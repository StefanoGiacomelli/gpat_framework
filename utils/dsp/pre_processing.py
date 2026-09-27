"""
Audio Preprocessing Module.

This module handles the core audio preprocessing pipeline including
resampling, channel conversion, and amplitude normalization.

Features:
    - Dynamic resampling with cached resamplers for heterogeneous data
    - Mono conversion with multiple modes (average, left, right, none)
    - Amplitude normalization: peak, RMS (dB), loudness weighting (A/B/C), LUFS

Author: Stefano Giacomelli, Ph.D. Candidate
Institution: Department of Engineering, Information Science & Mathematics, University of L'Aquila
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import pyloudnorm as pyln
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .globals import AudioConfig


class AudioPreprocessor(nn.Module):
    """Audio preprocessing module for consistent input handling.

    Handles resampling, channel conversion, and amplitude normalization
    to ensure all audio inputs are standardized for downstream processing.

    Attributes:
        target_sr: int, Target sample rate for all audio.
        mono_mode: str, Channel conversion mode ('average', 'left', 'right', 'none').
        normalization: str, Normalization strategy ('peak', 'rms_db', 'loudness_weighted', 'lufs', 'none').

    Example:
        Basic preprocessing pipeline::

            >>> from models.epanns.dsp.globals import AudioConfig
            >>> config = AudioConfig()
            >>> config.target_sample_rate = 48000
            >>> config.mono_mode = "average"
            >>> config.normalization = "peak"
            >>> preprocessor = AudioPreprocessor(config)
            >>> audio = torch.randn(4, 2, 96000)  # (B, C, T) stereo at 48kHz
            >>> processed, metadata = preprocessor(audio, sample_rate=48000)
            >>> print(processed.shape)  # (4, 1, 96000) - mono output
            >>> print(metadata['normalization_applied'])  # 'peak'

        LUFS normalization with peak limiting::

            >>> config.normalization = "lufs"
            >>> config.normalization_params = {
            ...     "lufs_target": -14.0,
            ...     "lufs_peak_dbfs": -1.0,
            ...     "lufs_filter_class": "K-weighting"
            ... }
            >>> preprocessor = AudioPreprocessor(config)
    """
    
    def __init__(self, config: "AudioConfig"):
        """Initialize audio preprocessor.

        Args:
            config: AudioConfig, Configuration instance from globals.py containing
                target_sample_rate, mono_mode, normalization, and normalization_params.
        """
        super(AudioPreprocessor, self).__init__()
        
        self.config = config
        
        # Extract key parameters
        self.target_sr = self.config.target_sample_rate
        self.mono_mode = self.config.mono_mode
        self.normalization = self.config.normalization
        
        # Advanced normalization parameters
        norm_params = self.config.normalization_params
        self.peak_target = norm_params.get("peak_target", 1.0)
        self.rms_target_db = norm_params.get("rms_target_db", -6.0)
        self.lufs_target = norm_params.get("lufs_target", -14.0)
        self.lufs_peak_dbfs = norm_params.get("lufs_peak_dbfs", -1.0)
        self.lufs_block_size = norm_params.get("lufs_block_size", None)
        self.lufs_filter_class = norm_params.get("lufs_filter_class", "K-weighting")
        self.weighting_curve = norm_params.get("weighting_curve", "A")
        
        # Resampler cache (dynamically created based on input sample rate(s) --> support heterogeneous data)
        self.resampler_cache = {}
        
    def _get_resampler(self, input_sr: int):
        """Get or create resampler for the given input sample rate.

        Resamplers are cached to avoid recreation overhead when processing
        audio from heterogeneous sources with different sample rates.

        Args:
            input_sr: int, Input sample rate in Hz.

        Returns:
            torchaudio.transforms.Resample or None: Resampler instance,
                or None if input_sr matches target_sr.
        """
        if input_sr == self.target_sr:
            return None
            
        if input_sr not in self.resampler_cache:
            self.resampler_cache[input_sr] = T.Resample(orig_freq=input_sr,
                                                        new_freq=self.target_sr,
                                                        resampling_method="sinc_interp_hann")
            
        return self.resampler_cache[input_sr]
    
    
    def _ensure_batch_dimension(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Ensure audio tensor has batch dimension.

        Args:
            audio: torch.Tensor, Input tensor of shape (T), (C, T) or (B, C, T).

        Returns:
            Tuple[torch.Tensor, bool]:
                - audio_batched: Tensor with batch dimension (B, C, T).
                - was_unbatched: Whether the input was originally unbatched.

        Raises:
            ValueError: If input tensor has unsupported dimensions.
        """
        if audio.ndim == 1:
            # Unbatched audio: (T) -> (1, 1, T)
            return audio.unsqueeze(0).unsqueeze(0), True
        elif audio.ndim == 2:
            # Add batch dimension: (C, T) -> (1, C, T)
            return audio.unsqueeze(0), True
        elif audio.ndim == 3:
            # Already batched: (B, C, T)
            return audio, False
        else:
            raise ValueError(f"Expected 1D, 2D or 3D audio tensor, got {audio.ndim}D")

    def _handle_channels(self, audio: torch.Tensor) -> torch.Tensor:
        """Handle multi-channel audio conversion to mono if specified.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Processed audio of shape (B, 1, T) if mono,
                (B, C, T) otherwise.

        Raises:
            ValueError: If mono_mode is unknown.
        """
        B, C, T = audio.shape
        
        if C == 1:
            return audio  # Already mono
        
        if self.mono_mode == "average":
            # Average all channels
            audio = audio.mean(dim=1, keepdim=True)
        elif self.mono_mode == "left":
            # Keep only left channel (first channel)
            audio = audio[:, :1, :]
        elif self.mono_mode == "right":
            # Keep only right channel (second channel if exists, otherwise first)
            channel_idx = min(1, C - 1)
            audio = audio[:, channel_idx:channel_idx+1, :]
        elif self.mono_mode == "none":
            # Keep all channels
            pass
        else:
            raise ValueError(f"Unknown mono_mode: {self.mono_mode}")
            
        return audio
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio according to specified strategy.

        Supports multiple normalization methods:
            - 'peak': Scale to configurable peak amplitude.
            - 'rms_db': Scale to target RMS level in dB.
            - 'loudness_weighted': Apply A/B/C weighting curves.
            - 'lufs': K-weighted LUFS normalization via pyloudnorm.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Normalized audio of shape (B, C, T).

        Raises:
            ValueError: If normalization method is unknown.
        """
        if self.normalization == "none":
            return audio
        
        B, C, T = audio.shape
        
        if self.normalization == "peak":
            # Peak normalization with configurable target
            peak = torch.amax(torch.abs(audio), dim=(-1, -2), keepdim=True)
            audio = torch.where(peak > 0, (audio / peak) * self.peak_target, audio)
                        
        elif self.normalization == "rms_db":
            # RMS normalization to dB target
            target_rms_linear = 10**(self.rms_target_db / 20.0)
            rms = torch.sqrt(torch.mean(audio**2, dim=(-1, -2), keepdim=True))
            audio = torch.where(rms > 0, (audio / rms) * target_rms_linear, audio)
            
        elif self.normalization == "loudness_weighted":
            # Apply loudness contouring (A, B, or C weighting)
            audio = self._apply_loudness_weighting(audio, self.weighting_curve)
            
        elif self.normalization == "lufs":
            # LUFS-based (K-weighting) normalization using pyloudnorm
            audio = self._apply_lufs_normalization(audio)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
            
        return audio
    
    def _apply_loudness_weighting(self, audio: torch.Tensor, weight_type: str = 'A') -> torch.Tensor:
        """Apply loudness contouring (A, B, or C weighting) to audio.

        Applies frequency-dependent gain curves in the frequency domain
        to model human loudness perception.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).
            weight_type: str, default='A'. Weighting curve ('A', 'B', 'C').

        Returns:
            torch.Tensor: Weighted audio of shape (B, C, T).

        Raises:
            ValueError: If weight_type is unknown.
        """
        B, C, T = audio.shape
        
        # Frequency vector for RFFT
        freqs = np.fft.rfftfreq(T, d=1/self.target_sr)
        
        # Weighting functions
        def R_A(f):
            return (12194**2 * f**4) / ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2))
        
        def R_B(f):
            return (12194**2 * f**3) / ((f**2 + 20.6**2) * np.sqrt(f**2 + 158.5**2) * (f**2 + 12194**2))
        
        def R_C(f):
            return (12194**2 * f**2) / ((f**2 + 20.6**2) * (f**2 + 12194**2))
        
        # Calculate weighting curve
        if weight_type == 'A':
            # Avoid log(0) by adding small epsilon to freqs
            safe_freqs = np.maximum(freqs, 1e-10)
            gain_db = 20 * np.log10(R_A(safe_freqs)) - 20 * np.log10(R_A(1000))
        elif weight_type == 'B':
            safe_freqs = np.maximum(freqs, 1e-10)
            gain_db = 20 * np.log10(R_B(safe_freqs)) - 20 * np.log10(R_B(1000))
        elif weight_type == 'C':
            safe_freqs = np.maximum(freqs, 1e-10)
            gain_db = 20 * np.log10(R_C(safe_freqs)) - 20 * np.log10(R_C(1000))
        else:
            raise ValueError(f"Unknown weighting type: {weight_type}")
        
        gain_linear = 10**(gain_db / 20)
        gain_tensor = torch.tensor(gain_linear, dtype=audio.dtype, device=audio.device)
        
        # Apply weighting in frequency domain
        weighted_audio = torch.zeros_like(audio)
        for b in range(B):
            for c in range(C):
                X = torch.fft.rfft(audio[b, c])
                X_weighted = X * gain_tensor
                weighted_audio[b, c] = torch.fft.irfft(X_weighted, n=T)
        
        return weighted_audio
    
    def _apply_lufs_normalization(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply LUFS-based normalization using pyloudnorm.

        Normalizes audio to target integrated loudness (LUFS) with optional
        true peak limiting. Falls back to original signal if measurement fails.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: LUFS-normalized audio of shape (B, C, T).

        Notes:
            Requires pyloudnorm library. Processing is done on CPU via NumPy.
        """
        B, C, T = audio.shape
        
        # Convert to numpy for pyloudnorm processing
        audio_np = audio.detach().cpu().numpy()
        normalized_np = np.zeros_like(audio_np)
        
        for b in range(B):
            for c in range(C):
                signal = audio_np[b, c, :]
                
                # Create loudness meter
                meter = pyln.Meter(self.target_sr, 
                                   block_size=self.lufs_block_size, 
                                   filter_class=self.lufs_filter_class)
                
                # Measure current loudness
                try:
                    current_loudness = meter.integrated_loudness(signal)
                    
                    # Check if measurement is valid
                    if current_loudness is None or np.isnan(current_loudness) or np.isinf(current_loudness):
                        raise ValueError("Invalid loudness measurement (None, NaN, or Inf)")
                    
                    # Normalize to target LUFS
                    normalized_signal = pyln.normalize.loudness(signal, current_loudness, self.lufs_target)
                    
                    # Optional peak limiting
                    if self.lufs_peak_dbfs is not None:
                        normalized_signal = pyln.normalize.peak(normalized_signal, self.lufs_peak_dbfs)
                    
                    normalized_np[b, c, :] = normalized_signal
                    
                except Exception as e:
                    # If LUFS measurement fails, fallback to original signal
                    print(f"Warning: LUFS normalization failed for batch {b}, channel {c}: {e}")
                    normalized_np[b, c, :] = signal
        
        # Convert back to torch tensor
        return torch.tensor(normalized_np, dtype=audio.dtype, device=audio.device)
    
    def _validate_input(self, audio: torch.Tensor, sample_rate: int) -> None:
        """Validate input audio and sample rate.

        Checks tensor type, dimensions, sample rate validity, and presence
        of NaN/Inf values.

        Args:
            audio: torch.Tensor, Input audio tensor.
            sample_rate: int, Input sample rate in Hz.

        Raises:
            ValueError: If any validation check fails.
        """
        if not isinstance(audio, torch.Tensor):
            raise ValueError("Audio must be a torch.Tensor")
            
        if audio.ndim not in [1, 2, 3]:
            raise ValueError(f"Audio must be 1D, 2D or 3D tensor, got {audio.ndim}D")

        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive integer, got {sample_rate}")
            
        # Check for NaN or infinite values
        if torch.isnan(audio).any():
            raise ValueError("Audio contains NaN values")
            
        if torch.isinf(audio).any():
            raise ValueError("Audio contains infinite values")
            
        # Check audio length
        if audio.shape[-1] == 0:
            raise ValueError("Audio cannot be empty")
    
    def forward(self, audio: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, dict]:
        """Preprocess raw audio for feature extraction.

        Applies the full preprocessing pipeline: validation, batch dimension
        handling, resampling, channel conversion, and normalization.

        Args:
            audio: torch.Tensor, Raw audio of shape (B, C, T) or (C, T).
            sample_rate: int, Original sample rate of the audio in Hz.

        Returns:
            Tuple[torch.Tensor, dict]:
                - preprocessed_audio: Standardized audio of shape (B, C, T).
                - metadata: Dictionary with preprocessing information including
                  original_shape, resampling_ratio, mono_conversion, etc.
        """
        # Validate inputs
        self._validate_input(audio, sample_rate)
        
        # Create metadata dictionary
        metadata = {"original_shape": audio.shape,
                    "original_sample_rate": sample_rate,
                    "target_sample_rate": self.target_sr,
                    "resampled": sample_rate != self.target_sr}
        
        # Ensure batch dimension
        audio, was_unbatched = self._ensure_batch_dimension(audio)
        metadata["was_unbatched"] = was_unbatched
        
        # Step 1: Resample if needed
        if sample_rate != self.target_sr:
            resampler = self._get_resampler(sample_rate)
            # Move resampler to same device as audio
            resampler = resampler.to(audio.device)
            audio = resampler(audio)
            metadata["resampling_ratio"] = self.target_sr / sample_rate
        
        # Step 2: Handle channels (mono conversion if specified)
        original_channels = audio.shape[1]
        audio = self._handle_channels(audio)
        metadata["original_channels"] = original_channels
        metadata["output_channels"] = audio.shape[1]
        metadata["mono_conversion"] = original_channels != audio.shape[1]
        
        # Step 3: Normalize audio
        audio = self._normalize_audio(audio)
        metadata["normalization_applied"] = self.normalization
        
        # Final validation
        assert audio.shape[0] >= 1, "Batch dimension lost"
        
        metadata["final_shape"] = audio.shape
        
        return audio, metadata
