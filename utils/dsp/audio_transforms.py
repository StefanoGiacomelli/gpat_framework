"""
Audio Transforms Module.

This module provides the core MultiTransformExtractor class and custom transform
implementations for multi-representation audio feature extraction. All transforms
support agnostic acceleration and share a common temporal scale (hop_length) for
consistent concatenation in multi-channel spectrograms.

Supported Transforms:
    - STFT: Short-Time Fourier Transform with magnitude & phase spectra
    - Mel-Spectrogram: Mel filterbank-based spectral representation
    - Bark Spectrogram: Psychoacoustic Bark-scale filterbank
    - Gammatonegram: ERB filterbank for auditory modeling
    - Constant-Q Transform: Log-frequency spectral representation
    - Cepstrogram: Short-Time Cepstrum for pitch/timbre analysis
    - Correlogram: Short-Time Autocorrelation for periodicity detection
    - Cochleagram: Biologically-inspired cochlear filtering (ERB + resonance envelope)

Author: Stefano Giacomelli, Ph.D. Candidate
Institution: Department of Engineering, Information Science & Mathematics, University of L'Aquila
License: MIT
"""

from typing import Dict, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from nnAudio.features import STFT, MelSpectrogram, CQT, Gammatonegram
import chcochleagram

if TYPE_CHECKING:
    from .globals import AudioConfig


class MultiTransformExtractor(nn.Module):
    """Multi-transform audio feature extractor.

    Extracts multiple spectral representations from raw audio and concatenates
    them along the channel dimension. All transforms share the same hop_length
    for temporal consistency, and outputs are standardized to common frequency
    and time dimensions through interpolation/truncation.

    The extractor supports dynamic resampling, duration handling (pad/crop),
    and optional per-channel MinMax normalization.

    Attributes:
        config: AudioConfig, Configuration instance from globals.py.
        active_transforms: List[str], Names of enabled transforms.
        target_sr: int, Target sample rate for all transforms.
        target_freq_bins: int, Standardized frequency dimension.
        target_time_frames: int, Standardized time dimension.

    Example:
        >>> from models.epanns.dsp.globals import AudioConfig
        >>> config = AudioConfig()
        >>> extractor = MultiTransformExtractor(config)
        >>> audio = torch.randn(4, 1, 480000)  # (B, C, T) at 48kHz
        >>> features = extractor(audio, sample_rate=48000)
        >>> print(features.shape)  # (4, N_transforms, 128, 1500)
    """
    
    def __init__(self, config: "AudioConfig"):
        super(MultiTransformExtractor, self).__init__()
        
        self.config = config
        
        # Extract parameters
        self.target_sr = self.config.target_sample_rate
        self.target_duration = self.config.target_duration
        self.common_hop_length = self.config.common_hop_length
        self.target_freq_bins = self.config.target_freq_bins
        
        # Derive active transforms from enabled flags
        self.active_transforms = [name for name, cfg in self.config.transforms_config.items() if cfg.get('enabled', False)]
        if not self.active_transforms:
            raise ValueError("At least one transform must be enabled in transforms_config")
        
        self.apply_minmax_norm = self.config.apply_minmax_normalization
        
        # Calculate target dimensions
        self.target_length = int(self.target_sr * self.target_duration)
        self.target_time_frames = int(self.target_length // self.common_hop_length)
        
        # Init resampler (will be configured dynamically)
        self.resampler = None
        self.current_input_sr = None
        
        # Init transform modules
        self.transforms = nn.ModuleDict()
        self._build_transforms()
        
    def _build_transforms(self):
        """Build transform modules based on enabled flags in configuration."""
        
        transforms_config = self.config.transforms_config
        
        for transform_name in self.active_transforms:
            if transform_name not in transforms_config:
                raise ValueError(f"Transform '{transform_name}' not found in configuration")
                
            transform_config = transforms_config[transform_name]
            
            # Build each transform module
            if transform_name == "stft":
                self.transforms[transform_name] = self._build_stft_transform(transform_config)
            elif transform_name == "mel_spectrogram":
                self.transforms[transform_name] = self._build_mel_transform(transform_config)
            elif transform_name == "constant_q":
                self.transforms[transform_name] = self._build_cqt_transform(transform_config)
            elif transform_name == "bark_spectrogram":
                self.transforms[transform_name] = self._build_bark_transform(transform_config)
            elif transform_name == "gammatone_spectrogram":
                self.transforms[transform_name] = self._build_erb_transform(transform_config)
            elif transform_name == "cepstrogram":
                self.transforms[transform_name] = self._build_cepstral_transform(transform_config)
            elif transform_name == "correlogram":
                self.transforms[transform_name] = self._build_autocorrelation_transform(transform_config)
            elif transform_name == "cochleagram":
                self.transforms[transform_name] = self._build_cochleagram_transform(transform_config)
            else:
                raise ValueError(f"Unknown transform: {transform_name}")
                
    def _build_stft_transform(self, config: Dict) -> nn.Module:
        """Build STFT transform module (Complex spectrum)."""
        return STFT(sr=self.target_sr,
                    n_fft=config["n_fft"],
                    win_length=config["win_length"],
                    hop_length=config["hop_length"],
                    window=config["window"],
                    center=config["center"],
                    pad_mode=config["pad_mode"],
                    trainable=False,
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                    output_format=config["output_format"],
                    verbose=False)
        
    def _build_mel_transform(self, config: Dict) -> nn.Module:
        """Build Mel-spectrogram transform module."""
        return MelSpectrogram(sr=self.target_sr,
                              n_fft=config["n_fft"],
                              win_length=config["win_length"],
                              hop_length=config["hop_length"],
                              window=config["window"],
                              center=config["center"],
                              pad_mode=config["pad_mode"],
                              n_mels=config["n_mels"],
                              power=config["power"],
                              htk=config["htk"],
                              fmin=config["fmin"],
                              fmax=config["fmax"],
                              trainable_mel=False,
                              trainable_STFT=False,
                              verbose=False)
        
    def _build_cqt_transform(self, config: Dict) -> nn.Module:
        """Build Constant-Q Transform module.""" 
        return CQT(sr=self.target_sr,
                   hop_length=config["hop_length"],
                   window=config["window"],
                   center=config["center"],
                   pad_mode=config["pad_mode"],
                   n_bins=config["n_bins"],
                   bins_per_octave=config["bins_per_octave"],
                   filter_scale=config["filter_scale"],
                   fmin=config["fmin"],
                   fmax=config["fmax"],
                   trainable=False,
                   output_format=config["output_format"],
                   verbose=False)
        
    def _build_bark_transform(self, config: Dict) -> nn.Module:
        """Build Bark spectrogram transform module."""
        return BarkSpectrogram(sample_rate=self.target_sr,
                               n_fft=config["n_fft"],
                               win_length=config["win_length"],
                               hop_length=config["hop_length"],
                               window=config["window"],
                               center=config["center"],
                               pad_mode=config["pad_mode"],
                               n_barks=config.get("n_bins", 64),
                               f_min=config["fmin"],
                               f_max=config["fmax"],
                               power=config.get("power", 2.0),
                               normalized=config.get("normalized", False))
        
    def _build_erb_transform(self, config: Dict) -> nn.Module:
        """Build ERB Gammatonegram transform module."""
        return Gammatonegram(sr=self.target_sr,
                             n_fft=config["n_fft"],
                             win_length=config["win_length"],
                             hop_length=config["hop_length"],
                             window=config["window"],
                             center=config["center"],
                             pad_mode=config["pad_mode"],
                             n_bins=config.get("n_bins", 64),
                             power=config.get("power", 2.0),
                             htk=config.get("htk", False),
                             fmin=config["fmin"],
                             fmax=config["fmax"],
                             trainable_bins=False,
                             trainable_STFT=False,
                             verbose=False)
        
    def _build_cepstral_transform(self, config: Dict) -> nn.Module:
        """Build Cepstrogram transform module."""
        return Cepstrogram(win_length=config["win_length"],
                           hop_length=config["hop_length"],
                           fft_length=config["fft_length"],
                           power=config["power"])

    def _build_autocorrelation_transform(self, config: Dict) -> nn.Module:
        """Build Correlogram transform module."""
        return Correlogram(win_length=config["win_length"],
                           hop_length=config["hop_length"],
                           normalize=config["normalize"],
                           window_type=config["window_type"])
    
    def _build_cochleagram_transform(self, config: Dict) -> nn.Module:
        """Build Cochleagram transform module."""
        return Cochleagram(target_sr=self.target_sr,
                           target_duration=self.target_duration,
                           n_filters=config["n_filters"],
                           low_lim=config["low_lim"],
                           high_lim=config["high_lim"],
                           sample_factor=config["sample_factor"],
                           full_filter=config["full_filter"],
                           pad_factor=config["pad_factor"],
                           use_rfft=config["use_rfft"],
                           envelope_sr=config["envelope_sr"],
                           downsampling_window_size=config["downsampling_window_size"],
                           compression_power=config["compression_power"],
                           compression_offset=config["compression_offset"],
                           compression_scale=config["compression_scale"],
                           compression_clip_value=config["compression_clip_value"])
    
    def _setup_resampler(self, input_sr: int) -> bool:
        """Configure resampler for given input sample rate.

        Args:
            input_sr: int, Input audio sample rate in Hz.

        Returns:
            bool: True if resampling is needed, False otherwise.
        """
        if input_sr != self.target_sr:
            if self.current_input_sr != input_sr:
                self.resampler = T.Resample(orig_freq=input_sr,
                                            new_freq=self.target_sr,
                                            resampling_method="sinc_interp_hann")
                self.current_input_sr = input_sr
            return True
        return False
        
    def _handle_duration(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or crop audio to target duration with center alignment.

        Behavior:
            - If length == target: returns unchanged.
            - If length < target: symmetric zero-padding (center alignment).
            - If length > target: center cropping to target length.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Audio tensor of shape (B, C, target_length).
        """
        B, C, T = audio.shape
        
        if T == self.target_length:
            return audio
        
        elif T < self.target_length:
            # Centered Zero-padding
            pad_total = self.target_length - T
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = F.pad(audio, (pad_left, pad_right), mode='constant', value=0)
        else:
            # Center cropping
            start = (T - self.target_length) // 2
            audio = audio[:, :, start: start + self.target_length]
            
        return audio
    
    def _standardize_frequency_dimension(self, features: torch.Tensor, transform_name: str) -> torch.Tensor:
        """Standardize frequency axis to target_freq_bins via interpolation or truncation.

        Aligns all transforms to a common frequency dimension for concatenation.
        Uses bilinear interpolation for upsampling and simple truncation (low-pass
        behavior) for downsampling.

        Args:
            features: torch.Tensor, Transform output of shape (B, C, H, W).
            transform_name: str, Transform identifier for debugging.

        Returns:
            torch.Tensor: Features with shape (B, C, target_freq_bins, W).

        Raises:
            ValueError: If input tensor is not 4D.
        """
        if len(features.shape) != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W) for {transform_name}, got {features.shape}")
        
        B, C, H, W = features.shape
        
        if H == self.target_freq_bins:
            return features
        elif H < self.target_freq_bins:
            # Interpolate to extend frequency dimension - spectral continuity
            features = F.interpolate(features,
                                     size=(self.target_freq_bins, W),
                                     mode='bilinear',
                                     align_corners=False)
        else:
            # Remove higher frequency bins (low-pass behavior)
            features = features[:, :, :self.target_freq_bins, :]
            
        return features
    
    def _apply_minmax_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """Apply per-channel MinMax normalization to [0, 1] range.

        Normalizes each batch-channel combination independently to preserve
        the relative characteristics of different transforms.

        Args:
            features: torch.Tensor, Feature tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized features in [0, 1] range, same shape as input.
        """
        if not self.apply_minmax_norm:
            return features
            
        B, C, H, W = features.shape
        normalized_features = torch.zeros_like(features)
        
        # Apply MinMax normalization independently for each batch and channel
        for b in range(B):
            for c in range(C):
                # Get feature map for this batch and channel
                feature_map = features[b, c, :, :]  # (H, W)
                
                # Calculate min and max values
                min_val = torch.min(feature_map)
                max_val = torch.max(feature_map)
                
                # Apply MinMax normalization if range is not zero
                if max_val > min_val:
                    normalized_features[b, c, :, :] = (feature_map - min_val) / (max_val - min_val)
                else:
                    # If min_val == max_val (constant features), set to 0
                    normalized_features[b, c, :, :] = torch.zeros_like(feature_map)
        
        return normalized_features
    
    def forward(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract multi-transform features from raw audio.

        Processing Pipeline:
            1. Resample to target sample rate if needed.
            2. Pad/crop to target duration (center alignment).
            3. Apply all enabled transforms with frequency standardization.
            4. Optional MinMax normalization per channel.
            5. Concatenate along channel dimension.

        Args:
            audio: torch.Tensor, Raw audio of shape (B, C, T).
            sample_rate: int, Original sample rate of the input audio.

        Returns:
            torch.Tensor: Concatenated features of shape 
                (B, N_transforms*C, target_freq_bins, target_time_frames).
        """
        # Step 1: Resample to target sample rate if needed
        needs_resampling = self._setup_resampler(sample_rate)
        if needs_resampling:
            audio = self.resampler(audio)
            
        # Step 2: Handle duration (pad/truncate to target length)
        audio = self._handle_duration(audio)

        # Step 3: Apply all active transforms (extraction --> freq axis standardization --> time axis standardization)
        transform_outputs = []
        
        for transform_name in self.active_transforms:
            # Apply transform
            transform_module = self.transforms[transform_name]
            
            if transform_name == "stft":
                # STFT returns separate magnitude and phase
                magnitude, phase = self._apply_stft_transform(audio, transform_module)
                
                # Standardize frequency dimensions for both
                magnitude = self._standardize_frequency_dimension(magnitude, f"{transform_name}_magnitude")
                phase = self._standardize_frequency_dimension(phase, f"{transform_name}_phase")
                
                # Ensure time dimensions match target
                if magnitude.shape[-1] != self.target_time_frames:
                    magnitude = F.interpolate(magnitude,
                                              size=(self.target_freq_bins, self.target_time_frames),
                                              mode='bilinear',
                                              align_corners=False)
                if phase.shape[-1] != self.target_time_frames:
                    phase = F.interpolate(phase,
                                          size=(self.target_freq_bins, self.target_time_frames),
                                          mode='bilinear',
                                          align_corners=False)

                # Apply MinMax normalization to magnitude and phase independently
                magnitude = self._apply_minmax_normalization(magnitude)
                phase = self._apply_minmax_normalization(phase)

                # Add both magnitude and phase to outputs
                transform_outputs.extend([magnitude, phase])
                                
            else:
                # Standard transform application
                features = self._apply_standard_transform(audio, transform_module, transform_name)
                features = self._standardize_frequency_dimension(features, transform_name)
                
                if features.shape[-1] != self.target_time_frames:
                    features = F.interpolate(features,
                                             size=(self.target_freq_bins, self.target_time_frames),
                                             mode='bilinear',
                                             align_corners=False)

                # Apply MinMax normalization to features
                features = self._apply_minmax_normalization(features)

                transform_outputs.append(features)
        
        # Step 4: Concatenate all transforms along channel dimension
        output_features = torch.cat(transform_outputs, dim=1)
        
        return output_features
    
    def _apply_stft_transform(self, audio: torch.Tensor, transform_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply STFT and return separate magnitude and phase spectrograms.

        Args:
            audio: torch.Tensor, Preprocessed audio of shape (B, C, target_length).
            transform_module: nn.Module, STFT module from nnAudio.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - magnitude: Magnitude spectrogram (B, C, Freq, Time).
                - phase: Phase spectrogram (B, C, Freq, Time).
        """
        B, C, T = audio.shape
        magnitude_list = []
        phase_list = []
        
        # nnAudio STFT requires (Batch, Samples) input per channel
        for c in range(C):
            input_ch = audio[:, c, :].float()  # (B, T)
            
            # Get complex spectrogram
            complex_spec = transform_module(input_ch)  # (B, Freq, Time, 2) or (B, Freq, Time)
            
            # Handle different output formats from nnAudio STFT
            if complex_spec.ndim == 4 and complex_spec.shape[-1] == 2:
                # Real/Imaginary format: (B, Freq, Time, 2)
                real_part = complex_spec[..., 0]  # (B, Freq, Time)
                imag_part = complex_spec[..., 1]  # (B, Freq, Time)
                complex_tensor = torch.complex(real_part, imag_part)
            else:
                # Already complex tensor: (B, Freq, Time)
                complex_tensor = complex_spec
            
            # Extract magnitude and phase
            magnitude = torch.abs(complex_tensor).unsqueeze(1)  # (B, 1, Freq, Time)
            phase = torch.angle(complex_tensor).unsqueeze(1)    # (B, 1, Freq, Time)
            
            magnitude_list.append(magnitude)
            phase_list.append(phase)
        
        # Stack channels: (B, C, Freq, Time)
        magnitude_output = torch.cat(magnitude_list, dim=1)
        phase_output = torch.cat(phase_list, dim=1)
        
        return magnitude_output, phase_output
    
    def _apply_standard_transform(self, audio: torch.Tensor, transform_module: nn.Module, transform_name: str) -> torch.Tensor:
        """Apply non-STFT transforms (Mel, Bark, Gammatone, CQT, Correlogram, Cepstrogram, Cochleagram).

        Handles per-channel processing for nnAudio/torchaudio transforms and
        direct batch processing for custom implementations.

        Args:
            audio: torch.Tensor, Preprocessed audio of shape (B, C, target_length).
            transform_module: nn.Module, Transform module instance.
            transform_name: str, Transform identifier for routing logic.

        Returns:
            torch.Tensor: Transform output of shape (B, C, Freq, Time).
        """
        B, C, T = audio.shape
        
        # Check if this requires per-channel processing (nnAudio + torchaudio transforms)
        if transform_name in ["mel_spectrogram", "gammatone_spectrogram", "constant_q", "bark_spectrogram"]:
            features_list = []
            for c in range(C):
                input_ch = audio[:, c, :].float()                                               # (B, T)
                output = transform_module(input_ch)                                             # (B, Freq, Time)
                features_list.append(output.unsqueeze(1))                                       # (B, 1, Freq, Time)
            
            features = torch.cat(features_list, dim=1)                                          # (B, C, Freq, Time)
            
        else:
            # Custom transforms (Correlogram, Cepstrogram, Cochleagram) handle (B, C, T) input
            features = transform_module(audio)                                                  # (B, C, H, W)
        
        return features
    
    def get_output_shape(self, input_channels: int = 1) -> Tuple[int, int, int]:
        """Compute expected output shape for given number of input channels.

        Args:
            input_channels: int, Number of input audio channels (default: 1).

        Returns:
            Tuple[int, int, int]: (output_channels, freq_bins, time_frames).
        """
        # Calculate total output channels
        total_channels = 0
        for transform_name in self.active_transforms:
            if transform_name == "stft":
                total_channels += input_channels * 2  # magnitude + phase
            else:
                total_channels += input_channels
        
        return (total_channels, self.target_freq_bins, self.target_time_frames)
    
    def get_transform_info(self) -> Dict[str, Dict]:
        """Retrieve configuration and metadata for all active transforms.

        Returns:
            Dict[str, Dict]: Mapping from transform name to info dict containing
                'config', 'output_channels', and 'native_freq_bins'.
        """
        info = {}
        for transform_name in self.active_transforms:
            config = self.config.transforms_config[transform_name]
            info[transform_name] = {"config": config,
                                    "output_channels": 2 if transform_name == "stft" else 1,  # Per input channel
                                    "native_freq_bins": "varies"}                             # Will be filled by actual implementations
        return info


# =============================================================================
# CUSTOM TRANSFORM IMPLEMENTATIONS
# =============================================================================

class Cepstrogram(nn.Module):
    """Short-Time Cepstrogram for quefrency-domain analysis.

    Computes the cepstrum (inverse FFT of log power spectrum) in overlapping
    frames, useful for pitch detection and spectral envelope analysis.

    Attributes:
        win_length: int, Window length in samples.
        hop_length: int, Hop length between frames.
        fft_length: int, FFT size for cepstrum computation.
        power: bool, If True use power spectrum, else magnitude spectrum.
    """
    def __init__(self, win_length=2048, hop_length=512, fft_length=4096, power=True):
        super(Cepstrogram, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.power = power
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cepstrogram from audio input.

        Args:
            x: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Cepstrogram of shape (B, C, Quefrency, Frames).

        Raises:
            ValueError: If input tensor has unsupported dimensions.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif x.dim() == 2:
            x = x.unsqueeze(0)               # [1, C, T]
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        B, C, T = x.shape
        window = torch.hann_window(self.win_length, device=x.device)

        cep_list = []
        for b in range(B):
            ch_list = []
            for c in range(C):
                sig = x[b, c]

                frames = sig.unfold(0, self.win_length, self.hop_length)            # [F, win_length]
                frames = frames * window                                            # Apply window
                spectrum = torch.fft.rfft(frames, n=self.fft_length, dim=-1)        # [F, Freqs]

                if self.power:
                    spectrum = torch.pow(torch.abs(spectrum), 2)
                else:
                    spectrum = torch.abs(spectrum)

                log_spec = torch.log10(spectrum + 1e-10)
                cepstrum = torch.fft.irfft(log_spec, n=self.fft_length, dim=-1)     # [F, Q]
                ch_list.append(cepstrum.T)                                          # [Q, F]
            cep_list.append(torch.stack(ch_list))                                   # [C, Q, F]

        cepstrogram = torch.stack(cep_list)                                         # [B, C, Q, F]
        
        return cepstrogram


class Correlogram(nn.Module):
    """Short-Time Autocorrelation (Correlogram) for periodicity analysis.

    Computes windowed autocorrelation via FFT for efficient pitch and
    periodicity detection. Useful for speech and music analysis.

    Attributes:
        win_length: int, Window length in samples.
        hop_length: int, Hop length between frames.
        normalize: bool, If True normalize by frame energy.
        window_type: str, Window function ('hann', 'hamming', or None).
    """
    
    def __init__(self, win_length=2048, hop_length=512, normalize=True, window_type='hann'):
        super(Correlogram, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.window_type = window_type
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute correlogram from audio input.

        Args:
            x: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Correlogram of shape (B, C, Frames, Lags).

        Raises:
            ValueError: If input tensor has unsupported dimensions.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        B, C, T = x.shape

        # Create window
        if self.window_type == 'hann':
            window = torch.hann_window(self.win_length, device=x.device)
        elif self.window_type == 'hamming':
            window = torch.hamming_window(self.win_length, device=x.device)
        else:
            window = torch.ones(self.win_length, device=x.device)

        # Frame extraction
        frames = x.unfold(dimension=-1, size=self.win_length, step=self.hop_length)  # [B, C, Frames, win_length]
        frames = frames * window.view(1, 1, 1, -1)

        # FFT & Power Spectrum
        fft_size = 2**(self.win_length-1).bit_length()
        spec = torch.fft.rfft(frames, n=fft_size, dim=-1)  # [B, C, Frames, Freqs]
        mag_sq = spec * torch.conj(spec)

        # Approximate autocorrelation via IFFT
        acorr = torch.fft.irfft(mag_sq, n=fft_size, dim=-1)  # [B, C, Frames, Time-lag]
        acorr = acorr[..., :self.win_length // 2 + 1]

        # Normalization
        if self.normalize:
            energy = frames.pow(2).sum(dim=-1, keepdim=True)
            acorr = acorr / (energy + 1e-10)

        return acorr


class Cochleagram(nn.Module):
    """Biologically-inspired cochlear filterbank representation.

    Simulates cochlear frequency decomposition using ERB (Equivalent Rectangular
    Bandwidth) filters with Hilbert envelope extraction, downsampling, and
    power-law compression. Implemented via the chcochleagram library.

    Attributes:
        target_sr: int, Target sample rate in Hz.
        target_duration: float, Expected input duration in seconds.
        n_filters: int, Number of ERB cochlear filters.
        envelope_sr: int, Envelope sampling rate after downsampling.
    """
    def __init__(self, 
                 target_sr=48000,
                 target_duration=10.0,
                 n_filters=128,
                 low_lim=20.0,
                 high_lim=20000.0,
                 sample_factor=4,
                 full_filter=False,
                 pad_factor=1.25,
                 use_rfft=True,
                 envelope_sr=200,
                 downsampling_window_size=1001,
                 compression_power=0.3,
                 compression_offset=1e-8,
                 compression_scale=1.0,
                 compression_clip_value=100.0):
        super(Cochleagram, self).__init__()
        
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
        self.n_filters = n_filters
        self.envelope_sr = envelope_sr
        
        # ERB filter configuration
        half_cos_filter_kwargs = {'n': n_filters,
                                  'low_lim': low_lim,
                                  'high_lim': high_lim,
                                  'sample_factor': sample_factor,
                                  'full_filter': full_filter}
        
        # Cochlear filter kwargs
        coch_filter_kwargs = {'use_rfft': use_rfft,
                              'pad_factor': pad_factor,
                              'filter_kwargs': half_cos_filter_kwargs}
        
        # Initialize filters
        self.filters = chcochleagram.cochlear_filters.ERBCosFilters(self.target_length,
                                                                    self.target_sr, 
                                                                    **coch_filter_kwargs)
        
        # Envelope extraction
        self.envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(self.target_length,
                                                                                               self.target_sr, 
                                                                                               use_rfft, 
                                                                                               pad_factor)
        
        # Downsampling
        downsampling_kwargs = {'window_size': downsampling_window_size}
        self.downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(self.target_sr, 
                                                                               envelope_sr, 
                                                                               **downsampling_kwargs)
        
        # Compression
        compression_kwargs = {'power': compression_power,
                              'offset': compression_offset,
                              'scale': compression_scale,
                              'clip_value': compression_clip_value}
        self.compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)
        
        # Build the complete cochleagram
        self.cochleagram = chcochleagram.cochleagram.Cochleagram(self.filters,
                                                                 self.envelope_extraction,
                                                                 self.downsampling_op,
                                                                 compression=self.compression)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cochleagram from audio input.

        Args:
            x: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Cochleagram of shape (B, C, n_filters, time_frames).

        Raises:
            ValueError: If input tensor has unsupported dimensions.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        elif x.dim() == 2:
            x = x.unsqueeze(0)               # [1, C, T] 
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        B, C, T = x.shape
        
        # Ensure input is correct length
        if T != self.target_length:
            if T < self.target_length:
                # Zero pad
                pad_total = self.target_length - T
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                x = torch.nn.functional.pad(x, (pad_left, pad_right), mode='constant', value=0)
            else:
                # Center crop
                start = (T - self.target_length) // 2
                x = x[:, :, start: start + self.target_length]
        
        coch_list = []
        for b in range(B):
            ch_list = []
            for c in range(C):
                # chcochleagram expects (batch_size, signal_length) for single channel processing
                input_signal = x[b, c, :].unsqueeze(0)  # (1, T)
                
                # Apply cochleagram
                coch_output = self.cochleagram(input_signal)  # (1, n_filters, time_frames)
                ch_list.append(coch_output.squeeze(0))  # (n_filters, time_frames)
            
            # Stack channels: (C, n_filters, time_frames)
            coch_batch = torch.stack(ch_list, dim=0)
            coch_list.append(coch_batch)
        
        # Stack batches: (B, C, n_filters, time_frames)
        cochleagram_output = torch.stack(coch_list, dim=0)
        
        return cochleagram_output


class BarkSpectrogram(nn.Module):
    """Bark-scale spectrogram with psychoacoustically accurate filterbank.

    Implements the Bark scale for human frequency perception using:
        - Traunmüller (1990) Hz-to-Bark conversion formula
        - Zwicker & Fastl critical bandwidth calculation
        - Raised cosine (rounded) filters for smooth band transitions

    The rounded filter shape better models the critical band masking behavior 
    of the human auditory system.

    Attributes:
        sample_rate: int, Audio sample rate in Hz.
        n_fft: int, FFT size for STFT computation.
        n_barks: int, Number of Bark frequency bands.
        f_min: float, Minimum frequency in Hz.
        f_max: float, Maximum frequency in Hz.
        power: float, Spectrum power (1.0=magnitude, 2.0=power).
    """
    
    def __init__(self, 
                 sample_rate=32000,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=320,
                 window='hann',
                 center=True,
                 pad_mode='constant',
                 n_barks=24,
                 f_min=20.0,
                 f_max=14000.0,
                 power=2.0,
                 normalized=False):
        super(BarkSpectrogram, self).__init__()
        
        # STFT module (GPU-accelerated, batch support)
        self.stft = STFT(n_fft=n_fft, 
                         win_length=win_length,
                         hop_length=hop_length,
                         window=window,
                         center=center,
                         pad_mode=pad_mode,
                         freq_bins=None,
                         sr=sample_rate,
                         output_format='Complex')
        
        self.power = power
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_barks = n_barks
        self.f_min = f_min
        self.f_max = f_max
        
        # Create Bark filterbank matrix and register as buffer (auto device transfer)
        bark_fb = self._create_bark_filterbank(sample_rate=sample_rate,
                                               n_fft=n_fft,
                                               n_barks=n_barks,
                                               f_min=f_min,
                                               f_max=f_max,
                                               normalized=normalized)
        
        self.register_buffer('bark_filterbank', bark_fb)
    
    def _hz_to_bark(self, f_hz: torch.Tensor) -> torch.Tensor:
        """Convert Hz to Bark scale using Traunmüller (1990) formula.

        Formula: Bark = 26.81 * f / (1960 + f) - 0.53
        With edge corrections for extreme frequencies.

        Args:
            f_hz: torch.Tensor, Frequency values in Hz.

        Returns:
            torch.Tensor: Corresponding Bark scale values.
        """
        bark = 26.81 * f_hz / (1960.0 + f_hz) - 0.53
        
        # Edge corrections (Traunmüller adjustments)
        bark = torch.where(bark < 2.0, bark + 0.15 * (2.0 - bark), bark)
        bark = torch.where(bark > 20.1, bark + 0.22 * (bark - 20.1), bark)
        
        return bark
    
    def _bark_to_hz(self, bark: torch.Tensor) -> torch.Tensor:
        """Convert Bark scale to Hz (inverse Traunmüller formula).

        Args:
            bark: torch.Tensor, Bark scale values.

        Returns:
            torch.Tensor: Corresponding frequency values in Hz.
        """
        # Clamp bark to valid range to avoid division issues
        bark_clamped = torch.clamp(bark, min=0.0, max=24.0)
        
        # Apply inverse formula
        numerator = 1960.0 * (bark_clamped + 0.53)
        denominator = 26.81 - bark_clamped - 0.53
        
        # Prevent division by zero
        denominator = torch.clamp(denominator, min=0.01)
        
        f_hz = numerator / denominator
        
        return f_hz
    
    def _critical_bandwidth(self, f_hz: torch.Tensor) -> torch.Tensor:
        """Calculate critical bandwidth using Zwicker & Fastl formula.

        Formula: CBW = 25 + 75 * (1 + 1.4 * f_kHz^2)^0.69 [Hz]

        Args:
            f_hz: torch.Tensor, Center frequency in Hz.

        Returns:
            torch.Tensor: Critical bandwidth in Hz.
        """
        f_khz = f_hz / 1000.0
        cbw = 25.0 + 75.0 * torch.pow(1.0 + 1.4 * f_khz**2, 0.69)
        
        return cbw
    
    def _create_bark_filterbank(self, 
                                sample_rate: int,
                                n_fft: int, 
                                n_barks: int,
                                f_min: float,
                                f_max: float,
                                normalized: bool = False) -> torch.Tensor:
        """Create Bark filterbank matrix with raised cosine filters.

        Uses raised cosine (rounded) windows instead of triangular filters 
        for smoother band transitions that better match human auditory perception.

        Args:
            sample_rate: int, Audio sample rate in Hz.
            n_fft: int, FFT size.
            n_barks: int, Number of Bark frequency bands.
            f_min: float, Minimum frequency in Hz.
            f_max: float, Maximum frequency in Hz.
            normalized: bool, default=False. Whether to normalize each filter
                to sum=1 for energy preservation.

        Returns:
            torch.Tensor: Filterbank matrix of shape (n_barks, n_fft // 2 + 1).
        """
        # FFT frequency bins (linear scale)
        n_freqs = n_fft // 2 + 1
        fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs)
        
        # Convert frequency limits to Bark scale
        bark_min = self._hz_to_bark(torch.tensor(f_min, dtype=torch.float32))
        bark_max = self._hz_to_bark(torch.tensor(f_max, dtype=torch.float32))
        
        # Create equally-spaced centers in Bark scale
        bark_centers = torch.linspace(bark_min, bark_max, n_barks)
        
        # Convert center frequencies back to Hz
        hz_centers = self._bark_to_hz(bark_centers)
        
        # Initialize filterbank matrix
        filterbank = torch.zeros(n_barks, n_freqs)
        
        # Create each Bark filter
        for bark_idx in range(n_barks):
            center = hz_centers[bark_idx]
            
            # Calculate critical bandwidth for this center frequency
            bandwidth = self._critical_bandwidth(center)
            
            # Calculate distances from center frequency
            distances = torch.abs(fft_freqs - center)
            
            # Apply raised cosine filter (smoother than triangular)
            # Only apply within bandwidth/2 radius
            mask = distances <= bandwidth / 2.0
            
            # Raised cosine formula: 0.5 * (1 + cos(2π * d / BW))
            # This creates a smooth, rounded filter shape
            filterbank[bark_idx, mask] = 0.5 * (1.0 + torch.cos(2.0 * torch.pi * distances[mask] / bandwidth))
        
        # Optional normalization
        if normalized:
            # Normalize each filter to sum = 1 (energy preservation)
            row_sums = filterbank.sum(dim=1, keepdim=True)
            filterbank = filterbank / (row_sums + 1e-10)
        
        return filterbank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Bark spectrogram from audio input.

        Args:
            x: torch.Tensor, Input audio of shape (B, T).

        Returns:
            torch.Tensor: Bark spectrogram of shape (B, n_barks, time_frames).
        """
        # 1. Compute STFT (GPU-accelerated via nnAudio)
        complex_spec = self.stft(x)  # Output format depends on nnAudio version
        
        # 2. Extract magnitude spectrum
        if complex_spec.is_complex():
            # Native complex tensor
            magnitude = torch.abs(complex_spec)
        else:
            # Real/Imaginary format: (B, Freq, Time, 2)
            if complex_spec.ndim == 4 and complex_spec.shape[-1] == 2:
                real = complex_spec[..., 0]
                imag = complex_spec[..., 1]
                magnitude = torch.sqrt(real**2 + imag**2)
            else:
                magnitude = torch.abs(complex_spec)
        
        # 3. Apply power
        power_spec = magnitude ** self.power  # (B, Freq, Time)
        
        # 4. Apply Bark filterbank via matrix multiplication
        # bark_filterbank: (n_barks, Freq)
        # power_spec: (B, Freq, Time)
        # Result: (B, n_barks, Time)
        bark_spec = torch.matmul(self.bark_filterbank, power_spec)
        
        return bark_spec
    
    def get_filterbank(self) -> torch.Tensor:
        """Return the Bark filterbank matrix for visualization.

        Returns:
            torch.Tensor: Filterbank matrix of shape (n_barks, n_freqs).
        """
        return self.bark_filterbank
    
    def get_center_frequencies(self) -> torch.Tensor:
        """Return center frequencies of Bark bands in Hz.

        Returns:
            torch.Tensor: Center frequencies of shape (n_barks,).
        """
        bark_min = self._hz_to_bark(torch.tensor(self.f_min, dtype=torch.float32))
        bark_max = self._hz_to_bark(torch.tensor(self.f_max, dtype=torch.float32))
        bark_centers = torch.linspace(bark_min, bark_max, self.n_barks)
        hz_centers = self._bark_to_hz(bark_centers)
        return hz_centers
