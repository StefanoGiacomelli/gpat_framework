"""
Time Domain Audio Augmentations.

This module implements various time-domain augmentation techniques for
audio signals. All augmentations operate on raw audio waveforms in the
format [batch, channels, samples].

Techniques:
    - Polarity Inversion: Signal polarity flip (* -1)
    - Random Amplitude Scaling: Random scaling with uniform/gaussian distribution
    - Time Roll: Circular temporal shift of the signal
    - Add Noise: White or Gaussian noise addition with controlled SNR

Author: Stefano Giacomelli, Ph.D. Candidate
Institution: Department of Engineering, Information Science & Mathematics, University of L'Aquila
License: MIT
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .globals import AugmentationConfig


class TimeDomainAugmenter(nn.Module):
    """Time-domain audio augmentation pipeline.

    Applies a sequence of time-domain augmentations to audio waveforms
    based on probabilistic selection and configuration parameters.
    Augmentations are only applied during training mode.

    Processing sequence:
        1. Polarity Inversion
        2. Random Amplitude Scaling
        3. Time Roll
        4. Add Noise

    Attributes:
        config: TimeDomainConfig, Configuration instance.
        enabled: bool, Whether augmentation is globally enabled.

    Example:
        Basic usage with all augmentations::

            >>> from models.epanns.dsp.globals import AugmentationConfig
            >>> config = AugmentationConfig()
            >>> # Enable specific augmentations
            >>> config.time_domain.polarity_inversion_enabled = True
            >>> config.time_domain.polarity_inversion_prob = 0.5
            >>> config.time_domain.amplitude_scaling_enabled = True
            >>> config.time_domain.amplitude_scaling_prob = 0.8
            >>> config.time_domain.amplitude_scale_range = [0.5, 1.5]
            >>> config.time_domain.amplitude_distribution = "uniform"  # or "gaussian"
            >>> config.time_domain.time_roll_enabled = True
            >>> config.time_domain.time_roll_prob = 0.5
            >>> config.time_domain.time_roll_range = [-0.1, 0.1]  # fraction of signal
            >>> config.time_domain.add_noise_enabled = True
            >>> config.time_domain.add_noise_prob = 0.3
            >>> config.time_domain.add_noise_snr_range = [10, 30]  # dB
            >>> config.time_domain.add_noise_type = "gaussian"  # or "white"
            >>> augmenter = TimeDomainAugmenter(config)
            >>> audio = torch.randn(8, 1, 480000)  # (B, C, T)
            >>> augmented = augmenter(audio, training=True)
    """
    
    def __init__(self, config: "AugmentationConfig"):
        """Initialize time-domain augmenter.

        Args:
            config: AugmentationConfig, Configuration instance containing
                time_domain settings and random_seed for reproducibility.
        """
        super().__init__()
        
        self.config = config.time_domain
        self.enabled = self.config.enabled
        
        # Random generators with seed offsets for reproducibility
        base_seed = config.random_seed
        self.generators = {"polarity_inversion": torch.Generator().manual_seed(base_seed + 1),
                           "random_amplitude_scaling": torch.Generator().manual_seed(base_seed + 2),
                           "time_roll": torch.Generator().manual_seed(base_seed + 3),
                           "add_noise": torch.Generator().manual_seed(base_seed + 4)}
        
    def forward(self, audio: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply time-domain augmentations to audio.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).
            training: bool, default=True. Whether in training mode.
                Augmentations are only applied when training=True.

        Returns:
            torch.Tensor: Augmented audio of shape (B, C, T).
        """
        if not training or not self.enabled:
            return audio
        
        # Apply augmentations in sequence (only if enabled)
        if self.config.polarity_inversion_enabled:
            audio = self._apply_polarity_inversion(audio)
        if self.config.amplitude_scaling_enabled:
            audio = self._apply_random_amplitude_scaling(audio)
        if self.config.time_roll_enabled:
            audio = self._apply_time_roll(audio)
        if self.config.add_noise_enabled:
            audio = self._apply_add_noise(audio)
        
        return audio
    
    def _apply_polarity_inversion(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply polarity inversion (signal * -1) based on probability.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Polarity-inverted audio of shape (B, C, T).
        """
        probability = self.config.polarity_inversion_prob
        
        if probability <= 0.:
            return audio
            
        batch_size = audio.shape[0]
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["polarity_inversion"])
        
        # Apply inversion where random prob > threshold
        apply_mask = (rand_probs > (1 - probability)).to(audio.device)
        
        # Reshape mask for broadcasting [batch, 1, 1]
        apply_mask = apply_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply polarity inversion: audio * (-1 if apply else 1)
        polarity_factor = torch.where(apply_mask, -1.0, 1.0)
        
        return audio * polarity_factor
    
    def _apply_random_amplitude_scaling(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling based on probability and distribution.

        Supports uniform and gaussian distributions for scale factor sampling.
        Gaussian uses 3-sigma rule with mean at center of range.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Amplitude-scaled audio of shape (B, C, T).
        """
        probability = self.config.amplitude_scaling_prob
        
        if probability <= 0:
            return audio
            
        batch_size = audio.shape[0]
        scale_range = self.config.amplitude_scale_range
        distribution = self.config.amplitude_distribution
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["random_amplitude_scaling"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Generate scaling factors
        if distribution == "uniform":
            scale_factors = torch.rand(batch_size, generator=self.generators["random_amplitude_scaling"])
            scale_factors = scale_range[0] + scale_factors * (scale_range[1] - scale_range[0])
        elif distribution == "gaussian":
            # Gaussian around mean of scale_range with std = range/6 (3-sigma rule)
            mean_scale = (scale_range[0] + scale_range[1]) / 2
            std_scale = (scale_range[1] - scale_range[0]) / 6
            scale_factors = torch.normal(mean_scale, std_scale, (batch_size,), generator=self.generators["random_amplitude_scaling"])
            # Clamp to range
            scale_factors = torch.clamp(scale_factors, scale_range[0], scale_range[1])
        else:
            scale_factors = torch.ones(batch_size)
        
        # Apply scaling only where mask is True, else use 1.0
        scale_factors = torch.where(apply_mask, scale_factors, 1.0)
        scale_factors = scale_factors.to(audio.device).unsqueeze(1).unsqueeze(2)
        
        return audio * scale_factors
    
    def _apply_time_roll(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply circular temporal shift based on probability.

        Shift amount is specified as a fraction of signal length.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Time-rolled audio of shape (B, C, T).
        """
        probability = self.config.time_roll_prob
        
        if probability <= 0:
            return audio
            
        batch_size, channels, samples = audio.shape
        shift_range = self.config.time_roll_range
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["time_roll"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Generate shift amounts as fraction of signal length
        shift_fractions = torch.rand(batch_size, generator=self.generators["time_roll"])
        shift_fractions = shift_range[0] + shift_fractions * (shift_range[1] - shift_range[0])
        
        # Convert to sample shifts
        shift_samples = (shift_fractions * samples).long()
        
        # Apply time roll for each sample in batch
        rolled_audio = audio.clone()
        for b in range(batch_size):
            if apply_mask[b]:
                rolled_audio[b] = torch.roll(audio[b], shift_samples[b].item(), dims=-1)
        
        return rolled_audio
    
    def _apply_add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add white or gaussian noise based on SNR and probability.

        Noise power is calculated from signal RMS and target SNR in dB.
        Supports 'gaussian' (zero-mean normal) and 'white' (uniform) noise types.

        Args:
            audio: torch.Tensor, Input audio of shape (B, C, T).

        Returns:
            torch.Tensor: Noisy audio of shape (B, C, T).
        """
        probability = self.config.add_noise_prob
        
        if probability <= 0:
            return audio
            
        batch_size = audio.shape[0]
        snr_range = self.config.add_noise_snr_range
        noise_type = self.config.add_noise_type
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["add_noise"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Generate SNR values for each sample
        snr_values = torch.rand(batch_size, generator=self.generators["add_noise"])
        snr_values = snr_range[0] + snr_values * (snr_range[1] - snr_range[0])
        
        # Apply noise to each sample in batch
        noisy_audio = audio.clone()
        for b in range(batch_size):
            if apply_mask[b]:
                signal = audio[b]  # [channels, samples]
                
                # Calculate signal power (RMS)
                signal_power = torch.mean(signal ** 2)
                
                # Calculate noise power from desired SNR
                snr_linear = 10 ** (snr_values[b] / 10.0)
                noise_power = signal_power / snr_linear
                
                # Generate noise
                if noise_type == "gaussian":
                    noise = torch.normal(0, math.sqrt(noise_power), signal.shape, 
                                       generator=self.generators["add_noise"]).to(audio.device)
                elif noise_type == "white":
                    noise = torch.rand(signal.shape, generator=self.generators["add_noise"])
                    noise = (noise - 0.5) * 2 * math.sqrt(3 * noise_power)  # Scale uniform to match power
                    noise = noise.to(audio.device)
                else:
                    noise = torch.zeros_like(signal)
                
                noisy_audio[b] = signal + noise
        
        return noisy_audio
