"""
Frequency Domain Audio Augmentations.

This module implements various frequency-domain augmentation techniques
for audio spectrograms. All augmentations operate on spectrograms in
the format [batch, channels, frequency, time].

Techniques:
    - SpecAugment: Time stretching, frequency masking, temporal masking
    - Spectral Patching: Patch-based masking with configurable overlap
      and mutation strategies (zero_out, noise_injection)

Author: Stefano Giacomelli, Ph.D. Candidate
Institution: Department of Engineering, Information Science & Mathematics, University of L'Aquila
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
import torchaudio.transforms as T

if TYPE_CHECKING:
    from .globals import AugmentationConfig


class FrequencyDomainAugmenter(nn.Module):
    """Frequency-domain audio augmentation pipeline.

    Applies augmentations to spectrograms in training mode. Supports two
    mutually exclusive modes: SpecAugment (time stretching, frequency masking,
    temporal masking) or Spectral Patching (patch-based masking).

    Attributes:
        config: FrequencyDomainConfig, Configuration instance.
        enabled: bool, Whether augmentation is globally enabled.
        mode: str, Augmentation mode ('specaugment' or 'patching').

    Example:
        Basic usage with SpecAugment mode::

            >>> from models.epanns.dsp.globals import AugmentationConfig
            >>> config = AugmentationConfig()
            >>> augmenter = FrequencyDomainAugmenter(config)
            >>> spectrograms = torch.randn(8, 4, 128, 1500)  # (B, C, F, T)
            >>> augmented = augmenter(spectrograms, training=True)

        Configure for Spectral Patching mode::

            >>> config.frequency_domain.mode = "patching"
            >>> config.frequency_domain.patching_enabled = True
            >>> config.frequency_domain.patching_prob = 0.5
            >>> config.frequency_domain.patching_patch_size = (16, 16)
            >>> augmenter = FrequencyDomainAugmenter(config)
    """

    def __init__(self, config: "AugmentationConfig"):
        """Initialize frequency-domain augmenter.

        Args:
            config: AugmentationConfig, Configuration instance (globals.py) containing
                frequency_domain settings and random_seed for reproducibility.
        """
        super().__init__()
        
        self.config = config.frequency_domain
        self.enabled = self.config.enabled
        
        # Mode selection (specaugment vs patching)
        self.mode = self.config.mode
        
        # Initialize SpecAugment transforms (torchaudio)
        self._init_specaugment_transforms()
        
        # Random generators with seed offsets for reproducibility
        base_seed = config.random_seed
        self.generators = {"time_stretching": torch.Generator().manual_seed(base_seed + 11),
                           "frequency_masking": torch.Generator().manual_seed(base_seed + 12),
                           "temporal_masking": torch.Generator().manual_seed(base_seed + 13),
                           "patching": torch.Generator().manual_seed(base_seed + 14)}
        
    def _init_specaugment_transforms(self):
        """Initialize torchaudio SpecAugment transforms.

        Creates FrequencyMasking and TimeMasking transforms from torchaudio
        with parameters from config. Falls back gracefully if initialization fails.
        """
        try:
            self.freq_masking = T.FrequencyMasking(freq_mask_param=self.config.freq_masking_param, iid_masks=True)
            
            self.time_masking = T.TimeMasking(time_mask_param=self.config.temporal_masking_param, iid_masks=True)
            
        except Exception as e:
            print(f"Warning: Could not initialize SpecAugment transforms: {e}")
            self.freq_masking = None
            self.time_masking = None
    
    def forward(self, spectrograms: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply frequency-domain augmentations to spectrograms.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).
            training: bool, default=True. Whether in training mode.
                Augmentations are only applied when training=True.

        Returns:
            torch.Tensor: Augmented spectrograms of shape (B, C, F, T).
        """
        if not training or not self.enabled:
            return spectrograms
        
        # SpecAugment OR Patching (mutually exclusive)
        if self.mode == "specaugment":
            spectrograms = self._apply_specaugment(spectrograms)
        elif self.mode == "patching":
            spectrograms = self._apply_spectral_patching(spectrograms)
        
        return spectrograms
    
    def _apply_specaugment(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment techniques to spectrograms.

        Sequentially applies time stretching, frequency masking, and temporal
        masking based on individual enable flags in config.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Augmented spectrograms of shape (B, C, F, T).

        References:
            Park et al., "SpecAugment: A Simple Data Augmentation Method
            for Automatic Speech Recognition", Interspeech 2019.
        """
        # Apply time stretching
        if self.config.time_stretching_enabled:
            spectrograms = self._apply_time_stretching(spectrograms)
        
        # Apply frequency masking
        if self.config.freq_masking_enabled:
            spectrograms = self._apply_frequency_masking(spectrograms)
        
        # Apply temporal masking
        if self.config.temporal_masking_enabled:
            spectrograms = self._apply_temporal_masking(spectrograms)
        
        return spectrograms
    
    def _apply_time_stretching(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Apply time stretching augmentation.

        Stretches or compresses the temporal dimension of spectrograms
        using bilinear interpolation, then crops/pads to original length.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Time-stretched spectrograms of shape (B, C, F, T).

        Example:
            Configure time stretching::

                >>> config.frequency_domain.time_stretching_enabled = True
                >>> config.frequency_domain.time_stretching_prob = 0.8
                >>> config.frequency_domain.time_stretching_rate_range = [0.8, 1.25]
                >>> # Rate < 1.0 compresses, rate > 1.0 stretches
        """
        probability = self.config.time_stretching_prob
        
        if probability <= 0:
            return spectrograms
        
        batch_size = spectrograms.shape[0]
        rate_range = self.config.time_stretching_rate_range
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["time_stretching"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Generate stretch rates
        stretch_rates = torch.rand(batch_size, generator=self.generators["time_stretching"])
        stretch_rates = rate_range[0] + stretch_rates * (rate_range[1] - rate_range[0])
        
        # Apply time stretching to each sample
        stretched_spectrograms = spectrograms.clone()
        for b in range(batch_size):
            if apply_mask[b]:
                # Apply time stretching using interpolation
                original_time = spectrograms.shape[-1]
                new_time = int(original_time * stretch_rates[b])
                
                # Interpolate along time dimension
                sample = spectrograms[b].unsqueeze(0)  # [1, C, F, T]
                stretched = F.interpolate(sample, size=(spectrograms.shape[-2], new_time), mode='bilinear', align_corners=False)

                # Crop or pad to original time dimension
                if new_time > original_time:
                    # Crop from center
                    start_idx = (new_time - original_time) // 2
                    stretched = stretched[:, :, :, start_idx:start_idx + original_time]
                elif new_time < original_time:
                    # Pad to center
                    pad_left = (original_time - new_time) // 2
                    pad_right = original_time - new_time - pad_left
                    stretched = F.pad(stretched, (pad_left, pad_right, 0, 0))
                
                stretched_spectrograms[b] = stretched.squeeze(0)
        
        return stretched_spectrograms
    
    def _apply_frequency_masking(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking augmentation.

        Masks random contiguous frequency bands with zeros using torchaudio
        FrequencyMasking transform.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Frequency-masked spectrograms of shape (B, C, F, T).

        Example:
            Configure frequency masking::

                >>> config.frequency_domain.freq_masking_enabled = True
                >>> config.frequency_domain.freq_masking_prob = 0.8
                >>> config.frequency_domain.freq_masking_param = 27
                >>> # freq_masking_param: max number of consecutive freq bins to mask
        """
        probability = self.config.freq_masking_prob
        
        if probability <= 0 or self.freq_masking is None:
            return spectrograms
        
        batch_size = spectrograms.shape[0]
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["frequency_masking"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Apply frequency masking to selected samples
        masked_spectrograms = spectrograms.clone()
        for b in range(batch_size):
            if apply_mask[b]:
                # Apply frequency masking to all channels of this sample
                sample = spectrograms[b]  # [C, F, T]
                for c in range(sample.shape[0]):
                    # Add batch dimension for torchaudio masking (expects at least 3D)
                    channel_spec = sample[c].unsqueeze(0)  # [1, F, T]
                    masked_channel = self.freq_masking(channel_spec)  # [1, F, T]
                    masked_spectrograms[b, c] = masked_channel.squeeze(0)  # [F, T]
        
        return masked_spectrograms
    
    def _apply_temporal_masking(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Apply temporal masking augmentation.

        Masks random contiguous time frames with zeros using torchaudio
        TimeMasking transform.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Time-masked spectrograms of shape (B, C, F, T).

        Example:
            Configure temporal masking::

                >>> config.frequency_domain.temporal_masking_enabled = True
                >>> config.frequency_domain.temporal_masking_prob = 0.8
                >>> config.frequency_domain.temporal_masking_param = 100
                >>> # temporal_masking_param: max number of consecutive time frames to mask
        """
        probability = self.config.temporal_masking_prob
        
        if probability <= 0 or self.time_masking is None:
            return spectrograms
        
        batch_size = spectrograms.shape[0]
        
        # Generate random probabilities for each sample in batch
        rand_probs = torch.rand(batch_size, generator=self.generators["temporal_masking"])
        apply_mask = (rand_probs > (1 - probability))
        
        # Apply temporal masking to selected samples
        masked_spectrograms = spectrograms.clone()
        for b in range(batch_size):
            if apply_mask[b]:
                # Apply temporal masking to all channels of this sample
                sample = spectrograms[b]  # [C, F, T]
                for c in range(sample.shape[0]):
                    # Add batch dimension for torchaudio masking (expects at least 3D)
                    channel_spec = sample[c].unsqueeze(0)  # [1, F, T]
                    masked_channel = self.time_masking(channel_spec)  # [1, F, T]
                    masked_spectrograms[b, c] = masked_channel.squeeze(0)  # [F, T]
        
        return masked_spectrograms
    
    def _apply_spectral_patching(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Apply spectral patching augmentation.

        Divides the spectrogram into overlapping patches and randomly masks
        some patches based on pass_probability. Supports multiple mutation
        strategies: 'zero_out' or 'noise_injection'.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Patch-augmented spectrograms of shape (B, C, F, T).

        Example:
            Configure spectral patching::

                >>> config.frequency_domain.mode = "patching"
                >>> config.frequency_domain.patching_enabled = True
                >>> config.frequency_domain.patching_prob = 0.5
                >>> config.frequency_domain.patching_patch_size = (16, 16)  # (F, T)
                >>> config.frequency_domain.patching_overlap_ratio = 0.25
                >>> config.frequency_domain.patching_pass_probability = 0.7
                >>> config.frequency_domain.patching_mutation_strategy = "zero_out"
                >>> # or "noise_injection" for Gaussian noise replacement
        """
        if not self.config.patching_enabled:
            return spectrograms
        
        probability = self.config.patching_prob
        
        if probability <= 0:
            return spectrograms
        
        batch_size = spectrograms.shape[0]
        
        # Check if patching should be applied to this batch
        rand_prob = torch.rand(1, generator=self.generators["patching"]).item()
        if rand_prob > probability:
            return spectrograms
        
        # Generate patch masks
        masks = self._generate_patch_masks(spectrograms)
        
        # Apply mutations based on strategy
        mutation_strategy = self.config.patching_mutation_strategy
        
        if mutation_strategy == "zero_out":
            return spectrograms * masks
        elif mutation_strategy == "noise_injection":
            noise = torch.randn_like(spectrograms) * 0.1
            return torch.where(masks == 1, spectrograms, noise)
        else:
            return spectrograms * masks
    
    def _generate_patch_masks(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """Generate patch masks for spectral patching.

        Creates a binary mask grid based on patch size and overlap ratio.
        Each patch independently passes or blocks signal based on pass_probability.

        Args:
            spectrograms: torch.Tensor, Input spectrograms of shape (B, C, F, T).

        Returns:
            torch.Tensor: Binary masks of shape (B, C, F, T) where 1=pass, 0=mask.
        """
        batch_size, channels, freq_bins, time_bins = spectrograms.shape
        patch_h, patch_w = self.config.patching_patch_size
        overlap_ratio = self.config.patching_overlap_ratio
        overlap_resolution = self.config.patching_overlap_resolution
        pass_probability = self.config.patching_pass_probability
        
        # Calculate stride based on overlap
        stride_h = max(1, int(patch_h * (1 - overlap_ratio)))
        stride_w = max(1, int(patch_w * (1 - overlap_ratio)))
        
        # Initialize masks
        masks = torch.zeros(batch_size, channels, freq_bins, time_bins, device=spectrograms.device)
        
        for b in range(batch_size):
            # Generate patch grid for this sample
            patch_mask = torch.zeros(freq_bins, time_bins, device=spectrograms.device)
            overlap_count = torch.zeros(freq_bins, time_bins, device=spectrograms.device)
            
            for f_start in range(0, freq_bins, stride_h):
                for t_start in range(0, time_bins, stride_w):
                    f_end = min(f_start + patch_h, freq_bins)
                    t_end = min(t_start + patch_w, time_bins)
                    
                    # Decide if this patch passes signal
                    patch_passes = torch.rand(1, generator=self.generators["patching"]).item() < pass_probability
                    patch_value = 1.0 if patch_passes else 0.0
                    
                    # Handle overlap resolution
                    if overlap_resolution == "last_wins":
                        patch_mask[f_start:f_end, t_start:t_end] = patch_value
                    elif overlap_resolution == "or_logic":
                        patch_mask[f_start:f_end, t_start:t_end] = torch.max(patch_mask[f_start:f_end, t_start:t_end],
                                                                             torch.full((f_end - f_start, t_end - t_start), patch_value, device=spectrograms.device))

                    overlap_count[f_start:f_end, t_start:t_end] += 1
            
            # Apply the same mask to all channels of this batch sample
            masks[b] = patch_mask.unsqueeze(0).expand(channels, -1, -1)
        
        return masks
