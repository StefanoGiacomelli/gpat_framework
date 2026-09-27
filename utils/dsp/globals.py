"""
Configuration data structures for DSP system.

Author: Stefano Giacomelli, Ph.D. Candidate
Institution: Department of Engineering, Information Science & Mathematics, University of L'Aquila
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


# =============================================================================
# AUDIO PROCESSING CONFIGURATION
# =============================================================================

@dataclass
class AudioConfig:
    """Audio preprocessing and transform extraction configuration."""
    
    # Target specifications
    target_sample_rate: int = 48000
    target_duration: float = 10.0
    target_freq_bins: int = 128
    common_hop_length: int = 320
    
    # Input normalization
    mono_mode: str = "average"  # 'average', 'left', 'right', 'none'
    normalization: str = "peak"  # 'peak', 'rms_db', 'lufs', 'loudness_weighted', 'none'
    
    # Amplitude Normalization
    normalization_params: Dict[str, Any] = field(default_factory=lambda: {"peak_target": 1.0,
                                                                          "rms_target_db": -6.0,
                                                                          "lufs_target": -14.0,
                                                                          "lufs_peak_dbfs": -1.0,
                                                                          "lufs_block_size": None,
                                                                          "lufs_filter_class": "K-weighting",
                                                                          "weighting_curve": "A"})
    
    # Output features Normalization
    apply_minmax_normalization: bool = True
    
    # Transform-specific configurations
    transforms_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "stft": {"enabled": True,
                 "n_fft": 1024,
                 "hop_length": 320,
                 "win_length": 1024,
                 "window": "hann",
                 "center": True,
                 "pad_mode": "constant",
                 "fmin": 50.0,
                 "fmax": 14000.0,
                 "return_complex": True,
                 "output_format": "Complex"},
        
        "mel_spectrogram": {"enabled": True,
                            "n_fft": 1024,
                            "hop_length": 320,
                            "win_length": 1024,
                            "window": "hann",
                            "center": True,
                            "pad_mode": "constant",
                            "n_mels": 64,
                            "power": 2.0,
                            "htk": False,
                            "fmin": 50.0,
                            "fmax": 14000.0},
        
        "bark_spectrogram": {"enabled": True,
                             "n_fft": 1024,
                             "hop_length": 320,
                             "win_length": 1024,
                             "window": "hann",
                             "center": True,
                             "pad_mode": "constant",
                             "n_bins": 24,
                             "power": 2.0,
                             "normalized": False,
                             "fmin": 50.0,
                             "fmax": 14000.0},
        
        "gammatone_spectrogram": {"enabled": True,
                                  "n_fft": 1024,
                                  "hop_length": 320,
                                  "win_length": 1024,
                                  "window": "hann",
                                  "center": True,
                                  "pad_mode": "constant",
                                  "n_bins": 64,
                                  "fmin": 50.0,
                                  "fmax": 14000.0},
        
        "constant_q": {"enabled": True,
                       "hop_length": 320,
                       "window": "hann",
                       "center": True,
                       "pad_mode": "constant",
                       "n_bins": 64,
                       "bins_per_octave": 12,
                       "filter_scale": 1.0,
                       "fmin": 50.0,
                       "fmax": 14000.0,
                       "output_format": "Magnitude"},
        
        "correlogram": {"enabled": True,
                        "win_length": 1024,
                        "hop_length": 320,
                        "normalize": True,
                        "window_type": "hann"},
        
        "cepstrogram": {"enabled": True,
                        "win_length": 1024,
                        "hop_length": 320,
                        "fft_length": 2048,
                        "power": True},
        
        "cochleagram": {"enabled": False,
                        "n_filters": 64,
                        "low_lim": 50.0,
                        "high_lim": 14000.0,
                        "sample_factor": 4,
                        "full_filter": False,
                        "pad_factor": 1.25,
                        "use_rfft": True,
                        "envelope_sr": 200,
                        "downsampling_window_size": 1001,
                        "compression_power": 0.3,
                        "compression_offset": 1e-8,
                        "compression_scale": 1.0,
                        "compression_clip_value": 100.0}
    })


# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================

@dataclass
class TimeDomainAugConfig:
    """Time-domain augmentation configuration."""
    enabled: bool = True
    
    # Polarity inversion
    polarity_inversion_enabled: bool = True
    polarity_inversion_prob: float = 0.5
    
    # Amplitude scaling
    amplitude_scaling_enabled: bool = True
    amplitude_scaling_prob: float = 0.5
    amplitude_scale_range: List[float] = field(default_factory=lambda: [0.5, 0.9])
    amplitude_distribution: str = "uniform"
    
    # Time roll
    time_roll_enabled: bool = True
    time_roll_prob: float = 0.5
    time_roll_range: List[float] = field(default_factory=lambda: [0.1, 0.9])
    
    # Add noise
    add_noise_enabled: bool = True
    add_noise_prob: float = 0.5
    add_noise_snr_range: List[float] = field(default_factory=lambda: [30, 60])
    add_noise_type: str = "white"


@dataclass
class FreqDomainAugConfig:
    """Frequency-domain augmentation configuration."""
    enabled: bool = True
    mode: str = "specaugment"  # "specaugment" or "patching"
    
    # SpecAugment
    time_stretching_enabled: bool = True
    time_stretching_prob: float = 0.5
    time_stretching_rate_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    
    freq_masking_enabled: bool = True
    freq_masking_prob: float = 0.5
    freq_masking_param: int = 80
    freq_masking_num_masks: int = 1
    
    temporal_masking_enabled: bool = True
    temporal_masking_prob: float = 0.5
    temporal_masking_param: int = 80
    temporal_masking_num_masks: int = 1
    
    # Patching
    patching_enabled: bool = True
    patching_prob: float = 0.5
    patching_patch_size: List[int] = field(default_factory=lambda: [16, 16])
    patching_overlap_ratio: float = 0.25
    patching_overlap_resolution: str = "last_wins"
    patching_pass_probability: float = 0.5
    patching_mutation_strategy: str = "zero_out"


@dataclass
class AugmentationConfig:
    """Complete augmentation configuration."""
    random_seed: int = 42
    time_domain: TimeDomainAugConfig = field(default_factory=TimeDomainAugConfig)
    frequency_domain: FreqDomainAugConfig = field(default_factory=FreqDomainAugConfig)
