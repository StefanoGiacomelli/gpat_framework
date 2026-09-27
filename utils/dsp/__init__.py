"""
DSP (Digital Signal Processing) module for EPANNs.

Provides audio preprocessing, transform extraction, and augmentation capabilities.

Authors: Stefano Giacomelli, Ph.D. candidate in ICT - DISIM dpt. University of L'Aquila, Italy
License: MIT License
"""

from .globals import AudioConfig, AugmentationConfig, TimeDomainAugConfig, FreqDomainAugConfig
from .pre_processing import AudioPreprocessor
from .audio_transforms import MultiTransformExtractor
from .time_domain_aug import TimeDomainAugmenter
from .freq_domain_aug import FrequencyDomainAugmenter

__all__ = ['AudioConfig',
           'AugmentationConfig',
           'TimeDomainAugConfig',
           'FreqDomainAugConfig',
           'AudioPreprocessor',
           'MultiTransformExtractor',
           'TimeDomainAugmenter',
           'FrequencyDomainAugmenter']
