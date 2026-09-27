"""AudioSet-Tools refactored package.

Utilities for parsing, filtering, grouping, inspecting, and downloading
YouTube-backed AudioSet Standard, AudioSet Strong, and VGGSound metadata.
"""

from .dataset import AudioDataset, load_dataset, merge_datasets
from .downloader import YouTubeDatasetDownloader, read_download_report, write_download_report
from .events import generate_event_tracks, iter_events, plot_events_pianoroll
from .filters import blacklist_labels, filter_labels, rebalance, select_range
from .models import (AUDIOSET, AUDIOSET_STRONG, VGGSOUND,
                     AudioRecord, DownloadJob, DownloadReportEntry, TemporalEvent,
                     is_valid_youtube_id)
from .parsers import detect_format, load_label_map, parse_metadata
from .salt import SaltResolver
from .utils import compute_stats, find_samples_by_samples, find_samples_by_youtube_ids, merge_files

__all__ = [
    "AUDIOSET",
    "AUDIOSET_STRONG",
    "VGGSOUND",
    "AudioDataset",
    "AudioRecord",
    "DownloadJob",
    "DownloadReportEntry",
    "SaltResolver",
    "TemporalEvent",
    "YouTubeDatasetDownloader",
    "blacklist_labels",
    "compute_stats",
    "detect_format",
    "filter_labels",
    "find_samples_by_samples",
    "find_samples_by_youtube_ids",
    "generate_event_tracks",
    "iter_events",
    "is_valid_youtube_id",
    "load_dataset",
    "load_label_map",
    "merge_datasets",
    "merge_files",
    "parse_metadata",
    "plot_events_pianoroll",
    "read_download_report",
    "rebalance",
    "select_range",
    "write_download_report",
]
