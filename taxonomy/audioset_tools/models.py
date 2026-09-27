"""Shared data models for AudioSet-Tools.

The package normalizes AudioSet Standard, AudioSet Strong, and VGGSound
metadata into the same record shape while preserving source fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


AUDIOSET = "AudioSet"
AUDIOSET_STRONG = "AudioSetStrong"
VGGSOUND = "VggSound"
NORMALIZED = "Normalized"

SUPPORTED_DATASETS = (AUDIOSET, AUDIOSET_STRONG, VGGSOUND)


@dataclass(frozen=True)
class TemporalEvent:
    """A time-aligned event inside a normalized record.

    Times are relative to the downloaded clip segment when source metadata
    provides relative event boundaries, as AudioSet Strong does.
    """

    start_seconds: float | None
    end_seconds: float | None
    original_label_id: str | None = None
    original_label: str | None = None
    salt_labels: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioRecord:
    """Normalized metadata record for a YouTube-backed audio clip."""

    dataset: str
    source_id: str
    youtube_id: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    original_label_ids: tuple[str, ...] = ()
    original_labels: tuple[str, ...] = ()
    salt_labels: tuple[str, ...] = ()
    split: str | None = None
    events: tuple[TemporalEvent, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.start_seconds is None or self.end_seconds is None:
            return None
        
        return self.end_seconds - self.start_seconds

    def labels(self, label_space: str) -> tuple[str, ...]:
        if label_space == "id":
            return self.original_label_ids
        if label_space == "original":
            return self.original_labels
        if label_space == "salt":
            return self.salt_labels
        if label_space == "auto":
            return tuple(dict.fromkeys(self.original_label_ids + self.original_labels + self.salt_labels))
        raise ValueError("label_space must be one of 'auto', 'id', 'original', or 'salt'.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        
        return data


@dataclass(frozen=True)
class DownloadJob:
    """Download request derived from a normalized record."""

    dataset: str
    source_id: str
    youtube_id: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    output_stem: str | None = None
    original_label_ids: tuple[str, ...] = ()
    original_labels: tuple[str, ...] = ()
    salt_labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: AudioRecord) -> "DownloadJob":
        return cls(dataset=record.dataset,
                   source_id=record.source_id,
                   youtube_id=record.youtube_id,
                   start_seconds=record.start_seconds,
                   end_seconds=record.end_seconds,
                   output_stem=record.source_id,
                   original_label_ids=record.original_label_ids,
                   original_labels=record.original_labels,
                   salt_labels=record.salt_labels,
                   metadata={"split": record.split, "source_path": record.source_path})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadReportEntry:
    """Structured status entry for one attempted or simulated download."""

    dataset: str
    source_id: str
    youtube_id: str
    start_seconds: float | None
    end_seconds: float | None
    requested_output_path: str
    final_output_path: str | None
    status: str
    success: bool
    error_category: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    processing_timestamp: str | None = None
    original_label_ids: tuple[str, ...] = ()
    original_labels: tuple[str, ...] = ()
    salt_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_output_stem(value: str) -> str:
    """Create a filesystem-safe, traceable output stem."""

    cleaned = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    stem = "".join(cleaned).strip("._")
    
    return stem or "audio"


def is_valid_youtube_id(value: str) -> bool:
    """Validate the canonical 11-character YouTube video ID shape."""

    if len(value) != 11:
        return False
    
    return all(char.isalnum() or char in ("-", "_") for char in value)


def path_to_str(path: Path | str | None) -> str | None:
    if path is None:
        return None
    
    return str(path)
