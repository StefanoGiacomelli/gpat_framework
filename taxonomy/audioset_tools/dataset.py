"""Unified dataset API for AudioSet Standard, AudioSet Strong, and VGGSound."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .models import DownloadJob, AudioRecord
from .parsers import detect_format, parse_metadata
from .salt import SaltResolver


@dataclass(frozen=True)
class AudioDataset:
    """Normalized dataset container with label-aware operations."""

    records: tuple[AudioRecord, ...]
    source_format: str | None = None
    salt_resolver: SaltResolver | None = None

    @classmethod
    def from_file(cls,
                  path: str | Path,
                  labels_file: str | Path | None = None,
                  salt_resolver: SaltResolver | None = None,
                  dataset: str | None = None) -> "AudioDataset":
        detected = dataset or detect_format(path)
        records = tuple(parse_metadata(path, labels_file, salt_resolver, detected))
        
        return cls(records, source_format=detected, salt_resolver=salt_resolver)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[AudioRecord]:
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def filter_labels(self,
                      labels: Sequence[str],
                      label_space: str = "auto",
                      match: str = "any",
                      include_descendants: bool = False,
                      case_sensitive: bool = False) -> "AudioDataset":
        """Filter records by labels in original, id, SALT, or auto label space.

        ``match`` accepts ``any``, ``all``, and ``none``. ``none`` is the
        blacklist mode from the previous scripts.
        """

        if match not in {"any", "all", "none"}:
            raise ValueError("match must be one of 'any', 'all', or 'none'.")

        target_labels = tuple(labels)
        if label_space == "salt" and include_descendants and self.salt_resolver:
            target_labels = self.salt_resolver.expand_salt_labels(target_labels, include_descendants=True)

        target = set(_label_key(label, case_sensitive) for label in target_labels)
        if not target:
            return self._replace(())

        selected = []
        for record in self.records:
            record_labels = set(_label_key(label, case_sensitive) for label in record.labels(label_space))
            if match == "any" and record_labels.intersection(target):
                selected.append(record)
            elif match == "all" and target.issubset(record_labels):
                selected.append(record)
            elif match == "none" and not record_labels.intersection(target):
                selected.append(record)

        return self._replace(tuple(selected))

    def select_range(self, start_idx: int, end_idx: int) -> "AudioDataset":
        """Select records by zero-based interval, replacing select_by_samp_idx."""

        return self._replace(self.records[start_idx:end_idx])

    def group_by(self, field: str) -> dict[object, "AudioDataset"]:
        """Group records by any AudioRecord attribute or raw metadata key."""

        groups: dict[object, list[AudioRecord]] = defaultdict(list)
        for record in self.records:
            value = getattr(record, field, record.raw.get(field))
            groups[value].append(record)
        
        return {value: self._replace(tuple(records)) for value, records in groups.items()}

    def label_counts(self, label_space: str = "original") -> Counter:
        counts = Counter()
        for record in self.records:
            counts.update(record.labels(label_space))
        
        return counts

    def summary(self) -> dict:
        datasets = Counter(record.dataset for record in self.records)
        splits = Counter(record.split for record in self.records if record.split)
        
        return {"records": len(self.records),
                "datasets": dict(datasets),
                "splits": dict(splits),
                "unique_youtube_ids": len({record.youtube_id for record in self.records}),
                "label_counts_original": dict(self.label_counts("original")),
                "label_counts_salt": dict(self.label_counts("salt"))}

    def find_by_youtube_ids(self, youtube_ids: Iterable[str]) -> "AudioDataset":
        target = set(youtube_ids)
        
        return self._replace(tuple(record for record in self.records if record.youtube_id in target))

    def rebalance(self,
                  focus_labels: Sequence[str] | None = None,
                  label_space: str = "original",
                  random_state: int | None = None) -> "AudioDataset":
        """Undersample to the smallest per-label bucket.

        This preserves the old rebalancing intent while operating on normalized
        records from all supported source formats.
        """

        rng = random.Random(random_state)
        focus = set(focus_labels or [])
        label_to_records: dict[str, list[AudioRecord]] = defaultdict(list)
        for record in self.records:
            labels = record.labels(label_space)
            if focus:
                labels = tuple(label for label in labels if label in focus)
            for label in set(labels):
                label_to_records[label].append(record)

        if not label_to_records:
            return self._replace(())

        target_count = min(len(records) for records in label_to_records.values())
        selected: list[AudioRecord] = []
        seen_keys = set()
        for label in sorted(label_to_records):
            bucket = list(label_to_records[label])
            rng.shuffle(bucket)
            added = 0
            for record in bucket:
                key = _record_key(record)
                if key not in seen_keys:
                    seen_keys.add(key)
                    selected.append(record)
                    added += 1
                if added >= target_count:
                    break
        
        return self._replace(tuple(selected))

    def deduplicate(self) -> "AudioDataset":
        seen = set()
        output = []
        for record in self.records:
            key = _record_key(record)
            if key not in seen:
                seen.add(key)
                output.append(record)
        
        return self._replace(tuple(output))

    def merge(self, *others: "AudioDataset", deduplicate: bool = True) -> "AudioDataset":
        merged = self._replace(self.records + tuple(record for other in others for record in other.records))
        
        return merged.deduplicate() if deduplicate else merged

    def to_download_jobs(self) -> tuple[DownloadJob, ...]:
        return tuple(DownloadJob.from_record(record) for record in self.records)

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self.records], handle, indent=2)

    def to_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        fields = ["dataset",
                  "source_id",
                  "youtube_id",
                  "start_seconds",
                  "end_seconds",
                  "original_label_ids",
                  "original_labels",
                  "salt_labels",
                  "split"]
        
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({"dataset": record.dataset,
                                 "source_id": record.source_id,
                                 "youtube_id": record.youtube_id,
                                 "start_seconds": record.start_seconds,
                                 "end_seconds": record.end_seconds,
                                 "original_label_ids": json.dumps(record.original_label_ids),
                                 "original_labels": json.dumps(record.original_labels),
                                 "salt_labels": json.dumps(record.salt_labels),
                                 "split": record.split})

    def _replace(self, records: tuple[AudioRecord, ...]) -> "AudioDataset":
        return AudioDataset(records, self.source_format, self.salt_resolver)


def load_dataset(path: str | Path,
                 labels_file: str | Path | None = None,
                 salt_resolver: SaltResolver | None = None,
                 dataset: str | None = None) -> AudioDataset:
    return AudioDataset.from_file(path, labels_file, salt_resolver, dataset)


def merge_datasets(*datasets: AudioDataset, deduplicate: bool = True) -> AudioDataset:
    if not datasets:
        return AudioDataset(())
    return datasets[0].merge(*datasets[1:], deduplicate=deduplicate)


def _label_key(label: str, case_sensitive: bool) -> str:
    text = str(label).strip()
    
    return text if case_sensitive else " ".join(text.lower().split())


def _record_key(record: AudioRecord) -> tuple:
    return (record.dataset,
            record.source_id,
            record.youtube_id,
            record.start_seconds,
            record.end_seconds,
            record.original_label_ids,
            record.original_labels)
