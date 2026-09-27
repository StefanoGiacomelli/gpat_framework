"""Metadata parsing and format detection for supported YouTube datasets."""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import AUDIOSET, AUDIOSET_STRONG, NORMALIZED, VGGSOUND, AudioRecord, TemporalEvent
from .salt import SaltResolver


def detect_format(path: str | Path) -> str:
    """Detect supported metadata format from file structure and content."""

    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix == ".json":
        with metadata_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list) and data and "youtube_id" in data[0]:
            return NORMALIZED
        raise ValueError("JSON metadata is only supported for audioset_tools normalized records.")

    first_lines = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for _ in range(8):
            line = handle.readline()
            if not line:
                break
            if line.strip():
                first_lines.append(line.rstrip("\n"))

    joined = "\n".join(first_lines)
    
    if "segment_id\tstart_time_seconds\tend_time_seconds\tlabel" in joined:
        return AUDIOSET_STRONG
    
    if first_lines and first_lines[0].split(",")[:4] == ["youtube_id",
                                                         "start_seconds",
                                                         "end_seconds",
                                                         "original_label_ids"]:
        return NORMALIZED
    
    if joined.startswith("# Segments csv created") or "# YTID" in joined:
        return AUDIOSET
    
    if suffix == ".csv":
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            sample = handle.readline()
        row = next(csv.reader([sample]))
        if len(row) >= 4 and _looks_like_vggsound_row(row):
            return VGGSOUND
        if row and row[0] in {"YTID", "yt_id", "youtube_id"}:
            return AUDIOSET

    raise ValueError(f"Unsupported metadata format for {metadata_path}. Supported formats are "
                      "AudioSet Standard CSV, AudioSet Strong TSV, VGGSound CSV, and "
                      "audioset_tools normalized JSON/CSV.")


def load_label_map(labels_file: str | Path | None) -> dict[str, str]:
    """Load MID/KG-id to display label mappings from local metadata files."""

    if labels_file is None:
        return {}
    labels_path = Path(labels_file)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with labels_path.open(newline="", encoding="utf-8") as handle:
        first = handle.readline()
        handle.seek(0)
        
        if "," in first and "mid" in first and "display_name" in first:
            reader = csv.DictReader(handle)
            
            return {row["mid"].strip(): row["display_name"].strip()
                    for row in reader
                    if row.get("mid") and row.get("display_name")}
        
        if "\t" in first:
            mapping: dict[str, str] = {}
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    if row[0] == "mid":
                        continue
                    mapping[row[0].strip()] = row[1].strip()
            
            return mapping

        reader = csv.DictReader(handle)
        
        if reader.fieldnames and {"KnowledgeGraphId", "Name"}.issubset(reader.fieldnames):
            return {row["KnowledgeGraphId"].strip(): row["Name"].strip()
                    for row in reader
                    if row.get("KnowledgeGraphId") and row.get("Name")}
    
    return {}


def parse_metadata(path: str | Path,
                   labels_file: str | Path | None = None,
                   salt_resolver: SaltResolver | None = None,
                   dataset: str | None = None) -> list[AudioRecord]:
    """Parse supported metadata into normalized records."""

    metadata_path = Path(path)
    detected = dataset or detect_format(metadata_path)
    label_map = load_label_map(labels_file)

    if detected == AUDIOSET:
        return list(_parse_audioset_standard(metadata_path, label_map, salt_resolver))
    
    if detected == AUDIOSET_STRONG:
        return list(_parse_audioset_strong(metadata_path, label_map, salt_resolver))
    
    if detected == VGGSOUND:
        return list(_parse_vggsound(metadata_path, salt_resolver))
    
    if detected == NORMALIZED:
        return list(_parse_normalized(metadata_path))
    
    raise ValueError(f"Unsupported dataset type: {detected}")


def _parse_audioset_standard(path: Path,
                             label_map: dict[str, str],
                             salt_resolver: SaltResolver | None) -> Iterable[AudioRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader((line for line in handle if not line.startswith("#")),
                            skipinitialspace=True)
        for row in reader:
            if not row or len(row) < 4:
                continue
            if row[0] in {"YTID", "yt_id", "youtube_id"}:
                continue
            youtube_id = row[0].strip()
            start = _parse_float(row[1])
            end = _parse_float(row[2])
            label_cell = ",".join(row[3:])
            label_ids = _parse_label_list(label_cell)
            labels = tuple(label_map.get(label_id, label_id) for label_id in label_ids)
            salt_labels = (salt_resolver.resolve(AUDIOSET, labels) if salt_resolver else ())
            source_id = f"{youtube_id}_{_fmt_time(start)}_{_fmt_time(end)}"
            
            yield AudioRecord(dataset=AUDIOSET,
                              source_id=source_id,
                              youtube_id=youtube_id,
                              start_seconds=start,
                              end_seconds=end,
                              original_label_ids=tuple(label_ids),
                              original_labels=labels,
                              salt_labels=salt_labels,
                              raw={"YTID": youtube_id,
                                   "start_seconds": row[1].strip(),
                                   "end_seconds": row[2].strip(),
                                   "positive_labels": label_cell.strip()},
                              source_path=str(path))


def _parse_audioset_strong(path: Path,
                           label_map: dict[str, str],
                           salt_resolver: SaltResolver | None) -> Iterable[AudioRecord]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"segment_id", "start_time_seconds", "end_time_seconds", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"AudioSet Strong file is missing columns: {sorted(missing)}")
        for row in reader:
            grouped[row["segment_id"].strip()].append(row)

    for segment_id, rows in grouped.items():
        youtube_id, clip_start = _parse_strong_segment_id(segment_id)
        label_ids = tuple(dict.fromkeys(row["label"].strip() for row in rows))
        labels = tuple(label_map.get(label_id, label_id) for label_id in label_ids)
        salt_labels = (salt_resolver.resolve(AUDIOSET_STRONG, labels) if salt_resolver else ())
        events = []
        
        for row in rows:
            label_id = row["label"].strip()
            label = label_map.get(label_id, label_id)
            event_salt = (salt_resolver.resolve(AUDIOSET_STRONG, (label,)) if salt_resolver else ())
            events.append(TemporalEvent(start_seconds=_parse_float(row["start_time_seconds"]),
                                        end_seconds=_parse_float(row["end_time_seconds"]),
                                        original_label_id=label_id,
                                        original_label=label,
                                        salt_labels=event_salt,
                                        raw=dict(row)))

        yield AudioRecord(dataset=AUDIOSET_STRONG,
                          source_id=segment_id,
                          youtube_id=youtube_id,
                          start_seconds=clip_start,
                          end_seconds=clip_start + 10.0 if clip_start is not None else None,
                          original_label_ids=label_ids,
                          original_labels=labels,
                          salt_labels=salt_labels,
                          events=tuple(events),
                          raw={"segment_id": segment_id, "row_count": len(rows)},
                          source_path=str(path))


def _parse_vggsound(path: Path,
                    salt_resolver: SaltResolver | None) -> Iterable[AudioRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 4:
                continue
            youtube_id = row[0].strip()
            if youtube_id in {"youtube_id", "yt_id"}:
                continue
            start = _parse_float(row[1])
            label = row[2].strip()
            split = row[3].strip()
            salt_labels = (salt_resolver.resolve(VGGSOUND, (label,)) if salt_resolver else ())
            source_id = f"{youtube_id}_{_fmt_time(start)}"
            
            yield AudioRecord(dataset=VGGSOUND,
                              source_id=source_id,
                              youtube_id=youtube_id,
                              start_seconds=start,
                              end_seconds=start + 10.0 if start is not None else None,
                              original_label_ids=(),
                              original_labels=(label,),
                              salt_labels=salt_labels,
                              split=split,
                              raw={"youtube_id": youtube_id,
                                   "start_seconds": row[1].strip(),
                                   "label": label,
                                   "split": split},
                              source_path=str(path))


def _parse_normalized(path: Path) -> Iterable[AudioRecord]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    for row in rows:
        if isinstance(row, AudioRecord):
            yield row
            continue
        events = tuple(TemporalEvent(start_seconds=event.get("start_seconds"),
                                     end_seconds=event.get("end_seconds"),
                                     original_label_id=event.get("original_label_id"),
                                     original_label=event.get("original_label"),
                                     salt_labels=tuple(event.get("salt_labels", ())),
                                     raw=event.get("raw", {}))
                       for event in row.get("events", ()))
        
        yield AudioRecord(dataset=row["dataset"],
                          source_id=row["source_id"],
                          youtube_id=row["youtube_id"],
                          start_seconds=row.get("start_seconds"),
                          end_seconds=row.get("end_seconds"),
                          original_label_ids=tuple(row.get("original_label_ids", ())),
                          original_labels=tuple(row.get("original_labels", ())),
                          salt_labels=tuple(row.get("salt_labels", ())),
                          split=row.get("split"),
                          events=events,
                          raw=row.get("raw", {}),
                          source_path=row.get("source_path") or str(path))



def _parse_strong_segment_id(segment_id: str) -> tuple[str, float | None]:
    youtube_id, sep, start_ms = segment_id.rpartition("_")
    if not sep:
        return segment_id, None
    try:
        return youtube_id, float(start_ms) / 1000.0
    except ValueError:
        return youtube_id, None


def _parse_label_list(value: str) -> tuple[str, ...]:
    text = value.strip().strip('"')
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return tuple(str(item).strip() for item in parsed if str(item).strip())
        except (SyntaxError, ValueError):
            pass
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _parse_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "na"
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _looks_like_vggsound_row(row: list[str]) -> bool:
    if len(row) < 4:
        return False
    return bool(row[0].strip()) and _parse_float(row[1]) is not None
