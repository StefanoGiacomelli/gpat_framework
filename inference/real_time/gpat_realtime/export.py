"""Interoperable export for GP-AT real-time inference results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .engine import RealtimeInferenceResult


def export_inference_result(
    result: RealtimeInferenceResult,
    output_prefix: Union[str, Path],
    include_csv: bool = True,
) -> Dict[str, str]:
    """Write ``.npz`` arrays, ``.json`` metadata, and optional ``.csv``."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")

    arrays = {}
    arrays.update(result.full_forward_arrays())
    arrays.update(result.selected_arrays())
    np.savez_compressed(arrays_path, **arrays)
    metadata_path.write_text(json.dumps(result.to_metadata(), indent=2), encoding="utf-8")

    written = {"arrays": str(arrays_path), "metadata": str(metadata_path)}
    if include_csv:
        _write_csv(result, arrays, csv_path)
        written["csv"] = str(csv_path)
    return written


def load_inference_export(
    output_prefix: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> Dict:
    prefix = Path(output_prefix)
    arrays_path = prefix if prefix.suffix == ".npz" else prefix.with_suffix(".npz")
    meta_path = Path(metadata_path) if metadata_path else arrays_path.with_suffix(".json")
    if not arrays_path.exists():
        raise FileNotFoundError(f"NPZ export not found: {arrays_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"JSON metadata not found: {meta_path}")
    with np.load(arrays_path) as arrays:
        loaded_arrays = {key: arrays[key] for key in arrays.files}
    return {
        "arrays": loaded_arrays,
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
    }


def _write_csv(result: RealtimeInferenceResult, arrays: Dict[str, np.ndarray], path: Path) -> None:
    states = list(result.tracker.states.values())
    columns = ["timestamp"]
    for state in states:
        stem = _safe_name(state.class_info.name, state.class_info.index)
        columns.extend([f"{stem}_probability", f"{stem}_window_seconds"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        timestamps = arrays["selected_timestamps"]
        for row_idx, timestamp in enumerate(timestamps):
            row = {"timestamp": float(timestamp)}
            for class_idx, state in enumerate(states):
                stem = _safe_name(state.class_info.name, state.class_info.index)
                row[f"{stem}_probability"] = float(arrays["selected_probabilities"][row_idx, class_idx])
                row[f"{stem}_window_seconds"] = float(
                    arrays["selected_window_durations_seconds"][row_idx, class_idx]
                )
            writer.writerow(row)


def _safe_name(name: str, index: int) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return f"class_{index}_{text or 'unnamed'}"
