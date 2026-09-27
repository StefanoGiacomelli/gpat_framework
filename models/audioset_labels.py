"""AudioSet class-order helpers shared by vendored GP-AT wrappers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

AUDIOSET_CLASS_COUNT = 527


def default_audioset_labels_path() -> Path:
    """Return the repository-local canonical AudioSet class-index metadata."""
    return Path(__file__).resolve().parents[1] / "datasets" / "AudioSet_meta" / "class_labels_indices.csv"


def load_canonical_audioset_mids(path: Optional[Path | str] = None) -> List[str]:
    """Load canonical AudioSet MIDs ordered by the explicit ``index`` column."""
    labels_path = Path(path) if path is not None else default_audioset_labels_path()
    if not labels_path.is_file():
        raise FileNotFoundError(
            "canonical AudioSet metadata not found: "
            f"{labels_path}. Expected datasets/AudioSet_meta/class_labels_indices.csv"
        )

    rows = []
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["index"]), row["mid"]))

    rows.sort(key=lambda item: item[0])
    indices = [index for index, _ in rows]
    expected = list(range(AUDIOSET_CLASS_COUNT))
    if indices != expected:
        raise ValueError(
            "AudioSet class index metadata must contain each canonical index "
            f"0..{AUDIOSET_CLASS_COUNT - 1} exactly once"
        )

    mids = [mid for _, mid in rows]
    if len(set(mids)) != AUDIOSET_CLASS_COUNT:
        raise ValueError("AudioSet MID metadata contains duplicate class identifiers")

    return mids


def canonical_reorder_index(
    source_mids: Sequence[str] | Iterable[str],
    *,
    canonical_mids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Return indices mapping a source MID order into canonical AudioSet order.

    For source predictions ``p`` arranged according to ``source_mids``, the
    canonical prediction matrix is ``p.index_select(-1, permutation)``.
    """
    source = list(source_mids)
    canonical = list(canonical_mids) if canonical_mids is not None else load_canonical_audioset_mids()

    if len(source) != AUDIOSET_CLASS_COUNT or len(set(source)) != AUDIOSET_CLASS_COUNT:
        raise ValueError(
            f"source AudioSet ordering must contain {AUDIOSET_CLASS_COUNT} unique MIDs"
        )
    if set(source) != set(canonical):
        missing = sorted(set(canonical) - set(source))
        extra = sorted(set(source) - set(canonical))
        raise ValueError(
            "source and canonical AudioSet MID sets differ: "
            f"missing={missing}, extra={extra}"
        )

    source_index = {mid: index for index, mid in enumerate(source)}
    permutation = torch.tensor(
        [source_index[mid] for mid in canonical],
        dtype=torch.long,
    )

    if sorted(permutation.tolist()) != list(range(AUDIOSET_CLASS_COUNT)):
        raise ValueError("AudioSet class-order permutation is not bijective")

    return permutation
