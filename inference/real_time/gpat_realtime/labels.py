"""AudioSet label resolution for GP-AT model outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


ClassSelector = Union[int, str]


@dataclass(frozen=True)
class ClassInfo:
    index: int
    mid: Optional[str]
    name: str
    selector: ClassSelector


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def default_audioset_labels_path(project_root: Optional[Union[str, Path]] = None) -> Path:
    root = Path(project_root) if project_root else project_root_from_file()
    return root / "datasets" / "AudioSet_meta" / "class_labels_indices.csv"


class AudioSetLabelResolver:
    """Resolve AudioSet class indices by display name, MID, or integer index."""

    def __init__(self, labels_path: Union[str, Path]):
        self.labels_path = Path(labels_path)
        if not self.labels_path.exists():
            raise FileNotFoundError(f"AudioSet label file not found: {self.labels_path}")
        self.by_index: Dict[int, ClassInfo] = {}
        self._by_name: Dict[str, int] = {}
        self._by_mid: Dict[str, int] = {}
        self._load()

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(value.strip().casefold().replace("_", " ").split())

    def _load(self) -> None:
        with self.labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"index", "mid", "display_name"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.labels_path} missing columns: {sorted(missing)}")
            for row in reader:
                index = int(row["index"])
                mid = row["mid"].strip()
                name = row["display_name"].strip()
                info = ClassInfo(index=index, mid=mid, name=name, selector=index)
                self.by_index[index] = info
                self._by_name[self._normalize_name(name)] = index
                if mid:
                    self._by_mid[mid] = index

    def resolve(self, selector: ClassSelector, output_size: Optional[int] = None) -> ClassInfo:
        if isinstance(selector, bool):
            raise ValueError("Boolean class selectors are invalid")
        if isinstance(selector, int):
            return self._resolve_index(selector, output_size, selector)
        text = str(selector).strip()
        if not text:
            raise ValueError("Class selector cannot be empty")
        if text.isdigit():
            return self._resolve_index(int(text), output_size, selector)
        if text in self._by_mid:
            return self._resolve_index(self._by_mid[text], output_size, selector)
        normalized = self._normalize_name(text)
        if normalized in self._by_name:
            return self._resolve_index(self._by_name[normalized], output_size, selector)
        raise ValueError(
            f"Unknown class selector '{selector}'. Use an AudioSet display name, MID, or output index."
        )

    def resolve_many(
        self, selectors: Iterable[ClassSelector], output_size: Optional[int] = None
    ) -> List[ClassInfo]:
        resolved: List[ClassInfo] = []
        seen = set()
        for selector in selectors:
            info = self.resolve(selector, output_size=output_size)
            if info.index in seen:
                raise ValueError(f"Duplicate monitored class index {info.index} ({info.name})")
            seen.add(info.index)
            resolved.append(info)
        if not resolved:
            raise ValueError("At least one monitored class must be selected")
        return resolved

    def _resolve_index(
        self, index: int, output_size: Optional[int], selector: ClassSelector
    ) -> ClassInfo:
        if index < 0:
            raise ValueError(f"Class index must be non-negative, got {index}")
        if output_size is not None and index >= output_size:
            raise ValueError(f"Class index {index} out of range for output size {output_size}")
        if index in self.by_index:
            info = self.by_index[index]
            return ClassInfo(index=info.index, mid=info.mid, name=info.name, selector=selector)
        return ClassInfo(index=index, mid=None, name=f"class_{index}", selector=selector)
