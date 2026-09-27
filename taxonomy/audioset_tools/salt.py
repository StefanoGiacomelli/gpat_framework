"""Optional SALT label resolution for AudioSet-Tools."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _norm_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


class SaltResolver:
    """Resolve dataset labels to SALT standardized labels.

    The resolver is deliberately lightweight: it can read SALT's TSV/JSON assets
    directly, and it can also be constructed from a py_salt explorer object when
    callers need py-salt's richer taxonomy navigation elsewhere.
    """

    def __init__(self,
                 mapping: dict[tuple[str, str], set[str]],
                 roots: dict | None = None) -> None:
        self._mapping = mapping
        self._roots = roots or {}
        self._children = self._build_children_index(self._roots)

    @classmethod
    def from_mapping_file(cls,
                          mapping_file: str | Path | None = None,
                          roots_file: str | Path | None = None) -> "SaltResolver":
        """Create a resolver from SALT asset files.

        Defaults point to the local vendored SALT assets relative to
        ``taxonomy/audioset_tools``.
        """

        package_dir = Path(__file__).resolve().parent
        taxonomy_dir = package_dir.parent
        if mapping_file is None:
            mapping_file = (taxonomy_dir / "salt" / "py-salt" / "assets" / "salt_event_mapping.tsv")
        if roots_file is None:
            roots_file = (taxonomy_dir / "salt" / "py-salt" / "assets" / "salt_event_roots.json")

        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(f"SALT mapping file not found: {mapping_path}")

        mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"standard_event", "dataset_label", "dataset"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"SALT mapping file is missing columns: {sorted(missing)}")
            for row in reader:
                dataset = row["dataset"].strip()
                dataset_label = row["dataset_label"].strip()
                standard_event = row["standard_event"].strip()
                if dataset and dataset_label and standard_event:
                    mapping[(dataset, _norm_label(dataset_label))].add(standard_event)

        roots = {}
        roots_path = Path(roots_file)
        if roots_path.exists():
            with roots_path.open(encoding="utf-8") as handle:
                roots = json.load(handle)

        return cls(dict(mapping), roots=roots)

    @classmethod
    def from_py_salt(cls, explorer=None) -> "SaltResolver":
        """Create a resolver using py-salt's default EventExplorer assets.

        This keeps ``audioset_tools`` usable without importing py_salt at module
        import time. If py_salt is unavailable to the caller's environment, this
        method raises an ImportError with a precise message.
        """

        if explorer is None:
            try:
                from py_salt import EventExplorer
            except ImportError as exc:
                raise ImportError("py_salt is not importable. Use SaltResolver.from_mapping_file() "
                                  "or install/configure py-salt on PYTHONPATH.") from exc
            explorer = EventExplorer()

        mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
        for _, row in explorer.map_df.iterrows():
            dataset = str(row["dataset"]).strip()
            dataset_label = str(row["dataset_label"]).strip()
            standard_event = str(row[explorer._std_col]).strip()
            if dataset and dataset_label and standard_event:
                mapping[(dataset, _norm_label(dataset_label))].add(standard_event)

        return cls(dict(mapping), roots=getattr(explorer, "roots", {}))

    def resolve(self, dataset: str, labels: Iterable[str]) -> tuple[str, ...]:
        """Resolve source labels for one mapped dataset into SALT labels."""

        resolved: list[str] = []
        seen = set()
        for label in labels:
            for salt_label in sorted(self._mapping.get((dataset, _norm_label(label)), ())):
                if salt_label not in seen:
                    seen.add(salt_label)
                    resolved.append(salt_label)
        
        return tuple(resolved)

    def expand_salt_labels(self,
                           labels: Iterable[str],
                           include_descendants: bool = False) -> tuple[str, ...]:
        """Expand SALT labels with descendants from SALT roots when requested."""

        result: list[str] = []
        seen = set()
        for label in labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
            if include_descendants:
                for descendant in self.descendants(label):
                    if descendant not in seen:
                        seen.add(descendant)
                        result.append(descendant)
        
        return tuple(result)

    def descendants(self, label: str) -> tuple[str, ...]:
        """Return descendants for a SALT label from the roots JSON."""

        output: list[str] = []
        queue = list(self._children.get(label, ()))
        while queue:
            current = queue.pop(0)
            output.append(current)
            queue.extend(self._children.get(current, ()))
        
        return tuple(output)

    @staticmethod
    def _build_children_index(roots: dict) -> dict[str, tuple[str, ...]]:
        children: dict[str, list[str]] = defaultdict(list)

        def walk(node: dict | list, parent: str | None = None) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if parent is not None:
                        children[parent].append(key)
                    walk(value, key)

        walk(roots)
        
        return {key: tuple(dict.fromkeys(values)) for key, values in children.items()}
