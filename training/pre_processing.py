"""Pre-processing utilities for hierarchy-aware labels."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor


def _default_hlp_lookup_table_path() -> Path:
    return Path(__file__).resolve().parent / "HLP_lookup_table.pt"


def load_hlp_lookup_table(table: Optional[Union[str, Path, Tensor]] = None) -> Tensor:
    """Load an HLP lookup table.

    The lookup table is square ``(classes, classes)`` and contains 1 at
    ``[child, ancestor]`` when the ancestor is an unambiguous parent or
    transitive ancestor of the child. ``None`` loads the local AudioSet table.
    """

    if table is None:
        path = _default_hlp_lookup_table_path()
        if not path.is_file():
            raise FileNotFoundError(
                "HLP lookup table asset is missing. Expected "
                f"{path}. Pass table explicitly to override."
            )
        loaded = torch.load(path, weights_only=True)
    elif isinstance(table, Tensor):
        loaded = table
    else:
        loaded = torch.load(Path(table), weights_only=True)
    if not isinstance(loaded, Tensor):
        raise ValueError("HLP lookup table must be a torch.Tensor")
    if loaded.ndim != 2 or loaded.shape[0] != loaded.shape[1]:
        raise ValueError("HLP lookup table must be square with shape (classes, classes)")
    
    return loaded


class HierarchicalLabelPropagation:
    """Hierarchical Label Propagation (HLP) for training labels.

    This follows the local HLP reference implementation and ICASSP 2025 paper:
    positive child labels are propagated upward only through unambiguous
    ancestor relations. For continuous values, the same max-propagation is
    differentiable almost everywhere and is valid when larger values mean
    stronger label confidence.
    """

    def __init__(self, lookup_table: Optional[Union[str, Path, Tensor]] = None) -> None:
        self.lookup_table = load_hlp_lookup_table(lookup_table)

    @property
    def shape(self) -> torch.Size:
        return self.lookup_table.shape

    def propagate(self, labels: Tensor) -> Tensor:
        """Propagate labels shaped ``(classes,)`` or ``(batch, classes)``."""

        if labels.ndim not in {1, 2}:
            raise ValueError("labels must have shape (classes,) or (batch, classes)")
        squeeze = labels.ndim == 1
        batch = labels.unsqueeze(0) if squeeze else labels
        if batch.shape[1] != self.lookup_table.shape[0]:
            raise ValueError(
                f"labels have {batch.shape[1]} classes but lookup table has "
                f"{self.lookup_table.shape[0]}"
            )
        table = self.lookup_table.to(device=batch.device, dtype=batch.dtype)
        propagated = (batch.unsqueeze(-1) * table).max(dim=1).values
        
        return propagated.squeeze(0) if squeeze else propagated


def hierarchical_label_propagation(labels: Tensor,
                                   lookup_table: Optional[Union[str, Path, Tensor]] = None) -> Tensor:
    """Functional wrapper for pre-training HLP over target labels."""

    return HierarchicalLabelPropagation(lookup_table).propagate(labels)
