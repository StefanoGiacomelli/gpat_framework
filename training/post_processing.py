"""Post-processing utilities for hierarchy-consistent predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

from torch import Tensor

try:
    from .pre_processing import HierarchicalLabelPropagation, load_hlp_lookup_table
except ImportError:  # Allows direct execution as a script from training/.
    from pre_processing import HierarchicalLabelPropagation, load_hlp_lookup_table

PostHLPMode = Literal["soft", "hard", "soft_then_threshold"]


class HierarchicalPredictionPropagation(HierarchicalLabelPropagation):
    """HLP applied to model outputs.

    The HLP paper describes post-processing as updating each unambiguous parent
    score with ``max(parent_score, child_score)``. ``soft`` mode returns these
    propagated scores. ``hard`` thresholds first and then propagates binary
    decisions. ``soft_then_threshold`` propagates scores before thresholding.
    """

    def postprocess(self,
                    scores: Tensor,
                    *,
                    threshold: float = 0.5,
                    mode: PostHLPMode = "soft") -> Tensor:
        if mode not in {"soft", "hard", "soft_then_threshold"}:
            raise ValueError("mode must be 'soft', 'hard', or 'soft_then_threshold'")
        if mode == "hard":
            return self.propagate((scores >= threshold).to(dtype=scores.dtype))
        propagated = self.propagate(scores)
        if mode == "soft_then_threshold":
            return (propagated >= threshold).to(dtype=scores.dtype)
        
        return propagated


def hierarchical_prediction_propagation(scores: Tensor,
                                        lookup_table: Optional[Union[str, Path, Tensor]] = None,
                                        *,
                                        threshold: float = 0.5,
                                        mode: PostHLPMode = "soft") -> Tensor:
    """Functional wrapper for HLP post-processing over prediction scores."""

    return HierarchicalPredictionPropagation(lookup_table).postprocess(scores, threshold=threshold, mode=mode)


__all__ = ["HierarchicalPredictionPropagation",
           "hierarchical_prediction_propagation",
           "load_hlp_lookup_table"]
