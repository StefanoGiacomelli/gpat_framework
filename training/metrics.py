"""Unified evaluation metrics for GP-AT training and inference."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
NoPositivePolicy = Literal["nan", "zero", "raise"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    
    return np.asarray(x)


def _validate_2d_same(target: np.ndarray, scores: np.ndarray) -> None:
    if target.shape != scores.shape:
        raise ValueError(f"target and scores must have the same shape, got {target.shape} and {scores.shape}")
    if target.ndim != 2:
        raise ValueError("expected arrays with shape (samples, classes)")


def micro_averaged_accuracy(scores: ArrayLike,
                            target: ArrayLike,
                            *,
                            threshold: float = 0.5) -> float:
    """Multi-label micro accuracy over all sample-class decisions.

    ``scores`` are thresholded with ``scores >= threshold``. The result is the
    fraction of correctly matched binary decisions across the flattened
    ``(samples, classes)`` matrix.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target).astype(bool)
    _validate_2d_same(y_true, y_score)
    if y_true.size == 0:
        return float("nan")
    
    return float(((y_score >= threshold) == y_true).mean())


def multilabel_f1(scores: ArrayLike,
                  target: ArrayLike,
                  *,
                  average: Literal["micro", "macro", "weighted", "samples"] = "micro",
                  threshold: float = 0.5,
                  zero_division: Union[int, float, Literal["warn"]] = 0) -> float:
    """Thresholded multi-label F1 using scikit-learn averaging semantics.

    This complements ranking metrics such as mAP and lwlrap by measuring the
    quality of binary decisions at a declared operating threshold. No threshold
    optimization is performed here.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target).astype(bool)
    _validate_2d_same(y_true, y_score)
    return float(f1_score(y_true, y_score >= threshold, average=average, zero_division=zero_division))


def average_precision_per_class(scores: ArrayLike,
                                target: ArrayLike,
                                *,
                                no_positive: NoPositivePolicy = "nan") -> np.ndarray:
    """Class-level AP values.

    Scores may be probabilities or arbitrary ranking scores. Classes with no
    positive targets return NaN by default, can return 0, or can raise.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target)
    _validate_2d_same(y_true, y_score)
    ap = np.empty(y_true.shape[1], dtype=float)
    for c in range(y_true.shape[1]):
        positives = np.sum(y_true[:, c] > 0)
        if positives == 0:
            if no_positive == "raise":
                raise ValueError(f"class {c} has no positive targets")
            ap[c] = np.nan if no_positive == "nan" else 0.0
        else:
            ap[c] = float(average_precision_score(y_true[:, c] > 0, y_score[:, c]))
    
    return ap


def mean_average_precision(scores: ArrayLike,
                           target: ArrayLike,
                           *,
                           no_positive: NoPositivePolicy = "nan",
                           ignore_nan: bool = True) -> float:
    """Mean AP over class-level AP values.

    By default, classes without positive examples are assigned NaN and ignored,
    matching common AudioSet-style class-wise mAP reporting.
    """

    ap = average_precision_per_class(scores, target, no_positive=no_positive)
    if ignore_nan:
        return float(np.nanmean(ap)) if not np.all(np.isnan(ap)) else float("nan")
    
    return float(np.mean(ap))


def _weighted_binary_clf_curve(y_true: np.ndarray,
                               y_score: np.ndarray,
                               *,
                               fps_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[order]
    y_score = y_score[order]
    fp_weight = np.ones_like(y_score, dtype=float) if fps_weight is None else np.asarray(fps_weight, dtype=float)[order]

    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true.astype(float))[threshold_idxs]
    fps = np.cumsum((~y_true).astype(float) * fp_weight)[threshold_idxs]
    
    return fps, tps, y_score[threshold_idxs]


def _weighted_average_precision(y_true: np.ndarray,
                                y_score: np.ndarray,
                                *,
                                fps_weight: Optional[np.ndarray] = None) -> float:
    fps, tps, thresholds = _weighted_binary_clf_curve(y_true, y_score, fps_weight=fps_weight)
    del thresholds
    if tps.size == 0 or tps[-1] == 0:
        return float("nan")
    precision = np.divide(tps, tps + fps, out=np.zeros_like(tps, dtype=float), where=(tps + fps) != 0)
    recall = tps / tps[-1]
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    
    return float(np.sum((recall[:-1] - recall[1:]) * precision[:-1]))


def _load_default_audioset_graph_distance() -> np.ndarray:
    path = Path(__file__).resolve().parent / "audioset_graph_distance.py"
    if not path.is_file():
        raise FileNotFoundError(
            "AudioSet graph-distance asset is missing. Expected "
            f"{path}. Pass graph_distance explicitly to override."
        )
    spec = importlib.util.spec_from_file_location("_gpat_audioset_graph_distance_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AudioSet graph distance module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return np.asarray(module.get_audioset_graph_distance(), dtype=float)


def _mask_distance_for_omap(distance: np.ndarray, threshold: float) -> np.ndarray:
    masked = np.asarray(distance, dtype=float).copy()
    masked[masked <= threshold] = 0.0
    off_diag = ~np.eye(masked.shape[0], dtype=bool)
    mean = masked[off_diag].mean()
    if mean > 1e-9:
        masked = masked / mean
    
    return masked


@dataclass(frozen=True)
class OmAPResult:
    average: float
    by_coarse_level: Dict[int, float]
    per_class_by_coarse_level: Dict[int, np.ndarray]


def ontology_mean_average_precision(scores: ArrayLike,
                                    target: ArrayLike,
                                    *,
                                    graph_distance: Optional[ArrayLike] = None) -> OmAPResult:
    """Ontology-aware mAP (OmAP) from Liu et al., Interspeech 2023.

    False positives are reweighted by the minimum ontology graph distance from
    the predicted class to the sample's positive target labels. The metric is
    evaluated for every coarse level ``lambda`` from 0 to ``max(distance)`` and
    averaged over classes and coarse levels.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target)
    _validate_2d_same(y_true, y_score)
    if np.any(y_true.sum(axis=1) <= 0):
        raise ValueError("OmAP requires at least one positive target per sample")

    distance = _load_default_audioset_graph_distance() if graph_distance is None else _to_numpy(graph_distance).astype(float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("graph_distance must be square")
    if distance.shape[0] != y_true.shape[1]:
        raise ValueError("graph_distance class count must match target shape")

    max_level = int(np.max(distance))
    per_level: Dict[int, np.ndarray] = {}
    means: Dict[int, float] = {}
    positive_indices = [np.flatnonzero(row > 0) for row in y_true]

    for level in range(max_level + 1):
        masked = _mask_distance_for_omap(distance, level)
        class_ap = np.empty(y_true.shape[1], dtype=float)
        for c in range(y_true.shape[1]):
            fps_weight = np.array([masked[pos, c].min() for pos in positive_indices], dtype=float)
            class_ap[c] = _weighted_average_precision(y_true[:, c] > 0, y_score[:, c], fps_weight=fps_weight)
        per_level[level] = class_ap
        means[level] = float(np.nanmean(class_ap)) if not np.all(np.isnan(class_ap)) else float("nan")

    return OmAPResult(average=float(np.nanmean(list(means.values()))),
                      by_coarse_level=means,
                      per_class_by_coarse_level=per_level)


def _one_sample_positive_class_precisions(scores: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=int)
    class_rankings[retrieved_classes] = np.arange(num_classes)
    retrieved_class_true = np.zeros(num_classes, dtype=bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    precision_at_hits = retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (
        1 + class_rankings[pos_class_indices].astype(float)
    )
    
    return pos_class_indices, precision_at_hits


@dataclass(frozen=True)
class LwlrapResult:
    overall: float
    per_class: np.ndarray
    class_weights: np.ndarray


def lwlrap(scores: ArrayLike, target: ArrayLike) -> LwlrapResult:
    """Label-weighted label-ranking average precision.

    This follows the DCASE reference implementation. Samples with no
    positive labels contribute no per-class precision. Classes with no positive
    labels receive per-class lwlrap 0 and class weight 0.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target)
    _validate_2d_same(y_true, y_score)
    num_samples, num_classes = y_score.shape
    precisions = np.zeros((num_samples, num_classes), dtype=float)
    for i in range(num_samples):
        pos, precision_at_hits = _one_sample_positive_class_precisions(y_score[i], y_true[i])
        precisions[i, pos] = precision_at_hits
    labels_per_class = np.sum(y_true > 0, axis=0)
    total_labels = float(np.sum(labels_per_class))
    class_weights = labels_per_class / total_labels if total_labels > 0 else np.zeros(num_classes)
    per_class = np.sum(precisions, axis=0) / np.maximum(1, labels_per_class)
    
    return LwlrapResult(overall=float(np.sum(per_class * class_weights)), per_class=per_class, class_weights=class_weights)


def _labels_from_scores_or_indices(values: ArrayLike) -> np.ndarray:
    arr = _to_numpy(values)
    if arr.ndim == 2:
        return arr.argmax(axis=1)
    if arr.ndim == 1:
        return arr
    raise ValueError("expected class indices with shape (samples,) or scores with shape (samples, classes)")


def _resolve_parent(label: Union[int, str],
                    class_to_parent: Optional[Mapping[Union[int, str], Union[int, str]]]) -> Union[int, str]:
    if class_to_parent is not None:
        return class_to_parent[label]
    if isinstance(label, str) and "-" in label:
        return label.split("-", 1)[0]
    raise ValueError("class_to_parent is required unless string labels use '<top>-<subclass>' format")


def hierarchical_accuracy(predictions: ArrayLike,
                          targets: ArrayLike,
                          *,
                          class_to_parent: Optional[Mapping[Union[int, str], Union[int, str]]] = None,
                          id_to_label: Optional[Mapping[int, str]] = None,
                          lambda_param: float = 0.5,
                          average: Literal["micro", "macro", "none"] = "macro") -> Union[float, Dict[Union[int, str], float]]:
    """Two-level hierarchical accuracy from the DCASE reference code.

    Exact second-level matches score 1. Wrong subclasses under the same
    top-level class score ``lambda_param``. Different top-level classes score 0.
    Macro averaging computes the per-true-class score and averages classes.
    """

    pred = _labels_from_scores_or_indices(predictions)
    true = _labels_from_scores_or_indices(targets)
    if pred.shape[0] != true.shape[0]:
        raise ValueError("predictions and targets must have the same number of samples")
    if id_to_label is not None:
        pred = np.array([id_to_label[int(x)] for x in pred], dtype=object)
        true = np.array([id_to_label[int(x)] for x in true], dtype=object)

    scores = np.empty(true.shape[0], dtype=float)
    for i, (p, t) in enumerate(zip(pred, true)):
        if p == t:
            scores[i] = 1.0
        elif _resolve_parent(p, class_to_parent) == _resolve_parent(t, class_to_parent):
            scores[i] = lambda_param
        else:
            scores[i] = 0.0
    if average == "micro":
        return float(scores.mean()) if scores.size else float("nan")
    per_class = {c: float(scores[true == c].mean()) for c in np.unique(true)}
    if average == "none":
        return per_class
    
    return float(np.mean(list(per_class.values()))) if per_class else float("nan")


@dataclass(frozen=True)
class HierarchicalPRF:
    precision: float
    recall: float
    f1: float
    per_class: Dict[Union[int, str], Tuple[float, float, float]]


@dataclass(frozen=True)
class HierarchicalMultilabelPRF:
    precision: float
    recall: float
    f1: float
    sample_precision: np.ndarray
    sample_recall: np.ndarray
    sample_f1: np.ndarray


def hierarchical_weighted_precision_recall_fscore(predictions: ArrayLike,
                                                  targets: ArrayLike,
                                                  *,
                                                  class_to_parent: Optional[Mapping[Union[int, str], Union[int, str]]] = None,
                                                  id_to_label: Optional[Mapping[int, str]] = None,
                                                  lambda_param: float = 0.75) -> HierarchicalPRF:
    """Hierarchical weighted PRF from ``dcase2026_metrics/evaluate.py``.

    For a class, precision averages samples predicted as that class; recall
    averages samples whose target is that class. Exact matches receive weight 1,
    same-top-level mistakes receive ``lambda_param``, and other mistakes 0.
    """

    pred = _labels_from_scores_or_indices(predictions)
    true = _labels_from_scores_or_indices(targets)
    if id_to_label is not None:
        pred = np.array([id_to_label[int(x)] for x in pred], dtype=object)
        true = np.array([id_to_label[int(x)] for x in true], dtype=object)
    classes = np.unique(true)
    per_class: Dict[Union[int, str], Tuple[float, float, float]] = {}

    def path(label: Union[int, str]) -> Tuple[Union[int, str], Union[int, str]]:
        return (_resolve_parent(label, class_to_parent), label)

    for c in classes:
        p_values = []
        r_values = []
        for p, t in zip(pred, true):
            pi = path(p)
            ti = path(t)
            intersection = len(set(pi).intersection(ti))
            w = 1.0 if p == t else (lambda_param if pi[0] == ti[0] else 0.0)
            if p == c:
                p_values.append((w * intersection) / len(pi))
            if t == c:
                r_values.append((w * intersection) / len(ti))
        precision = float(np.mean(p_values)) if p_values else 0.0
        recall = float(np.mean(r_values)) if r_values else 0.0
        fscore = 0.0 if precision == 0.0 and recall == 0.0 else 2 * precision * recall / (precision + recall)
        per_class[c] = (precision, recall, fscore)

    if not per_class:
        return HierarchicalPRF(float("nan"), float("nan"), float("nan"), per_class)
    values = np.array(list(per_class.values()), dtype=float)
    
    return HierarchicalPRF(float(values[:, 0].mean()), float(values[:, 1].mean()), float(values[:, 2].mean()), per_class)


def hierarchical_multilabel_precision_recall_fscore(scores: ArrayLike,
                                                    target: ArrayLike,
                                                    *,
                                                    lookup_table: Optional[Union[str, Path, torch.Tensor]] = None,
                                                    threshold: float = 0.5,
                                                    eps: float = 1e-12) -> HierarchicalMultilabelPRF:
    """Multi-label hierarchical PRF after HLP closure.

    Predictions are thresholded, then predictions and targets are propagated to
    unambiguous AudioSet ancestors through the HLP lookup table. Precision,
    recall, and F1 are macro-averaged over samples after set-style comparison
    of the propagated binary label vectors.
    """

    from .pre_processing import hierarchical_label_propagation

    y_score = torch.as_tensor(_to_numpy(scores), dtype=torch.float32)
    y_true = torch.as_tensor(_to_numpy(target), dtype=torch.float32)
    if y_score.shape != y_true.shape:
        raise ValueError(f"scores and target must have the same shape, got {tuple(y_score.shape)} and {tuple(y_true.shape)}")
    if y_score.ndim != 2:
        raise ValueError("expected arrays with shape (samples, classes)")

    pred = (y_score >= threshold).to(dtype=torch.float32)
    true = (y_true > 0).to(dtype=torch.float32)
    pred_h = hierarchical_label_propagation(pred, lookup_table=lookup_table) > 0
    true_h = hierarchical_label_propagation(true, lookup_table=lookup_table) > 0

    intersection = (pred_h & true_h).sum(dim=1).to(dtype=torch.float32)
    pred_count = pred_h.sum(dim=1).to(dtype=torch.float32)
    true_count = true_h.sum(dim=1).to(dtype=torch.float32)
    precision = torch.where(pred_count > 0, intersection / pred_count.clamp_min(eps), torch.zeros_like(intersection))
    recall = torch.where(true_count > 0, intersection / true_count.clamp_min(eps), torch.zeros_like(intersection))
    f1 = torch.where((precision + recall) > 0,
                     2 * precision * recall / (precision + recall).clamp_min(eps),
                     torch.zeros_like(precision))

    return HierarchicalMultilabelPRF(precision=float(precision.mean()) if precision.numel() else float("nan"),
                                     recall=float(recall.mean()) if recall.numel() else float("nan"),
                                     f1=float(f1.mean()) if f1.numel() else float("nan"),
                                     sample_precision=precision.numpy(),
                                     sample_recall=recall.numpy(),
                                     sample_f1=f1.numpy())


def hlp_expanded_mean_average_precision(scores: ArrayLike,
                                        target: ArrayLike,
                                        *,
                                        lookup_table: Optional[Union[str, Path, torch.Tensor]] = None,
                                        no_positive: NoPositivePolicy = "nan",
                                        ignore_nan: bool = True) -> float:
    """mAP after soft prediction propagation and target HLP closure."""

    from .pre_processing import hierarchical_label_propagation
    from .post_processing import hierarchical_prediction_propagation

    y_score = torch.as_tensor(_to_numpy(scores), dtype=torch.float32)
    y_true = torch.as_tensor(_to_numpy(target), dtype=torch.float32)
    if y_score.shape != y_true.shape:
        raise ValueError(f"scores and target must have the same shape, got {tuple(y_score.shape)} and {tuple(y_true.shape)}")
    if y_score.ndim != 2:
        raise ValueError("expected arrays with shape (samples, classes)")

    score_h = hierarchical_prediction_propagation(y_score, lookup_table=lookup_table, mode="soft")
    target_h = hierarchical_label_propagation((y_true > 0).to(dtype=torch.float32), lookup_table=lookup_table)
    
    return mean_average_precision(score_h, target_h, no_positive=no_positive, ignore_nan=ignore_nan)


def class_specific_f1(scores: ArrayLike,
                      target: ArrayLike,
                      classes: Union[int, str, Iterable[Union[int, str]]],
                      *,
                      label_to_index: Optional[Mapping[str, int]] = None,
                      threshold: float = 0.5,
                      zero_division: Union[int, float, Literal["warn"]] = 0) -> Union[float, Dict[Union[int, str], float]]:
    """F1 for caller-selected multi-label class(es).

    ``classes`` may contain integer indices or class names. Class names require
    an explicit ``label_to_index`` mapping; no AudioSet ordering is assumed.
    """

    y_score = _to_numpy(scores)
    y_true = _to_numpy(target)
    _validate_2d_same(y_true, y_score)
    single = isinstance(classes, (int, str))
    selected = [classes] if single else list(classes)
    result: Dict[Union[int, str], float] = {}
    binary = y_score >= threshold
    for cls in selected:
        if isinstance(cls, str):
            if label_to_index is None or cls not in label_to_index:
                raise ValueError(f"class name {cls!r} requires a label_to_index mapping")
            idx = label_to_index[cls]
        else:
            idx = int(cls)
        result[cls] = float(f1_score(y_true[:, idx] > 0, binary[:, idx], zero_division=zero_division))
    
    return result[selected[0]] if single else result
