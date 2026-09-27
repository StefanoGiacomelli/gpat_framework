"""Unified training losses for GP-AT models.

The public model wrappers in ``models/`` return sigmoid probabilities of
shape ``(batch, classes)``. These losses therefore accept probabilities by
default and expose an explicit ``from_logits`` switch for raw classifier heads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    _validate_reduction(reduction)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    
    return loss.mean()


def _validate_prediction_target(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target must have identical shapes, got "
                         f"{tuple(prediction.shape)} and {tuple(target.shape)}")
    if prediction.ndim < 2:
        raise ValueError("expected prediction and target with shape (batch, classes[, ...])")
    if not torch.is_floating_point(prediction):
        raise TypeError("prediction must be a floating point tensor")
    if not torch.is_floating_point(target):
        raise TypeError("target must be a floating point tensor")


def _validate_probabilities(prediction: Tensor, *, name: str = "prediction") -> None:
    if prediction.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if torch.any((prediction < 0) | (prediction > 1)):
        raise ValueError(f"{name} contains values outside [0, 1]; pass from_logits=True for logits")


def _as_tensor(value: Optional[Union[float, Sequence[float], Tensor]],
               *,
               like: Tensor) -> Optional[Tensor]:
    if value is None:
        return None
    if isinstance(value, Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _load_default_audioset_graph_distance() -> Tensor:
    path = Path(__file__).resolve().parent / "audioset_graph_distance.py"
    if not path.is_file():
        raise FileNotFoundError(
            "AudioSet graph-distance asset is missing. Expected "
            f"{path}. Pass graph_distance explicitly to override."
        )
    spec = importlib.util.spec_from_file_location("_gpat_audioset_graph_distance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load AudioSet graph distance module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return torch.as_tensor(module.get_audioset_graph_distance(), dtype=torch.float32)


def _prepare_graph_distance(graph_distance: Optional[Union[Tensor, Sequence[Sequence[float]]]],
                            *,
                            target: Tensor) -> Tensor:
    distance = (_load_default_audioset_graph_distance()
                if graph_distance is None
                else torch.as_tensor(graph_distance, dtype=target.dtype))
    distance = distance.to(device=target.device, dtype=target.dtype)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("graph_distance must be a square matrix with shape (classes, classes)")
    if distance.shape[0] != target.shape[1]:
        raise ValueError(f"graph_distance has {distance.shape[0]} classes but target has {target.shape[1]}")
    if torch.any(distance < 0):
        raise ValueError("graph_distance must contain non-negative distances")
    
    return distance


def binary_cross_entropy(prediction: Tensor,
                         target: Tensor,
                         *,
                         from_logits: bool = False,
                         weight: Optional[Tensor] = None,
                         pos_weight: Optional[Tensor] = None,
                         reduction: Reduction = "mean") -> Tensor:
    """Multi-label BCE for tensors shaped ``(batch, classes[, ...])``.

    ``prediction`` is interpreted as sigmoid probabilities unless
    ``from_logits=True``. ``target`` must have the same shape and contain
    multi-hot labels or soft labels in ``[0, 1]``. ``weight`` follows PyTorch's
    element-wise BCE weighting semantics. ``pos_weight`` is supported only for
    logits, matching ``torch.nn.functional.binary_cross_entropy_with_logits``.
    """

    _validate_prediction_target(prediction, target)
    _validate_reduction(reduction)
    if from_logits:
        return F.binary_cross_entropy_with_logits(prediction, target, weight=weight, pos_weight=pos_weight, reduction=reduction)
    if pos_weight is not None:
        raise ValueError("pos_weight requires from_logits=True")
    _validate_probabilities(prediction)
    
    return F.binary_cross_entropy(prediction, target, weight=weight, reduction=reduction)


class BinaryCrossEntropyLoss(nn.Module):
    """``nn.Module`` wrapper for :func:`binary_cross_entropy`."""

    def __init__(self,
                 *,
                 from_logits: bool = False,
                 weight: Optional[Tensor] = None,
                 pos_weight: Optional[Tensor] = None,
                 reduction: Reduction = "mean") -> None:
        super().__init__()
        self.from_logits = from_logits
        self.register_buffer("weight", weight if isinstance(weight, Tensor) else None)
        self.register_buffer("pos_weight", pos_weight if isinstance(pos_weight, Tensor) else None)
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(prediction,
                                    target,
                                    from_logits=self.from_logits,
                                    weight=self.weight,
                                    pos_weight=self.pos_weight,
                                    reduction=self.reduction)


def ontology_loss_weight(target: Tensor,
                         *,
                         graph_distance: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
                         beta: float = 1.0,
                         eps: float = 1e-12) -> Tensor:
    """Return the OBCE class-wise weight matrix from the reference algorithm.

    The reference source is
    ``training/ontology-aware-audio-tagging/ontology_audio_tagging/loss_and_eval_metric.py``.
    For each sample, class ``c`` receives the minimum ontology distance to any
    positive label, raised to ``beta``. The row is normalized by its maximum,
    target labels are set to weight 1, and the complete batch is divided by its
    mean to keep the expected BCE scale.
    """

    if target.ndim != 2:
        raise ValueError("target must have shape (batch, classes)")
    if not torch.is_floating_point(target):
        target = target.float()
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if torch.any(target.sum(dim=1) <= 0):
        raise ValueError("OBCE requires at least one positive target per sample")

    distance = _prepare_graph_distance(graph_distance, target=target)
    max_distance = distance.max()
    if max_distance <= 0:
        raise ValueError("graph_distance must contain at least one positive distance")

    graph_weight = (distance / max_distance).pow(beta)
    weighted_targets = target.unsqueeze(1) * graph_weight.unsqueeze(0)
    inf = torch.tensor(torch.inf, device=target.device, dtype=target.dtype)
    weighted_targets = torch.where(weighted_targets == 0, inf, weighted_targets)
    weight = weighted_targets.min(dim=2).values
    weight = torch.where(torch.isinf(weight), torch.zeros_like(weight), weight)

    row_max = weight.max(dim=1, keepdim=True).values.clamp_min(eps)
    weight = weight / row_max
    weight = torch.where(target > 0, torch.ones_like(weight), weight)
    
    return weight / weight.mean().clamp_min(eps)


def ontology_binary_cross_entropy(prediction: Tensor,
                                  target: Tensor,
                                  *,
                                  from_logits: bool = False,
                                  graph_distance: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
                                  beta: float = 1.0,
                                  ontology_balance: float = 0.5,
                                  reduction: Reduction = "mean",
                                  eps: float = 1e-7) -> Tensor:
    """Ontology-Based BCE (OBCE) from Liu et al., Interspeech 2023.

    The original formulation combines standard BCE and ontology-weighted BCE
    as ``(BCE + OBCE) / 2``. ``ontology_balance`` is the explicit generalized
    mixing coefficient for the ontology-weighted term:
    ``(1 - ontology_balance) * BCE + ontology_balance * OBCE``.
    The default ``0.5`` reproduces the reference implementation.
    """

    _validate_prediction_target(prediction, target)
    if not 0.0 <= ontology_balance <= 1.0:
        raise ValueError("ontology_balance must be in [0, 1]")
    _validate_reduction(reduction)

    if from_logits:
        elementwise = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    else:
        _validate_probabilities(prediction)
        elementwise = F.binary_cross_entropy(prediction.clamp(eps, 1.0 - eps), target, reduction="none")

    loss_weight = ontology_loss_weight(target, graph_distance=graph_distance, beta=beta)
    weighted = elementwise * loss_weight
    combined = (1.0 - ontology_balance) * elementwise + ontology_balance * weighted
    
    return _reduce(combined, reduction)


class OntologyBinaryCrossEntropyLoss(nn.Module):
    """``nn.Module`` wrapper for :func:`ontology_binary_cross_entropy`."""

    def __init__(self,
                 *,
                 from_logits: bool = False,
                 graph_distance: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
                 beta: float = 1.0,
                 ontology_balance: float = 0.5,
                 reduction: Reduction = "mean") -> None:
        super().__init__()
        self.from_logits = from_logits
        self.beta = beta
        self.ontology_balance = ontology_balance
        self.reduction = reduction
        graph = None if graph_distance is None else torch.as_tensor(graph_distance, dtype=torch.float32)
        self.register_buffer("graph_distance", graph)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return ontology_binary_cross_entropy(prediction,
                                             target,
                                             from_logits=self.from_logits,
                                             graph_distance=self.graph_distance,
                                             beta=self.beta,
                                             ontology_balance=self.ontology_balance,
                                             reduction=self.reduction)


def focal_loss(prediction: Tensor,
               target: Tensor,
               *,
               alpha: Optional[Union[float, Sequence[float], Tensor]] = None,
               gamma: float = 2.0,
               from_logits: bool = False,
               reduction: Reduction = "mean",
               eps: float = 1e-7) -> Tensor:
    """Multi-label sigmoid focal loss.

    This is the binary/multi-label variant: each class is treated as an
    independent Bernoulli target. ``prediction`` and ``target`` must have the
    same shape. Scores are probabilities unless ``from_logits=True``.
    ``alpha`` may be a scalar, class vector, or tensor broadcastable to the
    prediction shape.
    """

    _validate_prediction_target(prediction, target)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    _validate_reduction(reduction)

    if from_logits:
        bce = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        prob = torch.sigmoid(prediction)
    else:
        _validate_probabilities(prediction)
        prob = prediction.clamp(eps, 1.0 - eps)
        bce = F.binary_cross_entropy(prob, target, reduction="none")

    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    loss = bce * (1.0 - p_t).pow(gamma)
    alpha_t = _as_tensor(alpha, like=prediction)
    if alpha_t is not None:
        loss = loss * (alpha_t * target + (1.0 - alpha_t) * (1.0 - target))
    
    return _reduce(loss, reduction)


class FocalLoss(nn.Module):
    """``nn.Module`` wrapper for :func:`focal_loss`."""

    def __init__(self,
                 *,
                 alpha: Optional[Union[float, Sequence[float], Tensor]] = None,
                 gamma: float = 2.0,
                 from_logits: bool = False,
                 reduction: Reduction = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return focal_loss(prediction,
                          target,
                          alpha=self.alpha,
                          gamma=self.gamma,
                          from_logits=self.from_logits,
                          reduction=self.reduction)


def ontology_focal_loss(prediction: Tensor,
                        target: Tensor,
                        *,
                        graph_distance: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
                        beta: float = 1.0,
                        alpha: Optional[Union[float, Sequence[float], Tensor]] = None,
                        gamma: float = 2.0,
                        ontology_balance: float = 1.0,
                        from_logits: bool = False,
                        reduction: Reduction = "mean") -> Tensor:
    """Proposed ontology-aware focal loss.

    It is a rooted extension of sigmoid focal loss using the OBCE distance weight 
    ``r`` from Liu et al.: ``FL_ontology = FL * ((1-a) + a*r)``, where ``a`` is
    ``ontology_balance``. Setting ``ontology_balance=0`` recovers focal loss.
    """

    if not 0.0 <= ontology_balance <= 1.0:
        raise ValueError("ontology_balance must be in [0, 1]")
    elementwise = focal_loss(prediction,
                             target,
                             alpha=alpha,
                             gamma=gamma,
                             from_logits=from_logits,
                             reduction="none")
    weight = ontology_loss_weight(target, graph_distance=graph_distance, beta=beta)
    combined_weight = (1.0 - ontology_balance) + ontology_balance * weight
    
    return _reduce(elementwise * combined_weight, reduction)


def top_class_penalty(logits: Tensor,
                      target: Tensor,
                      top_class_of_class: Union[Tensor, Sequence[int]],
                      *,
                      reduction: Reduction = "mean") -> Tensor:
    """Top-class penalty from ``Hierarchical_Learning`` Eq. 5.

    ``logits`` has shape ``(batch, second_level_classes)``. ``target`` contains
    second-level class indices. ``top_class_of_class[c]`` gives the top-level
    class index for second-level class ``c``. The indicator uses ``argmax`` and
    is therefore not differentiable, matching the paper formulation.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    target = target.to(device=logits.device, dtype=torch.long)
    mapping = torch.as_tensor(top_class_of_class, device=logits.device, dtype=torch.long)
    if mapping.ndim != 1 or mapping.numel() != logits.shape[1]:
        raise ValueError("top_class_of_class must be a vector with one entry per class")
    pred = logits.argmax(dim=1)
    loss = (mapping[pred] != mapping[target]).to(dtype=logits.dtype)
    
    return _reduce(loss, reduction)


def supervised_top_class_contrastive_loss(embeddings: Tensor,
                                          top_targets: Tensor,
                                          *,
                                          temperature: float = 0.5,
                                          eps: float = 1e-12) -> Tensor:
    """Supervised contrastive top-class loss from ``Hierarchical_Learning``.

    Samples sharing the same top-level target are positives. Samples without a
    positive partner in the batch contribute zero to the batch mean.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (batch, embedding_dim)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    top_targets = top_targets.to(device=embeddings.device, dtype=torch.long)
    if top_targets.shape[0] != embeddings.shape[0]:
        raise ValueError("top_targets must contain one label per embedding")

    z = F.normalize(embeddings, dim=1)
    logits = z @ z.T / temperature
    batch = z.shape[0]
    eye = torch.eye(batch, device=z.device, dtype=torch.bool)
    positive_mask = (top_targets[:, None] == top_targets[None, :]) & ~eye
    denominator_mask = ~eye

    exp_logits = torch.exp(logits) * denominator_mask.to(dtype=z.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(eps))
    positives_per_row = positive_mask.sum(dim=1)
    row_loss = torch.zeros(batch, device=z.device, dtype=z.dtype)
    valid = positives_per_row > 0
    if valid.any():
        row_loss[valid] = -((log_prob * positive_mask.to(dtype=z.dtype)).sum(dim=1)[valid]
                            / positives_per_row[valid].to(dtype=z.dtype))
    
    return row_loss.mean()


class HierarchicalLoss(nn.Module):
    """Hierarchical single-label loss from ``Hierarchical_Learning``.

    The loss is ``cross_entropy + top_penalty_weight * L_top`` plus, when
    embeddings are supplied, ``contrastive_weight * L_contrastive``. It is for
    two-level single-label taxonomies such as BST, not multi-label AudioSet.
    """

    def __init__(self,
                 top_class_of_class: Union[Tensor, Sequence[int]],
                 *,
                 top_penalty_weight: float = 1.0,
                 contrastive_weight: float = 1.0,
                 temperature: float = 0.5,
                 include_cross_entropy: bool = True) -> None:
        super().__init__()
        self.register_buffer("top_class_of_class", torch.as_tensor(top_class_of_class, dtype=torch.long))
        self.top_penalty_weight = top_penalty_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.include_cross_entropy = include_cross_entropy

    def forward(self,
                logits: Tensor,
                target: Tensor,
                *,
                embeddings: Optional[Tensor] = None) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape (batch, classes)")
        target = target.to(device=logits.device, dtype=torch.long)
        mapping = self.top_class_of_class.to(device=logits.device)
        if mapping.numel() != logits.shape[1]:
            raise ValueError("top_class_of_class must contain one entry per class")

        total = logits.new_tensor(0.0)
        if self.include_cross_entropy:
            total = total + F.cross_entropy(logits, target)
        if self.top_penalty_weight:
            total = total + self.top_penalty_weight * top_class_penalty(logits, target, mapping)
        if embeddings is not None and self.contrastive_weight:
            total = total + self.contrastive_weight * supervised_top_class_contrastive_loss(
                embeddings, mapping[target], temperature=self.temperature
            )
        
        return total
