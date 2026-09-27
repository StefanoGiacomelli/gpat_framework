"""Metric finalization for unified GP-AT evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score

from training import metrics as gpat_metrics

from .datasets import DatasetSpec


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    
    return value


def metric_value(value: Any, *, label_space: str, **configuration: Any) -> Dict[str, Any]:
    return {"status": "computed",
            "value": json_safe(value),
            "label_space": label_space,
            "configuration": json_safe(configuration)}


def not_applicable(reason: str, *, label_space: Optional[str] = None, **configuration: Any) -> Dict[str, Any]:
    result = {"status": "not_applicable",
              "reason": reason,
              "configuration": json_safe(configuration)}
    if label_space is not None:
        result["label_space"] = label_space
    
    return result


class EvaluationMetricManager:
    """Accumulate prediction/target tensors and finalize dataset metrics."""

    def __init__(
        self,
        dataset_spec: DatasetSpec,
        *,
        threshold: float = 0.5,
        valid_class_indices: Optional[List[int]] = None,
        classifier_evaluable: bool = True,
    ):
        self.dataset_spec = dataset_spec
        self.threshold = threshold
        self.classifier_evaluable = bool(classifier_evaluable)
        if valid_class_indices is None:
            valid_class_indices = list(range(dataset_spec.class_count))
        self.valid_class_indices = [int(index) for index in valid_class_indices]
        self._valid_class_set = set(self.valid_class_indices)
        self._scores: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._metadata: List[Dict[str, Any]] = []
        self.failed_samples: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor, metadata: List[Dict[str, Any]]) -> None:
        self._metadata.extend(metadata)
        if not self.classifier_evaluable:
            return
        if scores.ndim != 2 or targets.ndim != 2:
            self.warnings.append(
                f"skipped metric update for non-2D tensors: scores={tuple(scores.shape)}, targets={tuple(targets.shape)}"
            )
            return
        if scores.shape != targets.shape:
            self.warnings.append(
                f"skipped metric update for mismatched tensors: scores={tuple(scores.shape)}, targets={tuple(targets.shape)}"
            )
            return
        if scores.shape[1] != self.dataset_spec.class_count:
            self.warnings.append(
                "skipped metric update because score/target class count does not match "
                f"dataset evaluation space: got {scores.shape[1]}, expected {self.dataset_spec.class_count}"
            )
            return
        self._scores.append(scores.detach().cpu().float())
        self._targets.append(targets.detach().cpu().float())

    def record_failed_samples(self, failed_metadata: List[Dict[str, Any]]) -> None:
        self.failed_samples.extend(failed_metadata)

    @property
    def evaluated_samples(self) -> int:
        if self._scores:
            return int(sum(batch.shape[0] for batch in self._scores))
        return len(self._metadata)

    @property
    def evaluated_batches(self) -> int:
        return len(self._scores)

    @property
    def hierarchy_metrics_applicable(self) -> bool:
        return (
            self.dataset_spec.supports_audioset_hierarchy
            and self.dataset_spec.class_count == 527
            and self.valid_class_indices == list(range(527))
        )

    def _metric_label_space(self) -> str:
        label_space = self.dataset_spec.evaluation_label_space
        if len(self.valid_class_indices) != self.dataset_spec.class_count:
            return (
                f"{label_space}; restricted to {len(self.valid_class_indices)}/"
                f"{self.dataset_spec.class_count} classes supported by the pretrained classifier"
            )
        return label_space

    def finalize(self) -> Dict[str, Any]:
        base_label_space = self.dataset_spec.evaluation_label_space
        metric_label_space = self._metric_label_space()

        if not self.classifier_evaluable:
            reason = "the loaded checkpoint does not provide a pretrained classifier for this evaluation label space"
            return {
                "standard": self._all_not_applicable(reason, metric_label_space),
                "ontology_aware": {"omap": not_applicable(reason, label_space=base_label_space)},
                "hierarchy_aware": {
                    "hierarchical_multilabel_prf": not_applicable(reason, label_space=base_label_space),
                    "hlp_expanded_map": not_applicable(reason, label_space=base_label_space),
                },
                "top10_class_f1": self._top10_results(None, None),
            }

        if not self.valid_class_indices:
            reason = "no evaluation classes remain after applying model-specific pretrained-output constraints"
            return {
                "standard": self._all_not_applicable(reason, metric_label_space),
                "ontology_aware": {"omap": not_applicable(reason, label_space=base_label_space)},
                "hierarchy_aware": {
                    "hierarchical_multilabel_prf": not_applicable(reason, label_space=base_label_space),
                    "hlp_expanded_map": not_applicable(reason, label_space=base_label_space),
                },
                "top10_class_f1": self._top10_results(None, None),
            }

        if not self._scores:
            reason = "no valid prediction batches were accumulated"
            return {
                "standard": self._all_not_applicable(reason, metric_label_space),
                "ontology_aware": {"omap": not_applicable(reason, label_space=base_label_space)},
                "hierarchy_aware": {
                    "hierarchical_multilabel_prf": not_applicable(reason, label_space=base_label_space),
                    "hlp_expanded_map": not_applicable(reason, label_space=base_label_space),
                },
                "top10_class_f1": self._top10_results(None, None),
            }

        full_scores = torch.cat(self._scores, dim=0).numpy()
        full_targets = torch.cat(self._targets, dim=0).numpy()
        valid = np.asarray(self.valid_class_indices, dtype=np.int64)
        scores = full_scores[:, valid]
        targets = full_targets[:, valid]

        common_configuration = {
            "evaluated_class_count": len(self.valid_class_indices),
            "evaluation_target_class_count": self.dataset_spec.class_count,
            "evaluated_class_indices": self.valid_class_indices,
            "excluded_class_indices": [
                index for index in range(self.dataset_spec.class_count)
                if index not in self._valid_class_set
            ],
        }

        standard = {
            "class_wise_ap": metric_value(
                gpat_metrics.average_precision_per_class(scores, targets),
                label_space=metric_label_space,
                no_positive="nan",
                **common_configuration,
            ),
            "map": metric_value(
                gpat_metrics.mean_average_precision(scores, targets),
                label_space=metric_label_space,
                no_positive="nan",
                ignore_nan=True,
                **common_configuration,
            ),
            "lwlrap": metric_value(
                gpat_metrics.lwlrap(scores, targets).overall,
                label_space=metric_label_space,
                **common_configuration,
            ),
            "micro_averaged_accuracy": metric_value(
                gpat_metrics.micro_averaged_accuracy(scores, targets, threshold=self.threshold),
                label_space=metric_label_space,
                threshold=self.threshold,
                **common_configuration,
            ),
            "micro_f1": metric_value(
                gpat_metrics.multilabel_f1(scores, targets, average="micro", threshold=self.threshold),
                label_space=metric_label_space,
                average="micro",
                threshold=self.threshold,
                **common_configuration,
            ),
            "macro_f1": metric_value(
                gpat_metrics.multilabel_f1(scores, targets, average="macro", threshold=self.threshold),
                label_space=metric_label_space,
                average="macro",
                threshold=self.threshold,
                **common_configuration,
            ),
        }

        if self.hierarchy_metrics_applicable:
            ontology: Dict[str, Any] = {}
            try:
                omap = gpat_metrics.ontology_mean_average_precision(full_scores, full_targets)
                ontology["omap"] = metric_value(
                    {"average": omap.average, "by_coarse_level": omap.by_coarse_level},
                    label_space=base_label_space,
                    graph_distance="training/audioset_graph_distance.py",
                )
            except Exception as exc:
                ontology["omap"] = not_applicable(
                    f"{type(exc).__name__}: {exc}", label_space=base_label_space
                )

            hierarchy: Dict[str, Any] = {}
            try:
                prf = gpat_metrics.hierarchical_multilabel_precision_recall_fscore(
                    full_scores, full_targets, threshold=self.threshold
                )
                hierarchy["hierarchical_multilabel_prf"] = metric_value(
                    {"precision": prf.precision, "recall": prf.recall, "f1": prf.f1},
                    label_space=f"{base_label_space} with AudioSet HLP closure",
                    threshold=self.threshold,
                    lookup_table="training/HLP_lookup_table.pt",
                    averaging="sample_macro",
                )
            except Exception as exc:
                hierarchy["hierarchical_multilabel_prf"] = not_applicable(
                    f"{type(exc).__name__}: {exc}", label_space=base_label_space
                )

            try:
                hierarchy["hlp_expanded_map"] = metric_value(
                    gpat_metrics.hlp_expanded_mean_average_precision(full_scores, full_targets),
                    label_space=f"{base_label_space} with soft AudioSet HLP prediction propagation and target closure",
                    lookup_table="training/HLP_lookup_table.pt",
                    no_positive="nan",
                    ignore_nan=True,
                )
            except Exception as exc:
                hierarchy["hlp_expanded_map"] = not_applicable(
                    f"{type(exc).__name__}: {exc}", label_space=base_label_space
                )
        else:
            reason = (
                "AudioSet ontology/hierarchy metrics require the complete canonical 527-class "
                "AudioSet evaluation space without model-specific class masking"
            )
            ontology = {"omap": not_applicable(reason, label_space=base_label_space)}
            hierarchy = {
                "hierarchical_multilabel_prf": not_applicable(reason, label_space=base_label_space),
                "hlp_expanded_map": not_applicable(reason, label_space=base_label_space),
            }

        return {
            "standard": standard,
            "ontology_aware": ontology,
            "hierarchy_aware": hierarchy,
            "top10_class_f1": self._top10_results(full_scores, full_targets),
        }

    def _all_not_applicable(self, reason: str, label_space: str) -> Dict[str, Any]:
        return {
            name: not_applicable(reason, label_space=label_space)
            for name in [
                "class_wise_ap",
                "map",
                "lwlrap",
                "micro_averaged_accuracy",
                "micro_f1",
                "macro_f1",
            ]
        }

    def _top10_results(
        self,
        scores: Optional[np.ndarray],
        targets: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        rows = []
        for item in self.dataset_spec.top10:
            row = asdict(item)
            row["threshold"] = self.threshold
            row["label_space"] = self.dataset_spec.evaluation_label_space
            evaluation_index = item.evaluation_index

            if not self.classifier_evaluable:
                row["status"] = "not_applicable"
                row["reason"] = "the loaded checkpoint does not provide a pretrained classifier"
                row["f1"] = None
            elif evaluation_index is None:
                row["status"] = "not_applicable"
                row["reason"] = "class has no evaluation-space index"
                row["f1"] = None
            elif evaluation_index not in self._valid_class_set:
                row["status"] = "not_applicable"
                row["reason"] = "class is not supported by the pretrained classifier output set"
                row["f1"] = None
            elif scores is None or targets is None:
                row["status"] = "not_applicable"
                row["reason"] = "no valid predictions were available"
                row["f1"] = None
            else:
                true = targets[:, evaluation_index] > 0.5
                pred_score = scores[:, evaluation_index]
                row["status"] = "evaluable"
                row["reason"] = None
                row["f1"] = float(
                    f1_score(true, pred_score >= self.threshold, zero_division=0)
                )
            rows.append(row)

        return {
            "selection_method": "top 10 native classes by positive sample count in full evaluation metadata",
            "tie_breaking_rule": "stable dataset class order, then native class ID",
            "frequencies_from_full_metadata": True,
            "limited_batches_can_affect_metric_values_but_not_selection": True,
            "classes": json_safe(rows),
        }
