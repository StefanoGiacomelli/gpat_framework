"""PyTorch Lightning evaluation module for GP-AT models."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None

from .datasets import DatasetSpec
from .metrics_manager import EvaluationMetricManager


class GPATEvaluationModule(pl.LightningModule if pl else nn.Module):
    """Inference-only Lightning module wrapping a local GP-AT model."""

    def __init__(self,
                 model: nn.Module,
                 *,
                 dataset_spec: DatasetSpec,
                 model_name: str,
                 model_class: str,
                 checkpoint_path: str,
                 threshold: float = 0.5):
        super().__init__()
        
        self.model = model
        self.model_name = model_name
        self.model_class = model_class
        self.checkpoint_path = checkpoint_path
        self.dataset_spec = dataset_spec
        self.classifier_evaluable = bool(
            getattr(model, "evaluation_classifier_pretrained", True)
        )
        raw_valid = getattr(model, "evaluation_valid_output_indices", None)
        self.model_valid_output_indices = (
            None if raw_valid is None else [int(index) for index in raw_valid]
        )
        self.evaluation_valid_class_indices = (
            dataset_spec.effective_evaluation_indices(self.model_valid_output_indices)
            if self.classifier_evaluable
            else []
        )
        self.metric_manager = EvaluationMetricManager(
            dataset_spec,
            threshold=threshold,
            valid_class_indices=self.evaluation_valid_class_indices,
            classifier_evaluable=self.classifier_evaluable,
        )
        self.threshold = threshold
        self._test_batch_count = 0

    def configure_optimizers(self):  # pragma: no cover - Lightning calls this defensively.
        return None

    def on_test_start(self) -> None:
        self.model.eval()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.model(waveform)
        scores = self._extract_scores(output)
        return self.dataset_spec.project_model_scores(
            scores,
            valid_model_output_indices=self.model_valid_output_indices,
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        failed_metadata = batch.get("failed_metadata", [])
        if failed_metadata:
            self.metric_manager.record_failed_samples(failed_metadata)
        if batch.get("skip_batch"):
            self._test_batch_count += 1
            return
        waveform = batch["waveform"]
        target = batch["target"]
        with torch.inference_mode():
            scores = self(waveform)
        self.metric_manager.update(scores, target, batch["metadata"])
        self._test_batch_count += 1

    @staticmethod
    def _extract_scores(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("clipwise_output", "probabilities", "probs", "scores", "logits"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    return value
        raise TypeError(f"could not extract a score tensor from model output type {type(output).__name__}")

    def structured_metrics(self) -> Dict[str, Any]:
        return self.metric_manager.finalize()

    @property
    def evaluated_samples(self) -> int:
        return self.metric_manager.evaluated_samples

    @property
    def evaluated_batches(self) -> int:
        return self._test_batch_count
