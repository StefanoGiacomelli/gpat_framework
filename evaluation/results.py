"""Structured JSON result construction and atomic writing."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None

from .datasets import DatasetSpec
from .metrics_manager import json_safe


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


class AtomicJSONWriter:
    """Write one final JSON result per model/dataset evaluation."""

    def __init__(self, output_dir: Path | str, *, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def path_for(self, model_name: str, dataset_name: str, checkpoint_path: Optional[str] = None) -> Path:
        base = f"{sanitize_name(model_name)}_{sanitize_name(dataset_name)}.json"
        path = self.output_dir / base
        if self.overwrite or not path.exists():
            return path
        stem = path.stem
        suffix = sanitize_name(Path(checkpoint_path or "checkpoint").stem)[:64]
        candidate = self.output_dir / f"{stem}_{suffix}.json"
        if self.overwrite or not candidate.exists():
            return candidate
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
        return self.output_dir / f"{stem}_{suffix}_{timestamp}.json"

    def write(self, result: Dict[str, Any], path: Path) -> Path:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
        
        return path


def build_result(*,
                 model_name: str,
                 model_class: str,
                 checkpoint_path: Path,
                 dataset_spec: DatasetSpec,
                 requested_accelerator: str,
                 effective_accelerator: str,
                 fallback_reason: Optional[str],
                 precision: str,
                 batch_size: int,
                 num_workers: int,
                 limit_batches: Optional[float | int],
                 seed: int,
                 threshold: float,
                 sample_rate: int,
                 metrics: Dict[str, Any],
                 evaluated_samples: int,
                 evaluated_batches: int,
                 duration_seconds: float,
                 warnings: list[str],
                 failed_samples: Optional[list[Dict[str, Any]]] = None,
                 classifier_evaluable: bool = True,
                 model_valid_output_indices: Optional[list[int]] = None,
                 evaluation_valid_class_indices: Optional[list[int]] = None,
                 status: str = "success") -> Dict[str, Any]:
    failed_samples = failed_samples or []
    
    return {"status": status,
            "run_identity": {"model_name": model_name,
                            "model_class": model_class,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_identifier": checkpoint_path.stem,
                            "dataset_name": dataset_spec.name,
                            "dataset_display_name": dataset_spec.display_name,
                            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()},
            
            "environment": {"python_version": sys.version,
                            "pytorch_version": torch.__version__,
                            "pytorch_lightning_version": getattr(pl, "__version__", None),
                            "operating_system": platform.platform(),
                            "requested_accelerator": requested_accelerator,
                            "effective_accelerator": effective_accelerator,
                            "precision": precision,
                            "fallback_reason": fallback_reason},
            
            "model_information": {"expected_sample_rate": sample_rate,
                                "output_class_count": 527,
                                "output_label_space": "AudioSet 527 MID order from datasets/AudioSet_meta/class_labels_indices.csv",
                                "output_semantics": "sigmoid probabilities according to local model wrappers",
                                "pretrained_classifier_evaluable": classifier_evaluable,
                                "pretrained_output_indices": model_valid_output_indices},
            
            "dataset_information": {"metadata_path": str(dataset_spec.metadata_path),
                                    "audio_root": str(dataset_spec.default_audio_root) if dataset_spec.default_audio_root else None,
                                    "split": dataset_spec.split,
                                    "total_metadata_samples": dataset_spec.total_metadata_samples,
                                    "successfully_evaluated_samples": evaluated_samples,
                                    "skipped_samples": len(dataset_spec.skipped_samples),
                                    "failed_samples": len(failed_samples),
                                    "native_class_count": dataset_spec.native_class_count,
                                    "evaluation_target_class_count": dataset_spec.class_count,
                                    "effective_evaluation_class_count": len(evaluation_valid_class_indices or []),
                                    "effective_evaluation_class_indices": evaluation_valid_class_indices,
                                    "native_label_space": dataset_spec.native_label_space,
                                    "class_frequency_distribution": dataset_spec.class_frequencies},
            
            "evaluation_configuration": {"batch_size": batch_size,
                                        "number_of_workers": num_workers,
                                        "limit_batches": limit_batches,
                                        "seed": seed,
                                        "threshold_values": {"default_binary_threshold": threshold},
                                        "top10_f1_selection_policy": "native evaluation-split positive-count ranking from complete metadata",
                                        "preprocessing": "mono downmix, resample to model sample rate, pad/crop to configured duration",
                                        "post_processing": (
                                            "dataset-specific score projection plus model-specific pretrained-output masking when required"
                                            if dataset_spec.prediction_projection is not None or model_valid_output_indices is not None
                                            else "none"
                                        ),
                                        "mapping_strategy": dataset_spec.compatibility.get("mapping_required")},
            
            "mapping_and_taxonomy_information": {"source_label_space": dataset_spec.native_label_space,
                                                "evaluation_label_space": dataset_spec.evaluation_label_space,
                                                "mapping_mechanism": dataset_spec.mapping_coverage.get("mapping_mechanism"),
                                                "hierarchy_source": dataset_spec.compatibility.get("ontology_hierarchy_source"),
                                                "ontology_source": dataset_spec.compatibility.get("ontology_hierarchy_source"),
                                                "mapping_coverage_statistics": dataset_spec.mapping_coverage},
            
            "compatibility": dataset_spec.compatibility,
            "metrics": metrics,
            
            "runtime_summary": {"total_evaluation_duration_seconds": duration_seconds,
                                "evaluated_batch_count": evaluated_batches,
                                "samples_per_second": evaluated_samples / duration_seconds if duration_seconds > 0 else None},
        
            "warnings_and_failures": {"warnings": warnings,
                                    "skipped_samples_preview": dataset_spec.skipped_samples[:50],
                                    "failed_samples_preview": failed_samples[:50]}}


class EvaluationTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.duration_seconds = time.perf_counter() - self.start
        return False
