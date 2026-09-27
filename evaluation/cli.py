"""Command-line orchestration for unified GP-AT evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import platform
import random
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

_CACHE_ROOT = Path(tempfile.gettempdir()) / "gpat_eval_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

try:
    import pytorch_lightning as pl
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pytorch_lightning is required for evaluation") from exc

from inference.profiling.profile_utils import DeviceManager, ModelRegistry, ModelSpec, default_model_specs

from .datasets import DATASET_NAMES, EvaluationDataModule, build_dataset_spec
from .lightning_module import GPATEvaluationModule
from .results import AtomicJSONWriter, EvaluationTimer, build_result


def _parse_limit_batches(value: Optional[str]) -> Optional[int | float]:
    if value is None:
        return None
    if "." in value:
        parsed = float(value)
        if not 0.0 < parsed <= 1.0:
            raise argparse.ArgumentTypeError("float --limit-batches must be in (0, 1]")
        return parsed
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("integer --limit-batches must be non-negative")
    
    return parsed


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local GP-AT models on supported metadata/audio datasets.")
    parser.add_argument("--all", action="store_true", help="Evaluate all discovered model/checkpoint combinations on all supported datasets.")
    parser.add_argument("--model", nargs="+", help="Model name(s) from the local registry.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path for a single selected model.")
    parser.add_argument("--dataset", nargs="+", choices=DATASET_NAMES, help="Dataset(s) to evaluate.")
    parser.add_argument("--models-dir", default="models", help="Local models directory.")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory for JSON result files.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--limit-batches", type=_parse_limit_batches, help="Lightning-compatible test batch limit: int count or float fraction.")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Requested accelerator.")
    parser.add_argument("--strict-device", action="store_true", help="Fail if the requested accelerator is unavailable.")
    parser.add_argument("--precision", default="32-true", help="Lightning precision setting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic orchestration.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for thresholded metrics.")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Pad/crop decoded audio to this duration.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing result JSON files.")
    parser.add_argument("--audio-root-audioset", help="Override AudioSet Standard audio root.")
    parser.add_argument("--audio-root-audioset-r", help="Override AudioSet-R audio root.")
    parser.add_argument("--audio-root-fsd50k", help="Override FSD50K eval audio root.")
    
    return parser.parse_args(argv)


def _auto_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_device(requested: str, *, strict: bool) -> tuple[str, str, Optional[str]]:
    normalized = _auto_accelerator() if requested == "auto" else requested
    selection = DeviceManager.select(normalized)
    if strict and selection.fallback_reason:
        raise RuntimeError(selection.fallback_reason)
    fallback = selection.fallback_reason
    if requested == "auto" and normalized == selection.effective:
        fallback = None
    
    return requested, selection.effective, fallback


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def _spec_with_checkpoint(spec: ModelSpec, checkpoint: Optional[str]) -> ModelSpec:
    if checkpoint is None:
        return spec
    path = Path(checkpoint).expanduser().resolve()
    
    return dataclasses.replace(spec, checkpoint_relative_path=str(path))


def _load_model(models_dir: str, spec: ModelSpec) -> tuple[torch.nn.Module, int, Path]:
    registry = ModelRegistry(models_dir, specs=[spec])
    checkpoint_path = registry.checkpoint_path(spec)
    
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    module = importlib.import_module(spec.module_path)
    model_class = getattr(module, spec.class_name)
    sample_rate = int(getattr(module, "SAMPLE_RATE"))
    model = model_class(**spec.init_kwargs)
    
    if hasattr(model, "load_pretrained"):
        model.load_pretrained(str(checkpoint_path))
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
    model.eval()
    
    return model, sample_rate, checkpoint_path


def _requested_specs(args: argparse.Namespace) -> List[ModelSpec]:
    registry = ModelRegistry(args.models_dir)
    if args.all:
        return registry.discover()
    requested = args.model or []
    if not requested:
        raise ValueError("select --all or provide --model")
    specs = registry.discover(requested)
    found = {spec.name for spec in specs}
    missing = sorted(set(requested) - found)
    if missing:
        known = ", ".join(spec.name for spec in default_model_specs())
        raise ValueError(f"unknown or unavailable model(s): {', '.join(missing)}. Known registry names: {known}")
    if args.checkpoint and len(specs) != 1:
        raise ValueError("--checkpoint can only be used with exactly one --model")
    
    return [_spec_with_checkpoint(spec, args.checkpoint) for spec in specs]


def _requested_datasets(args: argparse.Namespace) -> List[str]:
    if args.all:
        return list(DATASET_NAMES)
    return list(args.dataset or DATASET_NAMES)


def _audio_root_for(args: argparse.Namespace, dataset_name: str) -> Optional[str]:
    attr = {
        "audioset": "audio_root_audioset",
        "audioset_r": "audio_root_audioset_r",
        "fsd50k": "audio_root_fsd50k",
    }[dataset_name]
    return getattr(args, attr)


def _print_top10(dataset_name: str, top10) -> None:
    formatted = ", ".join(f"{item.rank}:{item.native_class_id}({item.positive_sample_count})" for item in top10)
    print(f"Top-10 {dataset_name}: {formatted}")


def evaluate_one(args: argparse.Namespace,
                 *,
                 spec: ModelSpec,
                 dataset_name: str,
                 writer: AtomicJSONWriter,
                 requested_accelerator: str,
                 effective_accelerator: str,
                 fallback_reason: Optional[str]) -> Path:
    print(f"Loading model={spec.name} checkpoint={spec.checkpoint_relative_path}")
    model, sample_rate, checkpoint_path = _load_model(args.models_dir, spec)
    print(f"Parsing dataset={dataset_name}")
    dataset_spec = build_dataset_spec(dataset_name, audio_root=_audio_root_for(args, dataset_name))
    print(f"Dataset {dataset_name}: metadata_samples={dataset_spec.total_metadata_samples} evaluable_audio={dataset_spec.evaluated_sample_count} classes={dataset_spec.class_count}")
    _print_top10(dataset_name, dataset_spec.top10)
    print(f"Device requested={requested_accelerator} effective={effective_accelerator} fallback={fallback_reason}")
    print(f"Active metrics: {', '.join(dataset_spec.compatibility.get('valid_metrics', [])) or 'none; not_applicable will be recorded'}")

    pin_memory = effective_accelerator == "cuda"
    
    datamodule = EvaluationDataModule(dataset_spec,
                                      sample_rate=sample_rate,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      duration_seconds=args.duration_seconds,
                                      pin_memory=pin_memory)
    
    lightning_module = GPATEvaluationModule(model,
                                            dataset_spec=dataset_spec,
                                            model_name=spec.name,
                                            model_class=spec.class_name,
                                            checkpoint_path=str(checkpoint_path),
                                            threshold=args.threshold)
    
    trainer_kwargs = {"accelerator": effective_accelerator,
                      "devices": 1,
                      "precision": args.precision,
                      "logger": False,
                      "enable_checkpointing": False}
    
    if args.limit_batches is not None:
        trainer_kwargs["limit_test_batches"] = args.limit_batches
    
    trainer = pl.Trainer(**trainer_kwargs)
    print(f"Starting evaluation: model={spec.name} dataset={dataset_name} batch_size={args.batch_size} limit_batches={args.limit_batches}")
    
    with EvaluationTimer() as timer:
        trainer.test(lightning_module, datamodule=datamodule, verbose=False)
    
    metrics = lightning_module.structured_metrics()
    warnings = list(lightning_module.metric_manager.warnings)
    failed_samples = list(lightning_module.metric_manager.failed_samples)
    warnings.extend(f"skipped {len(dataset_spec.skipped_samples)} samples before DataLoader because audio was missing" for _ in [0] if dataset_spec.skipped_samples)
    warnings.extend(f"skipped {len(failed_samples)} samples during DataLoader because audio could not be decoded" for _ in [0] if failed_samples)
    if not lightning_module.classifier_evaluable:
        warnings.append("loaded checkpoint has no pretrained classifier for quantitative tagging evaluation; metrics are recorded as not_applicable")
    elif len(lightning_module.evaluation_valid_class_indices) != dataset_spec.class_count:
        warnings.append(
            f"evaluation restricted to {len(lightning_module.evaluation_valid_class_indices)}/{dataset_spec.class_count} "
            "classes supported by the pretrained classifier and dataset projection"
        )
    
    result = build_result(model_name=spec.name,
                          model_class=spec.class_name,
                          checkpoint_path=checkpoint_path,
                          dataset_spec=dataset_spec,
                          requested_accelerator=requested_accelerator,
                          effective_accelerator=effective_accelerator,
                          fallback_reason=fallback_reason,
                          precision=args.precision,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          limit_batches=args.limit_batches,
                          seed=args.seed,
                          threshold=args.threshold,
                          sample_rate=sample_rate,
                          metrics=metrics,
                          evaluated_samples=lightning_module.evaluated_samples,
                          evaluated_batches=lightning_module.evaluated_batches,
                          duration_seconds=timer.duration_seconds,
                          warnings=warnings,
                          failed_samples=failed_samples,
                          classifier_evaluable=lightning_module.classifier_evaluable,
                          model_valid_output_indices=lightning_module.model_valid_output_indices,
                          evaluation_valid_class_indices=lightning_module.evaluation_valid_class_indices)
    
    output_path = writer.path_for(spec.name, dataset_name, str(checkpoint_path))
    writer.write(result, output_path)
    print(f"Completed model={spec.name} dataset={dataset_name}: saved {output_path}")
    
    return output_path


def _write_failure_result(writer: AtomicJSONWriter,
                          *,
                          spec: ModelSpec,
                          dataset_name: str,
                          exc: Exception,
                          requested_accelerator: str,
                          effective_accelerator: str,
                          fallback_reason: Optional[str]) -> Path:
    checkpoint = spec.checkpoint_relative_path
    
    result = {"status": "failed",
              "run_identity": {"model_name": spec.name,
                               "model_class": spec.class_name,
                               "checkpoint_path": checkpoint,
                               "checkpoint_identifier": Path(checkpoint).stem,
                               "dataset_name": dataset_name,
                               "evaluation_timestamp": datetime.now(timezone.utc).isoformat()},
              "environment": {"python_version": sys.version,
                              "pytorch_version": torch.__version__,
                              "pytorch_lightning_version": getattr(pl, "__version__", None),
                              "operating_system": platform.platform(),
                              "requested_accelerator": requested_accelerator,
                              "effective_accelerator": effective_accelerator,
                              "fallback_reason": fallback_reason},
              "failure": {"error_type": type(exc).__name__,
                          "error": str(exc),
                          "traceback": traceback.format_exc()}}
    
    output_path = writer.path_for(spec.name, dataset_name, checkpoint)
    writer.write(result, output_path)
    
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    
    try:
        _seed_everything(args.seed)
        requested_accelerator, effective_accelerator, fallback_reason = _select_device(args.accelerator, strict=args.strict_device)
        specs = _requested_specs(args)
        datasets = _requested_datasets(args)
    except Exception as exc:
        print(f"Configuration error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    
    writer = AtomicJSONWriter(args.output_dir, overwrite=args.overwrite)
    print(f"Evaluation mode={'full' if args.all else 'selective'} models={len(specs)} datasets={datasets}")
    
    failures = 0
    
    for spec in specs:
        for dataset_name in datasets:
            try:
                evaluate_one(args,
                             spec=spec,
                             dataset_name=dataset_name,
                             writer=writer,
                             requested_accelerator=requested_accelerator,
                             effective_accelerator=effective_accelerator,
                             fallback_reason=fallback_reason)
            except Exception as exc:
                failures += 1
                print(f"FAILED model={spec.name} dataset={dataset_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc()
                
                try:
                    failure_path = _write_failure_result(writer,
                                                         spec=spec,
                                                         dataset_name=dataset_name,
                                                         exc=exc,
                                                         requested_accelerator=requested_accelerator,
                                                         effective_accelerator=effective_accelerator,
                                                         fallback_reason=fallback_reason)
                    print(f"Recorded failure result: {failure_path}", file=sys.stderr)
                except Exception as write_exc:
                    print(f"Could not write failure result: {type(write_exc).__name__}: {write_exc}", file=sys.stderr)
                if not args.all:
                    return 2
    print(f"Evaluation attempts complete: combinations={len(specs) * len(datasets)} failures={failures}")
    
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
