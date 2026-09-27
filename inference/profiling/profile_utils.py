"""Unified profiling utilities for GP-AT PyTorch models.

The public entry point is :class:`ModelProfiler`. It loads one model from the
local ``models/`` tree, generates deterministic dummy waveform input, runs
warm-up and measured inference, and returns a JSON-compatible dictionary.

Timing boundary: measured time covers only ``model(input_tensor)`` execution.
Model construction, checkpoint loading, dummy input generation, warm-up,
complexity analysis, minimum-input search, and JSON serialization are outside
the measured timing vector.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats


BYTES_PER_DECIMAL_MB = 1_000_000
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASURED_RUNS = 20
DEFAULT_RANDOM_SEED = 42
SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}


@dataclass(frozen=True)
class ModelSpec:
    """Local model-loading metadata verified against the vendored wrappers."""

    name: str
    module_path: str
    class_name: str
    checkpoint_relative_path: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_rank: int = 2
    input_description: str = "mono waveform tensor shaped (batch, samples)"


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    effective: str
    fallback_reason: Optional[str]

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.effective)


def _json_safe(value: Any) -> Any:
    """Recursively convert common scientific Python values to JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    
    return value


def seconds_to_samples(duration_seconds: float, sample_rate: int) -> int:
    """Convert duration to samples using nearest-integer rounding.

    Rounding rule: ``round(duration_seconds * sample_rate)`` followed by a
    lower bound of one sample. Python's tie-to-even behavior is therefore the
    exact rule for half-sample ties.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    
    return max(1, int(round(duration_seconds * sample_rate)))


def default_model_specs() -> List[ModelSpec]:
    """Return the local AudioSet model catalog used by the profiler."""
    return [
        ModelSpec("ast", "models.ast.model", "ASTModel", "ast/audioset_10_10_0.4593.pth"),
        ModelSpec("audioclip", "models.audioclip.model", "AudioCLIP", "audioclip/AudioCLIP-Full-Training.pt"),
        ModelSpec("audiomae", "models.audiomae.model", "AudioMAE", "audiomae/finetuned.pth"),
        ModelSpec("beats", "models.beats.model", "BEATs", "beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"),
        ModelSpec("ced", "models.ced.model", "CEDBase", "ced/audiotransformer_base_mAP_4999.pt"),
        ModelSpec("clap", "models.clap.model", "CLAP", "clap/630k-audioset-fusion-best.pt"),
        ModelSpec("convnext", "models.convnext.model", "ConvNeXt", "convnext/convnext_tiny_471mAP.pth"),
        ModelSpec("efficientat_dymn", "models.efficientat.model_dymn", "EfficientAT_DyMN", "efficientat/dymn20_as_mAP_493.pt"),
        ModelSpec("efficientat_mn", "models.efficientat.model_mn", "EfficientAT_MN", "efficientat/mn40_as_ext_mAP_487.pt"),
        ModelSpec("epanns", "models.epanns.model", "EPANNs", "epanns/checkpoint_closeto_.44.pt"),
        ModelSpec("htsat", "models.htsat.model", "HTSAT", "htsat/HTSAT_AudioSet_Saved_3.ckpt"),
        ModelSpec("m2d", "models.m2d.model", "M2D", "m2d/weights_ep69it3124-0.47998.pth"),
        ModelSpec("panns_resnet38", "models.panns.model_resnet38", "ResNet38", "panns/ResNet38_mAP=0.434.pth"),
        ModelSpec(
            "panns_wavegram_logmel_cnn14",
            "models.panns.model_wavegram_logmel_cnn14",
            "Wavegram_Logmel_Cnn14",
            "panns/Wavegram_Logmel_Cnn14_mAP=0.439.pth",
        ),
        ModelSpec("passt", "models.passt.model", "PaSST", "passt/passt-s-kd-ap.486.pt"),
        ModelSpec(
            "psla",
            "models.psla.model",
            "EffNetAttention",
            "psla/as_mdl_0_wa.pth",
            init_kwargs={"label_dim": 527, "b": 2, "pretrain": False, "head_num": 4},
        ),
        ModelSpec("vggish", "models.vggish.model", "VGGish", "vggish/vggish_with_classifier.pth"),
        ModelSpec("yamnet", "models.yamnet.model", "YAMNet", "yamnet/yamnet.pth"),
    ]


class DeviceManager:
    """Validate requested devices and report explicit fallback details."""

    @staticmethod
    def select(requested_device: str) -> DeviceSelection:
        requested = requested_device.lower()
        if requested not in SUPPORTED_DEVICES:
            return DeviceSelection(requested=requested_device,
                                   effective="cpu",
                                   fallback_reason=f"unsupported requested device '{requested_device}'")
        if requested == "cpu":
            return DeviceSelection(requested=requested, effective="cpu", fallback_reason=None)
        if requested == "cuda" and torch.cuda.is_available():
            return DeviceSelection(requested=requested, effective="cuda", fallback_reason=None)
        if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceSelection(requested=requested, effective="mps", fallback_reason=None)
        if requested == "mps":
            return DeviceSelection(requested=requested, 
                                   effective="cpu", 
                                   fallback_reason=DeviceManager._mps_unavailable_reason())
        return DeviceSelection(requested=requested,
                               effective="cpu",
                               fallback_reason=f"requested device '{requested}' is not available")

    @staticmethod
    def _mps_unavailable_reason() -> str:
        if not hasattr(torch.backends, "mps"):
            return "requested device 'mps' is not supported by this PyTorch build"
        is_built = torch.backends.mps.is_built()
        is_available = torch.backends.mps.is_available()
        allocation_error = None
        try:
            _ = torch.ones(1, device="mps")
        except Exception as exc:
            allocation_error = f"{type(exc).__name__}: {exc}"
        details = f"torch.backends.mps.is_built()={is_built}, torch.backends.mps.is_available()={is_available}"
        if allocation_error:
            details = f"{details}, allocation_test_error={allocation_error}"
        return f"requested device 'mps' is not available ({details})"

    @staticmethod
    def synchronize(device: str) -> Optional[str]:
        """Synchronize asynchronous device work when PyTorch exposes a hook."""
        if device == "cuda":
            torch.cuda.synchronize()
            return "torch.cuda.synchronize"
        
        if device == "mps":
            sync = getattr(torch.mps, "synchronize", None)
            if callable(sync):
                sync()
                return "torch.mps.synchronize"            
            return None
        
        return None


class ModelRegistry:
    """Discover and load local GP-AT model wrappers."""

    def __init__(self, models_dir: Path | str = "models", specs: Optional[Iterable[ModelSpec]] = None):
        self.models_dir = Path(models_dir)
        self.specs = list(specs) if specs is not None else default_model_specs()

    def discover(self, model_names: Optional[Iterable[str]] = None) -> List[ModelSpec]:
        requested = set(model_names or [])
        discovered = []
        for spec in self.specs:
            if requested and spec.name not in requested:
                continue
            checkpoint_path = self.models_dir / spec.checkpoint_relative_path
            module_path = Path(*spec.module_path.split(".")).with_suffix(".py")
            if checkpoint_path.exists() and module_path.exists():
                discovered.append(spec)
        return discovered

    def checkpoint_path(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.checkpoint_relative_path

    def load(self, spec: ModelSpec) -> Tuple[nn.Module, Any, int]:
        module = importlib.import_module(spec.module_path)
        model_class = getattr(module, spec.class_name)
        sample_rate = int(getattr(module, "SAMPLE_RATE"))
        model = model_class(**spec.init_kwargs)
        checkpoint_path = self.checkpoint_path(spec)
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(str(checkpoint_path))
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state, strict=False)
        model.eval()
        
        return model, module, sample_rate


class StatisticsCalculator:
    """Statistical summaries for measured inference times or memory values."""

    @staticmethod
    def timing(values: List[float]) -> Dict[str, Any]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {}
        q25, q75 = np.percentile(arr, [25, 75])
        return _json_safe({"unit": "seconds",
                           "count": int(arr.size),
                           "min": float(np.min(arr)),
                           "max": float(np.max(arr)),
                           "mean": float(np.mean(arr)),
                           "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                           "median": float(np.median(arr)),
                           "skewness": float(scipy_stats.skew(arr, bias=False)) if arr.size > 2 else 0.0,
                           "kurtosis": float(scipy_stats.kurtosis(arr, fisher=True, bias=False)) if arr.size > 3 else 0.0,
                           "iqr": float(q75 - q25),
                           "p75": float(q75),
                           "p90": float(np.percentile(arr, 90))})

    @staticmethod
    def numeric(values: List[float], unit: str) -> Dict[str, Any]:
        if not values:
            return {"supported": False,
                    "unit": unit,
                    "values": [],
                    "mean": None,
                    "std": None}
        
        return _json_safe({"supported": True,
                           "unit": unit,
                           "values": values,
                           "mean": statistics.fmean(values),
                           "std": statistics.stdev(values) if len(values) > 1 else 0.0})


class ParameterProfiler:
    """Parameter counts and static parameter-memory estimates."""

    @staticmethod
    def profile(model: nn.Module) -> Dict[str, Any]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bytes_by_dtype: Dict[str, int] = {}
        for parameter in model.parameters():
            dtype = str(parameter.dtype)
            bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + parameter.numel() * parameter.element_size()
        total_bytes = sum(bytes_by_dtype.values())
        
        return _json_safe({"total_parameters": total,
                           "trainable_parameters": trainable,
                           "non_trainable_parameters": total - trainable,
                           "total_parameters_millions": total / 1_000_000,
                           "trainable_parameters_millions": trainable / 1_000_000,
                           "parameter_memory_mb": total_bytes / BYTES_PER_DECIMAL_MB,
                           "bytes_by_dtype": bytes_by_dtype})


class MemorySampler:
    """Device-specific runtime-memory snapshots around one inference call."""

    def __init__(self, device: str):
        self.device = device
        self.psutil_process = None
        if device == "cpu":
            try:
                import psutil
                self.psutil_process = psutil.Process()
            except Exception:
                self.psutil_process = None

    def model_loaded_state(self) -> Dict[str, Any]:
        if self.device == "cuda":
            return self._cuda_state("model_loaded")
        if self.device == "mps":
            return self._mps_state("model_loaded")
        
        return self._cpu_state("model_loaded")

    def before_run(self) -> Dict[str, Any]:
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            return self._cuda_state("before")
        if self.device == "mps":
            return self._mps_state("before")
        
        return self._cpu_state("before")

    def after_run(self) -> Dict[str, Any]:
        if self.device == "cuda":
            return self._cuda_state("after")
        if self.device == "mps":
            return self._mps_state("after")
        
        return self._cpu_state("after")

    def _cpu_state(self, phase: str) -> Dict[str, Any]:
        rss = self.psutil_process.memory_info().rss if self.psutil_process is not None else None
        
        return {"phase": phase, "process_rss_mb": None if rss is None else rss / BYTES_PER_DECIMAL_MB}

    def _cuda_state(self, phase: str) -> Dict[str, Any]:
        return {"phase": phase,
                "allocated_mb": torch.cuda.memory_allocated() / BYTES_PER_DECIMAL_MB,
                "reserved_mb": torch.cuda.memory_reserved() / BYTES_PER_DECIMAL_MB,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / BYTES_PER_DECIMAL_MB,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / BYTES_PER_DECIMAL_MB}

    def _mps_state(self, phase: str) -> Dict[str, Any]:
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        
        return {"phase": phase,
                "current_allocated_mb": current() / BYTES_PER_DECIMAL_MB if callable(current) else None,
                "driver_allocated_mb": driver() / BYTES_PER_DECIMAL_MB if callable(driver) else None,
                "recommended_max_memory_mb": recommended() / BYTES_PER_DECIMAL_MB if callable(recommended) else None}

    @staticmethod
    def summarize(device: str, before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Dict[str, Any]:
        if device == "cpu":
            values = [max(0.0, a["process_rss_mb"] - b["process_rss_mb"])
                      for b, a in zip(before, after)
                      if a.get("process_rss_mb") is not None and b.get("process_rss_mb") is not None]
            
            return {"increment": StatisticsCalculator.numeric(values, "decimal MB")}
        
        if device == "cuda":
            values = [max(0.0, a["peak_allocated_mb"] - b["allocated_mb"])
                      for b, a in zip(before, after)
                      if a.get("peak_allocated_mb") is not None and b.get("allocated_mb") is not None]
            
            return {"increment": StatisticsCalculator.numeric(values, "decimal MB")}
        
        if device == "mps":
            values = [max(0.0, a["current_allocated_mb"] - b["current_allocated_mb"])
                      for b, a in zip(before, after)
                      if a.get("current_allocated_mb") is not None and b.get("current_allocated_mb") is not None]
            
            return {"increment": StatisticsCalculator.numeric(values, "decimal MB")}
        
        return {"increment": StatisticsCalculator.numeric([], "decimal MB")}


class OutputValidator:
    """Minimal output predicate for model-agnostic forward success."""

    @staticmethod
    def is_valid(output: Any, batch_size: int = 1) -> bool:
        if isinstance(output, torch.Tensor):
            return output.numel() > 0 and (output.dim() == 0 or output.shape[0] == batch_size)
        if isinstance(output, (list, tuple)):
            return bool(output) and any(OutputValidator.is_valid(item, batch_size) for item in output)
        if isinstance(output, dict):
            return bool(output) and any(OutputValidator.is_valid(item, batch_size) for item in output.values())
        
        return output is not None


class ErrorClassifier:
    """Classify runtime failures for minimum-input search and reporting."""

    SHAPE_PATTERNS = (
        "input is too small",
        "calculated padded input size",
        "kernel size",
        "output size is too small",
        "negative dimension",
        "shape",
        "size mismatch",
        "mat1 and mat2",
        "padding size",
        "expected input",
        "invalid for input",
        "reshape",
        "view size is not compatible",
        "index out of range",
    )

    @staticmethod
    def classify(exc: BaseException) -> str:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return "out_of_memory"
        text = str(exc).lower()
        if "out of memory" in text:
            return "out_of_memory"
        if any(pattern in text for pattern in ErrorClassifier.SHAPE_PATTERNS):
            return "insufficient_input_or_shape"
        if "not implemented" in text or "not currently supported" in text:
            return "unsupported_operator"
        
        return "unrelated_runtime_error"


class InferenceRunner:
    """Run warm-up and measured inference."""

    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device

    def warmup(self, input_tensor: torch.Tensor, count: int) -> None:
        self.model.eval()
        with torch.inference_mode():
            for _ in range(count):
                _ = self.model(input_tensor)
                DeviceManager.synchronize(self.device)

    def measured(self, input_tensor: torch.Tensor, count: int) -> Tuple[List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
        times: List[float] = []
        before_memory: List[Dict[str, Any]] = []
        after_memory: List[Dict[str, Any]] = []
        sampler = MemorySampler(self.device)
        self.model.eval()
        
        with torch.inference_mode():
            for _ in range(count):
                before_memory.append(sampler.before_run())
                DeviceManager.synchronize(self.device)
                start = time.perf_counter()
                output = self.model(input_tensor)
                DeviceManager.synchronize(self.device)
                elapsed = time.perf_counter() - start
                if not OutputValidator.is_valid(output, batch_size=input_tensor.shape[0]):
                    raise RuntimeError("model inference completed but output failed the generic validity predicate")
                after_memory.append(sampler.after_run())
                times.append(elapsed)
        
        return times, before_memory, after_memory


class ComplexityProfiler:
    """Best-effort structural summaries and complexity estimates."""

    @staticmethod
    def profile(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        result = {"input_shape": list(input_tensor.shape),
                  "fvcore": ComplexityProfiler._fvcore(model, input_tensor),
                  "torchinfo": ComplexityProfiler._torchinfo(model, input_tensor)}
        fvcore = result["fvcore"]
        flops = fvcore.get("flops")
        result["flops"] = flops
        result["macs_estimated"] = flops / 2 if isinstance(flops, (int, float)) else None
        result["gflops"] = flops / 1e9 if isinstance(flops, (int, float)) else None
        result["gmacs_estimated"] = result["macs_estimated"] / 1e9 if result["macs_estimated"] is not None else None
        
        return _json_safe(result)

    @staticmethod
    def _fvcore(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        try:
            from fvcore.nn import FlopCountAnalysis

            analysis = FlopCountAnalysis(model, input_tensor)
            flops = int(analysis.total())
            unsupported = {str(k): int(v) for k, v in analysis.unsupported_ops().items()}
            uncalled = sorted(str(name) for name in analysis.uncalled_modules())
            
            return {"success": True,
                    "tool": "fvcore.nn.FlopCountAnalysis",
                    "flops": flops,
                    "unsupported_ops": unsupported,
                    "uncalled_modules": uncalled,
                    "warnings": ComplexityProfiler._coverage_warnings(unsupported, uncalled)}
        
        except Exception as exc:
            return {"success": False,
                    "tool": "fvcore.nn.FlopCountAnalysis",
                    "flops": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "warnings": ["fvcore complexity analysis failed; FLOPs/MACs are unavailable for this input."]}

    @staticmethod
    def _torchinfo(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        try:
            from torchinfo import summary
        
        except Exception as exc:
            return {"success": False,
                    "tool": "torchinfo.summary",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "warnings": ["torchinfo is not installed or could not be imported."]}
        try:
            info = summary(model, input_data=input_tensor, verbose=0)
            return {"success": True,
                    "tool": "torchinfo.summary",
                    "total_params": getattr(info, "total_params", None),
                    "trainable_params": getattr(info, "trainable_params", None),
                    "total_mult_adds": getattr(info, "total_mult_adds", None),
                    "summary": str(info)}
        
        except Exception as exc:
            return {"success": False,
                    "tool": "torchinfo.summary",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "warnings": ["torchinfo failed for this model/input; inference profiling may still be valid."]}

    @staticmethod
    def _coverage_warnings(unsupported: Dict[str, int], uncalled: List[str]) -> List[str]:
        warnings = []
        if unsupported:
            warnings.append("fvcore reported unsupported operators; FLOPs are a lower-bound/partial estimate.")
        if uncalled:
            warnings.append("fvcore reported uncalled modules for this input path.")
        
        return warnings


class MinimumInputFinder:
    """Find the shortest input that satisfies the generic forward predicate."""

    def __init__(self, model: nn.Module, sample_rate: int, device: str, make_input: Callable[[int], torch.Tensor]):
        self.model = model
        self.sample_rate = sample_rate
        self.device = device
        self.make_input = make_input

    def find(self,
             lower_seconds: float = 0.001,
             upper_seconds: float = DEFAULT_DURATION_SECONDS,
             max_upper_seconds: float = 10.0) -> Dict[str, Any]:
        lower_samples = seconds_to_samples(lower_seconds, self.sample_rate)
        upper_samples = seconds_to_samples(upper_seconds, self.sample_rate)
        max_upper_samples = seconds_to_samples(max_upper_seconds, self.sample_rate)
        attempts: List[Dict[str, Any]] = []

        while True:
            ok, cls, err = self._test(upper_samples)
            attempts.append({"samples": upper_samples, "ok": ok, "classification": cls, "error": err})
            if ok:
                break
            if cls not in {"insufficient_input_or_shape"}:
                return self._failed("upper_bound_failed_with_non_size_error", attempts)
            upper_samples *= 2
            if upper_samples > max_upper_samples:
                return self._failed("no_working_upper_bound", attempts)

        lo = max(1, lower_samples)
        hi = upper_samples
        while lo < hi:
            mid = (lo + hi) // 2
            ok, cls, err = self._test(mid)
            attempts.append({"samples": mid, "ok": ok, "classification": cls, "error": err})
            if ok:
                hi = mid
            elif cls == "insufficient_input_or_shape":
                lo = mid + 1
            else:
                return self._failed("search_aborted_with_non_size_error", attempts)

        return _json_safe({"status": "success",
                           "samples": hi,
                           "seconds": hi / self.sample_rate,
                           "sample_rate": self.sample_rate,
                           "search_configuration": {"initial_lower_bound_seconds": lower_seconds,
                                                    "initial_upper_bound_seconds": upper_seconds,
                                                    "max_upper_bound_seconds": max_upper_seconds},
                           "attempt_count": len(attempts),
                           "attempts_tail": attempts[-12:]})

    def _test(self, samples: int) -> Tuple[bool, str, Optional[str]]:
        try:
            input_tensor = self.make_input(samples)
            with torch.inference_mode():
                output = self.model(input_tensor)
                DeviceManager.synchronize(self.device)
            if OutputValidator.is_valid(output, batch_size=input_tensor.shape[0]):
                return True, "success", None
            return False, "invalid_output", "output failed generic validity predicate"
        
        except Exception as exc:
            return False, ErrorClassifier.classify(exc), str(exc)

    def _failed(self, status: str, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _json_safe({"status": status,
                           "samples": None,
                           "seconds": None,
                           "sample_rate": self.sample_rate,
                           "attempt_count": len(attempts),
                           "attempts_tail": attempts[-12:]})


class ModelProfiler:
    """Profile one local GP-AT model and return a JSON-compatible result."""

    def __init__(self,
                 models_dir: Path | str = "models",
                 requested_device: str = "cpu",
                 duration_seconds: float = DEFAULT_DURATION_SECONDS,
                 warmup_runs: int = DEFAULT_WARMUP_RUNS,
                 measured_runs: int = DEFAULT_MEASURED_RUNS,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 run_complexity: bool = True,
                 run_minimum_input: bool = True):
        
        self.registry = ModelRegistry(models_dir)
        self.device_selection = DeviceManager.select(requested_device)
        self.duration_seconds = duration_seconds
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.random_seed = random_seed
        self.run_complexity = run_complexity
        self.run_minimum_input = run_minimum_input

    def profile(self, spec: ModelSpec) -> Dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        result = self._base_result(spec, started_at)
        warnings: List[str] = []
        if self.device_selection.fallback_reason:
            warnings.append(self.device_selection.fallback_reason)

        try:
            model, module, sample_rate = self.registry.load(spec)
            result["model_identity"].update({"model_class": f"{spec.module_path}.{spec.class_name}",
                                             "checkpoint_path": str(self.registry.checkpoint_path(spec)),
                                             "model_configuration": spec.init_kwargs,
                                             "module_sample_rate": sample_rate})
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append(self._error("model_initialization_or_checkpoint_loading", exc))
            result["warnings"] = warnings
            return _json_safe(result)

        try:
            model.to(self.device_selection.torch_device)
            model.eval()
        
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append(self._error("device_transfer", exc))
            result["warnings"] = warnings
            return _json_safe(result)

        sample_count = seconds_to_samples(self.duration_seconds, sample_rate)
        reference_count = seconds_to_samples(DEFAULT_DURATION_SECONDS, sample_rate)
        input_tensor = self._make_input(sample_count, self.device_selection.torch_device)
        result["input"] = self._input_result(sample_rate, sample_count, input_tensor, reference_count, spec)
        result["parameters"] = ParameterProfiler.profile(model)

        sampler = MemorySampler(self.device_selection.effective)
        result["runtime_memory"]["model_loaded_state"] = sampler.model_loaded_state()

        if self.run_complexity:
            result["complexity"]["reference_10s"] = self._complexity_for(model, reference_count)
            if reference_count != sample_count:
                result["complexity"]["requested_duration"] = self._complexity_for(model, sample_count)
            else:
                result["complexity"]["requested_duration"] = result["complexity"]["reference_10s"]
        else:
            result["complexity"]["status"] = "skipped"

        if self.run_minimum_input:
            finder = MinimumInputFinder(model=model,
                                        sample_rate=sample_rate,
                                        device=self.device_selection.effective,
                                        make_input=lambda samples: self._make_input(samples, 
                                                                                    self.device_selection.torch_device))
            result["minimum_input"] = finder.find(upper_seconds=max(self.duration_seconds, DEFAULT_DURATION_SECONDS))
        else:
            result["minimum_input"] = {"status": "skipped"}

        try:
            runner = InferenceRunner(model, self.device_selection.effective)
            runner.warmup(input_tensor, self.warmup_runs)
            times, before_memory, after_memory = runner.measured(input_tensor, self.measured_runs)
            result["timing"] = {"warmup_count": self.warmup_runs,
                                "measured_run_count": self.measured_runs,
                                "measured_times": times,
                                "statistics": StatisticsCalculator.timing(times)}
            
            result["runtime_memory"]["per_run_before"] = before_memory
            result["runtime_memory"]["per_run_after"] = after_memory
            result["runtime_memory"]["summary"] = MemorySampler.summarize(self.device_selection.effective, before_memory, after_memory)
            
            if self.device_selection.effective == "cuda":
                result["cuda_vram"] = self._cuda_vram(result["runtime_memory"]["model_loaded_state"], before_memory, after_memory)
            
            result["status"] = "success"
        
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append(self._error("timed_inference", exc))

        result["warnings"] = warnings + self._collect_warnings(result)
        
        return _json_safe(result)

    def _make_input(self, samples: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + samples)
        tensor = torch.randn((1, samples), generator=generator, dtype=torch.float32)
        
        return tensor.to(device)

    def _complexity_for(self, model: nn.Module, samples: int) -> Dict[str, Any]:
        input_tensor = self._make_input(samples, self.device_selection.torch_device)
        try:
            return ComplexityProfiler.profile(model, input_tensor)
        
        except Exception as exc:
            return {"input_shape": list(input_tensor.shape),
                    "flops": None,
                    "macs_estimated": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc)}

    def _base_result(self, spec: ModelSpec, started_at: str) -> Dict[str, Any]:
        return {"schema_version": "v1",
                "status": "started",
                "model_identity": {"model_name": spec.name, "model_class": None, "checkpoint_path": None, "model_configuration": spec.init_kwargs},
                "environment": self._environment(started_at),
                "input": {},
                "timing": {},
                "parameters": {},
                "runtime_memory": {},
                "cuda_vram": None,
                "complexity": {},
                "minimum_input": {},
                "reproducibility": {"random_seed": self.random_seed,
                                    "dummy_input": "torch.randn((1, samples), dtype=torch.float32)",
                                    "dtype": "torch.float32",
                                    "model_mode": "model.eval()",
                                    "inference_context": "torch.inference_mode()",
                                    "gradient_disabled": True},
                "warnings": [],
                "errors": []}

    def _environment(self, started_at: str) -> Dict[str, Any]:
        return {"timestamp_utc": started_at,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "operating_system": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "requested_device": self.device_selection.requested,
                "effective_device": self.device_selection.effective,
                "fallback_reason": self.device_selection.fallback_reason,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def _input_result(self, sample_rate: int, samples: int, input_tensor: torch.Tensor, reference_samples: int, spec: ModelSpec) -> Dict[str, Any]:
        return {"sample_rate": sample_rate,
                "user_duration_seconds": self.duration_seconds,
                "number_of_samples": samples,
                "waveform_shape": list(input_tensor.shape),
                "input_rank": spec.input_rank,
                "input_description": spec.input_description,
                "reference_10s": {"duration_seconds": DEFAULT_DURATION_SECONDS,
                                  "number_of_samples": reference_samples,
                                  "waveform_shape": [1, reference_samples]}}

    @staticmethod
    def _cuda_vram(model_loaded: Dict[str, Any], before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"unit": "decimal MB",
                "model_loaded_state": model_loaded,
                "per_run_before": before,
                "per_run_after": after}

    @staticmethod
    def _error(stage: str, exc: BaseException) -> Dict[str, str]:
        return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}

    @staticmethod
    def _collect_warnings(result: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        for complexity in result.get("complexity", {}).values():
            if isinstance(complexity, dict):
                fvcore = complexity.get("fvcore", {})
                warnings.extend(fvcore.get("warnings", []))
                torchinfo = complexity.get("torchinfo", {})
                warnings.extend(torchinfo.get("warnings", []))
        memory_summary = result.get("runtime_memory", {}).get("summary", {})
        warnings.extend(memory_summary.get("limitations", []))
        
        return sorted(set(warnings))


class ResultsManager:
    """Write one JSON result per model profiling execution."""

    def __init__(self, output_dir: Path | str = "profile_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: Dict[str, Any]) -> Path:
        model_name = result.get("model_identity", {}).get("model_name", "unknown_model")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{model_name}_{stamp}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(result), handle, indent=2)
            handle.write("\n")
        
        return path
