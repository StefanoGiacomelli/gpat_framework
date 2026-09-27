"""Runtime and technical metrics for GP-AT inference workflows."""

from __future__ import annotations

import os
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional fallback
    psutil = None


@dataclass
class ForwardMetric:
    timestamp: float
    window_samples: int
    duration_seconds: float
    output_size: int


@dataclass
class BufferMetric:
    timestamp: float
    cursor_sample: int
    step_samples: int
    requested_window_samples: List[int]
    available_samples: int


@dataclass
class RuntimeMetrics:
    started_at_utc: str
    ended_at_utc: Optional[str] = None
    total_wall_seconds: Optional[float] = None
    audio_load_seconds: Optional[float] = None
    audio_resample_seconds: Optional[float] = None
    model_load_seconds: Optional[float] = None
    requested_device: Optional[str] = None
    effective_device: Optional[str] = None
    sample_rate: Optional[int] = None
    audio_duration_seconds: Optional[float] = None
    forward_metrics: List[ForwardMetric] = field(default_factory=list)
    buffer_metrics: List[BufferMetric] = field(default_factory=list)
    cpu_percent_samples: List[float] = field(default_factory=list)
    rss_mb_samples: List[float] = field(default_factory=list)

    def finish(self, started_perf: float) -> None:
        self.ended_at_utc = datetime.now(timezone.utc).isoformat()
        self.total_wall_seconds = time.perf_counter() - started_perf

    def to_dict(self) -> Dict:
        forward_times = [m.duration_seconds for m in self.forward_metrics]
        window_sizes = [m.window_samples for m in self.forward_metrics]
        total_forward = sum(forward_times)
        realtime_factor = None
        if self.audio_duration_seconds and self.total_wall_seconds and self.total_wall_seconds > 0:
            realtime_factor = self.audio_duration_seconds / self.total_wall_seconds
        return {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "process_id": os.getpid(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            },
            "device": {
                "requested": self.requested_device,
                "effective": self.effective_device,
            },
            "timing": {
                "total_wall_seconds": self.total_wall_seconds,
                "audio_load_seconds": self.audio_load_seconds,
                "audio_resample_seconds": self.audio_resample_seconds,
                "model_load_seconds": self.model_load_seconds,
                "total_forward_seconds": total_forward,
                "realtime_factor_audio_seconds_per_wall_second": realtime_factor,
                "forward_seconds": _summary(forward_times),
            },
            "resource_usage": {
                "cpu_percent": _summary(self.cpu_percent_samples),
                "process_rss_mb": _summary(self.rss_mb_samples),
                "num_samples": len(self.cpu_percent_samples),
            },
            "inference": {
                "forward_calls": len(self.forward_metrics),
                "window_samples": _summary(window_sizes),
                "window_seconds": _summary(
                    [w / self.sample_rate for w in window_sizes] if self.sample_rate else []
                ),
            },
            "buffer": {
                "updates": len(self.buffer_metrics),
                "step_samples": _summary([m.step_samples for m in self.buffer_metrics]),
                "available_samples": _summary([m.available_samples for m in self.buffer_metrics]),
                "requested_window_count": _summary(
                    [len(m.requested_window_samples) for m in self.buffer_metrics]
                ),
            },
        }


class RuntimeMetricsCollector:
    """Collect CPU/RAM samples while recording forward and buffer metrics."""

    def __init__(self, sample_interval: float = 0.1):
        self.metrics = RuntimeMetrics(started_at_utc=datetime.now(timezone.utc).isoformat())
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_perf = time.perf_counter()
        self._process = psutil.Process(os.getpid()) if psutil else None

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._process:
            self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> RuntimeMetrics:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.metrics.finish(self._started_perf)
        return self.metrics

    def record_forward(
        self,
        timestamp: float,
        window_samples: int,
        duration_seconds: float,
        output_size: int,
    ) -> None:
        self.metrics.forward_metrics.append(
            ForwardMetric(timestamp, window_samples, duration_seconds, output_size)
        )

    def record_buffer_update(
        self,
        timestamp: float,
        cursor_sample: int,
        step_samples: int,
        requested_window_samples: List[int],
        available_samples: int,
    ) -> None:
        self.metrics.buffer_metrics.append(
            BufferMetric(
                timestamp=timestamp,
                cursor_sample=cursor_sample,
                step_samples=step_samples,
                requested_window_samples=list(requested_window_samples),
                available_samples=available_samples,
            )
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            if self._process is None:
                continue
            try:
                self.metrics.cpu_percent_samples.append(float(self._process.cpu_percent(interval=None)))
                self.metrics.rss_mb_samples.append(
                    float(self._process.memory_info().rss) / (1024.0 * 1024.0)
                )
            except Exception:
                continue


def _summary(values: List[float]) -> Dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    numeric = [float(v) for v in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
    }
