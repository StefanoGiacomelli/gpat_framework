"""Core model-agnostic GP-AT inference engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from .adaptive import AdaptiveWindowConfig, MultiClassAdaptiveTracker
from .audio import AudioData, load_audio_file
from .labels import ClassInfo, ClassSelector
from .metrics import RuntimeMetrics, RuntimeMetricsCollector
from .models import GPATModelAdapter, load_model_adapter


@dataclass(frozen=True)
class RealtimeEngineConfig:
    initial_window_duration: float = 0.310
    min_window_duration: float = 0.310
    max_window_duration: float = 1.0
    threshold: float = 0.5
    adapt_width_coeff: float = 0.4
    pad_final_window: bool = True
    simulate_realtime: bool = False
    metrics_sample_interval: float = 0.1

    def adaptive_config(self) -> AdaptiveWindowConfig:
        return AdaptiveWindowConfig(
            initial_duration=self.initial_window_duration,
            min_duration=self.min_window_duration,
            max_duration=self.max_window_duration,
            threshold=self.threshold,
            adapt_width_coeff=self.adapt_width_coeff,
        )


@dataclass
class ForwardRecord:
    timestamp: float
    window_samples: int
    duration_seconds: float
    probabilities: np.ndarray


@dataclass
class RealtimeInferenceResult:
    model_info: Dict
    monitored_classes: List[ClassInfo]
    tracker: MultiClassAdaptiveTracker
    sample_rate: int
    source: Dict
    audio_info: Dict
    runtime_metrics: RuntimeMetrics
    forward_records: List[ForwardRecord] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        states = list(self.tracker.states.values())
        return len(states[0].timestamps) if states else 0

    def selected_arrays(self) -> Dict[str, np.ndarray]:
        states = list(self.tracker.states.values())
        timestamps = np.asarray(states[0].timestamps if states else [], dtype=np.float64)
        probabilities = np.stack(
            [np.asarray(state.probabilities, dtype=np.float32) for state in states],
            axis=1,
        ) if states else np.empty((0, 0), dtype=np.float32)
        windows = np.stack(
            [np.asarray(state.window_sizes_samples, dtype=np.int64) for state in states],
            axis=1,
        ) if states else np.empty((0, 0), dtype=np.int64)
        return {
            "selected_timestamps": timestamps,
            "selected_probabilities": probabilities,
            "selected_window_sizes_samples": windows,
            "selected_window_durations_seconds": windows.astype(np.float64) / self.sample_rate,
            "selected_class_indices": np.asarray([s.class_info.index for s in states], dtype=np.int64),
            "selected_forward_record_indices": np.stack(
                [np.asarray(s.forward_record_indices, dtype=np.int64) for s in states],
                axis=1,
            ) if states else np.empty((0, 0), dtype=np.int64),
        }

    def full_forward_arrays(self) -> Dict[str, np.ndarray]:
        if not self.forward_records:
            return {
                "forward_timestamps": np.array([], dtype=np.float64),
                "forward_window_sizes_samples": np.array([], dtype=np.int64),
                "forward_durations_seconds": np.array([], dtype=np.float64),
                "forward_probabilities": np.empty((0, 0), dtype=np.float32),
            }
        return {
            "forward_timestamps": np.asarray([r.timestamp for r in self.forward_records], dtype=np.float64),
            "forward_window_sizes_samples": np.asarray([r.window_samples for r in self.forward_records], dtype=np.int64),
            "forward_durations_seconds": np.asarray([r.duration_seconds for r in self.forward_records], dtype=np.float64),
            "forward_probabilities": np.stack([r.probabilities.astype(np.float32) for r in self.forward_records], axis=0),
        }

    def to_metadata(self) -> Dict:
        return {
            "schema_version": "gpat_realtime_v2",
            "execution_model": (
                "timeline-based replay or live block processing with model-specific "
                "resampling; file and YouTube workflows use decoded audio replay"
            ),
            "model": self.model_info,
            "sample_rate": self.sample_rate,
            "source": self.source,
            "audio": self.audio_info,
            "monitored_classes": [
                {
                    "index": info.index,
                    "mid": info.mid,
                    "name": info.name,
                    "selector": info.selector,
                }
                for info in self.monitored_classes
            ],
            "tracker": self.tracker.to_dict(),
            "runtime_metrics": self.runtime_metrics.to_dict(),
            "arrays_schema": {
                "forward_probabilities": "(forward_calls, model_classes), all classes from every model forward",
                "selected_probabilities": "(steps, monitored_classes), class-specific adaptive trajectories",
            },
        }


class GPATRealtimeEngine:
    def __init__(
        self,
        adapter: GPATModelAdapter,
        monitored_classes: Iterable[Union[ClassSelector, ClassInfo]],
        config: Optional[RealtimeEngineConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or RealtimeEngineConfig()
        class_infos: List[ClassInfo] = []
        selectors: List[ClassSelector] = []
        for item in monitored_classes:
            if isinstance(item, ClassInfo):
                class_infos.append(item)
            else:
                selectors.append(item)
        if selectors:
            class_infos.extend(adapter.resolve_classes(selectors))
        if not class_infos:
            raise ValueError("At least one monitored class must be selected")
        self.monitored_classes = class_infos

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        checkpoint_path: Optional[Union[str, Path]],
        monitored_classes: Iterable[ClassSelector],
        device: str = "auto",
        project_root: Optional[Union[str, Path]] = None,
        config: Optional[RealtimeEngineConfig] = None,
    ) -> "GPATRealtimeEngine":
        adapter = load_model_adapter(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
            project_root=project_root,
        )
        return cls(adapter=adapter, monitored_classes=monitored_classes, config=config)

    def run_file(self, audio_path: Union[str, Path]) -> RealtimeInferenceResult:
        audio = load_audio_file(audio_path, target_sample_rate=self.adapter.sample_rate)
        return self.run_audio_data(audio, source={"type": "file", "path": str(audio_path)})

    def run_audio_data(self, audio: AudioData, source: Optional[Dict] = None) -> RealtimeInferenceResult:
        metrics = RuntimeMetricsCollector(sample_interval=self.config.metrics_sample_interval)
        metrics.metrics.audio_load_seconds = audio.load_seconds
        metrics.metrics.audio_resample_seconds = audio.resample_seconds
        metrics.metrics.model_load_seconds = self.adapter.model_load_seconds
        metrics.metrics.requested_device = self.adapter.requested_device
        metrics.metrics.effective_device = self.adapter.effective_device
        metrics.metrics.sample_rate = self.adapter.sample_rate
        metrics.metrics.audio_duration_seconds = audio.duration_seconds
        metrics.start()
        try:
            result = self._run_samples(audio.samples, metrics, source or {"type": "array"}, audio)
        finally:
            runtime = metrics.stop()
        result.runtime_metrics = runtime
        return result

    def _run_samples(
        self,
        samples: np.ndarray,
        metrics: RuntimeMetricsCollector,
        source: Dict,
        audio: AudioData,
    ) -> RealtimeInferenceResult:
        waveform = np.asarray(samples, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform, got shape {waveform.shape}")
        tracker = MultiClassAdaptiveTracker(
            classes=self.monitored_classes,
            sample_rate=self.adapter.sample_rate,
            config=self.config.adaptive_config(),
        )
        forward_records: List[ForwardRecord] = []
        cursor = 0
        step = tracker.step_samples
        total_samples = int(waveform.size)

        while cursor < total_samples:
            timestamp = cursor / self.adapter.sample_rate
            requested = tracker.requested_window_sizes()
            metrics.record_buffer_update(
                timestamp=timestamp,
                cursor_sample=cursor,
                step_samples=step,
                requested_window_samples=requested,
                available_samples=max(0, total_samples - cursor),
            )
            outputs_by_window: Dict[int, np.ndarray] = {}
            record_indices: Dict[int, int] = {}
            for window_samples in requested:
                frame = self._slice_frame(waveform, cursor, window_samples)
                started = time.perf_counter()
                probabilities = self.adapter.predict_proba(frame).squeeze(0).numpy()
                duration = time.perf_counter() - started
                record_indices[window_samples] = len(forward_records)
                outputs_by_window[window_samples] = probabilities
                forward_records.append(
                    ForwardRecord(timestamp, window_samples, duration, probabilities)
                )
                metrics.record_forward(
                    timestamp=timestamp,
                    window_samples=window_samples,
                    duration_seconds=duration,
                    output_size=int(probabilities.shape[0]),
                )
            tracker.record_many(timestamp, outputs_by_window, record_indices)
            if self.config.simulate_realtime:
                time.sleep(step / self.adapter.sample_rate)
            cursor += step

        audio_info = {
            "path": audio.path,
            "duration_seconds": audio.duration_seconds,
            "sample_rate": audio.sample_rate,
            "original_sample_rate": audio.original_sample_rate,
            "channels": audio.channels,
            "decoder": audio.decoder,
            "num_samples": int(waveform.size),
        }
        return RealtimeInferenceResult(
            model_info=self.adapter.info.__dict__,
            monitored_classes=self.monitored_classes,
            tracker=tracker,
            sample_rate=self.adapter.sample_rate,
            source=source,
            audio_info=audio_info,
            runtime_metrics=metrics.metrics,
            forward_records=forward_records,
        )

    def _slice_frame(self, audio: np.ndarray, start: int, window_samples: int) -> np.ndarray:
        frame = audio[start:start + window_samples]
        if frame.size < window_samples and self.config.pad_final_window:
            frame = np.pad(frame, (0, window_samples - frame.size), mode="constant")
        return frame
