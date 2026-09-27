"""Independent adaptive temporal windows for monitored classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .labels import ClassInfo


@dataclass(frozen=True)
class AdaptiveWindowConfig:
    initial_duration: float = 0.310
    min_duration: float = 0.310
    max_duration: float = 1.0
    threshold: float = 0.5
    adapt_width_coeff: float = 0.4

    def validate(self) -> None:
        if self.initial_duration <= 0 or self.min_duration <= 0:
            raise ValueError("initial_duration and min_duration must be > 0")
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration must be >= min_duration")
        if not (self.min_duration <= self.initial_duration <= self.max_duration):
            raise ValueError("initial_duration must be inside [min_duration, max_duration]")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if self.adapt_width_coeff < 0:
            raise ValueError("adapt_width_coeff must be >= 0")


@dataclass
class ClassTrackerState:
    class_info: ClassInfo
    sample_rate: int
    config: AdaptiveWindowConfig
    current_window_samples: int = field(init=False)
    timestamps: List[float] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    window_sizes_samples: List[int] = field(default_factory=list)
    forward_record_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config.validate()
        self.current_window_samples = self._clamp(
            int(round(self.config.initial_duration * self.sample_rate))
        )

    @property
    def min_window_samples(self) -> int:
        return int(round(self.config.min_duration * self.sample_rate))

    @property
    def max_window_samples(self) -> int:
        return int(round(self.config.max_duration * self.sample_rate))

    def record(self, timestamp: float, probability: float, forward_record_index: int) -> None:
        probability = float(probability)
        self.timestamps.append(float(timestamp))
        self.probabilities.append(probability)
        self.window_sizes_samples.append(int(self.current_window_samples))
        self.forward_record_indices.append(int(forward_record_index))
        if probability >= self.config.threshold:
            increment = int(round(self.config.adapt_width_coeff * self.sample_rate))
            self.current_window_samples = self._clamp(self.current_window_samples + increment)
        else:
            self.current_window_samples = self.min_window_samples

    def _clamp(self, value: int) -> int:
        return max(self.min_window_samples, min(int(value), self.max_window_samples))

    def to_dict(self) -> Dict:
        return {
            "class_index": self.class_info.index,
            "class_mid": self.class_info.mid,
            "class_name": self.class_info.name,
            "selector": self.class_info.selector,
            "timestamps": self.timestamps,
            "probabilities": self.probabilities,
            "window_sizes_samples": self.window_sizes_samples,
            "window_durations_seconds": [
                value / self.sample_rate for value in self.window_sizes_samples
            ],
            "forward_record_indices": self.forward_record_indices,
            "current_window_seconds": self.current_window_samples / self.sample_rate,
        }


class MultiClassAdaptiveTracker:
    def __init__(self, classes: List[ClassInfo], sample_rate: int, config: AdaptiveWindowConfig):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if not classes:
            raise ValueError("At least one monitored class is required")
        self.classes = classes
        self.sample_rate = sample_rate
        self.config = config
        self.states: Dict[int, ClassTrackerState] = {
            info.index: ClassTrackerState(info, sample_rate, config) for info in classes
        }

    @property
    def step_samples(self) -> int:
        return min(state.min_window_samples for state in self.states.values())

    def requested_window_sizes(self) -> List[int]:
        return sorted({state.current_window_samples for state in self.states.values()})

    def record_many(self, timestamp: float, outputs_by_window: Dict[int, np.ndarray], record_indices: Dict[int, int]) -> None:
        for state in self.states.values():
            window = state.current_window_samples
            output = outputs_by_window[window]
            if state.class_info.index >= output.shape[0]:
                raise RuntimeError(
                    f"Class index {state.class_info.index} out of range for output size {output.shape[0]}"
                )
            state.record(
                timestamp=timestamp,
                probability=float(output[state.class_info.index]),
                forward_record_index=record_indices[window],
            )

    def to_dict(self) -> Dict:
        return {
            "sample_rate": self.sample_rate,
            "adaptation": {
                "rule": (
                    "Each monitored class owns an independent window. If its score is "
                    ">= threshold, its next window grows by adapt_width_coeff * sample_rate "
                    "samples; otherwise it resets to min_duration. Values are clamped to "
                    "[min_duration, max_duration]."
                ),
                "initial_duration": self.config.initial_duration,
                "min_duration": self.config.min_duration,
                "max_duration": self.config.max_duration,
                "threshold": self.config.threshold,
                "adapt_width_coeff": self.config.adapt_width_coeff,
                "hysteresis": None,
                "smoothing": None,
            },
            "classes": [state.to_dict() for state in self.states.values()],
        }
