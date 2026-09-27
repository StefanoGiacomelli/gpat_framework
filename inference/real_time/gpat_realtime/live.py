"""Fixed-window live microphone inference for GP-AT models."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence

import numpy as np

from .labels import AudioSetLabelResolver, default_audioset_labels_path
from .models import available_models, load_model_adapter

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StreamTiming:
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    chunk_seconds: float
    buffer_seconds: float
    window_samples: int
    hop_samples: int
    chunk_samples: int
    buffer_samples: int
    warnings: List[str]


class RecentAudioBuffer:
    """Small absolute-indexed buffer for live audio windows."""

    def __init__(self, capacity_samples: int):
        self.capacity_samples = int(capacity_samples)
        self.start_index = 0
        self.samples = np.zeros(0, dtype=np.float32)

    @property
    def end_index(self) -> int:
        return self.start_index + int(self.samples.size)

    def append(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if data.size == 0:
            return
        self.samples = np.concatenate([self.samples, data])
        overflow = int(self.samples.size) - self.capacity_samples
        if overflow > 0:
            self.samples = self.samples[overflow:]
            self.start_index += overflow

    def window_ending_at(self, end_index: int, window_samples: int) -> np.ndarray:
        start_index = int(end_index) - int(window_samples)
        if start_index < self.start_index or end_index > self.end_index:
            raise ValueError("Requested live window is outside the retained audio buffer")
        local_start = start_index - self.start_index
        local_end = local_start + int(window_samples)
        return self.samples[local_start:local_end].astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    model_names = [item["name"] for item in available_models()]
    parser = argparse.ArgumentParser(
        description=(
            "Live fixed-window GP-AT inference from an input audio device. "
            "The script interactively asks for the input device and channel(s), "
            "then continuously renders Top-N class probabilities."
        )
    )
    parser.add_argument("--model", choices=model_names, help="Model wrapper name.")
    parser.add_argument("--checkpoint", help="Path to the model checkpoint.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--window-size", type=float, help="Analysis window size in seconds.")
    parser.add_argument("--hop-size", type=float, default=None, help="Update hop size in seconds. Defaults to window size.")
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=None,
        help="sounddevice callback block size in seconds. Defaults to a protected low-latency value.",
    )
    parser.add_argument(
        "--buffer-size",
        type=float,
        default=None,
        help="Internal retained audio buffer in seconds. Defaults to a protected value >= window size.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of highest-probability classes to display.")
    parser.add_argument("--device-index", type=int, default=None, help="Input audio device index. If omitted, prompt.")
    parser.add_argument(
        "--channels",
        default=None,
        help="1-based channel list, e.g. '1' or '1,2'. If omitted, prompt. Multiple channels are averaged to mono.",
    )
    parser.add_argument(
        "--log-output",
        default=None,
        help="JSONL session log path. Defaults to realtime_results/live_rt_<timestamp>.jsonl.",
    )
    parser.add_argument("--bar-width", type=int, default=36, help="Terminal probability bar width.")
    parser.add_argument("--duration", type=float, default=None, help="Optional maximum session duration in seconds.")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    return parser


def validate_timing(
    sample_rate: int,
    window_seconds: float,
    hop_seconds: Optional[float],
    chunk_seconds: Optional[float],
    buffer_seconds: Optional[float],
) -> StreamTiming:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window_seconds <= 0:
        raise ValueError("--window-size must be > 0 seconds")
    hop = window_seconds if hop_seconds is None else float(hop_seconds)
    if hop <= 0:
        raise ValueError("--hop-size must be > 0 seconds")
    if hop > window_seconds:
        raise ValueError("--hop-size cannot be greater than --window-size, otherwise input windows are skipped")

    warnings: List[str] = []
    if chunk_seconds is None:
        chunk = min(0.100, max(0.020, hop / 4.0))
        warnings.append(f"chunk-size auto-selected at {chunk:.3f}s")
    else:
        chunk = float(chunk_seconds)
    if chunk <= 0:
        raise ValueError("--chunk-size must be > 0 seconds")

    if buffer_seconds is None:
        buffer = max(5.0, window_seconds * 3.0, window_seconds + hop * 3.0, chunk * 3.0)
        warnings.append(f"buffer-size auto-selected at {buffer:.3f}s")
    else:
        buffer = float(buffer_seconds)
    if buffer <= 0:
        raise ValueError("--buffer-size must be > 0 seconds")
    if buffer < window_seconds:
        raise ValueError("--buffer-size cannot be smaller than --window-size")
    if chunk > buffer:
        raise ValueError("--chunk-size cannot be greater than --buffer-size")
    if chunk > window_seconds:
        raise ValueError("--chunk-size cannot be greater than --window-size")
    if chunk > hop:
        warnings.append(
            "chunk-size is greater than hop-size; no windows are skipped, but display latency may increase"
        )

    return StreamTiming(
        sample_rate=int(sample_rate),
        window_seconds=float(window_seconds),
        hop_seconds=float(hop),
        chunk_seconds=float(chunk),
        buffer_seconds=float(buffer),
        window_samples=max(1, int(round(window_seconds * sample_rate))),
        hop_samples=max(1, int(round(hop * sample_rate))),
        chunk_samples=max(1, int(round(chunk * sample_rate))),
        buffer_samples=max(1, int(round(buffer * sample_rate))),
        warnings=warnings,
    )


def list_input_devices() -> List[Dict]:
    if sd is None:
        raise RuntimeError("sounddevice is not available")
    devices = sd.query_devices()
    rows = []
    for idx, info in enumerate(devices):
        channels = int(info.get("max_input_channels", 0))
        if channels > 0:
            rows.append(
                {
                    "index": idx,
                    "name": str(info.get("name", "unknown")),
                    "channels": channels,
                    "default_samplerate": float(info.get("default_samplerate", 0.0)),
                }
            )
    return rows


def print_input_devices(devices: Sequence[Dict]) -> None:
    print("\nInput audio devices:")
    if not devices:
        print("  No input audio devices found.")
        return
    for item in devices:
        print(
            f"  [{item['index']}] {item['name']} "
            f"({item['channels']} ch, default {item['default_samplerate']:.0f} Hz)"
        )


def choose_device(devices: Sequence[Dict], requested_index: Optional[int]) -> Dict:
    if not devices:
        raise RuntimeError("No input audio devices were found")
    if requested_index is not None:
        for item in devices:
            if int(item["index"]) == int(requested_index):
                return dict(item)
        raise ValueError(f"Input device index not found: {requested_index}")
    print_input_devices(devices)
    while True:
        raw = input("Select input device index: ").strip()
        if not raw:
            continue
        try:
            return choose_device(devices, int(raw))
        except Exception as exc:
            print(f"Invalid device selection: {exc}")


def parse_channels(value: str, max_channels: int) -> List[int]:
    try:
        selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    except Exception as exc:
        raise ValueError("channels must be a comma-separated list of 1-based indices") from exc
    if not selected:
        raise ValueError("at least one input channel must be selected")
    unique = []
    for channel in selected:
        if channel < 1 or channel > max_channels:
            raise ValueError(f"channel {channel} is outside 1..{max_channels}")
        if channel not in unique:
            unique.append(channel)
    return [channel - 1 for channel in unique]


def choose_channels(max_channels: int, requested_channels: Optional[str]) -> List[int]:
    if requested_channels:
        return parse_channels(requested_channels, max_channels)
    print(f"Available channels: 1..{max_channels}")
    while True:
        raw = input("Select channel(s), e.g. 1 or 1,2: ").strip()
        try:
            return parse_channels(raw, max_channels)
        except Exception as exc:
            print(f"Invalid channel selection: {exc}")


def default_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "realtime_results" / f"live_rt_{stamp}.jsonl"


def render_bars(
    probabilities: np.ndarray,
    top_n: int,
    resolver: AudioSetLabelResolver,
    output_size: int,
    bar_width: int,
    header: str,
) -> str:
    probs = np.asarray(probabilities, dtype=np.float32)
    count = min(max(1, top_n), int(probs.size))
    top_indices = np.argsort(probs)[::-1][:count]
    lines = [header, ""]
    for index in top_indices:
        value = float(probs[int(index)])
        try:
            label = resolver.resolve(int(index), output_size=output_size).name
        except Exception:
            label = f"class_{int(index)}"
        filled = int(round(max(0.0, min(1.0, value)) * bar_width))
        bar = "█" * filled + " " * max(0, bar_width - filled)
        lines.append(f"{value:6.3f} |{bar}| {label} [{int(index)}]")
    lines.append("")
    lines.append("Press Ctrl+C to stop.")
    return "\n".join(lines)


class InPlaceConsoleRenderer:
    """Overwrite a fixed terminal block without growing scrollback every update."""

    def __init__(self):
        self.previous_lines = 0

    def render(self, text: str) -> None:
        lines = text.splitlines()
        if self.previous_lines:
            sys.stdout.write(f"\033[{self.previous_lines}F")
        width = max(1, self._terminal_width())
        padded_count = max(self.previous_lines, len(lines))
        for idx in range(padded_count):
            line = lines[idx] if idx < len(lines) else ""
            sys.stdout.write("\033[2K")
            sys.stdout.write(line[: width - 1])
            if idx < padded_count - 1:
                sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.previous_lines = padded_count

    @staticmethod
    def _terminal_width() -> int:
        try:
            import shutil

            return shutil.get_terminal_size(fallback=(120, 24)).columns
        except Exception:
            return 120


def write_jsonl(handle, payload: Dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def mono_reduce(frame: np.ndarray, channels: Sequence[int]) -> np.ndarray:
    selected = frame[:, list(channels)].astype(np.float32, copy=False)
    if selected.ndim == 1 or selected.shape[1] == 1:
        return selected.reshape(-1)
    return np.mean(selected, axis=1, dtype=np.float32)


def run_live(args: argparse.Namespace) -> int:
    if sd is None:
        raise RuntimeError("sounddevice is not available")
    devices = list_input_devices()
    if args.list_devices:
        print_input_devices(devices)
        return 0
    missing = []
    if not args.model:
        missing.append("--model")
    if not args.checkpoint:
        missing.append("--checkpoint")
    if args.window_size is None:
        missing.append("--window-size")
    if missing:
        raise ValueError(f"Missing required argument(s) for live inference: {', '.join(missing)}")
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")
    if args.bar_width <= 0:
        raise ValueError("--bar-width must be > 0")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be > 0 seconds")

    print("Loading model and checkpoint...")
    adapter = load_model_adapter(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        project_root=PROJECT_ROOT,
    )
    timing = validate_timing(
        sample_rate=adapter.sample_rate,
        window_seconds=args.window_size,
        hop_seconds=args.hop_size,
        chunk_seconds=args.chunk_size,
        buffer_seconds=args.buffer_size,
    )
    for warning in timing.warnings:
        print(f"NOTE: {warning}")

    selected_device = choose_device(devices, args.device_index)
    channels = choose_channels(int(selected_device["channels"]), args.channels)
    channel_text = ",".join(str(channel + 1) for channel in channels)
    log_path = Path(args.log_output) if args.log_output else default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resolver = AudioSetLabelResolver(default_audioset_labels_path(PROJECT_ROOT))
    audio_queue: Queue[np.ndarray] = Queue(maxsize=64)
    buffer = RecentAudioBuffer(timing.buffer_samples)
    next_inference_end = timing.window_samples
    dropped_chunks = 0
    inference_count = 0
    started = time.perf_counter()
    stop_requested = False
    renderer = InPlaceConsoleRenderer()

    def request_stop(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        nonlocal dropped_chunks
        if status:
            print(f"\nInput stream status: {status}", file=sys.stderr)
        mono = mono_reduce(np.asarray(indata), channels)
        try:
            audio_queue.put_nowait(mono.copy())
        except Exception:
            dropped_chunks += 1

    header_base = (
        f"{args.model} | torch={adapter.effective_device} | input={selected_device['name']} "
        f"ch={channel_text} | sr={timing.sample_rate} Hz | "
        f"window={timing.window_seconds:.3f}s hop={timing.hop_seconds:.3f}s"
    )
    print(f"Writing live session log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        write_jsonl(
            log_handle,
            {
                "event": "session_start",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "checkpoint": str(args.checkpoint),
                "requested_device": args.device,
                "effective_device": adapter.effective_device,
                "sample_rate": timing.sample_rate,
                "window_seconds": timing.window_seconds,
                "hop_seconds": timing.hop_seconds,
                "chunk_seconds": timing.chunk_seconds,
                "buffer_seconds": timing.buffer_seconds,
                "input_device": selected_device,
                "channels_1_based": [channel + 1 for channel in channels],
                "top_n": args.top_n,
            },
        )
        try:
            with sd.InputStream(
                device=int(selected_device["index"]),
                channels=int(selected_device["channels"]),
                samplerate=timing.sample_rate,
                blocksize=timing.chunk_samples,
                dtype="float32",
                callback=callback,
            ):
                while not stop_requested:
                    if args.duration is not None and (time.perf_counter() - started) >= args.duration:
                        break
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    buffer.append(chunk)
                    while buffer.end_index >= next_inference_end:
                        frame = buffer.window_ending_at(next_inference_end, timing.window_samples)
                        t0 = time.perf_counter()
                        probabilities = adapter.predict_proba(frame).squeeze(0).numpy()
                        inference_seconds = time.perf_counter() - t0
                        timestamp = next_inference_end / timing.sample_rate
                        inference_count += 1
                        header = (
                            f"{header_base} | t={timestamp:8.3f}s | "
                            f"inference={inference_seconds * 1000.0:6.1f} ms | dropped={dropped_chunks}"
                        )
                        text = render_bars(
                            probabilities=probabilities,
                            top_n=args.top_n,
                            resolver=resolver,
                            output_size=adapter.num_classes,
                            bar_width=args.bar_width,
                            header=header,
                        )
                        renderer.render(text)

                        top_indices = np.argsort(probabilities)[::-1][: min(args.top_n, probabilities.size)]
                        top = []
                        for index in top_indices:
                            info = resolver.resolve(int(index), output_size=adapter.num_classes)
                            top.append(
                                {
                                    "index": int(index),
                                    "mid": info.mid,
                                    "name": info.name,
                                    "probability": float(probabilities[int(index)]),
                                }
                            )
                        write_jsonl(
                            log_handle,
                            {
                                "event": "inference",
                                "timestamp_seconds": float(timestamp),
                                "wall_time_seconds": float(time.perf_counter() - started),
                                "window_samples": timing.window_samples,
                                "hop_samples": timing.hop_samples,
                                "inference_seconds": float(inference_seconds),
                                "dropped_chunks": int(dropped_chunks),
                                "top": top,
                            },
                        )
                        next_inference_end += timing.hop_samples
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            write_jsonl(
                log_handle,
                {
                    "event": "session_end",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "wall_time_seconds": float(time.perf_counter() - started),
                    "inference_count": int(inference_count),
                    "dropped_chunks": int(dropped_chunks),
                },
            )
    print(f"\nStopped. Session log: {log_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_live(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
