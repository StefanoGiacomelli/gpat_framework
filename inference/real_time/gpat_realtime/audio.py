"""Audio loading, decoding, mono conversion, and resampling."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    original_sample_rate: Optional[int]
    duration_seconds: float
    path: Optional[str]
    decoder: str
    load_seconds: float
    resample_seconds: float
    channels: int


def supported_audio_extensions() -> str:
    return "*.wav *.flac *.aiff *.aif *.mp3 *.ogg *.m4a *.aac"


def load_audio_file(
    audio_path: Union[str, Path],
    target_sample_rate: int,
    normalize: bool = False,
) -> AudioData:
    """Load an audio file as mono float32 at ``target_sample_rate``.

    Decoding order is soundfile, ffmpeg, then librosa. This covers common PCM
    formats and compressed formats such as MP3/M4A when ffmpeg is installed.
    """

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be > 0")

    started = time.perf_counter()
    data, source_sr, channels, decoder = _decode_audio(path, target_sample_rate)
    load_seconds = time.perf_counter() - started

    if data.ndim > 1:
        channels = data.shape[1]
        data = np.mean(data, axis=1)
    else:
        channels = max(1, channels)
    data = np.asarray(data, dtype=np.float32)

    resample_seconds = 0.0
    if source_sr != target_sample_rate:
        resample_started = time.perf_counter()
        data = _resample(data, source_sr, target_sample_rate)
        resample_seconds = time.perf_counter() - resample_started

    if normalize:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 0:
            data = data / peak

    duration = len(data) / target_sample_rate
    return AudioData(
        samples=data.astype(np.float32, copy=False),
        sample_rate=target_sample_rate,
        original_sample_rate=source_sr,
        duration_seconds=duration,
        path=str(path),
        decoder=decoder,
        load_seconds=load_seconds,
        resample_seconds=resample_seconds,
        channels=channels,
    )


def prepare_audio_array(
    samples: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
    path: Optional[str] = None,
    decoder: str = "array",
    normalize: bool = False,
) -> AudioData:
    """Convert an in-memory waveform to mono float32 at the target sample rate."""

    started = time.perf_counter()
    data = np.asarray(samples, dtype=np.float32)
    channels = data.shape[1] if data.ndim > 1 else 1
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    load_seconds = time.perf_counter() - started
    resample_seconds = 0.0
    if source_sample_rate != target_sample_rate:
        resample_started = time.perf_counter()
        data = _resample(data, source_sample_rate, target_sample_rate)
        resample_seconds = time.perf_counter() - resample_started
    if normalize:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 0:
            data = data / peak
    return AudioData(
        samples=data.astype(np.float32, copy=False),
        sample_rate=target_sample_rate,
        original_sample_rate=source_sample_rate,
        duration_seconds=len(data) / target_sample_rate,
        path=path,
        decoder=decoder,
        load_seconds=load_seconds,
        resample_seconds=resample_seconds,
        channels=channels,
    )


def _decode_audio(path: Path, target_sample_rate: int):
    try:
        data, sr = sf.read(str(path), always_2d=False)
        channels = data.shape[1] if getattr(data, "ndim", 1) > 1 else 1
        return data, int(sr), channels, "soundfile"
    except Exception:
        pass

    if shutil.which("ffmpeg") is not None:
        return _decode_with_ffmpeg(path, target_sample_rate)

    try:
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=False)
        if data.ndim > 1:
            data = data.T
        channels = data.shape[1] if data.ndim > 1 else 1
        return data, int(sr), channels, "librosa"
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode {path}. soundfile failed, ffmpeg is not available, "
            f"and librosa failed: {exc}"
        ) from exc


def _decode_with_ffmpeg(path: Path, target_sample_rate: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Could not decode {path}. soundfile/librosa failed and ffmpeg is not available."
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(command, capture_output=True, check=False)
    if proc.returncode != 0:
        details = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {details}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    return data, target_sample_rate, 1, "ffmpeg"


def _resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is not None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(source_sr),
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ]
        proc = subprocess.run(
            command,
            input=np.asarray(data, dtype=np.float32).tobytes(),
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            details = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg resampling failed: {details}")
        return np.frombuffer(proc.stdout, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(data, orig_sr=source_sr, target_sr=target_sr).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Resampling requires ffmpeg or librosa: {exc}") from exc
