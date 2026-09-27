"""YouTube audio acquisition for synchronized replay inference."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})")


@dataclass(frozen=True)
class YouTubeAudioSource:
    url: str
    video_id: str
    audio_path: str
    title: Optional[str] = None


def extract_youtube_video_id(url: str) -> str:
    match = YOUTUBE_ID_RE.search(url)
    if match:
        return match.group(1)
    stripped = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", stripped):
        return stripped
    raise ValueError(f"Could not extract a YouTube video id from: {url}")


def acquire_youtube_audio(
    url: str,
    output_dir: Union[str, Path],
    sample_rate: int,
    overwrite: bool = False,
) -> YouTubeAudioSource:
    """Acquire YouTube audio as mono WAV at ``sample_rate``.

    This implements synchronized replay architecture: the browser/video player
    supplies user-facing playback while inference runs on separately decoded
    audio with the same timeline.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for YouTube audio decoding")
    video_id = extract_youtube_video_id(url)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{video_id}_{sample_rate}.wav"
    if wav_path.exists() and not overwrite:
        return YouTubeAudioSource(url=url, video_id=video_id, audio_path=str(wav_path))
    title = _download_with_python_ytdlp(url, wav_path, sample_rate)
    if title is None:
        title = _download_with_executable_ytdlp(url, wav_path, sample_rate)
    if not wav_path.exists():
        raise RuntimeError(f"YouTube acquisition completed but output file is missing: {wav_path}")
    return YouTubeAudioSource(url=url, video_id=video_id, audio_path=str(wav_path), title=title)


def _download_with_python_ytdlp(url: str, wav_path: Path, sample_rate: int) -> Optional[str]:
    try:
        import yt_dlp
    except Exception:
        return None
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(wav_path.with_suffix(".%(ext)s")),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": ["-ac", "1", "-ar", str(sample_rate)],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
        generated = wav_path.with_suffix(".wav")
        if generated != wav_path and generated.exists():
            generated.replace(wav_path)
        return info.get("title") if isinstance(info, dict) else None
    except Exception as exc:
        raise RuntimeError(f"yt_dlp Python package failed to acquire audio: {exc}") from exc


def _download_with_executable_ytdlp(url: str, wav_path: Path, sample_rate: int) -> Optional[str]:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not available as Python package or executable")
    template = str(wav_path.with_suffix(".%(ext)s"))
    command = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ac 1 -ar {sample_rate}",
        "--output",
        template,
        url,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"yt-dlp executable failed to acquire audio: {details}")
    generated = wav_path.with_suffix(".wav")
    if generated != wav_path and generated.exists():
        generated.replace(wav_path)
    return None
