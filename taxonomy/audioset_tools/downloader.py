"""YouTube audio downloader for normalized AudioSet-Tools records."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .models import DownloadJob, DownloadReportEntry, is_valid_youtube_id, safe_output_stem


class YouTubeDatasetDownloader:
    """Download and post-process YouTube clips from normalized jobs.

    This class adapts the useful behavior from the previous AudioSet-Tools
    downloader while avoiding in-place metadata mutation and text-only reports.
    """

    def __init__(self,
                 output_dir: str | Path,
                 target_sr: int = 44100,
                 channels: str = "stereo",
                 normalize: bool = False,
                 cookies_file: str | Path | None = None,
                 retries: int = 1,
                 sleep_seconds: float = 0.0,
                 verbose: bool = False,
                 dry_run: bool = False,) -> None:
        
        if channels not in {"stereo", "mono", "mono_split"}:
            raise ValueError("channels must be one of 'stereo', 'mono', or 'mono_split'.")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        self.channels = channels
        self.normalize = normalize
        self.cookies_file = Path(cookies_file) if cookies_file else None
        self.retries = max(0, retries)
        self.sleep_seconds = max(0.0, sleep_seconds)
        self.verbose = verbose
        self.dry_run = dry_run

    def download_dataset(self,
                         dataset,
                         limit: int | None = None,
                         report_path: str | Path | None = None) -> list[DownloadReportEntry]:
        return self.download_jobs(dataset.to_download_jobs(), limit=limit, report_path=report_path)

    def download_jobs(self,
                      jobs: Iterable[DownloadJob],
                      limit: int | None = None,
                      report_path: str | Path | None = None) -> list[DownloadReportEntry]:
        job_list = list(jobs)
        if limit is not None:
            job_list = job_list[:limit]

        report_entries: list[DownloadReportEntry] = []
        for job in tqdm(job_list, desc="Downloading YouTube clips"):
            report_entries.append(self.download_job(job))
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

        if report_path is not None:
            write_download_report(report_entries, report_path)
        
        return report_entries

    def download_job(self, job: DownloadJob) -> DownloadReportEntry:
        requested_path = self._output_path(job)
        if not is_valid_youtube_id(job.youtube_id):
            return self._report(job,
                                requested_path=requested_path,
                                final_path=None,
                                status="invalid_metadata",
                                success=False,
                                error_category="InvalidYouTubeId",
                                error_message=f"Malformed YouTube ID: {job.youtube_id}")
        if self.dry_run:
            return self._report(job,
                                requested_path=requested_path,
                                final_path=None,
                                status="skipped_dry_run",
                                success=True)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                final_paths = self._download_and_process(job, requested_path)
                return self._report(job,
                                    requested_path=requested_path,
                                    final_path=final_paths[0] if final_paths else None,
                                    status="downloaded",
                                    success=True,
                                    retry_count=attempt)
            except Exception as exc:  # noqa: BLE001 - categorized in report
                last_error = exc
                if self.verbose:
                    print(f"Download failed for {job.youtube_id} on attempt {attempt + 1}: {exc}")

        return self._report(job,
                            requested_path=requested_path,
                            final_path=None,
                            status="failed",
                            success=False,
                            error_category=type(last_error).__name__ if last_error else "UnknownError",
                            error_message=str(last_error) if last_error else "Unknown error",
                            retry_count=self.retries)

    def _download_and_process(self, job: DownloadJob, output_path: Path) -> list[Path]:
        try:
            import resampy
            import soundfile as sf
            import yt_dlp
        except ImportError as exc:
            raise ImportError("Downloading requires yt-dlp, soundfile, and resampy.") from exc

        with tempfile.TemporaryDirectory(prefix="audioset_tools_") as tmp:
            tmp_dir = Path(tmp)
            outtmpl = str(tmp_dir / f"{safe_output_stem(job.youtube_id)}.%(ext)s")
            ydl_opts = {"quiet": not self.verbose,
                        "no_warnings": not self.verbose,
                        "format": "bestaudio/best",
                        "outtmpl": outtmpl,
                        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}
            if self.cookies_file:
                ydl_opts["cookiefile"] = str(self.cookies_file)

            url = f"https://www.youtube.com/watch?v={job.youtube_id}"
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            wav_files = list(tmp_dir.glob("*.wav"))
            if not wav_files:
                raise RuntimeError("yt-dlp completed without producing a WAV file.")
            
            data, sr = sf.read(wav_files[0])

            if sr != self.target_sr:
                data = resampy.resample(data, sr, self.target_sr, axis=0)

            if job.start_seconds is not None and job.end_seconds is not None:
                start_idx = max(0, int(job.start_seconds * self.target_sr))
                end_idx = max(start_idx, int(job.end_seconds * self.target_sr))
                data = data[start_idx:end_idx]

            if self.normalize:
                peak = np.max(np.abs(data)) if data.size else 0.0
                if peak > 0:
                    data = data / peak

            output_path = _next_available_path(output_path)
            
            if self.channels == "mono" and data.ndim == 2:
                data = data.mean(axis=1)
                sf.write(output_path, data, self.target_sr)
                
                return [output_path]

            if self.channels == "mono_split" and data.ndim == 2 and data.shape[1] > 1:
                left = output_path.with_name(f"{output_path.stem}_Left{output_path.suffix}")
                right = output_path.with_name(f"{output_path.stem}_Right{output_path.suffix}")
                left = _next_available_path(left)
                right = _next_available_path(right)
                sf.write(left, data[:, 0], self.target_sr)
                sf.write(right, data[:, 1], self.target_sr)
                
                return [left, right]

            sf.write(output_path, data, self.target_sr)
            
            return [output_path]

    def _output_path(self, job: DownloadJob) -> Path:
        stem = safe_output_stem(job.output_stem or job.source_id or job.youtube_id)
        return self.output_dir / f"{stem}.wav"

    def _report(self,
                job: DownloadJob,
                requested_path: Path,
                final_path: Path | None,
                status: str,
                success: bool,
                error_category: str | None = None,
                error_message: str | None = None,
                retry_count: int = 0) -> DownloadReportEntry:
        
        return DownloadReportEntry(dataset=job.dataset,
                                   source_id=job.source_id,
                                   youtube_id=job.youtube_id,
                                   start_seconds=job.start_seconds,
                                   end_seconds=job.end_seconds,
                                   requested_output_path=str(requested_path),
                                   final_output_path=str(final_path) if final_path else None,
                                   status=status,
                                   success=success,
                                   error_category=error_category,
                                   error_message=error_message,
                                   retry_count=retry_count,
                                   processing_timestamp=datetime.now(timezone.utc).isoformat(),
                                   original_label_ids=job.original_label_ids,
                                   original_labels=job.original_labels,
                                   salt_labels=job.salt_labels)


def write_download_report(entries: Iterable[DownloadReportEntry],
                          path: str | Path) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([entry.to_dict() for entry in entries], handle, indent=2)


def read_download_report(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate an output filename for {path}.")
