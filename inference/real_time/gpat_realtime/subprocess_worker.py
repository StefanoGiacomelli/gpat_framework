#!/usr/bin/env python3
"""Subprocess worker used by the GUI to isolate native-library crashes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REAL_TIME_ROOT = Path(__file__).resolve().parents[1]
WORKER_PACKAGE_DIR = Path(__file__).resolve().parent
sys.path = [item for item in sys.path if Path(item or ".").resolve() != WORKER_PACKAGE_DIR]
for path in [str(REAL_TIME_ROOT), str(ROOT)]:
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(REAL_TIME_ROOT))

from gpat_realtime import (  # noqa: E402
    GPATRealtimeEngine,
    RealtimeEngineConfig,
    export_inference_result,
    load_audio_file,
)
from gpat_realtime.youtube import acquire_youtube_audio  # noqa: E402


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}), flush=True)


def build_config(payload: dict) -> RealtimeEngineConfig:
    config = payload["config"]
    return RealtimeEngineConfig(
        threshold=float(config["threshold"]),
        initial_window_duration=float(config["initial_window_duration"]),
        min_window_duration=float(config["min_window_duration"]),
        max_window_duration=float(config["max_window_duration"]),
        adapt_width_coeff=float(config["adapt_width_coeff"]),
    )


def run_request(payload: dict) -> None:
    emit("progress", message="Loading model and checkpoint")
    engine = GPATRealtimeEngine.from_model_name(
        model_name=payload["model"],
        checkpoint_path=payload["checkpoint"],
        monitored_classes=payload["classes"],
        device=payload["device"],
        project_root=ROOT,
        config=build_config(payload),
    )

    kind = payload["input_kind"]
    if kind == "file":
        emit("progress", message="Running inference on local audio")
        result = engine.run_file(payload["audio_path"])
    elif kind == "captured":
        emit("progress", message="Preparing captured audio")
        audio = load_audio_file(payload["audio_path"], target_sample_rate=engine.adapter.sample_rate)
        result = engine.run_audio_data(
            audio,
            source={"type": "audio_device", **payload["device_info"], "path": payload["audio_path"]},
        )
    elif kind == "youtube":
        emit("progress", message="YouTube: downloading audio with yt-dlp and converting with ffmpeg")
        source = acquire_youtube_audio(
            payload["youtube_url"],
            output_dir=Path(payload["work_dir"]) / "youtube_audio",
            sample_rate=engine.adapter.sample_rate,
            overwrite=False,
        )
        emit("progress", message=f"YouTube: decoded audio ready at {source.audio_path}")
        emit("progress", message="YouTube: running inference on decoded local audio")
        result = engine.run_file(source.audio_path)
        result.source.update(
            {
                "type": "youtube",
                "youtube_url": source.url,
                "video_id": source.video_id,
                "title": source.title,
            }
        )
    else:
        raise ValueError(f"Unsupported input kind: {kind}")

    emit("progress", message="Writing export files")
    written = export_inference_result(
        result,
        payload["output_path"],
        include_csv=bool(payload["include_csv"]),
    )
    result_path = Path(tempfile.gettempdir()) / "gpat_realtime_gui" / "last_result.pkl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    emit("finished", result_pickle=str(result_path), written=written)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        run_request(payload)
        return 0
    except Exception as exc:
        emit("failed", message=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
