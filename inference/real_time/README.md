# GP-AT Real-Time Inference

This directory contains the final model-agnostic real-time/replay inference workflow.

## Final Entry Points

### `main_rt_inference.py`

Runs fixed-window live inference from an input audio device. The script lists
available input devices/channels, asks the user which input to use, then prints
continuously updated Top-N class probabilities with terminal bars. It also writes
a JSONL session log under `realtime_results/` unless `--log-output` is provided.

```bash
.venv/bin/python main_rt_inference.py \
  --model panns_wavegram_logmel_cnn14 \
  --checkpoint models/panns/Wavegram_Logmel_Cnn14_mAP=0.439.pth \
  --device auto \
  --window-size 1.0 \
  --hop-size 0.5 \
  --top-n 10
```

### `main_inference_gui.py`

Launches a PySide6 desktop GUI with:

- model/checkpoint/device selection;
- local file drag/drop and file picker;
- local playback controls and volume;
- audio input device/channel capture with meter;
- YouTube URL ingestion through `yt-dlp`, followed by decoded-audio playback and inference;
- scrolling probability visualization;
- runtime metrics and export paths.
- `Clean Cache`, which removes GUI-generated temporary downloads, captured WAVs,
  request/result handoff files, and is also run automatically on exit.

The GUI runs inference in a separate Python subprocess, so native crashes from
model/audio dependencies do not terminate the desktop interface.

```bash
.venv/bin/python main_inference_gui.py
```

## Architecture

The `gpat_realtime` package isolates:

- model registry and adapter normalization for all local models in `models/`;
- AudioSet class name/MID/index resolution;
- audio decoding/resampling;
- independent per-class adaptive temporal windows;
- runtime technical metrics;
- structured export.

There are no application-specific assumptions about any monitored class.
