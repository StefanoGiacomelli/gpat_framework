#!/usr/bin/env python3
"""Desktop GUI for model-agnostic GP-AT real-time inference."""

from __future__ import annotations

import sys
import tempfile
import time
import json
import pickle
import shutil
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
REAL_TIME_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REAL_TIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_TIME_ROOT))

from PySide6.QtCore import QMimeData, QProcess, QProcessEnvironment, QPointF, QRectF, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QImage, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

from gpat_realtime import (  # noqa: E402
    AudioSetLabelResolver,
    available_models,
    default_audioset_labels_path,
    load_audio_file,
)
from gpat_realtime.youtube import extract_youtube_video_id  # noqa: E402


class DropBox(QFrame):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropBox")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel("Drop audio file here")
        title.setObjectName("dropTitle")
        subtitle = QLabel("or click Browse. Supported: wav, flac, aiff, mp3, ogg, m4a, aac.")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._has_file(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()

    @staticmethod
    def _has_file(mime: QMimeData) -> bool:
        return bool(mime.hasUrls() and mime.urls()[0].isLocalFile())


class SpectrogramProbabilityView(QWidget):
    """Spectrogram background with Top-N probability trajectories overlaid."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(260)
        self.spectrogram: Optional[np.ndarray] = None
        self.audio_duration = 0.0
        self.result = None
        self.top_n = 5
        self.playhead_seconds = 0.0
        self.visible_seconds = 0.0
        self._tracks: List[Tuple[int, str, np.ndarray]] = []
        self.colors = [
            QColor("#006d77"),
            QColor("#d62828"),
            QColor("#f77f00"),
            QColor("#3a86ff"),
            QColor("#6a4c93"),
            QColor("#2a9d8f"),
            QColor("#bc4749"),
            QColor("#5f0f40"),
            QColor("#4361ee"),
            QColor("#9b5de5"),
        ]

    def set_spectrogram(self, spectrogram: np.ndarray, duration_seconds: float) -> None:
        self.spectrogram = spectrogram
        self.audio_duration = max(0.0, float(duration_seconds))
        self.visible_seconds = self.audio_duration
        self.update()

    def set_result(self, result) -> None:
        self.result = result
        self._recompute_tracks()
        self.update()

    def set_top_n(self, value: int) -> None:
        self.top_n = max(1, int(value))
        self._recompute_tracks()
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead_seconds = max(0.0, float(seconds))
        self.update()

    def legend_text(self) -> str:
        if not self._tracks:
            return "No tracks to display yet"
        labels = []
        for pos, (idx, name, _) in enumerate(self._tracks):
            color = self.colors[pos % len(self.colors)].name()
            labels.append(
                f'<span style="color:{color}; font-weight:700;">{idx}: {escape(name)}</span>'
            )
        return " &nbsp; | &nbsp; ".join(labels)

    def _recompute_tracks(self) -> None:
        self._tracks = []
        if self.result is None or not self.result.forward_records:
            return
        arrays = self.result.full_forward_arrays()
        probs = arrays["forward_probabilities"]
        if probs.size == 0:
            return
        means = np.nanmean(probs, axis=0)
        top_indices = np.argsort(means)[::-1][: self.top_n]
        resolver = AudioSetLabelResolver(default_audioset_labels_path(ROOT))
        for index in top_indices:
            try:
                name = resolver.resolve(int(index), output_size=probs.shape[1]).name
            except Exception:
                name = f"class_{int(index)}"
            self._tracks.append((int(index), name, probs[:, int(index)]))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        plot_rect = self.rect().adjusted(54, 18, -18, -36)
        painter.setPen(QPen(QColor("#d8dee5"), 1))
        painter.drawRect(plot_rect)
        painter.setPen(QColor("#64717d"))
        painter.drawText(12, plot_rect.top() + 6, "1.0")
        painter.drawText(12, plot_rect.bottom(), "0.0")

        if self.spectrogram is None:
            painter.drawText(plot_rect, Qt.AlignCenter, "Select, capture, or acquire audio to show the spectrogram")
            return

        image = self._spectrogram_image(self.spectrogram)
        painter.setOpacity(0.46)
        painter.drawImage(plot_rect, image)
        painter.setOpacity(1.0)

        duration = self.audio_duration or 1.0
        xmin = 0.0
        xmax = duration
        if self.visible_seconds and self.visible_seconds < duration:
            xmin = min(max(0.0, self.playhead_seconds - self.visible_seconds * 0.25), duration - self.visible_seconds)
            xmax = xmin + self.visible_seconds
        painter.setPen(QColor("#64717d"))
        painter.drawText(plot_rect.left(), self.height() - 12, f"{xmin:.1f}s")
        painter.drawText(plot_rect.right() - 54, self.height() - 12, f"{xmax:.1f}s")

        def x_map(value):
            return plot_rect.left() + (float(value) - xmin) / max(1e-9, xmax - xmin) * plot_rect.width()

        def y_map(value):
            return plot_rect.top() + (1.0 - float(value)) * plot_rect.height()

        if self.result is None:
            painter.drawText(plot_rect, Qt.AlignCenter, "Run inference to overlay Top-N probability tracks")
            return

        arrays = self.result.full_forward_arrays()
        times = arrays["forward_timestamps"]
        if times.size == 0:
            return

        threshold = self.result.tracker.config.threshold
        painter.setPen(QPen(QColor("#8a96a3"), 1, Qt.DashLine))
        painter.drawLine(plot_rect.left(), y_map(threshold), plot_rect.right(), y_map(threshold))

        for track_pos, (_, _, values) in enumerate(self._tracks):
            painter.setPen(QPen(self.colors[track_pos % len(self.colors)], 2))
            previous = None
            for idx, timestamp in enumerate(times):
                if timestamp < xmin or timestamp > xmax:
                    continue
                point = QPointF(x_map(timestamp), y_map(values[idx]))
                if previous is not None:
                    painter.drawLine(previous, point)
                previous = point

        painter.setPen(QPen(QColor("#172026"), 1))
        playhead_x = x_map(self.playhead_seconds)
        painter.drawLine(playhead_x, plot_rect.top(), playhead_x, plot_rect.bottom())

    @staticmethod
    def _spectrogram_image(spec: np.ndarray) -> QImage:
        data = np.asarray(spec, dtype=np.float32)
        if data.size == 0:
            data = np.zeros((2, 2), dtype=np.float32)
        data = np.flipud(data)
        low, high = np.nanpercentile(data, [5, 98])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
        # Low-saturation blue-gray ramp so probability tracks remain readable.
        r = (238 - normalized * 72).astype(np.uint8)
        g = (244 - normalized * 84).astype(np.uint8)
        b = (248 - normalized * 92).astype(np.uint8)
        rgb = np.dstack([r, g, b]).copy()
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return image.copy()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GP-AT Real-Time Inference")
        self.resize(1320, 760)
        self.setMinimumSize(760, 520)
        self.result = None
        self.worker_process: Optional[QProcess] = None
        self.worker_request_path: Optional[Path] = None
        self.worker_failed_message: Optional[str] = None
        self.worker_stdout_buffer = ""
        self.worker_completed = False
        self.suppress_worker_finish_error = False
        self.capture_stream = None
        self.capture_chunks: List[np.ndarray] = []
        self.capture_sample_rate = 0
        self.capture_level = 0.0
        self.captured_audio_path: Optional[Path] = None
        self.youtube_audio_path: Optional[Path] = None
        self._recording_blink_counter = 0
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        self._build_ui()
        self._load_audio_devices()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda: self._cleanup_gui_cache(log=False))

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)

        root = QWidget()
        scroll.setWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(18, 14, 18, 14)
        page.setSpacing(10)

        title = QLabel("GP-AT Real-Time Inference")
        title.setObjectName("title")
        subtitle = QLabel("Choose a model, select an input source, preview the audio spectrogram, then run adaptive inference and export complete traces.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        page.addWidget(title)
        page.addWidget(subtitle)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self._build_model_panel(), 2)
        top.addWidget(self._build_adaptive_panel(), 1)
        page.addLayout(top)
        page.addWidget(self._build_output_panel())

        content = QHBoxLayout()
        content.setSpacing(12)
        source_col = QVBoxLayout()
        source_col.addWidget(self._build_source_panel(), 1)
        source_col.addWidget(self._build_log_panel())
        content.addLayout(source_col, 4)
        content.addWidget(self._build_visual_panel(), 6)
        page.addLayout(content, 1)

        self.setStyleSheet(STYLE)

    def _build_model_panel(self) -> QGroupBox:
        box = QGroupBox("Model and runtime")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        self.model_combo = QComboBox()
        for item in available_models():
            self.model_combo.addItem(f"{item['name']} ({item['sample_rate']} Hz)", item["name"])
        self.model_combo.currentIndexChanged.connect(self._apply_model_defaults)
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_button = QPushButton("Browse")
        self.checkpoint_button.clicked.connect(self._browse_checkpoint)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "mps", "cuda", "cpu"])
        grid.addWidget(QLabel("Model"), 0, 0)
        grid.addWidget(self.model_combo, 0, 1, 1, 2)
        grid.addWidget(QLabel("Checkpoint"), 1, 0)
        grid.addWidget(self.checkpoint_edit, 1, 1)
        grid.addWidget(self.checkpoint_button, 1, 2)
        grid.addWidget(QLabel("Compute device"), 2, 0)
        grid.addWidget(self.device_combo, 2, 1, 1, 2)
        return box

    def _build_adaptive_panel(self) -> QGroupBox:
        box = QGroupBox("Adaptive inference")
        grid = QGridLayout(box)
        self.threshold = self._spin(0, 100, 50)
        self.initial_window = self._spin(10, 10000, 310)
        self.max_window = self._spin(10, 20000, 1000)
        self.adapt_coeff = self._spin(0, 100, 40)
        self.top_n = self._spin(1, 25, 5)
        self.top_n.valueChanged.connect(lambda value: self._set_top_n(value))
        grid.addWidget(QLabel("Probability threshold (%)"), 0, 0)
        grid.addWidget(self.threshold, 0, 1)
        grid.addWidget(QLabel("Initial/min window size (ms)"), 1, 0)
        grid.addWidget(self.initial_window, 1, 1)
        grid.addWidget(QLabel("Maximum window size (ms)"), 2, 0)
        grid.addWidget(self.max_window, 2, 1)
        grid.addWidget(QLabel("Adaptation coefficient (%)"), 3, 0)
        grid.addWidget(self.adapt_coeff, 3, 1)
        grid.addWidget(QLabel("Display Top-N classes"), 4, 0)
        grid.addWidget(self.top_n, 4, 1)
        return box

    def _build_output_panel(self) -> QGroupBox:
        box = QGroupBox("Export")
        layout = QHBoxLayout(box)
        self.output_edit = QLineEdit(str(ROOT / "realtime_results" / "gui_run"))
        self.output_edit.setMaximumWidth(620)
        self.output_button = QPushButton("Choose path")
        self.output_button.clicked.connect(self._browse_output)
        self.csv_check = QCheckBox("Write CSV")
        self.csv_check.setChecked(True)
        self.clean_cache_button = QPushButton("Clean Cache")
        self.clean_cache_button.setObjectName("cleanCacheButton")
        self.clean_cache_button.clicked.connect(self._clean_cache_clicked)
        layout.addWidget(QLabel("Output path without extension"))
        layout.addWidget(self.output_edit, 1)
        layout.addWidget(self.output_button)
        layout.addWidget(self.csv_check)
        layout.addWidget(self.clean_cache_button)
        layout.addStretch(1)
        return box

    def _build_source_panel(self) -> QGroupBox:
        box = QGroupBox("Input source")
        layout = QVBoxLayout(box)
        buttons = QHBoxLayout()
        self.file_mode_button = QPushButton("File")
        self.file_mode_button.setObjectName("fileButton")
        self.device_mode_button = QPushButton("Audio device")
        self.device_mode_button.setObjectName("deviceButton")
        self.youtube_mode_button = QPushButton("▶ YouTube")
        self.youtube_mode_button.setObjectName("youtubeButton")
        for idx, button in enumerate([self.file_mode_button, self.device_mode_button, self.youtube_mode_button]):
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=idx: self._select_source(i))
            buttons.addWidget(button)
        self.file_mode_button.setChecked(True)
        layout.addLayout(buttons)
        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self._build_file_page())
        self.source_stack.addWidget(self._build_device_page())
        self.source_stack.addWidget(self._build_youtube_page())
        layout.addWidget(self.source_stack, 1)
        self.run_button = QPushButton("Run inference")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._run_selected_input)
        layout.addWidget(self.run_button)
        return box

    def _build_file_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel("Select an audio file. The spectrogram appears immediately; inference overlays Top-N probability tracks.")
        hint.setWordWrap(True)
        self.drop_box = DropBox()
        self.drop_box.fileDropped.connect(self._set_audio_file)
        self.audio_file_edit = QLineEdit()
        browse = QPushButton("Browse audio file")
        browse.clicked.connect(self._browse_audio)
        row = QHBoxLayout()
        row.addWidget(self.audio_file_edit)
        row.addWidget(browse)
        layout.addWidget(hint)
        layout.addWidget(self.drop_box, 1)
        layout.addLayout(row)
        return page

    def _build_device_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel("Record from an input device. Choose channel, monitor the level, stop capture, then run inference on the captured buffer.")
        hint.setWordWrap(True)
        self.device_input_combo = QComboBox()
        self.device_input_combo.currentIndexChanged.connect(self._refresh_channels)
        self.channel_combo = QComboBox()
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._load_audio_devices)
        self.capture_button = QPushButton("Start capture")
        self.capture_button.clicked.connect(self._toggle_capture)
        self.recording_indicator = QLabel("● REC")
        self.recording_indicator.setObjectName("recordingIndicator")
        self.recording_indicator.setVisible(False)
        grid = QGridLayout()
        grid.addWidget(QLabel("Input device"), 0, 0)
        grid.addWidget(self.device_input_combo, 0, 1)
        grid.addWidget(refresh, 0, 2)
        grid.addWidget(QLabel("Channel"), 1, 0)
        grid.addWidget(self.channel_combo, 1, 1, 1, 2)
        grid.addWidget(QLabel("Input meter"), 2, 0)
        grid.addWidget(self.level_meter, 2, 1, 1, 2)
        capture_row = QHBoxLayout()
        capture_row.addWidget(self.capture_button)
        capture_row.addWidget(self.recording_indicator)
        capture_row.addStretch(1)
        layout.addWidget(hint)
        layout.addLayout(grid)
        layout.addLayout(capture_row)
        layout.addStretch(1)
        return page

    def _build_youtube_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel(
            "Paste a YouTube URL. Run inference will download and decode the audio with yt-dlp/ffmpeg, "
            "then the common playback control above the spectrogram will play the decoded audio."
        )
        hint.setWordWrap(True)
        self.youtube_edit = QLineEdit()
        self.youtube_edit.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.youtube_audio_status = QLabel("No YouTube audio downloaded yet")
        self.youtube_audio_status.setObjectName("youtubeStatus")
        self.youtube_audio_status.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(self.youtube_edit)
        layout.addWidget(hint)
        layout.addLayout(row)
        layout.addWidget(self.youtube_audio_status)
        layout.addStretch(1)
        return page

    def _build_visual_panel(self) -> QGroupBox:
        box = QGroupBox("Spectrogram and probability tracks")
        layout = QVBoxLayout(box)
        playback = QHBoxLayout()
        self.play_button = QPushButton("Play/Pause file")
        self.play_button.clicked.connect(self._toggle_playback)
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.valueChanged.connect(self._set_volume)
        self.audio_output.setVolume(self.volume.value() / 100.0)
        playback.addWidget(self.play_button)
        playback.addWidget(QLabel("Playback volume"))
        playback.addWidget(self.volume, 1)
        layout.addLayout(playback)
        self.plot = SpectrogramProbabilityView()
        layout.addWidget(self.plot, 1)
        self.legend = QLabel("No tracks yet")
        self.legend.setObjectName("legend")
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMaximumHeight(135)
        layout.addWidget(self.metrics_view)
        return box

    def _build_log_panel(self) -> QGroupBox:
        box = QGroupBox("Status")
        layout = QVBoxLayout(box)
        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(100)
        layout.addWidget(self.status)
        return box

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _select_source(self, index: int) -> None:
        self.source_stack.setCurrentIndex(index)
        for i, button in enumerate([self.file_mode_button, self.device_mode_button, self.youtube_mode_button]):
            button.setChecked(i == index)
        self._update_playback_label()
        if index == 2:
            self.media_player.pause()
        if index == 0 and self.audio_file_edit.text().strip():
            self.media_player.setSource(QUrl.fromLocalFile(self.audio_file_edit.text().strip()))
        elif index == 1 and self.captured_audio_path is not None:
            self.media_player.setSource(QUrl.fromLocalFile(str(self.captured_audio_path)))
        elif index == 2 and self.youtube_audio_path is not None:
            self.media_player.setSource(QUrl.fromLocalFile(str(self.youtube_audio_path)))

    def _update_playback_label(self) -> None:
        labels = {
            0: "Play/Pause file",
            1: "Play/Pause recording",
            2: "Play/Pause YouTube audio",
        }
        self.play_button.setText(labels.get(self.source_stack.currentIndex(), "Play/Pause"))

    def _apply_model_defaults(self) -> None:
        # The wrappers accept one-dimensional raw waveforms; 310 ms remains the
        # conservative default used by the adaptive engine. The GUI treats this
        # value as both initial and minimum runtime window.
        self.initial_window.setValue(max(310, self.initial_window.value()))
        self.max_window.setValue(max(self.max_window.value(), self.initial_window.value()))

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(ROOT),
            "Checkpoints (*.pt *.pth *.ckpt);;All files (*)",
        )
        if path:
            self.checkpoint_edit.setText(path)

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            str(ROOT),
            "Audio files (*.wav *.flac *.aiff *.aif *.mp3 *.ogg *.m4a *.aac);;All files (*)",
        )
        if path:
            self._set_audio_file(path)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose output path without extension",
            self.output_edit.text(),
            "Output path (*)",
        )
        if path:
            self.output_edit.setText(str(Path(path).with_suffix("")))

    def _set_audio_file(self, path: str) -> None:
        self.audio_file_edit.setText(path)
        self.media_player.setSource(QUrl.fromLocalFile(path))
        self._log(f"Selected audio file: {path}")
        self._load_spectrogram_preview(path)

    def _load_spectrogram_preview(self, path: str) -> None:
        try:
            audio = load_audio_file(path, target_sample_rate=16000)
            self._set_spectrogram_from_samples(audio.samples, audio.sample_rate, audio.duration_seconds)
        except Exception as exc:
            self._log(f"Spectrogram preview failed: {exc}")

    def _set_spectrogram_from_samples(self, samples: np.ndarray, sample_rate: int, duration: float) -> None:
        spec = compute_preview_spectrogram(samples, sample_rate)
        self.plot.set_spectrogram(spec, duration)
        self.legend.setText("Spectrogram ready. Run inference to overlay probability tracks.")

    def _load_audio_devices(self) -> None:
        self.device_input_combo.blockSignals(True)
        self.device_input_combo.clear()
        self.channel_combo.clear()
        if sd is None:
            self.device_input_combo.addItem("sounddevice not available", None)
            self.device_input_combo.blockSignals(False)
            return
        devices = sd.query_devices()
        for idx, info in enumerate(devices):
            if int(info.get("max_input_channels", 0)) > 0:
                self.device_input_combo.addItem(f"{idx}: {info['name']}", idx)
        default_input = None
        try:
            default_device = sd.default.device
            if isinstance(default_device, (list, tuple)):
                default_input = int(default_device[0])
            else:
                default_input = int(default_device)
        except Exception:
            default_input = None
        if default_input is not None and default_input >= 0:
            default_combo_index = self.device_input_combo.findData(default_input)
            if default_combo_index >= 0:
                self.device_input_combo.setCurrentIndex(default_combo_index)
        self.device_input_combo.blockSignals(False)
        self._refresh_channels()

    def _refresh_channels(self) -> None:
        self.channel_combo.clear()
        if sd is None:
            return
        idx = self.device_input_combo.currentData()
        if idx is None:
            return
        info = sd.query_devices(idx)
        for channel in range(int(info.get("max_input_channels", 0))):
            self.channel_combo.addItem(f"Channel {channel + 1}", channel)

    def _toggle_capture(self) -> None:
        if self.capture_stream is None:
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self) -> None:
        if sd is None:
            self._error("sounddevice is not available")
            return
        device = self.device_input_combo.currentData()
        channel = self.channel_combo.currentData()
        if device is None or channel is None:
            self._error("Select an input device and channel")
            return
        info = sd.query_devices(device)
        self.capture_sample_rate = int(info["default_samplerate"])
        self.capture_chunks = []
        self.captured_audio_path = None
        self.media_player.stop()
        self.media_player.setSource(QUrl())

        def callback(indata, frames, timestamp, status):
            data = np.asarray(indata[:, int(channel)], dtype=np.float32).copy()
            self.capture_chunks.append(data)
            self.capture_level = min(1.0, float(np.sqrt(np.mean(np.square(data))) * 12.0))

        self.capture_stream = sd.InputStream(
            device=device,
            channels=int(info["max_input_channels"]),
            samplerate=self.capture_sample_rate,
            callback=callback,
        )
        self.capture_stream.start()
        self.capture_button.setText("Stop capture")
        self.recording_indicator.setVisible(True)
        self._recording_blink_counter = 0
        self._log(f"Capture started: {info['name']} @ {self.capture_sample_rate} Hz")

    def _stop_capture(self) -> None:
        if self.capture_stream is not None:
            self.capture_stream.stop()
            self.capture_stream.close()
            self.capture_stream = None
        self.capture_button.setText("Start capture")
        self.recording_indicator.setVisible(False)
        samples = np.concatenate(self.capture_chunks) if self.capture_chunks else np.array([], dtype=np.float32)
        seconds = len(samples) / max(1, self.capture_sample_rate)
        self._log(f"Capture stopped: {seconds:.2f}s recorded")
        if samples.size:
            self._set_spectrogram_from_samples(samples, self.capture_sample_rate, seconds)
            self._set_captured_playback_source(samples)

    def _set_captured_playback_source(self, samples: np.ndarray) -> None:
        output_dir = Path(tempfile.gettempdir()) / "gpat_realtime_gui" / "captures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"captured_{int(time.time() * 1000)}.wav"
        sf.write(path, np.asarray(samples, dtype=np.float32), self.capture_sample_rate)
        self.captured_audio_path = path
        self.media_player.setSource(QUrl.fromLocalFile(str(path)))
        self._log(f"Captured audio ready for playback: {path}")

    def _toggle_playback(self) -> None:
        if self.source_stack.currentIndex() == 2 and self.youtube_audio_path is None:
            self._error("Run inference first to download and decode the YouTube audio")
            return
        if self.source_stack.currentIndex() == 1 and self.captured_audio_path is None:
            self._error("Record audio before playback")
            return
        if self.source_stack.currentIndex() == 0 and not self.audio_file_edit.text().strip():
            self._error("Select an audio file before playback")
            return
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _set_volume(self, value: int) -> None:
        self.audio_output.setVolume(value / 100.0)

    def _run_selected_input(self) -> None:
        try:
            request = self._build_request()
        except Exception as exc:
            self._error(str(exc))
            return
        self.run_button.setEnabled(False)
        self.status.clear()
        self.metrics_view.clear()
        self._log("Starting inference")
        self._start_worker(request)

    def _start_worker(self, request: Dict) -> None:
        request_dir = Path(tempfile.gettempdir()) / "gpat_realtime_gui"
        request_dir.mkdir(parents=True, exist_ok=True)
        self.worker_request_path = request_dir / "last_request.json"
        self.worker_request_path.write_text(json.dumps(request), encoding="utf-8")
        self.worker_failed_message = None
        self.worker_stdout_buffer = ""
        self.worker_completed = False

        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        existing_pythonpath = environment.value("PYTHONPATH")
        pythonpath = f"{ROOT}:{REAL_TIME_ROOT}"
        if existing_pythonpath:
            pythonpath = f"{pythonpath}:{existing_pythonpath}"
        environment.insert("PYTHONPATH", pythonpath)
        environment.insert("NUMBA_DISABLE_JIT", "1")
        environment.insert("OMP_NUM_THREADS", "1")
        environment.insert("OPENBLAS_NUM_THREADS", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(ROOT))
        process.setProgram(sys.executable)
        process.setArguments(
            [
                str(REAL_TIME_ROOT / "gpat_realtime" / "subprocess_worker.py"),
                "--request-json",
                str(self.worker_request_path),
            ]
        )
        process.readyReadStandardOutput.connect(self._worker_stdout_ready)
        process.readyReadStandardError.connect(self._worker_stderr_ready)
        process.errorOccurred.connect(self._worker_process_error)
        process.finished.connect(self._worker_process_finished)
        self.worker_process = process
        process.start()

    def _worker_stdout_ready(self) -> None:
        if self.worker_process is None:
            return
        chunk = bytes(self.worker_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.worker_stdout_buffer += chunk
        while "\n" in self.worker_stdout_buffer:
            line, self.worker_stdout_buffer = self.worker_stdout_buffer.split("\n", 1)
            self._handle_worker_line(line.strip())

    def _worker_stderr_ready(self) -> None:
        if self.worker_process is None:
            return
        text = bytes(self.worker_process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            for line in text.splitlines():
                self._log(f"worker stderr: {line}")

    def _handle_worker_line(self, line: str) -> None:
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._log(line)
            return
        event = payload.get("event")
        if event == "progress":
            self._log(str(payload.get("message", "")))
        elif event == "failed":
            self.worker_failed_message = str(payload.get("message", "Worker failed"))
            self._log(f"ERROR: {self.worker_failed_message}")
            traceback_text = str(payload.get("traceback", "")).strip()
            if traceback_text:
                for line in traceback_text.splitlines()[-12:]:
                    self._log(f"traceback: {line}")
        elif event == "finished":
            self._load_worker_result(payload)

    def _load_worker_result(self, payload: Dict) -> None:
        try:
            result_path = Path(payload["result_pickle"])
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
            written = payload.get("written", {})
            self.worker_completed = True
            self._worker_finished(result, written)
        except Exception as exc:
            self.worker_failed_message = f"Could not load worker result: {exc}"
            self._log(f"ERROR: {self.worker_failed_message}")

    def _worker_process_error(self, error: QProcess.ProcessError) -> None:
        if self.suppress_worker_finish_error:
            return
        error_name = getattr(error, "name", str(error))
        self.worker_failed_message = f"Worker process error: {error_name}"
        self._log(f"ERROR: {self.worker_failed_message}")

    def _worker_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.worker_stdout_buffer.strip():
            self._handle_worker_line(self.worker_stdout_buffer.strip())
            self.worker_stdout_buffer = ""
        crashed = exit_status == QProcess.CrashExit
        self.worker_process = None
        if self.suppress_worker_finish_error:
            self.suppress_worker_finish_error = False
            return
        if crashed or exit_code != 0 or not self.worker_completed:
            message = self.worker_failed_message or f"Worker process failed: exit_code={exit_code}, crashed={crashed}"
            self._worker_failed(message)

    @staticmethod
    def _gui_cache_dir() -> Path:
        return Path(tempfile.gettempdir()) / "gpat_realtime_gui"

    def _clean_cache_clicked(self) -> None:
        try:
            removed = self._cleanup_gui_cache(log=True)
            if removed:
                self._log("GUI cache cleaned")
            else:
                self._log("GUI cache already empty")
        except Exception as exc:
            self._error(f"Could not clean GUI cache: {exc}")

    def _cleanup_gui_cache(self, log: bool = True) -> bool:
        cache_dir = self._gui_cache_dir().resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if cache_dir == temp_root or temp_root not in cache_dir.parents:
            raise RuntimeError(f"Refusing to delete unsafe cache path: {cache_dir}")

        if self.worker_process is not None:
            self.suppress_worker_finish_error = True
            self.worker_process.kill()
            self.worker_process.waitForFinished(2000)
            self.worker_process = None
        if self.capture_stream is not None:
            self._stop_capture()
        self.media_player.stop()
        self.media_player.setSource(QUrl())

        removed = False
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed = True
        self.worker_request_path = None
        self.worker_failed_message = None
        self.worker_stdout_buffer = ""
        self.worker_completed = False
        self.suppress_worker_finish_error = False
        self.captured_audio_path = None
        self.youtube_audio_path = None
        if hasattr(self, "youtube_audio_status"):
            self.youtube_audio_status.setText("No YouTube audio downloaded yet")
        if log and removed:
            self._log(f"Removed temporary GUI cache: {cache_dir}")
        return removed

    def closeEvent(self, event) -> None:
        try:
            self._cleanup_gui_cache(log=False)
        except Exception as exc:
            self._log(f"Cache cleanup on close failed: {exc}")
        super().closeEvent(event)

    def _build_request(self) -> Dict:
        checkpoint = self.checkpoint_edit.text().strip()
        if not checkpoint:
            raise ValueError("Select a checkpoint path")
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        output_path = self.output_edit.text().strip()
        if not output_path:
            raise ValueError("Choose an output path")
        initial_ms = self.initial_window.value()
        max_ms = max(self.max_window.value(), initial_ms)
        request = {
            "model": self.model_combo.currentData(),
            "checkpoint": checkpoint,
            # Track all classes in the engine enough to export all forward
            # probabilities; Top-N display is selected from full outputs.
            "classes": [0],
            "device": self.device_combo.currentText(),
            "config": {
                "threshold": self.threshold.value() / 100.0,
                "initial_window_duration": initial_ms / 1000.0,
                "min_window_duration": initial_ms / 1000.0,
                "max_window_duration": max_ms / 1000.0,
                "adapt_width_coeff": self.adapt_coeff.value() / 100.0,
            },
            "output_path": output_path,
            "include_csv": self.csv_check.isChecked(),
            "work_dir": str(Path(tempfile.gettempdir()) / "gpat_realtime_gui"),
        }
        index = self.source_stack.currentIndex()
        if index == 0:
            audio = self.audio_file_edit.text().strip()
            if not audio:
                raise ValueError("Select or drop an audio file")
            request.update({"input_kind": "file", "audio_path": audio})
        elif index == 1:
            if self.capture_stream is not None:
                self._stop_capture()
            if not self.capture_chunks:
                raise ValueError("Capture audio before running inference")
            if self.captured_audio_path is None:
                samples = np.concatenate(self.capture_chunks)
                self._set_captured_playback_source(samples)
            request.update(
                {
                    "input_kind": "captured",
                    "audio_path": str(self.captured_audio_path),
                    "device_info": {
                        "device": self.device_input_combo.currentText(),
                        "channel": self.channel_combo.currentText(),
                    },
                }
            )
        else:
            url = self.youtube_edit.text().strip()
            if not url:
                raise ValueError("Enter a YouTube URL")
            extract_youtube_video_id(url)
            self.youtube_audio_path = None
            self.youtube_audio_status.setText("Downloading and decoding YouTube audio...")
            request.update({"input_kind": "youtube", "youtube_url": url})
        return request

    def _worker_finished(self, result, written: Dict[str, str]) -> None:
        self.result = result
        self.plot.set_result(result)
        self._set_top_n(self.top_n.value())
        self._maybe_load_result_spectrogram(result)
        self.metrics_view.setPlainText(self._format_metrics(result, written))
        self.run_button.setEnabled(True)
        self._log("Inference completed")

    def _maybe_load_result_spectrogram(self, result) -> None:
        path = result.audio_info.get("path")
        if path and Path(path).exists():
            self._load_spectrogram_preview(path)
            self.plot.set_result(result)
            self._set_top_n(self.top_n.value())
            self.media_player.setSource(QUrl.fromLocalFile(path))
            if result.source.get("type") == "youtube":
                self.youtube_audio_path = Path(path)
                title = result.source.get("title") or result.source.get("video_id") or Path(path).name
                self.youtube_audio_status.setText(f"Decoded audio ready for playback: {title}")

    def _worker_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._error(message)

    def _set_top_n(self, value: int) -> None:
        self.plot.set_top_n(value)
        self.legend.setText(self.plot.legend_text())

    def _format_metrics(self, result, written: Dict[str, str]) -> str:
        metrics = result.runtime_metrics.to_dict()
        timing = metrics["timing"]
        resources = metrics["resource_usage"]
        lines = [
            "Runtime metrics",
            f"Model: {result.model_info['model_name']} on {result.model_info['effective_device']}",
            f"Audio: {result.audio_info['duration_seconds']:.3f}s, decoder={result.audio_info['decoder']}",
            f"Forward calls: {len(result.forward_records)}",
            f"Replay buffer updates: {len(result.runtime_metrics.buffer_metrics)}",
            f"Total wall time: {timing['total_wall_seconds']:.3f}s",
            f"Total forward time: {timing['total_forward_seconds']:.3f}s",
            f"CPU samples: {resources['num_samples']}",
            "Exports:",
        ]
        for key, path in written.items():
            lines.append(f"  {key}: {path}")
        return "\n".join(lines)

    def _tick(self) -> None:
        self.level_meter.setValue(int(self.capture_level * 100))
        if self.capture_stream is not None:
            self._recording_blink_counter = (self._recording_blink_counter + 1) % 8
            self.recording_indicator.setVisible(self._recording_blink_counter < 4)
        else:
            self.recording_indicator.setVisible(False)
        if self.media_player.duration() > 0:
            self.plot.set_playhead(self.media_player.position() / 1000.0)

    def _log(self, message: str) -> None:
        self.status.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _error(self, message: str) -> None:
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "GP-AT inference error", message)


def compute_preview_spectrogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        return np.zeros((64, 2), dtype=np.float32)
    frame = 1024
    hop = 512
    if data.size < frame:
        data = np.pad(data, (0, frame - data.size))
    windows = []
    window = np.hanning(frame).astype(np.float32)
    for start in range(0, max(1, data.size - frame + 1), hop):
        chunk = data[start:start + frame] * window
        windows.append(np.abs(np.fft.rfft(chunk))[:96])
    spec = np.asarray(windows, dtype=np.float32).T
    return 20.0 * np.log10(np.maximum(spec, 1e-8))


STYLE = """
* {
    color: #111820;
    selection-background-color: #d8e8eb;
    selection-color: #111820;
}
QMainWindow, QWidget {
    background: #f4f6f8;
    color: #111820;
}
QLabel {
    color: #111820;
    font-size: 13px;
    background: transparent;
}
QLabel#title {
    font-size: 24px;
    font-weight: 800;
}
QLabel#subtitle {
    color: #34424f;
    font-size: 13px;
}
QLabel#dropTitle {
    font-size: 18px;
    font-weight: 700;
}
QLabel#legend {
    color: #111820;
    font-weight: 700;
}
QLabel#recordingIndicator {
    color: #d00000;
    font-size: 14px;
    font-weight: 900;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d3dce4;
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    font-weight: 700;
    color: #111820;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #111820;
    background: #ffffff;
}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
    background: #ffffff;
    color: #111820;
    border: 1px solid #c8d3dd;
    border-radius: 6px;
    padding: 7px;
    selection-background-color: #cfe1e5;
    selection-color: #111820;
}
QComboBox QAbstractItemView {
    background: #2f3337;
    color: #f3f5f7;
    selection-background-color: #59646d;
    selection-color: #ffffff;
    outline: 0;
}
QComboBox QAbstractItemView::item {
    color: #f3f5f7;
    min-height: 24px;
    padding: 3px 8px;
}
QComboBox QAbstractItemView::item:selected {
    background: #59646d;
    color: #ffffff;
}
QSpinBox::up-button, QSpinBox::down-button {
    background: #eef2f5;
    border-left: 1px solid #c8d3dd;
    width: 18px;
}
QPushButton {
    background: #006d77;
    color: #ffffff;
    border: 1px solid #006d77;
    border-radius: 6px;
    padding: 8px 12px;
    font-weight: 700;
}
QPushButton:disabled {
    background: #95a3ad;
    border-color: #95a3ad;
}
QPushButton:checked {
    border: 2px solid #111820;
}
QPushButton#fileButton {
    background: #16803c;
    border-color: #16803c;
}
QPushButton#deviceButton {
    background: #1f5fbf;
    border-color: #1f5fbf;
}
QPushButton#youtubeButton {
    background: #ff0000;
    border-color: #ff0000;
}
QPushButton#runButton {
    background: #111820;
    border-color: #111820;
}
QPushButton#cleanCacheButton {
    background: #b00020;
    border-color: #b00020;
    color: #ffffff;
}
QPushButton#cleanCacheButton:hover {
    background: #8f001a;
    border-color: #8f001a;
}
QFrame#dropBox {
    background: #eef7f1;
    border: 2px dashed #16803c;
    border-radius: 8px;
    min-height: 105px;
}
QProgressBar {
    color: #111820;
    border: 1px solid #c8d3dd;
    border-radius: 6px;
    background: #ffffff;
    text-align: center;
}
QProgressBar::chunk {
    background: #2a9d8f;
    border-radius: 5px;
}
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 12))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
