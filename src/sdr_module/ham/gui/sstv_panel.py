"""
SSTV Image Viewer panel for receiving images from the ISS and other sources.

Provides:
- Start Decoder, a mode choice ("Auto (VIS)" or a fixed mode) and Start now
  (decode a transmission already under way, from its line syncs)
- Tune to ISS (145.800 MHz): FM, narrow deviation, receiver and decoder on
- A tone meter: the dominant audio tone and what it means
- Live image preview and reception progress
- Received images (also partial ones), kept across restarts
- Auto-save (PNG, via Qt; no extra imaging library needed)
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PyQt6.QtCore import (
        QEvent,
        QPointF,
        QRectF,
        QSize,
        QStandardPaths,
        Qt,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QDesktopServices,
        QFont,
        QFontMetricsF,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..sstv import SSTV_MODES, SSTVDecoder, SSTVImageViewer, SSTVModeSpec

logger = logging.getLogger(__name__)

# Most SSTV modes (Martin, Scottie, Robot 36) are 320 x 256: a 5:4 frame.
_ASPECT = 256.0 / 320.0
_NO_VALUE = "—"  # em dash
ISS_FREQUENCY_HZ = 145.800e6
ISS_SSTV_HINT = (
    "Press Tune to ISS (145.800 MHz), or tune to an SSTV signal (FM on VHF "
    "and UHF, USB or LSB on HF) and press Start Decoder."
)
# Seconds of silence while listening before the panel says no audio arrives.
NO_AUDIO_TIMEOUT_S = 3.0
NO_AUDIO_TEXT = (
    "⚠ No audio is reaching the decoder. Start the receiver, tuned to the "
    "SSTV signal in FM (or USB or LSB on HF)."
)
# An image in progress ends (kept as a partial image) when the audio stops
# for this long: its line timing can't be trusted afterwards.
AUDIO_GAP_S = 1.5
# Start Now with Mode "Auto (VIS)" listens this many seconds of audio for
# the line syncs that tell the mode (four syncs of the slowest mode, Scottie
# DX at 1.05 s a line, and some to spare).
START_NOW_AUTO_WAIT_S = 5.0
START_NOW_WAIT_TEXT = "Listening for line syncs to tell the mode…"
START_NOW_NO_MODE_TEXT = (
    "Can't tell the mode: no line syncs were heard. Choose the mode in the "
    "Mode list, then press Start Now."
)
START_NOW_NO_AUDIO_TEXT = (
    "Start Now heard no audio to decode. Start the receiver, tuned to the "
    "SSTV signal, then press Start Now."
)
# Rows the Received Images list grows to before it scrolls.
_HISTORY_MAX_ROWS = 5
# Saved images listed again after a restart (newest first).
_HISTORY_LOAD_LIMIT = 50
AUTO_MODE_TEXT = "Auto (VIS)"
# Modes in the Mode list, most used first within each family.
MODE_ORDER = (
    "Robot 36",
    "Robot 72",
    "Martin 1",
    "Martin 2",
    "Scottie 1",
    "Scottie 2",
    "Scottie DX",
    "PD 90",
    "PD 120",
    "PD 160",
    "PD 180",
    "PD 240",
    "PD 290",
    "Wraase SC2-180",
)
# sstv_20261030_123456_Martin_1.png / ..._Martin_1_partial_180of256.png, and
# a "_2" suffix for a second image saved in the same second.
_FILE_RE = re.compile(
    r"^sstv_(\d{8}_\d{6})(?:_(.+?))?(?:_partial_(\d+)of(\d+)(?:_\d+)?)?$",
    re.IGNORECASE,
)


def _theme() -> Any:
    """The GUI theme helpers, imported lazily.

    ``sdr_module.gui`` imports this panel while it initialises, so importing
    ``sdr_module.gui.themes`` at module level here would be circular.
    """
    from ...gui import themes

    return themes


def default_save_directory() -> Path:
    """Where received images are auto-saved (created on first save)."""
    base = ""
    if HAS_PYQT6:
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        )
    root = Path(base) if base else Path.home()
    return root / "SDR Module" / "SSTV"


def mode_by_name(name: str) -> Optional[SSTVModeSpec]:
    """The mode called ``name`` (None for Auto or an unknown name)."""
    for spec in SSTV_MODES.values():
        if spec.name == name:
            return spec
    return None


def min_partial_lines(spec: Optional[SSTVModeSpec]) -> int:
    """Fewest lines worth keeping as a partial image (5 %, at least 8)."""
    height = spec.height if spec else 256
    return max(8, int(round(0.05 * height)))


def rgb_to_qimage(image: np.ndarray) -> "QImage":
    """Copy an ``(H, W, 3)`` RGB array into a standalone ``QImage``.

    A single-channel ``(H, W)`` array is copied into all three channels; an
    alpha channel, if any, is dropped.

    Raises:
        ValueError: for any other shape (QImage would otherwise read past
            the end of the array).
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB or (H, W) single-channel array, got "
            f"shape {image.shape}"
        )
    data = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    h, w = data.shape[:2]
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qimage_to_rgb(image: "QImage") -> Optional[np.ndarray]:
    """An ``(H, W, 3)`` uint8 copy of ``image`` (None for a null image)."""
    if image.isNull():
        return None
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h, stride = rgb.width(), rgb.height(), rgb.bytesPerLine()
    buffer = rgb.constBits()
    buffer.setsize(stride * h)
    data = np.frombuffer(buffer, dtype=np.uint8).reshape(h, stride)
    return data[:, : 3 * w].reshape(h, w, 3).copy()


def save_rgb_image(image: np.ndarray, path: str) -> bool:
    """Save an RGB array as PNG/JPEG (format from the file extension)."""
    return bool(rgb_to_qimage(image).save(str(path)))


def _format_timestamp(stamp: str) -> str:
    """``20260926_101530`` -> ``2026-09-26 10:15:30`` (unchanged if unknown)."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(stamp, "%Y%m%d_%H%M%S"))
    except (TypeError, ValueError):
        return str(stamp)


def history_label(info: Dict[str, Any], today: Optional[str] = None) -> str:
    """List text for an image: ``12:34 · Martin 1`` or, for a partial one,
    ``12:34 · Martin 1 (partial, 180/256 lines)``; the date is added for
    other days (``Oct 29 12:34 · ...``)."""
    stamp = str(info.get("timestamp", ""))
    try:
        when = time.strptime(stamp, "%Y%m%d_%H%M%S")
    except ValueError:
        when_text = stamp
    else:
        day = time.strftime("%Y%m%d", when)
        if day == (today or time.strftime("%Y%m%d")):
            when_text = time.strftime("%H:%M", when)
        else:
            when_text = time.strftime("%b %d %H:%M", when)
    text = f"{when_text} · {info.get('mode', 'SSTV')}"
    rows = info.get("partial_lines")
    if rows:
        text += f" (partial, {rows}/{info.get('height', '?')} lines)"
    return text


def parse_image_filename(path: Path) -> Optional[Dict[str, Any]]:
    """History entry for a saved image named like the auto-save does
    (``sstv_<YYYYmmdd_HHMMSS>_<Mode>[_partial_<rows>of<height>].png``)."""
    match = _FILE_RE.match(path.stem)
    if not match:
        return None
    stamp, mode_text, rows, height = match.groups()
    name = (mode_text or "SSTV").replace("_", " ")
    spec = mode_by_name(name)
    if spec is None:
        # "..._Robot_36_2.png": a second image saved in the same second.
        base = re.match(r"^(.*) \d+$", name)
        if base and mode_by_name(base.group(1)):
            name = base.group(1)
            spec = mode_by_name(name)
    return {
        "image": None,  # loaded when shown
        "mode": name,
        "timestamp": stamp,
        "width": spec.width if spec else 0,
        "height": int(height) if height else (spec.height if spec else 0),
        "filename": str(path),
        "partial_lines": int(rows) if rows else None,
    }


class ImageDisplayWidget(QWidget if HAS_PYQT6 else object):
    """Widget for displaying SSTV images (aspect-correct, theme-aware)."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._scaled: Optional[QPixmap] = None
        self._scaled_for: Optional[QSize] = None
        self._current_line: int = 0
        self._total_lines: int = 0
        self._placeholder_title = "No image yet"
        self._placeholder_hint = ISS_SSTV_HINT

        self.setMinimumSize(200, 120)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setToolTip("Received SSTV image")

    # Size: keep the 5:4 SSTV frame so the image fills the width.
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(self.minimumHeight(), int(round(width * _ASPECT)))

    def sizeHint(self) -> "QSize":
        return QSize(320, 256)

    def set_placeholder(self, title: str, hint: str = "") -> None:
        """Text shown while there is no image."""
        self._placeholder_title = title
        self._placeholder_hint = hint
        if self._pixmap is None:
            self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to display."""
        self._image = image
        self._current_line = 0
        self._total_lines = 0
        self._update_pixmap()
        self.update()

    def set_partial_image(
        self, image: np.ndarray, current_line: int, total_lines: int
    ) -> None:
        """Set partial image during reception."""
        self._image = image
        self._current_line = current_line
        self._total_lines = total_lines
        self._update_pixmap()
        self.update()

    def clear(self) -> None:
        """Clear the display."""
        self._image = None
        self._pixmap = None
        self._scaled = None
        self._current_line = 0
        self._total_lines = 0
        self.update()

    def _update_pixmap(self) -> None:
        """Update the QPixmap from numpy array."""
        self._scaled = None
        if self._image is None:
            self._pixmap = None
            return
        self._pixmap = QPixmap.fromImage(rgb_to_qimage(self._image))

    def resizeEvent(self, event) -> None:
        self._scaled = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        """Paint the widget."""
        p = _theme().get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(frame, 6, 6)
        painter.fillPath(path, p.qcolor("plot_bg"))
        painter.setPen(QPen(p.qcolor("lcd_border"), 1))
        painter.drawPath(path)

        inner = self.rect().adjusted(1, 1, -1, -1)
        if self._pixmap is not None:
            if self._scaled is None or self._scaled_for != inner.size():
                self._scaled_for = inner.size()
                self._scaled = self._pixmap.scaled(
                    inner.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            scaled = self._scaled
            x = inner.x() + (inner.width() - scaled.width()) // 2
            y = inner.y() + (inner.height() - scaled.height()) // 2
            painter.save()
            painter.setClipPath(path)
            painter.drawPixmap(x, y, scaled)
            painter.restore()

            # Scan line marker while an image is coming in.
            if 0 < self._current_line < self._total_lines:
                progress_y = y + int(
                    scaled.height() * self._current_line / self._total_lines
                )
                painter.setPen(QPen(p.qcolor("plot_marker"), 2))
                painter.drawLine(x, progress_y, x + scaled.width(), progress_y)
        else:
            self._paint_placeholder(painter, inner, p)

        painter.end()

    def _paint_placeholder(self, painter: "QPainter", rect, p) -> None:
        """Empty-state text in the same style as the spectrum and waterfall."""
        # Imported here: sdr_module.gui imports this module while it
        # initialises (see _theme()).
        from ...gui.spectrum_widget import draw_placeholder

        font = QFont(self.font())
        if font.pointSizeF() <= 0 < font.pixelSize():
            # draw_placeholder sizes its title in points (+1 pt, bold); the
            # stylesheet gives widgets a pixel-sized font, which would make
            # the title smaller than the hint.
            font.setPointSizeF(font.pixelSize() * 72.0 / max(1, self.logicalDpiY()))
        draw_placeholder(
            painter,
            QRectF(rect),
            self._placeholder_title,
            self._placeholder_hint,
            font,
            p,
        )


class ToneMeterWidget(QWidget if HAS_PYQT6 else object):
    """The dominant audio tone on a 1000-2400 Hz scale.

    Ticks mark the SSTV tones: 1200 Hz sync, 1500 Hz (darkest pixel),
    1900 Hz leader and 2300 Hz (brightest pixel); the picture range
    (1500-2300 Hz) is shaded. Colours come from the theme palette at paint
    time.
    """

    LOW_HZ = 1000.0
    HIGH_HZ = 2400.0
    TICKS = (1200, 1500, 1900, 2300)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._tone: Optional[float] = None
        self._active = False
        self.setMinimumSize(200, 30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Tone meter")
        self.setToolTip(
            "Strongest audio tone. 1200 Hz: sync pulses; 1500-2300 Hz: picture "
            "(dark to bright, shaded); 1900 Hz: the leader tone before each "
            "image."
        )

    def sizeHint(self) -> "QSize":
        return QSize(320, 32)

    def set_reading(self, tone_hz: Optional[float], active: bool = True) -> None:
        """Show ``tone_hz`` (None: no tone); ``active`` False dims the meter."""
        tone = None if tone_hz is None else float(tone_hz)
        if tone == self._tone and bool(active) == self._active:
            return
        self._tone = tone
        self._active = bool(active)
        self.update()

    def tone(self) -> Optional[float]:
        return self._tone

    def _x(self, hz: float, left: float, width: float) -> float:
        frac = (hz - self.LOW_HZ) / (self.HIGH_HZ - self.LOW_HZ)
        return left + max(0.0, min(1.0, frac)) * width

    def paintEvent(self, event) -> None:
        p = _theme().get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        fm = QFontMetricsF(font)
        label_w = fm.horizontalAdvance("2300")
        left = label_w / 2 + 2
        width = max(10.0, self.width() - 2 * left)
        bar_top, bar_h = 3.0, 10.0
        bar = QRectF(left, bar_top, width, bar_h)
        # Track and the picture range.
        painter.setPen(QPen(p.qcolor("surface2"), 1))
        painter.setBrush(p.qcolor("surface0"))
        painter.drawRoundedRect(bar, 3, 3)
        picture = QRectF(
            self._x(1500, left, width),
            bar_top,
            self._x(2300, left, width) - self._x(1500, left, width),
            bar_h,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor("accent", 60 if self._active else 30))
        painter.drawRect(picture)
        # Ticks and their labels.
        text_color = p.qcolor("plot_axis" if self._active else "disabled")
        for hz in self.TICKS:
            x = self._x(hz, left, width)
            painter.setPen(QPen(text_color, 1))
            painter.drawLine(QPointF(x, bar_top - 1), QPointF(x, bar_top + bar_h + 2))
            label = str(hz)
            lw = fm.horizontalAdvance(label)
            painter.drawText(
                QRectF(x - lw / 2 - 1, bar_top + bar_h + 2, lw + 2, fm.height()),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
        # Marker: the dominant tone.
        if self._active and self._tone is not None:
            x = self._x(self._tone, left, width)
            color = p.qcolor("plot_marker")
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(x, bar_top - 2), QPointF(x, bar_top + bar_h + 1))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4, bar_top - 3),
                        QPointF(x + 4, bar_top - 3),
                        QPointF(x, bar_top + 2),
                    ]
                )
            )
        painter.end()


class SSTVPanel(QWidget if HAS_PYQT6 else object):
    """
    SSTV receiver panel.

    Displays received SSTV images and provides controls for reception.
    Feed the demodulated audio (FM, or USB or LSB on HF) to
    :meth:`process_audio` while :meth:`is_receiving` is true.
    """

    if HAS_PYQT6:
        start_requested = pyqtSignal()
        stop_requested = pyqtSignal()
        image_saved = pyqtSignal(str)  # filepath
        # Tune to ISS (145.800 MHz): the window tunes, sets FM and starts
        # receiving.
        tune_iss_requested = pyqtSignal()
        # "Switch to FM" in the panel's FM hint.
        switch_to_fm_requested = pyqtSignal()
        # The Mode choice changed ("Auto (VIS)" or a mode name).
        mode_choice_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._decoder: Optional[SSTVDecoder] = None
        # SSTVImageViewer creates its folder up front; point it at an existing
        # directory so constructing the panel never litters the working
        # directory, then aim it at the real (lazily created) save folder.
        self._viewer = SSTVImageViewer(save_dir=tempfile.gettempdir())
        self._viewer.save_dir = default_save_directory()
        self._is_receiving = False
        # True while the viewer shows an image from the list (complete or
        # partial), not the one coming in.
        self._showing_complete = False
        # monotonic() time of the last audio block (or of pressing Start).
        self._last_audio_time = 0.0
        # Partial display refresh is throttled; this marks a pending one.
        self._display_dirty = False
        self._last_display_update = 0.0
        # Saved images are listed once the panel is first shown (or the
        # folder changes), so tests and other folders are not scanned.
        self._history_loaded_from: Optional[Path] = None
        # The scroll area's viewport, followed so the picture fits it.
        self._watched: Optional[QWidget] = None
        # The mode the hint's Switch button asks for (see show_fm_needed).
        self._switch_to = "FM"
        # Start Now on Auto (VIS) waiting for the line syncs to tell the
        # mode: the seconds of audio heard since (None: not waiting).
        self._start_now_wait: Optional[float] = None
        # False from Start Now until audio arrives (a Start Now with no
        # audio at all is said so, not "the audio stopped").
        self._start_now_heard = True
        # True while the status says the mode couldn't be told (replaced
        # once line syncs that tell it are heard).
        self._mode_unknown_shown = False

        self._setup_ui()

        # Progress, tone meter and partial-image polling while receiving.
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(100)
        self._update_timer.timeout.connect(self._update_status)

        self._set_listening_ui(False)
        self._refresh_history_ui()
        self._set_idle_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup UI elements."""
        t = _theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Control buttons first, so they are visible without scrolling.
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._start_btn = QPushButton("Start Decoder")
        t.set_role(self._start_btn, "primary")
        self._start_btn.clicked.connect(self._on_start_stop_clicked)
        btn_layout.addWidget(self._start_btn)

        self._tune_iss_btn = QPushButton("Tune to ISS (145.800 MHz)")
        self._tune_iss_btn.setToolTip(
            "Tune to the ISS SSTV downlink, 145.800 MHz FM with 5 kHz "
            "deviation, start receiving and listen for images"
        )
        self._tune_iss_btn.clicked.connect(self._on_tune_iss_clicked)
        btn_layout.addWidget(self._tune_iss_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row_label = QLabel("Mode")
        mode_row.addWidget(mode_row_label)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem(AUTO_MODE_TEXT)
        self._mode_combo.setItemData(
            0,
            "Each image's mode comes from the VIS code sent before it",
            Qt.ItemDataRole.ToolTipRole,
        )
        for name in MODE_ORDER:
            spec = mode_by_name(name)
            if spec is None:
                continue
            self._mode_combo.addItem(name)
            self._mode_combo.setItemData(
                self._mode_combo.count() - 1,
                f"Decode every image as {name} ({spec.width} × {spec.height}, "
                f"{spec.duration_s:.0f} s), also without a VIS",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._mode_combo.setToolTip(
            "Auto (VIS) reads each image's mode from the VIS code sent before "
            "it. Pick a mode when the VIS is missed or wrong: images are then "
            "decoded in that mode, and its line syncs alone start an image."
        )
        self._mode_combo.setAccessibleName("SSTV mode")
        mode_row_label.setBuddy(self._mode_combo)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)
        mode_row.addWidget(self._mode_combo)

        self._start_now_btn = QPushButton("Start Now")
        self._start_now_btn.setToolTip(
            "Decode the transmission that is already under way (the start "
            "tone was missed), from its line syncs. Uses the mode above, or "
            "with Auto the mode the sync pulses fit (it listens a few seconds "
            "for them)."
        )
        self._start_now_btn.clicked.connect(self._on_start_now_clicked)
        mode_row.addWidget(self._start_now_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Shown when the receiver is in a mode whose audio can't carry SSTV
        # (AM, CW, I/Q): SSTV is FM on VHF/UHF, USB or LSB on HF. (Or in the
        # other sideband than the band's.)
        self._fm_callout = QFrame()
        t.set_role(self._fm_callout, "callout", "warning")
        fm_row = QHBoxLayout(self._fm_callout)
        fm_row.setContentsMargins(8, 4, 4, 4)
        fm_row.setSpacing(8)
        self._fm_label = QLabel()
        self._fm_label.setWordWrap(True)
        t.set_role(self._fm_label, None, "warning")
        fm_row.addWidget(self._fm_label, 1)
        self._switch_fm_btn = QPushButton("Switch to FM")
        self._switch_fm_btn.setToolTip(
            "Set Demodulation > Mode to FM with 5 kHz deviation (narrowband)"
        )
        self._switch_fm_btn.clicked.connect(self._on_switch_fm_clicked)
        # A click leaves the keyboard focus where it was (Space stays the
        # window's Start/Stop); the button hides once it has done its job.
        self._switch_fm_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        fm_row.addWidget(self._switch_fm_btn)
        self._fm_callout.setVisible(False)
        layout.addWidget(self._fm_callout)

        # Shown while listening if no audio arrives (e.g. receiver stopped).
        self._no_audio_label = QLabel(NO_AUDIO_TEXT)
        self._no_audio_label.setWordWrap(True)
        t.set_role(self._no_audio_label, "callout", "warning")
        self._no_audio_label.setVisible(False)
        layout.addWidget(self._no_audio_label)

        # What the decoder hears.
        self._tone_meter = ToneMeterWidget()
        layout.addWidget(self._tone_meter)
        self._tone_label = QLabel()
        t.set_role(self._tone_label, "hint")
        self._tone_label.setToolTip("What the decoder hears right now")
        layout.addWidget(self._tone_label)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        t.set_role(self._status_label, "hint")
        layout.addWidget(self._status_label)

        # Image display
        self._image_display = ImageDisplayWidget()
        layout.addWidget(self._image_display)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        # The status line reads "line N of M", so the bar carries no text.
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setToolTip("Lines of the current image received")
        layout.addWidget(self._progress_bar)

        # Mode / size readout under the image.
        info_layout = QHBoxLayout()
        info_layout.setSpacing(6)
        mode_caption = QLabel("MODE")
        t.set_role(mode_caption, "caption")
        info_layout.addWidget(mode_caption)
        self._mode_label = QLabel(_NO_VALUE)
        t.set_role(self._mode_label, "value")
        self._mode_label.setToolTip("SSTV mode of the image shown")
        info_layout.addWidget(self._mode_label)
        info_layout.addSpacing(12)
        size_caption = QLabel("SIZE")
        t.set_role(size_caption, "caption")
        info_layout.addWidget(size_caption)
        self._resolution_label = QLabel(_NO_VALUE)
        t.set_role(self._resolution_label, "value")
        self._resolution_label.setToolTip("Image size in pixels")
        info_layout.addWidget(self._resolution_label)
        info_layout.addStretch(1)
        layout.addLayout(info_layout)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addStretch(1)
        self._save_btn = QPushButton("Save Image...")
        self._save_btn.clicked.connect(self._on_save_clicked)
        action_row.addWidget(self._save_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        # Room for either caption ("Discard Image" while one comes in).
        metrics = self._clear_btn.fontMetrics()
        self._clear_btn.setMinimumWidth(metrics.horizontalAdvance("Discard Image") + 32)
        action_row.addWidget(self._clear_btn)
        layout.addLayout(action_row)

        # Image history
        history_group = QGroupBox("Received Images")
        history_layout = QVBoxLayout(history_group)
        history_layout.setSpacing(8)

        self._history_empty = QLabel()
        self._history_empty.setWordWrap(True)
        self._history_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.set_role(self._history_empty, "placeholder")
        history_layout.addWidget(self._history_empty)

        self._history_list = QListWidget()
        self._history_list.setToolTip(
            "Select an image to view it; right-click to open, show or delete it"
        )
        self._history_list.currentRowChanged.connect(self._on_history_row_changed)
        self._history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._history_list.customContextMenuRequested.connect(self._on_history_menu)
        history_layout.addWidget(self._history_list)

        # History navigation
        self._nav_row = QWidget()
        nav_layout = QHBoxLayout(self._nav_row)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(8)

        self._prev_btn = QPushButton("‹ Previous")
        self._prev_btn.setToolTip("Show the previous received image")
        self._prev_btn.clicked.connect(self._on_prev_clicked)
        nav_layout.addWidget(self._prev_btn)

        self._image_count_label = QLabel("No images")
        self._image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.set_role(self._image_count_label, "muted")
        nav_layout.addWidget(self._image_count_label, 1)

        self._next_btn = QPushButton("Next ›")
        self._next_btn.setToolTip("Show the next received image")
        self._next_btn.clicked.connect(self._on_next_clicked)
        nav_layout.addWidget(self._next_btn)

        history_layout.addWidget(self._nav_row)

        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(8)
        self._auto_save_check = QCheckBox("Auto-save received images")
        self._auto_save_check.setChecked(True)
        self._auto_save_check.toggled.connect(self._update_history_empty_text)
        folder_layout.addWidget(self._auto_save_check, 1)
        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        folder_layout.addWidget(self._open_folder_btn)
        history_layout.addLayout(folder_layout)

        layout.addWidget(history_group)
        layout.addStretch(1)

        self._update_folder_tooltips()
        self._update_history_empty_text()

    # ------------------------------------------------------------------
    # Decoder wiring
    # ------------------------------------------------------------------

    def set_decoder(self, decoder: SSTVDecoder) -> None:
        """Set the SSTV decoder instance."""
        self._decoder = decoder

        # Set callbacks
        self._decoder.set_on_mode_detected(self._on_mode_detected)
        self._decoder.set_on_line_decoded(self._on_line_decoded)
        self._decoder.set_on_image_complete(self._on_image_complete)
        self._decoder.set_on_image_lost(self._on_image_lost)
        self._decoder.set_forced_mode(self.selected_mode())

    def create_decoder(self, sample_rate: float = 48000.0) -> SSTVDecoder:
        """Create and configure a new decoder."""
        self._decoder = SSTVDecoder(sample_rate=sample_rate)
        self.set_decoder(self._decoder)
        return self._decoder

    def is_receiving(self) -> bool:
        """True while the decoder is listening for (or receiving) an image."""
        return self._is_receiving

    def selected_mode(self) -> Optional[SSTVModeSpec]:
        """The mode chosen in the Mode list (None for Auto (VIS))."""
        return mode_by_name(self._mode_combo.currentText())

    def mode_choice(self) -> str:
        """The Mode list's text ("Auto (VIS)" or a mode name)."""
        return self._mode_combo.currentText()

    def set_mode_choice(self, text: str) -> bool:
        """Select "Auto (VIS)" or a mode by name; False if unknown."""
        index = self._mode_combo.findText(text)
        if index < 0:
            return False
        self._mode_combo.setCurrentIndex(index)
        return True

    def start_listening(self) -> None:
        """Start the decoder if it is not listening yet."""
        if not self._is_receiving:
            self._on_start_clicked()

    def end_image(self, why: str) -> Optional[str]:
        """End the image coming in (if any) now, e.g. because the app
        closes: the lines received so far are kept as a partial image in
        Received Images (and auto-saved), as when the decoder is stopped.
        ``why`` completes "The image ended: ...". The decoder keeps
        listening. Returns the status note (None if nothing was kept)."""
        return self._end_image_early(why)

    def show_fm_needed(
        self, mode: Optional[str], switch_to: str = "FM", band: str = ""
    ) -> None:
        """Say in the panel that the receiver's ``mode`` (AM, CW, I/Q) can't
        carry SSTV, offering to switch to ``switch_to``: "FM" (VHF/UHF, e.g.
        the ISS) or "USB"/"LSB" (HF). With ``band`` ("40 m"), ``mode`` is
        the other sideband than the one SSTV is usually sent in there. None
        hides the hint. A hint that appears is scrolled into view."""
        wanted = switch_to if switch_to in ("USB", "LSB") else "FM"
        self._switch_to = wanted
        callout = self._fm_callout
        before = (callout.isVisible(), self._fm_label.text())
        if mode:
            if wanted == "FM":
                text = (
                    "SSTV on VHF and UHF (such as the ISS) is sent as FM audio; "
                    f"the receiver is in {mode}."
                )
                tip = "Set Demodulation > Mode to FM with 5 kHz deviation (narrowband)"
            elif band:
                text = (
                    f"SSTV on {band} is usually sent in {wanted}; the receiver "
                    f"is in {mode}."
                )
                tip = f"Set Demodulation > Mode to {wanted} (2.7 kHz filter)"
            else:
                text = f"SSTV on HF is sent in {wanted}; the receiver is in {mode}."
                tip = f"Set Demodulation > Mode to {wanted} (2.7 kHz filter)"
            self._fm_label.setText(text)
            self._switch_fm_btn.setText(f"Switch to {wanted}")
            self._switch_fm_btn.setToolTip(tip)
        elif callout.isVisible():
            focus = QApplication.focusWidget()
            if focus is not None and callout.isAncestorOf(focus):
                # (Hidden with the focus, it would pass on to the next
                # control, Auto-save at the bottom, scrolling the tab there.)
                self._start_btn.setFocus(Qt.FocusReason.OtherFocusReason)
        callout.setVisible(bool(mode))
        if mode and before != (True, self._fm_label.text()):
            QTimer.singleShot(0, self._reveal_callout)

    def _reveal_callout(self) -> None:
        """Scroll the tab so the mode hint is in view."""
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        if area is None or area.widget() is not self or self._fm_callout.isHidden():
            return
        area.ensureWidgetVisible(self._fm_callout, 0, 8)

    def switch_mode(self) -> str:
        """The mode the hint's Switch button asks for: "FM", "USB" or "LSB"."""
        return self._switch_to

    def process_audio(
        self, samples: np.ndarray, sample_rate: Optional[float] = None
    ) -> None:
        """Process demodulated audio (FM, USB or LSB) through the decoder.

        Args:
            samples: Mono float audio.
            sample_rate: Audio rate in Hz. When given and different from the
                decoder's, the decoder is recreated at this rate.
        """
        if not self._is_receiving:
            return
        if samples is not None and len(samples):
            now = time.monotonic()
            gap = now - self._last_audio_time
            self._last_audio_time = now
            # isHidden(), not isVisible(): audio usually arrives while this
            # tab is not the one on screen.
            if not self._no_audio_label.isHidden():
                self._no_audio_label.setVisible(False)
            decoder = self._decoder
            if gap > AUDIO_GAP_S and not self._start_now_heard:
                self._start_now_without_audio()
            elif gap > AUDIO_GAP_S and decoder is not None and decoder.is_receiving():
                # After a pause the line timing is gone: keep what came in.
                self._end_image_early("the audio stopped for a while")
            self._start_now_heard = True
        if sample_rate and (
            self._decoder is None
            or abs(float(self._decoder.sample_rate) - float(sample_rate)) > 1.0
        ):
            self.create_decoder(float(sample_rate))
        if self._decoder:
            self._decoder.process_audio(samples)
            if self._start_now_wait is not None and samples is not None:
                self._start_now_wait += len(samples) / float(self._decoder.sample_rate)
                self._try_pending_start_now()

    # ------------------------------------------------------------------
    # Save location and history
    # ------------------------------------------------------------------

    def save_directory(self) -> Path:
        """Folder that received images are auto-saved to."""
        return Path(self._viewer.save_dir)

    def set_save_directory(self, path: str) -> None:
        """Change the auto-save folder (created when the first image is saved)."""
        self._viewer.save_dir = Path(path)
        self._update_folder_tooltips()
        if self._history_loaded_from is not None:
            self.load_history()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._history_loaded_from is None:
            self.load_history()
        parent = self.parentWidget()
        if parent is not self._watched:
            if self._watched is not None:
                self._watched.removeEventFilter(self)
            self._watched = parent
            if parent is not None:
                parent.installEventFilter(self)
        self._fit_image_to_view()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._watched and event.type() == QEvent.Type.Resize:
            self._fit_image_to_view()
        return super().eventFilter(obj, event)

    def _fit_image_to_view(self) -> None:
        """In a short tab (a scroll area's viewport), keep the whole picture
        in view with its status line: the image gets narrower, not cut."""
        parent = self.parentWidget()
        if parent is None or parent.objectName() != "qt_scrollarea_viewport":
            return
        # Room for a two-line status ("Image received. Saved as ...").
        status = 2 * self._status_label.fontMetrics().lineSpacing()
        room = parent.height() - status - 16
        limit = max(self._image_display.minimumHeight(), room)
        if self._image_display.maximumHeight() != limit:
            self._image_display.setMaximumHeight(limit)

    def load_history(self) -> int:
        """List the images saved in the save folder (newest
        ``_HISTORY_LOAD_LIMIT``), replacing any listed from another folder.
        Images received in this session stay. Returns how many were added."""
        folder = self.save_directory()
        self._history_loaded_from = folder
        session = [e for e in self._viewer.images if not e.get("from_disk")]
        known = {e.get("filename") for e in session if e.get("filename")}
        entries: List[Dict[str, Any]] = []
        try:
            files = sorted(folder.glob("sstv_*.png"), reverse=True)
        except OSError as e:  # pragma: no cover - permissions, etc.
            logger.debug(f"Cannot list {folder}: {e}")
            files = []
        for path in files:
            if len(entries) >= _HISTORY_LOAD_LIMIT:
                break
            if str(path) in known:
                continue
            entry = parse_image_filename(path)
            if entry is not None:
                entry["from_disk"] = True
                entries.append(entry)
        entries.reverse()  # oldest first, like the session's list
        current = self._viewer.get_image_info()
        self._viewer.images = entries + session
        if current is not None:
            names = [e.get("filename") for e in self._viewer.images]
            stamps = [e.get("timestamp") for e in self._viewer.images]
            if current.get("filename") in names and current.get("filename"):
                self._viewer.current_index = names.index(current["filename"])
            elif current.get("timestamp") in stamps:
                self._viewer.current_index = stamps.index(current["timestamp"])
            else:
                self._viewer.current_index = len(self._viewer.images) - 1
        else:
            self._viewer.current_index = len(self._viewer.images) - 1
        self._rebuild_history_list()
        return len(entries)

    def _rebuild_history_list(self) -> None:
        self._history_list.blockSignals(True)
        self._history_list.clear()
        for entry in self._viewer.images:
            item = QListWidgetItem(history_label(entry))
            if entry.get("filename"):
                item.setToolTip(entry["filename"])
            self._history_list.addItem(item)
        self._history_list.setCurrentRow(-1)
        self._history_list.blockSignals(False)
        self._refresh_history_ui()

    def _update_history_empty_text(self, *_args) -> None:
        """Empty-state copy for Received Images (honest about auto-save)."""
        if self._auto_save_check.isChecked():
            text = (
                "No images yet. Each decoded image is listed here and saved "
                "to the image folder."
            )
        else:
            text = (
                "No images yet. Each decoded image is listed here; use Save "
                "Image... to keep one."
            )
        self._history_empty.setText(text)

    def _update_folder_tooltips(self) -> None:
        folder = str(self.save_directory())
        self._auto_save_check.setToolTip(
            f"Save every decoded image (and partial ones) as PNG to:\n{folder}"
        )
        self._open_folder_btn.setToolTip(f"Open the image folder:\n{folder}")

    def _on_open_folder_clicked(self) -> None:
        folder = self.save_directory()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._set_status(f"Cannot create {folder}: {e}", "danger")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._set_status(f"Images are saved in {folder}", "info")

    def _on_history_menu(self, pos) -> None:
        item = self._history_list.itemAt(pos)
        if item is None:
            return
        row = self._history_list.row(item)
        entry = self._viewer.images[row] if row < len(self._viewer.images) else None
        if entry is None:
            return
        path = entry.get("filename")
        menu = QMenu(self)
        open_action = menu.addAction("Open")
        folder_action = menu.addAction("Show in Folder")
        menu.addSeparator()
        delete_action = menu.addAction("Delete File...")
        for action in (open_action, folder_action, delete_action):
            action.setEnabled(bool(path))
        if not path:
            open_action.setToolTip("This image is not saved: use Save Image...")
        chosen = menu.exec(self._history_list.viewport().mapToGlobal(pos))
        menu.deleteLater()
        if chosen is open_action and path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        elif chosen is folder_action and path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
        elif chosen is delete_action and path:
            self._delete_entry(row, confirm=True)

    def _delete_entry(self, row: int, confirm: bool = True) -> bool:
        """Delete a listed image's file and remove it from the list."""
        if not 0 <= row < len(self._viewer.images):
            return False
        entry = self._viewer.images[row]
        path = entry.get("filename")
        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete Image",
                f"Delete {Path(path).name if path else 'this image'} from the "
                "disk? This cannot be undone.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                self._set_status(f"Could not delete {Path(path).name}: {e}", "danger")
                return False
        showing = row == self._viewer.current_index and self._showing_complete
        del self._viewer.images[row]
        if self._viewer.current_index >= row:
            self._viewer.current_index -= 1
        if showing:
            self._image_display.clear()
            self._showing_complete = False
            self._mode_label.setText(_NO_VALUE)
            self._resolution_label.setText(_NO_VALUE)
        self._rebuild_history_list()
        self._set_status("Image deleted.", None)
        return True

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, tone: Optional[str] = None) -> None:
        """Show an activity message above the image (hidden when empty)."""
        self._mode_unknown_shown = False
        self._status_label.setText(text)
        self._status_label.setVisible(bool(text))
        _theme().set_tone(self._status_label, tone)

    def _set_idle_status(self) -> None:
        # The viewer's placeholder already explains what to do.
        self._set_status("")
        self._image_display.set_placeholder("No image yet", ISS_SSTV_HINT)

    def _set_listening_ui(self, listening: bool) -> None:
        t = _theme()
        # The bar appears once an image starts arriving (mode detected).
        self._progress_bar.setVisible(False)
        self._no_audio_label.setVisible(False)
        if listening:
            self._start_btn.setText("Stop Decoder")
            self._start_btn.setToolTip("Stop listening for SSTV images")
            t.set_role(self._start_btn, None)
            self._image_display.set_placeholder(
                "Listening…",
                "Waiting for an SSTV start tone (VIS header) in the audio.",
            )
        else:
            self._start_btn.setText("Start Decoder")
            self._start_btn.setToolTip("Listen for SSTV images in the receiver's audio")
            t.set_role(self._start_btn, "primary")
            self._image_display.set_placeholder("No image yet", ISS_SSTV_HINT)
            self._tone_meter.set_reading(None, active=False)
            self._tone_label.setText("Not listening")
        self._update_action_buttons()

    def _decoder_busy(self) -> bool:
        """True while an image is coming in."""
        return bool(self._decoder is not None and self._decoder.is_receiving())

    def _update_action_buttons(self) -> None:
        busy = self._is_receiving and self._decoder_busy()
        can_save = (
            not busy
            and self._showing_complete
            and 0 <= self._viewer.current_index < self._viewer.get_image_count()
        )
        self._save_btn.setEnabled(can_save)
        if can_save:
            tip = "Save the image shown above to a PNG or JPEG file"
        elif busy:
            tip = (
                "Available when the image is complete, or as a partial image "
                "once the decoder stops"
            )
        else:
            tip = "Available once an image has been received"
        self._save_btn.setToolTip(tip)
        if busy:
            self._clear_btn.setText("Discard Image")
            self._clear_btn.setEnabled(True)
            self._clear_btn.setToolTip(
                "Discard the image being received and listen for the next one"
            )
            return
        self._clear_btn.setText("Clear")
        can_clear = self._image_display.has_image() or self._progress_bar.value() > 0
        self._clear_btn.setEnabled(can_clear)
        self._clear_btn.setToolTip(
            "Clear the viewer (received images stay in the list)"
            if can_clear
            else "Nothing to clear"
        )

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start_stop_clicked(self) -> None:
        if self._is_receiving:
            self._on_stop_clicked()
        else:
            self._on_start_clicked()

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if self._decoder is None:
            self.create_decoder()

        self._decoder.reset()
        self._decoder.set_forced_mode(self.selected_mode())
        self._is_receiving = True
        self._last_audio_time = time.monotonic()
        self._start_now_wait = None
        self._start_now_heard = True
        self._update_timer.start()
        self._set_listening_ui(True)
        self._tone_label.setText("Listening…")
        # Before the first image the viewer placeholder says "Listening…".
        self._set_status(
            "Listening for an SSTV signal…" if self._image_display.has_image() else "",
            "success",
        )
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        kept = self._end_image_early(None)
        self._is_receiving = False
        self._start_now_wait = None
        self._start_now_heard = True
        self._update_timer.stop()
        self._set_listening_ui(False)
        self.show_fm_needed(None)
        if kept:
            self._set_status(f"Decoder stopped. {kept}", None)
        else:
            self._set_status("Decoder stopped.", None)
        self.stop_requested.emit()

    def _on_start_now_clicked(self) -> None:
        """Start decoding a transmission already under way."""
        if not self._is_receiving:
            self._on_start_clicked()
        decoder = self._decoder
        if decoder is None:  # pragma: no cover - created by Start
            return
        mode = self.selected_mode()
        if decoder.is_receiving():
            self._end_image_early("Start Now began a new one")
        self._start_now_heard = False  # (until audio arrives)
        self._start_now_wait = None
        if not decoder.start_now(mode):
            # Auto (VIS), and no line syncs heard yet (the decoder may only
            # have started listening): listen a few seconds for them.
            self._start_now_wait = 0.0
            self._set_status(START_NOW_WAIT_TEXT, "info")
            return
        self._say_starting()

    def _say_starting(self) -> None:
        spec = self._decoder.get_mode() if self._decoder is not None else None
        name = spec.name if spec else "SSTV"
        self._set_status(f"Starting {name} from the next line sync…", "success")

    def _try_pending_start_now(self) -> None:
        """Start Now on Auto (VIS) is waiting for the line syncs to tell the
        mode: start as soon as they do, or give up after
        START_NOW_AUTO_WAIT_S seconds of audio."""
        decoder = self._decoder
        if decoder is None or self._start_now_wait is None:
            return
        if decoder.is_receiving():  # (a VIS started an image meanwhile)
            self._start_now_wait = None
            return
        if decoder.start_now(self.selected_mode()):
            self._start_now_wait = None
            self._say_starting()
            return
        if self._start_now_wait >= START_NOW_AUTO_WAIT_S:
            self._start_now_wait = None
            self._set_status(START_NOW_NO_MODE_TEXT, "warning")
            self._mode_unknown_shown = True

    def _start_now_without_audio(self) -> None:
        """Start Now was pressed and no audio has come since: there is
        nothing to decode (not "the audio stopped for a while")."""
        self._start_now_heard = True
        self._start_now_wait = None
        decoder = self._decoder
        if decoder is not None and decoder.is_receiving():
            decoder.stop()  # (no audio, so no lines to keep)
            self._show_listening_viewer()
        self._set_status(START_NOW_NO_AUDIO_TEXT, "warning")
        self._update_action_buttons()

    def _on_tune_iss_clicked(self) -> None:
        self.tune_iss_requested.emit()

    def _on_switch_fm_clicked(self) -> None:
        self.switch_to_fm_requested.emit()

    def _on_mode_combo_changed(self, _index: int) -> None:
        if self._decoder is not None:
            self._decoder.set_forced_mode(self.selected_mode())
        self.mode_choice_changed.emit(self._mode_combo.currentText())

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        image = self._current_entry_image()
        if image is None:
            return

        info = self._viewer.get_image_info() or {}
        folder = self.save_directory()
        saved = info.get("filename")
        if not folder.is_dir() or (
            saved and Path(saved).parent.resolve() == folder.resolve()
        ):
            # Already in the image folder (auto-saved): suggest a copy
            # elsewhere, so Received Images doesn't list it twice later.
            pictures = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.PicturesLocation
            )
            folder = Path(pictures) if pictures else Path.home()
        # The auto-save's name: read back with its mode after a restart.
        default_path = str(folder / self._image_file_name(info))

        filepath, selected = QFileDialog.getSaveFileName(
            self,
            "Save SSTV Image",
            default_path,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not filepath:
            return
        if not Path(filepath).suffix:
            filepath += ".jpg" if selected.startswith("JPEG") else ".png"

        if save_rgb_image(image, filepath):
            self._set_status(f"Saved {Path(filepath).name}", "success")
            self.image_saved.emit(filepath)
        else:
            self._set_status(f"Could not save {Path(filepath).name}", "danger")

    def _on_clear_clicked(self) -> None:
        """Clear the viewer; while an image comes in, discard it instead."""
        if self._is_receiving and self._decoder_busy():
            self._discard_image()
            return
        self._image_display.clear()
        self._showing_complete = False
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._mode_label.setText(_NO_VALUE)
        self._resolution_label.setText(_NO_VALUE)
        self._history_list.blockSignals(True)
        self._history_list.setCurrentRow(-1)
        self._history_list.blockSignals(False)

        if self._is_receiving:
            self._set_status("")
            self._image_display.set_placeholder(
                "Listening…",
                "Waiting for an SSTV start tone (VIS header) in the audio.",
            )
        else:
            self._set_idle_status()
        self._refresh_history_ui()
        self._update_action_buttons()

    def _discard_image(self) -> None:
        """Drop the image being received and keep listening."""
        if self._decoder is not None:
            self._decoder.stop()
        self._show_listening_viewer()
        self._set_status("Image discarded. Listening for the next one…", None)
        self._refresh_history_ui()
        self._update_action_buttons()

    def _show_listening_viewer(self) -> None:
        """Empty the viewer of an image that ended without being kept (and
        drop its pending redraw): back to "Listening…"."""
        self._display_dirty = False
        self._image_display.clear()
        self._showing_complete = False
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._mode_label.setText(_NO_VALUE)
        self._resolution_label.setText(_NO_VALUE)
        self._image_display.set_placeholder(
            "Listening…", "Waiting for an SSTV start tone (VIS header) in the audio."
        )

    def _on_prev_clicked(self) -> None:
        """Show previous image."""
        if self._viewer.current_index > 0:
            self._viewer.current_index -= 1
            self._show_viewer_image()

    def _on_next_clicked(self) -> None:
        """Show next image."""
        if self._viewer.current_index < self._viewer.get_image_count() - 1:
            self._viewer.current_index += 1
            self._show_viewer_image()

    def _on_history_row_changed(self, index: int) -> None:
        if 0 <= index < self._viewer.get_image_count():
            self._viewer.current_index = index
            self._show_viewer_image()

    def _current_entry_image(self) -> Optional[np.ndarray]:
        """The selected list entry's image (loaded from its file if needed)."""
        index = self._viewer.current_index
        if not 0 <= index < self._viewer.get_image_count():
            return None
        entry = self._viewer.images[index]
        image = entry.get("image")
        if image is None and entry.get("filename"):
            image = qimage_to_rgb(QImage(str(entry["filename"])))
            if image is not None:
                entry["image"] = image
                entry["height"] = entry.get("height") or image.shape[0]
                entry["width"] = entry.get("width") or image.shape[1]
        # In memory: this image, the newest and those without a file.
        self._drop_saved_pixels(keep=entry)
        return image

    def _drop_saved_pixels(self, keep: Optional[Dict[str, Any]] = None) -> None:
        """Free the pixels of the listed images that have a file (they are
        read back from it when shown), except ``keep`` and the newest one.
        Images without a file (auto-save off, or the save failed) keep
        theirs: there is no other copy."""
        entries = self._viewer.images
        newest = entries[-1] if entries else None
        for entry in entries:
            if entry is not keep and entry is not newest and entry.get("filename"):
                entry["image"] = None

    def _show_viewer_image(self) -> None:
        """Display the viewer's current image and sync the history UI."""
        image = self._current_entry_image()
        if image is None:
            info = self._viewer.get_image_info() or {}
            name = Path(info.get("filename") or "").name or "This image"
            self._set_status(f"{name} could not be read.", "warning")
            return
        self._image_display.set_image(image)
        self._showing_complete = True
        self._update_image_info()
        self._refresh_history_ui()

    # ------------------------------------------------------------------
    # Decoder callbacks
    # ------------------------------------------------------------------

    def _on_mode_detected(self, mode: SSTVModeSpec) -> None:
        """Callback when SSTV mode is detected."""
        self._start_now_wait = None  # (an image began: no Start Now waits)
        self._mode_label.setText(mode.name)
        self._resolution_label.setText(f"{mode.width} × {mode.height}")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._showing_complete = False
        self._set_status(f"Receiving {mode.name}…", "success")
        self._update_action_buttons()
        QTimer.singleShot(0, self.reveal_image)

    def reveal_image(self) -> None:
        """In a short tab, scroll so the status line and the picture are in
        view (an image is arriving or just arrived)."""
        self._fit_image_to_view()
        QTimer.singleShot(0, self._scroll_to_image)

    def _scroll_to_image(self) -> None:
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        if area is None or area.widget() is not self:
            return
        top = self._status_label.mapTo(self, self._status_label.rect().topLeft()).y()
        bottom = self._image_display.mapTo(
            self, self._image_display.rect().bottomLeft()
        ).y()
        bar = area.verticalScrollBar()
        visible = (bar.value(), bar.value() + area.viewport().height())
        if top >= visible[0] and bottom <= visible[1]:
            return
        bar.setValue(max(0, top - 4))

    def _on_line_decoded(self, line: int, line_data: np.ndarray) -> None:
        """Callback for each decoded line (the picture is redrawn at most ten
        times a second)."""
        decoder = self._decoder
        if decoder is None or decoder.state.image_data is None:
            return
        self._display_dirty = True
        now = time.monotonic()
        if now - self._last_display_update >= 0.1:
            self._refresh_partial_display()

    def _refresh_partial_display(self) -> None:
        decoder = self._decoder
        self._display_dirty = False
        if decoder is None or decoder.state.image_data is None:
            return
        if not decoder.is_receiving():
            # The image ended (complete, kept as partial, discarded or lost)
            # after this redraw was put off: nothing is coming in any more.
            return
        self._last_display_update = time.monotonic()
        mode = decoder.state.mode
        total = mode.height if mode else 1
        line = decoder.state.current_line
        self._image_display.set_partial_image(decoder.state.image_data, line, total)
        self._progress_bar.setValue(int(100 * line / max(1, total)))
        self._progress_bar.setVisible(True)
        prefix = f"Receiving {mode.name}" if mode else "Receiving"
        self._set_status(f"{prefix}: line {line} of {total}…", "success")
        self._showing_complete = False
        self._update_action_buttons()

    def _add_to_history(
        self, image: np.ndarray, mode: SSTVModeSpec, partial_lines: Optional[int]
    ) -> str:
        """List (and auto-save) a received image; returns a note for the
        status line."""
        index = self._viewer.add_image(image, mode, auto_save=False)
        entry = self._viewer.images[index]
        entry["partial_lines"] = partial_lines
        note = ""
        if self._auto_save_check.isChecked():
            path = self._auto_save(index, image)
            note = (
                f" Saved as {path.name}."
                if path
                else " It could not be saved automatically."
            )
        # Only the newest saved image stays in memory: the others are read
        # back from their files (a long ISS event lists hundreds).
        self._drop_saved_pixels()
        item = QListWidgetItem(history_label(entry))
        if entry.get("filename"):
            item.setToolTip(entry["filename"])
        self._history_list.addItem(item)
        return note

    def _on_image_complete(self, image: np.ndarray) -> None:
        """Callback when image is complete."""
        mode = self._decoder.get_mode() if self._decoder else None
        saved_note, tone = "", "success"

        if mode:
            saved_note = self._add_to_history(image, mode, None)
            if "could not" in saved_note:
                tone = "warning"

        # Update display
        self._image_display.set_image(image)
        self._display_dirty = False
        self._showing_complete = mode is not None
        self._progress_bar.setValue(100)
        # Done: the status line says so; the bar returns with the next image.
        self._progress_bar.setVisible(False)
        self._refresh_history_ui()

        if self._is_receiving:
            # Keep listening: ISS passes send a new image every few minutes
            # (the decoder re-arms itself).
            self._set_status(
                f"Image received.{saved_note} Listening for the next one…", tone
            )
        else:
            self._set_status(f"Image received.{saved_note}", tone)
        self._update_action_buttons()
        QTimer.singleShot(0, self.reveal_image)

    def _on_image_lost(self, image: np.ndarray, lines: int) -> None:
        """The decoder ended an image early (signal faded, or a new image
        started)."""
        reason = self._decoder.lost_reason if self._decoder else ""
        cut_short = reason == "a new image started"
        mode = self._decoder.get_mode() if self._decoder else None
        note = self._keep_partial(image, lines)
        if note is None and not cut_short:
            self._show_listening_viewer()
        if cut_short:
            text = "A new image started before the last one ended."
            text += f" {note}" if note else ""
            self._set_status(text, "warning")
        elif reason == "no line syncs were heard":
            name = mode.name if mode else "SSTV"
            self._set_status(
                f"No {name} line syncs were heard, so nothing was decoded. "
                "Check the Mode, or wait for the next image's start tone.",
                "warning",
            )
        elif note is None:
            self._set_status(
                "The signal faded before the image got going. Listening for the "
                "next one…",
                "warning",
            )
        else:
            self._set_status(
                f"The signal faded. {note} Listening for the next one…", "warning"
            )
        self._update_action_buttons()

    def _keep_partial(self, image: np.ndarray, lines: int) -> Optional[str]:
        """List a partial image (if it has enough lines); returns the status
        note or None."""
        decoder = self._decoder
        mode = decoder.get_mode() if decoder else None
        self._progress_bar.setVisible(False)
        if mode is None or lines < min_partial_lines(mode):
            return None
        lines = min(int(lines), mode.height)
        saved = self._add_to_history(image, mode, lines)
        self._image_display.set_image(image)
        self._display_dirty = False
        self._showing_complete = True
        self._refresh_history_ui()
        return (
            f"The partial image ({lines} of {mode.height} lines) is in Received "
            f"Images.{saved}"
        )

    def _end_image_early(self, why: Optional[str]) -> Optional[str]:
        """Stop the image in progress (if any), keeping it as a partial
        image. Returns the status note (None if nothing was kept)."""
        decoder = self._decoder
        if decoder is None or not decoder.is_receiving():
            return None
        image = decoder.get_image()
        lines = decoder.stop()
        if image is None:
            return None
        note = self._keep_partial(image.copy(), lines)
        if why and note:
            self._set_status(f"The image ended: {why}. {note}", "warning")
        elif why and self._is_receiving:
            # Too few lines to keep: back to "Listening…", not a stale
            # "Receiving …: line N" for an image that is over.
            self._show_listening_viewer()
            self._set_status(
                f"The image ended: {why}. Listening for the next one…", "warning"
            )
        self._update_action_buttons()
        return note

    @staticmethod
    def _image_file_name(info: Dict[str, Any], copy: int = 1) -> str:
        """``sstv_<stamp>_<Mode>[_partial_<rows>of<height>][_<copy>].png``:
        the name Received Images reads the time and mode back from."""
        mode_name = str(info.get("mode", "sstv")).replace(" ", "_")
        stamp = info.get("timestamp") or time.strftime("%Y%m%d_%H%M%S")
        suffix = ""
        if info.get("partial_lines"):
            suffix = f"_partial_{info['partial_lines']}of{info.get('height', 0)}"
        if copy > 1:
            suffix += f"_{copy}"
        return f"sstv_{stamp}_{mode_name}{suffix}.png"

    def _auto_save(self, index: int, image: np.ndarray) -> Optional[Path]:
        """Save a received image into the save folder; return its path."""
        info = self._viewer.images[index] if index < len(self._viewer.images) else {}
        folder = self.save_directory()
        path = folder / self._image_file_name(info)
        n = 2
        while path.exists():  # two images in the same second
            path = folder / self._image_file_name(info, n)
            n += 1
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Cannot create SSTV folder {folder}: {e}")
            return None
        if not save_rgb_image(image, str(path)):
            logger.warning(f"Failed to auto-save SSTV image to {path}")
            return None
        self._viewer.images[index]["filename"] = str(path)
        logger.info(f"Auto-saved SSTV image: {path}")
        self.image_saved.emit(str(path))
        return path

    # ------------------------------------------------------------------
    # Periodic / derived UI state
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        """Periodic status update."""
        if not self._is_receiving:
            return
        silent = time.monotonic() - self._last_audio_time
        if silent >= NO_AUDIO_TIMEOUT_S and self._no_audio_label.isHidden():
            self._no_audio_label.setVisible(True)
        decoder = self._decoder
        if decoder is None:
            return
        waiting = decoder.is_receiving() or self._start_now_wait is not None
        if silent > AUDIO_GAP_S and waiting and not self._start_now_heard:
            # Start Now, and no audio has come since.
            self._start_now_without_audio()
        elif silent > AUDIO_GAP_S and decoder.is_receiving():
            # The audio stopped (the receiver stopped, or left the SSTV
            # modes): keep what came in now, not when audio comes back.
            self._end_image_early("the audio stopped for a while")
        if self._display_dirty:
            self._refresh_partial_display()
        status = decoder.get_status()
        if status["is_receiving"]:
            progress = int(status["progress"] * 100)
            self._progress_bar.setValue(progress)
        self._update_tone(silent)
        self._update_action_buttons()

    def _update_tone(self, silent: float) -> None:
        decoder = self._decoder
        if decoder is None:
            return
        if silent >= NO_AUDIO_TIMEOUT_S:
            self._tone_meter.set_reading(None, active=True)
            self._tone_label.setText("No audio")
            return
        tone = decoder.get_tone()
        self._tone_meter.set_reading(tone["tone_hz"], active=True)
        self._tone_label.setText(tone["text"])
        guess = tone.get("mode_guess")
        if self._mode_unknown_shown and guess is not None:
            # Start Now couldn't tell the mode; now the syncs do.
            self._set_status(
                f"{guess.name} line syncs heard: press Start Now to decode the "
                "image under way.",
                "info",
            )

    def _refresh_history_ui(self) -> None:
        """Sync the list, counter, empty state and navigation buttons."""
        count = self._viewer.get_image_count()
        index = self._viewer.current_index
        has_images = count > 0
        self._history_empty.setVisible(not has_images)
        self._history_list.setVisible(has_images)
        self._nav_row.setVisible(has_images)
        self._fit_history_list()
        if has_images and self._image_display.has_image() and self._showing_complete:
            self._image_count_label.setText(f"{index + 1} of {count}")
            self._history_list.blockSignals(True)
            self._history_list.setCurrentRow(index)
            self._history_list.blockSignals(False)
        elif has_images:
            self._image_count_label.setText(f"{count} image{'s' if count != 1 else ''}")
        else:
            self._image_count_label.setText("No images")
        self._update_history_buttons()

    def _fit_history_list(self) -> None:
        """Size the list to its rows (up to a few), so one or two images do
        not sit in a mostly empty box."""
        count = self._history_list.count()
        if count == 0:
            return
        rows = min(count, _HISTORY_MAX_ROWS)
        row_h = max(self._history_list.sizeHintForRow(0), 1)
        frame = 2 * self._history_list.frameWidth()
        self._history_list.setFixedHeight(rows * row_h + frame + 2)

    def _update_history_buttons(self) -> None:
        """Update prev/next button states."""
        self._prev_btn.setEnabled(self._viewer.current_index > 0)
        self._next_btn.setEnabled(
            self._viewer.current_index < self._viewer.get_image_count() - 1
        )

    def _update_image_info(self) -> None:
        """Update display with current image info."""
        info = self._viewer.get_image_info()
        if info:
            self._mode_label.setText(info.get("mode", _NO_VALUE))
            w, h = info.get("width", 0), info.get("height", 0)
            self._resolution_label.setText(f"{w} × {h}")
            self._progress_bar.setValue(100)
        self._update_action_buttons()


__all__ = [
    "ImageDisplayWidget",
    "SSTVPanel",
    "ToneMeterWidget",
]
