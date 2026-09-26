"""
SSTV Image Viewer panel for receiving images from the ISS and other sources.

Provides:
- Live image preview during reception
- Reception progress indicator
- Image history browser
- Auto-save functionality (PNG, via Qt; no extra imaging library needed)
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

try:
    from PyQt6.QtCore import QRectF, QSize, QStandardPaths, Qt, QTimer, QUrl, pyqtSignal
    from PyQt6.QtGui import (
        QDesktopServices,
        QFont,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..sstv import SSTVDecoder, SSTVImageViewer, SSTVModeSpec

logger = logging.getLogger(__name__)

# Most SSTV modes (Martin, Scottie, Robot 36) are 320 x 256: a 5:4 frame.
_ASPECT = 256.0 / 320.0
_NO_VALUE = "—"  # em dash
ISS_SSTV_HINT = "Tune to 145.800 MHz FM (ISS SSTV), then press Start Decoder."
# Seconds of silence while listening before the panel says no audio arrives.
NO_AUDIO_TIMEOUT_S = 3.0
NO_AUDIO_TEXT = (
    "⚠ No audio is reaching the decoder. Start the receiver in FM mode, "
    "tuned to the SSTV signal."
)
# Rows the Received Images list grows to before it scrolls.
_HISTORY_MAX_ROWS = 5


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


def rgb_to_qimage(image: np.ndarray) -> "QImage":
    """Copy an ``(H, W, 3)`` RGB array into a standalone ``QImage``."""
    data = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
    h, w = data.shape[:2]
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def save_rgb_image(image: np.ndarray, path: str) -> bool:
    """Save an RGB array as PNG/JPEG (format from the file extension)."""
    return bool(rgb_to_qimage(image).save(str(path)))


def _format_timestamp(stamp: str) -> str:
    """``20260926_101530`` -> ``2026-09-26 10:15:30`` (unchanged if unknown)."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(stamp, "%Y%m%d_%H%M%S"))
    except (TypeError, ValueError):
        return str(stamp)


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

        self.setMinimumSize(200, 160)
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
        text_rect = rect.adjusted(16, 12, -16, -12)
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPixelSize(13)
        hint_font = QFont(self.font())
        hint_font.setItalic(True)
        hint_font.setPixelSize(11)

        painter.setFont(title_font)
        title_h = painter.fontMetrics().height()
        painter.setFont(hint_font)
        flags = Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap
        hint_bounds = painter.fontMetrics().boundingRect(
            text_rect, int(flags), self._placeholder_hint
        )
        hint_h = hint_bounds.height() if self._placeholder_hint else 0
        gap = 6 if self._placeholder_hint else 0
        top = text_rect.y() + (text_rect.height() - (title_h + gap + hint_h)) // 2

        painter.setFont(title_font)
        painter.setPen(p.qcolor("plot_text"))
        painter.drawText(
            text_rect.x(),
            top,
            text_rect.width(),
            title_h,
            int(Qt.AlignmentFlag.AlignHCenter),
            self._placeholder_title,
        )
        if self._placeholder_hint:
            painter.setFont(hint_font)
            painter.setPen(p.qcolor("plot_axis"))
            painter.drawText(
                text_rect.x(),
                top + title_h + gap,
                text_rect.width(),
                hint_h,
                int(flags),
                self._placeholder_hint,
            )


class SSTVPanel(QWidget if HAS_PYQT6 else object):
    """
    SSTV receiver panel.

    Displays received SSTV images and provides controls for reception.
    Feed FM-demodulated audio to :meth:`process_audio` while
    :meth:`is_receiving` is true.
    """

    if HAS_PYQT6:
        start_requested = pyqtSignal()
        stop_requested = pyqtSignal()
        image_saved = pyqtSignal(str)  # filepath

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
        # True while the viewer shows a finished image (not a partial one).
        self._showing_complete = False
        # monotonic() time of the last audio block (or of pressing Start).
        self._last_audio_time = 0.0

        self._setup_ui()

        # Progress polling while receiving.
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

        self._save_btn = QPushButton("Save Image...")
        self._save_btn.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(self._save_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        btn_layout.addWidget(self._clear_btn)

        layout.addLayout(btn_layout)

        # Shown while listening if no audio arrives (e.g. receiver stopped).
        self._no_audio_label = QLabel(NO_AUDIO_TEXT)
        self._no_audio_label.setWordWrap(True)
        t.set_role(self._no_audio_label, "callout", "warning")
        self._no_audio_label.setVisible(False)
        layout.addWidget(self._no_audio_label)

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
        self._mode_label.setToolTip("SSTV mode detected from the VIS header")
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
        self._history_list.setToolTip("Select an image to view it")
        self._history_list.currentRowChanged.connect(self._on_history_row_changed)
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

    def create_decoder(self, sample_rate: float = 48000.0) -> SSTVDecoder:
        """Create and configure a new decoder."""
        self._decoder = SSTVDecoder(sample_rate=sample_rate)
        self.set_decoder(self._decoder)
        return self._decoder

    def is_receiving(self) -> bool:
        """True while the decoder is listening for (or receiving) an image."""
        return self._is_receiving

    def process_audio(
        self, samples: np.ndarray, sample_rate: Optional[float] = None
    ) -> None:
        """Process FM-demodulated audio samples through the decoder.

        Args:
            samples: Mono float audio.
            sample_rate: Audio rate in Hz. When given and different from the
                decoder's, the decoder is recreated at this rate.
        """
        if not self._is_receiving:
            return
        if samples is not None and len(samples):
            self._last_audio_time = time.monotonic()
            # isHidden(), not isVisible(): audio usually arrives while this
            # tab is not the one on screen.
            if not self._no_audio_label.isHidden():
                self._no_audio_label.setVisible(False)
        if sample_rate and (
            self._decoder is None
            or abs(float(self._decoder.sample_rate) - float(sample_rate)) > 1.0
        ):
            self.create_decoder(float(sample_rate))
        if self._decoder:
            self._decoder.process_audio(samples)

    # ------------------------------------------------------------------
    # Save location
    # ------------------------------------------------------------------

    def save_directory(self) -> Path:
        """Folder that received images are auto-saved to."""
        return Path(self._viewer.save_dir)

    def set_save_directory(self, path: str) -> None:
        """Change the auto-save folder (created when the first image is saved)."""
        self._viewer.save_dir = Path(path)
        self._update_folder_tooltips()

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
            f"Save every decoded image as PNG to:\n{folder}"
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

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, tone: Optional[str] = None) -> None:
        """Show an activity message above the image (hidden when empty)."""
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
                "Waiting for an SSTV start tone (VIS header) on the FM audio.",
            )
        else:
            self._start_btn.setText("Start Decoder")
            self._start_btn.setToolTip(
                "Listen for SSTV images on the demodulated FM audio"
            )
            t.set_role(self._start_btn, "primary")
            self._image_display.set_placeholder("No image yet", ISS_SSTV_HINT)
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        can_save = (
            self._showing_complete and self._viewer.get_current_image() is not None
        )
        self._save_btn.setEnabled(can_save)
        self._save_btn.setToolTip(
            "Save the image shown above to a PNG or JPEG file"
            if can_save
            else "Available once an image has been fully received"
        )
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
        self._is_receiving = True
        self._last_audio_time = time.monotonic()
        self._update_timer.start()
        self._set_listening_ui(True)
        # Before the first image the viewer placeholder says "Listening…".
        self._set_status(
            "Listening for an SSTV signal…" if self._image_display.has_image() else "",
            "success",
        )
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self._is_receiving = False
        self._update_timer.stop()
        self._set_listening_ui(False)
        self._set_status("Decoder stopped.", None)
        self.stop_requested.emit()

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        image = self._viewer.get_current_image()
        if image is None:
            return

        info = self._viewer.get_image_info() or {}
        stamp = info.get("timestamp") or time.strftime("%Y%m%d_%H%M%S")
        folder = self.save_directory()
        if not folder.is_dir():
            pictures = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.PicturesLocation
            )
            folder = Path(pictures) if pictures else Path.home()
        default_path = str(folder / f"sstv_{stamp}.png")

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
        """Handle clear button click."""
        self._image_display.clear()
        self._showing_complete = False
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._mode_label.setText(_NO_VALUE)
        self._resolution_label.setText(_NO_VALUE)
        self._history_list.blockSignals(True)
        self._history_list.setCurrentRow(-1)
        self._history_list.blockSignals(False)

        if self._decoder:
            self._decoder.reset()
        if self._is_receiving:
            self._set_status("")
        else:
            self._set_idle_status()
        self._refresh_history_ui()
        self._update_action_buttons()

    def _on_prev_clicked(self) -> None:
        """Show previous image."""
        if self._viewer.prev_image() is not None:
            self._show_viewer_image()

    def _on_next_clicked(self) -> None:
        """Show next image."""
        if self._viewer.next_image() is not None:
            self._show_viewer_image()

    def _on_history_row_changed(self, index: int) -> None:
        if 0 <= index < self._viewer.get_image_count():
            self._viewer.current_index = index
            self._show_viewer_image()

    def _show_viewer_image(self) -> None:
        """Display the viewer's current image and sync the history UI."""
        image = self._viewer.get_current_image()
        if image is None:
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
        self._mode_label.setText(mode.name)
        self._resolution_label.setText(f"{mode.width} × {mode.height}")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._set_status(f"Receiving {mode.name}…", "success")

    def _on_line_decoded(self, line: int, line_data: np.ndarray) -> None:
        """Callback for each decoded line."""
        if self._decoder and self._decoder.state.image_data is not None:
            total = self._decoder.state.mode.height if self._decoder.state.mode else 1
            self._image_display.set_partial_image(
                self._decoder.state.image_data, line, total
            )
            self._progress_bar.setValue(int(100 * line / max(1, total)))
            self._progress_bar.setVisible(True)
            mode = self._decoder.state.mode
            prefix = f"Receiving {mode.name}" if mode else "Receiving"
            self._set_status(f"{prefix}: line {line} of {total}…", "success")
            self._showing_complete = False
            self._update_action_buttons()

    def _on_image_complete(self, image: np.ndarray) -> None:
        """Callback when image is complete."""
        mode = self._decoder.get_mode() if self._decoder else None
        saved_note, tone = "", "success"

        if mode:
            index = self._viewer.add_image(image, mode, auto_save=False)
            if self._auto_save_check.isChecked():
                path = self._auto_save(index, image)
                if path:
                    saved_note = f" Saved as {path.name}."
                else:
                    saved_note = " It could not be saved automatically."
                    tone = "warning"
            info = self._viewer.get_image_info()
            if info:
                item = QListWidgetItem(
                    f"{_format_timestamp(info['timestamp'])} · {info['mode']}"
                )
                if info.get("filename"):
                    item.setToolTip(info["filename"])
                self._history_list.addItem(item)

        # Update display
        self._image_display.set_image(image)
        self._showing_complete = mode is not None
        self._progress_bar.setValue(100)
        # Done: the status line says so; the bar returns with the next image.
        self._progress_bar.setVisible(False)
        self._refresh_history_ui()

        if self._is_receiving:
            # Keep listening: ISS passes send a new image every few minutes.
            QTimer.singleShot(0, self._rearm_decoder)
            self._set_status(
                f"Image received.{saved_note} Listening for the next one…", tone
            )
        else:
            self._set_status(f"Image received.{saved_note}", tone)
        self._update_action_buttons()

    def _rearm_decoder(self) -> None:
        if self._is_receiving and self._decoder is not None:
            self._decoder.reset()

    def _auto_save(self, index: int, image: np.ndarray) -> Optional[Path]:
        """Save a received image into the save folder; return its path."""
        info = self._viewer.get_image_info(index) or {}
        mode_name = str(info.get("mode", "sstv")).replace(" ", "_")
        stamp = info.get("timestamp") or time.strftime("%Y%m%d_%H%M%S")
        folder = self.save_directory()
        path = folder / f"sstv_{stamp}_{mode_name}.png"
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
        if self._decoder:
            status = self._decoder.get_status()

            if status["is_receiving"]:
                progress = int(status["progress"] * 100)
                self._progress_bar.setValue(progress)

    def _refresh_history_ui(self) -> None:
        """Sync the list, counter, empty state and navigation buttons."""
        count = self._viewer.get_image_count()
        index = self._viewer.current_index
        has_images = count > 0
        self._history_empty.setVisible(not has_images)
        self._history_list.setVisible(has_images)
        self._nav_row.setVisible(has_images)
        self._fit_history_list()
        if has_images and self._image_display.has_image():
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
]
