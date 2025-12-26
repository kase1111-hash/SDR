"""
SSTV Image Viewer panel for receiving images from the ISS and other sources.

Provides:
- Live image preview during reception
- Reception progress indicator
- Image history browser
- Auto-save functionality
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import time

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QProgressBar, QGroupBox,
        QListWidget, QListWidgetItem, QFileDialog, QSizePolicy
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..dsp.sstv import SSTVDecoder, SSTVImageViewer, SSTVModeSpec


class ImageDisplayWidget(QWidget if HAS_PYQT6 else object):
    """Widget for displaying SSTV images."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._current_line: int = 0
        self._total_lines: int = 0

        self.setMinimumSize(320, 256)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to display."""
        self._image = image
        self._update_pixmap()
        self.update()

    def set_partial_image(self, image: np.ndarray, current_line: int, total_lines: int) -> None:
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
        self._current_line = 0
        self._total_lines = 0
        self.update()

    def _update_pixmap(self) -> None:
        """Update the QPixmap from numpy array."""
        if self._image is None:
            self._pixmap = None
            return

        h, w, c = self._image.shape
        bytes_per_line = 3 * w

        # Ensure contiguous array
        img_data = np.ascontiguousarray(self._image)

        qimage = QImage(
            img_data.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self._pixmap = QPixmap.fromImage(qimage)

    def paintEvent(self, event) -> None:
        """Paint the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap:
            # Scale to fit while maintaining aspect ratio
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Center the image
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            # Draw progress line during reception
            if self._current_line > 0 and self._current_line < self._total_lines:
                progress_y = y + int(scaled.height() * self._current_line / self._total_lines)
                painter.setPen(QColor(0, 255, 0))
                painter.drawLine(x, progress_y, x + scaled.width(), progress_y)
        else:
            # Draw placeholder text
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "SSTV Image Viewer\n\nTune to 145.800 MHz\nto receive ISS images"
            )

        painter.end()


class SSTVPanel(QWidget if HAS_PYQT6 else object):
    """
    SSTV receiver panel.

    Displays received SSTV images and provides controls for reception.
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
        self._viewer = SSTVImageViewer()
        self._is_receiving = False

        self._setup_ui()

        # Update timer for progress
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_status)
        self._update_timer.start(100)

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)

        # Image display
        self._image_display = ImageDisplayWidget()
        layout.addWidget(self._image_display, stretch=1)

        # Status group
        status_group = QGroupBox("Reception Status")
        status_layout = QGridLayout(status_group)

        # Mode label
        status_layout.addWidget(QLabel("Mode:"), 0, 0)
        self._mode_label = QLabel("--")
        self._mode_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self._mode_label, 0, 1)

        # Resolution label
        status_layout.addWidget(QLabel("Resolution:"), 0, 2)
        self._resolution_label = QLabel("--")
        status_layout.addWidget(self._resolution_label, 0, 3)

        # Progress bar
        status_layout.addWidget(QLabel("Progress:"), 1, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        status_layout.addWidget(self._progress_bar, 1, 1, 1, 3)

        # Status text
        self._status_label = QLabel("Ready - Tune to ISS SSTV frequency (145.800 MHz)")
        self._status_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self._status_label, 2, 0, 1, 4)

        layout.addWidget(status_group)

        # Control buttons
        btn_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start Decoder")
        self._start_btn.clicked.connect(self._on_start_clicked)
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)
        btn_layout.addWidget(self._stop_btn)

        self._save_btn = QPushButton("Save Image")
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._save_btn.setEnabled(False)
        btn_layout.addWidget(self._save_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        btn_layout.addWidget(self._clear_btn)

        layout.addLayout(btn_layout)

        # Image history
        history_group = QGroupBox("Received Images")
        history_layout = QVBoxLayout(history_group)

        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(100)
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        history_layout.addWidget(self._history_list)

        # History navigation
        nav_layout = QHBoxLayout()

        self._prev_btn = QPushButton("< Previous")
        self._prev_btn.clicked.connect(self._on_prev_clicked)
        self._prev_btn.setEnabled(False)
        nav_layout.addWidget(self._prev_btn)

        self._image_count_label = QLabel("0 images")
        self._image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._image_count_label)

        self._next_btn = QPushButton("Next >")
        self._next_btn.clicked.connect(self._on_next_clicked)
        self._next_btn.setEnabled(False)
        nav_layout.addWidget(self._next_btn)

        history_layout.addLayout(nav_layout)

        layout.addWidget(history_group)

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

    def process_audio(self, samples: np.ndarray) -> None:
        """Process audio samples through the decoder."""
        if self._decoder and self._is_receiving:
            self._decoder.process_audio(samples)

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if self._decoder is None:
            self.create_decoder()

        self._decoder.reset()
        self._is_receiving = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Listening for SSTV signal...")
        self._status_label.setStyleSheet("color: #4a4;")
        self.start_requested.emit()

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        self._is_receiving = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Stopped")
        self._status_label.setStyleSheet("color: #888;")
        self.stop_requested.emit()

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        image = self._viewer.get_current_image()
        if image is None:
            return

        # Get save path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"sstv_{timestamp}.png"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save SSTV Image",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )

        if filepath:
            try:
                from PIL import Image
                img = Image.fromarray(image, 'RGB')
                img.save(filepath)
                self._status_label.setText(f"Saved: {Path(filepath).name}")
                self.image_saved.emit(filepath)
            except ImportError:
                # Fallback
                np.save(filepath + ".npy", image)
                self._status_label.setText(f"Saved as numpy: {Path(filepath).name}.npy")

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self._image_display.clear()
        self._progress_bar.setValue(0)
        self._mode_label.setText("--")
        self._resolution_label.setText("--")
        self._status_label.setText("Ready")
        self._status_label.setStyleSheet("color: #888;")
        self._save_btn.setEnabled(False)

        if self._decoder:
            self._decoder.reset()

    def _on_prev_clicked(self) -> None:
        """Show previous image."""
        image = self._viewer.prev_image()
        if image is not None:
            self._image_display.set_image(image)
            self._update_history_buttons()
            self._update_image_info()

    def _on_next_clicked(self) -> None:
        """Show next image."""
        image = self._viewer.next_image()
        if image is not None:
            self._image_display.set_image(image)
            self._update_history_buttons()
            self._update_image_info()

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle history item click."""
        index = self._history_list.row(item)
        if 0 <= index < self._viewer.get_image_count():
            self._viewer.current_index = index
            image = self._viewer.get_current_image()
            if image is not None:
                self._image_display.set_image(image)
                self._update_history_buttons()
                self._update_image_info()

    def _on_mode_detected(self, mode: SSTVModeSpec) -> None:
        """Callback when SSTV mode is detected."""
        self._mode_label.setText(mode.name)
        self._resolution_label.setText(f"{mode.width}x{mode.height}")
        self._status_label.setText(f"Receiving {mode.name}...")
        self._status_label.setStyleSheet("color: #4a4;")

    def _on_line_decoded(self, line: int, line_data: np.ndarray) -> None:
        """Callback for each decoded line."""
        if self._decoder and self._decoder.state.image_data is not None:
            total = self._decoder.state.mode.height if self._decoder.state.mode else 1
            self._image_display.set_partial_image(
                self._decoder.state.image_data,
                line,
                total
            )
            self._progress_bar.setValue(int(100 * line / total))

    def _on_image_complete(self, image: np.ndarray) -> None:
        """Callback when image is complete."""
        mode = self._decoder.get_mode() if self._decoder else None

        # Add to viewer
        if mode:
            self._viewer.add_image(image, mode, auto_save=True)

            # Update history list
            info = self._viewer.get_image_info()
            if info:
                item_text = f"{info['timestamp']} - {info['mode']}"
                self._history_list.addItem(item_text)

        # Update display
        self._image_display.set_image(image)
        self._progress_bar.setValue(100)
        self._status_label.setText("Image received!")
        self._status_label.setStyleSheet("color: #4a4; font-weight: bold;")
        self._save_btn.setEnabled(True)
        self._update_history_buttons()

        # Stop receiving
        self._is_receiving = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _update_status(self) -> None:
        """Periodic status update."""
        if self._decoder and self._is_receiving:
            status = self._decoder.get_status()

            if status["is_receiving"]:
                progress = int(status["progress"] * 100)
                self._progress_bar.setValue(progress)

        # Update image count
        count = self._viewer.get_image_count()
        self._image_count_label.setText(f"{count} image{'s' if count != 1 else ''}")

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
            self._mode_label.setText(info.get("mode", "--"))
            w, h = info.get("width", 0), info.get("height", 0)
            self._resolution_label.setText(f"{w}x{h}")


__all__ = [
    'ImageDisplayWidget',
    'SSTVPanel',
]
