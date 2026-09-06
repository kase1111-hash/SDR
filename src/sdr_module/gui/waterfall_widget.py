"""
Waterfall display widget.

Provides scrolling time-frequency visualization with:
- Configurable color maps
- Protocol highlighting
- Zoom and pan support
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen
    from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class WaterfallWidget(QWidget if HAS_PYQT6 else object):
    """
    Waterfall display widget.

    Shows scrolling spectrogram with time on Y-axis and frequency on X-axis.
    """

    # Color map definitions
    COLORMAPS = {
        "viridis": [
            (68, 1, 84),
            (72, 35, 116),
            (64, 67, 135),
            (52, 94, 141),
            (41, 120, 142),
            (32, 144, 140),
            (34, 167, 132),
            (68, 190, 112),
            (121, 209, 81),
            (189, 222, 38),
            (253, 231, 37),
        ],
        "plasma": [
            (13, 8, 135),
            (75, 3, 161),
            (125, 3, 168),
            (168, 34, 150),
            (203, 70, 121),
            (229, 107, 93),
            (248, 148, 65),
            (253, 195, 40),
            (240, 249, 33),
        ],
        "turbo": [
            (48, 18, 59),
            (86, 36, 163),
            (75, 107, 221),
            (42, 171, 226),
            (29, 223, 163),
            (109, 248, 101),
            (205, 233, 55),
            (252, 186, 47),
            (252, 108, 42),
            (210, 38, 39),
            (122, 4, 3),
        ],
        "grayscale": [
            (0, 0, 0),
            (28, 28, 28),
            (56, 56, 56),
            (85, 85, 85),
            (113, 113, 113),
            (141, 141, 141),
            (170, 170, 170),
            (198, 198, 198),
            (226, 226, 226),
            (255, 255, 255),
        ],
        "classic": [
            (0, 0, 50),
            (0, 0, 100),
            (0, 50, 150),
            (0, 100, 200),
            (0, 200, 200),
            (0, 200, 100),
            (100, 200, 0),
            (200, 200, 0),
            (255, 150, 0),
            (255, 50, 0),
            (255, 0, 0),
        ],
    }

    if HAS_PYQT6:
        frequency_clicked = pyqtSignal(float)  # Hz

    def __init__(self, parent=None, history_size: int = 500):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._history_size = history_size
        self._fft_size = 2048
        self._db_range = (-100, 0)
        self._center_freq = 100e6
        self._sample_rate = 2.4e6

        # Data storage
        self._history: deque = deque(maxlen=history_size)

        # Color map
        self._colormap_name = "turbo"
        self._colormap = self._build_colormap(self._colormap_name)

        # Image buffer. ``_image_rgb`` is the contiguous uint8 (H, W, 3) array
        # that ``_image`` (a QImage view) points at; we must keep a reference to
        # it alive for as long as the QImage exists, since QImage does not copy
        # the buffer it is constructed from.
        self._image: Optional[QImage] = None
        self._image_rgb: Optional[np.ndarray] = None

        # Highlights
        self._highlights: List[Tuple[int, int, int, int, QColor]] = []

        # Colors — matched to Catppuccin Mocha theme
        self._bg_color = QColor(17, 17, 27)  # #11111b crust
        self._text_color = QColor(186, 194, 222)  # #bac2de subtext1

        self.setMinimumHeight(200)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Controls bar
        controls_widget = QWidget()
        controls_widget.setFixedHeight(28)
        controls_widget.setStyleSheet(
            "background-color: #181825; border-bottom: 1px solid #313244;"
        )
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(8, 2, 8, 2)
        controls.setSpacing(6)

        title = QLabel("WATERFALL")
        title.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold;"
            "background: transparent;"
        )
        controls.addWidget(title)

        controls.addStretch()

        color_label = QLabel("COLOR")
        color_label.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold;"
            "background: transparent;"
        )
        controls.addWidget(color_label)
        self._color_combo = QComboBox()
        self._color_combo.addItems(list(self.COLORMAPS.keys()))
        self._color_combo.setCurrentText(self._colormap_name)
        self._color_combo.setFixedWidth(90)
        self._color_combo.currentTextChanged.connect(self._on_colormap_changed)
        controls.addWidget(self._color_combo)

        range_label = QLabel("RANGE")
        range_label.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold;"
            "background: transparent;"
        )
        controls.addWidget(range_label)
        self._range_combo = QComboBox()
        self._range_combo.addItems(["60 dB", "80 dB", "100 dB", "120 dB"])
        self._range_combo.setCurrentIndex(2)
        self._range_combo.setFixedWidth(70)
        self._range_combo.currentIndexChanged.connect(self._on_range_changed)
        controls.addWidget(self._range_combo)

        layout.addWidget(controls_widget)

    def _on_colormap_changed(self, name: str):
        """Handle colormap change."""
        self._colormap_name = name
        self._colormap = self._build_colormap(name)
        self._render_image()
        self.update()

    def _on_range_changed(self, index: int):
        """Handle range change."""
        ranges = [60, 80, 100, 120]
        self._db_range = (-ranges[index], 0)
        self._render_image()
        self.update()

    def _build_colormap(self, name: str) -> np.ndarray:
        """Build 256-entry colormap from definition."""
        if name not in self.COLORMAPS:
            name = "turbo"

        colors = self.COLORMAPS[name]
        n_colors = len(colors)

        # Interpolate to 256 entries
        colormap = np.zeros((256, 3), dtype=np.uint8)

        for i in range(256):
            # Find surrounding colors
            pos = i * (n_colors - 1) / 255
            idx = int(pos)
            frac = pos - idx

            if idx >= n_colors - 1:
                colormap[i] = colors[-1]
            else:
                c1 = np.array(colors[idx])
                c2 = np.array(colors[idx + 1])
                colormap[i] = (c1 * (1 - frac) + c2 * frac).astype(np.uint8)

        return colormap

    def add_line(self, power_db: np.ndarray):
        """
        Add a new spectrum line to the waterfall.

        Args:
            power_db: Power spectrum in dB
        """
        if len(power_db) == 0:
            # No spectrum data: record a neutral (min-dB) row so history stays
            # in sync with caller cadence but we skip the expensive interp path.
            power_db = np.full(self._fft_size, self._db_range[0], dtype=np.float32)
        elif len(power_db) != self._fft_size:
            # Resample if needed
            power_db = np.interp(
                np.linspace(0, 1, self._fft_size),
                np.linspace(0, 1, len(power_db)),
                power_db,
            )

        self._history.append(power_db.copy())
        self._render_image()
        self.update()

    def clear(self):
        """Clear the waterfall."""
        self._history.clear()
        self._image = None
        self._image_rgb = None
        self.update()

    def set_center_freq(self, center_freq: float) -> None:
        """Update the center frequency used for click-to-tune and labels."""
        self._center_freq = center_freq
        self.update()

    def set_sample_rate(self, sample_rate: float) -> None:
        """Update the sample rate used for click-to-tune and labels."""
        self._sample_rate = sample_rate
        self.update()

    def save_image(self, path: str) -> bool:
        """Save the current waterfall image to a file. Returns success."""
        if self._image is None:
            return False
        return bool(self._image.save(path))

    def mousePressEvent(self, event):
        """Emit frequency_clicked on left-click inside the image area."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        margin = 50  # matches _draw code; ~left axis margin
        right_pad = 10
        w = self.width() - margin - right_pad
        if w <= 0:
            return
        x = event.position().x() - margin
        if x < 0 or x > w:
            return
        frac = x / w
        freq_start = self._center_freq - self._sample_rate / 2
        freq = freq_start + frac * self._sample_rate
        self.frequency_clicked.emit(float(freq))

    def add_highlight(
        self,
        time_start: int,
        time_end: int,
        freq_start: int,
        freq_end: int,
        color: QColor,
    ):
        """Add a highlight region."""
        self._highlights.append((time_start, time_end, freq_start, freq_end, color))
        self.update()

    def clear_highlights(self):
        """Clear all highlights."""
        self._highlights.clear()
        self.update()

    def _render_image(self):
        """Rebuild the whole image buffer from history, vectorized.

        The waterfall is ``history_size`` rows by ``fft_size`` columns, with the
        newest line at the bottom. Rendering this per-pixel in Python (a nested
        ``QImage.pixel``/``setPixel`` scroll over ~1M pixels) took ~0.4 s per
        line, far above the ~33 ms display cadence, so acquisition froze the UI.

        Instead we map every history line to colormap indices and gather the RGB
        rows with NumPy in one pass, then wrap the resulting contiguous
        ``(H, W, 3)`` uint8 array in a QImage. The array is kept alive on
        ``self._image_rgb`` because QImage references the buffer without copying.
        """
        if len(self._history) == 0:
            self._image = None
            self._image_rgb = None
            return

        min_db, max_db = self._db_range
        db_range = max_db - min_db
        if db_range <= 0:
            db_range = 1.0

        # Background-filled buffer; history fills the bottom rows.
        buf = np.empty((self._history_size, self._fft_size, 3), dtype=np.uint8)
        buf[:, :, 0] = self._bg_color.red()
        buf[:, :, 1] = self._bg_color.green()
        buf[:, :, 2] = self._bg_color.blue()

        # Stack history into (n, fft_size); add_line guarantees each row is
        # already fft_size long.
        lines = np.stack(self._history).astype(np.float32, copy=False)
        normalized = np.clip((lines - min_db) / db_range, 0.0, 1.0)
        idx = (normalized * 255.0).astype(np.intp)  # (n, fft_size)
        rgb = self._colormap[idx]  # (n, fft_size, 3) uint8

        n = rgb.shape[0]
        buf[self._history_size - n :] = rgb

        # Keep the buffer alive: QImage does not copy it.
        self._image_rgb = np.ascontiguousarray(buf)
        self._image = QImage(
            self._image_rgb.data,
            self._fft_size,
            self._history_size,
            3 * self._fft_size,
            QImage.Format.Format_RGB888,
        )

    def paintEvent(self, event):
        """Paint the waterfall display."""
        painter = QPainter(self)

        width = self.width()
        height = self.height() - 25  # Leave room for controls
        margin = 50

        # Draw background
        painter.fillRect(0, 0, width, height, self._bg_color)

        # Draw waterfall image
        if self._image:
            target_rect = QRectF(margin, 0, width - margin - 10, height - 20)
            source_rect = QRectF(0, 0, self._image.width(), self._image.height())
            painter.drawImage(target_rect, self._image, source_rect)

        # Draw highlights
        for time_start, time_end, freq_start, freq_end, color in self._highlights:
            x1 = margin + freq_start * (width - margin - 10) // self._fft_size
            x2 = margin + freq_end * (width - margin - 10) // self._fft_size
            y1 = time_start * (height - 20) // self._history_size
            y2 = time_end * (height - 20) // self._history_size

            painter.setPen(QPen(color, 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

        # Draw time axis label
        painter.setPen(self._text_color)
        painter.drawText(5, height // 2, "Time")
