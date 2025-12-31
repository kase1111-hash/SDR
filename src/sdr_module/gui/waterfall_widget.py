"""
Waterfall display widget.

Provides scrolling time-frequency visualization with:
- Configurable color maps
- Protocol highlighting
- Zoom and pan support
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from collections import deque

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QPainter, QImage, QColor, QPen
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
            (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141),
            (41, 120, 142), (32, 144, 140), (34, 167, 132), (68, 190, 112),
            (121, 209, 81), (189, 222, 38), (253, 231, 37)
        ],
        "plasma": [
            (13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150),
            (203, 70, 121), (229, 107, 93), (248, 148, 65), (253, 195, 40),
            (240, 249, 33)
        ],
        "turbo": [
            (48, 18, 59), (86, 36, 163), (75, 107, 221), (42, 171, 226),
            (29, 223, 163), (109, 248, 101), (205, 233, 55), (252, 186, 47),
            (252, 108, 42), (210, 38, 39), (122, 4, 3)
        ],
        "grayscale": [
            (0, 0, 0), (28, 28, 28), (56, 56, 56), (85, 85, 85),
            (113, 113, 113), (141, 141, 141), (170, 170, 170), (198, 198, 198),
            (226, 226, 226), (255, 255, 255)
        ],
        "classic": [
            (0, 0, 50), (0, 0, 100), (0, 50, 150), (0, 100, 200),
            (0, 200, 200), (0, 200, 100), (100, 200, 0), (200, 200, 0),
            (255, 150, 0), (255, 50, 0), (255, 0, 0)
        ],
    }

    def __init__(self, parent=None, history_size: int = 500):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._history_size = history_size
        self._fft_size = 2048
        self._db_range = (-100, 0)

        # Data storage
        self._history: deque = deque(maxlen=history_size)

        # Color map
        self._colormap_name = "turbo"
        self._colormap = self._build_colormap(self._colormap_name)

        # Image buffer
        self._image: Optional[QImage] = None

        # Highlights
        self._highlights: List[Tuple[int, int, int, int, QColor]] = []

        # Colors
        self._bg_color = QColor(20, 20, 30)
        self._text_color = QColor(200, 200, 200)

        self.setMinimumHeight(200)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls bar
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(list(self.COLORMAPS.keys()))
        self._color_combo.setCurrentText(self._colormap_name)
        self._color_combo.currentTextChanged.connect(self._on_colormap_changed)
        controls.addWidget(self._color_combo)

        controls.addStretch()

        controls.addWidget(QLabel("Range:"))
        self._range_combo = QComboBox()
        self._range_combo.addItems(["60 dB", "80 dB", "100 dB", "120 dB"])
        self._range_combo.setCurrentIndex(2)
        self._range_combo.currentIndexChanged.connect(self._on_range_changed)
        controls.addWidget(self._range_combo)

        layout.addLayout(controls)

    def _on_colormap_changed(self, name: str):
        """Handle colormap change."""
        self._colormap_name = name
        self._colormap = self._build_colormap(name)
        self._rebuild_image()
        self.update()

    def _on_range_changed(self, index: int):
        """Handle range change."""
        ranges = [60, 80, 100, 120]
        self._db_range = (-ranges[index], 0)
        self._rebuild_image()
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
        if len(power_db) != self._fft_size:
            # Resample if needed
            power_db = np.interp(
                np.linspace(0, 1, self._fft_size),
                np.linspace(0, 1, len(power_db)),
                power_db
            )

        self._history.append(power_db.copy())
        self._update_image()
        self.update()

    def clear(self):
        """Clear the waterfall."""
        self._history.clear()
        self._image = None
        self.update()

    def add_highlight(
        self,
        time_start: int,
        time_end: int,
        freq_start: int,
        freq_end: int,
        color: QColor
    ):
        """Add a highlight region."""
        self._highlights.append((time_start, time_end, freq_start, freq_end, color))
        self.update()

    def clear_highlights(self):
        """Clear all highlights."""
        self._highlights.clear()
        self.update()

    def _update_image(self):
        """Update the image buffer with new data."""
        if len(self._history) == 0:
            return

        # Create image if needed
        if (self._image is None or
            self._image.width() != self._fft_size or
            self._image.height() != self._history_size):
            self._image = QImage(
                self._fft_size, self._history_size,
                QImage.Format.Format_RGB32
            )
            self._image.fill(self._bg_color)

        # Scroll image up
        if len(self._history) > 1:
            # Shift existing data up by one line
            for y in range(self._history_size - 1):
                for x in range(self._fft_size):
                    color = self._image.pixel(x, y + 1)
                    self._image.setPixel(x, y, color)

        # Add new line at bottom
        line = self._history[-1]
        min_db, max_db = self._db_range
        db_range = max_db - min_db

        for x in range(min(len(line), self._fft_size)):
            # Normalize to 0-255
            normalized = (line[x] - min_db) / db_range
            normalized = np.clip(normalized, 0, 1)
            idx = int(normalized * 255)

            r, g, b = self._colormap[idx]
            self._image.setPixel(x, self._history_size - 1, (255 << 24) | (r << 16) | (g << 8) | b)

    def _rebuild_image(self):
        """Rebuild entire image from history."""
        if len(self._history) == 0:
            return

        self._image = QImage(
            self._fft_size, self._history_size,
            QImage.Format.Format_RGB32
        )
        self._image.fill(self._bg_color)

        min_db, max_db = self._db_range
        db_range = max_db - min_db

        for y, line in enumerate(self._history):
            for x in range(min(len(line), self._fft_size)):
                normalized = (line[x] - min_db) / db_range
                normalized = np.clip(normalized, 0, 1)
                idx = int(normalized * 255)

                r, g, b = self._colormap[idx]
                row = self._history_size - len(self._history) + y
                if 0 <= row < self._history_size:
                    self._image.setPixel(x, row, (255 << 24) | (r << 16) | (g << 8) | b)

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
