"""
Spectrum analyzer widget.

Provides real-time spectrum visualization with:
- FFT-based power spectrum display
- Peak hold and averaging
- Configurable frequency span
- Marker support
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtGui import QPainter

try:
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class SpectrumWidget(QWidget if HAS_PYQT6 else object):
    """
    Spectrum analyzer display widget.

    Shows power spectrum with configurable averaging and peak hold.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._fft_size = 2048
        self._center_freq = 100e6
        self._sample_rate = 2.4e6
        self._db_range = (-100, 0)

        # Spectrum data
        self._spectrum = np.zeros(self._fft_size)
        self._peak_hold = np.full(self._fft_size, -120.0)
        self._average = np.zeros(self._fft_size)
        self._avg_count = 0
        self._avg_alpha = 0.3

        # Display options
        self._show_peak = True
        self._show_average = False
        self._grid_enabled = True

        # Colors
        self._bg_color = QColor(20, 20, 30)
        self._grid_color = QColor(60, 60, 80)
        self._spectrum_color = QColor(0, 255, 100)
        self._peak_color = QColor(255, 100, 100)
        self._avg_color = QColor(100, 100, 255)
        self._text_color = QColor(200, 200, 200)

        # Markers
        self._markers: List[Tuple[float, float]] = []  # (freq, power)

        self.setMinimumHeight(200)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls bar
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Avg:"))
        self._avg_combo = QComboBox()
        self._avg_combo.addItems(["Off", "2", "4", "8", "16", "32"])
        self._avg_combo.currentIndexChanged.connect(self._on_avg_changed)
        controls.addWidget(self._avg_combo)

        controls.addStretch()

        layout.addLayout(controls)

    def _on_avg_changed(self, index: int):
        """Handle averaging mode change."""
        if index == 0:
            self._show_average = False
        else:
            self._show_average = True
            # Alpha for exponential averaging
            n = [0, 2, 4, 8, 16, 32][index]
            self._avg_alpha = 2.0 / (n + 1)

        self._avg_count = 0
        self._average = np.zeros(self._fft_size)

    def update_spectrum(self, power_db: np.ndarray):
        """
        Update spectrum with new data.

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

        self._spectrum = power_db

        # Update peak hold
        self._peak_hold = np.maximum(self._peak_hold, power_db)

        # Update average
        if self._show_average:
            if self._avg_count == 0:
                self._average = power_db.copy()
            else:
                self._average = self._avg_alpha * power_db + (1 - self._avg_alpha) * self._average
            self._avg_count += 1

        self.update()

    def reset_peak(self):
        """Reset peak hold."""
        self._peak_hold = np.full(self._fft_size, -120.0)

    def reset_average(self):
        """Reset averaging."""
        self._average = np.zeros(self._fft_size)
        self._avg_count = 0

    def set_frequency_range(self, center_freq: float, sample_rate: float):
        """Set frequency range for display."""
        self._center_freq = center_freq
        self._sample_rate = sample_rate
        self.update()

    def set_db_range(self, min_db: float, max_db: float):
        """Set dB range for display."""
        self._db_range = (min_db, max_db)
        self.update()

    def paintEvent(self, event):
        """Paint the spectrum display."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get dimensions
        width = self.width()
        height = self.height() - 25  # Leave room for controls
        margin_left = 50
        margin_bottom = 30
        margin_right = 10
        margin_top = 10

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        # Draw background
        painter.fillRect(0, 0, width, height, self._bg_color)

        # Draw grid
        if self._grid_enabled:
            self._draw_grid(painter, margin_left, margin_top, plot_width, plot_height)

        # Draw spectrum
        self._draw_spectrum(painter, margin_left, margin_top, plot_width, plot_height)

        # Draw axes labels
        self._draw_labels(painter, margin_left, margin_top, plot_width, plot_height)

    def _draw_grid(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw grid lines."""
        painter.setPen(QPen(self._grid_color, 1, Qt.PenStyle.DotLine))

        # Vertical lines (frequency)
        for i in range(1, 10):
            px = x + i * w // 10
            painter.drawLine(px, y, px, y + h)

        # Horizontal lines (power)
        for i in range(1, 10):
            py = y + i * h // 10
            painter.drawLine(x, py, x + w, py)

    def _draw_spectrum(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw spectrum traces."""
        if len(self._spectrum) == 0:
            return

        min_db, max_db = self._db_range
        db_range = max_db - min_db

        def db_to_y(db: float) -> float:
            normalized = (db - min_db) / db_range
            return y + h - normalized * h

        def bin_to_x(bin_idx: int) -> float:
            return x + bin_idx * w / len(self._spectrum)

        # Draw peak hold
        if self._show_peak and np.any(self._peak_hold > -120):
            painter.setPen(QPen(self._peak_color, 1))
            path = QPainterPath()
            path.moveTo(bin_to_x(0), db_to_y(self._peak_hold[0]))
            for i in range(1, len(self._peak_hold)):
                path.lineTo(bin_to_x(i), db_to_y(self._peak_hold[i]))
            painter.drawPath(path)

        # Draw average
        if self._show_average and self._avg_count > 0:
            painter.setPen(QPen(self._avg_color, 1))
            path = QPainterPath()
            path.moveTo(bin_to_x(0), db_to_y(self._average[0]))
            for i in range(1, len(self._average)):
                path.lineTo(bin_to_x(i), db_to_y(self._average[i]))
            painter.drawPath(path)

        # Draw current spectrum
        painter.setPen(QPen(self._spectrum_color, 1))
        path = QPainterPath()
        path.moveTo(bin_to_x(0), db_to_y(self._spectrum[0]))
        for i in range(1, len(self._spectrum)):
            path.lineTo(bin_to_x(i), db_to_y(self._spectrum[i]))
        painter.drawPath(path)

    def _draw_labels(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw axis labels."""
        painter.setPen(self._text_color)
        font = QFont("Monospace", 8)
        painter.setFont(font)

        min_db, max_db = self._db_range

        # Y-axis (power) labels
        for i in range(0, 11, 2):
            db = min_db + i * (max_db - min_db) / 10
            py = y + h - i * h // 10
            painter.drawText(5, py + 4, f"{db:.0f}")

        # X-axis (frequency) labels
        freq_start = self._center_freq - self._sample_rate / 2
        freq_end = self._center_freq + self._sample_rate / 2

        for i in range(0, 11, 2):
            freq = freq_start + i * (freq_end - freq_start) / 10
            px = x + i * w // 10

            if freq >= 1e9:
                label = f"{freq/1e9:.3f}G"
            else:
                label = f"{freq/1e6:.2f}M"

            painter.drawText(px - 20, y + h + 15, label)
