"""
HAM Radio Signal Meter Widget.

Classic analog-style S-meter display like the ones on vintage Collins or Kenwood rigs.
Shows signal strength the way old HAMs expect to see it.

Features:
- Analog needle display with S1-S9 scale
- dB over S9 scale (+10, +20, +40, +60)
- RST readout
- Verbal report display
- Peak hold indicator
"""

from __future__ import annotations

import math
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QGroupBox, QSizePolicy, QComboBox
    )
    from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
    from PyQt6.QtGui import (
        QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
        QLinearGradient, QPainterPath, QPolygonF
    )
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..dsp.signal_meter import (
    SignalMeter, SignalReading, SignalMode,
    SignalHistory, S_METER_REFERENCE, S9_DBM
)


class AnalogMeterWidget(QWidget if HAS_PYQT6 else object):
    """
    Classic analog S-meter display.

    Draws a vintage-style meter with needle, S-unit scale, and dB over S9.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._s_units: float = 1.0
        self._peak_s_units: float = 1.0
        self._show_peak: bool = True

        self.setMinimumSize(280, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Colors
        self._bg_color = QColor(30, 30, 35)
        self._scale_color = QColor(220, 220, 200)
        self._needle_color = QColor(220, 60, 60)
        self._peak_color = QColor(60, 200, 60)
        self._s9_color = QColor(255, 200, 50)

    def set_value(self, s_units: float, peak_s_units: Optional[float] = None) -> None:
        """Set meter value in S-units."""
        self._s_units = max(0.0, min(15.0, s_units))
        if peak_s_units is not None:
            self._peak_s_units = max(0.0, min(15.0, peak_s_units))
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h - 20  # Pivot point at bottom center

        # Background
        painter.fillRect(self.rect(), self._bg_color)

        # Draw bezel/frame
        painter.setPen(QPen(QColor(80, 80, 85), 2))
        painter.drawRoundedRect(2, 2, w - 4, h - 4, 8, 8)

        # Draw scale arc
        self._draw_scale(painter, cx, cy, w, h)

        # Draw peak needle (if enabled)
        if self._show_peak:
            self._draw_needle(painter, cx, cy, self._peak_s_units,
                            self._peak_color, 2)

        # Draw main needle
        self._draw_needle(painter, cx, cy, self._s_units,
                         self._needle_color, 3)

        # Draw pivot cap
        painter.setBrush(QBrush(QColor(60, 60, 65)))
        painter.setPen(QPen(QColor(100, 100, 105), 1))
        painter.drawEllipse(QPointF(cx, cy), 8, 8)

        painter.end()

    def _draw_scale(self, painter: QPainter, cx: int, cy: int,
                    w: int, h: int) -> None:
        """Draw the S-meter scale."""
        # Scale arc parameters
        radius = min(w, h) * 0.7
        start_angle = 150  # degrees
        sweep_angle = -120  # negative for clockwise

        # Draw arc
        painter.setPen(QPen(self._scale_color, 2))

        # S-unit markers (S1-S9)
        for s in range(1, 10):
            angle = self._s_to_angle(float(s))
            self._draw_tick(painter, cx, cy, radius, angle,
                          str(s), is_major=(s == 9))

        # dB over S9 markers (+20, +40, +60)
        for db_over in [20, 40, 60]:
            s_units = 9 + db_over / 6.0  # 6 dB per S-unit
            angle = self._s_to_angle(s_units)
            self._draw_tick(painter, cx, cy, radius, angle,
                          f"+{db_over}", is_major=True, is_over=True)

        # Draw "S" label
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.setPen(self._scale_color)
        painter.drawText(20, h - 25, "S")

        # Draw "dB" label
        painter.drawText(w - 35, h - 25, "dB")

    def _draw_tick(self, painter: QPainter, cx: int, cy: int,
                   radius: float, angle: float, label: str,
                   is_major: bool = False, is_over: bool = False) -> None:
        """Draw a tick mark on the scale."""
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Tick lengths
        inner_r = radius - (15 if is_major else 8)
        outer_r = radius

        x1, y1 = cx + inner_r * cos_a, cy - inner_r * sin_a
        x2, y2 = cx + outer_r * cos_a, cy - outer_r * sin_a

        # Draw tick
        color = self._s9_color if is_over else self._scale_color
        painter.setPen(QPen(color, 2 if is_major else 1))
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Draw label
        if label:
            font_size = 10 if is_major else 8
            painter.setFont(QFont("Arial", font_size))
            painter.setPen(color)

            label_r = radius - 25
            lx, ly = cx + label_r * cos_a, cy - label_r * sin_a

            # Adjust position for text centering
            fm = QFontMetrics(painter.font())
            text_width = fm.horizontalAdvance(label)
            text_height = fm.height()

            painter.drawText(int(lx - text_width/2),
                           int(ly + text_height/4), label)

    def _draw_needle(self, painter: QPainter, cx: int, cy: int,
                     s_units: float, color: QColor, width: int) -> None:
        """Draw the meter needle."""
        angle = self._s_to_angle(s_units)
        rad = math.radians(angle)

        radius = min(self.width(), self.height()) * 0.6
        x = cx + radius * math.cos(rad)
        y = cy - radius * math.sin(rad)

        # Draw needle with gradient
        painter.setPen(QPen(color, width))
        painter.drawLine(QPointF(cx, cy), QPointF(x, y))

        # Draw needle tip
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(x, y), 3, 3)

    def _s_to_angle(self, s_units: float) -> float:
        """Convert S-units to display angle."""
        # Map S1 to 150°, S9+60 (15 units) to 30°
        return 150 - (s_units - 1) * (120 / 14)


class SignalMeterPanel(QWidget if HAS_PYQT6 else object):
    """
    Complete signal meter panel with analog display and digital readouts.

    Shows:
    - Analog S-meter
    - Digital S-meter reading
    - RST report
    - Verbal report
    - dBm value
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._meter = SignalMeter()
        self._history = SignalHistory()
        self._last_reading: Optional[SignalReading] = None

        self._setup_ui()

        # Update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(100)  # 10 Hz update

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Analog meter
        self._analog_meter = AnalogMeterWidget()
        layout.addWidget(self._analog_meter)

        # Digital readouts
        readout_group = QGroupBox("Signal Report")
        readout_layout = QGridLayout(readout_group)

        # S-Meter
        readout_layout.addWidget(QLabel("S-Meter:"), 0, 0)
        self._s_meter_label = QLabel("S1")
        self._s_meter_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #4f4;"
        )
        readout_layout.addWidget(self._s_meter_label, 0, 1)

        # dBm
        readout_layout.addWidget(QLabel("Power:"), 0, 2)
        self._dbm_label = QLabel("-121 dBm")
        self._dbm_label.setStyleSheet("font-size: 14px; color: #aaa;")
        readout_layout.addWidget(self._dbm_label, 0, 3)

        # RST
        readout_layout.addWidget(QLabel("RST:"), 1, 0)
        self._rst_label = QLabel("51")
        self._rst_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #ff4;"
        )
        readout_layout.addWidget(self._rst_label, 1, 1)

        # SNR
        readout_layout.addWidget(QLabel("SNR:"), 1, 2)
        self._snr_label = QLabel("0 dB")
        self._snr_label.setStyleSheet("font-size: 14px; color: #aaa;")
        readout_layout.addWidget(self._snr_label, 1, 3)

        # Verbal report
        readout_layout.addWidget(QLabel("Report:"), 2, 0)
        self._verbal_label = QLabel("One and one")
        self._verbal_label.setStyleSheet(
            "font-size: 14px; font-style: italic; color: #8af;"
        )
        readout_layout.addWidget(self._verbal_label, 2, 1, 1, 3)

        layout.addWidget(readout_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Phone (SSB/FM)", "CW (Morse)", "Digital"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        mode_layout.addStretch()

        # Peak hold toggle
        self._peak_label = QLabel("Peak: --")
        self._peak_label.setStyleSheet("color: #6c6;")
        mode_layout.addWidget(self._peak_label)

        layout.addLayout(mode_layout)

    def get_meter(self) -> SignalMeter:
        """Get the signal meter instance."""
        return self._meter

    def update_samples(self, samples: np.ndarray) -> None:
        """Update with new I/Q samples."""
        reading = self._meter.update(samples)
        self._last_reading = reading
        self._history.add(reading)

    def _update_display(self) -> None:
        """Update display elements."""
        if self._last_reading is None:
            return

        reading = self._last_reading

        # Update analog meter
        self._analog_meter.set_value(reading.s_units, reading.s_units + 0.5)

        # Update digital readouts
        self._s_meter_label.setText(reading.s_meter)
        self._dbm_label.setText(f"{reading.power_dbm:.0f} dBm")
        self._rst_label.setText(self._meter.get_rst())
        self._snr_label.setText(f"{reading.snr_db:.0f} dB")
        self._verbal_label.setText(self._meter.get_verbal_report())

        # Peak hold
        peak_s = 9 + (reading.peak_hold_dbm - S9_DBM) / 6.0
        if peak_s >= 9:
            over = reading.peak_hold_dbm - S9_DBM
            self._peak_label.setText(f"Peak: S9+{over:.0f}")
        else:
            self._peak_label.setText(f"Peak: S{int(peak_s)}")

        # Color code S-meter based on strength
        if reading.s_units >= 9:
            self._s_meter_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: #ff4;"
            )
        elif reading.s_units >= 5:
            self._s_meter_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: #4f4;"
            )
        else:
            self._s_meter_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: #888;"
            )

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode change."""
        modes = [SignalMode.PHONE, SignalMode.CW, SignalMode.DIGITAL]
        if 0 <= index < len(modes):
            self._meter.set_mode(modes[index])

    def get_qso_report(self) -> str:
        """Get report for QSO logging."""
        return self._meter.get_qso_report()

    def get_contest_report(self) -> str:
        """Get contest-style report."""
        return self._meter.get_contest_report()


class CompactSignalMeter(QWidget if HAS_PYQT6 else object):
    """
    Compact signal meter for embedding in other panels.

    Shows S-meter and RST in minimal space.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._meter = SignalMeter()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # S-Meter bar
        self._bar_label = QLabel("S: [░░░░░░░░░] S1")
        self._bar_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._bar_label)

        # RST
        self._rst_label = QLabel("RST: 51")
        self._rst_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._rst_label)

        layout.addStretch()

    def update_samples(self, samples: np.ndarray) -> None:
        """Update with samples."""
        reading = self._meter.update(samples)

        # Update bar graph
        bar = self._meter.get_bar_graph(9)
        self._bar_label.setText(f"S: {bar}")

        # Update RST
        self._rst_label.setText(f"RST: {self._meter.get_rst()}")


__all__ = [
    'AnalogMeterWidget',
    'SignalMeterPanel',
    'CompactSignalMeter',
]
