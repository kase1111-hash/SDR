"""
AM/FM Radio Tuner Widget - Vintage Car Radio Style.

A pop-out radio tuner styled like a classic car radio from the 1970s-80s.
Features retro aesthetics with modern SDR functionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
    from PyQt6.QtWidgets import (
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..dsp.demodulators import AMDemodulator, FMDemodulator

logger = logging.getLogger(__name__)


class RadioBand(Enum):
    """Radio frequency bands."""

    AM = "AM"
    FM = "FM"


@dataclass
class RadioPreset:
    """Radio station preset."""

    frequency_hz: float
    band: RadioBand
    name: str = ""


# Classic radio frequency ranges
AM_RANGE = (530e3, 1700e3)  # 530 kHz - 1700 kHz
FM_RANGE = (87.5e6, 108e6)  # 87.5 MHz - 108 MHz


class FrequencyDisplay(QWidget if HAS_PYQT6 else object):
    """
    LED-style frequency display.

    Mimics the orange/amber segmented displays of vintage car radios.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)
        self._frequency_hz = 101.1e6
        self._band = RadioBand.FM
        self.setMinimumSize(200, 60)
        self.setMaximumHeight(70)

    def set_frequency(self, freq_hz: float, band: RadioBand) -> None:
        """Set displayed frequency."""
        self._frequency_hz = freq_hz
        self._band = band
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the LED-style display."""
        if not HAS_PYQT6:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background - dark with slight gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(20, 20, 25))
        gradient.setColorAt(1, QColor(10, 10, 15))
        painter.fillRect(self.rect(), gradient)

        # Display bezel
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.drawRect(2, 2, self.width() - 4, self.height() - 4)

        # Inner bezel highlight
        painter.setPen(QPen(QColor(40, 40, 45), 1))
        painter.drawRect(5, 5, self.width() - 10, self.height() - 10)

        # Format frequency based on band
        if self._band == RadioBand.FM:
            freq_mhz = self._frequency_hz / 1e6
            freq_text = f"{freq_mhz:6.1f}"
            unit_text = "MHz"
        else:
            freq_khz = self._frequency_hz / 1e3
            freq_text = f"{freq_khz:6.0f}"
            unit_text = "kHz"

        # LED amber color
        led_color = QColor(255, 140, 0)
        led_glow = QColor(255, 180, 50, 100)

        # Draw frequency digits with glow effect
        font = QFont("Courier New", 28, QFont.Weight.Bold)
        painter.setFont(font)

        # Glow effect
        painter.setPen(led_glow)
        painter.drawText(15, 48, freq_text)

        # Main text
        painter.setPen(led_color)
        painter.drawText(14, 47, freq_text)

        # Draw band indicator
        band_font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(band_font)
        painter.setPen(led_color)
        painter.drawText(self.width() - 55, 30, self._band.value)

        # Draw unit
        unit_font = QFont("Arial", 10)
        painter.setFont(unit_font)
        painter.setPen(QColor(200, 100, 0))
        painter.drawText(self.width() - 55, 50, unit_text)

        painter.end()


class TuningDial(QWidget if HAS_PYQT6 else object):
    """
    Analog-style tuning dial.

    Rotary knob appearance with frequency indicator.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)

        self._min_freq = FM_RANGE[0]
        self._max_freq = FM_RANGE[1]
        self._frequency = 101.1e6
        self._dragging = False
        self._last_x = 0

        self.setMinimumSize(300, 80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_range(self, min_freq: float, max_freq: float) -> None:
        """Set frequency range."""
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._frequency = max(min_freq, min(self._frequency, max_freq))
        self.update()

    def set_frequency(self, freq: float) -> None:
        """Set current frequency."""
        self._frequency = max(self._min_freq, min(freq, self._max_freq))
        self.update()

    def get_frequency(self) -> float:
        """Get current frequency."""
        return self._frequency

    def paintEvent(self, event) -> None:
        """Draw the tuning dial."""
        if not HAS_PYQT6:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background track
        track_y = self.height() // 2
        track_height = 30

        # Chrome-look gradient for track
        track_gradient = QLinearGradient(
            0, track_y - track_height // 2, 0, track_y + track_height // 2
        )
        track_gradient.setColorAt(0, QColor(80, 80, 85))
        track_gradient.setColorAt(0.5, QColor(120, 120, 125))
        track_gradient.setColorAt(1, QColor(60, 60, 65))

        painter.fillRect(
            10,
            track_y - track_height // 2,
            self.width() - 20,
            track_height,
            track_gradient,
        )

        # Draw frequency scale markings
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Draw tick marks
        num_ticks = 10
        for i in range(num_ticks + 1):
            x = 20 + (self.width() - 40) * i / num_ticks
            painter.drawLine(int(x), track_y - 5, int(x), track_y + 5)

            # Frequency labels
            freq = self._min_freq + (self._max_freq - self._min_freq) * i / num_ticks
            if self._max_freq > 1e6:
                label = f"{freq/1e6:.0f}"
            else:
                label = f"{freq/1e3:.0f}"
            painter.drawText(int(x) - 10, track_y + 20, label)

        # Draw tuning indicator (red line)
        freq_ratio = (self._frequency - self._min_freq) / (
            self._max_freq - self._min_freq
        )
        indicator_x = 20 + (self.width() - 40) * freq_ratio

        # Indicator glow
        painter.setPen(QPen(QColor(255, 50, 50, 100), 6))
        painter.drawLine(int(indicator_x), track_y - 15, int(indicator_x), track_y + 15)

        # Indicator line
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.drawLine(int(indicator_x), track_y - 15, int(indicator_x), track_y + 15)

        painter.end()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for tuning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_x = event.position().x()
            self._update_frequency_from_position(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse drag for tuning."""
        if self._dragging:
            self._update_frequency_from_position(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release."""
        self._dragging = False

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for fine tuning."""
        # Fine tuning step
        if self._max_freq > 1e6:
            step = 100e3  # 100 kHz for FM
        else:
            step = 10e3  # 10 kHz for AM

        if event.angleDelta().y() > 0:
            self._frequency = min(self._frequency + step, self._max_freq)
        else:
            self._frequency = max(self._frequency - step, self._min_freq)

        self.update()
        self.frequency_changed.emit(self._frequency)

    def _update_frequency_from_position(self, x: float) -> None:
        """Update frequency based on mouse position."""
        ratio = (x - 20) / (self.width() - 40)
        ratio = max(0, min(1, ratio))
        self._frequency = self._min_freq + (self._max_freq - self._min_freq) * ratio
        self.update()
        self.frequency_changed.emit(self._frequency)


class PresetButton(QPushButton if HAS_PYQT6 else object):
    """
    Vintage-style radio preset button.

    Square chrome buttons with number labels.
    """

    def __init__(self, number: int, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(str(number), parent)
        self._number = number
        self._preset: Optional[RadioPreset] = None

        self.setFixedSize(50, 40)
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply vintage button style."""
        self.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #808080, stop:0.4 #606060,
                    stop:0.5 #505050, stop:1 #404040);
                border: 2px outset #707070;
                border-radius: 3px;
                color: #FFFFFF;
                font-family: Arial;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #909090, stop:0.4 #707070,
                    stop:0.5 #606060, stop:1 #505050);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:0.4 #505050,
                    stop:0.5 #606060, stop:1 #707070);
                border: 2px inset #505050;
            }
        """
        )

    def set_preset(self, preset: RadioPreset) -> None:
        """Store a preset for this button."""
        self._preset = preset
        if preset.name:
            self.setToolTip(f"{preset.name}\n{self._format_freq(preset)}")
        else:
            self.setToolTip(self._format_freq(preset))

    def get_preset(self) -> Optional[RadioPreset]:
        """Get stored preset."""
        return self._preset

    def _format_freq(self, preset: RadioPreset) -> str:
        """Format frequency for display."""
        if preset.band == RadioBand.FM:
            return f"{preset.frequency_hz / 1e6:.1f} FM"
        else:
            return f"{preset.frequency_hz / 1e3:.0f} AM"


class VolumeKnob(QWidget if HAS_PYQT6 else object):
    """
    Vintage rotary volume knob.
    """

    if HAS_PYQT6:
        volume_changed = pyqtSignal(int)

    def __init__(self, label: str = "VOL", parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Slider (vertical)
        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setFixedHeight(80)
        self._slider.setStyleSheet(
            """
            QSlider::groove:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #303030, stop:0.5 #404040, stop:1 #303030);
                width: 8px;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #808080, stop:0.5 #A0A0A0, stop:1 #808080);
                height: 20px;
                margin: 0 -4px;
                border-radius: 6px;
                border: 1px solid #606060;
            }
        """
        )
        self._slider.valueChanged.connect(self.volume_changed.emit)
        layout.addWidget(self._slider, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Label
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #C0C0C0; font-size: 10px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

    def value(self) -> int:
        """Get current value."""
        return self._slider.value()

    def setValue(self, value: int) -> None:
        """Set value."""
        self._slider.setValue(value)


class RadioTunerWidget(QDialog if HAS_PYQT6 else object):
    """
    AM/FM Radio Tuner - Vintage Car Radio Style.

    A standalone pop-out window styled like a classic car radio
    from the 1970s-80s era with chrome accents and amber displays.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float, str)  # freq_hz, band
        audio_output = pyqtSignal(np.ndarray)

    # Default FM presets (classic rock stations style)
    DEFAULT_FM_PRESETS = [
        RadioPreset(101.1e6, RadioBand.FM, "Rock"),
        RadioPreset(93.3e6, RadioBand.FM, "Classic"),
        RadioPreset(97.1e6, RadioBand.FM, "Pop"),
        RadioPreset(104.3e6, RadioBand.FM, "Jazz"),
        RadioPreset(88.5e6, RadioBand.FM, "NPR"),
        RadioPreset(99.5e6, RadioBand.FM, "Country"),
    ]

    DEFAULT_AM_PRESETS = [
        RadioPreset(880e3, RadioBand.AM, "News"),
        RadioPreset(1010e3, RadioBand.AM, "Talk"),
        RadioPreset(770e3, RadioBand.AM, "Sports"),
        RadioPreset(1050e3, RadioBand.AM, "Weather"),
        RadioPreset(660e3, RadioBand.AM, "News2"),
        RadioPreset(1260e3, RadioBand.AM, "Oldies"),
    ]

    def __init__(self, parent=None, sample_rate: float = 2.4e6):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required for RadioTunerWidget")

        super().__init__(parent)

        self._sample_rate = sample_rate
        self._band = RadioBand.FM
        self._frequency = 101.1e6
        self._volume = 50
        self._muted = False
        self._stereo = False

        # Demodulators
        self._am_demod = AMDemodulator(sample_rate)
        self._fm_demod = FMDemodulator(sample_rate)

        # FM presets (current band)
        self._fm_presets = list(self.DEFAULT_FM_PRESETS)
        self._am_presets = list(self.DEFAULT_AM_PRESETS)

        self._setup_ui()
        self._apply_style()
        self._connect_signals()

        # Start with FM band
        self._switch_band(RadioBand.FM)

        self.setWindowTitle("AM/FM Radio")
        self.setFixedSize(420, 280)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # === Top Section: Display and Controls ===
        top_section = QHBoxLayout()

        # Frequency display (left side)
        self._freq_display = FrequencyDisplay()
        top_section.addWidget(self._freq_display)

        top_section.addSpacing(15)

        # Volume and Tone controls (right side)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        self._vol_knob = VolumeKnob("VOL")
        controls_layout.addWidget(self._vol_knob)

        self._tone_knob = VolumeKnob("TONE")
        self._tone_knob.setValue(50)
        controls_layout.addWidget(self._tone_knob)

        self._balance_knob = VolumeKnob("BAL")
        self._balance_knob.setValue(50)
        controls_layout.addWidget(self._balance_knob)

        top_section.addLayout(controls_layout)
        main_layout.addLayout(top_section)

        # === Tuning Dial ===
        self._tuning_dial = TuningDial()
        main_layout.addWidget(self._tuning_dial)

        # === Band Selector and Presets ===
        bottom_section = QHBoxLayout()

        # AM/FM Band buttons
        band_layout = QVBoxLayout()
        band_layout.setSpacing(5)

        self._fm_btn = QPushButton("FM")
        self._fm_btn.setFixedSize(50, 35)
        self._fm_btn.setCheckable(True)
        self._fm_btn.setChecked(True)
        band_layout.addWidget(self._fm_btn)

        self._am_btn = QPushButton("AM")
        self._am_btn.setFixedSize(50, 35)
        self._am_btn.setCheckable(True)
        band_layout.addWidget(self._am_btn)

        bottom_section.addLayout(band_layout)
        bottom_section.addSpacing(20)

        # Preset buttons
        presets_layout = QHBoxLayout()
        presets_layout.setSpacing(8)

        self._preset_buttons: List[PresetButton] = []
        for i in range(1, 7):
            btn = PresetButton(i)
            self._preset_buttons.append(btn)
            presets_layout.addWidget(btn)

        bottom_section.addLayout(presets_layout)
        bottom_section.addStretch()

        # Stereo indicator
        self._stereo_indicator = QLabel("STEREO")
        self._stereo_indicator.setStyleSheet(
            """
            color: #404040;
            font-size: 11px;
            font-weight: bold;
            padding: 5px;
        """
        )
        bottom_section.addWidget(self._stereo_indicator)

        main_layout.addLayout(bottom_section)

        # === Power and Mute buttons ===
        button_row = QHBoxLayout()

        self._power_btn = QPushButton("⏻ POWER")
        self._power_btn.setCheckable(True)
        self._power_btn.setChecked(True)
        self._power_btn.setFixedWidth(80)
        button_row.addWidget(self._power_btn)

        self._mute_btn = QPushButton("🔇 MUTE")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setFixedWidth(80)
        button_row.addWidget(self._mute_btn)

        button_row.addStretch()

        # Seek buttons
        self._seek_down_btn = QPushButton("◀◀ SEEK")
        self._seek_down_btn.setFixedWidth(80)
        button_row.addWidget(self._seek_down_btn)

        self._seek_up_btn = QPushButton("SEEK ▶▶")
        self._seek_up_btn.setFixedWidth(80)
        button_row.addWidget(self._seek_up_btn)

        main_layout.addLayout(button_row)

    def _apply_style(self) -> None:
        """Apply vintage car radio styling."""
        self.setStyleSheet(
            """
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:0.1 #1a1a1a,
                    stop:0.9 #1a1a1a, stop:1 #0a0a0a);
                border: 3px solid #404040;
                border-radius: 10px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #606060, stop:0.4 #404040,
                    stop:0.5 #353535, stop:1 #2a2a2a);
                border: 2px outset #555555;
                border-radius: 5px;
                color: #E0E0E0;
                font-family: Arial;
                font-size: 11px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #707070, stop:0.4 #505050,
                    stop:0.5 #454545, stop:1 #3a3a3a);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:0.4 #353535,
                    stop:0.5 #404040, stop:1 #505050);
                border: 2px inset #404040;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF8C00, stop:0.4 #CC7000,
                    stop:0.5 #B06000, stop:1 #904000);
                color: #FFFFFF;
            }
        """
        )

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._tuning_dial.frequency_changed.connect(self._on_frequency_changed)
        self._vol_knob.volume_changed.connect(self._on_volume_changed)
        self._fm_btn.clicked.connect(lambda: self._switch_band(RadioBand.FM))
        self._am_btn.clicked.connect(lambda: self._switch_band(RadioBand.AM))
        self._power_btn.toggled.connect(self._on_power_toggled)
        self._mute_btn.toggled.connect(self._on_mute_toggled)
        self._seek_down_btn.clicked.connect(self._seek_down)
        self._seek_up_btn.clicked.connect(self._seek_up)

        # Preset buttons
        for i, btn in enumerate(self._preset_buttons):
            btn.clicked.connect(lambda checked, idx=i: self._on_preset_clicked(idx))

        # Long press to store preset
        for i, btn in enumerate(self._preset_buttons):
            btn.pressed.connect(lambda idx=i: self._start_preset_store(idx))

    def _switch_band(self, band: RadioBand) -> None:
        """Switch between AM and FM bands."""
        self._band = band

        if band == RadioBand.FM:
            self._fm_btn.setChecked(True)
            self._am_btn.setChecked(False)
            self._tuning_dial.set_range(*FM_RANGE)
            self._frequency = max(FM_RANGE[0], min(self._frequency, FM_RANGE[1]))
            self._update_presets(self._fm_presets)
        else:
            self._fm_btn.setChecked(False)
            self._am_btn.setChecked(True)
            self._tuning_dial.set_range(*AM_RANGE)
            self._frequency = max(AM_RANGE[0], min(self._frequency, AM_RANGE[1]))
            self._update_presets(self._am_presets)

        self._tuning_dial.set_frequency(self._frequency)
        self._freq_display.set_frequency(self._frequency, self._band)
        self.frequency_changed.emit(self._frequency, self._band.value)

    def _update_presets(self, presets: List[RadioPreset]) -> None:
        """Update preset buttons for current band."""
        for i, btn in enumerate(self._preset_buttons):
            if i < len(presets):
                btn.set_preset(presets[i])
            else:
                btn.set_preset(None)

    def _on_frequency_changed(self, freq: float) -> None:
        """Handle frequency change from tuning dial."""
        self._frequency = freq
        self._freq_display.set_frequency(freq, self._band)
        self.frequency_changed.emit(freq, self._band.value)

    def _on_volume_changed(self, value: int) -> None:
        """Handle volume change."""
        self._volume = value

    def _on_power_toggled(self, on: bool) -> None:
        """Handle power button toggle."""
        if on:
            self._freq_display.show()
        else:
            self._freq_display.hide()

    def _on_mute_toggled(self, muted: bool) -> None:
        """Handle mute button toggle."""
        self._muted = muted

    def _on_preset_clicked(self, index: int) -> None:
        """Handle preset button click."""
        presets = self._fm_presets if self._band == RadioBand.FM else self._am_presets
        if index < len(presets):
            preset = presets[index]
            self._frequency = preset.frequency_hz
            self._tuning_dial.set_frequency(self._frequency)
            self._freq_display.set_frequency(self._frequency, self._band)
            self.frequency_changed.emit(self._frequency, self._band.value)

    def _start_preset_store(self, index: int) -> None:
        """Start timer for long-press preset storage."""
        # In a full implementation, we'd use a timer to detect long press
        # For simplicity, we'll skip this for now
        pass

    def store_preset(self, index: int) -> None:
        """Store current frequency as preset."""
        preset = RadioPreset(self._frequency, self._band)
        if self._band == RadioBand.FM:
            self._fm_presets[index] = preset
        else:
            self._am_presets[index] = preset
        self._preset_buttons[index].set_preset(preset)

    def _seek_up(self) -> None:
        """Seek to next station (up frequency)."""
        if self._band == RadioBand.FM:
            step = 200e3  # 200 kHz step for FM
            max_freq = FM_RANGE[1]
        else:
            step = 10e3  # 10 kHz step for AM
            max_freq = AM_RANGE[1]

        self._frequency = min(self._frequency + step, max_freq)
        self._tuning_dial.set_frequency(self._frequency)
        self._freq_display.set_frequency(self._frequency, self._band)
        self.frequency_changed.emit(self._frequency, self._band.value)

    def _seek_down(self) -> None:
        """Seek to previous station (down frequency)."""
        if self._band == RadioBand.FM:
            step = 200e3
            min_freq = FM_RANGE[0]
        else:
            step = 10e3
            min_freq = AM_RANGE[0]

        self._frequency = max(self._frequency - step, min_freq)
        self._tuning_dial.set_frequency(self._frequency)
        self._freq_display.set_frequency(self._frequency, self._band)
        self.frequency_changed.emit(self._frequency, self._band.value)

    def set_stereo(self, stereo: bool) -> None:
        """Set stereo indicator state."""
        self._stereo = stereo
        if stereo:
            self._stereo_indicator.setStyleSheet(
                """
                color: #00FF00;
                font-size: 11px;
                font-weight: bold;
                padding: 5px;
            """
            )
        else:
            self._stereo_indicator.setStyleSheet(
                """
                color: #404040;
                font-size: 11px;
                font-weight: bold;
                padding: 5px;
            """
            )

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Process I/Q samples and return audio.

        Args:
            samples: Complex I/Q samples from SDR

        Returns:
            Demodulated audio samples
        """
        if not self._power_btn.isChecked():
            return np.zeros(len(samples), dtype=np.float32)

        # Demodulate based on band
        if self._band == RadioBand.FM:
            audio = self._fm_demod.demodulate(samples)
        else:
            audio = self._am_demod.demodulate(samples)

        # Apply volume
        if self._muted:
            audio = np.zeros_like(audio)
        else:
            volume_factor = self._volume / 100.0
            audio = audio * volume_factor

        return audio.astype(np.float32)

    def get_frequency(self) -> float:
        """Get current tuned frequency in Hz."""
        return self._frequency

    def get_band(self) -> RadioBand:
        """Get current band."""
        return self._band

    def set_frequency(self, freq_hz: float) -> None:
        """Set frequency externally."""
        if self._band == RadioBand.FM:
            freq_hz = max(FM_RANGE[0], min(freq_hz, FM_RANGE[1]))
        else:
            freq_hz = max(AM_RANGE[0], min(freq_hz, AM_RANGE[1]))

        self._frequency = freq_hz
        self._tuning_dial.set_frequency(freq_hz)
        self._freq_display.set_frequency(freq_hz, self._band)


def show_radio_tuner(parent=None, sample_rate: float = 2.4e6) -> RadioTunerWidget:
    """
    Show the radio tuner as a pop-out window.

    Args:
        parent: Parent widget
        sample_rate: SDR sample rate

    Returns:
        RadioTunerWidget instance
    """
    if not HAS_PYQT6:
        raise ImportError("PyQt6 is required for the radio tuner")

    tuner = RadioTunerWidget(parent, sample_rate)
    tuner.show()
    return tuner
