"""
AM/FM Radio Tuner Widget - car radio style.

A pop-out broadcast-band tuner laid out like a car radio: a frequency display,
a slide-rule tuning dial, AM/FM band buttons and six station presets. Tuning
here retunes the main receiver (through ``frequency_changed``).

All colors come from the active theme palette, so the tuner follows the
application's dark or light theme.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QFont,
        QFontMetricsF,
        QIcon,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QDialog,
        QDialogButtonBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QMenu,
        QPushButton,
        QSizePolicy,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ...dsp.demodulators import AMDemodulator, FMDemodulator

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

# Channel raster used when tuning: the dial, step buttons, wheel and arrow
# keys all land on these, so the display never shows an off-channel value.
AM_STEP_HZ = 10e3
FM_STEP_HZ = 100e3

# How long a preset button must be held to store the current station.
_LONG_PRESS_MS = 700
# Longest preset name (it has to fit a preset button).
_MAX_NAME = 24
# Seek: how often the receiver is asked whether the channel has a signal,
# and how long to wait for an answer before giving up.
_SEEK_POLL_MS = 30
_SEEK_PATIENCE_S = 2.0


def band_range(band: "RadioBand") -> tuple:
    """(min_hz, max_hz) of a band."""
    return FM_RANGE if band == RadioBand.FM else AM_RANGE


def band_step(band: "RadioBand") -> float:
    """Channel step of a band in Hz."""
    return FM_STEP_HZ if band == RadioBand.FM else AM_STEP_HZ


def band_for_frequency(freq_hz: float) -> Optional["RadioBand"]:
    """The broadcast band containing ``freq_hz``, or None."""
    if FM_RANGE[0] <= freq_hz <= FM_RANGE[1]:
        return RadioBand.FM
    if AM_RANGE[0] <= freq_hz <= AM_RANGE[1]:
        return RadioBand.AM
    return None


def on_raster(freq_hz: float, band: "RadioBand") -> bool:
    """Whether ``freq_hz`` is on the band's channel raster (100 kHz FM,
    10 kHz AM)."""
    step = band_step(band)
    channels = float(freq_hz) / step
    return abs(channels - round(channels)) < 1e-6


def format_station(freq_hz: float, band: "RadioBand") -> str:
    """Display text of a station frequency ("101.1 MHz", "1010 kHz").

    Exact: a frequency between FM channels shows as the receiver shows it
    ("100.137 MHz", "100.150 MHz"), an AM one to the hertz it needs
    ("1053 kHz", "1053.5 kHz").
    """
    if band == RadioBand.FM:
        if on_raster(freq_hz, band):
            return f"{freq_hz / 1e6:.1f} MHz"
        return format_receiver_frequency(freq_hz)
    khz = float(freq_hz) / 1e3
    text = f"{khz:.3f}".rstrip("0").rstrip(".")
    return f"{text} kHz"


def format_receiver_frequency(freq_hz: float) -> str:
    """Any receiver frequency, as the main window's toolbar shows it.

    MHz with 3 to 6 decimals: ``"146.520 MHz"``, ``"146.5125 MHz"``.
    """
    text = f"{max(0.0, float(freq_hz)) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def _themes() -> Any:
    """The GUI theme helpers, imported lazily.

    ``sdr_module.gui`` imports this module while it initialises, so importing
    ``sdr_module.gui.themes`` at module level here would be circular.
    """
    from ...gui import themes

    return themes


def _tip(key: str) -> str:
    """Short beginner tooltip from ``sdr_module.utils.tooltips`` (or "")."""
    try:
        from ...utils.tooltips import get_short_tip

        return get_short_tip(key) or ""
    except Exception:  # pragma: no cover - tooltips are optional
        return ""


def _arrow_icon(direction: int, token: str) -> "QIcon":
    """A small triangle icon (-1 left, +1 right) in a palette color."""
    p = _themes().get_palette()
    icon = QIcon()
    for mode, color_token in (
        (QIcon.Mode.Normal, token),
        (QIcon.Mode.Disabled, "disabled"),
    ):
        scale = 2
        pix = QPixmap(14 * scale, 14 * scale)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor(color_token))
        path = QPainterPath()
        if direction < 0:
            path.moveTo(10 * scale, 2 * scale)
            path.lineTo(3 * scale, 7 * scale)
            path.lineTo(10 * scale, 12 * scale)
        else:
            path.moveTo(4 * scale, 2 * scale)
            path.lineTo(11 * scale, 7 * scale)
            path.lineTo(4 * scale, 12 * scale)
        path.closeSubpath()
        painter.drawPath(path)
        painter.end()
        pix.setDevicePixelRatio(scale)
        icon.addPixmap(pix, mode)
    return icon


class FrequencyDisplay(QWidget if HAS_PYQT6 else object):
    """
    LCD-style frequency display.

    Shows the band, the tuned frequency in large digits, and a station line
    (the matching preset, or the band's range). Painted from the active
    theme's LCD colors.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)
        self._frequency_hz = 101.1e6
        self._band = RadioBand.FM
        self._station = ""
        self._stereo = False
        # Receiver frequency while it is outside both broadcast bands.
        self._off_band_hz: Optional[float] = None
        self.setMinimumSize(220, 84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> "QSize":
        """Preferred size."""
        return QSize(340, 96)

    def set_frequency(self, freq_hz: float, band: RadioBand) -> None:
        """Set displayed frequency."""
        self._frequency_hz = freq_hz
        self._band = band
        self.update()

    def set_station(self, text: str) -> None:
        """Set the small station line under the digits ("" for none)."""
        if text != self._station:
            self._station = text
            self.update()

    def set_stereo(self, stereo: bool) -> None:
        """Show or hide the STEREO indicator."""
        if bool(stereo) != self._stereo:
            self._stereo = bool(stereo)
            self.update()

    def set_off_band(self, receiver_hz: Optional[float]) -> None:
        """Show that the receiver is on ``receiver_hz``, outside both
        broadcast bands (dimmed digits, no band badge); ``None`` returns to
        the tuned station."""
        if receiver_hz != self._off_band_hz:
            self._off_band_hz = receiver_hz
            self.update()

    def _digits(self) -> tuple:
        if self._off_band_hz is not None:
            digits, _, unit = format_receiver_frequency(self._off_band_hz).partition(
                " "
            )
            return digits, unit
        digits, _, unit = format_station(self._frequency_hz, self._band).partition(" ")
        return digits, unit

    def paintEvent(self, event) -> None:
        """Draw the display."""
        if not HAS_PYQT6:
            return
        p = _themes().get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = float(self.width()), float(self.height())
        face = QRectF(0.5, 0.5, w - 1.0, h - 1.0)
        painter.setPen(QPen(p.qcolor("lcd_border"), 1))
        painter.setBrush(p.qcolor("lcd_bg"))
        painter.drawRoundedRect(face, 6, 6)

        pad = 10.0

        # Band badge, top left.
        badge_font = QFont(self.font())
        badge_font.setPixelSize(11)
        badge_font.setBold(True)
        bfm = QFontMetricsF(badge_font)
        painter.setFont(badge_font)
        off_band = self._off_band_hz is not None
        if not off_band:
            self._draw_badge(painter, pad, pad, self._band.value, bfm, p, "accent")
        if self._stereo and not off_band:
            text = "STEREO"
            bw = bfm.horizontalAdvance(text) + 12
            self._draw_badge(painter, w - pad - bw, pad, text, bfm, p, "success")

        # Station line along the bottom.
        station_font = QFont(self.font())
        station_font.setPixelSize(11)
        sfm = QFontMetricsF(station_font)
        station_h = sfm.height()

        # Frequency digits and unit, centred in the space between.
        digits, unit = self._digits()
        area_top = pad + bfm.height() * 0.4
        area_bottom = h - pad - station_h - 2
        digit_px = int(max(20, min(46, (area_bottom - area_top) * 0.95)))
        gap = 6.0
        while True:
            # Shrink long readouts (e.g. "1090.000 MHz") to fit the width.
            digit_font = _themes().mono_font(10, bold=True)
            digit_font.setPixelSize(digit_px)
            dfm = QFontMetricsF(digit_font)
            unit_font = QFont(self.font())
            unit_font.setPixelSize(max(11, int(digit_px * 0.34)))
            unit_font.setBold(True)
            ufm = QFontMetricsF(unit_font)
            dw = dfm.horizontalAdvance(digits)
            uw = ufm.horizontalAdvance(unit)
            if dw + gap + uw <= w - 2 * pad or digit_px <= 14:
                break
            digit_px -= 2

        x = (w - (dw + gap + uw)) / 2.0
        baseline = (area_top + area_bottom) / 2.0 + dfm.capHeight() / 2.0
        painter.setFont(digit_font)
        painter.setPen(p.qcolor("caption" if off_band else "lcd_text"))
        painter.drawText(QPointF(x, baseline), digits)
        painter.setFont(unit_font)
        painter.setPen(p.qcolor("caption" if off_band else "lcd_alt"))
        painter.drawText(QPointF(x + dw + gap, baseline), unit)

        painter.setFont(station_font)
        if off_band:
            text = "Receiver · outside the AM/FM broadcast bands"
            painter.setPen(p.qcolor("caption"))
        else:
            text = self._station or self._band_text()
            painter.setPen(p.qcolor("lcd_alt" if self._station else "caption"))
        painter.drawText(
            QRectF(pad, h - pad - station_h, w - 2 * pad, station_h),
            Qt.AlignmentFlag.AlignCenter,
            sfm.elidedText(text, Qt.TextElideMode.ElideRight, w - 2 * pad),
        )
        painter.end()

    def _band_text(self) -> str:
        lo, hi = band_range(self._band)
        if self._band == RadioBand.FM:
            return f"FM broadcast · {lo / 1e6:.1f}–{hi / 1e6:.0f} MHz"
        return f"AM broadcast · {lo / 1e3:.0f}–{hi / 1e3:.0f} kHz"

    @staticmethod
    def _draw_badge(painter, x, y, text, fm, p, tone) -> None:
        bw = fm.horizontalAdvance(text) + 12
        bh = fm.height() + 2
        rect = QRectF(x, y, bw, bh)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor(tone))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(p.qcolor("on_accent"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class TuningDial(QWidget if HAS_PYQT6 else object):
    """
    Slide-rule tuning dial.

    Drag or click to tune, scroll the wheel or use the arrow keys to step one
    channel. Preset stations are marked with dots under the scale.
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
        self._step = FM_STEP_HZ
        self._markers: List[float] = []
        self._dragging = False
        self._last_x = 0
        self._wheel_accum = 0
        # False while the receiver is outside the band: the pointer dims.
        self._active = True

        self.setAccessibleName("Tuning dial")
        self.setMinimumSize(240, 64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Drag or click to tune. Scroll or press Left/Right to step one "
            "channel; Page Up/Down steps ten."
        )

    def sizeHint(self) -> "QSize":
        """Preferred size."""
        return QSize(340, 68)

    # ------------------------------------------------------------------ API

    def set_range(self, min_freq: float, max_freq: float) -> None:
        """Set frequency range."""
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._frequency = max(min_freq, min(self._frequency, max_freq))
        self.update()

    def set_step(self, step_hz: float) -> None:
        """Channel step used for snapping and keyboard / wheel tuning."""
        self._step = max(1.0, float(step_hz))

    def set_markers(self, freqs: Sequence[float]) -> None:
        """Frequencies to mark on the scale (e.g. presets)."""
        self._markers = [float(f) for f in freqs]
        self.update()

    def set_frequency(self, freq: float) -> None:
        """Set current frequency."""
        self._frequency = max(self._min_freq, min(freq, self._max_freq))
        self.update()

    def get_frequency(self) -> float:
        """Get current frequency."""
        return self._frequency

    def set_active(self, active: bool) -> None:
        """Draw the pointer normally (True) or dimmed (False), e.g. while the
        receiver is tuned outside this band."""
        if bool(active) != self._active:
            self._active = bool(active)
            self.update()

    # ------------------------------------------------------------- geometry

    def _is_mhz(self) -> bool:
        return self._max_freq >= 10e6

    def _unit_hz(self) -> float:
        return 1e6 if self._is_mhz() else 1e3

    def _label_font(self) -> "QFont":
        font = QFont(self.font())
        font.setPixelSize(11)
        return font

    def _scale_bounds(self) -> tuple:
        """Left and right x of the frequency scale."""
        fm = QFontMetricsF(self._label_font())
        inset = 10.0 + fm.horizontalAdvance("1600") / 2.0
        return inset, max(inset + 1.0, self.width() - inset)

    def _x_for(self, freq: float) -> float:
        left, right = self._scale_bounds()
        span = max(1.0, self._max_freq - self._min_freq)
        return left + (freq - self._min_freq) / span * (right - left)

    def _snap(self, freq: float) -> float:
        snapped = round(freq / self._step) * self._step
        return max(self._min_freq, min(snapped, self._max_freq))

    def _tick_steps(self) -> tuple:
        """(major, minor) tick spacing in Hz so labels never collide."""
        unit = self._unit_hz()
        fm = QFontMetricsF(self._label_font())
        label_w = fm.horizontalAdvance("1600" if not self._is_mhz() else "108")
        left, right = self._scale_bounds()
        px_per_hz = (right - left) / max(1.0, self._max_freq - self._min_freq)
        for mantissa_exp in range(-1, 5):
            for m in (1, 2, 5):
                major = m * 10**mantissa_exp * unit
                if major * px_per_hz >= label_w + 10:
                    minor = major / (4 if m == 2 else 5)
                    return major, minor
        span = self._max_freq - self._min_freq
        return span, span / 5

    # ---------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:
        """Draw the tuning dial."""
        if not HAS_PYQT6:
            return
        p = _themes().get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = float(self.width()), float(self.height())
        face = QRectF(0.5, 0.5, w - 1.0, h - 1.0)
        border = "accent" if self.hasFocus() else "surface1"
        painter.setPen(QPen(p.qcolor(border), 1))
        painter.setBrush(p.qcolor("plot_bg"))
        painter.drawRoundedRect(face, 6, 6)

        font = self._label_font()
        painter.setFont(font)
        fm = QFontMetricsF(font)
        left, right = self._scale_bounds()
        base_y = h * 0.52  # scale baseline
        label_y = base_y + 6.0

        # Baseline
        painter.setPen(QPen(p.qcolor("plot_grid"), 1.5))
        painter.drawLine(QPointF(left, base_y), QPointF(right, base_y))

        # Ticks, and the label boxes under the major ones.
        major, minor = self._tick_steps()
        unit = self._unit_hz()
        first = math.ceil(self._min_freq / minor - 1e-9) * minor
        labels: List[tuple] = []
        f = first
        while f <= self._max_freq + 1e-6:
            x = self._x_for(f)
            is_major = abs(f / major - round(f / major)) < 1e-6
            length = 10.0 if is_major else 5.0
            painter.setPen(QPen(p.qcolor("plot_axis"), 1.3 if is_major else 1.0))
            painter.drawLine(QPointF(x, base_y - length), QPointF(x, base_y))
            if is_major:
                label = f"{f / unit:g}"
                tw = fm.horizontalAdvance(label)
                labels.append(
                    (label, QRectF(x - tw / 2 - 2, label_y, tw + 4, fm.height()))
                )
            f += minor

        # Preset markers
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor("accent"))
        for marker in self._markers:
            if self._min_freq <= marker <= self._max_freq:
                painter.drawEllipse(
                    QPointF(self._x_for(marker), base_y + 3.0), 2.5, 2.5
                )

        # Pointer: from the top down through the ticks, ending just above
        # the label row so it never runs through a number.
        x = self._x_for(self._frequency)
        top, bottom = 6.0, label_y - 1.0
        pointer = p.qcolor("plot_marker" if self._active else "disabled")
        painter.setPen(QPen(pointer, 2.0))
        painter.drawLine(QPointF(x, top + 5.0), QPointF(x, bottom))
        tri = QPainterPath()
        tri.moveTo(x - 5.0, top)
        tri.lineTo(x + 5.0, top)
        tri.lineTo(x, top + 7.0)
        tri.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pointer)
        painter.drawPath(tri)

        painter.setFont(font)
        painter.setPen(p.qcolor("plot_text"))
        for label, rect in labels:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Unit legend in the top corner the pointer is not in.
        cap = QFont(font)
        cap.setPixelSize(10)
        cap.setBold(True)
        painter.setFont(cap)
        cfm = QFontMetricsF(cap)
        unit_text = "MHz" if self._is_mhz() else "kHz"
        uw = cfm.horizontalAdvance(unit_text)
        unit_rect = QRectF(right - uw, 4.0, uw, cfm.height())
        if unit_rect.left() - 10.0 <= x:
            unit_rect.moveLeft(left)
        painter.setPen(p.qcolor("caption"))
        painter.drawText(
            unit_rect,
            Qt.AlignmentFlag.AlignCenter,
            unit_text,
        )
        painter.end()

    # --------------------------------------------------------------- events

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
        """Step one channel per wheel notch (120 units of angle delta).

        Touchpads and high-resolution wheels send many small deltas, so they
        are accumulated; otherwise one gentle swipe would jump many channels.
        """
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            if (delta > 0) != (self._wheel_accum > 0):
                self._wheel_accum = 0  # direction changed: start afresh
            self._wheel_accum += delta
            channels = int(self._wheel_accum / 120)
            if channels:
                self._wheel_accum -= channels * 120
                self.step(channels)
        event.accept()

    def keyPressEvent(self, event) -> None:
        """Arrow keys step one channel, Page Up/Down ten, Home/End the ends."""
        key = event.key()
        k = Qt.Key
        if key in (k.Key_Right, k.Key_Up):
            self.step(1)
        elif key in (k.Key_Left, k.Key_Down):
            self.step(-1)
        elif key == k.Key_PageUp:
            self.step(10)
        elif key == k.Key_PageDown:
            self.step(-10)
        elif key == k.Key_Home:
            self._tune_to(self._min_freq)
        elif key == k.Key_End:
            self._tune_to(self._max_freq)
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:
        """Show the focus border."""
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        """Hide the focus border."""
        super().focusOutEvent(event)
        self.update()

    def step(self, channels: int) -> None:
        """Tune up (positive) or down by whole channels (from a frequency
        between channels, the first step lands on the next one that way)."""
        position = self._frequency / self._step
        if abs(position - round(position)) < 1e-6:
            base = round(position)
        else:
            base = math.floor(position) if channels > 0 else math.ceil(position)
        self._tune_to((base + channels) * self._step)

    def tune_to(self, freq: float) -> None:
        """Tune to ``freq`` (on the channel raster), emitting the change."""
        self._tune_to(freq)

    def channel_range(self) -> tuple:
        """(lowest, highest) channel of the band."""
        return self._snap(self._min_freq), self._snap(self._max_freq)

    def _tune_to(self, freq: float) -> None:
        freq = self._snap(freq)
        # While dimmed, even the pointer's own position is a new choice: it
        # brings the receiver back to that station.
        if abs(freq - self._frequency) < 1.0 and self._active:
            return
        self._frequency = freq
        self.update()
        self.frequency_changed.emit(self._frequency)

    def _update_frequency_from_position(self, x: float) -> None:
        """Update frequency based on mouse position."""
        left, right = self._scale_bounds()
        ratio = (x - left) / max(1.0, right - left)
        ratio = max(0.0, min(1.0, ratio))
        self._tune_to(self._min_freq + (self._max_freq - self._min_freq) * ratio)


class PresetButton(QPushButton if HAS_PYQT6 else object):
    """
    Station preset button.

    Click to tune. Press and hold, or right-click, to store the current
    station in it; right-click also renames or clears it. Always two lines
    (number and name, then the frequency), so every button is as tall.
    """

    if HAS_PYQT6:
        store_requested = pyqtSignal()
        rename_requested = pyqtSignal()
        clear_requested = pyqtSignal()

    def __init__(self, number: int, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)
        self._number = number
        self._preset: Optional[RadioPreset] = None
        self._long_press_fired = False
        # Why storing isn't possible now (None: it is), e.g. while the
        # receiver is outside the broadcast bands.
        self._store_blocker: Optional[Callable[[], Optional[str]]] = None
        self._unreachable = ""  # why the device can't tune this preset

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(_LONG_PRESS_MS)
        self._hold_timer.timeout.connect(self._on_hold)

        self.setCheckable(True)
        self.setAutoDefault(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fit_two_lines()
        self.set_preset(None)

    def _fit_two_lines(self) -> None:
        """One height for every preset button: two lines of text."""
        lines = 2 * self.fontMetrics().lineSpacing()
        self.setFixedHeight(max(44, lines + 18))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (event.Type.FontChange, event.Type.StyleChange):
            self._fit_two_lines()

    def set_store_blocker(self, blocker: Optional[Callable[[], Optional[str]]]) -> None:
        """``blocker()`` says why storing isn't possible now (None: it is)."""
        self._store_blocker = blocker

    def set_unreachable(self, reason: str) -> None:
        """Why the connected device can't tune this preset ("" if it can):
        the button says "out of range" and the tooltip why (it stays
        enabled, so a station can still be stored in it, or it renamed)."""
        self._unreachable = reason
        self.set_preset(self._preset)

    def unreachable(self) -> str:
        return self._unreachable

    def set_preset(self, preset: Optional[RadioPreset]) -> None:
        """Store a preset for this button (``None`` leaves it empty)."""
        self._preset = preset
        how = "Right-click or press and hold to store the current station here."
        if preset is None:
            self.setText(f"&{self._number}\nEmpty")
            self.setToolTip(f"Preset {self._number} is empty. {how}")
            return
        freq = format_station(preset.frequency_hz, preset.band)
        line = f"{freq} (out of range)" if self._unreachable else freq
        if preset.name:
            self.setText(f"&{self._number}  {preset.name}\n{line}")
            tip = f"Tune to {preset.name} ({freq}). {how}"
        else:
            self.setText(f"&{self._number}\n{line}")
            tip = f"Tune to {freq}. {how} Right-click to name it."
        if self._unreachable:
            tip = f"{freq}: {self._unreachable} {how}"
        self.setToolTip(tip)

    def get_preset(self) -> Optional[RadioPreset]:
        """Get stored preset."""
        return self._preset

    # Long press / context menu to store -----------------------------------

    def mousePressEvent(self, event) -> None:
        """Start the hold timer."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._long_press_fired = False
            self._hold_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """A release after a long press stores; it does not also click."""
        self._hold_timer.stop()
        if self._long_press_fired and event.button() == Qt.MouseButton.LeftButton:
            self._long_press_fired = False
            # Release while "up": Qt clears its pressed state without clicking.
            self.setDown(False)
        super().mouseReleaseEvent(event)

    def _on_hold(self) -> None:
        if self.isDown():
            self._long_press_fired = True
            self.store_requested.emit()

    def contextMenuEvent(self, event) -> None:
        """Offer to tune to, store into, rename or clear this preset."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        tune = menu.addAction("&Tune to Preset")
        tune.setEnabled(self._preset is not None and not self._unreachable)
        store = menu.addAction("&Store Current Station Here")
        blocked = self._store_blocker() if self._store_blocker is not None else None
        if blocked:
            store.setEnabled(False)
            store.setToolTip(blocked)
        menu.addSeparator()
        rename = menu.addAction("&Rename...")
        rename.setEnabled(self._preset is not None)
        clear = menu.addAction("&Clear")
        clear.setEnabled(self._preset is not None)
        chosen = menu.exec(event.globalPos())
        # A new menu per right-click: free it rather than keep one per click.
        menu.deleteLater()
        if chosen is store:
            self.store_requested.emit()
        elif chosen is tune:
            self.click()
        elif chosen is rename:
            self.rename_requested.emit()
        elif chosen is clear:
            self.clear_requested.emit()


class _SeekButton(QPushButton if HAS_PYQT6 else object):
    """A seek arrow that remembers whether Shift was held when it was
    pressed (Shift+click steps one channel instead of seeking)."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)
        self.shift = False

    def mousePressEvent(self, event) -> None:
        self.shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        self.shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        super().keyPressEvent(event)

    def take_shift(self) -> bool:
        """Whether Shift was held for this click (then forgotten)."""
        shift, self.shift = self.shift, False
        return shift


class VolumeKnob(QWidget if HAS_PYQT6 else object):
    """
    Labelled vertical level slider (0-100).

    Not used by :class:`RadioTunerWidget` (the main window's volume control
    sets the level); kept for code that imports it.
    """

    if HAS_PYQT6:
        volume_changed = pyqtSignal(int)

    def __init__(self, label: str = "VOL", parent=None):
        if not HAS_PYQT6:
            return
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setMinimumHeight(72)
        self._slider.setToolTip(f"{label.title()}: 0-100")
        self._slider.valueChanged.connect(self.volume_changed.emit)
        layout.addWidget(self._slider, 1, Qt.AlignmentFlag.AlignHCenter)

        lbl = QLabel(label.upper())
        _themes().set_role(lbl, "caption")
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
    AM/FM Radio Tuner - car radio style.

    A standalone pop-out window with a frequency display, a slide-rule dial,
    band buttons and six presets per band. Tuning emits ``frequency_changed``
    so the main receiver follows.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float, str)  # freq_hz, band
        presets_changed = pyqtSignal()  # stored, renamed or cleared (save them)

    # Default presets: just frequencies (common broadcast channels; the demo
    # device has a station on each). Name them with right-click > Rename.
    DEFAULT_FM_PRESETS = [
        RadioPreset(101.1e6, RadioBand.FM),
        RadioPreset(93.3e6, RadioBand.FM),
        RadioPreset(97.1e6, RadioBand.FM),
        RadioPreset(104.3e6, RadioBand.FM),
        RadioPreset(88.5e6, RadioBand.FM),
        RadioPreset(99.5e6, RadioBand.FM),
    ]

    DEFAULT_AM_PRESETS = [
        RadioPreset(880e3, RadioBand.AM),
        RadioPreset(1010e3, RadioBand.AM),
        RadioPreset(770e3, RadioBand.AM),
        RadioPreset(1050e3, RadioBand.AM),
        RadioPreset(660e3, RadioBand.AM),
        RadioPreset(1260e3, RadioBand.AM),
    ]

    _FOOTER_HINT = "Tunes the main receiver. Use the toolbar's Audio button to listen."
    #: The footer while the main window has no device (as the S-meter says).
    _NO_RECEIVER_HINT = "No receiver: Device › Use Demo Device or Connect…"

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
        # Receiver frequency while it is outside both broadcast bands (the
        # tuner then keeps its own station but shows it is not playing).
        self._off_band_hz: Optional[float] = None

        # Last station tuned on each band, restored when switching back.
        self._band_frequency: Dict[RadioBand, float] = {
            RadioBand.FM: 101.1e6,
            RadioBand.AM: self.DEFAULT_AM_PRESETS[0].frequency_hz,
        }

        # Demodulators
        self._am_demod = AMDemodulator(sample_rate)
        self._fm_demod = FMDemodulator(sample_rate)

        # Presets per band
        self._fm_presets: List[Optional[RadioPreset]] = list(self.DEFAULT_FM_PRESETS)
        self._am_presets: List[Optional[RadioPreset]] = list(self.DEFAULT_AM_PRESETS)

        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.timeout.connect(self._reset_hint)

        # Seek: the receiver says whether the tuned channel has a signal
        # (see set_signal_probe); one channel at a time until it does.
        self._probe: Optional[Callable[[], Optional[str]]] = None
        # Which channels the receiver can tune now (see set_tune_check), and
        # the channel the seek asked for last.
        self._tune_check: Optional[Callable[[float], bool]] = None
        self._seek_target: Optional[float] = None
        self._seek_dir = 0
        self._seek_steps = 0
        self._seek_waited = 0.0
        self._seek_stepping = False
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(_SEEK_POLL_MS)
        self._seek_timer.timeout.connect(self._seek_poll)

        # What the connected device can tune (None: anything) and its name,
        # for the AM band's reach.
        self._device_range: Optional[tuple] = None
        self._device_name = ""
        # Whether the main window has a device at all (see
        # set_receiver_available).
        self._receiver_available = True

        self._setup_ui()
        self._connect_signals()

        # Start with FM band
        self._switch_band(RadioBand.FM)

        self.setWindowTitle("AM/FM Radio Tuner")
        # Resizable from the window frame. No QSizeGrip: it would sit on top
        # of the Close button in the bottom-right corner.
        self.resize(self.sizeHint().expandedTo(QSize(500, 0)))
        self._tuning_dial.setFocus()

    # ------------------------------------------------------------------- UI

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # === Band buttons and frequency display ===
        top = QHBoxLayout()
        top.setSpacing(8)

        band_col = QVBoxLayout()
        band_col.setSpacing(6)
        self._band_group = QButtonGroup(self)
        self._band_group.setExclusive(True)

        self._fm_btn = QPushButton("&FM")
        self._fm_btn.setToolTip(
            f"FM broadcast band, 87.5-108 MHz. {_tip('fm')}".strip()
        )
        self._am_btn = QPushButton("&AM")
        self._am_btn.setToolTip(
            f"AM (medium wave) broadcast band, 530-1700 kHz. {_tip('am')}".strip()
        )
        for btn in (self._fm_btn, self._am_btn):
            btn.setCheckable(True)
            btn.setAutoDefault(False)
            btn.setMinimumWidth(56)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            self._band_group.addButton(btn)
            band_col.addWidget(btn)
        self._fm_btn.setChecked(True)
        top.addLayout(band_col)

        self._freq_display = FrequencyDisplay()
        top.addWidget(self._freq_display, 1)
        main_layout.addLayout(top)

        # === Tuning dial with step buttons ===
        dial_row = QHBoxLayout()
        dial_row.setSpacing(6)
        self._seek_down_btn = _SeekButton()
        self._seek_up_btn = _SeekButton()
        self._seek_down_btn.setAccessibleName("Seek down")
        self._seek_up_btn.setAccessibleName("Seek up")
        self._refresh_icons()
        for btn in (self._seek_down_btn, self._seek_up_btn):
            btn.setAutoDefault(False)
            btn.setFixedWidth(34)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._tuning_dial = TuningDial()
        dial_row.addWidget(self._seek_down_btn)
        dial_row.addWidget(self._tuning_dial, 1)
        dial_row.addWidget(self._seek_up_btn)
        main_layout.addLayout(dial_row)

        # === Presets ===
        presets_box = QGroupBox("Presets")
        presets_box.setToolTip(
            "Click a preset to tune. Right-click or press and hold one to store "
            "the current station in it."
        )
        grid = QGridLayout(presets_box)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self._preset_buttons: List[PresetButton] = []
        for i in range(6):
            btn = PresetButton(i + 1)
            self._preset_buttons.append(btn)
            grid.addWidget(btn, i // 3, i % 3)
        main_layout.addWidget(presets_box)
        # A taller window keeps every row at its natural height (band and
        # step buttons would otherwise stretch) and the footer at the bottom.
        main_layout.addStretch(1)

        # === Footer: hint / feedback and Close ===
        footer = QHBoxLayout()
        footer.setSpacing(8)
        self._hint_label = QLabel(self._FOOTER_HINT)
        self._hint_label.setWordWrap(True)
        _themes().set_role(self._hint_label, "hint")
        footer.addWidget(self._hint_label, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setAutoDefault(False)
        footer.addWidget(buttons, 0, Qt.AlignmentFlag.AlignBottom)
        main_layout.addLayout(footer)

        # Reading order: band, step down, dial, step up, presets, Close.
        chain = [
            self._fm_btn,
            self._am_btn,
            self._seek_down_btn,
            self._tuning_dial,
            self._seek_up_btn,
            *self._preset_buttons,
        ]
        if close_btn is not None:
            chain.append(close_btn)
        for first, second in zip(chain, chain[1:], strict=False):
            QWidget.setTabOrder(first, second)

    def _refresh_icons(self, *_args) -> None:
        """(Re)paint the step-button arrows in the active theme's colors."""
        self._seek_down_btn.setIcon(_arrow_icon(-1, "text"))
        self._seek_up_btn.setIcon(_arrow_icon(1, "text"))

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        _themes().theme_notifier().theme_changed.connect(self._refresh_icons)
        self._tuning_dial.frequency_changed.connect(self._on_frequency_changed)
        self._fm_btn.clicked.connect(lambda: self._switch_band(RadioBand.FM))
        self._am_btn.clicked.connect(lambda: self._switch_band(RadioBand.AM))
        self._seek_down_btn.clicked.connect(self._seek_down)
        self._seek_up_btn.clicked.connect(self._seek_up)

        for i, btn in enumerate(self._preset_buttons):
            btn.clicked.connect(
                lambda _checked=False, idx=i: self._on_preset_clicked(idx)
            )
            btn.store_requested.connect(lambda idx=i: self.store_preset(idx))
            btn.rename_requested.connect(lambda idx=i: self.rename_preset(idx))
            btn.clear_requested.connect(lambda idx=i: self.clear_preset(idx))
            btn.set_store_blocker(self._store_blocker)

    # --------------------------------------------------------------- tuning

    def _current_presets(self) -> List[RadioPreset]:
        return self._fm_presets if self._band == RadioBand.FM else self._am_presets

    def _apply_frequency(self, freq_hz: float, emit: bool = True) -> None:
        """Clamp to the band, update every view and optionally emit."""
        if emit and not self._seek_stepping:
            self.stop_seek()  # tuning by hand ends a seek
        lo, hi = band_range(self._band)
        self._frequency = max(lo, min(float(freq_hz), hi))
        self._band_frequency[self._band] = self._frequency
        self._leave_off_band()
        self._tuning_dial.set_frequency(self._frequency)
        self._freq_display.set_frequency(self._frequency, self._band)
        self._refresh_preset_states()
        if emit:
            self.frequency_changed.emit(self._frequency, self._band.value)

    def _switch_band(self, band: RadioBand, emit: bool = True) -> None:
        """Switch between AM and FM bands, restoring that band's last station."""
        self._band = band
        (self._fm_btn if band == RadioBand.FM else self._am_btn).setChecked(True)
        self._tuning_dial.set_range(*band_range(band))
        self._tuning_dial.set_step(band_step(band))
        if emit:
            self.stop_seek()
        self._update_seek_tips()
        self._update_presets(self._current_presets())
        self._apply_frequency(self._reachable(band, self._band_frequency[band]), emit)
        if not self.is_seeking():
            self._hint_timer.stop()
            self._reset_hint()  # (the AM band's note, if any)

    def _reachable(self, band: RadioBand, freq_hz: float) -> float:
        """``freq_hz``, or (when the device can't tune it) the first preset
        of ``band`` it can, else its lowest reachable channel: a HackRF
        switched to AM lands on 1000 kHz or above instead of being refused."""
        if self._device_range is None:
            return freq_hz
        low, high = self._device_range
        if low <= freq_hz <= high:
            return freq_hz
        presets = self._fm_presets if band == RadioBand.FM else self._am_presets
        for preset in presets:
            if preset is not None and low <= preset.frequency_hz <= high:
                return preset.frequency_hz
        step = band_step(band)
        band_low, band_high = band_range(band)
        channel = math.ceil(max(low, band_low) / step) * step
        return min(max(channel, band_low), band_high)

    def _update_presets(self, presets: List[Optional[RadioPreset]]) -> None:
        """Update preset buttons for current band."""
        for i, btn in enumerate(self._preset_buttons):
            preset = presets[i] if i < len(presets) else None
            btn.set_preset(preset)
            btn.set_unreachable(self._unreachable_reason(preset))
        self._refresh_markers()
        self._refresh_preset_states()

    def _refresh_markers(self) -> None:
        """Mark the current band's presets on the dial."""
        self._tuning_dial.set_markers(
            [pr.frequency_hz for pr in self._current_presets() if pr is not None]
        )

    def _refresh_preset_states(self) -> None:
        """Check the preset matching the tuned station; name it on the display."""
        station = ""
        for i, btn in enumerate(self._preset_buttons):
            preset = btn.get_preset()
            match = (
                preset is not None
                and self._off_band_hz is None
                and preset.band == self._band
                and abs(preset.frequency_hz - self._frequency) < 1.0
            )
            btn.setChecked(match)
            if match and not station:
                station = f"Preset {i + 1}" + (
                    f" · {preset.name}" if preset.name else ""
                )
        if (
            not station
            and not on_raster(self._frequency, self._band)
            and not getattr(self, "_rds_text", "")
        ):
            station = "Between channels"
        self._freq_display.set_station(self._with_rds(station))

    def set_rds_text(self, text: str) -> None:
        """Show the station's RDS name (and RadioText) on the display's
        station line, after the preset it matches ("" for none). The main
        window sends it while broadcast FM is received."""
        text = str(text or "")
        if text != getattr(self, "_rds_text", ""):
            self._rds_text = text
            self._refresh_preset_states()

    def _with_rds(self, station: str) -> str:
        """The station line: the matching preset (if any), then the RDS
        name."""
        rds = str(getattr(self, "_rds_text", "") or "")
        if not rds or self._off_band_hz is not None or self._band != RadioBand.FM:
            return station
        return f"{station} · {rds}" if station else rds

    def _on_frequency_changed(self, freq: float) -> None:
        """Handle frequency change from tuning dial."""
        self._apply_frequency(freq)

    def _on_preset_clicked(self, index: int) -> None:
        """Handle preset button click."""
        presets = self._current_presets()
        reason = self._preset_buttons[index].unreachable()
        if index < len(presets) and presets[index] is not None and reason:
            self._refresh_preset_states()
            freq = format_station(presets[index].frequency_hz, self._band)
            self._show_hint(f"{freq}: {reason}", "warning")
        elif index < len(presets) and presets[index] is not None:
            self._apply_frequency(presets[index].frequency_hz)
        else:
            self._refresh_preset_states()
            self._show_hint(
                f"Preset {index + 1} is empty. Right-click or hold it to store "
                "the current station.",
                "warning",
            )

    def _store_blocker(self) -> Optional[str]:
        """Why the station can't be stored now (None: it can)."""
        if self._off_band_hz is not None:
            return (
                "Tune to an AM or FM station first: the receiver is on "
                f"{format_receiver_frequency(self._off_band_hz)}."
            )
        return None

    def store_preset(self, index: int) -> None:
        """Store the station tuned (the receiver's frequency) in preset
        ``index`` (refused, with a hint, while the receiver is outside the
        broadcast bands)."""
        blocked = self._store_blocker()
        if blocked:
            self._show_hint(blocked, "warning")
            return
        old = self._current_presets()[index]
        # A station stored over a named one keeps the name only if it is
        # the same station.
        name = ""
        if old is not None and abs(old.frequency_hz - self._frequency) < 1.0:
            name = old.name
        preset = RadioPreset(self._frequency, self._band, name)
        self._set_preset(index, preset)
        self._show_hint(
            f"Stored {format_station(self._frequency, self._band)} in preset "
            f"{index + 1}.",
            "success",
        )

    def _set_preset(self, index: int, preset: Optional[RadioPreset]) -> None:
        if self._band == RadioBand.FM:
            self._fm_presets[index] = preset
        else:
            self._am_presets[index] = preset
        button = self._preset_buttons[index]
        button.set_preset(preset)
        button.set_unreachable(self._unreachable_reason(preset))
        self._refresh_markers()
        self._refresh_preset_states()
        self.presets_changed.emit()

    def rename_preset(self, index: int, name: Optional[str] = None) -> None:
        """Name preset ``index`` (asks when ``name`` is None; an empty name
        removes it)."""
        preset = self._current_presets()[index]
        if preset is None:
            return
        if name is None:
            text, ok = QInputDialog.getText(
                self,
                "Rename Preset",
                f"Name for preset {index + 1} "
                f"({format_station(preset.frequency_hz, preset.band)}):",
                text=preset.name,
            )
            if not ok:
                return
            name = text
        name = " ".join(str(name).split())[:_MAX_NAME]
        self._set_preset(index, RadioPreset(preset.frequency_hz, preset.band, name))
        if name:
            self._show_hint(f"Preset {index + 1} is now {name}.", "success")
        else:
            self._show_hint(f"Preset {index + 1} has no name now.", None)

    def clear_preset(self, index: int) -> None:
        """Empty preset ``index``."""
        if self._current_presets()[index] is None:
            return
        self._set_preset(index, None)
        self._show_hint(f"Preset {index + 1} cleared.", None)

    def get_presets(self) -> Dict[str, list]:
        """Both bands' presets, to save: ``{"fm": [[hz, name] or None,
        ...], "am": [...]}``."""

        def pack(presets: List[Optional[RadioPreset]]) -> list:
            return [
                None if p is None else [float(p.frequency_hz), p.name] for p in presets
            ]

        return {"fm": pack(self._fm_presets), "am": pack(self._am_presets)}

    def set_presets(self, data: Dict[str, Any]) -> None:
        """Restore :meth:`get_presets` (entries outside their band, or that
        can't be read, keep the default)."""
        if not isinstance(data, dict):
            return
        for key, band, presets in (
            ("fm", RadioBand.FM, self._fm_presets),
            ("am", RadioBand.AM, self._am_presets),
        ):
            saved = data.get(key)
            if not isinstance(saved, list):
                continue
            low, high = band_range(band)
            for i, entry in enumerate(saved[: len(presets)]):
                if entry is None:
                    presets[i] = None
                    continue
                try:
                    freq = float(entry[0])
                    name = " ".join(str(entry[1] or "").split())[:_MAX_NAME]
                except (TypeError, ValueError, IndexError):
                    continue
                if low <= freq <= high and math.isfinite(freq):
                    presets[i] = RadioPreset(freq, band, name)
        self._update_presets(self._current_presets())

    def _show_hint(self, text: str, tone: Optional[str] = None) -> None:
        """Show feedback in the footer for a few seconds."""
        self._hint_label.setText(text)
        _themes().set_tone(self._hint_label, tone)
        self._hint_timer.start(4000)

    def _reset_hint(self) -> None:
        """Back to the standing footer text for the current state."""
        tone: Optional[str]
        if not self._receiver_available:
            text, tone = self._NO_RECEIVER_HINT, "warning"
        elif self._off_band_hz is not None:
            text = (
                f"The receiver is on {format_receiver_frequency(self._off_band_hz)}, "
                "outside the broadcast bands. Pick a preset or tune the dial to "
                "switch it to a station."
            )
            tone = "info"
        elif self._band == RadioBand.AM and self._am_note():
            text, tone = self._am_note(), "info"
        else:
            text, tone = self._FOOTER_HINT, None
        self._hint_label.setText(text)
        _themes().set_tone(self._hint_label, tone)

    # ------------------------------------------------------------- off band

    def is_off_band(self) -> bool:
        """True while the receiver is tuned outside both broadcast bands."""
        return self._off_band_hz is not None

    def _enter_off_band(self, receiver_hz: float) -> None:
        """Show that the receiver is on ``receiver_hz``, outside AM and FM.

        The tuner keeps its own station (the dial pointer dims there), no
        band or preset is selected, and the footer says where the receiver
        is. Tuning from here switches the receiver back to a station.
        """
        self._off_band_hz = float(receiver_hz)
        # An exclusive group keeps one button checked; lift that briefly.
        self._band_group.setExclusive(False)
        self._fm_btn.setChecked(False)
        self._am_btn.setChecked(False)
        self._band_group.setExclusive(True)
        self._tuning_dial.set_active(False)
        self._freq_display.set_off_band(self._off_band_hz)
        self._refresh_preset_states()
        self._hint_timer.stop()
        self._reset_hint()

    def _leave_off_band(self) -> None:
        if self._off_band_hz is None:
            return
        self._off_band_hz = None
        (self._fm_btn if self._band == RadioBand.FM else self._am_btn).setChecked(True)
        self._tuning_dial.set_active(True)
        self._freq_display.set_off_band(None)
        self._hint_timer.stop()
        self._reset_hint()

    # ------------------------------------------------------------- seek

    def set_signal_probe(self, probe: Optional[Callable[[], Optional[str]]]) -> None:
        """How seek knows a station is there: ``probe()`` returns "signal"
        (the tuned channel opens the squelch), "quiet", "wait" (ask again
        shortly), "refused" (the receiver didn't tune there) or None (the
        receiver isn't running). Without a probe the arrows step a channel."""
        self._probe = probe
        self._update_seek_tips()

    def set_tune_check(self, check: Optional[Callable[[float], bool]]) -> None:
        """Which channels seek may try: ``check(freq_hz)`` is False for a
        frequency the receiver can't tune now (e.g. outside a recording
        being played back). Seek goes round the channels it can tune and
        skips the rest, as it skips those outside the device's range (see
        :meth:`set_device_range`). None: only the device's range limits it."""
        self._tune_check = check

    def _seek_reachable(self, freq_hz: float) -> bool:
        """Whether seek may tune ``freq_hz`` (the device's range, then the
        receiver's own check)."""
        if self._device_range is not None:
            low, high = self._device_range
            if not low <= freq_hz <= high:
                return False
        check = self._tune_check
        return check is None or bool(check(freq_hz))

    def _seek_channels(self) -> List[float]:
        """The band's channels that seek may tune, lowest first."""
        low, high = self._tuning_dial.channel_range()
        step = band_step(self._band)
        first, last = int(round(low / step)), int(round(high / step))
        return [
            k * step for k in range(first, last + 1) if self._seek_reachable(k * step)
        ]

    def is_seeking(self) -> bool:
        return self._seek_dir != 0

    def _update_seek_tips(self) -> None:
        step = band_step(self._band) / 1e3
        if self._probe is None:
            down = f"Tune down one channel ({step:.0f} kHz)."
            up = f"Tune up one channel ({step:.0f} kHz)."
        else:
            down = (
                "Seek down: tune down until a station opens the squelch. "
                f"Shift+click: one channel ({step:.0f} kHz)."
            )
            up = (
                "Seek up: tune up until a station opens the squelch. "
                f"Shift+click: one channel ({step:.0f} kHz)."
            )
        self._seek_down_btn.setToolTip(down)
        self._seek_up_btn.setToolTip(up)

    def _seek_up(self) -> None:
        """Seek up (Shift, or without a receiver to ask: one channel)."""
        self._seek(1, self._seek_up_btn.take_shift())

    def _seek_down(self) -> None:
        """Seek down (Shift, or without a receiver to ask: one channel)."""
        self._seek(-1, self._seek_down_btn.take_shift())

    def _seek(self, direction: int, shift: bool = False) -> None:
        if self.is_seeking():
            self.stop_seek("Seek stopped.")
            return
        if shift or self._probe is None:
            self._tuning_dial.step(direction)
            return
        if self._probe() is None:
            self._show_hint(
                "Seek listens for stations: start the receiver first (Start or "
                "Space). Shift+click steps one channel.",
                "warning",
            )
            return
        self._seek_dir = direction
        self._seek_steps = 0
        self._seek_step()

    def _seek_step(self) -> None:
        """On to the next channel the receiver can tune (round the band, or
        the part of it the receiver can tune, at its ends), then listen."""
        channels = self._seek_channels()
        if not channels:
            self.stop_seek(
                f"Seek stopped: the receiver can't tune any {self._band.value} "
                "broadcast channel now.",
                "warning",
            )
            return
        here = self._tuning_dial.get_frequency()
        # (From between two channels the first step lands on the next one
        # that way.)
        if self._seek_dir > 0:
            ahead = [c for c in channels if c > here + 0.5]
            target = ahead[0] if ahead else channels[0]
        else:
            ahead = [c for c in channels if c < here - 0.5]
            target = ahead[-1] if ahead else channels[-1]
        self._seek_target = target
        self._seek_steps += 1
        self._seek_waited = 0.0
        self._seek_stepping = True
        try:
            if self._off_band_hz is not None:
                self._leave_off_band()
            self._tuning_dial.tune_to(target)
        finally:
            self._seek_stepping = False
        direction = "up" if self._seek_dir > 0 else "down"
        self._show_hint(
            f"Seeking {direction}… {format_station(self._frequency, self._band)}",
            "info",
        )
        self._hint_timer.stop()
        self._seek_timer.start()

    def _seek_poll(self) -> None:
        if not self.is_seeking():
            return
        state = self._probe() if self._probe is not None else None
        where = format_station(self._frequency, self._band)
        if state is None:
            self.stop_seek("Seek stopped: the receiver stopped.", "warning")
        elif state == "refused":
            # (The tuner is back on the frequency still tuned: name the one
            # the receiver refused.)
            if self._seek_target is not None:
                where = format_station(self._seek_target, self._band)
            self.stop_seek(f"Seek stopped: the receiver can't tune {where}.", "warning")
        elif state == "wait":
            self._seek_waited += _SEEK_POLL_MS / 1000.0
            if self._seek_waited > _SEEK_PATIENCE_S:
                self.stop_seek("Seek stopped: no samples from the receiver.", "warning")
            else:
                self._seek_timer.start()
        elif state == "signal":
            self.stop_seek(f"Found a station on {where}.", "success")
        else:
            # Round the band (or the part the receiver can tune) once.
            channels = self._seek_channels()
            if self._seek_steps >= len(channels):
                self.stop_seek(self._nothing_found_text(channels), "warning")
            else:
                self._seek_step()

    def _nothing_found_text(self, channels: List[float]) -> str:
        """What a seek that went round ``channels`` without a station says."""
        low, high = self._tuning_dial.channel_range()
        whole = int(round((high - low) / band_step(self._band))) + 1
        if not channels or len(channels) >= whole:
            return "No station strong enough to open the squelch anywhere in the band."
        if len(channels) == 1:
            return (
                "No station strong enough to open the squelch on "
                f"{format_station(channels[0], self._band)}, the only channel of "
                "the band the receiver can tune."
            )
        first = format_station(channels[0], self._band)
        last = format_station(channels[-1], self._band)
        return (
            f"No station strong enough to open the squelch from {first} to "
            f"{last}, the part of the band the receiver can tune."
        )

    def stop_seek(self, message: str = "", tone: Optional[str] = None) -> None:
        """End a seek (saying ``message``)."""
        if not self.is_seeking():
            return
        self._seek_dir = 0
        self._seek_timer.stop()
        if message:
            self._show_hint(message, tone)
        else:
            self._hint_timer.stop()
            self._reset_hint()

    def hideEvent(self, event) -> None:
        self.stop_seek()
        super().hideEvent(event)

    # ------------------------------------------------------------- device

    def set_receiver_available(self, available: bool) -> None:
        """Whether the main window has a device: without one the footer
        says so (presets and the dial still tune, nothing can play)."""
        available = bool(available)
        if available == self._receiver_available:
            return
        self._receiver_available = available
        self._hint_timer.stop()
        self._reset_hint()

    def set_device_range(
        self, low_hz: Optional[float], high_hz: Optional[float], name: str = ""
    ) -> None:
        """What the connected device can tune (None: anything, e.g. the
        demo device) and its name: presets it can't reach are disabled and
        the AM band is explained (or disabled) when it can't tune it."""
        if low_hz is None or high_hz is None:
            device_range = None
        else:
            device_range = (float(low_hz), float(high_hz))
        name = name or "The receiver"
        if (device_range, name) == (self._device_range, self._device_name):
            return  # (unchanged: keep the footer's feedback)
        self._device_range = device_range
        self._device_name = name
        am_low, am_high = AM_RANGE
        reachable = self._device_range is None or (
            self._device_range[0] <= am_high and self._device_range[1] >= am_low
        )
        tip = f"AM (medium wave) broadcast band, 530-1700 kHz. {_tip('am')}".strip()
        note = self._am_note()
        if note:
            tip = f"{tip}\n\n{note}"
        self._am_btn.setToolTip(tip)
        self._am_btn.setEnabled(reachable)
        if not reachable and self._band == RadioBand.AM and self._off_band_hz is None:
            self._switch_band(RadioBand.FM, emit=False)
        self._update_presets(self._current_presets())
        self._hint_timer.stop()
        self._reset_hint()

    def _am_note(self) -> str:
        """What limits the AM band on the connected device ("" if nothing)."""
        if self._device_range is None:
            return ""
        low, high = self._device_range
        am_low, am_high = AM_RANGE
        name = self._device_name
        span = f"{low / 1e6:g}–{high / 1e6:g} MHz"
        if low > am_high or high < am_low:
            return (
                f"{name} tunes {span}: the AM broadcast band (530–1700 kHz) is "
                "out of its reach."
            )
        if low > am_low:
            return (
                f"{name} tunes from {low / 1e3:.0f} kHz: AM stations below that "
                "are out of its reach."
            )
        if name == "RTL-SDR":
            return (
                "An RTL-SDR receives the AM band by direct sampling, which only "
                "some dongles support (such as the RTL-SDR Blog V3 and V4); "
                "others need an upconverter."
            )
        return ""

    def _unreachable_reason(self, preset: Optional[RadioPreset]) -> str:
        """Why the device can't tune ``preset`` ("" if it can)."""
        if preset is None or self._device_range is None:
            return ""
        low, high = self._device_range
        if low <= preset.frequency_hz <= high:
            return ""
        return (
            f"{self._device_name} can't tune it (it tunes "
            f"{low / 1e6:g}–{high / 1e6:g} MHz)."
        )

    def set_stereo(self, stereo: bool) -> None:
        """Set stereo indicator state."""
        self._stereo = bool(stereo)
        self._freq_display.set_stereo(self._stereo)

    def set_volume(self, value: int) -> None:
        """Set the output level (0-100) used by :meth:`process_samples`."""
        self._volume = max(0, min(100, int(value)))

    def set_muted(self, muted: bool) -> None:
        """Mute or unmute :meth:`process_samples` output."""
        self._muted = bool(muted)

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Process I/Q samples and return audio.

        Args:
            samples: Complex I/Q samples from SDR

        Returns:
            Demodulated audio samples
        """
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
        """The tuner's station in Hz (the one its dial and presets start from).

        While :meth:`is_off_band`, the receiver is elsewhere.
        """
        return self._frequency

    def get_band(self) -> RadioBand:
        """Get current band."""
        return self._band

    def set_frequency(self, freq_hz: float) -> None:
        """Follow the receiver's frequency (does not emit ``frequency_changed``).

        A frequency inside a broadcast band tunes the tuner there, switching
        bands if needed. Anything outside both bands leaves the tuner's
        station alone and shows that the receiver is elsewhere (see
        :meth:`is_off_band`), rather than a station the receiver is not on.
        """
        band = band_for_frequency(freq_hz)
        if band is None:
            self._enter_off_band(freq_hz)
        elif band != self._band:
            self._band_frequency[band] = freq_hz
            self._switch_band(band, emit=False)
        else:
            self._apply_frequency(freq_hz, emit=False)


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
