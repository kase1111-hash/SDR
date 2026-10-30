"""
HAM Radio Signal Meter Widget.

Classic analog-style S-meter display like the ones on vintage Collins or Kenwood rigs.
Shows signal strength the way old HAMs expect to see it.

Features:
- Analog needle display with S1-S9 scale
- dB over S9 scale (+10 ... +60)
- RST readout
- Verbal report display
- Peak hold indicator

Colors come from the active theme palette (see ``sdr_module.gui.themes``), so the
meter reads correctly in both the dark and the light theme.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QFont,
        QFontMetricsF,
        QGuiApplication,
        QPainter,
        QPainterPath,
        QPen,
        QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMenu,
        QSizePolicy,
        QStyle,
        QStyleOptionComboBox,
        QStylePainter,
        QToolButton,
        QVBoxLayout,
        QWidget,
        QWidgetAction,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..signal_meter import (
    PEAK_HOLD_S,
    S9_DBM,
    SignalHistory,
    SignalMeter,
    SignalMode,
    SignalReading,
)


def _themes() -> Any:
    """The GUI theme helpers, imported lazily.

    ``sdr_module.gui`` imports this module while it initialises, so importing
    ``sdr_module.gui.themes`` at module level here would be circular.
    """
    from ...gui import themes

    return themes


def get_palette() -> Any:
    """Active theme palette."""
    return _themes().get_palette()


def set_role(widget: Any, role: Optional[str] = None, tone: Any = ...) -> Any:
    """Tag ``widget`` with a stylesheet role (see ``gui.themes.set_role``)."""
    return _themes().set_role(widget, role, tone)


def set_tone(widget: Any, tone: Optional[str]) -> Any:
    """Set the color tone of ``widget`` (see ``gui.themes.set_tone``)."""
    return _themes().set_tone(widget, tone)


# Scale layout: S0..S9 fill the first 60 % of the arc, 0..+60 dB over S9 the
# rest. One S-unit (6 dB) and 10 dB over S9 are then the same arc length, so
# every tick on the dial is evenly spaced.
DB_PER_S_UNIT = 6.0
MAX_OVER_S9_DB = 60.0
MAX_S_UNITS = 9.0 + MAX_OVER_S9_DB / DB_PER_S_UNIT  # S9+60
_S9_FRACTION = 0.6

# Readings older than this (seconds) are shown as stale.
_STALE_AFTER_S = 3.0
# Seconds of level history in the sparkline.
HISTORY_S = 30.0

# Empty states: (title on the gauge, hint under it, the line in the report,
# the full sentence for the compact layout's one line).
_EMPTY_STATES = {
    "initial": (
        "No reading yet",
        "Start the receiver to measure.",
        "Start the receiver to measure.",
        "Waiting for a signal. Start the receiver.",
    ),
    "waiting": (
        "Waiting for samples…",
        "",
        "Waiting for samples…",
        "Waiting for samples…",
    ),
    "stopped": (
        "Receiver stopped",
        "Press Start (Space) to measure.",
        "Receiver stopped.",
        "Receiver stopped. Press Start (Space) to measure.",
    ),
    "no_device": (
        "No receiver",
        "Device › Use Demo Device or Connect…",
        "No receiver connected.",
        "No receiver: Device › Use Demo Device or Connect…",
    ),
    "stale": (
        "No new samples",
        "Start the receiver.",
        "No new samples.",
        "No new samples. Start the receiver.",
    ),
}

# Shown in place of a reading before data arrives; the em dash matches the
# rest of the app's empty values.
_EMPTY = "—"  # em dash


def dbm_to_s_units(dbm: float) -> float:
    """dBm to (uncapped above S9, up to S9+60) S-units for the analog scale."""
    s_units = 9.0 + (dbm - S9_DBM) / DB_PER_S_UNIT
    return max(0.0, min(MAX_S_UNITS, s_units))


def format_s_units(s_units: float) -> str:
    """Format S-units as an S-meter string ("S7", "S9", "S9+12")."""
    if s_units >= 9.0:
        over = (s_units - 9.0) * DB_PER_S_UNIT
        return "S9" if over < 0.5 else f"S9+{over:.0f}"
    return f"S{max(1, min(9, int(round(s_units))))}"


def strength_tone(s_units: float, signal: bool = True) -> Optional[str]:
    """Theme tone for a signal strength; matches the zones on the dial.

    ``signal`` False (nothing above the noise) is always muted: noise must
    not look like a good signal, whatever its level."""
    if not signal:
        return "muted"
    if s_units >= 9.0 + 40.0 / DB_PER_S_UNIT:
        return "danger"
    if s_units >= 9.0:
        return "warning"
    if s_units >= 5.0:
        return "success"
    return "muted"


class AnalogMeterWidget(QWidget if HAS_PYQT6 else object):
    """
    Classic analog S-meter display.

    Draws a vintage-style meter with needle, S-unit scale, and dB over S9. The
    arc, ticks, labels and needle share one geometry that scales with the
    widget, and every color is read from the active theme palette on paint.
    """

    # Half of the arc's sweep, in degrees either side of vertical.
    _HALF_SWEEP = 50.0
    _MARGIN = 10.0
    # Below _ROOMY_HEIGHT the face trades its margins and the room under the
    # arc for scale size, down to _MIN_HEIGHT. That keeps a usable gauge in
    # a short tab (1024x640 window) with the S-METER / RST readouts under it
    # still in view.
    _MIN_HEIGHT = 88
    _ROOMY_HEIGHT = 140.0
    _TIGHT_MARGIN = 4.0

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._s_units: float = 0.0
        self._peak_s_units: float = 0.0
        self._noise_s_units: Optional[float] = None
        self._show_peak: bool = True
        self._active: bool = False  # False until the first reading arrives
        # Empty-state text drawn on the face while there is no reading.
        self._placeholder: Tuple[str, str] = ("", "")

        # Preferred height fits the full-width scale; in a short panel the
        # gauge shrinks (down to its minimum) before anything scrolls.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumSize(220, self._MIN_HEIGHT)
        self.setToolTip(
            "Signal strength in S-units (6 dB each, S9 = -73 dBm), then dB over "
            "S9. The thin needle shows the peak hold; the shaded start of the "
            "scale is the noise in the channel."
        )

    # --------------------------------------------------------------- sizing

    def preferred_height(self, width: int) -> int:
        """Height that fits the whole scale when drawn at ``width``."""
        fm = QFontMetricsF(self._label_font(width))
        radius = self._radius_for_width(float(width), fm)
        return int(math.ceil(self._needed_height(radius, fm, width)))

    def sizeHint(self) -> "QSize":
        """Preferred size for a ~360 px wide side panel."""
        return QSize(360, self.preferred_height(360))

    def minimumSizeHint(self) -> "QSize":
        """Smallest useful size."""
        return QSize(220, self._MIN_HEIGHT)

    # ------------------------------------------------------------------ API

    def set_value(self, s_units: float, peak_s_units: Optional[float] = None) -> None:
        """Set meter value in S-units (above 9, 1 S-unit = 6 dB over S9)."""
        self._active = True
        self._s_units = max(0.0, min(MAX_S_UNITS, float(s_units)))
        if peak_s_units is not None:
            self._peak_s_units = max(0.0, min(MAX_S_UNITS, float(peak_s_units)))
        self.update()

    def set_active(self, active: bool) -> None:
        """Show (True) or dim (False) the needles, e.g. when no data arrives."""
        if self._active != bool(active):
            self._active = bool(active)
            self.update()

    def set_show_peak(self, show: bool) -> None:
        """Show or hide the peak-hold needle."""
        self._show_peak = bool(show)
        self.update()

    def set_placeholder(self, title: str, hint: str = "") -> None:
        """Text drawn on the face while the meter is inactive (no reading)."""
        if (title, hint) != self._placeholder:
            self._placeholder = (title, hint)
            if not self._active:
                self.update()

    def placeholder(self) -> Tuple[str, str]:
        return self._placeholder

    def set_noise(self, s_units: Optional[float]) -> None:
        """Mark the channel's noise level on the scale (None hides it)."""
        value = None if s_units is None else max(0.0, min(MAX_S_UNITS, float(s_units)))
        if value != self._noise_s_units:
            self._noise_s_units = value
            self.update()

    # ------------------------------------------------------------- geometry

    @staticmethod
    def _s_to_fraction(s_units: float) -> float:
        """Position of an S-unit value along the arc, 0 (left) to 1 (right)."""
        s_units = max(0.0, min(MAX_S_UNITS, s_units))
        if s_units <= 9.0:
            return s_units / 9.0 * _S9_FRACTION
        over_db = (s_units - 9.0) * DB_PER_S_UNIT
        return _S9_FRACTION + over_db / MAX_OVER_S9_DB * (1.0 - _S9_FRACTION)

    def _s_to_angle(self, s_units: float) -> float:
        """Convert S-units to a display angle in degrees (90 = straight up)."""
        f = self._s_to_fraction(s_units)
        return 90.0 + self._HALF_SWEEP - f * 2.0 * self._HALF_SWEEP

    @staticmethod
    def _over_to_s(db_over: float) -> float:
        return 9.0 + db_over / DB_PER_S_UNIT

    def _label_font(self, width: Optional[float] = None) -> "QFont":
        width = self.width() if width is None else width
        font = QFont(self.font())
        font.setPixelSize(max(9, min(13, int(width / 30))))
        font.setBold(True)
        return font

    def _gap(self, fm: "QFontMetricsF") -> float:
        """Space between the scale band and its labels."""
        return max(3.0, fm.height() * 0.25)

    @staticmethod
    def _band_width(width: float) -> float:
        """Thickness of the colored zone band."""
        return max(3.5, min(7.0, width * 0.012))

    @staticmethod
    def _label_extent(angle_deg: float, text_w: float, text_h: float) -> float:
        """How far a label box reaches along the radial direction at an angle."""
        a = math.radians(angle_deg)
        return text_w / 2.0 * abs(math.cos(a)) + text_h / 2.0 * abs(math.sin(a))

    def _label_radius(
        self, radius: float, angle_deg: float, text_w: float, fm: "QFontMetricsF"
    ) -> float:
        """Radius of a label's centre so its box clears the band by ``_gap``."""
        base = radius + self._band_width(self.width()) / 2.0 + self._gap(fm)
        return base + self._label_extent(angle_deg, text_w, fm.height())

    def _radius_for_width(self, width: float, fm: "QFontMetricsF") -> float:
        """Largest scale radius whose end labels fit inside ``width``."""
        end_angle = 90.0 - self._HALF_SWEEP
        lw = fm.horizontalAdvance("+60")
        base = self._band_width(width) / 2.0 + self._gap(fm)
        extent = self._label_extent(end_angle, lw, fm.height())
        cos_e = math.cos(math.radians(end_angle))
        radius = (width / 2.0 - self._MARGIN - lw / 2.0) / cos_e - base - extent
        return max(20.0, radius)

    def _squeeze(self, height: Optional[float]) -> float:
        """0 for a roomy face (or an unknown height), rising to 1 at the
        minimum height."""
        if height is None:
            return 0.0
        span = self._ROOMY_HEIGHT - self._MIN_HEIGHT
        return max(0.0, min(1.0, (self._ROOMY_HEIGHT - height) / span))

    def _v_margin(self, squeeze: float) -> float:
        """Space above the top label and below the legends."""
        return self._MARGIN - (self._MARGIN - self._TIGHT_MARGIN) * squeeze

    def _needed_height(
        self,
        radius: float,
        fm: "QFontMetricsF",
        width: Optional[float] = None,
        height: Optional[float] = None,
    ) -> float:
        """Face height for a scale of ``radius``: labels, arc and legends.

        ``height`` is the face's actual height, when known: a short face uses
        the tighter layout (see ``_squeeze``).
        """
        width = self.width() if width is None else width
        squeeze = self._squeeze(height)
        roomy = 1.0 - squeeze
        margin = self._v_margin(squeeze)
        cos_h = math.cos(math.radians(self._HALF_SWEEP))
        return (
            margin
            + fm.height()  # top label
            + self._gap(fm)
            + self._band_width(width) / 2.0
            + radius * (1.0 - cos_h)  # arc rise between its ends and its top
            # Room under the arc ends for the S / dB legends and enough of
            # the needle that its angle reads at a glance. Kept small so a
            # short, wide face still gets a full-width scale; a squeezed
            # face keeps just the legends.
            + max(fm.height() * (1.0 + 0.8 * roomy), radius * 0.18 * roomy)
            + margin
        )

    def _geometry(self, fm: "QFontMetricsF") -> Tuple[float, float, float]:
        """Return (pivot_x, pivot_y, scale_radius).

        The pivot sits below the face, as on a real rig's meter, so the scale
        is a wide, shallow arc and the needle rises from the bottom edge.
        """
        w, h = float(self.width()), float(self.height())
        radius = self._radius_for_width(w, fm)
        needed = self._needed_height(radius, fm, height=h)
        while needed > h + 0.5 and radius > 20.0:
            # Squeezed vertically: shrink the scale until it fits.
            radius = max(20.0, radius * (h / needed))
            needed = self._needed_height(radius, fm, height=h)
        top = (h - needed) / 2.0 + self._v_margin(self._squeeze(h))
        pivot_y = top + fm.height() + self._gap(fm) + self._band_width(w) / 2.0 + radius
        return w / 2.0, pivot_y, radius

    # ---------------------------------------------------------------- paint

    def paintEvent(self, event) -> None:
        """Paint the meter from the active palette."""
        p = get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Face
        face = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        face_path = QPainterPath()
        face_path.addRoundedRect(face, 6, 6)
        painter.fillPath(face_path, p.qcolor("plot_bg"))
        painter.setClipPath(face_path)

        font = self._label_font()
        painter.setFont(font)
        fm = QFontMetricsF(font)
        cx, cy, radius = self._geometry(fm)

        self._draw_scale(painter, p, cx, cy, radius, fm)
        if self._active and self._noise_s_units is not None:
            self._draw_noise(painter, p, cx, cy, radius)

        if self._show_peak and self._active and self._peak_s_units > self._s_units:
            self._draw_needle(
                painter,
                cx,
                cy,
                self._peak_s_units,
                p.qcolor("plot_marker"),
                radius,
                1.6,
                start=radius * 0.72,
            )

        needle_color = p.qcolor("text") if self._active else p.qcolor("disabled")
        self._draw_needle(
            painter,
            cx,
            cy,
            self._s_units if self._active else 0.0,
            needle_color,
            radius,
            4.0,
        )

        if not self._active and self._placeholder[0]:
            self._draw_placeholder(painter, p, face)

        painter.setClipping(False)
        painter.setPen(QPen(p.qcolor("surface1"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(face_path)
        painter.end()

    def _draw_placeholder(self, painter, p, face) -> None:
        """Empty-state text, in the style of the plots' empty states."""
        # Imported here: sdr_module.gui imports this module while it
        # initialises (see _themes()).
        from ...gui.spectrum_widget import draw_placeholder

        font = QFont(self.font())
        if font.pointSizeF() <= 0 < font.pixelSize():
            font.setPointSizeF(font.pixelSize() * 72.0 / max(1, self.logicalDpiY()))
        title, hint = self._placeholder
        draw_placeholder(painter, QRectF(face), title, hint, font, p)

    def _draw_noise(self, painter, p, cx, cy, radius) -> None:
        """A muted band inside the scale from S0 up to the noise level."""
        noise = self._noise_s_units
        if noise is None or noise <= 0.05:
            return
        band_w = self._band_width(self.width())
        inner = radius - band_w * 1.6
        rect = QRectF(cx - inner, cy - inner, inner * 2, inner * 2)
        a0 = self._s_to_angle(0.0)
        a1 = self._s_to_angle(noise)
        pen = QPen(p.qcolor("caption", 150), band_w * 1.2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(rect, int(a0 * 16), int((a1 - a0) * 16))

    def _zones(self) -> List[Tuple[float, float, str]]:
        """Colored scale zones as (from_s, to_s, palette token)."""
        return [
            (0.0, 5.0, "surface2"),
            (5.0, 9.0, "success"),
            (9.0, self._over_to_s(40.0), "warning"),
            (self._over_to_s(40.0), MAX_S_UNITS, "danger"),
        ]

    def _ticks(self) -> List[Tuple[float, str, bool]]:
        """(s_units, label, is_major) for every S-unit and every 10 dB over S9."""
        ticks: List[Tuple[float, str, bool]] = []
        for s in range(0, 10):
            ticks.append((float(s), str(s) if s else "", s % 2 == 1))
        for db in range(10, int(MAX_OVER_S9_DB) + 1, 10):
            ticks.append((self._over_to_s(db), f"+{db}", db % 20 == 0))
        return ticks

    def _scale_labels(
        self, fm: "QFontMetricsF", cx: float, cy: float, radius: float
    ) -> List[Tuple[str, "QRectF", bool]]:
        """(text, box, is_over_s9) for each scale label that is drawn.

        Every tick is labelled when there is room; otherwise only the major
        ones (odd S-units and +20/+40/+60), like a real rig's meter.
        """
        step_px = radius * math.radians(2 * self._HALF_SWEEP) / 15.0
        label_all = step_px >= fm.horizontalAdvance("+60") + 8
        labels = []
        for s, label, is_major in self._ticks():
            if not label or not (is_major or label_all):
                continue
            angle_deg = self._s_to_angle(s)
            angle = math.radians(angle_deg)
            tw = fm.horizontalAdvance(label)
            r_lab = self._label_radius(radius, angle_deg, tw, fm)
            lx, ly = cx + r_lab * math.cos(angle), cy - r_lab * math.sin(angle)
            box = QRectF(lx - tw / 2 - 1, ly - fm.height() / 2, tw + 2, fm.height())
            labels.append((label, box, s > 9.0))
        return labels

    def _draw_scale(self, painter, p, cx, cy, radius, fm) -> None:
        """Draw the zone band, ticks, labels and the S / dB legends."""
        band_w = self._band_width(self.width())
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # Zone band along the arc (QPainter angles: degrees * 16, CCW).
        for s_from, s_to, token in self._zones():
            a0 = self._s_to_angle(s_from)
            a1 = self._s_to_angle(s_to)
            pen = QPen(p.qcolor(token), band_w)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawArc(arc_rect, int(a0 * 16), int((a1 - a0) * 16))

        major = max(6.0, radius * 0.08)
        minor = major * 0.55
        inner_edge = radius - band_w / 2.0

        for s, _label, is_major in self._ticks():
            over = s > 9.0
            angle = math.radians(self._s_to_angle(s))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            length = major if (is_major or s == 9.0) else minor
            tick_color = p.qcolor("danger" if over else "plot_axis")
            painter.setPen(QPen(tick_color, 1.6 if is_major else 1.0))
            painter.drawLine(
                QPointF(cx + inner_edge * cos_a, cy - inner_edge * sin_a),
                QPointF(
                    cx + (inner_edge - length) * cos_a,
                    cy - (inner_edge - length) * sin_a,
                ),
            )

        for label, box, over in self._scale_labels(fm, cx, cy, radius):
            painter.setPen(p.qcolor("danger" if over else "plot_text"))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

        # Legends just outside and below the two ends of the arc.
        legend_font = QFont(painter.font())
        legend_font.setPixelSize(max(9, painter.font().pixelSize() - 2))
        painter.setFont(legend_font)
        lfm = QFontMetricsF(legend_font)
        end = math.radians(self._HALF_SWEEP)
        ex = radius * math.sin(end)
        top = cy - radius * math.cos(end) + band_w
        painter.setPen(p.qcolor("plot_axis"))
        lh = lfm.height()
        left_w = lfm.horizontalAdvance("S") + 2
        right_w = lfm.horizontalAdvance("dB") + 2
        painter.drawText(
            QRectF(cx - ex - left_w - 2, top, left_w, lh),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            "S",
        )
        painter.drawText(
            QRectF(cx + ex + 2, top, right_w, lh),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "dB",
        )
        painter.setFont(self._label_font())

    def _draw_needle(
        self,
        painter: "QPainter",
        cx: float,
        cy: float,
        s_units: float,
        color,
        radius: float,
        width: float,
        start: float = 0.0,
    ) -> None:
        """Draw a tapered needle from the pivot (or ``start``) past the arc."""
        angle = math.radians(self._s_to_angle(s_units))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Perpendicular unit vector (screen coordinates, y down).
        nx, ny = sin_a, cos_a
        length = radius + 3.0
        base_half = width / 2.0
        tip_half = max(0.5, width / 5.0)

        bx, by = cx + start * cos_a, cy - start * sin_a
        tx, ty = cx + length * cos_a, cy - length * sin_a
        path = QPainterPath()
        path.moveTo(bx + nx * base_half, by + ny * base_half)
        path.lineTo(tx + nx * tip_half, ty + ny * tip_half)
        path.lineTo(tx - nx * tip_half, ty - ny * tip_half)
        path.lineTo(bx - nx * base_half, by - ny * base_half)
        path.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(path)


class LevelSparkline(QWidget if HAS_PYQT6 else object):
    """The last ``HISTORY_S`` seconds of the signal level: fading (QSB) at
    a glance. The noise level is a dashed line; colours come from the theme
    palette at paint time."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._points: Deque[Tuple[float, float]] = deque()
        self._noise: Optional[float] = None
        self.setMinimumSize(80, 22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Signal level history")
        self.setToolTip(
            f"Signal level over the last {HISTORY_S:.0f} seconds (fading at a "
            "glance); the dashed line is the noise in the channel when the "
            "signal is near it. Cleared when you retune."
        )

    def sizeHint(self) -> "QSize":
        return QSize(200, 24)

    def add(self, when: float, dbm: float, noise_dbm: Optional[float] = None) -> None:
        self._points.append((float(when), float(dbm)))
        while self._points and when - self._points[0][0] > HISTORY_S:
            self._points.popleft()
        self._noise = noise_dbm
        self.update()

    def clear(self) -> None:
        self._points.clear()
        self._noise = None
        self.update()

    def values(self) -> List[float]:
        return [v for _t, v in self._points]

    def stats(self) -> Optional[Tuple[float, float, float]]:
        """``(min, mean, max)`` dBm of the history (None when empty)."""
        values = self.values()
        if not values:
            return None
        return min(values), float(np.mean(values)), max(values)

    def paintEvent(self, event) -> None:
        p = get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        painter.fillPath(path, p.qcolor("plot_bg"))
        painter.setPen(QPen(p.qcolor("surface1"), 1))
        painter.drawPath(path)
        points = list(self._points)
        if len(points) >= 2:
            values = [v for _t, v in points]
            low, high = min(values), max(values)
            if self._noise is not None and self._noise > low - 12.0:
                low = min(low, self._noise)  # noise near the trace: show it
            span = max(high - low, 12.0)
            mid = 0.5 * (high + low)
            low, high = mid - span / 2 - 1.0, mid + span / 2 + 1.0
            inner = rect.adjusted(3, 3, -3, -3)
            now = points[-1][0]

            def xy(t: float, v: float) -> QPointF:
                x = inner.right() - (now - t) / HISTORY_S * inner.width()
                y = inner.bottom() - (v - low) / (high - low) * inner.height()
                return QPointF(x, y)

            if self._noise is not None and low <= self._noise <= high:
                pen = QPen(p.qcolor("caption"), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                y = xy(now, self._noise).y()
                painter.drawLine(QPointF(inner.left(), y), QPointF(inner.right(), y))
            painter.setPen(QPen(p.qcolor("plot_trace"), 1.4))
            painter.drawPolyline(QPolygonF([xy(t, v) for t, v in points]))
        painter.end()


class _ReportModeCombo(QComboBox if HAS_PYQT6 else object):
    """A combo whose closed face can show a short text (``Auto · CW``) while
    its list shows the full item text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._face: Optional[str] = None

    def set_face_text(self, text: Optional[str]) -> None:
        self._face = text
        self.update()

    def face_text(self) -> str:
        return self._face or self.currentText()

    def sizeHint(self) -> "QSize":
        hint = super().sizeHint()
        texts = [self.itemText(i) for i in range(1, self.count())] + ["Auto · Digital"]
        fm = self.fontMetrics()
        widest = max(fm.horizontalAdvance(t) for t in texts)
        full = fm.horizontalAdvance(self.itemText(0)) if self.count() else 0
        return QSize(max(hint.width() - full + widest, 40), hint.height())

    def minimumSizeHint(self) -> "QSize":
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.face_text()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)


class _ClickableLabel(QLabel if HAS_PYQT6 else object):
    """A label that emits ``clicked`` (PEAK HOLD: click to reset)."""

    if HAS_PYQT6:
        clicked = pyqtSignal()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class SignalMeterPanel(QWidget if HAS_PYQT6 else object):
    """
    Complete signal meter panel with analog display and digital readouts.

    Shows:
    - Analog S-meter (with the noise level marked)
    - The level over the last 30 seconds, with min / avg / max
    - Digital S-meter reading, RST report and verbal report
    - Power (dBFS until calibrated, then dBm), SNR and peak hold

    Feed it the tuned channel with :meth:`update_samples` (and the channel's
    noise power, measured around the channel, for a true SNR); tell it when
    the receiver starts and stops with :meth:`set_receiving`, and call
    :meth:`reset_peak` after a retune or a mode change.
    """

    if HAS_PYQT6:
        # (calibration offset dB or None for uncalibrated, reference gain or
        # None): the window stores it per device type.
        calibration_changed = pyqtSignal(object, object)
        # A report was copied to the clipboard (its text).
        report_copied = pyqtSignal(str)

    _MODES = [
        (
            "Phone",
            SignalMode.PHONE,
            "Phone (SSB, AM, FM): RS report, 2 digits, e.g. 59",
        ),
        ("CW", SignalMode.CW, "CW (Morse): RST report with a tone digit, e.g. 599"),
        ("Digital", SignalMode.DIGITAL, "Digital modes: RS report, 2 digits"),
    ]
    AUTO_TEXT = "Auto (follow demodulation)"
    _AUTO_TIP = (
        "Report format follows Demodulation > Mode: CW gives an RST report "
        "with a tone digit (599); SSB, AM and FM give RS (59); I/Q gives the "
        "digital-mode RS"
    )

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._meter = SignalMeter()
        self._history = SignalHistory()
        self._last_reading: Optional[SignalReading] = None
        self._showing_data = False
        self._receiving: Optional[bool] = None  # None: not told (standalone)
        self._has_device = True
        self._demod_mode = ""
        self._frequency_source: Optional[Callable[[], float]] = None
        self._compact = False
        self._empty_hint = ""
        self._empty_short = ""
        self._roomy_min = 0  # height the full layout needs (px)
        self._watched: Optional[QWidget] = None
        self._last_history_time = 0.0

        self._setup_ui()

        # Update timer
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(100)  # 10 Hz update

    # ------------------------------------------------------------- building

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Analog meter
        self._analog_meter = AnalogMeterWidget()
        layout.addWidget(self._analog_meter)

        # Level history: sparkline and its min / avg / max.
        self._history_row = QWidget()
        history_layout = QHBoxLayout(self._history_row)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        self._sparkline = LevelSparkline()
        history_layout.addWidget(self._sparkline, 1)
        self._stats_label = set_role(QLabel(_EMPTY), "hint")
        self._stats_label.setToolTip(
            f"Lowest, average and highest level over the last {HISTORY_S:.0f} s"
        )
        history_layout.addWidget(self._stats_label)
        layout.addWidget(self._history_row)

        # Digital readouts: one grid so the three columns line up.
        #   S-METER   RST    REPORT
        #   S9+10     59     [Auto · Phone]
        #   Five and nine, 10 over   [Copy] [Calibrate]
        #   POWER     SNR    PEAK HOLD
        #   -33 dBFS  41 dB  S9+12
        readout_group = QGroupBox("Signal Report")
        self._readout_group = readout_group
        grid = QGridLayout(readout_group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(0)
        self._grid = grid

        def caption(text: str, tip: str) -> "QLabel":
            lbl = set_role(QLabel(text), "caption")
            lbl.setToolTip(tip)
            return lbl

        def readout(role: str, tip: str) -> "QLabel":
            lbl = set_role(QLabel(_EMPTY), role)
            lbl.setToolTip(tip)
            return lbl

        s_tip = (
            "Signal strength: S1 to S9 in 6 dB steps (S9 = -73 dBm), then dB "
            "over S9. Uncalibrated, it assumes 0 dBFS = -30 dBm."
        )
        rst_tip = (
            "RST report: Readability (1-5, from the signal-to-noise ratio), "
            "Strength (1-9) and, for CW, Tone (1-9). Shown once a signal "
            "stands above the noise."
        )
        mode_tip = (
            "Report format: phone and digital use RS (2 digits), CW adds a "
            "tone digit (RST, 3 digits). Auto follows Demodulation > Mode."
        )
        snr_tip = (
            "Signal-to-noise ratio: the signal's power over the noise in the "
            "channel, measured from the spectrum around it"
        )
        peak_tip = (
            f"Strongest recent reading: held {PEAK_HOLD_S:.0f} s, then falls "
            "back. Click to reset; it also resets when you retune or stop."
        )
        self._s_tip = s_tip

        self._s_caption = caption("S-METER", s_tip)
        grid.addWidget(self._s_caption, 0, 0)
        self._rst_caption = caption("RST", rst_tip)
        grid.addWidget(self._rst_caption, 0, 1)
        self._mode_caption = caption("REPORT", mode_tip)
        grid.addWidget(self._mode_caption, 0, 2)

        self._s_meter_label = readout("display", s_tip)
        grid.addWidget(self._s_meter_label, 1, 0)
        self._rst_label = readout("display", rst_tip)
        grid.addWidget(self._rst_label, 1, 1)

        self._mode_combo = _ReportModeCombo()
        self._mode_combo.addItem(self.AUTO_TEXT)
        self._mode_combo.setItemData(0, self._AUTO_TIP, Qt.ItemDataRole.ToolTipRole)
        for index, (text, _mode, tip) in enumerate(self._MODES, start=1):
            self._mode_combo.addItem(text)
            self._mode_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._mode_combo.setToolTip(mode_tip)
        self._mode_combo.setAccessibleName("Report mode")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(
            self._mode_combo,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        # Verbal report (or the empty-state hint), with Copy and Calibrate.
        self._verbal_label = QLabel()
        # No word wrap: a wrapping label would make the whole panel
        # height-for-width, so a short tab could not shrink the gauge.
        self._verbal_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._verbal_label.setToolTip("The report as you would say it on the air")
        self._verbal_label.setContentsMargins(0, 2, 0, 0)
        grid.addWidget(self._verbal_label, 2, 0, 1, 2)

        self._tools_box = QWidget()
        tools = QHBoxLayout(self._tools_box)
        tools.setContentsMargins(0, 2, 0, 0)
        tools.setSpacing(4)
        self._copy_btn = self._build_copy_button()
        tools.addWidget(self._copy_btn)
        self._cal_btn = self._build_calibration_button()
        tools.addWidget(self._cal_btn)
        tools.addStretch(1)
        grid.addWidget(self._tools_box, 2, 2)
        grid.setRowMinimumHeight(3, 8)  # gap before the measurement row

        self._power_caption = caption("POWER", "")
        grid.addWidget(self._power_caption, 4, 0)
        self._snr_caption = caption("SNR", snr_tip)
        grid.addWidget(self._snr_caption, 4, 1)
        self._peak_caption = caption("PEAK HOLD", peak_tip)
        grid.addWidget(self._peak_caption, 4, 2)
        self._dbm_label = readout("value", "")
        grid.addWidget(self._dbm_label, 5, 0)
        self._snr_label = readout("value", snr_tip)
        grid.addWidget(self._snr_label, 5, 1)
        self._peak_label = _ClickableLabel(_EMPTY)
        set_role(self._peak_label, "value")
        self._peak_label.setToolTip(peak_tip)
        self._peak_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._peak_label.clicked.connect(self.reset_peak)
        grid.addWidget(self._peak_label, 5, 2)

        # S-meter strings ("S9+30") are the widest; RST at most 3 digits.
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 4)

        layout.addWidget(readout_group)

        # Compact layout (short tab): the S-meter and RST readouts and the
        # report format move into one row, with power / SNR / peak on one
        # line under it (see set_compact).
        self._compact_box = QWidget()
        compact = QVBoxLayout(self._compact_box)
        compact.setContentsMargins(4, 0, 4, 0)
        compact.setSpacing(2)
        self._compact_row = QHBoxLayout()
        self._compact_row.setSpacing(12)
        self._compact_row.addStretch(1)
        compact.addLayout(self._compact_row)
        self._summary_label = set_role(QLabel(), "muted")
        self._summary_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._summary_label.setToolTip(
            f"Power, SNR and peak hold. Right-click to copy the report. {peak_tip}"
        )
        compact.addWidget(self._summary_label)
        # Added when there is room: the spoken report with Copy / Calibrate.
        self._compact_extra = QWidget()
        self._compact_extra_row = QHBoxLayout(self._compact_extra)
        self._compact_extra_row.setContentsMargins(0, 0, 0, 0)
        self._compact_extra_row.setSpacing(6)
        compact.addWidget(self._compact_extra)
        self._compact_box.hide()
        layout.addWidget(self._compact_box)
        layout.addStretch(1)

        for label in (
            self._s_meter_label,
            self._rst_label,
            self._verbal_label,
            self._dbm_label,
            self._snr_label,
            self._summary_label,
        ):
            label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            label.customContextMenuRequested.connect(
                lambda pos, w=label: self._copy_menu().exec(w.mapToGlobal(pos))
            )

        self._update_units()
        self._update_mode_face()
        self._show_empty_state("initial")

    def _build_copy_button(self) -> "QToolButton":
        button = QToolButton()
        button.setText("Copy")
        button.setToolTip(
            "Copy the signal report for a log or chat, e.g. "
            "'RST 59 · S9+20 · 14.250 MHz · 12:34 UTC'. The arrow offers RST "
            "only and a contest report."
        )
        button.setAccessibleName("Copy signal report")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        button.setMenu(self._copy_menu())
        button.clicked.connect(self.copy_report)
        return button

    def _copy_menu(self) -> "QMenu":
        menu = QMenu(self)
        menu.addAction("Copy Report", self.copy_report)
        menu.addAction("Copy RST", self.copy_rst)
        menu.addAction("Copy Contest Report (59/599)", self.copy_contest_report)
        return menu

    def _build_calibration_button(self) -> "QToolButton":
        button = QToolButton()
        button.setText("Calibrate")
        button.setAccessibleName("S-meter calibration")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self)
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(6)
        row.addWidget(QLabel("Offset"))
        self._cal_spin = QDoubleSpinBox()
        self._cal_spin.setRange(-150.0, 150.0)
        self._cal_spin.setDecimals(1)
        self._cal_spin.setSingleStep(0.5)
        self._cal_spin.setSuffix(" dB")
        self._cal_spin.setAccessibleName("Calibration offset")
        self._cal_spin.setToolTip(
            "Added to the uncalibrated reading (0 dBFS = -30 dBm) for this "
            "device type; saved for next time"
        )
        self._cal_spin.editingFinished.connect(self._on_cal_spin_done)
        row.addWidget(self._cal_spin)
        spin_action = QWidgetAction(menu)
        spin_action.setDefaultWidget(holder)
        menu.addAction(spin_action)
        self._cal_s9_action = menu.addAction(
            "Current Signal Is S9 (-73 dBm)", lambda: self.calibrate_to(S9_DBM)
        )
        self._cal_clear_action = menu.addAction(
            "Use Uncalibrated Readings", self.clear_calibration
        )
        menu.aboutToShow.connect(self._sync_cal_menu)
        button.setMenu(menu)
        self._cal_menu = menu
        return button

    # ----------------------------------------------------------- public API

    def get_meter(self) -> SignalMeter:
        """Get the signal meter instance."""
        return self._meter

    def update_samples(
        self, samples: np.ndarray, noise_dbfs: Optional[float] = None
    ) -> None:
        """Update with new I/Q samples of the tuned channel.

        Args:
            samples: The channel's complex samples.
            noise_dbfs: Noise power in the channel (dBFS), measured around
                it; without it the meter estimates the noise itself.
        """
        reading = self._meter.update(samples, noise_dbfs)
        self._last_reading = reading
        self._history.add(reading)

    def set_receiving(self, running: bool, has_device: bool = True) -> None:
        """The receiver started or stopped: show an empty state at once
        (not a stale reading) and forget the peak."""
        running = bool(running)
        changed = (running, bool(has_device)) != (self._receiving, self._has_device)
        self._receiving = running
        self._has_device = bool(has_device)
        if running and not changed:
            return
        if not running:
            self._last_reading = None
            self._meter.reset_peak()
            self._sparkline.clear()
            self._stats_label.setText(_EMPTY)
            self._show_empty_state("stopped" if has_device else "no_device")
        elif changed and self._last_reading is None:
            self._show_empty_state("waiting")

    def _clear_history(self) -> None:
        self._sparkline.clear()
        self._stats_label.setText(_EMPTY)

    def reset_peak(self) -> None:
        """Forget the peak hold (and the averaging and history): after a
        retune, a mode change, or a click on PEAK HOLD."""
        self._meter.reset_peak()
        self._clear_history()
        reading = self._last_reading
        if reading is not None:
            reading.peak_hold_dbm = reading.power_dbm
            if self._showing_data:
                self._peak_label.setText(
                    format_s_units(dbm_to_s_units(reading.power_dbm))
                )

    def set_demod_mode(self, mode: str) -> None:
        """Demodulation > Mode, for the Auto report format."""
        self._demod_mode = str(mode or "")
        self._apply_report_mode()

    def set_frequency_source(self, source: Optional[Callable[[], float]]) -> None:
        """Where the copied report gets the tuned frequency from."""
        self._frequency_source = source

    def set_gain(self, gain_db: Optional[float]) -> None:
        """The receiver's RF gain (None if unknown): a calibrated reading
        stays put when the gain changes."""
        self._meter.set_gain(gain_db)

    def set_calibration(
        self, offset_db: Optional[float], reference_gain_db: Optional[float] = None
    ) -> None:
        """Apply a stored calibration (None: uncalibrated). Does not emit
        ``calibration_changed``."""
        if offset_db is None:
            self._meter.clear_calibration()
        else:
            self._meter.set_calibration(float(offset_db), reference_gain_db)
        self._clear_history()
        self._update_units()

    def is_calibrated(self) -> bool:
        return self._meter.is_calibrated

    def calibrate_to(self, dbm: float) -> bool:
        """Calibrate so the current reading is ``dbm`` (e.g. S9 = -73 dBm).
        Returns False when there is no reading to calibrate against."""
        if self._last_reading is None:
            return False
        offset = self._meter.offset_for_level(dbm)
        self._apply_calibration(offset)
        return True

    def clear_calibration(self) -> None:
        self._meter.clear_calibration()
        self._clear_history()
        self._update_units()
        self.calibration_changed.emit(None, None)

    def get_qso_report(self) -> str:
        """Get report for QSO logging."""
        return self._meter.get_qso_report()

    def get_contest_report(self) -> str:
        """Get contest-style report."""
        return self._meter.get_contest_report()

    def report_text(self) -> Optional[str]:
        """The report to copy: 'RST 59 · S9+20 · 14.250 MHz · 12:34 UTC'
        (None while there is no reading)."""
        if not self._showing_data:
            return None
        frequency = None
        if self._frequency_source is not None:
            try:
                frequency = float(self._frequency_source())
            except Exception:  # pragma: no cover - defensive
                frequency = None
        return self._meter.get_log_report(frequency)

    def copy_report(self) -> Optional[str]:
        return self._copy(self.report_text())

    def copy_rst(self) -> Optional[str]:
        reading = self._last_reading
        if not self._showing_data or reading is None:
            return self._copy(None)
        if not reading.signal_present:
            return self._copy(None, "No signal above the noise: no RST to copy")
        return self._copy(self._meter.get_rst())

    def copy_contest_report(self) -> Optional[str]:
        return self._copy(self._meter.get_contest_report())

    def is_compact(self) -> bool:
        return self._compact

    # -------------------------------------------------------------- display

    def _copy(self, text: Optional[str], why: str = "") -> Optional[str]:
        if not text:
            self.report_copied.emit(why or "Nothing to copy yet: no reading")
            return None
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.report_copied.emit(f"Copied: {text}")
        return text

    def _apply_calibration(self, offset: float) -> None:
        gain = self._meter._gain_db
        self._meter.set_calibration(offset, gain)
        self._clear_history()
        self._update_units()
        self.calibration_changed.emit(float(offset), gain)

    def _on_cal_spin_done(self) -> None:
        value = float(self._cal_spin.value())
        if (
            self._meter.is_calibrated
            and abs(value - self._meter.calibration_offset_db) < 1e-6
        ):
            return
        if not self._meter.is_calibrated and value == 0.0:
            return
        self._apply_calibration(value)

    def _sync_cal_menu(self) -> None:
        self._cal_spin.blockSignals(True)
        self._cal_spin.setValue(
            self._meter.calibration_offset_db if self._meter.is_calibrated else 0.0
        )
        self._cal_spin.blockSignals(False)
        self._cal_s9_action.setEnabled(self._last_reading is not None)
        self._cal_clear_action.setEnabled(self._meter.is_calibrated)

    def _update_units(self) -> None:
        """POWER reads dBFS until calibrated, dBm after."""
        calibrated = self._meter.is_calibrated
        if calibrated:
            offset = self._meter.calibration_offset_db
            power_tip = (
                f"Input power in dBm, calibrated (offset {offset:+.1f} dB); it "
                "allows for RF gain changes when the gain is known"
            )
            self._cal_btn.setText("Calibrated")
            self._cal_btn.setToolTip(
                f"S-meter calibrated: offset {offset:+.1f} dB. Click to change it."
            )
            self._power_caption.setText("POWER")
            self._analog_meter.setToolTip(
                "Signal strength in S-units (6 dB each, S9 = -73 dBm, "
                "calibrated), then dB over S9. The thin needle shows the peak "
                "hold; the shaded start of the scale is the noise in the channel."
            )
        else:
            power_tip = (
                "Power in the channel relative to the SDR's full scale (dBFS). "
                "Uncalibrated: S-units take 0 dBFS as -30 dBm (~dBm). Use "
                "Calibrate for real dBm."
            )
            self._cal_btn.setText("Calibrate")
            self._cal_btn.setToolTip(
                "Uncalibrated: readings are dBFS, S-units assume 0 dBFS = -30 "
                "dBm (~dBm). Calibrate with a known signal or an offset."
            )
            self._power_caption.setText("POWER (UNCAL.)")
            self._analog_meter.setToolTip(
                "Signal strength in S-units (6 dB each, S9 = -73 dBm), then dB "
                "over S9. Uncalibrated: ~dBm, from 0 dBFS = -30 dBm. The thin "
                "needle shows the peak hold; the shaded start of the scale is "
                "the noise in the channel."
            )
        self._power_caption.setToolTip(power_tip)
        self._dbm_label.setToolTip(power_tip)
        self._s_meter_label.setToolTip(
            self._s_tip if not calibrated else self._s_tip.split(". Uncal")[0]
        )
        if self._showing_data and self._last_reading is not None:
            self._update_display()

    def _show_empty_state(self, state: str) -> None:
        """Clear the readouts and say why: ``state`` is a key of
        ``_EMPTY_STATES`` (or a hint of its own). The gauge shows the title
        and hint, the report row a short line, the compact layout's one
        line the whole sentence."""
        title, gauge_hint, short, sentence = _EMPTY_STATES.get(
            state, (state, "", state, state)
        )
        self._showing_data = False
        for lbl in (self._s_meter_label, self._rst_label):
            lbl.setText(_EMPTY)
            set_tone(lbl, "muted")
        if self._compact:
            self._rst_label.setText(f"RST {_EMPTY}")
        for lbl in (self._dbm_label, self._snr_label, self._peak_label):
            lbl.setText(_EMPTY)
        self._empty_hint = sentence
        self._empty_short = short
        self._summary_label.setText(sentence)
        # (In the compact layout the summary line carries the hint.)
        self._verbal_label.setText("" if self._compact else short)
        set_role(self._verbal_label, "hint", None)
        self._analog_meter.set_placeholder(title, gauge_hint)
        self._analog_meter.set_active(False)
        self._analog_meter.set_noise(None)
        self._copy_btn.setEnabled(False)

    def _update_display(self) -> None:
        """Update display elements."""
        reading = self._last_reading
        if reading is None:
            return
        if time.time() - reading.timestamp > _STALE_AFTER_S:
            if self._showing_data:
                self._show_empty_state("stale")
            return

        if not self._showing_data:
            self._showing_data = True
            set_role(self._verbal_label, "muted", None)
            self._copy_btn.setEnabled(True)
        reading = self._meter.get_reading()
        self._last_reading.peak_hold_dbm = reading.peak_hold_dbm

        # Analog meter: position from dBm so the needle can reach S9+60.
        s_units = dbm_to_s_units(reading.power_dbm)
        peak_s = dbm_to_s_units(reading.peak_hold_dbm)
        self._analog_meter.set_value(s_units, peak_s)
        self._analog_meter.set_noise(
            None if reading.noise_dbm is None else dbm_to_s_units(reading.noise_dbm)
        )

        # Digital readouts. The S reading uses the same 1 dB resolution as
        # the needle and PEAK HOLD (the core rounds to 10 dB above S9, which
        # could read below the peak); the spoken report keeps the 10 dB
        # steps that operators say on the air.
        signal = reading.signal_present
        s_text = format_s_units(s_units)
        self._s_meter_label.setText(s_text)
        set_tone(self._s_meter_label, strength_tone(s_units, signal))
        prefix = "RST " if self._compact else ""
        if signal:
            self._rst_label.setText(prefix + self._meter.get_rst())
            set_tone(self._rst_label, None)
            self._verbal_label.setText(self._meter.get_verbal_report())
        else:
            self._rst_label.setText(prefix + _EMPTY)
            set_tone(self._rst_label, "muted")
            self._verbal_label.setText("No signal above the noise")
        if reading.calibrated:
            power = f"{reading.power_dbm:.0f} dBm"
        else:
            power = f"{reading.power_dbfs:.0f} dBFS"
        self._dbm_label.setText(power)
        snr = (
            min(99.0, max(0.0, reading.snr_db)) if self._meter.has_noise_floor else None
        )
        snr_text = _EMPTY if snr is None else f"{snr:.0f} dB"
        self._snr_label.setText(snr_text)
        peak_text = format_s_units(peak_s)
        self._peak_label.setText(peak_text)
        self._summary_label.setText(f"{power} · SNR {snr_text} · peak {peak_text}")

        # Level history, ten points a second (not the old channel's samples
        # still arriving just after a retune).
        now = time.time()
        if now - self._last_history_time >= 0.095 and not self._meter.settling:
            self._last_history_time = now
            self._sparkline.add(now, reading.power_dbm, reading.noise_dbm)
            stats = self._sparkline.stats()
            if stats:
                low, avg, high = (format_s_units(dbm_to_s_units(v)) for v in stats)
                self._stats_label.setText(f"min {low} · avg {avg} · max {high}")
                unit = "dBm" if reading.calibrated else "~dBm"
                self._stats_label.setToolTip(
                    f"Last {HISTORY_S:.0f} s: min {stats[0]:.0f}, average "
                    f"{stats[1]:.0f}, max {stats[2]:.0f} {unit}"
                )

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode change."""
        self._apply_report_mode()

    def _resolved_mode(self) -> Tuple[str, SignalMode]:
        """(name, mode) of the report format in effect."""
        index = self._mode_combo.currentIndex()
        if 1 <= index <= len(self._MODES):
            text, mode, _tip = self._MODES[index - 1]
            return text, mode
        demod = self._demod_mode.upper()
        if demod == "CW":
            return "CW", SignalMode.CW
        if demod.startswith("NONE") or demod == "I/Q":
            return "Digital", SignalMode.DIGITAL
        return "Phone", SignalMode.PHONE

    def _apply_report_mode(self) -> None:
        _name, mode = self._resolved_mode()
        self._meter.set_mode(mode)
        self._update_mode_face()
        if self._showing_data and self._last_reading is not None:
            if self._last_reading.signal_present:
                prefix = "RST " if self._compact else ""
                self._rst_label.setText(prefix + self._meter.get_rst())

    def _update_mode_face(self) -> None:
        if self._mode_combo.currentIndex() == 0:
            name, _mode = self._resolved_mode()
            self._mode_combo.set_face_text(f"Auto · {name}")
        else:
            self._mode_combo.set_face_text(None)

    # ------------------------------------------------------------- layout

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._watch_viewport()
        self._update_compact()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_compact()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._watched and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._update_compact)
        return super().eventFilter(obj, event)

    def _watch_viewport(self) -> None:
        """Follow the scroll area's viewport (the panel itself never gets
        shorter than its minimum, so its own size can't say the tab is
        short)."""
        parent = self.parentWidget()
        if parent is not self._watched:
            if self._watched is not None:
                self._watched.removeEventFilter(self)
            self._watched = parent
            if parent is not None:
                parent.installEventFilter(self)

    def _available_height(self) -> int:
        parent = self.parentWidget()
        if parent is not None and parent.objectName() == "qt_scrollarea_viewport":
            return parent.height()
        return self.height()

    def _update_compact(self) -> None:
        """Compact whenever the full layout would need scrolling; in compact
        mode, the optional rows come back as far as they fit."""
        if not self._compact:
            self._roomy_min = self.minimumSizeHint().height()
        available = self._available_height()
        compact = 0 < available < self._roomy_min
        if compact != self._compact:
            self.set_compact(compact)
        if self._compact:
            self._fit_compact_extras(available)

    def _fit_compact_extras(self, available: int) -> None:
        """Show the report/tools row, then the level history, if they fit."""
        layout = self.layout()
        margins = layout.contentsMargins()
        spacing = layout.spacing()
        inner = self._compact_box.layout().spacing()
        core = (
            margins.top()
            + margins.bottom()
            + self._analog_meter.minimumSizeHint().height()
            + spacing
            + max(
                self._s_meter_label.sizeHint().height(),
                self._mode_combo.sizeHint().height(),
            )
            + inner
            + self._summary_label.sizeHint().height()
        )
        extra = self._compact_extra.sizeHint().height() + inner
        history = self._history_row.sizeHint().height() + spacing
        show_extra = available >= core + extra
        show_history = available >= core + (extra if show_extra else 0) + history
        if self._compact_extra.isHidden() == show_extra:
            self._compact_extra.setVisible(show_extra)
        if self._history_row.isHidden() == show_history:
            self._history_row.setVisible(show_history)

    def set_compact(self, compact: bool) -> None:
        """Compact layout for short tabs: the gauge, then the S-meter, RST
        and report format in one row and power / SNR / peak on one line;
        the spoken report with Copy / Calibrate and the level history come
        back as far as they fit (Copy also stays on the readouts'
        right-click menu)."""
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        movable = (
            self._s_meter_label,
            self._rst_label,
            self._mode_combo,
            self._verbal_label,
            self._tools_box,
        )
        for widget in movable:
            self._grid.removeWidget(widget)
            self._compact_row.removeWidget(widget)
            self._compact_extra_row.removeWidget(widget)
        if compact:
            self._compact_row.insertWidget(0, self._s_meter_label)
            self._compact_row.insertWidget(1, self._rst_label)
            self._compact_row.addWidget(
                self._mode_combo, 0, Qt.AlignmentFlag.AlignVCenter
            )
            self._compact_extra_row.addWidget(self._verbal_label, 1)
            self._compact_extra_row.addWidget(self._tools_box)
        else:
            self._grid.addWidget(self._s_meter_label, 1, 0)
            self._grid.addWidget(self._rst_label, 1, 1)
            self._grid.addWidget(
                self._mode_combo,
                1,
                2,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self._grid.addWidget(self._verbal_label, 2, 0, 1, 2)
            self._grid.addWidget(self._tools_box, 2, 2)
            self._history_row.setVisible(True)
        self._readout_group.setVisible(not compact)
        self._compact_box.setVisible(compact)
        if self._showing_data:
            self._update_display()
        else:
            self._summary_label.setText(self._empty_hint)
            self._verbal_label.setText("" if compact else self._empty_short)
            self._rst_label.setText(f"RST {_EMPTY}" if compact else _EMPTY)


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
        layout.setSpacing(8)

        # S-Meter bar
        # Empty bar and no reading until the first samples arrive.
        self._bar_label = QLabel(f"S: [{'░' * 9}] {_EMPTY}")
        self._bar_label.setFont(_themes().mono_font())
        self._bar_label.setToolTip("Signal strength in S-units")
        layout.addWidget(self._bar_label)

        # RST
        self._rst_label = set_role(QLabel(f"RST: {_EMPTY}"), "value")
        self._rst_label.setToolTip("Signal report (RST)")
        layout.addWidget(self._rst_label)

        layout.addStretch()

    def update_samples(self, samples: np.ndarray) -> None:
        """Update with samples."""
        self._meter.update(samples)

        # Update bar graph
        bar = self._meter.get_bar_graph(9)
        self._bar_label.setText(f"S: {bar}")

        # Update RST
        self._rst_label.setText(f"RST: {self._meter.get_rst()}")


__all__ = [
    "AnalogMeterWidget",
    "LevelSparkline",
    "SignalMeterPanel",
    "CompactSignalMeter",
]
