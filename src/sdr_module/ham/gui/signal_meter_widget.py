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
from typing import Any, List, Optional, Tuple

try:
    from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
    from PyQt6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen
    from PyQt6.QtWidgets import (
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..signal_meter import (
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


def strength_tone(s_units: float) -> Optional[str]:
    """Theme tone for a signal strength; matches the zones on the dial."""
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
        self._show_peak: bool = True
        self._active: bool = False  # False until the first reading arrives

        # Preferred height fits the full-width scale; in a short panel the
        # gauge shrinks (down to its minimum) before anything scrolls.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumSize(220, self._MIN_HEIGHT)
        self.setToolTip(
            "Signal strength in S-units (6 dB each, S9 = -73 dBm), then dB over "
            "S9. The thin needle shows the peak hold."
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

        painter.setClipping(False)
        painter.setPen(QPen(p.qcolor("surface1"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(face_path)
        painter.end()

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

    _MODES = [
        (
            "Phone",
            SignalMode.PHONE,
            "Phone (SSB, AM, FM): RS report, 2 digits, e.g. 59",
        ),
        ("CW", SignalMode.CW, "CW (Morse): RST report with a tone digit, e.g. 599"),
        ("Digital", SignalMode.DIGITAL, "Digital modes: RS report, 2 digits"),
    ]

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._meter = SignalMeter()
        self._history = SignalHistory()
        self._last_reading: Optional[SignalReading] = None
        self._showing_data = False

        self._setup_ui()

        # Update timer
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(100)  # 10 Hz update

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Analog meter
        self._analog_meter = AnalogMeterWidget()
        layout.addWidget(self._analog_meter)

        # Digital readouts: one grid so the three columns line up.
        #   S-METER   RST    MODE
        #   S9+10     59     [Phone]
        #   Five and nine, 10 over
        #   POWER     SNR    PEAK HOLD
        #   -63 dBm   41 dB  S9+12
        # Compact enough that the whole report fits under the gauge in the
        # S-Meter tab at the default window size.
        readout_group = QGroupBox("Signal Report")
        grid = QGridLayout(readout_group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(0)

        def caption(text: str, tip: str) -> "QLabel":
            lbl = set_role(QLabel(text), "caption")
            lbl.setToolTip(tip)
            return lbl

        def readout(role: str, tip: str) -> "QLabel":
            lbl = set_role(QLabel(_EMPTY), role)
            lbl.setToolTip(tip)
            return lbl

        s_tip = (
            "Signal strength: S1 to S9 in 6 dB steps (S9 = -73 dBm), then dB over S9"
        )
        rst_tip = (
            "RST report: Readability (1-5), Strength (1-9) and, for CW, Tone (1-9)"
        )
        mode_tip = (
            "Report format: phone and digital use RS (2 digits), CW adds a tone "
            "digit (RST, 3 digits)"
        )
        power_tip = (
            "Estimated input power (uncalibrated: relative to the SDR's full scale)"
        )
        snr_tip = "Signal level above the estimated noise floor"
        peak_tip = "Strongest recent reading; decays after 2 seconds"

        grid.addWidget(caption("S-METER", s_tip), 0, 0)
        grid.addWidget(caption("RST", rst_tip), 0, 1)
        grid.addWidget(caption("MODE", mode_tip), 0, 2)

        self._s_meter_label = readout("display", s_tip)
        grid.addWidget(self._s_meter_label, 1, 0)
        self._rst_label = readout("display", rst_tip)
        grid.addWidget(self._rst_label, 1, 1)

        self._mode_combo = QComboBox()
        for index, (text, _mode, tip) in enumerate(self._MODES):
            self._mode_combo.addItem(text)
            self._mode_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._mode_combo.setToolTip(mode_tip)
        self._mode_combo.setAccessibleName("Report mode")
        self._mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(
            self._mode_combo,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        # Verbal report (or the empty-state hint before any data arrives).
        self._verbal_label = QLabel()
        # No word wrap: a wrapping label would make the whole panel
        # height-for-width, so a short tab could not shrink the gauge.
        self._verbal_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._verbal_label.setToolTip("The report as you would say it on the air")
        self._verbal_label.setContentsMargins(0, 2, 0, 0)
        grid.addWidget(self._verbal_label, 2, 0, 1, 3)
        grid.setRowMinimumHeight(3, 10)  # gap before the measurement row

        grid.addWidget(caption("POWER", power_tip), 4, 0)
        grid.addWidget(caption("SNR", snr_tip), 4, 1)
        grid.addWidget(caption("PEAK HOLD", peak_tip), 4, 2)
        self._dbm_label = readout("value", power_tip)
        grid.addWidget(self._dbm_label, 5, 0)
        self._snr_label = readout("value", snr_tip)
        grid.addWidget(self._snr_label, 5, 1)
        self._peak_label = readout("value", peak_tip)
        grid.addWidget(self._peak_label, 5, 2)

        # S-meter strings ("S9+30") are the widest; RST at most 3 digits.
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 4)

        layout.addWidget(readout_group)
        layout.addStretch(1)

        self._show_empty_state("Waiting for a signal. Start the receiver.")

    # ----------------------------------------------------------- public API

    def get_meter(self) -> SignalMeter:
        """Get the signal meter instance."""
        return self._meter

    def update_samples(self, samples: np.ndarray) -> None:
        """Update with new I/Q samples."""
        reading = self._meter.update(samples)
        self._last_reading = reading
        self._history.add(reading)

    def get_qso_report(self) -> str:
        """Get report for QSO logging."""
        return self._meter.get_qso_report()

    def get_contest_report(self) -> str:
        """Get contest-style report."""
        return self._meter.get_contest_report()

    # -------------------------------------------------------------- display

    def _show_empty_state(self, hint: str) -> None:
        """Clear the readouts and show ``hint`` in place of the verbal report."""
        self._showing_data = False
        for lbl in (self._s_meter_label, self._rst_label):
            lbl.setText(_EMPTY)
            set_tone(lbl, "muted")
        for lbl in (self._dbm_label, self._snr_label, self._peak_label):
            lbl.setText(_EMPTY)
        self._verbal_label.setText(hint)
        set_role(self._verbal_label, "hint", None)
        self._analog_meter.set_active(False)

    def _update_display(self) -> None:
        """Update display elements."""
        if self._last_reading is None:
            return

        reading = self._last_reading
        if time.time() - reading.timestamp > _STALE_AFTER_S:
            if self._showing_data:
                self._show_empty_state("No new samples. Start the receiver.")
            return

        if not self._showing_data:
            self._showing_data = True
            set_role(self._verbal_label, "muted", None)
            set_tone(self._rst_label, None)

        # Analog meter: position from dBm so the needle can reach S9+60.
        s_units = dbm_to_s_units(reading.power_dbm)
        peak_s = dbm_to_s_units(reading.peak_hold_dbm)
        self._analog_meter.set_value(s_units, peak_s)

        # Digital readouts. The S reading uses the same 1 dB resolution as
        # the needle and PEAK HOLD (the core rounds to 10 dB above S9, which
        # could read below the peak); the spoken report keeps the 10 dB
        # steps that operators say on the air.
        self._s_meter_label.setText(format_s_units(s_units))
        set_tone(self._s_meter_label, strength_tone(s_units))
        self._rst_label.setText(self._meter.get_rst())
        self._verbal_label.setText(self._meter.get_verbal_report())
        self._dbm_label.setText(f"{reading.power_dbm:.0f} dBm")
        self._snr_label.setText(f"{reading.snr_db:.0f} dB")
        self._peak_label.setText(format_s_units(peak_s))

    def _on_mode_changed(self, index: int) -> None:
        """Handle mode change."""
        if 0 <= index < len(self._MODES):
            self._meter.set_mode(self._MODES[index][1])
            if self._showing_data:
                self._rst_label.setText(self._meter.get_rst())


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
    "SignalMeterPanel",
    "CompactSignalMeter",
]
