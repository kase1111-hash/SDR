"""
Waterfall display widget.

Provides scrolling time-frequency visualization with:
- A color-scale legend (dBFS) in the header whose floor and ceiling follow
  the noise floor and the strongest signal (Auto), or are dragged by hand
- A choice of history length (live, 1 min ... 1 h); a slower row keeps the
  strongest level that arrived during it, so short bursts stay visible
- Newest line at the top, a time axis with pause separators, and a hover
  readout of frequency, level, SNR and age
- Click to tune; drag to measure a signal's bandwidth, duration and peak;
  right-click for more (bookmark, copy, pause, levels, colors, save image)
- An optional frequency axis, for when the spectrum above is hidden

The plot shares its horizontal margins with the spectrum widget (see
:func:`~sdr_module.gui.spectrum_widget.plot_side_margins`), so a frequency
sits at the same x in both displays. Like the spectrum trace, the displayed
image is decimated to the plot width with a per-column maximum, so a narrow
carrier stays bright instead of being blurred away by image scaling.

Levels are stored as 1/8 dB integer codes and painted through a lookup table
(code -> color), so changing the levels, the colors or the theme only
re-maps pixels; it never re-processes the history.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spectrum_widget import (
    HEADER_MARGINS,
    PASSBAND_TOKEN,
    axis_font,
    columns_max,
    draw_focus_frame,
    draw_placeholder,
    draw_readout,
    format_mhz,
    freq_ticks,
    nice_ticks,
    normalize_passband,
    passband_px,
    plot_side_margins,
    readout_decimals,
    snap_step_hz,
)
from .themes import get_palette, set_role, theme_notifier

try:
    from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
    from PyQt6.QtGui import (
        QAction,
        QActionGroup,
        QColor,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QImage,
        QPainter,
        QPen,
        QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMenu,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


# Manual dynamic-range presets (dB below 0 dBFS), offered in the menus.
_RANGE_CHOICES = (60, 80, 100, 120)

# History lengths offered: (label, seconds of history; 0 = one row per update).
_HISTORY_CHOICES: Tuple[Tuple[str, float], ...] = (
    ("Live", 0.0),
    ("1 min", 60.0),
    ("5 min", 300.0),
    ("15 min", 900.0),
    ("1 h", 3600.0),
)

# Time-axis steps (seconds): short labels that fit the shared left margin.
_TIME_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)

# Level codes: code = round((dB - _Q_MIN_DB) * _Q_PER_DB), 1.._Q_CODES-1
# (-200 dBFS to about +56 dBFS in 1/8 dB steps); code 0 marks an empty row.
_Q_PER_DB = 8
_Q_MIN_DB = -200.0
_Q_CODES = 2048
_Q_MAX_DB = _Q_MIN_DB + (_Q_CODES - 1) / _Q_PER_DB

#: dBFS extent of the level bar (the handles move within it).
_LEVEL_SCALE_DB = (-140.0, 0.0)
#: Narrowest floor-to-ceiling span allowed, in dB.
_MIN_LEVEL_SPAN_DB = 10.0

# Auto levels: the floor sits this far below the noise floor (the median bin),
# the ceiling this far above the strongest signal, with at least this span.
# They move only when the target is this far from the current value, so the
# colors don't shimmer as the estimates wander.
_AUTO_FLOOR_BELOW_NOISE_DB = 6.0
_AUTO_CEILING_ABOVE_PEAK_DB = 4.0
_AUTO_MIN_SPAN_DB = 30.0
_AUTO_HYSTERESIS_DB = 3.0

#: A pause between two lines longer than this many typical line intervals
#: (and at least _GAP_MIN_S seconds) is drawn as a separator.
_GAP_FACTOR = 5.0
_GAP_MIN_S = 1.0

#: Height (px) of the passband bracket's end ticks on the top edge.
_PASSBAND_TICK_PX = 6


def time_step(raw: float, max_age: Optional[float] = None) -> float:
    """Smallest time-axis step (seconds) that is >= ``raw``.

    With ``max_age``, the step is also coarse enough that every tick label up
    to that age fits four characters (the shared left margin): a tall
    waterfall at 0.5 s steps would otherwise need ``10.5s``.
    """
    for step in _TIME_STEPS:
        if step < raw:
            continue
        if max_age is not None:
            ticks = nice_ticks(0.0, max_age, step)
            if ticks and len(format_age(ticks[-1], step)) > 4:
                continue
        return float(step)
    return float(_TIME_STEPS[-1])


def format_age(age: float, step: float) -> str:
    """Compact "seconds ago" label: ``0s``, ``0.5s``, ``15s``, ``2m``, ``1h``."""
    if age <= 0:
        return "0s"
    if step < 1:
        return f"{age:.1f}s"
    if step < 60:
        return f"{age:.0f}s"
    if step < 3600:
        return f"{age / 60:g}m"
    return f"{age / 3600:g}h"


def format_elapsed(age: float) -> str:
    """At most four characters for any age: ``4.2s``, ``65s``, ``12m``, ``3h``, ``5d``.

    (Years, ``3y``, only for absurdly old lines, to keep the promise.)
    """
    if not math.isfinite(age):
        return "--"
    # Round first, so 9.96 s reads "10s" rather than a five-character "10.0s".
    if round(age, 1) < 10:
        return f"{max(0.0, age):.1f}s"
    if age < 99.5:
        return f"{age:.0f}s"
    if age < 99.5 * 60:
        return f"{max(2.0, round(age / 60)):.0f}m"
    if age < 99.5 * 3600:
        return f"{max(2.0, round(age / 3600)):.0f}h"
    if age < 999.5 * 86400:
        return f"{max(5.0, round(age / 86400)):.0f}d"
    return f"{min(999.0, max(3.0, round(age / 31_557_600))):.0f}y"


def format_span_hz(hz: float) -> str:
    """A bandwidth for a readout: ``850 Hz``, ``12.5 kHz``, ``1.250 MHz``."""
    hz = abs(float(hz))
    if hz < 1e3:
        return f"{hz:.0f} Hz"
    if hz < 1e6:
        return f"{hz / 1e3:.3g} kHz" if hz < 1e5 else f"{hz / 1e3:.0f} kHz"
    return f"{hz / 1e6:.3f} MHz"


def format_duration(seconds: float) -> str:
    """A duration for a readout: ``250 ms``, ``1.25 s``, ``2 min 05 s``."""
    s = max(0.0, float(seconds))
    if s < 1.0:
        return f"{s * 1e3:.0f} ms"
    if s < 10.0:
        return f"{s:.2f} s"
    if s < 60.0:
        return f"{s:.1f} s"
    if s < 3600.0:
        minutes, rest = divmod(int(round(s)), 60)
        return f"{minutes} min {rest:02d} s"
    hours, rest = divmod(int(round(s)), 3600)
    return f"{hours} h {rest // 60:02d} min"


def history_label(seconds: float) -> str:
    """Menu/combo label of a history length (see ``_HISTORY_CHOICES``)."""
    for label, value in _HISTORY_CHOICES:
        if abs(value - seconds) < 1e-6:
            return label
    return format_duration(seconds)


def quantize_db(values: Any) -> np.ndarray:
    """dB values -> uint16 level codes (1/8 dB steps; never 0, the empty code).

    NaN and -inf map to the lowest code, +inf to the highest, like the
    colormap mapping they feed (below the floor / above the ceiling).
    """
    v = np.asarray(values, dtype=np.float32)
    v = np.nan_to_num(v, nan=_Q_MIN_DB, posinf=_Q_MAX_DB, neginf=_Q_MIN_DB)
    codes = np.rint((v - np.float32(_Q_MIN_DB)) * np.float32(_Q_PER_DB))
    return np.clip(codes, 1, _Q_CODES - 1).astype(np.uint16)


def auto_level_targets(noise_db: float, peak_db: float) -> Tuple[float, float]:
    """Floor and ceiling (dBFS) that frame the noise and the strongest signal.

    The floor sits a little below the noise floor, so noise is a dark texture
    and anything above it stands out; the ceiling sits just above the peak.
    Both are whole dB, inside the level bar's scale, at least
    ``_AUTO_MIN_SPAN_DB`` apart.
    """
    lo, hi = _LEVEL_SCALE_DB
    floor = math.floor(noise_db - _AUTO_FLOOR_BELOW_NOISE_DB)
    ceiling = math.ceil(
        max(peak_db + _AUTO_CEILING_ABOVE_PEAK_DB, floor + _AUTO_MIN_SPAN_DB)
    )
    ceiling = min(max(ceiling, lo + _AUTO_MIN_SPAN_DB), hi)
    floor = max(lo, min(floor, ceiling - _AUTO_MIN_SPAN_DB))
    return float(floor), float(ceiling)


class _LevelBar(QWidget if HAS_PYQT6 else object):
    """Color-scale legend with draggable floor and ceiling handles.

    The bar spans ``_LEVEL_SCALE_DB`` and is painted with the color each
    level gets, so it doubles as the waterfall's legend: everything below the
    floor shares the lowest color, everything above the ceiling the highest.
    """

    _PAD = 5  # px at each end, so a handle at the scale's end stays grabbable
    _BAR_H = 9
    _GRAB_PX = 7

    def __init__(self, owner: "WaterfallWidget"):
        super().__init__(owner)
        self._owner = owner
        self._font = axis_font()
        self._fm = QFontMetrics(self._font)
        self._drag: Optional[str] = None  # "floor", "ceiling" or "both"
        self._drag_anchor = 0.0
        self._drag_levels = (0.0, 0.0)
        self._gradient_key: Optional[tuple] = None
        self._gradient: Optional[np.ndarray] = None
        self._gradient_image: Optional[QImage] = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName("Waterfall color levels")
        self.setToolTip(
            "Color scale in dBFS: the color each signal level gets.\n"
            "Drag an end to set the floor or ceiling, drag the middle to "
            "shift both, or scroll to shift.\nDouble-click (or press Home) "
            "for Auto. Keys: Left/Right move the floor, Up/Down the ceiling."
        )

    def sizeHint(self):  # noqa: N802 - Qt override
        return QSize(172, self._BAR_H + self._fm.height() + 5)

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        return QSize(128, self._BAR_H + self._fm.height() + 5)

    # -- geometry --------------------------------------------------------
    def _span_px(self) -> Tuple[float, float]:
        return float(self._PAD), float(max(self._PAD + 1, self.width() - self._PAD))

    def _db_to_x(self, db: float) -> float:
        lo, hi = _LEVEL_SCALE_DB
        x0, x1 = self._span_px()
        frac = (min(max(db, lo), hi) - lo) / (hi - lo)
        return x0 + frac * (x1 - x0)

    def _x_to_db(self, x: float) -> float:
        lo, hi = _LEVEL_SCALE_DB
        x0, x1 = self._span_px()
        frac = (x - x0) / max(1.0, x1 - x0)
        return lo + min(max(frac, 0.0), 1.0) * (hi - lo)

    # -- painting ----------------------------------------------------------
    def _gradient_qimage(self, width: int) -> "QImage":
        owner = self._owner
        key = (width, owner._db_range, owner._colormap_name, id(owner._colormap))
        if self._gradient_image is None or self._gradient_key != key:
            lo, hi = _LEVEL_SCALE_DB
            dbs = np.linspace(lo, hi, max(1, width))
            rgb = owner._colormap[owner._level_index(dbs[np.newaxis, :])[0]]
            self._gradient = np.ascontiguousarray(rgb, dtype=np.uint8)
            self._gradient_image = QImage(
                self._gradient.data,
                self._gradient.shape[0],
                1,
                3 * self._gradient.shape[0],
                QImage.Format.Format_RGB888,
            )
            self._gradient_key = key
        return self._gradient_image

    def paintEvent(self, event):  # noqa: N802 - Qt override
        p = get_palette()
        owner = self._owner
        floor, ceiling = owner._db_range
        painter = QPainter(self)
        try:
            x0, x1 = self._span_px()
            bar = QRectF(x0, 1, x1 - x0, self._BAR_H)
            image = self._gradient_qimage(int(round(bar.width())))
            painter.drawImage(bar, image, QRectF(0, 0, image.width(), 1))
            # Dim what lies outside floor..ceiling: those levels are clamped.
            fx, cx = self._db_to_x(floor), self._db_to_x(ceiling)
            dim = p.qcolor("plot_bg", 150)
            if fx > bar.left():
                painter.fillRect(
                    QRectF(bar.left(), bar.top(), fx - bar.left(), bar.height()), dim
                )
            if cx < bar.right():
                painter.fillRect(
                    QRectF(cx, bar.top(), bar.right() - cx, bar.height()), dim
                )
            painter.setPen(QPen(p.qcolor("plot_grid"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bar.adjusted(-0.5, -0.5, 0.5, 0.5))
            # Handles: a text-colored post with a halo, a little taller than
            # the bar, so they read on any colormap.
            accent = self.hasFocus()
            for x in (fx, cx):
                xi = round(x)
                painter.fillRect(
                    QRectF(xi - 2, 0, 4, self._BAR_H + 3), p.qcolor("plot_bg")
                )
                painter.fillRect(
                    QRectF(xi - 1, 0, 2, self._BAR_H + 3),
                    p.qcolor("accent" if accent else "plot_text"),
                )
            # Values under their handles, pushed apart if they would touch.
            painter.setFont(self._font)
            fm = self._fm
            y = self._BAR_H + 3
            h = fm.height()
            left_text = f"{floor:.0f}"
            right_text = f"{ceiling:.0f} dBFS"
            lw = fm.horizontalAdvance(left_text)
            rw = fm.horizontalAdvance(right_text)
            if rw + lw + 6 > self.width():
                right_text = f"{ceiling:.0f}"
                rw = fm.horizontalAdvance(right_text)
            lx = min(max(0.0, fx - lw / 2), self.width() - lw - rw - 6)
            rx = min(max(lx + lw + 6, cx - rw / 2), self.width() - rw)
            painter.setPen(p.qcolor("plot_text"))
            painter.drawText(
                QRectF(lx, y, lw, h), int(Qt.AlignmentFlag.AlignLeft), left_text
            )
            painter.drawText(
                QRectF(rx, y, rw, h), int(Qt.AlignmentFlag.AlignLeft), right_text
            )
        finally:
            painter.end()

    # -- interaction -----------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        floor, ceiling = self._owner._db_range
        fx, cx = self._db_to_x(floor), self._db_to_x(ceiling)
        if abs(x - fx) <= self._GRAB_PX and abs(x - fx) <= abs(x - cx):
            self._drag = "floor"
        elif abs(x - cx) <= self._GRAB_PX:
            self._drag = "ceiling"
        elif fx < x < cx:
            self._drag = "both"
        else:
            # Outside the range: move the nearer end there.
            self._drag = "floor" if abs(x - fx) < abs(x - cx) else "ceiling"
            self._drag_to(x)
        self._drag_anchor = self._x_to_db(x)
        self._drag_levels = self._owner._db_range
        event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if self._drag is not None:
            self._drag_to(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _drag_to(self, x: float) -> None:
        db = round(self._x_to_db(x))
        floor, ceiling = self._owner._db_range
        if self._drag == "floor":
            floor = min(db, ceiling - _MIN_LEVEL_SPAN_DB)
        elif self._drag == "ceiling":
            ceiling = max(db, floor + _MIN_LEVEL_SPAN_DB)
        elif self._drag == "both":
            f0, c0 = self._drag_levels
            shift = round(self._x_to_db(x) - self._drag_anchor)
            lo, hi = _LEVEL_SCALE_DB
            shift = min(max(shift, lo - f0), hi - c0)
            floor, ceiling = f0 + shift, c0 + shift
        self._owner._user_set_levels(floor, ceiling)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt override
        self._owner.set_auto_levels(True)
        event.accept()

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        notches = event.angleDelta().y() / 120.0
        if not notches:
            super().wheelEvent(event)
            return
        self._owner._shift_levels(2.0 * notches)
        event.accept()

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        floor, ceiling = self._owner._db_range
        key = event.key()
        if key == Qt.Key.Key_Home:
            self._owner.set_auto_levels(True)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            sign = -1.0 if key == Qt.Key.Key_Left else 1.0
            self._owner._user_set_levels(
                min(floor + sign * step, ceiling - _MIN_LEVEL_SPAN_DB), ceiling
            )
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            sign = -1.0 if key == Qt.Key.Key_Down else 1.0
            self._owner._user_set_levels(
                floor, max(ceiling + sign * step, floor + _MIN_LEVEL_SPAN_DB)
            )
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def focusInEvent(self, event):  # noqa: N802 - Qt override
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event):  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self.update()


class _WaterfallCanvas(QWidget if HAS_PYQT6 else object):
    """The painted plot area below the waterfall header strip."""

    def __init__(self, owner: "WaterfallWidget"):
        super().__init__(owner)
        self._owner = owner
        self._press: Optional[QPointF] = None
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumHeight(120)

    def paintEvent(self, event):  # noqa: N802 - Qt override
        self._owner._paint_canvas(self)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        pos = event.position()
        if self._press is not None:
            moved = (pos - self._press).manhattanLength()
            if moved >= QApplication.startDragDistance() or self._owner._measuring:
                self._owner._drag_measure(self._press, pos)
        self._owner._set_hover(pos)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        self._owner._set_hover(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            # Tune on release, so a drag can measure instead.
            self._press = QPointF(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            press, self._press = self._press, None
            if self._owner._measuring:
                self._owner._finish_measure()
            elif self._owner._click_at(press):
                self._owner.clear_measurement()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        self._owner._show_context_menu(QPointF(event.pos()), event.globalPos())
        event.accept()


class WaterfallWidget(QWidget if HAS_PYQT6 else object):
    """
    Waterfall display widget.

    Shows scrolling spectrogram with time on Y-axis and frequency on X-axis.
    The newest line is at the top. Clicking emits :attr:`frequency_clicked`;
    dragging measures a region; the context menu offers
    :attr:`bookmark_requested` and the display options.
    """

    # Color map definitions (data-visualization colormaps, theme independent)
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
        #: A frequency (Hz) the user asked to bookmark from the context menu.
        bookmark_requested = pyqtSignal(float)
        #: Colors, history length or levels changed by the user (to persist).
        display_settings_changed = pyqtSignal()

    def __init__(self, parent=None, history_size: int = 500):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._history_size = history_size
        self._fft_size = 2048
        self._db_range: Tuple[float, float] = (-100.0, 0.0)
        self._auto_levels = False
        self._center_freq = 100e6
        self._sample_rate = 2.4e6
        self._history_s = 0.0  # 0 = one row per add_line (Live)
        self._paused = False
        self._skipped = 0  # lines dropped while paused
        self._freq_axis = False

        # Data (newest last): each row's dB line, its start time and the
        # history setting it was recorded with (so pauses are told apart
        # from slow rows).
        self._history: deque = deque(maxlen=history_size)
        self._times: deque = deque(maxlen=history_size)
        self._row_steps: deque = deque(maxlen=history_size)
        self._row_open = False  # the newest row may still gather lines
        self._row_end = 0.0  # when the newest row stops gathering

        # Level estimates (dBFS) for Auto levels, the legend and the readouts.
        self._noise_db: Optional[float] = None
        self._peak_db: Optional[float] = None

        # Color map
        self._colormap_name = "turbo"
        self._colormap = self._build_colormap(self._colormap_name)

        # Level codes (uint16, row 0 newest, 0 = no data) and the lookup
        # tables that turn codes into colors for the current levels, colors
        # and theme. ``_disp_*`` is what is drawn: the codes decimated to the
        # plot's pixel width with a per-column maximum, and their RGBX pixels.
        # QImages wrap these arrays without copying, so they are kept alive.
        self._codes: Optional[np.ndarray] = None
        self._lut_rgb: Optional[np.ndarray] = None
        self._lut32: Optional[np.ndarray] = None
        self._lut_key: Optional[tuple] = None
        self._disp_codes: Optional[np.ndarray] = None
        self._disp_rgb: Optional[np.ndarray] = None
        self._disp_image: Optional[QImage] = None
        self._version = 0  # bumps on every code or color change
        self._full_rgb: Optional[np.ndarray] = None
        self._full_image: Optional[QImage] = None
        self._full_version = -1

        # Highlights
        self._highlights: List[Tuple[int, int, int, int, Any]] = []
        # Demodulated channel: (low, high) Hz offsets from the center.
        self._passband: Optional[Tuple[float, float]] = None

        # Measurement: absolute frequencies (Hz) and row start times, so the
        # box follows the signal as rows scroll. None when there is none.
        self._measure: Optional[Dict[str, float]] = None
        self._measuring = False

        # Paint state
        self._font = axis_font()
        self._unit_font = QFont(self._font)
        self._unit_font.setBold(True)
        self._fm = QFontMetrics(self._font)
        self._hover: Optional[QPointF] = None

        self.setMinimumHeight(200)
        self._setup_ui()
        theme_notifier().theme_changed.connect(self._on_theme_changed)

    @property
    def _bg_color(self) -> "QColor":
        """Plot background of the active theme (fills rows with no data)."""
        return get_palette().qcolor("plot_bg")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Header strip (title, hint, pause, history, levels, options) above
        the plot canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        set_role(header, "header-strip")
        self._header = header
        controls = QHBoxLayout(header)
        controls.setContentsMargins(*HEADER_MARGINS)
        controls.setSpacing(8)

        title = QLabel("WATERFALL")
        set_role(title, "caption")
        controls.addWidget(title)

        hint = QLabel("Click to tune · drag to measure · right-click for more")
        set_role(hint, "hint")
        hint.setToolTip(
            "Each row is one spectrum; the newest row is at the top and older "
            "rows scroll down.\nClick to tune to a frequency; drag a box to "
            "measure a signal's bandwidth, duration and peak;\nright-click "
            "to bookmark, copy or change the display."
        )
        # Never let the optional hint raise the widget's minimum width:
        # _fit_header() hides it before it would be squeezed.
        hint.setMinimumWidth(1)
        controls.addWidget(hint)
        self._hint_label = hint

        controls.addStretch(1)

        # Header controls don't take focus on click, so Space/arrow keys keep
        # acting on the plot (start/stop, tuning). Tab still reaches them.
        self._pause_btn = QPushButton("Pause")
        set_role(self._pause_btn, "compact")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._pause_btn.setToolTip(
            "Freeze the waterfall to study it (P). The receiver keeps running; "
            "rows that arrive while paused are skipped."
        )
        self._pause_btn.toggled.connect(self.set_paused)
        # Wide enough for "Resume" too, so the header doesn't jump.
        resume = self._pause_btn.fontMetrics().horizontalAdvance("Resume")
        self._pause_btn.setMinimumWidth(
            max(self._pause_btn.sizeHint().width(), resume + 20)
        )
        controls.addWidget(self._pause_btn)

        self._history_caption = QLabel("HISTORY")
        set_role(self._history_caption, "caption")
        controls.addWidget(self._history_caption)
        self._history_combo = QComboBox()
        for label, seconds in _HISTORY_CHOICES:
            self._history_combo.addItem(label, seconds)
        self._history_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._history_combo.setToolTip(
            "How much time the waterfall spans. Live adds a row per update "
            "(about 15 s at full speed);\nlonger spans keep the strongest level "
            "of each row's interval, so short bursts stay visible."
        )
        self._history_caption.setToolTip(self._history_combo.toolTip())
        self._history_caption.setBuddy(self._history_combo)
        self._history_combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._history_combo.currentIndexChanged.connect(self._on_history_index_changed)
        controls.addWidget(self._history_combo)

        controls.addSpacing(4)

        self._levels_caption = QLabel("LEVELS")
        set_role(self._levels_caption, "caption")
        controls.addWidget(self._levels_caption)
        self._level_bar = _LevelBar(self)
        self._levels_caption.setToolTip(self._level_bar.toolTip())
        # Like the hint, captions never raise the minimum width: _fit_header()
        # hides them first when the strip gets narrow.
        for caption in (self._history_caption, self._levels_caption):
            caption.setMinimumWidth(1)
        controls.addWidget(self._level_bar)
        self._auto_btn = QPushButton("Auto")
        set_role(self._auto_btn, "compact")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._auto_btn.setToolTip(
            "Auto levels: the floor follows the noise floor and the ceiling "
            "the strongest signal, so weak signals stand out.\nDragging the "
            "color scale switches to manual levels."
        )
        self._auto_btn.toggled.connect(self.set_auto_levels)
        controls.addWidget(self._auto_btn)

        self._options_btn = QPushButton("⋯")
        set_role(self._options_btn, "compact")
        self._options_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._options_btn.setAccessibleName("Waterfall options")
        self._options_btn.setToolTip(
            "More: colors, level presets, clear history, save image"
        )
        self._options_btn.clicked.connect(self._show_options_menu)
        controls.addWidget(self._options_btn)

        # The actions the menus share (Clear keeps its state here).
        self._clear_action = QAction("&Clear History", self)
        self._clear_action.triggered.connect(self.clear)
        self._update_clear_action()

        layout.addWidget(header)

        self._canvas = _WaterfallCanvas(self)
        self._canvas.setAccessibleName("Waterfall plot")
        self._canvas.setAccessibleDescription(
            "Signal strength over time and frequency, newest at the top. "
            "Click to tune, drag to measure, right-click for options."
        )
        layout.addWidget(self._canvas, 1)
        self._update_level_description()

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_header()

    def _fit_header(self) -> None:
        """Hide optional header text (hint, then captions) when too narrow.

        Clipped text or squeezed controls look broken, so the extras simply
        disappear below the width they need, most expendable first.
        """
        layout = self._header.layout()
        if layout is None:
            return
        spacing = max(0, layout.spacing())
        optional = [self._hint_label, self._history_caption, self._levels_caption]
        base = layout.sizeHint().width()
        for widget in optional:
            if not widget.isHidden():
                base -= widget.sizeHint().width() + spacing
        available = self.width() - 24  # keep a gap before the controls
        keep = {}
        for widget in reversed(optional):  # the captions matter more
            need = widget.sizeHint().width() + spacing
            keep[widget] = base + need <= available
            if keep[widget]:
                base += need
        for widget in optional:
            widget.setHidden(not keep[widget])

    def focusInEvent(self, event):  # noqa: N802 - Qt override
        """Show the focus frame: Space and Left/Right now act on this plot."""
        super().focusInEvent(event)
        self._canvas.update()

    def focusOutEvent(self, event):  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self._canvas.update()

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        """P pauses; Esc clears a measurement. Other keys (Space, arrows)
        go to the main window, which starts/stops and tunes."""
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if event.key() == Qt.Key.Key_P:
                self.set_paused(not self._paused)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape and self._measure is not None:
                self.clear_measurement()
                event.accept()
                return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        """The context-menu key while the plot has focus."""
        if not self._canvas.geometry().contains(event.pos()):
            super().contextMenuEvent(event)
            return
        pos = QPointF(self._canvas.mapFrom(self, event.pos()))
        self._show_context_menu(pos, event.globalPos())
        event.accept()

    def _update_clear_action(self) -> None:
        has_data = len(self._history) > 0
        self._clear_action.setEnabled(has_data)
        tip = (
            "Erase the waterfall history"
            if has_data
            else "Nothing to clear yet: the waterfall is empty"
        )
        self._clear_action.setToolTip(tip)
        self._clear_action.setStatusTip(tip)

    def _on_history_index_changed(self, index: int) -> None:
        seconds = self._history_combo.itemData(index)
        if seconds is not None:
            self.set_history_seconds(float(seconds))

    def _on_color_index_changed(self, index: int) -> None:
        """Kept for callers of the former colors combo (index into COLORMAPS)."""
        names = list(self.COLORMAPS)
        if 0 <= index < len(names):
            self._on_colormap_changed(names[index])

    def _on_colormap_changed(self, name: str):
        """Handle colormap change."""
        if name not in self.COLORMAPS:
            name = "turbo"
        changed = name != self._colormap_name
        self._colormap_name = name
        self._colormap = self._build_colormap(name)
        self._refresh_colors()
        if changed:
            self.display_settings_changed.emit()

    def _on_range_changed(self, index: int):
        """Apply a manual range preset (index into ``_RANGE_CHOICES``)."""
        index = max(0, min(index, len(_RANGE_CHOICES) - 1))
        self.set_db_range(-float(_RANGE_CHOICES[index]), 0.0)

    def _on_theme_changed(self, _name: str = "") -> None:
        try:
            self._refresh_colors()
        except RuntimeError:  # pragma: no cover - widget already deleted
            pass

    def _refresh_colors(self) -> None:
        """Re-map pixels after a colors, levels or theme change."""
        self._ensure_lut()
        self._level_bar.update()
        self._canvas.update()

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_line(self, power_db: np.ndarray):
        """
        Add a new spectrum line to the waterfall.

        With a history length other than Live, lines arriving within one
        row's interval are merged into that row (the per-bin maximum), and
        the newest row updates in place until its interval ends.

        Args:
            power_db: Power spectrum in dB
        """
        if self._paused:
            self._skipped += 1
            return
        line = self._fit_line(power_db)
        now = time.monotonic()
        step = self._row_step()
        if step > 0 and self._row_open and self._history and now < self._row_end:
            merged = np.fmax(self._history[-1], line)
            self._history[-1] = merged
            self._write_top_row(merged, scroll=False)
        else:
            # Rows keep an exact cadence (each ends one interval after the
            # previous one ended), so N rows span N intervals even though
            # lines arrive on a coarser frame clock. After a pause the
            # cadence restarts from now.
            if step > 0 and self._row_open and now - self._row_end < step:
                self._row_end += step
            else:
                self._row_end = now + step
            self._history.append(line)
            self._times.append(now)
            self._row_steps.append(step)
            self._row_open = True
            self._write_top_row(line, scroll=True)
        self._observe_levels(self._history[-1])
        if not self._clear_action.isEnabled():
            self._update_clear_action()
        self._canvas.update()

    def clear(self):
        """Clear the waterfall."""
        self._history.clear()
        self._times.clear()
        self._row_steps.clear()
        self._row_open = False
        self._codes = None
        self._disp_codes = None
        self._disp_rgb = None
        self._disp_image = None
        self._version += 1
        self._measure = None
        self._measuring = False
        self._update_clear_action()
        self._canvas.update()

    def set_center_freq(self, center_freq: float) -> None:
        """Update the center frequency used for click-to-tune and labels."""
        self._center_freq = center_freq
        self._canvas.update()

    def set_sample_rate(self, sample_rate: float) -> None:
        """Update the sample rate used for click-to-tune and labels."""
        self._sample_rate = sample_rate
        self._canvas.update()

    def set_frequency_range(self, center_freq: float, sample_rate: float) -> None:
        """Set center frequency and span together (mirrors the spectrum)."""
        self._center_freq = center_freq
        self._sample_rate = sample_rate
        self._canvas.update()

    def set_passband(self, low_hz: Optional[float], high_hz: Optional[float]) -> None:
        """Mark the demodulated channel with a bracket on the top edge.

        Same arguments as :meth:`SpectrumWidget.set_passband
        <sdr_module.gui.spectrum_widget.SpectrumWidget.set_passband>`: Hz
        offsets from the tuned frequency; ``None`` or ``low_hz >= high_hz``
        hides it. Only the top edge is marked, so no signal is tinted.
        """
        band = normalize_passband(low_hz, high_hz)
        if band != self._passband:
            self._passband = band
            self._canvas.update()

    def passband(self) -> Optional[Tuple[float, float]]:
        """The marked channel as (low, high) Hz offsets, or None when hidden."""
        return self._passband

    def set_db_range(self, min_db: float, max_db: float):
        """Show ``min_db``..``max_db`` dBFS across the colors (manual levels)."""
        self._set_auto(False)
        self._set_levels(min_db, max_db)

    def levels(self) -> Tuple[float, float]:
        """The current floor and ceiling (dBFS) of the color scale."""
        return self._db_range

    def set_auto_levels(self, enabled: bool) -> None:
        """Let the levels follow the noise floor and the strongest signal."""
        enabled = bool(enabled)
        changed = enabled != self._auto_levels
        self._set_auto(enabled)
        if enabled:
            self._retarget_auto_levels(force=True)
        if changed:
            self.display_settings_changed.emit()

    def auto_levels(self) -> bool:
        return self._auto_levels

    def noise_floor_db(self) -> Optional[float]:
        """The estimated noise floor (median bin, dBFS), once data arrived."""
        return self._noise_db

    def set_history_seconds(self, seconds: float) -> None:
        """Span of the waterfall: 0 (Live, one row per update) or seconds of
        history (each row keeps the strongest level of its interval)."""
        values = [value for _label, value in _HISTORY_CHOICES]
        seconds = min(values, key=lambda v: abs(v - float(seconds)))
        changed = seconds != self._history_s
        self._history_s = seconds
        self._row_open = False  # the next line starts a row at the new pace
        idx = self._history_combo.findData(seconds)
        if idx >= 0 and idx != self._history_combo.currentIndex():
            self._history_combo.blockSignals(True)
            self._history_combo.setCurrentIndex(idx)
            self._history_combo.blockSignals(False)
        self._canvas.update()
        if changed:
            self.display_settings_changed.emit()

    def history_seconds(self) -> float:
        return self._history_s

    def set_paused(self, paused: bool) -> None:
        """Freeze the display (new lines are skipped) or resume it."""
        paused = bool(paused)
        if paused == self._paused:
            return
        self._paused = paused
        self._row_open = False
        if paused:
            self._skipped = 0
        if self._pause_btn.isChecked() != paused:
            self._pause_btn.blockSignals(True)
            self._pause_btn.setChecked(paused)
            self._pause_btn.blockSignals(False)
        self._pause_btn.setText("Resume" if paused else "Pause")
        self._canvas.update()

    def is_paused(self) -> bool:
        return self._paused

    def set_colormap(self, name: str) -> None:
        """Use one of :attr:`COLORMAPS` (unknown names fall back to turbo)."""
        self._on_colormap_changed(name)

    def colormap_name(self) -> str:
        return self._colormap_name

    def set_frequency_axis_visible(self, visible: bool) -> None:
        """Label frequencies under the plot (for when the spectrum is hidden)."""
        visible = bool(visible)
        if visible != self._freq_axis:
            self._freq_axis = visible
            self._canvas.update()

    def display_settings(self) -> Dict[str, Any]:
        """The user's display choices, for saving between sessions."""
        floor, ceiling = self._db_range
        return {
            "colormap": self._colormap_name,
            "history_s": self._history_s,
            "auto_levels": self._auto_levels,
            "floor_db": floor,
            "ceiling_db": ceiling,
        }

    def apply_display_settings(self, settings: Dict[str, Any]) -> None:
        """Restore :meth:`display_settings` output (unknown keys ignored)."""
        if not isinstance(settings, dict):
            return
        blocked = self.blockSignals(True)
        try:
            name = settings.get("colormap")
            if isinstance(name, str) and name in self.COLORMAPS:
                self._on_colormap_changed(name)
            try:
                self.set_history_seconds(
                    float(settings.get("history_s", self._history_s))
                )
            except (TypeError, ValueError):
                pass
            try:
                floor = float(settings.get("floor_db", self._db_range[0]))
                ceiling = float(settings.get("ceiling_db", self._db_range[1]))
                if math.isfinite(floor) and math.isfinite(ceiling):
                    self._set_levels(floor, ceiling)
            except (TypeError, ValueError):
                pass
            if "auto_levels" in settings:
                self.set_auto_levels(bool(settings.get("auto_levels")))
        finally:
            self.blockSignals(blocked)

    def measurement(self) -> Optional[Dict[str, float]]:
        """The measured region: ``f_low``/``f_high`` (Hz), ``bandwidth`` (Hz),
        ``center`` (Hz), ``duration`` (s) and ``peak_db`` (dBFS, or NaN), or
        None when nothing is measured."""
        if self._measure is None:
            return None
        return self._measure_stats()

    def clear_measurement(self) -> None:
        if self._measure is not None or self._measuring:
            self._measure = None
            self._measuring = False
            self._canvas.update()

    def save_image(self, path: str) -> bool:
        """Save the raw waterfall (one pixel per bin and row). Returns success."""
        image = self._image
        if image is None:
            return False
        return bool(image.save(path))

    def export_image(self, path: str) -> bool:
        """Save the plot as shown, with time and frequency axes (PNG)."""
        hover, axis = self._hover, self._freq_axis
        self._hover, self._freq_axis = None, True
        try:
            return bool(self._canvas.grab().save(path))
        finally:
            self._hover, self._freq_axis = hover, axis
            self._canvas.update()

    def frequency_at(self, x: float) -> Optional[float]:
        """Frequency (Hz) under canvas x-coordinate ``x``, or None outside."""
        plot = self._plot_rect()
        if plot.width() <= 0 or x < plot.left() or x > plot.right():
            return None
        frac = (x - plot.left()) / plot.width()
        return self._center_freq - self._sample_rate / 2 + frac * self._sample_rate

    def plot_rect(self) -> "QRectF":
        """The plot area in canvas coordinates (excludes the axis margins)."""
        return self._plot_rect()

    def add_highlight(
        self,
        time_start: int,
        time_end: int,
        freq_start: int,
        freq_end: int,
        color: Any,
    ):
        """Add a highlight region (image rows ``time_*``, FFT bins ``freq_*``)."""
        self._highlights.append((time_start, time_end, freq_start, freq_end, color))
        self._canvas.update()

    def clear_highlights(self):
        """Clear all highlights."""
        self._highlights.clear()
        self._canvas.update()

    # ------------------------------------------------------------------
    # Internals: levels
    # ------------------------------------------------------------------

    def _set_auto(self, enabled: bool) -> None:
        self._auto_levels = bool(enabled)
        if self._auto_btn.isChecked() != self._auto_levels:
            self._auto_btn.blockSignals(True)
            self._auto_btn.setChecked(self._auto_levels)
            self._auto_btn.blockSignals(False)
        self._update_level_description()

    def _set_levels(self, floor: float, ceiling: float) -> None:
        floor, ceiling = float(floor), float(ceiling)
        if not (math.isfinite(floor) and math.isfinite(ceiling)):
            return
        if ceiling - floor < 1.0:
            ceiling = floor + 1.0
        if (floor, ceiling) == self._db_range:
            return
        self._db_range = (floor, ceiling)
        self._update_level_description()
        self._refresh_colors()

    def _user_set_levels(self, floor: float, ceiling: float) -> None:
        """Levels chosen by hand (dragging the color scale): manual mode."""
        was_auto = self._auto_levels
        before = self._db_range
        self._set_auto(False)
        self._set_levels(floor, ceiling)
        if was_auto or self._db_range != before:
            self.display_settings_changed.emit()

    def _shift_levels(self, delta_db: float) -> None:
        floor, ceiling = self._db_range
        lo, hi = _LEVEL_SCALE_DB
        delta_db = min(max(delta_db, lo - floor), hi - ceiling)
        self._user_set_levels(floor + delta_db, ceiling + delta_db)

    def _observe_levels(self, row: np.ndarray) -> None:
        """Track the noise floor (median bin) and the strongest signal."""
        values = np.asarray(row, dtype=np.float32)
        finite = values[np.isfinite(values)]
        n = finite.size
        if n < 16:
            return
        top = n - 1 - n // 400  # ~99.75th percentile: ignores lone spikes
        part = np.partition(finite, (n // 2, top))
        noise, peak = float(part[n // 2]), float(part[top])
        if self._noise_db is None:
            self._noise_db = noise
        else:
            self._noise_db = 0.9 * self._noise_db + 0.1 * noise
        if self._peak_db is None or peak > self._peak_db:
            self._peak_db = peak  # a new signal shows at once...
        else:
            self._peak_db = 0.97 * self._peak_db + 0.03 * peak  # ...fades slowly
        if self._auto_levels:
            self._retarget_auto_levels()

    def _retarget_auto_levels(self, force: bool = False) -> None:
        if self._noise_db is None or self._peak_db is None:
            return
        floor, ceiling = auto_level_targets(self._noise_db, self._peak_db)
        cur_floor, cur_ceiling = self._db_range
        if (
            force
            or abs(floor - cur_floor) >= _AUTO_HYSTERESIS_DB
            or abs(ceiling - cur_ceiling) >= _AUTO_HYSTERESIS_DB
        ):
            self._set_levels(floor, ceiling)

    def _update_level_description(self) -> None:
        floor, ceiling = self._db_range
        mode = "automatic" if self._auto_levels else "manual"
        text = f"Floor {floor:.0f} dBFS, ceiling {ceiling:.0f} dBFS, {mode}"
        self._level_bar.setAccessibleDescription(text)
        self._level_bar.update()

    # ------------------------------------------------------------------
    # Internals: image
    # ------------------------------------------------------------------

    def _fit_line(self, power_db: Any) -> np.ndarray:
        """``power_db`` as a new float array of ``fft_size`` values."""
        power_db = np.asarray(power_db)
        if len(power_db) == 0:
            # No spectrum data: record a neutral (min-dB) row so history stays
            # in sync with caller cadence but we skip the expensive interp path.
            return np.full(self._fft_size, self._db_range[0], dtype=np.float32)
        if len(power_db) != self._fft_size:
            # Resample if needed
            return np.interp(
                np.linspace(0, 1, self._fft_size),
                np.linspace(0, 1, len(power_db)),
                power_db,
            )
        return np.array(power_db, copy=True)

    def _row_step(self) -> float:
        """Seconds each row gathers lines for (0 = Live: one row per line)."""
        return self._history_s / self._history_size if self._history_s > 0 else 0.0

    def _level_index(self, lines: np.ndarray) -> np.ndarray:
        """Map (n, fft_size) dB values to (n, fft_size) uint8 colormap indices.

        The mapping is monotonic, so the maximum of indices is the index of
        the maximum level (used for peak-preserving decimation).
        """
        min_db, max_db = self._db_range
        db_range = max_db - min_db
        if db_range <= 0:
            db_range = 1.0
        lines = np.nan_to_num(
            np.asarray(lines, dtype=np.float32),
            nan=min_db,
            posinf=max_db,
            neginf=min_db,
        )
        normalized = np.clip((lines - min_db) / db_range, 0.0, 1.0)
        return (normalized * 255.0).astype(np.uint8)

    def _colorize(self, lines: np.ndarray) -> np.ndarray:
        """Map (n, fft_size) dB values to (n, fft_size, 3) colormap RGB."""
        return self._colormap[self._level_index(lines)]

    def _ensure_lut(self) -> None:
        """(Re)build the code -> color tables for the current levels, colors
        and theme, and re-map the displayed pixels if they changed."""
        palette = get_palette()
        key = (self._db_range, self._colormap_name, id(self._colormap), palette.name)
        if self._lut32 is not None and key == self._lut_key:
            return
        dbs = _Q_MIN_DB + np.arange(_Q_CODES, dtype=np.float64) / _Q_PER_DB
        rgb = self._colormap[self._level_index(dbs[np.newaxis, :])[0]]
        bg = palette.qcolor("plot_bg")
        rgb[0] = (bg.red(), bg.green(), bg.blue())  # code 0: no data yet
        rgbx = np.empty((_Q_CODES, 4), dtype=np.uint8)
        rgbx[:, :3] = rgb
        rgbx[:, 3] = 255
        self._lut_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._lut32 = rgbx.view(np.uint32).reshape(_Q_CODES)
        self._lut_key = key
        self._version += 1
        if self._disp_codes is not None:
            self._regather_display()

    def _full_resolution_pixels(self) -> Optional[np.ndarray]:
        """The full-resolution image (rows x bins x RGB), row 0 newest."""
        if self._codes is None:
            return None
        self._ensure_lut()
        if self._full_rgb is None or self._full_version != self._version:
            self._full_rgb = np.ascontiguousarray(self._lut_rgb[self._codes])
            self._full_image = None
            self._full_version = self._version
        return self._full_rgb

    #: The full-resolution image as an array (see _full_resolution_pixels).
    _image_rgb = property(_full_resolution_pixels)

    @property
    def _image(self) -> Optional["QImage"]:
        """:attr:`_image_rgb` as a QImage (wrapping it, no copy)."""
        rgb = self._image_rgb
        if rgb is None:
            return None
        if self._full_image is None:
            self._full_image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                3 * rgb.shape[1],
                QImage.Format.Format_RGB888,
            )
        return self._full_image

    def _wrap_display(self) -> None:
        """(Re)wrap ``_disp_rgb`` in a QImage (no pixel copy)."""
        disp = self._disp_rgb
        self._disp_image = QImage(
            disp.data,
            disp.shape[1],
            disp.shape[0],
            4 * disp.shape[1],
            QImage.Format.Format_RGBX8888,
        )

    def _regather_display(self) -> None:
        """Display pixels from the display codes through the current table."""
        pixels = self._lut32[self._disp_codes]
        self._disp_rgb = pixels.view(np.uint8).reshape(pixels.shape + (4,))
        self._wrap_display()

    def _display_columns(self, plot: "QRectF") -> int:
        """Image columns the plot can show (device pixels, at most fft_size)."""
        dpr = self._canvas.devicePixelRatioF() or 1.0
        return int(max(1, min(self._fft_size, math.ceil(plot.width() * dpr))))

    def _display_image(self, cols: int) -> Optional["QImage"]:
        """The image to draw for a plot ``cols`` device pixels wide.

        The codes decimated to ``cols`` columns with a per-column maximum
        (cached, scrolled by :meth:`add_line`, rebuilt when the width
        changes), colored through the lookup table.
        """
        if self._codes is None:
            return None
        self._ensure_lut()
        cols = max(1, min(int(cols), self._fft_size))
        if (
            self._disp_rgb is None
            or self._disp_codes is None
            or self._disp_codes.shape[1] != cols
        ):
            pooled = columns_max(self._codes, cols, axis=1)
            if pooled is self._codes:
                pooled = pooled.copy()
            self._disp_codes = np.ascontiguousarray(pooled)
            self._regather_display()
        return self._disp_image

    def _write_top_row(self, line: np.ndarray, scroll: bool) -> None:
        """Put ``line`` in row 0, scrolling the older rows down if ``scroll``."""
        codes = self._codes
        if codes is None or codes.shape != (self._history_size, self._fft_size):
            self._render_image()
            return
        self._ensure_lut()
        row = quantize_db(line)
        if scroll:
            # NumPy handles the overlapping copies correctly.
            codes[1:] = codes[:-1]
        codes[0] = row
        self._version += 1
        disp, pixels = self._disp_codes, self._disp_rgb
        if disp is not None and pixels is not None:
            pooled = columns_max(row, disp.shape[1])
            if scroll:
                disp[1:] = disp[:-1]
                pixels[1:] = pixels[:-1]
            disp[0] = pooled
            pixels[0] = self._lut32[pooled].view(np.uint8).reshape(-1, 4)

    def _render_image(self):
        """Rebuild the level codes from the history, vectorized.

        The waterfall is ``history_size`` rows by ``fft_size`` columns, with
        the newest line at the top. New lines only scroll the codes (see
        :meth:`add_line`); colors come from a lookup table, so a change of
        levels, colors or theme re-maps pixels without coming here.
        """
        self._disp_codes = None
        self._disp_rgb = None
        self._disp_image = None
        self._version += 1
        if len(self._history) == 0:
            self._codes = None
            return
        # add_line guarantees each row is already fft_size long.
        lines = np.stack(list(reversed(self._history)))
        codes = np.zeros((self._history_size, self._fft_size), dtype=np.uint16)
        codes[: lines.shape[0]] = quantize_db(lines)
        self._codes = codes
        self._ensure_lut()

    # ------------------------------------------------------------------
    # Internals: interaction
    # ------------------------------------------------------------------

    def _margins(self) -> Tuple[int, int, int, int]:
        left, right = plot_side_margins(self._fm)
        bottom = 4 + (self._fm.height() + 8 if self._freq_axis else 0)
        return left, 4, right, bottom

    def _plot_rect(self) -> "QRectF":
        left, top, right, bottom = self._margins()
        w = self._canvas.width() - left - right
        h = self._canvas.height() - top - bottom
        return QRectF(left, top, max(0, w), max(0, h))

    def _hz_per_px(self) -> float:
        plot = self._plot_rect()
        return self._sample_rate / plot.width() if plot.width() > 0 else 0.0

    def _snapped_frequency_at(self, x: float) -> Optional[float]:
        freq = self.frequency_at(x)
        if freq is None:
            return None
        step = snap_step_hz(self._hz_per_px())
        return round(freq / step) * step

    def _freq_to_x(self, freq: float, plot: "QRectF") -> float:
        f_start = self._center_freq - self._sample_rate / 2
        return plot.left() + (freq - f_start) / self._sample_rate * plot.width()

    def _set_hover(self, pos: Optional["QPointF"]) -> None:
        self._hover = QPointF(pos) if pos is not None else None
        self._canvas.update()

    def _click_at(self, pos: "QPointF") -> bool:
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return False
        self.frequency_clicked.emit(float(freq))
        return True

    # -- rows and times ----------------------------------------------------
    def _arrival_times(self) -> np.ndarray:
        return np.fromiter(self._times, dtype=np.float64, count=len(self._times))

    def _seconds_per_row(self) -> Optional[float]:
        """Typical time between lines (median, so a pause doesn't skew it)."""
        if len(self._times) < 2:
            return None
        deltas = np.diff(self._arrival_times())
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            return None
        return float(np.median(deltas))

    def _row_ages(self) -> Optional[np.ndarray]:
        """Seconds between each image row's line and the newest (row 0 = 0)."""
        if not self._times:
            return None
        times = self._arrival_times()
        return times[-1] - times[::-1]

    def _row_steps_newest_first(self) -> Optional[np.ndarray]:
        """Each row's history setting (seconds per row), row 0 first, or None
        if the steps aren't known for every row."""
        if len(self._row_steps) != len(self._times):
            return None
        steps = np.fromiter(
            self._row_steps, dtype=np.float64, count=len(self._row_steps)
        )
        return steps[::-1]

    @staticmethod
    def _gap_rows(
        ages: np.ndarray, seconds_per_row: float, steps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Rows that start an older run after a pause in reception.

        ``steps`` (seconds per row, newest first) raises the threshold for
        rows recorded at a slower history setting, whose spacing is their
        interval, not a pause.
        """
        if len(ages) < 2:
            return np.empty(0, dtype=np.intp)
        threshold = np.full(
            len(ages) - 1, max(_GAP_FACTOR * seconds_per_row, _GAP_MIN_S)
        )
        if steps is not None and len(steps) == len(ages):
            # The spacing after row i is the older row (i + 1)'s interval.
            threshold = np.maximum(threshold, _GAP_FACTOR * steps[1:])
        return np.nonzero(np.diff(ages) > threshold)[0] + 1

    def _row_at_y(self, y: float, plot: "QRectF") -> float:
        """Fractional row index under canvas ``y`` (0 = newest row's top)."""
        rows_px = plot.height() / self._history_size
        return (y - plot.top()) / rows_px if rows_px > 0 else 0.0

    def _time_of_row(self, row: float) -> Optional[float]:
        """Start time of (fractional, clamped) ``row``."""
        n = len(self._times)
        if n == 0:
            return None
        times = self._arrival_times()[::-1]  # newest first
        return float(np.interp(min(max(row, 0.0), n - 1), np.arange(n), times))

    def _row_of_time(self, t: float) -> Optional[float]:
        """Fractional row whose start time is ``t`` (newest row = 0)."""
        n = len(self._times)
        if n == 0:
            return None
        times = self._arrival_times()  # oldest first, increasing
        rows = np.arange(n - 1, -1, -1, dtype=np.float64)
        return float(np.interp(t, times, rows))

    # -- measurement -------------------------------------------------------
    def _drag_measure(self, start: "QPointF", end: "QPointF") -> None:
        plot = self._plot_rect()
        if not self._history or plot.width() <= 0:
            return
        xs = [min(max(v, plot.left()), plot.right()) for v in (start.x(), end.x())]
        ys = [min(max(v, plot.top()), plot.bottom()) for v in (start.y(), end.y())]
        f0, f1 = (self.frequency_at(x) for x in xs)
        rows = sorted(self._row_at_y(y, plot) for y in ys)
        n = len(self._history)
        newest = int(min(max(math.floor(rows[0]), 0), n - 1))
        oldest = int(min(max(math.ceil(rows[1]) - 1, newest), n - 1))
        t_new, t_old = self._time_of_row(newest), self._time_of_row(oldest)
        if None in (f0, f1, t_new, t_old):
            return
        self._measure = {
            "f_low": min(f0, f1),
            "f_high": max(f0, f1),
            "t_new": t_new,
            "t_old": t_old,
        }
        self._measuring = True
        self._canvas.update()

    def _finish_measure(self) -> None:
        self._measuring = False
        m = self._measure
        if m is not None and m["f_high"] - m["f_low"] < max(1.0, self._hz_per_px()):
            self._measure = None  # a vertical line: nothing to measure
        self._canvas.update()

    def _measure_rows(self) -> Optional[Tuple[int, int]]:
        """(newest, oldest) row indices the measurement covers now."""
        m = self._measure
        if m is None or not self._times:
            return None
        if m["t_new"] < self._times[0] - 1e-9:
            return None  # scrolled out of the history
        newest = self._row_of_time(m["t_new"])
        oldest = self._row_of_time(max(m["t_old"], self._times[0]))
        if newest is None or oldest is None:
            return None
        return int(round(newest)), int(round(oldest))

    def _measure_stats(self) -> Dict[str, float]:
        m = self._measure
        assert m is not None
        f_low, f_high = m["f_low"], m["f_high"]
        spr = self._seconds_per_row() or 0.0
        rows = self._measure_rows()
        # A row spans its own interval: the newest row's time runs on to the
        # next row's start (or one typical row, for the newest of all).
        duration = max(0.0, m["t_new"] - m["t_old"])
        if rows is not None:
            newest = rows[0]
            if newest > 0:
                later = self._time_of_row(newest - 1)
                duration += max(0.0, (later or m["t_new"]) - m["t_new"])
            else:
                duration += spr
        peak = float("nan")
        peak_freq = float("nan")
        if rows is not None:
            newest, oldest = rows
            n_bins = self._fft_size
            f_start = self._center_freq - self._sample_rate / 2
            lo = int(math.floor((f_low - f_start) / self._sample_rate * n_bins))
            hi = int(math.ceil((f_high - f_start) / self._sample_rate * n_bins))
            lo, hi = max(0, lo), min(n_bins, max(hi, lo + 1))
            n = len(self._history)
            if lo < n_bins and hi > 0:
                block = np.stack(
                    [self._history[n - 1 - r][lo:hi] for r in range(newest, oldest + 1)]
                ).astype(np.float64)
                block[~np.isfinite(block)] = -np.inf
                if block.size and np.isfinite(block.max()):
                    peak = float(block.max())
                    col = int(np.unravel_index(np.argmax(block), block.shape)[1])
                    peak_freq = f_start + (lo + col + 0.5) / n_bins * self._sample_rate
        return {
            "f_low": f_low,
            "f_high": f_high,
            "bandwidth": f_high - f_low,
            "center": (f_low + f_high) / 2,
            "duration": duration,
            "peak_db": peak,
            "peak_freq": peak_freq,
        }

    def _measure_lines(self) -> List[str]:
        stats = self._measure_stats()
        decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
        lines = [
            f"Δf {format_span_hz(stats['bandwidth'])}"
            f"  Δt {format_duration(stats['duration'])}",
            f"Center {format_mhz(stats['center'], decimals)} MHz",
        ]
        if math.isfinite(stats["peak_db"]):
            peak = f"Peak {stats['peak_db']:.0f} dBFS"
            if self._noise_db is not None and stats["peak_db"] - self._noise_db >= 1:
                peak += f" · SNR {stats['peak_db'] - self._noise_db:.0f} dB"
            lines.append(peak)
        return lines

    # -- menus -------------------------------------------------------------
    def _display_menu(self, menu: "QMenu") -> None:
        """Add the display options (shared by the context and options menus)."""
        pause = menu.addAction("&Resume" if self._paused else "&Pause")
        pause.setShortcut("P")
        pause.triggered.connect(lambda: self.set_paused(not self._paused))

        history = menu.addMenu("&History")
        group = QActionGroup(history)
        for label, seconds in _HISTORY_CHOICES:
            act = history.addAction(label)
            act.setCheckable(True)
            act.setChecked(seconds == self._history_s)
            group.addAction(act)
            act.triggered.connect(
                lambda _c=False, s=seconds: self.set_history_seconds(s)
            )

        levels = menu.addMenu("&Levels")
        auto = levels.addAction("&Auto (follow noise and signals)")
        auto.setCheckable(True)
        auto.setChecked(self._auto_levels)
        auto.triggered.connect(self.set_auto_levels)
        levels.addSeparator()
        for index, db in enumerate(_RANGE_CHOICES):
            act = levels.addAction(f"{db} dB below full scale")
            act.triggered.connect(lambda _c=False, i=index: self._on_range_changed(i))
        if self._noise_db is not None:
            levels.addSeparator()
            snap = levels.addAction("&Fit to Current Signals")
            snap.setToolTip("Set manual levels once from the current noise and peak")
            snap.triggered.connect(self._fit_levels_once)

        colors = menu.addMenu("C&olors")
        group = QActionGroup(colors)
        for name in self.COLORMAPS:
            act = colors.addAction(name.capitalize())
            act.setCheckable(True)
            act.setChecked(name == self._colormap_name)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, n=name: self.set_colormap(n))

        menu.addSeparator()
        menu.addAction(self._clear_action)
        save = menu.addAction("&Save Image...")
        save.setEnabled(bool(self._history))
        save.triggered.connect(self._save_image_dialog)

    def _fit_levels_once(self) -> None:
        if self._noise_db is None or self._peak_db is None:
            return
        self._user_set_levels(*auto_level_targets(self._noise_db, self._peak_db))

    def _show_options_menu(self) -> None:
        menu = QMenu(self)
        self._display_menu(menu)
        button = self._options_btn
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        menu.deleteLater()

    def _show_context_menu(self, pos: "QPointF", global_pos) -> None:
        """Right-click: tune/bookmark/copy the frequency under the cursor, the
        measurement actions, then the display options."""
        menu = QMenu(self)
        freq = self._snapped_frequency_at(pos.x())
        if freq is not None:
            decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
            text = f"{format_mhz(freq, decimals)} MHz"
            tune = menu.addAction(f"&Tune to {text}")
            tune.triggered.connect(lambda: self.frequency_clicked.emit(float(freq)))
            mark = menu.addAction(f"&Bookmark {text}")
            mark.triggered.connect(lambda: self.bookmark_requested.emit(float(freq)))
            copy = menu.addAction("Copy &Frequency")
            copy.triggered.connect(
                lambda: self._copy_text(format_mhz(freq, max(decimals, 6)))
            )
            menu.addSeparator()
        if self._measure is not None:
            stats = self._measure_stats()
            center = stats["center"]
            tune_c = menu.addAction("Tune to Measured Ce&nter")
            tune_c.triggered.connect(lambda: self.frequency_clicked.emit(float(center)))
            copy_m = menu.addAction("Copy &Measurement")
            copy_m.triggered.connect(
                lambda: self._copy_text("\n".join(self._measure_lines()))
            )
            clear_m = menu.addAction("Clear M&easurement")
            clear_m.triggered.connect(self.clear_measurement)
            menu.addSeparator()
        self._display_menu(menu)
        menu.exec(global_pos)
        menu.deleteLater()

    @staticmethod
    def _copy_text(text: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    def _save_image_dialog(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Waterfall Image", "waterfall.png", "PNG image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not self.export_image(path):
            QMessageBox.warning(
                self, "Save Failed", f"Could not save the waterfall image to:\n{path}"
            )

    # ------------------------------------------------------------------
    # Internals: painting
    # ------------------------------------------------------------------

    def _paint_canvas(self, canvas: "QWidget") -> None:
        p = get_palette()
        painter = QPainter(canvas)
        try:
            painter.fillRect(canvas.rect(), p.qcolor("plot_bg"))
            plot = self._plot_rect()
            if plot.width() <= 2 or plot.height() <= 2:
                return
            painter.setFont(self._font)

            image = self._display_image(self._display_columns(plot))
            if image is not None:
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawImage(
                    plot, image, QRectF(0, 0, image.width(), image.height())
                )
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

            self._draw_time_axis(painter, plot, p)
            if self._freq_axis:
                self._draw_frequency_axis(painter, plot, p)

            # Frame around the plot, over the ends of the pause separators
            painter.setPen(QPen(p.qcolor("plot_grid"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(plot.adjusted(-0.5, -0.5, 0.5, 0.5))
            if self.hasFocus():
                draw_focus_frame(painter, plot, p)

            self._draw_highlights(painter, plot)

            self._draw_passband(painter, plot, p)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_center_marker(painter, plot, p)

            if self._codes is None:
                draw_placeholder(
                    painter,
                    plot,
                    "Waiting for data",
                    "The waterfall fills in once receiving starts",
                    canvas.font(),
                    p,
                )
            painter.setFont(self._font)
            self._draw_measurement(painter, plot, p)
            if self._paused:
                self._draw_paused_badge(painter, plot, p)
            if self._hover is not None and not self._measuring:
                # Also shown before any data: clicking tunes either way.
                self._draw_hover(painter, plot, p)
        finally:
            painter.end()

    def _time_axis_layout(
        self, plot: "QRectF"
    ) -> Tuple[List[float], List[Tuple[float, float, str]]]:
        """Where the time axis goes: ``(separator_ys, labels)``.

        ``separator_ys`` are the pauses in reception (one line across the
        plot each). Each label is ``(tick_y, label_y, text)``: its tick marks
        the row the age belongs to and its text is centered on ``label_y``,
        the tick's y kept inside the plot's vertical extent. The overlap
        checks use ``label_y``, where the text really is, so no two labels
        come closer than a line height. The newest line's ``0s`` always
        shows; then each pause's label (the older lines' age), then regular
        ticks within the newest continuous run, wherever they fit.
        """
        spr = self._seconds_per_row()
        ages = self._row_ages()
        if spr is None or ages is None or plot.height() <= 0:
            return [], []
        rows_px = plot.height() / self._history_size
        fh = self._fm.height()
        min_sep = fh + 2
        y_min = plot.top() + fh / 2
        y_max = max(y_min, plot.bottom() - fh / 2)
        labels: List[Tuple[float, float, str]] = []

        def place(tick_y: float, text: str) -> None:
            label_y = min(max(tick_y, y_min), y_max)
            if all(abs(label_y - other) >= min_sep for _t, other, _s in labels):
                labels.append((tick_y, label_y, text))

        gaps = self._gap_rows(ages, spr, self._row_steps_newest_first())
        separators = [round(plot.top() + row * rows_px) + 0.5 for row in gaps.tolist()]

        # Regular ticks within the newest continuous run of lines.
        run = ages[: int(gaps[0])] if gaps.size else ages
        step = time_step(
            spr * self._history_size * max(28.0, fh * 2.2) / plot.height(),
            max_age=float(run[-1]),
        )
        ticks: List[Tuple[float, str]] = []
        for age in nice_ticks(0.0, float(run[-1]) + 1e-9, step):
            # Fractional row between the two lines that bracket this age (the
            # run has no pauses, so the pair is always close together).
            i = min(int(np.searchsorted(run, age)), len(run) - 1)
            if i == 0:
                row = 0.0
            else:
                a0, a1 = float(run[i - 1]), float(run[i])
                row = i - 1 + ((age - a0) / (a1 - a0) if a1 > a0 else 1.0)
            ticks.append((plot.top() + (row + 0.5) * rows_px, format_age(age, step)))

        # "0s" first: right after Stop -> Start the newest rows sit just above
        # a pause, and their label must not be the older lines' age.
        if ticks:
            place(*ticks[0])
        for row, y in zip(gaps.tolist(), separators, strict=True):
            place(y, format_elapsed(float(ages[row])))
        for tick in ticks[1:]:
            place(*tick)
        return separators, labels

    def _draw_time_axis(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Seconds-ago labels in the left margin (0s at the newest line).

        Labels are placed at the rows whose lines actually arrived that long
        ago, so they stay right when reception was paused and resumed. Each
        pause gets a separator across the plot, labelled with the age of the
        older lines below it (see :meth:`_time_axis_layout`).
        """
        separators, labels = self._time_axis_layout(plot)
        if separators:
            painter.setPen(QPen(p.qcolor("plot_bg", 230), 1))
            for y in separators:
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        if not labels:
            return
        fh = self._fm.height()
        label_right = plot.left() - 6
        right_align = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tick_pen = QPen(p.qcolor("plot_grid"), 1)
        axis_color = p.qcolor("plot_axis")
        for tick_y, label_y, text in labels:
            # A 3 px tick ending on the frame (not on the image's first column).
            painter.setPen(tick_pen)
            painter.drawLine(
                QPointF(plot.left() - 4, tick_y), QPointF(plot.left() - 1, tick_y)
            )
            painter.setPen(axis_color)
            painter.drawText(
                QRectF(0, label_y - fh / 2, label_right - 2, fh), right_align, text
            )

    def _draw_frequency_axis(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """MHz labels under the plot, like the spectrum's (when it's hidden)."""
        fm = self._fm
        fh = fm.height()
        f_start = self._center_freq - self._sample_rate / 2
        f_end = self._center_freq + self._sample_rate / 2
        ticks, decimals = freq_ticks(f_start, f_end, plot.width(), fm)
        label_y = plot.bottom() + 5
        right_align = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        painter.setFont(self._unit_font)
        painter.setPen(p.qcolor("plot_axis"))
        painter.drawText(QRectF(0, label_y, plot.left() - 6, fh), right_align, "MHz")
        painter.setFont(self._font)
        tick_pen = QPen(p.qcolor("plot_grid"), 1)
        min_left = plot.left() + 3
        max_right = self._canvas.width() - 2
        last_right = -1e9
        center = int(Qt.AlignmentFlag.AlignCenter)
        for f in ticks:
            x = self._freq_to_x(f, plot)
            painter.setPen(tick_pen)
            xi = round(x) + 0.5
            painter.drawLine(
                QPointF(xi, plot.bottom() + 1), QPointF(xi, plot.bottom() + 4)
            )
            text = format_mhz(f, decimals)
            tw = fm.horizontalAdvance(text)
            centered = x - tw / 2
            left = min(max(centered, min_left), max_right - tw)
            if abs(left - centered) > tw * 0.3 or left < last_right + 8:
                continue
            painter.setPen(p.qcolor("plot_axis"))
            painter.drawText(QRectF(left, label_y, tw, fh), center, text)
            last_right = left + tw

    def _draw_highlights(self, painter: "QPainter", plot: "QRectF") -> None:
        if not self._highlights:
            return
        sx = plot.width() / self._fft_size
        sy = plot.height() / self._history_size
        for time_start, time_end, freq_start, freq_end, color in self._highlights:
            rect = QRectF(
                plot.left() + freq_start * sx,
                plot.top() + time_start * sy,
                (freq_end - freq_start) * sx,
                (time_end - time_start) * sy,
            )
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def _draw_passband(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """A thin bracket along the top edge spanning the demodulated channel.

        Drawn with a plot-colored halo so it reads on any colormap, and only
        over the newest few rows, so it never hides the signal history.
        """
        extent = passband_px(self._passband, plot, self._sample_rate)
        if extent is None:
            return
        x0, x1 = extent
        if x1 <= plot.left() or x0 >= plot.right():
            return
        top = plot.top()
        tick = _PASSBAND_TICK_PX
        halo = p.qcolor("plot_bg", 170)
        color = p.qcolor(PASSBAND_TOKEN)
        painter.save()
        painter.setClipRect(plot)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        # The bracket is a 2 px bar with a short tick down at each end, on a
        # 1 px plot-colored halo.
        ends = (x0, x1 - 1) if x1 - x0 >= 6 else ()
        painter.fillRect(QRectF(x0 - 1, top, x1 - x0 + 2, 3), halo)
        for x in ends:
            painter.fillRect(QRectF(x - 1, top + 3, 3, tick - 2), halo)
        painter.fillRect(QRectF(x0, top, x1 - x0, 2), color)
        for x in ends:
            painter.fillRect(QRectF(x, top + 2, 1, tick - 2), color)
        painter.restore()

    def _draw_center_marker(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Small flag on the top edge at the tuned (center) frequency."""
        x = round(plot.center().x()) + 0.5
        tri = QPolygonF(
            [
                QPointF(x - 5, plot.top()),
                QPointF(x + 5, plot.top()),
                QPointF(x, plot.top() + 6),
            ]
        )
        painter.setPen(QPen(p.qcolor("plot_bg"), 1))
        painter.setBrush(p.qcolor("plot_marker"))
        painter.drawPolygon(tri)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_measurement(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """The measured box, following its rows as they scroll, and its
        readout (bandwidth, duration, center, peak)."""
        m = self._measure
        rows = self._measure_rows()
        if m is None or rows is None:
            if m is not None and not self._measuring:
                self._measure = None  # scrolled out of the history
            return
        rows_px = plot.height() / self._history_size
        newest, oldest = rows
        x0 = self._freq_to_x(m["f_low"], plot)
        x1 = self._freq_to_x(m["f_high"], plot)
        box = QRectF(
            x0, plot.top() + newest * rows_px, x1 - x0, (oldest - newest + 1) * rows_px
        ).intersected(plot)
        if box.width() <= 0 or box.height() <= 0:
            return
        painter.save()
        painter.setClipRect(plot)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(box, p.qcolor("accent", 36))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(p.qcolor("plot_bg", 200), 3))
        painter.drawRect(box)
        pen = QPen(p.qcolor("plot_text"), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(box)
        painter.restore()
        self._draw_box_readout(painter, self._measure_lines(), box, plot, p)

    def _draw_box_readout(
        self, painter: "QPainter", lines: List[str], box: "QRectF", plot: "QRectF", p
    ) -> None:
        """A small boxed readout beside ``box``, kept inside the plot."""
        fm = self._fm
        pad_x, pad_y = 6, 3
        width = max(fm.horizontalAdvance(line) for line in lines) + 2 * pad_x
        height = fm.height() * len(lines) + 2 * pad_y
        x = box.right() + 8
        if x + width > plot.right() - 2:
            x = box.left() - 8 - width
        x = min(max(plot.left() + 2, x), plot.right() - 2 - width)
        y = min(max(plot.top() + 4, box.top()), plot.bottom() - 2 - height)
        frame = QRectF(x, y, width, height)
        painter.setPen(QPen(p.qcolor("plot_grid"), 1))
        painter.setBrush(p.qcolor("plot_bg", 235))
        painter.drawRoundedRect(frame, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(p.qcolor("plot_text"))
        for i, line in enumerate(lines):
            painter.drawText(
                QPointF(x + pad_x, y + pad_y + i * fm.height() + fm.ascent()), line
            )

    def _draw_paused_badge(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """A "Paused" tag in the plot's top-left corner while frozen."""
        fm = self._fm
        text = "PAUSED"
        if self._skipped:
            text += f" · {self._skipped} skipped"
        painter.setFont(self._unit_font)
        ufm = QFontMetrics(self._unit_font)
        width = ufm.horizontalAdvance(text) + 12
        badge = QRectF(plot.left() + 6, plot.top() + 10, width, fm.height() + 4)
        painter.setPen(QPen(p.qcolor("plot_marker"), 1))
        painter.setBrush(p.qcolor("plot_bg", 225))
        painter.drawRoundedRect(badge, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(p.qcolor("plot_marker"))
        painter.drawText(badge, int(Qt.AlignmentFlag.AlignCenter), text)
        painter.setFont(self._font)

    def _draw_hover(self, painter: "QPainter", plot: "QRectF", p) -> None:
        pos = self._hover
        if pos is None or not plot.contains(pos):
            return
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return
        x, y = pos.x(), pos.y()
        # Thin lines with a halo stay visible on any colormap; the row line
        # is fainter, so the frequency line reads as the one that tunes.
        painter.setPen(QPen(p.qcolor("plot_bg", 150), 3))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QPen(p.qcolor("plot_text", 220), 1))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QPen(p.qcolor("plot_text", 90), 1))
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
        lines = [f"{format_mhz(freq, decimals)} MHz"]
        row = int((y - plot.top()) / plot.height() * self._history_size)
        if 0 <= row < len(self._history):
            line = self._history[len(self._history) - 1 - row]
            n = len(line)
            # Strongest bin under this pixel, matching the decimated image.
            frac = (x - plot.left()) / plot.width()
            half = n / plot.width() / 2
            lo = max(0, min(n - 1, int(frac * n - half)))
            hi = min(n, max(lo + 1, int(frac * n + half) + 1))
            window = np.asarray(line[lo:hi], dtype=np.float64)
            finite = window[np.isfinite(window)]
            if finite.size:
                level = float(finite.max())
                detail = f"{level:.0f} dBFS"
                if self._noise_db is not None and level - self._noise_db >= 1:
                    detail += f" · SNR {level - self._noise_db:.0f} dB"
            else:
                detail = "-- dBFS"
            lines.append(detail)
            age = self._times[-1] - self._times[len(self._times) - 1 - row]
            lines.append(f"{format_elapsed(age)} ago")
        lines.append("Click to tune · drag to measure")
        draw_readout(painter, lines, x, plot, self._fm, p)
