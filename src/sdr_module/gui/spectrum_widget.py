"""
Spectrum analyzer widget.

Provides real-time spectrum visualization with:
- FFT-based power spectrum display (dBFS)
- Peak hold and exponential averaging
- Frequency axis with "nice" tick steps and a unit caption
- Hover readout (frequency and level), tuned-frequency marker, click-to-tune
- Zoom (x1 to x16, shared with the waterfall through a
  :class:`~sdr_module.gui.plot_view.PlotView`), Ctrl+wheel around the cursor
- A vertical range that follows the noise floor and the strongest signal
  (Auto), or a fixed one that the dB axis drags and scrolls
- A right-click menu (tune, bookmark, copy, zoom, peak hold, range, FFT size,
  window), labels at bookmarked frequencies, and a dimmed "Stopped" state

The widget is a header strip (title, hint and trace controls) above a plot
canvas. All plot colors come from the active theme palette (``plot_*``
tokens), so the plot follows the dark/light theme.

The module also exports small axis helpers (:func:`plot_side_margins`,
:func:`nice_step`, :func:`axis_font` ...) that the waterfall uses too, so the
spectrum and waterfall share their horizontal plot margins and their
frequency axes line up vertically.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..dsp.frame_spectrum import (
    DEFAULT_WINDOW,
    FFT_SIZES,
    WINDOW_HINTS,
    WINDOW_LABELS,
    WINDOWS,
    normalize_fft_size,
    normalize_window,
    rbw_hz,
)
from ..utils.tooltips import get_short_tip
from .plot_view import ZOOM_LEVELS, PlotView, normalize_zoom
from .themes import get_palette, mono_font, set_role, theme_notifier
from .tuning import snap_click_frequency

if TYPE_CHECKING:
    from PyQt6.QtGui import QPainter

try:
    from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import (
        QActionGroup,
        QBrush,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QLinearGradient,
        QPainter,
        QPen,
        QPixmap,
        QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMenu,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


# ---------------------------------------------------------------------------
# Axis helpers shared with the waterfall
# ---------------------------------------------------------------------------

#: Point size of axis tick labels and readouts drawn on the plots.
AXIS_FONT_PT = 8.0

#: Right plot margin in pixels (room for a clamped last tick label).
PLOT_MARGIN_RIGHT = 12

#: Content margins (left, top, right, bottom) of the header strip above each
#: plot. The right margin puts the last header control's right edge on the
#: plot frame (drawn one pixel outside the plot area), so they line up.
HEADER_MARGINS = (10, 3, PLOT_MARGIN_RIGHT - 1, 3)

#: Palette token that shades the demodulated channel (the passband).
PASSBAND_TOKEN = "accent"

#: Opacity (0-255) of the spectrum's passband (fill, edge lines) by theme
#: darkness: calmer on a white plot, where the accent is close to the trace.
PASSBAND_ALPHAS = {True: (40, 150), False: (34, 110)}

#: Narrowest the passband shading gets, in pixels.
PASSBAND_MIN_PX = 3.0

#: A jump of the noise floor (the trace's median) at least this big between
#: two frames is a gain or AGC change: peak hold and the average start over
#: (peaks held at the old gain would sit far above the new trace).
_PEAK_RESET_JUMP_DB = 6.0

#: Frames a click looks back over for the signal it snaps to (about half a
#: second at 30 frames/s): a station's whole width, and a keyed signal
#: between its dits (see snap_click_frequency's ``recent_db``).
SNAP_HOLD_FRAMES = 15

# Averaging combo entries: label -> number of frames (0 = off).
_AVG_CHOICES = (("Off", 0), ("2", 2), ("4", 4), ("8", 8), ("16", 16), ("32", 32))

#: Fixed vertical ranges offered (dB from the top of the plot to its floor).
RANGE_SPANS = (60, 80, 100, 120)
#: Where the vertical range may go (dBFS).
_RANGE_LIMITS_DB = (-200.0, 40.0)
#: Narrowest and widest vertical range (dB).
_RANGE_SPAN_LIMITS = (20.0, 160.0)
# Auto range: the floor this far below the noise floor, the top this far
# above the strongest signal, at least this span; both on a 5 dB grid and
# moved only when the target is a grid step away (so the axis doesn't twitch).
_AUTO_RANGE_BELOW_NOISE_DB = 15.0
_AUTO_RANGE_ABOVE_PEAK_DB = 10.0
_AUTO_RANGE_MIN_SPAN_DB = 50.0
_AUTO_RANGE_GRID_DB = 5.0
#: dB a notch of the wheel (or a key) moves the range.
_RANGE_WHEEL_DB = 5.0

#: Opacity of the plot-colored veil over a stopped (stale) plot.
STALE_VEIL_ALPHA = 150

#: Widest a bookmark label gets (its name is elided past that), in pixels.
BOOKMARK_LABEL_MAX_PX = 120

#: Rows at least this long (FFT bins) are decimated with reduceat (see
#: columns_max).
_REDUCEAT_MIN_BINS = 16384


def axis_font() -> "QFont":
    """Monospace font for plot tick labels and readouts."""
    return mono_font(AXIS_FONT_PT)


def plot_side_margins(fm: "QFontMetrics") -> Tuple[int, int]:
    """Left/right plot margins shared by the spectrum and the waterfall.

    Both plots compute their margins from the same axis font, so their plot
    areas start and end at the same x and a frequency lines up vertically.
    The left margin fits the widest y label (``-120``, ``dBFS``, ``60s``).
    """
    widest = max(fm.horizontalAdvance(s) for s in ("-120", "dBFS", "MHz", "120s"))
    return widest + 12, PLOT_MARGIN_RIGHT


def nice_step(raw: float) -> float:
    """Smallest step of the form 1, 2 or 5 x 10^n that is >= ``raw``."""
    if not math.isfinite(raw) or raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = 10.0**exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = mult * base
        if step >= raw * (1 - 1e-9):
            return step
    return 10.0 * base  # pragma: no cover - loop always returns


def nice_ticks(lo: float, hi: float, step: float) -> List[float]:
    """Multiples of ``step`` within ``[lo, hi]`` (inclusive, float-safe)."""
    if step <= 0 or not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        return []
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    if last - first > 1000:  # defensive: never loop forever on bad input
        return []
    return [k * step for k in range(first, last + 1)]


def mhz_decimals(step_hz: float) -> int:
    """Decimals needed to show MHz labels that are ``step_hz`` apart."""
    if step_hz <= 0:
        return 3
    return int(max(0, min(6, -math.floor(math.log10(step_hz / 1e6) + 1e-9))))


def format_mhz(freq_hz: float, decimals: int) -> str:
    """``100.2`` style MHz label (no unit); avoids a ``-0.0`` label."""
    value = freq_hz / 1e6
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def snap_step_hz(hz_per_px: float) -> float:
    """Resolution a click can meaningfully pick: 1, 10, 100 ... Hz."""
    if not math.isfinite(hz_per_px) or hz_per_px <= 1:
        return 1.0
    return 10.0 ** math.floor(math.log10(hz_per_px))


def readout_decimals(step_hz: float) -> int:
    """MHz decimals for a readout with ``step_hz`` resolution (3 to 6)."""
    if step_hz <= 0:
        return 6
    return int(max(3, min(6, round(6 - math.log10(step_hz)))))


def freq_ticks(
    f_start: float, f_end: float, width_px: float, fm: "QFontMetrics"
) -> Tuple[List[float], int]:
    """Nice frequency ticks for a plot ``width_px`` wide.

    Returns the tick frequencies (Hz) and the MHz decimals for their labels.
    The step grows until labels are at least one label width plus a gap apart.
    """
    span = f_end - f_start
    if span <= 0 or width_px <= 0:
        return [], 3
    step = nice_step(span * 60.0 / width_px)
    decimals = mhz_decimals(step)
    for _ in range(12):
        decimals = mhz_decimals(step)
        widest = max(
            fm.horizontalAdvance(format_mhz(f, decimals)) for f in (f_start, f_end)
        )
        if step / span * width_px >= widest + 16:
            break
        step = nice_step(step * 1.0001)
    return nice_ticks(f_start, f_end, step), decimals


def tuned_mhz_text(freq_hz: float, min_decimals: int = 3) -> str:
    """``"146.520 MHz"``, ``"99.82001 MHz"``: up to 1 Hz, at least
    ``min_decimals`` decimals (like the toolbar's FREQ readout)."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(min_decimals, '0')} MHz"


def format_hz(hz: float) -> str:
    """A small bandwidth with three significant digits: ``586 Hz``,
    ``1.76 kHz``, ``14.6 kHz``."""
    hz = abs(float(hz))
    if hz < 1e3:
        return f"{hz:.0f} Hz"
    if hz < 1e6:
        return f"{hz / 1e3:.3g} kHz"
    return f"{hz / 1e6:.3g} MHz"


class WheelTuner:
    """Turns mouse wheel turns over a plot into tuning steps: one per notch
    (high-resolution wheels and touchpads add up to notches), ten with
    Shift. Ctrl+wheel is not tuning (it is left for zooming)."""

    NOTCH = 120  # angleDelta of one wheel notch (eighths of a degree)

    def __init__(self) -> None:
        self._accum = 0

    def steps(self, event) -> Optional[int]:
        """Steps to tune for this wheel event (0: not a whole notch yet),
        or None when the event is not for tuning."""
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            return None
        angle = event.angleDelta()
        delta = angle.y() or angle.x()  # Shift turns some wheels sideways
        if not delta:
            return None
        if (delta > 0) != (self._accum > 0):
            self._accum = 0  # turned the other way: start afresh
        self._accum += delta
        notches = int(self._accum / self.NOTCH)
        self._accum -= notches * self.NOTCH
        if notches and mods & Qt.KeyboardModifier.ShiftModifier:
            notches *= 10
        return notches


class NotchCounter:
    """Whole wheel notches from wheel events (high-resolution wheels and
    touchpads add up), for zooming and for moving the dB range."""

    NOTCH = 120

    def __init__(self) -> None:
        self._accum = 0

    def notches(self, event) -> int:
        angle = event.angleDelta()
        delta = angle.y() or angle.x()
        if not delta:
            return 0
        if (delta > 0) != (self._accum > 0):
            self._accum = 0
        self._accum += delta
        notches = int(self._accum / self.NOTCH)
        self._accum -= notches * self.NOTCH
        return notches


def draw_readout(
    painter: "QPainter",
    lines: List[str],
    anchor_x: float,
    plot: "QRectF",
    fm: "QFontMetrics",
    palette,
    dim_last: bool = True,
    avoid: Optional["QRectF"] = None,
) -> "QRectF":
    """Draw a small boxed readout next to ``anchor_x`` at the top of ``plot``
    and return its box.

    The box flips to the left of the cursor near the right edge so it never
    leaves the plot. With ``avoid`` (another readout's box), it goes where it
    doesn't cover that box: on the other side of the cursor, or below (or
    above) that box; when there is no such place inside the plot, nothing is
    drawn (an empty box is returned).
    """
    pad_x, pad_y = 6, 3
    text_w = max(fm.horizontalAdvance(line) for line in lines)
    line_h = fm.height()
    box_w = text_w + 2 * pad_x
    box_h = line_h * len(lines) + 2 * pad_y

    def place(side: int, top: float) -> "QRectF":
        x = anchor_x + 10 if side > 0 else anchor_x - 10 - box_w
        if side > 0 and x + box_w > plot.right() - 2:
            x = anchor_x - 10 - box_w
        elif side < 0 and x < plot.left() + 2:
            x = anchor_x + 10
        x = min(max(plot.left() + 2, x), max(plot.left() + 2, plot.right() - 2 - box_w))
        return QRectF(x, top, box_w, box_h)

    box = place(1, plot.top() + 6)
    if avoid is not None and not avoid.isEmpty() and box.intersects(avoid):
        tops = (plot.top() + 6, avoid.bottom() + 4, avoid.top() - 4 - box_h)
        candidates = [place(side, top) for top in tops for side in (-1, 1)]
        free = [
            c
            for c in candidates
            if not c.intersects(avoid)
            and c.top() >= plot.top()
            and c.bottom() <= plot.bottom()
        ]
        if not free:
            return QRectF()
        box = free[0]
    x, y = box.left(), box.top()
    painter.setPen(QPen(palette.qcolor("plot_grid"), 1))
    painter.setBrush(palette.qcolor("plot_bg", 235))
    painter.drawRoundedRect(box, 3, 3)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for i, line in enumerate(lines):
        # The last line is a hint ("Click to tune"): quieter than the values.
        dim = dim_last and i == len(lines) - 1 and len(lines) > 1
        painter.setPen(palette.qcolor("plot_axis" if dim else "plot_text"))
        painter.drawText(
            QPointF(x + pad_x, y + pad_y + i * line_h + fm.ascent()),
            line,
        )
    return box


def draw_placeholder(
    painter: "QPainter", plot: "QRectF", title: str, hint: str, font: "QFont", palette
) -> None:
    """Centered two-line empty-state message inside ``plot``.

    The text sits on a plot-colored backdrop so grid lines and the tuned
    frequency marker don't run through it. The hint wraps on narrow plots.
    The title is bold and a step larger than ``font`` (the hint's font),
    whether ``font`` is sized in pixels (as the stylesheet sizes widget
    fonts) or in points.
    """
    title_font = QFont(font)
    if font.pixelSize() > 0:
        title_font.setPixelSize(font.pixelSize() + 2)
    else:
        title_font.setPointSizeF(max(8.0, font.pointSizeF() + 1))
    title_font.setBold(True)
    tfm = QFontMetrics(title_font)
    hfm = QFontMetrics(font)
    pad_x, pad_y, gap = 14.0, 8.0, 4.0
    avail = max(40.0, plot.width() - 2 * pad_x - 8)
    wrap = int(Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap)
    hint_box = hfm.boundingRect(QRectF(0, 0, avail, 10_000).toRect(), wrap, hint).size()
    text_w = min(avail, max(tfm.horizontalAdvance(title), hint_box.width()))
    total = tfm.height() + gap + hint_box.height()
    top = plot.center().y() - total / 2
    backdrop = QRectF(
        plot.center().x() - text_w / 2 - pad_x,
        top - pad_y,
        text_w + 2 * pad_x,
        total + 2 * pad_y,
    ).intersected(plot.adjusted(1, 1, -1, -1))
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(palette.qcolor("plot_bg"))
    painter.drawRoundedRect(backdrop, 6, 6)
    painter.restore()
    painter.setFont(title_font)
    painter.setPen(palette.qcolor("plot_text"))
    painter.drawText(
        QRectF(plot.left(), top, plot.width(), tfm.height()),
        int(Qt.AlignmentFlag.AlignCenter),
        title,
    )
    painter.setFont(font)
    painter.setPen(palette.qcolor("plot_axis"))
    painter.drawText(
        QRectF(
            plot.center().x() - avail / 2,
            top + tfm.height() + gap,
            avail,
            hint_box.height(),
        ),
        wrap,
        hint,
    )


def draw_stale(
    painter: "QPainter", plot: "QRectF", text: str, font: "QFont", palette
) -> "QRectF":
    """Dim ``plot`` (the data in it is no longer live) and say why in a
    centered badge (``"Stopped - press Space"``). Returns the badge."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.fillRect(plot, palette.qcolor("plot_bg", STALE_VEIL_ALPHA))
    badge_font = QFont(font)
    badge_font.setBold(True)
    fm = QFontMetrics(badge_font)
    width = min(plot.width() - 8, fm.horizontalAdvance(text) + 24)
    height = fm.height() + 10
    badge = QRectF(
        plot.center().x() - width / 2, plot.center().y() - height / 2, width, height
    )
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(palette.qcolor("plot_grid"), 1))
    painter.setBrush(palette.qcolor("plot_bg", 235))
    painter.drawRoundedRect(badge, 5, 5)
    painter.setFont(badge_font)
    painter.setPen(palette.qcolor("plot_text"))
    painter.drawText(
        badge,
        int(Qt.AlignmentFlag.AlignCenter),
        fm.elidedText(text, Qt.TextElideMode.ElideRight, int(width - 12)),
    )
    painter.restore()
    return badge


def normalize_passband(
    low_hz: Optional[float], high_hz: Optional[float]
) -> Optional[Tuple[float, float]]:
    """``(low_hz, high_hz)`` as floats, or None when it describes no band.

    None for either bound, a non-numeric or non-finite bound, or
    ``low_hz >= high_hz`` all mean "hide the passband".
    """
    if low_hz is None or high_hz is None:
        return None
    try:
        low, high = float(low_hz), float(high_hz)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        return None
    return low, high


def passband_px(
    passband: Optional[Tuple[float, float]],
    plot: "QRectF",
    span_hz: float,
    anchor_hz: float = 0.0,
    view_offset_hz: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Canvas x extent ``(left, right)`` of a passband drawn on ``plot``.

    ``passband`` holds offsets (Hz) from the receiver's center frequency
    (the LO); ``plot`` shows ``span_hz``. ``view_offset_hz`` is the LO's
    offset from the middle of the plot (0 unless zoomed in off-center).
    ``anchor_hz`` is the tuned frequency's offset from the LO (the VFO's; 0
    when it is at the LO). The extent is snapped to whole pixels and at
    least :data:`PASSBAND_MIN_PX` wide (a CW or SSB channel on a 2.4 MHz span
    is narrower than one pixel), growing away from the tuned frequency for a
    one-sided (SSB) band. It is not clipped to the plot. None when there is
    no passband or nothing to map it onto.
    """
    if passband is None or span_hz <= 0 or plot.width() <= 0:
        return None
    low, high = passband
    scale = plot.width() / span_hz
    center = plot.center().x() + view_offset_hz * scale
    x0, x1 = center + low * scale, center + high * scale
    if x1 - x0 < PASSBAND_MIN_PX:
        if low >= anchor_hz:  # upper sideband: grow to the right of the marker
            x1 = x0 + PASSBAND_MIN_PX
        elif high <= anchor_hz:  # lower sideband: grow to the left
            x0 = x1 - PASSBAND_MIN_PX
        else:
            mid = (x0 + x1) / 2
            x0, x1 = mid - PASSBAND_MIN_PX / 2, mid + PASSBAND_MIN_PX / 2
    return float(math.floor(x0 + 0.5)), float(math.floor(x1 + 0.5))


def draw_passband(
    painter: "QPainter",
    plot: "QRectF",
    extent: Tuple[float, float],
    palette,
    fill_alpha: int,
    edge_alpha: int,
) -> None:
    """Shade ``extent`` (see :func:`passband_px`) over the full plot height.

    Edge lines mark the band's ends when it is wide enough to have visible
    ends; everything is clipped to ``plot``.
    """
    x0, x1 = extent
    if x1 <= plot.left() or x0 >= plot.right():
        return
    painter.save()
    painter.setClipRect(plot)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if fill_alpha > 0:
        painter.fillRect(
            QRectF(x0, plot.top(), x1 - x0, plot.height()),
            palette.qcolor(PASSBAND_TOKEN, fill_alpha),
        )
    if x1 - x0 >= 6:
        painter.setPen(QPen(palette.qcolor(PASSBAND_TOKEN, edge_alpha), 1))
        for x in (x0 + 0.5, x1 - 0.5):
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
    painter.restore()


def draw_landing_mark(painter: "QPainter", plot: "QRectF", x: float, palette) -> None:
    """Where a click would tune, when that isn't under the cursor (it snaps
    onto a signal): a thin accent line with a small flag at the top."""
    painter.save()
    painter.setClipRect(plot)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    xi = round(x) + 0.5
    painter.setPen(QPen(palette.qcolor(PASSBAND_TOKEN, 170), 1))
    painter.drawLine(QPointF(xi, plot.top()), QPointF(xi, plot.bottom()))
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(palette.qcolor(PASSBAND_TOKEN))
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(xi - 4, plot.top()),
                QPointF(xi + 4, plot.top()),
                QPointF(xi, plot.top() + 5),
            ]
        )
    )
    painter.restore()


def draw_focus_frame(painter: "QPainter", plot: "QRectF", palette) -> None:
    """Accent frame around ``plot``: the plot has the keyboard focus.

    Two pixels wide: the plot's own 1 px frame plus one more pixel outward,
    so it never hides data. Drawn as exact 1 px lines, like the rest of the
    plot chrome.
    """
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(palette.qcolor("accent"), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(plot.adjusted(-0.5, -0.5, 0.5, 0.5))
    painter.drawRect(plot.adjusted(-1.5, -1.5, 1.5, 1.5))
    painter.restore()


def fit_header_hint(owner: "QWidget", header: "QWidget", hint: "QLabel") -> None:
    """Hide a header strip's hint text when the strip is too narrow for it.

    The hint is a nicety; clipping it (or squeezing the controls) looks
    broken, so it simply disappears below the width it needs.
    """
    layout = header.layout()
    if layout is None:
        return
    needed = layout.sizeHint().width() + 24  # keep a gap before the controls
    if hint.isHidden():
        needed += hint.sizeHint().width() + max(0, layout.spacing())
    hint.setHidden(owner.width() < needed)


def columns_max(data: np.ndarray, cols: int, axis: int = -1) -> np.ndarray:
    """Peak-preserving decimation of ``data`` to ``cols`` columns along ``axis``.

    Each output column is the maximum of the bins that fall into it, so a
    narrow carrier never disappears when 2048 bins share ~800 pixels. Works
    on a single trace (1-D) or on a stack of waterfall rows (2-D, ``axis=1``).
    """
    data = np.asarray(data)
    n = data.shape[axis] if data.ndim else 0
    if cols <= 0 or n == 0:
        shape = list(data.shape) or [0]
        shape[axis] = 0
        return np.empty(shape, dtype=data.dtype if data.ndim else np.float64)
    if n <= cols:
        return data
    edges = (np.arange(cols) * n // cols).astype(np.intp)
    if data.ndim == 1 or n >= _REDUCEAT_MIN_BINS:
        # (2-D: a whole history of long rows streams through reduceat about
        # twice as fast as through the dozens of gathers below.)
        return np.maximum.reduceat(data, edges, axis=axis)
    # 2-D: reduceat along an inner axis is slow; each column spans only a
    # few bins, so a handful of strided gathers is 3-4x faster.
    ends = np.append(edges[1:], n)
    out = np.take(data, edges, axis=axis)
    for k in range(1, int((ends - edges).max())):
        np.maximum(
            out, np.take(data, np.minimum(edges + k, ends - 1), axis=axis), out=out
        )
    return out


def bins_start(center_hz: float, span_hz: float, n: int) -> float:
    """Lower edge (Hz) of the first of the ``n`` FFT bins of a line received
    over ``span_hz`` around ``center_hz``. In ``fftshift`` order bin ``k``
    is centered at ``center - span/2 + k * span/n`` (the center frequency
    is bin ``n // 2``), so the bins start half a bin below
    ``center - span/2``."""
    return center_hz - span_hz / 2 - (span_hz / n / 2 if n > 0 else 0.0)


def visible_bins(
    n: int, data_start: float, data_span: float, view_start: float, view_span: float
) -> Optional[Tuple[int, int]]:
    """``(first, stop)``: the bins of an ``n``-bin line covering
    ``data_start`` .. ``data_start + data_span`` that overlap the view
    ``view_start`` .. ``view_start + view_span`` (None: none do).
    ``data_start`` is the lower edge of the first bin (for an FFT line,
    :func:`bins_start`)."""
    if n <= 0 or data_span <= 0 or view_span <= 0:
        return None
    bin_hz = data_span / n
    first = int(math.floor((view_start - data_start) / bin_hz + 1e-9))
    stop = int(math.ceil((view_start + view_span - data_start) / bin_hz - 1e-9))
    first, stop = max(0, first), min(n, stop)
    if stop <= first:
        return None
    return first, stop


def add_zoom_menu(menu: "QMenu", view: PlotView, choose) -> "QMenu":
    """A "Zoom" submenu (Full span, x2 ... x16) on ``menu``; ``choose(zoom)``
    applies a choice. Shared by the spectrum's and the waterfall's menus."""
    zoom = menu.addMenu("&Zoom")
    zoom.setToolTipsVisible(True)
    group = QActionGroup(zoom)
    for level in ZOOM_LEVELS:
        if level == 1:
            act = zoom.addAction("&Full Span")
            act.setToolTip("Show the whole span (Ctrl+0)")
        else:
            act = zoom.addAction(f"×&{level}")
            act.setToolTip(
                f"Show 1/{level} of the span around the tuned frequency "
                "(Ctrl+= and Ctrl+- step through these)"
            )
        act.setCheckable(True)
        act.setChecked(level == view.zoom)
        group.addAction(act)
        act.triggered.connect(lambda _c=False, z=level: choose(z))
    return zoom


def auto_range_targets(noise_db: float, peak_db: float) -> Tuple[float, float]:
    """Floor and top (dBFS) of an Auto vertical range: the noise floor in
    the lower part of the plot (with room below it for the noise's
    wiggles) and the strongest signal below the top, on a 5 dB grid, at
    least ``_AUTO_RANGE_MIN_SPAN_DB`` apart."""
    grid = _AUTO_RANGE_GRID_DB
    lo, hi = _RANGE_LIMITS_DB
    floor = grid * math.floor((noise_db - _AUTO_RANGE_BELOW_NOISE_DB) / grid)
    top = grid * math.ceil((peak_db + _AUTO_RANGE_ABOVE_PEAK_DB) / grid)
    top = max(top, floor + _AUTO_RANGE_MIN_SPAN_DB)
    top = min(max(top, lo + _AUTO_RANGE_MIN_SPAN_DB), hi)
    floor = max(lo, min(floor, top - _AUTO_RANGE_MIN_SPAN_DB))
    return float(floor), float(top)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class _SpectrumCanvas(QWidget if HAS_PYQT6 else object):
    """The painted plot area below the spectrum header strip."""

    def __init__(self, owner: "SpectrumWidget"):
        super().__init__(owner)
        self._owner = owner
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumHeight(120)

    def paintEvent(self, event):  # noqa: N802 - Qt override
        self._owner._paint_canvas(self)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        self._owner._invalidate_background()
        super().resizeEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        owner = self._owner
        pos = event.position()
        if owner._axis_drag is not None:
            owner._drag_range_to(pos.y())
            event.accept()
            return
        owner._set_hover(pos)
        self.setCursor(owner._cursor_at(pos))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        self._owner._set_hover(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        owner = self._owner
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if owner._on_db_axis(pos):
                owner._start_range_drag(pos.y())
                event.accept()
                return
            index = owner._bookmark_at(pos)
            if index is not None:
                owner.bookmark_clicked.emit(int(index))
                event.accept()
                return
            exact = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            if owner._click_at(pos, exact=exact):
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if self._owner._axis_drag is not None:
            self._owner._finish_range_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt override
        if self._owner._on_db_axis(event.position()):
            self._owner._finish_range_drag()
            self._owner.set_auto_range(True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        owner = self._owner
        pos = event.position()
        if owner._on_db_axis(pos):
            owner._wheel_range(event)
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            owner._wheel_zoom(event, pos)
            event.accept()
            return
        steps = owner._wheel_tuner.steps(event)
        if steps is None:
            super().wheelEvent(event)
            return
        if steps:
            owner.tune_steps.emit(steps)
        event.accept()

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        self._owner._show_context_menu(QPointF(event.pos()), event.globalPos())
        event.accept()


class SpectrumWidget(QWidget if HAS_PYQT6 else object):
    """
    Spectrum analyzer display widget.

    Shows power spectrum with configurable averaging and peak hold. Hovering
    the plot shows the frequency and level under the cursor; clicking emits
    :attr:`frequency_clicked` with that frequency (Hz), moved onto the signal
    under the click and the tuning step's raster once
    :meth:`set_click_snap` gave a step (Alt+click: exactly where clicked).
    The mouse wheel emits :attr:`tune_steps`; Ctrl+wheel zooms around the
    cursor (see :meth:`set_view_model`). The dashed marker is the tuned
    frequency (:meth:`set_tuned_frequency`, the center until set).

    Each trace keeps the center and span it was received with: after a
    retune it is drawn where its signals really are (or not at all), never
    relabelled with the new frequencies.
    """

    if HAS_PYQT6:
        frequency_clicked = pyqtSignal(float)  # Hz
        #: Tuning steps the mouse wheel asked for (negative: down).
        tune_steps = pyqtSignal(int)
        #: A frequency (Hz) the user asked to bookmark from the context menu.
        bookmark_requested = pyqtSignal(float)
        #: A bookmark label was clicked: the bookmark's index.
        bookmark_clicked = pyqtSignal(int)
        #: A short message for the status bar: (text, tone).
        message = pyqtSignal(str, str)
        #: AVG, Peak hold, range, labels, FFT size or window changed (save).
        display_settings_changed = pyqtSignal()
        #: The FFT size or window was chosen (points, window key).
        resolution_changed = pyqtSignal(int, str)
        #: The user zoomed or panned (Ctrl+wheel, the menus).
        view_changed = pyqtSignal()

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._fft_size = 2048
        self._window_name = DEFAULT_WINDOW
        self._center_freq = 100e6
        self._sample_rate = 2.4e6
        self._db_range = (-100, 0)
        # Auto range: follows the noise floor and the strongest signal.
        self._auto_range = False
        self._range_noise: Optional[float] = None
        self._range_peak: Optional[float] = None

        # Spectrum data, and the (center, span) it was received with
        self._spectrum = np.zeros(self._fft_size)
        # The highest and the lowest level per bin since the last reset
        # (NaN: none yet; fmax and fmin skip NaN).
        self._peak_hold = np.full(self._fft_size, np.nan)
        self._min_hold = np.full(self._fft_size, np.nan)
        self._average = np.zeros(self._fft_size)
        self._avg_count = 0
        self._avg_alpha = 0.3
        self._has_data = False
        self._trace_axis: Tuple[float, float] = (self._center_freq, self._sample_rate)
        # With ``reset_on_gain_jump``, a jump of the noise floor (the
        # trace's median, dBFS) resets the peak hold and the average: the
        # gain changed. (The main window turns it on.)
        self.reset_on_gain_jump = False
        self._floor_db: Optional[float] = None
        # Why the data is no longer live ("Stopped - press Space"), or None.
        self._stale: Optional[str] = None

        # Display options
        self._show_peak = True
        self._show_min = False
        self._show_average = False
        self._grid_enabled = True

        # Labelled markers (set_markers): (frequency Hz, label). (T10)
        self._markers: List[Tuple[float, str]] = []
        # Right-click actions others add (add_context_action): (text,
        # callback taking the frequency clicked, tooltip). (T10)
        self._context_actions: List[Tuple[str, Callable[[float], None], str]] = []
        # Demodulated channel: (low, high) Hz offsets from the center.
        self._passband: Optional[Tuple[float, float]] = None
        # The tuned frequency (None: the center), and how clicks snap: the
        # tuning step (0: to the pixel resolution only) and the mode's key.
        self._tuned_freq: Optional[float] = None
        self._snap_step = 0.0
        self._snap_mode = ""
        # The last live frames, where a click also looks for its signal.
        self._recent_frames: Deque[np.ndarray] = deque(maxlen=SNAP_HOLD_FRAMES)
        self._wheel_tuner = WheelTuner()
        self._zoom_wheel = NotchCounter()
        self._range_wheel = NotchCounter()
        # What part of the span is shown (shared with the waterfall by the
        # main window; see set_view_model).
        self._view = PlotView(self._center_freq, self._sample_rate)
        self._view.subscribe(self._on_view_changed)
        # (Tuned frequency, LO, span) the view last followed: it follows
        # again only when one of them changes (see set_tuned_frequency).
        self._follow_key: Optional[Tuple[float, float, float]] = None
        # Bookmarks labelled along the top: (freq Hz, name, index), by
        # frequency (and their frequencies alone, to find those in view),
        # and each label's elided text and width (measured once).
        self._bookmarks: List[Tuple[float, str, int]] = []
        self._bookmark_freqs: List[float] = []
        self._label_cache: Dict[Tuple[str, float], Tuple[str, int]] = {}
        self._show_bookmarks = True
        self._bookmark_boxes: List[Tuple["QRectF", int, float, str]] = []
        # Dragging the dB axis: (start y, range at the start).
        self._axis_drag: Optional[Tuple[float, Tuple[float, float]]] = None

        # Paint caches (rebuilt on resize, theme, range or axis changes)
        self._font = axis_font()
        self._unit_font = QFont(self._font)
        self._unit_font.setBold(True)
        self._fm = QFontMetrics(self._font)
        self._bg_pixmap: Optional[QPixmap] = None
        self._bg_key: Optional[tuple] = None
        self._plot_rect = QRectF()
        self._pens: dict = {}
        self._hover: Optional[QPointF] = None

        self.setMinimumHeight(200)
        self._setup_ui()
        theme_notifier().theme_changed.connect(self._on_theme_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Header strip (title, hint, trace controls) above the plot canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        set_role(header, "header-strip")
        self._header = header
        controls = QHBoxLayout(header)
        controls.setContentsMargins(*HEADER_MARGINS)
        controls.setSpacing(8)

        title = QLabel("SPECTRUM")
        set_role(title, "caption")
        controls.addWidget(title)

        hint = QLabel("Click to tune · scroll to step · Ctrl+scroll to zoom")
        set_role(hint, "hint")
        hint.setToolTip(
            "Click a signal to tune to it (Alt+click tunes exactly where you "
            "click). The dashed line marks the tuned frequency and the shaded "
            "band the channel being demodulated. The mouse wheel and, with the "
            "plot focused, Left/Right tune by the STEP in the receiver bar "
            "(Shift: ten steps; Ctrl+Left/Right: 1 MHz).\n"
            "Ctrl+wheel zooms around the pointer (View > Zoom: Ctrl+= and "
            "Ctrl+-, Ctrl+0 for the full span). Drag or scroll the dB axis to "
            "move the range; double-click it for Auto. Right-click for more."
        )
        # Never let the optional hint raise the widget's minimum width:
        # fit_header_hint() hides it before it would be squeezed.
        hint.setMinimumWidth(1)
        controls.addWidget(hint)
        self._hint_label = hint

        controls.addStretch(1)

        self._peak_check = QCheckBox("Peak hold")
        self._peak_check.setChecked(self._show_peak)
        self._peak_check.setToolTip(
            get_short_tip("averaging_peak_hold")
            + " Cleared automatically when you retune or the gain changes."
        )
        self._peak_check.toggled.connect(self._on_peak_toggled)
        # Header controls don't take focus on click, so Space/arrow keys keep
        # acting on the plot (start/stop, tuning). Tab still reaches them.
        self._peak_check.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        controls.addWidget(self._peak_check)

        self._peak_reset_btn = QPushButton("Reset")
        set_role(self._peak_reset_btn, "compact")
        self._peak_reset_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._peak_reset_btn.clicked.connect(self._on_reset_peak_clicked)
        controls.addWidget(self._peak_reset_btn)
        self._update_reset_button()

        controls.addSpacing(6)

        avg_label = QLabel("AVG")
        set_role(avg_label, "caption")
        controls.addWidget(avg_label)
        self._avg_combo = QComboBox()
        for text, frames in _AVG_CHOICES:
            self._avg_combo.addItem(text, frames)
        self._avg_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._avg_combo.setToolTip(
            "Smooth the trace by averaging over about this many frames. "
            "The live trace stays visible, dimmed, behind the average."
        )
        avg_label.setToolTip(self._avg_combo.toolTip())
        avg_label.setBuddy(self._avg_combo)
        self._avg_combo.currentIndexChanged.connect(self._on_avg_changed)
        self._avg_combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        controls.addWidget(self._avg_combo)

        self._options_btn = QPushButton("⋯")
        set_role(self._options_btn, "compact")
        self._options_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._options_btn.setAccessibleName("Spectrum options")
        self._options_btn.setToolTip(
            "More: FFT size and resolution, window, vertical range, zoom, "
            "bookmark labels"
        )
        self._options_btn.clicked.connect(self._show_options_menu)
        controls.addWidget(self._options_btn)

        layout.addWidget(header)

        self._canvas = _SpectrumCanvas(self)
        self._canvas.setAccessibleName("Spectrum plot")
        self._canvas.setAccessibleDescription(
            "Power versus frequency. A shaded band marks the demodulated "
            "channel. Click to tune to a frequency; scroll to tune by the step; "
            "Ctrl+scroll to zoom; right-click for options."
        )
        layout.addWidget(self._canvas, 1)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        fit_header_hint(self, self._header, self._hint_label)

    def focusInEvent(self, event):  # noqa: N802 - Qt override
        """Show the focus frame: Space and Left/Right now act on this plot."""
        super().focusInEvent(event)
        self._canvas.update()

    def focusOutEvent(self, event):  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self._canvas.update()

    def contextMenuEvent(self, event):  # noqa: N802 - Qt override
        """The context-menu key while the plot has focus."""
        if not self._canvas.geometry().contains(event.pos()):
            super().contextMenuEvent(event)
            return
        pos = QPointF(self._canvas.mapFrom(self, event.pos()))
        self._show_context_menu(pos, event.globalPos())
        event.accept()

    def _update_reset_button(self) -> None:
        holding = self._show_peak or self._show_min
        self._peak_reset_btn.setEnabled(holding)
        if self._show_peak and self._show_min:
            tip = "Clear the peak-hold and min-hold traces"
        elif self._show_min:
            tip = "Clear the min-hold trace"
        elif self._show_peak:
            tip = "Clear the peak-hold trace"
        else:
            tip = "Turn on Peak hold (or Min hold, in the ⋯ menu) to use Reset"
        self._peak_reset_btn.setToolTip(tip)

    def _on_peak_toggled(self, checked: bool) -> None:
        changed = bool(checked) != self._show_peak
        self._show_peak = bool(checked)
        self._update_reset_button()
        self._canvas.update()
        if changed:
            self._refit_range()
            self.display_settings_changed.emit()

    def _on_reset_peak_clicked(self) -> None:
        self.reset_peak()
        self._refit_range()  # (Range Auto lets go of what was held)

    def _on_avg_changed(self, index: int):
        """Handle averaging mode change."""
        if index <= 0:
            self._show_average = False
        else:
            self._show_average = True
            # Alpha for exponential averaging
            n = _AVG_CHOICES[min(index, len(_AVG_CHOICES) - 1)][1]
            self._avg_alpha = 2.0 / (n + 1)

        self._avg_count = 0
        self._average = np.zeros(self._fft_size)
        self._canvas.update()
        self.display_settings_changed.emit()

    def _on_theme_changed(self, _name: str = "") -> None:
        try:
            self._label_cache = {}
            self._invalidate_background()
            self._canvas.update()
        except RuntimeError:  # pragma: no cover - widget already deleted
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_spectrum(self, power_db: np.ndarray):
        """
        Update spectrum with new data.

        Args:
            power_db: Power spectrum in dB, over the span set with
                :meth:`set_frequency_range`. An empty array is ignored (the
                previous frame stays on screen).
        """
        power_db = np.asarray(power_db, dtype=np.float64).ravel()
        if power_db.size == 0:
            return
        if len(power_db) != self._fft_size:
            # Resample if needed
            power_db = np.interp(
                np.linspace(0, 1, self._fft_size),
                np.linspace(0, 1, len(power_db)),
                power_db,
            )

        axis = (self._center_freq, self._sample_rate)
        if self._has_data and axis != self._trace_axis:
            self._recent_frames.clear()
        if self.reset_on_gain_jump:
            sample = power_db[:: max(1, len(power_db) // 2048)]
            sample = sample[np.isfinite(sample)]
            floor = float(np.median(sample)) if sample.size else None
            if (
                floor is not None
                and self._floor_db is not None
                and abs(floor - self._floor_db) >= _PEAK_RESET_JUMP_DB
            ):
                # Another gain (or the AGC moved): what was held is history.
                self.reset_peak()
                self.reset_average()
            self._floor_db = floor
        self._spectrum = power_db
        self._trace_axis = axis
        self._has_data = True
        self._stale = None
        self._recent_frames.append(power_db)

        # Update peak hold (fmax ignores NaN bins instead of latching them)
        self._peak_hold = np.fmax(self._peak_hold, power_db)
        # ...and the min hold (fmin: likewise; NaN until a bin has data)
        self._min_hold = np.fmin(self._min_hold, power_db)

        # Update average
        if self._show_average:
            if self._avg_count == 0:
                self._average = power_db.copy()
            else:
                self._average = (
                    self._avg_alpha * power_db + (1 - self._avg_alpha) * self._average
                )
            self._avg_count += 1

        if self._auto_range:
            self._observe_range(self._main_trace())
        self._canvas.update()

    def reset_peak(self):
        """Reset peak hold (and min hold: they start over together)."""
        self._peak_hold = np.full(self._fft_size, np.nan)
        self._min_hold = np.full(self._fft_size, np.nan)
        self._canvas.update()

    def _refit_range(self) -> None:
        """Range Auto: fit the traces drawn at once (the user reset the
        holds, or showed or hid one) instead of easing toward them."""
        if self._auto_range and self._has_data:
            self._observe_range(self._main_trace(), force=True)

    def reset_average(self):
        """Reset averaging."""
        self._average = np.zeros(self._fft_size)
        self._avg_count = 0

    def set_frequency_range(self, center_freq: float, sample_rate: float):
        """Set frequency range for display."""
        changed = (center_freq, sample_rate) != (self._center_freq, self._sample_rate)
        self._center_freq = center_freq
        self._sample_rate = sample_rate
        if changed:
            self._view.set_full(center_freq, sample_rate)
            self._on_axis_changed()

    def set_center_freq(self, center_freq: float):
        """Convenience setter used by the main window."""
        self.set_frequency_range(center_freq, self._sample_rate)

    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the displayed span (equal to the complex sample rate)."""
        self.set_frequency_range(self._center_freq, sample_rate)

    def set_fft_size(self, size: int) -> None:
        """Expect spectra of ``size`` bins (the FFT size of the frames the
        main window computes). Clears the peak hold and the average."""
        size = int(size)
        if size <= 0 or size == self._fft_size:
            return
        self._fft_size = size
        if self._has_data:
            self._spectrum = np.interp(
                np.linspace(0, 1, size),
                np.linspace(0, 1, len(self._spectrum)),
                self._spectrum,
            )
        else:
            self._spectrum = np.zeros(size)
        self._recent_frames.clear()
        self.reset_peak()
        self.reset_average()
        self._invalidate_background()
        self._canvas.update()

    def fft_size(self) -> int:
        return self._fft_size

    def set_window_name(self, name: str) -> None:
        """The FFT window in use (for the menus and the RBW shown). Clears
        the holds and the average, as a new FFT size does: another window
        reads the noise per bin up to 4 dB differently."""
        name = normalize_window(name)
        if name != self._window_name:
            self._window_name = name
            self.reset_peak()
            self.reset_average()
            self._invalidate_background()
            self._canvas.update()

    def window_name(self) -> str:
        return self._window_name

    def resolution_bandwidth(self) -> float:
        """The resolution bandwidth (Hz) of the spectrum shown."""
        return rbw_hz(self._sample_rate, self._fft_size, self._window_name)

    def set_view_model(self, view: PlotView) -> None:
        """Show the part of the span ``view`` says (the main window gives
        the spectrum and the waterfall the same one, so they zoom
        together)."""
        if view is self._view:
            return
        self._view = view
        self._follow_key = None
        view.subscribe(self._on_view_changed)
        view.set_full(self._center_freq, self._sample_rate)
        self._on_view_changed()

    def view_model(self) -> PlotView:
        return self._view

    def set_passband(self, low_hz: Optional[float], high_hz: Optional[float]) -> None:
        """Shade the demodulated channel, ``low_hz`` to ``high_hz`` around the
        tuned (center) frequency.

        Both are offsets in Hz from the center frequency, e.g.
        ``(-12500, 12500)`` for 25 kHz FM, ``(0, 2800)`` for USB or
        ``(-250, 250)`` for CW. ``None`` for either, a non-finite value or
        ``low_hz >= high_hz`` hides the shading.
        """
        band = normalize_passband(low_hz, high_hz)
        if band == self._passband:
            return
        self._passband = band
        self._invalidate_background()
        self._canvas.update()

    def passband(self) -> Optional[Tuple[float, float]]:
        """The shaded channel as (low, high) Hz offsets, or None when hidden."""
        return self._passband

    def set_tuned_frequency(self, freq_hz: Optional[float]) -> None:
        """Put the tuned-frequency marker at ``freq_hz`` (the VFO, anywhere
        in the span); None puts it at the center frequency. A zoomed view
        follows it when tuning (or a retune) takes it out of sight."""
        value: Optional[float] = None
        if freq_hz is not None:
            value = float(freq_hz)
            if not math.isfinite(value):
                value = None
        if value != self._tuned_freq:
            self._tuned_freq = value
            self._invalidate_background()  # the passband's direction
            self._canvas.update()
        # Follow only when the tuning (or the receiver's span) changed: the
        # window also sends the same frequency again on Stop, a mode or
        # filter change..., and a view zoomed elsewhere (Ctrl+wheel, Zoom In
        # Here) must stay where the user put it.
        tuned = self.tuned_frequency()
        key = (tuned, self._view.full_center, self._view.full_span)
        if key != self._follow_key:
            self._follow_key = key
            self._view.follow(tuned)

    def tuned_frequency(self) -> float:
        """The frequency the marker shows (Hz)."""
        return self._center_freq if self._tuned_freq is None else self._tuned_freq

    def set_click_snap(self, step_hz: float, mode_key: str = "") -> None:
        """Make clicks tune onto the signal near them and the ``step_hz``
        raster (see :func:`~sdr_module.gui.tuning.snap_click_frequency`);
        ``mode_key`` (e.g. ``"usb"``) decides whether a click moves onto a
        signal. A step of 0 snaps to the pixel resolution only."""
        self._snap_step = float(step_hz) if step_hz and step_hz > 0 else 0.0
        self._snap_mode = str(mode_key or "")
        self._canvas.update()

    def set_db_range(self, min_db: float, max_db: float):
        """Show ``min_db``..``max_db`` dBFS (a fixed range: Auto turns off)."""
        self._auto_range = False
        self._set_range(min_db, max_db)

    def db_range(self) -> Tuple[float, float]:
        """The vertical range shown: (floor, top) in dBFS."""
        return (float(self._db_range[0]), float(self._db_range[1]))

    def set_auto_range(self, enabled: bool) -> None:
        """Let the vertical range follow the noise floor and the strongest
        signal (like the waterfall's Auto levels)."""
        enabled = bool(enabled)
        changed = enabled != self._auto_range
        self._auto_range = enabled
        if enabled and self._has_data:
            self._range_noise = self._range_peak = None
            self._observe_range(self._main_trace(), force=True)
        if changed:
            self._canvas.update()
            self.display_settings_changed.emit()

    def auto_range(self) -> bool:
        return self._auto_range

    def set_range_span(self, span_db: float) -> None:
        """A fixed range ``span_db`` deep, from the current top down."""
        span = min(max(float(span_db), _RANGE_SPAN_LIMITS[0]), _RANGE_SPAN_LIMITS[1])
        top = float(self._db_range[1])
        self._user_range(top - span, top)

    def shift_range(self, delta_db: float) -> None:
        """Move the (fixed) range up (positive) or down by ``delta_db``."""
        floor, top = self._db_range
        lo, hi = _RANGE_LIMITS_DB
        delta = min(max(float(delta_db), lo - floor), hi - top)
        self._user_range(floor + delta, top + delta)

    def set_peak_hold(self, enabled: bool) -> None:
        self._peak_check.setChecked(bool(enabled))

    def peak_hold(self) -> bool:
        return self._show_peak

    def set_min_hold(self, enabled: bool) -> None:
        """Show the lowest level each bin had since the last reset (the
        noise floor under intermittent signals)."""
        enabled = bool(enabled)
        if enabled == self._show_min:
            return
        self._show_min = enabled
        self._update_reset_button()
        self._refit_range()
        self._canvas.update()
        self.display_settings_changed.emit()

    def min_hold(self) -> bool:
        return self._show_min

    def set_average_frames(self, frames: int) -> None:
        """Average over about ``frames`` frames (0 or a value not offered:
        the nearest choice; 0 is Off)."""
        values = [f for _t, f in _AVG_CHOICES]
        try:
            frames = int(frames)
        except (TypeError, ValueError, OverflowError):  # (inf, NaN, junk)
            frames = 0
        index = min(range(len(values)), key=lambda i: abs(values[i] - frames))
        self._avg_combo.setCurrentIndex(index)

    def average_frames(self) -> int:
        return int(self._avg_combo.currentData() or 0)

    def set_stale(self, text: Optional[str]) -> None:
        """Dim the plot and say ``text`` (e.g. ``"Stopped - press Space"``)
        until the next spectrum arrives; None shows it as live again."""
        text = str(text) if text else None
        if text != self._stale:
            self._stale = text
            self._canvas.update()

    def stale_text(self) -> Optional[str]:
        """What the dimmed plot says, or None while it is live (or empty)."""
        return self._stale if self._has_data else None

    def clear(self) -> None:
        """Forget the trace (back to the "No spectrum yet" placeholder)."""
        self._has_data = False
        self._stale = None
        self._spectrum = np.zeros(self._fft_size)
        self._recent_frames.clear()
        self.reset_average()
        self.reset_peak()

    def set_bookmarks(self, bookmarks: Sequence[Tuple[float, str, int]]) -> None:
        """Label these bookmarks (frequency Hz, name, index) along the top of
        the plot; a click on a label emits :attr:`bookmark_clicked`."""
        items: List[Tuple[float, str, int]] = []
        for entry in bookmarks or ():
            try:
                freq, name, index = float(entry[0]), str(entry[1]), int(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(freq):
                items.append((freq, name, index))
        items.sort(key=lambda item: item[0])
        if items != self._bookmarks:
            self._bookmarks = items
            self._bookmark_freqs = [item[0] for item in items]
            self._label_cache = {}
            self._canvas.update()

    def set_bookmark_labels_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible != self._show_bookmarks:
            self._show_bookmarks = visible
            self._bookmark_boxes = []
            self._canvas.update()
            self.display_settings_changed.emit()

    def bookmark_labels_visible(self) -> bool:
        return self._show_bookmarks

    def display_settings(self) -> Dict[str, Any]:
        """The user's display choices, for saving between sessions."""
        floor, top = self._db_range
        return {
            "avg": self.average_frames(),
            "peak_hold": self._show_peak,
            "min_hold": self._show_min,
            "range_auto": self._auto_range,
            "range_floor": float(floor),
            "range_top": float(top),
            "fft_size": self._fft_size,
            "window": self._window_name,
            "zoom": self._view.zoom,
            "bookmark_labels": self._show_bookmarks,
        }

    def apply_display_settings(self, settings: Dict[str, Any]) -> None:
        """Restore :meth:`display_settings` output (unknown keys ignored).
        The FFT size, window and zoom are the main window's to apply (they
        concern the waterfall too); this restores the rest."""
        if not isinstance(settings, dict):
            return
        blocked = self.blockSignals(True)
        try:
            if "avg" in settings:
                self.set_average_frames(settings.get("avg"))
            if "peak_hold" in settings:
                self.set_peak_hold(bool(settings.get("peak_hold")))
            if "min_hold" in settings:
                self.set_min_hold(bool(settings.get("min_hold")))
            try:
                floor = float(settings.get("range_floor", self._db_range[0]))
                top = float(settings.get("range_top", self._db_range[1]))
                if math.isfinite(floor) and math.isfinite(top):
                    self._set_range(floor, top)
            except (TypeError, ValueError, OverflowError):
                pass
            if "range_auto" in settings:
                self.set_auto_range(bool(settings.get("range_auto")))
            if "bookmark_labels" in settings:
                self.set_bookmark_labels_visible(bool(settings.get("bookmark_labels")))
            if "window" in settings:
                self.set_window_name(str(settings.get("window")))
        finally:
            self.blockSignals(blocked)

    def frequency_at(self, x: float) -> Optional[float]:
        """Frequency (Hz) under canvas x-coordinate ``x``, or None outside."""
        plot = self._current_plot_rect()
        if plot.width() <= 0 or x < plot.left() or x > plot.right():
            return None
        return self._view.frequency_at((x - plot.left()) / plot.width())

    def plot_rect(self) -> "QRectF":
        """The plot area in canvas coordinates (excludes the axis margins)."""
        return QRectF(self._current_plot_rect())

    # -- markers and right-click actions (T10) ---------------------------------

    def set_markers(self, markers: Sequence[Tuple[float, str]]) -> None:
        """Label frequencies on the spectrum: a small triangle on the trace
        at each ``(frequency_hz, label)`` with its label above (e.g. the
        Signal tab's signals in view). An empty list clears them."""
        clean = [
            (float(f), str(label))
            for f, label in markers
            if isinstance(f, (int, float)) and math.isfinite(float(f))
        ]
        if clean != self._markers:
            self._markers = clean
            self._canvas.update()

    def markers(self) -> List[Tuple[float, str]]:
        """The labelled markers shown (see :meth:`set_markers`)."""
        return list(self._markers)

    def add_context_action(
        self, text: str, callback: Callable[[float], None], tip: str = ""
    ) -> None:
        """Add an item to the plot's right-click menu: ``callback`` gets
        the frequency (Hz) right-clicked."""
        self._context_actions.append((str(text), callback, str(tip)))

    def add_context_actions(self, menu: "QMenu", freq: float) -> None:
        """Put the added right-click items (:meth:`add_context_action`) in
        ``menu`` for ``freq`` (for a menu built elsewhere)."""
        for text, callback, tip in self._context_actions:
            action = menu.addAction(text)
            if tip:
                action.setToolTip(tip)
                action.setStatusTip(tip)
            action.triggered.connect(lambda _c=False, cb=callback, f=freq: cb(float(f)))

    # ------------------------------------------------------------------
    # Internals: interaction
    # ------------------------------------------------------------------

    def _on_axis_changed(self) -> None:
        # A peak-hold trace from another frequency or span is misleading
        # (and so are the recent frames a click snaps on).
        self.reset_peak()
        self.reset_average()
        self._recent_frames.clear()
        self._floor_db = None
        self._invalidate_background()
        self._canvas.update()

    def _on_view_changed(self) -> None:
        self._invalidate_background()
        self._canvas.update()

    def _hz_per_px(self) -> float:
        plot = self._current_plot_rect()
        return self._view.span / plot.width() if plot.width() > 0 else 0.0

    def _snapped_frequency_at(self, x: float) -> Optional[float]:
        freq = self.frequency_at(x)
        if freq is None:
            return None
        step = snap_step_hz(self._hz_per_px())
        return round(freq / step) * step

    def _trace_is_current(self) -> bool:
        return self._has_data and self._trace_axis == (
            self._center_freq,
            self._sample_rate,
        )

    def landing_frequency(self, x: float, exact: bool = False) -> Optional[float]:
        """Where a click at canvas x ``x`` tunes (None outside the plot):
        onto the signal there and the step's raster, or with ``exact``
        (Alt+click) just to the pixel's resolution."""
        if exact or self._snap_step <= 0:
            return self._snapped_frequency_at(x)
        freq = self.frequency_at(x)
        if freq is None:
            return None
        current = self._trace_is_current()
        return snap_click_frequency(
            freq,
            self._main_trace() if current else None,
            self._center_freq - self._sample_rate / 2,
            self._sample_rate,
            self._hz_per_px(),
            self._snap_step,
            self._snap_mode,
            recent_db=self._recent_hold() if current else None,
        )

    def _recent_hold(self) -> Optional[np.ndarray]:
        """The strongest level per bin over the last frames (None until
        there are two of the current size)."""
        size = len(self._spectrum)
        frames = [f for f in self._recent_frames if len(f) == size]
        if len(frames) < 2:
            return None
        hold = np.array(frames[0], dtype=np.float64)
        for frame in frames[1:]:
            np.fmax(hold, frame, out=hold)
        return hold

    def _tuned_offset(self) -> float:
        """The tuned frequency's offset from the center (Hz)."""
        return 0.0 if self._tuned_freq is None else self._tuned_freq - self._center_freq

    def _freq_x(self, freq_hz: float, plot: "QRectF") -> Optional[float]:
        """Canvas x of ``freq_hz`` (None outside the plot)."""
        if self._view.span <= 0 or plot.width() <= 0:
            return None
        x = plot.left() + self._view.fraction_of(freq_hz) * plot.width()
        if x < plot.left() - 0.5 or x > plot.right() + 0.5:
            return None
        return x

    def _set_hover(self, pos: Optional["QPointF"]) -> None:
        self._hover = QPointF(pos) if pos is not None else None
        self._canvas.update()

    def _click_at(self, pos: "QPointF", exact: bool = False) -> bool:
        """Emit frequency_clicked for a click inside the plot area."""
        freq = self.landing_frequency(pos.x(), exact=exact)
        if freq is None:
            return False
        self.frequency_clicked.emit(float(freq))
        return True

    def _level_at(self, x: float, plot: "QRectF") -> Optional[float]:
        """The main trace's level under canvas ``x`` (the strongest bin under
        the pixel, as drawn), or None where the trace has no data."""
        if not self._has_data:
            return None
        data = self._main_trace()
        n = len(data)
        center, span = self._trace_axis
        freq = self.frequency_at(x)
        if freq is None or n == 0 or span <= 0:
            return None
        bin_hz = span / n
        # (In bins from the first bin's lower edge: bin k is centered at
        # center - span/2 + k * bin_hz.)
        pos = (freq - (center - span / 2)) / bin_hz + 0.5
        half = max(0.5, self._hz_per_px() / bin_hz / 2)
        lo = int(math.floor(pos - half))
        hi = int(math.floor(pos + half)) + 1
        if hi <= 0 or lo >= n:
            return None
        window = np.asarray(data[max(0, lo) : min(n, max(hi, lo + 1))], dtype=float)
        finite = window[np.isfinite(window)]
        return float(finite.max()) if finite.size else None

    # -- vertical range ---------------------------------------------------
    def _set_range(self, floor: float, top: float) -> None:
        floor, top = float(floor), float(top)
        if not (math.isfinite(floor) and math.isfinite(top)):
            return
        if top - floor < 1.0:
            top = floor + 1.0
        if (floor, top) == tuple(self._db_range):
            return
        self._db_range = (floor, top)
        self._invalidate_background()
        self._canvas.update()

    def _user_range(self, floor: float, top: float) -> None:
        """A range chosen by hand (menu, drag, wheel): Auto turns off."""
        before = (self._auto_range, tuple(self._db_range))
        self._auto_range = False
        self._set_range(floor, top)
        if (self._auto_range, tuple(self._db_range)) != before:
            self.display_settings_changed.emit()

    def _observe_range(self, trace: np.ndarray, force: bool = False) -> None:
        """Auto range: track the noise floor (median) and the strongest
        signal, and move the range when they leave its comfortable part."""
        values = np.asarray(trace, dtype=np.float64)
        finite = values[np.isfinite(values)]
        n = finite.size
        if n < 16:
            return
        # The strongest bin counts, however narrow: a carrier is two or
        # three bins of a 32k FFT. (Its estimate falls slowly, so a lone
        # spike doesn't make the range jump.)
        noise = float(np.partition(finite, n // 2)[n // 2])
        peak = float(finite.max())
        # The holds drawn count too: a burst caught by Peak hold stays below
        # the top (where its level can be read), Min hold above the floor.
        if self._show_peak:
            held = self._peak_hold[np.isfinite(self._peak_hold)]
            if held.size:
                peak = max(peak, float(held.max()))
        if self._show_min:
            low = self._min_hold[np.isfinite(self._min_hold)]
            if low.size >= 16:
                noise = min(
                    noise, float(np.partition(low, low.size // 2)[low.size // 2])
                )
        if self._range_noise is None or force:
            self._range_noise = noise
        else:
            self._range_noise += 0.2 * (noise - self._range_noise)
        if self._range_peak is None or peak > self._range_peak or force:
            self._range_peak = peak
        else:
            self._range_peak += 0.03 * (peak - self._range_peak)
        floor, top = auto_range_targets(self._range_noise, self._range_peak)
        cur_floor, cur_top = self._db_range
        grid = _AUTO_RANGE_GRID_DB
        if (
            force
            or abs(floor - cur_floor) > grid
            or abs(top - cur_top) > grid
            or self._range_noise < cur_floor + 5.0  # the noise is clipped
            or self._range_peak > cur_top  # a signal is clipped
        ):
            self._set_range(floor, top)

    def _on_db_axis(self, pos: "QPointF") -> bool:
        """Whether ``pos`` is over the dB labels left of the plot."""
        plot = self._current_plot_rect()
        return (
            pos.x() < plot.left() - 1 and plot.top() - 4 <= pos.y() <= plot.bottom() + 4
        )

    def _cursor_at(self, pos: "QPointF") -> "Qt.CursorShape":
        if self._on_db_axis(pos):
            return Qt.CursorShape.SizeVerCursor
        if self._bookmark_at(pos) is not None:
            return Qt.CursorShape.PointingHandCursor
        return Qt.CursorShape.CrossCursor

    def _start_range_drag(self, y: float) -> None:
        self._axis_drag = (float(y), self.db_range())

    def _drag_range_to(self, y: float) -> None:
        if self._axis_drag is None:
            return
        start_y, (floor, top) = self._axis_drag
        plot = self._current_plot_rect()
        if plot.height() <= 0:
            return
        # Grab and move: the level under the pointer stays under it.
        delta = (y - start_y) / plot.height() * (top - floor)
        delta = round(delta)
        lo, hi = _RANGE_LIMITS_DB
        delta = min(max(delta, lo - floor), hi - top)
        self._user_range(floor + delta, top + delta)

    def _finish_range_drag(self) -> None:
        self._axis_drag = None

    def _wheel_range(self, event) -> None:
        notches = self._range_wheel.notches(event)
        if not notches:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            floor, top = self._db_range
            self.set_range_span((top - floor) - 10.0 * notches)
        else:
            self.shift_range(-_RANGE_WHEEL_DB * notches)

    # -- zoom ----------------------------------------------------------------
    def _wheel_zoom(self, event, pos: "QPointF") -> None:
        notches = self._zoom_wheel.notches(event)
        if not notches:
            return
        plot = self._current_plot_rect()
        anchor = self.frequency_at(pos.x())
        frac = (pos.x() - plot.left()) / plot.width() if plot.width() > 0 else 0.5
        if self._view.zoom_steps(notches, anchor, frac):
            self.view_changed.emit()

    def zoom_to(self, freq_hz: float, steps: int = 1) -> None:
        """Zoom in ``steps`` factors of two, centered on ``freq_hz``."""
        if self._view.zoom_steps(steps, freq_hz, 0.5):
            self.view_changed.emit()

    def set_zoom(self, zoom: int) -> None:
        """Zoom to ``zoom`` (1 to 16) around the tuned frequency."""
        anchor = self.tuned_frequency()
        if not self._view.in_span(anchor):
            anchor = None
        if self._view.set_zoom(normalize_zoom(zoom), anchor, 0.5):
            self.view_changed.emit()

    # -- bookmark labels -----------------------------------------------------
    def _bookmark_at(self, pos: "QPointF") -> Optional[int]:
        if not self._show_bookmarks:
            return None
        for box, index, _freq, _name in self._bookmark_boxes:
            if box.contains(pos):
                return index
        return None

    def _bookmark_box_at(self, pos: "QPointF"):
        for entry in self._bookmark_boxes if self._show_bookmarks else ():
            if entry[0].contains(pos):
                return entry
        return None

    # -- menus -------------------------------------------------------------
    def _copy_frequency(self, freq_hz: float, decimals: int) -> None:
        text = tuned_mhz_text(freq_hz, decimals)
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.message.emit(f"Copied {text}", "info")

    def _display_menu(self, menu: "QMenu") -> None:
        """The display options (shared by the context and options menus)."""
        peak = menu.addAction("Peak &Hold")
        peak.setCheckable(True)
        peak.setChecked(self._show_peak)
        peak.toggled.connect(self.set_peak_hold)
        low = menu.addAction("Mi&n Hold")
        low.setCheckable(True)
        low.setChecked(self._show_min)
        low.setToolTip("The lowest level of each bin since the last reset")
        low.toggled.connect(self.set_min_hold)
        clear_peak = menu.addAction("R&eset Holds")
        clear_peak.setEnabled(self._show_peak or self._show_min)
        clear_peak.triggered.connect(lambda _c=False: self._on_reset_peak_clicked())

        floor, top = self._db_range
        ranges = menu.addMenu("&Range")
        ranges.setToolTipsVisible(True)
        group = QActionGroup(ranges)
        auto = ranges.addAction("&Auto (follow noise and signals)")
        auto.setCheckable(True)
        auto.setChecked(self._auto_range)
        auto.setToolTip(
            "The floor follows the noise floor and the top the strongest "
            "signal. Dragging or scrolling the dB axis sets a fixed range."
        )
        group.addAction(auto)
        auto.triggered.connect(lambda _c=False: self.set_auto_range(True))
        ranges.addSeparator()
        for span in RANGE_SPANS:
            act = ranges.addAction(f"{span} dB")
            act.setCheckable(True)
            act.setChecked(not self._auto_range and abs((top - floor) - span) < 0.5)
            act.setToolTip(f"A fixed range {span} dB deep, from {top:.0f} dBFS down")
            group.addAction(act)
            act.triggered.connect(lambda _c=False, s=span: self.set_range_span(s))
        ranges.addSeparator()
        full = ranges.addAction("&Top at 0 dBFS")
        full.setToolTip("Move the range so its top is full scale")
        full.triggered.connect(
            lambda _c=False: self._user_range(
                -(self._db_range[1] - self._db_range[0]), 0.0
            )
        )

        add_zoom_menu(menu, self._view, self.set_zoom)

        sizes = menu.addMenu("FFT &Size")
        group = QActionGroup(sizes)
        for size in FFT_SIZES:
            rbw = rbw_hz(self._sample_rate, size, self._window_name)
            act = sizes.addAction(f"{size} points · RBW {format_hz(rbw)}")
            act.setCheckable(True)
            act.setChecked(size == self._fft_size)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, s=size: self._choose_resolution(s))

        windows = menu.addMenu("&Window")
        windows.setToolTipsVisible(True)
        group = QActionGroup(windows)
        for name in WINDOWS:
            act = windows.addAction(WINDOW_LABELS[name])
            act.setCheckable(True)
            act.setChecked(name == self._window_name)
            act.setToolTip(WINDOW_HINTS[name])
            group.addAction(act)
            act.triggered.connect(
                lambda _c=False, n=name: self._choose_resolution(window=n)
            )

        labels = menu.addAction("Bookmark &Labels")
        labels.setCheckable(True)
        labels.setChecked(self._show_bookmarks)
        labels.toggled.connect(self.set_bookmark_labels_visible)

    def _choose_resolution(
        self, size: Optional[int] = None, window: Optional[str] = None
    ) -> None:
        size = self._fft_size if size is None else normalize_fft_size(size)
        window = self._window_name if window is None else normalize_window(window)
        if (size, window) == (self._fft_size, self._window_name):
            return
        self.set_fft_size(size)
        self.set_window_name(window)
        self.resolution_changed.emit(size, window)
        self.display_settings_changed.emit()

    def _show_options_menu(self) -> None:
        menu = QMenu(self)
        self._display_menu(menu)
        button = self._options_btn
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        menu.deleteLater()

    def _show_context_menu(self, pos: "QPointF", global_pos) -> None:
        """Right-click: tune, bookmark, copy or zoom to the frequency under
        the cursor, then the display options."""
        menu = QMenu(self)
        freq = self._snapped_frequency_at(pos.x())
        if freq is not None:
            decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
            text = tuned_mhz_text(freq, decimals)
            tune = menu.addAction(f"&Tune to {text}")
            tune.triggered.connect(
                lambda _c=False: self.frequency_clicked.emit(float(freq))
            )
            mark = menu.addAction(f"&Bookmark {text}")
            mark.triggered.connect(
                lambda _c=False: self.bookmark_requested.emit(float(freq))
            )
            copy = menu.addAction(f"&Copy Frequency ({text})")
            copy.setToolTip(f"Copy {text} to the clipboard")
            copy.triggered.connect(
                lambda _c=False: self._copy_frequency(freq, decimals)
            )
            zoom_in = menu.addAction("Zoom &In Here")
            zoom_in.setEnabled(self._view.zoom < ZOOM_LEVELS[-1])
            zoom_in.triggered.connect(lambda _c=False: self.zoom_to(freq, 1))
            self.add_context_actions(menu, float(freq))  # (T10)
            menu.addSeparator()
        self._display_menu(menu)
        menu.exec(global_pos)
        menu.deleteLater()

    # ------------------------------------------------------------------
    # Internals: painting
    # ------------------------------------------------------------------

    def _invalidate_background(self) -> None:
        self._bg_key = None

    def _margins(self) -> Tuple[int, int, int, int]:
        """(left, top, right, bottom) plot margins inside the canvas."""
        left, right = plot_side_margins(self._fm)
        fh = self._fm.height()
        # Room for the dBFS caption and legend above the top (0 dB) label,
        # and for the frequency labels below the bottom dB label, each with
        # a few pixels of air so the corner captions don't touch the ticks.
        top = int(fh * 1.5) + 7
        bottom = int(fh * 1.5) + 9
        return left, top, right, bottom

    def _current_plot_rect(self) -> "QRectF":
        left, top, right, bottom = self._margins()
        w = self._canvas.width() - left - right
        h = self._canvas.height() - top - bottom
        return QRectF(left, top, max(0, w), max(0, h))

    def _ensure_background(self, canvas: "QWidget", p) -> None:
        """(Re)build the cached background: fill, grid, axes, labels."""
        dpr = canvas.devicePixelRatioF()
        key = (
            canvas.width(),
            canvas.height(),
            dpr,
            p.name,
            tuple(self._db_range),
            self._center_freq,
            self._sample_rate,
            self._view.start,
            self._view.span,
            self._grid_enabled,
            self._passband,
            self._tuned_freq,
        )
        if key == self._bg_key and self._bg_pixmap is not None:
            return
        self._bg_key = key
        self._plot_rect = self._current_plot_rect()
        self._pens = self._build_pens(self._plot_rect, p)

        pix = QPixmap(
            max(1, int(canvas.width() * dpr)), max(1, int(canvas.height() * dpr))
        )
        pix.setDevicePixelRatio(dpr)
        pix.fill(p.qcolor("plot_bg"))
        painter = QPainter(pix)
        try:
            self._draw_axes(painter, self._plot_rect, p)
            # Under the traces and the tuned-frequency marker (drawn per frame).
            extent = passband_px(
                self._passband,
                self._plot_rect,
                self._view.span,
                self._tuned_offset(),
                view_offset_hz=self._center_freq - self._view.center,
            )
            if extent is not None:
                fill, edge = PASSBAND_ALPHAS[bool(p.is_dark)]
                draw_passband(painter, self._plot_rect, extent, p, fill, edge)
        finally:
            painter.end()
        self._bg_pixmap = pix

    @staticmethod
    def _build_pens(plot: "QRectF", p) -> dict:
        """Pens for the per-frame drawing (rebuilt with the background)."""
        grad = QLinearGradient(0, plot.top(), 0, plot.bottom())
        grad.setColorAt(0.0, p.qcolor("plot_trace", 80 if p.is_dark else 60))
        grad.setColorAt(1.0, p.qcolor("plot_trace", 6))
        marker = QPen(p.qcolor("plot_marker", 190), 1, Qt.PenStyle.CustomDashLine)
        marker.setDashPattern([4.0, 3.0])
        return {
            "fill": QPen(QBrush(grad), 1),
            "trace": QPen(p.qcolor("plot_trace"), 1),
            "live_dim": QPen(p.qcolor("plot_trace", 80), 1),
            "peak": QPen(p.qcolor("plot_peak", 210), 1),
            "min": QPen(p.qcolor("plot_axis", 220), 1),
            "marker": marker,
            "cursor": QPen(p.qcolor("plot_text", 120), 1),
        }

    def _draw_axes(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Grid, frame, tick labels and unit captions."""
        fm = self._fm
        fh = fm.height()
        painter.setFont(self._font)
        grid_pen = QPen(p.qcolor("plot_grid"), 1)
        axis_color = p.qcolor("plot_axis")
        if plot.width() <= 0 or plot.height() <= 0:
            return

        min_db, max_db = self._db_range
        db_span = max_db - min_db
        f_start = self._view.start
        f_end = self._view.end
        span = self._view.span

        # --- y (power) ticks
        y_ticks: List[float] = []
        if db_span > 0:
            min_px = max(26.0, fh * 2.0)
            db_step = nice_step(db_span * min_px / plot.height())
            y_ticks = nice_ticks(min_db, max_db, db_step)

        def db_to_y(db: float) -> float:
            return plot.bottom() - (db - min_db) / db_span * plot.height()

        # --- x (frequency) ticks
        x_ticks, decimals = freq_ticks(f_start, f_end, plot.width(), fm)

        def f_to_x(f: float) -> float:
            return plot.left() + (f - f_start) / span * plot.width()

        # Grid
        if self._grid_enabled:
            painter.setPen(grid_pen)
            for db in y_ticks:
                y = round(db_to_y(db)) + 0.5
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            for f in x_ticks:
                x = round(f_to_x(f)) + 0.5
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        # Frame, just outside the plot area (same as the waterfall's, so the
        # two borders line up)
        painter.setPen(grid_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(plot.adjusted(-0.5, -0.5, 0.5, 0.5))

        # y labels, right-aligned in the left margin, centered on grid lines
        painter.setPen(axis_color)
        label_right = plot.left() - 6
        right_align = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        for db in y_ticks:
            y = db_to_y(db)
            text = f"{db:.0f}" if abs(db - round(db)) < 1e-6 else f"{db:.1f}"
            painter.drawText(QRectF(0, y - fh / 2, label_right, fh), right_align, text)

        # Unit captions (bold): dBFS above the y labels, MHz left of the
        # x labels.
        label_y = plot.bottom() + fh / 2 + 4
        painter.setFont(self._unit_font)
        painter.drawText(QRectF(0, 2, label_right, fh), right_align, "dBFS")
        painter.drawText(QRectF(0, label_y, label_right, fh), right_align, "MHz")
        painter.setFont(self._font)

        # x labels centered on their ticks, kept inside the canvas
        tick_pen = QPen(p.qcolor("plot_grid"), 1)
        min_left = plot.left() + 3
        max_right = self._canvas.width() - 2
        last_right = -1e9
        center = int(Qt.AlignmentFlag.AlignCenter)
        for f in x_ticks:
            x = f_to_x(f)
            painter.setPen(tick_pen)
            xi = round(x) + 0.5
            painter.drawLine(QPointF(xi, plot.bottom()), QPointF(xi, plot.bottom() + 3))
            text = format_mhz(f, decimals)
            tw = fm.horizontalAdvance(text)
            centered = x - tw / 2
            left = min(max(centered, min_left), max_right - tw)
            if abs(left - centered) > tw * 0.3:
                continue  # tick too close to an edge to label it clearly
            if left < last_right + 8:
                continue  # would overlap the previous label
            painter.setPen(axis_color)
            painter.drawText(QRectF(left, label_y, tw, fh), center, text)
            last_right = left + tw

    def _trace_xy(
        self,
        data: np.ndarray,
        plot: "QRectF",
        cols: int,
        axis: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Canvas x/y arrays for ``data`` (dB per bin over ``axis``, its
        (center, span); the trace's own by default): the part inside the
        view, decimated to the pixel columns it covers (``cols`` across the
        whole plot)."""
        min_db, max_db = self._db_range
        span_db = max_db - min_db
        n = len(data)
        if span_db <= 0 or n == 0 or cols <= 0 or plot.width() <= 0:
            return None
        center, span = axis if axis is not None else self._trace_axis
        start = center - span / 2
        v_start, v_span = self._view.start, self._view.span
        bins = visible_bins(n, bins_start(center, span, n), span, v_start, v_span)
        if bins is None:
            return None
        # One bin beyond each edge, so the line runs to the plot's edges
        # (the painter clips it there).
        first, stop = max(0, bins[0] - 1), min(n, bins[1] + 1)
        bin_hz = span / n
        clean = np.nan_to_num(
            np.asarray(data[first:stop], dtype=np.float64),
            nan=min_db - 10.0,
            posinf=max_db + 10.0,
            neginf=min_db - 10.0,
        )
        m = stop - first
        px_per_bin = bin_hz / v_span * plot.width()
        out_cols = max(1, min(int(math.ceil(m * px_per_bin)), cols + 2))
        # (Bin k is centered at start + k * bin_hz; a column of bins
        # edges..ends-1 at the middle of their centers.)
        if m > out_cols:
            edges = (np.arange(out_cols) * m // out_cols).astype(np.intp)
            values = np.maximum.reduceat(clean, edges)
            ends = np.append(edges[1:], m)
            centers = first + (edges + ends - 1) / 2.0
        else:
            values = clean
            centers = first + np.arange(m, dtype=np.float64)
        values = np.clip(values, min_db - 2.0, max_db + 2.0)
        xs = plot.left() + (start + centers * bin_hz - v_start) / v_span * plot.width()
        ys = plot.bottom() - (values - min_db) / span_db * plot.height()
        return xs, ys

    @staticmethod
    def _polyline(xy: Tuple[np.ndarray, np.ndarray]) -> "QPolygonF":
        xs, ys = xy
        return QPolygonF(
            [QPointF(x, y) for x, y in zip(xs.tolist(), ys.tolist(), strict=True)]
        )

    def _paint_canvas(self, canvas: "QWidget") -> None:
        p = get_palette()
        self._ensure_background(canvas, p)
        painter = QPainter(canvas)
        try:
            painter.drawPixmap(0, 0, self._bg_pixmap)
            plot = self._plot_rect
            if plot.width() <= 2 or plot.height() <= 2:
                return
            painter.setFont(self._font)
            self._draw_legend(painter, plot, p)

            painter.save()
            painter.setClipRect(plot)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._has_data:
                self._draw_traces(painter, plot, p)
                if self._markers:
                    self._draw_markers(painter, plot, p)
            self._draw_center_marker(painter, plot, p)
            painter.restore()
            if self._has_data and self._stale:
                draw_stale(painter, plot, self._stale, canvas.font(), p)
                painter.setFont(self._font)
            self._draw_bookmarks(painter, plot, p)
            if self.hasFocus():
                draw_focus_frame(painter, plot, p)

            if not self._has_data:
                draw_placeholder(
                    painter,
                    plot,
                    "No spectrum yet",
                    "Press Start (or Space) to begin receiving",
                    canvas.font(),
                    p,
                )
            if self._hover is not None:
                # Also shown before any data: clicking tunes either way.
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setFont(self._font)
                self._draw_hover(painter, plot, p)
        finally:
            painter.end()

    def _main_trace(self) -> np.ndarray:
        if self._show_average and self._avg_count > 0:
            return self._average
        return self._spectrum

    def _draw_traces(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Draw peak hold, the (dimmed) live trace and the main trace.

        Only exact 1 px pens are used: wider or fractional pens drop QPainter
        onto its slow path stroker (~30 ms for one trace instead of ~0.3 ms).
        """
        cols = max(1, int(plot.width()))
        averaging = self._show_average and self._avg_count > 0
        pens = self._pens

        main = self._trace_xy(self._main_trace(), plot, cols)

        # Soft fill under the main trace, drawn as one vertical line per
        # pixel column (a filled 900-point polygon costs ~10 ms per frame).
        if main is not None and len(main[0]) > 1:
            xs, ys = main
            lo = max(plot.left(), float(xs[0]))
            hi = min(plot.right(), float(xs[-1]))
            col_x = np.arange(math.floor(lo - plot.left()), math.ceil(hi - plot.left()))
            col_x = plot.left() + col_x + 0.5
            col_x = col_x[(col_x >= lo) & (col_x <= hi)]
            if col_x.size:
                col_y = np.interp(col_x, xs, ys)
                bottom = plot.bottom()
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                painter.setPen(pens["fill"])
                painter.drawLines(
                    [
                        QLineF(x, y, x, bottom)
                        for x, y in zip(col_x.tolist(), col_y.tolist(), strict=True)
                    ]
                )
                painter.restore()

        # Peak hold (once there is data; a bin without any is drawn below
        # the floor)
        if self._show_peak and np.isfinite(self._peak_hold).any():
            peak = self._trace_xy(self._peak_hold, plot, cols)
            if peak is not None:
                painter.setPen(pens["peak"])
                painter.drawPolyline(self._polyline(peak))

        # Min hold (once every bin has data)
        if self._show_min and np.isfinite(self._min_hold).all():
            low = self._trace_xy(self._min_hold, plot, cols)
            if low is not None:
                painter.setPen(pens["min"])
                painter.drawPolyline(self._polyline(low))

        # With averaging on, the live trace stays visible but dimmed.
        if averaging:
            live = self._trace_xy(self._spectrum, plot, cols)
            if live is not None:
                painter.setPen(pens["live_dim"])
                painter.drawPolyline(self._polyline(live))

        if main is not None:
            painter.setPen(pens["trace"])
            painter.drawPolyline(self._polyline(main))

    def _draw_markers(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """The labelled markers (set_markers): a triangle on the trace's
        highest point under the marker, the label above it. (T10)"""
        data = self._main_trace()
        n = len(data)
        min_db, max_db = self._db_range
        span = max_db - min_db
        if n == 0 or span <= 0:
            return
        fm = self._fm
        color = p.qcolor("accent")
        for freq, label in self._markers:
            x = self._freq_x(freq, plot)
            if x is None:
                continue
            # The trace covers the whole band whatever the zoom (its bin k
            # is centered at start + k * bin).
            start = self._center_freq - self._sample_rate / 2
            k = int(round((freq - start) / max(self._sample_rate, 1.0) * n))
            lo, hi = max(0, k - 2), min(n, k + 3)
            level = float(np.nanmax(data[lo:hi])) if hi > lo else min_db
            level = min(max(level, min_db), max_db)
            y = plot.bottom() - (level - min_db) / span * plot.height()
            tip = max(plot.top() + fm.height() + 6, y - 3)
            tri = QPolygonF(
                [
                    QPointF(x - 4, tip - 7),
                    QPointF(x + 4, tip - 7),
                    QPointF(x, tip),
                ]
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(tri)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if label:
                tw = fm.horizontalAdvance(label)
                painter.setPen(color)
                painter.drawText(
                    QRectF(x - tw / 2 - 2, tip - 8 - fm.height(), tw + 4, fm.height()),
                    int(Qt.AlignmentFlag.AlignCenter),
                    label,
                )

    def _draw_center_marker(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Dashed line + small flag at the tuned frequency (the VFO)."""
        tuned = self._freq_x(self.tuned_frequency(), plot)
        if tuned is None:
            return  # tuned outside the span shown
        x = round(tuned) + 0.5
        painter.setPen(self._pens["marker"])
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        tri = QPolygonF(
            [
                QPointF(x - 5, plot.top()),
                QPointF(x + 5, plot.top()),
                QPointF(x, plot.top() + 6),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(p.qcolor("plot_marker"))
        painter.drawPolygon(tri)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _info_text(self) -> str:
        """``"RBW 1.76 kHz · ×4"``: the resolution and the zoom, for the band
        above the plot."""
        text = f"RBW {format_hz(self.resolution_bandwidth())}"
        if self._view.is_zoomed():
            text += f" · ×{self._view.zoom}"
        return text

    def _trace_in_view(self) -> bool:
        """Whether any of the trace (received at its own axis) is in view:
        not after retuning away while stopped."""
        center, span = self._trace_axis
        return (
            visible_bins(
                len(self._spectrum),
                bins_start(center, span, len(self._spectrum)),
                span,
                self._view.start,
                self._view.span,
            )
            is not None
        )

    def _draw_legend(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Tiny legend in the band above the plot (right-aligned), and the
        resolution bandwidth and zoom at its left. A stopped plot retuned
        away from its last trace (nothing drawn) has no trace legend."""
        fm = self._fm
        y_mid = 2 + fm.height() / 2
        x = plot.right()
        if self._has_data and not (self._stale and not self._trace_in_view()):
            items = []
            averaging = self._show_average and self._avg_count > 0
            if self._stale:
                # Not live any more: the last trace received.
                if averaging:
                    frames = max(1, round(2.0 / max(self._avg_alpha, 1e-6) - 1))
                    items.append((f"Avg {frames}", p.qcolor("plot_trace")))
                else:
                    items.append(("Last", p.qcolor("plot_trace")))
            elif averaging:
                frames = max(1, round(2.0 / max(self._avg_alpha, 1e-6) - 1))
                items.append((f"Avg {frames}", p.qcolor("plot_trace")))
                items.append(("Live", p.qcolor("plot_trace", 110)))
            else:
                items.append(("Live", p.qcolor("plot_trace")))
            if self._show_peak:
                items.append(("Peak", p.qcolor("plot_peak")))
            if self._show_min:
                items.append(("Min", p.qcolor("plot_axis")))
            swatch, gap = 12, 12
            widths = [swatch + 4 + fm.horizontalAdvance(t) for t, _ in items]
            total = sum(widths) + gap * (len(items) - 1)
            start = plot.right() - total
            if start >= plot.left() + 60:
                x = start
                for (text, color), w in zip(items, widths, strict=True):
                    painter.setPen(QPen(color, 2))
                    painter.drawLine(QPointF(x, y_mid), QPointF(x + swatch, y_mid))
                    painter.setPen(p.qcolor("plot_axis"))
                    painter.drawText(QPointF(x + swatch + 4, 2 + fm.ascent()), text)
                    x += w + gap
                x = start
        info = self._info_text()
        if plot.left() + fm.horizontalAdvance(info) + 16 <= x:
            painter.setPen(p.qcolor("plot_axis"))
            painter.drawText(QPointF(plot.left() + 4, 2 + fm.ascent()), info)

    def _bookmark_label(self, freq: float, name: str) -> Tuple[str, int]:
        """A bookmark label's text (its name, or its frequency, elided) and
        width: measured once, not at every repaint."""
        key = (name, 0.0 if name else freq)
        found = self._label_cache.get(key)
        if found is None:
            text = self._fm.elidedText(
                name or tuned_mhz_text(freq),
                Qt.TextElideMode.ElideMiddle,
                BOOKMARK_LABEL_MAX_PX,
            )
            found = self._label_cache[key] = (text, self._fm.horizontalAdvance(text))
        return found

    def _draw_bookmarks(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Small labels at the bookmarked frequencies in view, along the top
        of the plot. Labels that would overlap the one before are left out
        (the nearer one to the tuned frequency wins), long names are
        elided."""
        self._bookmark_boxes = []
        if not self._show_bookmarks or not self._bookmarks or plot.width() <= 0:
            return
        fm = self._fm
        height = fm.height() + 2
        top = plot.top() + 9  # below the tuned-frequency flag
        pad = 4
        tuned = self.tuned_frequency()
        # Only the bookmarks in view (the list is in frequency order): a
        # long imported list costs nothing where it isn't shown.
        half_px = 0.5 * self._view.span / plot.width()
        first = bisect_left(self._bookmark_freqs, self._view.start - half_px)
        stop = bisect_right(self._bookmark_freqs, self._view.end + half_px)
        candidates = []
        for freq, name, index in self._bookmarks[first:stop]:
            x = self._freq_x(freq, plot)
            if x is None:
                continue
            text, text_width = self._bookmark_label(freq, name)
            width = text_width + 2 * pad
            left = min(max(plot.left() + 1, x - width / 2), plot.right() - 1 - width)
            candidates.append(
                (abs(freq - tuned), left, width, x, text, index, freq, name)
            )
        # Nearest the tuned frequency first, then skip whatever comes within
        # 3 px of a label placed already. The labels share one line, so the
        # placed ones are kept in order (lefts, rights) and each check looks
        # at the one placed just left of the candidate's right edge.
        lefts: List[float] = []
        rights: List[float] = []
        placed = []
        for entry in sorted(candidates, key=lambda c: c[0]):
            left, width = entry[1], entry[2]
            k = bisect_left(lefts, left + width + 3)
            if k > 0 and rights[k - 1] > left - 3:
                continue
            lefts.insert(k, left)
            rights.insert(k, left + width)
            placed.append(entry)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setFont(self._font)
        for _dist, left, width, x, text, index, freq, name in placed:
            box = QRectF(left, top, width, height)
            xi = round(x) + 0.5
            painter.setPen(QPen(p.qcolor("accent", 170), 1))
            painter.drawLine(QPointF(xi, box.bottom()), QPointF(xi, box.bottom() + 4))
            # (Translucent: a station under its label stays visible.)
            painter.fillRect(box, p.qcolor("plot_bg", 150))
            painter.setPen(QPen(p.qcolor("accent", 150), 1))
            painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.setPen(p.qcolor("plot_text"))
            painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)
            self._bookmark_boxes.append((box, index, freq, name))
        painter.restore()

    def _draw_hover(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Cursor line, a dot on the trace and a readout of the frequency
        and level under the cursor and where a click tunes (marked when it
        snaps away from the cursor)."""
        pos = self._hover
        if pos is None or not plot.contains(pos):
            return
        label = self._bookmark_box_at(pos)
        if label is not None:
            _box, _index, freq, name = label
            lines = [name or "Bookmark", tuned_mhz_text(freq), "Click to tune"]
            draw_readout(painter, lines, pos.x(), plot, self._fm, p)
            return
        exact = bool(
            QGuiApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier
        )
        freq = self.landing_frequency(pos.x(), exact=exact)
        if freq is None:
            return
        x = pos.x()
        painter.setPen(self._pens["cursor"])
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        landing_x = self._freq_x(freq, plot)
        if landing_x is not None and abs(landing_x - x) >= 2:
            draw_landing_mark(painter, plot, landing_x, p)

        decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
        # The frequency and the level under the cursor belong together; a
        # click that snaps to a signal or the step says where it tunes.
        here = self._snapped_frequency_at(x)
        here_text = tuned_mhz_text(freq if here is None else here, decimals)
        lines = [here_text]
        level = self._level_at(x, plot)
        if level is not None:
            lines.append(f"{level:.1f} dBFS")
            min_db, max_db = self._db_range
            if max_db > min_db and min_db <= level <= max_db:
                y = plot.bottom() - (level - min_db) / (max_db - min_db) * plot.height()
                painter.setPen(QPen(p.qcolor("plot_bg"), 1.5))
                painter.setBrush(p.qcolor("plot_text"))
                painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        landing_text = tuned_mhz_text(freq, decimals)
        lines.append(
            "Click to tune here"
            if landing_text == here_text
            else f"Click to tune to {landing_text} · Alt: exact"
        )
        draw_readout(painter, lines, x, plot, self._fm, p)
