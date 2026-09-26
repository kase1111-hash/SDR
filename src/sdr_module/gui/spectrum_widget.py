"""
Spectrum analyzer widget.

Provides real-time spectrum visualization with:
- FFT-based power spectrum display (dBFS)
- Peak hold and exponential averaging
- Frequency axis with "nice" tick steps and a unit caption
- Hover readout (frequency and level), tuned-frequency marker, click-to-tune

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
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from ..utils.tooltips import get_short_tip
from .themes import get_palette, mono_font, set_role, theme_notifier

if TYPE_CHECKING:
    from PyQt6.QtGui import QPainter

try:
    from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import (
        QBrush,
        QFont,
        QFontMetrics,
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

#: Floor used for "no peak yet" in the peak-hold trace.
_PEAK_FLOOR_DB = -120.0

# Averaging combo entries: label -> number of frames (0 = off).
_AVG_CHOICES = (("Off", 0), ("2", 2), ("4", 4), ("8", 8), ("16", 16), ("32", 32))


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


def draw_readout(
    painter: "QPainter",
    lines: List[str],
    anchor_x: float,
    plot: "QRectF",
    fm: "QFontMetrics",
    palette,
    dim_last: bool = True,
) -> None:
    """Draw a small boxed readout next to ``anchor_x`` at the top of ``plot``.

    The box flips to the left of the cursor near the right edge so it never
    leaves the plot.
    """
    pad_x, pad_y = 6, 3
    text_w = max(fm.horizontalAdvance(line) for line in lines)
    line_h = fm.height()
    box_w = text_w + 2 * pad_x
    box_h = line_h * len(lines) + 2 * pad_y
    x = anchor_x + 10
    if x + box_w > plot.right() - 2:
        x = anchor_x - 10 - box_w
    x = max(plot.left() + 2, x)
    y = plot.top() + 6
    box = QRectF(x, y, box_w, box_h)
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
    passband: Optional[Tuple[float, float]], plot: "QRectF", span_hz: float
) -> Optional[Tuple[float, float]]:
    """Canvas x extent ``(left, right)`` of a passband drawn on ``plot``.

    ``passband`` holds offsets (Hz) from the tuned frequency, which sits at the
    plot's horizontal center; ``plot`` shows ``span_hz``. The extent is snapped
    to whole pixels and at least :data:`PASSBAND_MIN_PX` wide (a CW or SSB
    channel on a 2.4 MHz span is narrower than one pixel), growing away from
    the tuned frequency for a one-sided (SSB) band. It is not clipped to the
    plot. None when there is no passband or nothing to map it onto.
    """
    if passband is None or span_hz <= 0 or plot.width() <= 0:
        return None
    low, high = passband
    scale = plot.width() / span_hz
    center = plot.center().x()
    x0, x1 = center + low * scale, center + high * scale
    if x1 - x0 < PASSBAND_MIN_PX:
        if low >= 0:  # upper sideband: grow to the right of the marker
            x1 = x0 + PASSBAND_MIN_PX
        elif high <= 0:  # lower sideband: grow to the left
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
    if data.ndim == 1:
        return np.maximum.reduceat(data, edges)
    # 2-D: reduceat along an inner axis is slow; each column spans only a
    # few bins, so a handful of strided gathers is 3-4x faster.
    ends = np.append(edges[1:], n)
    out = np.take(data, edges, axis=axis)
    for k in range(1, int((ends - edges).max())):
        np.maximum(
            out, np.take(data, np.minimum(edges + k, ends - 1), axis=axis), out=out
        )
    return out


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
        self._owner._set_hover(event.position())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        self._owner._set_hover(None)
        super().leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            if self._owner._click_at(event.position()):
                event.accept()
                return
        super().mousePressEvent(event)


class SpectrumWidget(QWidget if HAS_PYQT6 else object):
    """
    Spectrum analyzer display widget.

    Shows power spectrum with configurable averaging and peak hold. Hovering
    the plot shows the frequency and level under the cursor; clicking emits
    :attr:`frequency_clicked` with that frequency (Hz).
    """

    if HAS_PYQT6:
        frequency_clicked = pyqtSignal(float)  # Hz

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
        self._peak_hold = np.full(self._fft_size, _PEAK_FLOOR_DB)
        self._average = np.zeros(self._fft_size)
        self._avg_count = 0
        self._avg_alpha = 0.3
        self._has_data = False

        # Display options
        self._show_peak = True
        self._show_average = False
        self._grid_enabled = True

        # Markers
        self._markers: List[Tuple[float, float]] = []  # (freq, power)
        # Demodulated channel: (low, high) Hz offsets from the center.
        self._passband: Optional[Tuple[float, float]] = None

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

        hint = QLabel("Click to tune")
        set_role(hint, "hint")
        hint.setToolTip(
            "Click the plot to tune to that frequency. The dashed line marks "
            "the tuned frequency and the shaded band the channel being "
            "demodulated. With the plot focused, Left/Right step 10 kHz "
            "(Shift: 100 kHz, Ctrl: 1 MHz)."
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
            + " Cleared automatically when you retune."
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

        layout.addWidget(header)

        self._canvas = _SpectrumCanvas(self)
        self._canvas.setAccessibleName("Spectrum plot")
        self._canvas.setAccessibleDescription(
            "Power versus frequency. A shaded band marks the demodulated "
            "channel. Click to tune to a frequency."
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

    def _update_reset_button(self) -> None:
        self._peak_reset_btn.setEnabled(self._show_peak)
        self._peak_reset_btn.setToolTip(
            "Clear the peak-hold trace"
            if self._show_peak
            else "Turn on Peak hold to use Reset"
        )

    def _on_peak_toggled(self, checked: bool) -> None:
        self._show_peak = bool(checked)
        self._update_reset_button()
        self._canvas.update()

    def _on_reset_peak_clicked(self) -> None:
        self.reset_peak()

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

    def _on_theme_changed(self, _name: str = "") -> None:
        try:
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
            power_db: Power spectrum in dB. An empty array is ignored (the
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

        self._spectrum = power_db
        self._has_data = True

        # Update peak hold (fmax ignores NaN bins instead of latching them)
        self._peak_hold = np.fmax(self._peak_hold, power_db)

        # Update average
        if self._show_average:
            if self._avg_count == 0:
                self._average = power_db.copy()
            else:
                self._average = (
                    self._avg_alpha * power_db + (1 - self._avg_alpha) * self._average
                )
            self._avg_count += 1

        self._canvas.update()

    def reset_peak(self):
        """Reset peak hold."""
        self._peak_hold = np.full(self._fft_size, _PEAK_FLOOR_DB)
        self._canvas.update()

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
            self._on_axis_changed()

    def set_center_freq(self, center_freq: float):
        """Convenience setter used by the main window."""
        changed = center_freq != self._center_freq
        self._center_freq = center_freq
        if changed:
            self._on_axis_changed()

    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the displayed span (equal to the complex sample rate)."""
        self.set_frequency_range(self._center_freq, sample_rate)

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

    def set_db_range(self, min_db: float, max_db: float):
        """Set dB range for display."""
        self._db_range = (min_db, max_db)
        self._invalidate_background()
        self._canvas.update()

    def frequency_at(self, x: float) -> Optional[float]:
        """Frequency (Hz) under canvas x-coordinate ``x``, or None outside."""
        plot = self._current_plot_rect()
        if plot.width() <= 0 or x < plot.left() or x > plot.right():
            return None
        frac = (x - plot.left()) / plot.width()
        return self._center_freq - self._sample_rate / 2 + frac * self._sample_rate

    def plot_rect(self) -> "QRectF":
        """The plot area in canvas coordinates (excludes the axis margins)."""
        return QRectF(self._current_plot_rect())

    # ------------------------------------------------------------------
    # Internals: interaction
    # ------------------------------------------------------------------

    def _on_axis_changed(self) -> None:
        # A peak-hold trace from another frequency or span is misleading.
        self.reset_peak()
        self.reset_average()
        self._invalidate_background()
        self._canvas.update()

    def _hz_per_px(self) -> float:
        plot = self._current_plot_rect()
        return self._sample_rate / plot.width() if plot.width() > 0 else 0.0

    def _snapped_frequency_at(self, x: float) -> Optional[float]:
        freq = self.frequency_at(x)
        if freq is None:
            return None
        step = snap_step_hz(self._hz_per_px())
        return round(freq / step) * step

    def _set_hover(self, pos: Optional["QPointF"]) -> None:
        self._hover = QPointF(pos) if pos is not None else None
        self._canvas.update()

    def _click_at(self, pos: "QPointF") -> bool:
        """Emit frequency_clicked for a click inside the plot area."""
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return False
        self.frequency_clicked.emit(float(freq))
        return True

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
            self._db_range,
            self._center_freq,
            self._sample_rate,
            self._grid_enabled,
            self._passband,
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
            extent = passband_px(self._passband, self._plot_rect, self._sample_rate)
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
        f_start = self._center_freq - self._sample_rate / 2
        f_end = self._center_freq + self._sample_rate / 2

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
            return plot.left() + (f - f_start) / self._sample_rate * plot.width()

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
        self, data: np.ndarray, plot: "QRectF", cols: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Canvas x/y arrays for ``data`` (dB), decimated to the plot width."""
        min_db, max_db = self._db_range
        span = max_db - min_db
        if span <= 0 or len(data) == 0 or cols <= 0:
            return None
        clean = np.nan_to_num(
            np.asarray(data, dtype=np.float64),
            nan=min_db - 10.0,
            posinf=max_db + 10.0,
            neginf=min_db - 10.0,
        )
        values = columns_max(clean, cols)
        n = len(values)
        if n == 0:
            return None
        values = np.clip(values, min_db - 2.0, max_db + 2.0)
        if n == cols:
            xs = plot.left() + (np.arange(n) + 0.5) * (plot.width() / n)
        else:
            xs = plot.left() + np.arange(n) * (plot.width() / n)
        ys = plot.bottom() - (values - min_db) / span * plot.height()
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
            self._draw_center_marker(painter, plot, p)
            painter.restore()
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
        if main is not None:
            xs, ys = main
            col_x = plot.left() + np.arange(cols) + 0.5
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

        # Peak hold (only bins that have seen data)
        if self._show_peak and np.any(self._peak_hold > _PEAK_FLOOR_DB):
            peak = self._trace_xy(self._peak_hold, plot, cols)
            if peak is not None:
                painter.setPen(pens["peak"])
                painter.drawPolyline(self._polyline(peak))

        # With averaging on, the live trace stays visible but dimmed.
        if averaging:
            live = self._trace_xy(self._spectrum, plot, cols)
            if live is not None:
                painter.setPen(pens["live_dim"])
                painter.drawPolyline(self._polyline(live))

        if main is not None:
            painter.setPen(pens["trace"])
            painter.drawPolyline(self._polyline(main))

    def _draw_center_marker(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Dashed line + small flag at the tuned (center) frequency."""
        x = round(plot.center().x()) + 0.5
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

    def _draw_legend(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Tiny legend in the band above the plot (right-aligned)."""
        if not self._has_data:
            return
        items = []
        averaging = self._show_average and self._avg_count > 0
        if averaging:
            frames = max(1, round(2.0 / max(self._avg_alpha, 1e-6) - 1))
            items.append((f"Avg {frames}", p.qcolor("plot_trace")))
            items.append(("Live", p.qcolor("plot_trace", 110)))
        else:
            items.append(("Live", p.qcolor("plot_trace")))
        if self._show_peak:
            items.append(("Peak", p.qcolor("plot_peak")))
        fm = self._fm
        swatch, gap = 12, 12
        widths = [swatch + 4 + fm.horizontalAdvance(t) for t, _ in items]
        total = sum(widths) + gap * (len(items) - 1)
        x = plot.right() - total
        if x < plot.left() + 60:
            return  # too narrow to be useful
        y_mid = 2 + fm.height() / 2
        for (text, color), w in zip(items, widths, strict=True):
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(x, y_mid), QPointF(x + swatch, y_mid))
            painter.setPen(p.qcolor("plot_axis"))
            painter.drawText(QPointF(x + swatch + 4, 2 + fm.ascent()), text)
            x += w + gap

    def _draw_hover(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Cursor line, a dot on the trace and a frequency/level readout."""
        pos = self._hover
        if pos is None or not plot.contains(pos):
            return
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return
        x = pos.x()
        painter.setPen(self._pens["cursor"])
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        data = self._main_trace()
        lines = [
            f"{format_mhz(freq, readout_decimals(snap_step_hz(self._hz_per_px())))} MHz"
        ]
        if self._has_data and len(data):
            frac = (x - plot.left()) / plot.width()
            n = len(data)
            # Strongest bin under this pixel, matching what the trace shows.
            lo = int(frac * n - n / plot.width() / 2)
            hi = int(frac * n + n / plot.width() / 2) + 1
            lo, hi = max(0, lo), min(n, max(hi, lo + 1))
            window = np.asarray(data[lo:hi], dtype=np.float64)
            finite = window[np.isfinite(window)]
            if finite.size:
                level = float(finite.max())
                lines.append(f"{level:.1f} dBFS")
                min_db, max_db = self._db_range
                if max_db > min_db and min_db <= level <= max_db:
                    y = (
                        plot.bottom()
                        - (level - min_db) / (max_db - min_db) * plot.height()
                    )
                    painter.setPen(QPen(p.qcolor("plot_bg"), 1.5))
                    painter.setBrush(p.qcolor("plot_text"))
                    painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
        lines.append("Click to tune")
        draw_readout(painter, lines, x, plot, self._fm, p)
