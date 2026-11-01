"""
The Signal tab: identify the tuned signal, list the signals in view, look
at the tuned channel on a scope, and log the bursts in the span.

A tool panel (key ``signal``, Ctrl+9; Tools > Identify Signal, Ctrl+I,
starts an identification). The window's :class:`~.signal_tools.SignalTools`
feeds it and acts on what it asks for; the panel itself only shows things
and says what the user wants (signals), so it can be tested on its own.

Pages:

* **Identify**: what the signal at the tuned frequency is (type, likely
  mode and width, SNR, a confidence in words and per cent) with
  **Apply Mode** to listen to it that way; on/off keyed data also gets
  its pulses taken apart (rtl_433 style).
* **Signals**: the signals in view, about twice a second (frequency,
  level, SNR, width), double-click to tune, **Bookmark**, the strongest
  and the next one up or down; optional labelled markers on the spectrum.
  After File > Find Signals in Recording, the recording's signals with when
  they were heard; double-click plays the recording there.
* **Scope**: the tuned channel in time (magnitude, I/Q, phase,
  instantaneous frequency) or as a constellation with a BPSK / QPSK /
  8PSK / 16QAM overlay and its EVM; **Freeze** holds it.
* **Bursts**: transmissions that come and go in the span: time, frequency,
  length, level, what they are and their bits (hex), the pulse analysis of
  the selected one, CSV export.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QFontMetrics, QGuiApplication, QPainter, QPen
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

from ..ui.constellation import ConstellationDisplay, ModulationOverlay

if HAS_PYQT6:
    from .decoder_panel import ViewPlaceholder
    from .scanner_dialog import _NumericItem
    from .themes import get_palette, set_role, set_tone, theme_notifier

logger = logging.getLogger(__name__)

#: Page keys, in tab order.
PAGES = ("identify", "signals", "scope", "bursts")
_PAGE_TITLES = {
    "identify": "Identify",
    "signals": "Signals",
    "scope": "Scope",
    "bursts": "Bursts",
}

#: Scope views (item text, key).
SCOPE_VIEWS: Tuple[Tuple[str, str], ...] = (
    ("Magnitude", "magnitude"),
    ("I and Q", "iq"),
    ("Phase", "phase"),
    ("Frequency", "frequency"),
    ("Constellation", "constellation"),
)
#: Constellation overlays (item text, overlay).
OVERLAYS: Tuple[Tuple[str, Any], ...] = (
    ("No overlay", ModulationOverlay.NONE),
    ("BPSK", ModulationOverlay.BPSK),
    ("QPSK", ModulationOverlay.QPSK),
    ("8PSK", ModulationOverlay.PSK8),
    ("16QAM", ModulationOverlay.QAM16),
)
#: Time spans of the scope (ms).
SCOPE_SPANS_MS = (1.0, 5.0, 20.0, 100.0)
#: The Bursts list's empty state while it watches.
BURSTS_EMPTY = (
    "No bursts yet\nTransmissions that start and stop in the span are listed "
    "here, e.g. at 433.92 MHz."
)


def format_mhz(freq_hz: float) -> str:
    """``"433.920 MHz"`` (6 decimals when needed)."""
    mhz = float(freq_hz) / 1e6
    text = f"{mhz:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    frac = (frac + "000")[: max(3, len(frac))]
    return f"{whole}.{frac} MHz"


def format_width(hz: float) -> str:
    """``"12.5 kHz"``, ``"200 kHz"``, ``"1.2 MHz"``, ``"500 Hz"``."""
    hz = float(hz)
    if hz >= 1e6:
        return f"{hz / 1e6:.2f} MHz"
    if hz >= 100e3:
        return f"{hz / 1e3:.0f} kHz"
    if hz >= 1e3:
        return f"{round(hz / 1e3, 1):g} kHz"
    return f"{hz:.0f} Hz"


def format_signal_mhz(freq_hz: float, width_hz: float, unit: bool = True) -> str:
    """A signal's frequency: to the kHz for a wide one, to 100 Hz for a
    narrow one (CW, a carrier); ``unit=False`` leaves out " MHz" (a column
    headed MHz)."""
    decimals = 4 if 0 < float(width_hz) < 5e3 else 3
    text = f"{float(freq_hz) / 1e6:.{decimals}f}"
    return f"{text} MHz" if unit else text


def format_ago(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.5:
        return "now"
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    return f"{seconds / 60:.0f} min ago"


def format_clock_s(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    return f"{int(minutes)}:{sec:04.1f}"


def tune_frequency(signal: Any) -> float:
    """Where to tune for a listed signal: its centre, to 10 Hz for a narrow
    one (CW, a carrier) and to 100 Hz otherwise (not to the hertz the
    measurement happens to give)."""
    width = float(getattr(signal, "width_hz", 0.0))
    step = 10.0 if 0 < width < 5e3 else 100.0
    return round(float(signal.frequency_hz) / step) * step


def listening_text(mode: str, deviation: str, filter_hz: float) -> str:
    """``"FM · 5 kHz deviation · 12.5 kHz filter"``."""
    if not mode:
        return ""
    parts = [mode]
    if deviation:
        parts.append(f"{deviation} deviation")
    if filter_hz > 0:
        parts.append(f"{format_width(filter_hz)} filter")
    return " · ".join(parts)


if HAS_PYQT6:

    # ----------------------------------------------------------------------- #
    # The scope
    # ----------------------------------------------------------------------- #

    class ScopeView(QWidget):
        """The tuned channel in time, or as a constellation, painted with
        the theme's plot colours. :meth:`set_samples` hands it the newest
        channel samples; nothing is drawn while it is hidden (the window
        doesn't feed it then) or frozen."""

        def __init__(self, parent: Any = None):
            super().__init__(parent)
            self.setMinimumHeight(90)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self.setAccessibleName("Scope")
            self.view = "magnitude"
            self.span_ms = 5.0
            self.frozen = False
            self._samples = np.zeros(0, dtype=np.complex64)
            self._rate = 0.0
            self._constellation = ConstellationDisplay(max_points=2048)
            self.evm_percent: Optional[float] = None
            self._font = QFont(self.font())
            self._font.setPointSizeF(max(7.0, self._font.pointSizeF() * 0.85))
            # A bound method: disconnected when the widget is deleted (a
            # lambda would outlive it on the process-wide notifier).
            theme_notifier().theme_changed.connect(self._on_theme_changed)

        def _on_theme_changed(self, _name: str = "") -> None:
            try:
                self.update()
            except RuntimeError:  # pragma: no cover - already deleted
                pass

        # -- data ---------------------------------------------------------------

        def set_view(self, view: str) -> None:
            self.view = view
            self.update()

        def set_overlay(self, overlay: Any) -> None:
            self._constellation.overlay = overlay
            self.update()

        @property
        def overlay(self) -> Any:
            return self._constellation.overlay

        def set_span_ms(self, span_ms: float) -> None:
            self.span_ms = float(span_ms)
            self.update()

        def set_frozen(self, frozen: bool) -> None:
            self.frozen = bool(frozen)
            self.update()

        def clear(self) -> None:
            self._samples = np.zeros(0, dtype=np.complex64)
            self.evm_percent = None
            self.update()

        def wanted(self) -> int:
            """How many samples the current view shows."""
            if self._rate <= 0:
                return 4096
            if self.view == "constellation":
                return 2048
            return max(16, int(self.span_ms * 1e-3 * self._rate))

        def set_samples(self, samples: np.ndarray, rate: float) -> None:
            """The newest channel samples (complex, at ``rate``)."""
            if self.frozen:
                return
            x = np.asarray(samples)
            if rate != self._rate:
                self._rate = float(rate)
                self._samples = np.zeros(0, dtype=np.complex64)
            want = self.wanted()
            if len(x) >= want:
                self._samples = np.asarray(x[-want:], dtype=np.complex64)
            else:
                joined = np.concatenate((self._samples, x.astype(np.complex64)))
                self._samples = joined[-want:]
            self.update()

        def samples(self) -> np.ndarray:
            return self._samples

        # -- painting -------------------------------------------------------------

        def _plot_rect(self) -> QRectF:
            fm = QFontMetrics(self._font)
            left = fm.horizontalAdvance("-0.00") + 8
            return QRectF(
                left, 6, self.width() - left - 8, self.height() - fm.height() - 14
            )

        def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            p = get_palette()
            painter = QPainter(self)
            try:
                painter.fillRect(self.rect(), p.qcolor("plot_bg"))
                painter.setFont(self._font)
                plot = self._plot_rect()
                if plot.width() < 20 or plot.height() < 20:
                    return
                if not len(self._samples) or self._rate <= 0:
                    painter.setPen(p.qcolor("plot_axis"))
                    painter.drawText(
                        self.rect(),
                        int(Qt.AlignmentFlag.AlignCenter),
                        "No samples yet\nPress Start (Space): the tuned channel "
                        "shows here",
                    )
                    return
                if self.view == "constellation":
                    self._paint_constellation(painter, plot, p)
                else:
                    self._paint_time(painter, plot, p)
                if self.frozen:
                    painter.setPen(p.qcolor("plot_marker"))
                    painter.drawText(
                        plot.adjusted(0, 2, -4, 0),
                        int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                        "FROZEN",
                    )
            finally:
                painter.end()

        def _traces(self) -> Tuple[List[np.ndarray], float, float, str]:
            """The curves of the time view, their y range and unit."""
            x = self._samples.astype(np.complex128)
            if self.view == "iq":
                peak = float(np.max(np.abs(x))) or 1.0
                return [x.real, x.imag], -1.1 * peak, 1.1 * peak, ""
            if self.view == "phase":
                return [np.angle(x)], -math.pi, math.pi, "rad"
            if self.view == "frequency":
                f = np.angle(x[1:] * np.conj(x[:-1])) * self._rate / (2 * math.pi)
                if len(f) > 3:
                    f = np.convolve(f, np.ones(3) / 3.0, mode="same")
                top = float(np.percentile(np.abs(f), 99)) if len(f) else 1.0
                top = max(top * 1.2, 100.0)
                return [f / 1e3], -top / 1e3, top / 1e3, "kHz"
            mag = np.abs(x)
            top = float(np.max(mag)) or 1.0
            return [mag], 0.0, 1.1 * top, ""

        def _paint_time(self, painter: Any, plot: QRectF, p: Any) -> None:
            curves, lo, hi, unit = self._traces()
            if hi <= lo:
                hi = lo + 1.0
            grid = QPen(p.qcolor("plot_grid"), 1)
            painter.setPen(grid)
            painter.drawRect(plot)
            for frac in (0.25, 0.5, 0.75):
                y = plot.top() + frac * plot.height()
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                x = plot.left() + frac * plot.width()
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(p.qcolor("plot_axis"))
            fm = QFontMetrics(self._font)
            for value, y in ((hi, plot.top()), (lo, plot.bottom())):
                text = _short_number(value)
                painter.drawText(
                    QRectF(0, y - fm.height() / 2, plot.left() - 4, fm.height()),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    text,
                )
            n = max(len(c) for c in curves)
            span_ms = n / self._rate * 1e3
            painter.drawText(
                QRectF(plot.left(), plot.bottom() + 2, plot.width(), fm.height()),
                int(Qt.AlignmentFlag.AlignRight),
                f"{span_ms:.3g} ms{('  ·  ' + unit) if unit else ''}",
            )
            colors = ("plot_trace", "plot_peak")
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            cols = max(2, int(plot.width()))
            # (one or two curves: the trace colour, then the peak hold's)
            for curve, color in zip(curves, colors[: len(curves)], strict=True):
                xs, ys = _decimate(curve, cols)
                px = plot.left() + xs * plot.width()
                py = plot.bottom() - (ys - lo) / (hi - lo) * plot.height()
                points = [
                    QPointF(a, b) for a, b in zip(px.tolist(), py.tolist(), strict=True)
                ]
                painter.setPen(QPen(p.qcolor(color), 1))
                painter.drawPolyline(points)

        def _paint_constellation(self, painter: Any, plot: QRectF, p: Any) -> None:
            side = min(plot.width(), plot.height())
            box = QRectF(
                plot.center().x() - side / 2, plot.center().y() - side / 2, side, side
            )
            x = _derotate(self._samples, self._constellation.overlay)
            result = self._constellation.process(x)
            self._constellation.clear()
            ref = 1.6
            pts = result.i_data, result.q_data
            self.evm_percent = (
                result.stats.evm_percent
                if self._constellation.overlay != ModulationOverlay.NONE
                else None
            )
            grid = QPen(p.qcolor("plot_grid"), 1)
            painter.setPen(grid)
            painter.drawRect(box)
            c = box.center()
            painter.drawLine(QPointF(box.left(), c.y()), QPointF(box.right(), c.y()))
            painter.drawLine(QPointF(c.x(), box.top()), QPointF(c.x(), box.bottom()))
            scale = side / (2 * ref)

            def to_px(i: float, q: float) -> QPointF:
                return QPointF(c.x() + i * scale, c.y() - q * scale)

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(QPen(p.qcolor("plot_trace", 170), 2))
            i_data, q_data = pts
            keep = (np.abs(i_data) < ref) & (np.abs(q_data) < ref)
            painter.drawPoints(
                [
                    to_px(float(i), float(q))
                    for i, q in zip(i_data[keep], q_data[keep], strict=True)
                ]
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(p.qcolor("plot_marker"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for point in self._constellation.get_overlay_points():
                painter.drawEllipse(to_px(point.i, point.q), 5.0, 5.0)
            painter.setPen(p.qcolor("plot_axis"))
            fm = QFontMetrics(self._font)
            text = "I/Q"
            if self.evm_percent is not None:
                text += f"   EVM {self.evm_percent:.1f} %"
            painter.drawText(
                QRectF(box.left() + 4, box.top() + 2, box.width() - 8, fm.height()),
                int(Qt.AlignmentFlag.AlignLeft),
                text,
            )

    def _pause_fit(table: Any, rows: bool = False) -> List[int]:
        """Stop ``table``'s columns (and rows) resizing to their contents
        while it is filled: with that on, every item set made the headers
        measure every row again, so filling n rows cost n squared (1.3 s
        for 80). :func:`_resume_fit` has them measure once, after."""
        header = table.horizontalHeader()
        fit = QHeaderView.ResizeMode.ResizeToContents
        paused = [
            i for i in range(header.count()) if header.sectionResizeMode(i) == fit
        ]
        for i in paused:
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        if rows:
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        return paused

    def _resume_fit(table: Any, paused: List[int], rows: bool = False) -> None:
        header = table.horizontalHeader()
        for i in paused:
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        if rows:
            table.verticalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )

    class _ElidedLabel(QLabel):
        """A one-line label that shortens its text with "…" in the middle
        when it doesn't fit (:meth:`text` is still the whole text; the
        tooltip should say it all)."""

        def __init__(
            self,
            text: str = "",
            parent: Any = None,
            mode: Any = Qt.TextElideMode.ElideMiddle,
        ):
            super().__init__(parent)
            self._full = ""
            self._mode = mode
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.setText(text)

        def setText(self, text: str) -> None:  # noqa: N802 - Qt override
            self._full = str(text)
            self._elide()

        def text(self) -> str:
            return self._full

        def displayed_text(self) -> str:
            """What fits, as shown."""
            return super().text()

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            super().resizeEvent(event)
            self._elide()

        def _elide(self) -> None:
            shown = self.fontMetrics().elidedText(
                self._full, self._mode, max(0, self.width())
            )
            if super().text() != shown:
                super().setText(shown)

    def _short_number(value: float) -> str:
        a = abs(value)
        if a >= 100:
            return f"{value:.0f}"
        if a >= 1:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def _decimate(curve: np.ndarray, cols: int) -> Tuple[np.ndarray, np.ndarray]:
        """x (0-1) and y of a curve with at most ``2 * cols`` points: the
        minimum and maximum of each column, so peaks stay."""
        n = len(curve)
        if n <= 2 * cols:
            return np.linspace(0.0, 1.0, max(n, 1)), np.asarray(curve, dtype=float)
        per = n // cols
        m = per * cols
        block = np.asarray(curve[:m], dtype=float).reshape(cols, per)
        lo, hi = block.min(axis=1), block.max(axis=1)
        ys = np.empty(2 * cols)
        ys[0::2], ys[1::2] = lo, hi
        xs = np.repeat(np.linspace(0.0, 1.0, cols), 2)
        return xs, ys

    def _derotate(samples: np.ndarray, overlay: Any) -> np.ndarray:
        """Take out the carrier's frequency and phase with the M-th power
        of the overlay's order (so a PSK constellation stands still)."""
        order = {
            ModulationOverlay.BPSK: 2,
            ModulationOverlay.QPSK: 4,
            ModulationOverlay.PSK8: 8,
            ModulationOverlay.QAM16: 4,
        }.get(overlay, 0)
        x = np.asarray(samples, dtype=np.complex128)
        if not order or len(x) < 16:
            return x
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) or 1.0
        y = (x / rms) ** order
        step = np.angle(np.sum(y[1:] * np.conj(y[:-1]))) / order
        x = x * np.exp(-1j * step * np.arange(len(x)))
        y = (x / rms) ** order
        phase = np.angle(np.mean(y)) / order
        if overlay == ModulationOverlay.QPSK or overlay == ModulationOverlay.QAM16:
            phase -= math.pi / 4 if overlay == ModulationOverlay.QPSK else 0.0
        return x * np.exp(-1j * phase)

    # ----------------------------------------------------------------------- #
    # The panel
    # ----------------------------------------------------------------------- #

    class SignalPanel(QWidget):
        """The Signal tab (see the module docstring)."""

        identify_requested = pyqtSignal()
        apply_requested = pyqtSignal()
        tune_requested = pyqtSignal(float)
        bookmark_requested = pyqtSignal(float)
        markers_changed = pyqtSignal(bool)
        min_snr_changed = pyqtSignal(float)
        step_requested = pyqtSignal(str)  # "strongest", "up", "down"
        watch_changed = pyqtSignal(bool)
        play_requested = pyqtSignal(float, float)  # frequency, seconds
        live_requested = pyqtSignal()
        page_changed = pyqtSignal(str)
        scope_changed = pyqtSignal()
        #: A short note for the status bar (text, tone), e.g. "Copied ...".
        message = pyqtSignal(str, str)

        def __init__(self, parent: Any = None):
            super().__init__(parent)
            self._ident: Any = None
            self._signals: List[Any] = []
            self._recording: Optional[Tuple[str, List[Any], float]] = None
            self._bursts: List[Any] = []
            self._t0 = time.monotonic()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)
            self.tabs.setUsesScrollButtons(True)
            layout.addWidget(self.tabs)
            self._build_identify()
            self._build_signals()
            self._build_scope()
            self._build_bursts()
            self.tabs.currentChanged.connect(self._on_tab_changed)

        def _on_tab_changed(self, _index: int) -> None:
            try:
                self.page_changed.emit(self.current_page())
            except RuntimeError:  # pragma: no cover - being deleted
                pass

        # -- pages ----------------------------------------------------------------

        def current_page(self) -> str:
            index = self.tabs.currentIndex()
            return PAGES[index] if 0 <= index < len(PAGES) else ""

        def show_page(self, page: str) -> None:
            if page in PAGES:
                self.tabs.setCurrentIndex(PAGES.index(page))

        def page_visible(self, page: str) -> bool:
            """Whether ``page`` is on screen (the tab shown and in front)."""
            return self.isVisible() and self.current_page() == page

        def _page(self) -> Tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)
            return page, layout

        @staticmethod
        def _table(
            columns: Sequence[str], tips: Sequence[str], name: str
        ) -> QTableWidget:
            table = QTableWidget(0, len(columns))
            table.setHorizontalHeaderLabels(list(columns))
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
            header.setMinimumSectionSize(30)
            for col, tip in enumerate(tips):
                item = table.horizontalHeaderItem(col)
                if item is not None:
                    item.setToolTip(tip)
            table.verticalHeader().setVisible(False)
            table.setShowGrid(False)
            table.setWordWrap(False)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            table.setTabKeyNavigation(False)
            table.setAccessibleName(name)
            table.setMinimumHeight(64)
            return table

        # -- Identify ---------------------------------------------------------------

        def _build_identify(self) -> None:
            page, layout = self._page()
            row = QHBoxLayout()
            self.identify_btn = QPushButton("Identify")
            set_role(self.identify_btn, "primary")
            self.identify_btn.setToolTip(
                "Listen to the tuned frequency for about half a second (longer "
                "for a signal that comes and goes) and say what it is (Ctrl+I)"
            )
            self.identify_btn.clicked.connect(
                lambda _c=False: self.identify_requested.emit()
            )
            row.addWidget(self.identify_btn)
            self.apply_btn = QPushButton("Apply Mode")
            self.apply_btn.setToolTip(
                "Listen to it the suggested way: Mode, FM deviation and "
                "Bandwidth, tuned to the signal's centre"
            )
            self.apply_btn.setEnabled(False)
            self.apply_btn.clicked.connect(lambda _c=False: self.apply_requested.emit())
            row.addWidget(self.apply_btn)
            # (Next to Apply Mode: in view however long the card is.)
            self.copy_pulses_btn = QPushButton("Copy Pulse Analysis")
            set_role(self.copy_pulses_btn, "compact")
            self.copy_pulses_btn.setToolTip(
                "Copy the pulse analysis (laid out like rtl_433 -A, with a flex "
                "decoder line) to the clipboard"
            )
            self.copy_pulses_btn.clicked.connect(lambda _c=False: self._copy_pulses())
            row.addWidget(self.copy_pulses_btn)
            row.addStretch(1)
            layout.addLayout(row)
            self.identify_status = QLabel(
                "Tune to a signal (click it on the spectrum), then press "
                "Identify (Ctrl+I)."
            )
            set_role(self.identify_status, "hint")
            self.identify_status.setWordWrap(True)
            layout.addWidget(self.identify_status)

            card = QFrame()
            set_role(card, "card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(4)
            self.kind_label = QLabel("—")
            font = QFont(self.kind_label.font())
            font.setPointSizeF(font.pointSizeF() * 1.35)
            font.setBold(True)
            self.kind_label.setFont(font)
            self.kind_label.setWordWrap(True)
            self.kind_label.setAccessibleName("Signal type")
            card_layout.addWidget(self.kind_label)
            conf_row = QHBoxLayout()
            self.confidence_bar = QProgressBar()
            self.confidence_bar.setRange(0, 100)
            self.confidence_bar.setTextVisible(False)
            self.confidence_bar.setFixedHeight(8)
            self.confidence_bar.setAccessibleName("Confidence")
            conf_row.addWidget(self.confidence_bar, 1)
            self.confidence_label = QLabel("")
            set_role(self.confidence_label, "muted")
            conf_row.addWidget(self.confidence_label)
            card_layout.addLayout(conf_row)
            self.listen_label = QLabel("")
            set_role(self.listen_label, "value")
            self.listen_label.setWordWrap(True)
            card_layout.addWidget(self.listen_label)
            self.details_label = QLabel("")
            self.details_label.setWordWrap(True)
            self.details_label.setTextFormat(Qt.TextFormat.PlainText)
            card_layout.addWidget(self.details_label)
            self.pulses_label = QLabel("")
            self.pulses_label.setWordWrap(True)
            self.pulses_label.setTextFormat(Qt.TextFormat.PlainText)
            self.pulses_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            card_layout.addWidget(self.pulses_label)
            layout.addWidget(card)
            layout.addStretch(1)
            self._card = card
            self._pulse_text = ""
            self.show_identification(None)
            # The page scrolls on its own: a long card (a pulse analysis)
            # must not make every page of the tab taller than the column.
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(page)
            self.identify_area = area
            index = self.tabs.addTab(area, _PAGE_TITLES["identify"])
            self.tabs.setTabToolTip(index, "What is the signal at the tuned frequency?")

        def set_identify_status(self, text: str, tone: Optional[str] = None) -> None:
            self.identify_status.setText(text)
            set_tone(self.identify_status, tone)

        def set_identify_busy(self, busy: bool) -> None:
            self.identify_btn.setText("Listening…" if busy else "Identify")
            self.identify_btn.setEnabled(not busy)

        def identification(self) -> Any:
            return self._ident

        def show_identification(self, ident: Any) -> None:
            """Show an :class:`~sdr_module.dsp.signal_id.Identification`
            (None clears the card)."""
            self._ident = ident
            has = ident is not None
            self._card.setVisible(has)
            if not has:
                self.apply_btn.setEnabled(False)
                self._pulse_text = ""
                self.copy_pulses_btn.setVisible(False)
                return
            from ..dsp.signal_id import confidence_words

            self.kind_label.setText(ident.label)
            percent = int(round(100 * ident.confidence))
            self.confidence_bar.setValue(percent)
            words = confidence_words(ident.confidence)
            self.confidence_label.setText(f"{words.capitalize()} · {percent} %")
            tone = None
            if ident.confidence < 0.3:
                tone = "warning"
            set_tone(self.confidence_label, tone)
            listen = listening_text(ident.mode, ident.fm_deviation, ident.filter_hz)
            snr = ident.result.snr_db
            lines = []
            if listen:
                lines.append(f"Listen with: {listen}")
            if ident.is_signal and snr > -90:
                where = format_signal_mhz(ident.signal_hz, ident.result.bandwidth_hz)
                lines.append(f"SNR {snr:.0f} dB · at {where}")
            self.listen_label.setText("\n".join(lines))
            self.listen_label.setVisible(bool(lines))
            self.details_label.setText("\n".join(f"• {d}" for d in ident.details))
            self.apply_btn.setEnabled(bool(ident.mode) and ident.is_signal)
            pulses = getattr(ident, "pulses", None)
            if pulses is not None and getattr(pulses, "pulses_us", None):
                from ..dsp.bursts import format_analysis

                self._pulse_text = format_analysis(pulses, ident.signal_hz)
                self.pulses_label.setText(pulse_lines(pulses))
            else:
                self._pulse_text = ""
                self.pulses_label.setText("")
            self.pulses_label.setVisible(bool(self._pulse_text))
            self.copy_pulses_btn.setVisible(bool(self._pulse_text))

        def _copy_pulses(self) -> None:
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None and self._pulse_text:
                clipboard.setText(self._pulse_text)
                self.message.emit(
                    "Copied the pulse analysis (as rtl_433 -A lays it out)",
                    "success",
                )

        # -- Signals -------------------------------------------------------------------

        def _build_signals(self) -> None:
            page, layout = self._page()
            # One line on top: the live list's options, or (after Find
            # Signals in Recording) the recording's name and Back to Live,
            # so the buttons under the list stay in view.
            self.signals_top = QStackedWidget()
            live = QWidget()
            row = QHBoxLayout(live)
            row.setContentsMargins(0, 0, 0, 0)
            self.markers_check = QCheckBox("Mark on spectrum")
            self.markers_check.setToolTip(
                "Label the signals in view on the spectrum (numbered as in this list)"
            )
            self.markers_check.toggled.connect(self.markers_changed.emit)
            row.addWidget(self.markers_check)
            row.addStretch(1)
            snr_label = QLabel("SNR ≥")
            row.addWidget(snr_label)
            self.min_snr = QDoubleSpinBox()
            self.min_snr.setRange(3.0, 40.0)
            self.min_snr.setSingleStep(1.0)
            self.min_snr.setDecimals(0)
            self.min_snr.setSuffix(" dB")
            self.min_snr.setValue(10.0)
            self.min_snr.setToolTip(
                "List signals at least this far above the noise floor around them"
            )
            self.min_snr.setAccessibleName("Minimum SNR")
            snr_label.setBuddy(self.min_snr)
            self.min_snr.valueChanged.connect(self.min_snr_changed.emit)
            row.addWidget(self.min_snr)
            self.signals_top.addWidget(live)
            self.recording_row = QWidget()
            rec = QHBoxLayout(self.recording_row)
            rec.setContentsMargins(0, 0, 0, 0)
            self.recording_label = _ElidedLabel()
            set_role(self.recording_label, "value")
            rec.addWidget(self.recording_label, 1)
            self.live_btn = QPushButton("Back to Live")
            set_role(self.live_btn, "compact")
            self.live_btn.setToolTip("List the signals the receiver hears now again")
            self.live_btn.clicked.connect(lambda _c=False: self._back_to_live())
            rec.addWidget(self.live_btn)
            self.signals_top.addWidget(self.recording_row)
            # (As tall as one line: the label elides rather than wraps.)
            self.signals_top.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
            layout.addWidget(self.signals_top)
            self.signals_table = self._table(
                ["MHz", "dBFS", "SNR", "Width", "Heard"],
                [
                    "Where the signal's power is centred (double-click to tune). "
                    "The row number is its number on the spectrum's markers.",
                    "Its strongest bin, in dBFS (the spectrum's scale)",
                    "How far it stands above the noise floor around it (dB)",
                    "Its width within 20 dB of its peak",
                    "When it was last heard",
                ],
                "Signals in view",
            )
            # The frequency takes the room left (a stretched last column
            # wouldn't shrink below its default and the table scrolled
            # sideways in a 1024 px window).
            header = self.signals_table.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            # Row numbers: the markers' labels.
            self.signals_table.verticalHeader().setVisible(True)
            self.signals_table.verticalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            self.signals_table.itemDoubleClicked.connect(
                lambda _item: self._activate_selected()
            )
            self.signals_table.itemSelectionChanged.connect(self._update_signal_buttons)
            layout.addWidget(self.signals_table, 1)
            self.signals_empty = ViewPlaceholder(self.signals_table)
            buttons = QHBoxLayout()
            buttons.setSpacing(4)
            self.tune_btn = QPushButton("Tune")
            self.tune_btn.setToolTip("Tune to the selected signal (or double-click it)")
            self.tune_btn.clicked.connect(lambda _c=False: self._activate_selected())
            buttons.addWidget(self.tune_btn)
            self.bookmark_btn = QPushButton("Bookmark")
            self.bookmark_btn.setToolTip("Add the selected signal to the bookmarks")
            self.bookmark_btn.clicked.connect(
                lambda _c=False: self._bookmark_selected()
            )
            buttons.addWidget(self.bookmark_btn)
            buttons.addStretch(1)
            self.strongest_btn = QPushButton("Strongest")
            self.strongest_btn.setToolTip(
                "Tune to the strongest signal in view (Home with the plots focused)"
            )
            self.strongest_btn.clicked.connect(
                lambda _c=False: self.step_requested.emit("strongest")
            )
            buttons.addWidget(self.strongest_btn)
            self.down_btn = QPushButton("◀")
            self.down_btn.setToolTip(
                "Tune to the next signal below the tuned frequency (Page Down "
                "with the plots focused)"
            )
            self.down_btn.setAccessibleName("Next signal down")
            self.down_btn.clicked.connect(
                lambda _c=False: self.step_requested.emit("down")
            )
            buttons.addWidget(self.down_btn)
            self.up_btn = QPushButton("▶")
            self.up_btn.setToolTip(
                "Tune to the next signal above the tuned frequency (Page Up "
                "with the plots focused)"
            )
            self.up_btn.setAccessibleName("Next signal up")
            self.up_btn.clicked.connect(lambda _c=False: self.step_requested.emit("up"))
            buttons.addWidget(self.up_btn)
            for btn in (self.down_btn, self.up_btn):
                btn.setFixedWidth(btn.fontMetrics().horizontalAdvance("◀") + 28)
            layout.addLayout(buttons)
            self.set_signals_empty(
                "No signals yet\nPress Start (Space): the signals in view are "
                "listed here, about twice a second."
            )
            index = self.tabs.addTab(page, _PAGE_TITLES["signals"])
            self.tabs.setTabToolTip(
                index, "The signals in the span: tune, bookmark, mark them"
            )

        def showing_recording(self) -> bool:
            return self._recording is not None

        def set_signals_empty(self, text: str) -> None:
            if self._recording is not None:
                return
            self._signals = []
            self.signals_table.setRowCount(0)
            self.signals_empty.set_text(text)
            self.signals_empty.set_visible(True)
            self._update_signal_buttons()

        def set_signals(
            self, signals: Sequence[Any], tuned_hz: float, ages: Sequence[float]
        ) -> None:
            """The live list: :class:`~sdr_module.dsp.signal_list.SignalPeak`
            objects and how long ago each was heard."""
            if self._recording is not None:
                return
            selected = self.selected_frequency()
            table = self.signals_table
            table.setUpdatesEnabled(False)
            paused = _pause_fit(table, rows=True)
            try:
                table.setSortingEnabled(False)
                table.setRowCount(len(signals))
                for row, (sig, age) in enumerate(zip(signals, ages, strict=True)):
                    near = abs(sig.frequency_hz - tuned_hz) <= max(
                        0.5 * sig.width_hz, 500.0
                    )
                    items = [
                        _NumericItem(
                            format_signal_mhz(sig.frequency_hz, sig.width_hz, False),
                            sig.frequency_hz,
                        ),
                        _NumericItem(f"{sig.peak_dbfs:.0f}", sig.peak_dbfs),
                        _NumericItem(f"{sig.snr_db:.0f}", sig.snr_db),
                        _NumericItem(format_width(sig.width_hz), sig.width_hz),
                        _NumericItem(format_ago(age), -age, align_right=False),
                    ]
                    items[0].setData(Qt.ItemDataRole.UserRole + 2, row)
                    for col, item in enumerate(items):
                        if near:
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)
                            item.setToolTip("The tuned frequency is on this signal")
                        table.setItem(row, col, item)
                self._signals = list(signals)
            finally:
                _resume_fit(table, paused, rows=True)
                table.setUpdatesEnabled(True)
            self.signals_empty.set_visible(not signals)
            if not signals:
                self.signals_empty.set_text(
                    "No signals above the noise\nLower the SNR, or tune to a "
                    "busier band."
                )
            if selected is not None:
                self._select_frequency(selected)
            self._update_signal_buttons()

        def set_recording_signals(
            self, name: str, signals: Sequence[Any], seconds: float
        ) -> None:
            """The signals found in a recording (File > Find Signals in
            Recording): :class:`~sdr_module.dsp.signal_list.SurveySignal`."""
            self._recording = (name, list(signals), float(seconds))
            short = name if len(name) <= 30 else f"{name[:14]}…{name[-14:]}"
            self.recording_label.setText(f"In {short} ({format_clock_s(seconds)}):")
            self.recording_label.setToolTip(
                f"The signals found in {name} ({format_clock_s(seconds)} long). "
                "Double-click one to play the recording from just before it, "
                "tuned to it."
            )
            self.signals_top.setCurrentWidget(self.recording_row)
            self.markers_check.setEnabled(False)
            table = self.signals_table
            paused = _pause_fit(table, rows=True)
            table.setSortingEnabled(False)
            table.setRowCount(len(signals))
            for row, sig in enumerate(signals):
                heard = f"{format_clock_s(sig.first_s)}–{format_clock_s(sig.last_s)}"
                items = [
                    _NumericItem(
                        format_signal_mhz(sig.frequency_hz, sig.width_hz, False),
                        sig.frequency_hz,
                    ),
                    _NumericItem(f"{sig.peak_dbfs:.0f}", sig.peak_dbfs),
                    _NumericItem(f"{sig.snr_db:.0f}", sig.snr_db),
                    _NumericItem(format_width(sig.width_hz), sig.width_hz),
                    _NumericItem(heard, sig.first_s, align_right=False),
                ]
                items[0].setData(Qt.ItemDataRole.UserRole + 2, row)
                items[4].setToolTip(
                    f"First heard at {format_clock_s(sig.first_s)}, last at "
                    f"{format_clock_s(sig.last_s)} ({sig.duty:.0%} of the recording)"
                )
                for col, item in enumerate(items):
                    table.setItem(row, col, item)
            _resume_fit(table, paused, rows=True)
            self._signals = list(signals)
            self.signals_empty.set_visible(not signals)
            if not signals:
                self.signals_empty.set_text(
                    f"No signals in {name}\nNothing stood out of the noise "
                    "(SNR ≥ 10 dB)."
                )
            self.show_page("signals")
            self._update_signal_buttons()

        def _back_to_live(self) -> None:
            self._recording = None
            self.signals_top.setCurrentIndex(0)
            self.markers_check.setEnabled(True)
            self.set_signals_empty("Listing the signals in view…")
            self.live_requested.emit()

        def selected_signal(self) -> Any:
            rows = self.signals_table.selectionModel().selectedRows()
            if not rows:
                return None
            item = self.signals_table.item(rows[0].row(), 0)
            if item is None:
                return None
            index = item.data(Qt.ItemDataRole.UserRole + 2)
            if index is None:
                return None
            index = int(index)
            return self._signals[index] if 0 <= index < len(self._signals) else None

        def selected_frequency(self) -> Optional[float]:
            sig = self.selected_signal()
            return float(sig.frequency_hz) if sig is not None else None

        def _select_frequency(self, freq: float) -> None:
            best, best_d = None, None
            for row, sig in enumerate(self._signals):
                d = abs(sig.frequency_hz - freq)
                if d <= max(0.5 * sig.width_hz, 2e3) and (best_d is None or d < best_d):
                    best, best_d = row, d
            if best is not None:
                self.signals_table.selectRow(best)

        def signals(self) -> List[Any]:
            return list(self._signals)

        def _activate_selected(self) -> None:
            sig = self.selected_signal()
            if sig is None:
                return
            if self._recording is not None:
                self.play_requested.emit(tune_frequency(sig), float(sig.first_s))
            else:
                self.tune_requested.emit(tune_frequency(sig))

        def _bookmark_selected(self) -> None:
            sig = self.selected_signal()
            if sig is not None:
                self.bookmark_requested.emit(tune_frequency(sig))

        def _update_signal_buttons(self) -> None:
            has = self.selected_signal() is not None
            self.tune_btn.setEnabled(has)
            self.tune_btn.setText("Play" if self._recording is not None else "Tune")
            self.bookmark_btn.setEnabled(has)
            live = self._recording is None and bool(self._signals)
            for btn in (self.strongest_btn, self.down_btn, self.up_btn):
                btn.setEnabled(live)

        # -- Scope -------------------------------------------------------------------------

        def _build_scope(self) -> None:
            page, layout = self._page()
            row = QHBoxLayout()
            row.setSpacing(4)
            self.view_combo = QComboBox()
            for text, key in SCOPE_VIEWS:
                self.view_combo.addItem(text, key)
            self.view_combo.setToolTip(
                "What to show of the tuned channel: its magnitude, I and Q, "
                "phase or instantaneous frequency against time, or a "
                "constellation (I against Q)"
            )
            self.view_combo.setAccessibleName("Scope view")
            row.addWidget(self.view_combo)
            self.overlay_combo = QComboBox()
            for text, overlay in OVERLAYS:
                self.overlay_combo.addItem(text, overlay)
            self.overlay_combo.setToolTip(
                "Ideal points to compare with (the constellation is turned to "
                "match them) and the error vector magnitude (EVM) against them"
            )
            self.overlay_combo.setAccessibleName("Constellation overlay")
            row.addWidget(self.overlay_combo)
            self.span_combo = QComboBox()
            for ms in SCOPE_SPANS_MS:
                self.span_combo.addItem(f"{ms:g} ms", ms)
            self.span_combo.setCurrentIndex(1)
            self.span_combo.setToolTip("How much time the trace shows")
            self.span_combo.setAccessibleName("Time span")
            row.addWidget(self.span_combo)
            row.addStretch(1)
            self.freeze_btn = QPushButton("Freeze")
            self.freeze_btn.setCheckable(True)
            set_role(self.freeze_btn, "compact")
            self.freeze_btn.setToolTip("Hold the picture (click again to go on)")
            row.addWidget(self.freeze_btn)
            layout.addLayout(row)
            self.scope = ScopeView()
            layout.addWidget(self.scope, 1)
            self.scope_info = QLabel("")
            set_role(self.scope_info, "hint")
            self.scope_info.setWordWrap(True)
            layout.addWidget(self.scope_info)
            self.view_combo.currentIndexChanged.connect(self._on_scope_view)
            self.overlay_combo.currentIndexChanged.connect(
                lambda _i: self.scope.set_overlay(self.overlay_combo.currentData())
            )
            self.span_combo.currentIndexChanged.connect(
                lambda _i: self.scope.set_span_ms(float(self.span_combo.currentData()))
            )
            self.freeze_btn.toggled.connect(self.scope.set_frozen)
            self._on_scope_view()
            index = self.tabs.addTab(page, _PAGE_TITLES["scope"])
            self.tabs.setTabToolTip(
                index, "The tuned channel in time, or as a constellation"
            )

        def _on_scope_view(self, _index: int = 0) -> None:
            view = self.view_combo.currentData()
            self.scope.set_view(view)
            constellation = view == "constellation"
            # One of them at a time (the row stays narrow): the overlay for
            # a constellation, the time span otherwise.
            self.overlay_combo.setVisible(constellation)
            self.span_combo.setVisible(not constellation)
            self.scope_changed.emit()

        def set_scope_info(self, text: str) -> None:
            self.scope_info.setText(text)

        # -- Bursts --------------------------------------------------------------------------

        def _build_bursts(self) -> None:
            page, layout = self._page()
            row = QHBoxLayout()
            self.watch_check = QCheckBox("Watch in the background")
            self.watch_check.setChecked(False)
            self.watch_check.setToolTip(
                "Bursts (remote controls, sensors, data, voice) are logged "
                "while this page is on screen; tick this to log them in the "
                "background too (it costs a little CPU)"
            )
            self.watch_check.toggled.connect(self.watch_changed.emit)
            row.addWidget(self.watch_check)
            row.addStretch(1)
            self.clear_bursts_btn = QPushButton("Clear")
            set_role(self.clear_bursts_btn, "compact")
            self.clear_bursts_btn.setToolTip("Empty the list")
            self.clear_bursts_btn.clicked.connect(lambda _c=False: self.clear_bursts())
            row.addWidget(self.clear_bursts_btn)
            self.export_bursts_btn = QPushButton("Export CSV…")
            set_role(self.export_bursts_btn, "compact")
            self.export_bursts_btn.setToolTip("Save the list as a CSV file")
            row.addWidget(self.export_bursts_btn)
            layout.addLayout(row)
            self.bursts_table = self._table(
                ["Time", "MHz", "Length", "dBFS", "What"],
                [
                    "When it started",
                    "Where its power was centred (MHz)",
                    "How long it lasted",
                    "Its strongest level, in dBFS",
                    "What it looks like (OOK, FSK, voice...) and, for on/off "
                    "keyed data, its bits in hex",
                ],
                "Bursts",
            )
            self.bursts_table.itemSelectionChanged.connect(self._show_burst_details)
            self.bursts_table.itemDoubleClicked.connect(
                lambda _item: self._tune_selected_burst()
            )
            # The list is the page: at least three rows show, and the
            # details of the selected burst share the room under it (drag
            # the line between them); there are none until one is selected.
            table = self.bursts_table
            table.setMinimumHeight(
                table.horizontalHeader().sizeHint().height()
                + 3 * table.verticalHeader().defaultSectionSize()
                + 2 * table.frameWidth()
            )
            self.bursts_split = QSplitter(Qt.Orientation.Vertical)
            self.bursts_split.setChildrenCollapsible(False)
            self.bursts_split.addWidget(table)
            self.bursts_empty = ViewPlaceholder(self.bursts_table)
            self.burst_details = QPlainTextEdit()
            self.burst_details.setReadOnly(True)
            self.burst_details.setMinimumHeight(40)
            mono = QFont("monospace")
            mono.setStyleHint(QFont.StyleHint.Monospace)
            self.burst_details.setFont(mono)
            self.burst_details.setPlaceholderText(
                "Select a burst to see its pulses (as rtl_433 -A lays them out)"
            )
            self.burst_details.setAccessibleName("Burst details")
            self.bursts_split.addWidget(self.burst_details)
            self.bursts_split.setStretchFactor(0, 3)
            self.bursts_split.setStretchFactor(1, 2)
            self.burst_details.setVisible(False)
            layout.addWidget(self.bursts_split, 1)
            hint = "Select a burst to see its pulses; double-click one to tune to it."
            self.bursts_hint = _ElidedLabel(hint, mode=Qt.TextElideMode.ElideRight)
            self.bursts_hint.setToolTip(hint)
            set_role(self.bursts_hint, "hint")
            self.bursts_hint.setVisible(False)
            layout.addWidget(self.bursts_hint)
            self.set_bursts_empty(BURSTS_EMPTY)
            index = self.tabs.addTab(page, _PAGE_TITLES["bursts"])
            self.tabs.setTabToolTip(
                index, "Transmissions that come and go in the span, taken apart"
            )

        def set_bursts_empty(self, text: str) -> None:
            if self._bursts:
                return
            self.bursts_empty.set_text(text)
            self.bursts_empty.set_visible(True)

        def bursts(self) -> List[Any]:
            return list(self._bursts)

        def add_burst(self, report: Any, when: Optional[float] = None) -> None:
            """Add a :class:`~sdr_module.dsp.bursts.BurstReport` (newest at
            the top; at most 500 kept)."""
            burst = report.burst
            stamp = time.strftime("%H:%M:%S", time.localtime(when or time.time()))
            self._bursts.insert(0, (stamp, report))
            del self._bursts[500:]
            table = self.bursts_table
            paused = _pause_fit(table)
            table.insertRow(0)
            data = ""
            analysis = report.analysis
            if analysis is not None:
                rows = analysis.distinct_rows()
                if rows:
                    from ..dsp.bursts import bits_to_hex

                    bits, count = rows[0]
                    data = f"{{{len(bits)}}}{bits_to_hex(bits)[:24]}"
                    if count > 1:
                        data += f" ×{count}"
            kind = burst_kind_text(report)
            what = f"{kind} {data}".strip()
            freq_item = _NumericItem(f"{burst.center_hz / 1e6:.3f}", burst.center_hz)
            freq_item.setToolTip(format_mhz(burst.center_hz))
            what_item = QTableWidgetItem(what)
            what_item.setToolTip(what)
            items = [
                QTableWidgetItem(stamp),
                freq_item,
                _NumericItem(_duration_text(burst.duration_s), burst.duration_s),
                _NumericItem(f"{burst.peak_dbfs:.0f}", burst.peak_dbfs),
                what_item,
            ]
            for col, item in enumerate(items):
                table.setItem(0, col, item)
            while table.rowCount() > 500:
                table.removeRow(table.rowCount() - 1)
            _resume_fit(table, paused)
            self.bursts_empty.set_visible(False)
            self.bursts_hint.setVisible(self.burst_details.isHidden())

        def clear_bursts(self) -> None:
            self._bursts = []
            self.bursts_table.setRowCount(0)
            self._show_burst_details()
            self.set_bursts_empty("List cleared\nNew bursts in the span show up here.")

        def selected_burst(self) -> Any:
            rows = self.bursts_table.selectionModel().selectedRows()
            if not rows:
                return None
            row = rows[0].row()
            return self._bursts[row][1] if 0 <= row < len(self._bursts) else None

        def _show_burst_details(self) -> None:
            report = self.selected_burst()
            if report is None:
                self.burst_details.setPlainText("")
                self.burst_details.setVisible(False)
                self.bursts_hint.setVisible(bool(self._bursts))
                return
            self.burst_details.setPlainText(burst_details_text(report))
            if self.burst_details.isHidden():
                self.bursts_hint.setVisible(False)
                self.burst_details.setVisible(True)
                # The table keeps three rows or three fifths, whichever is
                # more; the selected burst stays in view.
                split = self.bursts_split
                total = sum(split.sizes()) or split.height()
                keep = max(self.bursts_table.minimumHeight(), int(0.6 * total))
                split.setSizes([keep, max(1, total - keep)])
                rows = self.bursts_table.selectionModel().selectedRows()
                if rows:
                    self.bursts_table.scrollTo(rows[0])

        def _tune_selected_burst(self) -> None:
            report = self.selected_burst()
            if report is not None:
                self.tune_requested.emit(float(report.burst.center_hz))

        def export_bursts(self, path: str) -> int:
            """Write the list to ``path`` as CSV; returns the rows written."""
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Time",
                        "Frequency (MHz)",
                        "Width (kHz)",
                        "Length (ms)",
                        "Peak (dBFS)",
                        "SNR (dB)",
                        "Type",
                        "Modulation",
                        "Bits",
                        "Hex",
                    ]
                )
                for stamp, report in reversed(self._bursts):
                    b = report.burst
                    a = report.analysis
                    rows = a.distinct_rows() if a is not None else []
                    bits = rows[0][0] if rows else ""
                    from ..dsp.bursts import bits_to_hex

                    writer.writerow(
                        [
                            stamp,
                            f"{b.center_hz / 1e6:.6f}",
                            f"{b.width_hz / 1e3:.1f}",
                            f"{b.duration_s * 1e3:.0f}",
                            f"{b.peak_dbfs:.1f}",
                            f"{b.snr_db:.1f}",
                            burst_kind_text(report),
                            a.modulation if a is not None else "",
                            len(bits) if bits else "",
                            bits_to_hex(bits) if bits else "",
                        ]
                    )
            return len(self._bursts)


def _duration_text(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.1f} s"
    return f"{seconds * 1e3:.0f} ms"


def burst_kind_text(report: Any) -> str:
    """What a burst is, in a few words: the pulse analysis's modulation
    for keyed data, else the classifier's label."""
    analysis = getattr(report, "analysis", None)
    result = getattr(report, "classification", None)
    kind = str(getattr(result, "kind", "") or "")
    if analysis is not None and analysis.pulses_us and kind in ("ook", "fsk", ""):
        return analysis.modulation
    if result is not None and kind not in ("", "unknown", "noise"):
        return str(getattr(result, "label", kind))
    if analysis is not None and analysis.pulses_us:
        return analysis.modulation
    return "Burst"


def pulse_lines(analysis: Any) -> str:
    """A few lines about a pulse analysis for the Identify card: the bits
    first (what it sent), then how it sends them."""
    from ..dsp.bursts import bits_to_hex

    lines = []
    rows = analysis.distinct_rows()
    if rows:
        bits, count = rows[0]
        repeat = f" (×{count})" if count > 1 else ""
        lines.append(f"Bits: {{{len(bits)}}}{bits_to_hex(bits)}{repeat}")
    lines.append(f"{analysis.description}")
    widths = ", ".join(f"{w:.0f}" for w, _c in analysis.pulse_widths[:4])
    if widths:
        lines.append(f"Pulses: {widths} µs")
    gaps = ", ".join(f"{w:.0f}" for w, _c in analysis.gap_widths[:4])
    if gaps:
        lines.append(f"Gaps: {gaps} µs")
    return "\n".join(lines)


def burst_details_text(report: Any) -> str:
    """The details of a burst: when, where, and its pulse analysis."""
    from ..dsp.bursts import format_analysis

    b = report.burst
    lines = [
        f"{format_mhz(b.center_hz)}, {format_width(b.width_hz)} wide, "
        f"{_duration_text(b.duration_s)}",
        f"Peak {b.peak_dbfs:.1f} dBFS, {b.snr_db:.0f} dB above the noise",
    ]
    result = getattr(report, "classification", None)
    if result is not None and getattr(result, "kind", "") not in ("", "noise"):
        lines.append(
            f"Looks like: {result.label} ({int(round(100 * result.confidence))} %)"
        )
    if report.analysis is not None and report.analysis.pulses_us:
        lines.append("")
        lines.append(format_analysis(report.analysis))
    return "\n".join(lines)
