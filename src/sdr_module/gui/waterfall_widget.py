"""
Waterfall display widget.

Provides scrolling time-frequency visualization with:
- Configurable color maps and dynamic range
- Newest line at the top, scrolling down (the usual SDR convention)
- Time axis (seconds before the newest line), pause separators, hover readout
- Click-to-tune, protocol highlighting

The plot shares its horizontal margins with the spectrum widget (see
:func:`~sdr_module.gui.spectrum_widget.plot_side_margins`), so a frequency
sits at the same x in both displays. Like the spectrum trace, the displayed
image is decimated to the plot width with a per-column maximum, so a narrow
carrier stays bright instead of being blurred away by image scaling.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .spectrum_widget import (
    axis_font,
    columns_max,
    draw_placeholder,
    draw_readout,
    fit_header_hint,
    format_mhz,
    nice_ticks,
    plot_side_margins,
    readout_decimals,
    snap_step_hz,
)
from .themes import get_palette, set_role, theme_notifier

try:
    from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QPolygonF
    from PyQt6.QtWidgets import (
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


# Dynamic-range choices (dB below 0 dBFS shown by the colormap).
_RANGE_CHOICES = (60, 80, 100, 120)

# Time-axis steps (seconds): short labels that fit the shared left margin.
_TIME_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)


def time_step(raw: float) -> float:
    """Smallest time-axis step (seconds) that is >= ``raw``."""
    for step in _TIME_STEPS:
        if step >= raw:
            return float(step)
    return float(_TIME_STEPS[-1])


#: A pause between two lines longer than this many typical line intervals
#: (and at least _GAP_MIN_S seconds) is drawn as a separator.
_GAP_FACTOR = 5.0
_GAP_MIN_S = 1.0


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
    """At most four characters for any age: ``4.2s``, ``65s``, ``12m``, ``3h``, ``5d``."""
    if age < 10:
        return f"{max(0.0, age):.1f}s"
    if age < 99.5:
        return f"{age:.0f}s"
    if age < 99.5 * 60:
        return f"{max(2.0, round(age / 60)):.0f}m"
    if age < 99.5 * 3600:
        return f"{max(2.0, round(age / 3600)):.0f}h"
    return f"{max(5.0, round(age / 86400)):.0f}d"


class _WaterfallCanvas(QWidget if HAS_PYQT6 else object):
    """The painted plot area below the waterfall header strip."""

    def __init__(self, owner: "WaterfallWidget"):
        super().__init__(owner)
        self._owner = owner
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumHeight(120)

    def paintEvent(self, event):  # noqa: N802 - Qt override
        self._owner._paint_canvas(self)

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


class WaterfallWidget(QWidget if HAS_PYQT6 else object):
    """
    Waterfall display widget.

    Shows scrolling spectrogram with time on Y-axis and frequency on X-axis.
    The newest line is at the top. Clicking emits :attr:`frequency_clicked`.
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

    def __init__(self, parent=None, history_size: int = 500):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Display settings
        self._history_size = history_size
        self._fft_size = 2048
        self._db_range = (-100, 0)
        self._center_freq = 100e6
        self._sample_rate = 2.4e6

        # Data storage (newest last) and the arrival time of each line
        self._history: deque = deque(maxlen=history_size)
        self._times: deque = deque(maxlen=history_size)

        # Color map
        self._colormap_name = "turbo"
        self._colormap = self._build_colormap(self._colormap_name)

        # Image buffer. ``_image_rgb`` is the contiguous uint8 (H, W, 3) array
        # that ``_image`` (a QImage view) points at; we must keep a reference to
        # it alive for as long as the QImage exists, since QImage does not copy
        # the buffer it is constructed from. Row 0 is the newest line.
        self._image: Optional[QImage] = None
        self._image_rgb: Optional[np.ndarray] = None
        self._image_theme: Optional[str] = None
        # Colormap indices of the same rows (uint8, row 0 newest), so the
        # display image can be re-decimated cheaply when the plot resizes.
        self._index_buf: Optional[np.ndarray] = None
        # What is actually drawn: the image decimated to the plot's pixel
        # width with a per-column maximum (RGBX, 4 bytes per pixel). None
        # until the first paint, or when the plot is at least fft_size wide.
        self._disp_rgb: Optional[np.ndarray] = None
        self._disp_image: Optional[QImage] = None
        self._lut: Optional[np.ndarray] = None
        self._lut_src: Optional[np.ndarray] = None

        # Highlights
        self._highlights: List[Tuple[int, int, int, int, QColor]] = []

        # Paint state
        self._font = axis_font()
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
        """Header strip (title, hint, colormap/range) above the plot canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        set_role(header, "header-strip")
        self._header = header
        controls = QHBoxLayout(header)
        controls.setContentsMargins(10, 3, 8, 3)
        controls.setSpacing(8)

        title = QLabel("WATERFALL")
        set_role(title, "caption")
        controls.addWidget(title)

        hint = QLabel("Newest at top · click to tune")
        set_role(hint, "hint")
        hint.setToolTip(
            "Each row is one spectrum; the newest row is at the top and older "
            "rows scroll down. Click to tune to a frequency."
        )
        # Never let the optional hint raise the widget's minimum width:
        # fit_header_hint() hides it before it would be squeezed.
        hint.setMinimumWidth(1)
        controls.addWidget(hint)
        self._hint_label = hint

        controls.addStretch(1)

        color_label = QLabel("COLORS")
        set_role(color_label, "caption")
        controls.addWidget(color_label)
        self._color_combo = QComboBox()
        for name in self.COLORMAPS:
            self._color_combo.addItem(name.capitalize(), name)
        self._color_combo.setCurrentIndex(
            self._color_combo.findData(self._colormap_name)
        )
        self._color_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._color_combo.setToolTip(
            "Color map used to paint signal strength (weak to strong)"
        )
        color_label.setToolTip(self._color_combo.toolTip())
        color_label.setBuddy(self._color_combo)
        self._color_combo.currentIndexChanged.connect(self._on_color_index_changed)
        # Header controls don't take focus on click, so Space/arrow keys keep
        # acting on the plot (start/stop, tuning). Tab still reaches them.
        self._color_combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        controls.addWidget(self._color_combo)

        controls.addSpacing(4)

        range_label = QLabel("RANGE")
        set_role(range_label, "caption")
        controls.addWidget(range_label)
        self._range_combo = QComboBox()
        for db in _RANGE_CHOICES:
            self._range_combo.addItem(f"{db} dB", db)
        self._range_combo.setCurrentIndex(2)
        self._range_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._range_combo.setToolTip(
            "Dynamic range shown, from 0 dBFS down. A smaller range gives more "
            "contrast; a larger one shows weaker signals."
        )
        range_label.setToolTip(self._range_combo.toolTip())
        range_label.setBuddy(self._range_combo)
        self._range_combo.currentIndexChanged.connect(self._on_range_changed)
        self._range_combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        controls.addWidget(self._range_combo)

        self._clear_btn = QPushButton("Clear")
        set_role(self._clear_btn, "compact")
        self._clear_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        controls.addWidget(self._clear_btn)
        self._update_clear_button()

        layout.addWidget(header)

        self._canvas = _WaterfallCanvas(self)
        self._canvas.setAccessibleName("Waterfall plot")
        self._canvas.setAccessibleDescription(
            "Signal strength over time and frequency, newest at the top. "
            "Click to tune to a frequency."
        )
        layout.addWidget(self._canvas, 1)

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        fit_header_hint(self, self._header, self._hint_label)

    def _update_clear_button(self) -> None:
        has_data = len(self._history) > 0
        if self._clear_btn.isEnabled() == has_data and self._clear_btn.toolTip():
            return
        self._clear_btn.setEnabled(has_data)
        self._clear_btn.setToolTip(
            "Erase the waterfall history"
            if has_data
            else "Nothing to clear yet: the waterfall is empty"
        )

    def _on_clear_clicked(self) -> None:
        self.clear()

    def _on_color_index_changed(self, index: int) -> None:
        name = self._color_combo.itemData(index)
        if name:
            self._on_colormap_changed(str(name))

    def _on_colormap_changed(self, name: str):
        """Handle colormap change."""
        self._colormap_name = name
        self._colormap = self._build_colormap(name)
        idx = self._color_combo.findData(name)
        if idx >= 0 and idx != self._color_combo.currentIndex():
            self._color_combo.blockSignals(True)
            self._color_combo.setCurrentIndex(idx)
            self._color_combo.blockSignals(False)
        self._render_image()
        self._canvas.update()

    def _on_range_changed(self, index: int):
        """Handle range change."""
        index = max(0, min(index, len(_RANGE_CHOICES) - 1))
        self._db_range = (-_RANGE_CHOICES[index], 0)
        if index != self._range_combo.currentIndex():
            self._range_combo.blockSignals(True)
            self._range_combo.setCurrentIndex(index)
            self._range_combo.blockSignals(False)
        self._render_image()
        self._canvas.update()

    def _on_theme_changed(self, _name: str = "") -> None:
        try:
            self._render_image()
            self._canvas.update()
        except RuntimeError:  # pragma: no cover - widget already deleted
            pass

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

        Args:
            power_db: Power spectrum in dB
        """
        if len(power_db) == 0:
            # No spectrum data: record a neutral (min-dB) row so history stays
            # in sync with caller cadence but we skip the expensive interp path.
            power_db = np.full(self._fft_size, self._db_range[0], dtype=np.float32)
        elif len(power_db) != self._fft_size:
            # Resample if needed
            power_db = np.interp(
                np.linspace(0, 1, self._fft_size),
                np.linspace(0, 1, len(power_db)),
                power_db,
            )

        line = np.array(power_db, copy=True)
        self._history.append(line)
        self._times.append(time.monotonic())

        buf = self._image_rgb
        index_buf = self._index_buf
        if (
            buf is None
            or index_buf is None
            or buf.shape != (self._history_size, self._fft_size, 3)
            or self._image_theme != get_palette().name
        ):
            self._render_image()
        else:
            # Scroll down one row and paint the new line at the top. NumPy
            # handles the overlapping copies correctly.
            idx = self._level_index(line[np.newaxis, :])[0]
            buf[1:] = buf[:-1]
            buf[0] = self._colormap[idx]
            index_buf[1:] = index_buf[:-1]
            index_buf[0] = idx
            disp = self._disp_rgb
            if disp is not None:
                disp[1:] = disp[:-1]
                disp[0] = self._rgbx_lut()[columns_max(idx, disp.shape[1])]
                self._wrap_display()
            self._wrap_image()
        if not self._clear_btn.isEnabled():
            self._update_clear_button()
        self._canvas.update()

    def clear(self):
        """Clear the waterfall."""
        self._history.clear()
        self._times.clear()
        self._image = None
        self._image_rgb = None
        self._index_buf = None
        self._disp_rgb = None
        self._disp_image = None
        self._update_clear_button()
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

    def save_image(self, path: str) -> bool:
        """Save the current waterfall image to a file. Returns success."""
        if self._image is None:
            return False
        return bool(self._image.save(path))

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
        color: QColor,
    ):
        """Add a highlight region (image rows ``time_*``, FFT bins ``freq_*``)."""
        self._highlights.append((time_start, time_end, freq_start, freq_end, color))
        self._canvas.update()

    def clear_highlights(self):
        """Clear all highlights."""
        self._highlights.clear()
        self._canvas.update()

    # ------------------------------------------------------------------
    # Internals: image
    # ------------------------------------------------------------------

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

    def _rgbx_lut(self) -> np.ndarray:
        """The colormap as 4-byte RGBX entries for the display image."""
        if self._lut is None or self._lut_src is not self._colormap:
            lut = np.empty((256, 4), dtype=np.uint8)
            lut[:, :3] = self._colormap
            lut[:, 3] = 255
            self._lut, self._lut_src = lut, self._colormap
        return self._lut

    def _wrap_image(self) -> None:
        """(Re)wrap ``_image_rgb`` in a QImage (no pixel copy)."""
        buf = self._image_rgb
        self._image = QImage(
            buf.data,
            self._fft_size,
            self._history_size,
            3 * self._fft_size,
            QImage.Format.Format_RGB888,
        )

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

    def _display_columns(self, plot: "QRectF") -> int:
        """Image columns the plot can show (device pixels, at most fft_size)."""
        dpr = self._canvas.devicePixelRatioF() or 1.0
        return int(max(1, min(self._fft_size, math.ceil(plot.width() * dpr))))

    def _display_image(self, cols: int) -> Optional["QImage"]:
        """The image to draw for a plot ``cols`` device pixels wide.

        Narrower than the FFT: a per-column-maximum decimation of the rows
        (cached, scrolled by :meth:`add_line`, rebuilt when the width, the
        colormap, the range or the theme changes). Otherwise the full image.
        """
        if self._image is None or self._index_buf is None:
            return self._image
        if cols >= self._fft_size:
            self._disp_rgb = None
            self._disp_image = None
            return self._image
        disp = self._disp_rgb
        if disp is None or disp.shape[1] != cols:
            n = len(self._history)
            bg = get_palette().qcolor("plot_bg")
            bg_px = np.array([bg.red(), bg.green(), bg.blue(), 255], dtype=np.uint8)
            # Work on 32-bit pixels: one gather per pixel instead of four.
            pixels = np.empty((self._history_size, cols), dtype=np.uint32)
            pixels[:] = bg_px.view(np.uint32)[0]
            if n:
                pooled = columns_max(self._index_buf[:n], cols, axis=1)
                pixels[:n] = self._rgbx_lut().view(np.uint32).reshape(256)[pooled]
            self._disp_rgb = pixels.view(np.uint8).reshape(self._history_size, cols, 4)
            self._wrap_display()
        return self._disp_image

    def _render_image(self):
        """Rebuild the whole image buffer from history, vectorized.

        The waterfall is ``history_size`` rows by ``fft_size`` columns, with the
        newest line at the top. Rendering this per-pixel in Python (a nested
        ``QImage.pixel``/``setPixel`` scroll over ~1M pixels) took ~0.4 s per
        line, far above the ~33 ms display cadence, so acquisition froze the UI.

        Instead we map every history line to colormap indices and gather the RGB
        rows with NumPy in one pass, then wrap the resulting contiguous
        ``(H, W, 3)`` uint8 array in a QImage. The array is kept alive on
        ``self._image_rgb`` because QImage references the buffer without copying.
        New lines only scroll the buffer (see :meth:`add_line`); a full rebuild
        runs when the colormap, range or theme changes.
        """
        palette = get_palette()
        self._image_theme = palette.name
        # The decimated display image is rebuilt lazily at the next paint.
        self._disp_rgb = None
        self._disp_image = None
        if len(self._history) == 0:
            self._image = None
            self._image_rgb = None
            self._index_buf = None
            return

        # Background-filled buffer; history fills the top rows, newest first.
        bg = palette.qcolor("plot_bg")
        buf = np.empty((self._history_size, self._fft_size, 3), dtype=np.uint8)
        buf[:, :, 0] = bg.red()
        buf[:, :, 1] = bg.green()
        buf[:, :, 2] = bg.blue()

        # add_line guarantees each row is already fft_size long.
        lines = np.stack(list(reversed(self._history)))
        idx = self._level_index(lines)
        index_buf = np.zeros((self._history_size, self._fft_size), dtype=np.uint8)
        index_buf[: idx.shape[0]] = idx
        self._index_buf = index_buf
        buf[: idx.shape[0]] = self._colormap[idx]

        # Keep the buffer alive: QImage does not copy it.
        self._image_rgb = np.ascontiguousarray(buf)
        self._wrap_image()

    # ------------------------------------------------------------------
    # Internals: interaction
    # ------------------------------------------------------------------

    def _margins(self) -> Tuple[int, int, int, int]:
        left, right = plot_side_margins(self._fm)
        return left, 4, right, 4

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

    def _set_hover(self, pos: Optional["QPointF"]) -> None:
        self._hover = QPointF(pos) if pos is not None else None
        self._canvas.update()

    def _click_at(self, pos: "QPointF") -> bool:
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return False
        self.frequency_clicked.emit(float(freq))
        return True

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

    @staticmethod
    def _gap_rows(ages: np.ndarray, seconds_per_row: float) -> np.ndarray:
        """Rows that start an older run after a pause in reception."""
        if len(ages) < 2:
            return np.empty(0, dtype=np.intp)
        threshold = max(_GAP_FACTOR * seconds_per_row, _GAP_MIN_S)
        return np.nonzero(np.diff(ages) > threshold)[0] + 1

    # ------------------------------------------------------------------
    # Internals: painting
    # ------------------------------------------------------------------

    def _paint_canvas(self, canvas: "QWidget") -> None:
        p = get_palette()
        if self._history and self._image_theme != p.name:
            self._render_image()
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

            # Frame around the plot
            painter.setPen(QPen(p.qcolor("plot_grid"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(plot.adjusted(-0.5, -0.5, 0.5, 0.5))

            self._draw_time_axis(painter, plot, p)
            self._draw_highlights(painter, plot)

            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_center_marker(painter, plot, p)

            if self._image is None:
                draw_placeholder(
                    painter,
                    plot,
                    "Waiting for data",
                    "The waterfall fills in once receiving starts",
                    canvas.font(),
                    p,
                )
            if self._hover is not None:
                # Also shown before any data: clicking tunes either way.
                painter.setFont(self._font)
                self._draw_hover(painter, plot, p)
        finally:
            painter.end()

    def _draw_time_axis(self, painter: "QPainter", plot: "QRectF", p) -> None:
        """Seconds-ago labels in the left margin (0s at the newest line).

        Labels are placed at the rows whose lines actually arrived that long
        ago, so they stay right when reception was paused and resumed. Each
        pause gets a separator across the plot, labelled with the age of the
        older lines below it.
        """
        spr = self._seconds_per_row()
        ages = self._row_ages()
        if spr is None or ages is None:
            return
        rows_px = plot.height() / self._history_size
        fm = self._fm
        fh = fm.height()
        label_right = plot.left() - 6
        right_align = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tick_pen = QPen(p.qcolor("plot_grid"), 1)
        axis_color = p.qcolor("plot_axis")
        min_sep = fh + 2

        def draw_label(y: float, text: str) -> None:
            # Keep the label inside the plot's vertical extent.
            y_label = min(max(y, plot.top() + fh / 2), plot.bottom() - fh / 2)
            painter.setPen(tick_pen)
            painter.drawLine(QPointF(plot.left() - 3, y), QPointF(plot.left(), y))
            painter.setPen(axis_color)
            painter.drawText(
                QRectF(0, y_label - fh / 2, label_right - 2, fh), right_align, text
            )

        # Pauses: a separator line across the plot plus the older lines' age.
        gaps = self._gap_rows(ages, spr)
        gap_ys: List[float] = []
        if gaps.size:
            separator = QPen(p.qcolor("plot_bg", 230), 1)
            for row in gaps.tolist():
                y = round(plot.top() + row * rows_px) + 0.5
                painter.setPen(separator)
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                if gap_ys and y - gap_ys[-1] < min_sep:
                    continue
                draw_label(y, format_elapsed(float(ages[row])))
                gap_ys.append(y)

        # Regular ticks within the newest continuous run of lines.
        run = ages[: int(gaps[0])] if gaps.size else ages
        step = time_step(spr * self._history_size * max(28.0, fh * 2.2) / plot.height())
        last_y = -math.inf
        for age in nice_ticks(0.0, float(run[-1]) + 1e-9, step):
            # Fractional row between the two lines that bracket this age (the
            # run has no pauses, so the pair is always close together).
            i = min(int(np.searchsorted(run, age)), len(run) - 1)
            if i == 0:
                row = 0.0
            else:
                a0, a1 = float(run[i - 1]), float(run[i])
                row = i - 1 + ((age - a0) / (a1 - a0) if a1 > a0 else 1.0)
            y = plot.top() + (row + 0.5) * rows_px
            if y - last_y < min_sep or any(abs(y - g) < min_sep for g in gap_ys):
                continue
            draw_label(y, format_age(age, step))
            last_y = y

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

    def _draw_hover(self, painter: "QPainter", plot: "QRectF", p) -> None:
        pos = self._hover
        if pos is None or not plot.contains(pos):
            return
        freq = self._snapped_frequency_at(pos.x())
        if freq is None:
            return
        x = pos.x()
        # A thin line with a halo stays visible on any colormap.
        painter.setPen(QPen(p.qcolor("plot_bg", 150), 3))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.setPen(QPen(p.qcolor("plot_text", 220), 1))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        decimals = readout_decimals(snap_step_hz(self._hz_per_px()))
        lines = [f"{format_mhz(freq, decimals)} MHz"]
        row = int((pos.y() - plot.top()) / plot.height() * self._history_size)
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
            detail = f"{finite.max():.0f} dBFS" if finite.size else "-- dBFS"
            age = self._times[-1] - self._times[len(self._times) - 1 - row]
            detail += f" · {format_elapsed(age)} ago"
            lines.append(detail)
        lines.append("Click to tune")
        draw_readout(painter, lines, x, plot, self._fm, p)
