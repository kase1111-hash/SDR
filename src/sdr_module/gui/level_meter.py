"""
Toolbar signal meter: the tuned channel's level against the squelch.

A compact bar (-120 to 0 dBFS) with the level in dBFS, the noise floor as a
tick, the squelch threshold as a marker you can drag, and a lamp that says
whether the squelch is open::

    [######|######:   ]  -46.2 dBFS  OPEN
            ^noise  ^threshold

It only paints what the main window hands it (:meth:`LevelMeter.set_reading`)
and reports a dragged threshold (:attr:`LevelMeter.threshold_changed`); the
squelch itself lives in the window. Colours come from the theme palette.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

try:
    from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
    from PyQt6.QtGui import QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
    from PyQt6.QtWidgets import QSizePolicy, QWidget

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

from .themes import get_palette, mono_font, theme_notifier

#: Squelch states the lamp shows: (lamp text, what it means).
STATES = {
    "open": ("OPEN", "the squelch is open: the signal plays"),
    "closed": ("SQL", "squelched: quieter than the threshold, so muted"),
    "off": ("OFF", "the squelch is off: everything plays, noise too"),
    "idle": ("I/Q", "no audio in None (I/Q) mode: choose AM, FM, USB, LSB or CW"),
    "none": ("—", "not receiving"),
}

_NO_VALUE = "—"


def _format_db(db: Optional[float]) -> str:
    if db is None or not math.isfinite(db):
        return f"{_NO_VALUE} dBFS"
    return f"{db:.1f} dBFS"


class LevelMeter(QWidget if HAS_PYQT6 else object):
    """The level of the tuned channel, with the squelch threshold.

    * :meth:`set_reading` shows a level, the channel's noise floor, the
      threshold in effect and the squelch state (a key of :data:`STATES`).
    * Pressing or dragging on the bar moves the threshold there and emits
      :attr:`threshold_changed` (dBFS), so the squelch can be set right
      where the signal is.
    * :meth:`text` and :meth:`setText` read and set the dBFS readout like
      the label this meter replaces.
    """

    if HAS_PYQT6:
        threshold_changed = pyqtSignal(float)

    MIN_DB = -120.0
    MAX_DB = 0.0
    BAR_WIDTH = 132
    _GAP = 6
    _TIP_INTERVAL_S = 0.25

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._level: Optional[float] = None
        self._noise: Optional[float] = None
        self._threshold: Optional[float] = None
        self._state = "none"
        self._auto = False
        self._dragging = False
        self._tip_due = 0.0
        self._font = mono_font(9.0)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Signal level and squelch")
        self._update_tooltip(force=True)
        # (apply_theme repaints every widget too; this covers a theme
        # switch while the meter is hidden.)
        theme_notifier().theme_changed.connect(self._on_theme_changed)

    # -- reading ------------------------------------------------------------

    def set_reading(
        self,
        level_db: Optional[float],
        noise_db: Optional[float] = None,
        threshold_db: Optional[float] = None,
        state: str = "none",
        auto: bool = False,
    ) -> None:
        """Show one reading (any value None: not shown)."""
        self._level = level_db
        self._noise = noise_db
        if not self._dragging:
            self._threshold = threshold_db
        self._state = state if state in STATES else "none"
        self._auto = bool(auto)
        self._update_tooltip()
        self.update()

    def clear(self) -> None:
        """No reading (not receiving)."""
        self.set_reading(None, None, self._threshold, "none", self._auto)

    def level(self) -> Optional[float]:
        return self._level

    def noise_floor(self) -> Optional[float]:
        return self._noise

    def threshold(self) -> Optional[float]:
        return self._threshold

    def state(self) -> str:
        return self._state

    def text(self) -> str:
        """The readout, e.g. ``"-46.2 dBFS"`` (``"— dBFS"`` without one)."""
        return _format_db(self._level)

    def setText(self, text: str) -> None:  # noqa: N802 (QLabel-like)
        """Set the readout from text like ``"-46.2 dBFS"``; anything else
        clears it."""
        try:
            value = float(str(text).split()[0])
        except (ValueError, IndexError):
            self.clear()
            return
        self.set_reading(value, self._noise, self._threshold, self._state, self._auto)

    def snr_db(self) -> Optional[float]:
        if self._level is None or self._noise is None:
            return None
        return self._level - self._noise

    # -- geometry -----------------------------------------------------------

    def _text_width(self) -> int:
        return QFontMetrics(self._font).horizontalAdvance("-120.0 dBFS") + 2

    def _lamp_width(self) -> int:
        return QFontMetrics(self._font).horizontalAdvance("OPEN") + 12

    def sizeHint(self) -> "QSize":  # noqa: N802 (Qt API)
        height = max(22, QFontMetrics(self._font).height() + 8)
        width = self.BAR_WIDTH + self._text_width() + self._lamp_width() + 2 * self._GAP
        return QSize(width, height)

    def minimumSizeHint(self) -> "QSize":  # noqa: N802 (Qt API)
        return self.sizeHint()

    def _bar_rect(self) -> "QRectF":
        h = self.height()
        bar_h = max(8.0, min(14.0, h - 8.0))
        return QRectF(1.0, (h - bar_h) / 2.0, float(self.BAR_WIDTH) - 2.0, bar_h)

    def _x_for(self, db: float) -> float:
        bar = self._bar_rect()
        frac = (db - self.MIN_DB) / (self.MAX_DB - self.MIN_DB)
        return bar.left() + min(max(frac, 0.0), 1.0) * bar.width()

    def db_at(self, x: float) -> float:
        """The level (dBFS) at canvas ``x`` on the bar."""
        bar = self._bar_rect()
        frac = (x - bar.left()) / max(bar.width(), 1.0)
        frac = min(max(frac, 0.0), 1.0)
        return self.MIN_DB + frac * (self.MAX_DB - self.MIN_DB)

    def _on_bar(self, x: float) -> bool:
        return x <= self.BAR_WIDTH + 2

    # -- mouse: drag the threshold -------------------------------------------

    def mousePressEvent(self, event):  # noqa: N802 (Qt API)
        x = event.position().x()
        if event.button() == Qt.MouseButton.LeftButton and self._on_bar(x):
            self._dragging = True
            self._move_threshold(x)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt API)
        x = event.position().x()
        if self._dragging:
            self._move_threshold(x)
            event.accept()
            return
        cursor = (
            Qt.CursorShape.SizeHorCursor
            if self._on_bar(x)
            else Qt.CursorShape.ArrowCursor
        )
        if self.cursor().shape() != cursor:
            self.setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt API)
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._move_threshold(event.position().x())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _move_threshold(self, x: float) -> None:
        db = float(round(self.db_at(x)))
        if db != self._threshold:
            self._threshold = db
            self._auto = False
            self._update_tooltip(force=True)
            self.update()
            self.threshold_changed.emit(db)

    # -- tooltip ------------------------------------------------------------

    def tooltip_text(self) -> str:
        lines = [f"Level of the tuned channel: {_format_db(self._level)}"]
        if self._noise is not None:
            snr = self.snr_db()
            noise = f"Noise floor: {_format_db(self._noise)}"
            if snr is not None:
                noise += f" (SNR {snr:.1f} dB)"
            lines.append(noise)
        lamp, meaning = STATES.get(self._state, STATES["none"])
        if self._state in ("open", "closed") and self._threshold is not None:
            how = "Auto" if self._auto else "set by hand"
            lines.append(f"Squelch: {_format_db(self._threshold)} ({how}), {meaning}")
        else:
            lines.append(f"Squelch: {meaning}")
        lines.append(
            "Drag on the bar to set the squelch threshold (the marker); the "
            "tick shows the noise floor."
        )
        return "\n".join(lines)

    def _update_tooltip(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._tip_due:
            return
        self._tip_due = now + self._TIP_INTERVAL_S
        tip = self.tooltip_text()
        if tip != self.toolTip():
            self.setToolTip(tip)
            self.setAccessibleDescription(tip.split("\n")[0])

    # -- painting -----------------------------------------------------------

    def _on_theme_changed(self, _name: str) -> None:
        try:
            self.update()
        except RuntimeError:  # pragma: no cover - widget already deleted
            pass

    def _colors(self) -> Tuple[str, str, str, str]:
        """(bar fill, readout text, lamp fill, lamp text) tokens."""
        state = self._state
        if state == "open":
            return "success", "lcd_text", "success_bg", "success"
        if state == "off":
            return "accent", "text", "surface0", "subtext"
        if state == "closed":
            return "surface2", "subtext", "surface0", "subtext"
        return "surface1", "subtext", "surface0", "subtext"

    def paintEvent(self, _event):  # noqa: N802 (Qt API)
        p = get_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill, text_color, lamp_fill, lamp_text = self._colors()
        bar = self._bar_rect()

        # Trough
        painter.setPen(QPen(p.qcolor("lcd_border"), 1.0))
        painter.setBrush(p.qcolor("lcd_bg"))
        painter.drawRoundedRect(bar, 3.0, 3.0)

        # Level
        if self._level is not None:
            x = self._x_for(self._level)
            if x > bar.left() + 1:
                level_rect = QRectF(bar.left(), bar.top(), x - bar.left(), bar.height())
                path = QPainterPath()
                path.addRoundedRect(level_rect.adjusted(1, 1, 0, -1), 2.0, 2.0)
                painter.fillPath(path, p.qcolor(fill))

        # Noise floor tick (below the bar's middle)
        if self._noise is not None:
            x = round(self._x_for(self._noise)) + 0.5
            painter.setPen(QPen(p.qcolor("plot_axis"), 1.0))
            painter.drawLine(QPointF(x, bar.center().y()), QPointF(x, bar.bottom() - 1))

        # Threshold marker (hidden while the squelch can't act)
        if self._threshold is not None and self._state not in ("off", "idle"):
            x = round(self._x_for(self._threshold)) + 0.5
            marker = p.qcolor("plot_marker")
            pen = QPen(marker, 2.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, bar.top() - 1), QPointF(x, bar.bottom() + 1))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(marker)
            top = bar.top() - 1
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 4.0, top - 4.0),
                        QPointF(x + 4.0, top - 4.0),
                        QPointF(x, top + 1.0),
                    ]
                )
            )

        # Readout
        painter.setFont(self._font)
        left = self.BAR_WIDTH + self._GAP
        text_rect = QRectF(left, 0, self._text_width(), self.height())
        painter.setPen(p.qcolor(text_color))
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self.text(),
        )

        # Lamp
        lamp_label = STATES.get(self._state, STATES["none"])[0]
        lamp = QRectF(
            text_rect.right() + self._GAP,
            (self.height() - (QFontMetrics(self._font).height() + 4)) / 2.0,
            self._lamp_width(),
            QFontMetrics(self._font).height() + 4,
        )
        painter.setPen(QPen(p.qcolor("lcd_border"), 1.0))
        painter.setBrush(p.qcolor(lamp_fill))
        painter.drawRoundedRect(lamp, lamp.height() / 2.0, lamp.height() / 2.0)
        painter.setPen(p.qcolor(lamp_text))
        painter.drawText(lamp, int(Qt.AlignmentFlag.AlignCenter), lamp_label)
        painter.end()
