"""
The receiver bar and the VFO-style FREQ readout.

:class:`ReceiverBar` is the second toolbar row: MODE, FILTER and STEP. MODE
and FILTER mirror the control panel's Demodulation > Mode and Receiver >
Bandwidth combo boxes both ways (the panel stays the source of truth), and
FILTER copies the Bandwidth combo's items whenever they change, so its
choices follow the mode wherever the panel's do. STEP is the tuning step:
what Left/Right, the mouse wheel over the plots and the Frequency field's
arrows tune by.

:class:`FrequencyDigits` turns the toolbar's FREQ readout (a plain QLabel
showing :func:`format_frequency` text) into a VFO display: it keeps one
width whatever the frequency, and scrolling over a digit tunes that digit.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, QPointF, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QPainter, QPen
    from PyQt6.QtWidgets import QComboBox, QLabel, QToolBar

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - no Qt
    HAS_PYQT6 = False

from .themes import get_palette, set_role
from .tuning import STEP_CHOICES_HZ, format_step, step_index

# ---------------------------------------------------------------------------
# FREQ readout
# ---------------------------------------------------------------------------

#: The widest text the readout shows (format_frequency: up to 6 decimals).
WIDEST_FREQUENCY_TEXT = "0000.000000 MHz"

#: While scrolling, the digit being tuned stays the same for this long after
#: the last notch, even if the text (and so the digit under the pointer)
#: shifts as trailing zeros come and go.
_DIGIT_LOCK_S = 0.6


class FrequencyDigits(QObject if HAS_PYQT6 else object):
    """VFO behaviour for a right-aligned frequency QLabel.

    * A fixed width: wide enough for :data:`WIDEST_FREQUENCY_TEXT`, so the
      toolbar after it never shifts while tuning (re-measured when the
      theme's font or padding changes).
    * Scroll over a digit to tune it: one notch is one unit of that digit
      (with carries), up for higher; :attr:`digit_tuned` gives the change in
      Hz. The digit under the pointer is underlined.

    Clicks are left to the label's owner (the window opens the Frequency
    field on a click).
    """

    if HAS_PYQT6:
        #: Hz to add to the tuned frequency (a scroll over a digit).
        digit_tuned = pyqtSignal(float)

    def __init__(self, label: "QLabel"):
        super().__init__(label)
        self._label = label
        self._hover_x: Optional[float] = None
        self._wheel_accum = 0
        self._locked: Optional[Tuple[float, float, float]] = None  # place, x, until
        label.setMouseTracking(True)
        label.installEventFilter(self)
        self.fit_width()

    # -- geometry -----------------------------------------------------------
    def fit_width(self) -> int:
        """Give the label the width of the widest frequency text; returns it
        (0 when the label is gone)."""
        label = self._label
        try:
            label.ensurePolished()
        except RuntimeError:  # deleted before a deferred refit ran
            return 0
        width = max(1, self._natural_width(WIDEST_FREQUENCY_TEXT))
        if label.minimumWidth() != width or label.maximumWidth() != width:
            label.setFixedWidth(width)
        return width

    def _natural_width(self, text: str) -> int:
        """The width the label asks for to show ``text``, worked out as
        QLabel's size hint does. (The hint itself can't be used: it never
        reports less than the label's minimum width, so once the width was
        fixed every refit, e.g. on a theme switch, made the label wider.)"""
        label = self._label
        fm = label.fontMetrics()
        flags = int(label.alignment().value) & ~int(
            (Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter).value
        )
        extra = 2 * label.margin()
        indent = label.indent()
        if indent < 0 and label.frameWidth() > 0:
            # QLabel's default indent inside a frame, as in its size hint.
            indent = fm.horizontalAdvance("x") - 2 * label.margin()
        if indent > 0:
            extra += indent
        margins = label.contentsMargins()  # frame: border and padding
        text_width = fm.boundingRect(0, 0, 2000, 2000, flags, text).width()
        return text_width + extra + margins.left() + margins.right()

    def _text_left(self) -> float:
        """Where the (right-aligned) text starts, in label coordinates."""
        label = self._label
        fm = label.fontMetrics()
        rect = label.contentsRect()
        indent = label.indent()
        if indent < 0 and label.frameWidth() > 0:
            # QLabel's own default: half an "x" inside a frame.
            indent = fm.horizontalAdvance("x") // 2 - label.margin()
        right = rect.right() + 1 - label.margin() - max(0, indent)
        return float(right - fm.horizontalAdvance(label.text()))

    def char_spans(self) -> List[Tuple[float, float]]:
        """``(left, right)`` x of each character of the label's text."""
        text = self._label.text()
        fm = self._label.fontMetrics()
        left = self._text_left()
        spans = []
        for i in range(len(text)):
            spans.append(
                (
                    left + fm.horizontalAdvance(text[:i]),
                    left + fm.horizontalAdvance(text[: i + 1]),
                )
            )
        return spans

    def digit_at(self, x: float) -> Optional[Tuple[int, float]]:
        """``(character index, Hz per unit)`` of the digit at ``x``."""
        text = self._label.text()
        number = text.split(" ", 1)[0]
        point = number.find(".")
        if point < 0:
            point = len(number)
        for i, (x0, x1) in enumerate(self.char_spans()):
            if i >= len(number):
                break
            if x0 <= x < x1 and number[i].isdigit():
                exponent = point - 1 - i if i < point else point - i
                return i, (10.0**exponent) * 1e6
        return None

    # -- events -------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if obj is not self._label:
            return False
        kind = event.type()
        if kind == QEvent.Type.Wheel:
            return self._wheel(event)
        if kind == QEvent.Type.MouseMove:
            self._set_hover(event.position().x())
        elif kind == QEvent.Type.Leave:
            self._set_hover(None)
        elif kind == QEvent.Type.Paint and self._hover_x is not None:
            # The label's own painting, then the underline on top.
            QLabel.paintEvent(self._label, event)
            self._paint_underline()
            return True
        elif kind in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            # The label takes the new font after this filter: measure then.
            QTimer.singleShot(0, self.fit_width)
        return False

    def _set_hover(self, x: Optional[float]) -> None:
        if x != self._hover_x:
            self._hover_x = x
            self._label.update()

    def _wheel(self, event) -> bool:
        delta = event.angleDelta().y() or event.angleDelta().x()
        x = event.position().x()
        now = time.monotonic()
        place: Optional[float] = None
        locked = self._locked
        if locked is not None and now < locked[2] and abs(x - locked[1]) <= 4:
            place = locked[0]  # still the same scroll: the same digit
        else:
            found = self.digit_at(x)
            place = found[1] if found else None
        if place is None:
            self._wheel_accum = 0
            event.ignore()
            return False
        self._wheel_accum += delta
        notches = int(self._wheel_accum / 120)
        self._locked = (place, x, now + _DIGIT_LOCK_S)
        if notches:
            self._wheel_accum -= notches * 120
            self.digit_tuned.emit(notches * place)
        event.accept()
        return True

    def _paint_underline(self) -> None:
        label = self._label
        if self._hover_x is None:
            return
        found = self.digit_at(self._hover_x)
        if found is None:
            return
        x0, x1 = self.char_spans()[found[0]]
        fm = label.fontMetrics()
        rect = label.contentsRect()
        baseline = rect.top() + (rect.height() - fm.height()) / 2 + fm.ascent()
        y = min(baseline + 3, label.height() - 3)
        painter = QPainter(label)
        try:
            pen = QPen(get_palette().qcolor("accent"), 2)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            # Letter spacing pads the right of each glyph: underline the
            # glyph itself.
            painter.drawLine(QPointF(x0 + 1, y), QPointF(x1 - 2, y))
        finally:
            painter.end()


# ---------------------------------------------------------------------------
# Receiver bar
# ---------------------------------------------------------------------------


class ReceiverBar(QToolBar if HAS_PYQT6 else object):
    """MODE, FILTER and STEP in a toolbar row of their own.

    ``mode_source`` and ``filter_source`` are the control panel's Mode and
    Bandwidth combo boxes: picking in the bar picks there (so every handler
    runs as usual), and changes there show here. :attr:`step_changed` says
    the user picked a step; :meth:`set_step` shows the step in use.
    """

    if HAS_PYQT6:
        #: The user picked a tuning step (Hz).
        step_changed = pyqtSignal(float)

    MODE_TIP = (
        "Demodulation mode: how the tuned signal becomes audio (the same "
        "choice as Demodulation > Mode in the control panel)"
    )
    FILTER_TIP = (
        "Channel filter: the width that is demodulated and measured around "
        "the tuned frequency (the same as Receiver > Filter)"
    )
    STEP_TIP = (
        "Tuning step: Left/Right, the mouse wheel over the spectrum or "
        "waterfall and the Frequency field's arrows tune by it (Shift: ten "
        "steps). [ and ] change it while the plots have the focus. Each mode "
        "remembers its own step."
    )

    def __init__(
        self,
        mode_source: "QComboBox",
        filter_source: "QComboBox",
        parent=None,
    ):
        super().__init__("Receiver", parent)
        self.setObjectName("receiverToolbar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

        self._mode_source = mode_source
        self._filter_source = filter_source
        # id(source combo) -> the model whose item changes are copied.
        self._watched_models: Dict[int, object] = {}
        self._mode_combo = self._mirror(
            mode_source, "MODE", "Demodulation mode", self.MODE_TIP
        )
        self.addSeparator()
        self._filter_combo = self._mirror(
            filter_source, "FILTER", "Channel filter", self.FILTER_TIP
        )
        self.addSeparator()

        self._step_combo = QComboBox()
        for step in STEP_CHOICES_HZ:
            self._step_combo.addItem(format_step(step), step)
        self._step_combo.setCurrentIndex(step_index(10e3))
        self._setup_combo(self._step_combo, "STEP", "Tuning step", self.STEP_TIP)
        self._step_combo.currentIndexChanged.connect(self._on_step_index_changed)

    # -- construction ---------------------------------------------------------
    def _setup_combo(self, combo: "QComboBox", caption: str, name: str, tip: str):
        label = QLabel(caption)
        set_role(label, "caption")
        label.setToolTip(tip)
        label.setBuddy(combo)
        combo.setToolTip(tip)
        combo.setAccessibleName(name)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        # Reachable with Tab, but a click doesn't take the keyboard focus from
        # the plots, where Space and the arrow keys act (like the plots'
        # header controls).
        combo.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.addWidget(label)
        self.addWidget(combo)
        return combo

    def _mirror(
        self, source: "QComboBox", caption: str, name: str, tip: str
    ) -> "QComboBox":
        combo = QComboBox()
        self._setup_combo(combo, caption, name, tip)
        self._copy_items(source, combo)
        combo.currentIndexChanged.connect(
            lambda index, s=source: self._to_source(s, index)
        )
        source.currentIndexChanged.connect(
            lambda _index, s=source, c=combo: self._from_source(s, c)
        )
        self._watch_model(source, combo)
        return combo

    def _watch_model(self, source: "QComboBox", combo: "QComboBox") -> None:
        """Copy ``source``'s items into ``combo`` whenever its model changes
        them (once per model: a combo can be given a new one)."""
        model = source.model()
        if self._watched_models.get(id(source)) is model:
            return
        self._watched_models[id(source)] = model
        for signal in (
            model.rowsInserted,
            model.rowsRemoved,
            model.rowsMoved,
            model.modelReset,
            model.layoutChanged,
            model.dataChanged,
        ):
            signal.connect(lambda *_args, s=source, c=combo: self._copy_items(s, c))

    # -- mirroring ------------------------------------------------------------
    @staticmethod
    def _copy_items(source: "QComboBox", combo: "QComboBox") -> None:
        """Make ``combo``'s items (text, tooltips) and choice ``source``'s."""
        items = [
            (source.itemText(i), source.itemData(i, Qt.ItemDataRole.ToolTipRole))
            for i in range(source.count())
        ]
        current = [
            (combo.itemText(i), combo.itemData(i, Qt.ItemDataRole.ToolTipRole))
            for i in range(combo.count())
        ]
        blocked = combo.blockSignals(True)
        try:
            if items != current:
                combo.clear()
                for i, (text, tip) in enumerate(items):
                    combo.addItem(text)
                    if tip:
                        combo.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
            combo.setCurrentIndex(source.currentIndex())
        finally:
            combo.blockSignals(blocked)

    @staticmethod
    def _to_source(source: "QComboBox", index: int) -> None:
        if 0 <= index < source.count() and index != source.currentIndex():
            source.setCurrentIndex(index)  # the panel's handlers run as usual

    def _from_source(self, source: "QComboBox", combo: "QComboBox") -> None:
        # Compare the items too, and watch the model: a combo given a new
        # model (setModel, which reports itself with currentIndexChanged) no
        # longer reports its item changes to the old model's signals.
        self._watch_model(source, combo)
        self._copy_items(source, combo)

    # -- step -------------------------------------------------------------------
    def _on_step_index_changed(self, index: int) -> None:
        step = self._step_combo.itemData(index)
        if step:
            self.step_changed.emit(float(step))

    def set_step(self, step_hz: float) -> None:
        """Show ``step_hz`` (or the nearest choice) without emitting."""
        index = step_index(step_hz)
        if index != self._step_combo.currentIndex():
            blocked = self._step_combo.blockSignals(True)
            self._step_combo.setCurrentIndex(index)
            self._step_combo.blockSignals(blocked)

    def step(self) -> float:
        """The step shown, in Hz."""
        return float(self._step_combo.currentData() or 10e3)

    @property
    def mode_combo(self) -> "QComboBox":
        return self._mode_combo

    @property
    def filter_combo(self) -> "QComboBox":
        return self._filter_combo

    @property
    def step_combo(self) -> "QComboBox":
        return self._step_combo
