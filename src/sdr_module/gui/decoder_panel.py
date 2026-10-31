"""
Protocol decoder output panel.

Shows what the live protocol decoder produces: a message table, a
plain-text log that is easy to copy, and running statistics; for ADS-B an
Aircraft view (one row per aircraft) and for Morse code a Text view with
the speed. "Auto" picks the protocol from where the receiver is tuned.
"""

from __future__ import annotations

import csv
import html
import math
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QBrush,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QKeySequence,
        QPalette,
        QTextBlockFormat,
        QTextCharFormat,
        QTextCursor,
        QTextLayout,
        QTextOption,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMenu,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QStyledItemDelegate,
        QTabBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .themes import get_palette, set_role, set_tone


def _format_mhz(freq_hz: float) -> str:
    """``1090e6 -> "1090.000 MHz"`` (3 to 6 decimals, like other readouts)."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


#: Selector entry that runs no decoder. The main window maps any name it
#: does not know (this one included) to "no live decoder".
PROTOCOL_OFF = "Off"

#: Selector entry that picks the protocol from the tuned frequency and mode
#: (see :func:`auto_protocol`); the panel then reports that protocol.
PROTOCOL_AUTO = "Auto"

#: Protocol name -> (what it carries, where to listen). The names are the
#: ones the main window maps to decoder implementations (and the keys of
#: ``sdr_module.dsp.live_decoders.LIVE_DECODERS``); keep them in step.
PROTOCOL_INFO: Dict[str, Tuple[str, str]] = {
    "POCSAG": (
        "Pager messages (512/1200/2400 baud)",
        "152–159 MHz or 929–932 MHz",
    ),
    "FLEX": ("Motorola FLEX pager messages (1600 bps)", "929–932 MHz"),
    "AX.25/APRS": (
        "Packet radio and APRS position reports",
        "144.390 MHz (N. America) or 144.800 MHz (Europe)",
    ),
    "ADS-B": ("Aircraft identity, altitude and position", "1090 MHz"),
    "ACARS": ("Aircraft data-link text messages (AM)", "129–137 MHz"),
    "RDS": ("Station name and radio text from FM broadcasts", "88–108 MHz"),
    "CW (Morse)": (
        "Morse code as text, with its speed",
        "a CW signal tuned in CW mode",
    ),
}

#: Protocol name -> (frequency in Hz, demod mode, FM deviation, what it is)
#: offered as a one-click "Tune to ..." in the empty message table. The
#: mode is what listening to it sounds best in (the decoder itself picks its
#: own channel): no audio for ADS-B's pulses, narrowband FM (5 kHz, a 25 kHz
#: channel) for pagers and packet, AM (10 kHz) for ACARS. Pagers use many
#: channels across their bands; theirs is a common one (the demo device's).
PROTOCOL_TUNE: Dict[str, Tuple[float, str, str, str]] = {
    "POCSAG": (
        152.24e6,
        "FM",
        "5 kHz",
        "a common VHF pager channel (pagers use many channels in 152–159 MHz "
        "and 929–932 MHz)",
    ),
    "FLEX": (
        931.9375e6,
        "FM",
        "5 kHz",
        "a common 900 MHz pager channel (FLEX pagers use 929–932 MHz)",
    ),
    "ADS-B": (1090e6, "None (I/Q)", "", "the ADS-B frequency used worldwide"),
    "AX.25/APRS": (
        144.39e6,
        "FM",
        "5 kHz",
        "the North American APRS frequency (Europe: 144.800 MHz)",
    ),
    "ACARS": (131.55e6, "AM", "", "the primary ACARS channel"),
    "RDS": (
        100.1e6,
        "FM",
        "75 kHz",
        "an FM broadcast station (any station that sends RDS will do)",
    ),
}

#: Protocol name -> tuning ranges (Hz) in which the receiver counts as
#: listening for it: around its channel, or anywhere in its band.
PROTOCOL_BANDS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "POCSAG": ((152e6, 159e6), (929e6, 932e6)),
    "FLEX": ((929e6, 932e6),),
    "AX.25/APRS": ((144.3775e6, 144.4025e6), (144.7875e6, 144.8125e6)),
    "ADS-B": ((1089e6, 1091e6),),
    "ACARS": ((129e6, 137e6),),
    "RDS": ((87.5e6, 108e6),),
}

#: Protocols listed but not decodable from live signals yet (shown disabled).
PROTOCOL_UNAVAILABLE: Dict[str, str] = {}

#: Empty-table hint per protocol whose signal is not at one frequency.
PROTOCOL_HINTS: Dict[str, str] = {
    "CW (Morse)": (
        "Tune to a CW signal in CW mode (the demo device has a beacon on 99.820 MHz)"
    ),
}

#: What the status line says when a protocol's decoder hears a signal it
#: can't decode for a while (default: check the frequency and protocol).
PROTOCOL_UNDECODED: Dict[str, str] = {
    "RDS": "No RDS from this station: not every station sends it",
}

#: What "Auto" picks, in order (the first that matches wins): Morse code
#: in CW mode, then by the tuned frequency.
_AUTO_BANDS: Tuple[Tuple[str, Tuple[Tuple[float, float], ...]], ...] = (
    ("ADS-B", ((1089e6, 1091e6),)),
    ("AX.25/APRS", ((144.3775e6, 144.4025e6), (144.7875e6, 144.8125e6))),
    ("ACARS", ((129e6, 137e6),)),
    ("POCSAG", ((152e6, 159e6),)),
    ("FLEX", ((929e6, 932e6),)),
    ("RDS", ((87.5e6, 108e6),)),
)
_AUTO_TIP = (
    "Pick the protocol from where the receiver is tuned: ADS-B on 1090 MHz, "
    "APRS on 144.390 or 144.800 MHz, ACARS on 129–137 MHz, POCSAG on "
    "152–159 MHz, FLEX on 929–932 MHz, RDS on FM broadcast (88–108 MHz), "
    "and Morse code in CW mode."
)


def auto_protocol(
    freq_hz: Optional[float], mode: Optional[str] = None
) -> Optional[str]:
    """The protocol "Auto" decodes when tuned to ``freq_hz`` in demod
    ``mode`` (None: nothing usually decodes there)."""
    if mode == "CW":
        return "CW (Morse)"
    if freq_hz is None:
        return None
    for name, bands in _AUTO_BANDS:
        if any(lo <= freq_hz <= hi for lo, hi in bands):
            return name
    return None


_START_HINT = "press Start (Space)"
# Shown for a value that isn't there (no message text, no last message).
_NO_VALUE = "–"
# Seconds of signal in the decoder's channel without a valid message before
# the status line says that something is there but does not decode.
_UNDECODED_SIGNAL_S = 6.0
# A last message older than this is no longer shown as good news.
_FRESH_S = 120.0
# Decoder units that are one table row each; a decoder counting something
# else (RDS groups, Morse characters) has it shown after its messages.
_ROW_UNITS = ("message", "page", "packet")


def _plural(count: int, unit: str) -> str:
    """``3, "page" -> "3 pages"``."""
    return f"{count:,} {unit}{'' if count == 1 else 's'}"


def _format_age(seconds: float) -> str:
    """``4.2 -> "4 s ago"``, ``130 -> "2 min ago"``."""
    seconds = max(0.0, float(seconds))
    if seconds < 2:
        return "just now"
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.0f} h ago"


# Table columns.
COL_TIME, COL_PROTOCOL, COL_ADDRESS, COL_MESSAGE = range(4)
_HEADERS = ("Time", "Protocol", "Address", "Message")
# Widest a size-to-contents column may grow before it elides.
_MAX_COL_WIDTH = {COL_PROTOCOL: 110, COL_ADDRESS: 170}
# Share of the table width the Message column keeps in a narrow panel: the
# Address column gives way first (down to _MIN_ADDRESS_WIDTH, eliding).
_MESSAGE_SHARE = 0.4
_MIN_MESSAGE_WIDTH = 120
_MIN_ADDRESS_WIDTH = 64
# Horizontal padding around cell text and (bold, padded) header text.
_CELL_PAD = 14
_HEADER_PAD = 20
# Lines of the selected message shown under the table before it elides.
_DETAIL_LINES = 3

# Item data role flagging a row whose message failed its integrity checks.
_INVALID_ROLE = Qt.ItemDataRole.UserRole.value + 1 if HAS_PYQT6 else 0


def placeholder_html(text: str) -> str:
    """Rich text for an empty-state message ``"Title\\nHint ..."``.

    The first line is the title, shown bold and upright like the painted
    placeholders of the plots ("No spectrum yet"); the lines after it are
    the hint. Titles are short phrases without a final period. A single
    line is shown as it is. Everything is escaped: the text is never markup.
    """
    title, sep, hint = text.partition("\n")
    if not sep:
        return html.escape(text)
    hint_html = "<br>".join(html.escape(line) for line in hint.split("\n"))
    return (
        '<p style="margin: 0 0 4px 0; font-weight: 600; font-style: normal;">'
        f'{html.escape(title)}</p><p style="margin: 0;">{hint_html}</p>'
    )


class ViewPlaceholder(QObject if HAS_PYQT6 else object):
    """Centered empty-state text laid over an item view's viewport.

    The label uses the ``placeholder`` role, lets clicks through and follows
    the viewport size. The text's first line is a title over the hint lines
    (see :func:`placeholder_html`). Call :meth:`set_text` and
    :meth:`set_visible` to update it. :meth:`set_action` adds an optional
    button under the text (e.g. "Tune to 1090.000 MHz"); clicking it emits
    :attr:`action_triggered`.
    """

    if HAS_PYQT6:
        action_triggered = pyqtSignal()

    _ACTION_GAP = 4

    def __init__(self, view: Any, text: str = ""):
        super().__init__(view)
        viewport = view.viewport()
        self._viewport = viewport
        self._text = text
        self.label = QLabel(placeholder_html(text), viewport)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        set_role(self.label, "placeholder")
        self.button = QPushButton(viewport)
        set_role(self.button, "primary")
        self.button.setAutoDefault(False)
        self.button.clicked.connect(lambda _checked=False: self.action_triggered.emit())
        self.button.hide()
        self._has_action = False
        self._layout()
        viewport.installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Resize:
            self._layout()
        return False

    def _layout(self) -> None:
        """Center the text (and the action button under it) in the view.

        When the view is too short for the text and the button, only the
        title is shown with the button, so the button is never cut off.
        """
        rect = self._viewport.rect()
        full = placeholder_html(self._text)
        if not self._has_action:
            if self.label.text() != full:
                self.label.setText(full)
            self.label.setGeometry(rect)
            return
        hint = self.button.sizeHint()
        width = rect.width()
        room = rect.height() - self._ACTION_GAP - hint.height()
        if self.label.text() != full:
            self.label.setText(full)
        title, sep, _hint = self._text.partition("\n")
        if sep and self.label.heightForWidth(width) > room:
            self.label.setText(
                '<p style="margin: 0; font-weight: 600; font-style: normal;">'
                f"{html.escape(title)}</p>"
            )
        text_h = max(0, min(self.label.heightForWidth(width), room))
        total = text_h + self._ACTION_GAP + hint.height()
        top = max(0, (rect.height() - total) // 2)
        self.label.setGeometry(0, top, width, text_h)
        button_w = min(hint.width(), max(width - 16, 0))
        self.button.setGeometry(
            (width - button_w) // 2,
            top + text_h + self._ACTION_GAP,
            button_w,
            hint.height(),
        )

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.label.setText(placeholder_html(text))
            self._layout()

    def set_action(self, text: Optional[str], tip: str = "") -> None:
        """Show a button labelled ``text`` under the message (None: none)."""
        self._has_action = bool(text)
        if text:
            self.button.setText(text)
            self.button.setToolTip(tip)
            self.button.setAccessibleName(text)
        self.button.setVisible(self._has_action and not self.label.isHidden())
        self._layout()

    def action_text(self) -> str:
        """Text of the action button ("" when there is none)."""
        return self.button.text() if self._has_action else ""

    def set_visible(self, visible: bool) -> None:
        self.label.setVisible(visible)
        self.button.setVisible(visible and self._has_action)

    def is_visible(self) -> bool:
        return not self.label.isHidden()

    def text(self) -> str:
        """The message as plain text (``"Title\\nHint ..."``)."""
        return self._text


class _MessageDelegate(QStyledItemDelegate if HAS_PYQT6 else object):
    """Tints rows that failed their checks, using the active theme colors."""

    def initStyleOption(self, option: Any, index: Any) -> None:  # noqa: N802
        super().initStyleOption(option, index)
        if index.data(_INVALID_ROLE):
            p = get_palette()
            option.backgroundBrush = QBrush(p.qcolor("danger_bg"))
            option.palette.setColor(QPalette.ColorRole.Text, p.qcolor("danger"))


class _MessageTable(QTableWidget if HAS_PYQT6 else object):
    """Message table: Ctrl+C copies whole rows, Esc clears the selection and
    the columns are refitted whenever the visible width changes."""

    def __init__(self, panel: "DecoderPanel"):
        super().__init__(0, len(_HEADERS))
        self._panel = panel
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        if obj is self.viewport() and event.type() == QEvent.Type.Resize:
            self._panel._fit_columns()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        if event.matches(QKeySequence.StandardKey.Copy):
            self._panel.copy_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.selectionModel().hasSelection():
            self.clearSelection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        menu = QMenu(self)
        copy_act = menu.addAction("Copy")
        copy_act.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy_act.setEnabled(bool(self.selectionModel().selectedRows()))
        select_act = menu.addAction("Select All")
        select_act.setEnabled(self.rowCount() > 0)
        chosen = menu.exec(event.globalPos())
        menu.deleteLater()  # a new menu is built for every right-click
        if chosen is copy_act:
            self._panel.copy_selected()
        elif chosen is select_act:
            self.selectAll()


def elide_lines(text: str, font: Any, width: int, max_lines: int) -> str:
    """Wrap ``text`` at ``width`` px and cut it to ``max_lines`` lines.

    Returns the text unchanged when it fits; otherwise the kept lines are
    joined with newlines and the last one ends in an ellipsis, so a
    word-wrapped label shows exactly ``max_lines`` lines.
    """
    width = max(int(width), 1)
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    spans: List[Tuple[int, int]] = []
    layout.beginLayout()
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        spans.append((line.textStart(), line.textLength()))
    layout.endLayout()
    if len(spans) <= max_lines:
        return text
    kept = [text[start : start + length].rstrip() for start, length in spans]
    rest = " ".join(text[spans[max_lines - 1][0] :].split())
    last = QFontMetrics(font).elidedText(rest, Qt.TextElideMode.ElideRight, width)
    return "\n".join(kept[: max_lines - 1] + [last])


class _MessageDetail(QFrame if HAS_PYQT6 else object):
    """Card under the table showing the selected message in full.

    Long messages are cut to a few lines (with an ellipsis) so the card
    never crowds out the table; the Log tab always has the complete text.
    """

    def __init__(self, parent: Any = None):
        super().__init__(parent)
        set_role(self, "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        self.meta = QLabel()
        set_role(self.meta, "hint")
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body = QLabel()
        self.body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # More about the message (an APRS path, an ADS-B message's kind...).
        self.details = QLabel()
        set_role(self.details, "hint")
        self.details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        # All wrap and never ask for more width than the panel has. Decoded
        # text is data, never markup ("<b>" in a page stays literal).
        for label in (self.meta, self.body, self.details):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(label)
        self.details.hide()
        self._full = ""
        self.hide()

    def show_text(
        self,
        meta: str,
        body: str,
        tone: Optional[str] = None,
        details: Optional[List[str]] = None,
    ) -> None:
        """Show ``body`` under a ``meta`` line (``tone`` colors the meta),
        with ``details`` lines under it."""
        self.meta.setText(meta)
        set_tone(self.meta, tone)
        self._full = body
        self.body.setVisible(bool(body))
        text = "\n".join(details or [])
        self.details.setText(text)
        self.details.setVisible(bool(text))
        self._render()
        self.show()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._render()

    def showEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._render()

    def _render(self) -> None:
        # Measure the frame itself: the layout's own geometry is reset
        # whenever it is invalidated (e.g. right after a tone change).
        margins = self.layout().contentsMargins()
        width = self.contentsRect().width() - margins.left() - margins.right()
        shown = elide_lines(self._full, self.body.font(), width, _DETAIL_LINES)
        self.body.setText(shown)
        self.body.setToolTip(self._full if shown != self._full else "")


class TailFollower(QObject if HAS_PYQT6 else object):
    """Keeps a scroll bar pinned to its end until the user scrolls away.

    Views lay out new rows lazily (and not at all while their tab is
    hidden), so instead of scrolling once per insert this re-pins the bar
    whenever its range grows, as long as the user was at the end.
    """

    def __init__(self, bar: Any):
        super().__init__(bar)
        self._bar = bar
        self.following = True
        bar.valueChanged.connect(self._on_value)
        bar.rangeChanged.connect(self._on_range)

    def _on_value(self, value: int) -> None:
        self.following = value >= self._bar.maximum() - 2

    def _on_range(self, _minimum: int, maximum: int) -> None:
        if self.following:
            self._bar.setValue(maximum)

    def reset(self) -> None:
        """Follow the end again (e.g. after the view was cleared)."""
        self.following = True


class _CornerTabBar(QTabBar if HAS_PYQT6 else object):
    """Tab bar whose tabs are as tall as the tab widget's corner widget.

    QTabWidget squeezes a corner widget into the tab bar's height. The
    underline (document mode) tabs are shorter than a push button, so the
    Clear / Export buttons in the corner were cut off at the bottom and
    read as two more tabs. Growing the tabs instead keeps the buttons whole
    and lines their text up with the tab titles.
    """

    # Rows QTabWidget takes from the tab bar's height before sizing the
    # corner widget (the tab/pane overlap).
    _OVERLAP = 1

    def tabSizeHint(self, index: int) -> Any:  # noqa: N802 (Qt API)
        hint = super().tabSizeHint(index)
        tabs = self.parentWidget()
        corner = (
            tabs.cornerWidget(Qt.Corner.TopRightCorner)
            if isinstance(tabs, QTabWidget)
            else None
        )
        if corner is not None and not corner.isHidden():
            hint.setHeight(
                max(hint.height(), corner.sizeHint().height() + self._OVERLAP)
            )
        return hint


class DecoderPanel(QWidget if HAS_PYQT6 else object):
    """
    Protocol decoder panel.

    Displays decoded messages and allows protocol selection. The
    ``_enabled_check`` box ("Decode") pauses decoding without losing the
    selected protocol; the main window reads it before feeding samples.

    While the table is empty it says where to listen and, for protocols with
    one well-known channel, offers a "Tune to ..." button that emits
    :attr:`tune_requested`. Tell the panel the tuned frequency and whether
    the receiver runs (:meth:`set_current_frequency`, :meth:`set_receiving`)
    and it confirms when it is listening in the right place.

    ``settings`` (the window's :class:`~.settings_store.GuiSettings`) keeps
    the folder Export last saved to.
    """

    if HAS_PYQT6:
        protocol_changed = pyqtSignal(str)
        # Decode File...: run the selected protocol over a recording
        # (the main window's File > Decode Recording).
        decode_file_requested = pyqtSignal()
        # freq_hz, label, demod mode and FM deviation ("" = keep the current
        # one): the same shape as BookmarksPanel.tune_requested.
        tune_requested = pyqtSignal(float, str, str, str)
        # Decode was ticked (True) or unticked by the user.
        decode_toggled = pyqtSignal(bool)

    def __init__(self, parent=None, settings: Any = None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        # Where Export saved last (file_dialogs' "decode" folder); without
        # settings it starts in Documents/SDR Recordings every time.
        self._settings = settings
        self._messages: List[Dict[str, Any]] = []
        self._max_messages = 1000
        # Totals since the last Clear (the table keeps only the newest rows).
        self._total = 0
        self._invalid = 0
        self._per_protocol: Counter = Counter()
        # Valid messages per protocol since Clear, and their count when the
        # protocol being decoded was chosen: what it decoded since then (or
        # since Clear) is what the status line and the window's status-bar
        # chip count, alike.
        self._valid_per_protocol: Counter = Counter()
        self._chosen_counts: Counter = Counter()
        # clock() time of each protocol's last valid message.
        self._row_times: Dict[str, float] = {}
        # Widest cell text seen per size-to-contents column, in px.
        self._col_content: Dict[int, int] = {}
        # Receiver state, when the main window reports it (None: unknown),
        # and the clock() time the tuned frequency last changed.
        self._tuned_hz: Optional[float] = None
        self._tuned_at = -math.inf
        self._receiving: Optional[bool] = None
        # The user's Decode choice; the box shows unticked while Off.
        self._decode_wanted = True
        # The live decoder's health (a LiveDecoderStatus; None: no decoder)
        # and the clock its last_time is measured with.
        self._decoder_status: Any = None
        self._clock = time.monotonic
        # The decoder's (frames, valid, failed) at the last Clear: the
        # status line and Stats count from there.
        self._count_base = (0, 0, 0)
        # The demodulation mode and the sample rate, when the main window
        # reports them (Auto reads the mode; messages keep the rate).
        self._mode: Optional[str] = None
        self._sample_rate: Optional[float] = None
        # What Auto chose (None: nothing to decode where it is tuned).
        self._auto_choice: Optional[str] = None
        # Messages shows only this address (an aircraft picked under
        # Aircraft), or "" for all, and how many of its messages are there.
        self._filter_address = ""
        self._filter_count = 0
        # The "Tune to ..." on offer (text, tooltip; None: none).
        self._tune_offer: Tuple[Optional[str], str] = (None, "")

        self._setup_ui()
        self._update_views()  # the protocol views hide until chosen
        self._update_state()
        self._update_stats()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Protocol selector and the decode on/off switch, side by side.
        proto_row = QHBoxLayout()
        proto_row.setSpacing(8)
        proto_label = QLabel("Protocol:")
        proto_row.addWidget(proto_label)

        self._proto_combo = QComboBox()
        self._proto_combo.addItem(PROTOCOL_OFF)
        self._proto_combo.setItemData(
            0,
            "No decoder. Choose a protocol to start decoding.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._proto_combo.addItem(PROTOCOL_AUTO)
        self._proto_combo.setItemData(1, _AUTO_TIP, Qt.ItemDataRole.ToolTipRole)
        for name, (what, where) in PROTOCOL_INFO.items():
            self._proto_combo.addItem(name)
            index = self._proto_combo.count() - 1
            tip = f"{what}. Usually {where}."
            unavailable = PROTOCOL_UNAVAILABLE.get(name)
            if unavailable:
                tip = f"{what}. {unavailable}"
                item = self._proto_combo.model().item(index)
                if item is not None:
                    item.setEnabled(False)
            self._proto_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._proto_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._proto_combo.setMaximumWidth(260)
        self._proto_combo.setAccessibleName("Decoder protocol")
        proto_label.setBuddy(self._proto_combo)
        self._proto_combo.currentTextChanged.connect(self._on_protocol_changed)
        proto_row.addWidget(self._proto_combo, 1)

        self._enabled_check = QCheckBox("Decode")
        self._enabled_check.setChecked(False)  # ticked once a protocol is chosen
        self._enabled_check.setAccessibleName("Decode messages")
        self._enabled_check.toggled.connect(self._on_decode_toggled)
        proto_row.addWidget(self._enabled_check)
        proto_row.addStretch(0)
        # Decode a recording instead of the live signal (File > Decode
        # Recording).
        self._decode_file_btn = QPushButton("Decode File...")
        self._decode_file_btn.setAutoDefault(False)
        self._decode_file_btn.setToolTip(
            "Decode a recorded I/Q file with a protocol, much faster than real "
            "time; the messages are listed here (File > Decode Recording)"
        )
        self._decode_file_btn.clicked.connect(
            lambda _checked=False: self.decode_file_requested.emit()
        )
        proto_row.addWidget(self._decode_file_btn)
        layout.addLayout(proto_row)

        # One line of decoder health: counts, signal, problems.
        self._status_label = QLabel()
        set_role(self._status_label, "hint")
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setWordWrap(True)
        self._status_label.setAccessibleName("Decoder status")
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self._status_label, 1)
        # "Tune to ..." beside it whenever the receiver is tuned elsewhere:
        # the empty table's own button goes once messages are listed.
        self._tune_btn = QPushButton()
        set_role(self._tune_btn, "compact")
        self._tune_btn.setAutoDefault(False)
        self._tune_btn.clicked.connect(lambda _checked=False: self._tune_to_protocol())
        self._tune_btn.hide()
        status_row.addWidget(self._tune_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(status_row)

        # Output tabs, with Clear / Export in the tab bar's corner so the
        # table keeps as much height as possible in the short right column.
        self._tabs = QTabWidget()
        self._tabs.setTabBar(_CornerTabBar(self._tabs))
        self._tabs.setDocumentMode(True)
        # No base line: it drew an empty tab-like outline between the last
        # tab and the corner buttons.
        self._tabs.tabBar().setDrawBase(False)

        # Regular push buttons, the same size as every other panel action.
        # The corner is set before any tab is added, so the tabs are sized
        # to fit it from the start (see _CornerTabBar).
        corner = QWidget()
        corner_row = QHBoxLayout(corner)
        corner_row.setContentsMargins(0, 0, 0, 2)
        corner_row.setSpacing(4)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(lambda _checked=False: self.clear())
        corner_row.addWidget(self._clear_btn)
        self._export_btn = QPushButton("Export...")
        self._export_btn.clicked.connect(lambda _checked=False: self._export_messages())
        corner_row.addWidget(self._export_btn)
        for button in (self._clear_btn, self._export_btn):
            button.setAutoDefault(False)
        self._tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        # Their width as regular buttons (they turn compact when a narrow
        # panel would otherwise hide a tab behind scroll arrows).
        self._corner_width = corner.sizeHint().width()

        self._table = _MessageTable(self)
        self._table.setAccessibleName("Decoded messages")
        # Tab moves on to the next control instead of walking the cells
        # (the arrow keys still move between rows).
        self._table.setTabKeyNavigation(False)
        self._table.setHorizontalHeaderLabels(list(_HEADERS))
        self._table.setItemDelegate(_MessageDelegate(self._table))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(
            self._table.fontMetrics().height() + 10
        )
        header = self._table.horizontalHeader()
        header.setHighlightSections(False)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setMinimumSectionSize(40)
        for col in (COL_TIME, COL_PROTOCOL, COL_ADDRESS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_MESSAGE, QHeaderView.ResizeMode.Stretch)
        self._table_tip = (
            "Decoded messages, newest at the bottom. Click a message to read "
            "it in full below; invalid messages (failed checks) are tinted "
            "red. Ctrl+C copies the selected rows."
        )
        self._table.setToolTip(self._table_tip)
        self._table.setMinimumHeight(110)
        # Stick to the newest message unless the user scrolled up to read.
        self._table_tail = TailFollower(self._table.verticalScrollBar())
        self._empty = ViewPlaceholder(self._table)
        self._empty.action_triggered.connect(self._tune_to_protocol)
        self._table.itemSelectionChanged.connect(self._update_detail)

        messages_page = QWidget()
        page_layout = QVBoxLayout(messages_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(4)
        # While an aircraft is picked under Aircraft, only its messages show.
        self._filter_bar = QWidget()
        filter_row = QHBoxLayout(self._filter_bar)
        filter_row.setContentsMargins(2, 0, 0, 0)
        filter_row.setSpacing(6)
        self._filter_label = QLabel()
        set_role(self._filter_label, "hint")
        set_tone(self._filter_label, "info")
        self._filter_label.setTextFormat(Qt.TextFormat.PlainText)
        self._filter_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        filter_row.addWidget(self._filter_label, 1)
        self._filter_clear = QPushButton("Show All")
        set_role(self._filter_clear, "compact")
        self._filter_clear.setAutoDefault(False)
        self._filter_clear.setToolTip("Show the messages of every aircraft again")
        self._filter_clear.clicked.connect(
            lambda _checked=False: self.set_address_filter("")
        )
        filter_row.addWidget(self._filter_clear)
        self._filter_bar.hide()
        page_layout.addWidget(self._filter_bar)
        page_layout.addWidget(self._table, 1)
        self._detail = _MessageDetail()
        page_layout.addWidget(self._detail)
        self._tabs.addTab(messages_page, "Messages")
        self._tabs.setTabToolTip(0, "Decoded messages as a table")
        self._messages_page = messages_page
        self._fit_columns()

        # Plain-text log of every message, wrapped so nothing is cut off.
        self._raw_output = QTextEdit()
        self._raw_output.setReadOnly(True)
        self._raw_output.setAcceptRichText(False)
        self._raw_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._raw_output.document().setMaximumBlockCount(self._max_messages)
        set_role(self._raw_output, "terminal")
        self._log_empty = ViewPlaceholder(
            self._raw_output,
            "No messages yet\nEvery decoded message is also listed here in "
            "full as plain text, easy to select and copy.",
        )
        self._log_tail = TailFollower(self._raw_output.verticalScrollBar())
        self._tabs.addTab(self._raw_output, "Log")
        self._tabs.setTabToolTip(1, "Plain-text log of decoded messages, easy to copy")

        self._tabs.addTab(self._build_stats(), "Stats")
        self._tabs.setTabToolTip(2, "Message counts since the last Clear")

        # Views for one protocol each, in front while it is chosen.
        from .decoder_views import AircraftView, MorseTextView

        self._aircraft_view = AircraftView(clock=lambda: self._clock())
        self._aircraft_view.aircraft_selected.connect(self.set_address_filter)
        self._aircraft_view.aircraft_activated.connect(self._show_aircraft_messages)
        self._aircraft_view.tune_requested.connect(self._tune_to_protocol)
        self._tabs.insertTab(0, self._aircraft_view, "Aircraft")
        self._tabs.setTabToolTip(
            0, "One row per aircraft heard (ADS-B), with its latest position"
        )
        self._cw_view = MorseTextView()
        self._cw_view.tune_requested.connect(self._tune_to_protocol)
        self._tabs.insertTab(0, self._cw_view, "Text")
        self._tabs.setTabToolTip(0, "The decoded Morse code as running text")
        # Which view belongs to which protocol.
        self._protocol_views = {
            "ADS-B": self._aircraft_view,
            "CW (Morse)": self._cw_view,
        }
        self._tabs.setCurrentWidget(self._messages_page)
        # The status line's "Tune to ..." stands in for the button of an
        # empty view only while that view is out of sight.
        self._tabs.currentChanged.connect(lambda _index: self._update_tune_button())

        layout.addWidget(self._tabs, 1)

    def _build_stats(self) -> "QWidget":
        """Statistics page: totals plus a per-protocol breakdown."""
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 10, 12, 10)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        def value_label() -> "QLabel":
            label = QLabel()
            set_role(label, "value")
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return label

        self._msg_count_label = value_label()
        form.addRow("Messages received", self._msg_count_label)
        self._valid_count_label = value_label()
        form.addRow("Valid", self._valid_count_label)
        self._invalid_count_label = value_label()
        invalid_caption = QLabel("Invalid")
        invalid_caption.setToolTip("Messages that failed a checksum or parity check")
        form.addRow(invalid_caption, self._invalid_count_label)
        self._last_msg_label = value_label()
        form.addRow("Last message", self._last_msg_label)
        self._kept_label = value_label()
        kept_caption = QLabel("In the table")
        kept_caption.setToolTip(
            f"The table, the Log and Export keep the newest {self._max_messages:,} "
            "messages"
        )
        form.addRow(kept_caption, self._kept_label)

        decoder_caption = QLabel("LIVE DECODER")
        set_role(decoder_caption, "caption")
        decoder_caption.setContentsMargins(0, 8, 0, 0)
        decoder_caption.setToolTip(
            "What the decoder found in the signal since it was started "
            "(choosing a protocol or changing the sample rate restarts it) "
            "or since Clear"
        )
        form.addRow(decoder_caption)
        self._frames_label = value_label()
        frames_caption = QLabel("Frames heard")
        frames_caption.setToolTip(
            "Transmissions, packets or blocks the decoder found and tried to "
            "decode (POCSAG and FLEX: transmissions; APRS: packets; ACARS: "
            "blocks)"
        )
        form.addRow(frames_caption, self._frames_label)
        self._decoded_label = value_label()
        decoded_caption = QLabel("Decoded")
        decoded_caption.setToolTip("Messages that passed every check")
        form.addRow(decoded_caption, self._decoded_label)
        self._failed_label = value_label()
        failed_caption = QLabel("Failed checks")
        failed_caption.setToolTip(
            "Frames or messages that failed a CRC, BCH or parity check: a weak "
            "or mistuned signal, or another station on the channel"
        )
        form.addRow(failed_caption, self._failed_label)
        self._signal_label = value_label()
        signal_caption = QLabel("Signal in channel")
        signal_caption.setToolTip(
            "Whether something stands above the noise in the channel the "
            "decoder listens to"
        )
        form.addRow(signal_caption, self._signal_label)

        by_proto = QLabel("BY PROTOCOL")
        set_role(by_proto, "caption")
        by_proto.setContentsMargins(0, 8, 0, 0)
        form.addRow(by_proto)
        self._proto_none = QLabel("No messages yet")
        set_role(self._proto_none, "hint")
        form.addRow(self._proto_none)
        # One "name  count" row per protocol seen, added as they appear.
        self._stats_form = form
        self._value_label = value_label
        self._proto_rows: Dict[str, Any] = {}

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        # Not a Tab stop: it has nothing to focus and no focus frame.
        area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        area.setWidget(page)
        return area

    def _header_width(self, col: int) -> int:
        """Width that shows a column's (bold, padded) header text in full."""
        bold = QFont(self._table.horizontalHeader().font())
        bold.setBold(True)
        return QFontMetrics(bold).horizontalAdvance(_HEADERS[col]) + _HEADER_PAD

    def _fit_columns(self) -> None:
        """Size Time/Protocol/Address to their contents and let Message
        stretch over the rest.

        In a narrow panel the Address column gives way (eliding, with the
        full text in its tooltip) so Message keeps at least
        ``_MESSAGE_SHARE`` of the width. Fonts are measured now rather than
        at construction, so the widths follow the applied stylesheet.
        """
        table = getattr(self, "_table", None)
        if table is None:
            return
        fm = table.fontMetrics()
        rows = table.verticalHeader()
        if rows.defaultSectionSize() != fm.height() + 10:
            rows.setDefaultSectionSize(fm.height() + 10)
        want: Dict[int, int] = {}
        for col in (COL_TIME, COL_PROTOCOL, COL_ADDRESS):
            if table.isColumnHidden(col):
                continue
            content = self._col_content.get(col, 0)
            if col == COL_TIME:
                content = fm.horizontalAdvance("00:00:00") + _CELL_PAD
            want[col] = max(self._header_width(col), content)

        avail = table.viewport().width()
        message = max(_MIN_MESSAGE_WIDTH, int(avail * _MESSAGE_SHARE))
        overflow = sum(want.values()) + message - avail
        if overflow > 0 and COL_ADDRESS in want:
            floor = min(
                want[COL_ADDRESS],
                max(_MIN_ADDRESS_WIDTH, self._header_width(COL_ADDRESS)),
            )
            want[COL_ADDRESS] = max(floor, want[COL_ADDRESS] - overflow)

        for col, width in want.items():
            if table.columnWidth(col) != width:
                table.setColumnWidth(col, width)

    def _grow_columns(self, texts: Dict[int, str]) -> None:
        """Note new cell text; refit when a column's contents got wider."""
        fm = self._table.fontMetrics()
        grown = False
        for col, text in texts.items():
            wanted = min(fm.horizontalAdvance(text) + _CELL_PAD, _MAX_COL_WIDTH[col])
            if wanted > self._col_content.get(col, 0):
                self._col_content[col] = wanted
                grown = True
        if grown:
            self._fit_columns()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def protocol(self) -> Optional[str]:
        """The protocol being decoded: the selected one, or Auto's choice;
        ``None`` when decoding is off (or Auto finds nothing to decode)."""
        text = self._proto_combo.currentText()
        if text == PROTOCOL_AUTO:
            return self._auto_choice
        return None if text == PROTOCOL_OFF else text

    def selection(self) -> str:
        """The selector's entry ("Off", "Auto" or a protocol name)."""
        return self._proto_combo.currentText()

    def is_auto(self) -> bool:
        return self.selection() == PROTOCOL_AUTO

    def is_decoding(self) -> bool:
        """True when a protocol is selected and Decode is ticked."""
        return self.protocol() is not None and self._enabled_check.isChecked()

    def _on_protocol_changed(self, text: str):
        """Handle protocol change."""
        if text == PROTOCOL_AUTO:
            self._auto_choice = auto_protocol(self._tuned_hz, self._mode)
        self._protocol_now_is(self.protocol())

    def _protocol_now_is(self, proto: Optional[str]) -> None:
        """The decoded protocol changed (to ``proto``; None: none)."""
        # The old decoder's status no longer applies; the main window sends
        # the new one's (or none) while handling the signal.
        self._decoder_status = None
        self._count_base = (0, 0, 0)
        self._chosen_counts = Counter(self._valid_per_protocol)
        self._update_views()
        self._update_state()
        self.protocol_changed.emit(proto or PROTOCOL_OFF)

    def _update_auto(self) -> None:
        """Auto: follow the tuned frequency and mode to another protocol."""
        if not self.is_auto():
            return
        choice = auto_protocol(self._tuned_hz, self._mode)
        if choice != self._auto_choice:
            self._auto_choice = choice
            self._protocol_now_is(choice)

    def _update_views(self) -> None:
        """Show the protocol's own view (Aircraft, Text), in front when the
        protocol was just chosen; the others' tabs hide."""
        proto = self.protocol()
        chosen = proto != getattr(self, "_views_for", None)
        self._views_for = proto
        for name, view in self._protocol_views.items():
            index = self._tabs.indexOf(view)
            shown = name == proto
            if self._tabs.isTabVisible(index) != shown:
                self._tabs.setTabVisible(index, shown)
            if shown and chosen:
                self._tabs.setCurrentWidget(view)
        # A view that went with its protocol leaves Messages in front. (What
        # is in front now: the tab in front before may be the view that
        # just went, while the new protocol's view is in front already.)
        current = self._tabs.currentWidget()
        if current in self._protocol_views.values() and not self._tabs.isTabVisible(
            self._tabs.indexOf(current)
        ):
            self._tabs.setCurrentWidget(self._messages_page)
        if proto != "ADS-B" and self._filter_address:
            self.set_address_filter("")
        self._fit_tab_bar()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._fit_tab_bar()

    def _fit_tab_bar(self) -> None:
        """Every tab title in sight: in a panel too narrow for the tabs and
        the regular Clear / Export buttons, the buttons turn compact."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        bar = tabs.tabBar()
        needed = sum(
            bar.tabSizeHint(i).width()
            for i in range(bar.count())
            if tabs.isTabVisible(i)
        )
        compact = needed + self._corner_width + 8 > tabs.width()
        role = "compact" if compact else ""
        for button in (self._clear_btn, self._export_btn):
            if (button.property("role") or "") != role:
                set_role(button, role or None)

    def _on_decode_toggled(self, checked: bool) -> None:
        # Only a choice made with a protocol selected counts; while Off the
        # box is unticked by the panel itself.
        if self.selection() != PROTOCOL_OFF:
            self._decode_wanted = bool(checked)
        self._update_state()
        self.decode_toggled.emit(bool(checked))  # (the window's chip: "paused")

    def set_current_frequency(self, freq_hz: float) -> None:
        """Tell the panel the tuned frequency (updates the empty-table hint;
        Auto may pick another protocol)."""
        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return
        if math.isfinite(freq_hz) and freq_hz != self._tuned_hz:
            self._tuned_hz = freq_hz
            self._tuned_at = self._clock()
            self._update_state()
            self._update_auto()

    def set_demod_mode(self, mode: Optional[str]) -> None:
        """Tell the panel the demodulation mode (Auto decodes Morse code
        in CW)."""
        if mode != self._mode:
            self._mode = mode
            self._update_auto()

    def set_sample_rate(self, rate_hz: Optional[float]) -> None:
        """Tell the panel the receiver's sample rate (kept with each
        message, and in the CSV export)."""
        self._sample_rate = float(rate_hz) if rate_hz else None

    # ------------------------------------------------------------------
    # Aircraft filter (ADS-B)
    # ------------------------------------------------------------------

    def set_address_filter(self, address: str) -> None:
        """Show only the messages from ``address`` ("" shows all) and say
        so above the table."""
        address = str(address or "")
        if address == self._filter_address:
            return
        self._filter_address = address
        for row, msg in enumerate(self._messages):
            hide = bool(address) and msg["address"] != address
            if self._table.isRowHidden(row) != hide:
                self._table.setRowHidden(row, hide)
        if address:
            self._filter_count = sum(
                1 for m in self._messages if m["address"] == address
            )
            self._show_filter_label()
            self._filter_bar.show()
            self._tabs.setTabText(self._tabs.indexOf(self._messages_page), "Messages*")
            self._tabs.setTabToolTip(
                self._tabs.indexOf(self._messages_page),
                f"Decoded messages from {self._filter_name()} only (Show All "
                "shows every one)",
            )
        else:
            self._filter_bar.hide()
            self._tabs.setTabText(self._tabs.indexOf(self._messages_page), "Messages")
            self._tabs.setTabToolTip(
                self._tabs.indexOf(self._messages_page), "Decoded messages as a table"
            )
            if self._aircraft_view.selected_icao():
                self._aircraft_view.select("")
        self._table.clearSelection()
        self._fit_tab_bar()

    def address_filter(self) -> str:
        return self._filter_address

    def _filter_name(self) -> str:
        """The filtered aircraft: its callsign and address, once heard."""
        address = self._filter_address
        aircraft = self._aircraft_view.tracker.get(address)
        if aircraft is not None and aircraft.callsign:
            return f"{aircraft.callsign} ({address})"
        return address

    def _show_filter_label(self) -> None:
        """Say whose messages show and how many (kept up to date as they
        come in)."""
        text = f"Messages from {self._filter_name()} only ({self._filter_count:,} here)"
        if self._filter_label.text() != text:
            self._filter_label.setText(text)

    def _show_aircraft_messages(self, icao: str) -> None:
        """Double-click (or Enter) on an aircraft: its messages."""
        if icao:
            self.set_address_filter(icao)
            self._tabs.setCurrentWidget(self._messages_page)

    def set_receiving(self, receiving: bool) -> None:
        """Tell the panel whether the receiver is running."""
        receiving = bool(receiving)
        if receiving != self._receiving:
            self._receiving = receiving
            self._update_state()

    def set_decoder_status(self, status: Any) -> None:
        """Show the live decoder's health: a
        :class:`~sdr_module.dsp.live_decoders.LiveDecoderStatus` (its
        counts, signal and any problem), or None when no decoder runs.

        Cheap enough to call for every block: labels change only when
        their text does.
        """
        problem_changed = getattr(status, "problem", None) != getattr(
            self._decoder_status, "problem", None
        )
        self._decoder_status = status
        if problem_changed:
            self._update_state()  # the empty-table hint names the problem
        else:
            self._update_status_line()
        self._update_decoder_stats()
        if self.protocol() == "CW (Morse)":
            self._cw_view.set_live(str(getattr(status, "live_text", "") or ""))
            self._cw_view.set_speed(
                str(getattr(status, "detail", "") or ""),
                bool(getattr(status, "signal_present", False)),
            )

    def _decoder_counts(self) -> Tuple[int, int, int]:
        """The decoder's (frames, valid, failed) since the last Clear."""
        status = self._decoder_status
        counts = tuple(
            int(getattr(status, attr, 0) or 0) for attr in ("frames", "valid", "failed")
        )
        if any(now < base for now, base in zip(counts, self._count_base, strict=True)):
            self._count_base = (0, 0, 0)  # a new decoder counts from zero
        frames, valid, failed = (
            now - base for now, base in zip(counts, self._count_base, strict=True)
        )
        return frames, valid, failed

    def decoder_status_text(self) -> str:
        """The status line's text ("" when hidden)."""
        return self._status_label.text() if not self._status_label.isHidden() else ""

    def add_view(self, widget: Any, title: str, tooltip: str = "") -> int:
        """Add an output view as a tab before Stats (e.g. an aircraft
        table); returns its index."""
        index = max(0, self._tabs.count() - 1)
        index = self._tabs.insertTab(index, widget, title)
        if tooltip:
            self._tabs.setTabToolTip(index, tooltip)
        return index

    def _tuned_for(self, proto: str) -> bool:
        """Whether the tuned frequency is where ``proto`` is heard."""
        freq = self._tuned_hz
        if freq is None:
            return False
        return any(lo <= freq <= hi for lo, hi in PROTOCOL_BANDS.get(proto, ()))

    def _tune_action(self, proto: str) -> Tuple[Optional[str], str]:
        """Text and tooltip of the empty table's "Tune to ..." button."""
        suggestion = PROTOCOL_TUNE.get(proto)
        if suggestion is None or self._tuned_for(proto):
            return None, ""
        # Offered only when something (the main window) acts on it.
        try:
            if self.receivers(self.tune_requested) <= 0:
                return None, ""
        except (TypeError, RuntimeError):  # pragma: no cover - defensive
            return None, ""
        freq, mode, _dev, what = suggestion
        text = f"Tune to {_format_mhz(freq)}"
        tip = f"Tune the receiver to {what}"
        if mode:
            tip += f", in {mode}"
        return text, tip + "."

    def _tune_to_protocol(self) -> None:
        """Emit :attr:`tune_requested` for the selected protocol's channel."""
        proto = self.protocol()
        suggestion = PROTOCOL_TUNE.get(proto or "")
        if proto is None or suggestion is None:
            return
        freq, mode, deviation, _what = suggestion
        self.tune_requested.emit(
            float(freq), f"{proto} ({_format_mhz(freq)})", mode, deviation
        )

    def _waiting_text(self, proto: str, where: str) -> str:
        """Empty-table hint for a selected, decoding protocol."""
        if proto in PROTOCOL_UNAVAILABLE:
            return f"{proto} can't be decoded yet\n{PROTOCOL_UNAVAILABLE[proto]}"
        problem = getattr(self._decoder_status, "problem", None)
        if problem:
            remedy = getattr(self._decoder_status, "remedy", None)
            return f"{proto} can't decode here\n{problem}." + (
                f" {remedy}" if remedy else ""
            )
        if self._tuned_for(proto):
            here = _format_mhz(self._tuned_hz or 0.0)
            if self._receiving:
                return f"Listening for {proto} on {here}\nDecoded messages appear here."
            return (
                f"Waiting for {proto} messages\n"
                f"Tuned to {here}: {_START_HINT} to decode."
            )
        hint = PROTOCOL_HINTS.get(proto)
        if hint:
            if self._receiving:
                return f"Waiting for {proto} messages\n{hint}."
            return f"Waiting for {proto} messages\n{hint}, then {_START_HINT}."
        if not where:
            return (
                f"Waiting for {proto} messages\n{_START_HINT.capitalize()} to decode."
            )
        if self._receiving:
            return f"Waiting for {proto} messages\nTune to {where} to decode."
        return f"Waiting for {proto} messages\nTune to {where}, then {_START_HINT}."

    def _update_state(self, *_args: Any) -> None:
        """Refresh tooltips, enabled states and the empty-table hint."""
        proto = self.protocol()
        selected = self.selection() != PROTOCOL_OFF
        check = self._enabled_check
        check.setEnabled(selected)
        # Unticked while Off (nothing decodes); the user's choice returns
        # once a protocol is selected.
        wanted = selected and self._decode_wanted
        if check.isChecked() != wanted:
            check.blockSignals(True)
            check.setChecked(wanted)
            check.blockSignals(False)
        action: Tuple[Optional[str], str] = (None, "")
        if proto is None and self.is_auto():
            here = _format_mhz(self._tuned_hz) if self._tuned_hz else "here"
            self._proto_combo.setToolTip(_AUTO_TIP)
            check.setToolTip("Untick to pause decoding without changing the protocol")
            empty = (
                f"Nothing to decode on {here}\n"
                "Auto decodes ADS-B on 1090 MHz, APRS on 144.390 MHz, ACARS on "
                "129–137 MHz, pagers on 152–159 and 929–932 MHz, RDS on FM "
                "broadcast and Morse code in CW mode."
            )
        elif proto is None:
            self._proto_combo.setToolTip(
                "Protocol to decode. Off: no decoder runs. After choosing one, "
                "the message table says where to listen."
            )
            check.setToolTip("Choose a protocol first")
            empty = (
                "No decoder selected\n"
                f"Choose a protocol above, tune to its frequency and "
                f"{_START_HINT}."
            )
        else:
            what, where = PROTOCOL_INFO.get(proto, ("", ""))
            tip = f"{proto}: {what}. Usually {where}." if what else proto
            if self.is_auto():
                tip = f"Auto chose {proto} for where the receiver is tuned. {tip}"
            self._proto_combo.setToolTip(tip)
            check.setToolTip("Untick to pause decoding without changing the protocol")
            if check.isChecked():
                empty = self._waiting_text(proto, where)
                action = self._tune_action(proto)
            else:
                empty = "Decoding paused\nTick Decode to resume."
        self._empty.set_text(empty)
        self._empty.set_action(*action)
        self._empty.set_visible(self._table.rowCount() == 0)
        # The protocol's own view (in front) says the same while empty.
        view = self._protocol_views.get(proto or "")
        if view is not None:
            view.set_waiting(self._view_hint(proto, empty), *action)
        self._tune_offer = action
        self._update_tune_button()
        self._update_buttons()
        self._update_status_line()
        self._update_decoder_stats()

    def _update_tune_button(self) -> None:
        """The status line's "Tune to ...": offered whenever the receiver is
        tuned elsewhere, whatever the table holds; hidden only while the
        view in front shows the same button in its empty state."""
        text, tip = self._tune_offer
        front = self._tabs.currentWidget()
        if front is self._messages_page:
            placeholder: Any = self._empty
        elif front in self._protocol_views.values():
            placeholder = front.placeholder()
        else:
            placeholder = None
        in_view = (
            placeholder is not None
            and placeholder.is_visible()
            and placeholder.action_text() == text
        )
        show = bool(text) and not in_view
        button = self._tune_btn
        if show and button.text() != text:
            button.setText(text)
            button.setAccessibleName(text)
        if show and button.toolTip() != tip:
            button.setToolTip(tip)
        if button.isHidden() == show:
            button.setVisible(show)

    def _view_hint(self, proto: Optional[str], empty: str) -> str:
        """The empty-state text for a protocol's view: the table's, but
        about that view once the receiver listens in the right place."""
        if proto == "ADS-B" and empty.startswith("Listening for"):
            title = empty.partition("\n")[0]
            return (
                f"{title}\nEach aircraft gets a row here as its messages are decoded."
            )
        if proto == "CW (Morse)" and empty.startswith("Listening for"):
            title = empty.partition("\n")[0]
            return f"{title}\nThe text appears here as it is received."
        return empty

    def _status_parts(self) -> Tuple[str, Optional[str], str]:
        """``(text, tone, tooltip)`` of the status line ("" hides it)."""
        text, tone, tip = self._status_core()
        if text and self.is_auto() and self.protocol() is not None:
            text = f"Auto: {text}"
        return text, tone, tip

    def _status_core(self) -> Tuple[str, Optional[str], str]:
        proto = self.protocol()
        if proto is None and self.is_auto():
            if not self._enabled_check.isChecked():
                return "Paused: tick Decode to resume", "muted", ""
            here = _format_mhz(self._tuned_hz) if self._tuned_hz else ""
            return (
                f"Auto: nothing to decode on {here}" if here else "Auto: waiting",
                "muted",
                _AUTO_TIP,
            )
        if proto is None or proto in PROTOCOL_UNAVAILABLE:
            return "", None, ""
        if not self._enabled_check.isChecked():
            return "Paused: tick Decode to resume", "muted", ""
        status = self._decoder_status
        problem = getattr(status, "problem", None)
        if problem:
            return problem, "warning", problem
        if not self._receiving:
            return f"Stopped: {_START_HINT} to decode", "muted", ""
        here = _format_mhz(self._tuned_hz) if self._tuned_hz else ""
        # One count with the status-bar chip: the protocol's messages since
        # it was chosen (or Clear). A decoder counting something else (RDS
        # groups, Morse characters) has its own count after it.
        unit = str(getattr(status, "unit", "message") or "message")
        rows = self.count_for(proto)
        row_unit = unit if unit in _ROW_UNITS else "message"
        frames, valid, failed = self._decoder_counts()
        tip = (
            (
                f"Since the decoder started (or Clear): {_plural(frames, 'frame')} "
                f"heard, {valid:,} decoded, {failed:,} failed their checks."
            )
            if status is not None
            else ""
        )
        if here and self._tuned_away(proto, status):
            # The count stays, but not as good news from here.
            earlier = f" ({_plural(rows, row_unit)} earlier)" if rows else ""
            return f"Not tuned to {proto}: listening on {here}{earlier}", None, tip
        if status is None:
            return (f"Listening on {here}" if here else "Listening"), None, ""
        # Seconds of signal since the last valid message: a channel that
        # decoded before but carries something else now says so too.
        undecoded = float(getattr(status, "undecoded_signal_s", 0.0) or 0.0)
        if undecoded >= _UNDECODED_SIGNAL_S:
            return (
                PROTOCOL_UNDECODED.get(
                    proto,
                    "Signal present but nothing decodes: check the frequency "
                    "and the protocol",
                ),
                "warning",
                tip,
            )
        detail = str(getattr(status, "detail", "") or "")
        if rows or valid:
            last = self._last_heard(proto, status)
            age = self._clock() - last if last is not None else 0.0
            count = _plural(rows, row_unit)
            if unit not in _ROW_UNITS:
                count += f" ({_plural(valid, unit)})"
            parts = [proto, detail, count, f"last {_format_age(age)}"]
            parts = [p for p in parts if p]
            if failed:
                parts.append(f"{failed:,} failed check{'' if failed == 1 else 's'}")
            return "  ·  ".join(parts), ("success" if age < _FRESH_S else None), tip
        if failed:
            return (
                f"{_plural(failed, unit)} failed their checks: the signal may be "
                "weak, mistuned or another protocol",
                "warning",
                tip,
            )
        if getattr(status, "signal_present", False):
            text = f"Signal on {here}: waiting for {proto} {unit}s"
            return text, "info", tip
        return (f"Listening on {here}" if here else "Listening"), None, tip

    def _tuned_away(self, proto: str, status: Any) -> bool:
        """Whether the receiver is tuned away from where ``proto`` is heard
        and nothing decoded since it was tuned there. (ADS-B decodes the
        whole span: off 1090 MHz with the span still holding it, its
        messages go on, and so does its count.)"""
        if proto not in PROTOCOL_BANDS or self._tuned_hz is None:
            return False
        if self._tuned_for(proto):
            return False
        last = getattr(status, "last_time", None)
        return last is None or last < self._tuned_at

    def _last_heard(self, proto: str, status: Any) -> Optional[float]:
        """clock() time of the protocol's last message or decoded unit."""
        times = [
            t
            for t in (getattr(status, "last_time", None), self._row_times.get(proto))
            if t is not None
        ]
        return max(times) if times else None

    def _update_status_line(self) -> None:
        text, tone, tip = self._status_parts()
        label = self._status_label
        if label.text() != text:
            label.setText(text)
        set_tone(label, tone)  # (a no-op when unchanged)
        if label.toolTip() != tip:
            label.setToolTip(tip)
        if label.isHidden() == bool(text):
            label.setVisible(bool(text))

    def _update_decoder_stats(self) -> None:
        """The Stats tab's live decoder counts."""
        status = self._decoder_status
        values = {
            self._frames_label: _NO_VALUE,
            self._decoded_label: _NO_VALUE,
            self._failed_label: _NO_VALUE,
            self._signal_label: _NO_VALUE,
        }
        failed = 0
        if status is not None and not getattr(status, "problem", None):
            frames, valid, failed = self._decoder_counts()
            values[self._frames_label] = f"{frames:,}"
            values[self._decoded_label] = f"{valid:,}"
            values[self._failed_label] = f"{failed:,}"
            if getattr(status, "signal_measured", False):
                values[self._signal_label] = (
                    "Yes" if getattr(status, "signal_present", False) else "No"
                )
        for label, text in values.items():
            if label.text() != text:
                label.setText(text)
        set_tone(self._failed_label, "danger" if failed else None)

    def _update_buttons(self) -> None:
        has = bool(self._messages)
        self._clear_btn.setEnabled(has)
        self._export_btn.setEnabled(has)
        self._clear_btn.setToolTip(
            "Remove all decoded messages and reset the statistics"
            if has
            else "Nothing to clear yet"
        )
        kept, total = len(self._messages), self._total
        if not has:
            export_tip = "No messages to export yet"
        elif total > kept:
            export_tip = (
                f"Save the newest {kept:,} of the {total:,} decoded messages to a "
                f"CSV file (the Decoder keeps the newest {self._max_messages:,})"
            )
        else:
            export_tip = "Save the decoded messages to a CSV file"
        if self._export_btn.toolTip() != export_tip:
            self._export_btn.setToolTip(export_tip)

    def _dropped_note(self) -> str:
        """What the table says once its oldest messages went ("" before)."""
        kept, total = len(self._messages), self._total
        if total <= kept:
            return ""
        return (
            f"Showing the newest {kept:,} of {total:,} messages since Clear: the "
            f"table, the Log and Export keep the newest {self._max_messages:,}."
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        protocol: str,
        address: str,
        content: str,
        valid: bool = True,
        raw: str = "",
        *,
        message: Any = None,
        details: Optional[List[str]] = None,
        freq_hz: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ):
        """
        Add a decoded message.

        Args:
            protocol: Protocol name
            address: Address/ID
            content: Message content
            valid: Whether message was valid
            raw: Raw hex data
            message: The decoded message itself (ADS-B messages feed the
                Aircraft view, Morse lines the Text view)
            details: More lines about it, for the card under the table
            freq_hz: Where it was received (default: the tuned frequency)
            sample_rate: The receiver's sample rate (default: as reported)
        """
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
        protocol, address, content = str(protocol), str(address), str(content)
        valid = bool(valid)
        if freq_hz is None:
            freq_hz = self._tuned_hz
        if sample_rate is None:
            sample_rate = self._sample_rate

        msg = {
            "time": timestamp,
            "date": now.strftime("%Y-%m-%d"),
            "protocol": protocol,
            "address": address,
            "content": content,
            "valid": valid,
            "raw": raw,
            "freq_hz": freq_hz,
            "sample_rate": sample_rate,
            "details": list(details or []),
        }
        self._messages.append(msg)
        self._total += 1
        if not valid:
            self._invalid += 1
        self._per_protocol[protocol] += 1
        if valid:
            self._valid_per_protocol[protocol] += 1
            self._row_times[protocol] = self._clock()

        # Keep the table and the stored list the same length.
        overflow = len(self._messages) - self._max_messages
        if overflow > 0:
            if self._filter_address:
                self._filter_count -= sum(
                    1
                    for m in self._messages[:overflow]
                    if m["address"] == self._filter_address
                )
            del self._messages[:overflow]
            for _ in range(overflow):
                self._table.removeRow(0)

        row = self._table.rowCount()
        self._table.insertRow(row)
        short_time = timestamp.split(".")[0]
        cells = {
            COL_TIME: (short_time, f"{msg['date']} {timestamp}"),
            COL_PROTOCOL: (protocol, protocol),
            COL_ADDRESS: (address, address),
            # A dash (like Stats) rather than "(empty)", which read as an error.
            COL_MESSAGE: (content or _NO_VALUE, content or "No message text"),
        }
        for col, (text, tip) in cells.items():
            item = QTableWidgetItem(text)
            if not valid:
                item.setData(_INVALID_ROLE, True)
                tip = f"{tip}\nInvalid: failed a checksum or parity check"
            if tip:
                item.setToolTip(tip)
            self._table.setItem(row, col, item)
        self._grow_columns({COL_PROTOCOL: protocol, COL_ADDRESS: address})
        if self._filter_address and address != self._filter_address:
            self._table.setRowHidden(row, True)
        elif self._filter_address:
            self._filter_count += 1

        self._append_log(msg)

        kind = type(message).__name__ if message is not None else ""
        if kind == "ADSBMessage" and valid:
            self._aircraft_view.update_message(message)
        elif kind == "MorseMessage":
            self._cw_view.add_line(str(getattr(message, "text", "") or content))
        if self._filter_address:
            self._show_filter_label()  # its count (and the callsign, once heard)

        self._empty.set_visible(False)
        self._log_empty.set_visible(False)
        self._update_tune_button()  # (the empty view's own button went)
        self._update_stats()
        self._update_buttons()
        self._update_status_line()  # (it counts the messages)

    def _append_log(self, msg: Dict[str, Any]) -> None:
        """Append one message to the log with a hanging indent."""
        doc = self._raw_output.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not doc.isEmpty():
            cursor.insertBlock()
        indent = self._raw_output.fontMetrics().horizontalAdvance("0000")
        block = QTextBlockFormat()
        block.setLeftMargin(indent)
        block.setTextIndent(-indent)
        block.setBottomMargin(2)
        cursor.setBlockFormat(block)
        bold = QTextCharFormat()
        bold.setFontWeight(QFont.Weight.Bold)
        plain = QTextCharFormat()
        cursor.insertText(msg["time"], bold)
        text = f"  {msg['protocol']}  {msg['address']}  {msg['content']}"
        if not msg["valid"]:
            text += "  [invalid]"
        if msg["raw"]:
            text += f"  raw: {msg['raw']}"
        cursor.insertText(text, plain)

    def add_adsb_message(
        self,
        icao: str,
        callsign: str = "",
        altitude: int = 0,
        lat: float = 0,
        lon: float = 0,
        speed: float = 0,
    ):
        """Add an ADS-B message with specific formatting."""
        parts = []
        if callsign:
            parts.append(str(callsign).strip())
        if altitude:
            parts.append(f"{altitude:,} ft")
        if lat and lon:
            parts.append(f"{lat:.4f}, {lon:.4f}")
        if speed:
            parts.append(f"{speed:.0f} kt")

        # Many replies carry only the aircraft's address; say so rather than
        # showing a bare dash.
        content = "  ·  ".join(parts) or "(aircraft address only)"
        self.add_message("ADS-B", str(icao), content)

    def add_pocsag_message(self, address: int, content: str, function: int = 0):
        """Add a POCSAG message."""
        addr_str = f"{address} (F{function})"
        self.add_message("POCSAG", addr_str, content)

    def add_aprs_message(
        self, source: str, dest: str, lat: float = 0, lon: float = 0, comment: str = ""
    ):
        """Add an APRS message."""
        parts = []
        if lat and lon:
            parts.append(f"{lat:.4f}, {lon:.4f}")
        if comment:
            parts.append(str(comment).strip())

        self.add_message("APRS", f"{source}>{dest}", "  ·  ".join(parts))

    def _update_stats(self):
        """Update statistics display."""
        total = self._total
        invalid = self._invalid

        self._msg_count_label.setText(f"{total:,}")
        self._valid_count_label.setText(f"{total - invalid:,}")
        self._invalid_count_label.setText(f"{invalid:,}")
        set_tone(self._invalid_count_label, "danger" if invalid else None)

        if self._messages:
            self._last_msg_label.setText(self._messages[-1].get("time", _NO_VALUE))
        else:
            self._last_msg_label.setText(_NO_VALUE)
        # The oldest messages go once the table is full: say so.
        note = self._dropped_note()
        kept = f"{len(self._messages):,}" + (" (the newest)" if note else "")
        if self._kept_label.text() != kept:
            self._kept_label.setText(kept)
        tip = f"{self._table_tip}\n{note}" if note else self._table_tip
        if self._table.toolTip() != tip:
            self._table.setToolTip(tip)

        if not self._per_protocol and self._proto_rows:
            for label in self._proto_rows.values():
                self._stats_form.removeRow(label)
            self._proto_rows.clear()
        for name, count in self._per_protocol.items():
            label = self._proto_rows.get(name)
            if label is None:
                label = self._value_label()
                self._stats_form.addRow(name, label)
                self._proto_rows[name] = label
            label.setText(f"{count:,}")
        self._proto_none.setVisible(not self._per_protocol)

        # With a single protocol the column only repeats the selector.
        hide_protocol = len(self._per_protocol) <= 1
        if hide_protocol != self._table.isColumnHidden(COL_PROTOCOL):
            self._table.setColumnHidden(COL_PROTOCOL, hide_protocol)
            self._fit_columns()

    def _update_detail(self) -> None:
        """Show the selected message in full under the table."""
        rows = sorted(i.row() for i in self._table.selectionModel().selectedRows())
        if len(rows) > 1:
            self._detail.show_text(
                f"{len(rows):,} messages selected",
                "Ctrl+C copies them as tab-separated text. Esc clears the "
                "selection.",
            )
            return
        if not rows or not 0 <= rows[0] < len(self._messages):
            self._detail.hide()
            return
        msg = self._messages[rows[0]]
        meta = [msg["time"], msg["protocol"]]
        if msg["address"]:
            meta.append(msg["address"])
        if msg.get("freq_hz"):
            meta.append(_format_mhz(msg["freq_hz"]))
        if not msg["valid"]:
            meta.append("failed checks")
        self._detail.show_text(
            "  ·  ".join(meta),
            msg["content"] or "(no message text)",
            None if msg["valid"] else "danger",
            msg.get("details"),
        )
        # The card takes height from the table; once that layout settles,
        # keep the selected row in view.
        QTimer.singleShot(0, self._ensure_selection_visible)

    def _ensure_selection_visible(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if len(rows) == 1:
            self._table.scrollTo(rows[0])

    def clear(self):
        """Clear all messages (and count the decoder's from here)."""
        status = self._decoder_status
        self._count_base = tuple(
            int(getattr(status, attr, 0) or 0) for attr in ("frames", "valid", "failed")
        )
        self._messages.clear()
        self._total = 0
        self._invalid = 0
        self._per_protocol.clear()
        self._valid_per_protocol.clear()
        self._chosen_counts.clear()
        self._row_times.clear()
        self._table.setRowCount(0)
        self._table_tail.reset()
        self._col_content.clear()
        self._fit_columns()
        self._detail.hide()
        self._raw_output.clear()
        self._log_tail.reset()
        self._log_empty.set_visible(True)
        self.set_address_filter("")
        self._aircraft_view.clear()
        self._cw_view.clear()
        self._update_stats()
        self._update_state()

    def copy_selected(self) -> int:
        """Copy the selected rows to the clipboard as tab-separated text."""
        rows = sorted(i.row() for i in self._table.selectionModel().selectedRows())
        lines = []
        for row in rows:
            if 0 <= row < len(self._messages) and not self._table.isRowHidden(row):
                m = self._messages[row]
                lines.append(
                    "\t".join((m["time"], m["protocol"], m["address"], m["content"]))
                )
        if lines:
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText("\n".join(lines))
        return len(lines)

    def _export_messages(self, path: Optional[str] = None) -> int:
        """
        Export messages to a CSV file.

        Args:
            path: Destination file; a file dialog is shown when omitted.

        Returns:
            Number of messages written (0 if cancelled or on error).
        """
        if not self._messages:
            return 0
        if not path:
            from . import file_dialogs

            # In Documents/SDR Recordings or the folder of the last export,
            # like the app's other files (a bare name started in the folder
            # the app was started from).
            default = datetime.now().strftime("decoded-%Y%m%d-%H%M%S.csv")
            path, _ = file_dialogs.save_file(
                self,
                self._settings,
                "decode",
                "Export Decoded Messages",
                default,
                "CSV files (*.csv);;All files (*)",
            )
        if not path:
            return 0
        if not os.path.splitext(path)[1]:
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                # The receive frequency and sample rate come last, so the
                # columns of older exports keep their places.
                writer.writerow(
                    [
                        "Date",
                        "Time",
                        "Protocol",
                        "Address",
                        "Content",
                        "Valid",
                        "Raw",
                        "Frequency (MHz)",
                        "Sample rate (MS/s)",
                    ]
                )
                for msg in self._messages:
                    freq = msg.get("freq_hz")
                    rate = msg.get("sample_rate")
                    writer.writerow(
                        [
                            msg.get("date", ""),
                            msg.get("time", ""),
                            msg.get("protocol", ""),
                            msg.get("address", ""),
                            msg.get("content", ""),
                            msg.get("valid", True),
                            msg.get("raw", ""),
                            f"{freq / 1e6:.6f}" if freq else "",
                            f"{rate / 1e6:g}" if rate else "",
                        ]
                    )
        except OSError as exc:
            QMessageBox.warning(
                self, "Export Failed", f"Could not write the file:\n{exc}"
            )
            return 0

        count = len(self._messages)
        if self._total > count:
            # The oldest went when the table was full: not in the file.
            text = (
                f"Exported the newest {count:,} of {self._total:,} messages to:\n"
                f"{path}\n\nThe Decoder keeps the newest {self._max_messages:,}; "
                "the older ones were not saved."
            )
        else:
            text = f"Exported {count:,} message{'' if count == 1 else 's'} to:\n{path}"
        QMessageBox.information(self, "Export Messages", text)
        return count

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)

    def count_for(self, protocol: Optional[str]) -> int:
        """Messages of ``protocol`` that passed their checks since it was
        chosen or since the last Clear, whichever came later: what the
        status line and the window's status-bar chip count (a decoder
        rebuilt for another sample rate goes on counting)."""
        name = protocol or ""
        return max(
            0,
            int(self._valid_per_protocol.get(name, 0))
            - int(self._chosen_counts.get(name, 0)),
        )
