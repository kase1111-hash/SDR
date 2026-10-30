"""
Views the Decoder panel adds for particular protocols.

- :class:`AircraftTracker` and :class:`AircraftView` (ADS-B): one row per
  aircraft, its latest callsign, altitude, speed, track, vertical rate,
  position and squawk, with how many messages it sent and when it was last
  heard. Rows dim after a minute without a message and go after five.
- :class:`MorseTextView` (CW): the decoded text as it comes in, the line
  still being received included, and the sending speed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QBrush, QFont, QKeySequence, QTextCursor
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPlainTextEdit,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

from .themes import get_palette, set_role, set_tone, theme_notifier

_NO_VALUE = "–"


# --------------------------------------------------------------------------- #
# ADS-B aircraft
# --------------------------------------------------------------------------- #


@dataclass
class Aircraft:
    """What is known about one aircraft (by its ICAO address)."""

    icao: str
    callsign: str = ""
    category: str = ""
    altitude: Optional[int] = None  # feet
    speed: Optional[float] = None  # knots, over the ground
    track: Optional[float] = None  # degrees
    vertical_rate: Optional[int] = None  # feet per minute
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    squawk: str = ""
    on_ground: bool = False
    messages: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def name(self) -> str:
        """The callsign, or the ICAO address until one is heard."""
        return self.callsign or self.icao


def format_age(seconds: float) -> str:
    """``0.5 -> "now"``, ``3 -> "3 s"``, ``75 -> "1 min"``."""
    seconds = max(0.0, float(seconds))
    if seconds < 2:
        return "now"
    if seconds < 60:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"


def format_vertical_rate(rate: Optional[int]) -> str:
    """``1200 -> "+1,200"`` (feet per minute), ``0 -> "level"``."""
    if rate is None:
        return _NO_VALUE
    if rate == 0:
        return "level"
    return f"{rate:+,}"


def describe_aircraft(aircraft: Aircraft, now: float) -> str:
    """One line about an aircraft (the Aircraft view's detail line)."""
    parts = [
        aircraft.callsign and f"{aircraft.callsign} ({aircraft.icao})" or aircraft.icao
    ]
    if aircraft.category:
        parts.append(aircraft.category)
    if aircraft.on_ground:
        parts.append("on the ground")
    elif aircraft.altitude is not None:
        altitude = f"{aircraft.altitude:,} ft"
        if aircraft.vertical_rate is not None:
            rate = aircraft.vertical_rate
            altitude += (
                ", level"
                if rate == 0
                else f", {'climbing' if rate > 0 else 'descending'} {abs(rate):,} ft/min"
            )
        parts.append(altitude)
    if aircraft.speed is not None:
        speed = f"{aircraft.speed:.0f} kt"
        if aircraft.track is not None:
            speed += f", track {round(aircraft.track) % 360}°"
        parts.append(speed)
    if aircraft.latitude is not None and aircraft.longitude is not None:
        parts.append(f"{aircraft.latitude:.4f}, {aircraft.longitude:.4f}")
    if aircraft.squawk:
        parts.append(f"squawk {aircraft.squawk}")
    count = aircraft.messages
    parts.append(f"{count:,} message{'' if count == 1 else 's'}")
    age = format_age(now - aircraft.last_seen)
    parts.append("last heard just now" if age == "now" else f"last heard {age} ago")
    return "  ·  ".join(parts)


class AircraftTracker:
    """Aircraft by ICAO address, from decoded ADS-B messages (no Qt).

    :meth:`update` takes every valid ADS-B message (the latest of each
    field wins, a field not in a message keeps its value); an aircraft not
    heard for ``STALE_S`` is stale, and after ``DROP_S`` it is dropped by
    :meth:`expire`.
    """

    STALE_S = 60.0
    DROP_S = 300.0

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._aircraft: Dict[str, Aircraft] = {}

    def __len__(self) -> int:
        return len(self._aircraft)

    def get(self, icao: str) -> Optional[Aircraft]:
        return self._aircraft.get(icao)

    def aircraft(self) -> List[Aircraft]:
        """Every aircraft, in the order they were first heard."""
        return list(self._aircraft.values())

    def clear(self) -> None:
        self._aircraft.clear()

    def is_stale(self, aircraft: Aircraft) -> bool:
        return self._clock() - aircraft.last_seen > self.STALE_S

    def update(self, msg: Any) -> Optional[Aircraft]:
        """Take one ADS-B message; returns its aircraft (None if the
        message is not a valid one with an address)."""
        icao = str(getattr(msg, "icao_address", "") or "").strip()
        if not icao or not getattr(msg, "valid", True):
            return None
        now = self._clock()
        aircraft = self._aircraft.get(icao)
        if aircraft is None:
            aircraft = self._aircraft[icao] = Aircraft(icao=icao, first_seen=now)
        aircraft.messages += 1
        aircraft.last_seen = now
        callsign = str(getattr(msg, "callsign", "") or "").strip()
        if callsign:
            aircraft.callsign = callsign
        category = str(getattr(msg, "category", "") or "")
        if category:
            aircraft.category = category
        altitude = int(getattr(msg, "altitude", 0) or 0)
        if altitude:
            aircraft.altitude = altitude
        type_code = int(getattr(msg, "type_code", 0) or 0)
        speed = float(getattr(msg, "velocity", 0.0) or 0.0)
        if type_code == 19 and speed:
            aircraft.speed = speed
            aircraft.track = float(getattr(msg, "heading", 0.0) or 0.0)
            if getattr(msg, "vertical_rate_known", True):
                aircraft.vertical_rate = int(getattr(msg, "vertical_rate", 0) or 0)
            else:
                aircraft.vertical_rate = None
        lat = float(getattr(msg, "latitude", 0.0) or 0.0)
        lon = float(getattr(msg, "longitude", 0.0) or 0.0)
        if lat or lon:
            aircraft.latitude, aircraft.longitude = lat, lon
        squawk = str(getattr(msg, "squawk", "") or "").strip()
        if squawk:
            aircraft.squawk = squawk
        df = int(getattr(msg, "downlink_format", 0) or 0)
        if df in (17, 18):
            if 5 <= type_code <= 8:
                aircraft.on_ground = True
            elif 9 <= type_code <= 19:
                aircraft.on_ground = False
        elif df in (0, 4, 5, 11, 20, 21):
            aircraft.on_ground = bool(getattr(msg, "on_ground", False))
        return aircraft

    def expire(self) -> List[str]:
        """Drop aircraft not heard for ``DROP_S``; returns their addresses."""
        cutoff = self._clock() - self.DROP_S
        gone = [icao for icao, a in self._aircraft.items() if a.last_seen < cutoff]
        for icao in gone:
            del self._aircraft[icao]
        return gone


# Aircraft table columns: the most telling first, the long ones last (a
# narrow panel scrolls sideways to them; the line under the table and the
# tooltips have everything).
(
    AC_NAME,
    AC_ALTITUDE,
    AC_SPEED,
    AC_TRACK,
    AC_VS,
    AC_SQUAWK,
    AC_SEEN,
    AC_MESSAGES,
    AC_POSITION,
    AC_ICAO,
) = range(10)
_AC_HEADERS = (
    ("Flight", "Callsign (flight number); the ICAO address until one is heard"),
    ("Altitude", "Barometric altitude in feet ('ground' on the ground)"),
    ("Speed", "Ground speed in knots"),
    ("Track", "Direction of travel over the ground, degrees from north"),
    ("V/S", "Vertical speed in feet per minute: + climbing, - descending"),
    ("Squawk", "Transponder code set by the crew (7500, 7600, 7700: emergencies)"),
    ("Seen", "Time since its last message (rows dim after a minute)"),
    ("Msgs", "Messages heard from this aircraft"),
    ("Position", "Latitude, longitude (degrees; south and west negative)"),
    ("ICAO", "The aircraft's 24-bit ICAO address (hex): it never changes"),
)


class AircraftView(QWidget if HAS_PYQT6 else object):
    """One row per aircraft (ADS-B), newest information in place.

    Selecting an aircraft emits :attr:`aircraft_selected` with its ICAO
    address ("" when the selection is cleared); double-click or Enter emits
    :attr:`aircraft_activated`.
    """

    if HAS_PYQT6:
        aircraft_selected = pyqtSignal(str)
        aircraft_activated = pyqtSignal(str)
        # Its empty state's "Tune to ..." button was clicked.
        tune_requested = pyqtSignal()

    def __init__(self, parent: Any = None, clock: Callable[[], float] = time.monotonic):
        super().__init__(parent)
        self._clock = clock
        self.tracker = AircraftTracker(clock)
        self._rows: Dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.table = QTableWidget(0, len(_AC_HEADERS))
        self.table.setAccessibleName("Aircraft")
        self.table.setHorizontalHeaderLabels([h for h, _tip in _AC_HEADERS])
        for col, (_title, tip) in enumerate(_AC_HEADERS):
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setTabKeyNavigation(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(
            self.table.fontMetrics().height() + 10
        )
        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setToolTip(
            "One row per aircraft heard, with its latest information. Click an "
            "aircraft to show only its messages under Messages (double-click "
            "to go there). Rows dim after a minute without a message and are "
            "removed after five."
        )
        self.table.setMinimumHeight(110)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.itemDoubleClicked.connect(
            lambda item: self.aircraft_activated.emit(self._icao_at(item.row()))
        )
        self.table.installEventFilter(self)
        layout.addWidget(self.table, 1)

        from .decoder_panel import ViewPlaceholder

        self._empty = ViewPlaceholder(
            self.table,
            "No aircraft yet\nEach aircraft gets a row here as its ADS-B "
            "messages are decoded: callsign, altitude, speed, track and "
            "position.",
        )
        self._empty.action_triggered.connect(self.tune_requested.emit)

        self.detail = QLabel()
        set_role(self.detail, "hint")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.detail.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.detail.setAccessibleName("Selected aircraft")
        self.detail.hide()
        layout.addWidget(self.detail)

        # Ages tick and rows dim or go without messages arriving.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        theme_notifier().theme_changed.connect(self._on_theme_changed)

    # -- API -----------------------------------------------------------------

    def update_message(self, msg: Any) -> None:
        """Take a decoded ADS-B message."""
        aircraft = self.tracker.update(msg)
        if aircraft is None:
            return
        row = self._rows.get(aircraft.icao)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows[aircraft.icao] = row
            for col in range(len(_AC_HEADERS)):
                self.table.setItem(row, col, QTableWidgetItem())
        self._fill(row, aircraft)
        self._empty.set_visible(False)
        if not self._timer.isActive():
            self._timer.start()
        if self.selected_icao() == aircraft.icao:
            self._show_detail(aircraft)

    def clear(self) -> None:
        self.tracker.clear()
        self._rows.clear()
        self.table.setRowCount(0)
        self.detail.hide()
        self._empty.set_visible(True)
        self._timer.stop()

    def set_waiting(
        self, text: str, action: Optional[str] = None, tip: str = ""
    ) -> None:
        """What the empty view says (where to listen) and its optional
        "Tune to ..." button, as the Decoder panel's table has them."""
        self._empty.set_text(text)
        self._empty.set_action(action, tip)
        self._empty.set_visible(not len(self.tracker))

    def placeholder(self) -> Any:
        return self._empty

    def count(self) -> int:
        return len(self.tracker)

    def selected_icao(self) -> str:
        rows = self.table.selectionModel().selectedRows()
        return self._icao_at(rows[0].row()) if rows else ""

    def select(self, icao: str) -> None:
        """Select an aircraft by address ("" clears the selection)."""
        row = self._rows.get(icao)
        if row is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)

    def refresh(self) -> None:
        """Update the ages, dim stale rows and drop old ones."""
        gone = self.tracker.expire()
        if gone:
            selected = self.selected_icao()
            for icao in gone:
                self._rows.pop(icao, None)
            self._rebuild(selected)
            if selected in gone:
                self._on_selection()  # nothing picked now
        for icao, row in self._rows.items():
            aircraft = self.tracker.get(icao)
            if aircraft is not None:
                self._fill(row, aircraft)
        current = self.tracker.get(self.selected_icao())
        if current is not None:
            self._show_detail(current)
        if not len(self.tracker):
            self._empty.set_visible(True)
            self._timer.stop()

    # -- internals -----------------------------------------------------------

    def _on_theme_changed(self, *_args: Any) -> None:
        self.refresh()  # the dimmed rows' color comes from the palette

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        if obj is self.table and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                icao = self.selected_icao()
                if icao:
                    self.aircraft_activated.emit(icao)
                    return True
            if event.key() == Qt.Key.Key_Escape and self.selected_icao():
                self.table.clearSelection()
                return True
            if event.matches(QKeySequence.StandardKey.Copy):
                current = self.tracker.get(self.selected_icao())
                if current is not None:
                    from PyQt6.QtGui import QGuiApplication

                    clipboard = QGuiApplication.clipboard()
                    if clipboard is not None:
                        clipboard.setText(describe_aircraft(current, self._clock()))
                    return True
        return super().eventFilter(obj, event)

    def _icao_at(self, row: int) -> str:
        for icao, r in self._rows.items():
            if r == row:
                return icao
        return ""

    def _rebuild(self, selected: str) -> None:
        """Rows again after aircraft were dropped (order kept), the aircraft
        ``selected`` still selected if it is there: quietly, so the
        Messages filter and its selection are left alone."""
        blocked = self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            self._rows.clear()
            for aircraft in self.tracker.aircraft():
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._rows[aircraft.icao] = row
                for col in range(len(_AC_HEADERS)):
                    self.table.setItem(row, col, QTableWidgetItem())
                self._fill(row, aircraft)
            if selected in self._rows:
                self.select(selected)
        finally:
            self.table.blockSignals(blocked)

    def _fill(self, row: int, aircraft: Aircraft) -> None:
        now = self._clock()
        stale = self.tracker.is_stale(aircraft)
        if aircraft.on_ground:
            altitude = "ground"
        elif aircraft.altitude is not None:
            altitude = f"{aircraft.altitude:,}"
        else:
            altitude = _NO_VALUE
        position = _NO_VALUE
        if aircraft.latitude is not None and aircraft.longitude is not None:
            position = f"{aircraft.latitude:.3f}, {aircraft.longitude:.3f}"
        values = {
            AC_NAME: aircraft.name,
            AC_ICAO: aircraft.icao,
            AC_ALTITUDE: altitude,
            AC_SPEED: (
                f"{aircraft.speed:.0f}" if aircraft.speed is not None else _NO_VALUE
            ),
            AC_TRACK: (
                f"{round(aircraft.track) % 360}°"
                if aircraft.track is not None
                else _NO_VALUE
            ),
            AC_VS: format_vertical_rate(aircraft.vertical_rate),
            AC_POSITION: position,
            AC_SQUAWK: aircraft.squawk or _NO_VALUE,
            AC_MESSAGES: f"{aircraft.messages:,}",
            AC_SEEN: format_age(now - aircraft.last_seen),
        }
        palette = get_palette()
        brush = QBrush(palette.qcolor("disabled" if stale else "text"))
        tip = describe_aircraft(aircraft, now)
        if stale:
            tip += "\nNot heard for over a minute: it may be out of range."
        for col, text in values.items():
            item = self.table.item(row, col)
            if item is None:
                continue
            if item.text() != text:
                item.setText(text)
            if col in (AC_SPEED, AC_TRACK, AC_MESSAGES, AC_SEEN, AC_ALTITUDE, AC_VS):
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            item.setForeground(brush)
            item.setToolTip(tip)
            if col == AC_NAME:
                font = QFont(item.font())
                font.setBold(not stale)
                item.setFont(font)

    def _on_selection(self) -> None:
        icao = self.selected_icao()
        aircraft = self.tracker.get(icao)
        if aircraft is None:
            self.detail.hide()
        else:
            self._show_detail(aircraft)
        self.aircraft_selected.emit(icao)

    def _show_detail(self, aircraft: Aircraft) -> None:
        text = describe_aircraft(aircraft, self._clock())
        if self.detail.text() != text:
            self.detail.setText(text)
        self.detail.show()


# --------------------------------------------------------------------------- #
# Morse code text
# --------------------------------------------------------------------------- #


class MorseTextView(QWidget if HAS_PYQT6 else object):
    """The decoded Morse text as it arrives: finished lines, then the line
    being received (in the accent color), and the speed above."""

    if HAS_PYQT6:
        # Its empty state's button was clicked.
        tune_requested = pyqtSignal()

    _MAX_LINES = 2000

    def __init__(self, parent: Any = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        caption = QLabel("SPEED")
        set_role(caption, "caption")
        row.addWidget(caption)
        self.speed = QLabel(_NO_VALUE)
        set_role(self.speed, "value")
        self.speed.setAccessibleName("Morse speed")
        self.speed.setToolTip(
            "Sending speed in words per minute, measured from the length of "
            "the dits and dashes"
        )
        row.addWidget(self.speed)
        row.addStretch(1)
        self.state = QLabel("")
        set_role(self.state, "hint")
        row.addWidget(self.state)
        layout.addLayout(row)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setAccessibleName("Decoded Morse text")
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text.document().setMaximumBlockCount(self._MAX_LINES)
        set_role(self.text, "terminal")
        self.text.setToolTip(
            "Decoded Morse code. A new line starts after a pause of a few "
            "seconds or a long carrier; the line still being received is "
            "shown last, in color. Select text to copy it."
        )
        layout.addWidget(self.text, 1)

        from .decoder_panel import TailFollower, ViewPlaceholder

        self._empty = ViewPlaceholder(
            self.text,
            "No Morse code yet\nTune to a CW signal in CW mode and press Start "
            "(Space). The demo device has a beacon on 99.820 MHz.",
        )
        self._empty.action_triggered.connect(self.tune_requested.emit)
        self._tail = TailFollower(self.text.verticalScrollBar())
        self._live = ""
        self._has_live = False  # the document's last block is the live line
        self._lines = 0
        theme_notifier().theme_changed.connect(self._restyle_live)

    def add_line(self, text: str) -> None:
        """A finished line of text (the live line is cleared: it was this
        line; the next :meth:`set_live` shows what follows)."""
        self._replace_live("")
        self._live = ""
        cursor = QTextCursor(self.text.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(str(text), self._format(False))
        self._lines += 1
        self._empty.set_visible(False)

    def set_live(self, text: str) -> None:
        """The line being received ("" when none)."""
        if text != self._live:
            self._live = text
            self._replace_live(text)

    def set_speed(self, wpm_text: str, receiving: bool) -> None:
        """Show the speed ("" before it is known) and whether a keyed signal
        is heard now."""
        speed = wpm_text or _NO_VALUE
        if self.speed.text() != speed:
            self.speed.setText(speed)
        state = "Receiving" if receiving else ""
        if self.state.text() != state:
            self.state.setText(state)
            set_tone(self.state, "success" if receiving else None)

    def clear(self) -> None:
        self.text.clear()
        self._has_live = False
        self._live = ""
        self._lines = 0
        self._tail.reset()
        self._empty.set_visible(True)
        self.set_speed("", False)

    def set_waiting(
        self, text: str, action: Optional[str] = None, tip: str = ""
    ) -> None:
        """What the empty view says (where to listen), as the Decoder
        panel's table says it."""
        self._empty.set_text(text)
        self._empty.set_action(action, tip)
        self._empty.set_visible(self.text.document().isEmpty())

    def placeholder(self) -> Any:
        return self._empty

    def plain_text(self) -> str:
        return self.text.toPlainText()

    # -- internals -----------------------------------------------------------

    def _format(self, live: bool) -> Any:
        from PyQt6.QtGui import QTextCharFormat

        fmt = QTextCharFormat()
        if live:
            fmt.setForeground(QBrush(get_palette().qcolor("accent")))
        return fmt

    def _replace_live(self, text: str) -> None:
        doc = self.text.document()
        cursor = QTextCursor(doc)
        if self._has_live:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()
            if text:
                cursor.insertText(text, self._format(True))
            else:
                # Remove the empty block (and the break before it).
                cursor.deletePreviousChar()
                self._has_live = False
            return
        if not text:
            return
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not doc.isEmpty():
            cursor.insertBlock()
        cursor.insertText(text, self._format(True))
        self._has_live = True
        self._empty.set_visible(False)

    def _restyle_live(self, *_args: Any) -> None:
        if self._has_live:
            live = self._live
            self._replace_live("")
            self._replace_live(live)
