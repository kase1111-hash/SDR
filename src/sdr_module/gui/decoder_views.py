"""
Views the Decoder panel adds for particular protocols.

- :class:`AircraftTracker` and :class:`AircraftView` (ADS-B): one row per
  aircraft, its latest callsign, altitude, speed, track, vertical rate,
  position and squawk, with how many messages it sent and when it was last
  heard. Rows dim after a minute without a message and go after five.
- :class:`MorseTextView` (CW): the decoded text as it comes in, the line
  still being received included, and the sending speed.

Both also show what File > Decode Recording finds in a recording, in a
list of its own per recording, picked under "Showing:" (:class:`SourceBar`)
above the view: what is heard live and what a recording holds are never
mixed. A recording's messages are timed by its own clock (its time of day
when it says when it was made, else the time into it), so its aircraft
neither dim nor go as the wall clock runs on.

What another receiver decodes (the Second Receiver window's, marked
"2nd RX") is live too: its aircraft join the live list (the same sky, one
row per aircraft whichever receiver heard it, saying which did and when),
and its Morse text is an entry of its own under "Showing:" (two receivers'
texts mixed line by line would make no sense).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QBrush, QFont, QFontMetrics, QKeySequence, QTextCursor
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
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
    #: When each receiver last heard it: "" this window's, else another
    #: receiver's mark ("2nd RX", the Second Receiver's).
    heard_by: Dict[str, float] = field(default_factory=dict)

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


def format_recording_time(t: float, absolute: bool) -> str:
    """When a recording's message came, by the recording's clock: its time
    of day (``absolute``: ``t`` is a time since 1970, shown in local time
    as the Messages table shows it, ``"07:00:01"``), else the time into the
    recording (``t`` seconds: ``"1:15"``, ``"1:02:05"``)."""
    if absolute:
        try:
            return datetime.fromtimestamp(float(t)).strftime("%H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return _NO_VALUE
    value = float(t)
    if not math.isfinite(value):
        return _NO_VALUE
    hours, rest = divmod(int(max(0.0, value)), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def recording_time(
    absolute: bool, at_s: Optional[float], when: Optional[datetime], last: float
) -> float:
    """A recording's message time on its own clock (see
    :func:`format_recording_time`): ``when`` (its time of day, given when
    the recording says when it was made) for an ``absolute`` clock, else
    the seconds ``at_s`` into it; ``last`` (the latest time so far) when
    the message has no such time, or now for an absolute clock."""
    if absolute:
        if when is not None:
            try:
                return float(when.timestamp())
            except (OverflowError, OSError, ValueError):
                pass
        return time.time()
    if at_s is not None:
        try:
            value = float(at_s)
        except (TypeError, ValueError):
            return last
        if math.isfinite(value):
            return value
    return last


def format_vertical_rate(rate: Optional[int]) -> str:
    """``1200 -> "+1,200"`` (feet per minute), ``0 -> "level"``."""
    if rate is None:
        return _NO_VALUE
    if rate == 0:
        return "level"
    return f"{rate:+,}"


def describe_aircraft(aircraft: Aircraft, now: float, heard: str = "") -> str:
    """One line about an aircraft (the Aircraft view's detail line).
    ``heard`` says when it was last heard instead of its age (a
    recording's aircraft: "last heard at 07:00:01 in adsb.sigmf-data")."""
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
    if heard:
        parts.append(heard)
        return "  ·  ".join(parts)
    parts.append(
        f"last heard {_ago(now - aircraft.last_seen)}{_heard_by(aircraft, now)}"
    )
    return "  ·  ".join(parts)


#: What the views call this window's own receiver beside another one.
MAIN_RECEIVER = "main receiver"


def _ago(seconds: float) -> str:
    """``"just now"``, ``"3 s ago"``, ``"2 min ago"``."""
    age = format_age(seconds)
    return "just now" if age == "now" else f"{age} ago"


def _heard_by(aircraft: Aircraft, now: float) -> str:
    """Which receivers heard a live aircraft, and when: "" when only this
    window's did (as it read before there was another), " by the 2nd RX"
    when only another did, else each with its own time ("(main receiver
    just now, 2nd RX 4 s ago)")."""
    others = [mark for mark in aircraft.heard_by if mark]
    if not others:
        return ""
    if len(aircraft.heard_by) == 1:
        return f" by the {others[0]}"
    each = [
        f"{mark or MAIN_RECEIVER} {_ago(now - seen)}"
        for mark, seen in sorted(aircraft.heard_by.items(), key=lambda kv: bool(kv[0]))
    ]
    return f" ({', '.join(each)})"


class AircraftTracker:
    """Aircraft by ICAO address, from decoded ADS-B messages (no Qt).

    :meth:`update` takes every valid ADS-B message (the latest of each
    field wins, a field not in a message keeps its value); an aircraft not
    heard for ``STALE_S`` is stale, and after ``DROP_S`` it is dropped by
    :meth:`expire`.

    ``ages`` False: a recording's aircraft, timed by the recording's clock
    (``update(msg, at=...)``), which are never stale and never dropped.
    """

    STALE_S = 60.0
    DROP_S = 300.0

    def __init__(self, clock: Callable[[], float] = time.monotonic, ages: bool = True):
        self._clock = clock
        self.ages = bool(ages)
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
        return self.ages and self._clock() - aircraft.last_seen > self.STALE_S

    def update(
        self, msg: Any, at: Optional[float] = None, receiver: str = ""
    ) -> Optional[Aircraft]:
        """Take one ADS-B message (heard ``at``, default: the clock's now,
        by ``receiver``: "" this window's, else another's mark); returns
        its aircraft (None if the message is not a valid one with an
        address)."""
        icao = str(getattr(msg, "icao_address", "") or "").strip()
        if not icao or not getattr(msg, "valid", True):
            return None
        now = self._clock() if at is None else float(at)
        aircraft = self._aircraft.get(icao)
        if aircraft is None:
            aircraft = self._aircraft[icao] = Aircraft(icao=icao, first_seen=now)
        aircraft.messages += 1
        aircraft.last_seen = now
        aircraft.heard_by[str(receiver or "")] = now
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
        if not self.ages:
            return []
        cutoff = self._clock() - self.DROP_S
        gone = [icao for icao, a in self._aircraft.items() if a.last_seen < cutoff]
        for icao in gone:
            del self._aircraft[icao]
        return gone


# Aircraft table columns: the most telling first, the long ones last (a
# narrow panel scrolls sideways to them; the line under the table and the
# tooltips have everything). When it was last heard and how often come
# early: the right column shows about six of them.
(
    AC_NAME,
    AC_ALTITUDE,
    AC_SPEED,
    AC_SEEN,
    AC_MESSAGES,
    AC_TRACK,
    AC_VS,
    AC_SQUAWK,
    AC_POSITION,
    AC_ICAO,
) = range(10)
_AC_HEADERS = (
    ("Flight", "Callsign (flight number); the ICAO address until one is heard"),
    ("Altitude", "Barometric altitude in feet ('ground' on the ground)"),
    ("Speed", "Ground speed in knots"),
    ("Seen", "Time since its last message (rows dim after a minute)"),
    ("Msgs", "Messages heard from this aircraft"),
    ("Track", "Direction of travel over the ground, degrees from north"),
    ("V/S", "Vertical speed in feet per minute: + climbing, - descending"),
    ("Squawk", "Transponder code set by the crew (7500, 7600, 7700: emergencies)"),
    ("Position", "Latitude, longitude (degrees; south and west negative)"),
    ("ICAO", "The aircraft's 24-bit ICAO address (hex): it never changes"),
)
# Columns of numbers (right-aligned).
_AC_NUMBERS = (AC_ALTITUDE, AC_SPEED, AC_TRACK, AC_VS, AC_SEEN, AC_MESSAGES)
# Item roles: the aircraft's ICAO address (on its Flight item) and a cell's
# sort key (numbers sort as numbers: "5,000" before "12,000").
_ICAO_ROLE = Qt.ItemDataRole.UserRole.value + 2 if HAS_PYQT6 else 0
_SORT_ROLE = Qt.ItemDataRole.UserRole.value + 3 if HAS_PYQT6 else 0
# Sort key of a value not known yet (it sorts before every known one).
_UNKNOWN = float("-inf")


def _number(value: Optional[float]) -> float:
    """A sort key: the value, or ``_UNKNOWN``."""
    return float(value) if value is not None else _UNKNOWN


class _AircraftItem(QTableWidgetItem if HAS_PYQT6 else object):
    """A cell that sorts by its key when it has one, else by its text."""

    def __lt__(self, other: Any) -> bool:
        mine = self.data(_SORT_ROLE)
        theirs = other.data(_SORT_ROLE) if isinstance(other, _AircraftItem) else None
        if mine is not None and theirs is not None:
            return bool(mine < theirs)
        return bool(super().__lt__(other))


#: What the source bar calls what is heard now.
LIVE = "Live"

#: How another receiver's list is told apart from a recording's in a view's
#: list of sources (a file name never holds it): see :func:`receiver_source`.
_RECEIVER_SOURCE = "\0"


def receiver_source(mark: str) -> str:
    """The source a view knows another receiver's live list by ("2nd RX":
    the Second Receiver's): for :meth:`MorseTextView.show_source` and
    :meth:`MorseTextView.plain_text`. Never a recording's name, even one
    named like the receiver."""
    return _RECEIVER_SOURCE + str(mark)


_LIVE_TABLE_TIP = (
    "One row per aircraft heard, with its latest information. Click an "
    "aircraft to show only its messages under Messages (double-click to go "
    "there); click a title to sort by it. Rows dim after a minute without a "
    "message and are removed after five."
)
_RECORDING_TABLE_TIP = (
    "One row per aircraft heard in {name}, as File > Decode Recording found "
    "them, with when each was last heard in it (the recording's own time). "
    "Click an aircraft to show only its messages under Messages "
    "(double-click to go there); click a title to sort by it. Showing: "
    "Live goes back to the aircraft heard now."
)
_LIVE_SEEN_TIP = _AC_HEADERS[AC_SEEN][1]
_RECORDING_SEEN_TIP = (
    "When it was last heard in the recording: its time of day when the "
    "recording says when it was made, else the time into the recording"
)


class SourceBar(QWidget if HAS_PYQT6 else object):
    """ "Showing: [Live]" above a protocol view: Live, or one of the
    recordings File > Decode Recording decoded, each in a list of its own
    (what is heard now and what a recording holds are never mixed). Hidden
    while there is no recording (a live view looks as it did). Picking one
    emits :attr:`chosen` with its name ("" for Live)."""

    if HAS_PYQT6:
        chosen = pyqtSignal(str)

    def __init__(self, what: str, parent: Any = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(6)
        label = QLabel("Showing:")
        set_role(label, "hint")
        row.addWidget(label)
        self.combo = QComboBox()
        self.combo.setAccessibleName(f"{what} shown")
        self.combo.setToolTip(
            f"Which {what.lower()} this view shows: {LIVE} (what the receiver "
            "decodes now) or a recording File > Decode Recording decoded, "
            "timed by the recording's clock. Clear empties them all."
        )
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo.setMinimumContentsLength(10)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label.setBuddy(self.combo)
        self.combo.activated.connect(self._on_activated)
        row.addWidget(self.combo, 1)
        self._what = what
        self._tip = self.combo.toolTip()
        self.hide()

    def set_sources(
        self, recordings: List[str], current: str, receivers: Any = ()
    ) -> None:
        """Offer Live, another receiver's live list for each mark in
        ``receivers`` ("2nd RX"), then ``recordings`` (in this order),
        ``current`` picked; shown while there is more than Live."""
        receivers = [str(mark) for mark in receivers]
        blocked = self.combo.blockSignals(True)
        try:
            self.combo.clear()
            self.combo.addItem(LIVE, "")
            tip = self._tip
            if receivers:
                self.combo.setItemData(
                    0,
                    f"What this window's receiver decodes now ({MAIN_RECEIVER})",
                    Qt.ItemDataRole.ToolTipRole,
                )
                tip = (
                    f"Which {self._what.lower()} this view shows: {LIVE} (what "
                    f"this window's receiver decodes now), "
                    f"{', '.join(receivers)} (what the Second Receiver window "
                    "decodes now) or a recording File > Decode Recording "
                    "decoded, timed by the recording's clock. Clear empties "
                    "them all."
                )
            for mark in receivers:
                self.combo.addItem(mark, receiver_source(mark))
                self.combo.setItemData(
                    self.combo.count() - 1,
                    f"What the Second Receiver window decodes now ({mark})",
                    Qt.ItemDataRole.ToolTipRole,
                )
            if self.combo.toolTip() != tip:
                self.combo.setToolTip(tip)
            for name in recordings:
                self.combo.addItem(name, name)
                self.combo.setItemData(
                    self.combo.count() - 1,
                    f"What File > Decode Recording found in {name}",
                    Qt.ItemDataRole.ToolTipRole,
                )
            index = self.combo.findData(current)
            self.combo.setCurrentIndex(max(0, index))
        finally:
            self.combo.blockSignals(blocked)
        self.setVisible(bool(recordings or receivers))

    def current_text(self) -> str:
        return self.combo.currentText()

    def current_source(self) -> str:
        return str(self.combo.currentData() or "")

    def texts(self) -> List[str]:
        return [self.combo.itemText(i) for i in range(self.combo.count())]

    def choose(self, source: str) -> None:
        """Pick ``source`` ("" for Live) as the user would."""
        index = self.combo.findData(str(source or ""))
        if index >= 0:
            self.combo.setCurrentIndex(index)
            self._on_activated(index)

    def _on_activated(self, index: int) -> None:
        self.chosen.emit(str(self.combo.itemData(index) or ""))


@dataclass
class _RecordedAircraft:
    """A recording's aircraft (see :class:`AircraftView`): timed by its own
    clock, ``absolute`` (its time of day) or into it."""

    absolute: bool
    latest: float = 0.0
    tracker: Any = None


#: Hints of the empty views: recordings show there too.
AIRCRAFT_RECORDING_HINT = (
    "File > Decode Recording lists the aircraft of a recording here too."
)
TEXT_RECORDING_HINT = "File > Decode Recording shows the text of a recording here too."


class AircraftView(QWidget if HAS_PYQT6 else object):
    """One row per aircraft (ADS-B), newest information in place.

    Selecting an aircraft emits :attr:`aircraft_selected` with its ICAO
    address ("" when the selection is cleared); double-click or Enter emits
    :attr:`aircraft_activated`.

    The aircraft heard live are in :attr:`tracker`; each recording's
    (``update_message(msg, source=name, ...)``, Decode Recording) in a list
    of its own, timed by its clock, which never dims or drops them. The
    table shows one list at a time, picked in the source bar above it (the
    newest recording when it comes); the live list goes on behind it.
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
        # Each recording's aircraft (Decode Recording), in the order they
        # came, and the list the table shows ("": live).
        self._recordings: Dict[str, _RecordedAircraft] = {}
        self._shown = ""
        # Other receivers whose aircraft joined the live list since Clear
        # (their marks, in the order they came: "2nd RX").
        self._others: Dict[str, None] = {}
        # Each aircraft's Flight item (its row moves when the list is
        # sorted: the row is looked up from the item).
        self._items: Dict[str, Any] = {}
        # Whether each row is shown dimmed (its brush and font are only
        # set when that changes).
        self._dimmed: Dict[str, bool] = {}
        # Widest text (pixels) each column has had, and the fonts measured.
        self._widest: Dict[int, int] = {}
        self._metrics: Dict[bool, Any] = {}
        self._fitting_held = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._source_bar = SourceBar("Aircraft")
        self._source_bar.chosen.connect(self.show_source)
        layout.addWidget(self._source_bar)
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
        # The columns are fitted to their contents only when a value is
        # wider than any before (see _fill): sizing them to their contents
        # on every change re-measured every row for each changed cell, and
        # tens of aircraft then saturated the GUI thread.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        # A click on a title sorts by it (numbers as numbers; again: the
        # other way); until then the aircraft are listed as first heard.
        # Qt's sort arrow takes room in every title (a column fewer would
        # fit): the sorted column's title shows the direction instead.
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)
        header.setSortIndicatorShown(False)
        header.sortIndicatorChanged.connect(self._on_sort_changed)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setToolTip(_LIVE_TABLE_TIP)
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
            f"position.\n{AIRCRAFT_RECORDING_HINT}",
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

    def update_message(
        self,
        msg: Any,
        source: str = "",
        at_s: Optional[float] = None,
        when: Optional[datetime] = None,
        receiver: str = "",
    ) -> bool:
        """Take a decoded ADS-B message: heard live, or (``source``) found
        in that recording by Decode Recording, ``at_s`` seconds into it and
        (``when``, if the recording says when it was made) at that time.
        True when it started the list of a recording (it is shown now).

        ``receiver``: heard live by another receiver ("2nd RX", the Second
        Receiver's): the same sky, so its aircraft are the live list's, a
        row per aircraft whichever receiver heard it, and say which did
        (see :func:`describe_aircraft`). True for that receiver's first
        aircraft since Clear (the view holds another receiver's now)."""
        source = str(source or "")
        if source:
            return self._recorded(msg, source, at_s, when)
        receiver = str(receiver or "")
        aircraft = self.tracker.update(msg, receiver=receiver)
        if aircraft is None:
            return False
        first = bool(receiver) and receiver not in self._others
        if first:
            self._others[receiver] = None
        if not self._timer.isActive():
            self._timer.start()  # (ages tick, the gone go, shown or not)
        if not self._shown:
            self._show_row(aircraft)
        return first

    def _recorded(
        self, msg: Any, source: str, at_s: Optional[float], when: Optional[datetime]
    ) -> bool:
        record = self._recordings.get(source)
        new = record is None
        if record is None:
            # Its clock: the time of day when the recording says when it
            # was made, else the time into it.
            record = _RecordedAircraft(absolute=when is not None or at_s is None)
            held = record
            record.tracker = AircraftTracker(clock=lambda: held.latest, ages=False)
        heard = recording_time(record.absolute, at_s, when, record.latest)
        aircraft = record.tracker.update(msg, at=heard)
        if aircraft is None:
            return False  # (a list only for a recording with aircraft)
        record.latest = max(record.latest, heard)
        if new:
            self._recordings[source] = record
            self.show_source(source)  # (the newest decode: what was asked)
            return True
        if self._shown == source:
            self._show_row(aircraft)
        return False

    def _show_row(self, aircraft: Aircraft) -> None:
        """An aircraft of the list shown, in its row (a new one at the
        bottom)."""
        if aircraft.icao not in self._items:
            self._add_row(aircraft.icao)
        self._fill(aircraft)
        self._empty.set_visible(False)
        if self.selected_icao() == aircraft.icao:
            self._show_detail(aircraft)

    def clear(self) -> None:
        self.tracker.clear()
        self._recordings.clear()
        self._others.clear()
        self._shown = ""
        self._source_bar.set_sources([], "")
        self._show_list_tips()
        self._items.clear()
        self._dimmed.clear()
        self._widest.clear()
        self.table.setRowCount(0)
        self.detail.hide()
        self._empty.set_visible(True)
        self._timer.stop()
        self._fit_columns()

    def forget_source(self, source: str) -> None:
        """Drop a recording's list (it is decoded again: it starts over);
        while it was shown, Live is shown again."""
        if self._recordings.pop(str(source or ""), None) is None:
            return
        if self._shown == source:
            self.show_source("")
        else:
            self._source_bar.set_sources(list(self._recordings), self._shown)

    def show_source(self, source: str) -> None:
        """Show the aircraft heard live ("") or those of a recording."""
        source = str(source or "")
        if source and source not in self._recordings:
            return
        changed = source != self._shown
        selected = self.selected_icao() if changed else ""
        self._shown = source
        self._source_bar.set_sources(list(self._recordings), source)
        if not changed:
            return
        self._show_list_tips()
        self._rebuild("")
        self.detail.hide()
        if selected:
            self.aircraft_selected.emit("")  # (its messages' filter goes)
        self._empty.set_visible(not len(self._shown_tracker()))

    def _show_list_tips(self) -> None:
        """The table's and the Seen column's tooltips, for the list shown."""
        if self._shown:
            tip = _RECORDING_TABLE_TIP.format(name=self._shown)
            seen = _RECORDING_SEEN_TIP
        else:
            tip, seen = _LIVE_TABLE_TIP, _LIVE_SEEN_TIP
        self.table.setToolTip(tip)
        item = self.table.horizontalHeaderItem(AC_SEEN)
        if item is not None:
            item.setToolTip(seen)

    def sources(self) -> List[str]:
        """The recordings with aircraft here, in the order they came."""
        return list(self._recordings)

    def shown_source(self) -> str:
        """The list shown: "" (live) or a recording's name."""
        return self._shown

    def has_recordings(self) -> bool:
        return bool(self._recordings)

    def other_receivers(self) -> List[str]:
        """The other receivers ("2nd RX") whose aircraft joined the live
        list since Clear, in the order they came."""
        return list(self._others)

    def has_other_receivers(self) -> bool:
        return bool(self._others)

    def tracker_for(self, source: str = "") -> Optional[AircraftTracker]:
        """The live aircraft (""), a recording's, or None (no such list)."""
        if not source:
            return self.tracker
        record = self._recordings.get(source)
        return record.tracker if record is not None else None

    def source_bar(self) -> "SourceBar":
        return self._source_bar

    def find(self, icao: str) -> Optional[Aircraft]:
        """An aircraft of the list shown, else heard live."""
        return self._shown_tracker().get(icao) or self.tracker.get(icao)

    def detail_text(self, icao: str) -> str:
        """The line about an aircraft of the list shown ("" if none)."""
        aircraft = self._shown_tracker().get(icao)
        if aircraft is None:
            return ""
        return describe_aircraft(aircraft, self._clock(), self._heard_text(aircraft))

    def set_waiting(
        self, text: str, action: Optional[str] = None, tip: str = ""
    ) -> None:
        """What the empty view says (where to listen) and its optional
        "Tune to ..." button, as the Decoder panel's table has them."""
        self._empty.set_text(text)
        self._empty.set_action(action, tip)
        self._empty.set_visible(not len(self._shown_tracker()))

    def placeholder(self) -> Any:
        return self._empty

    def count(self) -> int:
        """The aircraft of the list shown."""
        return len(self._shown_tracker())

    def selected_icao(self) -> str:
        rows = self.table.selectionModel().selectedRows()
        return self._icao_at(rows[0].row()) if rows else ""

    def select(self, icao: str) -> None:
        """Select an aircraft by address ("" clears the selection)."""
        item = self._items.get(icao)
        if item is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(item.row())

    def refresh(self) -> None:
        """Update the ages, dim stale rows and drop old ones (the live
        list, shown or not: a recording's aircraft keep their own time,
        and its rows are only filled in again)."""
        gone = self.tracker.expire()
        if gone and not self._shown:
            selected = self.selected_icao()
            for icao in gone:
                self._items.pop(icao, None)
            self._rebuild(selected)
            if selected in gone:
                self._on_selection()  # nothing picked now
        shown = self._shown_tracker()
        for icao in list(self._items):
            aircraft = shown.get(icao)
            if aircraft is not None:
                self._fill(aircraft)
        current = shown.get(self.selected_icao())
        if current is not None:
            self._show_detail(current)
        if not len(shown):
            self._empty.set_visible(True)
        if not len(self.tracker):
            self._timer.stop()  # (it ticks the live ages)

    def _shown_tracker(self) -> AircraftTracker:
        record = self._recordings.get(self._shown)
        return record.tracker if record is not None else self.tracker

    def _heard_text(self, aircraft: Aircraft) -> str:
        """When a recording's aircraft was last heard in it ("" live)."""
        record = self._recordings.get(self._shown)
        if record is None:
            return ""
        when = format_recording_time(aircraft.last_seen, record.absolute)
        if record.absolute:
            return f"last heard at {when} in {self._shown}"
        return f"last heard {when} into {self._shown}"

    # -- internals -----------------------------------------------------------

    def _on_theme_changed(self, *_args: Any) -> None:
        self._dimmed.clear()  # every row's color comes from the palette
        self._metrics.clear()
        held, self._fitting_held = self._fitting_held, True  # once, after
        try:
            self.refresh()
        finally:
            self._fitting_held = held
        self._fit_columns()

    def showEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._metrics.clear()
        self._fit_columns()  # in the fonts it is shown with

    def _add_row(self, icao: str) -> None:
        """A new, empty row for an aircraft at the bottom (a sorted list
        moves it to its place as its values are filled in)."""
        table = self.table
        row = table.rowCount()
        table.insertRow(row)
        # The sorted column's cell last: setting it may move the row, and
        # the others go to this row first.
        sorted_col = table.horizontalHeader().sortIndicatorSection()
        cols = [c for c in range(len(_AC_HEADERS)) if c != sorted_col]
        if 0 <= sorted_col < len(_AC_HEADERS):
            cols.append(sorted_col)
        for col in cols:
            item = _AircraftItem()
            if col in _AC_NUMBERS:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            if col == AC_NAME:
                item.setData(_ICAO_ROLE, icao)
                self._items[icao] = item
            table.setItem(row, col, item)

    def _on_sort_changed(self, section: int, order: Any) -> None:
        """The sorted column's title shows the direction (Qt's own arrow
        is hidden: it takes room in every title)."""
        arrow = " ▲" if order == Qt.SortOrder.AscendingOrder else " ▼"
        for col, (title, _tip) in enumerate(_AC_HEADERS):
            item = self.table.horizontalHeaderItem(col)
            text = title + (arrow if col == section else "")
            if item is not None and item.text() != text:
                item.setText(text)
        self._fit_columns()

    def _outgrows(self, col: int, text: str) -> bool:
        """Note a new text in column ``col``: whether it is wider than any
        the column had (the Flight column is measured bold, as fresh rows
        show it)."""
        bold = col == AC_NAME
        metrics = self._metrics.get(bold)
        if metrics is None:
            font = QFont(self.table.font())
            font.setBold(bold)
            metrics = self._metrics[bold] = QFontMetrics(font)
        width = metrics.horizontalAdvance(text)
        if width <= self._widest.get(col, 0):
            return False
        self._widest[col] = width
        return True

    def _fit_column(self, col: int) -> None:
        if not self._fitting_held:
            self.table.resizeColumnToContents(col)

    def _fit_columns(self) -> None:
        """Fit every column to its contents and its title."""
        if not self._fitting_held:
            self.table.resizeColumnsToContents()

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
                text = self.detail_text(self.selected_icao())
                if text:
                    from PyQt6.QtGui import QGuiApplication

                    clipboard = QGuiApplication.clipboard()
                    if clipboard is not None:
                        clipboard.setText(text)
                    return True
        return super().eventFilter(obj, event)

    def _icao_at(self, row: int) -> str:
        item = self.table.item(row, AC_NAME)
        return str(item.data(_ICAO_ROLE) or "") if item is not None else ""

    def _rebuild(self, selected: str) -> None:
        """Rows again after aircraft were dropped (order kept), the aircraft
        ``selected`` still selected if it is there: quietly, so the
        Messages filter and its selection are left alone."""
        blocked = self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        held, self._fitting_held = self._fitting_held, True  # once, at the end
        try:
            self.table.setRowCount(0)
            self._items.clear()
            self._dimmed.clear()
            self._widest.clear()  # the widest values may have gone
            for aircraft in self._shown_tracker().aircraft():
                self._add_row(aircraft.icao)
                self._fill(aircraft)
            if selected in self._items:
                self.select(selected)
        finally:
            self._fitting_held = held
            self._fit_columns()
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(blocked)

    def _fill(self, aircraft: Aircraft) -> None:
        """Show an aircraft in its row. Only what changed is set: a value
        wider than any before in its column widens that column, and the
        color and font change only when the row dims or brightens."""
        name_item = self._items.get(aircraft.icao)
        if name_item is None:
            return
        # The row's cells, taken first: in a sorted list a changed value
        # moves the row.
        row = name_item.row()
        items = [self.table.item(row, col) for col in range(len(_AC_HEADERS))]
        now = self._clock()
        record = self._recordings.get(self._shown)
        stale = self._shown_tracker().is_stale(aircraft)
        if record is None:
            seen = format_age(now - aircraft.last_seen)
            seen_key = now - aircraft.last_seen
        else:
            # When the recording heard it last (the latest first, as the
            # live list's youngest).
            seen = format_recording_time(aircraft.last_seen, record.absolute)
            seen_key = -aircraft.last_seen
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
            AC_SEEN: seen,
        }
        # Sort keys of the columns of numbers (unknown values sort first).
        keys = {
            AC_ALTITUDE: 0.0 if aircraft.on_ground else _number(aircraft.altitude),
            AC_SPEED: _number(aircraft.speed),
            AC_SEEN: seen_key,
            AC_MESSAGES: float(aircraft.messages),
            AC_TRACK: _number(aircraft.track),
            AC_VS: _number(aircraft.vertical_rate),
        }
        was_stale = self._dimmed.get(aircraft.icao)
        restyle = was_stale is not stale
        self._dimmed[aircraft.icao] = stale
        brush = (
            QBrush(get_palette().qcolor("disabled" if stale else "text"))
            if restyle
            else None
        )
        tip = describe_aircraft(aircraft, now, self._heard_text(aircraft))
        if stale:
            tip += "\nNot heard for over a minute: it may be out of range."
        # Heard again: its name is bold again, so maybe wider.
        grown = [AC_NAME] if was_stale and not stale else []
        for col, text in values.items():
            item = items[col]
            if item is None:
                continue
            key = keys.get(col)
            if key is not None and item.data(_SORT_ROLE) != key:
                item.setData(_SORT_ROLE, key)  # (before the text: sorted once)
            if item.text() != text:
                item.setText(text)
                if self._outgrows(col, text) and col not in grown:
                    grown.append(col)
            item.setToolTip(tip)  # (an unchanged value costs nothing)
            if brush is not None:
                item.setForeground(brush)
                if col == AC_NAME:
                    font = QFont(item.font())
                    font.setBold(not stale)
                    item.setFont(font)
        for col in grown:  # measured in the row's font
            self._fit_column(col)

    def _on_selection(self) -> None:
        icao = self.selected_icao()
        aircraft = self._shown_tracker().get(icao)
        if aircraft is None:
            self.detail.hide()
        else:
            self._show_detail(aircraft)
        self.aircraft_selected.emit(icao)

    def _show_detail(self, aircraft: Aircraft) -> None:
        text = describe_aircraft(aircraft, self._clock(), self._heard_text(aircraft))
        if self.detail.text() != text:
            self.detail.setText(text)
        self.detail.show()


# --------------------------------------------------------------------------- #
# Morse code text
# --------------------------------------------------------------------------- #


@dataclass
class _RecordedText:
    """A recording's Morse text (see :class:`MorseTextView`): its lines,
    each with its time by the recording's clock, and the last speed."""

    absolute: bool
    latest: float = 0.0
    lines: List[str] = field(default_factory=list)
    wpm: float = 0.0


class MorseTextView(QWidget if HAS_PYQT6 else object):
    """The decoded Morse text as it arrives: finished lines, then the line
    being received (in the accent color), and the speed above.

    A recording's text (``add_line(text, source=name, ...)``, Decode
    Recording) is kept apart from what is heard live, a list per recording,
    each line after its time by the recording's clock; the source bar above
    picks the one shown (the newest recording when it comes).

    Another receiver's text (``add_line(text, receiver="2nd RX")``, the
    Second Receiver's, on a frequency of its own) is an entry of its own
    there too, after Live (:func:`receiver_source` names it), each line as
    it ends; it is shown when it starts if Live has nothing yet."""

    if HAS_PYQT6:
        # Its empty state's button was clicked.
        tune_requested = pyqtSignal()

    _MAX_LINES = 2000

    def __init__(self, parent: Any = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        # Each recording's text (Decode Recording), in the order they came,
        # and the one shown ("": live; a receiver's: receiver_source(mark)).
        self._recordings: Dict[str, _RecordedText] = {}
        self._shown = ""
        # Each other receiver's live text since Clear, by its mark ("2nd
        # RX"), in the order they came.
        self._receivers: Dict[str, _RecordedText] = {}
        # The live speed and whether a signal is heard (shown when live is).
        self._live_speed = ("", False)
        self._source_bar = SourceBar("Text")
        self._source_bar.chosen.connect(self.show_source)
        layout.addWidget(self._source_bar)

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
        # A recording's text, in place of the live text while it is shown.
        self._recorded = QPlainTextEdit()
        self._recorded.setReadOnly(True)
        self._recorded.setAccessibleName("Morse text of the recording")
        self._recorded.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._recorded.document().setMaximumBlockCount(self._MAX_LINES)
        set_role(self._recorded, "terminal")
        self._recorded.hide()
        layout.addWidget(self._recorded, 1)

        from .decoder_panel import TailFollower, ViewPlaceholder

        self._empty = ViewPlaceholder(
            self.text,
            "No Morse code yet\nTune to a CW signal in CW mode and press Start "
            "(Space). The demo device has a beacon on 99.820 MHz.\n"
            f"{TEXT_RECORDING_HINT}",
        )
        self._empty.action_triggered.connect(self.tune_requested.emit)
        self._tail = TailFollower(self.text.verticalScrollBar())
        self._recorded_tail = TailFollower(self._recorded.verticalScrollBar())
        self._live = ""
        self._has_live = False  # the document's last block is the live line
        self._lines = 0
        theme_notifier().theme_changed.connect(self._restyle_live)

    def add_line(
        self,
        text: str,
        source: str = "",
        at_s: Optional[float] = None,
        when: Optional[datetime] = None,
        wpm: float = 0.0,
        receiver: str = "",
    ) -> bool:
        """A finished line of text (the live line is cleared: it was this
        line; the next :meth:`set_live` shows what follows). From a
        recording (``source``: Decode Recording), it goes to that
        recording's text, after its time there (``when`` it was sent, if
        the recording says when it was made, else ``at_s`` seconds into it),
        at ``wpm``. True when it started the text of a recording (it is
        shown now).

        From another receiver (``receiver``: "2nd RX"), it goes to that
        receiver's text, at ``wpm``. True when it started that text (an
        entry of its own; shown if Live has nothing yet)."""
        source = str(source or "")
        if source:
            return self._recorded_line(str(text), source, at_s, when, wpm)
        receiver = str(receiver or "")
        if receiver:
            return self._receiver_line(str(text), receiver, wpm)
        self._live_line(text)
        return False

    def _receiver_line(self, text: str, mark: str, wpm: float) -> bool:
        record = self._receivers.get(mark)
        new = record is None
        if record is None:
            record = self._receivers[mark] = _RecordedText(absolute=False)
        record.lines.append(text)
        del record.lines[: -self._MAX_LINES]
        try:
            record.wpm = float(wpm or 0.0)
        except (TypeError, ValueError):
            record.wpm = 0.0
        key = receiver_source(mark)
        if new:
            # Shown when the live text has nothing (the main receiver decodes
            # no Morse): else Live stays, the entry beside it.
            live_empty = self.text.document().isEmpty()
            self.show_source(key if not self._shown and live_empty else self._shown)
            return True
        if self._shown == key:
            self._append_shown(text)
        return False

    def _append_shown(self, line: str) -> None:
        """A new line of the list shown (a recording's or a receiver's)."""
        cursor = QTextCursor(self._recorded.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._recorded.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(line, self._format(False))
        self._show_speed()

    def _recorded_line(
        self,
        text: str,
        source: str,
        at_s: Optional[float],
        when: Optional[datetime],
        wpm: float,
    ) -> bool:
        record = self._recordings.get(source)
        new = record is None
        if record is None:
            record = self._recordings[source] = _RecordedText(
                absolute=when is not None or at_s is None
            )
        heard = recording_time(record.absolute, at_s, when, record.latest)
        record.latest = max(record.latest, heard)
        line = f"{format_recording_time(heard, record.absolute)}  {text}"
        record.lines.append(line)
        del record.lines[: -self._MAX_LINES]
        try:
            record.wpm = float(wpm or 0.0)
        except (TypeError, ValueError):
            record.wpm = 0.0
        if new:
            self.show_source(source)  # (the newest decode: what was asked)
            return True
        if self._shown == source:
            self._append_shown(line)
        return False

    def _live_line(self, text: str) -> None:
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
        is heard now (live: a recording shown shows its own speed)."""
        self._live_speed = (str(wpm_text or ""), bool(receiving))
        self._show_speed()

    def _show_speed(self) -> None:
        record = self._list(self._shown)
        if record is None:
            wpm_text, receiving = self._live_speed
        else:
            wpm_text = f"{record.wpm:.0f} WPM" if record.wpm > 0 else ""
            receiving = False
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
        self._recordings.clear()
        self._receivers.clear()
        self.show_source("")
        self._source_bar.set_sources([], "")
        self._empty.set_visible(True)
        self.set_speed("", False)

    def forget_source(self, source: str) -> None:
        """Drop a recording's text (it is decoded again: it starts over);
        while it was shown, the live text is shown again."""
        if self._recordings.pop(str(source or ""), None) is None:
            return
        if self._shown == source:
            self.show_source("")
        else:
            self._offer_sources()

    def show_source(self, source: str) -> None:
        """Show the live text (""), a recording's (its name) or another
        receiver's (:func:`receiver_source` of its mark)."""
        source = str(source or "")
        record = self._list(source)
        if source and record is None:
            return
        changed = source != self._shown
        self._shown = source
        self._offer_sources()
        if record is not None and changed:
            self._recorded.setPlainText("\n".join(record.lines))
            self._recorded_tail.reset()
            self._recorded.moveCursor(QTextCursor.MoveOperation.End)
            whose = (
                source[len(_RECEIVER_SOURCE) :]
                if source.startswith(_RECEIVER_SOURCE)
                else "recording"
            )
            self._recorded.setAccessibleName(f"Morse text of the {whose}")
        self.text.setVisible(record is None)
        self._recorded.setVisible(record is not None)
        if record is None:
            self._recorded.clear()  # (rebuilt when shown again)
        self._show_speed()

    def _list(self, source: str) -> Optional[_RecordedText]:
        """The list ``source`` names: a recording's, another receiver's, or
        None (live, or no such list)."""
        if source.startswith(_RECEIVER_SOURCE):
            return self._receivers.get(source[len(_RECEIVER_SOURCE) :])
        return self._recordings.get(source)

    def _offer_sources(self) -> None:
        self._source_bar.set_sources(
            list(self._recordings), self._shown, list(self._receivers)
        )

    def sources(self) -> List[str]:
        """The recordings with text here, in the order they came."""
        return list(self._recordings)

    def shown_source(self) -> str:
        """The text shown: "" (live), a recording's name, or another
        receiver's :func:`receiver_source`."""
        return self._shown

    def has_recordings(self) -> bool:
        return bool(self._recordings)

    def other_receivers(self) -> List[str]:
        """The other receivers ("2nd RX") whose text is here, since Clear,
        in the order they came."""
        return list(self._receivers)

    def has_other_receivers(self) -> bool:
        return bool(self._receivers)

    def source_bar(self) -> "SourceBar":
        return self._source_bar

    def shown_text(self) -> str:
        """The text shown (live, or the recording's)."""
        if self._shown:
            return self._recorded.toPlainText()
        return self.text.toPlainText()

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

    def plain_text(self, source: str = "") -> str:
        """The live text (the line being received included), a recording's
        or another receiver's (:func:`receiver_source`; "" for one not
        here)."""
        if not source:
            return self.text.toPlainText()
        record = self._list(str(source))
        return "\n".join(record.lines) if record is not None else ""

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
