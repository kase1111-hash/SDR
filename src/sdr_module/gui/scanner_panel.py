"""
The Scanner tab: sweep a band, stop on signals, or scan the bookmarks.

A non-modal tool panel (key ``scanner``, Ctrl+8; Tools > Frequency Scanner,
Ctrl+F, brings it up). The scan runs in the window's display loop (see
:mod:`sdr_module.gui.scan_engine`): the spectrum, waterfall, audio and
decoders keep running, so a hit can be heard where the scan stops, or with
Listen (double-click, Enter) on a result.

Modes: Sweep once, Sweep continuously, Stop on signal (listen while the
squelch is open, go on after a few seconds of quiet; Skip and Hold) and
Scan bookmarks (each bookmark in its own mode, filter and tone squelch,
passing by those marked Skip). Band presets carry how their signals are
listened to. The threshold is Auto by default: 12 dB above the noise floor
measured on each step. Settings and the last results are kept between
sessions; results can be bookmarked (one or all) and exported as CSV.

The window is the panel's host (:class:`ScannerHost`): it tunes, starts
the receiver, says whether a device is connected or a recording runs, and
adds bookmarks.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from bisect import bisect_left, bisect_right, insort
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core.band_plan import KHZ_BELOW_HZ, band_for, listening_for, short_khz
from ..dsp.sweep import (
    AUTO_MARGIN_DB,
    WIDE_SIGNAL_HZ,
    channel_count,
    format_duration,
    page_count,
    same_signal,
)
from ..utils.tooltips import get_short_tip
from .scan_engine import (
    ALL_SKIPPED,
    DWELL,
    LISTEN,
    MEASURE_S,
    SCAN_MODES,
    Found,
    Listening,
    ScanEngine,
    ScanHost,
    ScanSettings,
)
from .scanner_dialog import _CUSTOM, SCAN_PRESETS, _format_freq, _NumericItem

if HAS_PYQT6:
    from . import file_dialogs
    from .decoder_panel import DecoderChip, ViewPlaceholder
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

# Results table columns, and the item data role holding a row's result.
COL_FREQ, COL_LEVEL, COL_SNR, COL_MODE, COL_HEARD = range(5)
_HIT_ROLE = Qt.ItemDataRole.UserRole + 1 if HAS_PYQT6 else 0

#: A scan whose pass takes longer than this asks before it starts.
CONFIRM_SECONDS = 120.0
#: Bookmark All of more results than this asks first, and files them in a
#: group of their own (removed together with the group's Remove Group).
CONFIRM_BOOKMARKS = 20
#: A running scan that gets no blocks for this long stops (and says why).
NO_BLOCKS_S = 3.0
#: Display frames per second (the window's display timer).
_FRAME_S = 1.0 / 30.0
#: How many results are kept between sessions.
_SAVED_RESULTS = 500
#: A column clicked during a scan keeps the results sorted (re-sorted by
#: the tab's refresh) up to this many rows; longer lists are sorted once
#: per click, and again when the scan ends (a sort of 2000 takes ~50 ms).
_LIVE_SORT_ROWS = 1000
#: The fields' ranges.
_FREQ_RANGE_MHZ = (0.1, 6000.0)
_STEP_RANGE_KHZ = (1.0, 10000.0)
_RESUME_RANGE_S = (0.5, 60.0)
#: Saved result times later than this (the year 3000), before 1970 or NaN
#: (damaged settings) read as "never": no clock shows them.
_LAST_TIMESTAMP = 32503680000.0

_EMPTY_TEXTS = {
    "idle": "No results yet\nChoose a band or range and click Start Scan.",
    "scanning": (
        "Scanning for signals\n"
        "Those above the threshold appear here as they are found."
    ),
    "listening": (
        "Nothing heard yet\n" "The scan stops on each active channel and lists it here."
    ),
    "nothing": (
        "No signals above the threshold\n"
        "Lower the threshold (or tick Auto) or pick another range, then scan "
        "again."
    ),
    "cleared": "Results cleared\nClick Start Scan to scan again.",
    "stopped": "Stopped before any signal was found\nClick Start Scan to scan again.",
    "device": (
        "Connect a device to scan\n"
        "Use Demo Device tries it with simulated signals; Connect... opens an "
        "RTL-SDR or HackRF."
    ),
    "no_samples": (
        "The receiver returned no samples\n"
        "Check that it is plugged in and not used by another program."
    ),
    "refused": (
        "The receiver could not tune this range\n"
        "Choose a range inside what it tunes."
    ),
}


@dataclass
class ScanHit:
    """One row of the results: a signal found (or a channel heard)."""

    frequency_hz: float
    peak_db: float
    snr_db: float
    width_hz: float
    mode: str = ""
    fm_deviation: str = ""
    bandwidth: str = ""
    label: str = ""  # a bookmark's name, or ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 1
    tone_hz: float = 0.0  # a bookmark's CTCSS tone squelch (0: none)

    @property
    def listening(self) -> Listening:
        return self.mode, self.fm_deviation, self.bandwidth

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["ScanHit"]:
        try:
            hit = cls(
                frequency_hz=float(data["frequency_hz"]),
                peak_db=float(data.get("peak_db", -120.0)),
                snr_db=float(data.get("snr_db", 0.0)),
                width_hz=float(data.get("width_hz", 0.0)),
                mode=str(data.get("mode", "") or ""),
                fm_deviation=str(data.get("fm_deviation", "") or ""),
                bandwidth=str(data.get("bandwidth", "") or ""),
                label=str(data.get("label", "") or ""),
                first_seen=float(data.get("first_seen", 0.0)),
                last_seen=float(data.get("last_seen", 0.0)),
                count=int(data.get("count", 1)),
                tone_hz=float(data.get("tone_hz", 0.0) or 0.0),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(hit.frequency_hz) and hit.frequency_hz > 0):
            return None
        hit.first_seen = _valid_time(hit.first_seen)
        hit.last_seen = _valid_time(hit.last_seen)
        if not (math.isfinite(hit.tone_hz) and hit.tone_hz > 0):
            hit.tone_hz = 0.0
        return hit


def _band_name(label: str) -> str:
    """``"NOAA Weather"`` for ``"NOAA Weather (162.4–162.55 MHz)"``."""
    return label.split(" (", 1)[0]


def _say_freq(freq_hz: float) -> str:
    """A frequency in a sentence, as the window names it: ``"146.5125
    MHz"``; below 2 MHz in kHz (``"660 kHz"``, as FREQ shows it). The
    results column and the range fields stay in MHz (:func:`_format_freq`)."""
    if abs(float(freq_hz)) < KHZ_BELOW_HZ:
        return f"{short_khz(freq_hz)} kHz"
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def _bookmarked(hit: ScanHit) -> str:
    """What bookmarking one result says, as Ctrl+B and the Signal tab's
    Bookmark say it: the name the Bookmarks tab lists it under (the host
    names it so), its mode, and how to rename it."""
    from .bookmarks_panel import default_bookmark_name

    name = hit.label or default_bookmark_name(hit.frequency_hz)
    mode = "I/Q" if hit.mode.startswith("None") else hit.mode
    how = f" ({mode})" if mode else ""
    return f"Bookmarked {name}{how}. Rename it with F2 in the Bookmarks tab."


def _valid_time(timestamp: float) -> float:
    """A saved time (seconds since 1970), 0 ("never") if no clock shows it."""
    return timestamp if 0.0 <= timestamp < _LAST_TIMESTAMP else 0.0


def mode_text(mode: str, deviation: str = "") -> str:
    """Short mode text for the results: "AM", "WFM", "FM", "I/Q"."""
    if mode == "FM":
        return "WFM" if deviation == "75 kHz" else "FM"
    if mode.startswith("None"):
        return "I/Q"
    return mode


def guess_listening(frequency_hz: float, width_hz: float) -> Listening:
    """How to listen to a signal of a custom range: its band's (or
    preset's) usual mode; else broadcast FM for a wide signal; else
    "" (keep the receiver's mode)."""
    listening = listening_for(frequency_hz)
    if listening is not None:
        return listening
    if width_hz >= WIDE_SIGNAL_HZ:
        return "FM", "75 kHz", "200 kHz"
    return "", "", ""


class ScannerHost(ScanHost):
    """What the Scanner tab needs from the window, besides tuning (see
    :class:`~sdr_module.gui.scan_engine.ScanHost`)."""

    def scan_blocker(self) -> Tuple[str, str]:
        """``("", "")`` when a scan can start, else ``(kind, reason)``:
        kind "device" (none connected), "recording", "locked" (a
        recording plays back) or "transmitting"."""
        return "", ""

    def scan_receiving(self) -> bool:
        """Whether the receiver runs (blocks reach the scanner)."""
        return True

    def scan_start_receiver(self) -> bool:
        return True

    def scan_sample_rate(self) -> float:
        return 2.4e6

    def scan_settle_s(self) -> float:
        """Extra settling time after a retune (hardware buffers)."""
        return 0.0

    def scan_noise_floor_db(self) -> Optional[float]:
        """The spectrum's noise floor per bin now (dBFS), if known."""
        return None

    def scan_device_range(self) -> Optional[Tuple[float, float]]:
        return None

    def scan_begin(self) -> None:
        """A scan starts: remember where the receiver is tuned."""

    def scan_end(self, stay: bool) -> None:
        """A scan ended: back to where it started, unless ``stay`` (it
        stopped on a channel the user is listening to)."""

    def scan_listen(self, hit: ScanHit) -> bool:
        """Tune to a result in its mode (Listen)."""
        return False

    def scan_add_bookmarks(
        self, hits: List[ScanHit], group: str = ""
    ) -> Tuple[int, List[str]]:
        """Bookmark results (in ``group``, when given): ``(added, names
        already bookmarked)``."""
        return 0, []

    def scan_bookmark_list(self, group: str = "") -> List[Dict[str, Any]]:
        """The bookmarks a bookmark scan visits (see
        :func:`~sdr_module.gui.scan_engine.bookmark_scan_list`)."""
        return []

    def scan_bookmark_order(self) -> List[int]:
        """Every bookmark's index in the order the list shows them (the
        order a bookmark scan visits them); [] when it is the stored one."""
        return []

    def scan_bookmark_groups(self) -> List[str]:
        return []

    def scan_bookmark_counts(self) -> Tuple[int, int]:
        """``(bookmarks, marked Skip)``."""
        return 0, 0

    def scan_use_demo(self) -> None:
        pass

    def scan_connect(self) -> None:
        pass

    def scan_message(self, text: str, tone: Optional[str] = None) -> None:
        """Show a status-bar message."""

    def scan_show_panel(self) -> None:
        """Bring up the Scanner tab."""


class _ResultsTable(QTableWidget if HAS_PYQT6 else object):
    """The results list: it takes the height the tab has left (scrolling
    its rows) instead of asking for a table's usual 192 px, which in the
    short tab of a 1366x768 window pushed the buttons under it out of
    view."""

    def sizeHint(self):  # noqa: N802 (Qt API)
        hint = super().sizeHint()
        return QSize(hint.width(), self.minimumHeight())


class ScannerPanel(QWidget if HAS_PYQT6 else object):
    """The Scanner tab (see the module docstring)."""

    def __init__(self, parent=None, settings: Any = None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._settings = settings
        self._host: ScannerHost = ScannerHost()
        self._engine = ScanEngine(self._host, self._guess)
        self._connect_engine()
        self._hits: List[ScanHit] = []
        # The results by frequency, (frequency, index in _hits), the widest
        # one's width and each one's Frequency cell: a reading finds the
        # result it repeats, and its row, without visiting them all (in the
        # display loop, with thousands listed). See _rebuild_table.
        self._by_freq: List[Tuple[float, int]] = []
        self._widest_hz = 0.0
        self._cells: Dict[int, Any] = {}
        self._empty_reason = "idle"
        self._restoring = False
        self._blocker: Tuple[str, str] = ("", "")
        self._stay = False  # end the scan on the channel listened to
        self._restarting = False  # a new scan replaces the one running
        self._status_tone: Optional[str] = None
        # During a scan the table doesn't sort itself on each new result:
        # whether a column was clicked (then the refresh re-sorts), and
        # whether results came since the last sort.
        self._sort_live = False
        self._unsorted = False
        # Scan Bookmarks from Here: the tab's own (mode, group) to show
        # again when it ends (they are not saved meanwhile).
        self._mode_before_from_here: Optional[Tuple[str, str]] = None
        # The bookmark visits and counts of the group chosen, as the host
        # last gave them (refreshed when the bookmarks change).
        self._visits_cache: Optional[
            Tuple[str, List[Dict[str, Any]], Tuple[int, int]]
        ] = None
        self._setup_ui()
        # (See _show_results_buttons: after the layout's next pass.)
        self._buttons_timer = QTimer(self)
        self._buttons_timer.setSingleShot(True)
        self._buttons_timer.setInterval(0)
        self._buttons_timer.timeout.connect(self._ensure_buttons_visible)
        self._setup_chip()
        self._restore_settings()
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self._update_steps_hint()
        self._update_controls()

    # ------------------------------------------------------------------ #
    # Host
    # ------------------------------------------------------------------ #
    def set_host(self, host: ScannerHost) -> None:
        """Connect the panel to the window."""
        self._host = host
        self._visits_cache = None
        self._engine = ScanEngine(host, self._guess)
        self._connect_engine()
        # The groups are the window's bookmarks': now the saved one can be
        # chosen again.
        self._refresh_groups(prefer=getattr(self, "_saved_group", ""))
        self.refresh()

    @property
    def engine(self) -> ScanEngine:
        return self._engine

    def _connect_engine(self) -> None:
        self._engine.on_found = self._on_found
        self._engine.on_state = self._on_engine_state
        self._engine.on_finished = self._on_finished

    def _guess(self, reading: Any) -> Listening:
        return guess_listening(reading.frequency_hz, reading.width_hz)

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Band and mode side by side (their tooltips and accessible names
        # say what they are), then the range, step and threshold: compact
        # enough for the 380 px column of a 1024x640 window.
        top = QHBoxLayout()
        top.setSpacing(6)
        self._preset = QComboBox()
        self._preset.addItem(_CUSTOM)
        for preset in SCAN_PRESETS:
            # The band's name: "NOAA Weather (162.4–162.55 MHz)" didn't fit
            # the column. Its range is in the tooltip (and Start / End);
            # the full label is what is saved, as before.
            self._preset.addItem(_band_name(preset.label), preset.label)
            self._preset.setItemData(
                self._preset.count() - 1, preset.label, Qt.ItemDataRole.ToolTipRole
            )
        self._preset.setToolTip(
            "Band: fills in the range and step for a common band; signals found "
            "there are listened to in the band's mode (airband: AM)"
        )
        self._preset.setAccessibleName("Band")
        self._preset.currentIndexChanged.connect(self._on_preset_changed)
        top.addWidget(self._preset, 1)

        self._mode = QComboBox()
        for key, text in SCAN_MODES:
            self._mode.addItem(text, key)
        self._mode.setItemData(
            0,
            "One pass over the range, listing every signal found",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode.setItemData(
            1,
            "Pass over the range again and again (finds signals that come and "
            "go), until you stop it",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode.setItemData(
            2,
            "Sweep, and stop on each signal: listen while the squelch is open, "
            "go on after the Resume time of quiet",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode.setItemData(
            3,
            "Tune each bookmark in its own mode and stop on the active ones "
            "(bookmarks marked Skip are passed by)",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._mode.setToolTip(
            "What the scan does: sweep the range once or again and again, stop "
            "on each signal to listen, or scan your bookmarks"
        )
        self._mode.setAccessibleName("Scan mode")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        top.addWidget(self._mode, 1)
        for combo in (self._preset, self._mode):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._start = self._spin(*_FREQ_RANGE_MHZ, 3, 88.0, " MHz")
        self._start.setToolTip("First frequency to scan")
        self._add_field(grid, 0, 0, "&Start:", self._start)
        self._end = self._spin(*_FREQ_RANGE_MHZ, 3, 108.0, " MHz")
        self._end.setToolTip("Last frequency to scan")
        self._add_field(grid, 0, 2, "&End:", self._end)

        self._step = self._spin(*_STEP_RANGE_KHZ, 1, 200.0, " kHz")
        self._step.setToolTip(
            "Channel spacing: each channel reports the strongest signal within "
            "half a step either side.\nUse the band's channel spacing: 200 kHz "
            "for FM broadcast, 12.5-25 kHz for voice."
        )
        self._add_field(grid, 1, 0, "Ste&p:", self._step)

        # (Mnemonics: none of the menu bar's, F D R V T H, which Alt+ opens
        # while this tab shows.)
        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(4)
        self._auto = QCheckBox("A&uto")
        self._auto.setChecked(True)
        self._auto.setToolTip(
            f"Auto threshold: measure the noise floor on each step and list the "
            f"signals {AUTO_MARGIN_DB:.0f} dB above it (recommended)"
        )
        threshold_row.addWidget(self._auto)
        self._threshold = self._spin(-140.0, 0.0, 0, -76.0, " dBFS")
        self._threshold.setToolTip(
            "Only signals whose peak is above this level are listed: set it "
            "10-20 dB above the noise floor shown on the spectrum (untick "
            "Auto to set it; with Auto it shows the level in use)"
        )
        self._threshold.setAccessibleName("Threshold")
        self._threshold.setMinimumWidth(72)
        threshold_row.addWidget(self._threshold, 1)
        label = QLabel("Threshold:")
        label.setBuddy(self._auto)
        grid.addWidget(label, 1, 2)
        grid.addLayout(threshold_row, 1, 3)

        self._resume = self._spin(*_RESUME_RANGE_S, 1, 3.0, " s")
        self._resume.setToolTip(
            "Stop on signal and Scan bookmarks: go on after the channel has "
            "been quiet (squelch closed) this long"
        )
        self._resume_label = self._add_field(grid, 2, 0, "Resu&me:", self._resume)

        self._group = QComboBox()
        self._group.setToolTip("Scan bookmarks: all bookmarks, or one group")
        self._group.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._group.setMinimumContentsLength(6)
        self._group_label = self._add_field(grid, 2, 2, "&Group:", self._group)
        layout.addLayout(grid)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        set_role(self._hint, "hint")
        layout.addWidget(self._hint)

        for spin in (self._start, self._end, self._step):
            spin.valueChanged.connect(self._on_range_edited)
        self._threshold.valueChanged.connect(self._on_setting_changed)
        self._auto.toggled.connect(self._on_setting_changed)
        self._resume.valueChanged.connect(self._on_setting_changed)
        self._group.currentIndexChanged.connect(self._on_setting_changed)

        # Start/Stop (or, with no device, Use Demo Device / Connect...),
        # with the status beside it.
        self._run_row = QWidget()
        run_row = QHBoxLayout(self._run_row)
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(6)
        self._start_btn = QPushButton()
        widths = []
        for text, role in (("Start Scan", "primary"), ("Stop Scan", "danger")):
            self._start_btn.setText(text)
            set_role(self._start_btn, role)
            widths.append(self._start_btn.sizeHint().width())
        # Same width in both states, so the status text beside it stays put.
        self._start_btn.setMinimumWidth(max(widths) + 8)
        self._start_btn.setAutoDefault(False)
        self._start_btn.clicked.connect(self.toggle_scan)
        run_row.addWidget(self._start_btn)
        self._demo_btn = QPushButton("Use Demo Device")
        set_role(self._demo_btn, "primary")
        self._demo_btn.setToolTip("Try the scanner with the simulated demo device")
        self._demo_btn.clicked.connect(lambda _c=False: self._host.scan_use_demo())
        run_row.addWidget(self._demo_btn)
        self._connect_btn = QPushButton("Connect...")
        self._connect_btn.setToolTip("Choose and open an RTL-SDR or HackRF One")
        self._connect_btn.clicked.connect(lambda _c=False: self._host.scan_connect())
        run_row.addWidget(self._connect_btn)
        for btn in (self._demo_btn, self._connect_btn):
            btn.setAutoDefault(False)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        run_row.addWidget(self._status, 1)
        layout.addWidget(self._run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(6)
        self._progress.setAccessibleName("Scan progress")
        layout.addWidget(self._progress)

        # Stopped on a channel: what it is, Skip and Hold.
        self._listen_row = QWidget()
        listen_layout = QHBoxLayout(self._listen_row)
        listen_layout.setContentsMargins(0, 0, 0, 0)
        listen_layout.setSpacing(6)
        # (Ends in "…" where Skip and Hold leave too little room for a long
        # name; the tooltip has it all.)
        from .signal_panel import _ElidedLabel

        self._listen_label = _ElidedLabel(mode=Qt.TextElideMode.ElideRight)
        set_role(self._listen_label, "value")
        listen_layout.addWidget(self._listen_label, 1)
        self._skip_btn = QPushButton("S&kip")
        self._skip_btn.setToolTip(
            "Go on now, and don't stop on this channel again during this scan"
        )
        self._skip_btn.clicked.connect(lambda _c=False: self._engine.skip())
        listen_layout.addWidget(self._skip_btn)
        self._hold_btn = QPushButton("H&old")
        self._hold_btn.setCheckable(True)
        self._hold_btn.setToolTip(
            "Stay on this channel even when it goes quiet; click again to let "
            "the scan go on"
        )
        self._hold_btn.toggled.connect(self._engine_hold)
        listen_layout.addWidget(self._hold_btn)
        for btn in (self._skip_btn, self._hold_btn):
            btn.setAutoDefault(False)
        self._listen_row.setVisible(False)
        layout.addWidget(self._listen_row)

        # Results
        self._table = _ResultsTable(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Frequency", "Level", "SNR", "Mode", "Heard"]
        )
        header = self._table.horizontalHeader()
        # One line per result; the small columns as wide as their widest
        # text (_size_columns), the frequency takes what is left.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_FREQ, QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(30)
        self._table.setWordWrap(False)
        header.setSortIndicator(COL_FREQ, Qt.SortOrder.AscendingOrder)
        header.sortIndicatorChanged.connect(self._on_sort_changed)
        header_tips = (
            "Signal center frequency, to the nearest kHz (a broadcast FM "
            "station: its 100 kHz channel); a bookmark scan's results also "
            "name the bookmark",
            "Strongest FFT bin, in dB relative to full scale (same scale as "
            "the spectrum)",
            get_short_tip("snr"),
            "How Listen tunes it: the band's mode (or the bookmark's)",
            "When it was last found or heard (hover for how often)",
        )
        for col, tip in enumerate(header_tips):
            item = self._table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
                if col in (COL_FREQ, COL_LEVEL, COL_SNR):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Arrow keys pick a result and Tab moves on to the buttons.
        self._table.setTabKeyNavigation(False)
        self._table.setAccessibleName("Scan results")
        self._table.setToolTip(
            "Double-click a signal, or press Enter, to listen to it; "
            "right-click for more"
        )
        self._table.setMinimumHeight(90)
        self._table.itemSelectionChanged.connect(self._update_controls)
        self._table.itemDoubleClicked.connect(lambda _item: self.listen_selected())
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.installEventFilter(self)
        self._size_columns()
        layout.addWidget(self._table, 1)
        # Empty states: a bold title over what to do next.
        self._empty = ViewPlaceholder(self._table)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self._listen_btn = QPushButton("&Listen")
        self._listen_btn.clicked.connect(lambda _c=False: self.listen_selected())
        buttons.addWidget(self._listen_btn)
        self._bookmark_btn = QPushButton("&Bookmark")
        self._bookmark_btn.clicked.connect(lambda _c=False: self.bookmark_selected())
        buttons.addWidget(self._bookmark_btn)
        self._bookmark_all_btn = QPushButton("Bookmark &All")
        self._bookmark_all_btn.clicked.connect(lambda _c=False: self.bookmark_all())
        buttons.addWidget(self._bookmark_all_btn)
        buttons.addStretch(1)
        self._results_menu = QMenu(self)
        self._results_menu.setToolTipsVisible(True)
        self._export_action = self._results_menu.addAction("Export Results (CSV)...")
        self._export_action.setToolTip(
            "Save the results as a CSV file (frequency, level, SNR, mode, when "
            "heard)"
        )
        self._export_action.triggered.connect(lambda _c=False: self.export_csv())
        self._clear_action = self._results_menu.addAction("Clear Results")
        self._clear_action.setToolTip("Empty the results list")
        self._clear_action.triggered.connect(lambda _c=False: self.clear_results())
        # (Like the waterfall's options button: compact, a menu.)
        self._results_btn = QPushButton("⋯")
        set_role(self._results_btn, "compact")
        self._results_btn.setMenu(self._results_menu)
        self._results_btn.setAccessibleName("Results options")
        self._results_btn.setToolTip("Export the results as CSV, or clear them")
        buttons.addWidget(self._results_btn)
        for btn in (
            self._listen_btn,
            self._bookmark_btn,
            self._bookmark_all_btn,
            self._results_btn,
        ):
            btn.setAutoDefault(False)
        layout.addLayout(buttons)

    def _setup_chip(self) -> None:
        """The chip the window shows in its status bar while a scan runs
        (see :meth:`status_chip`): what the scan is doing, and a menu to
        stop, skip or hold it, or show this tab, from any tab. Short of
        room (AUDIO, AFC and the decoder chip at 1024 px) it is cut, as the
        decoder chip is, not the window widened (P-03)."""
        chip = DecoderChip("")
        set_role(chip, "compact")
        chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        chip.setAccessibleName("Scan in progress")
        menu = QMenu(chip)
        menu.setToolTipsVisible(True)
        stop = menu.addAction("Stop Scan")
        stop.setToolTip("Stop the scan")
        stop.triggered.connect(lambda _c=False: self.stop_scan())
        self._chip_skip = menu.addAction("Skip")
        self._chip_skip.setToolTip(self._skip_btn.toolTip())
        self._chip_skip.triggered.connect(lambda _c=False: self._engine.skip())
        self._chip_hold = menu.addAction("Hold")
        self._chip_hold.setCheckable(True)
        self._chip_hold.setToolTip(self._hold_btn.toolTip())
        self._chip_hold.triggered.connect(
            lambda on=False: self._hold_btn.setChecked(bool(on))
        )
        menu.addSeparator()
        show = menu.addAction("Show Scanner")
        show.setToolTip("Bring up the Scanner tab (Ctrl+F)")
        show.triggered.connect(lambda _c=False: self._host.scan_show_panel())
        chip.setMenu(menu)
        chip.hide()
        self._chip = chip

    def status_chip(self) -> "QPushButton":
        """The status-bar chip (the window adds it to its status bar)."""
        return self._chip

    def _update_chip(self) -> None:
        engine = self._engine
        chip = self._chip
        if not engine.active:
            if not chip.isHidden():
                chip.hide()
            return
        current = engine.current
        on_channel = engine.state in (DWELL, LISTEN) and current is not None
        if on_channel:
            # (Whole: short of room the chip is cut as it is drawn, and its
            # tooltip then starts with the whole text, the whole name too.)
            where = current.label or _say_freq(current.frequency_hz)
            text = f"Scanning · {where}" + (" · held" if engine.held else "")
        else:
            count = len(self._hits)
            verb = "heard" if engine.mode in ("stop", "bookmarks") else "found"
            text = f"Scanning · {count:,} {verb}"
        text = text.replace("&", "&&")  # (a name's "&" is no mnemonic)
        if chip.text() != text:
            chip.setText(text)
        tip = (
            f"{self._scanning_text()}. Click for Stop, Skip and Hold, or to "
            "show the Scanner tab (Ctrl+F)."
        )
        if chip.toolTip() != tip:
            chip.setToolTip(tip)
        can_skip = on_channel and engine.mode in ("stop", "bookmarks")
        self._chip_skip.setEnabled(can_skip)
        self._chip_hold.setEnabled(can_skip)
        self._chip_hold.setChecked(engine.held)
        if chip.isHidden():
            chip.show()

    def _size_columns(self) -> None:
        """Level, SNR, Mode and Heard as wide as their widest text."""
        cells = self._table.fontMetrics()
        header = self._table.horizontalHeader()
        titles = header.fontMetrics()
        widest = {
            COL_LEVEL: ("Level", ("-120 dBFS",)),
            COL_SNR: ("SNR", ("120 dB",)),
            COL_MODE: ("Mode", ("WFM", "USB", "I/Q")),
            COL_HEARD: ("Heard", ("00:00:00",)),
        }
        for col, (title, texts) in widest.items():
            width = max(
                titles.horizontalAdvance(title) + 20,
                max(cells.horizontalAdvance(t) for t in texts) + 14,
            )
            header.resizeSection(col, width)

    def changeEvent(self, event):  # noqa: N802 (Qt API)
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            if getattr(self, "_table", None) is not None:
                self._size_columns()

    @staticmethod
    def _add_field(grid, row, col, text, field, span=1) -> "QLabel":
        label = QLabel(text)
        label.setBuddy(field)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label, row, col)
        grid.addWidget(field, row, col + 1, 1, span)
        return label

    @staticmethod
    def _spin(low, high, decimals, value, suffix) -> "QDoubleSpinBox":
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.setKeyboardTracking(False)
        # (Its size hint fits "6000.000 MHz"; the column fits "108.000 MHz".)
        spin.setMinimumWidth(84)
        return spin

    def _set_status(self, text: str, tone: Optional[str] = None) -> None:
        self._status.setText(text)
        self._status.setToolTip(text)
        set_tone(self._status, tone)
        self._status_tone = tone

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        """Enter in the results listens to the selected signal."""
        if (
            obj is getattr(self, "_table", None)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and self._table.rowCount()
        ):
            if not self._table.selectionModel().selectedRows():
                row = self._table.currentRow()
                if row >= 0:
                    self._table.selectRow(row)
            self.listen_selected()
            return True
        return super().eventFilter(obj, event)

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event):  # noqa: N802 (Qt API)
        super().hideEvent(event)
        if not self._engine.active:
            self._timer.stop()

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def _setting(self, getter: str, key: str, default: Any) -> Any:
        settings = self._settings
        if settings is None:
            return default
        try:
            return getattr(settings, getter)(f"scanner/{key}", default)
        except Exception:  # pragma: no cover - a damaged setting
            return default

    def _save(self, key: str, value: Any) -> None:
        if self._settings is None or self._restoring:
            return
        try:
            self._settings.set(f"scanner/{key}", value)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not save scanner setting %s: %s", key, exc)

    def _restore_settings(self) -> None:
        self._restoring = True
        try:
            start = self._setting("get_float", "start_mhz", 88.0)
            end = self._setting("get_float", "end_mhz", 108.0)
            step = self._setting("get_float", "step_khz", 200.0)
            for spin, value in (
                (self._start, start),
                (self._end, end),
                (self._step, step),
            ):
                if math.isfinite(value):
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)
            preset = self._setting("get_str", "preset", SCAN_PRESETS[0].label)
            index = self._preset.findData(preset)
            if index < 0:
                index = self._preset.findText(preset)
            self._preset.blockSignals(True)
            self._preset.setCurrentIndex(index if index >= 0 else 0)
            self._preset.blockSignals(False)
            mode = self._setting("get_str", "mode", "once")
            index = self._mode.findData(mode)
            self._mode.setCurrentIndex(max(index, 0))
            self._auto.setChecked(
                bool(self._setting("get_bool", "auto_threshold", True))
            )
            threshold = self._setting("get_float", "threshold_db", -76.0)
            if math.isfinite(threshold):
                self._threshold.setValue(threshold)
            resume = self._setting("get_float", "resume_s", 3.0)
            if math.isfinite(resume):
                self._resume.setValue(resume)
            self._saved_group = self._setting("get_str", "bookmark_group", "")
            raw = self._setting("get_str", "results", "")
            self._hits = []
            if raw:
                try:
                    data = json.loads(raw)
                except ValueError:
                    data = []
                if isinstance(data, list):
                    for entry in data[:_SAVED_RESULTS]:
                        hit = (
                            ScanHit.from_dict(entry)
                            if isinstance(entry, dict)
                            else None
                        )
                        if hit is not None:
                            self._hits.append(hit)
            self._rebuild_table()
            self._table.setSortingEnabled(True)
        finally:
            self._restoring = False
        self._on_mode_changed()

    def _save_range(self) -> None:
        self._save("start_mhz", self._start.value())
        self._save("end_mhz", self._end.value())
        self._save("step_khz", self._step.value())
        label = self._preset.currentData()
        self._save(
            "preset", label if isinstance(label, str) else self._preset.currentText()
        )

    def _save_results(self) -> None:
        recent = sorted(self._hits, key=lambda h: -h.last_seen)[:_SAVED_RESULTS]
        try:
            self._save("results", json.dumps([h.to_dict() for h in recent]))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Could not save the scan results: %s", exc)

    # ------------------------------------------------------------------ #
    # Range and mode
    # ------------------------------------------------------------------ #
    def preset(self) -> Optional[Any]:
        index = self._preset.currentIndex()
        return SCAN_PRESETS[index - 1] if index > 0 else None

    def scan_mode(self) -> str:
        value = self._mode.currentData()
        return value if isinstance(value, str) else "once"

    def set_scan_mode(self, mode: str) -> None:
        index = self._mode.findData(mode)
        if index >= 0:
            self._mode.setCurrentIndex(index)

    def _on_preset_changed(self, index: int) -> None:
        preset = self.preset()
        if preset is not None:
            for spin, value in (
                (self._start, preset.start_mhz),
                (self._end, preset.end_mhz),
                (self._step, preset.step_khz),
            ):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._save_range()
        self._update_steps_hint()
        self._update_controls()

    def _on_range_edited(self) -> None:
        # Hand-edited values no longer match the chosen band.
        if self._preset.currentIndex() > 0:
            self._preset.blockSignals(True)
            self._preset.setCurrentIndex(0)
            self._preset.blockSignals(False)
        self._save_range()
        self._update_steps_hint()
        self._update_controls()

    def _on_mode_changed(self, _index: int = 0) -> None:
        self._save("mode", self.scan_mode())
        self._refresh_groups()
        self._update_steps_hint()
        self._update_controls()

    def _on_setting_changed(self, *_args) -> None:
        self._save("auto_threshold", self._auto.isChecked())
        if not self._auto.isChecked():
            self._save("threshold_db", self._threshold.value())
        self._save("resume_s", self._resume.value())
        group = self._group.currentData()
        if isinstance(group, str) and self._group.count():
            self._save("bookmark_group", group)
        self._update_steps_hint()
        self._update_controls()

    def _refresh_groups(self, prefer: Optional[str] = None) -> None:
        """The Group list: all bookmarks, or one of their groups (keeping
        the one chosen, or ``prefer``)."""
        wanted = prefer if prefer is not None else self._group.currentData()
        if not isinstance(wanted, str):
            wanted = getattr(self, "_saved_group", "")
        self._group.blockSignals(True)
        self._group.clear()
        # ("All": "All bookmarks" didn't fit the column; "Group: All".)
        self._group.addItem("All", "")
        self._group.setItemData(0, "All bookmarks", Qt.ItemDataRole.ToolTipRole)
        for group in self._host.scan_bookmark_groups():
            self._group.addItem(group, group)
        index = self._group.findData(wanted)
        self._group.setCurrentIndex(max(index, 0))
        self._group.blockSignals(False)

    def bookmarks_changed(self) -> None:
        """The bookmarks changed (the window calls this)."""
        self._visits_cache = None
        self._refresh_groups()
        self._update_steps_hint()
        self._update_controls()

    def _bookmark_visits(self) -> List[Dict[str, Any]]:
        """The bookmarks a bookmark scan of the chosen group visits. Kept
        until the bookmarks change: the tab's 500 ms refresh built the
        list twice over every bookmark (66 ms with 10,000)."""
        group = self._group_choice()
        cache = self._visits_cache
        if cache is None or cache[0] != group:
            cache = (
                group,
                self._host.scan_bookmark_list(group),
                self._host.scan_bookmark_counts(),
            )
            self._visits_cache = cache
        return cache[1]

    def _bookmark_counts(self) -> Tuple[int, int]:
        self._bookmark_visits()
        return self._visits_cache[2] if self._visits_cache else (0, 0)

    def _range_valid(self) -> bool:
        return self._end.value() > self._start.value()

    def _planned_sweep(self) -> Tuple[Optional[Tuple[float, float]], str]:
        """The range that will be swept, in Hz, and a note; ``(None,
        reason)`` when none of it can be tuned."""
        start_hz = self._start.value() * 1e6
        end_hz = self._end.value() * 1e6
        limits = self._host.scan_device_range()
        if limits is None:
            return (start_hz, end_hz), ""
        low, high = limits
        if end_hz < low or start_hz > high:
            return None, (
                f"This device tunes {_say_freq(low)} to {_say_freq(high)}. "
                "Choose a range inside it."
            )
        if start_hz < low or end_hz > high:
            start_hz, end_hz = max(start_hz, low), min(end_hz, high)
            return (start_hz, end_hz), (
                f"Limited to the device's range, {_say_freq(start_hz)} to "
                f"{_say_freq(end_hz)}."
            )
        return (start_hz, end_hz), ""

    def _page_seconds(self) -> float:
        """About how long one receiver step takes: the block after the
        retune is dropped, the tuner settles, then a block is measured."""
        frames = 1 + max(1, math.ceil(MEASURE_S / _FRAME_S))
        return frames * _FRAME_S + float(self._host.scan_settle_s())

    def estimate(self) -> Tuple[int, int, float]:
        """``(channels, receiver steps, seconds)`` of one pass."""
        sweep, _note = self._planned_sweep()
        if sweep is None or not self._range_valid():
            return 0, 0, 0.0
        step_hz = self._step.value() * 1e3
        rate = self._host.scan_sample_rate()
        channels = channel_count(sweep[0], sweep[1], step_hz)
        steps = page_count(sweep[0], sweep[1], step_hz, rate)
        return channels, steps, steps * self._page_seconds()

    def _auto_level(self) -> Optional[Tuple[float, float]]:
        """``(noise floor, threshold)`` Auto uses now, in dBFS, if known:
        the scan's last step, else the spectrum's noise floor."""
        noise = self._engine.noise_db if self._engine.active else None
        if noise is None or not math.isfinite(noise):
            noise = self._host.scan_noise_floor_db()
        if noise is None or not math.isfinite(noise):
            return None
        return noise, noise + AUTO_MARGIN_DB

    def _auto_text(self) -> str:
        level = self._auto_level()
        if level is None:
            return f"Auto: {AUTO_MARGIN_DB:.0f} dB over the noise floor."
        # The threshold field shows the level in use.
        self._threshold.blockSignals(True)
        self._threshold.setValue(round(level[1]))
        self._threshold.blockSignals(False)
        return (
            f"Auto: {AUTO_MARGIN_DB:.0f} dB over the noise floor "
            f"({level[0]:.0f} dBFS)."
        )

    def _update_steps_hint(self) -> None:
        mode = self.scan_mode()
        if mode == "bookmarks":
            total, skipped = self._bookmark_counts()
            visits = len(self._bookmark_visits())
            if not total:
                text = (
                    "No bookmarks yet: add some in the Bookmarks tab (Ctrl+B "
                    "bookmarks the tuned frequency)."
                )
                tone: Optional[str] = "warning"
            else:
                noun = "bookmark" if visits == 1 else "bookmarks"
                text = (
                    f"Visits {visits} {noun} in turn, each in its own mode, and "
                    "stops on the active ones (the squelch opens)."
                )
                if skipped:
                    verb = "is" if skipped == 1 else "are"
                    text += f" {skipped} marked Skip {verb} passed by."
                tone = None if visits else "warning"
            self._hint.setText(text)
            set_tone(self._hint, tone)
            return
        if not self._range_valid():
            self._hint.setText("End must be above Start.")
            set_tone(self._hint, "danger")
            return
        sweep, note = self._planned_sweep()
        if sweep is None:
            self._hint.setText(note)
            set_tone(self._hint, "danger")
            return
        channels, steps, seconds = self.estimate()
        noun = "channel" if channels == 1 else "channels"
        snoun = "receiver step" if steps == 1 else "receiver steps"
        per = " a pass" if mode != "once" else ""
        text = (
            f"{channels:,} {noun}, {steps:,} {snoun}: {format_duration(seconds)}{per}."
        )
        if self._auto.isChecked():
            text += " " + self._auto_text()
        else:
            text += f" Signals above {self._threshold.value():.0f} dBFS are listed."
        if note:
            text += " " + note
        self._hint.setText(text)
        set_tone(self._hint, "warning" if (note or seconds > CONFIRM_SECONDS) else None)

    def _group_choice(self) -> str:
        value = self._group.currentData()
        return value if isinstance(value, str) else ""

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    #: Why a running scan stopped, by what now blocks scanning.
    _STOPPED_BY = {
        "device": ("Scan stopped: the device was disconnected.", False),
        # A recording that started keeps the receiver where it is.
        "recording": (
            "Scan stopped: the recording keeps the receiver on one frequency.",
            True,
        ),
        "locked": ("Scan stopped: a recording is playing.", True),
        # Answering the channel the scan stopped on: the receiver comes back
        # on it when the transmission ends.
        "transmitting": ("Scan stopped: transmitting.", True),
    }

    def refresh(self) -> None:
        """Follow the window: device, recording, receiver, noise floor."""
        blocker = self._host.scan_blocker()
        engine = self._engine
        kind, reason = blocker
        if engine.active:
            if kind:
                text, stay = self._STOPPED_BY.get(
                    kind, (f"Scan stopped: {reason}", False)
                )
                self._blocker = blocker
                self.stop_scan(text, "warning", stay=stay)
            elif not self._host.scan_receiving():
                self.stop_scan(
                    "Scan stopped: the receiver was stopped.", "warning", stay=False
                )
            elif time.monotonic() - engine.last_block_at > NO_BLOCKS_S:
                # Running, but nothing arrives (a device that stalled): a
                # problem, listed in Tools > Error History.
                logger.warning(
                    "Frequency scan stopped: the receiver returned no samples "
                    "for %.0f s.",
                    NO_BLOCKS_S,
                )
                self.stop_scan(
                    "Scan stopped: the receiver returned no samples. Check that "
                    "it is plugged in and not used by another program.",
                    "warning",
                    stay=False,
                )
        if blocker != self._blocker:
            # (A scan stopped above already said why.)
            if not kind and self._status.text() == self._blocker[1]:
                self._set_status("")  # no longer true
            self._blocker = blocker
            # (A scan the window just stopped for it has said why: keep
            # that; the disabled Start button gives the reason.)
            shown = self._status.text()
            if kind and not engine.active and not shown.startswith("Scan stopped:"):
                self._set_status(reason, "warning")
        if not engine.active and self._auto.isChecked():
            self._update_steps_hint()
        if (
            engine.active
            and self._sort_live
            and self._unsorted
            and self._table.rowCount() <= _LIVE_SORT_ROWS
        ):
            self._resort()
        self._update_controls()

    def _update_controls(self) -> None:
        engine = self._engine
        scanning = engine.active
        mode = self.scan_mode()
        sweep_mode = mode != "bookmarks"
        for w in (self._preset, self._start, self._end, self._step):
            w.setEnabled(not scanning and sweep_mode)
        self._mode.setEnabled(not scanning)
        self._group.setEnabled(not scanning)
        self._auto.setEnabled(not scanning and sweep_mode)
        self._threshold.setEnabled(
            not scanning and sweep_mode and not self._auto.isChecked()
        )
        stops = mode in ("stop", "bookmarks")
        self._resume.setEnabled(not scanning and stops)
        self._resume_label.setEnabled(stops)
        self._group.setVisible(mode == "bookmarks")
        self._group_label.setVisible(mode == "bookmarks")
        self._resume.setVisible(stops)
        self._resume_label.setVisible(stops)

        kind, reason = self._blocker
        no_device = kind == "device" and not scanning
        self._start_btn.setVisible(not no_device)
        self._demo_btn.setVisible(no_device)
        self._connect_btn.setVisible(no_device)
        if scanning:
            self._start_btn.setText("Stop Scan")
            set_role(self._start_btn, "danger")
            self._start_btn.setEnabled(True)
            self._start_btn.setToolTip("Stop the scan")
        else:
            self._start_btn.setText("Start Scan")
            set_role(self._start_btn, "primary")
            why = reason if kind else self._cannot_start()
            self._start_btn.setEnabled(not why)
            self._start_btn.setToolTip(
                why or "Start scanning (it starts the receiver if needed)"
            )
            if kind and not self._status.text():
                self._set_status(reason, "warning")

        listening = scanning and engine.state in (DWELL, LISTEN)
        show_row = listening and engine.mode in ("stop", "bookmarks")
        if show_row and self._listen_row.isHidden():
            self._show_results_buttons()  # (the row pushes them down)
        self._listen_row.setVisible(show_row)

        has_rows = self._table.rowCount() > 0
        has_selection = bool(self._table.selectionModel().selectedRows())
        self._listen_btn.setEnabled(has_selection)
        self._listen_btn.setToolTip(
            "Tune to the selected signal in its mode and listen (stops a scan)"
            if has_selection
            else "Select a signal in the list first"
        )
        self._bookmark_btn.setEnabled(has_selection)
        self._bookmark_btn.setToolTip(
            "Add the selected signal to the bookmarks, in its mode"
            if has_selection
            else "Select a signal in the list first"
        )
        self._bookmark_all_btn.setEnabled(has_rows)
        self._bookmark_all_btn.setToolTip(
            "Add every signal in the list to the bookmarks (those already "
            "bookmarked are skipped)"
            if has_rows
            else "There are no results to bookmark yet"
        )
        self._export_action.setEnabled(has_rows)
        self._clear_action.setEnabled(has_rows and not scanning)
        self._update_empty_state()
        self._update_chip()

    def _cannot_start(self) -> str:
        """Why Start Scan is disabled ("" when it isn't)."""
        if self.scan_mode() == "bookmarks":
            if not self._bookmark_visits():
                total, _skipped = self._bookmark_counts()
                if not total:
                    return "No bookmarks to scan: add some in the Bookmarks tab"
                return "Every bookmark (of this group) is marked Skip"
            return ""
        if not self._range_valid():
            return "End must be above Start"
        sweep, note = self._planned_sweep()
        if sweep is None:
            return note
        return ""

    def _update_empty_state(self) -> None:
        if self._table.rowCount():
            self._empty.set_visible(False)
            return
        reason = self._empty_reason
        if self._blocker[0] == "device" and not self._engine.active:
            reason = "device"
        elif self._engine.active:
            reason = (
                "listening"
                if self._engine.mode in ("stop", "bookmarks")
                else "scanning"
            )
        self._empty.set_text(_EMPTY_TEXTS.get(reason, _EMPTY_TEXTS["idle"]))
        self._empty.set_visible(True)

    # ------------------------------------------------------------------ #
    # Scanning
    # ------------------------------------------------------------------ #
    def toggle_scan(self) -> None:
        if self._engine.active:
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self, start_index: Optional[int] = None) -> bool:
        """Start a scan with the panel's settings (``start_index``: the
        bookmark a bookmark scan starts at, else the first one the list
        shows). False if it can't.

        A scan running (Scan Bookmarks from Here during a sweep) is
        replaced: the receiver stays where it is meanwhile, and goes back
        to where the first scan began when the new one ends, or can't
        start."""
        restart = self._engine.active
        if restart:
            self._restarting = True
            try:
                self.stop_scan(stay=True)
            finally:
                self._restarting = False
        if self._launch(start_index, begin=not restart):
            return True
        if restart:
            self._host.scan_end(stay=False)  # back where the first began
        return False

    def _launch(self, start_index: Optional[int], begin: bool) -> bool:
        """:meth:`start_scan`'s work; ``begin``: the host remembers where
        the receiver is now (else where the scan replaced began)."""
        host = self._host
        self._visits_cache = None  # (a scan starts from the bookmarks now)
        kind, reason = host.scan_blocker()
        self._blocker = (kind, reason)
        if kind:
            self._set_status(reason, "warning")
            self._update_controls()
            return False
        why = self._cannot_start()
        if why:
            self._set_status(why + ".", "danger")
            self._update_controls()
            return False
        mode = self.scan_mode()
        settings = ScanSettings(mode=mode, resume_s=self._resume.value())
        if mode == "bookmarks":
            visits = [dict(visit) for visit in self._bookmark_visits()]
            settings.bookmarks = visits
            if start_index is not None:  # (else the first one listed)
                settings.start_index = self._first_visit(visits, start_index)
        else:
            sweep, note = self._planned_sweep()
            if sweep is None:  # (checked above)
                return False
            settings.start_hz, settings.end_hz = sweep
            settings.step_hz = self._step.value() * 1e3
            settings.threshold_db = (
                None if self._auto.isChecked() else self._threshold.value()
            )
            preset = self.preset()
            if preset is not None:
                settings.listening = preset.listening
                settings.min_width_hz = preset.min_width_khz * 1e3
            _channels, steps, seconds = self.estimate()
            if seconds > CONFIRM_SECONDS and not self._confirm_long_scan(
                steps, seconds
            ):
                return False
            self._range_note = f" {note}" if note else ""
        if not host.scan_receiving() and not host.scan_start_receiver():
            self._set_status(
                "The receiver didn't start, so the scan can't run.", "danger"
            )
            return False
        if begin:
            host.scan_begin()
        self._stay = False
        rate = host.scan_sample_rate()
        error = self._engine.start(settings, rate, host.scan_settle_s())
        if error:
            if begin:
                host.scan_end(stay=False)
            if error.startswith("error: "):
                error = f"The scan could not start: {error[len('error: '):]}."
            self._set_status(error, "danger")
            self._update_controls()
            return False
        # A new scan, new results.
        self._hits = []
        self._rebuild_table()
        self._table.setSortingEnabled(False)
        self._sort_live = self._unsorted = False
        self._empty_reason = "scanning"
        self._progress.setValue(0)
        self._timer.start()
        self._on_engine_state()
        self._reveal_results()
        return True

    def _first_visit(self, visits: List[Dict[str, Any]], index: int) -> int:
        """Where in ``visits`` (in the list's order) a scan from bookmark
        ``index`` begins: on it, or, when it is marked Skip, on the next
        one the list shows after it (the first one past the end)."""
        for i, visit in enumerate(visits):
            if visit.get("index", -1) == index:
                return i
        # (A host without the list's order: as stored.)
        order = getattr(self._host, "scan_bookmark_order", lambda: [])() or []
        listed = {stored: place for place, stored in enumerate(order)}
        start = listed.get(index, index)
        return next(
            (
                i
                for i, visit in enumerate(visits)
                if listed.get(visit.get("index", -1), visit.get("index", -1)) >= start
            ),
            0,
        )

    def _scroll_area(self) -> Optional["QScrollArea"]:
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        return area

    def _reveal_results(self) -> None:
        """In a short tab (1366x768), scroll so the status, the progress and
        the results are in view rather than the settings."""
        area = self._scroll_area()
        if area is None:
            return
        bar = area.verticalScrollBar()
        if bar is not None and bar.maximum() > 0:
            bar.setValue(min(bar.maximum(), self._run_row.y() - 6))
        self._show_results_buttons()

    def _show_results_buttons(self) -> None:
        """Bring the buttons under the results into view, once the layout
        has settled (the listen row or a longer status may push them
        down)."""
        self._buttons_timer.start()

    def _ensure_buttons_visible(self) -> None:
        area = self._scroll_area()
        if area is None or not self.isVisible():
            return
        # The layouts first, from the status row up (a status that wrapped,
        # the listen row shown), then the tab's height and scroll range, so
        # the buttons are measured where they now are.
        for widget in (self._run_row, self):
            layout = widget.layout()
            if layout is not None:
                layout.activate()
        QGuiApplication.sendEvent(area, QEvent(QEvent.Type.LayoutRequest))
        area.ensureWidgetVisible(self._results_btn, 0, 6)

    def scan_bookmarks_from(self, index: int) -> bool:
        """Scan all the bookmarks, starting at bookmark ``index`` (the
        Bookmarks tab's Scan Bookmarks from Here; it replaces a scan
        running, see :meth:`start_scan`). The tab shows Scan bookmarks of
        all groups meanwhile; its own mode and group (and the saved ones)
        come back when the scan ends."""
        before = self._mode_before_from_here or (
            self.scan_mode(),
            self._group_choice(),
        )
        self._show_mode_quietly("bookmarks", "")
        if self.start_scan(start_index=index):
            self._mode_before_from_here = before
            return True
        self._mode_before_from_here = None
        self._show_mode_quietly(*before)
        return False

    def _show_mode_quietly(self, mode: str, group: str) -> None:
        """Show ``mode`` and ``group`` without saving them."""
        restoring = self._restoring
        self._restoring = True
        try:
            self.set_scan_mode(mode)
            self._refresh_groups(prefer=group)
            self._update_steps_hint()
            self._update_controls()
        finally:
            self._restoring = restoring

    def _confirm_long_scan(self, steps: int, seconds: float) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Long Scan")
        once = self.scan_mode() == "once"
        box.setText(
            f"This scan takes {format_duration(seconds)}"
            + ("" if once else " per pass")
            + f" ({steps:,} receiver steps)."
        )
        box.setInformativeText(
            "A narrower range is quicker. Start it anyway? You can stop it at "
            "any time."
        )
        start = box.addButton("Start Scan", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        ok = box.clickedButton() is start
        box.deleteLater()
        return ok

    def stop_scan(
        self,
        text: Optional[str] = None,
        tone: Optional[str] = "muted",
        stay: Optional[bool] = None,
    ) -> None:
        """Stop a running scan. ``stay`` keeps the receiver where the scan
        is (default: when it stopped on a channel being listened to)."""
        engine = self._engine
        if not engine.active:
            return
        if stay is None:
            stay = engine.state == LISTEN
        self._stay = bool(stay)
        self._stop_text = (text, tone)
        engine.stop("stopped")

    def _engine_hold(self, on: bool) -> None:
        self._engine.hold(on)

    def _on_engine_state(self) -> None:
        engine = self._engine
        if not engine.active:
            return
        self._progress.setValue(int(engine.progress() * 1000))
        current = engine.current
        held = " (held)" if engine.held else ""
        if engine.state in (DWELL, LISTEN) and current is not None:
            name = f"{current.label} · " if current.label else ""
            listening = mode_text(current.listening[0], current.listening[1])
            freq = _say_freq(current.frequency_hz)
            if engine.state == LISTEN:
                self._listen_label.setText(
                    f"Listening: {name}{freq}"
                    + (f" · {listening}" if listening else "")
                    + held
                )
            else:
                self._listen_label.setText(f"Checking {name}{freq}{held}")
            self._listen_label.setToolTip(self._listen_label.text())
        # The button says whether the scan is held: Hold also holds it on a
        # channel being checked, and going on releases it.
        if self._hold_btn.isChecked() != engine.held:
            self._hold_btn.blockSignals(True)
            self._hold_btn.setChecked(engine.held)
            self._hold_btn.blockSignals(False)
        self._set_status(self._scanning_text(), None)
        self._update_controls()

    def _scanning_text(self) -> str:
        engine = self._engine
        count = len(self._hits)
        noun = "signal" if count == 1 else "signals"
        pass_text = f"Pass {engine.pass_number} · " if engine.mode != "once" else ""
        if engine.mode == "bookmarks":
            heard = "heard" if count else "heard yet"
            return f"{pass_text}{count} {noun} {heard}"
        page = engine.current_page()
        where = ""
        if page is not None and page.count:
            where = f"{_say_freq(page.first_hz)} – {_say_freq(page.last_hz)} · "
        found = "found" if engine.mode != "stop" else "heard"
        return f"{pass_text}{where}{count} {noun} {found}"

    def _on_found(self, found: Found) -> None:
        # (In the display loop: the engine's on_state, after the page's
        # readings, says how many were found and updates the buttons.)
        now = time.time()
        index = self._repeated_hit(found)
        if index >= 0:
            hit = self._hits[index]
            hit.count += 1
            hit.last_seen = now
            if found.peak_db > hit.peak_db:
                self._index_frequency(index, found.frequency_hz)
                hit.frequency_hz = found.frequency_hz
                hit.peak_db = found.peak_db
                hit.snr_db = found.snr_db
                hit.width_hz = found.width_hz
                self._note_width(hit.width_hz)
            self._render_hit(index)
            return
        peak = found.peak_db if math.isfinite(found.peak_db) else -120.0
        snr = found.snr_db if math.isfinite(found.snr_db) else 0.0
        demod, deviation, width = found.listening
        self._hits.append(
            ScanHit(
                found.frequency_hz,
                peak,
                snr,
                found.width_hz,
                demod,
                deviation,
                width,
                found.label,
                now,
                now,
                1,
                float(getattr(found, "tone_hz", 0.0) or 0.0),
            )
        )
        index = len(self._hits) - 1
        insort(self._by_freq, (found.frequency_hz, index))
        self._note_width(found.width_hz)
        self._render_hit(index)
        if self._table.rowCount() == 1:
            self._update_controls()  # the first: the list's buttons and empty state

    def _repeated_hit(self, found: Found) -> int:
        """The result ``found`` repeats (the first listed, if several), or
        -1: in a sweep the same signal (see ``same_signal``); in a bookmark
        scan the same bookmark (two on one frequency, a repeater's users
        with different tones, are two results)."""
        tone = float(getattr(found, "tone_hz", 0.0) or 0.0)
        bookmarks = self._engine.mode == "bookmarks"
        freq = found.frequency_hz
        if bookmarks:
            step, reach = 1.0, 10.0
        else:
            step = self._step.value() * 1e3
            reach = max(0.75 * step, 0.5 * self._widest_hz)
        # Only the results within reach of it can be the same.
        first = bisect_left(self._by_freq, (freq - reach,))
        last = bisect_right(self._by_freq, (freq + reach, math.inf))
        best = -1
        for _freq, index in self._by_freq[first:last]:
            if best >= 0 and index > best:
                continue
            hit = self._hits[index]
            if bookmarks:
                same = (
                    abs(hit.frequency_hz - freq) < 10.0
                    and hit.label == found.label
                    and hit.listening == tuple(found.listening)
                    and hit.tone_hz == tone
                )
            else:
                same = same_signal(hit.frequency_hz, hit.width_hz, freq, step)
            if same:
                best = index
        return best

    def _index_frequency(self, index: int, freq_hz: float) -> None:
        """Result ``index`` moves to ``freq_hz`` (a stronger reading)."""
        old = self._hits[index].frequency_hz
        at = bisect_left(self._by_freq, (old, index))
        if at < len(self._by_freq) and self._by_freq[at] == (old, index):
            del self._by_freq[at]
        insort(self._by_freq, (freq_hz, index))

    def _note_width(self, width_hz: float) -> None:
        if math.isfinite(width_hz) and width_hz > self._widest_hz:
            self._widest_hz = float(width_hz)

    def _on_finished(self, reason: str) -> None:
        engine = self._engine
        host = self._host
        if not self._restarting:  # (a new scan takes over where it is)
            host.scan_end(stay=self._stay)
        self._stay = False
        self._listen_row.setVisible(False)
        self._hold_btn.blockSignals(True)
        self._hold_btn.setChecked(False)
        self._hold_btn.blockSignals(False)
        self._table.setSortingEnabled(True)
        self._sort_live = self._unsorted = False
        if self._mode_before_from_here is not None and not self._restarting:
            # (Scan Bookmarks from Here ended: the tab's own mode again.)
            before, self._mode_before_from_here = self._mode_before_from_here, None
            self._show_mode_quietly(*before)
        count = len(self._hits)
        noun = "signal" if count == 1 else "signals"
        text, tone = getattr(self, "_stop_text", (None, "muted"))
        self._stop_text = (None, "muted")
        # A full bar for a pass that ran to its end (below), else an empty
        # one: a part-filled bar under a stopped scan looks like one going on.
        self._progress.setValue(0)
        if reason == "complete":
            noise = engine.noise_db
            floor = f" Noise floor {noise:.0f} dBFS." if noise is not None else ""
            refused = engine.steps_refused
            skipped = (
                f" {refused:,} steps the receiver couldn't tune were skipped."
                if refused
                else ""
            )
            if not engine.steps_measured:
                self._empty_reason = "refused" if refused else "no_samples"
                self._set_status(
                    _EMPTY_TEXTS[self._empty_reason].split("\n")[0] + ".", "warning"
                )
            else:
                self._empty_reason = "nothing"
                self._set_status(
                    f"Scan complete: {count} {noun} found.{floor}"
                    f"{getattr(self, '_range_note', '')}{skipped}",
                    "success" if count else None,
                )
                self._progress.setValue(1000)
        elif reason == "stopped":
            self._empty_reason = "stopped"
            if text:
                self._set_status(text, tone)
            else:
                verb = "heard" if engine.mode in ("stop", "bookmarks") else "found"
                self._set_status(f"Scan stopped. {count} {noun} {verb}.", "muted")
        elif reason == ALL_SKIPPED:
            self._empty_reason = "stopped"
            self._set_status(
                f"Scan stopped: every bookmark was skipped. {count} {noun} heard.",
                "muted",
            )
        else:
            message = (
                reason[len("error: ") :] if reason.startswith("error: ") else reason
            ).rstrip(".")
            self._empty_reason = "stopped"
            self._set_status(
                f"The scan stopped: {message}. {count} {noun} found. Details are "
                "in Tools > Error History.",
                "warning",
            )
            logger.warning("Frequency scan stopped: %s", message)
        if not self.isVisible():
            self._timer.stop()
            # (From another tab the status-bar chip just went: say why;
            # unless what stopped it (a transmission, a recording, the
            # device gone) has the status bar already.)
            if self._status.text() and not self._host.scan_blocker()[0]:
                tone = self._status_tone
                self._host.scan_message(
                    self._status.text(), None if tone == "muted" else tone
                )
        self._save_results()
        self._update_controls()
        self._show_results_buttons()

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def hits(self) -> List[ScanHit]:
        return list(self._hits)

    def _rebuild_table(self) -> None:
        """The table (and the index of the results, see ``_by_freq``) anew
        from ``_hits``."""
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        self._cells = {}  # (the rows' cells go)
        self._table.setRowCount(0)
        self._by_freq = sorted((h.frequency_hz, i) for i, h in enumerate(self._hits))
        self._widest_hz = 0.0
        for index, hit in enumerate(self._hits):
            self._note_width(hit.width_hz)
            self._render_hit(index)
        self._table.setSortingEnabled(sorting)
        self._update_controls()

    def _row_of(self, index: int) -> int:
        cell = self._cells.get(index)
        return self._table.row(cell) if cell is not None else -1

    def _on_sort_changed(self, _column: int, _order: Any) -> None:
        """A column header clicked: during a scan (when the table doesn't
        sort itself) sort now, and keep it sorted (see refresh)."""
        if self._engine.active and not self._table.isSortingEnabled():
            self._sort_live = True
            self._resort()

    def _resort(self) -> None:
        header = self._table.horizontalHeader()
        self._table.sortItems(
            header.sortIndicatorSection(), header.sortIndicatorOrder()
        )
        self._unsorted = False

    def _render_hit(self, index: int) -> None:
        hit = self._hits[index]
        row = self._row_of(index)
        self._unsorted = True
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        if row < 0:
            row = self._table.rowCount()
            self._table.insertRow(row)
        if hit.label:
            # A bookmark scan's result says which bookmark (two on one
            # repeater, with different tones, are told apart): its name
            # first, so a narrow column cuts the frequency's end (it is in
            # the tooltip), not the name. It still sorts by frequency.
            freq = _NumericItem(
                f"{hit.label} · {_format_freq(hit.frequency_hz)}",
                hit.frequency_hz,
                align_right=False,
            )
        else:
            freq = _NumericItem(_format_freq(hit.frequency_hz), hit.frequency_hz)
        freq.setData(_HIT_ROLE, index)
        # Whole dB in the narrow column (the tooltip has tenths).
        level = _NumericItem(f"{hit.peak_db:.0f} dBFS", hit.peak_db)
        snr = _NumericItem(f"{hit.snr_db:.0f} dB", hit.snr_db)
        mode = QTableWidgetItem(mode_text(hit.mode, hit.fm_deviation) or "—")
        heard_at = datetime.fromtimestamp(hit.last_seen) if hit.last_seen else None
        heard = _NumericItem(
            heard_at.strftime("%H:%M:%S") if heard_at else "—", hit.last_seen, False
        )
        tip = self._hit_tooltip(hit)
        for col, item in enumerate((freq, level, snr, mode, heard)):
            item.setToolTip(tip)
            self._table.setItem(row, col, item)
        self._cells[index] = freq
        self._table.setSortingEnabled(sorting)

    @staticmethod
    def _hit_tooltip(hit: ScanHit) -> str:
        lines = []
        if hit.label:
            lines.append(hit.label)
        band = band_for(hit.frequency_hz)
        lines.append(_say_freq(hit.frequency_hz) + (f" ({band.name})" if band else ""))
        lines.append(f"Peak {hit.peak_db:.1f} dBFS, SNR {hit.snr_db:.1f} dB")
        if hit.width_hz > 0:
            lines.append(f"Occupied width about {hit.width_hz / 1e3:.0f} kHz")
        mode = mode_text(hit.mode, hit.fm_deviation)
        if mode:
            width = f", {hit.bandwidth} filter" if hit.bandwidth else ""
            tone = (
                f", tone squelch {hit.tone_hz:.1f} Hz"
                if hit.tone_hz and hit.mode == "FM"
                else ""
            )
            lines.append(f"Listen: {mode}{width}{tone}")
        if hit.first_seen:
            first = datetime.fromtimestamp(hit.first_seen).strftime("%H:%M:%S")
            times = "once" if hit.count == 1 else f"{hit.count} times"
            lines.append(f"Heard {times}, first at {first}")
        lines.append("Double-click or Enter: listen")
        return "\n".join(lines)

    def selected_hit(self) -> Optional[ScanHit]:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), COL_FREQ)
        index = item.data(_HIT_ROLE) if item is not None else None
        if isinstance(index, int) and 0 <= index < len(self._hits):
            return self._hits[index]
        return None

    def selected_frequency(self) -> Optional[float]:
        """Frequency (Hz) of the selected result, if any."""
        hit = self.selected_hit()
        return hit.frequency_hz if hit is not None else None

    def listen_selected(self) -> None:
        """Tune to the selected result in its mode (stopping a scan)."""
        hit = self.selected_hit()
        if hit is None:
            return
        if self._engine.active:
            self.stop_scan(stay=True)
        if self._host.scan_listen(hit):
            mode = mode_text(hit.mode, hit.fm_deviation)
            self._set_status(
                f"Listening to {_say_freq(hit.frequency_hz)}"
                + (f" ({mode})." if mode else "."),
                "success",
            )

    def bookmark_selected(self) -> None:
        hit = self.selected_hit()
        if hit is not None:
            self._bookmark([hit])

    def bookmark_all(self) -> None:
        """Bookmark every result. More than :data:`CONFIRM_BOOKMARKS` ask
        first and go in a group of their own ("Scan 2026-10-31 22:00"), so
        they can be removed together."""
        if not self._hits:
            return
        hits = sorted(self._hits, key=lambda h: h.frequency_hz)
        group = ""
        if len(hits) > CONFIRM_BOOKMARKS:
            group = datetime.now().strftime("Scan %Y-%m-%d %H:%M")
            if not self._confirm_bookmark_all(len(hits), group):
                return
        self._bookmark(hits, group)

    def _confirm_bookmark_all(self, count: int, group: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Bookmark All")
        box.setText(f"Bookmark all {count:,} signals?")
        box.setInformativeText(
            f"They go in the group “{group}” of the Bookmarks tab (those "
            "already bookmarked are skipped). To remove them all, right-click "
            "the group there and choose Remove Group."
        )
        add = box.addButton("Bookmark All", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(add)
        box.setEscapeButton(cancel)
        box.exec()
        ok = box.clickedButton() is add
        box.deleteLater()
        return ok

    def _bookmark(self, hits: List[ScanHit], group: str = "") -> None:
        # (A group only for Bookmark All's many: hosts written before it
        # take the results alone.)
        extra = {"group": group} if group else {}
        added, existing = self._host.scan_add_bookmarks(hits, **extra)
        if len(hits) == 1:
            if added:
                text = _bookmarked(hits[0])
            else:
                text = f"Already bookmarked: {existing[0] if existing else ''}"
        else:
            text = f"Bookmarked {added} of {len(hits)} signals"
            if added and group:
                text += f" in the group “{group}”"
            text += f" ({len(existing)} already bookmarked)." if existing else "."
        tone = "success" if added else "info"
        self._set_status(text, tone)
        self._host.scan_message(text, tone)

    def clear_results(self) -> None:
        if self._engine.active:
            return
        self._hits = []
        self._rebuild_table()
        self._empty_reason = "cleared"
        self._progress.setValue(0)
        self._set_status("Results cleared.", "muted")
        self._save_results()
        self._update_controls()

    def export_csv(self, path: Optional[str] = None) -> int:
        """Save the results as CSV; returns the number of rows written."""
        if not self._hits:
            return 0
        if not path:
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            name = f"scan-{stamp}.csv"
            if self._settings is not None:
                # In Documents, then the folder used last (not the folder
                # the app was started from), as the other exports.
                path, _ = file_dialogs.save_file(
                    self,
                    self._settings,
                    "scan_results",
                    "Export Results (CSV)",
                    name,
                    "CSV (*.csv)",
                )
            else:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Export Results (CSV)", name, "CSV (*.csv)"
                )
        if not path:
            return 0
        if not path.lower().endswith(".csv"):
            path += ".csv"
        rows = sorted(self._hits, key=lambda h: h.frequency_hz)
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Frequency (MHz)",
                        "Level (dBFS)",
                        "SNR (dB)",
                        "Width (kHz)",
                        "Mode",
                        "FM deviation",
                        "Filter",
                        "Band",
                        "Name",
                        "First heard",
                        "Last heard",
                        "Times heard",
                    ]
                )
                for hit in rows:
                    band = band_for(hit.frequency_hz)
                    writer.writerow(
                        [
                            f"{hit.frequency_hz / 1e6:.6f}",
                            f"{hit.peak_db:.1f}",
                            f"{hit.snr_db:.1f}",
                            f"{hit.width_hz / 1e3:.1f}",
                            hit.mode,
                            hit.fm_deviation,
                            hit.bandwidth,
                            band.name if band else "",
                            hit.label,
                            _iso(hit.first_seen),
                            _iso(hit.last_seen),
                            hit.count,
                        ]
                    )
        except OSError as exc:
            logger.warning("Could not export the scan results: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
            return 0
        text = f"Exported {len(rows)} results to {path}"
        self._set_status(text, "success")
        self._host.scan_message(text, "success")
        return len(rows)

    def _show_context_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None:
            return
        self._table.selectRow(item.row())
        hit = self.selected_hit()
        if hit is None:
            return
        menu = QMenu(self)
        freq = _say_freq(hit.frequency_hz)
        listen = menu.addAction(f"Listen to {freq}")
        listen.triggered.connect(lambda _c=False: self.listen_selected())
        bookmark = menu.addAction("Bookmark")
        bookmark.triggered.connect(lambda _c=False: self.bookmark_selected())
        copy = menu.addAction("Copy Frequency")
        copy.triggered.connect(lambda _c=False, f=freq: self._copy(f))
        menu.exec(self._table.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _copy(self, text: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self._set_status(f"Copied {text}.", "info")
        # (In the status bar too, as the Bookmarks tab and the plots say it.)
        self._host.scan_message(f"Copied {text}", "info")


def _iso(timestamp: float) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


__all__ = [
    "CONFIRM_SECONDS",
    "ScanHit",
    "ScannerHost",
    "ScannerPanel",
    "guess_listening",
    "mode_text",
]
