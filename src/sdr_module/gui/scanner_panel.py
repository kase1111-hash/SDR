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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, Qt, QTimer
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

from ..core.band_plan import band_for, listening_for
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
    from .decoder_panel import ViewPlaceholder
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

# Results table columns, and the item data role holding a row's result.
COL_FREQ, COL_LEVEL, COL_SNR, COL_MODE, COL_HEARD = range(5)
_HIT_ROLE = Qt.ItemDataRole.UserRole + 1 if HAS_PYQT6 else 0

#: A scan whose pass takes longer than this asks before it starts.
CONFIRM_SECONDS = 120.0
#: A running scan that gets no blocks for this long stops (and says why).
NO_BLOCKS_S = 3.0
#: Display frames per second (the window's display timer).
_FRAME_S = 1.0 / 30.0
#: How many results are kept between sessions.
_SAVED_RESULTS = 500
#: The fields' ranges.
_FREQ_RANGE_MHZ = (0.1, 6000.0)
_STEP_RANGE_KHZ = (1.0, 10000.0)
_RESUME_RANGE_S = (0.5, 60.0)

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
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(hit.frequency_hz) and hit.frequency_hz > 0):
            return None
        return hit


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
        kind "device" (none connected), "recording" or "locked" (a
        recording plays back)."""
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

    def scan_add_bookmarks(self, hits: List[ScanHit]) -> Tuple[int, List[str]]:
        """Bookmark results: ``(added, names already bookmarked)``."""
        return 0, []

    def scan_bookmark_list(self, group: str = "") -> List[Dict[str, Any]]:
        """The bookmarks a bookmark scan visits (see
        :func:`~sdr_module.gui.scan_engine.bookmark_scan_list`)."""
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
        self._empty_reason = "idle"
        self._restoring = False
        self._blocker: Tuple[str, str] = ("", "")
        self._stay = False  # end the scan on the channel listened to
        self._setup_ui()
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
            self._preset.addItem(preset.label)
        self._preset.setToolTip(
            "Band: fills in the range and step for a common band; signals found "
            "there are listened to in the band's mode (airband: AM)"
        )
        self._preset.setAccessibleName("Band")
        self._preset.currentIndexChanged.connect(self._on_preset_changed)
        top.addWidget(self._preset, 3)

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
        top.addWidget(self._mode, 2)
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

        threshold_row = QHBoxLayout()
        threshold_row.setSpacing(4)
        self._auto = QCheckBox("Auto")
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
        label = QLabel("T&hreshold:")
        label.setBuddy(self._auto)
        grid.addWidget(label, 1, 2)
        grid.addLayout(threshold_row, 1, 3)

        self._resume = self._spin(*_RESUME_RANGE_S, 1, 3.0, " s")
        self._resume.setToolTip(
            "Stop on signal and Scan bookmarks: go on after the channel has "
            "been quiet (squelch closed) this long"
        )
        self._resume_label = self._add_field(grid, 2, 0, "&Resume:", self._resume)

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
        self._listen_label = QLabel()
        set_role(self._listen_label, "value")
        self._listen_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
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
        self._table = QTableWidget(0, 5)
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
        header_tips = (
            "Signal center frequency, to the nearest kHz (a broadcast FM "
            "station: its 100 kHz channel)",
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
        self._save("preset", self._preset.currentText())

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
        self._group.addItem("All bookmarks", "")
        for group in self._host.scan_bookmark_groups():
            self._group.addItem(group, group)
        index = self._group.findData(wanted)
        self._group.setCurrentIndex(max(index, 0))
        self._group.blockSignals(False)

    def bookmarks_changed(self) -> None:
        """The bookmarks changed (the window calls this)."""
        self._refresh_groups()
        self._update_steps_hint()
        self._update_controls()

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
                f"This device tunes {_format_freq(low)} to {_format_freq(high)}. "
                "Choose a range inside it."
            )
        if start_hz < low or end_hz > high:
            start_hz, end_hz = max(start_hz, low), min(end_hz, high)
            return (start_hz, end_hz), (
                f"Limited to the device's range, {_format_freq(start_hz)} to "
                f"{_format_freq(end_hz)}."
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
            total, skipped = self._host.scan_bookmark_counts()
            visits = len(self._host.scan_bookmark_list(self._group_choice()))
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
                    text += f" {skipped} marked Skip are passed by."
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
                # Running, but nothing arrives (a device that stalled).
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
            if kind and not engine.active:
                self._set_status(reason, "warning")
        if not engine.active and self._auto.isChecked():
            self._update_steps_hint()
        self._update_controls()

    def _update_controls(self) -> None:
        engine = self._engine
        scanning = engine.active
        mode = self.scan_mode()
        sweep_mode = mode != "bookmarks"
        for w in (self._preset, self._start, self._end, self._step):
            w.setEnabled(not scanning and sweep_mode)
        self._mode.setEnabled(not scanning)
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
        self._listen_row.setVisible(listening and engine.mode in ("stop", "bookmarks"))

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

    def _cannot_start(self) -> str:
        """Why Start Scan is disabled ("" when it isn't)."""
        if self.scan_mode() == "bookmarks":
            if not self._host.scan_bookmark_list(self._group_choice()):
                total, _skipped = self._host.scan_bookmark_counts()
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

    def start_scan(self, start_index: int = 0) -> bool:
        """Start a scan with the panel's settings (``start_index``: the
        bookmark a bookmark scan starts at). False if it can't."""
        host = self._host
        if self._engine.active:
            self.stop_scan(stay=True)  # (it would lose where the last began)
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
            visits = host.scan_bookmark_list(self._group_choice())
            settings.bookmarks = visits
            settings.start_index = next(
                (i for i, v in enumerate(visits) if v.get("index", -1) >= start_index),
                0,
            )
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
        host.scan_begin()
        self._stay = False
        rate = host.scan_sample_rate()
        error = self._engine.start(settings, rate, host.scan_settle_s())
        if error:
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
        self._empty_reason = "scanning"
        self._progress.setValue(0)
        self._timer.start()
        self._on_engine_state()
        self._reveal_results()
        return True

    def _reveal_results(self) -> None:
        """In a short tab (1366x768), scroll so the status, the progress and
        the results are in view rather than the settings."""
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        if area is None:
            return
        bar = area.verticalScrollBar()
        if bar is not None and bar.maximum() > 0:
            bar.setValue(min(bar.maximum(), self._run_row.y() - 6))

    def scan_bookmarks_from(self, index: int) -> bool:
        """Scan all the bookmarks, starting at bookmark ``index`` (the
        Bookmarks tab's Scan Bookmarks from Here)."""
        self.stop_scan(stay=True)
        self.set_scan_mode("bookmarks")
        self._group.setCurrentIndex(0)  # all bookmarks
        return self.start_scan(start_index=index)

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
        if engine.state in (DWELL, LISTEN) and current is not None:
            name = f"{current.label} · " if current.label else ""
            listening = mode_text(current.listening[0], current.listening[1])
            freq = _format_freq(current.frequency_hz)
            if engine.state == LISTEN:
                held = " (held)" if engine.held else ""
                self._listen_label.setText(
                    f"Listening: {name}{freq}"
                    + (f" · {listening}" if listening else "")
                    + held
                )
            else:
                self._listen_label.setText(f"Checking {name}{freq}")
            self._listen_label.setToolTip(self._listen_label.text())
        if engine.state != LISTEN and self._hold_btn.isChecked():
            self._hold_btn.blockSignals(True)
            self._hold_btn.setChecked(False)
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
            where = f"{_format_freq(page.first_hz)} – {_format_freq(page.last_hz)} · "
        found = "found" if engine.mode != "stop" else "heard"
        return f"{pass_text}{where}{count} {noun} {found}"

    def _on_found(self, found: Found) -> None:
        now = time.time()
        step = self._step.value() * 1e3 if self._engine.mode != "bookmarks" else 1.0
        for index, hit in enumerate(self._hits):
            same = (
                abs(hit.frequency_hz - found.frequency_hz) < 10.0
                if self._engine.mode == "bookmarks"
                else same_signal(
                    hit.frequency_hz, hit.width_hz, found.frequency_hz, step
                )
            )
            if same:
                hit.count += 1
                hit.last_seen = now
                if found.peak_db > hit.peak_db:
                    hit.frequency_hz = found.frequency_hz
                    hit.peak_db = found.peak_db
                    hit.snr_db = found.snr_db
                    hit.width_hz = found.width_hz
                self._render_hit(index)
                break
        else:
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
                )
            )
            self._render_hit(len(self._hits) - 1)
        self._set_status(self._scanning_text(), None)
        self._update_controls()

    def _on_finished(self, reason: str) -> None:
        engine = self._engine
        host = self._host
        host.scan_end(stay=self._stay)
        self._stay = False
        self._listen_row.setVisible(False)
        self._hold_btn.blockSignals(True)
        self._hold_btn.setChecked(False)
        self._hold_btn.blockSignals(False)
        self._table.setSortingEnabled(True)
        count = len(self._hits)
        noun = "signal" if count == 1 else "signals"
        text, tone = getattr(self, "_stop_text", (None, "muted"))
        self._stop_text = (None, "muted")
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
        self._save_results()
        self._update_controls()

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def hits(self) -> List[ScanHit]:
        return list(self._hits)

    def _rebuild_table(self) -> None:
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for index in range(len(self._hits)):
            self._render_hit(index)
        self._table.setSortingEnabled(sorting)
        self._update_controls()

    def _row_of(self, index: int) -> int:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_FREQ)
            if item is not None and item.data(_HIT_ROLE) == index:
                return row
        return -1

    def _render_hit(self, index: int) -> None:
        hit = self._hits[index]
        row = self._row_of(index)
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        if row < 0:
            row = self._table.rowCount()
            self._table.insertRow(row)
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
        self._table.setSortingEnabled(sorting)

    @staticmethod
    def _hit_tooltip(hit: ScanHit) -> str:
        lines = []
        if hit.label:
            lines.append(hit.label)
        band = band_for(hit.frequency_hz)
        lines.append(
            _format_freq(hit.frequency_hz) + (f" ({band.name})" if band else "")
        )
        lines.append(f"Peak {hit.peak_db:.1f} dBFS, SNR {hit.snr_db:.1f} dB")
        if hit.width_hz > 0:
            lines.append(f"Occupied width about {hit.width_hz / 1e3:.0f} kHz")
        mode = mode_text(hit.mode, hit.fm_deviation)
        if mode:
            width = f", {hit.bandwidth} filter" if hit.bandwidth else ""
            lines.append(f"Listen: {mode}{width}")
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
                f"Listening to {_format_freq(hit.frequency_hz)}"
                + (f" ({mode})." if mode else "."),
                "success",
            )

    def bookmark_selected(self) -> None:
        hit = self.selected_hit()
        if hit is not None:
            self._bookmark([hit])

    def bookmark_all(self) -> None:
        if self._hits:
            self._bookmark(sorted(self._hits, key=lambda h: h.frequency_hz))

    def _bookmark(self, hits: List[ScanHit]) -> None:
        added, existing = self._host.scan_add_bookmarks(hits)
        if len(hits) == 1:
            if added:
                text = f"Bookmarked {_format_freq(hits[0].frequency_hz)}."
            else:
                text = f"Already bookmarked: {existing[0] if existing else ''}"
        else:
            text = f"Bookmarked {added} of {len(hits)} signals"
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
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Results (CSV)", f"scan-{stamp}.csv", "CSV (*.csv)"
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
        freq = _format_freq(hit.frequency_hz)
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
