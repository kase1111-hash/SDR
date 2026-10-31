"""
Bookmarks / memory channels panel.

A list of saved frequencies the user can tune with a double-click (or
Enter). Stored via GuiSettings; survives restarts.

A bookmark remembers how it is listened to: the demodulation mode it was
saved with (chosen on Add, the current mode by default), the FM deviation,
the filter width and a CTCSS tone squelch (CHIRP channels carry their mode
and tone too), so tuning one also restores how to listen to it. Bookmarks
can be grouped (shown as folders), filtered, edited (Edit..., or the
right-click menu) and skipped by the Scanner's bookmark scan.

Channels import and export as CHIRP-compatible CSV (see
`sdr_module.core.chirp_csv`), so memories can be moved between this
application, CHIRP, and a handheld radio; a round trip keeps how each one
is listened to.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QGuiApplication
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core.band_plan import band_for, default_name, listening_for
from ..core.chirp_csv import (
    DEFAULT_DEMOD,
    ChirpCsvError,
    chirp_mode_to_demod,
    export_bookmarks_csv,
    is_chirp_channel,
    presets_to_channels,
    read_chirp_csv,
    write_chirp_csv,
)
from . import file_dialogs
from .bookmark_editor import bookmark_tone_hz
from .bookmark_editor import edit_bookmark as _edit_dialog
from .control_panel import DEMOD_MODES, MAX_FREQUENCY_HZ, MIN_FREQUENCY_HZ
from .decoder_panel import ViewPlaceholder
from .settings_store import (
    DEFAULT_FREQUENCY_HZ,
    GuiSettings,
    describe_dropped,
    valid_bookmarks,
)

logger = logging.getLogger(__name__)

CSV_FILTER = "CHIRP CSV (*.csv);;All files (*)"

# Tree columns.
COL_NAME, COL_FREQ, COL_MODE = range(3)

_EMPTY_TEXT = (
    "No bookmarks yet\n"
    "Press Ctrl+B to bookmark the current frequency, or enter one above "
    "and click Add. Use CSV to import a CHIRP channel list."
)


#: Keys a bookmark saved in this app has: every bookmark has a label and a
#: frequency; "demod" (a Mode list name) and "fm_deviation" (an FM deviation
#: choice such as "75 kHz") record how it was being listened to.
_BASIC_KEYS = frozenset(("label", "freq_hz", "demod", "fm_deviation"))
#: CHIRP's default mode; imported channels omit it when it is the default.
_CHIRP_DEFAULT_MODE = "FM"
#: Demodulator names of the control panel's Mode list.
_DEMOD_NAMES = frozenset(name for name, _tip in DEMOD_MODES)
#: The Mode list's "no audio" entry, which has no CHIRP equivalent, and
#: how it shows in the Mode column.
_IQ_DEMOD = "None (I/Q)"
_IQ_MODE_TEXT = "I/Q"
#: FM deviation choices for CHIRP's broadcast and narrowband FM modes.
_WFM_DEVIATION = "75 kHz"
_NFM_DEVIATION = "5 kHz"
#: Frequencies closer than this (Hz) are the same channel.
_SAME_CHANNEL_HZ = 10.0
#: Short names of the Add row's mode choices.
_ADD_MODE_TEXT = {_IQ_DEMOD: _IQ_MODE_TEXT}


def _num(value: Any, default: float = 0.0) -> float:
    """``float(value)``, or ``default`` for missing or malformed values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _plural(count: int, word: str) -> str:
    """Count with a plural-aware noun, e.g. "1 channel" or "3 channels"."""
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _trim_decimals(whole: str, point: str, frac: str) -> str:
    """Drop trailing zeros but keep at least 3 decimals ("146.520")."""
    return f"{whole}{point}{frac.rstrip('0').ljust(3, '0')}"


def format_mhz(freq_hz: Any) -> str:
    """Format a frequency as MHz with 3 to 6 decimals ("146.520 MHz")."""
    whole, point, frac = f"{_num(freq_hz) / 1e6:.6f}".partition(".")
    return f"{_trim_decimals(whole, point, frac)} MHz"


def default_bookmark_name(freq_hz: float) -> str:
    """A new bookmark's name: the matching preset's ("2m Calling"), else
    its band and frequency ("FM Broadcast 100.1"), else the frequency."""
    return default_name(freq_hz) or format_mhz(freq_hz)


def _mode(bookmark: Dict[str, Any]) -> str:
    """Mode of a bookmark as the list shows it: its listening mode ("AM",
    "WFM", "I/Q"), else its CHIRP mode; imported channels without one use
    CHIRP's FM; "" when it keeps the receiver's mode."""
    demod = str(bookmark.get("demod") or "")
    if demod == _IQ_DEMOD:
        return _IQ_MODE_TEXT
    if demod in _DEMOD_NAMES:
        chirp = _chirp_mode(demod, str(bookmark.get("fm_deviation") or ""))
        if chirp:
            return chirp
    mode = str(bookmark.get("mode") or "")
    if mode:
        return mode
    if is_chirp_channel(bookmark):
        return _CHIRP_DEFAULT_MODE
    return ""


def _chirp_mode(demod: str, fm_deviation: str = "") -> str:
    """CHIRP mode for a Mode list name ("" when CHIRP has none, e.g. I/Q)."""
    if demod == "FM":
        return "WFM" if fm_deviation == _WFM_DEVIATION else "FM"
    return demod if demod in ("AM", "USB", "LSB", "CW") else ""


def tune_mode(bookmark: Dict[str, Any]) -> Tuple[str, str]:
    """``(demod, fm_deviation)`` to listen to a bookmark with.

    ``demod`` is a Mode list name ("FM", "AM", "USB", ...) and
    ``fm_deviation`` an FM deviation choice ("5 kHz", "75 kHz"); either is
    empty when the bookmark doesn't say (old bookmarks keep the current
    mode). CHIRP modes are mapped: NFM/FM to FM at 5 kHz, WFM to FM at
    75 kHz, CWR to CW and so on.
    """
    demod = str(bookmark.get("demod") or "")
    if demod in _DEMOD_NAMES:
        deviation = str(bookmark.get("fm_deviation") or "") if demod == "FM" else ""
        return demod, deviation
    chirp = _mode(bookmark)
    if not chirp or chirp == _IQ_MODE_TEXT:
        return "", ""
    demod = chirp_mode_to_demod(chirp)
    if demod == DEFAULT_DEMOD or demod not in _DEMOD_NAMES:
        return "", ""  # e.g. RTTY or Auto: no matching demodulator here
    if demod == "FM":
        wide = chirp.strip().upper() == "WFM"
        return demod, _WFM_DEVIATION if wide else _NFM_DEVIATION
    return demod, ""


def listening(bookmark: Dict[str, Any]) -> Tuple[str, str, str, float]:
    """``(demod, fm_deviation, bandwidth, ctcss_tone_hz)`` a bookmark is
    listened to with (see :func:`tune_mode`): its own filter width ("" for
    the mode's usual one) and its CTCSS tone squelch (0: none)."""
    demod, deviation = tune_mode(bookmark)
    if not demod:
        return "", "", "", 0.0
    width = str(bookmark.get("bandwidth") or "")
    tone = bookmark_tone_hz(bookmark) if demod == "FM" else 0.0
    return demod, deviation, width, tone


def _new_bookmark(
    label: str,
    freq_hz: float,
    mode: str = "",
    fm_deviation: str = "",
    bandwidth: str = "",
) -> Dict[str, Any]:
    """A new bookmark (see :meth:`BookmarksPanel.add_bookmark`)."""
    bookmark: Dict[str, Any] = {"label": label, "freq_hz": float(freq_hz)}
    if mode in _DEMOD_NAMES:
        bookmark["demod"] = mode
        if mode == "FM" and fm_deviation:
            bookmark["fm_deviation"] = str(fm_deviation)
        chirp = _chirp_mode(mode, str(fm_deviation or ""))
        if chirp:
            bookmark["mode"] = chirp  # shown in the list, exported to CHIRP
        if bandwidth:
            bookmark["bandwidth"] = str(bandwidth)
    return bookmark


def same_channel(
    bookmark: Dict[str, Any], freq_hz: float, mode: str = "", deviation: str = ""
) -> bool:
    """Whether ``bookmark`` is ``freq_hz`` listened to in ``mode`` (and FM
    ``deviation``): a bookmark that keeps the receiver's mode, or a mode
    or deviation not given, matches any."""
    if abs(_num(bookmark.get("freq_hz")) - float(freq_hz)) >= _SAME_CHANNEL_HZ:
        return False
    demod, dev = tune_mode(bookmark)
    if demod and mode and demod != mode:
        return False
    if demod == "FM" and mode == "FM" and dev and deviation and dev != deviation:
        return False
    return True


def _offset_text(bookmark: Dict[str, Any]) -> str:
    """Short repeater-shift text such as "-0.6" (empty for simplex)."""
    duplex = str(bookmark.get("duplex") or "")
    if duplex in ("+", "-"):
        offset = abs(_num(bookmark.get("offset_hz"))) / 1e6
        return f"{duplex}{offset:g}"
    if duplex == "split":
        return "Split"
    if duplex == "off":
        return "RX only"
    return ""


def _details(bookmark: Dict[str, Any]) -> str:
    """Compact mode / tone / shift summary for the Mode column (and
    "· Skip" for a bookmark the bookmark scan passes by)."""
    parts = [p for p in (_mode(bookmark), str(bookmark.get("tone_mode") or "")) if p]
    shift = _offset_text(bookmark)
    if shift:
        parts.append(shift)
    text = " ".join(parts)
    if str(bookmark.get("skip") or "") == "S":
        text = f"{text} · Skip" if text else "Skip"
    return text


def _tooltip(bookmark: Dict[str, Any]) -> str:
    """Multi-line description of one bookmark for its tooltip."""
    lines = [
        str(bookmark.get("label", "")) or "(unnamed)",
        format_mhz(bookmark.get("freq_hz")),
    ]
    mode = _mode(bookmark)
    if mode:
        lines.append(f"Mode: {mode}")
    width = str(bookmark.get("bandwidth") or "")
    if width:
        lines.append(f"Filter: {width}")
    tone_mode = bookmark.get("tone_mode")
    if tone_mode in ("Tone", "TSQL"):
        key = "rtone" if tone_mode == "Tone" else "ctone"
        lines.append(f"{tone_mode}: {_num(bookmark.get(key), 88.5):g} Hz")
    elif tone_mode:
        lines.append(f"Tone mode: {tone_mode}")
    duplex = str(bookmark.get("duplex") or "")
    if duplex in ("+", "-"):
        lines.append(f"Shift: {_offset_text(bookmark)} MHz")
    elif duplex == "split":
        lines.append(f"TX: {format_mhz(bookmark.get('offset_hz'))}")
    elif duplex == "off":
        lines.append("Receive only")
    if bookmark.get("group"):
        lines.append(f"Group: {bookmark['group']}")
    if str(bookmark.get("skip") or "") == "S":
        lines.append("Skipped by Scan Bookmarks")
    if bookmark.get("comment"):
        lines.append(str(bookmark["comment"]))
    lines.append("Double-click or press Enter to tune; right-click for more")
    return "\n".join(lines)


def _matches(bookmark: Dict[str, Any], words: List[str]) -> bool:
    """Whether every word of a filter is in the bookmark's name, frequency,
    mode, group or comment."""
    haystack = " ".join(
        (
            str(bookmark.get("label", "")),
            format_mhz(bookmark.get("freq_hz")),
            f"{_num(bookmark.get('freq_hz')) / 1e6:.6f}",
            _details(bookmark),
            str(bookmark.get("group", "") or ""),
            str(bookmark.get("comment", "") or ""),
        )
    ).lower()
    return all(word in haystack for word in words)


class _MhzSpinBox(QDoubleSpinBox if HAS_PYQT6 else object):
    """MHz spin box that accepts 1 Hz steps but shows 3 to 6 decimals, like
    the list ("146.520 MHz", not "146.520000 MHz")."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        loc = self.locale()
        text = loc.toString(float(value), "f", self.decimals())
        text = text.replace(loc.groupSeparator(), "")
        whole, point, frac = text.partition(loc.decimalPoint())
        return _trim_decimals(whole, point, frac) if point else text


class BookmarksPanel(QWidget if HAS_PYQT6 else object):
    """Panel to manage saved frequency bookmarks."""

    if HAS_PYQT6:
        # freq_hz, label, demod mode and FM deviation to listen with (see
        # tune_mode(); both "" when the bookmark doesn't say: keep the
        # current mode). Slots may take just (freq_hz, label).
        tune_requested = pyqtSignal(float, str, str, str)
        # Emitted right after tune_requested with the whole bookmark, for
        # what tune_requested can't carry: its filter width and CTCSS tone
        # (see listening()).
        bookmark_tuned = pyqtSignal(object)
        # "Scan from Here": the index of the bookmark a bookmark scan
        # starts at.
        scan_requested = pyqtSignal(int)
        # A short message for the status bar: (text, tone).
        message = pyqtSignal(str, str)
        # Bookmarks were added, edited, removed or imported.
        bookmarks_changed = pyqtSignal()

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._settings = GuiSettings()
        self._bookmarks: List[Dict[str, Any]] = self._load_bookmarks()
        # The tuned frequency and a callable returning the current
        # (demod, fm_deviation), so "Add" can store how it is listened to.
        self._tuned_hz: Optional[float] = None
        self._mode_source: Optional[Callable[[], Tuple[str, str]]] = None
        # And the current filter width (Bandwidth), saved with it too.
        self._width_source: Optional[Callable[[], str]] = None
        # The Add row's Mode follows the frequency (and the tuned mode)
        # until the user picks one.
        self._mode_follows = True
        # Bookmark index -> its row; groups the user collapsed.
        self._items: Dict[int, "QTreeWidgetItem"] = {}
        self._collapsed: set = set()
        self._setup_ui()
        self._refresh_list()
        self._sync_add_mode()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Add row: name, frequency (MHz), mode, Add.
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Name (optional)")
        self._label_input.setClearButtonEnabled(True)
        self._label_input.setToolTip(
            "Name for the new bookmark (optional: when left empty it is named "
            "after the preset or band, e.g. “FM Broadcast 100.1”). "
            "Press Enter to add."
        )
        self._label_input.setAccessibleName("Bookmark name")
        self._label_input.setMinimumWidth(80)
        self._label_input.returnPressed.connect(self._add_current)
        add_row.addWidget(self._label_input, 1)

        self._freq_input = _MhzSpinBox()
        # Same range as the tuner, so Add never saves a clamped frequency.
        self._freq_input.setRange(MIN_FREQUENCY_HZ / 1e6, MAX_FREQUENCY_HZ / 1e6)
        self._freq_input.setDecimals(6)
        # Follows the tuner (set_current_frequency); this is its start-up value.
        self._freq_input.setValue(DEFAULT_FREQUENCY_HZ / 1e6)
        self._freq_input.setSuffix(" MHz")
        self._freq_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._freq_input.setAccelerated(True)
        self._freq_input.setToolTip("Frequency of the new bookmark, in MHz")
        self._freq_input.setAccessibleName("Bookmark frequency")
        self._freq_input.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        spin_edit = self._freq_input.lineEdit()
        if spin_edit is not None:
            spin_edit.returnPressed.connect(self._add_current)
        self._freq_input.valueChanged.connect(lambda _v: self._sync_add_mode())
        add_row.addWidget(self._freq_input)

        self._mode_input = QComboBox()
        for name, _tip in DEMOD_MODES:
            self._mode_input.addItem(_ADD_MODE_TEXT.get(name, name), name)
        self._mode_input.setToolTip(
            "How to listen to the new bookmark. It follows the mode you are "
            "listening in, or the band's usual mode for a frequency typed in "
            "(airband: AM); pick another to save that instead."
        )
        self._mode_input.setAccessibleName("Bookmark mode")
        self._mode_input.activated.connect(self._on_add_mode_picked)
        add_row.addWidget(self._mode_input)

        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip(
            "Save this name, frequency and mode as a bookmark (Enter)"
        )
        self._add_btn.setAutoDefault(False)
        self._add_btn.clicked.connect(self._add_current)
        add_row.addWidget(self._add_btn)
        layout.addLayout(add_row)

        # Filter: narrows the list to the bookmarks whose name, frequency,
        # mode, group or comment hold every word typed.
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter by name, frequency, mode or group")
        self._filter.setClearButtonEnabled(True)
        self._filter.setAccessibleName("Filter bookmarks")
        self._filter.setToolTip(
            "Show only the bookmarks whose name, frequency, mode, group or "
            "comment contain these words (Esc clears it)"
        )
        self._filter.textChanged.connect(lambda _t: self._apply_filter())
        self._filter.installEventFilter(self)
        layout.addWidget(self._filter)

        # Bookmark list: name | frequency | mode, in group folders.
        self._list = QTreeWidget()
        self._list.setColumnCount(3)
        self._list.setHeaderLabels(["Name", "Frequency", "Mode"])
        self._list.setRootIsDecorated(False)
        self._list.setUniformRowHeights(True)
        self._list.setAlternatingRowColors(True)
        self._list.setAllColumnsShowFocus(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        # (Header and two rows: with the filter row, the tab still fits a
        # 1024x640 window without scrolling.)
        self._list.setMinimumHeight(70)
        self._list.setAccessibleName("Bookmarks")
        header = self._list.header()
        header.setStretchLastSection(False)
        header.setHighlightSections(False)
        header.setSectionsMovable(False)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_FREQ, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_MODE, QHeaderView.ResizeMode.ResizeToContents)
        header_item = self._list.headerItem()
        if header_item is not None:
            header_item.setTextAlignment(
                COL_FREQ, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        self._list.itemDoubleClicked.connect(self._on_item_activated)
        self._list.itemSelectionChanged.connect(self._update_buttons)
        self._list.itemExpanded.connect(self._on_group_toggled)
        self._list.itemCollapsed.connect(self._on_group_toggled)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.installEventFilter(self)
        self._empty = ViewPlaceholder(self._list, _EMPTY_TEXT)
        layout.addWidget(self._list, 1)

        # Actions on the selection, plus CHIRP CSV import/export.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._tune_btn = QPushButton("Tune")
        self._tune_btn.clicked.connect(self._tune_selected)
        btn_row.addWidget(self._tune_btn)

        # Edit... (it renames too; so does F2 or the right-click menu).
        self._edit_btn = QPushButton("Edit...")
        self._edit_btn.clicked.connect(self._edit_selected)
        btn_row.addWidget(self._edit_btn)
        # The former Rename... button's name, kept for code that uses it.
        self._rename_btn = self._edit_btn

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._remove_btn)
        for btn in (self._tune_btn, self._edit_btn, self._remove_btn):
            btn.setAutoDefault(False)

        btn_row.addStretch(1)

        # Named like the File menu's commands (and the dialogs they open).
        self._csv_menu = QMenu(self)
        self._import_action = self._csv_menu.addAction(
            "Import Bookmarks (CHIRP CSV)..."
        )
        self._import_action.setToolTip(
            "Add a CHIRP CSV file's memory channels as bookmarks (commas, "
            "semicolons or tabs between the columns); frequencies already "
            "bookmarked are skipped when adding to the list"
        )
        self._import_action.triggered.connect(lambda _checked=False: self.import_csv())
        self._export_action = self._csv_menu.addAction(
            "Export Bookmarks (CHIRP CSV)..."
        )
        self._export_action.setToolTip(
            "Save the bookmarks as a CHIRP-compatible CSV file"
        )
        self._export_action.triggered.connect(lambda _checked=False: self.export_csv())
        self._csv_menu.addSeparator()
        self._presets_action = self._csv_menu.addAction(
            "Export Built-in Presets (CHIRP CSV)..."
        )
        self._presets_action.setToolTip(
            "Save the application's built-in presets (NOAA weather, "
            "calling frequencies, ACARS...) as a CHIRP CSV file, e.g. to load "
            "them into a handheld radio with CHIRP"
        )
        self._presets_action.triggered.connect(
            lambda _checked=False: self.export_presets_csv()
        )
        self._csv_menu.setToolTipsVisible(True)

        self._csv_btn = QPushButton("CSV")
        self._csv_btn.setMenu(self._csv_menu)
        self._csv_btn.setAutoDefault(False)
        self._csv_btn.setToolTip(
            "Import or export the bookmarks as a CHIRP-compatible CSV file, to "
            "move them between this app, CHIRP and a handheld radio's memories"
        )
        btn_row.addWidget(self._csv_btn)
        layout.addLayout(btn_row)

    # ---- Public API ----
    def add_bookmark(
        self,
        label: str,
        freq_hz: float,
        mode: str = "",
        fm_deviation: str = "",
        bandwidth: str = "",
    ) -> int:
        """Save a bookmark (public, so other parts of the UI can add them).

        Args:
            label: Name shown in the list.
            freq_hz: Frequency in Hz.
            mode: Demodulation mode to restore when it is tuned (a Mode list
                name such as "FM" or "USB"); "" leaves the mode alone.
            fm_deviation: FM deviation choice (e.g. "75 kHz"), for FM.
            bandwidth: Filter width to restore (e.g. "25 kHz"); "" for the
                mode's usual one.

        Returns:
            The new bookmark's index.
        """
        self._bookmarks.append(
            _new_bookmark(label, freq_hz, mode, fm_deviation, bandwidth)
        )
        self._save()
        index = len(self._bookmarks) - 1
        self._refresh_list(select=index)
        return index

    def add_unique_many(
        self, entries: List[Tuple[str, float, str, str, str]]
    ) -> List[Tuple[int, bool]]:
        """:meth:`add_unique` for many ``(label, freq_hz, mode,
        fm_deviation, bandwidth)`` at once (the Scanner's Bookmark All):
        saved, and the list rebuilt, once. An entry already bookmarked (or
        twice in ``entries``) isn't added again. Returns ``(index, added)``
        per entry; the last one added (else the first one already there) is
        selected."""
        results: List[Tuple[int, bool]] = []
        for label, freq_hz, mode, fm_deviation, bandwidth in entries:
            index = self.find_bookmark(freq_hz, mode, fm_deviation)
            if index >= 0:
                results.append((index, False))
                continue
            self._bookmarks.append(
                _new_bookmark(label, freq_hz, mode, fm_deviation, bandwidth)
            )
            results.append((len(self._bookmarks) - 1, True))
        added = [index for index, new in results if new]
        if added:
            self._save()
            self._refresh_list(select=added[-1])
        elif results:
            self.select_bookmark(results[0][0])
        return results

    def add_unique(
        self,
        label: str,
        freq_hz: float,
        mode: str = "",
        fm_deviation: str = "",
        bandwidth: str = "",
    ) -> Tuple[int, bool]:
        """Add a bookmark unless the same channel is already bookmarked
        (then select that one). Returns ``(index, added)``."""
        index = self.find_bookmark(freq_hz, mode, fm_deviation)
        if index >= 0:
            self.select_bookmark(index)
            return index, False
        return self.add_bookmark(label, freq_hz, mode, fm_deviation, bandwidth), True

    def find_bookmark(
        self, freq_hz: float, mode: str = "", fm_deviation: str = ""
    ) -> int:
        """Index of the bookmark of ``freq_hz`` in ``mode`` (see
        :func:`same_channel`), or -1."""
        for index, bookmark in enumerate(self._bookmarks):
            if same_channel(bookmark, freq_hz, mode, fm_deviation):
                return index
        return -1

    def set_tone(self, index: int, tone_hz: float) -> None:
        """Give bookmark ``index`` a CTCSS tone squelch (0: none), as
        CHIRP's TSQL tone."""
        if not 0 <= index < len(self._bookmarks):
            return
        b = self._bookmarks[index]
        if tone_hz and tone_hz > 0:
            b["tone_mode"] = "TSQL"
            b["ctone"] = float(tone_hz)
        elif str(b.get("tone_mode") or "") == "TSQL":
            b.pop("tone_mode", None)
            b.pop("ctone", None)
        self._save()
        self._refresh_list(select=index)

    def bookmark(self, index: int) -> Optional[Dict[str, Any]]:
        """A copy of bookmark ``index``, or None."""
        if 0 <= index < len(self._bookmarks):
            return dict(self._bookmarks[index])
        return None

    def bookmarks(self) -> List[Dict[str, Any]]:
        """Copies of all bookmarks, in order."""
        return [dict(b) for b in self._bookmarks]

    def groups(self) -> List[str]:
        """The group names in use, in the order they first appear."""
        names: List[str] = []
        for bookmark in self._bookmarks:
            group = str(bookmark.get("group", "") or "")
            if group and group not in names:
                names.append(group)
        return names

    def select_bookmark(self, index: int) -> None:
        """Select (and show) bookmark ``index``; clears a filter hiding it."""
        item = self._items.get(index)
        if item is None:
            return
        if item.isHidden():
            self._filter.clear()
        parent = item.parent()
        if parent is not None:
            parent.setExpanded(True)
        self._list.setCurrentItem(item)
        self._list.scrollToItem(item)

    def set_mode_source(self, source: Optional[Callable[[], Tuple[str, str]]]) -> None:
        """Set a callable returning the current ``(demod, fm_deviation)``.

        "Add" then saves the listening mode with the bookmark when its
        frequency is the tuned one (see :meth:`set_current_frequency`).
        """
        self._mode_source = source
        self._sync_add_mode()

    def set_width_source(self, source: Optional[Callable[[], str]]) -> None:
        """Set a callable returning the current filter width ("25 kHz"),
        saved with a bookmark added at the tuned frequency."""
        self._width_source = source

    def refresh_mode(self) -> None:
        """The listening mode changed: the Add row's Mode follows it."""
        self._sync_add_mode()

    def bookmark_count(self) -> int:
        """Number of saved bookmarks."""
        return len(self._bookmarks)

    # ---- CHIRP CSV import / export ----
    def import_csv(self, path: Optional[str] = None) -> int:
        """
        Load memory channels from a CHIRP CSV file.

        Prompts for the file (when `path` is omitted) and for whether to
        replace the existing channels or append to them. Appending skips
        channels that are already bookmarked.

        Args:
            path: CSV file to read; a file dialog is shown when omitted.

        Returns:
            Number of channels imported (0 if cancelled or on error).
        """
        if not path:
            # (In the folder last used for CHIRP files, else Documents: not
            # the folder the app was started from. mc:MC-U07)
            path, _ = file_dialogs.open_file(
                self,
                self._settings,
                "channels",
                "Import Bookmarks (CHIRP CSV)",
                CSV_FILTER,
            )
        if not path:
            return 0

        try:
            report = read_chirp_csv(path, strict=False)
        except ChirpCsvError as exc:
            logger.warning("Channel import failed: %s", exc)
            QMessageBox.warning(self, "Import Failed", str(exc))
            return 0

        if not report.channels:
            QMessageBox.information(
                self, "Import Bookmarks", "No channels found in that file."
            )
            return 0

        imported = [c.to_bookmark() for c in report.channels]
        duplicates: List[str] = []

        if self._bookmarks:
            choice = self._ask_replace_or_append(len(imported))
            if choice is None:
                return 0
            if choice == "replace":
                self._bookmarks = imported
            else:
                fresh = []
                for bookmark in imported:
                    demod, deviation = tune_mode(bookmark)
                    known = self.find_bookmark(
                        _num(bookmark.get("freq_hz")), demod, deviation
                    )
                    if known >= 0:
                        duplicates.append(str(bookmark.get("label") or "(unnamed)"))
                    else:
                        fresh.append(bookmark)
                imported = fresh
                self._bookmarks = self._bookmarks + imported
        else:
            self._bookmarks = imported

        self._save()
        self._refresh_list()

        notes = []
        if duplicates:
            names = ", ".join(duplicates[:10]) + ("..." if len(duplicates) > 10 else "")
            notes.append(f"Already bookmarked, so skipped: {names}")
        if report.skipped:
            notes.append(
                f"Skipped {_plural(len(report.skipped), 'unreadable row')}:\n"
                + "\n".join(report.skipped[:10])
            )
        if notes:
            first = (
                f"Imported {_plural(len(imported), 'bookmark')}."
                if imported
                else "Nothing new to import."
            )
            QMessageBox.information(
                self, "Import Bookmarks", first + "\n" + "\n".join(notes)
            )
        return len(imported)

    def _ask_replace_or_append(self, count: int) -> Optional[str]:
        """Ask how to merge imported channels: "replace", "append" or None."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Import Bookmarks")
        box.setText(f"Import {_plural(count, 'bookmark')}.")
        box.setInformativeText(
            f"Replace the existing {_plural(len(self._bookmarks), 'bookmark')}, "
            "or add the new ones after them (frequencies already bookmarked "
            "are skipped)?"
        )
        replace_btn = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
        append_btn = box.addButton("Append", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(append_btn)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()  # it is parented to the panel; don't keep it around
        if clicked is replace_btn:
            return "replace"
        if clicked is append_btn:
            return "append"
        return None

    def export_csv(self, path: Optional[str] = None) -> int:
        """
        Save the current channels to a CHIRP-compatible CSV file.

        Args:
            path: Destination file; a file dialog is shown when omitted.

        Returns:
            Number of channels written (0 if cancelled or on error).
        """
        if not self._bookmarks:
            QMessageBox.information(
                self, "Export Bookmarks", "There are no bookmarks to export."
            )
            return 0

        if not path:
            path, _ = file_dialogs.save_file(
                self,
                self._settings,
                "channels",
                "Export Bookmarks (CHIRP CSV)",
                "bookmarks.csv",
                CSV_FILTER,
            )
        if not path:
            return 0
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            count = export_bookmarks_csv(path, self._bookmarks)
        except OSError as exc:
            logger.warning("Channel export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
            return 0

        QMessageBox.information(
            self,
            "Export Bookmarks",
            f"Exported {_plural(count, 'bookmark')} to:\n{path}",
        )
        return count

    def export_presets_csv(self, path: Optional[str] = None) -> int:
        """Save the built-in RX presets as a CHIRP CSV file (as ``sdr-scan
        channels export --presets`` does). Returns the number written."""
        from ..core.frequency_manager import RX_PRESETS

        if not path:
            path, _ = file_dialogs.save_file(
                self,
                self._settings,
                "channels",
                "Export Built-in Presets (CHIRP CSV)",
                "sdr-module-presets.csv",
                CSV_FILTER,
            )
        if not path:
            return 0
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            count = write_chirp_csv(
                path, presets_to_channels(RX_PRESETS), renumber=True
            )
        except OSError as exc:
            logger.warning("Preset export failed: %s", exc)
            QMessageBox.warning(self, "Export Failed", str(exc))
            return 0
        QMessageBox.information(
            self,
            "Export Built-in Presets",
            f"Exported {_plural(count, 'preset')} to:\n{path}\n\nIt opens in "
            "CHIRP (File > Open), or import it here with CSV > Import Bookmarks.",
        )
        return count

    def set_current_frequency(self, freq_hz: float) -> None:
        """Update the 'add new' frequency to match the main tuner."""
        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(freq_hz):
            return
        self._tuned_hz = freq_hz
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(freq_hz / 1e6)
        self._freq_input.blockSignals(False)
        self._sync_add_mode()

    # ---- Internal ----
    def _load_bookmarks(self) -> List[Dict[str, Any]]:
        """The saved bookmarks. An entry of a damaged settings file that
        isn't one (null, a number, no usable frequency) is skipped with a
        log line: it must not stop the app starting."""
        bookmarks, dropped = valid_bookmarks(self._settings.get_bookmarks())
        if dropped:
            logger.warning(
                "Ignored %s in the saved bookmarks.", describe_dropped(dropped)
            )
        return bookmarks

    def _save(self) -> None:
        self._settings.set_bookmarks(self._bookmarks)
        self.bookmarks_changed.emit()

    def _refresh_list(self, select: Optional[int] = None):
        """Rebuild the rows from ``self._bookmarks`` (optionally selecting one).

        Bookmarks with a group are listed under a folder per group (in the
        order the groups first appear), the others after them.
        """
        self._list.clear()
        self._items = {}
        any_details = False
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        groups = self.groups()
        folders: Dict[str, QTreeWidgetItem] = {}
        bold = QFont(self._list.font())
        bold.setBold(True)
        for group in groups:
            folder = QTreeWidgetItem([group, "", ""])
            folder.setFont(COL_NAME, bold)
            folder.setFlags(Qt.ItemFlag.ItemIsEnabled)  # not a bookmark
            folders[group] = folder
            self._list.addTopLevelItem(folder)
        for idx, b in enumerate(self._bookmarks):
            freq = _num(b.get("freq_hz"))
            label = str(b.get("label", "")) or "(unnamed)"
            details = _details(b)
            any_details = any_details or bool(details)
            item = QTreeWidgetItem([label, format_mhz(freq), details])
            item.setTextAlignment(COL_FREQ, right)
            item.setData(COL_NAME, Qt.ItemDataRole.UserRole, idx)
            tip = _tooltip(b)
            for col in (COL_NAME, COL_FREQ, COL_MODE):
                item.setToolTip(col, tip)
            folder = folders.get(str(b.get("group", "") or ""))
            if folder is not None:
                folder.addChild(item)
            else:
                self._list.addTopLevelItem(item)
            self._items[idx] = item
        for group, folder in folders.items():
            count = folder.childCount()
            folder.setText(COL_FREQ, _plural(count, "bookmark"))
            folder.setTextAlignment(COL_FREQ, right)
            folder.setToolTip(
                COL_NAME, f"Group “{group}”: {_plural(count, 'bookmark')}"
            )
            folder.setExpanded(group not in self._collapsed)
        self._list.setRootIsDecorated(bool(folders))
        # Imported CHIRP channels carry mode/tone detail; hide the column
        # when no bookmark has any, so names get the width.
        self._list.setColumnHidden(COL_MODE, not any_details)
        self._apply_filter()
        if select is not None and select in self._items:
            self.select_bookmark(select)
        self._update_buttons()

    def _apply_filter(self) -> None:
        """Hide the bookmarks the filter doesn't match (and empty groups)."""
        words = self._filter.text().lower().split()
        shown = 0
        for idx, item in self._items.items():
            visible = not words or _matches(self._bookmarks[idx], words)
            item.setHidden(not visible)
            shown += visible
        for i in range(self._list.topLevelItemCount()):
            top = self._list.topLevelItem(i)
            if top is None or top.childCount() == 0:
                continue
            visible_children = any(
                not top.child(c).isHidden() for c in range(top.childCount())
            )
            top.setHidden(not visible_children)
            if words and visible_children:
                top.setExpanded(True)
        if not self._bookmarks:
            self._empty.set_text(_EMPTY_TEXT)
            self._empty.set_visible(True)
        elif not shown:
            text = self._filter.text().strip()
            self._empty.set_text(
                f"No bookmarks match “{text}”\n"
                f"Clear the filter (Esc) to see all {_plural(len(self._bookmarks), 'bookmark')}."
            )
            self._empty.set_visible(True)
        else:
            self._empty.set_visible(False)
        selected = self._list.selectedItems()
        if selected and selected[0].isHidden():
            self._list.clearSelection()

    def _on_group_toggled(self, item: "QTreeWidgetItem") -> None:
        if item.childCount():
            group = item.text(COL_NAME)
            if item.isExpanded():
                self._collapsed.discard(group)
            else:
                self._collapsed.add(group)

    def _update_buttons(self) -> None:
        """Enable the selection actions only when a bookmark is selected."""
        has_sel = self._selected_index() >= 0
        for btn, tip in (
            (self._tune_btn, "Tune the receiver to the selected bookmark (Enter)"),
            (
                self._edit_btn,
                "Edit the selected bookmark: name, frequency, mode, filter, "
                "tone, group (F2 renames it)",
            ),
            (self._remove_btn, "Delete the selected bookmark (Delete)"),
        ):
            btn.setEnabled(has_sel)
            btn.setToolTip(tip if has_sel else "Select a bookmark in the list first")
        has_any = bool(self._bookmarks)
        self._export_action.setEnabled(has_any)
        self._export_action.setToolTip(
            "Save the bookmarks as a CHIRP-compatible CSV file"
            if has_any
            else "There are no bookmarks to export yet"
        )

    def _selected_index(self) -> int:
        items = self._list.selectedItems()
        if not items:
            return -1
        return self._index_of(items[0])

    def _index_of(self, item: Optional["QTreeWidgetItem"]) -> int:
        if item is None:
            return -1
        idx = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self._bookmarks):
            return idx
        return -1

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 (Qt API)
        """List keys: Enter tunes, Delete/Backspace removes, F2 renames;
        Esc in the filter clears it."""
        # (Events arrive while the panel is still being built.)
        if obj is getattr(self, "_list", None) and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._tune_selected()
                return True
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._remove_selected()
                return True
            if key == Qt.Key.Key_F2:
                self._rename_selected()
                return True
        if (
            obj is getattr(self, "_filter", None)
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
            and self._filter.text()
        ):
            self._filter.clear()
            return True
        return super().eventFilter(obj, event)

    def _current_mode(self, freq_hz: float) -> Tuple[str, str]:
        """How the tuned frequency is listened to, if ``freq_hz`` is it."""
        if (
            self._mode_source is None
            or self._tuned_hz is None
            or abs(freq_hz - self._tuned_hz) >= 0.5
        ):
            return "", ""  # a typed-in frequency: its mode is unknown
        try:
            mode, deviation = self._mode_source()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not read the current mode: %s", exc)
            return "", ""
        return str(mode or ""), str(deviation or "")

    def _current_width(self) -> str:
        if self._width_source is None:
            return ""
        try:
            return str(self._width_source() or "")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not read the current filter width: %s", exc)
            return ""

    def _add_listening(self, freq_hz: float) -> Tuple[str, str, str]:
        """``(demod, fm_deviation, bandwidth)`` the Add row saves.

        The current mode: with its deviation and filter for the tuned
        frequency, or for a frequency typed in the same band. A frequency
        typed in another band gets that band's usual mode (airband: AM, not
        the broadcast FM being listened to). "" when nothing says (no
        receiver mode known).
        """
        current, deviation = self._current_mode(freq_hz)
        if current:
            return current, deviation, self._current_width()
        band = listening_for(freq_hz)
        tuned = self._tuned_hz
        same_band = tuned is not None and band_for(tuned) == band_for(freq_hz)
        if band is not None and not (same_band and self._mode_source is not None):
            return band
        if self._mode_source is not None:
            try:
                mode, deviation = self._mode_source()
            except Exception:  # pragma: no cover - defensive
                return "", "", ""
            mode, deviation = str(mode or ""), str(deviation or "")
            return mode, deviation if mode == "FM" else "", ""
        return "", "", ""

    def _sync_add_mode(self) -> None:
        """Show in the Add row's Mode what Add will save (unless the user
        picked a mode)."""
        if not self._mode_follows:
            return
        demod = self._add_listening(float(round(self._freq_input.value() * 1e6)))[0]
        # (Blank when nothing says how: Add then keeps the receiver's mode.)
        index = self._mode_input.findData(demod) if demod else -1
        self._mode_input.blockSignals(True)
        self._mode_input.setCurrentIndex(index)
        self._mode_input.blockSignals(False)

    def _on_add_mode_picked(self, _index: int) -> None:
        self._mode_follows = False

    def showEvent(self, event: Any) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._sync_add_mode()

    def _add_current(self):
        freq_hz = float(round(self._freq_input.value() * 1e6))
        label = self._label_input.text().strip() or default_bookmark_name(freq_hz)
        demod, deviation, width = self._add_listening(freq_hz)
        picked = self._mode_input.currentData()
        if not self._mode_follows and isinstance(picked, str) and picked != demod:
            # The user picked another mode: the band's settings for it, if
            # the band is heard in it, else the mode's usual ones.
            demod, deviation, width = picked, "", ""
            band = listening_for(freq_hz)
            if band is not None and band[0] == picked:
                demod, deviation, width = band
        index = self.find_bookmark(freq_hz, demod, deviation)
        if index >= 0:
            self.select_bookmark(index)
            name = str(self._bookmarks[index].get("label") or format_mhz(freq_hz))
            self.message.emit(f"Already bookmarked: {name}", "info")
        else:
            self.add_bookmark(label, freq_hz, demod, deviation, width)
            self.message.emit(f"Bookmarked {label}", "success")
        self._label_input.clear()
        self._mode_follows = True
        self._sync_add_mode()

    def _tune_index(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        b = self._bookmarks[idx]
        mode, deviation = tune_mode(b)
        self.tune_requested.emit(
            _num(b.get("freq_hz")), str(b.get("label", "")), mode, deviation
        )
        self.bookmark_tuned.emit(dict(b))

    def _tune_selected(self):
        self._tune_index(self._selected_index())

    def _on_item_activated(self, item: "QTreeWidgetItem", _column: int = 0):
        idx = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int):
            self._tune_index(idx)

    def _rename_selected(self):
        self._rename_index(self._selected_index())

    def _rename_index(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        current = str(self._bookmarks[idx].get("label", ""))
        new_label, ok = QInputDialog.getText(
            self, "Rename Bookmark", "Name:", text=current
        )
        if ok and new_label.strip():
            self._bookmarks[idx]["label"] = new_label.strip()
            self._save()
            self._refresh_list(select=idx)

    def _edit_selected(self) -> None:
        self.edit_bookmark(self._selected_index())

    def edit_bookmark(self, idx: int) -> bool:
        """Open the Edit Bookmark dialog for bookmark ``idx``; True when
        saved."""
        if not 0 <= idx < len(self._bookmarks):
            return False
        old = self._bookmarks[idx]
        edited = _edit_dialog(self, dict(old), tune_mode(old), self.groups())
        if edited is None:
            return False
        self._bookmarks[idx] = edited
        self._save()
        self._refresh_list(select=idx)
        return True

    def _show_context_menu(self, pos: Any) -> None:
        """Right-click: Tune, Edit, Rename, Copy Frequency, Remove, Scan
        from Here."""
        item = self._list.itemAt(pos)
        idx = self._index_of(item)
        if idx < 0:
            return
        self._list.setCurrentItem(item)
        menu = self._context_menu(idx)
        menu.exec(self._list.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _context_menu(self, idx: int) -> "QMenu":
        """The right-click menu for bookmark ``idx``."""
        b = self._bookmarks[idx]
        freq = format_mhz(b.get("freq_hz"))
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        tune = menu.addAction(f"Tune to {freq}")
        tune.triggered.connect(lambda _c=False, i=idx: self._tune_index(i))
        menu.addSeparator()
        edit = menu.addAction("Edit...")
        edit.triggered.connect(lambda _c=False, i=idx: self.edit_bookmark(i))
        rename = menu.addAction("Rename...\tF2")
        rename.triggered.connect(lambda _c=False, i=idx: self._rename_index(i))
        copy = menu.addAction("Copy Frequency")
        copy.setToolTip(f"Copy {freq} to the clipboard")
        copy.triggered.connect(lambda _c=False, i=idx: self._copy_frequency(i))
        up = menu.addAction("Move Up")
        up.setToolTip("Earlier in the list (and in an exported CHIRP file)")
        up.setEnabled(self._neighbour(idx, -1) >= 0)
        up.triggered.connect(lambda _c=False, i=idx: self.move_bookmark(i, -1))
        down = menu.addAction("Move Down")
        down.setToolTip("Later in the list (and in an exported CHIRP file)")
        down.setEnabled(self._neighbour(idx, 1) >= 0)
        down.triggered.connect(lambda _c=False, i=idx: self.move_bookmark(i, 1))
        menu.addSeparator()
        scan = menu.addAction("Scan Bookmarks from Here")
        scan.setToolTip(
            "Scan the bookmarks in the Scanner tab, starting with this one: it "
            "stops on each one that is active, in its mode"
        )
        scan.setEnabled(self.receivers(self.scan_requested) > 0)
        scan.triggered.connect(lambda _c=False, i=idx: self.scan_requested.emit(i))
        menu.addSeparator()
        remove = menu.addAction("Remove...\tDelete")
        remove.triggered.connect(lambda _c=False, i=idx: self._remove_index(i))
        return menu

    def _neighbour(self, idx: int, direction: int) -> int:
        """The bookmark shown just before (-1) or after (+1) bookmark
        ``idx`` in its group, or -1."""
        if not 0 <= idx < len(self._bookmarks):
            return -1
        group = str(self._bookmarks[idx].get("group", "") or "")
        j = idx + direction
        while 0 <= j < len(self._bookmarks):
            if str(self._bookmarks[j].get("group", "") or "") == group:
                return j
            j += direction
        return -1

    def move_bookmark(self, idx: int, direction: int) -> bool:
        """Swap bookmark ``idx`` with its neighbour in its group (the order
        of the list and of an exported CHIRP file); True when moved."""
        other = self._neighbour(idx, direction)
        if other < 0:
            return False
        items = self._bookmarks
        items[idx], items[other] = items[other], items[idx]
        self._save()
        self._refresh_list(select=other)
        return True

    def _copy_frequency(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        text = format_mhz(self._bookmarks[idx].get("freq_hz"))
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.message.emit(f"Copied {text}", "info")

    def _confirm_remove(self, bookmark: Dict[str, Any]) -> bool:
        """Ask before deleting a bookmark (there is no undo)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Remove Bookmark")
        name = str(bookmark.get("label", "")) or "(unnamed)"
        freq = format_mhz(_num(bookmark.get("freq_hz")))
        box.setText(f"Remove “{name}” ({freq})?")
        box.setInformativeText("This can't be undone.")
        remove_btn = box.addButton("&Remove", QMessageBox.ButtonRole.DestructiveRole)
        # Enter must not confirm an irreversible delete (a stray Delete or
        # Backspace followed by Enter would lose the bookmark); Alt+R does.
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        removed = box.clickedButton() is remove_btn
        box.deleteLater()  # it is parented to the panel; don't keep it around
        return removed

    def _remove_selected(self):
        self._remove_index(self._selected_index())

    def _remove_index(self, idx: int) -> None:
        if not 0 <= idx < len(self._bookmarks):
            return
        if not self._confirm_remove(self._bookmarks[idx]):
            return
        del self._bookmarks[idx]
        self._save()
        # Keep a selection nearby so repeated Delete presses keep working.
        self._refresh_list(
            select=min(idx, len(self._bookmarks) - 1) if self._bookmarks else None
        )
