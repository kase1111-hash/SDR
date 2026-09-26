"""
Main application window.

Provides the primary window with all panels and controls::

    +-----------------------------------------------------------------+
    | File  Device  Radio  View  Tools  Help                          |
    | [Start] [Record] | FREQ 100.000 MHz | LEVEL -40.0 dBFS          |
    +--------------------------------------+--------------------------+
    | spectrum                             | control panel (scrolls)  |
    |--------------------------------------|--------------------------|
    | waterfall                            | Decoder | Bookmarks | .. |
    +--------------------------------------+--------------------------+
    | [RUNNING] Demo device  2.4 MS/s   messages...           [REC]   |

All three splitters are user-resizable; their sizes persist between launches
and View > Reset Layout restores the defaults.
"""

from __future__ import annotations

import logging
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QEvent, Qt, QTimer
    from PyQt6.QtGui import QAction, QActionGroup, QFont, QFontMetrics, QKeySequence
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGroupBox,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .. import __version__
from ..devices.base import SDRDevice
from ..utils.tooltips import get_short_tip
from .audio_sink import AudioSink
from .bookmarks_panel import BookmarksPanel
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .settings_store import GuiSettings
from .spectrum_widget import SpectrumWidget
from .themes import (
    apply_theme,
    current_theme,
    normalize_theme,
    set_role,
    set_tone,
    theme_notifier,
)
from .waterfall_widget import WaterfallWidget

# Band presets shown in the Radio > Band Presets menu
BAND_PRESETS = [
    ("FM Broadcast", 100.1e6, "FM"),
    ("NOAA Weather", 162.55e6, "FM"),
    ("2m Ham (146.52)", 146.52e6, "FM"),
    ("70cm Ham (446)", 446.0e6, "FM"),
    ("Airband AM", 125.0e6, "AM"),
    ("ADS-B (1090)", 1090e6, "None (I/Q)"),
    ("ISM 433", 433.92e6, "FM"),
    ("ISM 915", 915e6, "FM"),
]

# Samples read per display frame; also the spectrum FFT length.
DISPLAY_BLOCK = 2048
# Equivalent noise bandwidth of the Hann window, in FFT bins.
_HANN_ENBW_BINS = 1.5
_DEFAULT_SAMPLE_RATE = 2.4e6
_AUDIBLE_MODES = ("AM", "FM", "USB", "LSB", "CW")
_NO_VALUE = "—"  # em dash for "not available"

# Optional ham radio panels
try:
    from ..ham.gui.callsign_panel import CallsignPanel
    from ..ham.gui.qrp_panel import QRPPanel
    from ..ham.gui.radio_tuner import RadioTunerWidget
    from ..ham.gui.signal_meter_widget import SignalMeterPanel
    from ..ham.gui.sstv_panel import SSTVPanel

    HAS_HAM_RADIO = True
except ImportError:
    HAS_HAM_RADIO = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_frequency(freq_hz: float) -> str:
    """Format a frequency in MHz with 3 to 6 decimals.

    ``100e6 -> "100.000 MHz"``, ``146.5125e6 -> "146.5125 MHz"``.
    """
    text = f"{max(0.0, float(freq_hz)) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def format_rate(rate_hz: float) -> str:
    """Format a sample rate, e.g. ``"2.4 MS/s"``."""
    return f"{float(rate_hz) / 1e6:.3f}".rstrip("0").rstrip(".") + " MS/s"


def format_bandwidth(bw_hz: float) -> str:
    """Format a bandwidth in kHz below 1 MHz and in MHz above."""
    bw_hz = float(bw_hz)
    if bw_hz >= 1e6:
        return f"{bw_hz / 1e6:.3f}".rstrip("0").rstrip(".") + " MHz"
    return f"{bw_hz / 1e3:.2f} kHz"


# ---------------------------------------------------------------------------
# Status bar message label
# ---------------------------------------------------------------------------


class _ElidedLabel(QLabel if HAS_PYQT6 else object):
    """A one-line label that ends in "…" instead of being cut off.

    ``text()`` returns the full text; only the painted text is shortened to
    the label's current width. It never asks for width of its own, so a long
    message can't raise the window's minimum width.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full_text = str(text or "")
        self._elide()

    def text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().changeEvent(event)
        # The stylesheet (theme switch) can change the font and padding.
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()

    def _elide(self) -> None:
        width = self.contentsRect().width()
        shown = self._full_text
        if width > 0 and self.fontMetrics().horizontalAdvance(shown) > width:
            shown = self.fontMetrics().elidedText(
                shown, Qt.TextElideMode.ElideRight, width
            )
        super().setText(shown)


# ---------------------------------------------------------------------------
# Info tab
# ---------------------------------------------------------------------------


class InfoPanel(QWidget if HAS_PYQT6 else object):
    """Read-only live summary of the receiver, shown in the Info tab."""

    # (group title, ((key, label, tooltip), ...))
    SECTIONS: Tuple[Tuple[str, Tuple[Tuple[str, str, str], ...]], ...] = (
        (
            "Receiver",
            (
                ("device", "Device", "The SDR samples are read from."),
                ("state", "State", "Whether samples are being acquired."),
                ("frequency", "Center frequency", get_short_tip("center_frequency")),
                ("sample_rate", "Sample rate", get_short_tip("sample_rate")),
                (
                    "span",
                    "Span",
                    "Width of the spectrum shown; equals the complex sample rate.",
                ),
                (
                    "rbw",
                    "RBW",
                    get_short_tip("resolution_bandwidth")
                    + f" ({DISPLAY_BLOCK}-point FFT, Hann window.)",
                ),
                ("demod", "Demodulation", "Selected in the Demodulation panel."),
                ("gain", "Gain", get_short_tip("gain")),
                ("squelch", "Squelch", "Audio is muted below this level."),
                (
                    "level",
                    "Peak level",
                    "Strongest bin of the current spectrum (0 dBFS = full scale).",
                ),
            ),
        ),
        (
            "Audio && Recording",
            (
                ("audio", "Audio output", "Toggle with Radio > Audio Output."),
                (
                    "recording",
                    "Recording",
                    "Toggle with the Record button or Ctrl+Shift+R.",
                ),
                (
                    "buffer",
                    "Buffer",
                    "I/Q samples held in memory. File > Save Recording writes "
                    "them to disk.",
                ),
            ),
        ),
        (
            "Application",
            (
                ("version", "Version", "SDR Module version."),
                ("qt", "Qt / PyQt", "Versions of the GUI toolkit."),
                ("theme", "Theme", "Change with View > Theme or Ctrl+T."),
            ),
        ),
    )

    TIPS = (
        "Click the spectrum or waterfall to tune there.",
        "Click the FREQ readout (or press Ctrl+L) to type a frequency.",
        "Left/Right tune by 10 kHz (Shift: 100 kHz, Ctrl: 1 MHz) "
        "while the plots have focus.",
        "Space starts or stops receiving while the plots have focus.",
        "Ctrl+B bookmarks the current frequency.",
        "F1 lists every keyboard shortcut.",
    )

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._values: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        for title, rows in self.SECTIONS:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            # Top-aligned, so a label lines up with the first line of a
            # wrapped value.
            form.setLabelAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(4)
            for key, label_text, tip in rows:
                label = QLabel(label_text)
                set_role(label, "muted")
                value = QLabel(_NO_VALUE)
                value.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
                )
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                # Long values (device names) wrap instead of forcing a
                # horizontal scrollbar in a narrow right column.
                value.setWordWrap(True)
                label.setToolTip(tip)
                value.setToolTip(tip)
                form.addRow(label, value)
                self._values[key] = value
            layout.addWidget(group)

        tips_group = QGroupBox("Quick Tips")
        tips_layout = QVBoxLayout(tips_group)
        tips = QLabel("\n".join(f"•  {tip}" for tip in self.TIPS))
        tips.setWordWrap(True)
        set_role(tips, "hint")
        tips_layout.addWidget(tips)
        layout.addWidget(tips_group)
        layout.addStretch(1)

    def set_value(self, key: str, text: str, tone: Optional[str] = None) -> None:
        """Show ``text`` for row ``key`` (optionally colored with a tone)."""
        label = self._values.get(key)
        if label is None:
            return
        if label.text() != text:
            label.setText(text)
        set_tone(label, tone)

    def value(self, key: str) -> str:
        """Current text of row ``key`` (empty if unknown)."""
        label = self._values.get(key)
        return label.text() if label is not None else ""


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class SDRMainWindow(QMainWindow if HAS_PYQT6 else object):
    """
    Main SDR application window.

    Contains:
    - Spectrum analyzer display
    - Waterfall display
    - Control panel (frequency, gain, bandwidth)
    - Protocol decoder output and other tool panels
    - Recording controls
    """

    # Frequency step for Left/Right, by modifier.
    _TUNE_STEPS_HZ = (
        {
            Qt.KeyboardModifier.NoModifier: 10e3,
            Qt.KeyboardModifier.ShiftModifier: 100e3,
            Qt.KeyboardModifier.ControlModifier: 1e6,
        }
        if HAS_PYQT6
        else {}
    )

    # Toolbar button captions.
    _START_TEXT = "▶  Start"
    _STOP_TEXT = "■  Stop"
    _RECORD_TEXT = "●  Record"

    def __init__(self, parent=None, demo_mode: bool = False):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required for the GUI")

        super().__init__(parent)

        self._device = None
        self._is_running = False
        self._recording = False
        self._samples_buffer: List[np.ndarray] = []
        self._demo_mode = False  # set by _start_demo_mode()
        self._radio_tuner = None  # Pop-out radio tuner window
        self._squelch_db = -80.0  # applied to spectrum gating
        # Cached analysis window for the spectrum display (built lazily to
        # match the sample-block length).
        self._spectrum_window: Optional[np.ndarray] = None
        self._spectrum_window_gain = 1.0
        self._agc_enabled = False
        self._recording_bytes = 0  # for free-space display
        self._recording_paused = False  # Pause in the Recording panel
        # Recording clock: seconds captured so far plus the start of the
        # current capturing stretch (None while paused or not receiving).
        self._rec_accum = 0.0
        self._rec_since: Optional[float] = None
        self._last_peak_db: Optional[float] = None
        self._known_device_names: Optional[set] = None  # set by _poll_hotplug
        self._layout_initialized = False
        self._splitters_restored = False
        self._panel_pages: Dict[str, QWidget] = {}
        self._panel_owners: Dict[str, QTabWidget] = {}
        self._panel_specs: List[Tuple[str, str, str]] = []

        # Live protocol decoder driven by the Decoder panel (None = Auto Detect
        # / no active decoder). Rebuilt when the panel's protocol changes.
        self._decoder: Any = None
        self._decoder_protocol: Any = None
        self._decoder_rate = 0.0  # sample rate the live decoder was built for

        # Persisted settings (frequency, gain, theme, bookmarks live here)
        self._settings = GuiSettings()
        # SDRApplication applies the persisted theme before the window exists;
        # mirror whatever is actually on screen.
        self._theme = current_theme()

        # Audio output
        self._audio = AudioSink()
        self._audio_enabled = self._settings.get_bool("audio_enabled", False)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_timers()

        # Connect signals
        self._connect_signals()
        theme_notifier().theme_changed.connect(self._on_theme_changed)

        # Restore persisted values
        self._restore_state()
        self._refresh_state_ui()

        # Auto-start demo mode
        if demo_mode:
            self._start_demo_mode()

        # First-run wizard
        if self._settings.is_first_run():
            self._run_first_run_wizard()

        # Start with the plots focused so Space and the arrow keys work.
        self._spectrum.setFocus()

        logger.info("Main window initialized")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("SDR Module")
        # Small enough for 1366x768 laptops; panels scroll instead of crushing.
        self.setMinimumSize(1024, 640)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(0)

        # Main horizontal splitter: displays | controls
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self._main_splitter)

        # Left side: spectrum over waterfall
        self._display_splitter = QSplitter(Qt.Orientation.Vertical)
        self._display_splitter.setChildrenCollapsible(False)
        self._spectrum = SpectrumWidget()
        self._waterfall = WaterfallWidget()
        for plot in (self._spectrum, self._waterfall):
            # Clicking a plot focuses it, so Left/Right tune afterwards
            # (see keyPressEvent).
            plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            self._display_splitter.addWidget(plot)
        self._display_splitter.setStretchFactor(0, 2)
        self._display_splitter.setStretchFactor(1, 3)
        self._main_splitter.addWidget(self._display_splitter)

        # Right side: control panel over the tabbed tool panels
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._control_panel = ControlPanel()
        self._right_splitter.addWidget(self._scrollable(self._control_panel))
        self._right_tabs = self._build_panel_tabs()
        self._right_splitter.addWidget(self._right_tabs)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 1)
        self._main_splitter.addWidget(self._right_splitter)

        # Extra window width goes to the displays, not the controls.
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)

    @staticmethod
    def _text_width(button: "QPushButton", *texts: str) -> int:
        """Width that fits any of ``texts`` in bold plus the button padding,
        so a toolbar button doesn't change size when its caption changes."""
        font = QFont(button.font())
        font.setBold(True)
        metrics = QFontMetrics(font)
        return max(metrics.horizontalAdvance(t) for t in texts) + 36

    @staticmethod
    def _scrollable(widget: "QWidget") -> "QWidget":
        """Return ``widget`` inside a frameless, resizable scroll area.

        A widget that already scrolls internally is returned unchanged so it
        is never wrapped twice.
        """
        if isinstance(widget, QScrollArea) or widget.findChild(QScrollArea):
            return widget
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        area.setWidget(widget)
        return area

    def _build_panel_tabs(self) -> "QTabWidget":
        """Create the right-hand tab widget holding the tool panels."""
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)

        self._decoder_panel = DecoderPanel()
        self._add_panel(
            tabs,
            "decoder",
            self._decoder_panel,
            "Decoder",
            "Live protocol decoder output (POCSAG, FLEX, ADS-B, ACARS, "
            "AX.25/APRS, RDS)",
        )

        self._bookmarks_panel = BookmarksPanel()
        self._add_panel(
            tabs,
            "bookmarks",
            self._bookmarks_panel,
            "Bookmarks",
            "Saved frequencies. Double-click one to tune; Ctrl+B adds the "
            "current frequency.",
        )

        # Optional ham radio panels, grouped under one tab so the top-level
        # tab bar fits a 360 px column without scroll arrows hiding tabs.
        if HAS_HAM_RADIO:
            self._ham_tabs = QTabWidget()
            self._ham_tabs.setDocumentMode(True)
            self._ham_tabs.setUsesScrollButtons(True)

            self._signal_meter_panel = SignalMeterPanel()
            self._add_panel(
                self._ham_tabs,
                "s_meter",
                self._signal_meter_panel,
                "S-Meter",
                "Signal strength in S-units with RST and signal report",
            )

            self._callsign_panel = CallsignPanel()
            self._add_panel(
                self._ham_tabs,
                "ham_id",
                self._callsign_panel,
                "Ham ID",
                "Station identification with your callsign (CW ID, HackRF only)",
            )

            self._sstv_panel = SSTVPanel()
            self._add_panel(
                self._ham_tabs,
                "sstv",
                self._sstv_panel,
                "SSTV",
                "Slow-scan TV image viewer, e.g. for ISS SSTV events",
            )

            self._qrp_panel = QRPPanel()
            self._add_panel(
                self._ham_tabs,
                "qrp",
                self._qrp_panel,
                "QRP",
                "Low-power (QRP) transmit power tools and compliance check",
            )

            index = tabs.addTab(self._ham_tabs, "Ham Radio")
            tabs.setTabToolTip(index, "S-meter, station ID, SSTV and QRP tools")

        self._info_panel = InfoPanel()
        self._add_panel(
            tabs,
            "info",
            self._info_panel,
            "Info",
            "Live receiver status, versions and quick tips",
        )
        return tabs

    def _add_panel(
        self, tabs: "QTabWidget", key: str, widget: "QWidget", title: str, tip: str
    ) -> None:
        """Add one tool panel as a (scrollable) tab of ``tabs``."""
        page = self._scrollable(widget)
        index = tabs.addTab(page, title)
        tabs.setTabToolTip(index, tip)
        self._panel_pages[key] = page
        self._panel_owners[key] = tabs
        self._panel_specs.append((key, title, tip))

    def _add_action(
        self,
        menu: Any,
        text: str,
        slot: Any = None,
        shortcut: Any = None,
        tip: str = "",
        checkable: bool = False,
        checked: bool = False,
    ) -> "QAction":
        """Create a menu action. Menu actions are the single owner of their
        keyboard shortcut, so no key is ever bound twice (which Qt treats as
        ambiguous and then fires neither)."""
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
            if slot is not None:
                action.toggled.connect(slot)
        elif slot is not None:
            action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # ---- File ----
        file_menu = menubar.addMenu("&File")
        self._add_action(
            file_menu,
            "&Open Recording...",
            self._open_recording,
            QKeySequence.StandardKey.Open,
            "Load a saved I/Q recording into the recording buffer",
        )
        self._add_action(
            file_menu,
            "&Save Recording...",
            self._save_recording,
            QKeySequence.StandardKey.Save,
            "Write the recorded I/Q samples to a file",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "&Import Channels (CHIRP CSV)...",
            self._import_channels_csv,
            tip="Load memory channels from a CHIRP-compatible CSV file",
        )
        self._add_action(
            file_menu,
            "&Export Channels (CHIRP CSV)...",
            self._export_channels_csv,
            tip="Save the bookmarks as a CHIRP-compatible CSV file",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Save S&creenshot...",
            self._save_screenshot,
            "Ctrl+P",
            "Save a PNG image of the window",
        )
        file_menu.addSeparator()
        # Some platforms (e.g. Windows) have no standard Quit key; use Ctrl+Q.
        quit_keys = QKeySequence.keyBindings(QKeySequence.StandardKey.Quit)
        self._add_action(
            file_menu,
            "E&xit",
            self.close,
            quit_keys[0] if quit_keys else "Ctrl+Q",
            "Close SDR Module",
        )

        # ---- Device ----
        device_menu = menubar.addMenu("&Device")
        self._connect_action = self._add_action(
            device_menu,
            "&Connect...",
            self._show_device_dialog,
            tip="Choose and open an RTL-SDR or HackRF One",
        )
        self._disconnect_action = self._add_action(
            device_menu,
            "&Disconnect",
            self._disconnect_device,
            tip="Stop receiving and close the current device",
        )
        device_menu.addSeparator()
        self._demo_action = self._add_action(
            device_menu,
            "Use De&mo Device",
            self._start_demo_mode,
            tip="Explore the app with a synthetic signal source (no hardware)",
        )
        self._add_action(
            device_menu,
            "&Scan for Devices",
            self._refresh_devices,
            tip="List the SDR devices currently plugged in",
        )

        # ---- Radio ----
        radio_menu = menubar.addMenu("&Radio")
        # Space is handled in keyPressEvent (so focused widgets keep it); the
        # "\tSpace" suffix only shows it in the menu's shortcut column.
        self._start_action = self._add_action(
            radio_menu,
            "&Start Receiving\tSpace",
            self._toggle_acquisition,
            tip="Start or stop acquiring samples",
        )
        self._record_action = QAction("&Record I/Q", self)
        self._record_action.setCheckable(True)
        self._record_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._record_action.setStatusTip("Capture raw I/Q samples into memory")
        self._record_action.triggered.connect(self._toggle_recording)
        radio_menu.addAction(self._record_action)
        radio_menu.addSeparator()
        audio_tip = (
            "Play demodulated AM/FM/SSB/CW audio when the signal is above squelch"
            if self._audio.available
            else "Audio output needs the PyQt6 QtMultimedia module"
        )
        self._audio_action = self._add_action(
            radio_menu,
            "&Audio Output",
            self._set_audio_enabled,
            tip=audio_tip,
            checkable=True,
            checked=self._audio_enabled and self._audio.available,
        )
        self._audio_action.setEnabled(self._audio.available)
        radio_menu.addSeparator()
        bands_menu = radio_menu.addMenu("Band &Presets")
        for label, freq_hz, mode in BAND_PRESETS:
            act = QAction(f"{label}\t{format_frequency(freq_hz)}", self)
            act.setStatusTip(f"Tune to {format_frequency(freq_hz)} in {mode} mode")
            act.triggered.connect(
                lambda _c=False, f=freq_hz, m=mode, n=label: self._apply_band_preset(
                    f, m, n
                )
            )
            bands_menu.addAction(act)
        self._add_action(
            radio_menu,
            "Enter &Frequency",
            self._focus_frequency_entry,
            "Ctrl+L",
            "Jump to the Frequency field to type a new center frequency",
        )
        self._add_action(
            radio_menu,
            "&Bookmark Current Frequency",
            self._bookmark_current_frequency,
            "Ctrl+B",
            "Add the current frequency to the Bookmarks panel",
        )

        # ---- View ----
        view_menu = menubar.addMenu("&View")
        self._spectrum_action = self._add_action(
            view_menu,
            "&Spectrum",
            self._spectrum.setVisible,
            tip="Show or hide the spectrum plot",
            checkable=True,
            checked=True,
        )
        self._waterfall_action = self._add_action(
            view_menu,
            "&Waterfall",
            self._waterfall.setVisible,
            tip="Show or hide the waterfall",
            checkable=True,
            checked=True,
        )
        for action in (self._spectrum_action, self._waterfall_action):
            action.toggled.connect(self._sync_plot_actions)
        view_menu.addSeparator()
        panels_menu = view_menu.addMenu("&Panels")
        for index, (key, title, tip) in enumerate(self._panel_specs):
            shortcut = f"Ctrl+{index + 1}" if index < 9 else None
            self._add_action(
                panels_menu,
                title,
                lambda k=key: self._show_panel(k),
                shortcut,
                tip,
            )
        theme_menu = view_menu.addMenu("&Theme")
        self._theme_actions: Dict[str, QAction] = {}
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for name, text in (("dark", "&Dark"), ("light", "&Light")):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setStatusTip(f"Use the {name} color theme")
            act.triggered.connect(lambda _c=False, n=name: self._set_theme(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act
        theme_menu.addSeparator()
        self._add_action(
            theme_menu,
            "&Toggle Theme",
            self._toggle_theme,
            "Ctrl+T",
            "Switch between the dark and light themes",
        )
        self._sync_theme_actions()
        view_menu.addSeparator()
        self._add_action(
            view_menu,
            "&Reset Layout",
            self._reset_layout,
            tip="Restore the default panel sizes and show both plots",
        )

        # ---- Tools ----
        tools_menu = menubar.addMenu("&Tools")
        self._add_action(
            tools_menu,
            "Frequency &Scanner...",
            self._show_scanner,
            "Ctrl+F",
            "Sweep a frequency range and list active channels",
        )
        self._add_action(
            tools_menu,
            "Protocol &Decoder",
            self._show_decoder_config,
            tip="Show the Decoder panel",
        )
        if HAS_HAM_RADIO:
            self._add_action(
                tools_menu,
                "AM/FM Radio &Tuner...",
                self._show_radio_tuner,
                "Ctrl+R",
                "Open the broadcast radio tuner window",
            )
        tools_menu.addSeparator()
        self._add_action(
            tools_menu,
            "&Error History...",
            self._show_error_history,
            "Ctrl+E",
            "Show recent warnings and errors",
        )

        # ---- Help ----
        help_menu = menubar.addMenu("&Help")
        self._add_action(
            help_menu,
            "&Keyboard Shortcuts...",
            self._show_help,
            "F1",
            "List every keyboard shortcut",
        )
        self._add_action(help_menu, "&About SDR Module...", self._show_about)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        # No "hide toolbar" context menu: there would be no way to get it back.
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(toolbar)
        self._toolbar = toolbar

        # Start/Stop: accent call to action while idle, a plain "Stop" while
        # running. Red is kept for Record, so "Stop" next to a red Record
        # button can't be mistaken for "stop recording".
        self._start_button = QPushButton(self._START_TEXT)
        self._start_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._start_button.setAccessibleName("Start or stop receiving")
        self._start_button.clicked.connect(self._toggle_acquisition)
        set_role(self._start_button, "primary")
        self._start_button.setMinimumWidth(
            self._text_width(self._start_button, self._START_TEXT, self._STOP_TEXT)
        )
        toolbar.addWidget(self._start_button)

        # Record: a toggle, filled red while recording.
        self._record_button = QPushButton(self._RECORD_TEXT)
        self._record_button.setCheckable(True)
        self._record_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._record_button.setAccessibleName("Record I/Q")
        self._record_button.setMinimumWidth(
            self._text_width(self._record_button, self._RECORD_TEXT)
        )
        self._record_button.clicked.connect(
            lambda _checked=False: self._record_action.trigger()
        )
        toolbar.addWidget(self._record_button)
        self._update_record_button_tip()

        toolbar.addSeparator()

        # Frequency readout
        freq_caption = QLabel("FREQ")
        set_role(freq_caption, "caption")
        toolbar.addWidget(freq_caption)
        self._freq_label = QLabel(format_frequency(100e6))
        set_role(self._freq_label, "lcd")
        self._freq_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._freq_label.setMinimumWidth(
            self._freq_label.fontMetrics().horizontalAdvance("0000.000 MHz") + 26
        )
        self._freq_label.setToolTip(
            "Center frequency. Click here (or press Ctrl+L) to type a new one. "
            "You can also click the spectrum or waterfall, or use Left/Right "
            "(10 kHz), Shift (100 kHz) and Ctrl (1 MHz) while the plots have "
            "focus."
        )
        self._freq_label.setAccessibleName("Center frequency")
        # Clicking the readout jumps to the Frequency field.
        self._freq_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._freq_label.installEventFilter(self)
        toolbar.addWidget(self._freq_label)

        toolbar.addSeparator()

        # Signal level readout
        level_caption = QLabel("LEVEL")
        set_role(level_caption, "caption")
        toolbar.addWidget(level_caption)
        self._level_label = QLabel(f"{_NO_VALUE} dBFS")
        set_role(self._level_label, "lcd-small")
        self._level_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._level_label.setMinimumWidth(
            self._level_label.fontMetrics().horizontalAdvance("-120.0 dBFS") + 20
        )
        self._level_label.setToolTip(
            "Peak level of the current spectrum (0 dBFS = full scale). "
            "Green while the signal is above the squelch threshold."
        )
        self._level_label.setAccessibleName("Signal level")
        toolbar.addWidget(self._level_label)

    def _setup_statusbar(self):
        """Setup status bar."""
        statusbar = QStatusBar()
        # Keep the state badge off the window edge. The window edges resize
        # the window; the size grip only drew a stray box in the corner.
        statusbar.setContentsMargins(8, 0, 8, 0)
        statusbar.setSizeGripEnabled(False)
        self.setStatusBar(statusbar)

        # Device state badge + device name + sample rate (left)
        self._state_badge = QLabel("NO DEVICE")
        set_role(self._state_badge, "badge", "warning")
        self._state_badge.setAccessibleName("Receiver state")
        statusbar.addWidget(self._state_badge)

        self._device_label = QLabel("No device")
        set_role(self._device_label, "value")
        self._device_label.setToolTip(
            "Current device. Change it with Device > Connect..."
        )
        statusbar.addWidget(self._device_label)

        self._rate_label = QLabel("")
        self._rate_label.setToolTip(get_short_tip("sample_rate"))
        statusbar.addWidget(self._rate_label)

        # Transient messages (neutral, success or error), auto-clearing. A
        # long message ends in "…" instead of raising the window's minimum
        # width; the full text is also in the tooltip.
        self._message_label = _ElidedLabel("")
        statusbar.addWidget(self._message_label, 1)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_status_message)

        # Recording indicator (right, only while recording)
        self._recording_label = QLabel("REC")
        set_role(self._recording_label, "badge", "danger")
        self._recording_label.setToolTip("Recording raw I/Q samples")
        self._recording_label.setVisible(False)
        self._recording_label.setAccessibleName("Recording time")
        statusbar.addPermanentWidget(self._recording_label)
        self._recording_info_label = QLabel("")
        self._recording_info_label.setToolTip(
            "Recorded size and free disk space in the working directory"
        )
        self._recording_info_label.setVisible(False)
        statusbar.addPermanentWidget(self._recording_info_label)

    def _setup_timers(self):
        """Setup update timers."""
        # Display update timer (30 Hz)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._update_display)
        self._display_timer.start(33)

        # Status update timer (5 Hz)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(200)

        # Device hot-plug polling (0.5 Hz)
        self._hotplug_timer = QTimer(self)
        self._hotplug_timer.timeout.connect(self._poll_hotplug)
        self._hotplug_timer.start(2000)

    def _connect_signals(self):
        """Connect control panel signals."""
        self._control_panel.frequency_changed.connect(self._on_frequency_changed)
        self._control_panel.gain_changed.connect(self._on_gain_changed)
        self._control_panel.bandwidth_changed.connect(self._on_bandwidth_changed)
        self._control_panel.squelch_changed.connect(self._on_squelch_changed)
        self._control_panel.agc_changed.connect(self._on_agc_changed)
        self._control_panel.demod_changed.connect(self._on_demod_changed)
        # The control panel's Record button drives the same recording as the
        # toolbar action (its signals were previously connected to nothing).
        self._control_panel.recording_started.connect(self._on_panel_record_started)
        self._control_panel.recording_stopped.connect(self._on_panel_record_stopped)
        self._control_panel.recording_paused.connect(self._on_panel_record_paused)
        # The decoder panel's protocol selector drives a live decoder over the
        # acquired samples (the panel was previously fed no data at all).
        self._decoder_panel.protocol_changed.connect(self._on_decoder_protocol_changed)
        self._bookmarks_panel.tune_requested.connect(self._on_bookmark_tune)
        if HAS_HAM_RADIO:
            self._callsign_panel.id_requested.connect(self._on_callsign_id_requested)
        # Click-to-tune from spectrum and waterfall. set_frequency keeps the
        # control panel, toolbar readout and plot axes in step.
        self._spectrum.frequency_clicked.connect(self.set_frequency)
        self._waterfall.frequency_clicked.connect(self.set_frequency)
        self._right_tabs.currentChanged.connect(self._on_panel_tab_changed)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if (
            obj is getattr(self, "_freq_label", None)
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._focus_frequency_entry()
            return True
        return super().eventFilter(obj, event)

    def _focus_frequency_entry(self) -> None:
        """Put the cursor in the Frequency field, ready to type."""
        entry = self._control_panel._freq_input
        field = getattr(entry, "_freq_input", entry)
        area = self._control_panel.findChild(QScrollArea)
        if area is None and isinstance(self._right_splitter.widget(0), QScrollArea):
            area = self._right_splitter.widget(0)
        if area is not None:
            area.ensureWidgetVisible(field)
        field.setFocus(Qt.FocusReason.ShortcutFocusReason)
        select_all = getattr(field, "selectAll", None)
        if callable(select_all):
            select_all()

    def keyPressEvent(self, event):
        """Space starts/stops receiving; Left/Right tune.

        These are handled here instead of as window-wide shortcuts, so they
        only act when the focused widget does not use the key itself: a
        focused slider, combo box, list, tab bar, button or text field keeps
        its normal arrow/Space behavior, and the key only reaches the window
        when nothing else consumed it (e.g. with the spectrum focused).
        """
        key = event.key()
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if key == Qt.Key.Key_Space and mods == Qt.KeyboardModifier.NoModifier:
            if not event.isAutoRepeat():
                self._toggle_acquisition()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = self._TUNE_STEPS_HZ.get(mods)
            if step is not None:
                self._nudge_frequency(step if key == Qt.Key.Key_Right else -step)
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Status bar messages
    # ------------------------------------------------------------------

    def _show_status_message(
        self, message: str, tone: Optional[str] = None, duration_ms: int = 4000
    ) -> None:
        """Show a transient message in the status bar.

        Args:
            message: Text to display
            tone: ``None`` (neutral), ``"success"``, ``"info"``, ``"warning"``
                or ``"danger"``
            duration_ms: How long to show before auto-clearing
        """
        self._message_label.setText(message)
        self._message_label.setToolTip(message)
        set_tone(self._message_label, tone)
        self._message_timer.start(max(500, int(duration_ms)))

    def _show_status_error(self, message: str, duration_ms: int = 6000) -> None:
        """Show a transient error message (red) in the status bar."""
        self._show_status_message(message, "danger", duration_ms)
        logger.warning(f"Status bar error: {message}")

    def _clear_status_message(self) -> None:
        self._message_label.setText("")
        self._message_label.setToolTip("")
        set_tone(self._message_label, None)

    # ------------------------------------------------------------------
    # State presentation
    # ------------------------------------------------------------------

    def _current_frequency(self) -> float:
        return float(self._control_panel._freq_input.get_frequency())

    def _device_sample_rate(self) -> float:
        """Sample rate of the current device (default 2.4 MS/s)."""
        dev = self._device
        if dev is None:
            return _DEFAULT_SAMPLE_RATE
        state = getattr(dev, "state", None)
        for value in (
            getattr(state, "sample_rate", None),
            getattr(dev, "sample_rate", None),
        ):
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if rate > 0:
                return rate
        return _DEFAULT_SAMPLE_RATE

    def _device_display_name(self) -> str:
        dev = self._device
        if dev is None:
            return "No device"
        if self._demo_mode:
            return "Demo device (synthetic signals)"
        name = getattr(getattr(dev, "info", None), "name", "") or ""
        return str(name) or type(dev).__name__

    def _refresh_state_ui(self) -> None:
        """Sync toolbar, status bar, menus and title with the device state."""
        has_device = self._device is not None
        running = self._is_running and has_device

        # Toolbar Start/Stop
        button = self._start_button
        button.setText(self._STOP_TEXT if running else self._START_TEXT)
        role = "" if running else "primary"
        if (button.property("role") or "") != role:
            set_role(button, role or None)
        if running:
            button.setToolTip("Stop receiving (Space)")
        elif has_device:
            button.setToolTip("Start receiving (Space)")
        else:
            button.setToolTip(
                "Start receiving (Space). No device is connected yet: you can "
                "connect one or use the demo device."
            )
        self._start_action.setText(
            "&Stop Receiving\tSpace" if running else "&Start Receiving\tSpace"
        )

        # Status bar
        if not has_device:
            self._state_badge.setText("NO DEVICE")
            set_tone(self._state_badge, "warning")
            self._state_badge.setToolTip(
                "Use Device > Connect... or Device > Use Demo Device"
            )
        elif running:
            self._state_badge.setText("RUNNING")
            set_tone(self._state_badge, "success")
            self._state_badge.setToolTip("Receiving samples")
        else:
            self._state_badge.setText("STOPPED")
            set_tone(self._state_badge, "muted")
            self._state_badge.setToolTip("Press Start or Space to receive")
        if has_device:
            self._device_label.setText(self._device_display_name())
        else:
            self._device_label.setText(
                "Press Start to connect a device or run the demo"
            )
        label_role = "value" if has_device else "muted"
        if self._device_label.property("role") != label_role:
            set_role(self._device_label, label_role)
        rate = self._device_sample_rate()
        self._rate_label.setText(format_rate(rate) if has_device else "")

        # Menus
        self._disconnect_action.setEnabled(has_device)
        self._demo_action.setEnabled(not has_device)

        # Level readout is only meaningful while samples flow
        if not running:
            self._last_peak_db = None
            self._level_label.setText(f"{_NO_VALUE} dBFS")
            set_tone(self._level_label, None)

        # Plot axes / click-to-tune span
        self._spectrum.set_frequency_range(self._current_frequency(), rate)
        self._waterfall.set_sample_rate(rate)

        # A live decoder is built for one sample rate; rebuild it when the
        # device (and so the rate) changes.
        if self._decoder is not None and self._decoder_rate != rate:
            self._rebuild_decoder()

        if self._recording:
            self._update_rec_clock()
            self._update_recording_status()

        self._update_window_title()
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _update_window_title(self) -> None:
        if self._device is None:
            self.setWindowTitle("SDR Module")
        elif self._demo_mode:
            self.setWindowTitle("Demo Mode — SDR Module")
        else:
            self.setWindowTitle(f"{self._device_display_name()} — SDR Module")

    def _refresh_info_panel(self) -> None:
        """Fill the Info tab from the current state."""
        p = self._info_panel
        has_device = self._device is not None
        running = self._is_running and has_device
        rate = self._device_sample_rate()

        p.set_value(
            "device", self._device_display_name(), None if has_device else "warning"
        )
        if not has_device:
            p.set_value("state", "Not connected", "warning")
        elif running:
            p.set_value("state", "Receiving", "success")
        else:
            p.set_value("state", "Stopped", "muted")
        p.set_value("frequency", format_frequency(self._current_frequency()))
        p.set_value("sample_rate", format_rate(rate) if has_device else _NO_VALUE)
        p.set_value("span", format_bandwidth(rate) if has_device else _NO_VALUE)
        p.set_value(
            "rbw",
            (
                format_bandwidth(_HANN_ENBW_BINS * rate / DISPLAY_BLOCK)
                if has_device
                else _NO_VALUE
            ),
        )
        p.set_value("demod", self._control_panel._demod_combo.currentText())
        if self._agc_enabled:
            p.set_value("gain", "Automatic (AGC)")
        else:
            p.set_value("gain", f"{self._control_panel._gain_slider.value()} dB")
        p.set_value("squelch", f"{self._squelch_db:.0f} dBFS")
        if running and self._last_peak_db is not None:
            p.set_value("level", f"{self._last_peak_db:.1f} dBFS")
        else:
            p.set_value("level", _NO_VALUE)

        if not self._audio.available:
            p.set_value("audio", "Unavailable", "muted")
        else:
            p.set_value("audio", "On" if self._audio_enabled else "Off")
        if not self._recording:
            p.set_value("recording", "Idle")
        elif self._recording_paused:
            p.set_value(
                "recording", f"Paused at {self._recording_elapsed_text()}", "warning"
            )
        elif not running:
            p.set_value("recording", "Armed, waiting for the receiver", "warning")
        else:
            p.set_value("recording", self._recording_elapsed_text(), "danger")
        count = sum(len(block) for block in self._samples_buffer)
        if count:
            p.set_value("buffer", f"{count:,} samples ({count * 8 / 1e6:.1f} MB)")
        else:
            p.set_value("buffer", "Empty")

        p.set_value("version", __version__)
        p.set_value("qt", f"{QT_VERSION_STR} / {PYQT_VERSION_STR}")
        p.set_value("theme", self._theme.title())

    def _on_panel_tab_changed(self, _index: int) -> None:
        if self._right_tabs.currentWidget() is self._panel_pages.get("info"):
            self._refresh_info_panel()

    def _show_panel(self, key: str) -> None:
        """Bring one of the right-hand tool panels to the front."""
        page = self._panel_pages.get(key)
        if page is None:
            return
        owner = self._panel_owners.get(key, self._right_tabs)
        if owner is not self._right_tabs:
            # A nested panel (e.g. the Ham Radio group): show its group first.
            self._right_tabs.setCurrentWidget(owner)
        owner.setCurrentWidget(page)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._layout_initialized:
            self._layout_initialized = True
            if not self._splitters_restored:
                self._apply_default_splitter_sizes()

    def _apply_default_splitter_sizes(self) -> None:
        """Default proportions: controls ~30% (380-440 px) wide, the control
        panel over ~55% of the right column, spectrum over 40% of the plots."""
        width = self._main_splitter.width()
        if width < 400:
            width = self.width() - 12
        right = int(min(440, max(380, width * 0.3)))
        self._main_splitter.setSizes([max(200, width - right), right])

        height = self._right_splitter.height()
        if height < 300:
            height = self.height() - 120
        top = int(height * 0.55)
        self._right_splitter.setSizes([top, max(120, height - top)])

        height = self._display_splitter.height()
        if height < 300:
            height = self.height() - 120
        spectrum = int(height * 0.4)
        self._display_splitter.setSizes([spectrum, max(120, height - spectrum)])

    def _sync_plot_actions(self, _checked: bool = True) -> None:
        """Keep at least one plot visible: the last visible plot's menu item
        is disabled so the display area can't be left empty."""
        actions = (self._spectrum_action, self._waterfall_action)
        shown = [a for a in actions if a.isChecked()]
        for action in actions:
            locked = len(shown) == 1 and action.isChecked()
            action.setEnabled(not locked)

    def _reset_layout(self) -> None:
        """Show both plots and restore the default splitter sizes."""
        for action in (self._spectrum_action, self._waterfall_action):
            action.setChecked(True)
        self._spectrum.setVisible(True)
        self._waterfall.setVisible(True)
        self._apply_default_splitter_sizes()
        self._show_status_message("Layout reset")

    def _apply_default_geometry(self) -> None:
        """Size the window to 1400x900, or to fit a smaller screen."""
        width, height = 1400, 900
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(width, int(avail.width() * 0.95))
            height = min(height, int(avail.height() * 0.92))
        self.resize(max(width, self.minimumWidth()), max(height, self.minimumHeight()))

    # ------------------------------------------------------------------
    # Recording sync between toolbar, menu and control panel
    # ------------------------------------------------------------------

    def _on_panel_record_started(self, fmt: str) -> None:
        """Start recording from the control panel's Record button."""
        self._start_recording()

    def _on_panel_record_stopped(self) -> None:
        """Stop recording from the control panel's Record button."""
        self._stop_recording()

    def _on_panel_record_paused(self, paused: bool) -> None:
        """Pause or resume capturing from the control panel's Pause button."""
        paused = bool(paused) and self._recording
        if paused == self._recording_paused:
            return
        self._recording_paused = paused
        self._update_rec_clock()
        self._update_recording_status()
        self._show_status_message(
            "Recording paused" if paused else "Recording resumed",
            "warning" if paused else None,
            2500,
        )

    def _sync_record_ui(self, recording: bool) -> None:
        for control in (self._record_action, self._record_button):
            control.blockSignals(True)
            control.setChecked(recording)
            control.blockSignals(False)
        role = "danger" if recording else ""
        if (self._record_button.property("role") or "") != role:
            set_role(self._record_button, role or None)
        self._update_record_button_tip()
        self._recording_label.setVisible(recording)
        self._recording_info_label.setVisible(recording)
        if recording:
            self._update_recording_status()

    def _update_record_button_tip(self) -> None:
        if self._recording:
            tip = "Stop recording (Ctrl+Shift+R)"
        else:
            tip = (
                "Record raw I/Q samples to memory (Ctrl+Shift+R). "
                "Save them with File > Save Recording."
            )
        self._record_button.setToolTip(tip)

    # Panel labels -> ProtocolType for the live decoder.
    _DECODER_PROTOCOLS = {
        "POCSAG": "pocsag",
        "FLEX": "flex",
        "AX.25/APRS": "ax25",
        "ADS-B": "adsb",
        "ACARS": "acars",
        "RDS": "rds",
    }

    def _on_decoder_protocol_changed(self, text: str) -> None:
        """Build (or clear) the live decoder for the panel's selected protocol."""
        from ..dsp.protocols import ProtocolType, create_protocol_decoder

        proto_value = self._DECODER_PROTOCOLS.get(text)
        if proto_value is None:
            # "Off" (or unknown): no live decoder.
            self._decoder = None
            self._decoder_protocol = None
            return

        rate = self._device_sample_rate()
        try:
            protocol = ProtocolType(proto_value)
            self._decoder = create_protocol_decoder(protocol, sample_rate=float(rate))
            self._decoder_protocol = protocol
            self._decoder_rate = rate
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not create %s decoder: %s", text, exc)
            self._decoder = None
            self._decoder_protocol = None

    def _rebuild_decoder(self) -> None:
        """Recreate the live decoder for the current device's sample rate."""
        self._on_decoder_protocol_changed(
            self._decoder_panel._proto_combo.currentText()
        )

    def _run_decoder(self, samples: np.ndarray) -> None:
        """Feed demodulated samples to the active decoder and show any messages."""
        if self._decoder is None or self._decoder_protocol is None:
            return
        if not self._decoder_panel._enabled_check.isChecked():
            return

        from ..dsp.protocols import demodulate_for_protocol

        baseband = demodulate_for_protocol(samples, self._decoder_protocol)
        if baseband is None:
            return
        try:
            messages = self._decoder.decode(baseband)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Decoder error: %s", exc)
            return
        for msg in messages:
            self._push_decoded_message(msg)

    def _push_decoded_message(self, msg: Any) -> None:
        """Render one decoded message into the decoder panel."""
        panel = self._decoder_panel
        name = type(msg).__name__
        try:
            if name == "POCSAGMessage":
                panel.add_pocsag_message(msg.address, msg.content, msg.function)
            elif name == "ADSBMessage":
                panel.add_adsb_message(
                    msg.icao_address,
                    msg.callsign,
                    msg.altitude,
                    msg.latitude,
                    msg.longitude,
                    msg.velocity,
                )
            elif name in ("AX25Frame", "APRSMessage"):
                panel.add_aprs_message(
                    getattr(msg, "source", ""),
                    getattr(msg, "destination", ""),
                    getattr(msg, "latitude", 0.0),
                    getattr(msg, "longitude", 0.0),
                    getattr(msg, "info", getattr(msg, "comment", "")),
                )
            else:
                address = str(getattr(msg, "address", getattr(msg, "icao_address", "")))
                content = str(getattr(msg, "content", getattr(msg, "info", "")))
                proto = getattr(getattr(msg, "protocol", None), "value", name)
                panel.add_message(
                    str(proto), address, content, getattr(msg, "valid", True)
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not render decoded message: %s", exc)

    # ------------------------------------------------------------------
    # Control panel handlers
    # ------------------------------------------------------------------

    def _on_frequency_changed(self, freq_hz: float):
        """Handle frequency change."""
        if self._device:
            try:
                self._device.set_frequency(freq_hz)
            except Exception as e:
                self._show_status_error(f"Could not tune the device: {e}")

        # Update the readout and the plots' axes / click-to-tune mapping
        self._freq_label.setText(format_frequency(freq_hz))
        self._spectrum.set_center_freq(freq_hz)
        self._waterfall.set_center_freq(freq_hz)

        logger.debug(f"Frequency changed to {freq_hz/1e6:.3f} MHz")

    def _on_gain_changed(self, gain_db: float):
        """Handle gain change."""
        if self._device:
            self._device.set_gain(gain_db)
        logger.debug(f"Gain changed to {gain_db:.1f} dB")

    def _on_bandwidth_changed(self, bw_hz: float):
        """Handle bandwidth change."""
        if self._device:
            self._device.set_bandwidth(bw_hz)
        logger.debug(f"Bandwidth changed to {bw_hz/1e3:.1f} kHz")

    def _on_squelch_changed(self, db: float):
        """Handle squelch threshold change."""
        self._squelch_db = float(db)
        self._settings.set("squelch_db", self._squelch_db)

    def _on_agc_changed(self, enabled: bool):
        """Handle AGC toggle."""
        self._agc_enabled = bool(enabled)
        self._settings.set("agc_enabled", self._agc_enabled)
        if self._device and hasattr(self._device, "set_gain_mode"):
            try:
                # set_gain_mode(auto: bool). Passing the strings "auto"/"manual"
                # made "manual" truthy, so unticking AGC left AGC on.
                self._device.set_gain_mode(self._agc_enabled)
            except Exception as e:
                logger.debug(f"Device AGC set failed: {e}")

    def _on_demod_changed(self, mode: str):
        """Handle demodulator selection change."""
        self._settings.set("demod_mode", mode)
        self._start_or_stop_audio(mode)

    def _start_or_stop_audio(self, mode: str):
        """Start audio for audible demods, stop otherwise."""
        if not self._audio_enabled:
            self._audio.stop()
            return
        if mode in _AUDIBLE_MODES:
            self._audio.start(48000)
        else:
            self._audio.stop()

    def _on_bookmark_tune(self, freq_hz: float, label: str):
        """Tune to a bookmarked frequency."""
        self.set_frequency(freq_hz)
        self._show_status_message(f"Tuned to {label}", "info", 2500)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _toggle_acquisition(self):
        """Toggle signal acquisition."""
        if self._is_running:
            self._stop_acquisition()
        else:
            self._start_acquisition()

    def _prompt_no_device(self) -> str:
        """Ask how to proceed when Start is pressed with no device.

        Returns ``"connect"``, ``"demo"`` or ``"cancel"``.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("No Device Connected")
        box.setText("No SDR device is connected.")
        box.setInformativeText(
            "Connect opens an RTL-SDR or HackRF One. Demo Mode explores the "
            "app with synthetic signals, no hardware needed."
        )
        connect_btn = box.addButton("&Connect...", QMessageBox.ButtonRole.AcceptRole)
        demo_btn = box.addButton("&Demo Mode", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(connect_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is connect_btn:
            return "connect"
        if clicked is demo_btn:
            return "demo"
        return "cancel"

    def _start_acquisition(self):
        """Start signal acquisition."""
        if self._is_running:
            return
        if not self._device:
            choice = self._prompt_no_device()
            if choice == "demo":
                self._start_demo_mode()
                return
            if choice != "connect" or not self._show_device_dialog():
                return

        try:
            started = self._device.start_rx()
        except Exception as e:
            logger.error(f"Could not start receiving: {e}")
            self._show_status_error(f"Could not start receiving: {e}")
            return
        if started is False:
            self._show_status_error("The device did not start streaming")
            return

        self._is_running = True
        self._refresh_state_ui()
        logger.info("Acquisition started")

    def _stop_acquisition(self):
        """Stop signal acquisition."""
        self._is_running = False

        # Update callsign panel
        if HAS_HAM_RADIO:
            self._callsign_panel.set_transmitting(False)

        if self._device:
            try:
                self._device.stop_rx()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"stop_rx failed: {e}")

        self._refresh_state_ui()
        logger.info("Acquisition stopped")

    def _on_callsign_id_requested(self):
        """Handle callsign ID request from the callsign panel."""
        callsign = self._callsign_panel.get_callsign()
        if not callsign:
            QMessageBox.warning(
                self, "No Callsign", "Please enter your callsign in the Ham ID panel."
            )
            self._show_panel("ham_id")
            return

        logger.info(f"Callsign ID requested: {callsign}")

        try:
            from ..ham.callsign import generate_tx_id

            settings = self._callsign_panel.get_settings()

            # Only CW ID is implemented for transmission. Refuse the other
            # modes rather than silently transmitting Morse under their label.
            mode = settings.get("mode", "CW")
            if mode != "CW":
                QMessageBox.information(
                    self,
                    "Mode Not Available",
                    f"{mode} identification is not implemented for transmission "
                    "yet; only CW (Morse) ID can be sent. Select CW (Morse) in "
                    "the Ham ID panel.",
                )
                return

            self._show_status_message(f"Preparing CW ID: DE {callsign}", "info")

            # Generate FM-modulated I/Q samples ready for transmission
            iq_samples = generate_tx_id(
                callsign,
                wpm=settings.get("cw_wpm", 20),
                tone_frequency=settings.get("cw_tone", 700),
                rf_sample_rate=2e6,
                fm_deviation=2500.0,  # Narrowband FM for CW
            )
            logger.info(f"Generated TX ID: {len(iq_samples)} I/Q samples")

            # Attempt transmission
            self._transmit_audio(iq_samples, callsign)

        except Exception as e:
            logger.error(f"Error generating callsign ID: {e}")
            QMessageBox.warning(
                self, "ID Error", f"Failed to generate callsign ID: {e}"
            )

    def _transmit_audio(self, iq_samples: np.ndarray, description: str = "audio"):
        """
        Transmit I/Q samples via HackRF.

        Args:
            iq_samples: Complex I/Q samples to transmit
            description: Description for logging/status
        """
        from ..core.frequency_manager import is_tx_allowed
        from ..devices.hackrf import HackRFDevice

        # Check if we have a TX-capable device
        if self._device is None:
            QMessageBox.warning(
                self,
                "No Device",
                "No SDR device connected. Connect a HackRF for transmission.",
            )
            return

        # Verify it's a HackRF (TX-capable)
        if not isinstance(self._device, HackRFDevice):
            QMessageBox.warning(
                self,
                "TX Not Supported",
                "Connected device does not support transmission.\n"
                "HackRF One is required for TX operations.",
            )
            return

        # Get current frequency for TX validation
        current_freq = self._current_frequency()
        bandwidth = 10e3  # Approximate CW bandwidth

        # Validate TX is allowed at this frequency
        allowed, reason = is_tx_allowed(current_freq, bandwidth)
        if not allowed:
            QMessageBox.critical(
                self,
                "TX Blocked",
                f"Transmission blocked at {current_freq/1e6:.3f} MHz:\n{reason}",
            )
            return

        # Stop RX if running (HackRF is half-duplex)
        was_running = self._is_running
        if was_running:
            self._stop_acquisition()

        try:
            # Update status
            self._show_status_message(f"Transmitting: {description}", "warning")
            if HAS_HAM_RADIO:
                self._callsign_panel.set_transmitting(True)

            # Configure TX gain
            self._device.set_tx_gain(20)  # Moderate TX power

            # Transmit the samples
            logger.info(
                f"Starting TX: {len(iq_samples)} samples at {current_freq/1e6:.3f} MHz"
            )

            # Use write_samples for one-shot transmission
            success = self._device.write_samples(iq_samples)

            if success:
                logger.info(f"TX complete: {description}")
                self._show_status_message(
                    f"Transmission complete: {description}", "success"
                )
            else:
                logger.error("TX failed")
                QMessageBox.warning(
                    self, "TX Failed", "Failed to transmit. Check device connection."
                )

        except Exception as e:
            logger.error(f"TX error: {e}")
            QMessageBox.warning(self, "TX Error", f"Transmission error: {e}")
        finally:
            if HAS_HAM_RADIO:
                self._callsign_panel.set_transmitting(False)

            # Restart RX if it was running
            if was_running:
                self._start_acquisition()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _toggle_recording(self, checked: bool):
        """Toggle recording from the toolbar button / menu action."""
        if checked:
            self._start_recording()
        else:
            self._stop_recording()
        # Keep the control panel's Record button in sync (without re-emitting).
        self._control_panel.set_recording_state(checked)

    def _start_recording(self):
        """Start recording."""
        self._recording = True
        self._recording_paused = False
        self._samples_buffer = []
        self._recording_bytes = 0
        self._rec_accum = 0.0
        self._rec_since = None
        self._update_rec_clock()
        self._sync_record_ui(True)
        if not self._is_running:
            self._show_status_message(
                "Recording armed: samples are captured once receiving starts "
                "(Space)",
                "warning",
                6000,
            )
        logger.info("Recording started")

    def _stop_recording(self):
        """Stop recording."""
        was_recording = self._recording
        self._recording = False
        self._recording_paused = False
        self._update_rec_clock()
        self._sync_record_ui(False)
        count = sum(len(block) for block in self._samples_buffer)
        if was_recording and count:
            self._show_status_message(
                f"Recorded {count:,} samples. Use File > Save Recording (Ctrl+S) "
                "to write them to disk.",
                "success",
                8000,
            )
        elif was_recording:
            self._show_status_message(
                "Recording stopped. Nothing was captured because the receiver "
                "was not running.",
                "warning",
                6000,
            )
        logger.info("Recording stopped")

    def _recording_capturing(self) -> bool:
        """True while samples are actually being added to the recording."""
        return (
            self._recording
            and not self._recording_paused
            and self._is_running
            and self._device is not None
        )

    def _update_rec_clock(self) -> None:
        """Bank the running stretch of the recording clock and restart it if
        capturing continues. Call whenever recording, pause or receiver state
        changes, so the clock only counts time that samples were captured."""
        now = time.monotonic()
        if self._rec_since is not None:
            self._rec_accum += max(0.0, now - self._rec_since)
            self._rec_since = None
        if self._recording_capturing():
            self._rec_since = now

    def _recording_elapsed(self) -> float:
        """Seconds spent capturing in the current recording."""
        elapsed = self._rec_accum
        if self._rec_since is not None:
            elapsed += max(0.0, time.monotonic() - self._rec_since)
        return elapsed

    def _recording_elapsed_text(self) -> str:
        elapsed = int(self._recording_elapsed())
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_recording_status(self) -> None:
        """Refresh the REC badge, size/free-space label and panel timer."""
        elapsed = self._recording_elapsed_text()
        badge = self._recording_label
        if self._recording_paused:
            badge.setText(f"PAUSED {elapsed}")
            set_tone(badge, "warning")
            badge.setToolTip("Recording paused. Resume it in the Recording panel.")
        elif not self._recording_capturing():
            badge.setText("REC ARMED")
            set_tone(badge, "warning")
            badge.setToolTip(
                "Recording is armed: samples are captured once receiving "
                "starts (Space)"
            )
        else:
            badge.setText(f"REC {elapsed}")
            set_tone(badge, "danger")
            badge.setToolTip("Recording raw I/Q samples")
        self._control_panel.update_record_time(int(self._recording_elapsed()))

        mb = self._recording_bytes / (1024 * 1024)
        # Recorded size + free space on the working-directory volume
        try:
            free_gb = shutil.disk_usage(".").free / (1024**3)
            info = f"{mb:.1f} MB · {free_gb:.1f} GB free"
        except OSError:
            info = f"{mb:.1f} MB"
        self._recording_info_label.setText(info)

    # ------------------------------------------------------------------
    # Periodic updates
    # ------------------------------------------------------------------

    def _read_block(self) -> Optional[np.ndarray]:
        """The next block of samples, without stalling the window.

        Hardware devices queue whole USB blocks from a background thread and
        ``read_samples`` waits up to a second for one by default. The display
        timer runs faster than blocks arrive, so waiting froze the GUI for
        most of every second; poll without waiting instead.
        """
        dev = self._device
        if isinstance(dev, SDRDevice):
            return dev.read_samples(DISPLAY_BLOCK, timeout=0.0)
        return dev.read_samples(DISPLAY_BLOCK)

    def _stream_died(self) -> bool:
        """Stop and tell the user if a hardware stream ended on its own
        (USB unplugged, read error); otherwise the window would sit on
        "RUNNING" with frozen plots."""
        dev = self._device
        if not isinstance(dev, SDRDevice):
            return False
        try:
            if dev.state.is_streaming:
                return False
        except Exception:  # pragma: no cover - defensive
            return False
        error = getattr(dev, "rx_error", None)
        self._stop_acquisition()
        detail = f": {error}" if error else ""
        self._show_status_error(
            f"The device stopped streaming{detail}. Check the USB connection, "
            "then press Start.",
            10000,
        )
        return True

    def _update_display(self):
        """Update spectrum and waterfall displays."""
        if not self._is_running or not self._device:
            return

        samples = self._read_block()
        if samples is None or len(samples) == 0:
            self._stream_died()
            return

        # The plots show the newest DISPLAY_BLOCK samples (a fixed FFT size, so
        # the RBW shown in the Info tab holds); everything else gets the block.
        display = samples[-DISPLAY_BLOCK:] if len(samples) > DISPLAY_BLOCK else samples

        # Compute spectrum (windowed, referenced to dBFS)
        power = self._power_spectrum_dbfs(display)

        # Update spectrum widget
        self._spectrum.update_spectrum(power)

        # Update waterfall
        self._waterfall.add_line(power)

        # Update level display (green while above the squelch threshold)
        peak_level = float(np.max(power))
        self._last_peak_db = peak_level
        self._level_label.setText(f"{peak_level:.1f} dBFS")
        set_tone(
            self._level_label, "success" if peak_level >= self._squelch_db else None
        )

        # Squelch + audio: demodulate when carrier is above threshold
        if self._audio_enabled and peak_level >= self._squelch_db:
            self._demodulate_and_play(samples)

        # Record if active (and not paused). The REC badge and the Recording
        # panel's timer are refreshed by the status timer.
        if self._recording and not self._recording_paused:
            self._samples_buffer.append(samples)
            # complex64 = 8 bytes/sample
            self._recording_bytes += len(samples) * 8

        # The S-meter panel measures the same I/Q blocks.
        if HAS_HAM_RADIO:
            try:
                self._signal_meter_panel.update_samples(samples)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("S-meter update failed: %s", exc)

        # Feed the live protocol decoder (if the Decoder panel selected one).
        self._run_decoder(samples)

    def _power_spectrum_dbfs(self, samples: np.ndarray) -> np.ndarray:
        """Windowed power spectrum in dBFS (full-scale sinusoid -> 0 dB).

        A raw ``20*log10(|FFT|)`` of an N-point block scales with N (an N=2048
        FFT of a full-scale tone peaks near +66 dB), so every bin saturated the
        top of the (-120, 0) dB display and pinned the -80 dB squelch open.

        A Hann window suppresses spectral leakage, and dividing the magnitude by
        the window's coherent gain (the sum of its samples) references the
        result to dBFS: a full-scale complex sinusoid peaks at 0 dB and real
        captures land in the display/squelch range as intended.
        """
        n = len(samples)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if self._spectrum_window is None or self._spectrum_window.shape[0] != n:
            # Periodic Hann (the form used for spectral analysis).
            self._spectrum_window = np.hanning(n + 1)[:-1].astype(np.float64)
            self._spectrum_window_gain = float(np.sum(self._spectrum_window))
        windowed = samples * self._spectrum_window
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        magnitude = np.abs(spectrum) / max(self._spectrum_window_gain, 1e-12)
        return (20.0 * np.log10(magnitude + 1e-12)).astype(np.float32)

    def _demodulate_and_play(self, samples: np.ndarray) -> None:
        """Demodulate samples to audio and push to the output sink."""
        mode = self._control_panel._demod_combo.currentText()
        if mode not in _AUDIBLE_MODES:
            return
        rate = self._device_sample_rate()
        try:
            # Simple demodulators sufficient for monitoring in the GUI
            if mode == "FM":
                # FM discriminator scaled by the selected deviation so a signal
                # at +/- that deviation reaches full-scale audio (this is what
                # the FM Dev control selects).
                prod = samples[1:] * np.conj(samples[:-1])
                inst_freq = np.angle(prod).astype(np.float32)
                deviation = self._control_panel.get_fm_deviation()
                gain = float(rate) / (2.0 * np.pi * max(deviation, 1.0))
                audio = np.clip(inst_freq * gain, -1.0, 1.0).astype(np.float32)
            else:
                if mode == "AM":
                    audio = (np.abs(samples) - np.mean(np.abs(samples))).astype(
                        np.float32
                    )
                else:  # SSB/CW: pass real part as crude envelope
                    audio = samples.real.astype(np.float32)
                # Peak-normalize the non-FM modes.
                peak = float(np.max(np.abs(audio)) + 1e-9)
                audio = audio / max(peak, 1e-6) * 0.5
            # Decimate to ~48 kHz assuming 2.4 MS/s (48x)
            decim = max(1, int(rate / 48000))
            self._audio.write(audio[::decim])
        except Exception as e:  # pragma: no cover
            logger.debug(f"Audio demod failed: {e}")

    def _update_status(self):
        """Update status bar (and the Info tab while it is visible)."""
        if self._recording:
            self._update_recording_status()
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def _show_device_dialog(self) -> bool:
        """Show the device connection dialog. Returns True if one was opened."""
        from .device_dialog import DeviceDialog

        dialog = DeviceDialog(self)
        if not dialog.exec():
            return False
        device = dialog.get_selected_device()
        if device is None:
            return False

        if self._device is not None and device is not self._device:
            self._release_device()
        self._device = device
        self._demo_mode = False
        # Bring the new device to what the controls show.
        self._on_frequency_changed(self._current_frequency())
        try:
            self._on_gain_changed(float(self._control_panel._gain_slider.value()))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not apply gain to new device: {e}")
        name = self._device_display_name()
        logger.info(f"Connected to {name}")
        self._refresh_state_ui()
        self._show_status_message(
            f"Connected to {name}. Press Start or Space to receive.", "success"
        )
        return True

    def _release_device(self) -> None:
        """Stop and close the current device (if any)."""
        if self._is_running:
            self._stop_acquisition()
        if self._device is not None:
            try:
                self._device.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Device close failed: {e}")
        self._device = None
        self._demo_mode = False

    def _disconnect_device(self):
        """Disconnect from device."""
        if self._device is None:
            return
        name = self._device_display_name()
        self._release_device()
        self._refresh_state_ui()
        self._show_status_message(f"Disconnected from {name}")
        logger.info("Device disconnected")

    def _refresh_devices(self):
        """Refresh device list by rescanning hardware."""
        from ..core.device_manager import DeviceManager

        logger.info("Refreshing device list...")
        try:
            manager = DeviceManager()
            devices = manager.scan_devices()
        except Exception as e:
            logger.error(f"Device scan failed: {e}")
            self._show_status_error(f"Device scan failed: {e}")
            return

        if not devices:
            QMessageBox.information(
                self,
                "No Devices Found",
                "No SDR devices were detected.\n\n"
                "Plug in an RTL-SDR or HackRF One and scan again, or use "
                "Device > Use Demo Device to try the app without hardware.",
            )
        else:
            names = "\n".join(f"  • {d}" for d in devices)
            QMessageBox.information(
                self,
                "Devices Found",
                f"Detected {len(devices)} device(s):\n\n{names}\n\n"
                "Use Device > Connect... to open one.",
            )

    def _poll_hotplug(self) -> None:
        """Poll for device hot-plug changes and notify on new devices."""
        try:
            from ..core.device_manager import DeviceManager

            devices = DeviceManager().scan_devices()
        except Exception:
            return
        names = {str(d) for d in devices}
        if self._known_device_names is None:
            # First poll after init: baseline silently. (An empty set is a
            # valid baseline, so a device plugged in after starting the app
            # with none attached is still announced.)
            self._known_device_names = names
            return
        added = names - self._known_device_names
        self._known_device_names = names
        if added:
            for name in added:
                logger.info(f"Device connected: {name}")
            self._show_status_message(
                f"New device detected: {next(iter(added))}. "
                "Use Device > Connect... to open it.",
                "info",
                8000,
            )

    def _start_demo_mode(self):
        """Start demo mode with synthetic signals."""
        from .device_dialog import MockDevice

        if self._device is not None and not self._demo_mode:
            self._release_device()
        if self._device is None:
            self._device = MockDevice()
        self._demo_mode = True
        self._on_frequency_changed(self._current_frequency())
        self._start_acquisition()
        self._show_status_message(
            "Demo mode: showing synthetic signals, no hardware needed", "info"
        )
        logger.info("Demo mode started")

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _open_recording(self):
        """Open a recording file and load its samples."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Recording",
            "",
            ";;".join(
                (
                    "I/Q Recordings (*.cf32 *.cs16 *.cs8 *.cu8 *.cf64 *.raw *.iq "
                    "*.bin *.sigmf-data *.sigmf-meta *.wav)",
                    "SigMF (*.sigmf-data *.sigmf-meta)",
                    "WAV Files (*.wav)",
                    "All Files (*)",
                )
            ),
        )
        if not filename:
            return

        from ..dsp.recording import load_iq_file

        try:
            samples, metadata = load_iq_file(filename)
        except Exception as e:
            logger.error(f"Failed to open recording: {e}")
            QMessageBox.warning(self, "Open Failed", f"Could not open recording:\n{e}")
            return

        if self._recording:
            # Don't mix live samples into the loaded file.
            self._toggle_recording(False)
        self._samples_buffer = [samples]
        if metadata.center_frequency > 0:
            self.set_frequency(metadata.center_frequency)
        logger.info(
            f"Loaded {len(samples)} samples from {filename} "
            f"(rate={metadata.sample_rate}, freq={metadata.center_frequency})"
        )
        unknown = "not stored in the file"
        rate = (
            format_rate(metadata.sample_rate) if metadata.sample_rate > 0 else unknown
        )
        freq = (
            format_frequency(metadata.center_frequency)
            if metadata.center_frequency > 0
            else unknown
        )
        self._show_status_message(
            f"Loaded {len(samples):,} samples into the recording buffer", "success"
        )
        QMessageBox.information(
            self,
            "Recording Loaded",
            f"Loaded {len(samples):,} samples into the recording buffer.\n\n"
            f"Sample rate: {rate}\n"
            f"Center frequency: {freq}\n\n"
            "Use File > Save Recording to write it in another format.",
        )

    # (file dialog filter, extension, FileFormat name, SampleFormat name)
    _SAVE_FORMATS: Tuple[Tuple[str, str, str, str], ...] = (
        ("Complex Float32 (*.cf32)", ".cf32", "RAW", "FLOAT32"),
        ("Complex Int16 (*.cs16)", ".cs16", "RAW", "INT16"),
        ("SigMF (*.sigmf-data)", ".sigmf-data", "SIGMF", "FLOAT32"),
        ("WAV, 16-bit I/Q (*.wav)", ".wav", "WAV", "INT16"),
        ("Raw I/Q, Float32 (*.raw)", ".raw", "RAW", "FLOAT32"),
    )

    def _panel_recording_extension(self) -> str:
        """Extension matching the Recording panel's Format choice."""
        try:
            fmt = self._control_panel.get_recording_format().lower()
        except Exception:  # pragma: no cover - defensive
            fmt = ""
        if fmt.startswith("wav"):
            return ".wav"
        if "sigmf" in fmt:
            return ".sigmf-data"
        return ".cf32"

    @classmethod
    def _resolve_save_format(
        cls, filename: str, selected_filter: str = ""
    ) -> Tuple[str, Tuple[str, str, str, str]]:
        """Pick the save format for ``filename``.

        A known extension wins; otherwise the selected filter's format is used
        and its extension appended (Qt's static dialog doesn't add one).
        """
        lower = filename.lower()
        if lower.endswith(".sigmf-meta"):
            # SigMF is saved as the data file; its .sigmf-meta is written next to it.
            filename = filename[: -len(".sigmf-meta")] + ".sigmf-data"
            lower = filename.lower()
        for spec in cls._SAVE_FORMATS:
            if lower.endswith(spec[1]):
                return filename, spec
        spec = next(
            (f for f in cls._SAVE_FORMATS if f[0] == selected_filter),
            cls._SAVE_FORMATS[0],
        )
        return filename + spec[1], spec

    def _save_recording(self):
        """Save the current recording buffer to an I/Q file."""
        if not self._samples_buffer:
            QMessageBox.information(
                self,
                "Nothing to Save",
                "No samples have been recorded yet.\n\n"
                "Start receiving, press Record (Ctrl+Shift+R), stop recording, "
                "then save.",
            )
            return

        # Preselect the format chosen in the Recording panel.
        ext = self._panel_recording_extension()
        initial = next(f[0] for f in self._SAVE_FORMATS if f[1] == ext)
        suggested = (
            f"iq_{self._current_frequency() / 1e6:.3f}MHz_"
            f"{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        )
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            suggested,
            ";;".join(f[0] for f in self._SAVE_FORMATS),
            initial,
        )
        if not filename:
            return

        from ..dsp.recording import FileFormat, SampleFormat, save_iq_file

        filename, spec = self._resolve_save_format(filename, selected_filter)
        fmt, sample_fmt = FileFormat[spec[2]], SampleFormat[spec[3]]

        samples = np.concatenate(self._samples_buffer).astype(np.complex64)
        sample_rate = self._device_sample_rate()
        center_freq = self._current_frequency()

        try:
            save_iq_file(
                filename,
                samples,
                sample_rate=sample_rate,
                center_frequency=center_freq,
                sample_format=sample_fmt,
                file_format=fmt,
            )
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            QMessageBox.warning(self, "Save Failed", f"Could not save recording:\n{e}")
            return

        logger.info(f"Saved {len(samples)} samples to {filename}")
        self._show_status_message(
            f"Saved {len(samples):,} samples to {filename}", "success", 6000
        )

    def _import_channels_csv(self) -> None:
        """Import memory channels from a CHIRP CSV into the bookmarks panel."""
        count = self._bookmarks_panel.import_csv()
        if count:
            self._show_panel("bookmarks")
            self._show_status_message(f"Imported {count} channel(s)", "success")

    def _export_channels_csv(self) -> None:
        """Export the saved channels to a CHIRP-compatible CSV file."""
        count = self._bookmarks_panel.export_csv()
        if count:
            self._show_status_message(f"Exported {count} channel(s)", "success")

    def _save_screenshot(self) -> None:
        """Save a PNG of the window (spectrum + waterfall + panels)."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "sdr.png", "PNG (*.png)"
        )
        if not filename:
            return
        pixmap = self.grab()
        if pixmap.save(filename):
            self._show_status_message(f"Screenshot saved to {filename}", "success")
        else:
            QMessageBox.warning(self, "Save Failed", "Could not save screenshot.")

    # ------------------------------------------------------------------
    # Tools and dialogs
    # ------------------------------------------------------------------

    def _show_scanner(self):
        """Show the frequency scanner dialog."""
        from .scanner_dialog import ScannerDialog

        dialog = ScannerDialog(self, device=self._device)
        dialog.exec()

    def _show_radio_tuner(self):
        """Show the AM/FM radio tuner pop-out window."""
        if self._radio_tuner is None:
            self._radio_tuner = RadioTunerWidget(self, self._device_sample_rate())
            # Connect frequency change to main tuner
            self._radio_tuner.frequency_changed.connect(
                self._on_radio_frequency_changed
            )

        self._radio_tuner.show()
        self._radio_tuner.raise_()
        self._radio_tuner.activateWindow()

    def _on_radio_frequency_changed(self, freq_hz: float, band: str):
        """Handle frequency change from radio tuner."""
        # set_frequency tunes the device (if any) and updates the control
        # panel, toolbar readout and plot axes together.
        self.set_frequency(freq_hz)
        self._show_status_message(
            f"Tuned to {format_frequency(freq_hz)} ({band})", "info", 2500
        )
        logger.info(f"Tuned to {freq_hz/1e6:.3f} MHz ({band})")

    def _show_decoder_config(self):
        """Bring the Decoder panel to the front."""
        self._show_panel("decoder")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SDR Module",
            f"<h3>SDR Module {__version__}</h3>"
            "<p>Software-defined radio for signal visualization, frequency "
            "analysis and protocol decoding.</p>"
            "<p><b>Hardware:</b> RTL-SDR and HackRF One, or the built-in "
            "demo device.</p>"
            "<p><b>Features:</b> spectrum and waterfall displays; AM, FM, SSB "
            "and CW demodulation; POCSAG, FLEX, ADS-B, ACARS, AX.25/APRS and "
            "RDS decoders; I/Q recording; ham radio tools.</p>"
            f"<p>Qt {QT_VERSION_STR} &middot; PyQt {PYQT_VERSION_STR}</p>",
        )

    def _show_help(self) -> None:
        """Open the keyboard shortcuts reference."""
        from .help_dialog import HelpDialog

        HelpDialog(self).exec()

    def _show_error_history(self) -> None:
        """Open the error history viewer."""
        from .error_log_dialog import ErrorLogDialog, install_history_handler

        install_history_handler()
        ErrorLogDialog(self).exec()

    # ------------------------------------------------------------------
    # Tuning helpers
    # ------------------------------------------------------------------

    def _nudge_frequency(self, offset_hz: float) -> None:
        """Adjust the center frequency by the given offset."""
        new_freq = max(0.0, self._current_frequency() + offset_hz)
        self.set_frequency(new_freq)

    def _apply_band_preset(self, freq_hz: float, mode: str, label: str) -> None:
        """Tune to a band preset and set matching demod mode."""
        self.set_frequency(freq_hz)
        idx = self._control_panel._demod_combo.findText(mode)
        if idx >= 0:
            self._control_panel._demod_combo.setCurrentIndex(idx)
        self._show_status_message(
            f"{label}: {format_frequency(freq_hz)}, {mode}", "info", 2500
        )

    def _bookmark_current_frequency(self) -> None:
        """Save the current tuner frequency into bookmarks."""
        freq = self._current_frequency()
        label = format_frequency(freq)
        self._bookmarks_panel.add_bookmark(label, freq)
        self._show_status_message(f"Bookmarked {label}", "success", 2500)

    # ------------------------------------------------------------------
    # Theme and audio
    # ------------------------------------------------------------------

    def _set_theme(self, name: str) -> None:
        """Apply and persist a theme ("dark" or "light")."""
        name = normalize_theme(name)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, name)  # emits theme_changed -> _on_theme_changed
        else:  # pragma: no cover - no QApplication
            self._on_theme_changed(name)
        self._settings.set("theme", name)

    def _toggle_theme(self) -> None:
        """Switch between the dark and light themes."""
        self._set_theme("light" if current_theme() == "dark" else "dark")
        self._show_status_message(f"{self._theme.title()} theme", None, 1500)

    def _on_theme_changed(self, name: str) -> None:
        try:
            self._theme = normalize_theme(name)
            self._sync_theme_actions()
            if self._info_panel.isVisible():
                self._refresh_info_panel()
        except RuntimeError:  # pragma: no cover - window already destroyed
            pass

    def _sync_theme_actions(self) -> None:
        action = self._theme_actions.get(self._theme)
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _set_audio_enabled(self, enabled: bool) -> None:
        """Toggle audio output on/off and persist the preference."""
        self._audio_enabled = bool(enabled)
        self._settings.set("audio_enabled", self._audio_enabled)
        # Apply immediately to current demod mode
        mode = self._control_panel._demod_combo.currentText()
        self._start_or_stop_audio(mode)
        if self._audio_enabled and mode not in _AUDIBLE_MODES:
            self._show_status_message(
                "Audio on. Choose AM, FM, USB, LSB or CW in Demodulation to hear "
                "signals.",
                "info",
                6000,
            )
        else:
            self._show_status_message(
                "Audio on" if self._audio_enabled else "Audio off", None, 2000
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore_state(self) -> None:
        """Restore persisted user settings on startup."""
        try:
            freq = self._settings.get_float("frequency_hz", 100e6)
            gain = self._settings.get_float("gain_db", 20.0)
            squelch = self._settings.get_float("squelch_db", -80.0)
            agc = self._settings.get_bool("agc_enabled", False)
            demod = self._settings.get_str("demod_mode", "None (I/Q)")
        except Exception as e:
            logger.debug(f"Settings restore failed: {e}")
            freq, gain, squelch, agc, demod = 100e6, 20.0, -80.0, False, "None (I/Q)"
        # set_frequency also updates the toolbar readout and plot axes.
        self.set_frequency(freq)
        self._control_panel.set_gain(gain)
        self._control_panel.set_squelch_db(squelch)
        self._control_panel.set_agc_enabled(agc)
        idx = self._control_panel._demod_combo.findText(demod)
        if idx >= 0:
            self._control_panel._demod_combo.setCurrentIndex(idx)

        # Window geometry
        geom = self._settings.load_geometry("main")
        restored = False
        if geom:
            try:
                restored = bool(self.restoreGeometry(geom))
            except Exception as e:
                logger.debug(f"Could not restore window geometry: {e}")
        if not restored:
            self._apply_default_geometry()

        # Splitter sizes (ignored if a saved pane is squeezed to nothing)
        restored_all = True
        for name, splitter in self._splitters().items():
            state = self._settings.load_geometry(name)
            ok = False
            if state:
                try:
                    ok = bool(splitter.restoreState(state))
                except Exception as e:
                    logger.debug(f"Could not restore {name}: {e}")
            if ok and min(splitter.sizes() or [0]) < 40:
                ok = False
            restored_all = restored_all and ok
        self._splitters_restored = restored_all

    def _splitters(self) -> Dict[str, "QSplitter"]:
        return {
            "main_splitter": self._main_splitter,
            "right_splitter": self._right_splitter,
            "display_splitter": self._display_splitter,
        }

    def _persist_state(self) -> None:
        """Save settings on exit."""
        try:
            freq = self._current_frequency()
            gain = float(self._control_panel._gain_slider.value())
            self._settings.set("frequency_hz", freq)
            self._settings.set("gain_db", gain)
            self._settings.set("squelch_db", self._squelch_db)
            self._settings.set("agc_enabled", self._agc_enabled)
            self._settings.set("theme", self._theme)
            self._settings.save_geometry("main", self.saveGeometry())
            if self._spectrum.isVisible() and self._waterfall.isVisible():
                for name, splitter in self._splitters().items():
                    self._settings.save_geometry(name, splitter.saveState())
            self._settings.sync()
        except Exception as e:
            logger.debug(f"Settings save failed: {e}")

    def _run_first_run_wizard(self) -> None:
        """Show the welcome wizard and tune to the chosen starting band."""
        from ..core.device_manager import DeviceManager
        from .first_run_wizard import FirstRunWizard

        try:
            hw_found = len(DeviceManager().scan_devices()) > 0
        except Exception:
            hw_found = False
        wiz = FirstRunWizard(self, hardware_found=hw_found)
        if wiz.exec():
            self.set_frequency(wiz.selected_frequency())
        self._settings.mark_first_run_done()

    def closeEvent(self, event):
        """Handle window close."""
        self._persist_state()
        for timer in (self._display_timer, self._status_timer, self._hotplug_timer):
            timer.stop()
        self._audio.stop()
        # Stops receiving and closes the device; a failing close() is logged
        # instead of leaving the window impossible to close.
        self._release_device()

        logger.info("Application closing")
        event.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frequency(self, freq_hz: float):
        """Set the center frequency (clamped to the tuning range)."""
        self._control_panel.set_frequency(freq_hz)
        # Use the value the control panel accepted, so the device, toolbar
        # readout and plot axes never disagree with the Frequency field.
        self._on_frequency_changed(self._current_frequency())

    def set_gain(self, gain_db: float):
        """Set the RF gain."""
        self._control_panel.set_gain(gain_db)
        self._on_gain_changed(gain_db)
