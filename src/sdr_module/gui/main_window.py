"""
Main application window.

Provides the primary window with all panels and controls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
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
from .audio_sink import AudioSink
from .bookmarks_panel import BookmarksPanel
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .settings_store import GuiSettings
from .spectrum_widget import SpectrumWidget
from .themes import get_stylesheet
from .waterfall_widget import WaterfallWidget

# Band presets shown in the Tools → Bands menu
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


class SDRWorker(QThread if HAS_PYQT6 else object):
    """Background worker for SDR data acquisition."""

    if HAS_PYQT6:
        samples_ready = pyqtSignal(np.ndarray)
        error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        if HAS_PYQT6:
            super().__init__(parent)
        self._running = False
        self._device = None
        self._sample_rate = 2.4e6
        self._center_freq = 100e6

    def set_device(self, device):
        """Set the SDR device."""
        self._device = device

    def run(self):
        """Main acquisition loop."""
        self._running = True

        while self._running and self._device:
            try:
                # Read samples from device
                samples = self._device.read_samples(16384)
                if samples is not None:
                    self.samples_ready.emit(samples)
            except Exception as e:
                self.error_occurred.emit(str(e))
                break

    def stop(self):
        """Stop acquisition."""
        self._running = False


class SDRMainWindow(QMainWindow if HAS_PYQT6 else object):
    """
    Main SDR application window.

    Contains:
    - Spectrum analyzer display
    - Waterfall display
    - Control panel (frequency, gain, bandwidth)
    - Protocol decoder output
    - Recording controls
    """

    def __init__(self, parent=None, demo_mode: bool = False):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required for the GUI")

        super().__init__(parent)

        self._device = None
        self._is_running = False
        self._recording = False
        self._samples_buffer = []
        self._demo_mode = demo_mode
        self._radio_tuner = None  # Pop-out radio tuner window
        self._squelch_db = -80.0  # applied to spectrum gating
        # Cached analysis window for the spectrum display (built lazily to
        # match the sample-block length).
        self._spectrum_window: Optional[np.ndarray] = None
        self._spectrum_window_gain = 1.0
        self._agc_enabled = False
        self._recording_bytes = 0  # for free-space display
        self._theme = "dark"
        self._known_device_names: set = set()

        # Live protocol decoder driven by the Decoder panel (None = Auto Detect
        # / no active decoder). Rebuilt when the panel's protocol changes.
        self._decoder: Any = None
        self._decoder_protocol: Any = None

        # Persisted settings (frequency, gain, theme, bookmarks live here)
        self._settings = GuiSettings()
        self._theme = self._settings.get_str("theme", "dark")

        # Audio output
        self._audio = AudioSink()
        self._audio_enabled = self._settings.get_bool("audio_enabled", False)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_timers()
        self._setup_shortcuts()

        # Connect signals
        self._connect_signals()

        # Restore persisted values
        self._restore_state()

        # Auto-start demo mode
        if demo_mode:
            self._start_demo_mode()

        # First-run wizard
        if self._settings.is_first_run():
            self._run_first_run_wizard()

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(f"SDR Module v{__version__}")
        self.setMinimumSize(1200, 800)

        # Central widget with splitters
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left side: Spectrum and waterfall
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(2)

        # Vertical splitter for spectrum/waterfall
        display_splitter = QSplitter(Qt.Orientation.Vertical)

        # Spectrum widget
        self._spectrum = SpectrumWidget()
        display_splitter.addWidget(self._spectrum)

        # Waterfall widget
        self._waterfall = WaterfallWidget()
        display_splitter.addWidget(self._waterfall)

        # Set initial sizes (40% spectrum, 60% waterfall)
        display_splitter.setSizes([300, 450])

        display_layout.addWidget(display_splitter)
        main_splitter.addWidget(display_widget)

        # Right side: Control panel and decoder
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Control panel
        self._control_panel = ControlPanel()
        right_layout.addWidget(self._control_panel)

        # Decoder panel in tabs
        decoder_tabs = QTabWidget()

        self._decoder_panel = DecoderPanel()
        decoder_tabs.addTab(self._decoder_panel, "Decoder")

        self._bookmarks_panel = BookmarksPanel()
        self._bookmarks_panel.tune_requested.connect(self._on_bookmark_tune)
        decoder_tabs.addTab(self._bookmarks_panel, "Bookmarks")

        # Optional ham radio panels
        if HAS_HAM_RADIO:
            self._callsign_panel = CallsignPanel()
            self._callsign_panel.id_requested.connect(self._on_callsign_id_requested)
            decoder_tabs.addTab(self._callsign_panel, "HAM ID")

            self._sstv_panel = SSTVPanel()
            decoder_tabs.addTab(self._sstv_panel, "SSTV/ISS")

            self._signal_meter_panel = SignalMeterPanel()
            decoder_tabs.addTab(self._signal_meter_panel, "S-Meter")

            self._qrp_panel = QRPPanel()
            decoder_tabs.addTab(self._qrp_panel, "QRP")

        # Info tab
        info_widget = QWidget()
        decoder_tabs.addTab(info_widget, "Info")

        right_layout.addWidget(decoder_tabs, 1)

        main_splitter.addWidget(right_widget)

        # Set splitter sizes (70% display, 30% controls)
        main_splitter.setSizes([800, 350])

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Recording...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_recording)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Recording...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_recording)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        import_channels_action = QAction("&Import Channels (CHIRP CSV)...", self)
        import_channels_action.setStatusTip(
            "Load memory channels from a CHIRP-compatible CSV file"
        )
        import_channels_action.triggered.connect(self._import_channels_csv)
        file_menu.addAction(import_channels_action)

        export_channels_action = QAction("&Export Channels (CHIRP CSV)...", self)
        export_channels_action.setStatusTip(
            "Save the saved channels as a CHIRP-compatible CSV file"
        )
        export_channels_action.triggered.connect(self._export_channels_csv)
        file_menu.addAction(export_channels_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Device menu
        device_menu = menubar.addMenu("&Device")

        connect_action = QAction("&Connect...", self)
        connect_action.triggered.connect(self._show_device_dialog)
        device_menu.addAction(connect_action)

        disconnect_action = QAction("&Disconnect", self)
        disconnect_action.triggered.connect(self._disconnect_device)
        device_menu.addAction(disconnect_action)

        device_menu.addSeparator()

        refresh_action = QAction("&Refresh Devices", self)
        refresh_action.triggered.connect(self._refresh_devices)
        device_menu.addAction(refresh_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        spectrum_action = QAction("&Spectrum", self)
        spectrum_action.setCheckable(True)
        spectrum_action.setChecked(True)
        spectrum_action.triggered.connect(lambda c: self._spectrum.setVisible(c))
        view_menu.addAction(spectrum_action)

        waterfall_action = QAction("&Waterfall", self)
        waterfall_action.setCheckable(True)
        waterfall_action.setChecked(True)
        waterfall_action.triggered.connect(lambda c: self._waterfall.setVisible(c))
        view_menu.addAction(waterfall_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        scanner_action = QAction("Frequency &Scanner...", self)
        scanner_action.triggered.connect(self._show_scanner)
        tools_menu.addAction(scanner_action)

        decoder_action = QAction("Protocol &Decoder...", self)
        decoder_action.triggered.connect(self._show_decoder_config)
        tools_menu.addAction(decoder_action)

        # Band presets submenu
        bands_menu = tools_menu.addMenu("&Bands")
        for label, freq_hz, mode in BAND_PRESETS:
            act = QAction(label, self)
            act.triggered.connect(
                lambda _c, f=freq_hz, m=mode, n=label: self._apply_band_preset(f, m, n)
            )
            bands_menu.addAction(act)

        tools_menu.addSeparator()

        bookmark_action = QAction("&Bookmark Current Frequency", self)
        bookmark_action.setShortcut("Ctrl+B")
        bookmark_action.triggered.connect(self._bookmark_current_frequency)
        tools_menu.addAction(bookmark_action)

        screenshot_action = QAction("Save &Screenshot...", self)
        screenshot_action.setShortcut("Ctrl+P")
        screenshot_action.triggered.connect(self._save_screenshot)
        tools_menu.addAction(screenshot_action)

        errors_action = QAction("&Error History...", self)
        errors_action.setShortcut("Ctrl+E")
        errors_action.triggered.connect(self._show_error_history)
        tools_menu.addAction(errors_action)

        tools_menu.addSeparator()

        theme_action = QAction("Toggle Light/Dark &Theme", self)
        theme_action.setShortcut("Ctrl+T")
        theme_action.triggered.connect(self._toggle_theme)
        tools_menu.addAction(theme_action)

        audio_action = QAction("&Audio Output", self)
        audio_action.setCheckable(True)
        audio_action.setChecked(self._audio_enabled)
        audio_action.toggled.connect(self._set_audio_enabled)
        tools_menu.addAction(audio_action)

        if HAS_HAM_RADIO:
            tools_menu.addSeparator()

            radio_action = QAction("AM/FM &Radio Tuner...", self)
            radio_action.setShortcut("Ctrl+R")
            radio_action.triggered.connect(self._show_radio_tuner)
            tools_menu.addAction(radio_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        help_action = QAction("Keyboard &Shortcuts...", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start/Stop button
        self._start_action = QAction("Start", self)
        self._start_action.triggered.connect(self._toggle_acquisition)
        toolbar.addAction(self._start_action)

        toolbar.addSeparator()

        # Record button
        self._record_action = QAction("Record", self)
        self._record_action.setCheckable(True)
        self._record_action.triggered.connect(self._toggle_recording)
        toolbar.addAction(self._record_action)

        toolbar.addSeparator()

        # Frequency display — styled as LCD readout
        freq_label = QLabel("FREQ")
        freq_label.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold; padding-right: 2px;"
        )
        toolbar.addWidget(freq_label)
        self._freq_label = QLabel("100.000 MHz")
        self._freq_label.setStyleSheet(
            "font-family: 'Consolas', 'SF Mono', 'Fira Code', monospace;"
            "font-size: 18px; font-weight: bold; color: #a6e3a1;"
            "background-color: #11111b; border: 1px solid #313244;"
            "border-radius: 4px; padding: 2px 10px; letter-spacing: 1px;"
        )
        toolbar.addWidget(self._freq_label)

        toolbar.addSeparator()

        # Signal level — styled as meter readout
        level_label = QLabel("LEVEL")
        level_label.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold; padding-right: 2px;"
        )
        toolbar.addWidget(level_label)
        self._level_label = QLabel("-80.0 dB")
        self._level_label.setStyleSheet(
            "font-family: 'Consolas', 'SF Mono', 'Fira Code', monospace;"
            "font-size: 14px; color: #f9e2af;"
            "background-color: #11111b; border: 1px solid #313244;"
            "border-radius: 4px; padding: 2px 8px;"
        )
        toolbar.addWidget(self._level_label)

    def _setup_statusbar(self):
        """Setup status bar."""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # Device status
        self._device_label = QLabel("No device")
        self._device_label.setStyleSheet(
            "color: #a6adc8; font-weight: bold; padding: 0 8px;"
        )
        statusbar.addWidget(self._device_label)

        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #313244; padding: 0 2px;")
        statusbar.addWidget(sep1)

        # Sample rate
        self._rate_label = QLabel("2.4 MS/s")
        self._rate_label.setStyleSheet(
            "font-family: monospace; color: #bac2de; padding: 0 4px;"
        )
        statusbar.addWidget(self._rate_label)

        # Buffer indicator
        self._buffer_progress = QProgressBar()
        self._buffer_progress.setFixedWidth(80)
        self._buffer_progress.setFixedHeight(14)
        self._buffer_progress.setRange(0, 100)
        self._buffer_progress.setValue(0)
        self._buffer_progress.setFormat("%p%")
        statusbar.addPermanentWidget(self._buffer_progress)

        # Recording status
        self._recording_label = QLabel("")
        self._recording_label.setStyleSheet("font-weight: bold; padding: 0 6px;")
        statusbar.addPermanentWidget(self._recording_label)

        # Error display (auto-clearing)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet(
            "color: #f38ba8; font-weight: bold; padding: 0 6px;"
        )
        statusbar.addPermanentWidget(self._error_label)
        self._error_clear_timer = QTimer(self)
        self._error_clear_timer.setSingleShot(True)
        self._error_clear_timer.timeout.connect(lambda: self._error_label.setText(""))

    def _show_status_error(self, message: str, duration_ms: int = 5000) -> None:
        """Show a transient error message in the status bar.

        Args:
            message: Error text to display
            duration_ms: How long to show before auto-clearing
        """
        self._error_label.setText(message)
        self._error_clear_timer.start(duration_ms)
        logger.warning(f"Status bar error: {message}")

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

    def _setup_shortcuts(self):
        """Install keyboard shortcuts."""
        # Space = start/stop
        sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc.activated.connect(self._toggle_acquisition)

        # Ctrl+Shift+R = record toggle
        sc = QShortcut(QKeySequence("Ctrl+Shift+R"), self)
        sc.activated.connect(
            lambda: self._record_action.setChecked(not self._record_action.isChecked())
        )

        # Tuning arrows
        for key, step in [
            (QKeySequence(Qt.Key.Key_Left), -10e3),
            (QKeySequence(Qt.Key.Key_Right), +10e3),
            (QKeySequence("Shift+Left"), -100e3),
            (QKeySequence("Shift+Right"), +100e3),
            (QKeySequence("Ctrl+Left"), -1e6),
            (QKeySequence("Ctrl+Right"), +1e6),
        ]:
            s = QShortcut(key, self)
            s.activated.connect(lambda off=step: self._nudge_frequency(off))

        # Ctrl+P = screenshot, Ctrl+T = theme toggle, Ctrl+B = add bookmark,
        # Ctrl+E = error history, F1 = help, Ctrl+F = scanner
        for ks, slot in [
            ("Ctrl+P", self._save_screenshot),
            ("Ctrl+T", self._toggle_theme),
            ("Ctrl+B", self._bookmark_current_frequency),
            ("Ctrl+E", self._show_error_history),
            ("F1", self._show_help),
            ("Ctrl+F", self._show_scanner),
        ]:
            s = QShortcut(QKeySequence(ks), self)
            s.activated.connect(slot)

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
        # The decoder panel's protocol selector drives a live decoder over the
        # acquired samples (the panel was previously fed no data at all).
        self._decoder_panel.protocol_changed.connect(self._on_decoder_protocol_changed)
        # Click-to-tune from spectrum and waterfall
        self._spectrum.frequency_clicked.connect(self._on_frequency_changed)
        self._waterfall.frequency_clicked.connect(self._on_frequency_changed)

    def _on_panel_record_started(self, fmt: str) -> None:
        """Start recording from the control panel's Record button."""
        self._start_recording()
        self._record_action.blockSignals(True)
        self._record_action.setChecked(True)
        self._record_action.blockSignals(False)

    def _on_panel_record_stopped(self) -> None:
        """Stop recording from the control panel's Record button."""
        self._stop_recording()
        self._record_action.blockSignals(True)
        self._record_action.setChecked(False)
        self._record_action.blockSignals(False)

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
            # "Auto Detect" (or unknown): no single live decoder.
            self._decoder = None
            self._decoder_protocol = None
            return

        rate = getattr(self._device, "sample_rate", None) or 2.4e6
        try:
            protocol = ProtocolType(proto_value)
            self._decoder = create_protocol_decoder(protocol, sample_rate=float(rate))
            self._decoder_protocol = protocol
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not create %s decoder: %s", text, exc)
            self._decoder = None
            self._decoder_protocol = None

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

    def _on_frequency_changed(self, freq_hz: float):
        """Handle frequency change."""
        if self._device:
            self._device.set_frequency(freq_hz)

        # Update display
        if freq_hz >= 1e9:
            self._freq_label.setText(f"{freq_hz/1e9:.6f} GHz")
        else:
            self._freq_label.setText(f"{freq_hz/1e6:.3f} MHz")

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
                self._device.set_gain_mode("auto" if enabled else "manual")
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
        audible = {"AM", "FM", "USB", "LSB", "CW"}
        if mode in audible:
            self._audio.start(48000)
        else:
            self._audio.stop()

    def _on_bookmark_tune(self, freq_hz: float, label: str):
        """Tune to a bookmarked frequency."""
        self.set_frequency(freq_hz)
        self._show_status_error(f"Tuned to {label}", duration_ms=2000)
        # The status-bar helper is error-styled; use it as a generic toast.

    def _toggle_acquisition(self):
        """Toggle signal acquisition."""
        if self._is_running:
            self._stop_acquisition()
        else:
            self._start_acquisition()

    def _start_acquisition(self):
        """Start signal acquisition."""
        if not self._device:
            QMessageBox.warning(
                self, "No Device", "Please connect to an SDR device first."
            )
            return

        self._is_running = True
        self._start_action.setText("Stop")
        self._device_label.setText("Running...")

        # Start device
        self._device.start_rx()

        logger.info("Acquisition started")

    def _stop_acquisition(self):
        """Stop signal acquisition."""
        self._is_running = False
        self._start_action.setText("Start")
        self._device_label.setText("Stopped")

        # Update callsign panel
        if HAS_HAM_RADIO:
            self._callsign_panel.set_transmitting(False)

        if self._device:
            self._device.stop_rx()

        logger.info("Acquisition stopped")

    def _on_callsign_id_requested(self):
        """Handle callsign ID request from the callsign panel."""
        callsign = self._callsign_panel.get_callsign()
        if not callsign:
            QMessageBox.warning(
                self, "No Callsign", "Please enter your callsign in the HAM ID panel."
            )
            return

        logger.info(f"Callsign ID requested: {callsign}")
        self._device_label.setText(f"ID: DE {callsign}")

        try:
            from ..ham.callsign import generate_tx_id

            settings = self._callsign_panel.get_settings()

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
        current_freq = self._control_panel._freq_input.get_frequency()
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
            self._device_label.setText(f"TX: {description}")
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
                self._device_label.setText("TX Complete")
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

    def _toggle_recording(self, checked: bool):
        """Toggle recording from the toolbar action."""
        if checked:
            self._start_recording()
        else:
            self._stop_recording()
        # Keep the control panel's Record button in sync (without re-emitting).
        self._control_panel.set_recording_state(checked)

    def _start_recording(self):
        """Start recording."""
        import time

        self._recording = True
        self._samples_buffer = []
        self._recording_bytes = 0
        self._recording_started_at = time.monotonic()
        self._recording_label.setText("  REC  ")
        self._recording_label.setStyleSheet(
            "color: #1e1e2e; font-weight: bold; background-color: #f38ba8;"
            "border-radius: 3px; padding: 1px 6px;"
        )
        logger.info("Recording started")

    def _stop_recording(self):
        """Stop recording."""
        self._recording = False
        self._recording_label.setText("")
        logger.info("Recording stopped")

    def _update_display(self):
        """Update spectrum and waterfall displays."""
        if not self._is_running:
            return

        # Generate demo data if no device
        if not self._device:
            # Demo: noise + some signals
            n = 2048
            noise = np.random.randn(n) + 1j * np.random.randn(n)
            noise *= 0.1

            # Add some fake signals
            t = np.arange(n)
            sig1 = 0.5 * np.exp(2j * np.pi * 0.1 * t)
            sig2 = 0.3 * np.exp(2j * np.pi * 0.25 * t)

            samples = noise + sig1 + sig2
        else:
            samples = self._device.read_samples(2048)
            if samples is None:
                return

        # Compute spectrum (windowed, referenced to dBFS)
        power = self._power_spectrum_dbfs(samples)

        # Update spectrum widget
        self._spectrum.update_spectrum(power)

        # Update waterfall
        self._waterfall.add_line(power)

        # Update level display
        peak_level = float(np.max(power))
        self._level_label.setText(f"{peak_level:.1f} dB")

        # Squelch + audio: demodulate when carrier is above threshold
        if self._audio_enabled and peak_level >= self._squelch_db:
            self._demodulate_and_play(samples)

        # Record if active
        if self._recording:
            self._samples_buffer.append(samples)
            # complex64 = 8 bytes/sample
            self._recording_bytes += len(samples) * 8
            # Advance the control panel's recording timer.
            started = getattr(self, "_recording_started_at", None)
            if started is not None:
                import time

                self._control_panel.update_record_time(int(time.monotonic() - started))

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
        if mode not in ("AM", "FM", "USB", "LSB", "CW"):
            return
        try:
            # Simple demodulators sufficient for monitoring in the GUI
            if mode == "FM":
                # FM discriminator
                prod = samples[1:] * np.conj(samples[:-1])
                audio = np.angle(prod).astype(np.float32)
            elif mode == "AM":
                audio = (np.abs(samples) - np.mean(np.abs(samples))).astype(np.float32)
            else:  # SSB/CW: pass real part as crude envelope
                audio = samples.real.astype(np.float32)
            # Normalize softly then write
            peak = float(np.max(np.abs(audio)) + 1e-9)
            audio = audio / max(peak, 1e-6) * 0.5
            # Decimate to ~48 kHz assuming 2.4 MS/s (48x)
            rate = (
                getattr(self._device, "sample_rate", 2.4e6) if self._device else 2.4e6
            )
            decim = max(1, int(rate / 48000))
            self._audio.write(audio[::decim])
        except Exception as e:  # pragma: no cover
            logger.debug(f"Audio demod failed: {e}")

    def _update_status(self):
        """Update status bar."""
        import shutil
        import time

        if self._recording:
            started = getattr(self, "_recording_started_at", time.monotonic())
            elapsed = int(time.monotonic() - started)
            mb = self._recording_bytes / (1024 * 1024)
            # Show elapsed time + size + free space on the cwd volume
            try:
                free_gb = shutil.disk_usage(".").free / (1024**3)
                free_str = f"{free_gb:.1f} GB free"
            except OSError:
                free_str = ""
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self._recording_label.setText(
                f"REC {h:02d}:{m:02d}:{s:02d}  {mb:.1f} MB  {free_str}"
            )

    def _show_device_dialog(self):
        """Show device connection dialog."""
        from .device_dialog import DeviceDialog

        dialog = DeviceDialog(self)
        if dialog.exec():
            self._device = dialog.get_selected_device()
            if self._device:
                self._device_label.setText(f"Connected: {self._device.info.name}")
                logger.info(f"Connected to {self._device.info.name}")

    def _disconnect_device(self):
        """Disconnect from device."""
        if self._is_running:
            self._stop_acquisition()

        if self._device:
            self._device.close()
            self._device = None
            self._device_label.setText("Disconnected")
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
            self._show_status_error(f"Scan failed: {e}")
            return

        if not devices:
            QMessageBox.information(
                self,
                "No Devices",
                "No SDR devices detected.\n\n"
                "Connect an RTL-SDR or HackRF One and try again,\n"
                "or launch with --demo for demo mode.",
            )
        else:
            names = "\n".join(f"  - {d}" for d in devices)
            QMessageBox.information(
                self,
                "Devices Found",
                f"Detected {len(devices)} device(s):\n\n{names}\n\n"
                "Use Device → Connect to open one.",
            )

    def _open_recording(self):
        """Open a recording file and load its samples."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Recording",
            "",
            "I/Q Files (*.raw *.cf32 *.cs16 *.cu8);;WAV Files (*.wav);;All Files (*)",
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

        self._samples_buffer = [samples]
        if metadata.center_frequency > 0:
            self.set_frequency(metadata.center_frequency)
        logger.info(
            f"Loaded {len(samples)} samples from {filename} "
            f"(rate={metadata.sample_rate}, freq={metadata.center_frequency})"
        )
        QMessageBox.information(
            self,
            "Recording Loaded",
            f"Loaded {len(samples):,} samples\n"
            f"Sample rate: {metadata.sample_rate/1e6:.3f} MS/s\n"
            f"Center freq: {metadata.center_frequency/1e6:.3f} MHz",
        )

    def _save_recording(self):
        """Save the current recording buffer to an I/Q file."""
        if not self._samples_buffer:
            QMessageBox.information(self, "No Data", "No recorded data to save.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            "",
            "Complex Float32 (*.cf32);;Complex Int16 (*.cs16);;Raw I/Q (*.raw);;WAV (*.wav)",
        )
        if not filename:
            return

        from ..dsp.recording import FileFormat, SampleFormat, save_iq_file

        # Pick format from extension / filter
        ext = filename.lower()
        if ext.endswith(".wav") or "WAV" in selected_filter:
            fmt, sample_fmt = FileFormat.WAV, SampleFormat.INT16
        elif ext.endswith(".cs16") or "Int16" in selected_filter:
            fmt, sample_fmt = FileFormat.RAW, SampleFormat.INT16
        else:
            fmt, sample_fmt = FileFormat.RAW, SampleFormat.FLOAT32

        samples = np.concatenate(self._samples_buffer).astype(np.complex64)
        sample_rate = (
            getattr(self._device, "sample_rate", 2.4e6) if self._device else 2.4e6
        )
        center_freq = self._control_panel._freq_input.get_frequency()

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
        QMessageBox.information(
            self,
            "Recording Saved",
            f"Saved {len(samples):,} samples to:\n{filename}",
        )

    def _show_scanner(self):
        """Show the frequency scanner dialog."""
        from .scanner_dialog import ScannerDialog

        dialog = ScannerDialog(self, device=self._device)
        dialog.exec()

    def _show_radio_tuner(self):
        """Show the AM/FM radio tuner pop-out window."""
        if self._radio_tuner is None:
            sample_rate = 2.4e6
            if self._device:
                sample_rate = getattr(self._device, "sample_rate", 2.4e6)
            self._radio_tuner = RadioTunerWidget(self, sample_rate)
            # Connect frequency change to main tuner
            self._radio_tuner.frequency_changed.connect(
                self._on_radio_frequency_changed
            )

        self._radio_tuner.show()
        self._radio_tuner.raise_()
        self._radio_tuner.activateWindow()

    def _on_radio_frequency_changed(self, freq_hz: float, band: str):
        """Handle frequency change from radio tuner."""
        if self._device:
            try:
                self._device.center_freq = freq_hz
                self._control_panel.set_frequency(freq_hz)
                self._spectrum.set_center_freq(freq_hz)
                self._waterfall.set_center_freq(freq_hz)
                logger.info(f"Tuned to {freq_hz/1e6:.3f} MHz ({band})")
            except Exception as e:
                logger.error(f"Failed to tune: {e}")
                self._show_status_error(f"Tune failed: {e}")

    def _show_decoder_config(self):
        """Focus the decoder panel tab."""
        # Decoder UI lives in the right-hand tab; surface it instead of a stub.
        QMessageBox.information(
            self,
            "Protocol Decoder",
            "Protocol decoding runs in the 'Decoder' tab on the right.\n\n"
            "Supported: ADS-B, POCSAG, FLEX, AX.25/APRS, RDS, ACARS.\n"
            "Start acquisition and received packets will appear there.",
        )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SDR Module",
            "<h3>SDR Module</h3>"
            f"<p>Version {__version__}</p>"
            "<p>A Software Defined Radio application for signal visualization, "
            "frequency analysis, and protocol decoding.</p>"
            "<p>Supports RTL-SDR and HackRF One devices.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Spectrum analyzer and waterfall display</li>"
            "<li>Multiple demodulation modes</li>"
            "<li>Protocol decoders (POCSAG, FLEX, ADS-B, ACARS, AX.25, RDS)</li>"
            "<li>I/Q recording and playback</li>"
            "</ul>",
        )

    # ---- New helpers ----

    def _nudge_frequency(self, offset_hz: float) -> None:
        """Adjust the center frequency by the given offset."""
        current = self._control_panel._freq_input.get_frequency()
        new_freq = max(0.0, current + offset_hz)
        self.set_frequency(new_freq)

    def _apply_band_preset(self, freq_hz: float, mode: str, label: str) -> None:
        """Tune to a band preset and set matching demod mode."""
        self.set_frequency(freq_hz)
        idx = self._control_panel._demod_combo.findText(mode)
        if idx >= 0:
            self._control_panel._demod_combo.setCurrentIndex(idx)
        self._show_status_error(f"Band: {label}", duration_ms=1500)

    def _bookmark_current_frequency(self) -> None:
        """Save the current tuner frequency into bookmarks."""
        freq = self._control_panel._freq_input.get_frequency()
        label = f"{freq/1e6:.3f} MHz"
        self._bookmarks_panel.add_bookmark(label, freq)
        self._show_status_error(f"Bookmarked {label}", duration_ms=1500)

    def _import_channels_csv(self) -> None:
        """Import memory channels from a CHIRP CSV into the bookmarks panel."""
        count = self._bookmarks_panel.import_csv()
        if count:
            self._show_status_error(f"Imported {count} channel(s)", duration_ms=2000)

    def _export_channels_csv(self) -> None:
        """Export the saved channels to a CHIRP-compatible CSV file."""
        count = self._bookmarks_panel.export_csv()
        if count:
            self._show_status_error(f"Exported {count} channel(s)", duration_ms=2000)

    def _save_screenshot(self) -> None:
        """Save a PNG of the window (spectrum + waterfall + panels)."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "sdr.png", "PNG (*.png)"
        )
        if not filename:
            return
        pixmap = self.grab()
        if pixmap.save(filename):
            self._show_status_error(f"Saved {filename}", duration_ms=2000)
        else:
            QMessageBox.warning(self, "Save Failed", "Could not save screenshot.")

    def _toggle_theme(self) -> None:
        """Switch between dark and light stylesheets."""
        self._theme = "light" if self._theme == "dark" else "dark"
        self._settings.set("theme", self._theme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_stylesheet(self._theme))
        self._show_status_error(f"Theme: {self._theme}", duration_ms=1500)

    def _show_help(self) -> None:
        """Open the keyboard shortcuts reference."""
        from .help_dialog import HelpDialog

        HelpDialog(self).exec()

    def _show_error_history(self) -> None:
        """Open the error history viewer."""
        from .error_log_dialog import ErrorLogDialog, install_history_handler

        install_history_handler()
        ErrorLogDialog(self).exec()

    def _set_audio_enabled(self, enabled: bool) -> None:
        """Toggle audio output on/off and persist the preference."""
        self._audio_enabled = bool(enabled)
        self._settings.set("audio_enabled", self._audio_enabled)
        # Apply immediately to current demod mode
        mode = self._control_panel._demod_combo.currentText()
        self._start_or_stop_audio(mode)

    def _poll_hotplug(self) -> None:
        """Poll for device hot-plug changes and notify on new devices."""
        try:
            from ..core.device_manager import DeviceManager

            devices = DeviceManager().scan_devices()
        except Exception:
            return
        names = {str(d) for d in devices}
        if not self._known_device_names:
            # First poll after init — baseline silently.
            self._known_device_names = names
            return
        added = names - self._known_device_names
        self._known_device_names = names
        if added:
            for name in added:
                logger.info(f"Device connected: {name}")
            self._show_status_error(
                f"New device: {next(iter(added))}", duration_ms=4000
            )

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
            return
        self._control_panel.set_frequency(freq)
        self._control_panel.set_gain(gain)
        self._control_panel.set_squelch_db(squelch)
        self._control_panel.set_agc_enabled(agc)
        idx = self._control_panel._demod_combo.findText(demod)
        if idx >= 0:
            self._control_panel._demod_combo.setCurrentIndex(idx)
        # Window geometry
        geom = self._settings.load_geometry("main")
        if geom:
            try:
                self.restoreGeometry(geom)
            except Exception as e:
                logger.debug(f"Could not restore window geometry: {e}")

    def _persist_state(self) -> None:
        """Save settings on exit."""
        try:
            freq = self._control_panel._freq_input.get_frequency()
            gain = float(self._control_panel._gain_slider.value())
            self._settings.set("frequency_hz", freq)
            self._settings.set("gain_db", gain)
            self._settings.set("squelch_db", self._squelch_db)
            self._settings.set("agc_enabled", self._agc_enabled)
            self._settings.set("theme", self._theme)
            self._settings.save_geometry("main", self.saveGeometry())
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
        self._audio.stop()
        if self._is_running:
            self._stop_acquisition()

        if self._device:
            self._device.close()

        logger.info("Application closing")
        event.accept()

    def _start_demo_mode(self):
        """Start demo mode with synthetic signals."""
        from .device_dialog import MockDevice

        self._device = MockDevice()
        self._device.start_rx()
        self._device_label.setText("Demo Mode")
        self._is_running = True
        self._start_action.setText("Stop")
        logger.info("Demo mode started")

    def set_frequency(self, freq_hz: float):
        """Set the center frequency."""
        self._control_panel.set_frequency(freq_hz)
        self._on_frequency_changed(freq_hz)

    def set_gain(self, gain_db: float):
        """Set the RF gain."""
        self._control_panel.set_gain(gain_db)
        self._on_gain_changed(gain_db)
