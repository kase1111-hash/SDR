"""
Main application window.

Provides the primary window with all panels and controls.
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QStatusBar, QToolBar, QMenuBar, QMenu,
        QLabel, QMessageBox, QFileDialog, QDockWidget,
        QTabWidget, QProgressBar
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt6.QtGui import QAction, QIcon, QKeySequence
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .spectrum_widget import SpectrumWidget
from .waterfall_widget import WaterfallWidget
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel

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

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        self._setup_timers()

        # Connect signals
        self._connect_signals()

        # Auto-start demo mode
        if demo_mode:
            self._start_demo_mode()

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("SDR Module")
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

        # Add placeholder for future tabs
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

        # Help menu
        help_menu = menubar.addMenu("&Help")

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

        # Frequency display
        toolbar.addWidget(QLabel("Freq: "))
        self._freq_label = QLabel("100.000 MHz")
        self._freq_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        toolbar.addWidget(self._freq_label)

        toolbar.addSeparator()

        # Signal level
        toolbar.addWidget(QLabel("Level: "))
        self._level_label = QLabel("-80.0 dB")
        toolbar.addWidget(self._level_label)

    def _setup_statusbar(self):
        """Setup status bar."""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # Device status
        self._device_label = QLabel("No device")
        statusbar.addWidget(self._device_label)

        # Sample rate
        self._rate_label = QLabel("Rate: 2.4 MS/s")
        statusbar.addWidget(self._rate_label)

        # Buffer indicator
        self._buffer_progress = QProgressBar()
        self._buffer_progress.setMaximumWidth(100)
        self._buffer_progress.setMaximumHeight(16)
        self._buffer_progress.setRange(0, 100)
        self._buffer_progress.setValue(0)
        statusbar.addPermanentWidget(self._buffer_progress)

        # Recording status
        self._recording_label = QLabel("")
        statusbar.addPermanentWidget(self._recording_label)

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

    def _connect_signals(self):
        """Connect control panel signals."""
        self._control_panel.frequency_changed.connect(self._on_frequency_changed)
        self._control_panel.gain_changed.connect(self._on_gain_changed)
        self._control_panel.bandwidth_changed.connect(self._on_bandwidth_changed)

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
                self, "No Device",
                "Please connect to an SDR device first."
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

        if self._device:
            self._device.stop_rx()

        logger.info("Acquisition stopped")

    def _toggle_recording(self, checked: bool):
        """Toggle recording."""
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Start recording."""
        self._recording = True
        self._samples_buffer = []
        self._recording_label.setText("REC")
        self._recording_label.setStyleSheet("color: red; font-weight: bold;")
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

        # Compute spectrum
        spectrum = np.fft.fftshift(np.fft.fft(samples))
        power = 20 * np.log10(np.abs(spectrum) + 1e-10)

        # Update spectrum widget
        self._spectrum.update_spectrum(power)

        # Update waterfall
        self._waterfall.add_line(power)

        # Update level display
        peak_level = np.max(power)
        self._level_label.setText(f"{peak_level:.1f} dB")

        # Record if active
        if self._recording:
            self._samples_buffer.append(samples)

    def _update_status(self):
        """Update status bar."""
        if self._recording:
            samples_count = sum(len(s) for s in self._samples_buffer)
            self._recording_label.setText(f"REC: {samples_count/1e6:.2f} MS")

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
        """Refresh device list."""
        logger.info("Refreshing device list...")
        # This would trigger a device rescan

    def _open_recording(self):
        """Open a recording file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Recording",
            "", "I/Q Files (*.raw *.cf32 *.cs16 *.cu8);;WAV Files (*.wav);;All Files (*)"
        )
        if filename:
            logger.info(f"Opening recording: {filename}")
            # Load and playback would go here

    def _save_recording(self):
        """Save the current recording."""
        if not self._samples_buffer:
            QMessageBox.information(
                self, "No Data",
                "No recorded data to save."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Recording",
            "", "Complex Float32 (*.cf32);;Complex Int16 (*.cs16);;WAV (*.wav)"
        )
        if filename:
            logger.info(f"Saving recording: {filename}")
            # Save logic would go here

    def _show_scanner(self):
        """Show frequency scanner dialog."""
        QMessageBox.information(
            self, "Scanner",
            "Frequency scanner will be implemented in a future update."
        )

    def _show_decoder_config(self):
        """Show decoder configuration dialog."""
        QMessageBox.information(
            self, "Decoder",
            "Protocol decoder configuration will be implemented in a future update."
        )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About SDR Module",
            "<h3>SDR Module</h3>"
            "<p>Version 0.1.0</p>"
            "<p>A Software Defined Radio application for signal visualization, "
            "frequency analysis, and protocol decoding.</p>"
            "<p>Supports RTL-SDR and HackRF One devices.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Spectrum analyzer and waterfall display</li>"
            "<li>Multiple demodulation modes</li>"
            "<li>Protocol decoders (POCSAG, FLEX, ADS-B, ACARS, etc.)</li>"
            "<li>I/Q recording and playback</li>"
            "<li>Plugin system for extensions</li>"
            "</ul>"
        )

    def closeEvent(self, event):
        """Handle window close."""
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
