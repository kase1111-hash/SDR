"""
Device selection dialog.

Allows user to select and configure SDR device.
"""

from typing import Optional, List

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QComboBox, QGroupBox, QPushButton,
        QTableWidget, QTableWidgetItem, QHeaderView,
        QDialogButtonBox, QMessageBox, QSpinBox
    )
    from PyQt6.QtCore import Qt
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import logging

logger = logging.getLogger(__name__)


class DeviceDialog(QDialog if HAS_PYQT6 else object):
    """
    Device selection and configuration dialog.

    Lists available SDR devices and allows connection.
    """

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._selected_device = None
        self._devices: List[dict] = []

        self.setWindowTitle("Select SDR Device")
        self.setMinimumSize(500, 400)

        self._setup_ui()
        self._refresh_devices()

    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)

        # Device list
        list_group = QGroupBox("Available Devices")
        list_layout = QVBoxLayout(list_group)

        self._device_table = QTableWidget()
        self._device_table.setColumnCount(4)
        self._device_table.setHorizontalHeaderLabels(["Type", "Name", "Serial", "Status"])
        self._device_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._device_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._device_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._device_table.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self._device_table)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_devices)
        list_layout.addWidget(refresh_btn)

        layout.addWidget(list_group)

        # Device settings
        settings_group = QGroupBox("Device Settings")
        settings_layout = QGridLayout(settings_group)

        # Sample rate
        settings_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self._rate_combo = QComboBox()
        self._rate_combo.addItems([
            "1.0 MS/s", "1.4 MS/s", "1.8 MS/s", "2.0 MS/s",
            "2.4 MS/s", "2.56 MS/s", "3.2 MS/s"
        ])
        self._rate_combo.setCurrentText("2.4 MS/s")
        settings_layout.addWidget(self._rate_combo, 0, 1)

        # PPM correction
        settings_layout.addWidget(QLabel("PPM Correction:"), 1, 0)
        self._ppm_spin = QSpinBox()
        self._ppm_spin.setRange(-100, 100)
        self._ppm_spin.setValue(0)
        settings_layout.addWidget(self._ppm_spin, 1, 1)

        # Direct sampling (RTL-SDR)
        settings_layout.addWidget(QLabel("Direct Sampling:"), 2, 0)
        self._direct_combo = QComboBox()
        self._direct_combo.addItems(["Off", "I-ADC", "Q-ADC"])
        settings_layout.addWidget(self._direct_combo, 2, 1)

        layout.addWidget(settings_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _refresh_devices(self):
        """Refresh device list."""
        self._devices.clear()
        self._device_table.setRowCount(0)

        # Try to enumerate devices
        try:
            self._enumerate_rtlsdr()
        except Exception as e:
            logger.debug(f"RTL-SDR enumeration failed: {e}")

        try:
            self._enumerate_hackrf()
        except Exception as e:
            logger.debug(f"HackRF enumeration failed: {e}")

        # Add demo device if no real devices found
        if not self._devices:
            self._add_device("Demo", "Demo Device", "N/A", "Available")

    def _enumerate_rtlsdr(self):
        """Enumerate RTL-SDR devices."""
        try:
            from sdr_module.devices.rtlsdr import RTLSDRDevice

            # Try to get device count
            device = RTLSDRDevice()
            devices = device.enumerate()

            for i, dev_info in enumerate(devices):
                self._add_device(
                    "RTL-SDR",
                    dev_info.get("name", f"RTL-SDR #{i}"),
                    dev_info.get("serial", "Unknown"),
                    "Available"
                )
                self._devices.append({
                    "type": "rtlsdr",
                    "index": i,
                    "info": dev_info
                })

        except ImportError:
            # Add mock device for testing
            self._add_device("RTL-SDR", "RTL-SDR (Mock)", "000001", "Demo Mode")
            self._devices.append({
                "type": "rtlsdr_mock",
                "index": 0,
                "info": {}
            })

    def _enumerate_hackrf(self):
        """Enumerate HackRF devices."""
        try:
            from sdr_module.devices.hackrf import HackRFDevice

            device = HackRFDevice()
            devices = device.enumerate()

            for i, dev_info in enumerate(devices):
                self._add_device(
                    "HackRF",
                    dev_info.get("name", f"HackRF #{i}"),
                    dev_info.get("serial", "Unknown"),
                    "Available"
                )
                self._devices.append({
                    "type": "hackrf",
                    "index": i,
                    "info": dev_info
                })

        except ImportError:
            # Add mock device for testing
            self._add_device("HackRF", "HackRF One (Mock)", "000002", "Demo Mode")
            self._devices.append({
                "type": "hackrf_mock",
                "index": 0,
                "info": {}
            })

    def _add_device(self, dev_type: str, name: str, serial: str, status: str):
        """Add device to table."""
        row = self._device_table.rowCount()
        self._device_table.insertRow(row)

        self._device_table.setItem(row, 0, QTableWidgetItem(dev_type))
        self._device_table.setItem(row, 1, QTableWidgetItem(name))
        self._device_table.setItem(row, 2, QTableWidgetItem(serial))
        self._device_table.setItem(row, 3, QTableWidgetItem(status))

    def _on_selection_changed(self):
        """Handle selection change."""
        rows = self._device_table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
            if row < len(self._devices):
                dev = self._devices[row]
                # Update settings based on device type
                if "rtlsdr" in dev.get("type", ""):
                    self._direct_combo.setEnabled(True)
                else:
                    self._direct_combo.setEnabled(False)
                    self._direct_combo.setCurrentIndex(0)

    def _on_accept(self):
        """Handle OK button."""
        rows = self._device_table.selectionModel().selectedRows()

        if not rows:
            QMessageBox.warning(
                self, "No Selection",
                "Please select a device."
            )
            return

        row = rows[0].row()
        if row >= len(self._devices):
            self.reject()
            return

        dev = self._devices[row]

        # Try to open device
        try:
            self._selected_device = self._open_device(dev)
            if self._selected_device:
                self.accept()
            else:
                QMessageBox.warning(
                    self, "Connection Failed",
                    "Failed to connect to the selected device."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error connecting to device: {e}"
            )

    def _open_device(self, dev: dict):
        """Open the specified device."""
        dev_type = dev.get("type", "")

        if dev_type == "rtlsdr":
            from sdr_module.devices.rtlsdr import RTLSDRDevice
            device = RTLSDRDevice()
            if device.open(dev.get("index", 0)):
                # Apply settings
                rate_str = self._rate_combo.currentText()
                rate = float(rate_str.split()[0]) * 1e6
                device.set_sample_rate(rate)
                return device

        elif dev_type == "hackrf":
            from sdr_module.devices.hackrf import HackRFDevice
            device = HackRFDevice()
            if device.open(dev.get("index", 0)):
                rate_str = self._rate_combo.currentText()
                rate = float(rate_str.split()[0]) * 1e6
                device.set_sample_rate(rate)
                return device

        # Return mock device for demo mode
        return MockDevice()

    def get_selected_device(self):
        """Get the selected and opened device."""
        return self._selected_device


class MockDevice:
    """Mock device for demo mode."""

    class MockInfo:
        name = "Demo Device"
        serial = "DEMO"

    def __init__(self):
        self.info = self.MockInfo()
        self._frequency = 100e6
        self._sample_rate = 2.4e6
        self._gain = 20
        self._running = False

    def set_frequency(self, freq: float) -> bool:
        self._frequency = freq
        return True

    def set_sample_rate(self, rate: float) -> bool:
        self._sample_rate = rate
        return True

    def set_gain(self, gain: float) -> bool:
        self._gain = gain
        return True

    def set_bandwidth(self, bw: float) -> bool:
        return True

    def start_rx(self) -> bool:
        self._running = True
        return True

    def stop_rx(self) -> bool:
        self._running = False
        return True

    def read_samples(self, num_samples: int):
        import numpy as np
        if not self._running:
            return None
        # Generate demo samples
        return np.random.randn(num_samples) * 0.1 + 1j * np.random.randn(num_samples) * 0.1

    def close(self):
        self._running = False
