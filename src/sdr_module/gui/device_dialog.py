"""
Device selection dialog.

Lists the SDR hardware that is actually connected (plus a Demo Device that
produces simulated signals), lets the user pick a sample rate and opens the
chosen device.

:class:`MockDevice` is the simulated receiver behind the Demo Device and the
``--demo`` command-line flag.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QHeaderView,
        QLabel,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..utils.tooltips import get_short_tip
from .settings_store import DEFAULT_FREQUENCY_HZ

if HAS_PYQT6:
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

# Sample rates offered in the dialog. Each device type only shows the ones
# inside its supported range (see _RATE_LIMITS).
_SAMPLE_RATES = (1.0e6, 1.4e6, 1.8e6, 2.0e6, 2.4e6, 2.56e6, 3.2e6, 4e6, 8e6, 10e6, 20e6)
_DEFAULT_RATE = 2.4e6

# (min, max) sample rate per device type, in samples/second.
_RATE_LIMITS: Dict[str, Tuple[float, float]] = {
    "rtlsdr": (0.9e6, 2.56e6),
    "hackrf": (2.0e6, 20e6),
    "demo": (1.0e6, 3.2e6),
}

# Python packages that provide each hardware driver, and how to install them.
_DRIVERS = {
    "rtlsdr": ("rtlsdr", "RTL-SDR", 'pip install "sdr-module[rtlsdr]"'),
    "hackrf": (
        "python_hackrf",
        "HackRF",
        'pip install "sdr-module[hackrf]" (needs the libhackrf system library)',
    ),
}


def _driver_installed(module_name: str) -> bool:
    """True if the Python binding for a hardware driver can be imported."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def missing_drivers() -> List[Tuple[str, str]]:
    """``(device label, install command)`` for each hardware driver whose
    Python binding isn't installed, so its devices can't be found."""
    return [
        (label, install)
        for module_name, label, install in _DRIVERS.values()
        if not _driver_installed(module_name)
    ]


def _format_rate(rate: float) -> str:
    """``2400000.0`` -> ``"2.4 MS/s"``."""
    return f"{rate / 1e6:g} MS/s"


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
        self._missing_drivers: List[Tuple[str, str]] = []

        self.setWindowTitle("Connect SDR Device")
        self.setMinimumSize(540, 480)

        self._setup_ui()
        self._refresh_devices()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _setup_ui(self):
        """Setup UI elements."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Device list
        list_group = QGroupBox("Available Devices")
        list_layout = QVBoxLayout(list_group)
        list_layout.setSpacing(8)

        self._device_table = QTableWidget()
        self._device_table.setColumnCount(4)
        self._device_table.setHorizontalHeaderLabels(
            ["Type", "Name", "Serial", "Status"]
        )
        header = self._device_table.horizontalHeader()
        # Type, Name and Status are short; the serial (32 hex digits on a
        # HackRF) takes the remaining width and is elided in the middle, with
        # the full serial in its tooltip, instead of crushing the Name column.
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._device_table.setWordWrap(False)
        self._device_table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._device_table.verticalHeader().setVisible(False)
        self._device_table.setShowGrid(False)
        self._device_table.setAlternatingRowColors(True)
        self._device_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._device_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._device_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        # Arrow keys pick a row; Tab moves on to the settings and buttons
        # instead of walking the cells (which trapped keyboard focus here).
        self._device_table.setTabKeyNavigation(False)
        self._device_table.setAccessibleName("Available devices")
        self._device_table.setToolTip(
            "Double-click a device, or press Enter, to connect to it."
        )
        self._device_table.itemSelectionChanged.connect(self._on_selection_changed)
        self._device_table.itemDoubleClicked.connect(lambda _item: self._on_accept())
        list_layout.addWidget(self._device_table, 1)

        # What was found, and how to fix it when nothing was.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        set_role(self._status_label, "callout", "info")
        list_layout.addWidget(self._status_label)

        self._driver_label = QLabel()
        self._driver_label.setWordWrap(True)
        self._driver_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        set_role(self._driver_label, "hint")
        list_layout.addWidget(self._driver_label)

        layout.addWidget(list_group, 1)

        # Device settings
        settings_group = QGroupBox("Device Settings")
        self._settings_form = form = QFormLayout(settings_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._rate_combo = QComboBox()
        self._rate_combo.setToolTip(get_short_tip("sample_rate"))
        self._populate_rates("demo")
        form.addRow("&Sample rate:", self._rate_combo)

        self._ppm_spin = QSpinBox()
        self._ppm_spin.setRange(-100, 100)
        self._ppm_spin.setValue(0)
        self._ppm_spin.setSuffix(" ppm")
        self._ppm_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("&Frequency correction:", self._ppm_spin)

        self._direct_combo = QComboBox()
        self._direct_combo.addItems(["Off", "I-ADC", "Q-ADC"])
        form.addRow("&Direct sampling:", self._direct_combo)

        layout.addWidget(settings_group)

        # Dialog buttons
        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._connect_btn = QPushButton("&Connect")
        set_role(self._connect_btn, "primary")
        self._connect_btn.setDefault(True)
        self._button_box.addButton(
            self._connect_btn, QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._refresh_btn = QPushButton("&Refresh")
        self._refresh_btn.setToolTip("Look for connected SDR hardware again")
        self._refresh_btn.setAutoDefault(False)
        self._refresh_btn.clicked.connect(self._refresh_devices)
        self._button_box.addButton(
            self._refresh_btn, QDialogButtonBox.ButtonRole.ResetRole
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)
        # Making Connect the default also makes it the button box's focus
        # proxy (set again when the box is reparented), so Tab into the box
        # jumped straight to Connect and never reached Refresh, which is laid
        # out before it. Without the proxy Tab visits every button in order.
        self._button_box.setFocusProxy(None)

        self._update_setting_controls(None)

    def _populate_rates(self, dev_kind: str) -> None:
        """Fill the sample-rate combo with the rates ``dev_kind`` supports."""
        current = self._rate_combo.currentData() or _DEFAULT_RATE
        low, high = _RATE_LIMITS.get(dev_kind, _RATE_LIMITS["demo"])
        rates = [r for r in _SAMPLE_RATES if low <= r <= high] or [_DEFAULT_RATE]
        self._rate_combo.blockSignals(True)
        self._rate_combo.clear()
        for rate in rates:
            self._rate_combo.addItem(_format_rate(rate), rate)
        # Keep the previous choice when it is still valid, else the closest.
        best = min(range(len(rates)), key=lambda i: abs(rates[i] - float(current)))
        self._rate_combo.setCurrentIndex(best)
        self._rate_combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Enumeration
    # ------------------------------------------------------------------ #
    def _refresh_devices(self):
        """Refresh device list."""
        self._devices.clear()
        self._missing_drivers = []
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

        hardware_count = len(self._devices)

        # The Demo Device is always offered, after any real hardware. Every
        # table row needs a matching entry in self._devices: _on_accept
        # rejects a selected row whose index has no entry.
        self._add_device("Demo", "Demo Device", "—", "Simulated signals")
        self._devices.append({"type": "demo", "index": 0, "info": {}})

        self._device_table.selectRow(0)
        self._update_status(hardware_count)

    def _enumerate_rtlsdr(self):
        """Enumerate RTL-SDR devices."""
        self._enumerate("rtlsdr")

    def _enumerate_hackrf(self):
        """Enumerate HackRF devices."""
        self._enumerate("hackrf")

    def _enumerate(self, dev_type: str) -> None:
        """List the connected devices of one type (skipped if no driver)."""
        module_name, label, install = _DRIVERS[dev_type]
        if not _driver_installed(module_name):
            self._missing_drivers.append((label, install))
            return

        if dev_type == "rtlsdr":
            from sdr_module.devices.rtlsdr import RTLSDRDevice as device_class
        else:
            from sdr_module.devices.hackrf import HackRFDevice as device_class

        for i, info in enumerate(device_class.list_devices()):
            name = getattr(info, "name", None) or f"{label} #{i}"
            serial = getattr(info, "serial", None) or "Unknown"
            index = getattr(info, "index", i)
            self._add_device(label, name, serial, "Available")
            self._devices.append({"type": dev_type, "index": index, "info": info})

    def _add_device(self, dev_type: str, name: str, serial: str, status: str):
        """Add device to table."""
        row = self._device_table.rowCount()
        self._device_table.insertRow(row)

        for col, text in enumerate((dev_type, name, serial, status)):
            item = QTableWidgetItem(text)
            if col == 2 and len(text) > 12:
                item.setToolTip(f"Serial number {text}")
            self._device_table.setItem(row, col, item)

    def _update_status(self, hardware_count: int) -> None:
        """Explain what was found, and what to do when nothing was."""
        # A leading glyph, as in the app's other callouts.
        if hardware_count:
            noun = "device" if hardware_count == 1 else "devices"
            text = (
                f"\u2713 Found {hardware_count} SDR {noun}. Select one and "
                "click Connect."
            )
            tone = "success"
        else:
            text = (
                "\u24d8 No SDR hardware found. Plug in an RTL-SDR or HackRF One "
                "and click Refresh, or connect the Demo Device to explore with "
                "simulated signals."
            )
            tone = "info"
        self._status_label.setText(text)
        set_tone(self._status_label, tone)

        notes = [
            f"{label} driver not installed: {install}"
            for label, install in self._missing_drivers
        ]
        self._driver_label.setText("\n".join(notes))
        self._driver_label.setVisible(bool(notes))

    # ------------------------------------------------------------------ #
    # Selection and settings
    # ------------------------------------------------------------------ #
    def _selected_entry(self) -> Optional[dict]:
        rows = self._device_table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return self._devices[row] if row < len(self._devices) else None

    def _on_selection_changed(self):
        """Handle selection change."""
        self._update_setting_controls(self._selected_entry())

    def _update_setting_controls(self, dev: Optional[dict]) -> None:
        """Enable only the settings the selected device's driver can apply."""
        self._connect_btn.setEnabled(dev is not None)
        self._connect_btn.setToolTip(
            "" if dev is not None else "Select a device in the list first"
        )
        dev_type = (dev or {}).get("type", "")
        kind = "rtlsdr" if "rtlsdr" in dev_type else dev_type
        if kind not in _RATE_LIMITS:
            kind = "demo"
        self._populate_rates(kind)

        device_class = self._device_class(dev_type)
        name = {"rtlsdr": "RTL-SDR", "hackrf": "HackRF"}.get(kind, "")
        unsupported = (
            f"Not supported by the {name} driver yet."
            if name
            else "Not needed for the Demo Device."
        )

        ppm_ok = device_class is not None and hasattr(
            device_class, "set_freq_correction"
        )
        self._ppm_spin.setEnabled(ppm_ok)
        self._ppm_spin.setToolTip(
            "Corrects the tuner's crystal error, in parts per million."
            if ppm_ok
            else unsupported
        )

        direct_ok = (
            kind == "rtlsdr"
            and device_class is not None
            and hasattr(device_class, "set_direct_sampling")
        )
        self._direct_combo.setEnabled(direct_ok)
        if not direct_ok:
            self._direct_combo.setCurrentIndex(0)
        self._direct_combo.setToolTip(
            get_short_tip("direct_sampling")
            if direct_ok
            else (
                unsupported
                if kind == "rtlsdr"
                else "RTL-SDR only. " + get_short_tip("direct_sampling")
            )
        )
        # Dim each label with its field so unavailable settings read as such.
        for field in (self._ppm_spin, self._direct_combo):
            label = self._settings_form.labelForField(field)
            if label is not None:
                label.setEnabled(field.isEnabled())
                label.setToolTip(field.toolTip())

    @staticmethod
    def _device_class(dev_type: str) -> Any:
        """Driver class for a device type, or None for the demo device."""
        try:
            if dev_type == "rtlsdr":
                from sdr_module.devices.rtlsdr import RTLSDRDevice

                return RTLSDRDevice
            if dev_type == "hackrf":
                from sdr_module.devices.hackrf import HackRFDevice

                return HackRFDevice
        except ImportError:
            return None
        return None

    def _selected_rate(self) -> float:
        rate = self._rate_combo.currentData()
        return float(rate) if rate else _DEFAULT_RATE

    # ------------------------------------------------------------------ #
    # Connect
    # ------------------------------------------------------------------ #
    def _on_accept(self):
        """Handle the Connect button."""
        rows = self._device_table.selectionModel().selectedRows()

        if not rows:
            QMessageBox.information(
                self, "No Device Selected", "Select a device in the list first."
            )
            return

        row = rows[0].row()
        if row >= len(self._devices):
            self.reject()
            return

        dev = self._devices[row]
        name_item = self._device_table.item(row, 1)
        name = name_item.text() if name_item is not None else "the device"

        # Try to open device
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._selected_device = self._open_device(dev)
        except Exception as e:
            self._selected_device = None
            QGuiApplication.restoreOverrideCursor()
            logger.error(f"Error connecting to {name}: {e}")
            QMessageBox.critical(
                self,
                "Connection Failed",
                f"Could not connect to {name}:\n{e}",
            )
            return
        QGuiApplication.restoreOverrideCursor()

        if self._selected_device:
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Connection Failed",
                f"Could not open {name}.\n\n"
                "Check that it is plugged in and not in use by another "
                "program, then click Refresh and try again. Details are in "
                "Tools > Error History.",
            )

    def _open_device(self, dev: dict):
        """Open the specified device.

        Returns the opened device, or None when real hardware fails to open
        (never a stand-in demo device, which would look like a connection).
        """
        dev_type = dev.get("type", "")
        rate = self._selected_rate()

        if dev_type in ("rtlsdr", "hackrf"):
            device_class = self._device_class(dev_type)
            if device_class is None:
                return None
            device = device_class()
            if not device.open(dev.get("index", 0)):
                return None
            device.set_sample_rate(rate)
            self._apply_optional_settings(device)
            return device

        # Demo device (and legacy mock entries): simulated signals.
        device = MockDevice()
        device.set_sample_rate(rate)
        return device

    def _apply_optional_settings(self, device) -> None:
        """Apply PPM / direct sampling when the driver supports them."""
        ppm = self._ppm_spin.value()
        if ppm and self._ppm_spin.isEnabled():
            try:
                device.set_freq_correction(ppm)
            except Exception as e:
                logger.warning(f"Could not apply {ppm} ppm correction: {e}")
        mode = self._direct_combo.currentIndex()
        if mode and self._direct_combo.isEnabled():
            try:
                device.set_direct_sampling(mode)
            except Exception as e:
                logger.warning(f"Could not enable direct sampling: {e}")

    def get_selected_device(self):
        """Get the selected and opened device."""
        return self._selected_device


# --------------------------------------------------------------------------- #
# Demo device
# --------------------------------------------------------------------------- #

# Simulated band plan: (frequency Hz, kind, level dBFS at 20 dB gain,
# (period s, on s) for intermittent signals or None for continuous).
# Levels are per-bin peaks on the main display's dBFS scale; for the Mode S
# replies on 1090 MHz ("modes"), the amplitude of the decodable airliner's
# pulses (see _SQUITTERS).
#
# Stations sit where the app sends people: the strongest one is on 100.1 MHz,
# the start-up frequency and the "FM Broadcast" preset of the welcome screen
# and Radio > Band Presets (it plays music, see _MUSIC_BARS), and the radio
# tuner's FM presets (88.5, 93.3, 97.1, 99.5, 101.1, 104.3 MHz) each find a
# station.
_DEMO_SIGNALS: Tuple[Tuple[float, str, float, Optional[Tuple[float, float]]], ...] = (
    # FM broadcast band
    (88.5e6, "wfm", -38.0, None),
    (89.3e6, "wfm", -52.0, None),
    (90.1e6, "wfm", -32.0, None),
    (91.5e6, "wfm", -46.0, None),
    (93.3e6, "wfm", -28.0, None),
    (94.7e6, "wfm", -50.0, None),
    (95.5e6, "wfm", -36.0, None),
    (97.1e6, "wfm", -24.0, None),
    (98.3e6, "wfm", -42.0, None),
    (99.5e6, "wfm", -34.0, None),
    (100.1e6, "wfm", -18.0, None),
    (101.1e6, "wfm", -44.0, None),
    (102.7e6, "wfm", -30.0, None),
    (103.5e6, "wfm", -54.0, None),
    (104.3e6, "wfm", -26.0, None),
    (105.9e6, "wfm", -40.0, None),
    (106.7e6, "wfm", -48.0, None),
    (107.5e6, "wfm", -34.0, None),
    # Narrowband carriers in the start-up view
    (99.82e6, "cw", -62.0, None),
    (100.715e6, "nfm", -58.0, (9.0, 5.0)),
    # Airband (AM)
    (118.1e6, "am", -52.0, (7.0, 3.0)),
    (119.1e6, "am", -46.0, (11.0, 4.0)),
    (121.5e6, "am", -60.0, (23.0, 3.0)),
    (124.35e6, "am", -50.0, (8.0, 5.0)),
    (125.0e6, "am", -44.0, None),
    (127.85e6, "am", -48.0, (13.0, 6.0)),
    # 2 m amateur band
    (144.39e6, "burst", -50.0, (4.0, 0.4)),
    (146.52e6, "nfm", -46.0, (12.0, 6.0)),
    (146.94e6, "nfm", -38.0, (10.0, 7.0)),
    (147.105e6, "nfm", -54.0, (15.0, 5.0)),
    # NOAA weather radio
    (162.4e6, "nfm", -52.0, None),
    (162.475e6, "nfm", -62.0, None),
    (162.55e6, "nfm", -44.0, None),
    # ISM 433 MHz and 70 cm
    (433.92e6, "burst", -40.0, (2.5, 0.3)),
    (434.1e6, "burst", -56.0, (6.0, 0.5)),
    (446.0e6, "nfm", -48.0, (14.0, 6.0)),
    (446.1e6, "nfm", -58.0, (9.0, 3.0)),
    # ISM 915 MHz and ADS-B
    (915.0e6, "burst", -50.0, (1.5, 0.2)),
    (1090.0e6, "modes", -26.0, None),
)

# Morse keying for the CW beacon ("VVV DE SDR"): 1 = key down, one dit each.
_MORSE = {"V": "...-", "D": "-..", "E": ".", "S": "...", "R": ".-."}


def _morse_keying(message: str) -> Tuple[int, ...]:
    units: List[int] = []
    for word in message.split():
        for letter in word:
            for symbol in _MORSE[letter]:
                units.extend([1] * (1 if symbol == "." else 3))
                units.append(0)
            units.extend([0, 0])
        units.extend([0] * 4)
    units.extend([1] * 30 + [0] * 7)  # long carrier, then repeat
    return tuple(units)


_CW_KEYING = _morse_keying("VVV DE SDR")
_CW_DIT_S = 0.08  # ~15 WPM
_CW_EDGE_S = 0.005  # keying rise/fall time (no key clicks)

# The station with a program worth listening to: a looping tune on the
# start-up frequency (settings_store.DEFAULT_FREQUENCY_HZ), the "FM
# Broadcast" preset and the demo's strongest station.
_MUSIC_STATION_HZ = 100.1e6
_MUSIC_SEED = next(i for i, s in enumerate(_DEMO_SIGNALS) if s[0] == _MUSIC_STATION_HZ)

# The tune: four bars of I-vi-IV-V in C major, each an eighth-note
# arpeggio (MIDI notes) over a bass note and a soft pad chord.
_MUSIC_BARS: Tuple[Tuple[Tuple[int, ...], int, Tuple[int, ...]], ...] = (
    ((72, 76, 79, 84, 88, 84, 79, 76), 48, (60, 64, 67)),  # C
    ((69, 72, 76, 81, 84, 81, 76, 72), 45, (57, 60, 64)),  # Am
    ((65, 69, 72, 77, 81, 77, 72, 69), 41, (53, 57, 60)),  # F
    ((67, 71, 74, 79, 83, 79, 74, 71), 43, (55, 59, 62)),  # G
)
_MUSIC_STEP_S = 0.25  # one arpeggio note (120 BPM eighth notes)

# Broadcast FM composite, as fractions of the 75 kHz peak deviation: the
# 19 kHz stereo pilot and the 57 kHz RDS subcarrier; program audio (L+R)
# and the L-R subcarrier at 38 kHz share the rest.
_WFM_DEVIATION_HZ = 75e3
_PILOT_HZ = 19e3
_PILOT_SHARE = 0.09
_RDS_SHARE = 0.04
_PROGRAM_SHARE = 1.0 - _PILOT_SHARE - _RDS_SHARE
# Broadcast FM pre-emphasis time constant (Americas and Korea): the
# station boosts the treble of its program audio before modulating, and the
# receiver's matching 75 us de-emphasis restores the balance (and cuts the
# hiss FM adds at high audio frequencies).
_EMPHASIS_TAU_S = 75e-6

# Modulation (audio, keying, data) is computed once per station as a loop at
# about this rate, then read back at the device rate by linear
# interpolation: far cheaper than synthesising every sample, and seamless
# because every loop is periodic.
_LOOP_RATE_HZ = 48e3
_TONE_LOOP_S = 4.0  # loops of tones (their pitch sweeps once per loop)
_LOOP_CACHE_SIZE = 24
_LOOP_CACHE: Dict[tuple, Any] = {}
_LOOP_LOCK = threading.Lock()
_RAMPS: Dict[int, np.ndarray] = {}

# Complex white noise (unit variance per component) that every demo device
# draws its noise floor from, at a random offset and phase each read.
_NOISE_POOL_SIZE = 1 << 20
_NOISE_POOL: Optional[np.ndarray] = None

# Carrier phase is built from BLOCK-sample tables (see MockDevice._phase).
_PHASE_BLOCK = 2048
# Reading this far behind real time (polling small blocks, or restarting
# after a pause), the demo skips ahead so timed signals keep wall-clock time.
_MAX_LAG_S = 0.5


def _noise_pool() -> np.ndarray:
    global _NOISE_POOL
    with _LOOP_LOCK:
        if _NOISE_POOL is None:
            rng = np.random.default_rng(0x5D12)
            _NOISE_POOL = rng.standard_normal(
                2 * _NOISE_POOL_SIZE, dtype=np.float32
            ).view(np.complex64)
    return _NOISE_POOL


def _ramp(factor: int, length: int) -> np.ndarray:
    """``(k % factor) / factor`` for ``k < length`` (interpolation weights)."""
    ramp = _RAMPS.get(factor)
    if ramp is None or len(ramp) < length:
        size = max(length, 1 << 16) + factor
        ramp = ((np.arange(size) % factor) / factor).astype(np.float32)
        _RAMPS[factor] = ramp
    return ramp


class _Loop:
    """A periodic real waveform stored at ``1/factor`` of the device rate."""

    __slots__ = ("values", "factor")

    def __init__(self, values: np.ndarray, factor: int):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.factor = int(factor)

    def read(self, pos: int, n: int) -> np.ndarray:
        """``n`` samples at the device rate from stream position ``pos``,
        linearly interpolated (float32)."""
        values, factor = self.values, self.factor
        first, skip = divmod(pos % (len(values) * factor), factor)
        count = (skip + n - 1) // factor + 2
        seg = values.take(np.arange(first, first + count), mode="wrap")
        if factor == 1:
            return seg[:n]
        out = np.repeat(np.diff(seg), factor)[skip : skip + n]
        out *= _ramp(factor, skip + n)[skip : skip + n]
        out += np.repeat(seg[:-1], factor)[skip : skip + n]
        return out


def _loop_timebase(
    rate: float, seconds: float, loop_hz: float = _LOOP_RATE_HZ
) -> Tuple[int, float, np.ndarray]:
    """``(factor, loop rate, sample times)`` for a loop of about ``seconds``
    at about ``loop_hz``."""
    factor = max(1, int(round(rate / loop_hz)))
    loop_rate = rate / factor
    size = max(2, int(round(seconds * loop_rate)))
    return factor, loop_rate, np.arange(size) / loop_rate


def _whole_cycles(freq: float, period: float) -> float:
    """``freq`` nudged to a whole number of cycles per loop (no seam)."""
    return max(1.0, round(freq * period)) / period


def _cos_cycles(cycles: np.ndarray) -> np.ndarray:
    """``cos(2 pi cycles)`` (float32: wrapped to one cycle first, it is
    exact enough and many times faster)."""
    return np.cos(((2 * np.pi) * (cycles - np.floor(cycles))).astype(np.float32))


def _tone(freq: float, t: np.ndarray, twist: float = 0.0) -> np.ndarray:
    """A steady tone with a whole number of cycles per loop."""
    period = len(t) * (t[1] - t[0])
    return _cos_cycles(_whole_cycles(freq, period) * t + twist / (2 * np.pi))


def _swept_tone(mean_hz: float, sweep_hz: float, t: np.ndarray, twist: float):
    """A tone whose pitch sweeps by +/- ``sweep_hz`` once per loop."""
    step = t[1] - t[0]
    period = len(t) * step
    sweep = np.sin(((2 * np.pi / period) * t + twist).astype(np.float32))
    freq = _whole_cycles(mean_hz, period) + sweep_hz * sweep.astype(float)
    return _cos_cycles(np.cumsum(freq) * step)


def _fm_phase(deviation_hz: np.ndarray, loop_rate: float) -> np.ndarray:
    """Phase (rad) of an FM modulation, closed over the loop and centred."""
    deviation_hz = deviation_hz - np.mean(deviation_hz)
    phase = (2 * np.pi / loop_rate) * np.cumsum(deviation_hz)
    return phase - np.mean(phase)


def _midi_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _pre_emphasis(signal: np.ndarray, rate: float) -> np.ndarray:
    """One loop of ``signal`` through 75 us FM pre-emphasis, ``1 + j 2 pi f
    tau``: the exact inverse of the receiver's one-pole de-emphasis.

    The filter is applied around the loop (by FFT, once per loop), so the
    loop stays seamless and reading it costs nothing extra.
    """
    spectrum = np.fft.rfft(np.asarray(signal, dtype=float))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / rate)
    spectrum *= 1.0 + (2j * np.pi * _EMPHASIS_TAU_S) * freqs
    return np.fft.irfft(spectrum, len(signal))


def _music_broadcast(
    loop_rate: float, size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(left, right, left_pre, right_pre)`` of the demo tune.

    ``left_pre``/``right_pre`` are the channels as broadcast, 75 us
    pre-emphasized and scaled so the louder one peaks at 1 (full
    deviation); ``left``/``right`` are the same channels as a listener hears
    them after the receiver's de-emphasis.
    """
    left, right = _music_tune(loop_rate, size)
    left_pre = _pre_emphasis(left, loop_rate)
    right_pre = _pre_emphasis(right, loop_rate)
    peak = max(float(np.max(np.abs(left_pre))), float(np.max(np.abs(right_pre))))
    scale = 1.0 / max(peak, 1e-9)
    return left * scale, right * scale, left_pre * scale, right_pre * scale


def _music_channels(loop_rate: float, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right channels of the demo tune as a listener hears them
    (after the receiver's 75 us de-emphasis), at the level the station
    broadcasts them: see :func:`_music_broadcast`."""
    left, right, _left_pre, _right_pre = _music_broadcast(loop_rate, size)
    return left, right


def _music_tune(loop_rate: float, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right channels of the demo tune, peak-normalized to 1."""
    left = np.zeros(size, dtype=np.float32)
    right = np.zeros(size, dtype=np.float32)
    envelopes: Dict[tuple, np.ndarray] = {}

    def envelope(length_s, attack_s, decay_s, release_s, gain) -> np.ndarray:
        key = (length_s, attack_s, decay_s, release_s, gain)
        env = envelopes.get(key)
        if env is None:
            t = np.arange(int(length_s * loop_rate)) / loop_rate
            env = gain * np.exp(-t / decay_s)
            rise = max(1, int(attack_s * loop_rate))
            env[:rise] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(rise) / rise)
            fall = max(1, int(release_s * loop_rate))
            env[-fall:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fall) / fall)
            env = envelopes[key] = env.astype(np.float32)
        return env

    def note(start_s, midi, env, pan):
        # A soft, slightly bright timbre: fundamental plus two harmonics.
        cycles = np.arange(len(env)) * (_midi_hz(midi) / loop_rate)
        phase = ((2 * np.pi) * (cycles - np.floor(cycles))).astype(np.float32)
        # sin x + 0.22 sin 2x + 0.06 sin 3x
        #   = sin x (1.18 + 0.44 cos x - 0.24 sin^2 x)
        s1 = np.sin(phase)
        wave = np.cos(phase)
        wave *= 0.44
        wave += 1.18
        wave -= 0.24 * s1 * s1
        wave *= s1
        wave *= env
        # Notes that ring past the end of the loop wrap round to its start.
        first = int(round(start_s * loop_rate)) % size
        head = min(len(wave), size - first)
        for channel, gain in ((left, pan[0]), (right, pan[1])):
            channel[first : first + head] += gain * wave[:head]
            channel[: len(wave) - head] += gain * wave[head:]

    bar_s = _MUSIC_STEP_S * 8
    pluck = envelope(1.2, 0.005, 0.45, 0.15, 0.30)
    bass_env = envelope(bar_s + 0.1, 0.02, 1.6, 0.25, 0.42)
    pad_env = envelope(bar_s + 0.3, 0.35, 8.0, 0.4, 0.07)
    for bar, (arpeggio, bass, pad) in enumerate(_MUSIC_BARS):
        start = bar * bar_s
        for step, midi in enumerate(arpeggio):
            note(start + step * _MUSIC_STEP_S, midi, pluck, (0.55, 1.0))
        note(start, bass, bass_env, (0.85, 0.85))
        for midi in pad:
            note(start - 0.1, midi, pad_env, (1.0, 0.45))
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-9)
    return left / peak, right / peak


def _build_loops(kind: str, variant: Any, rate: float):
    """The modulation loop(s) for one kind of demo signal (see _loops)."""
    if kind == "wfm":
        # (program phase, L-R subcarrier amplitude as a phase multiplier)
        if variant == "music":
            # The tune as broadcast: 75 us pre-emphasized, like the
            # receiver's broadcast FM de-emphasis expects.
            factor, loop_rate, t = _loop_timebase(
                rate, _MUSIC_STEP_S * 8 * len(_MUSIC_BARS)
            )
            *_heard, left, right = _music_broadcast(loop_rate, len(t))
            mono = (left + right) * (_PROGRAM_SHARE / 2)
            diff = (left - right) * (_PROGRAM_SHARE / 2)
        else:
            # Three tones for the program and a 2.5 kHz L-R tone.
            factor, loop_rate, t = _loop_timebase(rate, _TONE_LOOP_S)
            twist = variant * 2.1
            mono = 0.30 * _swept_tone(1.5e3, 400.0, t, twist)
            mono += 0.18 * _tone(3.7e3, t, twist)
            mono += 0.12 * _swept_tone(6.9e3, -1e3, t, twist)
            diff = 0.16 * _tone(2.5e3, t)
        # L-R rides on 2 x pilot: its phase is S(t) sin(2 theta) * dev / 38k.
        diff *= _WFM_DEVIATION_HZ / (2 * _PILOT_HZ)
        return (
            _Loop(_fm_phase(_WFM_DEVIATION_HZ * mono.astype(float), loop_rate), factor),
            _Loop(diff, factor),
        )
    if kind == "nfm":
        # Voice-like FM, +/-3 kHz peak deviation.
        factor, loop_rate, t = _loop_timebase(rate, _TONE_LOOP_S)
        twist = variant * 2.1
        deviation = 2.2e3 * _swept_tone(820.0, 200.0, t, twist)
        deviation += 1.2e3 * _swept_tone(1650.0, -250.0, t, twist)
        return _Loop(_fm_phase(deviation, loop_rate), factor)
    if kind == "am":
        factor, loop_rate, t = _loop_timebase(rate, _TONE_LOOP_S)
        envelope = 1.0 + 0.35 * _swept_tone(600.0, 150.0, t, variant * 2.1)
        envelope += 0.2 * _tone(1700.0, t, variant)
        return _Loop(envelope, factor)
    if kind == "burst":
        # 2-FSK data at 9.6 kbit/s, +/-20 kHz shift; as many ones as zeros.
        factor, loop_rate, t = _loop_timebase(rate, 0.5)
        count = int(np.ceil(len(t) * 9600 / loop_rate))
        bits = np.resize([1.0, -1.0], count)
        np.random.default_rng(variant).shuffle(bits)
        symbol = (np.arange(len(t)) * 9600 / loop_rate).astype(int)
        return _Loop(_fm_phase(20e3 * bits[symbol], loop_rate), factor)
    if kind == "cw":
        # Only the keying envelope: a low loop rate is plenty.
        factor, loop_rate, t = _loop_timebase(
            rate, len(_CW_KEYING) * _CW_DIT_S, _LOOP_RATE_HZ / 10
        )
        keyed = np.asarray(_CW_KEYING, dtype=float)[
            np.minimum((t / _CW_DIT_S).astype(int), len(_CW_KEYING) - 1)
        ]
        # Two passes of a circular moving average: S-shaped key edges.
        width = max(1, int(_CW_EDGE_S * loop_rate))
        for _ in range(2):
            sums = np.cumsum(np.concatenate((keyed[-width:], keyed)))
            keyed = (sums[width:] - sums[:-width]) / width
        return _Loop(keyed, factor)
    if kind == "modes":
        return _other_replies(rate)
    return None


def _loops(kind: str, seed: int, rate: float):
    """Cached modulation loop(s) of a demo signal at ``rate``."""
    if kind == "wfm" and seed == _MUSIC_SEED:
        variant: Any = "music"
    elif kind in ("cw", "modes"):
        variant = 0
    else:
        variant = seed % 3
    key = (kind, variant, float(rate))
    # The scanner reads from its own thread.
    with _LOOP_LOCK:
        loops = _LOOP_CACHE.get(key)
        if loops is None:
            loops = _build_loops(kind, variant, rate)
            while len(_LOOP_CACHE) >= _LOOP_CACHE_SIZE:
                _LOOP_CACHE.pop(next(iter(_LOOP_CACHE)))
            _LOOP_CACHE[key] = loops
    return loops


# --------------------------------------------------------------------------- #
# ADS-B: Mode S replies on 1090 MHz
# --------------------------------------------------------------------------- #
# 1090 MHz carries real Mode S frames, so the Decoder panel's ADS-B decoder
# has an aircraft to follow: an airliner whose transponder squitters its
# identification, position and velocity (DF17 extended squitters) and
# answers with its squawk (DF5), over weaker replies from other aircraft
# that no decoder can use (Mode A/C replies, and Mode S replies whose parity
# carries addresses never heard in a squitter).
_MODES_CRC_POLY = 0x1FFF409
_CPR_SCALE = 1 << 17  # 17-bit CPR coordinates
_M_PER_DEGREE = 111_195.0  # metres per degree of latitude
_CALLSIGN_CHARS = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

# The airliner: its identification squitter is the classic example frame
# 8D4840D6202CC371C32CE0576098 ("The 1090 MHz Riddle"), and it cruises
# west at FL380 from where that book's position example puts it, over the
# North Sea. The flight repeats every _FLIGHT_LOOP_S.
_AIRCRAFT_ICAO = 0x4840D6
_AIRCRAFT_CALLSIGN = "KLM1023"
_AIRCRAFT_SQUAWK = "3472"
_AIRCRAFT_ALTITUDE_FT = 38000
_AIRCRAFT_START = (52.2572, 3.9194)  # degrees north, east
_AIRCRAFT_TRACK_DEG = 265.0
_AIRCRAFT_SPEED_KT = 450.0
_FLIGHT_LOOP_S = 1200.0

# What it sends in each _SQUITTER_CYCLE_S: (time s, message). Each one is
# moved by up to +/- _SQUITTER_JITTER_S, as transponders randomize their
# squitter timing (which also puts its pulses at every phase of a
# receiver's sample clock).
_SQUITTER_CYCLE_S = 2.0
_SQUITTERS: Tuple[Tuple[float, str], ...] = (
    (0.10, "ident"),
    (0.35, "even"),
    (0.60, "velocity"),
    (0.85, "odd"),
    (1.10, "squawk"),
    (1.35, "even"),
    (1.60, "velocity"),
    (1.85, "odd"),
)
_SQUITTER_JITTER_S = 0.04
_LONGEST_REPLY_S = 120e-6  # 8 us preamble + 112 bits

# Other traffic: replies per second, the loop they repeat in (samples are
# made once per sample rate), and their levels relative to the airliner.
_REPLIES_PER_S = 300
_REPLY_LOOP_S = 3.1
_REPLY_DB = (-30.0, -6.0)


def _crc24(data: bytes) -> int:
    """Mode S parity (CRC-24, generator 0x1FFF409) of ``data``."""
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= _MODES_CRC_POLY
    return crc


def _extended_squitter(icao: int, me: int) -> bytes:
    """DF17 frame (capability 5: airborne) carrying the 56-bit ME field."""
    head = bytes([17 << 3 | 5]) + icao.to_bytes(3, "big") + me.to_bytes(7, "big")
    return head + _crc24(head).to_bytes(3, "big")


def _address_parity_reply(word: int, icao: int, long: bool = False) -> bytes:
    """A reply whose parity is its CRC XOR the aircraft address (DF0/4/5
    short, DF16/20/21 long); ``word`` is everything before the parity."""
    head = word.to_bytes(11 if long else 4, "big")
    return head + (_crc24(head) ^ icao).to_bytes(3, "big")


def _ident_me(callsign: str) -> int:
    """Aircraft identification (TC 4), eight 6-bit characters."""
    me = 4 << 3
    for char in callsign.upper().ljust(8)[:8]:
        index = _CALLSIGN_CHARS.find(char) if char != "#" else -1
        me = me << 6 | (index if index > 0 else 32)  # else a space
    return me


def _altitude_code(feet: int) -> int:
    """12-bit altitude field in 25 ft steps (Q bit set)."""
    n = (int(feet) + 1000) // 25
    return (n >> 4) << 5 | 0x10 | (n & 0x0F)


def _cpr_nl(lat: float) -> int:
    """Number of CPR longitude zones at latitude ``lat``."""
    if abs(lat) >= 87.0:
        return 1
    a = 1.0 - math.cos(math.pi / 30.0)
    return int(
        math.floor(
            2.0 * math.pi / math.acos(1.0 - a / math.cos(math.radians(lat)) ** 2)
        )
    )


def _position_me(lat: float, lon: float, feet: int, odd: bool) -> int:
    """Airborne position (TC 11) as an even or odd CPR frame."""
    i = int(odd)
    dlat = 360.0 / (60 - i)
    yz = math.floor(_CPR_SCALE * (lat % dlat) / dlat + 0.5)
    rlat = dlat * (yz / _CPR_SCALE + math.floor(lat / dlat))
    dlon = 360.0 / max(_cpr_nl(rlat) - i, 1)
    xz = math.floor(_CPR_SCALE * (lon % dlon) / dlon + 0.5)
    return (
        11 << 51
        | _altitude_code(feet) << 36
        | i << 34
        | (yz % _CPR_SCALE) << 17
        | (xz % _CPR_SCALE)
    )


def _velocity_me(speed_kt: float, track_deg: float) -> int:
    """Airborne velocity (TC 19, subtype 1: subsonic ground speed), level."""
    east = round(speed_kt * math.sin(math.radians(track_deg)))
    north = round(speed_kt * math.cos(math.radians(track_deg)))
    return (
        19 << 51
        | 1 << 48
        | int(east < 0) << 42
        | (abs(east) + 1) << 32
        | int(north < 0) << 31
        | (abs(north) + 1) << 21
        | 1 << 10  # vertical rate 0 ft/min
    )


def _identity_code(squawk: str) -> int:
    """13-bit identity field of a 4-digit (octal) squawk code, in the order
    C1 A1 C2 A2 C4 A4 X B1 D1 B2 D2 B4 D4."""
    a, b, c, d = (int(digit, 8) for digit in squawk)
    code = 0
    for digit, weight in (
        (c, 1), (a, 1), (c, 2), (a, 2), (c, 4), (a, 4), (0, 0),
        (b, 1), (d, 1), (b, 2), (d, 2), (b, 4), (d, 4),
    ):  # fmt: skip
        code = code << 1 | int(bool(digit & weight))
    return code


def _aircraft_position(t: float) -> Tuple[float, float]:
    """The airliner's latitude and longitude ``t`` s into the stream."""
    flown = (t % _FLIGHT_LOOP_S) * _AIRCRAFT_SPEED_KT * 1852.0 / 3600.0
    track = math.radians(_AIRCRAFT_TRACK_DEG)
    lat = _AIRCRAFT_START[0] + flown * math.cos(track) / _M_PER_DEGREE
    east = flown * math.sin(track) / (_M_PER_DEGREE * math.cos(math.radians(lat)))
    return lat, _AIRCRAFT_START[1] + east


def _aircraft_frame(message: str, t: float) -> bytes:
    """The airliner's ``message`` (see _SQUITTERS) sent at ``t`` s."""
    icao = _AIRCRAFT_ICAO
    if message == "ident":
        return _extended_squitter(icao, _ident_me(_AIRCRAFT_CALLSIGN))
    if message == "velocity":
        me = _velocity_me(_AIRCRAFT_SPEED_KT, _AIRCRAFT_TRACK_DEG)
        return _extended_squitter(icao, me)
    if message == "squawk":
        # DF5 surveillance identity reply, airborne (FS 0).
        return _address_parity_reply(5 << 27 | _identity_code(_AIRCRAFT_SQUAWK), icao)
    lat, lon = _aircraft_position(t)
    me = _position_me(lat, lon, _AIRCRAFT_ALTITUDE_FT, message == "odd")
    return _extended_squitter(icao, me)


def _chip_pulses(chips: np.ndarray, chip_us: float) -> np.ndarray:
    """``(start us, width us)`` rows of the runs of 1s in ``chips``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], chips, [0]))))
    starts, ends = edges[::2], edges[1::2]
    return np.column_stack((starts, ends - starts)) * chip_us


def _modes_pulses(frame: bytes) -> np.ndarray:
    """Pulses of a Mode S reply: the 8 us preamble (pulses at 0, 1, 3.5 and
    4.5 us), then 1 us per bit, pulse-position modulated: a 0.5 us pulse in
    the first half of the bit for a 1, in the second half for a 0."""
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8)).astype(np.int8)
    chips = np.zeros(16 + 2 * len(bits), dtype=np.int8)
    chips[[0, 2, 7, 9]] = 1
    chips[16::2] = bits
    chips[17::2] = 1 - bits
    return _chip_pulses(chips, 0.5)


def _mode_ac_pulses(code: int) -> np.ndarray:
    """Pulses of a Mode A/C reply: framing pulses 20.3 us apart with the 12
    code pulses between them, 1.45 us apart (the middle, X, left empty),
    each 0.45 us long."""
    slots = [0, 14] + [
        slot
        for bit, slot in enumerate((1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13))
        if code >> bit & 1
    ]
    starts = np.sort(np.asarray(slots, dtype=float)) * 1.45
    return np.column_stack((starts, np.full(len(starts), 0.45)))


def _pulse_envelope(
    pulses: np.ndarray, start: float, rate: float
) -> Tuple[int, np.ndarray]:
    """First sample and envelope (float32, 0 to 1) of ``pulses`` sent from
    stream position ``start`` (samples, fractional).

    Each sample holds the fraction of its period the pulses cover, as an
    integrating ADC sees them.
    """
    per_us = rate * 1e-6
    begin = pulses[:, 0]
    covered = np.cumsum(pulses[:, 1])
    first = int(math.floor(start))
    last = int(math.ceil(start + (begin[-1] + pulses[-1, 1]) * per_us))
    edges = (np.arange(first, last + 1) - start) / per_us
    knots = np.column_stack((begin, begin + pulses[:, 1])).ravel()
    area = np.column_stack((covered - pulses[:, 1], covered)).ravel()
    on = np.interp(edges, knots, area)
    return first, (np.diff(on) * per_us).astype(np.float32)


def _squitter_jitter(cycle: int, slot: int) -> float:
    """Timing offset (s) of one of the airliner's transmissions."""
    h = (cycle * 2654435761 + slot * 40503 + 0x5D12) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return (h / 0xFFFFFFFF * 2.0 - 1.0) * _SQUITTER_JITTER_S


def _squitter_spans(pos: int, n: int, rate: float) -> List[tuple]:
    """``(a, b, envelope)`` of the airliner's transmissions overlapping the
    ``n`` samples from stream position ``pos`` (``a:b`` in the block)."""
    t0, t1 = pos / rate, (pos + n) / rate
    cycle = _SQUITTER_CYCLE_S
    first = math.floor((t0 - _SQUITTER_JITTER_S - _LONGEST_REPLY_S) / cycle)
    last = math.floor((t1 + _SQUITTER_JITTER_S) / cycle)
    spans = []
    for c in range(first, last + 1):
        for slot, (offset, message) in enumerate(_SQUITTERS):
            t = c * cycle + offset + _squitter_jitter(c, slot)
            if t >= t1 or t + _LONGEST_REPLY_S <= t0:
                continue
            frame = _aircraft_frame(message, t)
            start, env = _pulse_envelope(_modes_pulses(frame), t * rate, rate)
            a, b = max(start, pos), min(start + len(env), pos + n)
            if b > a:
                spans.append((a - pos, b - pos, env[a - start : b - start]))
    return spans


class _Bursts:
    """Short transmissions at fixed places in a loop of ``period`` samples,
    stored sparsely (the loop is mostly silence)."""

    __slots__ = ("period", "starts", "ends", "offsets", "values", "longest")

    def __init__(self, period: int, bursts: List[Tuple[int, np.ndarray]]):
        bursts = sorted(bursts, key=lambda burst: burst[0])
        lengths = np.array([len(env) for _start, env in bursts], dtype=np.int64)
        self.period = int(period)
        self.starts = np.array([start for start, _env in bursts], dtype=np.int64)
        self.ends = self.starts + lengths
        self.offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        self.values = np.concatenate(
            [env for _start, env in bursts] or [np.zeros(0, dtype=np.float32)]
        )
        self.longest = int(lengths.max()) if len(lengths) else 0

    def spans(self, pos: int, n: int) -> List[tuple]:
        """``(a, b, envelope)`` of the bursts overlapping the ``n`` samples
        from stream position ``pos`` (``a:b`` in the block)."""
        spans = []
        end = pos + n
        base = (pos // self.period) * self.period
        while base < end:
            lo, hi = pos - base, end - base
            first = int(np.searchsorted(self.starts, lo - self.longest))
            last = int(np.searchsorted(self.starts, hi))
            for i in range(first, last):
                start, stop = int(self.starts[i]), int(self.ends[i])
                a, b = max(start, lo), min(stop, hi)
                if b > a:
                    at = int(self.offsets[i]) - start
                    spans.append(
                        (base + a - pos, base + b - pos, self.values[at + a : at + b])
                    )
            base += self.period
        return spans


def _other_replies(rate: float) -> _Bursts:
    """A loop of replies from other aircraft at ``rate``: Mode A/C replies
    and Mode S replies to addresses that never squitter, so a decoder drops
    them all."""
    rng = np.random.default_rng(1090)
    addresses = [
        a for a in rng.integers(1, 1 << 24, 12).tolist() if a != _AIRCRAFT_ICAO
    ]
    count = int(_REPLIES_PER_S * _REPLY_LOOP_S)
    times = np.sort(rng.uniform(0.0, _REPLY_LOOP_S - 2 * _LONGEST_REPLY_S, count))
    bursts = []
    for t in times.tolist():
        if rng.random() < 0.5:
            pulses = _mode_ac_pulses(int(rng.integers(0, 1 << 12)))
        else:
            df = int(rng.choice((0, 4, 5, 16, 20, 21)))
            long = df >= 16
            bits = 88 if long else 32
            payload = int.from_bytes(rng.bytes(11), "big") & ((1 << (bits - 5)) - 1)
            frame = _address_parity_reply(
                df << (bits - 5) | payload, int(rng.choice(addresses)), long
            )
            pulses = _modes_pulses(frame)
        start, env = _pulse_envelope(pulses, t * rate, rate)
        env *= np.float32(10.0 ** (rng.uniform(*_REPLY_DB) / 20.0))
        bursts.append((start, env))
    return _Bursts(int(round(_REPLY_LOOP_S * rate)), bursts)


def _on_ranges(
    duty: Optional[Tuple[float, float]], seed: int, t0: float, n: int, rate: float
) -> List[Tuple[int, int]]:
    """Sample ranges of a block starting at ``t0`` s in which a signal with
    ``duty`` = (period s, on s) is on (the whole block when ``None``)."""
    if duty is None:
        return [(0, n)]
    period, on = duty
    shift = (seed * 1.7) % period
    end = t0 + n / rate
    ranges = []
    cycle = math.floor((t0 + shift) / period)
    while True:
        start = cycle * period - shift
        if start >= end:
            return ranges
        a = max(0, int(math.ceil((start - t0) * rate)))
        b = min(n, int(math.ceil((start + on - t0) * rate)))
        if b > a:
            ranges.append((a, b))
        cycle += 1


class MockDevice:
    """Simulated receiver used by demo mode and the Demo Device.

    Generates a band of plausible signals at fixed frequencies (FM broadcast
    stations, airband AM, 2 m and 70 cm FM, NOAA weather, ISM bursts, a CW
    beacon, Mode S and Mode A/C replies on 1090 MHz and a few weak carriers
    everywhere else) on top of a realistic noise floor. Signals move across
    the display as you tune, so click-to-tune and the frequency scanner
    behave as they would with real hardware. Raising the gain lifts signals and noise together and, as on a
    real receiver, too much gain clips the ADC. With automatic gain
    (``set_gain_mode(True)``) the device picks the gain itself, keeping the
    signals in the passband clear of clipping.

    The strongest station, 100.1 MHz, broadcasts a looping tune in stereo
    FM with the usual 75 us pre-emphasis, so listening to it in FM mode
    (with the receiver's matching de-emphasis) plays music; the others carry
    tones, voice-like modulation, keyed CW or data bursts. On 1090 MHz an
    airliner's transponder sends real ADS-B frames a few times a second
    (KLM1023, ICAO 4840D6: identification, position, velocity and a squawk
    reply) that the ADS-B decoder decodes, among weaker replies from other
    aircraft that it drops.

    The samples form one continuous stream: each read carries on where the
    last one ended, so demodulated audio has no clicks between reads, and
    reading at the sample rate plays it in real time. ``read_samples`` is
    fast enough for that: at most about 0.04 us per sample (about 3 ms for
    a 30 Hz frame's worth, 0.034 s, at 2.4 MS/s), because the modulation is
    precomputed as loops and the noise floor comes from a shared pool.
    """

    class MockInfo:
        name = "Demo Device"
        serial = "DEMO"

    # Noise floor per FFT bin (Hann window, 2048 points) at 20 dB gain, and
    # the fixed ADC noise underneath it, both in dBFS.
    _NOISE_FLOOR_DB = -88.0
    _ADC_NOISE_DB = -104.0
    _REF_GAIN_DB = 20.0
    # Automatic gain keeps the passband's combined peak this far below full
    # scale, within the gain range an RTL-SDR tuner offers.
    _AGC_HEADROOM_DB = 10.0
    _AGC_RANGE_DB = (0.0, 40.0)

    def __init__(self):
        self.info = self.MockInfo()
        self._frequency = float(DEFAULT_FREQUENCY_HZ)
        self._sample_rate = 2.4e6
        self._gain = 20
        self._auto_gain = False
        self._running = False
        self._rng = np.random.default_rng()
        # Stream position (samples) of the next sample, the rate it counts
        # in, and the wall-clock time of position 0.
        self._stream_pos = 0
        self._stream_rate = 0.0
        self._stream_epoch: Optional[float] = None
        self._visible_key: Optional[Tuple[float, float]] = None
        self._visible: List[tuple] = []
        self._tables: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}

    # -- configuration (same API as the hardware drivers) ------------------ #
    def set_frequency(self, freq: float) -> bool:
        self._frequency = float(freq)
        return True

    def set_sample_rate(self, rate: float) -> bool:
        self._sample_rate = float(rate)
        return True

    def set_gain(self, gain: float) -> bool:
        """Set the manual gain (dB). With automatic gain on, it takes effect
        when automatic gain is switched off again."""
        self._gain = gain
        return True

    def set_gain_mode(self, auto: bool) -> bool:
        """Switch automatic gain (AGC) on or off, like the hardware drivers.

        With AGC on the gain follows what is in the passband, so the
        strongest signals stay about 10 dB below full scale and never clip;
        switching it off returns to the manual gain from :meth:`set_gain`.
        """
        self._auto_gain = bool(auto)
        return True

    def set_bandwidth(self, bw: float) -> bool:
        return True

    @property
    def frequency(self) -> float:
        """Current center frequency in Hz."""
        return self._frequency

    @property
    def sample_rate(self) -> float:
        """Current sample rate in samples/second."""
        return self._sample_rate

    @property
    def gain(self) -> float:
        """Current gain in dB (the automatic gain's choice while it is on)."""
        return self._gain_db(float(self._frequency), float(self._sample_rate) or 2.4e6)

    @property
    def gain_mode(self) -> str:
        """``"auto"`` while automatic gain is on, else ``"manual"``."""
        return "auto" if self._auto_gain else "manual"

    def _gain_db(self, center: float, rate: float) -> float:
        """The gain in effect: manual, or chosen for the current passband."""
        if not self._auto_gain:
            return float(self._gain)
        # Worst case, every signal peaks at once: keep that sum (plus the
        # noise peaks) the headroom below the ADC's full scale.
        peak = sum(
            amp
            for _offset, _kind, amp, _duty, _seed in self._visible_signals(center, rate)
        )
        peak += 4.0 * math.sqrt(2.0) * self._noise_sigma(self._NOISE_FLOOR_DB)
        gain = self._REF_GAIN_DB - self._AGC_HEADROOM_DB - 20.0 * math.log10(peak)
        low, high = self._AGC_RANGE_DB
        return float(min(max(gain, low), high))

    @property
    def is_streaming(self) -> bool:
        """True between ``start_rx()`` and ``stop_rx()``."""
        return self._running

    def start_rx(self) -> bool:
        self._running = True
        return True

    def stop_rx(self) -> bool:
        self._running = False
        return True

    # -- signal generation ------------------------------------------------- #
    def _visible_signals(self, center: float, rate: float) -> List[tuple]:
        """Signals inside the current passband (cached per tuning)."""
        key = (center, rate)
        if key == self._visible_key:
            return self._visible
        half = rate / 2.0
        signals = [
            (freq - center, kind, level, duty, idx)
            for idx, (freq, kind, level, duty) in enumerate(_DEMO_SIGNALS)
            if abs(freq - center) < half
        ]
        signals.extend(self._background_carriers(center, half))
        visible = []
        for offset, kind, level, duty, seed in signals:
            # Anti-alias filter roll-off over the outer 10% of the passband.
            edge = (abs(offset) / half - 0.8) / 0.2
            rolloff = 1.0 if edge <= 0 else math.cos(min(edge, 1.0) * math.pi / 2)
            if rolloff < 1e-3:
                continue
            amp = 10.0 ** (level / 20.0) * rolloff
            visible.append((offset, kind, amp, duty, seed))
        self._visible, self._visible_key = visible, key
        self._tables = {}
        return visible

    @staticmethod
    def _background_carriers(center: float, half: float) -> List[tuple]:
        """Weak narrowband carriers scattered deterministically everywhere."""
        cell = 150e3
        out = []
        first = int(math.floor((center - half) / cell))
        last = int(math.ceil((center + half) / cell))
        for c in range(first, last + 1):
            h = (c * 2654435761) & 0xFFFFFFFF
            if h % 100 >= 18:  # ~18% of cells hold a carrier
                continue
            freq = c * cell + ((h >> 8) % 30) * 5e3
            if abs(freq - center) >= half:
                continue
            kind = ("cw_steady", "nfm", "am")[(h >> 16) % 3]
            level = -80.0 + ((h >> 20) % 14)
            out.append((freq - center, kind, level, None, 1000 + c))
        return out

    @staticmethod
    def _noise_sigma(bin_db: float) -> float:
        """Per-component sigma of complex white noise whose FFT bins
        (2048-point Hann) sit at ``bin_db`` dBFS:
        bin power = 2*sigma^2 * sum(w^2)/sum(w)^2."""
        bin_factor = 1.5 / 2048.0
        return math.sqrt(10.0 ** (bin_db / 10.0) / bin_factor / 2)

    def _advance(self, n: int, rate: float) -> int:
        """Stream position of the next ``n`` samples (and move past them)."""
        now = time.monotonic()
        if rate != self._stream_rate:
            if self._stream_rate:
                # Same point in time at the new rate.
                self._stream_pos = int(self._stream_pos * rate / self._stream_rate)
            self._stream_rate = rate
        if self._stream_epoch is None:
            self._stream_epoch = now - self._stream_pos / rate
        # Reading slower than real time: skip ahead, so timed signals follow
        # the clock. Reading faster (a scan) just runs ahead.
        behind = (now - self._stream_epoch) * rate - (self._stream_pos + n)
        if behind > _MAX_LAG_S * rate:
            self._stream_pos += int(behind)
        pos = self._stream_pos
        self._stream_pos += n
        return pos

    def _table(self, freq: float, rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Phase (float32) and phasor (complex64) of ``freq`` over one block."""
        key = (freq, rate)
        table = self._tables.get(key)
        if table is None:
            cycles = np.mod(np.arange(_PHASE_BLOCK) * (freq / rate), 1.0)
            phase = (2 * np.pi) * cycles
            table = (phase.astype(np.float32), np.exp(1j * phase).astype(np.complex64))
            self._tables[key] = table
        return table

    def _block_starts(self, freq: float, rate: float, pos: int, n: int) -> np.ndarray:
        """Phase (rad, float64) of ``freq`` at the start of each block."""
        starts = pos + _PHASE_BLOCK * np.arange(-(-n // _PHASE_BLOCK))
        return (2 * np.pi) * np.mod(starts * (freq / rate), 1.0)

    def _phase(self, freq: float, rate: float, pos: int, n: int) -> np.ndarray:
        """Phase of a ``freq`` Hz carrier for ``n`` samples from ``pos``,
        in [0, 4 pi) (float32): continuous from one read to the next."""
        table = self._table(freq, rate)[0]
        starts = self._block_starts(freq, rate, pos, n).astype(np.float32)
        return (starts[:, None] + table[None, :]).ravel()[:n]

    def _carrier(
        self, freq: float, amp: float, rate: float, pos: int, n: int
    ) -> np.ndarray:
        """``amp * exp(j * phase)`` of a ``freq`` Hz carrier (complex64)."""
        table = self._table(freq, rate)[1]
        starts = (amp * np.exp(1j * self._block_starts(freq, rate, pos, n))).astype(
            np.complex64
        )
        return (starts[:, None] * table[None, :]).ravel()[:n]

    def _add_signal(self, out, kind, offset, amp, seed, rate, pos, stereo) -> None:
        """Add one signal to ``out`` (starting at stream position ``pos``).

        ``stereo`` gives the broadcast FM pilot/RDS phase and sin(2 theta)
        for the same samples.
        """
        n = len(out)
        if kind == "cw_steady":
            out += self._carrier(offset, amp, rate, pos, n)
            return
        if kind in ("am", "cw"):
            envelope = _loops(kind, seed, rate).read(pos, n)
            wave = self._carrier(offset, amp, rate, pos, n)
            wave *= envelope
            out += wave
            return
        if kind == "modes":
            # Short pulsed replies: only the samples they cover.
            spans = _loops(kind, seed, rate).spans(pos, n)
            spans += _squitter_spans(pos, n, rate)
            for a, b, envelope in spans:
                wave = self._carrier(offset, amp, rate, pos + a, b - a)
                wave *= envelope
                out[a:b] += wave
            return
        # Angle modulation: wfm, nfm, burst.
        loops = _loops(kind, seed, rate)
        phase = self._phase(offset, rate, pos, n)
        if kind == "wfm":
            program, subcarrier = loops
            pilot_rds, sin2 = stereo
            phase += program.read(pos, n)
            phase += pilot_rds
            diff = subcarrier.read(pos, n)
            diff *= sin2
            phase += diff
        else:
            phase += loops.read(pos, n)
        part = np.cos(phase)
        part *= amp
        out.real += part
        np.sin(phase, out=phase)
        phase *= amp
        out.imag += phase

    def _stereo_terms(self, rate: float, pos: int, n: int):
        """Broadcast FM pilot + RDS phase and sin(2 theta), theta the pilot's
        phase, shared by every FM station (they are locked to the stream)."""
        theta = self._phase(_PILOT_HZ, rate, pos, n)
        sin1 = np.sin(theta)
        sin2 = np.cos(theta)
        sin2 *= sin1
        sin2 *= 2.0
        # sin(3 theta) = sin(theta) * (3 - 4 sin^2(theta))
        sin3 = sin1 * sin1
        sin3 *= -4.0
        sin3 += 3.0
        sin3 *= sin1
        sin3 *= _RDS_SHARE * _WFM_DEVIATION_HZ / (3 * _PILOT_HZ)
        sin1 *= _PILOT_SHARE * _WFM_DEVIATION_HZ / _PILOT_HZ
        sin1 += sin3
        return sin1, sin2

    def _noise(self, n: int, sigma: float) -> np.ndarray:
        """``n`` samples of the noise floor (complex64)."""
        pool = _noise_pool()
        size = len(pool)
        out = np.empty(n, dtype=np.complex64)
        done = 0
        while done < n:
            start = int(self._rng.integers(size - min(n - done, size // 2) + 1))
            count = min(n - done, size - start)
            scale = np.complex64(sigma * np.exp(2j * np.pi * self._rng.random()))
            np.multiply(
                pool[start : start + count], scale, out=out[done : done + count]
            )
            done += count
        return out

    def read_samples(self, num_samples: int):
        if not self._running:
            return None
        n = int(num_samples)
        if n <= 0:
            return np.zeros(0, dtype=np.complex64)

        rate = float(self._sample_rate) or 2.4e6
        center = float(self._frequency)
        gain_lin = 10.0 ** ((self._gain_db(center, rate) - self._REF_GAIN_DB) / 20.0)
        pos = self._advance(n, rate)

        sigma_ant = self._noise_sigma(self._NOISE_FLOOR_DB)
        sigma = math.hypot(sigma_ant * gain_lin, self._noise_sigma(self._ADC_NOISE_DB))
        samples = self._noise(n, sigma)

        t0 = pos / rate
        stereo = None
        for offset, kind, amp, duty, seed in self._visible_signals(center, rate):
            for a, b in _on_ranges(duty, seed, t0, n, rate):
                if kind == "wfm" and stereo is None:
                    stereo = self._stereo_terms(rate, pos, n)
                terms = None if stereo is None else (stereo[0][a:b], stereo[1][a:b])
                self._add_signal(
                    samples[a:b],
                    kind,
                    offset,
                    amp * gain_lin,
                    seed,
                    rate,
                    pos + a,
                    terms,
                )

        # An 8-bit ADC clips at full scale: overdriving the gain distorts.
        flat = samples.view(np.float32)
        np.clip(flat, -1.0, 1.0, out=flat)
        return samples

    def close(self):
        self._running = False
