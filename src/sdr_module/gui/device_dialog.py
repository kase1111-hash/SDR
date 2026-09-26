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
# Levels are per-bin peaks on the main display's dBFS scale.
#
# Stations sit where the app sends people: the strongest one is on 100.1 MHz,
# the "FM Broadcast" preset of the welcome screen and Radio > Band Presets
# (and next to the 100 MHz start-up frequency), and the radio tuner's FM
# presets (88.5, 93.3, 97.1, 99.5, 101.1, 104.3 MHz) each find a station.
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
    # Narrowband carriers near the default 100 MHz view
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
    (1090.0e6, "pulse", -42.0, (0.5, 0.12)),
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


class MockDevice:
    """Simulated receiver used by demo mode and the Demo Device.

    Generates a band of plausible signals at fixed frequencies (FM broadcast
    stations, airband AM, 2 m and 70 cm FM, NOAA weather, ISM bursts, a CW
    beacon and a few weak carriers everywhere else) on top of a realistic
    noise floor. Signals move across the display as you tune, so
    click-to-tune and the frequency scanner behave as they would with real
    hardware. Raising the gain lifts signals and noise together and, as on a
    real receiver, too much gain clips the ADC. With automatic gain
    (``set_gain_mode(True)``) the device picks the gain itself, keeping the
    signals in the passband clear of clipping.

    ``read_samples`` is cheap (well under a millisecond for 2048 samples) so
    the display can poll it at 30 Hz.
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
        self._frequency = 100e6
        self._sample_rate = 2.4e6
        self._gain = 20
        self._auto_gain = False
        self._running = False
        self._rng = np.random.default_rng()
        self._sample_clock = 0  # samples generated so far (phase continuity)
        self._visible_key: Optional[Tuple[float, float]] = None
        self._visible: List[tuple] = []

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
            amp * (3.0 if kind == "pulse" else 1.0)
            for _offset, kind, amp, _duty, _seed in self._visible_signals(center, rate)
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

    @staticmethod
    def _active(duty: Optional[Tuple[float, float]], seed: int, now: float) -> bool:
        if duty is None:
            return True
        period, on = duty
        phase = (now + (seed * 1.7) % period) % period
        return phase < on

    def _modulated(self, kind, offset, amp, seed, t, now, n) -> Optional[np.ndarray]:
        """One signal's complex baseband contribution, or None when silent."""
        two_pi = 2.0 * np.pi
        carrier_phase = two_pi * offset * t
        # Slow per-signal audio variation, derived from wall-clock time.
        wobble = math.sin(now * 0.7 + seed) * 0.5 + 0.5

        if kind == "wfm":
            # Composite stereo baseband: program audio, 19 kHz pilot, L-R on a
            # 38 kHz subcarrier and RDS at 57 kHz; +/-75 kHz peak deviation.
            tones = (
                (1.1e3 + 800 * wobble, 0.30),
                (3.7e3, 0.18),
                (7.9e3 - 2e3 * wobble, 0.12),
                (19e3, 0.09),
                (38e3 - 2.5e3, 0.08),
                (38e3 + 2.5e3, 0.08),
                (57e3, 0.04),
            )
            phase = carrier_phase
            rs = seed * 0.37 + now * 3.1
            for i, (f_mod, weight) in enumerate(tones):
                phase = phase + (75e3 * weight / f_mod) * np.sin(
                    two_pi * f_mod * t + rs * (i + 1)
                )
            return amp * np.exp(1j * phase)

        if kind == "nfm":
            # Voice-like FM, +/-3 kHz deviation.
            f1, f2 = 620.0 + 400 * wobble, 1900.0 - 500 * wobble
            phase = (
                carrier_phase
                + (2.2e3 / f1) * np.sin(two_pi * f1 * t + now)
                + (1.2e3 / f2) * np.sin(two_pi * f2 * t + 2 * now)
            )
            return amp * np.exp(1j * phase)

        if kind == "am":
            f1, f2 = 450.0 + 300 * wobble, 1700.0
            envelope = (
                1.0
                + 0.35 * np.sin(two_pi * f1 * t + now)
                + 0.2 * np.sin(two_pi * f2 * t + 3 * now)
            )
            return amp * envelope * np.exp(1j * carrier_phase)

        if kind == "cw":
            unit = int(now / _CW_DIT_S) % len(_CW_KEYING)
            if not _CW_KEYING[unit]:
                return None
            return amp * np.exp(1j * carrier_phase)

        if kind == "cw_steady":
            return amp * np.exp(1j * carrier_phase)

        if kind == "burst":
            # 2-FSK data burst at 9.6 kbit/s, +/-20 kHz shift.
            bits_per_block = max(1, int(n * 9600 / self._sample_rate) + 1)
            bits = self._rng.integers(0, 2, bits_per_block) * 2 - 1
            idx = (np.arange(n) * bits_per_block) // n
            inst = offset + 20e3 * bits[idx]
            phase = two_pi * np.cumsum(inst) / self._sample_rate
            return amp * np.exp(1j * phase)

        if kind == "pulse":
            # ADS-B-like 1 us pulse-position bursts: wide, short and spiky.
            chips = self._rng.random(n) < 0.35
            return amp * 3.0 * chips * np.exp(1j * carrier_phase)

        return None

    def read_samples(self, num_samples: int):
        if not self._running:
            return None
        n = int(num_samples)
        if n <= 0:
            return np.zeros(0, dtype=np.complex64)

        rate = float(self._sample_rate) or 2.4e6
        center = float(self._frequency)
        gain_lin = 10.0 ** ((self._gain_db(center, rate) - self._REF_GAIN_DB) / 20.0)
        now = time.monotonic()

        sigma_ant = self._noise_sigma(self._NOISE_FLOOR_DB)
        sigma = math.hypot(sigma_ant * gain_lin, self._noise_sigma(self._ADC_NOISE_DB))
        samples = self._rng.standard_normal(2 * n).view(np.complex128) * sigma

        t = (self._sample_clock + np.arange(n)) / rate
        self._sample_clock = (self._sample_clock + n) % (1 << 40)
        for offset, kind, amp, duty, seed in self._visible_signals(center, rate):
            if not self._active(duty, seed, now):
                continue
            wave = self._modulated(kind, offset, amp * gain_lin, seed, t, now, n)
            if wave is not None:
                samples += wave

        # An 8-bit ADC clips at full scale: overdriving the gain distorts.
        np.clip(samples.real, -1.0, 1.0, out=samples.real)
        np.clip(samples.imag, -1.0, 1.0, out=samples.imag)
        return samples.astype(np.complex64)

    def close(self):
        self._running = False
