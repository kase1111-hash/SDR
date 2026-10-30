"""Tests for the GUI dialogs: device, scanner, first-run, help and errors."""

import logging
import os
import re
import time
import unittest

import numpy as np

try:
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance() or QApplication([])
except ImportError as _import_error:  # pragma: no cover - PyQt6 missing
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)

GUI_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "src", "sdr_module", "gui")
DIALOG_FILES = (
    "device_dialog.py",
    "scanner_dialog.py",
    "first_run_wizard.py",
    "help_dialog.py",
    "error_log_dialog.py",
)


def require_pyqt6(test_case: unittest.TestCase) -> None:
    if HAS_PYQT6:
        return
    if os.environ.get("SDR_REQUIRE_GUI"):
        test_case.fail(f"PyQt6 is required but failed to import: {PYQT6_IMPORT_ERROR}")
    test_case.skipTest("PyQt6 not available")


def _spectrum_db(samples):
    n = len(samples)
    window = np.hanning(n + 1)[:-1]
    spec = np.fft.fftshift(np.fft.fft(samples * window))
    return 20 * np.log10(np.abs(spec) / window.sum() + 1e-12)


class TestNoHardCodedColors(unittest.TestCase):
    """Dialogs take every color from the theme palette."""

    PATTERN = re.compile(
        r"#[0-9a-fA-F]{3,8}\b|rgba?\(|"
        r"\b(gray|grey|green|red|orange|blue|yellow|white|black)\b\s*[;\"']"
    )

    def test_dialog_sources_have_no_literal_colors(self):
        for name in DIALOG_FILES:
            with open(os.path.join(GUI_DIR, name), encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    self.assertIsNone(
                        self.PATTERN.search(line),
                        f"{name}:{lineno} hard-codes a color: {line.strip()}",
                    )


class TestMockDevice(unittest.TestCase):
    """The demo device produces a believable band, quickly."""

    def setUp(self):
        from sdr_module.gui.device_dialog import MockDevice

        self.dev = MockDevice()

    def test_no_samples_until_started(self):
        self.assertIsNone(self.dev.read_samples(2048))
        self.dev.start_rx()
        samples = self.dev.read_samples(2048)
        self.assertEqual(samples.shape, (2048,))
        self.assertEqual(samples.dtype, np.complex64)

    def test_carriers_stand_well_above_noise_floor(self):
        self.dev.start_rx()
        power = np.mean(
            [10 ** (_spectrum_db(self.dev.read_samples(2048)) / 10) for _ in range(8)],
            axis=0,
        )
        db = 10 * np.log10(power)
        floor = float(np.median(db))
        self.assertLess(floor, -80.0)  # a realistic, low noise floor
        self.assertGreater(float(db.max()) - floor, 40.0)  # a strong station

    def test_signals_stay_put_when_tuning(self):
        """A station at 100.1 MHz moves to the center when tuned to it."""
        self.dev.start_rx()
        self.dev.set_frequency(100.1e6)
        power = np.mean(
            [10 ** (_spectrum_db(self.dev.read_samples(2048)) / 10) for _ in range(8)],
            axis=0,
        )
        peak_bin = int(np.argmax(power))
        offset_hz = (peak_bin - 1024) * self.dev.sample_rate / 2048
        self.assertLess(abs(offset_hz), 120e3)
        self.assertEqual(self.dev.frequency, 100.1e6)

    def test_read_samples_is_fast(self):
        self.dev.start_rx()
        self.dev.read_samples(2048)
        start = time.perf_counter()
        for _ in range(30):
            self.dev.read_samples(2048)
        per_call = (time.perf_counter() - start) / 30
        self.assertLess(per_call, 0.01)  # polled at 30 Hz

    def test_excess_gain_clips_like_an_adc(self):
        self.dev.start_rx()
        self.dev.set_gain(60)
        samples = self.dev.read_samples(2048)
        self.assertLessEqual(float(np.max(np.abs(samples.real))), 1.0)


class TestDeviceDialog(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import device_dialog

        self.mod = device_dialog
        self.dialog = device_dialog.DeviceDialog()

    def test_demo_row_is_listed_and_preselected(self):
        table = self.dialog._device_table
        self.assertEqual(table.rowCount(), len(self.dialog._devices))
        self.assertEqual(self.dialog._devices[-1]["type"], "demo")
        self.assertTrue(table.selectionModel().selectedRows())
        self.assertTrue(self.dialog._connect_btn.isEnabled())
        self.assertTrue(self.dialog._connect_btn.isDefault())

    def test_demo_connect_applies_sample_rate(self):
        combo = self.dialog._rate_combo
        combo.setCurrentIndex(combo.findData(2.0e6))
        device = self.dialog._open_device({"type": "demo"})
        self.assertIsInstance(device, self.mod.MockDevice)
        self.assertEqual(device.sample_rate, 2.0e6)

    def test_failed_hardware_open_is_not_disguised_as_demo(self):
        class Unplugged:
            def open(self, index=0):
                return False

        self.dialog._device_class = lambda _t: Unplugged
        self.assertIsNone(self.dialog._open_device({"type": "rtlsdr", "index": 0}))

    def test_enumerates_hardware_through_list_devices(self):
        from sdr_module.devices.base import DeviceInfo
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        info = DeviceInfo("RTL-SDR #0", "00000001", "RTL-SDR Blog", "RTL2832U", 0)
        orig_installed = self.mod._driver_installed
        orig_list = RTLSDRDevice.list_devices
        try:
            self.mod._driver_installed = lambda name: name == "rtlsdr"
            RTLSDRDevice.list_devices = staticmethod(lambda: [info])
            self.dialog._refresh_devices()
        finally:
            self.mod._driver_installed = orig_installed
            RTLSDRDevice.list_devices = orig_list
        self.assertEqual(self.dialog._devices[0]["type"], "rtlsdr")
        self.assertEqual(self.dialog._device_table.item(0, 2).text(), "00000001")
        self.assertIn("Found 1 SDR device", self.dialog._status_label.text())
        # RTL-SDR rates stop at 2.56 MS/s.
        rates = [
            self.dialog._rate_combo.itemData(i)
            for i in range(self.dialog._rate_combo.count())
        ]
        self.assertLessEqual(max(rates), 2.56e6)

    def test_unsupported_settings_are_disabled_with_reason(self):
        self.assertFalse(self.dialog._ppm_spin.isEnabled())
        self.assertTrue(self.dialog._ppm_spin.toolTip())
        self.assertFalse(self.dialog._direct_combo.isEnabled())
        self.assertTrue(self.dialog._direct_combo.toolTip())


def _scanner_panel(device_range=None, blocker=("", ""), receiving=True):
    """A Scanner tab over a demo device (its host is a small fake window)."""
    from sdr_module.gui.device_dialog import MockDevice
    from sdr_module.gui.scanner_panel import ScannerHost, ScannerPanel

    dev = MockDevice()
    dev.start_rx()

    class Host(ScannerHost):
        listened = []
        started = 0
        state = {"receiving": receiving, "blocker": blocker}

        def scan_tune_lo(self, center_hz, vfo_hz=None):
            dev.set_frequency(center_hz)
            return True

        def scan_tune_channel(self, freq_hz, listening, tone_hz=0.0):
            return True

        def scan_lo_hz(self):
            return dev.frequency

        def scan_signal_present(self):
            return False

        def scan_blocker(self):
            return self.state["blocker"]

        def scan_receiving(self):
            return self.state["receiving"]

        def scan_start_receiver(self):
            Host.started += 1
            self.state["receiving"] = True
            return True

        def scan_sample_rate(self):
            return dev.sample_rate

        def scan_device_range(self):
            return device_range

        def scan_listen(self, hit):
            Host.listened.append(hit.frequency_hz)
            return True

    panel = ScannerPanel()
    host = Host()
    panel.set_host(host)
    return panel, host, dev


def _feed(panel, dev, seconds=10.0):
    for _ in range(int(seconds / 0.034)):
        if not panel.engine.active:
            break
        panel.engine.frame(dev.read_samples(int(dev.sample_rate * 0.034)))


class TestScanner(unittest.TestCase):
    """The frequency scanner (now the Scanner tab, scanning in the display
    loop; its measurements are in sdr_module.dsp.sweep)."""

    def setUp(self):
        require_pyqt6(self)

    def test_scan_reports_each_station_once_at_its_frequency(self):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings
        from tests.test_gui_scanner import _DemoHost

        dev = MockDevice()
        dev.start_rx()
        dev.set_frequency(145e6)
        engine = ScanEngine(_DemoHost(dev))
        hits = []
        engine.on_found = hits.append
        engine.start(ScanSettings("once", 99.9e6, 100.7e6, 200e3), dev.sample_rate, 0.0)
        for _ in range(40):
            engine.frame(dev.read_samples(int(dev.sample_rate * 0.034)))
        near = [h.frequency_hz for h in hits if abs(h.frequency_hz - 100.1e6) < 150e3]
        self.assertEqual(near, [100.1e6])  # once, on its channel
        self.assertIsNotNone(engine.noise_db)

    def test_measure_limits_peak_to_the_step(self):
        from sdr_module.dsp.sweep import SpectrumAverager, measure_page

        t = np.arange(2048 * 8)
        tone = np.exp(2j * np.pi * 0.1 * t).astype(np.complex64)
        averager = SpectrumAverager()
        averager.add(tone)
        power = averager.power_db()
        # The tone sits 240 kHz off center: outside a 200 kHz step's slice...
        readings, _n, _t = measure_page(power, 100e6, 2.4e6, [100e6], 200e3, -20.0)
        self.assertEqual(readings, [])
        # ...but inside a 600 kHz one, at the right frequency.
        readings, _n, _t = measure_page(power, 100e6, 2.4e6, [100e6], 600e3, -20.0)
        self.assertAlmostEqual(readings[0].frequency_hz, 100.24e6, delta=2e3)
        self.assertGreater(readings[0].peak_db, -3)

    def test_range_validation_and_presets(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        panel, _host, _dev = _scanner_panel()
        self.assertTrue(panel._start_btn.isEnabled())
        panel._end.setValue(50.0)  # below start
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertEqual(panel._preset.currentIndex(), 0)  # now "Custom"
        panel._preset.setCurrentIndex(2)
        preset = SCAN_PRESETS[1]
        self.assertEqual(
            (panel._start.value(), panel._end.value(), panel._step.value()),
            (preset.start_mhz, preset.end_mhz, preset.step_khz),
        )
        self.assertTrue(panel._start_btn.isEnabled())

    def test_empty_state_and_tune_to_result(self):
        from sdr_module.gui.scan_engine import Found

        panel, host, _dev = _scanner_panel()
        self.assertTrue(panel._empty.is_visible())
        self.assertFalse(panel._listen_btn.isEnabled())
        panel._on_found(Found(101.1e6, -40.0, -90.0, 150e3, ("FM", "75 kHz", "")))
        self.assertFalse(panel._empty.is_visible())
        self.assertEqual(panel._table.item(0, 2).text(), "50 dB")
        panel._table.selectRow(0)
        self.assertTrue(panel._listen_btn.isEnabled())
        panel.listen_selected()
        self.assertEqual(host.listened[-1], 101.1e6)

    def test_neighbouring_steps_merge_into_one_row(self):
        from sdr_module.gui.scan_engine import Found

        panel, _host, _dev = _scanner_panel()
        panel._on_found(Found(100.29e6, -35.0, -90.0, 5e3, ("", "", "")))
        panel._on_found(Found(100.31e6, -30.0, -90.0, 5e3, ("", "", "")))
        self.assertEqual(panel._table.rowCount(), 1)
        self.assertEqual(panel.hits()[0].peak_db, -30.0)


class TestFirstRunWizard(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_get_started_is_default_and_tunes(self):
        from sdr_module.gui.first_run_wizard import BAND_PRESETS, FirstRunWizard

        wiz = FirstRunWizard(hardware_found=False)
        self.assertTrue(wiz._start_btn.isDefault())
        self.assertEqual(wiz._status.property("tone"), "info")
        wiz._band.setCurrentText("2m Ham (144–148 MHz)")
        self.assertIn("146.520 MHz", wiz._band_hint.text())
        wiz._on_accept()
        self.assertEqual(wiz.selected_frequency(), BAND_PRESETS["2m Ham (144–148 MHz)"])

    def test_hardware_found_reads_as_success(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        wiz = FirstRunWizard(hardware_found=True)
        self.assertEqual(wiz._status.property("tone"), "success")


class TestHelpDialog(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_entries_follow_the_live_bindings(self):
        from sdr_module.gui.help_dialog import build_entries

        bindings = {
            "Ctrl+O": ("File", "Open Recording"),
            "Ctrl+J": ("Tools", "Jump Somewhere"),
        }
        entries = build_entries(bindings)
        keys = {k for _cat, ks, _d in entries for k in ks}
        self.assertIn("Ctrl+O", keys)
        self.assertNotIn("Ctrl+R", keys)  # not bound -> not listed
        # Unknown menu items are listed in sentence case like the rest.
        self.assertIn(("Tools", ("Ctrl+J",), "Jump somewhere"), entries)

    def test_reads_shortcuts_from_the_parent_window(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMainWindow

        from sdr_module.gui.help_dialog import HelpDialog

        class Window(QMainWindow):
            _TUNE_STEPS_HZ = {
                Qt.KeyboardModifier.NoModifier: 5e3,
                Qt.KeyboardModifier.ShiftModifier: 50e3,
            }

        win = Window()
        radio = win.menuBar().addMenu("&Radio")
        start = QAction("&Start Receiving\tSpace", win)
        radio.addAction(start)
        tools = win.menuBar().addMenu("&Tools")
        scan = QAction("Frequency &Scanner...", win)
        scan.setShortcut("Ctrl+F")
        tools.addAction(scan)

        entries = HelpDialog(win).shortcut_entries()
        by_key = {ks: (cat, d) for cat, ks, d in entries}
        self.assertEqual(by_key[("Space",)][0], "Radio")
        self.assertEqual(by_key[("Ctrl+F",)], ("Tools", "Open the frequency scanner"))
        self.assertEqual(
            by_key[("Left", "Right")], ("Tuning", "Tune down / up by 5 kHz")
        )
        self.assertIn(("Shift+Left", "Shift+Right"), by_key)
        self.assertNotIn(("Ctrl+Left", "Ctrl+Right"), by_key)
        self.assertNotIn(("Ctrl+E",), by_key)

    def test_static_table_without_parent_and_filter(self):
        from sdr_module.gui.help_dialog import SHORTCUTS, HelpDialog

        dialog = HelpDialog()
        self.assertEqual(len(dialog.shortcut_entries()), len(SHORTCUTS))
        dialog._filter.setText("scanner")
        visible = [s for row, s, _sec in dialog._rows if not row.isHidden()]
        # Ctrl+F, and Enter in the Scanner tab's results (mc:MC-11: the keys
        # lists handle themselves are listed too).
        self.assertEqual(len(visible), 2, visible)
        self.assertTrue(any("ctrl+f" in s for s in visible))
        self.assertTrue(any(s.startswith("scanner results") for s in visible))
        dialog._filter.setText("no such thing")
        self.assertFalse(dialog._no_match.isHidden())

    def test_shared_modifiers_are_compacted(self):
        from sdr_module.gui.help_dialog import _keycap_tokens

        self.assertEqual(
            _keycap_tokens(["Shift+Left", "Shift+Right"]), ["Shift", "+", "←", "/", "→"]
        )


class TestErrorLogDialog(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import error_log_dialog

        self.mod = error_log_dialog
        self.mod._HISTORY.clear_history()
        self.logger = logging.getLogger("sdr_module.test_dialogs")
        self.logger.addHandler(self.mod._HISTORY)
        self.logger.propagate = False

    def tearDown(self):
        self.logger.removeHandler(self.mod._HISTORY)
        self.mod._HISTORY.clear_history()

    def test_repeats_are_grouped_newest_first(self):
        self.logger.warning("driver missing")
        self.logger.error("open failed")
        self.logger.warning("driver missing")
        groups = self.mod.group_entries(self.mod._HISTORY.entries())
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["text"], "driver missing")
        self.assertEqual(groups[0]["count"], 2)
        # The legacy snapshot format is unchanged.
        ts, lvl, text = self.mod._HISTORY.snapshot()[1]
        self.assertEqual(text, "sdr_module.test_dialogs: open failed")
        self.assertEqual(lvl, logging.ERROR)

    def test_empty_state_then_rows_copy_and_clear(self):
        dialog = self.mod.ErrorLogDialog()
        self.assertFalse(dialog._empty.isHidden())
        self.assertFalse(dialog._clear_btn.isEnabled())
        # Enter must never clear the log.
        self.assertFalse(dialog._clear_btn.autoDefault())

        self.logger.error("tuner timeout")
        dialog._refresh_if_changed()
        self.assertEqual(dialog._table.rowCount(), 1)
        self.assertEqual(dialog._table.item(0, 1).text(), "Error")
        self.assertTrue(dialog._empty.isHidden())

        dialog._copy_all()
        self.assertIn("tuner timeout", QApplication.clipboard().text())

        dialog._clear()
        self.assertEqual(dialog._table.rowCount(), 0)
        self.assertEqual(self.mod._HISTORY.entries(), [])
        dialog.done(0)


class TestScannerReview(unittest.TestCase):
    """Sweeps measure the right frequency, fail fast and match the display."""

    def setUp(self):
        require_pyqt6(self)

    @staticmethod
    def _noise_block(block):
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(2 * block).view(np.complex128)
        return (noise * 1e-3).astype(np.complex64)

    def test_steps_outside_the_tuning_range_are_skipped(self):
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings
        from tests.test_gui_scanner import _demo, _DemoHost

        dev = _demo(1760e6)
        host = _DemoHost(dev)
        host.refuse_above = 1766e6  # where the receiver stops tuning
        engine = ScanEngine(host)
        hits = []
        engine.on_found = hits.append
        engine.start(
            ScanSettings("once", 1760e6, 1780e6, 1e6, -200.0), dev.sample_rate, 0.0
        )
        for _ in range(200):
            if not engine.active:
                break
            engine.frame(dev.read_samples(int(dev.sample_rate * 0.034)))
        self.assertGreater(engine.steps_refused, 0)
        self.assertGreater(engine.steps_measured, 0)
        # Nothing is reported at a frequency the device never tuned near.
        self.assertTrue(all(h.frequency_hz < 1766.6e6 for h in hits), hits)

    def test_dead_device_gives_up_after_two_steps(self):
        # A device that delivers nothing gives the scan no blocks: nothing is
        # reported, and when the window stops the dead stream (as it does
        # for a hardware stream that died) the scan stops and says so.
        panel, host, dev = _scanner_panel()
        self.assertTrue(panel.start_scan())
        for _ in range(10):
            panel.engine.frame(np.zeros(0, np.complex64))
        self.assertEqual(panel.hits(), [])
        host.state["receiving"] = False
        panel.refresh()
        self.assertFalse(panel.engine.active)
        self.assertIn("the receiver was stopped", panel._status.text())

    def test_stopped_demo_device_is_started_for_the_sweep_only(self):
        # The scan runs in the display loop, so a stopped receiver is
        # started for it (and keeps running, to listen to what it finds).
        panel, host, dev = _scanner_panel(receiving=False)
        panel._start.setValue(100.1)
        panel._end.setValue(100.5)
        self.assertTrue(panel.start_scan())
        self.assertEqual(host.started, 1)
        _feed(panel, dev)
        self.assertTrue(
            any(abs(h.frequency_hz - 100.1e6) < 50e3 for h in panel.hits()),
            panel.hits(),
        )

    def test_levels_do_not_depend_on_the_device_block_size(self):
        from sdr_module.dsp.sweep import SpectrumAverager, noise_floor_db

        floors = []
        for block in (8192, 262144):
            averager = SpectrumAverager()
            averager.add(self._noise_block(block))
            floors.append(noise_floor_db(averager.power_db()))
        self.assertAlmostEqual(floors[0], floors[1], delta=1.0)

    def test_dialog_keeps_the_sweep_inside_the_device_range(self):
        panel, _host, _dev = _scanner_panel(device_range=(24e6, 1766e6))
        # Partly outside: the hint says so before the scan starts.
        panel._start.setValue(1700.0)
        panel._end.setValue(1800.0)
        panel._step.setValue(1000.0)
        self.assertIn("67 channels", panel._hint.text())
        self.assertIn("Limited to the device's range", panel._hint.text())
        self.assertTrue(panel._start_btn.isEnabled())
        # Entirely outside: Start is disabled and says why.
        panel._start.setValue(1800.0)
        panel._end.setValue(1900.0)
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertIn("1766.000 MHz", panel._start_btn.toolTip())
        panel.toggle_scan()
        self.assertFalse(panel.engine.active)
        self.assertEqual(panel._status.property("tone"), "danger")
        self.assertIn("1766.000 MHz", panel._status.text())

    def test_no_device_explains_how_to_connect(self):
        panel, _host, _dev = _scanner_panel(
            blocker=(
                "device",
                "No receiver: use the demo device or connect an SDR to scan.",
            )
        )
        self.assertIn("Use Demo Device", panel._empty.text())
        self.assertIn("Connect...", panel._empty.text())
        self.assertFalse(panel._demo_btn.isHidden())
        self.assertFalse(panel._connect_btn.isHidden())

    def test_start_button_keeps_its_width(self):
        panel, _host, _dev = _scanner_panel()
        width = panel._start_btn.minimumWidth()
        self.assertTrue(panel.start_scan())
        self.assertEqual(panel._start_btn.text(), "Stop Scan")
        self.assertEqual(panel._start_btn.minimumWidth(), width)
        self.assertGreaterEqual(width, panel._start_btn.sizeHint().width())
        panel.stop_scan()


class TestFirstRunDemoMode(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_demo_is_ticked_only_without_hardware(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        self.assertTrue(FirstRunWizard(hardware_found=False).wants_demo_mode())
        self.assertFalse(FirstRunWizard(hardware_found=True).wants_demo_mode())

    def test_get_started_starts_demo_mode_on_the_window(self):
        from PyQt6.QtWidgets import QWidget

        from sdr_module.gui.first_run_wizard import FirstRunWizard

        class Window(QWidget):
            demo_started = 0

            def _start_demo_mode(self):
                self.demo_started += 1

        win = Window()
        FirstRunWizard(win, hardware_found=False)._on_accept()
        self.assertEqual(win.demo_started, 1)

        # With a listener connected, the signal is used instead.
        heard = []
        wiz = FirstRunWizard(win, hardware_found=False)
        wiz.demo_mode_requested.connect(lambda: heard.append(True))
        wiz._on_accept()
        self.assertEqual(heard, [True])
        self.assertEqual(win.demo_started, 1)

        # Unticked: no demo.
        wiz = FirstRunWizard(win, hardware_found=False)
        wiz._demo_check.setChecked(False)
        wiz._on_accept()
        self.assertEqual(win.demo_started, 1)


class TestHelpReview(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_static_categories_match_the_menus(self):
        from sdr_module.gui.help_dialog import SHORTCUTS

        menus = {"File", "Device", "Radio", "View", "Tools", "Help", "Tuning"}
        self.assertTrue({cat for cat, _k, _d in SHORTCUTS} <= menus)

    def test_panel_shortcuts_read_naturally(self):
        from sdr_module.gui.help_dialog import build_entries

        entries = build_entries({"Ctrl+1": ("View", "Panels › Decoder")})
        self.assertIn(("View", ("Ctrl+1",), "Show the Decoder panel"), entries)

    def test_keycap_column_fits_the_widest_keys(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = HelpDialog()
        widths = {caps.width() for caps in dialog._caps}
        self.assertEqual(len(widths), 1)
        widest = max(caps.sizeHint().width() for caps in dialog._caps)
        self.assertGreaterEqual(dialog._caps[0].minimumWidth(), widest)


class TestDeviceTableLayout(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_long_serial_is_elided_not_the_name(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QHeaderView

        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = DeviceDialog()
        table = dialog._device_table
        header = table.horizontalHeader()
        self.assertEqual(header.sectionResizeMode(2), QHeaderView.ResizeMode.Stretch)
        self.assertEqual(
            header.sectionResizeMode(1), QHeaderView.ResizeMode.ResizeToContents
        )
        self.assertFalse(table.wordWrap())
        self.assertEqual(table.textElideMode(), Qt.TextElideMode.ElideMiddle)
        serial = "0000000000000000a06063c8234e925f"
        dialog._add_device("HackRF", "HackRF One", serial, "Available")
        self.assertIn(serial, table.item(table.rowCount() - 1, 2).toolTip())


class TestErrorLogReview(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import error_log_dialog

        self.mod = error_log_dialog
        self.mod._HISTORY.clear_history()
        self.logger = logging.getLogger("sdr_module.test_dialogs_review")
        self.logger.addHandler(self.mod._HISTORY)
        self.logger.propagate = False

    def tearDown(self):
        self.logger.removeHandler(self.mod._HISTORY)
        self.mod._HISTORY.clear_history()

    def test_details_follow_a_growing_repeat_count(self):
        self.logger.warning("driver missing")
        self.logger.warning("driver missing")
        dialog = self.mod.ErrorLogDialog()
        dialog._table.selectRow(0)
        self.assertIn("2 times", dialog._details.toPlainText())
        self.logger.warning("driver missing")
        dialog._refresh_if_changed()
        self.assertIn("3 times", dialog._details.toPlainText())
        dialog.done(0)


if __name__ == "__main__":
    unittest.main()
