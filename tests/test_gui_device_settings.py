"""The connected device's settings in the window (T11).

Gain that matches the device (fg:FG-B15, rx:RX-14, ux:UX-B07, ux:UX-B08,
fg:FG-U07, rx:UX-10), the sample rate changed live (mc:MC-G04, rx:G-03,
ux:UX-G12, ux:UX-U15, tl:UX-13), the RTL-SDR's and HackRF's front end
(fg:FG-G23, mc:MC-G05, mc:MC-G06, tl:G-05, ux:UX-G07, ux:UX-G08,
tl:TOOLS-13, fg:FG-U05) and the Device menu (mc:MC-U16, mc:MC-U12).

No hardware: a real RTLSDRDevice on a fake ``rtlsdr`` module (pyrtlsdr's
surface) and a real HackRFDevice on the threaded fake python_hackrf
(tests/hackrf_fakes.py), connected the way Device > Connect... does.
"""

from __future__ import annotations

import logging
import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.hackrf_fakes import open_hackrf  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_station_tx import _Patch  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QMessageBox

E4000_GAINS = [-1.0, 1.5, 4.0, 6.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0, 29.0]


class FakeRtlSdr:
    """pyrtlsdr's RtlSdr (what the driver uses), recording what it is told."""

    gains_db = None  # the next device's valid_gains_db (None: R820T)
    fail = set()  # method names that raise

    def __init__(self, device_index=0):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        self.sample_rate = 2.048e6
        self.center_freq = 100e6
        self.gain = "auto"
        self.corr = 0
        self.direct = 0
        self.bias = False
        self.closed = False
        self.valid_gains_db = list(
            FakeRtlSdr.gains_db
            if FakeRtlSdr.gains_db is not None
            else RTLSDRDevice.VALID_GAINS
        )

    @staticmethod
    def get_device_serial_addresses():
        return ["00000001"]

    def _check(self, name):
        if name in FakeRtlSdr.fail:
            raise OSError(f"{name} failed")

    def set_freq_correction(self, ppm):
        self._check("set_freq_correction")
        if ppm == self.corr:
            self.closed = True  # pyrtlsdr closes the device on -2
            raise OSError("same ppm")
        self.corr = ppm

    def set_direct_sampling(self, mode):
        self._check("set_direct_sampling")
        self.direct = mode

    def set_bias_tee(self, enabled):
        self._check("set_bias_tee")
        self.bias = bool(enabled)

    def read_samples(self, n):
        import numpy as np

        return np.zeros(n, dtype=np.complex64)

    def close(self):
        self.closed = True


class _DeviceCase(_WindowTestCase):
    """A window (demo device) with helpers to connect a fake RTL-SDR or
    HackRF One as Device > Connect... does."""

    demo = True

    def setUp(self):
        super().setUp()
        self.patch = _Patch()
        self.addCleanup(self.patch.undo)
        self.patch.setattr(FakeRtlSdr, "gains_db", None)
        self.patch.setattr(FakeRtlSdr, "fail", set())
        rtlsdr = types.ModuleType("rtlsdr")
        librtlsdr = types.ModuleType("rtlsdr.librtlsdr")
        librtlsdr.rtlsdr_get_device_count = lambda: 1
        rtlsdr.RtlSdr = FakeRtlSdr
        rtlsdr.librtlsdr = librtlsdr
        self.patch.setitem(sys.modules, "rtlsdr", rtlsdr)
        self.patch.setitem(sys.modules, "rtlsdr.librtlsdr", librtlsdr)
        self.addCleanup(self._release)
        self.panel = self.win._control_panel

    def _release(self):
        self.win._tx_station.shutdown()
        if self.win._device is not None:
            self.win._release_device()

    def connect(self, dev):
        """What Device > Connect... does with the device it opened."""
        win = self.win
        if win._device is not None and win._device is not dev:
            win._release_device()
        win._device = dev
        win._demo_mode = False
        note = win._apply_controls_to_device()
        win._refresh_state_ui()
        return note

    def rtl(self, rate=2.4e6):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        dev = RTLSDRDevice()
        self.assertTrue(dev.open(0))
        dev.set_sample_rate(rate)
        self.connect(dev)
        return dev

    def hackrf(self, rate=10e6):
        dev, fake = open_hackrf(self.patch, rate=rate)
        self.connect(dev)
        return dev, fake

    def message(self):
        return self.win._message_label.text()

    def tone(self):
        return self.win._message_label.property("tone")

    def visible(self, widget):
        return self.panel._receiver_form.isRowVisible(widget)


# --------------------------------------------------------------------------- #
# Gain that matches the device
# --------------------------------------------------------------------------- #
class TestRtlGain(_DeviceCase):
    """fg:FG-B15, rx:RX-14: the RF gain steps through the tuner's table."""

    def test_the_slider_has_the_tuners_steps_and_shows_the_gain_applied(self):
        dev = self.rtl()
        slider = self.panel._gain_slider
        self.assertEqual((slider.minimum(), slider.maximum()), (0, 28))
        self.panel.set_agc_enabled(False)
        records = []
        handler = logging.Handler(logging.WARNING)
        handler.emit = records.append
        logger = logging.getLogger("sdr_module")
        logger.addHandler(handler)
        try:
            for index in range(slider.maximum() + 1):
                slider.setValue(index)
                applied = dev.state.gain
                self.assertEqual(applied, dev.get_tuner_gains()[index])
                self.assertEqual(
                    self.panel._gain_label.text(),
                    f"{applied:.1f}".rstrip("0").rstrip(".") + " dB",
                )
        finally:
            logger.removeHandler(handler)
        self.assertEqual(records, [])  # nothing "clamped" in Error History
        self.assertEqual(self.panel.gain_db(), 49.6)
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("gain"), "49.6 dB")

    def test_another_tuners_table(self):
        FakeRtlSdr.gains_db = E4000_GAINS
        dev = self.rtl()
        slider = self.panel._gain_slider
        self.assertEqual(slider.maximum(), len(E4000_GAINS) - 1)
        self.panel.set_agc_enabled(False)
        slider.setValue(0)
        self.assertEqual(dev.state.gain, -1.0)
        self.assertEqual(self.panel._gain_label.text(), "-1 dB")

    def test_set_gain_snaps_to_the_nearest_step(self):
        dev = self.rtl()
        self.panel.set_agc_enabled(False)
        self.win.set_gain(30)
        self.assertEqual(dev.state.gain, 29.7)
        self.assertEqual(self.panel._gain_label.text(), "29.7 dB")

    def test_tuner_agc_is_offered_and_says_auto(self):
        dev = self.rtl()
        self.assertTrue(self.panel._agc_check.isEnabled())
        self.panel.set_agc_enabled(True)
        self.assertEqual(dev.state.gain_mode, "auto")
        self.assertEqual(self.panel._gain_label.text(), "Auto")
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("gain"), "Auto (tuner AGC)")


class TestHackRFGain(_DeviceCase):
    """fg:FG-B15, ux:UX-B07, ux:UX-B08, ux:UX-G08: LNA, VGA and the RF amp;
    no AGC."""

    def test_lna_vga_and_amp_replace_the_rf_gain_slider(self):
        dev, fake = self.hackrf()
        p = self.panel
        self.assertFalse(self.visible(p._gain_slider.parentWidget()))
        for widget in (
            p._lna_slider.parentWidget(),
            p._vga_slider.parentWidget(),
            p._amp_check,
        ):
            self.assertTrue(self.visible(widget))
        # The driver's defaults to start with (LNA 16, VGA 20).
        self.assertEqual(p.hackrf_gains(), (16, 20, False))
        self.assertEqual((fake.lna, fake.vga, fake.amp), (16, 20, False))
        p._lna_slider.setValue(p._lna_slider.maximum())
        p._vga_slider.setValue(p._vga_slider.maximum())
        self.assertEqual((dev.lna_gain, dev.vga_gain), (40, 62))
        self.assertEqual((fake.lna, fake.vga), (40, 62))
        self.assertEqual(p._vga_label.text(), "62 dB")
        self.assertEqual(p.gain_db(), 102.0)

    def test_the_amp_switches_and_warns(self):
        dev, fake = self.hackrf()
        p = self.panel
        self.assertFalse(self.visible(p._amp_warning))
        p._amp_check.setChecked(True)
        self.assertTrue(dev.amp_enabled and fake.amp)
        self.assertTrue(self.visible(p._amp_warning))
        self.assertIn("overload", p._amp_warning.text())
        self.assertIn("overload", self.message())
        self.assertEqual(self.tone(), "warning")
        self.assertIn("overload", p._amp_check.toolTip())
        self.win._refresh_info_panel()
        self.assertEqual(
            self.win._info_panel.value("gain"), "50 dB (LNA 16 + VGA 20 + amp 14)"
        )
        p._amp_check.setChecked(False)
        self.assertFalse(fake.amp)
        self.win._refresh_info_panel()
        self.assertEqual(
            self.win._info_panel.value("gain"), "36 dB (LNA 16 + VGA 20, amp off)"
        )

    def test_no_agc_on_a_hackrf(self):
        dev, fake = self.hackrf()
        agc = self.panel._agc_check
        self.assertFalse(agc.isEnabled())
        self.assertFalse(agc.isChecked())
        self.assertIn("no automatic gain", agc.toolTip())
        # Even asked for directly, it is refused and says why.
        self.win._on_agc_changed(True)
        self.assertFalse(agc.isChecked())
        self.assertFalse(self.win._agc_enabled)
        self.assertEqual(dev.state.gain_mode, "manual")
        self.assertIn("no automatic gain", self.message())
        self.assertEqual(self.tone(), "warning")

    def test_saved_agc_does_not_carry_over_to_a_hackrf(self):
        self.panel.set_agc_enabled(True)  # on the demo device
        dev, fake = self.hackrf()
        self.assertFalse(self.panel.is_agc_enabled())
        self.assertFalse(self.win._agc_enabled)
        self.assertEqual((fake.lna, fake.vga), (16, 20))  # a manual gain set

    def test_the_transmit_power_is_left_alone(self):
        station = self.win._tx_station
        before = (station.power.gain_db, station.power.amp)
        dev, fake = self.hackrf()
        self.panel._amp_check.setChecked(True)
        self.panel._vga_slider.setValue(5)
        self.assertEqual((station.power.gain_db, station.power.amp), before)
        self.assertNotIn("tx/amp", self.store.values)


class TestAgcRefusals(_DeviceCase):
    def test_a_device_refusing_agc_is_reported_and_stays_manual(self):
        calls = []

        class Dev:
            sample_rate = 2.4e6

            def set_gain_mode(self, auto):
                calls.append(("mode", auto))
                return not auto

            def set_gain(self, db):
                calls.append(("gain", db))
                return True

            def set_frequency(self, freq):
                return True

            def close(self):
                pass

        self.win._release_device()
        self.win._device = Dev()
        self.panel.set_gain(33)
        self.panel.set_agc_enabled(True)
        self.assertFalse(self.panel.is_agc_enabled())
        self.assertFalse(self.panel._agc_check.isChecked())
        self.assertTrue(self.panel._gain_slider.isEnabled())
        self.assertEqual(calls[-2:], [("mode", False), ("gain", 33.0)])
        self.assertIn("didn't switch to automatic gain", self.message())
        self.win._device = None

    def test_tuner_agc_is_named_and_explained(self):
        agc = self.panel._agc_check
        self.assertEqual(agc.text(), "Tuner AGC")
        self.assertIn("Audio AGC", agc.toolTip())


class TestAutoGainReadout(_DeviceCase):
    """rx:UX-10: what the automatic gain chose, in the label and Info."""

    def test_the_demo_agc_gain_is_shown(self):
        from sdr_module.gui.device_settings import format_db

        self.win.set_frequency(100.1e6)
        self.panel.set_agc_enabled(True)
        chosen = self.win._device.gain
        text = f"Auto ({format_db(round(chosen, 1))})"
        self.assertEqual(self.panel._gain_label.text(), text)
        self.assertNotEqual(text, "Auto")
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("gain"), text)
        # It follows the band (another passband, another choice).
        self.win.set_frequency(162.55e6)
        self.win._update_status()
        now = self.win._device.gain
        self.assertNotEqual(now, chosen)
        self.assertEqual(
            self.panel._gain_label.text(), f"Auto ({format_db(round(now, 1))})"
        )

    def test_iq_mode_says_there_is_no_audio(self):
        self.win._set_demod_mode("None (I/Q)")
        self.win._audio_enabled = True
        with mock.patch.object(type(self.win), "_realtime", return_value=True):
            text, tone = self.win._audio_status()
        self.assertEqual(text, "On (no audio in I/Q mode)")
        self.win._set_demod_mode("FM")
        self.win._audio_enabled = True  # (no sound card here switched it off)
        with mock.patch.object(type(self.win), "_realtime", return_value=True):
            self.assertEqual(self.win._audio_status()[0], "On")


class TestGainMemory(_DeviceCase):
    def test_each_device_type_keeps_its_gain(self):
        self.panel.set_agc_enabled(False)
        self.win.set_gain(34)  # the demo device
        dev = self.rtl()
        self.panel.set_agc_enabled(False)
        self.win.set_gain(40)
        self.assertEqual(dev.state.gain, 40.2)
        dev_h, fake = self.hackrf()
        self.panel._lna_slider.setValue(3)  # 24 dB
        self.panel._amp_check.setChecked(True)
        self.win._release_device()
        self.win._start_demo_mode()
        self.assertEqual(self.win._device.gain, 34.0)
        self.assertEqual(self.panel._gain_label.text(), "34 dB")
        dev = self.rtl()
        self.assertEqual(dev.state.gain, 40.2)
        self.assertFalse(self.panel.is_agc_enabled())
        dev_h, fake = self.hackrf()
        self.assertEqual(self.panel.hackrf_gains(), (24, 20, True))
        self.assertEqual((fake.lna, fake.amp), (24, True))

    def test_the_global_gain_is_saved_in_db(self):
        self.rtl()
        self.panel.set_agc_enabled(False)
        self.win.set_gain(30)
        self.win._persist_state()
        self.assertEqual(self.store.values["gain_db"], 29.7)


# --------------------------------------------------------------------------- #
# Sample rate, live
# --------------------------------------------------------------------------- #
class TestSampleRate(_DeviceCase):
    def rates(self):
        combo = self.panel._rate_combo
        return [combo.itemData(i) for i in range(combo.count())]

    def labels(self):
        combo = self.panel._rate_combo
        return [combo.itemText(i) for i in range(combo.count())]

    def test_the_receiver_group_offers_the_devices_rates(self):
        combo = self.panel._rate_combo
        self.assertTrue(combo.isEnabled())
        self.assertEqual(max(self.rates()), 3.2e6)  # the demo device
        self.assertIn("1.8 MS/s (too low for ADS-B)", self.labels())
        self.assertIn("2 MS/s", self.labels())
        index = self.labels().index("1.8 MS/s (too low for ADS-B)")
        self.assertIn("ADS-B", combo.itemData(index, Qt.ItemDataRole.ToolTipRole))
        self.assertEqual(combo.currentText(), "2.4 MS/s")
        self.rtl()
        self.assertEqual(max(self.rates()), 2.56e6)
        self.hackrf()
        self.assertEqual((min(self.rates()), max(self.rates())), (2e6, 20e6))

    def test_choosing_a_rate_applies_it_live(self):
        win = self.win
        dev = win._device
        win._decoder_panel._proto_combo.setCurrentText("POCSAG")
        self.assertTrue(win._is_running)
        combo = self.panel._rate_combo
        index = self.rates().index(1.8e6)
        combo.setCurrentIndex(index)
        combo.activated.emit(index)
        self.assertIs(win._device, dev)  # not reconnected
        self.assertTrue(win._is_running)
        self.assertEqual(dev.sample_rate, 1.8e6)
        self.assertEqual(win._rate_label.text(), "1.8 MS/s")
        self.assertAlmostEqual(win._spectrum._sample_rate, 1.8e6)
        self.assertAlmostEqual(win._waterfall._sample_rate, 1.8e6)
        self.assertEqual(win._decoder.sample_rate, 1.8e6)
        self.assertEqual(self.store.values["device/demo/sample_rate"], 1.8e6)
        self.assertIn("1.8 MS/s", self.message())
        win._update_display()  # the receiver chain is rebuilt for the rate
        self.assertEqual(win._rx_chain.rate, 1.8e6)

    def test_too_low_for_adsb_is_said_when_adsb_decodes(self):
        self.win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        self.assertTrue(self.win._device_controls.choose_rate(1.8e6))
        self.assertEqual(self.win._device.sample_rate, 1.8e6)
        self.assertIn("ADS-B", self.message())

    def test_the_status_bar_rate_opens_the_menu(self):
        label = self.win._rate_label
        menu = self.win._device_controls.rate_menu
        self.assertEqual(label.cursor().shape(), Qt.CursorShape.PointingHandCursor)
        QTest.mouseClick(label, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(menu.isVisible())
        texts = [a.text() for a in menu.actions()]
        self.assertIn("1 MS/s (too low for ADS-B)", texts)
        checked = [a.text() for a in menu.actions() if a.isChecked()]
        self.assertEqual(checked, ["2.4 MS/s"])
        action = next(a for a in menu.actions() if a.text() == "3.2 MS/s")
        action.trigger()
        menu.hide()
        self.assertEqual(self.win._device.sample_rate, 3.2e6)
        self.assertEqual(label.text(), "3.2 MS/s")
        self.assertIn("Click to choose", label.toolTip())

    def test_device_menu_has_the_same_menu(self):
        menus = [a.menu() for a in self.win.menuBar().actions()]
        device = next(m for m in menus if m and m.title() == "&Device")
        self.assertIn(
            self.win._device_controls.rate_menu, [a.menu() for a in device.actions()]
        )

    def test_a_refused_pick_in_the_combo_goes_back(self):
        win = self.win
        win._toggle_recording(True)
        combo = self.panel._rate_combo
        index = self.rates().index(1.8e6)
        combo.setCurrentIndex(index)  # the user picks 1.8 MS/s...
        combo.activated.emit(index)
        self.assertEqual(win._device.sample_rate, 2.4e6)  # ...refused
        self.assertEqual(combo.currentText(), "2.4 MS/s")
        win._toggle_recording(False)

    def test_refused_while_recording_iq(self):
        win = self.win
        win._toggle_recording(True)
        self.assertTrue(win._recording)
        self.assertFalse(win._device_controls.choose_rate(1.8e6))
        self.assertEqual(win._device.sample_rate, 2.4e6)
        self.assertIn("Stop recording to change the sample rate", self.message())
        self.assertEqual(self.panel._rate_combo.currentText(), "2.4 MS/s")
        win._toggle_recording(False)

    def test_without_a_device_the_choice_waits(self):
        self.win._release_device()
        self.win._refresh_state_ui()
        combo = self.panel._rate_combo
        self.assertFalse(combo.isEnabled())
        self.assertIn("Connect a device", combo.toolTip())
        menu = self.win._device_controls.rate_menu
        self.win._device_controls._fill_rate_menu()
        self.assertEqual(len(menu.actions()), 1)
        self.assertFalse(menu.actions()[0].isEnabled())

    def test_the_rate_is_remembered_per_device_type(self):
        self.rtl()
        self.assertTrue(self.win._device_controls.choose_rate(1.4e6))
        self.assertEqual(self.store.values["device/rtlsdr/sample_rate"], 1.4e6)
        self.hackrf(rate=8e6)
        self.assertTrue(self.win._device_controls.choose_rate(20e6))
        self.assertEqual(self.store.values["device/hackrf/sample_rate"], 20e6)
        self.win._release_device()
        self.win._preferred_rate = None
        rates, fronts = self.win._device_controls.dialog_preferences()
        self.assertEqual(rates["rtlsdr"], 1.4e6)
        self.assertEqual(rates["hackrf"], 20e6)
        self.assertEqual(rates["demo"], 2.4e6)

    def test_use_demo_device_takes_its_remembered_rate(self):
        self.store.values["device/demo/sample_rate"] = 3.2e6
        self.win._release_device()
        self.win._preferred_rate = None
        self.win._start_demo_mode()
        self.assertEqual(self.win._device.sample_rate, 3.2e6)


class TestSampleRateDuringPlayback(_WindowTestCase):
    """While a recording plays, the file sets the rate: refused with the
    reason, and the choice is shown but off."""

    demo = True

    def test_refused_with_the_reason(self):
        import shutil
        import tempfile

        from tests.test_gui_playback import _fm_tone, _save_sigmf

        folder = tempfile.mkdtemp(prefix="sdr-t11-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.win._recording_folder = folder
        path = _save_sigmf(os.path.join(folder, "fm.sigmf-data"), _fm_tone(1.0))
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
            return_value=(path, ""),
        ):
            self.win._open_recording_action.trigger()
        self.settle()
        win = self.win
        self.assertTrue(win._playback.active)
        combo = win._control_panel._rate_combo
        self.assertFalse(combo.isEnabled())
        self.assertEqual(combo.currentText(), "0.24 MS/s (too low for ADS-B)")
        self.assertIn("recording sets the sample rate", combo.toolTip())
        self.assertFalse(win.set_sample_rate(2.4e6))
        self.assertEqual(win._device.sample_rate, 240e3)
        self.assertIn("recording sets the sample rate", win._message_label.text())
        # Gain stays off (as recorded), Tuner AGC too.
        self.assertFalse(win._control_panel._gain_slider.parentWidget().isEnabled())
        self.assertFalse(win._control_panel._agc_check.isEnabled())
        win._playback.close(restore=True)
        self.settle()
        self.assertTrue(combo.isEnabled())
        self.assertTrue(win._control_panel._agc_check.isEnabled())
        self.assertTrue(win._control_panel._gain_slider.parentWidget().isEnabled())


# --------------------------------------------------------------------------- #
# Front end: ppm, direct sampling, bias tee
# --------------------------------------------------------------------------- #
class TestRtlFrontEnd(_DeviceCase):
    def test_rows_show_for_the_rtl_sdr_only(self):
        p = self.panel
        rows = (p._ppm_spin, p._direct_combo, p._bias_check)
        self.assertEqual([self.visible(w) for w in rows], [False] * 3)  # demo
        self.rtl()
        self.assertEqual([self.visible(w) for w in rows], [True] * 3)
        self.hackrf()
        self.assertEqual([self.visible(w) for w in rows], [False, False, True])
        self.assertIn("3.3 V", p._bias_check.toolTip())
        self.assertIn("ppm", p._ppm_spin.toolTip())
        self.assertIn("calibrate", p._ppm_spin.toolTip())

    def test_ppm_is_applied_live_and_remembered(self):
        dev = self.rtl()
        spin = self.panel._ppm_spin
        spin.setValue(12)
        self.assertEqual(dev.freq_correction, 12)
        self.assertEqual(dev._device.corr, 12)
        self.assertEqual(self.store.values["device/rtlsdr/ppm"], 12)
        self.assertIn("+12 ppm", self.message())
        self.win._refresh_info_panel()
        self.assertIn("+12 ppm", self.win._info_panel.value("frontend"))

    def test_direct_sampling_choice_is_applied(self):
        dev = self.rtl()
        combo = self.panel._direct_combo
        self.assertEqual(combo.currentText(), "q")
        self.assertIn("24 MHz", combo.toolTip())
        self.assertTrue(self.win.set_frequency(7.1e6))
        self.assertEqual(dev._device.direct, 2)
        combo.setCurrentText("i")
        self.assertEqual(dev._device.direct, 1)
        self.assertIn("In use now", self.message())
        combo.setCurrentText("off")
        self.assertEqual(dev._device.direct, 0)
        self.assertEqual(self.store.values["device/rtlsdr/direct_sampling"], "off")
        self.win._refresh_info_panel()
        self.assertIn("HF: Off", self.win._info_panel.value("frontend"))

    def test_bias_tee_warns_and_is_switched_off_with_the_device(self):
        dev = self.rtl()
        handle = dev._device
        check = self.panel._bias_check
        self.assertFalse(self.visible(self.panel._bias_warning))
        check.setChecked(True)
        self.assertTrue(handle.bias)
        self.assertTrue(self.visible(self.panel._bias_warning))
        self.assertIn("antenna socket is powered", self.message())
        self.assertEqual(self.tone(), "warning")
        self.assertTrue(self.store.values["device/rtlsdr/bias_tee"])
        self.win._disconnect_device()
        self.assertFalse(handle.bias)

    def test_a_refused_change_goes_back_and_says_so(self):
        dev = self.rtl()
        FakeRtlSdr.fail = {"set_bias_tee"}
        self.panel._bias_check.setChecked(True)
        self.assertFalse(self.panel._bias_check.isChecked())
        self.assertFalse(dev.state.bias_tee_enabled)
        self.assertIn("didn't switch its bias tee on", self.message())
        self.assertEqual(self.tone(), "danger")
        self.assertNotIn("device/rtlsdr/bias_tee", self.store.values)

    def test_the_front_end_is_restored_on_connect(self):
        self.store.values.update(
            {
                "device/rtlsdr/ppm": -7,
                "device/rtlsdr/direct_sampling": "i",
                "device/rtlsdr/bias_tee": True,
            }
        )
        self.win.set_frequency(7.1e6)
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        dev = RTLSDRDevice()
        self.assertTrue(dev.open(0))
        note = self.connect(dev)
        handle = dev._device
        self.assertEqual(handle.corr, -7)
        self.assertEqual(handle.direct, 1)  # tuned at 7.1 MHz: the I input
        self.assertTrue(handle.bias)
        p = self.panel
        self.assertEqual(
            (
                p._ppm_spin.value(),
                p._direct_combo.currentText(),
                p._bias_check.isChecked(),
            ),
            (-7, "i", True),
        )
        self.assertIn("The bias tee is on", note)

    def test_hackrf_bias_tee(self):
        dev, fake = self.hackrf()
        self.panel._bias_check.setChecked(True)
        self.assertTrue(fake.antenna)
        self.assertTrue(self.store.values["device/hackrf/bias_tee"])
        self.panel._bias_check.setChecked(False)
        self.assertFalse(fake.antenna)


class TestConnectDialog(_DeviceCase):
    """tl:TOOLS-13, fg:FG-U05: no dead controls; the settings each device
    has, with what it used last."""

    def dialog(self):
        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = DeviceDialog(self.win)
        self.addCleanup(dialog.deleteLater)
        for kind, label, name in (
            ("rtlsdr", "RTL-SDR", "RTL-SDR #0"),
            ("hackrf", "HackRF", "HackRF One"),
        ):
            dialog._add_device(label, name, "0001", "Available")
            dialog._devices.append({"type": kind, "index": 0, "info": {}})
        return dialog

    def select(self, dialog, kind):
        for row, entry in enumerate(dialog._devices):
            if entry and entry.get("type") == kind:
                dialog._device_table.selectRow(row)
                return
        raise AssertionError(kind)

    def shown(self, dialog):
        form = dialog._settings_form
        return [
            form.isRowVisible(w)
            for w in (dialog._ppm_spin, dialog._direct_combo, dialog._bias_check)
        ]

    def test_rows_per_device(self):
        dialog = self.dialog()
        self.select(dialog, "demo")
        self.assertEqual(self.shown(dialog), [False, False, False])
        self.select(dialog, "rtlsdr")
        self.assertEqual(self.shown(dialog), [True, True, True])
        for field in (dialog._ppm_spin, dialog._direct_combo, dialog._bias_check):
            self.assertTrue(field.isEnabled())
            self.assertNotIn("yet", field.toolTip())
        self.select(dialog, "hackrf")
        self.assertEqual(self.shown(dialog), [False, False, True])

    def test_preselects_what_each_type_used_last(self):
        from sdr_module.gui.device_settings import FrontEnd

        dialog = self.dialog()
        dialog.set_preferences(
            {"rtlsdr": 1.4e6, "hackrf": 8e6, "demo": 2.4e6},
            {"rtlsdr": FrontEnd(ppm=5, direct_sampling="off", bias_tee=True)},
        )
        self.select(dialog, "rtlsdr")
        self.assertEqual(dialog._rate_combo.currentData(), 1.4e6)
        self.assertEqual(dialog._ppm_spin.value(), 5)
        self.assertEqual(dialog._direct_combo.currentData(), "off")
        self.assertTrue(dialog._bias_check.isChecked())
        self.select(dialog, "hackrf")
        self.assertEqual(dialog._rate_combo.currentData(), 8e6)
        dialog._ppm_spin.setValue(0)  # (hidden for a HackRF)
        self.select(dialog, "rtlsdr")
        self.assertEqual(dialog._ppm_spin.value(), 5)  # kept while switching
        kind, front = dialog.front_end_choice()
        self.assertEqual((kind, front.ppm, front.bias_tee), ("rtlsdr", 5, True))

    def test_connecting_applies_and_remembers_the_choice(self):
        from sdr_module.gui import device_dialog

        def fake_exec(dialog):
            dialog._add_device("RTL-SDR", "RTL-SDR #0", "0001", "Available")
            dialog._devices.append({"type": "rtlsdr", "index": 0, "info": {}})
            dialog._device_table.selectRow(dialog._device_table.rowCount() - 1)
            dialog._ppm_spin.setValue(9)
            dialog._bias_check.setChecked(True)
            dialog._rate_combo.setCurrentIndex(dialog._rate_combo.findData(1.8e6))
            dialog._on_accept()
            return dialog.result()

        with mock.patch.object(device_dialog.DeviceDialog, "exec", fake_exec):
            self.assertTrue(self.win._show_device_dialog())
        dev = self.win._device
        self.assertEqual(type(dev).__name__, "RTLSDRDevice")
        self.assertEqual(dev.freq_correction, 9)
        self.assertTrue(dev._device.bias)
        self.assertEqual(dev.state.sample_rate, 1.8e6)
        self.assertEqual(self.store.values["device/rtlsdr/ppm"], 9)
        self.assertTrue(self.store.values["device/rtlsdr/bias_tee"])
        self.assertEqual(self.panel._ppm_spin.value(), 9)
        self.assertEqual(self.store.values["device/rtlsdr/sample_rate"], 1.8e6)


# --------------------------------------------------------------------------- #
# Device menu
# --------------------------------------------------------------------------- #
class TestDemoAction(_DeviceCase):
    """mc:MC-U16"""

    def test_ticked_while_the_demo_is_in_use_and_available_with_hardware(self):
        action = self.win._demo_action
        self.assertTrue(action.isCheckable())
        self.assertTrue(action.isChecked())
        self.assertFalse(action.isEnabled())
        dev = self.rtl()
        self.assertFalse(action.isChecked())
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.assertTrue(self.win._demo_mode)
        self.assertTrue(action.isChecked())
        self.assertFalse(dev.is_open)  # the RTL-SDR was released

    def test_asks_first_when_a_recording_runs(self):
        dev = self.rtl()
        win = self.win
        win._start_acquisition()
        win._toggle_recording(True)
        self.assertTrue(win._recording)
        boxes = []

        def answer(choice):
            def run(box):
                boxes.append(box)
                if isinstance(box, QMessageBox):
                    for button in box.buttons():
                        if button.text().replace("&", "") == choice:
                            button.click()
                return 0

            return run

        win._exec_dialog = answer("Cancel")
        win._demo_action.trigger()
        self.assertIs(win._device, dev)
        self.assertTrue(win._recording)
        self.assertFalse(win._demo_action.isChecked())
        self.assertIn("switch from RTL-SDR #0 to the demo device", boxes[0].text())
        win._exec_dialog = answer("Stop and Switch")
        win._demo_action.trigger()
        self.assertTrue(win._demo_mode)
        self.assertFalse(win._recording)
        self.assertTrue(win._demo_action.isChecked())


class TestScanForDevices(_DeviceCase):
    """mc:MC-U12"""

    def scan(self, choice="", missing=True):
        from sdr_module.gui import main_window

        boxes = []

        def run(box):
            boxes.append(box)
            box.choice = choice
            return 1 if choice else 0

        self.win._exec_dialog = run
        with mock.patch.object(
            main_window, "_driver_importable", return_value=not missing
        ):
            self.win._hardware_classes = []
            with mock.patch.object(type(self.win), "_scan_hardware", return_value=[]):
                self.win._refresh_devices()
        return boxes[0]

    def test_shows_the_install_commands_selectable_and_monospace(self):
        box = self.scan()
        self.assertEqual(box.windowTitle(), "No Devices Found")
        text = box.commands_text()
        self.assertIn('RTL-SDR:  python -m pip install "sdr-module[rtlsdr]"', text)
        self.assertIn('python -m pip install "sdr-module[hackrf]"', text)
        self.assertIn("libhackrf", text)
        commands = box._commands
        self.assertTrue(commands.isReadOnly())
        self.assertEqual(commands.property("role"), "terminal")
        self.assertEqual(
            commands.lineWrapMode(), commands.LineWrapMode.NoWrap
        )  # never broken across lines
        texts = [
            b.text().replace("&", "") for b in box.findChildren(type(box._connect_btn))
        ]
        self.assertIn("Use Demo Device", texts)
        self.assertIn("Connect...", texts)

    def test_the_same_commands_as_the_connect_dialog(self):
        from sdr_module.gui.device_dialog import install_command, missing_drivers

        box = self.scan()
        text = box.commands_text()
        for _label, install in missing_drivers():
            # The command whole on its line, a note (libhackrf) under it.
            command, _sep, note = install.partition(" (")
            self.assertIn(command + "\n", text + "\n")
            if note:
                self.assertIn("(" + note, text)
        self.assertEqual(
            install_command("rtlsdr"), 'python -m pip install "sdr-module[rtlsdr]"'
        )

    def test_use_demo_device_from_the_box(self):
        dev = self.rtl()
        self.scan("demo")
        self.assertTrue(self.win._demo_mode)
        self.assertFalse(dev.is_open)

    def test_connect_from_the_box(self):
        with mock.patch.object(type(self.win), "_show_device_dialog") as connect:
            self.scan("connect")
        connect.assert_called_once()

    def test_devices_found(self):
        boxes = []

        def run(box):
            boxes.append(box)
            return 0

        self.win._exec_dialog = run
        with mock.patch.object(
            type(self.win), "_scan_hardware", return_value=[("k", "RTL-SDR #0")]
        ):
            self.win._refresh_devices()
        box = boxes[0]
        self.assertEqual(box.windowTitle(), "Devices Found")
        self.assertIn("RTL-SDR #0", box._summary.text())
        self.assertEqual(box.commands_text(), "")


if __name__ == "__main__":
    unittest.main()
