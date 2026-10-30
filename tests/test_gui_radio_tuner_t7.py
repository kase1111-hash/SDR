"""The AM/FM radio tuner (T7): presets kept between sessions and named,
even preset buttons, the receiver's exact frequency between channels, a
STEREO lamp that follows a real 19 kHz pilot, storing refused off-band, no
made-up station names, the AM band explained per device, real seek, and the
demo's AM broadcast stations."""

from __future__ import annotations

import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _fm(composite_hz, rate=480e3, seconds=0.2, deviation=75e3, snr_db=40.0, seed=1):
    """Complex FM of a composite of tones {freq: share of the deviation}."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    audio = sum(share * np.sin(2 * np.pi * f * t) for f, share in composite_hz.items())
    phase = 2 * np.pi * deviation * np.cumsum(audio) / rate
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (
        10 ** (-snr_db / 20) / np.sqrt(2)
    )
    return (np.exp(1j * phase) + noise).astype(np.complex64)


class TestStereoPilotDetector(unittest.TestCase):
    """TOOLS-11: STEREO from a measured 19 kHz pilot."""

    def feed(self, detector, signal, blocks=20):
        state = False
        for _ in range(blocks):
            state = detector.process(signal)
        return state

    def test_lights_on_a_pilot_and_not_without(self):
        from sdr_module.dsp.stereo_pilot import StereoPilotDetector

        program = {1500.0: 0.4, 6000.0: 0.2, 12000.0: 0.15}
        stereo = _fm({**program, 19000.0: 0.09})
        mono = _fm(program)
        detector = StereoPilotDetector(480e3)
        self.assertTrue(self.feed(detector, stereo))
        self.assertGreater(detector.ratio_db, 20)
        self.assertFalse(self.feed(detector, mono))  # goes off again
        detector.reset()
        self.assertFalse(self.feed(detector, mono))
        noise_only = _fm({}, deviation=0, snr_db=-10)
        detector.reset()
        self.assertFalse(self.feed(detector, noise_only))

    def test_it_does_not_flicker(self):
        from sdr_module.dsp.stereo_pilot import StereoPilotDetector

        stereo = _fm({1500.0: 0.4, 19000.0: 0.09})
        mono = _fm({1500.0: 0.4})
        detector = StereoPilotDetector(480e3)
        self.feed(detector, stereo)
        # One block without the pilot (a fade) doesn't turn it off.
        self.assertTrue(detector.process(mono[:1000]))
        self.assertTrue(detector.process(stereo))

    def test_too_narrow_a_channel_is_never_stereo(self):
        from sdr_module.dsp.stereo_pilot import StereoPilotDetector

        detector = StereoPilotDetector(40e3)
        self.assertFalse(detector.usable)
        self.assertFalse(detector.process(_fm({19000.0: 0.09}, rate=40e3)))

    def test_cheap(self):
        from sdr_module.dsp.stereo_pilot import StereoPilotDetector

        block = _fm({1500.0: 0.4, 19000.0: 0.09}, seconds=0.034)
        detector = StereoPilotDetector(480e3)
        detector.process(block)
        t0 = time.perf_counter()
        for _ in range(30):
            detector.process(block)
        per_block = (time.perf_counter() - t0) / 30
        self.assertLess(per_block, 0.003)  # well under a frame's 33 ms


class _TunerCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui import radio_tuner

        self.rt = radio_tuner
        self.tuner = radio_tuner.RadioTunerWidget(None, 2.4e6)
        self.emitted = []
        self.tuner.frequency_changed.connect(lambda f, b: self.emitted.append((f, b)))
        self.changes = []
        self.tuner.presets_changed.connect(lambda: self.changes.append(True))

    def tearDown(self):
        if hasattr(self, "tuner"):
            self.tuner.close()
            self.tuner.deleteLater()
            _settle()


class TestPresets(_TunerCase):
    def test_defaults_are_frequencies_without_made_up_names(self):
        """UX-U17: no "Rock", "NPR" or "News" stations that don't exist."""
        for preset in self.tuner.DEFAULT_FM_PRESETS + self.tuner.DEFAULT_AM_PRESETS:
            self.assertEqual(preset.name, "")
        self.assertEqual(self.tuner._preset_buttons[0].text(), "&1\n101.1 MHz")

    def test_rename_and_clear(self):
        """UX-11: presets can be named."""
        t = self.tuner
        t.rename_preset(1, "  Classic   rock ")
        self.assertEqual(t._preset_buttons[1].text(), "&2  Classic rock\n93.3 MHz")
        self.assertEqual(t.get_presets()["fm"][1], [93.3e6, "Classic rock"])
        t._preset_buttons[1].click()
        self.assertEqual(t._freq_display._station, "Preset 2 · Classic rock")
        t.rename_preset(1, "")
        self.assertEqual(t._preset_buttons[1].text(), "&2\n93.3 MHz")
        t.clear_preset(1)
        self.assertEqual(t._preset_buttons[1].text(), "&2\nEmpty")
        self.assertIsNone(t.get_presets()["fm"][1])
        self.assertEqual(len(self.changes), 3)

    def test_even_heights(self):
        """UX-11: empty, plain and named presets are equally tall."""
        t = self.tuner
        t.rename_preset(0, "A long preset name")
        t.clear_preset(2)
        t.show()
        _settle()
        heights = {b.height() for b in t._preset_buttons}
        self.assertEqual(len(heights), 1, heights)
        self.assertTrue(all("\n" in b.text() for b in t._preset_buttons))

    def test_storing_keeps_the_name_only_for_the_same_station(self):
        t = self.tuner
        t.rename_preset(0, "Home")
        t.store_preset(0)  # 101.1 MHz again
        self.assertEqual(t.get_presets()["fm"][0], [101.1e6, "Home"])
        t._seek_up_btn.click()
        t.store_preset(0)
        self.assertEqual(t.get_presets()["fm"][0], [101.2e6, ""])

    def test_set_presets_skips_what_it_cant_use(self):
        t = self.tuner
        t.set_presets(
            {
                "fm": [[97.5e6, "Mine"], None, ["x"], [150e6, "out of band"]],
                "am": [[1500e3, "MW"]],
            }
        )
        fm = t.get_presets()["fm"]
        self.assertEqual(fm[0], [97.5e6, "Mine"])
        self.assertIsNone(fm[1])
        self.assertEqual(fm[2], [97.1e6, ""])  # unreadable: default kept
        self.assertEqual(fm[3], [104.3e6, ""])  # outside FM: default kept
        self.assertEqual(t.get_presets()["am"][0], [1500e3, "MW"])
        t.set_presets("garbage")  # ignored

    def test_storing_off_band_is_refused(self):
        """TOOLS-12: no hidden station stored while the receiver is elsewhere."""
        t = self.tuner
        t.set_frequency(1090e6)
        self.assertTrue(t.is_off_band())
        before = t.get_presets()
        t.store_preset(3)
        self.assertEqual(t.get_presets(), before)
        self.assertEqual(
            t._hint_label.text(),
            "Tune to an AM or FM station first: the receiver is on 1090.000 MHz.",
        )
        self.assertEqual(t._hint_label.property("tone"), "warning")
        self.assertEqual(self.changes, [])
        self.assertIn("Tune to an AM or FM station", t._store_blocker())
        t.set_frequency(100.1e6)
        self.assertIsNone(t._store_blocker())


class TestExactFrequency(_TunerCase):
    def test_between_channels_shows_the_receivers_frequency(self):
        """TOOLS-10: 100.137 MHz is not shown as 100.1 MHz."""
        t = self.tuner
        t.set_frequency(100.137e6)
        self.assertEqual(t._freq_display._digits(), ("100.137", "MHz"))
        self.assertEqual(t._freq_display._station, "Between channels")
        t.set_frequency(100.15e6)
        self.assertEqual(t._freq_display._digits(), ("100.150", "MHz"))
        t.store_preset(4)
        self.assertEqual(t._preset_buttons[4].text(), "&5\n100.150 MHz")
        self.assertIn("Stored 100.150 MHz in preset 5", t._hint_label.text())
        t.set_frequency(1053.5e3)
        self.assertEqual(t._freq_display._digits(), ("1053.5", "kHz"))
        t.set_frequency(100.1e6)
        self.assertEqual(t._freq_display._digits(), ("100.1", "MHz"))
        self.assertEqual(
            self.rt.format_station(1010e3, self.rt.RadioBand.AM), "1010 kHz"
        )

    def test_stepping_from_between_channels_lands_on_the_next_one(self):
        t = self.tuner
        t.set_frequency(100.137e6)
        t._tuning_dial.step(1)
        self.assertAlmostEqual(t.get_frequency(), 100.2e6)
        t.set_frequency(100.137e6)
        t._tuning_dial.step(-1)
        self.assertAlmostEqual(t.get_frequency(), 100.1e6)


class TestAmBandPerDevice(_TunerCase):
    def test_hackrf_reaches_only_part_of_the_band(self):
        t = self.tuner
        t.set_device_range(1e6, 6e9, "HackRF One")
        t._am_btn.click()
        self.assertTrue(t._am_btn.isEnabled())
        # The band opens on a station the HackRF can tune (the last AM
        # station, 880 kHz, can't be): the first reachable preset.
        self.assertEqual(t.get_band(), self.rt.RadioBand.AM)
        self.assertEqual(t.get_frequency(), 1010e3)
        self.assertEqual(self.emitted[-1], (1010e3, "AM"))
        reachable = {
            b.get_preset().frequency_hz: b.isEnabled() for b in t._preset_buttons
        }
        self.assertEqual(
            reachable,
            {
                880e3: False,
                1010e3: True,
                770e3: False,
                1050e3: True,
                660e3: False,
                1260e3: True,
            },
        )
        self.assertIn("HackRF One can't tune it", t._preset_buttons[0].toolTip())
        self.assertIn("HackRF One tunes from 1000 kHz", t._hint_label.text())
        self.assertIn("HackRF One tunes from 1000 kHz", t._am_btn.toolTip())

    def test_a_device_without_mw_disables_am_and_says_why(self):
        t = self.tuner
        t._am_btn.click()
        t.set_device_range(24e6, 1766e6, "R820T dongle")
        self.assertFalse(t._am_btn.isEnabled())
        self.assertIn("out of its reach", t._am_btn.toolTip())
        self.assertEqual(t.get_band(), self.rt.RadioBand.FM)

    def test_rtl_sdr_note_and_demo_device(self):
        t = self.tuner
        t.set_device_range(0.5e6, 1766e6, "RTL-SDR")
        self.assertIn("direct sampling", t._am_btn.toolTip())
        t._am_btn.click()
        self.assertIn("direct sampling", t._hint_label.text())
        t.set_device_range(None, None, "Demo Device")
        self.assertNotIn("direct sampling", t._am_btn.toolTip())
        self.assertTrue(all(b.isEnabled() for b in t._preset_buttons))


class TestSeekWithoutReceiver(_TunerCase):
    def test_no_receiver_steps_one_channel(self):
        start = self.tuner.get_frequency()
        self.tuner._seek_up_btn.click()
        self.assertAlmostEqual(self.tuner.get_frequency(), start + 100e3)
        self.assertIn("one channel", self.tuner._seek_up_btn.toolTip())

    def test_receiver_stopped_says_so(self):
        self.tuner.set_signal_probe(lambda: None)
        start = self.tuner.get_frequency()
        self.tuner._seek_up_btn.click()
        self.assertEqual(self.tuner.get_frequency(), start)
        self.assertIn("start the receiver first", self.tuner._hint_label.text())
        self.assertIn("Seek up", self.tuner._seek_up_btn.toolTip())

    def test_seek_steps_until_a_signal_and_shift_steps_once(self):
        t = self.tuner
        stations = {101.4e6}
        t.set_signal_probe(
            lambda: "signal" if round(t.get_frequency()) in stations else "quiet"
        )
        t.show()
        t._seek_up_btn.click()  # from 101.1 MHz
        self.assertTrue(t.is_seeking())
        deadline = time.monotonic() + 3
        while t.is_seeking() and time.monotonic() < deadline:
            _settle()
            time.sleep(0.005)
        self.assertFalse(t.is_seeking())
        self.assertAlmostEqual(t.get_frequency(), 101.4e6)
        self.assertEqual(t._hint_label.text(), "Found a station on 101.4 MHz.")
        self.assertEqual(
            [round(f) for f, _b in self.emitted][-3:], [101200000, 101300000, 101400000]
        )
        QTest.mouseClick(
            t._seek_up_btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertAlmostEqual(t.get_frequency(), 101.5e6)
        self.assertFalse(t.is_seeking())

    def test_seek_goes_round_the_band_once_and_can_be_stopped(self):
        t = self.tuner
        t.set_signal_probe(lambda: "quiet")
        t.show()
        t._am_btn.click()
        t._seek_down_btn.click()
        self.assertTrue(t.is_seeking())
        t._seek_down_btn.click()  # again: stop
        self.assertFalse(t.is_seeking())
        self.assertEqual(t._hint_label.text(), "Seek stopped.")
        t._seek_down_btn.click()
        deadline = time.monotonic() + 10
        while t.is_seeking() and time.monotonic() < deadline:
            _settle()
        self.assertIn("No station strong enough", t._hint_label.text())


class TestTunerInTheWindow(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")

    def tearDown(self):
        tuner = getattr(self.win, "_radio_tuner", None) if HAS_PYQT6 else None
        if tuner is not None:
            tuner.close()
        super().tearDown()

    def run_receiver(self, seconds, until=None):
        """Drive the display loop (its timer is stopped in these tests)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.win._update_display()
            QApplication.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.02)
        return until() if until is not None else True

    def test_presets_survive_a_restart(self):
        """TOOLS-08/UX-B09."""
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.win.set_frequency(100.1e6)
        tuner.store_preset(5)
        self.win.set_frequency(97.1e6)
        tuner.store_preset(1)
        tuner.rename_preset(1, "Morning")
        self.win._persist_state()
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        other._show_radio_tuner()
        restored = other._radio_tuner.get_presets()["fm"]
        self.assertEqual(restored[5], [100.1e6, ""])
        self.assertEqual(restored[1], [97.1e6, "Morning"])
        other._radio_tuner.close()

    def test_stereo_follows_the_pilot(self):
        """TOOLS-11: the demo's 100.1 MHz station sends a pilot."""
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.win.set_frequency(100.1e6)
        self.assertTrue(self.run_receiver(4, lambda: tuner._stereo))
        # Not in AM (no pilot is looked for), nor once the receiver stops.
        tuner.frequency_changed.emit(1010e3, "AM")
        self.run_receiver(0.3)
        self.assertFalse(tuner._stereo)
        tuner.frequency_changed.emit(100.1e6, "FM")
        self.assertTrue(self.run_receiver(4, lambda: tuner._stereo))
        self.win._stop_acquisition()
        self.assertFalse(tuner._stereo)

    def test_seek_finds_the_next_demo_station(self):
        """UX-U17: the arrows seek until a station opens the squelch."""
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.win.set_frequency(101.3e6)
        self.run_receiver(0.5)
        tuner._seek_up_btn.click()
        self.assertTrue(tuner.is_seeking())
        self.assertTrue(self.run_receiver(15, lambda: not tuner.is_seeking()))
        self.assertAlmostEqual(self.win._current_frequency(), 102.7e6)
        self.assertEqual(tuner._hint_label.text(), "Found a station on 102.7 MHz.")

    def test_demo_am_stations_on_the_presets(self):
        """UX-12 (AM): every AM preset has a station in demo mode."""
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        tuner._am_btn.click()
        levels = {}
        for button in tuner._preset_buttons:
            button.click()
            self.assertEqual(self.win._control_panel.get_demod_mode(), "AM")
            self.run_receiver(0.4)
            levels[button.get_preset().frequency_hz] = self.win._tuner_signal
        self.assertTrue(all(levels.values()), levels)

    def test_device_range_reaches_the_tuner(self):
        from tests.test_gui_tuning import _hackrf

        self.win._release_device()
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.win._device = _hackrf()
        self.win._refresh_state_ui()
        self.assertIn("HackRF One tunes from 1000 kHz", tuner._am_note())
        # AM opens where the HackRF can tune (not refused, back on FM).
        tuner._am_btn.click()
        self.assertEqual(self.win._current_frequency(), 1010e3)
        self.assertEqual(self.win._control_panel.get_demod_mode(), "AM")
        self.assertEqual(tuner.get_band().value, "AM")
        self.win._device = None
        self.win._refresh_state_ui()
        self.assertEqual(tuner._am_note(), "")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDemoReadCost(unittest.TestCase):
    def test_am_band_reads_stay_fast(self):
        from sdr_module.gui.device_dialog import MockDevice

        dev = MockDevice()
        dev.set_frequency(960e3)  # all six AM stations in view
        dev.start_rx()
        n = 81920
        dev.read_samples(n)  # build the loops
        t0 = time.perf_counter()
        for _ in range(20):
            dev.read_samples(n)
        per_sample = (time.perf_counter() - t0) / (20 * n)
        self.assertLess(per_sample, 0.2e-6)


if __name__ == "__main__":
    unittest.main()
