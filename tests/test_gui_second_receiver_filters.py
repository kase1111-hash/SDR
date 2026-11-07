"""M-R2: the Second Receiver's own audio notch and low cut.

They existed only in the main window (its Audio group). The Second
Receiver window now has a Notch (Off, Auto, Manual at a frequency), the
Whistle row under it (Manual's frequency, or what Auto removes, as in the
main window) and a Low cut (Off, 100, 200, 300 Hz), with the main window's
behaviour: the same DSP (the receiver chain's ``ListeningFilters`` through
its options), greyed with the reason where they don't apply, saved under
``device/second_receiver/*`` (a damaged value is logged and replaced).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_fm_stereo import _tone  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

RATE = 2.4e6
BLOCK = 80_000
AUDIO = 48000.0


def _usb(seconds, tones, rate=RATE, seed=4):
    """USB at 0 Hz: each ``(audio_hz, level)`` a carrier that far above
    the tuned frequency (heard as that tone), in a little noise."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    iq = sum(level * np.exp(2j * np.pi * hz * t) for hz, level in tones)
    rng = np.random.default_rng(seed)
    iq = iq + 1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return iq.astype(np.complex64)


def _am(seconds, tones, rate=RATE):
    """AM at 0 Hz modulated by ``(audio_hz, depth)`` tones."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    envelope = 1.0 + sum(depth * np.sin(2 * np.pi * hz * t) for hz, depth in tones)
    return (0.3 * envelope).astype(np.complex64)


def _db(a, b):
    return 20 * np.log10(max(a, 1e-12) / max(b, 1e-12))


class _DspCase(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")

    @staticmethod
    def dsp(mode="USB", bandwidth=2.7e3, deviation=5e3):
        from sdr_module.gui.second_receiver import ReceiverDsp

        dsp = ReceiverDsp()
        dsp.rate = RATE
        dsp.mode = mode
        dsp.bandwidth_hz = bandwidth
        dsp.deviation_hz = deviation
        dsp.squelch.enabled = False  # (everything plays)
        return dsp

    @staticmethod
    def listen(dsp, iq):
        frames = [
            dsp.process(iq[start : start + BLOCK], True)
            for start in range(0, len(iq), BLOCK)
        ]
        audio = np.concatenate(
            [f.audio for f in frames if f.audio is not None and len(f.audio)]
        )
        return frames, audio


class TestTheNotch(_DspCase):
    def test_manual_takes_the_whistle_out_and_leaves_the_rest(self):
        iq = _usb(1.2, [(1000.0, 0.05), (600.0, 0.02)])
        _frames, before = self.listen(self.dsp(), iq)
        dsp = self.dsp()
        dsp.options.notch = "manual"
        dsp.options.notch_hz = 1000.0
        frames, after = self.listen(dsp, iq)
        tail = slice(int(0.6 * AUDIO), None)
        whistle = _db(_tone(before[tail], 1000, AUDIO), _tone(after[tail], 1000, AUDIO))
        # (The AGC brings what is left up: the ratio to the rest counts.)
        rest_before = _tone(before[tail], 600, AUDIO) / _tone(before[tail], 1000, AUDIO)
        rest_after = _tone(after[tail], 600, AUDIO) / _tone(after[tail], 1000, AUDIO)
        self.assertGreater(_db(rest_after, rest_before), 40.0)
        self.assertGreater(whistle, 0.0)
        self.assertEqual(frames[-1].notch_hz, 1000.0)

    def test_auto_finds_the_whistle_and_says_where(self):
        iq = _usb(2.0, [(1350.0, 0.05), (700.0, 0.01)])
        dsp = self.dsp()
        dsp.options.notch = "auto"
        frames, audio = self.listen(dsp, iq)
        found = frames[-1].notch_hz
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found, 1350.0, delta=1.0)
        tail = audio[-int(0.5 * AUDIO) :]
        self.assertGreater(_db(_tone(tail, 700, AUDIO), _tone(tail, 1350, AUDIO)), 20.0)

    def test_off_and_nothing_heard_say_none(self):
        dsp = self.dsp()
        frames, _audio = self.listen(dsp, _usb(0.3, [(1000.0, 0.05)]))
        self.assertIsNone(frames[-1].notch_hz)

    def test_another_station_tuned_searches_anew(self):
        # Auto forgets the last station's whistle once another is tuned.
        dsp = self.dsp()
        dsp.options.notch = "auto"
        dsp.tuned_hz = 14.2e6
        frames, _audio = self.listen(dsp, _usb(1.5, [(1350.0, 0.05)]))
        self.assertIsNotNone(frames[-1].notch_hz)
        dsp.tuned_hz = 14.25e6
        frames, _audio = self.listen(dsp, _usb(0.1, [(900.0, 0.01)], seed=5))
        self.assertIsNone(frames[-1].notch_hz)


class TestTheLowCut(_DspCase):
    def test_hum_goes_the_voice_stays(self):
        iq = _am(1.0, [(60.0, 0.3), (1000.0, 0.3)])
        _frames, before = self.listen(self.dsp("AM", 10e3), iq)
        dsp = self.dsp("AM", 10e3)
        dsp.options.low_cut_hz = 200.0
        _frames, after = self.listen(dsp, iq)
        tail = slice(int(0.5 * AUDIO), None)
        self.assertGreater(
            _db(_tone(before[tail], 60, AUDIO), _tone(after[tail], 60, AUDIO)), 30.0
        )
        self.assertLess(
            abs(_db(_tone(before[tail], 1000, AUDIO), _tone(after[tail], 1000, AUDIO))),
            0.5,
        )


class TestBroadcastFmIsLeftAlone(_DspCase):
    def test_notch_and_low_cut_dont_touch_it(self):
        n = int(0.5 * RATE)
        t = np.arange(n) / RATE
        phase = 2 * np.pi * 75e3 * np.cumsum(0.5 * np.sin(2 * np.pi * 1000 * t)) / RATE
        iq = np.exp(1j * phase).astype(np.complex64)
        _frames, plain = self.listen(self.dsp("FM", 200e3, 75e3), iq)
        dsp = self.dsp("FM", 200e3, 75e3)
        dsp.options.notch = "manual"
        dsp.options.notch_hz = 1000.0
        dsp.options.low_cut_hz = 300.0
        frames, filtered = self.listen(dsp, iq)
        np.testing.assert_array_equal(filtered, plain)
        self.assertIsNone(frames[-1].notch_hz)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheControls(_SecondCase):
    def test_notch_whistle_and_low_cut_rows(self):
        rx = self.rx
        rx.set_mode("USB")
        notch, low_cut = rx._notch_combo, rx._low_cut_combo
        self.assertEqual(
            [notch.itemText(i) for i in range(notch.count())], ["Off", "Auto", "Manual"]
        )
        self.assertEqual(
            [low_cut.itemText(i) for i in range(low_cut.count())],
            ["Off", "100 Hz", "200 Hz", "300 Hz"],
        )
        for combo in (notch, low_cut):
            self.assertTrue(combo.isEnabled())
            self.assertTrue(combo.toolTip())
            self.assertTrue(combo.accessibleName())
            for i in range(combo.count()):
                self.assertTrue(combo.itemData(i, 3))  # (a tooltip each)
        form = rx._audio_form
        self.assertFalse(form.isRowVisible(rx._notch_row))  # (Off)
        rx._notch_combo.setCurrentIndex(2)  # Manual: its frequency
        self.assertTrue(form.isRowVisible(rx._notch_row))
        self.assertIs(rx._notch_stack.currentWidget(), rx._notch_spin)
        self.assertTrue(rx._notch_spin.isEnabled())
        self.assertEqual(rx._notch_spin.minimum(), 200)
        self.assertEqual(rx._notch_spin.maximum(), 5000)
        self.assertTrue(rx._notch_spin.toolTip())
        rx._notch_spin.setValue(1234)
        self.assertEqual(rx._dsp.options.notch, "manual")
        self.assertEqual(rx._dsp.options.notch_hz, 1234.0)
        rx._notch_combo.setCurrentIndex(1)  # Auto: what it removes
        self.assertIs(rx._notch_stack.currentWidget(), rx._notch_found)
        self.assertEqual(rx._notch_found.text(), "None")
        self.assertEqual(rx._dsp.options.notch, "auto")
        rx._low_cut_combo.setCurrentIndex(2)
        self.assertEqual(rx._dsp.options.low_cut_hz, 200.0)

    def test_greyed_where_they_dont_apply_saying_why(self):
        rx = self.rx
        rx.set_mode("FM", "75 kHz", "200 kHz")
        self.assertFalse(rx._notch_combo.isEnabled())
        self.assertIn("broadcast FM", rx._notch_combo.toolTip())
        self.assertFalse(rx._low_cut_combo.isEnabled())
        self.assertIn("broadcast FM", rx._low_cut_combo.toolTip())
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(rx._notch_combo.isEnabled())
        self.assertTrue(rx._low_cut_combo.isEnabled())
        rx.set_mode("CW")
        self.assertTrue(rx._notch_combo.isEnabled())
        self.assertFalse(rx._low_cut_combo.isEnabled())
        self.assertIn("CW", rx._low_cut_combo.toolTip())
        rx.set_mode("None (I/Q)")
        self.assertFalse(rx._notch_combo.isEnabled())
        self.assertIn("no audio", rx._notch_combo.toolTip())
        self.assertFalse(rx._low_cut_combo.isEnabled())

    def test_saved_and_restored(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        rx = self.rx
        rx.set_mode("USB")
        rx._notch_combo.setCurrentIndex(2)
        rx._notch_spin.setValue(1500)
        rx._low_cut_combo.setCurrentIndex(3)
        values = self.store.values
        self.assertEqual(values["device/second_receiver/notch"], "manual")
        self.assertEqual(values["device/second_receiver/notch_hz"], 1500)
        self.assertEqual(values["device/second_receiver/low_cut"], "300")
        # (The main window's are its own.)
        self.assertNotIn("audio/notch", values)
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._notch_combo.currentData(), "manual")
        self.assertEqual(again._notch_spin.value(), 1500)
        self.assertEqual(again._low_cut_combo.currentData(), "300")
        self.assertEqual(again._dsp.options.notch, "manual")
        self.assertEqual(again._dsp.options.notch_hz, 1500.0)
        self.assertEqual(again._dsp.options.low_cut_hz, 300.0)

    def test_damaged_settings_are_logged_and_replaced(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.store.values.update(
            {
                "device/second_receiver/notch": "loud",
                "device/second_receiver/notch_hz": 99999,
                "device/second_receiver/low_cut": "40",
            }
        )
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._notch_combo.currentData(), "off")
        self.assertEqual(again._notch_spin.value(), 5000)  # (the nearest end)
        self.assertEqual(again._low_cut_combo.currentData(), "off")
        self.assertEqual(again._dsp.options.notch, "off")
        self.assertEqual(again._dsp.options.low_cut_hz, 0.0)
        text = "\n".join(logs.output)
        for key in ("notch", "notch_hz", "low_cut"):
            self.assertIn(f"device/second_receiver/{key}", text)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAutoSaysWhatItRemoves(_SecondCase):
    def test_the_whistle_row_shows_the_frequency_found(self):
        rx = self.rx
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        speaker = self.sink()
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_listen("second")
        rx.set_mode("USB")
        rx._squelch_check.setChecked(False)
        rx._notch_combo.setCurrentIndex(1)  # Auto
        rx._sync_audio_output()
        self.assertTrue(speaker.open)
        iq = _usb(2.0, [(1350.0, 0.05), (700.0, 0.01)], rate=rx._rate())
        block = int(rx._rate() / 30)
        for start in range(0, len(iq), block):
            rx._process(iq[start : start + block])
        found = rx._notch_found
        self.assertRegex(found.text(), r"^13(49|50|51) Hz$")
        self.assertEqual(found.property("tone"), "success")
        self.assertIn("1350", found.toolTip())
        # Manual shows its own frequency, not what Auto found.
        rx._notch_combo.setCurrentIndex(2)
        self.assertIs(rx._notch_stack.currentWidget(), rx._notch_spin)
        rx.stop()
        self.assertEqual(found.text(), "None")


if __name__ == "__main__":
    unittest.main()
