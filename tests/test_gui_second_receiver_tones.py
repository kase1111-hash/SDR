"""M-R3: the Second Receiver's own tone squelch (CTCSS tones, DCS codes).

It had none, so a repeater's shared channel couldn't be filtered there and
a bookmark with a tone, tuned there, played without it. Now it has the
main window's choices (Off, the 50 CTCSS tones, the 104 DCS codes in
normal then inverted polarity) and its Tone heard readout, in narrow FM
only (as there: at a broadcast deviation it isn't used), through the
same ``audio_chain.ToneSquelch``: a signal without the chosen tone or
code reads TONE on its LEVEL lamp and stays silent. Saved under
``device/second_receiver/*``; a bookmark tuned there applies its tone or
code as the main window does, and Bookmark This Frequency keeps it.
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


def _repeater(seconds=1.5, ctcss_hz=100.0, dcs=None, rate=RATE, voice_hz=1000.0):
    """Narrow FM at 0 Hz (5 kHz deviation): a voice tone and a sub-audible
    CTCSS tone (600 Hz deviation, as repeaters send it) or DCS code."""
    from sdr_module.dsp.dcs import dcs_waveform

    n = int(seconds * rate)
    t = np.arange(n) / rate
    deviation = 2.5e3 * np.sin(2 * np.pi * voice_hz * t)
    if dcs is not None:
        # (Made at 48 kHz and held: the code's bits are slow.)
        slow = dcs_waveform(dcs[0], dcs[1], AUDIO, samples=n // 50 + 1)
        deviation = deviation + 750.0 * np.repeat(slow, 50)[:n]
    elif ctcss_hz:
        deviation = deviation + 600.0 * np.sin(2 * np.pi * ctcss_hz * t)
    phase = 2 * np.pi * np.cumsum(deviation) / rate
    rng = np.random.default_rng(6)
    noise = 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (0.3 * np.exp(1j * phase) + noise).astype(np.complex64)


def _dsp(tone_hz=None, dcs=None, deviation=5e3, bandwidth=25e3):
    from sdr_module.gui.second_receiver import ReceiverDsp

    dsp = ReceiverDsp()
    dsp.rate = RATE
    dsp.mode = "FM"
    dsp.bandwidth_hz = bandwidth
    dsp.deviation_hz = deviation
    dsp.tone_squelch = (tone_hz, dcs)
    dsp.options.tone_squelch = tone_hz is not None or dcs is not None
    return dsp


def _listen(dsp, iq, play=True):
    frames = [
        dsp.process(iq[start : start + BLOCK], play)
        for start in range(0, len(iq), BLOCK)
    ]
    audio = [f.audio for f in frames if f.audio is not None and len(f.audio)]
    return frames, (np.concatenate(audio) if audio else np.zeros(0))


class _DspCase(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")


class TestCtcss(_DspCase):
    def test_the_right_tone_plays_without_its_tone(self):
        frames, audio = _listen(_dsp(tone_hz=100.0), _repeater())
        self.assertEqual(frames[-1].state, "open")
        tail = audio[-int(0.5 * AUDIO) :]
        self.assertGreater(_tone(tail, 1000, AUDIO), 0.2)
        # The 100 Hz tone is filtered out of what is heard (300 Hz cut).
        self.assertGreater(
            20 * np.log10(_tone(tail, 1000, AUDIO) / _tone(tail, 100, AUDIO)), 40
        )
        self.assertEqual(frames[-1].tone_heard, (100.0, None, True))

    def test_another_tone_reads_tone_and_stays_silent(self):
        frames, audio = _listen(_dsp(tone_hz=123.0), _repeater())
        self.assertEqual(frames[-1].state, "tone")
        tail = [f.audio for f in frames[-10:]]
        self.assertTrue(all(a is None or not len(a) for a in tail))
        self.assertEqual(frames[-1].tone_heard[0], 100.0)

    def test_the_tone_lamp_shows_with_the_speaker_off(self):
        # As in the main window: TONE whether the speaker plays or not.
        frames, _audio = _listen(_dsp(tone_hz=123.0), _repeater(), play=False)
        self.assertEqual(frames[-1].state, "tone")
        self.assertTrue(all(f.audio is None for f in frames))

    def test_without_a_tone_squelch_the_tone_heard_is_said(self):
        frames, audio = _listen(_dsp(), _repeater())
        self.assertEqual(frames[-1].state, "open")
        self.assertGreater(len(audio), 0.5 * AUDIO)
        self.assertEqual(frames[-1].tone_heard, (100.0, None, True))

    def test_not_used_at_a_broadcast_deviation(self):
        dsp = _dsp(tone_hz=123.0, deviation=75e3, bandwidth=200e3)
        frames, audio = _listen(dsp, _repeater())
        self.assertEqual(frames[-1].state, "open")
        self.assertGreater(len(audio), 0.5 * AUDIO)
        self.assertEqual(frames[-1].tone_heard, (None, None, False))

    def test_broadcast_fm_isnt_demodulated_for_a_tone_nobody_checks(self):
        # (Review: with a tone set, broadcast FM nobody listens to was
        # demodulated every block for a check it never makes.)
        dsp = _dsp(tone_hz=123.0, deviation=75e3, bandwidth=200e3)
        calls = []
        chain = dsp.chain()
        real = chain.demodulate
        chain.demodulate = lambda *a, **k: calls.append(1) or real(*a, **k)
        frames, _audio = _listen(dsp, _repeater(seconds=0.5), play=False)
        self.assertEqual(calls, [])
        self.assertEqual(frames[-1].tone_heard, (None, None, False))
        # Narrow FM still is (the TONE lamp, speaker off).
        narrow = _dsp(tone_hz=123.0)
        chain = narrow.chain()
        real_narrow = chain.demodulate
        chain.demodulate = lambda *a, **k: calls.append(1) or real_narrow(*a, **k)
        frames, _audio = _listen(narrow, _repeater(seconds=0.5), play=False)
        self.assertTrue(calls)
        self.assertEqual(frames[-1].state, "tone")

    def test_another_channel_is_heard_anew(self):
        # The last channel's tone must not keep the squelch open on one
        # without it.
        dsp = _dsp(tone_hz=100.0)
        dsp.tuned_hz = 145.29e6
        frames, _audio = _listen(dsp, _repeater())
        self.assertEqual(frames[-1].state, "open")
        dsp.tuned_hz = 145.31e6
        frames, _audio = _listen(dsp, _repeater(0.1, ctcss_hz=None))
        self.assertEqual(frames[0].state, "tone")


class TestDcs(_DspCase):
    def test_the_right_code_plays_another_doesnt(self):
        iq = _repeater(dcs=(23, False))
        frames, audio = _listen(_dsp(dcs=(23, False)), iq)
        self.assertEqual(frames[-1].state, "open")
        self.assertGreater(_tone(audio[-int(0.4 * AUDIO) :], 1000, AUDIO), 0.2)
        self.assertEqual(frames[-1].tone_heard, (None, (23, False), True))
        frames, _audio = _listen(_dsp(dcs=(25, False)), iq)
        self.assertEqual(frames[-1].state, "tone")


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheControls(_SecondCase):
    def test_the_choices_are_the_main_windows(self):
        from sdr_module.dsp.dcs import DCS_CODES
        from sdr_module.dsp.filters import CTCSS_FREQUENCIES

        rx = self.rx
        combo = rx._tone_combo
        values = [combo.itemData(i) for i in range(combo.count())]
        values = [v for v in values if v]  # (not the separators)
        self.assertEqual(values[0], "off")
        self.assertEqual(len(values), 1 + len(CTCSS_FREQUENCIES) + 2 * len(DCS_CODES))
        self.assertEqual(combo.itemText(values.index("100.0")), "100.0 Hz (1Z)")
        self.assertIn("D023N", values)
        self.assertIn("D023I", values)
        self.assertTrue(combo.accessibleName())
        index = combo.findData("D023I")
        self.assertIn("inverted", combo.itemData(index, 3))

    def test_narrow_fm_only(self):
        rx = self.rx
        form = rx._audio_form
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(rx._tone_combo.isEnabled())
        self.assertIn("CTCSS", rx._tone_combo.toolTip())
        self.assertTrue(form.isRowVisible(rx._tone_heard))
        rx.set_mode("AM")
        self.assertFalse(rx._tone_combo.isEnabled())
        self.assertIn("FM", rx._tone_combo.toolTip())
        self.assertFalse(form.isRowVisible(rx._tone_heard))
        # At a broadcast deviation a tone chosen is said not to be used.
        rx.set_mode("FM", "75 kHz", "200 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        self.assertIn("isn't used at 75 kHz", rx.message())

    def test_it_sets_the_receivers_tone_squelch(self):
        rx = self.rx
        rx.set_mode("FM", "5 kHz", "25 kHz")
        combo = rx._tone_combo
        combo.setCurrentIndex(combo.findData("100.0"))
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None))
        self.assertTrue(rx._dsp.options.tone_squelch)
        combo.setCurrentIndex(combo.findData("D754I"))
        self.assertEqual(rx._dsp.tone_squelch, (None, (754, True)))
        combo.setCurrentIndex(combo.findData("off"))
        self.assertEqual(rx._dsp.tone_squelch, (None, None))
        self.assertFalse(rx._dsp.options.tone_squelch)

    def test_saved_restored_and_damaged_values_logged(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        rx = self.rx
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("D023N"))
        values = self.store.values
        self.assertEqual(values["device/second_receiver/dcs_squelch"], "D023N")
        self.assertEqual(values["device/second_receiver/tone_squelch"], 0.0)
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._tone_combo.currentData(), "D023N")
        self.assertEqual(again._dsp.tone_squelch, (None, (23, False)))
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("123.0"))
        self.assertEqual(values["device/second_receiver/tone_squelch"], 123.0)
        self.assertEqual(values["device/second_receiver/dcs_squelch"], "")
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._dsp.tone_squelch, (123.0, None))
        values["device/second_receiver/tone_squelch"] = 12.3
        values["device/second_receiver/dcs_squelch"] = "D999Q"
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            damaged = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(damaged._tone_combo.currentData(), "off")
        self.assertEqual(damaged._dsp.tone_squelch, (None, None))
        text = "\n".join(logs.output)
        self.assertIn("device/second_receiver/tone_squelch", text)
        self.assertIn("device/second_receiver/dcs_squelch", text)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestInTheWindow(_SecondCase):
    def setUp(self):
        super().setUp()
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_listen("second")
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._sync_audio_output()

    def feed_iq(self, iq):
        rx = self.rx
        block = int(rx._rate() / 30)
        for start in range(0, len(iq), block):
            rx._process(iq[start : start + block])

    def test_tone_heard_and_the_tone_lamp(self):
        rx = self.rx
        self.feed_iq(_repeater(1.2, rate=rx._rate()))
        self.assertEqual(rx._tone_heard.text(), "100.0 Hz (1Z)")
        self.assertEqual(rx._tone_heard.property("tone"), "success")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("123.0"))
        self.speaker.blocks.clear()
        self.feed_iq(_repeater(1.2, rate=rx._rate()))
        self.assertEqual(rx._level_meter.state(), "tone")
        # (What played fades out in 5 ms; then silence.)
        self.assertLessEqual(len(self.speaker.samples()), 0.01 * AUDIO)
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        self.speaker.blocks.clear()
        self.feed_iq(_repeater(1.2, rate=rx._rate()))
        self.assertEqual(rx._level_meter.state(), "open")
        self.assertGreater(len(self.speaker.samples()), 0.5 * AUDIO)

    def test_a_dcs_code_heard(self):
        rx = self.rx
        self.feed_iq(_repeater(1.2, dcs=(23, False), rate=rx._rate()))
        self.assertEqual(rx._tone_heard.text(), "DCS 023 N")
        self.assertIn("DCS code 023", rx._tone_heard.toolTip())

    def test_stopped_nothing_is_heard(self):
        rx = self.rx
        self.feed_iq(_repeater(1.2, rate=rx._rate()))
        rx.stop()
        self.assertEqual(rx._tone_heard.text(), "None")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarks(_SecondCase):
    def setUp(self):
        super().setUp()
        panel = self.panel = self.win._bookmarks_panel
        tone = panel.add_bookmark("Repeater", 145.29e6, "FM", "5 kHz", "")
        panel.set_tone(tone, 100.0)
        code = panel.add_bookmark("Code rpt", 442.1e6, "FM", "5 kHz", "")
        panel.set_dcs(code, 23, False)
        panel.add_bookmark("Simplex", 146.52e6, "FM", "5 kHz", "")

    def bookmark(self, label):
        return next(b for b in self.panel.bookmarks() if b["label"] == label)

    def test_tuning_a_bookmark_applies_its_tone_or_code(self):
        rx = self.rx
        self.assertTrue(rx.tune_bookmark(self.bookmark("Repeater")))
        self.assertEqual(rx._tone_combo.currentData(), "100.0")
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None))
        self.assertIn("tone squelch 100.0 Hz", rx.message())
        self.assertTrue(rx.tune_bookmark(self.bookmark("Code rpt")))
        self.assertEqual(rx._tone_combo.currentData(), "D023N")
        self.assertIn("DCS 023 N", rx.message())
        # A channel without one: the tone squelch goes off.
        self.assertTrue(rx.tune_bookmark(self.bookmark("Simplex")))
        self.assertEqual(rx._tone_combo.currentData(), "off")
        self.assertEqual(rx._dsp.tone_squelch, (None, None))

    def test_bookmark_this_frequency_keeps_the_tone(self):
        rx = self.rx
        rx.tune(147.0e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("D754I"))
        rx.bookmark_frequency(rx.tuned_frequency())
        added = next(
            b for b in self.panel.bookmarks() if abs(b["freq_hz"] - 147.0e6) < 1
        )
        self.assertEqual(added.get("tone_mode"), "DTCS")
        self.assertEqual(added.get("dtcs"), 754)
        self.assertIn("DCS 754 I", rx.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTxMonitor(_SecondCase):
    """Check: Monitor (and Follow transmissions) hears the transmission.
    The main window's transmissions send no CTCSS tone or DCS code, so a
    tone squelch left on (a repeater listened to before) held the monitor
    silent, reading TONE."""

    def setUp(self):
        super().setUp()
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        station = self.win._tx_station
        station.device_problem = lambda: None
        self.spec = ("mcw", "FM", 16e3, "FM tone, 20 WPM")
        station.id_spec = lambda freq: self.spec
        self.win.set_frequency(146.52e6)
        rx = self.rx
        rx.set_listen("second")
        rx.tune(145.29e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")

    def test_monitoring_turns_the_tone_squelch_off(self):
        rx = self.rx
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        self.assertTrue(rx.monitor_tx())
        rx._frame_timer.stop()
        self.assertEqual((rx.tuned_frequency(), rx.mode()), (146.52e6, "FM"))
        self.assertEqual(rx._tone_combo.currentData(), "off")
        self.assertEqual(rx._dsp.tone_squelch, (None, None))
        self.assertIn("tone squelch off", rx.message())
        # What is sent (no tone) is heard, not held back as TONE.
        rx._sync_audio_output()
        self.speaker.blocks.clear()
        iq = _repeater(1.2, ctcss_hz=None, rate=rx._rate())
        block = int(rx._rate() / 30)
        for start in range(0, len(iq), block):
            rx._process(iq[start : start + block])
        self.assertEqual(rx._level_meter.state(), "open")
        self.assertGreater(len(self.speaker.samples()), 0.5 * AUDIO)

    def test_a_dcs_code_too_and_nothing_said_without_one(self):
        rx = self.rx
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("D023N"))
        self.assertTrue(rx.monitor_tx())
        rx._frame_timer.stop()
        self.assertEqual(rx._dsp.tone_squelch, (None, None))
        self.assertIn("tone squelch off", rx.message())
        self.assertTrue(rx.monitor_tx())  # (none set: nothing to say)
        self.assertNotIn("tone squelch", rx.message())

    def test_in_cw_the_tone_squelch_stays(self):
        # (It acts on narrow FM only: monitoring a CW ID leaves it be.)
        rx = self.rx
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        self.spec = ("cw", "CW", 100.0, "CW, 20 WPM")
        self.assertTrue(rx.monitor_tx())
        rx._frame_timer.stop()
        self.assertEqual(rx.mode(), "CW")
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None))
        self.assertNotIn("tone squelch", rx.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestToneHeardWhileStopped(_SecondCase):
    """Check: stopped, Tone heard said "None" (no tone on the channel) at
    a broadcast deviation too, where tones aren't looked for."""

    def test_it_says_whether_tones_are_looked_for(self):
        rx = self.rx
        rx.set_mode("FM", "75 kHz", "200 kHz")
        self.assertEqual(rx._tone_heard.text(), "Not checked")
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.assertEqual(rx._tone_heard.text(), "None")
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_mode("FM", "75 kHz", "200 kHz")
        rx._process(rx.device.read_samples(int(rx._rate() / 30)))
        rx.stop()
        self.assertEqual(rx._tone_heard.text(), "Not checked")
        self.assertIn("broadcast", rx._tone_heard.toolTip())


if __name__ == "__main__":
    unittest.main()
