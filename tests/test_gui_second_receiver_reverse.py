"""Reverse tone squelch (TSQL-R, DTCS-R) in the Second Receiver window
(post-merge, rx2 with the squelch lane's M-Q3).

The main window can reverse its tone squelch (Radio > Audio > Reverse Tone
Squelch, the Audio group's "Reverse: mute this tone"): the stations sending
the tone or code are muted, every other one plays. The Second Receiver
couldn't, and a TSQL-R or DTCS-R bookmark tuned there played with no tone
squelch at all (round 1: 145.290 MHz reverse 100 Hz gave tone None,
reverse False). Now its tone squelch has the same Reverse switch, saved
(``device/second_receiver/tone_reverse``; a damaged value is logged and
read as off), its Tone heard says when the tone heard is one it mutes, and
bookmarks apply and keep it as the main window's do.

A signal its squelch has just opened on is judged afresh, as in the main
window (``_route_audio``'s ``_tone_gate_open``), whatever the receiver
demodulated before it: the block that fades out as the squelch closes is
demodulated, and a station keying up right after it (one block later)
kept the last station's verdict, so a reverse tone squelch played the
station it mutes for its first 0.6 s. Its decoders, its I/Q recording and
Listen don't demodulate through its squelch (checked here too).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402
from tests.test_gui_second_receiver_tones import (  # noqa: E402
    BLOCK,
    RATE,
    _DspCase,
    _listen,
    _repeater,
)

KEY = "device/second_receiver/"


def _dsp(choice):
    from sdr_module.gui.second_receiver import ReceiverDsp

    dsp = ReceiverDsp()
    dsp.rate, dsp.mode, dsp.bandwidth_hz, dsp.deviation_hz = RATE, "FM", 25e3, 5e3
    dsp.tone_squelch = choice
    dsp.options.tone_squelch = True
    return dsp


def _heard(frames):
    """Frames whose audio is more than a whisper (a fade-out's tail aside)."""
    return [
        i
        for i, f in enumerate(frames)
        if f.audio is not None and len(f.audio) and np.max(np.abs(f.audio)) > 1e-3
    ]


def _blocks(iq):
    return iq[: len(iq) // BLOCK * BLOCK]


# ---------------------------------------------------------------------------
# The signal path
# ---------------------------------------------------------------------------


class TestReversed(_DspCase):
    def test_the_tone_is_muted_every_other_station_plays(self):
        frames, audio = _listen(_dsp((100.0, None, True)), _repeater(1.5))
        self.assertEqual(_heard(frames), [])
        self.assertEqual(frames[-1].state, "reverse")
        self.assertEqual(frames[-1].tone_heard[0], 100.0)
        for other in (None, 123.0):
            frames, audio = _listen(
                _dsp((100.0, None, True)), _repeater(1.5, ctcss_hz=other)
            )
            heard = _heard(frames)
            # Held while it is looked for (REVERSE_DECIDE_S), then heard.
            self.assertTrue(heard, other)
            self.assertGreaterEqual(heard[0] * BLOCK / RATE, 0.5, other)
            self.assertEqual(frames[-1].state, "open", other)

    def test_a_dcs_code_reversed(self):
        iq = _repeater(1.5, dcs=(23, False))
        frames, _audio = _listen(_dsp((None, (23, False), True)), iq)
        self.assertEqual(_heard(frames), [])
        self.assertEqual(frames[-1].state, "reverse")
        frames, _audio = _listen(_dsp((None, (25, False), True)), iq)
        self.assertTrue(_heard(frames))

    def test_two_values_still_mean_not_reversed(self):
        # (Library users and the lane's own tests set (tone, code).)
        dsp = _dsp((100.0, None))
        frames, _audio = _listen(dsp, _repeater(1.2))
        self.assertTrue(_heard(frames))
        self.assertFalse(dsp.tones.reverse)


class TestJudgedAfreshAsTheSquelchOpens(_DspCase):
    """A station without the tone (it plays), its carrier gone (the
    squelch closes after its hang), then the station the reverse tone
    squelch mutes keys up: never heard, however soon after the squelch
    closed it comes."""

    def test_the_muted_station_never_plays(self):
        rng = np.random.default_rng(9)
        for gap in range(9, 15):  # (blocks of noise: it closes after ~10)
            first = _blocks(_repeater(1.5, ctcss_hz=None))
            noise = 1e-3 * (
                rng.standard_normal(gap * BLOCK) + 1j * rng.standard_normal(gap * BLOCK)
            )
            muted = _blocks(_repeater(1.2, ctcss_hz=100.0))
            iq = np.concatenate((first, noise.astype(np.complex64), muted))
            dsp = _dsp((100.0, None, True))
            dsp.squelch.auto = False
            dsp.squelch.manual_db = -60.0
            frames = [
                dsp.process(iq[s : s + BLOCK], True) for s in range(0, len(iq), BLOCK)
            ]
            start = (len(first) + gap * BLOCK) // BLOCK
            states = [f.state for f in frames[len(first) // BLOCK : start]]
            if "closed" not in states:
                continue  # (it keyed up within the hang: one transmission)
            self.assertTrue(_heard(frames[: len(first) // BLOCK]), gap)
            self.assertEqual(_heard(frames[start:]), [], gap)
            self.assertEqual(frames[-1].state, "reverse", gap)


class TestTheLampSaysReverse(_DspCase):
    """(thr-2) While a reverse tone squelch holds a station silent, the
    lamp reads "reverse", as the main window's (level_meter.STATES: "the
    station sends the CTCSS tone or DCS code it mutes, or hasn't been heard
    long enough to tell"), not "tone" ("not the chosen CTCSS tone or DCS
    code": the opposite reason). Not reversed, "tone" as before."""

    def test_the_station_it_mutes(self):
        frames, _audio = _listen(_dsp((100.0, None, True)), _repeater(1.5))
        states = [f.state for f in frames]
        self.assertEqual(states[-1], "reverse")
        self.assertNotIn("tone", states)
        iq = _repeater(1.5, dcs=(23, False))
        frames, _audio = _listen(_dsp((None, (23, False), True)), iq)
        states = [f.state for f in frames]
        self.assertEqual(states[-1], "reverse")
        self.assertNotIn("tone", states)

    def test_another_station_while_it_decides(self):
        """Held for about half a second (it can't tell yet), then heard."""
        for other in (None, 123.0):
            iq = _repeater(1.5, ctcss_hz=other)
            frames, _audio = _listen(_dsp((100.0, None, True)), iq)
            held = [f.state for f in frames if f.state not in ("open", "closed")]
            self.assertTrue(held, other)
            self.assertEqual(set(held), {"reverse"}, other)
            self.assertEqual(frames[-1].state, "open", other)

    def test_not_reversed_reads_tone_as_before(self):
        frames, _audio = _listen(_dsp((123.0, None)), _repeater(1.5))
        self.assertEqual(frames[-1].state, "tone")
        self.assertNotIn("reverse", [f.state for f in frames])


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheReverseSwitch(_SecondCase):
    def choose(self, value):
        combo = self.rx._tone_combo
        combo.setCurrentIndex(combo.findData(value))

    def test_it_sits_under_the_tone_squelch_as_in_the_main_window(self):
        from sdr_module.gui.audio_chain import REVERSE_TONE_TIP

        rx = self.rx
        check = rx._tone_reverse_check
        rx.set_mode("FM", "5 kHz", "25 kHz")
        row = rx._audio_form.getWidgetPosition(check)[0]
        self.assertEqual(row, rx._audio_form.getWidgetPosition(rx._tone_combo)[0] + 1)
        self.assertTrue(check.accessibleName())
        # Nothing chosen: nothing to reverse, and it says what first.
        self.assertFalse(check.isEnabled())
        self.assertIn("Choose a CTCSS tone or DCS code", check.toolTip())
        self.choose("100.0")
        self.assertTrue(check.isEnabled())
        self.assertEqual(check.toolTip(), REVERSE_TONE_TIP)
        self.assertEqual(check.text(), "Reverse: mute this tone")
        self.choose("D023N")
        self.assertEqual(check.text(), "Reverse: mute this code")
        # FM only, as the tone squelch.
        rx.set_mode("AM")
        self.assertFalse(rx._audio_form.isRowVisible(check))
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(rx._audio_form.isRowVisible(check))

    def test_it_reverses_the_receivers_tone_squelch(self):
        rx = self.rx
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.choose("100.0")
        rx._tone_reverse_check.setChecked(True)
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None, True))
        self.assertIn("are muted", rx.message())
        self.choose("D754I")
        self.assertEqual(rx._dsp.tone_squelch, (None, (754, True), True))
        rx._tone_reverse_check.setChecked(False)
        self.assertEqual(rx._dsp.tone_squelch, (None, (754, True)))
        # Ticked with no tone chosen: nothing is reversed.
        self.choose("off")
        rx._tone_reverse_check.setChecked(True)
        self.assertEqual(rx._dsp.tone_squelch, (None, None))

    def test_saved_and_restored(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        rx = self.rx
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.choose("123.0")
        rx._tone_reverse_check.setChecked(True)
        self.assertIs(self.store.values[KEY + "tone_reverse"], True)
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertTrue(again._tone_reverse_check.isChecked())
        self.assertEqual(again._dsp.tone_squelch, (123.0, None, True))
        rx._tone_reverse_check.setChecked(False)
        self.assertIs(self.store.values[KEY + "tone_reverse"], False)

    def test_a_damaged_value_is_logged_and_off(self):
        from PyQt6.QtCore import QSettings

        from sdr_module.gui import settings_store
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        folder = tempfile.mkdtemp(prefix="rx2-reverse-")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "settings.ini")
        raw = QSettings(path, QSettings.Format.IniFormat)
        raw.setValue(KEY + "tone_squelch", 100.0)
        raw.setValue(KEY + "tone_reverse", "sometimes")
        raw.sync()
        with mock.patch.object(
            settings_store,
            "QSettings",
            lambda *_a: QSettings(path, QSettings.Format.IniFormat),
        ):
            store = settings_store.GuiSettings()
        kept = self.win._settings
        self.win._settings = store
        self.addCleanup(setattr, self.win, "_settings", kept)
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            damaged = self.extra(SecondReceiverWindow(self.win))
        self.assertFalse(damaged._tone_reverse_check.isChecked())
        self.assertEqual(damaged._dsp.tone_squelch, (100.0, None))
        self.assertTrue(any(KEY + "tone_reverse" in m for m in logs.output))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarks(_SecondCase):
    def setUp(self):
        super().setUp()
        panel = self.panel = self.win._bookmarks_panel
        index = panel.add_bookmark("Rpt reversed", 145.29e6, "FM", "5 kHz", "")
        panel.set_tone(index, 100.0, reverse=True)
        index = panel.add_bookmark("Code reversed", 442.1e6, "FM", "5 kHz", "")
        panel.set_dcs(index, 23, False, reverse=True)
        index = panel.add_bookmark("Rpt", 145.31e6, "FM", "5 kHz", "")
        panel.set_tone(index, 100.0)

    def bookmark(self, label):
        return next(b for b in self.panel.bookmarks() if b["label"] == label)

    def test_tuning_one_applies_its_reverse_tone_squelch(self):
        rx = self.rx
        self.assertTrue(rx.tune_bookmark(self.bookmark("Rpt reversed")))
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None, True))
        self.assertTrue(rx._tone_reverse_check.isChecked())
        self.assertIn("reverse tone squelch 100.0 Hz", rx.message())
        self.assertTrue(rx.tune_bookmark(self.bookmark("Code reversed")))
        self.assertEqual(rx._dsp.tone_squelch, (None, (23, False), True))
        self.assertIn("reverse DCS 023 N", rx.message())
        # One that isn't reversed turns Reverse off.
        self.assertTrue(rx.tune_bookmark(self.bookmark("Rpt")))
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None))
        self.assertFalse(rx._tone_reverse_check.isChecked())
        self.assertNotIn("reverse", rx.message())

    def test_the_main_window_agrees(self):
        # (The same bookmark tuned in the main window: round 1 found the
        # main window reversed on 100 Hz, this one without a tone squelch.)
        bookmark = self.bookmark("Rpt reversed")
        win = self.win
        self.assertTrue(win.set_frequency(145.29e6))
        win._on_bookmark_tuned(bookmark)
        self.assertTrue(win._tone_squelch.reverse)
        self.assertEqual(win._tone_squelch.tone_hz, 100.0)
        self.assertTrue(self.rx.tune_bookmark(bookmark))
        self.assertEqual(self.rx._dsp.tone_squelch, (100.0, None, True))

    def test_bookmark_this_frequency_keeps_it_reversed(self):
        rx = self.rx
        rx.tune(147.0e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("D754I"))
        rx._tone_reverse_check.setChecked(True)
        rx.bookmark_frequency(rx.tuned_frequency())
        added = next(
            b for b in self.panel.bookmarks() if abs(b["freq_hz"] - 147.0e6) < 1
        )
        self.assertEqual(added.get("tone_mode"), "DTCS-R")
        self.assertEqual(added.get("dtcs"), 754)
        self.assertIn("reverse DCS 754 I", rx.message())
        rx.tune(147.5e6, recenter=True)
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        rx.bookmark_frequency(rx.tuned_frequency())
        added = next(
            b for b in self.panel.bookmarks() if abs(b["freq_hz"] - 147.5e6) < 1
        )
        self.assertEqual(added.get("tone_mode"), "TSQL-R")
        self.assertIn("reverse tone squelch 100.0 Hz", rx.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTxMonitor(_SecondCase):
    def test_a_reverse_tone_squelch_counts_as_set_and_goes_off(self):
        station = self.win._tx_station
        station.device_problem = lambda: None
        station.id_spec = lambda freq: ("mcw", "FM", 16e3, "FM tone, 20 WPM")
        self.win.set_frequency(146.52e6)
        rx = self.rx
        rx.tune(145.29e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        rx._tone_reverse_check.setChecked(True)
        self.assertTrue(rx.monitor_tx())
        rx._frame_timer.stop()
        self.assertEqual(rx._dsp.tone_squelch, (None, None))
        self.assertIn("tone squelch off", rx.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheDemoRepeater(_SecondCase):
    """The demo's 2 m repeater (145.290 MHz, 100.0 Hz, 8 s on in every 12)
    with the tone squelch reversed on its tone: never heard, through a
    key-up, while this receiver decodes and records its I/Q and Listen
    moves to it, and the lamp and Tone heard say why."""

    def setUp(self):
        super().setUp()
        folder = tempfile.mkdtemp(prefix="rx2-reverse-rec-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.tune(145.29e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        rx._tone_reverse_check.setChecked(True)
        rx._decoder_combo.setCurrentIndex(rx._decoder_combo.findData("AX.25/APRS"))
        rx._set_recording(True)  # (shutting the window down ends it)

    def test_never_heard(self):
        rx = self.rx
        rx.set_listen("main")  # (not heard here: the TONE lamp only)
        rx._sync_audio_output()
        states = []
        for _ in range(150):  # 5 s
            rx._process(rx.device.read_samples(BLOCK))
            states.append(rx._level_meter.state())
        rx.set_listen("second")
        rx._sync_audio_output()
        for _ in range(240):  # 8 s more: a key-up at least
            rx._process(rx.device.read_samples(BLOCK))
            states.append(rx._level_meter.state())
        self.assertIn("reverse", states)  # (thr-2: not "tone")
        self.assertNotIn("tone", states)
        self.assertIn("closed", states)  # (and the gap between two)
        heard = self.speaker.samples()
        peak = float(np.max(np.abs(heard))) if len(heard) else 0.0
        self.assertLess(peak, 1e-3)
        self.assertTrue(rx.is_recording())

    def test_tone_heard_says_it_is_muted(self):
        rx = self.rx
        rx.set_listen("second")
        rx._sync_audio_output()
        for _ in range(390):
            rx._process(rx.device.read_samples(BLOCK))
            if rx._tone_heard.text() == "100.0 Hz (1Z)":
                break
        label = rx._tone_heard
        self.assertEqual(label.text(), "100.0 Hz (1Z)")
        self.assertEqual(label.property("tone"), "warning")
        self.assertIn("The reverse tone squelch mutes it", label.toolTip())
        # Not reversed: as before.
        rx._tone_reverse_check.setChecked(False)
        rx._show_tone_heard(100.0, True, None)
        self.assertEqual(label.property("tone"), "success")
        self.assertNotIn("mutes it", label.toolTip())

    def test_the_level_meter_gives_the_same_reason(self):
        """(thr-2) The LEVEL meter's tooltip agrees with Tone heard, as the
        main window's does: muted by the reverse tone squelch, not "not the
        chosen CTCSS tone or DCS code"."""
        from sdr_module.gui.level_meter import STATES

        rx = self.rx
        meter = rx._level_meter
        rx.set_listen("second")
        rx._sync_audio_output()
        for _ in range(390):
            rx._process(rx.device.read_samples(BLOCK))
            if rx._tone_heard.text() == "100.0 Hz (1Z)" and meter.state() != "closed":
                break
        self.assertIn("The reverse tone squelch mutes it", rx._tone_heard.toolTip())
        self.assertEqual(meter.state(), "reverse")
        tip = meter.tooltip_text()
        self.assertIn(STATES["reverse"][1], tip)
        self.assertIn("muted by the reverse tone squelch", tip)
        self.assertNotIn(STATES["tone"][1], tip)


if __name__ == "__main__":
    unittest.main()
