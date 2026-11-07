#!/usr/bin/env python3
"""
The squelch lane's check (M-Q1, M-Q2, M-Q3): what the checker found.

* The reverse tone squelch mutes its station from the first moment of each
  transmission also while Record Audio runs. Record Audio keeps the receiver
  demodulating while the squelch is closed, so the tone squelch heard the
  noise between transmissions as "no tone" for seconds: the station it mutes
  then played for its first half second, until its tone was found. Each
  signal the squelch opens on is now judged afresh, as it is without Record
  Audio.
* A scan begun with Reverse ticked puts it back ticked.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import pretend_audio_output  # noqa: E402
from tests.test_gui_tone_reverse import _demo_signal  # noqa: E402


def _transmission_start(kind, after=3.0):
    """A stream time (s) at which one of a demo repeater's transmissions
    starts, at least ``after`` seconds into the stream."""
    seed, (period, _on), _freq = _demo_signal(kind)
    start = -((seed * 1.7) % period)
    while start < after:
        start += period
    return start


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReverseWhileRecordingAudio(_WindowTestCase):
    """The demo's 2 m repeater (145.290 MHz) sends 100.0 Hz, its 70 cm one
    (442.100 MHz) DCS 023 N. Reverse on that tone or code, with Record Audio
    running from before the transmission starts."""

    demo = True

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.played = []
        self.win._audio.write = lambda audio: self.played.append(
            np.asarray(audio, float).copy()
        )
        self.folder = tempfile.mkdtemp(prefix="reverse-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.store.values["recording/folder"] = self.folder

    def tearDown(self):
        if HAS_PYQT6:
            self.win._stop_audio_recording(quiet=True)
        super().tearDown()

    def listen(self, kind, record=True, before=2.0, seconds=3.5):
        """Peak played per frame, with its stream time from the start of a
        transmission (the 2 s before it: the repeater off, noise)."""
        from sdr_module.gui import device_dialog

        win = self.win
        win._stop_acquisition()
        win._device.start_rx()
        rate = win._device_sample_rate()
        start = _transmission_start(kind)
        win._device._stream_pos = int((start - before) * rate)
        if record:
            self.assertTrue(win._start_audio_recording())
        frames = []
        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            for _ in range(int(seconds * 30)):
                win._pacer.reset()
                self.played.clear()
                block = win._device.read_samples(int(rate / 30))
                at = win._device._stream_pos / rate - start
                win._process_block(block)
                peak = max(
                    (float(np.max(np.abs(a))) for a in self.played if a.size),
                    default=0.0,
                )
                frames.append((at, peak, bool(win._squelch.is_open)))
        # (The squelch was closed before the transmission and opened on it.)
        self.assertFalse(any(open_ for at, _p, open_ in frames if at < -0.5))
        self.assertTrue(any(open_ for at, _p, open_ in frames if at > 0.3))
        return frames

    def tune_rpt(self, tone_hz):
        win = self.win
        win.set_frequency(145.29e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_tone_squelch_hz(tone_hz)
        win._control_panel.set_tone_reverse(True)

    def test_the_station_it_mutes_isnt_heard_as_it_starts(self):
        """(Before: its first 0.53 s played, at full level.)"""
        self.tune_rpt(100.0)
        frames = self.listen("rpt")
        self.assertLess(max(p for _at, p, _o in frames), 1e-3)
        self.assertEqual(self.win._squelch_state(True), "reverse")

    def test_a_dcs_station_it_mutes_isnt_heard_as_it_starts(self):
        win = self.win
        win.set_frequency(442.1e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_dcs_squelch(23, False)
        win._control_panel.set_tone_reverse(True)
        frames = self.listen("dcs")
        self.assertLess(max(p for _at, p, _o in frames), 1e-3)

    def test_every_other_station_still_plays_after_half_a_second(self):
        self.tune_rpt(123.0)
        frames = self.listen("rpt")
        first = next(at for at, p, _o in frames if p > 0.05)
        self.assertGreaterEqual(first, 0.4)
        self.assertLessEqual(first, 0.8)
        self.assertGreater(min(p for at, p, _o in frames if at > 1.0), 0.05)

    def test_without_record_audio_as_before(self):
        self.tune_rpt(100.0)
        frames = self.listen("rpt", record=False)
        self.assertLess(max(p for _at, p, _o in frames), 1e-3)

    def test_the_tone_squelch_still_opens_on_its_tone_while_recording(self):
        """(Not reversed: the tone is found again on each transmission,
        as without Record Audio.)"""
        self.tune_rpt(100.0)
        self.win._control_panel.set_tone_reverse(False)
        frames = self.listen("rpt")
        first = next(at for at, p, _o in frames if p > 0.05)
        self.assertLessEqual(first, 0.8)
        self.assertGreater(min(p for at, p, _o in frames if at > 1.0), 0.05)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheScanPutsBackAReversedSquelch(_WindowTestCase):
    """A bookmark scan begun with Reverse ticked ends with it ticked (the
    lane's own test began with it clear, which the scan's bookmarks clear
    anyway)."""

    demo = True

    def test_reverse_is_put_back(self):
        from sdr_module.gui import device_dialog
        from sdr_module.gui.scan_engine import LISTEN

        win, panel = self.win, self.win._control_panel
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_tone_squelch_hz(151.4)
        panel.set_tone_reverse(True)
        bookmarks = win._bookmarks_panel
        for index in reversed(range(len(bookmarks.bookmarks()))):
            bookmarks._bookmarks.pop(index)
        # (A plain one: the scan sets Reverse off for it.)
        index = bookmarks.add_bookmark("Plain", 145.29e6, "FM", "5 kHz")
        bookmarks.set_tone(index, 123.0)
        scanner = win._scanner_panel
        scanner.set_scan_mode("bookmarks")
        self.assertTrue(scanner.start_scan())
        seen_clear = False
        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            for _ in range(30):
                win._pacer.reset()
                win._process_block(
                    win._device.read_samples(int(win._device_sample_rate() / 30))
                )
                seen_clear = seen_clear or not panel.tone_reverse()
                if scanner.engine.state == LISTEN:
                    break
        self.assertTrue(seen_clear)  # (the bookmark's own squelch was set)
        scanner.stop_scan()
        self.assertEqual(panel.tone_squelch_hz(), 151.4)
        self.assertTrue(panel.tone_reverse())
        self.assertTrue(win._tone_squelch.reverse)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReversedAndPlainAreTwoResults(_WindowTestCase):
    """Two bookmarks alike but for Reverse (one muted while its tone is
    heard, the other only then) are two results of a bookmark scan."""

    def test_two_results(self):
        from sdr_module.gui.scan_engine import Found, ScanEngine

        scanner = self.win._scanner_panel

        def found(reverse):
            return Found(
                145.29e6,
                -50.0,
                -90.0,
                12.5e3,
                ("FM", "5 kHz", "12.5 kHz"),
                label="Rpt",
                tone_hz=100.0,
                reverse=reverse,
            )

        with mock.patch.object(
            ScanEngine, "mode", new_callable=mock.PropertyMock, return_value="bookmarks"
        ):
            scanner._on_found(found(False))
            scanner._on_found(found(True))
            scanner._on_found(found(True))
        hits = scanner.hits()
        self.assertEqual(len(hits), 2)
        self.assertEqual(
            sorted((hit.reverse, hit.count) for hit in hits), [(False, 1), (True, 2)]
        )


if __name__ == "__main__":
    unittest.main()
