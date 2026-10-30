#!/usr/bin/env python3
"""
Code review of the main window's state and tuning (area "mainstate"):
regression tests for the confirmed findings.

* MS-1: a scan stopped by a new sample rate, or by opening a recording,
  goes back to where it started (frequency, mode, filter and tone squelch),
  unless it stopped on a channel being listened to.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_review_mainstate.py``
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import _WindowTestCase  # noqa: E402

FRAME_S = 0.034


def _save_sigmf(path, rate, center, seconds=0.5, offset_hz=20e3):
    """A SigMF recording of one carrier ``offset_hz`` from ``center``."""
    from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

    t = np.arange(int(rate * seconds)) / rate
    samples = (0.5 * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)
    save_iq_file(path, samples, rate, center, SampleFormat.FLOAT32, FileFormat.SIGMF)
    return path


class _MainStateCase(_WindowTestCase):
    """The demo device; the display loop is run by hand (its timer is
    stopped); every status-bar message is kept in ``messages``."""

    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-mainstate-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = self.folder
        self.messages = []
        show = self.win._show_status_message

        def spy(text, tone=None, *args, **kwargs):
            self.messages.append((text, tone))
            return show(text, tone, *args, **kwargs)

        self.win._show_status_message = spy

    def frames(self, seconds, until=None):
        """Run the display loop for ``seconds`` of samples."""
        win = self.win
        for _ in range(int(seconds / FRAME_S)):
            win._pacer.reset()
            win._process_block(
                win._device.read_samples(int(win._device_sample_rate() * FRAME_S))
            )
            if until is not None and until():
                return True
        return False

    def recording(self, center, rate, name="take.sigmf-data", **kwargs):
        return _save_sigmf(os.path.join(self.folder, name), rate, center, **kwargs)

    def listening(self):
        """``(tuned frequency, mode, FM deviation, filter, tone squelch)``."""
        panel = self.win._control_panel
        return (
            self.win._current_frequency(),
            panel.get_demod_mode(),
            panel._fm_dev_combo.currentText(),
            panel._bw_combo.currentText(),
            float(panel.tone_squelch_hz()),
        )

    def status(self):
        label = self.win._message_label
        return label.text(), label.property("tone")


# ---------------------------------------------------------------------------
# MS-1: where a scan stopped by the window ends
# ---------------------------------------------------------------------------


class TestScanStoppedByTheWindow(_MainStateCase):
    """Broadcast FM on 100.1 MHz, then a bookmark scan reaches a quiet
    repeater (FM 5 kHz, 12.5 kHz filter, 123 Hz tone squelch). A new
    sample rate, or a recording opened, used to stop the scan right there:
    the receiver stayed on the repeater, tone squelch on (muting ordinary
    stations), and that was saved for the next session; Close Recording
    went back to the repeater too."""

    BEFORE = (100.1e6, "FM", "75 kHz", "200 kHz", 0.0)
    REPEATER = (146.97e6, "FM", "5 kHz", "12.5 kHz", 123.0)

    def setUp(self):
        super().setUp()
        win = self.win
        win._tune_to(100.1e6, recenter=True)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        win._control_panel.set_tone_squelch_hz(0.0)
        self.assertEqual(self.listening(), self.BEFORE)
        self.scanner = win._scanner_panel

    def scan_bookmark(self, freq, mode, deviation, width, tone=0.0):
        bookmarks = self.win._bookmarks_panel
        index = bookmarks.add_bookmark("Channel", freq, mode, deviation, width)
        if tone:
            bookmarks.set_tone(index, tone)
        self.scanner.set_scan_mode("bookmarks")
        self.assertTrue(self.scanner.start_scan())
        self.assertTrue(
            self.frames(1.0, lambda: abs(self.win._current_frequency() - freq) < 1)
        )

    def scan_to_the_repeater(self):
        self.scan_bookmark(146.97e6, "FM", "5 kHz", "12.5 kHz", 123.0)
        self.assertEqual(self.listening(), self.REPEATER)
        self.assertNotEqual(self.scanner.engine.state, "listen")  # nothing there

    def test_a_new_sample_rate_goes_back_to_where_the_scan_started(self):
        self.scan_to_the_repeater()
        self.assertTrue(self.win.set_sample_rate(2.048e6))
        self.frames(0.1)
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(
            self.scanner._status.text(),
            "Scan stopped: the sample rate changed. Start it again.",
        )
        self.assertEqual(self.listening(), self.BEFORE)
        self.assertAlmostEqual(self.win._lo_frequency(), 100.1e6)
        # ... and that is what the next session starts with.
        self.win.close()
        saved = self.store.values
        self.assertEqual(saved["frequency_hz"], 100.1e6)
        self.assertEqual(saved["fm_deviation"], "75 kHz")
        self.assertEqual(saved["bandwidth"], "200 kHz")
        self.assertEqual(saved["audio/tone_squelch"], 0.0)

    def test_a_new_sample_rate_keeps_a_channel_being_listened_to(self):
        """A scan stopped on a signal you are listening to stays there."""
        self.scan_bookmark(125.0e6, "AM", "", "10 kHz")  # the demo's tower
        self.assertTrue(self.frames(2.0, lambda: self.scanner.engine.state == "listen"))
        self.assertTrue(self.win.set_sample_rate(2.048e6))
        self.frames(0.1)
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(self.listening()[:2], (125.0e6, "AM"))

    def test_opening_a_recording_ends_the_scan_first(self):
        """Close Recording then goes back to where the scan started: the
        recording took its 'before' while the scan was still on."""
        self.scan_to_the_repeater()
        path = self.recording(145.0e6, 2.4e6)
        self.assertTrue(self.win._open_recording(path))
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(
            self.scanner._status.text(), "Scan stopped: a recording is playing."
        )
        self.win._playback.close(restore=True)
        self.assertEqual(self.listening(), self.BEFORE)
        self.assertAlmostEqual(self.win._lo_frequency(), 100.1e6)

    def test_a_recording_at_another_rate_says_a_recording_is_playing(self):
        """It said 'the sample rate changed' (that check came first)."""
        self.scan_to_the_repeater()
        self.assertTrue(self.win._open_recording(self.recording(145.0e6, 1.0e6)))
        self.frames(0.1)
        self.assertEqual(
            self.scanner._status.text(), "Scan stopped: a recording is playing."
        )
        self.win._playback.close(restore=True)
        self.assertEqual(self.listening(), self.BEFORE)

    def test_the_window_checks_the_recording_before_the_rate(self):
        """(A scan running when the LO is locked, e.g. by a path other than
        Open Recording, is stopped for the recording, not the rate.)"""
        self.scan_to_the_repeater()
        win = self.win
        win._lo_lock = (win._lo_frequency(), 1.0e6, "the recording")
        win._device.set_sample_rate(1.0e6)
        try:
            self.frames(0.1)
        finally:
            win._lo_lock = None
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(
            self.scanner._status.text(), "Scan stopped: a recording is playing."
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
