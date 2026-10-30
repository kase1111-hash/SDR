#!/usr/bin/env python3
"""
Code review of the main window's state and tuning (area "mainstate"):
regression tests for the confirmed findings.

* MS-1: a scan stopped by a new sample rate, or by opening a recording,
  goes back to where it started (frequency, mode, filter and tone squelch),
  unless it stopped on a channel being listened to.
* MS-2: Apply Preset with a frequency that is refused changes nothing
  else, and the refusal is not replaced by an "applied" confirmation.
* MS-4: a recording made above 6 GHz plays at its own frequency and is
  tuned inside its span.

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


# ---------------------------------------------------------------------------
# MS-2: the control panel's Apply Preset, refused
# ---------------------------------------------------------------------------


class TestApplyPresetRefused(_MainStateCase):
    """Presets > Aviation > ACARS (131.550 MHz, AM) while a 145 MHz
    recording plays: the frequency is refused (outside the recording). The
    panel still switched to AM, 15 kHz, and the window then replaced the
    refusal with "ACARS: 145.000 MHz, AM, 15 kHz bandwidth"."""

    def choose(self, category, name):
        panel = self.win._control_panel
        panel._category_combo.setCurrentText(category)
        panel._preset_combo.setCurrentText(name)
        self.assertEqual(panel._preset_combo.currentText(), name)
        return panel

    def test_a_refused_preset_changes_nothing_and_says_why(self):
        win = self.win
        self.assertTrue(win._open_recording(self.recording(145.0e6, 2.4e6)))
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        before = self.listening()
        panel = self.choose("Aviation", "ACARS")
        self.messages.clear()
        panel._apply_preset_btn.click()
        self.assertEqual(self.listening(), before)
        refusal = "131.550 MHz is outside the recording, which covers 143.8–146.2 MHz."
        self.assertEqual(self.messages, [(refusal, "danger")])
        self.assertEqual(self.status(), (refusal, "danger"))

    def test_a_preset_tuned_still_sets_its_mode_and_says_so(self):
        panel = self.choose("Aviation", "ACARS")
        panel._apply_preset_btn.click()
        self.assertEqual(self.win._current_frequency(), 131.55e6)
        self.assertEqual(panel.get_demod_mode(), "AM")
        text, tone = self.status()
        self.assertIn("ACARS", text)
        self.assertNotEqual(tone, "danger")


# ---------------------------------------------------------------------------
# MS-4: a recording above the receiver's 6 GHz
# ---------------------------------------------------------------------------


class TestRecordingAbove6GHz(_MainStateCase):
    """A SigMF capture of the QO-100 downlink (center 10489.75 MHz) played
    with the tuned frequency clamped to 6000 MHz: FREQ read 6000.000 MHz,
    the receiver mixed 4.5 GHz off, and every frequency in the file was
    refused as out of range."""

    CENTER = 10489.75e6

    def setUp(self):
        super().setUp()
        path = self.recording(self.CENTER, 2.4e6, seconds=1.0, offset_hz=50e3)
        self.assertTrue(self.win._open_recording(path))

    def covered(self):
        from sdr_module.gui.tuning import format_mhz_range

        return format_mhz_range(self.CENTER - 1.2e6, self.CENTER + 1.2e6)

    def test_it_plays_at_its_own_frequency(self):
        win = self.win
        self.assertEqual(win._lo_frequency(), self.CENTER)
        self.assertEqual(win._current_frequency(), self.CENTER)
        self.assertEqual(win._vfo_offset_hz(), 0.0)
        self.assertEqual(win._freq_label.text(), "10489.750 MHz")
        self.assertIn(f"Playing take.sigmf-data ({self.covered()})", self.status()[0])

    def test_a_signal_in_it_is_tuned_and_heard(self):
        win = self.win
        win._set_demod_mode("AM")
        self.assertTrue(win.set_frequency(self.CENTER + 50e3))
        self.assertEqual(win._current_frequency(), self.CENTER + 50e3)
        self.assertEqual(win._lo_frequency(), self.CENTER)
        self.assertEqual(win._vfo_offset_hz(), 50e3)
        self.assertEqual(win._receiver_chain().offset_hz, 50e3)
        self.assertEqual(win._freq_label.text(), "10489.800 MHz")
        for _ in range(6):
            win._process_block(win._device.read_samples(80000))
        self.assertGreater(win._last_peak_db, -20.0)  # the carrier, in the channel

    def test_frequencies_outside_it_are_refused_as_outside_the_recording(self):
        win = self.win
        self.assertFalse(win.set_frequency(10495e6))
        self.assertEqual(win._current_frequency(), self.CENTER)
        self.assertEqual(
            self.status(),
            (
                "10495.000 MHz is outside the recording, which covers "
                f"{self.covered()}.",
                "danger",
            ),
        )

    def test_typing_a_frequency_in_it(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication

        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "10490.5")
        self.assertFalse(field.property("tone"))  # (no red frame: it can be tuned)
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertEqual(self.win._current_frequency(), 10490.5e6)

    def test_closing_it_goes_back_to_the_receiver_and_its_range(self):
        win = self.win
        win._playback.close(restore=True)
        self.assertEqual(win._control_panel._freq_input.tuning_range(), (1e3, 6e9))
        self.assertEqual(win._current_frequency(), 100.1e6)
        self.assertEqual(win._lo_frequency(), 100.1e6)
        self.assertFalse(win.set_frequency(self.CENTER))  # the demo device: no
        self.assertIn("is out of range", self.status()[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
