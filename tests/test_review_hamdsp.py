"""Regression tests for the code review of the ham radio and DSP area.

One class per finding (HAMDSP-1 to HAMDSP-11): SSTV, the S-meter, QRP, the
AM/FM tuner, noise reduction and the CLI encoder.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_hamdsp.py``
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


class _HamWindowCase(_WindowTestCase):
    """The real window (demo device) with the ham radio panels."""

    demo = True

    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sstv = self.win._sstv_panel
        self.sstv.set_save_directory(self.tmp.name)
        self.win._audio.write = lambda audio: None

    def tearDown(self):
        if HAS_PYQT6 and getattr(self, "sstv", None) and self.sstv.is_receiving():
            self.sstv._start_btn.click()
        super().tearDown()


class TestTuneToIssRefused(_HamWindowCase):
    """HAMDSP-1: Tune to ISS ignored a refused tune. It said "Tuned to the
    ISS SSTV downlink", switched the station being heard to narrow FM and
    started the SSTV decoder while the receiver stayed where it was."""

    def assert_nothing_changed(self, freq_hz: float, mode: str, reason: str):
        win = self.win
        self.assertAlmostEqual(win._current_frequency(), freq_hz)
        self.assertEqual(win._control_panel.get_demod_mode(), mode)
        self.assertFalse(self.sstv.is_receiving(), "the decoder stays off")
        self.assertIn(reason, win._message_label.text())
        self.assertEqual(win._message_label.property("tone"), "danger")

    def test_refused_while_transmitting(self):
        win = self.win
        self.assertTrue(win.set_frequency(14.230e6))
        win._set_demod_mode("USB")
        win._tx_busy = lambda: True  # a transmission is on the air
        self.sstv._tune_iss_btn.click()
        settle()
        self.assert_nothing_changed(14.230e6, "USB", "Can't tune while transmitting")

    def test_refused_outside_the_recording_from_the_menu(self):
        win = self.win
        win._set_demod_mode("AM")
        win._lock_lo(100.1e6, 2.4e6, "the recording")
        tuned = win._current_frequency()
        win._tune_iss_from_menu()
        settle()
        self.assert_nothing_changed(tuned, "AM", "outside the recording")


FS = 48000.0
GUI_BLOCK = 1600


def _robot36_audio():
    from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card

    return encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)


class TestSstvImageKeptWhenAudioEnds(_HamWindowCase):
    """HAMDSP-2: an image coming in when the window closed was lost, and
    after the receiver stopped (or left FM) it waited, unsaved, for audio
    that never came. The lines received so far are now kept and
    auto-saved in both cases."""

    def receive_half(self):
        self.sstv._start_btn.click()
        audio = _robot36_audio()
        half = audio[: len(audio) // 2]
        for i in range(0, len(half), GUI_BLOCK):
            self.sstv.process_audio(half[i : i + GUI_BLOCK], FS)
        self.assertTrue(self.sstv._decoder.is_receiving())

    def saved_partials(self):
        return sorted(
            p.name for p in Path(self.tmp.name).glob("sstv_*_Robot_36_partial_*.png")
        )

    def test_closing_the_window_keeps_the_partial_image(self):
        self.receive_half()
        self.assertEqual(self.saved_partials(), [])
        self.win.close()
        self.assertEqual(len(self.saved_partials()), 1)
        self.assertEqual(self.sstv._history_list.count(), 1)
        self.assertIn("partial", self.sstv._history_list.item(0).text())

    def test_audio_that_stops_ends_the_image_without_more_audio(self):
        self.receive_half()
        self.win._stop_acquisition()  # no more audio comes
        self.sstv._last_audio_time -= 2.0  # ... for two seconds
        self.sstv._update_status()  # the panel's 100 ms tick
        self.assertFalse(self.sstv._decoder.is_receiving())
        self.assertTrue(self.sstv.is_receiving(), "still listening")
        self.assertEqual(len(self.saved_partials()), 1)
        self.assertIn("audio stopped", self.sstv._status_label.text())
        self.assertEqual(self.sstv._history_list.count(), 1)

    def test_a_short_silence_does_not_end_the_image(self):
        self.receive_half()
        self.sstv._last_audio_time -= 0.5
        self.sstv._update_status()
        self.assertTrue(self.sstv._decoder.is_receiving())
        self.assertEqual(self.saved_partials(), [])

    def test_a_few_lines_that_stop_leave_the_panel_listening(self):
        from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card

        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS, lines=4)
        self.sstv._start_btn.click()
        for i in range(0, len(audio), GUI_BLOCK):
            self.sstv.process_audio(audio[i : i + GUI_BLOCK], FS)
        self.assertTrue(self.sstv._decoder.is_receiving())
        self.sstv._last_audio_time -= 2.0
        self.sstv._update_status()
        self.assertFalse(self.sstv._decoder.is_receiving())
        self.assertEqual(self.sstv._history_list.count(), 0, "too few to keep")
        self.assertNotIn("Receiving", self.sstv._status_label.text())
        self.assertIn("audio stopped", self.sstv._status_label.text())
        self.assertFalse(self.sstv._image_display.has_image())
        self.assertTrue(self.sstv._progress_bar.isHidden())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
