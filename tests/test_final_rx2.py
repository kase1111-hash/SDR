#!/usr/bin/env python3
"""
Final fixes in the Second Receiver window, found by the last review and the
user walk of the smaller limitations (R7).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_rx2.py``
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

AUDIO = 48000.0


class _TypingCase(_SecondCase):
    """The second receiver started on its demo device as it opens: 145.800
    MHz, narrow FM (5 kHz deviation, 25 kHz filter)."""

    def setUp(self):
        super().setUp()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx.tune(145.8e6, recenter=True)

    def type_frequency(self, text):
        """Type ``text`` into Tuning > Frequency and press Enter."""
        self.rx.activateWindow()
        spin = self.rx._freq_input._freq_input
        spin.setFocus()
        self.settle()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, text)
        QTest.keyClick(spin, Qt.Key.Key_Return)

    def listening(self):
        rx = self.rx
        mode = rx.mode()
        deviation = rx._deviation_value() if mode == "FM" else ""
        return mode, deviation, rx._filter_combo.currentText()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestATypedFrequencyFollowsTheBandPlan(_TypingCase):
    """R7W-02: a broadcast station typed into the Second Receiver's
    Frequency (from 145.800 MHz, narrow FM) stayed in narrow FM, 5 kHz and
    a 25 kHz filter: clipped mono, no STEREO lamp, no word why. It now
    listens the way the band is usually heard, as the main window does:
    the main window's Radio > Choose Mode From Band Plan is the switch for
    both receivers."""

    def test_a_broadcast_station_is_heard_as_broadcast_fm_and_said_so(self):
        rx = self.rx
        self.type_frequency("97.1")
        self.assertEqual(rx.tuned_frequency(), 97.1e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.assertEqual(rx._dsp.deviation_hz, 75e3)
        self.assertEqual(rx._dsp.bandwidth_hz, 200e3)
        text = rx.message()
        self.assertTrue(
            text.startswith(
                "FM Broadcast: listening in FM (broadcast, 75 kHz deviation) "
                "with a 200 kHz filter, as the band is usually heard"
            ),
            text,
        )
        self.assertIn("Radio > Choose Mode From Band Plan", text)
        # The 2 m band again: narrow FM, as the main window hears it.
        self.type_frequency("145.29")
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertTrue(rx.message().startswith("2m Ham: listening in FM (narrow"))

    def test_a_tone_squelch_chosen_stays_through_the_bands(self):
        # (Broadcast FM isn't checked for a tone; back on the repeater the
        # tone chosen for it is.)
        rx = self.rx
        rx._set_tone_choice(100.0, None)
        self.type_frequency("97.1")
        self.assertEqual(rx._tone_choice(), (100.0, None))
        self.assertFalse(rx._dsp._tones_checked())
        self.type_frequency("145.29")
        self.assertEqual(rx._tone_choice(), (100.0, None))
        self.assertTrue(rx._dsp._tones_checked())

    def test_the_main_windows_rule_for_every_kind_of_place(self):
        rx = self.rx
        self.type_frequency("118.3")  # the airband: AM
        self.assertEqual(self.listening(), ("AM", "", "10 kHz"))
        self.type_frequency("7.03")  # QRP 40m CW, a preset channel: CW
        self.assertEqual(rx.mode(), "CW")
        self.assertTrue(rx.message().startswith("QRP 40m CW: listening in CW"))
        self.type_frequency("152.24")  # POCSAG: the preset's 12.5 kHz
        self.assertEqual(self.listening(), ("FM", "12.5 kHz", "25 kHz"))
        self.assertIn("POCSAG", rx.message())

    def test_the_last_bands_message_goes_with_it(self):
        # (As in the main window, LISTEN-13: "2m Ham: listening in FM ..."
        # stayed at 446.1 MHz.)
        rx = self.rx
        self.type_frequency("97.1")
        self.type_frequency("145.29")
        self.assertTrue(rx.message().startswith("2m Ham: listening in FM"))
        self.type_frequency("446.1")  # 70 cm: narrow FM already
        self.assertEqual(rx.tuned_frequency(), 446.1e6)
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertEqual(rx.message(), "")

    def test_a_mode_or_filter_chosen_since_takes_the_message_down(self):
        # (It said FM while the Mode box said AM; the main window says the
        # mode chosen instead.)
        rx = self.rx
        self.type_frequency("97.1")
        self.assertTrue(rx.message().startswith("FM Broadcast:"))
        rx._mode_combo.setCurrentText("AM")
        self.assertEqual(rx.message(), "")
        self.type_frequency("145.29")
        self.assertTrue(rx.message().startswith("2m Ham:"))
        rx._filter_combo.setCurrentText("12.5 kHz")
        self.assertEqual(rx.message(), "")
        self.type_frequency("97.1")
        self.assertTrue(rx.message().startswith("FM Broadcast:"))
        rx._set_deviation("25 kHz")
        self.assertEqual(rx.message(), "")
        # (Another message is not the band plan's: it stays.)
        rx._show_message("Something else", "info", 3000)
        rx._mode_combo.setCurrentText("AM")
        self.assertEqual(rx.mode(), "AM")
        self.assertEqual(rx.message(), "Something else")

    def test_within_the_band_your_mode_stays(self):
        rx = self.rx
        rx._mode_combo.setCurrentText("AM")  # picked by hand on 2 m
        self.type_frequency("146.52")
        self.assertEqual(rx.tuned_frequency(), 146.52e6)
        self.assertEqual(rx.mode(), "AM")
        self.assertNotIn("Choose Mode From Band Plan", rx.message())

    def test_a_mode_picked_after_the_frequency_stays(self):
        rx = self.rx
        self.type_frequency("97.1")
        rx.set_mode("FM", "5 kHz", "25 kHz")  # picked by hand there
        self.type_frequency("101.1")  # the same band
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))

    def test_the_fields_arrows_are_steps_and_keep_the_mode(self):
        # (As Left/Right on the plots and in the main window: from broadcast
        # FM at 107.9 MHz up into the airband, FM stays.)
        rx = self.rx
        rx.tune(107.9e6, recenter=True)
        rx.set_mode("FM", "75 kHz", "200 kHz")
        step = rx._step_hz
        self.rx.activateWindow()
        spin = rx._freq_input._freq_input
        spin.setFocus()
        self.settle()
        for _ in range(3):
            QTest.keyClick(spin, Qt.Key.Key_Up)
        self.assertAlmostEqual(rx.tuned_frequency(), 107.9e6 + 3 * step)
        self.assertGreater(rx.tuned_frequency(), 108e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))

    def test_the_main_windows_switch_turns_it_off_for_both(self):
        action = self.win._band_plan_action
        self.assertTrue(action.isChecked())
        self.assertIn("Second Receiver", action.toolTip())
        action.trigger()
        self.assertFalse(action.isChecked())
        self.type_frequency("97.1")
        self.assertEqual(self.rx.tuned_frequency(), 97.1e6)
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        action.trigger()
        self.type_frequency("118.3")
        self.assertEqual(self.rx.mode(), "AM")

    def test_a_bookmark_without_a_mode_gets_the_band_plan(self):
        rx = self.rx
        self.assertTrue(rx.tune_bookmark({"label": "Tower", "freq_hz": 118.3e6}))
        self.assertEqual(self.listening(), ("AM", "", "10 kHz"))
        self.assertTrue(rx.message().startswith("Airband: listening in AM"))

    def test_a_bookmark_with_a_mode_keeps_its_own(self):
        from sdr_module.gui.bookmarks_panel import _new_bookmark

        rx = self.rx
        bookmark = _new_bookmark("Talk", 97.1e6, "AM", "", "10 kHz")
        self.assertTrue(rx.tune_bookmark(bookmark))
        self.assertEqual(self.listening(), ("AM", "", "10 kHz"))
        self.assertEqual(rx.message(), "Tuned to Talk (AM)")

    def test_a_bookmark_in_the_bands_mode_keeps_its_deviation(self):
        # (Narrow FM on a broadcast station: FM is the band's mode too, but
        # the bookmark's 5 kHz and 25 kHz filter are its own.)
        from sdr_module.gui.bookmarks_panel import _new_bookmark

        rx = self.rx
        bookmark = _new_bookmark("Narrow", 97.1e6, "FM", "5 kHz", "25 kHz")
        self.assertTrue(rx.tune_bookmark(bookmark))
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertEqual(rx.message(), "Tuned to Narrow (FM)")
        # One without a mode where the band plan knows nothing keeps it too.
        self.assertTrue(rx.tune_bookmark({"label": "Somewhere", "freq_hz": 300e6}))
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertEqual(rx.message(), "Tuned to Somewhere (FM)")

    def test_where_the_band_plan_knows_nothing_the_mode_stays(self):
        # (No band, segment or preset channel there: nothing to go by, as
        # in the main window. The message about the band left goes.)
        rx = self.rx
        self.type_frequency("97.1")
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.type_frequency("300")
        self.assertEqual(rx.tuned_frequency(), 300e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.assertEqual(rx.message(), "")
        rx._mode_combo.setCurrentText("AM")  # picked by hand there
        self.type_frequency("250")
        self.assertEqual(rx.tuned_frequency(), 250e6)
        self.assertEqual(rx.mode(), "AM")
        self.assertEqual(rx.message(), "")

    def test_a_sideband_chosen_by_hand_takes_the_message_down(self):
        # (USB and LSB share the 2.7 kHz filter: only the mode changes, and
        # "listening in LSB" no longer says how this receiver listens.)
        rx = self.rx
        self.type_frequency("7.2")
        self.assertEqual(self.listening(), ("LSB", "", "2.7 kHz"))
        self.assertTrue(rx.message().startswith("40m Ham: listening in LSB"))
        rx._mode_combo.setCurrentText("USB")
        self.assertEqual(self.listening(), ("USB", "", "2.7 kHz"))
        self.assertEqual(rx.message(), "")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTypedStationPlaysInStereo(_TypingCase):
    """R7W-02 as the walker heard it: Listen set to this receiver, 97.1
    typed in. It played hard-clipped mono (peak 1.00, 55.6% of the samples
    at full scale) with the STEREO lamp hidden; now the demo station's
    stereo (its 3.7 kHz tone on the left only) plays, unclipped, with the
    lamp lit."""

    def setUp(self):
        super().setUp()
        from sdr_module.dsp.fm_stereo import StereoDecoder
        from sdr_module.gui import device_dialog

        for target, name, value in (
            (device_dialog, "_MAX_LAG_S", 1e9),
            (StereoDecoder, "RISE_S", 0.05),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        rx.device._rng = np.random.default_rng(3)
        rx.set_listen("second")
        rx._sync_audio_output()

    def heard(self):
        both = [
            b if b.ndim == 2 else np.stack((b, b), axis=1) for b in self.speaker.blocks
        ]
        return np.concatenate(both) if both else np.zeros((0, 2))

    def test_stereo_unclipped_with_the_lamp_lit(self):
        from tests.test_fm_stereo import _db, _tone

        rx = self.rx
        self.type_frequency("97.1")
        self.speaker.blocks.clear()
        self.feed(20)
        badge = rx._stereo_badge
        self.assertTrue(badge.isVisible())
        self.assertEqual(badge.text(), "STEREO")
        audio = self.heard()
        self.assertGreater(len(audio), int(0.5 * AUDIO))
        self.assertLess(float(np.max(np.abs(audio))), 0.99)
        self.assertLess(float(np.mean(np.abs(audio) >= 0.999)), 0.001)
        tail = audio[-int(0.25 * AUDIO) :]
        left = _tone(tail[:, 0], 3.7e3, AUDIO)
        right = _tone(tail[:, 1], 3.7e3, AUDIO)
        self.assertGreater(left, 0.1)
        self.assertGreater(_db(left, right), 35)


if __name__ == "__main__":
    unittest.main()
