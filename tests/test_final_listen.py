#!/usr/bin/env python3
"""
Final fixes in listening (the receiver chain, the band plan, presets and
the tuning messages), found by the end-to-end checks of the finished app.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_listen.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest


# --------------------------------------------------------------------------- #
# The band plan (no Qt)
# --------------------------------------------------------------------------- #


class TestBandPlanSegments(unittest.TestCase):
    """LISTEN-03, LISTEN-04, leftover 14: how a frequency is usually heard."""

    def test_cw_and_data_segments_of_ssb_bands_are_heard_in_cw(self):
        from sdr_module.core.band_plan import listening_for

        # 40 m and 80 m: LSB bands, whose low ends are CW/data only.
        self.assertEqual(listening_for(7.040e6), ("CW", "", "500 Hz"))
        self.assertEqual(listening_for(3.540e6), ("CW", "", "500 Hz"))
        self.assertEqual(listening_for(14.040e6), ("CW", "", "500 Hz"))
        # Their phone segments stay in the band's sideband.
        self.assertEqual(listening_for(7.200e6), ("LSB", "", "2.7 kHz"))
        self.assertEqual(listening_for(3.850e6), ("LSB", "", "2.7 kHz"))
        self.assertEqual(listening_for(14.200e6), ("USB", "", "2.7 kHz"))
        # 160 m is all modes: the band's LSB.
        self.assertEqual(listening_for(1.900e6), ("LSB", "", "2.7 kHz"))

    def test_digital_mode_channels_are_heard_in_usb(self):
        from sdr_module.core.band_plan import listening_for, segment_listening

        for freq in (1.840e6, 3.573e6, 7.074e6, 7.0475e6, 10.136e6, 14.074e6):
            self.assertEqual(listening_for(freq), ("USB", "", "3.0 kHz"), freq)
        # Up to 3 kHz above the dial (where the signals are).
        self.assertEqual(listening_for(7.0765e6), ("USB", "", "3.0 kHz"))
        self.assertEqual(segment_listening(7.074e6).name, "FT8 40m")
        self.assertTrue(segment_listening(7.074e6).channel)
        self.assertFalse(segment_listening(7.04e6).channel)
        self.assertIsNone(segment_listening(7.2e6))  # the band's own mode
        self.assertIsNone(segment_listening(100.1e6))

    def test_murs_and_frs_are_narrow_fm(self):
        from sdr_module.core.band_plan import band_for, listening_for

        self.assertEqual(band_for(151.85e6).name, "MURS")
        self.assertEqual(listening_for(151.85e6), ("FM", "2.5 kHz", "12.5 kHz"))
        self.assertEqual(listening_for(154.57e6), ("FM", "5 kHz", "25 kHz"))
        for freq in (462.5625e6, 462.7e6, 467.7125e6):
            self.assertEqual(band_for(freq).name, "FRS/GMRS", freq)
            self.assertEqual(listening_for(freq), ("FM", "2.5 kHz", "12.5 kHz"))
        self.assertIsNone(band_for(169.0e6))  # still nothing there

    def test_446_mhz_is_the_70cm_simplex_segment(self):
        # Leftover 14: the European PMR446 band (2.5 kHz deviation) claimed
        # 446.0-446.2 MHz, around the US 70 cm calling frequency, in a band
        # plan that is otherwise the US one.
        from sdr_module.core.band_plan import band_for, default_name, listening_for

        self.assertEqual(band_for(446.1e6).name, "70cm Ham")
        self.assertEqual(default_name(446.0e6), "70cm Calling")
        self.assertEqual(default_name(446.0006e6), "70cm Ham 446.0006")
        self.assertEqual(listening_for(446.1e6), ("FM", "5 kHz", "25 kHz"))


# --------------------------------------------------------------------------- #
# Choose Mode From Band Plan in the window
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBandPlanListeningSegments(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")  # broadcast FM
        self.win.set_frequency(100.1e6)
        self.panel = self.win._control_panel

    def type_frequency(self, text):
        spin = self.win._frequency_field()
        spin.setFocus()
        self.settle()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, text)
        QTest.keyClick(spin, Qt.Key.Key_Return)

    def listening(self):
        return (
            self.panel.get_demod_mode(),
            self.panel._fm_dev_combo.currentText(),
            self.panel._bw_combo.currentText(),
        )

    def message(self):
        return self.win._message_label.text()

    def test_the_cw_segment_of_40_m_is_heard_in_cw(self):
        # LISTEN-03: 7.040 MHz got LSB while FREQ's band plan line said
        # "CW/Data".
        self.type_frequency("7.04")
        self.assertEqual(self.panel.get_demod_mode(), "CW")
        self.assertEqual(self.panel._bw_combo.currentText(), "500 Hz")
        self.assertTrue(
            self.message().startswith("40m Ham, CW/data segment: listening in CW"),
            self.message(),
        )
        self.assertIn("as this part of the band is usually heard", self.message())

    def test_ft8_channels_are_heard_in_usb(self):
        for text in ("3.573", "7.074"):
            self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
            self.win.set_frequency(100.1e6)
            self.type_frequency(text)
            self.assertEqual(self.panel.get_demod_mode(), "USB", text)
            self.assertEqual(self.panel._bw_combo.currentText(), "3.0 kHz", text)
            self.assertIn("FT8", self.message())
            self.assertIn("as this channel is usually heard", self.message())

    def test_from_i_q_and_am_too(self):
        self.win._set_demod_mode("None (I/Q)")
        self.type_frequency("7.04")
        self.assertEqual(self.panel.get_demod_mode(), "CW")
        self.type_frequency("118.1")
        self.assertEqual(self.panel.get_demod_mode(), "AM")
        self.type_frequency("7.074")
        self.assertEqual(self.panel.get_demod_mode(), "USB")

    def test_the_phone_segment_stays_lsb(self):
        self.type_frequency("7.2")
        self.assertEqual(self.panel.get_demod_mode(), "LSB")

    def test_moving_into_the_cw_segment_of_the_band_follows_it(self):
        self.type_frequency("7.2")  # 40 m phone: LSB
        self.type_frequency("7.04")  # 40 m CW/data
        self.assertEqual(self.panel.get_demod_mode(), "CW")
        self.type_frequency("7.06")  # within the segment: CW stays
        self.assertEqual(self.panel.get_demod_mode(), "CW")

    def test_usb_stays_for_data_in_a_cw_data_segment(self):
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.type_frequency("7.04")  # RTTY and the digital modes are USB
        self.assertEqual(self.panel.get_demod_mode(), "USB")

    def test_cw_stays_cw_but_an_ft8_channel_is_usb(self):
        self.win.set_frequency(14.06e6)
        self.win._set_demod_mode("CW")
        self.type_frequency("7.03")
        self.assertEqual(self.panel.get_demod_mode(), "CW")
        self.type_frequency("7.074")
        self.assertEqual(self.panel.get_demod_mode(), "USB")

    def test_murs_and_frs_are_heard_in_narrow_fm(self):
        # LISTEN-04: from USB on 20 m they kept USB (no band in the table).
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.type_frequency("151.85")
        self.assertEqual(self.listening(), ("FM", "2.5 kHz", "12.5 kHz"))
        self.assertTrue(self.message().startswith("MURS: listening in FM"))
        self.type_frequency("154.57")  # MURS 4-5: wide (5 kHz deviation)
        self.assertEqual(self.listening(), ("FM", "2.5 kHz", "12.5 kHz"))
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.type_frequency("462.5625")
        self.assertEqual(self.listening(), ("FM", "2.5 kHz", "12.5 kHz"))
        self.assertTrue(self.message().startswith("FRS/GMRS: listening in FM"))
        self.type_frequency("467.7125")
        self.assertEqual(self.panel.get_demod_mode(), "FM")

    def test_murs_from_broadcast_fm_is_narrow(self):
        self.type_frequency("151.85")
        self.assertEqual(self.listening(), ("FM", "2.5 kHz", "12.5 kHz"))


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------- #
# The status line around tuning and mode changes
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTuningMessages(_WindowTestCase):
    """LISTEN-13: the status line describes the frequency and mode tuned."""

    def setUp(self):
        super().setUp()
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")  # broadcast FM
        self.win.set_frequency(100.1e6)
        self.panel = self.win._control_panel
        self.win._status_history.clear()

    type_frequency = TestBandPlanListeningSegments.type_frequency

    def message(self):
        return self.win._message_label.text()

    def history(self):
        return [entry.text for entry in self.win._status_history.entries()]

    def test_a_band_plan_message_goes_when_you_tune_out_of_its_band(self):
        self.type_frequency("145.29")
        self.assertTrue(self.message().startswith("2m Ham: listening"))
        self.type_frequency("146.94")  # the same band: still true
        self.assertTrue(self.message().startswith("2m Ham: listening"))
        self.type_frequency("446.1")  # 70 cm: listened to the same way
        self.assertNotIn("2m Ham", self.message())
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.win.set_frequency(100.1e6)
        self.type_frequency("144.39")  # APRS
        self.assertTrue(self.message().startswith("APRS: listening"))
        self.type_frequency("162550000")  # NOAA: the same narrow FM
        self.assertNotIn("APRS", self.message())

    def test_the_band_plan_says_once_what_it_set(self):
        # The mode change itself said "LSB: 2.7 kHz filter, 200-2900 Hz"
        # first (and "AM: 15 kHz" before "... AM with a 10 kHz filter").
        self.type_frequency("7.2")
        self.assertEqual(len(self.history()), 1, self.history())
        self.assertTrue(self.history()[0].startswith("40m Ham: listening in LSB"))

    def test_a_preset_says_once_what_it_set(self):
        self.assertTrue(self.win._apply_preset_named("Air Traffic Control"))
        self.assertEqual(
            self.history(), ["Air Traffic Control: 127.850 MHz, AM, 15 kHz filter"]
        )

    def test_i_q_mode_says_so(self):
        self.win._set_demod_mode("CW")
        self.win._set_demod_mode("None (I/Q)")
        self.assertTrue(self.message().startswith("None (I/Q): "), self.message())
        self.assertIn("no audio", self.message())
        self.assertNotIn("CW", self.message())

    def test_choosing_a_mode_by_hand_still_says_what_it_listens_with(self):
        self.win._set_demod_mode("USB")
        self.assertTrue(self.message().startswith("USB: "), self.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestPresetOutOfRange(_WindowTestCase):
    """Leftover 19: a preset outside the device's range retuned to the edge
    of the range (the Frequency field's clamped value) without a word."""

    def test_it_is_refused_with_the_reason(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._device_tuning_range = lambda: (24e6, 1766e6)
        win._apply_tuning_range()
        self.assertFalse(win._apply_preset_named("QRP 40m CW"))  # 7.030 MHz
        self.assertEqual(win._current_frequency(), 100.1e6)
        self.assertIn("7.030 MHz is out of range", win._message_label.text())
        self.assertEqual(win._control_panel.get_demod_mode(), "FM")


# --------------------------------------------------------------------------- #
# A preset's decoder
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestPresetDecoderLifetime(_WindowTestCase):
    """LISTEN-05: the decoder a preset turned on stayed on after another
    preset or a far tune (ADS-B decoding at 100.1 MHz; SSTV listening, and
    warning about the mode, on every airband tune)."""

    demo = True

    type_frequency = TestBandPlanListeningSegments.type_frequency

    def decoding(self):
        panel = self.win._decoder_panel
        return panel._proto_combo.currentText(), panel._enabled_check.isChecked()

    def history(self):
        return [entry.text for entry in self.win._status_history.entries()]

    def test_another_preset_turns_the_last_ones_decoder_off(self):
        self.assertTrue(self.win._apply_preset_named("ADS-B"))
        self.assertEqual(self.decoding(), ("ADS-B", True))
        self.assertTrue(self.win._apply_preset_named("FM Broadcast"))
        self.assertEqual(self.decoding(), ("ADS-B", False))
        self.assertIn("ADS-B decoder off", self.win._message_label.text())

    def test_a_preset_with_a_decoder_of_its_own_switches_to_it(self):
        self.win._apply_preset_named("ADS-B")
        self.win._apply_preset_named("ACARS")
        self.assertEqual(self.decoding(), ("ACARS", True))

    def test_a_far_tune_out_of_its_band_turns_it_off(self):
        self.win._apply_preset_named("ACARS")
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.type_frequency("131.6")  # still where ACARS is heard
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.type_frequency("100.1")
        self.assertEqual(self.decoding(), ("ACARS", False))
        self.assertTrue(
            any(text.startswith("ACARS decoder off") for text in self.history()),
            self.history(),
        )

    def test_a_decoder_you_changed_is_yours(self):
        self.win._apply_preset_named("ADS-B")
        combo = self.win._decoder_panel._proto_combo
        combo.setCurrentIndex(combo.findText("POCSAG"))
        self.win._decoder_panel._enabled_check.setChecked(True)
        self.type_frequency("100.1")
        self.assertEqual(self.decoding(), ("POCSAG", True))

    def test_the_sstv_preset_stops_listening_when_you_leave_it(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("no ham radio tools")
        win = self.win
        if not win._is_running:
            win._toggle_acquisition()
        self.assertTrue(win._apply_preset_named("ISS SSTV/Voice"))
        self.assertTrue(win._sstv_listening())
        win._status_history.clear()
        self.type_frequency("125")  # airband: AM
        self.assertFalse(win._sstv_listening())
        texts = self.history()
        self.assertFalse(any("SSTV decoding needs" in text for text in texts), texts)
        self.assertTrue(any(text.startswith("SSTV decoder stopped") for text in texts))
