#!/usr/bin/env python3
"""
Final fixes in listening (the receiver chain, the band plan, presets and
the tuning messages), found by the end-to-end checks of the finished app.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_listen.py``
"""

import os
import time
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


# --------------------------------------------------------------------------- #
# Noise reduction on a clean station
# --------------------------------------------------------------------------- #


def _chords(seconds, rate):
    """Music the noise estimate mistook for noise: chords held 3 s each,
    rich in 200-500 Hz harmonics."""
    import numpy as np

    t = np.arange(int(seconds * rate)) / rate
    out = np.zeros(len(t))
    for start, roots in ((0.0, (220.0, 277.2, 329.6)), (3.0, (246.9, 311.1, 370.0))):
        on = (t >= start) & (t < start + 3.0)
        for root in roots:
            for k, weight in ((1, 1.0), (2, 0.5), (3, 0.3)):
                out[on] += weight * np.sin(2 * np.pi * root * k * t[on])
    return 0.5 * out / np.max(np.abs(out))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNoiseReductionLeavesACleanSignalAlone(_WindowTestCase):
    """LISTEN-08: Strong took 2-3 dB off a clean music station (6 dB at
    200-500 Hz): its noise estimate learned held chords as noise."""

    def _listen(self, level, snr_db, deviation="75 kHz", width="200 kHz"):
        import numpy as np

        from tests.test_gui_squelch import (
            FRAME,
            RATE,
            Frames,
            fm,
            noise,
            pretend_audio_output,
        )

        pretend_audio_output(self.win)
        win = self.win
        win.set_frequency(100.1e6 if deviation == "75 kHz" else 146.52e6)
        win._set_demod_mode("AM")
        win._set_demod_mode("FM", deviation, width)
        win._control_panel.set_noise_reduction(level)
        win._reset_demodulator()
        audio = _chords(6.0, RATE)
        dev = 75e3 if deviation == "75 kHz" else 5e3
        # (``snr_db`` in the channel: the noise spreads over the span.)
        spread_db = 10 * np.log10(RATE / float(width.split()[0]) / 1e3)
        iq = fm(audio, dev, -20.0) + noise(
            len(audio), -20.0 - snr_db + spread_db, seed=8
        )
        frames = Frames(self, iq[: len(iq) // FRAME * FRAME]).run()
        out = frames.played()
        return out[int(1.5 * 48000) : int(5.5 * 48000)]

    @staticmethod
    def _rms_db(audio):
        import numpy as np

        return 10 * np.log10(np.mean(np.square(audio)) + 1e-30)

    def test_strong_and_light_leave_a_clean_station_at_its_level(self):
        off = self._rms_db(self._listen("off", 60.0))
        for level in ("light", "strong"):
            on = self._rms_db(self._listen(level, 60.0))
            self.assertAlmostEqual(on, off, delta=0.3, msg=level)  # was -2.7 dB

    def test_a_noisy_station_still_loses_its_hiss(self):
        import numpy as np

        # Narrow FM 15 dB above the noise: the hiss above 4 kHz still goes.
        def hiss(audio):
            spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
            freqs = np.fft.rfftfreq(len(audio), 1 / 48000)
            return float(spectrum[(freqs > 1.5e3) & (freqs < 3e3)].sum())

        off = hiss(self._listen("off", 15.0, "5 kHz", "25 kHz"))
        strong = hiss(self._listen("strong", 15.0, "5 kHz", "25 kHz"))
        self.assertLess(strong, 0.5 * off)


# --------------------------------------------------------------------------- #
# The tone squelch at a broadcast deviation
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestToneSquelchOnBroadcastFm(_WindowTestCase):
    """LISTEN-09: a tone squelch left set from a repeater chopped a
    broadcast station into ~20 % fragments (the music's bass notes passed
    for CTCSS tones) and Tone heard listed fake tones."""

    def test_broadcast_fm_ignores_the_tone_squelch(self):
        from tests.test_gui_audio_options import _voice
        from tests.test_gui_squelch import (
            FRAME,
            RATE,
            Frames,
            fm,
            noise,
            pretend_audio_output,
        )

        pretend_audio_output(self.win)
        panel = self.win._control_panel
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        panel.set_squelch_db(-90)
        panel.set_tone_squelch_hz(131.8)
        # A bass note right on a CTCSS tone (110.9 Hz) under the programme.
        audio = _voice(3.0, tone_hz=110.9)
        iq = fm(audio, 75e3, -30.0) + noise(int(3.0 * RATE), -95.0)
        n = len(iq) - len(iq) % FRAME
        self.win._reset_demodulator()
        frames = Frames(self, iq[:n]).run()
        self.assertTrue(all(len(a) for a in frames.audio[1:]))  # never muted
        self.assertEqual(panel._tone_heard.text(), "Not checked")
        self.assertNotEqual(self.win._squelch_state(True), "tone")

    def test_setting_one_on_broadcast_fm_says_it_is_for_narrow_fm(self):
        panel = self.win._control_panel
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        panel.set_tone_squelch_hz(100.0)
        self.assertIn("narrow FM", self.win._message_label.text())


# --------------------------------------------------------------------------- #
# The radio tuner's seek
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSeekNeedsABroadcastStation(_WindowTestCase):
    """LISTEN-10: FM seek stopped on 100.7 MHz for the demo's narrow FM
    carrier 15 kHz off the channel (its power is centred within half a
    100 kHz step), not on the next broadcast station."""

    def _decide(self, bumps):
        """The tuner's verdict on a 200 kHz FM channel at 100.7 MHz whose
        spectrum holds ``bumps`` ((offset Hz, half width Hz, dB above the
        noise), ...)."""
        import numpy as np

        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("no ham radio tools")
        win = self.win
        win.set_frequency(100.7e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        win._show_radio_tuner()
        chain = win._receiver_chain()
        n = 2048
        freqs = (np.arange(n) - n // 2) * chain.rate / n - chain.offset_hz
        power = np.full(n, -95.0) + np.random.default_rng(1).normal(0, 1, n)
        for offset, half, above in bumps:
            power[np.abs(freqs - offset) <= half] = -95.0 + above
        for _ in range(5):
            win._track_noise_floor(power, 80000)
        win._last_peak_db = -40.0  # (the channel opens the squelch)
        win._tuner_block(np.zeros(16, dtype=np.complex64), chain, power)
        return win._tuner_signal and win._tuner_centered

    def test_a_narrow_carrier_off_centre_is_no_station(self):
        self.assertFalse(self._decide([(15e3, 6e3, 40.0)]))

    def test_a_broadcast_station_is_one(self):
        self.assertTrue(self._decide([(0.0, 90e3, 40.0)]))

    def test_a_weak_broadcast_station_is_one_too(self):
        self.assertTrue(self._decide([(0.0, 40e3, 12.0)]))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTunerHintFollowsTheTuning(unittest.TestCase):
    """LISTEN-11: after a seek said "Found a station on 101.2 MHz.", a
    Shift+click (one channel up) or a retune elsewhere left that hint for
    up to 4 s on a tuner showing another station."""

    def setUp(self):
        from PyQt6.QtWidgets import QApplication

        from sdr_module.ham.gui import radio_tuner

        self.app = QApplication.instance() or QApplication([])
        self.tuner = t = radio_tuner.RadioTunerWidget(None, 2.4e6)
        t.set_signal_probe(
            lambda: "signal" if round(t.get_frequency()) == 101200000 else "quiet"
        )
        t.show()
        t._seek_up_btn.click()  # from 101.1 MHz
        deadline = time.monotonic() + 3
        while t.is_seeking() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertEqual(t._hint_label.text(), "Found a station on 101.2 MHz.")

    def tearDown(self):
        self.tuner.close()
        self.tuner.deleteLater()
        self.app.processEvents()

    def assert_standing_footer(self):
        t = self.tuner
        self.assertEqual(t._hint_label.text(), t._FOOTER_HINT)
        self.assertFalse(t._hint_label.property("tone"))  # (not "success")
        self.assertFalse(t._hint_timer.isActive())

    def test_shift_click_steps_and_drops_the_seek_result(self):
        t = self.tuner
        QTest.mouseClick(
            t._seek_up_btn, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertAlmostEqual(t.get_frequency(), 101.3e6)
        self.assertFalse(t.is_seeking())
        self.assert_standing_footer()

    def test_tuning_the_dial_drops_it(self):
        t = self.tuner
        t._tuning_dial.step(-1)
        self.assertAlmostEqual(t.get_frequency(), 101.1e6)
        self.assert_standing_footer()

    def test_a_retune_elsewhere_drops_it(self):
        """The main window retuned the receiver (a click on the spectrum)."""
        t = self.tuner
        t.set_frequency(100.9e6)
        self.assert_standing_footer()

    def test_the_receivers_echo_of_the_station_keeps_it(self):
        t = self.tuner
        t.set_frequency(101.2e6)
        self.assertEqual(t._hint_label.text(), "Found a station on 101.2 MHz.")


# --------------------------------------------------------------------------- #
# Frequencies below 2 MHz, and below 0 Hz
# --------------------------------------------------------------------------- #


class TestKilohertzBelowTwoMegahertz(unittest.TestCase):
    """LISTEN-12: medium-wave stations read "0.880 MHz" everywhere but in
    the radio tuner ("880 kHz", as listeners know them)."""

    def test_format_frequency(self):
        from sdr_module.gui.main_window import format_frequency

        self.assertEqual(format_frequency(880e3), "880 kHz")
        self.assertEqual(format_frequency(1260.5e3), "1260.5 kHz")
        self.assertEqual(format_frequency(198e3), "198 kHz")
        self.assertEqual(format_frequency(1999.999e3), "1999.999 kHz")
        self.assertEqual(format_frequency(2e6), "2.000 MHz")
        self.assertEqual(format_frequency(7.074e6), "7.074 MHz")
        self.assertEqual(format_frequency(100.1e6), "100.100 MHz")

    def test_signed(self):
        """LISTEN-14: a refused frequency below 0 Hz is named as asked."""
        from sdr_module.gui.main_window import format_frequency

        self.assertEqual(format_frequency(-300e3), "-300 kHz")
        self.assertEqual(format_frequency(-40e3), "-40 kHz")
        self.assertEqual(format_frequency(-2.5e6), "-2.500 MHz")
        self.assertEqual(format_frequency(0.0), "0 kHz")
        self.assertEqual(format_frequency(-0.0), "0 kHz")

    def test_bookmark_names(self):
        from sdr_module.core.band_plan import default_name

        self.assertEqual(default_name(880e3), "AM Broadcast 880 kHz")
        self.assertEqual(default_name(1260.5e3), "AM Broadcast 1260.5 kHz")
        self.assertEqual(default_name(198e3), "LW Broadcast 198 kHz")
        self.assertEqual(default_name(1.84e6), "160m Ham 1840 kHz")
        self.assertEqual(default_name(100.1e6), "FM Broadcast 100.1")
        self.assertEqual(default_name(7.074e6), "40m Ham 7.074")

    def test_band_plan_line(self):
        from sdr_module.core.band_info import band_plan_line

        self.assertTrue(
            band_plan_line(880e3).startswith("AM Broadcast · 525–1705 kHz"),
            band_plan_line(880e3),
        )
        self.assertIn("1.8–2 MHz", band_plan_line(1.84e6))
        self.assertIn("87.5–108 MHz", band_plan_line(100.1e6))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestKilohertzInTheWindow(_WindowTestCase):
    """LISTEN-12 in the window: FREQ, the title, the status line, Recent
    Frequencies and Ctrl+B's bookmark."""

    demo = True

    type_frequency = TestBandPlanListeningSegments.type_frequency

    def test_a_medium_wave_station(self):
        win = self.win
        self.type_frequency("880k")
        self.assertEqual(win._current_frequency(), 880e3)
        self.assertEqual(win._freq_label.text(), "880 kHz")
        self.assertTrue(win.windowTitle().startswith("880 kHz AM"), win.windowTitle())
        self.assertIn("525–1705 kHz", win._freq_label.toolTip())
        self.assertIn("No preset at 880 kHz.", win._control_panel._preset_info.text())
        win._commit_history_entry()
        entries = [a.text().replace("&", "") for a in win._recent_menu.actions()]
        self.assertTrue(any("880 kHz (AM)" in text for text in entries), entries)
        win._bookmark_current_frequency()
        self.assertIn("Bookmarked AM Broadcast 880 kHz (AM)", win._message_label.text())

    def test_the_radio_tuner_says_kilohertz_too(self):
        win = self.win
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        win._on_radio_frequency_changed(1260e3, "AM")
        self.assertEqual(win._message_label.text(), "Tuned to 1260 kHz (AM)")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBelowZeroHertz(_WindowTestCase):
    """LISTEN-14: "0.000 MHz is out of range" for a click at -0.3 MHz; the
    refusal stayed after a good tune; the span reached below 0 Hz."""

    demo = True

    type_frequency = TestBandPlanListeningSegments.type_frequency

    def test_the_refusal_names_the_frequency_asked(self):
        win = self.win
        win.set_frequency(660e3)
        self.assertFalse(win._tune_to(-300e3))
        self.assertTrue(
            win._message_label.text().startswith("-300 kHz is out of range"),
            win._message_label.text(),
        )
        self.assertEqual(win._current_frequency(), 660e3)

    def test_a_good_tune_clears_the_refusal(self):
        win = self.win
        win.set_frequency(660e3)
        self.assertFalse(win._tune_to(-300e3))
        self.assertTrue(win._tune_to(50e3))
        self.assertEqual(win._message_label.text(), "")
        self.assertFalse(win._message_timer.isActive())

    def test_other_messages_stay(self):
        win = self.win
        win._show_status_message("Something else", "info", 6000)
        self.assertTrue(win._tune_to(660e3))
        self.assertEqual(win._message_label.text(), "Something else")

    def test_the_span_starts_at_0_hz(self):
        win = self.win
        self.type_frequency("660k")
        rate = win._device_sample_rate()
        self.assertEqual(win._current_frequency(), 660e3)
        self.assertGreaterEqual(win._lo_hz - rate / 2, 0.0)
        self.assertTrue(win._channel_fits(660e3 - win._lo_hz))
        self.assertEqual(win._spectrum._center_freq, win._lo_hz)

    def test_lowest_channel_the_span_can_hold(self):
        """At 50 kHz the channel can't be 1.2 MHz from the LO: the LO goes
        as high as keeps it in the usable span."""
        win = self.win
        win._set_demod_mode("AM")
        self.assertTrue(win._tune_to(50e3))
        self.assertGreater(win._lo_hz, 50e3)
        self.assertTrue(win._channel_fits(50e3 - win._lo_hz))

    def test_above_half_the_span_the_lo_follows_as_before(self):
        win = self.win
        self.type_frequency("7.2")
        self.assertEqual(win._lo_hz, 7.2e6)


# --------------------------------------------------------------------------- #
# Track Signal (AFC)
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTrackSignalMovesEverything(_WindowTestCase):
    """Leftover 11: Track Signal moved the tuned frequency without the
    window's _on_frequency_changed: the title, the band plan, the Presets
    group and the frequency the next far tune starts from stayed where the
    signal was first tuned."""

    demo = True

    def test_a_drift_shows_everywhere_but_keeps_the_station(self):
        from unittest import mock

        win = self.win
        win.set_frequency(438.5e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        tools = win._signal_tools
        with (
            mock.patch.object(win, "_reset_signal_meter_peak") as peak,
            mock.patch.object(win, "_show_rds_station") as rds,
            mock.patch.object(win, "_tuner_retuned") as tuner,
            mock.patch.object(win, "_update_band_plan") as band_plan,
            mock.patch.object(win._control_panel, "show_preset_for") as preset,
        ):
            tools._move_vfo(2400.0)
        self.assertEqual(win._current_frequency(), 438.5024e6)
        self.assertEqual(win._freq_label.text(), "438.5024 MHz")
        self.assertTrue(win.windowTitle().startswith("438.5024 MHz"), win.windowTitle())
        self.assertEqual(win._shown_freq_hz, 438.5024e6)
        band_plan.assert_called_with(438.5024e6)
        preset.assert_called_with(438.5024e6)
        # The same signal: nothing of a new station starts over.
        peak.assert_not_called()
        rds.assert_not_called()
        tuner.assert_not_called()


# --------------------------------------------------------------------------- #
# Audio: Mute without an output, Ctrl+arrows in fields and lists
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestMuteWithoutAnOutput(_WindowTestCase):
    """Leftover 12: with no audio output device the Audio button said "No
    output" while Radio > Audio > Mute showed checked, as if muted by
    you."""

    def no_output(self):
        from unittest import mock

        return mock.patch.object(
            type(self.win),
            "_audio_output_problem",
            return_value="No audio output device",
        )

    def plugged_in(self):
        from unittest import mock

        return (
            mock.patch.object(
                type(self.win), "_audio_output_problem", return_value=None
            ),
            mock.patch.object(self.win._audio, "start", return_value=True),
        )

    def test_mute_is_not_checked_without_an_output(self):
        win = self.win
        with self.no_output():
            win._set_audio_enabled(True)
        self.assertEqual(win._audio_state(), "no_output")
        self.assertEqual(win._audio_button.text().strip(), "No output")
        self.assertFalse(win._mute_action.isChecked())

    def test_ctrl_m_there_tries_the_output_again(self):
        win = self.win
        with self.no_output():
            win._set_audio_enabled(True)
            win._mute_action.trigger()  # Ctrl+M
        self.assertEqual(win._audio_state(), "no_output")
        self.assertFalse(win._mute_action.isChecked())
        self.assertIn("No audio output device", win._message_label.text())
        self.assertNotEqual(self.store.values.get("audio_enabled"), False)
        present, opens = self.plugged_in()
        with present, opens:
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._mute_action.trigger()  # plugged in: it plays
            self.assertEqual(win._audio_state(), "on")
            self.assertFalse(win._mute_action.isChecked())
            win._mute_action.trigger()  # muted by you
        self.assertEqual(win._audio_state(), "muted")
        self.assertTrue(win._mute_action.isChecked())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestCtrlArrowsInFieldsAndLists(_WindowTestCase):
    """Leftover 13: Ctrl+Up / Ctrl+Down (the volume) took the keys from the
    Frequency field (whose Ctrl+arrow steps ten times as far) and from
    lists (which move the current row with them)."""

    def test_the_frequency_field_keeps_them(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._volume_slider.setValue(60)
        field = win._frequency_field()
        field.setFocus()
        self.settle()
        before = field.value()
        QTest.keyClick(field, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(win._volume_slider.value(), 60)
        self.assertGreater(field.value(), before)

    def test_a_list_keeps_them(self):
        win = self.win
        panel = win._bookmarks_panel
        panel.add_bookmark("One", 100.1e6)
        panel.add_bookmark("Two", 101.1e6)
        win._volume_slider.setValue(60)
        tree = panel._list
        win._show_panel("bookmarks")
        tree.setFocus()
        tree.setCurrentItem(tree.topLevelItem(0))
        self.settle()
        QTest.keyClick(tree, Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(win._volume_slider.value(), 60)
        self.assertIs(tree.currentItem(), tree.topLevelItem(1))

    def test_elsewhere_they_set_the_volume(self):
        win = self.win
        win._volume_slider.setValue(60)
        win._focus_plots()
        self.settle()
        QTest.keyClick(
            win._spectrum, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier
        )
        self.assertEqual(win._volume_slider.value(), 70)
