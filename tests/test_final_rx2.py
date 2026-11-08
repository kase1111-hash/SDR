#!/usr/bin/env python3
"""
Final fixes in the Second Receiver window, found by the last review and the
user walk of the smaller limitations (R7).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_rx2.py``
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
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


# --------------------------------------------------------------------------- #
# R8-01: the Second Receiver warns of clipped FM, by the main window's rule
# --------------------------------------------------------------------------- #

RATE = 2.4e6
BLOCK = 80_000  # (one display frame at 2.4 MS/s)


def _clipping_dsp(deviation=5e3, bandwidth=25e3):
    from sdr_module.gui.second_receiver import ReceiverDsp

    dsp = ReceiverDsp()
    dsp.rate = RATE
    dsp.mode = "FM"
    dsp.bandwidth_hz = bandwidth
    dsp.deviation_hz = deviation
    dsp.tuned_hz = 146.52e6
    return dsp


def _frames(dsp, iq, play=True):
    return [
        dsp.process(iq[start : start + BLOCK], play)
        for start in range(0, len(iq), BLOCK)
    ]


class _FakeClippingChain:
    """What the rule reads of a receiver chain: the last FM block's share
    of samples beyond full scale, and its swing (in multiples of the
    deviation it was heard at). Counts how often the swing was asked."""

    def __init__(self, fraction=0.5, swing=(2.0, 1.9)):
        self.last_clip_fraction = fraction
        self.swing = swing
        self.swings_asked = 0

    def clip_swing(self):
        self.swings_asked += 1
        return self.swing


class TestTheClippingRule(unittest.TestCase):
    """R8-01: the FM clipping warning's rule, out of the main window into
    one Qt-free helper (``audio_chain.ClipWarning``) both receivers use:
    a clear signal only, clipping for half a second, at most every 30 s,
    naming the deviation that holds the swing measured while it clipped."""

    def setUp(self):
        from sdr_module.gui.audio_chain import ClipWarning

        self.rule = ClipWarning()

    def run_blocks(self, chain, deviation, blocks, snr=30.0, start=100.0):
        """``blocks`` blocks of 1/30 s from ``start`` (s): when it warned."""
        said = []
        for i in range(blocks):
            now = start + i / 30
            if self.rule.update(chain, deviation, 1 / 30, snr, now):
                said.append(now)
        return said

    def test_half_a_second_of_clipping_then_at_most_every_30_s(self):
        chain = _FakeClippingChain()
        said = self.run_blocks(chain, 5e3, 20)
        self.assertEqual(len(said), 1)
        self.assertAlmostEqual(said[0] - 100.0, 15 / 30, delta=1.5 / 30)
        # Clipping on: not again within 30 s, again after.
        self.assertEqual(self.run_blocks(chain, 5e3, 30 * 25, start=101.0), [])
        later = self.run_blocks(chain, 5e3, 30, start=131.0)
        self.assertEqual(len(later), 1)

    def test_only_a_clear_signal_counts(self):
        # (Noise clips whatever the deviation: a squelch tail, an empty
        # channel with the squelch off.)
        chain = _FakeClippingChain()
        self.assertEqual(self.run_blocks(chain, 5e3, 60, snr=10.0), [])
        self.assertEqual(self.run_blocks(chain, 5e3, 60, snr=None), [])
        self.assertEqual(chain.swings_asked, 0)

    def test_no_cost_while_nothing_clips(self):
        chain = _FakeClippingChain(fraction=0.01)
        self.assertEqual(self.run_blocks(chain, 5e3, 90), [])
        self.assertEqual(chain.swings_asked, 0)

    def test_the_texts_name_what_holds_the_swing(self):
        cases = (
            # (deviation heard at, swing in its multiples, text)
            (
                5e3,
                (2.0, 1.9),
                "FM audio is clipping: this station deviates more than 5 kHz. "
                "Choose a larger FM deviation in Demodulation: 12.5 kHz holds it.",
            ),
            (
                5e3,
                (12.0, 8.0),
                "FM audio is clipping: this station deviates more than 5 kHz. "
                "Choose a larger FM deviation in Demodulation, such as 75 kHz "
                "(broadcast) for an FM radio station.",
            ),
            (
                75e3,
                (1.3, 1.1),
                "FM audio is clipping: this station deviates more than 75 kHz, "
                "the largest FM deviation offered.",
            ),
        )
        from sdr_module.gui.audio_chain import ClipWarning

        for deviation, swing, text in cases:
            rule = self.rule = ClipWarning()
            said = self.run_blocks(_FakeClippingChain(swing=swing), deviation, 20)
            self.assertEqual(len(said), 1)
            self.assertEqual(rule.text(deviation), text)
        # Where the deviation is chosen is the window's: the Second
        # Receiver's is in its Tuning group.
        self.assertEqual(
            rule.text(75e3, where="Tuning"),
            "FM audio is clipping: this station deviates more than 75 kHz, "
            "the largest FM deviation offered.",
        )

    @unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
    def test_the_main_window_uses_it(self):
        from sdr_module.gui import audio_chain, main_window

        self.assertEqual(main_window._CLIP_FRACTION, audio_chain.CLIP_FRACTION)
        self.assertEqual(main_window._CLIP_MIN_SNR_DB, audio_chain.CLIP_MIN_SNR_DB)
        self.assertEqual(main_window._CLIP_WARN_AFTER_S, audio_chain.CLIP_WARN_AFTER_S)
        self.assertEqual(main_window._CLIP_WARN_EVERY_S, audio_chain.CLIP_WARN_EVERY_S)


class TestTheSecondReceiversDspMeasuresClipping(unittest.TestCase):
    """R8-01 on the DSP worker: each frame says when the warning is due
    (``Frame.clip_warning``), by the main window's rule, for what this
    receiver demodulates; nothing while nothing clips."""

    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")

    def test_a_station_swinging_10_khz_heard_at_5_khz(self):
        from tests.test_gui_second_receiver import fm_signal

        dsp = _clipping_dsp()
        frames = _frames(dsp, fm_signal(int(1.5 * RATE), deviation=10e3))
        said = [(i, f.clip_warning) for i, f in enumerate(frames) if f.clip_warning]
        self.assertEqual(
            [text for _i, text in said],
            [
                "FM audio is clipping: this station deviates more than 5 kHz. "
                "Choose a larger FM deviation in Tuning: 12.5 kHz holds it."
            ],
        )
        # (After half a second of it, as in the main window.)
        self.assertGreaterEqual(said[0][0], 14)
        self.assertLessEqual(said[0][0], 20)
        # Clipping on: not again within 30 s.
        more = _frames(dsp, fm_signal(int(1.5 * RATE), deviation=10e3))
        self.assertFalse(any(f.clip_warning for f in more))

    def test_a_deviation_that_holds_it_never_warns(self):
        from tests.test_gui_second_receiver import fm_signal

        dsp = _clipping_dsp(deviation=12.5e3)
        frames = _frames(dsp, fm_signal(int(1.5 * RATE), deviation=10e3))
        self.assertFalse(any(f.clip_warning for f in frames))
        self.assertTrue(all(f.audio is not None for f in frames[2:]))

    def test_noise_doesnt_warn(self):
        # (An empty channel with the squelch off clips whatever the
        # deviation.)
        from tests.test_gui_second_receiver import noise_only

        dsp = _clipping_dsp()
        dsp.squelch.enabled = False
        frames = _frames(dsp, noise_only(int(1.5 * RATE)))
        self.assertTrue(all(f.audio is not None for f in frames[2:]))
        self.assertFalse(any(f.clip_warning for f in frames))

    def test_not_demodulated_not_measured(self):
        # (Nobody listens: as in the main window, nothing is demodulated,
        # so nothing is said.)
        from tests.test_gui_second_receiver import fm_signal

        dsp = _clipping_dsp()
        frames = _frames(dsp, fm_signal(int(1.5 * RATE), deviation=10e3), play=False)
        self.assertFalse(any(f.clip_warning for f in frames))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheSecondReceiverSaysItsFmClips(_TypingCase):
    """R8-01 as the walker met it: the band plan off (or narrow FM chosen
    by hand), 97.1 MHz in the Second Receiver at 5 kHz deviation, Listen:
    This receiver. It played clipped at full scale (peak 1.00 on 55% of
    the samples) and nothing said why, while the main window, tuned the
    same way, says "FM audio is clipping"; now its status line says so
    too, naming where its FM deviation is chosen."""

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()
        self.addCleanup(patcher.stop)
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        rx.set_listen("second")
        rx._sync_audio_output()
        self.win._band_plan_action.setChecked(False)
        self.type_frequency("97.1")
        rx.set_mode("FM", "5 kHz", "200 kHz")
        self.said = []
        shown = rx._show_message

        def show(text, *args, **kwargs):
            self.said.append(text)
            shown(text, *args, **kwargs)

        rx._show_message = show

    def test_the_status_line_says_it_clips(self):
        rx = self.rx
        self.feed(45)
        audio = np.concatenate([np.ravel(b) for b in self.speaker.blocks])
        self.assertGreater(float(np.mean(np.abs(audio) >= 0.999)), 0.2)
        text = (
            "FM audio is clipping: this station deviates more than 5 kHz. "
            "Choose a larger FM deviation in Tuning, such as 75 kHz "
            "(broadcast) for an FM radio station."
        )
        self.assertEqual(self.said, [text])
        self.assertEqual(rx.message(), text)
        self.assertEqual(rx._message_label.property("tone"), "warning")
        # At most every so often (it clips on).
        self.feed(45)
        self.assertEqual(self.said, [text])

    def test_at_75_khz_nothing_is_said(self):
        self.rx.set_mode("FM", "75 kHz", "200 kHz")
        self.feed(45)
        self.assertEqual(self.said, [])


# --------------------------------------------------------------------------- #
# R8-02: a lone preset channel keeps what you picked there
# --------------------------------------------------------------------------- #


class _BothWindowsCase(_TypingCase):
    """The same steps in the Second Receiver and in the main window."""

    def type_main(self, text):
        """Type ``text`` into the main window's Frequency box."""
        self.win.activateWindow()
        spin = self.win._frequency_field()
        spin.setFocus()
        self.settle()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, text)
        QTest.keyClick(spin, Qt.Key.Key_Return)

    def main_listening(self):
        panel = self.win._control_panel
        mode = panel.get_demod_mode()
        deviation = panel.get_fm_deviation_text() if mode == "FM" else ""
        return mode, deviation, panel._bw_combo.currentText()

    def windows(self):
        """For each window: (name, type a frequency, tune as a click,
        pick a deviation by hand, how it listens, its status line, clear
        its status line)."""
        rx, win = self.rx, self.win
        panel = win._control_panel
        return (
            (
                "Second Receiver",
                self.type_frequency,
                rx.tune,
                rx._set_deviation,
                self.listening,
                rx.message,
                rx._clear_message,
            ),
            (
                "main window",
                self.type_main,
                win.set_frequency,
                panel._fm_dev_combo.setCurrentText,
                self.main_listening,
                win._message_label.text,
                win._clear_status_message,
            ),
        )

    def start_on_2m(self):
        self.type_main("145.8")
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.type_frequency("145.8")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestALonePresetKeepsWhatYouPickedThere(_BothWindowsCase):
    """R8-02: POCSAG 152.240 MHz is a preset channel outside every band.
    With a deviation picked there by hand, a click 10 kHz away and 152.24
    typed again set the preset's 12.5 kHz again, with its message: outside
    every band the "same zone" test was skipped, so the short hop counted
    as a far tune. Inside a band the same steps keep your choice. Now the
    lone preset's neighbourhood (its channel width either side) is its
    zone, in both windows; a real far tune to it still applies the
    preset's listening."""

    def test_a_short_hop_off_the_channel_and_back_keeps_your_deviation(self):
        self.start_on_2m()
        for name, type_in, click, pick, listening, message, clear in self.windows():
            type_in("152.24")
            self.assertEqual(listening(), ("FM", "12.5 kHz", "25 kHz"), name)
            self.assertTrue(message().startswith("POCSAG: listening"), name)
            pick("5 kHz")
            self.assertEqual(listening(), ("FM", "5 kHz", "25 kHz"), name)
            for hop in (152.25e6, 152.2275e6, 152.265e6):  # (a click away)
                self.assertTrue(click(hop), name)
                clear()
                type_in("152.24")
                self.assertEqual(listening(), ("FM", "5 kHz", "25 kHz"), (name, hop))
                self.assertEqual(message(), "", name)  # (nothing set, nothing said)

    def test_a_far_tune_to_the_channel_still_hears_it_as_the_preset(self):
        self.start_on_2m()
        for name, type_in, _click, pick, listening, message, _clear in self.windows():
            # (From another band, from a frequency outside every band far
            # away, and from beyond the channel's neighbourhood.)
            for start in ("145.8", "300", "152.3"):
                type_in("152.24")
                pick("5 kHz")
                type_in(start)
                type_in("152.24")
                self.assertEqual(
                    listening(), ("FM", "12.5 kHz", "25 kHz"), (name, start)
                )
                self.assertTrue(message().startswith("POCSAG: listening"), name)

    def test_its_message_goes_with_a_far_tune_outside_every_band(self):
        # (As LISTEN-13: the message about where you came from no longer
        # applies. 300 MHz is in no band either, which read as the same
        # place as POCSAG's channel.)
        self.start_on_2m()
        for name, type_in, _click, _pick, _listening, message, _clear in self.windows():
            type_in("152.24")
            self.assertTrue(message().startswith("POCSAG: listening"), name)
            type_in("300")
            self.assertEqual(message(), "", name)

    def test_the_zone_of_a_lone_preset(self):
        from sdr_module.gui.main_window import SDRMainWindow

        zone = SDRMainWindow._band_plan_zone
        pocsag = zone(152.24e6)
        self.assertNotEqual(pocsag, ("", ""))
        for near in (152.215e6, 152.25e6, 152.265e6):
            self.assertEqual(zone(near), pocsag, near)
        for far in (152.2e6, 152.3e6, 300e6):
            self.assertEqual(zone(far), ("", ""), far)
        # Inside a band, as before.
        self.assertEqual(zone(145.8e6), zone(146.52e6))


# --------------------------------------------------------------------------- #
# R8-03: the band plan's message names a deviation as the list does
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheBandPlanNamesTheDeviationAsTheListDoes(_BothWindowsCase):
    """R8-03: typed 152.240 MHz (POCSAG), the band plan's message said "FM
    (narrow, 12.5 kHz deviation)" while the FM deviation list's item reads
    "12.5 kHz (wide voice)". The message now takes its words from the
    list's own table, in both windows: "FM (wide voice, 12.5 kHz
    deviation)"."""

    PLACES = (
        ("97.1", "75 kHz"),  # FM Broadcast
        ("145.29", "5 kHz"),  # 2m Ham
        ("152.24", "12.5 kHz"),  # POCSAG
        ("137.5", "25 kHz"),  # Weather Satellites
        ("151.88", "2.5 kHz"),  # MURS
    )

    def shown_deviations(self):
        """For each window: what its FM deviation list shows."""
        return (
            self.rx._deviation_combo.currentText(),
            self.win._control_panel._fm_dev_combo.current_label(),
        )

    def test_pocsag(self):
        self.start_on_2m()
        for name, type_in, *_rest, message, _clear in self.windows():
            type_in("152.24")
            self.assertTrue(
                message().startswith(
                    "POCSAG: listening in FM (wide voice, 12.5 kHz deviation) "
                    "with a 25 kHz filter, as this channel is usually heard"
                ),
                (name, message()),
            )
        self.assertEqual(self.shown_deviations(), ("12.5 kHz (wide voice)",) * 2)

    def test_every_deviation_in_the_lists_words(self):
        windows = self.windows()
        for text, value in self.PLACES:
            # (From a place heard otherwise: the band plan says how.)
            self.type_main("118.3")
            self.type_frequency("118.3")
            for _name, type_in, *_rest in windows:
                type_in(text)
            for (name, *_rest, message, _clear), shown in zip(
                windows, self.shown_deviations(), strict=True
            ):
                self.assertTrue(shown.startswith(f"{value} ("), (name, shown))
                words = shown[len(value) + 2 : -1]
                self.assertIn(
                    f"listening in FM ({words}, {value} deviation) with a ",
                    message(),
                    (name, text),
                )


# --------------------------------------------------------------------------- #
# R8-04: the demo receivers hear one sky
# --------------------------------------------------------------------------- #


class TestTheDemoAirlinerFliesOneSky(unittest.TestCase):
    """R8-04: each demo device flew the demo airliner on a clock of its own
    (from its first read), so the main window and the Second Receiver,
    started a few seconds apart on 1090 MHz, heard different position
    frames: the Aircraft view's Msgs read between one receiver's count and
    the sum (54 for 36 + 36, the Second Receiver started 3 s later). The
    app's demo receivers now share one sky: they hear the same replies,
    bit for bit, at the same moment. A device on its own, or one not of
    the app's receivers, flies the airliner as before."""

    RATE = 2.4e6
    BLOCK = 80_000  # (a display frame at 2.4 MS/s)

    def setUp(self):
        import weakref

        from sdr_module.gui import device_dialog

        self.dd = device_dialog
        for name, value in (
            ("_MAX_LAG_S", 1e9),  # (every sample is read, however slowly)
            ("_SKY_DEVICES", weakref.WeakSet()),  # (only this test's devices)
        ):
            patcher = mock.patch.object(device_dialog, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def demo(self, shared=True, seed=7):
        dev = self.dd.MockDevice()
        dev.shared_sky = shared  # (as the app's windows make theirs)
        dev.set_sample_rate(self.RATE)
        dev.set_frequency(1090e6)
        dev._rng = np.random.default_rng(seed)
        dev.start_rx()
        self.addCleanup(dev.close)
        return dev

    def started_earlier(self, dev, seconds):
        """``dev`` has streamed for ``seconds`` already (keeping up)."""
        dev.read_samples(self.BLOCK)
        dev._stream_pos += int(seconds * self.RATE)
        dev._stream_epoch -= seconds

    def decode(self, devices, seconds):
        """Each device's ADS-B messages over the same ``seconds``, read a
        display frame at a time, in turn: (time s, raw frame) lists."""
        from sdr_module.dsp.live_decoders import create_live_decoder

        decoders = [create_live_decoder("ADS-B", self.RATE) for _ in devices]
        heard = [[] for _ in devices]
        for step in range(int(seconds * self.RATE / self.BLOCK)):
            for dev, decoder, out in zip(devices, decoders, heard, strict=True):
                for msg in decoder.process(dev.read_samples(self.BLOCK)):
                    out.append((step * self.BLOCK / self.RATE, bytes(msg.raw_bits)))
        return heard

    def test_two_demo_receivers_hear_the_same_replies(self):
        from sdr_module.gui.decoder_views import AircraftTracker

        main = self.demo(seed=1)
        self.started_earlier(main, 3.0)  # (the Second Receiver comes later)
        second = self.demo(seed=2)
        ours, theirs = self.decode((main, second), 6.0)
        self.assertGreater(len(ours), 15)
        self.assertLessEqual(abs(len(ours) - len(theirs)), 1)  # (the ends)
        # Each reply the main receiver heard (away from the ends), the
        # Second Receiver heard too, bit for bit, at the same moment (the
        # frame it was read in, or the next).
        for at, frame in ours:
            if 0.2 < at < 5.8:
                self.assertTrue(
                    any(f == frame and abs(t - at) < 0.1 for t, f in theirs),
                    (at, frame.hex()),
                )
        # The Aircraft view counts each reply once.
        tracker = AircraftTracker(clock=lambda: 0.0)
        both = [(at, "", f) for at, f in ours] + [(at, "2nd RX", f) for at, f in theirs]
        for at, receiver, frame in sorted(both):  # (as they come)
            msg = SimpleNamespace(icao_address="4840D6", raw_bits=frame)
            tracker.update(msg, at=at, receiver=receiver)
        self.assertLessEqual(
            tracker.get("4840D6").messages, max(len(ours), len(theirs))
        )

    def test_a_device_on_its_own_flies_from_its_start(self):
        # Exactly the samples it gave before (its first half second: the
        # airliner's ident at 0.10 s and a position at 0.35 s): no other
        # demo receiver streams, or it isn't one of the app's, or the
        # other has stopped.
        n = int(0.5 * self.RATE)
        alone = self.demo(seed=3)
        samples = alone.read_samples(n)
        alone.close()
        live = self.demo(seed=4)
        self.started_earlier(live, 3.0)
        other = self.demo(shared=False, seed=3)
        np.testing.assert_array_equal(other.read_samples(n), samples)
        live.close()
        later = self.demo(seed=3)
        np.testing.assert_array_equal(later.read_samples(n), samples)


# --------------------------------------------------------------------------- #
# R8-05: the Second Receiver's Morse speed and Receiving in the Text view
# --------------------------------------------------------------------------- #

if HAS_PYQT6:
    from tests.test_gui_decoder_views import _settle
    from tests.test_gui_receiver_views import _Case as _ViewCase
    from tests.test_gui_receiver_views import _morse
    from tests.test_gui_second_receiver_decoders import _DecodeCase
else:  # pragma: no cover - environment
    _ViewCase = _DecodeCase = unittest.TestCase


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheSecondReceiversMorseSpeedInTheTextView(_ViewCase):
    """R8-05: above the Text view's "2nd RX" entry, SPEED was its last
    finished line's speed and "Receiving" never showed, while Live shows
    the main receiver's decoder's speed and state as it receives. The
    entry now shows the Second Receiver's own, from its decoder's status
    (made on its DSP worker), as Live does."""

    RX2 = "2nd RX"

    def setUp(self):
        super().setUp()
        self.view = self.panel._cw_view

    def shown(self):
        return self.view.speed.text(), self.view.state.text()

    def show_rx2(self):
        from sdr_module.gui.decoder_views import receiver_source

        self.view.source_bar().choose(receiver_source(self.RX2))
        _settle(1)

    def test_its_entry_shows_its_speed_and_receiving_as_it_receives(self):
        p = self.panel
        self.add(_morse("VVV DE SDR", wpm=15.0), receiver=self.RX2)
        self.show_rx2()
        self.assertEqual(self.shown(), ("15 WPM", ""))  # (its last line's)
        p.set_receiver_live(self.RX2, "VVV D", "18 WPM", True)
        self.assertEqual(self.view.shown_text(), "VVV DE SDR\nVVV D")
        self.assertEqual(self.shown(), ("18 WPM", "Receiving"))
        p.set_receiver_live(self.RX2, "VVV DE", "18 WPM", False)  # (key up)
        self.assertEqual(self.shown(), ("18 WPM", ""))
        p.set_receiver_live(self.RX2, "", "", False)  # (it stopped)
        self.assertEqual(self.shown(), ("15 WPM", ""))

    def test_live_keeps_this_windows_own(self):
        p, view = self.panel, self.view
        self.add(_morse("CQ DE MAIN"))
        view.set_speed("20 WPM", True)
        p.set_receiver_live(self.RX2, "VVV", "18 WPM", True)
        self.assertEqual(view.shown_source(), "")  # (Live stays)
        self.assertEqual(self.shown(), ("20 WPM", "Receiving"))
        self.show_rx2()
        self.assertEqual(self.shown(), ("18 WPM", "Receiving"))
        view.set_speed("21 WPM", False)  # (this window's, meanwhile)
        self.assertEqual(self.shown(), ("18 WPM", "Receiving"))
        view.source_bar().choose("")
        _settle(1)
        self.assertEqual(self.shown(), ("21 WPM", ""))

    def test_clear_forgets_it(self):
        from sdr_module.gui.decoder_views import _NO_VALUE

        p = self.panel
        p.set_receiver_live(self.RX2, "VVV", "18 WPM", True)  # (its entry, shown)
        self.assertEqual(self.shown(), ("18 WPM", "Receiving"))
        p.clear()
        _settle(1)
        self.assertEqual(self.shown(), (_NO_VALUE, ""))
        p.set_receiver_live(self.RX2, "", "18 WPM", True)  # (nothing to show yet)
        self.assertEqual(self.view.other_receivers(), [])
        self.assertEqual(self.shown(), (_NO_VALUE, ""))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheSecondReceiversMorseSpeedOnTheDemoBeacon(_DecodeCase):
    """R8-05 in the window: the Second Receiver on the demo's beacon
    (99.820 MHz, CW decoder), the Decoder tab's Text view on "2nd RX"."""

    def test_speed_and_receiving_follow_its_decoder(self):
        from sdr_module.gui.decoder_views import receiver_source

        panel, rx = self.panel, self.rx
        view = panel._cw_view
        panel._proto_combo.setCurrentText("POCSAG")  # (the main window's)
        self.settle()
        self.start_on(99.82e6, ("CW",))
        self.choose("CW (Morse)")
        self.run_seconds(4.0)
        self.assertEqual(view.shown_source(), receiver_source("2nd RX"))
        block = int(rx._rate() / 30)
        seen = set()
        for _ in range(45):  # (1.5 s more, frame by frame)
            rx._process(rx.device.read_samples(block))
            status = rx._decoder_status
            self.assertTrue(status.detail.endswith("WPM"), status.detail)
            self.assertEqual(view.speed.text(), status.detail)
            receiving = "Receiving" if status.signal_present else ""
            self.assertEqual(view.state.text(), receiving)
            seen.add(view.state.text())
        self.assertIn("Receiving", seen)
        # Decoding paused, or another protocol: nothing is received.
        rx._decode_check.setChecked(False)
        rx._process(rx.device.read_samples(block))
        self.assertEqual(view.state.text(), "")
        rx._decode_check.setChecked(True)
        for _ in range(15):
            rx._process(rx.device.read_samples(block))
        self.assertEqual(view.speed.text(), rx._decoder_status.detail)
        self.choose("POCSAG")
        rx._process(rx.device.read_samples(block))
        self.assertEqual(view.state.text(), "")
        self.choose("CW (Morse)")
        for _ in range(15):
            rx._process(rx.device.read_samples(block))
        # Stopped, it receives nothing.
        rx.stop()
        self.assertEqual(view.state.text(), "")


if __name__ == "__main__":
    unittest.main()
