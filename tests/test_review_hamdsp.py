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

import numpy as np

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


class TestSstvOnHf(_HamWindowCase):
    """HAMDSP-3: only FM audio reached the SSTV decoder, so HF SSTV (sent
    in USB or LSB) could never be decoded, and the panel said SSTV is FM
    audio and offered Switch to FM, which makes an HF image undecodable."""

    def tone_heard(self, freq_hz: float, mode: str, audio_hz: float) -> float:
        """Feed a steady SSB tone through the window; the decoder's tone."""
        win = self.win
        self.assertTrue(win.set_frequency(freq_hz))
        win._set_demod_mode(mode)
        self.sstv._start_btn.click()
        rate = win._device_sample_rate()
        offset = win._vfo_offset_hz() + (audio_hz if mode == "USB" else -audio_hz)
        n = int(rate / 30)
        phase = 0.0
        step = 2 * np.pi * offset / rate
        for _ in range(20):
            iq = (0.05 * np.exp(1j * (phase + step * np.arange(n)))).astype(
                np.complex64
            )
            phase += step * n
            win._process_block(iq)
        tone = self.sstv._decoder.get_tone()["tone_hz"]
        self.assertIsNotNone(tone, f"no tone reached the decoder in {mode}")
        return tone

    def test_usb_audio_reaches_the_decoder(self):
        self.assertAlmostEqual(self.tone_heard(14.230e6, "USB", 1900.0), 1900, delta=30)
        self.assertTrue(self.sstv._fm_callout.isHidden())

    def test_lsb_audio_reaches_the_decoder(self):
        self.assertAlmostEqual(self.tone_heard(7.171e6, "LSB", 1900.0), 1900, delta=30)
        self.assertTrue(self.sstv._fm_callout.isHidden())

    def test_start_in_usb_asks_for_nothing(self):
        win = self.win
        self.assertTrue(win.set_frequency(14.230e6))
        win._set_demod_mode("USB")
        self.sstv._start_btn.click()
        settle()
        self.assertTrue(self.sstv._fm_callout.isHidden())
        self.assertNotIn("needs FM", win._message_label.text())

    def switch_offered(self, freq_hz: float) -> str:
        win = self.win
        self.assertTrue(win.set_frequency(freq_hz))
        win._set_demod_mode("AM")
        self.sstv._start_btn.click()
        settle()
        self.assertFalse(self.sstv._fm_callout.isHidden())
        return self.sstv._switch_fm_btn.text()

    def test_am_on_20m_offers_usb(self):
        self.assertEqual(self.switch_offered(14.230e6), "Switch to USB")
        self.assertIn("USB", self.sstv._fm_label.text())
        self.assertIn("USB", self.win._message_label.text())
        self.sstv._switch_fm_btn.click()
        settle()
        self.assertEqual(self.win._control_panel.get_demod_mode(), "USB")
        self.assertTrue(self.sstv._fm_callout.isHidden())

    def test_am_on_40m_offers_lsb(self):
        self.assertEqual(self.switch_offered(7.171e6), "Switch to LSB")
        self.sstv._switch_fm_btn.click()
        self.assertEqual(self.win._control_panel.get_demod_mode(), "LSB")

    def test_am_on_2m_still_offers_narrow_fm(self):
        self.assertEqual(self.switch_offered(145.8e6), "Switch to FM")
        self.assertIn("FM", self.win._message_label.text())
        self.sstv._switch_fm_btn.click()
        panel = self.win._control_panel
        self.assertEqual(panel.get_demod_mode(), "FM")
        self.assertEqual(panel.get_fm_deviation_text(), "5 kHz")

    def test_cw_brings_the_hint_back(self):
        win = self.win
        self.assertTrue(win.set_frequency(14.230e6))
        win._set_demod_mode("USB")
        self.sstv._start_btn.click()
        win._set_demod_mode("CW")
        settle()
        self.assertFalse(self.sstv._fm_callout.isHidden())
        win._set_demod_mode("LSB")
        settle()
        self.assertTrue(self.sstv._fm_callout.isHidden())


class _SeekCase(unittest.TestCase):
    """The tuner alone, its seek driven by hand (a fake receiver)."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.ham.gui.radio_tuner import RadioTunerWidget

        self.tuner = RadioTunerWidget(None, 2.4e6)
        self.emitted = []
        self.tuner.frequency_changed.connect(lambda f, b: self.emitted.append(f))
        self.stations = set()
        self.tuner.set_signal_probe(
            lambda: (
                "signal"
                if round(self.tuner.get_frequency()) in self.stations
                else "quiet"
            )
        )
        self.tuner.show()

    def tearDown(self):
        if hasattr(self, "tuner"):
            self.tuner.close()
            self.tuner.deleteLater()
            settle()

    def seek(self, direction: int, polls: int = 400) -> None:
        tuner = self.tuner
        (tuner._seek_up if direction > 0 else tuner._seek_down)()
        for _ in range(polls):
            if not tuner.is_seeking():
                return
            tuner._seek_timer.stop()
            tuner._seek_poll()
        self.fail("the seek never ended")


class TestSeekWithinReach(_SeekCase):
    """HAMDSP-4: seek stepped over the whole band. On a HackRF (from 1 MHz)
    in AM it kept asking for 990 kHz (refused) until it had counted all
    118 channels, then said no station was anywhere in the band; seek up
    wrapped to 530 kHz the same way. A recording's span did the same."""

    def am_from(self, freq_hz: float) -> None:
        self.tuner.set_device_range(1e6, 6e9, "HackRF One")
        self.tuner._am_btn.click()
        self.tuner._tuning_dial.tune_to(freq_hz)
        self.emitted.clear()

    def test_seek_down_wraps_inside_the_device_range(self):
        self.am_from(1010e3)
        self.stations = {1260000}
        self.seek(-1)
        self.assertTrue(self.emitted)
        self.assertGreaterEqual(min(self.emitted), 1e6, "below the HackRF's range")
        self.assertEqual(self.emitted[:2], [1000e3, 1700e3])
        self.assertAlmostEqual(self.tuner.get_frequency(), 1260e3)
        self.assertEqual(self.tuner._hint_label.text(), "Found a station on 1260 kHz.")

    def test_seek_up_wraps_inside_the_device_range(self):
        self.am_from(1260e3)
        self.stations = {1010000}
        self.seek(1)
        self.assertGreaterEqual(min(self.emitted), 1e6)
        self.assertAlmostEqual(self.tuner.get_frequency(), 1010e3)

    def test_round_the_part_of_the_band_it_can_tune_once(self):
        self.am_from(1010e3)
        self.seek(-1)
        reachable = int(round((1700e3 - 1000e3) / 10e3)) + 1
        self.assertEqual(len(self.emitted), reachable)
        self.assertEqual(len(set(self.emitted)), reachable)
        self.assertEqual(
            self.tuner._hint_label.text(),
            "No station strong enough to open the squelch from 1000 kHz to "
            "1700 kHz, the part of the band the receiver can tune.",
        )

    def test_one_channel_it_can_tune_is_named_once(self):
        """A recording narrow enough to hold one channel: the seek said
        "from 1000 kHz to 1000 kHz"."""
        self.tuner._am_btn.click()
        self.tuner._tuning_dial.tune_to(1000e3)
        self.tuner.set_tune_check(lambda f: abs(f - 1000e3) < 1.0)
        self.seek(-1)
        self.assertEqual(
            self.tuner._hint_label.text(),
            "No station strong enough to open the squelch on 1000 kHz, the only "
            "channel of the band the receiver can tune.",
        )

    def test_the_whole_band_when_it_can_tune_it(self):
        self.tuner._am_btn.click()
        self.emitted.clear()
        self.seek(1)
        self.assertEqual(len(self.emitted), 118)
        self.assertEqual(
            self.tuner._hint_label.text(),
            "No station strong enough to open the squelch anywhere in the band.",
        )

    def test_seek_stays_inside_a_recording(self):
        self.tuner._am_btn.click()
        self.tuner._tuning_dial.tune_to(960e3)
        self.emitted.clear()
        self.tuner.set_tune_check(lambda f: 900e3 <= f <= 1100e3)
        self.stations = {1100000}
        self.seek(-1)
        self.assertTrue(all(900e3 <= f <= 1100e3 for f in self.emitted))
        self.assertAlmostEqual(self.tuner.get_frequency(), 1100e3)

    def test_nothing_it_can_tune(self):
        self.tuner.set_tune_check(lambda f: False)
        self.seek(1)
        self.assertEqual(self.emitted, [])
        self.assertIn("can't tune any", self.tuner._hint_label.text())


class TestSeekInTheWindow(_HamWindowCase):
    """HAMDSP-4 through the window: the device's range and a recording's
    span reach the seek, and a step the receiver refuses stops it."""

    def setUp(self):
        super().setUp()
        self.win._show_radio_tuner()
        self.tuner = self.win._radio_tuner
        self.tuner._am_btn.click()

    def tearDown(self):
        tuner = getattr(self, "tuner", None)
        if tuner is not None:
            tuner.close()
        super().tearDown()

    def spy_tuning(self):
        tunes = []
        tune = self.win.set_frequency

        def spy(freq):
            ok = tune(freq)
            tunes.append((freq, ok))
            return ok

        self.win.set_frequency = spy
        return tunes

    def test_seek_on_a_hackrf_finds_the_demo_station(self):
        from types import SimpleNamespace

        win, tuner = self.win, self.tuner
        win._device.spec = SimpleNamespace(freq_min=1e6, freq_max=6e9)
        win._apply_tuning_range()
        win._sync_tuner_device()
        tuner._tuning_dial.tune_to(1010e3)
        tunes = self.spy_tuning()
        device = win._device
        block = int(device.sample_rate / 30)
        tuner._seek_down()
        for _ in range(300):
            if not tuner.is_seeking():
                break
            tuner._seek_timer.stop()
            for _ in range(4):
                win._process_block(device.read_samples(block))
            tuner._seek_poll()
        self.assertFalse(tuner.is_seeking())
        self.assertTrue(all(ok for _f, ok in tunes), "a step was refused")
        self.assertGreaterEqual(min(f for f, _ok in tunes), 1e6)
        self.assertAlmostEqual(win._current_frequency(), 1260e3)
        self.assertEqual(tuner._hint_label.text(), "Found a station on 1260 kHz.")

    def test_seek_inside_a_recording_is_never_refused(self):
        win, tuner = self.win, self.tuner
        tuner._tuning_dial.tune_to(1000e3)
        win._lock_lo(1.0e6, 240e3, "the recording")
        tunes = self.spy_tuning()
        tuner.set_signal_probe(lambda: "quiet")
        tuner._seek_down()
        for _ in range(300):
            if not tuner.is_seeking():
                break
            tuner._seek_timer.stop()
            tuner._seek_poll()
        self.assertFalse(tuner.is_seeking())
        self.assertTrue(tunes)
        self.assertTrue(all(ok for _f, ok in tunes), "a step was refused")
        self.assertIn("the part of the band", tuner._hint_label.text())

    def test_a_refused_step_stops_the_seek(self):
        win, tuner = self.win, self.tuner
        tuner._tuning_dial.tune_to(1010e3)
        win._tx_busy = lambda: True  # every tune is refused now
        tuner._seek_down()
        tuner._seek_timer.stop()
        tuner._seek_poll()
        self.assertFalse(tuner.is_seeking())
        self.assertEqual(
            tuner._hint_label.text(), "Seek stopped: the receiver can't tune 1000 kHz."
        )
        self.assertIn("Can't tune while transmitting", win._message_label.text())


class _QrpCase(unittest.TestCase):
    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.core import frequency_manager
        from sdr_module.ham.gui.qrp_panel import QRPPanel

        self.fm = frequency_manager.get_frequency_manager()
        saved = self.fm.get_license_class()
        self.addCleanup(lambda: self.fm.set_license_class(saved))
        self.panel = QRPPanel()
        self.panel.show()
        settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.panel.deleteLater()
            settle()

    def license(self, name: str) -> None:
        from sdr_module.core.frequency_manager import LicenseClass

        self.fm.set_license_class(LicenseClass(name))
        self.panel.refresh_license()


# The five US 60 m channels (USB phone, CW and data only).
CHANNELS_60M = (5.3305e6, 5.3465e6, 5.357e6, 5.3715e6, 5.4035e6)


class TestQrpSidebandOn60m(_QrpCase):
    """HAMDSP-5: in Auto the card checked USB as "SSB", which the 60 m
    channels don't list (USB only): "Not permitted ... in SSB" in red, and
    the channel's 100 W limit never showed."""

    def test_usb_is_permitted_on_every_60m_channel(self):
        p = self.panel
        self.license("general")
        for freq in CHANNELS_60M:
            p.set_station(freq, "USB")
            self.assertEqual(p.mode(), "SSB", "the class is still judged as phone")
            self.assertEqual(
                p._legal_label.text(),
                f"At {freq / 1e6:.3f} MHz (General, SSB): legal limit 100 W PEP.",
            )
            self.assertEqual(p._legal_label.property("tone"), "success")
            self.assertEqual(p._legal_limit, 100.0)

    def test_lsb_is_still_refused_there(self):
        p = self.panel
        self.license("general")
        p.set_station(5.3305e6, "LSB")
        self.assertTrue(p._legal_label.text().startswith("Not permitted"))
        self.assertIn("LSB isn't allowed", p._legal_label.text())
        self.assertEqual(p._legal_label.property("tone"), "danger")

    def test_a_manual_ssb_choice_counts_either_sideband(self):
        p = self.panel
        self.license("general")
        p._mode_combo.setCurrentText("SSB")
        p.set_station(5.3305e6, "AM")  # (receiving in AM, sending SSB)
        self.assertIn("legal limit 100 W PEP", p._legal_label.text())
        self.assertEqual(p._legal_label.property("tone"), "success")
        p.set_station(14.25e6, "AM")  # 20 m General phone: both sidebands
        self.assertIn("legal limit", p._legal_label.text())
        self.assertEqual(p._legal_label.property("tone"), "success")

    def test_ssb_elsewhere_is_unchanged(self):
        p = self.panel
        self.license("technician")
        p.set_station(28.4e6, "USB")
        self.assertEqual(
            p._legal_label.text(),
            "At 28.400 MHz (Technician, SSB): legal limit 200 W PEP.",
        )


class TestQrpSidebandInTheWindow(_HamWindowCase):
    def test_qrp_card_on_60m_in_usb(self):
        from sdr_module.core import frequency_manager
        from sdr_module.core.frequency_manager import LicenseClass

        fm = frequency_manager.get_frequency_manager()
        saved = fm.get_license_class()
        self.addCleanup(lambda: fm.set_license_class(saved))
        fm.set_license_class(LicenseClass.GENERAL)
        qrp = self.win._qrp_panel
        qrp.refresh_license()
        self.assertTrue(self.win.set_frequency(5.3305e6))
        self.win._set_demod_mode("USB")
        self.assertEqual(
            qrp._legal_label.text(),
            "At 5.330 MHz (General, SSB): legal limit 100 W PEP.",
        )


# Saved ham/qrp values of the wrong shape (a hand edit, another version).
CORRUPT_QRP = (
    '{"qsos": 5}',
    '{"qsos": true}',
    '{"qsos": 2.5}',
    '{"qsos": [5, "x", null, [1, 2], {"km": Infinity, "watts": 1}, '
    '{"km": 10, "watts": 1}]}',
    '{"chain": {"stages": [[true, 1e12, 10, 0.5]]}}',
    '{"chain": {"stages": [[true, 10, 3000000000, 0.5]]}}',
    '{"chain": {"sdr_dbm": NaN, "stages": [[true, Infinity, -1e30, NaN]]}}',
)


class TestCorruptQrpSettings(unittest.TestCase):
    """HAMDSP-6: some wrongly typed ham/qrp values raised TypeError or
    OverflowError out of SDRMainWindow.__init__: the app would not start
    again until the value was deleted by hand."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)

    def build(self, raw: str):
        from sdr_module.gui import bookmarks_panel, main_window
        from tests.test_gui_chrome import _FakeSettings

        if not main_window.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        store = _FakeSettings()
        store.values["ham/qrp"] = raw
        saved = (main_window.GuiSettings, bookmarks_panel.GuiSettings)
        main_window.GuiSettings = lambda: store
        bookmarks_panel.GuiSettings = lambda: store
        try:
            win = main_window.SDRMainWindow(demo_mode=False)
        finally:
            main_window.GuiSettings, bookmarks_panel.GuiSettings = saved
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        win._confirm_discard_recording = lambda *_args: True
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.close)
        return win

    def test_the_window_starts_with_each_corrupt_value(self):
        for raw in CORRUPT_QRP:
            with self.subTest(raw=raw):
                win = self.build(raw)
                qrp = win._qrp_panel
                for stage in qrp._amp_calc._stages:
                    self.assertGreaterEqual(stage.gain.value(), stage.gain.minimum())
                    self.assertLessEqual(stage.gain.value(), stage.gain.maximum())
                    self.assertTrue(0.05 <= stage.efficiency <= 1.0)
                self.assertTrue(np.isfinite(qrp._amp_calc.get_output_dbm()))

    def test_good_entries_survive_among_bad_ones(self):
        win = self.build(CORRUPT_QRP[3])
        self.assertEqual(
            [(q["km"], q["watts"]) for q in win._qrp_panel._qsos], [(10.0, 1.0)]
        )

    def test_a_huge_gain_is_clamped_not_dropped(self):
        win = self.build(CORRUPT_QRP[4])
        stage = win._qrp_panel._amp_calc._stages[0]
        self.assertTrue(stage.check.isChecked())
        self.assertEqual(stage.gain.value(), stage.gain.maximum())
        self.assertEqual(stage.max_out.value(), 10)

    def test_anything_else_a_restore_raises_is_logged_not_fatal(self):
        from unittest import mock

        from sdr_module.ham.gui.qrp_panel import QRPPanel

        with mock.patch.object(QRPPanel, "set_state", side_effect=RuntimeError("x")):
            with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
                self.build('{"mode": "CW"}')
        self.assertTrue(any("QRP settings" in line for line in logs.output))


def _clip_with_bursts(fs: float = 8000.0, n: int = 8000):
    """Noise with a 1 kHz burst at 800-1600 and another in the last 600."""
    rng = np.random.default_rng(3)
    t = np.arange(n) / fs
    clip = 0.01 * rng.standard_normal(n)
    clip[800:1600] += 0.5 * np.sin(2 * np.pi * 1000 * t[800:1600])
    clip[n - 600 :] += 0.5 * np.sin(2 * np.pi * 1000 * t[n - 600 :])
    return clip


def _onset(y: np.ndarray, threshold: float = 0.1) -> int:
    return int(np.flatnonzero(np.abs(y) > threshold)[0])


class TestNoiseReductionOffline(unittest.TestCase):
    """HAMDSP-7: the spectral methods stream with a delay of
    latency_samples, and nothing returned the last latency_samples of the
    signal: a whole clip in one process() call came out shifted and with
    its end missing (a burst in the last 600 samples was lost)."""

    METHODS = ("SPECTRAL_SUBTRACTION", "WIENER")

    def make(self, method: str, **kw):
        from sdr_module.dsp.filters import (
            NoiseReduction,
            NoiseReductionConfig,
            NoiseReductionMethod,
        )

        config = NoiseReductionConfig(
            method=getattr(NoiseReductionMethod, method), fft_size=1024, **kw
        )
        return NoiseReduction(8000.0, config)

    def test_process_clip_lines_up_and_keeps_the_end(self):
        clip = _clip_with_bursts()
        for method in self.METHODS:
            with self.subTest(method=method):
                y = self.make(method).process_clip(clip)
                self.assertEqual(len(y), len(clip))
                self.assertLessEqual(abs(_onset(y) - _onset(clip)), 4)
                kept = np.sum(y[-600:] ** 2) / np.sum(clip[-600:] ** 2)
                self.assertGreater(kept, 0.8)

    def test_flush_returns_the_rest_of_the_stream(self):
        """With every gain at 1 the stream is the input, delayed: the
        blocks and then flush() give back all of it."""
        clip = _clip_with_bursts()
        nr = self.make("SPECTRAL_SUBTRACTION", subtraction_factor=0.0)
        delay = nr.latency_samples
        parts = [nr.process(clip[i : i + 333]) for i in range(0, len(clip), 333)]
        tail = nr.flush()
        self.assertEqual(len(tail), delay)
        y = np.concatenate(parts + [tail])
        np.testing.assert_allclose(y[delay:], clip, atol=1e-9)
        self.assertEqual(len(nr.flush()), 0, "nothing more is held")

    def test_blocks_then_flush_equal_one_clip(self):
        clip = _clip_with_bursts()
        for method in self.METHODS:
            with self.subTest(method=method):
                nr = self.make(method)
                parts = [nr.process(clip[i : i + 500]) for i in range(0, 8000, 500)]
                y = np.concatenate(parts + [nr.flush()])[nr.latency_samples :]
                np.testing.assert_allclose(
                    y, self.make(method).process_clip(clip), atol=1e-12
                )

    def test_flushing_does_not_count_as_quiet(self):
        """The zeros that push the tail out don't lower the noise estimate."""
        nr = self.make("SPECTRAL_SUBTRACTION")
        nr.process(_clip_with_bursts())
        before = nr.get_noise_spectrum()
        nr.flush()
        np.testing.assert_allclose(nr.get_noise_spectrum(), before)

    def test_complex_input_and_other_methods(self):
        nr = self.make("WIENER")
        iq = np.exp(2j * np.pi * 0.05 * np.arange(3000))
        y = nr.process_clip(iq)
        self.assertTrue(np.iscomplexobj(y))
        self.assertEqual(len(y), len(iq))
        median = self.make("MEDIAN")
        self.assertEqual(len(median.flush()), 0)
        self.assertEqual(len(median.process_clip(np.ones(100))), 100)

    def test_an_empty_clip_gives_an_empty_result_quietly(self):
        import warnings

        for method in self.METHODS + ("MEDIAN", "GATE"):
            with self.subTest(method=method), warnings.catch_warnings():
                warnings.simplefilter("error")  # ("Mean of empty slice")
                self.assertEqual(len(self.make(method).process_clip(np.zeros(0))), 0)
                iq = self.make(method).process_clip(np.zeros(0, np.complex64))
                self.assertTrue(np.iscomplexobj(iq))

    def test_a_stream_too_short_to_learn_from_leaves_no_noise_estimate(self):
        """Flushing a stream shorter than one hop learned the padding zeros
        as the noise: the next recording was cleaned less for a second or
        two (Wiener let twice the noise through). Nothing was heard before
        the zeros, so there is still no estimate, as in a fresh one."""
        rng = np.random.default_rng(9)
        clip = _clip_with_bursts() + 0.05 * rng.standard_normal(8000)
        for method in self.METHODS:
            with self.subTest(method=method):
                fresh = self.make(method)
                used = self.make(method)
                used.process(0.05 * rng.standard_normal(200))
                self.assertEqual(len(used.flush()), used.latency_samples)
                self.assertIsNone(used.get_noise_spectrum())
                self.assertFalse(used.noise_estimated)
                np.testing.assert_allclose(
                    used.process_clip(clip), fresh.process_clip(clip), atol=1e-12
                )


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float).ravel() - a.mean()
    b = b.astype(float).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))


class TestSstvInputThatEnds(unittest.TestCase):
    """HAMDSP-8: the last line waits for a few ms of audio after the
    transmission, so a recording that ends with it (the encoder's own
    output, a trimmed WAV) never completed: is_complete() stayed False
    (Martin 1 at 255/256 lines) and there was no way to say the input had
    ended."""

    def setUp(self):
        from sdr_module.ham.sstv import SSTVDecoder

        self.decoder = SSTVDecoder(FS)
        self.complete = []
        self.lost = []
        self.decoder.set_on_image_complete(lambda image: self.complete.append(image))
        self.decoder.set_on_image_lost(lambda image, rows: self.lost.append(rows))

    def audio(self, mode: str, **kw):
        from sdr_module.ham.sstv import find_mode
        from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card

        spec = find_mode(mode)
        card = sstv_test_card(spec.width, spec.height)
        lead = np.zeros(int(0.3 * FS), np.float32)
        return card, spec, np.concatenate([lead, encode_sstv(card, spec, FS, **kw)])

    def test_finish_completes_an_image_that_ends_the_input(self):
        for mode in ("Martin 1", "Scottie 1", "Robot 36", "PD 120"):
            with self.subTest(mode=mode):
                self.setUp()
                card, spec, audio = self.audio(mode)
                self.decoder.process_audio(audio)
                self.assertFalse(self.decoder.is_complete())
                self.assertTrue(self.decoder.finish())
                self.assertTrue(self.decoder.is_complete())
                self.assertFalse(self.decoder.is_receiving())
                self.assertEqual(self.decoder.state.current_line, spec.height)
                self.assertEqual(len(self.complete), 1)
                self.assertGreater(_correlation(self.complete[0], card), 0.98)

    def test_a_last_line_cut_short_still_completes(self):
        card, spec, audio = self.audio("Martin 1")
        cut = int(0.1 * spec.scan_time_ms / 1000.0 * FS)  # a tenth of a line
        self.decoder.process_audio(audio[:-cut])
        self.assertTrue(self.decoder.finish())
        self.assertGreater(_correlation(self.complete[0][:-1], card[:-1]), 0.98)

    def test_an_input_that_ends_mid_image_keeps_the_lines_received(self):
        card, spec, audio = self.audio("Robot 36")
        self.decoder.process_audio(audio[: int(len(audio) * 0.6)])
        self.assertFalse(self.decoder.finish())
        self.assertFalse(self.decoder.is_complete())
        self.assertFalse(self.decoder.is_receiving())
        self.assertEqual(self.decoder.lost_reason, "the audio ended")
        self.assertEqual(len(self.lost), 1)
        self.assertGreater(self.lost[0], 0.5 * spec.height)

    def test_finish_while_idle_does_nothing(self):
        self.decoder.process_audio(np.zeros(int(FS), np.float32))
        self.assertFalse(self.decoder.finish())
        self.assertEqual((self.complete, self.lost), ([], []))

    def test_an_image_that_completed_before_finish_stays_complete(self):
        """With audio after the transmission the image completes on its
        own: finish() says so (True) and reports nothing twice."""
        card, spec, audio = self.audio("Robot 36")
        self.decoder.process_audio(np.concatenate([audio, np.zeros(4800, np.float32)]))
        self.assertTrue(self.decoder.is_complete())
        self.assertTrue(self.decoder.finish())
        self.assertTrue(self.decoder.finish())
        self.assertEqual((len(self.complete), self.lost), (1, []))

    def test_finish_while_start_now_still_searches(self):
        from sdr_module.ham.sstv import find_mode

        self.decoder.process_audio(np.zeros(int(FS), np.float32))
        self.assertTrue(self.decoder.start_now(find_mode("Martin 1")))
        self.assertFalse(self.decoder.finish())
        self.assertFalse(self.decoder.is_receiving())
        self.assertEqual(self.decoder.lost_reason, "the audio ended")
        self.assertEqual(self.lost, [0])

    def test_decoding_goes_on_after_finish(self):
        card, spec, audio = self.audio("Robot 36")
        self.decoder.process_audio(audio)
        self.assertTrue(self.decoder.finish())
        before = self.decoder._freq.end
        self.assertEqual(before, len(audio), "the padding was taken out again")
        self.decoder.process_audio(audio)
        self.decoder.finish()
        self.assertEqual(len(self.complete), 2)
        self.assertGreater(_correlation(self.complete[1], card), 0.98)


class TestSstvHistoryMemory(unittest.TestCase):
    """HAMDSP-9: Received Images kept a copy of every image received in
    the session, also after auto-save had written it to a file: about
    1 MB per PD 120 image for as long as the app listened (190 MB after
    200 images). Saved images are now read back from their files."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.panel = SSTVPanel()
        self.panel.set_save_directory(self.tmp.name)
        self.panel.show()
        settle()
        self.panel._start_btn.click()  # listening, as through an ISS event

    def tearDown(self):
        panel = getattr(self, "panel", None)
        if panel is not None:
            if panel.is_receiving():
                panel._start_btn.click()
            panel.close()
            panel.deleteLater()
            settle()

    def receive(self, count: int, mode: str = "PD 120", flat: bool = False):
        """``count`` images through the decoder's completion callback (noise,
        or ``flat`` grey ones, whose files are small and quick to write)."""
        from sdr_module.ham.sstv import find_mode

        spec = find_mode(mode)
        decoder = self.panel._decoder
        rng = np.random.default_rng(len(self.panel._viewer.images))
        images = []
        for i in range(count):
            shape = (spec.height, spec.width, 3)
            if flat:
                image = np.full(shape, i % 256, np.uint8)
            else:
                image = rng.integers(0, 256, shape, np.uint8)
            decoder.state.mode = spec
            decoder.state.image_data = image
            decoder._on_image_complete(image)
            images.append(image.copy())
        return images

    def held(self):
        """The entries whose pixels are in memory."""
        return [e for e in self.panel._viewer.images if e.get("image") is not None]

    def held_rows(self):
        """Their rows in Received Images."""
        entries = self.panel._viewer.images
        return [i for i, e in enumerate(entries) if e.get("image") is not None]

    def test_saved_images_are_read_back_from_their_files(self):
        panel = self.panel
        images = self.receive(6)
        entries = panel._viewer.images
        self.assertEqual(len(entries), 6)
        self.assertTrue(all(e.get("filename") for e in entries))
        self.assertEqual(self.held_rows(), [5], "only the newest is held")
        np.testing.assert_array_equal(panel._viewer.get_current_image(), images[-1])
        for row in (1, 3, 0, 5, 2):
            panel._history_list.setCurrentRow(row)
            settle(1)
            self.assertEqual(panel._viewer.current_index, row)
            np.testing.assert_array_equal(panel._current_entry_image(), images[row])
            # The newest and the one shown.
            self.assertEqual(self.held_rows(), sorted({row, 5}))
        # Another image arrives: only it stays in memory.
        newest = self.receive(1)[0]
        self.assertEqual(self.held_rows(), [6])
        np.testing.assert_array_equal(panel._viewer.get_current_image(), newest)

    def test_images_that_were_not_saved_stay_in_memory(self):
        self.panel._auto_save_check.setChecked(False)
        images = self.receive(3)
        self.assertEqual(len(self.held()), 3, "there is no file to read back")
        self.panel._auto_save_check.setChecked(True)
        self.receive(2)
        held = self.held()
        self.assertEqual(len(held), 4, "the 3 unsaved ones and the newest")
        for entry, image in zip(held, images, strict=False):
            np.testing.assert_array_equal(entry["image"], image)

    def test_an_image_whose_save_failed_stays_in_memory(self):
        from unittest import mock

        from sdr_module.ham.gui import sstv_panel

        with mock.patch.object(sstv_panel, "save_rgb_image", return_value=False):
            first = self.receive(1)[0]
        self.receive(2)
        unsaved = self.panel._viewer.images[0]
        self.assertIsNone(unsaved.get("filename"))
        np.testing.assert_array_equal(unsaved["image"], first)
        self.assertEqual(len(self.held()), 2)

    def test_memory_stays_bounded(self):
        """40 PD 120 images held 38 MB of pixels (200 held 190 MB); now one."""
        self.receive(40, flat=True)
        self.assertEqual(len(self.panel._viewer.images), 40)
        held = sum(e["image"].nbytes for e in self.held())
        self.assertLess(held, 2 * 640 * 496 * 3)


SMETER_RATE = 12500.0  # an NBFM channel's samples, as a library caller has them


def _fm_voice(rng, n: int = 1250, snr_db: float = 60.0):
    """A voice-like NBFM signal (300-3000 Hz audio, 2.5 kHz deviation) that
    fills its 12.5 kHz channel, with noise; and the noise power (dBFS)."""
    freqs = np.fft.rfftfreq(n, 1 / SMETER_RATE)
    band = ((freqs > 300) & (freqs < 3000)).astype(float)
    audio = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * band, n)
    audio /= 3.0 * np.std(audio)
    signal = 0.03 * np.exp(2j * np.pi * 2500.0 * np.cumsum(audio) / SMETER_RATE)
    noise_p = float(np.mean(np.abs(signal) ** 2)) / 10 ** (snr_db / 10)
    noise = np.sqrt(noise_p / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return (signal + noise).astype(np.complex64), 10 * np.log10(noise_p)


class _MeterCase(unittest.TestCase):
    def setUp(self):
        from unittest import mock

        from sdr_module.ham import signal_meter as sm

        self.sm = sm
        self.now = 1000.0
        patcher = mock.patch.object(sm.time, "time", lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rng = np.random.default_rng(7)
        self.meter = sm.SignalMeter(SMETER_RATE)

    def run_for(self, seconds: float, snr_db: float = 60.0, **kw):
        """Blocks of 0.1 s for ``seconds``; the last reading."""
        reading = None
        for _ in range(int(round(seconds / 0.1))):
            self.now += 0.1
            block, _noise = _fm_voice(self.rng, snr_db=snr_db)
            reading = self.meter.update(block, **kw)
        return reading


class TestSignalMeterNoiseFloor(_MeterCase):
    """HAMDSP-10 (a): a floor set with set_noise_floor() was silently
    replaced 2 s later by the meter's own estimate (-120.5 -> -72.7 dBm,
    SNR 60 -> 12 dB, RST 59 -> 39); in 6bb37c7 it stayed."""

    def true_floor_dbm(self) -> float:
        _block, noise_dbfs = _fm_voice(self.rng)
        return noise_dbfs + self.sm.UNCALIBRATED_DBFS_TO_DBM

    def test_a_floor_set_once_stays_the_reference(self):
        floor = self.true_floor_dbm()
        self.meter.set_noise_floor(floor)
        for _ in range(4):
            reading = self.run_for(1.0)
            self.assertAlmostEqual(reading.noise_dbm, floor, delta=0.01)
            self.assertAlmostEqual(reading.snr_db, 60.0, delta=0.5)
            self.assertEqual(self.meter.get_rst(), "59")

    def test_a_retune_keeps_it(self):
        floor = self.true_floor_dbm()
        self.meter.set_noise_floor(floor)
        self.run_for(0.5)
        self.meter.reset_peak()
        self.assertTrue(self.meter.has_noise_floor)
        reading = self.run_for(3.0)
        self.assertAlmostEqual(reading.noise_dbm, floor, delta=0.01)

    def test_noise_given_with_the_updates_comes_first_while_it_comes(self):
        floor = self.true_floor_dbm()
        self.meter.set_noise_floor(floor)
        given = floor + 10.0 - self.sm.UNCALIBRATED_DBFS_TO_DBM  # in dBFS
        reading = self.run_for(3.0, noise_dbfs=given)
        self.assertAlmostEqual(
            reading.noise_dbm, given + self.sm.UNCALIBRATED_DBFS_TO_DBM, delta=0.1
        )
        reading = self.run_for(2.5)  # no more given: back to the floor set
        self.assertAlmostEqual(reading.noise_dbm, floor, delta=0.01)

    def test_clear_noise_floor_goes_back_to_the_estimate(self):
        self.meter.set_noise_floor(-60.0)  # far too high: SNR near 0
        self.assertLess(self.run_for(3.0).snr_db, 15.0)
        self.meter.clear_noise_floor()
        self.assertAlmostEqual(self.run_for(3.0).snr_db, 60.0, delta=1.5)

    def test_the_floor_is_in_dbm_and_follows_a_calibration(self):
        """As in 6bb37c7: SNR = level (dBm) - floor (dBm)."""
        floor = self.true_floor_dbm()
        self.meter.set_noise_floor(floor)
        self.meter.set_calibration(6.0)
        reading = self.run_for(1.0)
        self.assertAlmostEqual(reading.noise_dbm, floor, delta=0.01)
        self.assertAlmostEqual(reading.snr_db, 66.0, delta=0.5)


class TestSignalMeterFilledChannel(_MeterCase):
    """HAMDSP-10 (b): with no noise reference the meter took the median
    spectrum bin as the noise; a signal that fills the samples' band (an
    NBFM channel) is that median, so 60 dB SNR read 11.6 dB, RST 39."""

    def test_an_fm_channel_reads_its_snr(self):
        for snr in (60.0, 30.0, 12.0):
            with self.subTest(snr=snr):
                self.meter = self.sm.SignalMeter(SMETER_RATE)
                reading = self.run_for(2.0, snr_db=snr)
                self.assertAlmostEqual(reading.snr_db, snr, delta=1.5)
                if snr == 60.0:
                    self.assertEqual(self.meter.get_rst(), "59")

    def test_noise_alone_is_still_no_signal(self):
        for _ in range(30):
            self.now += 0.1
            hiss = 1e-3 * (
                self.rng.standard_normal(1250) + 1j * self.rng.standard_normal(1250)
            )
            reading = self.meter.update(hiss.astype(np.complex64))
        self.assertFalse(reading.signal_present)
        self.assertEqual(reading.readability, 1)
        self.assertAlmostEqual(reading.noise_dbm, reading.power_dbm, delta=1.0)

    def test_a_narrow_carrier_in_a_wide_span_is_unchanged(self):
        t = np.arange(4096) / 48000.0
        for _ in range(12):
            self.now += 0.1
            hiss = 1e-3 * (
                self.rng.standard_normal(4096) + 1j * self.rng.standard_normal(4096)
            )
            reading = self.meter.update(
                (0.05 * np.exp(2j * np.pi * 1000 * t) + hiss).astype(np.complex64)
            )
        expected = 10 * np.log10(0.05**2 / 2e-6)  # about 31 dB
        self.assertAlmostEqual(reading.snr_db, expected, delta=0.7)

    def test_compact_meter_takes_a_noise_reference(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.ham.gui.signal_meter_widget import CompactSignalMeter

        widget = CompactSignalMeter()
        self.addCleanup(widget.deleteLater)
        block, noise_dbfs = _fm_voice(self.rng, snr_db=40.0)
        widget.update_samples(block, noise_dbfs)
        self.assertAlmostEqual(widget.get_meter().get_reading().snr_db, 40.0, delta=0.5)


class TestEncodeFitsTheSampleRate(unittest.TestCase):
    """HAMDSP-11: sdr-scan encode took a carrier (with its shift) beyond the
    sample rate's Nyquist frequency and wrote the aliased signal without a
    word: ASCII at 8 kHz with --carrier 3900 put the 4400 Hz mark tone at
    -3600 Hz (mark and space swapped), PSK31 at 1 kHz with --carrier 1000
    landed on 0 Hz. The output is I/Q, so the limit is +-rate/2."""

    def encode(self, *args: str):
        import contextlib
        import io

        from sdr_module.cli import main

        out = Path(self.tmp.name) / "out.cf32"
        if out.exists():
            out.unlink()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            rc = main(["encode", *args, "--text", "CQ", "-o", str(out)])
        return rc, printed.getvalue(), out

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_carriers_past_nyquist_are_refused(self):
        for args, reach in (
            (("ascii", "--sample-rate", "8000", "--carrier", "3900"), "Hz reaches"),
            (("psk31", "--sample-rate", "1000", "--carrier", "1000"), "1030 Hz"),
            (("morse", "--sample-rate", "8000", "--carrier", "5000"), "Hz reaches"),
            (("rtty", "--sample-rate", "8000", "--carrier", "-3900"), "Hz reaches"),
        ):
            with self.subTest(args=args):
                rc, printed, out = self.encode(*args)
                self.assertEqual(rc, 1, printed)
                self.assertIn("Error:", printed)
                self.assertIn(reach, printed)
                self.assertFalse(out.exists(), "nothing is written")

    def test_the_error_says_what_fits(self):
        rc, printed, _out = self.encode(
            "ascii", "--sample-rate", "8000", "--carrier", "3900"
        )
        from sdr_module.protocols.text_modes import MODE_ASCII, fsk_bandwidth_hz

        width = fsk_bandwidth_hz(MODE_ASCII, 300.0, 1000.0)
        room = 4000.0 - width / 2.0
        self.assertIn("(±4000 Hz)", printed)
        self.assertIn(f"--carrier from {-room:.0f} to {room:.0f} Hz", printed)
        self.assertIn(f"--sample-rate of at least {2 * 3900 + width:.0f}", printed)

    def test_what_the_error_suggests_is_accepted(self):
        """The carriers and the rate the error names all fit. At its exact
        bounds it refused what it suggested: "--carrier 3010 Hz reaches
        4000 Hz, more than --sample-rate 8000 can hold (±4000 Hz): use a
        --carrier from -3010 to 3010 Hz, or a --sample-rate above 8000"."""
        import re

        for protocol, rate, carrier in (
            ("ascii", "8000", "3900"),
            ("ascii", "8000", "3010.5"),
            ("psk31", "1000", "1000"),
            ("morse", "8000", "5000"),
            ("rtty", "7999.999", "-3900"),
        ):
            with self.subTest(protocol=protocol, rate=rate, carrier=carrier):
                rc, printed, _out = self.encode(
                    protocol, "--sample-rate", rate, "--carrier", carrier
                )
                self.assertEqual(rc, 1, printed)
                reach = float(re.search(r"reaches ([\d.]+) Hz", printed).group(1))
                self.assertGreater(reach, float(rate) / 2.0, printed)
                low, high = re.search(
                    r"--carrier from (-?[\d.]+) to (-?[\d.]+) Hz", printed
                ).groups()
                least = re.search(r"--sample-rate of at least ([\d.]+)", printed)
                self.assertIsNotNone(least, printed)
                for args in (
                    ("--sample-rate", rate, "--carrier", low),
                    ("--sample-rate", rate, "--carrier", high),
                    ("--sample-rate", least.group(1), "--carrier", carrier),
                ):
                    rc, again, out = self.encode(protocol, *args)
                    self.assertEqual(rc, 0, f"{args}: {again}")
                    self.assertTrue(out.exists())

    def test_a_signal_wider_than_the_rate_asks_for_a_higher_rate(self):
        rc, printed, _out = self.encode(
            "ascii", "--sample-rate", "1000", "--carrier", "0"
        )
        self.assertEqual(rc, 1)
        self.assertNotIn("--carrier from", printed)
        self.assertIn("--sample-rate of at least", printed)

    def test_what_fits_is_written_and_lands_where_asked(self):
        # (protocol options, carrier, how far the strongest tone may be
        # from it: RTTY idles on the mark, 85 Hz above)
        for args, carrier, delta in (
            (("rtty", "--sample-rate", "8000", "--carrier", "1000"), 1000.0, 95.0),
            (("psk31", "--sample-rate", "1000", "--carrier", "-300"), -300.0, 20.0),
            (("morse", "--sample-rate", "8000", "--carrier", "3800"), 3800.0, 10.0),
        ):
            with self.subTest(args=args):
                rc, printed, out = self.encode(*args)
                self.assertEqual(rc, 0, printed)
                iq = np.fromfile(out, np.complex64)
                spectrum = np.abs(np.fft.fft(iq)) ** 2
                freqs = np.fft.fftfreq(len(iq), 1.0 / float(args[2]))
                peak = float(freqs[int(np.argmax(spectrum))])
                self.assertAlmostEqual(peak, carrier, delta=delta)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
