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
* MS-5: no Open Recording while transmitting; Disconnect ends a
  transmission before it closes the HackRF.
* MS-6: the Recording group's Pause shows the pause of the recording its
  Record choice shows.
* MS-7: a scan doesn't change the filter width each mode remembers.
* MS-8: a damaged squelch threshold setting (not a number, infinite or
  huge) doesn't stop the window from opening.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_review_mainstate.py``
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_station_tx import _TxCase  # noqa: E402

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
        self.store.set("recording/folder", self.folder)  # (audio recordings)
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
            self.scanner._status.text(), "Scan stopped: you opened a recording."
        )
        self.win._playback.close(restore=True)
        self.assertEqual(self.listening(), self.BEFORE)
        self.assertAlmostEqual(self.win._lo_frequency(), 100.1e6)

    def test_a_recording_at_another_rate_says_a_recording_is_playing(self):
        """It said 'the sample rate changed' (that check came first): the
        reason given is the recording."""
        self.scan_to_the_repeater()
        self.assertTrue(self.win._open_recording(self.recording(145.0e6, 1.0e6)))
        self.frames(0.1)
        self.assertEqual(
            self.scanner._status.text(), "Scan stopped: you opened a recording."
        )
        self.win._playback.close(restore=True)
        self.assertEqual(self.listening(), self.BEFORE)

    def test_a_recording_that_cant_be_opened(self):
        """(Second checker) The scan ends before the file is opened: when
        it then can't be (it went away, no permission), the Scanner tab
        said "a recording is playing" next to "Can't play ...". The
        receiver is back where the scan started either way."""
        from sdr_module.gui import iq_files

        self.scan_to_the_repeater()
        path = self.recording(145.0e6, 2.4e6)
        with mock.patch.object(
            iq_files.IqFileOptions,
            "open_player",
            side_effect=OSError("Permission denied"),
        ):
            self.assertFalse(self.win._open_recording(path))
        self.assertFalse(self.win._playback.active)
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(
            self.scanner._status.text(), "Scan stopped: you opened a recording."
        )
        self.assertEqual(
            self.status(), ("Can't play take.sigmf-data: Permission denied", "danger")
        )
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

    def test_below_the_receivers_range_is_still_refused(self):
        """Only the top of the receiver's range gives way to a recording: a
        recording centred at 0.5 MHz doesn't make 0 Hz and below tunable
        (the Frequency field can't hold them)."""
        win = self.win
        win._playback.close(restore=True)
        self.assertTrue(
            win._open_recording(self.recording(0.5e6, 2.4e6, "mw.sigmf-data"))
        )
        win._set_demod_mode("AM", None, "5 kHz")
        self.assertFalse(win.set_frequency(-100e3))
        self.assertIn("is out of range", self.status()[0])
        self.assertEqual((win._tuned_hz, win._current_frequency()), (0.5e6, 0.5e6))
        self.assertTrue(win.set_frequency(0.1e6))
        self.assertEqual((win._tuned_hz, win._current_frequency()), (0.1e6, 0.1e6))

    def test_closing_it_goes_back_to_the_receiver_and_its_range(self):
        win = self.win
        win._playback.close(restore=True)
        self.assertEqual(win._control_panel._freq_input.tuning_range(), (1e3, 6e9))
        self.assertEqual(win._current_frequency(), 100.1e6)
        self.assertEqual(win._lo_frequency(), 100.1e6)
        self.assertFalse(win.set_frequency(self.CENTER))  # the demo device: no
        self.assertIn("is out of range", self.status()[0])


# ---------------------------------------------------------------------------
# MS-5: recordings and transmitting
# ---------------------------------------------------------------------------


class TestPlaybackAndTransmitting(_TxCase):
    """Receiving on a HackRF, Send ID Now, then Open Recording while the ID
    is on the air: the file opened; when the ID ended, "receiving resumed"
    into the file, and Close Recording left the HackRF stopped. Disconnect
    in that state closed the HackRF under the transmit thread ("the HackRF
    stopped transmitting early")."""

    def setUp(self):
        super().setUp()
        folder = tempfile.mkdtemp(prefix="sdr-mainstate-tx-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.path = _save_sigmf(
            os.path.join(folder, "take.sigmf-data"), 2.4e6, 146.52e6
        )
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(10)  # (a long ID: still on the air below)
        self.dev, self.backend = self.hackrf(pace=True)
        self.win._start_acquisition()
        self.assertTrue(self.station.send_station_id())
        self.assertTrue(self.station.busy)

    def test_open_recording_waits_for_the_transmission_to_end(self):
        win = self.win
        self.assertFalse(win._open_recording(self.path))
        self.assertFalse(win._playback.active)
        self.assertIs(win._device, self.dev)
        self.assertEqual(
            self.status(),
            (
                "Wait until the transmission ends (or press Stop) to play a "
                "recording.",
                "warning",
            ),
        )
        self.assertTrue(self.station.busy)  # (on the air still)
        self.panel._id_now_btn.click()  # Stop
        self.wait_done(5)
        self.assertIs(win._device, self.dev)
        self.assertTrue(win._is_running)  # receiving on the HackRF again

    def test_disconnect_ends_the_transmission_before_closing_the_hackrf(self):
        """(With a recording opened by a path around Open Recording.)"""
        from sdr_module.gui.iq_files import describe_iq_file

        options, missing = describe_iq_file(self.path)
        self.assertEqual(missing, set())
        self.assertTrue(self.win._playback.open(options))
        on_air_when_closed = []
        close = self.dev.close

        def watched_close():
            on_air_when_closed.append(self.station.busy)
            return close()

        self.dev.close = watched_close
        self.win._disconnect_device()
        self.assertEqual(on_air_when_closed, [False])
        self.assertIsNone(self.win._device)
        self.assertEqual(self.station.engine.last_result.status, "stopped")


# ---------------------------------------------------------------------------
# MS-6: the Recording group's Pause and its Record choice
# ---------------------------------------------------------------------------


class TestPauseFollowsTheRecordChoice(_MainStateCase):
    """Pause a recording in the Recording group, then switch its Record
    choice to the other kind and back: the group said "● Recording" with
    "❚❚ Pause" while nothing was captured, and the next press of Pause only
    paused it (again)."""

    PAUSED = ("❚❚ Paused", "▶ Resume", True)

    def group(self):
        panel = self.win._control_panel
        return (
            panel._record_status.text(),
            panel._pause_btn.text(),
            panel._pause_btn.isChecked(),
        )

    def test_a_paused_iq_recording_still_shows_paused(self):
        win, panel = self.win, self.win._control_panel
        panel.set_record_kind("iq")
        panel._record_btn.click()
        self.frames(0.2)
        self.assertTrue(win._recording)
        panel._pause_btn.click()
        self.assertTrue(win._recording_paused)
        self.assertEqual(self.group(), self.PAUSED)
        panel.set_record_kind("audio")
        self.assertEqual(self.group()[0], "Ready")  # (no audio recording)
        panel.set_record_kind("iq")
        self.assertEqual(self.group(), self.PAUSED)
        self.assertTrue(panel.is_recording_paused())
        # One press resumes it.
        captured = win._buffer_sample_count()
        panel._pause_btn.click()
        self.assertFalse(win._recording_paused)
        self.assertEqual(self.group(), ("● Recording", "❚❚ Pause", False))
        self.frames(0.2)
        self.assertGreater(win._buffer_sample_count(), captured)
        win._toggle_recording(False)

    def test_a_paused_audio_recording_still_shows_paused(self):
        win, panel = self.win, self.win._control_panel
        panel.set_record_kind("audio")
        panel._record_btn.click()
        take = win._audio_take
        self.assertIsNotNone(take)
        panel._pause_btn.click()
        self.assertTrue(take.paused)
        panel.set_record_kind("iq")
        panel.set_record_kind("audio")
        self.assertEqual(self.group(), self.PAUSED)
        panel._pause_btn.click()
        self.assertFalse(take.paused)
        self.assertEqual(self.group(), ("● Recording audio", "❚❚ Pause", False))
        win._stop_audio_recording(quiet=True)

    def test_each_recording_shows_its_own_pause(self):
        """Both recording: the I/Q one paused, the audio one not."""
        win, panel = self.win, self.win._control_panel
        panel.set_record_kind("iq")
        panel._record_btn.click()
        panel._pause_btn.click()
        panel.set_record_kind("audio")
        panel._record_btn.click()
        self.assertIsNotNone(win._audio_take)
        self.assertEqual(self.group(), ("● Recording audio", "❚❚ Pause", False))
        panel.set_record_kind("iq")
        self.assertEqual(self.group(), self.PAUSED)
        panel.set_record_kind("audio")
        self.assertEqual(self.group(), ("● Recording audio", "❚❚ Pause", False))
        self.assertTrue(win._recording_paused)
        self.assertFalse(win._audio_take.paused)
        win._stop_audio_recording(quiet=True)
        win._toggle_recording(False)


# ---------------------------------------------------------------------------
# MS-7: scans and the filter width each mode remembers
# ---------------------------------------------------------------------------


class TestScansLeaveTheRememberedFilters(_MainStateCase):
    """AM is listened to with a 15 kHz filter (so AM remembers 15 kHz). A
    bookmark scan visiting an AM bookmark with a 5 kHz filter made AM
    remember 5 kHz: picking AM gave 5 kHz, in the next session too,
    although the scan tunes "quietly"."""

    def setUp(self):
        super().setUp()
        win, panel = self.win, self.win._control_panel
        win._tune_to(100.1e6, recenter=True)
        panel._demod_combo.setCurrentText("AM")
        panel._bw_combo.setCurrentText("15 kHz")  # by hand: remembered
        panel._demod_combo.setCurrentText("FM")
        self.assertEqual(panel.filter_memory()["AM"], {"width": "15 kHz"})
        self.memory = panel.filter_memory()
        win._bookmarks_panel.add_bookmark("Tower", 125.0e6, "AM", "", "5 kHz")
        self.scanner = win._scanner_panel
        self.scanner.set_scan_mode("bookmarks")
        self.assertTrue(self.scanner.start_scan())
        self.assertTrue(
            self.frames(1.0, lambda: abs(win._current_frequency() - 125.0e6) < 1)
        )
        self.assertEqual(panel._bw_combo.currentText(), "5 kHz")  # (its own)

    def test_a_scan_ending_where_it_started(self):
        panel = self.win._control_panel
        self.scanner.stop_scan(stay=False)
        self.assertEqual(panel.get_demod_mode(), "FM")
        self.assertEqual(panel.filter_memory(), self.memory)
        panel._demod_combo.setCurrentText("AM")
        self.assertEqual(panel._bw_combo.currentText(), "15 kHz")
        self.win.close()
        saved = json.loads(self.store.values["audio/mode_filters"])
        self.assertEqual(saved["AM"], {"width": "15 kHz"})

    def test_a_scan_staying_on_the_bookmark(self):
        """The bookmark's filter stays on (it is listened to), but AM still
        remembers the width you chose."""
        panel = self.win._control_panel
        self.scanner.stop_scan(stay=True)
        self.assertEqual(self.listening()[:2], (125.0e6, "AM"))
        self.assertEqual(panel._bw_combo.currentText(), "5 kHz")
        self.assertEqual(panel.filter_memory(), self.memory)

    def test_a_width_chosen_by_hand_is_remembered_again_after_it(self):
        panel = self.win._control_panel
        self.scanner.stop_scan(stay=True)
        panel._bw_combo.setCurrentText("8 kHz")
        self.assertEqual(panel.filter_memory()["AM"], {"width": "8 kHz"})


# ---------------------------------------------------------------------------
# MS-8: a damaged squelch setting
# ---------------------------------------------------------------------------


class TestDamagedSquelchSetting(unittest.TestCase):
    """squelch_db = nan, inf or 1e400 in the settings file (damaged, or
    edited by hand) stopped the window from opening, at every launch:
    ControlPanel.set_squelch_db's int(round(db)) raised in __init__ (and a
    huge finite value overflowed the slider)."""

    def window(self, squelch):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window

        store = _FakeSettings()
        store.values["squelch_db"] = squelch  # (QSettings gives text)
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", lambda: store)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.close)
        return win

    def test_the_window_opens_with_the_default_threshold(self):
        for value in ("nan", "inf", "-inf", "1e400"):
            with self.subTest(value=value):
                win = self.window(value)
                self.assertEqual(win._control_panel.get_squelch_db(), -80.0)
                self.assertEqual(win._squelch_db, -80.0)

    def test_a_huge_threshold_is_the_default_too(self):
        # (F-window, ROB-08: its nearest end, 0 dBFS, kept the squelch shut
        # for good; only the slider's range is ever saved.)
        for value in ("1e12", "50"):
            with self.subTest(value=value):
                win = self.window(value)
                self.assertEqual(win._control_panel.get_squelch_db(), -80.0)
                self.assertEqual(win._squelch_db, -80.0)

    def test_the_panel_ignores_a_threshold_that_is_not_a_number(self):
        require_pyqt6(self)
        from sdr_module.gui.control_panel import ControlPanel

        panel = ControlPanel()
        self.addCleanup(panel.deleteLater)
        panel.set_squelch_db(-50.4)
        self.assertEqual(panel.get_squelch_db(), -50.0)
        for value in (float("nan"), float("inf"), float("-inf"), None, "x"):
            panel.set_squelch_db(value)
            self.assertEqual(panel.get_squelch_db(), -50.0)
        panel.set_squelch_db(-1e300)
        self.assertEqual(panel.get_squelch_db(), -120.0)
        panel.set_squelch_db(1e300)
        self.assertEqual(panel.get_squelch_db(), 0.0)


# ---------------------------------------------------------------------------
# Checker, after MS-4: what the messages say a recording above 6 GHz covers
# ---------------------------------------------------------------------------


class TestFrequencyRangeText(unittest.TestCase):
    """format_mhz_range gave six significant digits: 10488.55–10490.95 MHz
    read "10488.5–10491 MHz", wider than the range itself."""

    def test_to_the_khz(self):
        from sdr_module.gui.tuning import format_mhz_range

        self.assertEqual(
            format_mhz_range(10488.55e6, 10490.95e6), "10488.55–10490.95 MHz"
        )
        self.assertEqual(
            format_mhz_range(1088.976e6, 1091.024e6), "1088.976–1091.024 MHz"
        )

    def test_as_before_where_six_digits_were_enough(self):
        from sdr_module.gui.tuning import format_mhz_range

        for (low, high), text in (
            ((500e3, 1.766e9), "0.5–1766 MHz"),
            ((1e6, 6e9), "1–6000 MHz"),
            ((1e3, 6e9), "0.001–6000 MHz"),
            ((143.8e6, 146.2e6), "143.8–146.2 MHz"),
            ((-0.7e6, 1.7e6), "-0.7–1.7 MHz"),
            ((145.8e6 - 1e-7, 146.2e6 + 1e-7), "145.8–146.2 MHz"),  # (float noise)
        ):
            with self.subTest(text=text):
                self.assertEqual(format_mhz_range(low, high), text)

    def test_never_wider_than_the_range(self):
        """A frequency outside the range never reads as inside it."""
        from sdr_module.gui.tuning import format_mhz_range

        self.assertEqual(format_mhz_range(143.8005e6, 146.2005e6), "143.801–146.2 MHz")

    def test_not_a_number_reads_as_such(self):
        """(Second checker) A recording's damaged centre (NaN or Infinity)
        made it raise: it reads as it did before rounding to the kHz."""
        from sdr_module.gui.tuning import format_mhz_range

        nan, inf = float("nan"), float("inf")
        for (low, high), text in (
            ((nan, nan), "nan–nan MHz"),
            ((inf, inf), "inf–inf MHz"),
            ((-inf, inf), "-inf–inf MHz"),
            ((144e6, inf), "144–inf MHz"),
        ):
            with self.subTest(text=text):
                self.assertEqual(format_mhz_range(low, high), text)


class TestARecordingWithADamagedCentre(_MainStateCase):
    """(Second checker) A SigMF whose core:frequency is NaN or Infinity
    (Python's json writes and reads those): Open Recording raised half way
    once the range text was given to the kHz (the LO on the file, the
    receiver stopped, the strip not shown). It opens as it did before, and
    Close Recording goes back to the receiver."""

    def damaged(self, value):
        path = self.recording(145.0e6, 2.4e6, f"damaged_{value}.sigmf-data")
        meta = path[: -len("data")] + "meta"
        with open(meta, encoding="utf-8") as f:
            data = json.load(f)
        data["captures"][0]["core:frequency"] = float(value)
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(data, f)  # ("NaN", "Infinity")
        return path

    def test_it_opens_and_closes(self):
        win = self.win
        before = (win._device, win._is_running, self.listening())
        for value in ("nan", "inf"):
            with self.subTest(value=value):
                win._open_recording(self.damaged(value))
                self.assertTrue(win._playback.active)
                win._playback.close(restore=True)
                self.assertFalse(win._playback.active)
                self.assertIsNone(win._lo_lock)
                self.assertEqual(win._lo_frequency(), 100.1e6)
                self.assertEqual(
                    (win._device, win._is_running, self.listening()), before
                )


class TestTheSpanOfARecordingAbove6GHz(_MainStateCase):
    """The QO-100 recording (10488.55–10490.95 MHz) once MS-4 made it
    tunable: "10490.980 MHz is outside the recording, which covers
    10488.5–10491 MHz" contradicted itself, and a frequency typed above it
    was refused with "the receiver tunes 0.001–10491 MHz" (the receiver
    doesn't; the recording ends there) and a hint to type it in kHz."""

    CENTER = 10489.75e6
    COVERED = "10488.55–10490.95 MHz"

    def setUp(self):
        super().setUp()
        path = self.recording(self.CENTER, 2.4e6, seconds=0.5, offset_hz=50e3)
        self.assertTrue(self.win._open_recording(path))

    def range_line(self):
        """The Frequency field's tooltip line that says what can be tuned."""
        tip = self.win._control_panel._freq_input._freq_input.toolTip()
        return tip.splitlines()[2].partition(". ")[0]

    def test_the_playing_message_gives_its_span(self):
        self.assertIn(f"Playing take.sigmf-data ({self.COVERED}).", self.status()[0])

    def test_a_refusal_never_shows_a_range_holding_the_frequency(self):
        win = self.win
        win._set_demod_mode("AM", None, "5 kHz")
        for freq, text in ((10490.98e6, "10490.980"), (10488.52e6, "10488.520")):
            with self.subTest(freq=text):
                self.assertFalse(win.set_frequency(freq))
                self.assertEqual(
                    self.status(),
                    (
                        f"{text} MHz is outside the recording, which covers "
                        f"{self.COVERED}.",
                        "danger",
                    ),
                )

    def test_typing_above_it_says_what_the_recording_covers(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication

        self.assertEqual(self.range_line(), f"The recording covers {self.COVERED}")
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "10491.5")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertEqual(self.win._current_frequency(), self.CENTER)
        self.assertEqual(
            self.status(),
            (
                f"10491.5 MHz is out of range: the recording covers {self.COVERED}.",
                "danger",
            ),
        )

    def test_closed_the_field_describes_the_receiver_again(self):
        self.win._playback.close(restore=True)
        self.assertEqual(self.range_line(), "Tunes 0.001–6000 MHz")

    def test_a_recording_below_6ghz_keeps_the_receivers_wording(self):
        """(Only a recording above the receiver's range changes it.)"""
        win = self.win
        win._playback.close(restore=True)
        path = self.recording(145.0e6, 2.4e6, "fm.sigmf-data")
        self.assertTrue(win._open_recording(path))
        self.assertEqual(self.range_line(), "Tunes 0.001–6000 MHz")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
