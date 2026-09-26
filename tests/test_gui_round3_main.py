#!/usr/bin/env python3
"""
Main window, launcher and audio sink: the third review round.

Recording "armed" state on the Recording panel, bookmarks and the Decoder
panel tuning with their modes, reading every queued hardware transfer,
the real-time demo device, the radio tuner following any frequency, the
100.1 MHz default, copy fixes, the first-run wizard's device name and
Connect, closing while recording, and AudioSink reporting failure.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round3_main.py``
"""

import argparse
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _fake_hardware_device,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication, QMessageBox


_saved_style = None


def setUpModule():  # noqa: N802 - unittest hook
    """Wiring tests: run on the unstyled app (much faster to build)."""
    global _saved_style
    if not HAS_PYQT6:
        return
    qapp = QApplication.instance()
    _saved_style = (qapp.styleSheet(), qapp.palette())
    qapp.setStyleSheet("")


def tearDownModule():  # noqa: N802 - unittest hook
    if not HAS_PYQT6 or _saved_style is None:
        return
    qapp = QApplication.instance()
    qapp.setStyleSheet(_saved_style[0])
    qapp.setPalette(_saved_style[1])


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


def _answer_message_box(button_text, seen=None):
    """A QMessageBox.exec replacement that clicks the button ``button_text``
    (and records the box's informative text in ``seen``)."""

    def fake_exec(box):
        if seen is not None:
            seen.append(box.informativeText())
        for button in box.buttons():
            if button.text().replace("&", "") == button_text:
                button.click()
                return 0
        raise AssertionError(f"no {button_text!r} button")

    return fake_exec


class _FakeClock:
    """Stands in for the ``time`` module inside main_window."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def strftime(self, fmt, *args):
        import time

        return time.strftime(fmt, *args)

    def advance(self, seconds):
        self.now += float(seconds)


class _FakeDemo:
    """A demo source that makes a steady carrier, costing ``cost`` seconds
    of (fake) time per sample."""

    def __init__(self, clock, cost=0.0, rate=2.4e6):
        self.clock = clock
        self.cost = float(cost)
        self.sample_rate = rate
        self.requests = []

    def read_samples(self, num_samples):
        self.requests.append(int(num_samples))
        self.clock.advance(self.cost * num_samples)
        return np.full(int(num_samples), 0.1, dtype=np.complex64)

    def set_frequency(self, freq):
        return True

    def start_rx(self):
        return True

    def stop_rx(self):
        return True

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 1. Recording armed
# ---------------------------------------------------------------------------


class TestRecordingArmed(_WindowTestCase):
    demo = True

    def test_panel_follows_armed_and_capturing(self):
        panel = self.win._control_panel
        self.win._stop_acquisition()
        self.win._record_button.click()  # receiver stopped: armed
        self.assertTrue(panel.is_recording_armed())
        self.assertIn("Armed", panel._record_status.text())
        self.assertEqual(self.win._recording_label.text(), "REC ARMED")

        self.win._start_acquisition()  # receiving: capturing
        self.assertFalse(panel.is_recording_armed())
        self.assertIn("Recording", panel._record_status.text())
        self.win._update_display()
        self.assertFalse(panel.is_recording_armed())

        self.win._stop_acquisition()  # armed again
        self.assertTrue(panel.is_recording_armed())
        self.win._record_button.click()  # stopped
        self.assertFalse(panel.is_recording_armed())
        self.assertEqual(panel._record_status.text(), "Ready")

    def test_panel_record_button_arms_too(self):
        panel = self.win._control_panel
        self.win._release_device()
        self.win._refresh_state_ui()
        panel._record_btn.click()
        self.assertTrue(self.win._recording)
        self.assertTrue(panel.is_recording_armed())
        panel._record_btn.click()
        self.assertFalse(self.win._recording)
        self.assertFalse(panel.is_recording_armed())

    def test_pause_shows_paused_not_armed(self):
        panel = self.win._control_panel
        self.win._stop_acquisition()
        self.win._record_button.click()
        panel._pause_btn.click()
        self.assertTrue(self.win._recording_paused)
        self.assertIn("Paused", panel._record_status.text())
        panel._pause_btn.click()
        self.assertIn("Armed", panel._record_status.text())
        self.win._record_button.click()


# ---------------------------------------------------------------------------
# 2-3. Bookmarks and the Decoder panel tune with their modes
# ---------------------------------------------------------------------------


class TestTuneWithModes(_WindowTestCase):
    def test_bookmark_tunes_frequency_and_mode(self):
        self.win._set_demod_mode("AM")
        self.win._bookmarks_panel.tune_requested.emit(
            146.52e6, "2m calling", "FM", "5 kHz"
        )
        panel = self.win._control_panel
        self.assertAlmostEqual(self.freq(), 146.52e6)
        self.assertEqual(panel.get_demod_mode(), "FM")
        self.assertEqual(panel.get_fm_deviation_text(), "5 kHz")
        self.assertEqual(self.win._message_label.text(), "Tuned to 2m calling (FM)")

    def test_bookmark_without_mode_keeps_the_mode(self):
        self.win._set_demod_mode("USB")
        self.win._on_bookmark_tune(7.074e6, "FT8")  # old two-argument shape
        self.assertEqual(self.win._control_panel.get_demod_mode(), "USB")
        self.assertEqual(self.win._message_label.text(), "Tuned to FT8")

    def test_ctrl_b_saves_mode_and_deviation(self):
        self.win.set_frequency(162.55e6)
        self.win._set_demod_mode("FM", "5 kHz")
        self.win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual(saved["demod"], "FM")
        self.assertEqual(saved["fm_deviation"], "5 kHz")
        self.assertIn("(FM)", self.win._message_label.text())
        # Double-clicking it later brings the mode back.
        self.win._set_demod_mode("AM")
        self.win.set_frequency(100e6)
        self.win._bookmarks_panel._tune_index(len(self.store.bookmarks) - 1)
        self.assertAlmostEqual(self.freq(), 162.55e6)
        self.assertEqual(self.win._control_panel.get_demod_mode(), "FM")

    def test_bookmarks_add_button_saves_the_listening_mode(self):
        panel = self.win._bookmarks_panel
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.assertEqual(panel._current_mode(14.2e6), ("USB", ""))

    def test_decoder_panel_tunes_and_follows_the_receiver(self):
        decoder = self.win._decoder_panel
        self.win.set_frequency(131.55e6)
        self.assertEqual(decoder._tuned_hz, 131.55e6)
        self.assertFalse(decoder._receiving)
        decoder.tune_requested.emit(144.39e6, "AX.25/APRS (144.390 MHz)", "FM", "5 kHz")
        self.assertAlmostEqual(self.freq(), 144.39e6)
        self.assertEqual(decoder._tuned_hz, self.freq())
        self.assertEqual(self.win._control_panel.get_demod_mode(), "FM")
        self.assertEqual(
            self.win._message_label.text(), "Tuned to AX.25/APRS (144.390 MHz, FM)"
        )

    def test_decoder_panel_knows_when_the_receiver_runs(self):
        decoder = self.win._decoder_panel
        self.win._start_demo_mode()
        self.assertTrue(decoder._receiving)
        self.win._stop_acquisition()
        self.assertFalse(decoder._receiving)
        self.win._start_acquisition()
        self.win._disconnect_device()
        self.assertFalse(decoder._receiving)


# ---------------------------------------------------------------------------
# 4. Hardware: every queued transfer per frame
# ---------------------------------------------------------------------------


class TestHardwareDrain(_WindowTestCase):
    def _device(self, blocks, rate):
        dev = _fake_hardware_device(blocks)
        dev._state.sample_rate = rate
        return dev

    def _keeps_up(self, rate):
        """Tell the pacer that frames take well under real time."""
        for _ in range(3):
            self.win._pacer.measured(int(rate), rate, 0.1)

    def tearDown(self):
        if HAS_PYQT6:
            self.win._recording = False
            self.win._is_running = False
            self.win._device = None
        super().tearDown()

    def test_high_rate_hackrf_is_read_in_full(self):
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        transfer = 131072  # one HackRF USB transfer
        blocks = [
            np.full(transfer, 0.01 * (i + 1), dtype=np.complex64) for i in range(8)
        ]
        dev = self._device(list(blocks), 10e6)
        self.win._device = dev
        self.win._is_running = True
        self.win._recording = True
        self.win._capture_pending = True
        self._keeps_up(10e6)
        seen = []
        self.win._spectrum.update_spectrum = seen.append
        self.win._update_display()
        # All eight transfers in one frame, none waited for...
        self.assertEqual(set(dev.timeouts), {0.0})
        recorded = np.concatenate(self.win._samples_buffer)
        self.assertEqual(len(recorded), 8 * transfer)
        np.testing.assert_array_equal(recorded, np.concatenate(blocks))
        # ...while the plots keep one fixed-size FFT of the newest samples.
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]), DISPLAY_BLOCK)

    def test_frames_are_capped(self):
        from sdr_module.gui import main_window as mw

        rate, transfer = 10e6, 131072
        dev = self._device(
            [np.zeros(transfer, dtype=np.complex64) for _ in range(60)], rate
        )
        self.win._device = dev
        # Until frames are known to keep up: a short probe...
        got = self.win._read_block()
        self.assertGreaterEqual(len(got), mw._PROBE_READ_S * rate)
        self.assertLess(len(got), mw._PROBE_READ_S * rate + transfer)
        # ...then at most a quarter second; the rest waits for the next one.
        self._keeps_up(rate)
        got = self.win._read_block()
        self.assertGreaterEqual(len(got), mw._MAX_READ_S * rate)
        self.assertLess(len(got), mw._MAX_READ_S * rate + transfer)
        self.assertGreater(60 - len(dev.timeouts), 0)

    def test_fast_device_counts_as_real_time(self):
        clock = _FakeClock()
        rate, transfer = 20e6, 131072
        per_frame = 5  # ~153 transfers/s at 30 frames/s
        blocks = []
        dev = self._device(blocks, rate)
        self.win._device = dev
        self.win._is_running = True
        with mock.patch.object(self.mw, "time", clock):
            for _ in range(40):
                blocks.extend(
                    np.zeros(transfer, dtype=np.complex64) for _ in range(per_frame)
                )
                self.win._update_display()
                clock.advance(per_frame * transfer / rate)
            self.assertTrue(self.win._realtime())
        self.assertFalse(self.win._pacer.overloaded)
        self.assertEqual(blocks, [])  # nothing left behind

    def test_a_computer_that_cannot_keep_up_reads_one_transfer(self):
        """Processing slower than real time: frames go back to one transfer
        each (the window stays responsive), audio is muted with a hint."""
        clock = _FakeClock()
        rate, transfer = 20e6, 131072
        blocks = []
        dev = self._device(blocks, rate)
        self.win._device = dev
        self.win._is_running = True
        self.win._set_demod_mode("FM")
        self.win._audio.write = lambda _audio: None
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-150)
        process = self.win._process_block

        def slow_process(samples):  # 1.5x real time
            clock.advance(1.5 * len(samples) / rate)
            process(samples)

        self.win._process_block = slow_process
        with mock.patch.object(self.mw, "time", clock):
            for _ in range(45):
                blocks.extend(
                    np.full(transfer, 0.1, dtype=np.complex64) for _ in range(5)
                )
                del blocks[:-100]  # the driver's queue holds 100 transfers
                before, tick = len(blocks), clock.now
                self.win._update_display()
                clock.now = max(clock.now, tick + 0.033)  # the display timer
            self.assertTrue(self.win._pacer.overloaded)
            self.assertEqual(before - len(blocks), 1)
            self.assertFalse(self.win._realtime())
        self.assertTrue(self.win._realtime_hint_shown)
        self.assertIn("20 MS/s", self.win._message_label.text())
        self.assertIn("lower sample rate", self.win._message_label.text())


# ---------------------------------------------------------------------------
# 5. The demo device in real time
# ---------------------------------------------------------------------------


class TestPacer(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window
        self.pacer = main_window._Pacer()

    def test_demo_reads_follow_wall_clock_time(self):
        mw, pacer = self.mw, self.pacer
        rate = 2.4e6
        self.assertEqual(pacer.demo_request(rate, 10.0), mw.DISPLAY_BLOCK)  # first
        self.assertEqual(pacer.demo_request(rate, 10.033), round(rate * 0.033))
        self.assertEqual(pacer.demo_request(rate, 10.0335), mw.DISPLAY_BLOCK)
        # Unknown speed: a long gap is capped at the probe length...
        probe = int(rate * mw._PROBE_READ_S)
        self.assertEqual(pacer.demo_request(rate, 12.0), probe)
        pacer.measured(80000, rate, 0.004)
        self.assertEqual(pacer.demo_request(rate, 14.0), probe)
        # ...once frames are known to keep up, at a quarter second.
        pacer.measured(80000, rate, 0.004)
        pacer.measured(80000, rate, 0.004)
        self.assertFalse(pacer.overloaded)
        self.assertEqual(pacer.demo_request(rate, 16.0), int(rate * mw._MAX_READ_S))
        pacer.restart()
        self.assertEqual(pacer.demo_request(rate, 20.0), mw.DISPLAY_BLOCK)

    def test_overload_falls_back_to_one_block_and_recovers(self):
        mw, pacer = self.mw, self.pacer
        rate = 2.4e6
        pacer.measured(1024, rate, 1.0)  # too short to judge
        self.assertFalse(pacer.overloaded)
        for _ in range(3):
            pacer.measured(80000, rate, 0.045)  # 45 ms for 33 ms of signal
        self.assertTrue(pacer.overloaded)
        self.assertEqual(pacer.read_limit(rate), 0.0)
        pacer.demo_request(rate, 1.0)
        self.assertEqual(pacer.demo_request(rate, 1.5), mw.DISPLAY_BLOCK)
        for _ in range(2):
            pacer.measured(80000, rate, 0.02)  # 0.6: in between
        self.assertTrue(pacer.overloaded)
        for _ in range(3):
            pacer.measured(80000, rate, 0.005)
        self.assertFalse(pacer.overloaded)
        pacer.reset()
        self.assertEqual(pacer.read_limit(rate), rate * mw._PROBE_READ_S)


class TestDemoRealTime(_WindowTestCase):
    demo = True

    def _run(self, cost, frames=45):
        clock = _FakeClock()
        source = _FakeDemo(clock, cost)
        spoken = []
        self.win._audio.write = spoken.append
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-100)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        with mock.patch.object(self.mw, "time", clock):
            self.win._stop_acquisition()
            self.win._device = source  # still in demo mode
            self.win._start_acquisition()
            for _ in range(frames):
                self.win._update_display()
                clock.advance(0.033)
            realtime = self.win._realtime()
        self.win._stop_acquisition()
        self.win._device = None
        self.win._demo_mode = False
        self.win._refresh_state_ui()
        return source, spoken, realtime

    def test_fast_demo_source_is_real_time_and_heard(self):
        source, spoken, realtime = self._run(cost=0.02e-6)
        self.assertTrue(realtime)
        self.assertTrue(spoken)
        self.assertFalse(self.win._realtime_hint_shown)
        # Reads cover the time that passed (33 ms at 2.4 MS/s), not a
        # 2048-sample display block.
        self.assertGreater(np.median(source.requests), 50000)

    def test_slow_demo_source_keeps_the_mute_with_hint_path(self):
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        source, _spoken, realtime = self._run(cost=0.6e-6)  # 1.44x real time
        self.assertFalse(realtime)
        self.assertEqual(source.requests[-1], DISPLAY_BLOCK)
        self.assertTrue(self.win._realtime_hint_shown)
        self.assertIn("demo device", self.win._message_label.text())


# ---------------------------------------------------------------------------
# 6. Radio tuner follows any frequency
# ---------------------------------------------------------------------------


class TestRadioTunerFollows(_WindowTestCase):
    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")

    def tearDown(self):
        if HAS_PYQT6 and getattr(self.win, "_radio_tuner", None) is not None:
            self.win._radio_tuner.close()
        super().tearDown()

    def test_off_band_frequencies_reach_the_tuner(self):
        self.win.set_frequency(433.92e6)
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.assertTrue(tuner.is_off_band())
        emitted = []
        tuner.frequency_changed.connect(lambda *a: emitted.append(a))
        self.win.set_frequency(98.5e6)
        self.assertFalse(tuner.is_off_band())
        self.assertAlmostEqual(tuner.get_frequency(), 98.5e6)
        self.win.set_frequency(446.0e6)
        self.assertTrue(tuner.is_off_band())
        # Back to exactly the tuner's station: it leaves off-band too.
        self.win.set_frequency(98.5e6)
        self.assertFalse(tuner.is_off_band())
        self.assertEqual(emitted, [])


# ---------------------------------------------------------------------------
# 7. Default frequency
# ---------------------------------------------------------------------------


class TestDefaultFrequency(_WindowTestCase):
    def test_window_starts_on_the_default(self):
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        self.assertEqual(DEFAULT_FREQUENCY_HZ, 100.1e6)
        self.assertAlmostEqual(self.freq(), DEFAULT_FREQUENCY_HZ)
        self.assertEqual(self.win._freq_label.text(), "100.100 MHz")
        # Broadcast FM: wideband deviation by default.
        self.assertEqual(self.win._control_panel.get_fm_deviation_text(), "75 kHz")

    def test_launchers_share_the_default(self):
        from sdr_module import cli
        from sdr_module.gui import __main__ as gui_main
        from sdr_module.gui import app
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        self.assertEqual(app._LAUNCH_DEFAULTS["frequency"], DEFAULT_FREQUENCY_HZ)
        with mock.patch.object(sys, "argv", ["sdr-gui"]):
            args = gui_main.parse_args()
        self.assertEqual(args.frequency, DEFAULT_FREQUENCY_HZ)
        gui_args = cli.create_parser().parse_args(["gui"])
        self.assertEqual(gui_args.frequency, DEFAULT_FREQUENCY_HZ)
        self.assertEqual(gui_args.sample_rate, app._LAUNCH_DEFAULTS["sample_rate"])

    def test_default_is_not_applied_over_the_saved_frequency(self):
        from sdr_module.gui.app import SDRApplication
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        launcher = SDRApplication(args=["sdr", "--demo"])
        self.assertFalse(
            launcher._should_apply({"frequency": DEFAULT_FREQUENCY_HZ}, "frequency")
        )
        self.assertTrue(launcher._should_apply({"frequency": 100e6}, "frequency"))
        typed = SDRApplication(args=["sdr", "-f", "100.1e6"])
        self.assertTrue(
            typed._should_apply({"frequency": DEFAULT_FREQUENCY_HZ}, "frequency")
        )


# ---------------------------------------------------------------------------
# 8. Copy
# ---------------------------------------------------------------------------


class TestCopy(_WindowTestCase):
    def test_receiving_wording(self):
        self.assertEqual(self.win._start_action.statusTip(), "Start or stop receiving")
        state_tip = self.win._info_panel._values["state"].toolTip()
        self.assertEqual(state_tip, "Whether the receiver is running.")

    def test_rds_is_not_advertised(self):
        tabs = self.win._right_tabs
        index = tabs.indexOf(self.win._panel_pages["decoder"])
        self.assertNotIn("RDS", tabs.tabToolTip(index))
        shown = []
        with mock.patch.object(
            QMessageBox, "about", side_effect=lambda *a: shown.append(a[2])
        ):
            self.win._show_about()
        self.assertNotIn("RDS", shown[0])
        self.assertIn("AX.25/APRS", shown[0])

    def test_quick_tips_use_keycaps_and_the_wizard_wording(self):
        from sdr_module.gui import first_run_wizard
        from sdr_module.gui.main_window import InfoPanel

        labels = self.win._info_panel.findChildren(type(self.win._freq_label))
        caps = [lb for lb in labels if lb.property("role") == "keycap"]
        self.assertEqual([c.text() for c in caps], [k for k, _t in InfoPanel.TIPS])
        self.assertFalse([lb for lb in labels if lb.property("role") == "badge"])
        wizard_tips = dict(first_run_wizard._TIPS)
        for keys, text in InfoPanel.TIPS:
            if keys in wizard_tips:
                self.assertEqual(text, wizard_tips[keys])


# ---------------------------------------------------------------------------
# 9. First-run wizard
# ---------------------------------------------------------------------------


class TestWizardDeviceName(_WindowTestCase):
    def test_wizard_names_the_found_radio_and_connects_after_tuning(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        seen = {}
        opened = []

        def fake_exec(wiz):
            seen["name"] = wiz._device_name
            wiz._band.setCurrentIndex(
                next(
                    i
                    for i in range(wiz._band.count())
                    if "NOAA" in wiz._band.itemText(i)
                )
            )
            wiz._on_accept()
            return wiz.result()

        self.win._scan_hardware = lambda: [("RTLSDRDevice:1#1", "RTL-SDR Blog V4")]
        with (
            mock.patch.object(FirstRunWizard, "exec", fake_exec),
            mock.patch.object(
                self.win,
                "_show_device_dialog",
                lambda start_after=False: opened.append((start_after, self.freq())),
            ),
        ):
            self.win._run_first_run_wizard()
        self.assertEqual(seen["name"], "RTL-SDR Blog V4")
        # Connect opens once the starting band is tuned.
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0][0])
        self.assertAlmostEqual(opened[0][1], 162.55e6)

    def test_no_hardware_passes_no_name(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        seen = {}

        def fake_exec(wiz):
            seen["name"] = wiz._device_name
            return 0

        self.win._scan_hardware = lambda: []
        with mock.patch.object(FirstRunWizard, "exec", fake_exec):
            self.win._run_first_run_wizard()
        self.assertEqual(seen["name"], "")


# ---------------------------------------------------------------------------
# 10. Closing while recording
# ---------------------------------------------------------------------------


class TestCloseWhileRecording(_WindowTestCase):
    demo = True

    def _record_live(self, frames=2):
        self.win._record_button.click()
        for _ in range(frames):
            self.win._update_display()
        self.assertTrue(self.win._recording)

    def _use_real_prompt(self, answer):
        from sdr_module.gui.main_window import SDRMainWindow

        del self.win._confirm_discard_recording  # the harness stub
        texts = []
        patcher = mock.patch.object(
            QMessageBox, "exec", _answer_message_box(answer, texts)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(
            setattr, self.win, "_confirm_discard_recording", lambda *_a: True
        )
        return SDRMainWindow._confirm_discard_recording, texts

    def test_cancel_keeps_recording_and_the_window(self):
        self._record_live()
        _confirm, texts = self._use_real_prompt("Cancel")
        self.win.close()
        self.assertTrue(self.win.isVisible())
        self.assertTrue(self.win._recording)
        self.assertTrue(self.win._record_button.isChecked())
        self.assertTrue(self.win._control_panel._record_btn.isChecked())
        self.assertIn("in progress", texts[0])
        before = sum(len(b) for b in self.win._samples_buffer)
        self.win._update_display()  # still capturing
        self.assertGreater(sum(len(b) for b in self.win._samples_buffer), before)

    def test_dont_save_stops_and_closes(self):
        self._record_live()
        self._use_real_prompt("Don't Save")
        self.win.close()
        self.assertFalse(self.win.isVisible())
        self.assertFalse(self.win._recording)
        self.assertEqual(self.win._samples_buffer, [])

    def test_save_stops_first_then_writes_everything(self):
        from sdr_module.dsp.recording import load_iq_file

        self._record_live()
        folder = tempfile.mkdtemp(prefix="sdr-round3-")
        path = os.path.join(folder, "take.cf32")
        states = []

        def fake_save(*_args):
            states.append(self.win._recording)
            return path, ""

        self._use_real_prompt("Save...")
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            side_effect=fake_save,
        ):
            self.win.close()
        self.assertEqual(states, [False])  # stopped before the file dialog
        self.assertFalse(self.win.isVisible())
        samples, _meta = load_iq_file(path)
        self.assertEqual(len(samples), sum(len(b) for b in self.win._samples_buffer))

    def test_armed_recording_with_nothing_to_lose_just_closes(self):
        self.win._stop_acquisition()
        self.win._record_button.click()  # armed, nothing captured
        _confirm, texts = self._use_real_prompt("Cancel")
        self.win.close()
        self.assertEqual(texts, [])  # no prompt
        self.assertFalse(self.win.isVisible())
        self.assertFalse(self.win._recording)


# ---------------------------------------------------------------------------
# 11. AudioSink
# ---------------------------------------------------------------------------


class TestAudioSink(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import audio_sink

        if not audio_sink.HAS_QT_AUDIO:
            self.skipTest("QtMultimedia not installed")
        self.mod = audio_sink

    def _device(self, null):
        device = mock.Mock()
        device.isNull.return_value = null
        return device

    def test_no_output_device_is_a_failure(self):
        sink = self.mod.AudioSink()
        with mock.patch.object(
            self.mod.QMediaDevices,
            "defaultAudioOutput",
            return_value=self._device(True),
        ):
            self.assertFalse(sink.start(48000))
        self.assertFalse(sink.is_open)

    def test_sink_that_does_not_open_is_a_failure(self):
        sink = self.mod.AudioSink()
        qsink = mock.Mock()
        qsink.start.return_value = None
        qsink.error.return_value = self.mod.QAudio.Error.OpenError
        with (
            mock.patch.object(
                self.mod.QMediaDevices,
                "defaultAudioOutput",
                return_value=self._device(False),
            ),
            mock.patch.object(self.mod, "QAudioSink", return_value=qsink),
        ):
            self.assertFalse(sink.start(48000))
        self.assertFalse(sink.is_open)
        sink.write(np.zeros(16, dtype=np.float32))  # a no-op, no error

    def test_error_state_is_a_failure_even_with_a_stream(self):
        sink = self.mod.AudioSink()
        qsink = mock.Mock()
        qsink.start.return_value = mock.Mock()
        qsink.error.return_value = self.mod.QAudio.Error.OpenError
        with (
            mock.patch.object(
                self.mod.QMediaDevices,
                "defaultAudioOutput",
                return_value=self._device(False),
            ),
            mock.patch.object(self.mod, "QAudioSink", return_value=qsink),
        ):
            self.assertFalse(sink.start(48000))
        self.assertFalse(sink.is_open)

    def test_open_output(self):
        sink = self.mod.AudioSink()
        stream = mock.Mock()
        qsink = mock.Mock()
        qsink.start.return_value = stream
        qsink.error.return_value = self.mod.QAudio.Error.NoError
        with (
            mock.patch.object(
                self.mod.QMediaDevices,
                "defaultAudioOutput",
                return_value=self._device(False),
            ),
            mock.patch.object(self.mod, "QAudioSink", return_value=qsink),
        ):
            self.assertTrue(sink.start(48000))
            self.assertTrue(sink.is_open)
            self.assertTrue(sink.start(48000))  # already open
        sink.write(np.zeros(16, dtype=np.float32))
        stream.write.assert_called_once()
        sink.stop()
        self.assertFalse(sink.is_open)


class TestAudioFailureInTheWindow(_WindowTestCase):
    def test_sink_failure_turns_audio_off_with_a_reason(self):
        with (
            mock.patch.object(
                type(self.win), "_audio_output_problem", return_value=None
            ),
            mock.patch.object(self.win._audio, "start", return_value=False),
        ):
            self.win._set_demod_mode("FM")
            self.win._audio_action.setChecked(True)
        self.assertFalse(self.win._audio_enabled)
        self.assertIn("could not be opened", self.win._message_label.text())

    def test_window_trusts_the_sink_result(self):
        with (
            mock.patch.object(
                type(self.win), "_audio_output_problem", return_value=None
            ),
            mock.patch.object(self.win._audio, "start", return_value=True),
        ):
            self.win._set_demod_mode("FM")
            self.win._audio_action.setChecked(True)
        self.assertTrue(self.win._audio_enabled)


# ---------------------------------------------------------------------------
# 12. Launcher options
# ---------------------------------------------------------------------------


class TestLauncherOptions(unittest.TestCase):
    def test_gui_main_sample_rate_help(self):
        from sdr_module.gui import __main__ as gui_main

        out = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["sdr-gui", "--help"]),
            redirect_stdout(out),
            self.assertRaises(SystemExit),
        ):
            gui_main.parse_args()
        text = " ".join(out.getvalue().split())
        self.assertIn("Sample rate in Hz for the demo device", text)
        self.assertIn("also preselected in Device > Connect", text)
        self.assertIn("100.1 MHz", text)

    def test_sdr_scan_gui_passes_the_sample_rate(self):
        from sdr_module import cli

        captured = {}

        class FakeApp:
            def is_available(self):
                return True

            def run(self, settings):
                captured.update(settings)
                return 0

        args = cli.create_parser().parse_args(["gui", "--demo", "-s", "1.024e6"])
        self.assertIsInstance(args, argparse.Namespace)
        with mock.patch("sdr_module.gui.app.SDRApplication", FakeApp):
            self.assertEqual(cli.cmd_gui(args), 0)
        self.assertEqual(captured["sample_rate"], 1.024e6)
        self.assertTrue(captured["demo_mode"])


if __name__ == "__main__":
    unittest.main()
