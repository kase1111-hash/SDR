#!/usr/bin/env python3
"""
The tuning model: a VFO inside the span the receiver captures (CONTRACTS
4.1), and tuning that a device refuses (fg:FG-B02, tl:TOOLS-05, mc:MC-U17).

The LO is the device's center (the middle of the plots); the tuned
frequency (the VFO) is what is demodulated, measured, bookmarked and shown
by FREQ. Tuning inside the usable span moves only the VFO; leaving it
recenters the LO; Radio > Keep Tuned Frequency Centered retunes the LO for
every tune (the old behaviour).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_gui_tuning.py``
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


def _flush_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _analytic(x: np.ndarray) -> np.ndarray:
    """The analytic signal of a real array (its envelope is the magnitude)."""
    n = len(x)
    gain = np.zeros(n)
    gain[0] = 1.0
    gain[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        gain[n // 2] = 1.0
    return np.fft.ifft(np.fft.fft(x) * gain)


# ---------------------------------------------------------------------------
# Real drivers on fake USB backends (their own range checks run)
# ---------------------------------------------------------------------------


class _FakeRtlBackend:
    """Just enough of pyrtlsdr's RtlSdr for the driver's tuning calls."""

    def __init__(self):
        self.center_freq = 100e6
        self.sample_rate = 2.4e6
        self.gain = "auto"

    def set_direct_sampling(self, mode):
        self.direct_sampling = mode

    def close(self):
        pass


class _FakeHackRFBackend:
    """Every pyhackrf_* call succeeds (and is recorded)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("pyhackrf_"):
            return lambda *args: self.calls.append((name, args)) or 0
        raise AttributeError(name)


def _rtlsdr():
    from sdr_module.devices.base import DeviceInfo
    from sdr_module.devices.rtlsdr import RTLSDRDevice

    dev = RTLSDRDevice()
    dev._device = _FakeRtlBackend()
    dev._is_open = True
    dev._state.frequency = 100e6
    dev._state.sample_rate = 2.4e6
    dev._info = DeviceInfo("Generic RTL2832U OEM", "00000001", "Realtek", "RTL")
    return dev


def _hackrf():
    from sdr_module.devices.base import DeviceInfo
    from sdr_module.devices.hackrf import HackRFDevice

    dev = HackRFDevice()
    dev._device = _FakeHackRFBackend()
    dev._is_open = True
    dev._state.frequency = 100e6
    dev._state.sample_rate = 8e6
    dev._info = DeviceInfo("HackRF One", "0000", "Great Scott Gadgets", "HackRF")
    return dev


class _TuningCase(_WindowTestCase):
    def lo(self):
        return self.win._lo_frequency()

    def connect(self, dev):
        """What Device > Connect... does with the device it opened."""
        self.win._device = dev
        self.win._demo_mode = False
        note = self.win._apply_controls_to_device()
        self.win._refresh_state_ui()
        return note

    def errors(self):
        label = self.win._message_label
        return label.text() if label.property("tone") == "danger" else ""


# ---------------------------------------------------------------------------
# The VFO inside the span
# ---------------------------------------------------------------------------


class TestVfoInsideTheSpan(_TuningCase):
    demo = True

    def test_starts_centered(self):
        self.assertEqual(self.freq(), 100.1e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.assertEqual(self.win._vfo_offset_hz(), 0.0)
        self.assertFalse(self.win._keep_centered_action.isChecked())

    def test_tuning_inside_the_span_moves_only_the_vfo(self):
        device = self.win._device
        self.assertTrue(self.win.set_frequency(100.5e6))
        self.assertEqual(self.freq(), 100.5e6)
        self.assertEqual(self.win._current_frequency(), 100.5e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.assertEqual(self.win._vfo_offset_hz(), 400e3)
        # The device (the LO) and the plots' axes stay put ...
        self.assertEqual(device.frequency, 100.1e6)
        for plot in (self.win._spectrum, self.win._waterfall):
            self.assertEqual(plot._center_freq, 100.1e6)
            # ... while the channel and the marker move to the VFO.
            self.assertEqual(plot.tuned_frequency(), 100.5e6)
            self.assertEqual(plot.passband(), (300e3, 500e3))  # 200 kHz FM
        self.assertEqual(self.win._freq_label.text(), "100.500 MHz")

    def test_leaving_the_usable_span_recenters_the_lo(self):
        device = self.win._device
        # 200 kHz FM may reach 0.4 x 2.4 MHz = 960 kHz from the LO.
        self.assertTrue(self.win.set_frequency(100.1e6 + 860e3))
        self.assertEqual(self.lo(), 100.1e6)
        self.assertTrue(self.win.set_frequency(100.1e6 + 870e3))
        self.assertEqual(self.lo(), 100.97e6)
        self.assertEqual(device.frequency, 100.97e6)
        self.assertEqual(self.win._vfo_offset_hz(), 0.0)
        for plot in (self.win._spectrum, self.win._waterfall):
            self.assertEqual(plot._center_freq, 100.97e6)
            self.assertEqual(plot.passband(), (-100e3, 100e3))

    def test_click_wheel_and_keys_tune_the_vfo(self):
        self.win._spectrum.frequency_clicked.emit(99.5e6)
        self.assertEqual((self.freq(), self.lo()), (99.5e6, 100.1e6))
        self.win._waterfall.frequency_clicked.emit(100.7e6)
        self.assertEqual((self.freq(), self.lo()), (100.7e6, 100.1e6))
        self.win._spectrum.tune_steps.emit(-2)  # 100 kHz steps for FM
        self.assertEqual((self.freq(), self.lo()), (100.5e6, 100.1e6))
        self.win._spectrum.setFocus()
        QApplication.processEvents()
        QTest.keyClick(self.win._spectrum, Qt.Key.Key_Left)
        self.assertEqual((self.freq(), self.lo()), (100.4e6, 100.1e6))

    def test_a_channel_too_wide_for_its_place_recenters(self):
        self.win.set_frequency(100.6e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.win._set_demod_mode("None (I/Q)", None, "2.4 MHz")
        self.assertEqual(self.lo(), 100.6e6)
        self.assertEqual(self.win._device.frequency, 100.6e6)
        self.assertEqual(self.freq(), 100.6e6)

    def test_a_smaller_span_recenters(self):
        self.win.set_frequency(100.9e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.win.set_sample_rate(1.0e6)  # +/-400 kHz usable: 800 kHz is out
        self.assertEqual(self.lo(), 100.9e6)
        self.assertEqual(self.freq(), 100.9e6)

    def test_the_receiver_listens_at_the_vfo(self):
        """Level, S-meter and audio all get the channel at the VFO: the
        demo's 99.5 MHz station, 600 kHz below the LO."""
        fed = []
        meter = getattr(self.win, "_signal_meter_panel", None)
        if meter is not None:
            # (The window also passes the channel's noise level.)
            meter.update_samples = lambda channel, *_noise: fed.append(
                np.asarray(channel)
            )

        def listen(freq):
            fed.clear()
            self.win.set_frequency(freq)
            levels = []
            for _ in range(6):
                self.win._update_display()
                levels.append(self.win._last_peak_db)
            return max(levels[2:])

        station = listen(99.5e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.assertEqual(self.win._rx_chain.offset_hz, -600e3)
        station_power = float(np.mean(np.abs(np.concatenate(fed)) ** 2)) if fed else 0
        empty = listen(100.45e6)  # 100.35-100.55 MHz: nothing on the air
        self.assertEqual(self.lo(), 100.1e6)
        empty_power = float(np.mean(np.abs(np.concatenate(fed)) ** 2)) if fed else 0
        self.assertGreater(station, -45.0)  # the station (-34 dBFS peak)
        self.assertLess(empty, -60.0)  # nothing there
        if meter is not None:
            self.assertGreater(station_power, 30 * empty_power)

    def test_moving_the_vfo_keeps_the_filters(self):
        self.win._update_display()
        chain = self.win._rx_chain
        self.assertIsNotNone(chain)
        self.win.set_frequency(100.3e6)
        self.win._update_display()
        self.assertIs(self.win._rx_chain, chain)
        self.assertEqual(chain.offset_hz, 200e3)

    def test_recordings_are_labelled_with_the_lo(self):
        self.win.set_frequency(100.5e6)
        self.win._toggle_recording(True)
        self.win._update_display()
        self.assertEqual(self.win._buffer_meta[0], 100.1e6)
        # Moving the VFO doesn't change the samples' center: no warning.
        self.win.set_frequency(100.6e6)
        self.win._update_display()
        self.assertFalse(self.win._retune_warned)
        # Moving the LO is refused while it records (a take holds one
        # center): the samples stay labelled right.
        self.assertFalse(self.win.set_frequency(145e6))
        self.win._update_display()
        self.assertEqual(self.win._lo_frequency(), 100.1e6)
        self.assertEqual(self.win._buffer_meta[0], 100.1e6)
        self.win._toggle_recording(False)
        self.win._discard_recording()

    def test_info_tab_shows_tuned_and_center(self):
        self.win.set_frequency(100.5e6)
        self.win._refresh_info_panel()
        info = self.win._info_panel
        self.assertEqual(info.value("frequency"), "100.500 MHz")
        self.assertEqual(info.value("lo"), "100.100 MHz")

    def test_panels_use_the_tuned_frequency(self):
        self.win.set_frequency(100.5e6)
        self.assertEqual(self.win._bookmarks_panel._tuned_hz, 100.5e6)
        self.assertEqual(self.win._decoder_panel._tuned_hz, 100.5e6)
        before = len(self.store.bookmarks)
        self.win._bookmark_current_frequency()
        self.assertEqual(len(self.store.bookmarks), before + 1)
        self.assertAlmostEqual(self.store.bookmarks[-1]["freq_hz"], 100.5e6)

    def test_radio_tuner_follows_the_tuned_frequency(self):
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.win.set_frequency(99.5e6)
        self.assertAlmostEqual(tuner._frequency, 99.5e6)
        tuner.close()

    def test_lo_is_saved_and_restored(self):
        self.win.set_frequency(100.5e6)
        self.win._persist_state()
        self.assertEqual(self.store.values["tuning/lo_hz"], 100.1e6)
        self.assertEqual(self.store.values["frequency_hz"], 100.5e6)
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertEqual(other._lo_frequency(), 100.1e6)
        self.assertEqual(other._current_frequency(), 100.5e6)
        self.assertEqual(other._spectrum._center_freq, 100.1e6)

    def test_contract_names(self):
        """The names other parts of the window use (CONTRACTS 4.1)."""
        self.win.set_frequency(99.9e6)
        self.assertEqual(self.win._current_frequency(), 99.9e6)
        self.assertEqual(self.win._lo_frequency(), 100.1e6)
        self.assertAlmostEqual(self.win._vfo_offset_hz(), -200e3)
        chain = self.win._receiver_chain()
        self.assertAlmostEqual(chain.offset_hz, -200e3)
        chain.set_offset(5e3)
        self.assertEqual(chain.offset_hz, 5e3)


class TestKeepTunedFrequencyCentered(_TuningCase):
    demo = True

    def test_the_option_restores_the_old_behaviour(self):
        device = self.win._device
        action = self.win._keep_centered_action
        self.assertTrue(action.isCheckable())
        self.assertIn(action, self._radio_menu_actions())
        self.win.set_frequency(100.5e6)
        self.assertEqual(self.lo(), 100.1e6)
        action.setChecked(True)  # centers right away
        self.assertEqual((self.lo(), device.frequency), (100.5e6, 100.5e6))
        self.assertIs(self.store.values["tuning/keep_centered"], True)
        for freq in (100.6e6, 100.65e6, 99.0e6):
            self.win.set_frequency(freq)  # every tune retunes the device
            self.assertEqual((self.freq(), self.lo(), device.frequency), (freq,) * 3)
        action.setChecked(False)
        self.assertIs(self.store.values["tuning/keep_centered"], False)
        self.win.set_frequency(99.2e6)
        self.assertEqual(self.lo(), 99.0e6)

    def test_the_option_is_remembered(self):
        self.store.values["tuning/keep_centered"] = True
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertTrue(other._keep_centered_action.isChecked())

    def _radio_menu_actions(self):
        for top in self.win.menuBar().actions():
            if top.text().replace("&", "") == "Radio":
                return top.menu().actions()
        return []


class TestNoClickWhenTheVfoMoves(unittest.TestCase):
    """Moving the channel within the span keeps the NCO's phase and the
    filters' state: no spike in the audio (a phase jump would make one)."""

    def test_fm_audio_has_no_spike_when_retuned(self):
        from sdr_module.gui import main_window as mw

        rate, block = 2.4e6, 80_000
        chain = mw._ReceiverChain(rate, "FM", 25e3, offset_hz=300e3)
        n = 12 * block
        tone = np.exp(2j * np.pi * 300e3 * np.arange(n) / rate).astype(np.complex64)
        audio, channel = [], []
        for i in range(12):
            if i == 6:
                chain.set_offset(300.5e3)  # 500 Hz: a small, steady step
            part = tone[i * block : (i + 1) * block]
            channel.append(chain.channelize(part))
            audio.append(chain.demodulate(channel[-1], 5e3))
        audio = np.concatenate(audio)
        settled = audio[len(audio) // 10 :]
        # A 500 Hz step at 5 kHz deviation is at most 0.1 of full scale
        # (and the audio's DC blocker takes even that away); a phase jump in
        # the mixer would spike toward full scale.
        self.assertLess(float(np.max(np.abs(settled))), 0.2)

        # The step really moved the channel: the tone now sits 500 Hz below
        # the VFO (measured on the channel, as the audio is DC-blocked).
        def mean_hz(blocks):
            x = np.concatenate(blocks)
            step = np.angle(x[1:] * np.conj(x[:-1]))
            return float(np.mean(step)) * chain.channel_rate / (2 * np.pi)

        self.assertAlmostEqual(
            mean_hz(channel[8:]) - mean_hz(channel[3:6]), -500.0, delta=20.0
        )

    def test_fine_tuning_across_the_lo_does_not_click(self):
        """USB in 10 Hz steps through offset 0 (the LO itself): the beat
        note keeps its level at every step. The NCO used to skip mixing at
        0 Hz and so jumped the phase there: the tone dipped to a fifth."""
        from sdr_module.gui import main_window as mw

        rate, block = 2.4e6, 80_000
        offsets = (17.0, 7.0, 0.0, -13.0, -23.0)
        chain = mw._ReceiverChain(rate, "USB", 3e3, offset_hz=offsets[0])
        audio, marks, start = [], [], 0
        for i in range(3 * len(offsets)):
            if i % 3 == 0:
                chain.set_offset(offsets[i // 3])
                marks.append(sum(len(a) for a in audio))
            t = (start + np.arange(block)) / rate
            start += block
            carrier = np.exp(2j * np.pi * 1000.0 * t).astype(np.complex64)
            audio.append(chain.demodulate(chain.channelize(carrier), 5e3))
        envelope = np.abs(_analytic(np.concatenate(audio)))
        for offset, mark in zip(offsets[1:], marks[1:], strict=True):
            level = float(np.median(envelope[mark - 2000 : mark - 200]))
            dip = float(envelope[mark : mark + 1200].min())
            self.assertGreater(dip, 0.8 * level, f"step to {offset:+g} Hz")


# ---------------------------------------------------------------------------
# Tuning a device can't do
# ---------------------------------------------------------------------------


class TestRefusedTuning(_TuningCase):
    def test_out_of_range_is_refused_and_explained(self):
        dev = _rtlsdr()
        self.connect(dev)
        self.assertEqual((self.freq(), self.lo()), (100.1e6, 100.1e6))
        self.assertEqual(dev.state.frequency, 100.1e6)
        self.assertFalse(self.win.set_frequency(2.4e9))
        # Field, readout, axes and the device all stay on 100.1 MHz ...
        self.assertEqual(self.freq(), 100.1e6)
        self.assertEqual(self.win._freq_label.text(), "100.100 MHz")
        self.assertEqual(self.win._spectrum._center_freq, 100.1e6)
        self.assertEqual(self.win._waterfall._center_freq, 100.1e6)
        self.assertEqual(dev.state.frequency, 100.1e6)
        self.assertEqual(dev._device.center_freq, 100.1e6)
        # ... and the status bar says why.
        self.assertEqual(
            self.errors(), "2400.000 MHz is out of range: RTL-SDR tunes 0.5–1766 MHz."
        )

    def test_typed_out_of_range_frequency_is_refused(self):
        dev = _rtlsdr()
        self.connect(dev)
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "2400")
        self.assertEqual(field.text(), "2400")  # the digits are kept
        self.assertEqual(field.property("tone"), "danger")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertIn("RTL-SDR tunes 0.5–1766 MHz", self.errors())
        self.assertEqual(self.freq(), 100.1e6)
        self.assertEqual(field.text(), "100.100")
        self.assertFalse(field.property("tone"))  # the red frame is gone
        self.assertEqual(dev.state.frequency, 100.1e6)
        # The field keeps the focus, to type again.
        self.assertTrue(field.hasFocus())

    def test_the_frequency_field_takes_the_device_range(self):
        entry = self.win._control_panel._freq_input
        self.assertEqual(entry.tuning_range(), (1e3, 6e9))
        self.assertIn("Tunes 0.001–6000 MHz", entry._freq_input.toolTip())
        self.connect(_rtlsdr())
        self.assertEqual(entry.tuning_range(), (0.5e6, 1.766e9))
        self.assertIn("RTL-SDR tunes 0.5–1766 MHz", entry._freq_input.toolTip())
        self.win._disconnect_device()
        self.assertEqual(entry.tuning_range(), (1e3, 6e9))
        self.connect(_hackrf())
        self.assertEqual(entry.tuning_range(), (1e6, 6e9))
        self.assertIn("HackRF One tunes 1–6000 MHz", entry._freq_input.toolTip())
        self.win._device = None

    def test_a_driver_that_says_no_within_its_range(self):
        dev = _rtlsdr()
        self.connect(dev)
        self.win._keep_centered_action.setChecked(True)  # every tune asks it
        with mock.patch.object(type(dev), "set_frequency", return_value=False):
            self.assertFalse(self.win.set_frequency(433.92e6))
        self.assertEqual((self.freq(), self.lo()), (100.1e6, 100.1e6))
        self.assertIn("RTL-SDR didn't tune to 433.920 MHz", self.errors())
        self.assertIn("it stays on 100.100 MHz", self.errors())
        with mock.patch.object(type(dev), "set_frequency", side_effect=OSError("USB")):
            self.assertFalse(self.win.set_frequency(144e6))
        self.assertEqual(self.freq(), 100.1e6)

    def test_radio_tuner_am_on_a_hackrf_is_refused(self):
        dev = _hackrf()
        self.connect(dev)
        mode = self.win._control_panel._demod_combo.currentText()
        self.win._on_radio_frequency_changed(880e3, "AM")
        self.assertEqual(self.freq(), 100.1e6)
        self.assertEqual(dev.state.frequency, 100.1e6)
        self.assertEqual(
            self.errors(), "880 kHz is out of range: HackRF One tunes 1–6000 MHz."
        )
        self.assertNotIn("Tuned to", self.win._message_label.text())
        self.assertEqual(self.win._control_panel._demod_combo.currentText(), mode)
        self.win._device = None

    def test_a_refused_bookmark_is_not_announced(self):
        self.connect(_hackrf())
        self.win._on_bookmark_tune(198e3, "LW", "AM")
        self.assertIn("HackRF One tunes 1–6000 MHz", self.errors())
        self.assertNotEqual(self.win._control_panel._demod_combo.currentText(), "AM")
        self.win._device = None

    def test_a_refused_back_keeps_the_history_where_the_receiver_is(self):
        """Back to a frequency the dongle can't tune moved the history
        anyway: Forward then offered the frequency already tuned, and the
        next frequency visited dropped it from the history."""
        history = self.win._tuning_history
        self.assertTrue(self.win.set_frequency(2.4e9))  # no device yet
        self.win._commit_history_entry()
        self.assertTrue(self.win.set_frequency(100.1e6))
        self.win._commit_history_entry()
        self.connect(_rtlsdr())
        self.win._tune_back()
        self.assertEqual(self.freq(), 100.1e6)
        self.assertIn("RTL-SDR tunes 0.5–1766 MHz", self.errors())
        self.assertEqual(history.current().freq_hz, 100.1e6)
        self.assertTrue(self.win._back_action.isEnabled())
        self.assertFalse(self.win._forward_action.isEnabled())
        self.assertTrue(self.win.set_frequency(100.3e6))
        self.win._commit_history_entry()
        self.assertEqual(
            [e.freq_hz for e in history._history], [2.4e9, 100.1e6, 100.3e6]
        )
        self.win._device = None

    def test_connecting_with_the_frequency_out_of_its_range(self):
        self.assertTrue(self.win.set_frequency(2.4e9))  # no device: fine
        dev = _rtlsdr()
        note = self.connect(dev)
        self.assertEqual(self.freq(), 100e6)  # where the dongle is
        self.assertEqual(self.lo(), 100e6)
        self.assertEqual(self.win._spectrum._center_freq, 100e6)
        self.assertEqual(
            note,
            "Tuned to 100.000 MHz: 2400.000 MHz is out of its range "
            "(RTL-SDR tunes 0.5–1766 MHz).",
        )

    def test_connect_dialog_says_why_it_moved(self):
        from sdr_module.gui import device_dialog

        self.win.set_frequency(2.4e9)
        dongle = _rtlsdr()
        dongle._is_open = False

        def fake_open(dev):
            dongle._is_open = True
            return dongle

        def fake_exec(dialog):
            dialog._open_device = fake_open
            dialog._on_accept()
            return dialog.result()

        with mock.patch.object(device_dialog.DeviceDialog, "exec", fake_exec):
            self.assertTrue(self.win._show_device_dialog())
        _flush_deletes()
        text = self.win._message_label.text()
        self.assertTrue(text.startswith("Connected to Generic RTL2832U OEM."), text)
        self.assertIn("2400.000 MHz is out of its range", text)
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        self.win._device = None


class TestRefusalsAndTheErrorHistory(_TuningCase):
    """Tools > Error History is for faults: a frequency out of range is
    explained in the status bar, not logged; a device that says no is."""

    def test_out_of_range_is_explained_not_logged(self):
        self.connect(_rtlsdr())
        with self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"):
            self.assertFalse(self.win.set_frequency(2.4e9))
        self.assertIn("RTL-SDR tunes", self.errors())

    def test_a_device_saying_no_is_logged(self):
        dev = _rtlsdr()
        self.connect(dev)
        self.win._keep_centered_action.setChecked(True)
        with (
            mock.patch.object(type(dev), "set_frequency", return_value=False),
            self.assertLogs("sdr_module.gui.main_window", level="WARNING"),
        ):
            self.assertFalse(self.win.set_frequency(433.92e6))

    def test_a_damaged_saved_frequency_is_clamped_quietly(self):
        self.store.values["frequency_hz"] = 1e12
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertEqual(other._current_frequency(), 6e9)
        self.assertEqual(other._lo_frequency(), 6e9)
        self.assertNotEqual(other._message_label.property("tone"), "danger")


class TestTransmitWhereChecked(_TuningCase):
    def test_the_lo_moves_to_the_tuned_frequency_before_transmitting(self):
        """The HackRF sends at its center (the LO): it is put on the tuned
        frequency before the driver's own legality check, which checks the
        emission's bandwidth and mode there (the window's check, made
        first, is of that same frequency)."""
        import sdr_module.devices.hackrf as hackrf_mod

        dev = _hackrf()
        self.connect(dev)
        self.win.set_frequency(100.9e6)  # 800 kHz up: the VFO only (8 MS/s)
        self.assertEqual(self.lo(), 100.1e6)
        self.store.values["tx/confirm"] = False  # (no first-use question)
        window_checks, driver_checks = [], []

        def window_allowed(freq, bandwidth=0, mode=""):
            window_checks.append((freq, mode))
            return True, None

        def driver_allowed(freq, bandwidth=0, mode=""):
            driver_checks.append((freq, dev.state.frequency, bandwidth, mode))
            return False, "not here"

        with (
            mock.patch(
                "sdr_module.core.frequency_manager.is_tx_allowed", window_allowed
            ),
            mock.patch.object(hackrf_mod, "is_tx_allowed", driver_allowed),
        ):
            self.win._transmit_audio(np.zeros(16, dtype=np.complex64), "test")
            self.assertTrue(self.win._tx_station.engine.wait(5))
        self.assertTrue(window_checks)
        self.assertTrue(all(freq == 100.9e6 for freq, _mode in window_checks))
        self.assertEqual(driver_checks, [(100.9e6, 100.9e6, 12.5e3, "FM")])
        self.assertEqual(self.lo(), 100.9e6)
        self.assertIn("test not sent on 100.900 MHz: not here", self.errors())
        self.win._device = None


class TestLockedLo(_TuningCase):
    """While a recording plays back the LO is the file's center (T5 calls
    _lock_lo / _unlock_lo)."""

    demo = True

    def test_frequencies_outside_the_locked_span_are_refused(self):
        device = self.win._device
        self.win._lock_lo(100.1e6, 2.4e6, "the recording")
        self.assertTrue(self.win.set_frequency(100.6e6))
        self.assertEqual(self.lo(), 100.1e6)
        self.assertFalse(self.win.set_frequency(101.5e6))
        self.assertEqual(self.freq(), 100.6e6)
        self.assertEqual(
            self.errors(),
            "101.500 MHz is outside the recording, which covers 98.9–101.3 MHz.",
        )
        self.win._keep_centered_action.setChecked(True)  # can't move the LO
        self.assertEqual(self.lo(), 100.1e6)
        self.win._unlock_lo()
        self.assertEqual(self.lo(), 100.6e6)  # centered, as the option says
        self.assertEqual(device.frequency, 100.6e6)

    def test_locking_brings_a_vfo_outside_the_span_inside(self):
        self.win.set_frequency(100.9e6)
        self.assertEqual(self.lo(), 100.1e6)
        self.win._lock_lo(100.1e6, 1.0e6, "the recording")  # +/-400 kHz usable
        self.assertEqual((self.freq(), self.lo()), (100.1e6, 100.1e6))
        self.win._unlock_lo()


class TestNewWindowSettings(unittest.TestCase):
    """The settings fake is shared with the chrome tests: the tuning keys
    use only its typed getters."""

    def test_fake_settings_cover_the_tuning_keys(self):
        store = _FakeSettings()
        self.assertEqual(store.get_str("tuning/recent", ""), "")
        self.assertFalse(store.get_bool("tuning/keep_centered", False))
        self.assertEqual(store.get_float("tuning/step/wfm", 100e3), 100e3)


if __name__ == "__main__":
    unittest.main()
