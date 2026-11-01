"""Device > Second Receiver... (gui/second_receiver.py): a second receiver
window bound to another device (or a second demo device).

Covers the signal path (ReceiverDsp), the device list and the rule that the
two windows never open the same device, tuning (VFO inside the span), mode
and filter, squelch, whose audio plays (Listen), the shared bookmarks, I/Q
recording, the transmit monitor (with a HackRF on the fake backend sending
through the transmit engine), keyboard, persistence and layout.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

from sdr_module.devices.base import DeviceInfo  # noqa: E402
from sdr_module.devices.hackrf import HackRFDevice  # noqa: E402
from sdr_module.devices.rtlsdr import RTLSDRDevice  # noqa: E402

RATE = 2.4e6
BLOCK = 80_000  # about one display frame at 2.4 MS/s


def fm_signal(n, offset_hz=0.0, deviation=5e3, tone_hz=1e3, level=0.3, noise=1e-3):
    """An FM carrier ``offset_hz`` from the center, modulated by a tone,
    in a little noise (complex64)."""
    t = np.arange(n) / RATE
    phase = 2 * np.pi * offset_hz * t + (deviation / tone_hz) * np.sin(
        2 * np.pi * tone_hz * t
    )
    rng = np.random.default_rng(1)
    hiss = noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (level * np.exp(1j * phase) + hiss).astype(np.complex64)


def noise_only(n, sigma=1e-3, seed=2):
    rng = np.random.default_rng(seed)
    return (sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )


# ---------------------------------------------------------------------------
# Signal path (no window)
# ---------------------------------------------------------------------------


class TestReceiverDsp(unittest.TestCase):
    def setUp(self):
        from sdr_module.gui.second_receiver import ReceiverDsp

        self.dsp = ReceiverDsp()
        self.dsp.rate = RATE
        self.dsp.mode = "FM"
        self.dsp.bandwidth_hz = 25e3
        self.dsp.deviation_hz = 5e3

    def run_blocks(self, make, count=12, play=True):
        frames = []
        for i in range(count):
            frames.append(self.dsp.process(make(i), play))
        return frames

    def test_a_station_opens_the_squelch_and_plays(self):
        frames = self.run_blocks(lambda i: fm_signal(BLOCK), count=24)
        last = frames[-1]
        self.assertEqual(last.state, "open")
        self.assertGreater(last.level_db, -20.0)
        audio = np.concatenate([f.audio for f in frames[2:] if f.audio is not None])
        self.assertGreater(len(audio), 0.7 * 48000)  # (22 frames: ~0.73 s)
        # The 1 kHz tone comes out (FM at the set deviation: near full scale).
        spectrum = np.abs(np.fft.rfft(audio[-24000:] * np.hanning(24000)))
        peak_hz = np.argmax(spectrum) * 48000 / 24000
        self.assertAlmostEqual(peak_hz, 1000.0, delta=20.0)
        self.assertGreater(float(np.max(np.abs(audio[-24000:]))), 0.5)

    def test_noise_keeps_the_auto_squelch_closed(self):
        frames = self.run_blocks(lambda i: noise_only(BLOCK, seed=i))
        self.assertEqual(frames[-1].state, "closed")
        self.assertTrue(all(f.audio is None or len(f.audio) == 0 for f in frames[2:]))
        # The noise level is known, and the threshold sits above it.
        self.assertIsNotNone(frames[-1].noise_db)
        self.assertGreater(frames[-1].threshold_db, frames[-1].noise_db)

    def test_squelch_off_plays_noise_too(self):
        self.dsp.squelch.enabled = False
        frames = self.run_blocks(lambda i: noise_only(BLOCK, seed=i))
        self.assertEqual(frames[-1].state, "off")
        self.assertIsNotNone(frames[-1].audio)
        self.assertGreater(len(frames[-1].audio), 0)

    def test_not_playing_demodulates_nothing(self):
        frames = self.run_blocks(lambda i: fm_signal(BLOCK), play=False)
        self.assertEqual(frames[-1].state, "open")  # (the squelch still works)
        self.assertTrue(all(f.audio is None for f in frames))

    def test_iq_mode_has_no_audio(self):
        self.dsp.mode = "None (I/Q)"
        self.dsp.bandwidth_hz = 200e3
        frames = self.run_blocks(lambda i: fm_signal(BLOCK), count=4)
        self.assertEqual(frames[-1].state, "idle")
        self.assertTrue(all(f.audio is None for f in frames))

    def test_the_channel_is_at_the_vfo(self):
        """A station 300 kHz off the LO is heard with the VFO there, not at
        the LO."""
        make = lambda i: fm_signal(BLOCK, offset_hz=300e3)  # noqa: E731
        self.dsp.offset_hz = 0.0
        self.assertEqual(self.run_blocks(make)[-1].state, "closed")
        self.dsp.reset()
        self.dsp.offset_hz = 300e3
        self.assertEqual(self.run_blocks(make)[-1].state, "open")

    def test_stopping_fades_out(self):
        self.run_blocks(lambda i: fm_signal(BLOCK))
        self.assertTrue(self.dsp.playing)
        tail = self.dsp.release()
        self.assertGreater(len(tail), 0)
        self.assertLess(abs(float(tail[-1])), 0.05)
        self.assertFalse(self.dsp.playing)

    def test_spectrum_is_in_dbfs(self):
        tone = (0.5 * np.exp(2j * np.pi * 0.1 * np.arange(2048))).astype(np.complex64)
        power = self.dsp.spectrum(tone)
        self.assertEqual(len(power), 2048)
        self.assertAlmostEqual(float(np.max(power)), 20 * np.log10(0.5), delta=0.5)


# ---------------------------------------------------------------------------
# Fake hardware: an RTL-SDR driver with two dongles, no USB
# ---------------------------------------------------------------------------


class FakeRtl(RTLSDRDevice):
    """Two RTL-SDR dongles (index 0 and 1); a dongle opens only once."""

    open_indexes: set = set()

    @staticmethod
    def list_devices():
        return [
            DeviceInfo(
                name=f"RTL-SDR #{i}",
                serial=f"0000000{i + 1}",
                manufacturer="test",
                product="RTL2832U",
                index=i,
            )
            for i in range(2)
        ]

    def __init__(self):
        super().__init__()
        self.closed = False
        self.frequencies = []

    def open(self, index=0):
        if index in FakeRtl.open_indexes:
            return False  # in use (the real driver fails to claim it)
        FakeRtl.open_indexes.add(index)
        self._info = self.list_devices()[index]
        self._is_open = True
        self._state.sample_rate = 2.4e6
        self._state.frequency = 100e6
        return True

    def close(self):
        if self._info is not None:
            FakeRtl.open_indexes.discard(self._info.index)
        self._is_open = False
        self.closed = True

    def set_frequency(self, freq_hz):
        self._state.frequency = float(freq_hz)
        self.frequencies.append(float(freq_hz))
        return True

    def set_sample_rate(self, rate_hz):
        self._state.sample_rate = float(rate_hz)
        return True

    def set_gain(self, gain_db):
        self._state.gain = float(gain_db)
        return True

    def set_gain_mode(self, auto):
        self._state.gain_mode = "auto" if auto else "manual"
        return True

    def start_rx(self, callback=None):
        self._state.is_streaming = True
        return True

    def stop_rx(self):
        self._state.is_streaming = False
        return True

    def read_samples(self, num_samples, timeout=1.0):
        if not self._state.is_streaming:
            return None
        return noise_only(2048, seed=len(self.frequencies))


class FakeHackRF(HackRFDevice):
    """One HackRF One, no USB; records its baseband filter widths."""

    open_now = False

    @staticmethod
    def list_devices():
        return [
            DeviceInfo(
                name="HackRF One",
                serial="0000000000fake77",
                manufacturer="test",
                product="HackRF One",
                index=0,
            )
        ]

    def __init__(self):
        super().__init__()
        self.filters = []

    def open(self, index=0):
        if FakeHackRF.open_now:
            return False
        FakeHackRF.open_now = True
        self._info = self.list_devices()[0]
        self._is_open = True
        self._state.sample_rate = 10e6
        self._state.frequency = 100e6
        return True

    def close(self):
        FakeHackRF.open_now = False
        self._is_open = False

    def set_frequency(self, freq_hz):
        self._state.frequency = float(freq_hz)
        return True

    def set_sample_rate(self, rate_hz):
        self._state.sample_rate = float(rate_hz)
        return True

    def set_bandwidth(self, bw_hz):
        self.filters.append(float(bw_hz))
        return True

    def set_gain(self, gain_db):
        return True

    def set_gain_mode(self, auto):
        return not auto  # (no automatic gain)

    def start_rx(self, callback=None):
        self._state.is_streaming = True
        return True

    def stop_rx(self):
        self._state.is_streaming = False
        return True

    def read_samples(self, num_samples, timeout=1.0):
        return noise_only(2048) if self._state.is_streaming else None


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


class _FakeSink:
    """Stands in for the speaker: records what is written."""

    available = True

    def __init__(self):
        self.blocks = []
        self.open = False
        self.volume = None

    @property
    def is_open(self):
        return self.open

    def start(self, rate=48000):
        self.open = True
        return True

    def stop(self):
        self.open = False

    def set_volume(self, volume):
        self.volume = volume

    def set_muted(self, muted):
        pass

    def output_name(self):
        return "Fake Speakers"

    def write(self, samples):
        if self.open and samples is not None and len(samples):
            self.blocks.append(np.asarray(samples, dtype=np.float32))

    def samples(self):
        return np.concatenate(self.blocks) if self.blocks else np.zeros(0)


class _SecondCase(_WindowTestCase):
    """A main window plus its second receiver (no timers running)."""

    def setUp(self):
        super().setUp()
        FakeRtl.open_indexes = set()
        self.host = self.win._second_rx
        self.rx = self.host.show()
        self.rx._status_timer.stop()
        self.extra_windows = []
        self.settle()

    def tearDown(self):
        # Before the main window goes: a receiver window outliving it would
        # keep its timers running.
        for window in self.extra_windows:
            window.shutdown()
            window.deleteLater()
        rx = self.host.receiver if HAS_PYQT6 else None
        if rx is not None:
            rx.shutdown()
        super().tearDown()

    def extra(self, window):
        """A further receiver window, shut down before the main window."""
        self.extra_windows.append(window)
        return window

    def frames(self, count=8, rx=None):
        rx = rx or self.rx
        for _ in range(count):
            rx._frame()
            time.sleep(0.005)

    def feed(self, count=8, rx=None):
        """``count`` display frames of the device's samples (BLOCK each,
        however fast the test runs)."""
        rx = rx or self.rx
        for _ in range(count):
            rx._process(rx.device.read_samples(BLOCK))

    def sink(self):
        sink = _FakeSink()
        self.rx._audio = sink
        self.rx._audio_open = False
        return sink


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestOpening(_SecondCase):
    def test_device_menu_opens_it(self):
        action = self.host.action
        self.assertEqual(action.text(), "&Second Receiver...")
        self.assertTrue(action.toolTip())
        device_menu = next(
            a.menu()
            for a in self.win.menuBar().actions()
            if a.text().replace("&", "") == "Device"
        )
        self.assertIn(action, device_menu.actions())
        # A window of its own (it can go behind the main window), not a
        # dialog stuck on top of it.
        self.assertIsNone(self.rx.parent())
        self.assertTrue(self.rx.isWindow())
        self.rx.close()
        self.settle()
        self.assertFalse(self.rx.isVisible())
        action.trigger()
        self.settle()
        self.assertTrue(self.host.receiver.isVisible())
        self.assertIs(self.host.receiver, self.rx)  # (one window, reused)
        # The plots have the focus, so Space and the arrows work at once.
        self.assertTrue(self.rx._plots_focused())

    def test_starts_on_a_demo_device_of_its_own(self):
        rx = self.rx
        self.assertEqual(rx.source, None)
        self.assertTrue(rx.start())
        self.assertTrue(rx.running)
        from sdr_module.gui.device_dialog import MockDevice

        self.assertIsInstance(rx.device, MockDevice)
        self.assertIsNot(rx.device, self.win._device)
        self.assertEqual(
            rx.windowTitle(), "Second Receiver — Demo Device (simulated signals)"
        )
        self.assertEqual(rx._state_badge.text(), "RUNNING")
        self.frames()
        self.assertTrue(rx._spectrum._has_data)
        self.assertIsNotNone(rx._level_meter.level())

    def test_stop_closes_the_device(self):
        rx = self.rx
        rx.start()
        device = rx.device
        self.assertTrue(device.is_streaming)
        rx.stop()
        self.assertFalse(rx.running)
        self.assertIsNone(rx.device)
        self.assertFalse(device.is_streaming)
        self.assertEqual(rx._state_badge.text(), "STOPPED")
        self.assertIn("closed", rx.message())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDevices(_SecondCase):
    """The two receivers never open the same device."""

    def setUp(self):
        super().setUp()
        self.win._hardware_classes = [FakeRtl]

    def main_has(self, index):
        device = FakeRtl()
        self.assertTrue(device.open(index))
        self.win._device = device
        self.win._demo_mode = False
        self.addCleanup(lambda: FakeRtl.open_indexes.discard(index))
        return device

    def test_the_main_windows_device_is_in_use(self):
        from sdr_module.gui.second_receiver import list_sources

        self.main_has(0)
        sources = list_sources(self.win)
        self.assertEqual(
            [s.key for s in sources], ["demo", "rtlsdr:00000001", "rtlsdr:00000002"]
        )
        self.assertEqual(sources[1].busy, "the main window")
        self.assertEqual(sources[1].label, "RTL-SDR #0 (in use by the main window)")
        self.assertEqual(sources[2].busy, "")
        self.rx._refresh_sources()
        combo = self.rx._source_combo
        self.assertFalse(combo.model().item(1).isEnabled())
        self.assertTrue(combo.model().item(2).isEnabled())

    def test_a_device_held_by_a_playing_recording_is_in_use_too(self):
        from types import SimpleNamespace

        from sdr_module.gui.second_receiver import list_sources

        held = FakeRtl()
        held.open(1)
        self.addCleanup(lambda: FakeRtl.open_indexes.discard(1))
        self.win._playback._previous = SimpleNamespace(device=held)
        self.addCleanup(lambda: setattr(self.win._playback, "_previous", None))
        busy = [s.name for s in list_sources(self.win) if s.busy]
        self.assertIn("RTL-SDR #1", busy)

    def test_it_opens_another_device_and_the_main_window_cant_take_it(self):
        main_device = self.main_has(0)
        rx = self.rx
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(2)  # RTL-SDR #1
        self.assertTrue(rx.start())
        self.assertEqual(rx.source.name, "RTL-SDR #1")
        self.assertTrue(rx.holds("rtlsdr", 1))
        problem = self.host.entry_problem({"type": "rtlsdr", "index": 1})
        self.assertIn("in use by the second receiver", problem)
        self.assertEqual(self.host.entry_problem({"type": "rtlsdr", "index": 0}), "")
        self.assertEqual(self.host.entry_problem({"type": "demo", "index": 0}), "")

        # Device > Connect... choosing it: refused before the main window
        # lets go of its own device.
        import sdr_module.gui.device_dialog as device_dialog

        results = {}

        class FakeDialog:
            _device_class = staticmethod(
                lambda kind: FakeRtl if kind == "rtlsdr" else None
            )

            def __init__(self, parent=None):
                pass

            def _open_device(self, entry):  # replaced by the window
                raise AssertionError("not wrapped")

            def exec(self):
                try:
                    self._open_device({"type": "rtlsdr", "index": 1})
                except Exception as e:
                    results["error"] = str(e)
                return 0

            def get_selected_device(self):
                return None

            def deleteLater(self):
                pass

        original = device_dialog.DeviceDialog
        device_dialog.DeviceDialog = FakeDialog
        self.addCleanup(lambda: setattr(device_dialog, "DeviceDialog", original))
        self.assertFalse(self.win._show_device_dialog())
        self.assertIn("RTL-SDR #1 is in use by the second receiver", results["error"])
        self.assertIs(self.win._device, main_device)
        self.assertFalse(main_device.closed)
        # Stopped, it holds nothing: the main window may have it.
        rx.stop()
        self.assertEqual(self.host.entry_problem({"type": "rtlsdr", "index": 1}), "")
        self.assertNotIn(1, FakeRtl.open_indexes)

    def test_opening_a_device_the_main_window_took_meanwhile_is_refused(self):
        self.rx._refresh_sources()
        self.rx._source_combo.setCurrentIndex(1)  # RTL-SDR #0, free so far
        self.main_has(0)
        self.assertFalse(self.rx.start())
        self.assertIn("in use by the main window", self.rx.message())
        self.assertFalse(self.rx.running)

    def test_switching_devices_while_running(self):
        rx = self.rx
        rx._refresh_sources()
        rx.start()  # demo
        rx._source_combo.setCurrentIndex(2)
        rx._on_source_activated(2)
        self.assertTrue(rx.running)
        self.assertEqual(rx.source.name, "RTL-SDR #1")
        self.assertEqual(
            self.store.values["device/second_receiver/device"], "rtlsdr:00000002"
        )
        # It tunes the new device to where the receiver was.
        self.assertAlmostEqual(rx.device.state.frequency, rx.lo_frequency())
        self.frames(3)
        self.assertTrue(rx._spectrum._has_data)

    def test_a_dead_stream_stops_with_the_reason(self):
        rx = self.rx
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(2)
        rx.start()
        device = rx.device
        device._state.is_streaming = False
        device._rx_error = "USB error"
        rx._frame()
        self.assertFalse(rx.running)
        self.assertTrue(device.closed)
        self.assertIn("stopped streaming: USB error", rx.message())

    def test_the_frequency_field_takes_the_devices_range_while_it_runs(self):
        rx = self.rx
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(2)  # RTL-SDR #1 (24-1766 MHz)
        rx.start()
        low, high = rx._freq_input.tuning_range()
        self.assertEqual(
            (low, high), (rx.device.spec.freq_min, rx.device.spec.freq_max)
        )
        self.assertFalse(rx.tune(2.4e9))
        self.assertIn("RTL-SDR tunes", rx.message())
        rx.stop()
        self.assertEqual(rx._freq_input.tuning_range(), (1e3, 6e9))
        self.assertTrue(rx.tune(2.4e9))  # (stopped: no device to refuse it)
        # Started again, the RTL-SDR brings it into its range and says so.
        rx.start()
        self.assertEqual(rx.tuned_frequency(), rx.device.spec.freq_max)
        self.assertIn("out of range", rx.message())

    def test_a_device_that_wont_tune_shows_where_it_is(self):
        rx = self.rx
        rx.tune(433.92e6, recenter=True)
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(2)
        original = FakeRtl.set_frequency
        FakeRtl.set_frequency = lambda self, freq: False
        self.addCleanup(lambda: setattr(FakeRtl, "set_frequency", original))
        rx.start()
        self.assertEqual(rx.lo_frequency(), 100e6)  # (where FakeRtl opens)
        self.assertEqual(rx.tuned_frequency(), 100e6)
        self.assertEqual(rx._spectrum._center_freq, 100e6)
        self.assertIn("didn't tune to 433.920 MHz; it is on 100.000 MHz", rx.message())

    def test_dongles_sharing_a_serial_are_told_apart(self):
        """Cheap RTL-SDRs often all say serial 00000001."""
        from sdr_module.gui.second_receiver import list_sources

        original = FakeRtl.list_devices

        def same_serial():
            infos = original()
            for info in infos:
                info.serial = "00000001"
            return infos

        FakeRtl.list_devices = staticmethod(same_serial)
        self.addCleanup(
            lambda: setattr(FakeRtl, "list_devices", staticmethod(original))
        )
        keys = [s.key for s in list_sources(self.win)]
        self.assertEqual(keys, ["demo", "rtlsdr:00000001", "rtlsdr:00000001#2"])
        rx = self.rx
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(2)
        rx._on_source_activated(2)
        self.assertEqual(rx._selected_source().name, "RTL-SDR #1")
        self.assertTrue(rx.start())
        self.assertTrue(rx.holds("rtlsdr", 1))

    def test_no_hardware_says_so(self):
        self.win._hardware_classes = []
        self.rx._refresh_sources()
        self.assertEqual(self.rx._source_combo.count(), 1)
        self.assertIn("No other SDR found", self.rx._device_note.text())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTuning(_SecondCase):
    def setUp(self):
        super().setUp()
        self.rx.start()
        self.rx.set_mode("FM", "5 kHz", "25 kHz")
        self.rx.tune(145.8e6, recenter=True)

    def test_inside_the_span_only_the_vfo_moves(self):
        rx = self.rx
        lo = rx.lo_frequency()
        self.assertTrue(rx.tune(145.6e6))
        self.assertEqual(rx.lo_frequency(), lo)
        self.assertEqual(rx.tuned_frequency(), 145.6e6)
        self.assertAlmostEqual(rx._dsp.offset_hz, -200e3)
        self.assertEqual(rx._freq_label.text(), "145.600 MHz")
        self.assertEqual(rx._freq_input.get_frequency(), 145.6e6)
        self.assertEqual(rx._spectrum.tuned_frequency(), 145.6e6)
        self.assertEqual(rx.device.frequency, lo)

    def test_outside_it_the_device_retunes(self):
        rx = self.rx
        self.assertTrue(rx.tune(162.55e6))
        self.assertEqual(rx.lo_frequency(), 162.55e6)
        self.assertEqual(rx.device.frequency, 162.55e6)
        self.assertEqual(rx._spectrum._center_freq, 162.55e6)

    def test_out_of_range_is_refused(self):
        rx = self.rx
        self.assertFalse(rx.tune(7e9))
        self.assertEqual(rx.tuned_frequency(), 145.8e6)
        self.assertIn("out of range", rx.message())

    def test_typed_clicked_scrolled_and_digit_tuning(self):
        rx = self.rx
        rx._freq_input.frequency_changed.emit(146.52e6)
        self.assertEqual(rx.tuned_frequency(), 146.52e6)
        rx._spectrum.frequency_clicked.emit(146.0e6)
        self.assertEqual(rx.tuned_frequency(), 146.0e6)
        step = rx._step_hz
        rx._waterfall.tune_steps.emit(2)
        self.assertAlmostEqual(rx.tuned_frequency(), 146.0e6 + 2 * step)
        rx._freq_digits.digit_tuned.emit(1e6)
        self.assertAlmostEqual(rx.tuned_frequency(), 147.0e6 + 2 * step)

    def test_mode_brings_its_filter_and_deviation(self):
        rx = self.rx
        rx._mode_combo.setCurrentText("AM")
        self.assertEqual(rx._filter_combo.currentText(), "10 kHz")
        self.assertEqual(
            [rx._filter_combo.itemText(i) for i in range(rx._filter_combo.count())],
            ["5 kHz", "6 kHz", "8 kHz", "10 kHz", "12 kHz", "15 kHz"],
        )
        self.assertFalse(rx._tuning_form.isRowVisible(rx._deviation_combo))
        rx._mode_combo.setCurrentText("USB")
        self.assertEqual(rx._filter_combo.currentText(), "2.7 kHz")
        self.assertEqual(rx._dsp.band_edges(), (200.0, 2900.0))
        lo, hi = rx._spectrum.passband()
        offset = rx.tuned_frequency() - rx.lo_frequency()
        self.assertAlmostEqual(lo, offset + 200.0)
        self.assertAlmostEqual(hi, offset + 2900.0)
        rx._mode_combo.setCurrentText("FM")
        self.assertTrue(rx._tuning_form.isRowVisible(rx._deviation_combo))
        self.assertEqual(rx._deviation_value(), "5 kHz")
        self.assertEqual(rx._filter_combo.currentText(), "25 kHz")
        rx.tune(100.1e6)
        rx._mode_combo.setCurrentText("AM")
        rx._mode_combo.setCurrentText("FM")  # broadcast band: wide FM
        self.assertEqual(rx._deviation_value(), "75 kHz")
        self.assertEqual(rx._filter_combo.currentText(), "200 kHz")

    def test_the_fm_deviation_is_never_shown_cut_off(self):
        """The label of 2.5 kHz ("2.5 kHz (narrow, 12.5 kHz channels)",
        what Monitor TX picks for an FM tone ID) is wider than the box in
        the controls column: the box shows the value instead of a label
        cut off mid-word; a label that fits shows whole (the list always
        does)."""
        rx = self.rx
        combo = rx._deviation_combo
        metrics = combo.fontMetrics()
        rx.set_mode("FM", "2.5 kHz", "12.5 kHz")
        self.assertEqual(combo.currentText(), "2.5 kHz (narrow, 12.5 kHz channels)")
        self.assertGreater(
            metrics.horizontalAdvance(combo.currentText()), combo.text_room()
        )
        self.assertEqual(combo.shown_text(), "2.5 kHz")
        rx.set_mode("FM", "5 kHz", "25 kHz")
        self.assertEqual(combo.shown_text(), "5 kHz (narrow voice)")
        combo.grab()  # (paints)
        # Without a shorter form, a text too wide ends in "…".
        rx._listen_combo.setItemText(0, "Main window " * 6)
        rx.set_listen("main")
        shown = rx._listen_combo.shown_text()
        self.assertTrue(shown.endswith("…"), shown)
        self.assertLessEqual(
            rx._listen_combo.fontMetrics().horizontalAdvance(shown),
            rx._listen_combo.text_room(),
        )

    def test_step_follows_the_mode_and_is_remembered(self):
        rx = self.rx
        rx._mode_combo.setCurrentText("AM")
        self.assertEqual(rx._step_hz, 10e3)
        rx._set_step(5e3)
        rx._mode_combo.setCurrentText("USB")
        self.assertEqual(rx._step_hz, 100.0)
        rx._mode_combo.setCurrentText("AM")
        self.assertEqual(rx._step_hz, 5e3)
        self.assertEqual(self.store.values["device/second_receiver/step/am"], 5e3)

    def test_keyboard(self):
        rx = self.rx
        rx._spectrum.setFocus()
        self.settle()
        start = rx.tuned_frequency()
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Right)
        self.assertAlmostEqual(rx.tuned_frequency(), start + rx._step_hz)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
        self.assertAlmostEqual(rx.tuned_frequency(), start - 9 * rx._step_hz)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Space)
        self.assertFalse(rx.running)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Space)
        self.assertTrue(rx.running)
        step = rx._step_hz
        QTest.keyClick(rx._spectrum, "]")
        self.assertGreater(rx._step_hz, step)
        # Space on a combo box is the combo's.
        rx._mode_combo.setFocus()
        self.settle()
        QTest.keyClick(rx._mode_combo, Qt.Key.Key_Space)
        self.assertTrue(rx.running)
        QApplication.processEvents()
        rx._mode_combo.hidePopup()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSampleRate(_SecondCase):
    def rates(self):
        combo = self.rx._rate_combo
        return [combo.itemData(i) for i in range(combo.count())]

    def choose(self, rate):
        combo = self.rx._rate_combo
        index = combo.findData(rate)
        self.assertGreaterEqual(index, 0)
        combo.setCurrentIndex(index)
        combo.activated.emit(index)

    def test_the_list_follows_the_device(self):
        from sdr_module.gui.second_receiver import SAMPLE_RATES

        self.assertEqual(self.rates(), list(SAMPLE_RATES["demo"]))
        self.assertEqual(self.rx._rate_combo.currentData(), 2.4e6)
        self.win._hardware_classes = [FakeRtl]
        self.rx._refresh_sources()
        self.rx._source_combo.setCurrentIndex(2)
        self.rx._on_source_activated(2)
        self.assertEqual(self.rates(), list(SAMPLE_RATES["rtlsdr"]))

    def test_changing_it_while_running(self):
        rx = self.rx
        rx.start()
        rx.tune(100.0e6, recenter=True)
        self.assertTrue(rx.tune(100.9e6))  # 900 kHz off the LO: fits 2.4 MS/s
        self.assertEqual(rx.lo_frequency(), 100.0e6)
        self.choose(1.0e6)
        self.assertEqual(rx.device.sample_rate, 1.0e6)
        self.assertEqual(rx._spectrum._sample_rate, 1.0e6)
        self.assertEqual(rx._rate_label.text(), "1 MS/s")
        # The channel no longer fits the narrower span: recentered on it.
        self.assertEqual(rx.lo_frequency(), 100.9e6)
        self.assertEqual(rx.tuned_frequency(), 100.9e6)
        self.feed(3)
        self.assertEqual(rx._dsp.chain().rate, 1.0e6)
        self.assertEqual(
            self.store.values["device/second_receiver/sample_rate/demo"], 1.0e6
        )

    def test_switching_devices_uses_the_new_devices_rates(self):
        rx = self.rx
        self.win._hardware_classes = [FakeRtl]
        rx._refresh_sources()
        self.choose(3.2e6)  # (demo only: an RTL-SDR stops at 2.56 MS/s)
        rx.start()
        self.assertEqual(rx.device.sample_rate, 3.2e6)
        rx._source_combo.setCurrentIndex(2)
        rx._on_source_activated(2)
        self.assertEqual(rx.source.kind, "rtlsdr")
        self.assertEqual(rx.device.state.sample_rate, 2.4e6)
        self.assertEqual(rx._rate_combo.currentData(), 2.4e6)

    def test_the_saved_device_comes_back_with_its_rates(self):
        from sdr_module.gui.second_receiver import SAMPLE_RATES, SecondReceiverWindow

        self.win._hardware_classes = [FakeRtl]
        self.rx.shutdown()
        self.store.values["device/second_receiver/device"] = "rtlsdr:00000002"
        self.store.values["device/second_receiver/sample_rate/rtlsdr"] = 1.8e6
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._selected_source().name, "RTL-SDR #1")
        rates = [
            again._rate_combo.itemData(i) for i in range(again._rate_combo.count())
        ]
        self.assertEqual(rates, list(SAMPLE_RATES["rtlsdr"]))
        self.assertEqual(again._rate_combo.currentData(), 1.8e6)

    def test_a_hackrf_gets_its_baseband_filter_for_the_rate(self):
        """As in the main window: 0.75x the sample rate (the whole span
        without what would alias into it)."""
        rx = self.rx
        FakeHackRF.open_now = False
        self.win._hardware_classes = [FakeHackRF]
        rx._refresh_sources()
        rx._source_combo.setCurrentIndex(1)
        rx._on_source_activated(1)
        self.assertEqual(self.rates()[-1], 20e6)
        self.choose(8e6)
        self.assertTrue(rx.start())
        self.assertEqual(rx.device.state.sample_rate, 8e6)
        self.assertEqual(rx.device.filters[-1], 6e6)
        self.choose(20e6)
        self.assertEqual(rx.device.filters[-1], 15e6)
        # No automatic gain on a HackRF: Auto says so and turns itself off.
        rx._gain_auto.setChecked(True)
        self.assertFalse(rx._gain_auto.isChecked())
        self.assertIn("has no automatic gain", rx.message())
        rx.stop()

    def test_refused_while_recording(self):
        rx = self.rx
        folder = tempfile.mkdtemp(prefix="rx2-rate-")
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, True))
        self.store.values["recording/folder"] = folder
        rx.start()
        rx._set_recording(True)
        self.choose(1.0e6)
        self.assertEqual(rx.device.sample_rate, 2.4e6)
        self.assertEqual(rx._rate_combo.currentData(), 2.4e6)
        self.assertIn("Stop recording to change the sample rate", rx.message())
        rx._set_recording(False)

    def test_the_device_opens_at_the_chosen_rate(self):
        rx = self.rx
        self.choose(3.2e6)
        rx.start()
        self.assertEqual(rx.device.sample_rate, 3.2e6)
        self.assertEqual(rx._spectrum._sample_rate, 3.2e6)

    def test_a_rate_the_device_refuses_isnt_remembered(self):
        """The rate remembered for a kind of device is one it took: a
        refused one would be asked for again every time it opens."""
        from sdr_module.gui.device_dialog import MockDevice

        key = "device/second_receiver/sample_rate/demo"
        rx = self.rx
        rx.start()
        original = MockDevice.set_sample_rate

        def refuse_1_ms(device, rate):
            return False if rate == 1.0e6 else original(device, rate)

        MockDevice.set_sample_rate = refuse_1_ms
        self.addCleanup(lambda: setattr(MockDevice, "set_sample_rate", original))
        self.choose(1.0e6)
        self.assertEqual(rx.device.sample_rate, 2.4e6)
        self.assertEqual(rx._rate_combo.currentData(), 2.4e6)
        self.assertIn("didn't take 1 MS/s", rx.message())
        self.assertNotEqual(self.store.values.get(key), 1.0e6)
        # One it takes is remembered, running or stopped.
        self.choose(1.8e6)
        self.assertEqual(rx.device.sample_rate, 1.8e6)
        self.assertEqual(self.store.values[key], 1.8e6)
        rx.stop()
        self.choose(3.2e6)
        self.assertEqual(self.store.values[key], 3.2e6)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSquelch(_SecondCase):
    def test_auto_by_default_and_a_dragged_threshold_turns_it_off(self):
        rx = self.rx
        gate = rx._dsp.squelch
        self.assertTrue(gate.enabled and gate.auto)
        self.assertFalse(rx._squelch_slider.isEnabled())
        rx._level_meter.threshold_changed.emit(-55.0)
        self.assertFalse(rx._squelch_auto.isChecked())
        self.assertFalse(gate.auto)
        self.assertEqual(gate.manual_db, -55.0)
        self.assertTrue(rx._squelch_slider.isEnabled())
        self.assertEqual(rx._squelch_label.text(), "-55 dBFS")
        self.assertEqual(self.store.values["device/second_receiver/squelch_db"], -55.0)

    def test_dragging_the_threshold_turns_the_squelch_on(self):
        """As on the main window's LEVEL: a threshold dragged with the
        squelch off turns it on (otherwise the drag shows nothing: the
        marker is hidden while the squelch is off)."""
        rx = self.rx
        rx._squelch_check.setChecked(False)
        rx._level_meter.threshold_changed.emit(-60.0)
        gate = rx._dsp.squelch
        self.assertTrue(rx._squelch_check.isChecked())
        self.assertTrue(gate.enabled)
        self.assertFalse(gate.auto)
        self.assertEqual(gate.manual_db, -60.0)
        self.assertTrue(rx._squelch_slider.isEnabled())
        self.assertEqual(rx._squelch_label.text(), "-60 dBFS")
        self.assertTrue(self.store.values["device/second_receiver/squelch_on"])

    def test_off_plays_everything(self):
        rx = self.rx
        rx._squelch_check.setChecked(False)
        self.assertFalse(rx._dsp.squelch.enabled)
        rx.start()
        rx.tune(150.0e6, recenter=True)  # (nothing there)
        self.frames(4)
        self.assertEqual(rx._level_meter.state(), "off")

    def test_the_level_meter_reads_the_tuned_channel(self):
        rx = self.rx
        rx.start()
        rx.tune(100.1e6, recenter=True)
        rx.set_mode("FM", "75 kHz", "200 kHz")
        self.feed(10)
        self.assertEqual(rx._level_meter.state(), "open")
        loud = rx._level_meter.level()
        rx.tune(100.4e6)  # an empty channel inside the span
        self.assertEqual(rx.lo_frequency(), 100.1e6)
        self.feed(15)  # (0.5 s: longer than the squelch's hang time)
        self.assertEqual(rx._level_meter.state(), "closed")
        self.assertLess(rx._level_meter.level(), loud - 30.0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestListen(_SecondCase):
    """Whose audio plays: the main window's, this one's, or both."""

    demo = True

    def setUp(self):
        super().setUp()
        self.main_sink = _FakeSink()
        self.main_sink.open = True
        self.win._audio = self.main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None  # (as with an output device)
        self.win.set_frequency(100.1e6)
        self.second_sink = self.sink()
        rx = self.rx
        rx.start()
        rx.tune(100.1e6, recenter=True)
        rx.set_mode("FM", "75 kHz", "200 kHz")

    def run_both(self, frames=10, second=True):
        """``frames`` display frames of both windows (``second`` False: the
        main window's alone), 1/30 s apart on a fake clock. (On the real
        clock a busy computer's late frames fell behind real time, which
        pauses the audio by design, and these counts came out short.)"""
        from sdr_module.gui import main_window, second_receiver
        from tests.test_gui_round3_main import _FakeClock

        self.main_sink.blocks.clear()
        self.second_sink.blocks.clear()
        clock = getattr(self, "clock", None)
        if clock is None:
            # (Going on from the real clock the pacers last read.)
            clock = self.clock = _FakeClock(time.monotonic())
        with (
            mock.patch.object(main_window, "time", clock),
            mock.patch.object(second_receiver, "time", clock),
        ):
            for _ in range(frames):
                clock.advance(1 / 30)
                self.win._update_display()
                if second:
                    self.rx._frame()

    def test_both_plays_both(self):
        self.rx.set_listen("both")
        self.run_both(12)
        self.assertGreater(len(self.main_sink.samples()), 4800)
        self.assertGreater(len(self.second_sink.samples()), 4800)
        self.assertFalse(self.host.mutes_main())

    def test_main_only(self):
        self.rx.set_listen("main")
        self.assertFalse(self.second_sink.open)
        self.run_both(12)
        self.assertGreater(len(self.main_sink.samples()), 4800)
        self.assertEqual(len(self.second_sink.samples()), 0)

    def test_second_only_mutes_the_main_window_until_it_closes(self):
        self.rx.set_listen("second")
        self.assertTrue(self.host.mutes_main())
        self.assertIn("Main receiver muted", self.win._message_label.text())
        self.run_both(4)  # (the main window's audio fades out)
        self.run_both(12)
        self.assertEqual(len(self.main_sink.samples()), 0)
        self.assertGreater(len(self.second_sink.samples()), 4800)
        self.assertIn("main window is muted", self.rx._audio_note.text())
        self.assertEqual(self.rx._audio_note.property("role"), "hint")
        self.assertEqual(self.rx._audio_note.property("tone"), "info")
        self.rx.close()
        self.settle()
        self.assertFalse(self.host.mutes_main())
        self.assertIn("audio is back", self.win._message_label.text())
        self.run_both(12, second=False)
        self.assertGreater(len(self.main_sink.samples()), 4800)

    def test_the_main_audio_switch_mutes_both(self):
        self.rx.set_listen("both")
        self.win._audio_enabled = False
        self.rx._update_status()
        self.assertFalse(self.second_sink.open)
        self.run_both(8)
        self.assertEqual(len(self.second_sink.samples()), 0)
        self.assertIn("Audio is off in the main window", self.rx._audio_note.text())

    def test_it_plays_through_the_main_windows_output(self):
        """Where the sink can choose an output device (Tools > Preferences),
        this receiver uses the one the main window plays through."""

        class ChoosingSink(_FakeSink):
            def __init__(self, output_id=""):
                super().__init__()
                self.output_id = output_id
                self.chosen = []

            def set_output_device(self, device_id, name=""):
                self.output_id = device_id
                self.chosen.append(device_id)

        self.win._audio = ChoosingSink("usb-headset")
        sink = ChoosingSink()
        self.rx._audio = sink
        self.rx._audio_open = False
        self.rx.set_listen("main")
        self.rx.set_listen("both")  # (opens this receiver's output)
        self.assertTrue(sink.open)
        self.assertEqual(sink.output_id, "usb-headset")
        self.win._audio.output_id = "speakers"  # chosen meanwhile
        self.rx._update_status()
        self.assertEqual(sink.chosen, ["usb-headset", "speakers"])
        self.rx._update_status()  # (no change: not chosen again)
        self.assertEqual(len(sink.chosen), 2)

    def test_volume_is_its_own(self):
        self.rx._volume_slider.setValue(35)
        self.assertAlmostEqual(self.second_sink.volume, 0.35)
        self.assertEqual(self.store.values["device/second_receiver/volume"], 35)

    def test_an_output_that_wont_open_says_so(self):
        """A second audio stream can fail to open (an output that takes one
        stream at a time): the note says this receiver is silent, not that
        both play."""

        class FailingSink(_FakeSink):
            def start(self, rate=48000):
                return False

        rx = self.rx
        rx.set_listen("main")
        rx._audio = FailingSink()
        rx._audio_open = False
        rx.set_listen("both")
        self.assertFalse(rx._audio_open)
        note = rx._audio_note.text()
        self.assertIn("audio output didn't open", note)
        self.assertNotIn("Both receivers play", note)
        self.assertEqual(rx._audio_note.property("tone"), "warning")
        # Only this receiver: the main window is muted too, so nothing plays.
        rx.set_listen("second")
        self.assertIn("audio output didn't open", rx._audio_note.text())
        self.assertIn("main window is muted too", rx._audio_note.text())
        # Not wanted (Main window), or opened at last: the usual note.
        rx.set_listen("main")
        self.assertIn("Only the main window plays", rx._audio_note.text())
        rx.set_listen("both")
        rx._audio = _FakeSink()
        rx._audio_retry_at = 0.0
        rx._update_status()
        self.assertTrue(rx._audio_open)
        self.assertIn("Both receivers play", rx._audio_note.text())
        self.assertIsNone(rx._audio_note.property("tone") or None)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarks(_SecondCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._bookmarks_panel
        self.panel.add_bookmark("Airport tower", 118.3e6, "AM", "", "8 kHz")
        self.panel.add_bookmark("NOAA 1", 162.55e6, "FM", "5 kHz", "")

    def test_no_bookmarks_says_how_to_add_one(self):
        for index in reversed(range(self.panel.bookmark_count())):
            self.panel._bookmarks.pop(index)
        self.panel.bookmarks_changed.emit()
        combo = self.rx._bookmark_combo
        self.assertEqual(combo.count(), 1)
        self.assertFalse(combo.isEnabled())
        self.assertIn("Ctrl+B", combo.itemText(0))

    def test_the_list_follows_the_main_windows_bookmarks(self):
        combo = self.rx._bookmark_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(texts[0], "Tune to a bookmark…")
        self.assertIn("Airport tower · 118.300 MHz", texts)
        self.assertIn("NOAA 1 · 162.550 MHz", texts)
        self.panel.add_bookmark("New", 433.92e6)
        self.assertEqual(combo.count(), len(texts) + 1)

    def test_tuning_a_bookmark_sets_its_mode_and_filter(self):
        rx = self.rx
        combo = rx._bookmark_combo
        index = next(
            i for i in range(combo.count()) if combo.itemText(i).startswith("Airport")
        )
        combo.setCurrentIndex(index)
        combo.activated.emit(index)
        self.assertEqual(rx.tuned_frequency(), 118.3e6)
        self.assertEqual(rx.mode(), "AM")
        self.assertEqual(rx._filter_combo.currentText(), "8 kHz")
        self.assertEqual(combo.currentIndex(), 0)
        self.assertIn("Tuned to Airport tower (AM)", rx.message())
        # A bookmark with only a deviation gets the usual channel for it.
        rx.tune_bookmark(self.panel.bookmark(1))
        self.assertEqual(rx.mode(), "FM")
        self.assertEqual(rx._deviation_value(), "5 kHz")
        self.assertEqual(rx._filter_combo.currentText(), "25 kHz")
        # The main window stays where it was.
        self.assertNotEqual(self.win._current_frequency(), 162.55e6)

    def test_bookmarking_adds_to_the_main_windows_list(self):
        rx = self.rx
        rx.tune(146.52e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        count = self.panel.bookmark_count()
        rx._bookmark_button.click()
        self.assertEqual(self.panel.bookmark_count(), count + 1)
        added = self.panel.bookmark(count)
        self.assertEqual(added["freq_hz"], 146.52e6)
        self.assertEqual(added["demod"], "FM")
        self.assertEqual(added["fm_deviation"], "5 kHz")
        self.assertEqual(added["bandwidth"], "25 kHz")
        self.assertIn("Bookmarked", rx.message())
        rx._bookmark_action.trigger()  # Ctrl+B: already there
        self.assertEqual(self.panel.bookmark_count(), count + 1)
        self.assertIn("Already bookmarked", rx.message())
        # A signal right-clicked on the waterfall.
        rx._waterfall.bookmark_requested.emit(146.94e6)
        self.assertEqual(self.panel.bookmark(count + 1)["freq_hz"], 146.94e6)

    def test_ctrl_b_in_this_window_bookmarks_this_receiver(self):
        rx = self.rx
        rx.tune(146.52e6, recenter=True)
        rx.activateWindow()
        rx._spectrum.setFocus()
        self.settle()
        if QApplication.activeWindow() is not rx:
            self.skipTest("the platform can't activate the window")
        count = self.panel.bookmark_count()
        QTest.keyClick(rx._spectrum, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.panel.bookmark_count(), count + 1)
        self.assertEqual(self.panel.bookmark(count)["freq_hz"], 146.52e6)
        self.assertNotEqual(self.win._current_frequency(), 146.52e6)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRecording(_SecondCase):
    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="rx2-rec-")
        self.store.values["recording/folder"] = self.folder
        self.addCleanup(self._remove_folder)

    def _remove_folder(self):
        import shutil

        shutil.rmtree(self.folder, ignore_errors=True)

    def test_record_needs_the_receiver_running(self):
        rx = self.rx
        self.assertFalse(rx._record_button.isEnabled())
        self.assertFalse(rx._set_recording(True))
        self.assertIn("Start this receiver first", rx.message())

    def test_records_a_sigmf_file_the_main_window_can_play(self):
        from sdr_module.dsp.recording import IQPlayer
        from sdr_module.gui.iq_files import describe_iq_file

        rx = self.rx
        rx.start()
        rx.tune(145.8e6, recenter=True)
        self.assertTrue(rx._record_button.isEnabled())
        rx._record_button.click()
        self.assertTrue(rx._record_button.isChecked())
        self.assertEqual(rx._record_button.property("role"), "danger")
        self.assertTrue(rx._rec_label.isVisible())
        recorder = rx._recorder
        self.frames(6)
        written = recorder.samples_written
        self.assertGreater(written, 0)
        rx._record_button.click()
        self.assertIsNone(rx._recorder)
        self.assertFalse(rx._rec_label.isVisible())
        files = sorted(os.listdir(self.folder))
        data = [f for f in files if f.endswith(".sigmf-data")]
        meta = [f for f in files if f.endswith(".sigmf-meta")]
        self.assertEqual(len(data), 1, files)
        self.assertEqual(len(meta), 1, files)
        self.assertTrue(data[0].startswith("iq_145.800MHz_2.4MSps_"))
        self.assertTrue(data[0].endswith("_rx2.sigmf-data"))
        path = os.path.join(self.folder, data[0])
        self.assertEqual(os.path.getsize(path), written * 8)
        with open(os.path.join(self.folder, meta[0])) as handle:
            info = json.load(handle)
        self.assertEqual(info["global"]["core:datatype"], "cf32_le")
        self.assertEqual(info["global"]["core:sample_rate"], 2.4e6)
        self.assertEqual(info["captures"][0]["core:frequency"], 145.8e6)
        # What File > Open Recording reads: nothing to ask.
        options, missing = describe_iq_file(path)
        self.assertEqual(missing, set())
        self.assertEqual(options.center_hz, 145.8e6)
        self.assertEqual(options.sample_rate, 2.4e6)
        self.assertEqual(options.total_samples, written)
        player = IQPlayer(path)
        self.assertEqual(player.total_samples, written)
        self.assertEqual(player.center_frequency, 145.8e6)
        player.close()
        self.assertIn("Saved", rx.message())

    def test_tuning_stays_inside_the_recorded_span(self):
        rx = self.rx
        rx.start()
        rx.tune(145.8e6, recenter=True)
        rx._set_recording(True)
        self.assertTrue(rx.tune(145.5e6))  # inside: fine
        self.assertEqual(rx.lo_frequency(), 145.8e6)
        self.assertFalse(rx.tune(162.55e6))  # outside: refused
        self.assertIn("Stop recording to tune there", rx.message())
        self.assertEqual(rx.lo_frequency(), 145.8e6)
        rx._set_recording(False)
        self.assertTrue(rx.tune(162.55e6))

    def test_stopping_the_receiver_finishes_the_file(self):
        rx = self.rx
        rx.start()
        rx._set_recording(True)
        self.frames(3)
        rx.stop()
        names = os.listdir(self.folder)
        self.assertTrue(any(n.endswith(".sigmf-meta") for n in names), names)

    def test_closing_the_main_window_finishes_the_file(self):
        rx = self.rx
        rx.start()
        rx._set_recording(True)
        self.frames(3)
        self.win.close()
        self.settle()
        names = os.listdir(self.folder)
        self.assertTrue(any(n.endswith(".sigmf-meta") for n in names), names)
        self.assertIsNone(rx.device)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestPersistence(_SecondCase):
    def test_settings_come_back(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        rx = self.rx
        rx.tune(433.92e6, recenter=True)
        rx.set_mode("AM", "", "12 kHz")
        rx.set_listen("second")
        rx._squelch_auto.setChecked(False)
        rx._squelch_slider.setValue(-70)
        rx._gain_slider.setValue(33)
        rx._follow_check.setChecked(True)
        rx._waterfall._on_colormap_changed("viridis")  # (the ⋯ menu's Colors)
        rx._waterfall._history_combo.setCurrentIndex(
            rx._waterfall._history_combo.findData(60.0)
        )
        rx.resize(rx.minimumWidth() + 40, 640)
        rx.close()
        self.settle()
        values = self.store.values
        self.assertEqual(values["device/second_receiver/frequency_hz"], 433.92e6)
        self.assertEqual(values["device/second_receiver/mode"], "AM")
        self.assertEqual(values["device/second_receiver/bandwidth"], "12 kHz")
        self.assertEqual(values["device/second_receiver/device"], "demo")
        self.assertEqual(values["device/second_receiver/listen"], "second")
        self.assertTrue(values["layout/second_receiver/geometry"])
        self.assertTrue(values["layout/second_receiver/splitter"])

        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._waterfall.colormap_name(), "viridis")
        self.assertEqual(again._waterfall.history_seconds(), 60.0)
        self.assertEqual(again.tuned_frequency(), 433.92e6)
        self.assertEqual(again.mode(), "AM")
        self.assertEqual(again._filter_combo.currentText(), "12 kHz")
        self.assertEqual(again.listen(), "second")
        self.assertFalse(again._squelch_auto.isChecked())
        self.assertEqual(again._dsp.squelch.manual_db, -70.0)
        self.assertEqual(again._gain_slider.value(), 33)
        self.assertTrue(again._follow_check.isChecked())
        # (The offscreen screen is 800 px wide: the width can't be checked.)
        self.assertEqual(again.size().height(), 640)

    def test_damaged_settings_fall_back(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.rx.shutdown()
        self.store.values.update(
            {
                "device/second_receiver/frequency_hz": float("nan"),
                "device/second_receiver/mode": "QAM",
                "device/second_receiver/listen": "loud",
                "device/second_receiver/squelch_db": float("nan"),
                "device/second_receiver/step/nfm": float("inf"),
                "layout/second_receiver/geometry": "not base64!",
            }
        )
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again.tuned_frequency(), 145.8e6)
        self.assertEqual(again.mode(), "FM")
        self.assertEqual(again.listen(), "both")
        self.assertEqual(again._dsp.squelch.manual_db, -80.0)
        self.assertEqual(again._step_hz, 12.5e3)
        self.assertGreaterEqual(again.width(), again.minimumWidth())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestLayout(_SecondCase):
    def test_fits_its_minimum_size_without_hidden_controls(self):
        rx = self.rx
        self.assertLessEqual(rx.minimumWidth(), 1024 - 100)
        self.assertLessEqual(rx.minimumHeight(), 640 - 60)
        rx.resize(rx.minimumWidth(), rx.minimumHeight())
        self.settle(10)
        toolbar = rx._toolbar
        for widget in (
            rx._start_button,
            rx._record_button,
            rx._freq_label,
            rx._level_meter,
        ):
            self.assertTrue(widget.isVisible(), widget)
            self.assertLessEqual(
                widget.mapTo(rx, widget.rect().topRight()).x(), rx.width()
            )
        # (Nothing in the toolbar's overflow menu.)
        extension = [
            child
            for child in toolbar.children()
            if type(child).__name__ == "QToolBarExtension"
        ]
        self.assertTrue(all(not child.isVisible() for child in extension))
        area = rx._controls_area
        self.assertEqual(
            area.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.assertGreaterEqual(
            area.viewport().width(), area.widget().minimumSizeHint().width()
        )

    def test_both_themes(self):
        from sdr_module.gui.themes import apply_theme, current_theme

        app = QApplication.instance()
        before = current_theme()
        self.addCleanup(lambda: apply_theme(app, before))
        self.rx.start()
        for theme in ("light", "dark"):
            apply_theme(app, theme)
            self.settle()
            self.frames(2)
            self.rx.grab()  # paints without error
        self.assertEqual(self.rx._start_button.text(), "■  Stop")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestShutdown(_SecondCase):
    def test_a_receiver_never_outlives_the_main_window_running(self):
        """A window the host doesn't know (the main window deleted without
        closing it) stops, finishing its recording and closing its device."""
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        folder = tempfile.mkdtemp(prefix="rx2-orphan-")
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, True))
        self.store.values["recording/folder"] = folder
        orphan = SecondReceiverWindow(self.win)
        self.assertTrue(orphan.start())
        orphan._set_recording(True)
        orphan._process(orphan.device.read_samples(BLOCK))
        device = orphan.device
        self.win.destroyed.emit()  # (as when the C++ window is deleted)
        self.assertFalse(orphan._frame_timer.isActive())
        self.assertFalse(orphan._status_timer.isActive())
        self.assertFalse(orphan.running)
        self.assertFalse(device.is_streaming)
        self.assertTrue(
            any(name.endswith(".sigmf-meta") for name in os.listdir(folder))
        )
        orphan._update_status()  # (inert: nothing of the main window's)
        orphan.shutdown()
        orphan.deleteLater()

    def test_closing_the_window_releases_everything(self):
        rx = self.rx
        rx.set_listen("second")
        rx.start()
        device = rx.device
        rx.close()
        self.settle()
        self.assertFalse(rx.running)
        self.assertFalse(device.is_streaming)
        self.assertFalse(rx._frame_timer.isActive())
        self.assertFalse(rx._status_timer.isActive())
        self.assertFalse(self.host.mutes_main())
        # Opened again, it is back in service with its settings.
        self.host.show()
        self.settle()
        self.assertTrue(rx._status_timer.isActive())
        self.assertTrue(self.host.mutes_main())
        self.assertTrue(rx.start())


# ---------------------------------------------------------------------------
# Transmit monitor
# ---------------------------------------------------------------------------


class TestTxListening(unittest.TestCase):
    def test_how_a_transmission_is_heard(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        listen = SecondReceiverWindow.tx_listening
        # The FM tone ID: 2.5 kHz deviation, 6.4 kHz wide.
        self.assertEqual(listen("FM", 6.4e3), ("FM", "2.5 kHz", "12.5 kHz"))
        self.assertEqual(listen("FM", 16e3), ("FM", "5 kHz", "25 kHz"))
        self.assertEqual(listen("FM", 180e3), ("FM", "75 kHz", "200 kHz"))
        self.assertEqual(listen("CW", 100.0), ("CW", "", ""))
        self.assertEqual(listen("AM", 6e3), ("AM", "", "6 kHz"))
        self.assertEqual(listen("SSB", 2.7e3), ("USB", "", ""))
        self.assertEqual(listen("LSB", 2.7e3), ("LSB", "", ""))
        # Text modes sent on the carrier (RTTY, PSK31, ASCII FSK: a few
        # hundred Hz around it) are heard with the CW beat note; wider data
        # (an AFSK FM carrier) in FM.
        self.assertEqual(listen("DATA", 500.0), ("CW", "", "1 kHz"))
        self.assertEqual(listen("DATA", 62.5), ("CW", "", "250 Hz"))
        self.assertEqual(listen("DATA", 6e3), ("FM", "2.5 kHz", "12.5 kHz"))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTxMonitorWithoutTransmitter(_SecondCase):
    def test_says_what_it_needs(self):
        rx = self.rx
        rx._update_tx_group()
        self.assertIsNone(rx.tx_target())
        self.assertFalse(rx._monitor_button.isEnabled())
        self.assertIn("Once it can, this receiver", rx._tx_label.text())
        self.assertIn("can't transmit: no device is connected", rx._tx_label.text())
        self.assertFalse(rx.monitor_tx())
        self.assertIn("Nothing to monitor", rx.message())

    def test_one_click_tunes_to_the_next_station_id(self):
        rx = self.rx
        station = self.win._tx_station
        station.device_problem = lambda: None
        station.id_spec = lambda freq: ("cw", "CW", 100.0, "CW, 20 WPM")
        self.win.set_frequency(144.05e6)
        rx._update_tx_group()
        self.assertTrue(rx._monitor_button.isEnabled())
        self.assertIn(
            "transmits on 144.050 MHz (CW for the next station ID)", rx._tx_label.text()
        )
        rx._monitor_button.click()
        self.assertTrue(rx.running)
        self.assertEqual(rx.tuned_frequency(), 144.05e6)
        self.assertEqual(rx.mode(), "CW")
        self.assertIn("Monitoring the next station ID on 144.050 MHz", rx.message())
        # (A demo device is honest about what it can hear.)
        self.assertIn("can't hear a real transmission", rx._tx_label.text())


if HAS_PYQT6:
    from tests.test_gui_station_tx import _TxCase
else:  # pragma: no cover - environment
    _TxCase = _WindowTestCase


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTxMonitorOnAir(_TxCase):
    """A HackRF (fake backend) in the main window sends a station ID
    through the transmit engine; the second receiver follows it and keeps
    receiving (full duplex)."""

    def setUp(self):
        super().setUp()
        self.rx = self.win._second_rx.show()
        self.rx._status_timer.stop()

    def tearDown(self):
        if hasattr(self, "rx"):
            self.rx.shutdown()
        super().tearDown()

    def test_follow_tunes_to_the_transmission_and_keeps_receiving(self):
        rx = self.rx
        rx._follow_check.setChecked(True)
        self.assertFalse(rx.running)  # (Follow starts it)
        self.panel.set_callsign("W1AW")
        self.hackrf(pace=True)  # the main window, on 146.520 MHz
        target = rx.tx_target()
        self.assertEqual(target[:2], (146.52e6, "FM"))
        self.assertTrue(self.station.send_station_id())
        self.assertTrue(self.station.busy)
        self.assertTrue(rx.running)
        self.assertEqual(rx.tuned_frequency(), 146.52e6)
        self.assertEqual(rx.mode(), "FM")
        self.assertEqual(rx._deviation_value(), "2.5 kHz")
        self.assertEqual(rx._filter_combo.currentText(), "12.5 kHz")
        rx._update_tx_group()
        # (How it is sent, as the main window says it: HAM-E2E-9.)
        self.assertIn(
            "On air: DE W1AW on 146.520 MHz (FM tone, 20 WPM).", rx._tx_label.text()
        )
        # (A hint colored by its tone: its role never changes.)
        self.assertEqual(rx._tx_label.property("role"), "hint")
        self.assertEqual(rx._tx_label.property("tone"), "danger")
        # Full duplex: the main receiver is held off, this one receives.
        self.assertFalse(self.win._is_running)
        for _ in range(3):
            rx._process(rx.device.read_samples(BLOCK))
        self.assertTrue(rx._spectrum._has_data)
        self.wait_done(30)
        self.assertTrue(self.station.engine.last_result.ok)
        self.assertTrue(rx.running)
        rx._update_tx_group()
        self.assertIn("The main window transmits on 146.520 MHz", rx._tx_label.text())

    def test_without_follow_it_stays_put(self):
        rx = self.rx
        rx.start()
        rx.tune(100.1e6, recenter=True)
        self.panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        self.assertTrue(self.station.send_station_id())
        self.assertEqual(rx.tuned_frequency(), 100.1e6)
        # One click while on the air.
        self.assertTrue(rx.monitor_tx())
        self.assertEqual(rx.tuned_frequency(), 146.52e6)
        self.station.stop()
        self.wait_done(10)
