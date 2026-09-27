#!/usr/bin/env python3
"""
Main window: the fourth review round.

The display pacer recovering from a passing overload, hardware filters
left to the sample rate (a HackRF's baseband filter follows it), broadcast
FM de-emphasis and pilot rejection, recordings streamed to a temporary
file, channel widths that suit a bookmark's mode, a receiver chain that
stays cheap at awkward sample rates and resamples to exactly 48 kHz, the
passband on the waterfall and ADS-B squawk codes.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round4_main.py``
"""

import os
import shutil
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_round3_main import _FakeClock, _FakeDemo  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


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


def _mw():
    from sdr_module.gui import main_window

    return main_window


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------


def _fm(audio, rate, deviation):
    """Complex baseband FM of ``audio`` (full scale = ``deviation``)."""
    phase = 2 * np.pi * deviation * np.cumsum(audio) / rate
    return np.exp(1j * phase).astype(np.complex64)


def _pre_emphasis(audio, rate, tau=75e-6):
    """Broadcast pre-emphasis, 1 + j 2 pi f tau (a whole number of cycles
    of each tone, so the FFT does it exactly)."""
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    return np.fft.irfft(spectrum * (1 + 2j * np.pi * freqs * tau), len(audio))


def _tone_level(audio, freq, rate):
    """Amplitude of the ``freq`` component of ``audio``."""
    t = np.arange(len(audio)) / rate
    window = np.hanning(len(audio))
    return (
        2
        * abs(np.sum(audio * window * np.exp(-2j * np.pi * freq * t)))
        / np.sum(window)
    )


def _receive(chain, iq, deviation, block=50_000):
    """Audio of ``iq`` through ``chain``, fed in blocks."""
    return np.concatenate(
        [
            chain.demodulate(chain.channelize(iq[i : i + block]), deviation)
            for i in range(0, len(iq), block)
        ]
    )


# ---------------------------------------------------------------------------
# 1. The pacer recovers from an overload
# ---------------------------------------------------------------------------


class TestPacerRecovers(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()
        self.pacer = self.mw._Pacer()

    def _overload(self, rate):
        for _ in range(3):
            self.pacer.measured(80000, rate, 0.045 * 80000 / (rate * 0.033))
        self.assertTrue(self.pacer.overloaded)

    def test_overloaded_demo_still_times_a_frame_every_second(self):
        mw, pacer, rate = self.mw, self.pacer, 2.4e6
        self._overload(rate)
        now = 10.0
        pacer.demo_request(rate, now)
        sizes = []
        for _ in range(70):  # over two seconds of frames
            now += 0.033
            sizes.append(pacer.demo_request(rate, now))
        probes = [s for s in sizes if s >= mw._TIMED_FRAME]
        self.assertIn(mw.DISPLAY_BLOCK, sizes)  # still one block per frame...
        self.assertEqual(len(probes), 2)  # ...but a timed frame once a second
        self.assertEqual(probes[0], round(rate * 0.033))

    def test_a_probe_is_long_enough_to_time_at_a_low_rate(self):
        mw, pacer, rate = self.mw, self.pacer, 250e3
        self._overload(rate)
        pacer.demo_request(rate, 0.0)
        sizes = [pacer.demo_request(rate, 0.033 * i) for i in range(1, 40)]
        self.assertGreaterEqual(max(sizes), mw._TIMED_FRAME)

    def test_fast_probes_recover_within_a_few_frames(self):
        mw, pacer, rate = self.mw, self.pacer, 2.4e6
        self._overload(rate)
        self.assertEqual(pacer.read_limit(rate, 5.0), 0.0)  # schedules a probe
        self.assertEqual(pacer.read_limit(rate, 5.5), 0.0)
        frames = 0
        now = 6.0  # due
        while pacer.overloaded and frames < 10:
            limit = pacer.read_limit(rate, now)
            self.assertGreater(limit, 0)  # after a fast probe, probe again
            pacer.measured(int(limit), rate, 0.1 * limit / rate)  # fast
            frames += 1
            now += 0.033
        self.assertFalse(pacer.overloaded)
        self.assertEqual(frames, 3)
        self.assertEqual(pacer.read_limit(rate, now), rate * mw._MAX_READ_S)

    def test_hardware_hiccup_with_short_transfers_recovers(self):
        """Transfers shorter than a timed frame: a probe drains several."""
        pacer, rate = self.pacer, 2.4e6
        self._overload(rate)
        pacer.read_limit(rate, 0.0)
        limits = [pacer.read_limit(rate, 0.033 * i) for i in range(1, 40)]
        self.assertTrue(any(limit >= self.mw._TIMED_FRAME for limit in limits))

    def test_start_forgets_an_overload(self):
        pacer, rate = self.pacer, 2.4e6
        self._overload(rate)
        pacer.restart()
        self.assertFalse(pacer.overloaded)
        self.assertEqual(pacer.read_limit(rate), rate * self.mw._PROBE_READ_S)


class TestDemoHiccup(_WindowTestCase):
    demo = True

    def test_a_hiccup_mutes_the_demo_only_while_it_lasts(self):
        clock = _FakeClock()
        source = _FakeDemo(clock, cost=0.02e-6)
        spoken = []
        self.win._audio.write = spoken.append
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-100)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")

        def run(seconds):
            end = clock.now + seconds
            while clock.now < end:
                self.win._update_display()
                clock.advance(0.033)

        with mock.patch.object(self.mw, "time", clock):
            self.win._stop_acquisition()
            self.win._device = source  # still in demo mode
            self.win._start_acquisition()
            run(1.0)
            self.assertTrue(spoken)
            self.assertFalse(self.win._pacer.overloaded)

            source.cost = 1.5 / source.sample_rate  # another program hogs the CPU
            run(1.0)
            self.assertTrue(self.win._pacer.overloaded)
            source.cost = 0.02e-6
            run(0.5)  # the hiccup is over, but nothing times it yet...
            before = len(spoken)
            self.assertTrue(self.win._realtime_hint_shown)
            self.assertIn("demo device", self.win._message_label.text())

            run(4.0)  # ...until the next probe frame
            self.assertFalse(self.win._pacer.overloaded)
            self.assertTrue(self.win._realtime())
            self.assertGreater(len(spoken), before)  # audio again
            self.assertFalse(self.win._realtime_hint_shown)  # hints next time
            self.assertGreater(np.median(source.requests[-10:]), 50000)
        self.win._stop_acquisition()
        self.win._device = None
        self.win._demo_mode = False
        self.win._refresh_state_ui()


# ---------------------------------------------------------------------------
# 2. Hardware filters follow the sample rate, not the channel
# ---------------------------------------------------------------------------


def _fake_hackrf(rate=10e6):
    from sdr_module.devices.hackrf import HackRFDevice

    class FakeHackRF(HackRFDevice):
        def __init__(self):
            super().__init__()
            self._state.sample_rate = rate
            self.bandwidths = []

        def set_bandwidth(self, bw_hz):
            self.bandwidths.append(bw_hz)
            return True

        def set_sample_rate(self, rate_hz):
            self._state.sample_rate = rate_hz
            return True

        def set_frequency(self, freq_hz):
            return True

        def close(self):
            pass

    return FakeHackRF()


class TestHardwareFilters(_WindowTestCase):
    def tearDown(self):
        if HAS_PYQT6:
            self.win._device = None
        super().tearDown()

    def test_hackrf_filter_follows_the_sample_rate_not_the_channel(self):
        device = _fake_hackrf(10e6)
        self.win._device = device
        self.win._apply_controls_to_device()  # as on connect
        self.assertEqual(device.bandwidths, [7.5e6])
        self.win._control_panel._bw_combo.setCurrentText("25 kHz")
        self.win._control_panel._bw_combo.setCurrentText("200 kHz")
        self.assertEqual(device.bandwidths, [7.5e6])  # the channel isn't sent
        self.win.set_sample_rate(20e6)
        self.assertEqual(device.bandwidths[-1], 15e6)

    def test_no_hardware_driver_gets_the_channel_width(self):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        mw = self.mw
        self.assertFalse(mw.SDRMainWindow._forwards_channel_width(_fake_hackrf()))
        self.assertFalse(
            mw.SDRMainWindow._forwards_channel_width(RTLSDRDevice.__new__(RTLSDRDevice))
        )

    def test_hackrf_driver_rounds_three_quarters_of_the_rate(self):
        from sdr_module.devices.hackrf import HackRFDevice

        widths = HackRFDevice.SUPPORTED_BANDWIDTHS
        for rate, expected in ((2.4e6, 1.75e6), (8e6, 6e6), (20e6, 15e6)):
            chosen = min(widths, key=lambda w, r=rate: abs(w - 0.75 * r))
            self.assertEqual(chosen, expected)


# ---------------------------------------------------------------------------
# 3. Broadcast FM: de-emphasis and no stereo pilot
# ---------------------------------------------------------------------------


class TestFmAudio(unittest.TestCase):
    RATE = 2.4e6

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def _level_db(self, audio, freq, amplitude):
        settled = audio[int(0.03 * 48000) :]
        return 20 * np.log10(_tone_level(settled, freq, 48000) / amplitude)

    def _tone(self, freq, deviation, amplitude, bandwidth, pre_emphasis):
        rate = self.RATE
        t = np.arange(int(0.2 * rate)) / rate
        audio = amplitude * np.sin(2 * np.pi * freq * t)
        if pre_emphasis:
            audio = _pre_emphasis(audio, rate)
        chain = self.mw._ReceiverChain(rate, "FM", bandwidth)
        out = _receive(chain, _fm(audio, rate, deviation), deviation)
        return self._level_db(out, freq, amplitude)

    def test_pre_emphasized_broadcast_audio_comes_out_flat(self):
        levels = [
            self._tone(f, 75e3, 0.15, 200e3, pre_emphasis=True)
            for f in (1e3, 5e3, 10e3, 15e3)
        ]
        for level in levels:
            self.assertAlmostEqual(level, 0.0, delta=0.5)
        # Without pre-emphasis the treble is cut, as the de-emphasis should.
        self.assertLess(self._tone(10e3, 75e3, 0.15, 200e3, False), -12.0)

    def test_stereo_pilot_is_removed(self):
        rate = self.RATE
        t = np.arange(int(0.2 * rate)) / rate
        # The pilot is added after the program's pre-emphasis.
        audio = _pre_emphasis(0.5 * np.sin(2 * np.pi * 1e3 * t), rate)
        audio += 0.1 * np.sin(2 * np.pi * 19e3 * t)
        chain = self.mw._ReceiverChain(rate, "FM", 200e3)
        out = _receive(chain, _fm(audio, rate, 75e3), 75e3)
        self.assertLess(self._level_db(out, 19e3, 0.1), -30.0)
        self.assertAlmostEqual(self._level_db(out, 1e3, 0.5), 0.0, delta=0.5)

    def test_narrowband_fm_is_not_de_emphasized(self):
        for freq in (1e3, 3e3):
            level = self._tone(freq, 5e3, 0.5, 25e3, pre_emphasis=False)
            self.assertAlmostEqual(level, 0.0, delta=0.5)

    def test_switching_to_broadcast_deviation_turns_de_emphasis_on(self):
        chain = self.mw._ReceiverChain(self.RATE, "FM", 200e3)
        block = _fm(np.zeros(24000), self.RATE, 5e3)
        chain.demodulate(chain.channelize(block), 5e3)
        self.assertFalse(chain._deemphasis)
        chain.demodulate(chain.channelize(block), 75e3)
        self.assertTrue(chain._deemphasis)


# ---------------------------------------------------------------------------
# 4. Recordings go to a temporary file
# ---------------------------------------------------------------------------


class _RecordingCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-round4-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = self.folder

    def _record(self, frames=3):
        self.win._record_button.click()
        for _ in range(frames):
            self.win._update_display()
        self.win._record_button.click()
        return self.win._samples_buffer

    def _temp_files(self):
        return sorted(
            name for name in os.listdir(self.folder) if name.endswith(".cf32")
        )


class TestRecordingOnDisk(_RecordingCase):
    def test_samples_stream_to_a_temporary_file(self):
        store = self._record()
        self.assertTrue(store)
        self.assertIsNotNone(store.path)
        self.assertEqual(os.path.dirname(store.path), self.folder)
        self.assertEqual(os.path.getsize(store.path), store.sample_count * 8)
        self.assertIsNone(store._memory)  # nothing kept in memory
        whole = np.concatenate(list(store))
        np.testing.assert_array_equal(whole, store.samples())
        self.assertEqual(self.win._buffer_sample_count(), len(whole))
        self.assertIn(
            f"{len(whole):,} samples", self.win._message_label.text()
        )  # "Recorded N samples. Use File > Save Recording ..."

    def test_saving_cf32_moves_the_file(self):
        store = self._record()
        samples = np.array(store.samples())
        target = os.path.join(self.folder, "saved", "take.cf32")
        os.makedirs(os.path.dirname(target))
        with mock.patch.object(
            self.mw.QFileDialog, "getSaveFileName", return_value=(target, "")
        ):
            self.assertTrue(self.win._save_recording())
        self.assertEqual(self._temp_files(), [])  # moved, not copied
        np.testing.assert_array_equal(np.fromfile(target, np.complex64), samples)
        # Still the buffer, and closing leaves the saved file alone.
        np.testing.assert_array_equal(self.win._samples_buffer.samples(), samples)
        self.win.close()
        self.assertTrue(os.path.exists(target))

    def test_saving_while_recording_copies(self):
        self.win._record_button.click()
        self.win._update_display()
        target = os.path.join(self.folder, "partial.cf32")
        with mock.patch.object(
            self.mw.QFileDialog, "getSaveFileName", return_value=(target, "")
        ):
            self.assertTrue(self.win._save_recording())
        self.assertEqual(len(self._temp_files()), 2)  # the take goes on
        count = self.win._buffer_sample_count()
        self.win._update_display()
        self.assertGreater(self.win._buffer_sample_count(), count)
        self.assertEqual(os.path.getsize(target), count * 8)
        self.win._record_button.click()

    def test_other_formats_are_written_a_chunk_at_a_time(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        store = self._record()
        whole = np.array(store.samples())
        freq, rate = self.win._buffer_meta
        for name, fmt, sample_fmt in (
            ("take.cs16", FileFormat.RAW, SampleFormat.INT16),
            ("take.wav", FileFormat.WAV, SampleFormat.INT16),
        ):
            target = os.path.join(self.folder, name)
            reference = os.path.join(self.folder, "ref-" + name)
            with (
                mock.patch.object(self.mw, "_SAVE_CHUNK", 1000),
                mock.patch.object(
                    self.mw.QFileDialog, "getSaveFileName", return_value=(target, "")
                ),
            ):
                self.assertTrue(self.win._save_recording())
            save_iq_file(reference, whole, rate, freq, sample_fmt, fmt)
            with open(target, "rb") as a, open(reference, "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_sigmf_keeps_its_metadata(self):
        import json

        self._record()
        target = os.path.join(self.folder, "take.sigmf-data")
        with mock.patch.object(
            self.mw.QFileDialog, "getSaveFileName", return_value=(target, "")
        ):
            self.assertTrue(self.win._save_recording())
        with open(os.path.join(self.folder, "take.sigmf-meta")) as handle:
            meta = json.load(handle)
        self.assertEqual(meta["global"]["core:datatype"], "cf32_le")
        self.assertAlmostEqual(meta["captures"][0]["core:frequency"], self.freq())

    def test_discarding_and_closing_delete_the_temporary_file(self):
        self._record()
        self.assertEqual(len(self._temp_files()), 1)
        self.win._discard_recording()
        self.assertEqual(self._temp_files(), [])
        self._record()
        self.assertEqual(len(self._temp_files()), 1)
        self.win.close()  # the harness answers "Don't Save"
        self.assertEqual(self._temp_files(), [])

    def test_a_new_take_replaces_the_old_file(self):
        self._record()
        first = self._temp_files()
        self.win._buffer_unsaved = False
        self._record()
        second = self._temp_files()
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second)

    def test_import_still_loads_into_memory(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        samples = (np.arange(5000) * (1 + 1j) / 5000).astype(np.complex64)
        path = os.path.join(self.folder, "import.cs16")
        save_iq_file(path, samples, 1e6, 433.92e6, SampleFormat.INT16, FileFormat.RAW)
        with (
            mock.patch.object(
                self.mw.QFileDialog, "getOpenFileName", return_value=(path, "")
            ),
            mock.patch.object(self.mw.QMessageBox, "information"),
        ):
            self.win._open_recording()
        store = self.win._samples_buffer
        self.assertIsNone(store.path)
        self.assertEqual(store.sample_count, len(samples))
        target = os.path.join(self.folder, "converted.cf32")
        with mock.patch.object(
            self.mw.QFileDialog, "getSaveFileName", return_value=(target, "")
        ):
            self.assertTrue(self.win._save_recording())
        np.testing.assert_allclose(
            np.fromfile(target, np.complex64), samples, atol=1e-4
        )


class TestLowDiskSpace(_RecordingCase):
    Usage = namedtuple("Usage", "total used free")

    def _free(self, megabytes):
        usage = self.Usage(10**12, 0, int(megabytes * 1024**2))
        return mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage)

    def test_recording_stops_with_a_warning_when_the_disk_fills(self):
        self.win._record_button.click()
        self.win._update_display()
        count = self.win._buffer_sample_count()
        self.assertGreater(count, 0)
        self.win._next_space_check = 0.0
        with self._free(420):
            self.win._update_display()
        self.assertFalse(self.win._recording)
        self.assertFalse(self.win._record_button.isChecked())
        self.assertFalse(self.win._control_panel._record_btn.isChecked())
        text = self.win._message_label.text()
        self.assertIn("Recording stopped", text)
        self.assertIn("420 MB of disk space", text)
        self.assertIn("Save Recording", text)
        self.assertEqual(self.win._message_label.property("tone"), "danger")
        self.assertEqual(self.win._buffer_sample_count(), count)  # kept

    def test_recording_does_not_start_on_a_full_disk(self):
        with self._free(100):
            self.win._record_button.click()
        self.assertFalse(self.win._recording)
        self.assertFalse(self.win._record_button.isChecked())
        self.assertIn("Can't record", self.win._message_label.text())

    def test_write_errors_stop_the_recording(self):
        self.win._record_button.click()
        self.win._update_display()
        with mock.patch.object(
            self.mw._SampleStore, "append", side_effect=OSError(28, "No space left")
        ):
            self.win._update_display()
        self.assertFalse(self.win._recording)
        self.assertIn("can't be written", self.win._message_label.text())

    def test_status_shows_the_space_on_the_recording_drive(self):
        self.win._record_button.click()
        with self._free(2.5 * 1024) as usage:
            self.win._update_status()
        usage.assert_called_with(self.folder)
        self.assertIn("2.5 GB free", self.win._recording_info_label.text())
        self.win._record_button.click()


class TestStaleRecordings(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()
        self.folder = tempfile.mkdtemp(prefix="sdr-round4-stale-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_leftovers_are_removed_but_live_files_and_others_kept(self):
        mw = self.mw
        stale = os.path.join(self.folder, "sdr-recording-99999999-dead.cf32")
        other = os.path.join(self.folder, "notes.cf32")
        for path in (stale, other):
            with open(path, "wb") as handle:
                handle.write(b"\0" * 64)
        live = mw._SampleStore(self.folder)
        with mock.patch.object(mw, "_cleaned_folders", set()):
            live.append(np.ones(16, np.complex64))  # cleans up on first use
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(other))
        self.assertTrue(os.path.exists(live.path))
        if mw.fcntl is not None:  # POSIX: the live file is locked
            self.assertEqual(mw._remove_stale_recordings(self.folder), 0)
            self.assertTrue(os.path.exists(live.path))
        path = live.path
        live.discard()
        self.assertFalse(os.path.exists(path))

    def test_a_forgotten_store_deletes_its_file(self):
        store = self.mw._SampleStore(self.folder)
        store.append(np.ones(16, np.complex64))
        path = store.path
        del store
        import gc

        gc.collect()
        self.assertFalse(os.path.exists(path))


# ---------------------------------------------------------------------------
# 5. Tuning picks a channel width for the mode
# ---------------------------------------------------------------------------


class TestTuningBandwidth(_WindowTestCase):
    def bandwidth(self):
        return self.win._control_panel._bw_combo.currentText()

    def test_decoder_suggestions_get_a_narrow_channel(self):
        decoder = self.win._decoder_panel
        decoder.tune_requested.emit(144.39e6, "AX.25/APRS (144.390 MHz)", "FM", "5 kHz")
        self.assertEqual(self.bandwidth(), "25 kHz")
        self.assertEqual(self.win._control_panel.get_fm_deviation_text(), "5 kHz")
        decoder.tune_requested.emit(131.55e6, "ACARS (131.550 MHz)", "AM", "")
        self.assertEqual(self.bandwidth(), "10 kHz")
        self.assertEqual(self.win._control_panel.get_demod_mode(), "AM")

    def test_bookmark_modes(self):
        cases = (
            ("FM", "75 kHz", "200 kHz"),
            ("FM", "25 kHz", "50 kHz"),
            ("FM", "12.5 kHz", "25 kHz"),
            ("FM", "2.5 kHz", "10 kHz"),
            ("AM", "", "10 kHz"),
            ("USB", "", "10 kHz"),
            ("LSB", "", "10 kHz"),
            ("CW", "", "10 kHz"),
        )
        for mode, deviation, expected in cases:
            self.win._control_panel._bw_combo.setCurrentText("2 MHz")
            self.win._on_bookmark_tune(146.52e6, "Test", mode, deviation)
            self.assertEqual(self.bandwidth(), expected, (mode, deviation))

    def test_fm_without_a_deviation_uses_the_current_one(self):
        self.win._set_demod_mode("FM", "75 kHz", "10 kHz")
        self.win._on_bookmark_tune(98.1e6, "Station", "FM", "")
        self.assertEqual(self.bandwidth(), "200 kHz")

    def test_no_mode_or_iq_keeps_the_bandwidth(self):
        self.win._control_panel._bw_combo.setCurrentText("2 MHz")
        self.win._on_bookmark_tune(433.92e6, "Old bookmark", "", "")
        self.assertEqual(self.bandwidth(), "2 MHz")
        self.win._on_bookmark_tune(1090e6, "ADS-B", "None (I/Q)", "")
        self.assertEqual(self.bandwidth(), "2 MHz")


# ---------------------------------------------------------------------------
# 6. The receiver chain at awkward sample rates
# ---------------------------------------------------------------------------


class TestAwkwardRates(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    @staticmethod
    def _smooth(n):
        for p in (2, 3, 5):
            while n % p == 0:
                n //= p
        return n == 1

    def test_decimation_is_in_small_factors_and_cheap_at_full_rate(self):
        for rate in (2.048e6, 2.4e6, 3.2e6, 8e6, 10e6, 20e6):
            for mode, bandwidth in (("FM", 200e3), ("FM", 25e3), ("USB", 10e3)):
                chain = self.mw._ReceiverChain(rate, mode, bandwidth)
                factor = rate / chain.intermediate_rate
                self.assertAlmostEqual(factor, round(factor))
                self.assertTrue(self._smooth(round(factor)), (rate, factor))
                self.assertGreaterEqual(chain.intermediate_rate, 48e3)
                self.assertLess(chain.intermediate_rate, 64e3)
                self.assertEqual(chain.audio_rate, 48000.0)
                first = chain._channel[0]
                if rate >= 8e6:  # only the kept outputs, few taps each
                    self.assertTrue(first._groups, (rate, mode))
                    self.assertLessEqual(first._groups, 10, (rate, mode))

    def test_output_is_exactly_48_khz(self):
        for rate in (8e6, 20e6, 2.048e6):
            chain = self.mw._ReceiverChain(rate, "FM", 200e3)
            rng = np.random.default_rng(4)
            total = 0
            fed = 0
            while fed < 2 * rate:
                size = int(rng.integers(20_000, 400_000))
                iq = np.ones(size, dtype=np.complex64)
                total += len(chain.demodulate(chain.channelize(iq), 75e3))
                fed += size
            self.assertAlmostEqual(total, fed / rate * 48000, delta=64, msg=rate)

    def test_tones_keep_their_pitch_and_blocks_join_seamlessly(self):
        rate = 8e6
        t = np.arange(int(0.25 * rate)) / rate
        iq = _fm(0.5 * np.sin(2 * np.pi * 1e3 * t), rate, 5e3)
        one = self.mw._ReceiverChain(rate, "FM", 25e3)
        whole = one.demodulate(one.channelize(iq), 5e3)
        split = self.mw._ReceiverChain(rate, "FM", 25e3)
        cuts = [0, 7, 100_003, 100_004, 777_777, 1_500_000, len(iq)]
        parts = np.concatenate(
            [
                split.demodulate(split.channelize(iq[a:b]), 5e3)
                for a, b in zip(cuts[:-1], cuts[1:], strict=True)
            ]
        )
        self.assertEqual(len(parts), len(whole))
        # (Past the first few ms, where the filters' start-up leaves a
        # near-zero signal whose phase is rounding noise.)
        np.testing.assert_allclose(parts[480:], whole[480:], atol=1e-4)
        settled = whole[2400:]
        spectrum = np.abs(np.fft.rfft(settled * np.hanning(len(settled))))
        peak = np.fft.rfftfreq(len(settled), 1 / 48000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 1000, delta=5)

    def test_ssb_and_am_resample_too(self):
        rate = 10e6
        t = np.arange(int(0.2 * rate)) / rate
        tone = np.exp(2j * np.pi * 1000 * t).astype(np.complex64)
        chain = self.mw._ReceiverChain(rate, "USB", 10e3)
        audio = _receive(chain, tone, 5e3)[4800:]
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        peak = np.fft.rfftfreq(len(audio), 1 / 48000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 1000, delta=10)
        am = (1 + 0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.complex64)
        chain = self.mw._ReceiverChain(rate, "AM", 10e3)
        audio = _receive(chain, am, 5e3)
        self.assertAlmostEqual(_tone_level(audio[4800:], 800, 48000), 0.5, delta=0.05)

    def test_polyphase_matches_filtering_every_sample(self):
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(30_011) + 1j * rng.standard_normal(30_011)).astype(
            np.complex64
        )
        taps = np.hanning(61).astype(np.float32)
        fast = self.mw._FirDecimator(taps, 15)
        self.assertTrue(fast._groups)
        slow = self.mw._FirDecimator(taps, 15)
        slow._groups = 0  # FFT overlap-save, then every 15th sample
        a = np.concatenate(
            [fast.process(x[i : i + 997]) for i in range(0, len(x), 997)]
        )
        b = slow.process(x)
        np.testing.assert_allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. The passband on both plots
# ---------------------------------------------------------------------------


class TestPassbandOnBothPlots(_WindowTestCase):
    def test_waterfall_marks_the_channel_like_the_spectrum(self):
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.assertEqual(self.win._spectrum._passband, (-12500.0, 12500.0))
        self.assertEqual(self.win._waterfall._passband, (-12500.0, 12500.0))
        self.win._set_demod_mode("USB")
        self.assertEqual(self.win._waterfall._passband, (0.0, 2800.0))
        self.assertEqual(self.win._spectrum._passband, (0.0, 2800.0))


# ---------------------------------------------------------------------------
# 8. ADS-B squawk codes
# ---------------------------------------------------------------------------


class TestAdsbSquawk(_WindowTestCase):
    def test_squawk_is_shown(self):
        from sdr_module.dsp.protocols import ADSBMessage, ProtocolType

        describe = self.mw.SDRMainWindow._describe_message
        base = dict(protocol=ProtocolType.ADSB, timestamp=0.0, raw_bits=b"", valid=True)
        address, content = describe(
            ADSBMessage(**base, icao_address="4840D6", squawk="7700")
        )
        self.assertEqual(address, "4840D6")
        self.assertEqual(content, "Squawk 7700")
        _address, content = describe(
            ADSBMessage(
                **base, icao_address="4840D6", callsign="KLM1023", altitude=38000
            )
        )
        self.assertEqual(content, "KLM1023  ·  38,000 ft")

    def test_demo_aircraft_is_decoded_with_its_squawk(self):
        clock = _FakeClock()
        rows = []
        panel = self.win._decoder_panel
        panel.add_message = lambda *args, **kwargs: rows.append(args)
        with mock.patch.object(self.mw, "time", clock):
            self.win._start_demo_mode()
            panel._proto_combo.setCurrentText("ADS-B")
            panel._tune_to_protocol()
            for _ in range(150):  # five seconds
                self.win._update_display()
                clock.advance(0.033)
        self.win._stop_acquisition()
        content = " ".join(str(row[2]) for row in rows)
        self.assertIn("KLM1023", content)
        self.assertIn("Squawk", content)


if __name__ == "__main__":
    unittest.main()
