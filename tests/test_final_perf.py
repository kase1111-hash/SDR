#!/usr/bin/env python3
"""
The display loop's cost at high sample rates (the final perf fixes).

At 20 MS/s (a HackRF) a display frame carries 666,667 samples every 33 ms.
The end-to-end checks found broadcast FM, None (I/Q) and the Decoder's RDS
taking more than that frame's time on a 4-core computer: the channel filter
computed each kept output in several passes (or filtered every sample by
FFT and kept one in five), the Decoder's RDS picked the station out of the
full-rate block a second time, and the plots transformed every sample even
of a catch-up frame several times longer. These tests pin the work each
part does (counts, not timings: they hold on a busy computer too).

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_perf.py``
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import _WindowTestCase, require_pyqt6  # noqa: E402

RATE = 20e6
FRAME = 666_667  # one display frame at 20 MS/s


def _mw():
    from sdr_module.gui import main_window

    return main_window


def _noise(n, seed=1):
    rng = np.random.default_rng(seed)
    return (0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )


class _CountDots:
    """Counts the dot products (rows) ``module._row_dot`` computes."""

    def __init__(self, module):
        self.calls = 0
        self.rows = 0
        self._module = module
        self._orig = module._row_dot

    def __enter__(self):
        def spy(rows, weights):
            self.calls += 1
            self.rows += int(np.asarray(rows).shape[0])
            return self._orig(rows, weights)

        self._patch = mock.patch.object(self._module, "_row_dot", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


# ---------------------------------------------------------------------------
# LISTEN-06, PLOTS-E2E-5, HAM-E2E-3: the receiver's channel filter
# ---------------------------------------------------------------------------


class TestReceiverChannelFilterCost(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def test_broadcast_fm_at_20_msps_computes_each_kept_output_once(self):
        # The 200 kHz channel's first stage (67 taps, decimating by 15) was
        # computed in five passes of short rows (one per group of 15 taps):
        # 7-9 ms of each 33 ms frame. Each kept output is one dot product.
        chain = self.mw._ReceiverChain(RATE, "FM", 200e3)
        outputs = 0
        with _CountDots(self.mw) as dots:
            x = _noise(FRAME)
            for stage in chain._channel:
                x = stage.process(x)
                if stage._groups:
                    outputs += len(x)
        self.assertGreater(outputs, 0)
        self.assertEqual(dots.rows, outputs)
        self.assertEqual(dots.calls, sum(1 for s in chain._channel if s._groups))

    def test_no_channel_at_20_msps_filters_every_sample_by_fft(self):
        # None (I/Q) with its 2.4 MHz filter decimates by 5: the 243-tap
        # filter ran as an FFT over all 666,667 samples and kept one output
        # in five (15-29 ms a frame, so I/Q at 20 MS/s could not keep up).
        for mode, width in (("None (I/Q)", 2.4e6), ("FM", 200e3), ("FM", 25e3)):
            chain = self.mw._ReceiverChain(RATE, mode, width)
            first = chain._channel[0]
            self.assertGreater(first._factor, 1, mode)
            self.assertTrue(first._groups, (mode, width))

    def test_kept_outputs_match_filtering_every_sample(self):
        # The direct form gives what filtering every sample by FFT and
        # keeping every factor-th one gives, for complex and real streams
        # cut into blocks of any size.
        rng = np.random.default_rng(3)
        for taps_n, factor in ((67, 15), (243, 5), (15, 2), (195, 3), (607, 135)):
            taps = np.hanning(taps_n + 2)[1:-1].astype(np.float32)
            taps /= taps.sum()
            for complex_ in (True, False):
                n = 50_021
                x = rng.standard_normal(n).astype(np.float32)
                if complex_:
                    x = (x + 1j * rng.standard_normal(n)).astype(np.complex64)
                fast = self.mw._FirDecimator(taps, factor)
                fast._groups = fast._groups or 1  # (the direct form)
                slow = self.mw._FirDecimator(taps, factor)
                slow._groups = 0  # FFT overlap-save, then every factor-th
                cuts = np.unique(
                    np.concatenate(([0, n], rng.integers(0, n, 9), [1, 2, 3]))
                )
                a = np.concatenate(
                    [
                        fast.process(x[i:j])
                        for i, j in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                b = slow.process(x)
                self.assertEqual(len(a), len(b), (taps_n, factor, complex_))
                self.assertEqual(a.dtype, b.dtype)
                np.testing.assert_allclose(
                    a, b, atol=2e-5, err_msg=str((taps_n, factor, complex_))
                )


# ---------------------------------------------------------------------------
# Leftover 6, PLOTS-E2E-12: the decoders' and Identify's channel
# ---------------------------------------------------------------------------


def _channelizer():
    from sdr_module.dsp import channelizer

    return channelizer


class TestDecoderChannelCost(unittest.TestCase):
    def _counted(self, chan, x):
        """``(dot products, kept outputs)`` of one block through ``chan``."""
        module = _channelizer()
        outputs = 0
        with _CountDots(module) as dots:
            y = chan._nco.process(x)
            for stage in chan._stages:
                y = stage.process(y)
                outputs += len(y)
        return dots.rows, outputs

    def test_identify_at_20_msps_computes_each_kept_output_once(self):
        # Identify on a 2.4 MHz I/Q channel at 20 MS/s took 32-115 ms of
        # each 33 ms block (each of its taps a strided pass over the
        # block), and 16 ms on a broadcast FM channel: each kept output is
        # now one dot product.
        from sdr_module.dsp.signal_id import SignalIdentifier, analysis_band

        for mode, width in (("None (I/Q)", 2.4e6), ("FM", 200e3), ("FM", 25e3)):
            band, _reach = analysis_band(mode, width, RATE)
            chan = SignalIdentifier(RATE, 150e3, band)._channel
            self.assertIsNotNone(chan, mode)
            rows, outputs = self._counted(chan, _noise(FRAME))
            self.assertEqual(rows, outputs, (mode, width))

    def test_live_decoders_at_20_msps_compute_each_kept_output_once(self):
        from sdr_module.dsp.live_decoders import LIVE_DECODERS, ADSBLiveDecoder

        for name, cls in LIVE_DECODERS.items():
            if cls is ADSBLiveDecoder:
                continue  # (no channel: the pulses at the full rate)
            chan = cls(RATE, 25e3)._channelizer
            rows, outputs = self._counted(chan, _noise(FRAME))
            self.assertEqual(rows, outputs, name)

    def test_stages_match_the_filter_computed_tap_by_tap(self):
        # Each stage's kept outputs are those of the plain convolution, for
        # complex and real streams cut into blocks of any size.
        module = _channelizer()
        rng = np.random.default_rng(5)
        for taps_n, factor in ((19, 3), (85, 2), (567, 135), (15, 1), (5, 2)):
            taps = rng.standard_normal(taps_n)
            taps = (taps / np.sum(np.abs(taps))).astype(np.float32)
            for complex_ in (True, False):
                n = 20_011
                x = rng.standard_normal(n).astype(np.float32)
                if complex_:
                    x = (x + 1j * rng.standard_normal(n)).astype(np.complex64)
                stage = module.FirDecimator(taps, factor)
                cuts = np.unique(
                    np.concatenate(([0, n], rng.integers(0, n, 9), [1, 2, 3]))
                )
                got = np.concatenate(
                    [
                        stage.process(x[i:j])
                        for i, j in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                full = np.convolve(x.astype(np.complex128), taps.astype(np.float64))
                want = full[:n:factor]
                self.assertEqual(len(got), len(want), (taps_n, factor))
                np.testing.assert_allclose(
                    got, want, atol=2e-5, err_msg=str((taps_n, factor, complex_))
                )


# ---------------------------------------------------------------------------
# Leftover 6, DEV-E2E-19: the plots' spectrum of a long frame
# ---------------------------------------------------------------------------


def _tone(n, freq, rate, amplitude=1.0):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


class TestFrameSpectrumWorkIsBounded(unittest.TestCase):
    CAP = 1 << 19

    def test_a_long_frame_is_analysed_in_evenly_spread_segments(self):
        # Every FFT segment of every frame: 325 at 20 MS/s (13-17 ms a
        # frame), 1024 for a 2 M-sample catch-up frame after a stall
        # (60-330 ms: a visible hitch that made the next frame later
        # still). With a cap, a long frame is analysed in evenly spread
        # segments, the newest included.
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        size = 2048
        n = 1 << 21
        data = _noise(n, seed=4) * np.float32(0.01) + _tone(n, 3e6, RATE, 0.1)
        whole = FrameSpectrumAnalyzer(size, threads=1).process(data)
        capped = FrameSpectrumAnalyzer(size, threads=1, max_samples=self.CAP)
        spread = capped.process(data)
        self.assertEqual(whole.segments, n // size)
        self.assertLessEqual(spread.segments, self.CAP // size)
        self.assertGreaterEqual(spread.segments, self.CAP // size // 2)
        # The same spectrum: the tone and the noise floor read the same.
        b = size // 2 + int(round(3e6 / (RATE / size)))
        self.assertAlmostEqual(
            float(spread.mean_db[b]), float(whole.mean_db[b]), delta=0.2
        )
        self.assertAlmostEqual(spread.noise_floor_db, whole.noise_floor_db, delta=0.5)

    def test_the_newest_segment_and_longer_bursts_still_show(self):
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        size = 2048
        n = 1 << 21
        burst = _tone(4 * size, -2e6, RATE, 0.1)
        b = size // 2 + int(round(-2e6 / (RATE / size)))
        for start in (0, 777_777, n - 4 * size, n - size):
            data = _noise(n, seed=6) * np.float32(0.001)
            stop = min(n, start + len(burst))
            data[start:stop] += burst[: stop - start]
            a = FrameSpectrumAnalyzer(size, threads=1, max_samples=self.CAP)
            spec = a.process(data)
            floor = float(np.median(spec.max_db))
            self.assertGreater(float(spec.max_db[b]), floor + 30, start)

    def test_frames_within_the_cap_are_analysed_whole(self):
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        a = FrameSpectrumAnalyzer(2048, threads=1, max_samples=self.CAP)
        self.assertEqual(a.max_samples, self.CAP)
        self.assertEqual(a.process(_noise(80_000)).segments, 39)  # 2.4 MS/s
        self.assertEqual(a.process(_noise(333_334)).segments, 162)  # 10 MS/s
        a.max_samples = None
        self.assertEqual(a.process(_noise(1 << 21)).segments, 1024)


class TestPlotsInTheWindow(_WindowTestCase):
    demo = True

    def test_a_catch_up_frame_costs_the_plots_no_more_than_a_usual_one(self):
        analyzer = self.win._frame_spectrum
        self.assertEqual(analyzer.max_samples, self.mw._PLOT_MAX_SAMPLES)
        frames = []
        process = analyzer.process

        def spy(samples):
            frame = process(samples)
            frames.append(frame)
            return frame

        analyzer.process = spy
        self.win._plot_frame(_noise(80_000))  # 2.4 MS/s: every sample
        self.win._plot_frame(_noise(1 << 21))  # a stall's backlog at 20 MS/s
        self.assertEqual(frames[0].segments, 80_000 // 2048)
        self.assertLessEqual(
            frames[1].segments, self.mw._PLOT_MAX_SAMPLES // analyzer.size
        )


# ---------------------------------------------------------------------------
# ROB-06: the Decoder's RDS and the receiver's broadcast FM channel
# ---------------------------------------------------------------------------


class TestRdsFromTheReceiversChannel(unittest.TestCase):
    def test_a_raw_decoder_decodes_a_channel_handed_to_it(self):
        from sdr_module.dsp.live_decoders import RDSLiveDecoder, describe_message
        from tests.test_rds import _wfm

        # The receiver's broadcast FM channel at 20 MS/s: 444 kS/s at 0 Hz.
        decoder = RDSLiveDecoder(RATE, 0.0)
        rate = RATE / 45
        x = _wfm(rate, 2.5, 0.0)
        own = mock.Mock(wraps=decoder._channelizer.process)
        decoder._channelizer.process = own
        messages = []
        block = int(rate / 30)
        for i in range(0, len(x), block):
            messages += decoder.process_channel(x[i : i + block], rate)
        rows = [describe_message(m) for m in messages]
        self.assertIn(("PI C0DE", "Station name: SDR DEMO"), rows)
        self.assertEqual(own.call_count, 0)
        status = decoder.status
        self.assertEqual(status.channel_rate, rate)
        self.assertGreater(status.valid, 0)
        self.assertAlmostEqual(status.stream_s, len(x) / rate, places=3)
        # Raw blocks again: its own channel, at its own rate (the station
        # it knows stays known).
        decoder.process(np.zeros(int(RATE / 30), dtype=np.complex64))
        self.assertEqual(own.call_count, 1)
        self.assertEqual(decoder.status.channel_rate, 250e3)
        self.assertEqual(decoder.station.ps_name, "SDR DEMO")


class TestDecoderRdsInTheWindow(_WindowTestCase):
    demo = True

    def test_the_decoders_rds_decodes_the_receivers_fm_channel(self):
        # ROB-06: with the receiver on broadcast FM the Decoder's RDS
        # picked the same station out of the raw block a second time (8-14
        # ms of each 33 ms frame at 20 MS/s), where the background RDS
        # took 1.2 ms from the receiver's own channel.
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.win._decoder_panel._proto_combo.setCurrentText("RDS")
        self.settle(1)
        decoder = self.win._decoder
        own = mock.Mock(wraps=decoder._channelizer.process)
        decoder._channelizer.process = own
        for _ in range(300):  # (up to 10 s of the demo station)
            self.win._process_block(self.win._device.read_samples(80000))
            if decoder.station.ps_complete:
                break
        self.assertTrue(decoder.station.ps_complete)
        self.assertEqual(own.call_count, 0)
        self.assertTrue(self.win._decoder_panel.count_for("RDS"))
        # A narrow FM channel can't carry the station's RDS: the decoder
        # picks the station out of the raw block itself again.
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.win._process_block(self.win._device.read_samples(80000))
        self.assertEqual(own.call_count, 1)


# ---------------------------------------------------------------------------
# DEC-E2E-03: a decoder that gets only part of the signal says so
# ---------------------------------------------------------------------------


def _hardware_case():
    from tests.test_review_mainloop import _HardwareCase

    return _HardwareCase


class TestDecoderSaysWhenSignalIsSkipped(_hardware_case()):
    """A HackRF at 20 MS/s on a computer whose frames take 1.5x real time
    (the pacer reads one transfer a frame; the driver drops the rest)."""

    def _decode(self, protocol):
        self.win._set_demod_mode("None (I/Q)", bandwidth="2 MHz")
        self.win._decoder_panel._proto_combo.setCurrentText(protocol)
        self.settle(1)
        self.win._decoder_panel.set_receiving(True)

    def test_the_decoder_status_says_it_misses_part_of_the_signal(self):
        # ADS-B's Tune to puts the receiver in None (I/Q): no speaker, so
        # the "can't keep up" hint never came, and the status line stayed
        # green while 72 % of the samples were dropped.
        self._decode("ADS-B")
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)
        panel = self.win._decoder_panel
        line = panel.decoder_status_text()
        self.assertIn("misses part of the signal", line)
        self.assertIn("20 MS/s", line)
        self.assertIn("Sample rate", line)
        self.assertEqual(panel._status_label.property("tone"), "warning")
        self.assertIn("misses part of the signal", self.win._message_label.text())
        # The computer keeps up again: the line is the decoder's own.
        self.LOAD = 0.3
        self.run_frames(300)
        self.assertFalse(self.win._pacer.overloaded)
        self.assertNotIn("misses", panel.decoder_status_text())

    def test_no_decoder_or_decoding_paused_says_nothing_of_it(self):
        self._decode("ADS-B")
        self.win._decoder_panel._enabled_check.setChecked(False)
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)
        self.assertNotIn("misses", self.win._decoder_panel.decoder_status_text())
        self.assertNotIn("misses", self.win._message_label.text())
