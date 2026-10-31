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

from tests.test_gui_chrome import require_pyqt6  # noqa: E402

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
                want = full[: n : factor]
                self.assertEqual(len(got), len(want), (taps_n, factor))
                np.testing.assert_allclose(
                    got, want, atol=2e-5, err_msg=str((taps_n, factor, complex_))
                )
