#!/usr/bin/env python3
"""
Tests for the whole-frame spectrum (``sdr_module.dsp.frame_spectrum``): dBFS
levels for every FFT size and window, the mean and per-bin maximum over a
frame's segments, samples carried between frames, short frames, and the
threaded transform giving the same result.
"""

import math
import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp import frame_spectrum as fs

RATE = 2.4e6


def tone(n, freq, amplitude=1.0, start=0):
    t = (np.arange(n) + start) / RATE
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def noise(n, power_db, seed=0):
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(10 ** (power_db / 10) / 2)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex64)


class TestHelpers(unittest.TestCase):
    def test_windows_and_their_noise_bandwidths(self):
        self.assertAlmostEqual(fs.enbw_bins("hann"), 1.5, places=3)
        self.assertAlmostEqual(fs.enbw_bins("blackman-harris"), 2.004, places=2)
        self.assertAlmostEqual(fs.enbw_bins("flat-top"), 3.77, places=2)
        for name in fs.WINDOWS:
            w = fs.make_window(name, 1024)
            self.assertEqual(w.dtype, np.float32)
            self.assertEqual(len(w), 1024)
            self.assertIn(name, fs.WINDOW_LABELS)
            self.assertIn(name, fs.WINDOW_HINTS)

    def test_rbw(self):
        self.assertAlmostEqual(fs.rbw_hz(RATE, 2048, "hann"), 1757.8, places=1)
        self.assertAlmostEqual(fs.rbw_hz(RATE, 32768, "hann"), 109.86, places=1)
        self.assertEqual(fs.rbw_hz(0, 2048), 0.0)

    def test_normalizers(self):
        self.assertEqual(fs.normalize_window("Blackman_Harris"), "blackman-harris")
        self.assertEqual(fs.normalize_window("flattop"), "flat-top")
        self.assertEqual(fs.normalize_window("nonsense"), "hann")
        self.assertEqual(fs.normalize_window(None), "hann")
        self.assertEqual(fs.normalize_fft_size(4000), 4096)
        self.assertEqual(fs.normalize_fft_size("8192"), 8192)
        self.assertEqual(fs.normalize_fft_size(10**9), 32768)
        self.assertEqual(fs.normalize_fft_size("junk"), 2048)
        self.assertEqual(fs.normalize_fft_size(-5), 2048)


class TestLevels(unittest.TestCase):
    def test_full_scale_tone_peaks_at_0_dbfs_for_every_size_and_window(self):
        for size in fs.FFT_SIZES:
            for window in fs.WINDOWS:
                a = fs.FrameSpectrumAnalyzer(size, window, threads=1)
                freq = 100 * RATE / size  # on a bin
                spec = a.process(tone(4 * size, freq))
                self.assertIsNotNone(spec)
                self.assertEqual(spec.size, size)
                peak = int(np.argmax(spec.mean_db))
                self.assertEqual(peak, size // 2 + 100, (size, window))
                self.assertAlmostEqual(float(spec.mean_db[peak]), 0.0, delta=0.05)
                self.assertAlmostEqual(float(spec.max_db[peak]), 0.0, delta=0.05)

    def test_flat_top_reads_a_tone_between_bins_exactly(self):
        a = fs.FrameSpectrumAnalyzer(2048, "flat-top", threads=1)
        freq = 100.5 * RATE / 2048  # half way between two bins
        spec = a.process(tone(8192, freq, amplitude=0.1))
        self.assertAlmostEqual(float(spec.mean_db.max()), -20.0, delta=0.05)
        hann = fs.FrameSpectrumAnalyzer(2048, "hann", threads=1)
        spec = hann.process(tone(8192, freq, amplitude=0.1))
        self.assertLess(float(spec.mean_db.max()), -21.0)  # scalloping loss

    def test_noise_per_bin_follows_the_noise_bandwidth(self):
        for window in fs.WINDOWS:
            a = fs.FrameSpectrumAnalyzer(2048, window, threads=1)
            spec = a.process(noise(400 * 2048, -40.0, seed=3))
            expected = -40.0 + 10 * math.log10(fs.enbw_bins(window) / 2048)
            mean_db = 10 * math.log10(float(np.mean(10 ** (spec.mean_db / 10))))
            self.assertAlmostEqual(mean_db, expected, delta=0.2, msg=window)
            # The mean of 400 segments is steady (one FFT scatters ~5.6 dB).
            self.assertLess(float(np.std(spec.mean_db)), 0.5)


class TestWholeFrame(unittest.TestCase):
    def test_a_burst_anywhere_in_the_frame_shows_in_the_maximum(self):
        size = 2048
        frame = noise(80000, -60.0, seed=5)
        burst = tone(2048, 300e3, amplitude=0.1)
        for start in (0, 30000, 80000 - 4096):
            data = frame.copy()
            data[start : start + 2048] += burst
            a = fs.FrameSpectrumAnalyzer(size, threads=1)
            spec = a.process(data)
            b = size // 2 + int(round(300e3 / (RATE / size)))
            noise_db = float(np.median(spec.max_db))
            self.assertGreater(float(spec.max_db[b - 2 : b + 3].max()), noise_db + 40)
            # The mean shows it, diluted by the 39 segments.
            self.assertGreater(float(spec.mean_db[b - 2 : b + 3].max()), noise_db + 15)
            self.assertEqual(spec.segments, 39)

    def test_every_sample_is_analysed_once(self):
        a = fs.FrameSpectrumAnalyzer(2048, threads=1)
        total = 0
        segments = 0
        rng = np.random.default_rng(2)
        for _ in range(50):
            n = int(rng.integers(2048, 9000))
            total += n
            segments += a.process(noise(n, -50.0)).segments
        # No sample is used twice and none is lost: what a frame leaves over
        # (less than one FFT) joins the next frame.
        self.assertEqual(segments, total // 2048)
        self.assertEqual(len(a._carry), total % 2048)

    def test_short_frames_wait_for_one_fft_then_use_the_newest_samples(self):
        a = fs.FrameSpectrumAnalyzer(8192, threads=1)
        results = [
            a.process(tone(1000, 50 * RATE / 8192, 0.5, start=i * 1000))
            for i in range(12)
        ]
        self.assertTrue(all(r is None for r in results[:8]))
        self.assertTrue(all(r is not None for r in results[9:]))
        peak = int(np.argmax(results[-1].mean_db))
        self.assertEqual(peak, 8192 // 2 + 50)
        self.assertAlmostEqual(float(results[-1].mean_db[peak]), -6.02, delta=0.1)

    def test_reset_and_configure_drop_old_samples(self):
        a = fs.FrameSpectrumAnalyzer(4096, threads=1)
        a.process(noise(5000, -50.0))
        self.assertGreater(len(a._carry), 0)
        a.reset()
        self.assertEqual(len(a._carry), 0)
        self.assertIsNone(a.process(noise(1000, -50.0)))
        self.assertTrue(a.configure(size=1024, window="flat-top"))
        self.assertFalse(a.configure(size=1024, window="flat-top"))
        self.assertEqual((a.size, a.window), (1024, "flat-top"))
        self.assertEqual(len(a._carry), 0)
        self.assertAlmostEqual(a.rbw(RATE), 3.77 * RATE / 1024, delta=5.0)

    def test_two_threads_give_the_same_spectrum(self):
        data = noise(700_000, -50.0, seed=9) + tone(700_000, 123e3, 0.01)
        one = fs.FrameSpectrumAnalyzer(2048, threads=1).process(data)
        two_analyzer = fs.FrameSpectrumAnalyzer(2048, threads=2)
        try:
            two = two_analyzer.process(data)
        finally:
            two_analyzer.close()
        np.testing.assert_array_equal(one.mean_db, two.mean_db)
        np.testing.assert_array_equal(one.max_db, two.max_db)
        self.assertEqual(one.segments, 341)

    def test_buffers_follow_the_frame_size(self):
        a = fs.FrameSpectrumAnalyzer(1024, threads=1)
        a.process(noise(100 * 1024, -50.0))
        big = a._rows
        a.process(noise(5 * 1024, -50.0))
        self.assertLess(a._rows, big)  # a much smaller frame frees memory
        spec = a.process(noise(3 * 1024 + 7, -50.0))
        self.assertEqual(spec.mean_db.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(spec.max_db)))

    def test_silence_is_finite(self):
        spec = fs.FrameSpectrumAnalyzer(2048, threads=1).process(
            np.zeros(4096, np.complex64)
        )
        self.assertTrue(np.all(spec.mean_db <= -200.0))
        self.assertTrue(np.all(np.isfinite(spec.mean_db)))


class TestOlderNumpy(unittest.TestCase):
    """NumPy 1.x, which the package still allows (numpy>=1.21), has no
    ``out`` argument to ``np.fft.fft``: the plots must work there too (a
    TypeError would stop every display frame, and the audio with it)."""

    def test_works_without_the_fft_out_argument(self):
        data = noise(200_000, -50.0, seed=4) + tone(200_000, 300e3, 0.01)
        reference = fs.FrameSpectrumAnalyzer(2048, threads=2)
        try:
            want = reference.process(data)
        finally:
            reference.close()
        numpy2_fft = np.fft.fft

        def numpy1_fft(a, n=None, axis=-1, norm=None):  # no out, double only
            return numpy2_fft(np.asarray(a, np.complex128), n, axis, norm)

        with mock.patch.object(np.fft, "fft", numpy1_fft):
            analyzer = fs.FrameSpectrumAnalyzer(2048, threads=2)
            try:
                got = analyzer.process(data)
            finally:
                analyzer.close()
        self.assertEqual(got.segments, want.segments)
        np.testing.assert_allclose(got.mean_db, want.mean_db, atol=0.01)
        np.testing.assert_allclose(got.max_db, want.max_db, atol=0.01)


if __name__ == "__main__":
    unittest.main()
