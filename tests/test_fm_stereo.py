#!/usr/bin/env python3
"""
Broadcast FM stereo (K-S1): the stereo decoder (dsp/fm_stereo.py), the
receiver chain's left and right channels, and the demo's stereo stations.

The stereo signals here come from an independent synthesizer of the
standard multiplex: ``(L+R)/2 + (L-R)/2 sin(2 phi) + pilot sin(phi)``, the
L-R carrier the pilot's second harmonic, both crossing zero upward
together (47 CFR 73.322).
"""

import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp import fm_stereo
from sdr_module.dsp.channelizer import lowpass_taps
from sdr_module.dsp.fm_stereo import StereoDecoder, stereo_noise_bandwidth

AUDIO = 48000.0


def _pre_emphasis(audio, rate, tau=75e-6):
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    return np.fft.irfft(spectrum * (1 + 2j * np.pi * freqs * tau), len(audio))


def _multiplex(left, right, rate, pilot=0.09, pilot_hz=19e3, phase=0.7):
    """The standard multiplex, scaled so the FM deviation is 1."""
    phi = 2 * np.pi * pilot_hz * np.arange(len(left)) / rate + phase
    return 0.9 * (
        (left + right) / 2 + (left - right) / 2 * np.sin(2 * phi)
    ) + pilot * np.sin(phi)


def _fm(mpx, rate, deviation=75e3):
    phase = 2 * np.pi * deviation * np.cumsum(mpx) / rate
    return np.exp(1j * phase).astype(np.complex64)


def _tone(x, freq, rate):
    """Amplitude of the ``freq`` component of ``x`` (least squares)."""
    t = np.arange(len(x)) / rate
    return float(
        np.hypot(
            2 * np.mean(x * np.cos(2 * np.pi * freq * t)),
            2 * np.mean(x * np.sin(2 * np.pi * freq * t)),
        )
    )


def _db(a, b):
    return 20 * np.log10(max(a, 1e-12) / max(b, 1e-12))


def _quick_rise():
    """The blend comes up from mono in 50 ms (not 0.75 s): shorter runs."""
    return mock.patch.object(StereoDecoder, "RISE_S", 0.05)


class TestStereoDecoder(unittest.TestCase):
    """The decoder on a multiplex given directly (no FM, no discriminator)."""

    RATE = 240e3

    def _decode(self, mpx, blocks=None):
        decoder = StereoDecoder(self.RATE, phase_step=False)
        if blocks is None:
            blocks = [len(mpx)]
        out, start = [], 0
        for size in blocks:
            out.append(decoder.process(mpx[start : start + size]))
            start += size
        return decoder, np.concatenate(out)

    def _audio(self, x):
        """The receiver's job: low-pass to 15 kHz (no de-emphasis here)."""
        taps = lowpass_taps(15e3, 18.5e3, self.RATE)
        return np.convolve(x, taps)[len(taps) // 2 : len(taps) // 2 + len(x)]

    def test_left_and_right_come_apart(self):
        with _quick_rise():
            n = int(0.5 * self.RATE)
            t = np.arange(n) / self.RATE
            left = 0.5 * np.sin(2 * np.pi * 1000 * t)
            right = 0.5 * np.sin(2 * np.pi * 3000 * t)
            decoder, out = self._decode(_multiplex(left, right, self.RATE))
        self.assertTrue(decoder.pilot)
        self.assertAlmostEqual(decoder.blend, 1.0, places=3)
        tail = slice(int(0.25 * self.RATE), None)
        difference = self._audio(out)[tail]  # = 0.9 (L - R) / 2
        mono = self._audio(_multiplex(left, right, self.RATE, pilot=0.0))[tail]
        heard_left, heard_right = mono + difference, mono - difference
        self.assertAlmostEqual(_tone(heard_left, 1000, self.RATE), 0.45, delta=0.005)
        self.assertAlmostEqual(_tone(heard_right, 3000, self.RATE), 0.45, delta=0.005)
        # Each tone stays on its own side: 50 dB and more.
        self.assertGreater(
            _db(
                _tone(heard_left, 1000, self.RATE), _tone(heard_right, 1000, self.RATE)
            ),
            50,
        )
        self.assertGreater(
            _db(
                _tone(heard_right, 3000, self.RATE), _tone(heard_left, 3000, self.RATE)
            ),
            50,
        )

    def test_a_pilot_off_frequency_is_followed(self):
        # A station's pilot a few Hz off (or the receiver's sample rate a
        # hundred ppm out): the phase is still right.
        with _quick_rise():
            n = int(0.7 * self.RATE)
            t = np.arange(n) / self.RATE
            left = 0.5 * np.sin(2 * np.pi * 1000 * t)
            for offset in (4.0, -7.0):
                mpx = _multiplex(left, 0 * left, self.RATE, pilot_hz=19e3 + offset)
                decoder, out = self._decode(mpx)
                self.assertAlmostEqual(decoder.offset_hz, offset, delta=0.2)
                difference = self._audio(out)[int(0.4 * self.RATE) :]
                # L - R = L: the difference carries the whole tone.
                self.assertAlmostEqual(
                    _tone(difference, 1000, self.RATE), 0.225, delta=0.002
                )

    def test_a_station_without_a_pilot_is_mono(self):
        n = int(0.5 * self.RATE)
        t = np.arange(n) / self.RATE
        left = 0.5 * np.sin(2 * np.pi * 1000 * t)
        decoder, out = self._decode(_multiplex(left, 0 * left, self.RATE, pilot=0.0))
        self.assertFalse(decoder.pilot)
        self.assertEqual(decoder.blend, 0.0)
        self.assertEqual(float(np.max(np.abs(out))), 0.0)

    def test_the_hiss_estimate_matches_the_hiss(self):
        # White noise on the multiplex (flat, not FM's rising noise, so the
        # check is simple): what stereo would add after the 15 kHz audio
        # filter is the noise density at the pilot times the stereo noise
        # bandwidth without de-emphasis.
        rng = np.random.default_rng(3)
        n = int(0.6 * self.RATE)
        zero = np.zeros(n)
        density = 1e-9  # per Hz (two-sided)
        noise = rng.standard_normal(n) * np.sqrt(density * self.RATE)
        decoder = StereoDecoder(self.RATE, phase_step=False)
        decoder.process(_multiplex(zero, zero, self.RATE) + noise)
        # (The estimate assumes FM's f^2 noise: compare its density.)
        hiss = 0.5 / 10 ** (decoder.snr_db / 10)
        estimated_density = hiss / stereo_noise_bandwidth()
        self.assertAlmostEqual(
            10 * np.log10(estimated_density / density), 0.0, delta=1.0
        )

    def test_the_blend_follows_the_noise(self):
        rng = np.random.default_rng(4)
        n = int(1.2 * self.RATE)
        t = np.arange(n) / self.RATE
        left = 0.5 * np.sin(2 * np.pi * 1000 * t)
        clean = _multiplex(left, 0 * left, self.RATE)
        blends = []
        for density in (1e-14, 1e-9, 1e-7):
            decoder = StereoDecoder(self.RATE, phase_step=False)
            noise = rng.standard_normal(n) * np.sqrt(density * self.RATE)
            decoder.process(clean + noise)
            blends.append(decoder.blend)
            # The hiss stereo adds once blended: never above the level of
            # full stereo at FULL_SNR_DB.
            hiss_db = -decoder.snr_db + 20 * np.log10(max(decoder.blend, 1e-9))
            self.assertLessEqual(hiss_db, -StereoDecoder.FULL_SNR_DB + 0.01)
        full, partly, mono = blends
        self.assertEqual(full, 1.0)
        self.assertGreater(partly, 0.05)
        self.assertLess(partly, 0.9)
        self.assertEqual(mono, 0.0)

    def test_the_blend_drops_at_once_and_comes_back_slowly(self):
        # A fade (noise for 100 ms): no burst of hiss, and no pumping after.
        rng = np.random.default_rng(5)
        rate = self.RATE
        t = np.arange(int(1.4 * rate)) / rate
        left = 0.5 * np.sin(2 * np.pi * 1000 * t)
        mpx = _multiplex(left, 0 * left, rate)
        fade = slice(int(0.9 * rate), int(1.0 * rate))
        mpx[fade] += rng.standard_normal(fade.stop - fade.start) * np.sqrt(1e-6 * rate)
        decoder = StereoDecoder(rate, phase_step=False)
        decoder.process(mpx[: int(0.9 * rate)])
        self.assertEqual(decoder.blend, 1.0)
        decoder.process(mpx[int(0.9 * rate) : int(0.92 * rate)])
        self.assertLess(decoder.blend, 0.1)  # within 20 ms
        decoder.process(mpx[int(0.92 * rate) : int(1.3 * rate)])
        # 0.3 s after the fade: back up 0.3 s / RISE_S of the way at most.
        self.assertGreater(decoder.blend, 0.2)
        self.assertLess(decoder.blend, 0.45)

    def test_blocks_of_any_size_give_the_same_output(self):
        with _quick_rise():
            rng = np.random.default_rng(6)
            n = int(0.4 * self.RATE)
            t = np.arange(n) / self.RATE
            left = 0.5 * np.sin(2 * np.pi * 1000 * t)
            mpx = _multiplex(left, 0 * left, self.RATE, pilot_hz=19003.0)
            mpx += rng.standard_normal(n) * 1e-4
            _decoder, whole = self._decode(mpx)
            sizes = list(rng.integers(1, 9000, size=200))
            _decoder, pieces = self._decode(mpx, sizes)
        self.assertEqual(len(pieces), len(whole))
        np.testing.assert_allclose(pieces, whole, atol=2e-5)

    def test_a_multiplex_rate_too_low_is_refused(self):
        with self.assertRaises(ValueError):
            StereoDecoder(fm_stereo.MIN_RATE / 2)

    def test_the_stereo_noise_bandwidth(self):
        # 75 us, 15 kHz: the closed form against a numerical integral.
        f = np.linspace(0.0, 15e3, 200001)
        fc = 1.0 / (2 * np.pi * 75e-6)
        density = ((38e3 - f) ** 2 + (38e3 + f) ** 2) / 19e3**2
        values = density / (1 + (f / fc) ** 2)
        numeric = 2 * float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(f)))
        self.assertAlmostEqual(stereo_noise_bandwidth() / numeric, 1.0, places=6)
        self.assertAlmostEqual(stereo_noise_bandwidth() / 1e3, 49.2, delta=0.1)


def _chain(rate, bandwidth=200e3, mode="FM"):
    from sdr_module.gui.main_window import _ReceiverChain

    return _ReceiverChain(rate, mode, bandwidth)


def _listen(chain, iq, deviation=75e3, block=None):
    """``(mono, left/right)`` of ``iq`` through ``chain`` a frame at a time
    (left/right are the mono audio twice where a block had no stereo)."""
    block = block or int(chain.rate / 30)
    mono, both = [], []
    for start in range(0, len(iq), block):
        audio = chain.demodulate(chain.channelize(iq[start : start + block]), deviation)
        mono.append(audio)
        pair = chain.stereo_audio
        both.append(np.stack((audio, audio), axis=1) if pair is None else pair)
    return np.concatenate(mono), np.concatenate(both)


class TestReceiverChainStereo(unittest.TestCase):
    """The main window's receiver chain plays broadcast FM in stereo."""

    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")
        patcher = _quick_rise()
        patcher.start()
        self.addCleanup(patcher.stop)

    def _station(self, rate, seconds=0.5, left_hz=1000.0, right_hz=3000.0):
        n = int(seconds * rate)
        t = np.arange(n) / rate
        left = _pre_emphasis(0.4 * np.sin(2 * np.pi * left_hz * t), rate)
        right = _pre_emphasis(0.4 * np.sin(2 * np.pi * right_hz * t), rate)
        return _fm(_multiplex(left, right, rate), rate)

    def test_left_and_right_at_the_receiver_rates(self):
        for rate in (1.024e6, 2.4e6, 3.2e6):
            with self.subTest(rate=rate):
                chain = _chain(rate)
                chain.stereo = True
                _mono, both = _listen(chain, self._station(rate))
                tail = both[int(0.25 * AUDIO) :]
                left, right = tail[:, 0], tail[:, 1]
                self.assertAlmostEqual(_tone(left, 1000, AUDIO), 0.36, delta=0.01)
                self.assertAlmostEqual(_tone(right, 3000, AUDIO), 0.36, delta=0.01)
                # 40 dB or more (the discriminator's droop made up: 35 dB
                # at 1 MS/s without it).
                self.assertGreater(
                    _db(_tone(left, 1000, AUDIO), _tone(right, 1000, AUDIO)), 40
                )
                self.assertGreater(
                    _db(_tone(right, 3000, AUDIO), _tone(left, 3000, AUDIO)), 40
                )
                self.assertGreater(chain.stereo_decoder.blend, 0.99)

    def test_the_mono_audio_is_unchanged(self):
        # (L+R)/2 everywhere: SSTV, the tone squelch, the clipping warning
        # and every mono listener hear exactly what they heard before.
        rate = 2.4e6
        iq = self._station(rate)
        plain = _chain(rate)
        mono_before, _both = _listen(plain, iq)
        chain = _chain(rate)
        chain.stereo = True
        mono, both = _listen(chain, iq)
        np.testing.assert_array_equal(mono, mono_before)
        np.testing.assert_allclose(both.mean(axis=1), mono, atol=1e-6)

    def test_stereo_back_on_mid_stream_lines_up(self):
        # Switched off and on again, the difference signal's filters start
        # afresh beside a mono stream that kept running: they take its
        # decimation phases and the resampler's time (frames of an odd
        # size leave them anywhere), or the two come out of step and left
        # and right only 30-45 dB apart (60-80 dB here).
        for rate in (1.024e6, 2.4e6):
            with self.subTest(rate=rate):
                chain = _chain(rate)
                iq = self._station(rate, seconds=0.8)
                block = int(rate / 30) + 777
                pairs, stopped = [], []
                for start in range(0, len(iq), block):
                    chain.stereo = not 0.15 <= start / rate < 0.3
                    audio = chain.demodulate(
                        chain.channelize(iq[start : start + block]), 75e3
                    )
                    if not chain.stereo:
                        stopped.append(chain.stereo_decoder is None)
                    pair = chain.stereo_audio
                    pairs.append(np.stack((audio, audio), 1) if pair is None else pair)
                self.assertTrue(any(stopped))  # (so it started over)
                tail = np.concatenate(pairs)[int(0.5 * AUDIO) :]
                left, right = tail[:, 0], tail[:, 1]
                self.assertGreater(
                    _db(_tone(left, 1000, AUDIO), _tone(right, 1000, AUDIO)), 55
                )
                self.assertGreater(
                    _db(_tone(right, 3000, AUDIO), _tone(left, 3000, AUDIO)), 55
                )

    def test_a_narrow_filter_plays_mono(self):
        # The 100 kHz filter cuts so much of a stereo station's FM spectrum
        # that left and right came out 6-8 dB apart (its distortion read as
        # hiss: 40 % stereo, "too weak"): mono, and the filter is the
        # reason given (see test_gui_stereo).
        rate = 2.4e6
        chain = _chain(rate, bandwidth=100e3)
        chain.stereo = True
        mono, both = _listen(chain, self._station(rate, seconds=0.4))
        self.assertIsNone(chain.stereo_decoder)
        np.testing.assert_array_equal(both[:, 0], mono)
        np.testing.assert_array_equal(both[:, 1], mono)

    def test_stereo_is_off_unless_asked_for(self):
        # (The second receiver's chains, tests and one-off demodulation.)
        rate = 2.4e6
        chain = _chain(rate)
        self.assertFalse(chain.stereo)
        _mono, both = _listen(chain, self._station(rate, 0.2))
        np.testing.assert_array_equal(both[:, 0], both[:, 1])
        self.assertIsNone(chain.stereo_decoder)

    def test_only_broadcast_fm_is_stereo(self):
        rate = 2.4e6
        iq = self._station(rate, 0.3)
        for mode, deviation, bandwidth in (
            ("FM", 5e3, 25e3),
            ("FM", 75e3, 50e3),  # (the filter too narrow for the multiplex)
            ("AM", 0.0, 10e3),
        ):
            with self.subTest(mode=mode, deviation=deviation, bandwidth=bandwidth):
                chain = _chain(rate, bandwidth, mode)
                chain.stereo = True
                _mono, both = _listen(chain, iq, deviation)
                np.testing.assert_array_equal(both[:, 0], both[:, 1])
                self.assertIsNone(chain.stereo_audio)

    def _switched(self, nr=None):
        """Left/right of a 200 Hz tone on the left only, stereo switched
        off a third of a second in (and back on later), and the largest
        step from one sample to the next on either side."""
        from sdr_module.gui.audio_chain import AudioOptions, make_noise_reduction

        rate = 2.4e6
        n = int(0.8 * rate)
        t = np.arange(n) / rate
        left = 0.4 * np.sin(2 * np.pi * 200 * t)
        iq = _fm(_multiplex(left, 0 * left, rate), rate)
        chain = _chain(rate)
        if nr:
            chain.options = AudioOptions()
            chain.options.noise_reduction = make_noise_reduction(nr, AUDIO)
        chain.stereo = True
        block = int(rate / 30)
        pairs, kinds = [], []
        for index, start in enumerate(range(0, n, block)):
            chain.stereo = not 10 <= index < 16
            audio = chain.demodulate(chain.channelize(iq[start : start + block]), 75e3)
            pair = chain.stereo_audio
            kinds.append(pair is not None)
            pairs.append(np.stack((audio, audio), 1) if pair is None else pair)
        both = np.concatenate(pairs)
        return chain, kinds, both

    def test_switching_stereo_off_fades_to_mono_and_back(self):
        for nr in (None, "light"):
            with self.subTest(noise_reduction=nr):
                chain, kinds, both = self._switched(nr)
                # Stereo until switched off, mono a block or two later (the
                # difference faded out over 20 ms and its filters, and the
                # noise reduction's 21 ms, emptied), stereo again after.
                self.assertTrue(all(kinds[3:10]))
                self.assertFalse(any(kinds[12:16]))
                self.assertTrue(all(kinds[18:]))
                # No click: no step from one sample to the next bigger than
                # the tone's own (a 200 Hz tone of 0.36: 0.0094).
                steps = np.max(np.abs(np.diff(both[int(0.05 * AUDIO) :], axis=0)))
                self.assertLess(steps, 0.0094 * 1.15)
                # (L+R)/2 is the mono audio throughout.
                self.assertGreater(chain.stereo_decoder.blend, 0.99)

    def test_noise_reduction_keeps_the_sides_apart(self):
        from sdr_module.gui.audio_chain import AudioOptions, make_noise_reduction

        rate = 2.4e6
        options = AudioOptions()
        options.noise_reduction = make_noise_reduction("light", AUDIO)
        chain = _chain(rate)
        chain.options = options
        chain.stereo = True
        mono, both = _listen(chain, self._station(rate, 0.8))
        tail = both[int(0.5 * AUDIO) :]
        left, right = tail[:, 0], tail[:, 1]
        self.assertGreater(_db(_tone(left, 1000, AUDIO), _tone(right, 1000, AUDIO)), 25)
        self.assertGreater(_db(_tone(right, 3000, AUDIO), _tone(left, 3000, AUDIO)), 25)
        np.testing.assert_allclose(both.mean(axis=1), mono, atol=1e-6)


class TestDemoStereo(unittest.TestCase):
    """The demo's FM stations send real stereo (different left and right)."""

    RATE = 2.4e6

    def setUp(self):
        from sdr_module.gui import device_dialog

        self.dd = device_dialog
        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _read(self, freq, seconds):
        device = self.dd.MockDevice()
        device.set_sample_rate(self.RATE)
        device.set_frequency(freq)
        device._running = True
        device._rng = np.random.default_rng(7)
        block = int(self.RATE / 30)
        return np.concatenate(
            [device.read_samples(block) for _ in range(int(seconds * 30))]
        )

    def test_the_multiplex_follows_the_standard(self):
        # Independently of the receiver: the 97.1 MHz station's 3.7 kHz tone
        # is on the left only, so L-R carries it, on the pilot's second
        # harmonic in phase (sin 2 phi for a pilot sin phi). It was sent in
        # quadrature, which a stereo decoder can't hear.
        iq = self._read(97.1e6, 0.25)
        mpx = np.angle(iq[1:] * np.conj(iq[:-1])) * self.RATE / (2 * np.pi * 75e3)
        t = np.arange(len(mpx)) / self.RATE
        # The pilot's phase: fit p sin(w t + phi).
        w = 2 * np.pi * 19e3
        a = 2 * np.mean(mpx * np.sin(w * t))
        b = 2 * np.mean(mpx * np.cos(w * t))
        phi = np.arctan2(b, a)
        self.assertAlmostEqual(np.hypot(a, b), 0.09, delta=0.005)
        taps = lowpass_taps(6e3, 9e3, self.RATE)

        def detect(carrier):
            x = np.convolve(2 * mpx * carrier, taps, mode="same")
            return _tone(x[len(taps) : -len(taps)], 3.7e3, self.RATE)

        in_phase = detect(np.sin(2 * (w * t + phi)))
        quadrature = detect(np.cos(2 * (w * t + phi)))
        self.assertAlmostEqual(in_phase, 0.18, delta=0.01)
        self.assertLess(quadrature, 0.03)

    def test_a_tone_station_is_heard_on_the_left(self):
        with _quick_rise():
            chain = _chain(self.RATE)
            chain.stereo = True
            _mono, both = _listen(chain, self._read(97.1e6, 0.6))
        tail = both[int(0.3 * AUDIO) :]
        left = _tone(tail[:, 0], 3.7e3, AUDIO)
        right = _tone(tail[:, 1], 3.7e3, AUDIO)
        self.assertGreater(left, 0.15)
        self.assertGreater(_db(left, right), 35)

    def test_the_music_station_plays_its_left_and_right(self):
        dd = self.dd
        with _quick_rise():
            chain = _chain(self.RATE)
            chain.stereo = True
            mono, both = _listen(chain, self._read(100.1e6, 1.0))
        factor = max(1, int(round(self.RATE / dd._LOOP_RATE_HZ)))
        loop_rate = self.RATE / factor
        size = dd._fft_friendly_size(
            int(round(dd._MUSIC_STEP_S * 8 * len(dd._MUSIC_BARS) * loop_rate))
        )
        left, right = dd._music_channels(loop_rate, size)
        start, count = int(0.4 * AUDIO), int(0.5 * AUDIO)
        heard = mono[start : start + count]
        # Line the tune up with what was heard (the filters' delay).
        reference = (left + right) / 2
        lags = range(0, 200)
        scores = [
            np.dot(heard, reference[(np.arange(start, start + count) - lag) % size])
            for lag in lags
        ]
        lag = lags[int(np.argmax(scores))]
        index = (np.arange(start, start + count) - lag) % size
        basis = np.stack((left[index], right[index]), axis=1)
        for side, (own, other) in ((0, (0, 1)), (1, (1, 0))):
            coef, *_ = np.linalg.lstsq(basis, both[start : start + count, side])
            self.assertGreater(coef[own], 0.8)
            self.assertLess(abs(coef[other]), 0.03)


if __name__ == "__main__":
    unittest.main()
