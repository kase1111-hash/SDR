#!/usr/bin/env python3
"""
Broadcast FM de-emphasis, 75 us or 50 us (M-A1): the Americas and South
Korea pre-emphasize with 75 us, Europe, Australia and most of the world
with 50 us, and a receiver set to the other one plays the treble too
bright or too dull. The receiver chain reads the time constant from its
``AudioOptions`` (``deemphasis_s``) on every block, for the mono program,
the stereo difference signal and the blend's hiss estimate alike, and a
change crossfades over 10 ms: no click.

The stations here are synthesized independently of the receiver: tones
pre-emphasized with ``1 + j 2 pi f tau`` (the exact inverse of the one-pole
de-emphasis), FM-modulated at 75 kHz deviation, sampled at 2.4 MS/s.
"""

import math
import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp.fm_stereo import StereoDecoder, stereo_noise_bandwidth

RATE = 2.4e6
AUDIO = 48000.0
#: Tones of the response test (Hz), not harmonically related.
FREQS = (100.0, 400.0, 1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10e3, 12e3, 14e3)


def _pre_emphasis(x, rate, tau):
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    return np.fft.irfft(spectrum * (1 + 2j * np.pi * freqs * tau), len(x))


def _tones(n, rate, amplitude, freqs=FREQS):
    t = np.arange(n) / rate
    return sum(
        amplitude * np.sin(2 * np.pi * f * t + 0.7 * k) for k, f in enumerate(freqs)
    )


def _multiplex(left, right, rate, pilot=0.09, phase=0.4):
    """The standard multiplex (47 CFR 73.322), deviation 1 = full scale."""
    phi = 2 * np.pi * 19e3 * np.arange(len(left)) / rate + phase
    return 0.9 * (
        (left + right) / 2 + (left - right) / 2 * np.sin(2 * phi)
    ) + pilot * np.sin(phi)


def _fm(mpx, rate=RATE, deviation=75e3):
    phase = 2 * np.pi * deviation * np.cumsum(mpx) / rate
    return np.exp(1j * phase).astype(np.complex64)


def _level(x, freq, rate=AUDIO):
    t = np.arange(len(x)) / rate
    c = 2 * np.mean(x * np.cos(2 * np.pi * freq * t))
    s = 2 * np.mean(x * np.sin(2 * np.pi * freq * t))
    return float(np.hypot(c, s))


def _db(a, b):
    return 20 * math.log10(max(a, 1e-15) / max(b, 1e-15))


def _chain(tau=None, mode="FM", bandwidth=200e3, stereo=False):
    from sdr_module.gui.audio_chain import AudioOptions
    from sdr_module.gui.main_window import _ReceiverChain

    chain = _ReceiverChain(RATE, mode, bandwidth)
    chain.options = AudioOptions()
    if tau is not None:
        chain.options.deemphasis_s = tau
    chain.stereo = stereo
    return chain


def _listen(chain, iq, deviation=75e3, block=80_000, before_block=None):
    """``(mono, (n, 2) left/right)`` of ``iq`` a 33 ms frame at a time;
    ``before_block(index)`` runs before each block (to change settings)."""
    mono, both = [], []
    for index, start in enumerate(range(0, len(iq), block)):
        if before_block is not None:
            before_block(index)
        audio = chain.demodulate(chain.channelize(iq[start : start + block]), deviation)
        mono.append(audio)
        pair = chain.stereo_audio
        both.append(np.stack((audio, audio), axis=1) if pair is None else pair)
    return np.concatenate(mono), np.concatenate(both)


def _quick_rise():
    return mock.patch.object(StereoDecoder, "RISE_S", 0.05)


class _ChainCase(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401  (main_window imports it)
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")


class TestTheChoices(unittest.TestCase):
    def test_75_and_50_us_with_75_the_default(self):
        from sdr_module.gui.audio_chain import AudioOptions
        from sdr_module.gui.deemphasis import (
            DEEMPHASIS_CHOICES,
            DEFAULT_DEEMPHASIS,
            deemphasis_key,
            deemphasis_seconds,
        )

        self.assertEqual(
            [key for key, _label, _tip in DEEMPHASIS_CHOICES], ["75", "50"]
        )
        self.assertEqual(DEFAULT_DEEMPHASIS, "75")
        for _key, label, tip in DEEMPHASIS_CHOICES:
            self.assertIn("µs", label)
            self.assertTrue(tip)
        self.assertIn("Americas", DEEMPHASIS_CHOICES[0][1])
        self.assertIn("Europe", DEEMPHASIS_CHOICES[1][1])
        self.assertEqual(deemphasis_seconds("75"), 75e-6)
        self.assertEqual(deemphasis_seconds(" 50 "), 50e-6)
        self.assertEqual(deemphasis_seconds(50), 50e-6)  # (as an INI file reads)
        for damaged in ("", "60", "loud", None, "nan", 25):
            self.assertIsNone(deemphasis_seconds(damaged), damaged)
        self.assertEqual(deemphasis_key(50e-6), "50")
        self.assertEqual(deemphasis_key(75e-6), "75")
        self.assertIsNone(deemphasis_key(25e-6))
        # Today's behaviour unless chosen otherwise.
        self.assertEqual(AudioOptions().deemphasis_s, 75e-6)


class TestTheResponse(_ChainCase):
    """What a station sent comes out as it was spoken, with the matching
    setting: flat from 100 Hz to 14 kHz."""

    def _response(self, sent_tau, heard_tau, amplitude=0.02):
        n = int(0.6 * RATE)
        program = _tones(n, RATE, amplitude)
        if sent_tau is not None:
            program = _pre_emphasis(program, RATE, sent_tau)
        chain = _chain(heard_tau)
        mono, _both = _listen(chain, _fm(program))
        tail = mono[int(0.2 * AUDIO) :]
        return chain, {f: _db(_level(tail, f), amplitude) for f in FREQS}

    def test_a_pre_emphasized_station_comes_out_flat_with_its_setting(self):
        for tau in (75e-6, 50e-6):
            with self.subTest(tau=tau):
                chain, response = self._response(tau, tau)
                self.assertEqual(chain.deemphasis_s, tau)
                for freq, gain in response.items():
                    self.assertAlmostEqual(gain, 0.0, delta=0.25, msg=f"{freq} Hz")

    def test_the_other_setting_tilts_the_treble(self):
        # A European (50 us) station heard with 75 us: dull; an American one
        # heard with 50 us: bright (the limitation this removes).
        _chain_, dull = self._response(50e-6, 75e-6)
        _chain_, bright = self._response(75e-6, 50e-6)
        expected = 20 * math.log10(
            abs(1 + 2j * math.pi * 10e3 * 75e-6) / abs(1 + 2j * math.pi * 10e3 * 50e-6)
        )
        self.assertAlmostEqual(dull[10e3], -expected, delta=0.3)
        self.assertAlmostEqual(bright[10e3], expected, delta=0.3)
        self.assertGreater(expected, 3.0)

    def test_3_db_down_at_the_corner_without_pre_emphasis(self):
        # The one-pole de-emphasis: -3 dB at 1 / (2 pi tau), 2122 Hz for
        # 75 us and 3183 Hz for 50 us.
        n = int(0.6 * RATE)
        for tau in (75e-6, 50e-6):
            with self.subTest(tau=tau):
                corner = 1.0 / (2 * math.pi * tau)
                freqs = (100.0, corner, 5000.0)
                chain = _chain(tau)
                mono, _both = _listen(chain, _fm(_tones(n, RATE, 0.1, freqs)))
                tail = mono[int(0.2 * AUDIO) :]
                low = _level(tail, 100.0)
                self.assertAlmostEqual(
                    _db(_level(tail, corner), low), -3.01, delta=0.15
                )
                expected = -10 * math.log10(1 + (2 * math.pi * 5000.0 * tau) ** 2)
                self.assertAlmostEqual(
                    _db(_level(tail, 5000.0), low), expected, delta=0.15
                )

    def test_a_chain_without_options_and_the_default_are_75_us(self):
        from sdr_module.gui.main_window import _ReceiverChain

        n = int(0.3 * RATE)
        iq = _fm(_pre_emphasis(_tones(n, RATE, 0.02), RATE, 75e-6))
        bare = _ReceiverChain(RATE, "FM", 200e3)
        plain, _ = _listen(bare, iq)
        default, _ = _listen(_chain(), iq)
        explicit, _ = _listen(_chain(75e-6), iq)
        np.testing.assert_array_equal(plain, default)
        np.testing.assert_array_equal(plain, explicit)
        self.assertEqual(bare.deemphasis_s, 75e-6)

    def test_narrow_fm_and_other_modes_are_unaffected(self):
        n = int(0.3 * RATE)
        t = np.arange(n) / RATE
        voice = 0.5 * np.sin(2 * np.pi * 1200 * t) + 0.3 * np.sin(2 * np.pi * 2900 * t)
        for mode, deviation, bandwidth, iq in (
            ("FM", 5e3, 12.5e3, _fm(voice, deviation=5e3)),
            ("FM", 12.5e3, 25e3, _fm(voice, deviation=12.5e3)),
            ("AM", 0.0, 10e3, (0.1 * (1 + 0.5 * voice)).astype(np.complex64)),
            (
                "USB",
                0.0,
                2.7e3,
                (0.1 * np.exp(2j * np.pi * 1e3 * t)).astype(np.complex64),
            ),
        ):
            with self.subTest(mode=mode, deviation=deviation):
                runs = []
                for tau in (75e-6, 50e-6):
                    chain = _chain(tau, mode, bandwidth)
                    runs.append(_listen(chain, iq, deviation)[0])
                np.testing.assert_array_equal(runs[0], runs[1])


class TestStereo(_ChainCase):
    """Left, right and the stereo difference are de-emphasized alike."""

    def setUp(self):
        super().setUp()
        patcher = _quick_rise()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_left_and_right_come_out_flat_and_apart(self):
        n = int(0.8 * RATE)
        amplitude = 0.012
        for tau in (50e-6, 75e-6):
            with self.subTest(tau=tau):
                left = _pre_emphasis(_tones(n, RATE, amplitude), RATE, tau)
                iq = _fm(_multiplex(left, np.zeros(n), RATE))
                chain = _chain(tau, stereo=True)
                _mono, both = _listen(chain, iq)
                self.assertGreater(chain.stereo_decoder.blend, 0.99)
                tail = both[int(0.4 * AUDIO) :]
                for freq in FREQS:
                    heard_left = _level(tail[:, 0], freq)
                    heard_right = _level(tail[:, 1], freq)
                    # (0.9: the program's share of the multiplex.)
                    self.assertAlmostEqual(
                        _db(heard_left, 0.9 * amplitude), 0.0, delta=0.3, msg=freq
                    )
                    self.assertGreater(_db(heard_left, heard_right), 40.0, freq)

    def test_the_blend_hears_the_hiss_through_the_same_time_constant(self):
        # 50 us lets more of the difference signal's hiss through than 75 us
        # (less treble cut): the same noise reads that much closer to full
        # scale, so a weak station blends toward mono a little sooner.
        ratio_db = 10 * math.log10(
            stereo_noise_bandwidth(50e-6) / stereo_noise_bandwidth(75e-6)
        )
        self.assertGreater(ratio_db, 0.5)
        n = int(0.8 * RATE)
        t = np.arange(n) / RATE
        left = 0.3 * np.sin(2 * np.pi * 1000 * t)
        rng = np.random.default_rng(3)
        noise = 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        iq = (_fm(_multiplex(left, 0 * left, RATE)) + noise).astype(np.complex64)
        readings = {}
        for tau in (75e-6, 50e-6):
            chain = _chain(tau, stereo=True)
            _listen(chain, iq)
            decoder = chain.stereo_decoder
            self.assertEqual(decoder.tau_s, tau)
            readings[tau] = decoder.snr_db
        self.assertAlmostEqual(readings[75e-6] - readings[50e-6], ratio_db, delta=0.05)


class TestChangingItWhilePlaying(_ChainCase):
    """Radio > Audio > De-emphasis applies at once, without a click: the
    new filters start from the stream's history and take over in 10 ms."""

    TONE_HZ = 10e3  # where 75 and 50 us differ most in the audio band

    def setUp(self):
        super().setUp()
        patcher = _quick_rise()
        patcher.start()
        self.addCleanup(patcher.stop)

    def _switched(self, stereo, switch_at=10, back_at=None, block=80_000 + 777):
        n = int(1.0 * RATE)
        t = np.arange(n) / RATE
        tone = 0.25 * np.sin(2 * np.pi * self.TONE_HZ * t)
        if stereo:
            iq = _fm(_multiplex(tone, 0 * tone, RATE))
        else:
            iq = _fm(tone)
        chain = _chain(75e-6, stereo=stereo)

        def before(index):
            if index == switch_at:
                chain.options.deemphasis_s = 50e-6
            if back_at is not None and index == back_at:
                chain.options.deemphasis_s = 75e-6

        mono, both = _listen(chain, iq, block=block, before_block=before)
        switch = int(switch_at * block / RATE * AUDIO)
        return chain, mono, both, switch

    def _expected(self, tau, amplitude=0.25):
        return amplitude / abs(1 + 2j * math.pi * self.TONE_HZ * tau)

    def test_mono_takes_the_new_setting_without_a_click(self):
        chain, mono, _both, switch = self._switched(stereo=False)
        self.assertEqual(chain.deemphasis_s, 50e-6)
        before = mono[switch - 9600 : switch - 480]
        after = mono[switch + 960 : switch + 9600]
        self.assertAlmostEqual(
            _level(before, self.TONE_HZ), self._expected(75e-6), delta=5e-4
        )
        self.assertAlmostEqual(
            _level(after, self.TONE_HZ), self._expected(50e-6), delta=5e-4
        )
        # No click: no step from one sample to the next bigger than the
        # louder tone's own (2 A sin(pi f / 48 kHz)).
        natural = 2 * self._expected(50e-6) * math.sin(math.pi * self.TONE_HZ / AUDIO)
        steps = np.abs(np.diff(mono[int(0.05 * AUDIO) :]))
        self.assertLess(float(np.max(steps)), natural * 1.03)

    def test_stereo_keeps_left_and_right_apart_through_the_change(self):
        chain, _mono, both, switch = self._switched(stereo=True, switch_at=12)
        decoder = chain.stereo_decoder
        self.assertIsNotNone(decoder)
        self.assertEqual(decoder.tau_s, 50e-6)
        self.assertGreater(decoder.blend, 0.99)
        after = both[switch + 960 :]
        left = _level(after[:, 0], self.TONE_HZ)
        right = _level(after[:, 1], self.TONE_HZ)
        self.assertAlmostEqual(left, 0.9 * self._expected(50e-6), delta=1e-3)
        self.assertGreater(_db(left, right), 40.0)
        natural = (
            2 * 0.9 * self._expected(50e-6) * math.sin(math.pi * self.TONE_HZ / AUDIO)
        )
        for side in (0, 1):
            steps = np.abs(np.diff(both[int(0.3 * AUDIO) :, side]))
            self.assertLess(float(np.max(steps)), natural * 1.03, side)

    def test_changed_twice_in_quick_succession(self):
        # Back to 75 us one block later (while the first change may still be
        # crossfading): it ends at 75 us, without a click.
        chain, mono, _both, switch = self._switched(
            stereo=False, switch_at=10, back_at=11, block=4000
        )
        self.assertEqual(chain.deemphasis_s, 75e-6)
        tail = mono[-9600:]
        self.assertAlmostEqual(
            _level(tail, self.TONE_HZ), self._expected(75e-6), delta=5e-4
        )
        natural = 2 * self._expected(50e-6) * math.sin(math.pi * self.TONE_HZ / AUDIO)
        steps = np.abs(np.diff(mono[int(0.05 * AUDIO) :]))
        self.assertLess(float(np.max(steps)), natural * 1.03)

    def test_a_new_stream_starts_with_the_setting(self):
        # Set while nothing played (the squelch closed): the next stream
        # starts with it, nothing to crossfade.
        n = int(0.4 * RATE)
        tone = 0.25 * np.sin(2 * np.pi * self.TONE_HZ * np.arange(n) / RATE)
        chain = _chain(75e-6)
        _listen(chain, _fm(tone[: n // 2]))
        chain.reset_demodulator()
        chain.options.deemphasis_s = 50e-6
        mono, _both = _listen(chain, _fm(tone[n // 2 :]))
        self.assertEqual(chain.deemphasis_s, 50e-6)
        self.assertAlmostEqual(
            _level(mono[2400:], self.TONE_HZ), self._expected(50e-6), delta=5e-4
        )

    def test_any_value_but_a_time_constant_is_75_us(self):
        n = int(0.2 * RATE)
        tone = 0.25 * np.sin(2 * np.pi * self.TONE_HZ * np.arange(n) / RATE)
        reference, _ = _listen(_chain(75e-6), _fm(tone))
        for damaged in (float("nan"), -50e-6, 0.0, "fifty", None, 1.0):
            with self.subTest(damaged=damaged):
                chain = _chain()
                chain.options.deemphasis_s = damaged
                mono, _ = _listen(chain, _fm(tone))
                self.assertEqual(chain.deemphasis_s, 75e-6)
                np.testing.assert_array_equal(mono, reference)


class TestTheDecodersTimeConstant(unittest.TestCase):
    """StereoDecoder.tau_s: the time constant the blend's hiss estimate
    assumes (settable while it runs; the estimate moves to it in 20 ms)."""

    def test_it_can_be_changed_while_it_runs(self):
        rate = 240e3
        n = int(0.4 * rate)
        rng = np.random.default_rng(5)
        t = np.arange(n) / rate
        left = 0.3 * np.sin(2 * np.pi * 1000 * t)
        mpx = _multiplex(left, 0 * left, rate) + 1e-3 * rng.standard_normal(n)
        switched = StereoDecoder(rate, phase_step=False)
        self.assertEqual(switched.tau_s, 75e-6)
        switched.process(mpx[: n // 2])
        switched.tau_s = 50e-6
        self.assertEqual(switched.tau_s, 50e-6)
        switched.process(mpx[n // 2 :])
        # The same samples heard with 50 us from the start, and with 75 us
        # throughout: the noise measured is the same, the hiss it means
        # differs by the noise bandwidths.
        fifty = StereoDecoder(rate, tau_s=50e-6, phase_step=False)
        fifty.process(mpx)
        seventy_five = StereoDecoder(rate, phase_step=False)
        seventy_five.process(mpx)
        ratio_db = 10 * math.log10(
            stereo_noise_bandwidth(50e-6) / stereo_noise_bandwidth(75e-6)
        )
        self.assertAlmostEqual(switched.snr_db, fifty.snr_db, delta=1e-4)
        self.assertAlmostEqual(seventy_five.snr_db - fifty.snr_db, ratio_db, delta=1e-4)

    def test_the_estimate_glides_to_a_new_time_constant(self):
        # Changed while it runs, the hiss estimate moves over 20 ms (the
        # blend never jumps): halfway there 10 ms after the change.
        rate = 240e3
        n = int(0.3 * rate)
        rng = np.random.default_rng(6)
        mpx = _multiplex(np.zeros(n), np.zeros(n), rate) + 1e-3 * rng.standard_normal(n)
        reference = StereoDecoder(rate, phase_step=False)
        reference.process(mpx[: n // 2])
        decoder = StereoDecoder(rate, phase_step=False)
        decoder.process(mpx[: n // 2])
        decoder.tau_s = 50e-6
        step = int(0.010 * rate)
        decoder.process(mpx[n // 2 : n // 2 + step])
        reference.process(mpx[n // 2 : n // 2 + step])
        ratio_db = 10 * math.log10(
            stereo_noise_bandwidth(50e-6) / stereo_noise_bandwidth(75e-6)
        )
        moved = reference.snr_db - decoder.snr_db
        self.assertGreater(moved, 0.3 * ratio_db)
        self.assertLess(moved, 0.7 * ratio_db)
        decoder.process(mpx[n // 2 + step :])
        reference.process(mpx[n // 2 + step :])
        self.assertAlmostEqual(reference.snr_db - decoder.snr_db, ratio_db, delta=1e-4)

    def test_a_time_constant_must_be_positive(self):
        decoder = StereoDecoder(240e3)
        for bad in (0.0, -1e-6, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                decoder.tau_s = bad
            with self.assertRaises(ValueError):
                StereoDecoder(240e3, tau_s=bad)


if __name__ == "__main__":
    unittest.main()
