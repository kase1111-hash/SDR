#!/usr/bin/env python3
"""
The DCS turn-off code (M-Q1): radios end a DCS transmission with a 134.4 Hz
tone, sent for 150-200 ms after the code while the carrier is still up (CML's
CMX882 datasheet: 134.4 +/- 0.5 Hz for 150-200 ms; Norcomm's NC105 manual: a
134.4 Hz sine for 180 ms, which disables the decoder within 35 ms). A
receiver that hears it closes its squelch at once, before the carrier drops,
so the squelch tail (the noise burst as the carrier goes) is never heard.

The code itself has no energy at 134.4 Hz (its bits are 1/134.4 s long, so
the spectrum of any bit pattern has a null there), which tells the tone from
the code; a low note, a voice's pitch or noise that comes with the code
leaves the code's own power in what is heard, which tells the turn-off code
(the code gone, only the tone left) from them.
"""

import unittest

import numpy as np

from tests.voice_synth import CLOSE_VOWEL, OPEN_VOWEL, pitched_voice, tone_level

RATE = 48000.0
LEVEL = 0.15  # DCS at 15 % of the deviation, as radios send it
BLOCK = 480  # 10 ms


def _as_radios_send_it(voice, corner=300.0, order=4):
    """Voice through a radio's gentle voice high-pass (Butterworth, in the
    frequency domain), peak 0.6."""
    spectrum = np.fft.rfft(voice)
    f = np.fft.rfftfreq(len(voice), 1.0 / RATE)
    spectrum /= np.sqrt(1.0 + (corner / np.maximum(f, 1e-9)) ** (2 * order))
    sent = np.fft.irfft(spectrum, len(voice))
    return 0.6 * sent / np.max(np.abs(sent))


def _carrier_gone(seconds, seed):
    """What the discriminator gives once the carrier has dropped: loud
    noise (a squelch tail)."""
    return 0.6 * np.random.default_rng(seed).standard_normal(int(seconds * RATE))


def _detector():
    from sdr_module.dsp.dcs import DcsDetector

    return DcsDetector(RATE)


def _ending(code, inverted=False, seconds=1.0, turn_off=True, **kw):
    """Half a second of quiet, a transmission of ``code`` (with its turn-off
    code), then the carrier gone; and where the turn-off code starts (s)."""
    from sdr_module.dsp.dcs import dcs_transmission

    quiet = np.zeros(int(0.5 * RATE))
    sent = LEVEL * dcs_transmission(
        code,
        inverted,
        rate=RATE,
        seconds=seconds,
        turn_off_s=None if turn_off else 0.0,
        **kw,
    )
    x = np.concatenate((quiet, sent, _carrier_gone(0.6, code)))
    return x, 0.5 + seconds


def _close_time(detector, x, code, inverted, block=BLOCK):
    """Seconds into ``x`` at which the squelch first closed after it had
    opened on ``code`` (None if it never opened, or never closed)."""
    opened = False
    for pos in range(0, len(x), block):
        detector.process(x[pos : pos + block])
        present = detector.is_present(code, inverted)
        if present:
            opened = True
        elif opened:
            return (pos + block) / RATE
    return None


class TestTurnOffEncoder(unittest.TestCase):
    def test_the_turn_off_code_is_a_134_4_hz_tone_for_180_ms(self):
        from sdr_module.dsp.dcs import DCS_TURN_OFF_HZ, DCS_TURN_OFF_S, dcs_turn_off

        self.assertEqual(DCS_TURN_OFF_HZ, 134.4)
        self.assertEqual(DCS_TURN_OFF_S, 0.18)
        x = dcs_turn_off(rate=RATE)
        self.assertEqual(len(x), int(round(0.18 * RATE)))
        # At the code's own level (+1/-1), all of it at 134.4 Hz.
        self.assertAlmostEqual(tone_level(x, 134.4, RATE), 1.0, delta=0.01)
        rest = x - np.sin(2 * np.pi * 134.4 * np.arange(len(x)) / RATE)
        self.assertLess(float(np.max(np.abs(rest))), 1e-9)
        # Pieces of it join up.
        joined = np.concatenate(
            (dcs_turn_off(rate=RATE, samples=1000), dcs_turn_off(rate=RATE, start=1000))
        )
        self.assertTrue(np.allclose(joined[: len(x)], x))

    def test_a_transmission_is_the_code_then_its_turn_off_code(self):
        from sdr_module.dsp.dcs import dcs_transmission

        x = dcs_transmission(23, rate=RATE, seconds=1.0)
        self.assertEqual(len(x), int(round(1.18 * RATE)))
        # The code first: read as 023 N.
        detector = _detector()
        detector.process(x[: int(1.0 * RATE)])
        self.assertEqual(detector.detected, (23, False))
        # Then the tone: almost all of the last 160 ms is 134.4 Hz.
        tail = x[int(1.02 * RATE) :]
        self.assertGreater(tone_level(tail, 134.4, RATE), 0.95)
        # Band-limited as the code is, the join included: next to nothing
        # above 400 Hz.
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / RATE)
        self.assertLess(spectrum[f > 400].sum(), 2e-3 * spectrum.sum())
        # Without its turn-off code, just the code.
        plain = dcs_transmission(23, rate=RATE, seconds=1.0, turn_off_s=0.0)
        self.assertEqual(len(plain), int(RATE))


class TestTheTurnOffCodeClosesTheSquelch(unittest.TestCase):
    def test_within_100_ms_of_its_start(self):
        """(Before: 0.35-0.40 s after the code stopped, turn-off code or
        not: the squelch tail was heard.)"""
        rng = np.random.default_rng(3)
        for code in (23, 115, 212, 365, 506, 754):
            for inverted in (False, True):
                x, turn_off_at = _ending(code, inverted)
                x = x + 0.01 * rng.standard_normal(len(x))
                detector = _detector()
                closed = _close_time(detector, x, code, inverted)
                self.assertIsNotNone(closed, (code, inverted))
                late = closed - turn_off_at
                self.assertLessEqual(late, 0.10, (code, inverted, late))
                self.assertGreater(late, 0.0, (code, inverted, late))
                self.assertTrue(detector.turned_off)
                self.assertEqual(detector.turn_offs, 1)
                self.assertIsNone(detector.detected)

    def test_every_standard_code(self):
        from sdr_module.dsp.dcs import DCS_CODES

        for code in DCS_CODES:
            for inverted in (False, True):
                x, turn_off_at = _ending(code, inverted, seconds=0.8)
                detector = _detector()
                closed = _close_time(detector, x, code, inverted, block=1600)
                self.assertIsNotNone(closed, (code, inverted))
                # (Within 100 ms, to the end of the 33 ms block it fell in.)
                self.assertLessEqual(closed - turn_off_at, 0.10 + 1600 / RATE)
                self.assertEqual(detector.turn_offs, 1, (code, inverted))

    def test_off_frequency_or_at_another_level(self):
        """CML's datasheet allows 134.4 +/- 0.5 Hz; some equipment sends
        about 136 Hz. Louder or quieter than the code, it still counts."""
        from sdr_module.dsp.dcs import dcs_waveform

        for hz, level in ((133.0, 1.0), (136.0, 1.0), (134.4, 0.5), (134.4, 2.0)):
            code = LEVEL * dcs_waveform(212, rate=RATE, seconds=1.0)
            t = np.arange(int(0.18 * RATE)) / RATE
            tone = LEVEL * level * np.sin(2 * np.pi * hz * t)
            x = np.concatenate((code, tone, _carrier_gone(0.5, 1)))
            detector = _detector()
            closed = _close_time(detector, x, 212, False)
            self.assertIsNotNone(closed, (hz, level))
            self.assertLessEqual(closed - 1.0, 0.10, (hz, level))
            self.assertEqual(detector.turn_offs, 1, (hz, level))

    def test_through_noise_while_the_code_still_reads(self):
        """At 9 dB signal to noise in the DCS band (the code reads every
        time), the turn-off code is still heard."""
        rng = np.random.default_rng(11)
        sigma = LEVEL * 10 ** (-9.0 / 20)
        for code in (23, 365, 754):
            x, turn_off_at = _ending(code, seconds=1.5)
            x = x + _band_noise(len(x), sigma, rng)
            detector = _detector()
            closed = _close_time(detector, x, code, False)
            self.assertIsNotNone(closed, code)
            self.assertLessEqual(closed - turn_off_at, 0.10, code)

    def test_the_next_transmission_is_heard_again(self):
        from sdr_module.dsp.dcs import dcs_transmission

        first, _at = _ending(23)
        first = first[: len(first) - len(first) % BLOCK]
        again = LEVEL * dcs_transmission(23, rate=RATE, seconds=1.5, turn_off_s=0.0)
        x = np.concatenate((first, again))
        detector = _detector()
        heard, turned_off = [], []
        for pos in range(0, len(x), BLOCK):
            detector.process(x[pos : pos + BLOCK])
            heard.append(detector.is_present(23, False))
            turned_off.append(detector.turned_off)
        start = len(first) // BLOCK
        self.assertFalse(heard[start])
        self.assertTrue(turned_off[start])
        self.assertIn(True, heard[start : start + 50])  # (within half a second)
        again_at = heard.index(True, start)
        self.assertFalse(any(turned_off[again_at:]))  # (a code heard since)
        self.assertEqual(detector.turn_offs, 1)

    def test_a_turn_off_code_alone_turns_nothing_off(self):
        """A tone near 134.4 Hz with no code before it (a CTCSS station on
        131.8 or 136.5 Hz, a lone tone) is no turn-off code: there is no
        code to turn off, and none is reported."""
        t = np.arange(int(2.0 * RATE)) / RATE
        for hz in (131.8, 134.4, 136.5):
            detector = _detector()
            x = LEVEL * np.sin(2 * np.pi * hz * t)
            for pos in range(0, len(x), 1600):
                detector.process(x[pos : pos + 1600])
                self.assertIsNone(detector.detected, hz)
            self.assertEqual(detector.turn_offs, 0, hz)
            self.assertFalse(detector.turned_off, hz)

    def test_any_block_size_gives_the_same_answers(self):
        x, _at = _ending(31, seconds=1.2)
        x = x + 0.02 * np.random.default_rng(2).standard_normal(len(x))
        x = x[: len(x) - len(x) % 4800]

        def trace(block):
            detector = _detector()
            out = []
            for pos in range(0, len(x), block):
                detector.process(x[pos : pos + block])
                if (pos + block) % 4800 == 0:
                    out.append((detector.detected, detector.turned_off))
            return out

        self.assertEqual(trace(4800), trace(160))
        self.assertEqual(trace(4800), trace(1600))

    def test_reset_forgets_it(self):
        x, _at = _ending(23)
        detector = _detector()
        detector.process(x)
        self.assertTrue(detector.turned_off)
        detector.reset()
        self.assertFalse(detector.turned_off)
        self.assertEqual(detector.turn_offs, 0)


def _band_noise(n, sigma, rng):
    """White noise with an rms of ``sigma`` in the 0-300 Hz DCS band (and
    as much per hertz above it, which the detector filters out): a weak
    station's hiss. Against the code's rms (``LEVEL``) that is its signal
    to noise in the DCS band."""
    return sigma * np.sqrt(RATE / 2 / 300.0) * rng.standard_normal(n)


class TestNoFalseTurnOffs(unittest.TestCase):
    """While the code is sent, nothing else turns the squelch off: voice
    (its pitch sweeps through 134 Hz all the time), a low note held near
    134 Hz, or a weak signal's noise."""

    def run_code(self, x, code=23, inverted=False):
        """(turn-off codes heard, blocks without the code once it was
        heard; None if it never was)."""
        detector = _detector()
        heard = []
        for pos in range(0, len(x), 1600):
            detector.process(x[pos : pos + 1600])
            heard.append(detector.is_present(code, inverted))
        if True not in heard:
            return detector.turn_offs, None
        opened = heard.index(True)
        return detector.turn_offs, heard[opened:].count(False)

    def code(self, seconds, code=23, inverted=False):
        from sdr_module.dsp.dcs import dcs_waveform

        return LEVEL * dcs_waveform(code, inverted, rate=RATE, seconds=seconds)

    def test_long_voice_as_radios_send_it(self):
        """Two minutes of a man's and a woman's voice, open and close
        vowels, through a radio's gentle 300 Hz voice high-pass."""
        for f0, formants, seed in (
            (110.0, CLOSE_VOWEL, 0),
            (130.0, OPEN_VOWEL, 1),
            (220.0, CLOSE_VOWEL, 2),
            (125.0, CLOSE_VOWEL, 3),
        ):
            voice = pitched_voice(30.0, RATE, seed=seed, f0=f0, formants=formants)
            x = self.code(30.0) + _as_radios_send_it(voice)
            offs, gaps = self.run_code(x)
            self.assertEqual((offs, gaps), (0, 0), (f0, seed))

    def test_voice_not_high_passed_at_all(self):
        """A man's voice at full level with nothing filtered out, its pitch
        going through 134 Hz again and again: far stronger there than the
        code."""
        for seed, f0 in ((4, 120.0), (5, 140.0)):
            voice = pitched_voice(30.0, RATE, seed=seed, f0=f0, formants=CLOSE_VOWEL)
            offs, _gaps = self.run_code(self.code(30.0) + voice)
            self.assertEqual(offs, 0, seed)

    def test_low_notes_near_134_hz(self):
        """A note held under the code (music, a hum), at a quarter of the
        code's level up to four times it: never taken for the turn-off
        code. (From about the code's own level up, a note off 134.4 Hz
        keeps the code from being read at all, as before.)"""
        t = np.arange(int(4.0 * RATE)) / RATE
        for hz in (130.8, 134.4, 138.6):  # (C3, the tone itself, C#3)
            for level in (0.25, 0.5, 0.75, 1.0, 2.0, 4.0):
                note = LEVEL * level * np.sin(2 * np.pi * hz * t)
                for code in (23, 116, 172, 445):  # (with long runs of a bit)
                    offs, gaps = self.run_code(self.code(4.0, code) + note, code)
                    self.assertEqual(offs, 0, (hz, level, code))
                    if level <= 0.75:
                        self.assertEqual(gaps, 0, (hz, level, code))

    def test_a_weak_signal(self):
        """A minute of the code in noise, 6 to 12 dB above it in the DCS
        band."""
        rng = np.random.default_rng(7)
        for snr_db in (6.0, 9.0, 12.0):
            sigma = LEVEL * 10 ** (-snr_db / 20)
            x = self.code(20.0, 365) + _band_noise(int(20.0 * RATE), sigma, rng)
            detector = _detector()
            for pos in range(0, len(x), 1600):
                detector.process(x[pos : pos + 1600])
            self.assertEqual(detector.turn_offs, 0, snr_db)


if __name__ == "__main__":
    unittest.main()
