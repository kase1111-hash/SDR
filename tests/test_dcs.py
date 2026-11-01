#!/usr/bin/env python3
"""
DCS (digital-coded squelch, K-A3): the codewords, an encoder written
from the standard, and the detector the tone squelch uses.

The standard (TIA-603 / Motorola DPL): each code sends a 23-bit Golay
(23,12) codeword over and over at 134.4 bit/s, NRZ on the FM carrier
under the voice. The 12 data bits are the code's three octal digits
(9 bits, least significant first) and a fixed 001; the 11 check bits
make the word a multiple of the generator polynomial
x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1. Inverted polarity sends every
bit inverted. The checks here are independent of the module's own
arithmetic: a polynomial division written out below, and pairs of codes
that radio manuals list as the same signal (023 normal is 047 inverted).
"""

import unittest

import numpy as np

from tests.voice_synth import pitched_voice

RATE = 48000.0
BIT_RATE = 134.4
#: x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1, bit i for x^i.
GOLAY = (1 << 11) | (1 << 10) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 2) | 1
#: Codes that send the same bits, one inverted (from radio manuals'
#: DCS tables): 023N = 047I and so on.
SAME_SIGNAL = {
    23: 47,
    25: 244,
    26: 464,
    31: 627,
    32: 51,
    43: 445,
    54: 413,
    65: 271,
    71: 306,
    72: 245,
    73: 506,
    74: 174,
}
FULL = (1 << 23) - 1


def remainder(word, poly=GOLAY):
    """``word`` modulo ``poly`` over GF(2) (bit i for x^i)."""
    degree = poly.bit_length() - 1
    while word and word.bit_length() - 1 >= degree:
        word ^= poly << (word.bit_length() - 1 - degree)
    return word


def rotations(word):
    return {((word >> s) | (word << (23 - s))) & FULL for s in range(23)}


def nrz(code, inverted=False, seconds=1.0, rate=RATE, bit_rate=BIT_RATE, start=0):
    """The bit stream as +1 / -1 levels (1: up), the encoder's job written
    out independently: bit k of the stream is bit (k mod 23) of the word."""
    from sdr_module.dsp.dcs import dcs_codeword

    word = dcs_codeword(code, inverted)
    n = int(seconds * rate)
    k = np.floor((np.arange(n) + start) * bit_rate / rate).astype(int) % 23
    bits = (word >> k) & 1
    return np.where(bits == 1, 1.0, -1.0)


class TestCodewords(unittest.TestCase):
    def test_the_standard_codes(self):
        from sdr_module.dsp.dcs import DCS_CODES

        self.assertEqual(len(DCS_CODES), 104)
        self.assertEqual(DCS_CODES[0], 23)
        self.assertEqual(DCS_CODES[-1], 754)
        self.assertEqual(list(DCS_CODES), sorted(DCS_CODES))
        for code in DCS_CODES:
            self.assertTrue(set(f"{code:03d}") <= set("01234567"), code)
        for missing in (24, 27, 100, 777):
            self.assertNotIn(missing, DCS_CODES)

    def test_data_bits_are_the_octal_code_then_001(self):
        from sdr_module.dsp.dcs import DCS_CODES, dcs_codeword

        for code in DCS_CODES:
            word = dcs_codeword(code)
            self.assertLess(word, 1 << 23)
            self.assertEqual(word & 0x1FF, int(str(code), 8), code)
            self.assertEqual((word >> 9) & 0x7, 0b100, code)  # 0, 0, 1 as sent

    def test_every_word_is_a_golay_codeword(self):
        from sdr_module.dsp.dcs import DCS_CODES, dcs_codeword

        words = [dcs_codeword(code) for code in DCS_CODES]
        for code, word in zip(DCS_CODES, words, strict=True):
            self.assertEqual(remainder(word), 0, code)
        # (23,12) Golay: any two codewords differ in at least 7 bits.
        for i, a in enumerate(words):
            for b in words[i + 1 :]:
                self.assertGreaterEqual(bin(a ^ b).count("1"), 7)
        self.assertEqual(dcs_codeword(23), 0x763813)

    def test_inverted_is_every_bit_inverted(self):
        from sdr_module.dsp.dcs import dcs_codeword

        for code in (23, 115, 754):
            self.assertEqual(dcs_codeword(code, True), dcs_codeword(code) ^ FULL)

    def test_the_pairs_radio_manuals_list_send_the_same_bits(self):
        from sdr_module.dsp.dcs import dcs_codeword, dcs_same_signal

        for normal, inverted in SAME_SIGNAL.items():
            self.assertIn(
                dcs_codeword(inverted, True), rotations(dcs_codeword(normal)), normal
            )
            self.assertEqual(dcs_same_signal(normal, False), (inverted, True))
            self.assertEqual(dcs_same_signal(inverted, True), (normal, False))

    def test_no_standard_code_is_another_one_shifted(self):
        """The receiver can't tell where a word starts: no code may be a
        rotation of another of the same polarity."""
        from sdr_module.dsp.dcs import DCS_CODES, dcs_codeword

        seen = {}
        for code in DCS_CODES:
            for word in rotations(dcs_codeword(code)):
                self.assertNotIn(word, seen, (code, seen.get(word)))
                seen[word] = code

    def test_names(self):
        from sdr_module.dsp.dcs import dcs_label, format_dcs, parse_dcs

        self.assertEqual(format_dcs(23, False), "D023N")
        self.assertEqual(format_dcs(754, True), "D754I")
        self.assertEqual(dcs_label(23, False), "DCS 023 N")
        self.assertEqual(dcs_label(754, True), "DCS 754 I")
        for text, parsed in (
            ("D023N", (23, False)),
            ("d023i", (23, True)),
            ("023", (23, False)),
            ("23", (23, False)),
            ("D754R", (754, True)),  # (R for reverse, as some radios write it)
            (" 115 n ", (115, False)),
            ("DCS 023 I", (23, True)),
        ):
            self.assertEqual(parse_dcs(text), parsed, text)
        for bad in ("", "D024N", "D999N", "08", "D023X", "abc", None, 23.5):
            self.assertIsNone(parse_dcs(bad), bad)
        # Digits other than 0-9 (a damaged setting or scan result) are no
        # code either; they used to raise ValueError ("²³") or pass ("٠٢٣").
        for bad in (
            "D\u00b2\u00b3N",
            "\u00b2\u00b3",
            "\u0660\u0662\u0663",
            "\uff10\uff12\uff13",
        ):
            self.assertIsNone(parse_dcs(bad), bad)


class TestEncoder(unittest.TestCase):
    def test_the_stream_is_the_codeword_least_significant_bit_first(self):
        from sdr_module.dsp.dcs import dcs_waveform

        levels = dcs_waveform(23, rate=RATE, seconds=0.6, smooth=False)
        self.assertTrue(np.array_equal(levels, nrz(23, seconds=0.6)))
        # Bit by bit at the middle of each bit: 1100100000 01 then the check
        # bits, the octal 023 backwards.
        middles = ((np.arange(46) + 0.5) * RATE / BIT_RATE).astype(int)
        bits = (levels[middles] > 0).astype(int)
        self.assertEqual("".join(map(str, bits[:12])), "110010000001")
        self.assertTrue(np.array_equal(bits[:23], bits[23:46]))

    def test_inverted_and_smoothed(self):
        from sdr_module.dsp.dcs import dcs_waveform

        plain = dcs_waveform(54, rate=RATE, seconds=0.5, smooth=False)
        flipped = dcs_waveform(54, True, rate=RATE, seconds=0.5, smooth=False)
        self.assertTrue(np.array_equal(plain, -flipped))
        smooth = dcs_waveform(54, rate=RATE, seconds=0.5)
        spectrum = np.abs(np.fft.rfft(smooth)) ** 2
        f = np.fft.rfftfreq(len(smooth), 1 / RATE)
        # Band-limited as transmitters send it: next to nothing above 300 Hz.
        self.assertLess(spectrum[f > 400].sum(), 1e-3 * spectrum.sum())


def _subaudible_voice(seconds, seed=1):
    """Voice as a transmitter sends it with DCS: nothing below 300 Hz."""
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, RATE, seed=seed)
    return SosFilter(highpass_sos(300.0, RATE)).process(voice)


class TestDetector(unittest.TestCase):
    """The detector on FM discriminator audio (full scale = the deviation):
    DCS at 15 % of it, as radios send it."""

    LEVEL = 0.15

    def detector(self):
        from sdr_module.dsp.dcs import DcsDetector

        return DcsDetector(RATE)

    def feed(self, detector, x, block=1600):
        found = []
        for pos in range(0, len(x), block):
            detector.process(x[pos : pos + block])
            found.append(detector.detected)
        return found

    def signal(self, code, inverted=False, seconds=1.0, **kw):
        from sdr_module.dsp.dcs import dcs_waveform

        return self.LEVEL * dcs_waveform(
            code, inverted, rate=RATE, seconds=seconds, **kw
        )

    def test_every_standard_code_is_recognised(self):
        from sdr_module.dsp.dcs import DCS_CODES, dcs_same_signal

        for code in DCS_CODES:
            detector = self.detector()
            self.feed(detector, self.signal(code, seconds=0.7))
            self.assertEqual(detector.detected, (code, False), code)
            self.assertTrue(detector.is_present(code, False), code)
            # Inverted: heard as the normal code that sends the same bits.
            detector = self.detector()
            self.feed(detector, self.signal(code, True, seconds=0.7))
            self.assertTrue(detector.is_present(code, True), code)
            twin = dcs_same_signal(code, True)
            if twin is not None:
                self.assertEqual(detector.detected, twin, code)

    def test_it_opens_only_on_its_code(self):
        detector = self.detector()
        self.feed(detector, self.signal(23, seconds=1.0))
        self.assertTrue(detector.is_present(23, False))
        self.assertTrue(detector.is_present(47, True))  # (the same bits)
        for other in ((25, False), (23, True), (47, False), (754, False)):
            self.assertFalse(detector.is_present(*other), other)

    def test_under_voice_with_noise_and_a_tuning_offset(self):
        rng = np.random.default_rng(5)
        for code, inverted in ((23, False), (365, False), (754, True)):
            x = self.signal(code, inverted, seconds=3.0)
            x = x + 0.6 * _subaudible_voice(3.0, seed=code % 7)
            x = x + 0.25  # (tuned 1.25 kHz off at 5 kHz deviation)
            x = x + 0.2 * rng.standard_normal(len(x))  # (a weak station's hiss)
            detector = self.detector()
            found = self.feed(detector, x)
            late = found[int(1.0 * RATE / 1600) :]
            expected = (code, inverted)
            if inverted:
                from sdr_module.dsp.dcs import dcs_same_signal

                expected = dcs_same_signal(code, inverted) or expected
            self.assertTrue(all(f == expected for f in late), (code, late))

    def test_a_transmitter_a_little_fast_or_slow(self):
        for bit_rate in (134.0, 134.8):
            detector = self.detector()
            self.feed(detector, self.signal(115, seconds=2.0, bit_rate=bit_rate))
            self.assertEqual(detector.detected, (115, False), bit_rate)

    def test_how_fast_it_opens_and_closes(self):
        quiet = np.zeros(int(1.0 * RATE))
        x = np.concatenate((quiet, self.signal(212, seconds=2.0), quiet))
        detector = self.detector()
        found = self.feed(detector, x, block=480)  # 10 ms steps
        times = [i * 0.01 for i, f in enumerate(found) if f == (212, False)]
        self.assertTrue(times)
        self.assertLess(times[0] - 1.0, 0.5)  # heard within half a second
        self.assertLess(times[-1] - 3.0, 0.45)  # and let go as quickly
        self.assertGreater(times[-1] - 3.0, 0.0)  # (not before it ends)

    def test_nothing_else_is_taken_for_a_code(self):
        from sdr_module.dsp.filters import CTCSS_FREQUENCIES

        rng = np.random.default_rng(9)
        cases = {"voice": 0.8 * pitched_voice(20.0, RATE, seed=3)}
        cases["noise"] = 0.3 * rng.standard_normal(int(10 * RATE))
        t = np.arange(int(1.5 * RATE)) / RATE
        for name, x in cases.items():
            detector = self.detector()
            self.assertTrue(all(f is None for f in self.feed(detector, x)), name)
        for tone in CTCSS_FREQUENCIES:
            detector = self.detector()
            x = self.LEVEL * np.sin(2 * np.pi * tone * t)
            self.assertTrue(all(f is None for f in self.feed(detector, x)), tone)

    def test_any_block_size_gives_the_same_answers(self):
        x = self.signal(31, seconds=1.5)
        x = x + 0.05 * np.random.default_rng(2).standard_normal(len(x))

        def trace(block):
            detector = self.detector()
            out = []
            for pos in range(0, len(x), block):
                detector.process(x[pos : pos + block])
                if (pos + block) % 4800 == 0:
                    out.append(detector.detected)
            return out

        self.assertEqual(trace(4800), trace(160))
        self.assertEqual(trace(4800), trace(1600))

    def test_the_ctcss_detector_takes_no_dcs_for_a_tone(self):
        from sdr_module.dsp.filters import CTCSSDetector

        for code in (23, 65, 134, 245, 365, 506, 754):
            detector = CTCSSDetector(RATE)
            x = self.signal(code, seconds=2.0)
            for pos in range(0, len(x), 1600):
                detector.process(x[pos : pos + 1600])
            self.assertIsNone(detector.detected_tone, code)


if __name__ == "__main__":
    unittest.main()
