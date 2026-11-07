#!/usr/bin/env python3
"""
The CTCSS reverse burst (M-Q2).

Many radios end a CTCSS transmission with a reverse burst (also called
squelch tail elimination): the microphone off, the tone goes on with its
phase shifted (180 degrees for about 150 ms on Kenwood's and most others',
120 degrees for about 180 ms on Motorola's; 135 and 240 are met too) before
the carrier drops, so the radios listening close their tone squelch before
the noise of the dropped carrier (the squelch tail). The tone squelch here
ignored it and closed only when the noise squelch did, after the tail: now
CTCSSDetector hears the shift and lets the tone go 70-80 ms into it.

The burst is looked for only while a tone is held, in the CTCSS band alone
(the voice above 300 Hz left out): each new sample ends three windows of
60 ms, and a burst is there when all three hold the tone and little else,
the latest at the same level as the one before it but shifted by more than
90 degrees, the one before that steady (within 45 degrees). So it is heard
whenever the last 150 ms before it carry no voice; with the voice right up
to it, most of the time (as before when it isn't: the tail is heard). Voice
as radios send it (high-passed below 300 Hz, even gently) never fakes one.
Plain NumPy.
"""

import unittest

import numpy as np

from tests.voice_synth import CLOSE_VOWEL, OPEN_VOWEL, pitched_voice

RATE = 48000.0
LEVEL = 0.12  # 12 % of the deviation, as repeaters send it


def tone_with_burst(
    tone_hz, seconds=2.0, shift_deg=180.0, burst_s=0.15, phase=1.0, error=0.0
):
    """A CTCSS tone for ``seconds``, then ``burst_s`` more of it shifted by
    ``shift_deg`` (over 2 ms, as a transmitter's tone filter smooths the
    step); ``error``: the tone that much off (0.005: 0.5 %). Written out
    here, so these tests don't lean on the encoder."""
    n = int(round((seconds + burst_s) * RATE))
    t = np.arange(n) / RATE
    ramp = np.clip((t - seconds) / 0.002, 0.0, 1.0)
    shift = np.deg2rad(shift_deg) * (0.5 - 0.5 * np.cos(np.pi * ramp))
    return LEVEL * np.cos(2 * np.pi * tone_hz * (1.0 + error) * t + phase + shift)


def voice(seconds, seed=1, f0=120.0, formants=OPEN_VOWEL, high_pass="steep"):
    """Voice as a radio sends it: high-passed at 300 Hz ("steep": 8th
    order, as most do; "gentle": first order, the least any does)."""
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    v = pitched_voice(seconds, RATE, seed=seed, f0=f0, formants=formants)
    if high_pass == "steep":
        v = SosFilter(highpass_sos(300.0, RATE)).process(v)
    elif high_pass == "gentle":
        a = 1.0 / (1.0 + 2 * np.pi * 300.0 / RATE)  # (one RC)
        out = np.empty_like(v)
        prev_x = prev_y = 0.0
        for k, value in enumerate(v):
            prev_y = a * (prev_y + value - prev_x)
            prev_x = value
            out[k] = prev_y
        v = out
    return 0.6 * v / float(np.max(np.abs(v)))


def ending(tone_hz, seconds=2.0, shift_deg=180.0, burst_s=0.15, talk=True, **kw):
    """FM audio of a transmission: the voice (up to the very end) and the
    tone for ``seconds``, the burst, then the carrier gone (loud noise)."""
    seed = kw.pop("seed", 1)
    x = tone_with_burst(tone_hz, seconds, shift_deg, burst_s, **kw)
    if talk:
        x[: int(seconds * RATE)] += voice(seconds, seed)
    gone = 0.6 * np.random.default_rng(seed).standard_normal(int(0.5 * RATE))
    return np.concatenate((x, gone))


def released(x, tone_hz, block=480, at=2.0):
    """``(seconds after at when the detector let the tone go (the end of
    that block), the detector)``; None: never let go (or never held)."""
    from sdr_module.dsp.filters import CTCSSDetector

    detector = CTCSSDetector(RATE)
    held = False
    for pos in range(0, len(x), block):
        detector.process(x[pos : pos + block])
        if detector.is_present(tone_hz):
            held = True
        elif held:
            return (pos + block) / RATE - at, detector
    return None, detector


def band_noise(n, snr_db, seed):
    """White noise whose power in the CTCSS band (0-300 Hz) is ``snr_db``
    below the tone's."""
    sigma = LEVEL / np.sqrt(2) * 10 ** (-snr_db / 20) * np.sqrt(RATE / 2 / 300.0)
    return sigma * np.random.default_rng(seed).standard_normal(n)


class TestTheDetectorHearsIt(unittest.TestCase):
    def test_every_tone_let_go_within_100_ms(self):
        from sdr_module.dsp.filters import CTCSS_FREQUENCIES

        for tone in CTCSS_FREQUENCIES:
            for shift in (180.0, 120.0):
                after, detector = released(ending(tone, shift_deg=shift), tone)
                self.assertIsNotNone(after, (tone, shift))
                self.assertGreaterEqual(after, 0.05, (tone, shift))  # (not before)
                self.assertLessEqual(after, 0.10, (tone, shift))
                self.assertEqual(detector.bursts, 1, (tone, shift))
                self.assertTrue(detector.burst_heard)

    def test_other_shifts_and_a_tone_off_frequency(self):
        for tone in (67.0, 103.5, 167.9, 250.3):
            for shift in (135.0, 240.0):
                after, _d = released(ending(tone, shift_deg=shift), tone)
                self.assertLessEqual(after, 0.10, (tone, shift))
            for error in (-0.005, 0.005):  # (the most the standard allows)
                x = ending(tone, shift_deg=120.0, error=error)
                after, _d = released(x, tone)
                self.assertLessEqual(after, 0.10, (tone, error))

    def test_in_33_ms_blocks_as_the_window_feeds_it(self):
        for tone in (71.9, 127.3, 203.5):
            after, _d = released(ending(tone, seed=3), tone, block=1600)
            self.assertLessEqual(after, 0.10 + 1e-9, tone)

    def talk_until(self, stop_s, count=12):
        """How many of ``count`` transmissions whose voice stopped
        ``stop_s`` before the burst were let go within 100 ms (33 ms
        blocks, as the window feeds it); the rest are let go as before."""
        from sdr_module.dsp.filters import CTCSS_FREQUENCIES

        quick = 0
        for i in range(count):
            tone = CTCSS_FREQUENCIES[(i * 7) % 50]
            x = tone_with_burst(tone, shift_deg=(180.0, 120.0)[i % 2])
            f0 = (100.0, 120.0, 150.0, 200.0, 240.0)[i % 5]
            v = voice(
                2.0, seed=300 + i, f0=f0, formants=(CLOSE_VOWEL, OPEN_VOWEL)[i % 2]
            )
            if stop_s:
                v[-int(stop_s * RATE) :] = 0.0
            x[: len(v)] += v
            gone = 0.6 * np.random.default_rng(i).standard_normal(24000)
            after, _d = released(np.concatenate((x, gone)), tone, block=1600)
            quick += after is not None and after <= 0.10 + 1e-9
        return quick

    def test_voice_until_shortly_before(self):
        """The voice stopped 150 ms before the burst: always heard."""
        self.assertEqual(self.talk_until(0.15), 12)

    def test_voice_to_the_last_moment(self):
        """The radio sends the burst the moment the key is let go: with
        the voice right up to it, most are still heard (87 of 100 measured
        with voice high-passed as most radios send it)."""
        self.assertGreaterEqual(self.talk_until(0.0), 9)

    def test_a_weak_signal(self):
        """Heard down to 6 dB signal to noise in the CTCSS band."""
        for i, tone in enumerate((77.0, 131.8, 192.8)):
            x = ending(tone, talk=False)
            x += band_noise(len(x), 6.0, seed=i)
            after, _d = released(x, tone)
            self.assertLessEqual(after, 0.10, tone)

    def test_without_a_burst_as_before(self):
        """A radio that sends none: the tone is held on as before (the
        noise squelch closes on the dropped carrier)."""
        after, detector = released(ending(100.0, burst_s=0.0), 100.0)
        self.assertTrue(after is None or after > 0.2)
        self.assertEqual(detector.bursts, 0)
        self.assertFalse(detector.burst_heard)


class TestNoFalseBursts(unittest.TestCase):
    def held_throughout(self, x, tone):
        from sdr_module.dsp.filters import CTCSSDetector

        detector = CTCSSDetector(RATE)
        held = []
        for pos in range(0, len(x), 1600):
            detector.process(x[pos : pos + 1600])
            held.append(detector.is_present(tone))
        return detector, held

    def test_voice_over_the_tone(self):
        """Two minutes of voice as radios send it (high-passed below 300 Hz,
        steeply or even gently; the tone off by up to the 0.5 % the
        standard allows): never a burst, the tone held."""
        cases = (
            (77.0, 110.0, CLOSE_VOWEL, "steep", 0.005),
            (136.5, 220.0, CLOSE_VOWEL, "steep", -0.005),
            (165.5, 180.0, OPEN_VOWEL, "gentle", 0.005),
            (173.8, 180.0, OPEN_VOWEL, "gentle", -0.005),
        )
        for i, (tone, f0, formants, high_pass, error) in enumerate(cases):
            v = voice(30.0, seed=40 + i, f0=f0, formants=formants, high_pass=high_pass)
            t = np.arange(len(v)) / RATE
            x = LEVEL * np.cos(2 * np.pi * tone * (1 + error) * t + 0.3) + v
            detector, held = self.held_throughout(x, tone)
            self.assertEqual(detector.bursts, 0, tone)
            self.assertNotIn(False, held[30:], tone)  # (held from the first second)

    def test_a_weak_signal(self):
        for i, tone in enumerate((94.8, 156.7)):
            n = int(30 * RATE)
            t = np.arange(n) / RATE
            x = LEVEL * np.cos(2 * np.pi * tone * t) + band_noise(n, 6.0, seed=50 + i)
            detector, _held = self.held_throughout(x, tone)
            self.assertEqual(detector.bursts, 0, tone)

    def test_another_station_after_a_gap_isnt_one(self):
        """One station drops and another (same tone, another phase) comes
        in 50 ms later: the gap's noise tells them apart. (Back to back, with
        no gap, the phase jump is what a burst is: it would count.)"""
        from sdr_module.dsp.filters import CTCSSDetector

        for k in range(6):
            t = np.arange(int(1.5 * RATE)) / RATE
            a = LEVEL * np.cos(2 * np.pi * 100.0 * t)
            b = LEVEL * np.cos(2 * np.pi * 100.0 * t + np.pi * (0.5 + k / 6))
            gap = 0.6 * np.random.default_rng(k).standard_normal(int(0.05 * RATE))
            x = np.concatenate((a, gap, b))
            detector = CTCSSDetector(RATE)
            for pos in range(0, len(x), 1600):
                detector.process(x[pos : pos + 1600])
            self.assertEqual(detector.bursts, 0, k)


class TestAfterTheBurst(unittest.TestCase):
    def test_found_again_only_in_what_follows(self):
        """The rest of the burst (the shifted tone, the carrier still up
        for a while) doesn't bring the tone back; a new transmission does,
        as any does."""
        from sdr_module.dsp.filters import CTCSSDetector

        burst = tone_with_burst(100.0, seconds=2.0, burst_s=0.25)
        silence = np.zeros(int(0.15 * RATE))
        t = np.arange(int(1.5 * RATE)) / RATE
        again = LEVEL * np.cos(2 * np.pi * 100.0 * t + 2.0)
        x = np.concatenate((burst, silence, again))
        detector = CTCSSDetector(RATE)
        state = []
        for pos in range(0, len(x), 480):
            detector.process(x[pos : pos + 480])
            state.append(
                (
                    (pos + 480) / RATE,
                    detector.is_present(100.0),
                    detector.burst_heard,
                    detector.heard_s,
                )
            )
        start_again = 2.0 + 0.25 + 0.15
        between = [held for at, held, _b, _h in state if 2.1 < at < start_again]
        self.assertNotIn(True, between)
        self.assertTrue(all(b for at, _held, b, _h in state if 2.1 < at < start_again))
        found = next(at for at, held, _b, _h in state if at > 2.1 and held)
        self.assertLess(found - start_again, 0.75)
        self.assertGreater(found - start_again, 0.3)
        self.assertFalse(state[-1][2])  # (a tone found again: no longer)
        self.assertEqual(detector.bursts, 1)
        # heard_s counts from the burst again (its rest not listened to).
        at_burst = [h for at, _held, _b, h in state if 2.08 < at < 2.12]
        self.assertLess(min(at_burst), 0.05)

    def test_reset_forgets_it(self):
        after, detector = released(ending(100.0, talk=False), 100.0)
        self.assertIsNotNone(after)
        detector.reset()
        self.assertEqual(detector.bursts, 0)
        self.assertFalse(detector.burst_heard)
        self.assertEqual(detector.heard_s, 0.0)


class TestTheEncoder(unittest.TestCase):
    def test_the_tone_then_its_burst(self):
        from sdr_module.dsp.filters import (
            CTCSS_BURST_DEG,
            CTCSS_BURST_S,
            ctcss_transmission,
        )

        self.assertEqual((CTCSS_BURST_DEG, CTCSS_BURST_S), (180.0, 0.15))
        x = ctcss_transmission(100.0, rate=RATE, seconds=1.0)
        self.assertEqual(len(x), int(round(1.15 * RATE)))
        self.assertAlmostEqual(float(np.max(np.abs(x))), 1.0, places=3)
        t = np.arange(len(x)) / RATE
        ref = np.exp(-2j * np.pi * 100.0 * t)
        before = np.sum(x[: int(RATE)] * ref[: int(RATE)])
        after = np.sum(x[int(1.01 * RATE) :] * ref[int(1.01 * RATE) :])
        angle = np.degrees(np.angle(after * np.conj(before)))
        self.assertAlmostEqual(abs(angle), 180.0, delta=1.0)
        # (The waveform doesn't jump: the shift is made over 5 ms, the
        # steepest step that of a 260 Hz tone; a jump would be up to 2.)
        self.assertLess(float(np.max(np.abs(np.diff(x)))), 2 * np.pi * 260 / RATE)
        # A Motorola-style 120 degrees, by samples; and none.
        x = ctcss_transmission(
            100.0, rate=RATE, samples=24000, burst_s=0.18, shift_deg=120.0
        )
        self.assertEqual(len(x), 24000 + int(round(0.18 * RATE)))
        after = np.sum(x[24600:] * ref[24600 : len(x)])
        before = np.sum(x[:24000] * ref[:24000])
        angle = np.degrees(np.angle(after * np.conj(before)))
        self.assertAlmostEqual(angle, 120.0, delta=1.0)
        plain = ctcss_transmission(100.0, rate=RATE, seconds=0.5, burst_s=0.0)
        self.assertTrue(np.allclose(plain, np.cos(2 * np.pi * 100.0 * t[:24000])))

    def test_what_it_sends_is_heard(self):
        from sdr_module.dsp.filters import ctcss_transmission

        x = LEVEL * ctcss_transmission(123.0, rate=RATE, seconds=2.0)
        x = np.concatenate((x, np.zeros(4800)))
        after, detector = released(x, 123.0)
        self.assertLessEqual(after, 0.10)
        self.assertEqual(detector.bursts, 1)


class TestTheToneSquelch(unittest.TestCase):
    def run_squelch(self, x, reverse=False, tone=100.0):
        from sdr_module.gui.audio_chain import ToneSquelch

        squelch = ToneSquelch(RATE)
        squelch.tone_hz = tone
        squelch.reverse = reverse
        plays = []
        for pos in range(0, len(x), 1600):
            piece = x[pos : pos + 1600]
            squelch.feed_dcs(piece)
            plays.append(((pos + 1600) / RATE, bool(squelch.process(piece))))
        return plays

    def test_closes_within_100_ms_of_the_burst(self):
        plays = self.run_squelch(ending(100.0))
        self.assertTrue(any(on for at, on in plays if 1.0 < at < 2.0))
        shut = next(at for at, on in plays if at > 2.0 and not on)
        self.assertLessEqual(shut - 2.0, 0.10 + 1e-9)
        self.assertNotIn(True, [on for at, on in plays if at > shut])

    def test_reversed_the_burst_doesnt_open_it(self):
        """The station the reverse tone squelch mutes ends with its burst:
        it decides afresh (muted) until the carrier has dropped."""
        plays = self.run_squelch(ending(100.0), reverse=True)
        self.assertNotIn(True, [on for _at, on in plays])
        # Another station (no tone) is heard after about half a second.
        plays = self.run_squelch(voice(2.0), reverse=True)
        first = next(at for at, on in plays if on)
        self.assertLessEqual(first, 0.6 + 1e-9)


if __name__ == "__main__":
    unittest.main()
