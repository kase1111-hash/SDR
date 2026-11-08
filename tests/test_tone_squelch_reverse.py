#!/usr/bin/env python3
"""
The reverse tone squelch (M-Q3), as CHIRP's TSQL-R and DTCS-R mean it (and
the Icom radios that have them: "when a signal with the matched tone is
received, the squelch closes"): muted while the chosen CTCSS tone or DCS code
is heard, open for every other station. ``ToneSquelch.reverse``.

It opens only once it is sure the tone or code isn't there: as long as the
detectors take to find one (CTCSS 0.5 s, DCS 0.35 s, a weak one a little
later), so a station it mutes isn't heard for the first half second of each
transmission either. Plain NumPy, no window.
"""

import unittest

import numpy as np

from tests.voice_synth import pitched_voice

RATE = 48000.0
BLOCK = 480  # 10 ms


def _voice(seconds, seed=1):
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, RATE, seed=seed)
    return 0.6 * SosFilter(highpass_sos(300.0, RATE)).process(voice)


def station(seconds, ctcss_hz=None, dcs=None, seed=1):
    """FM discriminator audio of a station: voice, with a CTCSS tone (12 %
    of the deviation) or a DCS code (15 %), or neither."""
    from sdr_module.dsp.dcs import dcs_waveform

    x = _voice(seconds, seed)
    if ctcss_hz is not None:
        t = np.arange(len(x)) / RATE
        x = x + 0.12 * np.sin(2 * np.pi * ctcss_hz * t)
    if dcs is not None:
        x = x + 0.15 * dcs_waveform(*dcs, rate=RATE, samples=len(x))
    return x


def squelch(tone_hz=None, dcs=None, reverse=True):
    from sdr_module.gui.audio_chain import ToneSquelch

    tones = ToneSquelch(RATE)
    tones.tone_hz = tone_hz
    tones.dcs = dcs
    tones.reverse = reverse
    return tones


def run(tones, x, block=BLOCK):
    """Whether each 10 ms block may play (the DCS detector fed first, as the
    window does)."""
    out = []
    for pos in range(0, len(x), block):
        piece = x[pos : pos + block]
        tones.feed_dcs(piece)
        out.append(bool(tones.process(piece)))
    return out


def opened_at(plays):
    """Seconds until the squelch first opened (None: never)."""
    return plays.index(True) * BLOCK / RATE if True in plays else None


class TestItIsOffByDefault(unittest.TestCase):
    def test_a_tone_squelch_is_not_reversed_unless_asked(self):
        """(Backwards compatible: code that sets only ``tone_hz`` and
        ``dcs``, as the Second Receiver window does, gets the tone squelch
        it had.)"""
        from sdr_module.gui.audio_chain import ToneSquelch

        tones = ToneSquelch(RATE)
        self.assertFalse(tones.reverse)
        tones.tone_hz = 100.0
        plays = run(tones, station(1.5, ctcss_hz=100.0))
        self.assertTrue(plays[-1])
        self.assertFalse(run(tones, station(1.0))[-1])


class TestReverseCtcss(unittest.TestCase):
    def test_the_station_with_the_tone_is_never_heard(self):
        tones = squelch(tone_hz=100.0)
        plays = run(tones, station(3.0, ctcss_hz=100.0))
        self.assertNotIn(True, plays)  # (its first half second neither)

    def test_every_other_station_is_heard_after_half_a_second(self):
        for tone in (None, 97.4, 103.5, 151.4):  # none, its neighbours, another
            tones = squelch(tone_hz=100.0)
            plays = run(tones, station(2.0, ctcss_hz=tone))
            at = opened_at(plays)
            self.assertIsNotNone(at, tone)
            self.assertLessEqual(at, 0.6, tone)
            self.assertGreaterEqual(at, 0.5, tone)
            self.assertNotIn(False, plays[int(0.6 * RATE / BLOCK) :], tone)

    def test_a_new_channel_decides_afresh(self):
        tones = squelch(tone_hz=100.0)
        self.assertTrue(run(tones, station(1.5))[-1])
        tones.reset()
        self.assertNotIn(True, run(tones, station(2.0, ctcss_hz=100.0)))

    def test_turned_back_to_normal(self):
        tones = squelch(tone_hz=100.0)
        x = station(1.5, ctcss_hz=100.0)
        self.assertFalse(run(tones, x)[-1])
        tones.reverse = False
        self.assertTrue(run(tones, x)[-1])


class TestReverseDcs(unittest.TestCase):
    def test_the_station_with_the_code_is_never_heard(self):
        tones = squelch(dcs=(23, False))
        self.assertNotIn(True, run(tones, station(3.0, dcs=(23, False))))
        # (Its twin of the other polarity sends the same bits: muted too.)
        tones = squelch(dcs=(47, True))
        self.assertNotIn(True, run(tones, station(3.0, dcs=(23, False))))

    def test_every_other_station_is_heard_after_half_a_second(self):
        for dcs in (None, (25, False), (23, True), (754, False)):
            tones = squelch(dcs=(23, False))
            plays = run(tones, station(2.0, dcs=dcs))
            at = opened_at(plays)
            self.assertIsNotNone(at, dcs)
            self.assertLessEqual(at, 0.6, dcs)
            self.assertNotIn(False, plays[int(0.6 * RATE / BLOCK) :], dcs)
        # A CTCSS station too.
        tones = squelch(dcs=(23, False))
        self.assertLessEqual(opened_at(run(tones, station(2.0, ctcss_hz=100.0))), 0.6)

    def test_its_turn_off_code_doesnt_open_it(self):
        """The station it mutes ends with its turn-off code: still muted
        (deciding afresh) until the carrier has dropped."""
        from sdr_module.dsp.dcs import dcs_transmission

        sent = 0.15 * dcs_transmission(23, rate=RATE, seconds=2.0)
        sent[: int(2.0 * RATE)] += _voice(2.0)
        quiet = 0.01 * np.random.default_rng(1).standard_normal(int(0.4 * RATE))
        tones = squelch(dcs=(23, False))
        plays = run(tones, np.concatenate((sent, quiet)))
        self.assertNotIn(True, plays)
        self.assertEqual(tones._dcs_detector.turn_offs, 1)


if __name__ == "__main__":
    unittest.main()
