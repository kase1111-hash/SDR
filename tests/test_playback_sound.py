"""The speaker's sound while a recording plays at another speed (M-P1).

``PlaybackSound`` time-stretches the demodulated audio of a recording
played at 0.5x, 2x or 4x to the pace the recording really plays at, and
keeps the speaker fed the way a sound card takes audio: without gaps and
without a growing delay, also on a computer that can't keep up (the
recording then plays more slowly, and the sound follows). Before, the
speaker was paused at any speed other than 1x.

The frames are driven here as the display loop drives them: regular
frames at 30 a second, or the real ``_Pacer`` with a processing cost per
sample (a computer that can't keep up), on made-up times. The speaker is
modelled after ``AudioSink`` (it pads silence when it runs dry and drops
the oldest audio past its high mark): a gap is time it had nothing to
play, mid-stream.
"""

import unittest

import numpy as np

from sdr_module.gui.audio_sink import AudioSink
from sdr_module.gui.playback_sound import PlaybackSound

try:
    from sdr_module.gui.main_window import _Pacer

    HAS_PACER = True
except ImportError:  # pragma: no cover - the window needs PyQt6
    HAS_PACER = False

RATE = 48000.0
TONE_HZ = 440.0


def _tone(start, count, amplitude=0.5):
    t = (np.arange(count) + start) / RATE
    return (amplitude * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)


class _Speaker:
    """What ``AudioSink`` does with what it is given (see its docstring)."""

    def __init__(self):
        self.queued = 0.0
        self.at = None
        self.started = False
        self.gaps = 0.0
        self.dropped = 0.0
        self.most = 0.0
        self.audio = []

    def advance(self, now):
        if self.at is not None:
            elapsed = now - self.at
            if self.started and self.queued < elapsed:
                self.gaps += elapsed - self.queued
            self.queued = max(0.0, self.queued - elapsed)
        self.at = now

    def write(self, out, now):
        self.advance(now)
        if not len(out):
            return
        self.audio.append(out)
        seconds = len(out) / RATE
        if self.queued == 0.0 and seconds < AudioSink.TARGET_S:
            self.queued = AudioSink.TARGET_S  # (dry: a cushion of silence)
        else:
            self.queued += seconds
        if self.queued > AudioSink.HIGH_S:
            drop = self.queued - max(AudioSink.TARGET_S, seconds)
            self.dropped += drop
            self.queued -= drop
        self.started = True
        self.most = max(self.most, self.queued)

    def stream(self):
        audio = [a if a.ndim == 1 else a[:, 0] for a in self.audio]
        return np.concatenate(audio) if audio else np.zeros(0, np.float32)


def _steps(y):
    """The largest second difference against a clean 440 Hz sine's."""
    clean = 0.5 * (2 * np.pi * TONE_HZ / RATE) ** 2
    return float(np.max(np.abs(np.diff(np.asarray(y, float), 2)))) / clean


class _Run:
    """Frames of a recording played at ``speed``: each frame reads what
    the time since the last one owes (``speed`` times real time), as the
    pacer does for a source that keeps up, ``fps`` frames a second with a
    little jitter; or, given ``load`` (seconds of processing per second of
    signal at 1x), what the real ``_Pacer`` reads on a computer that much
    too slow."""

    def __init__(self, speed, load=None, rate=2.4e6, seed=0):
        self.speed = float(speed)
        self.load = load
        self.rate = float(rate)
        self.sound = PlaybackSound(RATE, AudioSink.TARGET_S)
        self.speaker = _Speaker()
        self.rng = np.random.default_rng(seed)
        self.pacer = _Pacer() if load is not None else None
        self.now = 100.0
        self.position = 0.0  # audio samples of the recording played
        self.owed = 0.0
        self.last = None
        self.ratios = []
        self.delays = []  # (time, seconds of sound waiting)

    def frames(self, seconds, stall=None, signal=None):
        """Run ``seconds`` of frames (``stall``: a frame that late once)."""
        end = self.now + seconds
        while self.now < end:
            start = self.now
            if stall is not None and stall[0] <= self.now - 100.0:
                self.now += stall[1]
                stall = None
                start = self.now
            count = self.read()
            if count:
                n = count * RATE / self.rate
                first = int(self.position)
                self.position += n
                block = (signal or _tone)(first, int(self.position) - first)
                self.sound.note(len(block) / RATE, self.now, self.speed)
                out = self.sound.play(block, self.now)
                self.speaker.write(out, self.now)
                if self.sound.stretching:
                    self.ratios.append(self.sound.ratio)
                self.delays.append((self.now, self.speaker.queued + self.waiting()))
                if self.pacer is not None:
                    work = self.load * count / self.rate
                    self.now += work
                    self.pacer.measured(count, self.rate * self.speed, work)
            tick = (1 / 30.0) * (1.0 + 0.15 * self.rng.uniform(-1.0, 1.0))
            self.now = max(self.now, start + tick)
        self.speaker.advance(self.now)

    def read(self):
        if self.pacer is not None:
            return self.pacer.paced_request(self.rate * self.speed, self.now, self.rate)
        if self.last is None:
            self.last = self.now
            return 2048
        self.owed += self.rate * self.speed * (self.now - self.last)
        self.last = self.now
        if self.owed < 2048:
            return 0
        # (As the pacer: at most 0.25 s of signal a frame; what a stalled
        # frame couldn't take is dropped.)
        count = int(min(self.owed, 0.25 * self.rate * self.speed))
        self.owed = min(self.owed - count, 2048.0)
        return count

    def waiting(self):
        """Seconds of sound the stretch still holds (input at the ratio)."""
        sound = self.sound
        if not sound.stretching:
            return 0.0
        return sound.waiting_s

    def delay_after(self, t):
        return [d for when, d in self.delays if when - 100.0 >= t]


class TestOneX(unittest.TestCase):
    def test_a_recording_that_keeps_up_at_1x_plays_as_it_is(self):
        sound = PlaybackSound(RATE, AudioSink.TARGET_S)
        now = 10.0
        for i in range(60):
            block = _tone(1600 * i, 1600)
            sound.note(len(block) / RATE, now, 1.0)
            out = sound.play(block, now)
            self.assertIs(out, block)  # (bit for bit, no copy)
            self.assertFalse(sound.stretching)
            now += 1600 / RATE
        self.assertEqual(len(sound.finish()), 0)


class TestEverySpeedPlays(unittest.TestCase):
    """At 0.5x, 2x and 4x on a computer that keeps up."""

    def test_the_speaker_never_runs_dry_and_the_delay_doesnt_grow(self):
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                run = _Run(speed)
                run.frames(20.0)
                self.assertEqual(run.speaker.gaps, 0.0)
                self.assertEqual(run.speaker.dropped, 0.0)
                self.assertLess(run.speaker.most, AudioSink.HIGH_S)
                # The ratio is the speed, steadily.
                ratios = np.array(run.ratios[len(run.ratios) // 4 :])
                self.assertAlmostEqual(float(ratios.mean()), speed, delta=0.01 * speed)
                self.assertLess(float(ratios.std()), 0.02 * speed)
                # A short wait (at most about a third of a second), the
                # same at the end as after the first seconds.
                early, late = run.delay_after(3.0)[:30], run.delay_after(17.0)
                self.assertLess(max(late), 0.35)
                self.assertAlmostEqual(np.mean(late), np.mean(early), delta=0.03)

    def test_the_sound_keeps_its_pitch_and_never_steps(self):
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                run = _Run(speed)
                run.frames(4.0)
                y = run.speaker.stream()
                self.assertLess(_steps(y), 1.05)
                middle = y[len(y) // 4 : 3 * len(y) // 4].astype(float)
                pad = 8 * len(middle)
                s = np.abs(np.fft.rfft(middle * np.hanning(len(middle)), pad))
                peak = np.argmax(s) * RATE / pad
                self.assertAlmostEqual(peak, TONE_HZ, delta=0.5)

    def test_a_slow_recording_too(self):
        # A 48 kHz recording (HDSDR, SDR#) comes a display block (2048
        # samples) every few frames at 0.5x: the stretch holds enough to
        # play on between them.
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                run = _Run(speed, rate=48e3)
                run.frames(10.0)
                self.assertEqual(run.speaker.gaps, 0.0)
                self.assertEqual(run.speaker.dropped, 0.0)
                self.assertLess(max(run.delay_after(2.0)), 0.4)


@unittest.skipUnless(HAS_PACER, "the display loop's pacer needs PyQt6")
class TestAComputerThatCantKeepUp(unittest.TestCase):
    """The recording plays more slowly than chosen: the sound follows."""

    def test_4x_too_much_plays_what_it_manages_without_gaps(self):
        # 4x needs 1.2x real time here: the pacer reads about 1x a frame
        # and a probe frame at 4x once a second.
        run = _Run(4.0, load=0.3)
        run.frames(20.0)
        self.assertTrue(run.pacer.overloaded)
        pace = run.position / RATE / 20.0
        self.assertLess(pace, 2.0)
        self.assertEqual(run.speaker.gaps, 0.0)
        self.assertEqual(run.speaker.dropped, 0.0)
        ratios = np.array(run.ratios[len(run.ratios) // 4 :])
        self.assertAlmostEqual(float(ratios.mean()), pace, delta=0.1 * pace)
        # A steady tempo: the probe's 0.2 s is given out over the second
        # after it, not in a rush (and no dip before the next one).
        self.assertLess(float(ratios.std()), 0.08 * pace)
        self.assertLess(max(run.delay_after(5.0)), 0.45)

    def test_1x_too_much_stretches_to_what_it_manages(self):
        # At 1x the sound used to run dry again and again (a 0.12 s gap
        # every half second or so at 0.8x): now it plays at 0.8x.
        run = _Run(1.0, load=1.25)
        run.frames(15.0)
        self.assertTrue(run.sound.stretching)
        self.assertLess(run.speaker.gaps, 0.02)  # (taking over: a few ms)
        self.assertEqual(run.speaker.dropped, 0.0)
        ratios = np.array(run.ratios[len(run.ratios) // 4 :])
        self.assertAlmostEqual(float(ratios.mean()), 0.8, delta=0.05)

    def test_back_at_1x_it_plays_as_it_is_again(self):
        run = _Run(1.0, load=1.25)
        run.frames(6.0)
        self.assertTrue(run.sound.stretching)
        run.load = 0.05
        run.frames(12.0)
        self.assertFalse(run.sound.stretching)
        self.assertLess(_steps(run.speaker.stream()), 1.05)
        # As it is again, bit for bit.
        block = _tone(0, 1600)
        run.sound.note(len(block) / RATE, run.now, 1.0)
        self.assertIs(run.sound.play(block, run.now), block)

    def test_catching_up_again_at_4x_doesnt_pile_up_a_delay(self):
        run = _Run(4.0, load=0.3)
        run.frames(8.0)
        run.load = 0.05
        run.frames(12.0)
        self.assertLess(max(run.delay_after(8.0)), 0.5)
        self.assertLess(np.mean(run.delay_after(16.0)), 0.25)
        self.assertEqual(run.speaker.gaps, 0.0)


class TestStallsAndChanges(unittest.TestCase):
    def test_after_a_stall_the_delay_comes_back_down(self):
        # The window froze for 0.6 s: the speaker ran dry meanwhile (what
        # no sound card can avoid), then plays on, and the wait settles
        # back to where it was.
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                run = _Run(speed)
                run.frames(12.0, stall=(5.0, 0.6))
                self.assertLess(run.speaker.gaps, 0.6)
                self.assertEqual(run.speaker.dropped, 0.0)
                before = np.mean([d for t, d in run.delays if 3 < t - 100 < 5])
                after = np.mean(run.delay_after(10.0))
                self.assertAlmostEqual(after, before, delta=0.03)

    def test_another_speed_mid_sound_is_seamless(self):
        run = _Run(1.0)
        run.frames(2.0)
        for speed in (2.0, 4.0, 0.5, 1.0):
            run.speed = speed
            run.frames(3.0)
        self.assertEqual(run.speaker.gaps, 0.0)
        self.assertEqual(run.speaker.dropped, 0.0)
        self.assertLess(_steps(run.speaker.stream()), 1.05)
        self.assertFalse(run.sound.stretching)  # (back at 1x: as it is)

    def test_the_pace_follows_the_speed_chosen(self):
        sound = PlaybackSound(RATE, AudioSink.TARGET_S)
        sound.note(0.0, 0.0, 2.0)
        self.assertEqual(sound.speed, 2.0)
        self.assertAlmostEqual(sound.pace, 2.0)
        now = 0.0
        for _ in range(200):  # (the recording really plays at 1.5x)
            now += 1 / 30
            sound.note(1.5 / 30, now, 2.0)
        self.assertAlmostEqual(sound.pace, 1.5, delta=0.02)
        # A frame a second late (a stall) isn't the pace.
        now += 1.0
        sound.note(1.5 * 0.25, now, 2.0)
        self.assertAlmostEqual(sound.pace, 1.5, delta=0.02)


class TestTheSoundEnds(unittest.TestCase):
    def test_finish_plays_what_is_left_briefly_and_fades(self):
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                run = _Run(speed)
                run.frames(2.0)
                # (The window's AudioJoin fades the last block out.)
                fade = _tone(int(run.position), 240) * np.cos(
                    np.linspace(0, np.pi / 2, 240)
                ).astype(np.float32)
                run.speaker.write(run.sound.play(fade, run.now), run.now)
                tail = run.sound.finish()
                self.assertLessEqual(len(tail) / RATE, PlaybackSound.FLUSH_S + 0.026)
                self.assertLess(abs(float(tail[-1])), 1e-3)
                run.speaker.write(tail, run.now)
                # (A 5 ms fade bends a sine up to about 1.25 times as much
                # as the sine itself does; a click, a step, many times.)
                self.assertLess(_steps(run.speaker.stream()), 1.5)
                self.assertFalse(run.sound.stretching)
                self.assertEqual(run.sound.waiting_s, 0.0)

    def test_the_next_sound_starts_afresh(self):
        run = _Run(2.0)
        run.frames(2.0)
        run.sound.finish()
        run.speaker = _Speaker()
        run.frames(3.0)
        self.assertEqual(run.speaker.gaps, 0.0)
        self.assertLess(_steps(run.speaker.stream()), 1.05)

    def test_stereo(self):
        def stereo(first, count):
            left = _tone(first, count)
            return np.stack([left, 0.5 * left], axis=1)

        run = _Run(2.0)
        run.frames(3.0, signal=stereo)
        out = run.speaker.audio[-1]
        self.assertEqual(out.ndim, 2)
        self.assertEqual(run.speaker.gaps, 0.0)
        y = np.concatenate(run.speaker.audio)
        np.testing.assert_allclose(y[:, 1], 0.5 * y[:, 0], atol=1e-6)

    def test_reset_forgets_everything(self):
        run = _Run(4.0)
        run.frames(2.0)
        run.sound.reset()
        self.assertFalse(run.sound.stretching)
        self.assertEqual(run.sound.waiting_s, 0.0)
        self.assertAlmostEqual(run.sound.pace, 4.0)


if __name__ == "__main__":
    unittest.main()
