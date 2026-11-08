"""The files lane (M-F1, M-F2) and the playback lane (M-P1) together, and
the bookmarks the files and squelch (M-Q3) lanes both changed.

* A one-channel WAV file is read as I/Q at half its rate (M-F2): 8 kHz
  audio is 4 kS/s of I/Q, so the display loop reads a display block (2048
  samples, half a second of it) every 16 frames at 1x and every 31 at
  0.5x. Its sound at another speed (M-P1, time-stretched) was given out
  only with each block, a quarter of a second at most, and the pace it
  follows was measured from those blocks alone (a frame more than a
  quarter of a second after the last counted as a stall): the speaker ran
  dry most of the time, 3.5 s of every 4 for 8 kHz audio at 0.5x, 2.8 s
  for 22.05 kHz audio, 2.2 s for 8 kHz audio at 2x.
* Bookmarks below 1 kHz and above 6 GHz keep their frequency (M-F2) and
  bookmarks keep a reverse tone squelch, CHIRP's TSQL-R (M-Q3): one that
  is both, made while a recording above 6 GHz plays, is edited, exported
  to CHIRP CSV and imported again, and tuned while the recording plays.
"""

import io
import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_playback_sound import (  # noqa: E402
    AUDIO,
    _marked,
    _marker_seconds,
    _peak_hz,
    _SoundCase,
)
from tests.test_gui_wav_formats import AUDIO_RATE, _WavCase  # noqa: E402
from tests.test_playback_sound import _Speaker  # noqa: E402
from tests.wav_builders import iq_wav  # noqa: E402

#: A recording above the receiver's range (6 GHz): the 3 cm band.
X_BAND = 10.368e9
#: Its rate (I/Q) and the station's offset from its middle.
X_RATE = 240000
X_OFFSET = 30e3


def _tone_wav(path, rate, seconds, freq=1000.0, bursts=None):
    """A one-channel WAV file of a ``freq`` tone at ``rate`` (a little
    noise), or of tone bursts ``(on_s, off_s)``."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    x = 0.3 * np.sin(2 * np.pi * freq * t)
    if bursts is not None:
        on, off = bursts
        x = np.where((t % (on + off)) < on, x, 0.0)
    x = x + 0.001 * np.random.default_rng(4).standard_normal(n)
    iq_wav(path, x, "int", 2, int(rate))
    return path


def _marked_wav(path, rate, seconds, mark_s):
    """A one-channel WAV file of :func:`_marked`'s tone: its last
    ``mark_s`` at 600 Hz."""
    x = 0.3 * _marked(seconds, mark_s, rate)
    x = x + 0.001 * np.random.default_rng(4).standard_normal(len(x))
    iq_wav(path, x, "int", 2, int(rate))
    return path


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestSlowRecordingsAtOtherSpeeds(_SoundCase):
    """Sound at 0.5x and 2x of recordings slower than a display block a
    frame: telephone-rate and other low-rate audio in one-channel files."""

    def open_audio(self, rate, seconds=6.0, bursts=None, path=None):
        path = path or _tone_wav(
            os.path.join(self.folder, f"audio{rate}.wav"), rate, seconds, bursts=bursts
        )
        self.play(path)
        win = self.win
        self.assertTrue(win._playback.active, self.message())
        # (Heard at its own pitch: USB from 0 Hz, the 1 kHz tone at 1 kHz.)
        self.assertEqual(win._control_panel.get_demod_mode(), "USB")
        win._playback_bar.set_loop(True)
        win._control_panel.set_squelch_enabled(False)
        win._control_panel.set_audio_agc_speed("off")
        win._rx_history.clear()
        win._pacer.restart()

    def ticked_frames(self, seconds, tick, stop=None):
        """:meth:`frames`, with the playback's own timer ticked every 200
        ms from ``tick`` (its end, its position); ``stop``: no more frames
        (and no tick) after the frame where it turns true. Returns when it
        last ticked."""
        win = self.win
        end = self.clock.now + seconds
        while self.clock.now < end - 1e-9:
            start = self.clock.now
            win._update_display()
            done = stop is not None and stop()
            if not done and self.clock.now - tick >= win._playback._TICK_MS / 1e3:
                tick = self.clock.now
                win._playback._tick()
            self.clock.now = max(self.clock.now, start + 0.033)
            if done:
                break
        return tick

    def test_low_rate_audio_plays_without_gaps(self):
        win = self.win
        for rate, speed in ((8000, 0.5), (8000, 2.0), (11025, 0.5), (22050, 0.5)):
            with self.subTest(rate=rate, speed=speed):
                self.heard.clear()
                self.open_audio(rate)
                win._playback_bar.set_speed(speed)
                self.frames(2.0)
                device = win._device
                before = device.position + device.loops * device.total_samples
                start = self.clock.now
                self.frames(4.0)
                gaps, dropped = self.troubles_since(start)
                self.assertLess(gaps, 0.01)
                self.assertEqual(dropped, 0.0)
                sound = self.sound(since=start)
                self.assertAlmostEqual(len(sound) / AUDIO, 4.0, delta=0.15)
                self.assertAlmostEqual(_peak_hz(sound[: int(AUDIO)]), 1000.0, delta=2.0)
                # At the speed chosen: 4 s of the recording at 0.5x, 2 s ...
                after = device.position + device.loops * device.total_samples
                played = (after - before) / device.sample_rate
                self.assertAlmostEqual(played, 4.0 * speed, delta=2048 / (rate / 2))
                win._playback.close()

    def test_from_1x_to_another_speed_without_a_gap(self):
        """At 1x such a recording's blocks are heard as they are (half a
        second each for 8 kHz audio); another speed chosen then takes over
        that sound with the time-stretch. Its blocks are as far apart as
        before, so what the stretch keeps for the frames between them must
        be a block's worth from the first (kept for a pile-up's catch-up,
        it ran out: the speaker ran dry for 0.1 to 0.3 s in the first
        second)."""
        win = self.win
        for rate, speed in ((8000, 0.5), (8000, 4.0), (22050, 2.0), (22050, 4.0)):
            with self.subTest(rate=rate, speed=speed):
                # (The last one's sound has played out.)
                self.clock.advance(1.0)
                self.heard.clear()
                self.open_audio(rate, seconds=20.0)
                self.frames(3.0)
                start = self.clock.now
                win._playback_bar.set_speed(speed)
                self.frames(4.0)
                gaps, dropped = self.troubles_since(start)
                self.assertLess(gaps, 0.01)
                self.assertEqual(dropped, 0.0)
                win._playback.close()

    def test_a_sound_plays_on_between_blocks_from_its_start(self):
        """Tone bursts in 8 kHz audio at 0.5x with the squelch on: within
        each burst's sound (from the frame the squelch opens to the one it
        closes in) the speaker never runs dry, its first block included
        (taken for a pile-up of the stretch, it was rushed through)."""
        win = self.win
        self.open_audio(8000, seconds=10.0, bursts=(1.5, 1.0))
        panel = win._control_panel
        panel.set_squelch_auto(False)
        panel.set_squelch_enabled(True)
        panel.set_squelch_db(-40.0)
        update = win._update_display
        opened = []  # (fake time, the squelch open after the frame)

        def frame():
            update()
            opened.append((self.clock.now, win._squelch.is_open))

        win._update_display = frame
        win._playback_bar.set_speed(0.5)
        self.frames(1.0)
        start = self.clock.now
        self.frames(12.0)
        opens = sum(
            1
            for i in range(1, len(opened))
            if not opened[i - 1][1] and opened[i][1] and opened[i][0] >= start
        )
        self.assertGreaterEqual(opens, 2)
        model = _Speaker()
        model.listening = False
        state = dict(opened)
        for when, block in self.heard:
            if when < start:
                continue
            model.write(block, when)
            if len(block):
                model.listening = state.get(when, False)
        self.assertLess(model.gaps, 0.01)
        win._playback.close()

    def test_its_sound_ends_at_once_when_this_receiver_isnt_heard(self):
        """The Second Receiver's Listen playing only that one, or the Audio
        switched off, between two blocks of 8 kHz audio at 0.5x (a block a
        second): the sound's end (what the stretch holds, faded out) comes
        at once, and nothing after it, neither the sound between blocks
        nor its end again with the next block (a quarter of a second of
        this receiver, up to a second after its sound had stopped, over
        the Second Receiver's). Heard again, it plays on."""
        win = self.win
        self.open_audio(8000)
        win._playback_bar.set_speed(0.5)
        self.frames(2.0)

        def later(start):
            """What the speaker got after the frame at ``start``."""
            return [
                (round(t - start, 3), round(len(b) / AUDIO, 3))
                for t, b in self.heard
                if len(b) and t >= start + 0.03
            ]

        for phase in (0.0, 0.3, 0.6):  # (somewhere between two blocks)
            with self.subTest(phase=phase):
                self.frames(phase)
                self.assertTrue(win._playback_sound.stretching)
                with mock.patch.object(win._second_rx, "mutes_main", return_value=True):
                    start = self.clock.now
                    self.heard.clear()
                    self.frames(2.2)  # (two blocks come)
                    self.assertEqual(later(start), [])
                    self.assertLess(len(self.sound()) / AUDIO, 0.15)
                start = self.clock.now
                self.frames(2.0)
                self.assertGreater(len(self.sound(since=start)) / AUDIO, 1.0)
        self.frames(0.3)
        start = self.clock.now
        self.heard.clear()
        win._set_audio_enabled(False)
        self.frames(2.2)
        self.assertEqual(later(start), [])
        win._set_audio_enabled(True)
        start = self.clock.now
        self.frames(2.0)
        self.assertGreater(len(self.sound(since=start)) / AUDIO, 1.0)
        win._playback.close()

    def test_at_1x_too_its_sound_ends_at_once_when_this_receiver_isnt_heard(self):
        """At 1x the rest of a block of 8 kHz audio is given out as it
        plays (dsp-1): Listen in the Second Receiver window playing only
        that one, or the Audio switched off, between two blocks ends it at
        once (what is held, faded out), with nothing after it (it came back
        with the next block, up to half a second later). Heard again, it
        plays on."""
        win = self.win
        self.open_audio(8000)
        self.frames(2.0)

        def later(start):
            """What the speaker got after the frame at ``start``."""
            return [
                (round(t - start, 3), round(len(b) / AUDIO, 3))
                for t, b in self.heard
                if len(b) and t >= start + 0.03
            ]

        for phase in (0.0, 0.2, 0.4):  # (somewhere between two blocks)
            with self.subTest(phase=phase):
                self.frames(phase)
                self.assertGreater(win._playback_sound.waiting_s, 0.0)
                with mock.patch.object(win._second_rx, "mutes_main", return_value=True):
                    start = self.clock.now
                    self.heard.clear()
                    self.frames(1.2)  # (two blocks come)
                    self.assertEqual(later(start), [])
                    self.assertLess(len(self.sound()) / AUDIO, 0.15)
                    self.assertFalse(win._playback_sound.holding)
                start = self.clock.now
                self.frames(2.0)
                self.assertGreater(len(self.sound(since=start)) / AUDIO, 1.0)
        self.frames(0.3)
        start = self.clock.now
        self.heard.clear()
        win._set_audio_enabled(False)
        self.frames(1.2)
        self.assertEqual(later(start), [])
        win._set_audio_enabled(True)
        start = self.clock.now
        self.frames(2.0)
        self.assertGreater(len(self.sound(since=start)) / AUDIO, 1.0)
        win._playback.close()

    def test_1x_plays_the_recording_as_it_is(self):
        """At 1x a recording that keeps up is heard as it is: its blocks'
        own samples, in order, not stretched. A block longer than the
        speaker holds (8 kHz audio: half a second) is given out over the
        frames until the next one, never more at once than the speaker
        holds (handed over whole, AudioSink dropped the start of the next
        block or ran dry before it, about twice a second: dsp-1)."""
        win = self.win
        self.open_audio(8000)
        self.heard.clear()
        sound = win._playback_sound
        given = []
        play = sound.play

        def spy(block, now, last=False):
            given.append(np.array(block, dtype=np.float32))
            return play(block, now, last=last)

        with mock.patch.object(sound, "play", side_effect=spy):
            self.frames(3.0)
        self.assertFalse(sound.stretching)
        writes = [b for _t, b in self.heard if len(b)]
        self.assertGreater(len(writes), 60)  # (frame by frame)
        for block in writes:
            self.assertLessEqual(len(block) / AUDIO, sound.MOST_HELD_S)
        heard, fed = np.concatenate(writes), np.concatenate(given)
        np.testing.assert_array_equal(heard, fed[: len(heard)])
        self.assertGreater(len(heard), len(fed) - int(0.6 * AUDIO))
        win._playback.close()

    def test_1x_without_gaps_on_a_timer_that_varies(self):
        """Qt's 33 ms display timer fires every 32 ms or so, a millisecond
        either way: 8 and 11.025 kHz audio at 1x, a block every 16 or 17
        frames (12 for 11.025 kHz), plays without a gap or a drop (it had
        32 ms dropouts and skips about twice a second: dsp-1)."""
        win = self.win
        rng = np.random.default_rng(3)

        def frames(seconds):
            end = self.clock.now + seconds
            while self.clock.now < end - 1e-9:
                start = self.clock.now
                win._update_display()
                tick = 0.032 + float(rng.uniform(-0.001, 0.001))
                self.clock.now = max(self.clock.now, start + tick)

        for rate in (8000, 11025):
            with self.subTest(rate=rate):
                self.clock.advance(1.0)  # (the last one's sound has played)
                self.heard.clear()
                self.open_audio(rate)
                frames(1.5)
                start = self.clock.now
                frames(5.0)
                gaps, dropped = self.troubles_since(start)
                self.assertLess(gaps, 0.005)
                self.assertEqual(dropped, 0.0)
                self.assertFalse(win._playback_sound.stretching)
                win._playback.close()

    def test_22050_hz_audio_starts_without_a_gap(self):
        """22.05 kHz audio at 1x, a block every 6 frames (0.19 s): its
        blocks fit the speaker and pass through as they are, but the first
        went whole to a speaker that held nothing, with no silence in
        front: each block then lasted until the next, and on a timer 4 ms
        irregular the speaker ran dry for 30 ms in the first seconds."""
        win = self.win
        for seed in range(4):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                self.clock.advance(1.0)  # (the last one's sound has played)
                self.heard.clear()
                self.open_audio(22050)
                opened = self.clock.now
                end = opened + 4.0
                while self.clock.now < end - 1e-9:
                    start = self.clock.now
                    win._update_display()
                    tick = 0.032 + float(rng.uniform(-0.004, 0.004))
                    self.clock.now = max(self.clock.now, start + tick)
                gaps, dropped = self.troubles_since(opened)
                self.assertLess(gaps, 0.005)
                self.assertEqual(dropped, 0.0)
                win._playback.close()

    def test_1x_plays_a_recording_to_its_very_end(self):
        """Without Loop, 8 and 11.025 kHz audio at 1x plays to its last
        sample: the window stops once the sound has been given out, and
        its end follows it at once. (It stopped at the bar's first tick
        after the end of the file was read, with the last block given out
        as the speaker plays it: up to a quarter of a second of the
        recording's end was cut off.)"""
        win = self.win
        for rate in (8000, 11025):
            for phase in (0.0, 0.05, 0.1, 0.15):  # (when the bar's timer ticks)
                with self.subTest(rate=rate, phase=phase):
                    self.clock.advance(1.0)  # (the last one's sound has played)
                    seconds = 6 * 2 * 2048 / rate  # (six display blocks)
                    self.open_audio(rate, seconds=seconds)
                    win._playback_bar.set_loop(False)
                    self.heard.clear()
                    self.ticked_frames(seconds + 1.0, self.clock.now - phase)
                    self.assertFalse(win._is_running)
                    self.assertIn("End of audio", self.message())
                    self.assertGreater(len(self.sound()) / AUDIO, seconds - 0.01)
                    # (Nothing dry before the end: its decay came with it.)
                    last = max(when for when, block in self.heard if len(block))
                    model = _Speaker()
                    for when, block in self.heard:
                        model.write(block, when)
                    model.advance(last)
                    self.assertLess(model.gaps, 0.005)
                    self.assertEqual(model.dropped, 0.0)
                    win._playback.close()

    def test_every_speed_plays_a_recording_to_its_very_end(self):
        """Without Loop, 8 kHz audio plays its last block at the speed
        chosen, its pitch kept, and then the playback pauses (R8-06). At
        0.5x, 2x and 4x what the time-stretch held of it when the end was
        read went out at once when the playback paused, in at most 0.1 s:
        rushed, or cut short and faded (its last 0.3 s lasted 0.04-0.05 s
        at 0.5x, not 0.6 s, and 0.03-0.05 s at 2x). At 1x a last block
        short of a whole one went into the time-stretch (the pace measured
        sank with it), and out the same way."""
        win = self.win
        mark_s = 0.3
        # (These frames read 2048 samples, then 2112 at a time: the last
        # block holds 1500, the 0.3 s marked among them.)
        seconds = 2 * (2048 + 5 * 2112 + 1500) / 8000
        for speed in (0.5, 1.0, 2.0, 4.0):
            for phase in (0.0, 0.1):  # (when the bar's timer ticks)
                with self.subTest(speed=speed, phase=phase):
                    self.clock.advance(1.0)  # (the last one's sound has played)
                    path = os.path.join(self.folder, "end.wav")
                    self.open_audio(8000, path=_marked_wav(path, 8000, seconds, mark_s))
                    win._playback_bar.set_loop(False)
                    win._playback_bar.set_speed(speed)
                    win._rx_history.clear()
                    win._pacer.restart()
                    self.heard.clear()
                    paused = []

                    def stop(paused=paused):
                        if not win._is_running and not paused:
                            paused.append(self.clock.now)
                        return False

                    tick = self.clock.now - phase
                    self.ticked_frames(seconds / speed + 2.0, tick, stop)
                    self.assertFalse(win._is_running)
                    self.assertEqual(win._state_badge.text(), "ENDED")
                    self.assertIn("End of end.wav", self.message())
                    # Its last 0.3 s as long as the speed makes it...
                    self.assertAlmostEqual(
                        _marker_seconds(self.sound()),
                        mark_s / speed,
                        delta=0.1 * mark_s / speed + 0.02,
                    )
                    # ...with no gap before its end, which fades out...
                    writes = [(t, b) for t, b in self.heard if len(b)]
                    model = _Speaker()
                    for when, block in writes:
                        model.write(block, when)
                    self.assertLess(model.gaps, 0.005)
                    self.assertEqual(model.dropped, 0.0)
                    peak = float(np.max(np.abs(self.sound())))
                    self.assertLess(abs(float(writes[-1][1][-1])), 0.01 * peak)
                    # ...and nothing after it: then the playback paused.
                    self.assertLessEqual(writes[-1][0], paused[0])
                    win._playback.close()

    def test_loop_turned_on_as_the_end_plays_plays_on(self):
        """Loop turned on while the last of a recording's sound still plays
        (its end read, the window waiting for that sound): it plays on from
        the start, as with Loop on before (the end, read without it, still
        stopped it at the playback's next tick, though it had started
        again). (The file's last block, short of a whole one, may leave the
        speaker dry for that much before the start.)"""
        win = self.win
        seconds = 6 * 2 * 2048 / 8000
        self.open_audio(8000, seconds=seconds)
        win._playback_bar.set_loop(False)
        device = win._device
        tick = self.ticked_frames(
            seconds + 1.0, self.clock.now, lambda: device.finished
        )
        self.assertTrue(win._is_running)
        win._playback_bar.set_loop(True)
        start = self.clock.now
        self.ticked_frames(3.0, tick)
        self.assertTrue(win._is_running)
        self.assertFalse(device.finished)
        self.assertGreaterEqual(device.loops, 1)
        self.assertGreater(len(self.sound(since=start)) / AUDIO, 2.5)
        gaps, dropped = self.troubles_since(start + 1.0)  # (from the start on)
        self.assertLess(gaps, 0.005)
        self.assertEqual(dropped, 0.0)
        win._playback.close()

    def test_stalls_at_1x_dont_make_the_sound_later_and_later(self):
        """A stall of the window at 1x: the speaker runs dry, and starts
        again where the sound is now, as late as before. 22.05 kHz audio
        plays its blocks as they are, but a stall's catch-up is longer than
        the speaker holds: it was given out as a new sound's, with a new
        margin of silence in front, and stayed queued. Each stall made the
        sound 60 ms later for good."""
        win = self.win
        sound = win._playback_sound
        self.open_audio(22050, seconds=30.0)
        delays = []  # (time, seconds of sound in the speaker and held)

        def frames(seconds):
            end = self.clock.now + seconds
            while self.clock.now < end - 1e-9:
                start = self.clock.now
                win._update_display()
                waiting = self.speaker().queued + sound.waiting_s
                delays.append((self.clock.now, waiting, sound.holding))
                self.clock.now = max(self.clock.now, start + 0.033)

        def recent(seconds):
            return [d for d in delays if d[0] >= self.clock.now - seconds]

        frames(3.0)
        before = np.mean([d[1] for d in recent(2.0)])
        for _stall in range(5):
            self.clock.now += 0.3
            frames(2.0)
        frames(1.0)
        after = np.mean([d[1] for d in recent(2.0)])
        self.assertLess(after - before, 0.03)
        # (Its blocks play as they are again: none held between them.)
        self.assertLess(sum(d[2] for d in recent(2.0)), 10)
        start = self.clock.now
        frames(2.0)
        gaps, dropped = self.troubles_since(start)
        self.assertLess(gaps, 0.005)
        self.assertEqual(dropped, 0.0)
        win._playback.close()

    def test_an_audio_file_opened_while_the_last_sound_plays(self):
        """An 8 kHz audio file opened while the last recording's sound
        still plays, then another speed chosen, or back to 1x: no gap and
        nothing dropped. (The new file's first half-second block went to
        the speaker whole, on top of the last recording's end: AudioSink
        dropped that end, the sound was then fed as if the speaker still
        held it, and ran dry for about 0.1 s after the speed change.)"""
        win = self.win
        for speed in (2.0, 0.5):
            with self.subTest(speed=speed):
                self.clock.advance(1.0)  # (the last one's sound has played)
                self.open()  # (an FM recording: its sound plays)
                self.frames(1.5)
                self.open_audio(8000, seconds=20.0)
                opened = self.clock.now
                self.frames(3.4)
                win._playback_bar.set_speed(speed)
                self.frames(3.0)
                win._playback_bar.set_speed(1.0)
                self.frames(3.0)
                gaps, dropped = self.troubles_since(opened)
                self.assertLess(gaps, 0.005)
                self.assertEqual(dropped, 0.0)
                win._playback.close()

    def test_a_take_of_an_audio_recording_is_named_after_it(self):
        """Record Audio names a take after what was tuned. An audio
        recording's sound is listened to in USB from 1 Hz: its take was
        "0.000MHz_USB_<date>_<time>.wav"; now it is named after the
        recording. A take of an I/Q recording keeps its frequency."""
        import shutil
        import tempfile

        folder = tempfile.mkdtemp(prefix="audio-take-name-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        win = self.win
        self.open_audio(8000, seconds=2.0)
        self.assertTrue(win._start_audio_recording())
        self.assertIn("audio8000_USB_", self.message())
        self.frames(0.5)
        path = win._stop_audio_recording()
        self.assertRegex(os.path.basename(path), r"^audio8000_USB_\d{8}_\d{6}\.wav$")
        win._playback.close()
        self.open()  # (an FM recording at 100.0 MHz)
        self.assertTrue(win._start_audio_recording())
        self.frames(0.5)
        path = win._stop_audio_recording()
        self.assertRegex(os.path.basename(path), r"^\d+\.\d{3}MHz_FM_\d{8}_\d{6}\.wav$")
        win._playback.close()

    def test_record_audio_seek_pause_and_loop(self):
        """8 kHz audio at 0.5x with Record Audio on, through a seek, a
        pause, the file's end (Loop) and 2x: the take keeps the
        recording's own pace (what was played of it, at its own pitch),
        the speaker plays on after each restart, and nothing is given out
        between blocks once the sound has stopped."""
        import shutil
        import tempfile
        import wave

        folder = tempfile.mkdtemp(prefix="slow-take-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        win = self.win
        self.open_audio(8000, seconds=3.0)
        device = win._device

        def played():
            return (device.position + device.loops * device.total_samples) / 4000

        win._playback_bar.set_speed(0.5)
        self.frames(1.0)
        self.assertTrue(win._start_audio_recording())
        start = played()
        self.frames(2.0)
        jump = -played()
        win._playback.seek(2.5)  # (a little before the end: it loops)
        jump += played()
        self.assertFalse(win._playback_sound.stretching)
        self.assertEqual(len(win._playback_sound.more(self.clock.now)), 0)
        resumed = self.clock.now
        self.frames(3.0)
        self.assertGreaterEqual(device.loops, 1)
        gaps, _dropped = self.troubles_since(resumed + 0.3)
        self.assertLess(gaps, 0.01)
        win._toggle_acquisition()  # (Space: pause)
        self.assertFalse(win._playback_sound.stretching)
        self.frames(0.5)
        win._toggle_acquisition()
        win._playback_bar.set_speed(2.0)
        self.frames(2.0)
        # (What played of the recording: the seek skipped what lay between.)
        heard = played() - start - jump
        path = win._stop_audio_recording()
        with wave.open(path, "rb") as take:
            frames = take.getnframes()
            data = np.frombuffer(take.readframes(frames), dtype=np.int16)
        self.assertAlmostEqual(frames / AUDIO, heard, delta=2048 / 4000 + 0.1)
        self.assertAlmostEqual(_peak_hz(data[4800:52800]), 1000.0, delta=2.0)
        win._playback.close()


def _nfm(seconds, ctcss_hz, seed=1, rate=X_RATE, offset=X_OFFSET):
    """Narrow FM (5 kHz) tones with a CTCSS tone at 15 % of the deviation,
    ``offset`` Hz from the middle of an I/Q recording."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    audio = 0.15 * np.sin(2 * np.pi * ctcss_hz * t)
    for freq in (420.0, 830.0, 1270.0, 2010.0):
        audio += 0.15 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(5e3 * audio + offset) / rate
    x = 0.03 * np.exp(1j * phase)
    x += 1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x.astype(np.complex64)


class TestBookmarksOfBothKinds(_WavCase):
    """A reverse tone squelch (TSQL-R) on a frequency the receiver's own
    range doesn't cover (10368.030 MHz), made while a recording that does
    plays; and a bookmark below 1 kHz in a one-channel recording."""

    def _sink(self):
        from tests.test_gui_second_receiver import _FakeSink

        sink = _FakeSink()
        sink.open = True
        win = self.win
        win._audio, win._audio_enabled, win._audio_problem = sink, True, None
        win._audio_output_problem = lambda: None
        return sink

    def _loudest(self, sink, seconds):
        """The loudest sample the speaker got in each frame of ``seconds``."""
        from tests.test_gui_round3_main import _FakeClock

        win = self.win
        clock = _FakeClock()
        peaks = []
        with mock.patch.object(self.mw, "time", clock):
            win._pacer.restart()
            for _ in range(int(seconds / 0.03)):
                before = len(sink.blocks)
                clock.advance(0.03)
                win._update_display()
                peaks.append(
                    max(
                        (float(np.max(np.abs(b))) for b in sink.blocks[before:]),
                        default=0.0,
                    )
                )
        return peaks

    def _round_trip(self, bookmarks):
        """Export to CHIRP CSV and import again in their place."""
        from sdr_module.core.chirp_csv import export_bookmarks_csv

        panel = self.win._bookmarks_panel
        out = io.StringIO()
        export_bookmarks_csv(out, bookmarks)
        path = self.path("bookmarks.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(out.getvalue())
        panel._ask_replace_or_append = lambda count: "replace"
        self.assertEqual(panel.import_csv(path), len(bookmarks))
        return out.getvalue()

    def test_a_reverse_tone_squelch_above_6_ghz(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog, bookmark_squelch

        win = self.win
        panel = win._bookmarks_panel
        cp = win._control_panel
        station = X_BAND + X_OFFSET
        x = np.concatenate([_nfm(2.0, 100.0), _nfm(2.0, 123.0, seed=2)])
        path = self._wav("x_band.wav", x, "int", 3, "RIFF", X_BAND, rate=X_RATE)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.assertTrue(win.set_frequency(station), self.message())
        cp.set_tone_squelch_hz(100.0)
        cp.set_tone_reverse(True)
        # Ctrl+B keeps both: the frequency and the reverse tone squelch.
        win._bookmark_current_frequency()
        self.assertIn("(FM, reverse tone squelch 100.0 Hz)", self.message())
        made = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual(made["freq_hz"], station)
        self.assertEqual((made["tone_mode"], made["ctone"]), ("TSQL-R", 100.0))
        # The Add box shows it (not 6000 MHz), and Add saves it.
        self.assertAlmostEqual(panel._freq_input.value() * 1e6, station, delta=1.0)
        panel._confirm_add_another = lambda *args: True
        panel._label_input.setText("Added")
        panel._add_current()
        added = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual(added["freq_hz"], station)
        # The editor keeps both.
        dialog = BookmarkEditDialog(made, ("FM", "5 kHz"), [])
        self.addCleanup(dialog.deleteLater)
        self.assertTrue(dialog._reverse.isChecked())
        edited = dialog.bookmark()
        self.assertEqual(edited["freq_hz"], station)
        self.assertEqual(edited["tone_mode"], "TSQL-R")
        # CHIRP CSV and back.
        text = self._round_trip([edited, added])
        self.assertIn("10368.030000", text)
        self.assertIn("TSQL-R", text)
        back = panel.bookmark(0)
        self.assertEqual(back["freq_hz"], station)
        self.assertEqual(bookmark_squelch(back), (100.0, None, True))
        # Tuned while the recording plays: muted while its tone is sent,
        # the station with another tone heard.
        win._set_demod_mode("AM")
        cp.set_tone_reverse(False)
        cp.set_tone_squelch_hz(0.0)
        panel._tune_index(0)
        self.assertIn("(FM, reverse tone squelch 100.0 Hz)", self.message())
        self.assertEqual(win._current_frequency(), station)
        self.assertEqual((cp.tone_squelch_hz(), cp.tone_reverse()), (100.0, True))
        sink = self._sink()
        cp.set_squelch_db(-80)
        peaks = self._loudest(sink, 3.9)
        self.assertLess(max(peaks[5:60]), 1e-3)  # (its tone: muted)
        self.assertGreater(max(peaks[80:]), 0.1)  # (another tone: heard)
        # Without the recording the receiver can't tune it: said so.
        win._playback.close()
        before = win._current_frequency()
        panel._tune_index(0)
        self.assertEqual(win._current_frequency(), before)
        self.assertIn("10368.030 MHz is out of range", self.message())

    def test_a_bookmark_below_1_khz_goes_to_chirp_and_back(self):
        win = self.win
        panel = win._bookmarks_panel
        path = _tone_wav(self.path("audio.wav"), AUDIO_RATE, 2.0, freq=700.0)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        win._set_demod_mode("CW", None, "500 Hz")
        self.assertTrue(win.set_frequency(700.0), self.message())
        win._bookmark_current_frequency()
        made = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual((made["freq_hz"], made["demod"]), (700.0, "CW"))
        text = self._round_trip([made])
        self.assertIn("0.000700", text)
        self.assertEqual(panel.bookmark(0)["freq_hz"], 700.0)
        win._set_demod_mode("USB", None, "2.7 kHz")
        self.assertTrue(win.set_frequency(5000.0))
        panel._tune_index(0)
        self.assertEqual(win._current_frequency(), 700.0)
        self.assertEqual(win._control_panel.get_demod_mode(), "CW")
        win._playback.close()
        panel._tune_index(0)
        self.assertIn("0.7 kHz is out of range", self.message())
