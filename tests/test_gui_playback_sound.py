"""Sound at every playback speed (M-P1), in the window.

A recording played at 0.5x, 2x or 4x used to play no sound at all: the
Audio button read Paused ("Playing at 2x: audio plays at 1x only") while
the plots and decoders ran. Now the speaker plays it time-stretched to the
pace the recording really plays at (``PlaybackSound``), its pitch kept,
also when the computer can't keep up (it plays more slowly then, and the
sound follows); Record Audio records the recording's own sound at its own
pace (10 s of the recording make 10 s of WAV, whatever the speed).

The display loop runs on a fake clock, a frame every 33 ms; what the
speaker gets is captured from ``AudioSink.write``.
"""

import os
import shutil
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_playback import RATE, _fm_tone, _PlaybackCase  # noqa: E402
from tests.test_gui_round3_main import _FakeClock  # noqa: E402
from tests.test_gui_squelch import pretend_audio_output  # noqa: E402
from tests.test_playback_sound import _Speaker  # noqa: E402

AUDIO = 48000.0


def _peak_hz(audio, rate=AUDIO):
    """The strongest frequency of ``audio`` (interpolated)."""
    x = np.asarray(audio, dtype=float)
    pad = 8 * len(x)
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)), pad))
    k = int(np.argmax(s[1:])) + 1
    a, b, c = np.log(s[k - 1 : k + 2] + 1e-30)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * rate / pad


def _level(audio, freq, rate=AUDIO):
    """Amplitude of the ``freq`` component (least squares)."""
    x = np.asarray(audio, dtype=float)
    t = np.arange(len(x)) / rate
    return float(
        np.hypot(
            2 * np.mean(x * np.cos(2 * np.pi * freq * t)),
            2 * np.mean(x * np.sin(2 * np.pi * freq * t)),
        )
    )


#: A recording's last part is marked: a 600 Hz tone after a 1 kHz one.
MARK_HZ, TONE_HZ = 600.0, 1000.0


def _marked(seconds, mark_s, rate):
    """The tone a marked recording plays, ``seconds`` long at ``rate``: 1
    kHz, then ``MARK_HZ`` for its last ``mark_s`` (no step between)."""
    t = np.arange(int(round(seconds * rate))) / rate
    freq = np.where(t < seconds - mark_s, TONE_HZ, MARK_HZ)
    return np.sin(2 * np.pi * np.cumsum(freq) / rate)


def _marked_fm(seconds, mark_s, deviation=5e3, rate=RATE):
    """An FM recording of :func:`_marked`'s tone."""
    tone = _marked(seconds, mark_s, rate)
    x = 0.5 * np.exp(1j * 2 * np.pi * deviation * np.cumsum(tone) / rate)
    return x.astype(np.complex64)


def _marker_seconds(audio, rate=AUDIO):
    """How long ``audio`` plays the mark of :func:`_marked` (the end of the
    recording): from the first 10 ms in which it is louder than the tone
    before it to the last in which it is at least half as loud as usual.
    (10 ms holds whole cycles of both: neither leaks into the other.)"""
    hop, size = int(0.0025 * rate), int(0.01 * rate)
    marks, tones = [], []
    for i in range(0, len(audio) - size + 1, hop):
        piece = audio[i : i + size]
        marks.append(_level(piece, MARK_HZ, rate))
        tones.append(_level(piece, TONE_HZ, rate))
    marks, tones = np.array(marks), np.array(tones)
    louder = marks > tones
    if not louder.any():
        return 0.0
    usual = float(np.median(marks[louder]))
    on = np.nonzero(louder & (marks >= 0.5 * usual))[0]
    return (on[-1] - on[0]) * hop / rate + 0.5 * size / rate


class _SoundCase(_PlaybackCase):
    """A recording on a fake clock, a frame every 33 ms (or as soon as the
    last one ended), the speaker's audio captured with when it came."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.heard = []  # (fake time, block)
        self.win._audio.write = lambda block: self.heard.append(
            (self.clock.now, np.array(block, dtype=np.float32))
        )
        self.clock = _FakeClock()
        patcher = mock.patch.object(self.mw, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cost = 0.0  # processing seconds per second of signal at 1x

    def open(self, samples=None, rate=RATE, name="fm.sigmf-data", center=None):
        kwargs = {"rate": rate}
        if center is not None:
            kwargs["center"] = center
        self.play(self.make_file(samples, name=name, **kwargs))
        win = self.win
        self.assertTrue(win._playback.active)
        win._playback_bar.set_loop(True)
        win._control_panel.set_squelch_enabled(False)
        win._control_panel.set_audio_agc_speed("off")  # (levels to compare)
        win._rx_history.clear()  # (taken on the real clock)
        win._pacer.restart()
        process = win._process_block

        def costly(samples):
            # A computer that takes ``cost`` times the signal's time.
            self.clock.advance(self.cost * len(samples) / win._device_sample_rate())
            process(samples)

        win._process_block = costly

    def frames(self, seconds):
        win = self.win
        end = self.clock.now + seconds
        while self.clock.now < end - 1e-9:
            start = self.clock.now
            win._update_display()
            self.clock.now = max(self.clock.now, start + 0.033)

    def played(self):
        """Seconds of the recording played so far (Loop counted)."""
        device = self.win._device
        return (device.position + device.loops * device.total_samples) / RATE

    def sound(self, since=None):
        """What the speaker got (its first channel), from ``since`` on."""
        blocks = [b for t, b in self.heard if since is None or t >= since]
        blocks = [b if b.ndim == 1 else b[:, 0] for b in blocks if len(b)]
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def speaker(self, since=None):
        """The speaker as AudioSink plays what it got (gaps, drops)."""
        model = _Speaker()
        for when, block in self.heard:
            if since is None or when >= since:
                model.write(block, when)
        model.advance(self.clock.now)
        return model

    def troubles_since(self, start):
        """Seconds the speaker ran dry, and of audio dropped, after
        ``start``: the speaker followed from the first block it got (one
        started at ``start`` would start empty, and run dry where the real
        one still held what it got before)."""
        model = _Speaker()
        base = None
        for when, block in self.heard:
            if base is None and when >= start:
                model.advance(start)
                base = (model.gaps, model.dropped)
            model.write(block, when)
        model.advance(self.clock.now)
        if base is None:
            base = (model.gaps, model.dropped)
        return model.gaps - base[0], model.dropped - base[1]


class TestEverySpeedPlays(_SoundCase):
    def test_0_5x_2x_and_4x_play_at_the_recordings_pitch_in_time(self):
        self.open()
        win = self.win
        for speed in (0.5, 2.0, 4.0):
            with self.subTest(speed=speed):
                win._playback_bar.set_speed(speed)
                self.frames(1.0)  # (it settles in)
                start = self.clock.now
                before = self.played()
                self.frames(3.0)
                # The recording plays at the speed chosen...
                rate = (self.played() - before) / (self.clock.now - start)
                self.assertAlmostEqual(rate, speed, delta=0.05 * speed)
                # ...and the speaker plays all the while, without a gap:
                # as much sound as time went by, at the tone's own pitch.
                speaker = self.speaker(since=start)
                self.assertLess(speaker.gaps, 0.01)
                self.assertEqual(speaker.dropped, 0.0)
                audio = self.sound(since=start)
                self.assertAlmostEqual(len(audio) / AUDIO, 3.0, delta=0.15)
                self.assertAlmostEqual(_peak_hz(audio), 1000.0, delta=1.0)
                # The Audio button says Audio, not Paused.
                self.assertTrue(win._realtime())
                self.assertFalse(win._audio_held())
                self.assertEqual(win._audio_state(), "on")
                win._sync_audio_ui()
                self.assertEqual(win._audio_button.text(), win._AUDIO_ON_TEXT)

    def test_1x_plays_exactly_as_before(self):
        """At 1x on a computer that keeps up, the speaker gets the same
        blocks as without the stretch, bit for bit."""
        self.open()
        win = self.win

        def run():
            win._playback.seek(0.0)
            self.heard.clear()
            self.frames(2.0)
            return [b for _t, b in self.heard]

        with_it = run()
        self.assertFalse(win._playback_sound.stretching)
        with mock.patch.object(
            win._playback_sound,
            "play",
            side_effect=lambda block, now, last=False: block,
        ):
            without = run()
        self.assertEqual(len(with_it), len(without))
        for a, b in zip(with_it, without, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_the_notch_still_works_at_2x(self):
        self.open()
        win = self.win
        panel = win._control_panel
        win._playback_bar.set_speed(2.0)
        self.frames(1.0)
        start = self.clock.now
        self.frames(1.0)
        loud = _level(self.sound(since=start), 1000.0)
        self.assertGreater(loud, 0.05)
        # The notch takes the 1 kHz tone out of what plays at 2x too.
        panel.set_notch_mode("manual")
        panel.set_notch_hz(1000)
        self.frames(1.0)
        start = self.clock.now
        self.frames(1.0)
        notched = _level(self.sound(since=start), 1000.0)
        self.assertLess(20 * np.log10(notched / loud), -30.0)

    def test_the_squelch_closing_ends_the_sound_without_a_click(self):
        """A tone, then the carrier gone: the squelch closes and the sound
        ends, at 2x as at 1x, with nothing held back and no step the 1x
        sound (which never goes through the stretch) doesn't have too."""
        quiet = np.zeros(int(0.6 * RATE), np.complex64)
        self.open(np.concatenate([_fm_tone(1.0, deviation=2.5e3), quiet]))
        win = self.win
        win._playback_bar.set_loop(False)
        panel = win._control_panel
        panel.set_squelch_auto(False)
        panel.set_squelch_enabled(True)
        panel.set_squelch_db(-40.0)
        sharpest = {}
        for speed in (1.0, 2.0):
            win._playback.seek(0.0)
            win._playback_bar.set_speed(speed)
            self.heard.clear()
            self.frames(1.5 / speed)  # (the tone, then the quiet half second)
            self.assertFalse(win._squelch.is_open)
            sound = win._playback_sound
            self.assertFalse(sound.stretching)
            self.assertEqual(sound.waiting_s, 0.0)
            tail = [b for _t, b in self.heard if len(b)][-1]
            self.assertLess(float(np.max(np.abs(tail[-48:]))), 1e-3)
            audio = self.sound().astype(float)
            sharpest[speed] = float(np.max(np.abs(np.diff(audio, 2))))
        self.assertLessEqual(sharpest[2.0], 1.1 * sharpest[1.0])


class TestARecordingTheComputerCantKeepUpWith(_SoundCase):
    """A 1 MS/s recording whose processing costs more than real time."""

    def open_fast(self):
        x = _fm_tone(1.0, rate=1e6)
        self.open(x, rate=1e6, name="fast.sigmf-data")

    def played(self):
        device = self.win._device
        return (device.position + device.loops * device.total_samples) / 1e6

    def test_4x_too_much_plays_slower_and_the_sound_follows(self):
        self.open_fast()
        win = self.win
        win._playback_bar.set_speed(4.0)
        self.cost = 0.3  # (4x needs 1.2x real time)
        self.frames(3.0)
        start, before = self.clock.now, self.played()
        self.frames(4.0)
        pace = (self.played() - before) / (self.clock.now - start)
        self.assertTrue(win._pacer.overloaded)
        self.assertGreater(pace, 0.9)
        self.assertLess(pace, 2.0)
        speaker = self.speaker(since=start)
        self.assertLess(speaker.gaps, 0.02)
        self.assertEqual(speaker.dropped, 0.0)
        audio = self.sound(since=start)
        self.assertAlmostEqual(len(audio) / AUDIO, 4.0, delta=0.25)
        self.assertAlmostEqual(_peak_hz(audio), 1000.0, delta=1.0)
        self.assertEqual(win._audio_state(), "on")

    def test_1x_too_much_plays_slower_without_gaps(self):
        # It used to run dry again and again (a 0.12 s gap each time).
        self.open_fast()
        win = self.win
        self.cost = 1.25
        self.frames(4.0)
        start = self.clock.now
        self.frames(4.0)
        self.assertTrue(win._playback_sound.stretching)
        speaker = self.speaker(since=start)
        self.assertLess(speaker.gaps, 0.02)
        audio = self.sound(since=start)
        self.assertAlmostEqual(_peak_hz(audio), 1000.0, delta=1.0)

    def test_4x_with_frames_that_vary_plays_without_gaps(self):
        """Each frame costs 0.6 s per second of the recording at 1x (4x
        needs 2.4 times real time), give or take 10 %: the sound plays
        through. (Kept at just the longest recent wait, the speaker ran
        dry whenever a probe frame took a little longer than the longest
        so far: 0.04 s of gaps in these 30 s.)"""
        self.open_fast()
        win = self.win
        win._playback_bar.set_speed(4.0)
        rng = np.random.default_rng(2)
        process = win._process_block  # (costly: self.cost a second of signal)

        def varying(samples):
            share = self.cost * len(samples) / win._device_sample_rate()
            self.clock.advance(0.1 * share * float(rng.uniform(-1.0, 1.0)))
            process(samples)

        win._process_block = varying
        self.cost = 0.6
        self.frames(3.0)
        start = self.clock.now
        self.frames(30.0)
        self.assertTrue(win._pacer.overloaded)
        self.assertEqual(self.troubles_since(start), (0.0, 0.0))
        self.assertEqual(win._audio_state(), "on")

    def test_far_too_slow_pauses_and_says_why(self):
        # Below half of real time the sound pauses, as at 1x: the hint
        # names this computer, never "1x only".
        self.open_fast()
        win = self.win
        win._playback_bar.set_speed(4.0)
        self.cost = 3.0
        self.frames(4.0)
        self.assertFalse(win._realtime())
        self.assertEqual(win._audio_state(), "paused")
        hint = win._realtime_hint()
        self.assertNotIn("1x only", hint)
        self.assertIn("can't process the recording's 1 MS/s", hint)


class TestStereoAt2x(_SoundCase):
    def test_left_and_right_and_the_stereo_badge(self):
        from sdr_module.dsp.fm_stereo import StereoDecoder
        from tests.test_fm_stereo import _fm, _multiplex

        rate = 480e3
        t = np.arange(int(1.5 * rate)) / rate
        left = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
        right = 0.5 * np.sin(2 * np.pi * 400.0 * t)
        iq = _fm(_multiplex(left, right, rate), rate)
        patcher = mock.patch.object(StereoDecoder, "RISE_S", 0.05)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.open(iq, rate=rate, name="fm.sigmf-data", center=100.1e6)
        win = self.win
        self.assertEqual(win._control_panel.get_fm_deviation(), 75e3)
        win._playback_bar.set_speed(2.0)
        self.frames(1.5)
        start = self.clock.now
        self.frames(1.5)
        self.assertEqual(win._stereo_badge.text(), "STEREO")
        self.assertTrue(win._stereo_badge_action.isVisible())
        blocks = [b for t_, b in self.heard if t_ >= start and len(b)]
        self.assertTrue(all(b.ndim == 2 for b in blocks))
        stereo = np.concatenate(blocks)
        self.assertAlmostEqual(_peak_hz(stereo[:, 0]), 1000.0, delta=1.0)
        self.assertAlmostEqual(_peak_hz(stereo[:, 1]), 400.0, delta=1.0)


class TestRecordAudioDuringPlayback(_SoundCase):
    def setUp(self):
        super().setUp()
        self.folder_rec = tempfile.mkdtemp(prefix="playback-take-")
        self.addCleanup(shutil.rmtree, self.folder_rec, True)
        self.store.values["recording/folder"] = self.folder_rec

    def test_a_take_keeps_the_recordings_own_pace(self):
        """10 s of the recording make 10 s of WAV at any speed (what was
        received, at its own pitch and pace; the speaker's stretch, which
        follows how fast this computer plays it, isn't recorded)."""
        self.open()
        win = self.win
        for speed in (2.0, 0.5):
            with self.subTest(speed=speed):
                win._playback_bar.set_speed(speed)
                self.frames(0.5)
                self.assertTrue(win._start_audio_recording())
                self.assertIn("own pace", win._message_label.text())
                before = self.played()
                self.frames(2.0)
                win._update_audio_recording_status()
                self.assertIn("own pace", win._audio_rec_label.toolTip())
                played = self.played() - before
                path = win._stop_audio_recording()
                with wave.open(path, "rb") as take:
                    frames = take.getnframes()
                    data = np.frombuffer(take.readframes(frames), dtype=np.int16)
                self.assertAlmostEqual(frames / AUDIO, played, delta=0.05)
                self.assertAlmostEqual(_peak_hz(data[2400:]), 1000.0, delta=1.0)

    def test_at_1x_the_take_says_nothing_of_a_pace(self):
        self.open()
        self.assertTrue(self.win._start_audio_recording())
        self.assertNotIn("own pace", self.win._message_label.text())
        self.frames(0.5)
        self.win._update_audio_recording_status()
        self.assertNotIn("own pace", self.win._audio_rec_label.toolTip())
        self.win._stop_audio_recording()


class TestWhatStopsTheSound(_SoundCase):
    def test_listen_in_the_second_receiver_mutes_it(self):
        self.open()
        win = self.win
        win._playback_bar.set_speed(2.0)
        self.frames(1.0)
        with mock.patch.object(win._second_rx, "mutes_main", return_value=True):
            self.frames(0.1)  # (what played fades out)
            self.heard.clear()
            self.frames(0.5)
            self.assertEqual([b for _t, b in self.heard if len(b)], [])
            self.assertFalse(win._playback_sound.stretching)
            self.assertEqual(win._audio_state(), "second")
        start = self.clock.now
        self.frames(1.0)
        self.assertGreater(len(self.sound(since=start)) / AUDIO, 0.8)

    def test_audio_off_and_on_again_plays_on(self):
        """Audio off and straight on again opens the speaker anew, empty,
        while the stretch still thinks of what the old one held (the
        sound's end): the new output's cushion covers it, no gaps."""
        self.open()
        win = self.win
        win._playback_bar.set_speed(2.0)
        self.frames(1.5)
        win._set_audio_enabled(False)
        self.frames(0.07)  # (what played fades out)
        win._set_audio_enabled(True)
        start = self.clock.now
        self.frames(2.0)
        speaker = self.speaker(since=start)  # (the new output, empty)
        self.assertEqual(speaker.gaps, 0.0)
        self.assertEqual(speaker.dropped, 0.0)
        self.assertGreater(len(self.sound(since=start)) / AUDIO, 1.7)

    def test_a_squelch_that_closes_briefly_leaves_no_gaps(self):
        """Tone bursts 0.45 s apart: the squelch closes (after its hang)
        and opens again a frame or two later at 2x and 4x, while the
        speaker still plays the end of the last burst. No burst's sound
        runs the speaker dry mid-way (it did: 0.006 s of gaps in these 6 s
        at 2x, 0.74 s at 4x)."""
        burst = _fm_tone(0.6, deviation=2.5e3)
        quiet = np.zeros(int(0.45 * RATE), np.complex64)
        self.open(np.concatenate([burst, quiet] * 3))
        win = self.win
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
        for speed in (2.0, 4.0):
            with self.subTest(speed=speed):
                win._playback_bar.set_speed(speed)
                self.frames(1.0)
                start = self.clock.now
                self.frames(6.0)
                closes = sum(
                    1
                    for i in range(1, len(opened))
                    if opened[i - 1][1] and not opened[i][1] and opened[i][0] >= start
                )
                self.assertGreaterEqual(closes, 4)
                # Within each burst's sound (from its first block on to
                # the squelch closing), the speaker never runs dry.
                model = _Speaker()
                model.listening = False
                state = dict(opened)
                for when, block in self.heard:
                    if when < start:
                        continue
                    model.write(block, when)
                    if len(block):
                        model.listening = state.get(when, False)
                self.assertEqual(model.gaps, 0.0)
                self.assertEqual(model.dropped, 0.0)

    def test_a_seek_or_pause_ends_what_was_held(self):
        self.open()
        win = self.win
        win._playback_bar.set_speed(4.0)
        self.frames(1.0)
        self.assertTrue(win._playback_sound.stretching)
        win._playback.seek(0.2)
        self.assertFalse(win._playback_sound.stretching)
        self.frames(0.5)
        self.assertTrue(win._playback_sound.stretching)
        win._toggle_acquisition()  # (Space: pause)
        self.assertFalse(win._is_running)
        self.assertFalse(win._playback_sound.stretching)
        self.assertEqual(win._playback_sound.waiting_s, 0.0)


class TestTheEndOfARecording(_SoundCase):
    """Without Loop a recording's sound plays to its end at the speed
    chosen, and then the playback pauses (R8-06)."""

    def play_to_the_end(self, samples, speed, phase):
        """Play ``samples`` without Loop at ``speed``, the playback's own
        timer ticked every 200 ms from ``phase`` s before the start, until
        a second after it paused. Returns when the end was read and when
        it paused."""
        win = self.win
        self.open(samples)
        win._playback_bar.set_loop(False)
        win._playback_bar.set_speed(speed)
        win._rx_history.clear()
        win._pacer.restart()
        self.heard.clear()
        device = win._device
        read = paused = None
        began, tick = self.clock.now, self.clock.now - phase
        while paused is None or self.clock.now < paused + 1.0:
            start = self.clock.now
            win._update_display()
            if read is None and device.finished:
                read = self.clock.now
            if self.clock.now - tick >= win._playback._TICK_MS / 1e3 - 1e-9:
                tick = self.clock.now
                win._playback._tick()
                if paused is None and not win._is_running:
                    paused = self.clock.now
            self.clock.now = max(self.clock.now, start + 0.033)
            self.assertLess(self.clock.now - began, 60.0)
        return read, paused

    def test_its_end_plays_at_the_speed_chosen_then_it_pauses(self):
        """What the speaker was still to get went out at once when the
        playback paused, at its first tick after the end was read (up to
        0.2 s later): often after the speaker had run dry (30-100 ms), a
        burst from full level; at 0.5x, 2x and 4x the input the
        time-stretch held, rushed."""
        mark_s = 0.3
        for speed in (0.5, 1.0, 2.0, 4.0):
            for phase in (0.0, 0.05, 0.1, 0.15):
                with self.subTest(speed=speed, phase=phase):
                    self.clock.advance(1.0)  # (the last one's sound has played)
                    read, paused = self.play_to_the_end(
                        _marked_fm(1.6, mark_s), speed, phase
                    )
                    win = self.win
                    self.assertIsNotNone(read)
                    self.assertFalse(win._is_running)
                    self.assertEqual(win._state_badge.text(), "ENDED")
                    self.assertIn("End of fm.sigmf-data", self.message())
                    # Its last part as long as the speed makes it...
                    self.assertAlmostEqual(
                        _marker_seconds(self.sound()),
                        mark_s / speed,
                        delta=0.1 * mark_s / speed + 0.02,
                    )
                    # ...with no gap before its end, which fades out...
                    writes = [(t, b) for t, b in self.heard if len(b)]
                    last = writes[-1][0]
                    model = _Speaker()
                    for when, block in writes:
                        model.write(block, when)
                    self.assertLess(model.gaps, 0.005)
                    self.assertEqual(model.dropped, 0.0)
                    peak = float(np.max(np.abs(self.sound())))
                    self.assertLess(abs(float(writes[-1][1][-1])), 0.01 * peak)
                    # ...and nothing after it: then the playback paused.
                    self.assertLessEqual(last, paused)
                    win._playback.close()


class TestWhatItSays(_SoundCase):
    def test_the_speed_note_and_the_speed_tooltip(self):
        self.open()
        win = self.win
        bar = win._playback_bar
        bar.set_speed(2.0)
        self.assertEqual(self.message(), "Playing at 2x: the sound keeps its pitch.")
        bar.set_speed(0.5)
        self.assertEqual(self.message(), "Playing at 0.5x: the sound keeps its pitch.")
        bar.set_speed(4.0)
        self.assertEqual(
            self.message(),
            "Playing at 4x: the sound keeps its pitch, but speech is hard to "
            "follow this fast (2x or slower to listen).",
        )
        tip = bar._speed.toolTip()
        self.assertNotIn("1x only", tip)
        self.assertIn("keeps its pitch", tip)
        self.assertIn("4x", tip)
        self.assertNotIn("1x only", win._realtime_hint())
        bar.set_speed(1.0)
        self.assertEqual(self.message(), "Playing at 1x.")


class TestDecodersAt2x(_SoundCase):
    def test_aprs_decodes_at_2x_with_the_sound_on(self):
        from sdr_module.dsp import protocol_encoders as enc

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>Played back", RATE, ["WIDE1-1"]
        )
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, burst.astype(np.complex64), pad])
        self.open(x, name="aprs.sigmf-data", center=144.39e6)
        win = self.win
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        panel._enabled_check.setChecked(True)
        win._playback_bar.set_loop(False)
        win._playback_bar.set_speed(2.0)
        self.frames(1.5)
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(any("Played back" in t for t in texts), texts)
        self.assertGreater(len(self.sound()), 0)


if __name__ == "__main__" and HAS_PYQT6:
    unittest.main()
