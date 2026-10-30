#!/usr/bin/env python3
"""
AudioSink with a fake QAudioSink (rx:UX-15): partial writes, the jitter
buffer, deliberate drops and pads when the clocks drift, debug logging.

There is no sound card here, so the fake output plays a given number of
bytes per "tick" like a real one would, and only accepts what fits in its
buffer (``bytesFree``), returning short counts like QIODevice.write.
"""

import logging
import unittest
from unittest import mock

import numpy as np

from sdr_module.gui import audio_sink as audio_sink_module
from sdr_module.gui.audio_sink import AudioSink

RATE = 48000


class FakeOutput:
    """QAudioSink stand-in: a buffer of ``size`` bytes."""

    def __init__(self, size):
        self.size = size
        self.queued = 0
        self.volume = None

    def bufferSize(self):  # noqa: N802 (Qt API)
        return self.size

    def bytesFree(self):  # noqa: N802 (Qt API)
        return self.size - self.queued

    def setVolume(self, value):  # noqa: N802 (Qt API)
        self.volume = value

    def stop(self):
        pass

    def play(self, count):
        """The sound card takes ``count`` bytes; returns how many it had."""
        taken = min(count, self.queued)
        self.queued -= taken
        return taken


class FakeDevice:
    """The QIODevice QAudioSink.start() returns: accepts what fits."""

    def __init__(self, output, limit=None):
        self.output = output
        self.limit = limit  # accept at most this much per call
        self.received = bytearray()
        self.calls = []

    def write(self, data):
        self.calls.append(len(data))
        count = min(len(data), self.output.bytesFree())
        if self.limit is not None:
            count = min(count, self.limit)
        self.output.queued += count
        self.received += data[:count]
        return count


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _SinkCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        patcher = mock.patch.object(audio_sink_module.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, buffer_s=0.25, limit=None):
        sink = AudioSink()
        output = FakeOutput(int(buffer_s * RATE) * 2)
        device = FakeDevice(output, limit)
        sink._sink, sink._io, sink._sample_rate = output, device, RATE
        return sink, output, device

    @staticmethod
    def frame(seconds=1600 / RATE, value=0.25):
        return np.full(int(round(seconds * RATE)), value, dtype=np.float32)

    def stream(self, sink, output, frames, play_rate=1.0, frame_s=1600 / RATE):
        """Write ``frames`` frames of ``frame_s``, the sound card playing
        ``play_rate`` times the frame's worth between them."""
        played = 0
        for _ in range(frames):
            sink.write(self.frame(frame_s))
            self.clock.now += frame_s
            played += output.play(int(frame_s * RATE * play_rate) * 2)
        return played


class TestPartialWrites(_SinkCase):
    def test_what_does_not_fit_is_kept_for_the_next_write(self):
        sink, output, device = self.make(limit=1000)  # short writes
        samples = np.linspace(-0.5, 0.5, 4000).astype(np.float32)
        sink.write(samples)
        # Only 1000 bytes went out; the rest waits instead of being lost.
        self.assertEqual(len(device.received), 1000)
        self.assertGreater(len(sink._backlog), 0)
        cushion = len(sink._backlog) + 1000 - 8000  # the start-up silence
        for _ in range(40):
            self.clock.now += 0.01
            output.play(900)  # (never runs dry: no underrun padding)
            sink._pump(True)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
        # Every sample arrived once, in order, after the cushion.
        self.assertEqual(bytes(device.received), bytes(cushion) + pcm)
        self.assertEqual(len(sink._backlog), 0)
        self.assertEqual(sink.underruns, 0)

    def test_never_writes_more_than_bytes_free(self):
        sink, output, device = self.make(buffer_s=0.05)
        for _ in range(10):
            free_before = output.bytesFree()
            sink.write(self.frame(0.04))
            self.assertLessEqual(device.calls[-1], free_before)
            self.clock.now += 0.04
            output.play(int(0.04 * RATE) * 2)

    def test_a_refused_write_keeps_the_samples(self):
        sink, output, device = self.make()
        device.write = lambda data: -1
        sink.write(self.frame())
        self.assertGreaterEqual(len(sink._backlog), 3200)


class TestJitterBuffer(_SinkCase):
    def test_starts_with_a_cushion_and_rides_out_bursty_frames(self):
        sink, output, device = self.make()
        sink.write(self.frame())
        cushion = sink.queued_seconds()
        self.assertAlmostEqual(cushion, AudioSink.TARGET_S, delta=0.002)
        self.assertEqual(sink.underruns, 0)  # a stream start is not one
        # Frames arrive two at once every other tick (a slow display
        # frame): the cushion absorbs it, no underrun, no drop.
        for i in range(60):
            if i % 2:
                sink.write(self.frame())
                sink.write(self.frame())
            self.clock.now += 1600 / RATE
            output.play(3200)
        self.assertEqual(sink.underruns, 0)
        self.assertEqual(sink.overruns, 0)
        self.assertGreater(output.queued, 0)

    def test_a_fast_sound_card_pads_with_silence_and_logs(self):
        sink, output, _device = self.make()
        with self.assertLogs(audio_sink_module.logger, logging.DEBUG) as logs:
            self.stream(sink, output, 120, play_rate=1.2)
        self.assertGreater(sink.underruns, 0)
        self.assertTrue(any("underrun" in line for line in logs.output))
        self.assertTrue(all("DEBUG" in line for line in logs.output))

    def test_a_slow_sound_card_drops_old_audio_and_stays_bounded(self):
        sink, output, _device = self.make()
        with self.assertLogs(audio_sink_module.logger, logging.DEBUG) as logs:
            self.stream(sink, output, 400, play_rate=0.8)
        self.assertGreater(sink.overruns, 0)
        self.assertLessEqual(sink.queued_seconds(), AudioSink.HIGH_S + 0.04)
        self.assertTrue(any("overrun" in line for line in logs.output))

    def test_matched_clocks_neither_drop_nor_pad(self):
        sink, output, _device = self.make()
        self.stream(sink, output, 300, play_rate=1.0)
        self.assertEqual(sink.underruns, 0)
        self.assertEqual(sink.overruns, 0)
        self.assertAlmostEqual(sink.queued_seconds(), AudioSink.TARGET_S, delta=0.04)

    def test_resuming_after_silence_is_not_an_underrun(self):
        sink, output, _device = self.make()
        self.stream(sink, output, 10)
        self.clock.now += 2.0  # squelch closed for two seconds
        output.play(output.queued)
        sink.write(self.frame())
        self.assertEqual(sink.underruns, 0)
        self.assertAlmostEqual(sink.queued_seconds(), AudioSink.TARGET_S, delta=0.002)

    def test_stop_and_mute_discard_the_backlog(self):
        sink, output, device = self.make(limit=100)
        sink.write(self.frame())
        self.assertTrue(sink._backlog)
        sink.stop()
        self.assertFalse(sink._backlog)
        self.assertFalse(sink.is_open)
        sink.write(self.frame())  # closed: nothing happens
        self.assertEqual(sink.queued_seconds(), 0.0)
        sink, output, device = self.make(limit=100)
        sink.write(self.frame())
        sink.set_muted(True)
        self.assertFalse(sink._backlog)


if __name__ == "__main__":
    unittest.main()
