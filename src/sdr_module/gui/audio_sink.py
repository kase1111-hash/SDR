"""
Audio output sink.

Thin wrapper around QAudioSink (PyQt6.QtMultimedia). Accepts a mono
int16 stream and plays it on the default output device. Absent a working
Qt multimedia backend, falls back to a no-op so the rest of the GUI
still works.

``start()`` reports whether the output really opened: it returns False when
there is no output device or Qt could not open it, so callers can tell the
user instead of playing into nothing.

The samples arrive a display frame at a time (about every 33 ms, sometimes
two at once after a slow frame) while the sound card takes them at its own
pace, from its own clock. ``write`` therefore keeps a small jitter buffer:
it only hands the output what ``bytesFree()`` says fits and keeps the rest
for the next call, starts (and restarts after running dry) with a cushion
of silence, and drops the oldest audio when too much has piled up (the two
clocks drift apart, or the window stalled). Underruns and drops are logged
at debug level.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QIODevice
    from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

    HAS_QT_AUDIO = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_QT_AUDIO = False


class AudioSink:
    """Simple int16 mono audio sink.

    Call ``start(sample_rate)`` once, then push numpy float arrays with
    ``write(samples)``. Call ``stop()`` when done.
    """

    #: Audio kept queued (in the output plus not yet accepted by it): the
    #: jitter buffer. Running dry pads it back to TARGET_S with silence;
    #: more than HIGH_S drops the oldest audio back to TARGET_S.
    TARGET_S = 0.12
    HIGH_S = 0.35
    #: Size of the output's own buffer, asked for when it opens.
    BUFFER_S = 0.25
    #: A write within this long of the previous one continues the stream
    #: (running dry then is an underrun, not just the start of a stream).
    CONTINUOUS_S = 0.25

    def __init__(self) -> None:
        self._sink = None
        self._io: Optional["QIODevice"] = None
        self._sample_rate = 0
        self._muted = False
        self._volume = 0.7
        self._backlog = bytearray()  # PCM the output hasn't accepted yet
        self._last_write: Optional[float] = None
        self.underruns = 0  # the output ran dry mid-stream (padded)
        self.overruns = 0  # too much queued (oldest audio dropped)

    @property
    def available(self) -> bool:
        return HAS_QT_AUDIO

    @property
    def is_open(self) -> bool:
        """True while an output stream is open (``write`` plays sound)."""
        return self._io is not None

    def start(self, sample_rate: int = 48000) -> bool:
        """Open the default output at ``sample_rate`` Hz.

        Returns True when the output is open, False when nothing can play:
        no QtMultimedia, no output device, or Qt could not open it.
        """
        if not HAS_QT_AUDIO:
            logger.info("QtMultimedia not available; audio output disabled")
            return False
        if self.is_open and self._sample_rate == int(sample_rate):
            return True
        self.stop()
        try:
            device = QMediaDevices.defaultAudioOutput()
            if device.isNull():
                logger.info("No audio output device; audio output disabled")
                return False
            fmt = QAudioFormat()
            fmt.setSampleRate(int(sample_rate))
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

            sink = QAudioSink(device, fmt)
            sink.setVolume(self._volume)
            sink.setBufferSize(int(self.BUFFER_S * int(sample_rate)) * 2)
            io = sink.start()
            if io is None or sink.error() != QAudio.Error.NoError:
                logger.warning(
                    "Could not open the audio output (%s)", sink.error().name
                )
                sink.stop()
                return False
        except Exception as e:  # pragma: no cover - backend-dependent
            logger.warning(f"Could not start audio sink: {e}")
            return False
        self._sink = sink
        self._io = io
        self._sample_rate = int(sample_rate)
        self._backlog = bytearray()
        self._last_write = None
        return True

    def stop(self) -> None:
        if self._sink is not None:
            try:
                self._sink.stop()
            except Exception as e:  # pragma: no cover
                logger.debug(f"Audio sink stop failed: {e}")
        self._sink = None
        self._io = None
        self._sample_rate = 0
        self._backlog = bytearray()
        self._last_write = None

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, float(volume)))
        if self._sink is not None:
            self._sink.setVolume(self._volume)

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    def write(self, samples: np.ndarray) -> None:
        """Push mono float samples in [-1, 1].

        What the output can't take yet stays queued for the next call (see
        the module notes), so a partial write loses nothing.
        """
        if self._io is None or self._muted:
            return
        if samples is None or len(samples) == 0:
            return
        # Clip, scale to int16
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        now = time.monotonic()
        continuing = (
            self._last_write is not None and now - self._last_write <= self.CONTINUOUS_S
        )
        self._last_write = now
        self._backlog += pcm
        self._pump(continuing)

    def _bytes(self, seconds: float) -> int:
        """``seconds`` of audio in bytes (whole samples)."""
        return int(seconds * self._sample_rate) * 2

    def queued_seconds(self) -> float:
        """Audio waiting to be played: in the output and not yet taken."""
        if self._sink is None or self._sample_rate <= 0:
            return 0.0
        return self._queued_bytes() / (2.0 * self._sample_rate)

    def _in_output(self) -> int:
        try:
            return max(0, int(self._sink.bufferSize()) - int(self._sink.bytesFree()))
        except Exception:  # pragma: no cover - backend-dependent
            return 0

    def _queued_bytes(self) -> int:
        return self._in_output() + len(self._backlog)

    def _pump(self, continuing: bool) -> None:
        """Keep the jitter buffer near its target and hand the output what
        fits."""
        sink, io = self._sink, self._io
        if sink is None or io is None:
            return
        target = self._bytes(self.TARGET_S)
        in_output = self._in_output()
        queued = in_output + len(self._backlog)
        if in_output == 0 and 0 < len(self._backlog) < target:
            # Dry: start (or restart) with a cushion of silence.
            pad = target - len(self._backlog)
            self._backlog[0:0] = bytes(pad)
            if continuing:
                self.underruns += 1
                logger.debug(
                    "Audio underrun: the output ran dry; padded %.0f ms of silence",
                    1000.0 * pad / (2.0 * self._sample_rate),
                )
        elif queued > self._bytes(self.HIGH_S):
            # Too much queued (the clocks drift, or a stall delivered a
            # burst): drop the oldest audio not yet handed over.
            drop = min(len(self._backlog), queued - target)
            drop -= drop % 2
            if drop > 0:
                del self._backlog[:drop]
                self.overruns += 1
                logger.debug(
                    "Audio overrun: dropped %.0f ms (sound card and receiver "
                    "clocks drift, or the window stalled)",
                    1000.0 * drop / (2.0 * self._sample_rate),
                )
        try:
            free = int(sink.bytesFree())
        except Exception:  # pragma: no cover - backend-dependent
            free = len(self._backlog)
        count = min(free, len(self._backlog))
        count -= count % 2
        if count <= 0:
            return
        try:
            result = io.write(bytes(self._backlog[:count]))
        except Exception as e:  # pragma: no cover
            logger.debug(f"Audio write failed: {e}")
            return
        try:
            written = int(result)
        except (TypeError, ValueError):
            written = count  # no count reported: assume it took them all
        if written < 0:
            logger.debug("Audio write failed (the output refused the samples)")
            return
        written = min(written, count)
        del self._backlog[: written - written % 2]
