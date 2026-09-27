"""
Audio output sink.

Thin wrapper around QAudioSink (PyQt6.QtMultimedia). Accepts a mono
int16 stream and plays it on the default output device. Absent a working
Qt multimedia backend, falls back to a no-op so the rest of the GUI
still works.

``start()`` reports whether the output really opened: it returns False when
there is no output device or Qt could not open it, so callers can tell the
user instead of playing into nothing.
"""

from __future__ import annotations

import logging
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

    def __init__(self) -> None:
        self._sink = None
        self._io: Optional["QIODevice"] = None
        self._sample_rate = 0
        self._muted = False
        self._volume = 0.7

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

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, float(volume)))
        if self._sink is not None:
            self._sink.setVolume(self._volume)

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)

    def write(self, samples: np.ndarray) -> None:
        """Push mono float samples in [-1, 1]."""
        if self._io is None or self._muted:
            return
        if samples is None or len(samples) == 0:
            return
        # Clip, scale to int16
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        try:
            self._io.write(pcm)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Audio write failed: {e}")
