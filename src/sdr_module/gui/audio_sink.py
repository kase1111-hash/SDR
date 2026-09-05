"""
Audio output sink.

Thin wrapper around QAudioSink (PyQt6.QtMultimedia). Accepts a mono
int16 stream and plays it on the default output device. Absent a working
Qt multimedia backend, falls back to a no-op so the rest of the GUI
still works.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QIODevice
    from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

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

    def start(self, sample_rate: int = 48000) -> bool:
        if not HAS_QT_AUDIO:
            logger.info("QtMultimedia not available; audio output disabled")
            return False
        if self._sink is not None and self._sample_rate == sample_rate:
            return True
        self.stop()
        try:
            fmt = QAudioFormat()
            fmt.setSampleRate(int(sample_rate))
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

            device = QMediaDevices.defaultAudioOutput()
            self._sink = QAudioSink(device, fmt)
            self._sink.setVolume(self._volume)
            self._io = self._sink.start()
            self._sample_rate = int(sample_rate)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not start audio sink: {e}")
            self._sink = None
            self._io = None
            return False

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
