"""
Record what you hear: the demodulated audio as a WAV file.

An :class:`AudioTake` writes 48 kHz, 16-bit mono WAV from the audio the
receiver demodulates (before the volume control), whether or not the
speaker is on. While the squelch is closed it records silence, so the
recording keeps time with what was received; the audio fades in and out
over a few milliseconds where the squelch opens and closes, as the speaker
does. The WAV header is kept up to date as it records, so even a recording
cut short (a crash) plays.

Files are named after what was tuned: ``<freq>_<mode>_<date>_<time>.wav``
(:func:`audio_file_name`), e.g. ``145.800MHz_FM_20261030_142233.wav``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ..dsp.recording import AudioRecorder, AudioSampleFormat
from .audio_chain import AudioJoin

logger = logging.getLogger(__name__)

AUDIO_RATE = 48000


def audio_file_name(freq_hz: float, mode: str, when: Optional[float] = None) -> str:
    """``145.800MHz_FM_20261030_142233.wav`` (local time)."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
    mode = "".join(ch for ch in str(mode) if ch.isalnum()) or "audio"
    return f"{float(freq_hz) / 1e6:.3f}MHz_{mode}_{stamp}.wav"


class AudioTake:
    """One audio recording in progress (see the module docstring).

    Raises OSError from the constructor when the file can't be created.
    """

    def __init__(
        self,
        path: str,
        freq_hz: float = 0.0,
        mode: str = "",
        rate: int = AUDIO_RATE,
    ):
        self.path = str(path)
        self.rate = int(rate)
        self.freq_hz = float(freq_hz)
        self.mode = str(mode)
        self._recorder = AudioRecorder(
            sample_rate=self.rate,
            sample_format=AudioSampleFormat.INT16,
            channels=1,
            normalize=False,
        )
        self._recorder.set_metadata(source_frequency=self.freq_hz, demodulation=mode)
        self._recorder.start(self.path)
        self._join = AudioJoin(self.rate)
        self._carry = 0.0  # fraction of a sample of silence still owed
        self.samples = 0
        #: Seconds of audio that were heard (the squelch open).
        self.heard_s = 0.0
        self.paused = False
        self.closed = False
        #: What the last block came from (the window's receiver chain and
        #: settings): a block from another one is crossfaded in.
        self.stream_key: Optional[tuple] = None

    @property
    def seconds(self) -> float:
        return self.samples / float(self.rate)

    def write(self, audio: np.ndarray, open_: bool, continuous: bool = True) -> None:
        """A block of demodulated audio: recorded as it is while the
        squelch is ``open_``, as silence of the same length otherwise.
        ``continuous`` False: it doesn't follow the last block (a new
        receiver chain), so it is crossfaded in."""
        if self.paused or self.closed:
            return
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio)
        if n == 0:
            return
        join = self._join
        if open_:
            out = join.play(audio) if continuous else join.join(audio)
            self.heard_s += n / float(self.rate)
        elif join.playing:
            tail = join.fade_out(audio)
            out = np.zeros(n, dtype=np.float32)
            out[: len(tail)] = tail
        else:
            out = np.zeros(n, dtype=np.float32)
        self._write(out)

    def silence(self, seconds: float) -> None:
        """``seconds`` with no audio (a mode without audio, nothing
        demodulated): silence, after fading out what was playing."""
        if self.paused or self.closed or seconds <= 0:
            return
        wanted = seconds * self.rate + self._carry
        n = int(wanted)
        self._carry = wanted - n
        if n <= 0:
            return
        out = np.zeros(n, dtype=np.float32)
        if self._join.playing:
            tail = self._join.release()[:n]
            out[: len(tail)] = tail
        self._write(out)

    def _write(self, out: np.ndarray) -> None:
        self._recorder.write(np.clip(out, -1.0, 1.0))
        self.samples += len(out)

    def stop(self) -> float:
        """Finish the file; returns its length in seconds."""
        if not self.closed:
            self.closed = True
            if self._join.playing:
                self._write(self._join.release())
            try:
                self._recorder.stop()
            except Exception as e:  # pragma: no cover - disk trouble
                logger.warning(f"Could not finish the audio recording {self.path}: {e}")
        return self.seconds
