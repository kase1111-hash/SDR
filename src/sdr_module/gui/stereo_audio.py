"""
Broadcast FM stereo for the receiver chain: the stereo difference signal
through the same audio filters as the mono program, and the left and right
channels made from the two.

:class:`StereoPath` belongs to one ``_ReceiverChain`` (``main_window``). The
chain hands it each block of the FM multiplex before it turns that block
into mono audio; the path recovers L-R with
:class:`~sdr_module.dsp.fm_stereo.StereoDecoder` and runs it through fresh
copies of the chain's own audio stages (the 15 kHz low-pass with the 75 us
de-emphasis, the decimation, the resampler to 48 kHz, the DC blocker and the
noise reduction), started at the same decimation phases and times as the
mono stream's. The two come out aligned to the sample, so::

    left = mono + difference, right = mono - difference

and ``(left + right) / 2`` is exactly the mono audio the chain plays without
stereo. Plain NumPy, no Qt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..dsp.fm_stereo import MIN_RATE, StereoDecoder

__all__ = ["StereoPath", "AUDIBLE_DIFFERENCE", "MIN_FILTER_HZ"]

#: A difference signal below this (about -80 dBFS) is no stereo to play:
#: left and right are then the mono audio.
AUDIBLE_DIFFERENCE = 1e-4
#: Stereo switched off while it plays: the difference signal fades out
#: over this long (and back in, if it is switched on again meanwhile).
FADE_S = 0.02
#: The narrowest channel filter stereo plays with. Broadcast FM's 200 kHz
#: filter holds a stereo station's FM spectrum; the 100 kHz one cuts so
#: much of it that left and right came out only 6-8 dB apart (and its
#: distortion read as hiss): mono instead.
MIN_FILTER_HZ = 150e3


class StereoPath:
    """The stereo difference signal of a receiver chain (see the module
    docstring)."""

    def __init__(self, chain: Any):
        self.decoder = StereoDecoder(chain.channel_rate)
        #: False until it follows the chain's stream (see :meth:`sync`);
        #: set False to make the next block start afresh.
        self.running = False
        self._stage_cache: Dict[Tuple[bool, bool], List[Any]] = {}
        self._stages: List[Any] = []
        self._resampler: Any = None
        self._dc: Any = None
        self._nr: Any = None
        self._nr_source: Any = None
        self._fade = 1.0  # the difference signal's gain (see process)
        self._fade_step = 1.0 / (FADE_S * float(chain.channel_rate))

    @staticmethod
    def usable(chain: Any) -> bool:
        """Whether the chain's channel holds the whole multiplex (L-R
        reaches 53 kHz) and its filter the station's FM spectrum (see
        :data:`MIN_FILTER_HZ`)."""
        return (
            float(chain.channel_rate) >= MIN_RATE
            and float(chain.bandwidth) >= MIN_FILTER_HZ
        )

    # ------------------------------------------------------------------
    def sync(self, chain: Any) -> None:
        """Follow the chain's mono stream from where it is now (before it
        takes the next block): copies of its audio filters with empty
        histories, at the same decimation phases and output times, and the
        decoder starting afresh (mono, widening to stereo as the pilot is
        heard)."""
        key = (bool(chain._deemphasis), bool(chain._voice))
        stages = self._stage_cache.get(key)
        if stages is None:
            stages = self._stage_cache[key] = chain._build_audio_stages()
        for mine, theirs in zip(stages, chain._audio, strict=True):
            mine.reset()
            mine._phase = theirs._phase
        self._stages = stages
        theirs = chain._resampler
        if theirs is None:
            self._resampler = None
        else:
            mine = self._resampler
            if mine is None or mine.step != theirs.step:
                mine = self._resampler = type(theirs)(
                    chain.intermediate_rate, chain.audio_rate
                )
            mine.reset()
            mine._time = theirs._time
        source = chain._dc_blocker
        self._dc = chain._make_dc_blocker(source.corner_hz)
        self._dc._waiting = source._waiting
        if not source._waiting:
            self._dc._x = 0.0  # (what came before was silence)
        self._nr = self._nr_source = None
        self._fade = 1.0
        self.decoder.reset()
        self.running = True

    def process(
        self, mpx: np.ndarray, noise_reduction: Any = None, fade_out: bool = False
    ) -> np.ndarray:
        """The difference signal (float32, 48 kHz, as long as the mono
        audio of the same block) for a block of the multiplex (scaled so
        the deviation is 1). ``noise_reduction``: the chain's (the same
        kind is applied here). ``fade_out``: stereo is switched off: the
        difference fades out over ``FADE_S`` (and in again over as long
        once ``fade_out`` is False)."""
        difference = self.decoder.process(mpx)
        n = len(difference)
        if n and (fade_out or self._fade < 1.0):
            step = -self._fade_step if fade_out else self._fade_step
            ramp = self._fade + step * np.arange(1, n + 1, dtype=np.float64)
            ramp = np.clip(ramp, 0.0, 1.0)
            self._fade = float(ramp[-1])
            difference *= ramp.astype(np.float32)
        for stage in self._stages:
            difference = stage.process(difference)
        if self._resampler is not None:
            difference = self._resampler.process(difference)
        difference = self._dc.process(difference)
        nr = self._noise_reduction(noise_reduction)
        if nr is not None:
            difference = np.asarray(nr.process(difference), dtype=np.float32)
        return np.asarray(difference, dtype=np.float32)

    def _noise_reduction(self, source: Any) -> Any:
        """A noise reduction like the chain's, for the difference signal
        (its own stream: the same latency, so the two stay aligned)."""
        if source is None:
            self._nr = self._nr_source = None
            return None
        if source is not self._nr_source:
            self._nr_source = source
            self._nr = type(source)(source.sample_rate, source.config)
        if hasattr(source, "clean"):
            self._nr.clean = source.clean
        return self._nr

    @property
    def faded_out(self) -> bool:
        """Whether a fade out has silenced the difference signal (its
        filters may still be emptying: see :meth:`mix`)."""
        return self._fade <= 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def audible(difference: np.ndarray) -> bool:
        """Whether a difference signal is stereo to play (above
        :data:`AUDIBLE_DIFFERENCE`)."""
        values = np.asarray(difference)
        return bool(values.size) and float(np.max(np.abs(values))) >= (
            AUDIBLE_DIFFERENCE
        )

    @staticmethod
    def mix(mono: np.ndarray, difference: np.ndarray) -> Optional[np.ndarray]:
        """``(n, 2)`` left and right from the mono audio and the difference
        signal (clipped to [-1, 1]), or None when the difference is too
        small to hear (the mono audio is then what plays)."""
        mono = np.asarray(mono, dtype=np.float32)
        n = min(len(mono), len(difference))
        if n == 0 or not StereoPath.audible(difference[:n]):
            return None
        out = np.empty((len(mono), 2), dtype=np.float32)
        out[:, 0] = mono
        out[:, 1] = mono
        out[:n, 0] += difference[:n]
        out[:n, 1] -= difference[:n]
        return np.clip(out, -1.0, 1.0, out=out)
