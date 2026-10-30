"""
What happens to the tuned channel after the channel filter: its level, the
squelch, and the audio clean-up between the demodulator and the speaker.

Everything here is plain NumPy (no Qt), vectorised per block, so it can be
tested without a window and costs little at 30 frames a second:

* :func:`channel_power_db`: the level of a whole block of the tuned channel
  (what LEVEL, the squelch and the S-meter compare).
* :class:`NoiseFloorTracker`: the noise floor of the spectrum, and from it
  the noise power of a channel of a given width (for the Auto squelch and
  the meter's SNR).
* :class:`SquelchGate`: open at the threshold, close only after the level
  stays below it (with hysteresis) for a hang time.
* :class:`DcBlocker`, :class:`AudioAgc`: a one-pole DC blocker and a peak
  AGC with selectable speed, for the demodulated audio.
* :class:`AudioJoin`: short fades so the audio never steps (squelch opening
  or closing, a new receiver chain after a Bandwidth or Mode change).
* :class:`ToneSquelch`: CTCSS tone squelch on FM audio.
* :class:`IqConditioner`: optional DC-spike removal and I/Q balance
  correction of the raw samples.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..dsp.filters import (
    CTCSSDetector,
    NoiseReduction,
    NoiseReductionConfig,
    NoiseReductionMethod,
    ctcss_label,
)
from ..utils.iq import apply_iq_imbalance_correction, estimate_iq_imbalance

#: Level reported for an empty block (dBFS).
FLOOR_DB = -150.0

#: Short fade used on every audio discontinuity (squelch, chain change).
FADE_S = 0.005

# The mean of an exponential distribution (the power of a noise-only FFT
# bin) is 1/ln 2 times its median: +1.59 dB.
_MEDIAN_TO_MEAN_DB = 10.0 * math.log10(1.0 / math.log(2.0))
# Equivalent noise bandwidth of the spectrum's Hann window, in bins.
_HANN_ENBW_BINS = 1.5


def channel_power_db(samples: np.ndarray) -> float:
    """Mean power of a block of the tuned channel, in dBFS.

    A full-scale complex tone is 0 dBFS, like the spectrum's peak of the
    same tone, so LEVEL reads the same as the spectrum for a carrier; for
    noise or a wide signal it is the power in the whole channel (what an
    S-meter measures), not the strongest bin.
    """
    x = np.asarray(samples)
    if x.size == 0:
        return FLOOR_DB
    # (einsum works on strided views without copying, and without the
    # threaded BLAS a dot product could wake.)
    if np.iscomplexobj(x):
        re, im = x.real, x.imag
        total = float(np.einsum("i,i->", re, re)) + float(np.einsum("i,i->", im, im))
    else:
        total = float(np.einsum("i,i->", x, x))
    power = total / x.size
    if not power > 0.0 or not math.isfinite(power):
        return FLOOR_DB
    return max(FLOOR_DB, 10.0 * math.log10(power))


class NoiseFloorTracker:
    """The noise floor of the spectrum, per bin, and of a channel.

    Each spectrum line (dBFS per bin, as the spectrum shows it) gives the
    median of its central 80 % of bins (the band edges roll off), which a
    few strong signals barely move; the mean noise power per bin is 1.59 dB
    above that median. It is smoothed with a time constant of
    ``smoothing_s`` of signal time, but a change of more than ``jump_db``
    (a gain step, the AGC, another band) is taken at once: otherwise the
    Auto squelch would sit below the new noise for a moment and open.

    :meth:`channel_noise_db` scales the per-bin noise to a channel's noise
    bandwidth, giving the level an empty channel of that width reads in
    :func:`channel_power_db`.
    """

    def __init__(self, smoothing_s: float = 0.3, jump_db: float = 2.0):
        self.smoothing_s = float(smoothing_s)
        self.jump_db = float(jump_db)
        self._bin_db: Optional[float] = None
        self._bins = 0

    def reset(self) -> None:
        self._bin_db = None
        self._bins = 0

    @property
    def bin_db(self) -> Optional[float]:
        """Mean noise power of one spectrum bin (dBFS), once known."""
        return self._bin_db

    def update(self, power_db: np.ndarray, seconds: float) -> Optional[float]:
        """Take one spectrum line covering ``seconds`` of signal."""
        values = np.asarray(power_db, dtype=np.float32)
        n = values.size
        if n < 32:
            return self._bin_db
        edge = n // 10
        middle = values[edge : n - edge]
        finite = middle[np.isfinite(middle)]
        if finite.size < 16:
            return self._bin_db
        k = finite.size // 2
        median = float(np.partition(finite, k)[k])
        estimate = median + _MEDIAN_TO_MEAN_DB
        if (
            self._bin_db is None
            or self._bins != n
            or abs(estimate - self._bin_db) > self.jump_db
        ):
            self._bin_db = estimate
        else:
            alpha = 1.0 - math.exp(-max(0.0, seconds) / max(self.smoothing_s, 1e-3))
            self._bin_db += alpha * (estimate - self._bin_db)
        self._bins = n
        return self._bin_db

    def channel_noise_db(
        self, noise_bandwidth_hz: float, sample_rate: float
    ) -> Optional[float]:
        """Noise power (dBFS) of a channel ``noise_bandwidth_hz`` wide, for
        a spectrum of ``sample_rate`` span; None until a line arrived."""
        if self._bin_db is None or self._bins <= 0 or sample_rate <= 0:
            return None
        bin_hz = _HANN_ENBW_BINS * float(sample_rate) / self._bins
        width = max(float(noise_bandwidth_hz), 1.0)
        return self._bin_db + 10.0 * math.log10(width / bin_hz)


class LevelAverager:
    """Averages block levels (dBFS) over a short time, in power.

    A block of 33 ms is almost all its own level; blocks of a millisecond
    (the display loop catching up, or a narrow CW filter whose noise is
    correlated over tens of milliseconds) are averaged over about
    ``time_constant_s``, so a single noise peak can't open the squelch.
    """

    def __init__(self, time_constant_s: float = 0.025):
        self.time_constant_s = float(time_constant_s)
        self._power: Optional[float] = None

    def reset(self) -> None:
        self._power = None

    def update(self, level_db: float, seconds: float) -> float:
        power = 10.0 ** (float(level_db) / 10.0)
        if self._power is None:
            self._power = power
        else:
            alpha = 1.0 - math.exp(-max(0.0, seconds) / max(self.time_constant_s, 1e-6))
            self._power += alpha * (power - self._power)
        return max(FLOOR_DB, 10.0 * math.log10(max(self._power, 1e-30)))


class SquelchGate:
    """A squelch that opens cleanly and doesn't chatter.

    * It opens as soon as a block's level reaches the threshold.
    * Once open, it closes only after the level has stayed more than
      ``hysteresis_db`` below the threshold for ``hang_s`` seconds of
      signal: a fading signal, the gaps between Morse characters or a
      noise dip don't chop the audio.
    * With ``auto`` the threshold is the channel's noise level plus
      ``margin_db``, so it follows the gain, the AGC and the channel width
      by itself; otherwise it is ``manual_db``.
    * With ``enabled`` False it is always open.

    :meth:`update` returns ``"opened"`` or ``"closed"`` on a change (else
    ``""``), so the caller can fade the audio and reset the demodulator
    only when the squelch really closes.
    """

    def __init__(
        self,
        manual_db: float = -80.0,
        enabled: bool = True,
        auto: bool = True,
        margin_db: float = 7.0,
        hysteresis_db: float = 3.0,
        hang_s: float = 0.3,
    ):
        self.manual_db = float(manual_db)
        self.enabled = bool(enabled)
        self.auto = bool(auto)
        self.margin_db = float(margin_db)
        self.hysteresis_db = float(hysteresis_db)
        self.hang_s = float(hang_s)
        self.is_open = False
        self._below_s = 0.0
        self._last_threshold: Optional[float] = None

    def reset(self) -> None:
        """Closed, with no hang time running (a new stream)."""
        self.is_open = False
        self._below_s = 0.0

    def threshold_db(self, noise_db: Optional[float]) -> float:
        """The threshold in effect for a channel noise level ``noise_db``
        (the manual one while the noise level is unknown)."""
        if self.auto and noise_db is not None and math.isfinite(noise_db):
            return float(noise_db) + self.margin_db
        return self.manual_db

    @property
    def last_threshold_db(self) -> Optional[float]:
        """The threshold the last :meth:`update` used."""
        return self._last_threshold

    def update(self, level_db: float, noise_db: Optional[float], seconds: float) -> str:
        """Decide for one block of ``seconds`` at ``level_db``."""
        threshold = self.threshold_db(noise_db)
        self._last_threshold = threshold
        was_open = self.is_open
        if not self.enabled:
            self.is_open = True
            self._below_s = 0.0
        elif not self.is_open:
            if level_db >= threshold:
                self.is_open = True
                self._below_s = 0.0
        elif level_db >= threshold - self.hysteresis_db:
            self._below_s = 0.0
        else:
            self._below_s += max(0.0, float(seconds))
            if self._below_s >= self.hang_s:
                self.is_open = False
                self._below_s = 0.0
        if self.is_open != was_open:
            return "opened" if self.is_open else "closed"
        return ""


class DcBlocker:
    """One-pole DC blocker for real audio, vectorised.

    ``y[n] = x[n] - x[n-1] + a * y[n-1]`` with ``a = exp(-2 pi fc / rate)``:
    flat above a few times ``fc``, zero at DC. The recursion is evaluated a
    chunk at a time with a scaled cumulative sum (exact, no Python loop per
    sample).

    For ``startup`` samples after :meth:`reset` (while the filters in front
    of it fill) it outputs silence and only follows the input; then it
    starts from the last of them as the previous input, so a stream that
    starts with an offset starts at zero instead of stepping, and the noise
    of a filter's start-up doesn't linger in the recursion.
    """

    _CHUNK = 1024

    def __init__(self, corner_hz: float, rate: float, startup: int = 0):
        self.corner_hz = float(corner_hz)
        self.a = float(math.exp(-2.0 * math.pi * float(corner_hz) / float(rate)))
        powers = self.a ** np.arange(self._CHUNK, dtype=np.float64)
        self._powers = powers
        self._inverse = 1.0 / powers
        self.startup = max(0, int(startup))
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._y = 0.0
        self._waiting = self.startup

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64)
        n = x.size
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if self._waiting > 0:
            held = min(self._waiting, n)
            self._waiting -= held
            self._x = float(x[held - 1])
            self._y = 0.0
            if held == n:
                return np.zeros(n, dtype=np.float32)
            rest = self.process(x[held:])
            return np.concatenate((np.zeros(held, dtype=np.float32), rest))
        if self._x is None:
            self._x = float(x[0])
        out = np.empty(n, dtype=np.float64)
        prev_x, prev_y, a = self._x, self._y, self.a
        for start in range(0, n, self._CHUNK):
            chunk = x[start : start + self._CHUNK]
            m = chunk.size
            diff = np.empty(m, dtype=np.float64)
            diff[0] = chunk[0] - prev_x
            diff[1:] = chunk[1:] - chunk[:-1]
            p = self._powers[:m]
            y = p * (a * prev_y + np.cumsum(diff * self._inverse[:m]))
            out[start : start + m] = y
            prev_x, prev_y = float(chunk[-1]), float(y[-1])
        self._x, self._y = prev_x, prev_y
        return out.astype(np.float32)


#: Audio AGC speeds: the release time constant (seconds) of the peak AGC
#: (SSB and CW). ``"off"`` is a fixed gain.
AGC_SPEEDS: Dict[str, float] = {"fast": 0.1, "medium": 0.5, "slow": 2.0}
#: The same speeds for the AM carrier level (attack and release).
AM_CARRIER_TIME_S: Dict[str, float] = {"fast": 0.05, "medium": 0.15, "slow": 0.6}
AGC_SPEED_LABELS: Tuple[Tuple[str, str, str], ...] = (
    ("off", "Off", "No automatic level: loud signals clip, weak ones are quiet"),
    ("fast", "Fast", "Recovers in about 0.1 s: CW and fast-changing signals"),
    ("medium", "Medium", "Recovers in about 0.5 s: most voice"),
    ("slow", "Slow", "Recovers in about 2 s: steady SSB and AM voice"),
)
DEFAULT_AGC_SPEED = "medium"

#: CW beat note (pitch) range and default, in Hz.
CW_PITCH_RANGE = (400, 1000)
DEFAULT_CW_PITCH_HZ = 700


class AudioOptions:
    """Listening options the receiver chain reads on every block.

    The window keeps one instance and hands it to each chain it builds,
    so changing an option takes effect on the next block without
    rebuilding the chain (no click):

    * ``cw_pitch_hz``: the CW beat note.
    * ``agc_speed``: a key of :data:`AGC_SPEEDS`, or ``"off"`` (AM, SSB
      and CW; FM has no AGC).
    * ``noise_reduction``: a streaming
      :class:`~sdr_module.dsp.filters.NoiseReduction`, or None.
    """

    def __init__(self) -> None:
        self.cw_pitch_hz: float = float(DEFAULT_CW_PITCH_HZ)
        self.agc_speed: str = DEFAULT_AGC_SPEED
        self.noise_reduction: Optional[NoiseReduction] = None


class AudioAgc:
    """Peak AGC for demodulated audio (SSB, CW).

    The block is split into 5 ms pieces. Each piece's peak raises the
    tracked level at once (no overshoot: a piece is scaled by the gain its
    own peak needs, or the next piece's if that is louder); the level then
    falls with the release time constant of the speed. The gain is
    interpolated between pieces, so it never steps. The level never drops
    below ``floor`` (-80 dBFS), so an empty channel isn't turned up into
    full-scale hiss.

    Speed ``"off"`` applies ``fixed_gain`` instead.
    """

    PIECE_S = 0.005

    def __init__(
        self,
        rate: float,
        speed: str = DEFAULT_AGC_SPEED,
        target: float = 0.5,
        floor: float = 1e-4,
        fixed_gain: float = 100.0,
    ):
        self.rate = float(rate)
        self.speed = speed
        self.target = float(target)
        self.floor = float(floor)
        self.fixed_gain = float(fixed_gain)
        self.reset()

    def reset(self) -> None:
        self._level = 0.0
        self._gain: Optional[float] = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        n = x.size
        if n == 0:
            return x
        release_s = AGC_SPEEDS.get(self.speed)
        if release_s is None:  # off
            return np.clip(x * np.float32(self.fixed_gain), -1.0, 1.0)
        piece = max(1, int(round(self.PIECE_S * self.rate)))
        count = -(-n // piece)
        padded = np.zeros(count * piece, dtype=np.float32)
        padded[:n] = np.abs(x)
        peaks = padded.reshape(count, piece).max(axis=1).astype(np.float64)
        decay = math.exp(-piece / (release_s * self.rate))
        levels = np.empty(count, dtype=np.float64)
        level = self._level
        for i in range(count):
            level = max(float(peaks[i]), level * decay)
            levels[i] = level
        self._level = level
        gains = self.target / np.maximum(levels, self.floor)
        # A piece takes the smaller of its gain and the next one's: the
        # gain is already down when a loud piece starts.
        gains[:-1] = np.minimum(gains[:-1], gains[1:])
        start = gains[0] if self._gain is None else min(self._gain, gains[0])
        ends = np.concatenate(([start], gains))
        ramp = np.arange(1, piece + 1, dtype=np.float64) / piece
        curve = (ends[:-1, None] + (ends[1:] - ends[:-1])[:, None] * ramp).ravel()
        self._gain = float(gains[-1])
        out = x * curve[:n].astype(np.float32)
        return np.clip(out, -1.0, 1.0).astype(np.float32)


def raised_cosine(n: int) -> np.ndarray:
    """``n`` samples rising smoothly from 0 to (almost) 1."""
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    k = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (0.5 - 0.5 * np.cos(np.pi * k)).astype(np.float32)


class AudioJoin:
    """Keeps the audio sent to the speaker free of steps.

    * :meth:`fade_in` the first block after silence (the squelch opened).
    * :meth:`fade_out` the last one: it returns only the first ``FADE_S``,
      ramped down to zero (the squelch closed).
    * :meth:`join` a block that does not continue the previous one (a new
      receiver chain after a Bandwidth, Mode or deviation change starts
      from empty filters): it starts from the last sample played and
      crossfades into the new audio.
    """

    def __init__(self, rate: float):
        self.rate = float(rate)
        self._n = max(1, int(round(FADE_S * self.rate)))
        self._ramp = raised_cosine(self._n)
        self.last = 0.0
        self.playing = False

    def _ramp_for(self, n: int) -> np.ndarray:
        return self._ramp[:n] if n <= self._n else raised_cosine(n)

    def _remember(self, audio: np.ndarray) -> np.ndarray:
        if audio.size:
            self.last = float(audio[-1])
        self.playing = True
        return audio

    def play(self, audio: np.ndarray) -> np.ndarray:
        """A block that follows the previous one seamlessly."""
        if not self.playing:
            return self.fade_in(audio)
        return self._remember(np.asarray(audio, dtype=np.float32))

    def fade_in(self, audio: np.ndarray) -> np.ndarray:
        out = np.array(audio, dtype=np.float32, copy=True)
        m = min(self._n, out.size)
        out[:m] *= self._ramp[:m]
        return self._remember(out)

    def join(self, audio: np.ndarray) -> np.ndarray:
        """Crossfade from the last sample played into ``audio``."""
        if not self.playing:
            return self.fade_in(audio)
        out = np.array(audio, dtype=np.float32, copy=True)
        m = min(self._n, out.size)
        ramp = self._ramp[:m]
        out[:m] = np.float32(self.last) * (1.0 - ramp) + out[:m] * ramp
        return self._remember(out)

    def fade_out(self, audio: np.ndarray) -> np.ndarray:
        """The start of ``audio`` ramped down to silence."""
        out = np.array(audio[: self._n], dtype=np.float32, copy=True)
        if self.playing and out.size:
            # Continue from what was playing, then down to zero.
            ramp = 1.0 - self._ramp_for(out.size)
            out *= ramp
        elif not self.playing:
            out = np.zeros(0, dtype=np.float32)
        self.playing = False
        self.last = 0.0
        return out

    def stop(self) -> None:
        """Silence without a fade (nothing more can be played)."""
        self.playing = False
        self.last = 0.0


#: Noise reduction strengths: (key, label, tooltip).
NOISE_REDUCTION_LEVELS: Tuple[Tuple[str, str, str], ...] = (
    ("off", "Off", "No noise reduction"),
    (
        "light",
        "Light",
        "Takes the edge off steady hiss and keeps the voice natural",
    ),
    (
        "strong",
        "Strong",
        "Removes more hiss on weak signals; voices can sound a little watery",
    ),
)


def make_noise_reduction(level: str, rate: float) -> Optional[NoiseReduction]:
    """A streaming noise reduction for 48 kHz audio at ``level``
    (``"light"`` or ``"strong"``), or None for ``"off"``."""
    if level == "light":
        config = NoiseReductionConfig(
            method=NoiseReductionMethod.WIENER,
            fft_size=1024,
            overlap=0.5,
            gain_floor=0.3,
        )
    elif level == "strong":
        config = NoiseReductionConfig(
            method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
            fft_size=1024,
            overlap=0.5,
            subtraction_factor=2.0,
            floor_factor=0.12,
        )
    else:
        return None
    return NoiseReduction(rate, config)


class ToneSquelch:
    """CTCSS tone squelch for FM audio.

    ``tone_hz`` None: off (always open), but the strongest sub-audible tone
    is still reported in :attr:`detected_hz` for the readout. Otherwise the
    audio only plays while that tone is present; the adjacent EIA tones
    are rejected (see :class:`~sdr_module.dsp.filters.CTCSSDetector`).
    """

    def __init__(self, rate: float):
        self.rate = float(rate)
        self.tone_hz: Optional[float] = None
        self._detector = CTCSSDetector(self.rate)

    def reset(self) -> None:
        self._detector.reset()

    @property
    def detected_hz(self) -> Optional[float]:
        return self._detector.detected_tone

    def detected_label(self) -> str:
        tone = self.detected_hz
        return ctcss_label(tone) if tone is not None else ""

    def process(self, audio: np.ndarray) -> bool:
        """Feed FM audio; True while the speaker may play."""
        self._detector.process(audio)
        if self.tone_hz is None:
            return True
        return self._detector.is_present(self.tone_hz)


class IqConditioner:
    """Optional clean-up of the raw I/Q samples before anything else.

    * ``remove_dc``: subtracts the slowly tracked mean of the samples (time
      constant 0.2 s), which removes the spike most SDRs show at the
      center of the span (the LO leaking through) without touching
      signals more than a few hertz from it. A carrier exactly at the
      center is removed too.
    * ``balance_iq``: estimates the I/Q gain and phase error over a few
      seconds and corrects it, which suppresses mirror images of strong
      signals (on the other side of the center).

    Both are off by default and cost a few operations per sample.
    """

    DC_TIME_S = 0.2
    IQ_TIME_S = 2.0

    def __init__(self) -> None:
        self.remove_dc = False
        self.balance_iq = False
        self.reset()

    def reset(self) -> None:
        self._dc: Optional[complex] = None
        self._gain: Optional[float] = None
        self._phase: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.remove_dc or self.balance_iq

    @property
    def imbalance(self) -> Tuple[Optional[float], Optional[float]]:
        """The tracked (gain error, phase error in radians)."""
        return self._gain, self._phase

    def process(self, samples: np.ndarray, rate: float) -> np.ndarray:
        if not self.active:
            return samples
        x = np.asarray(samples, dtype=np.complex64)
        n = x.size
        if n == 0:
            return x
        if self.remove_dc:
            mean = complex(np.mean(x))
            if self._dc is None:
                start = end = mean
            else:
                alpha = 1.0 - math.exp(-n / (self.DC_TIME_S * max(rate, 1.0)))
                start = self._dc
                end = self._dc + alpha * (mean - self._dc)
            self._dc = end
            if n > 1 and start != end:
                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                x = x - (np.complex64(start) + np.complex64(end - start) * ramp)
            else:
                x = x - np.complex64(end)
        if self.balance_iq and n >= 64:
            gain, phase = estimate_iq_imbalance(x)
            gain, phase = float(gain), float(phase)
            if self._gain is None or self._phase is None:
                self._gain, self._phase = gain, phase
            else:
                alpha = 1.0 - math.exp(-n / (self.IQ_TIME_S * max(rate, 1.0)))
                self._gain += alpha * (gain - self._gain)
                self._phase += alpha * (phase - self._phase)
            x = apply_iq_imbalance_correction(x, self._gain, self._phase).astype(
                np.complex64
            )
        return x
