"""
Streaming channel selection: shift a channel to 0 Hz and decimate it.

The live protocol decoders (:mod:`sdr_module.dsp.live_decoders`) receive the
receiver's raw I/Q blocks, a whole span of up to 20 MS/s, but a pager, a
packet station or an ACARS burst occupies only a few kHz somewhere in it.
:class:`Channelizer` picks that channel out:

1. a phase-continuous NCO mixes the block by minus the channel's offset
   from the centre (the VFO offset), so the channel sits at 0 Hz;
2. FIR stages low-pass and decimate it to a protocol rate of about
   48 kHz: coarse stages that read the full-rate block once and only
   protect the channel from what would fold onto it, then the sharp
   channel filter at the lower rate. The split is planned for each rate
   (the cheapest one whose filters all reach their stopband). A much
   stronger station elsewhere in the span is removed (60 dB down) before
   anything is demodulated.

Everything carries its state between calls: a stream split into blocks of
any size gives the same output as the whole stream processed at once (to
floating-point rounding).

NumPy only (scipy is optional for this project); no Qt.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "lowpass_taps",
    "FirDecimator",
    "Nco",
    "Channelizer",
    "MovingAverage",
    "Delay",
    "FMDiscriminator",
    "decimation_factor",
]

# Longest filter designed (a guard against degenerate requests).
_MAX_TAPS = 2047
# Stopband attenuation of every channel filter stage.
_ATTEN_DB = 60.0


def _band_edges(pass_hz: float, stop_hz: float, rate_hz: float) -> Tuple[float, float]:
    """The pass and stop edges :func:`lowpass_taps` designs for (below the
    Nyquist frequency, the stop edge above the pass edge)."""
    nyquist = float(rate_hz) / 2.0
    pass_hz = max(0.0, min(float(pass_hz), 0.98 * nyquist))
    return pass_hz, min(max(float(stop_hz), pass_hz * 1.02 + 1.0), nyquist)


def _taps_needed(
    pass_hz: float, stop_hz: float, rate_hz: float, atten_db: float = _ATTEN_DB
) -> int:
    """The length (odd) of the filter :func:`lowpass_taps` designs, before
    it is cut to ``_MAX_TAPS`` (Kaiser's estimate)."""
    pass_hz, stop_hz = _band_edges(pass_hz, stop_hz, rate_hz)
    width = max((stop_hz - pass_hz) / float(rate_hz), 1e-6)  # cycles per sample
    taps = int(math.ceil((atten_db - 7.95) / (2.285 * 2.0 * math.pi * width))) + 1
    return max(taps, 5) | 1


def lowpass_taps(
    pass_hz: float, stop_hz: float, rate_hz: float, atten_db: float = _ATTEN_DB
) -> np.ndarray:
    """Kaiser-windowed sinc low-pass, flat to ``pass_hz`` and at least
    ``atten_db`` down from ``stop_hz``, with unity gain at DC (float32,
    odd length, linear phase).

    A filter that would need more than ``_MAX_TAPS`` taps is cut to that
    length and then misses its stopband; :class:`Channelizer` plans its
    stages so that none of them needs that many."""
    rate_hz = float(rate_hz)
    pass_hz, stop_hz = _band_edges(pass_hz, stop_hz, rate_hz)
    taps = min(_taps_needed(pass_hz, stop_hz, rate_hz, atten_db), _MAX_TAPS) | 1
    cutoff = (pass_hz + stop_hz) / 2.0 / rate_hz
    t = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * t)
    beta = (
        0.1102 * (atten_db - 8.7)
        if atten_db > 50
        else 0.5842 * (atten_db - 21) ** 0.4 + 0.07886 * (atten_db - 21)
    )
    h *= np.kaiser(taps, beta)
    return (h / np.sum(h)).astype(np.float32)


_VECDOT = getattr(np, "vecdot", None)  # numpy 2.0+


def _row_dot(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``rows @ weights`` for real ``weights``, without BLAS (a threaded BLAS
    spins its worker threads after every call, costing more than the work)."""
    if _VECDOT is not None:
        return _VECDOT(weights, rows)  # conj(weights) . row; weights are real
    return np.einsum("ij,j->i", rows, weights)


class FirDecimator:
    """Streaming FIR low-pass followed by integer decimation.

    Only the kept outputs are computed. A large factor works polyphase: the
    block is viewed as contiguous rows of ``factor`` samples and each group
    of ``factor`` (reversed) taps is dotted with every row. A small factor
    would make those rows too short to be efficient, so there each tap
    multiplies a strided view of the block instead. The filter history and
    the decimation phase carry over between calls, so block boundaries
    leave no trace.
    """

    _POLYPHASE_FROM = 8  # smallest factor computed row by row

    def __init__(self, taps: np.ndarray, factor: int = 1):
        taps = np.asarray(taps, dtype=np.float32).ravel()
        if not len(taps):
            taps = np.ones(1, dtype=np.float32)
        self.factor = max(1, int(factor))
        self.taps = taps
        # Output j is the window of len(taps) samples ending at its input
        # sample, dotted with the reversed taps.
        self._reversed = [float(w) for w in taps[::-1]]
        self._groups = 0
        if self.factor >= self._POLYPHASE_FROM:
            self._groups = -(-len(taps) // self.factor)
            padded = np.zeros(self._groups * self.factor, dtype=np.float32)
            padded[: len(taps)] = taps[::-1]
            # Row g multiplies input row (j + g) for output j.
            self._weights = padded.reshape(self._groups, self.factor)
        self._length = (
            self._groups * self.factor if self._groups else len(self._reversed)
        )
        self._history: Optional[np.ndarray] = None
        self._phase = 0  # index, in the next block, of the next kept sample

    def reset(self) -> None:
        """Forget the stream (filter history and decimation phase)."""
        self._history = None
        self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        dtype = np.complex64 if np.iscomplexobj(x) else np.float32
        factor = self.factor
        length = self._length
        if self._history is None or self._history.dtype != dtype:
            self._history = np.zeros(length - 1, dtype=dtype)
        n = len(x)
        if n == 0:
            return np.zeros(0, dtype=dtype)
        padded = np.concatenate((self._history, x.astype(dtype, copy=False)))
        self._history = padded[len(padded) - (length - 1) :].copy()
        phase = self._phase
        count = -(-(n - phase) // factor) if n > phase else 0
        self._phase = phase + count * factor - n
        if count <= 0:
            return np.zeros(0, dtype=dtype)
        if self._groups:
            rows = count + self._groups - 1
            view = padded[phase : phase + rows * factor].reshape(rows, factor)
            weights = self._weights
            out = _row_dot(view[0:count], weights[0])
            for g in range(1, self._groups):
                out += _row_dot(view[g : g + count], weights[g])
            return out.astype(dtype, copy=False)
        stop = phase + (count - 1) * factor + 1
        out = np.zeros(count, dtype=dtype)
        scratch = np.empty(count, dtype=dtype)
        for m, weight in enumerate(self._reversed):
            if weight:
                np.multiply(padded[phase + m : stop + m : factor], weight, out=scratch)
                out += scratch
        return out


class Nco:
    """Phase-continuous complex mixer: multiplies a stream by
    ``exp(-j 2 pi f t)``, moving a component at ``f`` Hz to 0 Hz.

    The phase is computed from a sample counter in float64 (no drift over
    hours), applied through a short table of phasors (cheap per sample).
    Changing the frequency keeps the phase continuous.
    """

    _BLOCK = 4096

    def __init__(self, rate: float, freq_hz: float = 0.0):
        self.rate = float(rate)
        self._freq = 0.0
        self._count = 0  # samples since the frequency was set
        self._phase0 = 0.0  # phase (cycles) at that point
        self._table: Optional[np.ndarray] = None
        self.set_frequency(freq_hz)

    @property
    def frequency(self) -> float:
        return self._freq

    def _phase_cycles(self) -> float:
        return (self._phase0 + self._freq * self._count / self.rate) % 1.0

    def set_frequency(self, freq_hz: float) -> None:
        freq_hz = float(freq_hz)
        if freq_hz == self._freq and self._table is not None:
            return
        self._phase0 = self._phase_cycles()
        self._count = 0
        self._freq = freq_hz
        k = np.arange(self._BLOCK)
        cycles = np.mod(k * (freq_hz / self.rate), 1.0)
        self._table = np.exp(-2j * np.pi * cycles).astype(np.complex64)

    def reset(self) -> None:
        self._phase0 = 0.0
        self._count = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        """``x`` mixed down by the NCO frequency (complex64)."""
        x = np.asarray(x)
        n = len(x)
        if self._freq == 0.0:
            self._count += n
            return x.astype(np.complex64, copy=False)
        if n == 0:
            return np.zeros(0, dtype=np.complex64)
        block = self._BLOCK
        starts = self._count + block * np.arange(-(-n // block))
        cycles = np.mod(self._phase0 + starts * (self._freq / self.rate), 1.0)
        phasors = np.exp(-2j * np.pi * cycles).astype(np.complex64)
        assert self._table is not None
        rot = (phasors[:, None] * self._table[None, :]).ravel()[:n]
        self._count += n
        return (x.astype(np.complex64, copy=False) * rot).astype(
            np.complex64, copy=False
        )


def _smooth_numbers(limit: int) -> List[int]:
    """Integers 1..limit with no prime factor above 5."""
    out = []
    power2 = 1
    while power2 <= limit:
        power3 = power2
        while power3 <= limit:
            value = power3
            while value <= limit:
                out.append(value)
                value *= 5
            power3 *= 3
        power2 *= 2
    return sorted(out)


def decimation_factor(rate: float, target: float, min_rate: float) -> int:
    """The decimation factor taking ``rate`` closest to ``target``.

    Only factors with no prime factor above 5 are used (they split into
    short, cheap stages), and the result never goes below ``min_rate``
    (the channel must still fit). 1 when ``rate`` is already low.
    """
    rate, target = float(rate), float(target)
    if rate <= target * 1.25 or rate / 2.0 < min_rate:
        return 1
    best, best_err = 1, abs(math.log(rate / target))
    for factor in _smooth_numbers(int(rate / max(min_rate, 1.0)) + 1):
        out = rate / factor
        if out < min_rate:
            continue
        err = abs(math.log(out / target))
        if err < best_err - 1e-12:
            best, best_err = factor, err
    return best


def _stage_cost(rate: float, factor: int, taps: int) -> float:
    """Rough CPU cost (ns per second of input at ``rate``) of a
    :class:`FirDecimator` stage, as measured with NumPy on 33 ms blocks:
    a polyphase row costs about 20 ns plus 0.4 ns per tap in it, the
    strided form (small factors) about 1 + 0.45 x ``factor`` ns per tap
    and output (its reads spread over more of the block), a NumPy call a
    few microseconds, and copying the input about 2 ns a sample."""
    outputs = rate / factor
    if factor >= FirDecimator._POLYPHASE_FROM:
        groups = -(-taps // factor)
        cost = groups * (outputs * (20.0 + 0.4 * factor) + 30 * 3000.0)
    else:
        cost = taps * (outputs * (1.0 + 0.45 * factor) + 30 * 2 * 2000.0)
    return cost + 2.0 * rate


# Most decimation stages a channel is split into.
_MAX_STAGES = 4


def _plan_stages(
    rate: float, factor: int, pass_hz: float, stop_hz: float
) -> List[Tuple[int, float]]:
    """How to decimate by ``factor`` from ``rate`` keeping +/- ``pass_hz``
    and removing everything beyond +/- ``stop_hz``: ``[(factor, stop
    edge)]`` per stage, the cheapest split (by :func:`_stage_cost`) into at
    most ``_MAX_STAGES`` stages whose filters all fit in ``_MAX_TAPS``.

    The last stage is the sharp channel filter (stop edge ``stop_hz``).
    Each stage before it only protects the channel from what its own
    decimation would fold onto it: everything from its output rate minus
    ``stop_hz`` up goes, so its transition band is wide and its filter
    short for its factor. A very large factor (a narrow channel at a high
    rate) needs several such stages: one alone would need tens of
    thousands of taps.
    """
    if factor <= 1:
        return [(1, stop_hz)] if stop_hz < 0.45 * rate else []

    def plan(cap: int) -> Optional[Tuple[float, Tuple[Tuple[int, float], ...]]]:
        @lru_cache(maxsize=None)
        def best(left: int, stages: int):
            # ``left``: the decimation still to do; the rate is then:
            r = rate * left / factor
            found = None
            taps = _taps_needed(pass_hz, stop_hz, r)
            if taps <= cap:  # the sharp filter takes the rest
                found = (_stage_cost(r, left, taps), ((left, stop_hz),))
            if stages > 1:
                for d in (d for d in range(2, left) if left % d == 0):
                    stop = r / d - stop_hz
                    if stop <= pass_hz:
                        continue
                    taps = _taps_needed(pass_hz, stop, r)
                    rest = best(left // d, stages - 1) if taps <= cap else None
                    if rest is not None:
                        cost = _stage_cost(r, d, taps) + rest[0]
                        if found is None or cost < found[0]:
                            found = (cost, ((d, stop),) + rest[1])
            return found

        return best(factor, _MAX_STAGES)

    # A channel too narrow for any split (a degenerate request) gets the
    # cheapest one, its filters cut to _MAX_TAPS.
    found = plan(_MAX_TAPS) or plan(1 << 62)
    assert found is not None
    return list(found[1])


class Channelizer:
    """Mix a channel at ``offset_hz`` to 0 Hz and decimate it to about
    ``target_rate`` (an integer factor of the input rate), keeping
    +/- ``pass_hz`` and removing everything beyond +/- ``stop_hz`` (60 dB).

    ``output_rate`` is the exact rate of what :meth:`process` returns. The
    decimation runs in stages (see :meth:`_design`): coarse ones whose
    filters only protect the final channel, then the sharp channel filter.
    """

    def __init__(
        self,
        rate: float,
        offset_hz: float = 0.0,
        pass_hz: float = 8e3,
        stop_hz: float = 12e3,
        target_rate: float = 48e3,
    ):
        rate = float(rate)
        if not rate > 0:
            raise ValueError(f"sample_rate must be positive, got {rate}")
        self.rate = rate
        stop_hz = min(float(stop_hz), 0.49 * rate)
        pass_hz = min(float(pass_hz), 0.8 * stop_hz)
        self.pass_hz = pass_hz
        self.stop_hz = stop_hz
        factor = decimation_factor(rate, target_rate, pass_hz + stop_hz)
        self.factor = factor
        self.output_rate = rate / factor
        self._nco = Nco(rate, offset_hz)
        self._stages = self._design(factor)

    def _design(self, factor: int) -> List[FirDecimator]:
        """Coarse stages, then the sharp channel filter (see
        :func:`_plan_stages`).

        The coarse stages' filters only protect the final channel from
        what would fold onto it, so they are short for their factors, and
        the first reads the full-rate samples just once. The sharp stage,
        at the lowest rate, removes everything beyond ``stop_hz``.
        """
        rate = self.rate
        stages: List[FirDecimator] = []
        for step, stop in _plan_stages(rate, factor, self.pass_hz, self.stop_hz):
            stages.append(FirDecimator(lowpass_taps(self.pass_hz, stop, rate), step))
            rate /= step
        return stages

    @property
    def offset_hz(self) -> float:
        return self._nco.frequency

    def set_offset(self, offset_hz: float) -> None:
        """Follow the channel to a new offset (phase-continuous)."""
        self._nco.set_frequency(offset_hz)

    def reset(self) -> None:
        self._nco.reset()
        for stage in self._stages:
            stage.reset()

    def process(self, iq: np.ndarray) -> np.ndarray:
        """The channel at 0 Hz, at ``output_rate`` (complex64)."""
        # Contiguous complex64 (a no-op for a device block): any layout or
        # type works, and the finite check below can view it as floats.
        x = np.ascontiguousarray(iq, dtype=np.complex64)
        if len(x) and not np.all(np.isfinite(x.view(np.float32))):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self._nco.process(x)
        for stage in self._stages:
            y = stage.process(y)
        return y.astype(np.complex64, copy=False)


class MovingAverage:
    """Streaming mean of the last ``length`` samples (real or complex)."""

    def __init__(self, length: int):
        self.length = max(1, int(length))
        self._history: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._history = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        n = len(x)
        length = self.length
        dtype = np.complex128 if np.iscomplexobj(x) else np.float64
        if length == 1:
            return x.astype(dtype)
        if self._history is None or self._history.dtype != dtype:
            self._history = np.zeros(length - 1, dtype=dtype)
        if n == 0:
            return np.zeros(0, dtype=dtype)
        buf = np.concatenate((self._history, x.astype(dtype, copy=False)))
        self._history = buf[len(buf) - (length - 1) :].copy()
        sums = np.cumsum(buf)
        sums = np.concatenate((np.zeros(1, dtype=dtype), sums))
        return (sums[length:] - sums[:-length]) / length


class Delay:
    """Streaming delay line of ``length`` samples."""

    def __init__(self, length: int):
        self.length = max(0, int(length))
        self._history: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._history = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.length == 0:
            return x
        if self._history is None or self._history.dtype != x.dtype:
            self._history = np.zeros(self.length, dtype=x.dtype)
        buf = np.concatenate((self._history, x))
        self._history = buf[len(buf) - self.length :].copy()
        return buf[: len(x)]


class FMDiscriminator:
    """Streaming FM discriminator: the instantaneous frequency (Hz) of a
    complex baseband, from the phase step between consecutive samples."""

    def __init__(self, rate: float):
        self.rate = float(rate)
        self._prev: complex = 0j

    def reset(self) -> None:
        self._prev = 0j

    def process(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.complex64)
        if not len(z):
            return np.zeros(0, dtype=np.float64)
        shifted = np.empty_like(z)
        shifted[0] = self._prev
        shifted[1:] = z[:-1]
        self._prev = complex(z[-1])
        step = np.angle(z * np.conj(shifted)).astype(np.float64)
        return step * (self.rate / (2.0 * np.pi))
