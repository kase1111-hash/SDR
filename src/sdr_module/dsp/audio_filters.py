"""
Listening filters for demodulated audio: the notch that removes a
heterodyne whistle, and the low cut that removes hum, rumble and the
sub-audible tones and DCS bits of narrow FM.

Everything streams (a block at a time gives exactly the audio of one long
block) and is plain NumPy (SciPy isn't required), cheap enough to run on
every display frame:

* :class:`SosFilter`: a cascade of second-order IIR sections (SciPy's
  ``sos`` layout) evaluated without a Python loop per sample: each
  section's poles become one-pole recursions, summed by a scaled
  cumulative sum (exact; numbers stay in range because the work is done
  in chunks).
* :func:`notch_sos` and :func:`highpass_sos`: a narrow notch (unity gain
  away from it), and an inverse-Chebyshev high-pass whose response is -3
  dB at the corner and at least ``stop_db`` down below about 0.8 times it.
* :class:`ToneFinder`: the strongest *steady* tone in the audio. A
  heterodyne stays on one frequency, to within a fraction of a hertz, for
  as long as it lasts; a voice's harmonics glide with its pitch, so they
  never hold still long enough to count.
* :class:`AudioNotch`: Off, Auto (the tone the finder hears, followed as
  it drifts) or a frequency set by hand, engaged and released over a few
  milliseconds so the audio never clicks.
* :class:`LowCut`: the high-pass at a chosen corner (or none), crossfaded
  when the corner changes.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: The notch's width (-3 dB), in Hz: wide enough that a tone set by hand
#: to the nearest hertz is still 35 dB down, narrow enough that nobody
#: hears the voice it takes away.
NOTCH_WIDTH_HZ = 60.0
#: Notch modes (see :class:`AudioNotch`).
NOTCH_MODES = ("off", "auto", "manual")


# --------------------------------------------------------------------------- #
# Designs
# --------------------------------------------------------------------------- #
def notch_sos(
    freq_hz: float, rate: float, width_hz: float = NOTCH_WIDTH_HZ
) -> np.ndarray:
    """A second-order notch at ``freq_hz``, ``width_hz`` wide at -3 dB, as
    one ``sos`` row ``[b0, b1, b2, 1, a1, a2]``.

    It is built from a second-order allpass, ``(1 + A(z)) / 2``: exactly
    zero at its frequency, exactly one at DC and at Nyquist, and the -3
    dB width is exactly ``width_hz`` whatever the frequency.
    """
    rate = float(rate)
    freq = min(max(float(freq_hz), 1.0), 0.499 * rate)
    width = min(max(float(width_hz), 0.1), 0.25 * rate)
    t = math.tan(math.pi * width / rate)
    a2 = (1.0 - t) / (1.0 + t)
    a1 = -(1.0 + a2) * math.cos(2.0 * math.pi * freq / rate)
    b0 = (1.0 + a2) / 2.0
    return np.array([[b0, a1, b0, 1.0, a1, a2]], dtype=np.float64)


def highpass_sos(
    corner_hz: float, rate: float, order: int = 8, stop_db: float = 40.0
) -> np.ndarray:
    """An inverse-Chebyshev (Chebyshev type II) high-pass, ``order // 2``
    ``sos`` rows: -3 dB at ``corner_hz``, flat above it, and at least
    ``stop_db`` down from DC up to its stopband edge (0.81 times the
    corner for order 8 and 40 dB). At 300 Hz that keeps every CTCSS tone
    up to 241.8 Hz 40 dB down (250.3 and 254.1 Hz about 30 dB) and a DCS
    bit stream with them, while voice from 350 Hz up is untouched.

    Designed in the analog domain and mapped with the bilinear transform,
    the stopband edge prewarped; gain 1 at Nyquist. ``order`` is even.
    """
    rate = float(rate)
    order = max(2, int(order) // 2 * 2)
    eps = 1.0 / math.sqrt(10.0 ** (float(stop_db) / 10.0) - 1.0)
    # The prototype's stopband starts at 1 rad/s and its -3 dB point lies
    # this much below it; scale so the -3 dB point lands on the corner.
    ratio = math.cosh(math.acosh(1.0 / eps) / order)
    stop_hz = min(max(float(corner_hz), 1.0), 0.45 * rate) / ratio
    mu = math.asinh(1.0 / eps) / order
    m = np.arange(-order + 1, order, 2)
    zeros = 1j / np.sin(m * np.pi / (2.0 * order))
    poles = -np.exp(1j * np.pi * m / (2.0 * order))
    poles = 1.0 / (math.sinh(mu) * poles.real + 1j * math.cosh(mu) * poles.imag)
    # Low-pass prototype to high-pass (s -> wo / s), then to z.
    wo = 2.0 * rate * math.tan(math.pi * stop_hz / rate)
    zeros_z = _bilinear(wo / zeros, rate)
    poles_z = _bilinear(wo / poles, rate)
    upper_p = sorted((complex(p) for p in poles_z if p.imag > 0), key=abs)
    upper_z = [complex(z) for z in zeros_z if z.imag > 0]
    rows = []
    for pole in upper_p:  # lowest Q first, each with its nearest zero pair
        distances = [abs(z - pole) for z in upper_z]
        zero = upper_z.pop(int(np.argmin(distances)))
        b = np.array([1.0, -2.0 * zero.real, abs(zero) ** 2])
        a = np.array([1.0, -2.0 * pole.real, abs(pole) ** 2])
        gain = (a[0] - a[1] + a[2]) / (b[0] - b[1] + b[2])  # 1 at Nyquist
        rows.append([*(gain * b), *a])
    return np.asarray(rows, dtype=np.float64)


def _bilinear(s: np.ndarray, rate: float) -> np.ndarray:
    k = 2.0 * rate
    return (k + s) / (k - s)


def sos_response(sos: np.ndarray, freqs_hz: Sequence[float], rate: float) -> np.ndarray:
    """The complex frequency response of ``sos`` at ``freqs_hz``."""
    z = np.exp(-2j * np.pi * np.asarray(freqs_hz, dtype=np.float64) / float(rate))
    h = np.ones(z.shape, dtype=np.complex128)
    for b0, b1, b2, a0, a1, a2 in np.asarray(sos, dtype=np.float64):
        h *= (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z)
    return h


# --------------------------------------------------------------------------- #
# The IIR engine
# --------------------------------------------------------------------------- #
class _Section:
    """One biquad: ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] -
    a2 y[n-2]``, a block at a time.

    The zeros (the FIR part) are three vectorised products. The initial
    conditions (the last two inputs and outputs) enter as a correction to
    the first two samples, and the poles then run from rest: complex
    conjugate poles as one complex one-pole recursion whose real part,
    scaled, is the output (partial fractions); real poles as two one-pole
    recursions in cascade. A one-pole recursion ``w[n] = p w[n-1] + u[n]``
    is ``p^n`` times the cumulative sum of ``p^-k u[k]``, computed in
    chunks short enough that ``p^-k`` stays far from overflowing.
    """

    _CHUNK = 2048

    def __init__(self, row: Sequence[float]) -> None:
        self.state = (0.0, 0.0, 0.0, 0.0)  # x[-1], x[-2], y[-1], y[-2]
        self.set(row)

    def set(self, row: Sequence[float]) -> None:
        b0, b1, b2, a0, a1, a2 = (float(v) for v in row)
        if a0 == 0.0:
            raise ValueError("a0 must not be zero")
        self.b = (b0 / a0, b1 / a0, b2 / a0)
        self.a = (a1 / a0, a2 / a0)
        a1, a2 = self.a
        disc = a1 * a1 - 4.0 * a2
        self.gain: Optional[complex] = None
        if disc < 0.0:
            pole = complex(-a1 / 2.0, math.sqrt(-disc) / 2.0)
            self.gain = pole / (pole - pole.conjugate())
            poles = [pole]
        else:
            root = math.sqrt(disc)
            poles = [complex((-a1 + root) / 2.0), complex((-a1 - root) / 2.0)]
        self.tables = [self._table(p) for p in poles]

    def _table(
        self, pole: complex
    ) -> Tuple[complex, int, Optional[np.ndarray], Optional[np.ndarray]]:
        """``(pole, chunk, p^k, p^-k)`` for k < chunk (no tables for a pole
        at zero: it passes its input)."""
        size = abs(pole)
        if size < 1e-9:
            return 0j, 0, None, None
        if size >= 1.0:
            raise ValueError("unstable filter: a pole on or outside the unit circle")
        decay = max(-math.log(size), 1e-12)
        chunk = max(16, min(self._CHUNK, int(300.0 / decay)))
        k = np.arange(chunk, dtype=np.float64)
        log = cmath.log(pole)  # (exp of a product: far cheaper than pole ** k)
        return pole, chunk, np.exp(k * log), np.exp(-k * log)

    @staticmethod
    def _pole(
        u: np.ndarray,
        table: Tuple[complex, int, Optional[np.ndarray], Optional[np.ndarray]],
    ) -> np.ndarray:
        pole, chunk, powers, inverse = table
        if powers is None or inverse is None:
            return u.astype(np.complex128)
        n = u.size
        if n <= chunk:
            return powers[:n] * np.cumsum(u * inverse[:n])
        out = np.empty(n, dtype=np.complex128)
        carry = 0j  # p times the last output of the chunk before
        for start in range(0, n, chunk):
            piece = u[start : start + chunk]
            m = piece.size
            w = powers[:m] * (carry + np.cumsum(piece * inverse[:m]))
            out[start : start + m] = w
            carry = pole * complex(w[-1])
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        b0, b1, b2 = self.b
        a1, a2 = self.a
        x1, x2, y1, y2 = self.state
        n = x.size
        v = b0 * x
        v[0] += b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        if n > 1:
            v[1:] += b1 * x[:-1]
            v[1] += b2 * x1 - a2 * y1
            if n > 2:
                v[2:] += b2 * x[:-2]
        if self.gain is not None:
            y = 2.0 * (self.gain * self._pole(v, self.tables[0])).real
        else:
            w = self._pole(v, self.tables[0]).real
            y = self._pole(w, self.tables[1]).real
        if n > 1:
            self.state = (float(x[-1]), float(x[-2]), float(y[-1]), float(y[-2]))
        else:
            self.state = (float(x[-1]), x1, float(y[-1]), y1)
        return y


class SosFilter:
    """A cascade of second-order sections (rows ``[b0, b1, b2, a0, a1,
    a2]``, SciPy's ``sos`` layout), streaming: :meth:`process` takes any
    block size and keeps the state (direct form) between blocks.

    :meth:`set_sos` changes the coefficients (same number of sections) and
    keeps the state, as a filter retuned between samples would: a small
    change is a small transient.
    """

    def __init__(self, sos: np.ndarray) -> None:
        rows = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        if rows.shape[1] != 6 or rows.shape[0] == 0:
            raise ValueError("sos must have rows of six coefficients")
        self._sections = [_Section(row) for row in rows]

    @property
    def sections(self) -> int:
        return len(self._sections)

    def set_sos(self, sos: np.ndarray) -> None:
        rows = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        if rows.shape != (len(self._sections), 6):
            raise ValueError("set_sos needs the same number of sections")
        for section, row in zip(self._sections, rows, strict=True):
            section.set(row)

    def reset(self) -> None:
        """Back to rest (a new stream)."""
        for section in self._sections:
            section.state = (0.0, 0.0, 0.0, 0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a block (float64 out)."""
        y = np.array(x, dtype=np.float64, copy=True).ravel()
        if y.size == 0:
            return y
        for section in self._sections:
            y = section.process(y)
        return y


# --------------------------------------------------------------------------- #
# Auto: finding the whistle
# --------------------------------------------------------------------------- #
class ToneFinder:
    """The strongest steady tone in audio, followed as it drifts.

    Every ``hop`` (about 43 ms at 48 kHz) a Hann-windowed spectrum of the
    last ~85 ms is searched (4096 points at 48 kHz, 11.7 Hz a bin) for
    peaks standing ``PROMINENCE_DB`` above the spectrum around them, in
    the band set with :meth:`set_band`. Each peak's frequency is refined
    to a small fraction of a hertz from how its phase turned since the
    last spectrum (a steady tone advances exactly by its frequency).

    A tone counts once a peak has been within ``NEAR_HZ`` of the same
    frequency in ``NEED`` of the last ``FRAMES`` spectra (about half a
    second), its refined frequencies no more than ``2 x SPREAD_HZ`` apart.
    A voice's harmonics glide with its pitch and come and go with its
    syllables, so they never qualify. Once found, the tone is followed (the
    median of its last five refined frequencies) while it stands
    ``KEEP_DB`` above its surroundings, and let go once it has been
    missing from three quarters of the last ``RELEASE_FRAMES`` spectra
    (about a second: a loud word over it doesn't release it).
    """

    SIZE_S = 0.085
    FRAMES = 14
    NEED = 12
    NEAR_HZ = 3.0
    SPREAD_HZ = 1.5
    PROMINENCE_DB = 12.0
    KEEP_DB = 6.0
    RELEASE_FRAMES = 24
    #: Peaks looked at per spectrum (the strongest), and the bins around a
    #: peak that make its surroundings (its own main lobe left out).
    _PEAKS = 6
    _AROUND = 13
    _LOBE = 2

    def __init__(self, rate: float, low_hz: float = 200.0, high_hz: float = 4500.0):
        self.rate = float(rate)
        size = 1 << max(8, int(round(math.log2(self.SIZE_S * self.rate))))
        self._size = size
        self._hop = size // 2
        self._window = np.hanning(size)
        self._bin_hz = self.rate / size
        self._prominence = 10.0 ** (self.PROMINENCE_DB / 10.0)
        self._keep = 10.0 ** (self.KEEP_DB / 10.0)
        self._low = float(low_hz)
        self._high = float(high_hz)
        self._skip: Optional[Tuple[float, float]] = None
        self.reset()

    # -- settings ------------------------------------------------------- #
    def set_band(
        self,
        low_hz: float,
        high_hz: float,
        skip: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Look for tones between ``low_hz`` and ``high_hz``, leaving out
        ``skip`` (``(low, high)``, e.g. the CW note); a tone being
        followed outside them is let go."""
        self._low, self._high = float(low_hz), float(high_hz)
        self._skip = None if skip is None else (float(skip[0]), float(skip[1]))
        if self._freq is not None and not self._allowed(self._freq):
            self._release()

    def _allowed(self, freq: float) -> bool:
        if not self._low <= freq <= self._high:
            return False
        skip = self._skip
        return skip is None or not skip[0] <= freq <= skip[1]

    # -- state ---------------------------------------------------------- #
    def reset(self) -> None:
        """Forget everything, the tone found too."""
        self._freq: Optional[float] = None
        self._history: List[List[Tuple[float, float]]] = []
        self.reset_stream()

    def reset_stream(self) -> None:
        """A new stream of the same channel (after a gap): the samples are
        dropped, the tone found is kept (it is checked again at once)."""
        self._tail = np.zeros(self._size, dtype=np.float64)
        self._seen = 0
        self._next = self._size
        self._prev: Optional[np.ndarray] = None
        self._recent: List[float] = []
        self._misses: List[bool] = []

    def _release(self) -> None:
        self._freq = None
        self._recent = []
        self._misses = []

    @property
    def frequency_hz(self) -> Optional[float]:
        """The tone being followed (Hz), or None."""
        return self._freq

    def until_next_frame(self) -> int:
        """Samples still to come before the next spectrum is taken."""
        return max(1, self._next - self._seen)

    # -- work ----------------------------------------------------------- #
    def process(self, audio: np.ndarray) -> Optional[float]:
        """Feed audio; returns :attr:`frequency_hz`."""
        x = np.asarray(audio, dtype=np.float64).ravel()
        pos = 0
        while pos < x.size:
            take = min(x.size - pos, self._next - self._seen)
            piece = x[pos : pos + take]
            if take >= self._size:
                self._tail = piece[-self._size :].copy()
            else:
                self._tail = np.concatenate((self._tail[take:], piece))
            pos += take
            self._seen += take
            if self._seen >= self._next:
                self._spectrum()
                self._next += self._hop
        return self._freq

    def _refine(self, spectrum: np.ndarray, k: int) -> float:
        """Bin ``k``'s tone frequency from its phase advance over a hop."""
        prev = self._prev
        if prev is None:
            return k * self._bin_hz
        turn = float(np.angle(spectrum[k] * np.conj(prev[k])))
        turn -= 2.0 * math.pi * k * self._hop / self._size
        turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
        return (k + turn * self._size / (2.0 * math.pi * self._hop)) * self._bin_hz

    def _floors(self, power: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """The level around each of ``bins``: the median of the bins
        ``_AROUND`` either side, its own main lobe left out."""
        lobe, around = self._LOBE, self._AROUND
        offsets = np.concatenate(
            (np.arange(-around, -lobe), np.arange(lobe + 1, around + 1))
        )
        return np.median(power[bins[:, None] + offsets[None, :]], axis=1)

    def _spectrum(self) -> None:
        spectrum = np.fft.rfft(self._tail * self._window)
        power = spectrum.real**2 + spectrum.imag**2
        guard = self._AROUND + 1
        first = max(guard, int(math.ceil(self._low / self._bin_hz)))
        last = min(power.size - guard - 1, int(self._high / self._bin_hz))
        peaks: List[Tuple[float, float]] = []
        if last > first:
            middle = power[first : last + 1]
            local = (middle >= power[first - 1 : last]) & (
                middle > power[first + 1 : last + 2]
            )
            skip = self._skip
            if skip is not None:
                hz = np.arange(first, last + 1) * self._bin_hz
                local &= (hz < skip[0]) | (hz > skip[1])
            index = np.flatnonzero(local)
            # Only the strongest few are judged.
            strongest = index[np.argsort(middle[index])[::-1][: self._PEAKS]] + first
            if strongest.size:
                floors = self._floors(power, strongest)
                for k, floor in zip(strongest, floors, strict=True):
                    level = float(power[k])
                    if level > 1e-9 and level >= self._prominence * float(floor):
                        peaks.append((self._refine(spectrum, int(k)), level))
        self._history.append(peaks)
        del self._history[: -self.FRAMES]
        if self._freq is None:
            self._find(peaks)
        else:
            self._follow(spectrum, power)
        self._prev = spectrum

    def _find(self, peaks: List[Tuple[float, float]]) -> None:
        for freq, _level in peaks:  # strongest first
            if not self._allowed(freq):
                continue
            near = [
                f
                for frame in self._history
                for f, _p in frame
                if abs(f - freq) <= self.NEAR_HZ
            ]
            if len(near) >= self.NEED and max(near) - min(near) <= 2.0 * self.SPREAD_HZ:
                self._recent = near[-5:]
                self._freq = float(np.median(self._recent))
                self._misses = []
                return

    def _follow(self, spectrum: np.ndarray, power: np.ndarray) -> None:
        freq = self._freq
        assert freq is not None
        k = int(round(freq / self._bin_hz))
        there: Optional[float] = None
        guard = self._AROUND + 1
        candidates = [
            kk
            for kk in (k - 1, k, k + 1)
            if guard <= kk < power.size - guard
            and power[kk] >= power[kk - 1]
            and power[kk] >= power[kk + 1]
            and power[kk] > 1e-9
        ]
        if candidates:
            floors = self._floors(power, np.asarray(candidates))
            for kk, floor in zip(candidates, floors, strict=True):
                if power[kk] >= self._keep * float(floor):
                    f = self._refine(spectrum, kk)
                    if abs(f - freq) <= self.NEAR_HZ:
                        there = f
        self._misses.append(there is None)
        del self._misses[: -self.RELEASE_FRAMES]
        if there is not None:
            self._recent = (self._recent + [there])[-5:]
            self._freq = float(np.median(self._recent))
        elif (
            len(self._misses) >= self.RELEASE_FRAMES
            and sum(self._misses) >= 0.75 * self.RELEASE_FRAMES
        ):
            self._release()


# --------------------------------------------------------------------------- #
# The notch
# --------------------------------------------------------------------------- #
class AudioNotch:
    """The audio notch: ``mode`` ``"off"``, ``"auto"`` (the tone the
    :attr:`finder` hears) or ``"manual"`` (``manual_hz``), read on every
    block, so changing them takes effect at once.

    It engages and lets go over ``RAMP_S`` (the notched audio is faded in
    or out against the audio as it was). A small move of its frequency (a
    drifting whistle, a step of the frequency box) retunes it in place; a
    jump fades it out and in again at the new frequency. A notch tuned
    afresh first runs over the last ``WARM_S`` of audio, so it starts in
    step with it. Off (and Auto with nothing found) the audio passes
    untouched.

    :meth:`process` gives the same audio whatever the block sizes.
    """

    RAMP_S = 0.02
    RETUNE_HZ = 5.0
    WARM_S = 0.03
    #: Moves smaller than this aren't followed (the notch is still 55 dB
    #: deep there; a new design costs more than the move is worth).
    STILL_HZ = 0.02

    def __init__(self, rate: float, width_hz: float = NOTCH_WIDTH_HZ) -> None:
        self.rate = float(rate)
        self.width_hz = float(width_hz)
        self.mode = "off"
        self.manual_hz = 1000.0
        self.finder = ToneFinder(self.rate)
        self._step = 1.0 / max(self.RAMP_S * self.rate, 1.0)
        self._warm = max(1, int(round(self.WARM_S * self.rate)))
        self._filter: Optional[SosFilter] = None
        self._freq: Optional[float] = None  # where the filter is tuned
        self._mix = 0.0  # how much of the notched audio is heard (0-1)
        self._recent = np.zeros(0, dtype=np.float64)  # the last WARM_S in
        self._finding = False

    @property
    def frequency_hz(self) -> Optional[float]:
        """The frequency being notched: the one set by hand, Auto's tone,
        or None (off, or Auto with nothing found)."""
        return self._wanted()

    def _wanted(self) -> Optional[float]:
        if self.mode == "manual":
            return self._manual()
        if self.mode == "auto":
            return self.finder.frequency_hz
        return None

    def _manual(self) -> Optional[float]:
        try:
            hz = float(self.manual_hz)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(hz):
            return None
        return min(max(hz, 20.0), 0.45 * self.rate)

    def reset(self) -> None:
        """Start afresh (another channel or mode): Auto searches anew."""
        self.finder.reset()
        self._finding = False
        self.reset_stream()

    def search_anew(self) -> None:
        """Another station tuned: Auto forgets the tone it found (the notch
        fades out, as when a tone ends) and searches afresh."""
        self.finder.reset()

    def reset_stream(self) -> None:
        """A new stream of the same channel (the squelch closed and opened
        again): the filter starts from rest, Auto keeps the tone it found."""
        self.finder.reset_stream()
        self._recent = np.zeros(0, dtype=np.float64)
        self._filter = None
        self._freq = None
        self._mix = 0.0
        if self._wanted() is not None:
            self._mix = 1.0  # (the audio itself fades in after a gap)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Notch a block of audio (float32 out)."""
        x = np.asarray(audio, dtype=np.float32).ravel()
        if x.size == 0:
            return x
        if self.mode != "auto" and self._finding:
            self.finder.reset()
            self._finding = False
        if self.mode != "auto":
            return self._apply(x, self._wanted())
        # Auto: a spectrum is taken at fixed sample counts, and what it
        # finds applies from there on (whatever the block sizes).
        self._finding = True
        out = np.empty_like(x)
        pos = 0
        while pos < x.size:
            take = min(x.size - pos, self.finder.until_next_frame())
            piece = x[pos : pos + take]
            out[pos : pos + take] = self._apply(piece, self.finder.frequency_hz)
            self.finder.process(piece)
            pos += take
        return out

    def _apply(self, x: np.ndarray, want: Optional[float]) -> np.ndarray:
        if want is not None and self._freq is not None:
            if abs(want - self._freq) > self.RETUNE_HZ and self._mix > 0.0:
                # A jump: out at the old frequency, then in at the new one.
                count = min(x.size, int(math.ceil(self._mix / self._step - 1e-9)))
                head = self._ramp(x[:count], None)
                if count == x.size:
                    return head
                return np.concatenate((head, self._apply(x[count:], want)))
        return self._ramp(x, want)

    def _ramp(self, x: np.ndarray, want: Optional[float]) -> np.ndarray:
        """``x`` notched at ``want`` (None: on the way out), fading."""
        target = 0.0 if want is None else 1.0
        if want is not None and (
            self._freq is None or abs(want - self._freq) > self.STILL_HZ
        ):
            fresh = (
                self._filter is None
                or self._freq is None
                or abs(want - self._freq) > self.RETUNE_HZ
            )
            self._tune(want, fresh)
        n = x.size
        if self._filter is None or (self._mix <= 0.0 and target <= 0.0):
            # Silent: the audio as it was (the filter rests until needed).
            # (No filter yet: a gap meant the notch to engage at once, but
            # nothing is wanted now. The audio fades in after a gap anyway.)
            self._filter = None
            self._freq = None
            self._mix = 0.0
            self._remember(x)
            return x
        notched = self._filter.process(x)
        if target == self._mix:
            mix: np.ndarray = np.full(n, self._mix)
        elif target > self._mix:
            mix = np.minimum(self._mix + self._step * np.arange(1, n + 1), target)
        else:
            mix = np.maximum(self._mix - self._step * np.arange(1, n + 1), target)
        self._mix = float(mix[-1])
        out = x + mix * (notched - x)
        self._remember(x)
        return out.astype(np.float32)

    def _tune(self, freq: float, fresh: bool) -> None:
        sos = notch_sos(freq, self.rate, self.width_hz)
        if fresh or self._filter is None:
            self._filter = SosFilter(sos)
            if self._recent.size:
                self._filter.process(self._recent)  # in step with the audio
        else:
            self._filter.set_sos(sos)
        self._freq = freq

    def _remember(self, x: np.ndarray) -> None:
        if x.size >= self._warm:
            self._recent = x[-self._warm :].astype(np.float64)
        else:
            self._recent = np.concatenate((self._recent, x))[-self._warm :]


# --------------------------------------------------------------------------- #
# The low cut
# --------------------------------------------------------------------------- #
class LowCut:
    """A high-pass of the audio (:func:`highpass_sos`) at a corner chosen
    on every block, or none: :meth:`process` ``(audio, corner_hz)``.

    A new corner (or switching it on or off) crossfades from the audio as
    it was to the new over ``FADE_S``, the new filter having first run
    over the last ``WARM_S`` of audio (in step with it): no click.
    Without a corner the audio passes untouched.
    """

    FADE_S = 0.01
    WARM_S = 0.05

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self._fade = max(1, int(round(self.FADE_S * self.rate)))
        self._warm = max(1, int(round(self.WARM_S * self.rate)))
        self.reset()

    @property
    def corner_hz(self) -> Optional[float]:
        """The corner in effect (Hz), or None."""
        return self._corner

    def reset(self) -> None:
        """Back to rest (a new stream; the corner stays)."""
        self._recent = np.zeros(0, dtype=np.float64)
        corner = getattr(self, "_corner", None)
        self._corner: Optional[float] = None
        self._filter: Optional[SosFilter] = None
        if corner is not None:
            self._corner = corner
            self._filter = SosFilter(highpass_sos(corner, self.rate))

    def process(self, audio: np.ndarray, corner_hz: Optional[float]) -> np.ndarray:
        """``audio`` high-passed at ``corner_hz`` (None or 0: as it is)."""
        x = np.asarray(audio, dtype=np.float32).ravel()
        if x.size == 0:
            return x
        corner: Optional[float] = None
        if corner_hz is not None and math.isfinite(corner_hz) and corner_hz > 0:
            corner = float(corner_hz)
        if corner != self._corner:
            old = (
                x.astype(np.float64)
                if self._filter is None
                else self._filter.process(x)
            )
            self._filter = None
            if corner is not None:
                self._filter = SosFilter(highpass_sos(corner, self.rate))
                if self._recent.size:
                    self._filter.process(self._recent)  # in step with the audio
            new = (
                x.astype(np.float64)
                if self._filter is None
                else self._filter.process(x)
            )
            m = min(x.size, self._fade)
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
            new[:m] = old[:m] + ramp * (new[:m] - old[:m])
            self._corner = corner
            out = new.astype(np.float32)
        elif self._filter is None:
            out = x
        else:
            out = self._filter.process(x).astype(np.float32)
        if x.size >= self._warm:
            self._recent = x[-self._warm :].astype(np.float64)
        else:
            self._recent = np.concatenate((self._recent, x))[-self._warm :]
        return out
