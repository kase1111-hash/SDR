"""
The spectrum of a whole display frame.

The display loop gets about 30 frames of I/Q samples a second (80,000
samples a frame at 2.4 MS/s, 670,000 at 20 MS/s). A single FFT of the newest
samples sees a few percent of them, so a burst earlier in the frame never
shows. :class:`FrameSpectrumAnalyzer` cuts the whole frame into FFT-size
segments, transforms them in one vectorised FFT and gives, per bin:

* the mean power over the segments: the spectrum trace. A steady signal
  reads the same as in one FFT, and the noise is much steadier;
* the strongest power over the segments: a waterfall row, where a burst in
  any part of the frame shows at its full level.

Samples left over at the end of a frame (less than one FFT) are analysed
with the next frame, so every sample counts once. A frame too short for one
FFT (a large FFT, a slow recording) is completed with the newest samples of
the frames before it.

With ``max_samples`` set, a frame longer than that (every frame at 20 MS/s,
or a catch-up frame after a stall, several times longer) is analysed in
evenly spread segments, the newest included, about ``max_samples`` of
them: the cost per frame stays bounded. The mean reads the same; a burst
shows in the maximum unless it is shorter than the gap between segments.

Levels are dBFS: a full-scale complex tone peaks at 0 dB whatever the FFT
size and window (the window's coherent gain is divided out). The noise level
per bin falls 3 dB per doubling of the FFT size (each bin is half as wide)
and depends on the window's equivalent noise bandwidth (:func:`enbw_bins`);
:func:`rbw_hz` gives the resolution bandwidth.

Large frames are transformed by two threads (NumPy's FFT releases the GIL),
in buffers reused from frame to frame: about 0.6 ms a frame at 2.4 MS/s and
4 ms at 20 MS/s on a laptop. The caller does the helper's half itself when
the helper hasn't started on it, so a busy computer (where the helper may
wait a while for a processor) never keeps it waiting.
"""

from __future__ import annotations

import inspect
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

#: FFT sizes offered for the display (points).
FFT_SIZES: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_FFT_SIZE = 2048

#: Windows offered for the display, by key.
WINDOWS: Tuple[str, ...] = ("hann", "blackman-harris", "flat-top")
DEFAULT_WINDOW = "hann"
WINDOW_LABELS: Dict[str, str] = {
    "hann": "Hann",
    "blackman-harris": "Blackman-Harris",
    "flat-top": "Flat-top",
}
#: What each window is for (menus and tooltips).
WINDOW_HINTS: Dict[str, str] = {
    "hann": "Good all-round choice: narrow peaks, low leakage",
    "blackman-harris": (
        "Keeps a strong signal's leakage 92 dB down, so a weak signal right "
        "next to it shows (slightly wider peaks)"
    ),
    "flat-top": (
        "Reads a carrier's level exactly wherever it falls between bins "
        "(much wider peaks)"
    ),
}

# Cosine-sum coefficients (periodic windows, as used for spectral analysis).
_COEFFICIENTS: Dict[str, Tuple[float, ...]] = {
    "hann": (0.5, 0.5),
    # 4-term minimum Blackman-Harris (-92 dB sidelobes).
    "blackman-harris": (0.35875, 0.48829, 0.14128, 0.01168),
    # Flat top (as MATLAB's flattopwin): scalloping loss under 0.01 dB.
    "flat-top": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
}

#: Frames at least this many samples are split over two threads.
_THREAD_MIN_SAMPLES = 1 << 16
#: Power floor before taking logs (-240 dBFS).
_POWER_FLOOR = 1e-24


def _fft_takes_out() -> bool:
    """Whether ``np.fft.fft`` writes into an ``out`` buffer (NumPy 2.0 and
    later; NumPy 1.x, which the package still allows, has no ``out``)."""
    try:
        return "out" in inspect.signature(np.fft.fft).parameters
    except (TypeError, ValueError):  # pragma: no cover - no signature
        return False


def normalize_window(name: Optional[str]) -> str:
    """A window key from :data:`WINDOWS` (unknown names: the default)."""
    key = str(name or "").strip().lower().replace("_", "-").replace(" ", "-")
    if key in ("flattop", "flat"):
        key = "flat-top"
    if key in ("blackmanharris", "bh"):
        key = "blackman-harris"
    return key if key in _COEFFICIENTS else DEFAULT_WINDOW


def normalize_fft_size(size: object) -> int:
    """The nearest of :data:`FFT_SIZES` to ``size`` (the default if not a
    number)."""
    try:
        value = float(size)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):  # (OverflowError: a huge int)
        return DEFAULT_FFT_SIZE
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_FFT_SIZE
    return min(FFT_SIZES, key=lambda s: abs(math.log2(s) - math.log2(value)))


def make_window(name: str, size: int) -> np.ndarray:
    """The periodic ``name`` window of ``size`` points (float32)."""
    coeffs = _COEFFICIENTS[normalize_window(name)]
    phase = 2.0 * np.pi * np.arange(int(size), dtype=np.float64) / int(size)
    window = np.zeros(int(size), dtype=np.float64)
    for k, a in enumerate(coeffs):
        window += (-1.0) ** k * a * np.cos(k * phase)
    return window.astype(np.float32)


_ENBW_CACHE: Dict[str, float] = {}


def enbw_bins(name: str) -> float:
    """Equivalent noise bandwidth of window ``name``, in FFT bins (Hann
    1.50, Blackman-Harris 2.00, flat-top 3.77)."""
    key = normalize_window(name)
    if key not in _ENBW_CACHE:
        w = make_window(key, 4096).astype(np.float64)
        _ENBW_CACHE[key] = float(len(w) * np.sum(w * w) / np.sum(w) ** 2)
    return _ENBW_CACHE[key]


def rbw_hz(sample_rate: float, size: int, window: str = DEFAULT_WINDOW) -> float:
    """Resolution bandwidth (Hz): the window's noise bandwidth in bins times
    the bin width."""
    if size <= 0 or sample_rate <= 0:
        return 0.0
    return enbw_bins(window) * float(sample_rate) / int(size)


@dataclass(frozen=True)
class FrameSpectrum:
    """One frame's spectrum (dBFS per bin, ``fftshift`` order: the lowest
    frequency first, the center at ``size // 2``)."""

    mean_db: np.ndarray
    max_db: np.ndarray
    #: FFT segments the frame gave (1 for a frame shorter than one FFT).
    segments: int

    @property
    def size(self) -> int:
        return int(self.mean_db.shape[0])

    @property
    def noise_floor_db(self) -> float:
        """The noise floor (dBFS): the mean noise power in one bin, for
        an SNR readout. The median bin of :attr:`mean_db` (so the signals
        in the frame don't count), corrected for how far the median of a
        mean of ``segments`` FFTs' noise sits below its mean (1.6 dB for
        one FFT, almost nothing for dozens). NaN when there is no data.

        (The median of :attr:`max_db`, the strongest of ``segments`` FFTs
        in each bin, sits 2 to 8 dB above the noise.)"""
        values = np.asarray(self.mean_db)
        values = values[:: max(1, len(values) // 2048)]  # (plenty for a median)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        k = max(1, int(self.segments))
        # The median of a mean of k exponential (noise power) values over
        # their mean: ln 2 for one, else Choi's approximation of the median
        # of a gamma(k) distribution.
        ratio = math.log(2.0) if k == 1 else 1.0 - 1.0 / (3 * k) + 8.0 / (405 * k * k)
        return float(np.median(values)) - 10.0 * math.log10(ratio)


class _Claim:
    """A piece of work that exactly one of two threads does: the first to
    :meth:`take` it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._taken = False

    def take(self) -> bool:
        with self._lock:
            taken, self._taken = self._taken, True
        return not taken


class FrameSpectrumAnalyzer:
    """Spectrum of every sample of each display frame (see the module).

    Not thread-safe: one caller (the display loop) at a time.
    """

    def __init__(
        self,
        size: int = DEFAULT_FFT_SIZE,
        window: str = DEFAULT_WINDOW,
        threads: int = 2,
        max_samples: Optional[int] = None,
    ):
        self._size = normalize_fft_size(size)
        self._window_name = normalize_window(window)
        self._threads = max(1, int(threads))
        self._max_samples: Optional[int] = None
        self.max_samples = max_samples
        self._fft_out = _fft_takes_out()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self._window = make_window(self._window_name, self._size)
        self._scale = 1.0 / float(np.sum(self._window, dtype=np.float64)) ** 2
        # Samples not analysed yet (fewer than one FFT), and the newest
        # ``size`` samples seen (to complete a short frame).
        self._carry = np.zeros(0, dtype=np.complex64)
        self._recent = np.zeros(0, dtype=np.complex64)
        self._seen = 0
        # Work buffers (rows x size), grown as frames need.
        self._rows = 0
        self._windowed: Optional[np.ndarray] = None
        self._spectra: Optional[np.ndarray] = None
        self._power: Optional[np.ndarray] = None
        self._scratch: Optional[np.ndarray] = None

    # -- settings ----------------------------------------------------------
    @property
    def size(self) -> int:
        return self._size

    @property
    def window(self) -> str:
        return self._window_name

    @property
    def max_samples(self) -> Optional[int]:
        """About the most samples one frame is analysed in (evenly spread
        FFT segments of a longer frame); None: every sample."""
        return self._max_samples

    @max_samples.setter
    def max_samples(self, value: Optional[int]) -> None:
        self._max_samples = None if value is None else max(1, int(value))

    def configure(
        self, size: Optional[int] = None, window: Optional[str] = None
    ) -> bool:
        """Use another FFT size and/or window. True when something changed
        (the samples kept for the next frame are dropped then)."""
        new_size = self._size if size is None else normalize_fft_size(size)
        new_window = self._window_name if window is None else normalize_window(window)
        if (new_size, new_window) == (self._size, self._window_name):
            return False
        self._size, self._window_name = new_size, new_window
        self._window = make_window(new_window, new_size)
        self._scale = 1.0 / float(np.sum(self._window, dtype=np.float64)) ** 2
        self._rows = 0
        self._windowed = self._spectra = self._power = self._scratch = None
        self.reset()
        return True

    def rbw(self, sample_rate: float) -> float:
        """Resolution bandwidth (Hz) at ``sample_rate``."""
        return rbw_hz(sample_rate, self._size, self._window_name)

    def reset(self) -> None:
        """Forget the samples of earlier frames (after a retune or a rate
        change: they belong to another frequency)."""
        self._carry = np.zeros(0, dtype=np.complex64)
        self._recent = np.zeros(0, dtype=np.complex64)
        self._seen = 0

    def close(self) -> None:
        """Stop the helper thread (it is started again when needed)."""
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)

    # -- analysis ----------------------------------------------------------
    def process(self, samples: np.ndarray) -> Optional[FrameSpectrum]:
        """The spectrum of ``samples`` (one frame), or None while fewer than
        one FFT of samples has arrived since :meth:`reset`."""
        x = np.asarray(samples)
        if x.dtype != np.complex64:
            x = x.astype(np.complex64)
        x = x.ravel()
        size = self._size
        self._seen += len(x)
        self._remember(x)
        if len(self._carry):
            data = np.concatenate((self._carry, x))
        else:
            data = x
        count = len(data) // size
        if count == 0:
            # Too short for one FFT of new samples: the newest FFT's worth
            # (with samples already shown), the rest kept for later.
            self._carry = np.array(data, dtype=np.complex64, copy=True)
            if len(self._recent) < size:
                return None
            segments = self._recent.reshape(1, size)
        else:
            used = count * size
            self._carry = np.array(data[used:], dtype=np.complex64, copy=True)
            segments = data[:used].reshape(count, size)
            limit = self._max_samples
            if limit is not None and used > limit:
                # Evenly spread segments, the newest one included.
                step = -(-count // max(1, limit // size))
                segments = segments[(count - 1) % step :: step]
        power = self._power_rows(segments)
        mean = power.mean(axis=0) if len(power) > 1 else power[0]
        peak = power.max(axis=0) if len(power) > 1 else power[0]
        return FrameSpectrum(self._to_db(mean), self._to_db(peak), len(power))

    def _remember(self, x: np.ndarray) -> None:
        size = self._size
        if len(x) >= size:
            self._recent = np.array(x[-size:], dtype=np.complex64, copy=True)
        else:
            self._recent = np.concatenate((self._recent, x))[-size:]

    def _ensure_buffers(self, rows: int) -> None:
        if self._windowed is not None and rows <= self._rows <= 4 * rows + 8:
            return
        capacity = max(rows, int(rows * 1.25) + 1)
        shape = (capacity, self._size)
        self._windowed = np.empty(shape, dtype=np.complex64)
        self._spectra = np.empty(shape, dtype=np.complex64)
        self._power = np.empty(shape, dtype=np.float32)
        self._scratch = np.empty(shape, dtype=np.float32)
        self._rows = capacity

    def _power_rows(self, segments: np.ndarray) -> np.ndarray:
        """|FFT|^2 of each windowed segment (rows x size, float32)."""
        rows = segments.shape[0]
        self._ensure_buffers(rows)
        assert self._power is not None
        split = (
            self._threads > 1 and rows >= 2 and rows * self._size >= _THREAD_MIN_SAMPLES
        )
        if split:
            # The helper takes the first half, unless this thread gets to it
            # first: on a busy computer the helper may not run for a while,
            # and waiting for it cost more than doing the half here.
            half = rows // 2
            claim = _Claim()
            future = self._submit(segments, 0, half, claim)
            self._transform(segments, half, rows)
            if claim.take():
                self._transform(segments, 0, half)
            elif future is not None:
                future.result()
        else:
            self._transform(segments, 0, rows)
        return self._power[:rows]

    def _submit(
        self, segments: np.ndarray, start: int, stop: int, claim: "_Claim"
    ) -> Optional[Future]:
        with self._pool_lock:
            if self._pool is None:
                try:
                    self._pool = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="frame-fft"
                    )
                except RuntimeError:  # pragma: no cover - interpreter exiting
                    return None
            pool = self._pool
        try:
            return pool.submit(self._transform_claimed, segments, start, stop, claim)
        except RuntimeError:  # pragma: no cover - pool shut down meanwhile
            return None

    def _transform_claimed(
        self, segments: np.ndarray, start: int, stop: int, claim: "_Claim"
    ) -> None:
        """The helper's part, unless the caller took it over meanwhile."""
        if claim.take():
            self._transform(segments, start, stop)

    def _transform(self, segments: np.ndarray, start: int, stop: int) -> None:
        if stop <= start:
            return
        assert self._windowed is not None and self._spectra is not None
        assert self._power is not None and self._scratch is not None
        w = self._windowed[start:stop]
        f = self._spectra[start:stop]
        p = self._power[start:stop]
        s = self._scratch[start:stop]
        np.multiply(segments[start:stop], self._window, out=w)
        if self._fft_out:
            np.fft.fft(w, axis=1, out=f)
        else:  # NumPy 1.x: no out argument (and the FFT is complex128)
            f[...] = np.fft.fft(w, axis=1)
        np.multiply(f.real, f.real, out=p)
        np.multiply(f.imag, f.imag, out=s)
        np.add(p, s, out=p)

    def _to_db(self, power: np.ndarray) -> np.ndarray:
        """Power per bin -> dBFS, in fftshift order (float32)."""
        scaled = np.fft.fftshift(power.astype(np.float64) * self._scale)
        return (10.0 * np.log10(np.maximum(scaled, _POWER_FLOOR))).astype(np.float32)


__all__ = [
    "DEFAULT_FFT_SIZE",
    "DEFAULT_WINDOW",
    "FFT_SIZES",
    "FrameSpectrum",
    "FrameSpectrumAnalyzer",
    "WINDOWS",
    "WINDOW_HINTS",
    "WINDOW_LABELS",
    "enbw_bins",
    "make_window",
    "normalize_fft_size",
    "normalize_window",
    "rbw_hz",
]
