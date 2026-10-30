"""
Signals in view: the signals of a spectrum, and of a whole recording.

:func:`find_signals` lists the signals of one (averaged) spectrum: each
one's frequency, peak level, SNR against the local noise floor and width,
as the Signal tab's list and the spectrum's peak markers show them.
:class:`SignalSurvey` runs it over a recording, slice by slice, and keeps
each signal once, with when it was first and last heard
(File > Find Signals in Recording).

Levels are dBFS per bin of a Hann-windowed FFT (the main display's scale).
No Qt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = [
    "SignalPeak",
    "find_signals",
    "local_floor_db",
    "SurveySignal",
    "SignalSurvey",
    "survey_nfft",
]

#: Bins either side of 0 Hz left out (most receivers' DC spike).
DC_GUARD_BINS = 2


@dataclass
class SignalPeak:
    """One signal of a spectrum."""

    frequency_hz: float  # absolute: the power centroid of its core
    peak_dbfs: float
    snr_db: float
    width_hz: float  # contiguous bins within 20 dB of the peak
    offset_hz: float = 0.0  # from the spectrum's centre


def local_floor_db(power_db: np.ndarray, fraction: float = 1.0 / 6.0) -> np.ndarray:
    """The noise floor under each bin (dB): a low quantile of the bins
    around it (``fraction`` of the span), corrected to the mean of noise.
    Signals stand above it; a sloping floor (a receiver's roll-off) is
    followed. Large spectra are pooled to at most 1024 bins first (in
    power), so it costs the same at any FFT size."""
    p = np.asarray(power_db, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p.copy()
    pool = max(1, n // 1024)
    lin = np.power(10.0, p / 10.0)
    if pool > 1:
        m = (n // pool) * pool
        pooled = lin[:m].reshape(-1, pool).mean(axis=1)
    else:
        pooled = lin
    q = len(pooled)
    width = int(max(9, min(q, round(q * fraction)))) | 1
    pad = width // 2
    padded = np.concatenate((pooled[:pad][::-1], pooled, pooled[-pad:][::-1]))
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)[:q]
    idx = np.arange(0, q, 4)
    low, mid = np.percentile(windows[idx], [20, 50], axis=1)
    # The 20th percentile of noise as a fraction of its mean depends on how
    # much the bins were averaged (0.22 for one FFT, near 1 for many), which
    # the spread of the bins tells: their 20th over their 50th percentile.
    spread = float(np.median(low / np.maximum(mid, 1e-300)))
    ratio = _q20_of_mean(_averaged_frames(spread))
    floor = np.interp(np.arange(q), idx, low) / ratio
    centres = (np.arange(q) + 0.5) * pool - 0.5
    full = np.interp(np.arange(n), centres, floor)
    return 10.0 * np.log10(np.maximum(full, 1e-30))


def _q20_of_mean(k: float) -> float:
    """The 20th percentile of the mean of ``k`` exponential (noise)
    values, over their mean."""
    from .classifiers import _gamma_quantile

    return max(1e-3, float(_gamma_quantile(0.2, k)))


def _averaged_frames(spread: float) -> float:
    """How many exponential values a noise bin is the mean of, from the
    ratio of its 20th to its 50th percentile (0.32 for one, near 1 for
    many)."""
    from .classifiers import _gamma_quantile

    if not math.isfinite(spread) or spread <= 0.33:
        return 1.0
    lo, hi = 1.0, 4096.0
    for _ in range(40):  # the ratio rises with k: bisect (in log k)
        k = math.sqrt(lo * hi)
        if _gamma_quantile(0.2, k) / _gamma_quantile(0.5, k) < spread:
            lo = k
        else:
            hi = k
    return math.sqrt(lo * hi)


def find_signals(
    power_db: np.ndarray,
    center_hz: float,
    span_hz: float,
    min_snr_db: float = 10.0,
    usable_fraction: float = 0.9,
    merge_hz: float = 3e3,
) -> List[SignalPeak]:
    """The signals of a spectrum (``power_db``: dBFS per bin, fftshifted,
    of a span ``span_hz`` wide around ``center_hz``) standing at least
    ``min_snr_db`` above the local noise floor, in frequency order. The
    outer edges (``usable_fraction``) and the DC spike are left out."""
    p = np.nan_to_num(np.asarray(power_db, dtype=np.float64), nan=-200.0)
    n = len(p)
    if n < 16 or not span_hz > 0:
        return []
    bin_hz = float(span_hz) / n
    floor = local_floor_db(p)
    smooth = np.convolve(p, np.ones(3) / 3.0, mode="same")
    above = smooth - floor >= float(min_snr_db)
    centre = n // 2
    edge = int(n * (1.0 - float(usable_fraction)) / 2.0)
    above[:edge] = False
    above[n - edge :] = False
    idx = np.flatnonzero(above)
    if not len(idx):
        return []
    gap = max(1, int(round(merge_hz / bin_hz)))
    breaks = np.flatnonzero(np.diff(idx) > gap)
    firsts = np.concatenate(([idx[0]], idx[breaks + 1]))
    lasts = np.concatenate((idx[breaks], [idx[-1]]))
    lin = np.power(10.0, p / 10.0)
    offsets = (np.arange(n) - centre) * bin_hz
    found: List[SignalPeak] = []
    for a, b in zip(firsts.tolist(), lasts.tolist(), strict=True):
        if a >= centre - DC_GUARD_BINS and b <= centre + DC_GUARD_BINS:
            continue  # only the DC spike at the LO
        k = a + int(np.argmax(p[a : b + 1]))
        peak = float(p[k])
        # The core: within 20 dB of the peak, contiguous.
        lo = hi = k
        while lo > 0 and p[lo - 1] >= peak - 20.0 and p[lo - 1] - floor[lo - 1] > 3:
            lo -= 1
        while hi < n - 1 and p[hi + 1] >= peak - 20.0 and p[hi + 1] - floor[hi + 1] > 3:
            hi += 1
        lo, hi = min(lo, a), max(hi, b)
        core = slice(max(lo, k - 64), min(hi, k + 64) + 1)
        weights = lin[core] * (p[core] >= peak - 12.0)
        offset = float(np.sum(offsets[core] * weights) / max(np.sum(weights), 1e-30))
        freq = float(center_hz) + offset
        width = (hi - lo + 1) * bin_hz
        if width >= 60e3 and 87.5e6 <= freq <= 108e6:
            # A broadcast station: on its 100 kHz channel (the centroid of
            # wide FM wanders with the programme).
            freq = round(freq / 100e3) * 100e3
            offset = freq - float(center_hz)
        found.append(
            SignalPeak(
                frequency_hz=freq,
                peak_dbfs=peak,
                snr_db=peak - float(floor[k]),
                width_hz=width,
                offset_hz=offset,
            )
        )
    # A much weaker "signal" inside a strong one's width is part of it, and
    # so is a second part of one station (on the same channel).
    kept: List[SignalPeak] = []
    for s in sorted(found, key=lambda s: -s.peak_dbfs):
        inside = any(
            (
                abs(s.offset_hz - k.offset_hz) <= 0.5 * k.width_hz
                and s.peak_dbfs <= k.peak_dbfs - 20.0
            )
            or abs(s.frequency_hz - k.frequency_hz) < 0.5
            for k in kept
        )
        if not inside:
            kept.append(s)
    return sorted(kept, key=lambda s: s.frequency_hz)


# --------------------------------------------------------------------------- #
# A whole recording
# --------------------------------------------------------------------------- #


@dataclass
class SurveySignal:
    """A signal found in a recording."""

    frequency_hz: float
    peak_dbfs: float
    snr_db: float
    width_hz: float
    first_s: float
    last_s: float
    slices: int = 1  # how many slices heard it
    total_slices: int = 1

    @property
    def duty(self) -> float:
        """Share of the recording it was heard in (0-1)."""
        return self.slices / max(1, self.total_slices)


def survey_nfft(rate: float, rbw_hz: float = 1.2e3) -> int:
    """FFT length giving about ``rbw_hz`` per bin (256 to 16384)."""
    want = max(1.0, float(rate) / float(rbw_hz))
    return int(min(16384, max(256, 1 << int(round(math.log2(want))))))


class SignalSurvey:
    """Finds the signals of a recording, slice by slice (see the module
    docstring). Feed its samples in order with :meth:`add` (any chunk
    size); :meth:`signals` lists what was found so far."""

    def __init__(
        self,
        rate: float,
        center_hz: float,
        slice_s: float = 0.25,
        min_snr_db: float = 10.0,
        nfft: Optional[int] = None,
    ):
        self.rate = float(rate)
        self.center_hz = float(center_hz)
        self.nfft = int(nfft or survey_nfft(self.rate))
        self.slice_samples = max(self.nfft, int(round(slice_s * self.rate)))
        self.min_snr_db = float(min_snr_db)
        self._window = np.hanning(self.nfft + 1)[:-1].astype(np.float32)
        self._scale = 1.0 / float(np.sum(self._window)) ** 2
        self._acc = np.zeros(self.nfft)
        self._top = np.zeros(self.nfft)
        # The slice's spectra one by one (when each signal came and went).
        self._rows: List[np.ndarray] = []
        self._segments = 0
        self._filled = 0  # samples of the current slice
        self._pending = np.zeros(0, dtype=np.complex64)
        self.position = 0  # samples seen
        self.slices = 0
        self._found: List[SurveySignal] = []

    def add(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.complex64)
        if len(self._pending):
            x = np.concatenate((self._pending, x))
        nfft = self.nfft
        count = len(x) // nfft
        if count:
            blocks = x[: count * nfft].reshape(count, nfft)
            # Slices end on whole segments.
            done = 0
            while done < count:
                room = max(1, (self.slice_samples - self._filled) // nfft)
                take = min(room, count - done)
                part = blocks[done : done + take]
                spec = np.fft.fft(part * self._window, axis=1)
                power = spec.real**2 + spec.imag**2
                self._acc += np.sum(power, axis=0)
                np.maximum(self._top, np.max(power, axis=0), out=self._top)
                self._rows.append(power.astype(np.float32))
                self._segments += take
                self._filled += take * nfft
                self.position += take * nfft
                done += take
                if self._filled >= self.slice_samples:
                    self._close_slice()
        self._pending = x[count * nfft :].copy()

    def finish(self) -> List[SurveySignal]:
        """End the last (partial) slice; returns :meth:`signals`."""
        if self._segments:
            self._close_slice()
        return self.signals()

    def _close_slice(self) -> None:
        power = np.fft.fftshift(self._acc / max(self._segments, 1)) * self._scale
        db = 10.0 * np.log10(np.maximum(power, 1e-30))
        # A signal's level: its strongest moment (a burst is on for part of
        # the slice only), where it was found.
        top = 10.0 * np.log10(
            np.maximum(np.fft.fftshift(self._top) * self._scale, 1e-30)
        )
        end_s = self.position / self.rate
        start_s = end_s - self._filled / self.rate
        self.slices += 1
        bin_hz = self.rate / self.nfft
        seg_s = self.nfft / self.rate
        rows = np.fft.fftshift(np.concatenate(self._rows), axes=1) * self._scale
        floor = np.power(10.0, local_floor_db(db) / 10.0)
        for peak in find_signals(db, self.center_hz, self.rate, self.min_snr_db):
            k = int(round(peak.offset_hz / bin_hz)) + self.nfft // 2
            lo, hi = max(0, k - 2), min(self.nfft, k + 3)
            peak.peak_dbfs = max(peak.peak_dbfs, float(np.max(top[lo:hi])))
            # When in the slice it was on: the segments where its bins stand
            # out (to a segment, not to the slice).
            near = slice(max(0, k - 1), min(self.nfft, k + 2))
            level = np.sum(rows[:, near], axis=1)
            limit = float(np.sum(floor[near])) * 10.0 ** (self.min_snr_db / 10.0)
            on = np.flatnonzero(level > limit)
            heard_from = start_s + float(on[0]) * seg_s if len(on) else start_s
            heard_to = start_s + float(on[-1] + 1) * seg_s if len(on) else end_s
            reach = max(0.5 * peak.width_hz, 2.5 * bin_hz, 3e3)
            match = None
            for known in self._found:
                if abs(known.frequency_hz - peak.frequency_hz) <= max(
                    reach, 0.5 * known.width_hz
                ):
                    match = known
                    break
            if match is None:
                self._found.append(
                    SurveySignal(
                        frequency_hz=peak.frequency_hz,
                        peak_dbfs=peak.peak_dbfs,
                        snr_db=peak.snr_db,
                        width_hz=peak.width_hz,
                        first_s=heard_from,
                        last_s=heard_to,
                    )
                )
                continue
            match.slices += 1
            match.last_s = heard_to
            if peak.peak_dbfs > match.peak_dbfs:
                match.peak_dbfs = peak.peak_dbfs
                match.frequency_hz = peak.frequency_hz
            match.snr_db = max(match.snr_db, peak.snr_db)
            match.width_hz = max(match.width_hz, peak.width_hz)
        self._acc[:] = 0.0
        self._top[:] = 0.0
        self._rows = []
        self._segments = 0
        self._filled = 0

    def signals(self) -> List[SurveySignal]:
        """What was found, in frequency order."""
        for s in self._found:
            s.total_slices = max(1, self.slices)
        return sorted(self._found, key=lambda s: s.frequency_hz)
