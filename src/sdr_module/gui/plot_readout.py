"""
What the plots' hover readouts say about the level under the pointer.

Qt-free: :class:`~sdr_module.gui.spectrum_widget.SpectrumWidget` and
:class:`~sdr_module.gui.waterfall_widget.WaterfallWidget` read through it.

A pixel of a plot can stand for many FFT bins (33 at 32768 points across
1000 pixels) and is drawn at the strongest of them, and a waterfall row is
the strongest of its frame's FFTs. Read as drawn, plain noise reads several
dB above its mean (how many depends on the FFT size, the zoom and the
sample rate) and so does a weak signal. A reading therefore takes the
averaged spectrum (the mean of the frame's FFTs) under the pointer and
judges it against what plain noise reads there:

* the noise floor (the mean noise power in a bin), and how many FFTs'
  worth of averaging its bins have, both measured on the line itself: the
  bins below its median are mostly noise, and the mean of ``k`` FFTs' noise
  spreads as a Gamma(k) distribution does (:func:`averages_from_spread`),
  so noise rougher than white counts as less averaged and is judged by
  how it really spreads;
* plain noise's strongest bin among those under the pointer stays below
  :meth:`NoiseModel.peak_limit` all but :data:`FALSE_ALARM` of the time,
  and their power average below a limit measured the same way from the
  line's averages over as many bins (:func:`mean_limit`; neighbouring bins
  of a windowed spectrum aren't independent).

A bin standing above that is a line (a carrier): its level is read. Bins
under the pointer that hold more power than noise but none of them a line
are a signal filling the pixel (a station wider than the pixel): their
power average is read. Otherwise it is noise: their average, ``noise``.
A waterfall row's strongest FFTs are judged too, as a burst shorter than
its frame (:func:`read_burst`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FALSE_ALARM",
    "NoiseModel",
    "Reading",
    "averages_from_spread",
    "mean_limit",
    "noise_model",
    "read_burst",
    "read_levels",
]

#: Plain noise reads as a signal at about one place in 200 (a third of
#: that each as a line, as a signal filling the pixel and as a burst).
FALSE_ALARM = 0.005
_EACH = FALSE_ALARM / 3.0

#: The 10th to the 50th percentile (dB) of the mean of ``k`` unit
#: exponentials (noise power averaged over ``k`` FFTs), exact from its CDF.
_SPREAD_DB: Tuple[Tuple[float, float], ...] = (
    (1, 8.1815),
    (2, 4.9912),
    (3, 3.8496),
    (4, 3.2317),
    (6, 2.5502),
    (8, 2.1673),
    (12, 1.7328),
    (16, 1.4831),
    (24, 1.1949),
    (32, 1.0270),
    (48, 0.8313),
    (64, 0.7164),
    (96, 0.5816),
    (128, 0.5020),
    (192, 0.4083),
)
#: Fewest bins a line must have for its noise to be measured.
_MIN_BINS = 32
#: Fewest bins under the pointer that can be a signal filling the pixel.
_MIN_FILL_BINS = 4
#: About how many of a line's bins its noise is measured on.
_SAMPLE_BINS = 4096

_NORMAL = NormalDist()


def averages_from_spread(spread_db: float) -> float:
    """How many FFTs' worth of averaging noise has when its 10th to 50th
    percentile span ``spread_db`` dB (1 for a single FFT's noise or anything
    rougher)."""
    s = float(spread_db)
    if not math.isfinite(s) or s >= _SPREAD_DB[0][1]:
        return 1.0
    last_k, last_s = _SPREAD_DB[-1]
    if s <= last_s:
        return last_k * (last_s / max(s, 1e-6)) ** 2  # (as 1 / sqrt(k))
    for (k0, s0), (k1, s1) in zip(_SPREAD_DB[:-1], _SPREAD_DB[1:], strict=True):
        if s1 <= s <= s0:
            frac = math.log(s0 / s) / math.log(s0 / s1)
            return float(math.exp(math.log(k0) + frac * math.log(k1 / k0)))
    return 1.0  # pragma: no cover - the table covers every spread


def _gamma_quantile(p: float, k: float) -> float:
    """Quantile ``p`` of the mean of ``k`` unit exponentials (Wilson-
    Hilferty: within a few per cent in the upper tail, even for k = 1)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-15)
    c = 1.0 / (9.0 * max(float(k), 0.5))
    return float(max(1e-6, (1.0 - c + _NORMAL.inv_cdf(p) * math.sqrt(c)) ** 3))


def _median_ratio(k: float) -> float:
    """The median of the mean of ``k`` unit exponentials (ln 2 for one)."""
    k = max(float(k), 1.0)
    return math.log(2.0) if k <= 1.0 else 1.0 - 1.0 / (3 * k) + 8.0 / (405 * k * k)


@dataclass(frozen=True)
class NoiseModel:
    """The noise of a spectrum line: its floor (the mean noise power in a
    bin, dBFS) and how many FFTs' worth of averaging its bins have."""

    floor_db: float
    averages: float

    def peak_limit(self, bins: float) -> float:
        """The power (over the floor) plain noise's strongest of ``bins``
        bins stays below all but a third of :data:`FALSE_ALARM` of the
        time."""
        p = (1.0 - _EACH) ** (1.0 / max(1.0, float(bins)))
        return _gamma_quantile(p, self.averages)


def _finite(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def noise_model(
    line_db: Sequence[float], floor_db: Optional[float] = None
) -> Optional[NoiseModel]:
    """The noise of ``line_db`` (dB per bin), measured from its bins below
    the median; ``floor_db`` is the mean noise in a bin when the caller
    knows it (else the median, corrected for the averaging measured). None
    for a line too short to tell."""
    v = _finite(line_db)
    if len(v) < _MIN_BINS:
        return None
    v = v[:: max(1, len(v) // _SAMPLE_BINS)]  # (plenty for two percentiles)
    p10, p50 = np.percentile(v, (10.0, 50.0))
    k = averages_from_spread(float(p50 - p10))
    if floor_db is None or not math.isfinite(float(floor_db)):
        floor_db = float(p50) - 10.0 * math.log10(_median_ratio(k))
    return NoiseModel(float(floor_db), k)


def mean_limit(line_db: Sequence[float], bins: int) -> Optional[float]:
    """The power (over the line's mean noise) the average of ``bins``
    neighbouring bins of plain noise stays below all but a third of
    :data:`FALSE_ALARM` of the time, measured from the line's own averages
    over ``bins`` bins (correlated neighbours spread more than independent
    ones would). None when ``bins`` is too few, or the line too short."""
    bins = int(bins)
    v = np.asarray(line_db, dtype=np.float64).ravel()
    if bins < _MIN_FILL_BINS or len(v) < bins * _MIN_BINS:
        return None
    count = len(v) // bins
    power = 10.0 ** (v[: count * bins].reshape(count, bins) / 10.0)
    means = power.mean(axis=1)
    means = means[np.isfinite(means) & (means > 0)]
    if len(means) < _MIN_BINS:
        return None
    db = 10.0 * np.log10(means)
    p10, p50 = np.percentile(db, (10.0, 50.0))
    k = averages_from_spread(float(p50 - p10))
    # (Relative to the averages' own mean noise, which is the bins'.)
    return _gamma_quantile(1.0 - _EACH, k)


@dataclass(frozen=True)
class Reading:
    """What a readout says about the level under the pointer."""

    level_db: float
    #: dB over the noise floor, None for plain noise.
    snr_db: Optional[float]
    #: The strongest of a frame's FFTs (a burst), not their mean.
    peak: bool = False

    @property
    def is_noise(self) -> bool:
        return self.snr_db is None

    def text(self, decimals: int = 1) -> str:
        """``-62.3 dBFS · SNR 31 dB``, ``-95.1 dBFS · noise`` or, for a
        burst, ``Peak -60.0 dBFS · SNR 33 dB``."""
        level = f"{self.level_db:.{decimals}f} dBFS"
        if self.peak:
            level = "Peak " + level
        if self.snr_db is None:
            return f"{level} · noise"
        return f"{level} · SNR {self.snr_db:.0f} dB"


def _snr_db(power: float) -> float:
    """dB of the power above the noise over the noise (``power`` is over
    the noise floor: 1 is the noise alone)."""
    return 10.0 * math.log10(max(power - 1.0, 1e-3))


def read_levels(
    levels_db: Sequence[float],
    model: NoiseModel,
    fill_limit: Optional[float] = None,
) -> Optional[Reading]:
    """The reading of the bins under the pointer (``levels_db``, the
    averaged spectrum in dBFS), judged against ``model``. ``fill_limit``:
    :func:`mean_limit` for as many bins (without it, a signal filling the
    pixel isn't told from noise unless one of its bins stands out).

    A line: the strongest bin stands out of the noise and of the other
    bins under the pointer (a carrier; read at its level). A signal filling
    the pixel: the bins' average holds more power than the noise's, none
    of them a line (read at their average, its power density). Otherwise
    noise (read at their average)."""
    v = _finite(levels_db)
    if not len(v):
        return None
    floor = model.floor_db
    power = 10.0 ** ((v - floor) / 10.0)
    m = len(v)
    top = int(np.argmax(power))
    peak = float(power[top])
    mean = float(power.mean())
    # (Beside the strongest bin, past the window's main lobe.)
    others = np.concatenate((power[: max(0, top - 1)], power[top + 2 :]))
    limit = model.peak_limit(m)
    if peak >= limit:
        if not len(others) or peak >= float(others.mean()) * limit:
            return Reading(floor + 10.0 * math.log10(peak), _snr_db(peak))
        # Above the noise, no more than the others: it fills the pixel.
        return Reading(floor + 10.0 * math.log10(mean), _snr_db(mean))
    if fill_limit is not None and m >= _MIN_FILL_BINS and mean >= fill_limit:
        return Reading(floor + 10.0 * math.log10(mean), _snr_db(mean))
    return Reading(floor + 10.0 * math.log10(mean), None)


def read_burst(
    peak_db: float,
    floor_db: float,
    rise_db: float,
    readings: int,
    steady_db: Optional[float] = None,
) -> Optional[Reading]:
    """A burst: the strongest of ``readings`` frame maxima (``peak_db``,
    each the strongest of a frame's FFTs) standing out of what the steady
    level under them (``steady_db``, the averaged spectrum's reading; the
    floor ``floor_db`` without one) reads as its strongest FFT. A frame's
    noise maximum sits ``rise_db`` above the floor at its median, which
    gives how many FFTs it is the strongest of (n, the median of the
    strongest of n unit exponentials being -ln(1 - 0.5^(1/n))). A steady
    level that varies like noise (the noise itself, a wide station) reaches
    ``readings`` x n FFTs' strongest; a carrier's varies less. None when
    the steady level explains ``peak_db`` all but a third of
    :data:`FALSE_ALARM` of the time."""
    values = (peak_db, floor_db, rise_db)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    median = 10.0 ** (max(float(rise_db), 0.0) / 10.0)
    n = max(1.0, math.log(0.5) / math.log(max(1e-12, 1.0 - math.exp(-median))))
    total = n * max(1, int(readings))
    limit = -math.log(max(1e-300, 1.0 - (1.0 - _EACH) ** (1.0 / total)))
    base = floor_db if steady_db is None else max(float(steady_db), floor_db)
    if 10.0 ** ((peak_db - base) / 10.0) < limit:
        return None
    return Reading(
        float(peak_db), _snr_db(10.0 ** ((peak_db - floor_db) / 10.0)), peak=True
    )
