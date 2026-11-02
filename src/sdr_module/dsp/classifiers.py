"""
Signal classification for automatic modulation recognition.

:class:`SignalClassifier` says what kind of signal a block of complex
baseband holds (the signal near 0 Hz): noise, an unmodulated carrier, CW
(Morse), AM, SSB, FM (narrow or broadcast), FSK, PSK, on/off keyed (OOK)
data bursts or short pulses (radar, Mode S / ADS-B replies), with its
occupied bandwidth, its signal-to-noise ratio and a confidence.

How it decides:

* **Noise** is measured honestly. The noise density comes from the caller
  (``noise_psd``: e.g. the receiver's spectrum outside the channel) or from
  the block itself, as a low quantile of the averaged spectrum *inside the
  passband* (``band_hz``, or the band a channel filter visibly passes).
  The stopband of a channel filter is never taken for noise: that gave
  50-150 dB "SNRs" and a confidence of 1.0 for anything, noise included.
  What stands above it is judged against how much the noise's spectrum
  actually varies (measured, not assumed from the averaging). A carrier's
  faint sidebands are judged with the spectrum mirrored about it (they
  are the same either side, the noise isn't), and FSK's two tones are
  taken as one signal when their power lies symmetric about the tuned
  frequency.
* **Keying** (CW, OOK, pulses, a transmission starting) is found from the
  power envelope: two clearly separated levels, the low one at the noise.
  The modulation of what is keyed is then judged on the "on" samples only.
* **Modulation** features: how much the envelope varies (from the power
  moments, beyond what the noise explains; over all of a carrier's
  sidebands), how far the instantaneous frequency swings (beyond the
  noise), whether it dwells on two values (FSK), the share of power in a
  carrier line, M-th power spectral lines (PSK) and the occupied width.
* Every candidate kind gets a score from these features; the best one
  wins, and the confidence falls when the runner-up is close (the features
  disagree) or the SNR is low. It is a heuristic, not a calibrated
  probability, and it never claims more than :data:`MAX_CONFIDENCE`.

No Qt; NumPy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .demodulators import ModulationType

__all__ = [
    "SignalType",
    "ClassificationResult",
    "SignalClassifier",
    "KIND_LABELS",
    "MAX_CONFIDENCE",
    "choose_nfft",
    "welch_psd",
    "noise_floor_psd",
    "noise_psd_from_bin_db",
]


class SignalType(Enum):
    """High-level signal type."""

    UNKNOWN = "unknown"
    NOISE = "noise"
    ANALOG = "analog"
    DIGITAL = "digital"
    PULSED = "pulsed"


#: What each fine-grained kind (``ClassificationResult.kind``) is called.
KIND_LABELS: Dict[str, str] = {
    "noise": "No signal (noise)",
    "carrier": "Unmodulated carrier",
    "cw": "CW (Morse code)",
    "am": "AM",
    "ssb": "SSB voice",
    "fm": "FM",
    "fsk": "FSK data",
    "psk": "PSK data",
    "ook": "OOK data bursts",
    "pulsed": "Short pulses",
    "unknown": "Unknown signal",
}

# kind -> (signal type, modulation)
_KIND_TYPES: Dict[str, Tuple[SignalType, Optional[ModulationType]]] = {
    "noise": (SignalType.NOISE, None),
    "carrier": (SignalType.ANALOG, ModulationType.CW),
    "cw": (SignalType.ANALOG, ModulationType.CW),
    "am": (SignalType.ANALOG, ModulationType.AM),
    "ssb": (SignalType.ANALOG, None),
    "fm": (SignalType.ANALOG, ModulationType.FM),
    "fsk": (SignalType.DIGITAL, ModulationType.FSK),
    "psk": (SignalType.DIGITAL, ModulationType.BPSK),
    "ook": (SignalType.DIGITAL, ModulationType.OOK),
    "pulsed": (SignalType.PULSED, None),
    "unknown": (SignalType.UNKNOWN, None),
}

#: The confidence is never higher than this (it is a heuristic).
MAX_CONFIDENCE = 0.95
#: Hann window equivalent noise bandwidth, in bins.
_HANN_ENBW = 1.5
#: The lower quartile of exponentially distributed power (noise), over its
#: mean: -ln(0.75).
_Q25_EXP = -math.log(0.75)
#: Keyed bursts whose mean level is at least this many times the keying
#: threshold (which noise alone passes about once in a thousand samples)
#: stand far out of any noise.
_STRONG_BURSTS = 4.0


@dataclass
class ClassificationResult:
    """Result of signal classification."""

    signal_type: SignalType
    modulation: Optional[ModulationType]
    confidence: float  # 0.0 to MAX_CONFIDENCE
    bandwidth_hz: float  # occupied: 99 % of the signal's power
    center_offset_hz: float  # where the signal's power is centred
    snr_db: float  # signal to noise in the occupied bandwidth
    features: Dict[str, Any]
    #: Fine-grained class: a key of :data:`KIND_LABELS`.
    kind: str = "unknown"
    #: Short plain-English reasons ("Constant envelope; ...").
    evidence: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """What the signal is called ("FM", "OOK data bursts", ...)."""
        return KIND_LABELS.get(self.kind, KIND_LABELS["unknown"])


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below ``lo``, 1 above ``hi``, linear in between."""
    if not math.isfinite(x):
        return 1.0 if x > 0 else 0.0
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


# Acklam's rational approximation of the inverse normal CDF.
_PPF_A = (
    -39.69683028665376,
    220.9460984245205,
    -275.9285104469687,
    138.3577518672690,
    -30.66479806614716,
    2.506628277459239,
)
_PPF_B = (
    -54.47609879822406,
    161.5858368580409,
    -155.6989798598866,
    66.80131188771972,
    -13.28068155288572,
    1.0,
)
_PPF_C = (
    -0.007784894002430293,
    -0.3223964580411365,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
)
_PPF_D = (
    0.007784695709041462,
    0.3224671290700398,
    2.445134137142996,
    3.754408661907416,
    1.0,
)


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _norm_ppf(p: float) -> float:
    """Inverse of the standard normal CDF (Acklam's approximation)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    if p > 0.5:
        return -_norm_ppf(1.0 - p)
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return _horner(_PPF_C, q) / _horner(_PPF_D, q)
    q = p - 0.5
    r = q * q
    return _horner(_PPF_A, r) * q / _horner(_PPF_B, r)


def _gamma_quantile(p: float, k: float) -> float:
    """Quantile ``p`` of the mean of ``k`` unit-mean exponential values,
    Gamma(k, 1/k) (Wilson-Hilferty approximation)."""
    k = max(float(k), 0.5)
    c = 1.0 / (9.0 * k)
    return float(max(1e-3, (1.0 - c + _norm_ppf(p) * math.sqrt(c)) ** 3))


def _gamma_cdf(x: float, k: float) -> float:
    """CDF of Gamma(k, 1/k) at ``x`` (Wilson-Hilferty)."""
    k = max(float(k), 0.5)
    c = 1.0 / (9.0 * k)
    z = (max(float(x), 1e-12) ** (1.0 / 3.0) - (1.0 - c)) / math.sqrt(c)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Spectrum helpers
# --------------------------------------------------------------------------- #


def choose_nfft(
    n: int, segments: int = 16, min_nfft: int = 64, max_nfft: int = 16384
) -> int:
    """FFT length for a Welch estimate of ``n`` samples with at least
    about ``segments`` half-overlapping segments (a power of two)."""
    n = int(n)
    if n < 2 * min_nfft:
        return max(8, 1 << max(3, int(math.floor(math.log2(max(n, 8))))))
    target = max(min_nfft, 2 * n // max(1, segments + 1))
    nfft = 1 << int(math.floor(math.log2(target)))
    return int(min(max(nfft, min_nfft), max_nfft))


def welch_psd(
    x: np.ndarray, rate: float, nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Welch power spectral density of complex ``x`` (Hann, 50 % overlap).

    Returns ``(freqs, psd, segments)``: frequencies in Hz and power per Hz,
    both fftshifted (0 Hz in the middle). White noise of total power ``s2``
    reads ``s2 / rate`` per Hz. Segments are transformed a batch at a time,
    so a long block needs no large temporaries.
    """
    data = np.asarray(x, dtype=np.complex64).ravel()
    n = len(data)
    if nfft is None:
        nfft = choose_nfft(n)
    nfft = int(max(8, nfft))
    if n < nfft:
        data = np.concatenate((data, np.zeros(nfft - n, dtype=np.complex64)))
        n = nfft
    hop = max(1, nfft // 2)
    count = 1 + (n - nfft) // hop
    window = np.hanning(nfft + 1)[:-1]
    win32 = window.astype(np.float32)
    acc = np.zeros(nfft, dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(data, nfft)[::hop][:count]
    batch = max(1, (1 << 19) // nfft)
    for start in range(0, count, batch):
        spec = np.fft.fft(view[start : start + batch] * win32, axis=1)
        acc += np.sum(
            spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2,
            axis=0,
        )
    psd = acc / (count * float(rate) * float(np.sum(window**2)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / float(rate)))
    return freqs, np.fft.fftshift(psd), int(count)


def noise_floor_psd(psd: np.ndarray, segments: int) -> float:
    """The noise density (the units of ``psd``) of Welch spectrum bins that
    are mostly noise, robust to signals occupying up to about 85 % of them.

    Averaged noise bins follow a Gamma distribution (``segments`` averaged
    periodograms, about 0.9 of them independent with 50 % overlap). Start
    from the 10th percentile corrected for that distribution, then take the
    median of the bins within a few standard deviations of the estimate
    (signals left out), corrected the same way.
    """
    values = np.asarray(psd, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 1e-30
    k = max(1.0, 0.9 * float(segments))
    mean = float(np.percentile(values, 10)) / _gamma_quantile(0.10, k)
    spread = 1.0 + 3.5 / math.sqrt(k)
    median_of_cut = _gamma_quantile(0.5 * _gamma_cdf(spread, k), k)
    for _ in range(3):
        kept = values[values < mean * spread]
        if len(kept) < 8:
            break
        mean = float(np.median(kept)) / median_of_cut
    return max(mean, 1e-30)


def noise_psd_from_bin_db(
    bin_db: float, fft_size: int, rate: float, enbw_bins: float = _HANN_ENBW
) -> float:
    """Noise density (power per Hz, full scale = 1) from the noise level of
    a Hann-windowed spectrum's bins in dBFS, referenced to the window's
    coherent gain as the main display draws it: white noise of total power
    ``s2`` reads ``s2 * enbw / N`` per bin."""
    per_bin = 10.0 ** (float(bin_db) / 10.0)
    return per_bin * float(fft_size) / (float(enbw_bins) * float(rate))


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Moving average over ``width`` neighbours (same length; the edges
    average what exists)."""
    v = np.asarray(values, dtype=np.float64)
    width = int(max(1, min(width, len(v))))
    if width == 1 or len(v) < 2:
        return v.copy()
    kernel = np.ones(width) / width
    ones = np.convolve(np.ones(len(v)), kernel, mode="same")
    return np.convolve(v, kernel, mode="same") / np.maximum(ones, 1e-12)


def _visible_band(psd: np.ndarray) -> np.ndarray:
    """The bins a channel filter passes, when its stopband shows (the
    outer bins far below the middle); otherwise every bin."""
    return _passband(psd)[0]


def _passband(psd: np.ndarray) -> Tuple[np.ndarray, bool]:
    """:func:`_visible_band`, and whether the passband found is flat all
    over (filtered noise, or a flat-topped signal with steep sides over an
    unfiltered floor: the caller tells them apart by the statistics)."""
    n = len(psd)
    mask = np.ones(n, dtype=bool)
    if n < 32:
        return mask, False
    db = 10.0 * np.log10(np.maximum(psd, 1e-30))
    edge_n = max(1, n // 20)
    edge = float(np.median(np.concatenate((db[:edge_n], db[-edge_n:]))))
    smooth = _smooth(db, max(1, n // 64))
    centre = float(np.percentile(db[n // 4 : 3 * n // 4], 20))
    flat_top = False
    if centre - edge >= 20.0:
        # A passband wider than about half the band.
        span = _run_around(smooth > 0.5 * (centre + edge), n // 2)
    else:
        found = _narrow_passband(db, smooth, edge)
        span = found[:2] if found is not None else None
        flat_top = bool(found[2]) if found is not None else False
    if span is None:
        return mask, False
    mask[:] = False
    mask[span[0] : span[1] + 1] = True
    return mask, flat_top


def _run_around(above: np.ndarray, mid: int) -> Optional[Tuple[int, int]]:
    """First and last index of the run of True in ``above`` that holds
    ``mid`` (or the True nearest it); None when there is none."""
    if not above[mid]:
        hits = np.flatnonzero(above)
        if not len(hits):
            return None
        mid = int(hits[np.argmin(np.abs(hits - mid))])
    lo = mid
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = mid
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    return lo, hi


def _narrow_passband(
    db: np.ndarray, smooth: np.ndarray, edge: float
) -> Optional[Tuple[int, int, bool]]:
    """A channel filter's passband narrower than half the band: centred
    on 0 Hz, far above the stopband, and flat (the noise) next to its
    steep sides, at the same level on both sides, which is the passband's
    floor. A signal over an unfiltered noise floor slopes down to the
    floor instead. Its sides are looked for 6 dB, 9 dB, ... below the
    level at 0 Hz (a signal there stands above the passband's noise).
    Returns ``(first bin, last bin, flat all over)`` or None."""
    n = len(db)
    mid = n // 2
    top = float(smooth[mid])
    if top - edge < 25.0:
        return None
    skip = max(1, n // 128)  # where the smoothing reaches past the side

    def level(a: int, b: int) -> float:
        return 10.0 * math.log10(max(float(np.mean(10.0 ** (db[a:b] / 10.0))), 1e-30))

    threshold = top - 6.0
    while threshold >= edge + 15.0:
        span = _run_around(smooth > threshold, mid)
        threshold -= 3.0
        if span is None:
            continue
        lo, hi = span
        width = hi - lo + 1
        strip = max(4, int(0.12 * width))
        if width < max(16, n // 32) or width < 2 * (skip + strip) + 4:
            continue
        if abs(0.5 * (lo + hi) - mid) > 0.15 * width:
            continue  # a channel filter is centred on 0 Hz
        half = strip // 2
        left_out = level(lo + skip, lo + skip + half)
        left_in = level(lo + skip + half, lo + skip + strip)
        right_in = level(hi + 1 - skip - strip, hi + 1 - skip - half)
        right_out = level(hi + 1 - skip - half, hi + 1 - skip)
        left = level(lo + skip, lo + skip + strip)
        right = level(hi + 1 - skip - strip, hi + 1 - skip)
        flat = abs(left_in - left_out) <= 2.0 and abs(right_in - right_out) <= 2.0
        even = abs(left - right) <= 3.0
        deep = min(left, right) - edge >= 25.0
        floor = max(left, right) <= float(np.median(db[lo : hi + 1])) + 1.5
        if flat and even and deep and floor:
            return lo, hi, top - max(left, right) < 6.0
    return None


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of the runs of True in ``mask``."""
    m = np.asarray(mask, dtype=np.int8)
    if not len(m):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def _fill_runs(mask: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> None:
    """Set ``mask`` True over the given runs (in place, vectorised)."""
    if not len(starts):
        return
    marks = np.zeros(len(mask) + 1, dtype=np.int64)
    np.add.at(marks, starts, 1)
    np.add.at(marks, starts + lengths, -1)
    mask |= np.cumsum(marks[:-1]) > 0


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean of ``width`` samples (same length) via cumsum."""
    width = int(max(1, width))
    v = np.asarray(values, dtype=np.float64)
    if width == 1 or len(v) <= width:
        return v.copy()
    c = np.cumsum(np.concatenate(([0.0], v)))
    out = (c[width:] - c[:-width]) / width
    pad_lo = (width - 1) // 2
    pad_hi = len(v) - len(out) - pad_lo
    return np.concatenate((np.full(pad_lo, out[0]), out, np.full(pad_hi, out[-1])))


def _narrow(
    x: np.ndarray,
    rate: float,
    centre_hz: float,
    half_hz: float,
    spectrum: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """The part of ``x`` within ``half_hz`` of ``centre_hz``, moved to 0 Hz
    and decimated (in the frequency domain) to about four times
    ``half_hz``. Returns ``(samples, new rate)``; amplitudes are kept.
    ``spectrum``: ``np.fft.fft(x)``, when the caller has it already.

    The band's edges are raised-cosine tapers from 0.8 to 1.2 times
    ``half_hz`` (half way down at ``half_hz``, about the same noise
    bandwidth as a brick wall): a brick wall rings for a long time after a
    strong signal is keyed off, which fills the gaps of Morse code."""
    n = len(x)
    if spectrum is None or len(spectrum) != n:
        spectrum = np.fft.fft(np.asarray(x, dtype=np.complex64))
    bin_hz = rate / n
    shift = int(round(centre_hz / bin_hz))
    spectrum = np.roll(spectrum, -shift)
    want = max(64, int(math.ceil(4.0 * half_hz / bin_hz)))
    size = 1 << int(math.ceil(math.log2(want)))
    if size >= n:
        size = n
    keep = int(min(size // 2, math.ceil(1.2 * half_hz / bin_hz)))
    f = np.arange(keep + 1) * bin_hz
    edge = np.clip((f - 0.8 * half_hz) / max(0.4 * half_hz, 1e-12), 0.0, 1.0)
    taper = 0.5 * (1.0 + np.cos(np.pi * edge))
    small = np.zeros(size, dtype=np.complex128)
    small[: keep + 1] = spectrum[: keep + 1] * taper
    if keep > 0:
        small[-keep:] = spectrum[-keep:] * taper[1:][::-1]
    out = np.fft.ifft(small) * (size / n)
    return out.astype(np.complex64), rate * size / n


def _gaussian(x: np.ndarray, freqs: np.ndarray, band: np.ndarray, rate: float) -> bool:
    """Whether what ``band`` passes is Gaussian noise: its power,
    normalised to its mean, follows the exponential distribution (a
    Kolmogorov-Smirnov test at 1 %). A modulated signal's doesn't: a
    constant envelope sits near 1, on/off keying near 0 and 2."""
    lo, hi = float(freqs[band][0]), float(freqs[band][-1])
    half = max(0.5 * (hi - lo), 1.0)
    xs, rate_n = _narrow(x, rate, 0.5 * (lo + hi), half)
    step = max(1, int(round(rate_n / (2.0 * half))))  # about independent
    p = (np.abs(xs.astype(np.complex128)) ** 2)[::step]
    mean = float(np.mean(p)) if len(p) else 0.0
    if mean <= 0 or len(p) < 32:
        return True
    p = np.sort(p / mean)
    count = len(p)
    model = 1.0 - np.exp(-p)
    above = np.arange(1, count + 1) / count - model
    below = model - np.arange(0, count) / count
    distance = float(max(np.max(above), np.max(below)))
    return distance <= max(0.06, 1.63 / math.sqrt(count))


def _burst_band_fraction(
    x: np.ndarray,
    on: np.ndarray,
    noise: float,
    rate: float,
    lo_hz: float,
    hi_hz: float,
    centre_hz: float,
    width_hz: float,
    nfft: int = 64,
) -> float:
    """How much of the bursts' power is spread across the band (``lo_hz``
    to ``hi_hz``) away from the signal (``width_hz`` wide at ``centre_hz``):
    about half for pulses as wide as the band (Mode S: the other half is
    the carrier line their keying leaves), small for a keyed carrier,
    however strong (its keying splatter stays near it). 0 unless
    the band away from the signal clearly stands out of the noise (density
    ``noise``) while the bursts are on. Every segment lies inside one "on"
    stretch (joining them would splatter)."""
    starts, lengths = _runs(on)
    keep = lengths >= nfft
    if not np.any(keep):
        return 0.0
    hop = nfft // 2
    firsts = [
        s + np.arange(0, ln - nfft + 1, hop)
        for s, ln in zip(starts[keep].tolist(), lengths[keep].tolist(), strict=True)
    ]
    seg = np.concatenate(firsts)[:4096]
    if len(seg) < 8:
        return 0.0
    window = np.hanning(nfft + 1)[:-1]
    data = np.asarray(x)[seg[:, None] + np.arange(nfft)[None, :]] * window
    spec = np.fft.fft(data, axis=1)
    psd = np.mean(np.abs(spec) ** 2, axis=0) / (float(rate) * float(np.sum(window**2)))
    psd = np.fft.fftshift(psd)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / float(rate)))
    inside = (freqs >= lo_hz) & (freqs <= hi_hz)
    core = np.abs(freqs - centre_hz) <= max(2.0 * width_hz, 2.0 * rate / nfft)
    away = inside & ~core
    if np.count_nonzero(away) < 4:
        return 0.0
    excess = np.maximum(psd - float(noise), 0.0)
    total = float(np.sum(excess[inside]))
    k = max(1.0, 0.9 * len(seg))
    stands_out = float(np.mean(psd[away])) > float(noise) * (
        1.0 + 3.0 / math.sqrt(k * np.count_nonzero(away))
    )
    if total <= 0.0 or not stands_out:
        return 0.0
    return float(np.sum(excess[away])) / total


def _kurtosis_z(power: np.ndarray, noise_power: float, band_fraction: float) -> float:
    """How far the power's kurtosis is from Gaussian noise's (2), in rough
    standard deviations: a constant envelope reads well below, pulses well
    above. Neighbouring samples of filtered noise are correlated, so only
    ``band_fraction`` of them count as independent."""
    p = np.asarray(power, dtype=np.float64)
    m2 = float(np.mean(p))
    if m2 <= 0 or len(p) < 16:
        return 0.0
    kurt = float(np.mean(p * p)) / (m2 * m2)
    n_eff = max(16.0, len(p) * min(1.0, max(float(band_fraction), 1e-3)))
    return (kurt - 2.0) / math.sqrt(20.0 / n_eff)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at which half of the total weight is reached."""
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(v[order][min(k, len(v) - 1)])


def _spread(ratio: np.ndarray, averages: float) -> float:
    """The noise's standard deviation in ``ratio`` (an averaged spectrum
    over the noise density: about 1 where there is only noise), measured
    from its lower values (signals lie above them), and never less than
    ``averages`` independent averages give. Neighbouring bins of a windowed
    spectrum are not independent, nor is every noise source white: taken as
    independent, the noise's own ripple stood out as signal all over a wide
    band and joined a narrow FM call to a station 385 kHz away."""
    floor = 1.0 / math.sqrt(max(float(averages), 1.0))
    if len(ratio) < 32:
        return floor
    low, mid = np.percentile(ratio, (5.0, 30.0))
    # (5th to 30th percentile of a normal spread: 1.12 standard deviations.)
    return max(floor, float(mid - low) / 1.1205)


def _tone_pair(
    centre: int,
    powers: np.ndarray,
    lo_core: np.ndarray,
    hi_core: np.ndarray,
    clear: np.ndarray,
    holds: np.ndarray,
    gap: int,
) -> Optional[Tuple[int, int]]:
    """Two groups of bins of about the same power (indices into the
    groups) to take as one signal because the tuned frequency (bin
    ``centre``) lies between them, about as far from their inner edges
    (cores ``lo_core``..``hi_core``), and no group ``holds`` it:
    ``(below, above)``, or None."""
    below = np.flatnonzero((hi_core < centre) & clear)
    above = np.flatnonzero((lo_core > centre) & clear)
    if np.any(holds) or not len(below) or not len(above):
        return None
    b = int(below[np.argmax(hi_core[below])])
    a = int(above[np.argmin(lo_core[above])])
    d_lo = centre - hi_core[b]
    d_hi = lo_core[a] - centre
    similar = 0.25 <= powers[b] / max(powers[a], 1e-30) <= 4.0
    if similar and max(d_lo, d_hi) <= 2.0 * max(min(d_lo, d_hi), gap):
        return b, a
    return None


def _symmetric_pair(
    starts: np.ndarray,
    lengths: np.ndarray,
    excess: np.ndarray,
    powers: np.ndarray,
    lo_core: np.ndarray,
    hi_core: np.ndarray,
    clear: np.ndarray,
    reach_bins: float,
    gap: int,
) -> Optional[Tuple[int, int]]:
    """The two tones of FSK, by their own symmetry about the tuned frequency
    (the middle bin): two groups of bins of about the same power, whose
    power is centred about as far below it as above it, within
    ``reach_bins``, while no group is centred on it. ``(below, above)``
    (indices into the groups), or None.

    Their inner edges say little: the noise in the dip between the tones
    joins a few bins to one tone's group or the other's, so one can reach
    up to the tuned frequency, or past it, while the other stops short."""
    centre = len(excess) // 2
    count = len(starts)
    if count < 2:
        return None
    cents = np.empty(count)
    for i, (s, n) in enumerate(zip(starts.tolist(), lengths.tolist(), strict=True)):
        part = excess[s : s + n]
        total = float(np.sum(part))
        cents[i] = (
            float(np.sum(part * np.arange(s, s + n))) / total
            if total > 0
            else s + 0.5 * (n - 1)
        )
    # A group is centred on the tuned frequency when its power is centred
    # in the middle half of its core around it (or within ``gap`` of it).
    half_core = 0.5 * (hi_core - lo_core + 1)
    centred = clear & (np.abs(cents - centre) <= np.maximum(0.5 * half_core, gap))
    if np.any(centred):
        return None
    below = np.flatnonzero(clear & (cents < centre))
    above = np.flatnonzero(clear & (cents > centre))
    if not len(below) or not len(above):
        return None
    b = int(below[np.argmax(cents[below])])
    a = int(above[np.argmin(cents[above])])
    d_lo = centre - cents[b]
    d_hi = cents[a] - centre
    similar = 0.25 <= powers[b] / max(powers[a], 1e-30) <= 4.0
    even = abs(d_hi - d_lo) <= 0.25 * (d_lo + d_hi)
    if similar and even and max(d_lo, d_hi) <= reach_bins:
        return b, a
    return None


def _sideband_reach(
    ratio: np.ndarray, band: np.ndarray, peak: int, averages: float, gap: int
) -> Tuple[int, int]:
    """How far (bins) from a carrier line at bin ``peak`` its sidebands
    stand out of the noise: ``(reach, edge)``. ``reach`` is the last bin of
    what stands out once smoothed, contiguous (breaks of at most ``gap``
    bins) from the line's main lobe outwards (0: nothing does); ``edge``
    is where the sidebands themselves end, the last bin within half the
    smoothing that still stands out when smoothed less (a strong sideband
    stands out smoothed up to half the smoothing past its end: a 400 Hz
    tone's AM read 1.6 kHz wide), the reach where nothing does.

    ``ratio`` is the averaged spectrum over the noise density. A carrier's
    modulation (AM's voice, FM's swing, keying) puts the same power either
    side of it, so each bin is averaged with its mirror image about the
    carrier: the sidebands keep their level and the noise, independent on
    the two sides, varies less. What stands out is judged against the
    spread this mirrored spectrum is measured to have (smoothed over 9 and
    33 bins, as single bins are judged), so weak sidebands count without
    taking the noise's ripple for signal. A strong signal elsewhere is
    mirrored too, but only what joins the carrier counts, and only while
    the power is on both sides: where one side holds far more than the
    other (more than noise makes them differ, and the other under a
    quarter of it), that is another station beside the carrier, and the
    sidebands end."""
    n = len(ratio)
    d = np.arange(3, min(peak, n - 1 - peak) + 1)  # past the Hann main lobe
    d = d[band[peak + d] & band[peak - d]]
    if len(d) < 32:
        return 0, 0
    above, below = ratio[peak + d], ratio[peak - d]
    mirrored = 0.5 * (above + below)
    reach = 0
    standing: Dict[int, np.ndarray] = {}  # (the bins standing out, by width)
    for width in (1, 9, 33):
        smooth = _smooth(mirrored, width)
        spread = _spread(smooth, 2.0 * width * averages)
        z = (smooth - 1.0) / spread
        # (Each side spreads sqrt(2) times as much as their mean, their
        # difference twice as much: 3.5 of its standard deviations.)
        up, down = _smooth(above, width) - 1.0, _smooth(below, width) - 1.0
        stronger, weaker = np.maximum(up, down), np.minimum(up, down)
        one_sided = (stronger - weaker > 7.0 * spread) & (weaker < 0.25 * stronger)
        hits = np.flatnonzero((z > 3.5) & ~one_sided)
        standing[width] = d[hits]
        if width == 1 or not len(hits) or d[hits[0]] > 3 + gap:
            continue  # (single bins only place the edge, below)
        breaks = np.flatnonzero(np.diff(d[hits]) > gap)
        last = hits[breaks[0]] if len(breaks) else hits[-1]
        reach = max(reach, int(d[last]))
    edge = reach
    for wider, narrower in ((33, 9), (9, 1)):
        near = standing[narrower]
        near = near[(near <= edge) & (near >= edge - (wider - 1) // 2)]
        if len(near):
            edge = int(near[-1])
    return reach, edge


def _power_span(excess: np.ndarray, offset: int) -> Tuple[int, int]:
    """First and last bin (plus ``offset``) holding the middle 98 % of the
    power in ``excess``."""
    total = float(np.sum(excess))
    if total <= 0 or not len(excess):
        return offset, offset + max(len(excess) - 1, 0)
    cum = np.cumsum(excess) / total
    lo = int(np.searchsorted(cum, 0.01))
    hi = int(np.searchsorted(cum, 0.99))
    return offset + lo, offset + min(hi, len(excess) - 1)


def _format_hz(hz: float) -> str:
    if hz >= 1e6:
        return f"{hz / 1e6:.2f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


class SignalClassifier:
    """
    Automatic signal classifier.

    Uses statistical features of the envelope, the instantaneous frequency
    and the spectrum to say what the signal near 0 Hz is (see the module
    docstring). Give it a channel: complex baseband with the signal of
    interest at (or near) 0 Hz, at ``sample_rate``, with some noise around
    it. About 0.1 to 1 s of samples works best.
    """

    #: Pulses shorter than this are "short pulses" (radar, Mode S).
    SHORT_PULSE_S = 4e-6
    #: Keyed elements this long or longer are Morse code (by hand or not).
    CW_ELEMENT_S = 0.015
    #: A lone "on" stretch this long is a transmission, not keying.
    TRANSMISSION_S = 0.02
    #: A signal narrower than this fraction of the band is cut out of it
    #: (filtered and decimated around itself) before it is judged.
    NARROW_FRACTION = 0.4
    #: Bursts of pulses (Mode S replies, radar) are at most this long.
    PULSE_BURST_S = 5e-3

    def __init__(self, sample_rate: float):
        """
        Initialize classifier.

        Args:
            sample_rate: Sample rate in Hz
        """
        self._sample_rate = float(sample_rate)

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        samples: np.ndarray,
        noise_psd: Optional[float] = None,
        band_hz: Optional[Tuple[float, float]] = None,
        search_hz: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify signal from I/Q samples.

        Args:
            samples: Complex I/Q samples (the signal near 0 Hz).
            noise_psd: The noise density, power per Hz in the samples'
                scale, when the caller knows it (e.g. from the spectrum
                outside the channel, see :func:`noise_psd_from_bin_db`);
                otherwise it is estimated from the passband.
            band_hz: ``(low, high)`` offsets from 0 Hz that the samples'
                channel filter passes, if known: noise and signal are only
                looked for there. Otherwise the passband is read off the
                spectrum (the whole band when no stopband shows).
            search_hz: Only a signal reaching within this many Hz of 0 Hz
                counts (the tuned frequency's signal, not a neighbour);
                otherwise the result is "noise", with the nearest signal's
                offset in ``features["nearest_signal_hz"]``. By default a
                signal within a quarter of the band is taken.

        Returns:
            ClassificationResult with the signal type, kind, features and
            the reasons for the decision.
        """
        x = np.asarray(samples).ravel()
        if not np.iscomplexobj(x):
            x = x.astype(np.complex64)
        x = np.where(np.isfinite(x), x, 0).astype(np.complex64, copy=False)
        n = len(x)
        mag = np.abs(x).astype(np.float64)
        features: Dict[str, Any] = {
            "mean_magnitude": float(np.mean(mag)) if n else 0.0,
            "std_magnitude": float(np.std(mag)) if n else 0.0,
            "max_magnitude": float(np.max(mag)) if n else 0.0,
        }
        if n < 64 or not np.any(mag > 0):
            return self._result(
                "unknown", 0.0, features, ["Too few samples to judge"], 0.0, 0.0, -99
            )

        rate = self._sample_rate
        freqs, psd, segments = welch_psd(x, rate)
        bin_hz = rate / len(psd)
        if band_hz is not None:
            lo, hi = sorted((float(band_hz[0]), float(band_hz[1])))
            band = (freqs >= lo - 0.5 * bin_hz) & (freqs <= hi + 0.5 * bin_hz)
            if np.count_nonzero(band) < 4:
                band = np.ones(len(psd), dtype=bool)
        else:
            band, flat_top = _passband(psd)
            if flat_top and not _gaussian(x, freqs, band, rate):
                # A flat-topped signal with steep sides (RRC-shaped PSK),
                # not a channel filter's noise: the floor is around it.
                band = np.ones(len(psd), dtype=bool)
        band_width = float(np.count_nonzero(band)) * bin_hz
        if noise_psd is not None and noise_psd > 0:
            noise = float(noise_psd)
            features["noise_source"] = "given"
        else:
            noise = noise_floor_psd(psd[band], segments)
            features["noise_source"] = "in-band"
        features.update(
            noise_psd=noise, band_hz=band_width, segments=segments, duration_s=n / rate
        )

        # -- is anything there? -------------------------------------------- #
        spec = self._spectrum_features(
            freqs, psd, band, noise, segments, rate, search_hz
        )
        features.update(spec)
        wide_power = power = mag**2
        kurt_z = _kurtosis_z(power, noise * band_width, band_width / rate)
        features["power_kurtosis_z"] = kurt_z
        spectral = spec["presence_z"] >= 6.0 and spec["presence_excess_db"] >= 1.5
        present = spectral or abs(kurt_z) >= 10.0
        if present and spec.get("nearest_signal_hz") is not None:
            present = False  # a signal, but not at the tuned frequency
        if not present:
            evidence = ["Nothing stands above the noise in the channel"]
            nearest = spec.get("nearest_signal_hz")
            if nearest is not None:
                evidence = [
                    "Nothing at the tuned frequency; the nearest signal is "
                    f"{_format_hz(abs(nearest))} "
                    f"{'above' if nearest > 0 else 'below'} it"
                ]
            elif spec["presence_z"] >= 3.0:
                evidence.append("A faint trace, too weak to judge")
            conf = self._noise_confidence(spec["presence_z"])
            snr = float(spec["snr_db"])
            return self._result("noise", conf, features, evidence, 0.0, 0.0, snr)

        # -- pulses as wide as the band (radar, Mode S): judge them whole --- #
        centre = float(spec["center_offset_hz"])
        width = float(spec["bandwidth_hz"])
        lo_edge = float(freqs[band][0]) if np.any(band) else -rate / 2
        hi_edge = float(freqs[band][-1]) if np.any(band) else rate / 2
        keying: Optional[Dict[str, Any]] = None
        if kurt_z >= 10.0:
            wide = self._keying(wide_power, noise * band_width, rate, band_width)
            wf = wide["features"]
            pulses = bool(wide["keyed"]) and (
                (
                    float(wf["on_median_s"]) < 3.0 * self.SHORT_PULSE_S
                    and int(wf.get("on_runs_real", 0)) >= 8
                )
                or float(wf.get("pulse_dips", 0.0)) >= 0.1
            )
            if (
                wide["keyed"]
                and not pulses
                and float(wf["on_median_s"]) <= self.PULSE_BURST_S
            ):
                # Bursts seen only averaged (pulses near the noise, like
                # Mode S at 10 dB): pulses as wide as the band when their
                # power spreads over most of it (a keyed carrier's doesn't).
                spread = _burst_band_fraction(
                    x, wide["on"], noise, rate, lo_edge, hi_edge, centre, width
                )
                wf["burst_band_fraction"] = spread
                pulses = spread >= 0.25
            if pulses:
                # The pulses must be the tuned signal's, not a strong burst
                # elsewhere in the band: where their power is centred.
                on_w = wide["on"]
                steps = (x[1:] * np.conj(x[:-1]))[on_w[1:] & on_w[:-1]]
                where = float(np.angle(np.sum(steps))) * rate / (2.0 * math.pi)
                reach = float(search_hz) if search_hz is not None else 0.25 * band_width
                if abs(where) <= max(reach, 2.0 * bin_hz):
                    keying = wide
        # -- otherwise cut the signal out of the band (more SNR per sample) -- #
        if keying is None and 0.0 < width < self.NARROW_FRACTION * band_width:
            half = max(0.75 * width, 3.0 * bin_hz, 60.0)
            # Not past the band's edges (the channel filter's stopband).
            half = min(half, max(hi_edge - centre, centre - lo_edge, 3.0 * bin_hz))
            # (The capture's spectrum, kept for a second cut-out.)
            spectrum = np.fft.fft(x)
            xs, rate_n = _narrow(x, rate, centre, half, spectrum)
            noise_power = noise * 2.0 * half
            features["narrowed_hz"] = 2.0 * half
        else:
            spectrum = None
            xs, rate_n = x, rate
            centre = 0.0
            half = 0.5 * band_width
            noise_power = noise * band_width
            features["narrowed_hz"] = 0.0
        features["analysis_rate"] = rate_n
        power = np.abs(xs).astype(np.float64) ** 2

        # -- keying (CW, OOK, pulses, a transmission starting) --------------- #
        if keying is None:
            keying = self._keying(power, noise_power, rate_n, 2.0 * half)
            if (
                keying["keyed"]
                and rate_n < rate
                and float(keying["features"].get("on_median_s", 1.0))
                <= self.PULSE_BURST_S
            ):
                # Bursts found in the cut-out: are they pulses as wide as
                # the band (Mode S near the noise), seen in the whole band?
                pick = (np.arange(n) * (rate_n / rate)).astype(np.int64)
                on_wide = keying["on"][np.minimum(pick, len(keying["on"]) - 1)]
                keying["features"]["burst_band_fraction"] = _burst_band_fraction(
                    x, on_wide, noise, rate, lo_edge, hi_edge, centre, width
                )
        features.update(keying["features"])
        on = keying["on"] if keying["keyed"] else None

        # -- the signal itself: envelope, frequency, PSK lines --------------- #
        width = max(width, 1.0)
        steady_on = on
        if on is not None:
            # The envelope away from the keying's edges (smeared over a few
            # times rate / width samples): a keyed carrier's own envelope
            # is steady, its ramps are not modulation.
            guard = int(math.ceil(2.0 * rate_n / max(2.0 * half, 1.0)))
            off_near = np.convolve(
                (~on).astype(np.int32), np.ones(2 * guard + 1), "same"
            )
            inner = on & (off_near == 0)
            if np.count_nonzero(inner) >= max(16, 0.25 * np.count_nonzero(on)):
                steady_on = inner
        on_power = power[steady_on] if steady_on is not None else power
        n_eff = 2.0 * half * len(on_power) / rate_n
        env = self._envelope_features(on_power, noise_power, n_eff)
        rho = float(env["signal_power"]) / max(noise_power, 1e-30)
        if on is None and features["narrowed_hz"] > 0.0:
            # A carrier can hold nearly all of the power and have sidebands
            # far wider than 99 % of it (AM's voice): its envelope is judged
            # over all of them (keying, frequency and phase where most of
            # its power is, as cut out above).
            limit = max(hi_edge - centre, centre - lo_edge, 3.0 * bin_hz)
            over_sidebands = self._sideband_envelope(
                x, rate, features, centre, half, limit, spectrum
            )
            if over_sidebands is not None:
                env = over_sidebands
        features.update(env)
        features["band_snr_db"] = 10.0 * math.log10(max(rho, 1e-30))
        snr_db = 10.0 * math.log10(max(env["signal_power"], 1e-30) / (noise * width))
        features.update(
            self._frequency_features(xs, rho, width, rate_n, 2.0 * half, on)
        )
        features.update(
            self._psk_features(xs[on] if on is not None else xs, rate_n, width)
        )

        trace = snr_db < 3.0 and float(spec["presence_excess_db"]) < 6.0
        if trace or not (spectral or keying["keyed"]):
            # A trace of something, too weak to say what it is. (Or only
            # the power's spread said something is there, and no keying
            # shows what: no spectrum peak clearly stands out.)
            evidence = ["Only a trace of a signal, too weak to judge"]
            return self._result("noise", 0.35, features, evidence, 0.0, 0.0, snr_db)
        kind, conf, evidence = self._decide(features, keying, snr_db)
        if kind == "carrier":
            # A line: as narrow as the analysis resolves, its SNR in that.
            width = float(features["bandwidth_hz"])
            snr_db = float(features.get("line_snr_db", snr_db))
        elif kind == "am" and float(features["bandwidth_hz"]) > width:
            # (Its sidebands' width, see _decide.)
            width = float(features["bandwidth_hz"])
            snr_db = 10.0 * math.log10(
                max(env["signal_power"], 1e-30) / (noise * width)
            )
        return self._result(
            kind,
            conf,
            features,
            evidence,
            width,
            float(features["center_offset_hz"]),
            snr_db,
        )

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #

    def _keying(
        self, power: np.ndarray, noise_power: float, rate: float, width: float
    ) -> Dict[str, Any]:
        """On/off structure of the power envelope.

        Keyed when the power switches (with hysteresis) between the noise
        and a steady level well above it: CW, OOK, pulses, or a
        transmission starting or ending in the block. ``on`` marks the
        samples to judge the modulation on (all of them for a continuous
        signal). Tried on the power as it is, then averaged over 4 and 16
        samples (a weak but slow keying, like CW near the noise, shows
        only once the noise is averaged down).
        """
        result: Dict[str, Any] = {}
        for span in (1, 4, 16):
            if span > 1 and len(power) < 64 * span:
                break
            result = self._keying_at(power, noise_power, rate, width, span)
            if result["keyed"]:
                break
        return result

    def _keying_at(
        self,
        power: np.ndarray,
        noise_power: float,
        rate: float,
        width: float,
        span: int,
    ) -> Dict[str, Any]:
        """:meth:`_keying` with the power averaged over ``span`` samples."""
        n = len(power)
        features: Dict[str, Any] = {
            "on_snr_db": -99.0,
            "on_fraction": 1.0,
            "keying_span": span,
        }
        result: Dict[str, Any] = {
            "keyed": False,
            "on": np.ones(n, dtype=bool),
            "features": features,
        }
        noise_power = max(float(noise_power), 1e-30)
        level = _moving_mean(power, span) if span > 1 else power
        # Noise alone (exponential power, or its average over ``span``)
        # passes ``high`` about once in n / 10 samples (at most 1e-3, at
        # least 1e-5 of them); once on, the signal stays on until it falls
        # near the noise.
        p_fa = 1.0 / min(max(n / 10.0, 1e3), 1e5)
        high_ratio = _gamma_quantile(1.0 - p_fa, float(span))
        if span == 1:
            high_ratio = math.log(1.0 / p_fa)
        high = noise_power * high_ratio
        low = noise_power * (1.0 + 0.25 * (high_ratio - 1.0))
        events = np.full(n, -1, dtype=np.int8)
        events[level < low] = 0
        events[level > high] = 1
        idx = np.where(events >= 0, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        state = events[idx]
        on = state == 1
        frac = float(np.mean(on))
        features["on_fraction"] = frac
        if np.count_nonzero(on) < 4:
            return result
        on_level = float(np.mean(power[on]))
        features["on_snr_db"] = 10.0 * math.log10(
            max(on_level - noise_power, 1e-30) / noise_power
        )
        if frac > 0.97 or not np.any(~on):
            return result
        # A dip of a sample inside an on stretch is noise, when the signal
        # can't change that fast (its width against the rate).
        close = int(round(0.3 * rate / max(width, 1.0)))
        if close >= 1:
            starts, lengths = _runs(~on)
            inner = (starts > 0) & (starts + lengths < n) & (lengths <= close)
            _fill_runs(on, starts[inner], lengths[inner])
        # The gaps' level away from the edges: a narrow signal's keying is
        # smeared over a few times rate / width samples (by the filter that
        # cut it out; a strong one rings longer), and those samples aren't
        # the noise. Their lower quartile, scaled to the median the noise
        # alone would have, is the more robust of the two.
        guard = int(math.ceil(2.0 * rate / max(width, 1.0)))
        quiet = ~on
        if guard > 1:
            near_on = np.convolve(on.astype(np.int32), np.ones(2 * guard + 1), "same")
            if np.count_nonzero(quiet & (near_on == 0)) >= 16:
                quiet &= near_on == 0
        if np.any(quiet):
            gap_power = power[quiet]
            off_level = min(
                float(np.median(gap_power)),
                float(np.percentile(gap_power, 25)) * (math.log(2.0) / _Q25_EXP),
            )
        else:
            off_level = on_level
        on_power = power[on]
        # The steady part of the on stretches (not their edges).
        on_power = on_power[on_power >= 0.5 * float(np.median(on_power))]
        mean_on = float(np.mean(on_power))
        var_on = float(np.var(on_power))
        # The variance a steady carrier plus this noise would have.
        steady_var = 2.0 * max(mean_on - noise_power, 0.0) * noise_power
        steady_var += noise_power**2
        level_spread = math.sqrt(max(var_on - steady_var, 0.0)) / max(
            mean_on - noise_power, 1e-30
        )
        features["on_level_spread"] = level_spread
        # The off samples are the noise: their median power is the noise's
        # (ln 2 of its mean for a single sample, nearer the mean averaged).
        off_ratio = off_level / noise_power
        features["off_is_noise"] = bool(off_ratio < (1.3 if span == 1 else 1.25))
        features["level_separation_db"] = 10.0 * math.log10(
            max(on_level, 1e-30) / max(off_level, 1e-30)
        )
        on_starts, on_lengths = _runs(on)
        off_starts, off_lengths = _runs(~on)
        inner_off = (off_starts > 0) & (off_starts + off_lengths < n)
        on_s = on_lengths / rate
        features["on_runs"] = int(len(on_lengths))
        # The typical on stretch: the one the median "on" sample is in (a
        # click or two beside a long transmission don't make it short).
        features["on_median_s"] = _weighted_median(on_s, on_lengths)
        features["on_max_s"] = float(np.max(on_s)) if len(on_s) else 0.0
        features["off_median_s"] = (
            float(np.median(off_lengths[inner_off])) / rate
            if np.any(inner_off)
            else 0.0
        )
        # Noise alone crosses ``high`` now and then for a sample or two:
        # real keying has on stretches longer than that (unless the signal
        # is as wide as the band, like Mode S pulses at the full rate, which
        # must then stand far out of the noise instead).
        wide = rate < 2.5 * max(width, 1.0) and span == 1
        min_run = 1 if wide else 3 * span
        real = on_lengths >= min_run
        features["on_runs_real"] = int(np.count_nonzero(real))
        # The stretches that hold something: not the noise's clicks (a
        # sample or two) beside a transmission starting or ending.
        if len(on_lengths):
            longest = float(np.max(on_lengths))
            held = on_lengths >= max(min_run, 0.1 * longest)
            features["on_runs_long"] = int(np.count_nonzero(held))
            # Long enough for a Morse element (a dit at 80 WPM): keying,
            # not the noise's clicks.
            keyed_runs = on_s >= 0.015
            features["on_runs_keyed"] = int(np.count_nonzero(keyed_runs))
            # Those whole inside the block: one cut by its start or end may
            # be the other end's carrier ringing round through the FFT that
            # cut the channel out (:func:`_narrow`), not a key-down.
            inside = (on_starts > 0) & (on_starts + on_lengths < n)
            features["on_runs_keyed_inside"] = int(
                np.count_nonzero(keyed_runs & inside)
            )
        enough = True
        if span == 1:
            expected = n * math.exp(-high_ratio)
            above = int(np.count_nonzero(power > high))
            # Far more samples above ``high`` than noise alone gives: a weak
            # signal that is always there crosses it now and then too, so
            # ten times as many. Bursts standing far above it (four times:
            # noise alone never gets there) need fewer: a capture of a few
            # Mode S replies, cut narrow, holds some 60 such samples.
            strong = on_level >= _STRONG_BURSTS * high
            enough = above >= (4.0 if strong else 10.0) * expected + 20.0
        else:
            # Averaged: the gaps must be real gaps too (not the troughs of
            # a modulated signal), long and at the noise.
            gaps = off_lengths[inner_off]
            enough = (
                len(gaps) > 0
                and float(np.median(gaps)) >= 2.0 * span
                and off_ratio < 1.15
            )
        # Isolated clicks (a strong signal switched on or off elsewhere) are
        # not keying: several on stretches, or a few long enough to hold
        # something.
        on_time = float(np.sum(on_lengths[real])) / rate
        n_real = int(np.count_nonzero(real))
        several = (
            n_real >= 4
            or (n_real >= 2 and on_time >= 2e-3)
            or (n_real >= 1 and features["on_max_s"] >= 0.02)
        )
        keyed = (
            features["off_is_noise"]
            and features["on_snr_db"] >= (12.0 if wide else 3.0)
            and (
                wide
                or level_spread < 0.45
                # Far above the noise, a modulated carrier never dips to
                # it: off at the noise is keying, however the top varies.
                or (features["on_snr_db"] >= 15.0 and level_spread < 0.8)
            )
            and frac >= 1e-3
            and several
            and float(np.sum(on_lengths[real])) >= 0.5 * float(np.sum(on_lengths))
            and enough
        )
        if keyed and span == 1:
            # Trains of short pulses (Mode S, radar) inside the on stretches:
            # deep dips (10 dB below the on level) a few microseconds long.
            dips = on & (power < 0.1 * float(np.median(power[on])))
            d_starts, d_lengths = _runs(dips)
            if len(d_lengths) >= 8:
                features["pulse_dips"] = float(np.count_nonzero(dips)) / max(
                    1, int(np.count_nonzero(on))
                )
                features["pulse_dip_s"] = float(np.median(d_lengths)) / rate
        if keyed:
            result["keyed"] = True
            result["on"] = on
        return result

    def _spectrum_features(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        band: np.ndarray,
        noise: float,
        segments: int,
        rate: float,
        search_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Occupied width, centre, carrier share and SNR, and how clearly
        anything stands above the noise (see :meth:`classify` for
        ``search_hz``)."""
        bin_hz = rate / len(psd)
        k = max(1.0, 0.9 * float(segments))
        out: Dict[str, Any] = {}
        excess = np.where(band, np.maximum(psd - noise, 0.0), 0.0)
        # The strongest local excess (3 bins): its significance in noise
        # standard deviations of an averaged bin, and its size in dB.
        smooth = _smooth(np.where(band, psd, noise), 3)
        ratio = np.where(band, smooth / noise, 0.0)
        best = float(np.max(ratio))
        spread = _spread(ratio[band], 3.0 * k)
        out["presence_z"] = (best - 1.0) / spread
        out["presence_excess_db"] = 10.0 * math.log10(max(best, 1e-30))
        out.update(
            bandwidth_hz=0.0,
            center_offset_hz=0.0,
            carrier_fraction=0.0,
            snr_db=-99.0,
            peak_offset_hz=0.0,
        )
        # The signal is the group of bins standing out of the noise (alone,
        # or smoothed over 9 bins for a wide, weak one) nearest 0 Hz; bins
        # less than a gap apart belong together (the lines of a modulated
        # signal), stray noise bins and other signals in the band don't.
        filled = np.where(band, psd, noise)
        wide = _smooth(filled, 9)
        broad = _smooth(filled, 33)
        # Significance (noise standard deviations, as measured) at each
        # smoothing: a narrow line, a signal a few bins wide, and broad weak
        # sidebands.
        z = np.maximum.reduce(
            [
                (smooth / noise - 1.0) / spread,
                (wide / noise - 1.0) / _spread(wide[band] / noise, 9.0 * k),
                (broad / noise - 1.0) / _spread(broad[band] / noise, 33.0 * k),
            ]
        )
        region = band & (z > 3.5)
        hits = np.flatnonzero(region)
        if not len(hits):
            return out
        gap = max(6, int(0.015 * np.count_nonzero(band)))
        breaks = np.flatnonzero(np.diff(hits) > gap)
        firsts = np.concatenate(([hits[0]], hits[breaks + 1]))
        lasts = np.concatenate((hits[breaks], [hits[-1]]))
        starts = firsts
        lengths = lasts - firsts + 1
        centre_bin = len(psd) // 2
        powers = np.array(
            [
                float(np.sum(excess[s : s + n]))
                for s, n in zip(starts.tolist(), lengths.tolist(), strict=True)
            ]
        )
        # Where each group's power is (its 98 % span): a strong signal's
        # faint skirts reaching the tuned frequency don't make it the one
        # there.
        cores = np.array(
            [
                _power_span(excess[s : s + n], s)
                for s, n in zip(starts.tolist(), lengths.tolist(), strict=True)
            ]
        ).reshape(-1, 2)
        lo_core, hi_core = cores[:, 0], cores[:, 1]
        # Only groups that clearly stand out count when there are any: a
        # stray noise bin near 0 Hz is not the signal at the tuned frequency
        # when the only real ones are elsewhere (nor between FSK's tones).
        peaks = np.array(
            [
                float(np.max(z[s : s + n]))
                for s, n in zip(starts.tolist(), lengths.tolist(), strict=True)
            ]
        )
        clear_ = peaks >= 6.0
        if not np.any(clear_):
            clear_ = np.ones(len(peaks), dtype=bool)
        reach = (
            float(search_hz)
            if search_hz is not None
            else 0.25 * float(np.count_nonzero(band)) * bin_hz
        )
        holds = (lo_core <= centre_bin + gap) & (hi_core >= centre_bin - gap) & clear_
        pair = _tone_pair(centre_bin, powers, lo_core, hi_core, clear_, holds, gap)
        if pair is None:
            pair = _symmetric_pair(
                starts,
                lengths,
                excess,
                powers,
                lo_core,
                hi_core,
                clear_,
                max(reach / bin_hz, float(gap)),
                gap,
            )
        if pair is not None:
            # Tuned between two humps of about the same power either side
            # (the two tones of FSK): one signal.
            b, a = pair
            starts = np.append(starts, starts[b])
            lengths = np.append(lengths, starts[a] + lengths[a] - starts[b])
            powers = np.append(powers, powers[a] + powers[b])
            lo_core = np.append(lo_core, lo_core[b])
            hi_core = np.append(hi_core, hi_core[a])
            holds = np.append(np.zeros(len(holds), dtype=bool), True)
            clear_ = np.append(clear_, True)
        dist = np.where(
            holds,
            0,
            np.minimum(np.abs(lo_core - centre_bin), np.abs(hi_core - centre_bin)),
        )
        near = (dist * bin_hz <= max(reach, gap * bin_hz)) & clear_
        if np.any(holds):
            pick = int(np.flatnonzero(holds)[np.argmax(powers[holds])])
        elif np.any(near):
            pick = int(np.argmax(np.where(near, powers, -1.0)))
        elif search_hz is not None:
            nearest = int(np.argmin(np.where(clear_, dist, np.iinfo(np.int64).max)))
            side = 1.0 if lo_core[nearest] > centre_bin else -1.0
            out["nearest_signal_hz"] = side * float(dist[nearest]) * bin_hz
            return out
        else:
            pick = int(np.argmax(np.where(clear_, powers, -1.0)))
        a0, a1 = int(starts[pick]), int(starts[pick] + lengths[pick])
        weights = np.zeros(len(psd))
        # Bins barely out of the noise count little (a stray noise bin must
        # not set the width).
        weights[a0:a1] = excess[a0:a1] * np.clip((z[a0:a1] - 3.5) / 3.5, 0.0, 1.0)
        signal = float(np.sum(weights)) * bin_hz
        if signal <= 0.0:
            return out
        peak = a0 + int(np.argmax(smooth[a0:a1]))
        out["peak_offset_hz"] = float(freqs[peak])
        # The core: the bins within 20 dB of the peak around it (the 99 %
        # width also counts a keyed carrier's faint key-click sidebands).
        core = smooth[a0:a1] >= 0.01 * float(smooth[peak])
        lo_c = hi_c = peak - a0
        while lo_c > 0 and core[lo_c - 1]:
            lo_c -= 1
        while hi_c < len(core) - 1 and core[hi_c + 1]:
            hi_c += 1
        out["core_width_hz"] = float((hi_c - lo_c + 1) * bin_hz)
        cum = np.cumsum(weights)
        cum /= cum[-1]
        lo = int(np.searchsorted(cum, 0.005))
        hi = int(np.searchsorted(cum, 0.995))
        width = max(bin_hz, (hi - lo + 1) * bin_hz)
        out["bandwidth_hz"] = float(width)
        out["center_offset_hz"] = float(np.sum(freqs * weights) / np.sum(weights))
        # Share of the signal's power in its strongest line (a carrier):
        # the Hann main lobe, +/- 2 bins around the peak, above the level of
        # the bins beside it (a smooth spectrum's top is not a line).
        a, b = max(0, peak - 2), min(len(psd), peak + 3)
        side = np.concatenate(
            (
                excess[max(a0, peak - 8) : max(a0, peak - 2)],
                excess[b : min(a1, peak + 9)],
            )
        )
        floor = float(np.median(side)) if len(side) else 0.0
        line = max(float(np.sum(excess[a:b])) - (b - a) * floor, 0.0) * bin_hz
        whole = float(np.sum(excess[a0:a1])) * bin_hz
        out["carrier_fraction"] = float(min(1.0, line / max(whole, 1e-30)))
        if out["carrier_fraction"] >= 0.25:
            # A carrier: how far its sidebands reach, however faint (AM's
            # voice can hold a few per cent of the power, spread wide), and
            # where they end (the reach runs past a strong one).
            reach_bins, edge_bins = _sideband_reach(psd / noise, band, peak, k, gap)
            out["sideband_hz"] = float(reach_bins * bin_hz)
            out["sideband_edge_hz"] = float(edge_bins * bin_hz)
        out["snr_db"] = 10.0 * math.log10(signal / (noise * width))
        # The strongest line's SNR in the resolution bandwidth (a carrier's
        # own width; its 99 % width above is mostly the noise's bins).
        out["line_width_hz"] = _HANN_ENBW * bin_hz
        out["line_snr_db"] = 10.0 * math.log10(
            max(line, 1e-30) / (noise * _HANN_ENBW * bin_hz)
        )
        return out

    def _sideband_envelope(
        self,
        x: np.ndarray,
        rate: float,
        features: Dict[str, Any],
        centre: float,
        half: float,
        limit: float,
        spectrum: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """:meth:`_envelope_features` over a carrier's sidebands
        (``features["sideband_hz"]`` either side of its line), when they
        reach past ``half`` of the cut-out at ``centre``; None otherwise.
        Never past ``limit`` from ``centre`` (the band's edges).
        ``spectrum``: the FFT of ``x``, when it was taken already."""
        sidebands = float(features.get("sideband_hz", 0.0))
        if sidebands <= 0.0:
            return None
        line_at = abs(float(features.get("peak_offset_hz", centre)) - centre)
        wide_half = min(line_at + sidebands, limit)
        if wide_half <= half:
            return None
        xs, rate_w = _narrow(x, rate, centre, wide_half, spectrum)
        power = np.abs(xs).astype(np.float64) ** 2
        noise_power = float(features["noise_psd"]) * 2.0 * wide_half
        features["sideband_cut_hz"] = 2.0 * wide_half
        n_eff = 2.0 * wide_half * len(power) / rate_w
        return self._envelope_features(power, noise_power, n_eff)

    @staticmethod
    def _envelope_features(
        power: np.ndarray, noise_power: float, n_eff: float
    ) -> Dict[str, Any]:
        """How much the signal's own power varies, from the power moments
        with the noise's share taken out (right at any SNR when the noise
        power is right): ``am_index`` is the standard deviation of the
        signal's power over its mean (0 for FM or a carrier; about 2 x
        depth x rms of the audio for AM). ``am_floor`` is what a constant
        envelope can read by chance with ``n_eff`` independent samples at
        this SNR (three standard errors)."""
        p = np.asarray(power, dtype=np.float64)
        m2 = float(np.mean(p))
        m4 = float(np.mean(p * p))
        s2 = max(float(noise_power), 0.0)
        signal = max(m2 - s2, 1e-30)
        # Signal + independent complex Gaussian noise:
        # E|x|^4 = E|s|^4 + 4 E|s|^2 s2 + 2 s2^2.
        s4 = m4 - 4.0 * signal * s2 - 2.0 * s2 * s2
        am_index = math.sqrt(max(s4 - signal * signal, 0.0)) / signal
        # A constant envelope: the power's variance is 2 S N + N^2, which
        # the sample variance estimates to about sqrt(2 / n) of itself.
        null_var = 2.0 * signal * s2 + s2 * s2
        floor = 3.0 * null_var * math.sqrt(2.0 / max(n_eff, 2.0))
        return {
            "signal_power": signal,
            "am_index": float(am_index),
            "am_floor": float(math.sqrt(floor) / signal),
            "power_kurtosis": m4 / max(m2 * m2, 1e-60),
        }

    def _frequency_features(
        self,
        xs: np.ndarray,
        rho: float,
        width_hz: float,
        rate: float,
        band: float,
        on: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Instantaneous frequency: offset, swing beyond the noise, and
        whether it dwells on two values (FSK). With ``on``, only the steps
        between two "on" samples count (not the gaps, nor the joins)."""
        out: Dict[str, Any] = {
            "mean_freq_hz": 0.0,
            "freq_dev_hz": 0.0,
            "freq_peak_hz": 0.0,
            "freq_p95_hz": 0.0,
            "freq_noise_hz": 0.0,
            "fm_index": 0.0,
            "freq_bimodality": 0.0,
            "freq_middle_fraction": 1.0,
            "freq_jump_kurtosis": 0.0,
        }
        if len(xs) < 8:
            return out
        x = xs.astype(np.complex128)
        prod = x[1:] * np.conj(x[:-1])
        if on is not None:
            prod = prod[on[1:] & on[:-1]]
            if len(prod) < 8:
                return out
        weight = np.abs(prod)
        wsum = float(np.sum(weight))
        if wsum <= 0:
            return out
        mean_step = float(np.angle(np.sum(prod)))
        out["mean_freq_hz"] = mean_step * rate / (2.0 * math.pi)
        # Steps relative to the mean (wrapped), in Hz, weighted by |prod|:
        # samples deep in the noise count little (fewer FM clicks).
        step = np.angle(prod * np.exp(-1j * mean_step)) * rate / (2.0 * math.pi)
        var = float(np.sum(weight * step * step)) / wsum
        # The discriminator's own noise: the phase noise (1 / (2 rho) rad^2)
        # differenced over one sample of noise ``band`` wide, less with the
        # weighting; clicks add more below about 10 dB.
        corr = 1.0 - float(np.sinc(min(band / rate, 1.0)))
        rho = max(rho, 1e-6)
        noise_var = (rate / (2.0 * math.pi)) ** 2 * corr * (0.6 / rho + 1.5 / rho**2)
        dev = math.sqrt(max(var - noise_var, 0.0))
        out["freq_dev_hz"] = dev
        out["freq_dev_raw_hz"] = math.sqrt(var)
        out["freq_noise_hz"] = math.sqrt(noise_var)
        out["fm_index"] = dev / max(0.5 * float(width_hz), 1.0)
        # The shape of the frequency's distribution, from the discriminator
        # averaged over about 1 / width (the fastest the signal's frequency
        # can change), weighted the same way: less noise, same plateaus.
        span = int(max(1, round(rate / max(float(width_hz), 1.0))))
        if span > 1 and len(step) > 4 * span:
            wsm = _moving_mean(weight, span)
            step = _moving_mean(weight * step, span) / np.maximum(wsm, 1e-30)
            weight = wsm
            wsum = float(np.sum(weight))
        # The peak swing (for the FM deviation to listen with): where 99.5 %
        # of the weight is, on the averaged discriminator (its noise mostly
        # averaged out; what is left is taken off). And the swing exceeded
        # 5 % of the time: near the peak for a tone, well below it for
        # voice (what a deviation clips more than 5 % of the time).
        mag = np.abs(step)
        order = np.argsort(mag)
        cum = np.cumsum(weight[order])
        if len(cum) and cum[-1] > 0:
            left = 3.0 * out["freq_noise_hz"] / max(span, 1)
            for key, share in (("freq_peak_hz", 0.995), ("freq_p95_hz", 0.95)):
                at = int(np.searchsorted(cum, share * cum[-1]))
                level = float(mag[order[min(at, len(order) - 1)]])
                out[key] = math.sqrt(max(level * level - left * left, 0.0))
        var_s = float(np.sum(weight * step * step)) / wsum
        if var_s > 1e-12:
            m3 = float(np.sum(weight * step**3)) / wsum
            m4 = float(np.sum(weight * step**4)) / wsum
            skew = m3 / var_s**1.5
            kurt = m4 / var_s**2
            out["freq_bimodality"] = (skew * skew + 1.0) / max(kurt, 1e-12)
            spread = float(np.sum(weight * np.abs(step))) / wsum
            middle = np.abs(step) < 0.5 * spread
            out["freq_middle_fraction"] = float(np.sum(weight * middle)) / wsum
        jumps = np.diff(step)
        if len(jumps) > 8:
            jv = float(np.mean(jumps**2))
            if jv > 0:
                out["freq_jump_kurtosis"] = float(np.mean(jumps**4)) / (jv * jv)
        return out

    @staticmethod
    def _psk_features(xs: np.ndarray, rate: float, width: float) -> Dict[str, Any]:
        """M-th power spectral lines: M-PSK raised to the M-th power is a
        carrier (a line) at M times its offset, here near 0 Hz (the signal
        was centred). FSK's lines sit at +/- M times its shift instead."""
        out: Dict[str, Any] = {"psk_line_2": 0.0, "psk_line_4": 0.0, "psk_line_8": 0.0}
        n = len(xs)
        if n < 64:
            return out
        nfft = 1 << int(min(15, max(6, math.floor(math.log2(n)))))
        seg = xs[:nfft].astype(np.complex128)
        seg = seg / max(math.sqrt(float(np.mean(np.abs(seg) ** 2))), 1e-30)
        window = np.hanning(len(seg))
        freqs = np.fft.fftfreq(nfft, 1.0 / rate)
        for m in (2, 4, 8):
            spec = np.abs(np.fft.fft((seg**m) * window, nfft)) ** 2
            total = float(np.sum(spec))
            k = int(np.argmax(spec))
            near = abs(freqs[k]) <= m * max(0.05 * width, 3.0 * rate / nfft)
            idx = np.arange(k - 2, k + 3) % nfft
            share = float(np.sum(spec[idx])) / max(total, 1e-30)
            out[f"psk_line_{m}"] = share if near else 0.0
        return out

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #

    def _scores(self, f: Dict[str, Any], keyed: bool) -> Tuple[Dict[str, float], bool]:
        """A score (0-1) for every candidate kind; and whether the keying
        is a transmission starting or ending rather than data or Morse."""
        width = float(f.get("bandwidth_hz", 0.0))
        am_index = float(f.get("am_index", 0.0))
        fm_index = float(f.get("fm_index", 0.0))
        dev = float(f.get("freq_dev_hz", 0.0))
        noise_hz = float(f.get("freq_noise_hz", 0.0))
        carrier = float(f.get("carrier_fraction", 0.0))
        bimod = float(f.get("freq_bimodality", 0.0))
        middle = float(f.get("freq_middle_fraction", 1.0))
        jump_kurt = float(f.get("freq_jump_kurtosis", 0.0))
        psk = max(
            _ramp(float(f.get("psk_line_2", 0.0)), 0.03, 0.1),
            _ramp(float(f.get("psk_line_4", 0.0)), 0.03, 0.1),
            _ramp(float(f.get("psk_line_8", 0.0)), 0.05, 0.15),
        )

        # Envelope variation beyond what chance gives at this SNR.
        am_excess = am_index - float(f.get("am_floor", 0.0))
        const_env = 1.0 - _ramp(am_excess, 0.06, 0.2)
        varying_env = _ramp(am_excess, 0.08, 0.2)
        has_carrier = _ramp(carrier, 0.25, 0.55)
        no_carrier = 1.0 - _ramp(carrier, 0.15, 0.4)
        # The frequency swings (beyond its noise), relative to the width.
        clear = _ramp(dev, max(30.0, 0.5 * noise_hz), max(120.0, 1.5 * noise_hz))
        swings = _ramp(fm_index, 0.1, 0.25) * clear
        steady = 1.0 - _ramp(fm_index, 0.08, 0.2) * clear
        # FSK dwells on two values: few samples in between, or (with noise)
        # a frequency that jumps rather than glides. A tone's FM is
        # bimodal too (arcsine), but glides through the middle.
        two_tone = clear * max(
            _ramp(bimod, 0.5, 0.7) * (1.0 - _ramp(middle, 0.1, 0.18)),
            _ramp(bimod, 0.45, 0.6)
            * _ramp(jump_kurt, 6.0, 12.0)
            * (1.0 - _ramp(middle, 0.15, 0.25)),
        )
        core = float(f.get("core_width_hz", width))
        narrow = 1.0 - _ramp(min(width, core), 400.0, 1500.0)
        # Phase keying holds the phase still, then jumps: most frequency
        # steps are near zero (FM glides; noise alone gives about 0.3).
        jumpy = _ramp(middle, 0.45, 0.6)

        scores: Dict[str, float] = {}
        transmission = False
        if keyed:
            on_s = float(f.get("on_median_s", 0.0))
            on_max = float(f.get("on_max_s", 0.0))
            # (Clicks of the noise in the gap don't count as stretches.)
            runs = int(f.get("on_runs_long", f.get("on_runs", 0)))
            total_on = float(f.get("on_fraction", 0.0)) * float(f.get("duration_s", 0))
            transmission = (
                runs <= 3
                and on_max >= self.TRANSMISSION_S
                and (total_on <= 0 or on_max >= 0.7 * total_on)
                # A narrow carrier switched on more than once at one steady
                # level is keyed (a Morse element or two before a long
                # one, one of them whole), not a transmission starting or
                # ending (whose voice varies the level).
                and not (
                    narrow > 0.5
                    and int(f.get("on_runs_keyed", 0)) >= 2
                    and int(f.get("on_runs_keyed_inside", 0)) >= 1
                    and float(f.get("on_level_spread", 1.0)) < 0.12
                )
            )
        if keyed and not transmission:
            short = 1.0 - _ramp(on_s, self.SHORT_PULSE_S, 3.0 * self.SHORT_PULSE_S)
            if int(f.get("on_runs_real", 0)) < 8:
                short *= 0.3  # a handful of pulses: clicks, not a pulse train
            dips = float(f.get("pulse_dips", 0.0))
            dip_s = float(f.get("pulse_dip_s", 1.0))
            train = _ramp(dips, 0.1, 0.25) * (
                1.0 - _ramp(dip_s, self.SHORT_PULSE_S, 3.0 * self.SHORT_PULSE_S)
            )
            # Bursts whose power fills the band: trains of pulses too short
            # to resolve near the noise (Mode S replies at 10 dB).
            spread = _ramp(float(f.get("burst_band_fraction", 0.0)), 0.2, 0.35)
            short = max(short, train, spread)
            slow = _ramp(on_s, 0.5 * self.CW_ELEMENT_S, self.CW_ELEMENT_S)
            scores["pulsed"] = short
            scores["cw"] = (1.0 - short) * slow * narrow * steady
            scores["ook"] = (
                (1.0 - short) * (1.0 - slow * narrow) * steady * (1.0 - two_tone)
            )
            scores["fsk"] = (1.0 - short) * two_tone * const_env
            scores["fm"] = (1.0 - short) * swings * const_env * (1.0 - two_tone) * 0.8
            return scores, transmission
        scores["carrier"] = const_env * steady * narrow * has_carrier
        scores["fm"] = const_env * swings * (1.0 - two_tone) * (1.0 - jumpy)
        scores["fsk"] = const_env * two_tone
        scores["am"] = varying_env * has_carrier * steady
        scores["ssb"] = (
            varying_env
            * no_carrier
            * (1.0 - psk)
            * (1.0 - two_tone)
            * _ramp(width, 300.0, 1000.0)
            * (1.0 - _ramp(width, 4500.0, 8000.0))
        )
        scores["psk"] = (
            psk * no_carrier * (1.0 - two_tone) * max(jumpy, 1.0 - swings * const_env)
        )
        if transmission:
            for key in scores:
                scores[key] *= 0.9  # judged on part of the block
        return scores, transmission

    def _decide(
        self, f: Dict[str, Any], keying: Dict[str, Any], snr_db: float
    ) -> Tuple[str, float, List[str]]:
        """Score every candidate kind, pick the best, and say why."""
        scores, transmission = self._scores(f, bool(keying["keyed"]))
        keyed = bool(keying["keyed"]) and not transmission
        f["transmission"] = transmission
        f["scores"] = {k: round(float(v), 3) for k, v in scores.items()}
        if not scores or max(scores.values()) < 0.2:
            conf = 0.25 * _ramp(snr_db, 3.0, 15.0)
            return "unknown", conf, ["The features don't clearly match a known kind"]
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        kind, best = ranked[0]
        second_kind, second = ranked[1] if len(ranked) > 1 else ("", 0.0)
        margin = 1.0 - 0.7 * (second / best if best > 0 else 1.0)
        band_snr = float(f.get("band_snr_db", snr_db))
        if kind == "carrier":
            # A line: its width is the resolution's, its SNR in that.
            snr_db = max(snr_db, float(f.get("line_snr_db", snr_db)))
            f["bandwidth_hz"] = float(f.get("line_width_hz", f["bandwidth_hz"]))
        elif kind == "am":
            # Its voice reaches as far as its sidebands do (the carrier
            # holds nearly all of its power, so 99 % of it is narrower).
            wide = 2.0 * float(f.get("sideband_edge_hz", 0.0))
            if wide > float(f["bandwidth_hz"]) > 0.0:
                snr_db -= 10.0 * math.log10(wide / float(f["bandwidth_hz"]))
                f["bandwidth_hz"] = wide
        snr_quality = 0.3 + 0.7 * _ramp(min(snr_db, band_snr + 10.0), 3.0, 18.0)
        conf = float(min(MAX_CONFIDENCE, best * margin * snr_quality))
        if kind == "carrier":
            # Weak modulation can hide under the noise: never sure there is
            # none, and less sure the less sidebands 10 dB below the carrier
            # would show in a 6 kHz voice channel around it.
            in_voice = float(f.get("line_snr_db", snr_db)) + 10.0 * math.log10(
                max(float(f.get("line_width_hz", 1.0)), 1e-3) / 6e3
            )
            conf = min(conf, 0.45 + 0.3 * _ramp(in_voice, 6.0, 16.0))
            if transmission:
                # Switched on or off while listening: a key held down for a
                # while (CW) looks the same; never sure it isn't keyed.
                conf = min(conf, 0.6)
        evidence = self._evidence(kind, f, keyed, transmission)
        if second_kind and second > 0.5 * best:
            evidence.append(
                f"Could also be {KIND_LABELS.get(second_kind, second_kind)}"
            )
        return kind, conf, evidence

    @staticmethod
    def _evidence(
        kind: str, f: Dict[str, Any], keyed: bool, transmission: bool
    ) -> List[str]:
        out: List[str] = []
        width = float(f.get("bandwidth_hz", 0.0))
        on_s = float(f.get("on_median_s", 0.0))
        if keyed:
            if kind == "pulsed" and float(f.get("burst_band_fraction", 0.0)) >= 0.25:
                out.append(
                    f"Bursts about {on_s * 1e6:.0f} µs long of pulses as wide "
                    "as the band"
                )
            elif kind == "pulsed":
                out.append(f"Pulses about {on_s * 1e6:.1f} µs long")
            elif kind == "cw":
                out.append(f"Keyed on and off; elements about {on_s * 1e3:.0f} ms")
            elif on_s < 0.01:
                out.append(f"Keyed on and off; pulses about {on_s * 1e6:.0f} µs")
            else:
                out.append(f"Keyed on and off; bursts about {on_s * 1e3:.0f} ms")
        elif transmission:
            out.append("The transmission starts or stops during the sample")
        dev = float(f.get("freq_dev_hz", 0.0))
        if kind == "fm":
            out.append(
                f"Constant envelope; the frequency swings about ±{dev / 1e3:.1f} kHz"
            )
        elif kind == "fsk":
            out.append(f"The frequency jumps between two values (±{dev / 1e3:.1f} kHz)")
        elif kind == "am":
            depth = float(f.get("am_index", 0.0))
            out.append(f"A carrier with sidebands; the power varies {depth:.0%}")
        elif kind == "ssb":
            out.append("The envelope varies and there is no carrier")
        elif kind == "carrier":
            out.append("A steady carrier with no modulation")
        elif kind == "psk":
            out.append(
                "Phase keyed: a line appears when the signal is raised to a power"
            )
        if width > 0:
            out.append(f"Occupies about {_format_hz(width)}")
        return out

    @staticmethod
    def _noise_confidence(z: float) -> float:
        """How sure "no signal" is: the cleaner the spectrum, the surer."""
        conf = 0.9 - 0.5 * _ramp(z, 2.5, 6.0)
        return float(min(MAX_CONFIDENCE, max(0.2, conf)))

    def _result(
        self,
        kind: str,
        confidence: float,
        features: Dict[str, Any],
        evidence: Sequence[str],
        bandwidth: float,
        centre: float,
        snr_db: float,
    ) -> ClassificationResult:
        signal_type, modulation = _KIND_TYPES.get(kind, _KIND_TYPES["unknown"])
        if kind == "psk":
            modulation = self._psk_order(features)
        snr = float(snr_db) if math.isfinite(float(snr_db)) else -99.0
        features["snr_db"] = snr
        features["kind"] = kind
        return ClassificationResult(
            signal_type=signal_type,
            modulation=modulation,
            confidence=float(min(MAX_CONFIDENCE, max(0.0, confidence))),
            bandwidth_hz=float(bandwidth),
            center_offset_hz=float(centre),
            snr_db=snr,
            features=features,
            kind=kind,
            evidence=list(evidence),
        )

    @staticmethod
    def _psk_order(features: Dict[str, Any]) -> Optional[ModulationType]:
        """BPSK (a squared line), QPSK (a 4th power line, the square has
        none) or neither."""
        line2 = float(features.get("psk_line_2", 0.0))
        line4 = float(features.get("psk_line_4", 0.0))
        if line2 >= 0.05 and line2 >= 0.5 * line4:
            return ModulationType.BPSK
        if line4 >= 0.05:
            return ModulationType.QPSK
        return None

    # ------------------------------------------------------------------ #
    # Kept for callers of the old helpers
    # ------------------------------------------------------------------ #

    def _estimate_snr(self, samples: np.ndarray) -> float:
        """SNR (dB) of ``samples`` in the signal's occupied bandwidth."""
        return float(self.classify(samples).snr_db)

    def _estimate_bandwidth(self, samples: np.ndarray) -> float:
        """Occupied bandwidth (99 % of the signal's power), Hz."""
        return float(self.classify(samples).bandwidth_hz)
