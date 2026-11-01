"""
Frequency sweeps: plan the receiver's steps across a range, measure each
step's spectrum and find the channels in use.

A receiver sees its whole sample rate at once, so a sweep doesn't visit
every channel: it steps the receiver (the LO) by about 80 % of the sample
rate (the usable middle of the span, inside the anti-alias filter's roll
off) and measures every channel of that page from one averaged spectrum.

Shared by the window's Scanner tab, which feeds it the blocks the display
loop reads, and ``sdr-scan scan``, which reads a device itself. Levels are
on the main display's scale: dBFS per bin of a 2048-point Hann-windowed
FFT, so a threshold read off the spectrum means the same here.

No Qt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.band_plan import band_for, preset_for
from ..core.frequency_manager import RX_PRESETS

#: FFT length of every level measurement (the main display's).
FFT_SIZE = 2048
#: The part of the span a page uses (the edges roll off).
USABLE_FRACTION = 0.8
#: Auto threshold: this far above the measured noise floor.
AUTO_MARGIN_DB = 12.0
# Reported frequencies. Narrow signals are measured to well within 1 kHz and
# are rounded to it. A wide signal (broadcast FM) found with a
# broadcast-sized step goes on the 100 kHz channel raster instead: its power
# centroid wanders by up to about 20 kHz with the modulation, so the raw
# value would tune the receiver off the station.
FINE_RASTER_HZ = 1e3
WIDE_RASTER_HZ = 100e3
WIDE_STEP_HZ = 100e3  # steps at least this big are broadcast-channel scans
WIDE_SIGNAL_HZ = 60e3  # occupied width (20 dB down) of a wide signal
# A sweep in steps of at most CHANNEL_STEP_HZ looks for channels (2 m,
# NOAA, 70 cm, airband), and a data signal's centroid (SSTV, AFSK) sits a
# kHz or two off its channel: a reading within PRESET_SNAP_HZ of a built-in
# channel (ISS 145.800, APRS 144.390) is that channel and, in an FM band,
# one within RASTER_SNAP_HZ of a CHANNEL_RASTER_HZ channel is on it (FM
# channels are on 5 kHz; the 12.5 kHz ones between, 2.5 kHz from them,
# are measured well within a kHz, so they keep their own frequency).
CHANNEL_STEP_HZ = 25e3
PRESET_SNAP_HZ = 2.5e3
CHANNEL_RASTER_HZ = 5e3
RASTER_SNAP_HZ = 1.5e3
#: The built-in channels a reading snaps to: not a band's own preset (FM
#: broadcast, ISM), nor a narrower channel's (SSB, CW), whose signal lies
#: beside its dial frequency, or must be tuned to the hertz.
_PRESET_CHANNELS: Tuple[float, ...] = tuple(
    float(p.frequency_hz)
    for p in RX_PRESETS
    if p.bandwidth_hz >= 10e3 and preset_for(p.frequency_hz) is not None
)
#: Bins either side of the LO left out of the search: most receivers show a
#: DC spike there, which would otherwise be a "signal" on every page.
DC_GUARD_BINS = 2
#: How much wider than it is a signal measures (20 dB down, through the Hann
#: window): about 3 bins; a pure tone measures 3 or 4 bins. A minimum width
#: is compared with the width less this, so a carrier isn't a station when
#: the bins are wide (9.8 kHz at 20 MS/s).
_WINDOW_SPREAD_BINS = 3


def power_spectrum_dbfs(samples: np.ndarray, max_segments: int = 4) -> np.ndarray:
    """Hann-windowed power spectrum in dBFS (a full-scale tone reads 0 dB).

    Uses the main display's FFT length and averages up to ``max_segments``
    of the newest segments, so levels match the spectrum whatever block size
    the device delivers. Shorter captures use a single FFT of their length.
    """
    x = np.asarray(samples)
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n >= FFT_SIZE:
        segments = max(1, min(int(max_segments), n // FFT_SIZE))
        blocks = x[n - segments * FFT_SIZE :].reshape(segments, FFT_SIZE)
        size = FFT_SIZE
    else:
        blocks = x.reshape(1, n)
        size = n
    window = np.hanning(size + 1)[:-1]
    gain = float(np.sum(window)) or 1.0
    spec = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = np.mean(np.abs(spec) ** 2, axis=0) / (gain * gain)
    return np.asarray(10.0 * np.log10(power + 1e-24), dtype=np.float64)


class SpectrumAverager:
    """Averages (in power) the spectra of the blocks of one sweep step.

    :meth:`add` takes up to ``max_segments`` of the newest FFT_SIZE
    segments of a block, so a step's measurement steadies over a few
    display frames whatever size they are.
    """

    def __init__(self, max_segments: int = 32):
        self.max_segments = max(1, int(max_segments))
        self._window = np.hanning(FFT_SIZE + 1)[:-1]
        self._scale = 1.0 / float(np.sum(self._window)) ** 2
        self._sum: Optional[np.ndarray] = None
        self.segments = 0

    def reset(self) -> None:
        self._sum = None
        self.segments = 0

    def add(self, samples: np.ndarray) -> int:
        """Add a block; returns the number of segments taken from it."""
        x = np.asarray(samples)
        count = min(self.max_segments, len(x) // FFT_SIZE)
        if count <= 0:
            return 0
        blocks = x[len(x) - count * FFT_SIZE :].reshape(count, FFT_SIZE)
        spec = np.fft.fft(blocks * self._window, axis=1)
        power = np.sum(spec.real**2 + spec.imag**2, axis=0) * self._scale
        self._sum = power if self._sum is None else self._sum + power
        self.segments += count
        return count

    def power_db(self) -> Optional[np.ndarray]:
        """The averaged spectrum (fftshifted, dBFS per bin), or None."""
        if self._sum is None or self.segments <= 0:
            return None
        mean = np.fft.fftshift(self._sum / self.segments)
        return np.asarray(10.0 * np.log10(mean + 1e-24), dtype=np.float64)


@dataclass(frozen=True)
class SweepPage:
    """One receiver step of a sweep: where the LO goes and which channels
    are measured there: ``count`` channels from channel ``first_index`` of
    the range (``start_hz`` every ``step_hz``). The channels are computed
    when needed, so a sweep of millions of channels costs nothing to plan."""

    lo_hz: float
    start_hz: float
    step_hz: float
    first_index: int
    count: int

    @property
    def channels(self) -> np.ndarray:
        """The channel centres (Hz), as :func:`channel_frequencies` gives
        them."""
        index = self.first_index + np.arange(self.count, dtype=np.float64)
        return np.asarray(self.start_hz + index * self.step_hz, dtype=np.float64)

    @property
    def first_hz(self) -> float:
        return self.start_hz + self.first_index * self.step_hz

    @property
    def last_hz(self) -> float:
        return self.start_hz + (self.first_index + self.count - 1) * self.step_hz

    def nearest_channel(self, freq_hz: float) -> float:
        """The page's channel nearest ``freq_hz``."""
        k = int(round((float(freq_hz) - self.first_hz) / self.step_hz))
        k = min(max(k, 0), self.count - 1)
        return self.start_hz + (self.first_index + k) * self.step_hz


@dataclass
class ChannelReading:
    """A signal found on one page."""

    frequency_hz: float  # where to tune (see report_frequency)
    peak_db: float  # strongest bin, dBFS
    noise_db: float  # the page's noise floor per bin, dBFS
    width_hz: float  # occupied width, 20 dB down from the peak

    @property
    def snr_db(self) -> float:
        return self.peak_db - self.noise_db


def channel_count(start_hz: float, end_hz: float, step_hz: float) -> int:
    """How many channels a range has (see :func:`channel_frequencies`)."""
    start, end, step = float(start_hz), float(end_hz), float(step_hz)
    if not (step > 0 and end >= start) or not math.isfinite(end - start):
        return 0
    return int(math.floor((end - start) / step + 0.5)) + 1


def channel_frequencies(start_hz: float, end_hz: float, step_hz: float) -> np.ndarray:
    """The channel centres of a range: from ``start_hz`` every ``step_hz``
    up to and including ``end_hz`` (within half a step)."""
    count = channel_count(start_hz, end_hz, step_hz)
    return float(start_hz) + np.arange(count, dtype=np.float64) * float(step_hz)


def _channels_per_page(step_hz: float, sample_rate: float, usable_fraction: float):
    """``(channels per page, LO shift)`` for a sweep (see plan_sweep)."""
    step = float(step_hz)
    usable = float(usable_fraction) * float(sample_rate)
    shift = min(0.5 * step, (DC_GUARD_BINS + 2) * float(sample_rate) / FFT_SIZE)
    per_page = max(1, int(math.floor((usable - step - 2.0 * shift) / step)) + 1)
    return per_page, shift


def page_count(
    start_hz: float,
    end_hz: float,
    step_hz: float,
    sample_rate: float,
    usable_fraction: float = USABLE_FRACTION,
) -> int:
    """How many receiver steps :func:`plan_sweep` makes (without making
    them)."""
    channels = channel_count(start_hz, end_hz, step_hz)
    if not channels or sample_rate <= 0:
        return 0
    per_page, _shift = _channels_per_page(step_hz, sample_rate, usable_fraction)
    return -(-channels // per_page)


def plan_sweep(
    start_hz: float,
    end_hz: float,
    step_hz: float,
    sample_rate: float,
    usable_fraction: float = USABLE_FRACTION,
) -> List[SweepPage]:
    """The receiver steps (pages) that cover a range.

    Consecutive channels share a page while the page, from half a step
    below its first channel to half a step above its last, fits the usable
    part of the span. A step wider than that gets a page per channel. The
    LO goes in the middle of each page, moved a few bins off a channel
    centre when it lands on one: the bins around the LO are left out of
    the search for peaks (the DC spike, see :data:`DC_GUARD_BINS`).
    """
    total = channel_count(start_hz, end_hz, step_hz)
    if not total or sample_rate <= 0:
        return []
    start, step = float(start_hz), float(step_hz)
    per_page, shift = _channels_per_page(step, sample_rate, usable_fraction)
    pages = []
    for first in range(0, total, per_page):
        count = min(per_page, total - first)
        lo = start + (first + 0.5 * (count - 1)) * step  # the middle
        # On a channel centre (an odd count; or one between two, closer
        # than the shift): beside it, not on it.
        nearest = 0.0 if count % 2 else 0.5 * step
        if nearest < shift:
            lo += shift
        pages.append(SweepPage(lo, start, step, first, count))
    return pages


def noise_floor_db(
    power_db: np.ndarray, usable_fraction: float = USABLE_FRACTION
) -> float:
    """The noise floor of a spectrum: the median of its usable bins."""
    power = np.asarray(power_db, dtype=np.float64)
    n = len(power)
    if n == 0:
        return float("nan")
    edge = int(n * (1.0 - float(usable_fraction)) / 2.0)
    middle = power[edge : n - edge] if n - 2 * edge > 0 else power
    return float(np.median(middle))


def report_frequency(
    freq_hz: float, width_hz: float, step_hz: Optional[float]
) -> float:
    """The frequency to list and tune to for a signal measured at
    ``freq_hz`` with occupied width ``width_hz`` in a sweep of ``step_hz``
    (``None`` outside a sweep): see :data:`WIDE_RASTER_HZ` and
    :data:`CHANNEL_STEP_HZ`."""
    if step_hz and step_hz >= WIDE_STEP_HZ and width_hz >= WIDE_SIGNAL_HZ:
        raster = WIDE_RASTER_HZ
    else:
        if step_hz and step_hz <= CHANNEL_STEP_HZ:
            channel = _channel_at(freq_hz)
            if channel is not None:
                return channel
        raster = FINE_RASTER_HZ
    return float(round(freq_hz / raster) * raster)


def _channel_at(freq_hz: float) -> Optional[float]:
    """The channel a channel scan's reading at ``freq_hz`` is on, when it
    is clearly one (see :data:`CHANNEL_STEP_HZ`); else None."""
    freq = float(freq_hz)
    near = [f for f in _PRESET_CHANNELS if abs(f - freq) <= PRESET_SNAP_HZ]
    if near:
        return min(near, key=lambda f: abs(f - freq))
    band = band_for(freq)
    if band is not None and band.mode == "FM":
        channel = round(freq / CHANNEL_RASTER_HZ) * CHANNEL_RASTER_HZ
        if abs(channel - freq) < RASTER_SNAP_HZ:
            return float(channel)
    return None


def _contiguous(above: np.ndarray, k: int) -> Tuple[int, int]:
    """``(lo, hi)``: the run of True values of ``above`` around index ``k``."""
    n = len(above)
    lo = hi = k
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < n - 1 and above[hi + 1]:
        hi += 1
    return lo, hi


def measure_page(
    power_db: np.ndarray,
    lo_hz: float,
    sample_rate: float,
    channels: Sequence[float],
    step_hz: float,
    threshold_db: Optional[float] = None,
    margin_db: float = AUTO_MARGIN_DB,
    usable_fraction: float = USABLE_FRACTION,
    min_width_hz: float = 0.0,
) -> Tuple[List[ChannelReading], float, float]:
    """Find the signals on one page.

    Each channel's slice (half a step either side of its centre) reports
    its strongest bin when that is above the threshold and is a peak, not
    the skirt of a stronger signal next door. A signal is measured whole:
    its centre is the power-weighted centroid of the contiguous bins within
    12 dB of its peak, its width the contiguous bins within 20 dB of the
    peak and clearly (6 dB) above the noise. Signals found by several
    channels (a wide one across narrow channels) are reported once.

    Args:
        power_db: The page's averaged spectrum (fftshifted, dBFS per bin).
        lo_hz: The LO (the spectrum's centre).
        sample_rate: The span.
        channels: Channel centres measured on this page.
        step_hz: The channel spacing.
        threshold_db: Peak level a signal must exceed; None for Auto (the
            noise floor plus ``margin_db``).
        min_width_hz: Signals narrower than this are ignored (a broadcast
            band scan lists stations, not stray carriers), the window's
            spread allowed for.

    Returns:
        ``(readings, noise_db, threshold_db)``, the readings in frequency
        order.
    """
    power = np.asarray(power_db, dtype=np.float64)
    n = len(power)
    if n < 8 or sample_rate <= 0 or not len(channels):
        return [], float("nan"), float("nan")
    noise = noise_floor_db(power, usable_fraction)
    threshold = noise + float(margin_db) if threshold_db is None else threshold_db
    bin_hz = float(sample_rate) / n
    center = n // 2
    half_usable = 0.5 * float(usable_fraction) * float(sample_rate)
    first_usable = max(0, center - int(math.floor(half_usable / bin_hz)))
    last_usable = min(n - 1, center + int(math.floor(half_usable / bin_hz)))
    # Signals are measured inside the usable span; peaks are looked for
    # there too, except right at the LO (the DC spike).
    searchable = np.zeros(n, dtype=bool)
    searchable[first_usable : last_usable + 1] = True
    peaks_allowed = searchable.copy()
    peaks_allowed[max(0, center - DC_GUARD_BINS) : center + DC_GUARD_BINS + 1] = False

    half = max(0.5 * float(step_hz), 0.5 * bin_hz)
    rel = np.asarray(channels, dtype=np.float64) - float(lo_hz)
    starts = np.clip(
        center + np.ceil((rel - half) / bin_hz - 1e-9).astype(int), 0, n - 1
    )
    stops = np.clip(
        center + np.floor((rel + half) / bin_hz + 1e-9).astype(int), 0, n - 1
    )
    # A slice narrower than a bin is the bin its start falls in.
    stops = np.maximum(stops, starts)
    masked = np.where(peaks_allowed, power, -np.inf)

    # Only the slices holding a bin above the threshold can report a signal
    # (on a page of noise, none): find those without visiting every slice
    # (16,000 of them at 20 MS/s in 1 kHz steps). The slices are in
    # frequency order, so those holding bin b are a contiguous run.
    hot = np.flatnonzero(masked > threshold)
    candidates: List[int] = []
    if hot.size:
        first_slice = np.searchsorted(stops, hot, side="left")
        last_slice = np.searchsorted(starts, hot, side="right") - 1
        runs = [
            np.arange(a, b + 1)
            for a, b in zip(first_slice.tolist(), last_slice.tolist(), strict=True)
            if b >= a
        ]
        touched = np.unique(np.concatenate(runs)) if runs else np.empty(0, int)
        # Narrow steps put many slices in one bin: each bin range once.
        ranges = np.unique(np.stack([starts[touched], stops[touched]], axis=1), axis=0)
        for first, last in ranges.tolist():
            segment = masked[first : last + 1]
            k = first + int(np.argmax(segment))
            peak = power[k]
            if not peak > threshold:
                continue
            # The skirt of a stronger signal in the next channel peaks at
            # the slice's edge, still rising: not a signal of this channel.
            if k == first and k > 0 and power[k - 1] > peak:
                continue
            if k == last and k < n - 1 and power[k + 1] > peak:
                continue
            candidates.append(k)

    readings: List[Tuple[int, ChannelReading, Tuple[int, int]]] = []
    offsets = (np.arange(n) - center) * bin_hz
    for k in sorted(set(candidates), key=lambda i: -power[i]):
        # A weaker peak inside a stronger signal already found is part of it.
        if any(lo <= k <= hi for _k, _r, (lo, hi) in readings):
            continue
        peak = float(power[k])
        lo, hi = _contiguous(searchable & (power >= peak - 12.0), k)
        weights = 10.0 ** (power[lo : hi + 1] / 10.0)
        centre = float(np.sum(offsets[lo : hi + 1] * weights) / np.sum(weights))
        floor = max(peak - 20.0, noise + 6.0)
        wlo, whi = _contiguous(searchable & (power >= floor), k)
        width = (whi - wlo + 1) * bin_hz
        if min_width_hz > 0 and width - _WINDOW_SPREAD_BINS * bin_hz < min_width_hz:
            continue
        freq = report_frequency(float(lo_hz) + centre, width, step_hz)
        readings.append(
            (k, ChannelReading(freq, peak, noise, width), (min(lo, wlo), max(hi, whi)))
        )
    found = sorted((r for _k, r, _span in readings), key=lambda r: r.frequency_hz)
    return found, noise, threshold


def same_signal(a_hz: float, a_width_hz: float, b_hz: float, step_hz: float) -> bool:
    """Whether a reading at ``b_hz`` is the signal already listed at
    ``a_hz`` (``a_width_hz`` wide): within three quarters of a step, or
    inside its occupied width."""
    reach = max(0.75 * float(step_hz), 0.5 * float(a_width_hz))
    return abs(float(a_hz) - float(b_hz)) < reach


def format_duration(seconds: float) -> str:
    """``"about 3 s"``, ``"about 2 min"``, ``"about 1 h 20 min"``."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"about {max(1, int(round(s)))} s"
    minutes = int(round(s / 60.0))
    if minutes < 60:
        return f"about {minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"about {hours} h {rest} min" if rest else f"about {hours} h"


__all__ = [
    "AUTO_MARGIN_DB",
    "ChannelReading",
    "FFT_SIZE",
    "SpectrumAverager",
    "SweepPage",
    "USABLE_FRACTION",
    "channel_count",
    "channel_frequencies",
    "format_duration",
    "measure_page",
    "noise_floor_db",
    "page_count",
    "plan_sweep",
    "power_spectrum_dbfs",
    "report_frequency",
    "same_signal",
]
