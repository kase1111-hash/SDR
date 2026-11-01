"""
Bursts: find the transmissions that come and go in the span, and take
apart on/off keyed (OOK) and FSK data the way rtl_433's pulse analyzer
does.

* :class:`BurstDetector` finds bursts in a stream of spectrum rows (dBFS
  per bin, 0 Hz in the middle, any size): a per-bin floor taken from the
  first row (what is on then is part of it) and adapted quickly at first,
  then slowly, to the mean level of the bins around (so noise never stands
  10 dB above it by chance), and under a burst at 1 dB a second (so a
  station that never stops becomes part of it); bins clearly above it
  (10 dB by default, over a few neighbouring bins), grouped into signals
  and followed from row to row. A burst ends when it has been gone for a
  couple of rows. No warm-up, no flood of noise "bursts", and none where
  the floor itself stands above the noise in a channel that never falls
  quiet (an occupied channel: a station's modulation coming and going, or
  a beacon's keying, is not a transmission there).
* :func:`analyze_pulses` takes the I/Q of one burst's channel and measures
  its pulses and gaps, guesses the modulation (pulse width, pulse position,
  PCM / NRZ, Manchester, FSK) and slices the bits, row by row, in hex.
  :func:`format_analysis` writes it up like ``rtl_433 -A``, with the flex
  decoder line to decode it with rtl_433.
* :class:`BurstWatcher` does both on the receiver's raw blocks: it makes
  each row from the whole block (the strongest of several FFTs, so a short
  burst anywhere in the block counts), cuts each burst's channel out while
  it lasts, and hands out a :class:`BurstReport` when it ends.

No Qt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .channelizer import Channelizer
from .classifiers import _gamma_quantile

__all__ = [
    "Burst",
    "BurstDetector",
    "PulseAnalysis",
    "analyze_pulses",
    "analyze_capture",
    "format_analysis",
    "BurstReport",
    "BurstWatcher",
    "analyze_report",
    "frame_rows",
]


# --------------------------------------------------------------------------- #
# Burst detection on spectrum rows
# --------------------------------------------------------------------------- #


@dataclass
class Burst:
    """One burst: when, where, how wide and how strong."""

    start_s: float
    end_s: float
    center_hz: float  # absolute (the row's centre plus the offset)
    width_hz: float
    peak_dbfs: float
    snr_db: float
    rows: int = 1
    #: Offset from the row's centre (Hz), where it was found.
    offset_hz: float = 0.0
    # Accumulators while active.
    lo_hz: float = 0.0
    hi_hz: float = 0.0
    _weight: float = 0.0
    _moment: float = 0.0
    _missing: int = 0
    ident: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _db_to_lin(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


class BurstDetector:
    """Find bursts in a stream of spectrum rows (see the module docstring).

    Args:
        snr_db: How far above the noise floor a bin must be.
        min_width_hz / max_width_hz: Widths of what counts as a burst.
        gap_rows: Rows a burst may be missing and still continue.
        merge_hz: Signals closer than this are one burst.
        settle_rows: Rows of fast adaptation at the start.
        slow_s: Time constant (s) of the floor once settled.
        absorb_db_per_s: How fast the floor rises under a burst: a signal
            that stays on (a station) becomes the floor, a 30 dB one after
            about 20 s at 1 dB/s.
        max_duration_s: A burst still on after this long is reported, so
            the activity list doesn't wait forever (where its channel never
            fell quiet, what follows is a station's, not a new burst).
        occupied_db: No burst starts where the floor stands this far above
            the noise around a station that never stops (or in the next
            bin), nor a weak one whose energy runs into such a place (0:
            anywhere): that channel is always occupied (a broadcast
            station), and its modulation rising and falling there is not a
            transmission. Nor where a channel has not fallen quiet for
            :attr:`ALWAYS_S` and its floor stands half that far above the
            noise (a beacon keying away). A transmission that comes and
            goes (a remote, a repeater between overs) falls back to the
            noise between its bursts, even where the floor still stands
            above it for a while (it rises under them). One standing
            :attr:`OVER_STATION_DB` past the burst threshold above the
            station's floor, outside the bins the station never leaves, is
            a burst all the same (a remote over a steady carrier, a call
            beside one).
    """

    #: Rows between two measurements of the noise under the floor.
    OCCUPANCY_EVERY = 8
    #: Seconds a channel must fall quiet (back near the noise) to count as
    #: free again: longer than the gaps of Morse keying at 12 WPM or more
    #: (a word's 7 dits), shorter than a repeater's pause between overs.
    REST_S = 0.75
    #: Seconds a channel goes on without falling quiet (for
    #: :attr:`REST_S`) before it counts as a station with its floor only
    #: half :attr:`occupied_db` up: Morse code never pauses that long, and
    #: a beacon keying away raises its floor only a few dB.
    ALWAYS_S = 10.0
    #: dB past ``snr_db`` a signal must stand above an occupied channel's
    #: floor (its highest where the station never stops), outside the bins
    #: the station never leaves, to be a burst over it or beside it: a
    #: station's own modulation stays in those bins (the demo's broadcast
    #: stations) or well under that; a remote's packet over a steady
    #: carrier (a spur), or a call beside one, stands far above it.
    OVER_STATION_DB = 10.0

    def __init__(
        self,
        snr_db: float = 10.0,
        min_width_hz: float = 1.5e3,
        max_width_hz: float = 1.5e6,
        gap_rows: int = 2,
        merge_hz: float = 10e3,
        settle_rows: int = 8,
        slow_s: float = 4.0,
        max_duration_s: float = 30.0,
        absorb_db_per_s: float = 1.0,
        occupied_db: float = 4.5,
    ):
        self.snr_db = float(snr_db)
        self.min_width_hz = float(min_width_hz)
        self.max_width_hz = float(max_width_hz)
        self.gap_rows = int(gap_rows)
        self.merge_hz = float(merge_hz)
        self.settle_rows = int(settle_rows)
        self.slow_s = float(slow_s)
        self.max_duration_s = float(max_duration_s)
        self.absorb_db_per_s = float(absorb_db_per_s)
        self.occupied_db = float(occupied_db)
        #: Signals that start in the same row this close are one burst.
        self.pair_hz = 60e3
        self._floor: Optional[np.ndarray] = None  # linear, per bin
        self._rows = 0
        self._last_t: Optional[float] = None
        self._key: Optional[Tuple[int, float, float]] = None
        self.active: List[Burst] = []
        self._next_id = 1
        # Per bin: when its level last stood ``occupied_db`` over the noise,
        # and when it had last been quiet for REST_S (a station that never
        # stops never is; a repeater between its overs is); and when the
        # floor was taken (no channel has been on for longer, as far as
        # anyone knows).
        self._last_on: Optional[np.ndarray] = None
        self._rested_at: Optional[np.ndarray] = None
        self._floor_t = 0.0
        # The noise under the floor (linear, per bin), and the row it was
        # measured at.
        self._noise: Optional[np.ndarray] = None
        self._noise_at = 0

    def reset(self) -> None:
        """Forget the floor and the bursts in progress (a retune)."""
        self._floor = None
        self._rows = 0
        self._last_t = None
        self._key = None
        self.active = []
        self._last_on = None
        self._rested_at = None
        self._noise = None

    @property
    def settled(self) -> bool:
        return self._floor is not None and self._rows >= self.settle_rows

    def noise_floor_db(self) -> Optional[np.ndarray]:
        if self._floor is None:
            return None
        return 10.0 * np.log10(np.maximum(self._floor, 1e-30))

    def process(
        self, row_db: np.ndarray, center_hz: float, span_hz: float, t: float
    ) -> List[Burst]:
        """Add one spectrum row (dBFS per bin, fftshifted) of a span
        ``span_hz`` wide around ``center_hz``, taken at time ``t`` (s).
        Returns the bursts that ended."""
        row = np.asarray(row_db, dtype=np.float64).ravel()
        n = len(row)
        if n < 8 or not span_hz > 0:
            return []
        key = (n, float(center_hz), float(span_hz))
        if key != self._key:
            # Another span: what was measured belongs to the old one.
            ended = self.flush(t)
            self._key = key
            self._floor = None
            self._rows = 0
        else:
            ended = []
        lin = _db_to_lin(np.nan_to_num(row, nan=-200.0, posinf=0.0, neginf=-200.0))
        if self._floor is not None and self._rows > self.settle_rows:
            # The whole floor moved 6 dB or more (the gain changed): start
            # again from this row rather than call the band a burst.
            moved = float(np.median(lin / np.maximum(self._floor, 1e-30)))
            if not 0.25 <= moved <= 4.0:
                ended += self.flush(t)
                self._floor = None
        if self._floor is None:
            # The first row is the floor (averaged over a few bins): what is
            # on now, stations included, is not a burst.
            self._floor = _local_mean(lin, 5)
            self._last_on = np.full(n, float(t))
            self._rested_at = np.full(n, -np.inf)
            self._floor_t = float(t)
            self._rows = 1
            self._last_t = t
            self._noise = None
            return ended
        dt = 1.0 / 30.0 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        self._rows += 1
        bin_hz = span_hz / n
        # Neighbouring bins together (3), against the floor.
        smooth = np.convolve(lin, np.ones(3) / 3.0, mode="same")
        ratio = smooth / np.maximum(self._floor, 1e-30)
        above = ratio > 10.0 ** (self.snr_db / 10.0)
        regions = _regions(above, max(1, int(round(self.merge_hz / bin_hz))))
        occupied = self._occupancy() if regions else None
        runs: Optional[Tuple[np.ndarray, np.ndarray]] = None
        found: List[Tuple[float, float, float, float, float, float, bool]] = []
        for a, b in regions:
            width = (b - a + 1) * bin_hz
            if width < self.min_width_hz or width > self.max_width_hz:
                continue
            seg = lin[a : b + 1]
            freqs = (np.arange(a, b + 1) - n // 2) * bin_hz
            weight = float(np.sum(seg))
            centre = float(np.sum(freqs * seg) / max(weight, 1e-30))
            peak = 10.0 * math.log10(max(float(np.max(seg)), 1e-30))
            floor = 10.0 * math.log10(
                max(float(np.mean(self._floor[a : b + 1])), 1e-30)
            )
            lo = (a - n // 2 - 0.5) * bin_hz
            hi = (b - n // 2 + 0.5) * bin_hz
            busy = False
            if occupied is not None:
                near, core, noise, station, own = occupied
                if np.any(near[a : b + 1]):
                    # An occupied channel's modulation, unless it stands far
                    # above the station past the bins it never leaves: a
                    # transmission over it or beside it (a remote's packet
                    # over a steady carrier, a call next to one).
                    past = seg[~own[a : b + 1]]
                    busy = not len(past) or float(np.max(past)) < float(
                        np.max(station[a : b + 1])
                    ) * 10.0 ** ((self.snr_db + self.OVER_STATION_DB) / 10.0)
                elif peak - floor < self.snr_db + 6.0:
                    # Weak, and its energy runs (never within 6 dB of the
                    # noise) into an occupied channel: that station's
                    # modulation reaching out.
                    if runs is None:
                        runs = _joined_runs(smooth > 4.0 * noise, core)
                    ids, joined = runs
                    busy = bool(np.any(np.isin(ids[a : b + 1], joined)))
            found.append((centre, lo, hi, peak, peak - floor, weight, busy))
        ended += self._track(found, center_hz, t, dt)
        # The floor follows the mean level of the bins around (both ways:
        # following the noise's dips only would sink it into the noise),
        # quickly while settling, then slowly; at once where the level fell
        # 6 dB below it (a station gone, the gain lowered); and under a
        # burst only rising, a decibel or so a second.
        alpha = 0.3 if self._rows <= self.settle_rows else min(1.0, dt / self.slow_s)
        busy = np.zeros(n, dtype=bool)
        for burst in self.active:
            a = int(n // 2 + math.floor(burst.lo_hz / bin_hz)) - 1
            b = int(n // 2 + math.ceil(burst.hi_hz / bin_hz)) + 1
            busy[max(0, a) : min(n, b + 1)] = True
        level = _local_mean(lin, 5)
        floor = self._floor
        # (Gone: well below it over 15 bins, which noise never is by chance.)
        gone = _local_mean(lin, 15) < 0.25 * floor
        rate = np.where(gone, 0.3, alpha)
        quiet = ~busy
        floor[quiet] += rate[quiet] * (level[quiet] - floor[quiet])
        if np.any(busy):
            step = 10.0 ** (self.absorb_db_per_s * dt / 10.0)
            floor[busy] = np.maximum(
                floor[busy], np.minimum(level[busy], floor[busy] * step)
            )
        self._note_rest(level, t)
        return ended

    def _note_rest(self, level: np.ndarray, t: float) -> None:
        """Note the bins whose level stands :attr:`occupied_db` over the
        noise in this row, and those that have been quiet for
        :attr:`REST_S` (with the noise last measured; none before)."""
        noise, on_at, rested = self._noise, self._last_on, self._rested_at
        if noise is None or on_at is None or rested is None:
            return
        if not len(noise) == len(on_at) == len(level):
            return
        on_at[level > noise * 10.0 ** (self.occupied_db / 10.0)] = t
        rested[t - on_at >= self.REST_S] = t

    def _occupancy(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """``(near, occupied, noise, station, own)``: the bins of a
        station that never stops (where the floor stands
        :attr:`occupied_db` above the noise under it, around a bin that
        hasn't been quiet for :attr:`REST_S` lately; or half that far,
        around one not quiet for :attr:`ALWAYS_S`), those with the next bin
        either side, the noise (linear) per bin, each station's floor
        (linear) at its highest in the bins it never leaves, over its near
        bins (0 elsewhere), and those bins with the next either side; None
        when that is off. A repeater's channel between its overs keeps a
        raised floor for a while (it rises under their bursts) but falls
        quiet: not occupied, nor the station's where it runs into one. The
        noise is measured every :attr:`OCCUPANCY_EVERY` rows (the floor
        moves slowly)."""
        floor = self._floor
        if floor is None or self.occupied_db <= 0:
            return None
        from .signal_list import DC_GUARD_BINS, local_floor_db

        if (
            self._noise is None
            or len(self._noise) != len(floor)
            or self._rows - self._noise_at >= self.OCCUPANCY_EVERY
        ):
            floor_db = 10.0 * np.log10(np.maximum(floor, 1e-30))
            self._noise = _db_to_lin(local_floor_db(floor_db))
            self._noise_at = self._rows
        noise = self._noise
        busy = floor > noise * 10.0 ** (self.occupied_db / 10.0)
        core = busy
        rested, now = self._rested_at, self._last_t
        if rested is not None and len(rested) == len(busy) and now is not None:
            # A raised floor counts around a channel on lately (a repeater's
            # falls quiet between its overs; one on when the floor was taken
            # counts as on); one raised half as far around a channel seen
            # never quiet for a long while (a beacon keying away).
            since = now - rested
            always = now - np.maximum(rested, self._floor_t) > self.ALWAYS_S
            always &= floor > noise * 10.0 ** (0.5 * self.occupied_db / 10.0)
            core = (busy & (since > self.REST_S)) | always
            busy = busy | always
        n = len(busy)
        # (Most receivers' DC spike sits at the centre: not a station.)
        dc = slice(max(0, n // 2 - DC_GUARD_BINS), n // 2 + DC_GUARD_BINS + 1)
        busy[dc] = False
        core[dc] = False
        ids, joined = _joined_runs(busy, core)
        busy = np.isin(ids, joined)
        # (A station's modulation reaching past its edge starts there.)
        near = busy.copy()
        near[1:] |= busy[:-1]
        near[:-1] |= busy[1:]
        # (Each station's floor at its highest where it never stops, over
        # its near bins; and those bins, with the next either side.)
        station = np.zeros(n)
        own = core.copy()
        if len(joined):
            run = ids[busy]
            firsts = np.flatnonzero(np.diff(run, prepend=-1))
            held = np.where(core[busy], floor[busy], 0.0)
            top = np.maximum.reduceat(held, firsts)
            station[busy] = np.repeat(top, np.diff(np.append(firsts, len(run))))
            station[1:] = np.maximum(station[1:], station[:-1] * busy[:-1])
            station[:-1] = np.maximum(station[:-1], station[1:] * busy[1:])
            own[1:] |= core[:-1]
            own[:-1] |= core[1:]
        return near, busy, noise, station, own

    def _track(
        self,
        found: Sequence[Tuple[float, float, float, float, float, float, bool]],
        center_hz: float,
        t: float,
        dt: float,
    ) -> List[Burst]:
        seen = set()
        fresh = set()  # the bursts this row starts
        for centre, lo, hi, peak, snr, weight, occupied in found:
            match = None
            for burst in self.active:
                if occupied and id(burst) in fresh:
                    continue  # (a fresh burst doesn't reach into a station)
                if (
                    lo <= burst.hi_hz + self.merge_hz
                    and hi >= burst.lo_hz - self.merge_hz
                ):
                    match = burst
                    break
                # Two signals that start together close by are one
                # transmission (the two tones of FSK).
                if (
                    burst.rows == 1
                    and id(burst) in seen
                    and lo <= burst.hi_hz + self.pair_hz
                    and hi >= burst.lo_hz - self.pair_hz
                ):
                    match = burst
                    break
            if match is None:
                if occupied:
                    continue  # an occupied channel's modulation, not a burst
                match = Burst(
                    start_s=t - dt,
                    end_s=t,
                    center_hz=center_hz + centre,
                    width_hz=hi - lo,
                    peak_dbfs=peak,
                    snr_db=snr,
                    rows=0,
                    offset_hz=centre,
                    lo_hz=lo,
                    hi_hz=hi,
                    ident=self._next_id,
                )
                self._next_id += 1
                self.active.append(match)
                fresh.add(id(match))
            match.rows += 1
            match.end_s = t
            match._missing = 0
            match.lo_hz = min(match.lo_hz, lo)
            match.hi_hz = max(match.hi_hz, hi)
            match.width_hz = match.hi_hz - match.lo_hz
            match.peak_dbfs = max(match.peak_dbfs, peak)
            match.snr_db = max(match.snr_db, snr)
            match._weight += weight
            match._moment += weight * centre
            match.offset_hz = match._moment / max(match._weight, 1e-30)
            match.center_hz = center_hz + match.offset_hz
            seen.add(id(match))
        ended = []
        for burst in list(self.active):
            if id(burst) not in seen:
                burst._missing += 1
            too_long = t - burst.start_s >= self.max_duration_s
            if burst._missing > self.gap_rows or too_long:
                self.active.remove(burst)
                ended.append(burst)
        return ended

    def flush(self, t: Optional[float] = None) -> List[Burst]:
        """End every burst in progress (the receiver stopped)."""
        ended = list(self.active)
        self.active = []
        return ended


def _local_mean(lin: np.ndarray, width: int) -> np.ndarray:
    """The mean of each bin and its neighbours (``width`` bins; mirrored
    at the ends, so the edges aren't pulled down)."""
    n = len(lin)
    pad = min(width // 2, n - 1)
    if pad <= 0:
        return np.asarray(lin, dtype=np.float64).copy()
    padded = np.concatenate((lin[1 : pad + 1][::-1], lin, lin[-pad - 1 : -1][::-1]))
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(padded, kernel, mode="valid")


def _regions(mask: np.ndarray, merge_bins: int) -> List[Tuple[int, int]]:
    """``(first, last)`` bins of the runs of ``mask``, runs closer than
    ``merge_bins`` joined."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > merge_bins)
    firsts = np.concatenate(([idx[0]], idx[breaks + 1]))
    lasts = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(firsts.tolist(), lasts.tolist(), strict=True))


def _joined_runs(mask: np.ndarray, core: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(ids, joined)``: each bin's run of ``mask`` (0 outside it) and the
    runs that contain a ``core`` bin."""
    starts = mask.copy()
    starts[1:] &= ~mask[:-1]
    ids = np.cumsum(starts) * mask
    joined = np.unique(ids[core & mask])
    return ids, joined[joined > 0]


def frame_rows(
    samples: np.ndarray, nfft: int = 1024, max_segments: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """``(strongest, mean)`` power per bin (dBFS, Hann, fftshifted) over
    the ``nfft``-sample segments of a block, so a burst anywhere in the
    block shows in the first. At most ``max_segments`` segments, spread
    over the block, are transformed (all of them at up to 2.4 MS/s)."""
    x = np.asarray(samples, dtype=np.complex64)
    n = len(x)
    if n < nfft:
        nfft = max(16, 1 << int(math.floor(math.log2(max(n, 16)))))
        if n < nfft:
            x = np.concatenate((x, np.zeros(nfft - n, dtype=np.complex64)))
            n = nfft
    count = n // nfft
    blocks = x[: count * nfft].reshape(count, nfft)
    if count > max_segments:
        pick = np.linspace(0, count - 1, max_segments).round().astype(int)
        blocks = blocks[pick]
    window = np.hanning(nfft + 1)[:-1].astype(np.float32)
    gain = float(np.sum(window)) ** 2
    spec = np.fft.fft(blocks * window, axis=1)
    power = (spec.real**2 + spec.imag**2) / gain
    top = np.fft.fftshift(np.max(power, axis=0))
    mean = np.fft.fftshift(np.mean(power, axis=0))
    return (
        10.0 * np.log10(np.maximum(top, 1e-30)),
        10.0 * np.log10(np.maximum(mean, 1e-30)),
    )


# --------------------------------------------------------------------------- #
# Pulse analysis (rtl_433 -A style)
# --------------------------------------------------------------------------- #


@dataclass
class PulseAnalysis:
    """What a burst's pulses say (times in microseconds)."""

    modulation: str  # "OOK PWM", "OOK PPM", "OOK PCM", "FSK PCM", ...
    description: str  # "Pulse width modulation", ...
    pulses_us: List[float]
    gaps_us: List[float]
    pulse_widths: List[Tuple[float, int]]  # (width, count) groups
    gap_widths: List[Tuple[float, int]]
    short_us: float = 0.0
    long_us: float = 0.0
    reset_us: float = 0.0
    gap_limit_us: float = 0.0
    sync_us: float = 0.0
    rows: List[str] = field(default_factory=list)  # bit strings
    level_dbfs: float = 0.0
    snr_db: float = 0.0
    duration_us: float = 0.0
    fsk_shift_hz: float = 0.0
    convention: str = ""

    @property
    def hex_rows(self) -> List[str]:
        return [bits_to_hex(r) for r in self.rows]

    @property
    def summary(self) -> str:
        """``"OOK PWM · 25 pulses · {24}5e3a1c"``."""
        parts = [self.modulation, f"{len(self.pulses_us)} pulses"]
        rows = self.distinct_rows()
        if rows:
            bits, count = rows[0]
            text = f"{{{len(bits)}}}{bits_to_hex(bits)}"
            if count > 1:
                text += f" ×{count}"
            parts.append(text)
        return " · ".join(parts)

    def distinct_rows(self) -> List[Tuple[str, int]]:
        """The bit rows with how often each repeats: the most repeated
        (the surest) first, then the longest."""
        counts: Dict[str, int] = {}
        for row in self.rows:
            if row:
                counts[row] = counts.get(row, 0) + 1
        return sorted(
            counts.items(),
            key=lambda item: (len(item[0]) < 8, -item[1], -len(item[0])),
        )


def bits_to_hex(bits: str) -> str:
    """``"0101 1110"`` → ``"5e"`` (padded with zeros to whole nibbles)."""
    bits = "".join(ch for ch in bits if ch in "01")
    if not bits:
        return ""
    pad = (-len(bits)) % 4
    padded = bits + "0" * pad
    return "".join(f"{int(padded[i : i + 4], 2):x}" for i in range(0, len(padded), 4))


def _cluster(values: Sequence[float], tolerance: float = 0.25) -> List[List[float]]:
    """Group widths that are within ``tolerance`` (relative) of each other
    (rtl_433's histogram bins)."""
    groups: List[List[float]] = []
    total = 0.0
    for v in sorted(float(x) for x in values):
        if groups and v <= total / len(groups[-1]) * (1.0 + tolerance):
            groups[-1].append(v)
            total += v
        else:
            groups.append([v])
            total = v
    return groups


def _hysteresis(power: np.ndarray, low: float, high: float) -> np.ndarray:
    n = len(power)
    events = np.full(n, -1, dtype=np.int8)
    events[power < low] = 0
    events[power > high] = 1
    idx = np.where(events >= 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    state = events[idx]
    state[: int(np.argmax(events >= 0))] = 0
    return state == 1


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def _cut_out(
    x: np.ndarray, rate: float
) -> Tuple[np.ndarray, float, float, Optional[float]]:
    """The burst cut out of ``x`` (all it occupies: both tones of FSK, with
    some room) and decimated: ``(samples, rate, width, noise power)``, the
    noise power per sample of the samples returned when they were cut out
    (None when ``x`` is returned as it is)."""
    from .classifiers import _narrow, _visible_band, noise_floor_psd, welch_psd

    freqs, psd, segments = welch_psd(x, rate)
    # The noise where the capture's channel filter passes it (its stopband
    # is not the noise).
    band = _visible_band(psd)
    noise = noise_floor_psd(psd[band], segments)
    k = max(1.0, 0.9 * float(segments))
    smooth = np.convolve(psd, np.ones(3) / 3.0, mode="same")
    z = np.where(band, (smooth / max(noise, 1e-30) - 1.0) * math.sqrt(3.0 * k), 0.0)
    # What stands out, the barely significant (stray noise bins) counting
    # little.
    excess = np.maximum(psd - noise, 0.0) * np.clip((z - 3.5) / 3.5, 0.0, 1.0)
    total = float(np.sum(excess))
    if total <= 0.0 or not np.any(z >= 6.0):
        return x, rate, rate, None  # nothing clearly there: noise stays noise
    # The middle 98 % of what stands out (its midpoint: FSK's tones may not
    # be sent equally often).
    cum = np.cumsum(excess) / total
    lo = int(np.searchsorted(cum, 0.01))
    hi = int(min(np.searchsorted(cum, 0.99), len(psd) - 1))
    bin_hz = rate / len(psd)
    width = float((hi - lo + 1) * bin_hz)
    centre = 0.5 * float(freqs[lo] + freqs[hi])
    half = min(max(0.75 * width, 5e3), 0.45 * rate)
    if half >= 0.45 * rate:
        return x, rate, width, None
    y, new_rate = _narrow(x, rate, centre, half)
    # (The cut's raised-cosine edges pass about 1.9 half widths of noise.)
    return y, new_rate, width, noise * 1.9 * half


def _clean_runs(on: np.ndarray, shortest: int) -> np.ndarray:
    """``on`` without gaps, then pulses, shorter than ``shortest`` samples
    (noise glitches at the edges and inside a pulse)."""
    on = on.copy()
    if shortest <= 1:
        return on
    starts, lengths = _runs(~on)
    inner = (starts > 0) & (starts + lengths < len(on)) & (lengths < shortest)
    on |= _span_mask(len(on), starts[inner], lengths[inner])
    starts, lengths = _runs(on)
    short = lengths < shortest
    on &= ~_span_mask(len(on), starts[short], lengths[short])
    return on


def _span_mask(n: int, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """A mask of ``n`` samples, True over the given runs (vectorised)."""
    marks = np.zeros(n + 1, dtype=np.int64)
    np.add.at(marks, starts, 1)
    np.add.at(marks, starts + lengths, -1)
    return np.cumsum(marks[:-1]) > 0


def analyze_pulses(
    iq: np.ndarray, rate: float, noise_power: Optional[float] = None
) -> Optional[PulseAnalysis]:
    """Measure the pulses of a burst's channel (complex baseband at
    ``rate``, the burst the strongest thing in it) and slice its bits.
    None when no pulses stand out. ``noise_power``: the noise power per
    sample in ``iq``, when the caller knows it."""
    x = np.asarray(iq, dtype=np.complex64)
    if len(x) < 64 or not rate > 0:
        return None
    x, rate, width, cut_noise = _cut_out(x, rate)
    if cut_noise is not None:
        noise_power = cut_noise
    n = len(x)
    if n < 32:
        return None
    power = (x.real.astype(np.float64) ** 2) + (x.imag.astype(np.float64) ** 2)
    # Average over half the time the signal's bandwidth allows it to change
    # (at least 4 us): the noise is averaged, the keying kept.
    span = max(1, int(round(rate * max(4e-6, 0.5 / max(width, 1.0)))))
    if span > 1:
        c = np.cumsum(np.concatenate(([0.0], power)))
        smooth = np.empty_like(power)
        smooth[span - 1 :] = (c[span:] - c[:-span]) / span
        smooth[: span - 1] = smooth[span - 1]
    else:
        smooth = power
    noise = float(np.percentile(smooth, 10))
    noise = max(noise, 1e-30)
    top = float(np.percentile(smooth, 99))
    # Noise alone (exponential power, averaged over ``span`` samples, about
    # half of them independent) reaches its 99.9th percentile once in a
    # thousand samples: pulses stand twice that far out, and 9 dB at least.
    # (Its mean from the spectrum when known: a burst that fills most of
    # the capture leaves too little noise for the quantiles to tell.)
    k = max(1.0, 0.5 * span)
    if noise_power is not None and noise_power > 0:
        mean_noise = float(noise_power)
    else:
        mean_noise = noise / _gamma_quantile(0.10, k)
    if top < 8.0 * noise or top < 2.0 * _gamma_quantile(0.999, k) * mean_noise:
        return None  # nothing to measure
    # Edges at half amplitude (a quarter of the pulses' power), where a
    # pulse's width is measured, with some hysteresis; never in the noise.
    high = max(0.35 * top, 4.0 * noise)
    low = max(0.18 * top, 2.0 * noise)
    if low >= high:
        mid = math.sqrt(top * noise)
        low, high = mid / 2.0, mid * 2.0
    on = _hysteresis(smooth, low, high)
    # Dips and blips shorter than the signal's bandwidth lets it change
    # (1.5 / width: an FSK burst's tone changes, through the cut-out's
    # filter) are not pulses or gaps.
    on = _clean_runs(on, max(2, span, int(round(1.5 * rate / max(width, 1.0)))))
    starts, lengths = _runs(on)
    if not len(starts):
        return None
    level = float(np.median(smooth[on]))
    us = 1e6 / rate
    pulses = (lengths * us).tolist()
    gaps = ((starts[1:] - (starts[:-1] + lengths[:-1])) * us).tolist()
    duration = float((starts[-1] + lengths[-1] - starts[0]) * us)
    fsk_shift = 0.0
    longest = float(np.max(lengths)) * us
    if duration > 2000.0 and longest >= 0.8 * float(np.sum(lengths)) * us:
        # One long transmission: frequency keyed? Slice the frequency.
        fsk = _fsk_pulses(x[on], rate)
        if fsk is not None:
            pulses, gaps, fsk_shift = fsk
    if not pulses:
        return None
    analysis = _guess(pulses, gaps)
    analysis.level_dbfs = 10.0 * math.log10(max(level, 1e-30))
    analysis.snr_db = 10.0 * math.log10(max(level - noise, 1e-30) / noise)
    analysis.duration_us = duration
    if fsk_shift:
        analysis.fsk_shift_hz = fsk_shift
        analysis.modulation = analysis.modulation.replace("OOK", "FSK")
    analysis.rows = [row[:MAX_ROW_BITS] for row in analysis.rows]
    return analysis


def _fsk_pulses(
    x: np.ndarray, rate: float
) -> Optional[Tuple[List[float], List[float], float]]:
    """Pulses (high tone) and gaps (low tone) of a constant-envelope burst,
    from its instantaneous frequency (rtl_433's FSK pulse demod)."""
    if len(x) < 64:
        return None
    step = np.angle(x[1:] * np.conj(x[:-1])) * rate / (2 * math.pi)
    span = max(1, int(round(rate * 10e-6)))
    if span > 1:
        c = np.cumsum(np.concatenate(([0.0], step)))
        step = (c[span:] - c[:-span]) / span
    hi = float(np.percentile(step, 90))
    lo = float(np.percentile(step, 10))
    if hi - lo < 1e3:
        return None
    mid = 0.5 * (hi + lo)
    high = _clean_runs(step > mid, max(2, span))
    starts, lengths = _runs(high)
    if len(starts) < 4:
        return None
    us = 1e6 / rate
    pulses = (lengths * us).tolist()
    gaps = ((starts[1:] - (starts[:-1] + lengths[:-1])) * us).tolist()
    return pulses, gaps, 0.5 * (hi - lo)


def _main_groups(groups: List[List[float]]) -> List[float]:
    """The means of the groups that matter: at least 2 members and a
    tenth of the biggest group (a stray width from noise is not one)."""
    if not groups:
        return []
    biggest = max(len(g) for g in groups)
    keep = [g for g in groups if len(g) >= max(2, 0.1 * biggest)]
    return [float(np.mean(g)) for g in (keep or groups)]


def _whole_units(values: Sequence[float], unit: float) -> float:
    """The share of ``values`` within 20 % of a whole number of ``unit``."""
    if unit <= 0 or not len(values):
        return 0.0
    ratio = np.asarray(values, dtype=np.float64) / unit
    near = np.abs(ratio - np.round(ratio)) <= 0.2 * np.maximum(1.0, np.round(ratio))
    return float(np.mean(near & (np.round(ratio) >= 1)))


def _guess(pulses: List[float], gaps: List[float]) -> PulseAnalysis:
    """Guess the modulation from the pulse and gap widths and slice the
    bits (rtl_433's conventions)."""
    p_groups = _cluster(pulses)
    g_all = _cluster(gaps) if gaps else []
    analysis = PulseAnalysis(
        modulation="OOK",
        description="Single pulse",
        pulses_us=[float(p) for p in pulses],
        gaps_us=[float(g) for g in gaps],
        pulse_widths=[(float(np.mean(g)), len(g)) for g in p_groups],
        gap_widths=[(float(np.mean(g)), len(g)) for g in g_all],
    )
    if len(pulses) == 1:
        analysis.description = "A single pulse (an unmodulated carrier burst)"
        return analysis
    # Long gaps between packet repeats (or before a sync) are not data: a
    # rare gap group well beyond the others is the reset.
    g_main = _main_groups(g_all)
    reset = 0.0
    rare = [
        float(np.mean(g))
        for g in g_all
        if g_main and float(np.mean(g)) >= 1.8 * max(g_main)
    ]
    if rare:
        reset = 0.5 * (min(rare) + max(g_main))
    elif len(g_main) >= 2 and g_main[-1] >= 3.0 * g_main[-2]:
        reset = 0.5 * (g_main[-1] + g_main[-2])
    analysis.reset_us = reset
    data_gaps = [g for g in gaps if not reset or g < reset]
    big_g = _main_groups(_cluster(data_gaps)) if data_gaps else []
    big_p = _main_groups(p_groups)
    unit = min(big_p + big_g) if (big_p or big_g) else 0.0
    whole = _whole_units(list(pulses) + data_gaps, unit)

    # Pulse width modulation keeps a fixed bit period: pulse + gap.
    def usual(width: float, means: Sequence[float]) -> bool:
        return any(abs(width - m) <= 0.3 * m for m in means)

    periods = [
        p + g
        for p, g in zip(pulses, gaps, strict=False)  # (no gap after the last)
        if (not reset or g < reset) and usual(p, big_p) and usual(g, big_g)
    ]
    fixed_period = bool(periods) and float(np.std(periods)) < 0.12 * float(
        np.mean(periods)
    )
    if len(big_p) == 1 and len(big_g) >= 2:
        # Fixed pulses, the gap carries the bit: pulse position.
        analysis.modulation = "OOK PPM"
        analysis.description = "Pulse position modulation (the gap carries the bit)"
        analysis.short_us, analysis.long_us = big_g[0], big_g[1]
        analysis.convention = "short gap = 0, long gap = 1"
        analysis.rows = _slice_ppm(gaps, analysis.short_us, analysis.long_us, reset)
    elif len(big_p) == 2 and (fixed_period or whole < 0.9):
        # Two pulse widths: pulse width modulation.
        analysis.modulation = "OOK PWM"
        analysis.description = "Pulse width modulation (the pulse carries the bit" + (
            ", fixed bit period)" if fixed_period else ")"
        )
        analysis.short_us, analysis.long_us = big_p[0], big_p[1]
        analysis.convention = "short pulse = 1, long pulse = 0"
        analysis.rows = _slice_pwm(
            pulses, gaps, analysis.short_us, analysis.long_us, reset
        )
        if big_g:
            analysis.gap_limit_us = max(big_g) * 1.5
    elif whole >= 0.8:
        analysis.modulation = "OOK PCM"
        analysis.description = (
            "Pulse code modulation (NRZ): every width is whole bits of "
            f"{unit:.0f} us"
        )
        analysis.short_us = analysis.long_us = unit
        analysis.convention = "each bit period on = 1, off = 0"
        analysis.rows = _slice_pcm(pulses, gaps, unit, reset)
    elif len(big_p) >= 2:
        analysis.modulation = "OOK PWM"
        analysis.description = "Pulse width modulation (the pulse carries the bit)"
        analysis.short_us, analysis.long_us = big_p[0], big_p[1]
        analysis.convention = "short pulse = 1, long pulse = 0"
        analysis.rows = _slice_pwm(
            pulses, gaps, analysis.short_us, analysis.long_us, reset
        )
    else:
        analysis.description = "Pulses without a clear pattern"
    return analysis


#: The longest bit row kept (the rest of a row this long is noise).
MAX_ROW_BITS = 1024


def _split_rows(gaps: Sequence[float], reset: float) -> List[int]:
    """Indices of the gaps that end a row."""
    if reset <= 0:
        return []
    return [i for i, g in enumerate(gaps) if g >= reset]


def _slice_pwm(
    pulses: Sequence[float],
    gaps: Sequence[float],
    short: float,
    long: float,
    reset: float,
) -> List[str]:
    threshold = 0.5 * (short + long)
    rows: List[str] = []
    row = ""
    ends = set(_split_rows(gaps, reset))
    for i, p in enumerate(pulses):
        if p > 2.5 * long or p < 0.5 * short:
            pass  # a sync pulse, or a glitch: no bit
        else:
            row += "1" if p < threshold else "0"
        if i in ends:
            rows.append(row)
            row = ""
    if row:
        rows.append(row)
    return [r for r in rows if r]


def _slice_ppm(
    gaps: Sequence[float], short: float, long: float, reset: float
) -> List[str]:
    threshold = 0.5 * (short + long)
    rows: List[str] = []
    row = ""
    for g in gaps:
        if reset and g >= reset:
            rows.append(row)
            row = ""
            continue
        if g > 2.5 * long or g < 0.5 * short:
            continue  # a sync gap, or a glitch
        row += "0" if g < threshold else "1"
    if row:
        rows.append(row)
    return [r for r in rows if r]


def _slice_pcm(
    pulses: Sequence[float], gaps: Sequence[float], unit: float, reset: float
) -> List[str]:
    rows: List[str] = []
    row = ""
    for i, p in enumerate(pulses):
        row += "1" * max(1, int(round(p / unit)))
        if i < len(gaps):
            g = gaps[i]
            if reset and g >= reset:
                rows.append(row)
                row = ""
            else:
                row += "0" * max(1, int(round(g / unit)))
    if row:
        rows.append(row)
    return [r for r in rows if r]


def analyze_capture(
    samples: np.ndarray, rate: float, result: Any
) -> Optional[PulseAnalysis]:
    """The pulse analysis of a classified capture (``result``: its
    :class:`~sdr_module.dsp.classifiers.ClassificationResult`): the signal
    cut out around its centre, then :func:`analyze_pulses`."""
    from .classifiers import _narrow

    x = np.asarray(samples, dtype=np.complex64)
    width = float(getattr(result, "bandwidth_hz", 0.0)) or 0.25 * rate
    centre = float(getattr(result, "center_offset_hz", 0.0))
    half = min(max(0.75 * width, 5e3), 0.45 * rate)
    if half < 0.45 * rate:
        x, rate = _narrow(x, rate, centre, half)
    noise = None
    features = getattr(result, "features", {}) or {}
    density = features.get("noise_psd")
    if density:
        noise = float(density) * 2.0 * half
    return analyze_pulses(x, rate, noise)


def format_analysis(analysis: PulseAnalysis, freq_hz: float = 0.0) -> str:
    """The analysis as text, laid out like ``rtl_433 -A``."""
    lines = []
    if freq_hz:
        lines.append(f"Frequency: {freq_hz / 1e6:.3f} MHz")
    lines.append(
        f"Total count: {len(analysis.pulses_us):4d},  width: "
        f"{analysis.duration_us / 1000:.2f} ms"
    )
    lines.append("Pulse width distribution:")
    for i, (width, count) in enumerate(analysis.pulse_widths):
        lines.append(f" [{i:2d}] count: {count:4d},  width: {width:7.0f} us")
    if analysis.gap_widths:
        lines.append("Gap width distribution:")
        for i, (width, count) in enumerate(analysis.gap_widths):
            lines.append(f" [{i:2d}] count: {count:4d},  width: {width:7.0f} us")
    lines.append(f"Level: {analysis.level_dbfs:.1f} dBFS, SNR {analysis.snr_db:.1f} dB")
    if analysis.fsk_shift_hz:
        lines.append(f"FSK shift: ±{analysis.fsk_shift_hz / 1e3:.1f} kHz")
    lines.append(f"Guessing modulation: {analysis.description}")
    if analysis.short_us:
        flex = _flex_line(analysis)
        if flex:
            lines.append(f"Use a flex decoder with -X '{flex}'")
    if analysis.convention:
        lines.append(f"Bits ({analysis.convention}):")
    for bits, count in analysis.distinct_rows():
        repeat = f"  (x{count})" if count > 1 else ""
        lines.append(f"codes     : {{{len(bits)}}}{bits_to_hex(bits)}{repeat}")
    return "\n".join(lines)


def _flex_line(analysis: PulseAnalysis) -> str:
    kind = {
        "OOK PWM": "OOK_PWM",
        "OOK PPM": "OOK_PPM",
        "OOK PCM": "OOK_PCM",
        "FSK PWM": "FSK_PWM",
        "FSK PPM": "FSK_PPM",
        "FSK PCM": "FSK_PCM",
    }.get(analysis.modulation)
    if kind is None:
        return ""
    short = int(round(analysis.short_us))
    long = int(round(analysis.long_us or analysis.short_us))
    reset = int(round(analysis.reset_us or 4 * long))
    parts = [f"n=name,m={kind},s={short},l={long},r={reset}"]
    if analysis.gap_limit_us:
        parts.append(f"g={int(round(analysis.gap_limit_us))}")
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# The watcher: bursts in the span from raw blocks
# --------------------------------------------------------------------------- #


@dataclass
class BurstReport:
    """A finished burst with what was learned about it."""

    burst: Burst
    analysis: Optional[PulseAnalysis] = None
    classification: Any = None
    #: The burst's channel I/Q, for :func:`analyze_report` (may be empty;
    #: up to a few MB: a list of bursts keeps what was learned instead).
    samples: Optional[np.ndarray] = None
    sample_rate: float = 0.0

    @property
    def kind_text(self) -> str:
        if self.analysis is not None and self.analysis.pulses_us:
            return self.analysis.modulation
        result = self.classification
        if result is not None:
            return str(getattr(result, "label", "") or "")
        return ""


class _Capture:
    """The channel of one burst, cut out while it lasts."""

    def __init__(self, rate: float, offset: float, width: float, max_s: float):
        # Roomy: the first row may have seen only part of the burst (one of
        # FSK's tones), which may not be its centre; the analysis cuts out
        # what it occupies.
        half = min(max(width + 50e3, 100e3), 0.45 * rate)
        self.channel = Channelizer(
            rate, offset, pass_hz=half, stop_hz=1.25 * half, target_rate=3.0 * half
        )
        self.rate = self.channel.output_rate
        self.parts: List[np.ndarray] = []
        self.count = 0
        self.max = int(max_s * self.rate)

    def add(self, block: np.ndarray) -> None:
        if self.count >= self.max:
            return
        y = self.channel.process(block)
        self.parts.append(y)
        self.count += len(y)

    def samples(self) -> np.ndarray:
        if not self.parts:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(self.parts)[: self.max]


def analyze_report(report: BurstReport) -> BurstReport:
    """Analyse a finished burst from the samples kept with it: its pulses
    (rtl_433 style) and its modulation (``report`` is filled in and
    returned). Takes 50-200 ms for a long burst: the window runs it off
    its display loop (see :class:`BurstWatcher`'s ``defer``)."""
    samples = report.samples
    if samples is None or not len(samples) or not report.sample_rate > 0:
        return report
    rate = float(report.sample_rate)
    try:
        report.analysis = analyze_pulses(samples, rate)
    except Exception:  # pragma: no cover - a bonus, never fatal
        report.analysis = None
    try:
        from .classifiers import SignalClassifier

        report.classification = SignalClassifier(rate).classify(
            samples, search_hz=0.5 * report.burst.width_hz + 5e3
        )
    except Exception:  # pragma: no cover - a bonus, never fatal
        report.classification = None
    return report


class BurstWatcher:
    """Bursts in the span, from the receiver's raw blocks (see the module
    docstring). Feed every block with :meth:`feed`; it returns the bursts
    that ended, analysed.

    ``analyze``: cut each burst's channel out and analyse its pulses
    (rtl_433 style) and its modulation; ``max_capture_s`` caps how much of
    one burst is kept, ``max_captures`` how many are cut out at once (each
    costs a channel filter on every block: on a busy band the others are
    listed without an analysis). ``defer``: hand the reports out with their
    samples but not analysed yet, for :func:`analyze_report` elsewhere (a
    worker thread: the analysis takes too long for a display loop).
    """

    #: Rows are made from FFTs this long.
    NFFT = 1024

    def __init__(
        self,
        snr_db: float = 10.0,
        analyze: bool = True,
        max_capture_s: float = 1.0,
        max_segments: int = 128,
        defer: bool = False,
        max_captures: int = 4,
    ):
        self.detector = BurstDetector(snr_db=snr_db)
        self.analyze = bool(analyze)
        self.defer = bool(defer)
        self.max_capture_s = float(max_capture_s)
        self.max_segments = int(max_segments)
        self.max_captures = int(max_captures)
        self._captures: Dict[int, _Capture] = {}
        self._previous: Optional[np.ndarray] = None
        self._key: Optional[Tuple[float, float]] = None
        self.time_s = 0.0
        #: Fraction of each block's samples the rows looked at (1 = all).
        self.coverage = 1.0

    def reset(self) -> None:
        self.detector.reset()
        self._captures = {}
        self._previous = None
        self._key = None

    def feed(
        self,
        block: np.ndarray,
        center_hz: float,
        rate: float,
        t: Optional[float] = None,
    ) -> List[BurstReport]:
        x = np.asarray(block, dtype=np.complex64)
        if not len(x) or not rate > 0:
            return []
        key = (float(center_hz), float(rate))
        reports: List[BurstReport] = []
        if key != self._key:
            reports += self._finish(self.detector.flush())
            self.detector.reset()
            self._captures = {}
            self._previous = None
            self._key = key
        if t is None:
            self.time_s += len(x) / rate
            t = self.time_s
        else:
            self.time_s = float(t)
        top, _mean = frame_rows(x, self.NFFT, self.max_segments)
        segments = len(x) // self.NFFT
        self.coverage = min(1.0, self.max_segments / max(segments, 1))
        before = {b.ident for b in self.detector.active}
        ended = self.detector.process(top, center_hz, rate, t)
        if self.analyze:
            for burst in self.detector.active:
                capture = self._captures.get(burst.ident)
                if capture is None:
                    taking = sum(1 for c in self._captures.values() if c.count < c.max)
                    if taking >= self.max_captures:
                        continue  # a busy band: listed without its samples
                    capture = _Capture(
                        rate, burst.offset_hz, burst.width_hz, self.max_capture_s
                    )
                    self._captures[burst.ident] = capture
                    if burst.ident not in before and self._previous is not None:
                        capture.add(self._previous)  # the start of it
                capture.add(x)
        self._previous = x
        reports += self._finish(ended)
        return reports

    def flush(self) -> List[BurstReport]:
        """End every burst in progress (the receiver stopped)."""
        reports = self._finish(self.detector.flush())
        self._captures = {}
        self._previous = None
        return reports

    def _finish(self, bursts: Sequence[Burst]) -> List[BurstReport]:
        reports = []
        for burst in bursts:
            capture = self._captures.pop(burst.ident, None)
            report = BurstReport(burst)
            if capture is not None and capture.count > 0:
                report.samples = capture.samples()
                report.sample_rate = capture.rate
                if not self.defer:
                    analyze_report(report)
            reports.append(report)
        return reports
