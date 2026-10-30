"""
Tuning helpers shared by the main window, the control panel and the plots.

Pure Python and NumPy (no Qt), so they are easy to test:

* the tuning **step** choices, the step each mode starts with and stepping
  on the step's raster (:func:`step_frequency`);
* reading a typed frequency (:func:`parse_frequency_text`): unit suffixes,
  e-notation, both decimal separators, and a bare number in Hz typed into
  the MHz field;
* snapping a click on the spectrum or waterfall to the signal under it
  (:func:`snap_click_frequency`);
* the back/forward tuning history and the recent frequencies
  (:class:`TuningHistory`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tuning steps
# ---------------------------------------------------------------------------

#: The STEP choices, in Hz. 8.33 kHz is exactly a third of 25 kHz (the
#: airband channel raster).
STEP_CHOICES_HZ: Tuple[float, ...] = (
    10.0,
    50.0,
    100.0,
    500.0,
    1e3,
    2.5e3,
    5e3,
    6.25e3,
    25e3 / 3.0,
    9e3,
    10e3,
    12.5e3,
    25e3,
    50e3,
    100e3,
    200e3,
    1e6,
)

#: The step each mode starts with (until the user picks another for it).
STEP_MODE_DEFAULTS: Dict[str, float] = {
    "wfm": 100e3,  # FM broadcast: the 100 kHz channel raster
    "nfm": 12.5e3,  # narrow FM voice and data channels
    "am": 10e3,  # AM broadcast raster
    "usb": 100.0,
    "lsb": 100.0,
    "cw": 10.0,
    "iq": 10e3,
}

#: What each mode key is called in messages.
STEP_MODE_NAMES: Dict[str, str] = {
    "wfm": "FM broadcast",
    "nfm": "narrow FM",
    "am": "AM",
    "usb": "USB",
    "lsb": "LSB",
    "cw": "CW",
    "iq": "I/Q",
}

#: FM with at least this deviation is broadcast (wideband) FM.
BROADCAST_DEVIATION_HZ = 50e3

#: A step that ends within this of a raster point lands on it (Hz): more
#: than the half hertz rounding to whole Hz leaves, less than the 1 Hz a
#: deliberate offset can be.
_ON_RASTER_HZ = 0.75


def step_mode_key(mode: str, fm_deviation_hz: float = 0.0) -> str:
    """The key a mode's tuning step is remembered under.

    ``("FM", 75e3)`` -> ``"wfm"``, ``("FM", 5e3)`` -> ``"nfm"``, ``"AM"`` ->
    ``"am"``, ``"None (I/Q)"`` -> ``"iq"``.
    """
    text = str(mode or "").strip()
    if text == "FM":
        try:
            wide = float(fm_deviation_hz) >= BROADCAST_DEVIATION_HZ
        except (TypeError, ValueError):
            wide = False
        return "wfm" if wide else "nfm"
    if text.startswith("None") or text.upper() in ("I/Q", "IQ", "RAW"):
        return "iq"
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    return key or "iq"


def default_step_hz(mode_key: str) -> float:
    """The step a mode starts with (10 kHz for an unknown mode)."""
    return STEP_MODE_DEFAULTS.get(mode_key, 10e3)


def nearest_step(step_hz: float) -> float:
    """The STEP choice closest to ``step_hz`` (on a log scale)."""
    try:
        value = float(step_hz)
    except (TypeError, ValueError):
        return 10e3
    if not (math.isfinite(value) and value > 0):
        return 10e3
    return min(STEP_CHOICES_HZ, key=lambda s: abs(math.log(s / value)))


def format_step(step_hz: float) -> str:
    """``12500 -> "12.5 kHz"``, ``25e3/3 -> "8.33 kHz"``, ``1e6 -> "1 MHz"``."""
    value = float(step_hz)
    for unit, scale in (("MHz", 1e6), ("kHz", 1e3)):
        if value >= scale:
            return f"{value / scale:.2f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{value:.0f} Hz"


def step_index(step_hz: float) -> int:
    """Index of ``step_hz`` (or the nearest choice) in :data:`STEP_CHOICES_HZ`."""
    return STEP_CHOICES_HZ.index(nearest_step(step_hz))


def step_frequency(freq_hz: float, step_hz: float, steps: int) -> float:
    """``freq_hz`` moved by ``steps`` tuning steps.

    The offset from the step's raster is kept (145.800001 MHz up 10 kHz is
    145.810001 MHz), except the sub-hertz drift that rounding to whole Hz
    leaves on a raster that isn't whole Hz: three 8.33 kHz steps from
    118 MHz land exactly on 118.025 MHz, not 1 Hz below.
    """
    step = float(step_hz)
    target = float(freq_hz) + int(steps) * step
    if step > 0:
        nearest = round(target / step) * step
        if abs(target - nearest) <= _ON_RASTER_HZ:
            target = nearest
    return target


# ---------------------------------------------------------------------------
# Typed frequencies
# ---------------------------------------------------------------------------

#: Unit suffixes (lowercase) -> Hz.
_SUFFIXES: Dict[str, float] = {
    "hz": 1.0,
    "k": 1e3,
    "khz": 1e3,
    "m": 1e6,
    "mhz": 1e6,
    "g": 1e9,
    "ghz": 1e9,
}
#: Unfinished suffixes while typing ("433.92 MH").
_PARTIAL_SUFFIXES = ("h", "kh", "mh", "gh")
_SPACES = re.compile(r"[\s\u00a0\u2009\u202f_']+")
_SUFFIX_RE = re.compile(r"([a-df-z]+)$", re.IGNORECASE)

#: The highest frequency any supported receiver tunes (HackRF One).
ANY_RECEIVER_MAX_HZ = 6e9

#: A bare number at least this large that can't be meant in the field's
#: unit (MHz: 1,000,000 MHz is far beyond any receiver) is read as Hz.
HZ_READING_MINIMUM = 1e6

ACCEPTABLE = "acceptable"
INTERMEDIATE = "intermediate"
INVALID = "invalid"


@dataclass(frozen=True)
class ParsedFrequency:
    """What a typed frequency means.

    ``state`` is :data:`ACCEPTABLE` (``hz`` is in range), :data:`INTERMEDIATE`
    (unfinished, or a number out of range: keep typing) or :data:`INVALID`
    (a character that can't belong to a frequency). ``hz`` is the value
    read (None when there is no number yet); ``read_as_hz`` says a bare
    number was taken as Hz. ``problem`` says, in plain words, why an
    :data:`INTERMEDIATE` or :data:`INVALID` text can't be tuned.
    """

    state: str
    hz: Optional[float] = None
    read_as_hz: bool = False
    problem: str = ""


def format_mhz_range(low_hz: float, high_hz: float) -> str:
    """``(500e3, 1.766e9) -> "0.5–1766 MHz"``."""
    return f"{low_hz / 1e6:g}–{high_hz / 1e6:g} MHz"


def _groups_thousands(digits: str, sep: str) -> bool:
    """Whether ``digits`` groups thousands with ``sep`` ("145.800.000"):
    groups of three after the first, none empty or short, as a slip of the
    finger would leave ("12..5", "145.800.0")."""
    pattern = r"[0-9]{1,3}(?:" + re.escape(sep) + r"[0-9]{3})+"
    return re.fullmatch(pattern, digits) is not None


def _normalize_number(
    mantissa: str, grouped: bool, group_separator: str
) -> Optional[str]:
    """``mantissa`` (digits, '.' and ',') as a Python float string, or None
    when its separators don't make a number ("12..5", "1.2.3")."""
    dots, commas = mantissa.count("."), mantissa.count(",")
    if dots and commas:
        # Both: the last one is the decimal separator, the other groups.
        last = "." if mantissa.rfind(".") > mantissa.rfind(",") else ","
        other = "," if last == "." else "."
        whole, _, frac = mantissa.rpartition(last)
        if not _groups_thousands(whole, other):
            return None
        return whole.replace(other, "") + "." + frac
    sep = "." if dots else ("," if commas else "")
    if not sep:
        return mantissa
    if mantissa.count(sep) > 1:
        # "145,800,000": grouping.
        return mantissa.replace(sep, "") if _groups_thousands(mantissa, sep) else None
    whole, _, frac = mantissa.partition(sep)
    if grouped and sep == group_separator and whole and len(frac) == 3:
        return whole + frac  # "7,074" in the kHz field: grouping
    return f"{whole}.{frac}"


def parse_frequency_text(
    text: str,
    unit_hz: float,
    minimum_hz: float,
    maximum_hz: float,
    *,
    grouped: bool = False,
    group_separator: str = ",",
    range_text: str = "",
) -> ParsedFrequency:
    """Read ``text`` typed into a frequency field shown in ``unit_hz``.

    Accepts a unit suffix in any case, with or without a space (``Hz``,
    ``k``/``kHz``, ``M``/``MHz``, ``G``/``GHz``), e-notation (``7.074e6``)
    and either decimal separator (``144,39`` in the MHz field). In the Hz
    and kHz units, which show digit grouping (``grouped``), and with a Hz or
    kHz suffix, a single ``group_separator`` followed by three digits groups
    digits (``7,074`` kHz is 7074 kHz).

    A bare number above the field's range in its unit that is at least
    :data:`HZ_READING_MINIMUM` and in range as Hz is read as Hz:
    ``162550000`` in the MHz field is 162.55 MHz. Out-of-range numbers are
    :data:`INTERMEDIATE`, so typing more digits is never refused.
    ``range_text`` (e.g. ``"RTL-SDR tunes 0.5–1766 MHz"``) explains the
    range in the problem text.
    """
    raw = str(text or "")
    compact = _SPACES.sub("", raw)
    if not compact:
        return ParsedFrequency(INTERMEDIATE)
    if compact[0] == "-" and compact[1:2] not in ("+", "-"):
        # Kept (not dropped, which would quietly tune the positive number)
        # and refused with the reason.
        rest = parse_frequency_text(
            compact[1:],
            unit_hz,
            minimum_hz,
            maximum_hz,
            grouped=grouped,
            group_separator=group_separator,
        )
        if rest.state == INVALID:
            return rest
        hz = None if rest.hz is None else -rest.hz
        return ParsedFrequency(
            INTERMEDIATE, hz, problem="A frequency can't be negative."
        )
    if compact[0] in "+-":
        return ParsedFrequency(INVALID, problem=_unreadable(raw))
    suffix = ""
    match = _SUFFIX_RE.search(compact)
    if match:
        suffix = match.group(1).lower()
        compact = compact[: match.start()]
    if suffix and suffix not in _SUFFIXES:
        if suffix in _PARTIAL_SUFFIXES:
            return ParsedFrequency(INTERMEDIATE, problem=_unreadable(raw))
        return ParsedFrequency(INVALID, problem=_unreadable(raw))
    # Number: digits and separators, then an optional exponent.
    mantissa, e, exponent = compact.replace("E", "e").partition("e")
    if not re.fullmatch(r"[0-9.,]*", mantissa):
        return ParsedFrequency(INVALID, problem=_unreadable(raw))
    if e and not re.fullmatch(r"[+-]?[0-9]*", exponent):
        return ParsedFrequency(INVALID, problem=_unreadable(raw))
    if e and suffix:
        return ParsedFrequency(INVALID, problem=_unreadable(raw))
    if not re.search(r"[0-9]", mantissa) or (e and not exponent.strip("+-")):
        return ParsedFrequency(INTERMEDIATE, problem=_unreadable(raw))
    grouping = suffix in ("hz", "k", "khz") if suffix else grouped
    number = _normalize_number(mantissa, grouping, group_separator)
    if number is None:
        return ParsedFrequency(INTERMEDIATE, problem=_unreadable(raw))
    try:
        value = float(f"{number}e{exponent}" if e else number)
    except ValueError:
        return ParsedFrequency(INTERMEDIATE, problem=_unreadable(raw))
    if not math.isfinite(value):
        return ParsedFrequency(INTERMEDIATE, problem=_unreadable(raw))
    scale = _SUFFIXES[suffix] if suffix else float(unit_hz)
    hz = value * scale
    if minimum_hz <= hz <= maximum_hz:
        return ParsedFrequency(ACCEPTABLE, hz)
    if (
        not suffix
        and scale > 1.0
        and hz > maximum_hz
        and value >= HZ_READING_MINIMUM
        and value == round(value)
        and minimum_hz <= value <= maximum_hz
    ):
        # "162550000" in the MHz field: clearly Hz.
        return ParsedFrequency(ACCEPTABLE, value, read_as_hz=True)
    return ParsedFrequency(
        INTERMEDIATE,
        hz,
        problem=_out_of_range(
            raw.strip(),
            suffix,
            scale,
            hz,
            minimum_hz,
            maximum_hz,
            range_text,
        ),
    )


_UNIT_NAMES = {1.0: "Hz", 1e3: "kHz", 1e6: "MHz", 1e9: "GHz"}


def _unreadable(text: str) -> str:
    return (
        f"Can't read “{text.strip()}” as a frequency. Type a number such as "
        "145.8, or add a unit: 433.92M, 7074k."
    )


def _out_of_range(
    text: str,
    suffix: str,
    scale: float,
    hz: float,
    minimum_hz: float,
    maximum_hz: float,
    range_text: str,
) -> str:
    shown = text if suffix else f"{text} {_UNIT_NAMES.get(scale, '')}".strip()
    limits = range_text or (
        f"the receiver tunes {format_mhz_range(minimum_hz, maximum_hz)}"
    )
    message = f"{shown} is out of range: {limits}."
    if not suffix and hz > ANY_RECEIVER_MAX_HZ:
        # Beyond every receiver, so probably meant in a smaller unit ("7000"
        # for 7000 kHz): say how to type that. Not for a reading with a
        # fraction of a hertz ("6000.000001" is 1 Hz above 6 GHz, not kHz).
        value = hz / scale
        for smaller, unit, typed in ((1e3, "kHz", "k"), (1.0, "Hz", " Hz")):
            reading = value * smaller
            if (
                smaller < scale
                and minimum_hz <= reading <= maximum_hz
                and abs(reading - round(reading)) < 1e-6
            ):
                message += f" For {text} {unit}, type {text}{typed}."
                break
    return message


# ---------------------------------------------------------------------------
# Click-to-tune snapping
# ---------------------------------------------------------------------------

#: Pixels either side of a click searched for the signal to tune to.
SNAP_SEARCH_PX = 6
#: A peak this far above the span's median (its noise floor) is a signal.
#: FFT noise bins scatter widely (exponential power): within a dozen bins
#: the strongest is often 6 dB above the median, rarely 10.
SNAP_MIN_SNR_DB = 10.0
#: Modes whose tuned frequency is not the signal's center (the suppressed
#: carrier is at the edge of a sideband signal), so a click isn't moved
#: onto the signal.
_EDGE_TUNED_MODES = ("usb", "lsb")


def _signal_center(power_db: np.ndarray, peak: int, threshold_db: float) -> float:
    """Fractional bin of the signal around ``peak``: a parabola through the
    top three bins for a narrow peak (a carrier, CW), else the power-weighted
    centroid of the contiguous bins within 6 dB of the peak."""
    n = len(power_db)
    first = last = peak
    limit = max(4, n // 8)  # a station never spans more than this
    while first > 0 and power_db[first - 1] >= threshold_db and peak - first < limit:
        first -= 1
    while last < n - 1 and power_db[last + 1] >= threshold_db and last - peak < limit:
        last += 1
    if last - first <= 2 and 0 < peak < n - 1:
        a, b, c = (float(power_db[peak + d]) for d in (-1, 0, 1))
        denominator = a - 2.0 * b + c
        if denominator < 0:
            return peak + 0.5 * (a - c) / denominator
        return float(peak)
    region = np.asarray(power_db[first : last + 1], dtype=np.float64)
    weights = np.power(10.0, (region - region.max()) / 10.0)
    return float(first + np.sum(weights * np.arange(len(region))) / np.sum(weights))


def _signal_near(
    click: float,
    power_db: Optional[np.ndarray],
    start_hz: float,
    span_hz: float,
    hz_per_px: float,
    search_px: float,
    min_snr_db: float,
) -> Optional[Tuple[float, float]]:
    """``(center Hz, bin width Hz)`` of a signal within ``search_px`` pixels
    of ``click`` in ``power_db`` (a peak ``min_snr_db`` above its median),
    or None."""
    data = None if power_db is None else np.asarray(power_db, dtype=np.float64)
    if (
        data is None
        or data.ndim != 1
        or len(data) < 8
        or not span_hz > 0
        or not np.all(np.isfinite(data))
    ):
        return None
    n = len(data)
    bin_hz = span_hz / n
    center_bin = (click - start_hz) / bin_hz
    half = max(1, int(round(search_px * hz_per_px / bin_hz)))
    lo = max(0, int(math.floor(center_bin)) - half)
    hi = min(n, int(math.ceil(center_bin)) + half + 1)
    if hi <= lo:
        return None
    peak = lo + int(np.argmax(data[lo:hi]))
    if data[peak] - float(np.median(data)) < min_snr_db:
        return None
    where = _signal_center(data, peak, float(data[peak]) - 6.0)
    return start_hz + where * bin_hz, bin_hz


def snap_click_frequency(
    click_hz: float,
    power_db: Optional[np.ndarray],
    start_hz: float,
    span_hz: float,
    hz_per_px: float,
    step_hz: float,
    mode_key: str = "",
    *,
    search_px: float = SNAP_SEARCH_PX,
    min_snr_db: float = SNAP_MIN_SNR_DB,
    recent_db: Optional[np.ndarray] = None,
) -> float:
    """Where a click at ``click_hz`` tunes.

    ``power_db`` is the spectrum (dB, ``fftshift`` order) the plot shows,
    covering ``start_hz`` to ``start_hz + span_hz``. A signal within
    ``search_px`` pixels of the click (a peak ``min_snr_db`` above the
    median) is tuned at its center: the peak of a narrow carrier, else the
    centroid of the -6 dB region. ``recent_db``, the strongest level per
    bin over the last moment (same bins), is searched first when given: it
    shows a station's whole width (one frame shows only where its
    modulation is at that instant, so a click on the side of a broadcast
    station landed tens of kHz off its center) and a keyed CW signal
    between two dits; ``power_db`` is searched when it shows nothing there
    (a weak signal stands out less above a max-hold's floor). That center
    moves onto the step's raster when the raster point is close (within a
    quarter step or a bin): a broadcast station lands on its 100 kHz
    channel, while a 2 m simplex channel (on a 5 kHz raster) is not dragged
    to the 12.5 kHz one. Clicks away from any signal, and in USB/LSB (where
    the dial is the signal's edge), go to the step's raster, or to the
    plot's pixel resolution when a step is finer than a pixel.
    """
    click = float(click_hz)
    step = float(step_hz) if step_hz and step_hz > 0 else 0.0
    pixel = _pixel_resolution(hz_per_px)
    candidate: Optional[float] = None
    bin_hz = 0.0
    if mode_key not in _EDGE_TUNED_MODES:
        for data in (recent_db, power_db):
            found = _signal_near(
                click, data, start_hz, span_hz, hz_per_px, search_px, min_snr_db
            )
            if found is not None:
                candidate, bin_hz = found
                break
    if candidate is None:
        if step >= pixel:
            return _on_raster(click, step)
        return round(click / pixel) * pixel
    if step > 0:
        raster = _on_raster(candidate, step)
        if abs(raster - candidate) <= max(0.25 * step, bin_hz):
            return raster
    fine = _pixel_resolution(bin_hz)
    if step > 0:
        fine = min(fine, step)
    return round(candidate / fine) * fine


def _on_raster(freq_hz: float, step_hz: float) -> float:
    return round(freq_hz / step_hz) * step_hz


def _pixel_resolution(hz: float) -> float:
    """1, 10, 100 ... Hz: the resolution ``hz`` per pixel (or bin) supports."""
    if not math.isfinite(hz) or hz <= 1.0:
        return 1.0
    return 10.0 ** math.floor(math.log10(hz))


# ---------------------------------------------------------------------------
# Tuning history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TunedEntry:
    """A frequency the receiver stayed on, with the mode it listened in."""

    freq_hz: float
    mode: str = ""
    deviation: str = ""

    def same_frequency(self, other: "TunedEntry") -> bool:
        return abs(self.freq_hz - other.freq_hz) < 0.5

    def same_listening(self, other: "TunedEntry") -> bool:
        """The same mode (and FM deviation)."""
        return (self.mode, self.deviation) == (other.mode, other.deviation)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"freq_hz": self.freq_hz, "mode": self.mode}
        if self.deviation:
            data["deviation"] = self.deviation
        return data

    @classmethod
    def from_dict(cls, data: Any) -> Optional["TunedEntry"]:
        if not isinstance(data, dict):
            return None
        try:
            freq = float(data.get("freq_hz"))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(freq) and freq > 0):
            return None
        return cls(
            freq, str(data.get("mode", "") or ""), str(data.get("deviation", "") or "")
        )


class TuningHistory:
    """Back/forward through the frequencies tuned, like a browser, plus the
    recent frequencies (most recent first, each once).

    :meth:`visit` records where the receiver settled. Visiting the current
    entry's frequency again only updates its mode; visiting another after
    going back drops the entries ahead (as a browser does). :meth:`back`
    and :meth:`forward` return the entry to tune to.
    """

    def __init__(
        self,
        recent: Sequence[TunedEntry] = (),
        recent_limit: int = 20,
        history_limit: int = 100,
    ):
        self.recent_limit = max(1, int(recent_limit))
        self.history_limit = max(2, int(history_limit))
        self._recent: List[TunedEntry] = []
        for entry in recent:
            if not any(entry.same_frequency(e) for e in self._recent):
                self._recent.append(entry)
        self._recent = self._recent[: self.recent_limit]
        # Oldest first, so Back after a restart goes to the last frequency.
        self._history: List[TunedEntry] = list(reversed(self._recent))
        self._pos = len(self._history) - 1

    # -- recording --------------------------------------------------------
    def current(self) -> Optional[TunedEntry]:
        return self._history[self._pos] if 0 <= self._pos < len(self._history) else None

    def visit(self, entry: TunedEntry) -> bool:
        """Record ``entry`` as where the receiver is now. True if the
        history or the recent list changed."""
        here = self.current()
        changed = False
        if here is not None and here.same_frequency(entry):
            if not here.same_listening(entry):
                self._history[self._pos] = entry
                changed = True
        else:
            del self._history[self._pos + 1 :]
            self._history.append(entry)
            if len(self._history) > self.history_limit:
                del self._history[: len(self._history) - self.history_limit]
            self._pos = len(self._history) - 1
            changed = True
        top = self._recent[0] if self._recent else None
        if top is None or not (top.same_frequency(entry) and top.same_listening(entry)):
            recent = [entry] + [e for e in self._recent if not e.same_frequency(entry)]
            self._recent = recent[: self.recent_limit]
            changed = True
        return changed

    # -- navigation -------------------------------------------------------
    def can_go_back(self) -> bool:
        return self._pos > 0

    def can_go_forward(self) -> bool:
        return self._pos < len(self._history) - 1

    def peek(self, delta: int) -> Optional[TunedEntry]:
        """The entry ``delta`` places from the current one (-1: what
        :meth:`back` would return), without moving."""
        index = self._pos + int(delta)
        if 0 <= index < len(self._history):
            return self._history[index]
        return None

    def back(self) -> Optional[TunedEntry]:
        if not self.can_go_back():
            return None
        self._pos -= 1
        return self._history[self._pos]

    def forward(self) -> Optional[TunedEntry]:
        if not self.can_go_forward():
            return None
        self._pos += 1
        return self._history[self._pos]

    # -- recent frequencies ---------------------------------------------------
    def recent(self) -> List[TunedEntry]:
        """The recent frequencies, most recent first, each once."""
        return list(self._recent)

    def clear_recent(self) -> None:
        """Forget the recent frequencies (the current one stays for Back)."""
        here = self.current()
        self._recent = []
        self._history = [here] if here is not None else []
        self._pos = len(self._history) - 1

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self._recent])

    @staticmethod
    def entries_from_json(raw: str) -> List[TunedEntry]:
        """Saved recent frequencies; unreadable data gives an empty list."""
        try:
            data = json.loads(raw) if raw else []
        except (TypeError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        entries = [TunedEntry.from_dict(item) for item in data]
        return [e for e in entries if e is not None]
