"""
One preset table for the whole window.

The frequency manager's presets (:data:`sdr_module.core.frequency_manager.
RX_PRESETS`: name, frequency, channel width, mode, category, and the live
decoder that reads it) are the table, and everything that offers presets
uses it, under the same names:

* Radio > Band Presets: the :data:`FAVOURITES` on top, then a submenu per
  category with every preset;
* the control panel's Presets group (a "Favourites" category first);
* the welcome screen's starting bands (each favourite's band, tuning the
  favourite);
* the Scanner tab's bands (each favourite's band with a range to sweep).

A favourite is a preset in a band: things that tune a channel are named by
the preset ("2m Calling"), things about the whole band by the band ("2m
Ham (144–148 MHz)", the name :mod:`sdr_module.core.band_plan` gives it).
Applying a preset sets the mode, the FM deviation and the filter it is
listened with (:func:`listening`) and turns its decoder on (ADS-B, APRS,
ACARS, POCSAG, FLEX; SSTV).

No widgets here (the Filter choices come from the control panel module).
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

from ..core.band_plan import Band, band_for, preset_listening
from ..core.frequency_manager import FrequencyPreset, get_frequency_manager

#: The Presets group's (and the menu's) first category.
FAVOURITES_CATEGORY = "Favourites"


class Favourite(NamedTuple):
    """A preset offered first (the menu's top, the welcome screen) and its
    band, with the range the Scanner tab sweeps there (none for ADS-B)."""

    preset: str  # the frequency manager preset's name
    band: str  # the band's name (core.band_plan)
    start_mhz: float = 0.0  # Scanner range (0: the Scanner doesn't offer it)
    end_mhz: float = 0.0
    step_khz: float = 0.0
    #: Signals narrower than this are not listed by the scanner (the
    #: broadcast band lists stations, not the stray carriers between them).
    min_width_khz: float = 0.0

    @property
    def scanned(self) -> bool:
        return self.end_mhz > self.start_mhz > 0.0


#: Each favourite's preset lands on a demo-device signal, so every one of
#: them can be tried without hardware.
FAVOURITES: Tuple[Favourite, ...] = (
    Favourite("FM Broadcast", "FM Broadcast", 88.0, 108.0, 200.0, 30.0),
    Favourite("NOAA Weather 1", "NOAA Weather", 162.4, 162.55, 25.0),
    Favourite("2m Calling", "2m Ham", 144.0, 148.0, 12.5),
    Favourite("70cm Calling", "70cm Ham", 420.0, 450.0, 25.0),
    Favourite("Air Traffic Control", "Airband", 118.0, 137.0, 25.0),
    Favourite("ADS-B", "ADS-B"),
    Favourite("ISM 433 MHz", "ISM 433", 433.05, 434.79, 25.0),
    Favourite("ISM 915 MHz", "ISM 915", 902.0, 928.0, 100.0),
)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def categories() -> List[str]:
    """The frequency manager's categories, in its order."""
    return list(get_frequency_manager().get_preset_categories())


def presets(category: Optional[str] = None) -> List[FrequencyPreset]:
    """The presets of ``category`` (:data:`FAVOURITES_CATEGORY`: the
    favourites; None: all of them)."""
    if category == FAVOURITES_CATEGORY:
        return favourites()
    return list(get_frequency_manager().get_rx_presets(category))


def preset_by_name(name: str) -> Optional[FrequencyPreset]:
    if not name:
        return None
    return get_frequency_manager().get_preset_by_name(str(name))


def favourites() -> List[FrequencyPreset]:
    """The favourites' presets, in :data:`FAVOURITES` order."""
    found = (preset_by_name(f.preset) for f in FAVOURITES)
    return [p for p in found if p is not None]


def is_favourite(name: str) -> bool:
    return any(f.preset == name for f in FAVOURITES)


def preset_at(
    freq_hz: float, category: Optional[str] = None, tolerance_hz: float = 500.0
) -> Optional[FrequencyPreset]:
    """The preset tuned to ``freq_hz`` (within ``tolerance_hz``), looked for
    in ``category`` first, then among the favourites, then everywhere."""
    try:
        freq = float(freq_hz)
    except (TypeError, ValueError):
        return None
    order: List[Optional[str]] = [category] if category else []
    if FAVOURITES_CATEGORY not in order:
        order.append(FAVOURITES_CATEGORY)
    order.append(None)  # every preset
    for where in order:
        for preset in presets(where):
            if abs(float(preset.frequency_hz) - freq) <= tolerance_hz:
                return preset
    return None


def listening(preset: FrequencyPreset) -> Tuple[str, str, str]:
    """``(mode, FM deviation, filter)`` the preset is listened with, in the
    Mode, FM deviation and Filter lists' words (``("FM", "75 kHz", "200
    kHz")`` for broadcast FM; ``("None (I/Q)", "", "2.4 MHz")`` for raw).
    The filter is the one Apply Preset selects: the narrowest of the mode's
    choices that holds the preset's channel (15 kHz -> 25 kHz in FM)."""
    from .control_panel import bandwidth_options, closest_bandwidth

    found = preset_listening(preset)
    if found is None:
        return str(preset.mode), "", ""
    mode, deviation, _width = found
    width = closest_bandwidth(bandwidth_options(mode), float(preset.bandwidth_hz))
    return mode, deviation, width


def decoder_for(preset: Optional[FrequencyPreset]) -> str:
    """The decoder the preset turns on ("" for none)."""
    return str(getattr(preset, "decoder", "") or "")


def band_of(favourite: Favourite) -> Optional[Band]:
    """The favourite's band (core.band_plan), looked up in the middle of
    its scan range or at its preset."""
    if favourite.scanned:
        middle = 0.5 * (favourite.start_mhz + favourite.end_mhz) * 1e6
        band = band_for(middle)
        if band is not None:
            return band
    preset = preset_by_name(favourite.preset)
    return band_for(preset.frequency_hz) if preset is not None else None


# --------------------------------------------------------------------------- #
# Texts
# --------------------------------------------------------------------------- #


def mhz(freq_hz: float) -> str:
    """``146.52e6`` -> ``"146.520 MHz"``, ``137.9125e6`` -> ``"137.9125 MHz"``."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def _short(value_mhz: float) -> str:
    return f"{value_mhz:.4f}".rstrip("0").rstrip(".")


def band_label(favourite: Favourite) -> str:
    """The band with its range: ``"2m Ham (144–148 MHz)"`` (the welcome
    screen and the Scanner tab); ADS-B: ``"ADS-B (1090 MHz)"``."""
    if favourite.scanned:
        span = f"{_short(favourite.start_mhz)}–{_short(favourite.end_mhz)} MHz"
    else:
        preset = preset_by_name(favourite.preset)
        span = mhz(preset.frequency_hz).replace(".000 ", " ") if preset else ""
    return f"{favourite.band} ({span})" if span else favourite.band


def describe(preset: FrequencyPreset) -> str:
    """``"146.520 MHz · FM, 5 kHz deviation, 25 kHz filter"`` (and the
    decoder it turns on), in the Mode list's words: ``"FM (broadcast)"``,
    ``"None (I/Q)"``, as the controls show them once it is applied."""
    mode, deviation, width = listening(preset)
    shown = mode
    if mode == "FM" and deviation == "75 kHz":
        shown = "FM (broadcast)"
    parts = [shown]
    if mode == "FM" and deviation and deviation != "75 kHz":
        parts.append(f"{deviation} deviation")
    if width:
        parts.append(f"{width} filter")
    text = f"{mhz(preset.frequency_hz)} · {', '.join(parts)}"
    decoder = decoder_for(preset)
    if decoder:
        text += f" · {decoder} decoder"
    return text


# --------------------------------------------------------------------------- #
# What the welcome screen and the Scanner tab offer
# --------------------------------------------------------------------------- #


class ScanBand(NamedTuple):
    """A band the Scanner tab offers (see scanner_dialog.ScanPreset)."""

    label: str
    start_mhz: float
    end_mhz: float
    step_khz: float
    mode: str
    fm_deviation: str
    bandwidth: str
    min_width_khz: float


def scan_bands() -> List[ScanBand]:
    """The favourites' bands with a range, in :data:`FAVOURITES` order,
    each listened to the way its band usually is."""
    bands: List[ScanBand] = []
    for favourite in FAVOURITES:
        if not favourite.scanned:
            continue
        band = band_of(favourite)
        if band is not None:
            mode, deviation, width = band.listening
        else:
            preset = preset_by_name(favourite.preset)
            if preset is None:
                continue
            mode, deviation, width = listening(preset)
        bands.append(
            ScanBand(
                band_label(favourite),
                favourite.start_mhz,
                favourite.end_mhz,
                favourite.step_khz,
                mode,
                deviation,
                width,
                favourite.min_width_khz,
            )
        )
    return bands


def welcome_bands() -> List[Tuple[str, FrequencyPreset]]:
    """``(band label, preset)`` for each favourite: the welcome screen's
    starting bands (the band, and the preset it tunes)."""
    items: List[Tuple[str, FrequencyPreset]] = []
    for favourite in FAVOURITES:
        preset = preset_by_name(favourite.preset)
        if preset is not None:
            items.append((band_label(favourite), preset))
    return items


def assign_mnemonics(labels: Sequence[str]) -> List[str]:
    """``labels`` with one "&" each, on a different letter or digit for
    every label (the first free one, preferring the start of a word)."""
    used = set()
    result = []
    for label in labels:
        clean = label.replace("&", "&&")
        starts = [
            i
            for i, c in enumerate(label)
            if c.isalnum() and (i == 0 or not label[i - 1].isalnum())
        ]
        rest = [i for i, c in enumerate(label) if c.isalnum() and i not in starts]
        chosen = next((i for i in starts + rest if label[i].lower() not in used), None)
        if chosen is None:
            result.append(clean)
            continue
        used.add(label[chosen].lower())
        # (Positions shift by one for every "&" doubled before ``chosen``.)
        shift = label[:chosen].count("&")
        at = chosen + shift
        result.append(clean[:at] + "&" + clean[at:])
    return result


def iter_menu() -> Iterator[Tuple[str, List[FrequencyPreset]]]:
    """``("", favourites)``, then ``(category, presets)`` per category:
    Radio > Band Presets."""
    yield "", favourites()
    for category in categories():
        yield category, presets(category)
