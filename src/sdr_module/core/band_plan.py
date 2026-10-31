"""
Band plan: what a frequency is, and how it is usually listened to.

Used to name new bookmarks ("FM Broadcast 100.1", "2m Calling") and to pick
a listening mode for a frequency that doesn't say (a scanner hit in a
custom range, a frequency typed into the Bookmarks panel).

The bands are the common receive bands of an RTL-SDR or HackRF, with the
mode and channel width each is usually heard in. Narrower bands inside
wider ones (ISM 433 inside 70 cm) take precedence. Within the amateur SSB
bands the CW/data segments are heard in CW and the digital-mode channels
(FT8, FT4) in USB (see :func:`segment_listening`). Like the frequency
manager's segments and license-free channels, the bands are the US ones.

Pure Python (no Qt, no NumPy), so the CLI can use it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple, Optional, Tuple

from .frequency_manager import AMATEUR_BAND_PRIVILEGES, RX_PRESETS, FrequencyPreset


@dataclass(frozen=True)
class Band:
    """One band: its name, edges and how it is usually listened to.

    Attributes:
        name: Short name, e.g. "FM Broadcast" or "2m Ham".
        start_hz: Lower edge in Hz.
        end_hz: Upper edge in Hz.
        mode: Demodulation > Mode item ("AM", "FM", "USB", "LSB", "CW" or
            "None (I/Q)").
        fm_deviation: FM deviation choice ("75 kHz", "5 kHz"); "" if not FM.
        bandwidth: Bandwidth choice (the channel width, "200 kHz").
    """

    name: str
    start_hz: float
    end_hz: float
    mode: str
    fm_deviation: str = ""
    bandwidth: str = ""

    def contains(self, freq_hz: float) -> bool:
        return self.start_hz <= float(freq_hz) <= self.end_hz

    @property
    def listening(self) -> Tuple[str, str, str]:
        """``(mode, fm_deviation, bandwidth)`` to listen to this band with."""
        return self.mode, self.fm_deviation, self.bandwidth


_IQ = "None (I/Q)"

#: Receive bands, roughly in frequency order. Where bands overlap the
#: narrowest one wins (see :func:`band_for`).
BANDS: Tuple[Band, ...] = (
    Band("LW Broadcast", 148.5e3, 283.5e3, "AM", "", "10 kHz"),
    Band("AM Broadcast", 525e3, 1705e3, "AM", "", "10 kHz"),
    Band("160m Ham", 1.8e6, 2.0e6, "LSB", "", "2.7 kHz"),
    Band("80m Ham", 3.5e6, 4.0e6, "LSB", "", "2.7 kHz"),
    Band("60m Ham", 5.3305e6, 5.4065e6, "USB", "", "2.7 kHz"),
    Band("40m Ham", 7.0e6, 7.3e6, "LSB", "", "2.7 kHz"),
    Band("30m Ham", 10.1e6, 10.15e6, "CW", "", "500 Hz"),
    Band("20m Ham", 14.0e6, 14.35e6, "USB", "", "2.7 kHz"),
    Band("17m Ham", 18.068e6, 18.168e6, "USB", "", "2.7 kHz"),
    Band("15m Ham", 21.0e6, 21.45e6, "USB", "", "2.7 kHz"),
    Band("12m Ham", 24.89e6, 24.99e6, "USB", "", "2.7 kHz"),
    Band("CB", 26.965e6, 27.405e6, "AM", "", "10 kHz"),
    Band("10m Ham", 28.0e6, 29.7e6, "USB", "", "2.7 kHz"),
    Band("6m Ham", 50.0e6, 54.0e6, "USB", "", "2.7 kHz"),
    Band("FM Broadcast", 87.5e6, 108.0e6, "FM", "75 kHz", "200 kHz"),
    Band("Airband", 108.0e6, 137.0e6, "AM", "", "10 kHz"),
    Band("Weather Satellites", 137.0e6, 138.0e6, "FM", "25 kHz", "50 kHz"),
    Band("2m Ham", 144.0e6, 148.0e6, "FM", "5 kHz", "25 kHz"),
    # MURS: channels 1-3 narrow (11.25 kHz), 4-5 wide (20 kHz).
    Band("MURS", 151.82e6, 151.94e6, "FM", "2.5 kHz", "12.5 kHz"),
    Band("MURS", 154.57e6, 154.6e6, "FM", "5 kHz", "25 kHz"),
    Band("Marine VHF", 156.0e6, 162.025e6, "FM", "5 kHz", "25 kHz"),
    Band("NOAA Weather", 162.4e6, 162.55e6, "FM", "5 kHz", "25 kHz"),
    Band("1.25m Ham", 222.0e6, 225.0e6, "FM", "5 kHz", "25 kHz"),
    Band("70cm Ham", 420.0e6, 450.0e6, "FM", "5 kHz", "25 kHz"),
    Band("ISM 433", 433.05e6, 434.79e6, _IQ, "", "500 kHz"),
    # FRS (12.5 kHz channels) shares these with GMRS.
    Band("FRS/GMRS", 462.55e6, 462.725e6, "FM", "2.5 kHz", "12.5 kHz"),
    Band("FRS/GMRS", 467.55e6, 467.725e6, "FM", "2.5 kHz", "12.5 kHz"),
    Band("ISM 868", 863.0e6, 870.0e6, _IQ, "", "500 kHz"),
    Band("ISM 915", 902.0e6, 928.0e6, _IQ, "", "2 MHz"),
    Band("Pagers", 929.0e6, 932.0e6, "FM", "5 kHz", "25 kHz"),
    Band("ADS-B", 1089.0e6, 1091.0e6, _IQ, "", "2.4 MHz"),
)

#: Digital-mode channels: the dial frequency (USB) of FT8 and FT4 on the HF
#: bands and 6 m. Their signals lie up to 3 kHz above it.
DATA_CHANNELS: Tuple[Tuple[str, float], ...] = (
    ("FT8 160m", 1.840e6),
    ("FT8 80m", 3.573e6),
    ("FT4 80m", 3.575e6),
    ("FT4 40m", 7.0475e6),
    ("FT8 40m", 7.074e6),
    ("FT8 30m", 10.136e6),
    ("FT4 30m", 10.140e6),
    ("FT8 20m", 14.074e6),
    ("FT4 20m", 14.080e6),
    ("FT8 17m", 18.100e6),
    ("FT4 17m", 18.104e6),
    ("FT8 15m", 21.074e6),
    ("FT4 15m", 21.140e6),
    ("FT8 12m", 24.915e6),
    ("FT4 12m", 24.919e6),
    ("FT8 10m", 28.074e6),
    ("FT4 10m", 28.180e6),
    ("FT8 6m", 50.313e6),
    ("FT4 6m", 50.318e6),
)
_DATA_LISTENING = ("USB", "", "3.0 kHz")
_CW_LISTENING = ("CW", "", "500 Hz")
#: Voice modes in the frequency manager's amateur segments.
_PHONE_MODES = frozenset(("SSB", "USB", "LSB", "AM", "FM"))

#: Built-in presets that stand for a whole band rather than one channel:
#: a bookmark there is named after the band and its frequency instead.
_BAND_PRESET_CATEGORIES = frozenset(("Broadcast", "ISM", "GNSS"))
#: How close (Hz) a frequency must be to a preset to take its name.
_PRESET_TOLERANCE_HZ = 500.0


def band_for(freq_hz: float, bands: Iterable[Band] = BANDS) -> Optional[Band]:
    """The narrowest band containing ``freq_hz``, or None."""
    try:
        freq = float(freq_hz)
    except (TypeError, ValueError):
        return None
    found = [band for band in bands if band.contains(freq)]
    if not found:
        return None
    return min(found, key=lambda band: band.end_hz - band.start_hz)


def preset_for(
    freq_hz: float, presets: Iterable[FrequencyPreset] = RX_PRESETS
) -> Optional[FrequencyPreset]:
    """The built-in channel preset at ``freq_hz`` (within 500 Hz), if any.

    Presets that stand for a whole band (FM broadcast, ISM) don't count.
    """
    try:
        freq = float(freq_hz)
    except (TypeError, ValueError):
        return None
    for preset in presets:
        if preset.category in _BAND_PRESET_CATEGORIES:
            continue
        if abs(float(preset.frequency_hz) - freq) <= _PRESET_TOLERANCE_HZ:
            return preset
    return None


#: Below this, frequencies are named in kHz ("AM Broadcast 880 kHz", as
#: long- and medium-wave stations are known); from it up, in MHz.
KHZ_BELOW_HZ = 2e6


def short_khz(freq_hz: float) -> str:
    """kHz without trailing zeros: ``880e3`` -> ``"880"``, ``1260.5e3`` ->
    ``"1260.5"``, ``-300e3`` -> ``"-300"``."""
    text = f"{float(freq_hz) / 1e3:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def short_mhz(freq_hz: float) -> str:
    """MHz without trailing zeros, keeping one decimal: ``100.1e6`` ->
    ``"100.1"``, ``146.52e6`` -> ``"146.52"``, ``125e6`` -> ``"125.0"``."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def default_name(freq_hz: float) -> str:
    """A name for a new bookmark at ``freq_hz``.

    The built-in channel preset's name when there is one ("2m Calling",
    "NOAA Weather 1"); otherwise the band and the frequency ("FM
    Broadcast 100.1", "Airband 118.1"; below 2 MHz in kHz: "AM Broadcast
    880 kHz"); otherwise the frequency ("7.074 MHz" style is left to the
    caller: this returns "" then).
    """
    preset = preset_for(freq_hz)
    if preset is not None:
        return str(preset.name)
    band = band_for(freq_hz)
    if band is None:
        return ""
    if abs(freq_hz) < KHZ_BELOW_HZ:
        return f"{band.name} {short_khz(freq_hz)} kHz"
    return f"{band.name} {short_mhz(freq_hz)}"


def listening_for(freq_hz: float) -> Optional[Tuple[str, str, str]]:
    """``(mode, fm_deviation, bandwidth)`` a frequency is usually heard
    with: its preset's, its amateur band segment's (see
    :func:`segment_listening`) or its band's, or None when none says."""
    preset = preset_for(freq_hz)
    if preset is not None:
        listening = preset_listening(preset)
        if listening is not None:
            return listening
    segment = segment_listening(freq_hz)
    if segment is not None:
        return segment.listening
    band = band_for(freq_hz)
    return band.listening if band is not None else None


class SegmentListening(NamedTuple):
    """How a part of an amateur band is usually heard, where that differs
    from the band's own mode."""

    name: str  # "FT8 40m", "40m Ham, CW/data segment"
    listening: Tuple[str, str, str]  # (mode, fm_deviation, bandwidth)
    channel: bool  # a channel (FT8, FT4), not a stretch of the band


def segment_listening(freq_hz: float) -> Optional[SegmentListening]:
    """How the part of an amateur band at ``freq_hz`` is usually heard,
    where that differs from the band's own mode: a digital-mode channel of
    :data:`DATA_CHANNELS` (from 500 Hz below its dial frequency to 3 kHz
    above) in USB with a 3 kHz filter, and the CW/data segment of an SSB
    band (40 m below 7.125 MHz: no voice there) in CW with a 500 Hz
    filter. None elsewhere."""
    try:
        freq = float(freq_hz)
    except (TypeError, ValueError):
        return None
    band = band_for(freq)
    if band is None or band.mode not in ("USB", "LSB", "CW"):
        return None
    near = [
        (abs(freq - dial), name)
        for name, dial in DATA_CHANNELS
        if dial - 500.0 <= freq <= dial + 3000.0
    ]
    if near:
        return SegmentListening(min(near)[1], _DATA_LISTENING, True)
    if band.mode != "CW" and _cw_data_only(freq):
        return SegmentListening(f"{band.name}, CW/data segment", _CW_LISTENING, False)
    return None


def _cw_data_only(freq_hz: float) -> bool:
    """Whether every amateur segment at ``freq_hz`` is for CW and data only
    (no voice; at an edge shared with a phone segment, it is voice)."""
    segments = [s for s in AMATEUR_BAND_PRIVILEGES if s.start_hz <= freq_hz <= s.end_hz]
    return bool(segments) and all(
        s.modes and not {m.upper() for m in s.modes} & _PHONE_MODES for s in segments
    )


def preset_listening(preset: FrequencyPreset) -> Optional[Tuple[str, str, str]]:
    """How a built-in preset is listened to, in Mode list terms."""
    mode = str(preset.mode).strip().upper()
    width = _width_text(float(preset.bandwidth_hz))
    if mode == "WFM":
        return "FM", "75 kHz", "200 kHz"
    if mode in ("FM", "NFM"):
        deviation = "25 kHz" if preset.bandwidth_hz >= 40e3 else "5 kHz"
        return "FM", deviation, width
    if mode in ("AM", "USB", "LSB", "CW"):
        return mode, "", width
    if mode == "RAW":
        return _IQ, "", width
    return None


def _width_text(width_hz: float) -> str:
    """``25e3`` -> ``"25 kHz"``, ``2.4e6`` -> ``"2.4 MHz"``, ``500`` ->
    ``"500 Hz"`` (the Bandwidth list's spelling)."""
    if width_hz >= 1e6:
        return f"{width_hz / 1e6:g} MHz"
    if width_hz >= 1e3:
        return f"{width_hz / 1e3:g} kHz"
    return f"{width_hz:g} Hz"


__all__ = [
    "BANDS",
    "DATA_CHANNELS",
    "KHZ_BELOW_HZ",
    "Band",
    "SegmentListening",
    "band_for",
    "default_name",
    "listening_for",
    "preset_for",
    "preset_listening",
    "segment_listening",
    "short_khz",
    "short_mhz",
]
