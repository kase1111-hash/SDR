"""
The band plan at a frequency, in one line.

For the Info tab's "Band plan" row and the FREQ readout's tooltip (the
receiver shows what it is tuned to; hams also want the segment, its modes,
who may transmit there and the power limit)::

    >>> band_plan_line(14.06e6)
    '20m Ham · 14.025–14.15 MHz · General and up: CW/Data, max 1500 W'
    >>> band_plan_line(100.1e6)
    'FM Broadcast · 87.5–108 MHz'

:func:`band_plan_details` adds what your own license allows there. The
tables are the frequency manager's (US FCC Part 97 segments, the
license-free CB/MURS/FRS channels and the transmit lockouts) and the
receive bands of :mod:`sdr_module.core.band_plan`.

Pure Python (no Qt).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from .band_plan import band_for
from .frequency_manager import (
    AMATEUR_BAND_PRIVILEGES,
    LICENSE_FREE_BANDS,
    TX_LOCKOUT_BANDS,
    BandPrivilege,
    LicenseClass,
)

#: The general amateur power limit (FCC 97.313: 1500 W PEP) where a
#: segment names none.
AMATEUR_POWER_LIMIT_W = 1500.0

_CLASS_ORDER = (
    LicenseClass.NONE,
    LicenseClass.TECHNICIAN,
    LicenseClass.GENERAL,
    LicenseClass.AMATEUR_EXTRA,
)
_CLASS_NAMES = {
    LicenseClass.NONE: "No license",
    LicenseClass.TECHNICIAN: "Technician",
    LicenseClass.GENERAL: "General",
    LicenseClass.AMATEUR_EXTRA: "Extra",
}
_PHONE = {"SSB", "USB", "LSB", "AM"}


def _mhz(freq_hz: float) -> str:
    """``14.025e6`` -> ``"14.025"``, ``144e6`` -> ``"144"``, ``87.5e6`` ->
    ``"87.5"``."""
    return f"{float(freq_hz) / 1e6:.4f}".rstrip("0").rstrip(".")


def _range(low: float, high: float) -> str:
    return f"{_mhz(low)}–{_mhz(high)} MHz"


def describe_modes(modes: Iterable[str]) -> str:
    """``{"CW", "DATA"}`` -> ``"CW/Data"``; nothing -> ``"all modes"``."""
    modes = {str(m).upper() for m in modes}
    if not modes:
        return "all modes"
    parts: List[str] = []
    if "CW" in modes:
        parts.append("CW")
    phone = modes & _PHONE
    if "SSB" in phone or len(phone) >= 2:
        parts.append("Phone")
    elif phone:
        parts.append(next(iter(phone)))
    if "FM" in modes:
        parts.append("FM")
    if "DATA" in modes:
        parts.append("Data")
    return "/".join(parts)


def _lowest_class(licenses: Set[LicenseClass]) -> LicenseClass:
    return min(licenses, key=_CLASS_ORDER.index)


def _who(licenses: Set[LicenseClass]) -> str:
    """Who a segment is listed for: "General and up" when it runs up to
    Amateur Extra, else the classes ("Technician")."""
    if not licenses:
        return "no one"
    ranks = sorted(_CLASS_ORDER.index(c) for c in licenses)
    lowest = _CLASS_ORDER[ranks[0]]
    if lowest == LicenseClass.NONE:
        return "license-free"
    top = len(_CLASS_ORDER) - 1
    if ranks == list(range(ranks[0], top + 1)):
        name = _CLASS_NAMES[lowest]
        return name if ranks[0] == top else f"{name} and up"
    return "/".join(_CLASS_NAMES[_CLASS_ORDER[r]] for r in ranks)


def _power(segment: BandPrivilege) -> float:
    limit = segment.max_power_watts
    return float(limit) if limit is not None else AMATEUR_POWER_LIMIT_W


def _watts(watts: float) -> str:
    return f"{watts:g} W"


def _segments(freq_hz: float, table: Iterable[BandPrivilege]) -> List[BandPrivilege]:
    return [s for s in table if s.start_hz <= freq_hz <= s.end_hz]


def _lockout_note(freq_hz: float) -> str:
    for band in TX_LOCKOUT_BANDS:
        if band.start_hz <= freq_hz <= band.end_hz:
            return f"transmitting locked out ({band.name})"
    return ""


def _ham_band_name(segment: BandPrivilege) -> str:
    """The amateur band a segment belongs to ("40m Ham"), from the receive
    bands (looked up in the middle of the segment, so the ISM 433 band
    inside 70 cm doesn't name the 70 cm segment)."""
    band = band_for(0.5 * (segment.start_hz + segment.end_hz))
    if band is not None and band.start_hz <= segment.start_hz <= band.end_hz:
        return band.name
    return segment.name.split(" ")[0]


def _segment_text(segment: BandPrivilege, with_range: bool) -> str:
    """``"General and up: CW/Data, max 1500 W"`` (and the segment's range
    when it differs from the one the line names)."""
    where = f" {_range(segment.start_hz, segment.end_hz)}" if with_range else ""
    return (
        f"{_who(segment.licenses)}: {describe_modes(segment.modes)}{where}, max "
        f"{_watts(_power(segment))}"
    )


def band_plan_line(freq_hz: float) -> str:
    """One line on the band plan at ``freq_hz`` ("" when the app knows no
    band there): the band and the segment, then who may transmit there,
    in which modes and with how much power (several segments can overlap:
    the one for the most license classes first), or the license-free
    channel, or the receive band; and a transmit lockout."""
    try:
        freq = float(freq_hz)
    except (TypeError, ValueError):
        return ""
    parts: List[str] = []
    free = _segments(freq, LICENSE_FREE_BANDS)
    ham = _segments(freq, AMATEUR_BAND_PRIVILEGES)
    if free:
        segment = free[0]
        parts.append(f"{segment.name} · {_range(segment.start_hz, segment.end_hz)}")
        parts.append(
            f"license-free: {describe_modes(segment.modes)}, max "
            f"{_watts(_power(segment))}"
        )
    elif ham:
        ham.sort(key=lambda s: (-len(s.licenses), -(s.end_hz - s.start_hz), s.start_hz))
        primary = ham[0]
        span = (primary.start_hz, primary.end_hz)
        parts.append(f"{_ham_band_name(primary)} · {_range(*span)}")
        parts.append(
            "; ".join(
                _segment_text(s, with_range=(s.start_hz, s.end_hz) != span) for s in ham
            )
        )
        band = band_for(freq)
        if band is not None and band.name != _ham_band_name(primary):
            parts.append(f"in {band.name} ({_range(band.start_hz, band.end_hz)})")
    else:
        band = band_for(freq)
        if band is not None:
            parts.append(f"{band.name} · {_range(band.start_hz, band.end_hz)}")
    lockout = _lockout_note(freq)
    if lockout:
        parts.append(lockout)
    return " · ".join(parts)


def license_note(freq_hz: float, license_class: Optional[LicenseClass]) -> str:
    """What ``license_class`` may do at ``freq_hz``: "Your license
    (General) allows CW/Data here, up to 1500 W." or "Your license
    (Technician) doesn't include 14.060 MHz." ("" outside the amateur and
    license-free bands)."""
    if license_class is None:
        return ""
    freq = float(freq_hz)
    name = _CLASS_NAMES.get(license_class, str(license_class))
    if _lockout_note(freq):
        return "Nobody may transmit here: the app locks transmitting out."
    for segment in _segments(freq, LICENSE_FREE_BANDS):
        if segment.is_granted_to(license_class):
            return (
                f"Anyone may transmit here ({describe_modes(segment.modes)}, up "
                f"to {_watts(_power(segment))})."
            )
    ham = _segments(freq, AMATEUR_BAND_PRIVILEGES)
    if not ham:
        return ""
    if license_class == LicenseClass.NONE:
        return "Transmitting here needs an amateur license (set it in Ham > Ham ID)."
    usable = [s for s in ham if s.is_granted_to(license_class)]
    if not usable:
        return f"Your license ({name}) doesn't include {_mhz(freq)} MHz."
    own = [s for s in usable if license_class in s.licenses] or usable
    segment = min(own, key=_power)
    return (
        f"Your license ({name}) allows {describe_modes(segment.modes)} here, up "
        f"to {_watts(_power(segment))}."
    )


def band_plan_details(
    freq_hz: float, license_class: Optional[LicenseClass] = None
) -> Tuple[str, str]:
    """``(line, note)``: :func:`band_plan_line` and :func:`license_note`."""
    return band_plan_line(freq_hz), license_note(freq_hz, license_class)


__all__ = [
    "AMATEUR_POWER_LIMIT_W",
    "band_plan_details",
    "band_plan_line",
    "describe_modes",
    "license_note",
]
