"""
CHIRP-compatible CSV import/export for saved channels (bookmarks / memories).

`CHIRP <https://chirpmyradio.com/>`_ stores memory channels in a generic CSV
file with a fixed 21-column header::

    Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,
    DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment,URCALL,
    RPT1CALL,RPT2CALL,DVCODE

Files written here use that exact header, column order, and value formatting
(frequencies in MHz with six decimals, DTCS codes zero-padded to three digits,
tones with one decimal, tuning step with two), so a file exported from this
application opens directly in CHIRP and can be uploaded to a radio.

Reading is deliberately more forgiving than writing: columns are matched by
name (case-insensitively), any subset of the optional columns may be present,
unknown columns are ignored, and a UTF-8 BOM (added by spreadsheet editors) is
stripped. Only ``Frequency`` is required. Files saved by a spreadsheet with
semicolons or tabs between the columns (and decimal commas, "146,520000")
read too.

How this application listens to a channel (its demodulator, FM deviation
and filter width, which CHIRP's Mode column can't express, e.g. airband AM
next to "keep the current mode") rides in the Comment column as a short tag,
``[sdr mode=FM | dev=12.5 kHz | bw=25 kHz]``, together with the bookmark's
group. CHIRP and radios keep comments as they are, so a file exported here
and imported again tunes every channel as before; the tag never shows in
the application.

This module is pure Python — no PyQt6, no NumPy — so it can be used from the
GUI, the CLI, or a script without pulling in the GUI stack.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# The CHIRP generic-CSV header, in order. Do not reorder: CHIRP writes and
# expects these exact names, and some tools read the file positionally.
CHIRP_COLUMNS: Sequence[str] = (
    "Location",
    "Name",
    "Frequency",
    "Duplex",
    "Offset",
    "Tone",
    "rToneFreq",
    "cToneFreq",
    "DtcsCode",
    "DtcsPolarity",
    "RxDtcsCode",
    "CrossMode",
    "Mode",
    "TStep",
    "Skip",
    "Power",
    "Comment",
    "URCALL",
    "RPT1CALL",
    "RPT2CALL",
    "DVCODE",
)

# Values CHIRP itself accepts. Anything else is passed through with a warning
# rather than dropped, so an unusual file still round-trips.
VALID_DUPLEX = ("", "+", "-", "split", "off")
VALID_TONE_MODES = ("", "Tone", "TSQL", "DTCS", "Cross", "TSQL-R", "DTCS-R")
VALID_DTCS_POLARITY = ("NN", "NR", "RN", "RR")
VALID_SKIP = ("", "S", "P")
VALID_MODES = (
    "WFM",
    "FM",
    "NFM",
    "AM",
    "NAM",
    "DV",
    "USB",
    "LSB",
    "CW",
    "CWR",
    "NCW",
    "NCWR",
    "RTTY",
    "RTTYR",
    "DIG",
    "PKT",
    "P25",
    "DMR",
    "DN",
    "FSK",
    "FSKR",
    "Auto",
)

# CHIRP mode -> demodulator name used by the GUI's demod selector.
_MODE_TO_DEMOD = {
    "WFM": "FM",
    "FM": "FM",
    "NFM": "FM",
    "DV": "FM",
    "PKT": "FM",
    "DIG": "FM",
    "P25": "FM",
    "DMR": "FM",
    "DN": "FM",
    "AM": "AM",
    "NAM": "AM",
    "USB": "USB",
    "LSB": "LSB",
    "CW": "CW",
    "CWR": "CW",
    "NCW": "CW",
    "NCWR": "CW",
}

# Demodulator name -> CHIRP mode. "None (I/Q)" has no CHIRP equivalent, so it
# is written as FM (CHIRP's default) to keep the file importable.
_DEMOD_TO_MODE = {
    "FM": "FM",
    "WFM": "WFM",
    "NFM": "NFM",
    "AM": "AM",
    "USB": "USB",
    "LSB": "LSB",
    "CW": "CW",
    "NONE (I/Q)": "FM",
    "RAW": "FM",
}

# Longest suffixes first so "kHz" is not matched as "Hz".
_HZ_PER_UNIT = (("ghz", 1e9), ("mhz", 1e6), ("khz", 1e3), ("hz", 1.0))

DEFAULT_DEMOD = "None (I/Q)"

# ---- How the application listens to a channel (the Comment tag) ---------- #
#: Bookmark keys the tag carries: how to listen, and the group.
LISTEN_KEYS = ("demod", "fm_deviation", "bandwidth", "group")
#: Demodulator names of the application's Mode list.
APP_DEMODS = ("AM", "FM", "USB", "LSB", "CW", DEFAULT_DEMOD)
#: The tag's word for "keep whatever mode the receiver is in" (a bookmark
#: saved without a mode) and for the no-audio mode.
_KEEP = "keep"
_IQ = "I/Q"
_TAG_RE = re.compile(r"\s*\[sdr\s+([^\]]*)\]\s*$")
_TAG_FIELDS = (("mode", "demod"), ("dev", "fm_deviation"), ("bw", "bandwidth"))


def _tag_value(value: Any) -> str:
    """A tag value: no separators or brackets inside it."""
    return re.sub(r"[|;\]\[=]", ",", str(value or "")).strip()


def listening_tag(bookmark: Mapping[str, Any]) -> str:
    """The Comment tag for how ``bookmark`` is listened to ("" for none).

    A bookmark with a ``demod`` (a Mode list name) gets its mode, FM
    deviation and width; one saved by this application without a mode
    gets ``mode=keep``; a CHIRP channel that only has CHIRP's own Mode
    gets no mode in the tag. The group goes in the tag too.
    """
    parts: List[str] = []
    demod = str(bookmark.get("demod") or "")
    if demod in APP_DEMODS:
        parts.append(f"mode={_IQ if demod == DEFAULT_DEMOD else demod}")
        deviation = _tag_value(bookmark.get("fm_deviation"))
        if demod == "FM" and deviation:
            parts.append(f"dev={deviation}")
        width = _tag_value(bookmark.get("bandwidth"))
        if width:
            parts.append(f"bw={width}")
    elif not is_chirp_channel(bookmark):
        parts.append(f"mode={_KEEP}")
    group = _tag_value(bookmark.get("group"))
    if group:
        parts.append(f"group={group}")
    # "|" between the fields: no CSV dialect separates columns with it, so
    # the tag survives a spreadsheet that saves with ";" or tabs.
    return f"[sdr {' | '.join(parts)}]" if parts else ""


def split_listening_tag(comment: str) -> Tuple[str, Dict[str, str]]:
    """``(comment without the tag, fields)`` for a Comment field.

    ``fields`` maps bookmark keys (``demod``, ``fm_deviation``,
    ``bandwidth``, ``group``) to the tag's values; ``demod`` is ``"keep"``
    for a bookmark that keeps the current mode. No tag: ``(comment, {})``.
    """
    text = str(comment or "")
    match = _TAG_RE.search(text)
    if match is None:
        return text, {}
    found: Dict[str, str] = {}
    for item in re.split(r"[|;]", match.group(1)):
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        for tag_key, bookmark_key in _TAG_FIELDS + (("group", "group"),):
            if key == tag_key and value:
                found[bookmark_key] = value
    demod = found.get("demod", "")
    if demod:
        if demod.lower() == _KEEP:
            found["demod"] = _KEEP
        elif demod.upper() in ("I/Q", "IQ", DEFAULT_DEMOD.upper()):
            found["demod"] = DEFAULT_DEMOD
        else:
            known = {d.upper(): d for d in APP_DEMODS}
            if demod.upper() in known:
                found["demod"] = known[demod.upper()]
            else:
                del found["demod"]  # not a mode this application has
    return text[: match.start()].rstrip(), found


def is_chirp_channel(bookmark: Mapping[str, Any]) -> bool:
    """Whether a stored bookmark came from a CHIRP channel: it has a CHIRP
    field this application doesn't set itself (a memory location, CHIRP's
    Mode, a repeater shift, a tone, a power level...). Such a channel
    without a ``demod`` is listened to in its CHIRP Mode (FM by default);
    a bookmark of this application without one keeps the current mode."""
    own = {"name", "frequency_hz", "comment", "skip"}
    chirp_only = {f.name for f in fields(Channel)} - own
    return bool(chirp_only & set(bookmark))


def chirp_mode_for(demod: str, fm_deviation: str = "") -> str:
    """CHIRP's Mode for an application mode: FM at 75 kHz is WFM, at 2.5
    kHz NFM; AM, USB, LSB and CW are themselves; I/Q (no CHIRP mode) is
    written as FM, CHIRP's default."""
    if demod == "FM":
        deviation = str(fm_deviation or "").replace(" ", "").lower()
        if deviation == "75khz":
            return "WFM"
        if deviation == "2.5khz":
            return "NFM"
        return "FM"
    if demod in ("AM", "USB", "LSB", "CW"):
        return demod
    return "FM"


class ChirpCsvError(ValueError):
    """Raised when a CSV file cannot be read as CHIRP channel data."""


def parse_freq(value: Union[str, float, int, None]) -> float:
    """
    Parse a CHIRP frequency field into Hz.

    CHIRP writes frequencies in MHz ("146.520000"), so a bare number is
    interpreted as MHz. A unit suffix ("146520 kHz") is honoured if present,
    and thousands separators are ignored. Blank values mean 0 Hz.

    Args:
        value: Field text, or a number already expressed in MHz.

    Returns:
        Frequency in Hz.

    Raises:
        ValueError: If the text is not a number.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(Decimal(str(value)) * 1_000_000)

    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0

    multiplier = 1e6  # bare numbers are MHz
    lowered = text.lower()
    for suffix, factor in _HZ_PER_UNIT:
        if lowered.endswith(suffix):
            text = text[: -len(suffix)].strip()
            multiplier = factor
            break

    try:
        # Decimal keeps 146.520000 MHz exactly 146520000 Hz instead of
        # 146520000.00000003.
        return float(Decimal(text) * Decimal(multiplier))
    except (InvalidOperation, ArithmeticError) as exc:
        raise ValueError(f"invalid frequency: {value!r}") from exc


def format_freq(freq_hz: float) -> str:
    """Format Hz as a CHIRP frequency field (MHz, six decimals)."""
    return f"{freq_hz / 1e6:0.6f}"


def chirp_mode_to_demod(mode: str) -> str:
    """Map a CHIRP mode name to the GUI's demodulator name."""
    return _MODE_TO_DEMOD.get(str(mode).strip().upper(), DEFAULT_DEMOD)


def demod_to_chirp_mode(demod: str) -> str:
    """Map a GUI demodulator name to a CHIRP mode name."""
    return _DEMOD_TO_MODE.get(str(demod).strip().upper(), "FM")


def _to_float(value: Any, default: float) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        text = str(value).strip()
        # int("023") is fine; int("023.0") is not, so go through float.
        return int(float(text)) if text else default
    except (TypeError, ValueError):
        return default


def _choice(value: Any, valid: Sequence[str], default: str, label: str) -> str:
    """Normalise a constrained field, keeping unknown values with a warning."""
    text = str(value or "").strip()
    if not text:
        return default
    for candidate in valid:
        if candidate and text.lower() == candidate.lower():
            return candidate
    logger.warning("Unrecognised %s value %r; keeping it as-is", label, text)
    return text


@dataclass
class Channel:
    """
    One saved channel, mirroring a CHIRP memory.

    Frequencies are stored in Hz (the convention used everywhere else in this
    package) and converted to/from CHIRP's MHz text at the file boundary.

    Attributes:
        name: Channel name / label.
        frequency_hz: Receive frequency in Hz.
        location: Memory slot number, or None to number sequentially on export.
        duplex: "", "+", "-", "split" (offset holds the TX frequency), or "off".
        offset_hz: Repeater shift in Hz, or the TX frequency when duplex="split".
        tone_mode: "", "Tone", "TSQL", "DTCS", "Cross", "TSQL-R", "DTCS-R".
        rtone: TX CTCSS tone in Hz.
        ctone: RX CTCSS tone in Hz.
        dtcs: TX DTCS code.
        dtcs_polarity: DTCS polarity ("NN", "NR", "RN", "RR").
        rx_dtcs: RX DTCS code (used with Cross modes).
        cross_mode: Cross-tone mode, e.g. "Tone->Tone".
        mode: CHIRP mode name (FM, NFM, AM, USB, ...).
        tuning_step_khz: Tuning step in kHz.
        skip: "" (scan), "S" (skip), or "P" (priority).
        power: Free-form power label, e.g. "5.0W".
        comment: Free-form comment.
        urcall/rpt1call/rpt2call/dv_code: D-STAR fields, blank for most radios.
    """

    name: str = ""
    frequency_hz: float = 0.0
    location: Optional[int] = None
    duplex: str = ""
    offset_hz: float = 0.0
    tone_mode: str = ""
    rtone: float = 88.5
    ctone: float = 88.5
    dtcs: int = 23
    dtcs_polarity: str = "NN"
    rx_dtcs: int = 23
    cross_mode: str = "Tone->Tone"
    mode: str = "FM"
    tuning_step_khz: float = 5.0
    skip: str = ""
    power: str = ""
    comment: str = ""
    urcall: str = ""
    rpt1call: str = ""
    rpt2call: str = ""
    dv_code: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.frequency_hz = float(self.frequency_hz)
        self.offset_hz = float(self.offset_hz)
        if self.frequency_hz < 0:
            raise ValueError(f"negative frequency: {self.frequency_hz}")
        self.duplex = _choice(self.duplex, VALID_DUPLEX, "", "Duplex")
        self.tone_mode = _choice(self.tone_mode, VALID_TONE_MODES, "", "Tone")
        self.dtcs_polarity = _choice(
            self.dtcs_polarity, VALID_DTCS_POLARITY, "NN", "DtcsPolarity"
        )
        self.skip = _choice(self.skip, VALID_SKIP, "", "Skip")
        self.mode = _choice(self.mode, VALID_MODES, "FM", "Mode")

    @property
    def demod(self) -> str:
        """The GUI demodulator name matching this channel's CHIRP mode."""
        return chirp_mode_to_demod(self.mode)

    @property
    def tx_frequency_hz(self) -> float:
        """
        Transmit frequency implied by duplex/offset.

        Returns the RX frequency for simplex channels. Returns 0.0 when duplex
        is "off" (receive only).
        """
        if self.duplex == "+":
            return self.frequency_hz + abs(self.offset_hz)
        if self.duplex == "-":
            return self.frequency_hz - abs(self.offset_hz)
        if self.duplex == "split":
            return self.offset_hz
        if self.duplex == "off":
            return 0.0
        return self.frequency_hz

    def to_row(self, location: Optional[int] = None) -> Dict[str, str]:
        """
        Render this channel as a CHIRP CSV row.

        Args:
            location: Overrides the channel's own location (used when a writer
                numbers rows sequentially).

        Returns:
            Mapping of CHIRP column name to formatted text.
        """
        slot = location if location is not None else self.location
        return {
            "Location": "" if slot is None else str(int(slot)),
            "Name": self.name,
            "Frequency": format_freq(self.frequency_hz),
            "Duplex": self.duplex,
            "Offset": format_freq(self.offset_hz),
            "Tone": self.tone_mode,
            "rToneFreq": f"{self.rtone:.1f}",
            "cToneFreq": f"{self.ctone:.1f}",
            "DtcsCode": f"{int(self.dtcs):03d}",
            "DtcsPolarity": self.dtcs_polarity,
            "RxDtcsCode": f"{int(self.rx_dtcs):03d}",
            "CrossMode": self.cross_mode,
            "Mode": self.mode,
            "TStep": f"{self.tuning_step_khz:.2f}",
            "Skip": self.skip,
            "Power": self.power,
            "Comment": self.comment,
            "URCALL": self.urcall,
            "RPT1CALL": self.rpt1call,
            "RPT2CALL": self.rpt2call,
            "DVCODE": self.dv_code,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Channel":
        """
        Build a channel from a CHIRP CSV row.

        Column names are matched case-insensitively; missing or blank columns
        fall back to CHIRP's defaults, and unknown columns are ignored.

        Args:
            row: Mapping of column name to text.

        Returns:
            The parsed channel.

        Raises:
            ValueError: If the Frequency column is present but unparseable.
        """
        get = {str(k).strip().lower(): v for k, v in row.items() if k is not None}.get

        location_text = str(get("location", "") or "").strip()
        return cls(
            name=str(get("name", "") or "").strip(),
            frequency_hz=parse_freq(get("frequency", "")),
            location=_to_int(location_text, 0) if location_text else None,
            duplex=str(get("duplex", "") or ""),
            offset_hz=parse_freq(get("offset", "")),
            tone_mode=str(get("tone", "") or ""),
            rtone=_to_float(get("rtonefreq", ""), 88.5),
            ctone=_to_float(get("ctonefreq", ""), 88.5),
            dtcs=_to_int(get("dtcscode", ""), 23),
            dtcs_polarity=str(get("dtcspolarity", "") or ""),
            rx_dtcs=_to_int(get("rxdtcscode", ""), 23),
            cross_mode=str(get("crossmode", "") or "").strip() or "Tone->Tone",
            mode=str(get("mode", "") or ""),
            tuning_step_khz=_to_float(get("tstep", ""), 5.0),
            skip=str(get("skip", "") or ""),
            power=str(get("power", "") or "").strip(),
            comment=str(get("comment", "") or "").strip(),
            urcall=str(get("urcall", "") or "").strip(),
            rpt1call=str(get("rpt1call", "") or "").strip(),
            rpt2call=str(get("rpt2call", "") or "").strip(),
            dv_code=str(get("dvcode", "") or "").strip(),
        )

    # ---- Bookmark bridge -------------------------------------------------
    def to_bookmark(self) -> Dict[str, Any]:
        """
        Convert to the bookmark dict shape stored in GUI settings.

        "label" and "freq_hz" are kept as the primary keys for backward
        compatibility with bookmarks saved before CSV support existed; the
        remaining CHIRP fields ride along so a GUI round-trip is lossless.
        A listening tag in the comment (see :func:`listening_tag`) becomes
        the bookmark's ``demod``, ``fm_deviation``, ``bandwidth`` and
        ``group`` again.
        """
        data: Dict[str, Any] = {
            "label": self.name,
            "freq_hz": self.frequency_hz,
        }
        comment, listen = split_listening_tag(self.comment)
        defaults = Channel()
        for f in fields(self):
            if f.name in ("name", "frequency_hz"):
                continue
            value = comment if f.name == "comment" else getattr(self, f.name)
            if value != getattr(defaults, f.name):
                data[f.name] = value
        demod = listen.get("demod", "")
        if demod == _KEEP:
            # Saved without a mode: it keeps the receiver's mode, as before
            # (its memory slot and CHIRP's default Mode don't change that).
            data.pop("location", None)
            data.pop("mode", None)
        elif demod:
            data["demod"] = demod
            if demod == "FM" and listen.get("fm_deviation"):
                data["fm_deviation"] = listen["fm_deviation"]
            if listen.get("bandwidth"):
                data["bandwidth"] = listen["bandwidth"]
        if listen.get("group"):
            data["group"] = listen["group"]
        return data

    @classmethod
    def from_bookmark(cls, bookmark: Mapping[str, Any]) -> "Channel":
        """
        Build a channel from a bookmark dict stored in GUI settings.

        Accepts legacy bookmarks that only carry "label" and "freq_hz". A
        bookmark's listening mode sets CHIRP's Mode (AM, WFM...) and, with
        its group, goes in the comment's listening tag.
        """
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {
            key: value for key, value in bookmark.items() if key in known
        }
        kwargs["name"] = str(bookmark.get("label", bookmark.get("name", "")) or "")
        kwargs["frequency_hz"] = float(
            bookmark.get("freq_hz", bookmark.get("frequency_hz", 0.0)) or 0.0
        )
        demod = str(bookmark.get("demod") or "")
        if demod in APP_DEMODS:
            kwargs["mode"] = chirp_mode_for(
                demod, str(bookmark.get("fm_deviation") or "")
            )
        tag = listening_tag(bookmark)
        if tag:
            comment, _old = split_listening_tag(str(kwargs.get("comment") or ""))
            kwargs["comment"] = f"{comment} {tag}".strip()
        return cls(**kwargs)


@dataclass
class ImportReport:
    """Outcome of reading a CSV file: the channels plus any skipped rows."""

    channels: List[Channel] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)


def _read_text(source: Union[str, Path, TextIO]) -> str:
    """The whole text of a path or an open text stream, without a BOM.

    A file that isn't UTF-8 (a spreadsheet saved it in the system's
    encoding) is read as Windows-1252, so accented names still import.
    """
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists():
            raise ChirpCsvError(f"file not found: {path}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ChirpCsvError(f"can't read {path}: {exc}") from exc
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1252", errors="replace")
    text = source.read()
    return text[1:] if text.startswith("﻿") else text


#: Column separators a spreadsheet may have saved the file with.
_DELIMITERS = (",", ";", "\t")
#: Number columns, which may use a decimal comma when the columns are
#: separated by semicolons or tabs ("146,520000").
_NUMBER_COLUMNS = ("frequency", "offset", "rtonefreq", "ctonefreq", "tstep")
_DECIMAL_COMMA = re.compile(r"^\s*[-+]?\d+,\d+\s*$")


def _detect_delimiter(text: str) -> str:
    """The column separator of a CSV text: the one its header uses most."""
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(_DELIMITERS, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def _fix_decimal_commas(row: Dict[str, Any]) -> Dict[str, Any]:
    """``"146,520000"`` -> ``"146.520000"`` in the number columns."""
    for key, value in row.items():
        if (
            key is not None
            and str(key).strip().lower() in _NUMBER_COLUMNS
            and isinstance(value, str)
            and _DECIMAL_COMMA.match(value)
        ):
            row[key] = value.replace(",", ".")
    return row


def read_chirp_csv(
    source: Union[str, Path, TextIO], *, strict: bool = True
) -> ImportReport:
    """
    Read channels from a CHIRP generic-CSV file.

    The columns may be separated by commas (CHIRP), semicolons or tabs (a
    spreadsheet's export); with semicolons or tabs the numbers may use a
    decimal comma.

    Args:
        source: Path to the CSV file, or an open text stream.
        strict: When True (default), a malformed row raises. When False, bad
            rows are skipped and described in the report's ``skipped`` list.

    Returns:
        An ImportReport holding the parsed channels (and skipped-row notes).

    Raises:
        ChirpCsvError: If the file is missing, has no header, lacks a
            "Frequency" column, or (in strict mode) contains a bad row.
    """
    text = _read_text(source)
    delimiter = _detect_delimiter(text)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    if not reader.fieldnames:
        raise ChirpCsvError("CSV file is empty (no header row)")

    headers = {
        str(name).strip().lower() for name in reader.fieldnames if name is not None
    }
    if "frequency" not in headers:
        raise ChirpCsvError(
            "not a CHIRP CSV: no 'Frequency' column "
            f"(found: {', '.join(str(n) for n in reader.fieldnames)})"
        )

    report = ImportReport()
    for line_no, row in enumerate(reader, start=2):
        # Blank lines and CHIRP's empty memory slots have no frequency.
        if not any(str(v or "").strip() for v in row.values()):
            continue
        if delimiter != ",":
            row = _fix_decimal_commas(row)
        try:
            channel = Channel.from_row(row)
        except ValueError as exc:
            message = f"line {line_no}: {exc}"
            if strict:
                raise ChirpCsvError(message) from exc
            logger.warning("Skipping bad channel row: %s", message)
            report.skipped.append(message)
            continue
        if channel.frequency_hz <= 0:
            message = f"line {line_no}: missing or zero frequency"
            if strict:
                raise ChirpCsvError(message)
            report.skipped.append(message)
            continue
        report.channels.append(channel)
    return report


def write_chirp_csv(
    destination: Union[str, Path, TextIO],
    channels: Iterable[Channel],
    *,
    renumber: bool = False,
    start_location: int = 0,
) -> int:
    """
    Write channels to a CHIRP generic-CSV file.

    Args:
        destination: Output path, or an open text stream (opened with
            ``newline=""``).
        channels: Channels to write, in order.
        renumber: When True, ignore each channel's location and number rows
            sequentially from ``start_location``.
        start_location: First memory slot number to assign.

    Returns:
        The number of channels written.
    """
    channels = list(channels)

    if isinstance(destination, (str, Path)):
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            return _write_rows(handle, channels, renumber, start_location)
    return _write_rows(destination, channels, renumber, start_location)


def _write_rows(
    stream: TextIO,
    channels: Sequence[Channel],
    renumber: bool,
    start_location: int,
) -> int:
    writer = csv.DictWriter(stream, fieldnames=list(CHIRP_COLUMNS))
    writer.writeheader()
    next_slot = start_location
    for channel in channels:
        if renumber or channel.location is None:
            slot = next_slot
        else:
            slot = channel.location
        next_slot = slot + 1
        writer.writerow(channel.to_row(location=slot))
    return len(channels)


# =============================================================================
# Bookmark-level helpers (what the GUI and CLI actually call)
# =============================================================================


def bookmarks_to_channels(bookmarks: Iterable[Mapping[str, Any]]) -> List[Channel]:
    """Convert stored bookmark dicts to channels."""
    return [Channel.from_bookmark(b) for b in bookmarks]


def channels_to_bookmarks(channels: Iterable[Channel]) -> List[Dict[str, Any]]:
    """Convert channels to the bookmark dict shape stored in GUI settings."""
    return [c.to_bookmark() for c in channels]


def export_bookmarks_csv(
    destination: Union[str, Path, TextIO],
    bookmarks: Iterable[Mapping[str, Any]],
    *,
    renumber: bool = True,
) -> int:
    """
    Write saved bookmarks to a CHIRP CSV file.

    Args:
        destination: Output path or open text stream.
        bookmarks: Bookmark dicts as stored in GUI settings.
        renumber: Number memory slots sequentially from 0 (CHIRP's usual shape).

    Returns:
        Number of channels written.
    """
    return write_chirp_csv(
        destination, bookmarks_to_channels(bookmarks), renumber=renumber
    )


def import_bookmarks_csv(
    source: Union[str, Path, TextIO], *, strict: bool = True
) -> List[Dict[str, Any]]:
    """
    Read a CHIRP CSV file into bookmark dicts ready for GUI settings.

    Args:
        source: Path to the CSV file, or an open text stream.
        strict: Propagate malformed rows as errors instead of skipping them.

    Returns:
        List of bookmark dicts.

    Raises:
        ChirpCsvError: If the file is not readable as CHIRP channel data.
    """
    return channels_to_bookmarks(read_chirp_csv(source, strict=strict).channels)


def presets_to_channels(presets: Iterable[Any]) -> List[Channel]:
    """
    Convert built-in `FrequencyPreset` objects to channels.

    Lets the shipped RX preset list be exported to CHIRP without the GUI.
    Each channel's comment carries the preset's listening tag (its mode and
    width), so importing the file here tunes every preset as the
    application's own preset list does.
    """
    from .band_plan import preset_listening

    channels: List[Channel] = []
    for preset in presets:
        comment = str(getattr(preset, "description", ""))
        listening = preset_listening(preset)
        if listening is not None:
            demod, deviation, width = listening
            tag = listening_tag(
                {"demod": demod, "fm_deviation": deviation, "bandwidth": width}
            )
            comment = f"{comment} {tag}".strip()
        channels.append(
            Channel(
                name=str(getattr(preset, "name", "")),
                frequency_hz=float(getattr(preset, "frequency_hz", 0.0)),
                mode=demod_to_chirp_mode(str(getattr(preset, "mode", "FM"))),
                comment=comment,
            )
        )
    return channels


__all__ = [
    "APP_DEMODS",
    "LISTEN_KEYS",
    "chirp_mode_for",
    "is_chirp_channel",
    "listening_tag",
    "split_listening_tag",
    "CHIRP_COLUMNS",
    "VALID_DUPLEX",
    "VALID_TONE_MODES",
    "VALID_DTCS_POLARITY",
    "VALID_SKIP",
    "VALID_MODES",
    "ChirpCsvError",
    "Channel",
    "ImportReport",
    "parse_freq",
    "format_freq",
    "chirp_mode_to_demod",
    "demod_to_chirp_mode",
    "read_chirp_csv",
    "write_chirp_csv",
    "bookmarks_to_channels",
    "channels_to_bookmarks",
    "export_bookmarks_csv",
    "import_bookmarks_csv",
    "presets_to_channels",
]
