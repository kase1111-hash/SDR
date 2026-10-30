"""
Recordings on disk: what a file holds, and asking for what it doesn't say.

A SigMF recording (and a raw or WAV recording this app saved, which gets a
SigMF-style ``<name>.sigmf-meta`` beside it) says how its samples are
stored, how fast they were taken and where it was tuned. A headerless file
(an rtl_sdr ``.bin``, a hackrf_transfer ``.iq``, a GNU Radio ``.dat``) says
none of that: :class:`ImportOptionsDialog` asks, prefilled from the file's
extension, its content and common file-name patterns
(:func:`guess_from_filename`), and checks the samples as you choose a
format. File > Open Recording, Convert Recording, Decode Recording and
Recording Info all use it.

:class:`RecordingInfoDialog` is File > Recording Info.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Set, Tuple

from ..dsp.recording import (
    FileFormat,
    IQPlayer,
    SampleCheck,
    SampleFormat,
    bytes_per_complex_sample,
    check_iq_samples,
    detect_format,
    get_file_info,
    guess_sample_format,
    sigmf_format,
)
from .file_dialogs import display_path

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import set_role, set_tone
    from .tuning import ACCEPTABLE, parse_frequency_text

logger = logging.getLogger(__name__)

#: Sample formats a headerless file can hold, as the dialog lists them.
FORMAT_CHOICES: Tuple[Tuple[SampleFormat, str], ...] = (
    (SampleFormat.UINT8, "cu8 – unsigned 8-bit (rtl_sdr)"),
    (SampleFormat.INT8, "cs8 – signed 8-bit (hackrf_transfer)"),
    (SampleFormat.INT16, "cs16 – signed 16-bit"),
    (SampleFormat.FLOAT32, "cf32 – 32-bit float (GNU Radio, GQRX)"),
    (SampleFormat.FLOAT64, "cf64 – 64-bit float"),
)
FORMAT_NAMES: Dict[SampleFormat, str] = dict(FORMAT_CHOICES)

#: The format a file's extension stands for (for a headerless file, only a
#: suggestion: the dialog asks). ``.bin`` is rtl_sdr's unsigned 8-bit.
EXTENSION_FORMATS: Dict[str, SampleFormat] = {
    ".cu8": SampleFormat.UINT8,
    ".cs8": SampleFormat.INT8,
    ".cs16": SampleFormat.INT16,
    ".cf32": SampleFormat.FLOAT32,
    ".cf64": SampleFormat.FLOAT64,
    ".bin": SampleFormat.UINT8,
    ".raw": SampleFormat.FLOAT32,
}

#: File dialog filter of every recording this app reads.
OPEN_FILTERS = ";;".join(
    (
        "I/Q recordings (*.sigmf-data *.sigmf-meta *.cf32 *.cs16 *.cs8 *.cu8 "
        "*.cf64 *.wav *.bin *.raw *.iq *.dat)",
        "SigMF (*.sigmf-data *.sigmf-meta)",
        "rtl_sdr / rtl_433 (*.bin *.cu8)",
        "HackRF (*.iq *.cs8 *.bin)",
        "WAV I/Q (*.wav)",
        "All files (*)",
    )
)

# Sample rates SDRs commonly run at: a bare number in a file name is read as
# a rate only if it is one of these (a time like 120000 is not).
_COMMON_RATES = {
    226000,
    240000,
    250000,
    256000,
    288000,
    300000,
    900001,
    960000,
    1000000,
    1024000,
    1200000,
    1400000,
    1536000,
    1600000,
    1800000,
    1920000,
    2000000,
    2048000,
    2400000,
    2560000,
    2880000,
    3000000,
    3200000,
    4000000,
    5000000,
    6000000,
    8000000,
    10000000,
    12500000,
    16000000,
    20000000,
}
_UNIT = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}
_MIN_CENTER_HZ = 1e3
_MAX_CENTER_HZ = 6e9
#: Frequencies the playback can be tuned to: what the receiver tunes.
PLAYBACK_RANGE = (_MIN_CENTER_HZ, _MAX_CENTER_HZ)


@dataclass
class FileNameHints:
    """What a recording's file name says."""

    sample_rate: Optional[float] = None
    center_hz: Optional[float] = None
    sample_format: Optional[SampleFormat] = None


def split_name(name: str) -> Tuple[str, str]:
    """``("capture", ".sigmf-data")`` for ``capture.sigmf-data``; a name
    without an extension (``capture_433.92MHz``) is all stem."""
    base = os.path.basename(name)
    lower = base.lower()
    for ext in (".sigmf-data", ".sigmf-meta"):
        if lower.endswith(ext):
            return base[: -len(ext)], base[-len(ext) :]
    stem, ext = os.path.splitext(base)
    if ext and not re.fullmatch(r"\.[A-Za-z][A-Za-z0-9-]{0,11}", ext):
        return base, ""  # ".92MHz" is part of the name
    return stem, ext


def _number(text: str) -> float:
    return float(text.replace("p", "."))


def guess_from_filename(name: str) -> FileNameHints:
    """Sample rate, centre frequency and format from a file name, as SDR
    programs name their recordings, e.g.:

    * ``iq_100.100MHz_2.4MSps_20261030_120000.sigmf-data`` (this app)
    * ``gqrx_20230101_123456_100100000_2400000_fc.raw`` (GQRX)
    * ``SDRSharp_20230101_123456Z_100100000Hz_IQ.wav`` (SDR#)
    * ``HDSDR_20230101_123456Z_100100kHz_RF.wav`` (HDSDR)
    * ``g001_433.92M_250k.cu8`` (rtl_433)
    * ``capture_1090MHz_2Msps.bin``

    Anything not found is None.
    """
    stem, ext = split_name(name)
    if stem.lower().endswith(tuple(EXTENSION_FORMATS)):  # take.cf32.sigmf-meta
        stem, inner = os.path.splitext(stem)
        ext = ext if ext.lower() not in (".sigmf-meta",) else inner
    hints = FileNameHints(sample_format=EXTENSION_FORMATS.get(ext.lower()))
    bare: List[str] = []
    for token in re.split(r"[_\s,;]+", stem):
        text = token.strip().lower()
        if not text:
            continue
        match = re.fullmatch(r"(\d+(?:[.p]\d+)?)([kmg]?)hz", text)
        if match:
            if hints.center_hz is None:
                hints.center_hz = _number(match.group(1)) * _UNIT[match.group(2)]
            continue
        match = re.fullmatch(
            r"(\d+(?:[.p]\d+)?)([km]?)(?:sps|s/s|sa/s|samples?|sr)", text
        )
        if match:
            if hints.sample_rate is None:
                hints.sample_rate = _number(match.group(1)) * _UNIT[match.group(2)]
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", text)  # rtl_433: 433.92M
        if match:
            value = _number(match.group(1)) * _UNIT[match.group(2)]
            if match.group(2) in "mg" and value >= 24e6 and hints.center_hz is None:
                hints.center_hz = value
            elif match.group(2) in "km" and value <= 25e6 and not hints.sample_rate:
                hints.sample_rate = value
            continue
        if re.fullmatch(r"\d{6,10}", text):
            bare.append(text)
    for text in bare:
        value = int(text)
        if (
            hints.center_hz is None
            and 24e6 <= value <= _MAX_CENTER_HZ
            and value % 1000 == 0  # (not a time stamp)
        ):
            hints.center_hz = float(value)
        elif hints.sample_rate is None and value in _COMMON_RATES:
            hints.sample_rate = float(value)
    for attr in ("center_hz", "sample_rate"):
        value = getattr(hints, attr)
        if value is not None and not (math.isfinite(value) and value > 0):
            setattr(hints, attr, None)
    return hints


def parse_rate_text(text: str) -> Optional[float]:
    """A sample rate typed as ``2.4``, ``2.4M``, ``2.4 MS/s``, ``2048000``,
    ``250k`` or ``250 kS/s``, in samples per second (None if it can't be
    read, or isn't between 1 kS/s and 200 MS/s). A bare number up to 100
    is in MS/s, up to 1000 in kS/s, above that in S/s."""
    compact = re.sub(r"\s+", "", str(text or "")).lower().replace(",", ".")
    match = re.fullmatch(
        r"(\d+(?:\.\d*)?|\.\d+)([kmg]?)(?:s/s|sps|sa/s|samples/s|hz)?", compact
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit:
        rate = value * _UNIT[unit]
    elif value <= 100:
        rate = value * 1e6
    elif value < 1000:
        rate = value * 1e3
    else:
        rate = value
    return rate if 1e3 <= rate <= 200e6 else None


def format_rate(rate_hz: float) -> str:
    """``2.4e6 -> "2.4 MS/s"``, ``250e3 -> "250 kS/s"``."""
    rate_hz = float(rate_hz)
    if rate_hz >= 1e6:
        return f"{rate_hz / 1e6:.6f}".rstrip("0").rstrip(".") + " MS/s"
    return f"{rate_hz / 1e3:.3f}".rstrip("0").rstrip(".") + " kS/s"


def format_mhz(freq_hz: float) -> str:
    """``100.1e6 -> "100.100 MHz"`` (3 to 6 decimals)."""
    text = f"{max(0.0, float(freq_hz)) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def format_duration(seconds: float) -> str:
    """``75.4 -> "1:15.4"``, ``3725 -> "1:02:05"``."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours, rest = divmod(int(round(seconds)), 3600)
        return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"
    minutes, secs = divmod(seconds, 60.0)
    return f"{int(minutes)}:{secs:04.1f}"


def format_datetime(text: str) -> str:
    """SigMF's ``2026-10-30T07:28:52.414781Z`` as ``2026-10-30 07:28:52
    UTC`` (the text itself if it can't be read)."""
    from datetime import datetime

    raw = str(text or "").strip()
    try:
        when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    zone = " UTC" if raw.endswith("Z") or when.utcoffset() is not None else ""
    return when.strftime("%Y-%m-%d %H:%M:%S") + zone


def format_size(size: float) -> str:
    size = float(size)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size:.0f} bytes"


@dataclass
class IqFileOptions:
    """How to read a recording: its sample format, rate and centre
    frequency (0 where unknown), from its metadata or the user."""

    path: str
    sample_format: SampleFormat = SampleFormat.FLOAT32
    sample_rate: float = 0.0
    center_hz: float = 0.0
    file_format: FileFormat = FileFormat.RAW
    total_samples: int = 0
    #: Where the rate and frequency came from, for Recording Info.
    source: str = ""
    #: Checked (or entered) by the user in the Import Options dialog.
    confirmed: bool = False

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def duration_s(self) -> float:
        return self.total_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    @property
    def headerless(self) -> bool:
        """A raw file whose format nothing but its name suggests."""
        return self.file_format is FileFormat.RAW and not self.source

    def open_player(self) -> IQPlayer:
        """An :class:`IQPlayer` reading the file with these settings."""
        sample_format = (
            self.sample_format if self.file_format is FileFormat.RAW else None
        )
        return IQPlayer(
            self.path,
            sample_format,
            self.sample_rate or None,
            self.center_hz if self.center_hz > 0 else None,
        )

    def count_samples(self, sample_format: SampleFormat) -> int:
        """Samples the file holds read as ``sample_format`` (raw files)."""
        if self.file_format is not FileFormat.RAW:
            return self.total_samples
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return 0
        return size // bytes_per_complex_sample(sample_format)


def describe_iq_file(path: str) -> Tuple[IqFileOptions, Set[str]]:
    """What a recording's own metadata says, and what it doesn't: a subset
    of ``{"format", "rate", "center"}`` (``"format"`` for a headerless
    file, whose format only its extension suggests).

    Raises ValueError when the file can't be read.
    """
    info = detect_format(path)
    if not info.is_valid:
        raise ValueError(info.error_message or "the file can't be read")
    meta = info.metadata or {}
    data_file = str(meta.get("data_file", "") or "")
    if info.file_format is FileFormat.SIGMF and (
        not data_file or data_file.lower().endswith(".sigmf-meta")
    ):
        stem = split_name(path)[0]
        raise ValueError(f"its samples are missing: {stem}.sigmf-data isn't next to it")
    if info.file_format is FileFormat.WAV and meta.get("wav_channels") == 1:
        raise ValueError(
            "it has one channel: it is audio, not I/Q (an I/Q WAV file has two, "
            "I and Q)"
        )
    width = meta.get("wav_sample_width")
    if info.file_format is FileFormat.WAV and width not in (None, 1, 2):
        # (The reader takes them for 16-bit: they would play as noise.)
        raise ValueError(
            f"its samples are {8 * int(width)}-bit, and this app reads 8- and "
            "16-bit I/Q WAV files"
        )
    datatype = str(meta.get("sigmf_datatype", "") or "")
    if datatype and sigmf_format(datatype) is None:
        raise ValueError(
            f"its sample type {datatype} isn't one this app reads (complex "
            "cu8, ci8, ci16_le, cf32_le or cf64_le)"
        )
    source = ""
    if info.file_format is FileFormat.SIGMF:
        source = "SigMF metadata"
    elif meta.get("detected_from") == "sidecar":
        source = (
            f"its metadata file ({os.path.basename(meta.get('metadata_file', ''))})"
        )
    elif info.file_format is FileFormat.WAV:
        source = "the WAV header"
    options = IqFileOptions(
        path=str(path),
        sample_format=info.sample_format,
        sample_rate=float(info.sample_rate or 0.0),
        center_hz=float(info.center_frequency or 0.0),
        file_format=info.file_format,
        total_samples=int(info.num_samples),
        source=source,
    )
    missing: Set[str] = set()
    if options.headerless:
        missing.add("format")
        guess = _content_guess(path)
        ext = split_name(path)[1].lower()
        if ext not in EXTENSION_FORMATS and guess is not None:
            options.sample_format = guess
        elif ext in EXTENSION_FORMATS:
            options.sample_format = EXTENSION_FORMATS[ext]
        options.total_samples = options.count_samples(options.sample_format)
    if not options.sample_rate:
        missing.add("rate")
    if not options.center_hz:
        missing.add("center")
    return options, missing


def _content_guess(path: str) -> Optional[SampleFormat]:
    try:
        with open(path, "rb") as handle:
            data = handle.read(65536)
    except OSError:
        return None
    return guess_sample_format(data) if len(data) >= 64 else None


#: What each use of a file needs to know (the others may stay unknown).
REQUIRED: Dict[str, Set[str]] = {
    "play": {"rate", "center"},
    "decode": {"rate"},
    "convert": set(),
    "info": set(),
}
_ACTION_TEXT = {
    "play": "&Play",
    "decode": "&Decode",
    "convert": "&Convert",
    "info": "&Show Info",
}


def needs_options(missing: Set[str], purpose: str) -> bool:
    """Whether to ask (Import Options) before ``purpose`` ("play",
    "convert", "decode", "info"): always for a headerless file; when a
    value the purpose needs is unknown; for a conversion, when the rate or
    frequency is unknown (so the new file isn't labelled wrongly)."""
    if "format" in missing:
        return True
    if missing & REQUIRED.get(purpose, set()):
        return True
    return purpose == "convert" and bool(missing)


if HAS_PYQT6:

    class ImportOptionsDialog(QDialog):
        """Asks how a recording's samples are stored (format), how fast
        they were taken (rate) and where it was tuned (centre frequency),
        for a file that doesn't say.

        Prefilled from the file's extension and content
        (:func:`describe_iq_file`), its name (:func:`guess_from_filename`)
        and ``guesses`` (e.g. the receiver's rate, marked as a guess). The
        samples are checked as the format changes. ``purpose`` ("play",
        "convert", "decode", "info") names the OK button and decides which
        values are required (:data:`REQUIRED`).
        """

        def __init__(
            self,
            options: IqFileOptions,
            missing: Set[str],
            purpose: str = "play",
            guesses: Optional[Dict[str, float]] = None,
            parent: Any = None,
        ):
            super().__init__(parent)
            self._options = options
            self._missing = set(missing)
            self._purpose = purpose
            self._required = REQUIRED.get(purpose, set())
            self._content = (
                _content_guess(options.path) if "format" in missing else None
            )
            self._result: Optional[IqFileOptions] = None
            self.setWindowTitle(f"Import Options – {options.name}")
            self.setMinimumWidth(500)
            self._build(guesses or {})
            self._update_check()

        # -- layout ---------------------------------------------------------

        def _build(self, guesses: Dict[str, float]) -> None:
            options = self._options
            hints = guess_from_filename(options.path)
            layout = QVBoxLayout(self)
            layout.setSpacing(10)

            if "format" in self._missing:
                intro = (
                    f"{options.name} has no header saying how its samples are "
                    "stored. Check the sample format, rate and center frequency "
                    "below."
                )
            else:
                intro = (
                    f"{options.name} doesn't say everything needed: fill in "
                    "what is missing."
                )
            intro_label = QLabel(intro)
            intro_label.setWordWrap(True)
            layout.addWidget(intro_label)

            form = QFormLayout()
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)

            self._format_combo = QComboBox()
            for fmt, text in FORMAT_CHOICES:
                self._format_combo.addItem(text, fmt)
            index = self._format_combo.findData(options.sample_format)
            self._format_combo.setCurrentIndex(max(0, index))
            editable_format = "format" in self._missing
            self._format_combo.setEnabled(editable_format)
            self._format_combo.setToolTip(
                "How each I and Q value is stored. rtl_sdr writes cu8, "
                "hackrf_transfer cs8; GNU Radio and GQRX write cf32."
                if editable_format
                else f"Stored in {options.source}."
            )
            self._format_combo.currentIndexChanged.connect(self._update_check)
            self._format_hint = QLabel()
            set_role(self._format_hint, "hint")
            self._use_guess = QPushButton()
            self._use_guess.setAutoDefault(False)
            self._use_guess.clicked.connect(self._apply_content_guess)
            form.addRow("Sample &format:", self._format_combo)
            hint_row = QHBoxLayout()
            hint_row.setContentsMargins(0, 0, 0, 0)
            hint_row.addWidget(self._format_hint, 1)
            hint_row.addWidget(self._use_guess)
            self._format_hint_row = QWidget()
            self._format_hint_row.setLayout(hint_row)
            form.addRow("", self._format_hint_row)

            self._rate_edit = QLineEdit()
            self._rate_edit.setAccessibleName("Sample rate")
            self._rate_edit.setToolTip(
                "Samples per second: 2.4 or 2.4M (MS/s), 2048000, 250k."
            )
            self._rate_hint = QLabel()
            set_role(self._rate_hint, "hint")
            rate, rate_note = self._prefill(
                "rate", options.sample_rate, hints.sample_rate, guesses.get("rate")
            )
            if rate:
                self._rate_edit.setText(format_rate(rate))
            self._rate_note = rate_note
            self._rate_prefill = self._rate_edit.text()
            self._rate_edit.setPlaceholderText(
                "Required" if "rate" in self._required else "Unknown"
            )
            self._rate_edit.setEnabled("rate" in self._missing)
            self._rate_edit.textChanged.connect(self._update_check)
            form.addRow("Sample &rate:", self._row(self._rate_edit, self._rate_hint))

            self._center_edit = QLineEdit()
            self._center_edit.setAccessibleName("Center frequency")
            self._center_edit.setToolTip(
                "Where the receiver was tuned (the middle of the recording), "
                "in MHz unless you type a unit: 433.92, 7074k, 1.09G."
            )
            self._center_hint = QLabel()
            set_role(self._center_hint, "hint")
            center, center_note = self._prefill(
                "center", options.center_hz, hints.center_hz, guesses.get("center")
            )
            if center:
                self._center_edit.setText(format_mhz(center))
            self._center_note = center_note
            self._center_prefill = self._center_edit.text()
            self._center_edit.setPlaceholderText(
                "Required" if "center" in self._required else "Unknown"
            )
            self._center_edit.setEnabled("center" in self._missing)
            self._center_edit.textChanged.connect(self._update_check)
            form.addRow(
                "&Center frequency:", self._row(self._center_edit, self._center_hint)
            )
            layout.addLayout(form)

            self._check_label = QLabel()
            self._check_label.setWordWrap(True)
            self._check_label.setAccessibleName("Sample check")
            set_role(self._check_label, "callout", "info")
            layout.addWidget(self._check_label)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            self._ok_button = QPushButton(_ACTION_TEXT.get(self._purpose, "&OK"))
            set_role(self._ok_button, "primary")
            self._ok_button.setDefault(True)
            buttons.addButton(self._ok_button, QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        @staticmethod
        def _row(edit: "QLineEdit", hint: "QLabel") -> "QWidget":
            row = QWidget()
            box = QHBoxLayout(row)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(8)
            box.addWidget(edit, 1)
            box.addWidget(hint)
            return row

        def _prefill(
            self,
            key: str,
            stored: float,
            from_name: Optional[float],
            guess: Optional[float],
        ) -> Tuple[float, str]:
            """The value a field starts with, and where it came from."""
            if stored and self._options.confirmed:
                return stored, "as you entered it"
            if stored:
                return stored, f"from {self._options.source or 'the file'}"
            if from_name:
                return from_name, "from the file name"
            if guess and key in self._required:
                what = "rate" if key == "rate" else "center frequency"
                return guess, f"a guess: the receiver's {what}"
            return 0.0, "not stored in the file"

        # -- values -----------------------------------------------------------

        def sample_format(self) -> SampleFormat:
            fmt = self._format_combo.currentData()
            return fmt if isinstance(fmt, SampleFormat) else SampleFormat.FLOAT32

        def sample_rate(self) -> Optional[float]:
            """The rate typed (0.0: left blank; None: can't be read)."""
            text = self._rate_edit.text().strip()
            if not text:
                return 0.0
            return parse_rate_text(text)

        def center_hz(self) -> Optional[float]:
            """The frequency typed (0.0: left blank; None: can't be read)."""
            text = self._center_edit.text().strip()
            if not text:
                return 0.0
            parsed = parse_frequency_text(text, 1e6, *PLAYBACK_RANGE)
            return parsed.hz if parsed.state == ACCEPTABLE else None

        def options(self) -> Optional[IqFileOptions]:
            """What was chosen (after OK), else None."""
            return self._result

        def _current(self) -> Tuple[Optional[IqFileOptions], str]:
            """The options as filled in, or None with the reason."""
            rate, center = self.sample_rate(), self.center_hz()
            if rate is None:
                return None, (
                    "Can't read the sample rate: type it in MS/s (2.4), with a "
                    "unit (250k, 2.048M) or in samples per second (2048000)."
                )
            if center is None:
                return None, (
                    "Can't read the center frequency: type it in MHz (433.92) "
                    "or with a unit (7074k, 1.09G); the receiver tunes "
                    "0.001–6000 MHz."
                )
            for key, value in (("rate", rate), ("center", center)):
                if key in self._required and not value:
                    what = "sample rate" if key == "rate" else "center frequency"
                    verb = _ACTION_TEXT.get(self._purpose, "&OK").replace("&", "")
                    return None, f"Enter the {what} to {verb.lower()} the file."
            fmt = self.sample_format()
            options = replace(
                self._options,
                sample_format=fmt,
                sample_rate=float(rate),
                center_hz=float(center),
                total_samples=self._options.count_samples(fmt),
                confirmed=True,
            )
            return options, ""

        # -- checks -----------------------------------------------------------

        def _apply_content_guess(self) -> None:
            if self._content is not None:
                self._format_combo.setCurrentIndex(
                    max(0, self._format_combo.findData(self._content))
                )

        def _update_check(self, *_args: Any) -> None:
            fmt = self.sample_format()
            guess = self._content
            hint = guess is not None and guess != fmt and self._format_combo.isEnabled()
            if hint and guess is not None:
                self._format_hint.setText(
                    f"The data looks more like {FORMAT_NAMES[guess].split(' – ')[0]}."
                )
                self._use_guess.setText(f"Use {guess.value}")
            if hint != self._format_hint_row.isVisibleTo(self):
                self._format_hint_row.setVisible(hint)
                # Fit the row in (or close the gap) instead of squeezing
                # the fields.
                self.adjustSize()
            self._rate_hint.setText(
                self._note(self._rate_edit, self._rate_prefill, self._rate_note)
            )
            self._center_hint.setText(
                self._note(self._center_edit, self._center_prefill, self._center_note)
            )
            options, problem = self._current()
            check = self._sample_check(fmt)
            if not check.ok:
                self._show_check(check.problem, "danger")
                self._ok_button.setEnabled(check.total > 0 or check.samples > 0)
                return
            self._ok_button.setEnabled(True)
            if problem:
                self._show_check(problem, "warning")
                return
            assert options is not None
            count = options.total_samples
            text = f"{count:,} samples"
            if options.sample_rate:
                text += f", {format_duration(options.duration_s)} long"
            if options.center_hz and options.sample_rate:
                half = options.sample_rate / 2.0
                text += (
                    f", {format_mhz(options.center_hz - half).split(' ')[0]}–"
                    f"{format_mhz(options.center_hz + half)}"
                )
            text += (
                f". The samples look right (peak {check.max_abs:.2f} of full scale)."
            )
            self._show_check(text, "success")

        @staticmethod
        def _note(edit: "QLineEdit", prefill: str, note: str) -> str:
            """Where the field's value came from (nothing once typed over)."""
            text = edit.text().strip()
            if not edit.isEnabled() or (text and text == prefill.strip()):
                return note
            return "" if text else "not stored in the file"

        def _sample_check(self, fmt: SampleFormat) -> SampleCheck:
            cache = getattr(self, "_checks", None)
            if cache is None:
                cache = self._checks = {}
            if fmt not in cache:
                read_as = fmt if self._options.file_format is FileFormat.RAW else None
                cache[fmt] = check_iq_samples(self._options.path, read_as)
            return cache[fmt]

        def _show_check(self, text: str, tone: str) -> None:
            self._check_label.setText(text)
            set_role(self._check_label, "callout", tone)

        def check_text(self) -> str:
            return self._check_label.text()

        def _on_accept(self) -> None:
            options, problem = self._current()
            if options is None:
                self._show_check(problem, "danger")
                return
            self._result = options
            self.accept()

    class RecordingInfoDialog(QDialog):
        """File > Recording Info: what a recording holds (from
        :func:`~sdr_module.dsp.recording.get_file_info`), whether its
        samples look right, and what it can be turned into. The buttons
        Play, Convert and Decode close it with :meth:`action` set."""

        ACTIONS = ("play", "convert", "decode")

        def __init__(self, options: IqFileOptions, parent: Any = None):
            super().__init__(parent)
            self._options = options
            self._action = ""
            self.setWindowTitle(f"Recording Info – {options.name}")
            self.setMinimumWidth(480)
            layout = QVBoxLayout(self)
            layout.setSpacing(10)
            self._form = QFormLayout()
            self._form.setHorizontalSpacing(14)
            self._form.setVerticalSpacing(5)
            self._values: Dict[str, QLabel] = {}
            for key, text in self.rows():
                label = QLabel(key)
                set_role(label, "muted")
                value = QLabel(text)
                value.setWordWrap(True)
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                self._form.addRow(label, value)
                self._values[key] = value
            layout.addLayout(self._form)
            check = check_iq_samples(
                options.path,
                (
                    options.sample_format
                    if options.file_format is FileFormat.RAW
                    else None
                ),
            )
            self._check = QLabel(
                f"The samples look right (peak {check.max_abs:.2f} of full scale)."
                if check.ok
                else check.problem
            )
            self._check.setWordWrap(True)
            set_role(self._check, "callout", "success" if check.ok else "warning")
            layout.addWidget(self._check)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            for action, text, tip in (
                ("play", "&Play", "Play the recording through the receiver"),
                (
                    "convert",
                    "C&onvert...",
                    "Load it to save it in another format",
                ),
                (
                    "decode",
                    "&Decode...",
                    "Run the Decoder panel's protocol over the whole file",
                ),
            ):
                button = QPushButton(text)
                button.setToolTip(tip)
                button.setAutoDefault(False)
                button.clicked.connect(lambda _c=False, a=action: self._choose(a))
                buttons.addButton(button, QDialogButtonBox.ButtonRole.ActionRole)
                if action == "play" and (
                    not options.sample_rate or not options.center_hz
                ):
                    button.setToolTip(
                        "Play the recording (you'll be asked for the rate and "
                        "center frequency it doesn't say)"
                    )
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def _choose(self, action: str) -> None:
            self._action = action
            self.accept()

        def action(self) -> str:
            """``"play"``, ``"convert"``, ``"decode"`` or ``""`` (closed)."""
            return self._action

        def value(self, key: str) -> str:
            label = self._values.get(key)
            return label.text() if label is not None else ""

        def rows(self) -> List[Tuple[str, str]]:
            """(label, text) of each fact shown."""
            options = self._options
            info = get_file_info(
                options.path,
                (
                    options.sample_format
                    if options.file_format is FileFormat.RAW
                    else None
                ),
                options.sample_rate or None,
                options.center_hz if options.confirmed else None,
            )
            unknown = "Unknown (not stored in the file)"
            kinds = {
                "sigmf": "SigMF recording",
                "wav": "WAV file, I/Q as stereo",
                "raw": "Raw I/Q samples (no header)",
            }
            kind = kinds.get(info["file_format"], info["file_format"])
            fmt = options.sample_format
            rate = float(info["sample_rate_hz"] or 0.0)
            center = float(info["center_frequency_hz"] or 0.0)
            rows = [
                ("File", options.name),
                (
                    "Folder",
                    display_path(os.path.dirname(os.path.abspath(options.path))),
                ),
                ("Type", kind),
                ("Sample format", FORMAT_NAMES.get(fmt, fmt.value)),
                ("Sample rate", format_rate(rate) if rate else unknown),
                ("Center frequency", format_mhz(center) if center else unknown),
            ]
            if rate and center:
                rows.append(
                    (
                        "Covers",
                        f"{format_mhz(center - rate / 2).split(' ')[0]}–"
                        f"{format_mhz(center + rate / 2)}",
                    )
                )
            rows += [
                (
                    "Duration",
                    (
                        format_duration(info["duration_seconds"])
                        if rate
                        else "Unknown (needs the sample rate)"
                    ),
                ),
                ("Samples", f"{int(info['num_samples']):,}"),
                ("Size", format_size(info["file_size_bytes"])),
            ]
            meta = info.get("metadata") or {}
            for key, label in (
                ("sigmf_datetime", "Recorded"),
                ("sigmf_hardware", "Hardware"),
                ("sigmf_description", "Description"),
            ):
                if meta.get(key):
                    text = str(meta[key])
                    if key == "sigmf_datetime":
                        text = format_datetime(text)
                    rows.append((label, text))
            if options.confirmed:
                source = "Entered by you (the file doesn't say)"
            elif options.source:
                source = options.source[0].upper() + options.source[1:]
            else:
                source = "None: only the file name's extension"
            rows.append(("Metadata", source))
            rows.append(
                (
                    "Converts to",
                    "SigMF, cf32, cs16, cu8, cs8, WAV and raw float32 "
                    "(File > Convert Recording)",
                )
            )
            return rows

    class DecodeOptionsDialog(QDialog):
        """File > Decode Recording: which protocol to decode, and where in
        the recording its channel is (a frequency inside the recorded span,
        or an offset from the center when the file doesn't say where it was
        tuned)."""

        def __init__(
            self,
            options: IqFileOptions,
            protocols: List[str],
            protocol: str = "",
            frequency_hz: float = 0.0,
            parent: Any = None,
        ):
            super().__init__(parent)
            self._options = options
            self._result: Optional[Tuple[str, float]] = None
            self.setWindowTitle(f"Decode Recording – {options.name}")
            self.setMinimumWidth(440)
            layout = QVBoxLayout(self)
            layout.setSpacing(10)
            intro = QLabel(
                f"Decode {options.name} ({format_duration(options.duration_s)}) "
                "much faster than real time. The messages go to the Decoder "
                "tab."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)
            form = QFormLayout()
            form.setHorizontalSpacing(12)
            self._protocol = QComboBox()
            self._protocol.addItems(list(protocols))
            if protocol in protocols:
                self._protocol.setCurrentText(protocol)
            self._protocol.setAccessibleName("Protocol")
            form.addRow("&Protocol:", self._protocol)
            self._freq = QLineEdit()
            self._has_center = options.center_hz > 0
            if self._has_center:
                self._freq.setAccessibleName("Frequency")
                self._freq.setToolTip(
                    "The frequency of the transmissions to decode, inside the "
                    "recording (MHz unless you type a unit)."
                )
                start = frequency_hz or options.center_hz
                self._freq.setText(format_mhz(start))
                form.addRow("&Frequency:", self._freq)
                half = options.sample_rate / 2.0
                span = (
                    f"The recording covers "
                    f"{format_mhz(options.center_hz - half).split(' ')[0]}–"
                    f"{format_mhz(options.center_hz + half)}."
                )
            else:
                self._freq.setAccessibleName("Offset from the center")
                self._freq.setToolTip(
                    "How far the transmissions are from the middle of the "
                    "recording, in kHz (0: in the middle)."
                )
                self._freq.setText("0")
                form.addRow("&Offset (kHz):", self._freq)
                span = (
                    "The file doesn't say where it was tuned: give the offset "
                    f"from its middle (within ±{options.sample_rate / 2e3:g} kHz)."
                )
            layout.addLayout(form)
            self._span_text = span
            self._note = QLabel(span)
            self._note.setWordWrap(True)
            set_role(self._note, "hint")
            layout.addWidget(self._note)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            ok = QPushButton("&Decode")
            set_role(ok, "primary")
            ok.setDefault(True)
            buttons.addButton(ok, QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def offset(self) -> Optional[float]:
            """The channel's offset from the recording's center (Hz), or None
            when what was typed can't be read or is outside the recording."""
            text = self._freq.text().strip()
            half = 0.5 * self._options.sample_rate
            if self._has_center:
                parsed = parse_frequency_text(text, 1e6, *PLAYBACK_RANGE)
                if parsed.state != ACCEPTABLE or parsed.hz is None:
                    return None
                offset = parsed.hz - self._options.center_hz
            else:
                try:
                    offset = float(text.replace(",", ".")) * 1e3
                except ValueError:
                    return None
            return offset if abs(offset) < half else None

        def _on_accept(self) -> None:
            offset = self.offset()
            if offset is None:
                what = "frequency" if self._has_center else "offset"
                self._note.setText(
                    f"That {what} isn't inside the recording. {self._span_text}"
                )
                set_tone(self._note, "danger")
                return
            self._result = (self._protocol.currentText(), offset)
            self.accept()

        def choice(self) -> Optional[Tuple[str, float]]:
            """``(protocol, offset Hz)`` after Decode, else None."""
            return self._result

        def set_frequency_text(self, text: str) -> None:
            self._freq.setText(text)

    class RecordingDetailsDialog(QDialog):
        """Save Recording, when the recording doesn't say its sample rate or
        center frequency (a converted headerless file): enter them, or leave
        them blank to save them as unknown. Nothing is filled in from the
        receiver unasked: the receiver's values are offered as buttons."""

        def __init__(
            self,
            rate: float,
            center: float,
            receiver_rate: float,
            receiver_center: float,
            parent: Any = None,
        ):
            super().__init__(parent)
            self.setWindowTitle("Recording Details")
            self.setMinimumWidth(420)
            self._result: Optional[Tuple[float, float]] = None
            layout = QVBoxLayout(self)
            layout.setSpacing(10)
            missing = [
                what
                for what, value in (("sample rate", rate), ("center frequency", center))
                if not value
            ]
            intro = QLabel(
                f"This recording doesn't say its {' or '.join(missing)}. Enter "
                "what you know so the saved file is labelled right, or leave a "
                "field blank to save it as unknown."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)
            form = QFormLayout()
            self._rate = QLineEdit(format_rate(rate) if rate else "")
            self._rate.setPlaceholderText("Unknown")
            self._rate.setAccessibleName("Sample rate")
            self._rate.setEnabled(not rate)
            self._center = QLineEdit(format_mhz(center) if center else "")
            self._center.setPlaceholderText("Unknown")
            self._center.setAccessibleName("Center frequency")
            self._center.setEnabled(not center)
            form.addRow("Sample &rate:", self._rate)
            form.addRow("&Center frequency:", self._center)
            layout.addLayout(form)
            # The receiver's values for what is missing, as far as it knows
            # them (no rate without a device).
            offer_rate = receiver_rate if receiver_rate > 0 and not rate else 0.0
            offer_center = (
                receiver_center if receiver_center > 0 and not center else 0.0
            )
            offered = [format_rate(offer_rate)] if offer_rate else []
            offered += [format_mhz(offer_center)] if offer_center else []
            if offered:
                use = QPushButton(f"Use the receiver's ({', '.join(offered)})")
                use.setAutoDefault(False)
                use.setToolTip(
                    "Only if this recording was made at what the receiver is "
                    "set to now (its center frequency and sample rate)"
                )

                def fill() -> None:
                    if offer_rate:
                        self._rate.setText(format_rate(offer_rate))
                    if offer_center:
                        self._center.setText(format_mhz(offer_center))

                use.clicked.connect(lambda _c=False: fill())
                layout.addWidget(use)
            self._problem = QLabel("")
            self._problem.setWordWrap(True)
            set_role(self._problem, "hint", "danger")
            layout.addWidget(self._problem)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            ok = QPushButton("&Continue")
            set_role(ok, "primary")
            ok.setDefault(True)
            buttons.addButton(ok, QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def _on_accept(self) -> None:
            rate_text = self._rate.text().strip()
            rate = parse_rate_text(rate_text) if rate_text else 0.0
            center_text = self._center.text().strip()
            center: Optional[float] = 0.0
            if center_text:
                parsed = parse_frequency_text(center_text, 1e6, *PLAYBACK_RANGE)
                center = parsed.hz if parsed.state == ACCEPTABLE else None
            if rate is None or center is None:
                self._problem.setText(
                    "Can't read the "
                    + ("sample rate" if rate is None else "center frequency")
                    + ": e.g. 2.4 (MS/s) and 433.92 (MHz)."
                )
                return
            self._result = (float(rate), float(center))
            self.accept()

        def values(self) -> Optional[Tuple[float, float]]:
            return self._result

        def fill(self, rate_text: str, center_text: str) -> None:
            if self._rate.isEnabled():
                self._rate.setText(rate_text)
            if self._center.isEnabled():
                self._center.setText(center_text)

    def ask_recording_details(
        parent: Any,
        rate: float,
        center: float,
        receiver_rate: float,
        receiver_center: float,
        exec_dialog: Optional[Any] = None,
    ) -> Optional[Tuple[float, float]]:
        """Ask for a recording's unknown rate or center (see
        :class:`RecordingDetailsDialog`); ``(rate, center)`` with 0 for
        unknown, or None when cancelled."""
        dialog = RecordingDetailsDialog(
            rate, center, receiver_rate, receiver_center, parent
        )
        accepted = exec_dialog(dialog) if exec_dialog else dialog.exec()
        values = dialog.values() if accepted else None
        if values is None:
            return None
        return (rate or values[0], center or values[1])
