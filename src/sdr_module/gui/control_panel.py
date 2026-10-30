"""
Device control panel widget.

Provides controls for:
- Frequency tuning (entry with unit selector and quick-tune steps)
- Receiver settings: RF gain / AGC and bandwidth
- Demodulation mode, FM deviation and squelch
- Frequency presets (with TX lockout indicators)
- I/Q recording
- License profile (transmit privileges)

The sections live in an internal scroll area, so the panel degrades
gracefully (scrolls instead of crushing its group boxes) when it is short.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, QSize, Qt, pyqtSignal
    from PyQt6.QtGui import QValidator
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QStyle,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core import frequency_manager as _frequency_manager_module
from ..core.frequency_manager import (
    AMATEUR_BAND_PRIVILEGES,
    POWER_HEADROOM_FACTOR,
    TX_POWER_WARNING,
    LicenseClass,
    get_frequency_manager,
)
from ..dsp.filters import CTCSS_FREQUENCIES, ctcss_label
from ..utils.tooltips import get_short_tip
from .audio_chain import (
    AGC_SPEED_LABELS,
    CW_PITCH_RANGE,
    DEFAULT_AGC_SPEED,
    DEFAULT_CW_PITCH_HZ,
    NOISE_REDUCTION_LEVELS,
)
from .device_settings import (
    ADSB_MIN_RATE,
    ADSB_RATE_TIP,
    AGC_TIP,
    AMP_TEXT,
    AMP_TIP,
    AMP_WARNING,
    BIAS_TEE_TEXT,
    BIAS_TEE_TIPS,
    BIAS_TEE_WARNING,
    DEFAULT_GAIN_MODEL,
    DIRECT_SAMPLING_CHOICES,
    DIRECT_SAMPLING_TIP,
    HACKRF_AMP_DB,
    HACKRF_DEFAULT_LNA,
    HACKRF_DEFAULT_VGA,
    HACKRF_LNA_STEPS,
    HACKRF_VGA_STEPS,
    PPM_HINT,
    PPM_RANGE,
    PPM_TIP,
    FrontEnd,
    GainModel,
    format_db,
    front_end_features,
    rate_choice_label,
    split_hackrf_gain,
)
from .settings_store import DEFAULT_FREQUENCY_HZ
from .themes import set_role, set_tone
from .tuning import (
    ACCEPTABLE,
    INTERMEDIATE,
    ParsedFrequency,
    format_mhz_range,
    parse_frequency_text,
    step_frequency,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tuning range of the frequency entry: the union of every supported
# receiver (RTL-SDR from 500 kHz, HackRF One up to 6 GHz). The Bookmarks
# frequency field uses the same range, so a bookmark always stores exactly
# the frequency the tuner showed.
MIN_FREQUENCY_HZ = 1e3
MAX_FREQUENCY_HZ = 6e9

# One arrow-key / wheel step of the frequency entry (matches the plots'
# Left/Right tuning step). Ctrl+wheel and Page Up/Down step ten times this.
FREQ_STEP_HZ = 10e3

# Display unit -> (multiplier, decimals). Every unit resolves to 1 Hz.
_UNITS: Dict[str, Tuple[float, int]] = {
    "Hz": (1.0, 0),
    "kHz": (1e3, 3),
    "MHz": (1e6, 6),
    "GHz": (1e9, 9),
}

# Units whose integer part gets digit grouping ("145,800,000 Hz" instead of
# an unreadable "145800000").
_GROUPED_UNITS = ("Hz", "kHz")

QUICK_TUNE_STEPS = (-1e6, -100e3, -10e3, 10e3, 100e3, 1e6)
_MINUS = "−"  # typographic minus for the step buttons' labels

# The channel widths of "None (I/Q)" mode (the wide list).
BANDWIDTH_OPTIONS = (
    "10 kHz",
    "25 kHz",
    "50 kHz",
    "100 kHz",
    "200 kHz",
    "500 kHz",
    "1 MHz",
    "2 MHz",
    "2.4 MHz",
)
DEFAULT_BANDWIDTH = "200 kHz"

# The Bandwidth choices of each mode: the channel for AM and FM, the filter
# for SSB and CW (what is heard, and what LEVEL, the squelch and the
# S-meter measure).
_SSB_WIDTHS = ("1.8 kHz", "2.4 kHz", "2.7 kHz", "3.0 kHz", "3.6 kHz")
MODE_BANDWIDTHS: Dict[str, Tuple[str, ...]] = {
    "None (I/Q)": BANDWIDTH_OPTIONS,
    "AM": ("5 kHz", "6 kHz", "8 kHz", "10 kHz", "12 kHz", "15 kHz"),
    "FM": ("12.5 kHz", "25 kHz", "50 kHz", "100 kHz", "200 kHz"),
    "USB": _SSB_WIDTHS,
    "LSB": _SSB_WIDTHS,
    "CW": ("100 Hz", "250 Hz", "500 Hz", "1 kHz"),
}

# FM is two different things: broadcast stations (wide, 75 kHz deviation)
# in the broadcast band, voice and data (narrow) elsewhere. Picking FM
# uses the settings of the one the tuned frequency is in.
FM_BROADCAST_PROFILE = "FM broadcast"
FM_BROADCAST_BAND = (87.5e6, 108e6)

# Bandwidth and FM deviation a mode starts with, until the user changes
# them (then they are remembered per mode): by filter profile.
MODE_FILTER_DEFAULTS: Dict[str, Tuple[str, str]] = {
    "None (I/Q)": (DEFAULT_BANDWIDTH, ""),
    "AM": ("10 kHz", ""),
    "FM": ("25 kHz", "5 kHz"),
    FM_BROADCAST_PROFILE: ("200 kHz", "75 kHz"),
    "USB": ("2.7 kHz", ""),
    "LSB": ("2.7 kHz", ""),
    "CW": ("500 Hz", ""),
}


def bandwidth_options(mode: str) -> Tuple[str, ...]:
    """The Bandwidth choices for ``mode`` (the wide list for I/Q)."""
    return MODE_BANDWIDTHS.get(mode, BANDWIDTH_OPTIONS)


def filter_profile(mode: str, freq_hz: float) -> str:
    """Which remembered filter settings ``mode`` uses at ``freq_hz``:
    the mode itself, or :data:`FM_BROADCAST_PROFILE` for FM in the
    broadcast band."""
    low, high = FM_BROADCAST_BAND
    if mode == "FM" and low <= float(freq_hz) <= high:
        return FM_BROADCAST_PROFILE
    return mode


def closest_bandwidth(options: Tuple[str, ...], bw_hz: float) -> str:
    """The narrowest of ``options`` that holds ``bw_hz``, else the widest."""
    values = [(_parse_hz(text), text) for text in options]
    if not values:
        return ""
    fitting = [v for v in values if v[0] >= float(bw_hz) * (1 - 1e-9)]
    return min(fitting)[1] if fitting else max(values)[1]


# Item text (read by the main window), item tooltip key.
DEMOD_MODES: Tuple[Tuple[str, str], ...] = (
    ("None (I/Q)", ""),
    ("AM", "am"),
    ("FM", "fm"),
    ("USB", "ssb"),
    ("LSB", "ssb"),
    ("CW", ""),
)

# (value, item text, item tooltip). The value ("75 kHz") is what settings,
# bookmarks and presets store; the item says what it is for.
FM_DEVIATIONS: Tuple[Tuple[str, str, str], ...] = (
    (
        "2.5 kHz",
        "2.5 kHz (narrow, 12.5 kHz channels)",
        "Narrow FM on 12.5 kHz channels: PMR446, business and public radio",
    ),
    (
        "5 kHz",
        "5 kHz (narrow voice)",
        "Narrowband FM voice on 25 kHz channels: amateur 2 m and 70 cm, NOAA "
        "weather, marine",
    ),
    ("12.5 kHz", "12.5 kHz (wide voice)", "Wide FM voice and data links"),
    (
        "25 kHz",
        "25 kHz (satellite APT)",
        "Wide FM: NOAA weather-satellite APT, analog TV sound",
    ),
    ("75 kHz", "75 kHz (broadcast)", "Broadcast FM radio (87.5-108 MHz)"),
)
DEFAULT_FM_DEVIATION = "25 kHz"

# Squelch threshold range (dBFS) of the slider and the toolbar meter.
SQUELCH_RANGE = (-120, 0)

# Named like File > Save Recording's filters, which preselect the choice.
# SigMF first: it keeps the frequency and sample rate in a form every SDR
# program reads.
RECORDING_FORMATS = (
    "SigMF, keeps frequency & rate",
    "Complex Float32 I/Q (.cf32)",
    "Complex Int16 I/Q (.cs16)",
    "WAV, 16-bit I/Q (.wav)",
    "Raw Float32 I/Q (.raw)",
)
#: The file extension of each of RECORDING_FORMATS (the save format).
RECORDING_FORMAT_EXTENSIONS = (".sigmf-data", ".cf32", ".cs16", ".wav", ".raw")
# What the Record buttons record: (key, Record list text, tooltip).
RECORD_KINDS: Tuple[Tuple[str, str, str], ...] = (
    (
        "iq",
        "I/Q samples",
        "The raw radio samples of the whole span: play, decode or convert "
        "them later (about 19 MB a second at 2.4 MS/s).",
    ),
    (
        "audio",
        "Audio (what you hear)",
        "The demodulated audio as a 48 kHz WAV file that any player plays, "
        "saved in the recordings folder as it records (Radio > Record Audio, "
        "Ctrl+Shift+A).",
    ),
)

# (license class, combo text, TX privileges summary)
LICENSE_CLASSES: Tuple[Tuple[LicenseClass, str, str], ...] = (
    (LicenseClass.NONE, "None (license-free)", "CB, MURS and FRS only"),
    (
        LicenseClass.TECHNICIAN,
        "Technician",
        "VHF/UHF, 10 m and limited HF CW",
    ),
    (LicenseClass.GENERAL, "General", "most HF bands plus VHF/UHF"),
    (LicenseClass.AMATEUR_EXTRA, "Amateur Extra", "full amateur privileges"),
)

# Preset mode -> Mode combo entry (presets use "WFM" and "RAW", which the
# Mode list calls FM with a 75 kHz deviation and "None (I/Q)").
_PRESET_DEMOD = {
    "FM": "FM",
    "WFM": "FM",
    "AM": "AM",
    "USB": "USB",
    "LSB": "LSB",
    "CW": "CW",
    "RAW": "None (I/Q)",
}
# How a preset's mode is shown in its details, in the Mode list's words.
_PRESET_MODE_LABEL = {"WFM": "FM (broadcast)", "RAW": "None (I/Q)"}

# Compact dummy-load reminder shown in the panel (full text in the tooltip).
TX_POWER_REMINDER = (
    "⚠ Before transmitting, check your output power with a 50 Ω dummy load "
    "and a wattmeter."
)

# Widest slider readout; both readouts share it so the sliders line up.
_VALUE_LABEL_WIDEST = "-120 dBFS"

_FORMAT_TIP = (
    "File format for the recorded I/Q samples; File > Save Recording offers "
    "it first. These are raw radio samples, not audio."
)

_APPLY_PRESET_TIP = "Tune to this preset and set its bandwidth and demodulation mode"

_STATUS_READY = "Ready"
_STATUS_RECORDING = "● Recording"
_STATUS_RECORDING_AUDIO = "● Recording audio"
_STATUS_PAUSED = "❚❚ Paused"
_STATUS_ARMED = "◌ Armed – waiting for the receiver"
_STATUS_WAITING = "◌ Waiting – resumes with the receiver"
_ARMED_TIP = "Samples are captured once the receiver starts (Space)"
_WAITING_TIP = "The receiver is stopped: recording resumes when it starts again (Space)"
_AUTO_SAVE_TIP = (
    "When an I/Q recording stops, save it in the recordings folder with an "
    "automatic name (iq_<frequency>_<rate>_<date>_<time>) in the format above, "
    "with no Save step. Audio recordings are always saved there."
)
_PAUSE_TEXT = "❚❚ Pause"
_RESUME_TEXT = "▶ Resume"
_RECORD_TEXT = "● Record"
_STOP_TEXT = "⏹ Stop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_hz(text: str) -> float:
    """Parse ``"2.4 MHz"`` / ``"25 kHz"`` / ``"500 Hz"`` into Hz."""
    value, _, unit = text.strip().partition(" ")
    scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(unit.strip(), 1.0)
    return float(value) * scale


def _elide_middle(text: str, limit: int) -> str:
    """``text`` shortened to ``limit`` characters with "…" in the middle."""
    if len(text) <= limit:
        return text
    keep = limit - 1
    return text[: (keep + 1) // 2] + "…" + text[len(text) - keep // 2 :]


def _format_mhz(freq_hz: float) -> str:
    """``146.52e6 -> "146.520 MHz"``, ``137.9125e6 -> "137.9125 MHz"``."""
    text = f"{float(freq_hz) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def _format_bandwidth(bw_hz: float) -> str:
    """``500 -> "500 Hz"``, ``2700 -> "2.7 kHz"``, ``2.4e6 -> "2.4 MHz"``."""
    bw_hz = float(bw_hz)
    if bw_hz >= 1e6:
        return f"{bw_hz / 1e6:.3f}".rstrip("0").rstrip(".") + " MHz"
    if bw_hz >= 1e3:
        return f"{bw_hz / 1e3:.2f}".rstrip("0").rstrip(".") + " kHz"
    return f"{bw_hz:.0f} Hz"


_LISTEN_OK = "Listen ✓"


def _power_limit_tip(legal: float, effective: Optional[float]) -> str:
    """Tooltip explaining the legal power limit and the app's TX guard."""
    pct = int(round(POWER_HEADROOM_FACTOR * 100))
    effective = effective or legal * POWER_HEADROOM_FACTOR
    return (
        f"Legal limit: {legal:g} W at your transmitter output. The app only "
        f"blocks power above {pct}% of it ({effective:g} W) to allow for "
        "cable loss; stay within the legal limit itself."
    )


def _in_amateur_band(freq_hz: float) -> bool:
    """Whether ``freq_hz`` lies in any amateur band (for any license class)."""
    return any(
        band.start_hz <= freq_hz <= band.end_hz for band in AMATEUR_BAND_PRIVILEGES
    )


@contextmanager
def _quiet_frequency_manager() -> Iterator[None]:
    """Silence the frequency manager's log while the panel only *inspects*.

    ``is_tx_allowed`` logs a warning for every blocked frequency, which is
    right for a real transmit attempt but turned merely browsing presets
    into a stream of warnings in the Error Log.
    """
    log = logging.getLogger(_frequency_manager_module.__name__)

    def _drop(_record: logging.LogRecord) -> bool:
        return False

    log.addFilter(_drop)
    try:
        yield
    finally:
        log.removeFilter(_drop)


class _WheelGuard(QObject if HAS_PYQT6 else object):
    """Send wheel events over *unfocused* inputs to the scroll area.

    Without this, scrolling the panel with the mouse wheel changes whatever
    combo box, slider or spin box happens to pass under the pointer (e.g.
    retuning or changing the gain by accident). Click an input first to
    adjust it with the wheel.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()  # ignored events propagate to the parent
            return True
        return False


# ---------------------------------------------------------------------------
# Frequency entry
# ---------------------------------------------------------------------------


class _FrequencySpinBox(QDoubleSpinBox if HAS_PYQT6 else object):
    """Spin box that keeps 1 Hz resolution but shows at least three
    decimals and drops the trailing zeros after them.

    In MHz that reads "100.000" or "137.9125", like the toolbar, the Bookmarks
    list and every other MHz readout, instead of an unreadable "100.000000".
    Units with three decimals or fewer (Hz, kHz) are shown unchanged.

    Typing understands units and notations (:func:`parse_frequency_text`:
    ``433.92M``, ``7074k``, ``7.074e6``, ``144,39``, ``162550000`` in Hz).
    Out-of-range text is kept (a red frame says it can't be tuned) instead of
    dropping the digit just typed; Enter or leaving the field with it keeps
    the tuned frequency and emits :attr:`text_rejected` with the reason. Esc
    cancels the edit. The arrows step by the tuning step.
    """

    if HAS_PYQT6:
        #: Typed text that can't be tuned was dropped (Enter or focus out):
        #: why, in plain words.
        text_rejected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._unit_hz = 1e6
        self._min_hz = MIN_FREQUENCY_HZ
        self._max_hz = MAX_FREQUENCY_HZ
        self._range_text = ""
        self._step_hz = FREQ_STEP_HZ
        self.lineEdit().textChanged.connect(self._update_tone)

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        text = super().textFromValue(value)
        whole, point, frac = text.partition(self.locale().decimalPoint())
        if not point or len(frac) <= 3:
            return text
        return f"{whole}{point}{frac.rstrip('0').ljust(3, '0')}"

    # -- reading typed text ------------------------------------------------
    def _parse(self, text: str) -> ParsedFrequency:
        return parse_frequency_text(
            text,
            self._unit_hz,
            self._min_hz,
            self._max_hz,
            grouped=self.isGroupSeparatorShown(),
            group_separator=self.locale().groupSeparator(),
            range_text=self._range_text,
        )

    def validate(self, text: str, pos: int):  # noqa: N802 (Qt API)
        state = self._parse(text).state
        if state == ACCEPTABLE:
            return QValidator.State.Acceptable, text, pos
        if state == INTERMEDIATE:
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos

    def valueFromText(self, text: str) -> float:  # noqa: N802 (Qt API)
        parsed = self._parse(text)
        if parsed.state != ACCEPTABLE or parsed.hz is None:
            return self.value()
        return parsed.hz / self._unit_hz

    def fixup(self, text: str) -> str:  # noqa: D102 (Qt API)
        # Nothing to repair: text that can't be tuned is reported (Enter or
        # focus out) and the field goes back to the tuned frequency.
        return text

    def _edited(self) -> bool:
        """Whether the text differs from the tuned value (typing pending)."""
        return self.lineEdit().text() != self.textFromValue(self.value())

    def _update_tone(self, text: str = "") -> None:
        """A red frame while the text is a number that can't be tuned."""
        parsed = self._parse(self.lineEdit().text())
        bad = parsed.state != ACCEPTABLE and parsed.hz is not None
        set_tone(self, "danger" if bad else None)  # (no-op when unchanged)

    def _report_unusable_text(self) -> None:
        text = self.lineEdit().text()
        if not text.strip() or not self._edited():
            return
        parsed = self._parse(text)
        if parsed.state != ACCEPTABLE:
            self.text_rejected.emit(parsed.problem or f"Can't tune to “{text}”.")

    def cancel_edit(self) -> None:
        """Drop typed text: show the tuned frequency again."""
        if self._edited():
            self.setValue(self.value())  # rewrites the text, emits nothing
        self._update_tone()

    # -- keys and stepping --------------------------------------------------
    def keyPressEvent(self, event):  # noqa: N802 (Qt API)
        key = event.key()
        if key == Qt.Key.Key_Escape:
            # Cancel the edit; the key goes on to the window, which returns
            # the focus to the plots.
            self.cancel_edit()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._report_unusable_text()
        super().keyPressEvent(event)
        # Qt rewrites the text (e.g. back to the tuned value) with the line
        # edit's signals blocked: refresh the frame here.
        self._update_tone()

    def focusOutEvent(self, event):  # noqa: N802 (Qt API)
        self._report_unusable_text()
        super().focusOutEvent(event)
        self._update_tone()

    def stepBy(self, steps: int) -> None:  # noqa: N802 (Qt API)
        """Arrows, Page Up/Down and the wheel: whole tuning steps (like
        Left/Right on the plots), free of the rounding drift a step that
        isn't whole Hz (8.33 kHz) would collect."""
        if self._edited():
            super().stepBy(steps)  # interprets the typed text first
            return
        current = round(self.value() * self._unit_hz)
        target = step_frequency(current, self._step_hz, steps)
        target = min(max(target, self._min_hz), self._max_hz)
        self.setValue(target / self._unit_hz)
        if self.style().styleHint(QStyle.StyleHint.SH_SpinBox_SelectOnStep, None, self):
            self.selectAll()


class SignedSpinBox(QSpinBox if HAS_PYQT6 else object):
    """A spin box that shows the sign: "+12 ppm", "-7 ppm", "0 ppm"."""

    def textFromValue(self, value: int) -> str:  # noqa: N802 (Qt API)
        return f"{value:+d}" if value else "0"


class _ValueComboBox(QComboBox if HAS_PYQT6 else object):
    """A combo box whose items show a label for a plain value, e.g.
    "75 kHz (broadcast)" for "75 kHz".

    Python code works with the value: :meth:`currentText` returns it, and
    :meth:`findText` and :meth:`setCurrentText` accept the value (or the
    label), so settings, bookmarks and presets keep storing "75 kHz".
    """

    def add_value(self, value: str, label: str, tip: str = "") -> None:
        self.addItem(label, value)
        if tip:
            self.setItemData(self.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)

    def current_label(self) -> str:
        """The text shown, e.g. ``"75 kHz (broadcast)"``."""
        return super().currentText()

    def currentText(self) -> str:  # noqa: N802 (Qt API)
        value = self.currentData()
        return value if isinstance(value, str) else super().currentText()

    def findText(  # noqa: N802 (Qt API)
        self,
        text: str,
        flags: "Qt.MatchFlag" = (
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive
            if HAS_PYQT6
            else None
        ),
    ) -> int:
        index = self.findData(text)
        return index if index >= 0 else super().findText(text, flags)

    def setCurrentText(self, text: str) -> None:  # noqa: N802 (Qt API)
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)


class FrequencyInput(QWidget if HAS_PYQT6 else object):
    """Frequency input widget with unit selection.

    Typing a value and pressing Enter (or leaving the field) tunes once,
    rather than retuning on every keystroke. Switching the unit keeps the
    tuned frequency and only changes how it is shown.

    The field accepts the connected receiver's tuning range
    (:meth:`set_tuning_range`); its arrows step by the tuning step
    (:meth:`set_step_hz`). Text that can't be tuned is reported through
    :attr:`entry_rejected` instead of being cut down to something that can.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)
        #: Typed text was not tuned: why (e.g. "2400 MHz is out of range:
        #: RTL-SDR tunes 0.5–1766 MHz.").
        entry_rejected = pyqtSignal(str)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._frequency_hz = float(DEFAULT_FREQUENCY_HZ)
        self._min_hz = MIN_FREQUENCY_HZ
        self._max_hz = MAX_FREQUENCY_HZ
        self._range_text = ""
        self._step_hz = FREQ_STEP_HZ

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Frequency value (in the selected unit)
        self._freq_input = _FrequencySpinBox()
        self._freq_input.setKeyboardTracking(False)
        self._freq_input.setAccelerated(True)
        self._freq_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._freq_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._freq_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._freq_input.setMinimumWidth(96)
        self._freq_input.setAccessibleName("Tuned frequency")
        self._freq_input.text_rejected.connect(self.entry_rejected)
        layout.addWidget(self._freq_input, 1)

        # Unit selector
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(list(_UNITS))
        self._unit_combo.setCurrentText("MHz")
        self._unit_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._unit_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._unit_combo.setToolTip(
            "Unit for the frequency entry (changing it keeps the tuned frequency)"
        )
        self._unit_combo.setAccessibleName("Frequency unit")
        layout.addWidget(self._unit_combo)

        self._apply_unit()
        self._update_tooltip()
        self._freq_input.valueChanged.connect(self._on_value_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)

    def _get_multiplier(self) -> float:
        """Get current unit multiplier."""
        return _UNITS.get(self._unit_combo.currentText(), _UNITS["MHz"])[0]

    def _apply_unit(self) -> None:
        """Show the current frequency in the selected unit (no signals)."""
        unit = self._unit_combo.currentText()
        mult, decimals = _UNITS.get(unit, _UNITS["MHz"])
        spin = self._freq_input
        spin.blockSignals(True)
        spin._unit_hz = mult
        spin._min_hz, spin._max_hz = self._min_hz, self._max_hz
        spin._range_text = self._range_text
        spin._step_hz = self._step_hz
        spin.setGroupSeparatorShown(unit in _GROUPED_UNITS)
        spin.setDecimals(decimals)
        spin.setRange(self._min_hz / mult, self._max_hz / mult)
        spin.setSingleStep(self._step_hz / mult)
        spin.setValue(self._frequency_hz / mult)
        spin.blockSignals(False)
        spin._update_tone()

    def _update_tooltip(self) -> None:
        step = _format_bandwidth(self._step_hz)
        page = _format_bandwidth(10 * self._step_hz)
        limits = self._range_text or (
            f"Tunes {format_mhz_range(self._min_hz, self._max_hz)}"
        )
        self._freq_input.setToolTip(
            "Tuned frequency: what you listen to, decode and measure.\n"
            "Type a value and press Enter (Esc cancels): 145.8, 433.92M, "
            "7074k, or 162550000 in Hz.\n"
            f"{limits}. The arrow keys step {step} (the tuning step; Page "
            f"Up/Down: {page}); click first to step with the wheel."
        )

    def set_tuning_range(
        self, minimum_hz: float, maximum_hz: float, range_text: str = ""
    ) -> None:
        """Accept ``minimum_hz`` to ``maximum_hz`` (the connected receiver's
        range). ``range_text`` explains it in messages and the tooltip, e.g.
        ``"RTL-SDR tunes 0.5–1766 MHz"``. The tuned frequency is kept: the
        caller retunes when it lies outside."""
        low, high = float(minimum_hz), float(maximum_hz)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            return
        self._min_hz, self._max_hz = low, high
        self._range_text = str(range_text or "")
        self._apply_unit()
        self._update_tooltip()

    def tuning_range(self) -> Tuple[float, float]:
        """The frequencies the field accepts, in Hz."""
        return self._min_hz, self._max_hz

    def set_step_hz(self, step_hz: float) -> None:
        """Make the arrows (and Page Up/Down, ten steps) tune by ``step_hz``."""
        try:
            step = float(step_hz)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(step) and step > 0) or step == self._step_hz:
            return
        self._step_hz = step
        self._freq_input._step_hz = step
        self._freq_input.setSingleStep(step / self._get_multiplier())
        self._update_tooltip()

    def step_hz(self) -> float:
        """The step the arrows tune by, in Hz."""
        return self._step_hz

    def cancel_edit(self) -> None:
        """Drop any typed text and show the tuned frequency again."""
        self._freq_input.cancel_edit()

    def _on_value_changed(self, value: float):
        """Handle value change."""
        # Every unit resolves to 1 Hz; rounding drops float noise
        # (145.8 * 1e6 = 145800000.00000003).
        self._frequency_hz = float(round(value * self._get_multiplier()))
        self.frequency_changed.emit(self._frequency_hz)

    def _on_unit_changed(self, index: int):
        """Handle unit change: same frequency, new unit."""
        self._apply_unit()

    def set_frequency(self, freq_hz: float):
        """Set frequency in Hz (clamped to the tuning range, no signal).

        Values that are not finite numbers (NaN, infinity, e.g. from a
        corrupted setting or ``-f nan``) are ignored: the entry keeps the
        frequency it had.
        """
        try:
            freq_hz = float(freq_hz)
        except (TypeError, ValueError):
            return
        if not math.isfinite(freq_hz):
            return
        self._frequency_hz = min(max(freq_hz, self._min_hz), self._max_hz)
        mult = self._get_multiplier()
        self._freq_input.blockSignals(True)
        self._freq_input.setValue(self._frequency_hz / mult)
        self._freq_input.blockSignals(False)
        self._freq_input._update_tone()

    def get_frequency(self) -> float:
        """Get frequency in Hz."""
        return self._frequency_hz


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class ControlPanel(QWidget if HAS_PYQT6 else object):
    """
    Device control panel.

    Provides controls for frequency, presets, gain, bandwidth, demodulation,
    squelch, recording and the transmit license profile.
    """

    if HAS_PYQT6:
        frequency_changed = pyqtSignal(float)
        gain_changed = pyqtSignal(float)
        bandwidth_changed = pyqtSignal(float)
        demod_changed = pyqtSignal(str)
        recording_started = pyqtSignal(str)  # format
        recording_stopped = pyqtSignal()
        # Pause (True) / resume (False). The Pause button is only enabled
        # when something is connected to this signal.
        recording_paused = pyqtSignal(bool)
        # What Record records changed ("iq" or "audio").
        record_kind_changed = pyqtSignal(str)
        # The take shown under the Record buttons: Save..., Discard (or
        # Unload) and Open Folder; the recordings folder and auto-save.
        take_save_requested = pyqtSignal()
        take_discard_requested = pyqtSignal()
        open_folder_requested = pyqtSignal()
        folder_change_requested = pyqtSignal()
        auto_save_changed = pyqtSignal(bool)
        license_changed = pyqtSignal(object)  # LicenseClass
        squelch_changed = pyqtSignal(float)  # dB
        agc_changed = pyqtSignal(bool)
        # Squelch on/off and Auto (the threshold follows the noise).
        squelch_enabled_changed = pyqtSignal(bool)
        squelch_auto_changed = pyqtSignal(bool)
        # The remembered Bandwidth / FM deviation of a mode changed.
        filter_memory_changed = pyqtSignal()
        # The Bandwidth choices were replaced (a mode with other widths).
        bandwidth_options_changed = pyqtSignal()
        cw_pitch_changed = pyqtSignal(int)  # Hz
        noise_reduction_changed = pyqtSignal(str)  # off / light / strong
        audio_agc_changed = pyqtSignal(str)  # off / fast / medium / slow
        tone_squelch_changed = pyqtSignal(float)  # Hz, 0 = off
        iq_options_changed = pyqtSignal(bool, bool)  # remove DC, balance I/Q
        # Receiver > Sample rate: a rate picked (applied live by the window).
        sample_rate_selected = pyqtSignal(float)
        # HackRF One: LNA and VGA gains (dB), and the +14 dB RF amp.
        hackrf_gain_changed = pyqtSignal(int, int)
        amp_changed = pyqtSignal(bool)
        # Front end of the connected device: frequency correction (ppm),
        # direct sampling ("q", "i" or "off") and the bias tee.
        ppm_changed = pyqtSignal(int)
        direct_sampling_changed = pyqtSignal(str)
        bias_tee_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._form_labels: List[QWidget] = []
        self._wheel_guard = _WheelGuard(self)
        # A recording is "armed" while it waits for the receiver to start
        # ("waiting" once it has captured something: it resumes).
        self._armed = False
        self._waiting = False
        # Bandwidth and FM deviation last used with each filter profile
        # (see filter_profile), and whether a mode change is applying them.
        self._filter_memory: Dict[str, Dict[str, str]] = {}
        self._applying_filter = False
        # The squelch slider shows the Auto threshold without emitting; a
        # threshold set by hand turns Auto off (and is emitted once).
        self._following_auto = False
        self._manual_threshold = False
        # How the RF gain is set on the connected device (its steps, or the
        # HackRF's LNA/VGA/amp), the gain an automatic gain chose (when the
        # device says) and the sample rate choices shown (to skip refilling
        # them).
        self._gain_model: GainModel = DEFAULT_GAIN_MODEL
        self._auto_gain_db: Optional[float] = None
        self._rate_choices: Tuple = ()
        self._current_rate = 0.0
        self._front_end_kind = ""
        self._setup_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup UI elements."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Everything scrolls vertically; the panel never scrolls sideways.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        outer.addWidget(self._scroll)

        content = QWidget()
        self._content = content
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 6)
        layout.setSpacing(8)

        # Where to listen first: Presets sit right under the frequency they
        # set (the quickest way for a newcomer to find something to hear),
        # then how to listen (mode, squelch) and the receiver settings. The
        # occasional sections (recording, license) come last.
        layout.addWidget(self._build_frequency_group())
        layout.addWidget(self._build_presets_group())
        layout.addWidget(self._build_demod_group())
        layout.addWidget(self._build_audio_group())
        layout.addWidget(self._build_receiver_group())
        layout.addWidget(self._build_recording_group())
        layout.addWidget(self._build_license_group())
        layout.addStretch()

        self._scroll.setWidget(content)

        self._align_form_labels()
        self._install_wheel_guard()

        # Initial state
        self._populate_preset_categories()
        self._sync_license_combo()
        self._update_fm_dev_enabled()
        self._update_mode_controls()
        self._update_record_controls(False)

    def minimumSizeHint(self) -> "QSize":  # noqa: N802 (Qt API)
        """Short (it scrolls) but never narrower than its content.

        The scroll area never scrolls sideways, so without this a splitter
        could squeeze the panel below its content width and clip the right
        edge of every group (the tool tabs alone allow ~290 px).
        """
        hint = super().minimumSizeHint()
        scroll = self._scroll
        bar = scroll.verticalScrollBar().sizeHint().width()
        if bar <= 0:
            bar = scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = self._content.minimumSizeHint().width() + bar + 2 * scroll.frameWidth()
        return QSize(max(hint.width(), width), hint.height())

    @staticmethod
    def _make_form(group: "QGroupBox") -> "QFormLayout":
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        return form

    def _add_row(self, form: "QFormLayout", text: str, field) -> "QLabel":
        """Add a labelled row, remembering the label for column alignment."""
        label = QLabel(text)
        if isinstance(field, QWidget):
            label.setBuddy(field)
        form.addRow(label, field)
        self._form_labels.append(label)
        return label

    @staticmethod
    def _shrinkable_combo(combo: "QComboBox", min_chars: int = 8) -> "QComboBox":
        """Let a combo box shrink below its longest item (no sideways scroll)."""
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(min_chars)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    @staticmethod
    def _value_label(text: str, widest: str) -> "QLabel":
        """Right-aligned readout next to a slider, wide enough not to jitter."""
        label = QLabel(text)
        set_role(label, "value")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.ensurePolished()
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance(widest) + 4)
        return label

    @staticmethod
    def _hint(text: str = "") -> "QLabel":
        label = QLabel(text)
        label.setWordWrap(True)
        set_role(label, "hint")
        return label

    def _align_form_labels(self) -> None:
        """Give every form's label column the same width, so fields line up."""
        if not self._form_labels:
            return
        for label in self._form_labels:
            label.ensurePolished()
        width = max(label.sizeHint().width() for label in self._form_labels)
        for label in self._form_labels:
            label.setMinimumWidth(width)

    def _install_wheel_guard(self) -> None:
        for kind in (QComboBox, QAbstractSpinBox, QSlider):
            for widget in self._content.findChildren(kind):
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                widget.installEventFilter(self._wheel_guard)

    # -- Frequency -------------------------------------------------------

    def _build_frequency_group(self) -> "QGroupBox":
        group = QGroupBox("Frequency")
        box = QVBoxLayout(group)
        box.setSpacing(8)

        self._freq_input = FrequencyInput()
        self._freq_input.frequency_changed.connect(self.frequency_changed)
        box.addWidget(self._freq_input)

        # Quick-tune steps: down steps | up steps
        steps = QHBoxLayout()
        steps.setSpacing(4)
        self._quick_tune_buttons: List[QPushButton] = []
        for i, offset in enumerate(QUICK_TUNE_STEPS):
            if i == len(QUICK_TUNE_STEPS) // 2:
                steps.addSpacing(6)
            # A true minus sign: in the proportional button font a hyphen is
            # narrower than "+", so "-10k" and "+10k" looked mismatched.
            btn = QPushButton(self._format_offset(offset).replace("-", _MINUS))
            set_role(btn, "compact")
            # An explicit minimum (text + padding) lets the six buttons share
            # a 360 px column; otherwise styles impose a ~80 px button minimum.
            btn.ensurePolished()
            btn.setMinimumWidth(btn.fontMetrics().horizontalAdvance(btn.text()) + 12)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            direction = "up" if offset > 0 else "down"
            spoken = f"Tune {direction} {_format_bandwidth(abs(offset))}"
            btn.setToolTip(spoken)
            # Screen readers announce this rather than "+10k".
            btn.setAccessibleName(spoken)
            btn.clicked.connect(lambda _checked=False, o=offset: self._quick_tune(o))
            steps.addWidget(btn)
            self._quick_tune_buttons.append(btn)
        box.addLayout(steps)
        return group

    # -- Presets ---------------------------------------------------------

    def _build_presets_group(self) -> "QGroupBox":
        group = QGroupBox("Presets")
        form = self._make_form(group)

        self._category_combo = self._shrinkable_combo(QComboBox())
        self._category_combo.setToolTip("Group of frequency presets to choose from")
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        self._add_row(form, "Category:", self._category_combo)

        # The preset combo gets the whole field width: sharing its row with
        # the Apply button truncated names such as "NOAA APT (NOAA-19)" at
        # 360 px, which made the three NOAA presets indistinguishable.
        self._preset_combo = self._shrinkable_combo(QComboBox())
        self._preset_combo.setToolTip("Frequency preset (details are shown below)")
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._add_row(form, "Preset:", self._preset_combo)

        # Preset details and its transmit status
        self._preset_info = self._hint()
        form.addRow(self._preset_info)
        self._preset_tx = self._hint()
        form.addRow(self._preset_tx)

        apply_row = QHBoxLayout()
        apply_row.setContentsMargins(0, 0, 0, 0)
        apply_row.addStretch(1)
        self._apply_preset_btn = QPushButton("Apply Preset")
        self._apply_preset_btn.setToolTip(_APPLY_PRESET_TIP)
        self._apply_preset_btn.clicked.connect(self._apply_preset)
        apply_row.addWidget(self._apply_preset_btn)
        form.addRow(apply_row)
        return group

    # -- Receiver --------------------------------------------------------

    def _build_receiver_group(self) -> "QGroupBox":
        group = QGroupBox("Receiver")
        form = self._make_form(group)
        self._receiver_form = form

        # The sample rate (the span), changed live: the connected device's
        # choices (set_sample_rates).
        self._rate_combo = self._shrinkable_combo(QComboBox(), 8)
        self._rate_combo.setAccessibleName("Sample rate")
        self._rate_combo.setToolTip(self._RATE_TIP)
        self._rate_combo.activated.connect(self._on_rate_activated)
        self._add_row(form, "Sample rate:", self._rate_combo)

        gain_row = QWidget()
        row = QHBoxLayout(gain_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 50)
        self._gain_slider.setValue(20)
        self._gain_slider.setPageStep(5)
        self._gain_slider.setToolTip(get_short_tip("gain"))
        self._gain_slider.setAccessibleName("RF gain")
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        row.addWidget(self._gain_slider, 1)
        self._gain_label = self._value_label("20 dB", _VALUE_LABEL_WIDEST)
        row.addWidget(self._gain_label)
        self._add_row(form, "RF gain:", gain_row).setBuddy(self._gain_slider)

        # HackRF One: its two gain stages and the RF amplifier (instead of
        # the RF gain slider).
        lna_row, self._lna_slider, self._lna_label = self._stage_slider(
            "LNA gain",
            HACKRF_LNA_STEPS,
            HACKRF_DEFAULT_LNA,
            "The HackRF's RF (LNA) gain: 0-40 dB in 8 dB steps. Raise it first "
            "for weak signals, then the VGA gain.",
        )
        self._add_row(form, "LNA gain:", lna_row).setBuddy(self._lna_slider)
        vga_row, self._vga_slider, self._vga_label = self._stage_slider(
            "VGA gain",
            HACKRF_VGA_STEPS,
            HACKRF_DEFAULT_VGA,
            "The HackRF's baseband (VGA) gain: 0-62 dB in 2 dB steps, after "
            "the LNA.",
        )
        self._add_row(form, "VGA gain:", vga_row).setBuddy(self._vga_slider)
        self._amp_check = QCheckBox(AMP_TEXT)
        self._amp_check.setToolTip(AMP_TIP)
        self._amp_check.toggled.connect(self._on_amp_toggled)
        form.addRow("", self._amp_check)
        self._amp_warning = self._hint(AMP_WARNING)
        set_tone(self._amp_warning, "warning")
        form.addRow(self._amp_warning)

        self._agc_check = QCheckBox("Tuner AGC")
        self._agc_check.setToolTip(AGC_TIP)
        self._agc_check.toggled.connect(self._on_agc_changed)
        form.addRow("", self._agc_check)

        self._bw_combo = self._shrinkable_combo(QComboBox(), 6)
        self._bw_combo.addItems(list(BANDWIDTH_OPTIONS))
        self._bw_combo.setCurrentText(DEFAULT_BANDWIDTH)
        self._bw_combo.setToolTip(get_short_tip("bandwidth"))
        self._bw_combo.currentTextChanged.connect(self._on_bandwidth_changed)
        self._add_row(form, "Bandwidth:", self._bw_combo)

        # Corrections of the raw samples (off by default).
        iq_row = QWidget()
        row = QHBoxLayout(iq_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self._dc_check = QCheckBox("Remove DC spike")
        self._dc_check.setToolTip(
            "Removes the spike most SDRs show at the center of the span (the "
            "tuner's own leakage). It also removes a carrier exactly at the "
            "center, so leave it off for an AM station tuned there."
        )
        self._iq_check = QCheckBox("Balance I/Q")
        self._iq_check.setToolTip(
            "Measures and corrects the I/Q gain and phase error, which shows "
            "as faint mirror images of strong signals on the other side of "
            "the center."
        )
        for check in (self._dc_check, self._iq_check):
            check.toggled.connect(self._on_iq_options_changed)
            row.addWidget(check)
        row.addStretch(1)
        form.addRow(iq_row)  # full width: the two say what they do

        # The connected device's front end (shown for the devices that have
        # each setting: see set_front_end).
        self._ppm_spin = SignedSpinBox()
        self._ppm_spin.setRange(*PPM_RANGE)
        self._ppm_spin.setSuffix(" ppm")
        self._ppm_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._ppm_spin.setKeyboardTracking(False)  # applied on Enter, not per digit
        self._ppm_spin.setAccessibleName("Frequency correction")
        self._ppm_spin.setToolTip(PPM_TIP)
        self._ppm_spin.valueChanged.connect(self._on_ppm_changed)
        self._add_row(form, "Correction:", self._ppm_spin).setToolTip(PPM_TIP)
        self._ppm_hint = self._hint(PPM_HINT)
        self._ppm_hint.setToolTip(PPM_TIP)
        form.addRow("", self._ppm_hint)

        self._direct_combo = self._shrinkable_combo(_ValueComboBox(), 8)
        for value, label, tip in DIRECT_SAMPLING_CHOICES:
            self._direct_combo.add_value(value, label, tip)
        self._direct_combo.setAccessibleName("Direct sampling")
        self._direct_combo.setToolTip(DIRECT_SAMPLING_TIP)
        self._direct_combo.currentIndexChanged.connect(self._on_direct_changed)
        self._add_row(form, "Direct sampling:", self._direct_combo).setToolTip(
            DIRECT_SAMPLING_TIP
        )

        self._bias_check = QCheckBox(BIAS_TEE_TEXT)
        self._bias_check.setToolTip(BIAS_TEE_TIPS["rtlsdr"])
        self._bias_check.toggled.connect(self._on_bias_toggled)
        form.addRow(self._bias_check)  # full width: it says what it does
        self._bias_warning = self._hint(BIAS_TEE_WARNING)
        set_tone(self._bias_warning, "warning")
        form.addRow(self._bias_warning)

        # Nothing connected yet: the RF gain slider, no device settings.
        for widget in (
            self._lna_slider.parentWidget(),
            self._vga_slider.parentWidget(),
            self._amp_check,
            self._amp_warning,
            self._ppm_spin,
            self._ppm_hint,
            self._direct_combo,
            self._bias_check,
            self._bias_warning,
        ):
            form.setRowVisible(widget, False)
        self.set_sample_rates((), 0.0, False)
        return group

    _RATE_TIP = (
        "Samples per second the receiver takes: the span shown (2.4 MS/s = "
        "2.4 MHz). A wider span shows more, a narrower one uses less CPU. "
        "Changing it doesn't reconnect the device (also: click the rate in "
        "the status bar, or Device > Sample Rate)."
    )

    def _stage_slider(
        self, name: str, steps: Tuple[int, ...], value: int, tip: str
    ) -> Tuple["QWidget", "QSlider", "QLabel"]:
        """A row widget with a slider over a gain stage's steps (its index)
        and a dB readout."""
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, len(steps) - 1)
        slider.setPageStep(1 if len(steps) < 10 else 5)
        slider.setValue(steps.index(value))
        slider.setToolTip(tip)
        slider.setAccessibleName(name)
        slider.valueChanged.connect(self._on_stage_changed)
        row.addWidget(slider, 1)
        label = self._value_label(format_db(value), _VALUE_LABEL_WIDEST)
        row.addWidget(label)
        return row_widget, slider, label

    # -- Demodulation ----------------------------------------------------

    def _build_demod_group(self) -> "QGroupBox":
        group = QGroupBox("Demodulation")
        form = self._make_form(group)

        self._demod_combo = self._shrinkable_combo(QComboBox(), 6)
        for index, (text, tip_key) in enumerate(DEMOD_MODES):
            self._demod_combo.addItem(text)
            tip = get_short_tip(tip_key) if tip_key else ""
            if text == "None (I/Q)":
                tip = "No audio: show the spectrum and record raw I/Q only."
            elif text == "CW":
                tip = "Morse code (continuous wave), heard as a tone."
            elif text == "USB":
                tip = f"Upper sideband. {tip} Usual for HF voice above 10 MHz."
            elif text == "LSB":
                tip = f"Lower sideband. {tip} Usual for HF voice below 10 MHz."
            self._demod_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._demod_combo.setToolTip("How the tuned signal is turned into audio")
        self._demod_combo.currentTextChanged.connect(self._on_demod_changed)
        self._add_row(form, "Mode:", self._demod_combo)

        self._fm_dev_combo = self._shrinkable_combo(_ValueComboBox(), 6)
        for value, text, tip in FM_DEVIATIONS:
            self._fm_dev_combo.add_value(value, text, tip)
        self._fm_dev_combo.setCurrentText(DEFAULT_FM_DEVIATION)
        self._fm_dev_combo.currentIndexChanged.connect(self._on_fm_deviation_changed)
        self._add_row(form, "FM deviation:", self._fm_dev_combo)

        # The CW beat note: only shown in CW.
        self._cw_pitch_spin = QSpinBox()
        self._cw_pitch_spin.setRange(*CW_PITCH_RANGE)
        self._cw_pitch_spin.setSingleStep(10)
        self._cw_pitch_spin.setValue(DEFAULT_CW_PITCH_HZ)
        self._cw_pitch_spin.setSuffix(" Hz")
        self._cw_pitch_spin.setToolTip(
            "Pitch of the CW tone you hear: the carrier comes out at this "
            "audio frequency (400-1000 Hz)."
        )
        self._cw_pitch_spin.setAccessibleName("CW pitch")
        self._cw_pitch_spin.valueChanged.connect(self.cw_pitch_changed)
        self._add_row(form, "CW pitch:", self._cw_pitch_spin)
        self._demod_form = form

        # Squelch: the row's label is its on/off switch; Auto follows the
        # channel's noise; the slider is the threshold (and shows the Auto
        # one while Auto is on).
        self._squelch_check = QCheckBox("Squelch:")
        self._squelch_check.setChecked(True)
        self._squelch_check.setToolTip(
            "Squelch on: the audio is muted while the tuned channel is "
            "quieter than the threshold. Off: everything plays, noise too."
        )
        self._squelch_check.toggled.connect(self._on_squelch_enabled)
        squelch_row = QWidget()
        row = QHBoxLayout(squelch_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._squelch_auto = QCheckBox("Auto")
        self._squelch_auto.setChecked(True)
        self._squelch_auto.setToolTip(
            "Sets the threshold a little (7 dB) above the channel's noise and "
            "keeps it there when the gain, the AGC or the Bandwidth change. "
            "Moving the slider sets it by hand instead."
        )
        self._squelch_auto.toggled.connect(self._on_squelch_auto)
        row.addWidget(self._squelch_auto)
        self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
        self._squelch_slider.setRange(*SQUELCH_RANGE)
        self._squelch_slider.setValue(-80)
        self._squelch_slider.setPageStep(5)
        self._squelch_slider.setToolTip(
            "Mutes audio while the tuned channel's level is below this "
            "threshold (dBFS). The LEVEL meter in the toolbar shows that "
            "level and this threshold; you can also drag it there."
        )
        self._squelch_slider.setAccessibleName("Squelch threshold")
        self._squelch_slider.valueChanged.connect(self._on_squelch_changed)
        row.addWidget(self._squelch_slider, 1)
        self._squelch_label = self._value_label("-80 dBFS", _VALUE_LABEL_WIDEST)
        row.addWidget(self._squelch_label)
        form.addRow(self._squelch_check, squelch_row)
        self._form_labels.append(self._squelch_check)
        return group

    # -- Audio -----------------------------------------------------------

    def _build_audio_group(self) -> "QGroupBox":
        group = QGroupBox("Audio")
        form = self._make_form(group)

        self._nr_combo = self._shrinkable_combo(_ValueComboBox(), 6)
        for key, text, tip in NOISE_REDUCTION_LEVELS:
            self._nr_combo.add_value(key, text, tip)
        self._nr_combo.setToolTip(
            "Reduces steady hiss in the audio. It learns the noise by "
            "itself; Strong removes more but can make voices sound watery."
        )
        self._nr_combo.currentIndexChanged.connect(
            lambda _i: self.noise_reduction_changed.emit(self._nr_combo.currentText())
        )
        self._add_row(form, "Noise reduction:", self._nr_combo)

        self._audio_agc_combo = self._shrinkable_combo(_ValueComboBox(), 6)
        for key, text, tip in AGC_SPEED_LABELS:
            self._audio_agc_combo.add_value(key, text, tip)
        self._audio_agc_combo.setCurrentText(DEFAULT_AGC_SPEED)
        self._audio_agc_combo.currentIndexChanged.connect(
            lambda _i: self.audio_agc_changed.emit(self._audio_agc_combo.currentText())
        )
        self._add_row(form, "Audio AGC:", self._audio_agc_combo)

        self._tone_combo = self._shrinkable_combo(_ValueComboBox(), 6)
        self._tone_combo.add_value("off", "Off", "Plays whatever is on the channel")
        for freq in CTCSS_FREQUENCIES:
            self._tone_combo.add_value(f"{freq:.1f}", ctcss_label(freq))
        self._tone_combo.setAccessibleName("Tone squelch")
        self._tone_combo.currentIndexChanged.connect(self._on_tone_changed)
        self._add_row(form, "Tone squelch:", self._tone_combo)

        # The CTCSS tone the station sends (FM only, like the tone squelch).
        self._tone_heard = QLabel("—")
        self._tone_heard.setAccessibleName("Tone heard")
        self._add_row(form, "Tone heard:", self._tone_heard)
        self._audio_form = form
        self.set_detected_tone(None)
        return group

    # -- Recording -------------------------------------------------------

    def _build_recording_group(self) -> "QGroupBox":
        group = QGroupBox("Recording")
        form = self._make_form(group)
        self._recording_form = form

        # What Record records: the I/Q samples or the audio heard.
        self._kind_combo = self._shrinkable_combo(QComboBox())
        for key, text, tip in RECORD_KINDS:
            self._kind_combo.addItem(text, key)
            self._kind_combo.setItemData(
                self._kind_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        self._kind_combo.setToolTip(
            "What the Record buttons record: the I/Q samples (raw radio "
            "samples of the whole span) or the audio you hear (WAV)."
        )
        self._kind_combo.currentIndexChanged.connect(self._on_record_kind_changed)
        self._add_row(form, "Record:", self._kind_combo)

        self._format_combo = self._shrinkable_combo(QComboBox())
        for text, ext in zip(
            RECORDING_FORMATS, RECORDING_FORMAT_EXTENSIONS, strict=True
        ):
            self._format_combo.addItem(text, ext)
        self._format_combo.setItemData(
            0,
            "SigMF (.sigmf-data and .sigmf-meta): keeps the frequency, sample "
            "rate and time, and SDR programs read it.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._format_combo.setToolTip(_FORMAT_TIP)
        self._format_label = self._add_row(form, "Format:", self._format_combo)

        # Where audio recordings (and auto-saved I/Q takes) go.
        self._folder_btn = QPushButton()
        self._folder_btn.setAutoDefault(False)
        self._folder_btn.setAccessibleName("Recordings folder")
        self._folder_btn.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._folder_btn.clicked.connect(
            lambda _checked=False: self.folder_change_requested.emit()
        )
        self._add_row(form, "Folder:", self._folder_btn)
        self._folder_path = ""

        self._auto_save_check = QCheckBox("Save automatically")
        self._auto_save_check.setToolTip(_AUTO_SAVE_TIP)
        self._auto_save_check.toggled.connect(self.auto_save_changed.emit)
        form.addRow("", self._auto_save_check)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        # Neutral at rest, red ("danger") only while recording, exactly like
        # the toolbar's Record button: red always means "recording now".
        self._record_btn = QPushButton(_RECORD_TEXT)
        self._record_btn.setCheckable(True)
        self._record_btn.setToolTip(
            "Start or stop capturing I/Q samples. Save them afterwards with "
            "Save... below or File > Save Recording."
        )
        self._record_btn.toggled.connect(self._on_record_toggled)
        buttons.addWidget(self._record_btn, 1)

        self._pause_btn = QPushButton(_PAUSE_TEXT)
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        buttons.addWidget(self._pause_btn, 1)
        form.addRow(buttons)

        status = QHBoxLayout()
        status.setSpacing(8)
        self._record_status = QLabel(_STATUS_READY)
        set_role(self._record_status, "hint")
        # Wraps rather than widening the panel for the longer armed text.
        self._record_status.setWordWrap(True)
        status.addWidget(self._record_status, 1)
        self._record_time = QLabel("00:00:00")
        set_role(self._record_time, "lcd-small")
        self._record_time.setToolTip("Elapsed recording time (hh:mm:ss)")
        status.addWidget(self._record_time)
        form.addRow(status)

        # The take after Stop ("Unsaved · 00:00:04 · 73 MB @ 100.100 MHz")
        # with Save... and Discard, or the file it was saved to.
        self._take_box = QWidget()
        take = QVBoxLayout(self._take_box)
        take.setContentsMargins(0, 0, 0, 0)
        take.setSpacing(4)
        self._take_label = QLabel()
        self._take_label.setWordWrap(True)
        self._take_label.setAccessibleName("Recording to save")
        take.addWidget(self._take_label)
        take_buttons = QHBoxLayout()
        take_buttons.setSpacing(6)
        self._take_save_btn = QPushButton("Save...")
        self._take_save_btn.setAutoDefault(False)
        self._take_save_btn.clicked.connect(
            lambda _checked=False: self.take_save_requested.emit()
        )
        take_buttons.addWidget(self._take_save_btn, 1)
        self._take_discard_btn = QPushButton("Discard")
        self._take_discard_btn.setAutoDefault(False)
        self._take_discard_btn.clicked.connect(
            lambda _checked=False: self.take_discard_requested.emit()
        )
        take_buttons.addWidget(self._take_discard_btn, 1)
        self._take_folder_btn = QPushButton("Open Folder")
        self._take_folder_btn.setAutoDefault(False)
        self._take_folder_btn.setToolTip("Show the saved file in your file manager")
        self._take_folder_btn.clicked.connect(
            lambda _checked=False: self.open_folder_requested.emit()
        )
        take_buttons.addWidget(self._take_folder_btn, 1)
        take.addLayout(take_buttons)
        form.addRow(self._take_box)
        self._take_box.setVisible(False)
        self.set_recordings_folder("")
        return group

    # -- License ---------------------------------------------------------

    def _build_license_group(self) -> "QGroupBox":
        group = QGroupBox("License Profile")
        form = self._make_form(group)

        self._license_combo = self._shrinkable_combo(QComboBox())
        for license_class, text, summary in LICENSE_CLASSES:
            self._license_combo.addItem(text, license_class)
            self._license_combo.setItemData(
                self._license_combo.count() - 1,
                f"Transmit privileges: {summary}",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._license_combo.setToolTip(
            "Your amateur license class. It decides where transmitting is "
            "allowed; receiving is never restricted."
        )
        self._license_combo.currentIndexChanged.connect(self._on_license_changed)
        self._add_row(form, "Class:", self._license_combo)

        self._license_info = self._hint()
        form.addRow(self._license_info)

        self._tx_warning = QLabel(TX_POWER_REMINDER)
        self._tx_warning.setWordWrap(True)
        set_role(self._tx_warning, "callout", "warning")
        self._tx_warning.setToolTip(TX_POWER_WARNING)
        form.addRow(self._tx_warning)
        return group

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

    def _format_offset(self, offset: float) -> str:
        """Format frequency offset for button label."""
        if abs(offset) >= 1e6:
            return f"{'+' if offset > 0 else ''}{offset/1e6:.0f}M"
        elif abs(offset) >= 1e3:
            return f"{'+' if offset > 0 else ''}{offset/1e3:.0f}k"
        else:
            return f"{'+' if offset > 0 else ''}{offset:.0f}"

    def _quick_tune(self, offset: float):
        """Quick tune by offset."""
        current = self._freq_input.get_frequency()
        self._freq_input.set_frequency(current + offset)
        new = self._freq_input.get_frequency()  # clamped to the tuning range
        if new != current:
            self.frequency_changed.emit(new)

    def set_frequency(self, freq_hz: float):
        """Set frequency (no ``frequency_changed`` signal)."""
        self._freq_input.set_frequency(freq_hz)

    # ------------------------------------------------------------------
    # Gain / AGC / bandwidth
    # ------------------------------------------------------------------

    def _step_db(self, index: int) -> float:
        """The gain (dB) of a step of the RF gain slider."""
        steps = self._gain_model.steps
        if not steps:
            return float(index)
        return float(steps[max(0, min(len(steps) - 1, int(index)))])

    def _on_gain_changed(self, value: int):
        """Handle gain change."""
        if self._agc_check.isChecked():
            # The tuner owns the gain while AGC is on; the new value is
            # applied when AGC is switched off.
            return
        db = self._step_db(value)
        self._gain_label.setText(format_db(db))
        self.gain_changed.emit(db)

    def _on_agc_changed(self, enabled: bool):
        """Handle AGC toggle."""
        enabled = bool(enabled)
        self._show_agc_state(enabled)
        self.agc_changed.emit(enabled)
        if not enabled and self._gain_model.kind != "hackrf":
            # Back to manual: re-apply the slider's gain. (Enabling AGC used
            # to emit gain_changed(-1), which set a fixed 0 dB manual gain
            # on the device and so switched AGC straight back off.)
            self.gain_changed.emit(self._step_db(self._gain_slider.value()))

    def _show_agc_state(self, enabled: bool) -> None:
        """The RF gain slider and readout for AGC on or off."""
        self._gain_slider.setEnabled(not enabled)
        if enabled:
            auto = self._auto_gain_db
            self._gain_label.setText(
                "Auto" if auto is None else f"Auto ({format_db(round(auto, 1))})"
            )
            self._gain_slider.setToolTip(
                "Gain is set automatically while Tuner AGC is on. Untick it to "
                "set it by hand."
            )
        else:
            self._gain_label.setText(
                format_db(self._step_db(self._gain_slider.value()))
            )
            self._gain_slider.setToolTip(self._gain_tip())

    def _gain_tip(self) -> str:
        model = self._gain_model
        if model.name and len(model.steps) > 1:
            return (
                f"{get_short_tip('gain')} The {model.name} has "
                f"{len(model.steps)} steps from {format_db(model.low)} to "
                f"{format_db(model.high)}."
            )
        return get_short_tip("gain")

    def is_agc_enabled(self) -> bool:
        """Whether AGC is currently enabled."""
        return bool(self._agc_check.isChecked())

    def set_agc_enabled(self, enabled: bool) -> None:
        """Toggle AGC."""
        self._agc_check.setChecked(bool(enabled))

    def set_agc_silently(self, enabled: bool) -> None:
        """Show AGC on or off without emitting anything (the window applies
        it to the device itself)."""
        self._agc_check.blockSignals(True)
        self._agc_check.setChecked(bool(enabled))
        self._agc_check.blockSignals(False)
        self._show_agc_state(bool(enabled))

    def show_auto_gain(self, gain_db: Optional[float]) -> None:
        """The gain the automatic gain chose (None: not known), shown as
        "Auto (25.9 dB)" while AGC is on."""
        self._auto_gain_db = None if gain_db is None else float(gain_db)
        if self._agc_check.isChecked():
            self._show_agc_state(True)

    def set_gain(self, gain_db: float):
        """Set gain (the nearest step the device has)."""
        if self._gain_model.kind == "hackrf":
            lna, vga = split_hackrf_gain(gain_db)
            self._lna_slider.setValue(HACKRF_LNA_STEPS.index(lna))
            self._vga_slider.setValue(HACKRF_VGA_STEPS.index(vga))
            return
        self._gain_slider.setValue(self._gain_model.nearest_index(gain_db))

    def set_gain_silently(self, gain_db: float) -> None:
        """Show ``gain_db`` (its nearest step) without emitting anything."""
        slider = self._gain_slider
        slider.blockSignals(True)
        slider.setValue(self._gain_model.nearest_index(gain_db))
        slider.blockSignals(False)
        self._show_agc_state(self._agc_check.isChecked())

    def gain_db(self) -> float:
        """The RF gain set by hand, in dB (a HackRF: LNA + VGA, plus 14 dB
        with the RF amp on)."""
        if self._gain_model.kind == "hackrf":
            lna, vga, amp = self.hackrf_gains()
            return float(lna + vga + (HACKRF_AMP_DB if amp else 0))
        return self._step_db(self._gain_slider.value())

    def slider_gain_db(self) -> float:
        """The RF gain slider's gain in dB (what the window saves as the
        gain to start with; a HackRF's LNA/VGA are remembered apart)."""
        return self._step_db(self._gain_slider.value())

    def gain_model(self) -> GainModel:
        return self._gain_model

    def set_gain_model(self, model: GainModel, locked: str = "") -> None:
        """Show the RF gain controls ``model`` describes (one slider over
        the device's gain steps, or a HackRF's LNA, VGA and RF amp), with
        Tuner AGC only where the hardware has it. The gain stays as close
        as the new steps allow. ``locked``: why the gain can't be set (a
        recording playing back), shown on the disabled controls. Nothing is
        emitted: the window applies the gain to the device itself."""
        old_db = self.gain_db() if self._gain_model.kind != "hackrf" else None
        changed = model != self._gain_model
        self._gain_model = model
        hackrf = model.kind == "hackrf"
        form = self._receiver_form
        form.setRowVisible(self._gain_slider.parentWidget(), not hackrf)
        for widget in (
            self._lna_slider.parentWidget(),
            self._vga_slider.parentWidget(),
            self._amp_check,
        ):
            form.setRowVisible(widget, hackrf)
        if changed and not hackrf:
            slider = self._gain_slider
            slider.blockSignals(True)
            slider.setRange(0, max(0, len(model.steps) - 1))
            slider.setPageStep(max(1, len(model.steps) // 10))
            if old_db is not None:
                slider.setValue(model.nearest_index(old_db))
            slider.blockSignals(False)
        if not model.agc and self._agc_check.isChecked():
            self.set_agc_silently(False)
        rows = [
            self._gain_slider.parentWidget(),
            self._lna_slider.parentWidget(),
            self._vga_slider.parentWidget(),
        ]
        for widget in rows + [self._amp_check]:
            widget.setEnabled(not locked)
            if locked:
                widget.setToolTip(locked)
            elif widget in rows:
                widget.setToolTip("")
            else:
                widget.setToolTip(AMP_TIP)
        self._agc_check.setEnabled(not locked and model.agc)
        if locked:
            self._agc_check.setToolTip(locked)
        else:
            self._agc_check.setToolTip(AGC_TIP if model.agc else model.agc_tip)
        self._show_agc_state(self._agc_check.isChecked())
        self._update_amp_warning()

    # -- HackRF One: LNA, VGA and the RF amp --------------------------------

    def hackrf_gains(self) -> Tuple[int, int, bool]:
        """``(lna, vga, amp)``: the HackRF gains shown."""
        lna = HACKRF_LNA_STEPS[self._lna_slider.value()]
        vga = HACKRF_VGA_STEPS[self._vga_slider.value()]
        return int(lna), int(vga), bool(self._amp_check.isChecked())

    def set_hackrf_gains(self, lna: int, vga: int, amp: bool) -> None:
        """Show HackRF gains (snapped to their steps) without emitting."""
        lna_i = min(
            range(len(HACKRF_LNA_STEPS)), key=lambda i: abs(HACKRF_LNA_STEPS[i] - lna)
        )
        vga_i = min(
            range(len(HACKRF_VGA_STEPS)), key=lambda i: abs(HACKRF_VGA_STEPS[i] - vga)
        )
        for widget, value in (
            (self._lna_slider, lna_i),
            (self._vga_slider, vga_i),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._amp_check.blockSignals(True)
        self._amp_check.setChecked(bool(amp))
        self._amp_check.blockSignals(False)
        self._show_stage_values()
        self._update_amp_warning()

    def _show_stage_values(self) -> None:
        lna, vga, _amp = self.hackrf_gains()
        self._lna_label.setText(format_db(lna))
        self._vga_label.setText(format_db(vga))

    def _on_stage_changed(self, _index: int) -> None:
        self._show_stage_values()
        lna, vga, _amp = self.hackrf_gains()
        self.hackrf_gain_changed.emit(lna, vga)

    def _on_amp_toggled(self, on: bool) -> None:
        self._update_amp_warning()
        self.amp_changed.emit(bool(on))

    def _update_amp_warning(self) -> None:
        show = self._gain_model.kind == "hackrf" and self._amp_check.isChecked()
        self._receiver_form.setRowVisible(self._amp_warning, show)

    # -- Sample rate ------------------------------------------------------------

    def set_sample_rates(
        self,
        rates: Sequence[float],
        current: float,
        enabled: bool = True,
        tip: str = "",
    ) -> None:
        """Offer ``rates`` (the connected device's) with ``current`` chosen
        (added when it isn't one of them, e.g. set on the command line).
        ``enabled`` False: shown but not changeable, ``tip`` says why."""
        values = sorted({float(r) for r in rates})
        if current > 0 and not any(abs(r - current) < 1.0 for r in values):
            values = sorted(values + [float(current)])
        key = (tuple(values), float(current), bool(enabled), tip)
        self._current_rate = float(current)
        combo = self._rate_combo
        if key == self._rate_choices:
            # The same choices: only show the rate in use again (a pick
            # that was refused goes back).
            index = combo.findData(float(current)) if current > 0 else -1
            if index >= 0 and index != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
            return
        self._rate_choices = key
        combo.blockSignals(True)
        combo.clear()
        for rate in values:
            combo.addItem(rate_choice_label(rate), rate)
            if rate < ADSB_MIN_RATE:
                combo.setItemData(
                    combo.count() - 1, ADSB_RATE_TIP, Qt.ItemDataRole.ToolTipRole
                )
        if values:
            best = min(range(len(values)), key=lambda i: abs(values[i] - current))
            combo.setCurrentIndex(best)
        else:
            combo.addItem("—", 0.0)
        combo.setEnabled(bool(enabled and values))
        combo.setToolTip(tip or self._RATE_TIP)
        combo.blockSignals(False)

    def _on_rate_activated(self, index: int) -> None:
        rate = float(self._rate_combo.itemData(index) or 0.0)
        if rate > 0 and abs(rate - self._current_rate) >= 1.0:
            self.sample_rate_selected.emit(rate)

    # -- Front end: frequency correction, direct sampling, bias tee ---------

    def set_front_end(self, kind: str, front: FrontEnd) -> None:
        """Show the front-end settings device ``kind`` has (an RTL-SDR: ppm,
        direct sampling, bias tee; a HackRF: bias tee; others: none) with
        ``front``'s values, without emitting anything."""
        features = front_end_features(kind)
        self._front_end_kind = kind
        widgets = (
            (self._ppm_spin, front.ppm),
            (self._direct_combo, front.direct_sampling),
            (self._bias_check, front.bias_tee),
        )
        for widget, _value in widgets:
            widget.blockSignals(True)
        self._ppm_spin.setValue(int(front.ppm))
        self._direct_combo.setCurrentText(front.direct_sampling)
        self._bias_check.setChecked(bool(front.bias_tee))
        for widget, _value in widgets:
            widget.blockSignals(False)
        if kind in BIAS_TEE_TIPS:
            self._bias_check.setToolTip(BIAS_TEE_TIPS[kind])
        form = self._receiver_form
        form.setRowVisible(self._ppm_spin, "ppm" in features)
        form.setRowVisible(self._ppm_hint, "ppm" in features)
        form.setRowVisible(self._direct_combo, "direct" in features)
        form.setRowVisible(self._bias_check, "bias_tee" in features)
        self._update_bias_warning()

    def front_end(self) -> FrontEnd:
        """The front-end settings shown."""
        return FrontEnd(
            ppm=int(self._ppm_spin.value()),
            direct_sampling=self._direct_combo.currentText(),
            bias_tee=bool(self._bias_check.isChecked()),
        )

    def _on_ppm_changed(self, value: int) -> None:
        self.ppm_changed.emit(int(value))

    def _on_direct_changed(self, _index: int) -> None:
        self.direct_sampling_changed.emit(self._direct_combo.currentText())

    def _on_bias_toggled(self, on: bool) -> None:
        self._update_bias_warning()
        self.bias_tee_changed.emit(bool(on))

    def _update_bias_warning(self) -> None:
        show = (
            "bias_tee" in front_end_features(self._front_end_kind)
            and self._bias_check.isChecked()
        )
        self._receiver_form.setRowVisible(self._bias_warning, show)

    def _on_bandwidth_changed(self, text: str):
        """Handle bandwidth change (remembered for the mode)."""
        if not text or self._applying_filter:
            return
        self._remember_filter("width", text)
        self.bandwidth_changed.emit(_parse_hz(text))

    # ------------------------------------------------------------------
    # Mode-dependent filter widths
    # ------------------------------------------------------------------

    def _profile(self, mode: Optional[str] = None) -> str:
        mode = self._demod_combo.currentText() if mode is None else mode
        return filter_profile(mode, self._freq_input.get_frequency())

    def _remember_filter(self, key: str, value: str) -> None:
        if not value:
            return
        memory = self._filter_memory.setdefault(self._profile(), {})
        if memory.get(key) != value:
            memory[key] = value
            self.filter_memory_changed.emit()

    def filter_memory(self) -> Dict[str, Dict[str, str]]:
        """The Bandwidth and FM deviation last used with each filter
        profile (a mode, or "FM broadcast"), for saving."""
        return {k: dict(v) for k, v in self._filter_memory.items()}

    def set_filter_memory(self, memory: Dict[str, Dict[str, str]]) -> None:
        """Restore :meth:`filter_memory` (ignores what doesn't fit)."""
        self._filter_memory = {}
        for profile, values in dict(memory or {}).items():
            if profile not in MODE_FILTER_DEFAULTS or not isinstance(values, dict):
                continue
            mode = "FM" if profile == FM_BROADCAST_PROFILE else profile
            kept: Dict[str, str] = {}
            width = values.get("width")
            if isinstance(width, str) and width in bandwidth_options(mode):
                kept["width"] = width
            deviation = values.get("deviation")
            if (
                isinstance(deviation, str)
                and self._fm_dev_combo.findText(deviation) >= 0
            ):
                kept["deviation"] = deviation
            if kept:
                self._filter_memory[profile] = kept

    def mode_filter(self, mode: str) -> Tuple[str, str]:
        """``(bandwidth, FM deviation)`` picking ``mode`` would apply at the
        tuned frequency: the ones last used with it, else its defaults."""
        profile = self._profile(mode)
        width, deviation = MODE_FILTER_DEFAULTS.get(profile, (DEFAULT_BANDWIDTH, ""))
        remembered = self._filter_memory.get(profile, {})
        return (
            remembered.get("width", width),
            remembered.get("deviation", deviation) if mode == "FM" else "",
        )

    def _set_bandwidth_options(self, options: Tuple[str, ...], selected: str) -> None:
        """Show ``options`` in the Bandwidth list with ``selected`` chosen
        (the caller sets ``_applying_filter``)."""
        combo = self._bw_combo
        current = tuple(combo.itemText(i) for i in range(combo.count()))
        if current != tuple(options):
            combo.clear()
            combo.addItems(list(options))
            self.bandwidth_options_changed.emit()
        index = combo.findText(selected)
        if index < 0:
            index = combo.findText(closest_bandwidth(options, _parse_hz(selected)))
        combo.setCurrentIndex(max(index, 0))

    def _apply_mode_filter(self, mode: str) -> None:
        """A new mode: its Bandwidth choices, and the width and FM
        deviation last used with it (or its defaults)."""
        width, deviation = self.mode_filter(mode)
        self._applying_filter = True
        try:
            if mode == "FM" and deviation:
                self._fm_dev_combo.setCurrentText(deviation)
            self._set_bandwidth_options(bandwidth_options(mode), width)
        finally:
            self._applying_filter = False
        self._update_bandwidth_tip()
        text = self._bw_combo.currentText()
        if text:
            self.bandwidth_changed.emit(_parse_hz(text))

    def _update_bandwidth_tip(self) -> None:
        mode = self._demod_combo.currentText()
        if mode in ("USB", "LSB"):
            side = "above" if mode == "USB" else "below"
            tip = (
                "Width of the SSB filter: it keeps from 200 Hz to 200 Hz plus "
                f"this {side} the carrier. Narrower cuts more noise and "
                "neighbours, wider sounds fuller."
            )
        elif mode == "CW":
            tip = (
                "Width of the CW filter around the carrier: narrower picks "
                "one signal out of a crowd, wider is easier to tune."
            )
        elif mode.startswith("None"):
            tip = (
                "Width of the channel the level meter measures (nothing is "
                "demodulated in I/Q mode)."
            )
        else:
            tip = get_short_tip("bandwidth")
        self._bw_combo.setToolTip(tip)

    def bandwidth_choices(self) -> Tuple[str, ...]:
        """The Bandwidth choices shown now (they depend on the mode)."""
        combo = self._bw_combo
        return tuple(combo.itemText(i) for i in range(combo.count()))

    # ------------------------------------------------------------------
    # Demodulation / squelch
    # ------------------------------------------------------------------

    def _on_demod_changed(self, text: str):
        """Handle demodulation mode change: the mode's filter choices and
        the width and deviation last used with it."""
        self._update_fm_dev_enabled()
        self._update_mode_controls()
        self._apply_mode_filter(text)
        self.demod_changed.emit(text)

    def _on_fm_deviation_changed(self, _index: int) -> None:
        if not self._applying_filter and self._demod_combo.currentText() == "FM":
            self._remember_filter("deviation", self._fm_dev_combo.currentText())

    def _update_mode_controls(self) -> None:
        """Controls that only apply to some modes."""
        mode = self._demod_combo.currentText()
        self._demod_form.setRowVisible(self._cw_pitch_spin, mode == "CW")
        agc_modes = ("AM", "USB", "LSB", "CW")
        self._audio_agc_combo.setEnabled(mode in agc_modes)
        if mode in agc_modes:
            self._audio_agc_combo.setToolTip(
                "How fast the audio level recovers after a loud signal "
                "(AM, SSB and CW)."
            )
        else:
            self._audio_agc_combo.setToolTip(
                "Only used in AM, USB, LSB and CW (FM audio is steady)."
            )
        is_fm = mode == "FM"
        self._tone_combo.setEnabled(is_fm)
        self._audio_form.setRowVisible(self._tone_heard, is_fm)
        self._tone_combo.setToolTip(
            "CTCSS tone squelch: only play stations that send this "
            "sub-audible tone (repeaters, PMR). Tone heard shows the one on "
            "the channel."
            if is_fm
            else "Only used in FM mode"
        )
        audible = not mode.startswith("None")
        self._nr_combo.setEnabled(audible)
        self._squelch_check.setEnabled(audible)
        self._update_squelch_controls()
        self._update_bandwidth_tip()

    def _update_fm_dev_enabled(self) -> None:
        is_fm = self._demod_combo.currentText() == "FM"
        self._fm_dev_combo.setEnabled(is_fm)
        if is_fm:
            self._fm_dev_combo.setToolTip(
                "Peak FM deviation of the station: it sets the audio level. "
                "75 kHz for broadcast FM, 5 kHz for narrowband voice. Too "
                "small a value distorts (the status bar warns)."
            )
        else:
            self._fm_dev_combo.setToolTip("Only used in FM mode")

    def get_demod_mode(self) -> str:
        """The selected demodulation mode (a :data:`DEMOD_MODES` name)."""
        return self._demod_combo.currentText()

    def get_fm_deviation_text(self) -> str:
        """The FM deviation choice as shown, e.g. ``"75 kHz"``."""
        return self._fm_dev_combo.currentText()

    def get_fm_deviation(self) -> float:
        """Return the selected FM deviation in Hz (default 25 kHz)."""
        text = self._fm_dev_combo.currentText()  # e.g. "25 kHz"
        try:
            return float(text.split()[0]) * 1e3
        except (ValueError, IndexError):
            return 25e3

    def _on_squelch_changed(self, value: int):
        """Handle squelch slider change: a threshold set by hand (Auto
        goes off)."""
        self._squelch_label.setText(f"{value} dBFS")
        if self._following_auto:
            return
        self._auto_off_by_hand()
        self.squelch_changed.emit(float(value))

    def _auto_off_by_hand(self) -> None:
        """A threshold set by hand ends Auto (the caller emits it)."""
        if self._squelch_auto.isChecked():
            self._manual_threshold = True
            try:
                self._squelch_auto.setChecked(False)  # -> squelch_auto_changed
            finally:
                self._manual_threshold = False

    def get_squelch_db(self) -> float:
        """Return the current squelch threshold in dB."""
        return float(self._squelch_slider.value())

    def set_squelch_db(self, db: float) -> None:
        """Set the squelch threshold in dB (a manual threshold: Auto goes
        off)."""
        value = int(round(db))
        low, high = SQUELCH_RANGE
        if self._squelch_slider.value() == min(max(value, low), high):
            # Same position: still a manual threshold from now on.
            self._squelch_label.setText(f"{self._squelch_slider.value()} dBFS")
            self._auto_off_by_hand()
            self.squelch_changed.emit(float(self._squelch_slider.value()))
            return
        self._squelch_slider.setValue(value)

    def show_auto_threshold(self, db: Optional[float]) -> None:
        """While Auto is on, show its threshold on the slider (no signal)."""
        if db is None or not self._squelch_auto.isChecked():
            return
        low, high = SQUELCH_RANGE
        value = int(round(min(max(db, low), high)))
        if value == self._squelch_slider.value():
            return
        self._following_auto = True
        try:
            self._squelch_slider.setValue(value)
        finally:
            self._following_auto = False

    def _on_squelch_enabled(self, enabled: bool) -> None:
        self._update_squelch_controls()
        self.squelch_enabled_changed.emit(bool(enabled))

    def _on_squelch_auto(self, auto: bool) -> None:
        self._update_squelch_controls()
        self.squelch_auto_changed.emit(bool(auto))
        if not auto and not self._manual_threshold:
            # Auto off keeps the threshold where Auto had it.
            self.squelch_changed.emit(float(self._squelch_slider.value()))

    def _update_squelch_controls(self) -> None:
        on = self._squelch_check.isChecked() and self._squelch_check.isEnabled()
        self._squelch_auto.setEnabled(on)
        self._squelch_slider.setEnabled(on)
        self._squelch_label.setEnabled(on)

    def is_squelch_enabled(self) -> bool:
        return bool(self._squelch_check.isChecked())

    def set_squelch_enabled(self, enabled: bool) -> None:
        self._squelch_check.setChecked(bool(enabled))

    def is_squelch_auto(self) -> bool:
        return bool(self._squelch_auto.isChecked())

    def set_squelch_auto(self, auto: bool) -> None:
        self._squelch_auto.setChecked(bool(auto))

    # ------------------------------------------------------------------
    # Audio options
    # ------------------------------------------------------------------

    def _on_tone_changed(self, _index: int) -> None:
        self.tone_squelch_changed.emit(self.tone_squelch_hz())

    def tone_squelch_hz(self) -> float:
        """The tone squelch's tone (Hz), 0 when it is off."""
        value = self._tone_combo.currentText()
        try:
            return float(value)
        except ValueError:
            return 0.0

    def set_tone_squelch_hz(self, freq: float) -> None:
        """Choose the tone squelch's tone (0 or an unknown tone: off)."""
        index = self._tone_combo.findText(f"{float(freq):.1f}") if freq else -1
        self._tone_combo.setCurrentIndex(max(index, 0))

    def set_detected_tone(self, freq: Optional[float], checked: bool = True) -> None:
        """Show the sub-audible tone heard on the channel (None: none;
        ``checked`` False: tones aren't looked for, at a broadcast FM
        deviation without a tone squelch)."""
        if not checked:
            text = "Not checked"
            tip = (
                "Tones aren't looked for at a broadcast FM deviation "
                "(broadcast stations send none). Choose 5 kHz (narrow voice) "
                "for a repeater or PMR channel to see the tone it sends."
            )
        elif freq is None:
            text = "None"
            tip = (
                "No sub-audible (CTCSS) tone heard on the channel. Repeaters "
                "and PMR radios often send one; set the tone squelch to it to "
                "hear only them."
            )
        else:
            text = ctcss_label(freq)
            tip = f"The station sends a {text} CTCSS tone"
        if self._tone_heard.text() != text:
            self._tone_heard.setText(text)
            self._tone_heard.setToolTip(tip)
            heard = checked and freq is not None
            set_tone(self._tone_heard, "success" if heard else "muted")

    def noise_reduction(self) -> str:
        return self._nr_combo.currentText()

    def set_noise_reduction(self, level: str) -> None:
        self._nr_combo.setCurrentText(level)

    def audio_agc_speed(self) -> str:
        return self._audio_agc_combo.currentText()

    def set_audio_agc_speed(self, speed: str) -> None:
        self._audio_agc_combo.setCurrentText(speed)

    def cw_pitch(self) -> int:
        return int(self._cw_pitch_spin.value())

    def set_cw_pitch(self, hz: float) -> None:
        self._cw_pitch_spin.setValue(int(round(hz)))

    def _on_iq_options_changed(self, _checked: bool = False) -> None:
        self.iq_options_changed.emit(
            self._dc_check.isChecked(), self._iq_check.isChecked()
        )

    def iq_options(self) -> Tuple[bool, bool]:
        """``(remove DC spike, balance I/Q)``."""
        return self._dc_check.isChecked(), self._iq_check.isChecked()

    def set_iq_options(self, remove_dc: bool, balance_iq: bool) -> None:
        self._dc_check.setChecked(bool(remove_dc))
        self._iq_check.setChecked(bool(balance_iq))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _pause_supported(self) -> bool:
        """True when something listens to :attr:`recording_paused`."""
        try:
            return self.receivers(self.recording_paused) > 0
        except (TypeError, RuntimeError):  # pragma: no cover - defensive
            return False

    def _update_record_controls(self, recording: bool) -> None:
        """Reflect the recording state on the Recording group (no signals)."""
        self._record_btn.setText(_STOP_TEXT if recording else _RECORD_TEXT)
        role = "danger" if recording else ""
        if (self._record_btn.property("role") or "") != role:
            set_role(self._record_btn, role or None)
        if not recording:
            self._armed = False
            self._waiting = False
        # The format only says how the take will be saved: it can change
        # while recording.
        self._format_combo.setToolTip(_FORMAT_TIP)

        can_pause = recording and self._pause_supported()
        if not recording or not can_pause:
            self._pause_btn.blockSignals(True)
            self._pause_btn.setChecked(False)
            self._pause_btn.blockSignals(False)
            self._pause_btn.setText(_PAUSE_TEXT)
        self._pause_btn.setEnabled(can_pause)
        if not recording:
            self._pause_btn.setToolTip("Start a recording to pause it")
        elif not can_pause:
            self._pause_btn.setToolTip("Pausing isn't available for this recording")
        else:
            self._pause_btn.setToolTip("Pause or resume capturing samples")

        self._update_record_status()
        if not recording:
            self._record_time.setText("00:00:00")

    def _update_record_status(self) -> None:
        recording = self._record_btn.isChecked()
        paused = recording and self._pause_btn.isChecked()
        armed = recording and self._armed and not paused
        waiting = armed and getattr(self, "_waiting", False)
        tip = ""
        if paused:
            text, tone = _STATUS_PAUSED, "warning"
        elif waiting:
            text, tone, tip = _STATUS_WAITING, "warning", _WAITING_TIP
        elif armed:
            text, tone, tip = _STATUS_ARMED, "warning", _ARMED_TIP
        elif recording:
            audio = self.record_kind() == "audio"
            text = _STATUS_RECORDING_AUDIO if audio else _STATUS_RECORDING
            tone = "danger"
        else:
            text, tone = _STATUS_READY, None
        self._record_status.setText(text)
        self._record_status.setToolTip(tip)
        set_tone(self._record_status, tone)
        # The timer only runs while samples are captured; idle or armed it is
        # muted (the LCD's default amber read as a warning).
        set_tone(self._record_time, "muted" if armed else tone or "muted")

    def _on_record_toggled(self, checked: bool):
        """Handle record button toggle."""
        self._update_record_controls(checked)
        if checked:
            self.recording_started.emit(self._format_combo.currentText())
        else:
            self.recording_stopped.emit()

    def _on_pause_toggled(self, paused: bool) -> None:
        self._pause_btn.setText(_RESUME_TEXT if paused else _PAUSE_TEXT)
        self._update_record_status()
        self.recording_paused.emit(bool(paused))

    def is_recording_paused(self) -> bool:
        """Whether the current recording is paused."""
        return self._record_btn.isChecked() and self._pause_btn.isChecked()

    def update_record_time(self, seconds: int):
        """Update recording time display."""
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        self._record_time.setText(f"{hours:02d}:{minutes:02d}:{secs:02d}")

    def set_recording_armed(self, armed: bool, waiting: bool = False) -> None:
        """Show whether the current recording is armed: started, but waiting
        for the receiver to run before any samples are captured. With
        ``waiting`` it has captured some already and resumes when the
        receiver starts again.

        Only matters while recording; stopping the recording clears it.
        """
        armed = bool(armed)
        waiting = armed and bool(waiting)
        if armed != self._armed or waiting != getattr(self, "_waiting", False):
            self._armed = armed
            self._waiting = waiting
            self._update_record_status()

    def is_recording_armed(self) -> bool:
        """Whether the panel shows the recording as armed (not capturing)."""
        return self._record_btn.isChecked() and self._armed

    def set_recording_state(self, recording: bool) -> None:
        """Reflect recording state on the panel without re-emitting signals.

        Lets the main window keep this panel's Record button in sync when
        recording is toggled from elsewhere (e.g. the toolbar action).
        """
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(bool(recording))
        self._record_btn.blockSignals(False)
        self._update_record_controls(bool(recording))

    def get_recording_format(self) -> str:
        """Return the currently selected recording format label."""
        return self._format_combo.currentText()

    def get_recording_extension(self) -> str:
        """The file extension of the selected recording format (".cf32")."""
        return str(self._format_combo.currentData() or "")

    # -- what is recorded, where, and the take -------------------------------

    def record_kind(self) -> str:
        """What the Record buttons record: ``"iq"`` or ``"audio"``."""
        return str(self._kind_combo.currentData() or "iq")

    def set_record_kind(self, kind: str) -> None:
        index = self._kind_combo.findData(kind)
        if index >= 0:
            self._kind_combo.setCurrentIndex(index)

    def _on_record_kind_changed(self, _index: int) -> None:
        kind = self.record_kind()
        audio = kind == "audio"
        # Audio is always 48 kHz WAV, saved in the folder as it records.
        self._recording_form.setRowVisible(self._format_combo, not audio)
        self._auto_save_check.setVisible(not audio)
        self._record_btn.setToolTip(
            "Start or stop recording what you hear (the demodulated audio) to "
            "a WAV file in the recordings folder."
            if audio
            else "Start or stop capturing I/Q samples. Save them afterwards with "
            "Save... below or File > Save Recording."
        )
        self.record_kind_changed.emit(kind)

    def set_recordings_folder(self, path: str) -> None:
        """Show where audio recordings and auto-saved takes go."""
        self._folder_path = str(path or "")
        name = os.path.basename(self._folder_path.rstrip("/\\")) or "Choose..."
        self._folder_btn.setText(name)
        self._folder_btn.setToolTip(
            f"{self._folder_path}\nClick to choose another folder for audio "
            "recordings and automatically saved I/Q takes."
            if self._folder_path
            else "Choose the folder for audio recordings and auto-saved takes."
        )

    def recordings_folder(self) -> str:
        return self._folder_path

    def is_auto_save(self) -> bool:
        return self._auto_save_check.isChecked()

    def set_auto_save(self, on: bool) -> None:
        self._auto_save_check.setChecked(bool(on))

    def show_take(self, text: str, state: Optional[str], tip: str = "") -> None:
        """The take under the Record buttons: ``state`` "unsaved" (Save...
        and Discard), "loaded" (a converted file: Save As... and Unload),
        "saved" (Open Folder) or None (nothing shown)."""
        self._take_box.setVisible(state is not None)
        if state is None:
            return
        # A long file name can't wrap: it is shortened in the middle (the
        # tooltip has all of it), so it never widens the panel.
        words = [_elide_middle(word, 34) for word in str(text).split(" ")]
        self._take_label.setText(" ".join(words))
        self._take_label.setToolTip(tip or text)
        set_tone(self._take_label, "warning" if state == "unsaved" else None)
        keep = state in ("unsaved", "loaded")
        self._take_save_btn.setVisible(keep)
        self._take_discard_btn.setVisible(keep)
        self._take_folder_btn.setVisible(state == "saved")
        if state == "loaded":
            self._take_save_btn.setText("Save As...")
            self._take_save_btn.setToolTip(
                "Write the loaded recording in another format (File > Save "
                "Recording, Ctrl+S)"
            )
            self._take_discard_btn.setText("Unload")
            self._take_discard_btn.setToolTip(
                "Forget the loaded recording (the file itself stays on disk)"
            )
        else:
            self._take_save_btn.setText("Save...")
            self._take_save_btn.setToolTip(
                "Write the recording to a file (File > Save Recording, Ctrl+S)"
            )
            self._take_discard_btn.setText("Discard")
            self._take_discard_btn.setToolTip("Delete the recording without saving it")

    def take_text(self) -> str:
        """The take line shown ("" when none is)."""
        return self._take_label.text() if self._take_box.isVisibleTo(self) else ""

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _populate_preset_categories(self):
        """Populate the preset category dropdown."""
        fm = get_frequency_manager()
        categories = fm.get_preset_categories()
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItems(categories)
        self._category_combo.blockSignals(False)
        self._on_category_changed(self._category_combo.currentText())

    def _on_category_changed(self, category: str):
        """Handle category selection change."""
        presets = get_frequency_manager().get_rx_presets(category) if category else []
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for preset in presets:
            self._preset_combo.addItem(preset.name)
            self._preset_combo.setItemData(
                self._preset_combo.count() - 1,
                f"{_format_mhz(preset.frequency_hz)}: {preset.description}",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._preset_combo.blockSignals(False)
        self._on_preset_changed(self._preset_combo.currentText())

    def _on_preset_changed(self, preset_name: str):
        """Show the selected preset's details and whether TX is allowed."""
        fm = get_frequency_manager()
        preset = fm.get_preset_by_name(preset_name) if preset_name else None
        self._apply_preset_btn.setEnabled(preset is not None)
        self._apply_preset_btn.setToolTip(
            _APPLY_PRESET_TIP if preset is not None else "Choose a preset first"
        )
        if preset is None:
            self._preset_info.setText("Choose a category and a preset to see it here.")
            self._preset_info.setToolTip("")
            self._preset_tx.setText("")
            self._preset_tx.setToolTip("")
            set_tone(self._preset_tx, None)
            return

        mode = _PRESET_MODE_LABEL.get(preset.mode, preset.mode)
        # Show the Bandwidth option Apply Preset will select (the narrowest
        # that fits the channel), not the channel's nominal width: "15 kHz"
        # here while Apply set 25 kHz made the details look wrong. (The
        # choices depend on the preset's mode.)
        bandwidth = self._preset_bandwidth_text(
            preset.bandwidth_hz, self._preset_mode(preset)
        )
        self._preset_info.setText(
            f"{_format_mhz(preset.frequency_hz)} · {bandwidth} · {mode}\n"
            f"{preset.description}"
        )
        nominal = _format_bandwidth(preset.bandwidth_hz)
        tip = ""
        if abs(_parse_hz(bandwidth) - preset.bandwidth_hz) > 0.5:
            fits = _parse_hz(bandwidth) >= preset.bandwidth_hz
            tip = (
                f"The channel is {nominal} wide; Apply Preset sets Bandwidth "
                f"to {bandwidth}, the "
                + ("narrowest option that fits it." if fits else "widest option.")
            )
        self._preset_info.setToolTip(tip)

        with _quiet_frequency_manager():
            allowed, reason = fm.is_tx_allowed(
                preset.frequency_hz, preset.bandwidth_hz, preset.mode
            )
        reason = reason or ""
        # Receiving is never restricted, so every line starts by saying so:
        # the transmit status must not read as "this preset won't work".
        tip = reason
        if allowed:
            legal = fm.get_power_limit(preset.frequency_hz, preset.mode)
            effective = fm.get_effective_power_limit(preset.frequency_hz, preset.mode)
            text = f"{_LISTEN_OK} · Transmit ✓ with your license"
            if legal:
                text = f"{_LISTEN_OK} · Transmit ✓ (legal limit {legal:g} W)"
                tip = _power_limit_tip(legal, effective)
            tone = "success"
        elif reason.startswith("LICENSE:"):
            # Not covered by the selected license class: informational, since
            # listening works the same.
            text = self._license_block_text(
                reason[len("LICENSE:") :].strip(), preset.frequency_hz, preset.mode
            )
            tone = None
        else:
            # Hardware lockout (GPS, aviation, marine distress ...)
            detail = reason.replace("TX BLOCKED:", "").split("[")[0].strip()
            band = detail.split(" - ")[0].strip()
            text = (
                f"{_LISTEN_OK} · Transmit locked out: {band} is protected"
                if band
                else f"{_LISTEN_OK} · Transmit locked out: protected frequency"
            )
            tone = "warning"
        self._preset_tx.setText(text)
        self._preset_tx.setToolTip(tip)
        set_tone(self._preset_tx, tone)

    @staticmethod
    def _license_block_text(reason: str, freq_hz: float = 0.0, mode: str = "") -> str:
        """Short, plain-language version of a license restriction."""
        lowered = reason.lower()
        prefix = f"{_LISTEN_OK} · Transmit:"
        outside = f"{prefix} not allowed here (outside the amateur bands)"
        if "no amateur license" in lowered:
            # Only an amateur frequency becomes usable with a license; FM
            # broadcast, aviation or paging never do.
            if freq_hz and not _in_amateur_band(freq_hz):
                return outside
            return f"{prefix} needs an amateur license"
        if "higher license class" in lowered:
            return f"{prefix} needs a higher license class"
        if "not in any amateur" in lowered:
            return outside
        if lowered.startswith("mode"):
            return f"{prefix} {mode or 'this mode'} isn't allowed in this band segment"
        return f"{prefix} not allowed ({reason})" if reason else f"{prefix} not allowed"

    def _closest_bandwidth_index(self, bw_hz: float) -> int:
        """Index of the narrowest bandwidth option that fits ``bw_hz``."""
        values = [
            _parse_hz(self._bw_combo.itemText(i)) for i in range(self._bw_combo.count())
        ]
        if not values:
            return -1
        fitting = [i for i, v in enumerate(values) if v >= bw_hz]
        if fitting:
            return min(fitting, key=lambda i: values[i])
        return max(range(len(values)), key=lambda i: values[i])

    def _preset_bandwidth_text(self, bw_hz: float, mode: Optional[str] = None) -> str:
        """The Bandwidth option Apply Preset picks for ``bw_hz``, as shown:
        the narrowest of ``mode``'s choices (the current ones without a
        mode) that holds it, else the widest."""
        if mode is None:
            index = self._closest_bandwidth_index(bw_hz)
            if index < 0:
                return _format_bandwidth(bw_hz)
            return self._bw_combo.itemText(index)
        return closest_bandwidth(bandwidth_options(mode), bw_hz)

    @staticmethod
    def _preset_mode(preset) -> str:
        """The Mode list entry a preset listens with."""
        return _PRESET_DEMOD.get(preset.mode, preset.mode)

    def _apply_preset(self):
        """Apply the selected preset: frequency, mode, FM deviation and
        bandwidth."""
        preset_name = self._preset_combo.currentText()
        if not preset_name:
            return

        fm = get_frequency_manager()
        preset = fm.get_preset_by_name(preset_name)
        if not preset:
            return

        # Set frequency
        self._freq_input.set_frequency(preset.frequency_hz)
        self.frequency_changed.emit(self._freq_input.get_frequency())

        # Mode first: the Bandwidth choices depend on it.
        if preset.mode in _PRESET_DEMOD:
            idx = self._demod_combo.findText(_PRESET_DEMOD[preset.mode])
            if idx >= 0:
                self._demod_combo.setCurrentIndex(idx)

        # FM deviation that suits the channel width
        if preset.mode in ("FM", "WFM"):
            if preset.mode == "WFM" or preset.bandwidth_hz >= 150e3:
                self._fm_dev_combo.setCurrentText("75 kHz")
            elif preset.bandwidth_hz >= 40e3:
                self._fm_dev_combo.setCurrentText("25 kHz")
            else:
                self._fm_dev_combo.setCurrentText("5 kHz")

        # Bandwidth: the narrowest of the mode's options that fits the
        # preset (the old substring match picked 25 kHz for 2.7 kHz SSB
        # and missed 2 MHz).
        index = self._closest_bandwidth_index(preset.bandwidth_hz)
        if index >= 0:
            self._bw_combo.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # License profile
    # ------------------------------------------------------------------

    def _sync_license_combo(self) -> None:
        """Show the frequency manager's current class (no signal)."""
        current = get_frequency_manager().get_license_class()
        index = self._license_combo.findData(current)
        self._license_combo.blockSignals(True)
        self._license_combo.setCurrentIndex(max(index, 0))
        self._license_combo.blockSignals(False)
        self._update_license_info()

    def _selected_license(self) -> LicenseClass:
        data = self._license_combo.currentData()
        return data if isinstance(data, LicenseClass) else LicenseClass.NONE

    def _update_license_info(self) -> None:
        license_class = self._selected_license()
        summary = next((s for lc, _t, s in LICENSE_CLASSES if lc == license_class), "")
        self._license_info.setText(f"Transmit privileges: {summary}.")

    def _on_license_changed(self, index: int):
        """Handle license class change."""
        license_class = self._selected_license()

        # Update frequency manager
        get_frequency_manager().set_license_class(license_class)
        self._update_license_info()

        # Emit signal
        self.license_changed.emit(license_class)

        # Refresh preset TX status display
        self._on_preset_changed(self._preset_combo.currentText())

    def set_license_class(self, license_class: LicenseClass):
        """Set the license class (for external control)."""
        idx = max(self._license_combo.findData(license_class), 0)
        if idx != self._license_combo.currentIndex():
            self._license_combo.setCurrentIndex(idx)  # -> _on_license_changed
        elif get_frequency_manager().get_license_class() != self._selected_license():
            # The combo already shows it but the frequency manager was changed
            # elsewhere: bring it back in line with what the panel shows.
            self._on_license_changed(idx)

    def get_license_class(self) -> LicenseClass:
        """Get the current license class."""
        return get_frequency_manager().get_license_class()
