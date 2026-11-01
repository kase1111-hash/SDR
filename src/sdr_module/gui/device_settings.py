"""
What the connected receiver offers, for the window's controls.

Qt-free helpers shared by the control panel's Receiver group, the Connect
dialog and the main window:

- :func:`device_kind` names the kind of device ("demo", "rtlsdr", "hackrf",
  "playback" for a recording playing back, "other", or "" for none). The
  settings below are remembered per kind (``device/<kind>/...``).
- :class:`GainModel` / :func:`gain_model_for`: how the RF gain is set on a
  device. An RTL-SDR steps through its tuner's own gain table (29 steps
  from 0 to 49.6 dB on an R820T); a HackRF One has an LNA (0-40 dB in 8 dB
  steps), a VGA (0-62 dB in 2 dB steps) and a +14 dB RF amplifier, and no
  automatic gain; the demo device (and anything else) 0-50 dB in 1 dB steps.
- :class:`FrontEnd`: frequency correction (ppm), direct sampling and the
  bias tee, and which kinds have them (:func:`front_end_features`).
- per-kind memory of the sample rate, the gain and the front end
  (:func:`load_rate`, :func:`load_gain`, :func:`load_front_end`, ...).

``settings`` is anything with ``get_str``/``get_float``/``get_int``/
``get_bool`` and ``set`` (:class:`~.settings_store.GuiSettings`).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

#: Settings key prefix of everything remembered per device kind.
KEY_PREFIX = "device/"

#: What each device kind is called in messages.
KIND_NAMES: Dict[str, str] = {
    "demo": "the demo device",
    "rtlsdr": "the RTL-SDR",
    "hackrf": "the HackRF One",
}

# --------------------------------------------------------------------------- #
# Device kinds
# --------------------------------------------------------------------------- #


def device_kind(device: Any) -> str:
    """``"demo"``, ``"rtlsdr"``, ``"hackrf"``, ``"playback"`` (a recording
    playing back), ``"other"`` (any other receiver), or ``""`` for none."""
    if device is None:
        return ""
    try:
        from .device_dialog import MockDevice

        if isinstance(device, MockDevice):
            return "demo"
    except ImportError:  # pragma: no cover - the GUI needs it
        pass
    try:
        from .playback import FilePlaybackDevice

        if isinstance(device, FilePlaybackDevice):
            return "playback"
    except ImportError:  # pragma: no cover - the GUI needs it
        pass
    try:
        from ..devices.hackrf import HackRFDevice

        if isinstance(device, HackRFDevice):
            return "hackrf"
    except ImportError:  # pragma: no cover - defensive
        pass
    try:
        from ..devices.rtlsdr import RTLSDRDevice

        if isinstance(device, RTLSDRDevice):
            return "rtlsdr"
    except ImportError:  # pragma: no cover - defensive
        pass
    return "other"


# --------------------------------------------------------------------------- #
# Sample rates
# --------------------------------------------------------------------------- #

#: Sample rates offered (Device > Connect..., Receiver > Sample rate, the
#: status bar). Each device kind only offers the ones inside its range.
SAMPLE_RATES: Tuple[float, ...] = (
    1.0e6,
    1.4e6,
    1.8e6,
    2.0e6,
    2.4e6,
    2.56e6,
    3.2e6,
    4e6,
    8e6,
    10e6,
    20e6,
)
DEFAULT_RATE = 2.4e6
#: (min, max) sample rate per device kind, in samples/second.
RATE_LIMITS: Dict[str, Tuple[float, float]] = {
    "rtlsdr": (0.9e6, 2.56e6),
    "hackrf": (2.0e6, 20e6),
    "demo": (1.0e6, 3.2e6),
}
#: The ADS-B decoder's lowest usable rate (its 0.5 us pulses need 2 MS/s).
ADSB_MIN_RATE = 2.0e6
#: The rate it hears weak (distant) aircraft best from: at 2.0-2.2 MS/s
#: strong replies decode as well, replies 15 dB above the noise 85-95 %
#: as often (98 % from 2.4 MS/s).
ADSB_GOOD_RATE = 2.4e6
ADSB_RATE_TIP = (
    "ADS-B (1090 MHz) needs 2 MS/s or more (2.4 MS/s or more hears weak, "
    "distant aircraft best); the other decoders work at any rate"
)


def format_rate(rate: float) -> str:
    """``2400000.0`` -> ``"2.4 MS/s"``."""
    return f"{float(rate) / 1e6:g} MS/s"


def sample_rates_for(kind: str) -> Tuple[float, ...]:
    """The sample rates ``kind`` offers ("other" and unknown kinds: the
    demo device's)."""
    low, high = RATE_LIMITS.get(kind, RATE_LIMITS["demo"])
    rates = tuple(rate for rate in SAMPLE_RATES if low <= rate <= high)
    return rates or (DEFAULT_RATE,)


def rate_in_range(kind: str, rate: Any) -> bool:
    """Whether a ``kind`` device runs at ``rate``: inside its
    :data:`RATE_LIMITS` (a kind without limits: any finite positive
    rate). The demo device's limit matters too: it makes its signals in
    software, and much faster than 3.2 MS/s the window can't keep up."""
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(value) and value > 0):
        return False
    limits = RATE_LIMITS.get(kind)
    if limits is None:
        return True
    low, high = limits
    return low - 0.5 <= value <= high + 0.5


def rate_choice_label(rate: float) -> str:
    """``"2.4 MS/s"``, or ``"1.8 MS/s (too low for ADS-B)"``."""
    label = format_rate(rate)
    if float(rate) < ADSB_MIN_RATE:
        label += " (too low for ADS-B)"
    return label


# --------------------------------------------------------------------------- #
# Gain
# --------------------------------------------------------------------------- #

#: The demo device's (and any other receiver's) RF gain: 0-50 dB, 1 dB steps.
RANGE_STEPS: Tuple[float, ...] = tuple(float(db) for db in range(0, 51))
#: HackRF One receive gain stages.
HACKRF_LNA_STEPS: Tuple[int, ...] = tuple(range(0, 41, 8))
HACKRF_VGA_STEPS: Tuple[int, ...] = tuple(range(0, 63, 2))
HACKRF_AMP_DB = 14
#: What the HackRF One starts with (its driver's defaults).
HACKRF_DEFAULT_LNA = 16
HACKRF_DEFAULT_VGA = 20

NO_AGC_HACKRF = (
    "The HackRF One has no automatic gain: set its LNA and VGA gains by hand."
)
AGC_TIP = (
    "Tuner AGC: the tuner sets its RF gain itself and the RF gain slider is "
    "off. Not the audio level (that is Audio > Audio AGC)."
)


@dataclass(frozen=True)
class GainModel:
    """How the RF gain is set on a device.

    ``kind`` is ``"steps"`` (one slider over ``steps``, the dB values the
    device has: a tuner's gain table, or 0-50 dB in 1 dB steps) or
    ``"hackrf"`` (LNA, VGA and the RF amp). ``agc``: the hardware has
    automatic gain (``agc_tip`` says why not, when it hasn't).
    """

    kind: str = "steps"
    steps: Tuple[float, ...] = RANGE_STEPS
    agc: bool = True
    agc_tip: str = ""
    name: str = ""

    @property
    def low(self) -> float:
        return float(self.steps[0]) if self.steps else 0.0

    @property
    def high(self) -> float:
        if self.kind == "hackrf":
            return float(HACKRF_LNA_STEPS[-1] + HACKRF_VGA_STEPS[-1])
        return float(self.steps[-1]) if self.steps else 0.0

    def nearest_index(self, gain_db: float) -> int:
        """Index of the step nearest ``gain_db`` (the lower one on a tie)."""
        if not self.steps:
            return 0
        try:
            value = float(gain_db)
        except (TypeError, ValueError):
            value = self.low
        if not math.isfinite(value):
            value = self.low
        return min(range(len(self.steps)), key=lambda i: abs(self.steps[i] - value))

    def snap(self, gain_db: float) -> float:
        """The step nearest ``gain_db``."""
        if self.kind == "hackrf":
            lna, vga = split_hackrf_gain(gain_db)
            return float(lna + vga)
        return float(self.steps[self.nearest_index(gain_db)]) if self.steps else 0.0


#: The model with no device, for the demo device and for unknown receivers.
DEFAULT_GAIN_MODEL = GainModel()


def split_hackrf_gain(gain_db: float) -> Tuple[int, int]:
    """``(lna, vga)`` for a total RX gain, as the HackRF driver's
    ``set_gain`` splits it: the LNA first (8 dB steps up to 40), the rest
    on the VGA (2 dB steps up to 62)."""
    try:
        total = float(gain_db)
    except (TypeError, ValueError):
        total = 0.0
    if not math.isfinite(total):
        total = 0.0
    total = max(0.0, min(float(HACKRF_LNA_STEPS[-1] + HACKRF_VGA_STEPS[-1]), total))
    lna = max(v for v in HACKRF_LNA_STEPS if v <= total) if total < 40 else 40
    vga = int((total - lna) / 2) * 2
    return int(lna), int(max(0, min(HACKRF_VGA_STEPS[-1], vga)))


def gain_model_for_kind(kind: str) -> GainModel:
    """How the RF gain of a ``kind`` device is set, before one is open
    (an RTL-SDR: the R820T's gain table, the common tuner's; the open
    device's own table is :func:`gain_model_for`'s)."""
    if kind == "hackrf":
        return GainModel(
            kind="hackrf",
            steps=tuple(float(v) for v in HACKRF_LNA_STEPS),
            agc=False,
            agc_tip=NO_AGC_HACKRF,
            name="HackRF One",
        )
    if kind == "rtlsdr":
        from ..devices.rtlsdr import RTLSDRDevice

        table = tuple(sorted({float(g) for g in RTLSDRDevice.VALID_GAINS}))
        return GainModel(kind="steps", steps=table, agc=True, name="RTL-SDR")
    return DEFAULT_GAIN_MODEL


def gain_model_for(device: Any) -> GainModel:
    """How the RF gain of ``device`` is set (see :class:`GainModel`)."""
    kind = device_kind(device)
    if kind == "hackrf":
        return gain_model_for_kind(kind)
    if kind == "rtlsdr":
        table: Tuple[float, ...] = ()
        try:
            table = tuple(sorted({float(g) for g in device.get_tuner_gains()}))
        except Exception as e:  # pragma: no cover - driver-dependent
            logger.debug(f"No RTL-SDR gain table: {e}")
        if table:
            return GainModel(kind="steps", steps=table, agc=True, name="RTL-SDR")
    return DEFAULT_GAIN_MODEL


def format_db(gain_db: float) -> str:
    """``29.7 -> "29.7 dB"``, ``20.0 -> "20 dB"``."""
    return f"{float(gain_db):.1f}".rstrip("0").rstrip(".") + " dB"


def auto_gain_db(device: Any) -> Optional[float]:
    """The gain an automatic gain chose, when the device says (the demo
    device does; an RTL-SDR's tuner AGC doesn't report it): None if not
    known or not automatic."""
    if device is None or device_kind(device) in ("rtlsdr", "hackrf", "playback"):
        return None
    try:
        if str(getattr(device, "gain_mode", "")) != "auto":
            return None
        value = float(device.gain)
    except (TypeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Front end: frequency correction, direct sampling, bias tee
# --------------------------------------------------------------------------- #

#: ``(value, label, tip)`` of the direct sampling choices (RTL-SDR).
DIRECT_SAMPLING_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    (
        "q",
        "Q branch (RTL-SDR Blog V3)",
        "Below 24 MHz the antenna goes straight to the ADC's Q input: HF and "
        "AM broadcast on an RTL-SDR Blog V3.",
    ),
    (
        "i",
        "I branch",
        "Below 24 MHz the antenna goes straight to the ADC's I input: for "
        "dongles modified for HF on the I input.",
    ),
    (
        "off",
        "Off (V4 or upconverter)",
        "Never use direct sampling: an RTL-SDR Blog V4 (it has an HF "
        "upconverter built in) or an external upconverter tunes HF through "
        "the tuner.",
    ),
)
DIRECT_SAMPLING_VALUES = tuple(value for value, _label, _tip in DIRECT_SAMPLING_CHOICES)

DIRECT_SAMPLING_TIP = (
    "How HF is received. The tuner can't go below 24 MHz, so there the "
    "RTL-SDR can feed the antenna straight to its ADC (direct sampling) to "
    "hear HF and AM broadcast, less sensitively than an HF receiver. Above "
    "24 MHz the tuner is always used."
)
PPM_TIP = (
    "Frequency correction: corrects the dongle's crystal error, in parts per "
    "million (cheap dongles are often 20-100 ppm off; 50 ppm is 22 kHz at "
    "433 MHz). To calibrate, tune a station whose frequency you know "
    "exactly (a NOAA weather station or a local FM broadcaster) and change "
    "the value until it sits on the tuning marker: raise it if the station "
    "shows up left of the marker, lower it if right."
)
PPM_HINT = (
    "Calibrate on a station you know: it should sit on the tuning marker "
    "(left of it: raise the value)."
)
PPM_RANGE = (-200, 200)
BIAS_TEE_TEXT = "Bias tee (powers an LNA or active antenna)"
BIAS_TEE_TIPS: Dict[str, str] = {
    "rtlsdr": (
        "Puts 4.5 V on the antenna socket to power an LNA or an active "
        "antenna through the coax (RTL-SDR Blog V3 and later). Only turn it on "
        "with such a device connected: an antenna that is a DC short (a loop, "
        "some dipoles) or a powered splitter can be damaged, or damage the "
        "dongle."
    ),
    "hackrf": (
        "Puts 3.3 V (at most 50 mA) on the antenna port to power an LNA or an "
        "active antenna while receiving; it is off while transmitting. Only "
        "turn it on with such a device connected: an antenna that is a DC "
        "short can be damaged."
    ),
}
BIAS_TEE_WARNING = (
    "⚠ Bias tee on: the antenna socket is powered. Only with an LNA or an "
    "active antenna connected."
)
AMP_TEXT = f"RF amp (+{HACKRF_AMP_DB} dB)"
AMP_TIP = (
    f"The HackRF's front-end amplifier: +{HACKRF_AMP_DB} dB for weak signals. "
    "Strong signals (a nearby FM or TV transmitter) then overload the "
    "receiver, filling the span with false signals; never feed the HackRF "
    "more than -5 dBm, and leave the amp off near transmitters."
)
AMP_WARNING = (
    "⚠ RF amp on: strong signals can overload the HackRF (false signals "
    "across the span)."
)


@dataclass
class FrontEnd:
    """The front-end settings of one device kind."""

    ppm: int = 0
    direct_sampling: str = "q"
    bias_tee: bool = False


def front_end_features(kind: str) -> FrozenSet[str]:
    """Which front-end settings ``kind`` has: ``"ppm"``, ``"direct"``,
    ``"bias_tee"``."""
    if kind == "rtlsdr":
        return frozenset({"ppm", "direct", "bias_tee"})
    if kind == "hackrf":
        return frozenset({"bias_tee"})
    return frozenset()


def _key(kind: str, name: str) -> str:
    return f"{KEY_PREFIX}{kind}/{name}"


def load_front_end(settings: Any, kind: str) -> FrontEnd:
    """The front end last used with ``kind`` (defaults if none)."""
    front = FrontEnd()
    if not kind or settings is None:
        return front
    try:
        front.ppm = int(settings.get_int(_key(kind, "ppm"), 0))
    except (TypeError, ValueError):
        front.ppm = 0
    front.ppm = max(PPM_RANGE[0], min(PPM_RANGE[1], front.ppm))
    direct = str(settings.get_str(_key(kind, "direct_sampling"), "q") or "q")
    front.direct_sampling = direct if direct in DIRECT_SAMPLING_VALUES else "q"
    try:
        front.bias_tee = bool(settings.get_bool(_key(kind, "bias_tee"), False))
    except (TypeError, ValueError):
        front.bias_tee = False
    return front


def save_front_end(settings: Any, kind: str, front: FrontEnd) -> None:
    """Remember ``front`` for ``kind`` (only what ``kind`` has)."""
    if not kind or settings is None:
        return
    features = front_end_features(kind)
    if "ppm" in features:
        settings.set(_key(kind, "ppm"), int(front.ppm))
    if "direct" in features:
        settings.set(_key(kind, "direct_sampling"), str(front.direct_sampling))
    if "bias_tee" in features:
        settings.set(_key(kind, "bias_tee"), bool(front.bias_tee))


# --------------------------------------------------------------------------- #
# Sample rate and gain memory
# --------------------------------------------------------------------------- #


def load_rate(settings: Any, kind: str) -> float:
    """The sample rate last used with ``kind`` (0 if none, or if the one
    saved is one ``kind`` doesn't run at: a damaged setting)."""
    if not kind or settings is None:
        return 0.0
    try:
        rate = float(settings.get_float(_key(kind, "sample_rate"), 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not rate_in_range(kind, rate):
        if rate:
            logger.debug(
                f"Ignoring a saved sample rate out of range for {kind}: {rate}"
            )
        return 0.0
    return rate


def save_rate(settings: Any, kind: str, rate: float) -> None:
    """Remember ``rate`` for ``kind`` (only one ``kind`` runs at)."""
    if kind and settings is not None and rate_in_range(kind, rate):
        settings.set(_key(kind, "sample_rate"), float(rate))


def load_gain(settings: Any, kind: str) -> Optional[Dict[str, Any]]:
    """The gain last used with ``kind``: ``{"db", "agc"}`` or, for a
    HackRF, ``{"lna", "vga", "amp"}``; None if nothing is saved."""
    if not kind or settings is None:
        return None
    raw = settings.get_str(_key(kind, "gain"), "")
    if not raw:
        return None
    try:
        saved = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring an unreadable saved gain for {kind}: {raw!r}")
        return None
    return saved if isinstance(saved, dict) else None


def save_gain(settings: Any, kind: str, gain: Dict[str, Any]) -> None:
    if kind and settings is not None:
        settings.set(_key(kind, "gain"), json.dumps(gain))


__all__ = [
    "AGC_TIP",
    "AMP_TEXT",
    "AMP_TIP",
    "AMP_WARNING",
    "BIAS_TEE_TEXT",
    "BIAS_TEE_TIPS",
    "BIAS_TEE_WARNING",
    "DEFAULT_GAIN_MODEL",
    "DIRECT_SAMPLING_CHOICES",
    "DIRECT_SAMPLING_TIP",
    "FrontEnd",
    "GainModel",
    "HACKRF_LNA_STEPS",
    "HACKRF_VGA_STEPS",
    "KIND_NAMES",
    "NO_AGC_HACKRF",
    "PPM_HINT",
    "PPM_RANGE",
    "PPM_TIP",
    "auto_gain_db",
    "device_kind",
    "format_db",
    "front_end_features",
    "gain_model_for",
    "gain_model_for_kind",
    "load_front_end",
    "load_gain",
    "load_rate",
    "rate_in_range",
    "save_front_end",
    "save_gain",
    "save_rate",
    "split_hackrf_gain",
]
