"""
Identify the signal at the tuned frequency: "what is this, and how do I
listen to it?"

:class:`SignalIdentifier` collects the receiver's blocks around the tuned
frequency (the VFO's offset from the LO, mixed to 0 Hz, decimated only
above about 3 MS/s), keeping the last second or so. Every half second
:meth:`SignalIdentifier.take` hands out what it holds; :func:`identify`
classifies that (:class:`~sdr_module.dsp.classifiers.SignalClassifier`)
and says how to listen to it (:func:`listening_for`): the mode, the FM
deviation and the filter width. A signal that comes and goes (a burst, an
intermittent transmission) is waited for, up to a few seconds.

On/off keyed data (433/868/915 MHz devices) also gets an rtl_433-style
pulse analysis (:mod:`sdr_module.dsp.bursts`): pulse and gap widths, the
modulation guess and the bits.

No Qt; the classification of a capture takes up to about 0.2 s, so the
window runs :func:`identify` off its GUI thread.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

from .channelizer import Channelizer
from .classifiers import (
    KIND_LABELS,
    MAX_CONFIDENCE,
    ClassificationResult,
    SignalClassifier,
)
from .nco import Nco

__all__ = [
    "Identification",
    "SignalIdentifier",
    "analysis_band",
    "identify",
    "listening_for",
    "confidence_words",
    "takes_needed",
    "CAPTURE_MAX_RATE",
]

#: Above this sample rate the capture is decimated (to about 2.4 MS/s).
CAPTURE_MAX_RATE = 3.2e6
#: How much the identifier keeps (seconds) and the most samples it holds.
KEEP_S = 1.0
KEEP_MAX_SAMPLES = 1 << 21
#: How long a capture is before it is judged, and how long a signal that
#: isn't there yet is waited for.
MIN_S = 0.5
WAIT_S = 4.0

# FM deviation choices (Hz), as Demodulation > FM deviation offers them.
# A station swinging beyond the deviation set clips its audio (the window
# warns "FM audio is clipping"), so the one suggested holds the measured
# peak swing with _DEVIATION_MARGIN to spare (the peak reads up to about
# 10 % low on tones). The measured rms swing times about 2.2 (voice; a
# tone is 1.4) estimates the peak when it wasn't measured.
_DEVIATIONS: Tuple[Tuple[float, str], ...] = (
    (2.5e3, "2.5 kHz"),
    (5e3, "5 kHz"),
    (12.5e3, "12.5 kHz"),
    (25e3, "25 kHz"),
    (75e3, "75 kHz"),
)
_DEVIATION_MARGIN = 1.1
_PEAK_PER_RMS = 2.2
# A peak swing this wide is a broadcast station's.
_BROADCAST_PEAK_HZ = 40e3


def _deviation_for(peak_hz: float) -> str:
    """The smallest FM deviation choice that holds a peak swing of
    ``peak_hz`` (with :data:`_DEVIATION_MARGIN` to spare), else the
    widest: the audio doesn't clip."""
    need = _DEVIATION_MARGIN * float(peak_hz)
    return next((text for hz, text in _DEVIATIONS if need <= hz), _DEVIATIONS[-1][1])


@dataclass
class Identification:
    """What the signal at the tuned frequency is, and how to listen to it."""

    result: ClassificationResult
    tuned_hz: float  # the frequency it was identified at
    label: str  # "Narrow FM (voice)", "AM", "OOK data bursts", ...
    mode: str = ""  # Demodulation > Mode to listen with ("" = no change)
    fm_deviation: str = ""  # FM deviation choice ("5 kHz"), FM only
    filter_hz: float = 0.0  # channel / filter width to listen with
    details: List[str] = field(default_factory=list)
    #: rtl_433-style pulse analysis of on/off keyed data (or None).
    pulses: Any = None
    #: Seconds of signal judged.
    seconds: float = 0.0

    @property
    def kind(self) -> str:
        return self.result.kind

    @property
    def confidence(self) -> float:
        return self.result.confidence

    @property
    def signal_hz(self) -> float:
        """Where the signal's power is centred (absolute frequency)."""
        return self.tuned_hz + self.result.center_offset_hz

    @property
    def is_signal(self) -> bool:
        return self.result.kind not in ("noise", "unknown")

    def summary(self) -> str:
        """One line: "Narrow FM (voice) · probably · SNR 34 dB"."""
        words = confidence_words(self.confidence)
        if not self.is_signal:
            return f"{self.label} · {words}"
        return f"{self.label} · {words} · SNR {self.result.snr_db:.0f} dB"


def confidence_words(confidence: float) -> str:
    """How sure, in words: "confident", "probably", "maybe", "can't tell"."""
    if confidence >= 0.75:
        return "confident"
    if confidence >= 0.5:
        return "probably"
    if confidence >= 0.3:
        return "maybe"
    return "can't tell"


def _format_hz(hz: float) -> str:
    hz = abs(float(hz))
    if hz >= 1e6:
        return f"{hz / 1e6:.2f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def listening_for(result: ClassificationResult, freq_hz: float) -> Identification:
    """How to listen to what ``result`` found at ``freq_hz``: the label,
    the mode, FM deviation and filter width, and plain-English details."""
    kind = result.kind
    f = result.features
    width = float(result.bandwidth_hz)
    label = KIND_LABELS.get(kind, KIND_LABELS["unknown"])
    mode, deviation, filter_hz = "", "", 0.0
    details: List[str] = []
    if kind == "fm":
        measured = float(f.get("freq_peak_hz", 0.0))
        peak = measured or _PEAK_PER_RMS * float(f.get("freq_dev_hz", 0.0))
        if measured > 0:
            # (The classifier's "swings about" is the rms swing; the
            # deviation suggested holds the peak.)
            details.append(f"The swing peaks at about ±{_format_hz(measured)}")
        deviation = _deviation_for(peak)
        plan = _band_listening(freq_hz)
        planned = _khz(plan[1]) if plan and plan[0] == "FM" else 0.0
        agrees = bool(planned) and 0.25 * planned <= peak
        agrees = agrees and _DEVIATION_MARGIN * peak <= planned
        if agrees:
            # What the band uses, when the swing measured agrees (and the
            # band's deviation holds it).
            deviation = plan[1] if plan else deviation
        broadcast = peak >= _BROADCAST_PEAK_HZ or (agrees and planned >= 75e3)
        if width >= 100e3 or broadcast:
            label, deviation, filter_hz = "Broadcast FM", "75 kHz", 200e3
        elif width >= 30e3 or _khz(deviation) >= 25e3:
            label, filter_hz = "Wide FM", max(1.2 * width, 50e3)
        else:
            label = "Narrow FM (voice or data)"
            channel = 25e3 if deviation == "5 kHz" else 12.5e3
            filter_hz = max(1.25 * width, channel)
        mode = "FM"
    elif kind == "am":
        mode, filter_hz = "AM", min(max(1.1 * width, 5e3), 15e3)
    elif kind == "ssb":
        mode = "USB" if freq_hz >= 10e6 else "LSB"
        filter_hz = 2.7e3
        details.append(
            f"{mode}: the usual sideband {'above' if mode == 'USB' else 'below'} "
            "10 MHz; tune so the voice sounds natural"
        )
    elif kind in ("cw", "carrier"):
        mode, filter_hz = "CW", 500.0
    elif kind == "fsk":
        peak = 1.3 * float(f.get("freq_dev_hz", 0.0))
        deviation = _deviation_for(peak)
        mode, filter_hz = "FM", max(1.2 * width, 12.5e3)
        details.append("Data: the Decoder tab decodes POCSAG, FLEX and similar")
    elif kind == "ook" and _mode_s_band(freq_hz):
        # 1090 MHz carries radar replies only. Through a filter narrower
        # than their 0.5 us pulses, each 64-120 us reply is one burst.
        label = "Mode S / ADS-B replies"
        mode, filter_hz = "None (I/Q)", 2e6
        details.append(
            "Each reply shows as one burst through this filter: Apply Mode "
            "widens it to 2 MHz, where its pulses show"
        )
        details.append("Decode them with the Decoder tab's ADS-B")
    elif kind == "ook":
        if width <= 14e3:
            mode, filter_hz = "AM", max(1.1 * width, 5e3)
        else:
            mode, filter_hz = "None (I/Q)", max(1.2 * width, 25e3)
        details.append("On/off keyed data, as remote controls and sensors send")
    elif kind == "pulsed":
        mode, filter_hz = "None (I/Q)", max(1.2 * width, 2e6)
        if _mode_s_band(freq_hz):
            label = "Mode S / ADS-B replies"
            details.append("Decode them with the Decoder tab's ADS-B")
    elif kind == "psk":
        if width < 3e3:
            mode, filter_hz = "USB", 2.7e3
        else:
            mode, filter_hz = "None (I/Q)", max(1.2 * width, 10e3)
    ident = Identification(
        result=result,
        tuned_hz=float(freq_hz),
        label=label,
        mode=mode,
        fm_deviation=deviation if mode == "FM" else "",
        filter_hz=float(filter_hz),
        details=details,
    )
    ident.details = _details(ident) + details
    return ident


def _mode_s_band(freq_hz: float) -> bool:
    """Whether ``freq_hz`` is on 1090 MHz (secondary radar replies)."""
    return abs(float(freq_hz) - 1090e6) < 2e6


def _band_listening(freq_hz: float) -> Optional[Tuple[str, str, str]]:
    """The band plan's ``(mode, FM deviation, filter)`` at ``freq_hz``
    (None outside the bands it knows)."""
    try:
        from ..core.band_plan import band_for, listening_for

        if band_for(freq_hz) is None:
            return None
        return listening_for(freq_hz)
    except Exception:  # pragma: no cover - the plan is a hint only
        return None


def _khz(text: str) -> float:
    """ "2.5 kHz" -> 2500.0 (0.0 when it isn't a width)."""
    try:
        value, unit = str(text).split()
        return float(value) * {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
    except (ValueError, KeyError):
        return 0.0


def _details(ident: Identification) -> List[str]:
    """Plain-English lines about the result (shown under the label)."""
    r = ident.result
    lines: List[str] = []
    if r.kind == "noise":
        lines.extend(r.evidence)
        return lines
    if r.bandwidth_hz > 0:
        lines.append(f"Occupies about {_format_hz(r.bandwidth_hz)}")
    offset = r.center_offset_hz
    if r.bandwidth_hz > 0 and abs(offset) >= max(0.1 * r.bandwidth_hz, 50.0):
        where = "above" if offset > 0 else "below"
        lines.append(f"Centred {_format_hz(offset)} {where} the tuned frequency")
    lines.extend(e for e in r.evidence if not e.startswith("Occupies"))
    return lines


def analysis_band(
    mode: str, channel_hz: float, rate: float, freq_hz: Optional[float] = None
) -> Tuple[float, float]:
    """``(analysis width, search reach)`` in Hz for identifying the signal
    at the tuned frequency while listening in ``mode`` with a channel (or
    filter) ``channel_hz`` wide: a band a few channels wide around it (the
    whole span for a wide I/Q channel, to see pulses as they are), and how
    far from the tuned frequency the signal may reach and still count.
    On 1090 MHz (``freq_hz``) at least 2 MHz whatever the mode: the radar
    replies' 0.5 us pulses show only that wide."""
    channel = max(float(channel_hz), 100.0)
    if freq_hz is not None and _mode_s_band(freq_hz):
        channel = max(channel, 2e6)
        mode = "None (I/Q)"
    if mode.startswith("None"):
        width = min(float(rate), max(channel, 100e3))
        reach = min(max(0.5 * channel, 5e3), 25e3)
    else:
        width = min(float(rate), min(max(4.0 * channel, 100e3), 800e3))
        reach = max(0.75 * channel, 3e3)
    return width, reach


def identify(
    samples: np.ndarray,
    rate: float,
    tuned_hz: float,
    search_hz: Optional[float] = None,
    noise_psd: Optional[float] = None,
    band_hz: Optional[Tuple[float, float]] = None,
) -> Identification:
    """Identify the signal at 0 Hz of ``samples`` (complex baseband at
    ``rate``, the tuned frequency ``tuned_hz`` at 0 Hz).

    ``search_hz``: how far from the tuned frequency the signal may reach
    and still count (default: 5 % of the rate, at least 12.5 kHz);
    ``band_hz``: the flat part of the capture's band (e.g. a
    :class:`SignalIdentifier`'s ``passband_hz``), where the noise is
    measured."""
    rate = float(rate)
    if search_hz is None:
        search_hz = max(12.5e3, 0.05 * rate)
    result = SignalClassifier(rate).classify(
        samples, noise_psd=noise_psd, band_hz=band_hz, search_hz=search_hz
    )
    half = 4.0 * float(search_hz)
    far = abs(result.center_offset_hz) > max(search_hz, 0.6 * result.bandwidth_hz) or (
        result.bandwidth_hz > 2.0 * half
    )
    if result.kind != "noise" and far and half < 0.4 * rate:
        # Centred well away: a stronger signal next door joined to the
        # tuned one (by its splatter). Look again, without it.
        from .classifiers import _narrow

        near, near_rate = _narrow(np.asarray(samples), rate, 0.0, half)
        result = SignalClassifier(near_rate).classify(
            near,
            noise_psd=noise_psd,
            band_hz=(-0.95 * half, 0.95 * half),
            search_hz=search_hz,
        )
        rate = near_rate
        samples = near
    ident = listening_for(result, tuned_hz)
    ident.seconds = len(samples) / rate if rate > 0 else 0.0
    if result.kind == "ook" and not _mode_s_band(tuned_hz):
        # (Not radar replies: their bits aren't a remote's.)
        try:
            from .bursts import analyze_capture

            ident.pulses = analyze_capture(samples, rate, result)
        except Exception:  # pragma: no cover - the analysis is a bonus
            ident.pulses = None
    return ident


class SignalIdentifier:
    """Collects about a second of the receiver's samples around the tuned
    frequency for :func:`identify`.

    Feed it every raw block (:meth:`feed`); after :data:`MIN_S` seconds
    :meth:`take` returns the capture (and again every half second while
    the caller keeps listening for a signal that hasn't shown up yet).
    The capture is the channel at the VFO mixed to 0 Hz at the device's
    rate (decimated to about 2.4 MS/s when the device is faster).
    """

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        analysis_hz: Optional[float] = None,
    ):
        """``analysis_hz``: how wide a band around the tuned frequency is
        kept (default: the whole span, at most about 2.4 MHz); see
        :func:`analysis_band`."""
        self.rate_in = float(sample_rate)
        self.offset_hz = float(offset_hz)
        wanted = float(analysis_hz) if analysis_hz else self.rate_in
        wanted = min(wanted, self.rate_in, 2.4e6)
        self.analysis_hz = wanted
        if wanted < 0.75 * self.rate_in:
            half = 0.5 * wanted
            self._channel: Optional[Channelizer] = Channelizer(
                self.rate_in,
                self.offset_hz,
                pass_hz=half,
                stop_hz=1.2 * half,
                target_rate=1.25 * wanted,
            )
            self.rate = self._channel.output_rate
            self._nco: Optional[Nco] = None
            #: The flat part of the capture's band (around 0 Hz).
            self.passband_hz = (-0.95 * half, 0.95 * half)
        else:
            self._channel = None
            self._nco = Nco(self.rate_in, self.offset_hz)
            self.rate = self.rate_in
            # (A receiver's own filter rolls off near the span's edges.)
            self.passband_hz = (-0.4 * self.rate, 0.4 * self.rate)
        self._keep = int(min(KEEP_MAX_SAMPLES, KEEP_S * self.rate))
        if self.rate > CAPTURE_MAX_RATE / 2:
            # The full span: 0.3 s (the analysis costs more).
            self._keep = int(min(self._keep, 0.3 * self.rate))
        self._blocks: Deque[np.ndarray] = deque()
        self._held = 0
        self.received = 0  # samples (at ``rate``) since the start
        self._next_take = int(MIN_S * self.rate)

    @property
    def seconds(self) -> float:
        """Seconds listened so far."""
        return self.received / self.rate if self.rate > 0 else 0.0

    def set_offset(self, offset_hz: float) -> None:
        """The VFO moved (inside the span): follow it, starting afresh."""
        if float(offset_hz) == self.offset_hz:
            return
        self.offset_hz = float(offset_hz)
        if self._channel is not None:
            self._channel.set_offset(offset_hz)
        if self._nco is not None:
            self._nco.frequency_hz = offset_hz
        self.restart()

    def restart(self) -> None:
        """Forget what was collected (and listen from now)."""
        self._blocks.clear()
        self._held = 0
        self.received = 0
        self._next_take = int(MIN_S * self.rate)

    def gap(self) -> None:
        """Samples were lost before the next block (the receiver dropped
        some, or skipped ahead): what is held is no longer one stretch of
        signal, so it is dropped and the next capture is due once
        :data:`MIN_S` of unbroken signal has come (how long it has listened
        in all still counts, for the wait)."""
        self._blocks.clear()
        self._held = 0
        self._next_take = self.received + int(MIN_S * self.rate)

    def feed(self, block: np.ndarray) -> None:
        """Add a raw block from the receiver (at ``rate_in``)."""
        x = np.asarray(block)
        if not len(x):
            return
        if self._channel is not None:
            y = self._channel.process(x)
        else:
            assert self._nco is not None
            y = self._nco.mix(x.astype(np.complex64, copy=False))
        if not len(y):
            return
        y = np.asarray(y, dtype=np.complex64)
        if len(y) > self._keep:
            y = y[-self._keep :]
        self._blocks.append(y.copy())
        self._held += len(y)
        self.received += len(y)
        while self._blocks and self._held - len(self._blocks[0]) >= self._keep:
            self._held -= len(self._blocks.popleft())

    def ready(self) -> bool:
        """A capture is due (half a second in, then every half second)."""
        return self.received >= self._next_take

    def take(self) -> np.ndarray:
        """The capture (the last second or so); the next one is due half a
        second later."""
        self._next_take = self.received + int(0.5 * self.rate)
        if not self._blocks:
            return np.zeros(0, dtype=np.complex64)
        data = np.concatenate(list(self._blocks))
        return data[-self._keep :]


def better(old: Optional[Identification], new: Identification) -> Identification:
    """Of two identifications of the same frequency, the more telling one:
    a signal over no signal; keying seen over what it looks like between
    or across its keys (CW over the steady carrier of a key held down, on/off
    keyed data over the short pulses a broken capture of it shows); else
    the more confident."""
    if old is None:
        return new
    if new.is_signal != old.is_signal:
        return new if new.is_signal else old
    for keyed, plain in (("cw", "carrier"), ("ook", "pulsed")):
        if {old.kind, new.kind} == {keyed, plain}:
            return old if old.kind == keyed else new
    return new if new.confidence >= old.confidence else old


def takes_needed(ident: Identification) -> int:
    """How many takes (half a second apart) Identify hears before it may
    stop on ``ident``: three for a steady carrier (a CW station holds its
    key down between its words: keying may still show), two for short
    pulses away from 1090 MHz (on/off keyed data broken by a gap in the
    stream looks like them), else one."""
    if ident.kind == "carrier":
        return 3
    if ident.kind == "pulsed" and not _mode_s_band(ident.tuned_hz):
        return 2
    return 1


# Re-exported for the window's wording of the cap.
MAX_IDENTIFY_CONFIDENCE = MAX_CONFIDENCE
