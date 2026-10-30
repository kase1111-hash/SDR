"""
Live protocol decoders: raw receiver I/Q in, decoded messages out.

The window hands the decoder every block it reads from the device: the
whole span (up to 20 MS/s) centred on the LO. A :class:`LiveDecoder` picks
out its own channel around the VFO (tuned frequency minus LO), demodulates
it the way its protocol is sent and runs the protocol decoder on the
result, keeping all of its state between calls: decoding a stream in 33 ms
blocks gives the same messages as decoding it in one go.

==========  =============================================================
Decoder     Chain
==========  =============================================================
POCSAG      channel (+/-8 kHz, ~48 kHz) -> FM discriminator -> DC removal
            -> 512, 1200 and 2400 baud slicers -> POCSAGDecoder
FLEX        as POCSAG -> 1600 baud slicer -> FLEXDecoder
AX.25/APRS  channel -> FM discriminator (audio) -> 1200/2200 Hz mark and
            space detectors over one bit -> slicer -> NRZI/HDLC -> AX.25
ACARS       channel (+/-5 kHz) -> AM envelope -> 1800 Hz complex mix and
            low-pass -> coherent MSK (rotation by 600 Hz, phase from a
            squaring loop) -> slicer -> ACARSDecoder
ADS-B       |I/Q| at the full device rate (at least 2 MS/s) -> ADSBDecoder
==========  =============================================================

Each decoder reports its health in :attr:`LiveDecoder.status` (frames
found, valid and failed, when the last message came, whether a signal is
in the channel and any problem, such as a sample rate too low for ADS-B),
which the Decoder panel shows.

To add a protocol, subclass :class:`LiveDecoder` (or :class:`ChannelDecoder`
for a narrow channel) and register the class in :data:`LIVE_DECODERS` under
the name the Decoder panel lists. NumPy only; no Qt.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type

import numpy as np

from .channelizer import (
    Channelizer,
    Delay,
    FirDecimator,
    FMDiscriminator,
    MovingAverage,
    Nco,
    lowpass_taps,
)
from .protocols import (
    ACARSDecoder,
    ADSBDecoder,
    AX25Decoder,
    DecodedMessage,
    FLEXDecoder,
    POCSAGDecoder,
    ProtocolDecoder,
)

__all__ = [
    "LiveDecoderStatus",
    "LiveDecoder",
    "ChannelDecoder",
    "POCSAGLiveDecoder",
    "FLEXLiveDecoder",
    "AX25LiveDecoder",
    "ACARSLiveDecoder",
    "ADSBLiveDecoder",
    "LIVE_DECODERS",
    "create_live_decoder",
    "describe_message",
    "format_rate",
]


def format_rate(rate_hz: float) -> str:
    """``2.4e6 -> "2.4 MS/s"``, ``48e3 -> "48 kS/s"``."""
    rate_hz = float(rate_hz)
    if rate_hz >= 1e6:
        return f"{rate_hz / 1e6:.3f}".rstrip("0").rstrip(".") + " MS/s"
    return f"{rate_hz / 1e3:.1f}".rstrip("0").rstrip(".") + " kS/s"


@dataclass
class LiveDecoderStatus:
    """What a live decoder has done so far (a snapshot)."""

    name: str = ""
    #: Noun for one decoded item: "page", "packet", "message".
    unit: str = "message"
    #: Frames (transmissions, packets, blocks) found in the signal.
    frames: int = 0
    #: Messages decoded that passed their checks.
    valid: int = 0
    #: Frames or messages that failed their checks (CRC, BCH, parity).
    failed: int = 0
    #: ``clock()`` time of the last valid message (None: none yet).
    last_time: Optional[float] = None
    #: A signal is in the decoder's channel now (or was a moment ago).
    signal_present: bool = False
    #: Whether the decoder measures ``signal_present`` at all.
    signal_measured: bool = False
    #: Seconds of signal heard in the channel since the last valid message.
    undecoded_signal_s: float = 0.0
    #: Why the decoder cannot work (shown instead of the counts), or None.
    problem: Optional[str] = None
    #: Rate of the samples given to :meth:`LiveDecoder.process`.
    sample_rate: float = 0.0
    #: Rate the channel is decoded at (the full rate for ADS-B).
    channel_rate: float = 0.0
    #: Where the channel is, relative to the centre of the samples.
    offset_hz: float = 0.0
    #: Seconds of samples processed.
    stream_s: float = 0.0


class LiveDecoder:
    """Base class: one protocol, fed raw I/Q blocks of any size.

    Subclasses implement :meth:`_decode` (and usually :meth:`_decoders`,
    the protocol decoders whose ``frames_seen``/``frames_failed`` counters
    feed the status).
    """

    #: Selector name, e.g. "POCSAG".
    name: str = ""
    #: Noun for one decoded item.
    unit: str = "message"
    #: Lowest usable sample rate (Hz); below it ``create_live_decoder``
    #: raises ValueError (ADS-B reports it as a problem instead).
    MIN_SAMPLE_RATE: float = 0.0

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        sample_rate = float(sample_rate)
        if not sample_rate > 0:
            raise ValueError(f"sample rate must be positive, got {sample_rate:g}")
        self.sample_rate = sample_rate
        self._offset = float(offset_hz)
        self._clock = clock
        self._valid = 0
        self._last_time: Optional[float] = None
        self._stream_s = 0.0
        self._problem: Optional[str] = None

    # -- API ----------------------------------------------------------------

    @property
    def offset_hz(self) -> float:
        return self._offset

    def set_offset(self, offset_hz: float) -> None:
        """Follow the VFO: the channel is now ``offset_hz`` from the centre."""
        self._offset = float(offset_hz)

    def process(self, iq: np.ndarray) -> List[DecodedMessage]:
        """Decode a block of raw I/Q (complex, any length); returns the
        messages that completed in it (valid ones and ones that failed
        their checks, ``valid`` False)."""
        iq = np.asarray(iq)
        if iq.ndim != 1:
            iq = iq.ravel()
        if not len(iq):
            return []
        self._stream_s += len(iq) / self.sample_rate
        messages = list(self._decode(iq))
        valid = sum(1 for m in messages if getattr(m, "valid", True))
        if valid:
            self._valid += valid
            self._last_time = self._clock()
            self._on_valid()
        return messages

    def reset(self) -> None:
        """Forget the stream and the counts (the settings stay)."""
        self._valid = 0
        self._last_time = None
        self._stream_s = 0.0
        for decoder in self._decoders():
            decoder.reset()
            for attr in ("frames_seen", "frames_failed"):
                if hasattr(decoder, attr):
                    setattr(decoder, attr, 0)

    @property
    def status(self) -> LiveDecoderStatus:
        frames = failed = 0
        for decoder in self._decoders():
            frames += int(getattr(decoder, "frames_seen", 0))
            failed += int(getattr(decoder, "frames_failed", 0))
        return LiveDecoderStatus(
            name=self.name,
            unit=self.unit,
            frames=max(frames, self._valid),
            valid=self._valid,
            failed=failed,
            last_time=self._last_time,
            problem=self._problem,
            sample_rate=self.sample_rate,
            channel_rate=self.sample_rate,
            offset_hz=self._offset,
            stream_s=self._stream_s,
        )

    # -- for subclasses -----------------------------------------------------

    def _decode(self, iq: np.ndarray) -> List[DecodedMessage]:
        raise NotImplementedError

    def _decoders(self) -> List[ProtocolDecoder]:
        return []

    def _on_valid(self) -> None:
        pass


class ChannelDecoder(LiveDecoder):
    """A protocol in a narrow channel: the I/Q is mixed down by the VFO
    offset and decimated to about ``TARGET_RATE`` (keeping +/- ``PASS_HZ``,
    60 dB down beyond +/- ``STOP_HZ``), then :meth:`_demodulate` turns the
    channel into messages. Also tracks whether a signal is in the channel:
    its power against the channel's noise floor, every 50 ms.
    """

    PASS_HZ = 8e3
    STOP_HZ = 12e3
    TARGET_RATE = 48e3
    MIN_SAMPLE_RATE = 16e3
    # Signal detection: power this far above the floor (the 20th
    # percentile of the last FLOOR_S seconds, once WARMUP_S are in).
    _SEGMENT_S = 0.05
    _PRESENT_DB = 6.0
    _FLOOR_S = 20.0
    _WARMUP_S = 2.0
    _HOLD_S = 1.0

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(sample_rate, offset_hz, clock)
        self._channelizer = Channelizer(
            self.sample_rate, offset_hz, self.PASS_HZ, self.STOP_HZ, self.TARGET_RATE
        )
        self.channel_rate = self._channelizer.output_rate
        self._segment_len = max(1, int(round(self._SEGMENT_S * self.channel_rate)))
        self._reset_presence()
        self._build()

    def _reset_presence(self) -> None:
        self._seg_sum = 0.0
        self._seg_count = 0
        self._powers: Deque[float] = deque(maxlen=int(self._FLOOR_S / self._SEGMENT_S))
        self._floor: Optional[float] = None
        self._since_present: Optional[float] = None  # seconds since last present
        self._undecoded = 0.0

    def _build(self) -> None:
        """Create the demodulator and protocol decoder(s)."""
        raise NotImplementedError

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        raise NotImplementedError

    def set_offset(self, offset_hz: float) -> None:
        super().set_offset(offset_hz)
        self._channelizer.set_offset(offset_hz)

    def reset(self) -> None:
        super().reset()
        self._channelizer.reset()
        self._reset_presence()
        self._build()

    def _decode(self, iq: np.ndarray) -> List[DecodedMessage]:
        channel = self._channelizer.process(iq)
        if not len(channel):
            return []
        self._measure(channel)
        return self._demodulate(channel)

    def _measure(self, channel: np.ndarray) -> None:
        """Track the channel power in 50 ms segments (block size has no
        effect) and whether a signal stands above the floor."""
        power = (
            channel.real.astype(np.float64) ** 2 + channel.imag.astype(np.float64) ** 2
        )
        pos = 0
        n = len(power)
        seg = self._segment_len
        while pos < n:
            take = min(seg - self._seg_count, n - pos)
            self._seg_sum += float(np.sum(power[pos : pos + take]))
            self._seg_count += take
            pos += take
            if self._seg_count < seg:
                break
            self._segment_done(self._seg_sum / seg)
            self._seg_sum, self._seg_count = 0.0, 0

    def _segment_done(self, power: float) -> None:
        self._powers.append(power)
        if len(self._powers) * self._SEGMENT_S < self._WARMUP_S:
            present = False
        else:
            if self._floor is None or len(self._powers) % 10 == 0:
                self._floor = float(np.percentile(np.asarray(self._powers), 20))
            present = power > self._floor * 10.0 ** (self._PRESENT_DB / 10.0)
        if present:
            self._since_present = 0.0
            self._undecoded += self._SEGMENT_S
        elif self._since_present is not None:
            self._since_present += self._SEGMENT_S

    def _on_valid(self) -> None:
        self._undecoded = 0.0

    @property
    def status(self) -> LiveDecoderStatus:
        base = super().status
        present = (
            self._since_present is not None and self._since_present <= self._HOLD_S
        )
        return replace(
            base,
            channel_rate=self.channel_rate,
            signal_present=present,
            signal_measured=True,
            undecoded_signal_s=self._undecoded,
        )


class _DCBlocker:
    """Removes a slowly varying offset (e.g. a mistuned carrier in an FM
    discriminator's output): the input minus its moving average over the
    last ``length`` samples (no delay is added to the signal)."""

    def __init__(self, length: int):
        self._avg = MovingAverage(max(1, int(length)))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x - self._avg.process(x)


class POCSAGLiveDecoder(ChannelDecoder):
    """POCSAG pages (512, 1200 and 2400 baud at once) on an NBFM channel."""

    name = "POCSAG"
    unit = "page"
    _DC_S = 0.25

    def _build(self) -> None:
        rate = self.channel_rate
        self._fm = FMDiscriminator(rate)
        self._dc = _DCBlocker(int(self._DC_S * rate))
        self._decoder = POCSAGDecoder(rate, 1200, auto_baud=True)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        baseband = self._dc.process(self._fm.process(channel))
        return list(self._decoder.decode(baseband))


class FLEXLiveDecoder(ChannelDecoder):
    """FLEX pages (1600 bps, 2-level FSK) on an NBFM channel."""

    name = "FLEX"
    unit = "page"
    _DC_S = 0.25

    def _build(self) -> None:
        rate = self.channel_rate
        self._fm = FMDiscriminator(rate)
        self._dc = _DCBlocker(int(self._DC_S * rate))
        self._decoder = FLEXDecoder(rate, 1600)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        baseband = self._dc.process(self._fm.process(channel))
        return list(self._decoder.decode(baseband))

    @property
    def status(self) -> LiveDecoderStatus:
        base = super().status
        decoder = self._decoder
        if decoder.unsupported_frames and not base.valid:
            return replace(
                base,
                problem=(
                    f"This channel sends FLEX at {decoder.last_unsupported}; "
                    "only 1600 bps 2-level FLEX decodes yet"
                ),
            )
        return base


class AX25LiveDecoder(ChannelDecoder):
    """AX.25 packet radio and APRS: 1200 baud Bell 202 AFSK on NBFM."""

    name = "AX.25/APRS"
    unit = "packet"
    MARK_HZ = 1200.0
    SPACE_HZ = 2200.0
    BAUD = 1200

    def _build(self) -> None:
        rate = self.channel_rate
        self._fm = FMDiscriminator(rate)
        self._mark = Nco(rate, self.MARK_HZ)
        self._space = Nco(rate, self.SPACE_HZ)
        bit = int(round(rate / self.BAUD))
        self._mark_avg = MovingAverage(bit)
        self._space_avg = MovingAverage(bit)
        self._decoder = AX25Decoder(rate, self.BAUD, matched_filter=False)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        audio = self._fm.process(channel)
        mark = np.abs(self._mark_avg.process(self._mark.process(audio)))
        space = np.abs(self._space_avg.process(self._space.process(audio)))
        # Mark minus space, normalised: twist (one tone louder, as after
        # pre-emphasis) and the audio level do not move the threshold.
        tones = (mark - space) / (mark + space + 1e-12)
        return list(self._decoder.decode(tones))


class ACARSLiveDecoder(ChannelDecoder):
    """ACARS: 2400 bit/s MSK (1200/2400 Hz tones) on an AM channel.

    The AM audio is mixed down by 1800 Hz (the middle of the two tones) and
    low-passed into a complex baseband whose phase turns by +90 degrees in a
    2400 Hz bit and -90 in a 1200 Hz bit. Turning it back by 90 degrees per
    bit (a 600 Hz rotation) leaves a phasor that stays put for a 2400 Hz
    bit and flips over for a 1200 Hz one: its real part, once the constant
    phase is removed, is the data itself (as acarsdec reads MSK), so a bit
    error stays one bit error. The phase comes from the phasor squared
    (which no longer depends on the data), averaged over 32 bits.
    """

    name = "ACARS"
    unit = "message"
    PASS_HZ = 5e3
    STOP_HZ = 8e3
    MIN_SAMPLE_RATE = 9.6e3
    CENTER_HZ = 1800.0
    BAUD = 2400
    _PHASE_BITS = 32

    def _build(self) -> None:
        rate = self.channel_rate
        # The carrier (the envelope's mean) over 1/600 s: its nulls at
        # multiples of 600 Hz leave the 1200/1800/2400 Hz audio untouched.
        self._dc = _DCBlocker(int(round(rate / 600.0)))
        self._mix = Nco(rate, self.CENTER_HZ)
        self._lowpass = FirDecimator(lowpass_taps(1500.0, 2300.0, rate))
        self._rotate = Nco(rate, self.BAUD / 4.0)
        length = int(round(rate * self._PHASE_BITS / self.BAUD)) | 1
        self._phase_avg = MovingAverage(length)
        self._phase_delay = Delay((length - 1) // 2)
        self._angle = 0.0  # unwrapped phase (x2) at the end of the last block
        self._decoder = ACARSDecoder(rate)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        envelope = np.abs(channel).astype(np.float64)
        audio = self._dc.process(envelope)
        phasor = self._rotate.process(self._lowpass.process(self._mix.process(audio)))
        squared = self._phase_avg.process(phasor.astype(np.complex128) ** 2)
        angle = np.unwrap(np.concatenate(([self._angle], np.angle(squared))))[1:]
        self._angle = float(angle[-1])
        centred = self._phase_delay.process(phasor.astype(np.complex128))
        data = np.real(centred * np.exp(-0.5j * angle))
        return list(self._decoder.decode(data))


class ADSBLiveDecoder(LiveDecoder):
    """ADS-B / Mode S on 1090 MHz: the pulse envelope at the full rate.

    The 0.5 us pulses need at least 2 MS/s; at a lower rate the decoder
    reports that as its status problem and decodes nothing. The envelope
    does not depend on where the signal is in the span, so the VFO offset
    is not used.
    """

    name = "ADS-B"
    unit = "message"
    MIN_SAMPLE_RATE = ADSBDecoder.WORK_RATE

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(sample_rate, offset_hz, clock)
        self._decoder: Optional[ADSBDecoder] = None
        if self.sample_rate < self.MIN_SAMPLE_RATE:
            self._problem = (
                f"ADS-B needs at least {format_rate(self.MIN_SAMPLE_RATE)}; "
                f"the device runs at {format_rate(self.sample_rate)}"
            )
        else:
            self._decoder = ADSBDecoder(self.sample_rate)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder] if self._decoder is not None else []

    def _decode(self, iq: np.ndarray) -> List[DecodedMessage]:
        if self._decoder is None:
            return []
        if np.iscomplexobj(iq):
            magnitude = np.abs(iq).astype(np.float32)
        else:
            magnitude = np.abs(np.asarray(iq, dtype=np.float32))
        return list(self._decoder.decode(magnitude))


#: Decoder panel name -> live decoder class. T6 adds "RDS" and "CW (Morse)".
LIVE_DECODERS: Dict[str, Type[LiveDecoder]] = {
    "POCSAG": POCSAGLiveDecoder,
    "FLEX": FLEXLiveDecoder,
    "AX.25/APRS": AX25LiveDecoder,
    "ADS-B": ADSBLiveDecoder,
    "ACARS": ACARSLiveDecoder,
}

# Other names a protocol is known by (the CLI's and ProtocolType values).
_ALIASES = {
    "pocsag": "POCSAG",
    "flex": "FLEX",
    "ax25": "AX.25/APRS",
    "aprs": "AX.25/APRS",
    "ax.25": "AX.25/APRS",
    "adsb": "ADS-B",
    "ads-b": "ADS-B",
    "acars": "ACARS",
}


def live_decoder_class(name: str) -> Optional[Type[LiveDecoder]]:
    """The live decoder class for a panel name or alias (None: unknown)."""
    cls = LIVE_DECODERS.get(name)
    if cls is None:
        alias = _ALIASES.get(str(name).strip().lower())
        cls = LIVE_DECODERS.get(alias) if alias else None
    return cls


def create_live_decoder(
    name: str, sample_rate: float, offset_hz: float = 0.0
) -> LiveDecoder:
    """A live decoder for protocol ``name`` (a Decoder panel name such as
    "AX.25/APRS", or an alias such as "aprs") at ``sample_rate``, its
    channel ``offset_hz`` from the centre of the samples.

    Raises:
        ValueError: unknown protocol, or a sample rate too low for it
            (except ADS-B, which reports a low rate in its status).
    """
    cls = live_decoder_class(name)
    if cls is None:
        raise ValueError(f"No live decoder for {name!r}")
    rate = float(sample_rate)
    if not rate > 0:
        raise ValueError(f"sample rate must be positive, got {rate:g}")
    if cls is not ADSBLiveDecoder and rate < cls.MIN_SAMPLE_RATE:
        raise ValueError(
            f"The sample rate is too low for {cls.name}: it needs at least "
            f"{format_rate(cls.MIN_SAMPLE_RATE)}, got {format_rate(rate)}"
        )
    return cls(rate, offset_hz)


def describe_message(msg: Any) -> Tuple[str, str]:
    """``(address, content)`` text for a decoded message, as the Decoder
    panel's table and the command line show it."""
    name = type(msg).__name__

    def text(attr: str) -> str:
        return str(getattr(msg, attr, "") or "").strip()

    def position() -> str:
        lat = getattr(msg, "latitude", 0.0) or 0.0
        lon = getattr(msg, "longitude", 0.0) or 0.0
        return f"{lat:.4f}, {lon:.4f}" if lat and lon else ""

    def joined(*parts: str) -> str:
        return "  ·  ".join(p for p in parts if p)

    if name == "POCSAGMessage":
        return f"{msg.address} (F{msg.function})", text("content")
    if name == "FLEXMessage":
        return str(getattr(msg, "capcode", "") or ""), text("content")
    if name == "ADSBMessage":
        altitude = getattr(msg, "altitude", 0) or 0
        speed = getattr(msg, "velocity", 0.0) or 0.0
        squawk = text("squawk")
        return text("icao_address"), joined(
            text("callsign"),
            f"{altitude:,} ft" if altitude else "",
            position(),
            f"{speed:.0f} kt" if speed else "",
            f"Squawk {squawk}" if squawk else "",
        )
    if name in ("AX25Frame", "APRSMessage"):
        source, dest = text("source"), text("destination")
        address = f"{source}>{dest}" if source or dest else ""
        return address, joined(position(), text("info") or text("comment"))
    if name == "ACARSMessage":
        return text("registration") or text("flight_id"), joined(
            text("label"), text("text")
        )
    if name == "RDSData":
        pi_code = getattr(msg, "pi_code", 0) or 0
        return (
            f"PI {pi_code:04X}" if pi_code else "",
            joined(text("ps_name"), text("radio_text")),
        )
    address = text("address") or text("icao_address")
    return address, text("content") or text("info")
