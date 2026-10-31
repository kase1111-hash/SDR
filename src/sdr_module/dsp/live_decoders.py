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
CW (Morse)  channel (+/-250 Hz, ~2 kHz) -> envelope -> KeyingDetector (peak
            and noise trackers, hysteresis) -> MorseDecoder (timing, speed)
RDS         channel (+/-100 kHz, ~250 kHz) -> FM discriminator (multiplex)
            -> RDSDemodulator (57 kHz locked to 3x the 19 kHz pilot, biphase,
            clock recovery, differential decoding) -> RDSDecoder
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
from .morse import KeyingDetector, MorseDecoder, MorseMessage
from .protocols import (
    APRS_TYPE_NAMES,
    ACARSDecoder,
    ADSBDecoder,
    AX25Decoder,
    DecodedMessage,
    FLEXDecoder,
    POCSAGDecoder,
    ProtocolDecoder,
    RDSData,
    RDSDecoder,
    parse_aprs_info,
)
from .rds import RDSDemodulator

__all__ = [
    "LiveDecoderStatus",
    "LiveDecoder",
    "ChannelDecoder",
    "POCSAGLiveDecoder",
    "FLEXLiveDecoder",
    "AX25LiveDecoder",
    "ACARSLiveDecoder",
    "ADSBLiveDecoder",
    "CWLiveDecoder",
    "RDSLiveDecoder",
    "LIVE_DECODERS",
    "create_live_decoder",
    "describe_message",
    "message_details",
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
    #: What to do about the problem (a sentence), or None.
    remedy: Optional[str] = None
    #: Rate of the samples given to :meth:`LiveDecoder.process`.
    sample_rate: float = 0.0
    #: Rate the channel is decoded at (the full rate for ADS-B).
    channel_rate: float = 0.0
    #: Where the channel is, relative to the centre of the samples.
    offset_hz: float = 0.0
    #: Seconds of samples processed.
    stream_s: float = 0.0
    #: A short summary of what is being decoded, shown after the protocol
    #: name (e.g. "18 WPM" for Morse, the station name for RDS), or "".
    detail: str = ""
    #: Text still being received (Morse: the line in progress), or "".
    live_text: str = ""


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
        self._remedy: Optional[str] = None

    # -- API ----------------------------------------------------------------

    @property
    def offset_hz(self) -> float:
        return self._offset

    def set_offset(self, offset_hz: float) -> None:
        """Follow the VFO: the channel is now ``offset_hz`` from the centre."""
        self._offset = float(offset_hz)

    def channel_changed(self) -> None:
        """The receiver was retuned (the LO, the VFO or both moved): what
        the channel carries now is something else, so forget what was
        measured in it (the counts and the protocol state stay)."""

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

    def flush(self) -> List[DecodedMessage]:
        """The end of the stream (e.g. of a recording): messages still being
        assembled, such as a line of Morse code in progress. Most decoders
        have none."""
        return []

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
            remedy=self._remedy,
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
    its power against the channel's noise floor, every 50 ms, and how much
    of that signal went by without a valid message (forgotten when the
    channel changes or stays quiet for ``_QUIET_S``).
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
    # Seconds without signal after which the undecoded signal heard
    # before no longer counts ("Signal present but nothing decodes" is
    # about what is on the channel now, not a minute ago).
    _QUIET_S = 30.0

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
        changed = float(offset_hz) != self._channelizer.offset_hz
        super().set_offset(offset_hz)
        self._channelizer.set_offset(offset_hz)
        if changed:
            self.channel_changed()

    def channel_changed(self) -> None:
        # A new channel has its own floor and signal: measure them afresh.
        self._reset_presence()

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
            if self._since_present >= self._QUIET_S:
                self._undecoded = 0.0  # the channel went quiet

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
    """FLEX pages (1600 bps, 2-level FSK) on an NBFM channel.

    Frames sent at a speed it can't decode yet (3200 or 6400 bps) are a
    status problem while they are heard: until ``_OTHER_SPEED_S`` (about
    three frames) after the last one, and only while no page has decoded
    since the channel changed.
    """

    name = "FLEX"
    unit = "page"
    _DC_S = 0.25
    _OTHER_SPEED_S = 6.0

    def _build(self) -> None:
        rate = self.channel_rate
        self._fm = FMDiscriminator(rate)
        self._dc = _DCBlocker(int(self._DC_S * rate))
        self._decoder = FLEXDecoder(rate, 1600)
        self._other_speed_s: Optional[float] = None  # stream time last heard
        self._valid_before = self._valid  # pages decoded before this channel

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def channel_changed(self) -> None:
        super().channel_changed()
        # What the old channel sent says nothing about this one.
        self._decoder.unsupported_frames = 0
        self._decoder.last_unsupported = ""
        self._other_speed_s = None
        self._valid_before = self._valid

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        before = self._decoder.unsupported_frames
        baseband = self._dc.process(self._fm.process(channel))
        messages: List[DecodedMessage] = list(self._decoder.decode(baseband))
        if self._decoder.unsupported_frames > before:
            self._other_speed_s = self._stream_s
        return messages

    @property
    def status(self) -> LiveDecoderStatus:
        base = super().status
        decoder = self._decoder
        heard = (
            self._other_speed_s is not None
            and self._stream_s - self._other_speed_s <= self._OTHER_SPEED_S
        )
        if heard and decoder.last_unsupported and self._valid == self._valid_before:
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


class CWLiveDecoder(ChannelDecoder):
    """Morse code (CW): an on/off keyed carrier at the tuned frequency.

    The channel is +/-250 Hz around the VFO at about 2 kHz (tune the
    carrier to within about 200 Hz, as the CW mode's 500 Hz filter hears
    it). Its envelope keys a :class:`~sdr_module.dsp.morse.KeyingDetector`
    and a :class:`~sdr_module.dsp.morse.MorseDecoder` turns the marks and
    gaps into text, learning the speed as it goes.

    Messages are lines of text (:class:`~sdr_module.dsp.morse.MorseMessage`):
    a line ends after a pause of 2.5 s, a long carrier or about 60
    characters. The status counts characters; its ``detail`` is the speed
    ("18 WPM") and its ``live_text`` the line still being received.
    Retuning ends the line heard on the old channel and learns the new
    channel's levels and speed afresh.
    """

    name = "CW (Morse)"
    unit = "character"
    PASS_HZ = 250.0
    STOP_HZ = 450.0
    TARGET_RATE = 2000.0
    MIN_SAMPLE_RATE = 4e3

    def _build(self) -> None:
        rate = self.channel_rate
        self._keyer = KeyingDetector(rate)
        self._morse = MorseDecoder(rate)
        self._held: List[DecodedMessage] = []  # lines ended by a retune
        self._chars = 0  # characters decoded (all channels)
        self._char_time: Optional[float] = None

    def channel_changed(self) -> None:
        super().channel_changed()
        # Another station: its line ends here, and its levels and speed
        # are its own.
        self._held.extend(self._counted(self._morse.flush))
        self._keyer.reset()
        self._morse.reset()

    def _counted(self, step: Callable[[], List[MorseMessage]]) -> List[MorseMessage]:
        """Run a step of the Morse decoder and count the characters it
        decoded."""
        before = self._morse.characters
        lines = step()
        new = self._morse.characters - before
        if new > 0:
            self._chars += new
            self._char_time = self._clock()
            self._undecoded = 0.0  # it decodes
        return lines

    def _demodulate(self, channel: np.ndarray) -> List[DecodedMessage]:
        lines: List[DecodedMessage] = list(self._held)
        self._held = []
        keying = self._keyer.process(np.abs(channel))
        lines += self._counted(lambda: self._morse.process(keying))
        return lines

    def flush(self) -> List[DecodedMessage]:
        lines: List[DecodedMessage] = list(self._held)
        self._held = []
        return lines + list(self._counted(self._morse.flush))

    @property
    def wpm(self) -> Optional[float]:
        """The measured speed, or None before there is one."""
        return self._morse.wpm if self._morse.calibrated else None

    @property
    def status(self) -> LiveDecoderStatus:
        base = super().status
        wpm = self.wpm
        return replace(
            base,
            frames=self._chars,
            valid=self._chars,
            last_time=self._char_time,
            detail=f"{wpm:.0f} WPM" if wpm else "",
            live_text=self._morse.line,
        )


class RDSLiveDecoder(LiveDecoder):
    """RDS (station name, RadioText, program type, clock) from the FM
    broadcast station at the tuned frequency.

    The station's channel (+/-100 kHz) is picked out at about 250 kS/s
    (240 kS/s from 2.4 MS/s) and FM-discriminated;
    :class:`~sdr_module.dsp.rds.RDSDemodulator` recovers
    the RDS bits from the multiplex (57 kHz locked to the 19 kHz pilot
    where there is one) and :class:`~sdr_module.dsp.protocols.RDSDecoder`
    finds the groups. With ``channel_ready`` the input already is the
    station's channel at 0 Hz (e.g. the receiver's own broadcast FM
    channel, at 192 kS/s or more), so none is picked out.

    Messages are the changes (an :class:`~sdr_module.dsp.protocols.RDSData`
    with ``update`` set): a whole new station name or RadioText, the
    program type and its name, the clock time. They wait until the
    station's PI code is known (it is in every one), and the program type
    waits for the station name, so the name comes first. The status counts
    groups; its ``detail`` is the station name. Retuning
    (``channel_changed``) forgets the station.
    """

    name = "RDS"
    unit = "group"
    PASS_HZ = 100e3
    STOP_HZ = 140e3
    # 250 kS/s: from 20 MS/s (a HackRF) that is a decimation by 80 = 16 x 5,
    # whose first stage costs half what 81 = 9 x 9 (246.9 kS/s) costs; from
    # the other usual rates it gives the same channel as 240 kS/s.
    TARGET_RATE = 250e3
    MIN_SAMPLE_RATE = 200e3
    #: Updates that are rows of their own (the PI code is in all of them).
    ROW_UPDATES = ("ps", "radio_text", "pty", "ptyn", "clock_time")

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        channel_ready: bool = False,
    ):
        super().__init__(sample_rate, offset_hz, clock)
        self._channelizer: Optional[Channelizer] = None
        if channel_ready:
            self.channel_rate = self.sample_rate
        else:
            self._channelizer = Channelizer(
                self.sample_rate,
                offset_hz,
                self.PASS_HZ,
                self.STOP_HZ,
                self.TARGET_RATE,
            )
            self.channel_rate = self._channelizer.output_rate
        self._group_time: Optional[float] = None
        self._pilot_s = 0.0  # seconds of pilot without RDS sync
        self._build()

    def _build(self) -> None:
        self._fm = FMDiscriminator(self.channel_rate)
        self._demod = RDSDemodulator(self.channel_rate)
        self._decoder = RDSDecoder(self._demod.work_rate)
        self._held: List[RDSData] = []  # updates waiting for the PI or PS

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder]

    def set_offset(self, offset_hz: float) -> None:
        changed = float(offset_hz) != self._offset
        super().set_offset(offset_hz)
        if self._channelizer is not None:
            self._channelizer.set_offset(offset_hz)
        if changed:
            self.channel_changed()

    def channel_changed(self) -> None:
        """Another station: forget this one's name, text and sync."""
        self._fm.reset()
        self._demod.reset()
        self._decoder.reset()
        self._held = []
        self._pilot_s = 0.0

    def reset(self) -> None:
        super().reset()
        if self._channelizer is not None:
            self._channelizer.reset()
        self._group_time = None
        self._pilot_s = 0.0
        self._build()

    @property
    def station(self) -> RDSData:
        """What is known about the station now (empty fields: not yet)."""
        return self._decoder.station()

    @property
    def synced(self) -> bool:
        return self._decoder.synced

    @property
    def pilot(self) -> bool:
        """Whether the station sends a 19 kHz stereo pilot."""
        return self._demod.pilot

    def _decode(self, iq: np.ndarray) -> List[DecodedMessage]:
        if self._channelizer is not None:
            channel = self._channelizer.process(iq)
        else:
            channel = np.asarray(iq, dtype=np.complex64)
        if not len(channel):
            return []
        groups = self._decoder.frames_seen - self._decoder.frames_failed
        updates = self._decoder.decode_bits(
            self._demod.process(self._fm.process(channel)), updates_only=True
        )
        if self._decoder.frames_seen - self._decoder.frames_failed > groups:
            self._group_time = self._clock()
        seconds = len(channel) / self.channel_rate
        if self._decoder.synced:
            self._pilot_s = 0.0
        elif self._demod.pilot:
            self._pilot_s += seconds
        return self._rows([u for u in updates if u.update in self.ROW_UPDATES])

    def _rows(self, updates: List[RDSData]) -> List[DecodedMessage]:
        """The updates that can be shown now: once the PI code is known
        (filled in), and the program type once the station name is."""
        self._held.extend(updates)
        station = self._decoder.station()
        if not self._decoder.pi_known:
            return []
        ready: List[DecodedMessage] = []
        waiting: List[RDSData] = []
        for update in self._held:
            if update.update in ("pty", "ptyn") and not station.ps_complete:
                waiting.append(update)
                continue
            ready.append(replace(update, pi_code=station.pi_code))
        # The name first: an earlier program type follows it.
        ready.sort(key=lambda u: 0 if getattr(u, "update", "") == "ps" else 1)
        self._held = waiting
        return ready

    @property
    def status(self) -> LiveDecoderStatus:
        base = super().status
        decoder = self._decoder
        station = decoder.station()
        good = decoder.frames_seen - decoder.frames_failed
        return replace(
            base,
            frames=decoder.frames_seen,
            valid=good,
            failed=decoder.frames_failed,
            last_time=self._group_time,
            channel_rate=self.channel_rate,
            signal_present=decoder.synced or self._demod.pilot,
            signal_measured=True,
            undecoded_signal_s=self._pilot_s,
            detail=station.ps_name if station.ps_complete else "",
        )


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
            self._remedy = (
                "Choose 2.4 MS/s or more in Receiver > Sample rate (or "
                "click the rate in the status bar)."
            )
        else:
            self._decoder = ADSBDecoder(self.sample_rate)

    def _decoders(self) -> List[ProtocolDecoder]:
        return [self._decoder] if self._decoder is not None else []

    def _decode(self, iq: np.ndarray) -> List[DecodedMessage]:
        if self._decoder is None:
            return []
        if np.iscomplexobj(iq):
            magnitude = np.abs(iq).astype(np.float32, copy=False)
        else:
            magnitude = np.abs(np.asarray(iq, dtype=np.float32))
        return list(self._decoder.decode(magnitude))


#: Decoder panel name -> live decoder class.
LIVE_DECODERS: Dict[str, Type[LiveDecoder]] = {
    "POCSAG": POCSAGLiveDecoder,
    "FLEX": FLEXLiveDecoder,
    "AX.25/APRS": AX25LiveDecoder,
    "ADS-B": ADSBLiveDecoder,
    "ACARS": ACARSLiveDecoder,
    "RDS": RDSLiveDecoder,
    "CW (Morse)": CWLiveDecoder,
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
    "cw": "CW (Morse)",
    "morse": "CW (Morse)",
    "rds": "RDS",
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


def _text(msg: Any, attr: str) -> str:
    return str(getattr(msg, attr, "") or "").strip()


def _joined(*parts: str) -> str:
    return "  ·  ".join(p for p in parts if p)


def _position(msg: Any) -> str:
    lat = getattr(msg, "latitude", 0.0) or 0.0
    lon = getattr(msg, "longitude", 0.0) or 0.0
    return f"{lat:.4f}, {lon:.4f}" if lat and lon else ""


def _vertical(rate: int) -> str:
    """``1200 -> "climbing 1,200 ft/min"``, ``0 -> "level"``."""
    if rate > 0:
        return f"climbing {rate:,} ft/min"
    if rate < 0:
        return f"descending {-rate:,} ft/min"
    return "level"


def _adsb_content(msg: Any) -> str:
    altitude = int(getattr(msg, "altitude", 0) or 0)
    speed = float(getattr(msg, "velocity", 0.0) or 0.0)
    parts = [_text(msg, "callsign"), _text(msg, "category")]
    if altitude:
        parts.append(f"{altitude:,} ft")
    parts.append(_position(msg))
    if speed and getattr(msg, "type_code", 0) == 19:
        heading = float(getattr(msg, "heading", 0.0) or 0.0)
        parts += [f"{speed:.0f} kt", f"track {round(heading) % 360}°"]
        if getattr(msg, "vertical_rate_known", True):
            parts.append(_vertical(int(getattr(msg, "vertical_rate", 0) or 0)))
    elif speed:
        parts.append(f"{speed:.0f} kt")
    squawk = _text(msg, "squawk")
    if squawk:
        parts.append(f"Squawk {squawk}")
    if getattr(msg, "on_ground", False):
        parts.append("on the ground")
    content = _joined(*parts)
    if content:
        return content
    df = int(getattr(msg, "downlink_format", 0) or 0)
    if df == 11:
        return "Address only (all-call reply)"
    if df in (17, 18):
        tc = int(getattr(msg, "type_code", 0) or 0)
        kind = _ADSB_KINDS.get(tc, f"type code {tc}")
        return f"{kind[0].upper()}{kind[1:]} (not read)"
    return "Address only"


def _aprs_fields(msg: Any) -> Optional[Dict[str, Any]]:
    """APRS fields of an AX.25 frame or APRS message (None: not APRS)."""
    if type(msg).__name__ == "APRSMessage":
        return {
            key: getattr(msg, key, "")
            for key in (
                "data_type",
                "latitude",
                "longitude",
                "altitude",
                "speed",
                "course",
                "symbol",
                "symbol_name",
                "addressee",
                "comment",
            )
        }
    info = _text(msg, "info")
    if not info or not getattr(msg, "valid", True):
        return None
    try:
        fields = parse_aprs_info(info)
    except Exception:  # a malformed report is shown as sent
        return None
    kind = str(fields["data_type"])
    if kind == "unknown" or (kind.startswith("position") and not fields["symbol"]):
        return None  # not APRS, or a position that doesn't read: as sent
    return fields


def _aprs_content(fields: Dict[str, Any]) -> str:
    kind = fields.get("data_type", "")
    comment = str(fields.get("comment", "") or "")
    addressee = str(fields.get("addressee", "") or "")
    if kind == "message":
        return f"To {addressee}: {comment}" if addressee else comment
    if kind == "mic-e":
        return "Mic-E position report (not decoded)"
    lat = float(fields.get("latitude", 0.0) or 0.0)
    lon = float(fields.get("longitude", 0.0) or 0.0)
    speed = float(fields.get("speed", 0.0) or 0.0)
    course = float(fields.get("course", 0.0) or 0.0)
    altitude = float(fields.get("altitude", 0.0) or 0.0)
    moving = ""
    if speed or course:
        moving = f"{speed:.0f} kt, course {course:.0f}°"
    return _joined(
        f"Object {addressee}" if kind in ("object", "item") and addressee else "",
        f"{lat:.4f}, {lon:.4f}" if lat and lon else "",
        str(fields.get("symbol_name", "") or ""),
        moving,
        f"{altitude:,.0f} ft" if altitude else "",
        comment,
    )


def describe_message(msg: Any) -> Tuple[str, str]:
    """``(address, content)`` text for a decoded message, as the Decoder
    panel's table and the command line show it."""
    name = type(msg).__name__

    def text(attr: str) -> str:
        return _text(msg, attr)

    if name == "POCSAGMessage":
        return f"{msg.address} (F{msg.function})", text("content")
    if name == "FLEXMessage":
        return str(getattr(msg, "capcode", "") or ""), text("content")
    if name == "ADSBMessage":
        return text("icao_address"), _adsb_content(msg)
    if name in ("AX25Frame", "APRSMessage"):
        source, dest = text("source"), text("destination")
        address = f"{source}>{dest}" if source or dest else ""
        fields = _aprs_fields(msg)
        if fields is not None:
            return address, _aprs_content(fields)
        return address, _joined(_position(msg), text("info") or text("comment"))
    if name == "ACARSMessage":
        registration, flight = text("registration"), text("flight_id")
        return registration or flight, _joined(
            f"Flight {flight}" if flight and registration else "",
            text("label"),
            text("text"),
        )
    if name == "MorseMessage":
        wpm = getattr(msg, "wpm", 0.0) or 0.0
        return (f"{wpm:.0f} WPM" if wpm else ""), text("text")
    if name == "RDSData":
        pi_code = getattr(msg, "pi_code", 0) or 0
        address = f"PI {pi_code:04X}" if pi_code else ""
        update = text("update")
        labelled = {
            "ps": ("Station name", text("ps_name")),
            "radio_text": ("Radio text", text("radio_text")),
            "pty": ("Program type", text("pty_name")),
            "ptyn": ("Program type name", text("ptyn")),
            "clock_time": ("Clock", text("clock_time")),
        }.get(update)
        if labelled is not None:
            label, value = labelled
            return address, f"{label}: {value}" if value else label
        return address, _joined(text("ps_name"), text("radio_text"))
    address = text("address") or text("icao_address")
    return address, text("content") or text("info")


#: ADS-B type codes -> what the message is.
_ADSB_KINDS = {
    **{tc: "identification" for tc in range(1, 5)},
    **{tc: "surface position" for tc in range(5, 9)},
    **{tc: "airborne position" for tc in range(9, 19)},
    19: "airborne velocity",
    **{tc: "airborne position (GNSS height)" for tc in range(20, 23)},
    28: "aircraft status",
    29: "target state",
    31: "operational status",
}
_ADSB_FORMATS = {
    0: "short air-air surveillance",
    4: "altitude reply",
    5: "identity reply",
    11: "all-call reply",
    16: "long air-air surveillance",
    17: "extended squitter",
    18: "extended squitter (non-transponder)",
    20: "altitude reply (Comm-B)",
    21: "identity reply (Comm-B)",
}


def message_details(msg: Any) -> List[str]:
    """Lines that say more about a decoded message than its row (for the
    Decoder panel's detail card): the path of an APRS packet, the kind of
    an ADS-B message, an ACARS message's flight and label, an RDS
    station's codes."""
    name = type(msg).__name__
    lines: List[str] = []
    if name == "ADSBMessage":
        df = int(getattr(msg, "downlink_format", 0) or 0)
        tc = int(getattr(msg, "type_code", 0) or 0)
        kind = f"DF{df} {_ADSB_FORMATS.get(df, '')}".strip()
        if df in (17, 18) and tc:
            kind += f", type code {tc}"
            if tc in _ADSB_KINDS:
                kind += f" ({_ADSB_KINDS[tc]})"
        lines.append(kind)
        category = _text(msg, "category")
        if category:
            lines.append(f"Category: {category}")
    elif name in ("AX25Frame", "APRSMessage"):
        path = list(
            getattr(msg, "digipeaters", None) or getattr(msg, "path", None) or []
        )
        lines.append(f"Path: via {', '.join(path)}" if path else "Path: direct")
        fields = _aprs_fields(msg)
        if fields is not None:
            kind = APRS_TYPE_NAMES.get(str(fields["data_type"]), "Other")
            symbol = str(fields.get("symbol") or "")
            if symbol:
                label = fields.get("symbol_name") or "symbol"
                kind += f" · {label} ({symbol})"
            lines.append(f"APRS: {kind}")
        info = _text(msg, "info")
        if info:
            lines.append(f"Sent: {info}")
    elif name == "ACARSMessage":
        parts = [
            (
                f"Registration {_text(msg, 'registration')}"
                if _text(msg, "registration")
                else ""
            ),
            f"flight {_text(msg, 'flight_id')}" if _text(msg, "flight_id") else "",
        ]
        if any(parts):
            lines.append(", ".join(p for p in parts if p))
        codes = [
            f"label {_text(msg, 'label')}" if _text(msg, "label") else "",
            f"block {_text(msg, 'block_id')}" if _text(msg, "block_id") else "",
            (
                f"message {_text(msg, 'message_number')}"
                if _text(msg, "message_number")
                else ""
            ),
            f"mode {_text(msg, 'mode')}" if _text(msg, "mode") else "",
        ]
        if any(codes):
            text = ", ".join(c for c in codes if c)
            lines.append(text[0].upper() + text[1:])
    elif name == "RDSData":
        pi_code = int(getattr(msg, "pi_code", 0) or 0)
        pty = int(getattr(msg, "pty", 0) or 0)
        pty_name = _text(msg, "pty_name")
        if pi_code:
            lines.append(
                f"PI {pi_code:04X} · program type {pty}"
                + (f" ({pty_name})" if pty_name else "")
            )
        flags = []
        if getattr(msg, "tp", False):
            flags.append("traffic program")
        if getattr(msg, "ta", False):
            flags.append("traffic announcement now")
        flags.append("music" if getattr(msg, "ms", False) else "speech")
        lines.append(", ".join(flags).capitalize())
        afs = list(getattr(msg, "af_list", None) or [])
        if afs:
            lines.append("Also on " + ", ".join(f"{f:.1f}" for f in afs) + " MHz")
    elif name == "MorseMessage":
        wpm = float(getattr(msg, "wpm", 0.0) or 0.0)
        if wpm:
            lines.append(f"Sent at {wpm:.0f} words per minute")
    return lines
