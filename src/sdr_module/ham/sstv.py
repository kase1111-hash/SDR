"""
SSTV (Slow Scan Television) decoder for receiving images from space.

Decodes SSTV transmissions from:
- the International Space Station (ISS) on 145.800 MHz FM,
- amateur radio operators,
- and anything else that sends the standard analog SSTV modes.

Supported modes (every entry of :data:`SSTV_MODES`): Robot 36 and 72,
Martin 1 and 2, Scottie 1, 2 and DX, PD 90, 120, 160, 180, 240 and 290, and
Wraase SC2-180.

How a transmission is decoded:

1. **Frequency track.** The audio goes through a complex band-pass filter
   (700-3000 Hz, the analytic signal of the SSTV tones) and a
   quadrature discriminator: one instantaneous-frequency estimate per audio
   sample.
2. **VIS header.** A 1900 Hz leader, a 1200 Hz break, the leader again,
   then a 30 ms 1200 Hz start bit, seven data bits (1100 Hz = 1, 1300 Hz =
   0, least significant first), an even parity bit and a 1200 Hz stop bit.
   The 7-bit code names the mode.
3. **Line sync.** Each line's 1200 Hz sync pulse is found by its trailing
   edge, near where the line timing says it should be. A least-squares fit
   through the measured syncs follows a sound card's sample-rate error
   (no slanted pictures) and ignores jitter; when syncs fade, the timing
   coasts on the fit.
4. **Pixels.** Each pixel is the mean frequency over its exact time slot
   (1500 Hz black to 2300 Hz white), laid out as the mode sends it: Martin
   (G, B, R after the sync), Scottie (G, B before the sync, R after it),
   Robot 36 (Y, then R-Y or B-Y on alternate lines, shared by the pair),
   Robot 72 (Y, R-Y, B-Y), PD (Y of two lines with their shared R-Y and
   B-Y) and Wraase SC2 (R, G, B). YUV modes use the ITU-R BT.601 studio
   range, as MMSSTV (the ISS's SSTV software) does.

Without a VIS, :meth:`SSTVDecoder.start_now` starts on the line syncs of a
transmission already under way.

Usage:
    decoder = SSTVDecoder(sample_rate=48000)
    decoder.process_audio(audio_samples)   # any block size

    if decoder.is_complete():
        image = decoder.get_image()
        decoder.save_image("iss_capture.png")

See :mod:`sdr_module.ham.sstv_encoder` for the matching encoder.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SSTVMode(Enum):
    """SSTV transmission modes."""

    # Robot modes
    ROBOT_36 = auto()  # 36 seconds, 320x240
    ROBOT_72 = auto()  # 72 seconds, 320x240

    # Scottie modes
    SCOTTIE_1 = auto()  # 110 seconds, 320x256
    SCOTTIE_2 = auto()  # 71 seconds, 320x256
    SCOTTIE_DX = auto()  # 269 seconds, 320x256

    # Martin modes
    MARTIN_1 = auto()  # 114 seconds, 320x256
    MARTIN_2 = auto()  # 58 seconds, 320x256

    # PD modes (ISS favorites)
    PD_90 = auto()  # 90 seconds, 320x256
    PD_120 = auto()  # 126 seconds, 640x496
    PD_160 = auto()  # 161 seconds, 512x400
    PD_180 = auto()  # 187 seconds, 640x496
    PD_240 = auto()  # 248 seconds, 640x496
    PD_290 = auto()  # 289 seconds, 800x616

    # Wraase modes
    WRAASE_SC2_180 = auto()  # 182 seconds, 320x256

    UNKNOWN = auto()


@dataclass
class SSTVModeSpec:
    """Specification for an SSTV mode.

    Timings are in milliseconds. ``scan_time_ms`` is the time from one sync
    pulse to the next: one image line, or two in the PD modes (see
    ``lines_per_sync``). ``channel_times_ms`` lists the scan time of each
    colour channel sent in that time, in transmission order.
    """

    name: str
    mode: SSTVMode
    vis_code: int  # Vertical Interval Signaling code (7 bits)
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    scan_time_ms: float  # Sync-to-sync period
    sync_pulse_ms: float  # Sync pulse duration (1200 Hz)
    sync_porch_ms: float  # Porch after the sync (1500 Hz)
    color_format: str  # "RGB", "GBR" (Martin/Scottie channel order) or "YUV"
    channel_times_ms: List[float]  # Scan time of each colour channel
    # Line layout: "robot36", "robot72", "martin", "scottie", "pd", "wraase".
    family: str = "martin"
    separator_ms: float = 0.0  # Gap between colour channels (1500 Hz)

    @property
    def lines_per_sync(self) -> int:
        """Image lines sent per sync pulse (2 in the PD modes)."""
        return 2 if self.family == "pd" else 1

    @property
    def sync_periods(self) -> int:
        """Sync pulses (line periods) in one image."""
        return self.height // self.lines_per_sync

    @property
    def duration_s(self) -> float:
        """Length of the image part of a transmission (after the VIS)."""
        pre = sum(seg[1] for seg in preamble_segments(self))
        return (pre + self.sync_periods * self.scan_time_ms) / 1000.0


# Robot 36/72: after the luma scan, a 4.5 ms separator (1500 Hz before R-Y,
# 2300 Hz before B-Y) and a 1.5 ms 1900 Hz porch come before each chroma scan.
_ROBOT_SEPARATOR_MS = 4.5
_ROBOT_CHROMA_PORCH_MS = 1.5
_ROBOT_CHROMA_PORCH_HZ = 1900.0


# SSTV Mode specifications, keyed by VIS code. Timings follow the SSTV
# handbook (JL Barber, "Proposal for SSTV Mode Specifications", Dayton
# 2000) and MMSSTV.
SSTV_MODES: Dict[int, SSTVModeSpec] = {
    # Robot modes - YUV format
    0x08: SSTVModeSpec(
        "Robot 36",
        SSTVMode.ROBOT_36,
        0x08,
        320,
        240,
        150.0,
        9.0,
        3.0,
        "YUV",
        [88.0, 44.0],
        family="robot36",
        separator_ms=_ROBOT_SEPARATOR_MS,
    ),
    0x0C: SSTVModeSpec(
        "Robot 72",
        SSTVMode.ROBOT_72,
        0x0C,
        320,
        240,
        300.0,
        9.0,
        3.0,
        "YUV",
        [138.0, 69.0, 69.0],
        family="robot72",
        separator_ms=_ROBOT_SEPARATOR_MS,
    ),
    # Scottie modes - GBR format, the sync sits between blue and red
    0x3C: SSTVModeSpec(
        "Scottie 1",
        SSTVMode.SCOTTIE_1,
        0x3C,
        320,
        256,
        428.22,
        9.0,
        1.5,
        "GBR",
        [138.24, 138.24, 138.24],
        family="scottie",
        separator_ms=1.5,
    ),
    0x38: SSTVModeSpec(
        "Scottie 2",
        SSTVMode.SCOTTIE_2,
        0x38,
        320,
        256,
        277.692,
        9.0,
        1.5,
        "GBR",
        [88.064, 88.064, 88.064],
        family="scottie",
        separator_ms=1.5,
    ),
    0x4C: SSTVModeSpec(
        "Scottie DX",
        SSTVMode.SCOTTIE_DX,
        0x4C,
        320,
        256,
        1050.3,
        9.0,
        1.5,
        "GBR",
        [345.6, 345.6, 345.6],
        family="scottie",
        separator_ms=1.5,
    ),
    # Martin modes - GBR format
    0x2C: SSTVModeSpec(
        "Martin 1",
        SSTVMode.MARTIN_1,
        0x2C,
        320,
        256,
        446.446,
        4.862,
        0.572,
        "GBR",
        [146.432, 146.432, 146.432],
        family="martin",
        separator_ms=0.572,
    ),
    0x28: SSTVModeSpec(
        "Martin 2",
        SSTVMode.MARTIN_2,
        0x28,
        320,
        256,
        226.798,
        4.862,
        0.572,
        "GBR",
        [73.216, 73.216, 73.216],
        family="martin",
        separator_ms=0.572,
    ),
    # PD modes - YUV, two image lines per sync: Y(even), R-Y, B-Y, Y(odd)
    0x63: SSTVModeSpec(
        "PD 90",
        SSTVMode.PD_90,
        0x63,  # standard PD90 VIS is 99 (0x63), not 93 (0x5D, which is PD50)
        320,
        256,
        703.04,
        20.0,
        2.08,
        "YUV",
        [170.24] * 4,
        family="pd",
    ),
    0x5F: SSTVModeSpec(
        "PD 120",
        SSTVMode.PD_120,
        0x5F,
        640,
        496,
        508.48,
        20.0,
        2.08,
        "YUV",
        [121.6] * 4,
        family="pd",
    ),
    0x62: SSTVModeSpec(
        "PD 160",
        SSTVMode.PD_160,
        0x62,
        512,
        400,
        804.416,
        20.0,
        2.08,
        "YUV",
        [195.584] * 4,
        family="pd",
    ),
    0x60: SSTVModeSpec(
        "PD 180",
        SSTVMode.PD_180,
        0x60,
        640,
        496,
        754.24,
        20.0,
        2.08,
        "YUV",
        [183.04] * 4,
        family="pd",
    ),
    0x61: SSTVModeSpec(
        "PD 240",
        SSTVMode.PD_240,
        0x61,
        640,
        496,
        1000.0,
        20.0,
        2.08,
        "YUV",
        [244.48] * 4,
        family="pd",
    ),
    0x5E: SSTVModeSpec(
        "PD 290",
        SSTVMode.PD_290,
        0x5E,  # standard PD290 VIS is 94 (0x5E), not 99 (0x63, which is PD90)
        800,
        616,
        937.28,
        20.0,
        2.08,
        "YUV",
        [228.8] * 4,
        family="pd",
    ),
    # Wraase SC2 - RGB format
    0x37: SSTVModeSpec(
        "Wraase SC2-180",
        SSTVMode.WRAASE_SC2_180,
        0x37,
        320,
        256,
        711.0225,
        5.5225,
        0.5,
        "RGB",
        [235.0, 235.0, 235.0],
        family="wraase",
    ),
}


def find_mode(mode: Union[str, int, SSTVMode, SSTVModeSpec]) -> SSTVModeSpec:
    """The :class:`SSTVModeSpec` for a mode name ("Martin 1"), VIS code,
    :class:`SSTVMode` or spec.

    Raises:
        KeyError: for an unknown mode.
    """
    if isinstance(mode, SSTVModeSpec):
        return mode
    if isinstance(mode, SSTVMode):
        for spec in SSTV_MODES.values():
            if spec.mode is mode:
                return spec
    elif isinstance(mode, (int, np.integer)):
        if int(mode) in SSTV_MODES:
            return SSTV_MODES[int(mode)]
    elif isinstance(mode, str):
        wanted = mode.replace("_", " ").replace("-", " ").lower().split()
        for spec in SSTV_MODES.values():
            if spec.name.replace("-", " ").lower().split() == wanted:
                return spec
    raise KeyError(f"Unknown SSTV mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Line layout (shared by the decoder and sstv_encoder)
# --------------------------------------------------------------------------- #
# One segment of a line: (kind, duration ms, frequency Hz). ``kind`` is
# "sync" (the 1200 Hz line sync), "tone" (a fixed tone: porch, separator),
# "parity" (Robot 36's separator: 1500 Hz before R-Y, 2300 Hz before B-Y) or
# a colour channel: "R", "G", "B", "Y", "Y0"/"Y1" (PD's two lines), "V"
# (R-Y) or "U" (B-Y). Channel segments carry frequency 0.
Segment = Tuple[str, float, float]

FREQ_SYNC_HZ = 1200.0
FREQ_BLACK_HZ = 1500.0
FREQ_WHITE_HZ = 2300.0
FREQ_LEADER_HZ = 1900.0
FREQ_VIS_ONE_HZ = 1100.0
FREQ_VIS_ZERO_HZ = 1300.0
VIS_BIT_MS = 30.0
LEADER_MS = 300.0
BREAK_MS = 10.0


def line_templates(spec: SSTVModeSpec) -> List[List[Segment]]:
    """Segments of one sync period, in transmission order.

    Robot 36 alternates two templates (even lines carry R-Y, odd lines
    B-Y); every other mode has one. Scottie lines start with the green
    scan: their sync sits between blue and red.
    """
    sync = ("sync", spec.sync_pulse_ms, FREQ_SYNC_HZ)
    porch = ("tone", spec.sync_porch_ms, FREQ_BLACK_HZ)
    sep = ("tone", spec.separator_ms, FREQ_BLACK_HZ)
    scans = spec.channel_times_ms
    family = spec.family
    if family == "martin":
        g, b, r = scans
        return [
            [sync, porch, ("G", g, 0.0), sep, ("B", b, 0.0), sep, ("R", r, 0.0), sep]
        ]
    if family == "scottie":
        g, b, r = scans
        return [[sep, ("G", g, 0.0), sep, ("B", b, 0.0), sync, porch, ("R", r, 0.0)]]
    if family == "robot36":
        y, c = scans
        chroma_porch = ("tone", _ROBOT_CHROMA_PORCH_MS, _ROBOT_CHROMA_PORCH_HZ)
        head = [sync, porch, ("Y", y, 0.0)]
        even = head + [
            ("parity", spec.separator_ms, FREQ_BLACK_HZ),
            chroma_porch,
            ("V", c, 0.0),
        ]
        odd = head + [
            ("parity", spec.separator_ms, FREQ_WHITE_HZ),
            chroma_porch,
            ("U", c, 0.0),
        ]
        return [even, odd]
    if family == "robot72":
        y, v, u = scans
        chroma_porch = ("tone", _ROBOT_CHROMA_PORCH_MS, _ROBOT_CHROMA_PORCH_HZ)
        return [
            [
                sync,
                porch,
                ("Y", y, 0.0),
                ("tone", spec.separator_ms, FREQ_BLACK_HZ),
                chroma_porch,
                ("V", v, 0.0),
                ("tone", spec.separator_ms, FREQ_WHITE_HZ),
                chroma_porch,
                ("U", u, 0.0),
            ]
        ]
    if family == "pd":
        y0, v, u, y1 = scans
        return [
            [
                sync,
                porch,
                ("Y0", y0, 0.0),
                ("V", v, 0.0),
                ("U", u, 0.0),
                ("Y1", y1, 0.0),
            ]
        ]
    if family == "wraase":
        r, g, b = scans
        return [[sync, porch, ("R", r, 0.0), ("G", g, 0.0), ("B", b, 0.0)]]
    raise ValueError(f"Unknown SSTV line layout: {family!r}")


def preamble_segments(spec: SSTVModeSpec) -> List[Segment]:
    """What comes between the VIS stop bit and the first line: Scottie's
    starting sync pulse (nothing for the other modes)."""
    if spec.family == "scottie":
        return [("sync", spec.sync_pulse_ms, FREQ_SYNC_HZ)]
    return []


def vis_segments(vis_code: int) -> List[Segment]:
    """The calibration header: leader, break, leader, then the VIS code
    (start bit, 7 data bits LSB first, even parity, stop bit)."""
    segments: List[Segment] = [
        ("tone", LEADER_MS, FREQ_LEADER_HZ),
        ("tone", BREAK_MS, FREQ_SYNC_HZ),
        ("tone", LEADER_MS, FREQ_LEADER_HZ),
        ("tone", VIS_BIT_MS, FREQ_SYNC_HZ),  # start bit
    ]
    ones = 0
    for bit in range(7):
        one = (int(vis_code) >> bit) & 1
        ones += one
        segments.append(
            ("tone", VIS_BIT_MS, FREQ_VIS_ONE_HZ if one else FREQ_VIS_ZERO_HZ)
        )
    segments.append(
        ("tone", VIS_BIT_MS, FREQ_VIS_ONE_HZ if ones % 2 else FREQ_VIS_ZERO_HZ)
    )
    segments.append(("tone", VIS_BIT_MS, FREQ_SYNC_HZ))  # stop bit
    return segments


def value_to_freq(values: np.ndarray) -> np.ndarray:
    """Pixel values (0-255) to tone frequencies (1500-2300 Hz)."""
    v = np.asarray(values, dtype=np.float64)
    return FREQ_BLACK_HZ + (FREQ_WHITE_HZ - FREQ_BLACK_HZ) * v / 255.0


def freq_to_value(freqs: np.ndarray) -> np.ndarray:
    """Tone frequencies to pixel values (float, clipped to 0-255)."""
    f = np.asarray(freqs, dtype=np.float64)
    return np.clip(
        (f - FREQ_BLACK_HZ) * (255.0 / (FREQ_WHITE_HZ - FREQ_BLACK_HZ)), 0, 255
    )


def rgb_to_ycc(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB to ITU-R BT.601 studio-range Y, R-Y (Cr) and B-Y (Cb), as floats."""
    rgb = np.asarray(image, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 16.0 + (65.738 * r + 129.057 * g + 25.064 * b) / 256.0
    cr = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0
    cb = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0
    return y, cr, cb


def ycc_to_rgb(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Studio-range Y, R-Y, B-Y to an ``(..., 3)`` uint8 RGB array (MMSSTV's
    conversion)."""
    y = np.asarray(y, dtype=np.float64) - 16.0
    cr = np.asarray(cr, dtype=np.float64) - 128.0
    cb = np.asarray(cb, dtype=np.float64) - 128.0
    r = 1.164457 * y + 1.596071 * cr
    g = 1.164457 * y - 0.813358 * cr - 0.392175 * cb
    b = 1.164457 * y + 2.017232 * cb
    rgb = np.stack((r, g, b), axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class _Slot:
    """A colour channel's scan, timed from the end of its line's sync."""

    channel: str
    start_ms: float
    scan_ms: float


@dataclass(frozen=True)
class _Layout:
    """Decoder view of a mode's line templates."""

    slots: Tuple[Tuple[_Slot, ...], ...]  # per template
    parity_ms: Optional[Tuple[float, float]]  # Robot 36 separator (start, len)
    before_ms: float  # earliest data a line needs, before its sync end
    after_ms: float  # latest data a line needs, after its sync end
    first_sync_end_ms: float  # from the VIS stop bit's end to sync 0's end


def _layout(spec: SSTVModeSpec) -> _Layout:
    templates = line_templates(spec)
    all_slots = []
    parity: Optional[Tuple[float, float]] = None
    before = 0.0
    after = 0.0
    sync_end = 0.0
    for template in templates:
        # Position of the sync pulse's end within the template.
        t = 0.0
        sync_end = 0.0
        for kind, dur, _freq in template:
            t += dur
            if kind == "sync":
                sync_end = t
        slots = []
        t = 0.0
        for kind, dur, _freq in template:
            start = t - sync_end
            if kind not in ("sync", "tone", "parity"):
                slots.append(_Slot(kind, start, dur))
            elif kind == "parity":
                parity = (start, dur)
            before = max(before, -start)
            after = max(after, start + dur)
            t += dur
        all_slots.append(tuple(slots))
    pre = sum(seg[1] for seg in preamble_segments(spec))
    return _Layout(tuple(all_slots), parity, before, after, pre + sync_end)


# --------------------------------------------------------------------------- #
# Signal processing helpers
# --------------------------------------------------------------------------- #

# Band of the analytic filter: the SSTV tones (1100-2300 Hz) with room for
# the picture's modulation.
_BAND_LOW_HZ = 700.0
_BAND_HIGH_HZ = 3000.0
_FILTER_MS = 1.5
# Audio kept for decoding (s): a line period (PD 290, Scottie DX: ~1 s) and
# the search margins fit comfortably.
_HISTORY_S = 2.5
# A sync pulse's trailing edge must score this well (0-1) to count, and this
# well to re-lock the line timing after a jump.
_SYNC_THRESHOLD = 0.5
_RELOCK_THRESHOLD = 0.65
# Pixels this close (ms) to a channel's start or end repeat their clean
# neighbour (the frequency track blurs across the edge).
_EDGE_MS = 0.3
# Lines without a sync before an image counts as lost (and at least this
# many seconds).
_LOST_LINES = 8
_LOST_S = 4.0


class _AnalyticFM:
    """Streaming instantaneous frequency of the SSTV tones.

    A complex band-pass FIR (a Kaiser-windowed low-pass shifted to the
    middle of the band) keeps only positive frequencies, which makes the
    analytic signal of the tones; the angle between consecutive samples is
    the instantaneous frequency. Filter and discriminator state carry over
    between blocks, so any block size gives the same track.
    """

    def __init__(self, sample_rate: float):
        fs = float(sample_rate)
        high = min(_BAND_HIGH_HZ, 0.45 * fs)
        low = min(_BAND_LOW_HZ, 0.5 * high)
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        ntaps = max(15, int(round(_FILTER_MS * fs / 1000.0)) | 1)
        n = np.arange(ntaps) - (ntaps - 1) / 2.0
        lowpass = np.sinc(2.0 * half * n / fs) * np.kaiser(ntaps, 5.0)
        lowpass /= np.sum(lowpass)
        # Gain 2: the analytic signal of A cos(wt) has magnitude A.
        taps = 2.0 * lowpass * np.exp(2j * np.pi * center * n / fs)
        self._taps_re = np.ascontiguousarray(taps.real)
        self._taps_im = np.ascontiguousarray(taps.imag)
        self._ntaps = ntaps
        self._scale = fs / (2.0 * np.pi)
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(self._ntaps - 1, dtype=np.float64)
        self._last = 0j

    def process(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(frequency Hz, |analytic|^2)`` for each sample of ``x``."""
        buf = np.concatenate((self._history, np.asarray(x, dtype=np.float64)))
        self._history = buf[len(buf) - (self._ntaps - 1) :].copy()
        re = np.convolve(buf, self._taps_re, mode="valid")
        im = np.convolve(buf, self._taps_im, mode="valid")
        z = re + 1j * im
        prev = np.empty_like(z)
        prev[0] = self._last
        prev[1:] = z[:-1]
        self._last = complex(z[-1])
        freq = np.angle(z * np.conj(prev)) * self._scale
        return freq.astype(np.float32), (re * re + im * im).astype(np.float32)


class _Track:
    """A bounded series with absolute indexing (old values are dropped).

    Values live in a growable buffer, so appending a block costs only that
    block, and dropping old values moves the rest once in a while.
    """

    def __init__(self, dtype: Any = np.float32):
        self._buf = np.zeros(4096, dtype=dtype)
        self._len = 0
        self.start = 0

    @property
    def data(self) -> np.ndarray:
        return self._buf[: self._len]

    @property
    def end(self) -> int:
        """Absolute index after the last value."""
        return self.start + self._len

    def append(self, values: np.ndarray) -> None:
        n = len(values)
        need = self._len + n
        if need > len(self._buf):
            grown = np.zeros(max(need, 2 * len(self._buf)), dtype=self._buf.dtype)
            grown[: self._len] = self._buf[: self._len]
            self._buf = grown
        self._buf[self._len : need] = values
        self._len = need

    def get(self, a: int, b: int) -> np.ndarray:
        """Values ``a:b`` (absolute), clipped to what is held (a view)."""
        a = max(int(a), self.start)
        b = min(int(b), self.end)
        if b <= a:
            return self._buf[:0]
        return self._buf[a - self.start : b - self.start]

    def keep_from(self, index: int) -> None:
        """Drop everything before absolute ``index``."""
        drop = min(int(index) - self.start, self._len)
        if drop > 0:
            rest = self._len - drop
            self._buf[:rest] = self._buf[drop : self._len]
            self._len = rest
            self.start += drop


class _SyncFit:
    """Where line period ``k``'s sync pulse ends.

    An exponentially weighted least-squares line through the measured sync
    ends, with a weak prior that pulls the slope to the expected period: it
    follows a sound card's sample-rate error (a picture would otherwise
    slant) and averages out jitter, and coasts through lost syncs.
    """

    _FORGET = 0.985  # per measured line: remembers ~70 lines
    _PRIOR = 0.05  # weight of the expected period against the data

    def __init__(self, period: float):
        self.prior_slope = float(period)
        self.reset()

    def reset(self, slope: Optional[float] = None) -> None:
        if slope is not None:
            self.prior_slope = float(slope)
        self.count = 0
        self._k0 = 0
        self._y0 = 0.0
        self._s = np.zeros(5)  # sum w, w k, w k^2, w y, w k y

    def add(self, k: int, y: float) -> None:
        if self.count == 0:
            self._k0, self._y0 = int(k), float(y)
        dk, dy = float(k - self._k0), float(y) - self._y0
        self._s *= self._FORGET
        self._s += (1.0, dk, dk * dk, dy, dk * dy)
        self.count += 1

    def _solve(self) -> Tuple[float, float]:
        s0, s1, s2, t0, t1 = self._s
        mu = self._PRIOR
        det = s0 * (s2 + mu) - s1 * s1
        if det <= 1e-12:
            return t0 / max(s0, 1e-12), self.prior_slope
        a = (t0 * (s2 + mu) - s1 * (t1 + mu * self.prior_slope)) / det
        b = (s0 * (t1 + mu * self.prior_slope) - s1 * t0) / det
        return a, b

    @property
    def slope(self) -> float:
        """Samples per line period."""
        if self.count == 0:
            return self.prior_slope
        return self._solve()[1]

    def predict(self, k: int) -> float:
        a, b = self._solve()
        return self._y0 + a + b * (k - self._k0)


@dataclass
class SSTVState:
    """Current state of SSTV decoding."""

    mode: Optional[SSTVModeSpec] = None
    is_receiving: bool = False
    current_line: int = 0  # image lines decoded so far
    line_buffer: List[float] = field(default_factory=list)  # unused, kept for API
    image_data: Optional[np.ndarray] = None
    start_time: float = 0.0
    last_sync_time: float = 0.0
    signal_strength: float = 0.0  # share of the audio in the SSTV band, 0-1
    vis_detected: bool = False
    complete: bool = False
    vis_code: Optional[int] = None
    # The whole calibration header was heard: leader, 10 ms break, leader
    # (the first leader and the break often fade; the VIS alone is enough).
    vis_header: bool = False
    sync_locked: bool = False


@dataclass
class _Reception:
    """Per-image decoding state."""

    spec: SSTVModeSpec
    layout: _Layout
    fit: _SyncFit
    period: float  # nominal samples per sync period
    next_k: int = 0
    first_end: Optional[float] = None  # predicted end of sync 0 (VIS start)
    acquire_from: Optional[int] = None  # start now: search from here
    misses: int = 0
    rows: int = 0
    last_good_rows: int = 0
    relock: List[Tuple[int, float]] = field(default_factory=list)
    chroma: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    luma: Dict[int, np.ndarray] = field(default_factory=dict)


# Line periods searched for when guessing a mode from sync pulses alone.
def _sync_guess_modes() -> List[SSTVModeSpec]:
    return sorted(SSTV_MODES.values(), key=lambda s: s.scan_time_ms)


class SSTVDecoder:
    """
    SSTV (Slow Scan Television) decoder.

    Decodes SSTV audio signals into images. Commonly used for receiving
    images from the International Space Station.

    Usage:
        decoder = SSTVDecoder(sample_rate=48000)

        # Feed audio samples (from FM demodulator output), any block size
        decoder.process_audio(audio_samples)

        # Check if image is complete
        if decoder.is_complete():
            image = decoder.get_image()
            decoder.save_image("received_image.png")

    Memory stays bounded however long it listens: only the last
    ``_HISTORY_S`` seconds of the frequency track are kept.
    """

    # SSTV frequency constants (Hz)
    FREQ_BLACK = 1500  # Black level
    FREQ_WHITE = 2300  # White level
    FREQ_SYNC = 1200  # Sync pulse
    FREQ_VIS_BIT_0 = 1300  # VIS bit 0
    FREQ_VIS_BIT_1 = 1100  # VIS bit 1
    FREQ_LEADER = 1900  # Leader tone
    FREQ_BREAK = 1200  # Break tone

    # Timing constants (ms)
    VIS_BIT_TIME = 30.0  # Each VIS bit duration
    LEADER_TIME = 300.0  # Leader tone duration

    def __init__(self, sample_rate: float = 48000.0):
        """
        Initialize SSTV decoder.

        Args:
            sample_rate: Audio sample rate in Hz (default: 48000)
        """
        self.sample_rate = float(sample_rate)
        self._spm = self.sample_rate / 1000.0  # samples per millisecond
        self._fm = _AnalyticFM(self.sample_rate)
        self._on_line_decoded: Optional[Callable[[int, np.ndarray], None]] = None
        self._on_image_complete: Optional[Callable[[np.ndarray], None]] = None
        self._on_mode_detected: Optional[Callable[[SSTVModeSpec], None]] = None
        self._on_image_lost: Optional[Callable[[np.ndarray, int], None]] = None
        self._forced_mode: Optional[SSTVModeSpec] = None
        self.reset()
        logger.info(f"SSTVDecoder initialized at {sample_rate} Hz sample rate")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset decoder state for new reception."""
        self.state = SSTVState()
        self._fm.reset()
        self._freq = _Track(np.float32)  # per-sample frequency (Hz)
        self._coarse = _Track(np.float32)  # 1 ms means of the frequency
        self._coarse_next = 0  # next 1 ms bin to compute
        self._rx: Optional[_Reception] = None
        self._vis_checked = 0  # coarse index up to which edges were looked at
        self._vis_pending: List[int] = []  # start-bit edges awaiting data
        self._vis_lockout = 0  # no new VIS before this coarse index
        self._pulse_checked = 0
        self._pulses: Deque[Tuple[int, int]] = deque(maxlen=12)  # (end ms, length)
        self._run_start: Optional[int] = None
        self._suppress_auto_until_quiet = False
        self._level = 0.0  # RMS of the latest audio
        self._in_band = 0.0  # share of that audio in the SSTV band
        self._tone_hz: Optional[float] = None
        self._activity = "none"
        # Why the last image ended early ("the signal faded", "a new image
        # started"), for the on_image_lost callback's caller.
        self.lost_reason = ""
        logger.debug("SSTV decoder reset")

    def set_on_line_decoded(self, callback: Callable[[int, np.ndarray], None]) -> None:
        """Set callback for when each line is decoded."""
        self._on_line_decoded = callback

    def set_on_image_complete(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for when image is complete."""
        self._on_image_complete = callback

    def set_on_mode_detected(self, callback: Callable[[SSTVModeSpec], None]) -> None:
        """Set callback for when SSTV mode is detected."""
        self._on_mode_detected = callback

    def set_on_image_lost(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """Set callback for an image that ended early: the signal faded, or a
        new VIS started another image. Gets ``(image, lines received)``."""
        self._on_image_lost = callback

    def set_forced_mode(self, mode: Optional[SSTVModeSpec]) -> None:
        """Decode every image in ``mode`` whatever its VIS says (None: the
        VIS decides). With a mode set, a train of line syncs that matches
        it also starts an image when the VIS was missed."""
        self._forced_mode = mode

    def get_forced_mode(self) -> Optional[SSTVModeSpec]:
        return self._forced_mode

    def process_audio(self, samples: np.ndarray) -> None:
        """
        Process audio samples from FM demodulator.

        Args:
            samples: Audio samples (float, mono), any block size
        """
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(x):
            return
        if not np.isfinite(x).all():
            # One NaN would stay in the filter history for good.
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        step = max(1, int(0.5 * self.sample_rate))
        for first in range(0, len(x), step):
            self._process_chunk(x[first : first + step])

    def start_now(self, mode: Optional[SSTVModeSpec] = None) -> bool:
        """Start an image now, from the line syncs, without a VIS.

        ``mode`` defaults to the forced mode, then to the mode the recent
        sync pulses fit. Returns False (and does nothing) when no mode is
        known.
        """
        spec = mode or self._forced_mode
        if spec is None:
            guess = self.guess_mode_from_syncs()
            spec = guess[0] if guess else None
        if spec is None:
            return False
        self._begin(spec, vis_code=None, first_end=None)
        return True

    def stop(self) -> int:
        """End the image being received (if any); returns its line count."""
        rx = self._rx
        rows = rx.rows if rx else 0
        self._end_reception()
        return rows

    def is_receiving(self) -> bool:
        """Check if currently receiving an image."""
        return self.state.is_receiving

    def is_complete(self) -> bool:
        """Check if image reception is complete."""
        return self.state.complete

    def get_progress(self) -> float:
        """
        Get reception progress.

        Returns:
            Progress as fraction 0.0 to 1.0
        """
        if self.state.mode is None:
            return 0.0
        return min(1.0, self.state.current_line / self.state.mode.height)

    def get_image(self) -> Optional[np.ndarray]:
        """
        Get the decoded image.

        Returns:
            RGB image as numpy array (H, W, 3) or None
        """
        return self.state.image_data

    def get_mode(self) -> Optional[SSTVModeSpec]:
        """Get detected SSTV mode."""
        return self.state.mode

    def get_status(self) -> dict:
        """Get decoder status."""
        tone = self.get_tone()
        return {
            "mode": self.state.mode.name if self.state.mode else None,
            "is_receiving": self.state.is_receiving,
            "complete": self.state.complete,
            "current_line": self.state.current_line,
            "total_lines": self.state.mode.height if self.state.mode else 0,
            "progress": self.get_progress(),
            "signal_strength": self.state.signal_strength,
            "sync_locked": self.state.sync_locked,
            "vis_code": self.state.vis_code,
            "vis_header": self.state.vis_header,
            "tone_hz": tone["tone_hz"],
            "activity": tone["activity"],
        }

    def get_tone(self) -> Dict[str, Any]:
        """What the decoder hears right now, for a tone meter.

        ``tone_hz``: the dominant tone over the last ~50 ms (None for noise
        or silence); ``in_band``: share of the audio power in the SSTV band
        (0-1); ``level``: audio RMS; ``activity``: "none" (silence),
        "noise", "leader", "vis", "sync", "picture", "tone" or "image"
        (receiving); ``text``: a short description; ``mode_guess``: the
        mode the recent sync pulses fit, or None.
        """
        guess = self.guess_mode_from_syncs()
        text = self._activity_text(guess)  # also sets _activity
        return {
            "tone_hz": self._tone_hz,
            "in_band": self._in_band,
            "level": self._level,
            "activity": self._activity,
            "text": text,
            "mode_guess": guess[0] if guess else None,
        }

    def guess_mode_from_syncs(self) -> Optional[Tuple[SSTVModeSpec, float]]:
        """``(mode, period ms)`` that the recent sync pulses fit, if any."""
        pulses = list(self._pulses)
        if len(pulses) < 4:
            return None
        now = self._coarse.end
        if now - pulses[-1][0] > 2500:
            return None  # stale
        ends = np.array([p[0] for p in pulses], dtype=float)
        lengths = np.array([p[1] for p in pulses], dtype=float)
        gaps = np.diff(ends)[-6:]
        best: Optional[Tuple[SSTVModeSpec, float]] = None
        best_err = 1e9
        for spec in _sync_guess_modes():
            period = spec.scan_time_ms
            ratio = gaps / period
            whole = np.rint(ratio)
            ok = (whole >= 1) & (whole <= 2) & (np.abs(gaps - whole * period) <= 2.5)
            if np.count_nonzero(ok) < 3 or np.count_nonzero(whole[ok] == 1) < 2:
                continue
            if abs(np.median(lengths[-len(gaps) :]) - spec.sync_pulse_ms) > max(
                2.5, 0.45 * spec.sync_pulse_ms
            ):
                continue
            err = float(np.mean(np.abs(gaps[ok] - whole[ok] * period)))
            if err < best_err:
                best, best_err = (spec, period), err
        return best

    def save_image(self, path: str) -> bool:
        """
        Save decoded image to file.

        Args:
            path: Output file path (PNG, JPG, etc.)

        Returns:
            True if saved successfully
        """
        if self.state.image_data is None:
            logger.error("No image data to save")
            return False

        try:
            # Try PIL first
            try:
                from PIL import Image

                img = Image.fromarray(self.state.image_data, "RGB")
                img.save(path)
                logger.info(f"Saved SSTV image to {path}")
                return True
            except ImportError:
                pass

            # Fallback to matplotlib
            try:
                import matplotlib.pyplot as plt

                plt.imsave(path, self.state.image_data)
                logger.info(f"Saved SSTV image to {path}")
                return True
            except ImportError:
                pass

            # Fallback to raw numpy save
            np.save(
                path.replace(".png", ".npy").replace(".jpg", ".npy"),
                self.state.image_data,
            )
            logger.info(
                f"Saved SSTV image data to {path}.npy (install PIL for image format)"
            )
            return True

        except (OSError, ValueError) as e:
            logger.error(f"Failed to save image: {e}")
            return False

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _process_chunk(self, x: np.ndarray) -> None:
        freq, power = self._fm.process(x)
        self._freq.append(freq)
        self._update_levels(x, freq, power)
        self._update_coarse()
        self._scan_pulses()
        self._scan_vis()
        if self._rx is not None:
            self._decode_available()
        self._maybe_auto_start()
        self._trim()

    def _update_levels(
        self, x: np.ndarray, freq: np.ndarray, power: np.ndarray
    ) -> None:
        """Audio level, in-band share and dominant tone of the last ~50 ms."""
        n = min(len(x), max(1, int(50 * self._spm)))
        tail = x[-n:].astype(np.float64)
        total = float(np.mean(tail * tail))
        self._level = math.sqrt(total)
        band = float(np.mean(power[-n:])) / 2.0  # analytic |z|^2 = 2 x mean x^2
        self._in_band = 0.0 if total <= 1e-12 else min(1.0, band / total)
        self.state.signal_strength = self._in_band
        if self._level < 1e-4 or self._in_band < 0.3:
            self._tone_hz = None
        else:
            recent = freq[-n:]
            self._tone_hz = float(np.median(recent))

    def _update_coarse(self) -> None:
        """Extend the 1 ms track (means of whole milliseconds of samples)."""
        spm = self._spm
        last = int((self._freq.end) // spm)  # bins whose samples all exist
        first = self._coarse_next
        if last <= first:
            return
        edges = np.rint(np.arange(first, last + 1) * spm).astype(np.int64)
        lo = int(edges[0])
        seg = self._freq.get(lo, int(edges[-1])).astype(np.float64)
        if len(seg) < int(edges[-1]) - lo:  # history trimmed (cannot happen)
            self._coarse_next = last
            return
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        rel = edges - lo
        means = (cs[rel[1:]] - cs[rel[:-1]]) / np.maximum(np.diff(rel), 1)
        if self._coarse.end != first:
            self._coarse.start = first - len(self._coarse.data)
        self._coarse.append(means.astype(np.float32))
        self._coarse_next = last

    def _trim(self) -> None:
        """Keep the last _HISTORY_S seconds (more only while a line needs it)."""
        keep = int(_HISTORY_S * self.sample_rate)
        start = self._freq.end - keep
        rx = self._rx
        if rx is not None and rx.acquire_from is not None:
            start = min(start, rx.acquire_from)
        elif rx is not None:
            need = self._predicted_end(rx, rx.next_k)
            if need is not None:
                start = min(start, int(need - (rx.layout.before_ms + 50.0) * self._spm))
        # Hard limit, whatever a reception wants.
        start = max(start, self._freq.end - 3 * keep)
        if start - self._freq.start > int(0.25 * self.sample_rate):
            self._freq.keep_from(start)
        coarse_start = self._coarse.end - int(_HISTORY_S * 1000)
        if coarse_start - self._coarse.start > 250:
            self._coarse.keep_from(coarse_start)

    # ------------------------------------------------------------------
    # Sync pulses heard while idle (mode guess, "Start now", auto start)
    # ------------------------------------------------------------------

    def _scan_pulses(self) -> None:
        """Note 3-25 ms runs below 1350 Hz (line sync candidates)."""
        coarse = self._coarse
        first = max(self._pulse_checked, coarse.start + 1)
        values = coarse.get(first - 1, coarse.end)
        if len(values) < 3:
            return
        low = values < 1350.0
        # Runs of `low`: starts where it rises, ends where it falls.
        change = np.diff(low.astype(np.int8))
        starts = list(np.nonzero(change == 1)[0] + first)
        ends = list(np.nonzero(change == -1)[0] + first)
        run_start = self._run_start
        events = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends])
        for index, kind in events:
            if kind == 1:
                run_start = index
            elif run_start is not None:
                length = index - run_start
                if 3 <= length <= 25:
                    self._pulses.append((int(index), int(length)))
                run_start = None
        self._run_start = run_start
        self._pulse_checked = coarse.end

    def _maybe_auto_start(self) -> None:
        """With a forced mode, a matching sync train starts an image."""
        if self._rx is not None or self._forced_mode is None:
            return
        guess = self.guess_mode_from_syncs()
        if guess is None:
            self._suppress_auto_until_quiet = False
            return
        if self._suppress_auto_until_quiet:
            return
        if guess[0].name == self._forced_mode.name:
            logger.info(f"SSTV: {guess[0].name} line syncs without a VIS; starting")
            self._begin(guess[0], vis_code=None, first_end=None)

    # ------------------------------------------------------------------
    # VIS header
    # ------------------------------------------------------------------

    def _scan_vis(self) -> None:
        coarse = self._coarse
        first = max(self._vis_checked, coarse.start + 120, self._vis_lockout)
        base = max(coarse.start, first - 101)
        values = coarse.get(base, coarse.end)
        if coarse.end - first >= 1 and len(values) > first - base:
            # Leader (1900 Hz) falling into the start bit (1200 Hz): edges
            # where at least 60% of the 95 ms before are within 100 Hz of
            # 1900 Hz. (Noise crosses 1550 Hz all the time, and its median
            # sits near 1900 Hz, but it is rarely that steady.)
            edges = np.nonzero((values[:-1] >= 1550.0) & (values[1:] < 1550.0))[0] + 1
            edges = edges[edges >= max(first - base, 100)]
            if len(edges):
                steady = np.concatenate(
                    ([0], np.cumsum(np.abs(values - FREQ_LEADER_HZ) < 100.0))
                )
                share = (steady[edges - 5] - steady[edges - 100]) / 95.0
                for e in edges[share >= 0.6]:
                    self._vis_pending.append(int(base + e))
            self._vis_checked = coarse.end
        pending = []
        for edge in self._vis_pending:
            if coarse.end < edge + 300:
                pending.append(edge)
                continue
            code = self._read_vis(edge)
            if code is not None and edge >= self._vis_lockout:
                self._on_vis(edge, code)
        self._vis_pending = [e for e in pending if e >= self._vis_lockout]

    def _read_vis(self, edge: int) -> Optional[int]:
        """The VIS code whose start bit begins at coarse index ``edge``."""
        coarse = self._coarse

        def median(a: int, b: int) -> float:
            seg = coarse.get(edge + a, edge + b)
            return float(np.median(seg)) if len(seg) else 0.0

        if abs(median(5, 25) - FREQ_SYNC_HZ) > 80.0:
            return None  # start bit
        if abs(median(275, 295) - FREQ_SYNC_HZ) > 80.0:
            return None  # stop bit
        bits = []
        for b in range(8):
            f = median(30 * (b + 1) + 5, 30 * (b + 2) - 5)
            if abs(f - FREQ_VIS_ONE_HZ) < 90.0:
                bits.append(1)
            elif abs(f - FREQ_VIS_ZERO_HZ) < 90.0:
                bits.append(0)
            else:
                return None
        if sum(bits[:7]) % 2 != bits[7]:
            logger.debug("SSTV VIS parity error")
            return None
        code = sum(bit << i for i, bit in enumerate(bits[:7]))
        if code not in SSTV_MODES:
            logger.debug(f"SSTV VIS code {code} is not a supported mode")
            return None
        return code

    def _heard_break(self, edge: int) -> bool:
        """The 10 ms 1200 Hz break between the two leaders, 300 ms before
        the start bit at coarse index ``edge``, with the first leader."""
        coarse = self._coarse
        gap = coarse.get(edge - 308, edge - 302)
        lead = coarse.get(edge - 560, edge - 320)
        if len(gap) < 4 or len(lead) < 120:
            return False
        return (
            abs(float(np.median(gap)) - FREQ_SYNC_HZ) < 100.0
            and abs(float(np.median(lead)) - FREQ_LEADER_HZ) < 80.0
        )

    def _on_vis(self, edge: int, code: int) -> None:
        # Refine the start bit's leading edge on the full-rate track.
        spm = self._spm
        lo = int((edge - 4) * spm)
        seg = self._freq.get(lo, int((edge + 3) * spm)).astype(np.float64)
        refined = edge * spm
        if len(seg) > 8:
            width = max(1, int(round(0.5 * spm)))
            smooth = np.convolve(seg, np.ones(width) / width, mode="same")
            cross = np.nonzero((smooth[:-1] >= 1550.0) & (smooth[1:] < 1550.0))[0]
            if len(cross):
                i = int(cross[-1])
                a, b = smooth[i], smooth[i + 1]
                frac = (a - 1550.0) / (a - b) if a != b else 0.0
                refined = lo + i + frac
        vis_end = refined + 10 * VIS_BIT_MS * spm
        spec = self._forced_mode or SSTV_MODES[code]
        header = self._heard_break(edge)
        logger.info(
            f"SSTV VIS {code} ({SSTV_MODES[code].name}"
            f"{', full header' if header else ', break not heard'}); "
            f"decoding {spec.name}"
        )
        if self._rx is not None:
            self._lose_image("a new image started")
        self._begin(spec, vis_code=code, first_end=vis_end)
        self.state.vis_header = header
        self._vis_lockout = edge + 300

    # ------------------------------------------------------------------
    # Reception
    # ------------------------------------------------------------------

    def _begin(
        self, spec: SSTVModeSpec, vis_code: Optional[int], first_end: Optional[float]
    ) -> None:
        layout = _layout(spec)
        period = spec.scan_time_ms * self._spm
        rx = _Reception(spec=spec, layout=layout, fit=_SyncFit(period), period=period)
        if first_end is not None:
            rx.first_end = first_end + layout.first_sync_end_ms * self._spm
        else:
            # Search the most recent line period that a line fits around.
            rx.acquire_from = max(
                self._freq.start,
                self._freq.end - int((spec.scan_time_ms + 100.0) * self._spm),
            )
        self._rx = rx
        self.state = SSTVState(
            mode=spec,
            is_receiving=True,
            image_data=np.zeros((spec.height, spec.width, 3), dtype=np.uint8),
            start_time=time.time(),
            vis_detected=vis_code is not None,
            vis_code=vis_code,
            signal_strength=self._in_band,
        )
        self._pulses.clear()
        logger.info(f"SSTV mode detected: {spec.name} ({spec.width}x{spec.height})")
        if self._on_mode_detected:
            self._on_mode_detected(spec)

    def _predicted_end(self, rx: _Reception, k: int) -> Optional[float]:
        if rx.fit.count:
            return rx.fit.predict(k)
        if rx.first_end is not None:
            return rx.first_end + k * rx.period
        return None

    def _decode_available(self) -> None:
        """Decode every line period whose audio has arrived."""
        rx = self._rx
        spm = self._spm
        while rx is not None and self._rx is rx:
            if rx.acquire_from is not None:
                if not self._acquire(rx):
                    return
                continue
            k = rx.next_k
            predicted = self._predicted_end(rx, k)
            if predicted is None:
                return
            tol = max(2.0 * spm, 0.006 * rx.period)
            wide = rx.misses >= 2
            half = 0.5 * rx.period if wide else max(2.5 * tol, 6.0 * spm)
            need = predicted + max(half, rx.layout.after_ms * spm) + 4.0 * spm
            if self._freq.end < need:
                return
            found, score = self._find_sync_end(predicted - half, predicted + half, rx)
            end = predicted
            if found is not None and score >= _SYNC_THRESHOLD:
                residual = found - predicted
                if abs(residual) <= tol or rx.fit.count == 0:
                    rx.fit.add(k, found)
                    rx.misses = 0
                    rx.relock.clear()
                    end = rx.fit.predict(k)
                else:
                    rx.misses += 1
                    if score >= _RELOCK_THRESHOLD:
                        self._try_relock(rx, k, found)
                        if rx.fit.count and not rx.relock:
                            end = rx.fit.predict(k)
            else:
                rx.misses += 1
            self.state.sync_locked = rx.misses == 0
            if rx.misses == 0:
                self.state.last_sync_time = time.time()
            self._decode_period(rx, k, end)
            if rx.misses == 0:
                rx.last_good_rows = rx.rows
            rx.next_k += 1
            if rx.next_k >= rx.spec.sync_periods:
                self._finish_image()
                return
            lost_after = max(
                _LOST_LINES, int(math.ceil(_LOST_S * 1000.0 / rx.spec.scan_time_ms))
            )
            if rx.misses >= lost_after:
                self._lose_image("the signal faded")
                return

    def _try_relock(self, rx: _Reception, k: int, found: float) -> None:
        """After a jump in the audio (lost samples), strong syncs keep
        turning up away from the prediction: three that agree re-anchor the
        line timing on them."""
        slope = rx.fit.slope
        rx.relock.append((k, found))
        rx.relock = rx.relock[-3:]
        if len(rx.relock) < 3:
            return
        offsets = [y - slope * (kk - k) for kk, y in rx.relock]
        if max(offsets) - min(offsets) <= 1.5 * self._spm:
            rx.fit.reset(slope)
            for kk, y in rx.relock:
                rx.fit.add(kk, y)
            rx.relock.clear()
            rx.misses = 0
            logger.info("SSTV: line timing re-locked")

    def _acquire(self, rx: _Reception) -> bool:
        """Start now: find the line phase from the strongest sync in the most
        recent period of audio. Returns True once found."""
        spm = self._spm
        before = rx.layout.before_ms * spm
        after = rx.layout.after_ms * spm
        lo = max(rx.acquire_from or 0, self._freq.start) + before + 2 * spm
        hi = lo + rx.period
        if self._freq.end < hi + after + 4 * spm:
            return False
        found, score = self._find_sync_end(lo, hi, rx)
        if found is None or score < _RELOCK_THRESHOLD:
            # Nothing sync-like yet: slide on by half a period.
            rx.acquire_from = int(lo - before - 2 * spm + 0.5 * rx.period)
            return False
        rx.acquire_from = None
        rx.first_end = found
        rx.fit.add(0, found)
        return True

    def _find_sync_end(
        self, lo: float, hi: float, rx: _Reception
    ) -> Tuple[Optional[float], float]:
        """Best sync trailing edge between ``lo`` and ``hi`` (sample index)
        and its score (1 for a clean edge).

        The score is the mean "sync-ness" (1 at 1200 Hz, 0 at 1500 Hz and
        above) over most of a pulse length before the edge, minus the same
        length after it, where the porch and the picture are: with equal
        windows the best score sits where the blurred edge is half-way, at
        the true edge, so the picture is not shifted.
        """
        spm = self._spm
        length = max(2, int(round(rx.spec.sync_pulse_ms * 0.85 * spm)))
        a = int(math.floor(lo)) - length
        b = int(math.ceil(hi)) + length
        seg = self._freq.get(a, b).astype(np.float64)
        if len(seg) < 2 * length + 2:
            return None, 0.0
        a = max(a, self._freq.start)
        syncness = np.clip(
            (FREQ_BLACK_HZ - seg) / (FREQ_BLACK_HZ - FREQ_SYNC_HZ), 0.0, 1.0
        )
        cs = np.concatenate(([0.0], np.cumsum(syncness)))
        # Edge between samples t-1 and t, for t in [length, len - length].
        t = np.arange(length, len(seg) - length + 1)
        score = (2.0 * cs[t] - cs[t - length] - cs[t + length]) / length
        best = int(np.argmax(score))
        top = float(score[best])
        # Sub-sample position: the vertex of a parabola through the peak.
        frac = 0.0
        if 0 < best < len(score) - 1:
            left, right = score[best - 1], score[best + 1]
            denom = left - 2.0 * top + right
            if denom < -1e-12:
                frac = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
        return a + float(t[best]) + frac - 0.5, top

    def _box_means(self, start: float, step: float, count: int) -> np.ndarray:
        """Mean frequency over ``count`` consecutive slots of ``step``
        samples from ``start`` (sample index; sample j covers j-0.5..j+0.5)."""
        stop = start + step * count
        a = int(math.floor(start)) - 1
        b = int(math.ceil(stop)) + 2
        seg = self._freq.get(a, b).astype(np.float64)
        a = max(a, self._freq.start)
        if len(seg) < 2:
            return np.full(count, FREQ_BLACK_HZ)
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        edges = start + step * np.arange(count + 1) - a + 0.5
        edges = np.clip(edges, 0.0, float(len(seg)))
        integral = np.interp(edges, np.arange(len(cs), dtype=np.float64), cs)
        return np.diff(integral) / max(step, 1e-9)

    def _decode_period(self, rx: _Reception, k: int, end: float) -> None:
        """Pixels of sync period ``k``, whose sync ends at ``end``."""
        spec = rx.spec
        spm = self._spm * (rx.fit.slope / rx.period if rx.fit.count >= 2 else 1.0)
        templates = rx.layout.slots
        index = k % len(templates)
        if rx.layout.parity_ms is not None:
            start, length = rx.layout.parity_ms
            f = float(
                np.median(self._box_means(end + start * spm, length * spm / 3, 3))
            )
            if abs(f - 1900.0) > 150.0:
                index = 0 if f < 1900.0 else 1
        channels: Dict[str, np.ndarray] = {}
        for slot in templates[index]:
            step = slot.scan_ms * spm / spec.width
            means = self._box_means(end + slot.start_ms * spm, step, spec.width)
            # The frequency track blurs ~0.3 ms across a channel's ends (into
            # the sync or porch next to it): those pixels repeat their
            # nearest clean neighbour instead of showing a coloured border.
            edge = int(math.ceil(_EDGE_MS * spec.width / slot.scan_ms))
            if 0 < edge < spec.width // 8:
                means[:edge] = means[edge]
                means[-edge:] = means[-edge - 1]
            channels[slot.channel] = freq_to_value(means)
        image = self.state.image_data
        if image is None:
            return
        family = spec.family
        rows: List[int] = []
        if family in ("martin", "scottie", "wraase"):
            row = k
            if row < spec.height:
                image[row] = (
                    np.stack((channels["R"], channels["G"], channels["B"]), axis=-1)
                    .round()
                    .astype(np.uint8)
                )
                rows.append(row)
        elif family == "robot72":
            row = k
            if row < spec.height:
                image[row] = ycc_to_rgb(channels["Y"], channels["V"], channels["U"])
                rows.append(row)
        elif family == "pd":
            for row, key in zip((2 * k, 2 * k + 1), ("Y0", "Y1"), strict=True):
                if row < spec.height:
                    image[row] = ycc_to_rgb(channels[key], channels["V"], channels["U"])
                    rows.append(row)
        elif family == "robot36":
            row = k
            if row < spec.height:
                rx.luma[row] = channels["Y"]
                key = "V" if "V" in channels else "U"
                pair = row // 2
                rx.chroma.setdefault(pair, {})[key] = channels[key]
                rows = self._robot36_rows(rx, pair, row)
                # Only the last two pairs are ever needed again.
                for old in [p for p in rx.chroma if p < pair - 1]:
                    rx.chroma.pop(old, None)
                for old in [r for r in rx.luma if r < row - 3]:
                    rx.luma.pop(old, None)
        for row in rows:
            rx.rows = max(rx.rows, row + 1)
        self.state.current_line = rx.rows
        if self._on_line_decoded:
            for row in rows:
                self._on_line_decoded(row, image[row])

    def _robot36_rows(self, rx: _Reception, pair: int, row: int) -> List[int]:
        """Robot 36 rows to (re)draw now that ``row`` arrived: both rows of
        its pair share one R-Y and one B-Y (taken from the neighbouring pair
        until the pair is complete)."""
        image = self.state.image_data
        if image is None:
            return []
        neutral = np.full(rx.spec.width, 128.0)

        def chroma(key: str) -> np.ndarray:
            for p in (pair, pair - 1):
                values = rx.chroma.get(p, {}).get(key)
                if values is not None:
                    return values
            return neutral

        v, u = chroma("V"), chroma("U")
        rows = []
        for r in (2 * pair, 2 * pair + 1):
            luma = rx.luma.get(r)
            if luma is not None and r < rx.spec.height and r <= row:
                image[r] = ycc_to_rgb(luma, v, u)
                rows.append(r)
        return rows

    def _finish_image(self) -> None:
        """Mark image as complete."""
        rx = self._rx
        self._rx = None
        self.state.complete = True
        self.state.is_receiving = False
        self.state.sync_locked = False
        duration = time.time() - self.state.start_time
        logger.info(
            f"SSTV image complete: {self.state.current_line} lines in {duration:.1f}s"
        )
        # The syncs heard during this image must not restart it.
        self._suppress_auto_until_quiet = rx is not None
        if self._on_image_complete and self.state.image_data is not None:
            self._on_image_complete(self.state.image_data)

    def _lose_image(self, why: str) -> None:
        """End an image early (keeping the lines that had a signal)."""
        rx = self._rx
        self._rx = None
        self.state.is_receiving = False
        self.state.sync_locked = False
        if rx is None:
            return
        rows = rx.last_good_rows if why == "the signal faded" else rx.rows
        self.lost_reason = why
        logger.info(f"SSTV image ended early ({why}) after {rows} lines")
        self.state.current_line = rows
        self._suppress_auto_until_quiet = True
        image = self.state.image_data
        if image is not None:
            image[rows:] = 0  # lines decoded after the signal faded are noise
        if self._on_image_lost and image is not None and rows > 0:
            self._on_image_lost(image.copy(), rows)

    def _end_reception(self) -> None:
        self._rx = None
        self.state.is_receiving = False
        self.state.sync_locked = False
        self._suppress_auto_until_quiet = True

    # ------------------------------------------------------------------
    # Activity (tone meter)
    # ------------------------------------------------------------------

    def _activity_text(self, guess: Optional[Tuple[SSTVModeSpec, float]]) -> str:
        """Short description of what is heard, and sets ``_activity``."""
        tone = self._tone_hz
        if self.state.is_receiving and self._rx is not None:
            self._activity = "image"
            if self._rx.acquire_from is not None:
                return "Looking for the line sync…"
            if self.state.sync_locked:
                return "Picture tones, sync locked"
            return "Picture tones, looking for the sync…"
        if self._level < 1e-4:
            self._activity = "none"
            return "No audio"
        if tone is None:
            self._activity = "noise"
            return "Noise only, no SSTV tones"
        if self._vis_pending:
            self._activity = "vis"
            return "Reading the VIS code…"
        if guess is not None:
            self._activity = "sync"
            spec, period = guess
            return f"Sync pulses every {period:.0f} ms ({spec.name}?)"
        if abs(tone - FREQ_LEADER_HZ) < 60.0:
            self._activity = "leader"
            return "Hearing 1900 Hz leader…"
        if abs(tone - FREQ_SYNC_HZ) < 60.0:
            self._activity = "sync"
            return "Hearing 1200 Hz sync tone…"
        if FREQ_BLACK_HZ - 30.0 <= tone <= FREQ_WHITE_HZ + 30.0:
            self._activity = "picture"
            return f"Hearing {tone:.0f} Hz: picture tones, no VIS yet"
        self._activity = "tone"
        return f"Hearing {tone:.0f} Hz"


class SSTVImageViewer:
    """
    Simple SSTV image viewer/manager.

    Manages received SSTV images with history and metadata.
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize viewer.

        Args:
            save_dir: Directory for auto-saving images
        """
        self.save_dir = Path(save_dir) if save_dir else Path("sstv_images")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.images: List[dict] = []
        self.current_index: int = -1

    def add_image(
        self, image: np.ndarray, mode: SSTVModeSpec, auto_save: bool = True
    ) -> int:
        """
        Add a received image.

        Args:
            image: Image data as numpy array
            mode: SSTV mode used
            auto_save: Whether to auto-save to disk

        Returns:
            Image index
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        entry = {
            "image": image.copy(),
            "mode": mode.name,
            "timestamp": timestamp,
            "width": mode.width,
            "height": mode.height,
            "filename": None,
        }

        if auto_save:
            filename = f"sstv_{timestamp}_{mode.name.replace(' ', '_')}.png"
            filepath = self.save_dir / filename

            try:
                from PIL import Image

                img = Image.fromarray(image, "RGB")
                img.save(str(filepath))
                entry["filename"] = str(filepath)
                logger.info(f"Auto-saved SSTV image: {filepath}")
            except (ImportError, OSError, ValueError) as e:
                logger.warning(f"Failed to auto-save: {e}")

        self.images.append(entry)
        self.current_index = len(self.images) - 1

        return self.current_index

    def get_current_image(self) -> Optional[np.ndarray]:
        """Get current image."""
        if 0 <= self.current_index < len(self.images):
            return self.images[self.current_index]["image"]
        return None

    def get_image_count(self) -> int:
        """Get number of stored images."""
        return len(self.images)

    def next_image(self) -> Optional[np.ndarray]:
        """Go to next image."""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            return self.get_current_image()
        return None

    def prev_image(self) -> Optional[np.ndarray]:
        """Go to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            return self.get_current_image()
        return None

    def get_image_info(self, index: Optional[int] = None) -> Optional[dict]:
        """Get info for an image."""
        idx = index if index is not None else self.current_index
        if 0 <= idx < len(self.images):
            info = self.images[idx].copy()
            del info["image"]  # Don't include raw data
            return info
        return None

    def clear(self) -> None:
        """Clear all stored images."""
        self.images.clear()
        self.current_index = -1


__all__ = [
    "SSTVMode",
    "SSTVModeSpec",
    "SSTV_MODES",
    "SSTVState",
    "SSTVDecoder",
    "SSTVImageViewer",
    "find_mode",
    "line_templates",
    "preamble_segments",
    "vis_segments",
    "value_to_freq",
    "freq_to_value",
    "rgb_to_ycc",
    "ycc_to_rgb",
]
