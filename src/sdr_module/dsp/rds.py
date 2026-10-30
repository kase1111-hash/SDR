"""
RDS physical layer: the FM broadcast multiplex in, RDS data bits out.

An FM broadcast station's multiplex (what the FM discriminator gives:
the instantaneous frequency) carries the program audio up to 15 kHz, a
19 kHz stereo pilot, the stereo difference signal at 23-53 kHz and RDS: a
57 kHz carrier, locked to the third harmonic of the pilot, suppressed and
BPSK modulated at 1187.5 bit/s. Each bit is sent biphase (a positive and a
negative half, or the reverse) after differential coding, and shaped so
the signal stays within +/-2.4 kHz of 57 kHz.

:class:`RDSDemodulator` reverses that, in blocks of any size (a stream
decoded 33 ms at a time gives the same bits as the whole of it):

1. The 57 kHz and 19 kHz regions are each mixed to 0 Hz, decimated to
   about 12 kHz and low-passed to +/-2.4 kHz (60 dB down by 4 kHz, clear
   of the stereo signal and the audio), by identical filters, so the two
   stay aligned.
2. Carrier phase: where the station sends a pilot (it stands well above
   the noise in its channel), the 57 kHz reference is the pilot's phase,
   smoothed over 25 ms, times three; what is left is a constant phase,
   found by squaring the signal (which removes the data) and averaging it
   over 40 ms. Without a pilot the same squaring loop tracks the carrier
   on its own.
3. The rotated signal's real part goes through the RDS matched filter and
   a symbol clock (``BitSlicer``) at 2375 half-bits per second.
4. Biphase: the half-bits are paired so that the two halves of a bit are
   opposite (which pairing does that is learned from the signal), the bit
   is the sign of their difference, and differential decoding (each bit
   against the one before) removes the BPSK sign ambiguity.

The bits go to :class:`~sdr_module.dsp.protocols.RDSDecoder` (block sync,
error correction, groups). NumPy only; no Qt.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .channelizer import Delay, FirDecimator, MovingAverage, Nco, lowpass_taps
from .protocols import BitSlicer

__all__ = ["RDSDemodulator", "matched_filter_taps"]

CARRIER_HZ = 57e3
PILOT_HZ = 19e3
BAUD = 1187.5


def _odd(value: float) -> int:
    return max(1, int(round(value))) | 1


def matched_filter_taps(rate: float, span_bits: float = 4.0) -> np.ndarray:
    """The RDS matched filter at ``rate``: the data-shaping response,
    cos(pi f td / 4) up to 2/td (td: one bit), over +/- ``span_bits`` bits
    (float64, unit DC gain is not needed: only signs matter)."""
    a = 1.0 / (4.0 * BAUD)
    half = int(round(span_bits / BAUD * rate))
    t = np.arange(-half, half + 1) / float(rate)
    taps = np.sinc(t / a + 0.5) + np.sinc(t / a - 0.5)
    return taps / np.sum(np.abs(taps))


class _Fir:
    """Streaming FIR (``np.convolve`` with the history carried over): for
    short filters at low rates."""

    def __init__(self, taps: np.ndarray):
        self.taps = np.asarray(taps, dtype=np.float64)
        self._history = np.zeros(len(self.taps) - 1)

    def reset(self) -> None:
        self._history[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return np.zeros(0)
        buf = np.concatenate((self._history, np.asarray(x, dtype=np.float64)))
        if len(self._history):
            self._history = buf[len(buf) - len(self._history) :].copy()
        return np.convolve(buf, self.taps, mode="valid")


class RDSDemodulator:
    """FM multiplex (Hz, real, at ``rate`` of at least about 130 kHz) to
    RDS data bits (0/1, differentially decoded), streaming."""

    PASS_HZ = 2.4e3
    STOP_HZ = 4.0e3
    TARGET_RATE = 12e3
    PILOT_AVG_S = 0.025
    PHASE_AVG_S = 0.04
    #: The pilot counts as present while its power (smoothed) is this many
    #: times the rest of its channel's power; it goes at ``PILOT_OFF``.
    PILOT_ON = 2.0
    PILOT_OFF = 0.5
    #: Per half-bit decay of the biphase pairing scores.
    PAIRING_DECAY = 0.95

    def __init__(self, rate: float):
        rate = float(rate)
        if rate < 2.0 * (CARRIER_HZ + self.STOP_HZ):
            raise ValueError(
                f"RDS needs the FM multiplex at {2 * (CARRIER_HZ + self.STOP_HZ) / 1e3:.0f} "
                f"kS/s or more, got {rate / 1e3:.0f} kS/s"
            )
        self.rate = rate
        # Both regions: mixed to 0 Hz, then one polyphase stage straight to
        # about 12 kHz (it keeps only what would fold into +/-4 kHz away)
        # and the sharp +/-2.4 kHz filter there. Identical filters keep the
        # pilot and the RDS signal aligned.
        factor = max(1, int(round(rate / self.TARGET_RATE)))
        work = rate / factor
        coarse = lowpass_taps(self.PASS_HZ, work - self.STOP_HZ, rate)
        sharp = lowpass_taps(self.PASS_HZ, self.STOP_HZ, work)
        self._mix = (Nco(rate, CARRIER_HZ), Nco(rate, PILOT_HZ))
        self._coarse = (FirDecimator(coarse, factor), FirDecimator(coarse, factor))
        self._sharp = (FirDecimator(sharp), FirDecimator(sharp))
        self.work_rate = work
        n_pilot = _odd(self.PILOT_AVG_S * work)
        self._pilot_avg = MovingAverage(n_pilot)
        self._pilot_power = MovingAverage(n_pilot)
        self._pilot_align = Delay((n_pilot - 1) // 2)
        n_phase = _odd(self.PHASE_AVG_S * work)
        self._phase_avg = MovingAverage(n_phase)
        self._phase_align = Delay((n_phase - 1) // 2)
        self._matched = _Fir(matched_filter_taps(work))
        self._slicer = BitSlicer(work, 2.0 * BAUD, integrate=False)
        self.reset()

    def reset(self) -> None:
        """Forget the stream."""
        for part in (
            *self._mix,
            *self._coarse,
            *self._sharp,
            self._pilot_avg,
            self._pilot_power,
            self._pilot_align,
            self._phase_avg,
            self._phase_align,
            self._matched,
            self._slicer,
        ):
            part.reset()
        self._angle = 0.0  # unwrapped twice-carrier phase at the last sample
        self._pilot = False  # a pilot is present
        self._prev = 0.0  # the last half-bit value
        self._half = 0  # half-bits so far
        self._scores = [0.0, 0.0]  # pairing scores by half-bit parity
        self._pairing = 1  # the parity of a bit's second half
        self._last = 0  # the last biphase bit (for differential decoding)
        self.quality = 0.0  # how cleanly the halves pair (0 noise .. 1 clean)

    @property
    def pilot(self) -> bool:
        """Whether the station's 19 kHz pilot is there (as of the last
        sample)."""
        return self._pilot

    def process(self, composite: np.ndarray) -> List[int]:
        """The RDS data bits found in a block of the FM multiplex (Hz)."""
        x = np.asarray(composite, dtype=np.float32).ravel()
        if not len(x):
            return []
        z, p = (
            self._sharp[i].process(self._coarse[i].process(self._mix[i].process(x)))
            for i in (0, 1)
        )
        if not len(z):
            return []
        z = z.astype(np.complex128)
        p = p.astype(np.complex128)
        z = self._pilot_align.process(z)
        mean = self._pilot_avg.process(p)
        power = self._pilot_power.process(p.real * p.real + p.imag * p.imag)
        coherent = mean.real * mean.real + mean.imag * mean.imag
        present = self._pilot_state(coherent / np.maximum(power - coherent, 1e-30))
        # The 57 kHz reference from the pilot: its phase times three.
        unit = mean / np.maximum(np.abs(mean), 1e-30)
        w = np.where(present, z * np.conj(unit * unit * unit), z)
        # The phase left over (or the whole carrier without a pilot): from
        # the signal squared, which no longer depends on the data.
        squared = self._phase_avg.process(w * w)
        angle = np.unwrap(np.concatenate(([self._angle], np.angle(squared))))[1:]
        self._angle = float(angle[-1])
        w = self._phase_align.process(w)
        data = np.real(w * np.exp(-0.5j * angle))
        values = self._slicer.process_values(self._matched.process(data))
        return self._biphase(values)

    def _pilot_state(self, ratio: np.ndarray) -> np.ndarray:
        """Pilot present (per sample), with hysteresis carried over."""
        state = np.where(
            ratio > self.PILOT_ON, 1.0, np.where(ratio < self.PILOT_OFF, 0.0, np.nan)
        )
        known = ~np.isnan(state)
        if not known.any():
            return np.full(len(ratio), self._pilot)
        index = np.where(known, np.arange(len(state)), -1)
        np.maximum.accumulate(index, out=index)
        filled = np.where(index >= 0, state[np.maximum(index, 0)], float(self._pilot))
        self._pilot = bool(filled[-1])
        return filled.astype(bool)

    def _biphase(self, values: List[float]) -> List[int]:
        """Pair half-bits into bits and decode them differentially."""
        bits: List[int] = []
        prev = self._prev
        half = self._half
        scores = self._scores
        pairing = self._pairing
        last = self._last
        decay = self.PAIRING_DECAY
        for value in values:
            diff = prev - value
            parity = half & 1
            scores[parity] = scores[parity] * decay + abs(diff)
            other = 1 - pairing
            if scores[other] > 1.15 * scores[pairing]:
                pairing = other
            if parity == pairing:
                biphase = 1 if diff > 0 else 0
                bits.append(biphase ^ last)
                last = biphase
            prev = value
            half += 1
        self._prev, self._half, self._pairing, self._last = prev, half, pairing, last
        total = scores[0] + scores[1]
        # Clean biphase: the true pairs score twice the others (quality 1);
        # noise scores both alike (0).
        self.quality = (
            min(1.0, 3.0 * abs(scores[0] - scores[1]) / total) if total else 0.0
        )
        return bits
