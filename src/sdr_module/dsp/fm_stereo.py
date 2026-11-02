"""
Broadcast FM stereo: the stereo difference signal out of the FM multiplex,
and how much of it to play.

A stereo station's multiplex (what the FM discriminator gives) carries the
mono program L+R up to 15 kHz, a 19 kHz pilot tone, and L-R as double
sideband with a suppressed 38 kHz carrier (23-53 kHz). That carrier is the
pilot's second harmonic: with the pilot sent as ``sin(phi)``, L-R is sent
as ``(L-R)/2 * sin(2 phi)`` (both cross zero upward together), next to
``(L+R)/2``.

:class:`StereoDecoder` recovers the pilot and demodulates L-R, a block of
the multiplex at a time:

1. Pilot: the multiplex is mixed by -19 kHz (the pilot to 0 Hz) and comes
   down to about 8 kHz in two stages that keep +/-1.5 kHz (the guard band
   between the program, which ends at 15 kHz, and L-R, which starts at
   23 kHz: 60 dB down from 3.5 kHz). Averaged over 15 ms, that is the
   pilot's phasor. The averaging and the filters make it late; the pilot's
   frequency offset (a station's own, or a receiver's sample-rate error:
   a few Hz) is measured from how fast the phasor turns, over a quarter of
   a second, and turns it forward by that delay. Its phase, doubled, is the
   38 kHz carrier (a pilot-derived carrier, as the hardware decoders use).
2. L-R: the multiplex times twice that carrier puts L-R at 0 Hz (and the
   rest above 15 kHz). A discriminator that takes the phase step from one
   sample to the next averages the frequency over that sample, which takes
   ``sinc(f / rate)`` off each component: 1 % at 38 kHz at 480 kS/s, 4 %
   at 250 kS/s, but next to nothing below 15 kHz. Unmatched, that alone
   limits the separation to 46 and 35 dB; L-R is turned up by as much
   (``phase_step``). The receiver low-passes it with the same audio
   filters (and the same de-emphasis) as the mono program, so the two stay
   aligned to the sample: left = mono + difference, right = mono -
   difference.
3. Blend: stereo is far noisier than mono (the FM noise rises with
   frequency, and L-R sits at 23-53 kHz: about 22 dB more hiss after
   de-emphasis). The noise left in the guard band once the pilot is taken
   out (averaged over 16 ms) gives the hiss stereo would add; while that
   is more than ``FULL_SNR_DB`` below full scale, the difference signal is
   turned down so the hiss stays there, and below ``MONO_SNR_DB`` it is
   gone (mono). No pilot (a mono station, or noise): mono. The blend falls
   at once (no burst of hiss when the signal fades) and comes back over
   ``RISE_S`` (no pumping).

Everything carries its state between calls, per sample: a stream decoded
33 ms at a time gives the same output as the whole of it at once (to
floating-point rounding). NumPy only; no Qt.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .channelizer import Delay, FirDecimator, MovingAverage, lowpass_taps

__all__ = [
    "PILOT_HZ",
    "MIN_RATE",
    "StereoDecoder",
    "stereo_noise_bandwidth",
]

PILOT_HZ = 19e3
#: Lowest multiplex rate that holds L-R (up to 53 kHz) clear of its images.
MIN_RATE = 120e3
#: Broadcast FM de-emphasis (Americas) and the audio bandwidth.
DEEMPHASIS_S = 75e-6
AUDIO_HZ = 15e3


def stereo_noise_bandwidth(
    tau_s: float = DEEMPHASIS_S, audio_hz: float = AUDIO_HZ, pilot_hz: float = PILOT_HZ
) -> float:
    """How much hiss the stereo difference signal adds, as a bandwidth (Hz):
    times the multiplex noise density at the pilot, it is the noise power
    in the left or right channel's audio (low-passed to ``audio_hz``,
    de-emphasized with ``tau_s``).

    FM noise rises with frequency (its density goes as f squared); the
    difference signal's comes from both sidebands, ``2 f_p - f`` and
    ``2 f_p + f``, and is de-emphasized like the program:
    ``integral of [(2fp - f)^2 + (2fp + f)^2] / fp^2 * 2 / (1 + (f/fc)^2)``
    from 0 to ``audio_hz`` (fc = 1 / (2 pi tau)). For 75 us and 15 kHz that
    is about 49 kHz; mono's own noise is 300 Hz of the same density."""
    fc = 1.0 / (2.0 * math.pi * float(tau_s))
    top = float(audio_hz)
    carrier_sq = (2.0 * pilot_hz) ** 2
    # integral of (A + f^2) / (1 + f^2/fc^2) df = fc^2 F + (A - fc^2) fc atan(F/fc)
    integral = fc * fc * top + (carrier_sq - fc * fc) * fc * math.atan(top / fc)
    return 4.0 * integral / (pilot_hz * pilot_hz)


class StereoDecoder:
    """The stereo difference signal of a broadcast FM multiplex (see the
    module docstring).

    Feed it the discriminator's output with :meth:`process`, scaled so the
    FM deviation is 1 (what the receiver's audio is scaled to), at ``rate``
    (at least :data:`MIN_RATE`). It returns ``2 * blend * mpx * sin(2 phi)``
    at the same rate: low-pass that like the mono program (15 kHz, the same
    de-emphasis) and it is ``blend * (L-R)/2``. ``phase_step`` (default):
    the multiplex comes from a discriminator that takes the phase step
    between samples, whose droop at 38 kHz is made up (see the module
    docstring); False for a multiplex that has none.

    After each block, :attr:`blend` is how much stereo plays (0 mono .. 1
    full separation), :attr:`target` how much the signal allows (the blend
    follows it: at once down, over ``RISE_S`` up), :attr:`pilot` whether a
    pilot is there, and :attr:`snr_db` how far below full scale the hiss of
    full stereo would be (None before anything was measured).
    """

    #: The blend: full separation while stereo's hiss is at least this far
    #: below full scale, turned down to keep it there below that, gone below
    #: ``MONO_SNR_DB`` (a fade over ``FADE_DB`` from there).
    FULL_SNR_DB = 50.0
    MONO_SNR_DB = 30.0
    FADE_DB = 4.0
    #: The blend comes back from mono to full stereo over this long (it
    #: falls at once, through the 16 ms noise average).
    RISE_S = 0.75
    #: The pilot counts as there while it stands this far above the noise
    #: in its own (15 ms) average, and goes below ``PILOT_OFF_DB``; and
    #: only at ``MIN_PILOT`` of the deviation or more (stations send 8-10
    #: %: on a clean signal without one, both it and the noise are only
    #: what the filters let through, whose ratio means nothing).
    PILOT_ON_DB = 15.0
    PILOT_OFF_DB = 9.0
    MIN_PILOT = 0.02

    _WORK_HZ = 8e3  # rate of the pilot's phasor
    _COARSE_HZ = 48e3  # rate after the first stage
    _PASS_HZ = 1.5e3  # the guard band kept around the pilot
    _STOP_HZ = 3.5e3
    _PHASOR_S = 0.015
    _NOISE_S = 0.016
    _OFFSET_S = 0.25
    _ROT_BLOCK = 4096  # samples of the 19 kHz phasor table

    def __init__(
        self,
        rate: float,
        tau_s: float = DEEMPHASIS_S,
        audio_hz: float = AUDIO_HZ,
        phase_step: bool = True,
    ):
        rate = float(rate)
        if rate < MIN_RATE:
            raise ValueError(
                f"FM stereo needs the multiplex at {MIN_RATE / 1e3:.0f} kS/s or "
                f"more, got {rate / 1e3:.0f} kS/s"
            )
        self.rate = rate
        self._noise_bw_stereo = stereo_noise_bandwidth(tau_s, audio_hz)
        # Twice the carrier (DSB), and the droop made up.
        droop = float(np.sinc(2.0 * PILOT_HZ / rate)) if phase_step else 1.0
        self._gain = np.float32(2.0 / droop)
        # The pilot to 0 Hz, then two stages to about 8 kHz: a short one
        # that only keeps what would fold onto +/-3.5 kHz, then the sharp
        # +/-1.5 kHz filter (60 dB down by 3.5 kHz: the program and L-R).
        first = max(1, int(round(rate / self._COARSE_HZ)))
        mid = rate / first
        second = max(1, int(round(mid / self._WORK_HZ)))
        self.factor = first * second
        self.work_rate = rate / self.factor
        coarse = lowpass_taps(self._PASS_HZ, mid - self._STOP_HZ, rate)
        sharp = lowpass_taps(self._PASS_HZ, self._STOP_HZ, mid)
        k = np.arange(self._ROT_BLOCK)
        self._rot_table = np.exp(
            -2j * np.pi * np.mod(k * (PILOT_HZ / rate), 1.0)
        ).astype(np.complex64)
        self._coarse = FirDecimator(coarse, first)
        self._sharp = FirDecimator(sharp, second)
        n_phasor = max(1, int(round(self._PHASOR_S * self.work_rate))) | 1
        self._phasor = MovingAverage(n_phasor)
        self._align = Delay((n_phasor - 1) // 2)
        self._noise = MovingAverage(max(1, int(round(self._NOISE_S * self.work_rate))))
        self._turn = MovingAverage(max(2, int(round(self._OFFSET_S * self.work_rate))))
        # How late the averaged phasor is (input samples), plus half a work
        # sample: each one is held for the ``factor`` input samples after
        # it, so it is turned forward to the middle of them.
        self._late = (
            (len(coarse) - 1) / 2.0
            + first * (len(sharp) - 1) / 2.0
            + self.factor * (n_phasor - 1) / 2.0
            + self.factor / 2.0
        )
        # Noise bandwidth (Hz) of what the residual measures: the sharp
        # filter's (complex: both sides of the pilot).
        taps = np.asarray(sharp, dtype=np.float64)
        self._measured_bw = mid * float(np.sum(taps * taps)) / float(np.sum(taps)) ** 2
        self._phasor_bw = self.work_rate / n_phasor
        self._rise = self.factor / (self.RISE_S * rate)  # blend per work sample
        # Work samples before the averages (and the filters before them)
        # have filled after a reset.
        self._warm_up = n_phasor + self._noise.length + len(sharp)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the station (a retune, a gap): mono until it is heard."""
        for part in (
            self._coarse,
            self._sharp,
            self._phasor,
            self._align,
            self._noise,
            self._turn,
        ):
            part.reset()
        self._count = 0  # input samples so far
        self._work = 0  # work samples so far
        self._last_phasor = 0j  # (for how fast it turns)
        self._anchor = 0j  # the carrier held now (scaled: silent at first)
        self._offset_per_work = 0.0  # the pilot's turn per work sample
        self.blend = 0.0
        self.target = 0.0
        self.pilot = False
        self.snr_db: Optional[float] = None
        self.pilot_db: Optional[float] = None

    @property
    def offset_hz(self) -> float:
        """The pilot's frequency offset from 19 kHz, as measured (Hz)."""
        return self._offset_per_work * self.work_rate / (2.0 * math.pi)

    # ------------------------------------------------------------------
    def process(self, mpx: np.ndarray) -> np.ndarray:
        """``2 * blend * mpx * sin(2 phi)`` (float32, same length): the
        stereo difference at 0 Hz once low-passed (see the class)."""
        x = np.asarray(mpx, dtype=np.float32).ravel()
        n = len(x)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # The pilot to 0 Hz (the same phasors make the 38 kHz carrier).
        start, factor = self._count, self.factor
        rot = self._rotation(start, n)
        work = self._sharp.process(self._coarse.process(x * rot))
        phasors, blends = self._track(work)
        # The carrier's amplitude: sqrt(2 * gain * blend), so that the
        # product below comes out times 2 * gain * blend (for the droop and
        # the blend) with no more passes over the block.
        scale = np.sqrt(blends * (2.0 * float(self._gain)))
        table = np.concatenate(([self._anchor], phasors * scale)).astype(np.complex64)
        # Work sample i is the pilot at input sample i * factor, held for
        # the factor samples from there; the one before this block's first
        # is still in force at its start (the table's first entry).
        first = self._work - len(work)  # index of work[0]
        anchors = np.arange(first - 1, self._work, dtype=np.int64) * factor
        lo = np.maximum(anchors, start)
        hi = np.minimum(anchors + factor, start + n)
        carrier = np.repeat(table, np.maximum(hi - lo, 0))
        if len(phasors):
            self._anchor = complex(table[-1])
        self._count += n
        # exp(j psi) (psi the pilot's phase), scaled: the pilot is cos(psi)
        # = sin(phi), phi = psi + 90 degrees, and L-R rides on sin(2 phi) =
        # -sin(2 psi) = -Im(exp(j psi)^2) = -2 Re Im of exp(j psi).
        carrier *= np.conj(rot)
        out = carrier.real * carrier.imag
        out *= x
        out *= np.float32(-1.0)
        return out.astype(np.float32, copy=False)

    def _rotation(self, start: int, n: int) -> np.ndarray:
        """``exp(-j 2 pi 19 kHz t)`` for ``n`` samples from sample ``start``
        (complex64: block phasors from the sample count in float64 times a
        table of one block, as :class:`~.channelizer.Nco` mixes; no drift
        over hours)."""
        block = self._ROT_BLOCK
        starts = start + block * np.arange(-(-n // block), dtype=np.int64)
        cycles = np.mod(starts * (PILOT_HZ / self.rate), 1.0)
        phasors = np.exp(-2j * np.pi * cycles).astype(np.complex64)
        return (phasors[:, None] * self._rot_table[None, :]).ravel()[:n]

    # ------------------------------------------------------------------
    def _track(self, work: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The pilot's unit phasor (turned forward to each work sample's own
        time) and the blend, per work sample."""
        m = len(work)
        if m == 0:
            return np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=np.float64)
        work = work.astype(np.complex128)
        phasor = self._phasor.process(work)
        centred = self._align.process(work)
        # How fast it turns (radians per work sample), over a quarter second.
        previous = np.concatenate(([self._last_phasor], phasor[:-1]))
        self._last_phasor = complex(phasor[-1])
        turn = self._turn.process(phasor * np.conj(previous))
        step = np.angle(turn)
        self._offset_per_work = float(step[-1])
        ahead = phasor * np.exp(1j * step * (self._late / self.factor))
        magnitude = np.abs(ahead)
        unit = np.where(magnitude > 0.0, ahead / np.maximum(magnitude, 1e-30), 1.0)
        # Noise: what is left of the guard band once the pilot is out. A
        # pilot off frequency turns during the average, which then holds
        # sin(M w/2) / (M sin(w/2)) of it (w its turn per sample): the
        # centred sample is scaled alike, or the turn reads as noise (a
        # pilot 7 Hz off blended a clean station 18 % toward mono).
        length = self._phasor.length
        held = np.sinc(length * step / (2 * np.pi)) / np.sinc(step / (2 * np.pi))
        residual = centred * held - phasor
        noise = self._noise.process(residual.real**2 + residual.imag**2)
        density = noise / self._measured_bw  # per Hz
        pilot_power = magnitude**2
        pilot_db = 10.0 * np.log10(
            (pilot_power + 1e-30) / (density * self._phasor_bw + 1e-20)
        )
        hiss = density * self._noise_bw_stereo
        snr_db = 10.0 * np.log10(0.5 / (hiss + 1e-12))
        # Until the averages have filled (after a reset), nothing is known.
        known = np.arange(self._work, self._work + m) >= self._warm_up
        pilot_db = np.where(2.0 * magnitude >= self.MIN_PILOT, pilot_db, -100.0)
        present = self._pilot_state(np.where(known, pilot_db, np.nan))
        target = np.power(10.0, np.minimum(snr_db - self.FULL_SNR_DB, 0.0) / 20.0)
        target *= np.clip((snr_db - self.MONO_SNR_DB) / self.FADE_DB, 0.0, 1.0)
        target = np.where(present, target, 0.0)
        blends = self._slew(target)
        self._work += m
        self.pilot = bool(present[-1])
        self.pilot_db = float(pilot_db[-1])
        self.snr_db = float(snr_db[-1])
        self.blend = float(blends[-1])
        self.target = float(target[-1])
        return unit, blends

    def _pilot_state(self, pilot_db: np.ndarray) -> np.ndarray:
        """Pilot there (per work sample), with hysteresis carried over (a
        NaN, not known yet, keeps the state: absent after a reset)."""
        state = np.where(
            pilot_db >= self.PILOT_ON_DB,
            1.0,
            np.where(pilot_db < self.PILOT_OFF_DB, 0.0, np.nan),
        )
        known = ~np.isnan(state)
        if not known.any():
            return np.full(len(pilot_db), self.pilot)
        index = np.where(known, np.arange(len(state)), -1)
        np.maximum.accumulate(index, out=index)
        filled = np.where(index >= 0, state[np.maximum(index, 0)], float(self.pilot))
        return filled.astype(bool)

    def _slew(self, target: np.ndarray) -> np.ndarray:
        """Follow ``target`` down at once and up by at most ``_rise`` per
        work sample: ``g[k] = min(target[k], g[k-1] + rise)``, vectorised
        as a running minimum."""
        rise = self._rise
        k = np.arange(1, len(target) + 1, dtype=np.float64)
        # g[k] = min(g0 + rise*k, min over i<=k of target[i] + rise*(k-i))
        candidates = np.minimum.accumulate(target - rise * k)
        out = np.minimum(self.blend + rise * k, candidates + rise * k)
        return np.clip(out, 0.0, 1.0)
