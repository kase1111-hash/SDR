"""
Stereo pilot detection for broadcast FM.

A stereo FM station adds a 19 kHz pilot tone to its multiplex signal (the
discriminator output), in the guard band between the program audio (up to
15 kHz) and the L-R subcarrier (23-53 kHz). :class:`StereoPilotDetector`
measures the power at 19 kHz against two empty spots of that guard band
(17 and 21 kHz), each block, smooths the ratio over about half a second and
turns STEREO on and off with hysteresis, so the lamp neither flickers nor
lights on noise.

It works on the tuned channel's I/Q (at the channel rate, at least ~48 kHz
wide either side): a single-bin DFT (the Goertzel measurement, vectorised)
on the newest few thousand discriminator samples of each block, so it costs
a fraction of a millisecond per block.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

PILOT_HZ = 19e3
_REFERENCE_HZ = (17e3, 21e3)
# Analysis length (samples at the channel rate) per block: ~8 ms at
# 480 kHz, 117 Hz bins, far finer than the 2 kHz to the reference spots.
_WINDOW = 4096
_ON_DB = 12.0  # pilot this far above the guard band: STEREO
_OFF_DB = 6.0  # and back off below this
_TAU_S = 0.4  # smoothing time constant


class StereoPilotDetector:
    """Whether the FM station in the channel sends a 19 kHz stereo pilot.

    Feed it each block of the tuned channel (complex, at ``rate``) with
    :meth:`process`; it returns (and :attr:`present` holds) the smoothed
    verdict. :meth:`reset` after a retune.
    """

    def __init__(self, rate: float):
        self.rate = float(rate)
        self._basis: Dict[int, np.ndarray] = {}
        self.reset()

    @property
    def usable(self) -> bool:
        """Whether the channel rate leaves room for the pilot (it must pass
        21 kHz, the upper reference)."""
        return self.rate > 2.2 * max(_REFERENCE_HZ)

    def reset(self) -> None:
        """Forget the station (after a retune or a gap)."""
        self.present = False
        self.ratio_db = 0.0
        self._warm = False

    def _bins(self, length: int) -> np.ndarray:
        """Windowed DFT rows for the pilot and the references (cached)."""
        basis = self._basis.get(length)
        if basis is None:
            k = np.arange(length, dtype=np.float64)
            window = 0.5 - 0.5 * np.cos(2.0 * np.pi * (k + 0.5) / length)
            freqs = (PILOT_HZ,) + _REFERENCE_HZ
            rows = [window * np.exp(-2j * np.pi * f * k / self.rate) for f in freqs]
            basis = np.asarray(rows, dtype=np.complex64)
            if len(self._basis) > 8:
                self._basis.clear()
            self._basis[length] = basis
        return basis

    def measure(self, channel: np.ndarray) -> Tuple[float, float]:
        """``(pilot-to-guard-band ratio in dB, seconds covered)`` of one block
        (no smoothing). Ratio 0 for too short a block."""
        x = np.asarray(channel, dtype=np.complex64)
        if len(x) < 2:
            return 0.0, 0.0
        tail = x[-(_WINDOW + 1) :]
        disc = np.angle(tail[1:] * np.conj(tail[:-1])).astype(np.float32)
        powers = np.abs(self._bins(len(disc)) @ disc) ** 2
        reference = float(np.mean(powers[1:])) + 1e-20
        ratio = 10.0 * math.log10(float(powers[0]) / reference + 1e-20)
        return ratio, len(x) / self.rate

    def process(self, channel: np.ndarray) -> bool:
        """Measure one block and return whether STEREO is lit."""
        if not self.usable or len(channel) < 64:
            return self.present
        ratio, seconds = self.measure(channel)
        if not self._warm:
            self.ratio_db, self._warm = ratio, True
        else:
            alpha = 1.0 - math.exp(-seconds / _TAU_S)
            self.ratio_db += alpha * (ratio - self.ratio_db)
        if self.present and self.ratio_db < _OFF_DB:
            self.present = False
        elif not self.present and self.ratio_db >= _ON_DB:
            self.present = True
        return self.present


__all__ = ["PILOT_HZ", "StereoPilotDetector"]
