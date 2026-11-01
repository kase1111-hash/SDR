"""
Numerically controlled oscillator (NCO) for streaming frequency shifts.

The receiver tunes a VFO inside the span the SDR captures: every I/Q block
is mixed by minus the VFO's offset from the LO, which moves the tuned
channel to 0 Hz before the channel filter. :class:`Nco` does that mixing a
block at a time with a phase that carries over between blocks, so a stream
split into blocks gives exactly the output of one long block, and changing
the frequency mid-stream bends the phase instead of jumping (no click).

It is cheap enough to run on every sample at 20 MS/s: a precomputed phasor
table of :data:`TABLE_SIZE` samples is multiplied by one phasor per table
length, two complex64 multiplications per sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Samples in the precomputed phasor table (one multiply per sample by the
#: table, and one per table length by the running phase).
TABLE_SIZE = 4096


class Nco:
    """Phase-continuous mixer: ``mix(x)`` returns ``x * exp(-2j*pi*f*t)``.

    ``frequency_hz`` is the shift: a signal at ``+frequency_hz`` comes out at
    0 Hz. ``t`` counts every sample mixed so far, so consecutive blocks join
    seamlessly; :meth:`reset` starts the phase at zero again.

    Thread safety: none (one stream, one thread).
    """

    def __init__(self, sample_rate: float, frequency_hz: float = 0.0):
        if not sample_rate > 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = float(sample_rate)
        self._frequency = 0.0
        # Phase of the next sample, in cycles, kept in [0, 1).
        self._cycles = 0.0
        self._table: Optional[np.ndarray] = None
        self.frequency_hz = frequency_hz

    @property
    def frequency_hz(self) -> float:
        """The shift in Hz (a signal at this offset is mixed to 0 Hz)."""
        return self._frequency

    @frequency_hz.setter
    def frequency_hz(self, value: float) -> None:
        freq = float(value)
        if not np.isfinite(freq):
            raise ValueError("frequency_hz must be finite")
        if freq != self._frequency:
            self._frequency = freq
            self._table = None  # rebuilt for the new step; the phase carries on

    @property
    def phase_cycles(self) -> float:
        """Phase of the next sample's phasor, in cycles (0 to 1)."""
        return self._cycles

    def reset(self) -> None:
        """Start the phase at zero again (a new, unrelated stream)."""
        self._cycles = 0.0

    def _step(self) -> float:
        """Phase advance per sample, in cycles (negative: mixing down)."""
        return -self._frequency / self.sample_rate

    def _phasors(self) -> np.ndarray:
        table = self._table
        if table is None:
            step = self._step()
            # Exact phases in float64 (reduced to one cycle) before rounding.
            cycles = np.mod(np.arange(TABLE_SIZE) * step, 1.0)
            table = np.exp(2j * np.pi * cycles).astype(np.complex64)
            self._table = table
        return table

    def mix(self, samples: np.ndarray) -> np.ndarray:
        """Shift ``samples`` (complex) down by :attr:`frequency_hz`.

        Returns complex64. With a zero shift the phase stands still: the
        input comes back turned by the phase reached so far (unchanged, and
        not copied, while that is zero), so moving the shift to 0 Hz and
        away again keeps the output continuous, as any other move does.
        """
        x = np.asarray(samples, dtype=np.complex64)
        n = len(x)
        if n == 0:
            return x
        if self._frequency == 0.0:
            if self._cycles == 0.0:
                return x
            return x * np.complex64(np.exp(2j * np.pi * self._cycles))
        step = self._step()
        table = self._phasors()
        size = TABLE_SIZE
        blocks = -(-n // size)
        # The running phase at the start of each table-length block, from
        # float64 cycles (reduced to one cycle, so precision never drifts).
        starts_cycles = np.mod(self._cycles + step * size * np.arange(blocks), 1.0)
        starts = np.exp(2j * np.pi * starts_cycles).astype(np.complex64)
        out = np.empty(n, dtype=np.complex64)
        whole = n // size
        if whole:
            view = out[: whole * size].reshape(whole, size)
            np.multiply(x[: whole * size].reshape(whole, size), table, out=view)
            view *= starts[:whole, None]
        rest = n - whole * size
        if rest:
            tail = out[whole * size :]
            np.multiply(x[whole * size :], table[:rest], out=tail)
            tail *= starts[whole]
        self._cycles = float(np.mod(self._cycles + step * n, 1.0))
        return out
