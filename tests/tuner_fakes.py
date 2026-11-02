"""Fake receivers for sweep and scan tests (not collected by pytest).

:class:`FakeTuner` hears carriers at fixed frequencies (``STATIONS``)
around wherever it is tuned, with a little noise, through the parts of the
:class:`~sdr_module.devices.base.SDRDevice` API a sweep uses
(``set_frequency``, ``start_rx``/``stop_rx``, queued ``read_samples(n,
timeout)``, ``state``, ``sample_rate``, ``spec`` and ``info``). Like hardware, a retune
leaves one block behind that was taken on the old frequency: queued at once,
or still "in flight" for ``in_flight_s`` (an RTL-SDR's USB transfer), so a
sweep that doesn't drop it measures the old frequency's signals at the new
one. Each fake is read by one thread at a time (a sweep's thread for its
device).
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

#: Carriers in the air: {frequency (Hz): level (dBFS)}.
STATIONS: Dict[float, float] = {
    88.5e6: -30.0,
    93.3e6: -25.0,
    100.1e6: -20.0,
    107.9e6: -40.0,
}


class FakeTuner:
    """A receiver hearing ``stations`` around its centre frequency.

    Args:
        name: Its ``info.name``.
        rate: Its sample rate.
        low_hz, high_hz: What it tunes (``spec.freq_min``/``freq_max``);
            set_frequency refuses anything else. Neither given: no spec.
        stations: ``{freq: dBFS}``, default ``STATIONS``.
        deliver: False: it never returns samples (unplugged, or used by
            another program).
        in_flight_s: How long after a retune the block taken on the old
            frequency arrives (0: it is queued at once).
        block: Samples per read.
    """

    def __init__(
        self,
        name: str = "Fake tuner",
        rate: float = 2.4e6,
        low_hz: Optional[float] = None,
        high_hz: Optional[float] = None,
        stations: Optional[Dict[float, float]] = None,
        deliver: bool = True,
        in_flight_s: float = 0.0,
        block: int = 32768,
        seed: int = 0,
    ) -> None:
        self.info = SimpleNamespace(name=name)
        if low_hz is None and high_hz is None:
            self.spec = None
        else:
            self.spec = SimpleNamespace(
                freq_min=0.0 if low_hz is None else float(low_hz),
                freq_max=math.inf if high_hz is None else float(high_hz),
            )
        self.sample_rate = float(rate)
        self.frequency = 0.0
        self.stations = dict(STATIONS if stations is None else stations)
        self.deliver = deliver
        self.in_flight_s = float(in_flight_s)
        self.block = int(block)
        #: LOs tuned, in order; reads; start_rx/stop_rx calls.
        self.tuned: List[float] = []
        self.reads = 0
        self.started = 0
        self.stopped = 0
        self.streaming = False
        #: Set while a read waits (for tests that cancel during a read).
        self.reading = threading.Event()
        self._rng = np.random.default_rng(seed)
        self._stale: Optional[tuple] = None  # (due time, frequency)

    @property
    def state(self) -> SimpleNamespace:
        """What it is doing, as an SDRDevice's ``state`` says it."""
        return SimpleNamespace(
            frequency=self.frequency,
            sample_rate=self.sample_rate,
            gain=0.0,
            is_streaming=self.streaming,
        )

    # -- settings ----------------------------------------------------------- #
    def set_frequency(self, freq: float) -> bool:
        spec = self.spec
        if spec is not None and not spec.freq_min <= freq <= spec.freq_max:
            return False
        self._stale = (time.monotonic() + self.in_flight_s, self.frequency)
        self.frequency = float(freq)
        self.tuned.append(float(freq))
        return True

    def set_sample_rate(self, rate: float) -> bool:
        self.sample_rate = float(rate)
        return True

    def set_gain_mode(self, auto: bool) -> bool:
        return True

    def set_gain(self, gain: float) -> bool:
        return True

    # -- streaming ---------------------------------------------------------- #
    def start_rx(self, callback=None) -> bool:
        self.started += 1
        self.streaming = True
        return True

    def stop_rx(self) -> bool:
        self.stopped += 1
        self.streaming = False
        return True

    def read_samples(self, n: int, timeout: float = 1.0):
        self.reads += 1
        self.reading.set()
        try:
            if not (self.streaming and self.deliver):
                time.sleep(min(float(timeout or 0.0), 0.005))
                return None
            if self._stale is not None:
                due, freq = self._stale
                wait = due - time.monotonic()
                if wait > 0:
                    if not timeout or wait > timeout:
                        time.sleep(float(timeout or 0.0))
                        return None  # not here yet
                    time.sleep(wait)
                self._stale = None
                return self.synthesize(freq)
            if not timeout:
                return None  # nothing queued: a drain ends here
            return self.synthesize(self.frequency)
        finally:
            self.reading.clear()

    def synthesize(self, centre: float) -> np.ndarray:
        """One block received around ``centre``."""
        count = self.block
        t = np.arange(count) / self.sample_rate
        x = (
            self._rng.standard_normal(count) + 1j * self._rng.standard_normal(count)
        ) * 1e-4
        for freq, level in self.stations.items():
            offset = freq - centre
            if abs(offset) < 0.5 * self.sample_rate:
                x = x + 10 ** (level / 20) * np.exp(2j * np.pi * offset * t)
        return x.astype(np.complex64)

    def tuned_inside(self, low: float, high: float) -> bool:
        """Whether every LO it tuned is within ``[low, high]``."""
        return all(low <= f <= high for f in self.tuned)
