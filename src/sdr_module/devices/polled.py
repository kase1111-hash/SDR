"""
The :class:`~.base.SDRDevice` streaming API for a source that makes samples
when asked.

Hardware drivers stream: ``start_rx(callback)`` hands each transfer to a
callback (or a queue) from a background thread, and ``state`` says what the
device is doing. The GUI's demo device (``gui.device_dialog.MockDevice``) and
similar sources instead make samples on request (``read_samples(n)``) and
have no ``state``, so code written for SDRDevice, such as
:class:`~sdr_module.core.dual_sdr.DualSDRController`, couldn't run on them.

:class:`PolledDevice` wraps such a source: a background thread reads it in
real time, ``block_s`` of signal at a time, and hands each block to the
callback (or the queue :meth:`~.base.SDRDevice.read_samples` takes from).
Settings go to the source and are mirrored in ``state``. It can only
receive. No Qt::

    from sdr_module.core.dual_sdr import DualSDRController
    from sdr_module.devices.polled import PolledDevice
    from sdr_module.gui.device_dialog import MockDevice

    controller = DualSDRController()
    controller.use_devices(
        rtlsdr=PolledDevice(MockDevice(), "Demo device A"),
        hackrf=PolledDevice(MockDevice(), "Demo device B"),
    )
    controller.start_dual_rx(on_first_samples, on_second_samples)
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .base import DeviceCapability, DeviceInfo, DeviceSpec, SDRDevice

logger = logging.getLogger(__name__)

#: The most signal one read takes after the thread was held up (seconds);
#: the rest is skipped, as a stalled hardware stream drops transfers.
MAX_CATCH_UP_S = 0.25


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class PolledDevice(SDRDevice):
    """A receive-only SDRDevice reading ``source`` in real time.

    ``source`` needs ``read_samples(n)`` (``n`` complex samples, or None
    while it isn't receiving) and ``set_frequency``, ``set_sample_rate``
    and ``start_rx()``/``stop_rx()``; ``set_gain``, ``set_gain_mode``,
    ``set_bandwidth``, ``close``, ``info``, ``spec`` and its ``frequency``,
    ``sample_rate`` and ``gain`` are used when it has them.
    """

    def __init__(self, source: Any, name: str = "", block_s: float = 0.01) -> None:
        super().__init__()
        self._source = source
        self.block_s = max(0.001, float(block_s))
        self._rx_error: Optional[str] = None
        info = getattr(source, "info", None)
        self._info = DeviceInfo(
            name=name or str(getattr(info, "name", "") or type(source).__name__),
            serial=str(getattr(info, "serial", "") or ""),
            manufacturer="",
            product=type(source).__name__,
            capabilities=[DeviceCapability.RX],
        )
        spec = getattr(source, "spec", None)
        self._spec = spec if isinstance(spec, DeviceSpec) else None
        self._state.frequency = _number(getattr(source, "frequency", 0.0))
        self._state.sample_rate = _number(getattr(source, "sample_rate", 0.0))
        self._state.gain = _number(getattr(source, "gain", 0.0))
        self._is_open = True

    @property
    def source(self) -> Any:
        """The wrapped source."""
        return self._source

    @property
    def rx_error(self) -> Optional[str]:
        """Why the stream ended on its own (None while it didn't)."""
        return self._rx_error

    # -- open / close -------------------------------------------------------

    def open(self, index: int = 0) -> bool:
        """The source is open already."""
        self._is_open = True
        return True

    def close(self) -> None:
        self.stop_rx()
        closer = getattr(self._source, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"{self._name}: close failed: {e}")
        self._is_open = False

    # -- settings -----------------------------------------------------------

    @property
    def _name(self) -> str:
        return self._info.name if self._info is not None else "PolledDevice"

    def _forward(self, method_name: str, value: Any) -> bool:
        """``source.<method_name>(value)``; False when it has no such
        method, refuses (returns False) or raises."""
        method = getattr(self._source, method_name, None)
        if not callable(method):
            return False
        try:
            return method(value) is not False
        except Exception as e:
            logger.warning(f"{self._name}: {method_name}({value!r}) failed: {e}")
            return False

    def set_frequency(self, freq_hz: float) -> bool:
        ok = self._forward("set_frequency", float(freq_hz))
        if ok:
            with self._state_lock:
                self._state.frequency = float(freq_hz)
        return ok

    def set_sample_rate(self, rate_hz: float) -> bool:
        ok = self._forward("set_sample_rate", float(rate_hz))
        if ok:
            with self._state_lock:
                self._state.sample_rate = float(rate_hz)
        return ok

    def set_bandwidth(self, bw_hz: float) -> bool:
        ok = self._forward("set_bandwidth", float(bw_hz))
        if ok:
            with self._state_lock:
                self._state.bandwidth = float(bw_hz)
        return ok

    def set_gain(self, gain_db: float) -> bool:
        ok = self._forward("set_gain", float(gain_db))
        if ok:
            with self._state_lock:
                self._state.gain = float(gain_db)
        return ok

    def set_gain_mode(self, auto: bool) -> bool:
        ok = self._forward("set_gain_mode", bool(auto))
        if ok:
            with self._state_lock:
                self._state.gain_mode = "auto" if auto else "manual"
        return ok

    # -- streaming ----------------------------------------------------------

    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Start the source and a thread handing its samples to ``callback``
        (or the queue :meth:`read_samples` reads) in real time."""
        with self._state_lock:
            if self._state.is_streaming:
                return True
        starter = getattr(self._source, "start_rx", None)
        if callable(starter):
            try:
                if starter() is False:
                    return False
            except Exception as e:
                logger.error(f"{self._name}: could not start receiving: {e}")
                return False
        self._rx_callback = callback
        self._rx_error = None
        self._stop_event.clear()
        with self._state_lock:
            self._state.is_streaming = True
        self._rx_thread = threading.Thread(
            target=self._run, name=f"polled-{self._name}", daemon=True
        )
        self._rx_thread.start()
        return True

    def _run(self) -> None:
        last = time.monotonic()
        owed = 0.0
        try:
            while not self._stop_event.wait(self.block_s):
                now = time.monotonic()
                with self._state_lock:
                    rate = float(self._state.sample_rate)
                owed += max(0.0, rate * (now - last))
                last = now
                owed = min(owed, rate * MAX_CATCH_UP_S)
                count = int(owed)
                if count <= 0:
                    continue
                owed -= count
                samples = self._source.read_samples(count)
                if samples is None or len(samples) == 0:
                    continue
                callback = self._rx_callback
                if callback is not None:
                    callback(samples)
                else:
                    try:
                        self._sample_queue.put_nowait(samples)
                    except queue.Full:
                        # (As the hardware drivers: the oldest stay.)
                        self._dropped_samples += len(samples)
        except Exception as e:
            if not self._stop_event.is_set():
                # The stream ended on its own (a source or callback error):
                # say so in state, as the hardware drivers do.
                self._rx_error = str(e)
                logger.error(f"{self._name}: receiving stopped: {e}")
                with self._state_lock:
                    self._state.is_streaming = False

    def stop_rx(self) -> bool:
        with self._state_lock:
            self._state.is_streaming = False
        self._stop_event.set()
        thread = self._rx_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)
            if thread.is_alive():  # pragma: no cover - a hung source
                logger.warning(f"{self._name}: the receive thread didn't stop")
        self._rx_thread = None
        stopper = getattr(self._source, "stop_rx", None)
        if callable(stopper):
            try:
                stopper()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"{self._name}: stop_rx failed: {e}")
        return True


__all__ = ["MAX_CATCH_UP_S", "PolledDevice"]
