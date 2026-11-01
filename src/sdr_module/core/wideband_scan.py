"""
Wideband scan: two receivers sweep one range at once, each its own part.

:class:`WidebandScan` drives one or two open devices (any
:class:`~sdr_module.devices.base.SDRDevice`: an RTL-SDR and a HackRF One,
two of a kind, or demo devices wrapped in
:class:`~sdr_module.devices.polled.PolledDevice`). The range's channels are
shared out by :class:`~sdr_module.dsp.sweep.SharedSweep`: each device first
measures the channels only it tunes, then the two meet in what both tune,
one from the bottom up and the other from the top down, so the faster one
(a wider span at a higher sample rate, or quicker steps) takes more of the
range. Every page fits its own device's sample rate and tuning range.

Each device works in its own thread, the way ``sdr-scan scan`` sweeps one
device: it tunes to a page, throws away the samples queued from before,
lets the tuner settle (a transfer in flight still holds the old frequency),
throws those away too, averages the spectrum of at least
``measure_samples`` samples and measures the page's channels (see
:func:`~sdr_module.dsp.sweep.measure_page`). A page a device can't measure
(it refused the frequency, or no samples came) goes to the other device;
a device that fails ``MAX_FAILURES`` pages in a row is left out of the rest
of the pass (and tried again in the next one), and the other measures what
it can of its part.

The signals found are merged in frequency order (:class:`SignalList`): a
signal seen by both devices, where their parts meet, or on two pages, is
listed once, with its strongest reading. Progress, pass reports and the
signals found are handed to callbacks in the thread that runs the scan
(:meth:`WidebandScan.run`, or the scan's own thread after
:meth:`WidebandScan.start`); :meth:`WidebandScan.cancel` stops it.

Used by :meth:`DualSDRController.start_wideband_scan
<sdr_module.core.dual_sdr.DualSDRController.start_wideband_scan>` and
``sdr-scan scan --dual``. No Qt.
"""

from __future__ import annotations

import bisect
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..dsp.sweep import (
    AUTO_MARGIN_DB,
    ChannelReading,
    SharedSweep,
    SpectrumAverager,
    SweepPage,
    SweepReceiver,
    measure_page,
    same_signal,
)

logger = logging.getLogger(__name__)

#: After each retune: seconds the tuner gets to settle (an RTL-SDR's USB
#: transfer in flight, 0.11 s at 2.4 MS/s, still holds the old frequency).
SETTLE_S = 0.12
#: Samples averaged at least per page, and how long to wait for them.
MEASURE_SAMPLES = 65536
READ_TIMEOUT_S = 1.0
#: Pages a device may fail in a row before it is left out of the pass.
MAX_FAILURES = 3

#: Why a scan ended (:attr:`WidebandScan.reason`): it swept the range (once,
#: or a continuous scan stopped by a device), it was cancelled, or no device
#: returned samples in a whole pass. Anything else is an error's text.
COMPLETE = "complete"
CANCELLED = "cancelled"
NO_SAMPLES = (
    "No receiver returned samples. Check that they are plugged in and not "
    "used by another program."
)

# A read waits at most this long, so cancel is prompt.
_READ_SLICE_S = 0.1
# How often an idle device looks for a page the other one gave back.
_IDLE_POLL_S = 0.01
# How long the scan waits for a device's thread to end after a pass.
_JOIN_GRACE_S = 2.0


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def device_rate(device: Any) -> float:
    """The sample rate ``device`` runs at (``state.sample_rate``, else its
    ``sample_rate``), or 0.0 when it doesn't say."""
    state = getattr(device, "state", None)
    for value in (
        getattr(state, "sample_rate", None),
        getattr(device, "sample_rate", None),
    ):
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return 0.0


def device_range(device: Any) -> Tuple[float, float]:
    """``(low_hz, high_hz)`` that ``device`` tunes (its ``spec``), or
    ``(0, inf)`` when it doesn't say (a demo device tunes anything)."""
    spec = getattr(device, "spec", None)
    low = getattr(spec, "freq_min", None)
    high = getattr(spec, "freq_max", None)
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        low, high = float(low), float(high)
        if math.isfinite(low) and 0 <= low < high:
            return low, high
    return 0.0, math.inf


def device_name(device: Any, fallback: str) -> str:
    """``device.info.name``, else ``fallback``."""
    name = getattr(getattr(device, "info", None), "name", "")
    return str(name or fallback)


def _mhz(freq_hz: float) -> str:
    return f"{freq_hz / 1e6:.3f} MHz"


def _span_text(low: float, high: float) -> str:
    top = "and up" if not math.isfinite(high) else f"- {high / 1e6:.3f} MHz"
    return f"{low / 1e6:.3f} {top}"


def _drain(device: Any) -> None:
    """Throw away the samples ``device`` has queued."""
    for _ in range(1000):
        try:
            block = device.read_samples(4096, timeout=0.0)
        except Exception:  # (a device that can't be drained this way)
            return
        if block is None or len(block) == 0:
            return


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class WidebandHit:
    """A signal a wideband scan found: its strongest reading, and which
    device took it (``receiver``, an index into the scan's devices)."""

    frequency_hz: float
    peak_db: float  # strongest bin, dBFS
    noise_db: float  # the page's noise floor per bin, dBFS
    width_hz: float  # occupied width, 20 dB down from the peak
    receiver: int
    receiver_name: str = ""

    @property
    def snr_db(self) -> float:
        return self.peak_db - self.noise_db


class SignalList:
    """The signals of a sweep, in frequency order, each listed once.

    :meth:`add` keeps a reading of a signal already listed (see
    :func:`~sdr_module.dsp.sweep.same_signal`: a wide station measured on
    two pages, or by both devices where their parts meet) as one entry,
    the stronger of the two; readings outside the range (half a step either
    side) are left out. Thread-safe.
    """

    def __init__(self, start_hz: float, end_hz: float, step_hz: float) -> None:
        self.step_hz = float(step_hz)
        self.low_hz = float(start_hz) - 0.5 * self.step_hz
        self.high_hz = float(end_hz) + 0.5 * self.step_hz
        self._hits: List[WidebandHit] = []
        self._freqs: List[float] = []
        self._widest = 0.0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._hits)

    def hits(self) -> List[WidebandHit]:
        """The signals, in frequency order (a copy)."""
        with self._lock:
            return list(self._hits)

    def _same(self, freq_hz: float) -> Optional[int]:
        """The index of the entry ``freq_hz`` is a reading of (lock held)."""
        reach = max(0.75 * self.step_hz, 0.5 * self._widest)
        first = bisect.bisect_left(self._freqs, freq_hz - reach)
        last = bisect.bisect_right(self._freqs, freq_hz + reach)
        near = [
            i
            for i in range(first, last)
            if same_signal(
                self._hits[i].frequency_hz,
                self._hits[i].width_hz,
                freq_hz,
                self.step_hz,
            )
        ]
        return min(near, key=lambda i: abs(self._freqs[i] - freq_hz)) if near else None

    def _insert(self, hit: WidebandHit) -> None:
        index = bisect.bisect(self._freqs, hit.frequency_hz)
        self._freqs.insert(index, hit.frequency_hz)
        self._hits.insert(index, hit)
        self._widest = max(self._widest, hit.width_hz)

    def add(
        self, reading: ChannelReading, receiver: int, name: str = ""
    ) -> Optional[WidebandHit]:
        """Add a reading taken by ``receiver``. Returns the new entry, or
        None when it is outside the range or of a signal already listed (a
        stronger reading takes that entry's place)."""
        freq = float(reading.frequency_hz)
        if not self.low_hz <= freq <= self.high_hz:
            return None
        hit = WidebandHit(
            freq,
            float(reading.peak_db),
            float(reading.noise_db),
            float(reading.width_hz),
            int(receiver),
            str(name),
        )
        with self._lock:
            known = self._same(freq)
            if known is None:
                self._insert(hit)
                return hit
            if hit.peak_db > self._hits[known].peak_db:
                del self._hits[known]
                del self._freqs[known]
                self._insert(hit)
            return None


@dataclass(frozen=True)
class WidebandProgress:
    """How far a scan is, after each page."""

    pass_number: int
    fraction: float  # of this pass, 0..1
    steps: Tuple[int, ...]  # pages each device measured in this pass
    found: int  # signals found so far (every pass)


@dataclass(frozen=True)
class WidebandPass:
    """What one pass of a scan did."""

    number: int
    seconds: float
    #: Signals found so far (every pass), and those first found in this one.
    found: int
    new: int
    #: Pages each device measured.
    steps: Tuple[int, ...]
    #: ``(first_hz, last_hz)`` of the channels each device measured.
    covered: Tuple[Tuple[Tuple[float, float], ...], ...]
    #: ``(first_hz, last_hz)`` of channels a device tunes that this pass
    #: didn't measure (a device stopped, or refused them).
    missed: Tuple[Tuple[float, float], ...]
    #: Why each device stopped before the range was swept ("" if it
    #: didn't): it was left out after failing pages in a row, didn't start
    #: streaming, or measured nothing (the other took the pages it gave
    #: back).
    stopped: Tuple[str, ...]
    #: Pages each device couldn't measure (the other measured those it
    #: tunes).
    failed: Tuple[int, ...] = ()
    #: False when the scan was cancelled during the pass.
    complete: bool = True


# --------------------------------------------------------------------------- #
# The scan
# --------------------------------------------------------------------------- #
class WidebandScan:
    """Sweep a range with one or two devices at once; see the module
    docstring.

    The devices must be open and not streaming: the scan starts their
    streams (``start_rx()``, read with ``read_samples(n, timeout)``) and
    stops them when it ends. Set their sample rate and gain first; each
    device's rate and tuning range (``spec``) are used as they are.

    Args:
        devices: One or two devices.
        start_hz, end_hz, step_hz: The channels to measure (``end_hz``
            included; each reports the strongest signal within half a step).
        threshold_db: Level (dBFS per bin) a signal must exceed; None:
            Auto, each page's noise floor plus ``margin_db``.
        min_width_hz: Narrower signals are not listed.
        continuous: Sweep again until cancelled (each signal is listed
            once, however many passes see it).
        settle_s, measure_samples, read_timeout_s: Per page: the tuner's
            settle time, the samples averaged, how long to wait for them.
        names: What to call the devices (default: their ``info.name``).
        on_found: ``on_found(WidebandHit)`` for each new signal.
        on_progress: ``on_progress(WidebandProgress)`` after each page.
        on_pass: ``on_pass(WidebandPass)`` at the end of each pass.
        on_finished: ``on_finished(reason)`` when the scan ends (see
            :attr:`reason`).

    Raises:
        ValueError: no device or more than two, a bad range, a device whose
            sample rate is unknown, or a range no device tunes; the message
            says which.

    Thread safety: :meth:`cancel`, :meth:`wait` and the properties can be
    used from any thread. The callbacks run one at a time, in the thread
    that runs the scan; they may call :meth:`cancel`.
    """

    def __init__(
        self,
        devices: Sequence[Any],
        start_hz: float,
        end_hz: float,
        step_hz: float,
        *,
        threshold_db: Optional[float] = None,
        margin_db: float = AUTO_MARGIN_DB,
        min_width_hz: float = 0.0,
        continuous: bool = False,
        settle_s: float = SETTLE_S,
        measure_samples: int = MEASURE_SAMPLES,
        read_timeout_s: float = READ_TIMEOUT_S,
        names: Optional[Sequence[str]] = None,
        on_found: Optional[Callable[[WidebandHit], None]] = None,
        on_progress: Optional[Callable[[WidebandProgress], None]] = None,
        on_pass: Optional[Callable[[WidebandPass], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
    ) -> None:
        devices = tuple(devices)
        if not devices:
            raise ValueError("A wideband scan needs a device to scan with.")
        if len(devices) > 2:
            raise ValueError("A wideband scan takes one or two devices.")
        start, end, step = float(start_hz), float(end_hz), float(step_hz)
        if not (math.isfinite(end - start) and step > 0 and end >= start):
            raise ValueError(
                "The end of the range must be above its start, and the step " "above 0."
            )
        given = list(names or [])
        labels = [
            str(given[i]) if i < len(given) and given[i] else device_name(d, "")
            for i, d in enumerate(devices)
        ]
        labels = [label or f"Receiver {i + 1}" for i, label in enumerate(labels)]
        if len(labels) == 2 and labels[0] == labels[1]:
            labels = [f"{labels[0]} 1", f"{labels[1]} 2"]
        rates = [device_rate(d) for d in devices]
        for label, rate in zip(labels, rates, strict=True):
            if rate <= 0:
                raise ValueError(
                    f"{label}'s sample rate is unknown: set it before scanning."
                )
        ranges = [device_range(d) for d in devices]
        sweep = SharedSweep(
            start,
            end,
            step,
            [
                SweepReceiver(rate, low, high)
                for rate, (low, high) in zip(rates, ranges, strict=True)
            ],
        )
        if sweep.covered_count == 0:
            tunes = "; ".join(
                f"{label} tunes {_span_text(low, high)}"
                for label, (low, high) in zip(labels, ranges, strict=True)
            )
            raise ValueError(
                f"No receiver tunes any of {start / 1e6:.3f} - {end / 1e6:.3f} "
                f"MHz: {tunes}."
            )
        self.devices: Tuple[Any, ...] = devices
        self.names: Tuple[str, ...] = tuple(labels)
        self.rates: Tuple[float, ...] = tuple(rates)
        self.ranges: Tuple[Tuple[float, float], ...] = tuple(ranges)
        self.start_hz, self.end_hz, self.step_hz = start, end, step
        self.threshold_db = threshold_db
        self.margin_db = float(margin_db)
        self.min_width_hz = float(min_width_hz)
        self.continuous = bool(continuous)
        self.settle_s = max(0.0, float(settle_s))
        self.measure_samples = max(1, int(measure_samples))
        self.read_timeout_s = max(0.0, float(read_timeout_s))
        self.sweep = sweep
        self.on_found = on_found
        self.on_progress = on_progress
        self.on_pass = on_pass
        self.on_finished = on_finished
        self._signals = SignalList(start, end, step)
        self._passes: List[WidebandPass] = []
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._ran = False
        self._running = False
        self._reason = ""
        self._pass_number = 0
        self._thread: Optional[threading.Thread] = None
        #: The exception that ended a scan run by :meth:`start`, if any.
        self.error: Optional[BaseException] = None

    # -- what it does ------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self._running

    @property
    def reason(self) -> str:
        """Why the scan ended: :data:`COMPLETE`, :data:`CANCELLED`,
        :data:`NO_SAMPLES` or an error's text; "" while it runs."""
        return self._reason

    @property
    def hits(self) -> List[WidebandHit]:
        """The signals found so far, in frequency order."""
        return self._signals.hits()

    @property
    def passes(self) -> List[WidebandPass]:
        """The passes done so far."""
        with self._lock:
            return list(self._passes)

    @property
    def pass_number(self) -> int:
        return self._pass_number

    @property
    def progress(self) -> float:
        """How far through the current pass (0..1)."""
        return self.sweep.progress

    def seconds_per_page(self) -> List[float]:
        """About how long each device takes for a page (its settle time,
        the samples it averages and a little for the rest)."""
        return [
            self.settle_s + self.measure_samples / rate + 0.02 for rate in self.rates
        ]

    def estimate_seconds(self) -> float:
        """About how long a pass takes (see :meth:`seconds_per_page`)."""
        return self.sweep.estimate_seconds(self.seconds_per_page())

    # -- control ----------------------------------------------------------- #
    def _claim_run(self) -> None:
        with self._lock:
            if self._ran:
                raise RuntimeError(
                    "A wideband scan runs once: make a new WidebandScan to scan "
                    "again."
                )
            self._ran = True
            self._running = True

    def run(self) -> List[WidebandHit]:
        """Scan in this thread until the range is swept (or, continuous,
        until cancelled); returns the signals found, in frequency order.
        Ctrl+C (KeyboardInterrupt) cancels the scan, stops the devices and
        is raised again."""
        self._claim_run()
        self._execute()
        return self.hits

    def start(self) -> None:
        """Scan in a thread of the scan's own (see :meth:`wait`)."""
        self._claim_run()
        thread = threading.Thread(
            target=self._run_in_background, name="wideband-scan", daemon=True
        )
        self._thread = thread
        thread.start()

    def _run_in_background(self) -> None:
        try:
            self._execute()
        except BaseException as exc:  # (its own thread: keep it, say it)
            self.error = exc
            logger.warning("The wideband scan stopped because of an error: %s", exc)

    def cancel(self) -> None:
        """Stop the scan (each device finishes or drops the page it is on)."""
        self._cancel.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until the scan has ended; False if it hasn't by
        ``timeout`` seconds."""
        return self._finished.wait(timeout)

    # -- running ----------------------------------------------------------- #
    def _execute(self) -> None:
        started: List[int] = []
        reason = ""
        stopped = ["" for _ in self.devices]
        try:
            for index, device in enumerate(self.devices):
                why = self._start_device(device)
                if why:
                    stopped[index] = why
                else:
                    started.append(index)
            if not started:
                reason = "; ".join(
                    f"{name} {why}"
                    for name, why in zip(self.names, stopped, strict=True)
                )
                reason = reason[:1].upper() + reason[1:] + "."
            while started and not self._cancel.is_set():
                self._pass_number += 1
                if self._pass_number > 1:
                    self.sweep.reset()
                report = self._run_pass(self._pass_number, started, stopped)
                with self._lock:
                    self._passes.append(report)
                if self.on_pass is not None:
                    self.on_pass(report)
                if self._cancel.is_set():
                    break
                if not any(report.steps):
                    reason = NO_SAMPLES
                    break
                if not self.continuous:
                    reason = COMPLETE
                    break
            if not reason:
                reason = CANCELLED
        except KeyboardInterrupt:
            reason = CANCELLED
            raise
        except BaseException as exc:
            reason = f"error: {exc}"
            raise
        finally:
            self._cancel.set()  # (any device thread still about ends)
            for index in started:
                self._stop_device(self.devices[index], self.names[index])
            self._reason = reason
            self._running = False
            try:
                if self.on_finished is not None:
                    self.on_finished(reason)
            except Exception as exc:
                logger.warning("The wideband scan's on_finished failed: %s", exc)
            finally:
                self._finished.set()

    def _start_device(self, device: Any) -> str:
        """Start ``device``'s stream; "" or why it didn't start."""
        try:
            started = device.start_rx()
        except Exception as exc:
            return f"did not start streaming: {exc}"
        return "did not start streaming" if started is False else ""

    @staticmethod
    def _stop_device(device: Any, name: str) -> None:
        try:
            device.stop_rx()
        except Exception as exc:
            logger.warning("%s did not stop streaming cleanly: %s", name, exc)

    def _run_pass(
        self, number: int, started: List[int], not_started: List[str]
    ) -> WidebandPass:
        """One pass: a thread per device, the coordinator (this thread)
        merging what they measure and telling the callbacks."""
        began = time.monotonic()
        count = len(self.devices)
        for index in range(count):
            if index not in started:
                self.sweep.give_up(index)
        events: "queue.Queue[tuple]" = queue.Queue()
        threads = [
            threading.Thread(
                target=self._work,
                args=(index, events),
                name=f"wideband-{self.names[index]}",
                daemon=True,
            )
            for index in started
        ]
        steps = [0] * count
        failed = [0] * count
        stopped = list(not_started)
        last_failure = [""] * count
        new = 0
        try:
            for thread in threads:
                thread.start()
            while True:
                try:
                    event = events.get(timeout=0.05)
                except queue.Empty:
                    if any(t.is_alive() for t in threads) or not events.empty():
                        continue
                    break
                if event[0] == "missed":
                    failed[event[1]] += 1
                    last_failure[event[1]] = event[2]
                new += self._handle(event, number, steps, stopped)
        except BaseException:
            self._cancel.set()
            raise
        finally:
            deadline = time.monotonic() + self._join_s()
            for thread in threads:
                thread.join(max(0.0, deadline - time.monotonic()))
                if thread.is_alive():  # pragma: no cover - a hung driver
                    logger.warning("%s is still busy after the scan", thread.name)
        for index in range(count):
            if not stopped[index] and failed[index] and not steps[index]:
                stopped[index] = last_failure[index]
        return WidebandPass(
            number=number,
            seconds=time.monotonic() - began,
            found=len(self._signals),
            new=new,
            steps=tuple(steps),
            covered=tuple(tuple(self.sweep.coverage(i)) for i in range(count)),
            missed=tuple(self.sweep.missed()),
            stopped=tuple(stopped),
            failed=tuple(failed),
            complete=not self._cancel.is_set(),
        )

    def _join_s(self) -> float:
        return self.settle_s + _READ_SLICE_S + _JOIN_GRACE_S

    def _handle(self, event: tuple, number: int, steps: List[int], stopped) -> int:
        """Act on a device thread's event; returns the new signals."""
        kind, index = event[0], event[1]
        new = 0
        if kind == "stopped":
            stopped[index] = event[2]
            logger.info("%s stopped scanning: %s", self.names[index], event[2])
            return 0
        if kind == "page":
            steps[index] += 1
            for reading in event[3]:
                hit = self._signals.add(reading, index, self.names[index])
                if hit is not None:
                    new += 1
                    if self.on_found is not None:
                        self.on_found(hit)
        if self.on_progress is not None:
            self.on_progress(
                WidebandProgress(
                    pass_number=number,
                    fraction=self.sweep.progress,
                    steps=tuple(steps),
                    found=len(self._signals),
                )
            )
        return new

    # -- a device's thread ------------------------------------------------- #
    def _work(self, index: int, events: "queue.Queue[tuple]") -> None:
        """Measure pages for device ``index`` until none is left for it."""
        failures = 0
        page: Optional[SweepPage] = None
        try:
            while not self._cancel.is_set():
                page = self.sweep.claim(index)
                if page is None:
                    if not self.sweep.more_may_come(index):
                        return
                    self._cancel.wait(_IDLE_POLL_S)
                    continue
                try:
                    outcome = self._measure(index, page)
                except Exception as exc:
                    outcome = f"error: {exc}"
                if outcome is None:  # cancelled
                    self.sweep.release(index, page)
                    return
                if isinstance(outcome, list):
                    failures = 0
                    self.sweep.done(index, page)
                    events.put(("page", index, page, outcome))
                    page = None
                    continue
                failures += 1
                self.sweep.release(index, page)
                page = None
                events.put(("missed", index, outcome))
                if failures >= MAX_FAILURES:
                    self.sweep.give_up(index)
                    events.put(
                        ("stopped", index, f"{outcome} ({failures} steps in a row)")
                    )
                    return
        except BaseException as exc:  # pragma: no cover - defensive
            if page is not None:
                self.sweep.release(index, page)
            self.sweep.give_up(index)
            events.put(("stopped", index, f"error: {exc}"))

    def _measure(self, index: int, page: SweepPage):
        """Measure ``page`` on device ``index``: the readings (a list),
        why it couldn't (a string), or None when cancelled."""
        device = self.devices[index]
        tuned = device.set_frequency(page.lo_hz)
        if tuned is False:
            return f"refused to tune to {_mhz(page.lo_hz)}"
        _drain(device)
        if self._cancel.wait(self.settle_s):
            return None
        _drain(device)
        samples = self._collect(device)
        if samples is None:
            return None
        averager = SpectrumAverager(max_segments=64)
        averager.add(samples)
        power = averager.power_db()
        if power is None:
            return "returned no samples"
        readings, _noise, _threshold = measure_page(
            power,
            page.lo_hz,
            self.rates[index],
            page.channels,
            self.step_hz,
            threshold_db=self.threshold_db,
            margin_db=self.margin_db,
            min_width_hz=self.min_width_hz,
        )
        return list(readings)

    def _collect(self, device: Any) -> Optional[np.ndarray]:
        """At least ``measure_samples`` samples (fewer if they don't come
        in time), or None when cancelled."""
        blocks: List[np.ndarray] = []
        taken = 0
        deadline = time.monotonic() + self.read_timeout_s
        while taken < self.measure_samples:
            if self._cancel.is_set():
                return None
            left = deadline - time.monotonic()
            if left <= 0:
                break
            block = device.read_samples(
                self.measure_samples, timeout=min(left, _READ_SLICE_S)
            )
            if block is None or len(block) == 0:
                self._cancel.wait(0.002)  # (a device that doesn't wait)
                continue
            blocks.append(np.asarray(block))
            taken += len(block)
        if not blocks:
            return np.zeros(0, dtype=np.complex64)
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)


__all__ = [
    "CANCELLED",
    "COMPLETE",
    "MAX_FAILURES",
    "MEASURE_SAMPLES",
    "NO_SAMPLES",
    "READ_TIMEOUT_S",
    "SETTLE_S",
    "SignalList",
    "WidebandHit",
    "WidebandPass",
    "WidebandProgress",
    "WidebandScan",
    "device_name",
    "device_range",
    "device_rate",
]
