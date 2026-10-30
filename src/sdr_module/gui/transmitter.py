"""
The transmit engine: sends an I/Q job on the connected HackRF One.

Everything the app transmits goes through :class:`Transmitter` (the Ham ID
panel's Send ID Now today; Transmit Text uses it too); nothing else calls the
HackRF's transmit API. A job (:class:`TxJob`) is a source of complex
samples at its own sample rate, the center frequency it is sent on, its mode
and occupied bandwidth, and a description.

What the engine does for every job:

* Checks it first (:meth:`Transmitter.check`): a TX-capable device, a
  sample rate the HackRF can send at, and the frequency manager's legality
  check with the emission's own occupied bandwidth and mode (and those of
  any station IDs it adds), never the receiver's filter. Refusals are plain
  sentences (:func:`describe_refusal`), ready for "Can't send: ...".
* Puts the HackRF at the job's sample rate (with its narrowest baseband
  filter that passes the job), the job's center frequency, the TX gain and
  amplifier chosen (:class:`TxPower`), and puts it all back afterwards, so
  the receiver resumes as it was.
* Streams the samples from a worker thread through the driver's
  ``start_tx`` callback, one block at a time: the window never waits on the
  radio, a long job is never cut short by a timeout, and memory stays at a
  block (sources such as :class:`~sdr_module.ham.callsign.MorseIdSource`
  generate each block when asked).
* Reports progress, can be stopped at any moment (:meth:`Transmitter.stop`),
  and says how it ended (:class:`TxResult`).
* Identifies the station automatically when asked (:class:`AutoIdPolicy`):
  an ID before the job, one every interval while it runs, and one after it.

The GUI polls the worker (a QTimer in the window's thread), so every signal
the engine emits arrives in the GUI thread.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PyQt6.QtCore import QObject, QTimer, pyqtSignal

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

from ..core import frequency_manager

logger = logging.getLogger(__name__)

# The HackRF's nominal front-end amplifier gain and TX VGA range.
AMP_GAIN_DB = 14.0
TX_GAIN_MAX_DB = 47
TX_GAIN_DEFAULT_DB = 20

# HackRF One's maximum TX power (TX VGA at 47 dB with the amplifier on), by
# frequency, from the hardware documentation: 5-15 dBm rising to 10 MHz,
# falling to 5 dBm at 2150 MHz, 13-15 dBm to 2750 MHz, 5-0 dBm to 4 GHz and
# 0 to -10 dBm to 6 GHz. Interpolated over log frequency; (MHz, dBm).
_MAX_TX_DBM = (
    (1.0, 5.0),
    (10.0, 15.0),
    (2150.0, 5.0),
    (2150.01, 13.0),
    (2750.0, 15.0),
    (2750.01, 5.0),
    (4000.0, 0.0),
    (6000.0, -10.0),
)

# Samples handed to the driver per read (the driver gathers them into its
# USB transfers of 131072 samples).
_BLOCK = 65536
# libhackrf keeps this many transfers of this many samples in flight: after
# the last samples are handed over, they take this long to go out.
_IN_FLIGHT_SAMPLES = 4 * 131072
# A device that takes no samples for this long has stalled.
_STALL_S = 3.0
# A station ID put into a job: a job that can't stop at a break of its own
# fades out and back in over this long (raised cosine), so the carrier
# never steps (key clicks).
_SPLICE_FADE_S = 0.005
# How often the GUI looks at the worker.
_POLL_MS = 100

_LICENSE_NAMES = {
    "none": "None",
    "technician": "Technician",
    "general": "General",
    "amateur_extra": "Amateur Extra",
}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def format_mhz(freq_hz: float) -> str:
    """``146.52e6 -> "146.520 MHz"`` (3 to 6 decimals, like FREQ)."""
    text = f"{max(0.0, float(freq_hz)) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def format_dbm_power(dbm: float) -> str:
    """``-31.4 -> "0.7 µW (-31 dBm)"``: the power in watts and in dBm."""
    watts = 10.0 ** ((float(dbm) - 30.0) / 10.0)
    try:
        from ..ham.gui.qrp_panel import format_watts

        power = format_watts(watts)
    except Exception:  # pragma: no cover - ham GUI unavailable
        from ..ham.qrp import format_power

        power = format_power(watts)
    sign = "-" if dbm < -0.05 else "+" if dbm > 0.05 else ""
    return f"{power} ({sign}{abs(dbm):.0f} dBm)"


def license_name(license_class: Any = None) -> str:
    """The license class in words ("None", "Technician", ...)."""
    if license_class is None:
        license_class = frequency_manager.get_license_class()
    value = str(getattr(license_class, "value", license_class))
    return _LICENSE_NAMES.get(value, value.replace("_", " ").title())


def describe_refusal(reason: Optional[str], freq_hz: float, mode: str = "") -> str:
    """The frequency manager's refusal as a plain sentence fragment.

    For "Can't send on 146.520 MHz: <this>": e.g. "your license (None)
    allows CB/MURS/FRS only", "FM isn't allowed in the 10m CW segment (CW
    and DATA only)", "1575.420 MHz is a protected frequency (GPS L1:
    aircraft navigation)".
    """
    text = (reason or "").strip()
    if text.startswith("TX BLOCKED:"):
        body = text[len("TX BLOCKED:") :].strip()
        name, _sep, rest = body.partition(" - ")
        rest = rest.split("[")[0].strip()
        what = rest.split(" - ")[-1].strip().lower() if rest else ""
        detail = f"{name}: {what}" if what else name
        return f"{format_mhz(freq_hz)} is a protected frequency ({detail})"
    if text.startswith("LICENSE:"):
        body = text[len("LICENSE:") :].strip()
        lic = license_name()
        if body.startswith("No amateur license"):
            return f"your license ({lic}) allows CB/MURS/FRS only"
        match = re.match(r"Mode '(.+)' not allowed in (.+) - allowed: (.+)$", body)
        if match:
            used, segment, allowed = match.groups()
            allowed_list = [a.strip() for a in allowed.split(",") if a.strip()]
            if len(allowed_list) > 1:
                allowed = ", ".join(allowed_list[:-1]) + " and " + allowed_list[-1]
            return f"{used} isn't allowed in the {segment} segment ({allowed} only)"
        match = re.match(r"Frequency ([\d.]+) MHz requires higher license class", body)
        if match:
            edge = float(match.group(1)) * 1e6
            where = _edge_words(edge, freq_hz)
            return f"your license ({lic}) doesn't include {where}"
        match = re.match(
            r"Frequency ([\d.]+) MHz is not in any amateur or license-free band", body
        )
        if match:
            edge = float(match.group(1)) * 1e6
            where = _edge_words(edge, freq_hz)
            return f"{where} is outside the amateur and license-free bands"
        return body[:1].lower() + body[1:] if body else "transmitting isn't allowed"
    if not text:
        return "transmitting isn't allowed here"
    return text[:1].lower() + text[1:]


def _edge_words(edge_hz: float, center_hz: float) -> str:
    """``"146.520 MHz"``, or for a signal edge ``"the signal's edge at
    143.997 MHz"`` (the frequency manager names the edge that failed)."""
    if abs(edge_hz - center_hz) < 500.0:
        return format_mhz(center_hz)
    return f"the signal's edge at {format_mhz(edge_hz)}"


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


def hackrf_max_output_dbm(freq_hz: float) -> float:
    """HackRF One's maximum output (dBm) at ``freq_hz``: TX gain 47 dB and
    the amplifier on (interpolated from the hardware documentation)."""
    mhz = min(max(float(freq_hz) / 1e6, _MAX_TX_DBM[0][0]), _MAX_TX_DBM[-1][0])
    xs = np.log10([point[0] for point in _MAX_TX_DBM])
    ys = [point[1] for point in _MAX_TX_DBM]
    return float(np.interp(math.log10(mhz), xs, ys))


@dataclass
class TxPower:
    """The HackRF's transmit power settings: TX VGA gain (0-47 dB) and the
    +14 dB front-end amplifier."""

    gain_db: int = TX_GAIN_DEFAULT_DB
    amp: bool = False

    def clamped(self) -> "TxPower":
        return TxPower(
            int(min(max(int(self.gain_db), 0), TX_GAIN_MAX_DB)), bool(self.amp)
        )

    def output_dbm(self, freq_hz: float) -> float:
        """Estimated output at the HackRF's antenna port (dBm): the maximum
        at this frequency, less the TX gain below 47 dB and the amplifier
        when it is off. Real output varies from unit to unit by a few dB."""
        power = self.clamped()
        drop = (TX_GAIN_MAX_DB - power.gain_db) + (0.0 if power.amp else AMP_GAIN_DB)
        return hackrf_max_output_dbm(freq_hz) - drop

    def describe(self, freq_hz: float) -> str:
        """``"TX gain 20 dB, amp off: about 0.7 µW (-31 dBm)"``."""
        power = self.clamped()
        amp = "amp on (+14 dB)" if power.amp else "amp off"
        estimate = format_dbm_power(self.output_dbm(freq_hz))
        return f"TX gain {power.gain_db} dB, {amp}: about {estimate}"


# --------------------------------------------------------------------------- #
# Jobs and sources
# --------------------------------------------------------------------------- #


class ArraySource:
    """Streams a complex array (the simplest job source)."""

    def __init__(self, samples: np.ndarray):
        self._samples = np.asarray(samples, dtype=np.complex64).ravel()
        self.total_samples = int(self._samples.size)
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def reset(self) -> None:
        self._pos = 0

    def read(self, count: int) -> np.ndarray:
        end = min(self._pos + max(0, int(count)), self.total_samples)
        block = self._samples[self._pos : end]
        self._pos = end
        return block


def _was_read(source: Any) -> bool:
    """Whether ``source`` has been read (its ``position`` is past 0)."""
    try:
        return int(getattr(source, "position", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def as_source(source: Any) -> Any:
    """A job's source as an object with ``read(n)`` and ``total_samples``
    (a plain array is wrapped in :class:`ArraySource`)."""
    if isinstance(source, np.ndarray) or isinstance(source, (list, tuple)):
        return ArraySource(np.asarray(source))
    if not (hasattr(source, "read") and hasattr(source, "total_samples")):
        raise TypeError("a job's source needs read(n) and total_samples")
    return source


@dataclass
class TxJob:
    """One transmission.

    Attributes:
        source: The samples: a complex array, or an object with
            ``read(n) -> complex64 array`` (empty at the end) and
            ``total_samples`` (generated as it is read). Sending reads it;
            a job sent again starts it again with its ``reset()`` (the
            built-in sources have one; one without is refused once read).
        sample_rate: The rate the samples are for (the HackRF is put at it).
        center_hz: Where the emission is centered (the HackRF's frequency;
            the samples are at baseband around 0 Hz).
        mode: What the frequency manager checks: "CW", "FM", "AM", "SSB",
            "USB", "LSB" or "DATA".
        bandwidth_hz: The emission's occupied bandwidth (checked against
            band and segment edges).
        description: What is sent, for messages ("DE W1AW").
        summary: How, for the confirmation ("FM tone, 20 WPM").
        is_id: The job is itself a station ID.
        identify: Wrap it in automatic station IDs (see
            :class:`AutoIdPolicy`); ignored for an ID.
    """

    source: Any
    sample_rate: float
    center_hz: float
    mode: str
    bandwidth_hz: float
    description: str
    summary: str = ""
    is_id: bool = False
    identify: bool = True

    def __post_init__(self) -> None:
        self.source = as_source(self.source)

    @property
    def total_samples(self) -> int:
        return int(self.source.total_samples)

    @property
    def duration_s(self) -> float:
        rate = float(self.sample_rate)
        return self.total_samples / rate if rate > 0 else 0.0


@dataclass
class AutoIdPolicy:
    """When the engine adds station IDs to a job (not to an ID itself).

    ``at_start``: an ID before the job. ``interval_s``: one whenever that
    long has passed since the last ID began (None: no interval IDs).
    ``at_end``: one after the job (the rules ask for an ID at the end of
    every communication and at least every 10 minutes during it).
    """

    at_start: bool = False
    at_end: bool = True
    interval_s: Optional[float] = 600.0

    def any(self) -> bool:
        return bool(self.at_start or self.at_end or self.interval_s)


@dataclass
class TxResult:
    """How a transmission ended.

    ``status``: "sent" (all of it), "stopped" (Stop pressed) or "failed".
    ``message``: a sentence for the status bar. ``seconds``: air time sent.
    """

    status: str
    message: str
    job: TxJob
    seconds: float = 0.0
    ids_sent: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "sent"


# Makes the ID that goes with a job: an ID job at the job's center and rate
# (or None when there is no callsign to send).
IdFactory = Callable[[TxJob], Optional[TxJob]]


class _Silence:
    """Key up (no carrier) for ``total_samples`` samples: the word space
    around a station ID put into a job."""

    def __init__(self, count: int):
        self.total_samples = max(0, int(count))
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def reset(self) -> None:
        self._pos = 0

    def read(self, count: int) -> np.ndarray:
        n = max(0, min(int(count), self.total_samples - self._pos))
        self._pos += n
        return np.zeros(n, dtype=np.complex64)


def _rise(k: np.ndarray, length: int) -> np.ndarray:
    """Raised-cosine rise over ``length`` samples at sample(s) ``k``."""
    x = np.clip((np.asarray(k, np.float64) + 0.5) / max(1, int(length)), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def id_goes_first(due: float, min_gap: float, job_length: float) -> bool:
    """Whether the interval ID that falls ``due`` into a job (``due`` and
    ``min_gap`` in the units of ``job_length``) goes before it instead: it
    would come before the job ends, and before half an interval of it
    (e.g. the last ID was 9 min 58 s ago and "At start" is off)."""
    return due < job_length and due < min_gap


class _Segment:
    """One part of a transmission: the job, a station ID, or the key-up
    around an ID put into the job."""

    JOB, ID, GAP = "job", "id", "gap"

    __slots__ = ("source", "is_id", "begun", "kind")

    def __init__(self, source: Any, is_id: bool, kind: Optional[str] = None):
        self.source = source
        self.is_id = bool(is_id)
        self.begun = False
        self.kind = kind or (self.ID if is_id else self.JOB)


class _Plan:
    """What a transmission sends, in order: the job, with station IDs
    before, during (every interval) and after it as the policy asks.

    An interval ID never cuts the job at a random place: it goes in at the
    last break before it is due that the job's source reports (the end of
    a word, see :meth:`~sdr_module.protocols.text_modes.TextTxSource.last_break`;
    the source then stops and starts again cleanly), else at the moment it
    is due, the job's carrier fading out and back in around it. A word
    space of key-up sets it apart on each side. When the interval would run
    out early in the job (a short job soon after the last transmission's
    ID), the ID goes before the job instead.

    :meth:`read` is called by the driver from libhackrf's USB thread; the
    GUI only reads the counters.
    """

    def __init__(
        self,
        job: TxJob,
        policy: Optional[AutoIdPolicy],
        make_id: Optional[IdFactory],
        since_last_id_s: Optional[float],
        cancel: threading.Event,
    ):
        self.job = job
        self.rate = float(job.sample_rate)
        self._cancel = cancel
        self._make_id = make_id
        self._segments: Deque[_Segment] = deque()
        self.ids_sent = 0
        self.ids_begun = 0
        # Samples handed over before the last ID began (None: no ID yet).
        self.last_id_began_at: Optional[int] = None
        self.error: Optional[str] = None
        self.exhausted = False
        self.handed = 0
        identify = bool(
            job.identify and not job.is_id and policy is not None and policy.any()
        ) and (make_id is not None)
        self._interval: Optional[int] = None
        if identify and policy is not None and policy.interval_s:
            self._interval = max(1, int(float(policy.interval_s) * self.rate))
        # Air time since the last ID began (samples): from an earlier
        # transmission, else from the start of this one.
        since = since_last_id_s if since_last_id_s is not None else 0.0
        self._since_id = int(max(0.0, since) * self.rate)
        # Job air time since the last ID ended: interval IDs never come back
        # to back (an ID longer than the interval would otherwise starve
        # the job); at least half an interval of the job goes between them.
        self._job_since_id = self._since_id
        self._min_gap = (self._interval or 0) // 2
        # The job's samples read so far, and where in them the next
        # interval ID goes (None: not chosen yet, or none falls in the job):
        # that ID, the key-up around it, and whether the job's source stops
        # there itself (else the plan fades the job out and back in).
        self._job_read = 0
        self._looked = False
        self._splice_at: Optional[int] = None
        self._splice_id: Optional[TxJob] = None
        self._splice_gap = 0
        self._splice_shaped = False
        self._resumed_at: Optional[int] = None
        self._fade = max(1, int(round(_SPLICE_FADE_S * self.rate)))
        # The IDs made up front (before and after the job): what is checked.
        self.id_jobs: List[TxJob] = []
        first = bool(identify and policy is not None and policy.at_start)
        if identify and not first and self._interval is not None:
            # An interval ID due early in the job goes before it.
            first = id_goes_first(
                self._interval - self._since_id, self._min_gap, job.total_samples
            )
        if first:
            self._add_id(front=False)
        self._segments.append(_Segment(job.source, job.is_id))
        if identify and policy is not None and policy.at_end:
            self._add_id(front=False)
        self.total = sum(int(seg.source.total_samples) for seg in self._segments)
        # What is checked before sending: the job and its IDs (with only
        # interval IDs, one made now stands for them: its mode and width).
        self.check_jobs: List[TxJob] = list(self.id_jobs)
        if identify and self._interval is not None and not self.check_jobs:
            sample = self._make_id(job) if self._make_id is not None else None
            if sample is not None:
                self.check_jobs.append(sample)

    def _new_id(self) -> Optional[TxJob]:
        if self._make_id is None:
            return None
        try:
            return self._make_id(self.job)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Could not make the station ID: {e}")
            return None

    def _add_id(self, front: bool) -> bool:
        id_job = self._new_id()
        if id_job is None:
            return False
        self.id_jobs.append(id_job)
        segment = _Segment(id_job.source, True)
        if front:
            self._segments.appendleft(segment)
        else:
            self._segments.append(segment)
        return True

    def _choose_splice(self, source: Any) -> None:
        """Where in the job the next interval ID goes: the last break the
        source reports that lets the ID begin by the time it is due, else
        that time (the job fades out and in around it); nothing when the
        job ends first."""
        self._looked = True
        assert self._interval is not None
        due = self._job_read + max(
            0, self._interval - self._since_id, self._min_gap - self._job_since_id
        )
        if due >= int(source.total_samples):
            return  # (the job ends first)
        id_job = self._new_id()
        if id_job is None:
            self._interval = None  # (no ID to send: no interval IDs)
            return
        word_space = float(getattr(id_job.source, "word_space_s", 0.0) or 0.0)
        gap = int(round(word_space * self.rate))
        # The ID itself (after its key-up) begins by the time it is due, but
        # never before half an interval of the job has gone out since the
        # last one (the IDs never starve the job), and with room for the
        # fade before it.
        latest = self._job_read + max(
            self._fade,
            self._interval - self._since_id - gap,
            self._min_gap - self._job_since_id,
        )
        at: Optional[int] = None
        finder = getattr(source, "last_break", None)
        if callable(finder):
            try:
                found = finder(latest)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not find a break for the station ID: {e}")
                found = None
            if found is not None and self._job_read < int(found) <= latest:
                at = int(found)
        shaped = False
        pauser = getattr(source, "pause_at", None)
        if at is not None and callable(pauser):
            before = int(source.total_samples)
            try:
                pauser(at)
                shaped = True
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"The job couldn't stop for the station ID: {e}")
            self.total += int(source.total_samples) - before
        self._splice_at = at if at is not None else latest
        self._splice_id, self._splice_gap = id_job, gap
        self._splice_shaped = shaped

    def _splice(self) -> None:
        """Put the interval ID chosen by :meth:`_choose_splice` in front of
        the rest of the job, with a word space of key-up on each side."""
        id_job, gap = self._splice_id, self._splice_gap
        self._splice_at = self._splice_id = None
        self._looked = False
        if id_job is None:  # pragma: no cover - defensive
            return
        parts = [_Segment(id_job.source, True)]
        if gap:
            parts = [
                _Segment(_Silence(gap), False, _Segment.GAP),
                parts[0],
                _Segment(_Silence(gap), False, _Segment.GAP),
            ]
        for part in reversed(parts):
            self._segments.appendleft(part)
        self.total += sum(int(part.source.total_samples) for part in parts)
        if not self._splice_shaped:
            self._resumed_at = self._job_read  # (fade the job back in)

    def _shape(self, block: np.ndarray) -> np.ndarray:
        """The job's ``block`` (from job sample :attr:`_job_read`) faded out
        before an interval ID and back in after it, where its source doesn't
        stop cleanly itself."""
        start, n, fade = self._job_read, len(block), self._fade
        gain: Optional[np.ndarray] = None
        where = start + np.arange(n)
        if (
            self._splice_at is not None
            and not self._splice_shaped
            and start + n > self._splice_at - fade
        ):
            gain = _rise(self._splice_at - 1 - where, fade)
        if self._resumed_at is not None:
            if start < self._resumed_at + fade:
                rise = _rise(where - self._resumed_at, fade)
                gain = rise if gain is None else gain * rise
            if start + n >= self._resumed_at + fade:
                self._resumed_at = None
        if gain is None:
            return block
        return (np.asarray(block) * gain).astype(np.complex64)

    @property
    def modes(self) -> Tuple[str, ...]:
        modes = [self.job.mode] + [j.mode for j in self.check_jobs]
        return tuple(dict.fromkeys(m for m in modes if m))

    @property
    def bandwidth_hz(self) -> float:
        return max([self.job.bandwidth_hz] + [j.bandwidth_hz for j in self.check_jobs])

    def next_id_in_s(self) -> Optional[float]:
        """Air time until the next interval ID, or None (no interval IDs)."""
        if self._interval is None or self.exhausted:
            return None
        at = self._splice_at  # (once: the transmit thread may change it)
        if at is not None:  # (its place in the job is chosen)
            return (max(0, at - self._job_read) + self._splice_gap) / self.rate
        due = max(self._interval - self._since_id, self._min_gap - self._job_since_id)
        return max(0, due) / self.rate

    def read(self) -> Optional[np.ndarray]:
        """The next block for the driver (None: nothing more to send)."""
        if self._cancel.is_set():
            return None
        try:
            while self._segments:
                segment = self._segments[0]
                is_job = segment.kind == _Segment.JOB
                if is_job and self._interval is not None:
                    if not self._looked:
                        self._choose_splice(segment.source)
                    if (
                        self._splice_at is not None
                        and self._job_read >= self._splice_at
                    ):
                        self._splice()
                        continue
                if segment.is_id and not segment.begun:
                    self._since_id = 0  # an ID begins
                    self._job_since_id = 0
                    self.ids_begun += 1
                    self.last_id_began_at = self.handed
                segment.begun = True
                count = _BLOCK
                if is_job and self._splice_at is not None:
                    # Up to the ID's place, the last block whole (its fade).
                    left = self._splice_at - self._job_read
                    count = left if left <= _BLOCK + self._fade else _BLOCK
                block = segment.source.read(count)
                if len(block):
                    if is_job:
                        block = self._shape(block)
                        self._job_read += len(block)
                        self._job_since_id += len(block)
                    self.handed += len(block)
                    self._since_id += len(block)
                    return block
                self._segments.popleft()
                if segment.is_id:
                    self.ids_sent += 1
        except Exception as e:
            logger.error(f"Transmit source failed: {e}")
            self.error = f"the samples could not be generated ({e})"
            return None
        self.exhausted = True
        return None


class _Run:
    """One transmission in progress (shared between the GUI and the worker)."""

    def __init__(
        self,
        job: TxJob,
        device: Any,
        plan: _Plan,
        saved: dict,
        cancel: threading.Event,
    ):
        self.job = job
        self.device = device
        self.plan = plan
        self.saved = saved
        self.cancel = cancel  # (the plan stops reading when it is set)
        self.done = threading.Event()
        self.status = "sent"
        self.message = ""
        self.started_at = time.monotonic()
        self.on_air_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.thread: Optional[threading.Thread] = None

    def air_seconds(self) -> float:
        """Air time so far (no more than what was handed to the radio)."""
        if self.on_air_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        handed = self.plan.handed / max(self.plan.rate, 1.0)
        return max(0.0, min(end - self.on_air_at, handed))


def _hackrf_class() -> Any:
    try:
        from ..devices.hackrf import HackRFDevice
    except Exception:  # pragma: no cover - defensive
        return None
    return HackRFDevice


def can_transmit(device: Any) -> bool:
    """Whether ``device`` is a TX-capable HackRF One."""
    cls = _hackrf_class()
    return device is not None and cls is not None and isinstance(device, cls)


def _state(device: Any) -> Any:
    return getattr(device, "state", None)


def _tx_filter(device: Any, occupied_hz: float, rate: float) -> float:
    """The narrowest baseband filter the device has that passes the job:
    its occupied bandwidth and 3/4 of its sample rate (the images of the
    DAC stay out)."""
    wanted = max(float(occupied_hz), 0.75 * float(rate))
    choices = sorted(getattr(device, "SUPPORTED_BANDWIDTHS", None) or [wanted])
    return next((bw for bw in choices if bw >= wanted), choices[-1])


class Transmitter(QObject if HAS_PYQT6 else object):  # type: ignore[misc]
    """Sends :class:`TxJob` s on a HackRF One, one at a time, without
    blocking the GUI (see the module docstring).

    Signals (all in the GUI thread):
        started(TxJob): the radio was set up and the transmission began.
        progress(float, float): seconds sent, seconds in all (it grows when
            an interval ID is added).
        finished(TxResult): it ended (sent, stopped or failed), and the
            device is back as it was.
        on_air_changed(bool): True from started to finished.
    """

    if HAS_PYQT6:
        started = pyqtSignal(object)
        progress = pyqtSignal(float, float)
        finished = pyqtSignal(object)
        on_air_changed = pyqtSignal(bool)

    def __init__(self, parent: Any = None, poll_ms: int = _POLL_MS):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required for the transmitter")
        super().__init__(parent)
        self._run: Optional[_Run] = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_ms))
        self._timer.timeout.connect(self._poll)
        self._last_id_at: Optional[float] = None  # monotonic start of last ID
        self.last_refusal = ""
        self.last_result: Optional[TxResult] = None

    # ------------------------------------------------------------- state

    @property
    def busy(self) -> bool:
        """True from :meth:`start` until ``finished`` was emitted."""
        return self._run is not None

    @property
    def job(self) -> Optional[TxJob]:
        return self._run.job if self._run is not None else None

    def seconds_sent(self) -> float:
        return self._run.air_seconds() if self._run is not None else 0.0

    def seconds_total(self) -> float:
        if self._run is None:
            return 0.0
        return self._run.plan.total / max(self._run.plan.rate, 1.0)

    def next_id_in(self) -> Optional[float]:
        """Seconds until the next automatic ID of the running transmission
        (None: none due, or not transmitting)."""
        if self._run is None:
            return None
        return self._run.plan.next_id_in_s()

    def seconds_since_last_id(self) -> Optional[float]:
        """Seconds since the last station ID began (None: none sent yet)."""
        if self._last_id_at is None:
            return None
        return time.monotonic() - self._last_id_at

    def expected_ids(self, job: TxJob, policy: Optional[AutoIdPolicy]) -> int:
        """About how many station IDs :meth:`start` adds to ``job`` with
        ``policy`` now (for the air time the confirmation gives): one
        before it (asked for, or the interval ID due early in it), one per
        interval while it runs, and one after it."""
        if job.is_id or not job.identify or policy is None or not policy.any():
            return 0
        first = bool(policy.at_start)
        interval = float(policy.interval_s or 0.0)
        during = 0
        if interval > 0:
            if not first:
                since = max(0.0, self.seconds_since_last_id() or 0.0)
                first = id_goes_first(interval - since, interval / 2.0, job.duration_s)
            during = int(job.duration_s // interval)
        return int(first) + during + int(bool(policy.at_end))

    # ------------------------------------------------------------ checks

    def check(
        self, job: TxJob, device: Any, id_jobs: Sequence[TxJob] = ()
    ) -> Optional[str]:
        """Why ``job`` can't be sent on ``device`` (None: it can).

        The device must be a HackRF One, idle, and able to send at the
        job's sample rate; the job must have samples; and the frequency
        manager must allow the job's emission (and each ID's) at its
        center, with the license class set.
        """
        if self.busy:
            return "a transmission is already on the air"
        if device is None:
            return "no device is connected. Connect a HackRF One"
        if not can_transmit(device):
            return "this device can only receive. Transmitting needs a HackRF One"
        if not getattr(device, "is_open", True):
            return "the HackRF is not open. Connect it again (Device > Connect...)"
        spec = getattr(device, "spec", None)
        low = float(getattr(spec, "sample_rate_min", 2e6) or 2e6)
        high = float(getattr(spec, "sample_rate_max", 20e6) or 20e6)
        rate = float(job.sample_rate)
        if not low - 1.0 <= rate <= high + 1.0:
            return (
                f"the HackRF sends at {low / 1e6:g}-{high / 1e6:g} MS/s, and this "
                f"transmission is made for {rate / 1e6:g} MS/s"
            )
        freq_low = float(getattr(spec, "freq_min", 1e6) or 1e6)
        freq_high = float(getattr(spec, "freq_max", 6e9) or 6e9)
        if not freq_low <= float(job.center_hz) <= freq_high:
            return f"the HackRF transmits on {freq_low / 1e6:g}-{freq_high / 1e6:g} MHz"
        if job.total_samples <= 0:
            return "there is nothing to send"
        if _was_read(job.source) and not callable(getattr(job.source, "reset", None)):
            return "this transmission was already sent: make a new one to send it again"
        for item in (job, *id_jobs):
            problem = legality_problem(item.center_hz, item.bandwidth_hz, item.mode)
            if problem:
                return problem
        return None

    # ------------------------------------------------------------ control

    def start(
        self,
        job: TxJob,
        device: Any,
        power: Optional[TxPower] = None,
        policy: Optional[AutoIdPolicy] = None,
        make_id: Optional[IdFactory] = None,
    ) -> bool:
        """Set the HackRF up for ``job`` and start sending it.

        False (and :attr:`last_refusal` says why) when it can't be sent;
        otherwise True, then ``started``, ``progress`` and ``finished``
        follow. The receiver must be stopped first (the HackRF is half
        duplex).
        """
        self.last_refusal = ""
        if self.busy:  # (before touching a job that may be the one on the air)
            self.last_refusal = "a transmission is already on the air"
            logger.info(f"Transmission refused: {self.last_refusal}")
            return False
        reset = getattr(job.source, "reset", None)
        if _was_read(job.source) and callable(reset):
            reset()  # (sent before: from its start again)
        power = (power or TxPower()).clamped()
        cancel = threading.Event()
        since = self.seconds_since_last_id()
        plan = _Plan(job, policy, make_id, since, cancel)
        problem = self.check(job, device, plan.check_jobs)
        if problem is None and getattr(_state(device), "is_streaming", False):
            problem = "the receiver is still running"
        if problem is None and getattr(_state(device), "is_transmitting", False):
            problem = "the HackRF is already transmitting"
        if problem is not None:
            self.last_refusal = problem
            logger.info(f"Transmission refused: {problem}")
            return False
        saved = self._prepare(job, device, power)
        if isinstance(saved, str):
            self.last_refusal = saved
            logger.warning(f"Transmission not started: {saved}")
            return False
        run = _Run(job, device, plan, saved, cancel)
        self._run = run
        thread = threading.Thread(
            target=self._work, args=(run,), name="sdr-transmit", daemon=True
        )
        run.thread = thread
        logger.info(
            f"Transmitting {job.description} on {format_mhz(job.center_hz)} "
            f"({job.mode}, {job.bandwidth_hz / 1e3:.2f} kHz, "
            f"{job.sample_rate / 1e6:g} MS/s, {power.describe(job.center_hz)})"
        )
        thread.start()
        self._timer.start()
        self.started.emit(job)
        self.on_air_changed.emit(True)
        return True

    def stop(self) -> None:
        """Stop the transmission now (``finished`` follows, "stopped")."""
        run = self._run
        if run is not None and not run.done.is_set():
            run.cancel.set()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop, wait for the worker and finish (e.g. before the device is
        closed or the window quits)."""
        run = self._run
        if run is None:
            return
        run.cancel.set()
        if run.thread is not None:
            run.thread.join(timeout)
        if not run.done.is_set():
            # The radio didn't let go (a hung USB link): give up on it, so
            # the window can close the device and transmit again later.
            logger.warning("The transmit thread didn't stop in time")
            run.status = "failed"
            run.message = "the HackRF didn't stop in time"
            if run.ended_at is None:
                run.ended_at = time.monotonic()
            run.done.set()
        self._poll()

    def wait(self, timeout: float = 10.0) -> bool:
        """Wait (pumping nothing) until the worker is done, then finish.
        For tests and scripts; True when it ended within ``timeout``."""
        run = self._run
        if run is None:
            return True
        ended = run.done.wait(timeout)
        if ended:
            if run.thread is not None:
                run.thread.join(1.0)
            self._poll()
        return ended

    # ------------------------------------------------------------ device

    def _prepare(self, job: TxJob, device: Any, power: TxPower) -> Any:
        """Put the HackRF at the job's rate, filter, frequency and power.

        Returns what to restore afterwards, or the reason it failed (the
        device is then put back at once).
        """
        state = _state(device)
        saved = {
            "rate": float(getattr(state, "sample_rate", 0.0) or 0.0),
            "bandwidth": float(getattr(state, "bandwidth", 0.0) or 0.0),
            "frequency": float(getattr(state, "frequency", 0.0) or 0.0),
            "amp": bool(getattr(device, "amp_enabled", False)),
        }
        try:
            rate = float(job.sample_rate)
            if abs(saved["rate"] - rate) > 0.5:
                if device.set_sample_rate(rate) is False:
                    raise RuntimeError(f"it didn't accept {rate / 1e6:g} MS/s")
                now = float(getattr(_state(device), "sample_rate", rate) or rate)
                if abs(now - rate) > 1.0:
                    raise RuntimeError(
                        f"it runs at {now / 1e6:g} MS/s, not {rate / 1e6:g}"
                    )
            # The narrowest baseband filter that passes the job (1.75 MHz,
            # the HackRF's narrowest, for anything up to 2.3 MS/s).
            device.set_bandwidth(_tx_filter(device, job.bandwidth_hz, rate))
            if abs(saved["frequency"] - float(job.center_hz)) > 0.5:
                if device.set_frequency(float(job.center_hz)) is False:
                    raise RuntimeError(f"it didn't tune to {format_mhz(job.center_hz)}")
            if device.set_tx_gain(power.gain_db) is False:
                raise RuntimeError("it didn't take the TX gain")
            if device.set_amp(power.amp) is False:
                raise RuntimeError("it didn't switch the amplifier")
        except Exception as e:
            self._restore(device, saved)
            return f"the HackRF couldn't be set up to transmit: {e}"
        return saved

    @staticmethod
    def _restore(device: Any, saved: dict) -> None:
        """Put the HackRF back as it was before the transmission."""
        state = _state(device)
        try:
            if (
                saved.get("rate")
                and abs(
                    float(getattr(state, "sample_rate", 0.0) or 0.0) - saved["rate"]
                )
                > 0.5
            ):
                device.set_sample_rate(saved["rate"])
            if saved.get("bandwidth"):
                device.set_bandwidth(saved["bandwidth"])
            if (
                saved.get("frequency")
                and abs(
                    float(getattr(_state(device), "frequency", 0.0) or 0.0)
                    - saved["frequency"]
                )
                > 0.5
            ):
                device.set_frequency(saved["frequency"])
            if bool(getattr(device, "amp_enabled", False)) != saved.get("amp", False):
                device.set_amp(saved.get("amp", False))
        except Exception as e:  # pragma: no cover - device-dependent
            logger.warning(f"Could not restore the HackRF after transmitting: {e}")

    # ------------------------------------------------------------ worker

    def _work(self, run: _Run) -> None:
        """Worker thread: start the HackRF streaming, wait for the end (or
        Stop, or a stall), let the last transfers go out, stop."""
        device, plan = run.device, run.plan
        waiter = getattr(device, "wait_tx_done", None)
        try:
            ok = device.start_tx(
                plan.read, bandwidth_hz=plan.bandwidth_hz, mode=plan.modes
            )
            if not ok:
                reason = getattr(device, "last_tx_error", None) or "it refused"
                run.status = "failed"
                run.message = describe_refusal(reason, run.job.center_hz)
                return
            run.on_air_at = time.monotonic()
            last_handed, last_change = plan.handed, run.on_air_at
            while True:
                if run.cancel.is_set():
                    run.status = "stopped"
                    break
                if callable(waiter):
                    if waiter(0.05):
                        break
                else:
                    if plan.exhausted:
                        break
                    time.sleep(0.05)
                now = time.monotonic()
                if plan.handed != last_handed:
                    last_handed, last_change = plan.handed, now
                elif now - last_change > _STALL_S:
                    run.status = "failed"
                    run.message = "the HackRF stopped taking samples"
                    break
            if run.status == "sent":
                error = plan.error or getattr(device, "last_tx_error", None)
                if error:
                    run.status = "failed"
                    run.message = error
                elif run.cancel.is_set():
                    run.status = "stopped"
                elif not plan.exhausted:
                    run.status = "failed"
                    run.message = "the HackRF stopped transmitting early"
                elif run.cancel.wait(
                    min(1.0, _IN_FLIGHT_SAMPLES / max(plan.rate, 1.0) + 0.05)
                ):
                    # Stopped while the last transfers were going out.
                    run.status = "stopped"
        except Exception as e:
            logger.error(f"Transmission failed: {e}")
            run.status = "failed"
            run.message = str(e)
        finally:
            try:
                device.stop_tx()
            except Exception as e:  # pragma: no cover - device-dependent
                logger.warning(f"stop_tx failed: {e}")
            run.ended_at = time.monotonic()
            run.done.set()

    # ------------------------------------------------------------ GUI side

    def _poll(self) -> None:
        run = self._run
        if run is None:
            self._timer.stop()
            return
        plan = run.plan
        self.progress.emit(run.air_seconds(), plan.total / max(plan.rate, 1.0))
        if not run.done.is_set():
            return
        if run.thread is not None:
            run.thread.join(1.0)
        self._timer.stop()
        self._restore(run.device, run.saved)
        self._run = None
        result = self._result(run)
        self.last_result = result
        logger.info(f"Transmission ended: {result.message}")
        self.on_air_changed.emit(False)
        self.finished.emit(result)

    def _result(self, run: _Run) -> TxResult:
        job, plan = run.job, run.plan
        seconds = run.air_seconds()
        if run.status == "sent":
            seconds = max(seconds, plan.handed / max(plan.rate, 1.0))
        if plan.last_id_began_at is not None and run.on_air_at is not None:
            # When the last ID began on air: the next transmission's
            # interval ID is timed from it.
            self._last_id_at = run.on_air_at + plan.last_id_began_at / max(
                plan.rate, 1.0
            )
        where = format_mhz(job.center_hz)
        if run.status == "sent":
            message = f"Sent {job.description} on {where} ({seconds:.1f} s)"
        elif run.status == "stopped":
            message = f"Stopped {job.description} on {where} after {seconds:.1f} s"
        else:
            reason = run.message or "it failed"
            message = f"{job.description} not sent on {where}: {reason}"
        return TxResult(run.status, message, job, seconds, plan.ids_sent)


@contextmanager
def quiet_checks() -> Iterator[None]:
    """Silence the frequency manager's log while only *inspecting* (the TX
    line, the QRP panel): it warns about every blocked frequency, which is
    right for a real attempt but would flood Tools > Error History while
    the user merely tunes around."""
    log = logging.getLogger(frequency_manager.__name__)

    def _drop(_record: logging.LogRecord) -> bool:
        return False

    log.addFilter(_drop)
    try:
        yield
    finally:
        log.removeFilter(_drop)


def legality_problem(
    center_hz: float, bandwidth_hz: float, mode: str, quiet: bool = False
) -> Optional[str]:
    """Why the frequency manager refuses this emission (None: allowed),
    as a plain sentence fragment (see :func:`describe_refusal`).
    ``quiet``: only looking (nothing is logged)."""
    if quiet:
        with quiet_checks():
            allowed, reason = frequency_manager.is_tx_allowed(
                float(center_hz), float(bandwidth_hz), mode
            )
    else:
        allowed, reason = frequency_manager.is_tx_allowed(
            float(center_hz), float(bandwidth_hz), mode
        )
    if allowed:
        return None
    return describe_refusal(reason, center_hz, mode)


__all__ = [
    "AMP_GAIN_DB",
    "ArraySource",
    "AutoIdPolicy",
    "TX_GAIN_DEFAULT_DB",
    "TX_GAIN_MAX_DB",
    "Transmitter",
    "TxJob",
    "TxPower",
    "TxResult",
    "as_source",
    "can_transmit",
    "describe_refusal",
    "format_dbm_power",
    "format_mhz",
    "hackrf_max_output_dbm",
    "id_goes_first",
    "legality_problem",
    "license_name",
    "quiet_checks",
]
