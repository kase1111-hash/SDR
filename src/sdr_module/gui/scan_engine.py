"""
The frequency scanner's engine: where to tune next, and what was found.

It is driven by the window's display loop: every block of samples the
receiver reads is also handed to :meth:`ScanEngine.frame`. The engine never
reads the device itself, so the spectrum, the waterfall, the audio and the
decoders keep running while it scans, and a signal it stops on is heard.
The window (the *host*, see :class:`ScanHost`) tunes the receiver for it,
mutes the audio between channels and says whether the squelch is open.

Modes (:data:`SCAN_MODES`):

* ``once`` / ``continuous``: the receiver (LO) steps across the range a
  page at a time, about 80 % of the sample rate per step; each page is
  measured from the averaged spectrum of its first blocks (see
  :mod:`sdr_module.dsp.sweep`) and every signal above the threshold is
  reported. ``continuous`` starts again at the end of the range.
* ``stop``: sweeps continuously, but stops on each signal found: it tunes to
  it in the band's mode, waits for the squelch to open, listens while it
  is open and goes on after ``resume_s`` seconds of quiet. Skip goes on at
  once (and passes that channel by for the rest of the scan); Hold stays.
* ``bookmarks``: tunes each bookmark in turn, in its own mode, filter and
  tone squelch, and stops on the ones that are active, as ``stop`` does.

Time is counted in samples (signal time), so the engine behaves the same
whatever the display frame rate, and tests can drive it block by block.
No Qt.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..dsp.sweep import (
    AUTO_MARGIN_DB,
    ChannelReading,
    SpectrumAverager,
    SweepPage,
    measure_page,
    noise_floor_db,
    plan_sweep,
    power_spectrum_dbfs,
    same_signal,
)

logger = logging.getLogger(__name__)

#: (key, menu text) of the scan modes.
SCAN_MODES: Tuple[Tuple[str, str], ...] = (
    ("once", "Sweep once"),
    ("continuous", "Sweep continuously"),
    ("stop", "Stop on signal"),
    ("bookmarks", "Scan bookmarks"),
)
MODE_NAMES = dict(SCAN_MODES)

#: Listening settings: (demod, fm_deviation, bandwidth); "" = keep.
Listening = Tuple[str, str, str]

# Engine states.
IDLE = "idle"
SETTLE = "settle"  # after a retune: blocks that may predate it are dropped
MEASURE = "measure"  # averaging a page's spectrum
DWELL = "dwell"  # on a channel, waiting for the squelch to open
LISTEN = "listen"  # stopped on an active channel

#: Seconds of samples averaged per page (a broadcast station's spectrum
#: needs about 30 ms to show its whole channel).
MEASURE_S = 0.03
#: Segments (of 2048 samples) at least averaged per page.
MIN_SEGMENTS = 8
#: Seconds the squelch gets to open on a channel before the scan goes on,
#: and how long a CTCSS tone gets to be recognised once it has.
DWELL_S = 0.25
TONE_WAIT_S = 0.9
#: Why a bookmark scan ended: Skip passed every bookmark by.
ALL_SKIPPED = "all skipped"
#: Why a sweep couldn't start, or stopped: no step of the range tuned.
NO_STEP_TUNED = "The receiver could not tune to any step of this range."


class ScanHost:
    """What the engine needs from the window (the main window implements it;
    tests fake it). All tuning goes through here."""

    def scan_tune_lo(self, center_hz: float, vfo_hz: Optional[float] = None) -> bool:
        """Retune the receiver to ``center_hz``, the tuned frequency in the
        middle (or at ``vfo_hz``, a channel near it). False if it can't."""
        raise NotImplementedError

    def scan_tune_channel(
        self, freq_hz: float, listening: Listening, tone_hz: float = 0.0
    ) -> bool:
        """Tune to ``freq_hz`` (inside the span if it fits) and listen to it
        with ``listening`` (and a CTCSS tone squelch, 0 for none).

        A bookmark with a DCS code squelch is tuned with the code as the
        keyword ``dcs`` too (``"D023N"``); a host that takes none is tuned
        as before."""
        raise NotImplementedError

    def scan_lo_moved(self) -> bool:
        """Whether the last tune retuned the receiver (its blocks may then
        hold samples from before), rather than moving within the span."""
        return True

    def scan_lo_hz(self) -> float:
        """The receiver's centre frequency."""
        raise NotImplementedError

    def scan_signal_present(self) -> bool:
        """Whether the tuned channel is heard: the squelch is open (and a
        tone squelch hears its tone)."""
        raise NotImplementedError

    def scan_squelch_open(self) -> bool:
        """Whether the squelch is open, whatever the tone squelch says."""
        return self.scan_signal_present()

    def scan_set_muted(self, muted: bool) -> None:
        """Mute the speaker while the scan moves between channels."""

    def scan_settled(self) -> None:
        """The blocks dropped after a retune are over (they may have held
        the last channel): what is heard is decided afresh from here."""

    def scan_refusal_reason(self) -> str:
        """What the receiver said when it last refused a tune (a driver's
        error), for the scan's one line about the steps it skipped (see
        :meth:`ScanEngine._tell_refused`); "" when it just said no."""
        return ""


@dataclass
class ScanSettings:
    """What to scan."""

    mode: str = "once"
    start_hz: float = 88e6
    end_hz: float = 108e6
    step_hz: float = 200e3
    #: Peak level (dBFS per bin) a signal must exceed; None: Auto, the
    #: measured noise floor plus ``margin_db``.
    threshold_db: Optional[float] = None
    margin_db: float = AUTO_MARGIN_DB
    #: Narrower signals are not listed (a broadcast band lists stations).
    min_width_hz: float = 0.0
    #: How a hit is listened to (the band preset's mode); None: guess per
    #: hit with ``guess_listening``.
    listening: Optional[Listening] = None
    #: Stop on signal / bookmarks: go on after this long without a signal.
    resume_s: float = 3.0
    #: Bookmarks mode: the bookmarks to visit, in order, and where to start.
    bookmarks: List[Dict[str, Any]] = field(default_factory=list)
    start_index: int = 0


@dataclass
class Found:
    """A signal the scan found (or a bookmark it stopped on)."""

    frequency_hz: float
    peak_db: float
    noise_db: float
    width_hz: float
    listening: Listening
    label: str = ""  # a bookmark's name
    heard: bool = False  # the scan stopped on it and the squelch opened
    tone_hz: float = 0.0  # a bookmark's CTCSS tone squelch (0: none)
    dcs: str = ""  # a bookmark's DCS code squelch ("D023N"; "": none)

    @property
    def snr_db(self) -> float:
        return self.peak_db - self.noise_db


class ScanEngine:
    """Runs one scan at a time; see the module docstring.

    Observers (plain callables, set by the UI):

    * ``on_found(found)``: a signal was found (sweeps) or heard (stop and
      bookmark scans);
    * ``on_state()``: the state, the page or the channel changed;
    * ``on_finished(reason)``: the scan ended (``"complete"``,
      ``"stopped"`` or an error text).
    """

    def __init__(
        self,
        host: ScanHost,
        guess_listening: Optional[Callable[[ChannelReading], Listening]] = None,
    ):
        self.host = host
        self.guess_listening = guess_listening or (lambda reading: ("", "", ""))
        self.on_found: Callable[[Found], None] = lambda found: None
        self.on_state: Callable[[], None] = lambda: None
        self.on_finished: Callable[[str], None] = lambda reason: None
        self.settings = ScanSettings()
        self.state = IDLE
        self.rate = 0.0
        self.settle_s = 0.0
        self.pages: List[SweepPage] = []
        self.page_index = 0
        self.pass_number = 0
        self.noise_db: Optional[float] = None
        self.threshold_db: Optional[float] = None
        self.found_count = 0
        self.steps_refused = 0
        self.steps_measured = 0
        # What the receiver refused to tune in this pass (told once, at its
        # end) and what was told last.
        self._refused_hz: List[float] = []
        self._refused_told: Set[float] = set()
        # The channel being dwelt on / listened to, and its bookmark index.
        self.current: Optional[Found] = None
        self.bookmark_index = -1
        self.held = False
        self._averager = SpectrumAverager(max_segments=32)
        self._skip_frames = 0
        self._settle_left = 0.0
        self._settle_then = MEASURE
        self._measured = 0
        self._candidates: List[Found] = []
        self._dwell_s = 0.0
        self._quiet_s = 0.0
        self._opened = False
        self._locked_out: List[Tuple[float, float]] = []  # (freq, width)
        self._skipped_visits: set = set()  # bookmark scan: indices skipped
        self._bookmarks_left = 0  # visits left in this pass
        self._starting = False
        self._start_failure = ""
        #: When the last block arrived (time.monotonic()), for a watchdog.
        self.last_block_at = 0.0

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self.state != IDLE

    @property
    def mode(self) -> str:
        return self.settings.mode

    def start(self, settings: ScanSettings, sample_rate: float, settle_s: float) -> str:
        """Start a scan; returns "" or why it can't start."""
        if self.active:
            self.stop("stopped")
        self.settings = settings
        self.rate = float(sample_rate)
        self.settle_s = max(0.0, float(settle_s))
        self.pass_number = 1
        self.found_count = 0
        self.steps_refused = 0
        self.steps_measured = 0
        self._refused_hz = []
        self._refused_told = set()
        self.noise_db = None
        self.threshold_db = None
        self.current = None
        self.held = False
        self._locked_out = []
        self._skipped_visits = set()
        self._candidates = []
        self.last_block_at = time.monotonic()
        if self.rate <= 0:
            return "The receiver's sample rate is unknown."
        if settings.mode == "bookmarks":
            if not settings.bookmarks:
                return "There are no bookmarks to scan."
            self.pages = []
            self.bookmark_index = (
                min(max(int(settings.start_index), 0), len(settings.bookmarks) - 1) - 1
            )
            self._bookmarks_left = len(settings.bookmarks)
            kick_off = self._next_bookmark
        else:
            self.pages = plan_sweep(
                settings.start_hz, settings.end_hz, settings.step_hz, self.rate
            )
            if not self.pages:
                return "End must be above Start."
            self.page_index = -1
            kick_off = self._next_page
        # A scan that can't even begin (nothing tunes) says so here, without
        # the observers hearing of a scan that never ran.
        self._starting = True
        self._start_failure = ""
        try:
            self.host.scan_set_muted(True)
            kick_off()
        except Exception as exc:  # a scan that broke as it began
            logger.warning("The scan could not start: %s", exc)
            self._finish(f"error: {exc}")
        finally:
            self._starting = False
        if not self.active:
            self.host.scan_set_muted(False)
            return self._start_failure or "The scan could not start."
        return ""

    def stop(self, reason: str = "stopped") -> None:
        """End the scan (the receiver stays where it is)."""
        if not self.active:
            return
        self._finish(reason)

    def skip(self) -> None:
        """Go on from the channel stopped on, and don't stop there again
        during this scan."""
        if self.state not in (DWELL, LISTEN) or self.current is None:
            return
        if self.settings.mode == "stop":
            self._locked_out.append((self.current.frequency_hz, self.current.width_hz))
        elif self.settings.mode == "bookmarks":
            self._skipped_visits.add(self.bookmark_index)
        self.held = False
        self._safely(self._go_on)

    def hold(self, on: bool) -> None:
        """Stay on the channel stopped on (until released, Skip or Stop)."""
        self.held = bool(on) and self.state in (DWELL, LISTEN)
        self._quiet_s = 0.0
        self.on_state()

    def _safely(self, action: Callable[[], None]) -> None:
        """Run a step of the scan; an error ends the scan, saying why,
        instead of leaving it half-way."""
        try:
            action()
        except Exception as exc:
            logger.warning("The scan stopped because of an error: %s", exc)
            self.stop(f"error: {exc}")

    def progress(self) -> float:
        """How far through the current pass (0..1)."""
        if self.settings.mode == "bookmarks":
            total = max(1, len(self.settings.bookmarks))
            return min(1.0, (total - self._bookmarks_left) / total)
        if not self.pages:
            return 0.0
        return min(1.0, max(0, self.page_index) / len(self.pages))

    def current_page(self) -> Optional[SweepPage]:
        if 0 <= self.page_index < len(self.pages):
            return self.pages[self.page_index]
        return None

    # ------------------------------------------------------------------ #
    # The display loop's blocks
    # ------------------------------------------------------------------ #
    def frame(self, samples: np.ndarray) -> None:
        """Take one block the receiver read (after the window processed it)."""
        if not self.active or samples is None or len(samples) == 0:
            return
        self.last_block_at = time.monotonic()
        seconds = len(samples) / self.rate
        try:
            if self.state == SETTLE:
                self._settle(len(samples))
            elif self.state == MEASURE:
                self._measure(samples)
            elif self.state in (DWELL, LISTEN):
                self._listen(samples, seconds)
        except Exception as exc:  # keep the display loop alive
            logger.warning("The scan stopped because of an error: %s", exc)
            self.stop(f"error: {exc}")

    def _settle(self, count: int) -> None:
        if self._skip_frames > 0:
            self._skip_frames -= 1  # may hold samples from before the retune
            if self._skip_frames or self._settle_left > 0:
                return
        else:
            self._settle_left -= count
            if self._settle_left > 0:
                return
        if self._settle_then == MEASURE:
            self.state = MEASURE
            self._averager.reset()
            self._measured = 0
        else:
            # (A host written before this hook may not have it.)
            settled = getattr(self.host, "scan_settled", None)
            if settled is not None:
                settled()
            self._begin_dwell()

    def _measure(self, samples: np.ndarray) -> None:
        self._averager.add(samples)
        self._measured += len(samples)
        if (
            self._measured < MEASURE_S * self.rate
            or self._averager.segments < MIN_SEGMENTS
        ):
            return
        page = self.current_page()
        power = self._averager.power_db()
        if page is None or power is None:
            self._next_page()
            return
        self.steps_measured += 1
        s = self.settings
        readings, noise, threshold = measure_page(
            power,
            self._receiver_lo(page),
            self.rate,
            page.channels,
            s.step_hz,
            threshold_db=s.threshold_db,
            margin_db=s.margin_db,
            min_width_hz=s.min_width_hz,
        )
        self.noise_db, self.threshold_db = noise, threshold
        found = []
        for reading in readings:
            low, high = s.start_hz - s.step_hz / 2, s.end_hz + s.step_hz / 2
            if not low <= reading.frequency_hz <= high:
                continue  # a signal just outside the range, seen from inside
            listening = s.listening or self.guess_listening(reading)
            found.append(
                Found(
                    reading.frequency_hz,
                    reading.peak_db,
                    reading.noise_db,
                    reading.width_hz,
                    listening,
                )
            )
        if s.mode in ("once", "continuous"):
            for item in found:
                self.found_count += 1
                self.on_found(item)
            self._next_page()
            return
        # Stop on signal: visit each one (not passed by) in turn.
        self._candidates = [
            item
            for item in found
            if not any(
                same_signal(f, w, item.frequency_hz, s.step_hz)
                for f, w in self._locked_out
            )
        ]
        self._next_candidate()

    def _receiver_lo(self, page: SweepPage) -> float:
        """The LO a page's blocks were received around: the host's (should
        the receiver not be exactly where the page was planned, the results
        still say where the signals are), else the planned one."""
        try:
            lo_hz = float(self.host.scan_lo_hz())
        except (NotImplementedError, TypeError, ValueError):
            return page.lo_hz
        return lo_hz if math.isfinite(lo_hz) else page.lo_hz

    def _listen(self, samples: np.ndarray, seconds: float) -> None:
        current = self.current
        if current is None:
            self._go_on()
            return
        heard = bool(self.host.scan_signal_present())
        if heard:
            self._measure_channel(samples)
        if self.state == DWELL:
            if heard:
                self._begin_listening()
                return
            if self.host.scan_squelch_open():
                # A signal the tone squelch holds back (the bookmark's tone,
                # or the receiver's own for a channel kept in its mode):
                # give its tone time to show.
                self._opened = True
            self._dwell_s += seconds
            limit = TONE_WAIT_S if self._opened else DWELL_S
            if self._dwell_s >= limit and not self.held:
                self._go_on()
            return
        # Listening: go on after resume_s of quiet (unless held).
        if heard:
            self._quiet_s = 0.0
            return
        self._quiet_s += seconds
        if self._quiet_s >= self.settings.resume_s and not self.held:
            self._go_on()

    def _measure_channel(self, samples: np.ndarray) -> None:
        """The level of the channel listened to (for the results)."""
        current = self.current
        if current is None:
            return
        power = power_spectrum_dbfs(samples, max_segments=8)
        n = len(power)
        if n < 32:
            return
        bin_hz = self.rate / n
        try:
            lo_hz = float(self.host.scan_lo_hz())
        except (NotImplementedError, TypeError, ValueError):
            return
        offset = current.frequency_hz - lo_hz
        half = max(0.5 * current.width_hz, 2.0 * bin_hz)
        first = max(int(n // 2 + math.floor((offset - half) / bin_hz)), 0)
        last = min(int(n // 2 + math.ceil((offset + half) / bin_hz)), n - 1)
        if last < first:
            return
        peak = float(np.max(power[first : last + 1]))
        if not math.isfinite(current.peak_db) or peak > current.peak_db:
            current.peak_db = peak
            current.noise_db = noise_floor_db(power)

    # ------------------------------------------------------------------ #
    # Moving on
    # ------------------------------------------------------------------ #
    def _finish(self, reason: str) -> None:
        self._tell_refused()
        self.state = IDLE
        self._candidates = []
        self.host.scan_set_muted(False)
        if self._starting:
            self._start_failure = reason
            return
        self.on_state()
        self.on_finished(reason)

    def _tell_refused(self) -> None:
        """One line in Tools > Error History for what the receiver refused
        to tune in this pass, not one a step (the window holds those back,
        see ``WindowScannerHost``); none when a continuous scan meets the
        ones it told already."""
        refused, self._refused_hz = self._refused_hz, []
        if not refused:
            self._refused_told = set()  # (it tunes them again)
            return
        # (A host written before this hook may not have it.)
        reason = str(getattr(self.host, "scan_refusal_reason", lambda: "")() or "")
        if set(refused) <= self._refused_told:
            return
        self._refused_told = set(refused)
        count = len(refused)
        noun = "bookmark" if self.settings.mode == "bookmarks" else "step"
        shown = ", ".join(f"{freq / 1e6:.3f}" for freq in refused[:3])
        more = f" and {count - 3} more" if count > 3 else ""
        said = f" The receiver said: {reason.rstrip('.')}." if reason else ""
        logger.warning(
            "Scan: %d %s%s the receiver couldn't tune %s skipped: %s MHz%s.%s",
            count,
            noun,
            "" if count == 1 else "s",
            "was" if count == 1 else "were",
            shown,
            more,
            said,
        )

    def _settle_after_tune(self, then: str, retuned: bool = True) -> None:
        """Drop the blocks that may predate a tune, then ``then``."""
        self.state = SETTLE
        self._settle_then = then
        self._skip_frames = 1 if retuned else 0
        self._settle_left = self.settle_s * self.rate if retuned else 0.0

    def _next_page(self) -> None:
        self.current = None
        for _attempt in range(len(self.pages) + 1):
            self.page_index += 1
            if self.page_index >= len(self.pages):
                if self.settings.mode == "once":
                    # (Still starting: every step was refused.)
                    self._finish(NO_STEP_TUNED if self._starting else "complete")
                    return
                self._tell_refused()
                self.pass_number += 1
                self.page_index = 0
            page = self.pages[self.page_index]
            # The tuned frequency shows the channel nearest the LO (the LO
            # sits a few bins off it), not an odd number.
            if self.host.scan_tune_lo(page.lo_hz, page.nearest_channel(page.lo_hz)):
                self.host.scan_set_muted(True)
                self._settle_after_tune(MEASURE)
                self.on_state()
                return
            # Out of what the receiver tunes (the range is planned inside
            # it, so this is rare): on to the next one.
            self.steps_refused += 1
            self._refused_hz.append(page.lo_hz)
        self._finish(NO_STEP_TUNED)

    def _next_candidate(self) -> None:
        while self._candidates:
            item = self._candidates.pop(0)
            self.current = item
            if self.host.scan_tune_channel(item.frequency_hz, item.listening, 0.0):
                self._settle_after_tune(DWELL, self.host.scan_lo_moved())
                if self.state == SETTLE and not self._skip_frames:
                    self._begin_dwell()
                else:
                    self.on_state()
                return
            self._refused_hz.append(item.frequency_hz)
        self.current = None
        self._next_page()

    def _next_bookmark(self) -> None:
        bookmarks = self.settings.bookmarks
        for _attempt in range(len(bookmarks)):
            if self._bookmarks_left <= 0:
                self._tell_refused()
                self.pass_number += 1
                self._bookmarks_left = len(bookmarks)
            self._bookmarks_left -= 1
            self.bookmark_index = (self.bookmark_index + 1) % len(bookmarks)
            if self.bookmark_index in self._skipped_visits:
                continue  # Skip passed it by for the rest of this scan
            b = bookmarks[self.bookmark_index]
            listening = (
                str(b.get("demod") or ""),
                str(b.get("fm_deviation") or ""),
                str(b.get("bandwidth") or ""),
            )
            tone = float(b.get("tone_hz") or 0.0)
            dcs = str(b.get("dcs") or "")
            freq = float(b.get("freq_hz") or 0.0)
            self.current = Found(
                freq,
                -math.inf,
                -math.inf,
                float(b.get("width_hz") or 10e3),
                listening,
                label=str(b.get("label") or ""),
                tone_hz=tone,
                dcs=dcs,
            )
            self.host.scan_set_muted(True)
            if dcs:
                tuned = self.host.scan_tune_channel(freq, listening, tone, dcs=dcs)
            else:
                tuned = self.host.scan_tune_channel(freq, listening, tone)
            if tuned:
                self._settle_after_tune(DWELL, self.host.scan_lo_moved())
                if self.state == SETTLE and not self._skip_frames:
                    self._begin_dwell()
                else:
                    self.on_state()
                return
            self.steps_refused += 1
            self._refused_hz.append(freq)
        if len(self._skipped_visits) >= len(bookmarks):
            self._finish(ALL_SKIPPED)
            return
        self._finish("The receiver could not tune to any of the bookmarks.")

    def _begin_dwell(self) -> None:
        self.state = DWELL
        self._dwell_s = 0.0
        self._quiet_s = 0.0
        self._opened = False
        # The squelch decides now: unmuted, so a signal is heard at once.
        self.host.scan_set_muted(False)
        self.on_state()

    def _begin_listening(self) -> None:
        self.state = LISTEN
        self._quiet_s = 0.0
        current = self.current
        if current is not None and not current.heard:
            current.heard = True
            self.found_count += 1
            self.on_found(current)
        self.on_state()

    def _go_on(self) -> None:
        """Leave the channel stopped on: the next candidate, page or
        bookmark."""
        self.held = False
        self.host.scan_set_muted(True)
        if self.settings.mode == "bookmarks":
            self._next_bookmark()
        else:
            self._next_candidate()


def bookmark_scan_list(
    bookmarks: Sequence[Dict[str, Any]],
    listening_of: Callable[[Dict[str, Any]], Tuple[str, str, str, float]],
    group: str = "",
    order: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """The bookmarks a bookmark scan visits, in order (``order``: their
    indices in the order to visit, e.g. as the Bookmarks list shows them;
    else as stored): those not marked Skip ("S"), of ``group`` when given,
    each with its ``index`` in the list, its listening settings and its
    CTCSS ``tone_hz``."""
    visits = []
    indices = range(len(bookmarks)) if order is None else order
    for index in indices:
        b = bookmarks[index]
        if str(b.get("skip") or "") == "S":
            continue
        if group and str(b.get("group") or "") != group:
            continue
        try:
            freq = float(b.get("freq_hz") or 0.0)
        except (TypeError, ValueError):
            continue
        if not freq > 0:
            continue
        demod, deviation, width, tone = listening_of(dict(b))
        visits.append(
            {
                "index": index,
                "label": str(b.get("label") or ""),
                "freq_hz": freq,
                "demod": demod,
                "fm_deviation": deviation,
                "bandwidth": width,
                "tone_hz": tone,
            }
        )
    return visits
