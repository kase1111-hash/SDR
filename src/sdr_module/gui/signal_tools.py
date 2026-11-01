"""
The window side of the Signal tab and of Radio > Track Signal (AFC).

:class:`SignalTools` is built by the main window with its
:class:`~.signal_panel.SignalPanel`. The display loop hands it every block
(:meth:`SignalTools.frame`: the raw samples, the tuned channel, the
spectrum frame and the receiver chain), and it does only what is on
screen or asked for:

* **Identify** (Tools > Identify Signal, Ctrl+I; "Identify Signal Here" on
  the spectrum and waterfall): collects about half a second around the
  tuned frequency (longer for a signal that comes and goes, up to 4 s) and
  classifies it in a worker thread; **Apply Mode** sets Mode, deviation and
  Bandwidth and centres the tuned frequency on the signal.
* **Signals in view**: the spectrum averaged over :data:`LIST_FRAMES`
  frames (half a second), its signals listed (and marked on the spectrum
  when asked); Home, Page Up and Page Down (plots focused) tune to the
  strongest, the next one up and the next one down.
* **Scope**: the tuned channel, only while the Scope page is on screen.
* **Bursts**: :class:`~sdr_module.dsp.bursts.BurstWatcher` over the span,
  while the Bursts page is on screen (or always, when asked).
* **Track Signal (AFC)**: :class:`~sdr_module.dsp.afc.SignalTracker`
  nudges the tuned frequency onto a drifting signal (the LO is retuned only
  when the tuned channel would leave the span); "AFC +1.2 kHz" in the
  status bar.
* **Find Signals in Recording** (File menu, the playback strip): runs
  :class:`~sdr_module.dsp.signal_list.SignalSurvey` over a file in the
  background and lists what it found; double-click plays the file there.
"""

from __future__ import annotations

import bisect
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..dsp.afc import TRACK_METHODS, SignalTracker
from ..dsp.bursts import BurstReport, BurstWatcher, analyze_report
from ..dsp.signal_id import (
    WAIT_S,
    Identification,
    SignalIdentifier,
    analysis_band,
    better,
    identify,
    takes_needed,
)
from ..dsp.signal_list import SignalPeak, SignalSurvey, find_signals

try:
    from PyQt6.QtCore import QObject, QTimer
    from PyQt6.QtWidgets import QLabel

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

logger = logging.getLogger(__name__)

#: The display frames each refresh of the signals list is measured from
#: (their mean spectrum). One frame shows a wide FM station only where its
#: modulation was at that instant, a narrow peak off its centre: a list
#: from a few frames held such pieces, and the keys tuned to them.
LIST_FRAMES = 15
#: How often the signals list is refreshed (s, at the display's 30 frames
#: a second), and how long a signal no longer heard stays listed.
LIST_EVERY_S = LIST_FRAMES / 30.0
FORGET_S = 10.0
#: Bursts waiting for their analysis (a busy band): beyond this many, a
#: burst is listed without one rather than late.
MAX_PENDING_BURSTS = 8
#: The Bursts list keeps this many (the newest).
MAX_LISTED_BURSTS = 500
#: The most of the receiver's time the bursts worker may spend taking
#: bursts apart (over the last few seconds of signal): beyond it, and while
#: the display loop can't keep up, a burst is listed without its analysis,
#: so the worker never starves the plots and the audio.
ANALYSIS_SHARE = 0.25
_ANALYSIS_WINDOW_S = 4.0
#: Broadcast FM: wide deviations Track Signal leaves alone.
_BROADCAST_DEVIATION_HZ = 50e3


def _format_offset(hz: float) -> str:
    """``"+1.2 kHz"``, ``"-350 Hz"``."""
    if abs(hz) >= 1e3:
        return f"{hz / 1e3:+.1f} kHz"
    return f"{hz:+.0f} Hz"


if HAS_PYQT6:

    from .signal_panel import tune_frequency
    from .themes import set_role, set_tone

    class SignalTools(QObject):
        """See the module docstring."""

        def __init__(self, window: Any, panel: Any):
            super().__init__(window)
            self._win = window
            self.panel = panel
            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="identify"
            )
            # Identify.
            self._identifier: Optional[SignalIdentifier] = None
            self._ident_job: Optional[Future] = None
            self._ident_best: Optional[Identification] = None
            self._ident_freq = 0.0
            self._ident_reach = 0.0
            self._ident_band: Tuple[float, float] = (0.0, 0.0)
            self._ident_takes = 0
            self._ident_kinds: List[str] = []  # what each take found
            self._ident_lost_s = 0.0  # signal dropped while listening
            # The device and its count of dropped samples at the last frame:
            # a capture must not span samples lost in between.
            self._drops: Optional[Tuple[int, int]] = None
            # Signals in view.
            self._acc: Optional[np.ndarray] = None
            self._acc_frames = 0
            self._acc_key: Optional[Tuple[int, float, float]] = None
            self._heard: List[Tuple[SignalPeak, float]] = []
            self._markers = False
            self._listing_until = 0.0
            # Whether the list is kept up to date (measured, and every frame
            # since; not while off screen or stopped), and the last mean
            # spectrum it was measured from, with its span and time.
            self._list_current = False
            self._last_mean: Optional[
                Tuple[Tuple[int, float, float], np.ndarray, float]
            ] = None
            # Bursts: each finished one is taken apart off the display loop
            # (50-200 ms each), and listed in order once it is.
            self._watcher = BurstWatcher(defer=True)
            self._watch_always = False
            self._burst_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="bursts"
            )
            self._burst_jobs: Deque[Tuple[Future, BurstReport, float]] = deque()
            # (signal time, seconds) of the analyses done lately.
            self._analysis_cost: Deque[Tuple[float, float]] = deque()
            self._bursts_paused = False  # (the display loop is behind)
            # Each listed burst's time on the waterfall's clock and its
            # frequencies (the selected one is outlined there), from the
            # watcher's time and the waterfall's at the end of each frame.
            self._burst_spans: Dict[int, Tuple[float, float, float, float]] = {}
            self._frame_times: Deque[Tuple[float, float]] = deque(maxlen=4096)
            self._bursts_shown = False  # (the Bursts page was on screen)
            self._burst_timer = QTimer(self)
            self._burst_timer.setInterval(100)
            self._burst_timer.timeout.connect(self._collect_bursts)
            # AFC.
            self._tracker: Optional[SignalTracker] = None
            self._tracking = False
            self._tracker_key: Optional[Tuple[str, float, float]] = None
            self._moving = False
            self._last_frame_t: Optional[float] = None
            self._afc_badge: Optional[QLabel] = None
            # A recording's signals (Find Signals in Recording).
            self._survey_options: Any = None
            self._shut = False  # the window closed: nothing more to do
            self._connect()

        # -- wiring ------------------------------------------------------------------

        def _connect(self) -> None:
            win, panel = self._win, self.panel
            panel.identify_requested.connect(lambda: self.identify())
            panel.apply_requested.connect(self.apply_identification)
            panel.tune_requested.connect(self._tune)
            panel.bookmark_requested.connect(win._bookmark_frequency)
            panel.markers_changed.connect(self.set_markers)
            panel.min_snr_changed.connect(lambda _v: self._min_snr_changed())
            panel.step_requested.connect(self.step)
            panel.watch_changed.connect(self._set_watch_always)
            panel.play_requested.connect(self.play_recording_at)
            panel.live_requested.connect(self._reset_list)
            panel.page_changed.connect(lambda _p: self._page_changed())
            panel.burst_selected.connect(self._mark_burst)
            panel.export_bursts_btn.clicked.connect(
                lambda _c=False: self.export_bursts()
            )
            panel.message.connect(
                lambda text, tone: win._show_status_message(text, tone or None, 3000)
            )
            for plot in (win._spectrum, win._waterfall):
                add = getattr(plot, "add_context_action", None)
                if callable(add):
                    add(
                        "&Identify Signal Here",
                        lambda freq: self.identify(freq),
                        "Tune here and say what the signal is (Signal tab)",
                    )
            bar = getattr(win, "_playback_bar", None)
            if bar is not None and hasattr(bar, "find_clicked"):
                bar.find_clicked.connect(self.find_in_current_recording)

        def shutdown(self) -> None:
            """The window closes: no more work in the background, and no
            more signals from the panel (its widgets emit some while Qt
            deletes them, when nothing of theirs may be touched). A burst
            in progress is dropped: the receiver stopping later (the
            device closing) has nothing left to hand the closed worker."""
            self._shut = True
            self._identifier = None
            self._tracking = False
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._burst_timer.stop()
            self._burst_jobs.clear()
            self._burst_pool.shutdown(wait=False, cancel_futures=True)
            self._watcher.reset()
            for obj in [self.panel, *self.panel.findChildren(QObject)]:
                obj.blockSignals(True)

        # -- the display loop -----------------------------------------------------------

        def frame(
            self,
            samples: np.ndarray,
            channel: np.ndarray,
            power: np.ndarray,
            chain: Any,
        ) -> None:
            """One block of the receiver (from the window's display loop)."""
            if self._shut:
                return
            now = time.monotonic()
            dt = 1.0 / 30.0
            rate = float(self._win._device_sample_rate())
            if rate > 0 and len(samples):
                dt = len(samples) / rate
            self._last_frame_t = now
            try:
                self._identify_frame(samples, self._samples_lost(rate))
                self._signals_frame(power, now)
                self._scope_frame(channel, chain)
                self._bursts_frame(samples, rate)
                self._afc_frame(samples, dt, chain)
            except Exception as exc:  # pragma: no cover - never stop the loop
                logger.warning("Signal tab: %s", exc, exc_info=True)

        def sync_state(self, running: bool, has_device: bool) -> None:
            """The receiver started, stopped or the device changed."""
            if running or self._shut:
                return
            if self._identifier is not None:
                self._identifier = None
                self.panel.set_identify_busy(False)
                self.panel.set_identify_status(
                    "Stopped before the signal could be identified. Press "
                    "Start (Space), then Identify.",
                    "warning",
                )
            self._queue_bursts(self._watcher.flush())
            if not self.panel.showing_recording():
                self.panel.set_signals_empty(
                    "Receiver stopped\nPress Start (Space): the signals in view "
                    "are listed here."
                    if has_device
                    else "No receiver\nUse Device > Use Demo Device or "
                    "Connect... to list the signals in view."
                )
            self._reset_list()
            self._apply_markers([])
            if self._tracking:
                self._set_paused_badge()

        def frequency_changed(self, freq_hz: float) -> None:
            """The tuned frequency changed (the window's
            ``_on_frequency_changed``): by AFC itself, or by the user."""
            if self._moving:
                return
            if self._identifier is not None:
                # Another frequency: listen there afresh.
                offset = float(self._win._vfo_offset_hz())
                if abs(freq_hz - self._ident_freq) > 0.5:
                    self._ident_freq = float(freq_hz)
                    self._identifier.set_offset(offset)
                    self._ident_best = None
                    self._ident_takes = 0
                    self._ident_kinds = []
                    self._ident_lost_s = 0.0
            if self._tracker is not None:
                self._tracker.retuned(float(self._win._vfo_offset_hz()))
                if self._win._is_running:
                    # Nothing measured here yet: not locked on.
                    self._update_badge(0.0, False, False, False)
                else:
                    self._set_paused_badge()

        # -- Identify -----------------------------------------------------------------------

        def identify(self, freq_hz: Optional[float] = None) -> bool:
            """Identify the signal at the tuned frequency (or tune to
            ``freq_hz`` first). Starts the receiver if it is stopped."""
            win = self._win
            win._show_panel("signal")
            self.panel.show_page("identify")
            if freq_hz is not None and not win.set_frequency(float(freq_hz)):
                return False  # refused: the status bar says why
            if win._device is None:
                self.panel.set_identify_status(
                    "No receiver: use Device > Use Demo Device or Connect..., "
                    "then Identify.",
                    "warning",
                )
                return False
            if not win._is_running:
                win._start_acquisition()
                if not win._is_running:
                    return False
            panel = win._control_panel
            mode = panel._demod_combo.currentText()
            rate = float(win._device_sample_rate())
            self._ident_freq = float(win._current_frequency())
            width, reach = analysis_band(
                mode, win._channel_bandwidth(), rate, self._ident_freq
            )
            self._identifier = SignalIdentifier(rate, win._vfo_offset_hz(), width)
            self._ident_reach = reach
            self._ident_band = self._identifier.passband_hz
            self._ident_best = None
            self._ident_takes = 0
            self._ident_kinds = []
            self._ident_lost_s = 0.0
            self.panel.set_identify_busy(True)
            self.panel.set_identify_status(
                f"Listening to {_mhz(self._ident_freq)}…", None
            )
            return True

        def _samples_lost(self, rate: float) -> float:
            """Seconds of signal the device dropped since the last frame (its
            queue overflowed, or the demo skipped ahead to catch up)."""
            device = self._win._device
            try:
                dropped = int(device.dropped_samples)
            except (AttributeError, TypeError, ValueError):
                self._drops = None
                return 0.0
            before, self._drops = self._drops, (id(device), dropped)
            if before is None or before[0] != id(device) or dropped <= before[1]:
                return 0.0
            return (dropped - before[1]) / rate if rate > 0 else 0.0

        def _identify_frame(self, samples: np.ndarray, lost_s: float = 0.0) -> None:
            ident = self._identifier
            job = self._ident_job
            if job is not None and job.done():
                self._ident_job = None
                try:
                    result = job.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Identify failed: %s", exc)
                    result = None
                if result is not None and ident is not None:
                    if abs(result.tuned_hz - self._ident_freq) < 0.5:
                        self._ident_takes += 1
                        self._ident_kinds.append(result.kind)
                        self._ident_best = better(self._ident_best, result)
                        self._maybe_finish()
            ident = self._identifier
            if ident is None:
                return
            if abs(ident.offset_hz - self._win._vfo_offset_hz()) > 0.5:
                ident.set_offset(self._win._vfo_offset_hz())
            if lost_s > 0:
                ident.gap()  # (no capture spans the samples lost)
                self._ident_lost_s += lost_s
            ident.feed(samples)
            seconds = ident.seconds
            if self._ident_job is None and ident.ready():
                capture = ident.take()
                rate, freq, reach, band = (
                    ident.rate,
                    self._ident_freq,
                    self._ident_reach,
                    self._ident_band,
                )
                self._ident_job = self._pool.submit(
                    identify, capture, rate, freq, reach, None, band
                )
            if self._ident_best is None or not self._ident_best.is_signal:
                self.panel.set_identify_status(
                    f"Listening to {_mhz(self._ident_freq)}… {seconds:.1f} s"
                    + ("" if seconds < 1.0 else " (waiting for a signal)"),
                    None,
                )
            if (
                self._ident_job is None
                and seconds + self._ident_lost_s >= WAIT_S
                and not ident.ready()
            ):
                # No unbroken half second to judge in all that time (the
                # receiver keeps dropping samples): what was heard, if any.
                self._maybe_finish(give_up=True)

        def _maybe_finish(self, give_up: bool = False) -> None:
            best = self._ident_best
            ident = self._identifier
            if ident is None:
                return
            if best is None:
                if give_up:
                    self._identifier = None
                    self.panel.set_identify_busy(False)
                    self.panel.set_identify_status(
                        f"Couldn't listen to {_mhz(self._ident_freq)} for half a "
                        "second without a break: the receiver keeps dropping "
                        "samples (this computer can't keep up). A lower sample "
                        "rate helps.",
                        "warning",
                    )
                return
            # Steady carriers and short pulses wait for more takes that
            # agree (keying, or the whole packet, may yet show).
            agree = self._ident_kinds.count(best.kind)
            enough = best.is_signal and (
                (best.confidence >= 0.6 and agree >= takes_needed(best))
                or self._ident_takes >= 3
            )
            if not enough and ident.seconds < WAIT_S and not give_up:
                return
            self._identifier = None
            self.panel.set_identify_busy(False)
            self.panel.show_identification(best)
            stamp = time.strftime("%H:%M:%S")
            if best.is_signal:
                self.panel.set_identify_status(
                    f"{_mhz(best.tuned_hz)} at {stamp}, from "
                    f"{best.seconds:.1f} s of signal. Apply Mode listens to it "
                    "that way.",
                    None,
                )
                self._win._show_status_message(
                    f"{_mhz(best.tuned_hz)}: {best.summary()}", "info", 5000
                )
            elif self._ident_lost_s >= 0.5:
                # (What it missed may have held the signal: say so, rather
                # than that nothing is there.)
                self.panel.set_identify_status(
                    f"Nothing identifiable at {_mhz(best.tuned_hz)} in "
                    f"{WAIT_S:.0f} s ({stamp}), but the receiver dropped "
                    f"{self._ident_lost_s:.1f} s of the signal meanwhile (this "
                    "computer can't keep up). Try again; a lower sample rate "
                    "helps.",
                    "warning",
                )
            else:
                self.panel.set_identify_status(
                    f"Nothing identifiable at {_mhz(best.tuned_hz)} in "
                    f"{WAIT_S:.0f} s ({stamp}). Tune onto a signal and try "
                    "again.",
                    "warning",
                )

        def apply_identification(self) -> None:
            """Apply Mode: listen to the identified signal as suggested."""
            ident = self.panel.identification()
            if ident is None or not ident.mode:
                return
            win = self._win
            from .control_panel import bandwidth_options, closest_bandwidth

            width = closest_bandwidth(bandwidth_options(ident.mode), ident.filter_hz)
            # The mode change alone would say the mode's last width and
            # deviation first ("FM: 200 kHz, 75 kHz (broadcast) deviation"),
            # before these are set; this says what it set, as the band plan
            # and the presets do.
            win._quiet_mode_change = True
            try:
                win._set_demod_mode(ident.mode, ident.fm_deviation or None, width)
            finally:
                win._quiet_mode_change = False
            target = self._listen_frequency(ident)
            if abs(target - win._current_frequency()) >= 1.0:
                win.set_frequency(target)
            win._show_status_message(
                f"Listening to {_mhz(win._current_frequency())} in "
                + ", ".join(
                    part
                    for part in (
                        ident.mode,
                        f"{ident.fm_deviation} deviation" if ident.fm_deviation else "",
                        f"{width} filter",
                    )
                    if part
                ),
                "success",
                5000,
            )

        @staticmethod
        def _listen_frequency(ident: Identification) -> float:
            """Where to tune to hear it: the signal's centre (CW: its peak;
            SSB: the suppressed carrier below or above the voice)."""
            r = ident.result
            freq = ident.tuned_hz
            if r.kind in ("cw", "carrier"):
                peak = float(r.features.get("peak_offset_hz", r.center_offset_hz))
                return round((freq + peak) / 10.0) * 10.0
            if r.kind == "ssb":
                half = 0.5 * r.bandwidth_hz + 200.0
                shift = -half if ident.mode == "USB" else half
                return round((freq + r.center_offset_hz + shift) / 10.0) * 10.0
            if abs(r.center_offset_hz) >= max(0.1 * r.bandwidth_hz, 50.0):
                return round((freq + r.center_offset_hz) / 100.0) * 100.0
            return freq

        # -- Signals in view ----------------------------------------------------------------------

        def _signals_active(self) -> bool:
            if self.panel.showing_recording():
                # (The list is the recording's: no live markers, whose
                # numbers would not be its rows.)
                return time.monotonic() < self._listing_until
            return (
                self._markers
                or self.panel.page_visible("signals")
                or time.monotonic() < self._listing_until
            )

        def _reset_list(self) -> None:
            self._acc = None
            self._acc_frames = 0
            self._heard = []
            self._list_current = False
            self._last_mean = None

        @staticmethod
        def _in_view(freq_hz: float, lo_hz: float, rate: float) -> bool:
            """Whether a list of the span ``rate`` wide around ``lo_hz``
            covers ``freq_hz`` (its edges are left out of the list)."""
            return abs(freq_hz - lo_hz) <= 0.45 * rate

        def _signals_frame(self, power: np.ndarray, now: float) -> None:
            if not self._signals_active():
                # Frames go by unmeasured: the list is not kept up to date,
                # and its next mean starts afresh (not with frames from
                # before).
                self._list_current = False
                self._acc = None
                self._acc_frames = 0
                return
            if power is None or not len(power):
                return
            win = self._win
            key = (
                len(power),
                float(win._lo_frequency()),
                float(win._device_sample_rate()),
            )
            if key != self._acc_key:
                # A new span: measured afresh. What was heard where it still
                # shows stays listed meanwhile (the keys step on across the
                # retune a step to a signal near the edge makes).
                kept = [
                    (s, t)
                    for s, t in self._heard
                    if self._in_view(s.frequency_hz, key[1], key[2])
                ]
                current = self._list_current and self._acc_key is not None
                self._acc_key = key
                self._reset_list()
                if current and kept:
                    self._heard = kept
                    self._list_current = True
            lin = np.power(10.0, np.asarray(power, dtype=np.float64) / 10.0)
            if self._acc is None:
                self._acc = lin
            else:
                self._acc = self._acc + lin
            self._acc_frames += 1
            if self._acc_frames < LIST_FRAMES:
                return
            avg_db = 10.0 * np.log10(np.maximum(self._acc / self._acc_frames, 1e-30))
            self._acc = None
            self._acc_frames = 0
            found = find_signals(
                avg_db, key[1], key[2], float(self.panel.min_snr.value())
            )
            self._last_mean = (key, avg_db, now)
            self._list_current = True
            self._remember(found, now)
            self._show_list(now, now)

        def _show_list(self, now: float, heard_at: float) -> None:
            """The list on the Signals page, and the markers of the signals
            heard at ``heard_at`` (the last measurement) on the spectrum."""
            live = not self.panel.showing_recording()
            if live:
                self.panel.set_signals(
                    [s for s, _t in self._heard],
                    float(self._win._current_frequency()),
                    [now - t for _s, t in self._heard],
                )
            current = [s for s, t in self._heard if t >= heard_at - 0.01]
            self._apply_markers(current if self._markers and live else [])

        def _min_snr_changed(self) -> None:
            """List again at the new threshold, at once, from the mean the
            list was last measured from: what was heard only at the old one
            goes (a fresh, short mean listed pieces of stations)."""
            last = self._last_mean
            if last is None or last[0] != self._acc_key:
                self._reset_list()
                return
            key, mean_db, when = last
            found = find_signals(
                mean_db, key[1], key[2], float(self.panel.min_snr.value())
            )
            self._heard = [(s, when) for s in found]
            self._show_list(time.monotonic(), when)

        def _remember(self, found: List[SignalPeak], now: float) -> None:
            """Merge a fresh list into the signals heard lately (a signal
            not heard for :data:`FORGET_S` seconds is dropped)."""
            kept: List[Tuple[SignalPeak, float]] = []
            used = set()
            for sig in found:
                # Every older entry of this signal is replaced by it.
                for i, (old, _t) in enumerate(self._heard):
                    if abs(old.frequency_hz - sig.frequency_hz) <= max(
                        0.5 * max(old.width_hz, sig.width_hz), 3e3
                    ):
                        used.add(i)
                kept.append((sig, now))
            for i, (old, t) in enumerate(self._heard):
                if i not in used and now - t < FORGET_S:
                    kept.append((old, t))
            self._heard = sorted(kept, key=lambda item: item[0].frequency_hz)

        def set_markers(self, on: bool) -> None:
            self._markers = bool(on)
            if not on:
                self._apply_markers([])

        def _apply_markers(self, signals: List[SignalPeak]) -> None:
            set_markers = getattr(self._win._spectrum, "set_markers", None)
            if not callable(set_markers):
                return
            index = {id(s): n + 1 for n, (s, _t) in enumerate(self._heard)}
            set_markers([(s.frequency_hz, str(index.get(id(s), ""))) for s in signals])

        def step(self, direction: str) -> bool:
            """Tune to the strongest signal in view ("strongest") or the next
            one above ("up") or below ("down") the tuned frequency."""
            win = self._win
            if not win._is_running:
                win._show_status_message(
                    "Press Start (Space) first: the signals in view are measured "
                    "while receiving.",
                    "info",
                    4000,
                )
                return False
            # (Pressing the keys keeps the list measured off the page.)
            self._ensure_listing()
            key = self._acc_key
            lo, rate = float(win._lo_frequency()), float(win._device_sample_rate())
            # What was heard in the span shown now (a step to a signal near
            # the edge retunes: the signals heard that the span still shows
            # are stepped to while it is measured).
            signals = [
                s for s, _t in self._heard if self._in_view(s.frequency_hz, lo, rate)
            ]
            if (
                key is None
                or not self._list_current
                or ((key[1], key[2]) != (lo, rate) and not signals)
            ):
                # Not measured (the list wasn't kept up to date off screen,
                # or it was of another span): measure now.
                win._show_status_message(
                    "Looking for signals… press it again in a moment.", None, 3000
                )
                return False
            tuned = float(win._current_frequency())
            target: Optional[SignalPeak] = None
            if direction == "strongest":
                if not signals:
                    win._show_status_message("No signals in view.", None, 3000)
                    return False
                target = max(signals, key=lambda s: s.peak_dbfs)
            elif direction == "up":
                ups = [
                    s
                    for s in signals
                    if s.frequency_hz > tuned + max(0.5 * s.width_hz, 1e3)
                ]
                target = min(ups, key=lambda s: s.frequency_hz) if ups else None
            else:
                downs = [
                    s
                    for s in signals
                    if s.frequency_hz < tuned - max(0.5 * s.width_hz, 1e3)
                ]
                target = max(downs, key=lambda s: s.frequency_hz) if downs else None
            if target is None:
                where = "above" if direction == "up" else "below"
                win._show_status_message(
                    f"No signal {where} {_mhz(tuned)} in view.", None, 3000
                )
                return False
            return bool(self._tune(tune_frequency(target)))

        def _ensure_listing(self) -> None:
            """List the signals for a while even off the Signals page (for
            Home / Page Up / Page Down: 30 s from the last press)."""
            self._listing_until = time.monotonic() + 30.0

        def _tune(self, freq: float) -> bool:
            return bool(self._win.set_frequency(float(freq)))

        def _page_changed(self) -> None:
            self._mark_burst(self.panel.selected_burst())
            if self.panel.page_visible("bursts") and not self._watch_always:
                self._watcher.reset()  # its floor is from when it was last shown

        # -- Scope ------------------------------------------------------------------------------

        def _scope_frame(self, channel: np.ndarray, chain: Any) -> None:
            panel = self.panel
            if not panel.page_visible("scope") or panel.scope.frozen:
                return
            if channel is None or not len(channel):
                return
            rate = float(getattr(chain, "channel_rate", 0.0) or 0.0)
            panel.scope.set_samples(channel, rate)
            info = (
                f"The tuned channel: {chain.mode}, "
                f"{_width_text(getattr(chain, 'bandwidth', 0.0))} filter, "
                f"{rate / 1e3:.0f} kS/s"
            )
            evm = panel.scope.evm_percent
            if evm is not None:
                info += f" · EVM {evm:.1f} %"
            panel.set_scope_info(info)

        # -- Bursts -------------------------------------------------------------------------------

        def _set_watch_always(self, on: bool) -> None:
            self._watch_always = bool(on)

        def _bursts_frame(self, samples: np.ndarray, rate: float) -> None:
            if self._burst_jobs:
                self._collect_bursts()
            if self.panel.page_visible("bursts") != self._bursts_shown:
                self._mark_burst(self.panel.selected_burst())
            if not (self._watch_always or self.panel.page_visible("bursts")):
                if self._watcher.detector.active:
                    self._queue_bursts(self._watcher.flush())
                return
            win = self._win
            if getattr(getattr(win, "_pacer", None), "overloaded", False):
                # One block a frame, a few per cent of the signal: bursts
                # would be cut up or missed, and their times wrong. Paused
                # until the display loop keeps up, then watched afresh.
                if not self._bursts_paused:
                    self._bursts_paused = True
                    self._queue_bursts(self._watcher.flush())
                    self._watcher.reset()
                    self.panel.set_bursts_paused(
                        "Paused: this computer can't keep up right now\nBursts "
                        "are logged again as soon as it catches up."
                    )
                return
            if self._bursts_paused:
                self._bursts_paused = False
                self.panel.set_bursts_paused("")
            reports = self._watcher.feed(samples, float(win._lo_frequency()), rate)
            clock = getattr(win, "_plot_clock", None)  # (this frame's row)
            if isinstance(clock, float):
                self._frame_times.append((self._watcher.time_s, clock))
            self._queue_bursts(reports)

        def _queue_bursts(self, reports: List[BurstReport]) -> None:
            """Take the finished bursts apart on the bursts worker (it takes
            50-200 ms each, too long for the display loop); they are listed
            in order as their analyses come back."""
            if self._shut:
                return  # (the worker is closed: the window is closing)
            frames: Optional[Tuple[List[float], List[float]]] = None
            for report in reports:
                burst = report.burst
                if burst.duration_s < 0.02 and burst.snr_db < 15:
                    continue  # a flicker
                # The list's Time is when it started, not when it was done.
                ago = max(0.0, self._watcher.time_s - burst.start_s)
                when = time.time() - ago
                if frames is None:
                    frames = (
                        [watched for watched, _row in self._frame_times],
                        [row for _watched, row in self._frame_times],
                    )
                self._remember_span(report, *frames)
                job: Optional[Future] = None
                if report.samples is not None and self._may_analyze():
                    try:
                        job = self._burst_pool.submit(_timed_analysis, report)
                    except RuntimeError:  # pragma: no cover - (shut down)
                        job = None
                if job is None:
                    # A busy band, or a busy computer: listed now, not taken
                    # apart.
                    job = Future()
                    job.set_result(0.0)
                self._burst_jobs.append((job, report, when))
            if self._burst_jobs:
                self._collect_bursts()
                if self._burst_jobs and not self._burst_timer.isActive():
                    self._burst_timer.start()

        def _remember_span(
            self, report: BurstReport, ends: List[float], rows: List[float]
        ) -> None:
            """Where ``report``'s burst is on the waterfall: the waterfall
            times (``rows``) of the frames its start and end were in, from
            the watcher's time at each frame's end (``ends``), and its
            frequencies."""
            if not ends:
                return
            last = len(ends) - 1
            burst = report.burst
            # The frames with its first and its last samples: the first that
            # ends after its start, and the first that ends at its end.
            first = min(bisect.bisect_right(ends, float(burst.start_s) + 1e-6), last)
            final = min(bisect.bisect_left(ends, float(burst.end_s) - 1e-6), last)
            half = 0.5 * float(burst.width_hz)
            self._burst_spans[id(report)] = (
                rows[first],
                rows[max(first, final)],
                float(burst.center_hz) - half,
                float(burst.center_hz) + half,
            )
            if len(self._burst_spans) > 2 * MAX_LISTED_BURSTS:
                # (Those the list no longer holds; not those waiting for it.)
                keep = {id(r) for _stamp, r in self.panel.bursts()}
                keep.update(id(r) for _job, r, _when in self._burst_jobs)
                keep.add(id(report))
                for key in [k for k in self._burst_spans if k not in keep]:
                    del self._burst_spans[key]

        def _mark_burst(self, report: Any = None) -> None:
            """Outline the selected burst on the waterfall, while the Bursts
            page is on screen."""
            self._bursts_shown = self.panel.page_visible("bursts")
            span = None if report is None else self._burst_spans.get(id(report))
            waterfall = self._win._waterfall
            if span is None or not self._bursts_shown:
                waterfall.clear_burst_mark()
            else:
                waterfall.mark_burst(*span)

        def _may_analyze(self) -> bool:
            """Whether the next burst may be taken apart (see
            :data:`ANALYSIS_SHARE`)."""
            if len(self._burst_jobs) >= MAX_PENDING_BURSTS:
                return False
            pacer = getattr(self._win, "_pacer", None)
            if getattr(pacer, "overloaded", False):
                return False  # the display loop is behind: no more work
            now = self._watcher.time_s
            cost = self._analysis_cost
            while cost and now - cost[0][0] > _ANALYSIS_WINDOW_S:
                cost.popleft()
            spent = sum(seconds for _t, seconds in cost)
            return spent < ANALYSIS_SHARE * _ANALYSIS_WINDOW_S

        def _collect_bursts(self) -> None:
            """List the bursts whose analysis came back, oldest first."""
            while self._burst_jobs and self._burst_jobs[0][0].done():
                job, report, when = self._burst_jobs.popleft()
                try:
                    seconds = float(job.result() or 0.0)
                except Exception as exc:  # pragma: no cover - a bonus only
                    logger.warning("Taking a burst apart failed: %s", exc)
                    seconds = 0.0
                if seconds > 0.0:
                    self._analysis_cost.append((self._watcher.time_s, seconds))
                # The list keeps what was learned, not the burst's samples
                # (up to a few MB each).
                report.samples = None
                self.panel.add_burst(report, when)
            if not self._burst_jobs:
                self._burst_timer.stop()

        def export_bursts(self) -> None:
            from . import file_dialogs

            win = self._win
            if not self.panel.bursts():
                win._show_status_message("No bursts to export yet.", None, 3000)
                return
            path, _f = file_dialogs.save_file(
                win,
                win._settings,
                "channels",
                "Export Bursts",
                time.strftime("bursts_%Y%m%d_%H%M%S.csv"),
                "CSV file (*.csv)",
            )
            if not path:
                return
            try:
                count = self.panel.export_bursts(path)
            except OSError as e:
                win._show_status_error(f"Could not save {path}: {e}")
                return
            win._show_status_message(f"Saved {count} bursts to {path}", "success", 5000)

        # -- Track Signal (AFC) ----------------------------------------------------------------------

        def _tracking_problem(self) -> Optional[str]:
            """Why tracking can't follow in the current mode (None: it can)."""
            panel = self._win._control_panel
            mode = panel._demod_combo.currentText()
            if mode not in TRACK_METHODS:
                return (
                    "Track Signal follows FM, AM and CW signals (SSB and I/Q "
                    "have no carrier to follow)."
                )
            if mode == "FM" and panel.get_fm_deviation() >= _BROADCAST_DEVIATION_HZ:
                return (
                    "Track Signal follows narrow FM, AM and CW (broadcast FM "
                    "stations don't drift)."
                )
            return None

        def set_tracking(self, on: bool) -> bool:
            """Radio > Track Signal (AFC)."""
            win = self._win
            if on:
                problem = self._tracking_problem()
                if problem:
                    action = getattr(win, "_track_action", None)
                    if action is not None and action.isChecked():
                        action.blockSignals(True)
                        action.setChecked(False)
                        action.blockSignals(False)
                    # (A mode it can't follow: a notice, not a problem for
                    # Tools > Error History.)
                    win._show_status_message(problem, "warning", 6000, log=False)
                    self._tracking = False
                    self._set_badge(None)
                    return False
            self._tracking = bool(on)
            self._tracker = None
            self._tracker_key = None
            if on:
                if win._is_running:
                    self._update_badge(0.0, False, False, False)
                else:
                    self._set_paused_badge()
                win._show_status_message(
                    "Track Signal: the tuned frequency follows the signal as it "
                    "drifts",
                    "info",
                    4000,
                )
            else:
                self._set_badge(None)
            return self._tracking

        @property
        def tracking(self) -> bool:
            return self._tracking

        def _stop_tracking(self, message: str, log: bool = True) -> None:
            """Track Signal turns itself off, its menu item and status-bar
            badge too, saying why (``log=False``: a notice, not a problem
            for Tools > Error History)."""
            self.set_tracking(False)
            action = getattr(self._win, "_track_action", None)
            if action is not None and action.isChecked():
                action.blockSignals(True)
                action.setChecked(False)
                action.blockSignals(False)
            self._win._show_status_message(message, "warning", 6000, log=log)

        def _set_paused_badge(self) -> None:
            self._set_badge(
                "AFC paused", "muted", "Track Signal resumes when receiving does"
            )

        def _afc_frame(self, samples: np.ndarray, dt: float, chain: Any) -> None:
            if not self._tracking:
                return
            win = self._win
            scanner = getattr(win, "_scanner_panel", None)
            if scanner is not None and getattr(scanner.engine, "active", False):
                self._set_badge("AFC paused", "muted", "Paused while a scan runs")
                self._tracker = None
                return
            panel = win._control_panel
            mode = panel._demod_combo.currentText()
            key = (
                mode,
                float(win._channel_bandwidth()),
                float(win._device_sample_rate()),
            )
            problem = self._tracking_problem()
            if problem is not None:
                self._stop_tracking("Track Signal turned off: " + problem, log=False)
                return
            if self._tracker is None or key != self._tracker_key:
                self._tracker = SignalTracker(
                    key[2], win._vfo_offset_hz(), mode, key[1]
                )
                self._tracker_key = key
            status = self._tracker.process(samples, dt)
            if status.nudge_hz:
                self._move_vfo(status.nudge_hz)
                if not self._tracking:
                    return  # it stopped there (the edge of a recording)
            self._update_badge(
                status.correction_hz, status.locked, status.lost, status.signal_present
            )

        def _move_vfo(self, delta_hz: float) -> None:
            """Move the tuned frequency by ``delta_hz`` for Track Signal: the
            same signal, so nothing that belongs to a new station is reset
            (S-meter peak, RDS, decoder status, tuning history). The LO is
            retuned only when the channel would leave the span."""
            win = self._win
            freq = float(win._current_frequency()) + float(delta_hz)
            lock = getattr(win, "_lo_lock", None)
            if lock is not None:
                center, span, _why = lock
                if not win._channel_fits(freq - center, rate=span):
                    self._stop_tracking(
                        "Track Signal stopped: the signal drifted out of the "
                        "recording."
                    )
                    return
            elif not win._channel_fits(freq - win._lo_frequency()):
                self._moving = True
                try:
                    tuned = win._tune_to(freq)
                finally:
                    self._moving = False
                if not tuned:
                    # (Refused, e.g. while an I/Q recording keeps its center:
                    # asked again every frame, the refusal repeated.)
                    recording = win._recording and not win._capture_pending
                    self._stop_tracking(
                        "Track Signal stopped: the signal drifted to the edge "
                        "of the I/Q recording's span."
                        if recording
                        else "Track Signal stopped: the receiver didn't retune "
                        "to follow the signal."
                    )
                    return
                if self._tracker is not None:
                    self._tracker.lo_moved(win._vfo_offset_hz())
                return
            self._moving = True
            try:
                win._tuned_hz = freq
                win._control_panel.set_frequency(freq)  # no signal
                # The live decoder follows the channel without starting over.
                win._decoder_tuned_hz = freq
                # Everything that shows the tuned frequency follows (FREQ,
                # the plots' channel, the title, the band plan, the Presets
                # group, the tuner, where Send ID and QRP transmit).
                win._on_frequency_changed(freq, same_signal=True)
            finally:
                self._moving = False

        def _update_badge(
            self, correction: float, locked: bool, lost: bool, present: bool
        ) -> None:
            if lost:
                self._set_badge(
                    f"AFC {_format_offset(correction)} · lost",
                    "warning",
                    "Track Signal: no signal to follow for a few seconds; it "
                    "holds the frequency and picks the signal up when it is back",
                )
            else:
                self._set_badge(
                    f"AFC {_format_offset(correction)}",
                    "success" if locked else "info",
                    "Track Signal (Radio menu): how far the tuned frequency has "
                    "been moved to follow the signal"
                    + (" (locked on)" if locked else ""),
                )

        def _set_badge(
            self, text: Optional[str], tone: Optional[str] = None, tip: str = ""
        ) -> None:
            if text is None:
                if self._afc_badge is not None:
                    self._afc_badge.setVisible(False)
                return
            if self._afc_badge is None:
                badge = QLabel("")
                set_role(badge, "badge", "info")
                badge.setAccessibleName("Track Signal (AFC)")
                self._win.statusBar().addPermanentWidget(badge)
                self._afc_badge = badge
            badge = self._afc_badge
            if badge.text() != text:
                badge.setText(text)
            set_tone(badge, tone)
            badge.setToolTip(tip)
            badge.setVisible(True)

        def badge_text(self) -> str:
            badge = self._afc_badge
            return badge.text() if badge is not None and badge.isVisible() else ""

        # -- Find Signals in Recording -----------------------------------------------------------

        def find_in_current_recording(self) -> bool:
            """The playback strip's Find Signals: the file playing."""
            playback = self._win._playback
            device = playback.device
            if device is None:
                return False
            return self.find_in_recording(device.options.path, device.options)

        def find_in_recording(self, path: str = "", options: Any = None) -> bool:
            """File > Find Signals in Recording: list the signals of a whole
            I/Q file (frequency, level, SNR, when heard), read in the
            background with progress and Cancel."""
            from .file_jobs import _chunks, run_job

            win = self._win
            path = path or win._choose_recording("Find Signals in Recording")
            if not path:
                return False
            options = win._recording_options(path, "find", options)
            if options is None:
                return False
            rate, center = float(options.sample_rate), float(options.center_hz)

            def work(report: Any) -> Tuple[List[Any], float]:
                survey = SignalSurvey(rate, center)
                for chunk in _chunks(options, report):
                    survey.add(chunk)
                return survey.finish(), survey.position / rate

            result = run_job(
                win,
                "Find Signals",
                f"Looking for signals in {options.name}…",
                work,
            )
            if result.error is not None:
                logger.warning(f"Finding signals in {path} failed: {result.error}")
                win._show_status_error(
                    f"Could not read {options.name}: {result.error}", log=False
                )
                return False
            if result.cancelled:
                win._show_status_message("Stopped looking for signals.", None, 4000)
                return False
            signals, seconds = result.value
            self._survey_options = options
            self.panel.set_recording_signals(options.name, signals, seconds)
            self._apply_markers([])
            win._show_panel("signal")
            self.panel.show_page("signals")
            win._show_status_message(
                f"Found {len(signals)} signal{'s' if len(signals) != 1 else ''} in "
                f"{options.name}. Double-click one to play it there.",
                "success" if signals else "info",
                6000,
            )
            return True

        def play_recording_at(self, freq_hz: float, seconds: float) -> bool:
            """Play the surveyed recording from just before ``seconds``,
            tuned to ``freq_hz`` (in its band's usual mode)."""
            win = self._win
            options = self._survey_options
            if options is None:
                return False
            playback = win._playback
            device = playback.device
            if device is None or device.options.path != options.path:
                if not win._open_recording(options.path, options):
                    return False
            playback.seek(max(0.0, float(seconds) - 0.5))
            # The band's usual mode first (a channel as wide as the one
            # listened to before may not fit the recording there), narrower
            # still when it is near the recording's edge.
            from ..core.band_plan import listening_for

            listen = listening_for(freq_hz)
            if listen is not None and listen[0]:
                mode, deviation, bandwidth = listen
                win._set_demod_mode(mode, deviation or None, bandwidth or None)
            if not win._channel_fits(
                float(freq_hz) - float(options.center_hz),
                rate=float(options.sample_rate),
            ):
                self._narrow_to_fit(float(freq_hz) - float(options.center_hz), options)
            if not win.set_frequency(float(freq_hz)):
                return False  # the status bar says why
            if not win._is_running:
                win._start_acquisition()
            return True

        def _narrow_to_fit(self, offset_hz: float, options: Any) -> None:
            """Pick the widest Bandwidth of the mode that fits the recording
            ``offset_hz`` from its centre."""
            from .control_panel import bandwidth_options

            win = self._win
            panel = win._control_panel
            mode = panel._demod_combo.currentText()
            for text in sorted(bandwidth_options(mode), key=lambda t: -_parse_width(t)):
                win._set_demod_mode(mode, None, text)
                if win._channel_fits(offset_hz, rate=float(options.sample_rate)):
                    return


def _timed_analysis(report: BurstReport) -> float:
    """Take a burst apart (on the bursts worker); returns the processor
    time it took (s): what it took from the display loop, however busy
    the computer is."""
    start = time.thread_time()
    analyze_report(report)
    return max(1e-6, time.thread_time() - start)


def _mhz(freq_hz: float) -> str:
    """The frequency as the window names it: ``"146.5125 MHz"``, below 2
    MHz in kHz (``"880 kHz"``, as FREQ shows it)."""
    from .signal_panel import format_mhz

    return format_mhz(freq_hz)


def _parse_width(text: str) -> float:
    """``"12.5 kHz"`` -> 12500.0 (0 when it isn't a width)."""
    try:
        value, unit = str(text).split()
        return float(value) * {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
    except (ValueError, KeyError):
        return 0.0


def _width_text(hz: float) -> str:
    hz = float(hz)
    if hz >= 1e6:
        return f"{hz / 1e6:.1f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:g} kHz"
    return f"{hz:.0f} Hz"


__all__ = ["SignalTools", "LIST_FRAMES", "LIST_EVERY_S", "FORGET_S"]
