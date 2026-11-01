"""
The main window as the Scanner tab's host.

:class:`WindowScannerHost` is what the scanner (``scanner_panel`` and
``scan_engine``) calls to tune the receiver, hear whether the squelch is
open, start the receiver, add bookmarks and so on. It keeps the window's
side of the scanner in one place, so ``main_window.py`` only needs a few
hooks: the panel, ``_scan_frame`` in the display loop and the mute in
``_route_audio``.

:class:`SecondReceiverPartner` is the Second Receiver window (Device >
Second Receiver...) sweeping part of the range with it: the Scanner tab's
"Use both receivers".
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..devices.base import SDRDevice
from ..dsp.dcs import dcs_label, format_dcs, parse_dcs
from .bookmark_editor import bookmark_dcs
from .bookmarks_panel import default_bookmark_name, listening
from .error_log_dialog import install_history_handler
from .scan_engine import Listening, ScanPartner, bookmark_scan_list
from .scanner_panel import ScanHit, ScannerHost

logger = logging.getLogger(__name__)

#: Modes with audio, where the squelch decides what is heard.
_AUDIBLE_MODES = ("AM", "FM", "USB", "LSB", "CW")
#: Extra time a hardware receiver gets after a retune: an RTL-SDR's USB
#: transfer in flight (0.11 s at 2.4 MS/s) still holds the old frequency.
_HARDWARE_SETTLE_S = 0.12
#: At most this many queued transfers are dropped after a retune (the
#: drivers queue up to 100).
_MAX_STALE_TRANSFERS = 256


def _parse_hz(text: str) -> Optional[float]:
    value, _sep, unit = str(text).strip().partition(" ")
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6}.get(unit.strip().lower())
    try:
        return float(value) * scale if scale else None
    except ValueError:
        return None


class _HeldBack(logging.Filter):
    """Keeps from a handler what this thread logs, in ``records``."""

    def __init__(self) -> None:
        super().__init__()
        self._thread = threading.get_ident()
        self.records: List[logging.LogRecord] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread != self._thread:
            return True  # (another thread's, while the step tunes)
        self.records.append(record)
        return False


def _text(record: logging.LogRecord) -> str:
    try:
        return record.getMessage()
    except Exception:  # (a malformed record: its bare message)
        return str(record.msg)


def _mhz(freq_hz: float) -> str:
    return f"{float(freq_hz) / 1e6:.3f} MHz"


class SecondReceiverPartner(ScanPartner):
    """The Second Receiver window sweeping with the main window (the
    Scanner tab's "Use both receivers"): the scan tunes it step by step and
    takes every block it reads (:meth:`attach`), its speaker is muted
    meanwhile, and it goes back where it was when the scan ends
    (:meth:`detach`). It leaves the scan, through ``left``, when it can't
    go on: stopped, recording, its sample rate or tuning changed by hand."""

    def __init__(self, receiver: Any) -> None:
        self.receiver = receiver
        self._rate = float(receiver.sample_rate())
        # (tuned, LO) before the scan, and what the scan tuned last.
        self._saved: Optional[Tuple[float, float]] = None
        self._expected: Optional[Tuple[float, float]] = None
        self._feed: Optional[Callable[[np.ndarray], None]] = None
        self._left: Optional[Callable[[str], None]] = None

    # ---- ScanPartner ------------------------------------------------------ #
    def scan_tune_lo(self, center_hz: float, vfo_hz: Optional[float] = None) -> bool:
        rx = self.receiver
        if not rx.tune(center_hz, recenter=True):
            return False
        if vfo_hz is not None and abs(vfo_hz - center_hz) >= 0.5:
            rx.tune(vfo_hz)  # (inside the span: only the VFO moves)
        self._expected = (rx.tuned_frequency(), rx.lo_frequency())
        return True

    def scan_lo_hz(self) -> float:
        return float(self.receiver.lo_frequency())

    def scan_sample_rate(self) -> float:
        return self._rate

    def scan_settle_s(self) -> float:
        hardware = isinstance(self.receiver.device, SDRDevice)
        return _HARDWARE_SETTLE_S if hardware else 0.0

    def scan_device_range(self) -> Optional[Tuple[float, float]]:
        low, high = self.receiver.tuning_range()
        return float(low), float(high)

    def scan_set_muted(self, muted: bool) -> None:
        self.receiver.set_scan_muted(muted)

    # ---- the scan's side -------------------------------------------------- #
    def problem(self) -> str:
        """Why it can't go on sweeping ("" when it can)."""
        rx = self.receiver
        if not rx.running:
            return "it was stopped"
        if rx.is_recording():
            return "it is recording, which keeps it on one frequency"
        if abs(float(rx.sample_rate()) - self._rate) > 0.5:
            return "its sample rate changed"
        if not self._where_left():
            return f"you tuned it to {_mhz(rx.tuned_frequency())}"
        return ""

    def _where_left(self) -> bool:
        """Whether it is still where the scan tuned it last (nobody, not
        the user nor Follow transmissions, has tuned it since)."""
        expected = self._expected
        if expected is None:
            return True
        rx = self.receiver
        tuned, lo = rx.tuned_frequency(), rx.lo_frequency()
        return abs(tuned - expected[0]) <= 1.0 and abs(lo - expected[1]) <= 1.0

    def attach(
        self, feed: Callable[[np.ndarray], None], left: Callable[[str], None]
    ) -> None:
        """Sweep with the scan: ``feed`` takes the blocks this receiver
        reads; ``left(why)`` when it can't go on (it is then detached)."""
        rx = self.receiver
        self._saved = (rx.tuned_frequency(), rx.lo_frequency())
        self._expected = self._saved
        self._feed, self._left = feed, left
        rx.set_scan_feed(self._block)

    def _block(self, samples: np.ndarray) -> None:
        why = self.problem()
        if why:
            left = self._left
            self.detach(restore=False)
            if left is not None:
                left(why)
            return
        feed = self._feed
        if feed is not None:
            feed(samples)

    def detach(self, restore: bool = True) -> None:
        """Out of the scan: unmuted, no longer feeding it, and (``restore``)
        back on the frequency and span it had before, unless someone tuned
        it since the scan did (then it stays there)."""
        rx = self.receiver
        saved, self._saved = self._saved, None
        self._feed = self._left = None
        try:
            restore = restore and self._where_left()
            self._expected = None
            rx.set_scan_feed(None)
            rx.set_scan_muted(False)
            if restore and saved is not None and rx.running:
                tuned, lo = saved
                # The span first (the plots where they were), then the VFO.
                if rx.tune(lo, recenter=True) and abs(tuned - lo) >= 0.5:
                    rx.tune(tuned)
        except RuntimeError as exc:  # pragma: no cover - its window deleted
            logger.debug("The second receiver is gone: %s", exc)


class WindowScannerHost(ScannerHost):
    """Adapts the main window (``SDRMainWindow``) to the scanner."""

    def __init__(self, window: Any):
        self._win = window
        # Where the receiver was before the scan: (frequency, LO, mode,
        # deviation, bandwidth, CTCSS tone squelch, DCS code squelch).
        self._saved: Optional[Tuple[float, float, str, str, str, float, str]] = None
        self._lo_moved = False
        #: The frequency the scan tuned last (None outside a scan): any
        #: other means the user tuned elsewhere.
        self.expected_hz: Optional[float] = None
        # What the driver said when the receiver last refused a step.
        self._refusal = ""

    # ---- tuning ----------------------------------------------------------- #
    def _quietly(self, tune) -> bool:
        """Tune without the status-bar messages and saved settings that a
        mode change brings (a memory scan changes modes all the time), and
        without the scan's filters becoming the widths each mode remembers."""
        win = self._win
        panel = win._control_panel
        restoring, remembering = win._restoring, panel.remember_filters
        win._restoring, panel.remember_filters = True, False
        try:
            return bool(tune())
        finally:
            win._restoring, panel.remember_filters = restoring, remembering

    def _tune_step(self, tune) -> bool:
        """Tune a step of the scan (quietly, see :meth:`_quietly`). What a
        refused step logs (the window's "didn't tune", the driver's error)
        stays out of Tools > Error History: the scan tells the steps it
        skipped in one line a pass (see ``ScanEngine._tell_refused``), not
        one or two a step. A step that tunes, or breaks, logs as usual."""
        history = install_history_handler()  # (installed at startup)
        held = _HeldBack()
        history.addFilter(held)
        ok: Optional[bool] = None
        try:
            ok = self._quietly(tune)
        finally:
            history.removeFilter(held)
            if ok is False:
                errors = [_text(r) for r in held.records if r.levelno >= logging.ERROR]
                self._refusal = errors[0] if errors else ""
            else:
                for record in held.records:
                    history.handle(record)
        return bool(ok)

    def scan_refusal_reason(self) -> str:
        reason, self._refusal = self._refusal, ""
        return reason

    def _after_tune(self) -> None:
        win = self._win
        # Scan steps don't join Tune Back / Recent Frequencies (where the
        # scan ends does, see scan_end).
        win._history_timer.stop()
        self._fresh_channel()
        self.expected_hz = win._current_frequency()
        if self._lo_moved:
            self._drop_stale_transfers()

    def _fresh_channel(self) -> None:
        """A new channel: its squelch decides afresh. Neither the last
        channel's hang time nor its samples still in the channel filter
        count as this one's signal (a station just left, by Skip or for its
        tone, opened the squelch on a quiet channel after it)."""
        win = self._win
        win._squelch.reset()
        win._level_average.reset()
        chain = getattr(win, "_rx_chain", None)
        if chain is not None:
            chain.reset_channel()

    def scan_settled(self) -> None:
        # The blocks dropped while settling may hold the last channel (a
        # hardware receiver's transfer in flight): what they opened, or
        # the tone they showed, doesn't count.
        self._fresh_channel()
        self._win._tone_squelch.reset()

    def _drop_stale_transfers(self) -> None:
        """After a retune, the transfers a hardware receiver has queued hold
        the old frequency: drop them, so the scan measures (and listens to)
        the new one. While the display loop keeps up there are hardly any;
        when it falls behind (a HackRF at a high rate) the queue is full of
        them, and a sweep listed the old step's signals at the new step's
        frequencies. They count as received (see ``_realtime``)."""
        win = self._win
        dev = win._device
        if not isinstance(dev, SDRDevice):
            return
        for _ in range(_MAX_STALE_TRANSFERS):
            try:
                block = dev.read_samples(16384, timeout=0.0)
            except Exception as exc:  # pragma: no cover - driver specific
                logger.debug("Could not drop queued samples: %s", exc)
                return
            if block is None or len(block) == 0:
                return
            win._note_block(len(block))

    def scan_tune_lo(self, center_hz: float, vfo_hz: Optional[float] = None) -> bool:
        win = self._win
        before = win._lo_frequency()
        win._scan_checks_tone = False
        # The step is measured around center_hz: with Radio > Keep Tuned
        # Frequency Centered, tuning the VFO would retune the LO onto it,
        # and so would a channel too wide to fit beside the LO (a 2 MHz
        # I/Q filter), so FREQ shows the LO instead.
        keep_centered = bool(getattr(win, "_keep_centered", lambda: False)())

        def tune() -> bool:
            if not win._tune_to(center_hz, recenter=True):
                return False
            if vfo_hz is not None and abs(vfo_hz - center_hz) >= 0.5:
                offset = vfo_hz - win._lo_frequency()
                if not keep_centered and win._channel_fits(offset):
                    win._tune_to(vfo_hz)  # inside the span: only the VFO moves
            return True

        ok = self._tune_step(tune)
        self._lo_moved = abs(win._lo_frequency() - before) >= 0.5
        if ok:
            self._after_tune()
        return ok

    def scan_tune_channel(
        self,
        freq_hz: float,
        listening: Listening,
        tone_hz: float = 0.0,
        dcs: str = "",
    ) -> bool:
        win = self._win
        before = win._lo_frequency()
        if not listening[0] and self._saved is not None:
            # "The receiver's mode" (an old bookmark, a signal of a range
            # without a band): the mode, filter and tone squelch the scan
            # began with, not whatever the channel visited last left.
            _freq, _lo, mode, deviation, width, tone_hz, dcs = self._saved
            listening = (mode, deviation, width)
        # A bookmark with a tone squelch (a CTCSS tone or a DCS code) is
        # judged by it, so the window checks it even with the speaker off
        # (see _route_audio).
        win._scan_checks_tone = bool(tone_hz and tone_hz > 0) or (
            parse_dcs(dcs) is not None
        )

        def tune() -> bool:
            if not win._tune_to(freq_hz):
                return False
            self._apply_listening(listening, tone_hz, dcs)
            return True

        ok = self._tune_step(tune)
        self._lo_moved = abs(win._lo_frequency() - before) >= 0.5
        if ok:
            self._after_tune()
        return ok

    def _apply_listening(
        self, listening: Listening, tone_hz: float, dcs: str = ""
    ) -> None:
        demod, deviation, width = listening
        if not demod:
            return
        win = self._win
        width = width or win._bandwidth_for_mode(demod, deviation) or ""
        win._set_demod_mode(demod, deviation or None, width or None)
        if demod == "FM":
            self._set_squelch(float(tone_hz or 0.0), dcs)

    def _set_squelch(self, tone_hz: float, dcs: str) -> None:
        """The tone squelch: a DCS code ("D023N"), else a CTCSS tone (0 for
        none)."""
        panel = self._win._control_panel
        code = parse_dcs(dcs) if dcs else None
        if code is not None:
            panel.set_dcs_squelch(*code)
        else:
            panel.set_tone_squelch_hz(tone_hz)

    def scan_lo_moved(self) -> bool:
        return self._lo_moved

    def scan_lo_hz(self) -> float:
        return float(self._win._lo_frequency())

    # ---- what is heard ---------------------------------------------------- #
    def _squelch_used(self) -> bool:
        win = self._win
        mode = win._control_panel.get_demod_mode()
        return mode in _AUDIBLE_MODES and bool(win._squelch.enabled)

    def _level_above_threshold(self) -> bool:
        """Without a squelch to go by (squelch off, or I/Q mode): the
        channel's level against the squelch's threshold."""
        win = self._win
        level = win._last_peak_db
        if level is None or not math.isfinite(level):
            return False
        noise = win._channel_noise_db(win._receiver_chain())
        return level >= win._squelch.threshold_db(noise)

    def scan_signal_present(self) -> bool:
        win = self._win
        if self._squelch_used():
            return bool(win._squelch.is_open) and not win._tone_blocked
        # (The tone squelch still decides with the squelch off: _route_audio
        # checks the tone of a channel scanned with one.)
        return self._level_above_threshold() and not win._tone_blocked

    def scan_squelch_open(self) -> bool:
        if self._squelch_used():
            return bool(self._win._squelch.is_open)
        return self._level_above_threshold()

    def scan_set_muted(self, muted: bool) -> None:
        self._win._scan_muted = bool(muted)

    # ---- the receiver ----------------------------------------------------- #
    def scan_blocker(self) -> Tuple[str, str]:
        win = self._win
        if win._device is None:
            return "device", (
                "No receiver: use the demo device or connect an SDR to scan."
            )
        lock = getattr(win, "_lo_lock", None)
        if lock is not None:
            return "locked", (
                f"Scanning needs the receiver, but {lock[2]} is playing. Stop "
                "it to scan."
            )
        if win._recording:
            return "recording", (
                "Stop recording to scan: a scan retunes the receiver, so the "
                "recording would mix frequencies."
            )
        if win._tx_busy():
            return "transmitting", (
                "Scanning needs the receiver, which is off while transmitting: "
                "scan when the transmission ends."
            )
        return "", ""

    def scan_receiving(self) -> bool:
        win = self._win
        return bool(win._is_running and win._device is not None)

    def scan_start_receiver(self) -> bool:
        win = self._win
        win._start_acquisition()
        return bool(win._is_running)

    def scan_sample_rate(self) -> float:
        return float(self._win._device_sample_rate())

    def scan_settle_s(self) -> float:
        return _HARDWARE_SETTLE_S if isinstance(self._win._device, SDRDevice) else 0.0

    def scan_noise_floor_db(self) -> Optional[float]:
        win = self._win
        if not (win._is_running and win._device is not None):
            return None
        return win._noise_floor.bin_db

    def scan_device_range(self) -> Optional[Tuple[float, float]]:
        return self._win._device_tuning_range()

    def scan_partner(self) -> Tuple[Optional[SecondReceiverPartner], str]:
        """The Second Receiver window to sweep with (the Scanner tab's "Use
        both receivers"), or None and why not; ("" when no such window is
        open: nothing to offer)."""
        host = getattr(self._win, "_second_rx", None)
        receiver = getattr(host, "receiver", None)
        if receiver is None or not receiver.isVisible():
            return None, ""
        if not receiver.running:
            return None, (
                "Start the second receiver (Device > Second Receiver..., Start) "
                "to sweep with it too."
            )
        if receiver.is_recording():
            return None, (
                "The second receiver is recording, which keeps it on one "
                "frequency: stop its recording to sweep with it."
            )
        return SecondReceiverPartner(receiver), ""

    def scan_begin(self) -> None:
        win = self._win
        panel = win._control_panel
        mode, deviation = win._listening_mode()
        code = panel.dcs_squelch()
        self._saved = (
            win._current_frequency(),
            win._lo_frequency(),
            mode,
            deviation,
            panel._bw_combo.currentText(),
            float(panel.tone_squelch_hz()),
            format_dcs(*code) if code else "",
        )
        self.expected_hz = win._current_frequency()

    def scan_end(self, stay: bool) -> None:
        """Back to where the scan started (with its mode, filter and tone
        squelch), unless ``stay``; either way the frequency now joins the
        tuning history once the receiver stays on it."""
        win = self._win
        win._scan_muted = False
        win._scan_checks_tone = False
        saved, self._saved = self._saved, None
        self.expected_hz = None
        # (A transmission stays where the scan is: it can't be retuned.)
        if not stay and saved is not None and not win._tx_busy():
            self._go_back(*saved)
        win._note_tuned()

    def _go_back(
        self,
        freq: float,
        lo: float,
        mode: str,
        deviation: str,
        width: str,
        tone: float,
        dcs: str = "",
    ) -> None:
        """To where the scan began, in the mode, filter and tone squelch it
        began with. A receiver that can't go back (it refused) stays on the
        scan's channel, in that channel's mode."""
        win = self._win
        off_centre = abs(lo - freq) >= 0.5

        def centre() -> bool:
            # The plots where they were too: the LO first (with the VFO on
            # it), so the scan's filter, maybe too wide to fit beside it,
            # doesn't move the LO onto the VFO.
            if off_centre and win._tune_to(lo, recenter=True):
                return True
            return bool(win._tune_to(freq, recenter=True))

        if not self._quietly(centre):
            return
        if mode:
            self._quietly(
                lambda: (
                    win._set_demod_mode(mode, deviation or None, width or None),
                    True,
                )[1]
            )
        # (Whatever the mode: the scan sets it for the FM channels it visits,
        # and it is kept, and saved, for when FM is next used.)
        self._set_squelch(tone, dcs)
        if off_centre and abs(win._lo_frequency() - lo) < 0.5:
            # Then the VFO, in the filter it was tuned with: only it moves.
            self._quietly(lambda: win._tune_to(freq))

    # ---- results ---------------------------------------------------------- #
    def scan_listen(self, hit: ScanHit) -> bool:
        """Tune to a result in its mode, filter and tone squelch (a
        bookmark's own, else none), and start the receiver if it is
        stopped, so it is heard."""
        win = self._win
        label = hit.label or default_bookmark_name(hit.frequency_hz)
        win._on_bookmark_tune(hit.frequency_hz, label, hit.mode, hit.fm_deviation)
        if abs(win._current_frequency() - hit.frequency_hz) >= 0.5:
            return False  # refused: the status bar says why
        if hit.mode and hit.bandwidth:
            win._set_demod_mode(hit.mode, hit.fm_deviation or None, hit.bandwidth)
        if hit.mode == "FM":
            tone = float(hit.tone_hz or 0.0)
            code = parse_dcs(hit.dcs) if hit.dcs else None
            self._set_squelch(tone, hit.dcs)
            if code is not None:
                squelch = dcs_label(*code)
            else:
                squelch = f"tone squelch {tone:.1f} Hz" if tone else ""
            if squelch:
                # (As tuning the bookmark says.)
                win._show_status_message(
                    f"Tuned to {label} (FM, {squelch})", "info", 3000
                )
        if win._device is not None and not win._is_running:
            win._start_acquisition()
        return True

    def scan_add_bookmarks(
        self, hits: List[ScanHit], group: str = ""
    ) -> Tuple[int, List[str]]:
        panel = self._win._bookmarks_panel
        names = [hit.label or default_bookmark_name(hit.frequency_hz) for hit in hits]
        # All at once: saved and listed once (Bookmark All of 500 results
        # took seconds one by one).
        results = panel.add_unique_many(
            [
                (name, hit.frequency_hz, hit.mode, hit.fm_deviation, hit.bandwidth)
                for name, hit in zip(names, hits, strict=True)
            ],
            group=group,
        )
        added, existing = 0, []
        for name, (index, new) in zip(names, results, strict=True):
            if new:
                added += 1
            else:
                bookmark = panel.bookmark(index) or {}
                existing.append(str(bookmark.get("label") or name))
        return added, existing

    def scan_bookmark_list(self, group: str = "") -> List[Dict[str, Any]]:
        panel = self._win._bookmarks_panel
        # In the order the Bookmarks list shows them (each group first).
        bookmarks = panel.bookmarks()
        visits = bookmark_scan_list(
            bookmarks, listening, group, order=panel.list_order()
        )
        for visit in visits:
            width = _parse_hz(visit.get("bandwidth") or "")
            visit["width_hz"] = width or 10e3
            # (A DCS code squelch, FM only, as the CTCSS tone is.)
            code = bookmark_dcs(bookmarks[visit["index"]])
            if visit.get("demod") == "FM" and code is not None:
                visit["dcs"] = format_dcs(*code)
        return visits

    def scan_bookmark_order(self) -> List[int]:
        return self._win._bookmarks_panel.list_order()

    def scan_bookmark_groups(self) -> List[str]:
        return self._win._bookmarks_panel.groups()

    def scan_bookmark_counts(self) -> Tuple[int, int]:
        bookmarks = self._win._bookmarks_panel.bookmarks()
        skipped = sum(1 for b in bookmarks if str(b.get("skip") or "") == "S")
        return len(bookmarks), skipped

    def scan_use_demo(self) -> None:
        self._win._start_demo_mode()

    def scan_connect(self) -> None:
        self._win._show_device_dialog(start_after=True)

    def scan_message(self, text: str, tone: Optional[str] = None) -> None:
        # (Not logged: the scan lists its problems itself, a scan that broke
        # or steps the receiver refused, whichever tab is shown. Its stop
        # for something you did is a notice: Error History lists problems.)
        self._win._show_status_message(text, tone, 4000, log=False)

    def scan_show_panel(self) -> None:
        self._win._show_panel("scanner", focus=True)
