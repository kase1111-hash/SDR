"""
The main window as the Scanner tab's host.

:class:`WindowScannerHost` is what the scanner (``scanner_panel`` and
``scan_engine``) calls to tune the receiver, hear whether the squelch is
open, start the receiver, add bookmarks and so on. It keeps the window's
side of the scanner in one place, so ``main_window.py`` only needs a few
hooks: the panel, ``_scan_frame`` in the display loop and the mute in
``_route_audio``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..devices.base import SDRDevice
from .bookmarks_panel import default_bookmark_name, listening
from .scan_engine import Listening, bookmark_scan_list
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


class WindowScannerHost(ScannerHost):
    """Adapts the main window (``SDRMainWindow``) to the scanner."""

    def __init__(self, window: Any):
        self._win = window
        # Where the receiver was before the scan: (frequency, LO, mode,
        # deviation, bandwidth, tone squelch).
        self._saved: Optional[Tuple[float, float, str, str, str, float]] = None
        self._lo_moved = False
        #: The frequency the scan tuned last (None outside a scan): any
        #: other means the user tuned elsewhere.
        self.expected_hz: Optional[float] = None

    # ---- tuning ----------------------------------------------------------- #
    def _quietly(self, tune) -> bool:
        """Tune without the status-bar messages and saved settings that a
        mode change brings (a memory scan changes modes all the time)."""
        win = self._win
        restoring = win._restoring
        win._restoring = True
        try:
            return bool(tune())
        finally:
            win._restoring = restoring

    def _after_tune(self) -> None:
        win = self._win
        # Scan steps don't join Tune Back / Recent Frequencies (where the
        # scan ends does, see scan_end).
        win._history_timer.stop()
        # A new channel: its squelch decides afresh (the last channel's
        # hang time must not count as this one's signal).
        win._squelch.reset()
        win._level_average.reset()
        self.expected_hz = win._current_frequency()
        if self._lo_moved:
            self._drop_stale_transfers()

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
        # so FREQ shows the LO instead.
        keep_centered = bool(getattr(win, "_keep_centered", lambda: False)())

        def tune() -> bool:
            if not win._tune_to(center_hz, recenter=True):
                return False
            if vfo_hz is not None and abs(vfo_hz - center_hz) >= 0.5:
                if not keep_centered:
                    win._tune_to(vfo_hz)  # inside the span: only the VFO moves
            return True

        ok = self._quietly(tune)
        self._lo_moved = abs(win._lo_frequency() - before) >= 0.5
        if ok:
            self._after_tune()
        return ok

    def scan_tune_channel(
        self, freq_hz: float, listening: Listening, tone_hz: float = 0.0
    ) -> bool:
        win = self._win
        before = win._lo_frequency()
        # A bookmark with a tone squelch is judged by its tone, so the
        # window checks it even with the speaker off (see _route_audio).
        win._scan_checks_tone = bool(tone_hz and tone_hz > 0)

        def tune() -> bool:
            if not win._tune_to(freq_hz):
                return False
            self._apply_listening(listening, tone_hz)
            return True

        ok = self._quietly(tune)
        self._lo_moved = abs(win._lo_frequency() - before) >= 0.5
        if ok:
            self._after_tune()
        return ok

    def _apply_listening(self, listening: Listening, tone_hz: float) -> None:
        demod, deviation, width = listening
        if not demod:
            return
        win = self._win
        width = width or win._bandwidth_for_mode(demod, deviation) or ""
        win._set_demod_mode(demod, deviation or None, width or None)
        if demod == "FM":
            win._control_panel.set_tone_squelch_hz(float(tone_hz or 0.0))

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
        return self._level_above_threshold()

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

    def scan_begin(self) -> None:
        win = self._win
        panel = win._control_panel
        mode, deviation = win._listening_mode()
        self._saved = (
            win._current_frequency(),
            win._lo_frequency(),
            mode,
            deviation,
            panel._bw_combo.currentText(),
            float(panel.tone_squelch_hz()),
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
        if not stay and saved is not None:
            freq, lo, mode, deviation, width, tone = saved

            def back() -> bool:
                # The plots where they were too: the LO, then the VFO in it.
                if abs(lo - freq) >= 0.5 and win._tune_to(lo, recenter=True):
                    return bool(win._tune_to(freq))
                return bool(win._tune_to(freq, recenter=True))

            self._quietly(back)
            if mode:
                self._quietly(
                    lambda: (
                        win._set_demod_mode(mode, deviation or None, width or None),
                        True,
                    )[1]
                )
                if mode == "FM":
                    win._control_panel.set_tone_squelch_hz(tone)
        win._note_tuned()

    # ---- results ---------------------------------------------------------- #
    def scan_listen(self, hit: ScanHit) -> bool:
        """Tune to a result in its mode, filter and (no) tone squelch, and
        start the receiver if it is stopped, so it is heard."""
        win = self._win
        label = hit.label or default_bookmark_name(hit.frequency_hz)
        win._on_bookmark_tune(hit.frequency_hz, label, hit.mode, hit.fm_deviation)
        if abs(win._current_frequency() - hit.frequency_hz) >= 0.5:
            return False  # refused: the status bar says why
        if hit.mode and hit.bandwidth:
            win._set_demod_mode(hit.mode, hit.fm_deviation or None, hit.bandwidth)
        if hit.mode == "FM":
            win._control_panel.set_tone_squelch_hz(0.0)
        if win._device is not None and not win._is_running:
            win._start_acquisition()
        return True

    def scan_add_bookmarks(self, hits: List[ScanHit]) -> Tuple[int, List[str]]:
        panel = self._win._bookmarks_panel
        names = [hit.label or default_bookmark_name(hit.frequency_hz) for hit in hits]
        # All at once: saved and listed once (Bookmark All of 500 results
        # took seconds one by one).
        results = panel.add_unique_many(
            [
                (name, hit.frequency_hz, hit.mode, hit.fm_deviation, hit.bandwidth)
                for name, hit in zip(names, hits, strict=True)
            ]
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
        visits = bookmark_scan_list(
            self._win._bookmarks_panel.bookmarks(), listening, group
        )
        for visit in visits:
            width = _parse_hz(visit.get("bandwidth") or "")
            visit["width_hz"] = width or 10e3
        return visits

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
        self._win._show_status_message(text, tone, 4000)
