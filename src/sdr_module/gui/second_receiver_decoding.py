"""
The Second Receiver's live decoder.

One protocol at a time, chosen in the window's Decoder group, decodes what
this receiver receives: the main window's live decoders
(:mod:`sdr_module.dsp.live_decoders`; each picks out its own channel at the
VFO, so every mode works and only the sample rate can rule one out, as
ADS-B below 2 MS/s). It decodes on the receiver's DSP worker
(``second_receiver.FrameWork``), never on the GUI thread; the frame hands
what it decoded back to the window, which puts the messages into the main
window's Decoder tab marked as this receiver's ("2nd RX · AX.25/APRS", at
this receiver's frequency and rate, as Decode Recording's rows are marked
"(file)"), and shows the last one and a compact status line.

:class:`ReceiverDecoding` (Qt-free) holds the decoder; :func:`decoder_line`
says how it is doing (the Decoder tab's status line, for this window);
:class:`DecoderRows` (Qt) is the Decoder group of the Second Receiver
window, a part of the window (settings ``device/second_receiver/decoder``
and ``decode``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

from ..dsp.live_decoders import (
    LIVE_DECODERS,
    create_live_decoder,
    describe_message,
    format_rate,
    live_decoder_class,
)
from .decoder_panel import (
    _FRESH_S,
    _ROW_UNITS,
    _UNDECODED_SIGNAL_S,
    PROTOCOL_INFO,
    PROTOCOL_UNDECODED,
    _format_age,
    _format_mhz,
    _plural,
)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .main_window import _ElidedLabel
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

__all__ = [
    "PROTOCOL_OFF",
    "PROTOCOLS",
    "RECEIVER_MARK",
    "ReceiverDecoding",
    "decoder_line",
    "rate_remedy",
]

#: The Second Receiver's settings (``second_receiver.KEY``).
KEY = "device/second_receiver/"
#: No decoder.
PROTOCOL_OFF = "Off"
#: The choices: Off, then the main window's live decoders.
PROTOCOLS: Tuple[str, ...] = (PROTOCOL_OFF,) + tuple(LIVE_DECODERS)
#: How the Decoder tab marks this receiver's rows ("2nd RX · POCSAG").
RECEIVER_MARK = "2nd RX"

#: (frames, valid, failed) of a decoder's status.
_Counts = Tuple[int, int, int]


def _counts(status: Any) -> _Counts:
    return tuple(  # type: ignore[return-value]
        int(getattr(status, name, 0) or 0) for name in ("frames", "valid", "failed")
    )


def rate_remedy(protocol: str, rates: Sequence[float]) -> str:
    """What to do when ``protocol`` needs a higher rate than this receiver
    runs at: the lowest of its device's ``rates`` the protocol takes, in
    this window's Sample rate list."""
    cls = live_decoder_class(protocol)
    low = float(getattr(cls, "MIN_SAMPLE_RATE", 0.0) or 0.0)
    usable = [float(rate) for rate in rates if float(rate) >= low]
    if not usable:
        return (
            f"This receiver's device doesn't run that fast: {protocol} needs "
            "another device."
        )
    return (
        f"Choose {format_rate(min(usable))} or more in Sample rate (this "
        "window's Device group)."
    )


class ReceiverDecoding:
    """The Second Receiver's live decoder (no Qt).

    The GUI thread chooses the protocol (:meth:`choose`, :meth:`flush`)
    only while no frame is on the DSP worker (the window settles the frame
    in flight first); the worker feeds each frame's raw samples to
    :meth:`process`, which follows the VFO and a retune, and takes the
    :meth:`status` snapshot the window shows. A decoder that fails is
    logged once and decodes nothing more from that frame (never stopping
    the receiver).
    """

    def __init__(self) -> None:
        self.protocol = PROTOCOL_OFF
        self.decoder: Any = None
        #: The rate the decoder was built for (Hz; 0: none).
        self.rate = 0.0
        #: Why the protocol's decoder couldn't be built, or None.
        self.problem: Optional[str] = None
        self._tuned: Optional[float] = None
        self._failed = False

    def choose(self, protocol: str, rate: float, offset_hz: float = 0.0) -> None:
        """Decode ``protocol`` (one of :data:`PROTOCOLS`; anything else is
        Off) from samples at ``rate``, its channel ``offset_hz`` from the
        LO: a new decoder (the old one's state and counts go)."""
        self.protocol = protocol if protocol in PROTOCOLS else PROTOCOL_OFF
        self.decoder = None
        self.problem = None
        self.rate = float(rate)
        self._tuned = None
        self._failed = False
        if self.protocol == PROTOCOL_OFF:
            return
        try:
            self.decoder = create_live_decoder(
                self.protocol, self.rate, float(offset_hz)
            )
        except Exception as exc:  # (a rate it can't take, or a bug)
            self.problem = f"The {self.protocol} decoder could not start: {exc}"

    def process(
        self, samples: Any, offset_hz: float, tuned_hz: Optional[float]
    ) -> List[Any]:
        """Decode one frame's raw samples (the DSP worker): the messages it
        completed (valid ones and ones that failed their checks)."""
        decoder = self.decoder
        if decoder is None:
            return []
        try:
            if float(offset_hz) != decoder.offset_hz:
                decoder.set_offset(float(offset_hz))
            if self._tuned is not None and tuned_hz != self._tuned:
                # What was measured on the old channel says nothing about
                # this one.
                decoder.channel_changed()
            self._tuned = tuned_hz
            return list(decoder.process(samples))
        except Exception as exc:  # pragma: no cover - defensive
            if not self._failed:
                self._failed = True
                logger.warning(f"The second receiver's {decoder.name} decoder: {exc}")
            return []

    def flush(self) -> List[Any]:
        """The end of the stream (Stop, another protocol): messages still
        being assembled (a Morse line in progress)."""
        decoder = self.decoder
        if decoder is None:
            return []
        try:
            return list(decoder.flush())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Second receiver decoder flush: {exc}")
            return []

    def status(self) -> Any:
        """The decoder's status snapshot (a ``LiveDecoderStatus``), or None
        without a decoder."""
        decoder = self.decoder
        return decoder.status if decoder is not None else None


def decoder_line(
    protocol: str,
    status: Any,
    *,
    decode: bool,
    running: bool,
    scanning: bool,
    rows: int,
    last_time: Optional[float],
    now: float,
    tuned_hz: Optional[float],
    problem: Optional[str] = None,
    remedy: str = "",
    base: _Counts = (0, 0, 0),
) -> Tuple[str, Optional[str], str]:
    """``(text, tone, tooltip)`` of the Decoder group's status line: the
    Decoder tab's status line (``DecoderPanel._status_core``) for this
    receiver. ``rows``: its messages in the Decoder tab since the protocol
    was chosen (or Clear), the last at ``last_time`` (``now``'s clock);
    ``problem``: why the decoder couldn't be built; ``remedy``: what to do
    about a rate too low; ``base``: the decoder's (frames, valid, failed)
    at the last Clear."""
    if protocol == PROTOCOL_OFF:
        return (
            "Choose a protocol to decode what this receiver hears.",
            "muted",
            "",
        )
    if not decode:
        return "Paused: tick Decode to resume", "muted", ""
    problem = problem or getattr(status, "problem", None)
    if problem:
        text = f"{problem}." if remedy else str(problem)
        if remedy:
            text += f" {remedy}"
        return text, "warning", text
    if not running:
        return "Stopped: press Start (Space) to decode", "muted", ""
    if scanning:
        return (
            "Paused while the Scanner sweeps with this receiver",
            "muted",
            "The Scanner tab's Use both receivers tunes this receiver step by "
            "step: it decodes again once the sweep ends.",
        )
    here = _format_mhz(tuned_hz) if tuned_hz else ""
    if status is None:
        return (f"Listening on {here}" if here else "Listening"), None, ""
    frames, valid, failed = (
        max(0, now_count - then)
        for now_count, then in zip(_counts(status), base, strict=True)
    )
    tip = (
        f"Since the decoder started (or Clear): {_plural(frames, 'frame')} "
        f"heard, {valid:,} decoded, {failed:,} failed their checks."
    )
    undecoded = float(getattr(status, "undecoded_signal_s", 0.0) or 0.0)
    if undecoded >= _UNDECODED_SIGNAL_S:
        return (
            PROTOCOL_UNDECODED.get(
                protocol,
                "Signal present but nothing decodes: check the frequency and "
                "the protocol",
            ),
            "warning",
            tip,
        )
    unit = str(getattr(status, "unit", "message") or "message")
    row_unit = unit if unit in _ROW_UNITS else "message"
    if rows or valid:
        times = [
            t for t in (getattr(status, "last_time", None), last_time) if t is not None
        ]
        age = now - max(times) if times else 0.0
        count = _plural(rows, row_unit)
        if unit not in _ROW_UNITS:
            count += f" ({_plural(valid, unit)})"
        detail = str(getattr(status, "detail", "") or "")
        parts = [protocol, detail, count, f"last {_format_age(age)}"]
        if failed:
            parts.append(f"{failed:,} failed check{'' if failed == 1 else 's'}")
        text = "  ·  ".join(part for part in parts if part)
        return text, ("success" if age < _FRESH_S else None), tip
    if failed:
        return (
            f"{_plural(failed, unit)} failed their checks: the signal may be "
            "weak, mistuned or another protocol",
            "warning",
            tip,
        )
    if getattr(status, "signal_present", False):
        return f"Signal on {here}: waiting for {protocol} {unit}s", "info", tip
    return (f"Listening on {here}" if here else "Listening"), None, tip


def last_message_text(msg: Any, when: Optional[float] = None) -> str:
    """The compact last-message line: its time, address and text."""
    address, content = describe_message(msg)
    stamp = time.strftime("%H:%M:%S", time.localtime(when))
    body = f"{address}: {content}" if address and content else (address or content)
    return f"{stamp}  {body}".strip()


if HAS_PYQT6:

    class DecoderRows:
        """Part of the Second Receiver window (a mixin of
        ``SecondReceiverWindow``): its Decoder group (the protocol, the
        Decode switch, the status line, the last message and Show in
        Decoder Tab), the decoder on its DSP worker (``self._work.
        decoding``) and its messages in the main window's Decoder tab.
        Uses the window's ``_win``, ``_work``, ``_combo``, ``_form``,
        ``_hint``, ``_save``, ``_settings``, ``_settle_frames``,
        ``_show_message``, ``_rate``, ``_rate_kind``, ``_running``,
        ``_scan_feed``, ``_tuned_hz``, ``_lo_hz`` and ``_main_gone``."""

        def _build_decoder_group(self) -> "QGroupBox":
            group = QGroupBox("Decoder")
            form = self._form(group)
            self._decoded_rows = 0
            self._decoded_at: Optional[float] = None
            self._decoder_status: Any = None
            self._decoder_base: _Counts = (0, 0, 0)
            self._decoder_said: Optional[Tuple[str, float]] = None
            combo = self._combo(8)
            for name in PROTOCOLS:
                combo.addItem(name, name)
                info = PROTOCOL_INFO.get(name)
                tip = (
                    f"{info[0]} ({info[1]})"
                    if info
                    else "No decoder: nothing is decoded here"
                )
                combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
            combo.setAccessibleName("Second receiver's decoder")
            combo.setToolTip(
                "Decode a protocol in what this receiver hears, on its own "
                "frequency (the decoder picks out its channel at the tuned "
                "frequency, whatever the Mode). Its messages go to the main "
                "window's Decoder tab, marked 2nd RX."
            )
            combo.currentIndexChanged.connect(self._on_decoder_chosen)
            self._decoder_combo = combo
            self._decode_check = QCheckBox("Decode")
            self._decode_check.setChecked(True)
            self._decode_check.setToolTip(
                "Untick to pause decoding here (the protocol stays chosen)"
            )
            self._decode_check.toggled.connect(self._on_decode_toggled)
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(combo, 1)
            row.addWidget(self._decode_check)
            label = QLabel("Protocol:")
            label.setBuddy(combo)
            form.addRow(label, row_widget)
            self._decoder_line = self._hint()
            self._decoder_line.setAccessibleName("Second receiver's decoder status")
            form.addRow(self._decoder_line)
            # The last message, on one line (the whole of it in its
            # tooltip and in the Decoder tab).
            self._decoder_last = _ElidedLabel("")
            set_role(self._decoder_last, "value")
            self._decoder_last.setAccessibleName(
                "Second receiver's last decoded message"
            )
            form.addRow(self._decoder_last)
            self._decoder_show = QPushButton("Show in Decoder Tab")
            self._decoder_show.setAutoDefault(False)
            self._decoder_show.setToolTip(
                "Bring the main window's Decoder tab to the front: this "
                "receiver's messages are there, marked 2nd RX"
            )
            self._decoder_show.clicked.connect(
                lambda _checked=False: self._show_decoder_tab()
            )
            form.addRow(self._decoder_show)
            self._decoder_form = form
            self._update_decoder_line()
            return group

        # -------------------------------------------------------- choosing

        def _decoder_protocol(self) -> str:
            name = self._decoder_combo.currentData()
            return name if name in PROTOCOLS else PROTOCOL_OFF

        def _on_decoder_chosen(self, *_args: Any) -> None:
            self._save(KEY + "decoder", self._decoder_protocol())
            self._rebuild_decoder()

        def _on_decode_toggled(self, on: bool) -> None:
            self._save(KEY + "decode", bool(on))
            self._update_decoder_line()

        def _decoding_now(self) -> bool:
            """Whether this frame's samples go to the decoder (the DSP
            worker decodes them): a protocol chosen, Decode ticked, and no
            scan sweeping with this receiver."""
            return (
                self._decode_check.isChecked()
                and self._work.decoding.decoder is not None
                and self._scan_feed is None
            )

        def _rebuild_decoder(self) -> None:
            """The decoder for the protocol chosen, at this receiver's rate
            (GUI thread). The frame on the worker is shown first (the
            decoder is the worker's meanwhile), and what the old decoder
            still held (a Morse line in progress) goes to the Decoder tab."""
            self._settle_frames()
            decoding = self._work.decoding
            self._push_decoded(decoding.flush(), self._tuned_hz)
            decoding.choose(
                self._decoder_protocol(), self._rate(), self._tuned_hz - self._lo_hz
            )
            self._decoded_rows = 0
            self._decoded_at = None
            self._decoder_base = (0, 0, 0)
            self._decoder_status = decoding.status()
            self._decoder_last.setText("")
            self._decoder_last.setToolTip("")
            self._say_decoder_problem()
            self._update_decoder_line()

        def _follow_decoder_rate(self) -> None:
            """A new sample rate (or another device): the decoder is built
            again for it, as the main window's is."""
            decoding = self._work.decoding
            if (
                decoding.protocol != PROTOCOL_OFF
                and abs(decoding.rate - self._rate()) >= 1.0
            ):
                self._rebuild_decoder()

        def _decoder_problem(self) -> Tuple[Optional[str], str]:
            """Why the decoder can't work here, and what to do about it
            (this window's own remedy for a rate too low)."""
            decoding = self._work.decoding
            problem = decoding.problem or getattr(self._decoder_status, "problem", None)
            if not problem:
                return None, ""
            cls = live_decoder_class(decoding.protocol)
            low = float(getattr(cls, "MIN_SAMPLE_RATE", 0.0) or 0.0)
            remedy = ""
            if self._rate() < low:
                from .second_receiver import SAMPLE_RATES

                remedy = rate_remedy(
                    decoding.protocol,
                    SAMPLE_RATES.get(self._rate_kind(), SAMPLE_RATES["demo"]),
                )
            return str(problem), remedy

        def _say_decoder_problem(self) -> None:
            """A decoder that can't work here (a rate too low): said in the
            status bar and logged, once per protocol and rate, as in the
            main window."""
            problem, remedy = self._decoder_problem()
            decoding = self._work.decoding
            key = (decoding.protocol, decoding.rate)
            if not problem or key == self._decoder_said:
                return
            self._decoder_said = key
            text = f"{problem}. {remedy}".strip() if remedy else problem
            logger.warning(f"Second receiver: {text}")
            if not self._restoring:
                self._show_message(text, "warning", 10000)

        # ------------------------------------------------------- messages

        def _push_decoded(self, messages: List[Any], tuned_hz: Any) -> None:
            """Messages this receiver decoded: into the main window's
            Decoder tab, marked as this receiver's (at its frequency and
            rate); the last one shown here."""
            if not messages or self._main_gone:
                return
            push = getattr(self._win, "_push_decoded_message", None)
            if not callable(push):
                return
            freq = float(tuned_hz) if tuned_hz is not None else self._tuned_hz
            for msg in messages:
                push(
                    msg,
                    freq_hz=freq,
                    sample_rate=self._rate(),
                    receiver=RECEIVER_MARK,
                )
                if not getattr(msg, "valid", True):
                    continue
                self._decoded_rows += 1
                self._decoded_at = time.monotonic()
                text = last_message_text(msg)
                self._decoder_last.setText(text)
                self._decoder_last.setToolTip(text)
            self._update_decoder_line()

        def _show_decoded(self, result: Any) -> None:
            """A frame's decoding (from the DSP worker): its messages and
            the decoder's status."""
            if result.decoder_status is not None:
                self._decoder_status = result.decoder_status
            if result.decoded:
                self._push_decoded(result.decoded, result.tuned_hz)
            elif result.decoder_status is not None:
                self._update_decoder_line()

        def _flush_decoder(self) -> None:
            """The stream ends (Stop): what the decoder still holds."""
            self._push_decoded(self._work.decoding.flush(), self._tuned_hz)

        def _on_decoder_tab_cleared(self) -> None:
            """The Decoder tab's Clear: this receiver's count and last
            message start afresh too (and the decoder forgets its notes)."""
            if self._main_gone:
                return
            self._settle_frames()
            decoding = self._work.decoding
            decoder = decoding.decoder
            if decoder is not None:
                decoder.messages_cleared()
            self._decoder_status = decoding.status()
            self._decoder_base = (
                _counts(self._decoder_status)
                if self._decoder_status is not None
                else (0, 0, 0)
            )
            self._decoded_rows = 0
            self._decoded_at = None
            self._decoder_last.setText("")
            self._decoder_last.setToolTip("")
            self._update_decoder_line()

        def _update_decoder_line(self) -> None:
            """The status line (refreshed by the status timer too: nothing
            is touched unless it changed)."""
            problem, remedy = self._decoder_problem()
            text, tone, tip = decoder_line(
                self._decoder_protocol(),
                self._decoder_status,
                decode=self._decode_check.isChecked(),
                running=self._running,
                scanning=self._scan_feed is not None,
                rows=self._decoded_rows,
                last_time=self._decoded_at,
                now=time.monotonic(),
                tuned_hz=self._tuned_hz,
                problem=problem,
                remedy=remedy,
                base=self._decoder_base,
            )
            label = self._decoder_line
            if label.text() != text:
                label.setText(text)
                label.updateGeometry()
            if label.toolTip() != tip:
                label.setToolTip(tip)
            set_tone(label, tone)
            self._decoder_show.setEnabled(not self._main_gone)

        def _show_decoder_tab(self) -> None:
            """Show in Decoder Tab: the main window's Decoder tab, in front."""
            win = self._win
            show = getattr(win, "_show_panel", None)
            if self._main_gone or not callable(show):
                return
            show("decoder")
            win.raise_()
            win.activateWindow()

        def _restore_decoder(self) -> None:
            """The saved protocol (a damaged one is logged and gives Off) and
            the Decode switch."""
            s = self._settings
            name = s.get_str(KEY + "decoder", PROTOCOL_OFF)
            if name not in PROTOCOLS:
                logger.warning(
                    f"Ignoring the damaged setting {KEY}decoder={name!r}: using "
                    "Off instead."
                )
                name = PROTOCOL_OFF
            self._decode_check.setChecked(s.get_bool(KEY + "decode", True))
            combo = self._decoder_combo
            index = max(0, combo.findData(name))
            if combo.currentIndex() != index:
                combo.setCurrentIndex(index)  # (builds the decoder)
            else:
                self._rebuild_decoder()
