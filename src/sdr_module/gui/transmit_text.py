"""
Tools > Transmit Text (Ctrl+Shift+T): send text in Morse code (CW), RTTY,
PSK31 or ASCII FSK.

A window of its own (it needs room to type, and the receiver's plots stay
in view while it is open):

* the mode and its settings (speed, baud rate, shift, the audio tone, and
  how it goes on the air: on the carrier, or as audio on an FM carrier);
* the text, with its length and air time, the characters the mode can't
  send (with a button that removes them) and macros that fill in your
  callsign from Ham Radio > Ham ID;
* **Preview** plays it through the speakers (what a receiver would play),
  **Save...** writes it as WAV audio (for a radio's data input) or as I/Q
  (SigMF or cf32, centered on the tuned frequency), and **Transmit** sends it
  on a HackRF One through the window's transmit engine
  (:meth:`~sdr_module.gui.main_window.SDRMainWindow._transmit_job`): the
  legality check, the first-use confirmation, your automatic station IDs,
  the ON AIR badge and Stop all apply. A line above the buttons says what
  Transmit would do now, or why it can't.

The text is turned into a signal by :mod:`sdr_module.protocols.text_modes`
(no Qt), the same code as the library encoders and ``sdr-scan encode``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QMimeData, QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

from ..protocols.text_modes import (
    ASCII_BAUDS,
    ASCII_SHIFTS,
    AUDIO_RANGE,
    EMISSION_DIRECT,
    EMISSION_FM,
    MAX_TEXT_CHARS,
    MODE_ASCII,
    MODE_CW,
    MODE_NAMES,
    MODE_PSK31,
    MODE_RTTY,
    RTTY_BAUDS,
    RTTY_SHIFTS,
    TEXT_MODES,
    TextSettings,
    TextSignal,
    bandwidth_rule_problem,
    char_name,
    format_number,
    remove_unsupported,
    write_audio_wav,
    write_iq_file,
)
from .audio_sink import AudioSink
from .themes import mono_font, set_role, set_tone
from .transmitter import (
    TxJob,
    format_dbm_power,
    format_mhz,
    id_goes_first,
    legality_problem,
)

logger = logging.getLogger(__name__)

#: Settings key (JSON): the mode, each mode's settings, the text, the file
#: format last saved.
SETTINGS_KEY = "tx/text"
#: Folder memory kind for Save (gui/file_dialogs.py).
FILE_KIND = "tx_text"

#: The rate a text is sent at: the HackRF's lowest (as the station ID), so
#: a narrow text costs the least and the engine's interval IDs, made inside
#: the transmit callback, keep up.
TX_SAMPLE_RATE = 2e6
#: Digital level of the transmitted signal (headroom below full scale).
TX_AMPLITUDE = 0.9
#: Audio and I/Q files are written at this rate.
FILE_RATE = 48000
#: Audio level of Preview and of a saved WAV (-6 dBFS, room for a radio's
#: data input to be set by its own level control).
AUDIO_LEVEL = 0.5

# "Sent as" per mode: (text, value, tooltip). "auto" (CW only) picks CW
# below 30 MHz and in the CW-only segments, an FM tone elsewhere above, as
# Send ID Now does.
_CW_EMISSIONS = (
    (
        "Auto (by frequency)",
        "auto",
        "A keyed carrier (CW) below 30 MHz and in the CW-only parts of 6 m "
        "and 2 m, where CW is heard in CW or SSB; an FM tone (MCW) elsewhere "
        "above, which FM receivers hear",
    ),
    (
        "CW (keyed carrier)",
        EMISSION_DIRECT,
        "The carrier keyed on and off on the tuned frequency (A1A): heard in "
        "CW or SSB, not in FM",
    ),
    (
        "FM tone (MCW)",
        EMISSION_FM,
        "The Morse as a keyed tone on an FM carrier (F2A): heard on FM "
        "receivers and repeaters",
    ),
)
_DATA_EMISSIONS = {
    MODE_RTTY: (
        (
            "FSK on the carrier",
            EMISSION_DIRECT,
            "The carrier shifts between mark (85 Hz above the tuned frequency at "
            "170 Hz shift) and space: what RTTY stations send. An SSB receiver "
            "in USB tuned 2.21 kHz below hears the usual 2125/2295 Hz tones",
        ),
        (
            "FM (audio tones)",
            EMISSION_FM,
            "The two audio tones on an FM carrier: heard on FM receivers",
        ),
    ),
    MODE_PSK31: (
        (
            "PSK on the carrier",
            EMISSION_DIRECT,
            "Phase-shift keying of the carrier on the tuned frequency, about 60 "
            "Hz wide: heard with an SSB receiver (in USB 1 kHz below, it plays "
            "as a 1 kHz tone)",
        ),
        (
            "FM (audio tone)",
            EMISSION_FM,
            "The PSK31 audio tone on an FM carrier: heard on FM receivers",
        ),
    ),
    MODE_ASCII: (
        (
            "FSK on the carrier",
            EMISSION_DIRECT,
            "The carrier shifts between mark (above the tuned frequency) and "
            "space (below), centered on the tuned frequency",
        ),
        (
            "FM (audio tones)",
            EMISSION_FM,
            "The two audio tones on an FM carrier: heard on FM receivers",
        ),
    ),
}

_MODE_ITEMS = (
    (MODE_CW, "CW (Morse code)", "On/off keyed Morse code, PARIS timing"),
    (MODE_RTTY, "RTTY", "Radio teletype: ITA2 (Baudot), 45.45 Bd and 170 Hz shift"),
    (MODE_PSK31, "PSK31", "Keyboard-to-keyboard PSK at 31.25 Bd, upper and lower case"),
    (MODE_ASCII, "ASCII FSK", "ASCII bytes as 8N1 FSK, like a serial port (Bell 103)"),
)

#: Macros: (button, text, tooltip); {call} is your callsign from Ham ID.
MACROS = (
    ("CQ", "CQ CQ CQ DE {call} {call} {call} K", "A general call: anyone may answer"),
    ("QRZ?", "QRZ? DE {call} K", "Ask who is calling you"),
    ("73", "TU 73 DE {call} SK", "End a contact: thank you, best regards, end"),
    ("My call", "{call}", "Your callsign"),
)

#: Save formats: (key, file filter, extension).
_SAVE_FORMATS = (
    ("wav", "WAV audio (*.wav)", ".wav"),
    ("sigmf", "SigMF I/Q recording (*.sigmf-data)", ".sigmf-data"),
    ("cf32", "Raw I/Q, complex float32 (*.cf32)", ".cf32"),
)

#: How often the "what Transmit would do" line is brought up to date.
_POLL_MS = 500


def format_seconds(seconds: float) -> str:
    """``14.23 -> "14.2 s"``, ``125 -> "2 min 05 s"``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    whole = int(round(seconds))
    return f"{whole // 60} min {whole % 60:02d} s"


def _in_amateur_band(freq_hz: float) -> bool:
    """Whether ``freq_hz`` is in an amateur band (for any license class)."""
    from ..core.frequency_manager import AMATEUR_BAND_PRIVILEGES

    return any(b.start_hz <= freq_hz <= b.end_hz for b in AMATEUR_BAND_PRIVILEGES)


def resolve_emission(mode: str, choice: str, freq_hz: float) -> str:
    """``"direct"`` or ``"fm"``: "auto" (CW) by frequency, as Send ID Now
    does (a keyed carrier below 30 MHz and in the CW-only segments, an FM
    tone elsewhere above)."""
    if choice in (EMISSION_DIRECT, EMISSION_FM):
        return choice
    if mode != MODE_CW:
        return EMISSION_DIRECT
    from ..ham.callsign import EMISSION_MCW, choose_id_emission

    return (
        EMISSION_FM if choose_id_emission(freq_hz) == EMISSION_MCW else EMISSION_DIRECT
    )


def _finite(values: Dict[str, Any], key: str, default: float) -> float:
    """``values[key]`` (``default`` when missing) as a number; ValueError
    when it isn't one (JSON allows NaN and Infinity)."""
    value = float(values.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{key} is not a number")
    return value


def snippet(text: str, limit: int = 24) -> str:
    """The start of ``text`` on one line, for messages: “CQ CQ DE W1AW…”."""
    line = " ".join(str(text).split())
    if len(line) > limit:
        line = line[: limit - 1].rstrip() + "…"
    return f"“{line}”"


if HAS_PYQT6:

    class _LimitedTextEdit(QPlainTextEdit):
        """A plain text box that holds at most ``limit`` characters: of what
        is typed, pasted or dropped, only what fits goes in (``overflow``
        says some didn't), and the text already there stays, so Undo works
        as usual."""

        overflow = pyqtSignal()

        def __init__(self, limit: int, parent: Any = None):
            super().__init__(parent)
            self.limit = int(limit)

        def room(self) -> int:
            """How many more characters fit (a selection is replaced)."""
            used = len(self.toPlainText()) - len(self.textCursor().selectedText())
            return max(0, self.limit - used)

        def insertFromMimeData(self, source):  # noqa: N802 - Qt naming
            if source is not None and source.hasText():
                # (As the box inserts it: CR LF and CR are new lines.)
                text = source.text().replace("\r\n", "\n").replace("\r", "\n")
                room = self.room()
                if len(text) > room:
                    self.overflow.emit()
                    fits = QMimeData()
                    fits.setText(text[:room])
                    source = fits
            super().insertFromMimeData(source)

        def keyPressEvent(self, event):  # noqa: N802 - Qt naming
            text = event.text().replace("\r", "\n")
            typed = bool(text) and (text.isprintable() or text == "\n")
            if typed and len(text) > self.room():
                self.overflow.emit()
                event.accept()
                return
            super().keyPressEvent(event)

    class TextPreview(QObject):
        """Plays a :class:`TextSignal`'s audio through an
        :class:`~.audio_sink.AudioSink` in real time (a little ahead of the
        clock, the way the receiver feeds it), so memory stays small for any
        length of text.

        Signals:
            playing_changed(bool)
            progress(float, float): seconds played, seconds in all.
        """

        playing_changed = pyqtSignal(bool)
        progress = pyqtSignal(float, float)

        RATE = FILE_RATE
        #: Audio handed to the output ahead of the clock (the sink's own
        #: buffer keeps playing through a late tick).
        LEAD_S = 0.15
        TICK_MS = 40

        def __init__(
            self, parent: Any = None, clock: Callable[[], float] = time.monotonic
        ):
            super().__init__(parent)
            self._clock = clock
            self._sink: Optional[AudioSink] = None
            self._signal: Optional[TextSignal] = None
            self._total = 0
            self._written = 0
            self._started = 0.0
            self._timer = QTimer(self)
            self._timer.setInterval(self.TICK_MS)
            self._timer.timeout.connect(self.tick)

        @property
        def playing(self) -> bool:
            return self._sink is not None

        @property
        def duration_s(self) -> float:
            return self._total / float(self.RATE)

        def start(
            self,
            signal: TextSignal,
            volume: float = 0.7,
            configure: Optional[Callable[[AudioSink], None]] = None,
        ) -> bool:
            """Play ``signal``; False when there is no audio output.
            ``configure(sink)`` sets the new sink up first (its output)."""
            self.stop()
            sink = AudioSink()
            sink.set_volume(volume)
            if configure is not None:
                configure(sink)
            if not sink.start(self.RATE):
                return False
            self._sink = sink
            self._signal = signal
            self._total = signal.sample_count(self.RATE)
            self._written = 0
            self._started = self._clock()
            self._timer.start()
            self.playing_changed.emit(True)
            self.tick()
            return True

        def tick(self) -> None:
            """Hand the output the audio due by now (plus the lead)."""
            sink, signal = self._sink, self._signal
            if sink is None or signal is None:
                return
            elapsed = self._clock() - self._started
            due = min(self._total, int((elapsed + self.LEAD_S) * self.RATE))
            if due > self._written:
                t = signal.times(self.RATE, self._written, due - self._written)
                sink.write(signal.audio(t) * AUDIO_LEVEL)
                self._written = due
            self.progress.emit(min(elapsed, self.duration_s), self.duration_s)
            if self._written >= self._total and elapsed >= self.duration_s + 0.25:
                self.stop()

        def stop(self) -> None:
            self._timer.stop()
            sink, self._sink = self._sink, None
            self._signal = None
            if sink is not None:
                sink.stop()
                self.playing_changed.emit(False)

    class TransmitTextDialog(QDialog):
        """The Transmit Text window (see the module docstring). Owned by
        the main window (``window``), which it reads: the tuned frequency,
        the device, the transmit station (``_tx_station``), the Ham ID
        panel and the settings."""

        def __init__(self, window: Any):
            super().__init__(window)
            self._win = window
            self._settings = window._settings
            # Each mode's own settings, and the "Sent as" choice per mode.
            self._mode_settings: Dict[str, TextSettings] = {
                m: TextSettings.for_mode(m) for m in TEXT_MODES
            }
            self._emission_choice: Dict[str, str] = {
                MODE_CW: "auto",
                MODE_RTTY: EMISSION_DIRECT,
                MODE_PSK31: EMISSION_DIRECT,
                MODE_ASCII: EMISSION_DIRECT,
            }
            self._save_format = "wav"
            self._loading = False
            self._signal_cache: Optional[Tuple[tuple, TextSignal]] = None
            # Our transmission while it is on the air (None otherwise).
            self._job: Optional[TxJob] = None
            # The last outcome (sent, stopped, refused, saved), shown until
            # something it depends on changes: (inputs, text, tone).
            self._outcome: Optional[Tuple[tuple, str, Optional[str]]] = None
            self._line_text = ""
            # The text before the last edit (to find what an edit added).
            self._last_text = ""

            self._preview = TextPreview(self)
            self._preview.playing_changed.connect(self._on_preview_playing)
            self._preview.progress.connect(self._on_preview_progress)

            self._setup_ui()
            self._load_settings()
            self._connect_engine()
            # The macros and the ID line follow Ham ID (callsign, auto-ID).
            panel = getattr(window, "_callsign_panel", None)
            if panel is not None:
                panel.callsign_changed.connect(self._on_ham_id_changed)
                panel.settings_changed.connect(self._on_ham_id_changed)

            self._poll = QTimer(self)
            self._poll.setInterval(_POLL_MS)
            self._poll.timeout.connect(self.refresh)

            self.setWindowTitle("Transmit Text")
            self.setModal(False)
            self.resize(self.sizeHint().expandedTo(self.minimumSizeHint()))
            self.refresh()

        # ------------------------------------------------------------ UI

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            # --- Mode and its settings -------------------------------
            mode_box = QGroupBox("Mode")
            grid = QGridLayout(mode_box)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(8)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)

            self._mode_combo = QComboBox()
            for index, (value, text, tip) in enumerate(_MODE_ITEMS):
                self._mode_combo.addItem(text, value)
                self._mode_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
            self._mode_combo.setToolTip("How the text is sent")
            self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
            self._add_grid_row(grid, 0, 0, "&Mode:", self._mode_combo)

            self._emission_combo = QComboBox()
            self._emission_combo.setToolTip(
                "How it goes on the air: on the carrier itself (heard in CW or "
                "SSB), or as audio on an FM carrier (heard on FM receivers)"
            )
            self._emission_combo.currentIndexChanged.connect(self._on_emission_changed)
            self._add_grid_row(grid, 0, 2, "Sent &as:", self._emission_combo)

            # Speed: WPM (CW), the baud rate (RTTY, ASCII) or fixed (PSK31).
            speed = QWidget()
            speed_row = QHBoxLayout(speed)
            speed_row.setContentsMargins(0, 0, 0, 0)
            self._wpm_spin = QSpinBox()
            self._wpm_spin.setRange(5, 60)
            self._wpm_spin.setSuffix(" WPM")
            self._wpm_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._wpm_spin.setToolTip("CW speed in words per minute (PARIS)")
            self._wpm_spin.valueChanged.connect(self._on_params_changed)
            speed_row.addWidget(self._wpm_spin)
            self._baud_combo = QComboBox()
            self._baud_combo.setToolTip(
                "Symbols per second. RTTY is 45.45 Bd almost everywhere. Below "
                "29.7 MHz US rules allow RTTY and data at most 2.8 kHz wide (the "
                "line above Transmit says when a setting is too wide)"
            )
            self._baud_combo.currentIndexChanged.connect(self._on_params_changed)
            speed_row.addWidget(self._baud_combo)
            self._fixed_rate = QLabel("31.25 Bd")
            self._fixed_rate.setToolTip("PSK31 always runs at 31.25 baud")
            set_role(self._fixed_rate, "value")
            speed_row.addWidget(self._fixed_rate)
            self._speed_label = self._add_grid_row(grid, 1, 0, "&Speed:", speed)

            self._shift_combo = QComboBox()
            self._shift_combo.setToolTip(
                "Distance between the mark and space tones. 170 Hz is the "
                "amateur RTTY standard"
            )
            self._shift_combo.currentIndexChanged.connect(self._on_params_changed)
            self._shift_label = self._add_grid_row(
                grid, 1, 2, "S&hift:", self._shift_combo
            )

            self._tone_spin = QSpinBox()
            self._tone_spin.setRange(int(AUDIO_RANGE[0]), int(AUDIO_RANGE[1]))
            self._tone_spin.setSingleStep(10)
            self._tone_spin.setSuffix(" Hz")
            self._tone_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._tone_spin.valueChanged.connect(self._on_params_changed)
            self._tone_label = self._add_grid_row(grid, 2, 0, "&Tone:", self._tone_spin)

            self._tones_hint = QLabel("")
            set_role(self._tones_hint, "hint")
            self._tones_hint.setWordWrap(True)
            grid.addWidget(self._tones_hint, 2, 2, 1, 2)
            layout.addWidget(mode_box)

            # --- The text --------------------------------------------
            text_box = QGroupBox("Text")
            text_layout = QVBoxLayout(text_box)
            text_layout.setSpacing(6)
            self._text_edit = _LimitedTextEdit(MAX_TEXT_CHARS)
            self._text_edit.overflow.connect(self._on_overflow)
            self._text_edit.setFont(mono_font(11))
            self._text_edit.setPlaceholderText(
                "Type what to send, or insert a macro below…"
            )
            self._text_edit.setAccessibleName("Text to send")
            self._text_edit.setToolTip(
                f"What to send, up to {MAX_TEXT_CHARS:,} characters. CW and RTTY "
                "send capitals; characters a mode can't send are listed below"
            )
            self._text_edit.setTabChangesFocus(True)
            self._text_edit.setMinimumHeight(90)
            self._text_edit.textChanged.connect(self._on_text_changed)
            text_layout.addWidget(self._text_edit, 1)

            self._count_label = QLabel("")
            set_role(self._count_label, "hint")
            self._count_label.setAccessibleName("Length and air time")
            text_layout.addWidget(self._count_label)

            # Characters the mode can't send, with a way to remove them.
            unsupported = QHBoxLayout()
            unsupported.setSpacing(8)
            self._unsupported_label = QLabel("")
            self._unsupported_label.setWordWrap(True)
            set_tone(self._unsupported_label, "warning")
            unsupported.addWidget(self._unsupported_label, 1)
            self._remove_button = QPushButton("Remove Them")
            set_role(self._remove_button, "compact")
            self._remove_button.setToolTip(
                "Delete the characters this mode can't send from the text"
            )
            self._remove_button.clicked.connect(self._remove_unsupported)
            unsupported.addWidget(self._remove_button, 0, Qt.AlignmentFlag.AlignTop)
            self._unsupported_row = QWidget()
            self._unsupported_row.setLayout(unsupported)
            unsupported.setContentsMargins(0, 0, 0, 0)
            self._unsupported_row.setVisible(False)
            text_layout.addWidget(self._unsupported_row)

            # Macros (your callsign from Ham ID).
            macros = QHBoxLayout()
            macros.setSpacing(6)
            caption = QLabel("Insert:")
            set_role(caption, "hint")
            macros.addWidget(caption)
            self._macro_buttons: List[QPushButton] = []
            for title, template, tip in MACROS:
                button = QPushButton(title)
                set_role(button, "compact")
                button.setProperty("macro_template", template)
                button.setProperty("macro_tip", tip)
                button.clicked.connect(
                    lambda _c=False, t=template: self.insert_macro(t)
                )
                macros.addWidget(button)
                self._macro_buttons.append(button)
            macros.addStretch(1)
            self._callsign_hint = QLabel("")
            set_role(self._callsign_hint, "hint")
            macros.addWidget(self._callsign_hint)
            self._ham_id_button = QPushButton("Ham ID…")
            set_role(self._ham_id_button, "compact")
            self._ham_id_button.setToolTip(
                "Open Ham Radio > Ham ID to enter your callsign (the macros and "
                "your automatic station ID use it)"
            )
            self._ham_id_button.clicked.connect(self._show_ham_id)
            macros.addWidget(self._ham_id_button)
            text_layout.addLayout(macros)
            layout.addWidget(text_box, 1)

            # --- What Transmit would do; progress ---------------------
            self._tx_label = QLabel("")
            self._tx_label.setWordWrap(True)
            self._tx_label.setAccessibleName("Transmit status")
            self._tx_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            set_role(self._tx_label, "hint")
            layout.addWidget(self._tx_label)

            self._progress = QProgressBar()
            self._progress.setRange(0, 1000)
            self._progress.setTextVisible(True)
            self._progress.setAccessibleName("Progress")
            self._progress.setToolTip("How far the preview or the transmission is")
            self._progress.setVisible(False)
            layout.addWidget(self._progress)

            # --- Actions ----------------------------------------------
            buttons = QHBoxLayout()
            buttons.setSpacing(8)
            self._preview_button = QPushButton("&Preview")
            self._preview_button.clicked.connect(self._on_preview_clicked)
            buttons.addWidget(self._preview_button)
            self._save_button = QPushButton("Sa&ve…")
            self._save_button.setToolTip(
                "Save as WAV audio (for a radio's data input, or any player), "
                "or as I/Q centered on the tuned frequency (SigMF or cf32, 48 "
                "kS/s): no device needed"
            )
            self._save_button.clicked.connect(self._on_save_clicked)
            buttons.addWidget(self._save_button)
            buttons.addStretch(1)
            self._close_button = QPushButton("Close")
            self._close_button.setToolTip(
                "Close this window (a transmission on the air goes on: Stop or "
                "the ON AIR badge ends it)"
            )
            self._close_button.clicked.connect(self.close)
            buttons.addWidget(self._close_button)
            self._transmit_button = QPushButton("T&ransmit")
            set_role(self._transmit_button, "danger")
            self._transmit_button.clicked.connect(self._on_transmit_clicked)
            buttons.addWidget(self._transmit_button)
            layout.addLayout(buttons)
            # No default button: Enter in a field (a speed, the tone, a
            # list) must not press one. A dialog otherwise makes its first
            # button the default, and Enter pressed Remove Them.
            for button in self.findChildren(QPushButton):
                button.setAutoDefault(False)
                button.setDefault(False)

            self.setMinimumWidth(520)

        def _add_grid_row(
            self, grid: QGridLayout, row: int, column: int, text: str, field: QWidget
        ) -> QLabel:
            label = QLabel(text)
            label.setBuddy(field)
            grid.addWidget(label, row, column)
            grid.addWidget(field, row, column + 1)
            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return label

        # ------------------------------------------------------ settings

        def _load_settings(self) -> None:
            raw = ""
            try:
                raw = str(self._settings.get_str(SETTINGS_KEY, "") or "")
            except Exception:  # pragma: no cover - defensive
                raw = ""
            data: Dict[str, Any] = {}
            if raw:
                try:
                    loaded = json.loads(raw)
                    data = loaded if isinstance(loaded, dict) else {}
                except ValueError:
                    logger.warning("Ignoring unreadable Transmit Text settings")
            try:
                self._load_mode_settings(data)
            except (TypeError, AttributeError, ValueError) as e:
                # (A value of a shape this doesn't know: the defaults stay.)
                logger.warning(f"Ignoring Transmit Text settings it can't use: {e}")
            mode = data.get("mode", MODE_RTTY)
            if not isinstance(mode, str) or mode not in TEXT_MODES:
                mode = MODE_RTTY
            index = self._mode_combo.findData(mode)
            self._loading = True
            try:
                self._mode_combo.setCurrentIndex(max(0, index))
                self._show_mode(self.mode())
                text = data.get("text", "")
                if isinstance(text, str):
                    self._text_edit.setPlainText(text[:MAX_TEXT_CHARS])
            finally:
                self._loading = False
            self._update_text_info()

        def _load_mode_settings(self, data: Dict[str, Any]) -> None:
            """Each mode's saved settings and the file format, where they
            are of the right type (anything else keeps the default)."""
            modes = data.get("modes")
            for mode, values in (modes if isinstance(modes, dict) else {}).items():
                if mode not in self._mode_settings or not isinstance(values, dict):
                    continue
                current = self._mode_settings[mode]
                try:
                    self._mode_settings[mode] = TextSettings.for_mode(
                        mode,
                        wpm=_finite(values, "wpm", current.wpm),
                        baud=_finite(values, "baud", current.baud),
                        shift_hz=_finite(values, "shift_hz", current.shift_hz),
                        audio_hz=_finite(values, "audio_hz", current.audio_hz),
                    )
                except (TypeError, ValueError):
                    continue
                emission = str(values.get("emission", ""))
                if emission in ("auto", EMISSION_DIRECT, EMISSION_FM):
                    if emission != "auto" or mode == MODE_CW:
                        self._emission_choice[mode] = emission
            save_format = data.get("save_format")
            if isinstance(save_format, str) and save_format in {
                key for key, _f, _e in _SAVE_FORMATS
            }:
                self._save_format = save_format

        def _save_settings(self) -> None:
            if self._loading:
                return
            data = {
                "mode": self.mode(),
                "text": self.text(),
                "save_format": self._save_format,
                "modes": {
                    mode: {
                        "wpm": s.wpm,
                        "baud": s.baud,
                        "shift_hz": s.shift_hz,
                        "audio_hz": s.audio_hz,
                        "emission": self._emission_choice[mode],
                    }
                    for mode, s in self._mode_settings.items()
                },
            }
            try:
                self._settings.set(SETTINGS_KEY, json.dumps(data))
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Could not save the Transmit Text settings: {e}")

        # ------------------------------------------------- mode controls

        def mode(self) -> str:
            return str(self._mode_combo.currentData() or MODE_RTTY)

        def text(self) -> str:
            return self._text_edit.toPlainText()

        def set_text(self, text: str) -> None:
            self._text_edit.setPlainText(text)

        def set_mode(self, mode: str) -> None:
            index = self._mode_combo.findData(mode)
            if index >= 0:
                self._mode_combo.setCurrentIndex(index)

        def settings(self) -> TextSettings:
            """The settings of the mode chosen."""
            return self._mode_settings[self.mode()]

        def emission_choice(self) -> str:
            return self._emission_choice[self.mode()]

        def emission(self, freq_hz: Optional[float] = None) -> str:
            """``"direct"`` or ``"fm"`` at ``freq_hz`` (default: tuned)."""
            freq = self._win._current_frequency() if freq_hz is None else freq_hz
            return resolve_emission(self.mode(), self.emission_choice(), float(freq))

        def _on_mode_changed(self, _index: int = 0) -> None:
            if self._loading:
                return
            self._show_mode(self.mode())
            self._after_change()

        def _show_mode(self, mode: str) -> None:
            """Fill the settings' controls with ``mode``'s own settings."""
            s = self._mode_settings[mode]
            widgets = (
                self._wpm_spin,
                self._baud_combo,
                self._shift_combo,
                self._tone_spin,
                self._emission_combo,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                cw, psk = mode == MODE_CW, mode == MODE_PSK31
                fsk = mode in (MODE_RTTY, MODE_ASCII)
                self._wpm_spin.setVisible(cw)
                self._wpm_spin.setValue(int(round(s.wpm)))
                self._baud_combo.setVisible(fsk)
                self._fixed_rate.setVisible(psk)
                self._shift_combo.setVisible(fsk)
                self._shift_label.setVisible(fsk)
                if fsk:
                    bauds = RTTY_BAUDS if mode == MODE_RTTY else ASCII_BAUDS
                    shifts = RTTY_SHIFTS if mode == MODE_RTTY else ASCII_SHIFTS
                    self._fill(self._baud_combo, bauds, s.baud, " Bd")
                    self._fill(self._shift_combo, shifts, s.shift_hz, " Hz")
                self._speed_label.setBuddy(
                    self._wpm_spin
                    if cw
                    else self._baud_combo if fsk else self._mode_combo
                )
                self._show_tone(s)
                self._tone_label.setText("&Tone:" if not fsk else "&Center:")
                self._tone_spin.setToolTip(self._tone_tip(mode))
                items = _CW_EMISSIONS if cw else _DATA_EMISSIONS[mode]
                self._emission_combo.clear()
                for index, (text, value, tip) in enumerate(items):
                    self._emission_combo.addItem(text, value)
                    self._emission_combo.setItemData(
                        index, tip, Qt.ItemDataRole.ToolTipRole
                    )
                index = self._emission_combo.findData(self._emission_choice[mode])
                self._emission_combo.setCurrentIndex(max(0, index))
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            self._update_tones_hint()

        @staticmethod
        def _fill(combo: QComboBox, values: tuple, current: float, unit: str) -> None:
            combo.clear()
            choices = list(values)
            if all(abs(v - current) > 1e-6 for v in choices):
                choices.append(current)  # (a value saved that isn't listed)
                choices.sort()
            for value in choices:
                combo.addItem(f"{format_number(value)}{unit}", float(value))
            index = next(
                (i for i, v in enumerate(choices) if abs(v - current) < 1e-6), 0
            )
            combo.setCurrentIndex(index)

        @staticmethod
        def _tone_tip(mode: str) -> str:
            if mode == MODE_CW:
                return (
                    "The pitch you hear in Preview and in a saved WAV, and the "
                    "tone of an FM tone (MCW) transmission. A keyed carrier has "
                    "no tone of its own: the listener's receiver makes it"
                )
            if mode == MODE_PSK31:
                return (
                    "Where the PSK31 signal sits in the audio (Preview, a saved "
                    "WAV, the FM emission); on the carrier it is the tuned "
                    "frequency itself"
                )
            return (
                "The center between the two tones in the audio (Preview, a "
                "saved WAV, the FM emission); mark is the higher tone, as USB "
                "data modes expect. 2210 Hz gives the usual 2125/2295 Hz"
            )

        def _on_emission_changed(self, _index: int = 0) -> None:
            value = self._emission_combo.currentData()
            if value is None:
                return
            self._emission_choice[self.mode()] = str(value)
            self._after_change()

        def _on_params_changed(self, *_args) -> None:
            mode = self.mode()
            s = self._mode_settings[mode]
            values: Dict[str, float] = {"audio_hz": float(self._tone_spin.value())}
            if mode == MODE_CW:
                values["wpm"] = float(self._wpm_spin.value())
            elif mode in (MODE_RTTY, MODE_ASCII):
                if self._baud_combo.currentData() is not None:
                    values["baud"] = float(self._baud_combo.currentData())
                if self._shift_combo.currentData() is not None:
                    values["shift_hz"] = float(self._shift_combo.currentData())
            self._mode_settings[mode] = TextSettings.for_mode(
                mode,
                wpm=values.get("wpm", s.wpm),
                baud=values.get("baud", s.baud),
                shift_hz=values.get("shift_hz", s.shift_hz),
                audio_hz=values["audio_hz"],
            )
            # (A wider shift narrows the center's range: show what is used.)
            self._show_tone(self._mode_settings[mode])
            self._update_tones_hint()
            self._after_change()

        def _show_tone(self, s: TextSettings) -> None:
            """The tone (CW, PSK31) or FSK center box: its range keeps every
            tone between 100 and 3500 Hz at this shift (a 100 Hz center
            with a 1000 Hz shift would put space at -400 Hz), and it shows
            the value that is used."""
            low, high = s.audio_bounds()
            blocked = self._tone_spin.blockSignals(True)
            try:
                self._tone_spin.setRange(int(math.ceil(low)), int(math.floor(high)))
                self._tone_spin.setValue(int(round(s.audio_hz)))
            finally:
                self._tone_spin.blockSignals(blocked)

        def _update_tones_hint(self) -> None:
            s = self.settings()
            if s.mode in (MODE_RTTY, MODE_ASCII):
                space, mark = s.tones()
                text = f"Mark {format_number(mark)} Hz, space {format_number(space)} Hz"
            elif s.mode == MODE_PSK31:
                text = "Upper and lower case, every ASCII character"
            else:
                text = "Prosigns: <SK>, <AR>, <KN>, <BT>"
            self._tones_hint.setText(text)

        def _after_change(self) -> None:
            self._signal_cache = None
            if self._preview.playing:
                self._preview.stop()
            self._update_text_info()
            self._save_settings()
            self.refresh()

        # ------------------------------------------------------------ text

        def _on_overflow(self) -> None:
            self._win._show_status_message(
                f"Transmit Text holds up to {MAX_TEXT_CHARS:,} characters: what "
                "didn't fit was left out.",
                "warning",
                5000,
            )

        def _on_text_changed(self) -> None:
            text = self.text()
            if len(text) > MAX_TEXT_CHARS:
                # (Typing, pasting and dropping stop at the limit; this is
                # for whatever else gets past it, e.g. set_text.)
                text = self._drop_excess(text)
                self._on_overflow()
            self._last_text = text
            if self._loading:
                return
            self._signal_cache = None
            if self._preview.playing:
                self._preview.stop()
            self._update_text_info()
            self.refresh()

        def _drop_excess(self, text: str) -> str:
            """Take what doesn't fit off the end of what the last edit put
            in, so the text that was already there stays. Returns the text
            kept."""
            old = self._last_text
            # The edit is where the old and new text differ: the inserted
            # text is text[start:end].
            limit = min(len(old), len(text))
            start = 0
            while start < limit and old[start] == text[start]:
                start += 1
            tail = 0
            while tail < limit - start and old[-1 - tail] == text[-1 - tail]:
                tail += 1
            end = len(text) - tail
            cut = end - (len(text) - MAX_TEXT_CHARS)
            if cut < start:  # (not one edit within the limit: cut the end)
                cut, end = MAX_TEXT_CHARS, len(text)

            def position(index: int) -> int:  # (Qt counts UTF-16 units)
                return len(text[:index].encode("utf-16-le", "surrogatepass")) // 2

            edit = self._text_edit
            edit.blockSignals(True)
            try:
                cursor = QTextCursor(edit.document())
                cursor.beginEditBlock()
                cursor.setPosition(position(cut))
                cursor.setPosition(position(end), QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                cursor.endEditBlock()
                view = edit.textCursor()
                view.setPosition(position(cut))
                edit.setTextCursor(view)
            finally:
                edit.blockSignals(False)
            return text[:cut] + text[end:]

        def current_signal(self) -> TextSignal:
            """The text as a signal with the settings chosen (cached)."""
            key = (self.text(), self.settings())
            cached = self._signal_cache
            if cached is not None and cached[0] == key:
                return cached[1]
            signal = TextSignal(key[0], key[1])
            self._signal_cache = (key, signal)
            return signal

        def _update_text_info(self) -> None:
            signal = self.current_signal()
            s = signal.settings
            typed = len(self.text())
            if signal.empty:
                count = "Nothing to send yet"
            else:
                n = typed  # as typed (what is left out is said below)
                rate = (
                    f"{format_number(s.wpm)} WPM"
                    if s.mode == MODE_CW
                    else f"{format_number(s.symbol_rate)} Bd"
                )
                count = (
                    f"{n:,} character{'' if n == 1 else 's'} · "
                    f"{format_seconds(signal.duration_s)} at {rate}"
                )
            if s.mode in (MODE_CW, MODE_RTTY) and any(c.islower() for c in self.text()):
                count += " · sent in capitals"
            if typed >= 0.9 * MAX_TEXT_CHARS:
                count += f" · {typed:,} of {MAX_TEXT_CHARS:,} characters"
            self._count_label.setText(count)
            check = signal.check
            if check.unsupported:
                name = MODE_NAMES[s.mode]
                skipped = check.skipped
                self._unsupported_label.setText(
                    f"{name} can't send {check.describe_unsupported()}: "
                    f"{'it is' if skipped == 1 else 'they are'} left out."
                )
                self._unsupported_label.setToolTip(
                    f"{name} has no "
                    + ", ".join(char_name(c) for c in check.unsupported)
                    + ". What is sent: "
                    + (check.sent or "nothing")
                )
            self._unsupported_row.setVisible(bool(check.unsupported))
            self._update_macros()

        def _remove_unsupported(self) -> None:
            """Delete the characters the mode can't send (an undoable edit)."""
            signal = self.current_signal()
            if not signal.check.unsupported:
                return
            cleaned = remove_unsupported(self.text(), signal.mode)
            cursor = self._text_edit.textCursor()
            cursor.beginEditBlock()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.insertText(cleaned)
            cursor.endEditBlock()
            self._text_edit.setFocus()

        # ---------------------------------------------------------- macros

        def _callsign(self) -> str:
            """Your callsign from Ham ID ("" when there is none yet)."""
            panel = getattr(self._win, "_callsign_panel", None)
            if panel is None or not panel.is_callsign_valid():
                return ""
            return str(panel.get_callsign())

        def _update_macros(self) -> None:
            call = self._callsign()
            for button in self._macro_buttons:
                template = str(button.property("macro_template"))
                tip = str(button.property("macro_tip"))
                button.setEnabled(bool(call))
                if call:
                    button.setToolTip(f"{tip}: inserts “{template.format(call=call)}”")
                else:
                    button.setToolTip(
                        f"{tip}. Enter your callsign in Ham Radio > Ham ID first"
                    )
            if call:
                self._callsign_hint.setText(f"as {call}")
                self._callsign_hint.setToolTip("Your callsign, from Ham Radio > Ham ID")
            else:
                self._callsign_hint.setText("Needs your callsign:")
                self._callsign_hint.setToolTip(
                    "The macros fill in your callsign from Ham Radio > Ham ID"
                )
            self._ham_id_button.setVisible(not call)

        def insert_macro(self, template: str) -> bool:
            """Insert ``template`` with your callsign at the cursor, with a
            space either side where it would run into other text."""
            call = self._callsign()
            if not call:
                self._win._show_status_message(
                    "Enter your callsign in Ham Radio > Ham ID to use the macros.",
                    "warning",
                    5000,
                )
                return False
            insert = template.format(call=call)
            cursor = self._text_edit.textCursor()
            text = self.text()
            before = text[: cursor.selectionStart()]
            after = text[cursor.selectionEnd() :]
            if before and not before[-1].isspace():
                insert = " " + insert
            if after and not after[0].isspace():
                insert += " "
            if len(insert) > self._text_edit.room():
                self._win._show_status_message(
                    f"The macro doesn't fit: Transmit Text holds up to "
                    f"{MAX_TEXT_CHARS:,} characters.",
                    "warning",
                    5000,
                )
                return False
            cursor.insertText(insert)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.setFocus()
            return True

        def _on_ham_id_changed(self, *_args) -> None:
            self._update_macros()
            if self.isVisible():
                self.refresh()

        def _show_ham_id(self) -> None:
            show = getattr(self._win, "_show_panel", None)
            if callable(show):
                show("ham_id", focus=True)
                self._win.raise_()
                self._win.activateWindow()

        # --------------------------------------------------------- readiness

        def _station(self) -> Any:
            return getattr(self._win, "_tx_station", None)

        def _our_job_on_air(self) -> bool:
            station = self._station()
            return bool(
                self._job is not None
                and station is not None
                and station.busy
                and station.engine.job is self._job
            )

        def _inputs(self) -> tuple:
            """What the outcome line depends on (a change drops it)."""
            win = self._win
            from ..core import frequency_manager

            return (
                self.text(),
                self.settings(),
                self.emission_choice(),
                round(float(win._current_frequency())),
                id(getattr(win, "_device", None)),
                bool(getattr(win, "_demo_mode", False)),
                str(frequency_manager.get_license_class().value),
                self._callsign(),
            )

        def readiness(self) -> Tuple[str, Optional[str], bool]:
            """``(line, tone, can transmit)``: what Transmit would do now,
            or why it can't."""
            win = self._win
            station = self._station()
            if station is None:  # pragma: no cover - the window always has one
                return "Transmitting isn't available.", "warning", False
            if self._our_job_on_air():
                return "On the air: Stop ends it.", "danger", False
            if station.busy:
                job = station.engine.job
                what = job.description if job is not None else "another transmission"
                return (
                    f"Wait: {what} is on the air (Radio > Stop Transmitting ends "
                    "it).",
                    "warning",
                    False,
                )
            signal = self.current_signal()
            if signal.empty and signal.check.unsupported:
                return (
                    f"Nothing to send: {MODE_NAMES[signal.mode]} can't send "
                    f"{signal.check.describe_unsupported()}.",
                    "warning",
                    False,
                )
            if signal.empty:
                return "Type the text to send, or insert a macro.", None, False
            freq = float(win._current_frequency())
            where = format_mhz(freq)
            problem = station.device_problem()
            if problem is not None:
                return (
                    f"Can't send: {problem}. (Preview and Save need no device.)",
                    "warning",
                    False,
                )
            emission = self.emission(freq)
            mode = signal.tx_mode(emission)
            width = signal.occupied_bandwidth_hz(emission)
            problem = bandwidth_rule_problem(freq, signal.mode, width)
            if problem is None:
                problem = legality_problem(freq, width, mode, quiet=True)
            if problem is not None:
                return f"Can't send on {where}: {problem}.", "danger", False
            job_s = signal.source(TX_SAMPLE_RATE, emission).duration_s
            ids, id_problem = self._id_words(freq, mode, job_s)
            if id_problem is not None:
                return f"Can't send on {where}: {id_problem}.", "danger", False
            problem = station.power_problem(freq, mode)
            if problem is not None:
                return f"Can't send on {where}: {problem}.", "danger", False
            estimate = format_dbm_power(station.power.output_dbm(freq)).split(" (")[0]
            how = signal.describe() + (", FM" if emission == EMISSION_FM else "")
            return (
                f"Will send on {where} · {how} · TX gain {station.power.gain_db} dB "
                f"(about {estimate}) · {ids}",
                "info",
                True,
            )

        def _id_words(
            self, freq: float, job_mode: Optional[str] = None, job_s: float = 0.0
        ) -> Tuple[str, Optional[str]]:
            """How the transmission is identified ("ID DE W1AW (CW, 20 WPM)
            at the start and end": the ID it gets, ``job_mode`` being its
            mode and ``job_s`` its air time), and why it can't be (None: it
            can)."""
            station = self._station()
            panel = getattr(self._win, "_callsign_panel", None)
            call = self._callsign()
            if not call:
                if _in_amateur_band(freq):
                    return "", (
                        "an amateur transmission must be identified: enter your "
                        "callsign in Ham Radio > Ham ID first"
                    )
                return "no station ID (no callsign in Ham ID)", None
            values = panel.auto_id_settings() if panel is not None else {}
            start, end = bool(values.get("at_start")), bool(values.get("at_end"))
            interval = values.get("interval_s")
            if not (start or end or interval):
                return "no automatic ID (Ham ID > Automatic ID is off)", None
            _emission, id_mode, id_width, summary = station.id_spec(freq, job_mode)
            problem = legality_problem(freq, id_width, id_mode, quiet=True)
            if problem is not None:
                return "", f"your station ID can't go there: {problem}"
            since = station.engine.seconds_since_last_id()
            due_first = bool(
                not start
                and interval
                and since is not None
                and id_goes_first(float(interval) - since, float(interval) / 2.0, job_s)
            )
            if start and end and not interval:
                when = "at the start and end"
            else:
                parts = []
                if start:
                    parts.append("at the start")
                elif due_first:  # (the interval since the last ID runs out)
                    parts.append(
                        f"at the start ({int(since // 60)} min since the last)"
                    )
                if interval:
                    parts.append(f"every {int(float(interval) // 60)} min")
                if end:
                    parts.append("at the end")
                when = (
                    parts[0]
                    if len(parts) == 1
                    else (", ".join(parts[:-1]) + " and " + parts[-1])
                )
            return f"ID DE {call} ({summary}) {when}", None

        def refresh(self) -> None:
            """Bring the line above the buttons and the buttons up to date."""
            if self._our_job_on_air():
                self._transmit_button.setText("Stop")
                self._transmit_button.setEnabled(True)
                self._transmit_button.setToolTip("Stop transmitting now")
                return
            ready, tone, can = self.readiness()
            text = ready
            if self._outcome is not None:
                if self._outcome[0] == self._inputs():
                    _inputs, text, tone = self._outcome
                else:
                    self._outcome = None
            self._set_line(text, tone)
            self._transmit_button.setText("T&ransmit")
            self._transmit_button.setEnabled(bool(can))
            self._transmit_button.setToolTip(
                "Transmit the text on the tuned frequency (HackRF One): checked, "
                "with your automatic station ID; the first time asks first"
                if can
                else ready
            )
            empty = self.current_signal().empty
            self._save_button.setEnabled(not empty)
            if not self._preview.playing:
                self._preview_button.setEnabled(not empty)
                self._preview_button.setToolTip(
                    "Type some text to preview it."
                    if empty
                    else "Play it through your speakers as a receiver would "
                    "(nothing is transmitted)"
                )

        def _set_line(self, text: str, tone: Optional[str]) -> None:
            if text != self._line_text:
                self._line_text = text
                self._tx_label.setText(text)
                self._tx_label.setToolTip(text)
            set_tone(self._tx_label, tone)

        def _set_outcome(self, text: str, tone: Optional[str]) -> None:
            self._outcome = (self._inputs(), text, tone)
            self.refresh()

        # --------------------------------------------------------- preview

        def _on_preview_clicked(self) -> None:
            if self._preview.playing:
                self._preview.stop()
                return
            signal = self.current_signal()
            if signal.empty:
                return
            volume = 0.7
            saved_volume = getattr(self._win, "_saved_volume", None)
            if callable(saved_volume):
                volume = saved_volume() / 100.0
            if not self._preview.start(signal, volume, self._configure_sink):
                self._set_outcome(
                    "No audio output to preview on. Save… writes a WAV file you "
                    "can play elsewhere.",
                    "warning",
                )
                return
            logger.info(f"Previewing {signal.describe()} ({signal.duration_s:.1f} s)")

        def _configure_sink(self, sink: AudioSink) -> None:
            """Play on the audio output chosen in Tools > Preferences, as the
            receiver does (when the sink can choose one; else the default)."""
            choose = getattr(sink, "set_output_device", None)
            if not callable(choose):
                return
            try:
                device = self._settings.get_str("prefs/audio_output", "")
                name = self._settings.get_str("prefs/audio_output_name", "")
                if device:
                    choose(device, name)
            except Exception as e:  # pragma: no cover - backend-dependent
                logger.debug(f"Preview plays on the default output: {e}")

        def _on_preview_playing(self, playing: bool) -> None:
            self._preview_button.setText("Stop &Preview" if playing else "&Preview")
            if playing:
                self._preview_button.setEnabled(True)
                self._preview_button.setToolTip("Stop the preview")
            if not self._our_job_on_air():
                self._progress.setVisible(playing)
            self.refresh()

        def _on_preview_progress(self, done: float, total: float) -> None:
            if self._our_job_on_air():
                return
            self._show_progress("Preview", done, total)

        def _show_progress(self, what: str, done: float, total: float) -> None:
            total = max(float(total), 1e-6)
            done = min(max(float(done), 0.0), total)
            self._progress.setVisible(True)
            self._progress.setValue(int(round(1000 * done / total)))
            self._progress.setFormat(f"{what} · {done:.1f} / {total:.1f} s")

        # ------------------------------------------------------------ save

        def _on_save_clicked(self) -> None:
            from . import file_dialogs
            from .file_jobs import run_job

            signal = self.current_signal()
            if signal.empty:
                return
            win = self._win
            freq = float(win._current_frequency())
            stamp = time.strftime("%Y%m%d_%H%M%S")
            mode = signal.mode.lower()
            ext = {key: e for key, _f, e in _SAVE_FORMATS}[self._save_format]
            name = f"{mode}_{freq / 1e6:.3f}MHz_{stamp}{ext}"
            filters = ";;".join(f for _key, f, _e in _SAVE_FORMATS)
            selected = {key: f for key, f, _e in _SAVE_FORMATS}[self._save_format]
            path, chosen = file_dialogs.save_file(
                self,
                self._settings,
                FILE_KIND,
                "Save Text Signal",
                name,
                filters,
                selected,
            )
            if not path:
                return
            kind = self._format_for(path, chosen)
            path = self._with_extension(path, kind)
            self._save_format = kind
            self._save_settings()
            emission = self.emission(freq)
            description = f"{signal.describe()}: {signal.check.sent[:200]}"

            def work(report: Callable[[float], None]) -> int:
                if kind == "wav":
                    return write_audio_wav(
                        path, signal, FILE_RATE, AUDIO_LEVEL, report=report
                    )
                source = signal.source(FILE_RATE, emission, amplitude=TX_AMPLITUDE)
                return write_iq_file(
                    path, source, kind, freq, description, report=report
                )

            result = run_job(self, "Saving", f"Saving {os.path.basename(path)}…", work)
            shown = file_dialogs.display_path(path)
            if result.cancelled:
                self._set_outcome("Not saved: cancelled.", None)
                return
            if not result.ok:
                error = result.error
                message = f"Couldn't save {os.path.basename(path)}: {error}"
                logger.error(message)
                win._show_status_error(message, log=False)
                self._set_outcome(message + ".", "danger")
                return
            what = (
                "audio"
                if kind == "wav"
                else f"I/Q at {format_mhz(freq)} ({FILE_RATE / 1000:g} kS/s)"
            )
            seconds = int(result.value or 0) / float(FILE_RATE)
            message = (
                f"Saved {MODE_NAMES[signal.mode]} {what} to {shown} "
                f"({format_seconds(seconds)})."
            )
            win._show_status_message(message, "success", 6000)
            self._set_outcome("✓ " + message, "success")

        @staticmethod
        def _format_for(path: str, chosen: str) -> str:
            lower = path.lower()
            for key, _f, ext in _SAVE_FORMATS:
                if lower.endswith(ext):
                    return key
            if lower.endswith(".sigmf-meta"):
                return "sigmf"
            for key, flt, _e in _SAVE_FORMATS:
                if chosen == flt:
                    return key
            return "wav"

        @staticmethod
        def _with_extension(path: str, kind: str) -> str:
            ext = {key: e for key, _f, e in _SAVE_FORMATS}[kind]
            lower = path.lower()
            if kind == "sigmf" and lower.endswith(".sigmf-meta"):
                return path[: -len(".sigmf-meta")] + ext
            return path if lower.endswith(ext) else path + ext

        # -------------------------------------------------------- transmit

        def _connect_engine(self) -> None:
            station = self._station()
            if station is None:  # pragma: no cover - defensive
                return
            station.engine.started.connect(self._on_tx_started)
            station.engine.progress.connect(self._on_tx_progress)
            station.engine.finished.connect(self._on_tx_finished)

        def _on_transmit_clicked(self) -> None:
            station = self._station()
            if station is None:
                return
            if self._our_job_on_air():
                station.stop()
                return
            line, tone, can = self.readiness()
            if not can:
                self._set_line(line, tone)
                return
            self._preview.stop()
            win = self._win
            signal = self.current_signal()
            freq = float(win._current_frequency())
            emission = self.emission(freq)
            source = signal.source(TX_SAMPLE_RATE, emission, amplitude=TX_AMPLITUDE)
            s = signal.settings
            if s.mode == MODE_CW:
                summary = f"{format_number(s.wpm)} WPM"
            elif s.mode == MODE_PSK31:
                summary = "31.25 Bd"
            else:
                summary = (
                    f"{format_number(s.baud)} Bd, {format_number(s.shift_hz)} Hz shift"
                )
            if emission == EMISSION_FM:
                summary += ", FM"
            # A new job (and source) for every send: a source is consumed.
            job = TxJob(
                source,
                TX_SAMPLE_RATE,
                freq,
                source.mode,
                source.occupied_bandwidth_hz,
                f"{MODE_NAMES[s.mode]} {snippet(signal.check.sent)}",
                summary,
            )
            self._job = job
            self._save_settings()
            if win._transmit_job(job):
                return
            self._job = None
            # Not sent: the window said why (or the question was cancelled).
            label = getattr(win, "_message_label", None)
            message = label.text() if label is not None else ""
            tone = (label.property("tone") or None) if label is not None else None
            if message:
                self._set_outcome(message, tone)
            else:  # pragma: no cover - the window always says why
                self._set_outcome("Not sent.", "warning")

        def _on_tx_started(self, job: Any) -> None:
            if job is self._job:
                self._preview.stop()
                self._show_progress("On the air", 0.0, job.duration_s)
                self._set_line(
                    f"On the air: {job.description} on {format_mhz(job.center_hz)}. "
                    "Stop ends it.",
                    "danger",
                )
            self.refresh()

        def _on_tx_progress(self, done: float, total: float) -> None:
            if not self._our_job_on_air():
                return
            self._show_progress("On the air", done, total)
            station = self._station()
            next_id = station.engine.next_id_in() if station is not None else None
            # (Only when an interval ID falls inside this transmission.)
            if next_id is not None and next_id < total - done:
                self._progress.setFormat(
                    self._progress.format() + f" · next ID in {format_seconds(next_id)}"
                )

        def _on_tx_finished(self, result: Any) -> None:
            if self._job is None or result.job is not self._job:
                self.refresh()
                return
            self._job = None
            self._progress.setVisible(self._preview.playing)
            stamp = time.strftime("%H:%M")
            if result.status == "sent":
                ids = f", with {result.ids_sent} ID" if result.ids_sent else ""
                ids += "s" if result.ids_sent > 1 else ""
                self._set_outcome(f"✓ {result.message}{ids} at {stamp}.", "success")
            elif result.status == "stopped":
                line = f"{result.message}."
                if (
                    not result.ids_sent
                    and self._callsign()
                    and _in_amateur_band(result.job.center_hz)
                ):
                    # (As the Ham ID tab says: no ID went out with it.)
                    line += (
                        " Send your ID (Radio > Send Station ID) to finish the contact."
                    )
                self._set_outcome(line, "warning")
            else:
                self._set_outcome(f"✗ {result.message}.", "danger")

        # ---------------------------------------------------------- window

        def focus_text(self) -> None:
            self._text_edit.setFocus()
            self._text_edit.moveCursor(QTextCursor.MoveOperation.End)

        def showEvent(self, event):  # noqa: N802 - Qt naming
            super().showEvent(event)
            self._update_macros()
            self.refresh()
            self._poll.start()

        def hideEvent(self, event):  # noqa: N802 - Qt naming
            self._poll.stop()
            self._preview.stop()
            self._save_settings()
            if self._our_job_on_air():
                job = self._job
                self._win._show_status_message(
                    f"Still transmitting {job.description if job else 'text'}: "
                    "Radio > Stop Transmitting (or the ON AIR badge) ends it.",
                    "danger",
                    8000,
                )
            super().hideEvent(event)

        def shutdown(self) -> None:
            """The window is closing: stop the preview, keep the draft."""
            self._poll.stop()
            self._preview.stop()
            self._save_settings()


__all__ = [
    "FILE_KIND",
    "MACROS",
    "SETTINGS_KEY",
    "TX_SAMPLE_RATE",
    "format_seconds",
    "resolve_emission",
    "snippet",
]
if HAS_PYQT6:
    __all__ += ["TextPreview", "TransmitTextDialog"]
