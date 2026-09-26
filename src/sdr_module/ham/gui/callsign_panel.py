"""
Callsign identification panel for HAM radio compliance.

Provides UI controls for:
- Callsign input (uppercased and validated as you type)
- Automatic ID settings
- ID mode selection (only CW can be transmitted today)
- Audible preview of the CW ID and a manual "Send ID Now" trigger
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QStandardItemModel, QValidator
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

logger = logging.getLogger(__name__)

# Characters a callsign may contain (portable suffixes use "/").
_CALLSIGN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
# Audio rate used for the CW preview.
_PREVIEW_RATE = 48000
# Combo index of the only mode that can actually be transmitted.
_CW_INDEX = 0
_INTERVAL_TIP = (
    "Time between automatic IDs. FCC rules require an ID at least every "
    "10 minutes and at the end of a communication."
)


def _theme() -> Any:
    """The GUI theme helpers, imported lazily.

    ``sdr_module.gui`` imports this panel while it initialises, so importing
    ``sdr_module.gui.themes`` at module level here would be circular.
    """
    from ...gui import themes

    return themes


def _morse_code() -> dict:
    """International Morse table used by the CW ID generator."""
    from ..callsign import MorseEncoder

    return MorseEncoder.MORSE_CODE


def morse_pattern(text: str) -> str:
    """Dots and dashes for ``text``, letters separated by spaces, words by " / "."""
    table = _morse_code()
    words = []
    for word in text.upper().split():
        words.append(" ".join(table[c] for c in word if c in table))
    return " / ".join(w for w in words if w)


def morse_duration(text: str, wpm: int) -> float:
    """Seconds needed to send ``text`` at ``wpm`` (PARIS timing, as generated)."""
    table = _morse_code()
    text = text.upper().strip()
    units = 0
    for i, char in enumerate(text):
        if char == " ":
            units += 7
            continue
        code = table.get(char)
        if not code:
            continue
        units += sum(1 if s == "." else 3 for s in code) + (len(code) - 1)
        if i < len(text) - 1 and text[i + 1] != " ":
            units += 3
    return units * 60.0 / (50.0 * max(1, wpm))


if HAS_PYQT6:

    class CallsignValidator(QValidator):
        """Uppercases input as it is typed and rejects non-callsign characters.

        Spaces (e.g. in pasted text) are dropped rather than rejecting the
        whole edit, and the cursor stays where the user left it.
        """

        def validate(self, text: str, pos: int):  # noqa: D401 - Qt API
            cleaned = []
            new_pos = pos
            for i, char in enumerate(text):
                if char.isspace():
                    if i < pos:
                        new_pos -= 1
                    continue
                cleaned.append(char.upper())
            result = "".join(cleaned)
            if any(c not in _CALLSIGN_CHARS for c in result):
                return (QValidator.State.Invalid, text, pos)
            return (QValidator.State.Acceptable, result, new_pos)

        def fixup(self, text: str) -> str:
            return "".join(c for c in text.upper() if c in _CALLSIGN_CHARS)


class CallsignPanel(QWidget if HAS_PYQT6 else object):
    """
    Callsign identification control panel.

    Allows HAM operators to configure automatic callsign identification
    to comply with FCC/regulatory requirements.
    """

    if HAS_PYQT6:
        callsign_changed = pyqtSignal(str)
        id_requested = pyqtSignal()  # Manual ID request
        settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._callsign = ""
        self._is_transmitting = False
        # None = unknown (the main window checks the device when sending).
        self._tx_available: Optional[bool] = None
        self._tx_unavailable_reason = ""
        self._id_timer = QTimer(self)
        self._id_timer.timeout.connect(self._update_countdown)
        self._seconds_until_id = 0
        self._preview_sink: Any = None
        self._preview_buffer: Any = None
        self._form_labels: list = []

        self._setup_ui()
        self._update_controls()
        self._refresh_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup the user interface."""
        t = _theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- Station callsign ------------------------------------------
        callsign_group = QGroupBox("Station Callsign")
        callsign_form = self._make_form(callsign_group)

        self._callsign_input = QLineEdit()
        self._callsign_input.setPlaceholderText("e.g. W1AW")
        self._callsign_input.setMaxLength(10)
        self._callsign_input.setFont(t.mono_font(bold=True))
        self._callsign_input.setValidator(CallsignValidator(self._callsign_input))
        self._callsign_input.setClearButtonEnabled(True)
        self._callsign_input.setToolTip(
            "Your amateur radio callsign. Letters are uppercased as you type; "
            "use / for portable suffixes (e.g. W1AW/P)."
        )
        self._callsign_input.textChanged.connect(self._on_callsign_changed)
        self._add_row(callsign_form, "Callsign:", self._callsign_input)

        # Validation / activity feedback, directly under the field.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        t.set_role(self._status_label, "hint")
        callsign_form.addRow(self._status_label)

        # What will be sent, in Morse, with its duration.
        self._morse_label = QLabel()
        self._morse_label.setWordWrap(True)
        self._morse_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._morse_label.setToolTip("The CW identification that will be sent")
        t.set_role(self._morse_label, "lcd-small")
        callsign_form.addRow(self._morse_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self._test_btn = QPushButton("Preview ID")
        self._test_btn.clicked.connect(self._on_test_clicked)
        btn_layout.addWidget(self._test_btn)

        self._id_now_btn = QPushButton("Send ID Now")
        t.set_role(self._id_now_btn, "danger")
        self._id_now_btn.clicked.connect(self._on_id_now_clicked)
        btn_layout.addWidget(self._id_now_btn)
        callsign_form.addRow(btn_layout)

        # Auto-ID countdown, shown only while transmitting.
        countdown_row = QHBoxLayout()
        countdown_row.setSpacing(8)
        self._countdown_label = QLabel("")
        t.set_role(self._countdown_label, "value")
        countdown_row.addWidget(self._countdown_label)
        self._id_progress = QProgressBar()
        self._id_progress.setRange(0, 600)
        self._id_progress.setValue(0)
        self._id_progress.setTextVisible(False)
        self._id_progress.setToolTip("Time left until the next automatic ID")
        countdown_row.addWidget(self._id_progress, 1)
        self._countdown_row = QWidget()
        self._countdown_row.setLayout(countdown_row)
        countdown_row.setContentsMargins(0, 0, 0, 0)
        self._countdown_row.setVisible(False)
        callsign_form.addRow(self._countdown_row)

        layout.addWidget(callsign_group)

        # --- Identification settings -----------------------------------
        settings_group = QGroupBox("ID Settings")
        form = self._make_form(settings_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(
            [
                "CW (Morse)",
                "Voice (not available)",
                "PSK31 (not available)",
                "RTTY (not available)",
            ]
        )
        # Only CW is implemented for transmission; the others stay listed so
        # saved settings round-trip, but cannot be picked.
        model = self._mode_combo.model()
        if isinstance(model, QStandardItemModel):
            for row in range(1, model.rowCount()):
                item = model.item(row)
                if item is not None:
                    item.setEnabled(False)
                    item.setToolTip("Not implemented yet; only CW ID can be sent")
        self._mode_combo.setToolTip("How the ID is sent. Only CW (Morse) is available.")
        self._mode_combo.currentIndexChanged.connect(self._on_settings_changed)
        self._add_row(form, "Mode:", self._mode_combo)

        self._wpm_spin = QSpinBox()
        self._wpm_spin.setRange(5, 50)
        self._wpm_spin.setValue(20)
        self._wpm_spin.setSuffix(" WPM")
        self._wpm_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._wpm_spin.setToolTip("CW sending speed in words per minute")
        self._wpm_spin.valueChanged.connect(self._on_settings_changed)
        self._add_row(form, "CW speed:", self._wpm_spin)

        self._tone_spin = QSpinBox()
        self._tone_spin.setRange(400, 1000)
        self._tone_spin.setSingleStep(10)
        self._tone_spin.setValue(700)
        self._tone_spin.setSuffix(" Hz")
        self._tone_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._tone_spin.setToolTip("CW audio tone (sidetone) frequency")
        self._tone_spin.valueChanged.connect(self._on_settings_changed)
        self._add_row(form, "Tone:", self._tone_spin)

        self._auto_id_check = QCheckBox("Every interval while transmitting")
        self._auto_id_check.setChecked(True)
        self._auto_id_check.setToolTip(
            "Automatically send your ID at the interval below during a transmission"
        )
        self._auto_id_check.stateChanged.connect(self._on_settings_changed)
        self._add_row(form, "Auto-ID:", self._auto_id_check)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 10)
        self._interval_spin.setValue(10)
        self._interval_spin.setSuffix(" min")
        self._interval_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._interval_spin.setToolTip(_INTERVAL_TIP)
        self._interval_spin.valueChanged.connect(self._on_settings_changed)
        self._add_row(form, "Interval:", self._interval_spin)

        self._id_start_check = QCheckBox("At start of transmission")
        self._id_start_check.setChecked(True)
        self._id_start_check.setToolTip("Send your ID when a transmission begins")
        self._id_start_check.stateChanged.connect(self._on_settings_changed)
        self._add_row(form, "Also ID:", self._id_start_check)

        self._id_end_check = QCheckBox("At end of transmission")
        self._id_end_check.setChecked(True)
        self._id_end_check.setToolTip(
            "Send your ID when a transmission ends (required by FCC rules)"
        )
        self._id_end_check.stateChanged.connect(self._on_settings_changed)
        form.addRow("", self._id_end_check)

        note = QLabel(
            "Sending an ID transmits on the current frequency and needs a "
            "TX-capable device (HackRF One)."
        )
        note.setWordWrap(True)
        t.set_role(note, "hint")
        form.addRow(note)

        layout.addWidget(settings_group)
        layout.addStretch(1)

        self._align_form_labels()

    @staticmethod
    def _make_form(group: "QGroupBox") -> "QFormLayout":
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        return form

    def _add_row(self, form: "QFormLayout", text: str, field: "QWidget") -> "QLabel":
        label = QLabel(text)
        label.setBuddy(field)
        form.addRow(label, field)
        self._form_labels.append((label, field))
        return label

    def _align_form_labels(self) -> None:
        """Line the labels up with their fields, in both directions.

        Both forms get the same label column width so the fields start at the
        same x. Each label is also made as tall as its field: QFormLayout caps
        a label at 7/4 of its own height, which leaves it a few pixels above
        the centre of a (taller) spin box.
        """
        if not self._form_labels:
            return
        for label, field in self._form_labels:
            label.ensurePolished()
            field.ensurePolished()
        width = max(label.sizeHint().width() for label, _ in self._form_labels)
        for label, field in self._form_labels:
            label.setMinimumWidth(width)
            label.setMinimumHeight(field.sizeHint().height())
            label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

    # ------------------------------------------------------------------
    # Callsign input
    # ------------------------------------------------------------------

    def _on_callsign_changed(self, text: str):
        """Handle callsign input change."""
        normalized = "".join(text.split()).upper()
        if normalized != text:
            # Programmatic text (set_callsign) bypasses the validator.
            pos = self._callsign_input.cursorPosition()
            self._callsign_input.blockSignals(True)
            self._callsign_input.setText(normalized)
            self._callsign_input.setCursorPosition(min(pos, len(normalized)))
            self._callsign_input.blockSignals(False)
        self._callsign = normalized

        self._update_controls()
        self._refresh_status()
        self.callsign_changed.emit(self._callsign)

    def _validate_callsign(self, callsign: str) -> bool:
        """Basic callsign format validation."""
        if not callsign or len(callsign) < 3:
            return False
        has_letter = any(c.isalpha() for c in callsign)
        has_number = any(c.isdigit() for c in callsign)
        valid_chars = all(c.isalnum() or c == "/" for c in callsign)
        return has_letter and has_number and valid_chars

    def _refresh_status(self) -> None:
        """Show validation feedback (or the TX state) under the callsign."""
        cs = self._callsign
        if self._is_transmitting:
            self._set_status(f"Transmitting ID: DE {cs}", "danger")
        elif not cs:
            self._set_status("Enter your callsign to enable station ID.", None)
        elif len(cs) < 3:
            self._set_status("Callsigns have at least 3 characters.", None)
        elif not self._validate_callsign(cs):
            self._set_status(
                "⚠ Unusual format: a callsign has letters and at least one "
                "digit, e.g. W1AW or VE3ABC.",
                "warning",
            )
        else:
            wpm = self._wpm_spin.value()
            seconds = morse_duration(f"DE {cs}", wpm)
            # Non-breaking spaces keep "5.8 s" and "20 WPM" together on wrap.
            self._set_status(
                f"✓ Ready to identify as {cs} · {seconds:.1f}\u00a0s at "
                f"{wpm}\u00a0WPM",
                "success",
            )
        self._refresh_morse()

    def _set_status(self, text: str, tone: Optional[str]) -> None:
        self._status_label.setText(text)
        _theme().set_tone(self._status_label, tone)

    def _refresh_morse(self) -> None:
        """Show the Morse that the ID will send.

        Only the dots and dashes: the ready line above carries the duration,
        so no digits or brackets mix into the code.
        """
        if len(self._callsign) < 3:
            self._morse_label.setVisible(False)
            return
        text = f"DE {self._callsign}"
        wpm = self._wpm_spin.value()
        seconds = morse_duration(text, wpm)
        self._morse_label.setText(morse_pattern(text))
        self._morse_label.setToolTip(
            f"“{text}” in Morse: the CW ID that will be sent "
            f"({seconds:.1f} s at {wpm} WPM, {self._tone_spin.value()} Hz tone)"
        )
        self._morse_label.setVisible(True)

    def _update_controls(self) -> None:
        """Enable the action buttons and explain why when they are not."""
        cs = self._callsign
        valid = self._validate_callsign(cs)

        can_preview = len(cs) >= 3
        self._test_btn.setEnabled(can_preview or self._preview_playing())
        if self._preview_playing():
            self._test_btn.setToolTip("Stop the ID preview")
        elif can_preview:
            self._test_btn.setToolTip(
                f"Play “DE {cs}” in CW through your speakers at the "
                "speed and tone set below. Nothing is transmitted."
            )
        else:
            self._test_btn.setToolTip("Enter your callsign to preview the ID.")

        if self._is_transmitting:
            enabled, tip = False, "An ID is being transmitted."
        elif not valid:
            enabled = False
            tip = (
                "Enter a valid callsign first (letters and at least one digit, "
                "e.g. W1AW)."
            )
        elif self._tx_available is False:
            enabled = False
            tip = self._tx_unavailable_reason or (
                "Connect a TX-capable device (HackRF One) to send an ID."
            )
        else:
            enabled = True
            tip = (
                f"Transmit “DE {cs}” in CW on the current frequency "
                "now. Needs a TX-capable device (HackRF One) and a frequency "
                "your license allows."
            )
        self._id_now_btn.setEnabled(enabled)
        self._id_now_btn.setToolTip(tip)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _on_settings_changed(self):
        """Handle settings change."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

        # CW speed/tone only matter for CW; the interval only for auto-ID.
        is_cw = self._mode_combo.currentIndex() == _CW_INDEX
        self._wpm_spin.setEnabled(is_cw)
        self._tone_spin.setEnabled(is_cw)
        auto = self._auto_id_check.isChecked()
        self._interval_spin.setEnabled(auto)
        self._interval_spin.setToolTip(
            _INTERVAL_TIP if auto else "Turn on Auto-ID to set the interval."
        )
        if self._preview_playing():
            # Keep the "Playing preview" message; only the Morse can change.
            self._refresh_morse()
        else:
            # The ready line shows the ID duration, which depends on the speed.
            self._refresh_status()

        if self._is_transmitting:
            if self._auto_id_check.isChecked() and not self._id_timer.isActive():
                self._reset_countdown()
                self._id_timer.start(1000)
            elif not self._auto_id_check.isChecked():
                self._id_timer.stop()
            self._countdown_row.setVisible(self._auto_id_check.isChecked())

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_id_now_clicked(self):
        """Handle manual ID request."""
        if self._callsign:
            self.id_requested.emit()
            self._reset_countdown()

    def _on_test_clicked(self):
        """Play (or stop) an audible preview of the CW ID."""
        if self._preview_playing():
            self._stop_preview()
            self._refresh_status()
            self._update_controls()
            return
        if not self._callsign:
            return
        try:
            from ..callsign import generate_cw_id

            audio = generate_cw_id(
                self._callsign,
                wpm=self._wpm_spin.value(),
                frequency=self._tone_spin.value(),
                sample_rate=_PREVIEW_RATE,
            )
        except Exception as e:
            logger.error(f"Error generating test ID: {e}")
            self._set_status(f"Could not generate the ID: {e}", "danger")
            return

        seconds = len(audio) / float(_PREVIEW_RATE)
        logger.info(f"Test ID generated: {len(audio)} samples")
        self._refresh_morse()
        if self._play_preview(audio):
            self._test_btn.setText("Stop Preview")
            self._set_status(
                f"Playing preview: DE {self._callsign} ({seconds:.1f} s)", "info"
            )
        else:
            self._set_status(
                "No audio output available; the Morse that would be sent is "
                "shown below.",
                "warning",
            )
        self._update_controls()

    def _play_preview(self, audio: np.ndarray) -> bool:
        """Play mono float ``audio`` once on the default output device."""
        try:
            from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
            from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
        except ImportError:
            return False
        try:
            device = QMediaDevices.defaultAudioOutput()
            if device.isNull():
                return False
            fmt = QAudioFormat()
            fmt.setSampleRate(_PREVIEW_RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if not device.isFormatSupported(fmt):
                return False
            pcm = (np.clip(audio, -1.0, 1.0) * 0.5 * 32767.0).astype(np.int16)
            self._stop_preview()
            buffer = QBuffer(self)
            buffer.setData(QByteArray(pcm.tobytes()))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            sink = QAudioSink(device, fmt, self)
            sink.stateChanged.connect(self._on_preview_state_changed)
            self._preview_sink, self._preview_buffer = sink, buffer
            sink.start(buffer)
            return True
        except Exception as e:  # pragma: no cover - backend dependent
            logger.debug(f"CW preview playback failed: {e}")
            self._stop_preview()
            return False

    def _preview_playing(self) -> bool:
        return self._preview_sink is not None

    def _on_preview_state_changed(self, state: Any) -> None:
        try:
            from PyQt6.QtMultimedia import QAudio
        except ImportError:  # pragma: no cover
            return
        if state in (QAudio.State.IdleState, QAudio.State.StoppedState):
            self._stop_preview()
            self._refresh_status()
            self._update_controls()

    def _stop_preview(self) -> None:
        sink, buffer = self._preview_sink, self._preview_buffer
        self._preview_sink = self._preview_buffer = None
        if sink is not None:
            try:
                sink.stateChanged.disconnect(self._on_preview_state_changed)
            except (TypeError, RuntimeError):
                pass
            sink.stop()
            sink.deleteLater()
        if buffer is not None:
            buffer.close()
            buffer.deleteLater()
        self._test_btn.setText("Preview ID")

    # ------------------------------------------------------------------
    # Transmission state
    # ------------------------------------------------------------------

    def set_transmitting(self, is_transmitting: bool):
        """Set transmission state."""
        self._is_transmitting = bool(is_transmitting)
        auto = self._auto_id_check.isChecked()
        self._countdown_row.setVisible(self._is_transmitting and auto)

        if self._is_transmitting:
            if auto:
                self._reset_countdown()
                self._id_timer.start(1000)  # Update every second
        else:
            self._id_timer.stop()
            self._countdown_label.setText("")
        self._refresh_status()
        self._update_controls()

    def set_tx_available(self, available: bool, reason: str = "") -> None:
        """Tell the panel whether the connected device can transmit.

        When ``available`` is False, "Send ID Now" is disabled and its tooltip
        shows ``reason``.
        """
        self._tx_available = bool(available)
        self._tx_unavailable_reason = reason
        self._update_controls()

    def _reset_countdown(self):
        """Reset the ID countdown."""
        self._seconds_until_id = self._interval_spin.value() * 60
        self._id_progress.setMaximum(self._seconds_until_id)
        self._id_progress.setValue(self._seconds_until_id)
        self._update_countdown_display()

    def _update_countdown(self):
        """Update the countdown timer."""
        if self._seconds_until_id > 0:
            self._seconds_until_id -= 1
            self._id_progress.setValue(self._seconds_until_id)
            self._update_countdown_display()

            if self._seconds_until_id == 0 and self._auto_id_check.isChecked():
                # Time for ID
                self.id_requested.emit()
                self._reset_countdown()

    def _update_countdown_display(self):
        """Update the countdown label."""
        minutes = self._seconds_until_id // 60
        seconds = self._seconds_until_id % 60
        self._countdown_label.setText(f"Next ID in {minutes:02d}:{seconds:02d}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_callsign(self) -> str:
        """Get the current callsign."""
        return self._callsign

    def set_callsign(self, callsign: str):
        """Set the callsign."""
        self._callsign_input.setText(callsign)

    def get_settings(self) -> dict:
        """Get all settings as a dictionary."""
        mode_map = {0: "CW", 1: "VOICE", 2: "PSK31", 3: "RTTY"}
        return {
            "callsign": self._callsign,
            "auto_id": self._auto_id_check.isChecked(),
            "id_at_start": self._id_start_check.isChecked(),
            "id_at_end": self._id_end_check.isChecked(),
            "mode": mode_map.get(self._mode_combo.currentIndex(), "CW"),
            "interval_minutes": self._interval_spin.value(),
            "cw_wpm": self._wpm_spin.value(),
            "cw_tone": self._tone_spin.value(),
        }

    def set_settings(self, settings: dict):
        """Apply settings from a dictionary."""
        if "callsign" in settings:
            self.set_callsign(settings["callsign"])
        if "auto_id" in settings:
            self._auto_id_check.setChecked(settings["auto_id"])
        if "id_at_start" in settings:
            self._id_start_check.setChecked(settings["id_at_start"])
        if "id_at_end" in settings:
            self._id_end_check.setChecked(settings["id_at_end"])
        if "mode" in settings:
            # Only CW can be sent, and the other entries cannot be picked in
            # the combo, so a saved non-CW mode falls back to CW instead of
            # selecting an entry the user could not have chosen.
            if str(settings["mode"]).upper() != "CW":
                logger.info(f"ID mode {settings['mode']!r} is not available; using CW")
            self._mode_combo.setCurrentIndex(_CW_INDEX)
        if "interval_minutes" in settings:
            self._interval_spin.setValue(settings["interval_minutes"])
        if "cw_wpm" in settings:
            self._wpm_spin.setValue(settings["cw_wpm"])
        if "cw_tone" in settings:
            self._tone_spin.setValue(settings["cw_tone"])
