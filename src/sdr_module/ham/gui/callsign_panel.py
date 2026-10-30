"""
Callsign identification panel for HAM radio compliance.

Provides UI controls for:
- Callsign input (uppercased and validated as you type)
- What sending the ID would do right now (or why it can't), from the window
- Audible preview of the CW ID and "Send ID Now" (Stop while on the air),
  with progress
- How the ID is sent (CW keyed carrier or FM tone), its speed and tone
- The HackRF's transmit power (TX gain and the +14 dB amplifier) with an
  estimate of the output
- Automatic ID for the app's other transmissions (start, interval, end)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

try:
    from PyQt6.QtCore import Qt, pyqtSignal
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
        QSlider,
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
# How the ID goes on air: (combo text, saved value, tooltip).
_EMISSIONS = (
    (
        "Auto (by frequency)",
        "auto",
        "CW (a keyed carrier) below 30 MHz, where HF stations listen for it; "
        "an FM tone above, which FM receivers hear",
    ),
    (
        "CW (keyed carrier)",
        "cw",
        "The carrier keyed on and off (A1A): heard in CW or SSB, not in FM",
    ),
    (
        "FM tone (MCW)",
        "mcw",
        "The Morse as a keyed audio tone on an FM carrier (F2A): heard on FM "
        "receivers and repeaters",
    ),
)
_TX_GAIN_MAX = 47
_AUTO_ID_NOTE = (
    "Automatic ID adds your ID to the app's other transmissions (such as "
    "Transmit Text) at their start, every interval and at their end. Send "
    "ID Now sends just the ID."
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


def _clock(seconds: float) -> str:
    """``75 -> "01:15"``."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


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

    The panel shows and edits; the window decides. It tells the panel what
    sending would do (:meth:`set_tx_status`), runs the transmission and
    reports it (:meth:`set_transmitting`, :meth:`set_tx_progress`,
    :meth:`set_id_countdown`), and shows the transmit power estimate
    (:meth:`set_power_estimate`).
    """

    if HAS_PYQT6:
        callsign_changed = pyqtSignal(str)
        id_requested = pyqtSignal()  # Send ID Now
        stop_requested = pyqtSignal()  # Stop (while on the air)
        settings_changed = pyqtSignal(dict)
        tx_power_changed = pyqtSignal(int, bool)  # TX gain dB, amplifier on

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")

        super().__init__(parent)

        self._callsign = ""
        self._is_transmitting = False
        # None = unknown (the window hasn't said): Send stays available and
        # the window checks when it is pressed.
        self._tx_available: Optional[bool] = None
        self._tx_unavailable_reason = ""
        self._tx_text = ""
        self._tx_tone: Optional[str] = None
        self._sending_text = ""
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

        # Line 1: the callsign and what the ID is.
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        t.set_role(self._status_label, "hint")
        callsign_form.addRow(self._status_label)

        # What will be sent, in Morse.
        self._morse_label = QLabel()
        self._morse_label.setWordWrap(True)
        self._morse_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._morse_label.setToolTip("The CW identification that will be sent")
        t.set_role(self._morse_label, "lcd-small")
        callsign_form.addRow(self._morse_label)

        # Line 2: what sending would do right now, or why it can't.
        self._tx_label = QLabel()
        self._tx_label.setWordWrap(True)
        self._tx_label.setAccessibleName("Transmit status")
        t.set_role(self._tx_label, "hint")
        self._tx_label.setVisible(False)
        callsign_form.addRow(self._tx_label)

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

        # Progress of the transmission on the air.
        self._tx_progress = QProgressBar()
        self._tx_progress.setRange(0, 1000)
        self._tx_progress.setTextVisible(True)
        self._tx_progress.setFormat("")
        self._tx_progress.setAccessibleName("Transmission progress")
        self._tx_progress.setVisible(False)
        callsign_form.addRow(self._tx_progress)

        # Automatic ID countdown, while a transmission is due one.
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

        self._emission_combo = QComboBox()
        for index, (text, value, tip) in enumerate(_EMISSIONS):
            self._emission_combo.addItem(text, value)
            self._emission_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        self._emission_combo.setToolTip(
            "How the Morse goes on air. Auto: CW (a keyed carrier) below 30 "
            "MHz, an FM tone (MCW) above."
        )
        self._emission_combo.currentIndexChanged.connect(self._on_settings_changed)
        self._add_row(form, "Sent as:", self._emission_combo)

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
        self._tone_spin.setToolTip(
            "The audio tone of the FM tone ID (and of the preview). A CW ID "
            "has no tone of its own: the listener's receiver makes it."
        )
        self._tone_spin.valueChanged.connect(self._on_settings_changed)
        self._add_row(form, "Tone:", self._tone_spin)

        layout.addWidget(settings_group)

        # --- Transmit power ---------------------------------------------
        power_group = QGroupBox("TX Power (HackRF)")
        power_form = self._make_form(power_group)

        gain_row = QHBoxLayout()
        gain_row.setSpacing(8)
        self._tx_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._tx_gain_slider.setRange(0, _TX_GAIN_MAX)
        self._tx_gain_slider.setPageStep(5)
        self._tx_gain_slider.setValue(20)
        self._tx_gain_slider.setAccessibleName("TX gain")
        self._tx_gain_slider.setToolTip(
            "The HackRF's transmit (VGA) gain, 0-47 dB: each dB raises the "
            "output by 1 dB"
        )
        gain_row.addWidget(self._tx_gain_slider, 1)
        self._tx_gain_spin = QSpinBox()
        self._tx_gain_spin.setRange(0, _TX_GAIN_MAX)
        self._tx_gain_spin.setValue(20)
        self._tx_gain_spin.setSuffix(" dB")
        self._tx_gain_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._tx_gain_spin.setToolTip(self._tx_gain_slider.toolTip())
        gain_row.addWidget(self._tx_gain_spin)
        self._tx_gain_slider.valueChanged.connect(self._tx_gain_spin.setValue)
        self._tx_gain_spin.valueChanged.connect(self._tx_gain_slider.setValue)
        self._tx_gain_spin.valueChanged.connect(self._on_tx_power_changed)
        gain_field = QWidget()
        gain_field.setLayout(gain_row)
        gain_row.setContentsMargins(0, 0, 0, 0)
        label = self._add_row(power_form, "TX gain:", gain_field)
        label.setBuddy(self._tx_gain_spin)

        self._amp_check = QCheckBox("+14 dB amplifier")
        self._amp_check.setToolTip(
            "The HackRF's front-end amplifier: about 14 dB more output. Leave "
            "it off unless you need the power."
        )
        self._amp_check.toggled.connect(self._on_tx_power_changed)
        self._add_row(power_form, "RF amp:", self._amp_check)

        self._power_label = QLabel("—")
        self._power_label.setWordWrap(True)
        self._power_label.setAccessibleName("Estimated output")
        t.set_role(self._power_label, "value")
        self._power_label.setToolTip(
            "Estimated output at the HackRF's antenna port at the tuned "
            "frequency. Units differ by a few dB: measure it before adding "
            "an amplifier."
        )
        self._add_row(power_form, "Output:", self._power_label)

        layout.addWidget(power_group)

        # --- Automatic ID ------------------------------------------------
        auto_group = QGroupBox("Automatic ID")
        auto_form = self._make_form(auto_group)

        self._auto_id_check = QCheckBox("Every interval while transmitting")
        self._auto_id_check.setChecked(True)
        self._auto_id_check.setToolTip(
            "Send your ID at the interval below during a long transmission"
        )
        self._auto_id_check.stateChanged.connect(self._on_settings_changed)
        self._add_row(auto_form, "Auto-ID:", self._auto_id_check)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 10)
        self._interval_spin.setValue(10)
        self._interval_spin.setSuffix(" min")
        self._interval_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._interval_spin.setToolTip(_INTERVAL_TIP)
        self._interval_spin.valueChanged.connect(self._on_settings_changed)
        self._add_row(auto_form, "Interval:", self._interval_spin)

        self._id_start_check = QCheckBox("At start of transmission")
        self._id_start_check.setChecked(True)
        self._id_start_check.setToolTip("Send your ID when a transmission begins")
        self._id_start_check.stateChanged.connect(self._on_settings_changed)
        self._add_row(auto_form, "Also ID:", self._id_start_check)

        self._id_end_check = QCheckBox("At end of transmission")
        self._id_end_check.setChecked(True)
        self._id_end_check.setToolTip(
            "Send your ID when a transmission ends (required by FCC rules)"
        )
        self._id_end_check.stateChanged.connect(self._on_settings_changed)
        auto_form.addRow("", self._id_end_check)

        self._auto_note = QLabel(_AUTO_ID_NOTE)
        self._auto_note.setWordWrap(True)
        t.set_role(self._auto_note, "hint")
        auto_form.addRow(self._auto_note)

        layout.addWidget(auto_group)
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

        Every form gets the same label column width so the fields start at
        the same x. Each label is also made as tall as its field: QFormLayout
        caps a label at 7/4 of its own height, which leaves it a few pixels
        above the centre of a (taller) spin box.
        """
        if not self._form_labels:
            return
        for label, field in self._form_labels:
            label.ensurePolished()
            field.ensurePolished()
        width = max(label.sizeHint().width() for label, _ in self._form_labels)
        for label, field in self._form_labels:
            label.setMinimumWidth(width)
            if field is not self._power_label:  # (a wrapping value grows)
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

    def is_callsign_valid(self) -> bool:
        """True when the callsign looks like one (letters and a digit)."""
        return self._validate_callsign(self._callsign)

    def _refresh_status(self) -> None:
        """Line 1: the callsign and the ID it makes (or the TX state)."""
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
            # Non-breaking spaces keep "4.0 s" and "20 WPM" together on wrap.
            self._set_status(
                f"✓ Callsign OK · DE {cs} = {seconds:.1f} s at " f"{wpm} WPM",
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

    def _refresh_tx_line(self) -> None:
        """Line 2: what sending does now (from the window), toned."""
        if self._is_transmitting and self._sending_text:
            text, tone = self._sending_text, "danger"
        else:
            text, tone = self._tx_text, self._tx_tone
        self._tx_label.setText(text)
        _theme().set_tone(self._tx_label, tone)
        self._tx_label.setVisible(bool(text))

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
            self._id_now_btn.setText("Stop")
            enabled, tip = True, "Stop transmitting now"
        else:
            self._id_now_btn.setText("Send ID Now")
            if not valid:
                enabled = False
                tip = (
                    "Enter a valid callsign first (letters and at least one "
                    "digit, e.g. W1AW)."
                )
            elif self._tx_available is False:
                enabled = False
                tip = self._tx_unavailable_reason or (
                    "Connect a TX-capable device (HackRF One) to send an ID."
                )
            else:
                enabled = True
                tip = (
                    f"Transmit “DE {cs}” on the tuned frequency now. Needs a "
                    "TX-capable device (HackRF One) and a frequency your "
                    "license allows."
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

    def _on_tx_power_changed(self, *_args) -> None:
        self.tx_power_changed.emit(
            self._tx_gain_spin.value(), self._amp_check.isChecked()
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_id_now_clicked(self):
        """Send ID Now, or Stop while transmitting."""
        if self._is_transmitting:
            self.stop_requested.emit()
            return
        if self._callsign:
            self.id_requested.emit()

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
    # Transmission state (driven by the window's transmitter)
    # ------------------------------------------------------------------

    def set_transmitting(self, is_transmitting: bool, text: str = "") -> None:
        """Show that a transmission is on the air (``text``: what, e.g.
        "Sending DE W1AW on 146.520 MHz…"): Send ID Now becomes Stop and the
        progress bar shows. The automatic-ID countdown is the window's to
        show (:meth:`set_id_countdown`); this never starts one."""
        self._is_transmitting = bool(is_transmitting)
        self._sending_text = text if self._is_transmitting else ""
        self._tx_progress.setVisible(self._is_transmitting)
        if self._is_transmitting:
            self._tx_progress.setValue(0)
            self._tx_progress.setFormat("Starting…")
        else:
            self.set_id_countdown(None)
        self._refresh_status()
        self._refresh_tx_line()
        self._update_controls()

    def is_transmitting(self) -> bool:
        return self._is_transmitting

    def set_tx_progress(self, done_s: float, total_s: float) -> None:
        """Air time sent so far and in all, while transmitting."""
        total = max(float(total_s), 1e-6)
        done = min(max(float(done_s), 0.0), total)
        self._tx_progress.setValue(int(round(1000 * done / total)))
        self._tx_progress.setFormat(f"{done:.1f} / {total:.1f} s")

    def set_id_countdown(
        self, seconds: Optional[float], interval_s: Optional[float] = None
    ) -> None:
        """Show "Next ID in mm:ss" (``seconds`` until the next automatic
        ID of the transmission on the air), or hide it (None)."""
        if seconds is None:
            self._countdown_row.setVisible(False)
            self._countdown_label.setText("")
            return
        span = int(
            round(interval_s if interval_s else self._interval_spin.value() * 60)
        )
        self._id_progress.setMaximum(max(1, span))
        self._id_progress.setValue(int(round(min(max(seconds, 0.0), span))))
        self._countdown_label.setText(f"Next ID in {_clock(seconds)}")
        self._countdown_row.setVisible(True)

    def set_tx_available(self, available: bool, reason: str = "") -> None:
        """Tell the panel whether the connected device can transmit.

        When ``available`` is False, "Send ID Now" is disabled and the
        reason is shown under the callsign (and as its tooltip).
        """
        self._tx_available = bool(available)
        self._tx_unavailable_reason = reason
        if available:
            self._tx_text, self._tx_tone = "", None
        else:
            self._tx_text, self._tx_tone = reason, "warning"
        self._refresh_tx_line()
        self._update_controls()

    def set_tx_status(
        self, text: str, tone: Optional[str], can_send: Optional[bool]
    ) -> None:
        """Line 2: what Send ID Now would do now ("Will send on 146.520 MHz
        · FM tone ID · ...") or why it can't ("Can't send: ..."), toned;
        ``can_send`` False disables Send ID Now with ``text`` as the reason
        (None: unknown, Send stays available)."""
        self._tx_available = can_send
        self._tx_unavailable_reason = text if can_send is False else ""
        self._tx_text, self._tx_tone = text, tone
        self._refresh_tx_line()
        self._update_controls()

    def tx_status_text(self) -> str:
        """Line 2 as shown (whether or not the tab is on show)."""
        return "" if self._tx_label.isHidden() else self._tx_label.text()

    # ------------------------------------------------------------------
    # Transmit power
    # ------------------------------------------------------------------

    def tx_power(self) -> tuple:
        """``(TX gain dB, amplifier on)``."""
        return self._tx_gain_spin.value(), self._amp_check.isChecked()

    def set_tx_power(self, gain_db: int, amp: bool) -> None:
        """Show these TX power settings (without emitting)."""
        for widget in (self._tx_gain_spin, self._tx_gain_slider, self._amp_check):
            widget.blockSignals(True)
        try:
            gain = int(min(max(int(gain_db), 0), _TX_GAIN_MAX))
            self._tx_gain_spin.setValue(gain)
            self._tx_gain_slider.setValue(gain)
            self._amp_check.setChecked(bool(amp))
        finally:
            for widget in (self._tx_gain_spin, self._tx_gain_slider, self._amp_check):
                widget.blockSignals(False)

    def set_power_estimate(
        self, text: str, tone: Optional[str] = None, tip: str = ""
    ) -> None:
        """The estimated output ("about 1 mW (0 dBm) at 146.520 MHz"), red
        (``tone`` "danger") when it is over a limit; ``tip`` says why."""
        self._power_label.setText(text or "—")
        _theme().set_tone(self._power_label, tone)
        if tip:
            self._power_label.setToolTip(tip)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_callsign(self) -> str:
        """Get the current callsign."""
        return self._callsign

    def set_callsign(self, callsign: str):
        """Set the callsign."""
        self._callsign_input.setText(callsign)

    def emission_preference(self) -> str:
        """How the ID is sent: "auto", "cw" or "mcw"."""
        return str(self._emission_combo.currentData() or "auto")

    def cw_wpm(self) -> int:
        return self._wpm_spin.value()

    def cw_tone(self) -> int:
        return self._tone_spin.value()

    def auto_id_settings(self) -> dict:
        """``{"at_start", "at_end", "interval_s"}`` (interval None: off)."""
        return {
            "at_start": self._id_start_check.isChecked(),
            "at_end": self._id_end_check.isChecked(),
            "interval_s": (
                self._interval_spin.value() * 60.0
                if self._auto_id_check.isChecked()
                else None
            ),
        }

    def get_settings(self) -> dict:
        """Get all settings as a dictionary."""
        mode_map = {0: "CW", 1: "VOICE", 2: "PSK31", 3: "RTTY"}
        return {
            "callsign": self._callsign,
            "auto_id": self._auto_id_check.isChecked(),
            "id_at_start": self._id_start_check.isChecked(),
            "id_at_end": self._id_end_check.isChecked(),
            "mode": mode_map.get(self._mode_combo.currentIndex(), "CW"),
            "emission": self.emission_preference(),
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
        if "emission" in settings:
            index = self._emission_combo.findData(str(settings["emission"]).lower())
            if index >= 0:
                self._emission_combo.setCurrentIndex(index)
        if "interval_minutes" in settings:
            self._interval_spin.setValue(settings["interval_minutes"])
        if "cw_wpm" in settings:
            self._wpm_spin.setValue(settings["cw_wpm"])
        if "cw_tone" in settings:
            self._tone_spin.setValue(settings["cw_tone"])
