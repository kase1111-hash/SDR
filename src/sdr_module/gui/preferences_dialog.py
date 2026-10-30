"""
Tools > Preferences (Ctrl+,): the few settings that aren't part of receiving.

- **Audio output**: which sound card plays the audio (Refresh lists one
  plugged in meanwhile); the toolbar's Audio button says which one plays.
- **Folders**: where recordings (audio recordings and automatically saved
  I/Q takes) and screenshots go by default (through :mod:`.file_dialogs`).
- **Recording temp folder**: where an I/Q recording is written while it
  records, until it is saved, with the space free there (a recording takes
  over a gigabyte a minute at 2.4 MS/s).
- **Transmitting**: ask before transmitting (the transmit engine's
  first-use confirmation, ``tx/confirm``).

The dialog only reads the settings; :meth:`PreferencesDialog.apply` writes
them when the user presses OK, and the main window then applies them.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

from . import file_dialogs
from .audio_sink import default_output_name, output_devices

if HAS_PYQT6:
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

#: Settings keys (prefs/ prefix; the confirmation is the transmitter's).
KEY_AUDIO_OUTPUT = "prefs/audio_output"
KEY_AUDIO_OUTPUT_NAME = "prefs/audio_output_name"
KEY_CAPTURE_FOLDER = "prefs/capture_folder"
KEY_TX_CONFIRM = "tx/confirm"

#: Bytes one I/Q sample takes while recording (complex64).
_BYTES_PER_SAMPLE = 8


def format_bytes(count: float) -> str:
    """``1.234e9 -> "1.2 GB"``."""
    value = float(count)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"  # pragma: no cover - unreachable


def free_space(folder: str) -> Optional[int]:
    """Bytes free on the drive ``folder`` is on (None if unknown)."""
    probe = folder
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return int(shutil.disk_usage(probe or folder).free)
    except OSError:
        return None


def usable_folder(folder: str) -> bool:
    """True if recordings can be written in ``folder``."""
    return bool(folder) and os.path.isdir(folder) and os.access(folder, os.W_OK)


def preferred_capture_folder(settings: Any) -> str:
    """The recording temp folder chosen in Preferences, if it can be used
    ("" for the default, or when the chosen one is gone: logged)."""
    try:
        folder = str(settings.get_str(KEY_CAPTURE_FOLDER, "") or "")
    except Exception:  # pragma: no cover - defensive
        return ""
    if folder and not usable_folder(folder):
        logger.warning(
            f"The recording temp folder {folder} can't be written to; "
            "recordings use the default one"
        )
        return ""
    return folder


@dataclass
class Preferences:
    """What the dialog holds (and :meth:`PreferencesDialog.apply` saves)."""

    audio_output: str = ""  # "" = the system default
    audio_output_name: str = ""
    recordings_folder: str = ""
    screenshots_folder: str = ""
    capture_folder: str = ""  # "" = the default
    tx_confirm: bool = True


class PreferencesDialog(QDialog if HAS_PYQT6 else object):
    """Tools > Preferences (see the module notes)."""

    def __init__(
        self,
        settings: Any,
        parent: Any = None,
        capture_default: str = "",
        sample_rate: float = 2.4e6,
    ):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._settings = settings
        self._capture_default = capture_default
        self._rate = float(sample_rate) if sample_rate > 0 else 2.4e6
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(520)
        self._initial = self._load()
        self._recordings = self._initial.recordings_folder
        self._screenshots = self._initial.screenshots_folder
        self._capture = self._initial.capture_folder

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self._build_audio())
        layout.addWidget(self._build_folders())
        layout.addWidget(self._build_capture())
        layout.addWidget(self._build_transmit())
        layout.addStretch(1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        set_role(ok, "primary")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._show_folders()

    # ------------------------------------------------------------------ load

    def _load(self) -> Preferences:
        s = self._settings
        return Preferences(
            audio_output=str(s.get_str(KEY_AUDIO_OUTPUT, "") or ""),
            audio_output_name=str(s.get_str(KEY_AUDIO_OUTPUT_NAME, "") or ""),
            recordings_folder=file_dialogs.recordings_folder(s),
            screenshots_folder=file_dialogs.start_folder(s, "screenshot"),
            capture_folder=str(s.get_str(KEY_CAPTURE_FOLDER, "") or ""),
            tx_confirm=bool(s.get_bool(KEY_TX_CONFIRM, True)),
        )

    # ----------------------------------------------------------------- build

    @staticmethod
    def _form(group: "QGroupBox") -> "QFormLayout":
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        return form

    @staticmethod
    def _hint(text: str = "") -> "QLabel":
        label = QLabel(text)
        label.setWordWrap(True)
        set_role(label, "hint")
        return label

    def _build_audio(self) -> "QGroupBox":
        group = QGroupBox("Audio Output")
        form = self._form(group)
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        self._output_combo = QComboBox()
        self._output_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._output_combo.setMinimumContentsLength(16)
        self._output_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._output_combo.setAccessibleName("Audio output device")
        self._output_combo.setToolTip(
            "The sound card (or headset, HDMI output, virtual cable...) the "
            "demodulated audio plays through. The toolbar's Audio button says "
            "which one plays."
        )
        box.addWidget(self._output_combo, 1)
        self._refresh_btn = QPushButton("&Refresh")
        self._refresh_btn.setToolTip(
            "List the audio outputs again (one plugged in now)"
        )
        self._refresh_btn.setAutoDefault(False)
        self._refresh_btn.clicked.connect(lambda: self._fill_outputs(keep=True))
        box.addWidget(self._refresh_btn)
        label = QLabel("&Output:")
        label.setBuddy(self._output_combo)
        form.addRow(label, row)
        self._output_hint = self._hint()
        form.addRow(self._output_hint)
        self._fill_outputs(keep=False)
        return group

    def _fill_outputs(self, keep: bool) -> None:
        """List the outputs; keep the one chosen (or the saved one)."""
        combo = self._output_combo
        wanted = combo.currentData() if keep and combo.count() else None
        if wanted is None:
            wanted = self._initial.audio_output
        devices = output_devices()
        default = default_output_name()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(
            f"System default ({default})" if default else "System default", ""
        )
        for device_id, description in devices:
            combo.addItem(description, device_id)
        index = combo.findData(wanted) if wanted else 0
        if index < 0:
            # Chosen before but not plugged in now: keep it, say so.
            name = self._initial.audio_output_name or "The chosen output"
            combo.addItem(f"{name} (not connected)", wanted)
            index = combo.count() - 1
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        if not devices:
            self._output_hint.setText(
                "No audio output was found. Plug in speakers or headphones, then "
                "press Refresh."
            )
            set_tone(self._output_hint, "warning")
        else:
            count = len(devices)
            self._output_hint.setText(
                f"{count} output{'s' if count != 1 else ''} found. The system "
                "default follows your operating system's sound settings."
            )
            set_tone(self._output_hint, None)

    def _folder_row(self, form: "QFormLayout", text: str, slot: Any) -> "QLabel":
        """A row with a folder (elided path) and a Change... button."""
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        path = QLabel()
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        set_role(path, "value")
        box.addWidget(path, 1)
        change = QPushButton("Change...")
        change.setAutoDefault(False)
        change.clicked.connect(slot)
        box.addWidget(change)
        label = QLabel(text)
        label.setBuddy(change)
        form.addRow(label, row)
        path.setAccessibleName(text.replace("&", "").rstrip(":"))
        change.setAccessibleName(f"Change {text.replace('&', '').rstrip(':').lower()}")
        return path

    def _build_folders(self) -> "QGroupBox":
        group = QGroupBox("Folders")
        form = self._form(group)
        self._recordings_label = self._folder_row(
            form, "&Recordings:", self._choose_recordings
        )
        self._recordings_label.setToolTip(
            "Where audio recordings and automatically saved I/Q recordings go "
            "(also the Recording group's Folder)"
        )
        self._screenshots_label = self._folder_row(
            form, "&Screenshots:", self._choose_screenshots
        )
        self._screenshots_label.setToolTip(
            "Where File > Save Screenshot... starts (it then remembers where you "
            "saved the last one)"
        )
        return group

    def _build_capture(self) -> "QGroupBox":
        group = QGroupBox("Recording Temp Folder")
        form = self._form(group)
        self._capture_label = self._folder_row(form, "&Folder:", self._choose_capture)
        self._capture_label.setToolTip(
            "Where an I/Q recording is written while it records, until you save "
            "it. Pick a drive with room: recordings stop when less than 500 MB "
            "is left."
        )
        self._default_btn = QPushButton("Use &Default")
        self._default_btn.setAutoDefault(False)
        self._default_btn.setToolTip("Go back to the default temp folder")
        self._default_btn.clicked.connect(self._use_default_capture)
        self._space_hint = self._hint()
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(self._space_hint, 1)
        box.addWidget(self._default_btn, 0, Qt.AlignmentFlag.AlignTop)
        form.addRow(row)
        return group

    def _build_transmit(self) -> "QGroupBox":
        group = QGroupBox("Transmitting")
        layout = QVBoxLayout(group)
        self._confirm_check = QCheckBox("&Ask before transmitting")
        self._confirm_check.setChecked(self._initial.tx_confirm)
        self._confirm_check.setToolTip(
            "Before a transmission (Send ID Now, a station ID), ask with what "
            "will be sent, where and at about what power. The license and "
            "band checks apply either way."
        )
        layout.addWidget(self._confirm_check)
        return group

    # --------------------------------------------------------------- folders

    def _show_folders(self) -> None:
        for label, folder in (
            (self._recordings_label, self._recordings),
            (self._screenshots_label, self._screenshots),
        ):
            label.setText(file_dialogs.display_path(folder))
            label.setToolTip(label.toolTip().split("\n")[0] + f"\n{folder}")
        capture = self._capture or self._capture_default
        shown = file_dialogs.display_path(capture)
        if not self._capture:
            shown += " (default)"
        self._capture_label.setText(shown)
        self._default_btn.setEnabled(bool(self._capture))
        free = free_space(capture) if capture else None
        per_minute = self._rate * _BYTES_PER_SAMPLE * 60.0
        text = (
            f"At {self._rate / 1e6:g} MS/s a recording takes "
            f"{format_bytes(per_minute)} a minute."
        )
        tone = None
        if free is not None:
            minutes = free / per_minute if per_minute > 0 else 0.0
            text = f"{format_bytes(free)} free there. " + text
            if minutes < 60:
                text += f" Room for about {max(0, int(minutes))} minutes."
            if free < 2e9:
                tone = "warning"
        if self._capture and not usable_folder(self._capture):
            text = (
                "This folder can't be written to: recordings use the default. " + text
            )
            tone = "danger"
        self._space_hint.setText(text)
        set_tone(self._space_hint, tone)

    def _choose(self, kind: str, caption: str, current: str) -> str:
        return file_dialogs.choose_folder(
            self, self._settings, kind, caption, current, remember=False
        )

    def _choose_recordings(self) -> None:
        folder = self._choose("iq", "Recordings Folder", self._recordings)
        if folder:
            self._recordings = folder
            self._show_folders()

    def _choose_screenshots(self) -> None:
        folder = self._choose("screenshot", "Screenshots Folder", self._screenshots)
        if folder:
            self._screenshots = folder
            self._show_folders()

    def _choose_capture(self) -> None:
        folder = self._choose(
            "iq",
            "Recording Temp Folder",
            self._capture or self._capture_default,
        )
        if folder:
            self._capture = "" if folder == self._capture_default else folder
            self._show_folders()

    def _use_default_capture(self) -> None:
        self._capture = ""
        self._show_folders()

    # ----------------------------------------------------------------- apply

    def preferences(self) -> Preferences:
        """What the dialog shows now."""
        combo = self._output_combo
        output = str(combo.currentData() or "")
        name = combo.currentText() if output else ""
        if name.endswith(" (not connected)"):
            name = name[: -len(" (not connected)")]
        return Preferences(
            audio_output=output,
            audio_output_name=name,
            recordings_folder=self._recordings,
            screenshots_folder=self._screenshots,
            capture_folder=self._capture,
            tx_confirm=bool(self._confirm_check.isChecked()),
        )

    def apply(self) -> Preferences:
        """Save what the dialog shows; returns it (the window applies it)."""
        prefs = self.preferences()
        s = self._settings
        s.set(KEY_AUDIO_OUTPUT, prefs.audio_output)
        s.set(KEY_AUDIO_OUTPUT_NAME, prefs.audio_output_name)
        if prefs.recordings_folder != self._initial.recordings_folder:
            file_dialogs.set_recordings_folder(s, prefs.recordings_folder)
        if prefs.screenshots_folder != self._initial.screenshots_folder:
            file_dialogs.remember_folder(s, "screenshot", prefs.screenshots_folder)
        s.set(KEY_CAPTURE_FOLDER, prefs.capture_folder)
        s.set(KEY_TX_CONFIRM, prefs.tx_confirm)
        return prefs

    def changed(self) -> bool:
        return self.preferences() != self._initial


__all__ = [
    "KEY_AUDIO_OUTPUT",
    "KEY_AUDIO_OUTPUT_NAME",
    "KEY_CAPTURE_FOLDER",
    "KEY_TX_CONFIRM",
    "Preferences",
    "PreferencesDialog",
    "format_bytes",
    "free_space",
    "preferred_capture_folder",
    "usable_folder",
]
