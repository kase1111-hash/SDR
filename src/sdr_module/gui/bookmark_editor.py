"""
Edit Bookmark dialog: everything about one saved channel.

Name, frequency, how it is listened to (mode, FM deviation, filter width,
CTCSS tone or DCS code squelch), its group, whether a bookmark scan skips
it, and a comment. Opened from the Bookmarks panel (Edit..., or the list's right-click
menu).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from ..core.chirp_csv import chirp_mode_for, is_chirp_channel
from ..dsp.dcs import DCS_CODES, dcs_label, format_dcs, parse_dcs
from ..dsp.filters import CTCSS_FREQUENCIES, ctcss_label
from .control_panel import (
    DEMOD_MODES,
    FM_DEVIATIONS,
    MAX_FREQUENCY_HZ,
    MIN_FREQUENCY_HZ,
    bandwidth_options,
)

if HAS_PYQT6:
    from .control_panel import _ValueComboBox
    from .themes import set_role

#: The Mode list's entry that leaves the receiver's mode as it is.
KEEP_MODE_TEXT = "Keep the receiver's mode"
_DEFAULT_WIDTH_TEXT = "The mode's usual width"
#: Repeater shift choices: (CHIRP Duplex, text).
_DUPLEX_CHOICES = (("", "None (simplex)"), ("+", "+ (above)"), ("-", "− (below)"))
_NO_TONE_TEXT = "None (hear every station)"
#: CHIRP's Mode for "no particular mode" (a CHIRP channel that keeps the
#: receiver's mode): it maps to no demodulator here.
_CHIRP_KEEP_MODE = "Auto"
#: CHIRP's default cToneFreq: a TSQL channel saved without its tone uses it.
_CHIRP_DEFAULT_TONE_HZ = 88.5


#: CHIRP Cross modes (transmit->receive) that receive with a CTCSS tone:
#: their receive tone is the cToneFreq too.
_CROSS_RECEIVE_TONE = ("Tone->Tone", "->Tone", "DTCS->Tone")


#: CHIRP Cross modes that receive with a DCS code: their receive code is
#: the RxDtcsCode.
_CROSS_RECEIVE_DCS = ("Tone->DTCS", "->DTCS", "DTCS->DTCS")
#: CHIRP's default DtcsCode, for a DTCS channel saved without one.
_CHIRP_DEFAULT_DCS = 23


def bookmark_tone_hz(bookmark: Dict[str, Any]) -> float:
    """The CTCSS tone (Hz) a bookmark's tone squelch listens for, 0 for
    none: CHIRP's TSQL tone, or a Cross channel's receive tone (its
    cToneFreq, 88.5 Hz when not saved). A DCS code is
    :func:`bookmark_dcs`'s; reverse tone squelch isn't supported: 0."""
    tone_mode = str(bookmark.get("tone_mode") or "")
    cross = str(bookmark.get("cross_mode") or "").replace(" ", "")
    if not (
        tone_mode == "TSQL" or (tone_mode == "Cross" and cross in _CROSS_RECEIVE_TONE)
    ):
        return 0.0
    raw = bookmark.get("ctone")
    try:
        tone = float(_CHIRP_DEFAULT_TONE_HZ if raw is None else raw)
    except (TypeError, ValueError, OverflowError):  # (a damaged setting)
        return 0.0
    return tone if math.isfinite(tone) and tone > 0 else 0.0


def bookmark_dcs(bookmark: Dict[str, Any]) -> Optional[Tuple[int, bool]]:
    """The DCS code a bookmark's squelch listens for, ``(code,
    inverted)``, or None: CHIRP's DTCS channel (its DtcsCode), or a Cross
    channel that receives with a code (its RxDtcsCode), with the receive
    polarity (the second letter of DtcsPolarity: R is inverted). A code
    that isn't one of the standard ones (or a damaged value) gives None.
    Reverse DCS (DTCS-R) isn't supported: None."""
    tone_mode = str(bookmark.get("tone_mode") or "")
    cross = str(bookmark.get("cross_mode") or "").replace(" ", "")
    if tone_mode == "DTCS":
        raw = bookmark.get("dtcs", _CHIRP_DEFAULT_DCS)
    elif tone_mode == "Cross" and cross in _CROSS_RECEIVE_DCS:
        raw = bookmark.get("rx_dtcs", _CHIRP_DEFAULT_DCS)
    else:
        return None
    try:
        number = float(_CHIRP_DEFAULT_DCS if raw is None else raw)
    except (TypeError, ValueError, OverflowError):  # (a damaged setting)
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    polarity = str(bookmark.get("dtcs_polarity") or "NN").strip().upper()
    found = parse_dcs(int(number))
    if found is None:
        return None
    return found[0], polarity[1:2] == "R"


class BookmarkEditDialog(QDialog if HAS_PYQT6 else object):
    """Edit one bookmark; :meth:`bookmark` returns the edited copy."""

    def __init__(
        self,
        bookmark: Dict[str, Any],
        listening: tuple = ("", ""),
        groups: Iterable[str] = (),
        parent=None,
    ):
        """
        Args:
            bookmark: The stored bookmark (not changed; see :meth:`bookmark`).
            listening: ``(demod, fm_deviation)`` it is listened to with now
                (``bookmarks_panel.tune_mode``), "" when it keeps the mode.
            groups: Existing group names, offered in the Group list.
        """
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._original = dict(bookmark)
        self._initial_tone = bookmark_tone_hz(bookmark)
        code = bookmark_dcs(bookmark)
        # The squelch as the Tone list holds it: a CTCSS tone (Hz, 0 for
        # none) or a DCS code ("D023N").
        self._initial_squelch: Any = format_dcs(*code) if code else self._initial_tone
        self.setWindowTitle("Edit Bookmark")
        self.setMinimumWidth(400)
        self._setup_ui(listening, list(groups))
        # How it is listened to as the dialog opens: Save rewrites the mode
        # (and CHIRP's Mode) only when the Mode or the FM deviation changed.
        self._opened_with = self._listening_choice()

    # ------------------------------------------------------------------ #
    def _setup_ui(self, listening: tuple, groups: List[str]) -> None:
        from .bookmarks_panel import _MhzSpinBox, format_mhz

        b = self._original
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._name = QLineEdit(str(b.get("label", "")))
        self._name.setPlaceholderText(format_mhz(b.get("freq_hz", 0.0)))
        self._name.setToolTip("Name shown in the list (the frequency when empty)")
        form.addRow("&Name:", self._name)

        self._freq = _MhzSpinBox()
        # (Up to the bookmark's own frequency: one made while a recording
        # above 6 GHz played was rewritten to 6000 MHz by Save.)
        try:
            saved = float(b.get("freq_hz", 0.0) or 0.0)
        except (TypeError, ValueError):
            saved = 0.0
        top = max(MAX_FREQUENCY_HZ, saved) if math.isfinite(saved) else MAX_FREQUENCY_HZ
        self._freq.setRange(MIN_FREQUENCY_HZ / 1e6, top / 1e6)
        self._freq.setDecimals(6)
        self._freq.setSuffix(" MHz")
        self._freq.setAlignment(Qt.AlignmentFlag.AlignRight)
        try:
            self._freq.setValue(float(b.get("freq_hz", 0.0) or 0.0) / 1e6)
        except (TypeError, ValueError):
            pass
        self._freq.setToolTip("Frequency, in MHz")
        form.addRow("&Frequency:", self._freq)

        self._mode = QComboBox()
        self._mode.addItem(KEEP_MODE_TEXT, "")
        for name, _tip in DEMOD_MODES:
            self._mode.addItem(name, name)
        self._mode.setToolTip(
            "How to listen to this bookmark: tuning it selects this mode "
            "(Keep the receiver's mode: tuning it leaves the mode alone)"
        )
        demod, deviation = (tuple(listening) + ("", ""))[:2]
        index = self._mode.findData(str(demod or ""))
        self._mode.setCurrentIndex(max(index, 0))
        form.addRow("&Mode:", self._mode)

        self._deviation = _ValueComboBox()
        for value, text, tip in FM_DEVIATIONS:
            self._deviation.add_value(value, text, tip)
        self._deviation.setToolTip("FM deviation (FM only)")
        self._deviation.setCurrentText(str(deviation or "5 kHz"))
        form.addRow("FM &deviation:", self._deviation)

        self._width = QComboBox()
        self._width.setToolTip(
            "Filter (channel) width to listen with; the mode's usual width "
            "when not set"
        )
        form.addRow("Filter &width:", self._width)

        self._tone = QComboBox()
        self._tone.addItem(_NO_TONE_TEXT, 0.0)
        for freq in CTCSS_FREQUENCIES:
            self._tone.addItem(ctcss_label(freq), float(freq))
        for inverted in (False, True):  # (DCS codes after the CTCSS tones)
            self._tone.insertSeparator(self._tone.count())
            for code in DCS_CODES:
                self._tone.addItem(
                    dcs_label(code, inverted), format_dcs(code, inverted)
                )
        self._tone.setToolTip(
            "Tone squelch (FM): tuning this bookmark only plays stations "
            "sending this sub-audible CTCSS tone or DCS code, as a repeater's "
            "users hear it"
        )
        initial = self._initial_squelch
        tone_index = self._tone.findData(initial) if initial else 0
        self._tone.setCurrentIndex(max(tone_index, 0))
        form.addRow("CTCSS/DCS &tone:", self._tone)

        self._group = QComboBox()
        self._group.setEditable(True)
        self._group.addItem("")
        for group in groups:
            if group:
                self._group.addItem(group)
        self._group.setCurrentText(str(b.get("group", "") or ""))
        self._group.setToolTip(
            "Group (folder) this bookmark is listed under; type a new name to "
            "make a group, or leave it empty"
        )
        line = self._group.lineEdit()
        if line is not None:
            line.setPlaceholderText("No group")
        form.addRow("&Group:", self._group)

        self._skip = QCheckBox("S&kip it when scanning bookmarks")
        self._skip.setChecked(str(b.get("skip", "") or "") == "S")
        self._skip.setToolTip(
            "The Scanner's Scan Bookmarks passes this bookmark by (CHIRP's "
            "Skip flag)"
        )
        form.addRow("", self._skip)

        # Repeater shift: where a radio loaded with this memory transmits
        # (CHIRP's Duplex and Offset; this receiver only listens).
        shift_row = QHBoxLayout()
        shift_row.setSpacing(6)
        self._duplex = QComboBox()
        for value, text in _DUPLEX_CHOICES:
            self._duplex.addItem(text, value)
        duplex = str(b.get("duplex", "") or "")
        if self._duplex.findData(duplex) < 0:
            self._duplex.addItem(duplex, duplex)  # e.g. split: kept as it is
        self._duplex.setCurrentIndex(max(self._duplex.findData(duplex), 0))
        self._duplex.setToolTip(
            "Repeater shift, for a radio this memory is exported to: it "
            "transmits this far above or below the frequency (CHIRP's Duplex)"
        )
        shift_row.addWidget(self._duplex, 1)
        self._offset = _MhzSpinBox()
        self._offset.setRange(0.0, 1000.0)
        self._offset.setDecimals(6)
        self._offset.setSuffix(" MHz")
        self._offset.setAlignment(Qt.AlignmentFlag.AlignRight)
        try:
            offset = abs(float(b.get("offset_hz", 0.0) or 0.0))
        except (TypeError, ValueError, OverflowError):  # (a damaged setting)
            offset = 0.0
        if math.isfinite(offset):
            self._offset.setValue(offset / 1e6)
        self._offset.setToolTip(
            "Repeater shift (CHIRP's Offset), e.g. 0.600 MHz on 2 m"
        )
        self._offset.setAccessibleName("Repeater shift")
        shift_row.addWidget(self._offset, 1)
        self._duplex.currentIndexChanged.connect(
            lambda _i: self._offset.setEnabled(self._duplex.currentData() in ("+", "-"))
        )
        self._offset.setEnabled(duplex in ("+", "-"))
        shift_label = QLabel("Repeater s&hift:")
        shift_label.setBuddy(self._duplex)
        form.addRow(shift_label, shift_row)

        self._comment = QLineEdit(str(b.get("comment", "") or ""))
        self._comment.setPlaceholderText("Optional")
        self._comment.setToolTip("A note for yourself (exported as CHIRP's Comment)")
        # (Not Alt+C or Alt+S: Linux desktops label the buttons "&Save" and
        # "&Cancel".)
        form.addRow("C&omment:", self._comment)
        layout.addLayout(form)

        self._hint = QLabel()
        set_role(self._hint, "hint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed()
        width = str(b.get("bandwidth", "") or "")
        if width:
            index = self._width.findData(width)
            if index < 0:
                self._width.addItem(width, width)
                index = self._width.count() - 1
            self._width.setCurrentIndex(index)

    def _on_mode_changed(self, _index: int = 0) -> None:
        """FM's deviation and tone apply to FM only; each mode has its own
        filter widths."""
        demod = self.demod()
        fm = demod == "FM"
        self._deviation.setEnabled(fm)
        self._tone.setEnabled(fm)
        current = self._width.currentData()
        self._width.blockSignals(True)
        self._width.clear()
        self._width.addItem(_DEFAULT_WIDTH_TEXT, "")
        if demod:
            for text in bandwidth_options(demod):
                self._width.addItem(text, text)
        index = self._width.findData(current) if current else 0
        self._width.setCurrentIndex(max(index, 0))
        self._width.setEnabled(bool(demod))
        self._width.blockSignals(False)
        if not demod:
            self._hint.setText(
                "Tuning this bookmark keeps the mode, filter and tone squelch "
                "you are listening with."
            )
        elif fm:
            self._hint.setText(
                "Tuning this bookmark selects FM with this deviation and "
                "filter, and sets the tone squelch to its tone or code (or off)."
            )
        else:
            self._hint.setText(f"Tuning this bookmark selects {demod}.")

    # ------------------------------------------------------------------ #
    def demod(self) -> str:
        """The chosen mode ("" = keep the receiver's mode)."""
        value = self._mode.currentData()
        return value if isinstance(value, str) else ""

    def _listening_choice(self) -> tuple:
        """``(mode, FM deviation)`` chosen (the deviation only in FM)."""
        demod = self.demod()
        return demod, self._deviation.currentText() if demod == "FM" else ""

    def bookmark(self) -> Dict[str, Any]:
        """The bookmark with the dialog's values."""
        from .bookmarks_panel import format_mhz

        b = dict(self._original)
        freq = float(round(self._freq.value() * 1e6))
        b["freq_hz"] = freq
        b["label"] = self._name.text().strip() or format_mhz(freq)
        demod, deviation = self._listening_choice()
        b.pop("bandwidth", None)
        width = self._width.currentData()
        if demod and isinstance(width, str) and width:
            b["bandwidth"] = width
        # The Mode list only approximates CHIRP's modes (DV, DMR and NFM
        # are listened to as FM, RTTY keeps the receiver's mode): unless the
        # mode or the deviation changed, the saved mode and CHIRP Mode stay
        # exactly as they were (a radio loaded with the channel needs them).
        if (demod, deviation) != self._opened_with:
            for key in ("demod", "fm_deviation"):
                b.pop(key, None)
            if demod:
                b["demod"] = demod
                if deviation:
                    b["fm_deviation"] = deviation
                chirp = chirp_mode_for(demod, deviation)
                if demod == "None (I/Q)" and not is_chirp_channel(self._original):
                    b.pop("mode", None)  # I/Q has no CHIRP mode to show
                else:
                    b["mode"] = chirp
            elif is_chirp_channel(self._original):
                # A CHIRP channel always has a Mode; "Auto" keeps the receiver's.
                b["mode"] = _CHIRP_KEEP_MODE
            else:
                b.pop("mode", None)
        # The tone field applies in FM only (greyed out otherwise): in the
        # other modes a channel's tone, e.g. a CHIRP memory's, is kept.
        if demod == "FM":
            choice = self._tone.currentData()
            if not isinstance(choice, (float, str)):
                choice = 0.0
            if choice != self._initial_squelch:
                code = parse_dcs(choice) if isinstance(choice, str) else None
                tone_mode = str(b.get("tone_mode") or "")
                if code is not None:
                    # Sent and heard with the same code (CHIRP's DTCS mode).
                    letter = "R" if code[1] else "N"
                    b["tone_mode"] = "DTCS"
                    b["dtcs"] = code[0]
                    b["dtcs_polarity"] = letter + letter
                elif isinstance(choice, float) and choice:
                    b["tone_mode"] = "TSQL"
                    b["ctone"] = choice
                elif tone_mode == "TSQL":
                    b.pop("tone_mode", None)
                    b.pop("ctone", None)
                elif tone_mode == "DTCS":
                    for key in ("tone_mode", "dtcs", "dtcs_polarity"):
                        b.pop(key, None)
        group = self._group.currentText().strip()
        if group:
            b["group"] = group
        else:
            b.pop("group", None)
        if self._skip.isChecked():
            b["skip"] = "S"
        elif str(b.get("skip", "") or "") == "S":
            b.pop("skip", None)
        duplex = self._duplex.currentData()
        duplex = duplex if isinstance(duplex, str) else ""
        if duplex in ("+", "-"):
            b["duplex"] = duplex
            b["offset_hz"] = float(round(self._offset.value() * 1e6))
        elif not duplex:
            b.pop("duplex", None)
            b.pop("offset_hz", None)
        comment = self._comment.text().strip()
        if comment:
            b["comment"] = comment
        else:
            b.pop("comment", None)
        return b


def edit_bookmark(
    parent: Any,
    bookmark: Dict[str, Any],
    listening: tuple,
    groups: Iterable[str],
) -> Optional[Dict[str, Any]]:
    """Show the dialog; the edited bookmark, or None when cancelled."""
    dialog = BookmarkEditDialog(bookmark, listening, groups, parent)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.bookmark()
    finally:
        dialog.deleteLater()
