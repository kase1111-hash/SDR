"""
The Second Receiver's listening options, as the main window has them.

* **Stereo**: broadcast FM plays in stereo where the station sends it,
  through the main window's own stereo path (``_ReceiverChain`` with
  ``stereo`` set, :class:`~.stereo_audio.StereoPath`); its STEREO lamp says
  what a station is received as and why, as the main window's badge does
  (:func:`stereo_reading`).
* **Notch** and **Low cut**: the receiver chain's own listening filters
  (``audio_chain.ListeningFilters``, set through the receiver's
  ``AudioOptions``): the Notch (Off, Auto, Manual at a frequency), the
  Whistle row under it (Manual's frequency, or every whistle Auto removes:
  :func:`notch_found_text`) and the Low cut (Off, 100, 200, 300 Hz),
  greyed with the reason where they don't apply (:func:`notch_unused`,
  :func:`low_cut_unused`), as in the main window's Audio group.
* **Tone squelch**: the main window's choices (Off, the 50 CTCSS tones,
  the 104 DCS codes in normal then inverted polarity: :func:`tone_choices`),
  its Reverse switch (CHIRP's TSQL-R and DTCS-R: the stations sending the
  tone or code muted) and its Tone heard readout (:func:`tone_heard_text`),
  through the same ``audio_chain.ToneSquelch``, in narrow FM only.

De-emphasis, Stereo Blend and Squelch Tail Elimination are the main
window's, for both receivers (``SecondReceiverWindow.follow_main_audio``).

What is decided here is Qt-free: the Second Receiver's DSP runs on a worker
thread (``second_receiver.FrameWork``), which hands the lamp's reading, the
whistles Auto found and the tone heard back to the window with each frame.
:class:`ListeningRows` (Qt) is a part of the Second Receiver window: its
rows in the Squelch and audio group, what they set on its DSP, and their
settings (``device/second_receiver/notch``, ``notch_hz``, ``low_cut``,
``tone_squelch``, ``dcs_squelch``, ``tone_reverse``).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from ..dsp.dcs import DCS_CODES, dcs_label, dcs_same_signal, format_dcs, parse_dcs
from ..dsp.filters import CTCSS_FREQUENCIES, ctcss_label
from .audio_chain import (
    DEFAULT_NOTCH_HZ,
    LOW_CUT_CHOICES,
    NOTCH_CHOICES,
    NOTCH_RANGE_HZ,
    REVERSE_TONE_TIP,
)

# (The main window's Audio group reads its readout's frequencies so.)
from .control_panel import _frequencies

# The main window's thresholds: STEREO lights at half the stereo
# difference or more (about 10 dB of separation) and goes out below a
# third of it (no flicker while the blend hovers); below
# _NEARLY_MONO_BLEND what plays is mono.
from .main_window import (
    _BROADCAST_DEVIATION_HZ,
    _NEARLY_MONO_BLEND,
    _STEREO_OFF_BLEND,
    _STEREO_ON_BLEND,
)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import set_tone

logger = logging.getLogger(__name__)

#: The Second Receiver's settings (``second_receiver.KEY``).
KEY = "device/second_receiver/"
#: The notch's choices (keys) and the low cut's.
_NOTCH_KEYS = tuple(key for key, _text, _tip in NOTCH_CHOICES)
_LOW_CUT_KEYS = tuple(key for key, _text, _tip in LOW_CUT_CHOICES)

__all__ = [
    "NO_STEREO_READING",
    "StereoReading",
    "ToneHeard",
    "low_cut_unused",
    "notch_found_text",
    "notch_unused",
    "parse_tone_choice",
    "stereo_reading",
    "tone_choices",
    "tone_heard_text",
]

#: ``(text, tone, tooltip)`` of the STEREO lamp: "STEREO" or "MONO" (""
#: hides it), its theme tone (None: plain) and why.
StereoReading = Tuple[str, Optional[str], str]
#: No broadcast FM station is heard: no lamp.
NO_STEREO_READING: StereoReading = ("", None, "")


def stereo_reading(
    received: bool,
    enabled: bool,
    usable: bool,
    decoder: Any,
    lit: bool,
    switch: str = "the Stereo switch",
) -> StereoReading:
    """The STEREO lamp for a broadcast FM station: ``received`` (one is
    above the squelch and someone listens), ``enabled`` (the Stereo switch
    is on), ``usable`` (the channel holds the stereo signal:
    ``StereoPath.usable``), ``decoder`` (the chain's stereo decoder while it
    follows the audio, else None), ``lit`` (the lamp said STEREO for the
    last block: it goes out only below a third of the stereo difference).
    ``switch`` names the switch that turns stereo off. The wording is the
    main window's badge's (``SDRMainWindow._show_stereo``), naming the
    Stereo Blend a blended or mono station is heard with: the main
    window's, which this receiver follows (``decoder.blending``)."""
    if not received:
        return NO_STEREO_READING
    if not enabled:
        return "MONO", None, f"Mono: {switch} is off."
    if not usable:
        return (
            "MONO",
            None,
            "Mono: the Filter is too narrow for the stereo signal. Broadcast "
            "FM's 200 kHz filter holds it.",
        )
    if decoder is None:
        return "MONO", None, "Mono: listening for the station's stereo pilot."
    blend = float(decoder.blend)
    blending = str(getattr(decoder, "blending", "") or "normal")
    setting = f"the main window's Radio > Audio > Stereo Blend: {blending.capitalize()}"
    if blend >= _STEREO_ON_BLEND or (lit and blend >= _STEREO_OFF_BLEND):
        how = (
            "Stereo: the station's 19 kHz pilot is locked and left and right "
            "are fully apart."
            if blend >= 0.95
            else "Stereo, blended partly toward mono to keep the hiss down: the "
            f"signal is a little weak ({setting})."
        )
        return (
            "STEREO",
            "success",
            f"{how} {switch[0].upper()}{switch[1:]} turns it off.",
        )
    if not decoder.pilot:
        return (
            "MONO",
            None,
            "Mono: this station sends no stereo pilot (it broadcasts in mono), "
            "or it is too weak to hear one.",
        )
    if float(decoder.target) >= _STEREO_ON_BLEND:
        return "MONO", None, "Mono for a moment: stereo is coming in."
    # (Under half the stereo difference: MONO, but said so when some of it
    # still plays, as the main window's badge does.)
    heard = (
        "Nearly mono: the signal is too weak for clean stereo, so it is "
        "blended most of the way toward mono"
        if blend > _NEARLY_MONO_BLEND
        else "Mono: the signal is too weak for clean stereo, so it plays in mono"
    )
    return (
        "MONO",
        None,
        f"{heard} to keep the hiss down ({setting}). Stereo comes back as the "
        "signal gets stronger.",
    )


# ---------------------------------------------------------------------------
# Notch and low cut (the main window's Audio group's rules and wording)
# ---------------------------------------------------------------------------

#: The Notch's and the Low cut's tooltips where they apply (the main
#: window's Audio group's: Auto removes up to three whistles, M-A3).
NOTCH_TIP = (
    "Removes whistles (heterodynes) from the audio. Auto finds up to three "
    "steady ones, the strongest first, and follows each as it drifts, leaving "
    "voice alone (for music use Off: a held note counts as a whistle); Manual "
    "removes the frequency you set (AM, SSB, CW and narrow FM)."
)
LOW_CUT_TIP = (
    "Cuts the audio below this frequency: mains hum and rumble out, the "
    "voice stays (AM, SSB and narrow FM). With a tone squelch set, narrow FM "
    "is cut below 300 Hz anyway (the tone or the DCS code)."
)


def notch_unused(mode: str, deviation_hz: float) -> str:
    """Why the notch isn't used in ``mode`` (at an FM ``deviation_hz``),
    "" where it is: AM, USB, LSB, CW and narrow FM (broadcast FM keeps its
    audio as it is; I/Q has none)."""
    if mode in ("AM", "USB", "LSB", "CW"):
        return ""
    if mode == "FM":
        if float(deviation_hz) < _BROADCAST_DEVIATION_HZ:
            return ""
        return "Not used in broadcast FM: its audio is kept as it is."
    return "Not used in None (I/Q) mode: there is no audio."


def low_cut_unused(mode: str, deviation_hz: float) -> str:
    """Why the low cut isn't used in ``mode``, "" where it is: AM, USB, LSB
    and narrow FM (not CW, whose filter sits around the note)."""
    if mode == "CW":
        return "Not used in CW: its filter sits around the CW note."
    return notch_unused(mode, deviation_hz)


def notch_found_text(found: Any) -> Tuple[str, str, str]:
    """Auto's readout, ``(text, tooltip, tone)``: what it removes, a
    whistle's frequency (Hz), several (a sequence, the strongest first: up
    to three) or None, in the main window's wording
    (``ControlPanel.show_notch_found``): several in order of frequency, so
    the row doesn't jump about when two are about as loud, the strongest
    first in the tooltip."""
    whistles = _frequencies(found)
    if not whistles:
        return (
            "None",
            "Auto hears no steady whistle on the channel: the audio plays as "
            "it is. It finds one within about half a second, and removes up to "
            "three at once.",
            "muted",
        )
    if len(whistles) == 1:
        hz = whistles[0]
        return (
            f"{hz:.0f} Hz",
            f"Auto removes a steady whistle at {hz:.1f} Hz",
            "success",
        )
    text = ", ".join(f"{hz:.0f}" for hz in sorted(whistles)) + " Hz"
    tip = (
        f"Auto removes {len(whistles)} steady whistles, the strongest first: "
        + ", ".join(f"{hz:.1f} Hz" for hz in whistles)
    )
    return text, tip, "success"


# ---------------------------------------------------------------------------
# Tone squelch (the main window's choices and Tone heard wording)
# ---------------------------------------------------------------------------

#: What the tone squelch hears: ``(CTCSS tone Hz or None, DCS (code,
#: inverted) or None, checked)``; ``checked`` False where tones aren't
#: looked for (not narrow FM).
ToneHeard = Tuple[Optional[float], Optional[Tuple[int, bool]], bool]

#: The Tone squelch's tooltip in FM, and elsewhere.
TONE_SQUELCH_TIP = (
    "Tone squelch: only play stations that send this sub-audible CTCSS tone "
    "or DCS code (repeaters, PMR); the tone or the code's bits are filtered "
    "out of what you hear. Tone heard shows the one on the channel. Squelch "
    "Tail Elimination (closing at a CTCSS station's reverse burst) is the "
    "main window's, for both receivers: Radio > Audio there."
)
TONE_SQUELCH_UNUSED = "Only used in FM mode"


def tone_choices() -> List[Optional[Tuple[str, str, str]]]:
    """The Tone squelch list, as the main window's: ``(value, text,
    tooltip)`` for Off, the CTCSS tones ("100.0") and the DCS codes, normal
    then inverted ("D023N", "D023I"), with None for a separator before each
    run of codes."""
    choices: List[Optional[Tuple[str, str, str]]] = [
        ("off", "Off", "Plays whatever is on the channel")
    ]
    for freq in CTCSS_FREQUENCIES:
        choices.append((f"{freq:.1f}", ctcss_label(freq), ""))
    for inverted in (False, True):
        choices.append(None)
        polarity = (
            "inverted polarity (also written I or R)" if inverted else "normal polarity"
        )
        for code in DCS_CODES:
            choices.append(
                (
                    format_dcs(code, inverted),
                    dcs_label(code, inverted),
                    f"DCS code {code:03d}, {polarity}",
                )
            )
    return choices


def parse_tone_choice(
    value: Any,
) -> Tuple[Optional[float], Optional[Tuple[int, bool]]]:
    """``(CTCSS tone, DCS code)`` of a Tone squelch choice ("off", "100.0",
    "D023N"): at most one of them set; ``(None, None)`` for Off or anything
    else."""
    text = str(value or "")
    if text.startswith("D"):
        return None, parse_dcs(text)
    try:
        tone = float(text)
    except ValueError:
        return None, None
    return (tone if tone > 0 else None), None


def tone_heard_text(
    freq: Optional[float],
    checked: bool = True,
    dcs: Optional[Tuple[int, bool]] = None,
    muted: bool = False,
) -> Tuple[str, str, str]:
    """Tone heard, ``(text, tooltip, tone)``: the sub-audible tone (or DCS
    code) on the channel, None for none, ``checked`` False where tones
    aren't looked for (a broadcast deviation); ``muted``: the one heard is
    the one a reverse tone squelch mutes (said, in the warning tone). The
    main window's wording (``ControlPanel.set_detected_tone``)."""
    mutes = ". The reverse tone squelch mutes it."
    if checked and dcs is not None:
        code, inverted = dcs
        tip = (
            f"The station sends DCS code {code:03d} "
            f"({'inverted' if inverted else 'normal'} polarity)"
        )
        twin = dcs_same_signal(code, inverted)
        if twin is not None:
            tip += f": the same signal as {dcs_label(*twin)}"
        if muted:
            return dcs_label(code, inverted), tip + mutes, "warning"
        return dcs_label(code, inverted), tip, "success"
    if not checked:
        return (
            "Not checked",
            "Tones aren't looked for at a broadcast FM deviation (broadcast "
            "stations send none). Choose 5 kHz (narrow voice) for a repeater or "
            "PMR channel to see the tone it sends.",
            "muted",
        )
    if freq is None:
        return (
            "None",
            "No sub-audible tone (CTCSS) or code (DCS) heard on the channel. "
            "Repeaters and PMR radios often send one; set the tone squelch to "
            "it to hear only them.",
            "muted",
        )
    label = ctcss_label(freq)
    tip = f"The station sends a {label} CTCSS tone"
    if muted:
        return label, tip + mutes, "warning"
    return label, tip, "success"


if HAS_PYQT6:

    class ListeningRows:
        """Part of the Second Receiver window (a mixin of
        ``SecondReceiverWindow``): the Tone squelch, Reverse, Tone heard,
        Notch, Whistle and Low cut rows of its Squelch and audio group, what they
        set on its DSP (``self._dsp.options``, read by the receiver chain
        on every block, and ``self._dsp.tone_squelch``: a change takes
        effect at once, without a click), and their settings. Uses the
        window's ``_combo``, ``_dsp``, ``_settings``, ``_save``,
        ``_restoring``, ``_running``, ``_show_message``, ``mode()`` and
        ``_deviation_hz()``."""

        def _build_tone_rows(self, form: Any) -> None:
            """The Tone squelch, Reverse and Tone heard rows, added to
            ``form``."""
            self._tone_form = form
            combo = self._combo(6)
            for choice in tone_choices():
                if choice is None:
                    combo.insertSeparator(combo.count())
                    continue
                value, text, tip = choice
                combo.addItem(text, value)
                if tip:
                    combo.setItemData(
                        combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                    )
            combo.setAccessibleName("Second receiver's tone squelch")
            self._tone_combo = combo
            label = QLabel("Tone squelch:")
            label.setBuddy(combo)
            form.addRow(label, combo)
            # (M-Q3) Reversed, as the main window's Audio group has it: the
            # stations sending the tone or code are muted, every other one
            # plays (CHIRP's TSQL-R and DTCS-R).
            self._tone_reverse_check = QCheckBox("Reverse: mute this tone")
            self._tone_reverse_check.setAccessibleName(
                "Second receiver's reverse tone squelch"
            )
            self._quiet_reverse = False  # (a bookmark's own message says it)
            form.addRow("", self._tone_reverse_check)
            combo.currentIndexChanged.connect(self._on_tone_changed)
            self._tone_reverse_check.toggled.connect(self._on_tone_reverse_toggled)
            # The tone (or code) the station sends (FM only).
            self._tone_heard = QLabel()
            self._tone_heard.setAccessibleName("Tone the second receiver hears")
            form.addRow(QLabel("Tone heard:"), self._tone_heard)
            self._update_tone_reverse()
            self._show_tone_heard(None, True, None)

        def _tone_choice(
            self,
        ) -> Tuple[Optional[float], Optional[Tuple[int, bool]]]:
            """The Tone squelch's ``(CTCSS tone, DCS code)``."""
            return parse_tone_choice(self._tone_combo.currentData())

        def _apply_tone(self) -> None:
            """The tone squelch chosen, reversed or not, to the DSP: one
            value its worker takes up at the next block (Reverse ticked with
            no tone or code chosen reverses nothing)."""
            tone, dcs = self._tone_choice()
            chosen = tone is not None or dcs is not None
            reverse = chosen and self._tone_reverse_check.isChecked()
            dsp = self._dsp
            dsp.tone_squelch = (tone, dcs, True) if reverse else (tone, dcs)
            dsp.options.tone_squelch = chosen
            self._update_tone_reverse()

        def _on_tone_changed(self, *_args: Any) -> None:
            tone, dcs = self._tone_choice()
            self._apply_tone()
            self._save(KEY + "tone_squelch", float(tone or 0.0))
            self._save(KEY + "dcs_squelch", format_dcs(*dcs) if dcs else "")
            self._say_tone_unused()

        def _on_tone_reverse_toggled(self, on: bool) -> None:
            """Reverse ticked or not: applied at once, saved, and said (as
            the main window says it), unless a bookmark's own message
            does."""
            self._apply_tone()
            self._save(KEY + "tone_reverse", bool(on))
            tone, dcs = self._tone_choice()
            if self._restoring or self._quiet_reverse:
                return
            if tone is None and dcs is None:
                return
            what = dcs_label(*dcs) if dcs is not None else f"the {tone:.1f} Hz tone"
            if on:
                text = (
                    f"Reverse tone squelch: stations sending {what} are muted, "
                    "every other station plays."
                )
            else:
                text = f"Tone squelch: only stations sending {what} play."
            self._show_message(text, "info", 4000)
            self._say_tone_unused()

        def _say_tone_unused(self) -> None:
            """A tone or code chosen at a broadcast deviation: say it isn't
            used there."""
            deviation = self._deviation_hz()
            if (
                self._dsp.options.tone_squelch
                and not self._restoring
                and self.mode() == "FM"
                and deviation >= _BROADCAST_DEVIATION_HZ
            ):
                self._show_message(
                    "Tone squelch works on narrow FM (repeaters, PMR): it isn't "
                    f"used at {deviation / 1e3:g} kHz deviation, so this station "
                    "plays as before.",
                    "info",
                    6000,
                )

        def _update_tone_reverse(self) -> None:
            """Reverse applies to a tone or code chosen, in FM (the main
            window's Audio group's rules and wording)."""
            check = self._tone_reverse_check
            tone, dcs = self._tone_choice()
            chosen = tone is not None or dcs is not None
            fm = self.mode() == "FM"
            check.setText(
                "Reverse: mute this code"
                if dcs is not None
                else "Reverse: mute this tone"
            )
            check.setEnabled(fm and chosen)
            if not fm:
                tip = TONE_SQUELCH_UNUSED
            elif not chosen:
                tip = (
                    "Choose a CTCSS tone or DCS code in Tone squelch first: Reverse "
                    "mutes the stations that send it."
                )
            else:
                tip = REVERSE_TONE_TIP
            check.setToolTip(tip)

        def _set_tone_choice(
            self,
            tone: Optional[float],
            dcs: Optional[Tuple[int, bool]],
            reverse: Optional[bool] = None,
        ) -> None:
            """Choose a CTCSS tone or a DCS code (one not in the list, or
            neither: Off), and (``reverse`` not None) whether it is
            reversed, without saying so (the caller does)."""
            combo = self._tone_combo
            if dcs is not None:
                index = combo.findData(format_dcs(*dcs))
            elif tone:
                index = combo.findData(f"{float(tone):.1f}")
            else:
                index = 0
            combo.setCurrentIndex(max(index, 0))
            if reverse is not None:
                self._quiet_reverse = True
                try:
                    self._tone_reverse_check.setChecked(bool(reverse))
                finally:
                    self._quiet_reverse = False

        def _show_tone_heard(
            self,
            freq: Optional[float],
            checked: bool,
            dcs: Optional[Tuple[int, bool]],
        ) -> None:
            """Tone heard (refreshed every frame: nothing is touched unless
            it changed); the one a reverse tone squelch mutes is said so,
            in the warning tone, as in the main window."""
            muted = False
            if checked and self._tone_reverse_check.isChecked():
                tone, chosen = self._tone_choice()
                if dcs is not None:
                    muted = chosen is not None and chosen in (
                        dcs,
                        dcs_same_signal(*dcs),
                    )
                elif freq is not None and tone is not None:
                    muted = abs(tone - float(freq)) < 0.05
            text, tip, tone_name = tone_heard_text(freq, checked, dcs, muted)
            label = self._tone_heard
            if label.text() != text or label.toolTip() != tip:
                label.setText(text)
                label.setToolTip(tip)
                set_tone(label, tone_name)

        def _show_no_tone_heard(self) -> None:
            """Nothing is received (stopped): Tone heard says None in narrow
            FM and Not checked at a broadcast deviation, as while receiving."""
            narrow = (
                self.mode() == "FM" and self._deviation_hz() < _BROADCAST_DEVIATION_HZ
            )
            self._show_tone_heard(None, narrow, None)

        def _build_filter_rows(self, form: Any) -> None:
            """The Notch, Whistle and Low cut rows, added to ``form``."""
            self._filter_form = form
            self._notch_combo = self._combo(6)
            for key, text, tip in NOTCH_CHOICES:
                self._notch_combo.addItem(text, key)
                self._notch_combo.setItemData(
                    self._notch_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                )
            self._notch_combo.setAccessibleName("Second receiver's notch")
            label = QLabel("Notch:")
            label.setBuddy(self._notch_combo)
            form.addRow(label, self._notch_combo)
            # Under it, in Auto and Manual: Manual's frequency, or the
            # whistle Auto removes (a row of its own, as in the main window).
            self._notch_spin = QSpinBox()
            self._notch_spin.setRange(*NOTCH_RANGE_HZ)
            self._notch_spin.setValue(DEFAULT_NOTCH_HZ)
            self._notch_spin.setSuffix(" Hz")
            self._notch_spin.setAccelerated(True)
            self._notch_spin.setAccessibleName("Second receiver's notch frequency")
            self._notch_spin.setToolTip(
                "The whistle's frequency, to remove it by hand (Manual; "
                f"{NOTCH_RANGE_HZ[0]}-{NOTCH_RANGE_HZ[1]} Hz). Type it, or use "
                "the arrows (Page Up/Down: 10 Hz) until the whistle is gone."
            )
            self._notch_found = QLabel()
            self._notch_found.setAccessibleName("Whistle the second receiver removes")
            self._notch_stack = QStackedWidget()
            self._notch_stack.addWidget(self._notch_spin)
            self._notch_stack.addWidget(self._notch_found)
            self._notch_stack.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
            notch_row = QWidget()
            row = QHBoxLayout(notch_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self._notch_stack)
            row.addStretch(1)
            label = QLabel("Whistle:")
            label.setBuddy(self._notch_spin)
            form.addRow(label, notch_row)
            self._notch_row = notch_row
            self._notch_label = label  # ("Whistles:" for several)
            form.setRowVisible(notch_row, False)  # (Off)
            self._notch_combo.currentIndexChanged.connect(self._on_notch_changed)
            self._notch_spin.valueChanged.connect(self._on_notch_changed)
            self._show_notch_found(None)

            self._low_cut_combo = self._combo(6)
            for key, text, tip in LOW_CUT_CHOICES:
                self._low_cut_combo.addItem(text, key)
                self._low_cut_combo.setItemData(
                    self._low_cut_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                )
            self._low_cut_combo.setAccessibleName("Second receiver's low cut")
            self._low_cut_combo.currentIndexChanged.connect(self._on_low_cut_changed)
            label = QLabel("Low cut:")
            label.setBuddy(self._low_cut_combo)
            form.addRow(label, self._low_cut_combo)

        def _notch_mode(self) -> str:
            """``"off"``, ``"auto"`` or ``"manual"``."""
            key = self._notch_combo.currentData()
            return key if key in _NOTCH_KEYS else "off"

        def _low_cut_key(self) -> str:
            """``"off"``, or the low cut's corner in Hz (``"100"``...)."""
            key = self._low_cut_combo.currentData()
            return key if key in _LOW_CUT_KEYS else "off"

        def _on_notch_changed(self, *_args: Any) -> None:
            options = self._dsp.options
            options.notch = self._notch_mode()
            options.notch_hz = float(self._notch_spin.value())
            self._update_filter_controls()
            self._save(KEY + "notch", options.notch)
            self._save(KEY + "notch_hz", int(self._notch_spin.value()))

        def _on_low_cut_changed(self, *_args: Any) -> None:
            key = self._low_cut_key()
            self._dsp.options.low_cut_hz = 0.0 if key == "off" else float(key)
            self._save(KEY + "low_cut", key)

        def _update_filter_controls(self) -> None:
            """Greyed, with the reason in the tooltip, where they don't
            apply (the mode, or broadcast FM); the Whistle row in Auto
            and Manual."""
            mode, deviation = self.mode(), self._deviation_hz()
            is_fm = mode == "FM"
            self._tone_combo.setEnabled(is_fm)
            self._tone_combo.setToolTip(
                TONE_SQUELCH_TIP if is_fm else TONE_SQUELCH_UNUSED
            )
            self._tone_form.setRowVisible(self._tone_heard, is_fm)
            self._tone_form.setRowVisible(self._tone_reverse_check, is_fm)
            self._update_tone_reverse()
            if not self._running:
                self._show_no_tone_heard()  # (frames say it while running)
            why = notch_unused(mode, deviation)
            notch = self._notch_mode()
            self._notch_combo.setEnabled(not why)
            self._notch_spin.setEnabled(not why and notch == "manual")
            self._notch_found.setEnabled(not why)
            if why:
                self._show_notch_found(None)  # (nothing is removed here)
            self._notch_stack.setCurrentWidget(
                self._notch_found if notch == "auto" else self._notch_spin
            )
            self._filter_form.setRowVisible(self._notch_row, notch != "off")
            self._notch_combo.setToolTip(why or NOTCH_TIP)
            why = low_cut_unused(mode, deviation)
            self._low_cut_combo.setEnabled(not why)
            self._low_cut_combo.setToolTip(why or LOW_CUT_TIP)

        def _show_notch_found(self, found: Any) -> None:
            """The Whistle row's readout in Auto: every whistle it removes
            (a frame's ``notches``, the strongest first; one frequency or
            None also do). Refreshed every frame: nothing is touched unless
            it changed."""
            text, tip, tone = notch_found_text(found)
            label = self._notch_found
            if label.text() != text or label.toolTip() != tip:
                label.setText(text)
                label.setToolTip(tip)
                set_tone(label, tone)
            caption = "Whistles:" if len(_frequencies(found)) > 1 else "Whistle:"
            if self._notch_label.text() != caption:
                self._notch_label.setText(caption)

        def _restore_filters(self) -> None:
            """The saved tone squelch (and whether it is reversed), Notch,
            its frequency and the Low cut, each on its own: a damaged value
            is logged and gives Off (the frequency: its box's nearest end),
            never stopping the others."""
            s = self._settings
            # (A damaged value is logged by the settings and read as off.)
            reverse = s.get_bool(KEY + "tone_reverse", False)
            self._tone_reverse_check.blockSignals(True)
            self._tone_reverse_check.setChecked(reverse)
            self._tone_reverse_check.blockSignals(False)
            tone = s.get_float(KEY + "tone_squelch", 0.0)
            index = self._tone_combo.findData(f"{tone:.1f}") if tone else 0
            if index < 0:
                logger.warning(
                    f"Ignoring the damaged setting {KEY}tone_squelch={tone!r}: no "
                    "tone squelch."
                )
                index = 0
            raw = s.get_str(KEY + "dcs_squelch", "")
            if raw:
                code = parse_dcs(raw)
                if code is None:
                    logger.warning(
                        f"Ignoring the damaged setting {KEY}dcs_squelch={raw!r}: no "
                        "DCS code squelch."
                    )
                else:
                    index = max(0, self._tone_combo.findData(format_dcs(*code)))
            self._tone_combo.setCurrentIndex(index)
            self._on_tone_changed()
            hz = s.get_int(KEY + "notch_hz", DEFAULT_NOTCH_HZ)
            low, high = NOTCH_RANGE_HZ
            kept = int(min(max(hz, low), high))
            if kept != hz:
                logger.warning(
                    f"Ignoring the damaged setting {KEY}notch_hz={hz!r}: using "
                    f"{kept} Hz instead."
                )
            self._notch_spin.setValue(kept)
            for key, choices, combo in (
                ("notch", _NOTCH_KEYS, self._notch_combo),
                ("low_cut", _LOW_CUT_KEYS, self._low_cut_combo),
            ):
                value = s.get_str(KEY + key, "off")
                if value not in choices:
                    logger.warning(
                        f"Ignoring the damaged setting {KEY}{key}={value!r}: "
                        "using Off instead."
                    )
                    value = "off"
                combo.setCurrentIndex(max(0, combo.findData(value)))
            # (The signals applied these; make sure even unchanged defaults
            # are in effect.)
            self._on_notch_changed()
            self._on_low_cut_changed()
