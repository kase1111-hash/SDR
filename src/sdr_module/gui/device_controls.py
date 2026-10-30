"""
The connected device's settings in the window: gain, sample rate, front end.

:class:`DeviceControls` (the main window's ``_device_controls``) keeps the
control panel's Receiver group in step with the device the window uses:

- **RF gain**: the device's own steps (an RTL-SDR's tuner gain table; a
  HackRF One's LNA, VGA and +14 dB RF amp, with an overload warning; the
  demo device 0-50 dB). Tuner AGC only where the hardware has it; a device
  that refuses automatic gain is reported and the gain stays manual. With
  AGC on, the readout says what the automatic gain chose when the device
  says ("Auto (25.9 dB)").
- **Sample rate**: Receiver > Sample rate, Device > Sample Rate and the
  status bar's rate (click it) offer the device's rates and apply one live
  through the window's ``set_sample_rate`` (no reconnecting); refused, with
  the reason, while a recording plays back, records or the HackRF
  transmits.
- **Front end**: an RTL-SDR's frequency correction (ppm), direct sampling
  (HF) and bias tee; a HackRF's bias tee (antenna power). Applied live.
- Per device kind (demo, RTL-SDR, HackRF) the gain, the sample rate and the
  front end are remembered (``device/<kind>/...``) and given back when a
  device of that kind is connected again.

The window calls :meth:`configure` when a device is connected (from
``_apply_controls_to_device``, before it is tuned), :meth:`sync` whenever
its state changes (``_refresh_state_ui``) and :meth:`apply_manual_gain`
after tuning. :class:`ScanResultsDialog` is Device > Scan for Devices.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PyQt6.QtCore import QObject, QPoint
    from PyQt6.QtGui import QAction, QActionGroup
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QLabel,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

from .device_settings import (
    ADSB_MIN_RATE,
    ADSB_RATE_TIP,
    AMP_WARNING,
    DIRECT_SAMPLING_CHOICES,
    HACKRF_AMP_DB,
    HACKRF_DEFAULT_LNA,
    HACKRF_DEFAULT_VGA,
    KIND_NAMES,
    FrontEnd,
    auto_gain_db,
    device_kind,
    format_db,
    format_rate,
    front_end_features,
    gain_model_for,
    load_front_end,
    load_gain,
    load_rate,
    rate_choice_label,
    sample_rates_for,
    save_front_end,
    save_gain,
    save_rate,
)

if HAS_PYQT6:
    from .themes import set_role

logger = logging.getLogger(__name__)

#: Device kinds whose gain, sample rate and front end are remembered.
REMEMBERED_KINDS = ("demo", "rtlsdr", "hackrf")
#: Why the gain can't be set while a recording plays back.
PLAYBACK_GAIN = "A recording plays with the gain it was made with."

_DIRECT_LABELS = {value: label for value, label, _tip in DIRECT_SAMPLING_CHOICES}
_BRANCH_VALUES = {0: "off", 1: "i", 2: "q"}


def _kind_name(kind: str) -> str:
    """ "the RTL-SDR", "the HackRF One", "the demo device", "the device"."""
    return KIND_NAMES.get(kind, "the device")


class DeviceControls(QObject if HAS_PYQT6 else object):
    """Keeps the Receiver group's device settings in step with the window's
    device (see the module notes)."""

    def __init__(self, window: Any):
        super().__init__(window)
        self._win = window
        self._panel = window._control_panel
        # (device id, kind, why the gain is locked) the controls show.
        self._shown_for: Optional[Tuple[int, str, str]] = None
        # While configure() fills the controls: nothing is saved.
        self._configuring = False
        #: Device > Sample Rate, also popped up by the status bar's rate.
        self.rate_menu = QMenu("Sample &Rate", window)
        self.rate_menu.setToolTipsVisible(True)
        self.rate_menu.aboutToShow.connect(self._fill_rate_menu)
        panel = self._panel
        panel.sample_rate_selected.connect(self.choose_rate)
        panel.hackrf_gain_changed.connect(self._on_hackrf_gain_changed)
        panel.amp_changed.connect(self._on_amp_changed)
        panel.ppm_changed.connect(self._on_ppm_changed)
        panel.direct_sampling_changed.connect(self._on_direct_changed)
        panel.bias_tee_changed.connect(self._on_bias_changed)

    # ------------------------------------------------------------------ state

    @property
    def _settings(self) -> Any:
        return self._win._settings

    def kind(self) -> str:
        """The kind of the window's device ("" with none)."""
        return device_kind(self._win._device)

    def _status(self, text: str, tone: Optional[str] = None, ms: int = 5000) -> None:
        self._win._show_status_message(text, tone, ms)

    def _refresh_info(self) -> None:
        win = self._win
        if win._info_panel.isVisible():
            win._refresh_info_panel()

    # --------------------------------------------------------- show / connect

    def sync(self) -> None:
        """Show the controls of the window's device (cheap when nothing
        changed): the window calls this whenever its device or state
        changes."""
        win = self._win
        dev = win._device
        kind = device_kind(dev)
        locked = PLAYBACK_GAIN if kind == "playback" else ""
        key = (id(dev) if dev is not None else 0, kind, locked)
        if key != self._shown_for:
            self._shown_for = key
            self._panel.set_gain_model(gain_model_for(dev), locked)
            self._panel.set_front_end(kind, self._front_end_of(dev, kind))
        self.sync_rates()
        self.show_auto_gain()

    def configure(self) -> str:
        """A device was just connected: show its controls with what was
        last used with its kind (the gain, Tuner AGC, the front end) and set
        its front end, before it is tuned (direct sampling depends on where
        it tunes). Returns a note for the connect message ("" if none)."""
        win = self._win
        dev = win._device
        kind = device_kind(dev)
        locked = PLAYBACK_GAIN if kind == "playback" else ""
        panel = self._panel
        self._configuring = True
        try:
            model = gain_model_for(dev)
            panel.set_gain_model(model, locked)
            saved = (
                load_gain(self._settings, kind) if kind in REMEMBERED_KINDS else None
            )
            if model.kind == "hackrf":
                lna, vga, amp = HACKRF_DEFAULT_LNA, HACKRF_DEFAULT_VGA, False
                if saved:
                    try:
                        lna = int(saved.get("lna", lna))
                        vga = int(saved.get("vga", vga))
                        amp = bool(saved.get("amp", amp))
                    except (TypeError, ValueError):
                        logger.debug(f"Ignoring an unreadable HackRF gain: {saved!r}")
                panel.set_hackrf_gains(lna, vga, amp)
                panel.set_agc_silently(False)
            else:
                agc = panel.is_agc_enabled()
                if saved:
                    try:
                        if "db" in saved:
                            panel.set_gain_silently(float(saved["db"]))
                        agc = bool(saved.get("agc", agc))
                    except (TypeError, ValueError):
                        logger.debug(f"Ignoring an unreadable saved gain: {saved!r}")
                panel.set_agc_silently(agc and model.agc)
            front = load_front_end(self._settings, kind)
            note = self._apply_front_end(dev, kind, front)
            panel.set_front_end(kind, self._front_end_of(dev, kind, front))
            self._shown_for = (id(dev) if dev is not None else 0, kind, locked)
        finally:
            self._configuring = False
        if kind in REMEMBERED_KINDS:
            save_rate(self._settings, kind, win._device_sample_rate())
        return note

    def apply_manual_gain(self) -> None:
        """Set the device to the gain shown (Tuner AGC off)."""
        panel = self._panel
        if panel.gain_model().kind == "hackrf":
            lna, vga, amp = panel.hackrf_gains()
            self._set_hackrf_gain(lna, vga)
            self._set_amp(amp, announce=False)
            self._hackrf_gain_applied()
        else:
            self._win._on_gain_changed(panel.gain_db())

    def _front_end_of(
        self, dev: Any, kind: str, front: Optional[FrontEnd] = None
    ) -> FrontEnd:
        """The front end to show: what ``dev`` says it uses, else what is
        remembered for ``kind``."""
        front = front if front is not None else load_front_end(self._settings, kind)
        if dev is None:
            return front
        features = front_end_features(kind)
        if "ppm" in features:
            try:
                front.ppm = int(getattr(dev, "freq_correction", front.ppm))
            except (TypeError, ValueError):
                pass
        if "direct" in features:
            branch = getattr(dev, "direct_sampling_branch", None)
            if branch in _BRANCH_VALUES:
                front.direct_sampling = _BRANCH_VALUES[branch]
        if "bias_tee" in features:
            state = getattr(dev, "state", None)
            if state is not None:
                front.bias_tee = bool(getattr(state, "bias_tee_enabled", False))
        return front

    def _apply_front_end(self, dev: Any, kind: str, front: FrontEnd) -> str:
        """Set ``dev``'s front end to ``front``; a note for the user."""
        features = front_end_features(kind)
        if dev is None or not features:
            return ""
        failed: List[str] = []
        if "direct" in features:
            if not self._call(dev, "set_direct_sampling", front.direct_sampling):
                failed.append("direct sampling")
        if "ppm" in features and int(front.ppm) != int(
            getattr(dev, "freq_correction", 0) or 0
        ):
            if not self._call(dev, "set_freq_correction", int(front.ppm)):
                failed.append(f"the {int(front.ppm):+d} ppm correction")
        bias_on = bool(getattr(getattr(dev, "state", None), "bias_tee_enabled", False))
        if "bias_tee" in features and bool(front.bias_tee) != bias_on:
            if not self._call(dev, "set_bias_tee", bool(front.bias_tee)):
                failed.append("the bias tee")
        notes = []
        if failed:
            notes.append(
                f"Couldn't set {' and '.join(failed)} (Tools > Error History says "
                "why)."
            )
        on = getattr(getattr(dev, "state", None), "bias_tee_enabled", False)
        if "bias_tee" in features and on:
            notes.append("The bias tee is on: the antenna socket is powered.")
        return " ".join(notes)

    @staticmethod
    def _call(dev: Any, name: str, *args: Any) -> bool:
        """``dev.name(*args)``; False if it refused, raised or has no such
        method (the driver logs why)."""
        method = getattr(dev, name, None)
        if not callable(method):
            return False
        try:
            return method(*args) is not False
        except NotImplementedError:
            return False
        except Exception as e:
            logger.warning(f"{name}{args} failed: {e}")
            return False

    # ------------------------------------------------------------------ gain

    def save_gain(self) -> None:
        """Remember the gain shown for the device kind (not while
        configuring, nor for kinds that aren't remembered)."""
        if self._configuring:
            return
        kind = self.kind()
        if kind not in REMEMBERED_KINDS:
            return
        panel = self._panel
        if panel.gain_model().kind == "hackrf":
            lna, vga, amp = panel.hackrf_gains()
            save_gain(self._settings, kind, {"lna": lna, "vga": vga, "amp": amp})
        else:
            save_gain(
                self._settings,
                kind,
                {"db": panel.gain_db(), "agc": panel.is_agc_enabled()},
            )

    def show_auto_gain(self) -> None:
        """While AGC is on: what it chose, when the device says."""
        win = self._win
        if getattr(win, "_agc_enabled", False):
            self._panel.show_auto_gain(auto_gain_db(win._device))
        else:
            self._panel.show_auto_gain(None)

    def agc_changed(self, enabled: bool, accepted: bool) -> None:
        """The window switched the device's automatic gain (Tuner AGC).
        ``accepted`` False: the device refused it."""
        if not accepted and enabled:
            model = self._panel.gain_model()
            name = _kind_name(self.kind())
            reason = model.agc_tip or (
                f"{name[:1].upper()}{name[1:]} didn't switch to automatic gain."
            )
            self._status(f"{reason} The gain stays as set.", "warning", 8000)
            # Back to manual: the box unticked, the device's gain mode and
            # the gain shown applied again.
            self._panel.set_agc_silently(False)
            self._win._on_agc_changed(False)
            self.apply_manual_gain()
            return
        if not accepted:
            self._win._show_status_error(
                "Couldn't switch the tuner AGC off; the gain may stay automatic."
            )
        self.show_auto_gain()
        self.save_gain()

    def gain_applied(self) -> None:
        """The window set a manual gain on the device: remember it."""
        self.save_gain()

    def _hackrf_gain_applied(self) -> None:
        """LNA, VGA or the RF amp changed: remember them; the S-meter's
        calibration and the Info tab follow."""
        win = self._win
        self.save_gain()
        meter = getattr(win, "_signal_meter_panel", None)
        if meter is not None:
            meter.set_gain(win._signal_meter_gain())
        self._refresh_info()

    def _set_hackrf_gain(self, lna: int, vga: int) -> bool:
        dev = self._win._device
        if device_kind(dev) != "hackrf":
            return True
        ok = self._call(dev, "set_lna_gain", int(lna)) and self._call(
            dev, "set_vga_gain", int(vga)
        )
        if not ok:
            self._win._show_status_error(
                f"The HackRF didn't take LNA {lna} dB / VGA {vga} dB (Tools > "
                "Error History says why)."
            )
        return ok

    def _on_hackrf_gain_changed(self, lna: int, vga: int) -> None:
        self._set_hackrf_gain(lna, vga)
        self._hackrf_gain_applied()

    def _set_amp(self, on: bool, announce: bool = True) -> bool:
        dev = self._win._device
        if device_kind(dev) != "hackrf":
            return True
        if not self._call(dev, "set_amp", bool(on)):
            lna, vga, _amp = self._panel.hackrf_gains()
            self._panel.set_hackrf_gains(lna, vga, bool(dev.amp_enabled))
            self._win._show_status_error(
                "The HackRF didn't switch its RF amp "
                + ("on" if on else "off")
                + " (Tools > Error History says why)."
            )
            return False
        if announce:
            if on:
                self._status(AMP_WARNING.lstrip("⚠ "), "warning", 8000)
            else:
                self._status("RF amp off.", None, 3000)
        return True

    def _on_amp_changed(self, on: bool) -> None:
        self._set_amp(on)
        self._hackrf_gain_applied()

    def gain_text(self) -> str:
        """The Info tab's Gain: what the device really uses."""
        win = self._win
        dev = win._device
        panel = self._panel
        kind = device_kind(dev)
        if kind == "playback":
            return "As recorded"
        if win._agc_enabled:
            auto = auto_gain_db(dev)
            if auto is not None:
                return f"Auto ({format_db(round(auto, 1))})"
            return "Auto (tuner AGC)" if kind == "rtlsdr" else "Automatic (AGC)"
        if panel.gain_model().kind == "hackrf":
            lna, vga, amp = panel.hackrf_gains()
            if kind == "hackrf":
                lna, vga = int(dev.lna_gain), int(dev.vga_gain)
                amp = bool(dev.amp_enabled)
            total = lna + vga + (HACKRF_AMP_DB if amp else 0)
            parts = f"LNA {lna} + VGA {vga}" + (
                f" + amp {HACKRF_AMP_DB}" if amp else ""
            )
            return f"{format_db(total)} ({parts}" + ("" if amp else ", amp off") + ")"
        value = panel.gain_db()
        if kind in ("rtlsdr", "other"):
            state_gain = getattr(getattr(dev, "state", None), "gain", None)
            if isinstance(state_gain, (int, float)) and kind == "rtlsdr":
                value = float(state_gain)
        elif kind == "demo":
            try:
                value = float(dev.gain)
            except (TypeError, ValueError, AttributeError):
                pass
        return format_db(value)

    def front_end_text(self) -> str:
        """The Info tab's Front end: frequency correction, direct sampling
        and bias tee of the device ("—" when it has none)."""
        dev = self._win._device
        kind = device_kind(dev)
        features = front_end_features(kind)
        if dev is None or not features:
            return "—"
        front = self._front_end_of(dev, kind)
        parts = []
        if "ppm" in features:
            parts.append(f"{front.ppm:+d} ppm")
        if "direct" in features:
            label = _DIRECT_LABELS.get(front.direct_sampling, front.direct_sampling)
            text = f"HF: {label}"
            if getattr(dev, "direct_sampling", False):
                text += ", in use"
            parts.append(text)
        if "bias_tee" in features:
            parts.append("bias tee on" if front.bias_tee else "bias tee off")
        return " · ".join(parts)

    # ------------------------------------------------------------- front end

    def _front_end_device(self, feature: str) -> Any:
        """The window's device when it has ``feature``, else None."""
        dev = self._win._device
        if dev is None or feature not in front_end_features(device_kind(dev)):
            return None
        return dev

    def _save_front_end(self) -> None:
        kind = self.kind()
        if kind in REMEMBERED_KINDS:
            save_front_end(self._settings, kind, self._panel.front_end())
        self._refresh_info()

    def _show_device_front_end(self) -> None:
        """Back to what the device really uses (after a refusal)."""
        dev = self._win._device
        kind = device_kind(dev)
        self._panel.set_front_end(kind, self._front_end_of(dev, kind))

    def _on_ppm_changed(self, ppm: int) -> None:
        dev = self._front_end_device("ppm")
        if dev is None:
            return
        if not self._call(dev, "set_freq_correction", int(ppm)):
            self._show_device_front_end()
            self._win._show_status_error(
                f"The RTL-SDR didn't take a {int(ppm):+d} ppm correction (Tools > "
                "Error History says why)."
            )
            return
        self._save_front_end()
        self._status(f"Frequency correction {int(ppm):+d} ppm.", None, 3000)

    def _on_direct_changed(self, value: str) -> None:
        dev = self._front_end_device("direct")
        if dev is None:
            return
        if not self._call(dev, "set_direct_sampling", value):
            self._show_device_front_end()
            self._win._show_status_error(
                "The RTL-SDR didn't change its direct sampling (Tools > Error "
                "History says why)."
            )
            return
        self._save_front_end()
        label = _DIRECT_LABELS.get(value, value)
        if value == "off":
            text = "Below 24 MHz the RTL-SDR now tunes through its tuner (no direct sampling)."
        else:
            text = f"Below 24 MHz the RTL-SDR now uses direct sampling ({label})."
        if getattr(dev, "direct_sampling", False):
            text += " In use now."
        self._status(text, None, 5000)

    def _on_bias_changed(self, on: bool) -> None:
        dev = self._front_end_device("bias_tee")
        if dev is None:
            return
        name = _kind_name(self.kind())
        if not self._call(dev, "set_bias_tee", bool(on)):
            self._show_device_front_end()
            self._win._show_status_error(
                f"{name[:1].upper()}{name[1:]} didn't switch its bias tee "
                + ("on" if on else "off")
                + " (an older driver library, or no bias tee: Tools > Error "
                "History says why)."
            )
            return
        self._save_front_end()
        if on:
            self._status(
                "Bias tee on: the antenna socket is powered. Only with an LNA or "
                "an active antenna connected.",
                "warning",
                8000,
            )
        else:
            self._status("Bias tee off: the antenna socket is no longer powered.")

    # ----------------------------------------------------------- sample rate

    def rate_blocked(self) -> str:
        """Why the sample rate can't be changed now ("" when it can)."""
        win = self._win
        if win._device is None:
            return (
                "Connect a device (Device > Connect...) or use the demo device to "
                "choose its sample rate."
            )
        if win._playback.active:
            return (
                f"The recording sets the sample rate "
                f"({format_rate(win._device_sample_rate())}) while it plays. "
                "Close it (File > Close Recording) to change the device's."
            )
        if win._recording:
            return (
                "Stop recording to change the sample rate: one I/Q recording "
                "holds one rate."
            )
        if win._tx_busy():
            return "Wait until the transmission ends to change the sample rate."
        return ""

    def sync_rates(self) -> None:
        """The Receiver group's Sample rate and the status bar's rate."""
        win = self._win
        dev = win._device
        kind = device_kind(dev)
        rate = win._device_sample_rate()
        if dev is None:
            self._panel.set_sample_rates((), 0.0, False, self.rate_blocked())
        elif kind == "playback":
            self._panel.set_sample_rates((rate,), rate, False, self.rate_blocked())
        else:
            self._panel.set_sample_rates(sample_rates_for(kind), rate, True)
        label = win._rate_label
        if dev is None:
            label.setToolTip(
                "Sample rate (the span). Connect a device to choose it, here or "
                "in Receiver > Sample rate."
            )
        elif kind == "playback":
            label.setToolTip(self.rate_blocked())
        else:
            label.setToolTip(
                f"Sample rate {format_rate(rate)}: the span shown. Click to choose "
                "another (also Receiver > Sample rate and Device > Sample Rate)."
            )

    def choose_rate(self, rate: float) -> bool:
        """A sample rate picked in the Receiver group or a rate menu: apply
        it live (or say why not, and show the rate in use again)."""
        ok = bool(self._win.set_sample_rate(rate))
        if not ok:
            self.sync_rates()
        return ok

    def rate_applied(self, rate: float) -> None:
        """The window's set_sample_rate applied ``rate``: remember it for
        the device kind and say what it means."""
        kind = self.kind()
        if kind in REMEMBERED_KINDS:
            save_rate(self._settings, kind, rate)
        text = f"Sample rate {format_rate(rate)}: the span is {rate / 1e6:g} MHz."
        tone = None
        if rate < ADSB_MIN_RATE and self._adsb_selected():
            text += " ADS-B needs at least 2 MS/s."
            tone = "warning"
        self._status(text, tone, 5000)

    def _adsb_selected(self) -> bool:
        decoder = getattr(self._win, "_decoder", None)
        name = str(getattr(decoder, "name", "") or "")
        return name.upper().startswith("ADS-B")

    def remembered_rate(self, kind: str) -> float:
        return load_rate(self._settings, kind)

    def _fill_rate_menu(self) -> None:
        menu = self.rate_menu
        menu.clear()
        win = self._win
        blocked = self.rate_blocked()
        dev = win._device
        if dev is None or win._playback.active:
            note = menu.addAction(blocked)
            note.setEnabled(False)
            return
        current = win._device_sample_rate()
        group = QActionGroup(menu)
        group.setExclusive(True)
        rates = list(sample_rates_for(device_kind(dev)))
        if not any(abs(r - current) < 1.0 for r in rates):
            rates = sorted(rates + [current])
        for rate in rates:
            action = QAction(rate_choice_label(rate), menu)
            action.setCheckable(True)
            action.setChecked(abs(rate - current) < 1.0)
            if rate < ADSB_MIN_RATE:
                action.setToolTip(ADSB_RATE_TIP)
            action.triggered.connect(lambda _c=False, r=rate: self.choose_rate(r))
            group.addAction(action)
            menu.addAction(action)
        if blocked:
            menu.addSeparator()
            note = menu.addAction(blocked)
            note.setEnabled(False)

    def popup_rate_menu(self, anchor: "QWidget") -> None:
        """Show the rate menu above ``anchor`` (the status bar's rate)."""
        menu = self.rate_menu
        self._fill_rate_menu()
        height = menu.sizeHint().height()
        pos = anchor.mapToGlobal(QPoint(0, 0))
        menu.popup(QPoint(pos.x(), max(0, pos.y() - height)))

    # ---------------------------------------------------------------- dialog

    def dialog_preferences(self) -> Tuple[Dict[str, float], Dict[str, FrontEnd]]:
        """For Device > Connect...: the rate to preselect and the front end
        to show for each device kind."""
        win = self._win
        dev = win._device
        current_kind, current_rate = device_kind(dev), 0.0
        if dev is not None:
            current_rate = win._device_sample_rate()
        if win._playback.active:
            # Not the recording's rate: the device's from before it.
            prev = getattr(win._playback, "_previous", None)
            current_kind = device_kind(getattr(prev, "device", None))
            current_rate = win._playback.previous_rate()
        preferred = getattr(win, "_preferred_rate", None)
        rates: Dict[str, float] = {}
        fronts: Dict[str, FrontEnd] = {}
        for kind in REMEMBERED_KINDS:
            if kind == current_kind and current_rate:
                rate = current_rate
            elif preferred:
                rate = float(preferred)
            else:
                rate = load_rate(self._settings, kind) or current_rate
            if rate:
                rates[kind] = float(rate)
            if kind == current_kind and dev is not None and kind != "playback":
                fronts[kind] = self._front_end_of(dev, kind)
            else:
                fronts[kind] = load_front_end(self._settings, kind)
        return rates, fronts

    def remember_dialog_choice(self, kind: str, front: Optional[FrontEnd]) -> None:
        """The Connect dialog opened a ``kind`` device with ``front``."""
        if kind in REMEMBERED_KINDS and front is not None:
            save_front_end(self._settings, kind, front)


# --------------------------------------------------------------------------- #
# Device > Scan for Devices
# --------------------------------------------------------------------------- #


class ScanResultsDialog(QDialog if HAS_PYQT6 else object):
    """What Device > Scan for Devices found, with what to do next: the
    devices (Connect...), or the drivers to install (a selectable command
    block) and Use Demo Device / Connect...."""

    #: What the user chose: "connect", "demo" or "" (Close).
    choice = ""

    def __init__(
        self,
        devices: Sequence[str],
        missing: Sequence[Tuple[str, str]],
        parent: Any = None,
    ):
        super().__init__(parent)
        self.choice = ""
        found = bool(devices)
        self.setWindowTitle("Devices Found" if found else "No Devices Found")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        if found:
            count = len(devices)
            text = f"Found {count} SDR device{'s' if count != 1 else ''}:\n"
            text += "\n".join(f"  •  {name}" for name in devices)
            text += "\n\nConnect... opens one."
        else:
            text = (
                "No SDR devices were found. Plug in an RTL-SDR or HackRF One and "
                "scan again, or try the app with the demo device's simulated "
                "signals."
            )
        self._summary = QLabel(text)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._commands: Optional[QPlainTextEdit] = None
        if missing:
            note = QLabel(
                "These drivers aren't installed, so their devices can't be found. "
                "Install them with:"
            )
            note.setWordWrap(True)
            layout.addWidget(note)
            commands = QPlainTextEdit()
            commands.setReadOnly(True)
            commands.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            commands.setPlainText(missing_driver_text(missing))
            commands.setAccessibleName("Install commands")
            commands.setToolTip("Select and copy a command (Ctrl+C)")
            set_role(commands, "terminal")
            text_lines = commands.toPlainText().splitlines() or [""]
            metrics = commands.fontMetrics()
            commands.setFixedHeight(
                int(metrics.lineSpacing() * (len(text_lines) + 0.8) + 14)
            )
            longest = max(metrics.horizontalAdvance(line) for line in text_lines)
            commands.setMinimumWidth(min(720, longest + 36))
            layout.addWidget(commands)
            self._commands = commands

        buttons = QDialogButtonBox()
        self._connect_btn = QPushButton("&Connect...")
        self._connect_btn.setToolTip("Open Device > Connect... to choose a device")
        buttons.addButton(self._connect_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        self._demo_btn: Optional[QPushButton] = None
        if not found:
            self._demo_btn = QPushButton("Use &Demo Device")
            self._demo_btn.setToolTip(
                "Explore the app with simulated signals (no hardware needed)"
            )
            buttons.addButton(self._demo_btn, QDialogButtonBox.ButtonRole.AcceptRole)
            set_role(self._demo_btn, "primary")
            self._demo_btn.setDefault(True)
        else:
            set_role(self._connect_btn, "primary")
            self._connect_btn.setDefault(True)
        close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.clicked.connect(self.reject)
        self._connect_btn.clicked.connect(lambda: self._choose("connect"))
        if self._demo_btn is not None:
            self._demo_btn.clicked.connect(lambda: self._choose("demo"))
        layout.addWidget(buttons)

    def _choose(self, choice: str) -> None:
        self.choice = choice
        self.accept()

    def commands_text(self) -> str:
        return self._commands.toPlainText() if self._commands is not None else ""


def missing_driver_text(missing: Sequence[Tuple[str, str]]) -> str:
    """One line per missing driver, ``RTL-SDR:  python -m pip install ...``,
    a note in brackets after a command ("(needs the libhackrf system
    library)") on a line of its own under it, so each command is whole on
    its line, ready to copy."""
    width = max((len(label) for label, _install in missing), default=0) + 3
    lines = []
    for label, install in missing:
        command, note = install, ""
        if install.endswith(")") and " (" in install:
            command, note = install.rsplit(" (", 1)
            note = "(" + note
        lines.append(f"{label + ':':<{width}}{command}")
        if note:
            lines.append(" " * width + note)
    return "\n".join(lines)


def confirm_switch_to_demo(window: Any) -> bool:
    """Use Demo Device while a recording runs on hardware: ask first (the
    recording would go on with simulated samples). True to go ahead (the
    recordings are stopped), False to keep the device."""
    iq = bool(window._recording)
    audio = window._audio_take is not None
    if not (iq or audio):
        return True
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Recording in Progress")
    box.setText(
        f"Stop recording and switch from {window._device_display_name()} to the "
        "demo device?"
    )
    detail = "The recording would otherwise go on with simulated signals."
    if iq:
        detail += (
            " The I/Q samples recorded so far stay in the Recording group, to save."
        )
    if audio:
        detail += " The audio recording is saved as it is."
    box.setInformativeText(detail)
    go = box.addButton("Stop and &Switch", QMessageBox.ButtonRole.AcceptRole)
    set_role(go, "primary")
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    window._exec_dialog(box)
    if box.clickedButton() is not go:
        return False
    if window._recording:
        window._toggle_recording(False)
    if window._audio_take is not None:
        window._stop_audio_recording()
    return True


__all__ = [
    "PLAYBACK_GAIN",
    "REMEMBERED_KINDS",
    "DeviceControls",
    "ScanResultsDialog",
    "confirm_switch_to_demo",
    "missing_driver_text",
]
