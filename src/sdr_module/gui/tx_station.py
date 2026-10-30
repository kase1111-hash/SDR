"""
The main window's transmit side.

:class:`StationTx` owns the :class:`~.transmitter.Transmitter` and connects it
to the window: the Ham ID panel (what Send ID Now would do, its progress and
Stop, the TX power), the red ON AIR badge in the toolbar, Radio > Send
Station ID / Stop Transmitting, the first-use confirmation, the receiver
(stopped for the transmission, resumed after it) and the status bar (every
message replaced by the outcome, so none goes stale).

Anything the app transmits goes through :meth:`StationTx.transmit` (Transmit
Text included): it checks the job, asks the first time, centers the HackRF on
the frequency and adds the station IDs the Ham ID panel's Automatic ID asks
for.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt
    from PyQt6.QtWidgets import QCheckBox, QLabel, QMessageBox

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

from .themes import set_role
from .transmitter import (
    AutoIdPolicy,
    Transmitter,
    TxJob,
    TxPower,
    TxResult,
    can_transmit,
    format_dbm_power,
    format_mhz,
    legality_problem,
)

logger = logging.getLogger(__name__)

# The rate a station ID is generated and sent at: the HackRF's lowest, so a
# narrow ID costs the least (the HackRF is put at it for the ID).
ID_SAMPLE_RATE = 2e6
# Peak deviation of the FM tone ID (narrow FM).
ID_DEVIATION_HZ = 2500.0

# Settings keys (tx/ prefix).
KEY_GAIN = "tx/gain_db"
KEY_AMP = "tx/amp"
KEY_CONFIRM = "tx/confirm"


class StationTx(QObject if HAS_PYQT6 else object):  # type: ignore[misc]
    """The window's transmit controller (see the module docstring)."""

    def __init__(self, window: Any):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(window)
        self._win = window
        self.engine = Transmitter(window)
        settings = window._settings
        self.power = TxPower(
            gain_db=settings.get_int(KEY_GAIN, TxPower().gain_db),
            amp=settings.get_bool(KEY_AMP, False),
        ).clamped()
        self._resume_rx = False
        self._badge: Optional[QLabel] = None
        self._badge_action: Any = None
        self._send_action: Any = None
        self._stop_action: Any = None
        # The last outcome, shown on the TX line until something it depends
        # on changes: (inputs it was for, text, tone).
        self._result_line: Optional[Tuple[tuple, str, Optional[str]]] = None
        self.engine.started.connect(self._on_started)
        self.engine.progress.connect(self._on_progress)
        self.engine.finished.connect(self._on_finished)

    # ------------------------------------------------------------ building

    def add_menu_actions(self, menu: Any) -> None:
        """Radio > Send Station ID and Stop Transmitting."""
        win = self._win
        self._send_action = win._add_action(
            menu,
            "Send Station &ID",
            self.send_station_id,
            tip="Transmit your callsign in Morse on the tuned frequency (Ham "
            "Radio > Ham ID: callsign, speed, how it is sent, TX power)",
        )
        self._stop_action = win._add_action(
            menu,
            "Stop &Transmitting",
            self.stop,
            tip="End the transmission on the air now",
        )
        self._stop_action.setEnabled(False)

    def add_toolbar_badge(self, toolbar: Any) -> None:
        """The red ON AIR badge (shown only while transmitting; a click
        stops the transmission)."""
        badge = QLabel("ON AIR")
        set_role(badge, "badge", "danger")
        badge.setAccessibleName("On the air")
        badge.setToolTip("Transmitting. Click to stop.")
        badge.setCursor(Qt.CursorShape.PointingHandCursor)
        badge.installEventFilter(self)
        self._badge = badge
        self._badge_action = toolbar.addWidget(badge)
        self._badge_action.setVisible(False)

    def connect_panel(self) -> None:
        """Wire the Ham ID panel (after the window built it)."""
        panel = self._panel()
        if panel is None:
            return
        panel.set_tx_power(self.power.gain_db, self.power.amp)
        panel.stop_requested.connect(self.stop)
        panel.tx_power_changed.connect(self._on_power_changed)
        panel.settings_changed.connect(lambda _settings: self.refresh())
        panel.callsign_changed.connect(lambda _text: self.refresh())

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if (
            obj is self._badge
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.stop()
            return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------ state

    @property
    def busy(self) -> bool:
        return self.engine.busy

    def _panel(self) -> Any:
        return getattr(self._win, "_callsign_panel", None)

    def _qrp(self) -> Any:
        return getattr(self._win, "_qrp_panel", None)

    def _device(self) -> Any:
        return getattr(self._win, "_device", None)

    def device_problem(self) -> Optional[str]:
        """Why the connected device can't transmit (None: it can)."""
        win = self._win
        dev = self._device()
        if dev is None:
            return "no device is connected. Connect a HackRF One (Device > Connect...)"
        if getattr(win, "_demo_mode", False):
            return "the demo device can't transmit. Connect a HackRF One"
        if not can_transmit(dev):
            name = (
                win._device_model_name() if hasattr(win, "_device_model_name") else ""
            )
            name = name or "this device"
            return f"the {name} can only receive. Connect a HackRF One"
        if getattr(win, "_lo_lock", None) is not None:
            return "a recording is playing back. Stop it to transmit"
        return None

    def id_spec(self, freq_hz: float) -> Tuple[str, str, float, str]:
        """``(emission, mode, occupied bandwidth, summary)`` of the station
        ID at ``freq_hz`` with the Ham ID panel's settings."""
        from ..ham.callsign import (
            EMISSION_MCW,
            choose_id_emission,
            cw_occupied_bandwidth,
            mcw_occupied_bandwidth,
        )

        panel = self._panel()
        pref = panel.emission_preference() if panel is not None else "auto"
        wpm = panel.cw_wpm() if panel is not None else 20
        tone = panel.cw_tone() if panel is not None else 700
        emission = choose_id_emission(freq_hz, pref)
        if emission == EMISSION_MCW:
            return (
                emission,
                "FM",
                mcw_occupied_bandwidth(tone, ID_DEVIATION_HZ),
                f"FM tone, {wpm} WPM",
            )
        return emission, "CW", cw_occupied_bandwidth(wpm), f"CW, {wpm} WPM"

    def power_problem(self, freq_hz: float, mode: str) -> Optional[str]:
        """Why the calculated output is too high (None: it isn't): over the
        QRP panel's power limit, or over the legal limit at ``freq_hz``."""
        qrp = self._qrp()
        if qrp is None:
            return None
        sdr_dbm = self.power.output_dbm(freq_hz)
        try:
            watts = qrp.calculated_output_watts(sdr_dbm)
        except Exception:  # pragma: no cover - defensive
            return None
        limit = qrp.power_limit_watts()
        if limit is not None and not qrp.within(watts, limit):
            return (
                f"the calculated output ({qrp.watts_text(watts)}) is over your "
                f"{qrp.watts_text(limit)} power limit (QRP tab). Lower the TX gain "
                "or raise the limit"
            )
        from ..core import frequency_manager

        legal = frequency_manager.get_power_limit(freq_hz, mode)
        if legal is not None and not qrp.within(watts, legal):
            return (
                f"the calculated output ({qrp.watts_text(watts)}) is over the "
                f"{qrp.watts_text(legal)} legal limit here"
            )
        return None

    def _inputs(self) -> tuple:
        """What the TX line depends on (a change replaces an old outcome)."""
        win = self._win
        panel = self._panel()
        from ..core import frequency_manager

        return (
            round(float(win._current_frequency())),
            id(self._device()),
            bool(getattr(win, "_demo_mode", False)),
            str(frequency_manager.get_license_class().value),
            panel.get_callsign() if panel is not None else "",
            tuple(sorted((panel.get_settings() if panel is not None else {}).items())),
            (self.power.gain_db, self.power.amp),
        )

    def readiness(self) -> Tuple[str, Optional[str], Optional[bool]]:
        """``(TX line, tone, can send)`` for Send ID Now right now."""
        win = self._win
        freq = float(win._current_frequency())
        where = format_mhz(freq)
        problem = self.device_problem()
        if problem is not None:
            return f"Can't send: {problem}.", "warning", False
        emission, mode, width, summary = self.id_spec(freq)
        problem = legality_problem(freq, width, mode, quiet=True)
        if problem is not None:
            return f"Can't send on {where}: {problem}.", "danger", False
        problem = self.power_problem(freq, mode)
        if problem is not None:
            return f"Can't send on {where}: {problem}.", "danger", False
        estimate = format_dbm_power(self.power.output_dbm(freq))
        how = "FM tone ID" if mode == "FM" else "CW ID"
        return (
            f"Will send on {where} · {how} · TX gain {self.power.gain_db} dB "
            f"(about {estimate.split(' (')[0]})",
            "info",
            True,
        )

    def refresh(self) -> None:
        """Bring the Ham ID panel's TX line, Send button and power estimate
        (and the QRP panel's SDR output) up to date."""
        panel = self._panel()
        win = self._win
        freq = float(win._current_frequency())
        self._update_qrp_source(freq)
        if panel is None:
            return
        self._update_power_label(freq)
        if self.busy:
            return  # the panel shows the transmission
        if self._send_action is not None:
            self._send_action.setEnabled(True)
        if self._result_line is not None and self._result_line[0] == self._inputs():
            _inputs, text, tone = self._result_line
            can = self.readiness()[2]
            panel.set_tx_status(text, tone, can)
            return
        self._result_line = None
        text, tone, can = self.readiness()
        panel.set_tx_status(text, tone, can)

    def _update_power_label(self, freq: float) -> None:
        panel = self._panel()
        dbm = self.power.output_dbm(freq)
        text = f"about {format_dbm_power(dbm)}"
        tip = (
            f"Estimated output at the HackRF's antenna port on {format_mhz(freq)} "
            f"({self.power.describe(freq)}). Units differ by a few dB: "
            "measure it before adding an amplifier."
        )
        problem = self.power_problem(freq, self.id_spec(freq)[1])
        tone = "danger" if problem else None
        if problem:
            tip += f"\n\n{problem[:1].upper()}{problem[1:]}."
        if not can_transmit(self._device()):
            tip += "\n\n(Only a HackRF One transmits.)"
        panel.set_power_estimate(text, tone, tip)

    def _update_qrp_source(self, freq: float) -> None:
        """The QRP calculator's SDR output: the HackRF's estimate while one
        is connected (read-only there), else the user's own figure."""
        qrp = self._qrp()
        if qrp is None:
            return
        if can_transmit(self._device()):
            qrp.set_sdr_output(
                self.power.output_dbm(freq),
                f"From Ham ID > TX Power: {self.power.describe(freq)}",
            )
        else:
            qrp.set_sdr_output(None)

    def _on_power_changed(self, gain_db: int, amp: bool) -> None:
        self.power = TxPower(int(gain_db), bool(amp)).clamped()
        settings = self._win._settings
        settings.set(KEY_GAIN, self.power.gain_db)
        settings.set(KEY_AMP, self.power.amp)
        self.refresh()

    # ------------------------------------------------------------ sending

    def make_id_job(
        self, freq_hz: float, rate: float = ID_SAMPLE_RATE
    ) -> Optional[TxJob]:
        """The station ID ("DE W1AW") at ``freq_hz`` with the Ham ID panel's
        settings, or None without a valid callsign."""
        from ..ham.callsign import MorseIdSource

        panel = self._panel()
        if panel is None or not panel.is_callsign_valid():
            return None
        emission, _mode, _width, summary = self.id_spec(freq_hz)
        callsign = panel.get_callsign()
        source = MorseIdSource(
            f"DE {callsign}",
            panel.cw_wpm(),
            rate,
            emission=emission,
            tone_hz=panel.cw_tone(),
            deviation_hz=ID_DEVIATION_HZ,
        )
        return TxJob(
            source,
            rate,
            float(freq_hz),
            source.mode,
            source.occupied_bandwidth_hz,
            f"DE {callsign}",
            summary,
            is_id=True,
            identify=False,
        )

    def _id_factory(self) -> Any:
        """Makes the IDs automatic ID adds to a job (settings taken now, so
        the factory can run on the transmit thread)."""
        from ..ham.callsign import MorseIdSource, choose_id_emission

        panel = self._panel()
        if panel is None or not panel.is_callsign_valid():
            return None
        callsign, wpm, tone = panel.get_callsign(), panel.cw_wpm(), panel.cw_tone()
        pref = panel.emission_preference()

        def make(job: TxJob) -> TxJob:
            emission = choose_id_emission(job.center_hz, pref)
            source = MorseIdSource(
                f"DE {callsign}",
                wpm,
                job.sample_rate,
                emission=emission,
                tone_hz=tone,
                deviation_hz=ID_DEVIATION_HZ,
            )
            return TxJob(
                source,
                job.sample_rate,
                job.center_hz,
                source.mode,
                source.occupied_bandwidth_hz,
                f"DE {callsign}",
                is_id=True,
                identify=False,
            )

        return make

    def _auto_id_policy(self) -> Optional[AutoIdPolicy]:
        panel = self._panel()
        if panel is None:
            return None
        values = panel.auto_id_settings()
        return AutoIdPolicy(values["at_start"], values["at_end"], values["interval_s"])

    def send_station_id(self) -> bool:
        """Send ID Now: the station ID on the tuned frequency."""
        win = self._win
        panel = self._panel()
        if self.busy:
            return False
        if panel is None or not panel.is_callsign_valid():
            win._show_status_message(
                "ID not sent: enter your callsign in Ham Radio > Ham ID first.",
                "warning",
                6000,
            )
            win._show_panel("ham_id")
            return False
        freq = float(win._current_frequency())
        job = self.make_id_job(freq)
        if job is None:  # pragma: no cover - checked above
            return False
        return self.transmit(job)

    def transmit(self, job: TxJob, confirm: bool = True) -> bool:
        """Send ``job`` (Send ID Now, Transmit Text, ...): check it, ask the
        first time (``confirm``), center the HackRF on ``job.center_hz``,
        stop the receiver, and start. False, with the reason in the status
        bar and on the Ham ID panel's TX line, when it isn't sent."""
        win = self._win
        if self.busy:
            win._show_status_message(
                "Already transmitting: wait for it to end or press Stop.", "warning"
            )
            return False
        what = "ID" if job.is_id else job.description
        problem = self.device_problem()
        id_jobs = []
        policy = None if job.is_id else self._auto_id_policy()
        make_id = None if job.is_id or not job.identify else self._id_factory()
        if problem is None and make_id is not None and policy is not None:
            sample = make_id(job)
            if sample is not None:
                id_jobs.append(sample)
        if problem is None:
            problem = self.engine.check(job, self._device(), id_jobs)
        if problem is None:
            problem = self.power_problem(job.center_hz, job.mode)
        if problem is not None:
            self._refuse(what, job.center_hz, problem)
            return False
        if confirm and not self.confirm(job):
            win._show_status_message(f"{what} not sent: cancelled.", None, 4000)
            return False
        # The HackRF transmits at its center (the LO): put it on the
        # frequency first, so it sends where it was checked.
        if abs(win._lo_frequency() - job.center_hz) >= 0.5 and not win._tune_to(
            job.center_hz, recenter=True
        ):
            return False  # (the status bar says why it couldn't tune)
        self._resume_rx = bool(win._is_running)
        if self._resume_rx:
            win._stop_acquisition()
        started = self.engine.start(
            job,
            self._device(),
            self.power,
            policy=policy,
            make_id=make_id,
        )
        if not started:
            self._refuse(what, job.center_hz, self.engine.last_refusal)
            if self._resume_rx:
                self._resume_rx = False
                win._start_acquisition()
            return False
        return True

    def _refuse(self, what: str, freq_hz: float, problem: str) -> None:
        """Say why nothing was sent (status bar and the TX line)."""
        win = self._win
        text = f"{what} not sent on {format_mhz(freq_hz)}: {problem}."
        win._show_status_message(text, "danger", 8000)
        logger.info(text)
        panel = self._panel()
        if panel is not None:
            self._result_line = (self._inputs(), f"Can't send: {problem}.", "danger")
            panel.set_tx_status(
                f"Can't send: {problem}.", "danger", self.readiness()[2]
            )

    def confirm(self, job: TxJob) -> bool:
        """The first-use question ("Transmit DE W1AW on 146.520 MHz (FM
        tone, 20 WPM, about 1 mW)?"); "Don't ask again" skips it from then
        on. True to transmit."""
        settings = self._win._settings
        if not settings.get_bool(KEY_CONFIRM, True):
            return True
        estimate = format_dbm_power(self.power.output_dbm(job.center_hz)).split(" (")[0]
        how = job.summary or job.mode
        box = QMessageBox(self._win)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Transmit?")
        box.setText(
            f"Transmit {job.description} on {format_mhz(job.center_hz)} "
            f"({how}, about {estimate})?"
        )
        box.setInformativeText(
            "The HackRF will transmit for about "
            f"{job.duration_s:.1f} s. Make sure an antenna or a dummy load is "
            "connected and that your license allows this frequency. Stop ends "
            "it at any time."
        )
        check = QCheckBox("Don't ask again")
        check.setToolTip("Transmit without asking from now on (the checks still apply)")
        box.setCheckBox(check)
        send = box.addButton("&Transmit", QMessageBox.ButtonRole.AcceptRole)
        set_role(send, "danger")
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        self._win._exec_dialog(box)
        accepted = box.clickedButton() is send
        if accepted and check.isChecked():
            settings.set(KEY_CONFIRM, False)
        return accepted

    def stop(self) -> None:
        """Stop the transmission on the air (Stop, the badge, the menu)."""
        if self.busy:
            self.engine.stop()
            self._win._show_status_message(
                "Stopping the transmission…", "warning", 3000
            )

    def shutdown(self) -> None:
        """Stop and wait (the device is about to close or change)."""
        if self.busy:
            self._resume_rx = False
            self.engine.shutdown()

    # ------------------------------------------------------------ engine

    def _on_started(self, job: TxJob) -> None:
        win = self._win
        where = format_mhz(job.center_hz)
        if self._badge_action is not None:
            self._badge_action.setVisible(True)
            self._badge.setToolTip(
                f"On the air: {job.description} on {where}. Click to stop."
            )
        if self._stop_action is not None:
            self._stop_action.setEnabled(True)
        if self._send_action is not None:
            self._send_action.setEnabled(False)
        panel = self._panel()
        if panel is not None:
            panel.set_transmitting(True, f"Sending {job.description} on {where}…")
        win._show_status_message(
            f"On air: {job.description} on {where} ({job.summary or job.mode}). "
            "Stop ends it.",
            "danger",
            int((job.duration_s + 10.0) * 1000),
        )

    def _on_progress(self, done_s: float, total_s: float) -> None:
        panel = self._panel()
        if panel is None:
            return
        panel.set_tx_progress(done_s, total_s)
        next_id = self.engine.next_id_in()
        policy = self._auto_id_policy()
        interval = policy.interval_s if policy is not None else None
        panel.set_id_countdown(next_id, interval)

    def _on_finished(self, result: TxResult) -> None:
        win = self._win
        if self._badge_action is not None:
            self._badge_action.setVisible(False)
        if self._stop_action is not None:
            self._stop_action.setEnabled(False)
        panel = self._panel()
        if panel is not None:
            panel.set_transmitting(False)
        stamp = time.strftime("%H:%M")
        if result.status == "sent":
            win._show_status_message(result.message, "success", 6000)
            line = f"✓ {result.message} at {stamp}."
            tone: Optional[str] = "success"
        elif result.status == "stopped":
            win._show_status_message(result.message, "warning", 6000)
            line = f"{result.message}."
            if result.job.is_id or result.ids_sent == 0:
                line += " Send your ID again to finish the contact."
            tone = "warning"
        else:
            win._show_status_error(result.message)
            line = f"✗ {result.message}."
            tone = "danger"
        # The receiver gets its own filter back (the transmitter restored
        # the rate, the frequency and the amplifier).
        try:
            win._apply_baseband_filter()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not reapply the baseband filter: {e}")
        win._refresh_state_ui()
        if self._resume_rx and win._device is not None:
            self._resume_rx = False
            win._start_acquisition()
        self._resume_rx = False
        self._result_line = (self._inputs(), line, tone)
        self.refresh()


__all__ = ["ID_DEVIATION_HZ", "ID_SAMPLE_RATE", "StationTx"]
