"""Transmitting from the window (gui/tx_station.py): Send ID Now through the
transmit engine, the Ham ID panel's TX line and power, the first-use
confirmation, the ON AIR badge, Stop, status messages that never go stale,
automatic ID and the guards while on the air.

A HackRFDevice runs on the threaded fake ``python_hackrf`` backend
(tests/hackrf_fakes.py); the demo device must refuse.
"""

from __future__ import annotations

import os
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.hackrf_fakes import analyse_cw, analyse_mcw, open_hackrf  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


class _Patch:
    """monkeypatch-like helper for unittest (undone in tearDown)."""

    def __init__(self):
        self._undo = []

    def setitem(self, mapping, key, value):
        missing = object()
        old = mapping.get(key, missing)
        self._undo.append((mapping, key, old, missing))
        mapping[key] = value

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name), None))
        setattr(obj, name, value)

    def undo(self):
        while self._undo:
            target, key, old, missing = self._undo.pop()
            if isinstance(target, dict):
                if old is missing:
                    target.pop(key, None)
                else:
                    target[key] = old
            else:
                setattr(target, key, old)


class _TxCase(_WindowTestCase):
    """A window with in-memory settings; helpers to attach a fake HackRF."""

    license = "technician"

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        from sdr_module.core.frequency_manager import LicenseClass

        self.patch = _Patch()
        self.addCleanup(self.patch.undo)
        self.win._control_panel.set_license_class(LicenseClass(self.license))
        self.addCleanup(
            lambda: self.win._control_panel.set_license_class(LicenseClass.NONE)
        )
        self.panel = self.win._callsign_panel
        self.station = self.win._tx_station
        # No modal box may open (the confirmation is tested on its own).
        self.boxes = []
        self.win._exec_dialog = lambda box: self.boxes.append(box) or 0
        self.store.values["tx/confirm"] = False

    def tearDown(self):
        if hasattr(self, "station"):
            self.station.shutdown()
        super().tearDown()

    def hackrf(self, rate=2.4e6, freq=146.52e6, **fake):
        """Connect a HackRF (fake backend) as Device > Connect does."""
        if self.win._current_frequency() != freq:
            self.win.set_frequency(freq)
        dev, backend = open_hackrf(self.patch, rate=rate, **fake)
        self.win._device = dev
        self.win._demo_mode = False
        self.win._apply_controls_to_device()
        self.win._refresh_state_ui()
        self.addCleanup(self._release)
        return dev, backend

    def _release(self):
        self.station.shutdown()
        if self.win._device is not None:
            self.win._release_device()

    def pump(self, seconds, until=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            QApplication.processEvents()
            if until is not None and until():
                return True
            time.sleep(0.01)
        return until() if until is not None else True

    def wait_done(self, timeout=30.0):
        engine = self.station.engine
        self.assertTrue(engine.wait(timeout))
        QApplication.processEvents()

    def status(self):
        return self.win._message_label.text(), self.win._message_label.property("tone")


class TestTxLine(_TxCase):
    """UX-01: what Send ID Now would do, or why it can't, under the callsign."""

    def test_callsign_line_and_tx_line_are_separate(self):
        self.panel.set_callsign("W1AW")
        self.assertEqual(
            self.panel._status_label.text(),
            "✓ Callsign OK · DE W1AW = 4.0 s at 20 WPM",
        )
        self.assertEqual(self.panel._status_label.property("tone"), "success")
        self.assertEqual(
            self.panel.tx_status_text(),
            "Can't send: no device is connected. Connect a HackRF One "
            "(Device > Connect...).",
        )
        self.assertFalse(self.panel._id_now_btn.isEnabled())

    def test_demo_device_refuses_with_its_reason(self):
        self.panel.set_callsign("W1AW")
        self.win._start_demo_mode()
        self.addCleanup(self.win._release_device)
        text = self.panel.tx_status_text()
        self.assertEqual(
            text, "Can't send: the demo device can't transmit. Connect a HackRF One."
        )
        self.assertEqual(self.panel._tx_label.property("tone"), "warning")
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertEqual(self.panel._id_now_btn.toolTip(), text)
        # Radio > Send Station ID says so too, without a dialog.
        self.win._on_callsign_id_requested()
        self.assertEqual(
            self.status(),
            (
                "ID not sent on 100.100 MHz: the demo device can't transmit. "
                "Connect a HackRF One.",
                "danger",
            ),
        )
        self.assertEqual(self.boxes, [])
        self.assertFalse(self.station.busy)

    def test_without_a_callsign_the_line_says_what_it_would_do(self):
        self.hackrf()
        self.assertTrue(
            self.panel.tx_status_text().startswith(
                "With a callsign, Send ID Now sends on 146.520 MHz · FM tone ID"
            )
        )
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertFalse(self.station.send_station_id())
        self.assertEqual(
            self.status(),
            (
                "ID not sent: enter your callsign in Ham Radio > Ham ID first.",
                "warning",
            ),
        )

    def test_an_unidentified_amateur_transmission_is_refused(self):
        """Without a callsign the IDs it needs can't be sent."""
        from sdr_module.gui.transmitter import TxJob

        self.hackrf()
        carrier = np.full(1000, 0.5, np.complex64)
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertFalse(self.win._transmit_job(job))
        self.assertIn("an amateur transmission must be identified", self.status()[0])

    def test_no_transmitting_while_a_recording_plays_back(self):
        self.panel.set_callsign("W1AW")
        self.hackrf()
        self.win._lock_lo(146.52e6, 2.4e6, "the recording")
        self.assertEqual(
            self.panel.tx_status_text(),
            "Can't send: a recording is playing back. Stop it to transmit.",
        )
        self.assertFalse(self.station.send_station_id())
        self.win._unlock_lo()
        self.assertTrue(self.panel.tx_status_text().startswith("Will send"))

    def test_license_is_checked_at_the_tuned_frequency(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_callsign("W1AW")
        self.hackrf(freq=100.1e6)
        self.win._control_panel.set_license_class(LicenseClass.NONE)
        self.assertEqual(
            self.panel.tx_status_text(),
            "Can't send on 100.100 MHz: your license (None) allows CB/MURS/FRS only.",
        )
        self.assertEqual(self.panel._tx_label.property("tone"), "danger")
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        # Tuning to where it is allowed updates the line.
        self.win._control_panel.set_license_class(LicenseClass.TECHNICIAN)
        self.win.set_frequency(146.52e6)
        text = self.panel.tx_status_text()
        self.assertTrue(text.startswith("Will send on 146.520 MHz · FM tone ID"), text)
        self.assertIn("TX gain 20 dB", text)
        self.assertTrue(self.panel._id_now_btn.isEnabled())

    def test_the_id_is_cw_on_hf_and_an_fm_tone_above(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_callsign("W1AW")
        self.hackrf(freq=146.52e6)
        self.assertIn("· FM tone ID ·", self.panel.tx_status_text())
        self.win._control_panel.set_license_class(LicenseClass.AMATEUR_EXTRA)
        self.win.set_frequency(14.05e6)
        self.assertIn("· CW ID ·", self.panel.tx_status_text())
        # An FM tone in a CW-only segment is refused, with the reason.
        index = self.panel._emission_combo.findData("mcw")
        self.panel._emission_combo.setCurrentIndex(index)
        self.assertEqual(
            self.panel.tx_status_text(),
            "Can't send on 14.050 MHz: FM isn't allowed in the 20m CW General "
            "segment (CW and DATA only).",
        )
        self.assertEqual(self.panel.get_settings()["emission"], "mcw")


class TestSendIdNow(_TxCase):
    def test_sends_without_blocking_and_comes_back(self):
        """HAM-02/HAM-04/UX-02: the window keeps working, the ID goes out at
        2 MS/s with progress, the ON AIR badge and Stop, and the receiver
        resumes at its own rate."""
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(rate=10e6, pace=True)
        self.win._start_acquisition()
        self.assertTrue(self.win._is_running)
        badge = self.station._badge_action
        self.assertFalse(badge.isVisible())

        t0 = time.perf_counter()
        self.panel._id_now_btn.click()
        self.assertLess(time.perf_counter() - t0, 0.3)
        self.assertTrue(self.station.busy)
        self.assertTrue(badge.isVisible())
        self.assertFalse(self.win._is_running)  # half duplex: RX stopped
        self.assertEqual(self.panel._id_now_btn.text(), "Stop")
        self.assertTrue(self.panel._tx_progress.isVisibleTo(self.panel))
        self.assertTrue(self.station._stop_action.isEnabled())
        text, tone = self.status()
        self.assertTrue(text.startswith("On air: DE W1AW on 146.520 MHz (FM tone"))
        self.assertEqual(tone, "danger")

        # While on the air: no tuning, no receiving (and the window answers).
        self.assertFalse(self.win.set_frequency(146.94e6))
        self.assertIn("Can't tune while transmitting", self.status()[0])
        self.win._start_acquisition()
        self.assertFalse(self.win._is_running)
        self.assertTrue(self.pump(1.5, lambda: self.panel._tx_progress.value() > 0))

        self.assertTrue(self.pump(10, lambda: not self.station.busy))
        self.assertFalse(badge.isVisible())
        text, tone = self.status()
        self.assertTrue(text.startswith("Sent DE W1AW on 146.520 MHz ("), text)
        self.assertEqual(tone, "success")
        self.assertTrue(self.panel.tx_status_text().startswith("✓ Sent DE W1AW"))
        self.assertTrue(self.win._is_running)  # receiving again
        self.assertEqual(dev.state.sample_rate, 10e6)
        self.assertEqual(backend.start_rate(), 2e6)
        on_air = analyse_mcw(backend.tx_iq(32), 2e6 / 32)
        self.assertEqual(on_air["text"], "DE W1AW")
        self.assertAlmostEqual(on_air["tone_hz"], 700, delta=5)
        self.win._stop_acquisition()

    def test_stop_ends_it_and_says_so(self):
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)
        self.hackrf(pace=True)
        self.panel._id_now_btn.click()
        self.assertTrue(self.station.busy)
        self.pump(0.4)
        self.panel._id_now_btn.click()  # Stop
        self.wait_done(5)
        self.assertFalse(self.station.busy)
        text, tone = self.status()
        self.assertTrue(text.startswith("Stopped DE W1AW on 146.520 MHz after"), text)
        self.assertEqual(tone, "warning")
        self.assertIn("Send your ID again", self.panel.tx_status_text())
        self.assertEqual(self.panel._id_now_btn.text(), "Send ID Now")

    def test_a_double_click_on_send_doesnt_stop_it(self):
        """Send ID Now becomes Stop under the pointer: the second click of
        a double-click doesn't end the ID the first one started."""
        self.panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        button = self.panel._id_now_btn
        button.click()
        self.assertTrue(self.station.busy)
        button.click()  # the double-click's second click, now on Stop
        QApplication.processEvents()
        self.assertTrue(self.station.busy)
        self.assertEqual(button.text(), "Stop")
        self.pump(self.panel._double_click_s() + 0.05)
        button.click()  # a click of its own stops it
        self.wait_done(5)
        self.assertEqual(self.station.engine.last_result.status, "stopped")

    def test_the_badge_stops_it(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        self.panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        self.station.send_station_id()
        QTest.mouseClick(self.station._badge, Qt.MouseButton.LeftButton)
        self.wait_done(5)
        self.assertEqual(self.station.engine.last_result.status, "stopped")

    def test_hf_cw_id_goes_out_as_a_keyed_carrier(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.win._control_panel.set_license_class(LicenseClass.AMATEUR_EXTRA)
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=14.05e6)
        self.assertTrue(self.station.send_station_id())
        self.wait_done(10)
        self.assertTrue(self.station.engine.last_result.ok)
        on_air = analyse_cw(backend.tx_iq(32), 2e6 / 32)
        self.assertEqual(on_air["text"], "DE W1AW")
        self.assertLess(on_air["carrier_off_level"], 0.01)

    def test_a_failure_replaces_every_earlier_message(self):
        """HAM-13: the outcome replaces "On air" (and nothing modal)."""
        import sdr_module.devices.hackrf as hackrf_mod

        self.panel.set_callsign("W1AW")
        self.hackrf()
        self.patch.setattr(
            hackrf_mod,
            "is_tx_allowed",
            lambda *a, **k: (False, "LICENSE: test says no"),
        )
        self.assertTrue(self.station.send_station_id())
        self.wait_done(5)
        text, tone = self.status()
        self.assertEqual(text, "DE W1AW not sent on 146.520 MHz: test says no")
        self.assertEqual(tone, "danger")
        self.assertFalse(self.panel.is_transmitting())
        self.assertIn("test says no", self.panel.tx_status_text())
        self.assertEqual(self.boxes, [])

    def test_a_refusal_is_said_at_once(self):
        """HAM-13: no "Preparing CW ID" left behind a refusal."""
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_callsign("W1AW")
        self.hackrf(freq=100.1e6)
        self.win._control_panel.set_license_class(LicenseClass.NONE)
        self.assertFalse(self.station.send_station_id())
        self.assertEqual(
            self.status(),
            (
                "ID not sent on 100.100 MHz: your license (None) allows "
                "CB/MURS/FRS only.",
                "danger",
            ),
        )

    def test_closing_the_window_stops_transmitting(self):
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(pace=True)
        self.station.send_station_id()
        self.assertTrue(self.station.busy)
        self.win.close()
        self.assertFalse(self.station.busy)
        self.assertFalse(dev.state.is_transmitting)
        self.assertTrue(backend.closed)


class TestConfirmation(_TxCase):
    def test_first_use_question_and_dont_ask_again(self):
        self.panel.set_callsign("W1AW")
        self.hackrf()
        self.store.values["tx/confirm"] = True
        seen = []

        def answer(box, transmit=True, never_again=True):
            seen.append(box.text())
            if never_again:
                box.checkBox().setChecked(True)
            buttons = {b.text().replace("&", ""): b for b in box.buttons()}
            button = buttons["Transmit"] if transmit else box.escapeButton()
            box.clickedButton = lambda b=button: b  # what exec() would record
            return 0

        self.win._exec_dialog = lambda box: answer(box, transmit=False)
        self.assertFalse(self.station.send_station_id())
        self.assertEqual(self.status(), ("ID not sent: cancelled.", None))
        self.assertTrue(self.store.values["tx/confirm"])  # still asks
        self.win._exec_dialog = answer
        self.assertTrue(self.station.send_station_id())
        self.assertEqual(
            seen[-1],
            "Transmit DE W1AW on 146.520 MHz (FM tone, 20 WPM, about 795 nW)?",
        )
        self.assertFalse(self.store.values["tx/confirm"])
        self.wait_done(10)
        self.win._exec_dialog = lambda box: self.fail("asked again")
        self.assertTrue(self.station.send_station_id())
        self.wait_done(10)


class TestTxPower(_TxCase):
    def test_gain_and_amp_are_used_saved_and_estimated(self):
        """GAP-04: TX gain and amplifier, with an estimated output."""
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf()
        self.assertEqual(self.panel._power_label.text(), "about 795 nW (-31 dBm)")
        self.panel._tx_gain_spin.setValue(37)
        self.panel._amp_check.setChecked(True)
        self.assertEqual(self.store.values["tx/gain_db"], 37)
        self.assertTrue(self.store.values["tx/amp"])
        self.assertEqual(self.panel._power_label.text(), "about 1 mW (0 dBm)")
        self.assertIn("about 1 mW", self.panel.tx_status_text())
        amp_during = []
        self.station.engine.started.connect(lambda _j: amp_during.append(backend.amp))
        self.station.send_station_id()
        self.wait_done(10)
        self.assertEqual(backend.txvga, 37)
        self.assertEqual(amp_during, [True])
        self.assertFalse(dev.amp_enabled)  # back off for receiving

    def test_qrp_takes_the_hackrf_output_and_its_limit_is_honoured(self):
        self.panel.set_callsign("W1AW")
        self.hackrf()
        qrp = self.win._qrp_panel
        spin = qrp._amp_calc._input_spin
        self.assertTrue(spin.isReadOnly())
        self.assertAlmostEqual(spin.value(), -31.0, delta=0.1)
        # The QRP kit at full drive puts out 5 W; a 1 W limit refuses it.
        self.panel._tx_gain_spin.setValue(47)
        self.panel._amp_check.setChecked(True)
        qrp._amp_calc.apply_preset("HackRF → 5 W QRP kit")
        qrp._set_quick_limit(1.0)
        text = self.panel.tx_status_text()
        self.assertTrue(text.startswith("Can't send on 146.520 MHz: the calculated"))
        self.assertIn("over your 1 W power limit (QRP tab)", text)
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertEqual(self.panel._power_label.property("tone"), "danger")
        self.assertFalse(self.station.send_station_id())
        qrp._limit_check.setChecked(False)
        self.assertTrue(self.panel._id_now_btn.isEnabled())
        # The refusal doesn't linger once the limit no longer applies.
        self.assertTrue(self.panel.tx_status_text().startswith("Will send"))
        # Without a HackRF the field is the user's again.
        self.win._release_device()
        self.win._refresh_state_ui()
        self.assertFalse(spin.isReadOnly())
        self.assertEqual(spin.value(), 0.0)

    def test_the_general_amateur_limit_is_honoured(self):
        """Where a segment names no limit, the 1500 W general amateur limit
        the QRP tab shows (red) refuses the transmission too."""
        self.panel.set_callsign("W1AW")
        self.hackrf()
        qrp = self.win._qrp_panel
        calc = qrp._amp_calc
        self.panel._tx_gain_spin.setValue(47)
        self.panel._amp_check.setChecked(True)
        for check, gain, ceiling in (
            (calc._driver_check, calc._driver_spin, calc._driver_max_spin),
            (calc._pa_check, calc._pa_spin, calc._pa_max_spin),
        ):
            check.setChecked(True)
            gain.setValue(40)
            ceiling.setValue(62)
        self.assertIn("is over it", qrp._legal_label.text())  # 1585 W
        text = self.panel.tx_status_text()
        self.assertTrue(text.startswith("Can't send on 146.520 MHz"), text)
        self.assertIn("over the 1500 W legal limit here", text)
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertFalse(self.station.send_station_id())
        self.assertFalse(self.station.busy)

    def test_qrp_shows_the_estimate_the_ham_id_tab_shows(self):
        self.hackrf()
        self.assertEqual(self.panel._power_label.text(), "about 795 nW (-31 dBm)")
        display = self.win._qrp_panel._power_display
        self.assertEqual(display._watts_label.text(), "795 nW")


class TestAutoId(_TxCase):
    def test_a_transmission_gets_its_ids(self):
        """HAM-08/GAP-12: Automatic ID drives real IDs for the app's other
        transmissions (start and end here)."""
        from sdr_module.gui.transmitter import TxJob

        self.panel.set_callsign("W1AW")
        self.panel._id_start_check.setChecked(True)
        self.panel._id_end_check.setChecked(True)
        dev, backend = self.hackrf()
        carrier = np.full(int(2e6), 0.9, np.complex64)  # 1 s at 2 MS/s
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertTrue(self.win._transmit_job(job))
        self.wait_done(20)
        result = self.station.engine.last_result
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.ids_sent, 2)
        self.assertEqual(
            analyse_mcw(backend.tx_iq(32), 2e6 / 32)["text"], "DE W1AW DE W1AW"
        )
        # Without them ticked, only the job goes out.
        self.panel._id_start_check.setChecked(False)
        self.panel._id_end_check.setChecked(False)
        self.panel._auto_id_check.setChecked(False)
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertTrue(self.win._transmit_job(job))
        self.wait_done(20)
        self.assertEqual(self.station.engine.last_result.ids_sent, 0)
        self.assertEqual(backend.tx_count(), len(carrier))

    def test_countdown_shows_while_an_interval_id_is_due(self):
        from sdr_module.gui.transmitter import TxJob

        self.panel.set_callsign("W1AW")
        self.panel._id_start_check.setChecked(False)
        self.panel._id_end_check.setChecked(False)
        self.panel._interval_spin.setValue(1)
        self.hackrf(pace=True)
        carrier = np.full(int(3 * 2e6), 0.9, np.complex64)
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertTrue(self.win._transmit_job(job))
        row, label = self.panel._countdown_row, self.panel._countdown_label
        self.assertTrue(self.pump(2, lambda: row.isVisibleTo(self.panel)))
        self.assertRegex(label.text(), r"^Next ID in 0[01]:\d\d$")
        # It counts down with the air time.
        self.assertTrue(
            self.pump(3, lambda: label.text().startswith("Next ID in 00:5"))
        )
        self.station.stop()
        self.wait_done(5)
        self.assertFalse(self.panel._countdown_row.isVisibleTo(self.panel))

    def _spy_status(self):
        shown = []
        real = self.win._show_status_message

        def spy(message, tone=None, duration_ms=4000):
            shown.append((message, tone, duration_ms))
            real(message, tone, duration_ms)

        self.win._show_status_message = spy
        return shown

    def test_on_air_stays_up_while_the_ids_go_out(self):
        """The status bar's "On air" lasts the whole transmission, the IDs
        Automatic ID adds included (it lasted the job's own air time plus
        10 s, so the status bar went blank while the IDs went out)."""
        from sdr_module.gui.transmitter import TxJob

        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # each ID: about 16.5 s
        self.hackrf(pace=True)
        shown = self._spy_status()
        carrier = np.full(int(2e6), 0.9, np.complex64)  # 1 s
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertTrue(self.win._transmit_job(job))
        total = self.station.engine.seconds_total()
        self.assertGreater(total, 30.0)
        on_air = [ms for text, _tone, ms in shown if text.startswith("On air:")]
        self.assertEqual(len(on_air), 1)
        self.assertGreaterEqual(on_air[0], total * 1000)
        # An interval ID makes the transmission longer: "On air" follows.
        shown.clear()
        self.station._on_progress(1.0, total + 20.0)
        on_air = [ms for text, _tone, ms in shown if text.startswith("On air:")]
        self.assertEqual(len(on_air), 1)
        self.assertGreaterEqual(on_air[0], (total + 19.0) * 1000)
        self.station.stop()
        self.wait_done(5)

    def test_the_ham_id_tab_says_what_is_on_the_air(self):
        """Another transmission isn't announced as "Transmitting ID"."""
        from sdr_module.gui.transmitter import TxJob

        self.panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        carrier = np.full(int(2e6), 0.9, np.complex64)
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertTrue(self.win._transmit_job(job))
        self.assertEqual(self.panel._status_label.text(), "On the air as W1AW")
        self.assertEqual(
            self.panel.tx_status_text(), "Sending a test carrier on 146.520 MHz…"
        )
        self.station.stop()
        self.wait_done(5)
        self.assertTrue(self.station.send_station_id())
        self.assertEqual(self.panel._status_label.text(), "Transmitting ID: DE W1AW")
        self.station.stop()
        self.wait_done(5)

    def test_the_confirmation_counts_the_station_ids(self):
        """The first-use question gives the whole air time, the IDs at the
        start and end included, not just the job's."""
        from sdr_module.gui.transmitter import TxJob

        self.panel.set_callsign("W1AW")
        self.hackrf()
        self.store.values["tx/confirm"] = True
        seen = []

        def answer(box):
            seen.append(box.informativeText())
            box.clickedButton = lambda: box.escapeButton()  # Cancel
            return 0

        self.win._exec_dialog = answer
        carrier = np.full(int(2e6), 0.9, np.complex64)  # 1 s
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        self.assertFalse(self.win._transmit_job(job))
        id_s = self.station.make_id_job(146.52e6).duration_s
        self.assertIn(f"about {1.0 + 2 * id_s:.1f} s with your station ID", seen[-1])
        self.assertTrue(self.station.send_station_id() is False)
        self.assertIn(f"about {id_s:.1f} s. Make sure", seen[-1])

    def test_an_id_that_wont_be_sent_isnt_checked(self):
        """With Automatic ID off, the ID's own emission (here an FM tone,
        not allowed in a CW segment) doesn't refuse a CW transmission."""
        from sdr_module.core.frequency_manager import LicenseClass
        from sdr_module.gui.transmitter import TxJob

        self.win._control_panel.set_license_class(LicenseClass.AMATEUR_EXTRA)
        self.panel.set_callsign("W1AW")
        self.panel._emission_combo.setCurrentIndex(
            self.panel._emission_combo.findData("mcw")
        )
        self.hackrf(freq=14.05e6)
        carrier = np.full(20000, 0.9, np.complex64)
        job = TxJob(carrier, 2e6, 14.05e6, "CW", 150.0, "a test carrier")
        self.assertFalse(self.win._transmit_job(job))  # its FM ID is refused
        self.assertIn("FM isn't allowed", self.status()[0])
        for check in (
            self.panel._auto_id_check,
            self.panel._id_start_check,
            self.panel._id_end_check,
        ):
            check.setChecked(False)
        self.assertTrue(self.win._transmit_job(job), self.status()[0])
        self.wait_done(10)
        self.assertTrue(self.station.engine.last_result.ok)


class TestMenusAndBadge(_TxCase):
    def test_radio_menu_has_send_and_stop(self):
        names = [
            a.text().replace("&", "")
            for a in self.win.menuBar().actions()[2].menu().actions()
        ]
        self.assertIn("Send Station ID", names)
        self.assertIn("Stop Transmitting", names)
        self.assertFalse(self.station._stop_action.isEnabled())

    def test_their_access_keys_are_their_own(self):
        """Alt+R then a letter reaches each directly (Stop Transmitting's
        T used to be Tuning Step's too)."""
        import re

        menu = next(
            a.menu()
            for a in self.win.menuBar().actions()
            if a.menu() is not None and self.station._stop_action in a.menu().actions()
        )
        keys = {}
        for action in menu.actions():
            match = re.search(r"&(\w)", action.text())
            if match:
                keys.setdefault(match.group(1).lower(), []).append(action.text())
        for action in (self.station._send_action, self.station._stop_action):
            letter = re.search(r"&(\w)", action.text()).group(1).lower()
            self.assertEqual(keys[letter], [action.text()])

    def test_on_air_badge_fits_a_1024_px_window(self):
        self.win.resize(1024, 640)
        self.station._badge_action.setVisible(True)
        self.settle(10)
        badge = self.station._badge
        self.assertTrue(badge.isVisible())
        right = badge.mapTo(self.win, badge.rect().topRight()).x()
        self.assertLessEqual(right, self.win.width())
        self.assertEqual(badge.text(), "ON AIR")
        self.assertEqual(badge.property("tone"), "danger")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNoModalBoxes(unittest.TestCase):
    def test_tx_station_opens_no_message_box_on_its_own(self):
        """Refusals and failures go to the status bar and the TX line; the
        only box is the (first-use) confirmation."""
        import inspect

        from sdr_module.gui import tx_station

        source = inspect.getsource(tx_station)
        self.assertEqual(source.count("QMessageBox("), 1)
        for name in ("warning(", "critical(", "information("):
            self.assertNotIn(f"QMessageBox.{name}", source)


if __name__ == "__main__":
    unittest.main()
