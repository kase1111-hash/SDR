"""The QRP panel: calculated output with the legal limit at the tuned
frequency, amplifier presets with saturation, the class badge agreeing with
the number, Auto mode, the saved settings and QSO tally (km/mi, delete,
reset, undo) and the contest exchange."""

from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def _settle():
    for _ in range(3):
        QApplication.processEvents()


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.core import frequency_manager
        from sdr_module.ham.gui.qrp_panel import QRPPanel

        self.fm = frequency_manager.get_frequency_manager()
        self._license = self.fm.get_license_class()
        self.addCleanup(lambda: self.fm.set_license_class(self._license))
        self.panel = QRPPanel()
        self.panel.resize(360, 900)
        self.panel.show()
        _settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.panel.deleteLater()
            _settle()

    def license(self, name):
        from sdr_module.core.frequency_manager import LicenseClass

        self.fm.set_license_class(LicenseClass(name))
        self.panel.refresh_license()


class TestCalculatedOutput(_PanelCase):
    def test_it_is_titled_as_a_calculation_and_starts_with_the_sdr_alone(self):
        """UX-06: not live power; the SDR's own milliwatt by default."""
        from PyQt6.QtWidgets import QGroupBox

        titles = [g.title() for g in self.panel.findChildren(QGroupBox)]
        self.assertIn("Calculated Output", titles)
        self.assertNotIn("TX Power", titles)
        calc = self.panel._amp_calc
        self.assertEqual(calc.preset(), "SDR only")
        self.assertFalse(calc._driver_check.isChecked())
        self.assertFalse(calc._pa_check.isChecked())
        self.assertEqual(self.panel._power_display._watts_label.text(), "1 mW")

    def test_badge_agrees_with_the_number(self):
        """UX-11: near a class limit the number shows enough digits."""
        from sdr_module.ham.gui.qrp_panel import format_class_watts

        display = self.panel._power_display
        table = {
            30.0: ("1 W", "QRPp"),
            30.1: ("1.02 W", "QRP"),
            37.0: ("5 W", "QRP"),
            37.1: ("5.1 W", "Low Power"),
            50.0: ("100 W", "Low Power"),
            50.1: ("102 W", "QRO"),
            20.0: ("100 mW", "QRPp"),
        }
        for dbm, (number, badge) in table.items():
            with self.subTest(dbm=dbm):
                display.set_power(dbm, "CW")
                self.assertEqual(display._watts_label.text(), number)
                self.assertEqual(display._status_label.text(), badge)
        self.assertEqual(format_class_watts(10 ** (0.04 / 10), "SSB"), "1 W")

    def test_legal_limit_and_privilege_at_the_tuned_frequency(self):
        """GAP-05: the license's limit (or refusal) where it is tuned, for
        the mode, refreshed on tune, mode and license changes."""
        p = self.panel
        self.license("technician")
        p.set_station(28.4e6, "USB")
        self.assertEqual(p.mode(), "SSB")
        self.assertEqual(
            p._legal_label.text(),
            "At 28.400 MHz (Technician, SSB): legal limit 200 W PEP.",
        )
        self.assertEqual(p._legal_label.property("tone"), "success")
        p.set_station(14.05e6, "CW")
        self.assertEqual(
            p._legal_label.text(),
            "Not permitted at 14.050 MHz in CW: your license (Technician) "
            "doesn't include 14.050 MHz.",
        )
        self.assertEqual(p._legal_label.property("tone"), "danger")
        self.license("general")
        p.set_station(10.12e6, "CW")
        self.assertIn("legal limit 200 W PEP", p._legal_label.text())
        p.set_station(146.52e6, "FM")
        self.assertEqual(
            p._legal_label.text(),
            "At 146.520 MHz (General, FM): legal limit 1500 W PEP (the general "
            "amateur limit).",
        )

    def test_badge_turns_red_over_the_legal_limit(self):
        p = self.panel
        self.license("none")
        p.set_station(151.82e6, "FM")  # MURS: 2 W
        p._amp_calc.apply_preset("HackRF → 5 W QRP kit")
        p._amp_calc._input_spin.setValue(15)
        badge = p._power_display._status_label
        self.assertEqual(badge.text(), "QRP")
        self.assertEqual(badge.property("tone"), "danger")
        self.assertIn(
            "Above the 2 W legal limit", p._power_display._detail_label.text()
        )
        self.assertIn("is over it", p._legal_label.text())
        p._amp_calc.apply_preset("SDR only")
        self.assertEqual(badge.property("tone"), "success")

    def test_auto_mode_follows_demodulation(self):
        """UX-07 (QRP): Auto first, resolved from the demodulation mode."""
        p = self.panel
        combo = p._mode_combo
        self.assertEqual(combo.itemText(0), "Auto (follow demodulation)")
        self.assertEqual(combo.currentIndex(), 0)
        for demod, mode in (
            ("CW", "CW"),
            ("USB", "SSB"),
            ("LSB", "SSB"),
            ("AM", "AM"),
            ("FM", "FM"),
            ("None (I/Q)", "Digital"),
        ):
            p.set_demod_mode(demod)
            self.assertEqual(p.mode(), mode)
            self.assertEqual(combo.face_text(), f"Auto · {mode}")
        # The QRP limit follows: 8 W is QRP in SSB, not in CW.
        p._amp_calc._input_spin.setValue(0)
        p._amp_calc._driver_check.setChecked(True)
        p._amp_calc._pa_check.setChecked(True)
        p._amp_calc._pa_spin.setValue(19)
        p.set_demod_mode("USB")
        self.assertEqual(p._power_display._status_label.text(), "QRP")
        p.set_demod_mode("CW")
        self.assertEqual(p._power_display._status_label.text(), "Low Power")
        p.set_demod_mode("None (I/Q)")  # digital: the 5 W limit, like CW
        self.assertEqual(p._power_display._status_label.text(), "Low Power")
        self.assertIn("5 W QRP limit", p._power_display._detail_label.text())
        combo.setCurrentText("SSB")  # a manual choice overrides
        self.assertEqual(p.mode(), "SSB")
        self.assertEqual(combo.face_text(), "SSB")


class TestAmplifierPresets(_PanelCase):
    def test_qrp_kit_saturates_at_five_watts(self):
        """GAP-06: the chain caps at each stage's maximum."""
        calc = self.panel._amp_calc
        calc.apply_preset("HackRF → 5 W QRP kit")
        calc._input_spin.setValue(15)
        self.assertTrue(calc._driver_check.isChecked() and calc._pa_check.isChecked())
        self.assertEqual(calc._driver_out.text(), "1 W")
        self.assertEqual(calc._stages[0].note.text(), "saturated")
        self.assertEqual(calc._stages[0].note.property("tone"), "warning")
        self.assertEqual(calc._pa_out.text(), "5 W")
        self.assertEqual(calc._stages[1].note.text(), "saturated")
        self.assertEqual(calc._output_label.text(), "5 W (+37 dBm)")
        self.assertEqual(calc._dc_label.text(), "11.7 W")
        calc._input_spin.setValue(0)
        self.assertEqual(calc._stages[0].note.text(), "")
        self.assertEqual(calc._output_label.text(), "1 W (+30 dBm)")

    def test_linear_preset_and_custom(self):
        calc = self.panel._amp_calc
        calc.apply_preset("HackRF → 100 W linear")
        calc._input_spin.setValue(15)
        self.assertEqual(calc._output_label.text(), "100 W (+50 dBm)")
        self.assertEqual(calc._dc_label.text(), "203 W")
        calc._pa_spin.setValue(10)  # editing a stage makes it Custom
        self.assertEqual(calc.preset(), "Custom")
        self.assertEqual(calc._output_label.text(), "20 W (+43 dBm)")
        calc._pa_max_spin.setValue(40)
        self.assertEqual(calc._output_label.text(), "10 W (+40 dBm)")
        self.assertEqual(calc._stages[1].note.text(), "saturated")

    def test_hackrf_output_is_read_only_while_connected(self):
        calc = self.panel._amp_calc
        calc._input_spin.setValue(-3)
        self.panel.set_sdr_output(-31.4, "From Ham ID > TX Power: TX gain 20 dB")
        self.assertTrue(calc._input_spin.isReadOnly())
        self.assertAlmostEqual(calc._input_spin.value(), -31.4)
        self.assertIn("Ham ID > TX Power", calc._input_spin.toolTip())
        self.assertAlmostEqual(
            self.panel.calculated_output_watts(), 10 ** (-6.14), places=9
        )
        self.panel.set_sdr_output(None)
        self.assertFalse(calc._input_spin.isReadOnly())
        self.assertEqual(calc._input_spin.value(), -3)


class TestTally(_PanelCase):
    def log(self, distance, watts):
        p = self.panel
        p._distance_spin.setValue(distance)
        p._mpw_power_spin.setValue(watts)
        p._log_qso_btn.click()

    def test_list_delete_reset_and_undo(self):
        """UX-12/GAP-07: the QSOs logged, with delete, reset and undo."""
        p = self.panel
        p._unit_combo.setCurrentIndex(p._unit_combo.findData("mi"))
        self.log(1000, 5)
        self.log(500, 1)
        self.log(200, 0.5)
        items = [p._qso_list.item(i).text() for i in range(p._qso_list.count())]
        self.assertEqual(len(items), 3)
        self.assertTrue(items[0].startswith("200 mi · 500 mW · 400 MPW"), items)
        self.assertTrue(items[2].startswith("1,000 mi · 5 W · 200 MPW"), items)
        self.assertEqual(p._best_mpw_label.text(), "500 MPW")
        self.assertEqual(p.get_controller().get_statistics()["total_qsos"], 3)

        p._qso_list.item(1).setSelected(True)  # 500 mi on 1 W
        p._delete_qso_btn.click()
        self.assertEqual(p._qso_list.count(), 2)
        self.assertEqual(p._best_mpw_label.text(), "400 MPW")
        self.assertTrue(p._undo_btn.isVisibleTo(p))
        p._undo_btn.click()
        self.assertEqual(p._qso_list.count(), 3)
        p._reset_btn.click()
        self.assertEqual(p._qso_list.count(), 0)
        self.assertFalse(p._stats_row.isVisibleTo(p))
        self.assertIn("Undo brings it back", p._mpw_feedback.text())
        p._undo_btn.click()
        self.assertEqual(p._qso_list.count(), 3)
        self.assertEqual(p.get_controller().get_statistics()["total_qsos"], 3)

    def test_kilometres(self):
        p = self.panel
        p._unit_combo.setCurrentIndex(p._unit_combo.findData("mi"))
        self.log(1000, 5)
        p._unit_combo.setCurrentIndex(p._unit_combo.findData("km"))
        self.assertEqual(p._distance_spin.suffix(), " km")
        self.assertEqual(p._distance_spin.value(), 1609)
        self.assertEqual(p._best_mpw_label.text(), "322 km/W")
        self.assertTrue(p._qso_list.item(0).text().startswith("1,609 km · 5 W"))
        self.assertEqual(p._mpw_group.title(), "Kilometres Per Watt")

    def test_state_round_trip(self):
        """GAP-07: settings and tally come back."""
        from sdr_module.ham.gui.qrp_panel import QRPPanel

        p = self.panel
        p._unit_combo.setCurrentIndex(p._unit_combo.findData("km"))
        p._amp_calc.apply_preset("HackRF → 5 W QRP kit")
        p._mode_combo.setCurrentText("SSB")
        p._set_quick_limit(1.0)
        p._state_edit.setText("ca")
        self.log(800, 2)
        state = json.loads(json.dumps(p.get_state()))

        other = QRPPanel()
        self.addCleanup(other.deleteLater)
        other.set_state(state)
        self.assertEqual(other._amp_calc.preset(), "HackRF → 5 W QRP kit")
        self.assertTrue(other._amp_calc._pa_check.isChecked())
        self.assertEqual(other.mode(), "SSB")
        self.assertTrue(other._limit_check.isChecked())
        self.assertAlmostEqual(other.power_limit_watts(), 1.0)
        self.assertEqual(other._unit_combo.currentData(), "km")
        self.assertEqual(other._qso_list.count(), 1)
        self.assertEqual(other._state_edit.text(), "ca")
        # Broken entries are skipped, not fatal.
        other.set_state({"qsos": [{"km": "x"}, None, {"km": 5, "watts": 1}], "mode": 7})
        self.assertEqual(other._qso_list.count(), 1)


class TestExchange(_PanelCase):
    def test_exchange_and_copy(self):
        """GAP-08/FG-G29: report, location and power, ready to copy."""
        p = self.panel
        p._mode_combo.setCurrentText("CW")
        self.assertEqual(p._exchange_label.text(), "599 1mW")
        p._amp_calc.apply_preset("HackRF → 5 W QRP kit")
        p._amp_calc._input_spin.setValue(15)
        p._state_edit.setText("ca")
        self.assertEqual(p._exchange_label.text(), "599 CA 5W")
        p._mode_combo.setCurrentText("SSB")
        self.assertEqual(p._exchange_label.text(), "59 CA 5W")
        p.set_rst_source(lambda: "57")
        self.assertEqual(p._exchange_label.text(), "57 CA 5W")
        copied = []
        p.exchange_copied.connect(copied.append)
        self.assertEqual(p.copy_exchange(), "57 CA 5W")
        self.assertEqual(QApplication.clipboard().text(), "57 CA 5W")
        self.assertEqual(copied, ["57 CA 5W"])


class TestQrpInTheWindow(_WindowTestCase):
    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.qrp = self.win._qrp_panel

    def test_follows_tuning_mode_and_license(self):
        from sdr_module.core.frequency_manager import LicenseClass

        panel = self.win._control_panel
        panel.set_license_class(LicenseClass.TECHNICIAN)
        self.addCleanup(lambda: panel.set_license_class(LicenseClass.NONE))
        self.win.set_frequency(28.4e6)
        self.win._set_demod_mode("USB")
        self.assertEqual(self.qrp.mode(), "SSB")
        self.assertIn("legal limit 200 W PEP", self.qrp._legal_label.text())
        panel.set_license_class(LicenseClass.NONE)
        self.assertTrue(self.qrp._legal_label.text().startswith("Not permitted"))

    def test_settings_and_tally_survive_a_restart(self):
        self.qrp._unit_combo.setCurrentIndex(self.qrp._unit_combo.findData("mi"))
        self.qrp._amp_calc.apply_preset("HackRF → 100 W linear")
        self.qrp._distance_spin.setValue(1200)
        self.qrp._log_qso_btn.click()
        self.assertIn("ham/qrp", self.store.values)
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        restored = other._qrp_panel
        self.assertEqual(restored._amp_calc.preset(), "HackRF → 100 W linear")
        self.assertEqual(restored._qso_list.count(), 1)
        self.assertTrue(restored._qso_list.item(0).text().startswith("1,200 mi"))

    def test_exchange_takes_the_s_meter_report(self):
        meter = self.win._signal_meter_panel
        self.assertIsNone(self.win._smeter_rst())  # no reading yet
        reading = type("Reading", (), {"signal_present": True})()
        meter._last_reading = reading
        meter._showing_data = True
        meter.get_meter().get_rst = lambda: "57"
        self.assertEqual(self.win._smeter_rst(), "57")
        self.qrp._update_exchange()
        self.assertTrue(self.qrp._exchange_label.text().startswith("57 "))


if __name__ == "__main__":
    unittest.main()
