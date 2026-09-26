#!/usr/bin/env python3
"""
Tests for the control panel (right column): layout, frequency entry,
receiver / demodulation / recording controls, presets and license profile.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_control.py``
"""

import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QFrame, QScrollArea, QWidget

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:  # pragma: no cover - environment
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)

SOURCE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "sdr_module"
    / "gui"
    / "control_panel.py"
)


def require_pyqt6(test_case: unittest.TestCase) -> None:
    """Skip when PyQt6 is unavailable (fail instead if SDR_REQUIRE_GUI is set)."""
    if HAS_PYQT6:
        return
    if os.environ.get("SDR_REQUIRE_GUI"):
        test_case.fail(
            f"PyQt6 is required but could not be imported: {PYQT6_IMPORT_ERROR}"
        )
    test_case.skipTest("PyQt6 not available")


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


class _PanelTestCase(unittest.TestCase):
    width = 360
    height = 700

    @classmethod
    def setUpClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        # Measure the panel the way the app shows it: themed.
        qapp = QApplication.instance()
        cls._old_theme = themes.current_theme()
        cls._old_sheet, cls._old_palette = qapp.styleSheet(), qapp.palette()
        themes.apply_theme(qapp, "dark")

    @classmethod
    def tearDownClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        themes.apply_theme(None, cls._old_theme)
        qapp.setStyleSheet(cls._old_sheet)
        qapp.setPalette(cls._old_palette)

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.core.frequency_manager import get_frequency_manager
        from sdr_module.gui.control_panel import ControlPanel

        # The frequency manager is a process-wide singleton: restore it.
        self._fm = get_frequency_manager()
        self._license = self._fm.get_license_class()
        self.panel = ControlPanel()
        self.panel.resize(self.width, self.height)
        self.panel.show()
        settle()

    def tearDown(self):
        if not HAS_PYQT6:
            return
        self.panel.close()
        self.panel.deleteLater()
        settle()
        self._fm.set_license_class(self._license)


class TestLayout(_PanelTestCase):
    def test_sections_scroll_inside_the_panel(self):
        area = self.panel.findChild(QScrollArea)
        self.assertIsNotNone(area)
        self.assertTrue(area.widgetResizable())
        self.assertEqual(area.frameShape(), QFrame.Shape.NoFrame)
        self.assertEqual(
            area.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Much shorter than its content: it scrolls instead of crushing.
        self.assertLess(self.panel.minimumSizeHint().height(), 200)

    def test_content_fits_360px_without_sideways_clipping(self):
        content = self.panel._content
        viewport = self.panel._scroll.viewport()
        self.assertLessEqual(content.minimumSizeHint().width(), viewport.width())
        self.assertLessEqual(content.width(), viewport.width())

    def test_quick_tune_buttons_are_compact_and_not_clipped(self):
        buttons = self.panel._quick_tune_buttons
        self.assertEqual(len(buttons), 6)
        for button in buttons:
            self.assertEqual(button.property("role"), "compact")
            self.assertTrue(button.toolTip())
            text_width = button.fontMetrics().horizontalAdvance(button.text())
            self.assertGreaterEqual(button.width(), text_width + 4, button.text())

    def test_merged_groups_keep_every_control(self):
        from PyQt6.QtWidgets import QGroupBox

        titles = [g.title() for g in self.panel.findChildren(QGroupBox)]
        self.assertEqual(
            titles,
            [
                "Frequency",
                "Receiver",
                "Demodulation",
                "Presets",
                "Recording",
                "License Profile",
            ],
        )
        for name in (
            "_gain_slider",
            "_agc_check",
            "_bw_combo",
            "_demod_combo",
            "_fm_dev_combo",
            "_squelch_slider",
            "_record_btn",
            "_pause_btn",
            "_format_combo",
            "_license_combo",
        ):
            self.assertIsInstance(getattr(self.panel, name), QWidget, name)

    def test_inputs_have_tooltips(self):
        p = self.panel
        for widget in (
            p._freq_input._freq_input,
            p._freq_input._unit_combo,
            p._category_combo,
            p._preset_combo,
            p._apply_preset_btn,
            p._gain_slider,
            p._agc_check,
            p._bw_combo,
            p._demod_combo,
            p._fm_dev_combo,
            p._squelch_slider,
            p._format_combo,
            p._record_btn,
            p._pause_btn,
            p._license_combo,
            p._tx_warning,
        ):
            self.assertTrue(widget.toolTip(), type(widget).__name__)


class TestTheming(_PanelTestCase):
    def test_no_hard_coded_colors_in_source(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"#[0-9a-fA-F]{3,8}\b", text))
        self.assertIsNone(re.search(r"rgba?\(", text))
        self.assertNotIn("setStyleSheet", text)

    def test_no_widget_has_an_inline_stylesheet(self):
        for widget in self.panel.findChildren(QWidget):
            self.assertEqual(widget.styleSheet(), "", type(widget).__name__)

    def test_status_labels_use_roles_and_tones(self):
        p = self.panel
        self.assertEqual(p._preset_info.property("role"), "hint")
        self.assertEqual(p._license_info.property("role"), "hint")
        self.assertEqual(p._tx_warning.property("role"), "callout")
        self.assertEqual(p._tx_warning.property("tone"), "warning")
        self.assertEqual(p._record_btn.property("role"), "danger")


class TestFrequencyEntry(_PanelTestCase):
    def setUp(self):
        super().setUp()
        self.entry = self.panel._freq_input
        self.events = []
        self.panel.frequency_changed.connect(self.events.append)

    def test_unit_switch_keeps_frequency_in_every_unit(self):
        self.panel.set_frequency(145.8e6)
        spin = self.entry._freq_input
        expected = {"Hz": 145800000.0, "kHz": 145800.0, "MHz": 145.8, "GHz": 0.1458}
        for unit, value in expected.items():
            self.entry._unit_combo.setCurrentText(unit)
            self.assertAlmostEqual(spin.value(), value, places=6, msg=unit)
            self.assertEqual(self.entry.get_frequency(), 145.8e6)
        self.assertEqual(self.events, [])  # changing the unit never retunes

    def test_external_set_frequency_does_not_echo(self):
        self.panel.set_frequency(433.92e6)
        self.assertEqual(self.panel._freq_input.get_frequency(), 433.92e6)
        self.assertAlmostEqual(self.entry._freq_input.value(), 433.92)
        self.assertEqual(self.events, [])

    def test_set_frequency_is_clamped_to_the_tuning_range(self):
        from sdr_module.gui.control_panel import MIN_FREQUENCY_HZ

        self.panel.set_frequency(-5e6)
        self.assertEqual(self.entry.get_frequency(), MIN_FREQUENCY_HZ)

    def test_typing_tunes_once_on_enter(self):
        spin = self.entry._freq_input
        self.assertFalse(spin.keyboardTracking())
        spin.setFocus()
        spin.selectAll()
        QTest.keyClicks(spin, "145.5")
        self.assertEqual(self.events, [])  # not on every keystroke
        QTest.keyClick(spin, Qt.Key.Key_Return)
        self.assertEqual(self.events, [145.5e6])

    def test_arrow_step_is_10_khz(self):
        self.panel.set_frequency(100e6)
        self.entry._freq_input.stepBy(1)
        self.assertEqual(self.events, [100.01e6])
        self.entry._unit_combo.setCurrentText("kHz")
        self.entry._freq_input.stepBy(-1)
        self.assertEqual(self.events[-1], 100e6)

    def test_quick_tune_emits_new_frequency(self):
        self.panel.set_frequency(100e6)
        plus_100k = self.panel._quick_tune_buttons[4]
        self.assertEqual(plus_100k.text(), "+100k")
        plus_100k.click()
        self.assertEqual(self.events, [100.1e6])
        self.assertEqual(self.panel._freq_input.get_frequency(), 100.1e6)


class TestReceiverControls(_PanelTestCase):
    def test_agc_disables_slider_without_sending_a_bogus_gain(self):
        gains, agc = [], []
        self.panel.gain_changed.connect(gains.append)
        self.panel.agc_changed.connect(agc.append)
        self.panel.set_gain(30)
        gains.clear()

        self.panel.set_agc_enabled(True)
        self.assertFalse(self.panel._gain_slider.isEnabled())
        self.assertEqual(self.panel._gain_label.text(), "Auto")
        self.assertEqual(agc, [True])
        self.assertEqual(gains, [])  # used to emit -1, i.e. 0 dB manual gain

        self.panel.set_agc_enabled(False)
        self.assertTrue(self.panel._gain_slider.isEnabled())
        self.assertEqual(agc, [True, False])
        self.assertEqual(gains, [30.0])  # manual gain restored
        self.assertEqual(self.panel._gain_label.text(), "30 dB")

    def test_fm_deviation_only_enabled_for_fm(self):
        combo = self.panel._demod_combo
        combo.setCurrentText("AM")
        self.assertFalse(self.panel._fm_dev_combo.isEnabled())
        self.assertIn("FM", self.panel._fm_dev_combo.toolTip())
        combo.setCurrentText("FM")
        self.assertTrue(self.panel._fm_dev_combo.isEnabled())

    def test_demod_items_keep_their_texts(self):
        combo = self.panel._demod_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(items, ["None (I/Q)", "AM", "FM", "USB", "LSB", "CW"])

    def test_squelch_readout_and_signal(self):
        values = []
        self.panel.squelch_changed.connect(values.append)
        self.panel.set_squelch_db(-60)
        self.assertEqual(values, [-60.0])
        self.assertEqual(self.panel.get_squelch_db(), -60.0)
        self.assertEqual(self.panel._squelch_label.text(), "-60 dBFS")

    def test_wheel_over_unfocused_slider_does_not_change_it(self):
        slider = self.panel._gain_slider
        start = slider.value()
        self.assertFalse(slider.hasFocus())
        center = QPointF(slider.width() / 2, slider.height() / 2)
        event = QWheelEvent(
            center,
            QPointF(slider.mapToGlobal(center.toPoint())),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(slider, event)
        self.assertEqual(slider.value(), start)


class TestRecording(_PanelTestCase):
    def test_pause_disabled_until_recording(self):
        self.assertFalse(self.panel._pause_btn.isEnabled())
        self.assertTrue(self.panel._pause_btn.toolTip())

    def test_pause_stays_disabled_without_a_listener(self):
        self.panel._record_btn.setChecked(True)
        self.assertFalse(self.panel._pause_btn.isEnabled())
        self.assertIn("isn't available", self.panel._pause_btn.toolTip())
        self.panel._record_btn.setChecked(False)

    def test_pause_and_resume_with_a_listener(self):
        paused = []
        self.panel.recording_paused.connect(paused.append)
        self.panel._record_btn.setChecked(True)
        self.assertEqual(self.panel._record_status.property("tone"), "danger")
        self.assertTrue(self.panel._pause_btn.isEnabled())
        self.assertFalse(self.panel._format_combo.isEnabled())

        self.panel._pause_btn.click()
        self.assertEqual(paused, [True])
        self.assertTrue(self.panel.is_recording_paused())
        self.assertIn("Paused", self.panel._record_status.text())
        self.assertEqual(self.panel._record_status.property("tone"), "warning")

        # Stopping from elsewhere resets pause without emitting anything.
        self.panel.set_recording_state(False)
        self.assertFalse(self.panel._pause_btn.isChecked())
        self.assertFalse(self.panel._pause_btn.isEnabled())
        self.assertTrue(self.panel._format_combo.isEnabled())
        self.assertEqual(self.panel._record_status.text(), "Ready")
        self.assertEqual(paused, [True])


class TestPresets(_PanelTestCase):
    def select(self, category: str, name: str) -> None:
        self.panel._category_combo.setCurrentText(category)
        self.panel._preset_combo.setCurrentText(name)

    def test_apply_picks_the_narrowest_bandwidth_that_fits(self):
        cases = {
            ("QRP", "QRP 20m SSB"): ("10 kHz", "USB"),  # was 25 kHz
            ("ISM", "ISM 915 MHz"): ("2 MHz", "None (I/Q)"),  # was unchanged
            ("Amateur", "2m Calling"): ("25 kHz", "FM"),
            ("Broadcast", "FM Broadcast"): ("200 kHz", "FM"),
        }
        for (category, name), (bw, mode) in cases.items():
            self.select(category, name)
            self.panel._apply_preset()
            self.assertEqual(self.panel._bw_combo.currentText(), bw, name)
            self.assertEqual(self.panel._demod_combo.currentText(), mode, name)

    def test_apply_tunes_and_sets_fm_deviation(self):
        events = []
        self.panel.frequency_changed.connect(events.append)
        self.select("Broadcast", "FM Broadcast")
        self.panel._apply_preset()
        self.assertEqual(events, [100e6])
        self.assertEqual(self.panel.get_fm_deviation(), 75e3)
        self.select("Amateur", "2m Calling")
        self.panel._apply_preset()
        self.assertEqual(self.panel.get_fm_deviation(), 5e3)

    def test_tx_status_tone_follows_license(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_license_class(LicenseClass.NONE)
        self.select("Amateur", "2m Calling")
        self.assertEqual(self.panel._preset_tx.property("tone"), "warning")
        self.panel.set_license_class(LicenseClass.TECHNICIAN)
        self.assertEqual(self.panel._preset_tx.property("tone"), "success")
        self.assertIn("TX allowed", self.panel._preset_tx.text())
        self.select("GNSS", "GPS L1 (C/A)")
        self.assertEqual(self.panel._preset_tx.property("tone"), "danger")
        self.assertIn("GPS L1", self.panel._preset_tx.text())

    def test_browsing_presets_does_not_flood_the_error_log(self):
        with self.assertNoLogs("sdr_module.core.frequency_manager", "WARNING"):
            self.select("GNSS", "GPS L1 (C/A)")
            self.select("Aviation", "ADS-B")

    def test_license_combo_shows_the_active_class(self):
        from sdr_module.core.frequency_manager import LicenseClass
        from sdr_module.gui.control_panel import ControlPanel

        self._fm.set_license_class(LicenseClass.GENERAL)
        panel = ControlPanel()
        try:
            self.assertEqual(panel._license_combo.currentText(), "General")
            self.assertIn("HF", panel._license_info.text())
        finally:
            panel.deleteLater()


class TestReviewFixes(_PanelTestCase):
    """Fixes from the adversarial review of the rebuilt panel."""

    def test_preset_names_are_not_truncated_at_360px(self):
        from sdr_module.core.frequency_manager import get_frequency_manager

        combo = self.panel._preset_combo
        fm = get_frequency_manager()
        names = [
            p.name for c in fm.get_preset_categories() for p in fm.get_rx_presets(c)
        ]
        longest = max(combo.fontMetrics().horizontalAdvance(n) for n in names)
        # Text + left padding + drop-down arrow area of the themed combo.
        self.assertLessEqual(longest + 32, combo.width())

    def test_apply_preset_has_its_own_row_and_explains_when_disabled(self):
        button = self.panel._apply_preset_btn
        self.assertEqual(button.text(), "Apply Preset")
        # Below the preset details, not squeezed next to the preset combo.
        content = self.panel._content
        self.assertGreater(
            button.mapTo(content, QPoint(0, 0)).y(),
            self.panel._preset_tx.mapTo(content, QPoint(0, 0)).y(),
        )
        self.panel._on_preset_changed("")
        self.assertFalse(button.isEnabled())
        self.assertIn("preset first", button.toolTip())

    def test_panel_never_reports_a_minimum_narrower_than_its_content(self):
        content = self.panel._content.minimumSizeHint().width()
        self.assertGreater(self.panel.minimumSizeHint().width(), content)
        self.assertLess(self.panel.minimumSizeHint().height(), 200)

    def test_hz_and_khz_show_digit_grouping(self):
        entry = self.panel._freq_input
        spin = entry._freq_input
        sep = spin.locale().groupSeparator()
        self.panel.set_frequency(145.8e6)
        entry._unit_combo.setCurrentText("Hz")
        self.assertIn(sep, spin.text())
        entry._unit_combo.setCurrentText("MHz")
        self.assertFalse(spin.isGroupSeparatorShown())
        self.assertEqual(entry.get_frequency(), 145.8e6)

    def test_idle_timer_is_muted_not_amber(self):
        timer = self.panel._record_time
        self.assertEqual(timer.property("tone"), "muted")
        self.panel.set_recording_state(True)
        self.assertEqual(timer.property("tone"), "danger")
        self.panel.set_recording_state(False)
        self.assertEqual(timer.property("tone"), "muted")

    def test_no_license_message_only_promises_amateur_frequencies(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_license_class(LicenseClass.NONE)
        self.panel._category_combo.setCurrentText("Broadcast")
        self.panel._preset_combo.setCurrentText("FM Broadcast")
        self.assertNotIn("amateur license", self.panel._preset_tx.text())
        self.assertIn("outside", self.panel._preset_tx.text())
        self.panel._category_combo.setCurrentText("Amateur")
        self.panel._preset_combo.setCurrentText("2m Calling")
        self.assertIn("amateur license", self.panel._preset_tx.text())

    def test_set_license_class_resyncs_the_frequency_manager(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_license_class(LicenseClass.GENERAL)
        self._fm.set_license_class(LicenseClass.NONE)  # changed elsewhere
        self.panel.set_license_class(LicenseClass.GENERAL)
        self.assertEqual(self._fm.get_license_class(), LicenseClass.GENERAL)

    def test_sideband_modes_have_distinct_tooltips(self):
        combo = self.panel._demod_combo
        tips = {
            combo.itemText(i): combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
            for i in range(combo.count())
        }
        self.assertIn("Upper", tips["USB"])
        self.assertIn("Lower", tips["LSB"])
        self.assertTrue(all(tips.values()))


if __name__ == "__main__":
    unittest.main()
