#!/usr/bin/env python3
"""
Tests for the ham radio tool panels: Ham ID (callsign), SSTV and QRP.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_ham1.py``
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import QApplication, QFrame, QScrollArea

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:  # pragma: no cover - environment
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)

ROOT = Path(__file__).resolve().parent.parent
HAM_GUI = ROOT / "src" / "sdr_module" / "ham" / "gui"
PANEL_FILES = ("callsign_panel.py", "sstv_panel.py", "qrp_panel.py")
COLUMN_WIDTH = 360


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


def form_label_offsets(widget):
    """(label text, |label centre - field centre|) for every form row."""
    from PyQt6.QtWidgets import QFormLayout, QLabel

    rows = []
    for form in widget.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if not (label and field and isinstance(label.widget(), QLabel)):
                continue
            if not label.widget().text():
                continue
            offset = abs(
                label.widget().geometry().center().y() - field.geometry().center().y()
            )
            rows.append((label.widget().text(), offset))
    return rows


class TestSources(unittest.TestCase):
    """Static checks that need no display."""

    def test_no_hard_coded_colors(self):
        pattern = re.compile(
            r"#[0-9a-fA-F]{3,8}\b|\brgb\(|QColor\(\s*\d"
            r"|\b(gray|grey|green|red|orange|blue|yellow|white|black)\b\s*[;\"']"
            r"|setStyleSheet\("
        )
        for name in PANEL_FILES:
            text = (HAM_GUI / name).read_text(encoding="utf-8")
            hits = [
                line.strip()
                for line in text.splitlines()
                if pattern.search(line) and not line.lstrip().startswith("#")
            ]
            self.assertEqual(hits, [], f"hard-coded colors in {name}")

    def test_importing_ham_panels_first_keeps_ham_tabs(self):
        """The panels import the theme lazily; a module-level import would be
        circular and make main_window silently drop the Ham Radio tabs."""
        code = (
            "import sdr_module.ham.gui.callsign_panel\n"
            "from sdr_module.gui import main_window\n"
            "print(main_window.HAS_HAM_RADIO)\n"
        )
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=120,
        )
        if "No module named 'PyQt6'" in result.stderr:
            self.skipTest("PyQt6 not available")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "True")


class TestMorseHelpers(unittest.TestCase):
    def test_pattern(self):
        from sdr_module.ham.gui.callsign_panel import morse_pattern

        self.assertEqual(morse_pattern("DE W1AW"), "-.. . / .-- .---- .- .--")

    def test_duration_matches_generated_audio(self):
        from sdr_module.ham.callsign import generate_cw_id
        from sdr_module.ham.gui.callsign_panel import morse_duration

        for wpm in (12, 20, 35):
            audio = generate_cw_id("VE3ABC/P", wpm=wpm, sample_rate=48000)
            self.assertAlmostEqual(
                morse_duration("DE VE3ABC/P", wpm), len(audio) / 48000.0, places=1
            )


class TestCallsignPanel(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui.callsign_panel import CallsignPanel

        self.panel = CallsignPanel()
        self.panel.resize(COLUMN_WIDTH, 600)
        self.panel.show()
        settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.panel.deleteLater()
            settle()

    def type_callsign(self, text: str) -> None:
        self.panel._callsign_input.clear()
        self.panel._callsign_input.setFocus()
        QTest.keyClicks(self.panel._callsign_input, text)
        settle()

    def test_fits_right_column(self):
        self.assertLessEqual(self.panel.minimumSizeHint().width(), COLUMN_WIDTH)

    def test_buttons_disabled_with_reason_until_callsign(self):
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertFalse(self.panel._test_btn.isEnabled())
        self.assertIn("callsign", self.panel._id_now_btn.toolTip().lower())
        self.assertIn("callsign", self.panel._test_btn.toolTip().lower())

    def test_typing_uppercases_and_enables_send(self):
        self.type_callsign("w1aw")
        self.assertEqual(self.panel._callsign_input.text(), "W1AW")
        self.assertEqual(self.panel.get_callsign(), "W1AW")
        self.assertTrue(self.panel._id_now_btn.isEnabled())
        self.assertTrue(self.panel._test_btn.isEnabled())
        self.assertEqual(self.panel._status_label.property("tone"), "success")
        self.assertIn(".--", self.panel._morse_label.text())

    def test_typing_in_the_middle_keeps_cursor(self):
        self.type_callsign("w1aw")
        self.panel._callsign_input.setCursorPosition(0)
        QTest.keyClicks(self.panel._callsign_input, "k")
        self.assertEqual(self.panel._callsign_input.text(), "KW1AW")
        self.assertEqual(self.panel._callsign_input.cursorPosition(), 1)

    def test_invalid_characters_rejected(self):
        self.type_callsign("w1#a w")
        self.assertEqual(self.panel.get_callsign(), "W1AW")

    def test_unusual_format_warns_and_blocks_send(self):
        self.type_callsign("abcd")
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertEqual(self.panel._status_label.property("tone"), "warning")
        self.assertIn("digit", self.panel._id_now_btn.toolTip())

    def test_send_id_emits_request(self):
        self.type_callsign("w1aw")
        spy = QSignalSpy(self.panel.id_requested)
        self.panel._id_now_btn.click()
        self.assertEqual(len(spy), 1)

    def test_tx_unavailable_disables_send_with_reason(self):
        self.type_callsign("w1aw")
        self.panel.set_tx_available(False, "Connect a HackRF One to transmit.")
        self.assertFalse(self.panel._id_now_btn.isEnabled())
        self.assertEqual(
            self.panel._id_now_btn.toolTip(), "Connect a HackRF One to transmit."
        )
        self.panel.set_tx_available(True)
        self.assertTrue(self.panel._id_now_btn.isEnabled())

    def test_set_callsign_normalizes(self):
        self.panel.set_callsign(" k1abc ")
        self.assertEqual(self.panel.get_callsign(), "K1ABC")
        self.assertEqual(self.panel._callsign_input.text(), "K1ABC")

    def test_transmitting_state_and_auto_id_gate(self):
        self.type_callsign("w1aw")
        self.panel.set_transmitting(True)
        self.assertTrue(self.panel._id_timer.isActive())
        self.assertTrue(self.panel._countdown_row.isVisible())
        self.assertEqual(self.panel._status_label.property("tone"), "danger")
        self.panel.set_transmitting(False)
        self.assertFalse(self.panel._id_timer.isActive())

        self.panel._auto_id_check.setChecked(False)
        self.assertFalse(self.panel._interval_spin.isEnabled())
        self.panel.set_transmitting(True)
        self.assertFalse(self.panel._id_timer.isActive())
        self.assertFalse(self.panel._countdown_row.isVisible())
        self.panel.set_transmitting(False)

    def test_unimplemented_modes_not_selectable(self):
        model = self.panel._mode_combo.model()
        self.assertTrue(model.item(0).isEnabled())
        for row in range(1, model.rowCount()):
            self.assertFalse(model.item(row).isEnabled())

    def test_settings_round_trip(self):
        settings = {
            "callsign": "g3abc",
            "auto_id": False,
            "id_at_start": False,
            "id_at_end": True,
            "mode": "CW",
            "interval_minutes": 5,
            "cw_wpm": 25,
            "cw_tone": 650,
        }
        self.panel.set_settings(settings)
        result = self.panel.get_settings()
        self.assertEqual(result["callsign"], "G3ABC")
        for key in ("auto_id", "id_at_start", "id_at_end", "interval_minutes"):
            self.assertEqual(result[key], settings[key])
        self.assertEqual(result["cw_wpm"], 25)
        self.assertEqual(result["cw_tone"], 650)

    def test_form_labels_centred_on_fields(self):
        rows = form_label_offsets(self.panel)
        self.assertGreaterEqual(len(rows), 5)
        for text, offset in rows:
            self.assertLessEqual(offset, 1, f"{text} sits {offset}px off its field")

    def test_morse_box_has_only_code_and_ready_line_has_duration(self):
        self.type_callsign("w1aw")
        self.assertRegex(self.panel._morse_label.text(), r"^[.\- /]+$")
        self.assertIn("20\u00a0WPM", self.panel._status_label.text())
        self.panel._wpm_spin.setValue(30)
        self.assertIn("30\u00a0WPM", self.panel._status_label.text())
        self.assertIn("30 WPM", self.panel._morse_label.toolTip())

    def test_interval_tooltip_explains_when_disabled(self):
        self.panel._auto_id_check.setChecked(False)
        self.assertFalse(self.panel._interval_spin.isEnabled())
        self.assertIn("Auto-ID", self.panel._interval_spin.toolTip())
        self.panel._auto_id_check.setChecked(True)
        self.assertIn("10 minutes", self.panel._interval_spin.toolTip())

    def test_unavailable_saved_mode_falls_back_to_cw(self):
        self.panel.set_settings({"mode": "VOICE"})
        self.assertEqual(self.panel._mode_combo.currentIndex(), 0)
        self.assertEqual(self.panel.get_settings()["mode"], "CW")
        self.assertTrue(self.panel._wpm_spin.isEnabled())

    def test_preview_without_audio_explains(self):
        self.type_callsign("w1aw")
        self.panel._play_preview = lambda audio: False
        self.panel._test_btn.click()
        self.assertEqual(self.panel._status_label.property("tone"), "warning")
        self.assertIn("audio", self.panel._status_label.text().lower())
        self.assertEqual(self.panel._test_btn.text(), "Preview ID")


class _SSTVBase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.panel = SSTVPanel()
        self.panel.set_save_directory(self.tmp.name)
        self.panel.resize(COLUMN_WIDTH, 800)
        self.panel.show()
        settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.panel.deleteLater()
            settle()
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    @staticmethod
    def mode():
        from sdr_module.ham.sstv import SSTV_MODES

        return next(iter(SSTV_MODES.values()))

    def receive_image(self):
        mode = self.mode()
        image = np.zeros((mode.height, mode.width, 3), np.uint8)
        image[..., 0] = 200
        self.panel._decoder.state.mode = mode
        self.panel._on_mode_detected(mode)
        self.panel._on_image_complete(image)
        settle()
        return image


class TestSSTVPanel(_SSTVBase):
    def test_fits_right_column(self):
        self.assertLessEqual(self.panel.minimumSizeHint().width(), COLUMN_WIDTH)

    def test_construction_does_not_create_folder_in_cwd(self):
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        with tempfile.TemporaryDirectory() as cwd:
            old = os.getcwd()
            os.chdir(cwd)
            try:
                panel = SSTVPanel()
                self.assertEqual(os.listdir(cwd), [])
                panel.deleteLater()
            finally:
                os.chdir(old)

    def test_initial_state(self):
        p = self.panel
        self.assertFalse(p._save_btn.isEnabled())
        self.assertFalse(p._clear_btn.isEnabled())
        self.assertTrue(p._save_btn.toolTip())
        self.assertTrue(p._history_empty.isVisible())
        self.assertFalse(p._history_list.isVisible())
        self.assertFalse(p._nav_row.isVisible())
        self.assertEqual(p._start_btn.text(), "Start Decoder")
        self.assertFalse(p.is_receiving())

    def test_start_stop_toggle(self):
        p = self.panel
        started = QSignalSpy(p.start_requested)
        stopped = QSignalSpy(p.stop_requested)
        p._start_btn.click()
        self.assertTrue(p.is_receiving())
        self.assertEqual(p._start_btn.text(), "Stop Decoder")
        # The bar only appears once an image starts arriving.
        self.assertFalse(p._progress_bar.isVisible())
        p._on_mode_detected(self.mode())
        self.assertTrue(p._progress_bar.isVisible())
        p._start_btn.click()
        self.assertFalse(p._progress_bar.isVisible())
        self.assertFalse(p.is_receiving())
        self.assertEqual(p._start_btn.text(), "Start Decoder")
        self.assertEqual((len(started), len(stopped)), (1, 1))

    def test_received_image_listed_and_auto_saved(self):
        p = self.panel
        p._start_btn.click()
        self.receive_image()
        self.assertEqual(p._history_list.count(), 1)
        self.assertTrue(p._history_list.isVisible())
        self.assertFalse(p._history_empty.isVisible())
        self.assertTrue(p._save_btn.isEnabled())
        self.assertTrue(p.is_receiving(), "keeps listening for the next image")
        saved = list(Path(self.tmp.name).glob("sstv_*.png"))
        self.assertEqual(len(saved), 1)
        self.assertFalse(QImage(str(saved[0])).isNull())
        self.assertEqual(p._status_label.property("tone"), "success")

    def test_clear_and_history_navigation(self):
        p = self.panel
        p._start_btn.click()
        self.receive_image()
        self.receive_image()
        self.assertEqual(p._image_count_label.text(), "2 of 2")
        p._prev_btn.click()
        self.assertEqual(p._image_count_label.text(), "1 of 2")
        self.assertTrue(p._next_btn.isEnabled())
        p._clear_btn.click()
        self.assertFalse(p._image_display.has_image())
        self.assertFalse(p._save_btn.isEnabled())
        self.assertEqual(p._image_count_label.text(), "2 images")

    def test_no_audio_warning_while_listening(self):
        p = self.panel
        p._start_btn.click()
        p._update_status()
        self.assertFalse(p._no_audio_label.isVisible(), "not straight away")
        p._last_audio_time -= 10  # pretend ten silent seconds went by
        p._update_status()
        self.assertTrue(p._no_audio_label.isVisible())
        self.assertEqual(p._no_audio_label.property("tone"), "warning")
        p.process_audio(np.zeros(480, np.float32))
        self.assertFalse(p._no_audio_label.isVisible(), "audio clears it")
        p._last_audio_time -= 10
        p._update_status()
        p._start_btn.click()  # stop
        self.assertFalse(p._no_audio_label.isVisible())

    def test_audio_clears_warning_while_tab_hidden(self):
        p = self.panel
        p._start_btn.click()
        p._last_audio_time -= 10
        p._update_status()
        p.hide()  # e.g. another tab is in front while the receiver runs
        p.process_audio(np.zeros(480, np.float32))
        p.show()
        settle()
        self.assertFalse(p._no_audio_label.isVisible())

    def test_empty_state_matches_auto_save(self):
        p = self.panel
        self.assertIn("saved", p._history_empty.text())
        p._auto_save_check.setChecked(False)
        self.assertNotIn("saved to", p._history_empty.text())
        self.assertIn("Save Image", p._history_empty.text())

    def test_history_list_fits_its_rows(self):
        p = self.panel
        p._start_btn.click()
        self.receive_image()
        one = p._history_list.height()
        self.receive_image()
        two = p._history_list.height()
        self.assertGreater(two, one)
        for _ in range(6):
            self.receive_image()
        # Capped: longer histories scroll inside the list.
        self.assertLess(p._history_list.height(), 8 * p._history_list.sizeHintForRow(0))
        self.assertEqual(p._history_list.count(), 8)

    def test_process_audio_follows_sample_rate(self):
        p = self.panel
        p._start_btn.click()
        p.process_audio(np.zeros(480, np.float32), sample_rate=50000.0)
        self.assertEqual(p._decoder.sample_rate, 50000.0)

    def test_save_rgb_image_without_pil(self):
        from sdr_module.ham.gui.sstv_panel import save_rgb_image

        image = np.zeros((10, 20, 3), np.uint8)
        path = os.path.join(self.tmp.name, "x.png")
        self.assertTrue(save_rgb_image(image, path))
        loaded = QImage(path)
        self.assertEqual((loaded.width(), loaded.height()), (20, 10))

    def test_viewer_background_follows_theme(self):
        from sdr_module.gui import themes
        from sdr_module.ham.gui.sstv_panel import ImageDisplayWidget

        widget = ImageDisplayWidget()
        widget.resize(240, 192)
        saved = themes._current_theme
        try:
            for name in ("dark", "light"):
                themes._current_theme = name
                pixel = widget.grab().toImage().pixelColor(12, 12)
                self.assertEqual(
                    pixel.name(), themes.get_palette(name).plot_bg, f"{name} theme"
                )
        finally:
            themes._current_theme = saved
            widget.deleteLater()


class TestQRPClassification(unittest.TestCase):
    def test_classify(self):
        from sdr_module.ham.gui.qrp_panel import classify_qrp

        self.assertEqual(classify_qrp(0.5, "CW")[0], "QRPp")
        self.assertEqual(classify_qrp(1.0, "CW")[0], "QRPp")
        self.assertEqual(classify_qrp(5.0, "CW")[:2], ("QRP", "success"))
        self.assertEqual(classify_qrp(8.0, "SSB")[:2], ("QRP", "success"))
        self.assertEqual(classify_qrp(8.0, "CW")[:2], ("Low Power", "warning"))
        self.assertEqual(classify_qrp(8.0, "FT8")[0], "Low Power")
        self.assertEqual(classify_qrp(200.0, "SSB")[:2], ("QRO", "danger"))


class TestQRPPanel(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui.qrp_panel import QRPPanel

        self.panel = QRPPanel()
        self.panel.resize(COLUMN_WIDTH, 400)
        self.panel.show()
        settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.close()
            self.panel.deleteLater()
            settle()

    def test_content_scrolls_instead_of_crushing(self):
        area = self.panel.findChild(QScrollArea)
        self.assertIsNotNone(area)
        self.assertTrue(area.widgetResizable())
        self.assertEqual(area.frameShape(), QFrame.Shape.NoFrame)
        content = area.widget()
        # The groups keep their natural height (taller than the tab)...
        self.assertGreaterEqual(content.height(), content.minimumSizeHint().height())
        self.assertGreater(content.height(), 400)
        # ...and nothing needs sideways scrolling in a 360 px column.
        self.assertLessEqual(content.minimumSizeHint().width(), COLUMN_WIDTH - 12)
        self.assertEqual(
            area.horizontalScrollBarPolicy(), Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

    def test_initial_power_matches_calculator(self):
        display = self.panel._power_display
        self.assertEqual(display._watts_label.text(), "1.0 W")
        self.assertEqual(display._dbm_label.text(), "+30.0 dBm")
        self.assertEqual(display._status_label.text(), "QRPp")

    def test_mode_changes_status(self):
        self.panel._amp_calc._pa_spin.setValue(19)  # ~7.9 W
        self.panel._mode_combo.setCurrentText("CW")
        self.assertEqual(self.panel._power_display._status_label.text(), "Low Power")
        self.panel._mode_combo.setCurrentText("SSB")
        self.assertEqual(self.panel._power_display._status_label.text(), "QRP")

    def test_presets_enable_limit_and_flag_excess(self):
        p = self.panel
        self.assertEqual(
            [b.text() for b in p._preset_buttons], ["QRPp 1 W", "CW 5 W", "SSB 10 W"]
        )
        spy = QSignalSpy(p.power_limit_changed)
        p._amp_calc._pa_spin.setValue(20)  # 10 W out
        p._preset_buttons[1].click()  # 5 W limit
        self.assertTrue(p._limit_check.isChecked())
        self.assertTrue(p._limit_spin.isEnabled())
        self.assertAlmostEqual(p.get_controller().get_power_limit(), 5.0)
        self.assertGreaterEqual(len(spy), 1)
        self.assertTrue(p._limit_status.isVisible())
        self.assertEqual(p._limit_status.property("tone"), "danger")
        p._preset_buttons[2].click()  # 10 W limit
        self.assertEqual(p._limit_status.property("tone"), "success")
        p._limit_check.setChecked(False)
        self.assertFalse(p._limit_status.isVisible())

    def test_miles_per_watt_updates_live_and_logs(self):
        p = self.panel
        p._distance_spin.setValue(1000)
        self.assertEqual(p._mpw_label.text(), "200 MPW")
        self.assertFalse(p._stats_row.isVisible())
        p._log_qso_btn.click()
        self.assertEqual(p._qso_count_label.text(), "1")
        self.assertEqual(p._best_mpw_label.text(), "200 MPW")
        self.assertTrue(p._stats_row.isVisible())
        self.assertIn("New best", p._mpw_feedback.text())

    def test_fields_share_one_left_edge_and_labels_centred(self):
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox

        fields = self.panel.findChildren(QAbstractSpinBox) + self.panel.findChildren(
            QComboBox
        )
        xs = {f.mapTo(self.panel, QPoint(0, 0)).x() for f in fields}
        self.assertEqual(len(xs), 1, f"field left edges differ: {sorted(xs)}")
        for text, offset in form_label_offsets(self.panel):
            self.assertLessEqual(offset, 1, f"{text} sits {offset}px off its field")

    def test_whole_db_step_at_a_limit_counts_as_within(self):
        from sdr_module.ham.gui.qrp_panel import classify_qrp

        p = self.panel
        p._amp_calc._pa_spin.setValue(17)  # +37 dBm = 5.01 W, shown as "5.0 W"
        self.assertEqual(p._power_display._watts_label.text(), "5.0 W")
        self.assertEqual(p._power_display._status_label.text(), "QRP")
        p._preset_buttons[1].click()  # 5 W limit
        self.assertEqual(p._limit_status.property("tone"), "success")
        self.assertEqual(classify_qrp(5.2, "CW")[0], "Low Power")

    def test_limit_spin_explains_when_disabled(self):
        p = self.panel
        self.assertFalse(p._limit_spin.isEnabled())
        self.assertIn("Warn when output exceeds", p._limit_spin.toolTip())
        p._limit_check.setChecked(True)
        self.assertNotIn("Turn on", p._limit_spin.toolTip())

    def test_bypassed_stage_is_muted(self):
        calc = self.panel._amp_calc
        calc._driver_check.setChecked(False)
        self.assertEqual(calc._driver_out.text(), "(bypassed)")
        self.assertEqual(calc._driver_out.property("tone"), "muted")
        self.assertFalse(calc._driver_spin.isEnabled())
        self.assertAlmostEqual(calc.get_output_dbm(), 10.0)


if __name__ == "__main__":
    unittest.main()
