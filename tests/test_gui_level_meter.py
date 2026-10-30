#!/usr/bin/env python3
"""
The toolbar's level meter (ux:UX-U04, rx:UX-07, ux:UX-U06): the channel
level as a bar with the dBFS number, the squelch threshold marked and
draggable (kept in sync with the slider), an open/closed lamp, and a
tooltip with the level, the noise floor and the SNR.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    Frames,
    carrier,
    noise,
    pretend_audio_output,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestLevelMeterWidget(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.level_meter import LevelMeter

        self.meter = LevelMeter()
        self.meter.resize(self.meter.sizeHint())
        self.meter.show()

    def tearDown(self):
        self.meter.close()
        self.meter.deleteLater()

    def test_readout_and_label_compatibility(self):
        m = self.meter
        self.assertEqual(m.text(), "— dBFS")
        m.set_reading(-46.25, -80.0, -73.0, "open")
        self.assertEqual(m.text(), "-46.2 dBFS")
        self.assertAlmostEqual(m.snr_db(), 33.75)
        m.setText("— dBFS")  # what the window does when it stops
        self.assertEqual(m.text(), "— dBFS")
        self.assertEqual(m.state(), "none")
        m.setText("-12.5 dBFS")
        self.assertEqual(m.level(), -12.5)

    def test_tooltip_has_level_noise_snr_and_squelch(self):
        m = self.meter
        m.set_reading(-46.2, -80.0, -73.0, "open", auto=True)
        m._update_tooltip(force=True)
        tip = m.toolTip()
        for part in (
            "channel",
            "-46.2 dBFS",
            "-80.0 dBFS",
            "SNR 33.8 dB",
            "Auto",
            "open",
        ):
            self.assertIn(part, tip)
        m.set_reading(-90.0, -80.0, -73.0, "closed")
        m._update_tooltip(force=True)
        self.assertIn("squelched", m.toolTip())
        self.assertIn("set by hand", m.toolTip())

    def test_dragging_sets_the_threshold(self):
        m = self.meter
        m.set_reading(-50.0, -80.0, -73.0, "closed")
        seen = []
        m.threshold_changed.connect(seen.append)
        bar = m._bar_rect()
        x_at = lambda db: int(round(m._x_for(db)))  # noqa: E731
        y = int(bar.center().y())
        QTest.mousePress(m, Qt.MouseButton.LeftButton, pos=QPoint(x_at(-60.0), y))
        QTest.mouseMove(m, QPoint(x_at(-40.0), y))
        QTest.mouseRelease(m, Qt.MouseButton.LeftButton, pos=QPoint(x_at(-40.0), y))
        self.assertTrue(seen)
        self.assertAlmostEqual(seen[0], -60.0, delta=1.0)
        self.assertAlmostEqual(seen[-1], -40.0, delta=1.0)
        self.assertAlmostEqual(m.threshold(), seen[-1])
        # A reading while dragging doesn't move the marker away.
        QTest.mousePress(m, Qt.MouseButton.LeftButton, pos=QPoint(x_at(-30.0), y))
        m.set_reading(-50.0, -80.0, -73.0, "closed")
        self.assertAlmostEqual(m.threshold(), -30.0, delta=1.0)
        QTest.mouseRelease(m, Qt.MouseButton.LeftButton, pos=QPoint(x_at(-30.0), y))

    def test_paints_in_both_themes(self):
        from sdr_module.gui import themes

        app = QApplication.instance()
        old_sheet, old_palette = app.styleSheet(), app.palette()
        old_theme = themes.current_theme()
        try:
            images = {}
            for theme in ("dark", "light"):
                themes.apply_theme(app, theme)
                self.meter.set_reading(-30.0, -80.0, -73.0, "open")
                images[theme] = self.meter.grab().toImage()
            # The open lamp is drawn in the theme's success colours.
            lamp = images["dark"].pixelColor(images["dark"].width() - 8, 6)
            self.assertNotEqual(
                lamp, images["light"].pixelColor(images["light"].width() - 8, 6)
            )
        finally:
            app.setStyleSheet(old_sheet)
            app.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestLevelMeterInTheWindow(_WindowTestCase):
    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.meter = self.win._level_meter
        self.panel = self.win._control_panel

    def test_it_replaces_the_level_text(self):
        self.assertIs(self.win._level_label, self.meter)
        self.assertIn(self.meter, self.win._toolbar.findChildren(type(self.meter)))
        self.assertEqual(self.meter.accessibleName(), "Signal level and squelch")
        self.assertTrue(self.meter.text().endswith("dBFS"))
        self.assertIn("channel", self.meter.toolTip())

    def test_reading_and_lamp_follow_the_channel_and_the_squelch(self):
        self.win.set_frequency(162.55e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        Frames(self, noise(15 * FRAME, -80.0, seed=11)).run()
        self.assertEqual(self.meter.state(), "closed")
        self.assertAlmostEqual(self.meter.level(), self.win._last_peak_db)
        self.assertLess(abs(self.meter.snr_db()), 2.5)  # empty: SNR ~ 0
        self.assertAlmostEqual(
            self.meter.threshold(), self.win._squelch.last_threshold_db
        )
        iq = noise(10 * FRAME, -80.0, seed=12) + carrier(10 * FRAME, 1e3, -50.0)
        Frames(self, iq).run()
        self.assertEqual(self.meter.state(), "open")
        self.assertGreater(self.meter.snr_db(), 25)
        self.panel.set_squelch_enabled(False)
        Frames(self, noise(3 * FRAME, -80.0, seed=13)).run()
        self.assertEqual(self.meter.state(), "off")
        self.win._set_demod_mode("None (I/Q)")
        Frames(self, noise(3 * FRAME, -80.0, seed=14)).run()
        self.assertEqual(self.meter.state(), "idle")  # nothing can play

    def test_dragging_on_the_meter_sets_the_squelch_and_the_slider(self):
        self.win.set_frequency(162.55e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.panel.set_squelch_enabled(False)
        Frames(self, noise(3 * FRAME, -80.0, seed=15)).run()
        self.meter.threshold_changed.emit(-42.0)  # what a drag emits
        self.assertTrue(self.panel.is_squelch_enabled())  # turned on
        self.assertFalse(self.panel.is_squelch_auto())
        self.assertEqual(self.panel.get_squelch_db(), -42.0)
        self.assertEqual(self.win._squelch.manual_db, -42.0)
        self.assertEqual(self.store.values["squelch_db"], -42.0)
        Frames(self, noise(2 * FRAME, -80.0, seed=16)).run()
        self.assertEqual(self.meter.threshold(), -42.0)
        # ...and the slider moves the marker.
        self.panel._squelch_slider.setValue(-65)
        Frames(self, noise(2 * FRAME, -80.0, seed=17)).run()
        self.assertEqual(self.meter.threshold(), -65.0)

    def test_stopping_clears_the_reading(self):
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        Frames(self, noise(3 * FRAME, -80.0, seed=18)).run()
        self.assertNotEqual(self.meter.text(), "— dBFS")
        self.win._refresh_state_ui()  # not running
        self.assertEqual(self.meter.text(), "— dBFS")

    def test_info_tab_shows_level_snr_and_squelch(self):
        self.win.set_frequency(162.55e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        iq = noise(10 * FRAME, -80.0, seed=19) + carrier(10 * FRAME, 1e3, -50.0)
        Frames(self, iq).run()
        self.win._is_running = True
        self.win._device = object()
        try:
            self.win._refresh_info_panel()
            info = self.win._info_panel
            self.assertIn("SNR", info.value("level"))
            self.assertIn("Auto", info.value("squelch"))
            self.assertIn("open", info.value("squelch"))
        finally:
            self.win._device = None
            self.win._is_running = False


if __name__ == "__main__":
    unittest.main()
