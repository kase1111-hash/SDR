#!/usr/bin/env python3
"""
Waterfall features: color levels (auto and manual), history length, pause,
measurement, the context menu, the frequency axis and persistence.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_waterfall_features.py``
"""

import json
import os
import types
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMenu

    from sdr_module.gui import waterfall_widget as wfm


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _noise_line(rng, floor=-85.0, carriers=((1024, -30.0),)):
    line = rng.normal(floor, 2.0, 2048)
    for bin_, level in carriers:
        line[bin_ - 3 : bin_ + 4] = level
    return line


class _Clock:
    """A settable stand-in for the waterfall module's ``time``."""

    def __init__(self, t=1000.0):
        self.t = t

    def monotonic(self):
        return self.t


class _WaterfallCase(unittest.TestCase):
    history_size = 100

    def setUp(self):
        require_pyqt6(self)
        self.clock = _Clock()
        patcher = mock.patch.object(
            wfm, "time", types.SimpleNamespace(monotonic=self.clock.monotonic)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.w = wfm.WaterfallWidget(history_size=self.history_size)
        self.w.resize(900, 360)
        self.w.show()
        _settle()
        self.rng = np.random.default_rng(7)

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()

    def feed(self, n, dt=1 / 30, **kwargs):
        for _ in range(n):
            self.clock.t += dt
            self.w.add_line(_noise_line(self.rng, **kwargs))


class TestLevels(_WaterfallCase):
    def test_quantized_codes_are_exact_for_eighth_db_values(self):
        codes = wfm.quantize_db(np.array([-100.0, -20.0, 0.0, -20.125]))
        back = wfm._Q_MIN_DB + codes.astype(float) / wfm._Q_PER_DB
        np.testing.assert_array_equal(back, [-100.0, -20.0, 0.0, -20.125])
        self.assertTrue((wfm.quantize_db([np.nan, -np.inf, np.inf]) > 0).all())

    def test_level_change_remaps_without_reprocessing_history(self):
        self.feed(20)
        with mock.patch.object(self.w, "_render_image") as render:
            self.w.set_db_range(-60.0, -20.0)
        render.assert_not_called()
        self.assertEqual(self.w.levels(), (-60.0, -20.0))
        self.assertFalse(self.w.auto_levels())
        # Pixels use the new levels: -30 dBFS maps through _level_index.
        expected = self.w._colormap[self.w._level_index(np.array([[-30.0]]))[0, 0]]
        pixel = self.w._image.pixelColor(1024, 0)
        self.assertEqual((pixel.red(), pixel.green(), pixel.blue()), tuple(expected))

    def test_display_pixels_follow_a_level_change(self):
        self.feed(20)
        plot = self.w.plot_rect()
        cols = self.w._display_columns(plot)
        self.w._display_image(cols)
        self.w.set_db_range(-70.0, -10.0)
        remapped = self.w._disp_rgb.copy()
        self.w._disp_rgb = None
        self.w._display_image(cols)
        np.testing.assert_array_equal(remapped, self.w._disp_rgb)

    def test_auto_levels_frame_noise_and_signal(self):
        self.w.set_auto_levels(True)
        self.feed(60)
        floor, ceiling = self.w.levels()
        noise = self.w.noise_floor_db()
        self.assertAlmostEqual(noise, -85.0, delta=1.0)
        self.assertLess(floor, noise)
        self.assertGreater(floor, noise - 10)
        self.assertGreater(ceiling, -30.0)
        self.assertLess(ceiling, -20.0)

    def test_auto_levels_ignore_small_wander(self):
        self.w.set_auto_levels(True)
        self.feed(60)
        before = self.w.levels()
        self.feed(60, floor=-84.0)  # 1 dB: inside the hysteresis
        self.assertEqual(self.w.levels(), before)
        self.feed(120, floor=-75.0)  # 10 dB: retargets
        self.assertGreater(self.w.levels()[0], before[0] + 5)

    def test_auto_targets_respect_scale_and_span(self):
        floor, ceiling = wfm.auto_level_targets(-85.0, -30.0)
        self.assertEqual((floor, ceiling), (-91.0, -26.0))
        floor, ceiling = wfm.auto_level_targets(-85.0, -84.0)  # noise only
        self.assertGreaterEqual(ceiling - floor, wfm._AUTO_MIN_SPAN_DB)
        floor, ceiling = wfm.auto_level_targets(-20.0, 5.0)  # hot signal
        self.assertLessEqual(ceiling, 0.0)
        self.assertGreaterEqual(ceiling - floor, wfm._AUTO_MIN_SPAN_DB)

    def test_level_bar_drag_switches_to_manual(self):
        self.w.set_auto_levels(True)
        self.feed(30)
        bar = self.w._level_bar
        floor, ceiling = self.w.levels()
        x = bar._db_to_x(floor)
        y = bar.height() // 2
        QTest.mousePress(bar, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint())
        QTest.mouseMove(bar, QPointF(bar._db_to_x(floor - 20), y).toPoint())
        QTest.mouseRelease(
            bar,
            Qt.MouseButton.LeftButton,
            pos=QPointF(bar._db_to_x(floor - 20), y).toPoint(),
        )
        self.assertFalse(self.w.auto_levels())
        self.assertLess(self.w.levels()[0], floor - 10)
        self.assertEqual(self.w.levels()[1], ceiling)
        QTest.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint())
        self.assertTrue(self.w.auto_levels())

    def test_level_bar_keys(self):
        self.w.set_db_range(-90.0, -20.0)
        bar = self.w._level_bar
        QTest.keyClick(bar, Qt.Key.Key_Left)
        self.assertEqual(self.w.levels(), (-91.0, -20.0))
        QTest.keyClick(bar, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.w.levels(), (-91.0, -15.0))
        QTest.keyClick(bar, Qt.Key.Key_Home)
        self.assertTrue(self.w.auto_levels())

    def test_level_bar_never_inverts(self):
        self.w.set_db_range(-50.0, -40.0)
        bar = self.w._level_bar
        for _ in range(5):
            QTest.keyClick(bar, Qt.Key.Key_Right)
        floor, ceiling = self.w.levels()
        self.assertGreaterEqual(ceiling - floor, wfm._MIN_LEVEL_SPAN_DB)


class TestHistoryAndPause(_WaterfallCase):
    history_size = 500

    def test_live_adds_one_row_per_line(self):
        self.feed(40)
        self.assertEqual(len(self.w._history), 40)

    def test_one_minute_spans_a_minute_at_30_fps(self):
        self.w.set_history_seconds(60)
        self.feed(30 * 60)
        self.assertEqual(len(self.w._history), 500)
        ages = self.w._row_ages()
        self.assertAlmostEqual(ages[-1], 60.0 - 0.12, delta=0.2)

    def test_slow_rows_keep_the_strongest_level(self):
        self.w.set_history_seconds(300)  # 0.6 s per row
        self.clock.t += 1
        quiet = np.full(2048, -90.0)
        burst = quiet.copy()
        burst[500] = -25.0  # one frame of a short burst
        self.w.add_line(quiet)
        self.clock.t += 0.1
        self.w.add_line(burst)
        self.clock.t += 0.1
        self.w.add_line(quiet)
        self.assertEqual(len(self.w._history), 1)
        self.assertEqual(self.w._history[-1][500], -25.0)

    def test_history_menu_and_combo_agree(self):
        self.w._history_combo.setCurrentIndex(self.w._history_combo.findData(900.0))
        self.assertEqual(self.w.history_seconds(), 900.0)
        self.w.set_history_seconds(3600)
        self.assertEqual(self.w._history_combo.currentData(), 3600.0)

    def test_slow_rows_are_not_mistaken_for_pauses(self):
        self.feed(90)  # Live rows
        self.w.set_history_seconds(3600)  # 7.2 s per row
        self.feed(30 * 30)
        _separators, _labels = self.w._time_axis_layout(self.w.plot_rect())
        self.assertEqual(_separators, [])

    def test_pause_skips_lines_and_resumes(self):
        self.feed(10)
        self.w.set_paused(True)
        self.assertEqual(self.w._pause_btn.text(), "Resume")
        self.assertTrue(self.w._pause_btn.isChecked())
        self.feed(10)
        self.assertEqual(len(self.w._history), 10)
        self.assertEqual(self.w._skipped, 10)
        self.w._canvas.grab()  # the PAUSED badge paints
        self.w._pause_btn.click()
        self.assertFalse(self.w.is_paused())
        self.feed(5)
        self.assertEqual(len(self.w._history), 15)

    def test_p_key_toggles_pause(self):
        QTest.keyClick(self.w, Qt.Key.Key_P)
        self.assertTrue(self.w.is_paused())
        QTest.keyClick(self.w, Qt.Key.Key_P)
        self.assertFalse(self.w.is_paused())


class TestMeasureAndClick(_WaterfallCase):
    def _box(self, frac0, frac1, y0, y1):
        plot = self.w.plot_rect()
        a = QPointF(plot.left() + plot.width() * frac0, plot.top() + y0)
        b = QPointF(plot.left() + plot.width() * frac1, plot.top() + y1)
        return a, b

    def test_click_tunes_and_drag_measures(self):
        self.feed(60)
        tuned = []
        self.w.frequency_clicked.connect(tuned.append)
        canvas = self.w._canvas
        a, b = self._box(0.45, 0.56, 10, 90)
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=a.toPoint())
        self.assertEqual(len(tuned), 1)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=a.toPoint())
        QTest.mouseMove(canvas, b.toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=b.toPoint())
        self.assertEqual(len(tuned), 1)  # a drag doesn't tune
        stats = self.w.measurement()
        self.assertIsNotNone(stats)
        span = self.w._sample_rate * 0.11
        self.assertAlmostEqual(stats["bandwidth"], span, delta=span * 0.05)
        self.assertAlmostEqual(stats["peak_db"], -30.0, places=3)
        self.assertGreater(stats["duration"], 0.0)

    def test_measurement_follows_scrolling_rows(self):
        self.feed(60)
        a, b = self._box(0.45, 0.56, 10, 60)
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        first = self.w.measurement()
        rows_before = self.w._measure_rows()
        self.feed(10)
        rows_after = self.w._measure_rows()
        self.assertEqual(rows_after[0], rows_before[0] + 10)
        self.assertAlmostEqual(self.w.measurement()["duration"], first["duration"], 3)

    def test_measurement_drops_when_scrolled_out(self):
        self.feed(60)
        a, b = self._box(0.45, 0.56, 10, 40)
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        self.feed(self.history_size + 5)
        self.w._canvas.grab()
        self.assertIsNone(self.w.measurement())

    def test_escape_clears_measurement(self):
        self.feed(30)
        a, b = self._box(0.4, 0.6, 5, 40)
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        QTest.keyClick(self.w, Qt.Key.Key_Escape)
        self.assertIsNone(self.w.measurement())

    def test_readout_lines(self):
        self.feed(60)
        a, b = self._box(0.45, 0.56, 10, 90)
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        lines = self.w._measure_lines()
        self.assertTrue(lines[0].startswith("Δf "))
        self.assertIn("Δt ", lines[0])
        self.assertTrue(lines[1].startswith("Center "))
        self.assertIn("SNR", lines[2])

    def test_formatters(self):
        self.assertEqual(wfm.format_span_hz(850), "850 Hz")
        self.assertEqual(wfm.format_span_hz(12500), "12.5 kHz")
        self.assertEqual(wfm.format_span_hz(200e3), "200 kHz")
        self.assertEqual(wfm.format_span_hz(1.25e6), "1.250 MHz")
        self.assertEqual(wfm.format_duration(0.25), "250 ms")
        self.assertEqual(wfm.format_duration(1.25), "1.25 s")
        self.assertEqual(wfm.format_duration(125), "2 min 05 s")


class TestMenusAndSettings(_WaterfallCase):
    def _menu_actions(self, pos):
        captured = {}

        def fake_exec(menu, *_args):
            def walk(m):
                for act in m.actions():
                    captured[act.text().replace("&", "")] = act
                    if act.menu():
                        walk(act.menu())

            walk(menu)

        with mock.patch.object(QMenu, "exec", fake_exec):
            self.w._show_context_menu(pos, self.w._canvas.mapToGlobal(pos.toPoint()))
        return captured

    def test_context_menu_tunes_and_bookmarks_the_clicked_frequency(self):
        self.feed(20)
        plot = self.w.plot_rect()
        pos = QPointF(plot.left() + plot.width() * 0.75, plot.top() + 30)
        expected = self.w._snapped_frequency_at(pos.x())
        actions = self._menu_actions(pos)
        tuned, marked = [], []
        self.w.frequency_clicked.connect(tuned.append)
        self.w.bookmark_requested.connect(marked.append)
        tune = next(a for t, a in actions.items() if t.startswith("Tune to 1"))
        mark = next(a for t, a in actions.items() if t.startswith("Bookmark "))
        tune.trigger()
        mark.trigger()
        self.assertEqual(tuned, [expected])
        self.assertEqual(marked, [expected])
        for name in (
            "Pause",
            "Clear History",
            "Save Image...",
            "Auto (follow noise and signals)",
        ):
            self.assertIn(name, actions)

    def test_context_menu_mnemonics_are_unique(self):
        self.feed(20)
        plot = self.w.plot_rect()
        a = QPointF(plot.left() + plot.width() * 0.4, plot.top() + 5)
        b = QPointF(plot.left() + plot.width() * 0.6, plot.top() + 40)
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        seen = {}

        def fake_exec(menu, *_args):
            for act in menu.actions():
                text = act.text()
                if "&" in text:
                    key = text[text.index("&") + 1].lower()
                    self.assertNotIn(key, seen, f"{text!r} vs {seen.get(key)!r}")
                    seen[key] = text

        with mock.patch.object(QMenu, "exec", fake_exec):
            self.w._show_context_menu(a, self.w._canvas.mapToGlobal(a.toPoint()))
        self.assertGreater(len(seen), 8)

    def test_display_settings_round_trip(self):
        self.w.set_colormap("viridis")
        self.w.set_history_seconds(300)
        self.w.set_db_range(-95.0, -25.0)
        saved = self.w.display_settings()
        other = wfm.WaterfallWidget(history_size=10)
        try:
            other.apply_display_settings(json.loads(json.dumps(saved)))
            self.assertEqual(other.display_settings(), saved)
            other.apply_display_settings({"history_s": "junk", "floor_db": None})
            other.apply_display_settings("not a dict")
            self.assertEqual(other.colormap_name(), "viridis")
        finally:
            other.deleteLater()

    def test_settings_changes_are_announced(self):
        changes = []
        self.w.display_settings_changed.connect(lambda: changes.append(1))
        self.w.set_colormap("plasma")
        self.w.set_history_seconds(60)
        self.w.set_auto_levels(True)
        self.assertEqual(len(changes), 3)

    def test_frequency_axis_takes_room_below_the_plot(self):
        before = self.w.plot_rect()
        self.w.set_frequency_axis_visible(True)
        after = self.w.plot_rect()
        self.assertLess(after.height(), before.height())
        self.feed(5)
        self.w._canvas.grab()  # draws the MHz labels

    def test_export_image_includes_axes(self):
        import tempfile

        self.feed(30)
        path = os.path.join(tempfile.mkdtemp(), "wf.png")
        self.assertTrue(self.w.export_image(path))
        self.assertGreater(os.path.getsize(path), 1000)
        self.assertFalse(self.w._freq_axis)  # restored after the export

    def test_narrow_header_hides_captions_before_squeezing(self):
        self.w.resize(560, 300)
        _settle()
        self.assertTrue(self.w._hint_label.isHidden())
        self.w.resize(1400, 300)
        _settle()
        self.assertFalse(self.w._levels_caption.isHidden())


class TestMainWindowWiring(_WindowTestCase):
    demo = True

    def test_defaults_auto_levels_and_one_minute(self):
        wf = self.win._waterfall
        self.assertTrue(wf.auto_levels())
        self.assertEqual(wf.history_seconds(), 60.0)

    def test_settings_saved_on_change(self):
        self.win._waterfall.set_colormap("classic")
        saved = json.loads(self.store.values["waterfall_display"])
        self.assertEqual(saved["colormap"], "classic")

    def test_bookmark_from_the_waterfall(self):
        panel = self.win._bookmarks_panel
        before = len(panel._bookmarks)
        self.win._waterfall.bookmark_requested.emit(101.2e6)
        self.assertEqual(len(panel._bookmarks), before + 1)
        self.assertAlmostEqual(panel._bookmarks[-1]["freq_hz"], 101.2e6)

    def test_hiding_the_spectrum_shows_waterfall_frequencies(self):
        self.win._spectrum_action.setChecked(False)
        self.assertTrue(self.win._waterfall._freq_axis)
        self.win._spectrum_action.setChecked(True)
        self.assertFalse(self.win._waterfall._freq_axis)


if __name__ == "__main__":
    unittest.main()
