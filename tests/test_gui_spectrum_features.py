#!/usr/bin/env python3
"""
Spectrum features: the vertical range (Auto or fixed, dragged and scrolled
on the dB axis; rx:G-02/G-08, ux:UX-G06), the right-click menu (tune,
bookmark, copy, zoom, peak hold, range, FFT size, window), FFT size and
window for both plots with the RBW shown (fg:FG-G16), a LEVEL / squelch
measurement that doesn't change with them, bookmark labels along the top
(ux:UX-U16) and the display choices remembered (rx:UX-12).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_spectrum_features.py``
"""

import json
import os
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
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMenu

    from sdr_module.gui import spectrum_widget as swm

RATE = 2.4e6
CENTER = 100e6
N = 2048


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _line(noise=-110.0, carriers=((100.3e6, -40.0),), seed=0, n=N):
    rng = np.random.default_rng(seed)
    line = rng.normal(noise, 0.5, n)
    for freq, level in carriers:
        b = int(round(n / 2 + (freq - CENTER) / (RATE / n)))
        line[b] = level
    return line


def _menu(show):
    """{plain text: action} of the menu ``show()`` opens, and the menus."""
    captured, menus = {}, []

    def fake_exec(menu, *_args):
        def walk(m):
            menus.append(m)
            for act in m.actions():
                captured[act.text().replace("&", "").split("\t")[0]] = act
                if act.menu():
                    walk(act.menu())

        walk(menu)

    with mock.patch.object(QMenu, "exec", fake_exec):
        show()
    return captured, menus


def _wheel(widget, pos, notches=1, mods=None):
    event = QWheelEvent(
        QPointF(pos),
        QPointF(widget.mapToGlobal(QPointF(pos).toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        mods if mods is not None else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


class _SpectrumCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.sp = swm.SpectrumWidget()
        self.sp.set_frequency_range(CENTER, RATE)
        self.sp.resize(1000, 400)
        self.sp.show()
        self.addCleanup(self.sp.deleteLater)
        _settle()


class TestVerticalRange(_SpectrumCase):
    def test_auto_follows_the_noise_floor_and_the_strongest_signal(self):
        sp = self.sp
        sp.set_auto_range(True)
        for i in range(10):
            sp.update_spectrum(_line(seed=i))
        floor, top = sp.db_range()
        self.assertTrue(floor <= -120.0 <= floor + 15, (floor, top))  # room below
        self.assertTrue(-40.0 < top <= -20.0, (floor, top))
        self.assertEqual(floor % 5, 0)
        # A hotter receiver (gain up 50 dB): the range moves with it.
        for i in range(30):
            sp.update_spectrum(_line(noise=-60.0, carriers=((100.3e6, 10.0),), seed=i))
        floor2, top2 = sp.db_range()
        self.assertGreater(floor2, floor + 40)
        self.assertGreaterEqual(top2, 10.0)

    def test_auto_ignores_small_wander(self):
        sp = self.sp
        sp.set_auto_range(True)
        for i in range(10):
            sp.update_spectrum(_line(seed=i))
        before = sp.db_range()
        for i in range(10):
            sp.update_spectrum(_line(noise=-108.0, seed=20 + i))
        self.assertEqual(sp.db_range(), before)

    def test_fixed_ranges_and_auto_off(self):
        sp = self.sp
        sp.set_auto_range(True)
        sp.update_spectrum(_line())
        top = sp.db_range()[1]
        sp.set_range_span(80)
        self.assertFalse(sp.auto_range())
        self.assertEqual(sp.db_range(), (top - 80, top))
        sp.set_db_range(-120, -20)
        self.assertEqual(sp.db_range(), (-120.0, -20.0))

    def test_drag_scroll_and_double_click_the_db_axis(self):
        sp = self.sp
        sp.set_db_range(-100, 0)
        canvas = sp._canvas
        plot = sp.plot_rect()
        x = plot.left() - 15  # over the dB labels
        y = plot.center().y()
        self.assertTrue(sp._on_db_axis(QPointF(x, y)))
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint())
        QTest.mouseMove(canvas, QPointF(x, y + plot.height() / 10).toPoint())
        QTest.mouseRelease(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=QPointF(x, y + plot.height() / 10).toPoint(),
        )
        # Dragged down a tenth of the height: the scale moved 10 dB with it.
        self.assertEqual(sp.db_range(), (-90.0, 10.0))
        _wheel(canvas, QPointF(x, y), 2)  # up two notches: 10 dB down
        self.assertEqual(sp.db_range(), (-100.0, 0.0))
        _wheel(canvas, QPointF(x, y), 2, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(sp.db_range(), (-80.0, 0.0))  # 20 dB less deep
        sp.update_spectrum(_line())
        QTest.mouseDClick(
            canvas, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint()
        )
        self.assertTrue(sp.auto_range())
        self.assertNotEqual(sp.db_range(), (-80.0, 0.0))

    def test_the_wheel_over_the_plot_still_tunes(self):
        sp = self.sp
        steps = []
        sp.tune_steps.connect(steps.append)
        _wheel(sp._canvas, sp.plot_rect().center(), 1)
        self.assertEqual(steps, [1])
        self.assertEqual(sp.db_range(), (-100, 0))


class TestPeakHoldAfterAGainChange(_SpectrumCase):
    """ux:UX-U09: a gain or AGC change (the noise floor jumps) restarts
    the peak hold, which would otherwise sit far above the new trace."""

    def test_a_jump_of_the_noise_floor_restarts_peak_hold(self):
        sp = self.sp
        sp.reset_on_gain_jump = True
        for i in range(5):
            sp.update_spectrum(_line(noise=-110.0, seed=i))
        self.assertGreater(sp._peak_hold.max(), -45.0)
        # 2 dB of wander keeps what was held...
        sp.update_spectrum(_line(noise=-108.0, carriers=(), seed=9))
        self.assertGreater(sp._peak_hold.max(), -45.0)
        # ...30 dB less gain starts over.
        low = _line(noise=-140.0, carriers=((100.3e6, -70.0),), seed=10)
        sp.update_spectrum(low)
        self.assertAlmostEqual(float(sp._peak_hold.max()), -70.0, delta=0.01)

    def test_off_by_default_for_a_standalone_plot(self):
        sp = self.sp
        self.assertFalse(sp.reset_on_gain_jump)
        sp.update_spectrum(np.full(N, -60.0))
        sp.update_spectrum(np.full(N, -90.0))
        self.assertTrue(np.all(sp._peak_hold == -60.0))


class TestContextMenu(_SpectrumCase):
    def actions(self, frac=0.75):
        plot = self.sp.plot_rect()
        pos = QPointF(plot.left() + plot.width() * frac, plot.center().y())
        return _menu(lambda: self.sp._show_context_menu(pos, QPoint(0, 0))), pos

    def test_tune_bookmark_copy_and_the_display_options(self):
        sp = self.sp
        sp.update_spectrum(_line())
        (actions, menus), pos = self.actions()
        expected = sp._snapped_frequency_at(pos.x())
        tuned, marked, messages = [], [], []
        sp.frequency_clicked.connect(tuned.append)
        sp.bookmark_requested.connect(marked.append)
        sp.message.connect(lambda text, tone: messages.append(text))
        next(a for t, a in actions.items() if t.startswith("Tune to ")).trigger()
        next(a for t, a in actions.items() if t.startswith("Bookmark 1")).trigger()
        next(a for t, a in actions.items() if t.startswith("Copy Frequency")).trigger()
        self.assertEqual(tuned, [expected])
        self.assertEqual(marked, [expected])
        copied = QApplication.clipboard().text()
        self.assertTrue(copied.endswith(" MHz"), copied)
        self.assertEqual(messages, [f"Copied {copied}"])
        for name in (
            "Zoom In Here",
            "Peak Hold",
            "Min Hold",
            "Reset Holds",
            "Range",
            "Auto (follow noise and signals)",
            "80 dB",
            "Zoom",
            "×4",
            "FFT Size",
            "Window",
            "Blackman-Harris",
            "Flat-top",
            "Bookmark Labels",
        ):
            self.assertIn(name, actions)
        actions["Peak Hold"].trigger()
        self.assertFalse(sp.peak_hold())
        actions["80 dB"].trigger()
        self.assertEqual(sp.db_range()[1] - sp.db_range()[0], 80)

    def test_mnemonics_are_unique_in_each_menu(self):
        (_actions, menus), _pos = self.actions()
        for menu in menus:
            seen = {}
            for act in menu.actions():
                text = act.text()
                if "&" in text:
                    key = text[text.index("&") + 1].lower()
                    self.assertNotIn(key, seen, f"{text!r} vs {seen.get(key)!r}")
                    seen[key] = text

    def test_fft_size_and_window_from_the_menu(self):
        sp = self.sp
        chosen = []
        sp.resolution_changed.connect(
            lambda size, window: chosen.append((size, window))
        )
        (actions, _menus), _pos = self.actions()
        item = next(a for t, a in actions.items() if t.startswith("8192 points"))
        self.assertIn("RBW 439 Hz", item.text())
        item.trigger()
        self.assertEqual(chosen[-1], (8192, "hann"))
        self.assertEqual(sp.fft_size(), 8192)
        (actions, _menus), _pos = self.actions()
        actions["Flat-top"].trigger()
        self.assertEqual(chosen[-1], (8192, "flat-top"))
        self.assertAlmostEqual(sp.resolution_bandwidth(), 3.77 * RATE / 8192, delta=5)
        self.assertIn("RBW 1.1 kHz", sp._info_text())

    def test_options_button_opens_the_same_display_options(self):
        actions, _menus = _menu(self.sp._show_options_menu)
        for name in (
            "Peak Hold",
            "Range",
            "Zoom",
            "FFT Size",
            "Window",
            "Bookmark Labels",
        ):
            self.assertIn(name, actions)


class TestStaleLegend(_SpectrumCase):
    def legend(self):
        texts = []

        class Painter:
            def setPen(self, *_a):
                pass

            def drawLine(self, *_a):
                pass

            def drawText(self, *args):
                texts.append(args[-1])

        from sdr_module.gui.themes import get_palette

        self.sp._draw_legend(Painter(), self.sp.plot_rect(), get_palette())
        return texts

    def test_the_legend_says_last_while_stopped_and_shows_the_rbw(self):
        sp = self.sp
        sp.update_spectrum(_line())
        self.assertIn("Live", self.legend())
        sp.set_stale("Stopped - press Space")
        texts = self.legend()
        self.assertNotIn("Live", texts)
        self.assertIn("Last", texts)
        self.assertIn("RBW 1.76 kHz", texts)
        sp._canvas.grab()  # the dimmed plot and its badge paint
        sp.update_spectrum(_line(seed=2))
        self.assertIsNone(sp.stale_text())


class TestMinHold(_SpectrumCase):
    """fg:FG-G16: a min-hold trace next to peak hold."""

    def test_min_hold_keeps_the_lowest_level_per_bin(self):
        sp = self.sp
        self.assertFalse(sp.min_hold())
        changes = []
        sp.display_settings_changed.connect(lambda: changes.append(1))
        sp.set_min_hold(True)
        self.assertEqual(len(changes), 1)
        sp.update_spectrum(np.full(N, -60.0))
        sp.update_spectrum(np.full(N, -80.0))
        sp.update_spectrum(np.full(N, -70.0))
        self.assertTrue(np.all(sp._min_hold == -80.0))
        self.assertTrue(np.all(sp._peak_hold == -60.0))
        sp._canvas.grab()  # draws the min trace
        # Reset clears both holds; a retune too.
        sp._peak_reset_btn.click()
        self.assertTrue(np.all(np.isnan(sp._min_hold)))
        sp.update_spectrum(np.full(N, -75.0))
        sp.set_center_freq(101e6)
        self.assertTrue(np.all(np.isnan(sp._min_hold)))

    def test_reset_button_follows_either_hold_and_the_setting_persists(self):
        sp = self.sp
        sp.set_peak_hold(False)
        self.assertFalse(sp._peak_reset_btn.isEnabled())
        sp.set_min_hold(True)
        self.assertTrue(sp._peak_reset_btn.isEnabled())
        self.assertIn("min-hold", sp._peak_reset_btn.toolTip())
        saved = sp.display_settings()
        self.assertTrue(saved["min_hold"])
        other = swm.SpectrumWidget()
        try:
            other.apply_display_settings(saved)
            self.assertTrue(other.min_hold())
            self.assertFalse(other.peak_hold())
        finally:
            other.deleteLater()

    def test_the_legend_lists_min(self):
        sp = self.sp
        sp.set_min_hold(True)
        sp.update_spectrum(_line())
        texts = []

        class Painter:
            def setPen(self, *_a):
                pass

            def drawLine(self, *_a):
                pass

            def drawText(self, *args):
                texts.append(args[-1])

        from sdr_module.gui.themes import get_palette

        sp._draw_legend(Painter(), sp.plot_rect(), get_palette())
        self.assertIn("Min", texts)


class TestClear(_SpectrumCase):
    def test_clear_brings_back_the_placeholder(self):
        sp = self.sp
        sp.update_spectrum(_line())
        sp.set_stale("Stopped - press Space")
        sp.clear()
        self.assertFalse(sp._has_data)
        self.assertIsNone(sp.stale_text())
        self.assertTrue(np.all(sp._peak_hold <= -120.0))
        sp._canvas.grab()  # the "No spectrum yet" placeholder paints


class TestBookmarkLabels(_SpectrumCase):
    def boxes(self):
        self.sp._canvas.grab()
        return self.sp._bookmark_boxes

    def test_labels_for_bookmarks_in_view_and_a_click_tunes(self):
        sp = self.sp
        sp.set_bookmarks(
            [
                (100.1e6, "FM Broadcast 100.1", 0),
                (99.5e6, "Classic", 1),
                (145.8e6, "ISS", 2),  # outside the span
            ]
        )
        boxes = self.boxes()
        self.assertEqual(sorted(index for _b, index, _f, _n in boxes), [0, 1])
        clicked = []
        sp.bookmark_clicked.connect(clicked.append)
        box = next(b for b, index, _f, _n in boxes if index == 1)
        QTest.mouseClick(
            sp._canvas, Qt.MouseButton.LeftButton, pos=box.center().toPoint()
        )
        self.assertEqual(clicked, [1])
        for box, _index, _f, _n in boxes:
            self.assertTrue(sp.plot_rect().contains(box))

    def test_overlapping_labels_are_left_out_and_long_names_elided(self):
        sp = self.sp
        sp.set_tuned_frequency(100.2e6)
        sp.set_bookmarks(
            [
                (100.2e6, "Near the tuned frequency", 0),
                (100.201e6, "Right next to it", 1),
                (100.6e6, "A very long bookmark name that goes on and on and on", 2),
            ]
        )
        boxes = self.boxes()
        shown = sorted(index for _b, index, _f, _n in boxes)
        self.assertEqual(shown, [0, 2])  # the one nearest the tuned wins
        for box, _index, _f, _n in boxes:
            self.assertLessEqual(box.width(), swm.BOOKMARK_LABEL_MAX_PX + 10)

    def test_the_labels_can_be_turned_off(self):
        sp = self.sp
        sp.set_bookmarks([(100.1e6, "FM", 0)])
        self.assertTrue(self.boxes())
        sp.set_bookmark_labels_visible(False)
        self.assertEqual(self.boxes(), [])
        self.assertIsNone(sp._bookmark_at(sp.plot_rect().center()))


class TestWindowSpectrumSettings(_WindowTestCase):
    demo = True

    def frames(self, n=6, samples=80000):
        win = self.win
        for _ in range(n):
            win._process_block(win._device.read_samples(samples))

    def test_fft_size_and_window_reach_both_plots_and_the_info_tab(self):
        win = self.win
        win._spectrum._choose_resolution(8192, "blackman-harris")
        self.assertEqual(win._frame_spectrum.size, 8192)
        self.assertEqual(win._frame_spectrum.window, "blackman-harris")
        self.assertEqual(win._waterfall.fft_size(), 8192)
        self.assertIn("8192-point FFT", win._message_label.text())
        self.frames(3)
        self.assertEqual(len(win._spectrum._spectrum), 8192)
        self.assertEqual(len(win._waterfall._history[-1]), 8192)
        win._show_panel("info")
        win._refresh_info_panel()
        rbw = win._info_panel.value("rbw")
        self.assertIn("8192-point FFT", rbw)
        self.assertIn("Blackman-Harris", rbw)

    def test_the_rbw_reads_the_same_everywhere(self):
        # The menu and the band above the plot say "RBW 110 Hz"; the status
        # message and the Info tab said "0.11 kHz" for the same thing.
        win = self.win
        win._spectrum._choose_resolution(32768, "hann")
        shown = swm.format_hz(win._spectrum.resolution_bandwidth())
        self.assertEqual(shown, "110 Hz")
        self.assertIn(f"RBW {shown}", win._spectrum._info_text())
        self.assertIn(f"RBW {shown}", win._message_label.text())
        win._show_panel("info")
        win._refresh_info_panel()
        self.assertTrue(win._info_panel.value("rbw").startswith(f"{shown} ("))

    def test_level_and_squelch_measure_the_same_whatever_the_display(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._stop_acquisition()
        rng = np.random.default_rng(3)
        n = 80000
        block = (rng.normal(0, 0.003, n) + 1j * rng.normal(0, 0.003, n)).astype(
            np.complex64
        )

        def measure():
            # The same start each time: a fresh receiver chain and trackers.
            win._rx_chain = None
            win._noise_floor.reset()
            win._level_average.reset()
            win._process_block(block.copy())
            return win._noise_floor.bin_db, win._last_peak_db

        default = measure()
        win._spectrum._choose_resolution(32768, "flat-top")
        win._zoom_in_action.trigger()
        changed = measure()
        self.assertEqual(default, changed)

    def test_spectrum_display_choices_survive_a_restart(self):
        win = self.win
        sp = win._spectrum
        sp._choose_resolution(4096, "flat-top")
        sp.set_range_span(60)
        sp.set_bookmark_labels_visible(False)
        win._zoom_in_action.trigger()
        win._persist_state()
        saved = json.loads(self.store.values["spectrum_display"])
        self.assertEqual(saved["fft_size"], 4096)
        self.assertEqual(saved["window"], "flat-top")
        self.assertFalse(saved["range_auto"])
        self.assertEqual(saved["zoom"], 2)
        self.assertFalse(saved["bookmark_labels"])
        other = self.mw.SDRMainWindow()
        try:
            for timer in (
                other._display_timer,
                other._status_timer,
                other._hotplug_timer,
            ):
                timer.stop()
            osp = other._spectrum
            self.assertEqual(osp.fft_size(), 4096)
            self.assertEqual(other._frame_spectrum.window, "flat-top")
            self.assertEqual(other._waterfall.fft_size(), 4096)
            self.assertFalse(osp.auto_range())
            floor, top = osp.db_range()
            self.assertEqual(top - floor, 60)
            self.assertFalse(osp.bookmark_labels_visible())
            self.assertEqual(other._plot_view.zoom, 2)
        finally:
            other.close()
            other.deleteLater()

    def test_auto_range_is_the_default(self):
        self.assertTrue(self.win._spectrum.auto_range())
        self.assertTrue(self.win._spectrum.reset_on_gain_jump)

    def test_peak_hold_starts_off_in_the_window(self):
        # ux:UX-U09: one trace on a first start; Peak hold is in the header.
        sp = self.win._spectrum
        self.assertFalse(sp.peak_hold())
        self.assertFalse(sp._peak_check.isChecked())
        self.frames(3)
        texts = []

        class Painter:
            def setPen(self, *_a):
                pass

            def drawLine(self, *_a):
                pass

            def drawText(self, *args):
                texts.append(args[-1])

        sp._draw_legend(Painter(), sp.plot_rect(), swm.get_palette())
        self.assertIn("Live", texts)
        self.assertNotIn("Peak", texts)

    def test_bookmark_labels_follow_the_bookmarks_and_tune(self):
        win = self.win
        panel = win._bookmarks_panel
        win.set_frequency(100.1e6)
        win._bookmark_frequency(99.5e6)
        labels = {round(freq) for freq, _name, _i in win._spectrum._bookmarks}
        self.assertIn(round(99.5e6), labels)
        index = next(
            i for f, _n, i in win._spectrum._bookmarks if round(f) == round(99.5e6)
        )
        win._spectrum.bookmark_clicked.emit(index)
        self.assertAlmostEqual(win._current_frequency(), 99.5e6)
        self.assertGreater(panel.bookmark_count(), 0)

    def test_spectrum_bookmark_from_the_context_menu(self):
        win = self.win
        panel = win._bookmarks_panel
        before = panel.bookmark_count()
        win._spectrum.bookmark_requested.emit(101.3e6)
        self.assertEqual(panel.bookmark_count(), before + 1)


if __name__ == "__main__":
    unittest.main()
