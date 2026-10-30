#!/usr/bin/env python3
"""
Spectrum and waterfall display behaviour in the main window: every sample of
a frame is analysed (a short burst shows), the waterfall fills at once, P
pauses it from either plot, Copy Frequency confirms with the unit, and the
spectrum's display choices are remembered.

Regression tests for rx:UX-04, rx:UX-02 / ux:UX-U02, rx:UX-13, rx:UX-11 and
rx:UX-12.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_plot_display.py``
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

RATE = 2.4e6


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _noise(n, level_db=-70.0, seed=0):
    rng = np.random.default_rng(seed)
    sigma = 10 ** (level_db / 20) / np.sqrt(2)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex64)


def _menu_actions(show_menu):
    """Every action of the menu ``show_menu()`` would open, by plain text."""
    captured = {}

    def fake_exec(menu, *_args):
        def walk(m):
            for act in m.actions():
                captured[act.text().replace("&", "").split("\t")[0]] = act
                if act.menu():
                    walk(act.menu())

        walk(menu)

    with mock.patch.object(QMenu, "exec", fake_exec):
        show_menu()
    return captured


class TestWholeFrame(_WindowTestCase):
    """rx:UX-04: the plots analyse the whole frame, not its last 2048
    samples."""

    demo = True

    def setUp(self):
        super().setUp()
        self.win._stop_acquisition()
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("None (I/Q)")

    def test_a_short_burst_early_in_the_frame_reaches_the_waterfall(self):
        win = self.win
        win._waterfall.set_history_seconds(0)
        n = 80000  # one 33 ms frame at 2.4 MS/s
        frame = _noise(n)
        t = np.arange(4800) / RATE  # a 2 ms burst at +300 kHz, early on
        frame[1000:5800] += (0.1 * np.exp(2j * np.pi * 300e3 * t)).astype(np.complex64)
        win._process_block(frame)
        row = np.asarray(win._waterfall._history[-1], dtype=float)
        n_bins = len(row)
        burst_bin = n_bins // 2 + int(round(300e3 / (RATE / n_bins)))
        noise = float(np.median(row))
        self.assertGreater(row[burst_bin - 2 : burst_bin + 3].max(), noise + 30.0)
        # The spectrum's trace (the frame's mean power) shows it too.
        trace = win._spectrum._spectrum
        self.assertGreater(
            trace[burst_bin - 2 : burst_bin + 3].max(), float(np.median(trace)) + 15.0
        )

    def test_the_trace_is_the_mean_of_the_frame(self):
        """Averaging the frame's segments makes the noise trace smooth: its
        bins scatter far less than one 2048-point FFT's."""
        win = self.win
        win._process_block(_noise(80000, seed=4))
        trace = np.asarray(win._spectrum._spectrum, dtype=float)
        core = trace[200:-200]
        self.assertLess(float(np.std(core)), 2.0)  # one FFT: about 5.6 dB


class TestFastFill(unittest.TestCase):
    """rx:UX-02 / ux:UX-U02: with a minute of history the waterfall is not
    blank for a minute."""

    def setUp(self):
        require_pyqt6(self)
        self.t = 1000.0
        patcher = mock.patch.object(
            wfm, "time", types.SimpleNamespace(monotonic=lambda: self.t)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.w = wfm.WaterfallWidget()
        self.w.resize(900, 420)
        self.w.show()
        _settle()
        self.w.set_history_seconds(60)

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()

    def feed(self, seconds, fps=30):
        rng = np.random.default_rng(1)
        for _ in range(int(seconds * fps)):
            self.t += 1 / fps
            line = rng.normal(-85, 2, 2048)
            line[1000:1010] = -30.0
            self.w.add_line(line)

    def test_rows_arrive_at_the_live_rate_until_the_history_is_full(self):
        self.feed(1.0)
        self.assertEqual(len(self.w._history), 30)

    def test_the_rows_fill_the_plot_from_the_first_second(self):
        self.feed(1.0)
        image = self.w._canvas.grab().toImage()
        plot = self.w.plot_rect()
        from sdr_module.gui.themes import get_palette

        bg = get_palette().qcolor("plot_bg").rgb()
        x = int(plot.left() + plot.width() * 0.25)
        y = int(plot.bottom() - 4)  # near the bottom of the plot
        self.assertNotEqual(image.pixel(x, y), bg)

    def test_it_settles_into_the_chosen_interval(self):
        self.feed(25.0)  # more than 500 rows at 30 per second
        rows = len(self.w._history)
        self.assertEqual(rows, 500)
        self.feed(1.2)
        steps = list(self.w._row_steps)[-5:]
        self.assertTrue(all(abs(s - 0.12) < 1e-9 for s in steps), steps)
        # The time axis covers what the rows hold: about 25 s, not 60 s.
        ages = self.w._row_ages()
        self.assertLess(ages[-1], 30.0)


class TestPauseKey(_WindowTestCase):
    """rx:UX-13: P pauses the waterfall whenever a plot has the focus."""

    def test_p_with_the_spectrum_focused(self):
        win = self.win
        win._spectrum.setFocus()
        self.settle()
        self.assertTrue(win._spectrum.hasFocus())
        QTest.keyClick(win._spectrum, Qt.Key.Key_P)
        self.assertTrue(win._waterfall.is_paused())
        QTest.keyClick(win._spectrum, Qt.Key.Key_P)
        self.assertFalse(win._waterfall.is_paused())

    def test_p_with_a_header_combo_focused_is_left_to_it(self):
        win = self.win
        combo = win._spectrum._avg_combo
        combo.setFocus()
        self.settle()
        QTest.keyClick(combo, Qt.Key.Key_P)
        self.assertFalse(win._waterfall.is_paused())

    def test_p_in_a_text_field_types(self):
        win = self.win
        field = win._frequency_field()
        field.setFocus()
        self.settle()
        QTest.keyClick(field, Qt.Key.Key_P)
        self.assertFalse(win._waterfall.is_paused())


class TestCopyFrequency(_WindowTestCase):
    """rx:UX-11: Copy Frequency says what it copied, with the unit."""

    def test_waterfall_copy_confirms_in_the_status_bar(self):
        wf = self.win._waterfall
        plot = wf.plot_rect()
        pos = QPointF(plot.left() + plot.width() * 0.75, plot.top() + 20)
        actions = _menu_actions(
            lambda: wf._show_context_menu(pos, wf._canvas.mapToGlobal(pos.toPoint()))
        )
        copy = next(a for t, a in actions.items() if t.startswith("Copy Frequency"))
        copy.trigger()
        clipboard = QApplication.clipboard().text()
        self.assertTrue(clipboard.endswith("MHz"), clipboard)
        message = self.win._message_label.text()
        self.assertIn("Copied", message)
        self.assertIn(clipboard, message)


class TestSpectrumSettingsRemembered(_WindowTestCase):
    """rx:UX-12: the spectrum's AVG and Peak hold survive a restart."""

    def test_avg_and_peak_hold_are_saved_and_restored(self):
        sp = self.win._spectrum
        sp._avg_combo.setCurrentIndex(3)  # 8 frames
        sp._peak_check.setChecked(False)
        self.win._persist_state()
        saved = json.loads(self.store.values["spectrum_display"])
        self.assertEqual(saved["avg"], 8)
        self.assertFalse(saved["peak_hold"])
        other = self.mw.SDRMainWindow()
        try:
            self.assertEqual(other._spectrum._avg_combo.currentText(), "8")
            self.assertFalse(other._spectrum._peak_check.isChecked())
        finally:
            for timer in (
                other._display_timer,
                other._status_timer,
                other._hotplug_timer,
            ):
                timer.stop()
            other.close()
            other.deleteLater()


if __name__ == "__main__":
    unittest.main()
