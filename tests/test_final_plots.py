"""Final fixes for the plots and the Signal tab, found by the end-to-end
journeys (PLOTS-E2E-*, ROB-*) and the leftovers list.

Each test names the problem it covers; they run the real window offscreen
on the demo device (or the panel on its own where that is enough).
"""

import os
import unittest
from itertools import combinations

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_signal_tab import FRAME_S, _SignalCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest


def _burst_report(freq_hz, start_s=1.0):
    from sdr_module.dsp.bursts import Burst, BurstReport

    return BurstReport(Burst(start_s, start_s + 0.35, freq_hz, 20e3, -40.0, 35.0))


def _survey_signal(freq_hz, width_hz=12e3):
    from sdr_module.dsp.signal_list import SurveySignal

    return SurveySignal(freq_hz, -50.0, 25.0, width_hz, 0.0, 6.0, 4, 4)


class _PlotsCase(_WindowTestCase):
    demo = True

    def at(self, width, height):
        """The window at ``width`` x ``height`` with its default layout (as
        a window opened at that size has it)."""
        self.win.resize(width, height)
        self.settle()
        self.win._apply_default_splitter_sizes()
        self.settle()

    def in_view(self, widget):
        """Whether ``widget`` is entirely inside the Signal tab's viewport
        (no scrolling needed to reach it)."""
        area = self.win._panel_pages["signal"]
        viewport = area.viewport()
        top = widget.mapTo(viewport, QPoint(0, 0)).y()
        return top >= 0 and top + widget.height() <= viewport.height()

    @staticmethod
    def rows_in_view(table):
        viewport = table.viewport()
        return sum(
            1
            for row in range(table.rowCount())
            if table.visualRect(table.model().index(row, 0)).top() >= 0
            and table.visualRect(table.model().index(row, 0)).bottom()
            <= viewport.height()
        )


class TestBurstsLayout(_PlotsCase):
    """PLOTS-E2E-1 / ROB-11: at 1366x768 the Bursts list showed one row (64
    px) over an empty 134 px details box."""

    def show_bursts(self, count=6):
        panel = self.win._signal_panel
        self.win._show_signal_page("bursts")
        for i in range(count):
            panel.add_burst(_burst_report(433.92e6 + 1e3 * i, start_s=i))
        self.settle()
        return panel

    def check(self, width, height):
        self.at(width, height)
        panel = self.show_bursts()
        table = panel.bursts_table
        self.assertGreaterEqual(self.rows_in_view(table), 3, (width, height))
        # No details yet: a one-line hint says how to get them.
        self.assertTrue(panel.burst_details.isHidden())
        self.assertTrue(panel.bursts_hint.isVisible())
        # A burst selected: its details under the list, which keeps three
        # rows and the larger share.
        table.selectRow(2)
        self.settle()
        self.assertTrue(panel.burst_details.isVisible())
        self.assertFalse(panel.bursts_hint.isVisible())
        from sdr_module.gui.signal_panel import burst_details_text

        self.assertEqual(
            panel.burst_details.toPlainText(),
            burst_details_text(panel.selected_burst()),
        )
        self.assertGreaterEqual(self.rows_in_view(table), 3, (width, height))
        self.assertGreaterEqual(table.height(), panel.burst_details.height())
        # Nothing selected: the list gets the room back.
        table.clearSelection()
        self.settle()
        self.assertTrue(panel.burst_details.isHidden())

    def test_at_1366x768(self):
        self.check(1366, 768)

    def test_at_1024x640(self):
        self.check(1024, 640)

    def test_clear_hides_the_details(self):
        panel = self.show_bursts()
        panel.bursts_table.selectRow(0)
        self.settle()
        panel.clear_bursts()
        self.settle()
        self.assertTrue(panel.burst_details.isHidden())
        self.assertFalse(panel.bursts_hint.isVisible())


class TestFindSignalsLayout(_PlotsCase):
    """PLOTS-E2E-6: after Find Signals in Recording, the 'In <file>: Back to
    Live' line pushed Play / Bookmark / Strongest / ◀ / ▶ below the tab's
    view at 1366x768."""

    def test_the_buttons_stay_in_view(self):
        self.at(1366, 768)
        panel = self.win._signal_panel
        self.win._show_signal_page("signals")
        signals = [_survey_signal(100.1e6 + 0.2e6 * i) for i in range(5)]
        panel.set_recording_signals("demo_take.sigmf-data", signals, 6.0)
        self.settle()
        area = self.win._panel_pages["signal"]
        self.assertEqual(area.verticalScrollBar().maximum(), 0)
        for button in (
            panel.tune_btn,
            panel.bookmark_btn,
            panel.strongest_btn,
            panel.down_btn,
            panel.up_btn,
            panel.live_btn,
        ):
            self.assertTrue(self.in_view(button), button.text())
        self.assertEqual(panel.tune_btn.text(), "Play")
        self.assertIn("demo_take.sigmf-data", panel.recording_label.text())
        # On the line of the live list's options (which don't apply).
        self.assertFalse(panel.min_snr.isVisible())
        self.assertFalse(panel.markers_check.isVisible())
        panel.live_btn.click()
        self.settle()
        self.assertTrue(panel.min_snr.isVisible())
        self.assertTrue(panel.markers_check.isVisible())
        self.assertFalse(panel.live_btn.isVisible())

    def test_a_long_name_fits_on_its_line(self):
        self.at(1024, 640)
        panel = self.win._signal_panel
        self.win._show_signal_page("signals")
        name = "a_very_long_recording_name_from_last_saturday_night.sigmf-data"
        panel.set_recording_signals(name, [_survey_signal(100.1e6)], 6.0)
        self.settle()
        label = panel.recording_label
        self.assertIn("a_very_long_re", label.text())  # (the whole text)
        self.assertLessEqual(
            label.fontMetrics().horizontalAdvance(label.displayed_text()),
            label.width(),
        )
        self.assertIn(name, label.toolTip())
        self.assertTrue(self.in_view(panel.live_btn))


class TestSignalsListCost(_PlotsCase):
    """ROB-05: refreshing the Signals list cost rows squared on the GUI
    thread (each item set made the 'resize to contents' headers measure
    every row again): 1.3 s for 80 rows, a 100-170 ms stall twice a second
    with the 22 signals of the FM band at 20 MS/s."""

    def test_eighty_rows_refresh_in_a_few_ms(self):
        import time

        from sdr_module.dsp.signal_list import SignalPeak

        panel = self.win._signal_panel
        self.win._show_signal_page("signals")
        signals = [
            SignalPeak(88e6 + 0.25e6 * i, -40.0 - i % 7, 20.0 + i % 9, 150e3)
            for i in range(80)
        ]
        ages = [0.5 * (i % 5) for i in range(80)]
        panel.set_signals(signals, 100e6, ages)  # (the first fill)
        self.settle()
        start = time.thread_time()
        panel.set_signals(signals, 100e6, ages)
        self.settle(1)
        spent = time.thread_time() - start
        self.assertLess(spent, 0.15, f"{spent * 1e3:.0f} ms of processor time")
        table = panel.signals_table
        self.assertEqual(table.rowCount(), 80)
        # Still sized to what they hold.
        width = table.fontMetrics().horizontalAdvance(table.item(0, 3).text())
        self.assertGreaterEqual(table.columnWidth(3), width)


class TestWaterfallSaveImage(_PlotsCase):
    """PLOTS-E2E-7: Waterfall > Save Image... opened in the folder the app
    was started from, always offered 'waterfall.png', forgot the folder and
    said nothing after saving."""

    def test_like_save_screenshot(self):
        import re
        import shutil
        import tempfile
        from unittest import mock

        from sdr_module.gui import file_dialogs

        other = tempfile.mkdtemp(prefix="wf-images-")
        self.addCleanup(shutil.rmtree, other, True)
        offered = []

        def save(_parent, caption, path, *_args):
            offered.append((caption, path))
            return os.path.join(other, os.path.basename(path)), "PNG image (*.png)"

        wf = self.win._waterfall
        with mock.patch.object(
            file_dialogs.QFileDialog, "getSaveFileName", side_effect=save
        ):
            wf._save_image_dialog()
            first = sorted(os.listdir(other))
            self.assertIn(
                "Saved the waterfall image to", self.win._message_label.text()
            )
            wf._save_image_dialog()
        caption, path = offered[0]
        self.assertEqual(caption, "Save Waterfall Image")
        self.assertEqual(
            os.path.dirname(path), file_dialogs.default_folder("screenshot")
        )
        self.assertRegex(
            os.path.basename(path), r"^sdr_waterfall_\d+\.\d{3}MHz_\d{8}_\d{6}\.png$"
        )
        self.assertTrue(re.search(r"\.png$", first[0]), first)
        # The folder used is remembered (in the window's settings).
        self.assertEqual(os.path.dirname(offered[1][1]), other)
        self.assertEqual(
            file_dialogs.remembered_folder(self.win._settings, "screenshot"), other
        )

    def test_a_plot_without_settings_remembers_while_the_app_runs(self):
        import shutil
        import tempfile
        from unittest import mock

        from sdr_module.gui import file_dialogs
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        other = tempfile.mkdtemp(prefix="wf-images-")
        self.addCleanup(shutil.rmtree, other, True)
        offered = []

        def save(_parent, _caption, path, *_args):
            offered.append(path)
            return os.path.join(other, "mine.png"), ""

        wf = WaterfallWidget()
        self.addCleanup(wf.deleteLater)
        with mock.patch.object(
            file_dialogs.QFileDialog, "getSaveFileName", side_effect=save
        ):
            wf._save_image_dialog()
            wf._save_image_dialog()
        self.assertEqual(os.path.dirname(offered[1]), other)


class TestWaterfallNoiseBrightness(unittest.TestCase):
    """PLOTS-E2E-13: a waterfall row is the strongest of its frame's FFTs,
    so its noise read brighter the more there were: rows after a stall
    (about 1.6 dB) and the rows of a 1 or 5 min history (merging frames)
    looked brighter, one-block frames (a busy computer) 6.5 dB darker,
    like a change in the band. Rows are coloured at one noise brightness;
    their levels (what the readouts say) stay as measured."""

    BINS = 1024

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)

    def line(self, ffts, rng, floor_db=-80.0):
        power = rng.exponential(1.0, (ffts, self.BINS)).max(axis=0)
        return floor_db + 10.0 * np.log10(power)

    @staticmethod
    def shown_db(wf):
        codes = wf._row_codes[-1][1].astype(float)
        return float(np.median(codes / 8.0 - 200.0))

    def test_rows_of_any_length_show_the_same_noise(self):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        rng = np.random.default_rng(3)
        wf = WaterfallWidget()
        self.addCleanup(wf.deleteLater)
        wf.set_frequency_range(100e6, 2.4e6)
        normal = []
        for i in range(60):
            wf.add_line(self.line(39, rng), timestamp=i / 30, noise_db=-80.0)
            normal.append(self.shown_db(wf))
        for ffts in (290, 1):
            wf.add_line(self.line(ffts, rng), timestamp=2.1, noise_db=-80.0)
            raw = float(np.median(wf._history[-1]))
            self.assertGreater(abs(raw - float(np.median(normal))), 1.2, ffts)
            self.assertAlmostEqual(self.shown_db(wf), np.median(normal), delta=0.6)
        # A real change of the noise floor still shows.
        wf.add_line(self.line(39, rng, -70.0), timestamp=2.2, noise_db=-70.0)
        self.assertAlmostEqual(self.shown_db(wf) - np.median(normal), 10.0, delta=1.0)

    def test_a_row_merging_frames_is_as_bright(self):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        rng = np.random.default_rng(4)
        wf = WaterfallWidget(history_size=50)
        self.addCleanup(wf.deleteLater)
        wf.set_frequency_range(100e6, 2.4e6)
        wf.set_history_seconds(60.0)  # a row every 1.2 s, once full
        t = 0.0
        while len(wf._history) < 50:  # (the fast fill: a row per line)
            wf.add_line(self.line(39, rng), timestamp=t, noise_db=-80.0)
            t += 1 / 30
        single = self.shown_db(wf)
        rows = len(wf._history)
        for _ in range(30):  # one 1.2 s row, about 30 frames merged
            t += 1 / 30
            wf.add_line(self.line(39, rng), timestamp=t, noise_db=-80.0)
        self.assertGreater(
            float(np.median(wf._history[-1])) - single, 1.0
        )  # (its levels: the strongest of 30 frames)
        self.assertAlmostEqual(self.shown_db(wf), single, delta=0.6)
        self.assertLessEqual(len(wf._history), rows + 2)


class TestSignalsList(_SignalCase):
    """Leftovers 1 and 17, PLOTS-E2E-20: the Signals list was measured
    every half second of the wall clock, the first time from a single
    frame (after a reset or a new SNR threshold), in which a wide FM station
    shows only where its modulation is at that instant: pieces of stations
    off their centres were listed (and kept 10 s). Page Up tuned to one,
    and, when it retuned the receiver, Page Down said "Looking for
    signals"; on a busy computer the keys also found the list too old."""

    STATIONS = (99.5e6, 99.82e6, 100.1e6, 101.1e6)  # in view (100.715: off)

    def setUp(self):
        super().setUp()
        self.win._device._rng = np.random.default_rng(17)  # (its noise)
        self.listen(100.3e6, ("FM", "75 kHz", "200 kHz"))
        self.win._show_signal_page("signals")

    def run_frames(self, count):
        self.frames(count * FRAME_S + 1e-9)

    def freqs(self):
        """The listed frequencies, to the kHz."""
        return sorted(round(s.frequency_hz, -3) for s in self.panel.signals())

    def assert_no_pieces(self, stations=STATIONS):
        for freq in self.freqs():
            self.assertLess(
                min(abs(freq - want) for want in stations), 1e3, self.freqs()
            )

    def key(self, key):
        QTest.keyClick(self.win._waterfall, key)
        return self.win._current_frequency()

    def test_a_list_is_measured_from_many_frames(self):
        self.run_frames(3)
        self.assertEqual(self.tools._heard, [])  # (not from a frame or two)
        self.run_frames(40)
        self.assertEqual(self.freqs(), list(self.STATIONS))
        widths = {round(s.frequency_hz): s.width_hz for s in self.panel.signals()}
        self.assertGreater(widths[round(101.1e6)], 90e3)

    def test_the_keys_step_on_across_a_retune(self):
        self.run_frames(40)
        self.win._waterfall.setFocus()
        self.settle()
        self.assertEqual(self.key(Qt.Key.Key_Home), 100.1e6)
        lo = self.win._lo_frequency()
        self.assertEqual(self.key(Qt.Key.Key_PageUp), 101.1e6)
        self.assertNotEqual(self.win._lo_frequency(), lo)  # (near the edge)
        self.assertEqual(self.key(Qt.Key.Key_PageDown), 100.1e6)
        self.assertNotIn("Looking for signals", self.win._message_label.text())
        self.assertAlmostEqual(self.key(Qt.Key.Key_PageDown), 99.82e6, delta=50)
        self.run_frames(20)  # (measured again in the new span)
        self.assertEqual(self.key(Qt.Key.Key_PageDown), 99.5e6)

    def test_a_slow_computer_lists_the_same(self):
        """(Leftover 1: on a busy machine a frame took longer than the half
        second the list was measured over.)"""
        import types
        from unittest import mock

        from sdr_module.gui import signal_tools

        clock = [1000.0]

        def monotonic():
            clock[0] += 0.7  # every look at the clock: a slow frame later
            return clock[0]

        fake = types.SimpleNamespace(monotonic=monotonic)
        with mock.patch.object(signal_tools, "time", fake):
            self.run_frames(40)
            self.assertEqual(self.freqs(), list(self.STATIONS))
            self.win._waterfall.setFocus()
            self.settle()
            self.assertEqual(self.key(Qt.Key.Key_Home), 100.1e6)
            self.assertEqual(self.key(Qt.Key.Key_PageUp), 101.1e6)
            self.assertEqual(self.key(Qt.Key.Key_PageDown), 100.1e6)

    def test_a_new_snr_threshold_lists_again_at_once(self):
        self.run_frames(40)
        self.panel.min_snr.setValue(30.0)
        self.assertEqual(self.freqs(), [99.5e6, 100.1e6])
        self.run_frames(60)
        self.assert_no_pieces()
        self.win._waterfall.setFocus()
        self.settle()
        self.assertEqual(self.key(Qt.Key.Key_Home), 100.1e6)
        up = self.key(Qt.Key.Key_PageUp)
        self.assertIn(up, (100.1e6, 101.1e6))  # (101.1 MHz: about 30 dB)
        self.panel.min_snr.setValue(10.0)
        self.assertEqual(self.freqs(), list(self.STATIONS))


class TestIdentifyThroughDrops(_SignalCase):
    """PLOTS-E2E-2: when the receiver drops samples every second or so, the
    half seconds left whole may hear only noise (the remote's packets fell
    in the gaps), and Identify said "Nothing identifiable ... Tune onto a
    signal and try again", as if nothing were there, though seconds of the
    signal had been lost."""

    def run_with_drops(self, freq, drop_s):
        self.listen(freq, ("None (I/Q)", None, "200 kHz"))
        self.assertTrue(self.tools.identify())
        dev = self.win._device
        rate = self.win._device_sample_rate()
        for _ in range(12):
            if self.tools._identifier is None:
                break
            self.frames(0.6)  # (half a second whole: a take)
            if drop_s:
                dev._stream_pos += int(drop_s * rate)
                dev.dropped_samples += int(drop_s * rate)
        for job in [self.tools._ident_job] if self.tools._ident_job else []:
            job.result(timeout=30)
        self.frames(0.2)
        self.assertIsNone(self.tools._identifier)
        return self.panel.identify_status.text()

    def test_nothing_heard_says_samples_were_lost(self):
        status = self.run_with_drops(430.5e6, 0.5)  # (nothing there)
        self.assertIn("dropped", status)
        self.assertNotIn("Tune onto a signal", status)

    def test_nothing_there_says_so(self):
        status = self.run_with_drops(430.5e6, 0.0)
        self.assertIn("Tune onto a signal", status)


class TestBurstsPausedWithRows(_SignalCase):
    """While the display loop is behind, bursts are not logged (ROB-03):
    only an empty list said so; with bursts in it, it just stopped growing
    without a word. The line under the list says why, until it catches
    up."""

    def test_the_line_under_the_list_says_why(self):
        win, panel = self.win, self.panel
        self.listen(433.92e6, ("None (I/Q)", None, None))
        win._show_signal_page("bursts")
        self.assertTrue(self.frames(6.0, until=lambda: bool(panel.bursts())))
        self.bursts_done()
        hint = panel.bursts_hint
        normal = hint.text()
        self.assertTrue(hint.isVisible())
        for _ in range(int(1.0 / FRAME_S)):  # (one block a frame)
            win._pacer.reset()
            win._pacer.overloaded = True
            win._process_block(win._device.read_samples(2048))
        self.assertTrue(hint.isVisible())
        self.assertIn("can't keep up", hint.text())
        self.assertIn("can't keep up", hint.toolTip())
        self.assertEqual(hint.property("tone"), "warning")
        self.frames(0.5)
        self.assertEqual(hint.text(), normal)
        self.assertFalse(hint.property("tone"))


class TestBurstOnTheWaterfall(_SignalCase):
    """Leftover 4: the Signal tab's bursts were never shown on the
    waterfall. The selected one is outlined there, around the rows it was
    received in, while the Bursts page is on screen."""

    def test_the_selected_burst_is_outlined(self):
        win, panel = self.win, self.panel
        wf = win._waterfall
        self.listen(433.92e6, ("None (I/Q)", None, None))
        win._show_signal_page("bursts")
        self.frames(3.4)
        self.bursts_done()
        reports = [r for _stamp, r in panel.bursts()]
        row = next(
            i for i, r in enumerate(reports) if abs(r.burst.center_hz - 433.92e6) < 20e3
        )
        burst = reports[row].burst
        self.assertIsNone(wf.burst_mark_rect())
        panel.bursts_table.selectRow(row)
        rect = wf.burst_mark_rect()
        self.assertIsNotNone(rect)
        plot = wf.plot_rect()
        # Around its frequency...
        self.assertLessEqual(wf.frequency_at(rect.left()), burst.center_hz)
        self.assertGreaterEqual(wf.frequency_at(rect.right()), burst.center_hz)
        # ...and the rows it was received in (and no others).
        rows_px = plot.height() / wf._rows_shown()
        first = int(round((rect.top() - plot.top()) / rows_px))
        last = int(round((rect.bottom() - plot.top()) / rows_px)) - 1

        def level(row):
            return wf.level_at(rect.center().x(), plot.top() + (row + 0.5) * rows_px)

        inside = [level(row) for row in range(first, last + 1)]
        self.assertGreater(min(inside), burst.peak_dbfs - 6.0, inside)
        for row in (first - 1, last + 1):
            if 0 <= row < len(wf._times):
                self.assertLess(level(row), burst.peak_dbfs - 20.0, row)
        wf._canvas.grab()  # (paints it)
        # Another page, or no selection: no outline.
        win._show_signal_page("signals")
        self.assertIsNone(wf.burst_mark_rect())
        win._show_signal_page("bursts")
        self.assertIsNotNone(wf.burst_mark_rect())
        panel.bursts_table.clearSelection()
        self.assertIsNone(wf.burst_mark_rect())


class TestHoverReadsOnePoint(unittest.TestCase):
    """PLOTS-E2E-19: with the pointer on the edge of a station the hover
    read the station's frequency (where a click lands) with the level under
    the pointer, as if they were one point. The first line is now the
    pointer's own frequency, and the hint says where a click tunes."""

    CENTER = 100.1e6
    RATE = 2.4e6
    BINS = 2048

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        self.rng = np.random.default_rng(5)

    def line(self):
        """Noise with a 100 kHz wide station at 100.5 MHz."""
        offsets = (np.arange(self.BINS) - self.BINS / 2) * self.RATE / self.BINS
        line = self.rng.normal(-85.0, 1.5, self.BINS)
        station = np.abs(self.CENTER + offsets - 100.5e6) <= 50e3
        line[station] = self.rng.normal(-45.0, 1.0, int(station.sum()))
        return line

    def readout(self, module, widget, x, y):
        from unittest import mock

        from PyQt6.QtCore import QPointF

        seen = []

        def record(_painter, lines, *_args, **_kwargs):
            seen.append(list(lines))

        with mock.patch.object(module, "draw_readout", record):
            widget._set_hover(QPointF(x, y))
            widget._canvas.grab()
        widget._set_hover(None)
        self.assertTrue(seen)
        return seen[-1]

    def check(self, module, widget, landing_at, y_of):
        from sdr_module.gui.spectrum_widget import (
            readout_decimals,
            snap_step_hz,
            tuned_mhz_text,
        )

        widget.resize(900, 360)
        widget.show()
        self.addCleanup(widget.close)
        plot = widget.plot_rect()
        x = plot.left() + widget._view.fraction_of(100.535e6) * plot.width()
        y = y_of(plot)
        decimals = readout_decimals(snap_step_hz(widget._hz_per_px()))
        here = widget._snapped_frequency_at(x)
        self.assertAlmostEqual(here, 100.535e6, delta=3e3)

        widget.set_click_snap(100e3, "fm")  # a click snaps to the station
        landing = landing_at(x, y)
        self.assertAlmostEqual(landing, 100.5e6, delta=1e3)
        lines = self.readout(module, widget, x, y)
        self.assertEqual(lines[0], tuned_mhz_text(here, decimals), lines)
        self.assertTrue(
            lines[-1].startswith(
                f"Click to tune to {tuned_mhz_text(landing, decimals)}"
            ),
            lines,
        )
        self.assertTrue(any("dBFS" in line for line in lines[1:-1]), lines)

        widget.set_click_snap(0)  # a click tunes where the pointer is
        lines = self.readout(module, widget, x, y)
        self.assertEqual(lines[0], tuned_mhz_text(here, decimals), lines)
        self.assertTrue(lines[-1].startswith("Click to tune here"), lines)

    def test_spectrum(self):
        import sdr_module.gui.spectrum_widget as module

        widget = module.SpectrumWidget()
        self.addCleanup(widget.deleteLater)
        widget.set_frequency_range(self.CENTER, self.RATE)
        for _ in range(4):
            widget.update_spectrum(self.line())
        self.check(
            module,
            widget,
            lambda x, _y: widget.landing_frequency(x),
            lambda plot: plot.center().y(),
        )

    def test_waterfall(self):
        import sdr_module.gui.waterfall_widget as module

        widget = module.WaterfallWidget(history_size=100)
        self.addCleanup(widget.deleteLater)
        widget.set_frequency_range(self.CENTER, self.RATE)
        for i in range(40):
            widget.add_line(self.line(), timestamp=i / 30, noise_db=-85.0)
        self.check(
            module,
            widget,
            lambda x, y: widget.landing_frequency(x, y),
            lambda plot: plot.top() + 3,
        )


class TestTimeAxisAfterAPause(unittest.TestCase):
    """PLOTS-E2E-14: after Stop and Start the waterfall's time axis read
    0s, 1s and the pause's age, then nothing for the older rows below it
    (most of the plot): only the newest run had regular ticks. Each run
    now has its own, counting on from its pause's age."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)

    def layout(self, times, history_size=300):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        wf = WaterfallWidget(history_size=history_size)
        self.addCleanup(wf.deleteLater)
        wf.set_frequency_range(100e6, 2.4e6)
        wf.resize(940, 372)
        wf.show()
        self.addCleanup(wf.close)
        wf._times.clear()
        wf._times.extend(times)
        plot = wf.plot_rect()
        separators, labels = wf._time_axis_layout(plot)
        return wf, plot, separators, labels

    def check_ages(self, wf, plot, separators, labels):
        """Every regular label's text is the age of the row its tick marks
        (a pause's label sits on its separator, between two rows)."""
        from sdr_module.gui.waterfall_widget import format_elapsed

        ages = wf._row_ages()
        rows_px = plot.height() / wf._rows_shown()
        for tick_y, _y, text in labels:
            if tick_y in separators:
                continue
            row = (tick_y - plot.top()) / rows_px - 0.5
            age = float(np.interp(row, np.arange(len(ages)), ages))
            if text.endswith("m"):
                self.assertAlmostEqual(float(text[:-1]) * 60, age, delta=30, msg=text)
            else:
                self.assertIn(text, (f"{age:.0f}s", f"{age:.1f}s", format_elapsed(age)))

    def test_older_rows_keep_their_ages(self):
        # 5.5 s of lines, Stop for about 2.3 s, then 1.5 s after Start.
        times = [1000 + i / 30 for i in range(165)]
        times += [1007.8 + i / 30 for i in range(45)]
        wf, plot, separators, labels = self.layout(times)
        self.assertEqual(len(separators), 1)
        below = [label for label in labels if label[0] > separators[0] + 1]
        self.assertGreaterEqual(len(below), 3, labels)
        self.check_ages(wf, plot, separators, labels)
        fh = wf._fm.height()
        for (_t1, y1, s1), (_t2, y2, s2) in combinations(labels, 2):
            self.assertGreaterEqual(abs(y1 - y2), fh + 2, (s1, s2))

    def test_an_hour_old_run_reads_in_minutes(self):
        # A run from an hour ago keeps labels that fit the margin, and the
        # newest run keeps its fine ticks.
        times = [1000 + i / 30 for i in range(165)]
        times += [4600 + i / 30 for i in range(135)]
        wf, plot, separators, labels = self.layout(times)
        self.assertEqual(len(separators), 1)
        self.assertTrue(all(len(text) <= 4 for _t, _y, text in labels), labels)
        above = [label for label in labels if label[0] < separators[0]]
        self.assertGreaterEqual(len(above), 3, labels)
        self.check_ages(wf, plot, separators, labels)


class TestCompactCheckedButtons(unittest.TestCase):
    """PLOTS-E2E-9: the Scope's Freeze button read 'reez' while frozen:
    a checked, focused button took the wide padding of the theme's double
    focus ring, whatever its role (compact buttons are narrow)."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)

    def test_freeze_reads_whole_when_frozen(self):
        from PyQt6.QtWidgets import QApplication, QStyle, QStyleOptionButton

        from sdr_module.gui.signal_panel import SignalPanel
        from sdr_module.gui.themes import get_stylesheet

        def content_width(button):
            option = QStyleOptionButton()
            button.initStyleOption(option)
            rect = button.style().subElementRect(
                QStyle.SubElement.SE_PushButtonContents, option, button
            )
            return rect.width()

        for theme in ("dark", "light"):
            panel = SignalPanel()
            self.addCleanup(panel.deleteLater)
            panel.setStyleSheet(get_stylesheet(theme))
            panel.resize(400, 300)
            panel.show()
            panel.show_page("scope")
            QApplication.processEvents()
            button = panel.freeze_btn
            text = button.fontMetrics().horizontalAdvance(button.text())
            self.assertGreaterEqual(content_width(button), text, theme)
            button.setChecked(True)
            button.setFocus()
            QApplication.processEvents()
            self.assertTrue(button.hasFocus())
            self.assertGreaterEqual(content_width(button), text, theme)
            panel.close()


class TestSignalPanelDetails(unittest.TestCase):
    """The Signal tab's small things: the scope's axis (PLOTS-E2E-10), the
    Identify card's filter (-11), the bursts' bits, CSV and tuning (-17)
    and one EVM (-18)."""

    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.gui.signal_panel import SignalPanel

        self.panel = SignalPanel()
        self.addCleanup(self.panel.deleteLater)

    def test_the_scope_axis_tells_small_levels_apart(self):
        """PLOTS-E2E-10: the magnitude axis read '0.00 ... 0.00' for the
        remote (2.5e-3), the CW beacon (4.6e-4) and an airband station."""
        from sdr_module.gui.signal_panel import _short_number

        assert _short_number(1.1 * 2.5e-3) == "2.8m"
        assert _short_number(4.6e-4 * 1.1) == "506µ"
        assert _short_number(-2.5e-3) == "-2.5m"
        assert _short_number(3.1e-8) == "31n"
        assert _short_number(0.15) == "0.15"
        assert _short_number(0.0) == "0.00"
        assert _short_number(3.14159) == "3.1"
        assert _short_number(150.0) == "150"
        for value in (2.5e-3, 4.6e-4, 3.1e-4, 7e-6, -9.9e-3):
            assert len(_short_number(value)) <= 5, value
            assert _short_number(1.1 * value) != _short_number(0.0)

    def test_the_card_names_the_filter_apply_mode_sets(self):
        """PLOTS-E2E-11: the card said '61.3 kHz filter' for the 434.10 MHz
        FSK burst, and Apply Mode set 100 kHz (the Filter list's choice)."""
        from sdr_module.dsp.classifiers import ClassificationResult, SignalType
        from sdr_module.dsp.signal_id import Identification

        result = ClassificationResult(
            SignalType.DIGITAL, None, 0.8, 51e3, 0.0, 25.0, {}, kind="fsk"
        )
        ident = Identification(
            result, 434.1e6, "FSK data", "FM", "25 kHz", 61.3e3, ["Occupies 51 kHz"]
        )
        self.panel.show_identification(ident)
        text = self.panel.listen_label.text()
        self.assertIn("FM · 25 kHz deviation · 100 kHz filter", text)
        self.assertNotIn("61.3", text)

    def report(self, bits, kind="ook", label_mod="OOK PWM", count=8):
        from sdr_module.dsp.bursts import Burst, BurstReport, PulseAnalysis
        from sdr_module.dsp.classifiers import ClassificationResult, SignalType

        analysis = PulseAnalysis(label_mod, "", [350.0] * 10, [], [], [])
        analysis.rows = [bits] * count
        result = ClassificationResult(
            SignalType.DIGITAL, None, 0.9, 50e3, 0.0, 30.0, {}, kind=kind
        )
        burst = Burst(1.0, 1.4, 434.100256870e6, 53.9e3, -56.0, 22.0)
        return BurstReport(burst, analysis=analysis, classification=result)

    def test_long_bits_are_cut_with_an_ellipsis(self):
        """PLOTS-E2E-17: 'FSK PCM {136}c60bc73fd91824c57fe5df4f': cut at 24
        hex digits without saying so, the tooltip as cut."""
        bits = "1100" * 34  # {136}cccc... (34 hex digits)
        self.panel.add_burst(self.report(bits, "fsk", "FSK PCM", 1))
        item = self.panel.bursts_table.item(0, 4)
        self.assertEqual(item.text(), "FSK PCM {136}" + "c" * 24 + "…")
        self.assertEqual(item.toolTip(), "FSK PCM {136}" + "c" * 34)
        short = "1010" * 6
        self.panel.add_burst(self.report(short))
        self.assertEqual(
            self.panel.bursts_table.item(0, 4).text(), "OOK PWM {24}aaaaaa ×8"
        )

    def test_the_csv_type_is_what_it_is(self):
        """PLOTS-E2E-17: Type repeated Modulation ('OOK PWM','OOK PWM')."""
        import csv
        import shutil
        import tempfile

        self.panel.add_burst(self.report("1010" * 6))
        folder = tempfile.mkdtemp(prefix="bursts-")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "b.csv")
        self.panel.export_bursts(path)
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["Type"], "OOK data bursts")
        self.assertEqual(rows[0]["Modulation"], "OOK PWM")

    def test_double_click_tunes_to_a_round_frequency(self):
        """PLOTS-E2E-17: the 434.1 MHz FSK burst tuned to 434.100256870 MHz
        (now 434.1003, to 100 Hz as the Signals list does)."""
        self.panel.add_burst(self.report("1010" * 6, "fsk", "FSK PCM"))
        tuned = []
        self.panel.tune_requested.connect(tuned.append)
        self.panel.bursts_table.selectRow(0)
        self.panel._tune_selected_burst()
        self.assertEqual(tuned, [434.1003e6])

    def test_a_new_burst_leaves_the_selected_one_shown(self):
        """A new burst arriving at the top of the list moved the details
        (and the waterfall's outline, leftover 4) to the burst that took
        the selected one's place, while the selection stayed on its row."""
        from sdr_module.dsp.bursts import Burst, BurstReport
        from sdr_module.gui.signal_panel import burst_details_text

        panel = self.panel
        reports = [
            BurstReport(Burst(i, i + 0.3, f, 20e3, -40.0, 30.0))
            for i, f in enumerate((433.92e6, 433.93e6, 433.94e6))
        ]
        for report in reports:
            panel.add_burst(report)
        panel.bursts_table.selectRow(1)  # (433.93 MHz)
        selected = panel.selected_burst()
        self.assertIs(selected, reports[1])
        marked = []
        panel.burst_selected.connect(marked.append)
        panel.add_burst(BurstReport(Burst(5.0, 5.3, 434.1e6, 50e3, -56.0, 22.0)))
        self.assertIs(panel.selected_burst(), selected)
        self.assertEqual(
            panel.burst_details.toPlainText(), burst_details_text(selected)
        )
        self.assertTrue(all(report is selected for report in marked), marked)

    def test_one_evm(self):
        """PLOTS-E2E-18: the plot said 'EVM 43.6 %' and the line under it
        44.2 %: one was measured when painting, the other read the paint
        before. Both are measured once per set of samples."""
        from sdr_module.ui.constellation import ModulationOverlay

        scope = self.panel.scope
        scope.set_view("constellation")
        scope.set_overlay(ModulationOverlay.QPSK)
        rng = np.random.default_rng(1)
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, 4096)))
        noise = 0.3 * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        scope.set_samples((symbols + noise).astype(np.complex64), 48e3)
        evm = scope.evm_percent
        self.assertIsNotNone(evm)
        scope.resize(300, 200)
        scope.grab()  # (painting doesn't measure it again)
        self.assertEqual(scope.evm_percent, evm)
        scope.set_overlay(ModulationOverlay.NONE)
        self.assertIsNone(scope.evm_percent)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
