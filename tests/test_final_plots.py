"""Final fixes for the plots and the Signal tab, found by the end-to-end
journeys (PLOTS-E2E-*, ROB-*) and the leftovers list.

Each test names the problem it covers; they run the real window offscreen
on the demo device (or the panel on its own where that is enough).
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


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
