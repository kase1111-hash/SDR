"""The window area on the merged build (I-window): what two areas' final
fixes do together, and the window's integration to-dos (the Second
Receiver at its minimum width, the right column on wide windows). Each
test failed before its fix.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_final_integration_window.py``
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication, QLabel


def _in_each_theme(check):
    """Run ``check(theme)`` in each theme the app runs with (their fonts and
    padding are not the bare Fusion style's), then put the old look back."""
    from sdr_module.gui import themes

    qapp = QApplication.instance()
    sheet, palette, theme_now = (
        qapp.styleSheet(),
        qapp.palette(),
        themes.current_theme(),
    )
    try:
        for theme in themes.THEMES:
            themes.apply_theme(qapp, theme)
            check(theme)
    finally:
        qapp.setStyleSheet(sheet)
        qapp.setPalette(palette)
        themes.apply_theme(None, theme_now)


# --------------------------------------------------------------------------- #
# The Second Receiver at its minimum width (860 px: its default on a small
# screen). F-record gave its demo device the main window's longer name and
# put the saved file in Stop's message; neither fitted there.
# --------------------------------------------------------------------------- #
class TestSecondReceiverAtItsMinimumWidth(_SecondCase):
    def narrow(self):
        rx = self.rx
        rx.resize(rx.minimumWidth(), rx.minimumHeight())
        self.settle(5)
        self.assertEqual(rx.width(), 860)

    def test_the_device_list_shows_the_whole_name(self):
        # It read "Demo Device (simulated si" (the tooltip had the rest).
        from PyQt6.QtWidgets import QStyle, QStyleOptionComboBox

        from sdr_module.gui.second_receiver import DEMO_NAME

        combo = self.rx._source_combo
        self.assertEqual(combo.currentText(), DEMO_NAME)

        def check(theme):
            self.narrow()
            option = QStyleOptionComboBox()
            combo.initStyleOption(option)
            field = combo.style().subControlRect(
                QStyle.ComplexControl.CC_ComboBox,
                option,
                QStyle.SubControl.SC_ComboBoxEditField,
                combo,
            )
            need = combo.fontMetrics().horizontalAdvance(DEMO_NAME)
            self.assertGreaterEqual(field.width(), need, theme)

        _in_each_theme(check)

    def test_stop_shows_where_the_recording_went(self):
        # Stop said "Stopped; Demo Device (simulated signals) is closed.
        # Saved ~/Documents/SDR Recordi…": the file name was cut off.
        folder = tempfile.mkdtemp(prefix="iw-rx2-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        rx = self.rx
        rx._audio = _FakeSink()

        def check(theme):
            self.narrow()
            self.assertTrue(rx.start())
            self.assertTrue(rx._set_recording(True))
            name = os.path.basename(rx._record_path)
            for _ in range(3):
                rx._frame()
            rx.stop()
            self.settle(3)
            shown = QLabel.text(rx._message_label)  # (as elided on screen)
            self.assertIn(name, shown, theme)
            text = rx.message()
            self.assertIn("Demo Device (simulated signals) is closed", text)
            self.assertIn("File > Open Recording", text)
            self.assertEqual(rx._message_label.property("tone"), "success")

        _in_each_theme(check)


# --------------------------------------------------------------------------- #
# The right column on wide windows: it stayed 440 px at most, so the
# Decoder's Aircraft list scrolled sideways at every window size.
# --------------------------------------------------------------------------- #
class TestRightColumnOnWideWindows(_WindowTestCase):
    demo = True

    def at(self, width, height, default=False):
        win = self.win
        win.resize(width, height)
        self.settle(5)
        if default:
            win._apply_default_splitter_sizes()
            self.settle(5)
        return win._main_splitter.sizes()

    def test_laptop_sizes_keep_their_column(self):
        # (The plots there keep every pixel they had.)
        self.assertAlmostEqual(self.at(1366, 768, default=True)[1], 406, delta=2)
        self.assertAlmostEqual(self.at(1024, 640, default=True)[1], 400, delta=2)

    def test_a_wide_window_gives_the_column_its_share(self):
        self.at(1366, 768, default=True)
        plots, right = self.at(1920, 1080)
        self.assertAlmostEqual(right / (plots + right), 0.3, delta=0.01)
        self.assertGreaterEqual(plots, 1300)
        # Back to the laptop size: the column as it was.
        self.assertAlmostEqual(self.at(1366, 768)[1], 406, delta=2)
        # On a very wide window it stops at what the Aircraft list needs.
        plots, right = self.at(2560, 1440)
        self.assertLessEqual(right, 660)
        self.assertGreaterEqual(right, 640)

    def test_the_aircraft_list_fits_a_wide_window(self):
        from tests.test_gui_decoder_views import _adsb

        self.at(2560, 1440, default=True)
        win = self.win
        win._show_panel("decoder")
        panel = win._decoder_panel
        view = panel._aircraft_view
        panel._tabs.setCurrentWidget(view)
        for msg in (
            _adsb(
                icao_address="4840D6", downlink_format=17, type_code=4,
                callsign="KLM1023",
            ),
            _adsb(
                icao_address="4840D6", downlink_format=17, type_code=11,
                altitude=38000, latitude=-33.8688, longitude=151.2093,
            ),
            _adsb(
                icao_address="4840D6", downlink_format=17, type_code=19,
                velocity=450.0, heading=265.0, vertical_rate=-1500,
            ),
            _adsb(icao_address="4840D6", downlink_format=5, squawk="7421"),
        ):  # fmt: skip
            view.update_message(msg)
        self.settle(5)
        table = view.table
        last = table.columnCount() - 1
        need = sum(table.columnWidth(c) for c in range(last)) + max(
            table.sizeHintForColumn(last),
            table.horizontalHeader().sectionSizeHint(last),
        )
        self.assertGreaterEqual(table.viewport().width(), need)

    def test_a_saved_layout_keeps_its_share_on_another_screen(self):
        # QSplitter restores the column's width in pixels: a layout saved at
        # 1920 px (a 572-px column) took 572 of 1366 px, leaving the plots
        # 778. It is the share that comes back.
        self.at(1920, 1080, default=True)
        self.win._persist_state()
        second = self.mw.SDRMainWindow(demo_mode=True)
        self.addCleanup(second.deleteLater)
        self.addCleanup(second.close)
        for timer in (second._display_timer, second._status_timer):
            timer.stop()
        second.resize(1366, 768)
        second.show()
        self.settle(10)
        self.assertTrue(second._splitters_restored)
        self.assertAlmostEqual(second._main_splitter.sizes()[1], 406, delta=2)

    def test_a_dragged_column_keeps_its_share(self):
        win = self.win
        self.at(1366, 768, default=True)
        splitter = win._main_splitter
        total = sum(splitter.sizes())
        splitter.setSizes([total - 480, 480])
        splitter.splitterMoved.emit(total - 480, 1)  # (as a drag does)
        plots, right = self.at(1600, 900)
        self.assertAlmostEqual(right / (plots + right), 480 / total, delta=0.01)
        # Layout reset gives the default share back.
        win._reset_layout()
        self.settle(5)
        plots, right = self.at(1920, 1080)
        self.assertAlmostEqual(right / (plots + right), 0.3, delta=0.01)


if __name__ == "__main__":
    unittest.main()
