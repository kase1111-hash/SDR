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

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
