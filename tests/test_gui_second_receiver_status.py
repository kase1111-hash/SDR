"""M-R5: a long message in the Second Receiver's status line is readable in
full.

Its status line ended a long message in "…" with the whole text only in
its tooltip (at its minimum width, 860 px, the message area keeps about
520 px beside RUNNING, the device and the rate). Now a message the line
cuts short also shows whole in a panel just above the status bar while it
lasts, in its tone (the main window's ``status_history.MessageOverflow``);
Esc or a click hides it until the next message, and a message that fits
shows as before.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _SecondCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

#: One of the longest messages there: the recording's gaps (_note_record_lost).
LONG = (
    "The recording has gaps: 1.8 s of signal was lost because this "
    "computer's disk couldn't keep up with 3.2 MS/s. A lower sample rate "
    "helps."
)
SHORT = "Stopped; Demo Device is closed."


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _StatusCase(_SecondCase):
    def setUp(self):
        super().setUp()
        rx = self.rx
        self.assertTrue(rx.start())  # (RUNNING, the device and the rate shown)
        rx._frame_timer.stop()
        rx.resize(rx.minimumWidth(), rx.minimumHeight())
        self.settle(10)

    def panel(self):
        return self.rx._message_overflow


class TestTheWholeMessage(_StatusCase):
    def test_a_long_message_shows_whole_above_the_status_bar(self):
        rx = self.rx
        self.assertEqual((rx.width(), rx.height()), (860, 560))
        rx._show_message(LONG, "warning", 8000)
        self.settle(5)
        self.assertTrue(rx._message_label.is_elided())  # (cut in the line)
        panel = self.panel()
        self.assertTrue(panel.isVisible())
        self.assertEqual(panel.message(), LONG)
        self.assertEqual(panel.text(), LONG)
        self.assertEqual(panel.property("tone"), "warning")
        # Just above the status bar, inside the window.
        bar_top = rx.statusBar().mapTo(rx, QPoint(0, 0)).y()
        box = panel.geometry()
        self.assertLessEqual(box.bottom(), bar_top)
        self.assertGreaterEqual(box.left(), 0)
        self.assertLessEqual(box.right(), rx.width())
        self.assertGreater(box.top(), 0)
        # Screen readers get the whole text too.
        self.assertEqual(rx._message_label.accessibleDescription(), LONG)
        # Its tooltip says only what works here (no status history).
        self.assertNotIn("last 20", panel.toolTip())

    def test_a_message_that_fits_shows_as_before(self):
        rx = self.rx
        rx._show_message(SHORT, None, 4000)
        self.settle(5)
        self.assertFalse(rx._message_label.is_elided())
        self.assertFalse(self.panel().isVisible())

    def test_it_goes_with_its_message(self):
        rx = self.rx
        rx._show_message(LONG, "danger", 8000)
        self.assertTrue(self.panel().isVisible())
        rx._message_timer.timeout.emit()  # (its time is up)
        self.assertFalse(self.panel().isVisible())
        self.assertEqual(rx._message_label.accessibleDescription(), "")
        rx._show_message(LONG, "danger", 8000)
        rx._show_message(SHORT, None, 4000)  # (another in its place)
        self.assertFalse(self.panel().isVisible())

    def test_esc_hides_it_and_still_goes_back_to_the_plots(self):
        rx = self.rx
        rx._show_message(LONG, "warning", 8000)
        field = rx._freq_input._freq_input
        field.setFocus()
        self.settle()
        QTest.keyClick(field, Qt.Key.Key_Escape)
        self.assertFalse(self.panel().isVisible())
        self.assertEqual(rx.message(), LONG)  # (still in the status line)
        self.assertTrue(rx._plots_focused())
        # Hidden for that message; the next one shows.
        rx._show_message(LONG, "warning", 8000)
        self.assertFalse(self.panel().isVisible())
        rx._show_message(LONG + " Again.", "warning", 8000)
        self.assertTrue(self.panel().isVisible())

    def test_a_click_hides_it(self):
        rx = self.rx
        rx._show_message(LONG, "warning", 8000)
        QTest.mouseClick(self.panel(), Qt.MouseButton.LeftButton)
        QTest.qWait(QApplication.doubleClickInterval() + 50)
        self.assertFalse(self.panel().isVisible())
        self.assertEqual(rx.message(), LONG)

    def test_in_both_themes(self):
        from sdr_module.gui.themes import apply_theme, current_theme

        app = QApplication.instance()
        before = current_theme()
        self.addCleanup(lambda: apply_theme(app, before))
        rx = self.rx
        for theme in ("light", "dark"):
            apply_theme(app, theme)
            self.settle(5)
            rx._show_message(LONG, "danger", 8000)
            self.settle(5)
            self.assertTrue(self.panel().isVisible(), theme)
            self.assertEqual(self.panel().property("tone"), "danger")
            rx.grab()  # (paints)
            rx._clear_message()


if __name__ == "__main__":
    unittest.main()
