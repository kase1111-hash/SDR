"""K-W1: a status message too long for the status bar is readable in full.

At 1024 px with every status chip shown (REC, the recording's size, AUDIO,
the decoder chip and AFC) the message area keeps 220 px, so a long message
was cut: "Stop recording to change the sample rate: one I/Q recording holds
one rate." read "Stop recording to change the sam…", the rest only in its
tooltip and in the status history (a click away). Such a message now shows
in full in a small panel just above the status bar, in the message's tone,
for as long as the message shows; a click or Esc hides it. The status bar
and the plots stay where they are (each message used to move the status
bar's chips a pixel and back), and a message that fits shows as before.
"""

import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, QRect, Qt
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLabel

    from tests.test_gui_chrome import _WindowTestCase
else:  # pragma: no cover - no Qt
    _WindowTestCase = unittest.TestCase

# What the sample rate's refusal says while recording (device_controls).
REFUSAL = "Stop recording to change the sample rate: one I/Q recording holds one rate."
# One of the longest messages: the take's gaps (_report_lost_samples).
GAPS = (
    "The recording has gaps: 1.2 s of signal was lost because this computer "
    "(or its disk) couldn't keep up with 2.4 MS/s. A lower sample rate "
    "(Receiver > Sample rate) helps."
)
SHORT = "Retuned to 100.100 MHz"


def _themes(case, check):
    from tests.test_gui_coherent_window import _ThemedSizes

    _ThemedSizes.each_theme(case, check)


def _second_click(widget, pos):
    """A double-click's second click as Qt delivers it to ``widget`` when
    the widget the first click pressed is gone: the double-click event,
    then the release."""
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    left = Qt.MouseButton.LeftButton
    for kind, buttons in (
        (QEvent.Type.MouseButtonDblClick, left),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            kind,
            QPointF(pos),
            QPointF(widget.mapToGlobal(pos)),
            left,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(widget, event)


class _StatusCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.win._status_history.clear()

    def every_chip(self):
        """REC, the size, AUDIO, the decoder chip and AFC, as at 1024 px in
        tests/test_final_window.py::TestStatusBarRoom."""
        from sdr_module.gui.themes import set_role

        win = self.win
        win._recording_label.setText("REC 00:12")
        win._recording_label.setVisible(True)
        win._recording_info_label.setText("26.7 MB · 17.7 GB free")
        win._recording_info_label.setVisible(True)
        win._audio_rec_label.setText("AUDIO 00:05")
        win._audio_rec_label.setVisible(True)
        win._decoder_chip.setText("POCSAG · 6 msgs")
        win._decoder_chip.setVisible(True)
        afc = getattr(self, "_afc", None)
        if afc is None:
            afc = self._afc = QLabel("AFC +2.2 kHz")  # (as Track Signal shows it)
            set_role(afc, "badge", "info")
            win.statusBar().addPermanentWidget(afc)
            self.addCleanup(afc.deleteLater)
        afc.setVisible(True)
        self.settle(10)

    def panel(self):
        return self.win._message_overflow

    def shown_text(self):
        """What the status bar's message area draws (cut short or not)."""
        return QLabel.text(self.win._message_label)

    def layout(self):
        """Where the status bar, its message and the plots are."""
        win = self.win
        return [
            win.statusBar().geometry(),
            win._message_label.geometry(),
            win._main_splitter.geometry(),
            win._spectrum.geometry(),
            win._waterfall.geometry(),
            win._right_tabs.geometry(),
        ]

    def in_window(self, widget):
        return QRect(widget.mapTo(self.win, QPoint(0, 0)), widget.size())

    def assert_shows_in_full(self, text, where):
        """The panel shows ``text`` whole, just above the status bar, inside
        the window, its text starting where the message's text starts."""
        win, panel = self.win, self.panel()
        self.assertTrue(panel.isVisible(), where)
        self.assertEqual(panel.text(), text, where)
        self.assertFalse(panel.isWindow(), where)  # (part of the window)
        self.assertIs(panel.window(), win, where)
        box = panel.geometry()
        bar = win.statusBar().geometry()
        self.assertLess(box.bottom(), bar.top(), where)  # above the status bar
        self.assertGreaterEqual(box.bottom(), bar.top() - 8, where)  # just above
        self.assertGreaterEqual(box.left(), 0, where)
        self.assertLessEqual(box.right(), win.width() - 1, where)
        self.assertGreater(box.top(), win._spectrum.mapTo(win, QPoint(0, 0)).y())
        message = self.in_window(win._message_label)
        label = win._message_label
        self.assertEqual(
            box.left() + panel.contentsRect().left() + panel.indent(),
            message.left() + label.contentsRect().left(),
            where,
        )
        self.assertGreaterEqual(box.width(), message.width() - 3, where)
        # Every word fits in the panel (word-wrapped where its text runs).
        inner = panel.contentsRect()
        inner.setLeft(inner.left() + max(0, panel.indent()))
        needed = QFontMetrics(panel.font()).boundingRect(
            QRect(0, 0, inner.width(), 10_000),
            int(Qt.TextFlag.TextWordWrap),
            text,
        )
        self.assertLessEqual(needed.height(), inner.height(), where)
        self.assertLessEqual(needed.width(), inner.width(), where)

    def lines(self):
        """How many lines the panel's text takes."""
        panel = self.panel()
        inner = panel.contentsRect().height()
        return round(inner / QFontMetrics(panel.font()).lineSpacing())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestALongMessageIsReadableInFull(_StatusCase):
    def test_at_1024_px_with_every_chip(self):
        _themes(self, self.check_1024)

    def check_1024(self, theme):
        win = self.win
        win.resize(1024, 640)
        self.every_chip()
        before = self.layout()
        self.assertEqual(win._message_label.width(), 220, theme)  # (as before)
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.settle(5)
        # The status bar still cuts it (it keeps its layout)...
        self.assertEqual(win._message_label.text(), REFUSAL, theme)
        self.assertTrue(self.shown_text().endswith("…"), self.shown_text())
        # ...and the whole message is on screen meanwhile, on one line
        # (there is room for it from where the message starts).
        self.assert_shows_in_full(REFUSAL, theme)
        self.assertEqual(self.lines(), 1, theme)
        self.assertEqual(self.panel().property("tone"), "warning", theme)
        # Nothing moved: not the status bar, its message, nor the plots.
        self.assertEqual(self.layout(), before, theme)
        self.assertEqual(win.width(), 1024, theme)
        self.assertLessEqual(win.minimumSizeHint().width(), 1024, theme)
        # Said once (the panel is the same message, not another one).
        self.assertEqual(
            [entry.text for entry in win._status_history.entries()], [REFUSAL]
        )
        win._clear_status_message()

    def test_screen_readers_get_the_whole_text(self):
        win = self.win
        win.resize(1024, 640)
        self.every_chip()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertEqual(win._message_label.accessibleDescription(), REFUSAL)
        panel = self.panel()
        self.assertTrue(panel.accessibleName())
        self.assertEqual(panel.accessibleDescription(), REFUSAL)
        win._show_status_message(SHORT)
        self.assertEqual(win._message_label.accessibleDescription(), SHORT)
        win._clear_status_message()
        self.assertEqual(win._message_label.accessibleDescription(), "")

    def test_at_1366_and_1920_px(self):
        _themes(self, self.check_wider)

    def check_wider(self, theme):
        win = self.win
        for width, height in ((1366, 768), (1920, 1080)):
            win.resize(width, height)
            self.every_chip()
            for text in (REFUSAL, GAPS):
                where = (theme, width, text[:20])
                before = self.layout()
                win._show_status_message(text, "info", 8000)
                self.settle(5)
                cut = self.shown_text() != text
                self.assertEqual(self.panel().isVisible(), cut, where)
                if cut:
                    self.assert_shows_in_full(text, where)
                self.assertEqual(self.layout(), before, where)
                self.assertEqual(win.width(), width, where)
        # The longest one is cut even at 1366 px with every chip.
        win.resize(1366, 768)
        self.settle(10)
        win._show_status_message(GAPS, "warning", 8000, log=False)
        self.settle(5)
        self.assert_shows_in_full(GAPS, theme)
        win._clear_status_message()

    def test_a_message_that_fits_shows_as_before(self):
        win = self.win
        win.resize(1024, 640)
        self.every_chip()
        win._show_status_message(SHORT, "success")
        self.settle(5)
        self.assertEqual(self.shown_text(), SHORT)
        self.assertFalse(self.panel().isVisible())
        self.assertEqual(win._message_label.property("tone"), "success")
        # ...and as long as the status bar has room, a long one does too.
        for label in (
            win._recording_label,
            win._recording_info_label,
            win._audio_rec_label,
            win._decoder_chip,
            self._afc,
        ):
            label.setVisible(False)
        win.resize(1920, 1080)
        self.settle(10)
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.settle(5)
        self.assertEqual(self.shown_text(), REFUSAL)
        self.assertFalse(self.panel().isVisible())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheStatusBarHoldsStill(_StatusCase):
    """At 1024 px with every chip, each message and each clear moved the
    status bar's chips and its message one pixel and back: the device's
    name and the recording's size, both cut short, asked for one pixel
    more or less depending on where their text was cut ("26.7 MB · …" or
    "26.7 MB · 1…"), and the status bar shared the room out anew at every
    message."""

    def places(self):
        from PyQt6.QtWidgets import QWidget

        bar = self.win.statusBar()
        return [
            (type(widget).__name__, widget.geometry())
            for widget in bar.findChildren(QWidget)
            if widget.isVisible() and widget.parentWidget() is bar
        ]

    def test_messages_dont_move_it(self):
        _themes(self, self.check_still)

    def check_still(self, theme):
        win = self.win
        win.resize(1024, 640)
        self.every_chip()
        win._clear_status_message()
        self.settle(5)
        first = self.places()
        for text in (SHORT, REFUSAL, GAPS, SHORT):
            win._show_status_message(text, "info", 8000)
            self.settle(5)
            self.assertEqual(self.places(), first, (theme, text))
            win._clear_status_message()
            self.settle(5)
            self.assertEqual(self.places(), first, (theme, "cleared"))

    def test_a_shortened_label_asks_for_its_whole_text(self):
        """(Its size hint is the whole text's, however it is cut.)"""
        _themes(self, self.check_hint)

    def check_hint(self, theme):
        win = self.win
        label = win._recording_info_label
        text = "26.7 MB · 17.7 GB free"
        whole = QLabel(text, win.statusBar())  # (as the status bar styles it)
        self.addCleanup(whole.deleteLater)
        whole.ensurePolished()
        label.setText(text)
        label.setVisible(True)
        for width in range(label.minimumWidth(), whole.sizeHint().width() + 4):
            label.resize(width, label.height())
            self.assertEqual(
                label.sizeHint().width(),
                whole.sizeHint().width(),
                (theme, width, QLabel.text(label)),
            )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestThePanelGoesWithItsMessage(_StatusCase):
    def setUp(self):
        super().setUp()
        self.win.resize(1024, 640)
        self.every_chip()

    def test_it_lasts_as_long_as_the_message(self):
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertTrue(panel.isVisible())
        win._clear_status_message()  # (what the message's timer does)
        self.assertFalse(panel.isVisible())
        # A shorter message in its place: no panel.
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        win._show_status_message(SHORT)
        self.assertFalse(panel.isVisible())
        # Another long one: the panel says it, in its tone (in two lines:
        # it is longer than the window has room for from there).
        win._show_status_message(GAPS, "danger", 8000, log=False)
        self.assert_shows_in_full(GAPS, "gaps")
        self.assertEqual(self.lines(), 2)
        self.assertEqual(panel.property("tone"), "danger")
        win._clear_status_message()

    def test_the_timer_clears_it(self):
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 500, log=False)
        self.assertTrue(panel.isVisible())
        win._message_timer.timeout.emit()
        self.assertFalse(panel.isVisible())
        self.assertEqual(win._message_label.text(), "")

    def click_away(self, panel):
        """A click on the panel, and the time a double-click takes."""
        QTest.mouseClick(panel, Qt.MouseButton.LeftButton)
        QTest.qWait(QApplication.doubleClickInterval() + 50)

    def test_a_click_hides_it(self):
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.click_away(panel)
        self.assertFalse(panel.isVisible())
        # The message stays in the status bar (and in the history).
        self.assertEqual(win._message_label.text(), REFUSAL)
        self.assertEqual(len(win._status_history), 1)
        # Hidden for that message: a resize doesn't bring it back...
        win.resize(1030, 640)
        self.settle(10)
        win.resize(1024, 640)
        self.settle(10)
        self.assertFalse(panel.isVisible())
        # ...nor the same refusal said again meanwhile.
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertFalse(panel.isVisible())
        # The next message shows again.
        win._show_status_message(GAPS, "warning", 8000, log=False)
        self.assertTrue(panel.isVisible())
        win._clear_status_message()
        # And the refusal once more, after it was gone.
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertTrue(panel.isVisible())
        win._clear_status_message()

    def test_a_double_click_stays_on_it(self):
        """The click that hides it doesn't let a double-click's second
        click through to what is beneath: a button there would be pressed
        (at 1024 px it covers the Bookmarks tab's Tune, Edit... and Remove)
        and the waterfall would tune (a press and a release on it tune)."""
        from PyQt6.QtWidgets import QPushButton

        win, panel = self.win, self.panel()
        win._show_status_message(GAPS, "warning", 8000, log=False)
        self.settle(5)
        beneath = QPushButton("Remove", win)
        self.addCleanup(beneath.deleteLater)
        beneath.setGeometry(panel.geometry())
        beneath.show()
        panel.raise_()
        pressed = []
        beneath.clicked.connect(lambda: pressed.append(True))
        spot = QPoint(panel.width() // 2, panel.height() // 2)
        where = panel.mapTo(win, spot)
        QTest.mouseClick(panel, Qt.MouseButton.LeftButton, pos=spot)
        # The second click goes to whatever is at that spot now: if what was
        # pressed first went away meanwhile, Qt gives the widget beneath the
        # double-click event alone (QWidget takes it for a press), and the
        # release (QWidget.mouseDoubleClickEvent).
        second = win.childAt(where)
        _second_click(second, second.mapFrom(win, where))
        self.settle(5)
        self.assertEqual(pressed, [])
        QTest.qWait(QApplication.doubleClickInterval() + 50)
        self.assertFalse(panel.isVisible())
        self.assertEqual(pressed, [])
        # (Then what is beneath is there to click, as before.)
        self.assertIs(win.childAt(where), beneath)
        win._clear_status_message()

    def test_esc_hides_it(self):
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        win._spectrum.setFocus()
        self.settle()
        QTest.keyClick(win._spectrum, Qt.Key.Key_Escape)
        self.assertFalse(panel.isVisible())
        self.assertEqual(win._message_label.text(), REFUSAL)
        # Esc in the Frequency field too (it also cancels the typing there).
        win._show_status_message(GAPS, "warning", 8000, log=False)
        field = win._frequency_field()
        field.setFocus()
        self.settle()
        QTest.keyClick(field, Qt.Key.Key_Escape)
        self.assertFalse(panel.isVisible())
        win._clear_status_message()

    def test_the_history_takes_its_place(self):
        """A click on the message lists the last messages, over the place
        the panel had: the panel goes."""
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        QTest.mouseClick(win._message_label, Qt.MouseButton.LeftButton)
        popup = win._status_popup
        self.addCleanup(popup.close)
        self.assertTrue(popup.isVisible())
        self.assertFalse(panel.isVisible())
        self.assertIn(REFUSAL, popup.texts()[0])

    def test_it_follows_the_window(self):
        """Wide enough, the message fits and the panel goes; narrower again
        (the message still showing), it comes back, in place."""
        win, panel = self.win, self.panel()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertTrue(panel.isVisible())
        win.resize(1920, 1080)
        self.settle(10)
        self.assertEqual(self.shown_text(), REFUSAL)
        self.assertFalse(panel.isVisible())
        win.resize(1024, 640)
        self.settle(10)
        self.assert_shows_in_full(REFUSAL, "back at 1024")
        # Taller: still just above the status bar.
        win.resize(1024, 760)
        self.settle(10)
        self.assert_shows_in_full(REFUSAL, "taller")
        # A chip that goes (the decoder stopped) makes room: it may fit.
        win._decoder_chip.setVisible(False)
        win._audio_rec_label.setVisible(False)
        win._recording_info_label.setVisible(False)
        win._recording_label.setVisible(False)
        self._afc.setVisible(False)
        self.settle(10)
        self.assertEqual(panel.isVisible(), self.shown_text() != REFUSAL)
        win._clear_status_message()

    def test_a_huge_message_stays_in_the_window(self):
        """An internal error's text can be of any length: the panel takes at
        most five lines, ending in "…" (the whole message stays in the
        status history and is what screen readers get)."""
        win, panel = self.win, self.panel()
        huge = (
            "Internal error: ValueError: "
            + "operands could not be broadcast together " * 60
            + ". Details are in Tools > Error History."
        )
        win._show_status_message(huge, "danger", 8000, log=False)
        self.settle(5)
        self.assertTrue(panel.isVisible())
        self.assertLessEqual(self.lines(), 5)
        self.assertEqual(self.lines(), 5)
        self.assertTrue(panel.text().endswith("…"), panel.text()[-20:])
        self.assertTrue(huge.startswith(panel.text()[:-1].rstrip()))
        self.assertGreater(panel.y(), win._spectrum.mapTo(win, QPoint(0, 0)).y())
        bar = win.statusBar().geometry()
        self.assertLess(panel.geometry().bottom(), bar.top())
        self.assertEqual(panel.message(), huge)
        self.assertEqual(panel.accessibleDescription(), huge)
        self.assertEqual(win._status_history.entries()[0].text, huge)
        # A message that fits in five lines is whole again.
        win._show_status_message(GAPS, "warning", 8000, log=False)
        self.assert_shows_in_full(GAPS, "after a huge one")
        win._clear_status_message()

    def test_while_the_window_is_hidden(self):
        """Said while the window is out of sight (minimized): shown once it
        is back, while the message lasts."""
        win, panel = self.win, self.panel()
        win.hide()
        self.settle()
        win._show_status_message(REFUSAL, "warning", 8000, log=False)
        self.assertFalse(panel.isVisible())
        win.show()
        self.settle(10)
        self.assert_shows_in_full(REFUSAL, "shown again")
        win._clear_status_message()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestThePlaybackStripStaysInReach(_StatusCase):
    """While a recording plays, its strip (Pause, the position, Loop, the
    speed, Signals... and Close) is right above the status bar, and the
    panel went over it: for the 8 s of what Open Recording says ("Playing
    ... · 2m Calling: listening in FM ...") a click on the speed or on
    Close only hid the panel. The panel now sits just above the strip."""

    def setUp(self):
        super().setUp()
        import numpy as np

        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        folder = tempfile.mkdtemp(prefix="sdr-status-overflow-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.path = os.path.join(folder, "a_take_from_the_two_metre_band.sigmf-data")
        rate = 2.4e6
        t = np.arange(int(rate * 0.5)) / rate
        samples = (0.5 * np.exp(2j * np.pi * 50e3 * t)).astype(np.complex64)
        save_iq_file(
            self.path, samples, rate, 146.52e6, SampleFormat.FLOAT32, FileFormat.SIGMF
        )

    def assert_strip_in_reach(self, where):
        """The panel is just above the strip, and each of the strip's
        controls is what a click on its middle reaches."""
        from PyQt6.QtWidgets import QAbstractButton, QAbstractSlider, QComboBox

        win, panel = self.win, self.panel()
        strip = self.in_window(win._playback_bar)
        box = panel.geometry()
        self.assertFalse(box.intersects(strip), where)
        self.assertLess(box.bottom(), strip.top(), where)
        self.assertGreaterEqual(box.bottom(), strip.top() - 8, where)
        controls = [
            widget
            for widget in win._playback_bar.findChildren(QAbstractButton)
            + win._playback_bar.findChildren(QAbstractSlider)
            + win._playback_bar.findChildren(QComboBox)
            if widget.isVisible()
        ]
        self.assertGreaterEqual(len(controls), 4, where)  # (Pause ... Close)
        for control in controls:
            hit = win.childAt(control.mapTo(win, control.rect().center()))
            self.assertTrue(
                hit is control or control.isAncestorOf(hit),
                (where, control.accessibleName() or control.objectName(), hit),
            )

    def test_the_strip_stays_in_reach(self):
        _themes(self, self.check_strip)

    def check_strip(self, theme):
        win, panel = self.win, self.panel()
        for width, height in ((1024, 640), (1366, 768)):
            where = (theme, width)
            win.resize(width, height)
            self.settle(10)
            self.assertTrue(win._open_recording(self.path), where)
            self.settle(10)
            self.assertTrue(win._playback_bar.isVisible(), where)
            # What Open Recording says (cut at 1024 px), then a long warning.
            opened = win._message_label.text()
            self.assertTrue(opened.startswith("Playing a_take"), opened)
            if width == 1024:
                self.assertTrue(panel.isVisible(), where)
                self.assertEqual(panel.text(), opened, where)
                self.assert_strip_in_reach((where, "opened"))
            win._show_status_message(GAPS, "warning", 8000, log=False)
            self.settle(5)
            self.assertTrue(panel.isVisible(), where)
            self.assertEqual(panel.text(), GAPS, where)
            self.assert_strip_in_reach((where, "gaps"))
            # Without the strip, just above the status bar again (should
            # the status bar, with the device's name back, cut it).
            win._playback.close()
            self.settle(10)
            self.assertFalse(win._playback_bar.isVisible(), where)
            win._show_status_message(GAPS, "warning", 8000, log=False)
            self.settle(5)
            cut = self.shown_text() != GAPS
            self.assertEqual(panel.isVisible(), cut, where)
            self.assertTrue(cut or width > 1024, where)
            if cut:
                self.assert_shows_in_full(GAPS, where)
            win._clear_status_message()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestThePanelsColors(_StatusCase):
    """The theme's colors: the tone's tint and border (the message's own
    color), in both themes, and again after a theme switch."""

    def test_in_both_themes(self):
        _themes(self, self.check_colors)

    def check_colors(self, theme):
        from sdr_module.gui.themes import get_palette

        win, panel = self.win, self.panel()
        win.resize(1024, 640)
        self.every_chip()
        palette = get_palette(theme)
        for tone, fill, edge in (
            ("warning", palette.warning_bg, palette.warning),
            ("danger", palette.danger_bg, palette.danger),
            ("info", palette.info_bg, palette.info),
            (None, palette.mantle, palette.surface2),
        ):
            win._show_status_message(REFUSAL, tone, 8000, log=False)
            self.settle(5)
            self.assertTrue(panel.isVisible(), (theme, tone))
            image = panel.grab().toImage()
            middle = image.height() // 2
            self.assertEqual(image.pixelColor(0, middle).name(), edge, (theme, tone))
            self.assertEqual(image.pixelColor(3, middle).name(), fill, (theme, tone))
            win._clear_status_message()


if __name__ == "__main__":
    unittest.main()
