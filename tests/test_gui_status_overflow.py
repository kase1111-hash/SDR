"""K-W1: the status bar holds still, and a long message is readable in full.

At 1024 px with every status chip shown (REC, the recording's size, AUDIO,
the decoder chip and AFC) the message area keeps 220 px. Each message moved
the status bar's chips and its message a pixel and back.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QLabel

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


if __name__ == "__main__":
    unittest.main()
