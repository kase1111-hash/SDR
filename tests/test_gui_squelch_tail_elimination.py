#!/usr/bin/env python3
"""
Radio > Audio > Squelch Tail Elimination in the window.

The CTCSS reverse burst a radio sends as its user stops talking closes the
tone squelch before the squelch tail (M-Q2); a voice pitched near the tone
can sound like one now and then and mute the station for about 0.8 s. The
switch (radios' STE) turns the burst on or off: a checkable item right
after Reverse Tone Squelch, on by default, saved as audio/tail_elimination
(a damaged value is logged and read as on), applied at once to the tone
squelch, so also to Scan Bookmarks, which judges a channel by it. The DCS
turn-off code isn't covered (see tests/test_squelch_tail_elimination.py).
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_ctcss_burst import ending_transmission  # noqa: E402
from tests.test_gui_dcs import _menu  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    pretend_audio_output,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

ITEM = "Squelch Tail Elimination"
FRAME_S = FRAME / RATE
BURST_AT = 2.0  # (ending_transmission: the burst starts here...)
DROP_AT = 2.15  # (...and the carrier drops here)


def _audio_menu_texts(win):
    return [
        a.text()
        for a in _menu(win, "Radio", "Audio").menu().actions()
        if not a.isSeparator()
    ]


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheMenuItem(_WindowTestCase):
    def test_after_reverse_tone_squelch_on_by_default(self):
        win = self.win
        action = _menu(win, "Radio", "Audio", ITEM)
        texts = [t.replace("&", "") for t in _audio_menu_texts(win)]
        self.assertEqual(texts.index(ITEM), texts.index("Reverse Tone Squelch") + 1)
        self.assertTrue(action.isCheckable())
        self.assertTrue(action.isChecked())
        self.assertTrue(action.isEnabled())
        self.assertTrue(win._tone_squelch.tail_elimination)
        # What it does and why one might turn it off.
        tip = action.statusTip()
        self.assertEqual(tip, action.toolTip())
        for words in ("reverse burst", "squelch tail", "Turn it off", "mid-"):
            self.assertIn(words, tip)
        # Its letter is its own in the menu.
        keys = [t[t.index("&") + 1].lower() for t in _audio_menu_texts(win) if "&" in t]
        self.assertEqual(len(keys), len(set(keys)), keys)

    def test_turned_off_and_on_applied_and_saved(self):
        win = self.win
        action = _menu(win, "Radio", "Audio", ITEM)
        with self.assertNoLogs(level="WARNING"):  # (what was asked: no problem)
            action.trigger()
        self.assertFalse(action.isChecked())
        self.assertFalse(win._tone_squelch.tail_elimination)
        self.assertIs(self.store.values["audio/tail_elimination"], False)
        self.assertIn("Squelch tail elimination off", win._message_label.text())
        action.trigger()
        self.assertTrue(win._tone_squelch.tail_elimination)
        self.assertIs(self.store.values["audio/tail_elimination"], True)
        self.assertIn("Squelch tail elimination on", win._message_label.text())

    def test_set_from_code_the_item_follows(self):
        win = self.win
        action = _menu(win, "Radio", "Audio", ITEM)
        win._set_tail_elimination(False)
        self.assertFalse(action.isChecked())
        self.assertFalse(win._tone_squelch.tail_elimination)
        win._clear_status_message()
        win._set_tail_elimination(False)  # (no change: nothing said)
        self.assertEqual(win._message_label.text(), "")

    def test_a_tone_squelch_chosen_later_keeps_it(self):
        """The switch is the receiver's, not the tone's: choosing a tone,
        a DCS code or Reverse leaves it as set."""
        win, panel = self.win, self.win._control_panel
        win._set_tail_elimination(False)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_tone_squelch_hz(100.0)
        panel.set_tone_reverse(True)
        panel.set_dcs_squelch(23)
        self.assertFalse(win._tone_squelch.tail_elimination)


class TestTheSetting(_IniSettings):
    def window(self, **values):
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", **values)
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        return win

    def test_restored(self):
        win = self.window(audio__tail_elimination="false")
        self.assertFalse(win._tone_squelch.tail_elimination)
        self.assertFalse(_menu(win, "Radio", "Audio", ITEM).isChecked())

    def test_on_when_nothing_is_saved(self):
        win = self.window()
        self.assertTrue(win._tone_squelch.tail_elimination)
        self.assertTrue(_menu(win, "Radio", "Audio", ITEM).isChecked())

    def test_a_damaged_value_is_on_and_logged(self):
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            win = self.window(audio__tail_elimination="maybe")
        self.assertTrue(win._tone_squelch.tail_elimination)
        self.assertTrue(_menu(win, "Radio", "Audio", ITEM).isChecked())
        self.assertTrue(any("audio/tail_elimination" in m for m in logs.output))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestInTheReceiver(_WindowTestCase):
    """A 2 m channel (FM, 5 kHz) with a 100.0 Hz tone squelch: a
    transmission ending with its reverse burst, through the real chain."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(146.52e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_squelch_db(-90)
        self.panel.set_tone_squelch_hz(100.0)

    def listen(self, scan=False):
        """Per frame: (end time s, peak played, lamp state, whether a scan
        would count the channel as busy)."""
        win = self.win
        iq, _dropped = ending_transmission()
        win._reset_demodulator()
        host = win._scanner_host
        if scan:
            # As Scan Bookmarks tunes a bookmark with a tone squelch.
            listening = ("FM", "5 kHz", "12.5 kHz")
            self.assertTrue(host.scan_tune_channel(146.52e6, listening, 100.0))
            host.scan_settled()
        frames = Frames(self, iq)
        out = []
        while frames.pos + FRAME <= len(iq):
            frames.run(frames=1)
            audio = frames.audio[-1]
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            out.append(
                (
                    frames.pos / RATE,
                    peak,
                    win._squelch_state(True),
                    host.scan_signal_present(),
                )
            )
        return out

    def test_on_it_shuts_at_the_burst(self):
        frames = self.listen()
        shut = next(t for t, _p, s, _b in frames if t > BURST_AT and s == "tone")
        self.assertLessEqual(shut - BURST_AT, 0.10 + FRAME_S)

    def test_off_the_tail_is_heard_as_before(self):
        """Off, the station plays until its carrier drops, and its squelch
        tail is heard until the noise squelch closes (as before M-Q2)."""
        self.win._set_tail_elimination(False)
        frames = self.listen()
        during = [p for t, p, _s, _b in frames if BURST_AT < t <= DROP_AT - 0.03]
        self.assertTrue(during)
        self.assertGreater(min(during), 1e-3)  # (the burst plays)
        opened = next(t for t, _p, s, _b in frames if s == "open")
        self.assertLess(opened, 1.0)
        states = [s for t, _p, s, _b in frames if opened <= t <= DROP_AT + FRAME_S]
        self.assertNotIn("tone", states)
        tail = [t for t, p, _s, _b in frames if t > DROP_AT and p > 1e-3]
        self.assertTrue(tail)  # (the noise as the carrier drops)
        self.assertLess(max(tail) - DROP_AT, 0.45)

    def test_scan_bookmarks_follows_it(self):
        """Scan Bookmarks judges a channel with a tone squelch by the tone
        squelch: on, the channel is free from the burst; off, only once
        the carrier has dropped."""
        frames = self.listen(scan=True)
        busy = [t for t, _p, _s, b in frames if b]
        self.assertTrue(busy)
        free = next(t for t, _p, _s, b in frames if t > BURST_AT and not b)
        self.assertLessEqual(free - BURST_AT, 0.10 + FRAME_S)
        self.win._set_tail_elimination(False)
        frames = self.listen(scan=True)
        free = next(t for t, _p, _s, b in frames if t > BURST_AT and not b)
        self.assertGreater(free, DROP_AT)


if __name__ == "__main__":
    unittest.main()
