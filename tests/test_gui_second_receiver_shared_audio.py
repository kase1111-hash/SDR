"""One audio setting for both receivers (post-merge, rx2 with the audio and
squelch lanes).

Radio > Audio's De-emphasis (75 or 50 us) and Stereo Blend (M-A1, M-A2)
and Squelch Tail Elimination (the CTCSS reverse burst, M-Q2 and its
switch) were the main window's only: the Second Receiver window ran its
own ``AudioOptions`` defaults (75 us, Normal) and a ``ToneSquelch`` that
always closed at a reverse burst, whatever the menu said. Now the Second
Receiver follows them when it opens and whenever they change, and says
so where the user looks: the menu items' tips name both receivers, and
the Second Receiver's own Stereo switch, tone squelch and STEREO lamp say
where those settings are.

Signals: an unemphasized broadcast FM station sending 1 and 10 kHz
(``tests/test_gui_deemphasis.py``), the demo's weak 88.5 MHz stereo
station, and a CTCSS transmission ending with its reverse burst
(``tests/test_gui_ctcss_burst.py``).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_ctcss_burst import ending_transmission  # noqa: E402
from tests.test_gui_deemphasis import (  # noqa: E402
    _expected_treble_db,
    _station,
    _treble_db,
)
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402
from tests.test_gui_second_receiver_stereo import _StereoWindowCase  # noqa: E402

AUDIO = 48000.0
BURST_AT = 2.0  # (ending_transmission: the burst starts here...)
DROP_AT = 2.15  # (...and the carrier drops here)


def _menu(win, *path):
    """The menu (or action) at ``path`` of menu titles, '&' ignored."""
    actions = win.menuBar().actions()
    found = None
    for title in path:
        found = next(
            a for a in actions if a.text().replace("&", "").split("\t")[0] == title
        )
        actions = found.menu().actions() if found.menu() is not None else []
    return found


class _Listening(_SecondCase):
    """The second receiver running, Listen: This receiver, its speaker a
    fake one that keeps what it is given."""

    def setUp(self):
        super().setUp()
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_listen("second")

    def frames(self, iq):
        """Feed ``iq`` a display frame at a time: per frame ``(end time s,
        peak played, LEVEL lamp)``."""
        rx = self.rx
        block = int(rx._rate() / 30)
        out = []
        for start in range(0, len(iq) - block + 1, block):
            before = len(self.speaker.blocks)
            rx._process(iq[start : start + block])
            played = self.speaker.blocks[before:]
            peak = max((float(np.max(np.abs(b))) for b in played if len(b)), default=0)
            out.append(((start + block) / rx._rate(), peak, rx._level_meter.state()))
        return out

    def heard(self):
        return self.speaker.samples()


# ---------------------------------------------------------------------------
# It follows Radio > Audio
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestItFollowsRadioAudio(_SecondCase):
    def test_it_takes_them_when_it_opens(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        win = self.win
        win._set_deemphasis("50")
        win._set_stereo_blend("more")
        win._set_tail_elimination(False)
        again = self.extra(SecondReceiverWindow(win))
        self.assertEqual(again._dsp.options.deemphasis_s, 50e-6)
        self.assertEqual(again._dsp.options.stereo_blend, "more")
        self.assertFalse(again._dsp.tones.tail_elimination)

    def test_it_follows_a_change_while_open(self):
        win, dsp = self.win, self.rx._dsp
        self.assertEqual(dsp.options.deemphasis_s, 75e-6)
        self.assertEqual(dsp.options.stereo_blend, "normal")
        self.assertTrue(dsp.tones.tail_elimination)
        win._set_deemphasis("50")
        self.assertEqual(dsp.options.deemphasis_s, 50e-6)
        win._set_stereo_blend("less")
        self.assertEqual(dsp.options.stereo_blend, "less")
        win._set_tail_elimination(False)
        self.assertFalse(dsp.tones.tail_elimination)
        win._set_tail_elimination(True)
        self.assertTrue(dsp.tones.tail_elimination)
        win._set_deemphasis("75")
        self.assertEqual(dsp.options.deemphasis_s, 75e-6)

    def test_from_the_menus_too(self):
        dsp = self.rx._dsp
        _menu(
            self.win,
            "Radio",
            "Audio",
            "De-emphasis",
            "50 µs (Europe, Australia, most of the world)",
        ).trigger()
        _menu(
            self.win, "Radio", "Audio", "Stereo Blend", "More (quieter weak stations)"
        ).trigger()
        _menu(self.win, "Radio", "Audio", "Squelch Tail Elimination").trigger()
        self.assertEqual(dsp.options.deemphasis_s, 50e-6)
        self.assertEqual(dsp.options.stereo_blend, "more")
        self.assertFalse(dsp.tones.tail_elimination)

    def test_its_own_settings_stay_its_own(self):
        # The notch, the low cut and the tone squelch are this window's;
        # following the main window's audio settings doesn't touch them.
        rx = self.rx
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._notch_combo.setCurrentIndex(rx._notch_combo.findData("auto"))
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        self.win._set_deemphasis("50")
        self.win._set_tail_elimination(False)
        self.assertEqual(rx._dsp.options.notch, "auto")
        self.assertEqual(rx._dsp.tone_squelch, (100.0, None))
        self.assertEqual(self.win._audio_options.notch, "off")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestWhereTheUserLooks(_SecondCase):
    def test_the_menu_items_say_both_receivers(self):
        for path in (
            ("Radio", "Audio", "De-emphasis"),
            ("Radio", "Audio", "Stereo Blend"),
            ("Radio", "Audio", "Squelch Tail Elimination"),
        ):
            action = _menu(self.win, *path)
            for tip in (action.statusTip(), action.toolTip()):
                self.assertIn("Second Receiver", tip, path)
        # Its Stereo switch is its own: the main window's says so.
        stereo = _menu(self.win, "Radio", "Audio", "Stereo")
        self.assertIn("Second Receiver", stereo.statusTip())
        self.assertIn("own", stereo.statusTip())

    def test_its_controls_say_where_the_shared_ones_are(self):
        rx = self.rx
        tip = rx._stereo_check.toolTip()
        for words in ("De-emphasis", "Stereo Blend", "main window", "Radio > Audio"):
            self.assertIn(words, tip)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        tip = rx._tone_combo.toolTip()
        self.assertIn("Squelch Tail Elimination", tip)
        self.assertIn("main window", tip)


# ---------------------------------------------------------------------------
# Heard in the Second Receiver
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestA50UsStation(_Listening):
    """An unemphasized broadcast station, so what is heard shows the
    de-emphasis itself: 10 kHz against 1 kHz is -12.8 dB through 75 us and
    -9.9 dB through 50 us."""

    def setUp(self):
        super().setUp()
        rx = self.rx
        rx.tune(100.1e6, recenter=True)
        rx.set_mode("FM", "75 kHz", "200 kHz")
        rx._squelch_check.setChecked(False)
        rx._sync_audio_output()

    def test_heard_through_50_us_when_the_main_window_says_so(self):
        self.win._set_deemphasis("50")
        self.frames(_station(1.5))
        tail = self.heard()[int(0.5 * AUDIO) :]
        self.assertAlmostEqual(_treble_db(tail), _expected_treble_db(50e-6), delta=0.2)
        self.assertEqual(self.rx._dsp.chain().deemphasis_s, 50e-6)

    def test_a_change_while_it_plays(self):
        iq = _station(2.5)
        half = len(iq) // 2 // 80_000 * 80_000
        self.frames(iq[:half])
        before = self.heard()[int(0.4 * AUDIO) :]
        self.assertAlmostEqual(
            _treble_db(before), _expected_treble_db(75e-6), delta=0.2
        )
        self.speaker.blocks.clear()
        self.win._set_deemphasis("50")
        self.frames(iq[half:])
        after = self.heard()[int(0.1 * AUDIO) :]
        self.assertAlmostEqual(_treble_db(after), _expected_treble_db(50e-6), delta=0.2)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheBlendHeard(_StereoWindowCase):
    """The demo's 88.5 MHz stereo station is weak: Normal blends it partly
    toward mono, Less plays it in full stereo, More nearly in mono (as in
    the main window, tests/test_gui_stereo_blend.py). The lamp says which,
    and where the setting is."""

    def blended(self, key):
        self.win._set_stereo_blend(key)
        self.listen(88.5e6)
        self.feed(14)
        decoder = self.rx._dsp.chain().stereo_decoder
        return decoder, self.rx._stereo_badge

    def test_more_plays_it_nearly_in_mono(self):
        decoder, lamp = self.blended("more")
        self.assertEqual(decoder.blending, "more")
        self.assertLess(decoder.blend, 0.5)
        self.assertEqual(lamp.text(), "MONO")
        tip = lamp.toolTip()
        self.assertIn("Stereo Blend: More", tip)
        self.assertIn("main window", tip)
        self.assertIn("most of the way toward mono", tip)
        self.assertNotIn("plays in mono", tip)  # (some stereo still plays)

    def test_less_plays_it_in_full_stereo(self):
        decoder, lamp = self.blended("less")
        self.assertEqual(decoder.blending, "less")
        self.assertGreaterEqual(decoder.blend, 0.95)
        self.assertEqual(lamp.text(), "STEREO")
        self.assertIn("fully apart", lamp.toolTip())

    def test_normal_blends_it_partly_and_says_so(self):
        decoder, lamp = self.blended("normal")
        self.assertEqual(decoder.blending, "normal")
        self.assertGreater(decoder.blend, 0.5)
        self.assertLess(decoder.blend, 0.95)
        self.assertEqual(lamp.text(), "STEREO")
        self.assertIn("blended partly", lamp.toolTip())
        self.assertIn("Stereo Blend: Normal", lamp.toolTip())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheReverseBurstHeard(_Listening):
    """A 2 m channel (FM, 5 kHz) with a 100.0 Hz tone squelch in the Second
    Receiver: a transmission ending with its reverse burst."""

    def setUp(self):
        super().setUp()
        rx = self.rx
        rx.tune(146.52e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._squelch_auto.setChecked(False)
        rx._squelch_slider.setValue(-90)
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        rx._sync_audio_output()

    def test_on_it_shuts_at_the_burst(self):
        frames = self.frames(ending_transmission()[0])
        shut = next(t for t, _p, s in frames if t > BURST_AT and s == "tone")
        self.assertLessEqual(shut - BURST_AT, 0.10 + 1 / 30)

    def test_off_in_the_main_window_the_tail_is_heard(self):
        """Squelch Tail Elimination off: the station plays until its carrier
        drops, and its squelch tail is heard, here as in the main window."""
        self.win._set_tail_elimination(False)
        frames = self.frames(ending_transmission()[0])
        during = [p for t, p, _s in frames if BURST_AT < t <= DROP_AT - 0.03]
        self.assertTrue(during)
        self.assertGreater(min(during), 1e-3)  # (the burst plays)
        opened = next(t for t, _p, s in frames if s == "open")
        states = [s for t, _p, s in frames if opened <= t <= DROP_AT + 1 / 30]
        self.assertNotIn("tone", states)
        tail = [t for t, p, _s in frames if t > DROP_AT and p > 1e-3]
        self.assertTrue(tail)  # (the noise as the carrier drops)
        # Back on: the next transmission shuts at its burst again.
        self.win._set_tail_elimination(True)
        frames = self.frames(ending_transmission()[0])
        shut = next(t for t, _p, s in frames if t > BURST_AT and s == "tone")
        self.assertLessEqual(shut - BURST_AT, 0.10 + 1 / 30)


if __name__ == "__main__":
    unittest.main()
