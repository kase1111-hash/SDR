#!/usr/bin/env python3
"""
The audio notch in the window (K-A1, the audit's FG-G14): Audio group >
Notch (Off, Auto, Manual and its frequency), Radio > Audio > Notch, where
it applies (AM, SSB, CW and narrow FM; not broadcast FM or I/Q), what the
receiver does with it, and its settings (saved, and safe when damaged).
"""

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    carrier,
    noise,
    pretend_audio_output,
)
from tests.voice_synth import pitched_voice, tone_level  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

AUDIO_RATE = 48000.0


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


def _analytic(x):
    """The analytic signal of real audio (an upper sideband of it)."""
    n = len(x)
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.fft.ifft(spectrum * h)


def _upsample(audio):
    """48 kHz audio held for the 2.4 MS/s I/Q (the channel filter takes
    the copies away)."""
    return np.repeat(audio, int(RATE / AUDIO_RATE))


def _whole_frames(iq):
    n = len(iq) - len(iq) % FRAME
    return iq[:n].astype(np.complex64)


def usb_with_a_whistle(seconds, whistle_hz=1300.0, seed=6):
    """USB voice at -50 dBFS and a carrier ``whistle_hz`` above the tuned
    frequency (heard as a ``whistle_hz`` tone) at -56 dBFS."""
    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    iq = 10 ** (-50 / 20) * _upsample(_analytic(voice))
    iq = iq + carrier(len(iq), whistle_hz, -56.0)
    return _whole_frames(iq + noise(len(iq), -100.0, seed=seed))


def am_with_a_whistle(seconds, whistle_hz=1000.0, seed=4):
    """AM voice (60 %) at -40 dBFS and a carrier ``whistle_hz`` away at
    -54 dBFS: a whistle at a fifth of the carrier."""
    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    envelope = 10 ** (-40 / 20) * (1.0 + _upsample(voice))
    iq = envelope + carrier(len(envelope), whistle_hz, -54.0)
    return _whole_frames(iq + noise(len(iq), -95.0, seed=seed))


def _window_of(audio, start_s, end_s):
    return audio[int(start_s * AUDIO_RATE) : int(end_s * AUDIO_RATE)]


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNotchControls(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel

    def test_the_audio_group_has_a_notch(self):
        from sdr_module.gui.audio_chain import DEFAULT_NOTCH_HZ, NOTCH_RANGE_HZ

        p = self.panel
        combo, spin = p._notch_combo, p._notch_spin
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())], ["Off", "Auto", "Manual"]
        )
        for widget in (combo, spin):
            self.assertTrue(widget.toolTip(), widget)
        self.assertEqual(combo.accessibleName(), "Notch")
        self.assertEqual(spin.accessibleName(), "Notch frequency")
        self.assertEqual((spin.minimum(), spin.maximum()), NOTCH_RANGE_HZ)
        self.assertEqual(spin.suffix(), " Hz")
        self.assertEqual(p.notch_mode(), "off")
        self.assertEqual(p.notch_hz(), DEFAULT_NOTCH_HZ)
        # The frequency is set by hand in Manual only.
        self.win._set_demod_mode("USB")
        self.assertFalse(spin.isEnabled())
        p.set_notch_mode("manual")
        self.assertTrue(spin.isEnabled())
        p.set_notch_mode("auto")
        self.assertFalse(p._notch_spin.isVisible() and spin.isEnabled())

    def test_it_takes_no_width_from_the_right_column(self):
        """At 1024 px the right column is 400 px wide: the frequency box
        beside the Notch choice made the control panel 404 px. The row
        with the whistle (Manual's frequency, or the one Auto removes) is
        under the choice, about as wide as the Audio group's lists, and
        shows in Auto and Manual only."""
        from sdr_module.gui import themes

        p = self.panel
        form = p._audio_form
        self.assertFalse(form.isRowVisible(p._notch_row))
        for mode in ("auto", "manual"):
            p.set_notch_mode(mode)
            self.assertTrue(form.isRowVisible(p._notch_row), mode)
        p.set_notch_mode("off")
        self.assertFalse(form.isRowVisible(p._notch_row))
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in (None, *themes.THEMES):
                if theme is not None:
                    themes.apply_theme(qapp, theme)
                for mode in ("auto", "manual"):
                    p.set_notch_mode(mode)
                    self.settle(3)
                    widest = p._nr_combo.minimumSizeHint().width()
                    for field in (p._notch_combo, p._notch_row):
                        width = field.minimumSizeHint().width()
                        self.assertLessEqual(width, widest + 6, (theme, mode))
                    self.assertLessEqual(p.minimumSizeHint().width(), 400, theme)
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)

    def test_only_where_it_applies(self):
        p, win = self.panel, self.win
        for mode in ("AM", "USB", "LSB", "CW"):
            win._set_demod_mode(mode)
            self.assertTrue(p._notch_combo.isEnabled(), mode)
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(p._notch_combo.isEnabled())
        win._set_demod_mode("FM", "75 kHz", "200 kHz")  # broadcast keeps its audio
        self.assertFalse(p._notch_combo.isEnabled())
        self.assertIn("broadcast", p._notch_combo.toolTip())
        p._fm_dev_combo.setCurrentText("12.5 kHz")  # (the deviation alone)
        self.assertTrue(p._notch_combo.isEnabled())
        win._set_demod_mode("None (I/Q)")
        self.assertFalse(p._notch_combo.isEnabled())
        self.assertFalse(p._notch_spin.isEnabled())

    def test_radio_audio_notch_follows_the_control(self):
        win, p = self.win, self.panel
        win._set_demod_mode("USB")
        _menu(win, "Radio", "Audio", "Notch", "Auto").trigger()
        self.assertEqual(p.notch_mode(), "auto")
        p.set_notch_mode("manual")
        self.assertTrue(_menu(win, "Radio", "Audio", "Notch", "Manual").isChecked())
        self.assertFalse(_menu(win, "Radio", "Audio", "Notch", "Auto").isChecked())
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertFalse(_menu(win, "Radio", "Audio", "Notch").isEnabled())
        win._set_demod_mode("AM")
        self.assertTrue(_menu(win, "Radio", "Audio", "Notch").isEnabled())

    def test_notch_frequency_goes_to_its_box_in_manual(self):
        win, p = self.win, self.panel
        win._set_demod_mode("USB")
        win.activateWindow()
        _menu(win, "Radio", "Audio", "Notch", "Notch Frequency...").trigger()
        self.settle(5)
        self.assertEqual(p.notch_mode(), "manual")
        self.assertIs(QApplication.focusWidget(), p._notch_spin)

    def test_choices_reach_the_receiver_and_are_saved(self):
        p, win = self.panel, self.win
        win._set_demod_mode("USB")
        p.set_notch_mode("manual")
        p.set_notch_hz(1234)
        options = win._audio_options
        self.assertEqual((options.notch, options.notch_hz), ("manual", 1234.0))
        values = self.store.values
        self.assertEqual(values["audio/notch"], "manual")
        self.assertEqual(values["audio/notch_hz"], 1234)
        p.set_notch_mode("auto")
        self.assertEqual(options.notch, "auto")
        win._persist_state()
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel.notch_mode(), "auto")
        self.assertEqual(other._control_panel.notch_hz(), 1234)
        self.assertEqual(other._audio_options.notch, "auto")
        self.assertEqual(other._audio_options.notch_hz, 1234.0)

    def test_out_of_range_frequencies_go_to_the_nearest_end(self):
        from sdr_module.gui.audio_chain import NOTCH_RANGE_HZ

        p = self.panel
        p.set_notch_hz(10)
        self.assertEqual(p.notch_hz(), NOTCH_RANGE_HZ[0])
        p.set_notch_hz(1e9)
        self.assertEqual(p.notch_hz(), NOTCH_RANGE_HZ[1])
        p.set_notch_hz(1500)
        for value in (float("nan"), float("inf"), "x", None):
            p.set_notch_hz(value)  # (not a number: left as it was)
            self.assertEqual(p.notch_hz(), 1500, value)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNotchInTheReceiver(_WindowTestCase):
    """What is played (48 kHz audio captured from the speaker's write)."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.panel.set_squelch_enabled(False)
        self.panel.set_audio_agc_speed("off")  # (a fixed gain: compare levels)

    def listen(self, iq):
        self.win._reset_demodulator()
        return Frames(self, iq).run().played()

    def test_turned_off_while_muted_the_first_block_plays(self):
        """Manual notch, the speaker off for a moment (Mute; a squelch that
        closes does the same), the notch turned Off meanwhile: the first
        block once the speaker is on again is played. (The notch was to
        engage at once and tripped an assertion: the window dropped that
        block, and did the same tuning to another station in Auto.)"""
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.panel.set_notch_mode("manual")
        self.panel.set_notch_hz(1300)
        self.win._reset_demodulator()
        frames = Frames(self, usb_with_a_whistle(1.0))
        frames.run(5)
        self.win._set_audio_enabled(False)
        frames.run(2)  # (the block that fades out, then the demodulator reset)
        self.panel.set_notch_mode("off")
        self.win._set_audio_enabled(True)
        frames.run(1)
        self.assertGreater(len(frames.audio[-1]), 0)

    def test_auto_removes_a_whistle_from_ssb_and_shows_it(self):
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.panel.set_notch_mode("auto")
        audio = self.listen(usb_with_a_whistle(4.0))
        before = tone_level(_window_of(audio, 0.1, 0.4), 1300.0)
        after = tone_level(_window_of(audio, 2.0, 4.0), 1300.0)
        self.assertGreater(before, 0.05)
        self.assertLess(20 * math.log10(after / before), -30.0)
        self.assertEqual(self.panel._notch_found.text(), "1300 Hz")
        notched = self.win._receiver_chain().notch_frequency_hz
        self.assertAlmostEqual(notched, 1300.0, delta=0.5)

    def test_another_station_is_searched_anew(self):
        """Tuning elsewhere forgets the whistle Auto found (it was the last
        station's: it used to stay notched, and shown, for another second);
        Track Signal following the same signal keeps it."""
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.panel.set_notch_mode("auto")
        self.listen(usb_with_a_whistle(2.0))
        chain = self.win._receiver_chain()
        self.assertAlmostEqual(chain.notch_frequency_hz, 1300.0, delta=0.5)
        self.win._on_frequency_changed(14.2e6, same_signal=True)
        self.assertAlmostEqual(chain.notch_frequency_hz, 1300.0, delta=0.5)
        self.win.set_frequency(14.205e6)
        self.assertIs(self.win._receiver_chain(), chain)  # (only retuned)
        self.assertIsNone(chain.notch_frequency_hz)

    def test_by_hand_it_removes_an_am_heterodyne(self):
        self.win.set_frequency(6.0e6)  # (shortwave: the LO is on it)
        self.win._set_demod_mode("AM")
        self.panel.set_notch_mode("manual")
        self.panel.set_notch_hz(1000)
        audio = self.listen(am_with_a_whistle(2.0))
        heard = tone_level(_window_of(audio, 0.5, 2.0), 1000.0)
        self.panel.set_notch_mode("off")
        plain = tone_level(
            _window_of(self.listen(am_with_a_whistle(2.0)), 0.5, 2.0), 1000.0
        )
        self.assertGreater(plain, 0.05)
        self.assertLess(20 * math.log10(heard / plain), -30.0)

    def test_auto_leaves_the_cw_note_alone(self):
        """In CW the note at the CW pitch is the signal itself: a steady
        carrier (a long dash, a beacon) is never notched."""
        self.win.set_frequency(7.03e6)
        self.win._set_demod_mode("CW")
        self.panel.set_notch_mode("auto")
        pitch = self.panel.cw_pitch()
        iq = carrier(int(3.0 * RATE) // FRAME * FRAME, 0.0, -60.0)
        iq = iq + noise(len(iq), -100.0)
        audio = self.listen(iq)
        late = tone_level(_window_of(audio, 2.0, 3.0), float(pitch))
        early = tone_level(_window_of(audio, 0.3, 0.8), float(pitch))
        self.assertAlmostEqual(20 * math.log10(late / early), 0.0, delta=1.0)
        self.assertIsNone(self.win._receiver_chain().notch_frequency_hz)

    def test_broadcast_fm_keeps_its_audio(self):
        from tests.test_gui_squelch import fm

        self.panel.set_notch_mode("manual")
        self.panel.set_notch_hz(1000)
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        n = int(2.0 * RATE) // FRAME * FRAME
        program = 0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / RATE)
        audio = self.listen(fm(program, 75e3, -30.0) + noise(n, -90.0))
        # (75 us de-emphasis: 1 kHz comes out about 0.3 dB down.)
        self.assertGreater(tone_level(_window_of(audio, 0.5, 2.0), 1000.0), 0.4)
        self.assertIsNone(self.win._receiver_chain().notch_frequency_hz)


class TestDamagedNotchSettings(_IniSettings):
    """Saved values that aren't a notch mode or a frequency in range."""

    def window(self, **values):
        from unittest import mock

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
        win._confirm_discard_recording = lambda *_a: True

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        return win

    def test_an_unknown_mode_is_off_and_logged(self):
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            win = self.window(audio__notch="loud")
        self.assertEqual(win._control_panel.notch_mode(), "off")
        self.assertEqual(win._audio_options.notch, "off")
        self.assertTrue(any("audio/notch" in line for line in logs.output))

    def test_a_frequency_out_of_range_is_the_nearest_end(self):
        from sdr_module.gui.audio_chain import DEFAULT_NOTCH_HZ, NOTCH_RANGE_HZ

        for raw, expected in (("99999", NOTCH_RANGE_HZ[1]), ("3", NOTCH_RANGE_HZ[0])):
            with self.subTest(raw=raw):
                with self.assertLogs(level="WARNING") as logs:
                    win = self.window(audio__notch="manual", audio__notch_hz=raw)
                self.assertEqual(win._control_panel.notch_hz(), expected)
                self.assertEqual(win._audio_options.notch_hz, float(expected))
                self.assertTrue(any("audio/notch_hz" in m for m in logs.output))
        win = self.window(audio__notch_hz="garbage")
        self.assertEqual(win._control_panel.notch_hz(), DEFAULT_NOTCH_HZ)


if __name__ == "__main__":
    unittest.main()
