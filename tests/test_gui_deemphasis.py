#!/usr/bin/env python3
"""
Radio > Audio > De-emphasis in the window (M-A1): 75 us (the Americas,
South Korea, the default) or 50 us (Europe, Australia, most of the world),
saved (audio/deemphasis), safe when damaged, and applied at once to what
plays, to Record Audio and to the radio tuner's station (the tuner tunes
the main receiver), in mono and in stereo.
"""

import math
import os
import shutil
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.dsp.fm_stereo import StereoDecoder  # noqa: E402
from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    fm,
    noise,
    pretend_audio_output,
)

if HAS_PYQT6:
    from PyQt6.QtGui import QActionGroup
    from PyQt6.QtWidgets import QApplication

AUDIO = 48000.0
LOW_HZ, HIGH_HZ = 1000.0, 10e3
FIFTY = "50 µs (Europe, Australia, most of the world)"
SEVENTY_FIVE = "75 µs (the Americas, South Korea)"


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


def _level(x, freq):
    t = np.arange(len(x)) / AUDIO
    return float(
        np.hypot(
            2 * np.mean(x * np.cos(2 * np.pi * freq * t)),
            2 * np.mean(x * np.sin(2 * np.pi * freq * t)),
        )
    )


def _station(seconds):
    """A mono broadcast FM station (no pilot) sending 1 and 10 kHz at
    0.25 of the deviation each, unemphasized, at -30 dBFS."""
    n = int(seconds * RATE) // FRAME * FRAME
    t = np.arange(n) / RATE
    program = 0.25 * np.sin(2 * np.pi * LOW_HZ * t) + 0.25 * np.sin(
        2 * np.pi * HIGH_HZ * t
    )
    return (fm(program, 75e3, -30.0) + noise(n, -100.0)).astype(np.complex64)


def _treble_db(audio):
    """How far the 10 kHz tone is below the 1 kHz one (dB)."""
    return 20 * math.log10(_level(audio, HIGH_HZ) / _level(audio, LOW_HZ))


def _expected_treble_db(tau):
    """1/|1 + j 2 pi f tau| at 10 kHz against 1 kHz (dB)."""
    return -10 * math.log10(
        (1 + (2 * math.pi * HIGH_HZ * tau) ** 2)
        / (1 + (2 * math.pi * LOW_HZ * tau) ** 2)
    )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheMenu(_WindowTestCase):
    def test_radio_audio_offers_75_and_50_us(self):
        menu = _menu(self.win, "Radio", "Audio", "De-emphasis").menu()
        actions = [a for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(
            [a.text().replace("&", "") for a in actions],
            [
                SEVENTY_FIVE,
                FIFTY,
            ],
        )
        for action in actions:
            self.assertTrue(action.isCheckable())
            self.assertTrue(action.toolTip())
            self.assertIsInstance(action.actionGroup(), QActionGroup)
            self.assertTrue(action.actionGroup().isExclusive())
        self.assertTrue(actions[0].isChecked())  # (the default: as before)
        self.assertFalse(actions[1].isChecked())
        self.assertEqual(self.win._audio_options.deemphasis_s, 75e-6)
        self.assertTrue(menu.menuAction().toolTip())
        # It sits with Stereo, after Output Device....
        audio = [
            a.text().replace("&", "")
            for a in _menu(self.win, "Radio", "Audio").menu().actions()
            if not a.isSeparator()
        ]
        self.assertLess(audio.index("Stereo"), audio.index("De-emphasis"))
        self.assertLess(audio.index("Output Device..."), audio.index("Stereo"))

    def test_choosing_50_us_applies_at_once_and_is_saved(self):
        fifty = _menu(
            self.win,
            "Radio",
            "Audio",
            "De-emphasis",
            FIFTY,
        )
        seventy_five = _menu(
            self.win,
            "Radio",
            "Audio",
            "De-emphasis",
            SEVENTY_FIVE,
        )
        with self.assertNoLogs(level="WARNING"):  # (what was asked: no problem)
            fifty.trigger()
        self.assertTrue(fifty.isChecked())
        self.assertFalse(seventy_five.isChecked())
        self.assertEqual(self.win._audio_options.deemphasis_s, 50e-6)
        self.assertEqual(self.store.values["audio/deemphasis"], "50")
        self.assertIn("50 µs", self.win._message_label.text())
        seventy_five.trigger()
        self.assertEqual(self.win._audio_options.deemphasis_s, 75e-6)
        self.assertEqual(self.store.values["audio/deemphasis"], "75")
        self.assertIn("75 µs", self.win._message_label.text())

    def test_a_saved_choice_is_restored(self):
        self.store.values["audio/deemphasis"] = "50"
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertEqual(other._audio_options.deemphasis_s, 50e-6)
        fifty = _menu(
            other,
            "Radio",
            "Audio",
            "De-emphasis",
            FIFTY,
        )
        self.assertTrue(fifty.isChecked())


class TestADamagedSetting(_IniSettings):
    """A value that is no choice gives 75 us (today's behaviour), logged."""

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

    def test_anything_but_75_or_50_is_75_us(self):
        for raw in ("60", "loud", "0.00005"):
            with self.subTest(raw=raw):
                with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
                    win = self.window(audio__deemphasis=raw)
                self.assertEqual(win._audio_options.deemphasis_s, 75e-6)
                self.assertTrue(any("audio/deemphasis" in m for m in logs.output))
                checked = _menu(
                    win,
                    "Radio",
                    "Audio",
                    "De-emphasis",
                    SEVENTY_FIVE,
                )
                self.assertTrue(checked.isChecked())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestWhatPlays(_WindowTestCase):
    """What the speaker and Record Audio get (48 kHz audio captured)."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.panel.set_squelch_enabled(False)
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.win._reset_demodulator()

    def choose(self, key):
        label = {
            "75": SEVENTY_FIVE,
            "50": FIFTY,
        }[key]
        _menu(self.win, "Radio", "Audio", "De-emphasis", label).trigger()

    def test_the_speaker_follows_a_change_while_it_plays(self):
        frames = Frames(self, _station(2.0))
        frames.run(20)
        self.choose("50")
        frames.run()
        audio = [block if block.ndim == 1 else block[:, 0] for block in frames.audio]
        before = np.concatenate(audio[8:19])
        after = np.concatenate(audio[24:])
        self.assertAlmostEqual(
            _treble_db(before), _expected_treble_db(75e-6), delta=0.2
        )
        self.assertAlmostEqual(_treble_db(after), _expected_treble_db(50e-6), delta=0.2)
        self.assertTrue(all(len(block) for block in frames.audio[1:]))
        self.assertEqual(self.win._receiver_chain().deemphasis_s, 50e-6)

    def test_record_audio_records_with_it(self):
        folder = tempfile.mkdtemp(prefix="deemph-rec-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        self.choose("50")
        frames = Frames(self, _station(1.5))
        self.assertTrue(self.win._start_audio_recording())
        frames.run()
        path = self.win._stop_audio_recording()
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        heard = data.reshape(-1, channels)[:, 0].astype(float) / 32768.0
        tail = heard[int(0.5 * AUDIO) :]
        self.assertAlmostEqual(_treble_db(tail), _expected_treble_db(50e-6), delta=0.2)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestStereoStations(_WindowTestCase):
    """The demo's 100.1 MHz stereo station: left, right and the blend's
    hiss estimate are heard through the setting; the radio tuner, which
    tunes the main receiver, hears it too."""

    FRAME = 80_000

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        for target, name, value in (
            (device_dialog, "_MAX_LAG_S", 1e9),
            (StereoDecoder, "RISE_S", 0.05),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        pretend_audio_output(self.win)
        self.written = []
        self.win._audio.write = self.written.append

    def frames(self, device, count):
        for _ in range(count):
            self.win._process_block(device.read_samples(self.FRAME))

    def device(self, freq=100.1e6):
        from sdr_module.gui.device_dialog import MockDevice

        device = MockDevice()
        device.set_sample_rate(2.4e6)
        device.set_frequency(freq)
        device._running = True
        device._rng = np.random.default_rng(11)
        return device

    def test_stereo_plays_through_it(self):
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        device = self.device()
        self.frames(device, 8)
        _menu(
            self.win,
            "Radio",
            "Audio",
            "De-emphasis",
            FIFTY,
        ).trigger()
        self.written.clear()
        self.frames(device, 8)
        chain = self.win._receiver_chain()
        self.assertEqual(chain.deemphasis_s, 50e-6)
        self.assertEqual(chain.stereo_decoder.tau_s, 50e-6)
        stereo = [block for block in self.written if np.ndim(block) == 2]
        self.assertGreaterEqual(len(stereo), 6)
        self.assertEqual(self.win._stereo_badge.text(), "STEREO")

    def test_the_radio_tuner_hears_it(self):
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        _menu(
            self.win,
            "Radio",
            "Audio",
            "De-emphasis",
            FIFTY,
        ).trigger()
        self.win.set_frequency(98.0e6)  # (the tuner opens on its FM band)
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.addCleanup(tuner.close)
        tuner._tuning_dial.tune_to(100.1e6)
        self.settle()
        self.assertEqual(self.win._current_frequency(), 100.1e6)
        self.frames(self.device(), 12)
        chain = self.win._receiver_chain()
        self.assertEqual(chain.deemphasis_s, 50e-6)
        self.assertEqual(chain.stereo_decoder.tau_s, 50e-6)
        self.assertTrue(tuner._stereo)  # (its lamp: stereo decoded)


if __name__ == "__main__":
    unittest.main()
