#!/usr/bin/env python3
"""
Dialogs: the third review round.

The keyboard shortcut list checked against the real main window (F6, the
Import Recording wording, sentence case), the welcome screen's and the
scanner's band presets in the main window's order, and the demo device
delivering its signal in real time: fast reads, one continuous stream
whatever the block sizes, and music on 100.1 MHz that FM-demodulates to
clean audio (and that the scanner still reports on its channel).

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round3_dialogs.py``
"""

import os
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QLabel


def _title_case_words(text):
    """Words after the first that are capitalized like a title."""
    return [
        word
        for word in text.split()[1:]
        if word[:1].isupper() and word[1:].isalpha() and word[1:].islower()
    ]


# --------------------------------------------------------------------------- #
# Keyboard shortcuts
# --------------------------------------------------------------------------- #
class TestShortcutTable(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_descriptions_are_in_sentence_case(self):
        from sdr_module.gui.help_dialog import SHORTCUTS

        for _category, _keys, description in SHORTCUTS:
            # "SDR Module" is the product's name.
            self.assertEqual(
                [w for w in _title_case_words(description) if w != "Module"],
                [],
                description,
            )

    def test_import_and_focus_entries(self):
        from sdr_module.gui.help_dialog import SHORTCUTS

        table = {keys: (cat, desc) for cat, keys, desc in SHORTCUTS}
        self.assertEqual(
            table[("Ctrl+O",)], ("File", "Import a recording to convert its format")
        )
        self.assertEqual(
            table[("F6",)], ("View", "Put the keyboard focus on the spectrum")
        )

    def test_unknown_menu_items_read_in_sentence_case(self):
        from sdr_module.gui.help_dialog import _describe_action

        self.assertEqual(_describe_action("Focus Spectrum"), "Focus spectrum")
        self.assertEqual(
            _describe_action("Import Channels (CHIRP CSV)"),
            "Import channels (CHIRP CSV)",
        )
        self.assertEqual(_describe_action("About SDR Module"), "About SDR Module")
        self.assertEqual(_describe_action("Record I/Q"), "Record I/Q")
        self.assertEqual(_describe_action("Panels › S-Meter"), "Show the S-Meter panel")

    def test_footer_points_to_f6(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = HelpDialog()
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        self.assertTrue(
            any("press F6 or Esc (or click the spectrum) first." in t for t in texts),
            texts,
        )
        dialog.deleteLater()


class TestShortcutsMatchTheMainWindow(_WindowTestCase):
    """Every table entry is a real shortcut, filed under the menu holding it,
    and every menu shortcut has a written description."""

    def test_table_matches_the_menus(self):
        from sdr_module.gui.help_dialog import SHORTCUTS, HelpDialog, collect_bindings

        bindings = collect_bindings(self.win)
        entries = HelpDialog(self.win).shortcut_entries()
        listed = {keys: (cat, desc) for cat, keys, desc in entries}
        for category, keys, description in SHORTCUTS:
            if keys == ("Ctrl+R",) and not self.mw.HAS_HAM_RADIO:
                continue
            self.assertIn(keys, listed, f"{keys} is not bound by the window")
            self.assertEqual(listed[keys], (category, description))
            # Arrow keys are handled by the window itself, not a menu.
            menu = bindings.get(keys[0], ("",))[0]
            if menu:
                self.assertEqual(menu, category, keys)
        table_keys = {k for _c, keys, _d in SHORTCUTS for k in keys}
        for category, keys, description in entries:
            if set(keys) <= table_keys:
                continue
            # Only the View > Panels items are described by rule.
            self.assertEqual(category, "View", keys)
            self.assertRegex(description, r"^Show the .+ panel$")

    def test_f6_is_listed_under_view(self):
        from sdr_module.gui.help_dialog import HelpDialog

        entries = HelpDialog(self.win).shortcut_entries()
        self.assertIn(
            ("View", ("F6",), "Put the keyboard focus on the spectrum"), entries
        )


# --------------------------------------------------------------------------- #
# Band presets
# --------------------------------------------------------------------------- #
class TestBandPresetOrder(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_welcome_bands_follow_the_band_presets_menu(self):
        from sdr_module.gui import main_window
        from sdr_module.gui.first_run_wizard import BAND_PRESETS

        menu = main_window.BAND_PRESETS
        self.assertEqual(list(BAND_PRESETS.values()), [p.frequency_hz for p in menu])
        for name, preset in zip(BAND_PRESETS, menu, strict=True):
            self.assertTrue(
                name.startswith(main_window._plain(preset.label) + " ("), name
            )

    def test_scanner_bands_keep_the_same_order_and_names(self):
        from sdr_module.gui.first_run_wizard import BAND_PRESETS
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        names = list(BAND_PRESETS)
        labels = [label for label, *_rest in SCAN_PRESETS]
        self.assertTrue(set(labels) <= set(names), labels)
        self.assertEqual(labels, sorted(labels, key=names.index))
        # The default range (88-108 MHz) is the first preset.
        self.assertEqual(SCAN_PRESETS[0][1:3], (88.0, 108.0))

    def test_welcome_starts_on_the_default_frequency(self):
        from sdr_module.gui.first_run_wizard import BAND_PRESETS, FirstRunWizard
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        wiz = FirstRunWizard(hardware_found=False, missing_drivers=[])
        self.assertEqual(BAND_PRESETS[wiz._band.currentText()], DEFAULT_FREQUENCY_HZ)
        self.assertEqual(wiz.selected_frequency(), DEFAULT_FREQUENCY_HZ)
        wiz.deleteLater()


# --------------------------------------------------------------------------- #
# Demo device in real time
# --------------------------------------------------------------------------- #
def _demo(freq=100.1e6, rate=2.4e6, noise=True):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(freq)
    dev.start_rx()
    if not noise:
        dev._noise = lambda n, sigma: np.zeros(n, dtype=np.complex64)
    return dev


class TestDemoDeviceSpeed(unittest.TestCase):
    def test_a_frame_of_signal_takes_a_fraction_of_its_duration(self):
        # The busiest view (three FM stations and narrow carriers). A frame
        # at 30 Hz is 0.034 s of signal; reading it takes a few ms.
        for freq in (100.1e6, 100.0e6, 125e6):
            dev = _demo(freq)
            n = int(dev.sample_rate * 0.034)
            dev.read_samples(n)  # builds the modulation loops once
            times = []
            for _ in range(8):
                start = time.perf_counter()
                samples = dev.read_samples(n)
                times.append(time.perf_counter() - start)
            self.assertEqual(samples.shape, (n,))
            self.assertLess(float(np.median(times)), 0.034 / 2, freq)

    def test_the_largest_read_is_well_under_real_time(self):
        dev = _demo()
        n = int(dev.sample_rate * 0.25)
        dev.read_samples(n)
        start = time.perf_counter()
        dev.read_samples(n)
        self.assertLess(time.perf_counter() - start, 0.25 / 2)

    def test_starts_on_the_default_frequency(self):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        self.assertEqual(MockDevice().frequency, DEFAULT_FREQUENCY_HZ)


class TestDemoDeviceStream(unittest.TestCase):
    def test_reads_of_any_size_make_one_continuous_stream(self):
        # Every kind of signal: FM broadcast, CW, NFM, AM, bursts, pulses.
        for freq in (100.0e6, 125e6, 433.92e6, 1090e6):
            whole = _demo(freq, noise=False).read_samples(700_000)
            dev = _demo(freq, noise=False)
            parts, got = [], 0
            for size in (81_600, 2048, 1, 300_000, 12_345, 7, 400_000):
                size = min(size, len(whole) - got)
                parts.append(dev.read_samples(size))
                got += size
            joined = np.concatenate(parts)
            self.assertEqual(len(joined), len(whole))
            np.testing.assert_allclose(joined, whole, atol=2e-5, err_msg=str(freq))

    def test_reading_behind_real_time_skips_ahead(self):
        from sdr_module.gui import device_dialog

        dev = _demo()
        clock = mock.Mock(return_value=100.0)
        with mock.patch.object(device_dialog.time, "monotonic", clock):
            dev.read_samples(2048)
            clock.return_value = 110.0  # ten seconds later
            dev.read_samples(2048)
        # The second block ends "now": 10 s of signal on from the first.
        self.assertAlmostEqual(dev._stream_pos / dev.sample_rate, 10.0, delta=0.01)

    def test_timed_signals_switch_within_a_block(self):
        from sdr_module.gui.device_dialog import _on_ranges

        rate = 1000.0
        # On for 0.25 s of every second (seed 0: no offset), over 2 s.
        self.assertEqual(
            _on_ranges((1.0, 0.25), 0, 0.0, 2000, rate), [(0, 250), (1000, 1250)]
        )
        self.assertEqual(_on_ranges(None, 3, 5.0, 10, rate), [(0, 10)])


class TestDemoMusic(unittest.TestCase):
    """100.1 MHz plays a tune: FM-demodulated it is the tune, cleanly."""

    def setUp(self):
        require_pyqt6(self)

    def _reference(self, rate, start_pos, audio_rate, count):
        """The tune's program audio (L+R) as the receiver should output it."""
        from sdr_module.gui import device_dialog as dd

        program = dd._loops("wfm", dd._MUSIC_SEED, rate)[0]
        factor = program.factor
        left, right = dd._music_channels(rate / factor, len(program.values))
        mono = (left + right) / 2 * dd._PROGRAM_SHARE
        positions = (start_pos + np.arange(count) * (rate / audio_rate)) / factor
        return np.interp(positions % len(mono), np.arange(len(mono)), mono)

    @staticmethod
    def _audio_band(signal, audio_rate, top_hz=15e3):
        spectrum = np.fft.rfft(signal)
        spectrum[np.fft.rfftfreq(len(signal), 1 / audio_rate) > top_hz] = 0
        return np.fft.irfft(spectrum, len(signal))

    def test_fm_broadcast_audio_is_the_tune(self):
        from sdr_module.gui import main_window as mw

        rate = 2.4e6
        dev = _demo(100.1e6, rate)
        chain = mw._ReceiverChain(rate, "FM", 200e3)
        start = dev._stream_pos
        blocks = []
        for size in (81_600, 2048, 250_000, 60_000) * 6:  # about 2.4 s
            channel = chain.channelize(dev.read_samples(size))
            blocks.append(chain.demodulate(channel, 75e3))
        audio = np.concatenate(blocks).astype(float)
        self.assertLess(float(np.mean(np.abs(audio) > 0.98)), 0.001)  # no clipping
        self.assertGreater(float(np.sqrt(np.mean(audio**2))), 0.1)  # loud enough

        reference = self._reference(rate, start, chain.audio_rate, len(audio))
        skip = int(0.2 * chain.audio_rate)  # filters settling
        heard = self._audio_band(audio[skip:], chain.audio_rate)
        best = -np.inf
        for lag in range(0, 60):  # the receiver's filter delay
            a = heard[lag:]
            r = self._audio_band(reference[skip : skip + len(a)], chain.audio_rate)
            noise = np.sum((a - r) ** 2)
            best = max(best, 10 * np.log10(np.sum(r**2) / max(noise, 1e-20)))
        self.assertGreater(best, 30.0)  # clean: no clicks, no distortion

    def test_scanner_finds_the_music_station_on_its_channel(self):
        from sdr_module.gui.scanner_dialog import _ScanWorker

        dev = _demo(100.1e6)
        worker = _ScanWorker(dev, 100.1e6, 100.1e6, 200e3, -60.0)
        reports = set()
        for _ in range(24):  # all through the tune's loop
            reports.add(worker._measure(100.1e6, 200e3)[0])
            dev.read_samples(int(dev.sample_rate * 0.3))
        self.assertEqual(reports, {100.1e6})


if __name__ == "__main__":
    unittest.main()
