#!/usr/bin/env python3
"""
Dialogs and the demo device: the fourth review round.

The scanner's empty table reads like the app's other empty states (a bold
title over an italic hint); the shortcut list names Esc next to F6; the
demo's music station is broadcast with 75 us pre-emphasis, so the
receiver's de-emphasis gives back the tune with its treble intact; and
1090 MHz carries real Mode S frames that the ADS-B decoder follows through
the main window.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round4_dialogs.py``
"""

import math
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
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow

KLM1023_IDENT = "8D4840D6202CC371C32CE0576098"


def _settle(n=5):
    for _ in range(n):
        QApplication.processEvents()


def _demo(freq=1090e6, rate=2.4e6):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(freq)
    dev.start_rx()
    return dev


# --------------------------------------------------------------------------- #
# Scanner empty state
# --------------------------------------------------------------------------- #
class TestScannerEmptyState(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self._dialogs = []

    def tearDown(self):
        for dialog in self._dialogs:
            dialog.close()
            dialog.deleteLater()
        _settle()

    def dialog(self, device="demo"):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.scanner_dialog import ScannerDialog

        dialog = ScannerDialog(device=MockDevice() if device == "demo" else device)
        self._dialogs.append(dialog)
        dialog.resize(620, 600)
        dialog.show()
        _settle()
        return dialog

    def test_idle_table_shows_a_title_over_a_hint(self):
        from sdr_module.gui.decoder_panel import placeholder_html

        dialog = self.dialog()
        empty = dialog._empty
        self.assertTrue(empty.is_visible())
        self.assertEqual(
            empty.text(), "No results yet\nChoose a range and click Start Scan."
        )
        # Rendered like the other empty states: bold title, italic hint.
        self.assertEqual(empty.label.textFormat(), Qt.TextFormat.RichText)
        self.assertEqual(empty.label.text(), placeholder_html(empty.text()))
        self.assertIn("font-weight: 600", empty.label.text())
        self.assertEqual(empty.label.property("role"), "placeholder")

    def test_every_state_has_a_short_title_and_a_hint(self):
        dialog = self.dialog()
        texts = [dialog._empty.text()]
        dialog._update_empty_state(True)  # scanning
        texts.append(dialog._empty.text())
        with mock.patch.object(dialog, "_worker", object()):
            dialog._update_empty_state(False)  # a sweep found nothing
            texts.append(dialog._empty.text())
        texts.append(self.dialog(device=None)._empty.text())
        self.assertEqual(len(set(texts)), 4)
        for text in texts:
            title, _sep, hint = text.partition("\n")
            self.assertTrue(hint, text)
            self.assertFalse(title.endswith("."), title)
            self.assertLess(len(title), 40, title)
        self.assertIn("Device > Connect", texts[-1])
        self.assertIn("Use Demo Device", texts[-1])

    def test_placeholder_covers_the_table_and_hides_with_results(self):
        dialog = self.dialog()
        viewport = dialog._table.viewport()
        self.assertEqual(dialog._empty.label.geometry(), viewport.rect())
        dialog.resize(700, 660)
        _settle()
        self.assertEqual(dialog._empty.label.geometry(), viewport.rect())
        dialog._on_hit(100.1e6, -30.0, -90.0)
        self.assertFalse(dialog._empty.is_visible())
        dialog._clear_results()
        self.assertTrue(dialog._empty.is_visible())


# --------------------------------------------------------------------------- #
# Help: Esc
# --------------------------------------------------------------------------- #
class TestHelpEsc(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_esc_follows_f6_in_the_table(self):
        from sdr_module.gui.help_dialog import SHORTCUTS

        keys = [k for _c, k, _d in SHORTCUTS]
        esc = keys.index(("Esc",))
        self.assertEqual(keys[esc - 1], ("F6",))
        self.assertEqual(
            SHORTCUTS[esc],
            ("View", ("Esc",), "Return the keyboard focus to the spectrum"),
        )

    def test_footer_names_esc(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = HelpDialog()
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        self.assertTrue(
            any(
                t.endswith("press F6 or Esc (or click the spectrum) first.")
                for t in texts
            ),
            texts,
        )
        dialog.deleteLater()

    def test_esc_is_listed_for_windows_that_handle_it(self):
        from sdr_module.gui.help_dialog import HelpDialog

        class Plain(QMainWindow):
            pass

        class Focusing(QMainWindow):
            def _focus_plots(self):
                pass

        for cls, listed in ((Plain, False), (Focusing, True)):
            win = cls()
            entries = HelpDialog(win).shortcut_entries()
            esc = [e for e in entries if e[1] == ("Esc",)]
            self.assertEqual(
                esc,
                (
                    [("View", ("Esc",), "Return the keyboard focus to the spectrum")]
                    if listed
                    else []
                ),
            )
            win.deleteLater()
        _settle()


class TestHelpEscInTheMainWindow(_WindowTestCase):
    def test_esc_is_listed_and_returns_focus_to_the_spectrum(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = HelpDialog(self.win)
        self.assertIn(
            ("View", ("Esc",), "Return the keyboard focus to the spectrum"),
            dialog.shortcut_entries(),
        )
        dialog.deleteLater()

        combo = next(
            c
            for c in self.win._control_panel.findChildren(QComboBox)
            if c.isVisible() and c.isEnabled()
        )
        combo.setFocus()
        self.settle()
        self.assertIs(QApplication.focusWidget(), combo)
        QTest.keyClick(combo, Qt.Key.Key_Escape)
        self.settle()
        self.assertIs(QApplication.focusWidget(), self.win._spectrum)


# --------------------------------------------------------------------------- #
# Demo FM: 75 us pre-emphasis
# --------------------------------------------------------------------------- #
class TestDemoPreEmphasis(unittest.TestCase):
    def test_pre_emphasis_is_first_order_75_us(self):
        from sdr_module.gui.device_dialog import _pre_emphasis

        rate, size = 48e3, 48_000  # one second: whole cycles of each tone
        t = np.arange(size) / rate
        for freq in (100.0, 1e3, 2122.0, 10e3):
            out = _pre_emphasis(np.cos(2 * np.pi * freq * t), rate)
            response = 1 + 2j * np.pi * freq * 75e-6
            expected = np.abs(response) * np.cos(
                2 * np.pi * freq * t + np.angle(response)
            )
            np.testing.assert_allclose(out, expected, atol=1e-9, err_msg=str(freq))

    def test_music_is_broadcast_pre_emphasized_at_full_deviation(self):
        from sdr_module.gui import device_dialog as dd

        rate = 2.4e6
        program, _subcarrier = dd._loops("wfm", dd._MUSIC_SEED, rate)
        loop_rate = rate / program.factor
        left, right, left_pre, right_pre = dd._music_broadcast(
            loop_rate, len(program.values)
        )
        # (float32 channels: equal to about 1e-7)
        np.testing.assert_allclose(
            left_pre, dd._pre_emphasis(left, loop_rate), atol=1e-5
        )
        np.testing.assert_allclose(
            right_pre, dd._pre_emphasis(right, loop_rate), atol=1e-5
        )
        # The louder broadcast channel is at full scale, never beyond.
        peak = max(np.max(np.abs(left_pre)), np.max(np.abs(right_pre)))
        self.assertAlmostEqual(float(peak), 1.0, places=6)
        # The program's frequency deviation (phase slope) is the
        # pre-emphasized mix, within the 75 kHz of broadcast FM.
        phase = program.values.astype(float)
        deviation = (
            np.diff(np.concatenate((phase, phase[:1]))) * loop_rate / (2 * np.pi)
        )
        mono_pre = (left_pre + right_pre) / 2 * dd._PROGRAM_SHARE * 75e3
        mono_pre = np.roll(mono_pre - np.mean(mono_pre), -1)
        self.assertLess(float(np.max(np.abs(deviation - mono_pre))), 75.0)  # Hz
        self.assertLessEqual(float(np.max(np.abs(deviation))), 75e3)

    def test_program_loop_is_seamless(self):
        from sdr_module.gui import device_dialog as dd

        program = dd._loops("wfm", dd._MUSIC_SEED, 2.4e6)[0].values.astype(float)
        steps = np.abs(np.diff(program))
        seam = abs(program[0] - program[-1])
        self.assertLessEqual(seam, float(np.max(steps)))

    def test_receiver_hears_the_tune_with_its_treble(self):
        require_pyqt6(self)
        from sdr_module.gui import device_dialog as dd
        from sdr_module.gui import main_window as mw

        rate = 2.4e6
        dev = _demo(100.1e6, rate)
        dev._noise = lambda n, sigma: np.zeros(n, dtype=np.complex64)
        chain = mw._ReceiverChain(rate, "FM", 200e3)
        start = dev._stream_pos
        audio = np.concatenate(
            [
                chain.demodulate(chain.channelize(dev.read_samples(240_000)), 75e3)
                for _ in range(30)
            ]
        ).astype(float)

        program = dd._loops("wfm", dd._MUSIC_SEED, rate)[0]
        left, right = dd._music_channels(rate / program.factor, len(program.values))
        mono = (left + right) / 2 * dd._PROGRAM_SHARE
        where = (start + np.arange(len(audio)) * (rate / chain.audio_rate)) / (
            program.factor
        )
        tune = np.interp(where % len(mono), np.arange(len(mono)), mono)

        skip = int(0.2 * chain.audio_rate)
        heard = np.abs(np.fft.rfft(audio[skip:])) ** 2
        sent = np.abs(np.fft.rfft(tune[skip:])) ** 2
        freqs = np.fft.rfftfreq(len(audio) - skip, 1 / chain.audio_rate)
        for low, high in ((100, 500), (500, 1500), (1500, 3000), (3000, 6000)):
            band = (freqs >= low) & (freqs < high)
            ratio_db = 10 * np.log10(heard[band].sum() / sent[band].sum())
            self.assertLess(abs(ratio_db), 1.0, (low, high, ratio_db))

    def test_reads_stay_fast(self):
        # About 0.03 us per sample on a desktop; the bound leaves room for
        # a busy test machine.
        for freq in (100.1e6, 1090e6):
            dev = _demo(freq)
            n = int(dev.sample_rate * 0.034)
            dev.read_samples(n)
            times = []
            for _ in range(10):
                start = time.perf_counter()
                dev.read_samples(n)
                times.append(time.perf_counter() - start)
            per_sample_us = float(np.median(times)) / n * 1e6
            self.assertLess(per_sample_us, 0.15, freq)


# --------------------------------------------------------------------------- #
# Demo ADS-B on 1090 MHz
# --------------------------------------------------------------------------- #
class TestDemoModeSFrames(unittest.TestCase):
    def test_identification_is_the_classic_klm1023_frame(self):
        from sdr_module.gui import device_dialog as dd

        self.assertEqual(dd._aircraft_frame("ident", 0.0).hex().upper(), KLM1023_IDENT)
        # The parity of the published example pair checks out too.
        for frame in ("8D40621D58C386435CC412692AD6", "8D40621D58C382D690C8AC2863A7"):
            data = bytes.fromhex(frame)
            self.assertEqual(dd._crc24(data[:11]), int.from_bytes(data[11:], "big"))

    def test_every_frame_passes_the_decoders_checks(self):
        from sdr_module.dsp.protocols import ADSBDecoder
        from sdr_module.gui import device_dialog as dd

        decoder = ADSBDecoder(2e6)
        fields = {}
        for t in (3.0, 3.5):
            for _offset, message in dd._SQUITTERS:
                msg = decoder._parse_frame(dd._aircraft_frame(message, t))
                self.assertIsNotNone(msg, message)
                self.assertEqual(msg.icao_address, "4840D6")
                fields.setdefault(message, msg)
        self.assertEqual(fields["ident"].callsign, "KLM1023")
        self.assertEqual(fields["even"].altitude, 38000)
        self.assertEqual(fields["odd"].altitude, 38000)
        self.assertEqual(round(fields["velocity"].velocity), 450)
        self.assertAlmostEqual(fields["velocity"].heading, 265.0, delta=0.5)
        self.assertEqual(fields["velocity"].vertical_rate, 0)
        self.assertEqual(fields["squawk"].downlink_format, 5)
        self.assertEqual(fields["squawk"].squawk, "3472")

    def test_positions_decode_along_the_whole_flight(self):
        from sdr_module.dsp.protocols import ADSBDecoder
        from sdr_module.gui import device_dialog as dd

        for t in np.arange(0.0, dd._FLIGHT_LOOP_S, 23.7):
            decoder = ADSBDecoder(2e6)
            decoder._parse_frame(dd._aircraft_frame("even", t))
            msg = decoder._parse_frame(dd._aircraft_frame("odd", t + 0.5))
            lat, lon = dd._aircraft_position(t + 0.5)
            self.assertAlmostEqual(msg.latitude, lat, delta=0.001, msg=t)
            self.assertAlmostEqual(msg.longitude, lon, delta=0.001, msg=t)
        # It cruises west at 450 kt: about 116 m per half second.
        lat0, lon0 = dd._aircraft_position(10.0)
        lat1, lon1 = dd._aircraft_position(10.5)
        metres = math.hypot(
            (lat1 - lat0) * 111_195,
            (lon1 - lon0) * 111_195 * math.cos(math.radians(lat0)),
        )
        self.assertAlmostEqual(metres, 450 * 1852 / 3600 / 2, delta=1.0)
        self.assertLess(lon1, lon0)

    def test_pulses_follow_mode_s_timing(self):
        from sdr_module.gui import device_dialog as dd

        frame = bytes.fromhex(KLM1023_IDENT)
        bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
        chips = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # 8 us preamble
        for bit in bits:
            chips += [1, 0] if bit else [0, 1]
        # At 2 MS/s one sample is one 0.5 us chip.
        first, env = dd._pulse_envelope(dd._modes_pulses(frame), 100.0, 2e6)
        self.assertEqual(first, 100)
        np.testing.assert_array_equal(env, np.array(chips, dtype=np.float32))
        # At 2.4 MS/s the samples share the chips: the same 58 us of pulses
        # (4 preamble + 112 bit pulses of 0.5 us), whatever the phase.
        for start in (0.0, 0.3, 0.77):
            _first, env = dd._pulse_envelope(dd._modes_pulses(frame), start, 2.4e6)
            self.assertAlmostEqual(float(env.sum()), 58.0 * 2.4, places=3)
            self.assertLessEqual(float(env.max()), 1.0 + 1e-6)

    def test_replies_stand_well_above_the_noise(self):
        from sdr_module.gui import device_dialog as dd

        dev = dd.MockDevice()
        level = next(lvl for f, kind, lvl, _d in dd._DEMO_SIGNALS if kind == "modes")
        sigma = dev._noise_sigma(dev._NOISE_FLOOR_DB)
        self.assertGreater(20 * math.log10(10 ** (level / 20) / sigma), 25.0)


class TestDemoModeSDecoding(unittest.TestCase):
    """The demo stream at 1090 MHz, through the window's decoder path."""

    @staticmethod
    def _decode(rate, seconds, device=None):
        from sdr_module.dsp.protocols import (
            ADSBDecoder,
            ProtocolType,
            demodulate_for_protocol,
        )

        dev = device or _demo(1090e6, rate)
        decoder = ADSBDecoder(rate)
        rng = np.random.default_rng(int(rate))
        messages, total = [], 0
        while total < rate * seconds:
            n = int(rng.integers(2048, 120_000))
            total += n
            baseband = demodulate_for_protocol(dev.read_samples(n), ProtocolType.ADSB)
            messages += decoder.decode(baseband)
        return messages

    def test_the_airliner_decodes_at_every_ads_b_rate(self):
        from sdr_module.gui import device_dialog as dd

        seconds = 6.0
        sent = seconds / dd._SQUITTER_CYCLE_S * len(dd._SQUITTERS)
        for rate in (2.0e6, 2.4e6, 3.2e6):
            messages = self._decode(rate, seconds)
            self.assertEqual({m.icao_address for m in messages}, {"4840D6"}, rate)
            self.assertGreater(len(messages), 0.6 * sent, rate)
            self.assertIn("KLM1023", {m.callsign for m in messages}, rate)
            self.assertIn("3472", {m.squawk for m in messages}, rate)
            self.assertIn(38000, {m.altitude for m in messages}, rate)
            self.assertIn(450, {round(m.velocity) for m in messages}, rate)
            located = [m for m in messages if m.latitude]
            self.assertTrue(located, rate)
            for msg in located:
                self.assertAlmostEqual(msg.latitude, 52.25, delta=0.1)
                self.assertAlmostEqual(msg.longitude, 3.9, delta=0.1)

    def test_other_replies_never_decode(self):
        from sdr_module.gui import device_dialog as dd

        dev = _demo(1090e6, 2.4e6)
        with mock.patch.object(dd, "_squitter_spans", lambda pos, n, rate: []):
            samples = dev.read_samples(int(2.4e6 * 0.5))
            # The other aircraft are on the air...
            self.assertGreater(float(np.max(np.abs(samples))), 0.01)
            dev = _demo(1090e6, 2.4e6)
            self.assertEqual(self._decode(2.4e6, 4.0, device=dev), [])


class TestDemoAdsbInTheWindow(_WindowTestCase):
    demo = True

    def test_the_decoder_panel_follows_klm1023(self):
        win = self.win
        dev = win._device
        if not dev.is_streaming:
            dev.start_rx()
        win.set_frequency(1090e6)
        self.assertEqual(dev.frequency, 1090e6)
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("ADS-B")
        panel._enabled_check.setChecked(True)
        self.assertIsNotNone(win._decoder)

        # Five seconds of signal, a display frame's worth at a time.
        block = int(win._device_sample_rate() / 10)
        for _ in range(50):
            win._process_block(dev.read_samples(block))
        rows = [(m["address"], m["content"]) for m in panel._messages]
        self.assertTrue(rows)
        self.assertEqual({address for address, _c in rows}, {"4840D6"})
        contents = [content for _a, content in rows]
        self.assertIn("KLM1023", contents)
        self.assertTrue(any("450 kt" in c for c in contents), contents)
        located = [c for c in contents if "38,000 ft" in c and "52.2" in c]
        self.assertTrue(located, contents)


if __name__ == "__main__":
    unittest.main()
