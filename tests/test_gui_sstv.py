"""SSTV panel and its main-window wiring.

hm:HAM-01 (decoding through the real window: FM at 5 kHz deviation, noise,
clock error), hm:GAP-10 (the demo device's SSTV test card on 145.800 MHz),
hm:GAP-11 (mode choice and Start now), hm:UX-09 (partial images), hm:HAM-10
(Clear while receiving), hm:GAP-09 (history across restarts), hm:UX-03
(tone meter) and hm:UX-04 (Tune to ISS, Switch to FM).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_sstv.py``
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtTest import QSignalSpy
    from PyQt6.QtWidgets import QApplication

from sdr_module.ham.sstv import find_mode  # noqa: E402
from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card  # noqa: E402

FS = 48000.0
GUI_BLOCK = 1600


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float).ravel() - a.mean()
    b = b.astype(float).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        self.tmp = tempfile.TemporaryDirectory()
        self.panel = SSTVPanel()
        self.panel.set_save_directory(self.tmp.name)
        self.panel.resize(360, 900)
        self.panel.show()
        settle()

    def tearDown(self):
        if hasattr(self, "panel"):
            if self.panel.is_receiving():
                self.panel._start_btn.click()
            self.panel.close()
            self.panel.deleteLater()
            settle()
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def feed(self, audio: np.ndarray) -> None:
        for i in range(0, len(audio), GUI_BLOCK):
            self.panel.process_audio(audio[i : i + GUI_BLOCK], FS)

    def history_texts(self):
        lst = self.panel._history_list
        return [lst.item(i).text() for i in range(lst.count())]


class TestModeChoiceAndStartNow(_PanelCase):
    def test_mode_list_offers_auto_and_every_mode(self):
        from sdr_module.ham.sstv import SSTV_MODES

        combo = self.panel._mode_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(items[0], "Auto (VIS)")
        self.assertEqual(set(items[1:]), {s.name for s in SSTV_MODES.values()})
        self.assertEqual(combo.accessibleName(), "SSTV mode")
        for i in range(combo.count()):
            self.assertTrue(combo.itemData(i, 3))  # Qt.ToolTipRole

    def test_choosing_a_mode_forces_it(self):
        changed = QSignalSpy(self.panel.mode_choice_changed)
        self.assertTrue(self.panel.set_mode_choice("Martin 1"))
        self.assertEqual(changed[-1][0], "Martin 1")
        self.panel._start_btn.click()
        self.assertEqual(self.panel._decoder.get_forced_mode().name, "Martin 1")
        self.panel.set_mode_choice("Auto (VIS)")
        self.assertIsNone(self.panel._decoder.get_forced_mode())
        self.assertFalse(self.panel.set_mode_choice("Martin 9"))

    def test_start_now_decodes_a_transmission_under_way(self):
        spec = find_mode("Martin 2")
        card = sstv_test_card(spec.width, spec.height)
        audio = encode_sstv(card, spec, FS, vis=False)
        start = int(len(audio) * 0.25)
        self.panel.set_mode_choice("Martin 2")
        self.panel._start_btn.click()
        self.feed(audio[start : start + int(2 * FS)])
        self.panel._start_now_btn.click()
        self.assertIn("Martin 2", self.panel._status_label.text())
        self.feed(audio[start + int(2 * FS) : start + int(14 * FS)])
        decoder = self.panel._decoder
        self.assertTrue(decoder.is_receiving())
        rows = decoder.state.current_line - 1
        self.assertGreater(rows, 40)
        best = max(
            correlation(decoder.get_image()[:rows], card[o : o + rows])
            for o in range(0, spec.height - rows)
        )
        self.assertGreater(best, 0.97)

    def test_start_now_on_auto_without_syncs_asks_for_a_mode(self):
        self.panel._start_now_btn.click()
        self.assertTrue(self.panel.is_receiving(), "Start now also starts listening")
        self.assertEqual(self.panel._status_label.property("tone"), "warning")
        self.assertIn("Mode", self.panel._status_label.text())
        self.assertFalse(self.panel._decoder.is_receiving())


class TestPartialImages(_PanelCase):
    def receive_half(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(audio[: len(audio) // 2])
        self.panel._update_status()

    def test_stop_keeps_and_saves_the_partial_image(self):
        self.receive_half()
        self.assertFalse(self.panel._save_btn.isEnabled(), "not while receiving")
        self.panel._start_btn.click()  # Stop Decoder
        texts = self.history_texts()
        self.assertEqual(len(texts), 1)
        self.assertRegex(
            texts[0], r"^\d\d:\d\d · Robot 36 \(partial, 1\d\d/240 lines\)$"
        )
        saved = list(Path(self.tmp.name).glob("sstv_*_Robot_36_partial_*of240.png"))
        self.assertEqual(len(saved), 1)
        self.assertTrue(
            self.panel._save_btn.isEnabled(), "a partial image can be saved"
        )
        self.assertIn("partial image", self.panel._status_label.text())
        self.assertTrue(self.panel._image_display.has_image())

    def test_signal_loss_lists_the_partial_image(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        rng = np.random.default_rng(2)
        noise = (0.3 * rng.standard_normal(int(6 * FS))).astype(np.float32)
        self.panel._start_btn.click()
        self.feed(np.concatenate([audio[: int(len(audio) * 0.6)], noise]))
        self.assertTrue(self.panel.is_receiving(), "keeps listening")
        self.assertEqual(len(self.history_texts()), 1)
        self.assertIn("partial", self.history_texts()[0])
        self.assertEqual(self.panel._status_label.property("tone"), "warning")
        self.assertIn("faded", self.panel._status_label.text())

    def test_a_new_image_cuts_the_last_one_short(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(audio[: len(audio) // 2])
        statuses = []
        self.panel._decoder.set_on_image_lost(
            lambda image, lines: (
                self.panel._on_image_lost(image, lines),
                statuses.append(self.panel._status_label.text()),
            )
        )
        self.feed(audio[: int(2 * FS)])  # the next VIS arrives
        self.assertEqual(len(self.history_texts()), 1)
        self.assertIn("partial", self.history_texts()[0])
        self.assertTrue(statuses[0].startswith("A new image started before"))
        self.assertTrue(self.panel._decoder.is_receiving(), "decoding the new one")

    def test_a_few_lines_are_not_kept(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS, lines=4)
        self.panel._start_btn.click()
        self.feed(audio)
        self.panel._start_btn.click()
        self.assertEqual(self.history_texts(), [])

    def test_clear_while_receiving_discards_explicitly(self):
        """hm:HAM-10: Clear no longer silently aborts the image."""
        self.receive_half()
        clear = self.panel._clear_btn
        self.assertEqual(clear.text(), "Discard Image")
        self.assertIn("Discard the image being received", clear.toolTip())
        clear.click()
        self.assertFalse(self.panel._decoder.is_receiving())
        self.assertTrue(self.panel.is_receiving(), "still listening")
        self.assertEqual(
            self.panel._status_label.text(),
            "Image discarded. Listening for the next one…",
        )
        self.assertEqual(self.history_texts(), [])
        self.panel._update_status()
        self.assertEqual(clear.text(), "Clear")

    def test_audio_gap_ends_the_image_as_partial(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(audio[: len(audio) // 2])
        self.panel._last_audio_time -= 5.0  # the receiver paused
        self.feed(audio[len(audio) // 2 : len(audio) // 2 + 4800])
        self.assertEqual(len(self.history_texts()), 1)
        self.assertIn("audio stopped", self.panel._status_label.text())


class TestHistoryAcrossRestarts(_PanelCase):
    def write_image(self, name, height=256):
        from sdr_module.ham.gui.sstv_panel import save_rgb_image

        path = Path(self.tmp.name) / name
        save_rgb_image(sstv_test_card(320, height), str(path))
        return path

    def test_saved_images_are_listed_and_open(self):
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        self.write_image("sstv_20261029_101530_Martin_1.png")
        self.write_image("sstv_20261029_111500_Robot_36_partial_180of240.png", 240)
        self.write_image("not_an_sstv_image.png")
        panel = SSTVPanel()
        try:
            panel.set_save_directory(self.tmp.name)
            panel.show()
            settle()
            lst = panel._history_list
            self.assertEqual(
                [lst.item(i).text() for i in range(lst.count())],
                [
                    "Oct 29 10:15 · Martin 1",
                    "Oct 29 11:15 · Robot 36 (partial, 180/240 lines)",
                ],
            )
            lst.setCurrentRow(0)
            settle()
            self.assertTrue(panel._image_display.has_image())
            self.assertEqual(panel._mode_label.text(), "Martin 1")
            self.assertEqual(panel._resolution_label.text(), "320 × 256")
            self.assertTrue(panel._save_btn.isEnabled())
            # Delete removes the file and the entry.
            self.assertTrue(panel._delete_entry(0, confirm=False))
            self.assertFalse(
                (Path(self.tmp.name) / "sstv_20261029_101530_Martin_1.png").exists()
            )
            self.assertEqual(lst.count(), 1)
        finally:
            panel.deleteLater()

    def test_received_images_come_back_after_a_restart(self):
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(np.concatenate([audio, np.zeros(int(0.5 * FS), np.float32)]))
        self.assertEqual(len(self.history_texts()), 1)
        again = SSTVPanel()
        try:
            again.set_save_directory(self.tmp.name)
            again.show()
            settle()
            self.assertEqual(again._history_list.count(), 1)
            self.assertIn("Robot 36", again._history_list.item(0).text())
        finally:
            again.deleteLater()

    def test_file_names_parse(self):
        from sdr_module.ham.gui.sstv_panel import parse_image_filename

        entry = parse_image_filename(Path("sstv_20261030_123456_Robot_36_2.png"))
        self.assertEqual((entry["mode"], entry["height"]), ("Robot 36", 240))
        entry = parse_image_filename(
            Path("sstv_20261030_123456_PD_120_partial_180of496.png")
        )
        self.assertEqual(
            (entry["mode"], entry["partial_lines"], entry["height"]),
            ("PD 120", 180, 496),
        )
        entry = parse_image_filename(Path("sstv_20261030_123456.png"))
        self.assertEqual(entry["mode"], "SSTV")  # saved by hand
        self.assertIsNone(parse_image_filename(Path("holiday.png")))

    def test_history_label_format(self):
        from sdr_module.ham.gui.sstv_panel import history_label

        info = {
            "timestamp": "20261030_123456",
            "mode": "Martin 1",
            "height": 256,
            "partial_lines": 180,
        }
        self.assertEqual(
            history_label(info, today="20261030"),
            "12:34 · Martin 1 (partial, 180/256 lines)",
        )
        info["partial_lines"] = None
        self.assertEqual(
            history_label(info, today="20261031"), "Oct 30 12:34 · Martin 1"
        )


class TestToneMeterAndHints(_PanelCase):
    def test_tone_meter_follows_what_is_heard(self):
        meter = self.panel._tone_meter
        self.assertEqual(self.panel._tone_label.text(), "Not listening")
        self.panel._start_btn.click()
        t = np.arange(int(0.5 * FS)) / FS
        self.feed(np.sin(2 * np.pi * 1900 * t).astype(np.float32))
        self.panel._update_status()
        self.assertEqual(self.panel._tone_label.text(), "Hearing 1900 Hz leader…")
        self.assertAlmostEqual(meter.tone(), 1900, delta=10)
        rng = np.random.default_rng(1)
        self.feed((0.3 * rng.standard_normal(int(0.5 * FS))).astype(np.float32))
        self.panel._update_status()
        self.assertEqual(self.panel._tone_label.text(), "Noise only, no SSTV tones")
        self.assertIsNone(meter.tone())
        self.panel._last_audio_time -= 10
        self.panel._update_status()
        self.assertEqual(self.panel._tone_label.text(), "No audio")
        self.assertEqual(meter.accessibleName(), "Tone meter")
        self.assertIn("1200 Hz", meter.toolTip())

    def test_tone_meter_paints_in_both_themes(self):
        from sdr_module.gui import themes

        meter = self.panel._tone_meter
        meter.set_reading(2100.0)
        saved = themes._current_theme
        try:
            for name in ("dark", "light"):
                themes._current_theme = name
                self.assertFalse(meter.grab().isNull())
        finally:
            themes._current_theme = saved

    def test_fm_hint_and_buttons(self):
        switch = QSignalSpy(self.panel.switch_to_fm_requested)
        tune = QSignalSpy(self.panel.tune_iss_requested)
        self.panel.show_fm_needed("AM")
        settle()
        self.assertTrue(self.panel._fm_callout.isVisible())
        self.assertIn("AM", self.panel._fm_label.text())
        self.panel._switch_fm_btn.click()
        self.assertEqual(len(switch), 1)
        self.panel.show_fm_needed(None)
        self.assertFalse(self.panel._fm_callout.isVisible())
        self.panel._tune_iss_btn.click()
        self.assertEqual(len(tune), 1)
        self.assertIn("145.800 MHz", self.panel._tune_iss_btn.text())


class TestShortTab(unittest.TestCase):
    def test_picture_fits_a_short_tab_and_is_revealed(self):
        require_pyqt6(self)
        from PyQt6.QtCore import QPoint, QRect
        from PyQt6.QtWidgets import QFrame, QScrollArea

        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        panel = SSTVPanel()
        panel.set_save_directory(tmp.name)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(panel)
        area.resize(376, 200)
        area.show()
        self.addCleanup(area.deleteLater)
        settle(4)
        panel._start_btn.click()
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        for i in range(0, len(audio) + 24000, GUI_BLOCK):
            panel.process_audio(audio[i : i + GUI_BLOCK], FS)
        settle(4)
        panel._start_btn.click()
        viewport = area.viewport()
        display = panel._image_display
        self.assertLessEqual(display.height(), viewport.height())
        rect = QRect(display.mapTo(viewport, QPoint(0, 0)), display.size())
        self.assertGreaterEqual(rect.top(), 0)
        self.assertLessEqual(rect.bottom(), viewport.height())
        self.assertTrue(display.has_image())


class TestSSTVInTheWindow(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.panel = self.win._sstv_panel
        self.panel.set_save_directory(self.tmp.name)
        self.win._audio.write = lambda audio: None

    def tearDown(self):
        if HAS_PYQT6 and getattr(self, "panel", None) and self.panel.is_receiving():
            self.panel._start_btn.click()
        super().tearDown()
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_tune_to_iss(self):
        self.win._stop_acquisition()
        self.win._set_demod_mode("AM")
        self.panel._tune_iss_btn.click()
        panel = self.win._control_panel
        self.assertAlmostEqual(self.win._current_frequency(), 145.8e6)
        self.assertEqual(panel._demod_combo.currentText(), "FM")
        self.assertEqual(panel.get_fm_deviation_text(), "5 kHz")
        self.assertEqual(panel._bw_combo.currentText(), "25 kHz")
        self.assertTrue(self.win._is_running)
        self.assertTrue(self.panel.is_receiving())
        self.assertIn("ISS", self.win._message_label.text())

    def test_tune_to_iss_from_the_tools_menu(self):
        actions = {}
        for top in self.win.menuBar().actions():
            if top.menu() is not None:
                for a in top.menu().actions():
                    actions[a.text().replace("&", "")] = a
        action = actions["SSTV: Tune to ISS (145.800 MHz)"]
        self.assertTrue(action.statusTip())
        action.trigger()
        self.assertAlmostEqual(self.win._current_frequency(), 145.8e6)
        self.assertTrue(self.panel.is_receiving())
        self.assertIs(self.win._ham_tabs.currentWidget(), self.win._panel_pages["sstv"])

    def test_tune_to_iss_without_a_device(self):
        self.win._disconnect_device()
        self.panel._tune_iss_btn.click()
        self.assertAlmostEqual(self.win._current_frequency(), 145.8e6)
        self.assertIn("No receiver", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")

    def test_start_decoder_in_am_offers_switch_to_fm(self):
        self.win._set_demod_mode("AM")
        self.panel._start_btn.click()
        settle()
        # (isHidden: the SSTV tab need not be the one on screen.)
        self.assertFalse(self.panel._fm_callout.isHidden())
        self.assertIn("FM", self.win._message_label.text())  # status bar too
        self.panel._switch_fm_btn.click()
        settle()
        self.assertEqual(self.win._control_panel._demod_combo.currentText(), "FM")
        self.assertEqual(self.win._control_panel.get_fm_deviation_text(), "5 kHz")
        self.assertTrue(self.panel._fm_callout.isHidden())
        # Leaving FM while listening brings the hint back.
        self.win._set_demod_mode("USB")
        settle()
        self.assertFalse(self.panel._fm_callout.isHidden())

    def test_mode_choice_is_remembered(self):
        self.panel.set_mode_choice("PD 120")
        self.assertEqual(self.store.values.get("ham/sstv_mode"), "PD 120")
        self.panel.set_mode_choice("Auto (VIS)")
        self.store.values["ham/sstv_mode"] = "Scottie 1"
        self.win._restore_sstv_and_smeter()
        self.assertEqual(self.panel.mode_choice(), "Scottie 1")

    def test_demo_test_card_decodes(self):
        """hm:GAP-10: Tune to ISS on the demo device, and the next test card
        (Robot 36, every 75 s, the first 8 s into the demo) arrives through
        the window's channel filter, FM demodulator and audio path."""
        self.panel._tune_iss_btn.click()
        device = self.win._device
        rate = device.sample_rate
        device._stream_pos = int(7.0 * rate)  # just before the first image
        device._stream_epoch = None
        block = int(rate / 30)
        for _ in range(int(40 * 30)):
            self.win._process_block(device.read_samples(block))
            if self.panel._history_list.count():
                break
        self.assertEqual(self.panel._history_list.count(), 1)
        self.assertRegex(self.panel._history_list.item(0).text(), r"· Robot 36$")
        image = self.panel._viewer.get_current_image()
        card = sstv_test_card(320, 240, "SDR MODULE")
        self.assertGreater(correlation(image, card), 0.97)
        self.assertIn("Image received", self.panel._status_label.text())


class TestSSTVThroughFM(_WindowTestCase):
    """hm:HAM-01 end to end: an SSTV image FM-modulated at 5 kHz deviation,
    with noise (20 dB carrier to noise in the 25 kHz channel) and the
    sender's clock 0.05% fast, through _process_block, the receiver chain
    and the panel."""

    def test_robot36_through_the_window(self):
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        panel = self.win._sstv_panel
        panel.set_save_directory(tmp.name)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        panel._start_btn.click()
        self.addCleanup(lambda: panel.is_receiving() and panel._start_btn.click())
        card = sstv_test_card(320, 240)
        audio = encode_sstv(card, "Robot 36", FS * 1.0005)
        audio = np.concatenate([np.zeros(int(0.3 * FS), np.float32), audio])
        audio = np.concatenate([audio, np.zeros(int(0.6 * FS), np.float32)])
        rate = self.win._device_sample_rate()
        up = int(round(rate / FS))
        amplitude = 0.1
        # 20 dB C/N in 25 kHz: noise power over the span = signal / 100 *
        # span / 25 kHz.
        sigma = np.sqrt(amplitude**2 / 100.0 * rate / 25e3 / 2.0)
        rng = np.random.default_rng(4)
        pool = (
            (rng.standard_normal(2 * (1 << 20)) * sigma)
            .astype(np.float32)
            .view(np.complex64)
        )
        phase0 = 0.0
        step = 2 * np.pi * 5000.0 / rate
        ramp = np.arange(1, up + 1, dtype=np.float64) / up
        last = 0.0
        started = time.perf_counter()
        for first in range(0, len(audio), GUI_BLOCK):
            chunk = audio[first : first + GUI_BLOCK].astype(np.float64)
            # Linear interpolation of the audio to the device rate.
            prev = np.concatenate(([last], chunk[:-1]))
            last = float(chunk[-1])
            fine = (prev[:, None] + (chunk - prev)[:, None] * ramp[None, :]).ravel()
            phase = phase0 + step * np.cumsum(fine)
            phase0 = float(phase[-1])
            iq = (amplitude * np.exp(1j * phase)).astype(np.complex64)
            offset = int(rng.integers(0, len(pool) - len(iq)))
            iq += pool[offset : offset + len(iq)]
            self.win._process_block(iq)
        elapsed = time.perf_counter() - started
        self.assertEqual(panel._history_list.count(), 1, f"took {elapsed:.1f} s")
        self.assertEqual(panel._mode_label.text(), "Robot 36")
        image = panel._viewer.get_current_image()
        self.assertGreater(correlation(image, card), 0.97)
