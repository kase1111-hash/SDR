#!/usr/bin/env python3
"""
Integration tests for the main window's receiver, recording, audio, preset,
keyboard-focus and error-handling behavior (the "main" UI review fixes).

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration_main.py``
"""

import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMessageBox


_saved_style = None


def setUpModule():  # noqa: N802 - unittest hook
    """Wiring tests: run on the unstyled app (much faster to build)."""
    global _saved_style
    if not HAS_PYQT6:
        return
    qapp = QApplication.instance()
    _saved_style = (qapp.styleSheet(), qapp.palette())
    qapp.setStyleSheet("")


def tearDownModule():  # noqa: N802 - unittest hook
    if not HAS_PYQT6 or _saved_style is None:
        return
    qapp = QApplication.instance()
    qapp.setStyleSheet(_saved_style[0])
    qapp.setPalette(_saved_style[1])


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


def _flush_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _answer_message_box(button_text):
    """A QMessageBox.exec replacement that clicks the button ``button_text``."""

    def fake_exec(box):
        for button in box.buttons():
            if button.text().replace("&", "") == button_text:
                button.click()
                return 0
        raise AssertionError(f"no {button_text!r} button")

    return fake_exec


def _action(window, shortcut):
    return next(
        a for a in window.findChildren(QAction) if a.shortcut().toString() == shortcut
    )


class TestReceiverChain(unittest.TestCase):
    """The channel filter and demodulators, without a window."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window

    def test_blocks_give_the_same_output_as_one_long_block(self):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal(9000) + 1j * rng.standard_normal(9000)).astype(
            np.complex64
        )
        for mode, bw in (("FM", 25e3), ("FM", 200e3), ("AM", 25e3), ("USB", 10e3)):
            one = self.mw._ReceiverChain(2.4e6, mode, bw).channelize(x)
            chain = self.mw._ReceiverChain(2.4e6, mode, bw)
            parts = np.concatenate(
                [chain.channelize(x[i : i + 700]) for i in range(0, len(x), 700)]
            )
            np.testing.assert_allclose(one, parts, atol=1e-4, err_msg=mode)

    def test_channel_filter_rejects_a_neighbour(self):
        t = np.arange(48000) / 2.4e6
        chain = self.mw._ReceiverChain(2.4e6, "FM", 25e3)
        inside = chain.channelize(np.exp(2j * np.pi * 8e3 * t).astype(np.complex64))
        chain = self.mw._ReceiverChain(2.4e6, "FM", 25e3)
        outside = chain.channelize(np.exp(2j * np.pi * 75e3 * t).astype(np.complex64))
        self.assertGreater(np.mean(np.abs(inside[20:])), 0.9)
        self.assertLess(np.mean(np.abs(outside[20:])), 1e-3)

    def test_sidebands_and_cw_beat_note(self):
        t = np.arange(240000) / 2.4e6

        def loudest(mode, iq):
            chain = self.mw._ReceiverChain(2.4e6, mode, 10e3)
            audio = chain.demodulate(chain.channelize(iq), 5e3)[2000:]
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / chain.audio_rate)
            return freqs[np.argmax(spectrum)], float(np.sqrt(np.mean(audio**2)))

        upper = np.exp(2j * np.pi * 1000 * t).astype(np.complex64)
        freq, rms = loudest("USB", upper)
        self.assertAlmostEqual(freq, 1000, delta=30)
        self.assertGreater(rms, 0.2)
        self.assertLess(loudest("LSB", upper)[1], 0.05)  # wrong sideband
        freq, rms = loudest("CW", np.ones(len(t), dtype=np.complex64))
        self.assertAlmostEqual(freq, 700, delta=30)  # carrier -> 700 Hz tone

    def test_weaker_demo_station_is_heard_next_to_a_stronger_one(self):
        from sdr_module.gui.device_dialog import MockDevice

        device = MockDevice()
        device.set_frequency(99.5e6)  # 100.1 MHz is 16 dB stronger
        device.start_rx()
        chain = self.mw._ReceiverChain(2.4e6, "FM", 200e3)
        audio = np.concatenate(
            [
                chain.demodulate(chain.channelize(device.read_samples(2048)), 75e3)
                for _ in range(30)
            ]
        )
        self.assertLess(np.mean(np.abs(audio) > 0.98), 0.02)  # was 99.6%
        self.assertGreater(np.std(audio), 0.1)


class TestChannelMeasurements(_WindowTestCase):
    demo = True

    def _settle_on(self, freq, bandwidth="200 kHz", mode="FM"):
        self.win.set_frequency(freq)
        self.win._set_demod_mode(mode, "75 kHz", bandwidth)
        for _ in range(8):
            self.win._update_display()

    def test_level_measures_the_tuned_channel(self):
        self._settle_on(100.5e6)  # empty channel; 100.1 MHz is in view
        empty = self.win._last_peak_db
        self._settle_on(100.1e6)
        station = self.win._last_peak_db
        self.assertLess(empty, -60)
        self.assertGreater(station, -40)
        self.assertIn("channel", self.win._level_label.toolTip())

    def test_squelch_closes_on_an_empty_channel(self):
        spoken = []
        self.win._audio.write = spoken.append
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-60)
        self._settle_on(100.5e6)
        self.assertEqual(spoken, [])
        self._settle_on(100.1e6)
        self.assertTrue(spoken)

    def test_s_meter_reads_the_channel(self):
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")

        def reading(freq):
            self._settle_on(freq, "25 kHz")
            return self.win._signal_meter_panel._last_reading.power_dbm

        station, empty = reading(100.1e6), reading(100.9e6)
        self.assertGreater(station - empty, 30)  # was the same S9+30 on both

    def test_rtl_bandwidth_does_not_change_its_sample_rate(self):
        from sdr_module.devices.rtlsdr import RTLSDRDevice

        class FakeRtl(RTLSDRDevice):
            def __init__(self):
                super().__init__()
                self._state.sample_rate = 2.4e6
                self.calls = []

            def set_bandwidth(self, bw_hz):
                self.calls.append(bw_hz)
                self._state.sample_rate = bw_hz
                return True

        self.win._release_device()
        self.win._device = device = FakeRtl()
        self.win._refresh_state_ui()
        self.win._control_panel._bw_combo.setCurrentText("25 kHz")
        self.assertEqual(device.calls, [])
        self.assertEqual(self.win._rate_label.text(), "2.4 MS/s")
        self.win._device = None

    def test_bandwidth_errors_are_reported_and_rate_changes_followed(self):
        class Device:
            sample_rate = 2.4e6

            def set_bandwidth(self, bw_hz):
                Device.sample_rate = 1.2e6
                raise RuntimeError("filter busy")

        self.win._release_device()
        self.win._device = Device()
        self.win._control_panel._bw_combo.setCurrentText("50 kHz")
        self.assertIn("filter busy", self.win._message_label.text())
        self.assertEqual(self.win._rate_label.text(), "1.2 MS/s")
        self.win._device = None
        self.win._refresh_state_ui()

    def test_no_audio_from_a_device_that_is_not_real_time(self):
        spoken = []
        self.win._audio.write = spoken.append
        self.win._audio_enabled = True
        self.win._control_panel.set_squelch_db(-120)
        now = time.monotonic()
        self.win._rx_history.extend((now - 1.5 + i * 0.1, 2048) for i in range(15))
        self.win.set_frequency(100.3e6)
        self.win._route_audio(np.ones(64, dtype=np.complex64), 0.0)
        self.assertEqual(spoken, [])
        self.assertIn("demo device", self.win._message_label.text())


class TestRecordingProtection(_WindowTestCase):
    demo = True

    def _record(self, blocks=2):
        self.win._record_button.click()
        for _ in range(blocks):
            self.win._update_display()
        self.win._record_button.click()

    def _real_confirm(self, answer):
        """Use the real prompt, answered with ``answer``."""
        from sdr_module.gui.main_window import SDRMainWindow

        del self.win._confirm_discard_recording  # the harness stub
        patcher = mock.patch.object(QMessageBox, "exec", _answer_message_box(answer))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(
            setattr, self.win, "_confirm_discard_recording", lambda *_a: True
        )
        return SDRMainWindow._confirm_discard_recording

    def test_recording_again_asks_and_cancel_keeps_the_take(self):
        self._record()
        count = sum(len(b) for b in self.win._samples_buffer)
        self._real_confirm("Cancel")
        self.win._record_button.click()
        self.assertFalse(self.win._recording)
        self.assertFalse(self.win._record_button.isChecked())
        self.assertFalse(self.win._control_panel._record_btn.isChecked())
        self.assertEqual(sum(len(b) for b in self.win._samples_buffer), count)

    def test_dont_save_discards_and_clears_the_save_hint(self):
        self._record()
        self.assertIn("Save Recording", self.win._message_label.text())
        self._real_confirm("Don't Save")
        self.win._stop_acquisition()  # nothing new is captured
        self.win._record_button.click()
        self.assertTrue(self.win._recording)
        self.assertEqual(self.win._samples_buffer, [])
        self.assertNotIn("Save Recording", self.win._message_label.text())
        self.win._record_button.click()

    def test_armed_recording_that_captures_nothing_keeps_the_previous_one(self):
        self._record()
        self.win._buffer_unsaved = False  # e.g. it was saved
        before = list(self.win._samples_buffer)
        self.win._stop_acquisition()
        self.win._record_button.click()  # armed, nothing arrives
        self.win._record_button.click()
        self.assertEqual(len(self.win._samples_buffer), len(before))
        self.assertIs(self.win._samples_buffer[0], before[0])

    def test_closing_with_an_unsaved_recording_can_be_cancelled(self):
        self._record()
        self._real_confirm("Cancel")
        self.win.close()
        self.assertTrue(self.win.isVisible())
        self.assertTrue(self.win._samples_buffer)

    def test_saved_recording_does_not_ask(self):
        self._record()
        folder = tempfile.mkdtemp(prefix="sdr-main-")
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(os.path.join(folder, "take.cf32"), ""),
        ):
            self.assertTrue(self.win._save_recording())
        confirm = self._real_confirm("Cancel")  # would cancel if it asked
        self.assertTrue(confirm(self.win, "closing"))

    def test_save_uses_the_capture_frequency_and_rate(self):
        from sdr_module.dsp.recording import load_iq_file

        self.win.set_frequency(100e6)
        self._record()
        self.win.set_frequency(433.92e6)  # retuned after recording
        folder = tempfile.mkdtemp(prefix="sdr-main-")
        seen = {}

        def fake_save(parent, caption, directory, filters, initial):
            seen["suggested"] = directory
            return os.path.join(folder, "take.sigmf-data"), ""

        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            side_effect=fake_save,
        ):
            self.win._save_recording()
        self.assertTrue(seen["suggested"].startswith("iq_100.000MHz_"))
        _samples, meta = load_iq_file(os.path.join(folder, "take.sigmf-data"))
        self.assertAlmostEqual(meta.center_frequency, 100e6)

        # An imported file keeps its own sample rate when converted.
        with (
            mock.patch(
                "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
                return_value=(os.path.join(folder, "take.sigmf-data"), ""),
            ),
            mock.patch.object(QMessageBox, "information"),
            mock.patch(
                "sdr_module.dsp.recording.load_iq_file",
                return_value=(
                    np.zeros(100, np.complex64),
                    mock.Mock(sample_rate=2.0e6, center_frequency=145.8e6),
                ),
            ),
        ):
            self.win._open_recording()
        self.assertFalse(self.win._buffer_unsaved)
        self.win.set_sample_rate(2.4e6)
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(os.path.join(folder, "converted.sigmf-data"), ""),
        ):
            self.win._save_recording()
        _samples, meta = load_iq_file(os.path.join(folder, "converted.sigmf-data"))
        self.assertAlmostEqual(meta.sample_rate, 2.0e6)
        self.assertAlmostEqual(meta.center_frequency, 145.8e6)

    def test_retuning_while_recording_warns(self):
        self.win._record_button.click()
        self.win._update_display()
        self.win.set_frequency(101e6)
        self.win._update_display()
        self.assertIn("Retuned while recording", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        self.win._record_button.click()


class TestDecodedMessages(_WindowTestCase):
    def test_invalid_messages_are_marked_and_empty_failures_dropped(self):
        from sdr_module.dsp.protocols import AX25Frame, POCSAGMessage, ProtocolType

        panel = self.win._decoder_panel
        self.win._push_decoded_message(
            AX25Frame(
                protocol=ProtocolType.AX25,
                timestamp=0,
                raw_bits=b"",
                valid=False,
                error_message="CRC mismatch",
            )
        )
        self.assertEqual(panel._table.rowCount(), 0)
        self.win._push_decoded_message(
            POCSAGMessage(
                protocol=ProtocolType.POCSAG,
                timestamp=0,
                raw_bits=b"\x01",
                valid=False,
                error_message="parity",
                address=1234,
                content="HELLO",
            )
        )
        self.win._push_decoded_message(
            AX25Frame(
                protocol=ProtocolType.AX25,
                timestamp=0,
                raw_bits=b"",
                valid=True,
                source="W1AW",
                destination="APRS",
                info="hi",
            )
        )
        self.assertEqual(panel._table.rowCount(), 2)
        self.assertEqual(panel._invalid, 1)
        self.assertEqual(panel._messages[0]["protocol"], "POCSAG")
        self.assertFalse(panel._messages[0]["valid"])
        self.assertIn("parity", panel._messages[0]["content"])
        self.assertEqual(panel._messages[1]["protocol"], "AX.25/APRS")
        self.assertEqual(panel._messages[1]["address"], "W1AW>APRS")


class TestAudioOutput(_WindowTestCase):
    def test_no_output_device_turns_audio_back_off_and_says_why(self):
        with mock.patch.object(
            type(self.win),
            "_audio_output_problem",
            return_value="No audio output device",
        ):
            self.win._audio_action.setChecked(True)
        self.assertFalse(self.win._audio_enabled)
        self.assertFalse(self.win._audio_action.isChecked())
        self.assertFalse(self.win._audio_button.isChecked())
        self.assertIn("No audio output device", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        self.assertNotEqual(self.store.values.get("audio_enabled"), False)
        text, tone = self.win._audio_status()
        self.assertTrue(text.startswith("Unavailable"))
        self.assertEqual(tone, "warning")

    def test_toolbar_button_and_menu_item_are_one_switch(self):
        with (
            mock.patch.object(
                type(self.win), "_audio_output_problem", return_value=None
            ),
            mock.patch.object(self.win._audio, "start", return_value=True),
            mock.patch.object(self.win._audio, "_io", object(), create=True),
        ):
            self.win._set_demod_mode("FM")
            self.win._audio_button.click()  # on
            self.assertTrue(self.win._audio_enabled)
            self.assertTrue(self.win._audio_action.isChecked())
            self.assertEqual(self.win._audio_button.text().strip(), "Audio")
            self.win._audio_action.trigger()  # off from the menu
        self.assertFalse(self.win._audio_enabled)
        self.assertFalse(self.win._audio_button.isChecked())
        self.assertEqual(self.win._audio_button.text().strip(), "Muted")
        self.assertIs(self.store.values["audio_enabled"], False)

    def test_audio_is_on_by_default_and_volume_is_remembered(self):
        self.assertTrue(self.win._settings.get_bool("audio_enabled", True))
        with mock.patch.object(self.win._audio, "set_volume") as set_volume:
            self.win._volume_slider.setValue(35)
        set_volume.assert_called_with(0.35)
        self.assertEqual(self.store.values["audio_volume"], 35)
        self.assertEqual(self.win._volume_slider.accessibleName(), "Volume")


class TestPresetsAndDefaults(_WindowTestCase):
    def test_band_presets_set_mode_deviation_and_bandwidth(self):
        from sdr_module.gui.main_window import BAND_PRESETS

        panel = self.win._control_panel
        for preset in BAND_PRESETS:
            panel._bw_combo.setCurrentText("1 MHz")
            self.win._apply_band_preset(preset.frequency_hz, preset.mode, preset.label)
            self.assertEqual(panel._demod_combo.currentText(), preset.mode)
            self.assertEqual(panel._bw_combo.currentText(), preset.bandwidth)
            if preset.fm_deviation:
                self.assertEqual(panel._fm_dev_combo.currentText(), preset.fm_deviation)
            self.assertIn("bandwidth", self.win._message_label.text())
            self.assertNotIn("&", self.win._message_label.text())
        # ISM matches the Control Panel presets: raw I/Q.
        ism = [p for p in BAND_PRESETS if "ISM" in p.label]
        self.assertTrue(all(p.mode == "None (I/Q)" for p in ism))

    def test_band_preset_menu_has_unique_mnemonics_and_no_repeated_numbers(self):
        menu = next(
            a.menu()
            for a in self.win.findChildren(QAction)
            if a.menu() is not None and a.text() == "Band &Presets"
        )
        labels = [a.text().split("\t")[0] for a in menu.actions()]
        keys = [label[label.index("&") + 1].lower() for label in labels]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(any("(" in label for label in labels))

    def test_panels_menu_has_mnemonics(self):
        texts = [_action(self.win, f"Ctrl+{i}").text() for i in range(1, 3)]
        self.assertEqual(texts, ["&Decoder", "&Bookmarks"])

    def test_panel_apply_preset_is_confirmed(self):
        panel = self.win._control_panel
        panel._category_combo.setCurrentText("Weather")
        panel._preset_combo.setCurrentText("NOAA Weather 1")
        self.win._clear_status_message()
        panel._apply_preset_btn.click()
        self.assertIn("NOAA Weather 1", self.win._message_label.text())
        self.assertIn("162.550 MHz", self.win._message_label.text())

    def test_fresh_start_is_fm_with_broadcast_deviation(self):
        panel = self.win._control_panel
        self.assertEqual(panel._demod_combo.currentText(), "FM")
        self.assertEqual(panel._fm_dev_combo.currentText(), "75 kHz")

    def test_deviation_bandwidth_and_format_are_restored(self):
        panel = self.win._control_panel
        panel._fm_dev_combo.setCurrentText("12.5 kHz")
        panel._bw_combo.setCurrentText("25 kHz")
        sigmf = next(
            panel._format_combo.itemText(i)
            for i in range(panel._format_combo.count())
            if "SigMF" in panel._format_combo.itemText(i)
        )
        panel._format_combo.setCurrentText(sigmf)
        self.win._show_panel("bookmarks")
        self.win._persist_state()
        other = self.mw.SDRMainWindow()
        other._confirm_discard_recording = lambda *_a: True
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        restored = other._control_panel
        self.assertEqual(restored._fm_dev_combo.currentText(), "12.5 kHz")
        self.assertEqual(restored._bw_combo.currentText(), "25 kHz")
        self.assertEqual(restored._format_combo.currentText(), sigmf)
        self.assertEqual(other._current_panel_key(), "bookmarks")

    def test_demo_mode_leaves_the_no_audio_mode(self):
        self.win._set_demod_mode("None (I/Q)")
        self.win.set_frequency(100.3e6)
        self.win._start_demo_mode()
        panel = self.win._control_panel
        self.assertEqual(panel._demod_combo.currentText(), "FM")
        self.assertEqual(panel._fm_dev_combo.currentText(), "75 kHz")
        self.assertIn("Mode set to FM", self.win._message_label.text())
        self.assertNotIn("simulated", self.win._message_label.text())
        self.assertEqual(
            self.win._device_display_name(), "Demo Device (simulated signals)"
        )

    def test_welcome_screen_can_be_reopened(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        shown = []
        with mock.patch.object(
            FirstRunWizard, "exec", lambda wiz: shown.append(wiz) or 0
        ):
            next(
                a for a in self.win.findChildren(QAction) if "Welcome" in a.text()
            ).trigger()
        _flush_deletes()
        self.assertEqual(len(shown), 1)


class TestSmallFixes(_WindowTestCase):
    def test_screenshot_without_extension_is_saved_as_png(self):
        folder = tempfile.mkdtemp(prefix="sdr-main-")
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(os.path.join(folder, "shot"), "PNG (*.png)"),
        ):
            self.win._save_screenshot()
        self.assertTrue(os.path.exists(os.path.join(folder, "shot.png")))
        self.assertIn("shot.png", self.win._message_label.text())

    def test_start_then_connect_says_receiving(self):
        from sdr_module.gui.device_dialog import MockDevice

        self.win._prompt_no_device = lambda: "connect"

        def fake_dialog(start_after=False):
            self.win._device = MockDevice()
            self.win._refresh_state_ui()
            if start_after:
                self.win._start_acquisition()
                self.win._show_status_message("Receiving from Demo Device", "success")
            return True

        with mock.patch.object(self.win, "_show_device_dialog", fake_dialog):
            self.win._start_acquisition()
        self.assertTrue(self.win._is_running)
        self.assertNotIn("Press Start", self.win._message_label.text())

    def test_no_device_prompt_recommends_connect(self):
        seen = {}

        def fake_exec(box):
            seen.update(
                {b.text().replace("&", ""): b.property("role") for b in box.buttons()}
            )
            return 0

        with mock.patch.object(QMessageBox, "exec", fake_exec):
            self.win._prompt_no_device()
        self.assertEqual(seen["Connect..."], "primary")

    def test_info_quick_tips_are_keycap_rows(self):
        from sdr_module.gui.main_window import InfoPanel

        caps = [
            label.text()
            for label in self.win._info_panel.findChildren(type(self.win._freq_label))
            if label.property("role") == "keycap"
        ]
        self.assertEqual(caps, [keys for keys, _text in InfoPanel.TIPS])


class TestKeyboardFocus(_WindowTestCase):
    def test_plots_are_reachable_with_tab(self):
        for plot in (self.win._spectrum, self.win._waterfall):
            self.assertEqual(plot.focusPolicy(), Qt.FocusPolicy.StrongFocus)

    def test_f6_and_esc_return_to_the_spectrum(self):
        slider = self.win._control_panel._gain_slider
        slider.setFocus()
        self.settle()
        _action(self.win, "F6").trigger()
        self.settle()
        self.assertTrue(self.win._spectrum.hasFocus())
        slider.setFocus()
        self.settle()
        QTest.keyClick(slider, Qt.Key.Key_Escape)
        self.settle()
        self.assertTrue(self.win._spectrum.hasFocus())

    def test_ctrl_l_then_enter_tunes_and_returns_to_the_plots(self):
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "146.52")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertAlmostEqual(self.freq(), 146.52e6)
        self.assertTrue(self.win._spectrum.hasFocus())

    def test_panel_shortcut_moves_focus_into_the_panel(self):
        _action(self.win, "Ctrl+2").trigger()
        self.settle()
        page = self.win._panel_pages["bookmarks"]
        self.assertTrue(page.isAncestorOf(QApplication.focusWidget()))


class TestExceptHook(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.app import SDRApplication

        self.store = _FakeSettings()
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", lambda: self.store)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.win = main_window.SDRMainWindow()
        for timer in (
            self.win._display_timer,
            self.win._status_timer,
            self.win._hotplug_timer,
        ):
            timer.stop()
        self.addCleanup(self.win.deleteLater)
        self.addCleanup(self.win.close)
        self.launcher = SDRApplication(args=["sdr"])
        self.launcher._app = QApplication.instance()
        self.launcher._main_window = self.win

    @staticmethod
    def _error():
        try:
            raise ValueError("cannot convert float NaN to integer")
        except ValueError:
            return sys.exc_info()

    def test_error_is_logged_and_reported_instead_of_aborting(self):
        with self.assertLogs("sdr_module.gui.app", level=logging.ERROR):
            self.launcher._handle_exception(*self._error())
        self.assertIn("Internal error", self.win._message_label.text())
        # The same error again right away (e.g. every frame) isn't re-logged.
        with self.assertNoLogs("sdr_module.gui.app", level=logging.ERROR):
            self.launcher._handle_exception(*self._error())

    def test_worker_thread_errors_are_logged_only(self):
        self.win._clear_status_message()
        with self.assertLogs("sdr_module.gui.app", level=logging.ERROR):
            thread = threading.Thread(
                target=lambda: self.launcher._handle_exception(*self._error())
            )
            thread.start()
            thread.join()
        self.assertEqual(self.win._message_label.text(), "")

    def test_run_installs_and_restores_the_hook(self):
        seen = []
        before = sys.excepthook
        with mock.patch.object(
            self.launcher,
            "_run",
            side_effect=lambda settings: seen.append(sys.excepthook) or 0,
        ):
            self.assertEqual(self.launcher.run({}), 0)
        self.assertEqual(seen, [self.launcher._handle_exception])
        self.assertIs(sys.excepthook, before)


if __name__ == "__main__":
    unittest.main()
