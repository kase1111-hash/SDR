#!/usr/bin/env python3
"""
Mode-appropriate filter widths and FM deviation.

* ux:UX-B02 / rx:UX-05: picking a mode applies its width (and FM's
  deviation), remembered per mode once changed.
* rx:UX-09: the FM deviation items say what they are for, and clipping FM
  audio is reported in the status bar.
* rx:RX-11 / ux:UX-B10 / rx:G-09: the Bandwidth choices depend on the mode
  (SSB and CW filter widths), the Info tab shows the real passband, and a
  CW pitch control appears in CW.
"""

import json
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    fm,
    noise,
    pretend_audio_output,
)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestModeWidths(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel

    def pick(self, mode):
        """Choose a mode in the Mode list, as a user does."""
        self.panel._demod_combo.setCurrentText(mode)

    def width(self):
        return self.panel._bw_combo.currentText()

    def choices(self):
        return self.panel.bandwidth_choices()

    def test_each_mode_offers_its_own_widths(self):
        from sdr_module.gui.control_panel import MODE_BANDWIDTHS

        self.pick("USB")
        self.assertEqual(
            self.choices(), ("1.8 kHz", "2.4 kHz", "2.7 kHz", "3.0 kHz", "3.6 kHz")
        )
        self.pick("CW")
        self.assertEqual(self.choices(), ("100 Hz", "250 Hz", "500 Hz", "1 kHz"))
        self.pick("AM")
        self.assertEqual(self.choices()[0], "5 kHz")
        self.assertEqual(self.choices()[-1], "15 kHz")
        self.pick("FM")
        self.assertEqual(self.choices()[0], "12.5 kHz")
        self.assertEqual(self.choices()[-1], "200 kHz")
        self.pick("None (I/Q)")
        self.assertEqual(self.choices(), MODE_BANDWIDTHS["None (I/Q)"])
        self.assertIn("2.4 MHz", self.choices())

    def test_picking_am_after_broadcast_fm_gets_an_am_channel(self):
        # UX-B02: AM kept 200 kHz and played a station 100 kHz away.
        self.win.set_frequency(125.0e6)
        self.assertEqual(self.width(), "200 kHz")
        self.pick("AM")
        self.assertEqual(self.width(), "10 kHz")
        chain = self.win._receiver_chain()
        self.assertEqual(chain.band_edges(), (-5000.0, 5000.0))
        self.assertIn("AM: 10 kHz", self.win._message_label.text())

    def test_picking_fm_in_the_broadcast_band_gets_broadcast_settings(self):
        self.win._apply_band_preset(125.0e6, "AM", "&Airband AM")
        self.win.set_frequency(100.1e6)
        self.pick("FM")
        self.assertEqual(self.width(), "200 kHz")
        self.assertEqual(self.panel.get_fm_deviation_text(), "75 kHz")
        # Outside the broadcast band FM is narrow voice.
        self.win.set_frequency(146.52e6)
        self.pick("AM")
        self.pick("FM")
        self.assertEqual(self.width(), "25 kHz")
        self.assertEqual(self.panel.get_fm_deviation_text(), "5 kHz")

    def test_widths_are_remembered_per_mode_and_saved(self):
        self.pick("USB")
        self.panel._bw_combo.setCurrentText("1.8 kHz")
        self.pick("CW")
        self.panel._bw_combo.setCurrentText("250 Hz")
        self.pick("AM")
        self.assertEqual(self.width(), "10 kHz")
        self.pick("USB")
        self.assertEqual(self.width(), "1.8 kHz")
        self.pick("CW")
        self.assertEqual(self.width(), "250 Hz")
        saved = json.loads(self.store.values["audio/mode_filters"])
        self.assertEqual(saved["USB"]["width"], "1.8 kHz")
        self.assertEqual(saved["CW"]["width"], "250 Hz")
        # A new window remembers them.
        self.win._persist_state()
        other = self.mw.SDRMainWindow()
        other._confirm_discard_recording = lambda *_a: True
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        other._control_panel._demod_combo.setCurrentText("USB")
        self.assertEqual(other._control_panel._bw_combo.currentText(), "1.8 kHz")

    def test_broadcast_and_narrow_fm_are_remembered_apart(self):
        self.win.set_frequency(146.52e6)
        self.pick("AM")
        self.pick("FM")
        self.panel._bw_combo.setCurrentText("12.5 kHz")
        self.panel._fm_dev_combo.setCurrentText("2.5 kHz")
        self.win.set_frequency(100.1e6)
        self.pick("AM")
        self.pick("FM")
        self.assertEqual(
            (self.width(), self.panel.get_fm_deviation_text()), ("200 kHz", "75 kHz")
        )
        self.win.set_frequency(446.0e6)
        self.pick("AM")
        self.pick("FM")
        self.assertEqual(
            (self.width(), self.panel.get_fm_deviation_text()), ("12.5 kHz", "2.5 kHz")
        )

    def test_presets_bookmarks_and_restore_set_their_own_widths(self):
        from sdr_module.gui.main_window import BAND_PRESETS

        for preset in BAND_PRESETS:
            self.win._apply_band_preset(preset.frequency_hz, preset.mode, preset.label)
            self.assertEqual(self.width(), preset.bandwidth, preset.label)
            self.assertIn(preset.bandwidth, self.choices())
        self.win._on_bookmark_tune(144.39e6, "APRS", "FM", "5 kHz")
        self.assertEqual(self.width(), "25 kHz")
        self.panel._category_combo.setCurrentText("QRP")
        self.panel._preset_combo.setCurrentText("QRP 20m SSB")
        self.panel._apply_preset()
        self.assertEqual(
            (self.panel.get_demod_mode(), self.width()), ("USB", "2.7 kHz")
        )
        # Restore: the saved mode and width win over the remembered ones.
        self.store.values.update(
            {"demod_mode": "CW", "bandwidth": "100 Hz", "frequency_hz": 7.03e6}
        )
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel.get_demod_mode(), "CW")
        self.assertEqual(other._control_panel._bw_combo.currentText(), "100 Hz")

    def test_a_width_from_an_older_version_does_not_stick(self):
        self.store.values.update({"demod_mode": "USB", "bandwidth": "10 kHz"})
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel._bw_combo.currentText(), "2.7 kHz")

    def test_info_tab_and_messages_show_the_real_filter(self):
        # RX-11: Info said "Bandwidth: 2.4 MHz" in USB.
        self.pick("USB")
        self.panel._bw_combo.setCurrentText("2.4 kHz")
        self.win._refresh_info_panel()
        self.assertEqual(
            self.win._info_panel.value("bandwidth"), "2.4 kHz (USB, 200-2600 Hz)"
        )
        self.assertEqual(self.win._passband_edges(), (200.0, 2600.0))
        self.assertEqual(self.win._spectrum.passband(), (200.0, 2600.0))
        self.pick("CW")
        self.win._refresh_info_panel()
        self.assertIn("CW, 700 Hz tone", self.win._info_panel.value("bandwidth"))
        self.assertIn("500 Hz filter", self.win._message_label.text())

    def test_cw_pitch_only_in_cw(self):
        form = self.panel._demod_form
        self.pick("FM")
        self.assertFalse(form.isRowVisible(self.panel._cw_pitch_spin))
        self.pick("CW")
        self.assertTrue(form.isRowVisible(self.panel._cw_pitch_spin))
        self.assertEqual(
            (self.panel._cw_pitch_spin.minimum(), self.panel._cw_pitch_spin.maximum()),
            (400, 1000),
        )
        self.panel.set_cw_pitch(550)
        self.assertEqual(self.win._audio_options.cw_pitch_hz, 550.0)
        self.assertEqual(self.store.values["audio/cw_pitch"], 550)

    def test_bandwidth_tooltip_explains_the_mode(self):
        self.pick("USB")
        self.assertIn("SSB filter", self.panel._bw_combo.toolTip())
        self.pick("CW")
        self.assertIn("CW filter", self.panel._bw_combo.toolTip())
        self.pick("FM")
        self.assertIn("channel", self.panel._bw_combo.toolTip())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestFmDeviation(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel

    def test_items_say_what_they_are_for(self):
        combo = self.panel._fm_dev_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        self.assertIn("75 kHz (broadcast)", labels)
        self.assertIn("5 kHz (narrow voice)", labels)
        combo.setCurrentText("75 kHz")  # the value still works...
        self.assertEqual(combo.current_label(), "75 kHz (broadcast)")
        self.assertEqual(self.panel.get_fm_deviation_text(), "75 kHz")
        self.assertEqual(self.panel.get_fm_deviation(), 75e3)
        combo.setCurrentText("5 kHz (narrow voice)")  # ...and so does the label
        self.assertEqual(self.store.values["fm_deviation"], "5 kHz")

    def test_clipping_fm_audio_is_reported(self):
        pretend_audio_output(self.win)
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "200 kHz")
        self.panel.set_squelch_db(-100)
        t = np.arange(40 * FRAME) / RATE
        iq = fm(np.sin(2 * np.pi * 1000 * t), 60e3, -30.0) + noise(40 * FRAME, -90.0)
        Frames(self, iq).run()
        text = self.win._message_label.text()
        self.assertIn("clipping", text)
        self.assertIn("75 kHz (broadcast)", text)
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        # Once is enough for a while.
        self.win._clear_status_message()
        Frames(self, iq).run()
        self.assertNotIn("clipping", self.win._message_label.text())

    def test_squelch_tails_of_noise_do_not_warn(self):
        # The noise heard for the hang time after each transmission clips;
        # it used to add up over a few transmissions into a false warning.
        pretend_audio_output(self.win)
        self.win.set_frequency(145.29e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        t = np.arange(20 * FRAME) / RATE
        voice = fm(0.7 * np.sin(2 * np.pi * 800 * t), 5e3, -45.0)
        burst = noise(40 * FRAME, -85.0)
        burst[: 20 * FRAME] += voice
        self.win._clear_status_message()
        for _ in range(5):  # five transmissions, each with its tail
            Frames(self, burst).run()
        self.assertNotIn("clipping", self.win._message_label.text())

    def test_a_matching_deviation_does_not_warn(self):
        pretend_audio_output(self.win)
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.panel.set_squelch_db(-100)
        t = np.arange(40 * FRAME) / RATE
        iq = fm(0.9 * np.sin(2 * np.pi * 1000 * t), 60e3, -30.0)
        self.win._clear_status_message()
        Frames(self, iq).run()
        self.assertNotIn("clipping", self.win._message_label.text())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNoClickOnChanges(_WindowTestCase):
    """rx:RX-15: a new receiver chain (Bandwidth, deviation or mode
    change) used to start from empty filters: a step to 0 and back."""

    demo = True

    def _seam(self, freq, mode, deviation, bandwidth, change):
        from sdr_module.gui.device_dialog import MockDevice

        pretend_audio_output(self.win)
        panel = self.win._control_panel
        device = MockDevice()
        device.set_frequency(freq)
        device.start_rx()
        self.win.set_frequency(freq)
        self.win._set_demod_mode("AM" if mode != "AM" else "FM")
        self.win._set_demod_mode(mode, deviation, bandwidth)
        panel.set_squelch_db(-120)
        self.win._reset_demodulator()
        written = []
        self.win._audio.write = lambda a: written.append(np.array(a, copy=True))
        for _ in range(6):
            self.win._process_block(device.read_samples(FRAME))
        before = np.concatenate(written)
        change(panel)
        written.clear()
        self.win._process_block(device.read_samples(FRAME))
        after = np.concatenate(written)
        steps = np.abs(np.diff(before[2400:]))
        seam = np.abs(np.diff(np.concatenate((before[-5:], after[:300]))))
        return float(before[-1]), float(after[0]), float(seam.max()), steps

    def test_new_chains_join_the_old_audio(self):
        cases = (
            (
                100.1e6,
                "FM",
                "75 kHz",
                "200 kHz",
                lambda p: p._bw_combo.setCurrentText("100 kHz"),
            ),
            (
                125.0e6,
                "AM",
                None,
                "10 kHz",
                lambda p: p._bw_combo.setCurrentText("15 kHz"),
            ),
            (
                162.55e6,
                "FM",
                "5 kHz",
                "25 kHz",
                lambda p: p._fm_dev_combo.setCurrentText("12.5 kHz"),
            ),
            (
                99.8185e6,
                "USB",
                None,
                "2.7 kHz",
                lambda p: p._bw_combo.setCurrentText("1.8 kHz"),
            ),
            (
                99.82e6,
                "CW",
                None,
                "500 Hz",
                lambda p: p._demod_combo.setCurrentText("USB"),
            ),
        )
        for freq, mode, deviation, bandwidth, change in cases:
            last, first, seam, steps = self._seam(
                freq, mode, deviation, bandwidth, change
            )
            self.assertAlmostEqual(first, last, delta=0.02, msg=mode)  # was 0.506 -> 0
            self.assertLessEqual(seam, max(1.5 * np.percentile(steps, 99), 0.05), mode)
