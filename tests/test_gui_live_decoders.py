"""Live decoders in the window and the Decoder panel's health line.

FG-B01 / TOOLS-01: the window fed the raw span to protocol decoders that
could not read it, so only ADS-B ever decoded; now each protocol has a live
decoder that picks out its channel at the VFO. TOOLS-03 / TOOLS-04: ADS-B at
1.8 MS/s said "Listening" forever and was never rebuilt when the rate
became valid. UX-01 / FG-U03: the panel had no health status. UX-B11:
"Tune to 1090.000 MHz" kept FM, so ADS-B pulses reached the speaker.
FG-U04: the demo had nothing but ADS-B to decode.
"""

import logging
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
    from PyQt6.QtWidgets import QApplication


def _settle(n=5):
    for _ in range(n):
        QApplication.processEvents()


# --------------------------------------------------------------------------- #
# The window builds, feeds and rebuilds the live decoder
# --------------------------------------------------------------------------- #


class TestLiveDecoderWiring(_WindowTestCase):
    def select(self, name):
        self.win._decoder_panel._proto_combo.setCurrentText(name)
        _settle(2)
        return self.win._decoder

    def test_selecting_a_protocol_builds_its_live_decoder(self):
        from sdr_module.dsp.live_decoders import POCSAGLiveDecoder

        decoder = self.select("POCSAG")
        self.assertIsInstance(decoder, POCSAGLiveDecoder)
        self.assertEqual(decoder.sample_rate, self.win._device_sample_rate())
        self.assertEqual(self.win._decoder_rate, self.win._device_sample_rate())
        self.assertEqual(decoder.offset_hz, 0.0)
        self.select("Off")
        self.assertIsNone(self.win._decoder)
        self.assertEqual(self.win._decoder_panel.decoder_status_text(), "")

    def test_every_raw_block_reaches_the_decoder(self):
        decoder = self.select("AX.25/APRS")
        seen = []
        decoder.process = lambda block: seen.append(len(block)) or []
        block = np.zeros(80_000, dtype=np.complex64)
        self.win._process_block(block)
        self.assertEqual(seen, [80_000])
        # Decode unticked: paused, nothing is fed.
        self.win._decoder_panel._enabled_check.setChecked(False)
        self.win._process_block(block)
        self.assertEqual(seen, [80_000])
        self.assertEqual(
            self.win._decoder_panel.decoder_status_text(),
            "Paused: tick Decode to resume",
        )

    def test_the_decoder_follows_the_vfo_offset(self):
        # T1 adds _vfo_offset_hz (tuned frequency minus LO); until then it is
        # read with a fallback of 0.
        self.win._vfo_offset_hz = lambda: 250e3
        decoder = self.select("ACARS")
        self.assertEqual(decoder.offset_hz, 250e3)
        self.win._vfo_offset_hz = lambda: -120e3
        self.win._run_decoder(np.zeros(4096, dtype=np.complex64))
        self.assertEqual(decoder.offset_hz, -120e3)

    def test_an_offset_reader_that_fails_counts_as_no_offset(self):
        def broken():
            raise AttributeError("half-built window")

        self.win._vfo_offset_hz = broken
        self.assertEqual(self.win._decoder_offset_hz(), 0.0)

    def test_rows_and_status_come_from_the_decoder(self):
        from sdr_module.dsp.protocols import POCSAGMessage, ProtocolType

        decoder = self.select("POCSAG")
        page = POCSAGMessage(
            protocol=ProtocolType.POCSAG,
            timestamp=0.0,
            raw_bits=b"",
            valid=True,
            address=1234567,
            function=3,
            content="HELLO",
        )
        decoder._decode = lambda iq: [page]
        self.win._decoder_panel.set_receiving(True)
        self.win._run_decoder(np.zeros(1000, dtype=np.complex64))
        panel = self.win._decoder_panel
        self.assertEqual(
            [(m["address"], m["content"]) for m in panel._messages],
            [("1234567 (F3)", "HELLO")],
        )
        self.assertTrue(panel.decoder_status_text().startswith("POCSAG  ·  1 page  ·"))
        self.assertEqual(panel._decoded_label.text(), "1")

    def test_retuning_forgets_the_signal_heard_on_the_old_channel(self):
        # Regression: "Signal present but nothing decodes" stayed on after
        # retuning to a quiet channel (seen in the window going from a
        # 146.940 MHz voice channel to 150.000 MHz with POCSAG selected).
        decoder = self.select("POCSAG")
        panel = self.win._decoder_panel
        block = np.zeros(4096, dtype=np.complex64)
        panel.set_receiving(True)
        self.win._run_decoder(block)
        decoder._undecoded = 10.0  # as after 10 s of undecodable signal
        decoder._since_present = 0.0
        self.win._run_decoder(block)
        self.assertTrue(
            panel.decoder_status_text().startswith("Signal present but nothing decodes")
        )
        self.win.set_frequency(self.win._current_frequency() + 1e6)
        _settle(2)
        panel.set_receiving(True)
        self.win._run_decoder(block)
        self.assertEqual(decoder.status.undecoded_signal_s, 0.0)
        self.assertNotIn("nothing decodes", panel.decoder_status_text())


class TestLiveDecoderRates(_WindowTestCase):
    """With the demo device, whose sample rate can be changed."""

    demo = True

    def select(self, name):
        self.win._decoder_panel._proto_combo.setCurrentText(name)
        _settle(2)
        return self.win._decoder

    def test_ads_b_below_2_ms_s_says_so_and_recovers(self):
        win = self.win
        win.set_sample_rate(1.8e6)
        self.select("ADS-B")
        panel = win._decoder_panel
        problem = "ADS-B needs at least 2 MS/s; the device runs at 1.8 MS/s"
        self.assertEqual(panel.decoder_status_text(), problem)
        self.assertEqual(panel._status_label.property("tone"), "warning")
        self.assertIn("ADS-B can't decode here", panel._empty.text())
        self.assertIn("Choose a higher sample rate", panel._empty.text())
        self.assertIn("Choose a higher sample rate", win._message_label.text())
        # A valid rate rebuilds it (TOOLS-04).
        win.set_sample_rate(2.4e6)
        self.assertEqual(win._decoder.sample_rate, 2.4e6)
        self.assertIsNone(win._decoder.status.problem)
        self.assertNotEqual(panel.decoder_status_text(), problem)
        self.assertNotIn("can't decode", panel._empty.text())

    def test_a_failed_build_is_retried_when_the_rate_changes(self):
        from sdr_module.dsp import live_decoders

        real = live_decoders.create_live_decoder
        calls = []

        def failing(*args, **kwargs):
            calls.append(args)
            raise RuntimeError("no luck")

        with mock.patch.object(live_decoders, "create_live_decoder", failing):
            with self.assertLogs("sdr_module.gui.main_window", logging.WARNING):
                self.select("POCSAG")
        self.assertIsNone(self.win._decoder)
        self.assertIn("could not start", self.win._decoder_panel.decoder_status_text())
        self.assertEqual(len(calls), 1)
        self.assertIs(live_decoders.create_live_decoder, real)
        self.win.set_sample_rate(1.024e6)
        self.assertIsNotNone(self.win._decoder)
        self.assertEqual(self.win._decoder.sample_rate, 1.024e6)


# --------------------------------------------------------------------------- #
# "Tune to ..." picks a mode that suits listening (UX-B11)
# --------------------------------------------------------------------------- #


class TestTuneToModes(_WindowTestCase):
    def tune(self, name):
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText(name)
        self.win.set_frequency(100e6)  # away from it: the button shows
        _settle(2)
        self.assertTrue(panel._empty.action_text().startswith("Tune to"), name)
        panel._empty.button.click()
        _settle(2)
        cp = self.win._control_panel
        return (
            cp.get_demod_mode(),
            cp.get_fm_deviation_text(),
            cp._bw_combo.currentText(),
        )

    def test_modes(self):
        mode, _dev, _bw = self.tune("ADS-B")
        self.assertEqual(mode, "None (I/Q)")  # no audio from the pulses
        self.assertAlmostEqual(self.freq(), 1090e6)
        for name, freq in (
            ("AX.25/APRS", 144.39e6),
            ("POCSAG", 152.24e6),
            ("FLEX", 931.9375e6),
        ):
            self.assertEqual(self.tune(name), ("FM", "5 kHz", "25 kHz"), name)
            self.assertAlmostEqual(self.freq(), freq)
        mode, _dev, bandwidth = self.tune("ACARS")
        self.assertEqual((mode, bandwidth), ("AM", "10 kHz"))
        self.assertAlmostEqual(self.freq(), 131.55e6)


# --------------------------------------------------------------------------- #
# The demo's transmitters decode in the window
# --------------------------------------------------------------------------- #


class TestDemoDecodesInTheWindow(_WindowTestCase):
    demo = True

    # name -> (rows expected from the first transmission(s), seconds of
    # stream that holds them, counting from the start of the schedule)
    EXPECTED = {
        "AX.25/APRS": (
            [("N0CALL-9>APZ001", "!4903.50N/07201.75W>Demo mobile on 144.390 MHz")],
            2.5,
        ),
        "POCSAG": (
            [("1234567 (F3)", "SDR Module demo: POCSAG 1200 page on 152.240 MHz")],
            5.0,
        ),
        "FLEX": (
            [("1234567", "SDR Module demo: FLEX 1600 page on 931.9375 MHz")],
            7.5,
        ),
        "ACARS": (
            [("N601SD", "H1  ·  DEMO MESSAGE FROM SDR MODULE - ETA 1435Z")],
            7.8,
        ),
    }

    def run_protocol(self, name):
        win = self.win
        panel = win._decoder_panel
        panel.clear()
        panel._proto_combo.setCurrentText(name)
        win.set_frequency(100.1e6)
        _settle(2)
        panel._empty.button.click()  # "Tune to ..."
        _settle(2)
        dev = win._device
        if not dev.is_streaming:
            dev.start_rx()
        dev._stream_pos = 0  # the start of every schedule
        expected, seconds = self.EXPECTED[name]
        rate = win._device_sample_rate()
        block = int(rate * 0.1)
        for _ in range(int(seconds / 0.1) + 1):
            win._process_block(dev.read_samples(block))
        rows = [(m["address"], m["content"]) for m in panel._messages if m["valid"]]
        self.assertEqual(rows[: len(expected)], expected, name)
        self.assertIn(f"{name}  ·  ", panel.decoder_status_text())

    def test_ads_b_keeps_the_speaker_quiet(self):
        # UX-B11: "Tune to 1090.000 MHz" kept FM, so the pulse bursts were
        # played as noise; I/Q mode sends nothing to the speaker.
        win = self.win
        written = []
        win._audio.write = lambda audio: written.append(len(audio))
        win._audio_enabled = True
        win._realtime = lambda: True
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("ADS-B")
        win.set_frequency(100.1e6)
        _settle(2)
        panel._empty.button.click()
        _settle(2)
        self.assertEqual(win._control_panel.get_demod_mode(), "None (I/Q)")
        dev = win._device
        if not dev.is_streaming:
            dev.start_rx()
        block = int(win._device_sample_rate() * 0.1)
        for _ in range(30):
            win._process_block(dev.read_samples(block))
        self.assertEqual(written, [])
        self.assertEqual({m["address"] for m in panel._messages}, {"4840D6"})

    def test_aprs(self):
        self.run_protocol("AX.25/APRS")

    def test_pocsag(self):
        self.run_protocol("POCSAG")

    def test_flex(self):
        self.run_protocol("FLEX")

    def test_acars(self):
        self.run_protocol("ACARS")


# --------------------------------------------------------------------------- #
# The Decoder panel's status line and stats
# --------------------------------------------------------------------------- #


class TestDecoderPanelStatus(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self.panel = decoder_panel.DecoderPanel()
        self.panel.resize(380, 420)
        self.now = 1000.0
        self.panel._clock = lambda: self.now

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)

    def status(self, **fields):
        from sdr_module.dsp.live_decoders import LiveDecoderStatus

        base = dict(name="POCSAG", unit="page", signal_measured=True)
        base.update(fields)
        return LiveDecoderStatus(**base)

    def show(self, status=None, receiving=True, freq=152.24e6):
        p = self.panel
        p._proto_combo.setCurrentText("POCSAG")
        p.set_current_frequency(freq)
        p.set_receiving(receiving)
        p.set_decoder_status(status)
        return p.decoder_status_text(), p._status_label.property("tone") or None

    def test_off_hides_the_line(self):
        self.assertEqual(self.panel.decoder_status_text(), "")
        self.assertTrue(self.panel._status_label.isHidden())

    def test_stopped_and_listening(self):
        self.assertEqual(
            self.show(self.status(), receiving=False),
            ("Stopped: press Start (Space) to decode", "muted"),
        )
        self.assertEqual(self.show(self.status()), ("Listening on 152.240 MHz", None))
        self.assertEqual(
            self.show(self.status(), freq=100e6)[0],
            "Not tuned to POCSAG: listening on 100.000 MHz",
        )

    def test_counts_and_age(self):
        text, tone = self.show(self.status(valid=3, last_time=self.now - 4))
        self.assertEqual(text, "POCSAG  ·  3 pages  ·  last 4 s ago")
        self.assertEqual(tone, "success")
        text, tone = self.show(self.status(valid=1, failed=2, last_time=self.now - 600))
        self.assertEqual(
            text, "POCSAG  ·  1 page  ·  last 10 min ago  ·  2 failed checks"
        )
        self.assertIsNone(tone)
        text, _tone = self.show(self.status(valid=2, failed=1, last_time=self.now))
        self.assertEqual(
            text, "POCSAG  ·  2 pages  ·  last just now  ·  1 failed check"
        )

    def test_clear_counts_from_zero_again(self):
        p = self.panel
        self.show(self.status(frames=6, valid=5, failed=1, last_time=self.now - 3))
        p.clear()
        p.set_decoder_status(
            self.status(frames=6, valid=5, failed=1, last_time=self.now)
        )
        self.assertEqual(p.decoder_status_text(), "Listening on 152.240 MHz")
        self.assertEqual(
            (p._frames_label.text(), p._decoded_label.text(), p._failed_label.text()),
            ("0", "0", "0"),
        )
        p.set_decoder_status(
            self.status(frames=7, valid=6, failed=1, last_time=self.now)
        )
        self.assertEqual(p.decoder_status_text(), "POCSAG  ·  1 page  ·  last just now")
        # A new decoder (fewer counts than at Clear) counts from its own zero.
        p.set_decoder_status(self.status(frames=1, valid=1, last_time=self.now))
        self.assertEqual(p.decoder_status_text(), "POCSAG  ·  1 page  ·  last just now")
        self.assertEqual(p._frames_label.text(), "1")

    def test_failures_and_undecodable_signal_are_flagged(self):
        text, tone = self.show(self.status(failed=2, frames=2))
        self.assertTrue(text.startswith("2 pages failed their checks"), text)
        self.assertEqual(tone, "warning")
        text, tone = self.show(self.status(signal_present=True, undecoded_signal_s=1))
        self.assertEqual(
            (text, tone), ("Signal on 152.240 MHz: waiting for POCSAG pages", "info")
        )
        warning = (
            "Signal present but nothing decodes: check the frequency and "
            "the protocol",
            "warning",
        )
        text, tone = self.show(self.status(signal_present=True, undecoded_signal_s=9))
        self.assertEqual((text, tone), warning)
        # Also after earlier successes (e.g. retuned to another signal).
        status = self.status(
            valid=4, last_time=self.now - 30, signal_present=True, undecoded_signal_s=9
        )
        self.assertEqual(self.show(status), warning)

    def test_a_problem_wins_and_reaches_the_empty_table(self):
        text, tone = self.show(self.status(problem="Something is wrong"))
        self.assertEqual((text, tone), ("Something is wrong", "warning"))
        self.assertIn("POCSAG can't decode here", self.panel._empty.text())
        self.show(self.status())
        self.assertNotIn("can't decode", self.panel._empty.text())

    def test_stats_tab_counts(self):
        p = self.panel
        self.show(None)
        self.assertEqual(p._frames_label.text(), "–")
        self.show(self.status(frames=5, valid=3, failed=2, signal_present=True))
        self.assertEqual(
            (p._frames_label.text(), p._decoded_label.text(), p._failed_label.text()),
            ("5", "3", "2"),
        )
        self.assertEqual(p._signal_label.text(), "Yes")
        self.assertEqual(p._failed_label.property("tone"), "danger")
        self.show(self.status(frames=5, valid=5))
        self.assertFalse(p._failed_label.property("tone"))
        self.show(self.status(signal_measured=False))
        self.assertEqual(p._signal_label.text(), "–")

    def test_views_can_be_added_before_stats(self):
        from PyQt6.QtWidgets import QLabel

        p = self.panel
        index = p.add_view(QLabel("aircraft"), "Aircraft", "Aircraft heard")
        self.assertEqual(p._tabs.tabText(index), "Aircraft")
        self.assertEqual(p._tabs.tabText(p._tabs.count() - 1), "Stats")
        self.assertEqual(p._tabs.tabToolTip(index), "Aircraft heard")

    def test_a_short_table_keeps_the_tune_button_whole(self):
        # At 1024x640 the table is about 86 px tall: the hint lines give way
        # to the button (it used to be cut off at the bottom).
        p = self.panel
        p.tune_requested.connect(lambda *_args: None)
        p._proto_combo.setCurrentText("AX.25/APRS")
        p.set_receiving(True)
        p.set_current_frequency(100e6)
        p.show()
        p._table.setFixedHeight(90)  # (about what the themed window leaves)
        _settle(3)
        viewport = p._table.viewport().rect()
        button = p._empty.button.geometry()
        self.assertTrue(viewport.contains(button), (viewport, button))
        self.assertIn("Waiting for AX.25/APRS messages", p._empty.label.text())
        self.assertNotIn("(N. America)", p._empty.label.text())  # title only
        self.assertIn("Tune to 144.390 MHz", p._empty.text())  # the full text
        p._table.setMinimumHeight(110)
        p._table.setMaximumHeight(16777215)
        p.resize(380, 520)
        _settle(3)
        self.assertGreater(p._table.viewport().height(), 200)
        self.assertIn("(N. America)", p._empty.label.text())  # room again

    def test_pager_wording(self):
        # UX-10: two bands are alternatives, not both at once.
        p = self.panel
        p.tune_requested.connect(lambda *_args: None)  # a listener: the button
        p._proto_combo.setCurrentText("POCSAG")
        p.set_receiving(True)
        p.set_current_frequency(100e6)
        self.assertIn("Tune to 152–159 MHz or 929–932 MHz to decode.", p._empty.text())
        self.assertEqual(p._empty.action_text(), "Tune to 152.240 MHz")


class TestDeviceDialogRates(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_rates_too_low_for_ads_b_say_so(self):
        # TOOLS-03: 1, 1.4 and 1.8 MS/s were offered with no hint that the
        # ADS-B decoder then decodes nothing.
        from PyQt6.QtCore import Qt

        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = DeviceDialog()
        try:
            combo = dialog._rate_combo
            for i in range(combo.count()):
                rate = combo.itemData(i)
                low = rate < 2e6
                self.assertEqual(combo.itemText(i).endswith("(too low for ADS-B)"), low)
                tip = combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or ""
                self.assertEqual("needs at least 2 MS/s" in tip, low)
            self.assertTrue(any(combo.itemData(i) < 2e6 for i in range(combo.count())))
        finally:
            dialog.close()
            dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Demo device transmitters (FG-U04)
# --------------------------------------------------------------------------- #


class TestDemoTransmitters(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    @staticmethod
    def demo(freq, rate=1.024e6, noise=True):
        from sdr_module.gui.device_dialog import MockDevice

        dev = MockDevice()
        dev.set_sample_rate(rate)
        dev.set_frequency(freq)
        dev.start_rx()
        if not noise:
            dev._noise = lambda n, sigma: np.zeros(n, dtype=np.complex64)
        return dev

    def test_the_band_plan_has_the_data_transmitters(self):
        from sdr_module.gui.device_dialog import _DEMO_SIGNALS

        kinds = {freq: kind for freq, kind, _level, _duty in _DEMO_SIGNALS}
        self.assertEqual(kinds[144.39e6], "aprs")  # was a random 9.6 kb/s burst
        self.assertEqual(kinds[152.24e6], "pocsag")
        self.assertEqual(kinds[931.9375e6], "flex")
        self.assertEqual(kinds[131.55e6], "acars")

    def test_no_background_carrier_sits_on_a_data_channel(self):
        from sdr_module.gui.device_dialog import MockDevice

        for freq in (144.39e6, 152.24e6, 931.9375e6, 131.55e6):
            carriers = MockDevice._background_carriers(freq, 1.2e6)
            near = [c for c in carriers if abs(c[0]) < 20e3]
            self.assertEqual(near, [], freq)

    def test_each_transmitter_decodes_from_the_stream(self):
        from sdr_module.dsp.live_decoders import create_live_decoder

        for name, freq, seconds, count in (
            ("AX.25/APRS", 144.39e6, 2.5, 1),
            ("POCSAG", 152.24e6, 5.0, 1),
            ("FLEX", 931.9375e6, 7.5, 1),
            ("ACARS", 131.55e6, 7.8, 1),
        ):
            dev = self.demo(freq)
            decoder = create_live_decoder(name, dev.sample_rate)
            block = int(dev.sample_rate * 0.034)
            messages = []
            for _ in range(int(seconds / 0.034)):
                messages += decoder.process(dev.read_samples(block))
            self.assertEqual(sum(m.valid for m in messages), count, name)
            self.assertFalse([m for m in messages if not m.valid], name)

    def test_reads_stay_fast_while_transmitting(self):
        from sdr_module.gui import device_dialog as dd

        for kind, freq in (("aprs", 144.39e6), ("flex", 931.9375e6)):
            dev = self.demo(freq, rate=2.4e6)
            n = int(dev.sample_rate * 0.034)
            start_s = dd._DATA_SCHEDULE[kind][1][0][0]
            dev.read_samples(n)  # builds the transmissions once
            dev._stream_pos = int((start_s + 0.1) * dev.sample_rate)
            times = []
            for _ in range(10):
                begin = time.perf_counter()
                dev.read_samples(n)
                times.append(time.perf_counter() - begin)
            per_sample_us = float(np.median(times)) / n * 1e6
            self.assertLess(per_sample_us, 0.15, kind)

    def test_reads_of_any_size_make_one_continuous_stream(self):
        from sdr_module.gui import device_dialog as dd

        for kind, freq in (("aprs", 144.39e6), ("acars", 131.55e6)):
            whole_dev = self.demo(freq, noise=False)
            first = dd._DATA_SCHEDULE[kind][1][0][0]
            whole_dev._stream_pos = int((first - 0.1) * whole_dev.sample_rate)
            whole = whole_dev.read_samples(700_000)
            dev = self.demo(freq, noise=False)
            dev._stream_pos = whole_dev._stream_pos - 700_000
            parts, got = [], 0
            for size in (81_600, 2048, 1, 300_000, 12_345, 7, 400_000):
                size = min(size, len(whole) - got)
                parts.append(dev.read_samples(size))
                got += size
            joined = np.concatenate(parts)
            self.assertGreater(float(np.max(np.abs(whole))), 1e-3)  # on air
            np.testing.assert_allclose(joined, whole, atol=2e-5, err_msg=str(freq))


if __name__ == "__main__":
    unittest.main()
