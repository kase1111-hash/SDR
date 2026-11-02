"""K-P3: Identify listens to the receiver's own channel when that is the band
it analyzes.

In None (I/Q) Identify analyzes the receiver's filter width around the
tuned frequency, the very channel the receiver filters out of every block.
At a high sample rate Identify filtered and decimated the raw block again
with a channelizer of its own: 9.5 ms of a 20 MS/s display frame, which
overloaded the display loop (81-87 % of the time, 40-50 % of the signal
dropped by the driver) and broke Identify's own captures. Now it takes the
receiver's channel as it is when it matches (centered on the tuned
frequency, flat to the edges of the analysis band, decimated); otherwise
(other modes, the demo's rates where Identify only mixes the span) nothing
changes.
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.dsp.signal_id import MIN_S, SignalIdentifier  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

RATE = 8e6  # a HackRF rate where Identify filters and decimates the span
FRAME_S = 1 / 30


def _fm(n, rate, start=0, deviation=5e3, tone_hz=1e3, level=0.3, noise=2e-3):
    """Narrow FM with a tone at 0 Hz (the tuned frequency), plus noise."""
    t = (start + np.arange(n)) / rate
    phase = (deviation / tone_hz) * np.sin(2 * np.pi * tone_hz * t)
    rng = np.random.default_rng(start // max(n, 1) + 1)
    x = level * np.exp(1j * phase) + noise * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return x.astype(np.complex64)


def _cw(n, rate, start=0, element_s=0.06, level=0.3, noise=2e-3):
    """A keyed carrier at 0 Hz (two elements on, one off), plus noise."""
    t = (start + np.arange(n)) / rate
    key = (np.floor(t / element_s) % 3) != 2
    rng = np.random.default_rng(start // max(n, 1) + 7)
    x = level * key + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x.astype(np.complex64)


class TestAnIdentifierForTheReceiversChannel(unittest.TestCase):
    def test_the_channel_is_kept_as_it_is(self):
        ident = SignalIdentifier(4e6, 1.3e6, 2.4e6, channel_pass_hz=1.2e6)
        self.assertEqual(ident.rate, 4e6)
        self.assertEqual(ident.offset_hz, 1.3e6)
        self.assertAlmostEqual(ident.passband_hz[0], -0.95 * 1.2e6)
        self.assertAlmostEqual(ident.passband_hz[1], 0.95 * 1.2e6)
        block = (np.arange(200_000) * (1 + 1j) * 1e-6).astype(np.complex64)
        for _ in range(int(MIN_S * 4e6 / len(block)) + 1):
            ident.feed(block)
        self.assertTrue(ident.ready())
        np.testing.assert_array_equal(ident.take()[-len(block) :], block)

    def test_a_vfo_move_listens_afresh(self):
        ident = SignalIdentifier(4e6, 0.0, 2.4e6, channel_pass_hz=1.2e6)
        ident.feed(np.ones(1000, np.complex64))
        ident.set_offset(250e3)  # (the receiver's channel follows the VFO)
        self.assertEqual(ident.received, 0)
        self.assertEqual(ident.offset_hz, 250e3)

    def test_a_capture_is_judged_at_the_rate_asked(self):
        # A keyed carrier taken at 4 MS/s, judged at 2.67 MS/s (the rate
        # Identify's own filter gives a 2.4 MHz band at 8 MS/s): as if it
        # had been taken at that rate. (Judged at 4 MS/s, its line looks
        # twice as wide as it is and CW was "maybe".)
        from sdr_module.dsp.signal_id import identify

        band = (-0.95 * 1.2e6, 0.95 * 1.2e6)
        slow, fast = 8e6 / 3, 4e6
        own = identify(_cw(int(0.3 * slow), slow), slow, 145.5e6, 25e3, None, band)
        taken = _cw(int(0.3 * fast), fast)
        judged = identify(taken, fast, 145.5e6, 25e3, None, band, analysis_rate=slow)
        self.assertEqual(judged.kind, "cw")
        self.assertAlmostEqual(judged.confidence, own.confidence, delta=0.05)
        self.assertAlmostEqual(judged.seconds, 0.3, places=3)
        ident = SignalIdentifier(fast, 0.0, 2.4e6, channel_pass_hz=1.2e6)
        self.assertEqual(ident.analysis_rate, fast)  # (as it is, by default)
        ident = SignalIdentifier(
            fast, 0.0, 2.4e6, channel_pass_hz=1.2e6, analysis_rate=slow
        )
        self.assertEqual(ident.analysis_rate, slow)
        self.assertEqual(ident.rate, fast)  # (what it keeps)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _HighRateCase(_WindowTestCase):
    """The window on a hardware-like device at RATE (frames run here),
    listening in None (I/Q) 2.4 MHz; Identify's jobs run as handed over."""

    def setUp(self):
        super().setUp()
        from tests.test_final_plots import _RunAtOnce
        from tests.test_review_mainloop import _queued_hardware

        win = self.win
        self.dev = _queued_hardware(RATE)
        win._device = self.dev
        win._is_running = True
        win._pacer.reset()
        win._set_demod_mode("None (I/Q)", "", "2.4 MHz")
        self.assertEqual(win._channel_bandwidth(), 2.4e6)
        self.tools = win._signal_tools
        self.tools._pool = _RunAtOnce()
        self.sent = 0
        self.signal = _fm  # (what is on the air)

    def tearDown(self):
        if HAS_PYQT6:
            self.win._is_running = False
            self.win._device = None
        super().tearDown()

    def frames(self, seconds, until=None):
        win = self.win
        n = int(RATE * FRAME_S)
        for _ in range(max(1, int(round(seconds / FRAME_S)))):
            win._process_block(self.signal(n, RATE, self.sent))
            self.sent += n
            if until is not None and until():
                return True
        return False

    def identify(self):
        self.assertTrue(self.tools.identify())
        done = self.frames(3.0, until=lambda: self.tools._identifier is None)
        self.assertTrue(done, self.win._signal_panel.identify_status.text())
        return self.win._signal_panel.identification()


class TestIdentifyTakesTheReceiversChannel(_HighRateCase):
    def test_no_second_channel_filter_runs(self):
        from sdr_module.dsp import channelizer

        calls = []
        real = channelizer.Channelizer.process

        def counted(self_, block):
            calls.append(len(block))
            return real(self_, block)

        with mock.patch.object(channelizer.Channelizer, "process", counted):
            self.frames(0.2)  # (the receiver alone)
            before = len(calls)
            found = self.identify()
        self.assertEqual(len(calls), before)  # none for Identify
        self.assertIsNotNone(found)
        self.assertTrue(found.is_signal, found)

    def test_the_same_verdict_as_filtering_its_own(self):
        from sdr_module.gui import signal_tools

        for width in ("2.4 MHz", "200 kHz"):
            self.win._set_demod_mode("None (I/Q)", "", width)
            self.win._process_block(_fm(int(RATE * FRAME_S), RATE))
            self.sent = 0
            reused = self.identify()
            self.assertIsNotNone(self.tools._ident_chain, width)
            self.sent = 0
            with mock.patch.object(
                signal_tools, "receiver_channel_band", lambda *args: 0.0
            ):
                own = self.identify()
            self.assertEqual(reused.kind, own.kind, width)
            self.assertEqual(reused.is_signal, own.is_signal, width)
            self.assertEqual(reused.label, own.label, width)
            self.assertEqual(
                (reused.mode, reused.fm_deviation), (own.mode, own.fm_deviation)
            )
            self.assertLess(abs(reused.signal_hz - own.signal_hz), 2e3, width)

    def test_a_keyed_carrier_is_judged_as_its_own_filter_judges_it(self):
        # The receiver's 2.4 MHz channel comes at 4 MS/s, Identify's own
        # filter at 2.67 MS/s: judged at 4 MS/s, Morse's narrow line looked
        # half again as wide, and the verdict was "CW ... maybe" (0.49)
        # where its own filter says "confident" (0.86), and took three
        # takes. The same signal now gets the same verdict either way.
        from sdr_module.dsp.signal_id import confidence_words
        from sdr_module.gui import signal_tools

        self.signal = _cw
        reused = self.identify()
        self.assertIsNotNone(self.tools._ident_chain)
        self.sent = 0
        with mock.patch.object(
            signal_tools, "receiver_channel_band", lambda *args: 0.0
        ):
            own = self.identify()
        self.assertEqual((reused.kind, own.kind), ("cw", "cw"))
        self.assertEqual(
            confidence_words(reused.confidence), confidence_words(own.confidence)
        )
        self.assertAlmostEqual(reused.confidence, own.confidence, delta=0.05)

    def test_a_new_filter_while_listening_is_followed(self):
        tools = self.tools
        win = self.win
        self.assertTrue(tools.identify())
        self.frames(0.2)
        win._set_demod_mode("None (I/Q)", "", "1 MHz")  # (a new channel)
        self.frames(FRAME_S)
        # Listening to the new channel at its rate, or to the raw blocks
        # (never to a channel at a rate it doesn't expect).
        ident = tools._identifier
        chain = win._receiver_chain()
        if tools._ident_chain is not None:
            self.assertIs(tools._ident_chain, chain)
            self.assertEqual(ident.rate_in, chain.channel_rate)
        else:
            self.assertEqual(ident.rate_in, RATE)
        done = self.frames(3.0, until=lambda: tools._identifier is None)
        self.assertTrue(done, win._signal_panel.identify_status.text())
        self.assertTrue(win._signal_panel.identification().is_signal)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNothingChangesAtTheDemosRates(_WindowTestCase):
    demo = True

    def test_identify_keeps_its_own_capture(self):
        win = self.win
        win.set_frequency(100.1e6)
        if not win._is_running:
            win._start_acquisition()
        win._set_demod_mode("None (I/Q)", "", "2.4 MHz")
        win._update_display()
        self.assertTrue(win._signal_tools.identify())
        ident = win._signal_tools._identifier
        self.assertEqual(ident.rate, win._device_sample_rate())
        self.assertIsNone(ident._channel)  # (the span, mixed only)
        self.assertIsNone(win._signal_tools._ident_chain)
        # A narrower filter: filtered by Identify itself, as before.
        win._set_demod_mode("None (I/Q)", "", "200 kHz")
        win._update_display()
        self.assertTrue(win._signal_tools.identify())
        self.assertIsNotNone(win._signal_tools._identifier._channel)
        self.assertIsNone(win._signal_tools._ident_chain)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestWithoutTheChannel(_HighRateCase):
    def test_a_frame_without_the_channel_falls_back_to_the_raw_block(self):
        tools = self.tools
        self.assertTrue(tools.identify())
        self.assertIsNotNone(tools._ident_chain)
        n = int(RATE * FRAME_S)
        tools._identify_frame(_fm(n, RATE), 0.0)  # (no channel, no chain)
        self.assertIsNone(tools._ident_chain)
        self.assertIsNotNone(tools._identifier._channel)  # (its own filter)
        self.assertGreater(tools._identifier.received, 0)
