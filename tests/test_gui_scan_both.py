#!/usr/bin/env python3
"""
The Scanner tab's "Use both receivers" (J-dual): a range sweep shared by the
main window's receiver and the Second Receiver window's.

* The engine (gui/scan_engine.py) with a ScanPartner: each receiver takes
  pages of a SharedSweep (dsp/sweep.py), sized for its own sample rate and
  inside what it tunes, the main one from the bottom up and the second from
  the top down; the results are one list; progress, Stop, continuous
  passes; a partner that leaves (or refuses a step) hands its part to the
  main receiver.
* The tab and the window (gui/scanner_panel.py, gui/scanner_host.py,
  gui/second_receiver.py): the choice shows while a second receiver runs,
  for range sweeps; a sweep with both finds what one finds in about half
  the steps, mutes both and puts both back where they were; the second
  receiver stopping or being tuned elsewhere takes it out of the scan.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_scan_both.py``
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

FRAME_S = 0.034


def _demo(rate, freq=100.1e6):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(freq)
    dev.start_rx()
    return dev


def _fm_stations():
    from sdr_module.gui.device_dialog import _DEMO_SIGNALS

    return {f for f, kind, _l, _d in _DEMO_SIGNALS if kind == "wfm"}


class _Receiver:
    """A ScanHost (or ScanPartner) over a demo device."""

    def __init__(self, dev, tunes=None, refuse_above=None):
        self.dev = dev
        self.lo = dev.frequency
        self.vfo = dev.frequency
        self.range = tunes
        self.refuse_above = refuse_above
        self.tunes = []
        self.muted = []

    def scan_tune_lo(self, center, vfo=None):
        if self.refuse_above is not None and center > self.refuse_above:
            return False
        self.dev.set_frequency(center)
        self.lo = center
        self.vfo = center if vfo is None else vfo
        self.tunes.append(center)
        return True

    def scan_tune_channel(self, freq, listening, tone=0.0):
        raise AssertionError("a range sweep tunes whole steps only")

    def scan_lo_moved(self):
        return True

    def scan_lo_hz(self):
        return self.lo

    def scan_signal_present(self):
        return False

    def scan_set_muted(self, muted):
        self.muted.append(bool(muted))

    def scan_sample_rate(self):
        return self.dev.sample_rate

    def scan_settle_s(self):
        return 0.0

    def scan_device_range(self):
        return self.range


def _block(receiver):
    dev = receiver.dev
    return dev.read_samples(int(dev.sample_rate * FRAME_S))


def _run(engine, main, partner=None, seconds=10.0, until=None):
    """Feed the engine about 30 blocks a second of signal from each
    receiver (the two display loops); returns the main receiver's count."""
    frames = 0
    for _ in range(int(seconds / FRAME_S)):
        if not engine.active or (until is not None and until()):
            break
        engine.frame(_block(main))
        frames += 1
        if partner is not None and engine.active:
            engine.partner_frame(_block(partner))
    return frames


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TestSharedSweep(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def engine(self, host):
        from sdr_module.gui.scan_engine import ScanEngine

        engine = ScanEngine(host)
        self.found, self.finished = [], []
        engine.on_found = self.found.append
        engine.on_finished = self.finished.append
        return engine

    def fm_band(self, mode="once"):
        from sdr_module.gui.scan_engine import ScanSettings

        return ScanSettings(mode, 88e6, 108e6, 200e3, min_width_hz=30e3)

    def test_two_receivers_sweep_complementary_parts_in_about_half_the_steps(self):
        # One receiver first, for comparison.
        alone = _Receiver(_demo(2.4e6))
        engine = self.engine(alone)
        self.assertEqual(engine.start(self.fm_band(), 2.4e6, 0.0), "")
        frames_alone = _run(engine, alone)
        self.assertEqual(self.finished, ["complete"])
        one = {f.frequency_hz for f in self.found}

        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(3.2e6, 98e6))
        engine = self.engine(main)
        self.assertEqual(engine.start(self.fm_band(), 2.4e6, 0.0, partner=partner), "")
        self.assertIs(engine.partner, partner)
        progress = []
        engine.on_state = lambda: progress.append(engine.progress())
        frames = _run(engine, main, partner)
        self.assertEqual(self.finished, ["complete"])
        both = {f.frequency_hz for f in self.found}
        # The same stations (the tab lists each once: a strong one beside a
        # step's edge is read by both steps, with one receiver as with two).
        self.assertEqual(both, one)
        self.assertEqual(both, _fm_stations())
        # Each swept its own part: the main receiver from the bottom up, the
        # second from the top down, and they met in between.
        self.assertTrue(main.tunes and partner.tunes)
        self.assertLess(max(main.tunes), min(partner.tunes))
        self.assertEqual(main.tunes, sorted(main.tunes))
        self.assertEqual(partner.tunes, sorted(partner.tunes, reverse=True))
        # In about half the time (the second, at 3.2 MS/s, takes more).
        self.assertLessEqual(frames, 0.65 * frames_alone)
        self.assertGreater(len(partner.tunes), len(main.tunes) - 2)
        self.assertEqual(progress, sorted(progress))
        self.assertGreater(progress[-1], 0.85)
        # Both muted while sweeping, unmuted at the end.
        for receiver in (main, partner):
            self.assertTrue(receiver.muted[0])
            self.assertFalse(receiver.muted[-1])
        self.assertIsNone(engine.partner)

    def test_continuous_passes_and_stop(self):
        from sdr_module.gui.scan_engine import ScanSettings

        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(2.4e6))
        engine = self.engine(main)
        settings = ScanSettings("continuous", 160e6, 166e6, 25e3)
        self.assertEqual(engine.start(settings, 2.4e6, 0.0, partner=partner), "")
        _run(engine, main, partner, seconds=1.5)
        self.assertTrue(engine.active)
        self.assertGreaterEqual(engine.pass_number, 3)
        engine.stop()
        self.assertEqual(self.finished, ["stopped"])
        self.assertFalse(main.muted[-1])
        self.assertFalse(partner.muted[-1])
        noaa = {round(f.frequency_hz) for f in self.found}
        self.assertTrue({162_400_000, 162_550_000} <= noaa, noaa)

    def test_a_partner_that_leaves_hands_the_rest_to_the_main_receiver(self):
        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(2.4e6))
        engine = self.engine(main)
        engine.start(self.fm_band(), 2.4e6, 0.0, partner=partner)
        _run(engine, main, partner, seconds=4 * FRAME_S)
        tuned = len(partner.tunes)
        self.assertGreater(tuned, 0)
        self.assertIsNotNone(engine.partner_page())
        engine.drop_partner()
        self.assertIsNone(engine.partner)
        self.assertIsNone(engine.partner_page())
        self.assertFalse(partner.muted[-1])  # (its own again)
        _run(engine, main, partner)  # (its blocks are ignored now)
        self.assertEqual(self.finished, ["complete"])
        self.assertEqual(len(partner.tunes), tuned)
        self.assertEqual({f.frequency_hz for f in self.found}, _fm_stations())

    def test_a_step_the_partner_refuses_is_swept_by_the_main_receiver(self):
        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(2.4e6), refuse_above=100e6)
        engine = self.engine(main)
        engine.start(self.fm_band(), 2.4e6, 0.0, partner=partner)
        _run(engine, main, partner)
        self.assertEqual(self.finished, ["complete"])
        self.assertGreater(engine.steps_refused, 0)
        self.assertEqual({f.frequency_hz for f in self.found}, _fm_stations())
        self.assertTrue(all(lo <= 100e6 for lo in partner.tunes))
        self.assertGreater(max(main.tunes), 100e6)

    def test_each_receiver_stays_inside_what_it_tunes(self):
        main = _Receiver(_demo(2.4e6), tunes=(24e6, 99e6))
        partner = _Receiver(_demo(2.4e6), tunes=(96e6, 1766e6))
        engine = self.engine(main)
        engine.start(self.fm_band(), 2.4e6, 0.0, partner=partner)
        _run(engine, main, partner)
        self.assertEqual(self.finished, ["complete"])
        self.assertTrue(all(lo <= 99e6 for lo in main.tunes))
        self.assertTrue(all(lo >= 96e6 for lo in partner.tunes))
        self.assertEqual({f.frequency_hz for f in self.found}, _fm_stations())
        # (What no one sweeps once the partner leaves.)
        self.assertEqual(engine.partner_only(), [(99.2e6, 108e6)])

    def test_stop_on_signal_and_bookmark_scans_use_the_main_receiver_only(self):
        from sdr_module.gui.scan_engine import ScanSettings

        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(2.4e6))
        engine = self.engine(main)
        engine.start(
            ScanSettings("stop", 88e6, 90e6, 200e3), 2.4e6, 0.0, partner=partner
        )
        self.assertIsNone(engine.partner)
        _run(engine, main, partner, seconds=0.3)
        self.assertEqual(partner.tunes, [])
        engine.stop()

    def test_a_partner_without_blocks_is_counted_quiet(self):
        main = _Receiver(_demo(2.4e6))
        partner = _Receiver(_demo(2.4e6))
        engine = self.engine(main)
        engine.start(self.fm_band(), 2.4e6, 0.0, partner=partner)
        engine.partner_frame(_block(partner))
        self.assertLess(engine.partner_quiet_s(), 0.5)
        with mock.patch(
            "sdr_module.gui.scan_engine.time.monotonic",
            return_value=engine._shares[1].last_block_at + 7.0,
        ):
            self.assertAlmostEqual(engine.partner_quiet_s(), 7.0, delta=0.01)
        engine.stop()
        self.assertEqual(engine.partner_quiet_s(), 0.0)


# --------------------------------------------------------------------------- #
# The tab and the window
# --------------------------------------------------------------------------- #
class _BothCase(_WindowTestCase):
    """The main window (demo) and its Second Receiver window on a second
    demo device at 3.2 MS/s, the display loops driven by hand."""

    demo = True

    def setUp(self):
        super().setUp()
        self.panel = self.win._scanner_panel
        self.win._show_panel("scanner")
        self.rx = None

    def tearDown(self):
        rx = self.win._second_rx.receiver if HAS_PYQT6 else None
        if rx is not None:
            rx.shutdown()
        super().tearDown()

    def open_second(self, start=True):
        rx = self.win._second_rx.show()
        rx._status_timer.stop()
        index = rx._rate_combo.findData(3.2e6)
        rx._rate_combo.setCurrentIndex(index)
        rx._on_rate_activated(index)
        if start:
            self.assertTrue(rx.start())
            rx._frame_timer.stop()
            # A display frame's worth of samples, however fast the test runs.
            rx._read = lambda: rx.device.read_samples(int(rx._rate() * FRAME_S))
        self.rx = rx
        self.settle()
        return rx

    def frames(self, seconds, until=None, second=True):
        win = self.win
        for _ in range(int(seconds / FRAME_S)):
            win._pacer.reset()
            win._process_block(
                win._device.read_samples(int(win._device_sample_rate() * FRAME_S))
            )
            if second and self.rx is not None and self.rx.running:
                self.rx._frame()
            if until is not None and until():
                return True
        return False

    def fm_sweep(self):
        panel = self.panel
        panel._preset.setCurrentIndex(1)  # FM broadcast, 200 kHz channels
        panel.set_scan_mode("once")


class TestUseBothReceivers(_BothCase):
    def test_offered_while_a_second_receiver_runs_for_range_sweeps(self):
        panel = self.panel
        panel.refresh()
        self.assertTrue(panel._both.isHidden())  # (no second receiver)
        rx = self.open_second(start=False)
        panel.refresh()
        self.assertFalse(panel._both.isHidden())
        self.assertFalse(panel._both.isEnabled())
        self.assertIn("Start", panel._both.toolTip())
        rx.start()
        rx._frame_timer.stop()
        panel.refresh()
        self.assertTrue(panel._both.isEnabled())
        self.assertIn("second receiver", panel._both.toolTip())
        for mode, shown in (
            ("stop", False),
            ("bookmarks", False),
            ("continuous", True),
        ):
            panel.set_scan_mode(mode)
            self.assertEqual(not panel._both.isHidden(), shown, mode)
        rx.stop()
        panel.refresh()
        self.assertFalse(panel._both.isEnabled())

    def test_the_choice_is_remembered(self):
        self.open_second()
        self.panel.refresh()
        self.panel._both.setChecked(True)
        self.assertTrue(self.store.values.get("scanner/both_receivers"))
        from sdr_module.gui.scanner_panel import ScannerPanel

        again = ScannerPanel(settings=self.store)
        try:
            self.assertTrue(again._both.isChecked())
        finally:
            again.deleteLater()

    def test_a_sweep_with_both_finds_what_one_finds_and_puts_both_back(self):
        win, panel = self.win, self.panel
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        rx = self.open_second()
        self.assertTrue(rx.tune(146.52e6))
        self.assertTrue(rx.tune(146.94e6))  # (the VFO moves within its span)
        rx_lo = rx.lo_frequency()
        self.assertNotAlmostEqual(rx_lo, 146.94e6)
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        self.assertIn("both receivers", panel._hint.text())
        played = []
        real_job = rx._frame_job

        def job(samples):
            played.append(rx._audio_open and rx._plays_here() and not rx._scan_muted)
            return real_job(samples)

        rx._frame_job = job
        tunes = []
        real_tune = rx.tune
        rx.tune = lambda freq, recenter=False: (
            tunes.append(freq) or real_tune(freq, recenter)
        )
        self.assertTrue(panel.start_scan())
        self.assertIs(panel.engine.partner.receiver, rx)
        self.assertTrue(win._scan_muted)
        self.assertTrue(rx._scan_muted)
        done = self.frames(5.0, until=lambda: not panel.engine.active)
        self.assertTrue(done)
        self.assertEqual(len(panel.hits()), 18)
        self.assertIn(
            "Scan complete: 18 signals found with both receivers", panel._status.text()
        )
        self.assertGreater(len(tunes), 4)  # (it swept part of the band)
        self.assertTrue(all(88e6 <= f <= 108e6 for f in tunes[:-2]))
        self.assertFalse(any(played[1:-2]))  # muted while it swept
        # Both back where they were, unmuted, no longer fed to the scan.
        self.assertAlmostEqual(win._current_frequency(), 100.1e6)
        self.assertFalse(win._scan_muted)
        self.assertAlmostEqual(rx.tuned_frequency(), 146.94e6)
        self.assertAlmostEqual(rx.lo_frequency(), rx_lo)
        self.assertFalse(rx._scan_muted)
        self.assertIsNone(rx._scan_feed)

    def test_both_take_half_the_frames_of_one(self):
        panel = self.panel
        self.open_second()
        self.fm_sweep()
        panel.refresh()
        counts = []
        for both in (False, True):
            panel._both.setChecked(both)
            self.assertTrue(panel.start_scan())
            frames = 0
            while panel.engine.active and frames < 400:
                self.frames(FRAME_S)
                frames += 1
            counts.append(frames)
            self.assertEqual(len(panel.hits()), 18)
        self.assertLessEqual(counts[1], 0.7 * counts[0], counts)

    def test_the_second_receiver_stopping_leaves_the_main_to_finish(self):
        win, panel = self.win, self.panel
        rx = self.open_second()
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        said = []
        real_show = win._show_status_message
        win._show_status_message = lambda text, *a, **k: (
            said.append(text) or real_show(text, *a, **k)
        )
        self.assertTrue(panel.start_scan())
        self.frames(6 * FRAME_S)
        self.assertTrue(panel.engine.active)
        rx.stop()
        panel.refresh()
        self.assertIsNone(panel.engine.partner)
        self.assertTrue(any("second receiver" in t.lower() for t in said), said)
        self.assertTrue(self.frames(8.0, until=lambda: not panel.engine.active))
        self.assertEqual(len(panel.hits()), 18)
        self.assertIn("Scan complete: 18 signals found.", panel._status.text())

    def test_tuning_the_second_receiver_takes_it_out_of_the_scan(self):
        panel = self.panel
        rx = self.open_second()
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        self.frames(4 * FRAME_S)
        self.assertTrue(rx.tune(446.0e6, recenter=True))  # (the user)
        self.frames(2 * FRAME_S)
        self.assertIsNone(panel.engine.partner)
        self.assertIsNone(rx._scan_feed)
        self.assertFalse(rx._scan_muted)
        self.assertTrue(self.frames(8.0, until=lambda: not panel.engine.active))
        self.assertEqual(len(panel.hits()), 18)
        # It stays where it was tuned.
        self.assertAlmostEqual(rx.tuned_frequency(), 446.0e6)

    def test_a_receiver_tuned_elsewhere_just_before_the_end_stays_there(self):
        """The user (or Follow transmissions) tunes the second receiver and
        the scan ends before its next block: it isn't put back."""
        panel = self.panel
        rx = self.open_second()
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        self.frames(3 * FRAME_S)
        self.assertTrue(rx.tune(446.0e6, recenter=True))
        panel.stop_scan()
        self.assertFalse(panel.engine.active)
        self.assertAlmostEqual(rx.tuned_frequency(), 446.0e6)
        self.assertAlmostEqual(rx.lo_frequency(), 446.0e6)
        self.assertIsNone(rx._scan_feed)
        self.assertFalse(rx._scan_muted)

    def test_without_the_choice_the_second_receiver_is_left_alone(self):
        panel = self.panel
        rx = self.open_second()
        self.assertTrue(rx.tune(146.52e6))
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(False)
        self.assertTrue(panel.start_scan())
        self.assertIsNone(panel.engine.partner)
        self.frames(1.0)
        self.assertAlmostEqual(rx.tuned_frequency(), 146.52e6)
        self.assertFalse(rx._scan_muted)
        panel.stop_scan()


class _FakePartner(_Receiver):
    """A ScanPartner the tab can attach and detach (as the window's
    SecondReceiverPartner)."""

    def __init__(self, dev, tunes=None):
        super().__init__(dev, tunes=tunes)
        self.feed = self.left = None
        self.detached = []
        self.trouble = ""

    def problem(self):
        return self.trouble

    def attach(self, feed, left):
        self.feed, self.left = feed, left

    def detach(self, restore=True):
        self.feed = self.left = None
        self.detached.append(restore)


class TestTheTabWithFakes(unittest.TestCase):
    """The tab over fake receivers: an RTL-SDR-like main receiver (up to
    1766 MHz) and a second one that tunes higher."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.scanner_panel import ScannerPanel
        from tests.test_gui_scanner import FakeWindow

        self.host = FakeWindow()
        self.host.range = (24e6, 1766e6)
        self.partner = _FakePartner(_demo(2.4e6, 1800e6), tunes=(1e6, 3000e6))
        self.offer = (self.partner, "")
        self.host.scan_partner = lambda: self.offer
        self.panel = ScannerPanel(settings=_FakeSettings())
        self.panel.set_host(self.host)
        self.panel.show()
        QApplication.processEvents()

    def tearDown(self):
        self.panel.engine.stop()
        self.panel.close()
        self.panel.deleteLater()
        QApplication.processEvents()

    def test_the_range_is_what_either_receiver_tunes(self):
        panel = self.panel
        panel._start.setValue(1700.0)
        panel._end.setValue(1900.0)
        panel._step.setValue(100.0)
        self.assertIn("Limited to the device's range", panel._hint.text())
        panel._both.setChecked(True)
        self.assertNotIn("Limited", panel._hint.text())
        self.assertIn("with both receivers", panel._hint.text())
        panel._end.setValue(3500.0)
        self.assertIn("Limited to the receivers' range", panel._hint.text())
        self.assertIn("to 3000.000 MHz", panel._hint.text())
        panel._start.setValue(3200.0)
        self.assertIn(
            "The two receivers tune 1000 kHz to 3000.000 MHz", panel._hint.text()
        )

    def test_leaving_names_what_only_it_tunes_and_the_main_finishes(self):
        panel = self.panel
        panel._start.setValue(1700.0)
        panel._end.setValue(1900.0)
        panel._step.setValue(100.0)
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        self.assertIsNotNone(self.partner.feed)
        for _ in range(3):
            panel.engine.frame(_block(self.host))
            self.partner.feed(_block(self.partner))
        self.partner.trouble = "it was stopped"
        panel.refresh()
        self.assertEqual(self.partner.detached, [False])
        self.assertIsNone(panel.engine.partner)
        text, tone = self.host.messages[-1]
        self.assertIn("The second receiver left the scan (it was stopped)", text)
        self.assertIn("except 1766.100 MHz – 1900.000 MHz, which only it tunes", text)
        for _ in range(200):
            if not panel.engine.active:
                break
            panel.engine.frame(_block(self.host))
        self.assertFalse(panel.engine.active)
        self.assertIn("Scan complete", panel._status.text())
        self.assertNotIn("both receivers", panel._status.text())
        self.assertEqual(self.partner.detached, [False])  # (not put back)

    def test_a_scan_that_cannot_start_puts_the_second_receiver_back(self):
        panel = self.panel
        panel._both.setChecked(True)
        self.host.scan_sample_rate = lambda: 0.0  # (the engine refuses to start)
        self.assertFalse(panel.start_scan())
        self.assertEqual(self.partner.detached, [True])
        self.assertIsNone(panel._partner)

    def test_a_receiver_offered_but_stopped_since_is_not_used(self):
        panel = self.panel
        panel._both.setChecked(True)
        self.partner.trouble = "it was stopped"
        self.assertTrue(panel.start_scan())
        self.assertIsNone(panel.engine.partner)
        self.assertIsNone(self.partner.feed)
        panel.stop_scan()


class TestTheChoiceWithoutAWindow(unittest.TestCase):
    """The tab alone (a host without a second receiver): no choice shown."""

    def test_no_choice_without_a_second_receiver(self):
        require_pyqt6(self)
        from sdr_module.gui.scanner_panel import ScannerPanel

        panel = ScannerPanel(settings=_FakeSettings())
        panel.show()
        QApplication.processEvents()
        try:
            panel.refresh()
            self.assertTrue(panel._both.isHidden())
        finally:
            panel.close()
            panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
