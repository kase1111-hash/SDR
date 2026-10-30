#!/usr/bin/env python3
"""
The Scanner tab (T8): a non-modal scanner driven by the display loop.

* The engine (gui/scan_engine.py): sweeps page by page, stops on signals,
  scans bookmarks, mutes between channels, survives errors.
* The tab (gui/scanner_panel.py): presets with modes, Auto threshold,
  estimates and the long-scan question, empty states, results (merge,
  listen, bookmark, export, kept between sessions), no-device buttons,
  recording.
* The window: Ctrl+F / Ctrl+8 show the tab, the display keeps running,
  hits are heard in their band's mode, recording and manual tuning stop a
  scan cleanly.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_scanner.py``
"""

import csv
import json
import os
import tempfile
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
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

FRAME_S = 0.034


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _demo(freq=100.1e6):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_frequency(freq)
    dev.start_rx()
    return dev


class _DemoHost:
    """A ScanHost over the demo device: the 'squelch' is open while the
    tuned frequency is one of ``active``."""

    def __init__(self, dev, active=()):
        self.dev = dev
        self.lo = dev.frequency
        self.vfo = dev.frequency
        self.active = list(active)
        self.carriers = []  # squelch open, but the tone squelch says no
        self.muted = []
        self.tunes = []
        self.moved = True
        self.refuse_above = None

    def scan_tune_lo(self, center, vfo=None):
        if self.refuse_above is not None and center > self.refuse_above:
            return False
        self.dev.set_frequency(center)
        self.lo = center
        self.vfo = center if vfo is None else vfo
        self.moved = True
        return True

    def scan_tune_channel(self, freq, listening, tone=0.0):
        self.tunes.append((freq, tuple(listening), tone))
        self.moved = abs(freq - self.lo) > 0.4 * self.dev.sample_rate
        if self.moved:
            self.dev.set_frequency(freq)
            self.lo = freq
        self.vfo = freq
        return True

    def scan_lo_moved(self):
        return self.moved

    def scan_lo_hz(self):
        return self.lo

    def scan_signal_present(self):
        return any(abs(self.vfo - f) < 2e3 for f in self.active)

    def scan_squelch_open(self):
        return self.scan_signal_present() or any(
            abs(self.vfo - f) < 2e3 for f in self.carriers
        )

    def scan_set_muted(self, muted):
        self.muted.append(bool(muted))


def _run(engine, dev, seconds=10.0, until=None):
    """Feed the engine demo blocks (about 30 per second of signal)."""
    frames = 0
    for _ in range(int(seconds / FRAME_S)):
        if not engine.active or (until is not None and until()):
            break
        engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
        frames += 1
    return frames


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class TestScanEngine(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def engine(self, host, **listening):
        from sdr_module.gui.scan_engine import ScanEngine

        engine = ScanEngine(host)
        self.found, self.finished = [], []
        engine.on_found = self.found.append
        engine.on_finished = self.finished.append
        return engine

    def test_sweep_once_finds_the_fm_band_in_about_two_frames_per_step(self):
        from sdr_module.gui.scan_engine import ScanSettings

        dev = _demo()
        host = _DemoHost(dev)
        engine = self.engine(host)
        settings = ScanSettings("once", 88e6, 108e6, 200e3, min_width_hz=30e3)
        self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
        frames = _run(engine, dev)
        self.assertEqual(self.finished, ["complete"])
        self.assertLessEqual(frames, 2 * len(engine.pages) + 2)
        stations = {
            f
            for f, kind, _l, _d in __import__(
                "sdr_module.gui.device_dialog", fromlist=["_DEMO_SIGNALS"]
            )._DEMO_SIGNALS
            if kind == "wfm"
        }
        reported = {f.frequency_hz for f in self.found}
        self.assertEqual(stations, reported)  # (each on its 100 kHz channel)
        # Muted while it sweeps, unmuted at the end.
        self.assertTrue(host.muted[0])
        self.assertFalse(host.muted[-1])
        self.assertEqual(engine.noise_db, engine.noise_db)  # measured
        self.assertAlmostEqual(engine.noise_db, -88.0, delta=2.0)

    def test_continuous_sweeps_start_again(self):
        from sdr_module.gui.scan_engine import ScanSettings

        dev = _demo()
        engine = self.engine(_DemoHost(dev))
        engine.start(
            ScanSettings("continuous", 162.4e6, 162.55e6, 25e3), dev.sample_rate, 0.0
        )
        _run(engine, dev, seconds=0.5)
        self.assertTrue(engine.active)
        self.assertGreaterEqual(engine.pass_number, 3)
        engine.stop()
        self.assertEqual(self.finished, ["stopped"])
        noaa = sorted({round(f.frequency_hz) for f in self.found})
        self.assertEqual(noaa, [162400000, 162475000, 162550000])

    def test_stop_on_signal_listens_then_goes_on_after_the_quiet_time(self):
        from sdr_module.gui.scan_engine import DWELL, LISTEN, ScanSettings

        dev = _demo(146e6)
        host = _DemoHost(dev, active=[146.52e6])
        engine = self.engine(host)
        settings = ScanSettings(
            "stop",
            146.4e6,
            146.6e6,
            12.5e3,
            listening=("FM", "5 kHz", "25 kHz"),
            resume_s=0.5,
        )
        engine.start(settings, dev.sample_rate, 0.0)
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        self.assertEqual(engine.state, LISTEN)
        self.assertEqual(engine.current.frequency_hz, 146.52e6)
        self.assertEqual(host.tunes[-1], (146.52e6, ("FM", "5 kHz", "25 kHz"), 0.0))
        self.assertFalse(host.muted[-1])  # heard
        self.assertEqual([f.frequency_hz for f in self.found], [146.52e6])
        self.assertTrue(self.found[0].heard)
        # It stays while the squelch is open...
        _run(engine, dev, seconds=1.0)
        self.assertEqual(engine.state, LISTEN)
        # ...and goes on half a second after it closes.
        host.active = []
        _run(engine, dev, seconds=0.45)
        self.assertIn(engine.state, (LISTEN, DWELL))
        _run(engine, dev, seconds=0.2)
        self.assertNotEqual(engine.state, LISTEN)
        self.assertTrue(engine.active)

    def test_hold_stays_and_skip_passes_the_channel_by(self):
        from sdr_module.gui.scan_engine import LISTEN, ScanSettings

        dev = _demo(146e6)
        host = _DemoHost(dev, active=[146.52e6])
        engine = self.engine(host)
        engine.start(
            ScanSettings("stop", 146.4e6, 146.6e6, 12.5e3, resume_s=0.2),
            dev.sample_rate,
            0.0,
        )
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        engine.hold(True)
        host.active = []  # quiet, but held
        _run(engine, dev, seconds=1.0)
        self.assertEqual(engine.state, LISTEN)
        self.assertTrue(engine.held)
        host.active = [146.52e6]
        engine.skip()
        self.assertFalse(engine.held)
        listened = []
        for _ in range(int(3.0 / FRAME_S)):
            engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
            if engine.state == LISTEN:
                listened.append(engine.current.frequency_hz)
        self.assertNotIn(146.52e6, listened)  # passed by for this scan
        self.assertGreaterEqual(engine.pass_number, 2)

    def test_bookmark_scan_visits_each_in_its_mode_and_stops_on_the_active(self):
        from sdr_module.gui.scan_engine import LISTEN, ScanSettings, bookmark_scan_list

        bookmarks = [
            {"label": "Tower", "freq_hz": 118.1e6, "demod": "AM"},
            {"label": "Skipped", "freq_hz": 119.1e6, "demod": "AM", "skip": "S"},
            {"label": "Approach", "freq_hz": 125.0e6, "demod": "AM"},
            {
                "label": "Rptr",
                "freq_hz": 145.29e6,
                "demod": "FM",
                "fm_deviation": "5 kHz",
                "tone_mode": "TSQL",
                "ctone": 100.0,
                "group": "Ham",
            },
        ]

        def listening(b):
            tone = float(b.get("ctone") or 0.0) if b.get("tone_mode") == "TSQL" else 0.0
            return b.get("demod", ""), b.get("fm_deviation", ""), "", tone

        visits = bookmark_scan_list(bookmarks, listening)
        self.assertEqual([v["label"] for v in visits], ["Tower", "Approach", "Rptr"])
        self.assertEqual(visits[1]["index"], 2)
        self.assertEqual(
            [v["label"] for v in bookmark_scan_list(bookmarks, listening, "Ham")],
            ["Rptr"],
        )
        dev = _demo(118e6)
        host = _DemoHost(dev, active=[125.0e6])
        engine = self.engine(host)
        engine.start(
            ScanSettings("bookmarks", bookmarks=visits, resume_s=0.3),
            dev.sample_rate,
            0.0,
        )
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        self.assertEqual(engine.current.label, "Approach")
        self.assertEqual(
            [t[:2] for t in host.tunes[:2]],
            [(118.1e6, ("AM", "", "")), (125.0e6, ("AM", "", ""))],
        )
        self.assertEqual([f.label for f in self.found], ["Approach"])
        self.assertGreater(self.found[0].peak_db, -70)  # its measured level
        # Quiet again: on to the repeater, with its CTCSS tone.
        host.active = []
        _run(engine, dev, seconds=0.5)
        self.assertEqual(host.tunes[2], (145.29e6, ("FM", "5 kHz", ""), 100.0))

    def test_skip_passes_a_bookmark_by_for_the_rest_of_the_scan(self):
        """Skip in a bookmark scan does what its tooltip says (as in Stop on
        signal): that bookmark isn't stopped on again during this scan. A
        carrier that never stops used to hold the scan there every pass."""
        from sdr_module.gui.scan_engine import ALL_SKIPPED, LISTEN, ScanSettings

        bookmarks = [
            {"label": "A", "freq_hz": 118.1e6, "demod": "AM"},
            {"label": "B", "freq_hz": 125.0e6, "demod": "AM"},
        ]
        dev = _demo(118e6)
        host = _DemoHost(dev, active=[118.1e6, 125.0e6])
        engine = self.engine(host)
        engine.start(
            ScanSettings("bookmarks", bookmarks=bookmarks, resume_s=0.2),
            dev.sample_rate,
            0.0,
        )
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        self.assertEqual(engine.current.label, "A")
        engine.skip()
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        self.assertEqual(engine.current.label, "B")
        # B goes quiet: the scan goes on, passing A by.
        host.active = [118.1e6]
        visited = set()
        for _ in range(int(2.0 / FRAME_S)):
            engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
            visited.add(engine.current.label)
        self.assertEqual(visited, {"B"})
        self.assertGreaterEqual(engine.pass_number, 3)
        # Skipping the last one ends the scan, saying why.
        host.active = [125.0e6]
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        engine.skip()
        self.assertFalse(engine.active)
        self.assertEqual(self.finished, [ALL_SKIPPED])
        # A new scan visits them all again.
        engine.start(
            ScanSettings("bookmarks", bookmarks=bookmarks), dev.sample_rate, 0.0
        )
        self.assertEqual(engine.current.label, "A")
        engine.stop()

    def test_a_carrier_without_the_tone_gets_time_then_is_passed_by(self):
        from sdr_module.gui.scan_engine import DWELL, TONE_WAIT_S, ScanSettings

        bookmarks = [
            {"label": "A", "freq_hz": 145.29e6, "demod": "FM", "tone_hz": 100.0},
            {"label": "B", "freq_hz": 146.52e6, "demod": "FM"},
        ]
        dev = _demo(146e6)
        host = _DemoHost(dev)
        host.carriers = [145.29e6]  # the squelch opens, the tone never comes
        engine = self.engine(host)
        engine.start(
            ScanSettings("bookmarks", bookmarks=bookmarks), dev.sample_rate, 0.0
        )
        self.assertEqual(engine.current.label, "A")
        _run(engine, dev, seconds=TONE_WAIT_S - 0.15)
        self.assertEqual((engine.state, engine.current.label), (DWELL, "A"))
        _run(engine, dev, seconds=0.3)
        self.assertEqual(engine.current.label, "B")
        self.assertEqual(self.found, [])

    def test_refused_steps_are_skipped_and_all_refused_stops(self):
        from sdr_module.gui.scan_engine import ScanSettings

        dev = _demo()
        host = _DemoHost(dev)
        host.refuse_above = 100e6
        engine = self.engine(host)
        engine.start(ScanSettings("once", 98e6, 104e6, 200e3), dev.sample_rate, 0.0)
        _run(engine, dev)
        self.assertEqual(self.finished, ["complete"])
        self.assertGreater(engine.steps_refused, 0)
        self.assertTrue(all(f.frequency_hz < 101e6 for f in self.found))
        host.refuse_above = 0.0
        error = engine.start(
            ScanSettings("continuous", 98e6, 104e6, 200e3), dev.sample_rate, 0.0
        )
        self.assertFalse(engine.active)
        self.assertIn("could not tune", error)
        self.assertEqual(self.finished, ["complete"])  # (only the first scan's)
        self.assertFalse(host.muted[-1])

    def test_short_blocks_wait_and_errors_stop_cleanly(self):
        from sdr_module.gui import scan_engine
        from sdr_module.gui.scan_engine import MEASURE, ScanSettings

        dev = _demo()
        engine = self.engine(_DemoHost(dev))
        engine.start(ScanSettings("once", 100e6, 101e6, 200e3), dev.sample_rate, 0.0)
        engine.frame(dev.read_samples(1000))  # the block after the retune
        for _ in range(5):
            engine.frame(np.ones(1, np.complex64))  # too short to measure
        self.assertEqual(engine.state, MEASURE)
        with (
            mock.patch.object(
                scan_engine, "measure_page", side_effect=ValueError("bad")
            ),
            self.assertLogs("sdr_module.gui.scan_engine", "WARNING"),
        ):
            _run(engine, dev, seconds=0.5)
        self.assertFalse(engine.active)
        self.assertEqual(self.finished, ["error: bad"])

    def test_an_error_while_skipping_or_starting_ends_the_scan(self):
        from sdr_module.gui.scan_engine import LISTEN, ScanSettings

        dev = _demo(146e6)
        host = _DemoHost(dev, active=[146.52e6])
        engine = self.engine(host)
        engine.start(
            ScanSettings("stop", 146.4e6, 146.6e6, 12.5e3), dev.sample_rate, 0.0
        )
        _run(engine, dev, until=lambda: engine.state == LISTEN)
        host.scan_tune_lo = mock.Mock(side_effect=RuntimeError("tuner gone"))
        with self.assertLogs("sdr_module.gui.scan_engine", "WARNING"):
            engine.skip()  # the next page's retune fails
        self.assertFalse(engine.active)
        self.assertEqual(self.finished[-1], "error: tuner gone")
        with self.assertLogs("sdr_module.gui.scan_engine", "WARNING"):
            error = engine.start(
                ScanSettings("once", 100e6, 101e6, 200e3), dev.sample_rate, 0.0
            )
        self.assertEqual(error, "error: tuner gone")
        self.assertFalse(engine.active)

    def test_hardware_settles_longer(self):
        from sdr_module.gui.scan_engine import MEASURE, SETTLE, ScanSettings

        dev = _demo()
        engine = self.engine(_DemoHost(dev))
        engine.start(ScanSettings("once", 100e6, 101e6, 200e3), dev.sample_rate, 0.12)
        block = lambda: dev.read_samples(int(dev.sample_rate * FRAME_S))  # noqa: E731
        engine.frame(block())  # dropped: may predate the retune
        for _ in range(3):
            engine.frame(block())
            self.assertEqual(engine.state, SETTLE)
        engine.frame(block())
        self.assertEqual(engine.state, MEASURE)

    def test_bad_settings(self):
        from sdr_module.gui.scan_engine import ScanSettings

        dev = _demo()
        engine = self.engine(_DemoHost(dev))
        self.assertIn(
            "End must be above",
            engine.start(ScanSettings("once", 108e6, 88e6), dev.sample_rate, 0),
        )
        self.assertIn(
            "no bookmarks", engine.start(ScanSettings("bookmarks"), dev.sample_rate, 0)
        )
        self.assertFalse(engine.active)


# --------------------------------------------------------------------------- #
# The tab, with a fake window
# --------------------------------------------------------------------------- #
class FakeWindow:
    """A ScannerHost: a demo device behind it, everything recorded."""

    def __init__(self):
        from sdr_module.gui.scanner_panel import ScannerHost

        self.base = ScannerHost
        self.dev = _demo()
        self.blocker = ("", "")
        self.receiving = True
        self.started = 0
        self.range = None
        self.listened = []
        self.bookmarked = []
        self.known = set()
        self.messages = []
        self.begun = 0
        self.ended = []
        self.demo = self.connect = 0
        self.bookmarks = []
        self.groups = []
        self.noise = -88.0
        self.settle = 0.0
        self.lo = self.dev.frequency
        self.muted = False

    # ScanHost
    def scan_tune_lo(self, center, vfo=None):
        self.dev.set_frequency(center)
        self.lo = center
        return True

    def scan_tune_channel(self, freq, listening, tone=0.0):
        return True

    def scan_lo_moved(self):
        return False

    def scan_lo_hz(self):
        return self.lo

    def scan_signal_present(self):
        return False

    def scan_squelch_open(self):
        return False

    def scan_set_muted(self, muted):
        self.muted = muted

    # ScannerHost
    def scan_blocker(self):
        return self.blocker

    def scan_receiving(self):
        return self.receiving

    def scan_start_receiver(self):
        self.started += 1
        self.receiving = True
        return True

    def scan_sample_rate(self):
        return self.dev.sample_rate

    def scan_settle_s(self):
        return self.settle

    def scan_noise_floor_db(self):
        return self.noise

    def scan_device_range(self):
        return self.range

    def scan_begin(self):
        self.begun += 1

    def scan_end(self, stay):
        self.ended.append(stay)

    def scan_listen(self, hit):
        self.listened.append(hit)
        return True

    def scan_add_bookmarks(self, hits):
        added, existing = 0, []
        for hit in hits:
            if hit.frequency_hz in self.known:
                existing.append(f"{hit.frequency_hz / 1e6:.3f}")
            else:
                self.known.add(hit.frequency_hz)
                self.bookmarked.append(hit)
                added += 1
        return added, existing

    def scan_bookmark_list(self, group=""):
        return [b for b in self.bookmarks if not group or b.get("group") == group]

    def scan_bookmark_groups(self):
        return list(self.groups)

    def scan_bookmark_counts(self):
        return len(self.bookmarks), 0

    def scan_use_demo(self):
        self.demo += 1

    def scan_connect(self):
        self.connect += 1

    def scan_message(self, text, tone=None):
        self.messages.append((text, tone))


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.store = _FakeSettings()
        self.host = FakeWindow()
        self._panels = []

    def tearDown(self):
        for panel in self._panels:
            panel.engine.stop()
            panel.close()
            panel.deleteLater()
        _settle()

    def panel(self, host=None, show=True):
        from sdr_module.gui.scanner_panel import ScannerPanel

        panel = ScannerPanel(settings=self.store)
        panel.set_host(host or self.host)
        panel.resize(420, 560)
        if show:
            panel.show()
            _settle()
        self._panels.append(panel)
        return panel

    def run_scan(self, panel, seconds=10.0):
        dev = self.host.dev
        for _ in range(int(seconds / FRAME_S)):
            if not panel.engine.active:
                break
            panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))


class TestScannerTab(_PanelCase):
    def test_presets_fill_the_range_and_carry_how_hits_are_heard(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        panel = self.panel()
        self.assertEqual(panel._preset.currentText(), SCAN_PRESETS[0].label)
        panel._preset.setCurrentIndex(5)  # Airband AM
        air = SCAN_PRESETS[4]
        self.assertEqual(
            (panel._start.value(), panel._end.value(), panel._step.value()),
            (air.start_mhz, air.end_mhz, air.step_khz),
        )
        self.assertEqual(air.listening, ("AM", "", "10 kHz"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        self.assertEqual(panel.engine.settings.listening, ("AM", "", "10 kHz"))
        self.run_scan(panel)
        hits = {round(h.frequency_hz): h for h in panel.hits()}
        self.assertIn(125000000, hits)
        self.assertEqual(hits[125000000].listening, ("AM", "", "10 kHz"))
        # Editing a field means a custom range.
        panel._end.setValue(130.0)
        self.assertEqual(panel._preset.currentIndex(), 0)

    def test_custom_ranges_guess_the_mode_from_the_band_or_the_width(self):
        from sdr_module.gui.scanner_panel import guess_listening

        self.assertEqual(guess_listening(118.3e6, 8e3), ("AM", "", "10 kHz"))
        self.assertEqual(guess_listening(98.1e6, 150e3), ("FM", "75 kHz", "200 kHz"))
        self.assertEqual(guess_listening(2.5e9, 150e3), ("FM", "75 kHz", "200 kHz"))
        self.assertEqual(guess_listening(2.5e9, 5e3), ("", "", ""))

    def test_auto_threshold_is_on_and_the_hint_shows_the_noise_floor(self):
        panel = self.panel()
        self.assertTrue(panel._auto.isChecked())
        self.assertFalse(panel._threshold.isEnabled())
        self.assertIn("Auto: 12 dB over the noise floor (-88 dBFS)", panel._hint.text())
        self.assertIn("101 channels, 12 receiver steps", panel._hint.text())
        self.assertEqual(panel._threshold.value(), -76)  # the level in use
        panel._auto.setChecked(False)
        self.assertTrue(panel._threshold.isEnabled())
        panel._threshold.setValue(-60)
        self.assertIn("Signals above -60 dBFS", panel._hint.text())
        panel.start_scan()
        self.assertEqual(panel.engine.settings.threshold_db, -60)

    def test_settings_and_results_are_kept_between_sessions(self):
        panel = self.panel()
        panel._preset.setCurrentIndex(2)  # NOAA
        panel.set_scan_mode("continuous")
        panel._auto.setChecked(False)
        panel._threshold.setValue(-70)
        panel._resume.setValue(5.0)
        panel.start_scan()
        self.run_scan(panel, seconds=0.3)
        panel.stop_scan()
        self.assertTrue(panel.hits())
        saved = json.loads(self.store.values["scanner/results"])
        self.assertEqual(len(saved), len(panel.hits()))
        again = self.panel()
        self.assertEqual(again._preset.currentText(), panel._preset.currentText())
        self.assertEqual(again.scan_mode(), "continuous")
        self.assertFalse(again._auto.isChecked())
        self.assertEqual(again._threshold.value(), -70)
        self.assertEqual(again._resume.value(), 5.0)
        self.assertEqual(
            sorted(h.frequency_hz for h in again.hits()),
            sorted(h.frequency_hz for h in panel.hits()),
        )
        self.assertEqual(again._table.rowCount(), len(panel.hits()))

    def test_the_bookmark_group_is_kept_between_sessions(self):
        self.host.groups = ["Airport", "Ham"]
        self.host.bookmarks = [
            {"index": 0, "label": "A", "freq_hz": 118.1e6, "group": "Ham"}
        ]
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        panel._group.setCurrentIndex(panel._group.findData("Ham"))
        self.assertEqual(self.store.values["scanner/bookmark_group"], "Ham")
        again = self.panel()
        self.assertEqual(again._group.currentData(), "Ham")
        self.assertEqual(again.scan_mode(), "bookmarks")

    def test_damaged_saved_results_are_ignored(self):
        self.store.values["scanner/results"] = "not json"
        self.assertEqual(self.panel().hits(), [])
        self.store.values["scanner/results"] = json.dumps(
            [{"x": 1}, {"frequency_hz": -1}]
        )
        self.assertEqual(self.panel().hits(), [])

    def test_neighbouring_readings_merge_into_one_row(self):
        from sdr_module.gui.scan_engine import Found

        panel = self.panel()
        panel._on_found(Found(100.29e6, -35.0, -90.0, 5e3, ("FM", "75 kHz", "")))
        panel._on_found(Found(100.31e6, -30.0, -90.0, 5e3, ("FM", "75 kHz", "")))
        self.assertEqual(panel._table.rowCount(), 1)
        hit = panel.hits()[0]
        self.assertEqual(
            (hit.peak_db, hit.frequency_hz, hit.count), (-30.0, 100.31e6, 2)
        )
        self.assertEqual(panel._table.item(0, 2).text(), "60 dB")

    def test_a_bookmark_scan_lists_each_bookmark_it_heard(self):
        """Two bookmarks on one frequency (a repeater's users with
        different tones) are two results: the second used to be counted
        as the first, under the first one's name."""
        from sdr_module.gui.scan_engine import Found

        visit = {"freq_hz": 145.29e6, "demod": "FM", "fm_deviation": "5 kHz"}
        self.host.bookmarks = [
            dict(visit, index=0, label="Club", bandwidth="", tone_hz=0.0),
            dict(visit, index=1, label="Club 100", bandwidth="", tone_hz=100.0),
        ]
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        for label in ("Club", "Club 100", "Club"):
            panel._on_found(
                Found(145.29e6, -50.0, -90.0, 10e3, ("FM", "5 kHz", ""), label=label)
            )
        self.assertEqual(
            sorted((hit.label, hit.count) for hit in panel.hits()),
            [("Club", 2), ("Club 100", 1)],
        )
        self.assertEqual(panel._table.rowCount(), 2)
        panel.stop_scan()

    def test_empty_states_say_what_happened(self):
        """TOOLS-09: after Clear or an early Stop the table doesn't claim
        that nothing was above the threshold."""
        from sdr_module.gui.scan_engine import Found

        panel = self.panel()
        texts = {"idle": panel._empty.text()}
        panel.start_scan()
        texts["scanning"] = panel._empty.text()
        panel.stop_scan()
        texts["stopped"] = panel._empty.text()
        panel._on_found(Found(100.1e6, -30.0, -90.0, 150e3, ("FM", "", "")))
        self.assertFalse(panel._empty.is_visible())
        panel.clear_results()
        texts["cleared"] = panel._empty.text()
        self.assertTrue(panel._empty.is_visible())
        self.assertEqual(panel._status.text(), "Results cleared.")
        panel._auto.setChecked(False)
        panel._threshold.setValue(0.0)
        panel.start_scan()
        self.run_scan(panel)
        texts["nothing"] = panel._empty.text()
        self.assertTrue(texts["stopped"].startswith("Stopped before any signal"))
        self.assertTrue(texts["cleared"].startswith("Results cleared"))
        self.assertTrue(texts["nothing"].startswith("No signals above the threshold"))
        self.assertTrue(texts["idle"].startswith("No results yet"))
        self.assertTrue(texts["scanning"].startswith("Scanning for signals"))
        for text in texts.values():
            title, _sep, hint = text.partition("\n")
            self.assertTrue(hint, text)
            self.assertFalse(title.endswith("."), title)

    def test_without_a_device_it_offers_the_demo_and_connect(self):
        """MC-U15: the tab itself offers Use Demo Device and Connect..."""
        self.host.blocker = (
            "device",
            "No receiver: use the demo device or connect an SDR to scan.",
        )
        panel = self.panel()
        self.assertTrue(panel._start_btn.isHidden())
        self.assertFalse(panel._demo_btn.isHidden())
        self.assertFalse(panel._connect_btn.isHidden())
        self.assertIn("Connect a device to scan", panel._empty.text())
        self.assertIn("No receiver", panel._status.text())
        panel._demo_btn.click()
        panel._connect_btn.click()
        self.assertEqual((self.host.demo, self.host.connect), (1, 1))
        # Never a fabricated scan without a device.
        self.assertFalse(panel.start_scan())
        self.assertFalse(panel.engine.active)
        self.assertEqual(panel._table.rowCount(), 0)
        # A device arrives: Start Scan is back.
        self.host.blocker = ("", "")
        panel.refresh()
        self.assertFalse(panel._start_btn.isHidden())
        self.assertTrue(panel._demo_btn.isHidden())
        self.assertTrue(panel._start_btn.isEnabled())

    def test_recording_prevents_scanning_and_says_why(self):
        """TOOLS-06 / MC-14: no scan while recording I/Q, with the reason."""
        self.host.blocker = ("recording", "Stop recording to scan: a scan retunes ...")
        panel = self.panel()
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertIn("Stop recording to scan", panel._start_btn.toolTip())
        self.assertIn("Stop recording to scan", panel._status.text())
        self.assertFalse(panel.start_scan())
        # A recording that starts during a scan stops it.
        self.host.blocker = ("", "")
        panel.refresh()
        self.assertTrue(panel.start_scan())
        self.host.blocker = ("recording", "Stop recording to scan")
        panel.refresh()
        self.assertFalse(panel.engine.active)
        self.assertIn("Scan stopped", panel._status.text())

    def test_the_sweep_stays_inside_the_device_range(self):
        self.host.range = (24e6, 1766e6)
        panel = self.panel()
        panel._start.setValue(1700.0)
        panel._end.setValue(1800.0)
        panel._step.setValue(1000.0)
        self.assertIn("67 channels", panel._hint.text())
        self.assertIn("Limited to the device's range", panel._hint.text())
        self.assertTrue(panel._start_btn.isEnabled())
        panel.start_scan()
        self.assertLessEqual(panel.engine.settings.end_hz, 1766e6)
        panel.stop_scan()
        panel._start.setValue(1800.0)
        panel._end.setValue(1900.0)
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertIn("1766.000 MHz", panel._start_btn.toolTip())
        self.assertFalse(panel.start_scan())
        self.assertEqual(panel._status.property("tone"), "danger")

    def test_a_huge_scan_shows_its_time_and_asks_first(self):
        """UX-05: an estimate in the hint, and a question before a scan of
        more than five minutes a pass."""
        from sdr_module.gui.scanner_panel import CONFIRM_SECONDS

        panel = self.panel()
        panel._start.setValue(0.1)
        panel._end.setValue(6000.0)
        panel._step.setValue(1.0)
        channels, steps, seconds = panel.estimate()
        self.assertGreater(seconds, CONFIRM_SECONDS)
        self.assertIn("5,999,901 channels", panel._hint.text())
        self.assertIn("about", panel._hint.text())
        self.assertEqual(panel._hint.property("tone"), "warning")
        with mock.patch.object(panel, "_confirm_long_scan", return_value=False) as ask:
            self.assertFalse(panel.start_scan())
        ask.assert_called_once()
        self.assertFalse(panel.engine.active)
        with mock.patch.object(panel, "_confirm_long_scan", return_value=True):
            self.assertTrue(panel.start_scan())
        # A short scan doesn't ask.
        panel.stop_scan()
        panel._preset.setCurrentIndex(1)
        with mock.patch.object(panel, "_confirm_long_scan") as ask:
            self.assertTrue(panel.start_scan())
        ask.assert_not_called()

    def test_the_long_scan_question_defaults_to_cancel(self):
        panel = self.panel()
        seen = {}

        def answer():
            box = QApplication.activeModalWidget()
            if isinstance(box, QMessageBox):
                seen["default"] = box.defaultButton().text()
                seen["text"] = box.text()
                box.reject()

        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, answer)
        self.assertFalse(panel._confirm_long_scan(3125, 3600.0))
        self.assertEqual(seen["default"], "Cancel")
        self.assertIn("about 1 h", seen["text"])

    def test_starts_the_receiver_when_stopped(self):
        self.host.receiving = False
        panel = self.panel()
        self.assertTrue(panel.start_scan())
        self.assertEqual(self.host.started, 1)
        self.assertEqual(self.host.begun, 1)

    def test_a_scan_that_gets_no_blocks_gives_up(self):
        from sdr_module.gui import scanner_panel

        panel = self.panel()
        panel.start_scan()
        panel.refresh()
        self.assertTrue(panel.engine.active)  # (just started)
        panel.engine.last_block_at -= scanner_panel.NO_BLOCKS_S + 1
        panel.refresh()
        self.assertFalse(panel.engine.active)
        self.assertIn("returned no samples", panel._status.text())
        self.assertEqual(self.host.ended, [False])

    def test_starting_again_keeps_where_the_first_began(self):
        panel = self.panel()
        panel.start_scan()
        panel.start_scan()
        self.assertEqual(self.host.begun, 2)
        self.assertEqual(self.host.ended, [True])  # the first one, on the spot
        panel.stop_scan()
        self.assertEqual(self.host.ended, [True, False])

    def test_the_receiver_stopping_stops_the_scan(self):
        panel = self.panel()
        panel.start_scan()
        self.host.receiving = False
        panel.refresh()
        self.assertFalse(panel.engine.active)
        self.assertIn("the receiver was stopped", panel._status.text())
        self.assertEqual(self.host.ended, [False])  # back where it started

    def test_listen_bookmark_and_export(self):
        panel = self.panel()
        panel._preset.setCurrentIndex(2)  # NOAA
        panel.start_scan()
        self.run_scan(panel)
        self.assertEqual(panel._table.rowCount(), 3)
        self.assertFalse(panel._listen_btn.isEnabled())
        panel._table.selectRow(0)
        self.assertTrue(panel._listen_btn.isEnabled())
        panel.listen_selected()
        self.assertEqual(len(self.host.listened), 1)
        self.assertEqual(self.host.listened[0].listening, ("FM", "5 kHz", "25 kHz"))
        panel.bookmark_selected()
        self.assertEqual(len(self.host.bookmarked), 1)
        panel.bookmark_all()
        self.assertEqual(len(self.host.bookmarked), 3)
        self.assertIn("1 already bookmarked", panel._status.text())
        panel.bookmark_selected()
        self.assertIn("Already bookmarked", panel._status.text())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scan")
            self.assertEqual(panel.export_csv(path), 3)
            with open(path + ".csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(
            [r["Frequency (MHz)"] for r in rows],
            ["162.400000", "162.475000", "162.550000"],
        )
        self.assertEqual(rows[0]["Mode"], "FM")
        self.assertEqual(rows[0]["Band"], "NOAA Weather")
        self.assertTrue(rows[0]["Last heard"])

    def test_listen_stops_a_running_scan_on_the_spot(self):
        panel = self.panel()
        panel._preset.setCurrentIndex(2)
        panel.set_scan_mode("continuous")
        panel.start_scan()
        self.run_scan(panel, seconds=0.3)
        panel._table.selectRow(0)
        panel.listen_selected()
        self.assertFalse(panel.engine.active)
        self.assertEqual(self.host.ended, [True])  # stays (then tunes the hit)
        self.assertEqual(len(self.host.listened), 1)

    def test_enter_and_double_click_listen(self):
        from sdr_module.gui.scan_engine import Found

        panel = self.panel()
        for freq in (100.1e6, 101.1e6, 102.7e6):
            panel._on_found(
                Found(freq, -30.0, -88.0, 150e3, ("FM", "75 kHz", "200 kHz"))
            )
        panel._table.setFocus()
        panel._table.selectRow(1)
        QTest.keyClick(panel._table, Qt.Key.Key_Return)
        self.assertEqual([h.frequency_hz for h in self.host.listened], [101.1e6])
        QTest.keyClick(panel._table, Qt.Key.Key_Down)
        QTest.keyClick(panel._table, Qt.Key.Key_Enter)
        self.assertEqual(self.host.listened[-1].frequency_hz, 102.7e6)
        self.assertFalse(panel.engine.active)  # Enter never starts a scan
        self.assertEqual(panel._table.rowCount(), 3)

    def test_keyboard_and_mnemonics(self):
        import re

        from PyQt6.QtWidgets import QAbstractButton, QLabel

        panel = self.panel()
        self.assertFalse(panel._table.tabKeyNavigation())
        self.assertEqual(panel._table.accessibleName(), "Scan results")
        keys = {}
        for widget in panel.findChildren((QLabel, QAbstractButton)):
            match = re.search(r"&([^&])", widget.text())
            if match:
                keys.setdefault(match.group(1).lower(), []).append(widget.text())
        self.assertEqual({k: v for k, v in keys.items() if len(v) > 1}, {})

    def test_start_button_keeps_its_width(self):
        panel = self.panel()
        width = panel._start_btn.minimumWidth()
        panel.start_scan()
        self.assertEqual(panel._start_btn.text(), "Stop Scan")
        self.assertEqual(panel._start_btn.minimumWidth(), width)
        self.assertGreaterEqual(width, panel._start_btn.sizeHint().width())

    def test_bookmark_mode_controls(self):
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertIn("No bookmarks", panel._start_btn.toolTip())
        self.assertFalse(panel._start.isEnabled())
        self.assertFalse(panel._group.isHidden())
        self.assertFalse(panel._resume.isHidden())
        self.host.bookmarks = [
            {
                "index": 0,
                "label": "A",
                "freq_hz": 146.52e6,
                "demod": "FM",
                "group": "Ham",
            },
            {"index": 3, "label": "B", "freq_hz": 125.0e6, "demod": "AM"},
        ]
        self.host.groups = ["Ham"]
        panel.bookmarks_changed()
        self.assertTrue(panel._start_btn.isEnabled())
        self.assertIn("Visits 2 bookmarks", panel._hint.text())
        self.assertTrue(panel.start_scan(start_index=3))
        self.assertEqual(panel.engine.current.label, "B")
        panel.stop_scan()
        panel._group.setCurrentIndex(1)
        self.assertIn("Visits 1 bookmark", panel._hint.text())
        panel.set_scan_mode("once")
        self.assertTrue(panel._group.isHidden())
        self.assertTrue(panel._resume.isHidden())

    def test_listening_row_skip_and_hold(self):
        from sdr_module.gui.scan_engine import LISTEN

        host = FakeWindow()
        host.scan_signal_present = lambda: True
        panel = self.panel(host)
        panel.set_scan_mode("stop")
        panel._preset.setCurrentIndex(2)
        panel.start_scan()
        dev = host.dev
        for _ in range(60):
            panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
            if panel.engine.state == LISTEN:
                break
        self.assertEqual(panel.engine.state, LISTEN)
        self.assertFalse(panel._listen_row.isHidden())
        self.assertIn("Listening: ", panel._listen_label.text())
        self.assertIn("162.", panel._listen_label.text())
        panel._hold_btn.click()
        self.assertTrue(panel.engine.held)
        self.assertIn("(held)", panel._listen_label.text())
        first = panel.engine.current.frequency_hz
        panel._skip_btn.click()
        self.assertFalse(panel.engine.held)
        self.assertFalse(panel._hold_btn.isChecked())
        self.assertNotEqual(panel.engine.current.frequency_hz, first)
        # Stop while listening: the receiver stays on the channel.
        for _ in range(30):
            panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
            if panel.engine.state == LISTEN:
                break
        panel.toggle_scan()
        self.assertEqual(host.ended, [True])
        self.assertTrue(panel._listen_row.isHidden())

    def test_skipping_every_bookmark_ends_the_scan_and_says_so(self):
        from sdr_module.gui.scan_engine import LISTEN

        host = FakeWindow()
        host.scan_signal_present = lambda: True
        visit = {"demod": "FM", "fm_deviation": "5 kHz", "bandwidth": ""}
        host.bookmarks = [
            dict(visit, index=0, label="A", freq_hz=162.4e6, tone_hz=0.0),
            dict(visit, index=1, label="B", freq_hz=162.55e6, tone_hz=0.0),
        ]
        panel = self.panel(host)
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        dev = host.dev
        for label in ("A", "B"):
            panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
            self.assertEqual(
                (panel.engine.state, panel.engine.current.label), (LISTEN, label)
            )
            panel._skip_btn.click()
        self.assertFalse(panel.engine.active)
        self.assertEqual(
            panel._status.text(),
            "Scan stopped: every bookmark was skipped. 2 signals heard.",
        )
        self.assertEqual(host.ended, [False])  # back where it started
        self.assertIn("Start Scan", panel._start_btn.text())


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
def _pretend_audio(win):
    win._audio_output_problem = lambda: None
    win._audio.start = lambda *_a, **_k: True
    win._audio_enabled = True


class _ScanWindowCase(_WindowTestCase):
    demo = True

    def frames(self, seconds, until=None):
        """Run the display loop (the timer is stopped in these tests)."""
        frames = int(seconds / FRAME_S)
        for _ in range(frames):
            self.win._pacer.reset()
            self.win._process_block(
                self.win._device.read_samples(
                    int(self.win._device_sample_rate() * FRAME_S)
                )
            )
            if until is not None and until():
                return True
        return False


class TestScannerInTheWindow(_ScanWindowCase):
    def test_ctrl_f_shows_the_tab_and_nothing_stops(self):
        """UX-B06 / FG-U08 / UX-02: no modal dialog, the display timer is
        never stopped."""
        self.win._display_timer.start()
        QTest.keyClick(self.win, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.win._current_panel_key(), "scanner")
        self.assertIsNone(QApplication.activeModalWidget())
        self.assertEqual(self.win.findChildren(QDialog), [])
        self.assertTrue(self.win._display_timer.isActive())
        self.win._display_timer.stop()

    def test_ctrl_8_and_the_tab_before_info(self):
        tabs = self.win._right_tabs
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertEqual(titles[-2:], ["Scanner", "Info"])
        keys = [key for key, _t, _tip in self.win._panel_specs]
        self.assertEqual(keys.index("scanner"), 7)  # Ctrl+8
        self.assertEqual(
            keys[:7],
            ["decoder", "bookmarks", "s_meter", "ham_id", "sstv", "qrp", "info"][
                : len(keys[:7])
            ],
        )
        self.win._show_panel("info")
        QTest.keyClick(self.win, Qt.Key.Key_8, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.win._current_panel_key(), "scanner")

    def test_a_sweep_runs_in_the_display_loop_and_restores_the_tuning(self):
        win, panel = self.win, self.win._scanner_panel
        win._show_panel("scanner")
        win.set_frequency(100.1e6)
        win.set_frequency(100.5e6)  # the VFO moves within the span
        lo = win._lo_frequency()
        self.assertNotAlmostEqual(lo, 100.5e6)
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        panel._preset.setCurrentIndex(1)  # FM broadcast
        panel.set_scan_mode("once")
        lines = []
        win._waterfall.add_line = lambda power: lines.append(len(power))
        self.assertTrue(panel.start_scan())
        self.assertTrue(win._scan_muted)
        self.frames(5.0, until=lambda: not panel.engine.active)
        self.assertFalse(panel.engine.active)
        self.assertEqual(len(panel.hits()), 18)
        self.assertIn("Scan complete: 18 signals found", panel._status.text())
        self.assertGreater(len(lines), 20)  # the plots kept running
        # Back where it was (the plots too), unmuted.
        self.assertAlmostEqual(win._current_frequency(), 100.1e6)
        self.assertAlmostEqual(win._lo_frequency(), lo)
        self.assertEqual(win._listening_mode(), ("FM", "75 kHz"))
        self.assertFalse(win._scan_muted)
        # Scan steps didn't fill Recent Frequencies.
        win._commit_pending_history()
        recent = [round(e.freq_hz) for e in win._tuning_history.recent()]
        self.assertLessEqual(len(recent), 2)

    def test_the_tuned_frequency_is_a_channel_while_sweeping(self):
        win, panel = self.win, self.win._scanner_panel
        panel._preset.setCurrentIndex(1)  # FM broadcast, 200 kHz channels
        panel.set_scan_mode("continuous")
        panel.start_scan()
        seen = set()
        for _ in range(12):
            self.frames(FRAME_S)
            seen.add(round(win._current_frequency()))
            # (The LO sits a few bins off the channel: its DC spike.)
            self.assertNotEqual(win._lo_frequency(), win._current_frequency())
        self.assertGreater(len(seen), 3)
        for freq in seen:
            self.assertEqual((freq - 88_000_000) % 200_000, 0, freq)
        panel.stop_scan()

    def test_losing_the_device_says_so(self):
        win, panel = self.win, self.win._scanner_panel
        panel._preset.setCurrentIndex(1)
        panel.start_scan()
        panel.stop_scan()
        self.assertIn("Scan stopped", panel._status.text())
        win._release_device()
        panel.refresh()
        self.assertIn("No receiver", panel._status.text())
        self.assertFalse(panel._demo_btn.isHidden())
        # And a scan running when it goes: stopped, saying why.
        win._start_demo_mode()
        panel.refresh()
        self.assertTrue(panel.start_scan())
        win._release_device()
        panel.refresh()
        self.assertFalse(panel.engine.active)
        self.assertIn("the device was disconnected", panel._status.text())

    def test_a_new_sample_rate_or_a_playback_stops_the_scan(self):
        win, panel = self.win, self.win._scanner_panel
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        panel.start_scan()
        self.frames(0.2)
        win.set_sample_rate(2.048e6)
        self.frames(0.1)
        self.assertFalse(panel.engine.active)
        self.assertIn("sample rate changed", panel._status.text())
        panel.start_scan()
        self.frames(0.1)
        win._lo_lock = (win._lo_frequency(), 2.048e6, "the recording")
        self.frames(0.1)
        self.assertFalse(panel.engine.active)
        self.assertIn("a recording is playing", panel._status.text())
        win._lo_lock = None
        panel.refresh()

    def test_listen_starts_a_stopped_receiver(self):
        from sdr_module.gui.scanner_panel import ScanHit

        win = self.win
        win._stop_acquisition()
        hit = ScanHit(125.0e6, -44.0, 44.0, 8e3, "AM", "", "10 kHz")
        self.assertTrue(win._scanner_host.scan_listen(hit))
        self.assertTrue(win._is_running)
        self.assertEqual(win._control_panel.get_demod_mode(), "AM")

    def test_an_airband_hit_is_heard_in_am(self):
        """TOOLS-07: Listen on a hit of the Airband preset tunes AM 10 kHz
        (it used to keep FM and play silence)."""
        win, panel = self.win, self.win._scanner_panel
        _pretend_audio(win)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        panel._preset.setCurrentIndex(5)  # Airband AM
        panel.start_scan()
        self.frames(5.0, until=lambda: not panel.engine.active)
        rows = {
            panel._table.item(r, 0).text(): r for r in range(panel._table.rowCount())
        }
        self.assertIn("125.000 MHz", rows)
        panel._table.selectRow(rows["125.000 MHz"])
        written = []
        win._audio.write = lambda audio: written.append(np.array(audio, copy=True))
        panel.listen_selected()
        self.assertAlmostEqual(win._current_frequency(), 125.0e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "AM")
        self.assertEqual(win._control_panel._bw_combo.currentText(), "10 kHz")
        self.frames(1.0)
        audio = np.concatenate(written) if written else np.zeros(1)
        self.assertGreater(float(np.sqrt(np.mean(audio**2))), 0.02)

    def test_stop_on_signal_is_heard_while_scanning(self):
        """UX-02 / G-04: the scan stops on an active channel, unmuted, in the
        band's mode, and the speaker plays it."""
        win, panel = self.win, self.win._scanner_panel
        _pretend_audio(win)
        written = []
        win._audio.write = lambda audio: written.append(len(audio))
        panel._preset.setCurrentIndex(2)  # NOAA: continuous stations
        panel.set_scan_mode("stop")
        panel.start_scan()
        from sdr_module.gui.scan_engine import LISTEN

        self.assertTrue(self.frames(5.0, until=lambda: panel.engine.state == LISTEN))
        self.assertFalse(win._scan_muted)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))
        written.clear()
        self.frames(1.0)
        self.assertEqual(panel.engine.state, LISTEN)
        self.assertGreater(sum(written), 30000)  # most of a second of audio
        self.assertFalse(panel._listen_row.isHidden())
        # Stop while listening: it stays on the channel.
        tuned = win._current_frequency()
        panel.toggle_scan()
        self.assertAlmostEqual(win._current_frequency(), tuned)
        self.assertIn(round(tuned), (162400000, 162475000, 162550000))

    def test_recording_blocks_scanning_and_stops_a_running_scan(self):
        """TOOLS-06 / MC-14: no samples silently dropped while the REC clock
        runs: a scan can't start while recording, and a recording started
        during a scan stops it before the receiver is retuned again."""
        win, panel = self.win, self.win._scanner_panel
        win._toggle_recording(True)
        self.assertTrue(win._recording)
        panel.refresh()
        self.assertFalse(panel.start_scan())
        self.assertIn("Stop recording to scan", panel._status.text())
        win._toggle_recording(False)
        panel.refresh()
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        self.frames(0.3)
        win._toggle_recording(True)
        lo = win._lo_frequency()
        before = win._buffer_sample_count()
        self.frames(0.5)
        self.assertFalse(panel.engine.active)
        self.assertIn("recording", panel._status.text())
        panel.refresh()  # (the tab's timer) keeps saying why it stopped
        self.assertIn("Scan stopped", panel._status.text())
        self.assertIn("Stop recording to scan", panel._start_btn.toolTip())
        self.assertAlmostEqual(win._lo_frequency(), lo)  # no more retunes
        self.assertGreater(win._buffer_sample_count(), before)
        # One centre frequency for the whole take, never retuned during it.
        self.assertAlmostEqual(win._buffer_meta[0], lo)
        self.assertFalse(win._retune_warned)
        win._toggle_recording(False)

    def test_tuning_by_hand_stops_the_scan_where_you_tuned(self):
        win, panel = self.win, self.win._scanner_panel
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        panel.start_scan()
        self.frames(0.2)
        win.set_frequency(144.39e6)
        self.frames(0.1)
        self.assertFalse(panel.engine.active)
        self.assertIn("you tuned to 144.390 MHz", panel._status.text())
        self.assertAlmostEqual(win._current_frequency(), 144.39e6)

    def test_closing_during_a_scan_saves_the_frequency_it_started_on(self):
        win, panel = self.win, self.win._scanner_panel
        win.set_frequency(100.1e6)
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        panel.start_scan()
        self.frames(0.3)
        self.assertNotAlmostEqual(win._current_frequency(), 100.1e6)
        win.close()
        self.assertAlmostEqual(self.store.values["frequency_hz"], 100.1e6)

    def test_bookmark_hits_from_the_scanner(self):
        """G-08: hits go to the bookmarks one by one or all, named after
        their band, in their mode, without duplicates."""
        win, panel = self.win, self.win._scanner_panel
        panel._preset.setCurrentIndex(2)  # NOAA
        panel.start_scan()
        self.frames(3.0, until=lambda: not panel.engine.active)
        panel._table.selectRow(0)
        panel.bookmark_selected()
        bookmarks = win._bookmarks_panel.bookmarks()
        self.assertEqual(len(bookmarks), 1)
        self.assertEqual(bookmarks[0]["demod"], "FM")
        self.assertEqual(bookmarks[0]["fm_deviation"], "5 kHz")
        self.assertTrue(bookmarks[0]["label"].startswith("NOAA Weather"))
        panel.bookmark_all()
        self.assertEqual(win._bookmarks_panel.bookmark_count(), 3)
        self.assertIn("1 already bookmarked", panel._status.text())

    def test_bookmark_all_saves_the_bookmarks_once(self):
        """Bookmark All saved and rebuilt the bookmark list once per result:
        with the 500 results kept, the window froze for about 5 s."""
        from sdr_module.gui.scanner_panel import ScanHit

        win, panel = self.win, self.win._scanner_panel
        bookmarks = win._bookmarks_panel
        bookmarks.add_bookmark("Already", 440.0e6, "FM", "5 kHz")
        hits = [
            ScanHit(440.0e6 + i * 25e3, -60.0, 20.0, 5e3, "FM", "5 kHz", "25 kHz")
            for i in range(300)
        ]
        hits.append(ScanHit(440.025e6, -61.0, 19.0, 5e3, "FM", "5 kHz", "25 kHz"))
        panel._hits = hits
        panel._rebuild_table()
        saves = []
        real_save = self.store.set_bookmarks
        self.store.set_bookmarks = lambda items: (
            saves.append(len(items)),
            real_save(items),
        )
        with mock.patch.object(
            bookmarks, "_refresh_list", wraps=bookmarks._refresh_list
        ) as refresh:
            panel.bookmark_all()
        self.assertEqual(saves, [300])  # once, with every new one
        self.assertEqual(refresh.call_count, 1)
        self.assertEqual(bookmarks.bookmark_count(), 300)
        self.assertEqual(
            panel._status.text(),
            "Bookmarked 299 of 301 signals (2 already bookmarked).",
        )
        added = bookmarks.bookmark(1)
        self.assertEqual(
            (added["label"], added["demod"], added["bandwidth"]),
            ("70cm Ham 440.025", "FM", "25 kHz"),
        )
        # One result: the bookmark already there is selected.
        bookmarks._list.clearSelection()
        panel._bookmark([hits[0]])
        self.assertEqual(bookmarks._selected_index(), 0)
        self.assertEqual(panel._status.text(), "Already bookmarked: Already")

    def test_keep_tuned_frequency_centered_keeps_each_step_where_planned(self):
        """A sweep measures each step around the LO it planned. With Radio >
        Keep Tuned Frequency Centered on, putting FREQ on a channel retuned
        the LO onto it, so 2 m signals were listed up to half a step
        (6.25 kHz) off: 147.105 MHz as 147.099 MHz."""
        win, panel = self.win, self.win._scanner_panel
        win._keep_centered_action.setChecked(True)
        panel._preset.setCurrentIndex(3)  # 2m Ham, 12.5 kHz channels
        panel.set_scan_mode("once")
        los = []
        retune = win._retune_lo

        def spy(lo_hz):
            los.append(lo_hz)
            return retune(lo_hz)

        win._retune_lo = spy
        self.assertTrue(panel.start_scan())
        self.frames(10.0, until=lambda: not panel.engine.active)
        self.assertIn("Scan complete", panel._status.text())
        planned = [page.lo_hz for page in panel.engine.pages]
        self.assertEqual(los[: len(planned)], planned)
        found = [round(hit.frequency_hz) for hit in panel.hits()]
        self.assertIn(147_105_000, found)

    def test_a_bookmark_scan_checks_the_tone_with_the_speaker_off(self):
        """A bookmark's CTCSS tone decides whether the scan stops on it,
        speaker on or off: with the audio off the tone wasn't checked, so
        the scan stopped on any carrier there."""
        from sdr_module.gui.scan_engine import LISTEN

        win, panel = self.win, self.win._scanner_panel
        win._audio_enabled = False
        bookmarks = win._bookmarks_panel
        index = bookmarks.add_bookmark("Club, wrong tone", 145.29e6, "FM", "5 kHz")
        bookmarks.set_tone(index, 123.0)  # the demo repeater sends 100.0 Hz
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        self.assertFalse(self.frames(3.0, until=lambda: panel.engine.state == LISTEN))
        self.assertEqual(panel.hits(), [])
        panel.stop_scan()
        # Its own tone: heard (the repeater keeps sending for a few more s).
        bookmarks.set_tone(index, 100.0)
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(3.0, until=lambda: panel.engine.state == LISTEN))
        panel.stop_scan()
        self.assertFalse(win._scan_checks_tone)

    def test_transfers_queued_before_a_retune_are_not_measured(self):
        """A hardware receiver's queued transfers hold the frequency it was
        on: with the display loop behind (a HackRF at a high rate) a sweep
        measured them as the new step's, and listed a carrier at 100.300
        MHz as 100.805 MHz. They are dropped after each retune."""
        import queue

        from sdr_module.devices.base import SDRDevice

        rate, carrier = 2.4e6, 100.3e6

        class Behind(SDRDevice):
            """Each transfer holds the carrier as heard at the LO it was
            made at; the display loop takes them one per frame, 20 late."""

            def __init__(self):
                super().__init__()
                self._state.sample_rate = rate
                self._state.is_streaming = True
                self.lo = 100.0e6
                self.rng = np.random.default_rng(1)

            def transfer(self):
                n = 65536
                t = np.arange(n) / rate
                noise = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
                tone = 0.05 * np.exp(2j * np.pi * (carrier - self.lo) * t)
                try:
                    self._sample_queue.put_nowait(
                        (tone + 1e-4 * noise).astype(np.complex64)
                    )
                except queue.Full:
                    pass

            def set_frequency(self, freq_hz):
                self.lo = float(freq_hz)
                return True

            def open(self, index=0):
                return True

            def close(self):
                pass

            def set_sample_rate(self, rate_hz):
                return True

            def set_bandwidth(self, bw_hz):
                return True

            def set_gain(self, gain_db):
                return True

            def set_gain_mode(self, auto):
                return True

            def start_rx(self, callback=None):
                return True

            def stop_rx(self):
                return True

        win, panel = self.win, self.win._scanner_panel
        dev = Behind()
        for _ in range(20):
            dev.transfer()
        win._device = dev
        panel._preset.setCurrentIndex(0)  # custom: one step, 100-101 MHz
        panel._start.setValue(100.0)
        panel._end.setValue(101.0)
        panel._step.setValue(25.0)
        panel.set_scan_mode("once")
        received = len(win._rx_history)
        self.assertTrue(panel.start_scan())
        for _ in range(100):
            dev.transfer()
            win._pacer.reset()
            win._process_block(dev.read_samples(65536, timeout=0.0))
            if not panel.engine.active:
                break
        self.assertIn("Scan complete", panel._status.text())
        self.assertEqual([round(h.frequency_hz) for h in panel.hits()], [100_300_000])
        # (The dropped transfers count as received.)
        self.assertGreaterEqual(len(win._rx_history) - received, 20)
        win._stop_acquisition()
        win._device = None

    def test_the_tab_leaves_the_menu_bar_mnemonics_alone(self):
        """Alt+R and Alt+H open the Radio and Help menus: with the tab
        shown, Resume and Threshold had them too, so every other press
        moved the focus instead."""
        import re

        from PyQt6.QtWidgets import QAbstractButton, QLabel

        win, panel = self.win, self.win._scanner_panel
        menus = set()
        for action in win.menuBar().actions():
            match = re.search(r"&([^&])", action.text())
            if match:
                menus.add(match.group(1).lower())
        self.assertTrue({"r", "h", "t", "v", "d", "f"} <= menus)
        used = []
        for widget in panel.findChildren((QLabel, QAbstractButton)):
            match = re.search(r"&([^&])", widget.text())
            if match:
                used.append(match.group(1).lower())
                self.assertNotIn(match.group(1).lower(), menus, widget.text())
        self.assertGreaterEqual(len(used), 8)

    def test_scan_bookmarks_from_here(self):
        win = self.win
        bookmarks = win._bookmarks_panel
        bookmarks.add_bookmark("Calling", 146.52e6, "FM", "5 kHz")
        bookmarks.add_bookmark("Weather", 162.55e6, "FM", "5 kHz")
        menu = bookmarks._context_menu(1)
        texts = [a.text().split("\t")[0] for a in menu.actions() if a.text()]
        self.assertIn("Scan Bookmarks from Here", texts)
        action = next(
            a for a in menu.actions() if a.text() == "Scan Bookmarks from Here"
        )
        self.assertTrue(action.isEnabled())
        panel = win._scanner_panel
        panel._group.addItem("Ham", "Ham")
        panel._group.setCurrentIndex(1)  # a group chosen before...
        action.trigger()
        self.assertEqual(panel._group.currentIndex(), 0)  # ...all bookmarks
        self.assertEqual(win._current_panel_key(), "scanner")
        self.assertTrue(panel.engine.active)
        self.assertEqual(panel.scan_mode(), "bookmarks")
        self.assertEqual(panel.engine.current.label, "Weather")
        self.assertAlmostEqual(win._current_frequency(), 162.55e6)
        from sdr_module.gui.scan_engine import LISTEN

        self.assertTrue(self.frames(2.0, until=lambda: panel.engine.state == LISTEN))
        panel.stop_scan()

    def test_no_device_offers_the_demo(self):
        win, panel = self.win, self.win._scanner_panel
        win._release_device()
        win._refresh_state_ui()
        panel.refresh()
        self.assertFalse(panel._demo_btn.isHidden())
        panel._demo_btn.click()
        self.assertIsNotNone(win._device)
        panel.refresh()
        self.assertTrue(panel._demo_btn.isHidden())
        self.assertTrue(panel._start_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
