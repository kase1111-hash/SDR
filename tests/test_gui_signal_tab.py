"""The Signal tab and Radio > Track Signal (AFC) in the window (T10).

Covers fg:FG-G09 / mc:MC-G08 / ux:UX-G10 (Identify, Apply Mode, "Identify
Signal Here"), fg:FG-G17 / rx:G-10 (signals in view, markers, Home / Page
Up / Page Down), fg:FG-G18 / mc:MC-G10 / ux:UX-G16 (the scope), fg:FG-G19 /
fg:FG-G20 (bursts and the OOK analysis), fg:FG-G11 / mc:MC-G09 / rx:G-07 /
ux:UX-G15 (Track Signal), fg:FG-G02 (Find Signals in Recording) and
fg:FG-U02 (a home for the analysis tools).
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

FRAME_S = 0.034


class _SignalCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        # Every demo sample is read (no skipping ahead on a busy machine).
        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tools = self.win._signal_tools
        self.panel = self.win._signal_panel

    def frames(self, seconds, until=None, pause=0.0):
        """Run the display loop (the timer is stopped in these tests)."""
        win = self.win
        for _ in range(int(seconds / FRAME_S)):
            win._pacer.reset()
            win._process_block(
                win._device.read_samples(int(win._device_sample_rate() * FRAME_S))
            )
            if pause:
                time.sleep(pause)  # the identify worker's turn
            if until is not None and until():
                return True
        return False

    def listen(self, freq, mode=None):
        win = self.win
        win.set_frequency(freq)
        if mode is not None:
            win._set_demod_mode(*mode)
        if not win._is_running:
            win._start_acquisition()
        self.assertTrue(win._is_running)

    def identified(self, seconds=8.0):
        """Run frames until the identification is shown."""
        done = self.frames(
            seconds, until=lambda: self.tools._identifier is None, pause=0.003
        )
        self.assertTrue(done, self.panel.identify_status.text())
        return self.panel.identification()

    def bursts_done(self):
        """Wait for the bursts being taken apart (on their worker) and list
        them, as the window's timer does."""
        for job, _report, _when in list(self.tools._burst_jobs):
            job.result(timeout=30)
        self.tools._collect_bursts()
        self.assertFalse(self.tools._burst_jobs)


class TestPlace(_SignalCase):
    def test_a_tab_before_info_on_ctrl_9(self):
        """FG-U02: the analysis tools have a home: the Signal tab."""
        tabs = self.win._right_tabs
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertEqual(titles[-2:], ["Signal", "Info"])
        keys = [key for key, _t, _tip in self.win._panel_specs]
        self.assertEqual(keys.index("signal"), 8)  # Ctrl+9
        self.win._show_panel("info")
        QTest.keyClick(self.win, Qt.Key.Key_9, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.win._current_panel_key(), "signal")
        self.assertTrue(tabs.tabToolTip(tabs.indexOf(self.win._panel_pages["signal"])))

    def test_menus(self):
        win = self.win
        self.assertTrue(win._track_action.isCheckable())
        self.assertIn("AFC", win._track_action.text().replace("&", ""))
        found = {}
        for action in win.findChildren(type(win._track_action)):
            text = action.text().replace("&", "")
            found[text.split("\t")[0]] = action
        identify = found["Identify Signal"]
        self.assertEqual(identify.shortcut(), QKeySequence("Ctrl+I"))
        self.assertIn("Find Signals in Recording...", found)
        for name in ("Signals in View", "Scope", "Bursts"):
            found[name].trigger()
            self.assertEqual(win._current_panel_key(), "signal")
        self.assertEqual(self.panel.current_page(), "bursts")
        for name in (
            "Tune to Strongest Signal",
            "Next Signal Up",
            "Next Signal Down",
        ):
            self.assertIn(name, found)
            self.assertTrue(found[name].toolTip())

    def test_pages_and_their_empty_states(self):
        self.assertEqual(
            [self.panel.tabs.tabText(i) for i in range(self.panel.tabs.count())],
            ["Identify", "Signals", "Scope", "Bursts"],
        )
        self.assertIn("Identify (Ctrl+I)", self.panel.identify_status.text())
        self.assertFalse(self.panel.apply_btn.isEnabled())
        self.assertTrue(self.panel.signals_empty.text())
        for widget in (
            self.panel.identify_btn,
            self.panel.apply_btn,
            self.panel.markers_check,
            self.panel.min_snr,
            self.panel.view_combo,
            self.panel.overlay_combo,
            self.panel.freeze_btn,
            self.panel.watch_check,
        ):
            self.assertTrue(widget.toolTip(), widget)


class TestIdentify(_SignalCase):
    def test_ctrl_i_identifies_the_broadcast_station(self):
        self.listen(100.1e6, ("FM", "75 kHz", "200 kHz"))
        self.win._show_panel("info")
        QTest.keyClick(self.win, Qt.Key.Key_I, Qt.KeyboardModifier.ControlModifier)
        self.settle()
        self.assertEqual(self.win._current_panel_key(), "signal")
        self.assertEqual(self.panel.current_page(), "identify")
        self.assertFalse(self.panel.identify_btn.isEnabled())  # listening
        ident = self.identified()
        self.assertEqual(ident.kind, "fm")
        self.assertEqual(ident.label, "Broadcast FM")
        self.assertEqual((ident.mode, ident.fm_deviation), ("FM", "75 kHz"))
        self.assertTrue(self.panel.identify_btn.isEnabled())
        self.assertEqual(self.panel.kind_label.text(), "Broadcast FM")
        self.assertIn("FM · 75 kHz deviation", self.panel.listen_label.text())
        self.assertIn("SNR", self.panel.listen_label.text())
        self.assertRegex(self.panel.confidence_label.text(), r"^(Confident|Probably)")

    def test_apply_mode_listens_the_suggested_way(self):
        """NOAA weather radio heard in AM: Identify says narrow FM, Apply
        Mode switches to FM with its deviation and filter."""
        self.listen(162.55e6, ("AM", None, "10 kHz"))
        self.assertTrue(self.tools.identify())
        ident = self.identified()
        self.assertEqual(ident.kind, "fm")
        self.assertTrue(ident.label.startswith("Narrow FM"))
        self.assertTrue(self.panel.apply_btn.isEnabled())
        self.panel.apply_btn.click()
        self.assertEqual(self.win._listening_mode(), ("FM", "5 kHz"))
        self.assertEqual(self.win._control_panel._bw_combo.currentText(), "25 kHz")
        self.assertAlmostEqual(self.win._current_frequency(), 162.55e6, delta=200)

    def test_the_remote_control_is_taken_apart(self):
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.tools.identify()
        ident = self.identified()
        self.assertEqual(ident.kind, "ook")
        self.assertIsNotNone(ident.pulses)
        self.assertIn("{25}a1c5e38", self.panel.pulses_label.text())
        self.assertTrue(self.panel.copy_pulses_btn.isVisible())
        self.panel.copy_pulses_btn.click()
        text = QApplication.clipboard().text()
        self.assertIn("Guessing modulation: Pulse width modulation", text)
        self.assertIn("-X 'n=name,m=OOK_PWM", text)
        # The long card scrolls in its page; it doesn't make the other pages
        # taller than the column (their buttons fell out of view at 768 px).
        self.assertIs(self.panel.tabs.widget(0), self.panel.identify_area)
        self.assertLess(self.panel.minimumSizeHint().height(), 260)

    def test_identify_here_from_the_plots(self):
        """The spectrum's and the waterfall's right-click menus: "Identify
        Signal Here" tunes there and identifies."""
        self.listen(100.3e6)
        for plot in (self.win._spectrum, self.win._waterfall):
            texts = [t for t, _cb, _tip in plot.__dict__.get("_context_actions", [])]
            self.assertIn("&Identify Signal Here", texts)
        _text, callback, _tip = self.win._spectrum._context_actions[0]
        callback(99.5e6)
        self.assertAlmostEqual(self.win._current_frequency(), 99.5e6)
        self.assertIsNotNone(self.tools._identifier)
        ident = self.identified()
        self.assertAlmostEqual(ident.tuned_hz, 99.5e6)
        self.assertEqual(ident.kind, "fm")

    def test_identify_here_is_in_the_menus_the_plots_open(self):
        """After each plot's own Tune / Bookmark / Copy / Zoom items."""
        from PyQt6.QtCore import QPoint, QPointF
        from PyQt6.QtWidgets import QMenu

        self.listen(100.3e6)
        self.frames(0.5)
        for plot in (self.win._spectrum, self.win._waterfall):
            seen = []

            def fake_exec(menu, *_args, seen=seen):
                seen.extend(a.text().replace("&", "") for a in menu.actions())

            with mock.patch.object(QMenu, "exec", fake_exec):
                plot._show_context_menu(QPointF(plot.plot_rect().center()), QPoint())
            self.assertIn("Identify Signal Here", seen, plot)
            self.assertLess(
                [t.startswith("Tune to") for t in seen].index(True),
                seen.index("Identify Signal Here"),
            )

    def test_an_empty_channel_is_said_to_be_empty(self):
        self.listen(300.5e6, ("FM", "5 kHz", "12.5 kHz"))
        self.tools.identify()
        ident = self.identified(seconds=10.0)
        self.assertFalse(ident.is_signal)
        self.assertIn("Nothing identifiable", self.panel.identify_status.text())
        self.assertFalse(self.panel.apply_btn.isEnabled())

    def test_without_a_receiver(self):
        self.win._release_device()
        self.win._refresh_state_ui()
        self.assertFalse(self.tools.identify())
        self.assertIn("No receiver", self.panel.identify_status.text())

    def test_stopping_says_so(self):
        self.listen(100.1e6)
        self.tools.identify()
        self.win._stop_acquisition()
        self.assertIsNone(self.tools._identifier)
        self.assertIn("Stopped", self.panel.identify_status.text())
        self.assertTrue(self.panel.identify_btn.isEnabled())


class TestSignalsInView(_SignalCase):
    def setUp(self):
        super().setUp()
        self.listen(100.3e6, ("FM", "75 kHz", "200 kHz"))
        self.win._show_signal_page("signals")
        self.frames(1.2)

    def freqs(self):
        return [s.frequency_hz for s in self.panel.signals()]

    def test_the_list(self):
        freqs = self.freqs()
        for want in (99.5e6, 100.1e6, 101.1e6):
            self.assertTrue(any(abs(f - want) < 1e3 for f in freqs), (want, freqs))
        table = self.panel.signals_table
        self.assertEqual(table.rowCount(), len(freqs))
        headers = [table.horizontalHeaderItem(c).text() for c in range(5)]
        self.assertEqual(headers, ["MHz", "dBFS", "SNR", "Width", "Heard"])
        row = freqs.index(100.1e6)
        self.assertEqual(table.item(row, 0).text(), "100.100")
        self.assertEqual(table.item(row, 4).text(), "now")
        self.assertGreater(float(table.item(row, 2).text()), 40)

    def test_double_click_tunes_and_bookmark_adds(self):
        row = self.freqs().index(99.5e6)
        self.panel.signals_table.selectRow(row)
        self.panel._activate_selected()  # (a double-click)
        self.assertAlmostEqual(self.win._current_frequency(), 99.5e6)
        before = len(self.win._bookmarks_panel.bookmarks())
        self.panel.bookmark_btn.click()
        marks = self.win._bookmarks_panel.bookmarks()
        self.assertEqual(len(marks), before + 1)
        self.assertAlmostEqual(marks[-1]["freq_hz"], 99.5e6)

    def test_markers_on_the_spectrum(self):
        self.assertEqual(self.win._spectrum.markers(), [])
        self.panel.markers_check.setChecked(True)
        self.frames(0.6)
        markers = self.win._spectrum.markers()
        self.assertTrue(markers)
        numbers = {round(f): label for f, label in markers}
        freqs = self.freqs()
        self.assertEqual(numbers[round(100.1e6)], str(freqs.index(100.1e6) + 1))
        self.win._spectrum.grab()  # paints them
        self.panel.markers_check.setChecked(False)
        self.assertEqual(self.win._spectrum.markers(), [])

    def test_home_page_up_and_page_down(self):
        self.win._waterfall.setFocus()
        self.settle()
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_Home)
        self.assertAlmostEqual(self.win._current_frequency(), 100.1e6)  # strongest
        # (Tuned to the signal's centre to 10 Hz, or 100 Hz for a wide one.)
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_PageUp)
        up = self.win._current_frequency()
        above = sorted(f for f in self.freqs() if f > 100.1e6 + 1e3)
        self.assertAlmostEqual(up, above[0], delta=50)
        self.assertEqual(up % 10, 0)
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_PageDown)
        self.assertAlmostEqual(self.win._current_frequency(), 100.1e6)
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_PageDown)
        below = sorted(f for f in self.freqs() if f < 100.1e6 - 1e3)
        self.assertAlmostEqual(self.win._current_frequency(), below[-1], delta=50)

    def test_not_measured_off_the_page(self):
        self.win._show_panel("info")
        self.tools._reset_list()
        self.frames(0.6)
        self.assertEqual(self.tools._heard, [])
        self.assertIsNone(self.tools._acc)

    def test_the_keys_never_use_an_old_list(self):
        """Off the Signals page the list is kept for 30 s after a key; a
        key long after that (here: in another band) used the old list, and
        Page Down at 433.92 MHz tuned back to 101.1 MHz. It measures the
        signals in view afresh instead."""
        win = self.win
        win._show_panel("info")
        self.tools._listing_until = time.monotonic() - 1.0  # 30 s are over
        win.set_frequency(433.92e6)
        win._set_demod_mode("None (I/Q)", None, None)
        self.frames(0.3)
        self.assertTrue(self.tools._heard)  # (what was heard on the FM band)
        self.assertFalse(self.tools.step("down"))
        self.assertAlmostEqual(win._current_frequency(), 433.92e6)
        self.assertIn("Looking for signals", win._message_label.text())
        self.frames(0.6)  # measured now, in this span
        self.assertTrue(self.tools.step("up"), win._message_label.text())
        self.assertGreater(win._current_frequency(), 433.92e6)
        self.assertLess(win._current_frequency(), 433.92e6 + 1.2e6)


class TestScope(_SignalCase):
    def test_only_while_on_screen(self):
        self.listen(162.55e6, ("FM", "5 kHz", "12.5 kHz"))
        self.win._show_signal_page("identify")
        self.frames(0.3)
        self.assertEqual(len(self.panel.scope.samples()), 0)
        self.win._show_signal_page("scope")
        self.frames(0.3)
        self.assertGreater(len(self.panel.scope.samples()), 0)
        self.assertIn("The tuned channel: FM", self.panel.scope_info.text())

    def test_views_freeze_and_both_themes(self):
        from sdr_module.gui import themes

        self.listen(162.55e6, ("FM", "5 kHz", "12.5 kHz"))
        self.win._show_signal_page("scope")
        self.frames(0.3)
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in themes.THEMES:
                themes.apply_theme(qapp, theme)
                for index in range(self.panel.view_combo.count()):
                    self.panel.view_combo.setCurrentIndex(index)
                    self.frames(0.07)
                    image = self.panel.scope.grab().toImage()
                    self.assertFalse(image.isNull())
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)
        # The constellation's overlay replaces the time span in the row.
        self.panel.view_combo.setCurrentIndex(self.panel.view_combo.count() - 1)
        self.assertFalse(self.panel.overlay_combo.isHidden())
        self.assertTrue(self.panel.span_combo.isHidden())
        self.panel.freeze_btn.click()
        held = self.panel.scope.samples().copy()
        self.frames(0.2)
        np.testing.assert_array_equal(self.panel.scope.samples(), held)
        self.panel.freeze_btn.click()
        self.frames(0.1)
        self.assertFalse(np.array_equal(self.panel.scope.samples(), held))

    def test_qpsk_overlay_and_evm(self):
        from sdr_module.gui.signal_panel import ScopeView
        from sdr_module.ui.constellation import ModulationOverlay

        scope = ScopeView()
        self.addCleanup(scope.deleteLater)
        rng = np.random.default_rng(1)
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, 4000)))
        noisy = symbols * 0.3 * np.exp(0.4j) + 0.01 * (
            rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
        )
        scope.set_view("constellation")
        scope.set_overlay(ModulationOverlay.QPSK)
        scope.set_samples(noisy.astype(np.complex64), 4800.0)
        scope.resize(300, 300)
        scope.grab()
        self.assertIsNotNone(scope.evm_percent)
        self.assertLess(scope.evm_percent, 10.0)  # scaled and turned to fit


class TestConstellationLevel(unittest.TestCase):
    """mc:MC-G10: the constellation divided the points by the peak sample,
    so a noisy signal's points all fell short of the ideal ones and the EVM
    read 14 % for a clean (25 dB) QPSK signal; they are scaled by their
    average level now."""

    def test_a_clean_qpsk_signal_reads_a_small_evm(self):
        from sdr_module.ui.constellation import ConstellationDisplay, ModulationOverlay

        rng = np.random.default_rng(2)
        n = 2000
        symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, n)))
        sigma = 10 ** (-25 / 20) / np.sqrt(2)  # 25 dB SNR
        x = 0.3 * (
            symbols + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        )
        display = ConstellationDisplay(overlay=ModulationOverlay.QPSK, max_points=n)
        evm = display.process(x.astype(np.complex64)).stats.evm_percent
        self.assertLess(evm, 8.0)  # the noise's own: about 5.6 %


class TestBursts(_SignalCase):
    def test_the_remote_is_logged_with_its_bits(self):
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.win._show_signal_page("bursts")
        self.frames(3.4)
        self.bursts_done()
        reports = [r for _stamp, r in self.panel.bursts()]
        remote = [r for r in reports if abs(r.burst.center_hz - 433.92e6) < 20e3]
        self.assertTrue(remote, [r.burst.center_hz for r in reports])
        table = self.panel.bursts_table
        row = [r for _s, r in self.panel.bursts()].index(remote[0])
        self.assertIn("OOK PWM {25}a1c5e38 ×8", table.item(row, 4).text())
        table.selectRow(row)
        self.assertIn("Guessing modulation", self.panel.burst_details.toPlainText())
        # Double-click tunes to it.
        self.panel._tune_selected_burst()
        self.assertAlmostEqual(self.win._current_frequency(), 433.92e6, delta=5e3)
        folder = tempfile.mkdtemp(prefix="bursts-")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "b.csv")
        self.assertEqual(self.panel.export_bursts(path), len(reports))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("OOK PWM", text)
        self.assertIn("a1c5e38", text)

    def test_not_watched_off_the_page_unless_asked(self):
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.win._show_signal_page("identify")
        self.frames(3.0)
        self.assertEqual(self.panel.bursts(), [])
        self.assertFalse(self.tools._burst_jobs)
        self.panel.watch_check.setChecked(True)  # "Watch in the background"
        self.frames(3.4)
        self.bursts_done()
        self.assertTrue(self.panel.bursts())

    def test_taken_apart_off_the_display_loop(self):
        """A finished burst's pulse analysis and classification take 50 to
        300 ms: they ran in the display loop, which stalled for that long
        every time a burst ended. They run on the bursts worker, and the
        list gets them in order."""
        import threading

        from sdr_module.gui import signal_tools

        threads = []

        def analyze(report):
            threads.append(threading.current_thread())
            return real(report)

        real = signal_tools.analyze_report
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.win._show_signal_page("bursts")
        with mock.patch.object(signal_tools, "analyze_report", analyze):
            self.frames(3.4)
            self.bursts_done()
        self.assertTrue(threads)
        self.assertNotIn(threading.main_thread(), threads)
        reports = [r for _stamp, r in self.panel.bursts()]
        self.assertTrue(reports)
        self.assertTrue(all(r.analysis is not None for r in reports), reports)
        # Newest first, as they ended.
        starts = [r.burst.start_s for r in reports]
        self.assertEqual(starts, sorted(starts, reverse=True))

    def test_the_list_keeps_no_samples(self):
        """ROB-03: each listed burst kept its channel's samples (up to 1 s,
        a few MB each): 500 of them held about 300 MB. The list keeps what
        was learned (the analysis), not the samples."""
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.win._show_signal_page("bursts")
        self.frames(3.4)
        self.bursts_done()
        reports = [r for _stamp, r in self.panel.bursts()]
        self.assertTrue(reports)
        self.assertTrue(any(r.analysis is not None for r in reports))
        self.assertEqual([r.samples for r in reports], [None] * len(reports))

    def test_broadcast_fm_lists_no_bursts(self):
        """ROB-03: on the first-launch tuning (100.1 MHz) the broadcast
        stations' sidebands were listed as 'bursts', four a second."""
        self.listen(100.1e6, ("FM", "75 kHz", "200 kHz"))
        self.win._device._rng = np.random.default_rng(3)  # (the same noise)
        self.win._show_signal_page("bursts")
        self.frames(8.0)
        self.bursts_done()
        listed = [r.burst.center_hz for _s, r in self.panel.bursts()]
        for station in (99.5e6, 100.1e6, 101.1e6):
            self.assertFalse(
                [f for f in listed if abs(f - station) < 150e3], (station, listed)
            )

    def overloaded_frames(self, seconds):
        """Frames as a computer that can't keep up runs them: one block."""
        for _ in range(int(seconds / FRAME_S)):
            self.win._pacer.reset()
            self.win._pacer.overloaded = True
            self.win._process_block(self.win._device.read_samples(2048))

    def test_paused_while_the_display_loop_is_behind(self):
        """ROB-03: while the window can't keep up, each frame reads one block
        (a few per cent of the signal): bursts came out cut up or as
        stations' sidebands, and the worker taking them apart competed
        with the display loop when it could least afford it. Watching
        pauses (the empty list says why) and goes on once it keeps up."""
        from sdr_module.gui import signal_tools
        from sdr_module.gui.signal_panel import BURSTS_EMPTY

        self.listen(100.1e6, ("FM", "75 kHz", "200 kHz"))
        self.win._show_signal_page("bursts")
        watcher = self.tools._watcher
        with mock.patch.object(watcher, "feed", wraps=watcher.feed) as fed:
            self.overloaded_frames(2.0)
            fed.assert_not_called()
            self.assertIn("can't keep up", self.panel.bursts_empty.text())
            self.assertTrue(self.panel.bursts_empty.is_visible())
            self.frames(0.5)
            fed.assert_called()
        self.assertEqual(self.panel.bursts_empty.text(), BURSTS_EMPTY)
        # A burst in progress when it falls behind is listed, not analysed.
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.assertTrue(self.frames(5.0, until=lambda: bool(watcher.detector.active)))
        self.bursts_done()
        before = len(self.panel.bursts())
        analyze = mock.Mock(side_effect=signal_tools.analyze_report)
        with mock.patch.object(signal_tools, "analyze_report", analyze):
            self.overloaded_frames(1.0)
            self.bursts_done()
        analyze.assert_not_called()
        new = [r for _s, r in self.panel.bursts()][: len(self.panel.bursts()) - before]
        self.assertTrue(new)
        self.assertTrue(all(r.analysis is None for r in new))

    def test_the_worker_keeps_to_its_share_of_the_time(self):
        """A busy band can't keep the bursts worker busy all the time: over
        :data:`ANALYSIS_SHARE` of the last few seconds, the next bursts are
        listed without their analysis."""
        from sdr_module.dsp.bursts import Burst, BurstReport
        from sdr_module.gui import signal_tools

        tools = self.tools
        tools._watcher.time_s = 100.0
        window = signal_tools._ANALYSIS_WINDOW_S
        tools._analysis_cost.extend(
            [(99.0, 0.6 * signal_tools.ANALYSIS_SHARE * window)] * 2
        )
        self.assertFalse(tools._may_analyze())
        tools._watcher.time_s = 99.0 + window + 0.1  # those are old now
        self.assertTrue(tools._may_analyze())
        report = BurstReport(
            Burst(1.0, 1.1, 433.92e6, 10e3, -40.0, 30.0),
            samples=np.zeros(4096, np.complex64),
            sample_rate=300e3,
        )
        tools._analysis_cost.extend([(tools._watcher.time_s, 10.0)])
        tools._queue_bursts([report])
        self.bursts_done()
        self.assertEqual([r for _s, r in self.panel.bursts()], [report])
        self.assertIsNone(report.analysis)
        self.assertIsNone(report.samples)

    def test_closing_with_a_burst_in_progress_closes_the_device(self):
        """ROB-04: closing the window while a burst was in progress raised
        'cannot schedule new futures after shutdown' (stopping the
        receiver handed the burst to the closed worker), so the device was
        never closed (a HackRF's bias tee stayed on), nor the temporary
        recording file and the FFT helper."""
        self.listen(433.92e6, ("None (I/Q)", None, None))
        self.win._show_signal_page("bursts")
        self.assertTrue(
            self.frames(5.0, until=lambda: bool(self.tools._watcher.detector.active))
        )
        device = self.win._device
        spectrum = self.win._frame_spectrum
        buffer = self.win._samples_buffer
        with (
            mock.patch.object(device, "close", wraps=device.close) as close,
            mock.patch.object(
                spectrum, "close", wraps=spectrum.close
            ) as spectrum_close,
            mock.patch.object(buffer, "discard", wraps=buffer.discard) as discard,
            self.assertNoLogs("sdr_module", level="ERROR"),
        ):
            self.win.close()
        close.assert_called_once_with()
        spectrum_close.assert_called_once_with()
        discard.assert_called_once_with()
        self.assertIsNone(self.win._device)

    def test_closing_still_closes_the_device_when_a_step_fails(self):
        """Closing goes on past a step that fails (it is logged): the device
        is closed whatever happened before."""
        self.listen(433.92e6, ("None (I/Q)", None, None))
        device = self.win._device
        spectrum = self.win._frame_spectrum
        with (
            mock.patch.object(
                self.win, "_stop_acquisition", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(device, "close", wraps=device.close) as close,
            mock.patch.object(
                spectrum, "close", wraps=spectrum.close
            ) as spectrum_close,
            self.assertLogs("sdr_module.gui.main_window", level="ERROR") as logs,
        ):
            self.win.close()
        close.assert_called_once_with()
        spectrum_close.assert_called_once_with()
        self.assertIn("boom", "\n".join(logs.output))


class TestTrackSignal(_SignalCase):
    def test_broadcast_fm_is_refused(self):
        self.listen(100.1e6, ("FM", "75 kHz", "200 kHz"))
        self.win._track_action.trigger()
        self.assertFalse(self.win._track_action.isChecked())
        self.assertFalse(self.tools.tracking)
        self.assertIn("broadcast FM", self.win._message_label.text())

    def test_ssb_is_refused(self):
        self.listen(14.2e6, ("USB", None, "2.7 kHz"))
        self.assertFalse(self.tools.set_tracking(True))
        self.assertIn("SSB", self.win._message_label.text())

    def test_the_drifting_handheld_is_followed(self):
        """UX-G15: the demo's 70 cm handheld drifts +/- 2.5 kHz over 40 s;
        the tuned frequency follows it, the LO stays put, and the status
        bar says how far ("AFC +2.4 kHz")."""
        from sdr_module.gui import device_dialog

        self.listen(438.5e6, ("FM", "5 kHz", "12.5 kHz"))
        lo = self.win._lo_frequency()
        start = self.win._device._stream_pos / self.win._device_sample_rate()
        self.win._track_action.trigger()
        self.assertTrue(self.tools.tracking)
        self.frames(8.0)
        t = self.win._device._stream_pos / self.win._device_sample_rate()
        drift = device_dialog._DRIFT_HZ * np.sin(
            2 * np.pi * t / device_dialog._DRIFT_PERIOD_S
        )
        self.assertGreater(t - start, 7.0)
        moved = self.win._current_frequency() - 438.5e6
        self.assertAlmostEqual(moved, drift, delta=150.0)
        self.assertEqual(self.win._lo_frequency(), lo)
        self.assertRegex(self.tools.badge_text(), r"^AFC \+2\.\d kHz$")
        # Turned off: the badge goes, the frequency stays.
        self.win._track_action.trigger()
        self.assertFalse(self.tools.tracking)
        self.assertEqual(self.tools.badge_text(), "")

    def test_tuning_by_hand_starts_over(self):
        self.listen(438.5e6, ("FM", "5 kHz", "12.5 kHz"))
        self.win._track_action.trigger()
        self.frames(3.0)
        self.assertNotEqual(self.tools.badge_text(), "AFC +0 Hz")
        self.win.set_frequency(438.6e6)
        self.assertEqual(self.tools.badge_text(), "AFC +0 Hz")
        self.assertAlmostEqual(self.win._current_frequency(), 438.6e6)
        # Nothing measured here yet: not shown as locked on (it was green,
        # "locked on", before anything was heard).
        badge = self.tools._afc_badge
        self.assertNotEqual(badge.property("tone"), "success")
        self.assertNotIn("locked on", badge.toolTip())

    def test_drifting_out_of_a_recording_turns_it_off(self):
        """Playing a recording, Track Signal stops when the signal drifts
        out of the file's span: the menu item went on looking checked and
        the badge on showing the last correction."""
        win = self.win
        self.listen(438.5e6, ("FM", "5 kHz", "12.5 kHz"))
        win._track_action.trigger()
        self.assertTrue(win._track_action.isChecked())
        # (A recording whose span ends just above the signal.)
        win._lo_lock = (438.5e6 - 1e6 + 4e3, 2e6, "the recording")
        try:
            self.frames(8.0, until=lambda: not self.tools.tracking)
        finally:
            win._lo_lock = None
        self.assertFalse(self.tools.tracking)
        self.assertFalse(win._track_action.isChecked())
        self.assertEqual(self.tools.badge_text(), "")
        self.assertIn("drifted out of the recording", win._message_label.text())

    def test_turned_on_while_stopped_says_paused(self):
        self.listen(438.5e6, ("FM", "5 kHz", "12.5 kHz"))
        self.win._stop_acquisition()
        self.assertFalse(self.win._is_running)
        self.win._track_action.trigger()
        self.assertTrue(self.tools.tracking)
        self.assertEqual(self.tools.badge_text(), "AFC paused")


class TestFindSignalsInRecording(_SignalCase):
    RATE = 240e3

    def setUp(self):
        super().setUp()
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        self.folder = tempfile.mkdtemp(prefix="sdr-find-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        n = int(3.0 * self.RATE)
        t = np.arange(n) / self.RATE
        rng = np.random.default_rng(6)
        x = 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x += 0.1 * np.exp(2j * np.pi * 30e3 * t)  # all the time
        on = (t >= 1.5) & (t < 2.0)
        x[on] += 0.1 * np.exp(2j * np.pi * -50e3 * t[on])  # from 1.5 s
        self.path = os.path.join(self.folder, "two.sigmf-data")
        save_iq_file(
            self.path,
            x.astype(np.complex64),
            self.RATE,
            145e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )

    def test_lists_them_and_plays_from_one(self):
        self.listen(100.3e6)
        self.panel.markers_check.setChecked(True)
        self.win._show_signal_page("signals")
        self.frames(0.6)
        self.assertTrue(self.win._spectrum.markers())  # the live signals
        self.assertTrue(self.tools.find_in_recording(self.path))
        # The live markers go: their numbers would not be the list's rows.
        self.assertEqual(self.win._spectrum.markers(), [])
        self.assertFalse(self.panel.markers_check.isEnabled())
        self.assertEqual(self.win._current_panel_key(), "signal")
        self.assertEqual(self.panel.current_page(), "signals")
        self.assertTrue(self.panel.showing_recording())
        found = self.panel.signals()
        self.assertEqual(
            [round(s.frequency_hz - 145e6, -2) for s in found], [-50e3, 30e3]
        )
        self.assertIn("two.sigmf-data", self.panel.recording_label.text())
        table = self.panel.signals_table
        self.assertEqual(table.item(0, 4).text(), "0:01.5–0:02.0")
        self.assertEqual(self.panel.tune_btn.text(), "Play")
        self.assertIn("Found 2 signals", self.win._message_label.text())
        # Double-click the burst: the recording plays from just before it,
        # tuned to it.
        table.selectRow(0)
        self.panel._activate_selected()
        self.assertTrue(self.win._playback.active)
        self.assertAlmostEqual(self.win._current_frequency(), 145e6 - 50e3, delta=200)
        self.assertTrue(self.win._is_running)
        position = self.win._playback.device.position_s
        self.assertAlmostEqual(position, 1.0, delta=0.3)
        # The playback strip has the same: Signals...
        with mock.patch.object(self.tools, "find_in_recording") as find:
            self.win._playback_bar._find.click()
        find.assert_called_once()
        self.assertEqual(find.call_args[0][0], self.path)
        # Back to Live lists what the receiver hears again.
        self.panel.live_btn.click()
        self.assertFalse(self.panel.showing_recording())

    def test_a_file_without_a_rate_asks(self):
        """A headerless file needs its rate and centre: Import Options."""
        raw = os.path.join(self.folder, "raw.cf32")
        np.zeros(4800, np.complex64).tofile(raw)
        with mock.patch.object(
            self.win, "_recording_options", return_value=None
        ) as ask:
            self.assertFalse(self.tools.find_in_recording(raw))
        self.assertEqual(ask.call_args[0][:2], (raw, "find"))


if __name__ == "__main__":
    unittest.main()
