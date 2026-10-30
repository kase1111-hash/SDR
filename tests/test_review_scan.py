#!/usr/bin/env python3
"""
Regression tests for the code review of the scanner and bookmarks (area
"scan"): one test (or a few) per confirmed finding, named after its id.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_scan.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402
from tests.test_gui_scanner import (  # noqa: E402
    FRAME_S,
    FakeWindow,
    _demo,
    _DemoHost,
    _PanelCase,
    _run,
    _ScanWindowCase,
)

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication


def _preset_index(panel, prefix):
    """The Band list's index of the preset whose label starts with ``prefix``."""
    for i in range(panel._preset.count()):
        if panel._preset.itemText(i).startswith(prefix):
            return i
    raise ValueError(prefix)


# --------------------------------------------------------------------------- #
# SCAN-01: a sweep started with a wide filter measured around the wrong LO
# --------------------------------------------------------------------------- #
class TestScan01SweepWithAWideFilter(_ScanWindowCase):
    def test_each_step_is_measured_around_the_lo_it_was_planned_on(self):
        """I/Q with a 2 MHz filter (what Listen on an ISM 915 result
        selects): the nearest channel doesn't fit beside the LO, so tuning
        the VFO there moved the LO by half a step, and every result was
        off by it (125.000 MHz listed as 124.988 MHz)."""
        win, panel, host = self.win, self.win._scanner_panel, self.win._scanner_host
        win._set_demod_mode("None (I/Q)", None, "2 MHz")
        self.assertEqual(win._control_panel._bw_combo.currentText(), "2 MHz")
        moved = []
        tune_lo = host.scan_tune_lo

        def spy(center, vfo=None):
            ok = tune_lo(center, vfo)
            moved.append(round(win._lo_frequency() - center))
            return ok

        host.scan_tune_lo = spy
        panel._preset.setCurrentIndex(_preset_index(panel, "Airband"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan(), panel._status.text())
        self.frames(10.0, until=lambda: not panel.engine.active)
        self.assertFalse(panel.engine.active)
        self.assertEqual(set(moved), {0})
        found = {round(h.frequency_hz) for h in panel.hits()}
        self.assertIn(125_000_000, found)  # the demo's steady AM station
        self.assertFalse({124_988_000, 124_987_500} & found, sorted(found))

    def test_listen_on_a_result_found_with_a_wide_filter_hears_it(self):
        win, panel = self.win, self.win._scanner_panel
        win._set_demod_mode("None (I/Q)", None, "2 MHz")
        panel._preset.setCurrentIndex(_preset_index(panel, "Airband"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        self.frames(10.0, until=lambda: not panel.engine.active)
        rows = [
            r
            for r in range(panel._table.rowCount())
            if panel._table.item(r, 0).text() == "125.000 MHz"
        ]
        self.assertEqual(len(rows), 1, [h.frequency_hz for h in panel.hits()])
        panel._table.selectRow(rows[0])
        panel.listen_selected()
        self.settle()
        self.assertAlmostEqual(win._current_frequency(), 125.0e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "AM")
        self.frames(1.0)
        self.assertTrue(win._squelch.is_open)


class TestScan01EngineMeasuresAroundTheReceiversLo(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_a_host_that_puts_the_lo_elsewhere_still_gets_true_frequencies(self):
        """The engine measures each page around the LO the host reports, not
        the one it planned: results stay right even when the receiver can't
        put its LO exactly there."""
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        class Shifted(_DemoHost):
            def scan_tune_lo(self, center, vfo=None):
                return super().scan_tune_lo(center + 12.5e3, vfo)

        dev = _demo(162.4e6)
        engine = ScanEngine(Shifted(dev))
        found = []
        engine.on_found = found.append
        settings = ScanSettings("once", 162.4e6, 162.55e6, 25e3)
        self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
        _run(engine, dev)
        self.assertFalse(engine.active)
        noaa = sorted({round(f.frequency_hz) for f in found})
        self.assertEqual(noaa, [162400000, 162475000, 162550000])


# --------------------------------------------------------------------------- #
# SCAN-03: Edit... rewrote an imported channel's CHIRP Mode
# --------------------------------------------------------------------------- #
_CHIRP_MODES = (
    "NFM",
    "DV",
    "DMR",
    "P25",
    "DN",
    "PKT",
    "RTTY",
    "FSK",
    "NAM",
    "CWR",
    "NCW",
    "WFM",
    "AM",
    "FM",
    "USB",
    "Auto",
)


def _chirp_csv(modes):
    rows = "".join(
        f"{i},CH{m},{145.0 + i * 0.025:.6f},,0.000000,,88.5,88.5,023,NN,{m},5.00,,,\n"
        for i, m in enumerate(modes, 1)
    )
    return (
        "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
        "DtcsPolarity,Mode,TStep,Skip,Comment,URCALL\n" + rows
    )


class _BookmarksCase(unittest.TestCase):
    """A Bookmarks panel with in-memory settings."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel
        from tests.test_gui_chrome import _FakeSettings

        self.mod = bookmarks_panel
        self.store = _FakeSettings()
        self._real_settings = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: self.store
        self._panels = []

    def tearDown(self):
        self.mod.GuiSettings = self._real_settings
        for panel in self._panels:
            panel.close()
            panel.deleteLater()
        QApplication.processEvents()

    def panel(self, bookmarks=()):
        self.store.bookmarks = [dict(b) for b in bookmarks]
        panel = self.mod.BookmarksPanel()
        self._panels.append(panel)
        return panel

    def edit_with(self, panel, index, change):
        """Run the panel's Edit... on ``index``, with ``change(dialog)``
        done in the dialog before Save."""
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        def dialog(parent, bookmark, listening, groups):
            box = BookmarkEditDialog(bookmark, listening, groups, parent)
            try:
                change(box)
                return box.bookmark()
            finally:
                box.deleteLater()

        real = self.mod._edit_dialog
        self.mod._edit_dialog = dialog
        try:
            return panel.edit_bookmark(index)
        finally:
            self.mod._edit_dialog = real


class TestScan03EditKeepsTheChirpMode(_BookmarksCase):
    def test_renaming_an_imported_channel_changes_nothing_else(self):
        import io

        from sdr_module.core.chirp_csv import export_bookmarks_csv, read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(_CHIRP_MODES))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        for index, mode in enumerate(_CHIRP_MODES):
            before = panel.bookmark(index)

            def rename(box):
                box._name.setText(box._name.text() + " renamed")

            self.assertTrue(self.edit_with(panel, index, rename), mode)
            after = panel.bookmark(index)
            self.assertEqual(after.pop("label"), before.pop("label") + " renamed")
            self.assertEqual(after, before, mode)
        out = io.StringIO()
        export_bookmarks_csv(out, panel.bookmarks())
        out.seek(0)
        self.assertEqual([c.mode for c in read_chirp_csv(out)], list(_CHIRP_MODES))

    def test_a_new_filter_width_keeps_the_mode_and_survives_a_round_trip(self):
        import io

        from sdr_module.core.chirp_csv import export_bookmarks_csv, read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(("DV", "NFM")))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        for index in (0, 1):
            self.edit_with(
                panel,
                index,
                lambda box: box._width.setCurrentIndex(box._width.findData("12.5 kHz")),
            )
        dv, nfm = panel.bookmarks()
        self.assertEqual((dv["mode"], dv["bandwidth"]), ("DV", "12.5 kHz"))
        self.assertEqual((nfm["mode"], nfm["bandwidth"]), ("NFM", "12.5 kHz"))
        self.assertEqual(self.mod.listening(dv), ("FM", "5 kHz", "12.5 kHz", 0.0))
        out = io.StringIO()
        export_bookmarks_csv(out, [dv, nfm])
        out.seek(0)
        back = [c.to_bookmark() for c in read_chirp_csv(out)]
        self.assertEqual([b["mode"] for b in back], ["DV", "NFM"])
        self.assertEqual(
            [self.mod.listening(b) for b in back],
            [
                ("FM", "5 kHz", "12.5 kHz", 0.0),
                ("FM", "5 kHz", "12.5 kHz", 0.0),
            ],
        )

    def test_a_new_mode_or_deviation_still_sets_the_chirp_mode(self):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(("DV", "FM", "NFM")))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        self.edit_with(
            panel, 0, lambda box: box._mode.setCurrentIndex(box._mode.findData("AM"))
        )
        self.edit_with(panel, 1, lambda box: box._deviation.setCurrentText("2.5 kHz"))

        # Away and back again is no change.
        def away_and_back(box):
            box._mode.setCurrentIndex(box._mode.findData("USB"))
            box._mode.setCurrentIndex(box._mode.findData("FM"))

        self.edit_with(panel, 2, away_and_back)
        am, narrow, nfm = panel.bookmarks()
        self.assertEqual((am["demod"], am["mode"]), ("AM", "AM"))
        self.assertEqual(
            (narrow["demod"], narrow["fm_deviation"], narrow["mode"]),
            ("FM", "2.5 kHz", "NFM"),
        )
        self.assertEqual(nfm["mode"], "NFM")
        self.assertNotIn("demod", nfm)


# --------------------------------------------------------------------------- #
# SCAN-02: a CTCSS tone squelch of 88.5 Hz (CHIRP's default) was lost
# --------------------------------------------------------------------------- #
_TONE_CSV = (
    "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
    "DtcsPolarity,Mode,TStep,Skip,Comment\n"
    "1,RPT885,145.290000,-,0.600000,TSQL,88.5,88.5,023,NN,FM,5.00,,\n"
    "2,RPT100,145.310000,-,0.600000,TSQL,88.5,100.0,023,NN,FM,5.00,,\n"
    "3,ENC885,145.330000,-,0.600000,Tone,88.5,88.5,023,NN,FM,5.00,,\n"
    "4,DCS023,145.350000,-,0.600000,DTCS,88.5,88.5,023,NN,FM,5.00,,\n"
)


class TestScan02ToneSquelchOf885Hz(_BookmarksCase):
    def test_an_imported_tsql_channel_keeps_its_885_hz_tone(self):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv
        from sdr_module.gui.bookmark_editor import bookmark_tone_hz

        rpt885, rpt100, enc885, dcs023 = [
            c.to_bookmark() for c in read_chirp_csv(io.StringIO(_TONE_CSV))
        ]
        self.assertEqual(rpt885["ctone"], 88.5)
        self.assertEqual(bookmark_tone_hz(rpt885), 88.5)
        self.assertEqual(self.mod.listening(rpt885), ("FM", "5 kHz", "", 88.5))
        self.assertEqual(self.mod.listening(rpt100), ("FM", "5 kHz", "", 100.0))
        # The tones a radio uses for the other Tone modes are kept too.
        self.assertEqual(enc885["rtone"], 88.5)
        self.assertEqual((dcs023["dtcs"], dcs023["dtcs_polarity"]), (23, "NN"))
        # A transmit-only tone is no tone squelch.
        self.assertEqual(self.mod.listening(enc885)[3], 0.0)

    def test_the_apps_own_885_hz_tone_survives_a_round_trip(self):
        import io

        from sdr_module.core.chirp_csv import export_bookmarks_csv, read_chirp_csv

        panel = self.panel()
        index = panel.add_bookmark("Club", 145.29e6, "FM", "5 kHz", "12.5 kHz")
        panel.set_tone(index, 88.5)
        before = self.mod.listening(panel.bookmark(index))
        self.assertEqual(before, ("FM", "5 kHz", "12.5 kHz", 88.5))
        out = io.StringIO()
        export_bookmarks_csv(out, panel.bookmarks())
        out.seek(0)
        (back,) = [c.to_bookmark() for c in read_chirp_csv(out)]
        self.assertEqual(self.mod.listening(back), before)

    def test_a_tsql_bookmark_saved_without_its_tone_listens_for_885_hz(self):
        """Bookmarks imported before the fix have no "ctone": CHIRP's
        default tone is what the radio uses, and what the tooltip says."""
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog, bookmark_tone_hz

        saved = {
            "label": "RPT",
            "freq_hz": 145.29e6,
            "location": 1,
            "tone_mode": "TSQL",
        }
        self.assertEqual(bookmark_tone_hz(saved), 88.5)
        self.assertIn("TSQL: 88.5 Hz", self.mod._tooltip(saved))
        dialog = BookmarkEditDialog(saved, self.mod.tune_mode(saved))
        self.assertEqual(dialog._tone.currentData(), 88.5)
        self.assertEqual(dialog.bookmark().get("tone_mode"), "TSQL")
        dialog.deleteLater()


class TestScan02ToneSquelchInTheWindow(_ScanWindowCase):
    def test_tuning_an_imported_885_hz_channel_sets_its_tone_squelch(self):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        panel = self.win._bookmarks_panel
        panel._bookmarks = [
            c.to_bookmark() for c in read_chirp_csv(io.StringIO(_TONE_CSV))
        ][:1]
        panel._refresh_list()
        self.win._control_panel.set_tone_squelch_hz(0.0)
        panel._tune_index(0)
        self.settle()
        self.assertAlmostEqual(self.win._current_frequency(), 145.29e6)
        self.assertEqual(self.win._control_panel.tone_squelch_hz(), 88.5)


# --------------------------------------------------------------------------- #
# SCAN-04 / 05 / 06: how a bookmark scan listens, and what it leaves behind
# --------------------------------------------------------------------------- #
def _pretend_audio(win):
    win._audio_output_problem = lambda: None
    win._audio.start = lambda *_a, **_k: True
    win._audio.write = lambda *_a, **_k: None
    win._audio_enabled = True


class _BookmarkScanCase(_ScanWindowCase):
    def scan(self, bookmarks, seconds=3.0, until=None):
        """Start a bookmark scan of ``bookmarks``; run until ``until()``
        (default: it stops on a channel) or ``seconds``."""
        from sdr_module.gui.scan_engine import LISTEN

        marks = self.win._bookmarks_panel
        marks._bookmarks = [dict(b) for b in bookmarks]
        marks._save()
        marks._refresh_list()
        panel = self.win._scanner_panel
        panel.set_scan_mode("bookmarks")
        panel._refresh_groups()
        self.assertTrue(panel.start_scan(), panel._status.text())
        engine = panel.engine
        return self.frames(seconds, until=until or (lambda: engine.state == LISTEN))

    def listening_now(self):
        cp = self.win._control_panel
        return (
            self.win._listening_mode(),
            cp._bw_combo.currentText(),
            cp.tone_squelch_hz(),
        )


_CLUB = {"label": "Club", "freq_hz": 145.29e6, "demod": "FM", "fm_deviation": "5 kHz"}
_QUIET_TONE = {
    "label": "Quiet FM tone 100",
    "freq_hz": 145.5e6,
    "demod": "FM",
    "fm_deviation": "5 kHz",
    "mode": "FM",
    "tone_mode": "TSQL",
    "ctone": 100.0,
}
_QUIET_AM = {
    "label": "Quiet AM",
    "freq_hz": 133.0e6,
    "demod": "AM",
    "mode": "AM",
    "bandwidth": "6 kHz",
}
#: An old bookmark without a mode: tuning it keeps the receiver's mode.
_LEGACY = {"label": "Old 146.94", "freq_hz": 146.94e6}


class TestScan04ToneWithTheSquelchOff(_BookmarkScanCase):
    def test_a_wrong_tone_is_passed_by_with_the_squelch_off(self):
        """The demo repeater on 145.290 MHz sends 100.0 Hz: a bookmark with
        123.0 Hz must not stop the scan, squelch on or off."""
        self.win._control_panel.set_squelch_enabled(False)
        self.settle()
        self.assertFalse(self.win._squelch.enabled)
        wrong = dict(_CLUB, label="Club, wrong tone", tone_mode="TSQL", ctone=123.0)
        self.assertFalse(self.scan([wrong]))
        self.assertEqual(self.win._scanner_panel.hits(), [])
        self.win._scanner_panel.stop_scan()

    def test_the_right_tone_is_heard_with_the_squelch_off(self):
        self.win._control_panel.set_squelch_enabled(False)
        self.settle()
        right = dict(_CLUB, tone_mode="TSQL", ctone=100.0)
        self.assertTrue(self.scan([right]))
        self.assertEqual([h.label for h in self.win._scanner_panel.hits()], ["Club"])
        self.win._scanner_panel.stop_scan()


class TestScan05ToneSquelchRestored(_BookmarkScanCase):
    def test_a_scan_from_am_leaves_no_tone_squelch_behind(self):
        win, cp = self.win, self.win._control_panel
        win.set_frequency(125.0e6)
        win._set_demod_mode("AM", None, "10 kHz")
        cp.set_tone_squelch_hz(0.0)
        self.scan([_QUIET_TONE], seconds=0.5)
        self.assertEqual(cp.tone_squelch_hz(), 100.0)  # (the scan's)
        win._scanner_panel.stop_scan()
        self.settle()
        self.assertEqual(win._listening_mode(), ("AM", ""))
        self.assertEqual(cp.tone_squelch_hz(), 0.0)
        self.assertIsNone(win._tone_squelch.tone_hz)
        win._persist_audio_settings()
        self.assertEqual(self.store.values.get("audio/tone_squelch"), 0.0)

    def test_a_scan_from_am_gives_back_the_users_own_tone(self):
        win, cp = self.win, self.win._control_panel
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        cp.set_tone_squelch_hz(100.0)
        win.set_frequency(125.0e6)
        win._set_demod_mode("AM", None, "10 kHz")
        quiet = dict(_QUIET_TONE, label="Quiet FM", tone_mode="", ctone=88.5)
        self.scan([quiet], seconds=0.5)
        self.assertEqual(cp.tone_squelch_hz(), 0.0)  # (the scan's)
        win._scanner_panel.stop_scan()
        self.settle()
        self.assertEqual(win._listening_mode(), ("AM", ""))
        self.assertEqual(cp.tone_squelch_hz(), 100.0)


class TestScan06KeepModeBookmarksInAScan(_BookmarkScanCase):
    def setUp(self):
        super().setUp()
        self.win.set_frequency(146.94e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")  # no tone squelch
        self.win._control_panel.set_tone_squelch_hz(0.0)
        self.user = self.listening_now()
        self.seen = []
        engine = self.win._scanner_panel.engine

        def on_legacy():
            current = engine.current
            if current is not None and current.label == _LEGACY["label"]:
                self.seen.append(self.listening_now())
            return engine.state == "listen" and current.label == _LEGACY["label"]

        self.on_legacy = on_legacy

    def check_heard_as_the_user_listens(self, before):
        self.assertTrue(self.scan([before, _LEGACY], seconds=4.0, until=self.on_legacy))
        self.assertTrue(self.seen)
        self.assertEqual(set(self.seen), {self.user})
        self.assertEqual(
            [h.label for h in self.win._scanner_panel.hits()], [_LEGACY["label"]]
        )
        self.win._scanner_panel.stop_scan()

    def test_after_a_bookmark_with_a_tone_speaker_on(self):
        _pretend_audio(self.win)
        self.check_heard_as_the_user_listens(_QUIET_TONE)

    def test_after_a_bookmark_with_a_tone_speaker_off(self):
        self.win._audio_enabled = False
        self.check_heard_as_the_user_listens(_QUIET_TONE)

    def test_after_an_am_bookmark(self):
        _pretend_audio(self.win)
        self.check_heard_as_the_user_listens(_QUIET_AM)

    def test_the_users_own_tone_squelch_decides_speaker_on_or_off(self):
        """Keep the receiver's mode means its tone squelch too: the demo
        repeater on 145.290 MHz sends 100.0 Hz (and keys up for its first
        12 s: the cases that hear it come first)."""
        legacy = {"label": "Old 145.29", "freq_hz": 145.29e6}
        for speaker in (True, False):
            for tone, heard in ((100.0, True), (123.0, False)):
                if speaker:
                    _pretend_audio(self.win)
                else:
                    self.win._audio_enabled = False
                self.win.set_frequency(145.29e6)
                self.win._control_panel.set_tone_squelch_hz(tone)
                self.assertEqual(
                    self.scan([_QUIET_AM, legacy], seconds=3.0), heard, (speaker, tone)
                )
                self.win._scanner_panel.stop_scan()
                self.assertEqual(self.win._control_panel.tone_squelch_hz(), tone)


# --------------------------------------------------------------------------- #
# SCAN-07: Hold clicked while the scan was "Checking" a channel
# --------------------------------------------------------------------------- #
class TestScan07HoldWhileChecking(_PanelCase):
    def test_hold_in_dwell_shows_held_and_a_second_click_lets_it_go_on(self):
        from sdr_module.gui.scan_engine import DWELL, DWELL_S

        host = FakeWindow()  # every channel quiet: the scan only checks them
        host.bookmarks = [
            {"index": 0, "label": "Quiet A", "freq_hz": 145.5e6, "demod": "FM"},
            {"index": 1, "label": "Quiet B", "freq_hz": 145.6e6, "demod": "FM"},
        ]
        panel = self.panel(host)
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan(), panel._status.text())
        engine = panel.engine
        self.assertEqual((engine.state, engine.current.label), (DWELL, "Quiet A"))
        self.assertFalse(panel._listen_row.isHidden())
        self.assertIn("Checking Quiet A", panel._listen_label.text())
        panel._hold_btn.click()
        self.assertTrue(engine.held)
        self.assertTrue(panel._hold_btn.isChecked())
        self.assertIn("(held)", panel._listen_label.text())
        self.run_scan(panel, seconds=4 * DWELL_S)
        self.assertEqual((engine.state, engine.current.label), (DWELL, "Quiet A"))
        self.assertTrue(panel._hold_btn.isChecked())  # (each new state kept it)
        # Click again: the scan goes on (it has checked Quiet A long enough).
        panel._hold_btn.click()
        self.assertFalse(engine.held)
        self.assertFalse(panel._hold_btn.isChecked())
        self.run_scan(panel, seconds=2 * FRAME_S)
        self.assertEqual(engine.current.label, "Quiet B")
        self.assertIn("Checking Quiet B", panel._listen_label.text())
        self.assertNotIn("(held)", panel._listen_label.text())


# --------------------------------------------------------------------------- #
# SCAN-08: transmitting while a scan is stopped on a channel
# --------------------------------------------------------------------------- #
class TestScan08TransmittingDuringAScan(_BookmarkScanCase):
    def setUp(self):
        super().setUp()
        win = self.win
        # The user was listening to airband in AM before scanning.
        win.set_frequency(125.0e6)
        win._set_demod_mode("AM", None, "10 kHz")
        self.messages = []
        show = win._show_status_message

        def spy(text, *args, **kwargs):
            self.messages.append(text)
            return show(text, *args, **kwargs)

        win._show_status_message = spy

    def transmit(self):
        """What StationTx.transmit does: the HackRF is busy (sending where
        the receiver is tuned) and the receiver stops for it."""
        self.win._tx_busy = lambda: True
        self.win._stop_acquisition()

    def receive_again(self):
        self.win._tx_busy = lambda: False
        self.win._start_acquisition()

    def test_answering_the_channel_the_scan_stopped_on_stays_on_it(self):
        win, panel = self.win, self.win._scanner_panel
        self.assertTrue(self.scan([_CLUB], seconds=4.0))  # the demo repeater
        self.transmit()
        self.messages.clear()
        panel.refresh()  # (the tab's 500 ms timer)
        self.assertFalse(panel.engine.active)
        self.assertEqual(panel._status.text(), "Scan stopped: transmitting.")
        self.assertEqual(self.messages, [])  # no "Can't tune while transmitting"
        self.assertAlmostEqual(win._current_frequency(), 145.29e6)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))
        self.receive_again()
        self.assertAlmostEqual(win._current_frequency(), 145.29e6)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))
        # No scan while transmitting: Start says why.
        self.transmit()
        panel.refresh()
        self.assertFalse(panel._start_btn.isEnabled())
        self.assertIn("transmitting", panel._start_btn.toolTip())
        self.assertFalse(panel.start_scan())
        self.receive_again()
        panel.refresh()
        self.assertTrue(panel._start_btn.isEnabled())

    def test_a_scan_that_cant_go_back_keeps_the_channels_mode(self):
        """Stopping where the receiver can't be retuned: it stays on the
        channel in the channel's mode, not in the mode of where it began."""
        from sdr_module.gui.scan_engine import DWELL

        win, panel = self.win, self.win._scanner_panel
        quiet = dict(_QUIET_TONE, label="Quiet FM", tone_mode="")
        self.scan([quiet], seconds=1.0, until=lambda: panel.engine.state == DWELL)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))
        win._tune_to = lambda *_a, **_k: False  # the receiver refuses
        panel.stop_scan()  # (stay=False: not listening to anything)
        self.assertAlmostEqual(win._current_frequency(), 145.5e6)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))

    def test_stop_pressed_as_a_transmission_starts(self):
        """Stop pressed on a channel being checked, before the tab's refresh
        saw the transmission: the scan doesn't try to go back (the receiver
        can't be retuned while it transmits), so no "Can't tune while
        transmitting" replaces the on-air status."""
        from sdr_module.gui.scan_engine import DWELL

        win, panel = self.win, self.win._scanner_panel
        quiet = dict(_QUIET_TONE, label="Quiet FM", tone_mode="")
        self.scan([quiet], seconds=1.0, until=lambda: panel.engine.state == DWELL)
        self.transmit()
        self.messages.clear()
        panel.stop_scan()  # (stay=False: not listening to anything)
        self.assertFalse(panel.engine.active)
        self.assertEqual(self.messages, [])
        self.assertAlmostEqual(win._current_frequency(), 145.5e6)
        self.assertEqual(win._listening_mode(), ("FM", "5 kHz"))
        self.receive_again()


# --------------------------------------------------------------------------- #
# SCAN-09: sdr-scan scan --rate the driver can't do
# --------------------------------------------------------------------------- #
class TestScan09CliRateTheDeviceRuns(unittest.TestCase):
    """Through the real RTLSDRDevice and DeviceManager, with pyrtlsdr
    faked: the "hardware" samples at the rate the driver actually set."""

    CARRIERS = (100.1e6, 101.3e6)

    def scan(self, rate):
        import contextlib
        import io
        import sys
        import types
        from unittest import mock

        import numpy as np

        from sdr_module import cli

        carriers = self.CARRIERS

        class RtlSdr:
            def __init__(self, device_index=0):
                self.sample_rate, self.center_freq, self.gain = 2.4e6, 100e6, "auto"
                self._rng = np.random.default_rng(1)

            @staticmethod
            def get_device_serial_addresses():
                return ["00000001"]

            def set_manual_gain_enabled(self, *_args):
                pass

            def get_gains(self):
                return [0, 100, 200]

            def read_samples(self, n):
                t = np.arange(n) / self.sample_rate
                x = (
                    self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
                ) * 1e-4
                for freq in carriers:
                    offset = freq - self.center_freq
                    if abs(offset) < self.sample_rate / 2:
                        x = x + 0.05 * np.exp(2j * np.pi * offset * t)
                return x.astype(np.complex64)

            def close(self):
                pass

        class _Lib:
            @staticmethod
            def rtlsdr_get_device_count():
                return 1

        fake = types.ModuleType("rtlsdr")
        fake.RtlSdr, fake.librtlsdr = RtlSdr, _Lib
        out = io.StringIO()
        with (
            mock.patch.dict(sys.modules, {"rtlsdr": fake}),
            mock.patch.object(cli, "_LIVE_SETTLE_S", 0.0),
            mock.patch(
                "sdr_module.devices.hackrf.HackRFDevice.list_devices", return_value=[]
            ),
            contextlib.redirect_stdout(out),
        ):
            rc = cli.main(
                ["scan", "--start", "99", "--end", "102", "--single", "--rate", rate]
            )
        return rc, out.getvalue()

    def test_a_rate_the_driver_clamps_is_scanned_at_the_rate_it_runs(self):
        rc, out = self.scan("3.2e6")  # an RTL-SDR runs at 2.56 MS/s at most
        self.assertEqual(rc, 0, out)
        self.assertIn("100.1000 MHz", out)
        self.assertIn("101.3000 MHz", out)
        self.assertIn("(2.56 MS/s)", out)
        self.assertIn("can't run at 3.2 MS/s", out)

    def test_a_rate_it_runs_at_says_nothing_more(self):
        rc, out = self.scan("2.4e6")
        self.assertEqual(rc, 0, out)
        self.assertIn("100.1000 MHz", out)
        self.assertIn("101.3000 MHz", out)
        self.assertIn("(2.4 MS/s)", out)
        self.assertNotIn("can't run", out)


# --------------------------------------------------------------------------- #
# SCAN-10: a minimal CSV's Mode=FM rows imported as "keep the receiver's mode"
# --------------------------------------------------------------------------- #
class TestScan10ModeColumnOfAMinimalCsv(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def bookmarks(self, text):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        return {c.name: c.to_bookmark() for c in read_chirp_csv(io.StringIO(text))}

    def test_a_mode_column_saying_fm_tunes_fm(self):
        from sdr_module.gui.bookmarks_panel import _mode, tune_mode

        marks = self.bookmarks(
            "Name,Frequency,Mode\n"
            "RPT FM,145.290000,FM\n"
            "RPT NFM,145.300000,NFM\n"
            "TWR AM,118.100000,AM\n"
        )
        self.assertEqual(marks["RPT FM"]["mode"], "FM")
        self.assertEqual(_mode(marks["RPT FM"]), "FM")
        self.assertEqual(tune_mode(marks["RPT FM"]), ("FM", "5 kHz"))
        self.assertEqual(tune_mode(marks["RPT NFM"]), ("FM", "5 kHz"))
        self.assertEqual(tune_mode(marks["TWR AM"]), ("AM", ""))

    def test_a_file_that_doesnt_say_still_keeps_the_receivers_mode(self):
        from sdr_module.gui.bookmarks_panel import tune_mode

        marks = self.bookmarks("Name,Frequency,Mode\nA,146.520000,\n")
        marks.update(self.bookmarks("Name,Frequency\nB,162.550000\n"))
        for name in ("A", "B"):
            self.assertNotIn("mode", marks[name], name)
            self.assertEqual(tune_mode(marks[name]), ("", ""), name)


# --------------------------------------------------------------------------- #
# SCAN-11: the frequency was restored before the mode when a scan ended
# --------------------------------------------------------------------------- #
class TestScan11PlotsGoBackToo(_BookmarkScanCase):
    def test_after_a_wide_bookmark_the_lo_and_the_vfo_go_back(self):
        from sdr_module.gui.scan_engine import DWELL

        win, panel = self.win, self.win._scanner_panel
        win.set_frequency(146.0e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win.set_frequency(146.9e6)  # the VFO moves within the span
        before = (win._lo_frequency(), win._current_frequency(), self.listening_now())
        self.assertAlmostEqual(before[0], 146.0e6)
        wide = {
            "label": "Quiet WFM",
            "freq_hz": 87.7e6,
            "demod": "FM",
            "fm_deviation": "75 kHz",
            "bandwidth": "200 kHz",
        }
        engine = panel.engine
        self.assertTrue(
            self.scan(
                [wide],
                seconds=2.0,
                until=lambda: engine.state == DWELL
                and engine.current.label == wide["label"],
            )
        )
        self.assertEqual(win._control_panel._bw_combo.currentText(), "200 kHz")
        panel.stop_scan()  # (not listening to anything: it goes back)
        after = (win._lo_frequency(), win._current_frequency(), self.listening_now())
        self.assertEqual(after, before)


# --------------------------------------------------------------------------- #
# SCAN-12: a damaged time in the saved results stopped the app from starting
# --------------------------------------------------------------------------- #
class TestScan12DamagedSavedResults(_PanelCase):
    def test_times_that_cant_be_shown_read_as_never(self):
        import json

        good = 1.7e9
        entries = []
        for key, value in (
            ("last_seen", 1e20),
            ("last_seen", float("nan")),
            ("last_seen", -1e12),
            ("first_seen", 1e20),
            ("first_seen", float("inf")),
        ):
            entry = {
                "frequency_hz": 100.1e6 + len(entries) * 1e6,
                "first_seen": good,
                "last_seen": good,
            }
            entry[key] = value
            entries.append(entry)
        entries.append({"frequency_hz": 110e6, "count": float("inf")})
        self.store.values["scanner/results"] = json.dumps(entries)
        panel = self.panel()  # (it used to raise here, and so did the window)
        hits = panel.hits()
        self.assertEqual(len(hits), 5)  # (a count of infinity isn't a result)
        for hit in hits:
            for when in (hit.first_seen, hit.last_seen):
                self.assertTrue(when in (0.0, good), when)
        heard = [panel._table.item(r, 4).text() for r in range(panel._table.rowCount())]
        self.assertEqual(heard.count("—"), 3)
        for row in range(panel._table.rowCount()):
            self.assertTrue(panel._table.item(row, 0).toolTip())


class TestScan12TheWindowStarts(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_the_window_is_built_with_a_damaged_result_saved(self):
        import json

        from sdr_module.gui import bookmarks_panel, main_window
        from tests.test_gui_chrome import _FakeSettings

        store = _FakeSettings()
        store.values["scanner/results"] = json.dumps(
            [{"frequency_hz": 100.1e6, "last_seen": 1e20, "first_seen": 1e20}]
        )
        patched = [
            (main_window, "GuiSettings", main_window.GuiSettings),
            (bookmarks_panel, "GuiSettings", bookmarks_panel.GuiSettings),
        ]
        main_window.GuiSettings = lambda: store
        bookmarks_panel.GuiSettings = lambda: store
        try:
            win = main_window.SDRMainWindow(demo_mode=False)
            for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
                timer.stop()
            self.assertEqual(len(win._scanner_panel.hits()), 1)
            win.close()
            win.deleteLater()
            QApplication.processEvents()
        finally:
            for module, name, value in patched:
                setattr(module, name, value)


# --------------------------------------------------------------------------- #
# SCAN-13: a "Sweep once" that can't tune any step said "complete"
# --------------------------------------------------------------------------- #
class TestScan13NoStepTuned(_PanelCase):
    def test_once_says_the_receiver_could_not_tune_like_continuous(self):
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        dev = _demo()
        host = _DemoHost(dev)
        host.refuse_above = 0.0  # every step is refused
        engine = ScanEngine(host)
        finished = []
        engine.on_finished = finished.append
        for mode in ("once", "continuous"):
            error = engine.start(
                ScanSettings(mode, 118e6, 137e6, 25e3), dev.sample_rate, 0.0
            )
            self.assertFalse(engine.active)
            self.assertEqual(
                error, "The receiver could not tune to any step of this range.", mode
            )
        self.assertEqual(finished, [])  # (no scan ran)

    def test_the_tab_says_why_in_words(self):
        host = FakeWindow()
        host.scan_tune_lo = lambda center, vfo=None: False
        panel = self.panel(host)
        panel.set_scan_mode("once")
        self.assertFalse(panel.start_scan())
        self.assertEqual(
            panel._status.text(),
            "The receiver could not tune to any step of this range.",
        )


# --------------------------------------------------------------------------- #
# SCAN-14: each reading cost time in proportion to the results listed
# --------------------------------------------------------------------------- #
def _merged_like_before(readings, step_hz, bookmarks=False):
    """The results the Scanner tab listed for ``readings`` before the
    index: each reading merges into the first result it repeats."""
    from sdr_module.dsp.sweep import same_signal

    hits = []  # [frequency, peak, width, label, listening, count]
    for found in readings:
        for hit in hits:
            if bookmarks:
                same = (
                    abs(hit[0] - found.frequency_hz) < 10.0
                    and hit[3] == found.label
                    and hit[4] == tuple(found.listening)
                )
            else:
                same = same_signal(hit[0], hit[2], found.frequency_hz, step_hz)
            if same:
                hit[5] += 1
                if found.peak_db > hit[1]:
                    hit[0], hit[1], hit[2] = (
                        found.frequency_hz,
                        found.peak_db,
                        found.width_hz,
                    )
                break
        else:
            hits.append(
                [
                    found.frequency_hz,
                    found.peak_db,
                    found.width_hz,
                    found.label,
                    tuple(found.listening),
                    1,
                ]
            )
    return [(h[0], h[1], h[2], h[3], h[5]) for h in hits]


class TestScan14ReadingsCostTheSame(_PanelCase):
    def readings(self, count, seed, bookmarks=False):
        import numpy as np

        from sdr_module.gui.scan_engine import Found

        rng = np.random.default_rng(seed)
        out = []
        for _ in range(count):
            if bookmarks:
                k = int(rng.integers(0, 12))
                out.append(
                    Found(
                        145.0e6 + (k % 4) * 25e3,
                        float(rng.uniform(-80, -20)),
                        -90.0,
                        10e3,
                        ("FM", "5 kHz", ""),
                        label=f"B{k}",
                    )
                )
            else:
                width = float(rng.choice([5e3, 12e3, 150e3, 400e3]))
                out.append(
                    Found(
                        100e6 + float(rng.integers(0, 400)) * 37e3,
                        float(rng.uniform(-80, -20)),
                        -90.0,
                        width,
                        ("FM", "75 kHz", ""),
                    )
                )
        return out

    def test_readings_merge_as_they_did(self):
        panel = self.panel()
        panel._step.setValue(25.0)
        readings = self.readings(600, seed=7)
        for found in readings:
            panel._on_found(found)
        got = [
            (h.frequency_hz, h.peak_db, h.width_hz, h.label, h.count)
            for h in panel.hits()
        ]
        self.assertEqual(got, _merged_like_before(readings, 25e3))
        self.assertEqual(panel._table.rowCount(), len(got))
        # Every row shows its own result.
        for row in range(panel._table.rowCount()):
            cell = panel._table.item(row, 0)
            hit = panel.hits()[cell.data(Qt.ItemDataRole.UserRole + 1)]
            self.assertEqual(cell.text(), f"{hit.frequency_hz / 1e6:.3f} MHz")

    def test_bookmark_readings_merge_as_they_did(self):
        self.host.bookmarks = [
            {"index": 0, "label": "B", "freq_hz": 145.0e6, "demod": "FM"}
        ]
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        readings = self.readings(200, seed=3, bookmarks=True)
        for found in readings:
            panel._on_found(found)
        got = [
            (h.frequency_hz, h.peak_db, h.width_hz, h.label, h.count)
            for h in panel.hits()
        ]
        self.assertEqual(got, _merged_like_before(readings, 1.0, bookmarks=True))
        panel.stop_scan()

    def test_a_reading_doesnt_visit_every_result(self):
        from unittest import mock

        from sdr_module.gui import scanner_panel
        from sdr_module.gui.scan_engine import Found
        from sdr_module.gui.scanner_panel import ScanHit

        panel = self.panel()
        panel.set_scan_mode("continuous")
        panel._step.setValue(150.0)
        panel._hits = [
            ScanHit(30e6 + i * 150e3, -60.0, 20.0, 5e3, "FM", "5 kHz", "25 kHz")
            for i in range(2000)
        ]
        panel._rebuild_table()
        panel._table.setSortingEnabled(False)  # (as during a scan)
        compared, cells, refreshed = [], [], []
        same = scanner_panel.same_signal
        item = panel._table.item
        with (
            mock.patch.object(
                scanner_panel,
                "same_signal",
                side_effect=lambda *a: compared.append(a) or same(*a),
            ),
            mock.patch.object(
                panel._table, "item", side_effect=lambda *a: cells.append(a) or item(*a)
            ),
            mock.patch.object(
                panel, "_update_controls", side_effect=lambda: refreshed.append(1)
            ),
        ):
            for i in range(0, 2000, 50):  # repeats, spread over the list
                panel._on_found(
                    Found(30e6 + i * 150e3, -59.0, -80.0, 5e3, ("FM", "5 kHz", ""))
                )
            for i in range(40):  # and new ones
                panel._on_found(
                    Found(400e6 + i * 150e3, -59.0, -80.0, 5e3, ("FM", "5 kHz", ""))
                )
        self.assertLessEqual(len(compared), 4 * 80)
        self.assertLessEqual(len(cells), 80)
        self.assertEqual(refreshed, [])  # (once a page, by the engine's on_state)
        self.assertEqual(len(panel.hits()), 2040)
        self.assertEqual(panel.hits()[0].count, 2)

    def test_the_status_still_counts_every_signal_found(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        panel = self.panel()
        index = [p.label for p in SCAN_PRESETS].index(SCAN_PRESETS[0].label) + 1
        panel._preset.setCurrentIndex(index)  # FM broadcast
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        self.run_scan(panel, seconds=3.0)
        count = len(panel.hits())
        self.assertGreater(count, 10)
        self.assertIn(f"{count} signals found", panel._status.text())
        self.assertFalse(panel._empty.is_visible())
        self.assertTrue(panel._bookmark_all_btn.isEnabled())
        panel.stop_scan()


# --------------------------------------------------------------------------- #
# SCAN-15: Record Audio during a sweep recorded what the scan had muted
# --------------------------------------------------------------------------- #
class TestScan15RecordAudioDuringASweep(_ScanWindowCase):
    def test_the_recording_is_silent_where_the_speaker_was_muted(self):
        import tempfile

        import numpy as np

        from sdr_module.gui import file_dialogs

        win, panel = self.win, self.win._scanner_panel
        folder = tempfile.mkdtemp(prefix="scan15-")
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        file_dialogs.set_recordings_folder(self.store, folder)
        _pretend_audio(win)
        heard = []
        win._audio.write = lambda block, *_a, **_k: heard.append(len(block))
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertTrue(win._start_audio_recording())
        take = win._audio_take
        written = []
        write = take.write

        def spy(audio, open_, continuous=True):
            loud = float(np.max(np.abs(audio))) if len(audio) else 0.0
            written.append((bool(win._scan_muted), bool(open_), loud))
            return write(audio, open_, continuous=continuous)

        take.write = spy
        panel._preset.setCurrentIndex(_preset_index(panel, "FM Broadcast"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        heard.clear()
        self.frames(6.0, until=lambda: not panel.engine.active)
        self.assertFalse(panel.engine.active)
        self.assertTrue(any(muted for muted, _o, _l in written))
        self.assertEqual(heard, [])  # (the speaker played nothing meanwhile)
        self.assertEqual(
            [entry for entry in written if entry[0] and entry[1]], []
        )  # nothing recorded as heard while muted
        self.assertTrue(win._stop_audio_recording())


class TestScanToneBlockStaysOnItsFmChannel(_BookmarkScanCase):
    """Found checking the fixes: Record Audio demodulates every block, and
    outside FM nothing cleared "the tone squelch holds this signal back",
    so the FM channel's verdict stuck to the AM channel after it. A bookmark
    scan never stopped on an active AM channel that followed a channel
    sending the wrong tone (squelch on, and since SCAN-04 squelch off)."""

    def scan_while_recording(self, squelch):
        import tempfile

        from sdr_module.gui import file_dialogs

        win, panel = self.win, self.win._scanner_panel
        folder = tempfile.mkdtemp(prefix="scan-tone-")
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        file_dialogs.set_recordings_folder(self.store, folder)
        _pretend_audio(win)
        win._control_panel.set_squelch_enabled(squelch)
        self.settle()
        # (A second of signal first: the squelch knows the noise floor, so
        # it opens on the repeater at once and its tone is checked.)
        self.frames(1.0)
        self.assertTrue(win._start_audio_recording())
        wrong = dict(_CLUB, label="Club, wrong tone", tone_mode="TSQL", ctone=123.0)
        tower = {
            "label": "Tower",
            "freq_hz": 125.0e6,  # the demo's steady AM station
            "demod": "AM",
            "mode": "AM",
            "bandwidth": "10 kHz",
        }
        heard = self.scan([wrong, tower], seconds=4.0)
        hits = [h.label for h in panel.hits()]
        blocked = win._tone_blocked
        panel.stop_scan()
        self.assertTrue(win._stop_audio_recording())
        self.assertTrue(heard, "the scan never stopped on the AM station")
        self.assertEqual(hits, ["Tower"])
        self.assertFalse(blocked)

    def test_squelch_on(self):
        self.scan_while_recording(squelch=True)

    def test_squelch_off(self):
        self.scan_while_recording(squelch=False)


# --------------------------------------------------------------------------- #
# SCAN-17: at 20 MS/s the FM Broadcast preset listed a stray carrier
# --------------------------------------------------------------------------- #
class TestScan17MinimumWidthAtHighRates(unittest.TestCase):
    """measure_page on a synthetic FM broadcast page (200 kHz steps, the
    preset's 30 kHz minimum width)."""

    def page(self, rate, signals, seed=7):
        """``(readings)`` for a page around 99.82 MHz holding ``signals``:
        ``(offset in bins from 99.82 MHz, level dBFS, width Hz or 0)``."""
        import numpy as np

        from sdr_module.dsp.sweep import SpectrumAverager, measure_page, plan_sweep

        rng = np.random.default_rng(seed)
        pages = plan_sweep(88e6, 108e6, 200e3, rate)
        page = next(p for p in pages if p.first_hz <= 99.82e6 <= p.last_hz)
        bin_hz = rate / 2048
        n = 2048 * 32
        t = np.arange(n) / rate
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 10 ** (-60 / 20)
        x = x.astype(np.complex128)
        centre = round((99.82e6 - page.lo_hz) / bin_hz)
        for bins, level, width in signals:
            offset = (centre + bins) * bin_hz
            if width:  # band-limited noise, `width` wide
                noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
                spectrum = np.fft.fft(noise)
                freqs = np.fft.fftfreq(n, 1 / rate)
                spectrum[np.abs(freqs) > width / 2] = 0
                shaped = np.fft.ifft(spectrum)
                shaped *= 10 ** (level / 20) / np.sqrt(np.mean(np.abs(shaped) ** 2))
                x += shaped * np.exp(2j * np.pi * offset * t)
            else:  # a pure carrier
                x += 10 ** (level / 20) * np.exp(2j * np.pi * offset * t)
        averager = SpectrumAverager(32)
        averager.add(x.astype(np.complex64))
        readings, _noise, _threshold = measure_page(
            averager.power_db(),
            page.lo_hz,
            rate,
            page.channels,
            200e3,
            min_width_hz=30e3,
        )
        return readings

    def test_a_carrier_isnt_a_station_whatever_the_rate(self):
        for rate in (2.4e6, 10e6, 20e6):
            for between in (0.0, 0.5):  # on a bin, half-way between two
                self.assertEqual(
                    self.page(rate, [(between, -30.0, 0)]), [], (rate, between)
                )

    def test_stations_still_are(self):
        for rate in (2.4e6, 10e6, 20e6):
            readings = self.page(rate, [(0, -40.0, 150e3)])
            self.assertEqual(len(readings), 1, rate)
            self.assertAlmostEqual(readings[0].frequency_hz, 99.8e6, delta=100e3)
        # Narrow, but wider than the minimum.
        self.assertEqual(len(self.page(2.4e6, [(0, -40.0, 40e3)])), 1)

    def test_without_a_minimum_a_carrier_is_listed(self):
        import numpy as np

        from sdr_module.dsp.sweep import SpectrumAverager, measure_page

        rate, lo = 20e6, 100e6
        n = 2048 * 32
        t = np.arange(n) / rate
        x = 1e-3 * (np.random.default_rng(1).standard_normal(n) + 0j)
        x = x + 0.03 * np.exp(2j * np.pi * 1.0e6 * t)
        averager = SpectrumAverager(32)
        averager.add(x.astype(np.complex64))
        readings, _n, _t = measure_page(
            averager.power_db(), lo, rate, [lo + 1.0e6], 25e3, min_width_hz=0.0
        )
        self.assertEqual([round(r.frequency_hz) for r in readings], [101000000])


# --------------------------------------------------------------------------- #
# SCAN-18: NaN / inf in a CSV
# --------------------------------------------------------------------------- #
_ODD_HEADER = (
    "Location,Name,Frequency,Offset,rToneFreq,cToneFreq,DtcsCode,RxDtcsCode,"
    "Mode,TStep\n"
)


class TestScan18NotNumbersInACsv(unittest.TestCase):
    def read(self, rows, strict=False):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        return read_chirp_csv(io.StringIO(_ODD_HEADER + rows), strict=strict)

    def test_a_frequency_that_isnt_a_number_skips_the_row_and_says_so(self):
        report = self.read(
            "1,NAN,nan,0,88.5,88.5,023,023,FM,5\n"
            "2,INF,inf,0,88.5,88.5,023,023,FM,5\n"
            "3,MINF,-Infinity,0,88.5,88.5,023,023,FM,5\n"
            "4,SHIFT,146.94,inf,88.5,88.5,023,023,FM,5\n"
            "5,OK,146.52,0,88.5,88.5,023,023,FM,5\n"
        )
        self.assertEqual([c.name for c in report.channels], ["OK"])
        self.assertEqual(len(report.skipped), 4)
        self.assertTrue(report.skipped[0].startswith("line 2:"), report.skipped)

    def test_strict_reading_says_which_line(self):
        from sdr_module.core.chirp_csv import ChirpCsvError

        with self.assertRaises(ChirpCsvError) as raised:
            self.read("1,NAN,nan,0,88.5,88.5,023,023,FM,5\n", strict=True)
        self.assertIn("line 2", str(raised.exception))

    def test_other_columns_that_arent_numbers_fall_back_to_chirps_defaults(self):
        report = self.read(
            "inf,A,146.52,0,inf,nan,inf,-inf,FM,inf\n"
            "nan,B,146.55,0,88.5,100.0,inf,023,FM,5\n"
        )
        self.assertEqual([c.name for c in report.channels], ["A", "B"])
        a, b = report.channels
        self.assertEqual((a.rtone, a.ctone, a.dtcs, a.rx_dtcs), (88.5, 88.5, 23, 23))
        self.assertEqual(a.tuning_step_khz, 5.0)
        self.assertEqual((b.ctone, b.dtcs), (100.0, 23))
        self.assertEqual(report.skipped, [])
        for channel in report.channels:
            channel.to_row()  # (written back without trouble)

    def test_import_channels_in_the_tab_takes_them(self):
        import tempfile

        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel
        from tests.test_gui_chrome import _FakeSettings

        store = _FakeSettings()
        real = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: store
        self.addCleanup(setattr, bookmarks_panel, "GuiSettings", real)
        panel = bookmarks_panel.BookmarksPanel()
        self.addCleanup(panel.deleteLater)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(_ODD_HEADER + "1,X,146.52,0,88.5,88.5,inf,023,FM,5\n")
        self.addCleanup(os.unlink, handle.name)
        self.assertEqual(panel.import_csv(handle.name), 1)
        self.assertEqual(panel.bookmark(0)["label"], "X")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
