#!/usr/bin/env python3
"""
DCS in bookmarks, CHIRP files and the Scanner (K-A3): a channel's receive
code is applied when it is tuned (CHIRP's DTCS, and the Cross modes that
receive with a code), Ctrl+B keeps the code the receiver is set to, Edit
Bookmark offers the codes, the bookmark scan judges a channel by its code
and its results keep it. With the demo device: the 70 cm repeater on
442.100 MHz sends DCS 023.
"""

import io
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

_CSV = (
    "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
    "DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment\n"
    "1,DCS RPT,442.100000,+,5.000000,DTCS,88.5,88.5,023,NN,023,Tone->Tone,FM,"
    "5.00,,,\n"
    "2,DCS INV,442.200000,,0.000000,DTCS,88.5,88.5,754,NR,023,Tone->Tone,FM,"
    "5.00,,,\n"
    "3,SPLIT,442.300000,,0.000000,Cross,88.5,88.5,023,RN,155,DTCS->DTCS,FM,"
    "5.00,,,\n"
    "4,TO DCS,442.400000,,0.000000,Cross,100.0,123.0,023,NN,365,Tone->DTCS,FM,"
    "5.00,,,\n"
    "5,ODD,442.500000,,0.000000,DTCS,88.5,88.5,024,NN,023,Tone->Tone,FM,"
    "5.00,,,\n"
)


def _imported():
    from sdr_module.core.chirp_csv import read_chirp_csv

    report = read_chirp_csv(io.StringIO(_CSV), strict=False)
    return {c.name: c.to_bookmark() for c in report.channels}


class TestBookmarkDcs(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_the_receive_code_of_each_tone_mode(self):
        from sdr_module.gui.bookmark_editor import bookmark_dcs, bookmark_tone_hz

        marks = _imported()
        self.assertEqual(bookmark_dcs(marks["DCS RPT"]), (23, False))
        # Polarity: transmit then receive (CHIRP's "NR": receives inverted).
        self.assertEqual(bookmark_dcs(marks["DCS INV"]), (754, True))
        # Cross channels receive with RxDtcsCode.
        self.assertEqual(bookmark_dcs(marks["SPLIT"]), (155, False))
        self.assertEqual(bookmark_dcs(marks["TO DCS"]), (365, False))
        self.assertEqual(bookmark_tone_hz(marks["TO DCS"]), 0.0)
        # A code that isn't a standard one: no code squelch.
        self.assertIsNone(bookmark_dcs(marks["ODD"]))
        for damaged in (
            {"tone_mode": "DTCS", "dtcs": "abc"},
            {"tone_mode": "DTCS", "dtcs": float("nan")},
            {"tone_mode": "DTCS", "dtcs": 23, "dtcs_polarity": 7},
            {"tone_mode": "TSQL", "ctone": 100.0},
            {"tone_mode": "Cross", "cross_mode": "DTCS->"},
            {},
        ):
            if damaged.get("dtcs_polarity") == 7:
                self.assertEqual(bookmark_dcs(damaged), (23, False))
            else:
                self.assertIsNone(bookmark_dcs(damaged), damaged)

    def test_the_tooltip_names_the_code(self):
        from sdr_module.gui import bookmarks_panel

        marks = _imported()
        self.assertIn("DCS 023 N", bookmarks_panel._tooltip(marks["DCS RPT"]))
        tip = bookmarks_panel._tooltip(marks["TO DCS"])
        self.assertIn("Cross (Tone->DTCS): receives DCS 365 N", tip)
        self.assertNotIn("not supported", tip)
        self.assertIn("not supported", bookmarks_panel._tooltip(marks["ODD"]))

    def test_listening_keeps_its_shape(self):
        from sdr_module.gui.bookmarks_panel import listening

        # (demod, deviation, filter, CTCSS tone): a DCS channel has no tone.
        self.assertEqual(listening(_imported()["DCS RPT"]), ("FM", "5 kHz", "", 0.0))

    def test_a_code_set_by_this_app_is_no_chirp_channel(self):
        from sdr_module.core.chirp_csv import Channel, is_chirp_channel

        bookmark = {
            "label": "Rpt",
            "freq_hz": 442.1e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
            "tone_mode": "DTCS",
            "dtcs": 23,
            "dtcs_polarity": "NN",
        }
        self.assertFalse(is_chirp_channel(bookmark))
        row = Channel.from_bookmark(bookmark).to_row()
        self.assertEqual(
            (row["Tone"], row["DtcsCode"], row["DtcsPolarity"]), ("DTCS", "023", "NN")
        )


class TestEditBookmarkOffersCodes(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def dialog(self, bookmark):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        dialog = BookmarkEditDialog(bookmark, ("FM", "5 kHz"), [])
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_choose_a_code_or_none(self):
        base = {"label": "Rpt", "freq_hz": 442.1e6, "demod": "FM"}
        dialog = self.dialog(base)
        tone = dialog._tone
        index = tone.findData("D754I")
        self.assertGreater(index, tone.findData(254.1))
        self.assertEqual(tone.itemText(index), "DCS 754 I")
        tone.setCurrentIndex(index)
        edited = dialog.bookmark()
        self.assertEqual(
            (edited["tone_mode"], edited["dtcs"], edited["dtcs_polarity"]),
            ("DTCS", 754, "RR"),
        )
        # Opened on a DCS channel, it shows its code; None removes it.
        dialog = self.dialog(edited)
        self.assertEqual(dialog._tone.currentData(), "D754I")
        dialog._tone.setCurrentIndex(0)
        cleared = dialog.bookmark()
        for key in ("tone_mode", "dtcs", "dtcs_polarity"):
            self.assertNotIn(key, cleared)
        # Untouched, a CHIRP channel's code fields stay exactly as they were.
        split = _imported()["SPLIT"]
        kept = self.dialog(split).bookmark()
        for key in ("tone_mode", "cross_mode", "dtcs", "rx_dtcs", "dtcs_polarity"):
            self.assertEqual(kept.get(key), split.get(key), key)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDcsBookmarksInTheWindow(_WindowTestCase):
    def test_tuning_a_dcs_bookmark_sets_its_code(self):
        win, panel = self.win, self.win._bookmarks_panel
        panel.add_bookmark("DCS rpt", 442.1e6, "FM", "5 kHz", "12.5 kHz")
        panel.set_dcs(0, 23, False)
        panel.add_bookmark("Simplex", 446.5e6, "FM", "5 kHz")
        win._set_demod_mode("AM")
        panel._tune_index(0)
        self.assertAlmostEqual(win._current_frequency(), 442.1e6)
        self.assertEqual(win._control_panel.dcs_squelch(), (23, False))
        self.assertEqual(win._tone_squelch.dcs, (23, False))
        self.assertIn("Tuned to DCS rpt (FM, DCS 023 N)", win._message_label.text())
        # A bookmark without a tone or code turns it off.
        panel._tune_index(1)
        self.assertIsNone(win._control_panel.dcs_squelch())
        self.assertIsNone(win._tone_squelch.dcs)

    def test_an_imported_chirp_channel_applies_its_code(self):
        win, panel = self.win, self.win._bookmarks_panel
        for bookmark in _imported().values():
            panel.add_bookmark(bookmark["label"], bookmark["freq_hz"])
            index = len(panel.bookmarks()) - 1
            panel._bookmarks[index] = dict(bookmark)
        names = [b["label"] for b in panel.bookmarks()]
        panel._tune_index(names.index("DCS INV"))
        self.assertEqual(win._control_panel.dcs_squelch(), (754, True))
        panel._tune_index(names.index("SPLIT"))
        self.assertEqual(win._control_panel.dcs_squelch(), (155, False))

    def test_ctrl_b_keeps_the_code(self):
        win = self.win
        win.set_frequency(442.1e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_dcs_squelch(23)
        win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual(
            (saved["tone_mode"], saved["dtcs"], saved["dtcs_polarity"]),
            ("DTCS", 23, "NN"),
        )
        self.assertNotIn("ctone", saved)
        self.assertIn("(FM, DCS 023 N)", win._message_label.text())


class TestScanCarriesTheCode(unittest.TestCase):
    """The engine and the results (no window)."""

    def setUp(self):
        require_pyqt6(self)

    def test_the_engine_tunes_with_the_code_and_the_result_keeps_it(self):
        from sdr_module.gui.scan_engine import DWELL, ScanEngine, ScanSettings

        calls = []

        class Host:
            def scan_tune_channel(self, freq, listening, tone=0.0, dcs=""):
                calls.append((freq, listening, tone, dcs))
                return True

            def scan_lo_moved(self):
                return False

            def scan_lo_hz(self):
                return 442.0e6

            def scan_signal_present(self):
                return True

            def scan_set_muted(self, muted):
                pass

        engine = ScanEngine(Host())
        found = []
        engine.on_found = found.append
        visit = {
            "index": 0,
            "label": "Rpt",
            "freq_hz": 442.1e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
            "bandwidth": "",
            "tone_hz": 0.0,
            "dcs": "D023N",
        }
        engine.start(ScanSettings("bookmarks", bookmarks=[visit]), 2.4e6, 0.0)
        self.assertEqual(calls[0][3], "D023N")
        self.assertEqual(engine.current.dcs, "D023N")
        self.assertIn(engine.state, (DWELL, "settle"))

    def test_a_host_without_codes_is_tuned_as_before(self):
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        calls = []

        class OldHost:
            def scan_tune_channel(self, freq, listening, tone=0.0):
                calls.append((freq, listening, tone))
                return True

            def scan_lo_moved(self):
                return False

            def scan_lo_hz(self):
                return 146.0e6

            def scan_signal_present(self):
                return False

            def scan_set_muted(self, muted):
                pass

        engine = ScanEngine(OldHost())
        visit = {"index": 0, "label": "A", "freq_hz": 146.52e6, "tone_hz": 100.0}
        engine.start(ScanSettings("bookmarks", bookmarks=[visit]), 2.4e6, 0.0)
        self.assertEqual(calls, [(146.52e6, ("", "", ""), 100.0)])

    def test_results_keep_the_code(self):
        from sdr_module.gui.scanner_panel import ScanHit

        hit = ScanHit(442.1e6, -50.0, 30.0, 10e3, "FM", "5 kHz", "", "Rpt", dcs="D023N")
        again = ScanHit.from_dict(hit.to_dict())
        self.assertEqual(again.dcs, "D023N")
        for damaged in ("D999X", 23, None, "nan"):
            data = dict(hit.to_dict(), dcs=damaged)
            self.assertEqual(ScanHit.from_dict(data).dcs, "", damaged)
        self.assertEqual(ScanHit.from_dict({"frequency_hz": 1e8}).dcs, "")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDcsBookmarkScanOnTheDemo(_WindowTestCase):
    """The demo's 70 cm repeater (442.100 MHz) sends DCS 023 for its first
    5 s: a bookmark scan stops on it with its own code, not another."""

    demo = True

    def frames(self, seconds, until=None):
        from unittest import mock

        from sdr_module.gui import device_dialog

        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            for _ in range(int(seconds * 30)):
                self.win._pacer.reset()
                self.win._process_block(
                    self.win._device.read_samples(
                        int(self.win._device_sample_rate() / 30)
                    )
                )
                if until is not None and until():
                    return True
        return False

    def test_the_scan_stops_on_its_code_and_listen_sets_it(self):
        from sdr_module.gui.scan_engine import LISTEN

        win, scanner = self.win, self.win._scanner_panel
        win._stop_acquisition()
        win._device.start_rx()
        bookmarks = win._bookmarks_panel
        index = bookmarks.add_bookmark("Wrong code", 442.1e6, "FM", "5 kHz")
        bookmarks.set_dcs(index, 25, False)
        scanner.set_scan_mode("bookmarks")
        self.assertTrue(scanner.start_scan())
        self.assertFalse(self.frames(1.5, until=lambda: scanner.engine.state == LISTEN))
        scanner.stop_scan()
        self.assertEqual(scanner.hits(), [])
        bookmarks.set_dcs(index, 23, False)
        self.assertTrue(scanner.start_scan())
        self.assertTrue(self.frames(2.5, until=lambda: scanner.engine.state == LISTEN))
        scanner.stop_scan()
        hits = scanner.hits()
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].dcs, "D023N")
        self.assertIn("DCS 023 N", scanner._hit_tooltip(hits[0]))
        # Somewhere else without a code squelch; then Listen to the result.
        win.set_frequency(100.1e6)
        win._control_panel.set_tone_squelch_hz(0.0)
        scanner._table.selectRow(0)
        scanner.listen_selected()
        self.assertAlmostEqual(win._current_frequency(), 442.1e6)
        self.assertEqual(win._control_panel.dcs_squelch(), (23, False))
        self.assertIn("DCS 023 N", win._message_label.text())

    def test_tone_heard_on_the_demo_repeater(self):
        from tests.test_gui_squelch import pretend_audio_output

        win = self.win
        pretend_audio_output(win)  # (heard while it plays, as CTCSS is)
        win._audio.write = lambda _audio: None
        win._stop_acquisition()
        win._device.start_rx()
        win.set_frequency(442.1e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        heard = win._control_panel._tone_heard
        self.assertTrue(self.frames(2.0, until=lambda: heard.text() == "DCS 023 N"))


if __name__ == "__main__":
    unittest.main()
