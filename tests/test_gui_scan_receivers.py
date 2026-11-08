"""M-S1: the Scanner tab says which receiver found each signal.

With "Use both receivers" (J-dual) the two receivers sweep a range
together and the results are one list, but nothing said which receiver
read each signal (``sdr-scan scan --dual`` does: its Receiver column). The
two receivers' levels aren't calibrated against each other, so a level
means little without knowing whose it is.

Now each result of a sweep with both receivers says whose it is: Main (the
main window's), Second (the Second Receiver window's) or Both (a signal
both read, where their parts of the range meet; its level and SNR are the
stronger reading, and its tooltip says whose and what the other read), in
an RX column (Main, 2nd, Both) shown only when the sweep used both
receivers, after SNR (the level and SNR are its reading; the levels drop
their unit then, so that the frequency keeps its room), in the results'
tooltips, kept through sorting, Listen and Bookmark, saved with the
results, and in Export Results (CSV: Receiver and Level from).
Listen on a result only the second receiver can tune (above what the main
receiver tunes) tunes the Second Receiver window there. Single-receiver
scans look exactly as before.
"""

import csv
import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import _FakeSettings, require_pyqt6  # noqa: E402
from tests.test_gui_scan_both import (  # noqa: E402
    FRAME_S,
    _block,
    _BothCase,
    _demo,
    _FakePartner,
    _Receiver,
    _run,
)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - the GUI tests need PyQt6
    pass

OLD_CSV_HEADER = [
    "Frequency (MHz)",
    "Level (dBFS)",
    "SNR (dB)",
    "Width (kHz)",
    "Mode",
    "FM deviation",
    "Filter",
    "Band",
    "Name",
    "First heard",
    "Last heard",
    "Times heard",
]


# --------------------------------------------------------------------------- #
# The engine says whose reading each signal is
# --------------------------------------------------------------------------- #
class TestTheEngine(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def sweep(self, partner=None):
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        main = _Receiver(_demo(2.4e6))
        engine = ScanEngine(main)
        found = []
        engine.on_found = found.append
        settings = ScanSettings("once", 88e6, 108e6, 200e3, min_width_hz=30e3)
        self.assertEqual(engine.start(settings, 2.4e6, 0.0, partner=partner), "")
        _run(engine, main, partner)
        self.assertFalse(engine.active)
        return main, found

    def test_a_sweep_with_both_says_which_receiver_read_each_signal(self):
        from sdr_module.gui.scan_engine import MAIN, PARTNER

        partner = _Receiver(_demo(3.2e6, 98e6))
        main, found = self.sweep(partner)
        receivers = {f.receiver for f in found}
        self.assertEqual(receivers, {MAIN, PARTNER})
        # Each signal is in the part of the range its receiver swept.
        top = max(main.tunes) + 0.5 * 0.8 * 2.4e6
        for item in found:
            if item.receiver == MAIN:
                self.assertLessEqual(item.frequency_hz, top + 200e3)
            else:
                self.assertGreater(item.frequency_hz, max(main.tunes) - 1.3e6)

    def test_a_sweep_with_one_says_nothing(self):
        _main, found = self.sweep()
        self.assertTrue(found)
        self.assertTrue(all(f.receiver is None for f in found))


# --------------------------------------------------------------------------- #
# The tab (fake receivers: an RTL-SDR-like main one up to 1766 MHz, and a
# second one that tunes higher)
# --------------------------------------------------------------------------- #
class _TabCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import scanner_panel
        from sdr_module.gui.scanner_panel import ScannerPanel
        from tests.test_gui_scanner import FakeWindow

        self.mod = scanner_panel
        self.host = FakeWindow()
        self.host.range = (24e6, 1766e6)
        self.partner = _FakePartner(_demo(2.4e6, 1800e6), tunes=(1e6, 3000e6))
        self.offer = (self.partner, "")
        self.host.scan_partner = lambda: self.offer
        # The window's Second Receiver, as Listen finds it.
        self.second_listened = []
        self.second_answer = (True, "")
        self.host.scan_listen_second = (
            lambda hit: self.second_listened.append(hit) or self.second_answer
        )
        self.store = _FakeSettings()
        self.panel = ScannerPanel(settings=self.store)
        self.panel.set_host(self.host)
        self.panel.resize(420, 600)
        self.panel.show()
        QApplication.processEvents()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        self.panel.engine.stop()
        self.panel.close()
        self.panel.deleteLater()
        QApplication.processEvents()

    def found(self, freq, peak, receiver, width=12e3, noise=-90.0):
        from sdr_module.gui.scan_engine import Found

        item = Found(freq, peak, noise, width, ("FM", "5 kHz", "12.5 kHz"))
        item.receiver = receiver
        return item

    def cell(self, row, col):
        return self.panel._table.item(row, col).text()

    def row_of(self, freq):
        table = self.panel._table
        for row in range(table.rowCount()):
            hit = self.panel._hits[table.item(row, self.mod.COL_FREQ).data(
                self.mod._HIT_ROLE)]  # fmt: skip
            if abs(hit.frequency_hz - freq) < 1.0:
                return row
        raise AssertionError(f"no row for {freq}")

    def sweep_with_both(self):
        """A sweep of 1700-1900 MHz with both receivers, run to its end."""
        panel = self.panel
        panel._start.setValue(1700.0)
        panel._end.setValue(1900.0)
        panel._step.setValue(100.0)
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        for _ in range(int(20 / FRAME_S)):
            if not panel.engine.active:
                break
            panel.engine.frame(_block(self.host))
            if panel.engine.active and self.partner.feed is not None:
                self.partner.feed(_block(self.partner))
        self.assertFalse(panel.engine.active)

    def export(self):
        path = os.path.join(self.tmp.name, "scan.csv")
        self.panel.export_csv(path)
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.reader(fh))


class TestTheResults(_TabCase):
    def test_a_sweep_with_both_shows_a_receiver_column(self):
        panel, mod = self.panel, self.mod
        table = panel._table
        self.assertTrue(table.isColumnHidden(mod.COL_RECEIVER))
        panel._on_found(self.found(1720e6, -50.0, 0))
        panel._on_found(self.found(1850e6, -40.0, 1))
        self.assertFalse(table.isColumnHidden(mod.COL_RECEIVER))
        title = table.horizontalHeaderItem(mod.COL_RECEIVER)
        self.assertEqual(title.text(), "RX")
        self.assertIn("Second Receiver", title.toolTip())
        self.assertEqual(self.cell(self.row_of(1720e6), mod.COL_RECEIVER), "Main")
        self.assertEqual(self.cell(self.row_of(1850e6), mod.COL_RECEIVER), "2nd")
        tip = table.item(self.row_of(1850e6), mod.COL_LEVEL).toolTip()
        self.assertIn("Read by the second receiver", tip)
        self.assertEqual([h.receiver for h in panel.hits()], ["main", "second"])

    def test_the_level_header_says_whose_scale_each_level_is_on(self):
        """ui-6 (final review): with the RX column shown, the Level header's
        tooltip still said every level was on the spectrum's scale, which a
        2nd result's isn't (the Second Receiver's own dBFS, as the RX
        header and the row's tooltip say)."""
        from sdr_module.gui.scanner_panel import ScannerPanel

        panel, mod = self.panel, self.mod
        title = panel._table.horizontalHeaderItem(mod.COL_LEVEL)
        single = title.toolTip()
        self.assertEqual(
            single,
            "Strongest FFT bin, in dB relative to full scale (same scale as "
            "the spectrum)",
        )  # (as before)
        panel._on_found(self.found(1720e6, -50.0, 0))
        panel._on_found(self.found(1850e6, -40.0, 1))  # (the second's)
        both = title.toolTip()
        self.assertNotIn("same scale as the spectrum", both)
        for words in (
            "relative to full scale",  # (the cells drop their unit)
            "the receiver that read it",
            "the Second Receiver's",
            "aren't calibrated against each other",
        ):
            self.assertIn(words, both)
        # Restored with the results at startup.
        panel._save_results()
        again = ScannerPanel(settings=self.store)
        self.addCleanup(again.deleteLater)
        self.assertEqual(
            again._table.horizontalHeaderItem(mod.COL_LEVEL).toolTip(), both
        )
        # Results of one receiver again (Clear, or a new scan): as before.
        panel.clear_results()
        self.assertTrue(panel._table.isColumnHidden(mod.COL_RECEIVER))
        self.assertEqual(title.toolTip(), single)

    def test_the_frequency_keeps_its_room(self):
        # The right column is about 384 px inside at 1024x640 (and 370 at
        # 1366x768): "1850.000 MHz" still fits beside the RX column.
        panel, mod = self.panel, self.mod
        table = panel._table
        panel.resize(400, 600)
        QApplication.processEvents()
        panel._on_found(self.found(1720e6, -52.4, None))
        self.assertEqual(self.cell(0, mod.COL_LEVEL), "-52 dBFS")  # (as before)
        panel._on_found(self.found(1850e6, -40.0, 1))
        QApplication.processEvents()
        header = table.horizontalHeader()
        self.assertEqual(header.visualIndex(mod.COL_RECEIVER), mod.COL_SNR + 1)
        self.assertEqual(self.cell(self.row_of(1720e6), mod.COL_LEVEL), "-52")
        self.assertEqual(self.cell(self.row_of(1850e6), mod.COL_LEVEL), "-40")
        cells = table.fontMetrics()
        self.assertGreaterEqual(
            table.columnWidth(mod.COL_FREQ),
            cells.horizontalAdvance("1850.000 MHz") + 14,
        )
        for col in (mod.COL_LEVEL, mod.COL_SNR, mod.COL_RECEIVER, mod.COL_MODE,
                    mod.COL_HEARD):  # fmt: skip
            for row in range(table.rowCount()):
                text = self.cell(row, col)
                self.assertLessEqual(
                    cells.horizontalAdvance(text) + 8, table.columnWidth(col), text
                )
        # A scan with one receiver: the unit is back.
        panel._both.setChecked(False)
        panel._start.setValue(1700.0)
        panel._end.setValue(1760.0)
        self.assertTrue(panel.start_scan())
        panel._on_found(self.found(1720e6, -52.4, None))
        self.assertEqual(self.cell(0, mod.COL_LEVEL), "-52 dBFS")
        panel.stop_scan()

    def test_a_signal_both_read_says_both_and_whose_level_it_shows(self):
        panel, mod = self.panel, self.mod
        panel._on_found(self.found(1766.0e6, -55.0, 0))
        panel._on_found(self.found(1766.0e6, -48.0, 1))  # (stronger: second's)
        panel._on_found(self.found(1766.0e6, -60.0, 0))  # (weaker: kept)
        self.assertEqual(len(panel.hits()), 1)
        hit = panel.hits()[0]
        self.assertEqual(hit.peak_db, -48.0)
        self.assertEqual(hit.receiver, "second")
        self.assertEqual(hit.readings, {"main": -55.0, "second": -48.0})
        self.assertEqual(self.cell(0, mod.COL_RECEIVER), "Both")
        tip = panel._table.item(0, mod.COL_LEVEL).toolTip()
        self.assertIn("Read by both receivers", tip)
        self.assertIn("the second receiver's", tip)
        self.assertIn("main receiver read -55.0 dBFS", tip)

    def test_kept_through_sorting(self):
        panel, mod = self.panel, self.mod
        for freq, peak, receiver in (
            (1720e6, -50.0, 0),
            (1850e6, -40.0, 1),
            (1700e6, -30.0, 0),
            (1890e6, -70.0, 1),
        ):
            panel._on_found(self.found(freq, peak, receiver))
        expected = {1720e6: "Main", 1850e6: "2nd", 1700e6: "Main",
                    1890e6: "2nd"}  # fmt: skip
        header = panel._table.horizontalHeader()
        for col in (mod.COL_LEVEL, mod.COL_RECEIVER, mod.COL_FREQ):
            for order in (Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder):
                header.setSortIndicator(col, order)
                panel._resort()
                for freq, text in expected.items():
                    self.assertEqual(
                        self.cell(self.row_of(freq), mod.COL_RECEIVER), text
                    )
        # By receiver: the main one's first, then both, then the second's
        # (their parts of the range in order), not "2nd" before "Main".
        panel._on_found(self.found(1720e6, -45.0, 1))  # (both read 1720 MHz)
        header.setSortIndicator(mod.COL_RECEIVER, Qt.SortOrder.AscendingOrder)
        panel._resort()
        texts = [self.cell(r, mod.COL_RECEIVER) for r in range(4)]
        self.assertEqual(texts, ["Main", "Both", "2nd", "2nd"])
        header.setSortIndicator(mod.COL_RECEIVER, Qt.SortOrder.DescendingOrder)
        panel._resort()
        texts = [self.cell(r, mod.COL_RECEIVER) for r in range(4)]
        self.assertEqual(texts, ["2nd", "2nd", "Both", "Main"])

    def test_export_has_the_receiver(self):
        panel = self.panel
        panel._on_found(self.found(1720e6, -50.0, 0))
        panel._on_found(self.found(1850e6, -40.0, 1))
        panel._on_found(self.found(1766e6, -55.0, 0))
        panel._on_found(self.found(1766e6, -48.0, 1))
        rows = self.export()
        self.assertEqual(rows[0], OLD_CSV_HEADER + ["Receiver", "Level from"])
        by_freq = {row[0]: row[-2:] for row in rows[1:]}
        self.assertEqual(by_freq["1720.000000"], ["Main", "Main"])
        self.assertEqual(by_freq["1850.000000"], ["Second", "Second"])
        self.assertEqual(by_freq["1766.000000"], ["Both", "Second"])

    def test_saved_and_restored_with_the_results(self):
        from sdr_module.gui.scanner_panel import ScannerPanel

        panel, mod = self.panel, self.mod
        panel._on_found(self.found(1720e6, -50.0, 0))
        panel._on_found(self.found(1766e6, -55.0, 0))
        panel._on_found(self.found(1766e6, -48.0, 1))
        panel._save_results()
        again = ScannerPanel(settings=self.store)
        self.addCleanup(again.deleteLater)
        hits = sorted(again.hits(), key=lambda h: h.frequency_hz)
        self.assertEqual([h.receiver for h in hits], ["main", "second"])
        self.assertEqual(hits[1].readings, {"main": -55.0, "second": -48.0})
        self.assertFalse(again._table.isColumnHidden(mod.COL_RECEIVER))
        levels = [again._table.item(r, mod.COL_LEVEL).text() for r in range(2)]
        self.assertEqual(sorted(levels), ["-48", "-50"])
        # Damaged values are dropped, the result kept.
        raw = json.loads(self.store.values["scanner/results"])
        raw[0]["receiver"] = 7
        raw[0]["readings"] = {"main": "x", "third": -1.0, "second": float("nan")}
        self.store.values["scanner/results"] = json.dumps(raw)
        damaged = ScannerPanel(settings=self.store)
        self.addCleanup(damaged.deleteLater)
        hit = [h for h in damaged.hits() if h.frequency_hz == raw[0]["frequency_hz"]]
        self.assertEqual((hit[0].receiver, hit[0].readings), ("", {}))

    def test_single_receiver_scans_look_exactly_as_before(self):
        panel, mod = self.panel, self.mod
        panel._start.setValue(1700.0)
        panel._end.setValue(1760.0)
        panel._step.setValue(100.0)
        panel._both.setChecked(False)
        self.assertTrue(panel.start_scan())
        for _ in range(int(10 / FRAME_S)):
            if not panel.engine.active:
                break
            panel.engine.frame(_block(self.host))
        panel._on_found(self.found(1720e6, -50.0, None))
        self.assertTrue(panel._table.isColumnHidden(mod.COL_RECEIVER))
        self.assertTrue(all(h.receiver == "" for h in panel.hits()))
        self.assertNotIn("receiver", panel._table.item(0, mod.COL_LEVEL).toolTip())
        self.assertEqual(self.export()[0], OLD_CSV_HEADER)

    def test_a_real_sweep_with_both_fills_it_in(self):
        panel, mod = self.panel, self.mod
        self.sweep_with_both()
        hits = panel.hits()
        self.assertTrue(hits)
        self.assertTrue(all(h.receiver in ("main", "second") for h in hits))
        self.assertFalse(panel._table.isColumnHidden(mod.COL_RECEIVER))
        # What only the second receiver tunes (above 1766 MHz) it read.
        for hit in hits:
            if hit.frequency_hz > 1766.1e6:
                self.assertEqual(hit.receiver, "second", hit.frequency_hz)

    def test_a_new_scan_with_one_receiver_hides_the_column_again(self):
        panel, mod = self.panel, self.mod
        panel._on_found(self.found(1720e6, -50.0, 0))
        self.assertFalse(panel._table.isColumnHidden(mod.COL_RECEIVER))
        panel._both.setChecked(False)
        panel._start.setValue(1700.0)
        panel._end.setValue(1760.0)
        self.assertTrue(panel.start_scan())
        self.assertTrue(panel._table.isColumnHidden(mod.COL_RECEIVER))
        panel.stop_scan()


class TestListenAndBookmark(_TabCase):
    def select(self, freq):
        self.panel._table.selectRow(self.row_of(freq))
        QApplication.processEvents()

    def test_listen_tunes_the_main_receiver_where_it_tunes(self):
        panel = self.panel
        panel._on_found(self.found(1720e6, -50.0, 1))  # (second's, main tunes it)
        self.select(1720e6)
        panel.listen_selected()
        self.assertEqual([h.frequency_hz for h in self.host.listened], [1720e6])
        self.assertEqual(self.host.listened[0].receiver, "second")
        self.assertEqual(self.second_listened, [])

    def test_listen_tunes_the_second_receiver_where_only_it_tunes(self):
        panel = self.panel
        panel._on_found(self.found(1850e6, -40.0, 1))
        self.select(1850e6)
        panel.listen_selected()
        self.assertEqual(self.host.listened, [])
        self.assertEqual([h.frequency_hz for h in self.second_listened], [1850e6])
        self.assertIn("in the Second Receiver window", panel._status.text())
        self.assertIn("1850.000 MHz", panel._status.text())

    def test_a_silent_second_receiver_says_how_to_hear_it(self):
        panel = self.panel
        panel._on_found(self.found(1850e6, -40.0, 1))
        self.second_answer = (True, "its Listen is Main window: choose This "
                              "receiver or Both (mixed) there to hear it.")  # fmt: skip
        self.select(1850e6)
        panel.listen_selected()
        text = panel._status.text()
        self.assertIn("Tuned the Second Receiver window to 1850.000 MHz", text)
        self.assertIn("choose This receiver or Both (mixed) there", text)
        self.assertNotIn("Listening", text)

    def test_without_the_second_receiver_it_says_why_not(self):
        panel = self.panel
        panel._on_found(self.found(1850e6, -40.0, 1))
        self.second_answer = (False, "start it in its window, then Listen again.")
        self.select(1850e6)
        panel.listen_selected()
        self.assertEqual(self.host.listened, [])
        text = panel._status.text()
        self.assertIn("Only the second receiver tunes 1850.000 MHz", text)
        self.assertIn("start it in its window", text)
        # A host without a second receiver at all.
        del self.host.scan_listen_second
        panel.listen_selected()
        self.assertEqual(self.host.listened, [])
        text = panel._status.text()
        self.assertIn("Only the second receiver tunes 1850.000 MHz", text)
        self.assertIn("Device > Second Receiver", text)

    def test_bookmark_keeps_the_column_and_the_receiver(self):
        panel, mod = self.panel, self.mod
        panel._on_found(self.found(1850e6, -40.0, 1))
        self.select(1850e6)
        panel.bookmark_selected()
        self.assertEqual(self.host.bookmarked[-1].receiver, "second")
        self.assertFalse(panel._table.isColumnHidden(mod.COL_RECEIVER))
        self.assertEqual(self.cell(0, mod.COL_RECEIVER), "2nd")


# --------------------------------------------------------------------------- #
# In the window: a sweep with the Second Receiver window
# --------------------------------------------------------------------------- #
class TestInTheWindow(_BothCase):
    def test_a_sweep_with_both_says_which_read_each(self):
        from sdr_module.gui import scanner_panel as mod

        panel = self.panel
        self.open_second()
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(5.0, until=lambda: not panel.engine.active))
        hits = panel.hits()
        self.assertEqual(len(hits), 18)
        self.assertEqual({h.receiver for h in hits}, {"main", "second"})
        self.assertFalse(panel._table.isColumnHidden(mod.COL_RECEIVER))
        texts = {
            panel._table.item(r, mod.COL_RECEIVER).text()
            for r in range(panel._table.rowCount())
        }
        self.assertTrue({"Main", "2nd"} <= texts, texts)
        # The main receiver swept up from the bottom, the second down.
        mains = [h.frequency_hz for h in hits if h.receiver == "main"]
        seconds = [h.frequency_hz for h in hits if h.receiver == "second"]
        self.assertLess(min(mains), min(seconds))
        self.assertLess(max(mains), max(seconds))

    def sweep_and_pick(self):
        """A sweep of the FM band with both receivers, and the highest
        result the second receiver read, as if the main receiver tuned
        only below it (as an RTL-SDR beside a HackRF One above 1766 MHz)."""
        panel = self.panel
        self.fm_sweep()
        panel.refresh()
        panel._both.setChecked(True)
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(5.0, until=lambda: not panel.engine.active))
        hit = max(
            (h for h in panel.hits() if h.receiver == "second"),
            key=lambda h: h.frequency_hz,
        )
        host = panel._host
        host.scan_device_range = lambda: (24e6, hit.frequency_hz - 1e6)
        table = panel._table
        for row in range(table.rowCount()):
            if panel.selected_hit() is hit:
                break
            table.selectRow(row)
        self.assertIs(panel.selected_hit(), hit)
        return hit

    def test_listen_tunes_the_second_receiver_window(self):
        from sdr_module.gui.second_receiver import LISTEN_BOTH, LISTEN_MAIN

        panel = self.panel
        rx = self.open_second()
        rx.set_listen(LISTEN_BOTH)
        hit = self.sweep_and_pick()
        main_before = self.win._current_frequency()
        panel.listen_selected()
        self.assertAlmostEqual(rx.tuned_frequency(), hit.frequency_hz, delta=1.0)
        self.assertEqual((rx.mode(), rx._deviation_hz()), ("FM", 75e3))
        self.assertEqual(self.win._current_frequency(), main_before)
        text = panel._status.text()
        self.assertIn("in the Second Receiver window", text)
        # Its Listen set to the main window's audio: tuned, and it says so.
        rx.set_listen(LISTEN_MAIN)
        rx.tune(hit.frequency_hz - 2e6)
        panel.listen_selected()
        self.assertAlmostEqual(rx.tuned_frequency(), hit.frequency_hz, delta=1.0)
        text = panel._status.text()
        self.assertIn("Tuned the Second Receiver window", text)
        self.assertIn("This receiver or Both", text)

    def test_listen_says_what_to_do_when_the_second_receiver_cant(self):
        panel = self.panel
        rx = self.open_second()
        hit = self.sweep_and_pick()
        main_before = self.win._current_frequency()
        rx.stop()
        panel.listen_selected()
        text = panel._status.text()
        self.assertIn("Only the second receiver tunes", text)
        self.assertIn("start it", text)
        self.assertEqual(self.win._current_frequency(), main_before)
        # Out of what it tunes: refused, and its window says why.
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        hit.frequency_hz = 7e9
        panel.listen_selected()
        text = panel._status.text()
        self.assertIn("Only the second receiver tunes", text)
        self.assertIn("its window says why", text)
        rx.hide()
        panel.listen_selected()
        self.assertIn("Device > Second Receiver", panel._status.text())


if __name__ == "__main__":
    unittest.main()
