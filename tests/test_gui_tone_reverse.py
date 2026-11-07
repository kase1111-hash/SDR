#!/usr/bin/env python3
"""
The reverse tone squelch in the window (M-Q3): CHIRP's TSQL-R and DTCS-R.

CHIRP's meaning, followed here: TSQL-R is a reverse tone squelch on the
channel's cToneFreq (the tone CHIRP shows for TSQL; Icom's radios, which
have the mode, use their tone squelch frequency), DTCS-R a reverse DCS
squelch on its DtcsCode, received with DtcsPolarity's second letter. As
Icom's manuals put it: "when a signal with the matched tone is received, the
squelch closes": muted while that tone or code is heard, every other station
heard. The Audio group's Reverse (and Radio > Audio > Reverse Tone Squelch)
sets it; bookmarks keep it (CHIRP's own modes, so files round-trip), tuning
one applies it, and a bookmark scan counts a reversed bookmark while its tone
is absent. Old bookmarks and settings load as before.
"""

import io
import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_dcs import _menu, narrow_fm  # noqa: E402
from tests.test_gui_squelch import FRAME, Frames, pretend_audio_output  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

_HEADER = (
    "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
    "DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment\n"
)
_CSV = _HEADER + (
    "1,NOT RPT,145.290000,-,0.600000,TSQL-R,88.5,100.0,023,NN,023,Tone->Tone,FM,"
    "5.00,,,\n"
    "2,NO DCS,442.100000,+,5.000000,DTCS-R,88.5,88.5,754,NR,023,Tone->Tone,FM,"
    "5.00,,,\n"
    "3,PLAIN,145.330000,,0.000000,TSQL,88.5,123.0,023,NN,023,Tone->Tone,FM,"
    "5.00,,,\n"
)


def _imported():
    from sdr_module.core.chirp_csv import read_chirp_csv

    report = read_chirp_csv(io.StringIO(_CSV), strict=False)
    return {c.name: c.to_bookmark() for c in report.channels}


class TestBookmarkSquelch(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_chirps_reverse_modes(self):
        from sdr_module.gui.bookmark_editor import bookmark_squelch

        marks = _imported()
        self.assertEqual(bookmark_squelch(marks["NOT RPT"]), (100.0, None, True))
        # DTCS-R: its DtcsCode, received with the polarity's second letter.
        self.assertEqual(bookmark_squelch(marks["NO DCS"]), (0.0, (754, True), True))
        self.assertEqual(bookmark_squelch(marks["PLAIN"]), (123.0, None, False))
        # CHIRP's defaults where a field wasn't saved.
        self.assertEqual(bookmark_squelch({"tone_mode": "TSQL-R"}), (88.5, None, True))
        self.assertEqual(
            bookmark_squelch({"tone_mode": "DTCS-R"}), (0.0, (23, False), True)
        )

    def test_damaged_or_unknown_is_no_squelch(self):
        from sdr_module.gui.bookmark_editor import bookmark_squelch

        for bookmark in (
            {"tone_mode": "TSQL-R", "ctone": "abc"},
            {"tone_mode": "TSQL-R", "ctone": float("nan")},
            {"tone_mode": "DTCS-R", "dtcs": 24},  # (no standard code)
            {"tone_mode": "DTCS-R", "dtcs": "x"},
            {},
            {"tone_mode": "Tone", "rtone": 100.0},  # (sends a tone, hears all)
        ):
            self.assertEqual(bookmark_squelch(bookmark), (0.0, None, False), bookmark)

    def test_the_older_helpers_keep_their_meaning(self):
        """``bookmark_tone_hz`` and ``bookmark_dcs`` still give the normal
        squelch only (0 / None for a reverse one), as code written for them
        (the Second Receiver window) expects."""
        from sdr_module.gui.bookmark_editor import bookmark_dcs, bookmark_tone_hz
        from sdr_module.gui.bookmarks_panel import listening

        marks = _imported()
        self.assertEqual(bookmark_tone_hz(marks["NOT RPT"]), 0.0)
        self.assertIsNone(bookmark_dcs(marks["NO DCS"]))
        self.assertEqual(bookmark_tone_hz(marks["PLAIN"]), 123.0)
        self.assertEqual(listening(marks["NOT RPT"])[3], 0.0)

    def test_tooltips_describe_them(self):
        from sdr_module.gui import bookmarks_panel

        marks = _imported()
        tip = bookmarks_panel._tooltip(marks["NOT RPT"])
        self.assertNotIn("not supported", tip)
        self.assertIn(
            "Reverse tone squelch (TSQL-R): muted while 100 Hz is heard, every "
            "other station plays",
            tip,
        )
        tip = bookmarks_panel._tooltip(marks["NO DCS"])
        self.assertNotIn("not supported", tip)
        self.assertIn("Reverse DCS (DTCS-R): muted while DCS 754 I is heard", tip)


class TestChirpRoundTrip(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def rows(self, text):
        import csv

        return list(csv.DictReader(io.StringIO(text)))

    def test_imported_and_exported_back_as_they_were(self):
        from sdr_module.core.chirp_csv import export_bookmarks_csv

        out = io.StringIO()
        export_bookmarks_csv(out, list(_imported().values()))
        before, after = self.rows(_CSV), self.rows(out.getvalue())
        self.assertEqual(len(after), len(before))
        for old, new in zip(before, after, strict=True):
            for column in (
                "Frequency",
                "Tone",
                "rToneFreq",
                "cToneFreq",
                "DtcsCode",
                "DtcsPolarity",
                "Mode",
            ):
                self.assertEqual(new[column], old[column], (old["Name"], column))

    def test_one_made_here_goes_to_chirp_and_back(self):
        from sdr_module.core.chirp_csv import (
            export_bookmarks_csv,
            import_bookmarks_csv,
            is_chirp_channel,
        )
        from sdr_module.gui.bookmark_editor import bookmark_squelch

        made = [
            {
                "label": "Not him",
                "freq_hz": 145.29e6,
                "demod": "FM",
                "fm_deviation": "5 kHz",
                "tone_mode": "TSQL-R",
                "ctone": 100.0,
            },
            {
                "label": "No code",
                "freq_hz": 442.1e6,
                "demod": "FM",
                "fm_deviation": "5 kHz",
                "tone_mode": "DTCS-R",
                "dtcs": 23,
                "dtcs_polarity": "NN",
            },
        ]
        for bookmark in made:
            self.assertFalse(is_chirp_channel(bookmark))  # (this app's own)
        out = io.StringIO()
        export_bookmarks_csv(out, made)
        rows = self.rows(out.getvalue())
        self.assertEqual(
            [(r["Tone"], r["cToneFreq"], r["DtcsCode"]) for r in rows],
            [("TSQL-R", "100.0", "023"), ("DTCS-R", "88.5", "023")],
        )
        back = import_bookmarks_csv(io.StringIO(out.getvalue()))
        self.assertEqual(
            [bookmark_squelch(b) for b in back],
            [(100.0, None, True), (0.0, (23, False), True)],
        )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReverseInTheAudioGroup(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel
        self.check = self.panel._tone_reverse_check

    def test_offered_under_the_tone_squelch_with_a_clear_explanation(self):
        win, panel, check = self.win, self.panel, self.check
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_tone_squelch_hz(0.0)
        self.assertTrue(check.isVisibleTo(panel))
        self.assertFalse(check.isEnabled())  # (Off: nothing to reverse)
        self.assertIn("Choose a CTCSS tone or DCS code", check.toolTip())
        panel.set_tone_squelch_hz(100.0)
        self.assertTrue(check.isEnabled())
        self.assertIn("mute the stations", check.toolTip())
        self.assertIn("TSQL-R and DTCS-R", check.toolTip())
        self.assertIn("half a second", check.toolTip())
        self.assertIn("Reverse", panel._tone_combo.toolTip())
        self.assertEqual(check.text(), "Reverse: mute this tone")
        panel.set_dcs_squelch(23)
        self.assertEqual(check.text(), "Reverse: mute this code")
        panel.set_tone_squelch_hz(100.0)
        self.assertEqual(check.text(), "Reverse: mute this tone")
        # Under the Tone squelch list, in FM only.
        form = panel._audio_form
        row = form.getWidgetPosition(check)[0]
        self.assertEqual(form.getWidgetPosition(panel._tone_combo)[0], row - 1)
        win._set_demod_mode("AM")
        self.assertFalse(check.isVisibleTo(panel))

    def test_it_reaches_the_tone_squelch_and_is_saved(self):
        win, panel, check = self.win, self.panel, self.check
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_tone_squelch_hz(100.0)
        check.setChecked(True)
        self.assertTrue(win._tone_squelch.reverse)
        self.assertTrue(panel.tone_reverse())
        self.assertIs(self.store.get_bool("audio/tone_reverse", False), True)
        self.assertIn(
            "Reverse tone squelch: stations sending the 100.0 Hz tone are muted",
            win._message_label.text(),
        )
        panel.set_tone_reverse(False)
        self.assertFalse(win._tone_squelch.reverse)
        self.assertIn("only stations sending", win._message_label.text())

    def test_radio_audio_reverse_tone_squelch_follows_it(self):
        win, panel, check = self.win, self.panel, self.check
        action = _menu(win, "Radio", "Audio", "Reverse Tone Squelch")
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_tone_squelch_hz(0.0)
        self.assertFalse(action.isEnabled())
        panel.set_dcs_squelch(23)
        self.assertTrue(action.isEnabled())
        action.trigger()
        self.assertTrue(check.isChecked())
        self.assertTrue(win._tone_squelch.reverse)
        check.setChecked(False)
        self.assertFalse(action.isChecked())
        self.assertIn("TSQL-R", action.toolTip() or action.statusTip())

    def test_the_info_tab_says_reverse(self):
        win, panel = self.win, self.panel
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        panel.set_dcs_squelch(23)
        panel.set_tone_reverse(True)
        text, _tone = win._squelch_description(False)
        self.assertIn("reverse DCS 023 N", text)
        panel.set_tone_squelch_hz(100.0)
        text, _tone = win._squelch_description(False)
        self.assertIn("reverse tone 100.0 Hz", text)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReverseInTheReceiver(_WindowTestCase):
    """A 70 cm channel (FM, 5 kHz) through the real receiver chain."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(446.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_squelch_db(-90)

    def listen(self, iq):
        self.win._reset_demodulator()
        frames = Frames(self, iq)
        states, played = [], []
        while frames.pos + FRAME <= len(iq):
            frames.run(frames=1)
            states.append(self.win._squelch_state(True))
            audio = frames.audio[-1]
            played.append(float(np.max(np.abs(audio))) if audio.size else 0.0)
        return played, states

    def test_the_station_with_the_tone_is_muted_the_rest_play(self):
        panel, win = self.panel, self.win
        panel.set_tone_squelch_hz(100.0)
        panel.set_tone_reverse(True)
        played, states = self.listen(narrow_fm(2.0, ctcss_hz=100.0))
        self.assertLess(max(played), 1e-3)  # (never, its start neither)
        self.assertEqual(states[-1], "reverse")
        self.assertFalse(win._scanner_host.scan_signal_present())
        meter = win._level_meter
        self.assertIn("reverse tone squelch", meter.tooltip_text())
        heard = panel._tone_heard
        self.assertIn("100.0 Hz", heard.text())
        self.assertIn("reverse tone squelch mutes it", heard.toolTip())
        self.assertEqual(heard.property("tone"), "warning")
        # Another station: heard after about half a second.
        played, states = self.listen(narrow_fm(2.0, ctcss_hz=123.0))
        first = next(i for i, peak in enumerate(played) if peak > 0.05)
        self.assertLessEqual(first * FRAME / 2.4e6, 0.7)
        self.assertEqual(states[-1], "open")
        self.assertTrue(win._scanner_host.scan_signal_present())
        self.assertNotIn("mutes it", heard.toolTip())
        played, _states = self.listen(narrow_fm(2.0))
        self.assertGreater(max(played[-20:]), 0.05)

    def test_a_reverse_dcs_squelch(self):
        panel = self.panel
        panel.set_dcs_squelch(23)
        panel.set_tone_reverse(True)
        played, states = self.listen(narrow_fm(2.0, code=23))
        self.assertLess(max(played), 1e-3)
        self.assertEqual(states[-1], "reverse")
        self.assertEqual(panel._tone_heard.property("tone"), "warning")
        # (The same bits: 047 inverted is muted too.)
        panel.set_dcs_squelch(47, True)
        played, _states = self.listen(narrow_fm(2.0, code=23))
        self.assertLess(max(played), 1e-3)
        played, states = self.listen(narrow_fm(2.0, code=25))
        self.assertGreater(max(played[-20:]), 0.05)
        self.assertEqual(states[-1], "open")


class TestReverseSettings(_IniSettings):
    def window(self, **values):
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", **values)
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        return win

    def test_restored(self):
        win = self.window(audio__tone_squelch="100.0", audio__tone_reverse="true")
        self.assertTrue(win._control_panel.tone_reverse())
        self.assertTrue(win._tone_squelch.reverse)
        self.assertEqual(win._tone_squelch.tone_hz, 100.0)

    def test_old_settings_load_unchanged(self):
        win = self.window(audio__tone_squelch="100.0")
        self.assertFalse(win._control_panel.tone_reverse())
        self.assertFalse(win._tone_squelch.reverse)
        self.assertEqual(win._tone_squelch.tone_hz, 100.0)

    def test_a_damaged_value_is_no_reverse_and_logged(self):
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            win = self.window(audio__tone_reverse="maybe")
        self.assertFalse(win._tone_squelch.reverse)
        self.assertTrue(any("audio/tone_reverse" in m for m in logs.output))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReverseBookmarksInTheWindow(_WindowTestCase):
    def add(self, bookmark):
        panel = self.win._bookmarks_panel
        panel.add_bookmark(bookmark["label"], bookmark["freq_hz"])
        index = len(panel.bookmarks()) - 1
        panel._bookmarks[index] = dict(bookmark)
        return index

    def test_tuning_one_applies_it(self):
        win, panel = self.win, self.win._bookmarks_panel
        marks = _imported()
        win._set_demod_mode("AM")
        panel._tune_index(self.add(marks["NOT RPT"]))
        cp = win._control_panel
        self.assertEqual((cp.tone_squelch_hz(), cp.tone_reverse()), (100.0, True))
        self.assertTrue(win._tone_squelch.reverse)
        self.assertIn(
            "Tuned to NOT RPT (FM, reverse tone squelch 100.0 Hz)",
            win._message_label.text(),
        )
        panel._tune_index(self.add(marks["NO DCS"]))
        self.assertEqual((cp.dcs_squelch(), cp.tone_reverse()), ((754, True), True))
        self.assertIn("(FM, reverse DCS 754 I)", win._message_label.text())
        # A bookmark with a normal tone squelch isn't reversed.
        panel._tune_index(self.add(marks["PLAIN"]))
        self.assertEqual((cp.tone_squelch_hz(), cp.tone_reverse()), (123.0, False))
        self.assertFalse(win._tone_squelch.reverse)

    def test_one_message_for_the_tune(self):
        win, panel = self.win, self.win._bookmarks_panel
        said = []
        original = win._show_status_message

        def record(text, *args, **kwargs):
            said.append(text)
            return original(text, *args, **kwargs)

        win._show_status_message = record
        panel._tune_index(self.add(_imported()["NOT RPT"]))
        self.assertFalse([t for t in said if t.startswith("Reverse tone squelch")])

    def test_ctrl_b_keeps_it(self):
        win = self.win
        win.set_frequency(145.29e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_tone_squelch_hz(100.0)
        win._control_panel.set_tone_reverse(True)
        win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual((saved["tone_mode"], saved["ctone"]), ("TSQL-R", 100.0))
        self.assertIn("(FM, reverse tone squelch 100.0 Hz)", win._message_label.text())
        win._control_panel.set_dcs_squelch(23)
        win.set_frequency(442.1e6)
        win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual((saved["tone_mode"], saved["dtcs"]), ("DTCS-R", 23))


class TestEditBookmarkReverse(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def dialog(self, bookmark):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        dialog = BookmarkEditDialog(bookmark, ("FM", "5 kHz"), [])
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_offered_and_saved(self):
        base = {"label": "Rpt", "freq_hz": 145.29e6, "demod": "FM"}
        dialog = self.dialog(base)
        self.assertFalse(dialog._reverse.isEnabled())  # (no tone chosen)
        dialog._tone.setCurrentIndex(dialog._tone.findData(100.0))
        self.assertTrue(dialog._reverse.isEnabled())
        dialog._reverse.setChecked(True)
        edited = dialog.bookmark()
        self.assertEqual((edited["tone_mode"], edited["ctone"]), ("TSQL-R", 100.0))
        # Opened again it shows it; a code instead keeps it reversed.
        dialog = self.dialog(edited)
        self.assertTrue(dialog._reverse.isChecked())
        dialog._tone.setCurrentIndex(dialog._tone.findData("D023N"))
        coded = dialog.bookmark()
        self.assertEqual(
            (coded["tone_mode"], coded["dtcs"], coded["dtcs_polarity"]),
            ("DTCS-R", 23, "NN"),
        )
        # None: no squelch at all.
        dialog = self.dialog(coded)
        dialog._tone.setCurrentIndex(0)
        cleared = dialog.bookmark()
        for key in ("tone_mode", "dtcs", "dtcs_polarity"):
            self.assertNotIn(key, cleared)

    def test_only_reversed_keeps_the_rest_as_it_was(self):
        """Reverse ticked or unticked alone changes the mode only: a CHIRP
        channel's other tone fields (its polarities) stay as they were."""
        marks = _imported()
        dialog = self.dialog(marks["NO DCS"])
        self.assertTrue(dialog._reverse.isChecked())
        dialog._reverse.setChecked(False)
        edited = dialog.bookmark()
        self.assertEqual(edited["tone_mode"], "DTCS")
        for key in ("dtcs", "dtcs_polarity", "ctone", "rtone"):
            self.assertEqual(edited.get(key), marks["NO DCS"].get(key), key)
        untouched = self.dialog(marks["NOT RPT"]).bookmark()
        self.assertEqual(untouched["tone_mode"], "TSQL-R")


def _demo_signal(kind):
    """(seed, (period, on), frequency) of a demo repeater."""
    from sdr_module.gui.device_dialog import _DEMO_SIGNALS

    for seed, (freq, k, _level, duty) in enumerate(_DEMO_SIGNALS):
        if k == kind:
            return seed, duty, freq
    raise AssertionError(kind)


def _on_at(kind, into=1.0):
    """A stream time (s) ``into`` seconds into one of a demo repeater's
    transmissions."""
    seed, (period, _on), _freq = _demo_signal(kind)
    shift = (seed * 1.7) % period
    start = -shift
    while start < 0.5:
        start += period
    return start + into


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestReverseBookmarkScanOnTheDemo(_WindowTestCase):
    """The demo's 2 m repeater (145.290 MHz) sends 100.0 Hz, its 70 cm one
    (442.100 MHz) DCS 023 N: a reversed bookmark is a hit only where the
    tone or code it mutes isn't sent."""

    demo = True

    def frames(self, seconds, until=None):
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

    def scan(self, kind, freq, tone=0.0, code=None):
        from sdr_module.gui.scan_engine import LISTEN

        win, scanner = self.win, self.win._scanner_panel
        win._stop_acquisition()
        win._device.start_rx()
        rate = win._device_sample_rate()
        win._device._stream_pos = int(_on_at(kind) * rate)
        bookmarks = win._bookmarks_panel
        for index in reversed(range(len(bookmarks.bookmarks()))):
            bookmarks._bookmarks.pop(index)
        index = bookmarks.add_bookmark("Reversed", freq, "FM", "5 kHz")
        if code is not None:
            bookmarks.set_dcs(index, code, False, reverse=True)
        else:
            bookmarks.set_tone(index, tone, reverse=True)
        scanner.set_scan_mode("bookmarks")
        self.assertTrue(scanner.start_scan())
        stopped = self.frames(2.5, until=lambda: scanner.engine.state == LISTEN)
        scanner.stop_scan()
        return stopped, scanner.hits()

    def test_muted_where_its_tone_is_sent(self):
        stopped, hits = self.scan("rpt", 145.29e6, tone=100.0)
        self.assertFalse(stopped)
        self.assertEqual(hits, [])

    def test_heard_where_another_tone_is(self):
        stopped, hits = self.scan("rpt", 145.29e6, tone=123.0)
        self.assertTrue(stopped)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].reverse)
        self.assertEqual(hits[0].tone_hz, 123.0)
        tip = self.win._scanner_panel._hit_tooltip(hits[0])
        self.assertIn("reverse tone squelch 123.0 Hz", tip)
        # Listen sets it reversed.
        win = self.win
        win.set_frequency(100.1e6)
        win._control_panel.set_tone_squelch_hz(0.0)
        win._control_panel.set_tone_reverse(False)
        scanner = win._scanner_panel
        scanner._table.selectRow(0)
        scanner.listen_selected()
        self.assertEqual(win._control_panel.tone_squelch_hz(), 123.0)
        self.assertTrue(win._control_panel.tone_reverse())
        self.assertIn("reverse tone squelch 123.0 Hz", win._message_label.text())

    def test_dcs_muted_where_its_code_is_sent_heard_where_another(self):
        stopped, hits = self.scan("dcs", 442.1e6, code=23)
        self.assertFalse(stopped)
        self.assertEqual(hits, [])
        stopped, hits = self.scan("dcs", 442.1e6, code=25)
        self.assertTrue(stopped)
        self.assertEqual((hits[0].dcs, hits[0].reverse), ("D025N", True))

    def test_the_scan_puts_back_the_squelch_it_began_with(self):
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_tone_squelch_hz(151.4)
        said = []
        show = win._show_status_message

        def spy(text, *args, **kwargs):
            said.append(text)
            return show(text, *args, **kwargs)

        with mock.patch.object(win, "_show_status_message", spy):
            stopped, _hits = self.scan("rpt", 145.29e6, tone=100.0)
        self.assertFalse(stopped)  # (stopped while it moved on: it goes back)
        self.assertEqual(win._control_panel.tone_squelch_hz(), 151.4)
        self.assertFalse(win._control_panel.tone_reverse())
        self.assertFalse(win._tone_squelch.reverse)
        # The Reverse it sets for each bookmark (and back) isn't said each
        # time: the status bar keeps the scan's own messages.
        self.assertTrue(said)
        self.assertEqual([t for t in said if "Tone squelch:" in t], [])
        self.assertEqual([t for t in said if "Reverse tone squelch:" in t], [])


class TestScanResultsKeepIt(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_saved_and_loaded(self):
        from sdr_module.gui.scanner_panel import ScanHit

        hit = ScanHit(145.29e6, -50.0, 30.0, 10e3, "FM", "5 kHz", "", "Rpt", 0, 0, 1)
        hit.tone_hz, hit.reverse = 100.0, True
        self.assertTrue(ScanHit.from_dict(hit.to_dict()).reverse)
        # Older results (no "reverse") and damaged values: not reversed.
        data = hit.to_dict()
        del data["reverse"]
        self.assertFalse(ScanHit.from_dict(data).reverse)
        for damaged in ("yes", 1, "true", None):
            data = dict(hit.to_dict(), reverse=damaged)
            self.assertFalse(ScanHit.from_dict(data).reverse, damaged)
        # (Nothing to reverse without a tone or code.)
        data = dict(hit.to_dict(), tone_hz=0.0)
        self.assertFalse(ScanHit.from_dict(data).reverse)

    def test_an_older_host_is_tuned_as_before(self):
        """A host written for the old ``scan_tune_channel`` (no ``reverse``)
        is called without it for a bookmark that isn't reversed."""
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        calls = []

        class Host:
            def scan_tune_channel(self, freq, listening, tone=0.0):
                calls.append((freq, tone))
                return True

            def scan_lo_moved(self):
                return False

            def scan_lo_hz(self):
                return 145.0e6

            def scan_signal_present(self):
                return False

            def scan_set_muted(self, muted):
                pass

        visit = {
            "index": 0,
            "label": "Rpt",
            "freq_hz": 145.29e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
            "bandwidth": "",
            "tone_hz": 100.0,
        }
        engine = ScanEngine(Host())
        engine.start(ScanSettings("bookmarks", bookmarks=[visit]), 2.4e6, 0.0)
        self.assertEqual(calls, [(145.29e6, 100.0)])
        self.assertFalse(engine.current.reverse)


if __name__ == "__main__":
    unittest.main()
