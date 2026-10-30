"""F1 (Help > Keyboard Shortcuts) lists the keys the app really answers to.

mc:MC-11: the list missed every key a list, a field or a window handles
itself (Enter, F2 and Delete in Bookmarks, the waterfall's COLORS scale,
playback, the second receiver...), and its filter found "Ctrl+S" inside
"Ctrl+Shift+R". Here every row of ``help_dialog.PANEL_KEYS`` is pressed on
the real widget (one test per row id, and a test that no row goes
unchecked), the plot keys are pressed on the plots, and every shortcut of
the menus is in the dialog.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from sdr_module.gui.help_dialog import PANEL_KEYS, PLOT_KEYS, HelpDialog

SHIFT = Qt.KeyboardModifier.ShiftModifier if HAS_PYQT6 else None
CTRL = Qt.KeyboardModifier.ControlModifier if HAS_PYQT6 else None


def _action(window, keys):
    """The enabled action of ``window`` bound to ``keys``."""
    wanted = QKeySequence(keys)
    for action in window.findChildren(QAction):
        if action.isEnabled() and wanted in action.shortcuts():
            return action
    raise AssertionError(f"nothing in {window!r} is bound to {keys}")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestEveryPanelKeyIsReal(_WindowTestCase):
    """One ``test_key_<id>`` per PANEL_KEYS row: it presses the row's keys
    on the widget the row names and checks they do what the row says."""

    def test_every_row_is_checked_here(self):
        rows = [row[0] for row in PANEL_KEYS]
        self.assertEqual(len(rows), len(set(rows)), "duplicate ids")
        checks = {
            name[len("test_key_") :]
            for name in dir(type(self))
            if name.startswith("test_key_")
        }
        self.assertEqual(set(rows), checks)

    def keys_of(self, row_id):
        return next(keys for rid, _s, keys, _d in PANEL_KEYS if rid == row_id)

    # -- the Frequency field ---------------------------------------------

    def field(self):
        spin = self.win._frequency_field()  # the spin box in FrequencyInput
        spin.setFocus()
        self.settle()
        self.assertTrue(spin.hasFocus())
        return spin

    def test_key_freq_step(self):
        self.assertEqual(self.keys_of("freq_step"), ("Up", "Down"))
        spin = self.field()
        start = self.win._current_frequency()
        step = spin._step_hz
        QTest.keyClick(spin, Qt.Key.Key_Up)
        self.assertAlmostEqual(self.win._current_frequency(), start + step)
        QTest.keyClick(spin, Qt.Key.Key_Down)
        self.assertAlmostEqual(self.win._current_frequency(), start)

    def test_key_freq_page(self):
        self.assertEqual(self.keys_of("freq_page"), ("PgUp", "PgDown"))
        spin = self.field()
        start = self.win._current_frequency()
        step = spin._step_hz
        QTest.keyClick(spin, Qt.Key.Key_PageUp)
        self.assertAlmostEqual(self.win._current_frequency(), start + 10 * step)
        QTest.keyClick(spin, Qt.Key.Key_PageDown)
        self.assertAlmostEqual(self.win._current_frequency(), start)

    def test_key_freq_enter(self):
        spin = self.field()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "100.7")
        self.assertEqual(self.win._current_frequency(), 100.1e6)  # not yet
        QTest.keyClick(spin, Qt.Key.Key_Return)
        self.assertAlmostEqual(self.win._current_frequency(), 100.7e6)

    def test_key_freq_esc(self):
        spin = self.field()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "123")
        QTest.keyClick(spin, Qt.Key.Key_Escape)
        self.settle()
        self.assertEqual(self.win._current_frequency(), 100.1e6)
        self.assertEqual(spin.text(), spin.textFromValue(spin.value()))
        self.assertTrue(self.win._plots_focused())

    # -- the Bookmarks list ----------------------------------------------

    def bookmarks(self, count=1):
        panel = self.win._bookmarks_panel
        for i in range(count):
            panel.add_bookmark(f"Test {i}", 146.52e6 + i * 25e3, "FM", "5 kHz", "")
        last = panel.bookmark_count() - 1
        panel._refresh_list(select=last)
        panel._list.setFocus()
        self.settle()
        self.assertEqual(panel._selected_index(), last)
        return panel, last

    def test_key_bm_enter(self):
        panel, last = self.bookmarks()
        self.assertEqual(self.keys_of("bm_enter"), ("Return",))
        QTest.keyClick(panel._list, Qt.Key.Key_Return)
        self.assertEqual(self.win._current_frequency(), panel.bookmark(last)["freq_hz"])

    def test_key_bm_f2(self):
        panel, last = self.bookmarks()
        with mock.patch(
            "sdr_module.gui.bookmarks_panel.QInputDialog.getText",
            return_value=("Renamed", True),
        ) as ask:
            QTest.keyClick(panel._list, Qt.Key.Key_F2)
        ask.assert_called_once()
        self.assertEqual(panel.bookmark(last)["label"], "Renamed")

    def test_key_bm_delete(self):
        panel, _last = self.bookmarks(count=2)
        self.assertEqual(self.keys_of("bm_delete"), ("Del", "Backspace"))
        count = panel.bookmark_count()
        with mock.patch.object(panel, "_confirm_remove", return_value=True):
            QTest.keyClick(panel._list, Qt.Key.Key_Delete)
            self.assertEqual(panel.bookmark_count(), count - 1)
            QTest.keyClick(panel._list, Qt.Key.Key_Backspace)
            self.assertEqual(panel.bookmark_count(), count - 2)

    def test_key_bm_filter(self):
        panel, _last = self.bookmarks()
        panel._filter.setText("Test")
        panel._filter.setFocus()
        self.settle()
        QTest.keyClick(panel._filter, Qt.Key.Key_Escape)
        self.assertEqual(panel._filter.text(), "")

    # -- the Decoder messages --------------------------------------------

    def decoder_rows(self):
        panel = self.win._decoder_panel
        panel.add_message("POCSAG", "1234567", "first page")
        panel.add_message("POCSAG", "1234567", "second page")
        table = panel._table
        table.selectRow(1)
        table.setFocus()
        self.settle()
        return panel, table

    def test_key_dec_copy(self):
        _panel, table = self.decoder_rows()
        QGuiApplication.clipboard().setText("")
        QTest.keyClick(table, Qt.Key.Key_C, CTRL)
        self.assertIn("second page", QGuiApplication.clipboard().text())

    def test_key_dec_esc(self):
        _panel, table = self.decoder_rows()
        self.assertTrue(table.selectionModel().hasSelection())
        QTest.keyClick(table, Qt.Key.Key_Escape)
        self.assertFalse(table.selectionModel().hasSelection())

    def test_key_air_enter(self):
        from sdr_module.dsp.live_decoders import describe_message, message_details
        from sdr_module.dsp.protocols import ADSBMessage, ProtocolType

        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("ADS-B")
        self.settle()
        msg = ADSBMessage(
            protocol=ProtocolType.ADSB,
            timestamp=0.0,
            raw_bits=b"",
            valid=True,
            icao_address="4840D6",
            downlink_format=17,
            type_code=4,
            callsign="KLM1023",
        )
        address, content = describe_message(msg)
        panel.add_message(
            "ADS-B", address, content, message=msg, details=message_details(msg)
        )
        view = panel._aircraft_view
        panel._tabs.setCurrentWidget(view)
        view.select("4840D6")
        view.table.setFocus()
        self.settle()
        QTest.keyClick(view.table, Qt.Key.Key_Return)
        self.settle()
        self.assertIs(panel._tabs.currentWidget(), panel._messages_page)
        self.assertEqual(panel.address_filter(), "4840D6")

    # -- the Scanner results ---------------------------------------------

    def test_key_scan_enter(self):
        from sdr_module.gui.scan_engine import Found

        panel = self.win._scanner_panel
        self.win._show_panel("scanner")
        for freq in (146.52e6, 146.94e6):
            panel._on_found(Found(freq, -40.0, -95.0, 8e3, ("FM", "5 kHz", "25 kHz")))
        table = panel._table
        table.setFocus()
        table.selectRow(1)
        self.settle()
        QTest.keyClick(table, Qt.Key.Key_Return)
        self.assertEqual(self.win._current_frequency(), 146.94e6)

    # -- the waterfall's COLORS scale ------------------------------------

    def colors_scale(self):
        waterfall = self.win._waterfall
        waterfall._user_set_levels(-100.0, -40.0)  # manual levels
        self.assertFalse(waterfall.auto_levels())
        bar = waterfall._level_bar
        self.assertEqual(waterfall._levels_caption.text(), "COLORS")
        bar.setFocus()
        self.settle()
        return waterfall, bar

    def test_key_wf_auto(self):
        waterfall, bar = self.colors_scale()
        QTest.keyClick(bar, Qt.Key.Key_Home)
        self.assertTrue(waterfall.auto_levels())

    def test_key_wf_floor(self):
        waterfall, bar = self.colors_scale()
        QTest.keyClick(bar, Qt.Key.Key_Left)
        self.assertEqual(waterfall.levels(), (-101.0, -40.0))
        QTest.keyClick(bar, Qt.Key.Key_Right, SHIFT)
        self.assertEqual(waterfall.levels(), (-96.0, -40.0))

    def test_key_wf_ceiling(self):
        waterfall, bar = self.colors_scale()
        self.assertEqual(self.keys_of("wf_ceiling"), ("Down", "Up"))
        QTest.keyClick(bar, Qt.Key.Key_Down)
        self.assertEqual(waterfall.levels(), (-100.0, -41.0))
        QTest.keyClick(bar, Qt.Key.Key_Up, SHIFT)
        self.assertEqual(waterfall.levels(), (-100.0, -36.0))

    # -- playing a recording ---------------------------------------------

    def play(self, seconds=40.0, rate=24e3):
        """Play a ``seconds`` long (silent) recording."""
        from sdr_module.dsp.recording import (
            FileFormat,
            SampleFormat,
            save_iq_file,
        )

        folder = tempfile.mkdtemp(prefix="sdr-help-keys-")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "quiet.sigmf-data")
        samples = np.zeros(int(seconds * rate), dtype=np.complex64)
        save_iq_file(path, samples, rate, 145e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
            return_value=(path, ""),
        ):
            self.win._open_recording_action.trigger()
        self.settle()
        self.assertTrue(self.win._playback.active)
        return self.win._playback_bar

    def test_key_pb_space(self):
        self.play(seconds=1.0)
        win = self.win
        win._spectrum.setFocus()
        self.settle()
        QTest.keyClick(win._spectrum, Qt.Key.Key_Space)
        self.assertFalse(win._is_running)
        self.assertEqual(win._state_badge.text(), "PAUSED")
        QTest.keyClick(win._spectrum, Qt.Key.Key_Space)
        self.assertTrue(win._is_running)

    def slider(self):
        bar = self.play()
        slider = bar._slider
        slider.setFocus()
        self.settle()
        self.assertTrue(slider.hasFocus())
        return slider

    def test_key_pb_skip(self):
        self.assertEqual(self.keys_of("pb_skip"), ("Left", "Right"))
        slider = self.slider()
        start = self.win._device.position_s
        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertAlmostEqual(self.win._device.position_s, start + 5.0, places=2)
        QTest.keyClick(slider, Qt.Key.Key_Left)
        self.assertAlmostEqual(self.win._device.position_s, start, places=2)
        self.assertIn("5 s", next(d for i, _s, _k, d in PANEL_KEYS if i == "pb_skip"))

    def test_key_pb_page(self):
        # The row reads "Back / ahead": PgDown goes back, PgUp ahead.
        self.assertEqual(self.keys_of("pb_page"), ("PgDown", "PgUp"))
        slider = self.slider()
        start = self.win._device.position_s
        QTest.keyClick(slider, Qt.Key.Key_PageUp)
        self.assertAlmostEqual(self.win._device.position_s, start + 30.0, places=2)
        QTest.keyClick(slider, Qt.Key.Key_PageDown)
        self.assertAlmostEqual(self.win._device.position_s, start, places=2)

    # -- the second receiver window --------------------------------------

    def second(self):
        host = self.win._second_rx
        rx = host.show()
        rx._status_timer.stop()
        self.addCleanup(rx.shutdown)
        rx.start()
        rx._spectrum.setFocus()
        self.settle()
        return rx

    def test_key_rx2_space(self):
        rx = self.second()
        self.assertTrue(rx.running)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Space)
        self.assertFalse(rx.running)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Space)
        self.assertTrue(rx.running)

    def test_key_rx2_tune(self):
        rx = self.second()
        start = rx.tuned_frequency()
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Right)
        self.assertAlmostEqual(rx.tuned_frequency(), start + rx._step_hz)
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Left, SHIFT)
        self.assertAlmostEqual(rx.tuned_frequency(), start - 9 * rx._step_hz)
        before = rx.tuned_frequency()
        QTest.keyClick(rx._spectrum, Qt.Key.Key_Right, CTRL)
        self.assertAlmostEqual(rx.tuned_frequency(), before + 1e6)

    def test_key_rx2_step(self):
        rx = self.second()
        step = rx._step_hz
        QTest.keyClick(rx._spectrum, "]")
        self.assertGreater(rx._step_hz, step)
        QTest.keyClick(rx._spectrum, "[")
        self.assertEqual(rx._step_hz, step)

    def test_key_rx2_freq(self):
        rx = self.second()
        _action(rx, "Ctrl+L").trigger()
        self.settle()
        entry = getattr(rx._freq_input, "_freq_input", rx._freq_input)
        self.assertIs(QApplication.focusWidget(), entry)

    def test_key_rx2_bookmark(self):
        rx = self.second()
        panel = self.win._bookmarks_panel
        count = panel.bookmark_count()
        rx.tune(146.52e6, recenter=True)
        _action(rx, "Ctrl+B").trigger()
        self.assertEqual(panel.bookmark_count(), count + 1)
        self.assertEqual(panel.bookmark(count)["freq_hz"], 146.52e6)

    def test_key_rx2_record(self):
        folder = tempfile.mkdtemp(prefix="sdr-help-keys-rx2-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        rx = self.second()
        action = _action(rx, "Ctrl+Shift+R")
        action.trigger()
        self.assertIsNotNone(rx._recorder)
        action.trigger()
        self.assertIsNone(rx._recorder)

    def test_key_rx2_close(self):
        rx = self.second()
        self.assertTrue(rx.isVisible())
        _action(rx, "Ctrl+W").trigger()
        self.settle()
        self.assertFalse(rx.isVisible())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestPlotKeys(_WindowTestCase):
    """PLOT_KEYS: P, M and Esc with a plot focused."""

    def test_rows(self):
        self.assertEqual(
            [keys for _c, keys, _d in PLOT_KEYS], [("P",), ("M",), ("Esc",)]
        )

    def test_p_pauses_the_waterfall_from_either_plot(self):
        win = self.win
        for plot in (win._spectrum, win._waterfall):
            plot.setFocus()
            self.settle()
            paused = win._waterfall.is_paused()
            QTest.keyClick(plot, Qt.Key.Key_P)
            self.assertEqual(win._waterfall.is_paused(), not paused, plot)

    def test_m_mutes_with_the_plots_focused(self):
        win = self.win
        # (An output device that opens: without one, unmuting can't work.)
        with (
            mock.patch.object(type(win), "_audio_output_problem", return_value=None),
            mock.patch.object(win._audio, "start", return_value=True),
        ):
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._set_audio_enabled(True)
            win._spectrum.setFocus()
            self.settle()
            self.assertFalse(win._mute_action.isChecked())
            QTest.keyClick(win._spectrum, Qt.Key.Key_M)
            self.assertTrue(win._mute_action.isChecked())
            self.assertEqual(win._audio_state(), "muted")
            QTest.keyClick(win._spectrum, Qt.Key.Key_M)
            self.assertFalse(win._mute_action.isChecked())
            self.assertEqual(win._audio_state(), "on")

    def test_esc_clears_the_waterfall_measurement(self):
        waterfall = self.win._waterfall
        cleared = []
        waterfall.setFocus()
        self.settle()
        waterfall._measure = object()  # a measurement is shown
        with mock.patch.object(
            waterfall, "clear_measurement", lambda: cleared.append(1)
        ):
            QTest.keyClick(waterfall, Qt.Key.Key_Escape)
        self.assertEqual(cleared, [1])
        waterfall._measure = None


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheDialogListsTheBindings(_WindowTestCase):
    def dialog(self):
        dialog = HelpDialog(self.win)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_every_menu_shortcut_is_listed(self):
        from sdr_module.gui.help_dialog import _portable

        listed = {
            _portable(k)
            for _c, keys, _d in self.dialog().shortcut_entries()
            for k in keys
        }
        missing = []
        for action in self.win.findChildren(QAction):
            if not action.isVisible():
                continue
            for seq in action.shortcuts():
                key = seq.toString(QKeySequence.SequenceFormat.PortableText)
                if key and key not in listed:
                    missing.append((key, action.text()))
        self.assertEqual(missing, [])

    def test_panel_keys_are_shown_in_their_sections(self):
        dialog = self.dialog()
        captions = {
            label.text()
            for label in dialog.findChildren(type(dialog._no_match))
            if label.property("role") == "caption"
        }
        for _id, section, _keys, _desc in PANEL_KEYS:
            self.assertIn(section.upper(), captions)

    def visible(self, dialog, text):
        dialog._filter.setText(text)
        return [search for row, search, _sec in dialog._rows if not row.isHidden()]

    def test_a_key_filter_matches_whole_keys(self):
        dialog = self.dialog()
        found = self.visible(dialog, "Ctrl+S")
        self.assertEqual(len(found), 1, found)
        self.assertIn("ctrl+s", found[0])
        self.assertNotIn("ctrl+shift+r", " ".join(found))
        f2 = self.visible(dialog, "f2")
        self.assertEqual(len(f2), 1, f2)
        self.assertIn("rename", f2[0])
        deleting = self.visible(dialog, "delete")
        self.assertTrue(any("remove the selected bookmark" in s for s in deleting))

    def test_words_still_find_descriptions(self):
        dialog = self.dialog()
        self.assertTrue(
            any(
                "rename the selected bookmark" in s
                for s in self.visible(dialog, "rename")
            )
        )
        self.assertTrue(any("mute" in s for s in self.visible(dialog, "mute")))

    def test_typing_a_word_never_flashes_no_match(self):
        # T12 review: the first letter of a word ("t" of "tune") read as a
        # key nothing is bound to, and the list went empty ("No shortcuts
        # match") until the second letter.
        dialog = self.dialog()
        for text in ("t", "tu", "tune", "c", "ctrl", "ctrl+"):
            found = self.visible(dialog, text)
            self.assertTrue(found, text)
            self.assertTrue(dialog._no_match.isHidden(), text)

    def test_a_single_key_finds_its_combinations(self):
        dialog = self.dialog()
        muting = self.visible(dialog, "m")
        self.assertTrue(any(s.startswith("plots") for s in muting), muting)
        self.assertTrue(any("ctrl+m" in s for s in muting), muting)
        # Every row shown has the key M: alone or last (Ctrl+M).
        self.assertTrue(all(" m" in s or "+m" in s for s in muting), muting)
        self.assertTrue(any("ctrl+1" in s for s in self.visible(dialog, "1")))
        ctrl = self.visible(dialog, "Ctrl+")
        self.assertTrue(all("ctrl+" in s for s in ctrl), ctrl)
        self.assertTrue(any("ctrl+s" in s for s in ctrl), ctrl)


if __name__ == "__main__":
    unittest.main()
