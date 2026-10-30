"""Tools > Transmit Text (gui/transmit_text.py): the window, its counter,
reported characters and macros; Preview through the audio output; Save as
WAV, SigMF and cf32; and Transmit through the window's transmit engine on a
HackRF One (the threaded fake python_hackrf backend, tests/hackrf_fakes.py):
what goes on the air is read back and decoded, with its station IDs. The
demo device refuses, with its reason.
"""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.gui.transmitter import format_mhz  # noqa: E402
from tests import text_demod as demod  # noqa: E402
from tests.hackrf_fakes import analyse_cw, analyse_mcw  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtGui import QAction, QKeySequence
    from PyQt6.QtWidgets import QApplication, QFileDialog

RATE = 2e6


def _varicode_table():
    from sdr_module.protocols.text_modes import VARICODE

    return {code: char for char, code in VARICODE.items()}


class _TextCase(_TxCase):
    """A window with the Transmit Text window open (General class, so HF
    data and CW are allowed)."""

    license = "general"

    def setUp(self):
        super().setUp()
        self.win._show_transmit_text()
        self.dlg = self.win._transmit_text
        self.addCleanup(self.dlg.close)
        self.dlg.set_text("")

    def line(self):
        return self.dlg._tx_label.text(), self.dlg._tx_label.property("tone") or None

    def refresh(self):
        self.dlg.refresh()
        QApplication.processEvents()


class TestWindow(_TextCase):
    def test_menu_action_and_shortcut(self):
        actions = [
            a for a in self.win.findChildren(QAction) if a.text() == "Transmit Te&xt..."
        ]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].shortcut(), QKeySequence("Ctrl+Shift+T"))
        self.dlg.close()
        actions[0].trigger()
        self.assertIs(self.win._transmit_text, self.dlg)  # the same window
        self.assertTrue(self.dlg.isVisible())
        self.assertFalse(self.dlg.isModal())

    def test_counter_and_air_time(self):
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("CQ CQ CQ DE W1AW W1AW W1AW K")
        self.assertEqual(
            self.dlg._count_label.text(), "28 characters · 6.4 s at 45.45 Bd"
        )
        self.dlg.set_mode("CW")
        self.assertEqual(
            self.dlg._count_label.text(), "28 characters · 18.7 s at 20 WPM"
        )
        self.dlg._wpm_spin.setValue(25)
        self.assertIn("at 25 WPM", self.dlg._count_label.text())
        self.dlg.set_text("")
        self.assertEqual(self.dlg._count_label.text(), "Nothing to send yet")
        self.assertFalse(self.dlg._save_button.isEnabled())
        self.assertFalse(self.dlg._preview_button.isEnabled())

    def test_characters_a_mode_lacks_are_reported_and_removable(self):
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("50% @ home")
        self.assertTrue(self.dlg._unsupported_row.isVisibleTo(self.dlg))
        self.assertEqual(
            self.dlg._unsupported_label.text(),
            "RTTY can't send %, @: they are left out.",
        )
        self.assertEqual(
            self.dlg._count_label.text(),
            "10 characters · 2.5 s at 45.45 Bd · sent in capitals",
        )
        self.dlg._remove_button.click()
        self.assertEqual(self.dlg.text(), "50  home")
        self.assertFalse(self.dlg._unsupported_row.isVisibleTo(self.dlg))
        self.dlg._text_edit.undo()  # (one undoable edit)
        self.assertEqual(self.dlg.text(), "50% @ home")
        # PSK31 has every ASCII character.
        self.dlg.set_mode("PSK31")
        self.assertFalse(self.dlg._unsupported_row.isVisibleTo(self.dlg))

    def test_remove_keeps_good_prosigns(self):
        self.dlg.set_mode("CW")
        self.dlg.set_text("73 <SK> 5% <#>")
        self.dlg._remove_button.click()
        self.assertEqual(self.dlg.text(), "73 <SK> 5 ")
        self.assertFalse(self.dlg._unsupported_row.isVisibleTo(self.dlg))

    def test_macros_use_the_ham_id_callsign(self):
        self.assertFalse(self.dlg._macro_buttons[0].isEnabled())
        self.assertTrue(self.dlg._ham_id_button.isVisibleTo(self.dlg))
        self.assertFalse(self.dlg.insert_macro("CQ DE {call}"))
        self.panel.set_callsign("W1AW")  # (the window follows Ham ID)
        self.assertTrue(self.dlg._macro_buttons[0].isEnabled())
        self.assertFalse(self.dlg._ham_id_button.isVisibleTo(self.dlg))
        self.dlg._macro_buttons[0].click()  # CQ
        self.assertEqual(self.dlg.text(), "CQ CQ CQ DE W1AW W1AW W1AW K")
        self.dlg._macro_buttons[2].click()  # 73, after a space
        self.assertEqual(
            self.dlg.text(), "CQ CQ CQ DE W1AW W1AW W1AW K TU 73 DE W1AW SK"
        )

    def test_ham_id_button_opens_the_panel(self):
        self.win._show_panel("decoder")
        self.dlg._ham_id_button.click()
        self.assertIs(
            self.win._ham_tabs.currentWidget(), self.win._panel_pages["ham_id"]
        )

    def test_settings_and_text_are_remembered(self):
        self.dlg.set_mode("CW")
        self.dlg._wpm_spin.setValue(28)
        self.dlg.set_mode("ASCII")
        self.dlg._baud_combo.setCurrentIndex(self.dlg._baud_combo.findData(110.0))
        self.dlg.set_text("hello")
        self.dlg.close()
        saved = json.loads(self.store.values["tx/text"])
        self.assertEqual(saved["mode"], "ASCII")
        self.assertEqual(saved["text"], "hello")
        self.assertEqual(saved["modes"]["CW"]["wpm"], 28)
        from sdr_module.gui.transmit_text import TransmitTextDialog

        again = TransmitTextDialog(self.win)
        self.addCleanup(again.deleteLater)
        self.assertEqual((again.mode(), again.text()), ("ASCII", "hello"))
        self.assertEqual(again.settings().baud, 110.0)
        again.set_mode("CW")
        self.assertEqual(again._wpm_spin.value(), 28)

    def test_unreadable_settings_are_ignored(self):
        from sdr_module.gui.transmit_text import TransmitTextDialog

        self.store.values["tx/text"] = "{not json"
        fresh = TransmitTextDialog(self.win)
        self.addCleanup(fresh.deleteLater)
        self.assertEqual((fresh.mode(), fresh.text()), ("RTTY", ""))
        self.store.values["tx/text"] = '{"mode": "CW", "modes": {"CW": {"wpm": "x"}}}'
        fresh = TransmitTextDialog(self.win)
        self.addCleanup(fresh.deleteLater)
        self.assertEqual((fresh.mode(), fresh.settings().wpm), ("CW", 20.0))
        # NaN or Infinity (JSON accepts them) made the window fail to open.
        self.store.values["tx/text"] = (
            '{"mode": "CW", "modes": {"CW": {"wpm": NaN, "audio_hz": Infinity}}}'
        )
        fresh = TransmitTextDialog(self.win)
        self.addCleanup(fresh.deleteLater)
        self.assertEqual(
            (fresh.mode(), fresh.settings().wpm, fresh.settings().audio_hz),
            ("CW", 20.0, 700.0),
        )

    def test_text_is_limited(self):
        from sdr_module.protocols.text_modes import MAX_TEXT_CHARS

        self.dlg.set_text("E" * (MAX_TEXT_CHARS + 50))
        self.assertEqual(len(self.dlg.text()), MAX_TEXT_CHARS)
        self.assertIn("up to 2,000 characters", self.status()[0])
        self.assertIn("2,000 of 2,000 characters", self.dlg._count_label.text())

    def test_a_paste_past_the_limit_keeps_the_text_already_there(self):
        """What doesn't fit is cut from the paste: the end of the text
        already there was cut off before, and the undo history cleared, so
        it couldn't be brought back."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        from sdr_module.protocols.text_modes import MAX_TEXT_CHARS

        edit = self.dlg._text_edit
        original = "".join(chr(65 + i % 26) for i in range(MAX_TEXT_CHARS - 10))
        cursor = edit.textCursor()
        cursor.insertText(original)
        cursor.setPosition(10)
        edit.setTextCursor(cursor)
        QApplication.clipboard().setText("x" * 50)
        edit.paste()
        self.assertEqual(self.dlg.text(), original[:10] + "x" * 10 + original[10:])
        self.assertIn("up to 2,000 characters", self.status()[0])
        edit.undo()  # the text from before the paste
        self.assertEqual(self.dlg.text(), original)
        edit.redo()
        # Full: typing adds nothing, a macro that doesn't fit is refused,
        # and deleting still works.
        full = self.dlg.text()
        cursor = edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        edit.setTextCursor(cursor)
        QTest.keyClicks(edit, "zz")
        QTest.keyClick(edit, Qt.Key.Key_Return)
        self.assertEqual(self.dlg.text(), full)
        self.panel.set_callsign("W1AW")
        self.assertFalse(self.dlg.insert_macro("CQ DE {call}"))
        self.assertEqual(self.dlg.text(), full)
        QTest.keyClick(edit, Qt.Key.Key_Backspace)
        self.assertEqual(self.dlg.text(), full[:-1])

    def test_enter_in_a_field_presses_no_button(self):
        """The window made its first button (Remove Them) the default, so
        Enter in the speed or tone box deleted characters from the text."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QPushButton

        self.dlg.set_mode("CW")
        self.dlg.set_text("CQ 50% DE TEST")
        self.assertTrue(self.dlg._unsupported_row.isVisibleTo(self.dlg))
        buttons = self.dlg.findChildren(QPushButton)
        self.assertEqual([b.text() for b in buttons if b.isDefault()], [])
        for field in (self.dlg._wpm_spin, self.dlg._tone_spin, self.dlg._mode_combo):
            field.setFocus()
            QTest.keyClick(field, Qt.Key.Key_Return)
            self.assertEqual(self.dlg.text(), "CQ 50% DE TEST")

    def test_the_fsk_center_keeps_both_tones_in_the_audio(self):
        """A 100 Hz center with a 1000 Hz shift showed "space -400 Hz" (and
        played it as 400 Hz)."""
        self.dlg.set_mode("ASCII")
        self.dlg._shift_combo.setCurrentIndex(self.dlg._shift_combo.findData(1000.0))
        self.dlg._tone_spin.setValue(100)
        self.assertEqual(self.dlg._tone_spin.minimum(), 600)
        self.assertEqual(self.dlg._tone_spin.value(), 600)
        self.assertEqual(self.dlg.settings().tones(), (100.0, 1100.0))
        self.assertEqual(self.dlg._tones_hint.text(), "Mark 1100 Hz, space 100 Hz")
        # CW has the whole tone range.
        self.dlg.set_mode("CW")
        self.assertEqual(
            (self.dlg._tone_spin.minimum(), self.dlg._tone_spin.maximum()),
            (100, 3500),
        )

    def test_nothing_sendable_says_so(self):
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("@@@%")
        self.refresh()
        self.assertEqual(
            self.line(), ("Nothing to send: RTTY can't send @ (3x), %.", "warning")
        )
        self.assertFalse(self.dlg._transmit_button.isEnabled())

    def test_the_speed_tooltip_states_todays_rule(self):
        self.dlg.set_mode("RTTY")
        tip = self.dlg._baud_combo.toolTip()
        self.assertNotIn("300 Bd", tip)  # (the old US HF limit)
        self.assertIn("2.8 kHz", tip)

    def test_ham_id_tab_opens_it(self):
        self.dlg.close()
        self.panel._text_tx_btn.click()
        self.assertTrue(self.dlg.isVisible())
        self.assertIn("Ctrl+Shift+T", self.panel._text_tx_btn.toolTip())

    def test_capitals_are_mentioned_where_only_capitals_go(self):
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("cq de w1aw")
        self.assertTrue(self.dlg._count_label.text().endswith("· sent in capitals"))
        self.dlg.set_mode("PSK31")
        self.assertNotIn("capitals", self.dlg._count_label.text())
        self.dlg.set_mode("CW")
        self.assertIn("capitals", self.dlg._count_label.text())
        self.dlg.set_text("CQ DE W1AW")
        self.assertNotIn("capitals", self.dlg._count_label.text())

    def test_access_keys_are_unique(self):
        import re

        from PyQt6.QtWidgets import QAbstractButton, QGroupBox, QLabel

        keys = []
        for widget in self.dlg.findChildren((QLabel, QAbstractButton, QGroupBox)):
            text = widget.title() if isinstance(widget, QGroupBox) else widget.text()
            match = re.search(r"&([^&])", text or "")
            if match and widget.isVisibleTo(self.dlg):
                keys.append(match.group(1).lower())
        self.assertEqual(len(keys), len(set(keys)), keys)

    def test_every_control_says_what_it_does(self):
        from PyQt6.QtWidgets import (
            QAbstractButton,
            QAbstractSpinBox,
            QComboBox,
            QPlainTextEdit,
            QProgressBar,
        )

        kinds = (QAbstractButton, QAbstractSpinBox, QComboBox, QPlainTextEdit)
        for widget in self.dlg.findChildren(kinds + (QProgressBar,)):
            if widget.parent() is not None and widget.parent().inherits("QComboBox"):
                continue
            self.assertTrue(
                widget.toolTip() or widget.accessibleName(),
                widget.objectName()
                or type(widget).__name__ + " " + str(getattr(widget, "text", str)()),
            )

    def test_speed_access_key_goes_to_the_control_shown(self):
        self.dlg.set_mode("CW")
        self.assertIs(self.dlg._speed_label.buddy(), self.dlg._wpm_spin)
        self.dlg.set_mode("ASCII")
        self.assertIs(self.dlg._speed_label.buddy(), self.dlg._baud_combo)

    def test_mode_controls(self):
        self.dlg.set_mode("RTTY")
        self.assertTrue(self.dlg._shift_combo.isVisibleTo(self.dlg))
        self.assertEqual(self.dlg._tones_hint.text(), "Mark 2295 Hz, space 2125 Hz")
        self.assertEqual(self.dlg._tone_label.text(), "&Center:")
        self.dlg.set_mode("PSK31")
        self.assertFalse(self.dlg._shift_combo.isVisibleTo(self.dlg))
        self.assertTrue(self.dlg._fixed_rate.isVisibleTo(self.dlg))
        self.dlg.set_mode("CW")
        self.assertTrue(self.dlg._wpm_spin.isVisibleTo(self.dlg))
        self.assertEqual(
            [self.dlg._emission_combo.itemData(i) for i in range(3)],
            ["auto", "direct", "fm"],
        )


class TestDemoDevice(_TextCase):
    def test_refuses_clearly_and_still_previews_and_saves(self):
        self.panel.set_callsign("W1AW")
        self.win._start_demo_mode()
        self.addCleanup(self.win._release_device)
        self.dlg.set_text("CQ DE W1AW")
        self.refresh()
        self.assertEqual(
            self.line(),
            (
                "Can't send: the demo device can't transmit. Connect a HackRF "
                "One. (Preview and Save need no device.)",
                "warning",
            ),
        )
        self.assertFalse(self.dlg._transmit_button.isEnabled())
        self.assertIn("demo device", self.dlg._transmit_button.toolTip())
        self.assertTrue(self.dlg._preview_button.isEnabled())
        self.assertTrue(self.dlg._save_button.isEnabled())
        # Even if pressed anyway, nothing is sent.
        self.dlg._on_transmit_clicked()
        self.assertFalse(self.station.busy)
        self.assertEqual(self.boxes, [])


class TestTransmit(_TextCase):
    def _send(self, text, mode, freq=14.08e6, **fake):
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=freq, **fake)
        self.dlg.set_mode(mode)
        self.dlg.set_text(text)
        self.refresh()
        line, tone = self.line()
        self.assertTrue(line.startswith(f"Will send on {format_mhz(freq)}"), line)
        self.assertTrue(self.dlg._transmit_button.isEnabled())
        self.dlg._transmit_button.click()
        return dev, backend

    def _job_samples(self, backend, signal_seconds):
        """The job's own part of what went on air (between the IDs)."""
        id_job = self.station.make_id_job(self.win._current_frequency())
        start = id_job.total_samples
        return backend.tx_iq()[start : start + int(round(signal_seconds * RATE))]

    def test_rtty_goes_out_with_its_ids_and_decodes(self):
        dev, backend = self._send("CQ CQ DE W1AW 599 73", "RTTY")
        self.assertTrue(self.station.busy)
        self.assertTrue(self.station._badge_action.isVisible())  # ON AIR
        self.assertEqual(self.dlg._transmit_button.text(), "Stop")
        self.assertTrue(self.dlg._progress.isVisibleTo(self.dlg))
        self.assertTrue(self.dlg._progress.format().startswith("On the air"))
        signal = self.dlg.current_signal()
        self.wait_done(60)
        self.assertEqual(backend.start_rate(), RATE)  # a low TX rate
        result = self.station.engine.last_result
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.ids_sent, 2)
        iq = self._job_samples(backend, signal.duration_s)
        small = iq[: len(iq) // 50 * 50].reshape(-1, 50).mean(axis=1)
        self.assertEqual(demod.rtty_decode(small, RATE / 50), "CQ CQ DE W1AW 599 73")
        # The IDs: keyed carrier (CW) on HF, before and after.
        whole = backend.tx_iq(50)
        n_id = self.station.make_id_job(14.08e6).total_samples // 50
        self.assertEqual(analyse_cw(whole[:n_id], RATE / 50)["text"], "DE W1AW")
        self.assertEqual(analyse_cw(whole[-n_id:], RATE / 50)["text"], "DE W1AW")
        # Back to idle, with the outcome.
        self.assertEqual(self.dlg._transmit_button.text(), "T&ransmit")
        self.assertFalse(self.station._badge_action.isVisible())
        text, tone = self.line()
        self.assertTrue(
            text.startswith("✓ Sent RTTY “CQ CQ DE W1AW 599 73” on 14.080 MHz ("),
            text,
        )
        self.assertIn("with 2 IDs", text)
        self.assertEqual(tone, "success")
        self.assertEqual(self.status()[1], "success")

    def test_the_line_says_what_it_will_do(self):
        self.panel.set_callsign("W1AW")
        self.hackrf(freq=14.08e6)
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("RYRY")
        self.refresh()
        text, tone = self.line()
        self.assertEqual(tone, "info")
        self.assertTrue(
            text.startswith(
                "Will send on 14.080 MHz · RTTY 45.45 Bd, 170 Hz shift · TX gain "
                "20 dB (about "
            ),
            text,
        )
        self.assertTrue(
            text.endswith("· ID DE W1AW at the start, every 10 min and at the end"),
            text,
        )
        self.panel._auto_id_check.setChecked(False)
        self.refresh()
        self.assertTrue(self.line()[0].endswith("· ID DE W1AW at the start and end"))

    def test_psk31_goes_out_and_decodes(self):
        text = "Hello de W1AW: 73!"
        dev, backend = self._send(text, "PSK31", freq=14.0703e6)
        signal = self.dlg.current_signal()
        self.wait_done(60)
        self.assertTrue(self.station.engine.last_result.ok)
        iq = self._job_samples(backend, signal.duration_s)
        small = iq[: len(iq) // 100 * 100].reshape(-1, 100).mean(axis=1)
        self.assertEqual(demod.psk31_decode(small, RATE / 100, _varicode_table()), text)

    def test_cw_text_and_its_id(self):
        dev, backend = self._send("CQ TEST", "CW", freq=14.05e6)
        self.wait_done(60)
        self.assertTrue(self.station.engine.last_result.ok)
        result = analyse_cw(backend.tx_iq(20), RATE / 20)
        self.assertEqual(result["text"], "DE W1AW CQ TEST DE W1AW")

    def test_fm_tone_cw_on_2m(self):
        """CW "Auto" is an FM tone above 30 MHz (like the ID)."""
        self.dlg.set_mode("CW")
        self.assertEqual(self.dlg.emission(146.52e6), "fm")
        self.assertEqual(self.dlg.emission(14.05e6), "direct")
        dev, backend = self._send("TEST", "CW", freq=146.52e6)
        self.wait_done(60)
        result = analyse_mcw(backend.tx_iq(32), RATE / 32)
        self.assertEqual(result["text"], "DE W1AW TEST DE W1AW")

    def test_ascii_fm_audio_on_2m(self):
        self.dlg.set_mode("ASCII")
        self.dlg._emission_combo.setCurrentIndex(
            self.dlg._emission_combo.findData("fm")
        )
        dev, backend = self._send("hi 2m", "ASCII", freq=146.52e6)
        signal = self.dlg.current_signal()
        self.wait_done(60)
        self.assertTrue(self.station.engine.last_result.ok)
        from sdr_module.protocols.text_modes import FM_LEAD_S

        id_len = self.station.make_id_job(146.52e6).total_samples
        start = id_len + int(FM_LEAD_S * RATE)
        iq = backend.tx_iq()[start : start + int(signal.duration_s * RATE)]
        # FM-demodulate, then read the Bell 103 tones.
        small = iq[: len(iq) // 40 * 40].reshape(-1, 40).mean(axis=1)
        audio = demod.instantaneous_frequency(small, RATE / 40)
        self.assertEqual(
            demod.ascii_decode(demod.analytic(audio), RATE / 40, 1170.0, 300.0),
            b"hi 2m",
        )

    def test_stop_from_the_window(self):
        dev, backend = self._send("THE QUICK BROWN FOX " * 3, "RTTY", pace=True)
        self.assertTrue(self.pump(3.0, lambda: backend.tx_count() > RATE))
        self.dlg._transmit_button.click()  # Stop
        self.wait_done(10)
        result = self.station.engine.last_result
        self.assertEqual(result.status, "stopped")
        text, tone = self.line()
        self.assertTrue(text.startswith("Stopped RTTY “THE QUICK BROWN FOX"), text)
        self.assertEqual(tone, "warning")
        # Stopped during the first ID: no ID went out, so it says to send one
        # (as the Ham ID tab does).
        self.assertEqual(result.ids_sent, 0)
        self.assertTrue(
            text.endswith(
                "Send your ID (Radio > Send Station ID) to finish the contact."
            ),
            text,
        )

    def test_closing_the_window_keeps_sending_and_says_so(self):
        dev, backend = self._send("THE QUICK BROWN FOX " * 3, "RTTY", pace=True)
        self.assertTrue(self.station.busy)
        self.dlg.close()
        self.assertTrue(self.station.busy)  # (Stop ends it, not Close)
        text, tone = self.status()
        self.assertTrue(text.startswith("Still transmitting RTTY “THE QUICK"), text)
        self.assertEqual(tone, "danger")
        self.win._show_transmit_text()  # back: Stop and the progress
        self.assertEqual(self.dlg._transmit_button.text(), "Stop")
        self.station.stop()
        self.wait_done(10)

    def test_each_send_is_a_new_job(self):
        dev, backend = self._send("AB", "RTTY")
        self.wait_done(60)
        first = self.station.engine.last_result.job
        self.dlg._transmit_button.click()
        self.wait_done(60)
        second = self.station.engine.last_result.job
        self.assertIsNot(first, second)
        self.assertIsNot(first.source, second.source)
        self.assertTrue(self.station.engine.last_result.ok)

    def test_refused_without_a_callsign(self):
        self.hackrf(freq=14.08e6)
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("TEST")
        self.refresh()
        self.assertEqual(
            self.line(),
            (
                "Can't send on 14.080 MHz: an amateur transmission must be "
                "identified: enter your callsign in Ham > Ham ID first.",
                "danger",
            ),
        )
        self.assertFalse(self.dlg._transmit_button.isEnabled())

    def test_hf_data_width_and_license_are_checked(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_callsign("W1AW")
        self.hackrf(freq=14.08e6)
        self.dlg.set_mode("ASCII")
        self.dlg._baud_combo.setCurrentIndex(self.dlg._baud_combo.findData(1200.0))
        self.dlg.set_text("fast")
        self.refresh()
        self.assertEqual(self.line()[1], "info")  # (no baud limit on HF now)
        self.dlg._emission_combo.setCurrentIndex(
            self.dlg._emission_combo.findData("fm")
        )
        self.refresh()
        text, tone = self.line()
        width = self.dlg.current_signal().occupied_bandwidth_hz("fm")
        self.assertEqual(
            text,
            f"Can't send on 14.080 MHz: the signal is {width / 1e3:.1f} kHz wide, "
            "and RTTY and data below 29.7 MHz may be at most 2.8 kHz wide (choose "
            "a lower baud rate or shift, or send it on the carrier).",
        )
        self.assertEqual(tone, "danger")
        self.dlg._emission_combo.setCurrentIndex(
            self.dlg._emission_combo.findData("direct")
        )
        self.win._control_panel.set_license_class(LicenseClass.NONE)
        self.refresh()
        self.assertEqual(
            self.line()[0],
            "Can't send on 14.080 MHz: your license (None) allows CB/MURS/FRS only.",
        )
        self.assertFalse(self.dlg._transmit_button.isEnabled())

    def test_first_use_question(self):
        self.store.values["tx/confirm"] = True
        seen = []

        def answer(box, transmit=False):
            seen.append(box.text())
            buttons = {b.text().replace("&", ""): b for b in box.buttons()}
            button = buttons["Transmit"] if transmit else box.escapeButton()
            box.clickedButton = lambda b=button: b
            return 0

        self.win._exec_dialog = answer
        self._send("CQ DE W1AW", "RTTY")
        self.assertFalse(self.station.busy)
        self.assertTrue(
            seen[0].startswith(
                "Transmit RTTY “CQ DE W1AW” on 14.080 MHz (45.45 Bd, 170 Hz "
                "shift, about "
            ),
            seen[0],
        )
        self.assertEqual(self.line(), ("RTTY “CQ DE W1AW” not sent: cancelled.", None))

    def test_another_transmission_on_the_air_holds_it(self):
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=14.05e6, pace=True)
        self.dlg.set_mode("CW")
        self.dlg.set_text("TEST")
        self.assertTrue(self.station.send_station_id())
        self.refresh()
        text, tone = self.line()
        self.assertEqual(
            text,
            "Wait: DE W1AW is on the air (Radio > Stop Transmitting ends it).",
        )
        self.assertFalse(self.dlg._transmit_button.isEnabled())
        self.station.stop()
        self.wait_done(10)


class TestPreview(_TextCase):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import audio_sink

        self.written = []
        self.opened = []
        self.patch.setattr(
            audio_sink.AudioSink,
            "start",
            lambda sink, rate=48000: self.opened.append(rate) or True,
        )
        self.patch.setattr(
            audio_sink.AudioSink,
            "write",
            lambda sink, samples: self.written.append(np.array(samples)),
        )
        self.patch.setattr(audio_sink.AudioSink, "stop", lambda sink: None)
        self.now = [100.0]
        self.dlg._preview._clock = lambda: self.now[0]

    def _play_to_end(self, step=0.04):
        preview = self.dlg._preview
        while preview.playing:
            self.now[0] += step
            preview.tick()
        return np.concatenate(self.written)

    def test_preview_plays_what_a_receiver_hears(self):
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("RY TEST 123")
        self.dlg._preview_button.click()
        self.assertEqual(self.opened, [48000])
        self.assertEqual(self.dlg._preview_button.text(), "Stop &Preview")
        self.assertTrue(self.dlg._progress.isVisibleTo(self.dlg))
        audio = self._play_to_end()
        self.assertEqual(len(audio), self.dlg.current_signal().sample_count(48000))
        self.assertLessEqual(np.max(np.abs(audio)), 0.5 + 1e-9)
        self.assertEqual(
            demod.rtty_decode(demod.analytic(audio), 48000.0, 2210.0), "RY TEST 123"
        )
        self.assertEqual(self.dlg._preview_button.text(), "&Preview")
        self.assertFalse(self.dlg._progress.isVisibleTo(self.dlg))

    def test_preview_is_fed_in_real_time(self):
        self.dlg.set_mode("CW")
        self.dlg.set_text("PARIS PARIS")
        self.dlg._preview_button.click()
        # At the start: only the lead (0.15 s) is handed over.
        self.assertEqual(sum(len(a) for a in self.written), int(0.15 * 48000))
        self.now[0] += 1.0
        self.dlg._preview.tick()
        self.assertEqual(sum(len(a) for a in self.written), int(1.15 * 48000))
        self.dlg._preview_button.click()  # Stop Preview
        self.assertFalse(self.dlg._preview.playing)

    def test_editing_stops_the_preview(self):
        self.dlg.set_mode("PSK31")
        self.dlg.set_text("hello")
        self.dlg._preview_button.click()
        self.assertTrue(self.dlg._preview.playing)
        self.dlg.set_text("hello there")
        self.assertFalse(self.dlg._preview.playing)

    def test_preview_uses_the_output_chosen_in_preferences(self):
        from sdr_module.gui import audio_sink

        chosen = []
        # (AudioSink chooses its output once Preferences has one: T11.)
        cls = audio_sink.AudioSink
        before = cls.__dict__.get("set_output_device")
        cls.set_output_device = lambda sink, device, name="": chosen.append(
            (device, name)
        )
        if before is None:
            self.addCleanup(delattr, cls, "set_output_device")
        else:
            self.addCleanup(setattr, cls, "set_output_device", before)
        self.store.values["prefs/audio_output"] = "usb-headset"
        self.store.values["prefs/audio_output_name"] = "USB Headset"
        self.dlg.set_text("TEST")
        self.dlg._preview_button.click()
        self.assertEqual(chosen, [("usb-headset", "USB Headset")])

    def test_no_audio_output_says_so(self):
        from sdr_module.gui import audio_sink

        self.patch.setattr(
            audio_sink.AudioSink, "start", lambda sink, rate=48000: False
        )
        self.dlg.set_text("TEST")
        self.dlg._preview_button.click()
        self.assertFalse(self.dlg._preview.playing)
        self.assertEqual(
            self.line(),
            (
                "No audio output to preview on. Save… writes a WAV file you can "
                "play elsewhere.",
                "warning",
            ),
        )


class TestSave(_TextCase):
    def _save_as(self, name, chosen_filter=""):
        import tempfile

        folder = tempfile.mkdtemp(prefix="t14_save_")
        path = os.path.join(folder, name)
        asked = []

        def fake_dialog(parent, caption, start, filters, selected):
            asked.append((caption, start, filters, selected))
            return path, chosen_filter or selected

        self.patch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
        self.dlg._save_button.click()
        return folder, asked

    def test_wav_for_a_radios_data_input(self):
        import wave

        self.win.set_frequency(14.08e6)
        self.dlg.set_mode("RTTY")
        self.dlg.set_text("CQ DE W1AW")
        folder, asked = self._save_as("cq.wav")
        caption, start, filters, selected = asked[0]
        self.assertEqual(selected, "WAV audio (*.wav)")
        self.assertIn("SigMF I/Q recording (*.sigmf-data)", filters)
        self.assertTrue(os.path.basename(start).startswith("rtty_14.080MHz_"))
        path = os.path.join(folder, "cq.wav")
        with wave.open(path) as w:
            self.assertEqual((w.getnchannels(), w.getframerate()), (1, 48000))
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32767.0
        self.assertEqual(
            demod.rtty_decode(demod.analytic(pcm), 48000.0, 2210.0), "CQ DE W1AW"
        )
        text, tone = self.line()
        self.assertTrue(text.startswith("✓ Saved RTTY audio to "), text)
        self.assertEqual(tone, "success")
        self.assertEqual(self.status()[1], "success")

    def test_sigmf_keeps_the_frequency(self):
        from sdr_module.dsp.recording import load_iq_file

        self.win.set_frequency(14.0703e6)
        self.dlg.set_mode("PSK31")
        self.dlg.set_text("SigMF test")
        folder, _asked = self._save_as("psk", "SigMF I/Q recording (*.sigmf-data)")
        data = os.path.join(folder, "psk.sigmf-data")
        meta = json.load(open(os.path.join(folder, "psk.sigmf-meta")))
        self.assertEqual(meta["captures"][0]["core:frequency"], 14.0703e6)
        self.assertEqual(meta["global"]["core:sample_rate"], 48000)
        self.assertEqual(meta["global"]["core:description"], "PSK31: SigMF test")
        samples, _meta = load_iq_file(data)
        self.assertEqual(
            demod.psk31_decode(samples, 48000.0, _varicode_table()), "SigMF test"
        )
        # The format is remembered for the next Save.
        self.assertEqual(
            json.loads(self.store.values["tx/text"])["save_format"], "sigmf"
        )
        self.assertIn("I/Q at 14.0703 MHz", self.line()[0])

    def test_cf32_with_its_metadata(self):
        """The I/Q is what would go on the air: CW "Auto" is a keyed carrier
        on HF and an FM tone (MCW) on 2 m."""
        from sdr_module.dsp.recording import find_sidecar

        self.dlg.set_mode("CW")
        self.dlg.set_text("VVV")
        for freq, analyse in ((7.03e6, analyse_cw), (144.5e6, analyse_mcw)):
            self.win.set_frequency(freq)
            folder, _asked = self._save_as("cw.cf32")
            path = os.path.join(folder, "cw.cf32")
            iq = np.fromfile(path, np.complex64)
            self.assertEqual(analyse(iq, 48000.0)["text"], "VVV")
            info = find_sidecar(path)
            self.assertEqual((info.sample_rate, info.center_frequency), (48000.0, freq))

    def test_a_failed_save_says_why(self):
        missing = os.path.join(self._missing_folder(), "nowhere", "x.wav")
        self.patch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a: (missing, "WAV audio (*.wav)")),
        )
        self.dlg.set_text("TEST")
        self.dlg._save_button.click()
        text, tone = self.line()
        self.assertTrue(text.startswith("Couldn't save x.wav: "), text)
        self.assertEqual(tone, "danger")
        self.assertEqual(self.status()[1], "danger")
        self.assertFalse(os.path.exists(missing))

    def _missing_folder(self):
        import tempfile

        return tempfile.mkdtemp(prefix="t14_missing_")

    def test_cancelled_dialog_saves_nothing(self):
        self.patch.setattr(
            QFileDialog, "getSaveFileName", staticmethod(lambda *a: ("", ""))
        )
        self.dlg.set_text("TEST")
        before = self.line()
        self.dlg._save_button.click()
        self.assertEqual(self.line(), before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
