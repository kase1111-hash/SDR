#!/usr/bin/env python3
"""
T12: one coherent window. Menus and shortcuts (no key bound twice), the
tool-tab bar (fits, maximize, default tab), the toolbar at 1024 px, the
Audio button's states and keys, the status bar (clickable, recent
messages, warnings logged), the window title and the band plan line.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_coherent_window.py``
"""

import logging
import os
import unittest
from collections import defaultdict
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction, QKeySequence, QShortcut
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QToolButton

    from sdr_module.gui import themes


def _portable(seq) -> str:
    return seq.toString(QKeySequence.SequenceFormat.PortableText)


def _click(widget) -> None:
    """A left click (press and release) on ``widget``."""
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)


def _menu(win, *path):
    """The QMenu at ``path`` (menu titles without "&")."""
    menu = None
    actions = win.menuBar().actions()
    for title in path:
        menu = next(
            a.menu()
            for a in actions
            if a.menu() is not None and a.text().replace("&", "") == title
        )
        actions = menu.actions()
    return menu


def _item(menu, text):
    """The action of ``menu`` whose text (without "&" and the shortcut
    column) is ``text``."""
    for action in menu.actions():
        if action.text().split("\t")[0].replace("&", "") == text:
            return action
    raise AssertionError(f"no {text!r} in {[a.text() for a in menu.actions()]}")


class _ThemedSizes:
    """Run a check in both themes (the app always runs with one: it gives
    some bars tighter or wider spacing than bare Fusion)."""

    def each_theme(self, check):
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in themes.THEMES:
                themes.apply_theme(qapp, theme)
                check(theme)
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


# ---------------------------------------------------------------------------
# Menus and shortcuts
# ---------------------------------------------------------------------------


class TestShortcutsAreBoundOnce(_WindowTestCase):
    def test_no_key_is_bound_to_two_commands(self):
        """Walk every QAction (and QShortcut) of the window: a key bound
        twice is ambiguous, and Qt then fires neither."""
        bound = defaultdict(list)
        for action in self.win.findChildren(QAction):
            if action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut:
                continue  # (local to its widget)
            for seq in action.shortcuts():
                bound[_portable(seq)].append(action)
        for shortcut in self.win.findChildren(QShortcut):
            for seq in shortcut.keys():
                bound[_portable(seq)].append(shortcut)
        twice = {
            key: [getattr(a, "text", lambda: "QShortcut")() for a in owners]
            for key, owners in bound.items()
            if len(owners) > 1
        }
        self.assertEqual(twice, {})

    def test_a_key_shown_after_a_tab_is_free_or_the_same_command(self):
        """Menu items show keys the window handles itself after a tab
        ("Start Receiving\\tSpace"); one that shows a key another action
        binds must do the same as it (Tools > SSTV Decoder shows Ctrl+5,
        View > Panels > SSTV's key)."""
        real = {}
        for action in self.win.findChildren(QAction):
            for seq in action.shortcuts():
                real[_portable(seq)] = action
        shown = 0
        for action in self.win.findChildren(QAction):
            text = action.text()
            if "\t" not in text:
                continue
            key = _portable(QKeySequence(text.split("\t", 1)[1]))
            owner = real.get(key)
            if owner is None or owner is action:
                continue
            shown += 1
            self.assertTrue(action.data(), (text, owner.text()))
            self.assertEqual(action.data(), owner.data(), (text, owner.text()))
        self.assertGreaterEqual(shown, 4)  # the Tools menu's panels

    def test_the_new_keys_are_where_the_contract_puts_them(self):
        keys = {}
        for action in self.win.findChildren(QAction):
            for seq in action.shortcuts():
                keys[_portable(seq)] = action
        self.assertIs(keys["Ctrl+M"], self.win._mute_action)
        self.assertIs(keys["Ctrl+Up"], self.win._volume_up_action)
        self.assertIs(keys["Ctrl+Down"], self.win._volume_down_action)
        for n, key in enumerate(
            ["decoder", "bookmarks", "s_meter", "ham_id", "sstv", "qrp", "info"],
            start=1,
        ):
            self.assertIs(keys[f"Ctrl+{n}"], self.win._panel_actions[key])
        self.assertIs(keys["Ctrl+8"], self.win._panel_actions["scanner"])
        self.assertIs(keys["Ctrl+9"], self.win._panel_actions["signal"])


class TestMenus(_WindowTestCase):
    def test_ham_tools_are_in_the_tools_menu(self):
        tools = _menu(self.win, "Tools")
        for text, key in (
            ("S-Meter", "s_meter"),
            ("SSTV Decoder", "sstv"),
            ("Ham ID (Station ID)", "ham_id"),
            ("QRP Calculator", "qrp"),
            ("Protocol Decoder", "decoder"),
        ):
            action = _item(tools, text)
            self.assertTrue(action.statusTip())
            action.trigger()
            self.settle()
            self.assertEqual(self.win._current_panel_key(), key, text)
        self.assertIn("Ctrl+5", _item(tools, "SSTV Decoder").text())
        _item(tools, "Transmit Text...")
        _item(tools, "SSTV: Tune to ISS (145.800 MHz)")

    def test_send_station_id_is_in_the_radio_menu(self):
        radio = _menu(self.win, "Radio")
        _item(radio, "Send Station ID")
        _item(radio, "Stop Transmitting")

    def test_view_panels_follow_the_tab_bar_with_their_keys(self):
        panels = _menu(self.win, "View", "Panels")
        rows = [
            (a.text().replace("&", ""), _portable(a.shortcut()))
            for a in panels.actions()
            if not a.isSeparator()
        ]
        self.assertEqual(
            rows,
            [
                ("Decoder", "Ctrl+1"),
                ("Bookmarks", "Ctrl+2"),
                ("S-Meter", "Ctrl+3"),
                ("Ham ID", "Ctrl+4"),
                ("SSTV", "Ctrl+5"),
                ("QRP", "Ctrl+6"),
                ("Scanner", "Ctrl+8"),
                ("Signal", "Ctrl+9"),
                ("Info", "Ctrl+7"),
            ],
        )
        # The Ham panels in a group of their own (between separators).
        kinds = ["-" if a.isSeparator() else "x" for a in panels.actions()]
        self.assertEqual("".join(kinds), "xx-xxxx-xxx")

    def test_every_panel_key_shows_its_panel(self):
        for key, action in self.win._panel_actions.items():
            action.trigger()
            self.settle()
            self.assertEqual(self.win._current_panel_key(), key)


# ---------------------------------------------------------------------------
# Tool tabs
# ---------------------------------------------------------------------------


class TestToolTabs(_WindowTestCase, _ThemedSizes):
    def _arrows_visible(self, bar):
        return any(b.isVisible() for b in bar.findChildren(QToolButton))

    def test_every_tab_shows_without_scroll_arrows(self):
        """At 1366x768 and 1024x640, in both themes, with the maximize
        arrows in the corner and the Decoder counting new messages."""
        tabs = self.win._right_tabs
        bar = tabs.tabBar()

        def check(theme):
            for size in ((1366, 768), (1024, 640)):
                self.win.resize(*size)
                self.settle()
                self.win._apply_default_splitter_sizes()
                self.settle()
                for unread in (0, 12, 5000):
                    self.win._decoder_unread = unread
                    self.win._update_decoder_tab_title()
                    self.settle()
                    total = sum(bar.tabRect(i).width() for i in range(bar.count()))
                    self.assertLessEqual(total, bar.width(), (theme, size, unread))
                    self.assertFalse(self._arrows_visible(bar), (theme, size, unread))
                    self.assertTrue(self.win._tool_tabs.button.isVisible())

        self.each_theme(check)
        self.win._decoder_unread = 5000
        self.win._update_decoder_tab_title()
        index = tabs.indexOf(self.win._panel_pages["decoder"])
        self.assertEqual(tabs.tabText(index), "Decoder (99+)")
        self.assertIn("5,000 new", tabs.tabToolTip(index))

    def test_the_ham_tab_is_short_and_says_what_it_holds(self):
        tabs = self.win._right_tabs
        index = tabs.indexOf(self.win._ham_tabs)
        self.assertEqual(tabs.tabText(index), "Ham")
        self.assertIn("Ham ID", tabs.tabToolTip(index))

    def test_bookmarks_is_the_first_launch_tab(self):
        self.assertEqual(self.win._current_panel_key(), "bookmarks")

    def test_maximize_gives_the_tabs_the_whole_column_and_is_remembered(self):
        pane = self.win._right_splitter.widget(0)
        button = self.win._tool_tabs.button
        self.assertTrue(pane.isVisible())
        _click(button)
        self.settle()
        self.assertFalse(pane.isVisible())
        self.assertTrue(self.win._tools_max_action.isChecked())
        self.assertGreater(self.win._right_tabs.height(), 500)
        self.assertIs(self.store.values["layout/tools_maximized"], True)
        # The menu item brings the control panel back.
        self.win._tools_max_action.trigger()
        self.settle()
        self.assertTrue(pane.isVisible())
        self.assertFalse(button.isChecked())
        self.assertIs(self.store.values["layout/tools_maximized"], False)

    def test_double_clicking_a_tab_toggles_it(self):
        pane = self.win._right_splitter.widget(0)
        bar = self.win._right_tabs.tabBar()
        bar.tabBarDoubleClicked.emit(1)
        self.settle()
        self.assertFalse(pane.isVisible())
        self.win._ham_tabs.tabBar().tabBarDoubleClicked.emit(0)
        self.settle()
        self.assertTrue(pane.isVisible())

    def test_ctrl_l_brings_the_control_panel_back(self):
        self.win._tool_tabs.set_maximized(True)
        self.settle()
        QTest.keySequence(self.win, QKeySequence("Ctrl+L"))
        self.settle()
        self.assertTrue(self.win._right_splitter.widget(0).isVisible())
        self.assertTrue(self.win._frequency_field().hasFocus())

    def test_a_new_window_opens_maximized_when_left_so(self):
        self.store.values["layout/tools_maximized"] = True
        win = self.mw.SDRMainWindow()
        try:
            for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
                timer.stop()
            win.show()
            self.settle()
            self.assertTrue(win._tool_tabs.maximized)
            self.assertFalse(win._right_splitter.widget(0).isVisible())
            self.assertTrue(win._tools_max_action.isChecked())
        finally:
            win._confirm_discard_recording = lambda *_a: True
            win.close()
            win.deleteLater()
            self.settle()


class TestToolbarFits(_WindowTestCase, _ThemedSizes):
    def test_nothing_falls_into_the_extension_menu_at_1024_px(self):
        def check(theme):
            self.win.resize(1024, 640)
            self.settle(10)
            for toolbar in (self.win._toolbar, self.win._receiver_bar):
                ext = [
                    b
                    for b in toolbar.findChildren(QToolButton)
                    if b.objectName() == "qt_toolbar_ext_button"
                ]
                self.assertFalse(any(b.isVisible() for b in ext), (theme, toolbar))
                for action in toolbar.actions():
                    widget = toolbar.widgetForAction(action)
                    if widget is None or not action.isVisible():
                        continue
                    self.assertTrue(widget.isVisible(), (theme, widget))
                    right = widget.mapTo(self.win, widget.rect().topRight()).x()
                    self.assertLessEqual(right, self.win.width(), (theme, widget))
            # The level meter stays in the top row, by FREQ.
            self.assertTrue(self.win._level_meter.isVisible())

        self.each_theme(check)

    def test_audio_and_volume_are_in_the_listening_row(self):
        bar = self.win._receiver_bar
        widgets = [bar.widgetForAction(a) for a in bar.actions()]
        self.assertIn(self.win._audio_button, widgets)
        self.assertIn(self.win._volume_slider, widgets)


# ---------------------------------------------------------------------------
# Audio: four states, Mute and the volume from the keyboard
# ---------------------------------------------------------------------------


class TestAudioStates(_WindowTestCase):
    def _audio(self, problem=None, opens=True):
        return (
            mock.patch.object(
                type(self.win), "_audio_output_problem", return_value=problem
            ),
            mock.patch.object(self.win._audio, "start", return_value=opens),
        )

    def test_four_states_say_why(self):
        win = self.win
        no_output, opens = self._audio(problem=None)
        with no_output, opens:
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._set_audio_enabled(True)
            self.assertEqual(win._audio_state(), "on")
            self.assertEqual(win._audio_button.text().strip(), "Audio")
            win._set_demod_mode("None (I/Q)")
            self.assertEqual(win._audio_state(), "silent")
            self.assertEqual(win._audio_button.text().strip(), "No audio")
            self.assertIn("I/Q", win._audio_button.toolTip())
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._mute_action.trigger()  # muted by you
            self.assertEqual(win._audio_state(), "muted")
            self.assertEqual(win._audio_button.text().strip(), "Muted")
            self.assertIsNone(win._audio_button.property("tone") or None)
        missing, opens = self._audio(problem="No audio output device")
        with missing, opens:
            win._mute_action.trigger()  # unmute: there is no output
        self.assertEqual(win._audio_state(), "no_output")
        self.assertEqual(win._audio_button.text().strip(), "No output")
        self.assertEqual(win._audio_button.property("tone"), "warning")
        self.assertEqual(win._audio_button.accessibleName(), "No audio output device")

    def test_no_output_click_tries_again_then_offers_the_output_choice(self):
        win = self.win
        missing, opens = self._audio(problem="No audio output device")
        with missing, opens:
            win._set_audio_enabled(True)
            self.assertEqual(win._audio_state(), "no_output")
            with (
                mock.patch(
                    "sdr_module.gui.audio_sink.output_devices",
                    return_value=[("usb", "USB Headset")],
                ),
                mock.patch.object(win, "_show_preferences") as prefs,
            ):
                win._audio_button.click()
            prefs.assert_called_once()
        # Plugged in: the click plays at once.
        present, opens = self._audio(problem=None)
        with present, opens:
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._audio_button.click()
        self.assertEqual(win._audio_state(), "on")

    def test_ctrl_m_and_m_mute(self):
        win = self.win
        present, opens = self._audio(problem=None)
        with present, opens:
            win._set_demod_mode("FM", "5 kHz", "25 kHz")
            win._set_audio_enabled(True)
            QTest.keySequence(win, QKeySequence("Ctrl+M"))
            self.assertFalse(win._audio_enabled)
            win._focus_plots()
            self.settle()
            QTest.keyClick(win._spectrum, Qt.Key.Key_M)
            self.assertTrue(win._audio_enabled)
            # Not while typing.
            field = win._frequency_field()
            field.setFocus()
            self.settle()
            QTest.keyClick(field, Qt.Key.Key_M)
            self.assertTrue(win._audio_enabled)

    def test_volume_keys_step_by_ten_and_say_so(self):
        win = self.win
        win._volume_slider.setValue(60)
        QTest.keySequence(win, QKeySequence("Ctrl+Up"))
        self.assertEqual(win._volume_slider.value(), 70)
        self.assertTrue(win._message_label.text().startswith("Volume 70%"))
        for _ in range(3):
            QTest.keySequence(win, QKeySequence("Ctrl+Down"))
        self.assertEqual(win._volume_slider.value(), 40)
        self.assertEqual(self.store.values["audio_volume"], 40)
        # A flash, not kept in the recent messages.
        self.assertFalse(any("Volume" in e.text for e in win._status_history.entries()))


# ---------------------------------------------------------------------------
# Status bar and title
# ---------------------------------------------------------------------------


class TestStatusBar(_WindowTestCase):
    def test_the_last_twenty_messages_open_from_the_message_area(self):
        win = self.win
        for i in range(25):
            win._show_status_message(f"Message {i}", "warning" if i == 24 else None)
        entries = win._status_history.entries()
        self.assertEqual(len(entries), 20)
        self.assertEqual(entries[0].text, "Message 24")
        self.assertEqual(entries[-1].text, "Message 5")
        win._clear_status_message()
        _click(win._message_label)
        self.settle()
        popup = win._status_popup
        self.assertTrue(popup.isVisible())
        texts = popup.texts()
        self.assertEqual(len(texts), 20)
        self.assertTrue(texts[0].endswith("Message 24"))
        popup.close()

    def test_a_repeated_message_counts_instead_of_filling_the_list(self):
        for _ in range(3):
            self.win._show_status_message("Already bookmarked: X", "info")
        entries = self.win._status_history.entries()
        self.assertEqual(entries[0].count, 3)
        self.assertEqual(len([e for e in entries if "Already" in e.text]), 1)

    def test_warnings_go_to_error_history_without_a_prefix(self):
        from sdr_module.gui.error_log_dialog import install_history_handler

        history = install_history_handler()
        history.clear_history()
        self.win._show_status_message("Disk nearly full", "warning")
        self.win._show_status_message("Tuned to 145.800 MHz", "info")
        self.win._show_status_error("The device didn't start")
        self.win._show_status_message("Logged by the caller", "warning", log=False)
        texts = [text for _t, level, _n, text in history.entries()]
        self.assertEqual(texts, ["Disk nearly full", "The device didn't start"])
        self.assertTrue(
            all(level == logging.WARNING for _t, level, _n, _x in history.entries())
        )
        history.clear_history()

    def test_badge_and_device_name_do_what_helps(self):
        win = self.win
        with mock.patch.object(win, "_prompt_no_device", return_value="cancel") as ask:
            _click(win._state_badge)
            _click(win._device_label)
        self.assertEqual(ask.call_count, 2)
        self.assertEqual(
            win._state_badge.cursor().shape(), Qt.CursorShape.PointingHandCursor
        )
        win._start_demo_mode()
        self.settle()
        self.assertTrue(win._is_running)
        self.assertEqual(win._state_badge.cursor().shape(), Qt.CursorShape.ArrowCursor)
        win._stop_acquisition()
        self.assertIn("Click", win._state_badge.toolTip())
        _click(win._state_badge)  # STOPPED: receive
        self.assertTrue(win._is_running)
        with mock.patch.object(win, "_show_device_dialog") as connect:
            _click(win._device_label)
        connect.assert_called_once_with()

    def test_rate_click_opens_the_sample_rate_menu(self):
        with mock.patch.object(self.win._device_controls, "popup_rate_menu") as popup:
            _click(self.win._rate_label)
        popup.assert_called_once()


class TestWindowTitle(_WindowTestCase):
    def test_title_names_the_frequency_mode_device_and_recording(self):
        win = self.win
        self.assertEqual(win.windowTitle(), "SDR Module")
        win._start_demo_mode()
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertEqual(win.windowTitle(), "100.100 MHz FM — Demo Mode — SDR Module")
        win._record_action.trigger()
        win._update_status()
        self.assertEqual(
            win.windowTitle(), "● REC · 100.100 MHz FM — Demo Mode — SDR Module"
        )
        win._record_action.trigger()
        win.set_frequency(145.8e6)
        self.assertTrue(win.windowTitle().startswith("145.800 MHz FM — "))
        win._set_demod_mode("None (I/Q)")
        self.assertTrue(win.windowTitle().startswith("145.800 MHz I/Q — "))


# ---------------------------------------------------------------------------
# The band plan at the tuned frequency
# ---------------------------------------------------------------------------


class TestBandPlanLine(unittest.TestCase):
    def test_lines(self):
        from sdr_module.core.band_info import band_plan_line

        self.assertEqual(
            band_plan_line(14.06e6),
            "20m Ham · 14.025–14.15 MHz · General and up: CW/Data, max 1500 W",
        )
        self.assertEqual(band_plan_line(100.1e6), "FM Broadcast · 87.5–108 MHz")
        self.assertIn("Technician: CW, max 200 W", band_plan_line(7.05e6))
        self.assertIn("transmitting locked out", band_plan_line(121.5e6))
        self.assertIn("license-free: FM, max 2 W", band_plan_line(151.88e6))
        self.assertTrue(band_plan_line(433.92e6).startswith("70cm Ham"))
        self.assertEqual(band_plan_line(300e6), "")

    def test_license_note(self):
        from sdr_module.core.band_info import license_note
        from sdr_module.core.frequency_manager import LicenseClass

        self.assertIn("doesn't include", license_note(14.06e6, LicenseClass.TECHNICIAN))
        self.assertIn("up to 200 W", license_note(7.05e6, LicenseClass.TECHNICIAN))
        self.assertIn(
            "needs an amateur license", license_note(14.06e6, LicenseClass.NONE)
        )
        self.assertEqual(license_note(100.1e6, LicenseClass.GENERAL), "")


class TestBandPlanInTheWindow(_WindowTestCase):
    def test_info_tab_and_freq_tooltip_follow_tuning(self):
        win = self.win
        win.set_frequency(14.06e6)
        win._show_panel("info")
        self.settle()
        self.assertTrue(win._info_panel.value("band").startswith("20m Ham"))
        self.assertIn("Band plan: 20m Ham", win._freq_label.toolTip())
        win.set_frequency(100.1e6)
        self.assertEqual(win._info_panel.value("band"), "FM Broadcast · 87.5–108 MHz")


# ---------------------------------------------------------------------------
# Help: the web pages, version info, the welcome screen again (MUST 5)
# ---------------------------------------------------------------------------


class TestHelpMenu(_WindowTestCase):
    def test_items(self):
        texts = [
            a.text().split("\t")[0].replace("&", "")
            for a in _menu(self.win, "Help").actions()
            if not a.isSeparator()
        ]
        self.assertEqual(
            texts,
            [
                "Welcome and Quick Start...",
                "Keyboard Shortcuts...",
                "Online Documentation",
                "Report a Problem...",
                "About SDR Module...",
            ],
        )

    def test_online_documentation_opens_the_readme(self):
        with mock.patch(
            "sdr_module.gui.version_info.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            _item(_menu(self.win, "Help"), "Online Documentation").trigger()
        self.assertEqual(
            open_url.call_args[0][0].toString(),
            "https://github.com/kase1111-hash/SDR#readme",
        )

    def test_report_a_problem_fills_in_the_version_info(self):
        from urllib.parse import parse_qs, urlsplit

        from sdr_module import __version__

        with mock.patch(
            "sdr_module.gui.version_info.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            _item(_menu(self.win, "Help"), "Report a Problem...").trigger()
        url = urlsplit(bytes(open_url.call_args[0][0].toEncoded()).decode())
        self.assertEqual(
            f"{url.scheme}://{url.netloc}{url.path}",
            "https://github.com/kase1111-hash/SDR/issues/new",
        )
        body = parse_qs(url.query)["body"][0]
        self.assertIn(f"SDR Module {__version__}", body)
        self.assertIn("Device: No device", body)
        # On the clipboard as well, and the status bar says so.
        self.assertIn(f"SDR Module {__version__}", QApplication.clipboard().text())
        self.assertIn("issue tracker", self.win._message_label.text())

    def test_without_a_browser_the_address_is_given(self):
        with mock.patch(
            "sdr_module.gui.version_info.QDesktopServices.openUrl", return_value=False
        ):
            self.win._report_problem()
        self.assertIn(
            "https://github.com/kase1111-hash/SDR/issues",
            self.win._message_label.text(),
        )

    def test_about_and_error_history_copy_the_version_info(self):
        from sdr_module.gui.error_log_dialog import ErrorLogDialog
        from sdr_module.gui.version_info import AboutDialog

        self.win._start_demo_mode()
        about = AboutDialog(self.win)
        self.addCleanup(about.deleteLater)
        QApplication.clipboard().setText("")
        about.copy_button.click()
        text = QApplication.clipboard().text()
        for part in ("SDR Module", "Python", "Qt ", "NumPy", "Driver for RTL-SDR"):
            self.assertIn(part, text)
        self.assertIn("Device: Demo Device", text)
        self.assertEqual(about.feedback.text(), "Copied to the clipboard.")
        self.assertIn("Device: Demo Device", about.info.toPlainText())

        history = ErrorLogDialog(self.win)
        self.addCleanup(history.deleteLater)
        QApplication.clipboard().setText("")
        history._version_btn.click()
        self.assertIn("Device: Demo Device", QApplication.clipboard().text())
        self.assertIn("Version info copied", history._summary.text())


class TestWelcomeAgain(_WindowTestCase):
    """mc:MC-12: Help > Welcome while the demo ran said "Mode set to FM"
    (then tuned an I/Q band) and offered to start the demo again."""

    def run_welcome(self, band, seen):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        def fake_exec(wiz):
            seen["status"] = wiz._status.text()
            seen["demo_offered"] = not wiz._demo_check.isHidden()
            seen["connect_offered"] = not wiz._connect_check.isHidden()
            wiz._band.setCurrentText(band)
            seen["hint"] = wiz._band_hint.text()
            wiz._on_accept()
            return wiz.result()

        with mock.patch.object(FirstRunWizard, "exec", fake_exec):
            self.win._show_welcome()

    def test_rerun_while_the_demo_runs(self):
        win = self.win
        win._start_demo_mode()
        seen = {}
        self.run_welcome("ADS-B (1090 MHz)", seen)
        self.assertIn("Demo Mode is running", seen["status"])
        self.assertFalse(seen["demo_offered"])
        self.assertFalse(seen["connect_offered"])  # (no SDR was found)
        self.assertTrue(win._demo_mode and win._is_running)
        self.assertEqual(win._current_frequency(), 1090e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "None (I/Q)")
        message = win._message_label.text()
        self.assertTrue(message.startswith("ADS-B: 1090.000 MHz, I/Q"), message)
        self.assertIn("ADS-B decoder on", message)
        self.assertNotIn("Mode set to FM", message)
        self.run_welcome("ISM 433 (433.05–434.79 MHz)", seen)
        self.assertEqual(
            win._message_label.text(), "ISM 433 MHz: 433.920 MHz, I/Q, 500 kHz filter"
        )
        self.assertEqual(win._control_panel._bw_combo.currentText(), "500 kHz")

    def test_first_run_starts_the_demo_then_tunes_the_band(self):
        win = self.win
        seen = {}
        self.run_welcome("2m Ham (144–148 MHz)", seen)
        self.assertTrue(seen["demo_offered"])
        self.assertEqual(
            seen["hint"],
            "Tunes to 2m Calling: 146.520 MHz · FM, 5 kHz deviation, 25 kHz filter.",
        )
        self.assertTrue(win._demo_mode and win._is_running)
        self.assertEqual(win._current_frequency(), 146.52e6)
        self.assertEqual(win._control_panel.get_fm_deviation(), 5e3)
        self.assertEqual(
            win._message_label.text(),
            "Demo mode (simulated signals). 2m Calling: 146.520 MHz, FM, 25 kHz "
            "filter",
        )


# ---------------------------------------------------------------------------
# Small chrome bugs (MUST 6)
# ---------------------------------------------------------------------------


class TestHidingTheFocusedPlot(_WindowTestCase):
    """mc:MC-09: hiding the focused waterfall threw the focus into the
    Frequency field (Space typed there, Up retuned 10 kHz)."""

    def test_the_focus_goes_to_the_other_plot(self):
        win = self.win
        win._waterfall.setFocus()
        self.settle()
        win._waterfall_action.trigger()  # View > Waterfall off
        self.settle()
        self.assertFalse(win._waterfall.isVisible())
        self.assertTrue(win._spectrum.hasFocus())
        freq = win._current_frequency()
        QTest.keyClick(win._spectrum, Qt.Key.Key_Up)
        self.assertEqual(win._current_frequency(), freq)
        win._waterfall_action.trigger()
        win._spectrum.setFocus()
        self.settle()
        win._spectrum_action.trigger()  # and the other way round
        self.settle()
        self.assertTrue(win._waterfall.hasFocus())

    def test_a_plot_without_the_focus_leaves_it_alone(self):
        win = self.win
        field = win._frequency_field()
        field.setFocus()
        self.settle()
        win._waterfall_action.trigger()
        self.settle()
        self.assertTrue(field.hasFocus())


class TestDefaultSplitAtTheMinimumSize(_WindowTestCase, _ThemedSizes):
    """mc:MC-10, ux:UX-B15: at 1024x640 (with the theme) the tool tabs'
    minimum height pushed the split into the Presets group."""

    def check(self, theme, width, height):
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QGroupBox

        win = self.win
        win._layout_initialized = False
        win._splitters_restored = False
        win.resize(width, height)
        self.settle()
        win._apply_default_splitter_sizes()
        self.settle()
        pane = win._right_splitter.widget(0)
        split = win._right_splitter.sizes()[0]
        for group in win._control_panel.findChildren(QGroupBox):
            if not group.isVisibleTo(win._control_panel):
                continue
            top = group.mapTo(pane, QPoint(0, 0)).y()
            self.assertFalse(
                top < split < top + group.height(),
                f"{theme} {width}x{height}: split at {split} cuts {group.title()}",
            )
        # Apply Preset is whole above the split, or under it (scrolled to).
        button = win._control_panel._apply_preset_btn
        top = button.mapTo(pane, QPoint(0, 0)).y()
        bottom = top + button.height()
        self.assertTrue(
            bottom <= split or top >= split,
            f"{theme} {width}x{height}: Apply Preset ({top}-{bottom}) cut at {split}",
        )

    def test_no_group_is_cut(self):
        def check(theme):
            for width, height in ((1024, 640), (1024, 600), (1366, 768)):
                self.check(theme, width, height)

        self.each_theme(check)


class TestScreenshotFormats(_WindowTestCase):
    """mc:MC-15: "shot.jpg" got PNG bytes; mc:MC-U07: a fixed name in the
    working folder."""

    def save(self, typed, chosen="PNG image (*.png)"):
        import tempfile

        folder = tempfile.mkdtemp(prefix="sdr-shot-")
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        with mock.patch(
            "sdr_module.gui.file_dialogs.QFileDialog.getSaveFileName",
            return_value=(os.path.join(folder, typed), chosen),
        ) as ask:
            self.win._save_screenshot()
        return folder, ask

    def head(self, path, n=4):
        with open(path, "rb") as f:
            return f.read(n)

    def test_the_format_follows_the_extension(self):
        for typed, magic in (
            ("shot.png", b"\x89PNG"),
            ("shot.jpg", b"\xff\xd8\xff"),
            ("shot.jpeg", b"\xff\xd8\xff"),
            ("shot.bmp", b"BM"),
        ):
            with self.subTest(typed=typed):
                folder, _ask = self.save(typed)
                path = os.path.join(folder, typed)
                self.assertTrue(self.head(path).startswith(magic))
                self.assertIn(typed, self.win._message_label.text())

    def test_no_extension_takes_the_chosen_format(self):
        folder, _ask = self.save("shot", "JPEG image (*.jpg *.jpeg)")
        self.assertTrue(
            self.head(os.path.join(folder, "shot.jpg")).startswith(b"\xff\xd8")
        )
        folder, _ask = self.save("notes.txt")
        self.assertTrue(
            self.head(os.path.join(folder, "notes.txt.png")).startswith(b"\x89PNG")
        )

    def test_the_dialog_offers_the_formats_and_a_timestamped_name(self):
        import re

        _folder, ask = self.save("x.png")
        args = ask.call_args[0]
        self.assertEqual(
            args[3], "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp)"
        )
        self.assertRegex(
            os.path.basename(args[2]), r"^sdr_\d+\.\d{3}MHz_\d{8}_\d{6}\.png$"
        )
        self.assertTrue(
            re.search(r"Screenshot saved to .*x\.png", self.win._message_label.text())
        )


class TestLayoutIsRemembered(_WindowTestCase):
    """mc:MC-U13: hiding a plot and the Frequency field's unit came back
    as the defaults after a restart."""

    def reopen(self):
        win = self.mw.SDRMainWindow()
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.close)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        win._confirm_discard_recording = lambda *_a: True
        win.show()
        self.settle()
        return win

    def test_plots_and_unit_come_back(self):
        win = self.win
        win._spectrum_action.trigger()  # a waterfall-only layout
        self.assertIs(self.store.values["layout/spectrum_visible"], False)
        win._control_panel._freq_input._unit_combo.setCurrentText("kHz")
        win._persist_state()
        self.assertEqual(self.store.values["layout/freq_unit"], "kHz")
        again = self.reopen()
        self.assertFalse(again._spectrum.isVisible())
        self.assertTrue(again._waterfall.isVisible())
        self.assertFalse(again._spectrum_action.isChecked())
        self.assertFalse(again._waterfall_action.isEnabled())  # the last plot
        self.assertEqual(
            again._control_panel._freq_input._unit_combo.currentText(), "kHz"
        )
        # Reset Layout shows both plots again, and that is remembered too.
        again._reset_layout()
        self.assertIs(self.store.values["layout/spectrum_visible"], True)


class TestConnectDialogDemoRow(unittest.TestCase):
    """mc:MC-U02: the Connect dialog's demo row says what it is."""

    def test_the_demo_row_is_use_demo_device(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = DeviceDialog()
        self.addCleanup(dialog.deleteLater)
        table = dialog._device_table
        row = table.rowCount() - 1
        self.assertEqual(table.item(row, 1).text(), "Demo Device")
        for col in range(table.columnCount()):
            self.assertIn("Device > Use Demo Device", table.item(row, col).toolTip())


class TestChirpDialogsRememberTheirFolder(_WindowTestCase):
    """mc:MC-U07: Import/Export Bookmarks (CHIRP CSV) opened in the folder
    the app was started from and forgot the last one."""

    def test_export_starts_in_documents_then_where_it_last_saved(self):
        import shutil
        import tempfile

        from sdr_module.gui import file_dialogs

        panel = self.win._bookmarks_panel
        panel.add_bookmark("Tower", 118.3e6, "AM", "", "")
        folder = tempfile.mkdtemp(prefix="sdr-chirp-")
        self.addCleanup(shutil.rmtree, folder, True)
        default = file_dialogs.start_folder(self.store, "channels")
        target = os.path.join(folder, "radio.csv")
        with (
            mock.patch(
                "sdr_module.gui.file_dialogs.QFileDialog.getSaveFileName",
                return_value=(target, ""),
            ) as ask,
            mock.patch("sdr_module.gui.bookmarks_panel.QMessageBox.information"),
        ):
            self.assertEqual(panel.export_csv(), 1)
            first = ask.call_args[0][2]
            panel.export_csv()
            second = ask.call_args[0][2]
        self.assertEqual(first, os.path.join(default, "bookmarks.csv"))
        self.assertEqual(second, os.path.join(folder, "bookmarks.csv"))
        self.assertTrue(os.path.exists(target))


class TestErrorHistorySummary(_WindowTestCase):
    """mc:MC-U19: "Saved to /a/long/path..." was cut at the dialog edge."""

    def test_a_long_line_is_shortened_in_the_middle_with_the_whole_in_a_tooltip(self):
        from PyQt6.QtWidgets import QLabel

        from sdr_module.gui.error_log_dialog import ErrorLogDialog

        dialog = ErrorLogDialog(self.win)
        self.addCleanup(dialog.deleteLater)
        dialog.resize(520, 400)
        dialog.show()
        self.settle()
        path = "/home/someone/" + "a-very-long-folder-name/" * 8 + "history.txt"
        dialog._flash(f"Saved to {path}")
        self.settle()
        summary = dialog._summary
        shown = QLabel.text(summary)
        self.assertIn("…", shown)
        self.assertTrue(shown.startswith("Saved to /home"))
        self.assertTrue(shown.endswith("history.txt"))
        self.assertEqual(summary.toolTip(), f"Saved to {path}")
        self.assertLessEqual(
            summary.fontMetrics().horizontalAdvance(shown),
            summary.contentsRect().width(),
        )


if __name__ == "__main__":
    unittest.main()
