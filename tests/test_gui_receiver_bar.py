#!/usr/bin/env python3
"""
The receiver bar (MODE, FILTER, STEP; CONTRACTS 4.2), the tuning step and
what tunes by it (keys, the mouse wheel, the Frequency field's arrows), the
fixed-width VFO-style FREQ readout (mc:MC-13), keys that a focused control
owns (mc:MC-U09) and the back/forward tuning history (ux:UX-G13).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_gui_receiver_bar.py``
"""

import json
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton


def _wheel(widget, notches=1, modifiers=None, pos=None, angle=120):
    """Send a mouse wheel turn of ``notches`` x ``angle`` eighths of a degree."""
    mods = modifiers if modifiers is not None else Qt.KeyboardModifier.NoModifier
    where = QPointF(pos) if pos is not None else QPointF(widget.rect().center())
    event = QWheelEvent(
        where,
        QPointF(widget.mapToGlobal(where.toPoint())),
        QPoint(0, 0),
        QPoint(0, int(angle * notches)),
        Qt.MouseButton.NoButton,
        mods,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


class TestReceiverBar(_WindowTestCase):
    def bar(self):
        return self.win._receiver_bar

    def test_a_second_toolbar_row_with_mode_filter_and_step(self):
        bar = self.bar()
        self.assertTrue(self.win.toolBarBreak(bar))
        self.assertFalse(bar.isMovable())
        captions = [
            w.text()
            for w in bar.findChildren(QLabel)
            if w.property("role") == "caption"
        ]
        self.assertEqual(captions, ["MODE", "FILTER", "STEP"])
        for combo, name in (
            (bar.mode_combo, "Demodulation mode"),
            (bar.filter_combo, "Channel filter"),
            (bar.step_combo, "Tuning step"),
        ):
            self.assertEqual(combo.accessibleName(), name)
            self.assertTrue(combo.toolTip())
            # Tab reaches it; a click leaves the keys with the plots.
            self.assertEqual(combo.focusPolicy(), Qt.FocusPolicy.TabFocus)
        self.assertEqual(len(bar.findChildren(QComboBox)), 3)

    def test_mode_mirrors_the_control_panel_both_ways(self):
        panel_combo = self.win._control_panel._demod_combo
        bar_combo = self.bar().mode_combo
        self.assertEqual(
            [bar_combo.itemText(i) for i in range(bar_combo.count())],
            [panel_combo.itemText(i) for i in range(panel_combo.count())],
        )
        self.assertEqual(bar_combo.currentText(), panel_combo.currentText())
        bar_combo.setCurrentText("AM")
        self.assertEqual(panel_combo.currentText(), "AM")
        self.assertEqual(self.store.values["demod_mode"], "AM")  # handlers ran
        panel_combo.setCurrentText("USB")
        self.assertEqual(bar_combo.currentText(), "USB")
        self.assertTrue(bar_combo.itemData(3, Qt.ItemDataRole.ToolTipRole))

    def test_filter_mirrors_the_bandwidth_combo_both_ways(self):
        panel_combo = self.win._control_panel._bw_combo
        bar_combo = self.bar().filter_combo
        bar_combo.setCurrentText("25 kHz")
        self.assertEqual(panel_combo.currentText(), "25 kHz")
        self.assertEqual(self.win._channel_bandwidth(), 25e3)
        panel_combo.setCurrentText("2.4 MHz")
        self.assertEqual(bar_combo.currentText(), "2.4 MHz")

    def test_filter_copies_the_bandwidth_items_whenever_they_change(self):
        panel_combo = self.win._control_panel._bw_combo
        bar_combo = self.bar().filter_combo

        def items():
            return [bar_combo.itemText(i) for i in range(bar_combo.count())]

        # As a mode-dependent filter list would (T4): new items, new choice.
        panel_combo.clear()
        panel_combo.addItems(["1.8 kHz", "2.4 kHz", "2.7 kHz", "3.0 kHz"])
        panel_combo.setCurrentIndex(2)
        self.assertEqual(items(), ["1.8 kHz", "2.4 kHz", "2.7 kHz", "3.0 kHz"])
        self.assertEqual(bar_combo.currentText(), "2.7 kHz")
        panel_combo.insertItem(0, "1.5 kHz")
        self.assertEqual(items()[0], "1.5 kHz")
        self.assertEqual(bar_combo.currentText(), "2.7 kHz")
        panel_combo.setItemText(1, "1.9 kHz")
        self.assertEqual(items()[1], "1.9 kHz")
        panel_combo.removeItem(0)
        self.assertEqual(items(), ["1.9 kHz", "2.4 kHz", "2.7 kHz", "3.0 kHz"])
        bar_combo.setCurrentText("3.0 kHz")
        self.assertEqual(panel_combo.currentText(), "3.0 kHz")

    def test_filter_follows_a_bandwidth_combo_given_a_new_model(self):
        from PyQt6.QtGui import QStandardItem, QStandardItemModel

        panel_combo = self.win._control_panel._bw_combo
        bar_combo = self.bar().filter_combo
        # As many items as before, so only their texts tell the change.
        texts = [f"{w} Hz" for w in range(100, 100 * (panel_combo.count() + 1), 100)]
        model = QStandardItemModel()
        for text in texts:
            model.appendRow(QStandardItem(text))
        panel_combo.setModel(model)
        panel_combo.setCurrentIndex(1)
        self.assertEqual(
            [bar_combo.itemText(i) for i in range(bar_combo.count())], texts
        )
        self.assertEqual(bar_combo.currentText(), "200 Hz")
        # Later changes to the new model (no change of choice) show too.
        model.item(0).setText("150 Hz")
        model.appendRow(QStandardItem("5 kHz"))
        self.assertEqual(
            [bar_combo.itemText(i) for i in range(bar_combo.count())],
            ["150 Hz"] + texts[1:] + ["5 kHz"],
        )
        self.assertEqual(bar_combo.currentText(), "200 Hz")

    def test_step_choices(self):
        combo = self.bar().step_combo
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())],
            [
                "10 Hz",
                "50 Hz",
                "100 Hz",
                "500 Hz",
                "1 kHz",
                "2.5 kHz",
                "5 kHz",
                "6.25 kHz",
                "8.33 kHz",
                "9 kHz",
                "10 kHz",
                "12.5 kHz",
                "25 kHz",
                "50 kHz",
                "100 kHz",
                "200 kHz",
                "1 MHz",
            ],
        )

    def test_everything_fits_a_1024_px_window(self):
        self.win.resize(1024, 640)
        self.win.show()
        self.settle(10)
        for toolbar in (self.win._toolbar, self.bar()):
            widgets = [
                toolbar.widgetForAction(a)
                for a in toolbar.actions()
                if toolbar.widgetForAction(a) is not None
            ]
            self.assertTrue(widgets)
            for widget in widgets:
                self.assertTrue(widget.isVisible(), (toolbar.objectName(), widget))
                right = widget.mapTo(self.win, widget.rect().topRight()).x()
                self.assertLessEqual(right, self.win.width())


class TestTuningStep(_WindowTestCase):
    demo = True

    def bar(self):
        return self.win._receiver_bar

    def use(self, mode, deviation=None):
        self.win._set_demod_mode(mode, deviation)

    def test_each_mode_starts_with_its_step(self):
        for mode, deviation, step, text in (
            ("FM", "75 kHz", 100e3, "100 kHz"),
            ("FM", "5 kHz", 12.5e3, "12.5 kHz"),
            ("AM", None, 10e3, "10 kHz"),
            ("USB", None, 100.0, "100 Hz"),
            ("LSB", None, 100.0, "100 Hz"),
            ("CW", None, 10.0, "10 Hz"),
            ("None (I/Q)", None, 10e3, "10 kHz"),
        ):
            self.use(mode, deviation)
            self.assertEqual(self.win._step_hz, step, (mode, deviation))
            self.assertEqual(self.bar().step_combo.currentText(), text)
            self.assertTrue(self.win._step_actions[step].isChecked())
            steps = self.win._TUNE_STEPS_HZ
            self.assertEqual(steps[Qt.KeyboardModifier.NoModifier], step)
            self.assertEqual(steps[Qt.KeyboardModifier.ShiftModifier], 10 * step)
            self.assertEqual(steps[Qt.KeyboardModifier.ControlModifier], 1e6)
            self.assertEqual(self.win._control_panel._freq_input.step_hz(), step)

    def test_each_mode_remembers_the_step_chosen_for_it(self):
        self.use("AM")
        self.bar().step_combo.setCurrentText("9 kHz")
        self.assertEqual(self.win._step_hz, 9e3)
        self.assertEqual(self.store.values["tuning/step/am"], 9e3)
        self.use("USB")
        self.assertEqual(self.win._step_hz, 100.0)
        self.use("AM")
        self.assertEqual(self.win._step_hz, 9e3)
        self.assertEqual(self.bar().step_combo.currentText(), "9 kHz")
        # The menu picks too, and is remembered the same way.
        self.win._step_actions[25e3 / 3].trigger()
        self.assertEqual(self.store.values["tuning/step/am"], 25e3 / 3)
        self.assertEqual(self.bar().step_combo.currentText(), "8.33 kHz")

    def test_a_new_window_uses_the_remembered_step(self):
        self.store.values["tuning/step/wfm"] = 200e3
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertEqual(other._step_hz, 200e3)
        self.assertEqual(other._receiver_bar.step_combo.currentText(), "200 kHz")

    def test_f1_lists_the_arrow_keys_with_the_step(self):
        from sdr_module.gui.help_dialog import HelpDialog

        def rows():
            dialog = HelpDialog(self.win)
            self.addCleanup(dialog.deleteLater)
            return {k: d for c, k, d in dialog.shortcut_entries() if c == "Tuning"}

        listed = rows()
        self.assertEqual(
            listed[("Left", "Right")], "Tune down / up by one tuning step (100 kHz)"
        )
        self.assertEqual(
            listed[("Shift+Left", "Shift+Right")],
            "Tune down / up by ten tuning steps (1 MHz)",
        )
        self.assertEqual(listed[("Ctrl+Left", "Ctrl+Right")], "Tune down / up by 1 MHz")
        self.use("USB")
        self.assertEqual(
            rows()[("Left", "Right")], "Tune down / up by one tuning step (100 Hz)"
        )

    def test_arrow_keys_tune_by_the_step(self):
        self.use("FM", "5 kHz")  # 12.5 kHz
        self.win._spectrum.setFocus()
        self.settle()
        start = self.freq()
        QTest.keyClick(self.win._spectrum, Qt.Key.Key_Right)
        self.assertAlmostEqual(self.freq(), start + 12.5e3)
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertAlmostEqual(self.freq(), start + 137.5e3)
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier
        )
        self.assertAlmostEqual(self.freq(), start - 862.5e3)

    def test_brackets_change_the_step_with_the_plots_focused(self):
        self.win._waterfall.setFocus()
        self.settle()
        self.assertEqual(self.win._step_hz, 100e3)
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_BracketLeft)
        self.assertEqual(self.win._step_hz, 50e3)
        self.assertIn("Tuning step 50 kHz", self.win._message_label.text())
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_BracketRight)
        QTest.keyClick(self.win._waterfall, Qt.Key.Key_BracketRight)
        self.assertEqual(self.win._step_hz, 200e3)
        for _ in range(5):
            QTest.keyClick(self.win._waterfall, Qt.Key.Key_BracketRight)
        self.assertEqual(self.win._step_hz, 1e6)  # the largest
        # Not while another control has the focus.
        slider = self.win._control_panel._gain_slider
        slider.setFocus()
        self.settle()
        QTest.keyClick(slider, Qt.Key.Key_BracketLeft)
        self.assertEqual(self.win._step_hz, 1e6)

    def test_the_wheel_over_the_plots_tunes_by_the_step(self):
        start = self.freq()
        for canvas in (self.win._spectrum._canvas, self.win._waterfall._canvas):
            before = self.freq()
            _wheel(canvas, 1)
            self.assertAlmostEqual(self.freq(), before + 100e3)
            _wheel(canvas, -1, Qt.KeyboardModifier.ShiftModifier)
            self.assertAlmostEqual(self.freq(), before - 900e3)
            _wheel(canvas, 1, Qt.KeyboardModifier.ShiftModifier)
        self.assertAlmostEqual(self.freq(), start + 200e3)
        # Ctrl+wheel is left for zooming; half notches add up.
        before = self.freq()
        _wheel(self.win._spectrum._canvas, 1, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.freq(), before)
        _wheel(self.win._spectrum._canvas, 1, angle=60)
        self.assertEqual(self.freq(), before)
        _wheel(self.win._spectrum._canvas, 1, angle=60)
        self.assertAlmostEqual(self.freq(), before + 100e3)


class TestFreqReadout(_WindowTestCase):
    def test_fixed_width_so_the_toolbar_never_shifts(self):
        self.win.resize(1366, 768)
        self.win.show()
        self.settle(10)
        label = self.win._freq_label
        self.assertEqual(label.minimumWidth(), label.maximumWidth())
        level_x = self.win._level_label.mapTo(self.win, QPoint(0, 0)).x()
        for freq in (1090.000001e6, 146.5125e6, 24e6, 5999.999999e6):
            self.win.set_frequency(freq)
            self.settle(3)
            self.assertEqual(
                self.win._level_label.mapTo(self.win, QPoint(0, 0)).x(), level_x
            )
            # Every frequency's text fits inside the frame. (Not checked with
            # the size hint: QLabel never hints less than its fixed width.)
            spans = self.win._freq_digits.char_spans()
            self.assertGreaterEqual(spans[0][0], label.contentsRect().left())
            self.assertLessEqual(spans[-1][1], label.contentsRect().right() + 1)

    def test_the_wheel_over_a_digit_tunes_that_digit(self):
        self.win.resize(1366, 768)
        self.win.show()
        self.settle(10)
        label = self.win._freq_label
        digits = self.win._freq_digits
        self.assertEqual(label.text(), "100.100 MHz")
        spans = digits.char_spans()

        def center(i):
            x0, x1 = spans[i]
            return QPointF((x0 + x1) / 2, label.height() / 2)

        self.assertEqual(digits.digit_at(center(0).x()), (0, 100e6))
        self.assertEqual(digits.digit_at(center(2).x()), (2, 1e6))
        self.assertEqual(digits.digit_at(center(4).x()), (4, 100e3))
        self.assertEqual(digits.digit_at(center(6).x()), (6, 1e3))
        self.assertIsNone(digits.digit_at(center(3).x()))  # the point
        self.assertIsNone(digits.digit_at(center(8).x()))  # "MHz"
        _wheel(label, 1, pos=center(4))  # +100 kHz
        self.assertEqual(self.freq(), 100.2e6)
        _wheel(label, -3, pos=center(6))  # -3 kHz
        self.assertEqual(self.freq(), 100.197e6)
        _wheel(label, 1, pos=center(2))  # +1 MHz
        self.assertEqual(self.freq(), 101.197e6)
        before = self.freq()
        _wheel(label, 1, pos=center(8))  # not a digit: nothing
        self.assertEqual(self.freq(), before)
        # Clicking still opens the Frequency field.
        QTest.mouseClick(label, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(self.win._frequency_field().hasFocus())

    def test_scrolling_keeps_its_digit_when_the_text_shifts(self):
        self.win.resize(1366, 768)
        self.win.show()
        self.settle(10)
        label = self.win._freq_label
        self.win.set_frequency(146.5125e6)
        self.assertEqual(label.text(), "146.5125 MHz")
        x0, x1 = self.win._freq_digits.char_spans()[7]  # the 100 Hz digit
        where = QPointF((x0 + x1) / 2, label.height() / 2)
        for _ in range(6):  # 146.5125 -> 146.512 (shorter text) -> 146.5119
            _wheel(label, -1, pos=where)
        self.assertAlmostEqual(self.freq(), 146.5119e6, places=0)

    def test_theme_switches_and_refits_keep_the_width(self):
        """Each refit measured the label's size hint, which QLabel never
        lets fall below the width fixed before: every theme switch added the
        width of the characters FREQ wasn't showing (about 59 px)."""
        self.win.resize(1366, 768)
        self.win.show()
        self.settle(10)
        label = self.win._freq_label
        digits = self.win._freq_digits
        self.win._toggle_theme()  # a real theme's style (the tests start bare)
        self.settle(5)
        width = label.width()
        self.assertEqual(digits.fit_width(), width)
        self.assertEqual(digits.fit_width(), width)
        level_x = self.win._level_label.mapTo(self.win, QPoint(0, 0)).x()
        for _ in range(4):  # the other theme and back, twice
            self.win._toggle_theme()
            self.settle(5)
        self.assertEqual(label.width(), width)
        self.assertEqual(
            self.win._level_label.mapTo(self.win, QPoint(0, 0)).x(), level_x
        )
        # The widest frequency fits inside the frame.
        self.win.set_frequency(5999.999999e6)
        self.settle(3)
        self.assertEqual(label.text(), "5999.999999 MHz")
        spans = digits.char_spans()
        self.assertGreaterEqual(spans[0][0], label.contentsRect().left())
        self.assertLessEqual(spans[-1][1], label.contentsRect().right() + 1)

    def test_the_width_follows_a_new_font(self):
        from PyQt6.QtGui import QFont

        label = self.win._freq_label
        before = label.width() if label.isVisible() else label.minimumWidth()
        font = QFont(label.font())
        font.setPixelSize(max(8, label.fontInfo().pixelSize()) * 2)
        label.setFont(font)
        self.settle(3)
        self.assertGreater(label.minimumWidth(), before)
        self.assertEqual(label.minimumWidth(), label.maximumWidth())
        self.assertEqual(label.minimumWidth(), self.win._freq_digits.fit_width())

    def test_digit_geometry_matches_the_painted_text(self):
        """The digit boxes are where the label really paints its glyphs."""
        self.win.resize(1366, 768)
        self.win.show()
        self.settle(10)
        label = self.win._freq_label
        image = label.grab().toImage()
        background = image.pixelColor(3, label.height() // 2)
        inked = [
            x
            for x in range(image.width())
            if any(
                image.pixelColor(x, y) != background
                for y in range(4, image.height() - 4)
            )
        ]
        spans = self.win._freq_digits.char_spans()
        self.assertTrue(inked)
        # The glyphs start in the first character's box and end in the last.
        self.assertLessEqual(spans[0][0] - 1, min(inked))
        self.assertLess(min(inked), spans[0][1])
        self.assertLessEqual(max(inked), spans[-1][1] + 1)


class TestKeysAFocusedControlOwns(_WindowTestCase):
    """mc:MC-U09: Space and the arrows act on the receiver only when the
    focused control has no use for them."""

    def setUp(self):
        super().setUp()
        self.prompted = []
        self.win._prompt_no_device = lambda: self.prompted.append(1) or "cancel"

    def press(self, widget, key, mods=None):
        widget.setFocus()
        self.settle()
        self.assertTrue(widget.hasFocus())
        if mods is None:
            QTest.keyClick(widget, key)
        else:
            QTest.keyClick(widget, key, mods)
        self.settle()

    def test_space_on_a_slider_or_tab_bar_does_not_start_receiving(self):
        self.press(self.win._control_panel._squelch_slider, Qt.Key.Key_Space)
        self.press(self.win._control_panel._gain_slider, Qt.Key.Key_Space)
        self.press(self.win._right_tabs.tabBar(), Qt.Key.Key_Space)
        self.assertEqual(self.prompted, [])
        self.press(self.win._spectrum, Qt.Key.Key_Space)
        self.assertEqual(self.prompted, [1])

    def test_arrows_on_a_combo_or_button_do_not_tune(self):
        start = self.freq()
        bar = self.win._receiver_bar
        index = bar.mode_combo.currentIndex()
        self.press(bar.mode_combo, Qt.Key.Key_Left)
        self.press(bar.step_combo, Qt.Key.Key_Right)
        button = next(
            b
            for b in self.win._control_panel.findChildren(QPushButton)
            if "Tune up" in b.accessibleName()
        )
        self.press(button, Qt.Key.Key_Right)
        self.assertEqual(self.freq(), start)
        self.assertEqual(bar.mode_combo.currentIndex(), index)
        self.press(self.win._spectrum, Qt.Key.Key_Right)
        self.assertNotEqual(self.freq(), start)


class TestTuningHistory(_WindowTestCase):
    demo = True

    def visit(self, freq, mode=None, deviation=None):
        self.assertTrue(self.win.set_frequency(freq))
        if mode is not None:
            self.win._set_demod_mode(mode, deviation)
        self.win._commit_history_entry()  # as after a moment on it

    def test_back_and_forward(self):
        self.win._commit_history_entry()  # 100.1 MHz, FM
        self.assertFalse(self.win._back_action.isEnabled())
        self.visit(99.5e6)
        self.visit(125e6, "AM")
        self.assertTrue(self.win._back_action.isEnabled())
        self.assertEqual(self.win._back_action.shortcut().toString(), "Alt+Left")
        self.assertEqual(self.win._forward_action.shortcut().toString(), "Alt+Right")
        self.assertIn("99.500 MHz (FM)", self.win._back_action.toolTip())
        self.win._back_action.trigger()
        self.assertEqual(self.freq(), 99.5e6)
        self.assertEqual(self.win._control_panel.get_demod_mode(), "FM")
        self.assertIn("Back to 99.500 MHz (FM)", self.win._message_label.text())
        self.win._back_action.trigger()
        self.assertEqual(self.freq(), 100.1e6)
        self.assertFalse(self.win._back_action.isEnabled())
        self.win._forward_action.trigger()
        self.win._forward_action.trigger()
        self.assertEqual(self.freq(), 125e6)
        self.assertEqual(self.win._control_panel.get_demod_mode(), "AM")
        self.assertFalse(self.win._forward_action.isEnabled())

    def test_stepping_through_a_band_adds_one_entry(self):
        self.win._commit_history_entry()
        for _ in range(5):
            self.win._tune_steps(1)  # no moment on any step
        self.win._commit_history_entry()
        recent = self.win._tuning_history.recent()
        self.assertEqual([e.freq_hz for e in recent], [100.6e6, 100.1e6])
        self.assertFalse(self.win._history_timer.isActive())

    def test_back_right_after_a_jump_goes_to_the_frequency_before(self):
        self.win._commit_history_entry()
        self.win.set_frequency(99.5e6)  # not committed yet
        self.win._tune_back()
        self.assertEqual(self.freq(), 100.1e6)

    def test_closing_keeps_the_frequency_just_tuned(self):
        self.win._commit_history_entry()
        self.win.set_frequency(99.5e6)  # not yet a moment on it
        self.assertTrue(self.win._history_timer.isActive())
        self.win._persist_state = lambda: None
        self.win.close()
        saved = json.loads(self.store.values["tuning/recent"])
        self.assertEqual(saved[0]["freq_hz"], 99.5e6)

    def test_recent_frequencies_menu_and_persistence(self):
        self.win._commit_history_entry()
        self.visit(146.52e6, "FM", "5 kHz")
        self.visit(7.074e6, "USB")
        menu = self.win._recent_menu
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(
            texts,
            [
                "7.074 MHz (USB)",
                "146.520 MHz (FM)",
                "100.100 MHz (FM)",
                "&Clear Recent Frequencies",
            ],
        )
        saved = json.loads(self.store.values["tuning/recent"])
        self.assertEqual(
            saved[1], {"freq_hz": 146.52e6, "mode": "FM", "deviation": "5 kHz"}
        )
        menu.actions()[1].trigger()  # 146.520 MHz (FM)
        self.assertEqual(self.freq(), 146.52e6)
        self.assertEqual(self.win._control_panel.get_fm_deviation_text(), "5 kHz")
        # A new window offers them, and Back goes to the one before.
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        texts = [a.text() for a in other._recent_menu.actions() if not a.isSeparator()]
        self.assertEqual(texts[0], "7.074 MHz (USB)")
        self.assertTrue(other._back_action.isEnabled())
        # Clearing.
        clear = [a for a in menu.actions() if "Clear" in a.text()][0]
        clear.trigger()
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(
            texts, ["No recent frequencies yet", "&Clear Recent Frequencies"]
        )
        self.assertEqual(json.loads(self.store.values["tuning/recent"]), [])


if __name__ == "__main__":
    unittest.main()
