#!/usr/bin/env python3
"""
The Demo Device runs in real time whichever way it is opened: Device >
Use Demo Device, Device > Connect... (its Demo row), or Start with no device
> Connect... (mc:MC-01, rx:RX-01, tl:TOOLS-02, ux:UX-B01).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_gui_tuning_demo.py``
"""

import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QCoreApplication, QEvent


def _flush_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


class TestDemoThroughConnectDialog(_WindowTestCase):
    """Device > Connect... with the Demo row selected."""

    def _connect_demo(self, rate=None, start_after=False):
        from sdr_module.gui import device_dialog

        def fake_exec(dialog):
            # No hardware here: the Demo Device is the only (preselected) row.
            row = dialog._device_table.rowCount() - 1
            dialog._device_table.selectRow(row)
            if rate is not None:
                dialog._rate_combo.setCurrentIndex(dialog._rate_combo.findData(rate))
            dialog._on_accept()
            return dialog.result()

        with mock.patch.object(device_dialog.DeviceDialog, "exec", fake_exec):
            result = self.win._show_device_dialog(start_after=start_after)
        _flush_deletes()
        return result

    def _paced_read(self, wait_s=0.12):
        """Samples one display frame takes after ``wait_s`` seconds."""
        self.win._read_block()  # the pacer's first read: one block
        time.sleep(wait_s)
        block = self.win._read_block()
        return 0 if block is None else len(block)

    def test_demo_row_is_demo_mode_and_paced_in_real_time(self):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        self.assertTrue(self._connect_demo(rate=1.0e6, start_after=True))
        self.assertIsInstance(self.win._device, MockDevice)
        self.assertTrue(self.win._demo_mode)
        self.assertTrue(self.win._is_running)
        self.assertEqual(
            self.win._device_display_name(), "Demo Device (simulated signals)"
        )
        # (The title names the frequency and mode too: mc:MC-U11.)
        self.assertEqual(
            self.win.windowTitle(), "100.100 MHz FM — Demo Mode — SDR Module"
        )
        # A frame takes what the time since the last one covers (capped at
        # the pacer's probe limit, 0.05 s, until frames are known to keep
        # up), not one 2048-sample display block.
        count = self._paced_read()
        self.assertGreater(count, 10 * DISPLAY_BLOCK)
        self.assertGreaterEqual(count, int(0.04 * 1.0e6))

    def test_start_with_no_device_then_connect_is_paced(self):
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        self.win._prompt_no_device = lambda: "connect"
        from sdr_module.gui import device_dialog

        def fake_exec(dialog):
            dialog._device_table.selectRow(dialog._device_table.rowCount() - 1)
            dialog._on_accept()
            return dialog.result()

        with mock.patch.object(device_dialog.DeviceDialog, "exec", fake_exec):
            self.win._toggle_acquisition()
        _flush_deletes()
        self.assertTrue(self.win._is_running)
        self.assertTrue(self.win._demo_mode)
        self.assertGreater(self._paced_read(), 10 * DISPLAY_BLOCK)

    def test_any_mock_device_is_paced_even_without_the_flag(self):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        dev = MockDevice()
        dev.start_rx()
        self.win._device = dev
        self.win._demo_mode = False
        self.win._pacer.reset()
        self.assertGreater(self._paced_read(), 10 * DISPLAY_BLOCK)
        self.win._device = None

    def test_device_change_starts_pacing_and_history_afresh(self):
        self.win._rx_history.append((time.monotonic() - 5.0, 10))
        self.win._pacer.overloaded = True
        self.win._realtime_hint_shown = True
        self.assertTrue(self._connect_demo())
        self.assertEqual(len(self.win._rx_history), 0)
        self.assertFalse(self.win._pacer.overloaded)
        self.assertIsNone(self.win._pacer._last)
        self.assertFalse(self.win._realtime_hint_shown)

    def test_reconnecting_the_running_demo_at_another_rate_stays_real_time(self):
        """The demo's own advice ("a lower sample rate in Device >
        Connect... helps") leads here: the demo must stay paced."""
        from sdr_module.gui.main_window import DISPLAY_BLOCK

        self.win._start_demo_mode()
        self.assertTrue(self.win._is_running)
        self.assertTrue(self._connect_demo(rate=1.0e6))
        self.assertTrue(self.win._demo_mode)
        self.assertTrue(self.win._is_running)
        self.assertEqual(self.win._device_sample_rate(), 1.0e6)
        self.assertGreater(self._paced_read(), 10 * DISPLAY_BLOCK)
        # A later Device > Use Demo Device keeps the rate chosen here.
        self.assertEqual(self.win._preferred_rate, 1.0e6)


if __name__ == "__main__":
    unittest.main()
