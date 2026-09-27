#!/usr/bin/env python3
"""
The default right-column split never cuts through a control panel section.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_layout_split.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QApplication, QGroupBox


class TestDefaultSplitLandsBetweenSections(_WindowTestCase):
    def _sections_straddling_split(self, width, height):
        win = self.win
        win._layout_initialized = False
        win._splitters_restored = False
        win.resize(width, height)
        win.show()
        for _ in range(5):
            QApplication.processEvents()
        win._apply_default_splitter_sizes()
        for _ in range(3):
            QApplication.processEvents()
        pane = win._right_splitter.widget(0)
        split = win._right_splitter.sizes()[0]
        cut = []
        for group in win._control_panel.findChildren(QGroupBox):
            if not group.isVisibleTo(win._control_panel):
                continue
            top = group.mapTo(pane, QPoint(0, 0)).y()
            bottom = top + group.height()
            if top < split < bottom:
                cut.append(group.title())
        return split, cut

    def test_common_window_sizes(self):
        for width, height in ((1024, 640), (1280, 720), (1400, 900), (1920, 1080)):
            with self.subTest(size=(width, height)):
                split, cut = self._sections_straddling_split(width, height)
                self.assertEqual(cut, [], f"split at {split}px cuts {cut}")


if __name__ == "__main__":
    unittest.main()
