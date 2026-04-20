"""
Help dialog — keyboard shortcuts reference.
"""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QTextBrowser,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


_HELP_HTML = """
<h2>SDR Module — Keyboard Shortcuts</h2>

<h3>Acquisition</h3>
<table>
<tr><td><b>Space</b></td><td>Start / Stop acquisition</td></tr>
<tr><td><b>Ctrl+Shift+R</b></td><td>Toggle recording</td></tr>
</table>

<h3>Tuning</h3>
<table>
<tr><td><b>Left / Right</b></td><td>Tune by 10 kHz</td></tr>
<tr><td><b>Shift+Left / Right</b></td><td>Tune by 100 kHz</td></tr>
<tr><td><b>Ctrl+Left / Right</b></td><td>Tune by 1 MHz</td></tr>
<tr><td><b>Click on spectrum/waterfall</b></td><td>Tune to clicked frequency</td></tr>
</table>

<h3>File</h3>
<table>
<tr><td><b>Ctrl+O</b></td><td>Open recording</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save recording</td></tr>
<tr><td><b>Ctrl+P</b></td><td>Save screenshot (waterfall + spectrum)</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>Quit</td></tr>
</table>

<h3>Tools</h3>
<table>
<tr><td><b>Ctrl+R</b></td><td>AM/FM Radio Tuner (if ham module installed)</td></tr>
<tr><td><b>Ctrl+F</b></td><td>Frequency scanner</td></tr>
<tr><td><b>Ctrl+B</b></td><td>Add current frequency to bookmarks</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Show error history</td></tr>
</table>

<h3>View</h3>
<table>
<tr><td><b>Ctrl+T</b></td><td>Toggle light/dark theme</td></tr>
<tr><td><b>F1</b></td><td>This help dialog</td></tr>
</table>

<p><i>Settings (theme, bookmarks, frequency, gain) persist between launches.</i></p>
"""


class HelpDialog(QDialog if HAS_PYQT6 else object):
    """Shortcut reference dialog."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Help — Shortcuts")
        self.resize(500, 520)

        layout = QVBoxLayout(self)
        viewer = QTextBrowser()
        viewer.setHtml(_HELP_HTML)
        layout.addWidget(viewer)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
