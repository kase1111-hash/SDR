"""
The right column's tool tabs: give them the whole column on demand.

The tool tabs (Decoder, Bookmarks, Ham, Scanner, Signal, Info) share
the right column with the control panel, which leaves them 200-330 px of
height on a laptop screen: a few rows of decoded messages, a third of the
QRP calculator. :class:`ToolTabsMaximizer` lets them take the whole column
(the control panel steps aside) and back:

* the arrow button in the tab bar's corner,
* a double-click on a tab (of the tool tabs or of the Ham group),
* View > Maximize Tool Tabs.

The choice is remembered (``layout/tools_maximized``). Anything that needs
the control panel (Radio > Enter Frequency, Ctrl+L) brings it back first
(:meth:`ToolTabsMaximizer.restore`).
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from PyQt6.QtCore import QObject, QPointF, Qt
    from PyQt6.QtGui import QIcon, QPainter, QPen, QPixmap
    from PyQt6.QtWidgets import QToolButton

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - no Qt
    HAS_PYQT6 = False

from .themes import get_palette, theme_notifier

#: Settings key: whether the tool tabs fill the right column.
KEY_MAXIMIZED = "layout/tools_maximized"

MAXIMIZE_TIP = (
    "Give the tool tabs the whole right column: the control panel steps "
    "aside (double-clicking a tab does the same; View > Maximize Tool Tabs)"
)
RESTORE_TIP = "Bring the control panel back above the tool tabs (or double-click a tab)"


def chevron_icon(up: bool) -> "QIcon":
    """A chevron pointing up (maximize: the tabs move up over the control
    panel) or down (restore), in the theme's text color."""
    color = get_palette().qcolor("text")
    icon = QIcon()
    for size in (16, 32):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        k = size / 16.0
        pen = QPen(color)
        pen.setWidthF(1.6 * k)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        # Two stacked chevrons: "all the way up" / "back down".
        for y in (5.0, 9.5):
            if up:
                points = [(4.0, y + 3.0), (8.0, y - 1.0), (12.0, y + 3.0)]
            else:
                points = [(4.0, y - 1.5), (8.0, y + 2.5), (12.0, y - 1.5)]
            for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
                painter.drawLine(QPointF(x0 * k, y0 * k), QPointF(x1 * k, y1 * k))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class ToolTabsMaximizer(QObject if HAS_PYQT6 else object):
    """Lets the window's tool tabs fill the right column (see the module
    docstring). ``window`` is the main window: its ``_right_splitter``
    (control pane over the tool tabs), ``_right_tabs`` and ``_settings``."""

    def __init__(self, window: Any):
        super().__init__(window)
        self._win = window
        self._maximized = False
        self.action: Optional[Any] = None  # View > Maximize Tool Tabs
        button = QToolButton()
        button.setObjectName("toolTabsMaximize")
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName("Maximize tool tabs")
        button.toggled.connect(self.set_maximized)
        self.button = button
        tabs = window._right_tabs
        tabs.setCornerWidget(button, Qt.Corner.TopRightCorner)
        tabs.tabBar().tabBarDoubleClicked.connect(self._double_clicked)
        ham = getattr(window, "_ham_tabs", None)
        if ham is not None:
            ham.tabBar().tabBarDoubleClicked.connect(self._double_clicked)
        theme_notifier().theme_changed.connect(self._on_theme_changed)
        self._sync()

    # -- state ---------------------------------------------------------------
    @property
    def maximized(self) -> bool:
        return self._maximized

    def set_action(self, action: Any) -> None:
        """The View menu's checkable action that mirrors the button."""
        self.action = action
        action.toggled.connect(self.set_maximized)
        self._sync()

    def set_maximized(self, on: bool, announce: bool = True) -> None:
        """Give the tool tabs the whole column (``on``) or share it with the
        control panel again; remembered."""
        on = bool(on)
        if on == self._maximized:
            self._sync()
            return
        self._maximized = on
        pane = self._win._right_splitter.widget(0)
        if pane is not None:
            pane.setVisible(not on)
        self._sync()
        settings = getattr(self._win, "_settings", None)
        if settings is not None and not getattr(self._win, "_restoring", False):
            settings.set(KEY_MAXIMIZED, on)
        if announce:
            text = (
                "The tool tabs fill the column. Double-click a tab (or the "
                "arrows in the tab bar) to bring the control panel back."
                if on
                else "Control panel back above the tool tabs"
            )
            self._win._show_status_message(text, None, 4000)

    def restore(self) -> None:
        """Bring the control panel back (something needs it, e.g. Ctrl+L)."""
        if self._maximized:
            self.set_maximized(False, announce=False)

    def restore_saved(self) -> None:
        """Apply the remembered choice (at start-up)."""
        settings = getattr(self._win, "_settings", None)
        if settings is not None and settings.get_bool(KEY_MAXIMIZED, False):
            self.set_maximized(True, announce=False)

    # -- internals ---------------------------------------------------------------
    def _double_clicked(self, index: int) -> None:
        if index >= 0:
            self.set_maximized(not self._maximized)

    def _on_theme_changed(self, _name: str) -> None:
        try:
            self._sync()
        except RuntimeError:  # pragma: no cover - window already destroyed
            pass

    def _sync(self) -> None:
        on = self._maximized
        for control in (self.button, self.action):
            if control is not None and control.isChecked() != on:
                control.blockSignals(True)
                control.setChecked(on)
                control.blockSignals(False)
        self.button.setIcon(chevron_icon(up=not on))
        tip = RESTORE_TIP if on else MAXIMIZE_TIP
        self.button.setToolTip(tip)
        self.button.setAccessibleName(
            "Restore the control panel" if on else "Maximize tool tabs"
        )
