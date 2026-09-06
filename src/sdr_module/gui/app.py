"""
SDR Application entry point.

Provides the main application class and initialization.
"""

import logging
import sys
from typing import Any, Dict, List, Optional

from .. import __version__
from .themes import get_stylesheet

logger = logging.getLogger(__name__)


def check_pyqt6() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: F401

        return True
    except ImportError:
        return False


_DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 8px;
    padding: 12px 6px 6px 6px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 6px;
    padding: 4px;
}
QToolBar QLabel {
    background: transparent;
    color: #a6adc8;
    padding: 0 2px;
}
QToolBar QToolButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
}
QToolBar QToolButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}
QToolBar QToolButton:pressed {
    background-color: #585b70;
}
QToolBar QToolButton:checked {
    background-color: #f38ba8;
    color: #1e1e2e;
    border-color: #f38ba8;
}
QStatusBar {
    background-color: #181825;
    border-top: 1px solid #313244;
    color: #a6adc8;
    font-size: 11px;
}
QStatusBar QLabel {
    background: transparent;
    color: #a6adc8;
    padding: 0 6px;
}
QStatusBar::item {
    border: none;
}
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected {
    background-color: #313244;
    border-radius: 4px;
}
QMenu {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #313244;
    border-radius: 3px;
}
QMenu::separator {
    height: 1px;
    background-color: #313244;
    margin: 4px 8px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 14px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:checked {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 8px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #585b70;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #313244;
}
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 20px;
}
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #585b70;
}
QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #89b4fa;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background-color: #b4d0fb;
}
QSlider::sub-page:horizontal {
    background-color: #45475a;
    border-radius: 3px;
}
QCheckBox {
    spacing: 6px;
    color: #cdd6f4;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #45475a;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background-color: #1e1e2e;
    top: -1px;
}
QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #313244;
}
QTableWidget {
    background-color: #181825;
    color: #cdd6f4;
    gridline-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #313244;
}
QTableWidget::item:alternate {
    background-color: #1e1e2e;
}
QHeaderView::section {
    background-color: #181825;
    color: #a6adc8;
    border: none;
    border-bottom: 1px solid #45475a;
    border-right: 1px solid #313244;
    padding: 4px 8px;
    font-weight: bold;
}
QTextEdit {
    background-color: #181825;
    color: #a6e3a1;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #313244;
}
QProgressBar {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QSplitter::handle {
    background-color: #313244;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QScrollBar:vertical {
    background-color: #181825;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background-color: #181825;
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #585b70;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""


class SDRApplication:
    """
    Main SDR application.

    Handles application initialization, event loop, and cleanup.

    Usage:
        app = SDRApplication()
        app.run()
    """

    def __init__(self, args: Optional[List[str]] = None):
        """
        Initialize the SDR application.

        Args:
            args: Command line arguments (uses sys.argv if None)
        """
        self._args = args if args is not None else sys.argv
        self._app = None
        self._main_window = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def is_available(self) -> bool:
        """Check if PyQt6 is available."""
        return check_pyqt6()

    def run(self, settings: Optional[Dict[str, Any]] = None) -> int:
        """
        Run the application.

        Args:
            settings: Optional settings dictionary with keys:
                - frequency: Initial frequency in Hz
                - sample_rate: Sample rate in Hz
                - gain: RF gain in dB
                - demo_mode: Run in demo mode

        Returns:
            Exit code (0 for success)
        """
        if not check_pyqt6():
            logger.error("PyQt6 is required but not installed.")
            logger.error('Install it with: python -m pip install "sdr-module[gui]"')
            print(
                'Error: PyQt6 is required. Install it with: python -m pip install "sdr-module[gui]"'
            )
            return 1

        settings = settings or {}

        try:
            from PyQt6.QtWidgets import QApplication

            from .main_window import SDRMainWindow

            # Create application
            self._app = QApplication(self._args)
            self._app.setApplicationName("SDR Module")
            self._app.setApplicationVersion(__version__)
            self._app.setOrganizationName("SDR Module Team")

            # Set application style
            self._app.setStyle("Fusion")

            # Apply theme (persisted user preference, default dark)
            try:
                from .settings_store import GuiSettings

                theme = GuiSettings().get_str("theme", "dark")
            except Exception:
                theme = "dark"
            self._app.setStyleSheet(get_stylesheet(theme))

            # Install error-history handler so the Error History dialog works
            try:
                from .error_log_dialog import install_history_handler

                install_history_handler()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Error-history handler not installed: {e}")

            # Create and show main window
            demo_mode = settings.get("demo_mode", False)
            self._main_window = SDRMainWindow(demo_mode=demo_mode)

            # Apply initial settings
            if "frequency" in settings:
                self._main_window.set_frequency(settings["frequency"])
            if "gain" in settings:
                self._main_window.set_gain(settings["gain"])

            self._main_window.show()

            logger.info("SDR Module GUI started")
            if demo_mode:
                logger.info("Running in demo mode")

            # Run event loop
            return self._app.exec()

        except Exception as e:
            logger.exception(f"Application error: {e}")
            return 1

    def quit(self) -> None:
        """Quit the application."""
        if self._app:
            self._app.quit()


def main():
    """Main entry point for command line."""
    app = SDRApplication()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
