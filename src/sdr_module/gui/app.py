"""
SDR Application entry point.

Provides the main application class and initialization.
"""

import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def check_pyqt6() -> bool:
    """Check if PyQt6 is available."""
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: F401

        return True
    except ImportError:
        return False


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
            logger.error("Install with: pip install PyQt6")
            print("Error: PyQt6 is required. Install with: pip install PyQt6")
            return 1

        settings = settings or {}

        try:
            from PyQt6.QtWidgets import QApplication

            from .main_window import SDRMainWindow

            # Create application
            self._app = QApplication(self._args)
            self._app.setApplicationName("SDR Module")
            self._app.setApplicationVersion("0.1.0")
            self._app.setOrganizationName("SDR Module Team")

            # Set application style
            self._app.setStyle("Fusion")

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
