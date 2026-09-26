"""
SDR Application entry point.

Provides the main application class and initialization.
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import __version__
from .settings_store import DEFAULT_FREQUENCY_HZ
from .themes import apply_theme

logger = logging.getLogger(__name__)

# Defaults the GUI launchers pass when no value is given on the command line.
_LAUNCH_DEFAULTS: Dict[str, float] = {
    "frequency": DEFAULT_FREQUENCY_HZ,
    "gain": 20.0,
    "sample_rate": 2.4e6,
}
_LAUNCH_FLAGS: Dict[str, Tuple[str, str]] = {
    "frequency": ("-f", "--frequency"),
    "gain": ("-g", "--gain"),
    "sample_rate": ("-s", "--sample-rate"),
}


# Normalized (x, y) points of the spectrum trace drawn on the app icon.
_ICON_TRACE = (
    (0.14, 0.72),
    (0.27, 0.70),
    (0.35, 0.52),
    (0.41, 0.68),
    (0.50, 0.24),
    (0.59, 0.68),
    (0.65, 0.56),
    (0.73, 0.70),
    (0.86, 0.72),
)


def make_app_icon() -> Any:
    """The window/taskbar icon: a spectrum peak on an accent tile.

    Drawn from the active theme palette, so no image file is needed.
    """
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import QIcon, QPainter, QPainterPath, QPen, QPixmap

    from .themes import get_palette

    palette = get_palette()
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inset = size * 0.04
        tile = QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(palette.qcolor("accent"))
        painter.drawRoundedRect(tile, size * 0.22, size * 0.22)

        path = QPainterPath()
        for i, (x, y) in enumerate(_ICON_TRACE):
            point = QPointF(x * size, y * size)
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        pen = QPen(palette.qcolor("on_accent"))
        pen.setWidthF(max(1.5, size * 0.06))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


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
        # Last unhandled error: (signature, monotonic time), to log a
        # repeating one (e.g. in the 30 Hz display timer) only now and then.
        self._last_error: Optional[Tuple[Any, float]] = None

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
                - sample_rate: Sample rate in Hz, used by the demo device and
                  preselected in Device > Connect (hardware is opened there)
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
        previous_hook = sys.excepthook
        # PyQt6 aborts the process on an exception escaping a slot, virtual
        # or thread unless a hook is installed: log it and carry on instead.
        sys.excepthook = self._handle_exception

        try:
            return self._run(settings)
        finally:
            sys.excepthook = previous_hook

    def _handle_exception(self, exc_type, exc, tb) -> None:
        """sys.excepthook while the GUI runs.

        An unhandled error inside Qt (a slot, paint event or worker thread)
        is logged, so it shows in Tools > Error History, and reported in the
        status bar, instead of aborting the app and losing unsaved work.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            if self._app is not None:
                self._app.quit()
            return
        frame = tb
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        where = (
            (frame.tb_frame.f_code.co_filename, frame.tb_lineno)
            if frame is not None
            else None
        )
        signature = (exc_type, where)
        now = time.monotonic()
        last = self._last_error
        if last is not None and last[0] == signature and now - last[1] < 5.0:
            logger.debug("Unhandled error again: %s", exc)
            return
        self._last_error = (signature, now)
        logger.error(
            "Unhandled error: %s: %s",
            exc_type.__name__,
            exc,
            exc_info=(exc_type, exc, tb),
        )
        # Widgets may only be touched from the GUI thread; an error in a
        # worker thread is logged only.
        if not self._in_gui_thread():
            return
        window = self._main_window
        report = getattr(window, "_show_status_error", None)
        if callable(report):
            try:
                report(
                    f"Internal error: {exc_type.__name__}: {exc}. Details are in "
                    "Tools > Error History.",
                    10000,
                )
            except Exception as e:  # pragma: no cover - window already gone
                logger.debug(f"Could not report the error in the window: {e}")

    def _in_gui_thread(self) -> bool:
        if self._app is None:
            return False
        try:
            from PyQt6.QtCore import QThread

            return QThread.currentThread() is self._app.thread()
        except Exception:  # pragma: no cover - defensive
            return False

    def _run(self, settings: Dict[str, Any]) -> int:
        """Create the application and main window and run the event loop."""
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
            apply_theme(self._app, theme)
            try:
                self._app.setWindowIcon(make_app_icon())
            except Exception as e:  # pragma: no cover - cosmetic only
                logger.debug(f"App icon not set: {e}")

            # Install error-history handler so the Error History dialog works
            try:
                from .error_log_dialog import install_history_handler

                install_history_handler()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Error-history handler not installed: {e}")

            # Create and show main window
            demo_mode = settings.get("demo_mode", False)
            self._main_window = SDRMainWindow(demo_mode=demo_mode)

            # Apply command-line values, but don't let a launcher default
            # override the frequency/gain restored from the last session (or
            # chosen in the first-run wizard).
            if self._should_apply(settings, "frequency"):
                self._main_window.set_frequency(float(settings["frequency"]))
            if self._should_apply(settings, "gain"):
                self._main_window.set_gain(float(settings["gain"]))
            if self._should_apply(settings, "sample_rate"):
                self._main_window.set_sample_rate(float(settings["sample_rate"]))

            self._main_window.show()

            logger.info("SDR Module GUI started")
            if demo_mode:
                logger.info("Running in demo mode")

            # Run event loop
            return self._app.exec()

        except Exception as e:
            logger.exception(f"Application error: {e}")
            self._show_fatal_error(e)
            return 1

    def _should_apply(self, settings: Dict[str, Any], key: str) -> bool:
        """Whether a launch setting should override the persisted value.

        ``python -m sdr_module.gui`` and ``sdr-scan gui`` always pass their
        argparse defaults (``_LAUNCH_DEFAULTS``: 100.1 MHz, 20 dB, 2.4 MS/s).
        A value that differs from the default was clearly requested; one
        equal to it only counts if its flag was actually typed on the command
        line.
        """
        value = settings.get(key)
        if value is None or key not in _LAUNCH_DEFAULTS:
            return False
        try:
            if float(value) != _LAUNCH_DEFAULTS[key]:
                return True
        except (TypeError, ValueError):
            return False
        short, long_ = _LAUNCH_FLAGS[key]
        for arg in self._args[1:]:
            name = str(arg).split("=", 1)[0]
            if name == short or (arg.startswith(short) and not arg.startswith("--")):
                return True  # -f 144e6 or -f144e6
            if name.startswith("--") and len(name) > 3 and long_.startswith(name):
                return True  # --frequency, --frequency=..., or a prefix (--freq)
        return False

    def _show_fatal_error(self, error: Exception) -> None:
        """Tell the user why the GUI could not start (not only the log)."""
        if self._app is None:
            return
        try:
            from PyQt6.QtWidgets import QMessageBox

            detail = str(error) or type(error).__name__
            QMessageBox.critical(
                None,
                "SDR Module Could Not Start",
                f"SDR Module could not start:\n\n{detail}\n\n"
                "The full error report is in the terminal output.",
            )
        except Exception as exc:  # pragma: no cover - best effort only
            logger.debug(f"Could not show the startup error dialog: {exc}")

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
