"""
Version information, the project's web pages and the About box.

For bug reports: Help > About and Tools > Error History have **Copy Version
Info** (the app, Python, Qt, NumPy, the system, the SDR drivers installed,
and the device in use); Help > Report a Problem opens the issue tracker
with it filled in. Help > Online Documentation opens the README.

The URLs are the project's (``pyproject.toml`` [project.urls], the README's
links); the installed package's metadata is read when it has them.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
from typing import Any, Dict, Optional
from urllib.parse import urlencode

try:
    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, Qt, QUrl
    from PyQt6.QtGui import QDesktopServices, QGuiApplication
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - no Qt
    HAS_PYQT6 = False

from .. import __version__

if HAS_PYQT6:
    from .themes import set_role

logger = logging.getLogger(__name__)

#: The project's pages (pyproject.toml [project.urls]; the README links them).
REPOSITORY_URL = "https://github.com/kase1111-hash/SDR"
DOCS_URL = f"{REPOSITORY_URL}#readme"
ISSUES_URL = f"{REPOSITORY_URL}/issues"
NEW_ISSUE_URL = f"{ISSUES_URL}/new"

# The SDR drivers (label, Python package).
_DRIVERS = (("RTL-SDR", "rtlsdr"), ("HackRF One", "python_hackrf"))


def project_urls() -> Dict[str, str]:
    """``{"Homepage": ..., "Issues": ...}`` from the installed package's
    metadata, else the built-in ones."""
    urls = {
        "Homepage": REPOSITORY_URL,
        "Repository": REPOSITORY_URL,
        "Issues": ISSUES_URL,
    }
    try:
        from importlib.metadata import metadata

        for entry in metadata("sdr-module").get_all("Project-URL") or []:
            name, _sep, url = str(entry).partition(",")
            if url.strip().startswith("https://"):
                urls[name.strip()] = url.strip()
    except Exception as e:  # not installed, or no metadata: the built-in ones
        logger.debug(f"No project URLs in the package metadata: {e}")
    return urls


def _package_version(module: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, packages_distributions
        from importlib.metadata import version as dist_version

        dists = packages_distributions().get(module) or [module]
        for dist in dists:
            try:
                return dist_version(dist)
            except PackageNotFoundError:
                continue
    except Exception as e:  # pragma: no cover - old importlib.metadata
        logger.debug(f"No version for {module}: {e}")
    return "installed"


def driver_lines() -> Dict[str, str]:
    """``{"RTL-SDR": "pyrtlsdr 0.3.0", "HackRF One": "not installed"}``."""
    lines = {}
    for label, module in _DRIVERS:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        lines[label] = (
            f"{module} {_package_version(module)}" if found else "not installed"
        )
    return lines


def version_info(window: Any = None) -> str:
    """Plain text for a bug report: the app, Python, Qt, NumPy, the system,
    the drivers and (with the main ``window``) the device in use."""
    import numpy as np

    lines = [f"SDR Module {__version__}"]
    lines.append(
        f"Python {platform.python_version()} ({platform.python_implementation()})"
    )
    if HAS_PYQT6:
        lines.append(f"Qt {QT_VERSION_STR}, PyQt {PYQT_VERSION_STR}")
    lines.append(f"NumPy {np.__version__}")
    lines.append(f"System: {platform.platform()}")
    for label, text in driver_lines().items():
        lines.append(f"Driver for {label}: {text}")
    if window is not None:
        lines.extend(_window_lines(window))
    return "\n".join(lines)


def _window_lines(window: Any) -> list:
    lines = []
    name = getattr(window, "_device_display_name", None)
    rate = getattr(window, "_device_sample_rate", None)
    if callable(name):
        device = str(name())
        if getattr(window, "__dict__", {}).get("_device") is not None and callable(
            rate
        ):
            device += f" at {float(rate()) / 1e6:g} MS/s"
        lines.append(f"Device: {device}")
    audio = getattr(window, "__dict__", {}).get("_audio")
    output = getattr(audio, "output_name", None)
    if callable(output):
        problem = getattr(window, "__dict__", {}).get("_audio_problem")
        lines.append(f"Audio output: {output()}" + (f" ({problem})" if problem else ""))
    theme = getattr(window, "__dict__", {}).get("_theme")
    if theme:
        lines.append(f"Theme: {theme}")
    return lines


def report_url(info: str) -> str:
    """The issue tracker's "new issue" page with ``info`` in its body."""
    body = (
        "What happened, and what did you expect?\n\n\n"
        "Steps to reproduce:\n1. \n2. \n\n"
        "Version info (Help > About > Copy Version Info):\n```\n"
        f"{info}\n```\n"
    )
    issues = project_urls().get("Issues", ISSUES_URL).rstrip("/")
    return f"{issues}/new?{urlencode({'body': body})}"


def copy_version_info(window: Any = None) -> str:
    """Put :func:`version_info` on the clipboard; returns it."""
    text = version_info(window)
    if HAS_PYQT6:
        QGuiApplication.clipboard().setText(text)
    return text


def open_url(url: str) -> bool:
    """Open ``url`` in the browser (False when the system can't)."""
    if not HAS_PYQT6:
        return False
    return bool(QDesktopServices.openUrl(QUrl(url)))


class AboutDialog(QDialog if HAS_PYQT6 else object):
    """Help > About: what the app is, its links, and the version info with
    Copy Version Info (for a bug report)."""

    def __init__(self, window: Any = None, parent: Optional[Any] = None):
        super().__init__(parent if parent is not None else window)
        self.setWindowTitle("About SDR Module")
        self.setMinimumWidth(480)
        self._window = window
        urls = project_urls()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        title = QLabel(f"SDR Module {__version__}")
        set_role(title, "display")
        layout.addWidget(title)
        text = QLabel(
            "<p>Software-defined radio for signal visualization, frequency "
            "analysis and protocol decoding.</p>"
            "<p><b>Hardware:</b> RTL-SDR and HackRF One, or the built-in demo "
            "device.</p>"
            "<p><b>Features:</b> spectrum and waterfall; AM, FM, SSB and CW; "
            "POCSAG, FLEX, ADS-B, ACARS, AX.25/APRS, RDS and Morse code "
            "decoders; signal identification, scanner and bookmarks; I/Q and "
            "audio recording and playback; a second receiver; ham radio tools "
            "(S-meter, station ID, SSTV, QRP, Transmit Text).</p>"
            f'<p><a href="{urls.get("Homepage", REPOSITORY_URL)}#readme">'
            "Documentation</a> &middot; "
            f'<a href="{urls.get("Issues", ISSUES_URL)}">Report a problem</a>'
            "</p>"
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)
        caption = QLabel("VERSION INFO")
        set_role(caption, "caption")
        layout.addWidget(caption)
        self.info = QPlainTextEdit(version_info(window))
        self.info.setReadOnly(True)
        self.info.setAccessibleName("Version info")
        self.info.setMaximumHeight(150)
        set_role(self.info, "terminal")
        layout.addWidget(self.info)
        self.feedback = QLabel("")
        set_role(self.feedback, "hint")
        layout.addWidget(self.feedback)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.copy_button = QPushButton("&Copy Version Info")
        self.copy_button.setToolTip(
            "Copy the versions, the system and the device in use, to paste "
            "into a bug report"
        )
        self.copy_button.setAutoDefault(False)
        self.copy_button.clicked.connect(self.copy)
        buttons.addButton(self.copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        close = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close is not None:
            close.setDefault(True)
        layout.addWidget(buttons)

    def copy(self) -> None:
        copy_version_info(self._window)
        self.feedback.setText("Copied to the clipboard.")
