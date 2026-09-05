# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SDR Module.

Builds a self-contained, portable folder (dist/sdr-module/) with a
``sdr-scan`` executable that can run from a USB drive without a Python
installation.

Usage:
    pyinstaller sdr_module.spec                       # CLI + GUI (default)
    SDR_BUILD_GUI=0 pyinstaller sdr_module.spec       # CLI only, no PyQt6

Requirements:
    python -m pip install -e ".[gui]" pyinstaller     # or "." for CLI only

The build scripts (build_portable.sh, build_windows.bat, build_windows.ps1)
wrap this file and set SDR_BUILD_GUI for you.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# Include the PyQt6 GUI unless explicitly disabled.
INCLUDE_GUI = os.environ.get("SDR_BUILD_GUI", "1") != "0"

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821 - provided by PyInstaller
SRC_PATH = PROJECT_ROOT / "src"


def _wanted(module_name: str) -> bool:
    """Drop the Qt-dependent subpackages when building without the GUI."""
    if INCLUDE_GUI:
        return True
    return ".gui" not in module_name


# Every sdr_module submodule is imported lazily somewhere (decoders, ham
# features, GUI panels), so collect them all rather than maintaining a list.
hidden_imports = collect_submodules("sdr_module", filter=_wanted)

# Optional DSP acceleration: bundle SciPy only when it is installed.
try:
    import scipy  # noqa: F401

    hidden_imports += ["scipy.signal", "scipy.fft"]
except ImportError:
    pass

excludes = ["tkinter", "matplotlib"]

if INCLUDE_GUI:
    hidden_imports += [
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtMultimedia",
    ]
else:
    excludes += [
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "sdr_module.gui",
        "sdr_module.ham.gui",
    ]

datas = [
    (str(PROJECT_ROOT / "README.md"), "."),
    (str(PROJECT_ROOT / "LICENSE.md"), "."),
    (str(PROJECT_ROOT / "CHANGELOG.md"), "."),
]

# The entry script must live outside the package: running a package module
# directly breaks its relative imports ("attempted relative import with no
# known parent package").
a = Analysis(  # noqa: F821 - provided by PyInstaller
    [str(PROJECT_ROOT / "tools" / "pyinstaller_entry.py")],
    pathex=[str(SRC_PATH)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821 - provided by PyInstaller

# Console executable: `sdr-scan --help` must work from a terminal, and
# `sdr-scan gui` launches the Qt window from the same binary.
exe = EXE(  # noqa: F821 - provided by PyInstaller
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sdr-scan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add an icon path here if one is added to the repo.
)

coll = COLLECT(  # noqa: F821 - provided by PyInstaller
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="sdr-module",
)
