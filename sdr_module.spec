# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SDR Module.

This creates a standalone portable executable for the SDR Module application.
Can be run from a USB drive without installation.

Usage:
    pyinstaller sdr_module.spec           # CLI only
    pyinstaller sdr_module.spec --gui     # With GUI support

Requirements:
    pip install pyinstaller
    pip install PyQt6  # Optional, for GUI
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Check if GUI should be included
INCLUDE_GUI = '--gui' in sys.argv or os.environ.get('SDR_BUILD_GUI', '0') == '1'

# Project paths
PROJECT_ROOT = Path(SPECPATH)
SRC_PATH = PROJECT_ROOT / 'src'

# Hidden imports for core functionality
hidden_imports = [
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'scipy.ndimage',
    'sdr_module',
    'sdr_module.core',
    'sdr_module.core.sample_buffer',
    'sdr_module.core.device_manager',
    'sdr_module.core.dual_sdr',
    'sdr_module.dsp',
    'sdr_module.dsp.spectrum',
    'sdr_module.dsp.filters',
    'sdr_module.dsp.demodulators',
    'sdr_module.dsp.classifiers',
    'sdr_module.dsp.frequency_lock',
    'sdr_module.dsp.afc',
    'sdr_module.dsp.scanner',
    'sdr_module.dsp.protocols',
    'sdr_module.dsp.recording',
    'sdr_module.plugins',
    'sdr_module.plugins.base',
    'sdr_module.plugins.manager',
    'sdr_module.plugins.registry',
    'sdr_module.ui',
    'sdr_module.ui.waterfall',
    'sdr_module.ui.constellation',
    'sdr_module.ui.time_domain',
    'sdr_module.ui.signal_meter',
    'sdr_module.ui.packet_display',
    'sdr_module.utils',
    'sdr_module.utils.conversions',
]

# Excludes
excludes = ['tkinter']

# Add GUI imports if building with GUI
if INCLUDE_GUI:
    hidden_imports.extend([
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'sdr_module.gui',
        'sdr_module.gui.app',
        'sdr_module.gui.main_window',
        'sdr_module.gui.spectrum_widget',
        'sdr_module.gui.waterfall_widget',
        'sdr_module.gui.control_panel',
        'sdr_module.gui.decoder_panel',
        'sdr_module.gui.device_dialog',
    ])
else:
    excludes.extend(['PyQt5', 'PyQt6', 'PySide2', 'PySide6'])

# Collect all Python files from the sdr_module package
a = Analysis(
    [str(SRC_PATH / 'sdr_module' / 'cli.py')],
    pathex=[str(SRC_PATH)],
    binaries=[],
    datas=[
        # Include example plugins for portability
        (str(PROJECT_ROOT / 'examples' / 'plugins'), 'plugins'),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Main CLI executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sdr-scan',
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
    icon=None,  # Add icon path if available: 'assets/icon.ico'
)

# Collect all files into portable folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sdr-module',
)
