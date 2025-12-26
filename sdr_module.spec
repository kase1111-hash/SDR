# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SDR Module.

This creates a standalone Windows executable for the SDR Module application.

Usage:
    pyinstaller sdr_module.spec

Requirements:
    pip install pyinstaller
"""

import sys
from pathlib import Path

block_cipher = None

# Project paths
PROJECT_ROOT = Path(SPECPATH)
SRC_PATH = PROJECT_ROOT / 'src'

# Collect all Python files from the sdr_module package
a = Analysis(
    [str(SRC_PATH / 'sdr_module' / 'cli.py')],
    pathex=[str(SRC_PATH)],
    binaries=[],
    datas=[
        # Include any data files if needed
        # (str(PROJECT_ROOT / 'data'), 'data'),
    ],
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.fft',
        'scipy.ndimage',
        'matplotlib',
        'matplotlib.pyplot',
        'sdr_module',
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
        'sdr_module.ui',
        'sdr_module.ui.waterfall',
        'sdr_module.ui.constellation',
        'sdr_module.ui.time_domain',
        'sdr_module.ui.signal_meter',
        'sdr_module.ui.packet_display',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
