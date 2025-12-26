@echo off
REM ============================================================================
REM SDR Module Windows Build Script
REM ============================================================================
REM This script builds a standalone Windows executable for the SDR Module.
REM
REM Prerequisites:
REM   - Python 3.8 or higher
REM   - pip (Python package manager)
REM
REM Usage:
REM   build_windows.bat [options]
REM
REM Options:
REM   --clean     Clean build directories before building
REM   --install   Install the package in development mode first
REM   --no-upx    Disable UPX compression (faster build, larger exe)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo   SDR Module Windows Build Script
echo ============================================
echo.

REM Parse command line arguments
set CLEAN=0
set INSTALL=0
set NO_UPX=0

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--clean" set CLEAN=1
if /i "%~1"=="--install" set INSTALL=1
if /i "%~1"=="--no-upx" set NO_UPX=1
shift
goto :parse_args
:end_parse

REM Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    exit /b 1
)
python --version
echo.

REM Check pip installation
echo [2/6] Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please ensure pip is installed with Python
    exit /b 1
)
pip --version
echo.

REM Clean build directories if requested
if %CLEAN%==1 (
    echo [3/6] Cleaning build directories...
    if exist "build" rmdir /s /q build
    if exist "dist" rmdir /s /q dist
    if exist "*.egg-info" rmdir /s /q *.egg-info
    echo Cleaned.
    echo.
) else (
    echo [3/6] Skipping clean (use --clean to enable)
    echo.
)

REM Install development dependencies
echo [4/6] Installing build dependencies...
pip install --upgrade pip setuptools wheel >nul 2>&1
pip install pyinstaller numpy scipy matplotlib >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some optional dependencies failed to install
    echo Continuing with available dependencies...
)
echo Dependencies installed.
echo.

REM Install package in development mode if requested
if %INSTALL%==1 (
    echo [5/6] Installing SDR Module in development mode...
    pip install -e . >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Development install failed
        echo Continuing anyway...
    )
    echo Package installed.
    echo.
) else (
    echo [5/6] Skipping package install (use --install to enable)
    echo.
)

REM Build the executable
echo [6/6] Building Windows executable...
echo.

if %NO_UPX%==1 (
    echo Building without UPX compression...
    pyinstaller --noconfirm --clean sdr_module.spec --upx-dir=""
) else (
    pyinstaller --noconfirm --clean sdr_module.spec
)

if errorlevel 1 (
    echo.
    echo ============================================
    echo   BUILD FAILED
    echo ============================================
    echo.
    echo Check the error messages above for details.
    exit /b 1
)

echo.
echo ============================================
echo   BUILD SUCCESSFUL
echo ============================================
echo.
echo Executable location: dist\sdr-module\sdr-scan.exe
echo.
echo To run:
echo   cd dist\sdr-module
echo   sdr-scan.exe --help
echo.
echo To create an installer, run:
echo   build_installer.bat
echo.

endlocal
