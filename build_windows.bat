@echo off
REM ============================================================================
REM SDR Module Windows Build Script
REM ============================================================================
REM Builds a standalone Windows folder (dist\sdr-module\sdr-scan.exe) with
REM PyInstaller. The GUI is included by default.
REM
REM Prerequisites:
REM   - Python 3.10 or higher on PATH
REM
REM Usage:
REM   build_windows.bat [options]
REM
REM Options:
REM   --clean     Clean build directories before building
REM   --no-gui    Build the command-line tool only (no PyQt6)
REM   --no-upx    Disable UPX compression (faster build, larger exe)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo   SDR Module Windows Build Script
echo ============================================
echo.

set CLEAN=0
set NO_GUI=0
set NO_UPX=0

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--clean" set CLEAN=1
if /i "%~1"=="--no-gui" set NO_GUI=1
if /i "%~1"=="--no-upx" set NO_UPX=1
shift
goto :parse_args
:end_parse

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    exit /b 1
)
python --version
echo.

REM Always call pip through the interpreter so both agree on the environment.
echo [2/5] Checking pip installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available for this Python
    exit /b 1
)
python -m pip --version
echo.

if %CLEAN%==1 (
    echo [3/5] Cleaning build directories...
    if exist "build" rmdir /s /q build
    if exist "dist" rmdir /s /q dist
    echo Cleaned.
    echo.
) else (
    echo [3/5] Skipping clean (use --clean to enable)
    echo.
)

echo [4/5] Installing the package and build dependencies...
python -m pip install --upgrade pip
if %NO_GUI%==1 (
    python -m pip install -e . pyinstaller
    set SDR_BUILD_GUI=0
) else (
    python -m pip install -e ".[gui]" pyinstaller
    set SDR_BUILD_GUI=1
)
if errorlevel 1 (
    echo ERROR: dependency installation failed
    exit /b 1
)
echo.

for /f "delims=" %%v in ('python tools\get_version.py') do set APPVER=%%v
echo [5/5] Building sdr-module %APPVER% (GUI included: %SDR_BUILD_GUI%)...
echo.

if %NO_UPX%==1 (
    python -m PyInstaller --noconfirm --clean sdr_module.spec --upx-dir=""
) else (
    python -m PyInstaller --noconfirm --clean sdr_module.spec
)

if errorlevel 1 (
    echo.
    echo ============================================
    echo   BUILD FAILED
    echo ============================================
    exit /b 1
)

REM Record the version for installer.iss (read via #include).
echo #define MyAppVersion "%APPVER%"> installer_version.iss

echo.
echo ============================================
echo   BUILD SUCCESSFUL
echo ============================================
echo.
echo Executable location: dist\sdr-module\sdr-scan.exe
echo.
echo To run:
echo   cd dist\sdr-module
echo   sdr-scan.exe gui --demo
echo.
echo To create an installer, run:
echo   build_installer.bat
echo.

endlocal
