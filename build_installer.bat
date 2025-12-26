@echo off
REM ============================================================================
REM SDR Module Windows Installer Build Script
REM ============================================================================
REM This script creates a Windows installer using Inno Setup.
REM
REM Prerequisites:
REM   - Inno Setup 6.x installed (https://jrsoftware.org/isinfo.php)
REM   - Built executable in dist\sdr-module\ directory (run build_windows.bat first)
REM
REM Usage:
REM   build_installer.bat
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo   SDR Module Installer Build Script
echo ============================================
echo.

REM Check if PyInstaller build exists
if not exist "dist\sdr-module\sdr-scan.exe" (
    echo ERROR: Executable not found at dist\sdr-module\sdr-scan.exe
    echo.
    echo Please run build_windows.bat first to create the executable.
    echo.
    exit /b 1
)

echo [1/3] Checking for Inno Setup...

REM Try to find Inno Setup compiler
set ISCC=""

REM Check common installation paths
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
)

REM Check if ISCC is in PATH
where iscc >nul 2>&1
if not errorlevel 1 (
    set ISCC=iscc
)

if %ISCC%=="" (
    echo ERROR: Inno Setup not found
    echo.
    echo Please install Inno Setup 6 from:
    echo   https://jrsoftware.org/isdl.php
    echo.
    echo Or add ISCC.exe to your PATH.
    exit /b 1
)

echo Found Inno Setup: %ISCC%
echo.

REM Create output directory
echo [2/3] Creating installer output directory...
if not exist "installer_output" mkdir installer_output
echo.

REM Build the installer
echo [3/3] Building installer...
echo.

%ISCC% installer.iss

if errorlevel 1 (
    echo.
    echo ============================================
    echo   INSTALLER BUILD FAILED
    echo ============================================
    echo.
    echo Check the error messages above for details.
    exit /b 1
)

echo.
echo ============================================
echo   INSTALLER BUILD SUCCESSFUL
echo ============================================
echo.
echo Installer location: installer_output\SDR-Module-0.1.0-Setup.exe
echo.
echo To install:
echo   Run the installer and follow the prompts.
echo.

endlocal
