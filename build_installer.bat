@echo off
REM ============================================================================
REM SDR Module Windows Installer Build Script
REM ============================================================================
REM Creates installer_output\SDR-Module-<version>-Setup.exe with Inno Setup.
REM
REM Prerequisites:
REM   - Inno Setup 6.x installed (https://jrsoftware.org/isinfo.php)
REM   - Built executable in dist\sdr-module\ (run build_windows.bat first)
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

if not exist "dist\sdr-module\sdr-scan.exe" (
    echo ERROR: Executable not found at dist\sdr-module\sdr-scan.exe
    echo.
    echo Please run build_windows.bat first to create the executable.
    exit /b 1
)

echo [1/3] Checking for Inno Setup...

set ISCC=""
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
)
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

echo [2/3] Reading version...
for /f "delims=" %%v in ('python tools\get_version.py') do set APPVER=%%v
if "%APPVER%"=="" (
    echo ERROR: could not read the version from src\sdr_module\__init__.py
    exit /b 1
)
echo Version: %APPVER%
if not exist "installer_output" mkdir installer_output
echo.

echo [3/3] Building installer...
echo.

%ISCC% /DMyAppVersion=%APPVER% installer.iss

if errorlevel 1 (
    echo.
    echo ============================================
    echo   INSTALLER BUILD FAILED
    echo ============================================
    exit /b 1
)

echo.
echo ============================================
echo   INSTALLER BUILD SUCCESSFUL
echo ============================================
echo.
echo Installer location: installer_output\SDR-Module-%APPVER%-Setup.exe
echo.

endlocal
