<#
.SYNOPSIS
    SDR Module Windows Build Script (PowerShell)

.DESCRIPTION
    This script builds a standalone Windows executable for the SDR Module.
    It handles dependency installation, PyInstaller compilation, and optional
    Inno Setup installer creation.

.PARAMETER Clean
    Clean build directories before building.

.PARAMETER NoGUI
    Build the command-line tool only (no PyQt6 in the bundle).

.PARAMETER NoUPX
    Disable UPX compression (faster build, larger executable).

.PARAMETER CreateInstaller
    Also create the Windows installer after building.

.EXAMPLE
    .\build_windows.ps1
    Basic build with default options.

.EXAMPLE
    .\build_windows.ps1 -Clean -CreateInstaller
    Full clean build with installer creation.

.NOTES
    Prerequisites:
    - Python 3.10 or higher
    - pip (Python package manager)
    - Inno Setup 6 (optional, for installer creation)
#>

param(
    [switch]$Clean,
    [switch]$NoGUI,
    [switch]$NoUPX,
    [switch]$CreateInstaller
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SDR Module Windows Build Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Step 2: Check pip
Write-Host "[2/7] Checking pip installation..." -ForegroundColor Yellow
try {
    # Always call pip through the interpreter so both agree on the environment.
    $pipVersion = python -m pip --version 2>&1
    Write-Host "  Found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: pip is not installed" -ForegroundColor Red
    exit 1
}

# Step 3: Clean if requested
Write-Host "[3/7] Cleaning build directories..." -ForegroundColor Yellow
if ($Clean) {
    $dirsToClean = @("build", "dist", "installer_output")
    foreach ($dir in $dirsToClean) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir
            Write-Host "  Removed: $dir" -ForegroundColor Gray
        }
    }
    Get-ChildItem -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force
    Write-Host "  Cleaned." -ForegroundColor Green
} else {
    Write-Host "  Skipped (use -Clean to enable)" -ForegroundColor Gray
}

# Step 4: Install dependencies
Write-Host "[4/7] Installing build dependencies..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip | Out-Null
    if ($NoGUI) {
        python -m pip install -e . pyinstaller | Out-Null
        $env:SDR_BUILD_GUI = "0"
    } else {
        python -m pip install -e ".[gui]" pyinstaller | Out-Null
        $env:SDR_BUILD_GUI = "1"
    }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    Write-Host "  Dependencies installed." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Dependency installation failed: $_" -ForegroundColor Red
    exit 1
}

# Step 5: Read the version from the single source of truth
Write-Host "[5/7] Reading package version..." -ForegroundColor Yellow
$appVersion = (python tools/get_version.py).Trim()
if (-not $appVersion) {
    Write-Host "ERROR: could not read the version from src/sdr_module/__init__.py" -ForegroundColor Red
    exit 1
}
Write-Host "  Version: $appVersion (GUI included: $($env:SDR_BUILD_GUI))" -ForegroundColor Green

# Step 6: Build executable
Write-Host "[6/7] Building Windows executable..." -ForegroundColor Yellow
Write-Host ""

$pyinstallerArgs = @("--noconfirm", "--clean", "sdr_module.spec")

if ($NoUPX) {
    $pyinstallerArgs += '--upx-dir=""'
}

try {
    & python -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed"
    }
} catch {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  BUILD FAILED" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "  Executable built successfully." -ForegroundColor Green

# Step 7: Create installer if requested
Write-Host "[7/7] Creating installer..." -ForegroundColor Yellow
if ($CreateInstaller) {
    # Find Inno Setup
    $isccPaths = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )

    $iscc = $null
    foreach ($path in $isccPaths) {
        if (Test-Path $path) {
            $iscc = $path
            break
        }
    }

    # Try PATH
    if (-not $iscc) {
        $iscc = Get-Command iscc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    }

    if ($iscc) {
        Write-Host "  Found Inno Setup: $iscc" -ForegroundColor Gray

        # Create output directory
        if (-not (Test-Path "installer_output")) {
            New-Item -ItemType Directory -Path "installer_output" | Out-Null
        }

        try {
            & $iscc "/DMyAppVersion=$appVersion" installer.iss
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  Installer created." -ForegroundColor Green
            } else {
                Write-Host "  Warning: Installer creation failed" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  Warning: Installer creation failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Inno Setup not found. Install from: https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipped (use -CreateInstaller to enable)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Executable location: dist\sdr-module\sdr-scan.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run:" -ForegroundColor White
Write-Host "  cd dist\sdr-module" -ForegroundColor Gray
Write-Host "  .\sdr-scan.exe gui --demo" -ForegroundColor Gray
Write-Host ""

if ($CreateInstaller -and (Test-Path "installer_output\SDR-Module-$appVersion-Setup.exe")) {
    Write-Host "Installer location: installer_output\SDR-Module-$appVersion-Setup.exe" -ForegroundColor Cyan
    Write-Host ""
}
