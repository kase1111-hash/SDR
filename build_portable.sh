#!/bin/bash
# Build a portable SDR Module folder for Linux/macOS with PyInstaller.
#
# Usage: ./build_portable.sh [--no-gui] [--clean]
#
#   --no-gui   Build the command-line tool only (no PyQt6 in the bundle).
#   --clean    Remove build/ and dist/ before building.
#
# Output: dist/sdr-module/ containing the `sdr-scan` executable, a launcher
# script and a README. Copy the folder to a USB drive to run it elsewhere.
#
# The bundle does not include the RTL-SDR / HackRF USB drivers: the host
# system still needs librtlsdr / libhackrf (plus udev rules on Linux).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INCLUDE_GUI=1
CLEAN=0
for arg in "$@"; do
    case $arg in
        --no-gui) INCLUDE_GUI=0 ;;
        --gui) INCLUDE_GUI=1 ;;   # kept for backwards compatibility (now the default)
        --clean) CLEAN=1 ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Always go through `python3 -m pip` so pip and the interpreter agree.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "Error: Python 3.10+ is required" >&2
    exit 1
fi

echo "=========================================="
echo "  SDR Module Portable Build"
echo "=========================================="
echo "Interpreter: $PY ($("$PY" --version 2>&1))"

if [ "$CLEAN" -eq 1 ]; then
    echo "Cleaning previous build..."
    rm -rf build/ dist/
fi

echo "Installing build dependencies..."
if [ "$INCLUDE_GUI" -eq 1 ]; then
    "$PY" -m pip install -q -e ".[gui]" pyinstaller
    export SDR_BUILD_GUI=1
else
    "$PY" -m pip install -q -e . pyinstaller
    export SDR_BUILD_GUI=0
fi

VERSION="$("$PY" tools/get_version.py)"
echo "Building sdr-module $VERSION (GUI: $INCLUDE_GUI)..."
"$PY" -m PyInstaller sdr_module.spec --noconfirm

DIST_DIR="dist/sdr-module"
mkdir -p "$DIST_DIR/recordings"

cat > "$DIST_DIR/sdr-module.sh" << 'LAUNCHER'
#!/bin/bash
# SDR Module portable launcher: runs sdr-scan from this folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/sdr-scan" "$@"
LAUNCHER
chmod +x "$DIST_DIR/sdr-module.sh"

cat > "$DIST_DIR/README.txt" << README
SDR Module $VERSION - Portable Edition
=====================================

Run from this folder (or a USB drive); no Python installation needed.

USAGE
-----
    ./sdr-module.sh info          # Build and capability summary
    ./sdr-module.sh devices       # List connected SDRs
    ./sdr-module.sh gui           # Graphical interface (needs a display)
    ./sdr-module.sh gui --demo    # GUI with synthetic signals, no hardware
    ./sdr-module.sh --help        # All commands

DIRECTORIES
-----------
recordings/   Suggested location for I/Q recordings (choose it in the GUI)

Settings persist in the user's profile (Qt QSettings for the GUI,
~/.config/sdr_module/config.json for the library), not in this folder.

REQUIREMENTS
------------
- RTL-SDR: librtlsdr installed on the host (and udev rules on Linux)
- HackRF One: libhackrf installed on the host
Without hardware, use demo mode: ./sdr-module.sh gui --demo
README

echo ""
echo "=========================================="
echo "  Build Complete: $DIST_DIR"
echo "=========================================="
ls -la "$DIST_DIR" | head -20
