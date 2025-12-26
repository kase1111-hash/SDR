#!/bin/bash
# Build portable SDR Module for Linux/macOS
# Usage: ./build_portable.sh [--gui] [--clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
INCLUDE_GUI=0
CLEAN=0
for arg in "$@"; do
    case $arg in
        --gui) INCLUDE_GUI=1 ;;
        --clean) CLEAN=1 ;;
    esac
done

echo "=========================================="
echo "  SDR Module Portable Build"
echo "=========================================="

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning previous build..."
    rm -rf build/ dist/ *.egg-info
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e . -q
pip install pyinstaller -q

if [ $INCLUDE_GUI -eq 1 ]; then
    echo "Including GUI support..."
    pip install PyQt6 -q
    export SDR_BUILD_GUI=1
fi

# Build
echo "Building portable executable..."
pyinstaller sdr_module.spec --noconfirm

# Create portable structure
DIST_DIR="dist/sdr-module"
echo "Creating portable structure..."

# Create directories for portable data
mkdir -p "$DIST_DIR/config"
mkdir -p "$DIST_DIR/recordings"
mkdir -p "$DIST_DIR/plugins"

# Create portable config
cat > "$DIST_DIR/config/settings.json" << 'EOF'
{
    "portable": true,
    "frequency": 100000000,
    "sample_rate": 2400000,
    "gain": 20,
    "recordings_dir": "./recordings",
    "plugins_dir": "./plugins"
}
EOF

# Create launcher script
cat > "$DIST_DIR/sdr-module.sh" << 'EOF'
#!/bin/bash
# SDR Module Portable Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export SDR_PORTABLE=1
export SDR_CONFIG_DIR="$SCRIPT_DIR/config"
./sdr-scan "$@"
EOF
chmod +x "$DIST_DIR/sdr-module.sh"

# Create README
cat > "$DIST_DIR/README.txt" << 'EOF'
SDR Module - Portable Edition
=============================

This is a portable version that can run from a USB drive.

USAGE:
------
Linux/macOS:
    ./sdr-module.sh info        # Show module info
    ./sdr-module.sh devices     # List SDR devices
    ./sdr-module.sh scan        # Frequency scanner

Windows:
    sdr-scan.exe info
    sdr-scan.exe devices
    sdr-scan.exe scan

DIRECTORIES:
------------
config/      - Configuration files (portable settings)
recordings/  - Saved I/Q recordings
plugins/     - Custom plugins

REQUIREMENTS:
-------------
- RTL-SDR or HackRF One device
- USB drivers installed on host system

For GUI version, run: sdr-scan gui --demo
EOF

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo ""
echo "Portable folder: $DIST_DIR"
echo ""
echo "To use, copy the 'sdr-module' folder to a USB drive."
echo ""
ls -la "$DIST_DIR"
