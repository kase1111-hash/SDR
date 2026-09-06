# Claude.md - SDR Module Project Guide

## Project Overview

SDR Module is a Software Defined Radio framework for dual-SDR operation using RTL-SDR and HackRF One devices. It provides signal analysis, visualization, classification, and protocol decoding capabilities for amateur radio operators and signal analysts.

**Version**: 0.2.0 (Beta)
**Python**: 3.10 - 3.14
**License**: MIT

## Directory Structure

```
src/sdr_module/          # Main source code
├── core/                # Device management, dual-SDR controller, config
├── devices/             # Hardware drivers (RTL-SDR via pyrtlsdr, HackRF via python_hackrf)
├── dsp/                 # Digital signal processing (spectrum, demodulators, filters)
├── gui/                 # PyQt6 graphical interface
├── ham/                 # Optional ham radio features (signal meter, QRP, SSTV, callsign)
│   └── gui/             # Ham radio GUI widgets (radio tuner, panels)
├── protocols/           # Protocol encoders/decoders
├── ui/                  # Visualization components (waterfall, constellation)
├── utils/               # Helper utilities (conversions, I/Q tools)
└── cli.py               # Command-line interface

packages/
└── sdr-antenna-array/   # Standalone antenna array package

tests/                   # pytest test suite
examples/                # Example scripts
tools/                   # Utility tools
```

## Key Technologies

- **NumPy** - Core numerical computing (required)
- **pyrtlsdr** / **python_hackrf** - SDR device drivers (optional)
- **scipy** - Advanced signal processing
- **PyQt6** - GUI framework
- **pytest** - Testing framework

## Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=sdr_module --cov-report=term-missing

# Format code
black src tests
isort src tests

# Type checking
mypy src

# Linting
ruff check src tests

# Run CLI
sdr-scan --help
```

## Coding Standards

### Formatting
- **Black**: 88 character line length
- **isort**: Black-compatible import sorting
- **Type hints**: Required, checked with mypy

### Naming Conventions
- Classes: `PascalCase` (e.g., `DualSDRController`, `SpectrumAnalyzer`)
- Functions/methods: `snake_case` (e.g., `set_frequency`, `demodulate`)
- Constants: `UPPER_SNAKE_CASE`
- Private members: Leading underscore (e.g., `_state`, `_lock`)

### Architecture Patterns

1. **Abstract Base Classes** - Used for device, demodulator, and protocol interfaces
2. **Dataclasses** - Used for configuration and state objects (often immutable)
3. **Enums** - Extensive use for type-safe constants (OperationMode, ModulationType, etc.)
4. **Thread Safety** - RLock used for shared state; thread-safety documented in docstrings

### Documentation
- Module docstrings describing purpose
- Function docstrings with Args, Returns, and Raises sections
- Thread-safety notes on critical classes

## Key Modules

### Core (`core/`)
- `chirp_csv.py` - CHIRP-compatible CSV import/export for saved channels
- `device_manager.py` - Device enumeration and lifecycle management
- `dual_sdr.py` - Dual-SDR controller with 5 operation modes
- `config.py` - Configuration management with validation

### DSP (`dsp/`)
- `spectrum.py` - FFT-based spectrum analyzer
- `demodulators.py` - AM, FM, SSB, CW, OOK, FSK, PSK, GFSK, MSK, QAM
- `filters.py` - FIR filters, AGC, decimators, interpolators
- `classifiers.py` - Signal classification with confidence scoring
- `protocols.py` - Protocol decoders (POCSAG, AX.25/APRS, RDS, ADS-B, FLEX, ACARS)
- `recording.py` - I/Q recording/playback (WAV, raw binary, SigMF)

### Devices (`devices/`)
- `base.py` - Abstract SDR device interface
- `rtlsdr.py` - RTL-SDR driver
- `hackrf.py` - HackRF One driver

### Protocols (`protocols/`)
- `base.py` - Protocol decoder base class and types
- `detector.py` - Automatic protocol detection
- `encoders.py` - Text encoders (RTTY, Morse, ASCII FSK, PSK31)

### Ham Radio (`ham/`) - Optional
- `signal_meter.py` - S-unit signal meter with RST reporting
- `qrp.py` - QRP (low-power) operations support
- `callsign.py` - Automatic CW callsign identification
- `sstv.py` - SSTV image decoder
- `gui/` - Radio tuner, QRP panel, SSTV panel, signal meter widget


## Testing

Tests use pytest with mocking for hardware simulation. Demo mode available for testing without physical devices.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_spectrum.py

# Run with verbose output
pytest -v --tb=short
```

## Hardware Support

- **RTL-SDR**: 500 kHz - 1.7 GHz, RX only
- **HackRF One**: 1 MHz - 6 GHz, TX/RX

Operation Modes:
- `DUAL_RX` - Simultaneous monitoring of two frequencies
- `FULL_DUPLEX` - RTL-SDR RX + HackRF TX
- `TX_MONITOR` - Monitor own transmission quality
- `WIDEBAND_SCAN` - Coordinated spectrum scanning
- `RELAY` - Receive-and-retransmit operations

## Important Notes

- Hardware drivers are optional; graceful degradation if missing
- GUI module may have relaxed mypy rules for PyQt6 compatibility
- Use demo mode for development without physical SDR devices
