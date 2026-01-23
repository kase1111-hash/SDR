# SDR Module

A dual-SDR framework for RTL-SDR and HackRF One, providing signal visualization, frequency analysis, signal classification, protocol identification, and amateur radio capabilities.

[![CI](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Capabilities
- **Dual-SDR Support**: Simultaneous operation of RTL-SDR and HackRF One
- **Signal Visualization**: Real-time spectrum analyzer, waterfall display, constellation diagrams
- **Signal Classification**: Automatic detection of analog/digital modulation types
- **Protocol Detection**: Identify and decode common protocols (ADS-B, POCSAG, LoRa, APRS, etc.)
- **Full-Duplex**: RTL-SDR receive + HackRF transmit simultaneously
- **Plugin System**: Extensible architecture for custom decoders and processors

### HAM Radio Features
- **AM/FM Radio Tuner**: Vintage car radio-style interface with presets
- **Signal Meter**: S-unit meter with RST reporting (S1-S9, S9+dB)
- **Callsign Identification**: Automatic CW callsign transmission for FCC compliance
- **SSTV Decoder**: Receive slow-scan television images from ISS and satellites
- **QRP Operations**: Power calculations and amplifier chain management for low-power ops

### DSP & Processing
- **Demodulators**: AM, FM, SSB, CW, OOK, ASK, FSK, PSK, GFSK, MSK, QAM
- **Filters**: FIR filters, AGC, squelch, noise reduction, CTCSS tone detection
- **Frequency Control**: AFC (Automatic Frequency Control), frequency locking
- **Recording**: I/Q recording/playback in WAV, raw binary, and SigMF formats
- **Text Encoding**: RTTY, Morse code, ASCII FSK, PSK31 encoders

## Hardware Support

| Device | Mode | Frequency Range | Bandwidth |
|--------|------|-----------------|-----------|
| RTL-SDR | RX | 500 kHz - 1.7 GHz | 2.4 MHz |
| HackRF One | TX/RX | 1 MHz - 6 GHz | 20 MHz |

**Combined Coverage**: 500 kHz - 6 GHz with 22.4 MHz combined bandwidth

## Installation

### Basic Installation

```bash
pip install sdr-module
```

### With Hardware Support

```bash
# RTL-SDR support
pip install sdr-module[rtlsdr]

# HackRF support
pip install sdr-module[hackrf]

# Full installation (all features)
pip install sdr-module[full]
```

### From Source

```bash
git clone https://github.com/kase1111-hash/SDR.git
cd SDR
pip install -e ".[full]"
```

## Quick Start

### Graphical User Interface

```bash
# Launch GUI with connected hardware
sdr-scan gui

# Demo mode (no hardware required)
sdr-scan gui --demo

# With specific frequency and gain
sdr-scan gui --demo --frequency 100000000 --gain 20
```

The GUI includes:
- Real-time spectrum analyzer with peak detection
- Waterfall display with time-frequency visualization
- Control panel for frequency, gain, and mode settings
- Protocol decoder panel for decoded data output
- AM/FM radio tuner with vintage car radio styling
- Signal meter with S-units display
- SSTV image viewer for satellite reception

### Command Line

```bash
# Display module information
sdr-scan info

# List available SDR devices
sdr-scan devices

# Scan FM broadcast band
sdr-scan scan --start 88 --end 108

# Scan with custom parameters
sdr-scan scan --start 430 --end 440 --step 25 --threshold -50
```

### Python API

```python
from sdr_module import DeviceManager, DualSDRController

# Initialize device manager
manager = DeviceManager()
devices = manager.scan_devices()

# Create dual-SDR controller
controller = DualSDRController()
controller.initialize()

# Set frequencies
controller.set_rtlsdr_frequency(433.92e6)  # ISM band
controller.set_hackrf_frequency(915e6)     # ISM band

# Start dual receive
controller.start_dual_rx()

# Read samples
samples = controller.read_rtlsdr_samples(262144)
```

### Signal Processing

```python
from sdr_module.dsp import SpectrumAnalyzer, SignalClassifier

# Analyze spectrum
analyzer = SpectrumAnalyzer(fft_size=1024)
result = analyzer.compute_spectrum(
    samples,
    center_freq=433.92e6,
    sample_rate=2.4e6
)

# Classify signal
classifier = SignalClassifier(sample_rate=2.4e6)
classification = classifier.classify(samples)
print(f"Signal type: {classification.signal_type}")
print(f"Modulation: {classification.modulation}")
print(f"Confidence: {classification.confidence:.2%}")
```

### Demodulation

```python
from sdr_module.dsp.demodulators import FMDemodulator, AMDemodulator

# FM demodulation
fm_demod = FMDemodulator(sample_rate=2.4e6)
audio = fm_demod.demodulate(samples)

# AM demodulation
am_demod = AMDemodulator(sample_rate=2.4e6)
audio = am_demod.demodulate(samples)
```

### Protocol Encoding

```bash
# Encode text as Morse code
sdr-scan encode morse --text "CQ CQ CQ DE W1ABC" --output morse.iq --wpm 20

# Encode as RTTY
sdr-scan encode rtty --text "HELLO WORLD" --output rtty.iq

# Encode as PSK31
sdr-scan encode psk31 --text "hello world" --output psk31.iq

# Encode as ASCII FSK
sdr-scan encode ascii --text "Test message" --output ascii.iq
```

### Recording and Playback

```python
from sdr_module.dsp.recording import IQRecorder, RecordingFormat

# Record I/Q samples
recorder = IQRecorder(sample_rate=2.4e6, format=RecordingFormat.SIGMF)
recorder.start("recording.sigmf")
# ... capture samples ...
recorder.stop()

# Playback
from sdr_module.dsp.recording import IQPlayer
player = IQPlayer("recording.sigmf")
samples = player.read_samples(262144)
```

## Supported Modulations

**Analog**: AM, FM, SSB (USB/LSB), CW

**Digital**: ASK/OOK, FSK (2FSK, 4FSK), PSK (BPSK, QPSK, 8PSK), QAM, GFSK, MSK

## Supported Protocols

| Category | Protocols |
|----------|-----------|
| ISM Band | 433/868/915 MHz devices, weather sensors, remote controls |
| Aviation | ADS-B, ACARS |
| Paging | POCSAG, FLEX |
| Amateur Radio | AX.25, APRS |
| Trunking | P25, DMR, TETRA |
| IoT | LoRa, Zigbee, Z-Wave |
| Space | ISS SSTV, Meteor-M2 |
| Broadcast | RDS (FM Radio Data System) |

## Dual-SDR Operation Modes

| Mode | RTL-SDR | HackRF One | Use Case |
|------|---------|------------|----------|
| DUAL_RX | RX @ Freq A | RX @ Freq B | Monitor two bands simultaneously |
| FULL_DUPLEX | RX @ Freq A | TX @ Freq B | Transceiver with simultaneous RX |
| TX_MONITOR | RX @ TX Freq | TX | Monitor own transmission quality |
| WIDEBAND_SCAN | Scan 0-1.7 GHz | Scan 1.7-6 GHz | Cover full spectrum faster |
| RELAY | RX Input | TX Output | Receive-and-retransmit applications |

## Project Structure

```
sdr-module/
├── src/sdr_module/
│   ├── core/          # Device management, dual-SDR controller, configuration
│   ├── devices/       # RTL-SDR and HackRF drivers
│   ├── dsp/           # Signal processing (spectrum, demodulators, filters, etc.)
│   ├── gui/           # PyQt6 graphical interface
│   ├── plugins/       # Plugin system architecture
│   ├── protocols/     # Protocol encoders/decoders
│   ├── ui/            # Visualization components (waterfall, constellation)
│   └── utils/         # Helper utilities (conversions, I/Q tools)
├── tests/             # Test suite (13 test modules)
├── examples/          # Example scripts and plugins
└── tools/             # Utility tools (text encoder)
```

## Plugin System

Extend functionality with custom plugins:

```python
from sdr_module.plugins import ProtocolPlugin, PluginMetadata, PluginType

class MyProtocolDecoder(ProtocolPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-protocol",
            version="1.0.0",
            plugin_type=PluginType.PROTOCOL,
            description="Custom protocol decoder"
        )

    def decode(self, samples: np.ndarray) -> dict:
        # Your decoding logic here
        return {"data": decoded_data}
```

Plugin types available:
- `ProtocolPlugin`: Custom protocol decoders
- `DemodulatorPlugin`: Custom demodulation algorithms
- `DevicePlugin`: Custom SDR device drivers
- `ProcessorPlugin`: Custom signal processing blocks

## Development

### Setup

```bash
git clone https://github.com/kase1111-hash/SDR.git
cd SDR
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sdr_module

# Run specific test file
pytest tests/test_dual_sdr.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/sdr_module
```

## Building

### Linux Portable Build

```bash
./build_portable.sh
```

### Windows Build

```batch
REM Basic build
build_windows.bat

REM Full build with installer
build_windows.bat --clean --install
```

### Windows Installer

```batch
build_installer.bat
```

Output: `installer_output/SDR-Module-0.1.0-Setup.exe`

## Documentation

- [SPEC_SHEET.md](SPEC_SHEET.md) - Detailed technical specifications
- [CHANGELOG.md](CHANGELOG.md) - Version history and release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy and vulnerability reporting
- [tools/README_TEXT_ENCODER.md](tools/README_TEXT_ENCODER.md) - Text encoder tool guide

## Requirements

- Python 3.9+ (tested on 3.9, 3.10, 3.11, 3.12, 3.13)
- NumPy ≥1.21.0

**Optional Dependencies:**
- pyrtlsdr ≥0.2.92 (RTL-SDR support)
- hackrf ≥1.0.0 (HackRF support)
- scipy ≥1.7.0 (advanced DSP)
- matplotlib ≥3.4.0 (plotting)
- PyQt6 (GUI application)

## Safety Features

The software includes hard-coded TX frequency lockouts for safety:
- GPS/GNSS frequencies (L1, L2, L5, GLONASS, Galileo, BeiDou)
- Aviation emergency frequencies (121.5 MHz, 243.0 MHz)
- ADS-B/Mode S (1030 MHz, 1090 MHz)
- Emergency beacons (406.0-406.1 MHz)
- Marine distress (156.8 MHz)
- Cellular bands

See [SPEC_SHEET.md](SPEC_SHEET.md) for complete details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a Pull Request.

Areas where contributions are appreciated:
- New protocol decoders
- Additional modulation support
- GUI improvements
- Documentation
- Bug fixes and optimizations

For security issues, please review our [Security Policy](SECURITY.md).

## Links

- [GitHub Repository](https://github.com/kase1111-hash/SDR)
- [Issue Tracker](https://github.com/kase1111-hash/SDR/issues)
