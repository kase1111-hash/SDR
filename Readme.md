# SDR Module

A dual-SDR framework for RTL-SDR and HackRF One, providing signal visualization, frequency analysis, signal classification, and protocol identification capabilities.

[![CI](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Dual-SDR Support**: Simultaneous operation of RTL-SDR and HackRF One
- **Signal Visualization**: Real-time spectrum, waterfall, constellation diagrams
- **Signal Classification**: Automatic detection of analog/digital modulation types
- **Protocol Detection**: Identify common protocols (ADS-B, POCSAG, LoRa, etc.)
- **Full-Duplex**: RTL-SDR receive + HackRF transmit simultaneously
- **Plugin System**: Extensible architecture for custom decoders

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

### Command Line

```bash
# List available devices
sdr-scan devices

# Scan FM broadcast band
sdr-scan scan --start 88 --end 108

# Launch GUI (requires PyQt6)
sdr-scan gui
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
```

### Signal Processing

```python
from sdr_module.dsp import SpectrumAnalyzer, SignalClassifier

# Analyze spectrum
analyzer = SpectrumAnalyzer(fft_size=1024)
result = analyzer.compute_spectrum(samples, center_freq=433.92e6, sample_rate=2.4e6)

# Classify signal
classifier = SignalClassifier(sample_rate=2.4e6)
classification = classifier.classify(samples)
print(f"Signal type: {classification.signal_type}")
print(f"Confidence: {classification.confidence:.2%}")
```

### Protocol Encoding

```bash
# Encode text as Morse code
sdr-scan encode morse --text "CQ CQ CQ" --output morse.iq --wpm 20

# Encode as RTTY
sdr-scan encode rtty --text "HELLO WORLD" --output rtty.iq
```

## Supported Modulations

**Analog**: AM, FM, SSB, CW

**Digital**: ASK/OOK, FSK, PSK, QAM, GFSK, MSK

## Supported Protocols

| Category | Protocols |
|----------|-----------|
| ISM Band | 433/868/915 MHz devices |
| Aviation | ADS-B, ACARS |
| Paging | POCSAG, FLEX |
| Amateur Radio | AX.25, APRS |
| Trunking | P25, DMR, TETRA |
| IoT | LoRa, Zigbee, Z-Wave |
| Space | ISS SSTV, Meteor-M2 |

## Project Structure

```
sdr-module/
├── src/sdr_module/
│   ├── core/          # Device management, configuration
│   ├── devices/       # RTL-SDR, HackRF drivers
│   ├── dsp/           # Signal processing algorithms
│   ├── gui/           # PyQt6 graphical interface
│   ├── plugins/       # Plugin system
│   ├── protocols/     # Protocol encoders/decoders
│   ├── ui/            # Visualization components
│   └── utils/         # Helper utilities
├── tests/             # Test suite
├── examples/          # Example scripts
└── tools/             # Utility tools
```

## Development

### Setup

```bash
git clone https://github.com/kase1111-hash/SDR.git
cd SDR
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
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

## Documentation

See [SPEC_SHEET.md](SPEC_SHEET.md) for detailed technical specifications.

## Requirements

- Python 3.9+
- NumPy
- Optional: pyrtlsdr, hackrf, scipy, matplotlib, PyQt6

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub Repository](https://github.com/kase1111-hash/SDR)
- [Issue Tracker](https://github.com/kase1111-hash/SDR/issues)
