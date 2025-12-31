# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Python 3.13 support

### Fixed
- Test warnings in test_framework.py (PytestReturnNotNoneWarning)

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Dual-SDR support for RTL-SDR and HackRF One
- Real-time spectrum analyzer with FFT-based visualization
- Waterfall display with time-frequency representation
- Signal classification for analog and digital modulations
- Protocol detection framework
- Demodulators: AM, FM, SSB, CW, OOK, FSK
- Filter bank with low-pass, high-pass, band-pass, and notch filters
- Automatic Frequency Control (AFC)
- Frequency scanning capability
- I/Q recording and playback (WAV, raw, SigMF formats)
- Text encoding (RTTY, Morse, ASCII, PSK31)
- SSTV decoder for satellite image reception
- HAM radio callsign identification
- S-unit signal meter with RST reporting
- QRP (low power) operations support
- Plugin system architecture
- PyQt6 GUI application
- Command-line interface (`sdr-scan`)
- Comprehensive configuration system with presets
- Tooltip system for RF/DSP terminology

### Hardware Support
- RTL-SDR (RX only): 500 kHz - 1.7 GHz, 2.4 MHz bandwidth
- HackRF One (TX/RX): 1 MHz - 6 GHz, 20 MHz bandwidth
- Dual-SDR operation modes: DUAL_RX, FULL_DUPLEX, TX_MONITOR, WIDEBAND_SCAN

### Protocols
- ISM band devices (433/868/915 MHz)
- Amateur Radio (AX.25, APRS)
- Aviation (ADS-B, ACARS)
- Paging (POCSAG, FLEX)
- Trunking (P25, DMR, TETRA)
- IoT (LoRa, Zigbee, Z-Wave)

[Unreleased]: https://github.com/kase1111-hash/SDR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kase1111-hash/SDR/releases/tag/v0.1.0
