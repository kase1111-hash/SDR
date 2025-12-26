# Software Defined Radio (SDR) Module - Specification Sheet

## 1. Overview

This module is a Software Defined Radio application that provides signal visualization, frequency analysis, signal classification, and protocol identification capabilities.

### 1.1 Target Hardware Configuration

This system is designed for a **dual-SDR setup** using:

| Device | Role | Key Capability |
|--------|------|----------------|
| **RTL-SDR** | Primary RX | Low-cost wideband receiver (500 kHz - 1.7 GHz) |
| **HackRF One** | TX/RX | Wideband transceiver (1 MHz - 6 GHz) |

### 1.2 Combined System Capabilities

| Capability | Specification |
|------------|---------------|
| Frequency Coverage | 500 kHz - 6 GHz (combined) |
| Simultaneous RX Channels | 2 (RTL-SDR + HackRF) |
| Transmit Capability | Yes (HackRF One) |
| Full-Duplex Operation | Yes (RTL-SDR RX + HackRF TX) |
| Maximum Combined Bandwidth | 22.4 MHz (2.4 + 20 MHz) |

---

## 2. Core Features

### 2.1 Signal Visualization & Graphing

| Feature | Description |
|---------|-------------|
| Real-time Spectrum Display | FFT-based frequency spectrum visualization |
| Waterfall Display | Time-frequency representation showing signal history |
| Time Domain View | Amplitude vs. time waveform display |
| Constellation Diagram | I/Q signal visualization for modulation analysis |
| Signal Strength Meter | Real-time RSSI/signal level indicator |

### 2.2 Frequency Tuning & Analysis

| Feature | Description |
|---------|-------------|
| Center Frequency Selection | Ability to tune to specific frequencies |
| Frequency Locking | "Zero in" on detected signals |
| Bandwidth Selection | Adjustable receive bandwidth |
| Automatic Frequency Control (AFC) | Automatic drift compensation |
| Frequency Scanning | Sweep across frequency ranges to detect activity |

---

## 3. Signal Classification

### 3.1 Analog vs. Digital Detection

| Classification | Detection Method |
|----------------|------------------|
| Analog Signals | Continuous waveform analysis, modulation depth measurement |
| Digital Signals | Symbol rate detection, discrete level identification |

### 3.2 Supported Analog Modulation Types

| Modulation | Description |
|------------|-------------|
| AM | Amplitude Modulation |
| FM | Frequency Modulation |
| SSB | Single Sideband (USB/LSB) |
| CW | Continuous Wave (Morse) |

### 3.3 Supported Digital Modulation Types

| Modulation | Description |
|------------|-------------|
| ASK/OOK | Amplitude Shift Keying / On-Off Keying |
| FSK | Frequency Shift Keying (2FSK, 4FSK, etc.) |
| PSK | Phase Shift Keying (BPSK, QPSK, 8PSK) |
| QAM | Quadrature Amplitude Modulation |
| GFSK | Gaussian Frequency Shift Keying |
| MSK | Minimum Shift Keying |

---

## 4. Protocol Identification

### 4.1 Protocol Analysis Capabilities

| Feature | Description |
|---------|-------------|
| Preamble Detection | Identifies sync patterns and preambles |
| Bit Rate Estimation | Automatic symbol/bit rate calculation |
| Frame Structure Analysis | Packet/frame boundary detection |
| Protocol Matching | Compares against known protocol signatures |

### 4.2 Common Protocols (Target Support)

| Category | Protocols |
|----------|-----------|
| ISM Band | 433 MHz, 868 MHz, 915 MHz devices |
| Wireless Sensors | Weather stations, temperature sensors |
| Remote Controls | Garage doors, car key fobs |
| Paging | POCSAG, FLEX |
| Amateur Radio | AX.25, APRS |
| Aviation | ADS-B, ACARS |
| Trunking | P25, DMR, TETRA |
| IoT | LoRa, Zigbee, Z-Wave |

---

## 5. Standard SDR Features

### 5.1 Signal Processing

| Feature | Description |
|---------|-------------|
| Filtering | Low-pass, high-pass, band-pass, notch filters |
| Decimation | Sample rate reduction |
| Resampling | Sample rate conversion |
| AGC | Automatic Gain Control |
| Squelch | Signal-level based audio muting |
| Noise Reduction | DSP-based noise filtering |

### 5.2 Recording & Playback

| Feature | Description |
|---------|-------------|
| I/Q Recording | Raw baseband signal capture |
| Audio Recording | Demodulated audio capture |
| Playback | Replay recorded I/Q files |
| File Formats | WAV, raw I/Q, SigMF |

### 5.3 Demodulation

| Feature | Description |
|---------|-------------|
| Multi-mode Demod | Support for various modulation schemes |
| Audio Output | Speaker/headphone output for audio signals |
| Data Output | Decoded digital data stream |

---

## 6. Technical Specifications

### 6.1 RF Performance Specifications

#### RTL-SDR (Receive Only)

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Frequency Range | 500 kHz - 1.766 GHz | 24-1766 MHz native; HF via direct sampling |
| Instantaneous Bandwidth | 2.4 MHz | 2.56 MHz max (may drop samples) |
| Maximum Sample Rate | 2.56 MS/s | 2.4 MS/s recommended for stability |
| ADC Resolution | 8-bit | RTL2832U chipset |
| Effective Number of Bits (ENOB) | ~7 bits | Actual usable resolution |
| Dynamic Range | ~42 dB | Limited by 8-bit ADC |
| Noise Figure | 6-8 dB | R820T2 tuner, frequency dependent |
| Sensitivity | -130 dBm | With external LNA |
| Maximum Input Power | +10 dBm | Do not exceed |
| Frequency Accuracy | 1 PPM | V3 with TCXO |
| Tuner Chip | R820T2 | Low-noise silicon tuner |
| Bias Tee | 4.5V DC, 180mA | Software-enabled (V3+) |

#### HackRF One (Transmit & Receive)

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Frequency Range | 1 MHz - 6 GHz | Full coverage |
| Instantaneous Bandwidth | 20 MHz | Maximum supported |
| Maximum Sample Rate | 20 MS/s | 8 MS/s minimum recommended |
| ADC/DAC Resolution | 8-bit | MAX5864 chipset |
| Effective Number of Bits (ENOB) | ~7 bits | Actual usable resolution |
| Dynamic Range | 48 dB | Limited by 8-bit ADC |
| Noise Figure | 10-15 dB | Varies by frequency; LNA recommended |
| Sensitivity | -115 dBm | With RX amp enabled |
| Maximum Input Power | -5 dBm | **CAUTION: Exceeding will cause damage** |
| TX Output Power | -10 to +15 dBm | Frequency dependent |
| Frequency Accuracy | 20 PPM | Standard crystal (TCXO upgrade available) |
| Half-Duplex | Yes | Cannot TX and RX simultaneously |
| RX/TX Amp | +14 dB nominal | Software-controlled |
| Antenna Port Power | 50mA @ 3.3V | Software-controlled |

#### Combined System Performance

| Parameter | RTL-SDR | HackRF One | Combined |
|-----------|---------|------------|----------|
| Frequency Range | 500 kHz - 1.7 GHz | 1 MHz - 6 GHz | **500 kHz - 6 GHz** |
| Bandwidth | 2.4 MHz | 20 MHz | **22.4 MHz** (parallel) |
| Sample Rate | 2.56 MS/s | 20 MS/s | **22.56 MS/s** (parallel) |
| TX Capability | No | Yes | **Yes** |
| Full Duplex | N/A | No | **Yes** (RTL RX + HackRF TX) |

### 6.2 Signal Processing Parameters

| Parameter | Specification |
|-----------|---------------|
| FFT Size | 1024 / 2048 / 4096 / 8192 / 16384 points (configurable) |
| FFT Window Functions | Hamming, Hann, Blackman, Blackman-Harris, Flat-top |
| FFT Overlap | 0% - 75% (configurable) |
| Averaging Modes | RMS, Peak Hold, Min Hold, Linear |
| DC Offset Removal | Automatic DC spike suppression |
| I/Q Imbalance Correction | Automatic gain and phase correction |
| Sample Rate Conversion | Arbitrary resampling supported |
| Filter Transition Bandwidth | Configurable (sharper = more CPU) |

### 6.3 Primary Hardware Specifications

#### RTL-SDR (Your Device)

| Component | Specification |
|-----------|---------------|
| Model | RTL-SDR Blog V3 (or compatible) |
| ADC Chip | RTL2832U |
| Tuner Chip | R820T2 |
| USB Interface | USB 2.0 High-Speed |
| Connector | SMA Female |
| Impedance | 50 Ω |
| Power | Bus-powered (~300mA) |
| TCXO | 1 PPM (V3), 0.5 PPM (V4) |
| Direct Sampling | Q-branch for HF (500 kHz - 24 MHz) |
| Case | Aluminum with thermal pad |

#### HackRF One (Your Device)

| Component | Specification |
|-----------|---------------|
| Model | HackRF One (Great Scott Gadgets) |
| Baseband Chip | MAX2837 (2.3-2.7 GHz IF) |
| ADC/DAC Chip | MAX5864 |
| RF Frontend | RFFC5071 (wideband mixer) |
| Processor | NXP LPC4320 (ARM Cortex-M4/M0) |
| USB Interface | USB 2.0 High-Speed |
| Connector | SMA Female |
| Impedance | 50 Ω |
| Power | Bus-powered (~500mA RX, ~900mA TX) |
| Clock | 20 PPM crystal (10 MHz external ref supported) |
| Expansion | Header for add-on boards (Opera Cake, etc.) |
| Open Source | Fully open hardware/firmware |

### 6.4 Dual-SDR Operation Modes

| Mode | RTL-SDR | HackRF One | Use Case |
|------|---------|------------|----------|
| **Dual RX** | RX @ Freq A | RX @ Freq B | Monitor two bands simultaneously |
| **Full Duplex** | RX @ Freq A | TX @ Freq B | Transceiver with simultaneous RX monitoring |
| **TX + Monitor** | RX @ TX Freq | TX | Monitor own transmission quality |
| **Wideband Scan** | Scan 0-1.7 GHz | Scan 1.7-6 GHz | Cover full spectrum faster |
| **Backup RX** | Primary RX | Secondary RX | Redundancy / comparison |
| **Signal Relay** | RX Input | TX Output | Receive-and-retransmit applications |

### 6.5 Dual-SDR Synchronization

| Parameter | Specification |
|-----------|---------------|
| Clock Sync | Independent (no hardware sync) |
| Software Sync | Sample timestamp alignment via USB SOF |
| Timing Accuracy | ~1 ms (USB latency limited) |
| External Reference | HackRF supports 10 MHz ext clock input |
| GPS Sync | Via external GPSDO to HackRF clock input |

> **Note**: RTL-SDR and HackRF One do not share a common clock. For applications requiring precise timing synchronization (e.g., TDOA), use software-based correlation or an external reference clock with HackRF.

### 6.6 Clock & Timing Specifications

| Parameter | RTL-SDR | HackRF One |
|-----------|---------|------------|
| Internal Clock | TCXO | Crystal (TCXO mod available) |
| Frequency Stability | 1 PPM | 20 PPM (standard) |
| External Clock Input | No | Yes (10 MHz SMA) |
| External Clock Output | No | Yes (10 MHz SMA) |
| GPS/PPS Support | No | Via external GPSDO |

### 6.7 Software Requirements

| Component | Requirement |
|-----------|-------------|
| Platform | Cross-platform (Linux, Windows, macOS) |
| Minimum RAM | 4 GB (8 GB recommended for dual-SDR) |
| CPU | Multi-core x86_64 or ARM64 (4+ cores recommended) |
| GPU Acceleration | Optional (OpenGL for visualization) |
| USB | 2x USB 2.0 ports (separate controllers recommended) |
| Dependencies | libusb, FFTW3, rtl-sdr, hackrf libraries |
| API | Python bindings, C/C++ API |

### 6.8 Dual-SDR Software Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| Driver | rtl-sdr | RTL-SDR device control |
| Driver | libhackrf | HackRF device control |
| Abstraction | SoapySDR | Unified SDR API (recommended) |
| DSP | GNU Radio | Signal processing framework |
| DSP | LiquidDSP | Lightweight DSP library |
| Application | Custom / GQRX / SDR++ | User interface |

---

## 7. Dual-SDR Use Cases

### 7.1 Simultaneous Receive (Dual RX)

Monitor two different frequencies at the same time.

| Application | RTL-SDR Frequency | HackRF Frequency | Description |
|-------------|-------------------|------------------|-------------|
| ADS-B + ACARS | 1090 MHz | 131.550 MHz | Aircraft tracking + voice |
| ISM Band Monitoring | 433 MHz | 915 MHz | Dual ISM band coverage |
| VHF + UHF Ham | 146 MHz | 446 MHz | 2m and 70cm bands |
| FM Broadcast + DAB | 98 MHz | 225 MHz | Analog + digital radio |
| Marine + Air | 156.8 MHz (Ch 16) | 121.5 MHz | Distress frequencies |
| Trunking + Control | 460 MHz (voice) | 851 MHz (control) | P25/DMR systems |

### 7.2 Full-Duplex Transceiver

Use RTL-SDR for receive while HackRF transmits.

| Application | RTL-SDR (RX) | HackRF (TX) | Description |
|-------------|--------------|-------------|-------------|
| Repeater | Input Freq | Output Freq | Simplex/duplex relay |
| Transponder | Uplink | Downlink | Satellite-style relay |
| SIGINT Training | Any | Test signal | Generate and capture signals |
| Protocol Testing | Device RX Freq | Device TX Freq | Stimulate and monitor devices |

### 7.3 Transmit Monitoring

Monitor your own transmissions for quality assurance.

| Setup | RTL-SDR | HackRF | Purpose |
|-------|---------|--------|---------|
| Spectrum Monitor | Same freq (attenuated) | TX | Check spectral purity |
| Modulation Check | Same freq | TX | Verify modulation quality |
| Spurious Monitor | Harmonic freq | TX | Detect spurious emissions |

### 7.4 Extended Frequency Coverage

Combine devices to cover wider spectrum.

| Band | Device | Frequency Range |
|------|--------|-----------------|
| HF (Direct Sampling) | RTL-SDR | 500 kHz - 24 MHz |
| VHF/UHF | RTL-SDR | 24 MHz - 1.7 GHz |
| UHF/Microwave | HackRF | 1.7 GHz - 6 GHz |

### 7.5 Signal Analysis & Classification Pipeline

```
┌─────────────┐     ┌─────────────────────────────┐
│   RTL-SDR   │────▶│  Primary Signal Capture     │
│  (RX Only)  │     │  • Signal detection         │
└─────────────┘     │  • Frequency identification │
                    │  • Initial classification   │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
┌─────────────┐     ┌─────────────────────────────┐
│  HackRF One │────▶│  Wideband Analysis          │
│   (RX/TX)   │     │  • 20 MHz capture           │
└─────────────┘     │  • Protocol decoding        │
                    │  • TX for active testing    │
                    └─────────────────────────────┘
```

---

## 8. User Interface Components

| Component | Description |
|-----------|-------------|
| Spectrum Analyzer | Main frequency display with adjustable span |
| Waterfall | Scrolling time-frequency display |
| Control Panel | Frequency, gain, bandwidth controls |
| Signal Classifier | Displays detected modulation type |
| Protocol Decoder | Shows identified protocol and decoded data |
| Recording Controls | Start/stop recording, file management |

---

## 9. Hardware Interface Specifications

### 9.1 USB Interface

| Parameter | Specification |
|-----------|---------------|
| USB Standard | USB 2.0 High-Speed / USB 3.0 SuperSpeed |
| USB 2.0 Data Rate | 480 Mbps (theoretical), ~35 MB/s practical |
| USB 3.0 Data Rate | 5 Gbps (theoretical), ~400 MB/s practical |
| Connector Type | USB Type-A, USB Type-C (device dependent) |
| Cable Length | ≤ 3m recommended for USB 2.0, ≤ 2m for USB 3.0 |
| Power Delivery | Bus-powered (500mA USB 2.0, 900mA USB 3.0) |

### 9.2 RF Connectors

| Parameter | Specification |
|-----------|---------------|
| Antenna Connector | SMA Female (most devices) |
| Impedance | 50 Ω |
| VSWR | < 2:1 (typical) |
| Clock I/O | SMA Female (supported devices) |
| GPIO/Expansion | Device-specific headers |

### 9.3 Data Formats

| Format | Description |
|--------|-------------|
| I/Q Sample Format | 8-bit unsigned, 16-bit signed, 32-bit float |
| Byte Order | Little-endian (I, Q interleaved) |
| Raw File Format | .raw, .cu8, .cs8, .cs16, .cf32 |
| Metadata Format | SigMF (Signal Metadata Format) |
| Audio Export | WAV (PCM 16-bit, 44.1/48 kHz) |

---

## 10. Physical & Environmental Specifications

### 10.1 Physical Dimensions

| Device Type | Dimensions (L × W × H) | Weight |
|-------------|------------------------|--------|
| USB Dongle (RTL-SDR) | 65 × 25 × 10 mm | ~25 g |
| Portable (HackRF) | 120 × 75 × 15 mm | ~200 g |
| Desktop (USRP) | 160 × 100 × 30 mm | ~500 g |

### 10.2 Environmental Conditions

| Parameter | Operating | Storage |
|-----------|-----------|---------|
| Temperature | 0°C to +55°C | -20°C to +70°C |
| Humidity | 10% to 90% RH (non-condensing) | 5% to 95% RH |
| Altitude | Up to 3,000 m | Up to 12,000 m |

### 10.3 Power Requirements

| Parameter | Specification |
|-----------|---------------|
| Input Voltage | 5V DC (USB bus power) |
| Current Draw (RX) | 200 - 500 mA typical |
| Current Draw (TX) | 300 - 900 mA typical |
| External Power | 5V/2A recommended for TX-capable devices |
| Bias Tee Output | 4.5V DC, 180mA max (supported devices) |

---

## 11. Compliance & Regulatory

### 11.1 Regulatory Notices

| Region | Certification | Notes |
|--------|---------------|-------|
| USA | FCC Part 15 | Receive-only exempt; TX requires license |
| Europe | CE Mark | RED compliance for TX devices |
| Canada | ISED | RSS-210 for unlicensed operation |
| Japan | TELEC | Certification required for TX |
| Australia | ACMA | Class license for certain bands |

### 11.2 Transmit Considerations

| Requirement | Description |
|-------------|-------------|
| Amateur License | Required for TX on amateur bands |
| ISM Bands | Limited TX power without license (varies by region) |
| Spurious Emissions | User responsible for filtering |
| Frequency Coordination | Required for certain services |

### 11.3 Legal Notice

> **WARNING**: Transmitting on frequencies without proper authorization is illegal in most jurisdictions. Users are responsible for compliance with all applicable laws and regulations. This software is intended for educational, amateur radio, and authorized research purposes only.

### 11.4 TX Frequency Lockouts (Safety Feature)

The software includes **hard-coded TX frequency lockouts** to prevent accidental transmission on critical safety frequencies.

| Category | Frequencies | Reason |
|----------|-------------|--------|
| **GPS/GNSS** | L1 (1575.42 MHz), L2 (1227.60 MHz), L5 (1176.45 MHz), GLONASS, Galileo, BeiDou | **CRITICAL SAFETY** - GPS spoofing can cause aircraft navigation failures |
| **Aviation Emergency** | 121.5 MHz, 243.0 MHz | International distress frequencies |
| **ADS-B/Mode S** | 1030 MHz, 1090 MHz | Aircraft collision avoidance transponders |
| **ELT/EPIRB** | 406.0-406.1 MHz | Search and rescue emergency beacons |
| **Marine Distress** | 156.8 MHz (Ch 16) | VHF marine distress and calling |
| **Cellular** | 698-806 MHz, 824-894 MHz, 1850-1995 MHz | Commercial cellular bands |

> **Note**: These lockouts **cannot be disabled** in software. TX attempts to these frequencies will be blocked with an error message.

### 11.5 HAM Radio Callsign Identification

For amateur radio compliance, the software includes automatic callsign identification:

| Feature | Description |
|---------|-------------|
| Automatic ID | Transmits callsign at start/end of transmission and every 10 minutes |
| CW Mode | Morse code identification at configurable WPM (5-50) |
| Tone Frequency | Configurable sidetone (default: 700 Hz) |
| "DE" Prefix | Proper amateur radio format: "DE [CALLSIGN]" |
| GUI Panel | Callsign input, countdown timer, manual ID button |

---

## 12. Future Considerations

| Feature | Description |
|---------|-------------|
| Plugin Architecture | Extensible protocol decoder support |
| Remote Operation | Network-based SDR control |
| Database Integration | Signal/protocol signature database |
| Machine Learning | AI-based signal classification |
| Multi-SDR Support | Multiple receiver operation |

---

## 13. Development Phases

### Phase 1: Core Infrastructure
- [x] Hardware abstraction layer
- [x] Basic signal acquisition
- [x] FFT and spectrum display
- [x] Waterfall display

### Phase 2: Signal Analysis
- [x] Analog/digital signal detection
- [x] Modulation classification
- [x] Basic demodulation (AM, FM, SSB)

### Phase 3: Protocol Identification
- [x] Symbol timing recovery
- [x] Frame synchronization
- [x] Protocol signature matching
- [x] Decoder framework

### Phase 4: Advanced Features
- [x] Recording/playback
- [x] Plugin system
- [x] Advanced protocols (ADS-B, ACARS, FLEX, AX.25/APRS, RDS, POCSAG)
- [x] Main GUI application (PyQt6-based with spectrum, waterfall, controls, decoder panels)

---

## 14. Windows Build & Installation

### 14.1 Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.8 or higher |
| pip | Python package manager |
| Inno Setup | 6.x (optional, for creating installer) |

### 14.2 Build Files

| File | Purpose |
|------|---------|
| `build_windows.bat` | Batch script for building Windows executable |
| `build_windows.ps1` | PowerShell script (alternative) |
| `build_installer.bat` | Creates Windows installer with Inno Setup |
| `sdr_module.spec` | PyInstaller specification file |
| `installer.iss` | Inno Setup installer script |

### 14.3 Quick Build (Command Prompt)

```batch
REM Basic build
build_windows.bat

REM Full clean build with development install
build_windows.bat --clean --install
```

### 14.4 Quick Build (PowerShell)

```powershell
# Basic build
.\build_windows.ps1

# Full clean build with installer
.\build_windows.ps1 -Clean -Install -CreateInstaller
```

### 14.5 Build Options

| Option | Batch | PowerShell | Description |
|--------|-------|------------|-------------|
| Clean | `--clean` | `-Clean` | Remove build directories before building |
| Install | `--install` | `-Install` | Install package in development mode |
| No UPX | `--no-upx` | `-NoUPX` | Disable UPX compression |
| Installer | N/A | `-CreateInstaller` | Create Windows installer |

### 14.6 Creating the Installer

After building the executable:

```batch
build_installer.bat
```

Or with PowerShell:
```powershell
.\build_windows.ps1 -CreateInstaller
```

### 14.7 Output Locations

| Output | Location |
|--------|----------|
| Executable | `dist\sdr-module\sdr-scan.exe` |
| Installer | `installer_output\SDR-Module-0.1.0-Setup.exe` |

### 14.8 Manual Installation

If not using the installer:

1. Copy the `dist\sdr-module` folder to your preferred location
2. Optionally add the folder to your system PATH
3. Run `sdr-scan.exe --help` to verify installation

---

*Document Version: 3.5*
*Last Updated: 2025-12-26*

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-25 | Initial specification document |
| 2.0 | 2025-12-25 | Added quantitative RF specs, hardware compatibility matrix, interface specs, physical/environmental specs, compliance section |
| 3.0 | 2025-12-25 | Tuned for dual-SDR setup (RTL-SDR + HackRF One); added device-specific specs, dual-SDR operation modes, synchronization, use cases, software stack |
| 3.1 | 2025-12-26 | Added Windows build and installation documentation (Section 14) |
| 3.2 | 2025-12-26 | Implemented plugin system architecture; updated development phases to reflect current implementation status |
| 3.3 | 2025-12-26 | Added advanced protocol decoders: ADS-B, ACARS, FLEX; verified AX.25/APRS, RDS, POCSAG implementations |
| 3.4 | 2025-12-26 | Implemented Main GUI application with PyQt6: spectrum analyzer, waterfall display, control panel, protocol decoder panel, recording controls, device dialog |
| 3.5 | 2025-12-26 | Added TX frequency lockouts for safety (GPS, aviation, emergency, cellular); HAM radio callsign identification; RX presets with GUI selector |
