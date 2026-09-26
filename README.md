# SDR Module

A dual-SDR framework for simultaneous RTL-SDR + HackRF One operation, with signal visualization, protocol decoding, and a PyQt6 GUI.

[![CI](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

- **Dual-SDR control**: Operate RTL-SDR and HackRF One simultaneously in five modes (dual RX, full-duplex, TX monitor, wideband scan, relay)
- **Signal processing**: Spectrum analysis, demodulation (AM/FM/SSB/CW/OOK/FSK/PSK/QAM), filtering, AGC, and signal classification
- **Protocol decoding**: ADS-B, POCSAG, FLEX, AX.25/APRS, RDS, ACARS
- **GUI**: PyQt6 application with spectrum analyzer, waterfall, click-to-tune, live protocol decoder, bookmarks, band presets, audio output, ham radio tools, dark/light themes, keyboard shortcuts, a no-hardware demo mode, and persisted state

## Hardware Support

| Device | Mode | Frequency Range | Bandwidth |
|--------|------|-----------------|-----------|
| RTL-SDR | RX only | 500 kHz - 1.7 GHz | 2.4 MHz |
| HackRF One | TX/RX | 1 MHz - 6 GHz | 20 MHz |

**Combined coverage**: 500 kHz - 6 GHz

## Known Limitations

- **Real-time bandwidth**: Pure Python + NumPy DSP cannot sustain the full 20 MHz HackRF bandwidth in real-time. Effective throughput depends on FFT size, demodulator complexity, and host CPU. For wideband capture, record I/Q to disk and process offline.
- **HackRF half-duplex**: HackRF One cannot TX and RX simultaneously. Full-duplex mode pairs RTL-SDR (RX) with HackRF (TX).
- **RTL-SDR RX only**: The RTL-SDR cannot transmit.
- **No GPU acceleration**: All signal processing runs on CPU via NumPy.

## Installation

Install from source (the package is not yet published to PyPI):

```bash
git clone https://github.com/kase1111-hash/SDR.git
cd SDR

# Basic (offline analysis, encoding, decoding — no hardware drivers)
python -m pip install -e .

# With optional extras
python -m pip install -e ".[rtlsdr]"   # RTL-SDR driver (pyrtlsdr)
python -m pip install -e ".[hackrf]"   # HackRF driver (python_hackrf)
python -m pip install -e ".[gui]"      # PyQt6 GUI
python -m pip install -e ".[dsp]"      # SciPy-accelerated DSP
python -m pip install -e ".[images]"   # Pillow, for SSTV image export
python -m pip install -e ".[full]"     # rtlsdr + gui + dsp + images (wheels only)
```

### Hardware drivers (system libraries required)

The Python driver packages wrap system libraries that must be installed first:

- **RTL-SDR** (`[rtlsdr]` → `pyrtlsdr`) needs **librtlsdr**.
  Debian/Ubuntu: `sudo apt install librtlsdr-dev`; macOS: `brew install librtlsdr`.
  On Linux, install the RTL-SDR udev rules so the device is usable without root.
- **HackRF** (`[hackrf]` → `python_hackrf`) needs **libhackrf** *and a C
  compiler and CMake* (python_hackrf builds from source).
  Debian/Ubuntu: `sudo apt install libhackrf-dev build-essential cmake`;
  macOS: `brew install hackrf cmake`.
  Because it builds from source, `[hackrf]` is deliberately left out of
  `[full]`; add it explicitly once the toolchain is present.

Without a driver installed, `sdr-scan devices` reports no hardware and the GUI
still runs in demo mode (`sdr-scan gui --demo`); offline analysis, decoding,
and encoding work with no drivers at all.

> **Tip:** Always install with `python -m pip` rather than a bare `pip`. On some
> systems the `pip` on your `PATH` belongs to a different interpreter than
> `python`, which silently installs packages where `python` can't import them
> (the classic `ModuleNotFoundError: No module named 'numpy'` right after a
> "successful" install). Routing through `python -m pip` guarantees they agree.

## Quick Start

### GUI

```bash
sdr-scan gui              # Launch with connected hardware
sdr-scan gui --demo       # Demo mode (no hardware required)
```

On first launch the GUI shows a welcome wizard that says whether it found an
SDR, offers to connect it or start demo mode, and picks a starting band
(Help → Welcome and Quick Start shows it again). With nothing saved yet the
receiver starts on **100.1 MHz in FM**. `--frequency`/`-f` (Hz) and
`--gain`/`-g` override the remembered values; `--sample-rate`/`-s` sets the
demo device's rate and the one Device → Connect preselects. Press **F1** at
any time for the shortcut reference.

**Demo mode** (`--demo`, Device → Use Demo Device, or the Demo Device entry
in Device → Connect) simulates FM broadcast stations, airband AM, 2 m, 70 cm
and NOAA weather FM, ISM and APRS bursts, ADS-B pulses and a CW beacon on a
realistic noise floor, streamed in real time. The strongest station,
100.1 MHz, plays a looping tune in FM, so audio works without hardware; gain,
AGC and the frequency scanner behave as they would with a real receiver.

### Command line

```bash
sdr-scan info                                      # Build + capability summary
sdr-scan devices                                   # Scan for connected SDRs
sdr-scan scan --start 88 --end 108 --step 100      # Live sweep (needs hardware)
sdr-scan encode morse --text "HELLO" -o out.cf32   # Encode text to raw I/Q
sdr-scan encode morse --text "HELLO" -o out.wav    # ...or to a playable WAV
```

#### Offline analysis (no hardware required)

Record I/Q to disk (via the GUI or any SDR tool), then analyze it offline:

```bash
# Detect signals in a capture via FFT peak detection
sdr-scan scan --input capture.cf32 --sample-rate 2400000 --center 100e6 --threshold -40

# Decode a protocol from a capture (POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS)
# (RDS needs 57 kHz subcarrier recovery and is not decodable from a raw I/Q capture yet)
sdr-scan decode adsb --input adsb_1090.cf32 --sample-rate 2000000
```

#### Memory channels (CHIRP-compatible CSV)

Saved channels move in and out as [CHIRP](https://chirpmyradio.com/) generic
CSV files, so the same file works in CHIRP and can be uploaded to a handheld:

```bash
sdr-scan channels export my-channels.csv     # Saved channels -> CHIRP CSV
sdr-scan channels export bands.csv --presets # ...or the built-in RX presets
sdr-scan channels import my-channels.csv     # CHIRP CSV -> saved channels
sdr-scan channels import more.csv --append   # Add instead of replacing
sdr-scan channels list my-channels.csv       # Show a file's channels
sdr-scan channels list                       # Show the saved channels
```

Files use CHIRP's exact 21-column header (`Location,Name,Frequency,Duplex,
Offset,Tone,...`), frequencies in MHz. Import is forgiving: any subset of the
optional columns works, headers are matched case-insensitively, extra columns
are ignored, and repeater shift, tone/DTCS, mode, tuning step, skip, power,
and comment fields are preserved on a round trip. Reading and writing saved
channels needs PyQt6 (they live in the GUI's settings store); `list` and
`export --presets` on a file work without it.

The sample format, rate, and center frequency are auto-detected for WAV and
SigMF files (pass either the `.sigmf-data` or `.sigmf-meta` path); for
headerless raw captures pass `--format` (`cu8`, `cs8`, `cs16`, `cf32`,
`cf64`) and `--sample-rate`. `cu8` is the RTL-SDR native format. Note that
ADS-B requires a capture of at least 2 MHz sample rate (Mode S bits are
1 µs wide).

### Python API

```python
from sdr_module import DeviceManager, DualSDRController

# Scan for devices
manager = DeviceManager()
devices = manager.scan_devices()

# Dual-SDR operation
controller = DualSDRController()
controller.initialize()
controller.set_rtlsdr_frequency(433.92e6)
controller.set_hackrf_frequency(915e6)
controller.start_dual_rx()

samples = controller.read_rtlsdr_samples(262144)
```

### Signal Processing

```python
from sdr_module.dsp import SpectrumAnalyzer, SignalClassifier
from sdr_module.dsp.demodulators import FMDemodulator

# Spectrum analysis
analyzer = SpectrumAnalyzer(fft_size=1024)
result = analyzer.compute_spectrum(samples, center_freq=433.92e6, sample_rate=2.4e6)

# Signal classification
classifier = SignalClassifier(sample_rate=2.4e6)
classification = classifier.classify(samples)

# FM demodulation
fm_demod = FMDemodulator(sample_rate=2.4e6)
audio = fm_demod.demodulate(samples)
```

### Protocol Decoding

```python
from sdr_module.dsp.protocols import create_protocol_decoder, ProtocolType

decoder = create_protocol_decoder(ProtocolType.ADSB, sample_rate=2e6)
messages = decoder.decode(samples)
for msg in messages:
    print(f"ICAO: {msg.icao_address}, Alt: {msg.altitude}")
```

## GUI Features

The window has a toolbar (Start/Stop, Record, Audio on/off with volume, the
FREQ readout and the tuned channel's LEVEL in dBFS), the spectrum over the
waterfall on the left, the control panel on the right, and below it the tool
tabs: **Decoder**, **Bookmarks**, **Ham Radio** (S-Meter, Ham ID, SSTV, QRP)
and **Info**. The status bar shows the receiver state, device and sample
rate, short feedback messages, and a REC badge with the recorded size and
free disk space while recording.

| Feature | How |
|---|---|
| Click-to-tune | Left-click anywhere on the spectrum or waterfall; the shaded band on the spectrum is the channel being demodulated |
| Type a frequency | `Ctrl+L` (Radio → Enter Frequency) or click the FREQ readout; Enter tunes and returns to the plots |
| Keyboard tuning | `←`/`→` ±10 kHz, `Shift+←`/`→` ±100 kHz, `Ctrl+←`/`→` ±1 MHz while the plots have focus (`F6` or `Esc` focuses the spectrum) |
| Start / stop receiving | `Space`, the toolbar's Start button, or Radio → Start Receiving |
| Record I/Q | `Ctrl+Shift+R`, the toolbar's Record button, or the control panel's Recording section (with Pause); samples are held in memory until saved |
| Save / import recording | `Ctrl+S` File → Save Recording (cf32, cs16, SigMF, 16-bit I/Q WAV, raw); `Ctrl+O` File → Import Recording loads a file to save it in another format (no playback yet) |
| Screenshot | `Ctrl+P` File → Save Screenshot (PNG of the whole window) |
| Bookmarks | `Ctrl+B` (Radio → Bookmark Current Frequency) saves the frequency with its mode; in the Bookmarks tab double-click or `Enter` tunes, `F2` renames, `Delete` removes |
| Import / export channels | Bookmarks tab → CSV button, or File → Import/Export Channels (CHIRP CSV) |
| Band presets | Radio → Band Presets (FM Broadcast, NOAA Weather, 2m Ham, 70cm Ham, Airband AM, ADS-B, ISM 433, ISM 915), which also set mode and bandwidth; the control panel's Presets section has the full categorized list |
| Protocol decoder | Decoder tab: POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS from the live signal, with message table, log, stats and CSV export |
| Frequency scanner | `Ctrl+F` Tools → Frequency Scanner: non-blocking sweep with band presets, progress bar and hit list; double-click a hit to tune |
| Audio output | On by default: toolbar Audio button and volume slider, or Radio → Audio Output. AM, FM, USB, LSB and CW play through the default sound device while the channel is above squelch |
| Squelch + AGC | Control panel: Demodulation → Squelch, Receiver → Automatic gain (AGC); the LEVEL readout turns green above squelch |
| AM/FM radio tuner | `Ctrl+R` Tools → AM/FM Radio Tuner (see [Ham Radio Features](#optional-ham-radio-features)) |
| Tool panels | `Ctrl+1` … `Ctrl+7` (View → Panels) jump to Decoder, Bookmarks, S-Meter, Ham ID, SSTV, QRP and Info |
| Dark / light theme | `Ctrl+T` or View → Theme; switches live |
| Error history | `Ctrl+E` Tools → Error History: the last 500 warning / error log records, with copy and export |
| Help / shortcuts | `F1`: every shortcut, grouped by menu and filterable |

Settings persist across launches via `QSettings`: frequency, gain, AGC,
squelch, bandwidth, demodulation mode and FM deviation, theme, audio on/off
and volume, recording format, the open tool tab, license class, Ham ID
settings, window and splitter sizes, and bookmarks.

The GUI has a dark (default) and a light theme, both built from the palette
tokens in `sdr_module/gui/themes.py`; widgets take their colors from style
roles and tones there rather than hard-coding them, so custom-painted views
(spectrum, waterfall axes, S-meter, radio tuner) follow a theme switch too.

## Supported Protocols

| Category | Protocols | Status |
|----------|-----------|--------|
| Aviation | ADS-B, ACARS | Implemented |
| Paging | POCSAG, FLEX | Implemented |
| Amateur Radio | AX.25, APRS | Implemented |
| Broadcast | RDS (FM Radio Data System) | Implemented |
| ISM Band | 433/868/915 MHz devices, weather sensors | Implemented (OOK/FSK) |

## Dual-SDR Operation Modes

| Mode | RTL-SDR | HackRF One | Use Case |
|------|---------|------------|----------|
| DUAL_RX | RX @ Freq A | RX @ Freq B | Monitor two bands simultaneously |
| FULL_DUPLEX | RX @ Freq A | TX @ Freq B | Transceiver with simultaneous RX |
| TX_MONITOR | RX @ TX Freq | TX | Monitor own transmission quality |
| WIDEBAND_SCAN | Scan 0-1.7 GHz | Scan 1.7-6 GHz | Cover full spectrum faster |
| RELAY | RX Input | TX Output | Receive-and-retransmit |

## Safety

Hard-coded TX frequency lockouts prevent transmission on protected
frequencies: GPS/GNSS, aviation (121.5/243.0 MHz), ADS-B (1030/1090 MHz),
emergency beacons (406 MHz), marine distress (156.8 MHz), and cellular
bands. License-class enforcement blocks TX on ham bands unless the
configured class has privileges there. See
[SPEC_SHEET.md](SPEC_SHEET.md) for the full list.

The lockout path is covered by `tests/test_frequency_manager.py` (61 tests).

## Optional: Ham Radio Features

The `sdr_module.ham` subpackage provides amateur radio functionality:

- **AM/FM Radio Tuner** laid out like a car radio (frequency display,
  slide-rule dial, AM/FM buttons, six presets per band); it tunes the main
  receiver, and audio plays through the main window's Audio button
- **Signal Meter** with S-units (S1-S9, S9+dB) and RST reporting
- **Callsign ID** for automatic CW identification (transmitting needs a HackRF One)
- **SSTV Decoder** for ISS image reception
- **QRP Operations** with power calculations

In the GUI the meter, callsign ID, SSTV and QRP tools are the S-Meter,
Ham ID, SSTV and QRP sub-tabs of the **Ham Radio** tab; the tuner opens from
Tools → AM/FM Radio Tuner (`Ctrl+R`).

```python
from sdr_module.ham import SignalMeter, QRPController
from sdr_module.ham.gui import RadioTunerWidget
```

## Optional: Antenna Array

The `sdr-antenna-array` package (in `packages/sdr-antenna-array/`) provides multi-SDR antenna array support:

- Beamforming (delay-and-sum, MVDR/Capon)
- Direction of arrival estimation (MUSIC, beamscan)
- Array calibration

```bash
python -m pip install -e packages/sdr-antenna-array/
```

## Project Structure

```
sdr-module/
├── src/sdr_module/
│   ├── core/          # Device management, dual-SDR controller, config, frequency manager
│   ├── devices/       # RTL-SDR and HackRF drivers
│   ├── dsp/           # Signal processing, demodulators, protocol decoders, recording
│   ├── gui/           # PyQt6 GUI: main window, panels, dialogs, themes, settings store
│   ├── ham/           # Optional ham radio features + radio tuner UI
│   ├── protocols/     # Protocol encoders + detector
│   ├── ui/            # Visualization components (waterfall, constellation, time-domain)
│   └── utils/         # Helper utilities
├── packages/
│   └── sdr-antenna-array/  # Standalone antenna array package
├── tests/             # Test suite (~1,550 tests, plus ~130 in sdr-antenna-array)
├── examples/          # Example scripts
└── tools/             # Dev helpers
```

## Development

```bash
git clone https://github.com/kase1111-hash/SDR.git
cd SDR
python -m pip install -e ".[dev]"
pytest                                # Run tests
pytest --cov=sdr_module               # With coverage
ruff check src/ tests/
black --check src/ tests/
isort --check-only src/ tests/
mypy src/sdr_module --ignore-missing-imports
```

## Requirements

- Python 3.10 - 3.14
- NumPy >= 1.21.0
- **Optional**: pyrtlsdr (RTL-SDR), python_hackrf (HackRF), scipy (advanced DSP), PyQt6 (GUI, includes multimedia for audio output), Pillow (SSTV image export)

See [Hardware drivers](#hardware-drivers-system-libraries-required) for the
system libraries the driver packages require.

## License

MIT License - see [LICENSE](LICENSE.md) for details.

## Links

- [GitHub Repository](https://github.com/kase1111-hash/SDR)
- [Issue Tracker](https://github.com/kase1111-hash/SDR/issues)
- [Technical Specifications](SPEC_SHEET.md)
- [Changelog](CHANGELOG.md)
- [Audit History](AUDIT_HISTORY.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
