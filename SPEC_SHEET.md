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

### 4.2 Supported Protocols

| Category | Protocols | Status |
|----------|-----------|--------|
| ISM Band | 433 MHz, 868 MHz, 915 MHz devices | Implemented |
| Wireless Sensors | Weather stations, temperature sensors | Implemented (via OOK/FSK decoding) |
| Remote Controls | Garage doors, car key fobs | Implemented (via OOK/ASK decoding) |
| Paging | POCSAG, FLEX | Implemented |
| Amateur Radio | AX.25, APRS | Implemented |
| Aviation | ADS-B, ACARS | Implemented |
| Broadcast | RDS (FM Radio Data System) | Implemented |

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
| Playback | Replay recorded I/Q files (Python API; the GUI imports a file only to convert its format) |
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
| GPU Acceleration | None (CPU only, NumPy-based DSP) |
| USB | 2x USB 2.0 ports (separate controllers recommended) |
| Dependencies | libusb, rtl-sdr, hackrf libraries |
| API | Python API (sdr_module package) |

### 6.8 Software Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| Driver | pyrtlsdr | RTL-SDR Python bindings |
| Driver | python_hackrf | HackRF Python bindings |
| DSP | NumPy / SciPy | Signal processing |
| GUI | PyQt6 | User interface |
| Application | sdr-module | This project |

### 6.9 DSP Performance Characteristics

All signal processing runs in Python/NumPy on CPU. The following are realistic throughput expectations:

| Operation | FFT/Block Size | Approx. Throughput | Notes |
|-----------|---------------|-------------------|-------|
| Spectrum (FFT) | 4096 | ~50 MS/s | NumPy FFT, single core |
| Spectrum (FFT) | 8192 | ~30 MS/s | Larger FFT reduces throughput |
| FM Demodulation | N/A | ~2.4 MS/s real-time | Matches RTL-SDR max rate |
| AM Demodulation | N/A | ~2.4 MS/s real-time | Envelope detection is lightweight |
| Protocol Decoding | N/A | Varies | Bit-level processing, lower throughput |
| Filter (FIR, 101 taps) | N/A | ~5 MS/s | NumPy convolution |

**Key constraints:**
- The RTL-SDR's 2.4 MHz bandwidth is generally processable in real-time
- The HackRF's 20 MHz bandwidth exceeds real-time DSP capacity for most operations
- For wideband HackRF capture, record raw I/Q and process offline
- GUI display updates are capped at 30 Hz to prevent UI thread starvation
- Running multiple demodulators simultaneously reduces per-chain throughput

These figures are approximate and vary by CPU. Measured on a typical 4-core x86_64 system.

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
| UHF Monitoring | 460 MHz | 851 MHz | Dual UHF band monitoring |

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

### 8.1 Main window

```
+------------------------------------------------------------------+
| File  Device  Radio  View  Tools  Help                           |
| [Start] [Record] | [Audio] vol | FREQ 100.100 MHz | LEVEL dBFS   |
+---------------------------------------+--------------------------+
| spectrum                              | control panel (scrolls)  |
|---------------------------------------|--------------------------|
| waterfall                             | tabs: Decoder, Bookmarks,|
|                                       |       Ham Radio, Info    |
+---------------------------------------+--------------------------+
| RUNNING  Demo Device  2.4 MS/s  messages...      REC 00:01:23    |
+------------------------------------------------------------------+
```

The three splitters (plots | right column, spectrum / waterfall, control
panel / tabs) are resizable; their sizes persist and View → Reset Layout
restores the defaults. The minimum window size (1024 × 640) fits a
1366 × 768 screen; the control panel and the tool tabs scroll instead of
being crushed.

| Component | Description |
|-----------|-------------|
| Toolbar | Start/Stop; Record (red while recording); Audio on/off with a volume slider; FREQ readout (click it, or press `Ctrl+L`, to type a frequency); LEVEL readout: strongest signal inside the tuned channel in dBFS, green while above squelch |
| Spectrum Analyzer | 2048-point Hann-windowed FFT in dBFS; peak hold (with Reset) and averaging (Off, 2–32 frames); hover readout; dashed tuned-frequency marker and a shaded band for the demodulated channel (passband); left-click to tune |
| Waterfall | Newest line at the top with a time axis (pauses in reception drawn as separators). Header: Pause (`P`); History (Live, 1 min, 5 min, 15 min, 1 h; slower rows keep the per-bin maximum of their interval, and rows keep an exact cadence); LEVELS, a draggable color-scale legend in dBFS with Auto (floor 6 dB under the median-bin noise floor, ceiling 4 dB over the strongest signal, at least 30 dB apart, moving only on 3 dB changes); ⋯ menu. Hover readout: frequency, level, SNR over the noise floor, age. Left-click tunes; dragging measures a box (Δf, Δt, center, peak and SNR) that follows its rows as they scroll (`Esc` clears it). Right-click: tune/bookmark/copy the frequency, measurement actions, pause, history, levels (Auto, 60–120 dB presets, fit once), colors (Turbo, Viridis, Plasma, Grayscale, Classic), Clear History, Save Image (PNG with axes). Its frequency axis lines up with the spectrum's, and it draws its own when the spectrum is hidden. Colors, history and levels persist |
| Control Panel | **Frequency** (entry with Hz/kHz/MHz/GHz units, −1M … +1M step buttons); **Presets** (category and preset, with listen/transmit status, Apply Preset); **Demodulation** (None (I/Q), AM, FM, USB, LSB, CW; FM deviation; squelch); **Receiver** (RF gain, automatic gain (AGC), channel bandwidth); **Recording** (format, Record/Pause, status and elapsed time); **License Profile** (class, TX privileges, dummy-load reminder) |
| Decoder tab | Live protocol decoder: POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS (RDS is listed but needs 57 kHz subcarrier recovery, so it cannot be decoded live yet); Messages table, plain-text Log and Stats; Clear and Export (CSV); a one-click "Tune to …" for protocols with a well-known channel (ADS-B, APRS, ACARS) |
| Bookmarks tab | Saved channels with name, frequency and mode; Add (name and frequency), Tune, Rename, Remove; double-click or `Enter` tunes (restoring the saved mode), `F2` renames, `Delete` removes; CSV button to import/export CHIRP-compatible CSV |
| Ham Radio tab | Sub-tabs **S-Meter** (analog meter of the tuned channel, RST and verbal report, peak hold), **Ham ID** (callsign, CW speed and tone, preview, Send ID Now; transmits with a HackRF One only), **SSTV** (decoder fed with FM audio, live image, history, auto-save) and **QRP** (TX power, amplifier chain, power limit, miles per watt) |
| Info tab | Live summary: device, state, center frequency, sample rate, span, RBW, demodulation, bandwidth, gain, squelch, channel level; audio, recording and buffer state; version, Qt/PyQt and theme; quick tips |
| Status Bar | State badge (NO DEVICE / STOPPED / RUNNING), device name and sample rate; short feedback messages colored by tone (info, success, warning, error) that clear themselves; while recording, a REC badge with the elapsed time (REC ARMED until the receiver runs, PAUSED while paused) plus recorded size and free disk space |

### 8.2 Menus and dialogs

| Menu / Dialog | Purpose |
|---|---|
| File → Import Recording (convert format) | `Ctrl+O`; load an I/Q file (cf32, cs16, cs8, cu8, cf64, raw, SigMF, WAV) into the recording buffer so it can be saved in another format (no playback yet) |
| File → Save Recording | `Ctrl+S`; write the recording as Complex Float32 (.cf32), Complex Int16 (.cs16), SigMF (.sigmf-data plus .sigmf-meta), 16-bit I/Q WAV or raw float32, with the frequency and sample rate it was captured at |
| File → Import / Export Channels (CHIRP CSV) | Memory channels as CHIRP-compatible CSV (replace or append on import) |
| File → Save Screenshot | `Ctrl+P`; PNG of the whole window |
| File → Exit | `Ctrl+Q` (or the platform's Quit key); offers to save an unsaved recording first |
| Device → Connect... | Device dialog: detected RTL-SDR / HackRF One plus the Demo Device, and the sample rate |
| Device → Disconnect | Stop receiving and close the device |
| Device → Use Demo Device | Simulated signals, no hardware needed (see 8.7) |
| Device → Scan for Devices | List the SDRs plugged in (when none are found, also which drivers are not installed) |
| Radio → Start / Stop Receiving | `Space` |
| Radio → Record I/Q | `Ctrl+Shift+R`; the same recording as the toolbar and the control panel |
| Radio → Audio Output (toggle) | Same switch as the toolbar's Audio button; on by default; plays AM, FM, USB, LSB and CW through the default audio device while the channel is above squelch |
| Radio → Band Presets | FM Broadcast, NOAA Weather, 2m Ham, 70cm Ham, Airband AM, ADS-B, ISM 433, ISM 915: tunes and sets the mode, FM deviation and bandwidth |
| Radio → Enter Frequency | `Ctrl+L`; focus the Frequency field; Enter tunes and returns to the plots |
| Radio → Bookmark Current Frequency | `Ctrl+B`; saved with the current mode |
| View → Spectrum / Waterfall | Show or hide each plot (one always stays visible) |
| View → Panels | `Ctrl+1` … `Ctrl+7`: Decoder, Bookmarks, S-Meter, Ham ID, SSTV, QRP, Info |
| View → Focus Spectrum | `F6` |
| View → Theme → Dark / Light / Toggle Theme | `Ctrl+T` toggles; persists across launches |
| View → Reset Layout | Default panel sizes, both plots shown |
| Tools → Frequency Scanner | `Ctrl+F`; non-blocking sweep dialog with band presets, progress bar and hit table (Frequency, Peak Level, SNR); double-click or Tune to Signal tunes the receiver |
| Tools → Protocol Decoder | Brings the Decoder tab to the front |
| Tools → AM/FM Radio Tuner | `Ctrl+R`; pop-out broadcast tuner (Section 12) |
| Tools → Error History | `Ctrl+E`; last 500 warning/error log records, Copy All, Export, Clear |
| Help → Welcome and Quick Start | The first-run wizard again |
| Help → Keyboard Shortcuts | `F1`; every shortcut, grouped by menu and filterable |
| Help → About SDR Module | Version and feature summary |

Pressing Start with no device asks whether to connect one or start Demo
Mode. Starting a new recording, importing a file or closing the window with
an unsaved recording offers to save it first.

### 8.3 Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Space` | Start / stop receiving |
| `Ctrl+Shift+R` | Start / stop recording |
| `Ctrl+L` | Type a new center frequency (Enter tunes and returns to the plots) |
| `Ctrl+B` | Bookmark the current frequency |
| `←` / `→` | Tune down / up 10 kHz |
| `Shift+←` / `Shift+→` | Tune down / up 100 kHz |
| `Ctrl+←` / `Ctrl+→` | Tune down / up 1 MHz |
| `F6` | Put the keyboard focus on the spectrum |
| `Esc` | Return the keyboard focus to the spectrum |
| `Ctrl+1` … `Ctrl+7` | Show the Decoder, Bookmarks, S-Meter, Ham ID, SSTV, QRP or Info panel |
| `Ctrl+T` | Switch between the dark and light themes |
| `Ctrl+O` | Import a recording to convert its format |
| `Ctrl+S` | Save the recording |
| `Ctrl+P` | Save a screenshot |
| `Ctrl+Q` | Exit |
| `Ctrl+F` | Frequency scanner |
| `Ctrl+R` | AM/FM radio tuner |
| `Ctrl+E` | Error history |
| `F1` | Keyboard shortcut reference |
| Click on spectrum / waterfall | Tune to the clicked frequency |
| Drag on the waterfall | Measure bandwidth, duration and peak |
| Right-click on the waterfall | Tune, bookmark or copy that frequency; display options |
| `P` (waterfall focused) | Pause or resume the waterfall |

`Space` and the arrow keys act when the focused control does not use them
itself (a focused slider, list or text field keeps its usual keys); press
`F6` or `Esc`, or click a plot, first. Without the optional ham radio
panels the Info panel is `Ctrl+3`. In the Bookmarks list, `Enter` tunes,
`F2` renames and `Delete` removes.

### 8.4 Persisted state

Stored via `QSettings` (platform-specific backing store: Windows registry,
macOS plist, Linux `~/.config/SDR Module Team/SDR Module.conf`):

- Center frequency (100.1 MHz, in FM, when nothing is saved yet), RF gain,
  AGC, squelch threshold, channel bandwidth.
- Demodulation mode and FM deviation, recording format.
- Theme, audio output on/off and volume.
- The tool tab that was open, license class and Ham ID settings (callsign,
  CW speed and tone, automatic ID).
- Main window geometry and splitter sizes.
- Saved channels (frequency, label, demodulation mode and FM deviation, and
  CHIRP fields: mode, duplex/offset, tone, tuning step, skip, power,
  comment).
- First-run wizard completion flag.

### 8.5 First-run wizard

Launched automatically the first time the GUI starts, and again from
Help → Welcome and Quick Start. It says whether SDR hardware was found (and
how to install a missing driver), offers to connect the device it found or
start Demo Mode (simulated signals), and tunes to a chosen starting band in
that band's mode. It also shows a few quick tips.

### 8.6 Themes

Two themes, dark (the default) and light, switch live from View → Theme or
`Ctrl+T` and persist across launches. Every color comes from the palette
tokens in `sdr_module/gui/themes.py`: widgets are tagged with style roles
and tones (`set_role` / `set_tone`) instead of hard-coded colors, and the
custom-painted views (spectrum, waterfall axes, S-meter, radio tuner) read
the active palette and repaint when the theme changes. Keyboard focus is
always visible: a 2 px ring on buttons, check boxes and slider handles, an
accent outline on tabs, lists and text views, and a focus frame on the
plots. The waterfall colormaps are the same in both themes.

### 8.7 Demo mode

Started with `--demo`, Device → Use Demo Device, the Demo Device entry in
Device → Connect, or the first-run wizard. The simulated receiver produces
FM broadcast stations across 88–108 MHz, airband AM, 2 m and 70 cm FM,
NOAA weather, APRS and ISM bursts, decodable ADS-B frames at 1090 MHz
(flight KLM1023 with position, speed and squawk), a CW beacon
and weak carriers elsewhere on a realistic noise floor. The signals sit at
fixed frequencies and move across the display as you tune, so click-to-tune
and the frequency scanner work as with real hardware; raising the gain lifts
signals and noise together and too much gain clips, while AGC keeps the
signals clear of clipping. The strongest station, 100.1 MHz, carries a
looping tune in stereo FM, so listening in FM mode plays music. Samples
stream in real time for gap-free audio; a computer too slow to keep up gets
a status-bar hint and muted audio instead of a stutter. Starting the demo
from "None (I/Q)" switches the mode to FM.

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
| I/Q Sample Format | 8-bit unsigned, 16-bit signed, 32-bit float, 64-bit float |
| Byte Order | Little-endian (I, Q interleaved) |
| Raw File Format | .raw, .cu8, .cs8, .cs16, .cf32, .cf64 |
| Metadata Format | SigMF (Signal Metadata Format) |
| Audio Export | WAV (PCM 16-bit, 44.1/48 kHz) |
| Memory Channels | CHIRP generic CSV (21-column header; MHz frequencies; duplex/offset, CTCSS/DTCS, mode, tuning step, skip, power, comment) |

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
| GUI Panel | Ham Radio → Ham ID tab: callsign input, CW speed and tone, Preview ID, Send ID Now (HackRF One only), countdown to the next automatic ID while transmitting |

### 11.6 SSTV Image Reception (ISS/Space)

The software includes an SSTV (Slow Scan Television) decoder for receiving images from the International Space Station and other amateur sources.

| Feature | Description |
|---------|-------------|
| ISS SSTV Preset | Pre-configured 145.800 MHz with FM demodulation |
| Supported Modes | PD120, PD180 (ISS favorites), Robot, Martin, Scottie |
| Live Preview | Real-time image display as lines are decoded |
| Auto-Save | Automatic saving of completed images |
| Image History | Browse previously received images |
| VIS Detection | Automatic mode detection from VIS header |
| GUI Panel | Ham Radio → SSTV tab (Start Decoder); decodes the receiver's FM audio (Demodulation → Mode: FM), whether or not the speaker is on |

**Space Frequency Presets:**

| Preset | Frequency | Description |
|--------|-----------|-------------|
| ISS SSTV/Voice | 145.800 MHz | Slow Scan TV and voice downlink |
| ISS APRS | 145.825 MHz | Packet radio / APRS digipeater |
| ISS Packet | 437.550 MHz | UHF packet downlink |
| Meteor-M2 LRPT | 137.100 MHz | Russian weather satellite |
| SO-50 | 436.795 MHz | Amateur satellite |

### 11.7 HAM Radio Signal Meter (S-Units / RST)

Classic signal strength reporting using formats every amateur radio operator knows.

**S-Meter Scale (IARU Region 1, 50Ω):**

| S-Unit | dBm | Description |
|--------|-----|-------------|
| S1 | -121 | Barely perceptible |
| S3 | -109 | Weak |
| S5 | -97 | Moderate |
| S7 | -85 | Good |
| S9 | -73 | Very strong (reference) |
| S9+20 | -53 | Extremely strong |
| S9+40 | -33 | Full scale |

**RST Reporting System:**

| Code | Meaning |
|------|---------|
| R (1-5) | Readability: 1=unreadable, 5=perfect |
| S (1-9) | Strength: corresponds to S-meter |
| T (1-9) | Tone (CW only): 9=perfect tone |

**Example Reports:**

| Report | Meaning |
|--------|---------|
| "59" | Perfectly readable, very strong (phone) |
| "599" | Perfect readability, strength, tone (CW) |
| "Five and nine, twenty over" | S9+20 dB |
| "57" | Readable, moderate signal |

### 11.8 QRP (Low Power) Operations

Support for QRP (5W CW / 10W SSB) and QRPp (milliwatt) operation.

**Power Classifications:**

| Class | CW Power | SSB Power | Description |
|-------|----------|-----------|-------------|
| QRPp | < 1W | < 1W | Milliwatt operation |
| QRP | ≤ 5W | ≤ 10W | Standard QRP |
| Low Power | ≤ 100W | ≤ 100W | Reduced power |
| QRO | > 100W | > 100W | Full power |

**QRP Calling Frequencies:**

| Band | CW | SSB |
|------|-----|-----|
| 80m | 3.560 MHz | - |
| 40m | 7.030 MHz | 7.285 MHz |
| 30m | 10.106 MHz | - |
| 20m | 14.060 MHz | 14.285 MHz |
| 15m | 21.060 MHz | 21.385 MHz |
| 10m | 28.060 MHz | 28.360 MHz |

**QRP Features:**

| Feature | Description |
|---------|-------------|
| Power Display | Shows watts, mW, and dBm simultaneously |
| TX Power Limiter | Configurable limit for QRP compliance |
| Amplifier Calculator | HackRF → Driver → PA chain calculation |
| Miles-per-Watt | Track QSO distances and efficiency |
| Contest Exchange | Format/parse RST + power exchanges |

**HackRF QRP Chain Example:**

```
HackRF (0 dBm, 1mW) → Driver (+20dB) → PA (+17dB) = 5W QRP
```

### 11.9 License Profiles

TX permissions are enforced based on the operator's amateur radio license class.

**License Classes:**

| Class | Description | HF Privileges | VHF/UHF |
|-------|-------------|---------------|---------|
| None | No license | License-free only | License-free only |
| Technician | Entry level | 10m, limited CW on 80/40/15m | Full |
| General | Intermediate | Most HF with sub-band limits | Full |
| Amateur Extra | Full privileges | All amateur bands | Full |

**License-Free Bands (No License Required):**

| Service | Frequency | Power Limit | Modes |
|---------|-----------|-------------|-------|
| CB Radio | 26.965-27.405 MHz | 4W AM, 12W SSB | AM, SSB |
| MURS | 151.82-154.60 MHz | 2W | FM |
| FRS | 462-467 MHz | 0.5-2W | FM |

**Amateur Band Privileges (Examples):**

| Band | Technician | General | Extra |
|------|------------|---------|-------|
| 160m | ✓ | ✓ | ✓ |
| 80m | CW only (3.525-3.6) | Phone 3.8-4.0, CW 3.525-3.6 | Full |
| 40m | CW only (7.025-7.125) | Phone 7.175-7.3, CW 7.025-7.125 | Full |
| 20m | ✗ | Phone 14.225-14.35, CW 14.025-14.15 | Full |
| 10m | ✓ (28.0-28.5) | ✓ | ✓ |
| 6m | ✓ | ✓ | ✓ |
| 2m | ✓ | ✓ | ✓ |
| 70cm | ✓ | ✓ | ✓ |

**TX Validation:**
- All TX requests are validated against current license class
- Hardware lockouts (GPS, aviation, emergency) always apply regardless of license
- Mode restrictions enforced per band segment
- Power limits enforced where applicable

**Power Headroom (150%):**

Power limits allow 150% of the legal limit to account for:
- Cable and connector losses
- Filter insertion loss
- Amplifier efficiency variations
- Measurement uncertainty

| Band | Legal Limit | Effective Max |
|------|-------------|---------------|
| CB | 12W | 18W |
| 10m Tech | 200W | 300W |
| 30m | 200W | 300W |
| 60m | 100W | 150W |

⚠️ **IMPORTANT**: Before transmitting, test actual broadcast power with a 50Ω dummy load and power meter. The configured limit is not measured output—verify radiated power at the antenna base.

---

## 12. AM/FM Radio Tuner

The software includes a car-radio-style AM/FM tuner for broadcast radio: a
pop-out window (Tools → AM/FM Radio Tuner, `Ctrl+R`) that tunes the main
receiver. Audio plays through the main window, with its toolbar Audio button
and volume slider.

### 12.1 Frequency Bands

| Band | Frequency Range | Step Size | Modulation |
|------|-----------------|-----------|------------|
| AM | 530 kHz - 1700 kHz | 10 kHz | Amplitude Modulation |
| FM | 87.5 MHz - 108 MHz | 100 kHz | Frequency Modulation |

### 12.2 User Interface Features

| Feature | Description |
|---------|-------------|
| **Frequency Display** | LCD-style readout: band badge, large frequency digits and a station line (the matching preset, or the band's range) |
| **Tuning Dial** | Slide-rule scale with dots marking the band's presets; drag or click to tune; mouse wheel or `←`/`→` step one channel, `Page Up`/`Page Down` ten, `Home`/`End` jump to the band edges |
| **Step Buttons** | ◀ / ▶ either side of the dial tune one channel; hold to keep tuning |
| **AM/FM Buttons** | Switch bands; each band returns to the last station tuned on it |
| **Preset Buttons** | 6 station presets per band (12 total); click to tune; right-click or press and hold to store the current station |
| **Receiver Link** | Tuning here retunes the main receiver and sets FM (75 kHz deviation, 200 kHz bandwidth) or AM (10 kHz bandwidth). The tuner opens on the receiver's frequency and follows it; while the receiver is outside both broadcast bands the display dims and shows the receiver's frequency instead of a station |

Stored presets are kept until SDR Module closes; they are not saved between
launches.

### 12.3 Default Presets

**FM Presets (example):**

| Preset | Frequency | Label |
|--------|-----------|-------|
| 1 | 101.1 MHz | Rock |
| 2 | 93.3 MHz | Classic |
| 3 | 97.1 MHz | Pop |
| 4 | 104.3 MHz | Jazz |
| 5 | 88.5 MHz | NPR |
| 6 | 99.5 MHz | Country |

**AM Presets (example):**

| Preset | Frequency | Label |
|--------|-----------|-------|
| 1 | 880 kHz | News |
| 2 | 1010 kHz | Talk |
| 3 | 770 kHz | Sports |
| 4 | 1050 kHz | Weather |
| 5 | 660 kHz | News2 |
| 6 | 1260 kHz | Oldies |

### 12.4 Styling

The tuner is laid out like a car radio: band buttons beside the frequency
display, the dial between its step buttons, and two rows of preset buttons.
It is drawn entirely from the active theme palette (Section 8.6), so it
follows the dark or light theme and switches live:
- The display uses the theme's LCD colors (the same as the toolbar's FREQ readout)
- The dial uses the plot colors, with the tuned-frequency marker color for its pointer
- The selected band and the preset on the tuned station are accent-filled

### 12.5 Integration

```python
from sdr_module.ham.gui.radio_tuner import RadioTunerWidget, show_radio_tuner

# Launch as standalone window
tuner = show_radio_tuner(sample_rate=2.4e6)

# Or create it for an existing window (it is a pop-out dialog)
tuner = RadioTunerWidget(parent=main_window, sample_rate=2.4e6)
tuner.frequency_changed.connect(on_station)  # (freq_hz, "AM" or "FM")
tuner.set_frequency(98.3e6)  # follow the receiver (does not emit)

# Process samples through tuner
audio = tuner.process_samples(iq_samples)  # scaled by set_volume() / set_muted()

# Get current frequency
freq = tuner.get_frequency()
band = tuner.get_band()  # RadioBand.AM or RadioBand.FM

# Show the STEREO badge on the display (no stereo detection is built in)
tuner.set_stereo(True)
```

---

## 13. Future Considerations

| Feature | Description |
|---------|-------------|
| Remote Operation | Network-based SDR control |
| Database Integration | Signal/protocol signature database |
| Machine Learning | AI-based signal classification |
| Multi-SDR Support | Multiple receiver operation |
| Spectral Signature Library | Community-sourced signal database |

---

## 14. Development Phases

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
- [x] Advanced protocols (ADS-B, ACARS, FLEX, AX.25/APRS, RDS, POCSAG)
- [x] Main GUI application (PyQt6-based with spectrum, waterfall, controls, decoder panels)

---

## 15. Windows Build & Installation

### 15.1 Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or higher |
| pip | Python package manager |
| Inno Setup | 6.x (optional, for creating installer) |

### 15.2 Build Files

| File | Purpose |
|------|---------|
| `build_windows.bat` | Batch script for building Windows executable |
| `build_windows.ps1` | PowerShell script (alternative) |
| `build_installer.bat` | Creates Windows installer with Inno Setup |
| `sdr_module.spec` | PyInstaller specification file |
| `installer.iss` | Inno Setup installer script |

### 15.3 Quick Build (Command Prompt)

```batch
REM Basic build
build_windows.bat

REM Full clean build
build_windows.bat --clean
```

### 15.4 Quick Build (PowerShell)

```powershell
# Basic build
.\build_windows.ps1

# Full clean build with installer
.\build_windows.ps1 -Clean -CreateInstaller
```

### 15.5 Build Options

| Option | Batch | PowerShell | Description |
|--------|-------|------------|-------------|
| Clean | `--clean` | `-Clean` | Remove build directories before building |
| No GUI | `--no-gui` | `-NoGUI` | Build the CLI only (skip PyQt6 GUI) |
| No UPX | `--no-upx` | `-NoUPX` | Disable UPX compression |
| Installer | N/A | `-CreateInstaller` | Create Windows installer |

### 15.6 Creating the Installer

After building the executable:

```batch
build_installer.bat
```

Or with PowerShell:
```powershell
.\build_windows.ps1 -CreateInstaller
```

### 15.7 Output Locations

| Output | Location |
|--------|----------|
| Executable | `dist\sdr-module\sdr-scan.exe` |
| Installer | `installer_output\SDR-Module-0.2.0-Setup.exe` (version injected by the build) |

### 15.8 Manual Installation

If not using the installer:

1. Copy the `dist\sdr-module` folder to your preferred location
2. Optionally add the folder to your system PATH
3. Run `sdr-scan.exe --help` to verify installation

---

*Document Version: 4.3*
*Last Updated: 2026-09-26*

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-25 | Initial specification document |
| 2.0 | 2025-12-25 | Added quantitative RF specs, hardware compatibility matrix, interface specs, physical/environmental specs, compliance section |
| 3.0 | 2025-12-25 | Tuned for dual-SDR setup (RTL-SDR + HackRF One); added device-specific specs, dual-SDR operation modes, synchronization, use cases, software stack |
| 3.1 | 2025-12-26 | Added Windows build and installation documentation (Section 15) |
| 3.2 | 2025-12-26 | Implemented plugin system architecture; updated development phases to reflect current implementation status |
| 3.3 | 2025-12-26 | Added advanced protocol decoders: ADS-B, ACARS, FLEX; verified AX.25/APRS, RDS, POCSAG implementations |
| 3.4 | 2025-12-26 | Implemented Main GUI application with PyQt6: spectrum analyzer, waterfall display, control panel, protocol decoder panel, recording controls, device dialog |
| 3.5 | 2025-12-26 | Added TX frequency lockouts for safety (GPS, aviation, emergency, cellular); HAM radio callsign identification; RX presets with GUI selector |
| 3.6 | 2025-12-26 | Added SSTV decoder for ISS image reception; space/satellite RX presets (ISS, Meteor-M2, SO-50); GUI image viewer with live preview |
| 3.7 | 2025-12-26 | Added HAM radio signal meter with S-units (S1-S9, S9+dB) and RST reporting; analog meter GUI; verbal reports ("five and nine, twenty over") |
| 3.8 | 2025-12-26 | Added QRP operations module: power conversion (dBm↔watts), TX limiter for QRP compliance, amplifier chain calculator, miles-per-watt tracker; QRP calling frequency presets (80m-10m CW/SSB) |
| 3.9 | 2025-12-26 | Added license profiles (None, Technician, General, Amateur Extra); TX lockouts enforced by license class; license-free bands (CB, MURS, FRS); GUI license selector |
| 4.0 | 2025-12-26 | Added 150% power headroom for TX limits (accounts for cable/filter losses); dummy load testing warning; shows legal vs effective power limits in GUI |
| 4.1 | 2025-12-26 | Removed NatLangChain blockchain radio protocol integration |
| 4.2 | 2026-01-01 | Added AM/FM Radio Tuner documentation (Section 12); vintage car radio-style interface with presets, tuning dial, and volume controls |
| 4.3 | 2026-09-26 | Section 8 rewritten for the reworked GUI: window layout, Radio menu, tool tabs (Decoder, Bookmarks, Ham Radio, Info), full shortcut list, status bar, persisted state, themes and demo mode; SigMF recording; Section 12 matched to the current radio tuner (volume, tone, balance, seek and power controls removed; audio via the main window) |
