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
| Real-time Spectrum Display | FFT of each display frame (1024–32768 points; Hann, Blackman-Harris or Flat-top window) in dBFS: every sample up to 15 MS/s, at 20 MS/s half of them in evenly spread pieces. With peak and min hold, averaging, an automatic or fixed vertical range, zoom (×2–×16) and bookmark labels |
| Waterfall Display | Signal history from live to 1 hour; each row keeps the frequency it was received on, so old rows stay true after a retune; drag to measure bandwidth, duration, peak and SNR |
| Time Domain View | Signal tab → Scope: the tuned channel's magnitude, I and Q, phase or instantaneous frequency over 1, 5, 20 or 100 ms |
| Constellation Diagram | Signal tab → Scope → Constellation, with a BPSK, QPSK, 8PSK or 16QAM overlay and the EVM (the points are the channel's samples, not symbol-timed) |
| Signal Strength Meter | The toolbar's LEVEL meter (channel power in dBFS, noise floor, squelch state) and Ham → S-Meter (S-units, RST, SNR) |
| Signal List and Bursts | Signal tab → Signals (the signals in view) and Bursts (transmissions that come and go, not stations that never stop; on/off keyed ones taken apart into pulses and bits) |

### 2.2 Frequency Tuning & Analysis

| Feature | Description |
|---------|-------------|
| Center Frequency Selection | The tuned frequency moves within the span and the receiver retunes only near its edges (Radio → Keep Tuned Frequency Centered retunes every time). Type a frequency with its unit (`Ctrl+L`), step by the tuning step with the arrow keys or the mouse wheel, scroll a digit of FREQ, go back and forward through the frequencies tuned. Below 2 MHz frequencies read in kHz ("880 kHz", also in the Signal and Scanner tabs' messages; their lists' frequency columns, the Decoder tab's status line and transmit messages stay in MHz), and near 0 Hz the span starts at 0 Hz. A frequency the device can't tune (or one below 0 Hz) is refused with the reason, and so is one outside the span while an I/Q recording has samples (5.2) |
| Listening by Band Plan | Radio → Choose Mode From Band Plan (on by default): a typed frequency, or a bookmark or recent frequency without a mode of its own, in another band or another part of a ham band is heard the way it usually is, and the status bar says so ("40m Ham, CW/data segment: listening in CW with a 500 Hz filter ..."). A preset channel its preset's way; the FT8 and FT4 channels in USB (3 kHz filter); the CW/data segments of the HF ham bands and 6 m (40 m below 7.125 MHz, 20 m below 14.15 MHz ...) in CW (500 Hz); elsewhere the band's own mode, e.g. LSB on 160, 80 and 40 m, USB on 60 m and from 20 m up, CW on 30 m, narrow FM on 2 m, 70 cm, MURS and FRS/GMRS, AM on the airband. USB is kept on the LSB bands and in the CW/data segments, 30 m included (FT8, RTTY and other data are sent in USB), CW is kept where the segment lists CW (not on an FT8 or FT4 channel), and presets and bookmarks with a mode keep theirs |
| Frequency Locking | A click on the spectrum or waterfall lands on the signal's peak or center; Signal → Identify → Apply Mode centers the tuned frequency on the signal |
| Bandwidth Selection | A filter per mode: USB/LSB 1.8–3.6 kHz, CW 100 Hz–1 kHz, AM 5–15 kHz, FM 12.5–200 kHz, None (I/Q) 10 kHz–2.4 MHz; each mode remembers its own |
| Automatic Frequency Control (AFC) | Radio → Track Signal (AFC): the tuned frequency follows a drifting narrow FM, AM or CW signal (not SSB, I/Q or broadcast FM) |
| Frequency Scanning | The Scanner tab sweeps a band once or continuously, stops on signals, or scans the bookmarks, while you keep listening; `sdr-scan scan` sweeps a connected RTL-SDR or HackRF One (or a recording) from the command line |
| Signal Identification | Signal → Identify (`Ctrl+I`): what the signal is, how sure, and the mode, deviation and filter to listen with (an FM deviation that doesn't clip the station) |

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

Decoded live in the Decoder tab, from a whole file with File → Decode
Recording, and from the command line with `sdr-scan decode` (`pocsag`,
`flex`, `ax25`/`aprs`, `adsb`, `acars`, `rds`, `cw`). Each decoder picks out
its own channel at the tuned frequency, whatever the Mode and Filter are
(ADS-B reads the pulses across the whole span).

| Category | Protocols | Status |
|----------|-----------|--------|
| Paging | POCSAG (512, 1200 and 2400 baud, either polarity), FLEX (1600 bps) | Implemented. FLEX at 3200 and 6400 bps is recognized and reported as not supported |
| Amateur Radio | AX.25/APRS (1200 baud AFSK) | Implemented; APRS positions, messages, status and objects are read |
| Aviation | ADS-B (1090 MHz), ACARS (2400 bit/s MSK on AM) | Implemented. ADS-B needs at least 2 MS/s (2.4 MS/s or more recommended); the Aircraft tab lists one row per aircraft |
| Broadcast | RDS (FM Radio Data System) | Implemented: station name, RadioText, program type, clock, traffic flags, alternative frequencies; the name also shows beside FREQ (Radio → RDS Station Names) and on the radio tuner |
| Morse Code | CW (Morse), 5–60 WPM | Implemented: the text and its speed in the Decoder's Text tab; tune the carrier to within about 200 Hz, as CW mode hears it |
| Slow-Scan TV | SSTV (Robot, Martin, Scottie, PD, Wraase SC2-180) | Implemented in Ham → SSTV (Section 11.6) |
| ISM Band (433, 868, 915 MHz) | Sensors, remote controls and other on/off keyed (OOK) or FSK devices | Analyzed, not decoded: Signal tab → Bursts and Identify give the modulation (pulse width, pulse position or PCM), the pulse and gap widths and the bits in hex, laid out like `rtl_433 -A` with an rtl_433 flex decoder line. No device is decoded by brand or model (no temperatures, no button names) |
| Text Modes | CW, RTTY, PSK31, ASCII FSK | Sent with Tools → Transmit Text (HackRF One, or saved as audio or I/Q) and written to a file by `sdr-scan encode`. RTTY, PSK31 and ASCII FSK are not decoded; CW is (Morse Code above) |

Decoder → Protocol: Auto picks the protocol from where you are tuned: ADS-B
on 1090 MHz, APRS on 144.390 / 144.800 MHz, ACARS on 129–137 MHz, POCSAG on
152–159 MHz, FLEX on 929–932 MHz, RDS on broadcast FM, Morse code in CW mode.

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
| I/Q Recording | Radio → Record I/Q (`Ctrl+Shift+R`), or Record in the toolbar or the Recording group: the raw samples, kept in a temporary file until saved (or saved automatically to the recordings folder), with pause and resume. A recording holds one device, one sample rate and one center frequency: the rate can't be changed while recording, and once it has samples the receiver doesn't retune (tuning moves within the span; a frequency outside it is refused: "145.800 MHz is outside the I/Q recording's span (98.9–101.3 MHz): stop recording to retune there."); Device → Connect..., Use Demo Device and Disconnect ask before they end it. While the display can't keep up, a take still gets every sample, from the demo device as from hardware; signal lost all the same (a long freeze, a disk too slow) is shown on the take line ("· 2.1 s lost") and in a warning. It stops when less than 500 MB is free (sizes are decimal MB and GB) |
| Audio Recording | Radio → Record Audio (`Ctrl+Shift+A`): what you hear as a 48 kHz 16-bit mono WAV, silent while the squelch is closed so the file keeps time |
| Saved Takes | A take saved as SigMF records the device, where it was tuned ("Tuned to 145.290 MHz (FM)"; a take of a recording playing says it was played back from that file) and its start to the millisecond. Saving a quiet take as cu8 or cs8 asks first, since 8 bits keep little of a weak signal |
| Playback | File → Open Recording (`Ctrl+O`) plays a file through the receiver (spectrum, waterfall, audio, decoders, both recorders) in real time, read from disk as it plays, with a playback strip (position, Loop, 0.5x–4x; audio at 1x only). It is heard the way its band usually is (Radio → Choose Mode From Band Plan: a 2 m recording opened from broadcast FM plays in narrow FM). Tune anywhere inside the recording's span. Every sample is read, however slowly the file plays, so the decoders miss nothing; on a computer that can't keep up it still plays as near to 1x (or the slower speed chosen) as the computer manages, and faster than 1x the status bar says when it plays much slower than chosen ("Playing at about 1.2x, not 4x ..."). A decoder that needs a higher sample rate than the recording's says so, as Decode Recording does ("ADS-B needs at least 2 MS/s; this recording is 1 MS/s"). Close Recording goes back to the device, frequency, mode, FM deviation and filter from before, and says in the same message when going back turned off a decoder that a preset turned on while the file played |
| Convert | File → Convert Recording (`Ctrl+Shift+O`) loads a file in the background and saves it in another format |
| Decode a Recording | File → Decode Recording runs any of the seven live decoders over a whole file, faster than real time (a protocol the recording's rate is too low for, such as ADS-B below 2 MS/s, is grayed out); its rows are marked "(file)", with the recording's own time when it has one, and stay out of the live Aircraft and Text views. `sdr-scan decode` does the same from the command line |
| Inspect a Recording | File → Recording Info (format, rate, frequency, span, length, the hardware it was made with, and whether the samples look right, with their peak in dBFS) and File → Find Signals in Recording (the signals in it and when they were heard) |
| File Formats | Saved as SigMF, cf32, cs16, 16-bit I/Q WAV, raw float32, cu8 (rtl_sdr) or cs8 (hackrf_transfer); opened from all of these and cf64. Files without a header (.bin, .iq, .dat, .raw ...) get their format, rate and frequency from the Import Options dialog, prefilled from the extension, the data and the file name (a quiet 16-bit capture is taken for cs16, not cs8; a center up to 300 GHz, for a capture through a downconverter). The sample check says when the format chosen is probably wrong, the samples clip or the file is silent. Raw and WAV files are saved with a SigMF metadata file beside them (`take.cf32.sigmf-meta`), so their rate and frequency are kept |
| Limits | A WAV file holds at most 4 GiB (about 7 minutes at 2.4 MS/s; checked before the free space). One-channel (audio) WAV files, 24- and 32-bit WAV files and big-endian or real-valued SigMF data can't be played |

### 5.3 Demodulation

| Feature | Description |
|---------|-------------|
| Multi-mode Demod | None (I/Q, no audio), AM, FM, USB, LSB and CW. FM is broadcast (75 kHz deviation, 75 µs de-emphasis) in 87.5–108 MHz and narrow (5 kHz) elsewhere, with 2.5, 12.5 and 25 kHz to choose; at 2.5 and 5 kHz the voice audio is cut off above about 3 kHz, so a weak station doesn't hiss above the voice. CW has a 400–1000 Hz pitch. In USB, LSB and CW the filter is the audio filter (USB at 2.7 kHz hears 200–2900 Hz) |
| Squelch | Level squelch with Auto (7 dB above the channel's own noise); it opens at the threshold and closes only after 0.3 s at least 3 dB below it (0.5 s in CW). CTCSS tone squelch with the 50 standard tones (no DCS), in FM up to 25 kHz deviation: at a broadcast deviation it isn't used and Tone heard reads "Not checked" |
| Audio Processing | Noise reduction (Off, Light, Strong; it leaves an FM or AM station 30 dB or more above its noise alone), audio AGC (AM, SSB, CW), DC blocking in FM, carrier and DC-spike rejection in SSB. AM transmissions fade in and out without a pop or thump |
| Audio Output | Mono, 48 kHz, to the output chosen in Tools → Preferences (the system default otherwise); Mute and volume in Radio → Audio. Broadcast stereo is not decoded (the radio tuner's STEREO lamp only shows that a pilot is received) |
| Data Output | The Decoder tab's live decoders (CSV export), the RDS station name, SSTV pictures, and the Signal tab's identification and burst analysis |

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
| FFT Size | 1024 / 2048 / 4096 / 8192 / 16384 / 32768 points (the display: spectrum ⋯ → FFT Size, 2048 by default) |
| FFT Window Functions | Rectangular, Hann, Hamming, Blackman, Blackman-Harris, Flat-top (library); the display offers Hann, Blackman-Harris and Flat-top |
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

The modes of the library's `DualSDRController` (`sdr_module.core.dual_sdr`),
and what the GUI offers for each:

| Mode | RTL-SDR | HackRF One | Use Case | Status |
|------|---------|------------|----------|--------|
| **Dual RX** | RX @ Freq A | RX @ Freq B | Monitor two bands at once, or compare two receivers on one | Implemented. GUI: Device → Second Receiver... (`Ctrl+Shift+N`) runs a second receiver window on any RTL-SDR or HackRF One the main window isn't using, or on a second demo device |
| **Full Duplex** | RX @ Freq A | TX @ Freq B | Transceiver with simultaneous RX monitoring | Implemented. GUI: the second receiver keeps receiving while the main window's HackRF One transmits (the main window's own receiver pauses: the HackRF is half duplex) |
| **TX + Monitor** | RX @ TX Freq | TX | Monitor own transmission quality | Implemented. GUI: Second Receiver → Transmit monitor → Monitor TX tunes it to the transmission in its mode; Follow transmissions does that for every transmission |
| **Wideband Scan** | Scan 0-1.7 GHz | Scan 1.7-6 GHz | Cover full spectrum faster | **Not implemented**: `set_mode` refuses it (`NotImplementedError`). Scan with one device at a time: the Scanner tab, or `sdr-scan scan --device rtlsdr` and `--device hackrf` |
| **Signal Relay** | RX Input | TX Output | Receive-and-retransmit applications | **Not implemented** (refused): retransmitting what is received needs checks the controller doesn't make |

Without hardware, `DualSDRController.use_devices()` takes devices that are
already open, and `sdr_module.devices.polled.PolledDevice` lets the demo
device (`MockDevice`) stand in for either receiver. Full Duplex and TX +
Monitor refuse a "HackRF" that can't transmit and leave the mode and the
receiver's frequency as they were.

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
| FM Demodulation | N/A | 20 MS/s real-time | The HackRF's top rate, with the plots and audio running; the channel filter computes only the samples it keeps |
| AM Demodulation | N/A | 20 MS/s real-time | Envelope detection is lightweight |
| Protocol Decoding | N/A | Varies | Bit-level processing, lower throughput |
| Filter (FIR, 101 taps) | N/A | ~5 MS/s | NumPy convolution |

**Key constraints:**
- The RTL-SDR's 2.4 MHz bandwidth is generally processable in real-time
- At the HackRF's 20 MS/s, listening keeps up on a typical 4-core computer (a 33 ms frame of broadcast FM takes about 15 ms of processor time, 11–12 ms of it on the display thread and the rest on the plots' helper thread; the plots analyze half the samples); with more running at once it may not, and the app says so (the Audio button reads Paused; the Decoder says it misses part of the signal; when the app's own Decode Recording, Convert Recording, Save Recording or Find Signals is what keeps the computer busy, both name that job)
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

> **Note**: Retransmitting what is received (the repeater and transponder
> rows) needs the Signal Relay mode, which is not implemented (6.4). What the
> app transmits today is a station ID or text from Tools → Transmit Text,
> through the checks in 11.2.

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
+----------------------------------------------------------------------------+
| File  Device  Radio  View  Tools  Help                                     |
| [Start] [Record] (ON AIR) | FREQ 100.100 MHz SDR DEMO | LEVEL ==|- OPEN    |
| MODE [FM] | FILTER [200 kHz] | STEP [100 kHz] | [Audio] --o-- volume       |
+--------------------------------------------+-------------------------------+
| SPECTRUM           Peak hold  AVG  ...     | control panel (scrolls):      |
| RBW 1.76 kHz                               |  Frequency, Demodulation,     |
|                                            |  Audio, Presets,              |
|--------------------------------------------|  Receiver, Recording          |
| WATERFALL ?  Pause  HISTORY  COLORS  ...   |-------------------------------|
|                                            | Decoder  Bookmarks  Ham       |
|                                            | Scanner  Signal  Info    ^^   |
| (playback strip while a recording plays)   |                               |
+--------------------------------------------+-------------------------------+
| RUNNING  Demo Device  2.4 MS/s  messages...  REC 00:01:23  ADS-B · 37 msgs |
+----------------------------------------------------------------------------+
```

The window first opens at 1400 × 900, less on a smaller screen (at most 95%
of its width and 92% of its height: about 1300 × 670 on a 1366 × 768 laptop
with a 40 px taskbar); its minimum, 1024 × 640, fits a 1366 × 768 laptop.
The three splitters (plots | right column, spectrum / waterfall, control
panel / tool tabs) are resizable and remembered; View → Reset Layout restores
them and shows both plots. The right column takes 30% of the window's
width, at least 400 px and at most 660 px (406 px at 1366 px, about 570 px
at 1920 px; at 660 px the Decoder's Aircraft list shows all ten columns
without scrolling sideways), and keeps that share as the window is resized:
a width you drag it to is kept as a share too, a layout saved on one screen
opens with the same share on another, and Reset Layout goes back to 30%.
The control panel and the tool tabs scroll instead of being crushed. The
control panel puts how you listen first: in a window at least 678 px tall
the default split shows Mode, Filter, FM deviation and Squelch; in a shorter
one, such as the window that first opens on a 1366 × 768 laptop with a 40 px
taskbar, the tool tabs keep their minimum height, so the default split falls
after the Frequency group and the rest of the panel is a scroll away (View →
Reset Layout once the window is at least that tall shows them). When
the window's height changes, the split between the control panel and the
tool tabs keeps the panel's share of the column (the default's, or where you
dragged it) and settles in the nearest gap between two groups, so no group
is cut in half. View → Maximize Tool Tabs, the double-chevron button at the
right of the tab bar or a double-click on any tab gives the tool tabs the
whole right column (remembered; `Ctrl+L` brings the control panel back).
The window title says where the receiver is:
`● REC · 100.100 MHz FM — Demo Mode — SDR Module`.

| Component | Description |
|-----------|-------------|
| Toolbar | **Start/Stop** (Play/Pause while a recording plays; off while transmitting, its tooltip saying whether receiving resumes when the transmission ends); **Record** (red while recording; it records I/Q or audio, as the Recording group's Record choice says); **ON AIR** (red, only while transmitting; click to stop); **FREQ**, the tuned frequency in a fixed width (click it or press `Ctrl+L` to type one; scroll over a digit to tune that digit; its tooltip gives the band plan there), with the station's RDS name beside it while a broadcast station sends one (hover for the RadioText, program type and PI code); **LEVEL**, a meter of the tuned channel's power in dBFS with a tick at the noise floor, the squelch threshold as a marker (press or drag on the bar to set it) and a lamp: OPEN, SQL (squelched), TONE (a signal without the chosen CTCSS tone), OFF (squelch off) or I/Q (no audio) |
| Receiver bar | The second toolbar row: **MODE** and **FILTER** (the same controls as the control panel's Mode and Filter); **STEP** (10 Hz … 1 MHz, remembered per mode), what `←`/`→`, the mouse wheel over the plots and the Frequency field's arrows tune by; the **Audio** button and the volume slider. The Audio button reads Audio (playing), Muted, Muted (2nd RX) (the second receiver's Listen plays only that receiver; unmuting then says to choose Main window or Both there), Paused (amber: the signal received isn't real time, because this computer can't keep up or a recording plays faster or slower than 1x; the tooltip says why, the status bar at most every 30 s), No output (no audio device: a click tries again or opens Preferences on the output choice) or No audio (None (I/Q) has no audio). When the output chosen in Preferences is unplugged, the status bar says once which output plays instead |
| Spectrum | Each display frame is analyzed (every sample up to 15 MS/s; at 20 MS/s half of them, in evenly spread pieces that include the newest, so a burst shorter than the gap between pieces, 0.1 ms at 2048 points, may not show); the trace is the mean power in dBFS. Header: Peak hold with Reset (which also clears Min hold), AVG (Off, 2–32 frames) and ⋯ (display options); above the plot the resolution and zoom ("RBW 1.76 kHz · ×4"). FFT size 1024–32768 points; window Hann, Blackman-Harris or Flat-top. Range: Auto (follows the noise floor and the strongest signal in 5 dB steps), 60/80/100/120 dB or Top at 0 dBFS; drag the dB labels to move the range, scroll over them to move it 5 dB (`Shift`: 10 dB shallower or deeper), double-click them for Auto. Bookmarks in view are labeled along the top (click one to tune it). A dashed marker and a shaded band show the tuned channel; the hover readout gives the frequency and level under the pointer, and where a click tunes when it snaps to a signal ("Click to tune to 99.500 MHz"). Click tunes the signal near the pointer (`Alt`+click: exactly there), the wheel tunes by STEP (`Shift`: ten steps), `Ctrl`+wheel zooms (×2–×16) around the pointer. Right-click: Tune to, Bookmark, Copy Frequency, Zoom In Here and Identify Signal Here at that frequency, then Peak Hold, Min Hold, Reset Holds, Range, Zoom, FFT Size, Window and Bookmark Labels |
| Waterfall | Newest row at the top with a time axis (rows from before a Stop are labeled too). Each row keeps the strongest level per bin and the frequency it was received on, so after a retune the older rows move with their signals ("Not received then" where a row has no data); every row's noise is drawn equally bright (after a stall, in a 1 or 5 min history, on a busy computer and once it catches up), so only a real change of the noise floor shows. Header: **?** (the usage hint), Pause (`P`), HISTORY (Live, 1 min, 5 min, 15 min, 1 h; a long history first fills at the live rate), the COLORS scale in dBFS (drag it; Auto by default) and ⋯. Hover readout: the frequency and level under the pointer, SNR over the noise floor, age, and where a click tunes when it snaps to a signal. Click tunes; dragging measures a box (Δf, Δt, center, peak, SNR); `Esc` clears it. Right-click: Tune to, Bookmark, Copy Frequency, Zoom In Here, Identify Signal Here; Tune to Measured Center, Copy Measurement, Clear Measurement; Pause, History, Color Levels (Auto, 60–120 dB below full scale, Fit to Current Signals; the one in use is checked), Colors (Viridis, Plasma, Turbo (the default), Grayscale, Classic), Zoom, Clear History, Save Image... (PNG, in the screenshots folder, then the last one used, with a dated name). It zooms with the spectrum and draws its own frequency axis when the spectrum is hidden; while the Signal tab's Bursts page is shown, the selected burst is outlined on it. Both plots dim and say why when nothing is being received ("Stopped - press Space"; while transmitting, "Transmitting - receiving resumes when it ends") |
| Playback strip | Under the plots while a recording plays (File → Open Recording): Play/Pause, position slider, elapsed and total time, Loop, speed (0.5x, 1x, 2x, 4x; audio plays at 1x only), the file name, Signals… (Find Signals in Recording for this file) and Close (back to the device, frequency, mode, FM deviation, filter and receiving state from before). On a narrow window the PLAYBACK caption, then the file name (the status bar names it), make room for the position slider |
| Frequency group | Frequency field with a unit (Hz, kHz, MHz, GHz); it reads `433.92M`, `7074k`, `1.09 GHz` or a number in Hz, refuses a frequency the device can't tune with the reason, and `Esc` cancels typing. A retune never changes what you are typing, and typing stops a running scan. Buttons −1M, −100k, −10k, +10k, +100k, +1M |
| Demodulation group | Mode (None (I/Q), AM, FM, USB, LSB, CW), Filter (choices per mode, right under Mode as in the receiver bar), FM deviation (FM only: 2.5, 5, 12.5, 25 or 75 kHz), CW pitch (CW only, 400–1000 Hz), Squelch (on/off, Auto: 7 dB over the channel's noise, threshold slider) |
| Audio group | Noise reduction (Off, Light, Strong), Audio AGC (Off, Fast, Medium, Slow; AM, SSB and CW), Tone squelch (the 50 CTCSS tones, FM; not used at a broadcast deviation, and setting one there says so) and Tone heard |
| Presets group | Category (Favorites first) and Preset, which follows the tuned frequency ("No preset at 162.500 MHz"); Apply Preset sets the mode, FM deviation, filter and the preset's decoder (a preset the device can't tune is refused with the reason). That decoder goes off again, and the status bar says so, when you tune away from the preset (a tune with a message of its own starts it with that note: "ACARS decoder off · FM Broadcast: listening in FM ...") or apply another that doesn't use it (the Decoder tab's Decode is unticked for it; a protocol you choose next decodes). A "Listen ✓ · Transmit …" line shows only while a HackRF One is connected; its legal-limit tooltip says the app won't transmit when the output calculated on the QRP tab is above the limit |
| Receiver group | Sample rate (applied live); RF gain (an RTL-SDR steps through its tuner's own gains) or, on a HackRF One, LNA gain, VGA gain and RF amp (+14 dB); Tuner AGC (only where the hardware has it); Remove DC spike; Balance I/Q; Correction (ppm) and Direct sampling for an RTL-SDR; Bias tee (RTL-SDR, HackRF One), with a warning while on. Gain and front-end settings are remembered per device type |
| Recording group | Record (I/Q samples or Audio), Format (SigMF first, then cf32, cs16, 16-bit I/Q WAV and raw float32, named as in File → Save Recording, which also writes cu8 and cs8), Folder, Save automatically, Record / Pause with the elapsed time, then the take: "Unsaved · 00:04 · 77 MB @ 100.100 MHz" (with "· 2.1 s lost" when signal was lost) with Save... and Discard, a file loaded by Convert Recording ("Loaded …") with Save As... and Unload, or the saved file's name with Open Folder. While the kind the Record choice doesn't show is recording, the status line names it: "● Recording audio 00:06 (Ctrl+Shift+A stops it)" (in amber while it is paused, armed or waiting) |

The License Profile group sits in Ham → Ham ID, next to the callsign.

| Tool tab | What it does |
|----------|--------------|
| Decoder | Protocol (Off, Auto, POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS, RDS, CW (Morse)), Decode (untick to pause), Decode Recording..., and a status line that says what the decoder hears ("AX.25/APRS · 3 packets · last 4 s ago", "Not tuned to POCSAG: listening on 145.000 MHz (3 pages earlier)"; in amber when this computer can't keep up and the decoder misses part of the signal, naming the app's own Decode Recording, Convert Recording, Save Recording or Find Signals when that is what keeps it busy). Whenever the receiver is tuned elsewhere, "Tune to …" the protocol's usual channel sits beside the status line (or in the empty table), and listens the way that band's preset does (ACARS with its 15 kHz filter); for CW it offers the demo device's beacon, or "Listen in CW" in another mode. Tabs: Messages (rows that fail their checks are tinted red; a Protocol column while rows of another protocol are listed; Decode Recording's rows marked "(file)"; a detail card for the selected row; Clear; Export... to CSV, which starts in the recordings folder), Log, Stats (its "In the table" count says when only the newest 1,000 messages are kept, as the table's and Export's tooltips do), plus Aircraft for ADS-B (one row per aircraft; Seen and Msgs right after Speed; a click on a column title sorts by it) and Text for CW (the text as it arrives, with its speed). Switching between ADS-B and CW brings that protocol's view to the front. While the tab is out of sight its title counts new messages ("Decoder (12)", up to 99+) |
| Bookmarks | Add row (name, frequency, mode; a blank name is filled from the preset or band; a name typed for a channel already bookmarked is kept, with an offer to add it as another bookmark there), a filter box, the list (groups as folders; several bookmarks can be selected), Tune, Edit... (name, frequency, mode, FM deviation, filter, CTCSS tone, group, repeater shift, Skip, comment), Remove and CSV (Import / Export Bookmarks (CHIRP CSV), Export Built-in Presets (CHIRP CSV)). Right-click: Tune, Edit..., Rename..., Copy Frequency, Move Up/Down, Scan Bookmarks from Here, Remove... (Remove N Bookmarks... with several selected; on a group's folder, Remove Group..., one question for the whole group). Tuning a bookmark restores its mode, deviation, filter and tone squelch; `Ctrl+B` saves the tone squelch too and says so ("Bookmarked 2m Calling (FM, tone squelch 100.0 Hz)"), and on a frequency already saved in that mode selects that bookmark instead of adding it twice. The tab shown on a first launch |
| Ham | Four inner tabs. **S-Meter**: the tuned channel in S-units with RST (readability from the measured SNR), a spoken report, peak hold, a 30 s level history, Calibrate (dBFS until calibrated; saved per device type) and Copy (the report with the S reading shown: "RST 59 · S9+23 · 14.250 MHz · 12:34 UTC"). **Ham ID**: callsign, License Profile, CW speed (5–20 WPM) and tone, Sent as (Auto, CW, FM tone), TX Power (HackRF: TX gain, +14 dB amplifier), Automatic ID (start, every 1–10 min, end), Preview ID (on the Tools → Preferences output, at the app's volume), Send ID Now (HackRF One only; Stop while on the air), Transmit Text…, and a line saying what Send ID Now would do or why it can't. **SSTV**: Tune to ISS (145.800 MHz), Mode (Auto (VIS) or one of 14 modes), Start Decoder, Start Now (on Auto it listens up to 5 s for the line syncs that tell the mode), a tone meter, the picture (after a restart, the newest received image) and Received Images (kept across restarts; auto-save); in the wrong mode or sideband for the band it offers Switch to FM, LSB or USB. **QRP**: Calculated Output through an amplifier chain (presets, saturation), the legal limit at the tuned frequency, a power limit, Miles Per Watt and Contest Exchange |
| Scanner | Band (FM Broadcast, NOAA Weather, 2m Ham, 70cm Ham, Airband, ISM 433, ISM 915, or a custom range), Mode (Sweep once, Sweep continuously, Stop on signal, Scan bookmarks with a Group, or All for every group), Threshold (Auto: 12 dB over the measured noise floor), Resume delay, and how long a pass takes. It scans the receiver's own samples, so audio and plots keep running; a chip in the status bar shows a running scan from any tab, and typing in the Frequency field stops it. A bookmark scan starts at the top of the list and follows its order (each group, then the rest). Results: Frequency, Level, SNR, Mode, Heard (a click on a column title sorts, also during a scan; a narrow signal is listed on its channel, 145.800 MHz rather than 145.799; a bookmark scan's results carry the bookmark's name, "Rpt 100 · 145.290 MHz"); Listen (or double-click, `Enter`; a bookmark keeps its tone squelch), Bookmark, Bookmark All (more than 20 results: it asks first and files them in a group of their own, "Scan 2026-10-31 22:00"), ⋯ (Export Results (CSV)..., starting in Documents, then the last folder used; Clear Results). While stopped on a signal: Skip and Hold. With no device it offers Use Demo Device and Connect... |
| Signal | **Identify** (`Ctrl+I`): what the tuned signal is, how sure, how to listen to it (mode, deviation, filter, SNR) and why, with Apply Mode (which sets that mode, deviation and filter); for FM it suggests a deviation that doesn't clip the station and says how far its swing peaks; on/off keyed bursts are taken apart as rtl_433's analyzer does, their bits first, with Copy Pulse Analysis beside Apply Mode. A capture never spans samples the receiver dropped; it says so when the receiver keeps dropping them, or when nothing was heard in what was left. **Signals**: the signals in view (MHz, dBFS, SNR, Width, Heard), measured over 15 frames so a station is listed whole, with Tune, Bookmark, Strongest, ◀ / ▶, Mark on spectrum and SNR ≥. **Scope**: the tuned channel as Magnitude, I and Q, Phase or Frequency over 1–100 ms, or a Constellation with a BPSK/QPSK/8PSK/16QAM overlay and its EVM; Freeze. **Bursts**: transmissions that come and go (Time, MHz, Length, dBFS, What, bits in hex), not stations that never stop (broadcast FM, a beacon keying without a pause), though a transmission far above such a station is listed; a burst's details open under the list when one is selected, and it is outlined on the waterfall; double-click tunes to it; Clear, Export CSV..., Watch in the background. While the computer can't keep up, watching pauses and the line under the list says why. A page does its work only while it is shown, unless Bursts is told to watch in the background |
| Info | Receiver: Device, State, Tuned, Band plan, Center (LO), Sample rate, Span, RBW, Demodulation, Filter, Gain, Front end, Squelch, Channel level. Audio & Recording: Audio output (it also says when the chosen output is unplugged and which plays instead, or that the Second Receiver's Listen mutes this one), Recording, Buffer. Application: Version, Qt / PyQt, Theme. Quick Tips |

The tab bar reads Decoder, Bookmarks, Ham, Scanner, Signal, Info; `Ctrl+1` …
`Ctrl+9` open them (8.3).

| Status bar part | Shows | A click |
|-----------------|-------|---------|
| State badge | NO DEVICE, STOPPED, RUNNING; PLAYING, PAUSED or ENDED while a recording plays; ON AIR while transmitting | NO DEVICE or STOPPED: start (with no device it asks whether to connect one or start Demo Mode); ON AIR: stop transmitting |
| Device name | The device in use, e.g. "Demo Device (simulated signals)"; on a narrow window it is shortened first (whole in its tooltip) | Device → Connect... (with no device, the same choice as Start) |
| Sample rate | "2.4 MS/s" | The sample rate menu (Device → Sample Rate) |
| Messages | Short feedback colored by tone (info, success, warning, error) that clears itself, with room kept for it at 1024 px; problems also go to Tools → Error History, notices such as "Recording paused" don't | The last 20 messages (time, ×2 for repeats), with Copy and Error History... |
| REC | REC 00:01:23 while recording I/Q (REC ARMED before the first samples, REC WAITING while the receiver is stopped mid-recording, PAUSED), with the recorded size and the free disk space in decimal MB and GB ("26.7 MB · 17.7 GB free"; recording stops below 500 MB; on a narrow window this gives way first, whole in its tooltip) | — |
| AUDIO | AUDIO 00:12 while Record Audio runs | — |
| Decoder chip | "ADS-B · 37 msgs" while a decoder runs ("paused" as soon as Decode is unticked; "can't decode" in amber when the decoder can't work, such as ADS-B below 2 MS/s, with the reason in its tooltip and the remedy when there is one; a recording's rate can't be changed, so then only the reason: "ADS-B needs at least 2 MS/s; this recording is 1 MS/s"). When the bar is short of room (REC, AUDIO and AFC shown at 1024 px) it is cut short to fit, keeping at least its first part ("CW (Morse) · …"), and its tooltip then starts with the whole text | Opens the Decoder tab |
| Scan chip | "Scanning · 7 found" ("· 3 heard" when it stops on signals or scans bookmarks), or the channel it is on ("Scanning · Rpt 100", with "· held" while held), while a scan runs; a scan that ends while the Scanner tab is out of sight says how in the messages (a preset or bookmark tuned from another tab keeps its own message after "Scan stopped · "). Short of room, it gives way as the decoder chip does, keeping at least "Scanning · …", with the whole text first in its tooltip | A menu: Stop Scan, Skip and Hold (while it is stopped on a channel), Show Scanner |
| AFC | AFC +1.2 kHz (green when locked on), "lost", "paused" while Radio → Track Signal (AFC) is on | — |

### 8.2 Menus and dialogs

Generated from the menus the window builds. A key in parentheses is shown in
the menu but handled elsewhere: by View → Panels, or by the window itself
when the focused control doesn't use the key (8.3).

**File**

| Item | Key | What it does |
|------|-----|--------------|
| Open Recording... | `Ctrl+O` | Play an I/Q file through the receiver (spectrum, waterfall, audio, decoders, recorders) with the playback strip, heard the way its band usually is (Radio → Choose Mode From Band Plan) |
| Close Recording | | Stop playing; back to the device, frequency, mode, FM deviation and filter used before (enabled while a file plays). Its message also says when going back turned off a decoder that a preset turned on while the file played |
| Recording Info... | | What a file holds: type, format, rate, center frequency, span, length, size, the hardware it was made with (when the file says), whether the samples look right (with their peak in dBFS); Play, Convert... and Decode... from there |
| Decode Recording... | | Run one of the Decoder's seven protocols over a whole file, faster than real time, in the background. The frequency follows the protocol chosen (unless you typed one), and a protocol the file's rate is too low for is grayed out with the reason. Messages go to the Decoder tab marked "(file)", with the recording's own time when it has one |
| Find Signals in Recording... | | List a file's signals (frequency, level, SNR, width, first and last heard) in the Signal tab; double-click one to play from just before it |
| Convert Recording... | `Ctrl+Shift+O` | Load a file in the background to save it in another format; the receiver isn't retuned. The summary offers Save As..., Play and Tune There |
| Save Recording... | `Ctrl+S` | Write the recorded or converted samples as SigMF, cf32, cs16, 16-bit I/Q WAV, raw float32, cu8 (rtl_sdr) or cs8 (hackrf_transfer). A quiet take saved as cu8 or cs8 asks first ("Quiet Recording": Save Anyway or Cancel); saving over the file being converted is refused ("Can't Save There") |
| Import Bookmarks (CHIRP CSV)... | | Add bookmarks from a CHIRP CSV file (replace, or append without duplicates) |
| Export Bookmarks (CHIRP CSV)... | | Save the bookmarks for CHIRP or a radio |
| Save Screenshot... | `Ctrl+P` | The window as PNG, JPEG or BMP, in the screenshots folder |
| Exit | `Ctrl+Q` | Close SDR Module (the platform's Quit key where it has one) |

**Device**

| Item | Key | What it does |
|------|-----|--------------|
| Connect... | | Choose and open a detected RTL-SDR or HackRF One (or the Demo Device), with its sample rate and front-end settings; when that would end a recording it asks first |
| Disconnect | | Stop receiving and close the device (a transmission is ended first; a recording taking its samples from it asks first: "Stop recording and disconnect ...?") |
| Sample Rate ▸ | | The connected device's rates, applied live (RTL-SDR 1, 1.4, 1.8, 2, 2.4 and 2.56 MS/s; HackRF One 2, 2.4, 2.56, 3.2, 4, 8, 10 and 20 MS/s; demo device 1–3.2 MS/s); rates under 2 MS/s are marked "(too low for ADS-B)". Refused while a recording plays, while recording I/Q and while transmitting |
| Use Demo Device | | Simulated signals, no hardware needed (8.7); replaces a connected device (when that would end a recording it asks first) and is ticked while in use |
| Find Connected Devices... | | List the SDRs plugged in, with Connect...; with none, the commands that install a missing driver and Use Demo Device (while the demo device runs it says so, and Close is the default) |
| Second Receiver... | `Ctrl+Shift+N` | A second receiver window on another device or a second demo device (below) |

**Radio**

| Item | Key | What it does |
|------|-----|--------------|
| Start Receiving / Stop Receiving | (`Space`) | Start or stop the receiver; play or pause a recording (off while transmitting) |
| Record I/Q | `Ctrl+Shift+R` | Start or stop an I/Q recording (kept in a temporary file until saved) |
| Record Audio | `Ctrl+Shift+A` | Start or stop recording what you hear: a 48 kHz 16-bit mono WAV in the recordings folder |
| Audio ▸ Mute | `Ctrl+M` | Silence the speaker or play again (also `M` with the plots focused, and the Audio button); with no output device it isn't shown checked and tries the output again |
| Audio ▸ Volume Up / Volume Down | `Ctrl+Up` / `Ctrl+Down` | 10% steps (a focused number field or list keeps these keys: the Frequency field steps ten times as far, a list moves its current row) |
| Audio ▸ Output Device... | | Tools → Preferences, on the audio output |
| Audio ▸ Noise Reduction ▸ | | Off, Light, Strong (the Audio group's Noise reduction; grayed out in None (I/Q)) |
| Audio ▸ Audio AGC ▸ | | Off, Fast, Medium, Slow (AM, SSB and CW; grayed out in other modes) |
| Audio ▸ Tone Squelch... | | Bring the control panel's Tone squelch into view and put the focus on it (outside FM it says to choose FM in Radio → Mode first) |
| Audio ▸ CW Pitch... | | The same for the CW pitch (CW) |
| Mode ▸ | | None (I/Q), AM, FM, USB, LSB, CW: the same choice as MODE in the receiver bar |
| Squelch ▸ Squelch On / Auto Threshold | | The Demodulation group's Squelch and Auto (grayed out when those controls are: Auto Threshold with the squelch off, both in None (I/Q)) |
| Band Presets ▸ | | The favorites (FM Broadcast, NOAA Weather 1, 2m Calling, 70cm Calling, Air Traffic Control, ADS-B, ISM 433 MHz, ISM 915 MHz), then a submenu per category (listed below). Each item shows its frequency and tunes it with its mode, deviation, filter and decoder (the decoder goes off again when you tune away from the preset) |
| Enter Frequency | `Ctrl+L` | Focus the Frequency field; `Enter` tunes and returns to the plots |
| Bookmark Current Frequency | `Ctrl+B` | Save it with the mode, filter and tone squelch |
| Tune Back / Tune Forward | `Alt+Left` / `Alt+Right` | Through the frequencies (and modes) you stayed on |
| Recent Frequencies ▸ | | The last 20 with their modes, and Clear Recent Frequencies |
| Tuning Step ▸ | (`[` / `]`) | Smaller Step, Larger Step, and the steps from 10 Hz to 1 MHz (10, 50, 100, 500 Hz; 1, 2.5, 5, 6.25, 8.33, 9, 10, 12.5, 25, 50, 100, 200 kHz; 1 MHz) |
| Keep Tuned Frequency Centered | | Retune the receiver on every tune (off, the default: the tuned channel moves within the span). Below half the sample rate (1.2 MHz at 2.4 MS/s) it can't center the station: the span then starts at about 0 Hz |
| Choose Mode From Band Plan | | On by default: a far tune without a mode of its own listens the way that band, or that part of a ham band, is usually heard (2.2) |
| RDS Station Names | | On by default: decode a broadcast station's RDS in the background (its name beside FREQ and on the radio tuner) |
| Track Signal (AFC) | | Keep a drifting signal centered (narrow FM, AM, CW) |
| Send Station ID | | Transmit your callsign in Morse on the tuned frequency (HackRF One; settings in Ham → Ham ID); with another device it says why it can't |
| Stop Transmitting | | End the transmission now (enabled while on the air) |

Band Presets categories: Amateur (2m Calling, 70cm Calling, APRS), Aviation
(ADS-B, ACARS, Aviation Emergency, Air Traffic Control), Broadcast (FM
Broadcast), GNSS (GPS L1 (C/A), GPS L2, GPS L5, GLONASS L1), ISM (ISM 433 MHz,
ISM 915 MHz), Marine (Marine Ch 16), Paging (POCSAG/FLEX, POCSAG, FLEX), QRP
(the CW and SSB calling frequencies from 80 m to 10 m), Space (ISS SSTV/Voice,
ISS APRS, ISS Packet (UHF), ISS Repeater Output, Meteor-M2 LRPT, SO-50
Downlink) and Weather (NOAA Weather 1 and 2, NOAA APT for NOAA-15, -18 and
-19).

**View**

| Item | Key | What it does |
|------|-----|--------------|
| Spectrum / Waterfall | | Show or hide each plot (one always stays; remembered) |
| Zoom ▸ Zoom In / Zoom Out / Full Span | `Ctrl+=` or `Ctrl++` / `Ctrl+-` / `Ctrl+0` | Both plots, ×2 to ×16, around the tuned frequency |
| Panels ▸ | `Ctrl+1` … `Ctrl+9` | Decoder (1), Bookmarks (2); S-Meter (3), Ham ID (4), SSTV (5), QRP (6); Scanner (8), Signal (9), Info (7), in tab-bar order |
| Maximize Tool Tabs | | Give the tool tabs the whole right column |
| Focus Spectrum | `F6` | Put the keyboard focus on the spectrum |
| Theme ▸ Dark / Light / Toggle Theme | `Ctrl+T` (Toggle) | Switch themes (8.6) |
| Reset Layout | | Default panel sizes, both plots shown, tool tabs back in place |

**Tools**

| Item | Key | What it does |
|------|-----|--------------|
| Frequency Scanner | `Ctrl+F` | Show the Scanner tab |
| Protocol Decoder | (`Ctrl+1`) | Show the Decoder tab |
| Identify Signal | `Ctrl+I` | Identify the signal at the tuned frequency in the Signal tab (starts the receiver if needed) |
| Signal ▸ Identify / Signals in View / Scope / Bursts | | Show that page of the Signal tab (Identify also identifies the signal, as `Ctrl+I` does) |
| Signal ▸ Tune to Strongest Signal / Next Signal Up / Next Signal Down | (`Home` / `PgUp` / `PgDown`) | Tune the signals in view (the keys work with the plots focused) |
| AM/FM Radio Tuner... | `Ctrl+R` | The broadcast tuner window (Section 12) |
| S-Meter / SSTV Decoder / Ham ID (Station ID) / QRP Calculator | (`Ctrl+3` / `Ctrl+5` / `Ctrl+4` / `Ctrl+6`) | Show that Ham tab |
| SSTV: Tune to ISS (145.800 MHz) | | Tune the ISS SSTV downlink (FM, 5 kHz deviation), start receiving and listen for images |
| Transmit Text... | `Ctrl+Shift+T` | Send text in CW, RTTY, PSK31 or ASCII FSK (below) |
| Error History... | `Ctrl+E` | The last 500 problems (warnings and errors; notices such as "Recording paused", and refusals that are no failure, are left out): Copy All, Export..., Copy Version Info, Clear |
| Preferences... | `Ctrl+,` | Audio output, default folders, recording temp folder, Ask before transmitting |

**Help**

| Item | Key | What it does |
|------|-----|--------------|
| Welcome and Quick Start... | | The first-run wizard again (8.5) |
| Keyboard Shortcuts... | `F1` | Every key and mouse gesture (8.3), with a filter that takes a key ("Ctrl+S", "f2") or words in any order ("mouse wheel"; "wheel" also finds the Scroll rows) |
| Online Documentation | | The user guide (the README) in your browser |
| Report a Problem... | | The issue tracker's new-issue page with the version info filled in (also copied to the clipboard) |
| About SDR Module... | | What the app does, links, and the version info (app, Python, Qt/PyQt, NumPy, system, SDR drivers, device, audio output) with Copy Version Info |

**Dialogs and windows**

| Dialog | Opened from | What it holds |
|--------|-------------|---------------|
| Connect | Device → Connect... | Available Devices (the detected RTL-SDR and HackRF One, and the Demo Device), Sample rate, and only the settings the selected device has: Frequency correction and Direct sampling (RTL-SDR), Bias tee (RTL-SDR, HackRF One). A device the second receiver holds is listed "In use by the second receiver", grayed out with the reason in its tooltip, and Connect is off for it. While a recording takes its samples from the current device, Connect asks first ("Stop recording and switch from ... to ...?"; Cancel keeps the dialog open and changes nothing) |
| No Device | Start (or the status bar) with no device | "No SDR hardware detected." with Demo Mode as the default button, or the radios found by name ("Found RTL-SDR #0 and HackRF One, not opened yet.") with Connect... as the default |
| Devices Found / No Devices Found | Device → Find Connected Devices... | As in the Device menu above |
| Second Receiver | Device → Second Receiver... | Its own Start/Stop, Record, FREQ and LEVEL, spectrum and waterfall; Device (any RTL-SDR or HackRF One the main window isn't using, or a second demo device; sample rate; gain, or LNA/VGA/amp; Direct sampling and Correction for an RTL-SDR), Tuning (frequency, mode, filter, FM deviation, step), Squelch, Listen (Main window, This receiver, Both (mixed)) with Volume, the main window's bookmarks, Record (SigMF in the recordings folder) and Transmit monitor (Monitor TX, Follow transmissions, and a line saying how a transmission is sent), for when the main window has a HackRF One. Its spectrum starts like the main window's (Peak hold off, range Auto) and keeps its own choices; with Auto gain the gain reads "Auto" (or "Auto (25.9 dB)" where the device says what it chose, as the demo device does). While the window is minimized or hidden it skips its plots and keeps receiving and playing. Stop closes its device and, when it was recording, names the saved file, then its folder ("Stopped. Saved iq_…_rx2.sigmf-data (00:02, 51 MB) in ~/Documents/SDR Recordings. …"), so the file's name stays in view at the window's smallest width, 860 px, where the Device list, as wide as its group, shows "Demo Device (simulated signals)" whole |
| Import Options | Opening, converting, decoding or inspecting a file without a header (.bin, .raw, .iq, .dat, .cu8, .cs16 ...), or a SigMF file whose rate or center is unusable | Sample format (cu8, cs8, cs16, cf32, cf64), sample rate and center frequency (up to 300 GHz), prefilled from the extension, the data and the file name, with a check of whether the samples look right (their peak in dBFS; it says when the format is probably wrong, the samples clip or the file is silent) |
| Decode Recording | File → Decode Recording..., Decoder → Decode Recording... | Protocol and frequency (the frequency follows the protocol unless you typed one; a protocol the file's rate is too low for is grayed out); progress and Cancel |
| Convert / Save progress | File → Convert Recording..., Save Recording... | Progress and Cancel (while a Convert, Save or Decode runs, only its progress dialog takes clicks and keys); after Convert a summary with Save As..., Play and Tune There |
| Edit Bookmark | Bookmarks → Edit... | Every field of a bookmark (see the Bookmarks tab) |
| Transmit Text | Tools → Transmit Text..., Ham → Ham ID → Transmit Text… | Mode (CW, RTTY, PSK31, ASCII FSK) and its settings, sent on the carrier or as FM; the text with its length and air time and the characters the mode can't send (Remove Them); Insert (CQ, QRZ?, 73, My call); Preview through the speakers; Save... (48 kHz 16-bit mono WAV, or I/Q as SigMF or cf32); Transmit (HackRF One; it asks before each transmission until you tick Don't ask again), with a line saying what it would do or why it can't |
| AM/FM Radio Tuner | Tools → AM/FM Radio Tuner... | Section 12 |
| Preferences | Tools → Preferences..., Radio → Audio → Output Device... | Audio Output (with Refresh), Folders (recordings, screenshots), Recording Temp Folder (its free space, Use Default), Transmitting (Ask before transmitting) |
| Error History | Tools → Error History..., the status bar's message list | The last 500 problems (warnings and errors) |
| Keyboard Shortcuts | Help → Keyboard Shortcuts... (`F1`) | 8.3 |
| Welcome | Help → Welcome and Quick Start... | 8.5 |

Starting a new recording, converting a file or closing the window with an
unsaved recording offers to save it first (with Save automatically on,
closing saves a running recording without asking). Opening a file while a
recording runs asks "Stop recording and play <file>?", and Device →
Connect..., Use Demo Device and Disconnect ask the same before they end a
recording ("Stop recording and switch from ... to ...?", "Stop recording
and disconnect ...?"; Cancel keeps everything as it was). Bookmark All of
more than 20 scan results, and saving a quiet take as cu8 or cs8, ask first
too. Every transmission asks for confirmation until you tick Don't ask again
(Tools → Preferences turns the question back on).

### 8.3 Keyboard shortcuts

What Help → Keyboard Shortcuts (`F1`) lists, in its order (the nine
`Ctrl`+number rows are one row here). F1 lists only keys the window really
answers to: each menu key is checked against the bindings installed, and a
test presses each key of the lists, fields and windows on the real widget.
F1 names `Enter` "Return" and `Delete` "Del".

| Keys | Action |
|------|--------|
| **Radio** | |
| `Space` | Start or stop the receiver (play or pause a recording) |
| `Ctrl+Shift+R` | Start or stop recording I/Q samples |
| `Ctrl+Shift+A` | Start or stop recording the audio you hear |
| `Ctrl+L` | Type a frequency to tune to |
| `Ctrl+B` | Bookmark the current frequency |
| `Alt+←` / `Alt+→` | Go back / forward through the frequencies you tuned |
| `[` / `]` | Smaller / larger tuning step (with the plots focused) |
| `Ctrl+M` | Mute or unmute the audio (`M` with the plots focused) |
| `Ctrl+↑` / `Ctrl+↓` | Volume up / down by 10% |
| **Tuning** | |
| `←` / `→` | Tune down / up by one tuning step (F1 shows the step in use, e.g. 100 kHz) |
| `Shift+←` / `Shift+→` | Tune down / up by ten tuning steps (F1 shows the amount, e.g. 1 MHz) |
| `Ctrl+←` / `Ctrl+→` | Tune down / up by 1 MHz |
| **File** | |
| `Ctrl+O` | Open a recording and play it back |
| `Ctrl+Shift+O` | Convert a recording to another format |
| `Ctrl+S` | Save the recording |
| `Ctrl+P` | Save a screenshot |
| `Ctrl+Q` | Exit SDR Module |
| **Device** | |
| `Ctrl+Shift+N` | Open the second receiver window |
| **Tools** | |
| `Ctrl+F` | Open the frequency scanner |
| `Ctrl+R` | Open the AM/FM radio tuner |
| `Ctrl+Shift+T` | Send text in CW, RTTY, PSK31 or ASCII FSK |
| `Ctrl+E` | Show the error history |
| `Ctrl+,` | Open the preferences (audio output, folders) |
| `Ctrl+I` | Identify the signal at the tuned frequency (Signal tab) |
| `Home` | Tune to the strongest signal in view (plots focused) |
| `PgUp` / `PgDown` | Tune to the next signal up / down (plots focused) |
| **View** | |
| `F6` | Put the keyboard focus on the spectrum |
| `Esc` | Return the keyboard focus to the spectrum |
| `Ctrl+T` | Switch between the light and dark themes |
| `Ctrl+=` / `Ctrl++` | Zoom the spectrum and waterfall in around the tuned frequency |
| `Ctrl+-` | Zoom the spectrum and waterfall out |
| `Ctrl+0` | Zoom out to the full span |
| `Ctrl+1` … `Ctrl+9` | Show the Decoder (1), Bookmarks (2), S-Meter (3), Ham ID (4), SSTV (5), QRP (6), Scanner (8), Signal (9) or Info (7) panel |
| **Help** | |
| `F1` | Show this list of keyboard shortcuts |
| **Plots** | |
| `P` | Pause or resume the waterfall (either plot focused) |
| `M` | Mute or unmute the audio (like `Ctrl+M`) |
| `Esc` | Clear the waterfall measurement, if one is shown |
| **Frequency field** | |
| `↑` / `↓` | Up / down by one tuning step |
| `PgUp` / `PgDown` | Up / down by ten tuning steps |
| `Enter` | Tune to the frequency typed |
| `Esc` | Cancel typing, back to the plots |
| **Bookmarks list** | |
| `Enter` | Tune to the selected bookmark |
| `F2` | Rename the selected bookmark |
| `Delete` / `Backspace` | Remove the selected bookmark |
| `Esc` | Clear the filter (in the filter box) |
| **Decoder messages** | |
| `Ctrl+C` | Copy the selected messages |
| `Esc` | Clear the selection |
| `Enter` | Aircraft tab: show the selected aircraft's messages (or double-click it) |
| `Ctrl+C` | Aircraft tab: copy the selected aircraft's line |
| **Scanner results** | |
| `Enter` | Listen to the selected signal |
| **Waterfall COLORS scale** | |
| `Home` | Auto levels (also double-click) |
| `←` / `→` | Move the floor down / up by 1 dB (`Shift`: 5 dB) |
| `↓` / `↑` | Move the ceiling down / up by 1 dB (`Shift`: 5 dB) |
| **Playing a recording** | |
| `Space` | Play / pause (plots focused; the toolbar's Play) |
| `←` / `→` | Back / ahead 5 s (position slider focused) |
| `PgDown` / `PgUp` | Back / ahead 30 s (position slider focused) |
| **Second receiver window** | |
| `Space` | Start or stop it |
| `←` / `→` | Tune by its step (`Shift`: ten steps, `Ctrl`: 1 MHz) |
| `[` / `]` | Smaller / larger step |
| `Ctrl+L` | Type a frequency |
| `Ctrl+B` | Bookmark its frequency (in the main window's Bookmarks) |
| `Ctrl+Shift+R` | Record its I/Q |
| `Ctrl+W` | Close it |
| **Mouse** | |
| Click | Tune to the signal clicked on the spectrum or waterfall |
| `Alt`+click | Tune exactly where you click, without snapping to a signal |
| Scroll | Tune by the step over the spectrum or waterfall (`Shift`: ten steps) |
| Scroll | Tune one digit over a digit of the FREQ readout |
| Drag | Measure a signal's bandwidth, duration and peak on the waterfall |
| `Ctrl`+scroll | Zoom the spectrum and waterfall around the pointer |
| Right-click | Tune, bookmark, copy or zoom at a frequency on either plot; options |
| Drag | Move the spectrum's range (over its dB labels; double-click: Auto) |
| Scroll | Move the spectrum's range by 5 dB (over its dB labels) |
| `Shift`+scroll | Make the spectrum's range 10 dB shallower / deeper (over its dB labels) |
| Click | Tune a bookmark from its label along the top of the spectrum |
| Scroll | Shift the waterfall's color levels (over its COLORS scale) |
| Double-click | Give the tool tabs the whole column (on a tab) and back |

`Space`, the arrow keys and the other plot keys act when the focused control
does not use them itself (a focused slider, list or text field keeps its
usual keys); if they seem to do nothing, press `F6` or `Esc`, or click a
plot, first. `Ctrl+↑` / `Ctrl+↓` work the same way: a focused number field
or list keeps them (the Frequency field steps ten times as far, a list moves
its current row), and anywhere else they set the volume.

### 8.4 Persisted state

Stored with `QSettings` (Windows registry, macOS plist, Linux
`~/.config/SDR Module Team/SDR Module.conf`):

| Key or family | What it holds |
|---------------|---------------|
| `frequency_hz`, `tuning/lo_hz` | The tuned frequency and the receiver's center (100.1 MHz in FM when nothing is saved) |
| `demod_mode`, `fm_deviation`, `bandwidth` | Mode, FM deviation and filter |
| `gain_db`, `agc_enabled` | RF gain slider and Tuner AGC |
| `squelch_db`, `squelch/enabled`, `squelch/auto` | Squelch threshold, on/off, Auto |
| `audio/*` | Noise reduction, audio AGC speed, CW pitch, tone squelch, Remove DC spike, Balance I/Q, each mode's own filter and deviation (`audio/mode_filters`) |
| `audio_enabled`, `audio_volume` | Audio on/off and the volume |
| `tuning/step/<mode>`, `tuning/recent` | The step chosen for each mode; Recent Frequencies (the last 20, with modes) |
| `tuning/keep_centered`, `tuning/band_plan_mode`, `decoder/rds_auto` | Radio → Keep Tuned Frequency Centered, Choose Mode From Band Plan, RDS Station Names |
| `device/<kind>/…` (demo, rtlsdr, hackrf) | Per device type: sample rate, gain (with LNA/VGA/amp and Tuner AGC), frequency correction, direct sampling, bias tee |
| `prefs/*` | The audio output device (id and name) and the recording temp folder |
| `recording/*`, `recording_format` | What Record records, Save automatically, the recordings folder, the last folder used for each kind of file (I/Q, audio, decode, screenshots, bookmarks, ...), the Recording group's format |
| `spectrum_display`, `waterfall_display` | Spectrum: AVG, peak and min hold, range, FFT size, window, zoom, bookmark labels. Waterfall: colors, history, levels |
| `geom/*`, `layout/*`, `panel` | Window geometry and splitter sizes; which plots are shown; Maximize Tool Tabs; the Frequency field's unit; the tool tab that was open |
| `theme` | Dark or light |
| `bookmarks` | The bookmarks: name, frequency, demodulation mode, FM deviation, filter, group, and the CHIRP fields (mode, duplex/offset, tone, tuning step, skip, power, comment) |
| `scanner/*` | The Scanner's band, range, step, mode, threshold, resume delay, bookmark group and its last results |
| `license_class`, `ham_id`, `ham/*` | License class; Ham ID (callsign, speed, tone, how the ID is sent, automatic ID); QRP tab (chain, mode, limit, units, logged QSOs, location); S-meter calibration per device type; SSTV mode; the radio tuner's presets |
| `tx/*` | TX gain and amplifier, Ask before transmitting (`tx/confirm`), and Transmit Text's mode, settings, draft and save format (`tx/text`) |
| `device/second_receiver/*`, `layout/second_receiver/*` | The second receiver's device, rate, frequency, mode, filter, deviation, steps, squelch, Listen, volume, gain and Follow transmissions; its window size, splitters, spectrum (Peak hold, averaging, range) and waterfall settings |
| `first_run_done` | Whether the first-run wizard has been shown |

A damaged or hand-edited settings file no longer stops the app starting. A
saved value the app can't have written (text where a number belongs, a
number that isn't finite or doesn't fit the control, a yes/no that reads as
neither, a recordings folder that isn't a full path) gives that setting's
default instead of turning a feature on or off, and the first time it is
read the log says so (Tools → Error History: "Ignoring the damaged setting
..."). A CW pitch outside 400–1000 Hz takes the nearest end; a squelch
threshold outside the slider's range, the -80 dBFS default; Recent
Frequencies outside 1 kHz–300 GHz are dropped; an unreadable window size or
split, or a damaged FFT size, window or zoom, gives the default layout or
display. The spectrum's and waterfall's other display options (holds,
averaging, range, bookmark labels, colors, history, levels) are checked too:
a damaged one gives its default and is logged ("Ignoring the damaged
spectrum setting peak_hold='nope': using False instead."), and an averaging
or history length the menus don't offer gives the nearest one they do, also
logged. So are the second receiver's, except that its waterfall settings,
when they can't be read at all, are dropped without a log line (its
waterfall then starts with the defaults). In the bookmarks, an entry that
isn't a bookmark is skipped and one whose frequency is unusable is listed at
0 Hz, to edit or remove. A damaged `first_run_done` counts as done, and only
an explicit no in `tx/confirm` turns off the question before transmitting. A
Recording group format saved under an earlier name is found by its file
extension.

Files are kept outside the settings: recordings in Documents/SDR Recordings
and screenshots in Pictures (Tools → Preferences changes both), an unsaved
I/Q recording in the cache folder (or the Recording temp folder), and SSTV
images in Pictures/SDR Module/SSTV.

### 8.5 First-run wizard

Shown the first time the GUI starts (Get Started and Skip both count) and
again from Help → Welcome and Quick Start. A callout says whether SDR
hardware was found, names it, and gives the command that installs a missing
driver; reopened while the demo runs, it says "Demo Mode is running".
**Starting Band** lists the favorite bands (FM Broadcast (88–108 MHz),
NOAA Weather, 2m Ham, 70cm Ham, Airband, ADS-B (1090 MHz), ISM 433, ISM 915)
with a line saying which preset it tunes and how ("Tunes to 2m Calling:
146.520 MHz · FM, 5 kHz deviation, 25 kHz filter"). Below it, "Start Demo
Mode (simulated signals)" (ticked when no hardware was found) or "Connect
<device> and start receiving" (when one was). **Quick Tips** lists Click,
`←`/`→`, `Space`, `Ctrl+I`, `Ctrl+Shift+N` and `F1`. **Get Started** starts
Demo Mode if ticked, applies the band's preset (mode, filter and decoder) and
says in one status message what was set; with "Connect ..." ticked it then
opens Device → Connect so receiving starts on that band. **Skip** closes
without tuning.

### 8.6 Themes

Two themes, dark (the default) and light, switch live from View → Theme or
`Ctrl+T` and persist across launches. Every color comes from the palette
tokens in `sdr_module/gui/themes.py`: widgets are tagged with style roles
and tones (`set_role` / `set_tone`) instead of hard-coded colors, and the
custom-painted views (spectrum, waterfall axes, level meter, S-meter, SSTV
tone meter, Signal tab scope, radio tuner) read the active palette and
repaint when the theme changes. Keyboard focus is always visible: a 2 px ring
on buttons, check boxes and slider handles, an accent outline on tabs, lists
and text views, and a focus frame on the plots. The waterfall colormaps are
the same in both themes.

### 8.7 Demo mode

Started with `--demo` (`sdr-scan gui --demo` or `python -m sdr_module.gui
--demo`), Device → Use Demo Device, the Demo Device entry in Device →
Connect, the No Device box or the first-run wizard. The simulated receiver
puts signals at fixed frequencies on a realistic noise floor, so they move
across the display as you tune, and click-to-tune, the scanner and the
decoders work as with real hardware.
Raising the gain lifts signals and noise together and too much gain clips;
Tuner AGC picks the gain itself and keeps the signals clear of clipping.

| Frequency | What is there | Try it with |
|-----------|---------------|-------------|
| 660–1260 kHz | AM broadcast stations on the radio tuner's six AM presets (660, 770, 880, 1010, 1050, 1260 kHz), playing the demo tune | AM/FM Radio Tuner |
| 88.5–107.5 MHz | FM broadcast stations, one on each of the radio tuner's FM presets. The strongest, 100.1 MHz, plays a looping tune in stereo FM (75 µs pre-emphasis) and sends RDS: "SDR DEMO", a RadioText, program type Top 40 | FM; the RDS name beside FREQ; Decoder → RDS |
| 99.820 MHz | CW beacon sending "VVV DE SDR" | CW mode; Decoder → CW (Morse) |
| 118.1–127.85 MHz | Airband AM stations | AM; Scanner → Airband |
| 131.550 MHz | ACARS messages (AM), every 8–13 s | Decoder → ACARS |
| 144.390 MHz | APRS AFSK1200 position reports from three stations, every 8–13 s | Decoder → AX.25/APRS |
| 145.290 MHz | 2 m repeater with a 100.0 Hz CTCSS tone, 8 s in every 12 | Audio → Tone squelch, Tone heard |
| 145.800 MHz | SSTV test card (Robot 36, NBFM) every 75 s, the first 8 s after the demo starts | Ham → SSTV → Tune to ISS |
| 146.520–147.105 MHz, 446.000, 446.100 MHz | 2 m and 70 cm voice-like narrow FM that comes and goes | FM, 5 kHz deviation |
| 152.240 MHz | POCSAG 1200 pages (alphanumeric and numeric), every 8–13 s | Decoder → POCSAG |
| 162.400–162.550 MHz | NOAA weather radio (narrow FM) | Band Presets → NOAA Weather 1 |
| 433.920 MHz | A remote control of the EV1527 kind (on/off keyed, pulse width 350/1050 µs, code 0x5E3A1C sent eight times every 2.5 s); FSK bursts on 434.100 MHz | Signal → Bursts, Identify |
| 438.500 MHz | A 70 cm handheld whose narrow FM drifts ±2.5 kHz over 40 s | Radio → Track Signal (AFC) |
| 915 MHz | ISM bursts | Signal → Bursts |
| 931.9375 MHz | FLEX 1600 pages, every 8–13 s | Decoder → FLEX |
| 1090 MHz | ADS-B from an airliner (KLM1023, ICAO 4840D6: identification, position, velocity, squawk) a few times a second, among weaker Mode S and Mode A/C replies the decoder drops | Decoder → ADS-B (2 MS/s or more) |

Weak carriers sit everywhere else. The demo device runs at 1–3.2 MS/s (1,
1.4, 1.8, 2, 2.4, 2.56 or 3.2 MS/s; 2.4 MS/s by default, then the rate it
had last; `--sample-rate` chooses one) and refuses any other rate. Samples
stream in real time for gap-free audio; on a computer too slow to keep up
the audio pauses (the Audio button reads Paused, and the status bar says
why) until it catches up, while an I/Q recording still gets every sample.
Starting the demo with Device → Use Demo Device or `--demo` while the mode
is None (I/Q) switches it to FM. The demo device can't transmit.

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
| I/Q Sample Format | 8-bit unsigned (cu8) and signed (cs8), 16-bit signed, 32-bit float, 64-bit float |
| Byte Order | Little-endian (I, Q interleaved) |
| Raw File Format | .raw, .cu8, .cs8, .cs16, .cf32, .cf64 |
| Metadata Format | SigMF (Signal Metadata Format) |
| Audio Export | WAV, PCM 16-bit mono at 48 kHz (Radio → Record Audio; Transmit Text's Save...). Another rate comes only from the command line: `sdr-scan encode ... -o file.wav --sample-rate <Hz>` writes the encoded text as a 16-bit two-channel (I and Q) WAV, at 48 kHz by default |
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

**What the transmit path checks.** Everything the app transmits (Ham → Ham ID
→ Send ID Now, Radio → Send Station ID, Tools → Transmit Text, and the
automatic IDs added to them) goes through one transmit engine. It refuses,
with the reason in the status bar and on the Ham ID or Transmit Text line
("Can't send on 100.100 MHz: your license (None) allows CB/MURS/FRS only."),
unless all of these hold:

| Check | Rule |
|-------|------|
| Device | A HackRF One is connected and open (the demo device and an RTL-SDR only receive), no recording is playing back, and no other transmission is on the air |
| Frequency | Within the HackRF's 1 MHz – 6 GHz |
| Lockouts | No part of the emission's occupied bandwidth overlaps a locked band (11.4) |
| License | The license class (Ham → Ham ID → License Profile) allows the mode at the center and at both edges of the emission, for the transmission and each automatic ID (11.9); 50.0–50.1 MHz and 144.0–144.1 MHz are CW only. CW is allowed on every amateur frequency the class covers, phone segments included (FCC 97.305(a)), so a data or SSB transmission there can carry its automatic CW ID (PSK31 on 14.230 MHz for a General); other modes keep their segments. A refusal names the frequency to as many decimals as it needs, and a band the class has no part of ("your license (Technician) doesn't include 60 m (5.3305 MHz)") |
| Identification | A transmission in an amateur band needs your callsign (Ham → Ham ID) for its ID |
| Bandwidth | RTTY and data below 29.7 MHz are at most 2.8 kHz wide (Transmit Text refuses wider) |
| Power | The calculated station output (TX gain, +14 dB amp and the QRP tab's amplifier chain) is within the QRP tab's power limit, when that is on, and within the legal limit at that frequency (1500 W PEP where a segment names none) |
| Antenna port | The bias tee is off; if it can't be switched off, nothing is sent |
| Confirmation | Each transmission asks first ("Transmit DE W1AW on 146.520 MHz (FM tone, 20 WPM, about 1 mW)?", with the air time and the output through the QRP tab's amplifiers; Cancel is the default) until you tick Don't ask again; Tools → Preferences → Ask before transmitting turns the question back on (a damaged setting never turns it off) |

Transmissions are made at 2 MS/s with the HackRF's narrowest filter, and the
HackRF is retuned to the tuned frequency first, so it sends where it was
checked. While on the air the receiver pauses (the HackRF is half duplex)
and the toolbar's Start and Radio → Start Receiving are off. Receiving
resumes afterwards if it was running before, and the plots read
"Transmitting - receiving resumes when it ends"; otherwise it stays off and
they read "Transmitting - press Space to receive once it ends". Record I/Q
or Record Audio started on the air captures once receiving starts again.
Tuning, the sample rate and the TX power can't be changed; Stop, the ON AIR
badge and Radio → Stop Transmitting end it. The status bar says "On air: ...
Stop ends it." meanwhile, and again once a notice that took its place (Space
refused, say) has gone. The HackRF driver checks the
lockouts and the license again for the emission's own bandwidth and mode,
and again whenever the frequency, filter or amplifier changes while it
transmits. Station IDs go out at no more than 20 WPM.

### 11.3 Legal Notice

> **WARNING**: Transmitting on frequencies without proper authorization is illegal in most jurisdictions. Users are responsible for compliance with all applicable laws and regulations. This software is intended for educational, amateur radio, and authorized research purposes only.

### 11.4 TX Frequency Lockouts (Safety Feature)

The software includes **hard-coded TX frequency lockouts** to prevent accidental transmission on critical safety frequencies.

| Category | Locked ranges | Reason |
|----------|---------------|--------|
| **GPS/GNSS** | GPS L1 1560.42–1590.42 MHz, L2 1212.6–1242.6 MHz, L5 1161.45–1191.45 MHz (±15 MHz around 1575.42, 1227.60 and 1176.45 MHz); GLONASS L1 1588–1610 MHz and L2 1232–1249 MHz; Galileo E1 1560.42–1590.42 MHz and E5 1151.45–1232.14 MHz; BeiDou B1 1546.098–1576.098 MHz | **CRITICAL SAFETY** - GPS spoofing can cause aircraft navigation failures |
| **Aviation Emergency** | 121.4–121.6 MHz, 242.9–243.1 MHz | International distress frequencies |
| **ADS-B/Mode S** | 1088–1092 MHz (1090 MHz replies), 1028–1032 MHz (1030 MHz interrogations) | Aircraft collision avoidance transponders |
| **ELT/EPIRB** | 405.5–406.6 MHz | Search and rescue emergency beacons |
| **Marine Distress** | 156.75–156.85 MHz (Ch 16) | VHF marine distress and calling |
| **Cellular** | 698-806 MHz, 824-894 MHz, 1850-1995 MHz | Commercial cellular bands |

> **Note**: These lockouts **cannot be disabled** in the app (the library can add more lockouts, not remove these). A transmission is refused when any part of its occupied bandwidth overlaps a locked range, both by the app's transmit engine and by the HackRF driver (11.2), with the reason in the status bar. The band plan line (FREQ's tooltip, Info → Band plan) shows "transmitting locked out" on these frequencies.

### 11.5 HAM Radio Callsign Identification

For amateur radio compliance, the software includes automatic callsign identification:

| Feature | Description |
|---------|-------------|
| Automatic ID | Sends your callsign at the start, every interval (1–10 min, 10 by default) and at the end of a transmission, as ticked in Ham → Ham ID → Automatic ID; an interval ID goes into Transmit Text's text at a word gap, with the carrier faded around it. An ID that is due anyway goes before the text |
| CW Mode | Morse code identification at 5–20 WPM (an ID sent by an automatic keyer may not be faster, FCC 97.119(b)(1)) |
| Emission | Sent as Auto: a keyed carrier (CW) below 30 MHz and in the CW-only segments, otherwise an FM carrier with the Morse sent as a tone (MCW); an automatic ID goes out the way its transmission does. Or CW / FM tone chosen by hand (an FM tone ID in a CW-only segment is refused) |
| Tone Frequency | The FM tone ID's tone, 400–1000 Hz (default: 700 Hz) |
| "DE" Prefix | Proper amateur radio format: "DE [CALLSIGN]" |
| GUI Panel | Ham → Ham ID tab: callsign, License Profile, CW speed and tone, Sent as, TX Power (HackRF), Automatic ID, Preview ID, Send ID Now (HackRF One only; Stop while on the air, with progress) and a countdown to the next automatic ID; also Radio → Send Station ID |

### 11.6 SSTV Image Reception (ISS/Space)

The software includes an SSTV (Slow Scan Television) decoder for receiving images from the International Space Station and other amateur sources.

| Feature | Description |
|---------|-------------|
| ISS SSTV Preset | Pre-configured 145.800 MHz with FM demodulation; Tune to ISS (145.800 MHz) on the SSTV tab or in the Tools menu tunes it, starts the receiver and the decoder |
| Supported Modes | Robot 36 and 72, Martin 1 and 2, Scottie 1, 2 and DX, PD 90, 120, 160, 180, 240 and 290 (PD120 and PD180 are the ISS favorites), Wraase SC2-180 |
| Live Preview | Real-time image display as lines are decoded, following the line syncs (a slightly fast or slow sound-card clock doesn't slant the picture) |
| Auto-Save | Automatic saving of received images (partial ones too) to Pictures/SDR Module/SSTV. An image that ends with a recording played to its end is completed; one cut off earlier is kept as a partial image ("The audio ended before the image did.") |
| Image History | The newest 50 images in the save folder are listed again after a restart, and the newest is shown |
| VIS Detection | Automatic mode detection from VIS header (Mode: Auto (VIS)), or a fixed mode; Start Now decodes a transmission already under way from its line syncs (on Auto it listens up to 5 s for the syncs that tell the mode, and says so when no audio comes) |
| GUI Panel | Ham → SSTV tab (Start Decoder); decodes the receiver's FM, USB or LSB audio, whether or not the speaker is on (in another mode it offers Switch to FM, or USB/LSB on HF; in the other sideband than the band's usual one, LSB on 160, 80 and 40 m and USB on 60 m and from 20 m to 6 m, it offers that one) |

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

TX permissions are enforced based on the operator's amateur radio license class, chosen in Ham → Ham ID → License Profile (remembered).

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
| 6m | ✓ (50.0-50.1 CW only) | ✓ (50.0-50.1 CW only) | ✓ (50.0-50.1 CW only) |
| 2m | ✓ (144.0-144.1 CW only) | ✓ (144.0-144.1 CW only) | ✓ (144.0-144.1 CW only) |
| 70cm | ✓ | ✓ | ✓ |

**TX Validation:**
- All TX requests are validated against current license class, at the emission's center and at both edges of its occupied bandwidth
- Hardware lockouts (GPS, aviation, emergency) always apply regardless of license
- Mode restrictions enforced per band segment, for the emission's own mode (and its automatic ID's); CW is allowed in every amateur segment granted to the class, phone segments included (FCC 97.305(a))
- Power: the app's transmit path refuses a calculated station output above the legal limit (11.2)

**Power Headroom (150%):**

The frequency manager also gives an effective limit of 150% of the legal
limit (`get_effective_power_limit()`), to account for:
- Cable and connector losses
- Filter insertion loss
- Amplifier efficiency variations
- Measurement uncertainty

The app's transmit path doesn't use it: Send ID Now and Transmit Text refuse
a calculated output above the legal limit itself.

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
receiver. Audio plays through the main window, with the Audio button and
volume slider in its receiver bar.

### 12.1 Frequency Bands

| Band | Frequency Range | Step Size | Modulation |
|------|-----------------|-----------|------------|
| AM | 530 kHz - 1700 kHz | 10 kHz | Amplitude Modulation |
| FM | 87.5 MHz - 108 MHz | 100 kHz | Frequency Modulation |

### 12.2 User Interface Features

| Feature | Description |
|---------|-------------|
| **Frequency Display** | LCD-style readout: band badge, large frequency digits and a station line (the matching preset, then the station's RDS name and RadioText on FM); between channels it shows the receiver's exact frequency ("Between channels"). STEREO lights while a 19 kHz stereo pilot is measured on a broadcast FM station (the audio itself is mono) |
| **Tuning Dial** | Slide-rule scale with dots marking the band's presets; drag or click to tune; mouse wheel or `←`/`→` step one channel, `Page Up`/`Page Down` ten, `Home`/`End` jump to the band edges |
| **Seek Buttons** | ◀ / ▶ either side of the dial seek: they step until a station opens the squelch, once round the band at most (click again to stop); on FM it must be a broadcast station (at least 60 kHz of the channel 10 dB above the noise, or a stereo pilot), not a narrow carrier. `Shift`+click steps one channel. They need the receiver running, and on a device or recording that covers only part of the band they search the part it can tune; during playback each channel is checked in the band's own mode (FM 200 kHz, AM 10 kHz), switched to first. The "Found a station" message goes as soon as you step, turn the dial or retune elsewhere |
| **AM/FM Buttons** | Switch bands; each band returns to the last station tuned on it |
| **Preset Buttons** | 6 station presets per band (12 total); click to tune; right-click for Tune to Preset, Store Current Station Here, Rename... and Clear (press and hold also stores). A preset the device can't tune reads "(out of range)" (a HackRF One tunes from 1 MHz, so the AM presets below it can't be reached); storing is refused while the receiver is outside both broadcast bands |
| **Receiver Link** | Tuning here retunes the main receiver and sets FM (75 kHz deviation, 200 kHz filter) or AM (10 kHz filter). The tuner opens on the receiver's frequency and follows it; while the receiver is outside both broadcast bands the display dims and shows the receiver's frequency instead of a station. With no device the footer says "No receiver: Device › Use Demo Device or Connect…" |

Presets, with their names, are saved between launches.

### 12.3 Default Presets

The default presets are frequencies only (name them with right-click →
Rename...); the demo device has a station on each.

| Preset | FM | AM |
|--------|----|----|
| 1 | 101.1 MHz | 880 kHz |
| 2 | 93.3 MHz | 1010 kHz |
| 3 | 97.1 MHz | 770 kHz |
| 4 | 104.3 MHz | 1050 kHz |
| 5 | 88.5 MHz | 660 kHz |
| 6 | 99.5 MHz | 1260 kHz |

### 12.4 Styling

The tuner is laid out like a car radio: band buttons beside the frequency
display, the dial between its seek buttons, and two rows of preset buttons.
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

# Show the STEREO badge and the RDS station line (the main window sets them
# from its own 19 kHz pilot measurement and RDS decoder)
tuner.set_stereo(True)
tuner.set_rds_text("SDR DEMO")
```

---

## 13. Future Considerations

| Feature | Description |
|---------|-------------|
| Remote Operation | Network-based SDR control |
| Database Integration | Signal/protocol signature database |
| Machine Learning | AI-based signal classification |
| Multi-SDR Support | More than two receivers at once (the GUI runs two: Device → Second Receiver...) |
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

*Document Version: 4.6*
*Last Updated: 2026-11-01*

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
| 4.4 | 2026-10-30 | Section 8 rewritten from the code for the reworked GUI: receiver bar, Scanner, Signal and Ham tabs, playback strip, clickable status bar, every menu item and key, the F1 list, settings keys, demo signals by frequency. Sections 2, 4.2, 5.2, 5.3, 6.4, 11 and 12 brought in line: RDS and Morse decoding, ISM burst analysis, playback and conversion, dual-SDR modes implemented or not, transmit checks and lockout ranges, AM/FM tuner seek and saved presets |
| 4.5 | 2026-11-01 | Section 8 checked against the code again after the final fixes: the control panel's order (Filter under Mode), Radio → Mode and Squelch and the Audio submenu's new items, the Audio button's Muted (2nd RX) and Paused, the scan chip, the decoder chip's "can't decode", the Decoder, Bookmarks, Scanner and Signal tabs, F1's new rows, damaged saved settings. Band-plan listening by segment (2.2), I/Q recordings that hold their center frequency and playback that restores the listening (5.2), the plots' per-frame analysis at 20 MS/s, the audio export formats (9.3), the transmit path: CW on every amateur frequency the license covers, the receiver and recordings while on the air (11.2), SSTV's Start Now and sideband hint (11.6) and the radio tuner's seek (12.2) |
| 4.6 | 2026-11-01 | Checked against the code again after the last fixes: the right column's share of the window's width, the window height that shows the listening controls, the decoder chip on a crowded status bar, unmuting while the second receiver plays and a preset's decoder going off (8.1); Close Recording's message and the Second Receiver at its smallest width (8.2); damaged spectrum and waterfall settings logged (8.4); playback that keeps real time on a busy computer (5.2) and the 20 MS/s frame cost (6.9) |
| 4.7 | 2026-11-01 | Checked against the code again after the final cross-area fixes: Identify's FM deviation and kHz below 2 MHz in the Signal and Scanner tabs (2.2, 8.1); the scan chip on a crowded status bar, a tune that stops a scan or turns a preset's decoder off, the waterfall's noise after the display falls behind (8.1); a recording whose sample rate is too low for the decoder (5.2, 8.1); the app's own jobs named as what keeps the computer busy (6.9, 8.1); "On air" in the status bar (11.2) |
