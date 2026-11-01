# SDR Module

A dual-SDR framework for simultaneous RTL-SDR + HackRF One operation, with signal visualization, protocol decoding, and a PyQt6 GUI.

[![CI](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

- **Listening**: AM, FM (broadcast and narrow), USB, LSB and CW audio with squelch, noise reduction, audio AGC and CTCSS tone squelch; spectrum and waterfall with click-to-tune, zoom and measurements; band presets, bookmarks and a scanner
- **Protocol decoding**: ADS-B, ACARS, POCSAG, FLEX, AX.25/APRS, RDS and Morse code (CW), from the live signal or from a recording; SSTV pictures
- **Signal analysis**: identify the signal at the tuned frequency, list the signals in view, a scope and constellation, and a log of the bursts in the span with on/off keyed data taken apart (like rtl_433's pulse analyzer)
- **Recording and playback**: record I/Q (SigMF, cf32, cs16, cu8, cs8, WAV, raw) or the audio you hear; play a recording back through the receiver, convert it, decode it or find its signals
- **Two receivers at once**: a second receiver window on another device (two frequencies at once, receiving while the HackRF transmits, monitoring your own transmission), and `DualSDRController` for scripts (dual RX, full duplex, TX monitor)
- **Transmitting (HackRF One only)**: your station ID in Morse, and text in CW, RTTY, PSK31 or ASCII FSK, behind hard-coded frequency lockouts and license, band-segment and power checks
- **Ham radio tools**: S-meter with RST reports, callsign ID, SSTV, QRP power calculator and an AM/FM radio tuner
- **Command line**: `sdr-scan` sweeps a band, finds and decodes signals in recordings, encodes text to I/Q or WAV, and moves bookmarks to and from CHIRP
- **Signal processing library**: spectrum analysis, demodulation (AM/FM/SSB/CW/OOK/FSK/PSK/QAM), filtering, AGC and signal classification in Python and NumPy
- **No hardware needed to try it**: a demo device with simulated stations and decodable data traffic

## Hardware Support

| Device | Mode | Frequency Range | Bandwidth |
|--------|------|-----------------|-----------|
| RTL-SDR | RX only | 500 kHz - 1.7 GHz | 2.4 MHz |
| HackRF One | TX/RX | 1 MHz - 6 GHz | 20 MHz |

**Combined coverage**: 500 kHz - 6 GHz

## Known Limitations

- **Real-time bandwidth**: all signal processing is Python and NumPy. On a typical 4-core computer a HackRF One at 20 MS/s keeps up while you listen (a display frame of broadcast FM takes about 15 ms of its 33 ms, None (I/Q) with the 2.4 MHz filter about 21 ms), but a slower or busier computer can fall behind. Then the audio pauses (the Audio button reads **Paused** and the status bar says why), the spectrum and decoders skip ahead (the Decoder's status line says its decoder misses part of the signal), and an I/Q recording still keeps every sample the device delivers (if samples are lost anyway, the recording says how much). A lower sample rate (Device → Sample Rate) helps. A recording playing back (File → Open Recording) skips nothing: it plays more slowly instead, at least at 1x when the computer manages that, else as fast as it can. The spectrum and waterfall analyze at most about half a million samples of each frame: every sample up to 15 MS/s, half of them at 20 MS/s, so at that rate a burst shorter than about 0.1 ms (at the default FFT size; 1.6 ms at 32768 points) may not show in the waterfall. For wideband capture, record I/Q to disk and process offline.
- **HackRF half-duplex**: HackRF One cannot TX and RX simultaneously: the main window's receiver pauses while it transmits (the spectrum and waterfall say so and Start is off; receiving resumes when the transmission ends, if it was on before). A second device in Device → Second Receiver... keeps receiving.
- **Transmitting needs a HackRF One**: the RTL-SDR and the demo device only receive.
- **US rules**: the transmit checks, license classes and band plan follow the US FCC Part 97 rules. Elsewhere they are only a rough guide, and you remain responsible for what you transmit.
- **ADS-B** needs a sample rate of at least 2 MS/s; at 2 and 2.048 MS/s many replies are still missed, so use 2.4 MS/s or more.
- **FLEX** decodes 1600 bps (2-level) frames only; 3200 and 6400 bps frames are recognized and reported as not supported yet.
- **Decoder messages**: the Decoder tab keeps the newest 1,000 messages, and Export saves only those. Messages from Decode Recording are listed in the Messages table and the Log, not in the Aircraft or Text views.
- **Morse**: an E or a T sent as a word of its own (the "E E" of a "TU E E" sign-off) is dropped, because that is what static crashes look like.
- **Audio is mono**: broadcast FM plays without stereo (the radio tuner's STEREO lamp only shows the pilot), None (I/Q) mode has no audio, and a recording plays sound at 1x speed only.
- **Tone squelch** is CTCSS only (no DCS), and works on narrow FM only: a tone set for a repeater isn't applied to a broadcast FM station.
- **Best guesses, not certainties**: Identify Signal says how sure it is and is never certain; the Scope's constellation plots channel samples, not recovered symbols; ISM-band devices are taken apart into pulses and bits, but there are no device-specific decoders; the S-meter reads dBFS until you calibrate it.
- **Plot readings**: each waterfall row shows the strongest level of its frame, and each pixel the strongest of its FFT bins, so hovering reads weak signals a few dB high (and at an FFT size of 32768 can give plain noise an SNR); the spectrum's Range → Auto fits the whole span, even when zoomed.
- **WAV recordings** hold 4 GiB at most (about 7 minutes at 2.4 MS/s): a longer recording can't be saved as WAV (Save automatically saves it as SigMF instead).
- **Dual-SDR library**: the wideband scan and relay modes are not implemented (see [Dual-SDR Operation Modes](#dual-sdr-operation-modes)).
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
python -m sdr_module gui --demo   # The same, if sdr-scan isn't on your PATH
```

On first launch a welcome screen says whether it found an SDR (and, when a
driver is missing, the command that installs it), offers to connect the SDR
or start Demo Mode, picks a starting band (FM Broadcast, NOAA Weather,
2m Ham, 70cm Ham, Airband, ADS-B, ISM 433 or ISM 915, each tuning a preset
with something to hear in demo mode) and shows a few quick tips. Help →
Welcome and Quick Start shows it again. With nothing saved yet the receiver
starts on **100.1 MHz in FM**. `--frequency`/`-f` (Hz) and `--gain`/`-g`
override the remembered values; `--sample-rate`/`-s` sets the demo device's
rate and the one Device → Connect preselects.

The keys a new user needs (`F1` lists every one, filterable):

| Key | What it does |
|---|---|
| `Space` | Start or stop receiving (play or pause a recording) |
| Click, `←`/`→`, mouse wheel | Tune on the spectrum or waterfall: a click lands on the signal, the arrows and wheel move one tuning step (`Shift`: ten steps, `Ctrl`+arrow: 1 MHz) |
| `Ctrl+L` | Type a frequency: `145.8`, `433.92M`, `7074k` (or click the FREQ readout) |
| `Ctrl+M` | Mute or unmute (`M` with the plots focused); `Ctrl+Up`/`Ctrl+Down` set the volume (except in a number field or list, which keeps them) |
| `Ctrl+B` | Bookmark the tuned frequency |
| `Ctrl+I` | What is this signal? (the Signal tab) |
| `Ctrl+1` … `Ctrl+9` | Show a tool tab (Decoder, Bookmarks, the Ham tools, Info, Scanner, Signal) |
| `F6` or `Esc` | Put the focus back on the spectrum: `Space` and the arrows work while a plot has the focus |

**Demo mode** (`--demo`, Device → Use Demo Device, or the Demo Device entry
in Device → Connect) streams simulated signals in real time on a realistic
noise floor; gain, AGC, squelch, recording, the scanner and every decoder
behave as they would with a real receiver. Where to look:

| Frequency | What the demo sends |
|---|---|
| 88.5–107.5 MHz | FM broadcast stations; the strongest, 100.1 MHz, plays a tune and sends RDS ("SDR DEMO") |
| 660–1260 kHz | AM broadcast stations on the radio tuner's six AM presets |
| 99.820 MHz | A CW beacon, "VVV DE SDR" |
| 118.1–127.85 MHz, 162.400–162.550 MHz | Airband AM; NOAA weather FM |
| 144.390, 152.240, 931.9375, 131.550 MHz | APRS, POCSAG, FLEX and ACARS traffic, every 8–13 s |
| 145.290 MHz | A 2 m repeater sending a 100.0 Hz CTCSS tone |
| 145.800 MHz | An SSTV test card (Robot 36) every 75 s |
| 433.920 MHz | A remote control (on/off keyed), for the Signal tab's Bursts |
| 438.500 MHz | A handheld drifting ±2.5 kHz, for Radio → Track Signal (AFC) |
| 1090 MHz | ADS-B from a simulated flight, KLM1023 |

The demo device can't transmit, and a second demo device in the second
receiver window can't hear a real transmission.

### Command line

`sdr-scan` (or `python -m sdr_module`) has seven commands; `sdr-scan <command>
--help` lists each one's options.

```bash
sdr-scan info                  # Version, supported hardware and features
sdr-scan devices               # List the SDRs plugged in (and any driver to install)
sdr-scan gui --demo            # The GUI (see above)
```

#### Sweep a band (needs an RTL-SDR or HackRF One)

```bash
sdr-scan scan --start 88 --end 108 --step 200 --single     # One sweep of the FM band
sdr-scan scan --start 118 --end 137 --step 25              # Airband, until Ctrl+C
sdr-scan scan --device hackrf --start 144 --end 148 --step 12.5 --gain 30 --threshold -60
```

The receiver steps by about 80% of its sample rate (`--rate`, default 2.4
MS/s) and prints each signal as it is found (frequency, level, SNR, band),
then a summary per sweep. The threshold is the noise floor + 12 dB unless
you give one in dBFS, and the gain is automatic unless you give `--gain`.
With no SDR connected it says so and exits with an error.

#### Offline analysis (no hardware required)

Record I/Q to disk (via the GUI, `rtl_sdr` or `hackrf_transfer`), then analyze it offline:

```bash
# Detect signals in a capture via FFT peak detection
sdr-scan scan --input fm.sigmf-data --threshold auto

# Decode a protocol: pocsag, flex, ax25 (or aprs), adsb, acars, rds or cw
sdr-scan decode rds --input fm.sigmf-data --offset 100000   # A station 100 kHz above the center
sdr-scan decode cw --input fm.sigmf-data --offset -180000   # Morse code 180 kHz below it
sdr-scan decode aprs --input aprs.cu8 --format cu8 --sample-rate 1024000
sdr-scan decode adsb --input adsb.cu8 --format cu8 --sample-rate 2400000
```

```
$ sdr-scan decode rds --input fm.sigmf-data --offset 100000
Loaded 11264000 samples (11.000s at 1.024 Msps)
Decoded 3 RDS message(s)
  [0] PI C0DE  Station name: SDR DEMO
  [1] PI C0DE  Program type: Top 40
  [2] PI C0DE  Radio text: Now playing: the SDR Module demo tune (I-vi-IV-V in C)
```

`decode` uses the same decoders as the GUI's Decoder tab: it picks out the
channel `--offset` Hz from the recording's center (default: the center),
counts only the messages that pass their checks, and lists the others too
with `--show-invalid`.

SigMF files give the sample format, rate, and center frequency (pass either
the `.sigmf-data` or `.sigmf-meta` path), and so does the `.sigmf-meta` file
the GUI writes beside its raw and WAV recordings; a WAV file gives its rate.
For other headerless raw captures pass `--sample-rate`, and `--format`
(`cu8`, `cs8`, `cs16`, `cf32`, `cf64`) unless the file's extension is one of
those. `cu8` is the RTL-SDR native format (what `rtl_sdr` writes), `cs8`
what `hackrf_transfer` writes. ADS-B needs a capture of at least 2 MS/s
(Mode S bits are 1 µs wide; 2.4 MS/s or more decodes best), RDS at least
200 kS/s.

#### Encode text (no hardware required)

```bash
sdr-scan encode morse --text "CQ CQ DE N0CALL K" -o cq.wav     # Morse at 20 WPM (--wpm)
sdr-scan encode rtty --text "RYRY DE N0CALL 73" -o rtty.wav    # 45.45 Bd, 170 Hz shift
sdr-scan encode psk31 --text "CQ CQ DE N0CALL" -o psk31.cf32   # Raw cf32 I/Q
sdr-scan encode ascii --text "Hello" --shift 200 --carrier 1170 -o bell103.wav  # 300 Bd, Bell 103 tones
```

The signal is centered on `--carrier` (default 1000 Hz) at `--sample-rate`
(default 48000). A `.wav` name writes a 16-bit two-channel I/Q WAV that
plays as audio; any other name writes raw cf32. `--baud` and `--shift` set
RTTY and ASCII. Characters a mode can't send are listed ("Skipped 2
characters RTTY can't send: @, %"), and without `--text` the text is read
from standard input. What you encode decodes again:
`sdr-scan decode cw --input cq.wav --offset 1000`.

#### Memory channels (CHIRP-compatible CSV)

Saved channels (the GUI's bookmarks) move in and out as
[CHIRP](https://chirpmyradio.com/) generic CSV files, so the same file works
in CHIRP and can be uploaded to a handheld:

```bash
sdr-scan channels export my-channels.csv     # Saved channels -> CHIRP CSV
sdr-scan channels export bands.csv --presets # ...or the built-in RX presets
sdr-scan channels import my-channels.csv     # CHIRP CSV -> saved channels
sdr-scan channels import more.csv --append   # Add instead of replacing
sdr-scan channels import odd.csv --skip-bad  # Skip unreadable rows instead of stopping
sdr-scan channels list my-channels.csv       # Show a file's channels
sdr-scan channels list                       # Show the saved channels
```

Files use CHIRP's exact 21-column header (`Location,Name,Frequency,Duplex,
Offset,Tone,...`), frequencies in MHz. Import is forgiving: any subset of the
optional columns works, headers are matched case-insensitively, extra columns
are ignored, files from a spreadsheet with semicolons or tabs between the
columns (and decimal commas) are read, empty memory slots are skipped, and
repeater shift, tone/DTCS, mode, tuning step, skip, power, and comment fields
are preserved on a round trip. Saved channels are exported and listed in the
order the Bookmarks tab shows them (each group, then the rest).
Reading and writing saved channels needs PyQt6 (they live in the GUI's
settings store); `list` and `export --presets` on a file work without it.

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

The spectrum sits over the waterfall on the left. On the right are the
control panel (Frequency, Demodulation, Audio, Presets, Receiver and
Recording) and the tool tabs: **Decoder**, **Bookmarks**, **Ham** (S-Meter,
Ham ID, SSTV, QRP), **Scanner**, **Signal** and **Info**. The toolbar's
first row has Start/Stop, Record, the FREQ readout and the LEVEL meter; the
second has MODE, FILTER, STEP and the Audio button with its volume. `F1`
lists every key, and [SPEC_SHEET.md section 8](SPEC_SHEET.md#8-user-interface-components)
is the full UI reference.

### Listening and tuning

- **Tune** by clicking a signal (`Alt`+click: exactly there), with `←`/`→`
  or the mouse wheel over the plots, by scrolling over a digit of FREQ, or
  by typing (`Ctrl+L`). The tuned channel moves within the span and the
  receiver retunes only near its edges (Radio → Keep Tuned Frequency
  Centered retunes every time). Below 2 MHz FREQ reads kHz; a station below
  half the sample rate (1.2 MHz at 2.4 MS/s, 10 MHz at 20 MS/s) can't be
  centered, so it sits left of center with the span starting at 0 Hz (a
  little below it at the lowest frequencies). Radio → Tune Back / Tune
  Forward (`Alt+←`/`Alt+→`) and Recent Frequencies return to where you
  were; `[`/`]` change the step.
- **MODE** (AM, FM, USB, LSB, CW, None (I/Q); also Radio → Mode) brings a
  filter to suit it, and each mode remembers the FILTER and STEP you choose.
  A frequency typed (or a bookmark without a mode) in another band, or in
  another part of an HF band, is heard the way it usually is there (Radio →
  Choose Mode From Band Plan): narrow FM in MURS and FRS/GMRS, for example,
  and on the HF bands and 6 m, CW in the CW/data segments (40 m below 7.125
  MHz, 20 m below 14.15 MHz...) unless you are listening in USB, and USB on
  the FT8 and FT4 channels.
- **Radio → Band Presets** has eight favorites and every preset by
  category (also in the Presets group); a preset sets the mode and filter
  and turns its decoder on (ADS-B, APRS, ACARS, POCSAG, FLEX, or SSTV for
  the ISS). Tuning out of that protocol's band, or choosing a preset that
  doesn't use it, turns the decoder off again and the status bar says so
  (unless you have changed the decoder since). A preset the device can't
  tune is refused with the reason.
- **Squelch** is on and **Auto** (7 dB above the channel's noise) by
  default. The LEVEL meter shows the channel level, the noise floor and the
  threshold (drag on it to set the threshold), with an OPEN, SQL, TONE, OFF
  or I/Q lamp. The Audio group has noise reduction, audio AGC and CTCSS tone
  squelch. At any window size, Radio → Squelch (Squelch On, Auto Threshold)
  and Radio → Audio (Mute `Ctrl+M`, the volume, Noise Reduction, Audio AGC,
  Tone Squelch... and CW Pitch...) reach the same settings, grayed out
  when the control panel's own control is (Auto Threshold with the squelch
  off, say).
- On broadcast FM the station's **RDS** name shows next to FREQ. Tools →
  AM/FM Radio Tuner (`Ctrl+R`) is a car-radio style tuner with presets and
  seek, and Radio → Track Signal (AFC) follows a drifting signal.

### The spectrum and waterfall

- The spectrum's **⋯** button (or right-click) sets the FFT size
  (1024–32768), the window, the range (Auto or fixed), Peak and Min hold, and
  the bookmark labels along its top (click one to tune it). Drag or scroll
  the dB labels to move the range (`Shift`+scroll makes it deeper or
  shallower).
- **Zoom** ×2 to ×16: View → Zoom (`Ctrl+=`, `Ctrl+-`, `Ctrl+0`) or
  `Ctrl`+wheel over a plot; both plots zoom together.
- **Waterfall**: the COLORS scale is its legend (Auto, or drag its ends),
  HISTORY runs from Live to 1 h, and Pause (`P`) freezes it. Hover for
  frequency, level, SNR and age; drag a box to measure bandwidth, duration
  and peak; right-click to tune, bookmark, copy or identify a frequency, or
  save the image.

### Decoding

- **Decoder** tab (`Ctrl+1`): POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS, RDS,
  CW (Morse), or **Auto** (chosen from where you are tuned); **Tune to ...**
  goes to the protocol's usual channel and listens there the way its preset
  does (for CW: the demo device's beacon, or Listen in CW). A status line
  says what the decoder hears (or that you are tuned away from its band, or
  that it misses part of the signal because the computer or device can't
  keep up), messages that fail their checks are tinted red, and a click
  shows a message's details. ADS-B adds an **Aircraft** tab, CW a **Text**
  tab. The table keeps the newest 1,000 messages; **Export...** saves them
  as CSV.
- While the tab is out of sight its title counts the new messages, and
  while a decoder runs a status-bar chip shows how many it has decoded,
  "paused" while Decode is unticked, or "can't decode" with the reason in
  its tooltip (click it for the tab). **Decode Recording...** (on the tab,
  or File → Decode Recording...) runs a decoder over a whole recording,
  much faster than real time; protocols the recording's sample rate is too
  low for are grayed out, and its messages are marked "(file)".

### Recording and playback

- **Record** in the toolbar, Radio → Record I/Q (`Ctrl+Shift+R`) or Radio →
  Record Audio (`Ctrl+Shift+A`, a WAV of what you hear). The Recording group
  picks what Record records, the format, the folder and Save automatically;
  otherwise File → Save Recording (`Ctrl+S`) writes SigMF, cf32, cs16, cu8,
  cs8, WAV or raw (it asks first before saving a quiet take in cu8 or cs8,
  whose 8 bits would lose most of it). Recording stops before the drive
  fills (below 500 MB free).
- While an I/Q recording runs, the receiver stays on the recording's center
  frequency: you can tune within the span, and a frequency outside it is
  refused until you stop recording.
- **File → Open Recording** (`Ctrl+O`) plays a recording through the
  receiver (audio, decoders, S-meter and SSTV included) with a playback
  strip: Play/Pause (`Space`), position, Loop, speed (0.5x–4x) and Close.
  It is heard the way its band usually is (with Radio → Choose Mode From
  Band Plan on), and Close goes back to the device, frequency and mode you
  had before. File → Convert Recording (`Ctrl+Shift+O`) and Find Signals in
  Recording read whole files in the background; Recording Info shows what a
  file holds. A file without a header asks for its format, rate and
  frequency.

### Scanning and bookmarks

- **Scanner** tab (`Ctrl+F`): sweep a band or a custom range once or
  continuously, stop on each signal while it is active (with Skip and Hold),
  or scan your bookmarks (in the order the Bookmarks tab lists them). The
  plots and decoders keep running while it scans; the speaker is muted
  between channels and plays when the scan stops on one. A status-bar chip
  shows a running scan from any tab (click it for Stop, Skip, Hold or the
  Scanner tab), and typing a frequency stops the scan. Results can be
  listened to, bookmarked or exported; Bookmark All asks first when there
  are more than 20 and files them in a group of their own.
- **Bookmarks** tab (`Ctrl+2`): `Ctrl+B` saves the tuned frequency with its
  mode, filter and tone squelch. Double-click or `Enter` tunes, Edit...
  changes everything, `F2` renames, `Delete` removes the selected
  bookmarks; groups (right-click one to remove it), a filter box, and File →
  Import / Export Bookmarks (CHIRP CSV).

### The Signal tab

The **Signal** tab (`Ctrl+9`, or Tools → Signal):

- **Identify** (`Ctrl+I`) says what the tuned signal probably is, how sure
  it is and how to listen to it (Apply Mode sets that mode and filter, and
  for FM a deviation that doesn't clip the station), and takes on/off keyed
  data apart as rtl_433's pulse analyzer does.
- **Signals** lists the signals in view (`Home`, `Page Up`, `Page Down` tune
  to the strongest and the next ones), **Scope** shows the channel against
  time or as a constellation, and **Bursts** logs the transmissions that
  come and go (a station that is always on isn't one), with their bits in
  hex; the selected burst is outlined on the waterfall.

### Ham radio tools

The **Ham** tab has the **S-Meter** (`Ctrl+3`: S-units and RST from the
measured SNR, uncalibrated until you calibrate it), **Ham ID** (`Ctrl+4`:
callsign, License Profile, how the ID is sent, TX power, Send ID Now),
**SSTV** (`Ctrl+5`: Robot, Martin, Scottie, PD and Wraase pictures from FM,
USB or LSB audio, offering a switch to the band's usual sideband when you
are in the other; Tune to ISS sets everything up) and **QRP** (`Ctrl+6`:
calculated output through your amplifiers, the legal limit at the tuned
frequency, miles per watt).

### Transmitting (HackRF One only)

- Radio → **Send Station ID** (or Send ID Now in Ham ID) sends your callsign
  in Morse. Tools → **Transmit Text** (`Ctrl+Shift+T`) sends CW, RTTY, PSK31
  or ASCII FSK; Preview and Save... work without a device.
- Every transmission goes through the checks in [Safety](#safety), and a
  line says what would be sent or why it can't be. A red **ON AIR** badge
  shows while transmitting: click it, or Radio → Stop Transmitting, to stop.
  Meanwhile the spectrum and waterfall say "Transmitting" and Start is off;
  receiving resumes when the transmission ends, if it was on before.

### Devices and settings

- **Device** menu: Connect..., Disconnect, Sample Rate, Use Demo Device,
  Find Connected Devices... (with the commands that install a missing
  driver) and Second Receiver... (`Ctrl+Shift+N`, see
  [Dual-SDR Operation Modes](#dual-sdr-operation-modes)). Connect... and
  Disconnect ask before they end a recording, and Connect... shows the
  second receiver's device as in use.
- The **Receiver** group matches the device: an RTL-SDR's tuner gain steps,
  a HackRF One's LNA and VGA gain and RF amp, Tuner AGC where the hardware
  has it, the sample rate (changed live), frequency correction and direct
  sampling for HF (RTL-SDR), the bias tee, Remove DC spike and Balance I/Q.
  Each device type remembers its own settings.
- **Tools → Preferences** (`Ctrl+,`): the audio output, default folders, the
  recording temp folder and whether to ask before transmitting. Settings
  persist across launches via `QSettings`. A damaged or hand-edited
  settings file doesn't stop the app from starting: a value it can't use
  gives that setting's default (or the nearest value it allows). A damaged
  bookmark is skipped, or listed at 0 Hz to edit or remove, and Tools →
  Error History says so.

### The window

- **Menus**: File (recordings, bookmarks, screenshots), Device, Radio
  (receiving, recording, audio, mode, squelch, presets, tuning,
  transmitting), View (plots, zoom, panels, theme, layout), Tools (scanner,
  decoder, Signal, ham tools, Transmit Text, Error History, Preferences) and
  Help.
- **Tool tabs**: `Ctrl+1` … `Ctrl+9` (View → Panels). Double-click a tab, or
  View → Maximize Tool Tabs, to give them the whole right column. **Info**
  shows the receiver's state and the band plan at the tuned frequency.
- **Status bar**: click the state badge to start, the device to connect, the
  sample rate to change it, or the message for the last 20; badges show
  recording and AFC, chips the decoder and a running scan.
- **Themes**: dark (default) and light, View → Theme (`Ctrl+T`).
- **Layout**: the right column takes 30% of the window's width (at least
  400 px, at most 660 px) and keeps that share, or the one you drag its
  edge to, as the window is resized; View → Reset Layout restores the
  panel sizes.
- **Help**: Keyboard Shortcuts (`F1`), Welcome and Quick Start, Online
  Documentation (this README), Report a Problem... and About. Tools → Error
  History (`Ctrl+E`) keeps the last 500 problems (warnings and errors); a
  notice, such as what to do first, shows only in the status bar.

## Supported Protocols

| Category | Protocols | Status |
|----------|-----------|--------|
| Aviation | ADS-B, ACARS | Decoded live and from recordings |
| Paging | POCSAG (512, 1200, 2400 baud), FLEX | Decoded live and from recordings; FLEX at 1600 bps only |
| Amateur Radio | AX.25/APRS (1200 baud AFSK), Morse code (CW, 5-60 WPM) | Decoded live and from recordings |
| Amateur Radio | SSTV (Robot, Martin, Scottie, PD, Wraase) | Decoded live and while a recording plays (Ham → SSTV) |
| Broadcast | RDS (station name, RadioText, program type, clock) | Decoded live and from recordings |
| ISM Band | 433/868/915 MHz remotes and sensors (OOK, FSK) | Pulse and bit analysis in the Signal tab; no device decoders |
| Text modes | CW, RTTY, PSK31, ASCII FSK | Encoded (`sdr-scan encode`, Transmit Text); sent by a HackRF One |

"Live" is the GUI's Decoder tab; "from recordings" is File → Decode
Recording or `sdr-scan decode`.

## Dual-SDR Operation Modes

In the GUI, **Device → Second Receiver...** (`Ctrl+Shift+N`) opens a second,
lighter receiver in its own window, on another device: any RTL-SDR or HackRF
One the main window isn't using, or a second Demo Device to try it without
hardware. It has its own FREQ and LEVEL readouts, spectrum and waterfall,
mode, filter, squelch, gain and sample rate (but no decoders or tool tabs).
**Listen** chooses whose audio plays: the main window's, this receiver's, or
both mixed (the main window's Audio button reads Muted (2nd RX) while only
this one plays). It shares the main window's bookmarks and records its own
I/Q to SigMF (Stop says where the file went). Minimized or hidden, it keeps
receiving and playing but skips its plots, leaving more time for the main
window.

When the main window has a HackRF One, the second receiver's **Transmit
monitor** group covers the transmit uses: it keeps receiving while the
HackRF transmits (full duplex), **Monitor TX** tunes it to the transmit
frequency in the transmission's mode, and **Follow transmissions** does that
for every transmission. Keep its antenna away from the transmitter, or use
40–60 dB of attenuation.

For scripts, `DualSDRController` (`sdr_module.core.dual_sdr`) runs these
modes:

| Mode | RTL-SDR | HackRF One | Status |
|------|---------|------------|--------|
| DUAL_RX | RX @ Freq A | RX @ Freq B | Implemented |
| FULL_DUPLEX | RX @ Freq A | TX @ Freq B | Implemented |
| TX_MONITOR | RX @ TX Freq | TX | Implemented |
| WIDEBAND_SCAN | Scan 0-1.7 GHz | Scan 1.7-6 GHz | Not implemented: `set_mode` raises `NotImplementedError` (scan with one device at a time instead) |
| RELAY | RX Input | TX Output | Not implemented: `set_mode` raises `NotImplementedError` |

`DualSDRController.supported_modes()` lists the implemented ones. Full
duplex and TX monitor refuse a device in the HackRF's place that can't
transmit. `use_devices()` takes devices that are open already, so dual
receive can be tried with two demo devices:

```python
from sdr_module.core.dual_sdr import DualSDRController
from sdr_module.devices.polled import PolledDevice
from sdr_module.gui.device_dialog import MockDevice

def on_a(samples): ...   # called with each block from the first device
def on_b(samples): ...   # ...and from the second

controller = DualSDRController()
controller.use_devices(PolledDevice(MockDevice(), "Demo A"),
                       PolledDevice(MockDevice(), "Demo B"))
controller.set_rtlsdr_frequency(100.1e6)
controller.set_hackrf_frequency(144.39e6)
controller.start_dual_rx(on_a, on_b)
...
controller.stop_all()
```

## Safety

Everything the app transmits goes through one transmit engine, and a
transmission is refused, with the reason in the status bar, unless every
check passes:

- **Hard-coded TX frequency lockouts**, whatever the license: GPS/GNSS,
  aviation (121.5/243.0 MHz), ADS-B (1030/1090 MHz), emergency beacons
  (406 MHz), marine distress (156.8 MHz), and cellular bands.
- **License class** (Ham → Ham ID → License Profile): the band, the segment
  and the mode must be allowed for your class, judged on what is really sent
  (its occupied bandwidth and its mode). CW is allowed on every amateur
  frequency your class may transmit on (FCC 97.305(a)), so a data or SSB
  transmission in a phone segment can carry its automatic CW ID; CW-only
  segments, such as 50.0-50.1 and 144.0-144.1 MHz, refuse FM, data and SSB.
- **Power**: the calculated station output (TX gain, the HackRF's amp and the
  amplifier chain on the QRP tab) must stay within the legal limit there, and
  within the QRP tab's power limit while its "Warn when output exceeds the
  limit" box is ticked.
- **Identification**: an amateur transmission needs your callsign; the
  automatic ID goes at the start, every interval and at the end, as ticked
  in Ham ID (at 20 WPM or slower).
- **Confirmation**: each transmission asks first until you tick "Don't ask
  again" (Tools → Preferences turns the question back on). Nothing is sent
  from the demo device or an RTL-SDR, or while a recording plays back, and
  the bias tee is off while the HackRF transmits.

The HackRF driver checks the lockouts and the license itself as well, so
scripts that transmit through the library are checked too. These checks
follow US rules and are not legal advice: you are responsible for what you
transmit. See [SPEC_SHEET.md](SPEC_SHEET.md) for the full list.

The lockout path is covered by `tests/test_frequency_manager.py` (61 tests).

## Optional: Ham Radio Features

The `sdr_module.ham` subpackage provides amateur radio functionality:

- **AM/FM Radio Tuner** laid out like a car radio (frequency display,
  slide-rule dial, AM/FM buttons, six remembered presets per band, seek);
  it tunes the main receiver, and audio plays through the main window
- **Signal Meter** with S-units (S1-S9, S9+dB) and RST reporting
- **Callsign ID** for automatic CW identification (transmitting needs a HackRF One)
- **SSTV Decoder** for ISS and HF images (call `SSTVDecoder.finish()` after
  the last block of a file)
- **QRP Operations** with power calculations

In the GUI the meter, callsign ID, SSTV and QRP tools are the S-Meter,
Ham ID, SSTV and QRP sub-tabs of the **Ham** tab; the tuner opens from
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
│   ├── cli.py         # The sdr-scan command
│   ├── core/          # Device management, dual-SDR controller, config, frequency manager, band plan, CHIRP CSV
│   ├── devices/       # RTL-SDR and HackRF drivers, a streaming wrapper for the demo device
│   ├── dsp/           # Signal processing, demodulators, live protocol decoders (RDS, Morse...), bursts, sweeps, recording
│   ├── gui/           # PyQt6 GUI: main window, panels and tabs, second receiver, transmit engine, playback, demo device, themes
│   ├── ham/           # Optional ham radio features + radio tuner UI
│   ├── protocols/     # Text-mode encoders (CW, RTTY, PSK31, ASCII) + protocol detector
│   ├── ui/            # Visualization components (waterfall, constellation, time-domain)
│   └── utils/         # Helper utilities
├── packages/
│   └── sdr-antenna-array/  # Standalone antenna array package
├── tests/             # Test suite (~4,200 tests, plus ~130 in sdr-antenna-array)
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
