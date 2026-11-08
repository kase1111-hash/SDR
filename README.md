# SDR Module

A dual-SDR framework for simultaneous RTL-SDR + HackRF One operation, with signal visualization, protocol decoding, and a PyQt6 GUI.

[![CI](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml/badge.svg)](https://github.com/kase1111-hash/SDR/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Does

- **Listening**: AM, FM (broadcast, in stereo, and narrow), USB, LSB and CW audio with squelch, noise reduction, audio AGC, a notch for up to three whistles, a low cut and CTCSS or DCS tone squelch (or reverse tone squelch); spectrum and waterfall with click-to-tune, zoom and measurements; band presets, bookmarks and a scanner
- **Protocol decoding**: ADS-B, ACARS, POCSAG, FLEX (1600, 3200 and 6400 bps), AX.25/APRS, RDS and Morse code (CW), from the live signal (in either receiver window) or from a recording (I/Q, or a receiver's recorded audio); SSTV pictures
- **Signal analysis**: identify the signal at the tuned frequency, list the signals in view, a scope and constellation, and a log of the bursts in the span with on/off keyed data taken apart (like rtl_433's pulse analyzer)
- **Recording and playback**: record I/Q (SigMF, cf32, cs16, cu8, cs8, WAV of any length, raw) or the audio you hear (in stereo from broadcast FM); play a recording back through the receiver, with its sound at 0.5x to 4x, convert it, decode it or find its signals (WAV files of 8 to 32 bits or floating point, and one-channel audio files, too)
- **Two receivers at once**: a second receiver window on another device, with its own notch, low cut, tone squelch and decoder (two frequencies at once, receiving while the HackRF transmits, monitoring your own transmission, sweeping a band together with the main window), and `DualSDRController` for scripts (dual RX, full duplex, TX monitor, wideband scan)
- **Transmitting (HackRF One only)**: your station ID in Morse, and text in CW, RTTY, PSK31 or ASCII FSK, behind hard-coded frequency lockouts and license, band-segment and power checks
- **Ham radio tools**: S-meter with RST reports, callsign ID, SSTV, QRP power calculator and an AM/FM radio tuner
- **Command line**: `sdr-scan` sweeps a band (with one SDR, or two at once), finds and decodes signals in recordings, encodes text to I/Q or WAV, and moves bookmarks to and from CHIRP
- **Signal processing library**: spectrum analysis, demodulation (AM/FM/SSB/CW/OOK/FSK/PSK/QAM), filtering, AGC and signal classification in Python and NumPy
- **No hardware needed to try it**: a demo device with simulated stations and decodable data traffic

## Hardware Support

| Device | Mode | Frequency Range | Bandwidth |
|--------|------|-----------------|-----------|
| RTL-SDR | RX only | 500 kHz - 1.7 GHz | 2.4 MHz |
| HackRF One | TX/RX | 1 MHz - 6 GHz | 20 MHz |

**Combined coverage**: 500 kHz - 6 GHz

## Known Limitations

- **Real-time bandwidth**: all signal processing is Python and NumPy. On a typical 4-core computer a HackRF One at 20 MS/s keeps up while you listen (a display frame of broadcast FM takes about 15 ms of its 33 ms, about 1 ms more in stereo; None (I/Q) with the 2.4 MHz filter about 21 ms), but a slower or busier computer can fall behind. Then the audio pauses (the Audio button reads **Paused** and the status bar says why, naming this computer, the device or one of the app's own jobs such as Decode Recording, and follows a change of cause), the spectrum and decoders skip ahead (the Decoder's status line says its decoder misses part of the signal), and an I/Q recording still keeps every sample the device delivers. A lower sample rate (Device → Sample Rate) helps. An I/Q recording is written to disk by a thread of its own, so writing it takes little of the receiver's time; if samples are lost anyway (the device drops them, or the disk falls more than about a second of signal behind), the recording says how much, and on a disk slower than the signal Stop takes up to about a second to finish writing. The Second Receiver window does its signal processing (its decoder too) on a thread of its own, but the two windows still share this computer. A recording playing back (File → Open Recording) skips nothing: it plays more slowly instead, at least at 1x when the computer manages that, else as fast as it can, and its sound follows the pace it really plays at. The spectrum and waterfall analyze at most about half a million samples of each frame: every sample up to 15 MS/s, half of them at 20 MS/s, so at that rate a burst shorter than about 0.1 ms (at the default FFT size; 1.6 ms at 32768 points) may not show in the waterfall. For wideband capture, record I/Q to disk and process offline.
- **Tested without real radios**: the parts only real hardware can exercise were tested against simulated devices and fake drivers: transmitting on a HackRF One, an RTL-SDR's direct sampling, the bias tee, and two SDRs sweeping together. FM stereo (with its blend and 50 µs de-emphasis), the notch's whistles, the DCS tone squelch and its turn-off code, the CTCSS reverse burst, the reverse tone squelch, FLEX at 3200 and 6400 bps, ADS-B at 2 and 2.048 MS/s and the pages, packets and Morse in a receiver's recorded audio were checked against the project's own encoders, synthetic signals and the demo device, not against off-air recordings or a real radio: multipath, SCA subcarriers and a weak station's stereo blend on real FM stations, a real 50 µs station, real heterodynes, a real radio's DCS (the code, bit order and polarity follow the standard as written), turn-off code, reverse burst and reverse tone squelch, real FLEX transmitters, a real receiver's recorded audio and how a real receiver's filters shape Mode S replies at 2 MS/s are untested. WAV files of 24 and 32 bits and floating point were checked against files ffmpeg wrote and the project's own (24-bit samples in 32-bit words against the project's own only), not against files from other SDR programs or ALSA's `arecord`, and the sound at other playback speeds was measured without a sound card, not listened to.
- **HackRF half-duplex**: HackRF One cannot TX and RX simultaneously: the main window's receiver pauses while it transmits (the spectrum and waterfall say so and Start is off; receiving resumes when the transmission ends, if it was on before). A second device in Device → Second Receiver... keeps receiving.
- **Transmitting needs a HackRF One**: the RTL-SDR and the demo device only receive.
- **US rules**: the transmit checks, license classes and band plan follow the US FCC Part 97 rules. Elsewhere they are only a rough guide, and you remain responsible for what you transmit.
- **No relay**: retransmitting what you receive (the dual-SDR library's RELAY mode) is deliberately not implemented: it is restricted almost everywhere, and the app can't tell whether a relay would be legal.
- **ADS-B** needs a sample rate of at least 2 MS/s, and 2.4 MS/s or more hears weak, distant aircraft best: at 2 and 2.048 MS/s weak replies are missed more often. Below 2.5 MS/s the decoder's work per second of signal is capped, so in a very busy sky (more than about 15,000 replies a second) it reads the clearest replies first and leaves some. It says how many ("Busy sky: about 870 replies a second weren't read") in the Decoder tab's status line, where that takes the message count's place while it lasts (the count stays in the line's tooltip and the status bar's decoder chip), in the Second Receiver's decoder line, once in the status bar and in Decode Recording's summary. 2.56 MS/s or more leaves none unread, though in a synthetic busy sky it decoded about as many replies as 2.4 MS/s (3.2 MS/s about a fifth more).
- **FLEX** decodes 1600, 3200 and 6400 bps, but 4-level frames (3200 bps 4-level and 6400 bps) need about 3 dB more signal than 2-level ones, and near that limit a page can now and then show as valid with one word wrong.
- **Decoder messages**: the Decoder tab's table and Log show the newest 1,000 messages. Export saves every message since Clear, kept meanwhile in a file in the system's temporary folder; if that disk fills, Export saves the messages kept until then and the newest 1,000, and says how many are missing. A recording decoded with File → Decode Recording keeps the name it was first listed under for the session, after Clear too (the first file of a name keeps the plain name; a later one of the same name is listed with its folder). In the Aircraft view a reply both receivers heard counts once when the two copies come within 2 s of each other (further apart, twice), and Msgs can read a little low when the two receivers miss different repeats of a frame; the Second Receiver's aircraft and Morse text stay listed after its window closes, until they age out (aircraft) or Clear.
- **Morse**: an E or a T sent as a word of its own shows when it is keyed like the code around it (the "E E" of a "TU E E" sign-off), but one with nothing but pauses around it is still dropped, because that is what static crashes look like; in weak code (below about 6 dB SNR) lone letters may be dropped too.
- **Audio**: None (I/Q) mode has no audio. Stereo needs broadcast FM's 200 kHz filter (with a narrower one it plays mono, and the STEREO badge says why). De-emphasis is 75 or 50 µs (there is no setting for a station sent without pre-emphasis, nor for 25 µs), and De-emphasis, Stereo Blend and Squelch Tail Elimination are each one setting for both receivers, not one per bookmark or region: listening across a border, you switch by hand. While a weak station is partly blended toward mono, the blend follows its noise from moment to moment (by a few per cent), so its stereo isn't quite steady. The FM clipping warning (in both windows) names the deviation that holds the swing heard through the filter in use: with a 25 kHz filter on a broadcast station it says 25 kHz, with the 200 kHz filter 75 kHz (broadcast).
- **Notch**: Auto removes up to three whistles at once. A fourth plays on until it is clearly (6 dB) stronger than the weakest of them, whose notch it then takes, and two whistles closer than about 23 Hz count as one (its notch takes the other only partly); Manual removes one frequency. Auto takes about half a second to find each whistle and a second to let one go, removes a drifting whistle only partly (the notch trails it), and lets go of one sweeping faster than about 20 Hz a second until it holds still; a whistle much weaker than the voice, in the voice's loudest band, can take 2–5 s to be found.
- **Tone squelch** works on narrow FM only: a tone or code set for a repeater isn't applied to a broadcast FM station. DCS needs a little more signal than CTCSS (about 5 dB signal to noise below 300 Hz, where its bits are). The squelch tail (the noise as the carrier drops) is cut only where the radio marks its end: a DCS radio that sends no turn-off code is closed about a third of a second after its code stops, and a CTCSS radio that sends no reverse burst (some cheap radios send a "55 Hz tail tone" instead, which isn't looked for) is heard to the end, its squelch tail too. Without a tone squelch set, narrow FM plays a station's tone or code bits under the voice (Low cut 300 Hz takes them out). Tone heard can name a tone that isn't sent: a steady 1 kHz test tone on a channel without one reads as 199.5 Hz.
- **Reverse burst and reverse tone squelch**: a voice pitched near a CTCSS station's tone, sent with little filtering below 300 Hz, can now and then sound like its reverse burst and cut the station off for about 0.8 s mid-transmission (turning Radio → Audio → Squelch Tail Elimination off stops that, and lets the tail through again); another station keying up on the same tone within about 20 ms of the first one dropping can count as a burst too. With the voice right up to the burst, the burst is missed 13–32 % of the time and the tail is heard. A reverse tone squelch holds each transmission about half a second while it looks for the tone or code it mutes, so the stations it plays start that much late, and the station it mutes is heard for its first half second while the squelch is off, or when it keys up within the squelch's 0.3 s hang after another station's carrier drops.
- **The Second Receiver window** has no status history (the main window's status bar keeps the last 20 messages): a long message shows whole in a panel above its status bar while it lasts, at most five lines of it (all of it is in the status line's tooltip). Monitor TX in FM turns its tone squelch off and leaves it off (tuning a bookmark with a tone sets one again).
- **Band plan listening** (Radio → Choose Mode From Band Plan, for both receivers) compares a typed frequency only with the one you leave, not with where you chose how to listen: outside every band, a frequency near a preset channel but off it (152.250 MHz, beside POCSAG's 152.240) applies nothing, and typing the channel next then keeps the mode you came with. In the main window, an FM deviation chosen by hand leaves the band plan's message up for its few seconds.
- **Scanning with two receivers**: two devices at most. Their levels aren't calibrated against each other (each reading is in its own receiver's dBFS, and each device keeps its own gain). Stop on signal and Scan bookmarks use the main receiver only. Listen tunes the Second Receiver only for a result the main receiver can't tune (one both can tune is heard with the main one, whichever read it), and a bookmark made from a result doesn't record which receiver read it.
- **Recordings at other speeds**: the sound keeps its pitch at every speed, but at 4x speech is hard to follow (4x is for finding where something is said, 2x or slower for listening), and music at 0.5x sounds a little "phasey" on held chords. A computer much too slow for the speed (4x on one that manages about 0.8x) still breaks the sound up now and then; the status bar says a lower speed helps. A moment's freeze of the window is a gap at every speed, as at 1x (an audio recording at 1x then goes on from where the recording has got to). Short silences (20–130 ms) remain where a slow audio recording's stretched sound starts (8 kHz audio set to 2x as it opens), and when Loop is turned on after the end's sound has died away but before the playback pauses. At 4x the squelch decides on 0.13 s of the recording at a time, so it can cut up to about 60 ms of the last syllable before it closes. ENDED can show up to about 0.2 s before or 0.1 s after the sound stops, and a recording's last 20 ms or so stay in the demodulator's filters.
- **Audio recordings** (one-channel WAV files of 200 kHz or less) are played as I/Q at half their rate: one plays its sound from 200 Hz up to the top of the USB filter (2.9 kHz with the default 2.7 kHz filter, 3.8 kHz at most), a slow one only to under a quarter of its rate (8 kHz audio to about 1.9 kHz), and nothing above that is heard; the status line says which part plays (giving the file's span in MHz: "0–0.003 MHz" for 6 kHz audio). The plots move a display block at a time (twice a second for 8 kHz audio), and a tone below about 300 Hz (or at the very edges of a faster capture) shows a faint mirror image at the other edge of the span. Find Signals' double-click tunes a tone in the mode in use: in USB, as audio opens, that puts the tone at 0 Hz, where it isn't heard (CW plays it at its pitch). The Decoder tab's protocols decode the sound with File → Decode Recording and `sdr-scan decode`, not as it plays (but for Morse), and never ADS-B or RDS; SSTV isn't among Decode Recording's decoders. Pages (POCSAG, FLEX) decode only from a receiver's FM discriminator audio (a scanner's speaker audio of a pager, de-emphasised and cut below 300 Hz, holds none; APRS packets decode from speaker audio too), and Decode Recording doesn't find a Morse tone's pitch by itself: type it ("700 Hz"; `sdr-scan decode cw` finds it).
- **Best guesses, not certainties**: Identify Signal says how sure it is and is never certain; the Scope's constellation plots channel samples, not recovered symbols; ISM-band devices are taken apart into pulses and bits, but there are no device-specific decoders; the S-meter reads dBFS until you calibrate it.
- **Plot readings**: at an FFT size of 32768 a pixel holds dozens of bins (about 35 in a window 1366 px wide, more in a smaller one), so a weak carrier needs about 10 dB over the noise before the hover reads it as a signal rather than "noise" (zoomed in, or at a smaller FFT size, it reads sooner); at the edges of a broadcast FM station the waterfall's hover can still say "Peak".
- **WAV files**: past 4 GiB (about 7 minutes of I/Q at 2.4 MS/s) a WAV recording is saved as RF64, the WAV format for bigger files, which programs that read only classic WAV files can't open (SigMF, cs16 and cf32 have no such limit); the app writes WAV as 16-bit integers only. WAV files of more than two channels, of 64-bit integers or 16-bit floating point, or compressed (A-law, µ-law, ADPCM, MP3 in WAV) are refused, as are 24-bit samples in 32-bit words that lie neither in each word's top three bytes nor in its low three (packed samples under a wrong header, say) and any other samples stored in words wider than they are (16-bit samples in 32-bit words, say). The app tells where they lie from a few stretches of the samples, so a file silent in all of those is read as 32-bit samples.
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
| `F6` or `Esc` | Put the focus back on the spectrum: `Space` and the arrows work while a plot has the focus (`Esc` also hides the full text of a long status message, shown above the status bar) |

**Demo mode** (`--demo`, Device → Use Demo Device, or the Demo Device entry
in Device → Connect) streams simulated signals in real time on a realistic
noise floor; gain, AGC, squelch, recording, the scanner and every decoder
behave as they would with a real receiver. Where to look:

| Frequency | What the demo sends |
|---|---|
| 88.5–107.5 MHz | FM broadcast stations in stereo; the strongest, 100.1 MHz, plays a tune and sends RDS ("SDR DEMO"), the others three tones, one of them on the left channel only |
| 660–1260 kHz | AM broadcast stations on the radio tuner's six AM presets |
| 99.820 MHz | A CW beacon, "VVV DE SDR" |
| 118.1–127.85 MHz, 162.400–162.550 MHz | Airband AM; NOAA weather FM |
| 144.390, 152.240, 931.9375, 131.550 MHz | APRS, POCSAG, FLEX (1600, 3200 and 6400 bps) and ACARS traffic, every 3.5–13 s |
| 145.290 MHz | A 2 m repeater sending a 100.0 Hz CTCSS tone, 8 s in every 12, each transmission ending with a reverse burst |
| 145.800 MHz | An SSTV test card (Robot 36) every 75 s |
| 433.920 MHz | A remote control (on/off keyed), for the Signal tab's Bursts |
| 438.500 MHz | A handheld drifting ±2.5 kHz, for Radio → Track Signal (AFC) |
| 442.100 MHz | A 70 cm repeater sending DCS code 023 (DCS 023 N), 8 s in every 12, each transmission ending with the DCS turn-off code |
| 1090 MHz | ADS-B from a simulated flight, KLM1023 |

The demo device can't transmit, and a second demo device in the second
receiver window can't hear a real transmission. When a demo device runs at
a sample rate it hasn't used recently, it first builds its stations'
signals for that rate, and the display holds for a moment (about a third of
a second for the FM band).

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
sdr-scan scan --dual --start 88 --end 108 --single         # Two SDRs share one sweep
sdr-scan scan --dual --rate 2.4e6,20e6 --start 24 --end 1900   # The second (a HackRF One) at 20 MS/s
```

The receiver steps by about 80% of its sample rate (`--rate`, default 2.4
MS/s) and prints each signal as it is found (frequency, level, SNR, band),
then a summary per sweep. The threshold is the noise floor + 12 dB unless
you give one in dBFS, and the gain is automatic unless you give `--gain`.
With no SDR connected it says so and exits with an error.

`--dual` sweeps with the first two SDRs found at once (an RTL-SDR and a
HackRF One, or two of a kind; with `--device`, two of that kind). Each
first takes what only it tunes (above 1.766 GHz, only the HackRF One), then
they share the rest from opposite ends until they meet, so the faster one
takes more. The header says what each receiver sweeps, each signal is
printed with the receiver that found it, each sweep ends with what each one
swept, and at the end (or at Ctrl+C) every signal is listed once more in
frequency order, each once (a station both receivers heard with the
stronger reading). `--rate` takes one rate for both or one each, and
`--gain` sets both. A receiver that returns no samples (unplugged, or in
use by another program) is left out after three steps in a row and the
other sweeps what it can of its part. The header's time per sweep is an
estimate that runs low with an RTL-SDR. Every step is measured with a
2048-point FFT, so at 20 MS/s a HackRF One reads in 9.8 kHz bins, coarse
for 12.5 kHz channel steps.

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
the GUI writes beside its raw and WAV recordings; a WAV file (classic, or
RF64 and Wave64 past 4 GiB; 8- to 32-bit integer or floating-point samples)
gives its rate.
For other headerless raw captures pass `--sample-rate`, and `--format`
(`cu8`, `cs8`, `cs16`, `cf32`, `cf64`) unless the file's extension is one of
those. `cu8` is the RTL-SDR native format (what `rtl_sdr` writes), `cs8`
what `hackrf_transfer` writes. ADS-B needs a capture of at least 2 MS/s
(Mode S bits are 1 µs wide; 2.4 MS/s or more decodes best), RDS at least
200 kS/s.

A one-channel WAV file at an audio rate (200 kHz or less: a receiver's
sound) is decoded as a receiver's audio: `aprs`, `pocsag` and `flex` from
its FM discriminator's output, `acars` from its AM detector's, and `cw` at
its Morse tone's pitch, found by itself or given with `--pitch 700`
(`sdr-scan decode cw --input cq_audio.wav`). Otherwise (an IF or RF sampled
directly, `adsb` and `rds`, or `cw` with `--offset`) a one-channel file is
read as I/Q at half its rate, centered on a quarter of it unless a
`.sigmf-meta` beside it places it elsewhere (an audio-rate file it places is
read as I/Q there too); `--offset` counts from there. For a one-channel file
`--sample-rate` is the file's own rate, and `--max-samples` counts its own
samples when its sound is decoded as audio or its Morse at its pitch, I/Q
samples when it is read as I/Q.

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
repeater shift, tone/DTCS (the reverse TSQL-R and DTCS-R too), mode, tuning
step, skip, power, and comment fields are preserved on a round trip. Saved
channels are exported and listed in the order the Bookmarks tab shows them
(each group, then the rest).
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
second has MODE, FILTER, STEP, the Audio button with its volume and, on
broadcast FM, the STEREO badge. `F1`
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
  (unless you have changed the decoder since); File → Close Recording, back
  on the preset's channel, turns on again the decoder a recording turned
  off. A preset the device can't tune is refused with the reason.
- **Squelch** is on and **Auto** (7 dB above the channel's noise) by
  default. The LEVEL meter shows the channel level, the noise floor and the
  threshold (drag on it to set the threshold), with an OPEN, SQL, TONE, OFF
  or I/Q lamp. The Audio group has noise reduction, audio AGC, Notch, Low
  cut and Tone squelch (with its Reverse switch). At any window size, Radio
  → Squelch (Squelch On, Auto Threshold) and Radio → Audio (Mute `Ctrl+M`,
  the volume, Noise Reduction, Audio AGC, Notch, Low Cut, Tone Squelch...,
  Reverse Tone Squelch and CW Pitch...) reach the same settings, grayed out
  when the control panel's own control is (Auto Threshold with the squelch
  off, say). Stereo, Stereo Blend, De-emphasis and Squelch Tail Elimination
  are in Radio → Audio only; the Second Receiver uses the last three too (it
  has a Stereo switch of its own).
- **Notch** (Audio group, or Radio → Audio → Notch) takes whistles
  (heterodynes) out of the audio. **Auto** finds up to three steady ones at
  once, each within about half a second, and follows each as it drifts,
  leaving voice alone (the Whistle row under it lists what it removes:
  "Whistles: 700, 1900, 2600 Hz"); it is for voice: a note held in music
  counts as a whistle, so use Off for music. **Manual** removes the
  frequency set in the Whistle row (200–5000 Hz; Radio → Audio → Notch →
  Notch Frequency... chooses Manual and goes to it). **Low cut** (Off, 100,
  200 or 300 Hz; also Radio → Audio → Low Cut) takes out mains hum and
  rumble: 100 Hz the hum, 200 Hz its harmonics too, 300 Hz leaves the voice
  only. Both work in AM, SSB and narrow FM, the notch in CW too; broadcast
  FM is played as it comes.
- **Tone squelch** (Audio group, or Radio → Audio → Tone Squelch...) plays
  only a repeater or PMR radio that sends the chosen sub-audible signal: one
  of the 50 CTCSS tones or of the 104 DCS codes, normal ("DCS 023 N") or
  inverted ("DCS 023 I"); a code's twin of the other polarity opens it too,
  as on a radio. Another tone or code, or none, reads TONE on the LEVEL lamp
  and stays silent, and **Tone heard** names the tone or code the station
  sends. It works on narrow FM, which it then cuts below 300 Hz so the tone
  or the code's bits aren't heard. **Reverse: mute this tone** (under it,
  once a tone or code is chosen; also Radio → Audio → Reverse Tone Squelch)
  does the opposite, as CHIRP's TSQL-R and DTCS-R do: the station sending
  that tone or code is muted (TONE, with Tone heard in the warning color)
  and every other one plays, about half a second into each transmission,
  once the tone or code isn't found. When a transmission ends, the squelch
  closes before the noise of the dropping carrier (the squelch tail)
  wherever the radio marks the end: at a DCS station's turn-off code, and at
  a CTCSS station's reverse burst while Radio → Audio → **Squelch Tail
  Elimination** is on (the default; turn it off if a station is cut off for
  a moment mid-transmission).
- **Broadcast FM plays in stereo** where the station sends it (Radio → Audio
  → Stereo, on by default; off, it plays mono). A weak or noisy station
  blends toward mono by itself to keep the hiss down, as far as Radio →
  Audio → **Stereo Blend** says: Less (more stereo, more hiss), Normal or
  More (quieter weak stations). Radio → Audio → **De-emphasis** matches the
  stations where you listen: 75 µs (the Americas, South Korea; the default)
  or 50 µs (Europe, Australia, most of the world); with the wrong one music
  sounds a little dull or bright. While a broadcast station is above the
  squelch, the badge after the volume reads STEREO or MONO, speaker on or
  off (not while the audio is paused or the Second Receiver's Listen plays
  only that receiver, unless Record Audio is recording in stereo), and its
  tooltip says why (no stereo pilot, too weak, Stereo off, or a filter
  narrower than broadcast FM's 200 kHz); the radio tuner's STEREO lamp
  lights when the badge reads STEREO.
- On broadcast FM the station's **RDS** name shows next to FREQ. Tools →
  AM/FM Radio Tuner (`Ctrl+R`) is a car-radio style tuner with presets and
  seek, and Radio → Track Signal (AFC) follows a drifting signal.

### The spectrum and waterfall

- The spectrum's **⋯** button (or right-click) sets the FFT size
  (1024–32768), the window, the range (Auto or fixed), Peak and Min hold, and
  the bookmark labels along its top (click one to tune it). Drag or scroll
  the dB labels to move the range (`Shift`+scroll makes it deeper or
  shallower). Hovering it reads the level under the pointer with its SNR,
  or "noise" where there is nothing but noise.
- **Zoom** ×2 to ×16: View → Zoom (`Ctrl+=`, `Ctrl+-`, `Ctrl+0`) or
  `Ctrl`+wheel over a plot; both plots zoom together. Zoomed, the
  spectrum's Range → Auto and the waterfall's Auto colors fit the part in
  view, and at ×1 the whole span again.
- **Waterfall**: the COLORS scale is its legend (Auto, or drag its ends),
  HISTORY runs from Live to 1 h, and Pause (`P`) freezes it. Hover for
  frequency, level and age: the row's average level there with its SNR,
  "noise" where there is nothing but noise, or "Peak" and its SNR for a
  short burst. Drag a box to measure bandwidth, duration and peak;
  right-click to tune, bookmark, copy or identify a frequency, or save the
  image.

### Decoding

- **Decoder** tab (`Ctrl+1`): POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS, RDS,
  CW (Morse), or **Auto** (chosen from where you are tuned); **Tune to ...**
  goes to the protocol's usual channel and listens there the way its preset
  does (for CW: the demo device's beacon, or Listen in CW). A status line
  says what the decoder hears (or that you are tuned away from its band, or
  that it misses part of the signal because the computer or device can't
  keep up), messages that fail their checks are tinted red, and a click
  shows a message's details. ADS-B adds an **Aircraft** tab, CW a **Text**
  tab, and a FLEX page's address names the phase and speed it came in on
  ("1000999 (D, 6400 bps)"). The Messages table and the Log show the newest
  1,000 messages; **Export...** saves every message since Clear as CSV.
- While the tab is out of sight its title counts the new messages, and while
  a decoder runs a status-bar chip shows how many it has decoded, "paused"
  while you have unticked Decode, or "can't decode" with the reason in its
  tooltip (click it for the tab). **Decode Recording...** (on the tab, or
  File → Decode Recording...) runs a decoder over a whole recording, much
  faster than real time; protocols the recording's sample rate is too low
  for are grayed out, and its messages are marked "(file)". A one-channel
  WAV file of a receiver's audio is decoded as such: AX.25/APRS, POCSAG and
  FLEX as FM audio (a pager's as the FM discriminator gives it, not a
  speaker's) and ACARS as AM audio (8 kHz or more; ACARS 9.6 kHz), and
  Morse with CW (Morse) at the tone's pitch, typed in the Frequency box
  ("700 Hz"); the dialog says which decode. As such a file plays, CW decodes
  its tones, while AX.25/APRS, POCSAG, FLEX and ACARS say to use Decode
  Recording.
- The **Aircraft** and **Text** views also show what Decode Recording found
  and what the Second Receiver window decodes. A **Showing:** list above the
  view picks Live (what this window's receiver hears now), the Second
  Receiver's Morse text ("2nd RX", with its own speed and Receiving), or a
  decoded recording by its file name (with its folder when two share a
  name). A recording's list keeps the recording's clock: Seen is the time of
  day it was heard when the recording says when it was made (SigMF), else
  the time into it ("1:15"), and its rows never dim or go; decoding the file
  again starts its list over, and Clear empties every list. The Second
  Receiver's aircraft join the live list, one row per aircraft whichever
  receiver heard it, and the line under the table says which did and when
  ("last heard just now (main receiver just now, 2nd RX 4 s ago)").
- **A busy sky**: below 2.5 MS/s ADS-B reads the clearest replies first and
  says how many it leaves unread (at 2 MS/s: "Busy sky: about 870 replies a
  second weren't read (the clearest are read first). 2.4 MS/s reads more of
  them, 2.56 MS/s or more every one.") in the tab's status line (and the
  Second Receiver's decoder line), once in the status bar and in Decode
  Recording's summary.

### Recording and playback

- **Record** in the toolbar, Radio → Record I/Q (`Ctrl+Shift+R`) or Radio →
  Record Audio (`Ctrl+Shift+A`, a WAV of what you hear, after the notch and
  the low cut: in stereo when it starts on broadcast FM with Stereo on, else
  in mono, for the whole take; while a recording plays at another speed, it
  keeps the recording's own pace, 10 s of the recording making 10 s of
  audio, and a take of an audio recording is named after it). The Recording
  group picks what Record records, the format, the folder and Save
  automatically; otherwise File → Save Recording (`Ctrl+S`) writes SigMF,
  cf32, cs16, cu8, cs8, WAV or raw (it asks first before saving a quiet take
  in cu8 or cs8, whose 8 bits would lose most of it). A WAV file past 4 GiB
  (about 7 minutes of I/Q at 2.4 MS/s, or 6 hours of stereo audio) is
  written as RF64, the WAV format for bigger files, and the status bar says
  so. Recording stops before the drive fills (below 500 MB free).
- While an I/Q recording runs, the receiver stays on the recording's center
  frequency: you can tune within the span, and a frequency outside it is
  refused until you stop recording.
- **File → Open Recording** (`Ctrl+O`) plays a recording through the
  receiver (audio, decoders, S-meter and SSTV included) with a playback
  strip: Play/Pause (`Space`), position, Loop, speed (0.5x–4x) and Close.
  The sound plays at every speed, time-stretched so its pitch stays (a voice
  sounds like the same voice, faster or slower): 4x is for finding where
  something is said, 2x or slower for listening, and the status bar says so.
  On a computer that can't keep up the recording plays more slowly, and its
  sound follows. It is heard the way its band usually is (with Radio →
  Choose Mode From Band Plan on), and Close goes back to the device,
  frequency and mode you had before. File → Convert Recording
  (`Ctrl+Shift+O`) and Find Signals in Recording read whole files in the
  background; Recording Info shows what a file holds.
- WAV files from this app or another program open, convert and decode like
  any other: 8-, 16-, 24- and 32-bit integer or 32- and 64-bit
  floating-point samples, and past 4 GiB RF64, BW64 or Sony Wave64 (`.w64`).
  Recording Info says how the samples are stored ("Sample format: 24-bit
  integer"); 24-bit samples in 32-bit words (as ALSA's `arecord -f S24_LE`
  wrote them) are read where the samples show which bytes they fill.
- A **one-channel WAV file** is read as the real signal it holds, at its own
  frequencies (0 Hz to half its rate, each once: 0–24 kHz for 48 kHz audio).
  Audio (200 kHz or less) opens in USB from 0 Hz, and the status line says
  which part of its sound plays at its own pitch ("0.2–2.9 kHz" with the 2.7
  kHz filter; with Radio → Choose Mode From Band Plan off the mode stays).
  Any frequency in it can be tuned, by clicking, typing ("700 Hz") or
  double-clicking a Find Signals result: CW on a Morse tone, say. A capture
  of an IF, or of RF sampled directly, plays its stations at their own
  frequencies, and a `.sigmf-meta` beside it can place it elsewhere (an IF
  at 10.7 MHz).
- **Import Options** asks for what a file doesn't say (a raw file's format,
  rate and frequency, where a WAV file was tuned, a one-channel file's own
  rate when its header has none), filled in from its name where it can be.
  Recording Info and Decode Recording take the center frequency from the
  file's name when nothing else gives it (as SDR# and HDSDR name their
  files), and Recording Info the sample rate too.

### Scanning and bookmarks

- **Scanner** tab (`Ctrl+F`): sweep a band or a custom range once or
  continuously, stop on each signal while it is active (with Skip and Hold),
  or scan your bookmarks (in the order the Bookmarks tab lists them; a
  bookmark with a CTCSS tone or DCS code counts only while its tone or code
  is heard, a reversed one only while it isn't). The plots and decoders keep
  running while it scans; the speaker is muted between channels and plays
  when the scan stops on one. A status-bar chip shows a running scan from
  any tab (click it for Stop, Skip, Hold or the Scanner tab), and typing a
  frequency stops the scan. Results can be listened to, bookmarked or
  exported (Export Results adds a Squelch column for bookmarks with a tone
  squelch: "reverse tone squelch 123.0 Hz"); Bookmark All asks first when
  there are more than 20 and files them in a group of their own.
- **Use both receivers** (`Alt+I`; on the Scanner tab for Sweep once and
  Sweep continuously while the Device → Second Receiver... window is open
  and receiving, not recording) sweeps the range with both receivers at
  once, each its own part from opposite ends until they meet, so a pass
  takes about half as long. The results are one list, each signal once, with
  an **RX** column saying which receiver read it: Main, 2nd or Both (where
  their parts meet; its tooltip says whose level shows and what the other
  read). Each level is in the dBFS of the receiver that read it, as the
  Level header's tooltip says, and Export Results adds Receiver and Level
  from columns. Listen on a result only the second receiver tunes (above
  1766 MHz with an RTL-SDR beside a HackRF One) tunes the Second Receiver
  window there. The second receiver's speaker is muted while it sweeps, and
  it goes back to where it was when the scan ends; if it can't go on (you
  tune or stop it, say), the status bar says why and the main receiver
  sweeps the rest.
- **Bookmarks** tab (`Ctrl+2`): `Ctrl+B` saves the tuned frequency with its
  mode, filter and tone squelch (a CTCSS tone or DCS code, reversed or not).
  Double-click or `Enter` tunes, Edit... changes everything (Reverse: mute
  this tone or code included), `F2` renames, `Delete` removes the selected
  bookmarks; groups (right-click one to remove it), a filter box, and File →
  Import / Export Bookmarks (CHIRP CSV: a channel's TSQL tone or DTCS code,
  or the tone or code a Cross channel receives with, becomes its tone
  squelch, and TSQL-R or DTCS-R a reverse one).

### The Signal tab

The **Signal** tab (`Ctrl+9`, or Tools → Signal):

- **Identify** (`Ctrl+I`) says what the tuned signal probably is, how sure
  it is and how to listen to it (Apply Mode sets that mode and filter: for
  AM as wide as the station's sidebands reach, for FM with a deviation that
  doesn't clip the station), and takes on/off keyed data apart as rtl_433's
  pulse analyzer does.
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
  recording and AFC, chips the decoder and a running scan. A message the
  status bar has to cut short shows whole, in the message's color, in a
  panel just above it (above the playback strip while a recording plays)
  for as long as the message shows; the panel covers the bottom of the
  plots and of the right column meanwhile, and a click on it or `Esc`
  hides it.
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
| Paging | POCSAG (512, 1200, 2400 baud), FLEX (1600, 3200 and 6400 bps) | Decoded live and from recordings |
| Amateur Radio | AX.25/APRS (1200 baud AFSK), Morse code (CW, 5-60 WPM) | Decoded live and from recordings |
| Amateur Radio | SSTV (Robot, Martin, Scottie, PD, Wraase) | Decoded live and while a recording plays (Ham → SSTV) |
| Broadcast | RDS (station name, RadioText, program type, clock) | Decoded live and from recordings |
| ISM Band | 433/868/915 MHz remotes and sensors (OOK, FSK) | Pulse and bit analysis in the Signal tab; no device decoders |
| Text modes | CW, RTTY, PSK31, ASCII FSK | Encoded (`sdr-scan encode`, Transmit Text); sent by a HackRF One |

"Live" is the GUI's Decoder tab (and the Second Receiver window's Decoder
group); "from recordings" is File → Decode Recording or `sdr-scan decode`,
which also decode a receiver's audio recorded in a one-channel WAV file
(AX.25/APRS, POCSAG, FLEX and ACARS, and Morse at its pitch).

## Dual-SDR Operation Modes

In the GUI, **Device → Second Receiver...** (`Ctrl+Shift+N`) opens a second,
lighter receiver in its own window, on another device: any RTL-SDR or HackRF
One the main window isn't using, or a second Demo Device to try it without
hardware. It has its own FREQ and LEVEL readouts, spectrum and waterfall,
mode, filter, squelch, gain and sample rate (but no tool tabs). **Listen**
chooses whose audio plays: the main window's, this receiver's, or both mixed
(the main window's Audio button reads Muted (2nd RX) while only this one
plays). Its **Squelch and audio** group has a Tone squelch (CTCSS or DCS,
with **Reverse: mute this tone** and Tone heard), Notch (Auto names every
whistle it removes) and Low cut, as the main window's Audio group does, and
**Stereo** (broadcast FM in stereo, with a STEREO lamp in its status bar).
That switch is its own, while Radio → Audio → De-emphasis, Stereo Blend and
Squelch Tail Elimination, and Radio → Choose Mode From Band Plan (a
frequency typed in another band is heard the way that band usually is), are
the main window's, for both receivers. It warns in its status line when FM
audio clips, and a message its status line has to cut short shows whole in a
panel above it.

Its **Decoder** group decodes one protocol (POCSAG, FLEX, AX.25/APRS, ADS-B,
ACARS, RDS or CW), with a status line in the Decoder tab's words (a busy
ADS-B sky included), its last message and **Show in Decoder Tab**, which
brings that tab forward: its messages join the main window's Decoder tab
marked "2nd RX · AX.25/APRS", its aircraft the Aircraft view and its Morse
text the Text view (see [Decoding](#decoding)). It shares the main window's
bookmarks (a bookmark's tone squelch, reversed or not, applies there too)
and records its own I/Q to SigMF (Stop says where the file went). Its
signal processing, decoder included, runs on a thread of its own; minimized
or hidden, it keeps receiving and playing but skips its plots, leaving more
time for the main window. While it receives, the Scanner tab's **Use both
receivers** sweeps a range with both devices at once (see [Scanning and
bookmarks](#scanning-and-bookmarks)), its decoder paused meanwhile.

When the main window has a HackRF One, the second receiver's **Transmit
monitor** group covers the transmit uses: it keeps receiving while the
HackRF transmits (full duplex), **Monitor TX** tunes it to the transmit
frequency in the transmission's mode (in FM with its tone squelch off,
since the main window's transmissions send no tone), and **Follow
transmissions** does that for every transmission. Keep its antenna away from
the transmitter, or use 40–60 dB of attenuation.

For scripts, `DualSDRController` (`sdr_module.core.dual_sdr`) runs these
modes:

| Mode | RTL-SDR | HackRF One | Status |
|------|---------|------------|--------|
| DUAL_RX | RX @ Freq A | RX @ Freq B | Implemented |
| FULL_DUPLEX | RX @ Freq A | TX @ Freq B | Implemented |
| TX_MONITOR | RX @ TX Freq | TX | Implemented |
| WIDEBAND_SCAN | Sweeps part of a range | Sweeps the rest at once (alone above 1.766 GHz) | Implemented: `start_wideband_scan()`; also `sdr-scan scan --dual`, and the Scanner tab's Use both receivers |
| RELAY | RX Input | TX Output | Not implemented, on purpose: `set_mode` raises `NotImplementedError` saying why |

`DualSDRController.supported_modes()` lists the implemented ones. Full
duplex and TX monitor refuse a device in the HackRF's place that can't
transmit. `start_wideband_scan(start_hz, end_hz, step_hz)` has both devices
sweep the range at once (`continuous=True` sweeps until stopped): each
first takes what only it tunes, then they share the rest from opposite ends
until they meet, so the faster one takes more. It runs in threads of its
own; `wideband_scan` holds its progress and the signals found, merged in
frequency order with each signal once, and `stop_wideband_scan()` or
`stop_all()` stops it. While it runs, the other modes and retuning are
refused; with one device, that one sweeps alone. Relay (receive and
retransmit) is refused on purpose: retransmitting what you receive is
restricted almost everywhere, and the library can't tell whether a relay
would be legal. `use_devices()` takes devices that are open already, so dual
receive and the wideband scan can be tried with two demo devices:

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

controller.start_wideband_scan(88e6, 108e6, 100e3)   # Both sweep 88-108 MHz at once
scan = controller.wideband_scan
scan.wait()
for hit in scan.hits:                                # In frequency order, each signal once
    print(f"{hit.frequency_hz / 1e6:.3f} MHz {hit.peak_db:.1f} dBFS {hit.receiver_name}")
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
  slide-rule dial, AM/FM buttons, six remembered presets per band, seek,
  a STEREO lamp); it tunes the main receiver, and audio plays through the
  main window
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
├── tests/             # Test suite (~5,800 tests, plus ~130 in sdr-antenna-array)
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
