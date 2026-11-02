# Claude.md - SDR Module Project Guide

## Project Overview

SDR Module is a Software Defined Radio framework for dual-SDR operation using RTL-SDR and HackRF One devices. It provides signal analysis, visualization, classification, and protocol decoding capabilities for amateur radio operators and signal analysts.

Most features are used through the PyQt6 GUI (`python -m sdr_module.gui`): spectrum and waterfall, a receiver with audio (broadcast FM in stereo), live decoders, recording and playback, a scanner, signal identification, a second receiver window, and transmitting on a HackRF One with license and power checks. The `sdr-scan` CLI covers scanning (with one SDR, or two at once: `sdr-scan scan --dual`), decoding, encoding and CHIRP channel files. The demo device simulates signals (including decodable traffic), so receiving, decoding, recording, scanning and the signal tools work without hardware; transmitting needs a HackRF One, and each device's own gain and front-end settings need that device.

**Version**: 0.2.0 (Beta)
**Python**: 3.10 - 3.14
**License**: MIT

## Directory Structure

```
src/sdr_module/          # Main source code
├── core/                # Device management, dual-SDR controller and its wideband scan,
│                        #   config, frequency manager (presets, TX lockouts, licenses),
│                        #   band plan, CHIRP CSV
├── devices/             # Hardware drivers (RTL-SDR via pyrtlsdr, HackRF via python_hackrf)
├── dsp/                 # Signal processing (spectrum, demodulators, filters, live decoders,
│                        #   sweeps, signal ID, bursts, AFC, recording, WAV files, FM stereo,
│                        #   audio notch and low cut, DCS)
├── gui/                 # PyQt6 graphical interface (see Key Modules)
├── ham/                 # Optional ham radio features (signal meter, QRP, SSTV, callsign ID)
│   └── gui/             # Ham tab panels (S-Meter, Ham ID, SSTV, QRP), AM/FM radio tuner
├── protocols/           # Decoder framework, protocol detector, text encoders
├── ui/                  # Older visualization components (the Signal tab's Scope uses
│                        #   constellation.py)
├── utils/               # Helper utilities (conversions, I/Q tools, tooltip texts)
└── cli.py               # Command-line interface (sdr-scan)

packages/
└── sdr-antenna-array/   # Standalone antenna array package (own tests)

tests/                   # pytest test suite (conftest.py, hardware fakes, signal helpers)
examples/                # Example scripts
tools/                   # Utility tools (version, PyInstaller entry)
deferred/                # Standalone text-encoder script and example (not packaged)
.github/workflows/       # CI (tests on Linux/Windows/macOS, lint, mypy, security, package build), CodeQL, release
```

## Key Technologies

- **NumPy** - Core numerical computing (required); all DSP is NumPy
- **pyrtlsdr** / **python_hackrf** - SDR device drivers (optional; python_hackrf builds against the system libhackrf)
- **PyQt6** - GUI framework; QtMultimedia plays the audio (without it the GUI runs silent)
- **Pillow** - Optional: PNG/JPEG export of SSTV images
- **scipy** - Optional `[dsp]` extra; no module imports it
- **pytest** / **hypothesis** - Testing framework and property tests

## Development Commands

```bash
# Install with dev dependencies (includes PyQt6)
pip install -e ".[dev]"

# Run the GUI with the demo device (simulated signals, no hardware)
python -m sdr_module.gui --demo

# Run tests with coverage (offscreen: no display needed)
QT_QPA_PLATFORM=offscreen pytest --cov=sdr_module --cov-report=term-missing

# Format code
black src tests packages
isort src tests packages

# Type checking (as CI runs it)
mypy src/sdr_module

# Linting
ruff check src tests packages

# Run CLI
sdr-scan --help
```

## Coding Standards

### Formatting
- **Black**: 88 character line length
- **isort**: Black-compatible import sorting
- **Type hints**: Required for new code; mypy checks everything except `gui/` and `ham/gui/`, whose errors `pyproject.toml` ignores

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
5. **Qt-free logic** - What the GUI computes lives in modules without Qt where it can (`dsp/`, `core/`, and `gui/tuning.py`, `audio_chain.py`, `stereo_audio.py`, `scan_engine.py`, `device_settings.py`, `plot_view.py`, `plot_readout.py`, `rds_station.py`, `decoder_spool.py`, `frame_pipeline.py`), so it is tested a block at a time without a window. Panels show state and emit signals; the window, or a host object such as `WindowScannerHost`, acts on them.

### Documentation
- Module docstrings describing purpose
- Function docstrings with Args, Returns, and Raises sections
- Thread-safety notes on critical classes

## Key Modules

### Core (`core/`)
- `band_info.py` - The band plan line at a frequency (Info tab, FREQ tooltip)
- `band_plan.py` - Which band a frequency is in and how it is usually listened to (CW in the CW/data segments of the amateur SSB bands, USB on the FT8/FT4 channels); `KHZ_BELOW_HZ`: frequencies below 2 MHz are named in kHz. A preset's FM deviation comes from its `fm_deviation` (`FrequencyPreset`: the pagers' "12.5 kHz", their ±4.5–4.8 kHz swing clipped at 5 kHz), else from its channel width (`preset_listening`); the control panel's Apply Preset and the Decoder's `PROTOCOL_TUNE` agree with it
- `chirp_csv.py` - CHIRP-compatible CSV import/export for saved channels
- `config.py` - Configuration management with validation
- `device_manager.py` - Device enumeration and lifecycle management
- `dual_sdr.py` - Dual-SDR controller (modes: see Hardware Support)
- `wideband_scan.py` - `WidebandScan`: one or two open devices sweep one range at once, each in a thread of its own (a step: tune, drop what was queued, let the tuner settle, measure), sharing the range through `dsp/sweep.py`'s `SharedSweep`; the signals are merged in frequency order, each once (`SignalList`: a station both saw keeps its stronger reading). A page one device can't measure (it refused the frequency, returned no samples or raised) goes to the other; a device that fails `MAX_FAILURES` pages in a row is left out of the rest of that pass (and tried again in the next). Callbacks run one at a time in the scan's thread. Used by `DualSDRController.start_wideband_scan` and `sdr-scan scan --dual`
- `frequency_manager.py` - TX lockouts, license profiles, band privileges, RX presets (`RX_PRESETS`; a preset can name the decoder it turns on, and the window turns that decoder off again once you tune away from where it is heard: `_release_preset_decoder`)
- `sample_buffer.py` - Thread-safe circular sample buffers

### DSP (`dsp/`)
- `spectrum.py` - FFT-based spectrum analyzer
- `frame_spectrum.py` - Spectrum of a whole display frame (mean for the spectrum, max for the waterfall; a frame longer than `max_samples` in evenly spread segments), FFT sizes and windows
- `demodulators.py` - AM, FM, SSB, CW, OOK, FSK, PSK, GFSK, MSK, QAM
- `filters.py` - FIR filters, resamplers, AGC, squelch, CTCSS detection, noise reduction (a `Squelch` given `SquelchConfig(mode=SquelchMode.DCS)` still raises `NotImplementedError`: DCS is `dcs.py`'s `DcsDetector`)
- `audio_filters.py` - Listening filters for demodulated audio, streaming (any block sizes give the same audio) and NumPy only: `SosFilter` (biquads in SciPy's `sos` layout, without SciPy), `AudioNotch` (`"off"`, `"auto"`: the strongest steady whistle its `ToneFinder` hears, followed as it drifts, or `"manual"`) and `LowCut` (an inverse-Chebyshev high-pass); both engage, move and let go without a click
- `dcs.py` - DCS (digital-coded squelch): the 104 standard codes, an encoder (`dcs_waveform`: tests and the demo device), `format_dcs` / `parse_dcs` (`"D023N"`), `dcs_label` (`"DCS 023 N"`), `dcs_same_signal` (a code's twin of the other polarity sends the same bits: 023 N and 047 I) and `DcsDetector`, fed FM discriminator audio before any DC blocking
- `nco.py`, `channelizer.py` - Streaming frequency shift and decimation: a channel out of the span
- `live_decoders.py` - Live decoders on raw I/Q (`LIVE_DECODERS`: POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS, RDS, CW); `create_live_decoder(..., recording=True)` is for a recording playing: a rate too low for the protocol is then named as the recording's ("this recording is 1 MS/s"), with no advice to change the sample rate, which a recording can't; `describe_message` gives a FLEX page's address with its phase and speed ("1000999 (D, 6400 bps)"), as the Decoder's table, details card and Export show it
- `protocols.py` - Protocol decoders (POCSAG, AX.25/APRS, RDS, ADS-B, FLEX, ACARS), bit slicer, BCH, APRS parsing. `FLEXDecoder` reads 1600, 3200 and 6400 bps frames in every phase (`FLEXMessage.baud_rate`, `phase`; a 3200-symbol/s frame needs 9.6 kS/s or more, else it is counted as unsupported), following the input's DC between frames and holding it through each one; below 2.5 MS/s `ADSBDecoder` reads Mode S through `mode_s.py`
- `mode_s.py` - `ModeSDemodulator`: Mode S replies at 2.0 to 2.5 MS/s, where a sample spans most of a chip: each reply's start time, pulse level and noise fitted from its preamble, its bits read at that time or, for a reply that then fails its check, found by a Viterbi search. The fits and searches have a budget per second of signal (`FITS_PER_S`, `SEARCHES_PER_S`): over it the clearest preambles are read first and the rest counted in `skipped`
- `protocol_encoders.py` - POCSAG, FLEX (every speed: `FLEX_SPEEDS`, `flex_baseband(..., speed=)`), AX.25/APRS, ACARS, Morse and RDS transmitters (tests and the demo device; there is no ADS-B encoder, the demo builds its Mode S frames in `device_dialog.py`)
- `rds.py`, `morse.py`, `stereo_pilot.py` - RDS physical layer, streaming Morse decoder, 19 kHz pilot detector. A word that is one E or one T is what a crash of static keys, so `MorseDecoder` shows one beside longer words (or in a line of such words) only when it was keyed like code: steady marks (`KeyingDetector.steady`) at the speed already established, and a word gap from that code before or after it (the `KEYED_*` settings; one alone between two pauses still isn't shown, and code sent letter by letter shows them as before); a caller of `process(keying)` that passes no `steady` never has one counted as keyed (lone letters behave as before)
- `fm_stereo.py` - `StereoDecoder`: broadcast FM's stereo difference signal from the multiplex (the 19 kHz pilot followed, L-R taken off its 38 kHz subcarrier) and how much of it to play: the blend keeps stereo's hiss 50 dB below full scale (`FULL_SNR_DB`), goes to mono below 30 dB (`MONO_SNR_DB`) and without a pilot, falls at once and widens back over `RISE_S`
- `classifiers.py`, `signal_id.py` - Signal classification with confidence; identifying the tuned signal and how to listen to it (Signal tab > Identify). The FM deviation it suggests must hold the station's swing, or Apply Mode leads straight to "FM audio is clipping" (shown when more than 5 % of the audio is beyond the deviation), and be no wider, or the station is heard quieter (`listening_for` keeps the band plan's deviation where that holds the swing and the peak is at least a quarter of it): `_swing_to_hold` takes the larger of the peak (`freq_peak_hz`, the swing exceeded 0.5 % of the time) and, with 10 % to spare, the swing exceeded 5 % of the time (`freq_p95_hz`). Voice reaches its peak only now and then, so it is held at its peak (a 2 m station limited at ±5 kHz gets 5 kHz); a tone stays near its peak and keeps the margin. How far a spectrum bin stands above the noise is judged against the noise's measured spread (`_spread` in `_spectrum_features`), not the spread independent averages would have: neighbouring bins of a windowed spectrum aren't independent and the demo's noise is rougher than white, so otherwise the noise's ripple counts as signal all over a wide capture (Identify in broadcast FM looks 800 kHz wide) and joins a call to a station hundreds of kHz away. An AM station is as wide as its sidebands reach (`sideband_edge_hz`), judged with the spectrum mirrored about the carrier (the sidebands are alike either side, the noise isn't). `SignalIdentifier(..., channel_pass_hz=, analysis_rate=)` takes a receiver's channel as it is and judges it at the rate its own filter would have given (see `signal_tools.py`)
- `signal_list.py`, `bursts.py` - Signals in a spectrum or a recording; bursts and rtl_433-style pulse analysis
- `afc.py` - Signal tracking (Radio > Track Signal (AFC))
- `sweep.py` - Sweep planning and measurement (Scanner tab, `sdr-scan scan`); `SharedSweep` shares one sweep between two receivers (`SweepReceiver`: a sample rate and a tuning range): each first takes what only it tunes, then they meet from opposite ends, each page sized for its own rate (the wideband scan, the Scanner tab's Use both receivers); `scanner.py` is the older library scanner
- `recording.py` - I/Q recording/playback (raw cu8/cs8/cs16/cf32/cf64, WAV, SigMF), `.sigmf-meta` sidecars, format guessing; WAV goes through `wav_file.py` (`IQRecorder.start` with `expected_samples` that fit writes a classic file; without them, a file that turns into RF64 if it grows past 4 GiB; `IQPlayer`, `AudioPlayer` and `FormatDetector` read classic WAV, RF64, BW64 and Wave64 through `read_wav_header`, never the `wave` module)
- `wav_file.py` - WAV files of any size (standard library only): `WavWriter` writes integer PCM, a classic file while it fits and RF64 past 4 GiB, in place (a size known to fit gets the classic 44-byte header, any other an 80-byte header whose JUNK chunk becomes RF64's ds64), the header up to date after every write; `read_wav_header` reads classic, RF64, BW64 and Wave64 files and counts one cut short or never finished as what it holds; `WavError` is both a `ValueError` and a `wave.Error`
- `background_writer.py` - `BackgroundWriter`: blocks written in order on a thread of their own from a queue of at most `max_bytes`; `put()` copies the block, or returns False when there is no room (the caller counts it as lost and says so); the first write error stops it and waits in `error` for the caller's thread; `flush()`, `close()`, `abort()`

### Devices (`devices/`)
- `base.py` - Abstract SDR device interface (`start_rx(callback)`, queued `read_samples`, TX)
- `rtlsdr.py` - RTL-SDR driver (tuner gain table, ppm, direct sampling, bias tee)
- `hackrf.py` - HackRF One driver (LNA/VGA/amp, bias tee, half-duplex TX with its own legality check)
- `polled.py` - `PolledDevice`: the streaming API for a source that makes samples on request (the demo device)

### Protocols (`protocols/`)
- `base.py` - Protocol decoder base class and types
- `detector.py` - Automatic protocol detection (tries every live decoder that works at the sample rate)
- `text_modes.py` - Text to signal for CW, RTTY (ITA2), PSK31 (Varicode) and ASCII FSK; used by Transmit Text and `sdr-scan encode`
- `encoder.py`, `encoders.py` - Encoder framework and the RTTY, Morse, ASCII FSK and PSK31 encoders

### Ham Radio (`ham/`) - Optional
- `signal_meter.py` - S-unit signal meter with RST reporting
- `qrp.py` - QRP (low-power) operations support
- `callsign.py` - CW and FM-tone (MCW) station ID (`MorseIdSource`)
- `sstv.py`, `sstv_encoder.py` - SSTV decoder (Robot, Martin, Scottie, PD and Wraase modes, VIS code) and encoder (tests, demo device)
- `gui/` - Ham tab: `signal_meter_widget.py` (S-Meter), `callsign_panel.py` (Ham ID: callsign, station ID, TX power, automatic ID; the window adds the License Profile group), `sstv_panel.py` (SSTV), `qrp_panel.py` (QRP calculator); `radio_tuner.py` (Tools > AM/FM Radio Tuner)

### GUI (`gui/`)

Window and receiver:
- `app.py`, `__main__.py` - `SDRApplication`, `python -m sdr_module.gui` (theme, `--demo`; an unhandled exception is logged and shown in the status bar)
- `main_window.py` - `SDRMainWindow` and what only it uses:
  - menus (`_setup_menus`), toolbar, receiver bar, status bar, tool tabs (`_build_panel_tabs`); the right column takes 30% of the window's width, 400-660 px (`_RIGHT_COLUMN_SHARE`), and keeps that share, or the one its handle was dragged to, as the window is resized (`_keep_right_column_share`)
  - tuning: `set_frequency` / `_tune_to`, `_retune_lo`, `_lock_lo` (see Important Notes)
  - the display loop: `_update_display` -> `_read_block` (paced by `_Pacer`) -> `_process_block`
  - `_ReceiverChain` (channel filter, demodulators, 48 kHz audio; the second receiver reuses it): `demodulate()` returns mono audio in every mode, through the Audio group's notch and low cut (`_listen`, an `audio_chain.ListeningFilters`), and while its `stereo` is set, broadcast FM's left and right go to `stereo_audio` (see Important Notes); `_SampleStore` (the I/Q recording buffer, a temporary file its writer thread fills), `InfoPanel` (Info tab)
  - the status bar's whole-message panel (`_message_overflow`) and the hints that follow their cause (`_skip_follow`, `_audio_follow`); Radio > Audio > Stereo (`_set_stereo_enabled`) and the receiver bar's STEREO badge (`_show_stereo`)
- `control_panel.py` - Right column, listening first: Frequency, Demodulation, Audio, Presets, Receiver and Recording groups (its License Profile group moves to Ham > Ham ID); `SAVE_FORMATS`, the one table of recording formats (Recording group, File > Save Recording). The Audio group's Notch (Off, Auto, Manual; the Whistle row under it holds Manual's frequency box or what Auto removes) and Low cut are greyed, their tooltips saying why, where they don't apply; its Tone squelch list holds Off, the 50 CTCSS tones and the 104 DCS codes (normal, then inverted)
- `receiver_bar.py` - Second toolbar row (MODE, FILTER, STEP) and the FREQ readout's digit tuning
- `level_meter.py` - Toolbar LEVEL meter with the draggable squelch threshold
- `tuning.py` - Steps, typed-frequency parsing, click-to-tune snapping, back/forward history
- `audio_chain.py` - Channel level, noise floor, squelch, audio AGC, noise reduction, CTCSS and DCS tone squelch (`ToneSquelch`; `feed_dcs` takes the discriminator audio before the DC blocker), the listening filters (`ListeningFilters`: the notch in AM, SSB, CW and narrow FM, the low cut in AM, SSB and narrow FM, neither in broadcast FM; narrow FM with a tone squelch set is heard from `SUBAUDIBLE_CUT_HZ`, 300 Hz, up), `AudioJoin` (mono or `(n, 2)` blocks), DC/I-Q clean-up
- `audio_sink.py` - Audio output on the device chosen in Preferences (a no-op without QtMultimedia); the main window's (`channels=2`) opens in stereo where the output can and plays mono on both channels, so stereo coming and going never reopens it
- `stereo_audio.py` - `StereoPath`: broadcast FM's stereo difference signal (`dsp/fm_stereo.py`) through copies of its chain's audio stages, so left = mono + difference and right = mono - difference line up to the sample; `usable(chain)`: a channel of at least 120 kS/s (`fm_stereo.MIN_RATE`: L-R reaches 53 kHz) through a filter of at least 150 kHz (`MIN_FILTER_HZ`; broadcast FM's is 200 kHz)
- `rds_station.py` - Background RDS of the tuned FM station (the name next to FREQ)
- `tool_tabs.py`, `status_history.py` - Maximize Tool Tabs; the status bar's last 20 messages, `MessageOverflow` (a message the status bar cuts short, shown whole just above it) and `HintFollower` (Qt-free: when a hint is said again because its cause changed)

Plots:
- `spectrum_widget.py` - Spectrum: click-to-tune, zoom, Auto/fixed range (zoomed in, Range > Auto fits the part in view), peak/min hold, FFT size and window, bookmark labels, right-click menu; the windows call `update_spectrum(trace)` with the trace alone (tests replace it with a one-argument callable)
- `waterfall_widget.py` - Waterfall: rows keep the frequency they were received on, history length, measurements, COLORS scale (zoomed in, Color Levels > Auto and Fit to Current Signals fit the part in view). Each row also keeps its frame's averaged spectrum for the hover: pass it (`add_line(..., mean_db=frame.mean_db)`, as both windows do), since a row with `noise_db` alone reads only bursts ("Peak") and noise
- `plot_readout.py` - What the hover says over either plot, in both windows: the averaged spectrum under the pointer judged against the noise measured on that same line, so plain noise reads "noise" at any FFT size and a signal its own level (at 32768 points a weak carrier only from about 10 dB over the noise in a bin, sooner zoomed in: see Important Notes); a waterfall row's strongest FFTs count only where they stand out of that level ("Peak": a burst shorter than the frame)
- `plot_view.py` - The zoom (x1-x16) both plots share

Tool tabs (Decoder, Bookmarks, Ham, Scanner, Signal, Info):
- `decoder_panel.py`, `decoder_views.py` - Decoder tab: messages, detail card, stats, Auto, Aircraft view (ADS-B), Text view (Morse). The table and Log keep the newest 1,000 messages; Export saves every message since Clear (`decoder_spool.py`, see Important Notes). From window code, untick Decode with `pause_decoding()` (as tuning away from a preset does), not `_enabled_check.setChecked(False)`, which counts as the user's own pause and carries over to the next protocol chosen; `paused_by_window()` tells such a pause from the user's (the decoder chip hides for it), and Close Recording back on a preset's channel turns on again the preset's decoder the recording turned off (`_resume_preset_decoder`). `DecoderChip` is the status bar's decoder chip (the scan chip is one too): short of room, it is shortened to fit, keeping at least its first part ("CW (Morse) · …"), with the whole text in its tooltip; `text()` stays whole
- `decoder_spool.py` - `MessageSpool`: every decoded message since Clear, as the CSV row Export writes, in a file of the system's temporary folder (`spool_folder()`)
- `bookmarks_panel.py`, `bookmark_editor.py` - Bookmarks tab: groups, filter, Edit Bookmark, CHIRP CSV; `bookmark_dcs`: the DCS code a channel's squelch listens for (CHIRP's DTCS, and the Cross modes that receive with a code; DTCS-R isn't supported)
- `scanner_panel.py` - Scanner tab, and the status-bar chip of a running scan (Stop Scan, Skip, Hold, Show Scanner); `scan_engine.py` its engine, fed by the display loop; `scanner_host.py` the window as its host; `scanner_dialog.py` scan band presets and table helpers (no dialog any more). Use both receivers (Alt+I, `scanner/both_receivers`; for Sweep once and Sweep continuously while the Second Receiver window runs): the engine shares a `SharedSweep` with a `ScanPartner` (`SecondReceiverPartner`, whose blocks come in through `partner_frame`): each first takes what only it tunes, then they meet from opposite ends (`SharedSweep.sweeps_up`: the one whose part of the range starts lower goes up, of two that start together the one whose part ends lower; where both tune the same part, such as the whole range, the main receiver goes up and the second comes down); a partner that leaves (stopped, tuned by hand, recording, a new sample rate, 3 s without samples: `NO_BLOCKS_S`) hands the rest to the main receiver (`drop_partner`; what only the partner tunes is left out, and the status bar names it). Stop on signal and Scan bookmarks use the main receiver only
- `signal_panel.py` - Signal tab (Identify, Signals, Scope, Bursts); `signal_tools.py` its window side, Track Signal (AFC) and Find Signals in Recording. Above 3.2 MS/s in None (I/Q) with a 100 kHz to 2.4 MHz filter, Identify listens to the receiver's channel (`receiver_channel_band`) instead of filtering the raw block again, and brings each capture down to its own rate on its worker

Devices:
- `device_dialog.py` - Device > Connect... dialog and `MockDevice`, the demo device (its docstring describes the simulated stations, `_DEMO_SIGNALS` lists them)
- `device_settings.py` - Device kinds, gain models, sample-rate tables, front end, per-kind memory
- `device_controls.py` - Keeps the Receiver group in step with the device: gain, live sample rate, ppm, direct sampling, bias tee

Recordings and files:
- `playback.py` - File > Open Recording: `FilePlaybackDevice`, the playback strip, `PlaybackController`
- `iq_files.py` - What a file holds; Import Options (headerless files), Recording Info and decode dialogs; `format_size`, how sizes are shown (decimal kB, MB, GB; Tools > Preferences has its own `format_bytes`, also decimal)
- `file_jobs.py` - `run_job` (worker thread, progress, Cancel) and the load, write and decode jobs; `rf64_note`, what a save message adds when the file became RF64
- `file_dialogs.py` - File dialogs that remember a folder per kind of file; the recordings folder
- `audio_recording.py` - Radio > Record Audio (48 kHz WAV of what you hear, after the notch and low cut: two channels when it starts on broadcast FM with Radio > Audio > Stereo on, kept for the whole take; RF64 past 4 GiB)

Transmit and second receiver:
- `transmitter.py` - Transmit engine: `Transmitter`, `TxJob`, `TxPower`, `AutoIdPolicy`, legality checks
- `tx_station.py` - `StationTx`, the window's transmit side (Ham ID panel, ON AIR, confirmation, automatic ID)
- `transmit_text.py` - Tools > Transmit Text: CW, RTTY, PSK31, ASCII FSK with Preview, Save, Transmit
- `second_receiver.py` - Device > Second Receiver... window (`SecondReceiverHost`, `ReceiverDsp`): each frame's DSP and plot spectrum run on a worker a frame behind the window (`FrameWork`, see Important Notes); it plays mono, has no notch, low cut or tone squelch (CTCSS or DCS), records SigMF through a `BackgroundWriter`, and sweeps with the Scanner tab's Use both receivers (`set_scan_feed`, `set_scan_muted`)
- `frame_pipeline.py` - `FramePipeline`: one frame's job at a time on a worker thread, a frame behind the display loop (`submit`; `collect` waits for the job in flight and re-raises its exception)

Presets, help and settings:
- `presets.py` - The one preset table: Radio > Band Presets, the Presets group, the welcome screen, the Scanner's bands
- `help_dialog.py` - F1 Keyboard Shortcuts: `SHORTCUTS` (the window's keys, checked against its real bindings), `PLOT_KEYS`, `PANEL_KEYS` (keys a list, field or window handles itself) and `MOUSE_ACTIONS`
- `first_run_wizard.py`, `version_info.py` - Welcome and Quick Start; About, Copy Version Info, Report a Problem
- `error_log_dialog.py`, `preferences_dialog.py` - Tools > Error History; Tools > Preferences
- `settings_store.py` - `GuiSettings`, a typed QSettings wrapper whose getters give the default for a damaged value (see Important Notes); `usable_bookmarks`
- `themes.py` - Dark and light palettes, stylesheet roles and tones

## Testing

Tests use pytest (about 170 modules): DSP, drivers against fake backends, the CLI, and GUI tests that build the real main window. The suite is large; run the modules for the code you change.

```bash
# GUI tests need no display (CI sets this for every run; some
# GUI test modules don't set it themselves)
export QT_QPA_PLATFORM=offscreen

# Run all tests
pytest

# Run specific test file, or matching tests
pytest tests/test_gui_tuning.py
pytest tests/test_gui_scanner.py -k bookmark

# Fail instead of skipping the GUI tests when PyQt6 is missing (as CI does)
SDR_REQUIRE_GUI=1 pytest tests/test_gui_chrome.py

# The antenna-array package's own tests (plain `pytest` runs tests/ only;
# CI runs these separately)
pytest packages/sdr-antenna-array/tests
```

- **`tests/conftest.py`**: an exception raised in a Qt slot or event handler would make PyQt abort the whole run ("Fatal Python error: Aborted", output lost); the conftest records it instead, and the test it happened in fails with the traceback (`tests/test_qt_callback_errors.py` checks this). It also runs pending `deleteLater()` deletions after each test, closes windows a module leaves open, restores the theme, and points the file dialogs' Documents and Pictures folders at temporary ones. Keys are not reset between tests: a `QTest` click with a modifier leaves the key down for the whole app (a later test's `selectRow` then extends the selection), so release it (`QTest.keyRelease(widget, Qt.Key.Key_Shift)` after a Shift+click, as the radio tuner's seek tests do).
- **Window tests** mostly build on `_WindowTestCase` (`tests/test_gui_chrome.py`): a real `SDRMainWindow` with in-memory settings (`_FakeSettings` in place of `GuiSettings`) and its timers stopped, so a test runs frames by calling `_update_display()`; `demo = True` starts the demo device. Code that creates `GuiSettings()` itself uses the real settings file (`~/.config/SDR Module Team/SDR Module.conf` on Linux); set `XDG_CONFIG_HOME` to a scratch folder to keep a run away from your own. On broadcast FM the main window's speaker gets `(n, 2)` blocks (Radio > Audio > Stereo is on by default), so a fake sink that collects them must take both shapes (as `_FakeSink` in `tests/test_gui_second_receiver.py` does).
- **`tests/test_final_*.py`**: the last fixes, found by using the finished app end to end, a module per area (`decode`, `ham`, `listen`, `perf`, `plots`, `record`, `scan`, `window`); `test_final_scan_check.py` holds what the check of the scan fixes found. What two areas' fixes do together, found on the merged build, is in `test_final_integration.py`, `test_final_integration_ham.py`, `test_final_integration_scan.py`, `test_final_integration_window.py` and `tests/test_integration_decode.py`; the listen, perf, plots and record areas added theirs to their own modules, and the last cross-area fixes are in `test_final_polish.py` and `test_final_finish.py`. Each regression test failed before its fix (a few guard tests pin what must not change); run the area's modules when you change its code.
- **Former known limitations** (removed since) have modules of their own, held to the same rule: broadcast FM stereo (`test_fm_stereo.py`, `test_gui_stereo.py`); the notch, low cut and DCS (`test_audio_notch.py`, `test_audio_low_cut.py`, `test_dcs.py`, `test_gui_audio_notch.py`, `test_gui_audio_low_cut.py`, `test_gui_dcs.py`, `test_gui_dcs_bookmarks.py`); ADS-B below 2.5 MS/s and its budget (`test_adsb_low_rates.py`, `test_adsb_bounded_cost.py`); FLEX at 3200 and 6400 bps (`test_flex_speeds.py`, `test_flex_phase_speed_shown.py`); Morse's lone E and T (`test_morse_lone_letters.py`); Export's spool (`test_gui_decoder_spool.py`); the plots' readings and zoom (`test_plot_readouts.py`, `test_gui_waterfall_zoom_levels.py`); Identify (`test_identify_weak_signals.py`, `test_identify_reuses_channel.py`); the status bar (`test_gui_status_overflow.py`, `test_gui_hint_causes.py`); RF64 (`test_wav_rf64.py`, `test_gui_wav_rf64.py`); the take's writer and the second receiver's worker (`test_background_writer.py`, `test_gui_recording_writer.py`, `test_frame_pipeline.py`, `test_gui_second_receiver_thread.py`); the wideband scan (`test_wideband_sweep.py`, `test_wideband_scan.py`, `test_dual_sdr_wideband.py`, `test_cli_scan_dual.py`, `test_demo_two_devices.py`, `test_gui_scan_both.py`).
- **Timing-sensitive tests are made deterministic**, not given wider margins (they must pass on a busy computer):
  - run the display loop on a fake clock: `_FakeClock` (`tests/test_gui_round3_main.py`) patched in as the module's `time` (`mock.patch.object(main_window, "time", clock)`; `second_receiver` too) and advanced between `_update_display()` calls, as the playback and second-receiver Listen tests do. For a computer that can't keep up, advance it inside the frame by what the work would cost: `_FakeDemo` (same module) costs fake time per sample, and `TestPlaybackWhileOverloaded` (`tests/test_final_perf.py`) wraps `_process_block` to cost `load` times the time its samples cover at 1x, and a recording's `read_samples` to wait for the disk; `_Pacer` can also be driven on its own with made-up times (`TestSlowRecordingRidesOutHiccups`)
  - read every demo sample: patch `device_dialog._MAX_LAG_S` to `1e9`, or the demo device skips ahead to the wall clock on a slow run and its timed signals move (`_SignalCase` in `tests/test_gui_signal_tab.py`); fix a frame's size by replacing `win._pacer.demo_request`, and seed the demo's noise with `win._device._rng = np.random.default_rng(n)`; run Identify's jobs as they are handed over (`_RunAtOnce` in `tests/test_final_plots.py`, in place of `SignalTools._pool`), so which takes are judged doesn't depend on its worker thread
  - count work instead of timing it (`tests/test_final_perf.py` counts the outputs a filter computes and the samples the plots analyze); bring about the state a test needs (a progress dialog shown, a transfer in flight queued, a receiver silent for 3 s during a scan: move the scan engine's `last_block_at` back) instead of sleeping; stop a panel's own refresh timer when the test plants a reading, or call the timer's slot yourself (`ScannerPanel.refresh()`)
  - wait for the threads behind the frames, never sleep for them: on a fake clock the frames outrun real time, so flush the take's writer before the test reads the take (`win._samples_buffer.flush()` after each frame, as `tests/test_review_mainloop.py` does); the second receiver's `_frame()` hands each frame's DSP to its worker and presents the frame before, while `_process(samples)` runs one frame start to finish on the calling thread; to test what must wait for the frame in flight, hold the worker's job on a `threading.Event` (`tests/test_gui_second_receiver_thread.py`)
- **Damaged settings**: `_IniSettings` (`tests/test_final_window.py`) runs the real `GuiSettings` on an INI file in a temporary folder, written as a damaged settings file would be. A damaged value is logged once: per key for a `GuiSettings` instance, but once per process for a bad recordings folder (`file_dialogs._BAD_FOLDERS_LOGGED`), so a test that asserts that warning patches the set empty.
- **WAV past 4 GiB**: patch `wav_file.RIFF_SIZE_LIMIT` lower rather than write 4 GiB (`tests/test_gui_wav_rf64.py`); the tests that do need a 5 GiB file make a sparse one, and are skipped where the file system makes no sparse files.
- **Fake hardware backends** (no hardware or driver packages needed):
  - `tests/test_rtlsdr_driver.py` - `FakeRtlSdr`, installed as the `rtlsdr` module; it has only pyrtlsdr's real methods, so a wrong driver call fails. `tests/test_gui_device_settings.py` has one with tuner gain tables and front-end calls for window tests.
  - `tests/test_hackrf_driver.py` - `FakeHackRFDevice`, installed as `python_hackrf`; answers only to real python_hackrf method names.
  - `tests/hackrf_fakes.py` - `ThreadedFakeHackRF` (its TX callback runs in its own thread like libhackrf's USB thread; optionally paced or stalled), `open_hackrf(monkeypatch, rate)`, and helpers that decode what went on air (`analyse_cw`, `analyse_mcw`); used by the transmit and device tests. `ThreadedFakeHackRF(rx_stations={freq_hz: dBFS})` also receives, a transfer at a time from a thread of its own at the sample rate (scans through the real `HackRFDevice`); without `rx_stations` it receives nothing.
  - `tests/tuner_fakes.py` - `FakeTuner`: carriers at fixed frequencies (`STATIONS`) around wherever it is tuned, through the parts of the device API a sweep uses; like hardware, a retune leaves one block from the old frequency behind (queued at once, or `in_flight_s` later). Used by the wideband scan tests (`test_wideband_scan.py`, `test_dual_sdr_wideband.py`) and the `sdr-scan scan --dual` tests (`test_cli_scan_dual.py`).
- **Signals**: `MockDevice` (the demo device) for receive tests, `PolledDevice(MockDevice(), name)` for dual-SDR tests (two of them run the wideband scan without hardware), `tests/signal_synth.py` (synthetic signals, `add_noise`), `tests/text_demod.py` (independent demodulators for the text modes), `tests/voice_synth.py` (voice with a gliding pitch through formants: its harmonics move as speech's do, so it tells a steady whistle from voice where band-limited noise can't; `CLOSE_VOWEL` puts the first formant at 300 Hz).
- **Checked against this project's own models, not the air**: the hardware paths (HackRF transmit, RTL-SDR direct sampling, the bias tee, the dual-SDR scans through the real drivers) are tested against the fake backends, and FM stereo, DCS and FLEX at 3200 and 6400 bps against this project's own encoders and synthetic signals (ADS-B below 2.5 MS/s against modelled replies and the demo's airliner); no off-air recording or real radio was used. A decoder and an encoder written from the same reading of a standard agree even where that reading is wrong (DCS polarity, FLEX's Gray mapping and phase order), so change such a detail only against the standard, never to make a round trip pass.
- **Demo stations** (MHz) to try features on: 100.1 FM with RDS, in stereo (the other demo FM stations send their 3.7 kHz tone on the left only), 144.390 APRS, 152.240 POCSAG, 931.9375 FLEX (1600, 3200 and 6400 bps), 131.550 ACARS (AM), 1090 ADS-B, 145.800 SSTV (every 75 s), 145.290 repeater with a 100.0 Hz tone, 442.100 repeater with DCS 023 N (8 s in every 12), 99.820 CW beacon, 433.920 OOK remote, 438.500 drifting handheld; AM broadcast on the radio tuner's six AM presets (660-1260 kHz). The `MockDevice` docstring describes them; `_DEMO_SIGNALS` in `gui/device_dialog.py` lists every simulated signal.

## Hardware Support

- **RTL-SDR**: 500 kHz - 1.766 GHz (below 24 MHz through direct sampling), RX only, 1.0 - 2.56 MS/s in the GUI
- **HackRF One**: 1 MHz - 6 GHz, TX/RX (half duplex), 2 - 20 MS/s. Its firmware switches the RF amp and antenna power off whenever it stops; the driver sends the amp setting again before every RX or TX stream, powers the antenna port (bias tee) again for each reception, and switches it off before transmitting (never powered on the air).
- **Demo device**: simulated signals, 1 - 3.2 MS/s, can't transmit

Dual-SDR operation modes (`core/dual_sdr.py`; `OperationMode.implemented`, `DualSDRController.supported_modes()`):

| Mode | What it does | `DualSDRController` | In the GUI |
| --- | --- | --- | --- |
| `DUAL_RX` | Two receivers on two frequencies | Implemented | Device > Second Receiver... with any free RTL-SDR, HackRF One or a second demo device |
| `FULL_DUPLEX` | RTL-SDR RX + HackRF TX | Implemented | The second receiver keeps receiving while the main window's HackRF transmits |
| `TX_MONITOR` | Monitor own transmission quality | Implemented (attenuate the receiver) | Second receiver's Transmit monitor group: Monitor TX, Follow transmissions |
| `WIDEBAND_SCAN` | Both devices sweep one range at once, each its own part | Implemented: `start_wideband_scan()` / `stop_wideband_scan()` (`core/wideband_scan.py`); `sdr-scan scan --dual` | Scanner tab > Use both receivers (Sweep once or continuously, with Device > Second Receiver... running) |
| `RELAY` | Receive-and-retransmit | Deliberately not implemented: `set_mode` raises `NotImplementedError` saying why (retransmitting what you receive is restricted almost everywhere, e.g. FCC 97.113) | No |

`DualSDRController.use_devices(rtlsdr, hackrf)` takes devices that are already open (two `PolledDevice(MockDevice(), ...)` try dual receive and the wideband scan without hardware). While a wideband scan runs the devices are its own: the other modes, `set_mode`, `use_devices` and `set_rtlsdr_frequency` / `set_hackrf_frequency` are refused (with the reason in the log), and `stop_all()` stops the scan first. Two devices at most, each at its own sample rate and gain (`start_wideband_scan` keeps what each is set to; `sdr-scan scan --dual` takes one `--rate` or two, `2.4e6,20e6`, and its `--gain` sets the same gain on both); each reading is in the dBFS of the receiver that took it (levels aren't calibrated between them). The GUI's second receiver doesn't use the controller: its `ReceiverDsp` runs its own `_ReceiverChain` (the main window's channel filter and demodulators) on its own device's blocks, and the transmissions it monitors come from the main window's transmitter. Nor does the Scanner tab's Use both receivers: the scan engine tunes the second receiver window step by step and takes the blocks its display loop reads (see Key Modules: `scanner_panel.py`).

## Important Notes

### The display loop
- A 33 ms timer calls `SDRMainWindow._update_display`: `_read_block` takes what arrived since the last frame, and `_process_block` hands that one block to everything that works on the signal (plots, level and squelch, receiver chain and audio, RDS, radio tuner, `SignalTools`, the I/Q recording, S-meter, live decoder, scanner)
- Hardware is polled without waiting (`read_samples(..., timeout=0.0)`); the demo device and a recording playing back are paced to real time by `_Pacer` (a recording never runs ahead)
- `_Pacer.measured` times each frame against the signal time it covers (leaving out the time a recording's read waited for the disk). When the computer can't keep up (`_pacer.overloaded`), frames drop to one block and a full frame is tried once a second to recover; meanwhile the Audio button reads Paused (the status bar says why, at most every 30 s), a running live decoder says it misses part of the signal (not while a Scanner scan runs: `_sync_decoder_skipping`), the Bursts page stops logging and says why, and the waterfall draws the one-block frames' rows as bright as the others without letting them set how bright that is (`_plot_frame` passes `add_line(..., usual=not self._pacer.overloaded)`; the second receiver does too). A recording playing back loses nothing by waiting, so its frames don't drop to one block: they still take what real time owes at 1x (or at a slower speed chosen), at most 50 ms of it a frame (`_Pacer.paced_request`), so it plays at that pace where the computer can, else as fast as it manages (at 1x its audio pauses only below half of real time), and the decoder, which hears all of it, doesn't say it misses part (`_samples_skipped`); nor does the Bursts page pause (`_bursts_frame` pauses only for a device), and Paused's hint gives no advice to lower the sample rate, which a recording can't change (`_realtime_hint`). An I/Q recording still gets every sample (every queued hardware transfer, and the demo's backlog through `_read_demo`; only a disk a second behind loses blocks, see Never block the GUI thread), and what a source skipped (`dropped_samples`) is reported as lost
- New per-frame work goes in `_process_block`, does nothing while its view is hidden or its feature is off, and stays small (measured on a 4-CPU computer, on noise and on the demo's stations, a live decoder takes about 0.3-2.5 ms of the 33 ms at 2.4 MS/s, ADS-B the most; below 2.5 MS/s ADS-B takes about 5 ms in a busy sky, its budget capping any input at about 7-8 ms; a HackRF delivers up to 20 MS/s). Count time in samples where you can, as `scan_engine.py` does, so tests can drive it block by block
- At 20 MS/s a frame holds about 670,000 samples, so keep each part's cost bounded: filter down to the channel first and compute only the outputs kept (the receiver's `_FirDecimator` takes one dot product per kept output wherever that is cheaper than filtering every sample by FFT); reuse the receiver's channel rather than filtering the full-rate block again (the Decoder's RDS takes it through `RDSLiveDecoder.process_channel`, Identify through `receiver_channel_band`); cap a long frame's work (the plots analyze at most `_PLOT_MAX_SAMPLES`, in evenly spread segments); give work that grows with what is received a budget per second of signal (ADS-B's Mode S fit below 2.5 MS/s: `ModeSDemodulator.FITS_PER_S`, `SEARCHES_PER_S`; beyond it the clearest replies are read first); go through a large block in cache-sized pieces (Remove DC spike and Balance I/Q). `tests/test_final_perf.py` (and `tests/test_adsb_bounded_cost.py` for ADS-B) pins these by counting the work
- Nothing else takes samples from the main window's device: the scanner, decoders and signal tools get the loop's blocks (the scanner's host only discards transfers queued before a retune, as `_retune_lo` does)
- Measurements don't depend on the plots' FFT size, window or zoom: LEVEL and the squelch use the channel's own power, the noise floor (Auto squelch, the S-meter's SNR) a fixed 2048-point spectrum (`_power_spectrum_dbfs`)
- The plots' hover readings (`plot_readout.py`) judge the averaged spectrum against the noise measured on that same line, so plain noise reads "noise" at any FFT size; a pixel that holds many bins (about 33 at 32768 points on a 1000 px plot) still needs a carrier about 10 dB over the noise in a bin before it reads as a signal every time (less once zoomed in, or at a smaller FFT size). The zoom changes only what Range > Auto and Color Levels > Auto fit, never a reading: the hover, the measurement box's Peak and the SNRs judge against the whole span's noise

### Tuning: the VFO inside the span
- The LO (`_lo_hz`, the device's center frequency) is the middle of the plots; the tuned frequency (the VFO) can sit anywhere in the span. The receiver chain, decoders and S-meter work at `_vfo_offset_hz()`
- Tune through `set_frequency` / `_tune_to`, never the device's `set_frequency`. They move only the VFO while the channel fits the inner 80% of the span (`_USABLE_SPAN_FRACTION`) and retune the LO otherwise (always, with Radio > Keep Tuned Frequency Centered), to where `_lo_for` puts it: on the tuned frequency, except below half the sample rate, where it goes to half the sample rate so the span starts at 0 Hz (below about a tenth of the rate, only as high as keeps the channel in the usable span, so the span starts a little below 0 Hz). `_tune_to(recenter=True)` puts the LO exactly on the frequency (the Scanner's steps and the transmit path rely on it)
- A frequency out of the device's range, outside a recording that is playing (`_lock_lo`), asked for while transmitting, or outside the usable span while an I/Q recording has samples is refused, with the reason in the status bar and everything left on the frequency tuned (`_refuse_tuning`: red, but amber while transmitting, as every "not while transmitting" notice is). A take holds one center frequency, as it holds one sample rate and one device: while it has samples, tuning inside the span moves only the VFO (Keep Tuned Frequency Centered too), and Device > Connect..., Disconnect and Use Demo Device ask before they end it (`confirm_stop_recordings` in `device_controls.py`)
- After a hardware retune, `_retune_lo` drops the samples taken on the old frequency. Recordings are labeled with the LO (their center frequency), not the tuned frequency; a SigMF take's description names the tuned frequency and mode (`_take_details`)

### Never block the GUI thread
- Long jobs run in workers with progress and Cancel: `file_jobs.run_job(parent, title, label, work)` runs `work(report)` in a `QThread`, shows a progress dialog after 400 ms, holds all keyboard and mouse input except the dialog's own (Cancel) meanwhile (the spectrum keeps running) and returns a `JobResult`. Convert, save and decode recordings, Find Signals in Recording, the Decoder's Export and Transmit Text > Save use it; a cancelled or failed write leaves no partial file. While a job runs, `file_jobs.busy_with()` gives its title, and the Decoder's and the audio's hints name that job, not the device or the sample rate, as what makes the display loop skip, also for as long after it as that skipping is still measured (`_busy_job_now`, `_JOB_SKIP_S`), then in the past tense ("Decode Recording kept this computer busy, ..."; `busy_with()` no longer names it). So title a job with its command's name ("Decode Recording", "Save Text Signal"): the hints read "Save Text Signal keeps this computer busy, so audio is paused meanwhile." (`TestEveryJobTitleReadsInTheHints` in `tests/test_final_finish.py` checks every `run_job` title)
- The transmitter streams from its own thread, polled by a timer; Identify and burst analysis run in a thread pool, and the bursts worker spends at most a quarter of the signal's time taking bursts apart (`ANALYSIS_SHARE`, timed in its own processor time; beyond it, a burst is listed without its analysis)
- Don't make the display loop wait for a thread that may not get a processor on a busy computer: the plots hand half of a large frame's FFTs to a helper thread, and the loop does that half itself when the helper hasn't started on it
- The I/Q recording is written by the take's own writer thread: `_capture` only copies each block into `_SampleStore.queue()` (a `BackgroundWriter` that holds up to a second of signal, 200 MB at most: `_capture_queue_bytes`), so a frame never waits for the disk. A block the queue has no room for (a disk that can't keep up) is left out of the take, counted and said as samples a device drops are ("The recording has gaps: ..."). A write that failed on the writer thread is raised once, on the GUI thread, by the next `queue()` or by `_take_write_failed()` from the status timer, and stops the recording saying why. Stop first waits until the queue is written (`_take_write_failed(wait=True)`), so the take is whole before it is offered for saving or saved automatically; the store's readers (`samples()`, iterating, saving) wait for the queue too, and the free-space check counts what is still queued, so read a take only through its store. The second receiver's recording writes the same way (`_record_writer`)
- The second receiver's frame (`_frame` -> `_feed`) runs in three parts: the GUI thread's (`_frame` reads; `_frame_job` records, gives the real-time hint and decides whether the plots want the frame), `FrameWork` on its `FramePipeline` worker (channel filter, demodulation, level and squelch, the plots' spectrum) and `_present` on the GUI thread a frame later (paint and play). `_feed` collects the frame in flight before it submits the next, waiting while that one still runs: the worker is never more than a frame behind, and a slow worker slows the second receiver's frames (its pacer sees them) rather than piling up work. While a frame is in flight, what the DSP and the analyzer keep belongs to the worker: call `_settle_frames()` (it presents the frame in flight) before anything resets or reconfigures them (a new sample rate or FFT size, a retune, Stop); the settings a user changes (mode, filter, tuning, squelch level) are plain values the next frame reads. Put new per-frame work in the part for its thread. Both windows share one GUI thread, and the worker competes with it for Python's GIL: it takes part of the second receiver's work off the GUI thread, not all of it
- Widgets are touched only from the GUI thread: a worker's job (`FrameWork`) and a writer's write function never touch Qt; they hand their results back to it

### Broadcast FM stereo: what hears stereo, what hears mono
- Only what you hear and a stereo Record Audio take get left and right. `_ReceiverChain.demodulate()` returns the mono program, (L+R)/2, in every mode (the same samples as without stereo); with the chain's `stereo` set (the window sets it from Radio > Audio > Stereo and who listens, below), on a broadcast deviation and through a filter `StereoPath.usable` accepts, `stereo_audio` holds the block's `(n, 2)` left and right whenever there is stereo to play (None otherwise). `_route_audio` gives `heard` (left/right, else the mono audio) to the speaker, its fade-out and the Record Audio take. Everything else keeps the mono audio or the I/Q: the squelch, LEVEL and the S-meter (the channel's own power), the tone squelch and SSTV (`pre_nr_audio`; DCS reads `fm_raw_audio`), the clipping warning, RDS, the decoders, and the Second Receiver window, which plays mono
- Stereo is decoded only for someone: for the speaker (and its fade-out), a two-channel take, and the STEREO badge and the radio tuner's lamp while a broadcast station is above the squelch, but not while nobody can hear this receiver (`_audio_held()`: a recording playing at a speed other than 1x, a computer that can't keep up, the Second Receiver's Listen playing only that receiver) unless a stereo take records it, and never for SSTV, a tone check or a mono take alone. Through a filter too narrow for it the badge says so without decoding anything
- `StereoPath.sync` copies the chain's own audio stages (`_build_audio_stages` and their `_phase`), its resampler's `_time` and its DC blocker's state; left and right line up only while those copies match, so a change to those classes must keep them in step. The notch and low cut leave broadcast FM untouched (`_listen` returns its audio as it came): if they ever filter it, the difference signal needs the same filter, in `StereoPath.process` before its noise reduction
- The speaker opens in two channels and plays mono on both, so stereo coming and going never reopens it (no gap, no click); `AudioJoin`, the sink and `AudioTake` take mono or `(n, 2)` blocks. A take keeps the channels it started with: two when it starts on broadcast FM with Stereo on (its mono stretches on both channels), else one

### Export: every decoded message since Clear
- The Decoder's table and Log keep the newest 1,000 messages; each message is also appended, as the row Export writes (`_export_row`, `EXPORT_COLUMNS`), to a `MessageSpool` file in the system's temporary folder, written through, so a row is there whole or not at all. Export (a `run_job`, "Export Decoded Messages") writes the spool's rows as they are, then the table's newer messages: a new column means changing `EXPORT_COLUMNS` and `_export_row` together
- Clear empties the spool, closing the window deletes it (`discard_spool` in `closeEvent`), and a spool a crashed session left is deleted by the next one (`remove_stale_spools`; a spool in use is locked). When the disk fills, the spool keeps what it has and takes no more until Clear: `spool_failed` puts one warning in the status bar (and Error History), and Export says how many messages it saved and why the others are missing (those between the last one spooled and the table's newest 1,000 are lost)

### Settings survive damaged values
- A settings file can hold anything (a hand edit, another version). `GuiSettings` getters give the default, logged once per key, for a value of the wrong type, a NaN or infinite float or an integer beyond 32 bits; saved bookmarks come back through `usable_bookmarks` (an entry that isn't a bookmark is skipped, one without a usable frequency is listed at 0 Hz)
- The getters can't know a key's range, its choices or the shape of JSON stored in it: check those where you read it and fall back to the default, or to the nearest valid value where that is harmless (the end of the tuning range; never the costliest FFT size or a squelch that never opens), and log it. The plots' saved display settings (JSON) show how: `apply_display_settings(settings, defaults=...)` reads each yes/no with `saved_flag` (never `bool()`, which reads "no" as yes) and each number with `saved_number`, and logs each damaged value with what it used instead (`log_damaged_setting` in `spectrum_widget.py`; an average or a history length the menus don't offer gives the nearest one). JSON of another shape than the one saved (a list or a number where an object belongs) is as damaged as JSON that can't be read: give the defaults and log it, as the plots' restores in both windows (`_saved_choices` in `second_receiver.py`) and the Ham ID, QRP and radio-preset restores do. Restore each value on its own (`restore()` in `_restore_audio_settings`, which also brings back the Notch, its frequency, the Low cut and the DCS code: an unknown choice reads Off, a notch frequency out of range its box's nearest end, a damaged DCS code is ignored, each logged): a damaged setting must never stop the others, the app from starting or a window from opening
- Safety settings fall back to the safe side: an unreadable `tx/confirm` asks before transmitting

### Both themes
- Every color comes from the `Palette` in `themes.py` (Catppuccin Mocha dark, Latte light). Tag widgets with `set_role` / `set_tone`; in `paintEvent` read `get_palette()` and drop cached pixmaps on `theme_notifier().theme_changed`. Never hard-code a color
- Check new UI in both themes (View > Theme), at the minimum window size, 1024x640, and at 1366x768 (a laptop screen); the Second Receiver window at its own minimum, 860x560. At 1024 px the status bar keeps 220 px for messages (`_MESSAGE_MIN_WIDTH`): a status-bar chip whose text can grow should give way rather than widen the window, as the decoder and scan chips (`DecoderChip`) do. A message cut short there shows whole just above the status bar while it lasts (`MessageOverflow`, see Keys, menus and messages), in its tone in both themes

### Transmit only through the transmitter
- Everything the GUI sends goes through `StationTx.transmit(TxJob)` -> `Transmitter`, the only GUI code that calls the HackRF's `start_tx`. Send ID Now and Radio > Send Station ID reach it through `StationTx.send_station_id`; Transmit Text and any other job through `SDRMainWindow._transmit_job(TxJob)`
- That path checks the device can transmit (the demo device and RTL-SDR refuse, with the reason), the license class, band segment and protected frequencies for the emission's own mode and occupied bandwidth (CW is allowed in every segment the class holds, FCC 97.305(a)), a callsign for amateur transmissions, and the power limits. It asks before each transmission until "Don't ask again" is ticked (`tx/confirm`; Ask before transmitting in Tools > Preferences turns the question back on), adds the automatic station IDs, shows ON AIR, pauses receiving (half duplex), puts the HackRF back afterwards and ends with the outcome in the status bar
- While on the air the transmit state wins: `_show_tx_state(True)` (called by `StationTx` when a transmission starts, and by `_refresh_state_ui` while `_tx_busy()`) turns Start and Radio > Start Receiving off and dims the plots with "Transmitting - receiving resumes when it ends", or "Transmitting - press Space to receive once it ends" when the receiver was stopped before (it stays stopped: `_tx_resumes_rx()`); `_refresh_state_ui` turns them back on afterwards. Update Start and the plots' state through `_refresh_state_ui`, so a transmission always shows
- A `TxJob`'s source can make its samples as they are read (`read(n)`, `total_samples`), so a long transmission never has to sit in memory
- Don't call `start_tx` / `write_samples` from GUI code, and don't loosen a check to make a test pass: use the fake HackRF backends

### Keys, menus and messages
- `_add_action` makes a menu action the only owner of its shortcut; `tests/test_gui_coherent_window.py` fails on a key bound twice, and `tests/test_final_window.py` on an access letter (`&`) used twice in one menu
- Keys that must leave a focused slider, list or text field alone (Space, Left/Right, `[` `]`, P, M, Home, Page Up/Down) are handled in `keyPressEvent`, not bound as shortcuts; a menu item shows such a key after a tab (`"&Start Receiving\tSpace"`). A window shortcut that a focused widget should keep is handed to it in `SDRMainWindow.event`, which accepts its `ShortcutOverride` (Ctrl+Up/Ctrl+Down: Volume Up/Down, but ten steps in a number field and a row up or down in a list)
- A menu item that changes a control panel setting follows its control: checked as it is, and disabled while it is (`follow()` in `_add_listening_menus`: Radio > Mode, Radio > Squelch, and Radio > Audio's Noise Reduction, Audio AGC, Notch and Low Cut). Radio > Audio > Stereo has no control in the panel: it is a checkable action of its own, saved at once (`audio/stereo`)
- List every new key in `help_dialog.py`: a window key in `SHORTCUTS` (F1 checks the table against the real bindings), a key a focused plot handles in `PLOT_KEYS`, and a key a list, field or window handles itself in `PANEL_KEYS` with an id. `tests/test_gui_help_keys.py` presses each `PANEL_KEYS` row on its real widget in a `test_key_<id>` test and fails when a row has none
- A refusal says what and why ("2400.000 MHz is out of range: RTL-SDR tunes 0.5–1766 MHz."). Use `_show_status_message(text, tone, ms)` / `_show_status_error`. A long action ends with its outcome in the status bar, never a stale "Preparing...", and a message that stops being true is replaced or cleared (a refusal by the next good tune, "Recording armed..." once samples arrive)
- A message too long for the status bar (with every chip shown it keeps 220 px at 1024 px; at 1366 px messages over about 70 characters don't fit) also shows whole in a panel just above it for as long as it shows (`MessageOverflow`, the window's `_message_overflow`): starting above the message's first letter, in its tone, wrapped to the window, five lines at most (then "…"; the message label's tooltip and the status history hold all of it), and above the playback strip while a recording plays (`keep_clear_of`). Nothing else moves; a click on it, or Esc in the window (`keyPressEvent`, where Esc still returns the focus to the plots; a focused control that uses Esc, such as the Decoder's list with a message selected, the Bookmarks filter with text in it or the waterfall's measurement, takes the first one), hides it until the next message. It follows `_show_status_message` and `_clear_status_message`, which also make the message the label's accessible description, so never set the message label's text directly. Still put what matters first: the panel covers the bottom of the plots and the right column while it shows
- A hint that names its cause (why a live decoder misses part of the signal, why audio is paused) is said once per episode, and again, once, when the cause changes while the status bar still shows the hint and the new cause has held a second (`HintFollower`: `_skip_follow`, `_audio_follow`, `HINT_SETTLE_S`). Blame this computer only while it is overloaded or was within the seconds real time is judged over (`_computer_behind()`), never for an overload from before a new sample rate, device or recording
- Tools > Error History lists problems only. It takes every WARNING (or worse) any logger writes, and the status bar logs what it shows in the "warning" tone or with `_show_status_error` unless you pass `log=False` (a refusal `_show_status_message` shows in "danger", such as a frequency out of range, is not logged): pass it when you logged already (a problem is listed once) and for a notice that is no problem, such as what to do first ("ID not sent: enter your callsign in Ham > Ham ID first."), what the app did as asked (a scan stopped by Space, Open Recording or a new sample rate; "Recording paused", "Recording armed...") or what a feature can't do in the mode chosen (Track Signal in SSB, I/Q or broadcast FM). A part that logs its own problems shows its messages with `log=False` (`WindowScannerHost.scan_message`: the scan logs a step that broke, and a receiver that returns no samples, whichever tab is shown). Warn only about what is in use: a paused decoder doesn't warn that the sample rate is too low for it. List the same problem once: a decoder that can't work at the rate is logged once for the device and once for a recording until the decoder or the rate changes (`_decoder_logged`), though the status bar, the Decoder line and its chip say it at every rebuild. Tests check that a notice stays out with `assertNoLogs(level="WARNING")`
- One action, one message: an action that sets the mode along the way (the band plan, a preset, a bookmark's or the Decoder's Tune to, the Signal tab's Apply Mode, going back when a recording closes) sets it quietly (`_quiet_mode_change`; presets: `_applying_preset`) and says once what it set; the mode change alone would first name the mode's last filter ("AM: 6 kHz"). A message that something said along the way would replace carries it instead: Open Recording in another band leads the band plan's line with "Playing fm8.sigmf-meta (98.9–101.3 MHz) · " (`_listen_by_band_plan(..., lead=)`, `playback.playing_line`), and in the same band puts the decoder's note after the file ("Playing air125.sigmf-data (...) · ACARS decoder off. Click ..."; `_release_note`), and a speed other than 1x says the audio's own hint at once and marks it said (`set_speed`). A closing message carries what its steps just said rather than replacing it (Close Recording: "Closed aprs2m.sigmf-data. Back to Demo Device (simulated signals). AX.25/APRS decoder off: ..."), and so does a message that would replace one just said: a tune that turned a preset's decoder off leads its own line with "ACARS decoder off · " (`_keeping_release_note`, once), and a scan stopped by a tune from another tab says "Scan stopped · " before the tune's own message (`_scan_frame`): one posted before the tune is done, back in the event loop or at the next frame (`_tune_said`), not one something else posts after it
- Show frequencies with `format_frequency` (`main_window.py`: kHz below 2 MHz, MHz with 3 to 6 decimals above; the Signal tab's `format_mhz` and the Scanner's `_say_freq` name them the same way, while a list's column in MHz stays in MHz) and sizes with `iq_files.format_size`; use American spelling (`TestTheWindowsWords` in `tests/test_final_window.py` checks every text the main window shows) and the controls' own names ("Filter", not "Bandwidth")

### General
- Hardware drivers are optional; graceful degradation if missing
- Closing the window releases and closes the device, discards the temporary recording file and the Decoder's spool, and stops the plots' FFT helper, one step at a time (`quietly()` in `SDRMainWindow.closeEvent`): a step that fails is logged and the next still runs, so the device is closed. The parts' `shutdown()` calls before those steps are not wrapped: a part with background workers must stop them there without raising, and ignore frames after it (`SignalTools.shutdown`)
- Create a widget you hand to a dialog with the dialog as its parent (`QCheckBox("Don't ask again", box)` before `box.setCheckBox(...)`): PyQt's `QMessageBox.setCheckBox` leaves a parentless box to Python, which deletes it while the dialog still holds it, and a restyle (a theme change) then crashes
- mypy ignores errors in `sdr_module.gui.*` and `sdr_module.ham.gui.*`; the rest is type-checked in CI
- Use demo mode for development without physical SDR devices (`--demo`, or Device > Use Demo Device)
