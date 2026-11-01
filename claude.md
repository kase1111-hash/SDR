# Claude.md - SDR Module Project Guide

## Project Overview

SDR Module is a Software Defined Radio framework for dual-SDR operation using RTL-SDR and HackRF One devices. It provides signal analysis, visualization, classification, and protocol decoding capabilities for amateur radio operators and signal analysts.

Most features are used through the PyQt6 GUI (`python -m sdr_module.gui`): spectrum and waterfall, a receiver with audio, live decoders, recording and playback, a scanner, signal identification, a second receiver window, and transmitting on a HackRF One with license and power checks. The `sdr-scan` CLI covers scanning, decoding, encoding and CHIRP channel files. The demo device simulates signals (including decodable traffic), so receiving, decoding, recording, scanning and the signal tools work without hardware; transmitting needs a HackRF One, and each device's own gain and front-end settings need that device.

**Version**: 0.2.0 (Beta)
**Python**: 3.10 - 3.14
**License**: MIT

## Directory Structure

```
src/sdr_module/          # Main source code
├── core/                # Device management, dual-SDR controller, config, frequency manager
│                        #   (presets, TX lockouts, licenses), band plan, CHIRP CSV
├── devices/             # Hardware drivers (RTL-SDR via pyrtlsdr, HackRF via python_hackrf)
├── dsp/                 # Signal processing (spectrum, demodulators, filters, live decoders,
│                        #   sweeps, signal ID, bursts, AFC, recording)
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
5. **Qt-free logic** - What the GUI computes lives in modules without Qt where it can (`dsp/`, `core/`, and `gui/tuning.py`, `audio_chain.py`, `scan_engine.py`, `device_settings.py`, `plot_view.py`, `rds_station.py`), so it is tested a block at a time without a window. Panels show state and emit signals; the window, or a host object such as `WindowScannerHost`, acts on them.

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
- `frequency_manager.py` - TX lockouts, license profiles, band privileges, RX presets (`RX_PRESETS`; a preset can name the decoder it turns on, and the window turns that decoder off again once you tune away from where it is heard: `_release_preset_decoder`)
- `sample_buffer.py` - Thread-safe circular sample buffers

### DSP (`dsp/`)
- `spectrum.py` - FFT-based spectrum analyzer
- `frame_spectrum.py` - Spectrum of a whole display frame (mean for the spectrum, max for the waterfall; a frame longer than `max_samples` in evenly spread segments), FFT sizes and windows
- `demodulators.py` - AM, FM, SSB, CW, OOK, FSK, PSK, GFSK, MSK, QAM
- `filters.py` - FIR filters, resamplers, AGC, squelch, CTCSS detection, noise reduction
- `nco.py`, `channelizer.py` - Streaming frequency shift and decimation: a channel out of the span
- `live_decoders.py` - Live decoders on raw I/Q (`LIVE_DECODERS`: POCSAG, FLEX, AX.25/APRS, ADS-B, ACARS, RDS, CW); `create_live_decoder(..., recording=True)` is for a recording playing: a rate too low for the protocol is then named as the recording's ("this recording is 1 MS/s"), with no advice to change the sample rate, which a recording can't
- `protocols.py` - Protocol decoders (POCSAG, AX.25/APRS, RDS, ADS-B, FLEX, ACARS), bit slicer, BCH, APRS parsing
- `protocol_encoders.py` - POCSAG, FLEX, AX.25/APRS, ACARS, Morse and RDS transmitters (tests and the demo device; there is no ADS-B encoder, the demo builds its Mode S frames in `device_dialog.py`)
- `rds.py`, `morse.py`, `stereo_pilot.py` - RDS physical layer, streaming Morse decoder, 19 kHz pilot detector
- `classifiers.py`, `signal_id.py` - Signal classification with confidence; identifying the tuned signal and how to listen to it (Signal tab > Identify). The FM deviation it suggests must hold the station's swing, or Apply Mode leads straight to "FM audio is clipping" (shown when more than 5 % of the audio is beyond the deviation), and be no wider, or the station is heard quieter (`listening_for` keeps the band plan's deviation where that holds the swing and the peak is at least a quarter of it): `_swing_to_hold` takes the larger of the peak (`freq_peak_hz`, the swing exceeded 0.5 % of the time) and, with 10 % to spare, the swing exceeded 5 % of the time (`freq_p95_hz`). Voice reaches its peak only now and then, so it is held at its peak (a 2 m station limited at ±5 kHz gets 5 kHz); a tone stays near its peak and keeps the margin. How far a spectrum bin stands above the noise is judged against the noise's measured spread (`_spread` in `_spectrum_features`), not the spread independent averages would have: neighbouring bins of a windowed spectrum aren't independent and the demo's noise is rougher than white, so otherwise the noise's ripple counts as signal all over a wide capture (Identify in broadcast FM looks 800 kHz wide) and joins a call to a station hundreds of kHz away
- `signal_list.py`, `bursts.py` - Signals in a spectrum or a recording; bursts and rtl_433-style pulse analysis
- `afc.py` - Signal tracking (Radio > Track Signal (AFC))
- `sweep.py` - Sweep planning and measurement (Scanner tab, `sdr-scan scan`); `scanner.py` is the older library scanner
- `recording.py` - I/Q recording/playback (raw cu8/cs8/cs16/cf32/cf64, WAV, SigMF), `.sigmf-meta` sidecars, format guessing

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
  - `_ReceiverChain` (channel filter, demodulators, 48 kHz audio; the second receiver reuses it), `_SampleStore` (the I/Q recording buffer, a temporary file), `InfoPanel` (Info tab)
- `control_panel.py` - Right column, listening first: Frequency, Demodulation, Audio, Presets, Receiver and Recording groups (its License Profile group moves to Ham > Ham ID); `SAVE_FORMATS`, the one table of recording formats (Recording group, File > Save Recording)
- `receiver_bar.py` - Second toolbar row (MODE, FILTER, STEP) and the FREQ readout's digit tuning
- `level_meter.py` - Toolbar LEVEL meter with the draggable squelch threshold
- `tuning.py` - Steps, typed-frequency parsing, click-to-tune snapping, back/forward history
- `audio_chain.py` - Channel level, noise floor, squelch, audio AGC, noise reduction, CTCSS tone squelch, DC/I-Q clean-up
- `audio_sink.py` - Audio output on the device chosen in Preferences (a no-op without QtMultimedia)
- `rds_station.py` - Background RDS of the tuned FM station (the name next to FREQ)
- `tool_tabs.py`, `status_history.py` - Maximize Tool Tabs; the status bar's last 20 messages

Plots:
- `spectrum_widget.py` - Spectrum: click-to-tune, zoom, Auto/fixed range, peak/min hold, FFT size and window, bookmark labels, right-click menu
- `waterfall_widget.py` - Waterfall: rows keep the frequency they were received on, history length, measurements, COLORS scale
- `plot_view.py` - The zoom (x1-x16) both plots share

Tool tabs (Decoder, Bookmarks, Ham, Scanner, Signal, Info):
- `decoder_panel.py`, `decoder_views.py` - Decoder tab: messages, detail card, stats, Auto, Aircraft view (ADS-B), Text view (Morse). From window code, untick Decode with `pause_decoding()` (as tuning away from a preset does), not `_enabled_check.setChecked(False)`, which counts as the user's own pause and carries over to the next protocol chosen; `paused_by_window()` tells such a pause from the user's (the decoder chip hides for it), and Close Recording back on a preset's channel turns on again the preset's decoder the recording turned off (`_resume_preset_decoder`). `DecoderChip` is the status bar's decoder chip (the scan chip is one too): short of room, it is shortened to fit, keeping at least its first part ("CW (Morse) · …"), with the whole text in its tooltip; `text()` stays whole
- `bookmarks_panel.py`, `bookmark_editor.py` - Bookmarks tab: groups, filter, Edit Bookmark, CHIRP CSV
- `scanner_panel.py` - Scanner tab, and the status-bar chip of a running scan (Stop Scan, Skip, Hold, Show Scanner); `scan_engine.py` its engine, fed by the display loop; `scanner_host.py` the window as its host; `scanner_dialog.py` scan band presets and table helpers (no dialog any more)
- `signal_panel.py` - Signal tab (Identify, Signals, Scope, Bursts); `signal_tools.py` its window side, Track Signal (AFC) and Find Signals in Recording

Devices:
- `device_dialog.py` - Device > Connect... dialog and `MockDevice`, the demo device (its docstring describes the simulated stations, `_DEMO_SIGNALS` lists them)
- `device_settings.py` - Device kinds, gain models, sample-rate tables, front end, per-kind memory
- `device_controls.py` - Keeps the Receiver group in step with the device: gain, live sample rate, ppm, direct sampling, bias tee

Recordings and files:
- `playback.py` - File > Open Recording: `FilePlaybackDevice`, the playback strip, `PlaybackController`
- `iq_files.py` - What a file holds; Import Options (headerless files), Recording Info and decode dialogs; `format_size`, how sizes are shown (decimal kB, MB, GB; Tools > Preferences has its own `format_bytes`, also decimal)
- `file_jobs.py` - `run_job` (worker thread, progress, Cancel) and the load, write and decode jobs
- `file_dialogs.py` - File dialogs that remember a folder per kind of file; the recordings folder
- `audio_recording.py` - Radio > Record Audio (48 kHz WAV of what you hear)

Transmit and second receiver:
- `transmitter.py` - Transmit engine: `Transmitter`, `TxJob`, `TxPower`, `AutoIdPolicy`, legality checks
- `tx_station.py` - `StationTx`, the window's transmit side (Ham ID panel, ON AIR, confirmation, automatic ID)
- `transmit_text.py` - Tools > Transmit Text: CW, RTTY, PSK31, ASCII FSK with Preview, Save, Transmit
- `second_receiver.py` - Device > Second Receiver... window (`SecondReceiverHost`, `ReceiverDsp`)

Presets, help and settings:
- `presets.py` - The one preset table: Radio > Band Presets, the Presets group, the welcome screen, the Scanner's bands
- `help_dialog.py` - F1 Keyboard Shortcuts: `SHORTCUTS` (the window's keys, checked against its real bindings), `PLOT_KEYS`, `PANEL_KEYS` (keys a list, field or window handles itself) and `MOUSE_ACTIONS`
- `first_run_wizard.py`, `version_info.py` - Welcome and Quick Start; About, Copy Version Info, Report a Problem
- `error_log_dialog.py`, `preferences_dialog.py` - Tools > Error History; Tools > Preferences
- `settings_store.py` - `GuiSettings`, a typed QSettings wrapper whose getters give the default for a damaged value (see Important Notes); `usable_bookmarks`
- `themes.py` - Dark and light palettes, stylesheet roles and tones

## Testing

Tests use pytest (about 130 modules): DSP, drivers against fake backends, the CLI, and GUI tests that build the real main window. The suite is large; run the modules for the code you change.

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
- **Window tests** mostly build on `_WindowTestCase` (`tests/test_gui_chrome.py`): a real `SDRMainWindow` with in-memory settings (`_FakeSettings` in place of `GuiSettings`) and its timers stopped, so a test runs frames by calling `_update_display()`; `demo = True` starts the demo device. Code that creates `GuiSettings()` itself uses the real settings file (`~/.config/SDR Module Team/SDR Module.conf` on Linux); set `XDG_CONFIG_HOME` to a scratch folder to keep a run away from your own.
- **`tests/test_final_*.py`**: the last fixes, found by using the finished app end to end, a module per area (`decode`, `ham`, `listen`, `perf`, `plots`, `record`, `scan`, `window`); `test_final_scan_check.py` holds what the check of the scan fixes found. What two areas' fixes do together, found on the merged build, is in `test_final_integration.py`, `test_final_integration_ham.py`, `test_final_integration_scan.py`, `test_final_integration_window.py` and `tests/test_integration_decode.py`; the listen, perf, plots and record areas added theirs to their own modules, and the last cross-area fixes are in `test_final_polish.py` and `test_final_finish.py`. Each regression test failed before its fix (a few guard tests pin what must not change); run the area's modules when you change its code.
- **Timing-sensitive tests are made deterministic**, not given wider margins (they must pass on a busy computer):
  - run the display loop on a fake clock: `_FakeClock` (`tests/test_gui_round3_main.py`) patched in as the module's `time` (`mock.patch.object(main_window, "time", clock)`; `second_receiver` too) and advanced between `_update_display()` calls, as the playback and second-receiver Listen tests do. For a computer that can't keep up, advance it inside the frame by what the work would cost: `_FakeDemo` (same module) costs fake time per sample, and `TestPlaybackWhileOverloaded` (`tests/test_final_perf.py`) wraps `_process_block` to cost `load` times the time its samples cover at 1x, and a recording's `read_samples` to wait for the disk; `_Pacer` can also be driven on its own with made-up times (`TestSlowRecordingRidesOutHiccups`)
  - read every demo sample: patch `device_dialog._MAX_LAG_S` to `1e9`, or the demo device skips ahead to the wall clock on a slow run and its timed signals move (`_SignalCase` in `tests/test_gui_signal_tab.py`); fix a frame's size by replacing `win._pacer.demo_request`, and seed the demo's noise with `win._device._rng = np.random.default_rng(n)`; run Identify's jobs as they are handed over (`_RunAtOnce` in `tests/test_final_plots.py`, in place of `SignalTools._pool`), so which takes are judged doesn't depend on its worker thread
  - count work instead of timing it (`tests/test_final_perf.py` counts the outputs a filter computes and the samples the plots analyze); bring about the state a test needs (a progress dialog shown, a transfer in flight queued, a receiver silent for 3 s during a scan: move the scan engine's `last_block_at` back) instead of sleeping; stop a panel's own refresh timer when the test plants a reading, or call the timer's slot yourself (`ScannerPanel.refresh()`)
- **Damaged settings**: `_IniSettings` (`tests/test_final_window.py`) runs the real `GuiSettings` on an INI file in a temporary folder, written as a damaged settings file would be. A damaged value is logged once: per key for a `GuiSettings` instance, but once per process for a bad recordings folder (`file_dialogs._BAD_FOLDERS_LOGGED`), so a test that asserts that warning patches the set empty.
- **Fake hardware backends** (no hardware or driver packages needed):
  - `tests/test_rtlsdr_driver.py` - `FakeRtlSdr`, installed as the `rtlsdr` module; it has only pyrtlsdr's real methods, so a wrong driver call fails. `tests/test_gui_device_settings.py` has one with tuner gain tables and front-end calls for window tests.
  - `tests/test_hackrf_driver.py` - `FakeHackRFDevice`, installed as `python_hackrf`; answers only to real python_hackrf method names.
  - `tests/hackrf_fakes.py` - `ThreadedFakeHackRF` (its TX callback runs in its own thread like libhackrf's USB thread; optionally paced or stalled), `open_hackrf(monkeypatch, rate)`, and helpers that decode what went on air (`analyse_cw`, `analyse_mcw`); used by the transmit and device tests.
- **Signals**: `MockDevice` (the demo device) for receive tests, `PolledDevice(MockDevice(), name)` for dual-SDR tests, `tests/signal_synth.py` (synthetic signals, `add_noise`), `tests/text_demod.py` (independent demodulators for the text modes).
- **Demo stations** (MHz) to try features on: 100.1 FM with RDS, 144.390 APRS, 152.240 POCSAG, 931.9375 FLEX, 131.550 ACARS (AM), 1090 ADS-B, 145.800 SSTV (every 75 s), 145.290 repeater with a 100.0 Hz tone, 99.820 CW beacon, 433.920 OOK remote, 438.500 drifting handheld; AM broadcast on the radio tuner's six AM presets (660-1260 kHz). The `MockDevice` docstring describes them; `_DEMO_SIGNALS` in `gui/device_dialog.py` lists every simulated signal.

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
| `WIDEBAND_SCAN` | Coordinated spectrum scanning | Not implemented: `set_mode` raises `NotImplementedError` | No; scan with one device (Scanner tab, `sdr-scan scan --device ...`) |
| `RELAY` | Receive-and-retransmit | Not implemented: `set_mode` raises `NotImplementedError` | No |

`DualSDRController.use_devices(rtlsdr, hackrf)` takes devices that are already open (two `PolledDevice(MockDevice(), ...)` try dual receive without hardware). The GUI's second receiver doesn't use the controller: its `ReceiverDsp` runs its own `_ReceiverChain` (the main window's channel filter and demodulators) on its own device's blocks, and the transmissions it monitors come from the main window's transmitter.

## Important Notes

### The display loop
- A 33 ms timer calls `SDRMainWindow._update_display`: `_read_block` takes what arrived since the last frame, and `_process_block` hands that one block to everything that works on the signal (plots, level and squelch, receiver chain and audio, RDS, radio tuner, `SignalTools`, the I/Q recording, S-meter, live decoder, scanner)
- Hardware is polled without waiting (`read_samples(..., timeout=0.0)`); the demo device and a recording playing back are paced to real time by `_Pacer` (a recording never runs ahead)
- `_Pacer.measured` times each frame against the signal time it covers (leaving out the time a recording's read waited for the disk). When the computer can't keep up (`_pacer.overloaded`), frames drop to one block and a full frame is tried once a second to recover; meanwhile the Audio button reads Paused (the status bar says why, at most every 30 s), a running live decoder says it misses part of the signal (not while a Scanner scan runs: `_sync_decoder_skipping`), the Bursts page stops logging and says why, and the waterfall draws the one-block frames' rows as bright as the others without letting them set how bright that is (`_plot_frame` passes `add_line(..., usual=not self._pacer.overloaded)`; the second receiver does too). A recording playing back loses nothing by waiting, so its frames don't drop to one block: they still take what real time owes at 1x (or at a slower speed chosen), at most 50 ms of it a frame (`_Pacer.paced_request`), so it plays at that pace where the computer can, else as fast as it manages (at 1x its audio pauses only below half of real time), and the decoder, which hears all of it, doesn't say it misses part (`_samples_skipped`); nor does the Bursts page pause (`_bursts_frame` pauses only for a device), and Paused's hint gives no advice to lower the sample rate, which a recording can't change (`_realtime_hint`). An I/Q recording still gets every sample (every queued hardware transfer, and the demo's backlog through `_read_demo`), and what a source skipped (`dropped_samples`) is reported as lost
- New per-frame work goes in `_process_block`, does nothing while its view is hidden or its feature is off, and stays small (measured on a 4-CPU computer, on noise and on the demo's stations, a live decoder takes about 0.3-2.5 ms of the 33 ms at 2.4 MS/s, ADS-B the most; a HackRF delivers up to 20 MS/s). Count time in samples where you can, as `scan_engine.py` does, so tests can drive it block by block
- At 20 MS/s a frame holds about 670,000 samples, so keep each part's cost bounded: filter down to the channel first and compute only the outputs kept (the receiver's `_FirDecimator` takes one dot product per kept output wherever that is cheaper than filtering every sample by FFT); reuse the receiver's channel rather than filtering the full-rate block again (the Decoder's RDS takes it through `RDSLiveDecoder.process_channel`); cap a long frame's work (the plots analyze at most `_PLOT_MAX_SAMPLES`, in evenly spread segments); go through a large block in cache-sized pieces (Remove DC spike and Balance I/Q). `tests/test_final_perf.py` pins these by counting the work
- Nothing else takes samples from the main window's device: the scanner, decoders and signal tools get the loop's blocks (the scanner's host only discards transfers queued before a retune, as `_retune_lo` does)
- Measurements don't depend on the plots' FFT size, window or zoom: LEVEL and the squelch use the channel's own power, the noise floor (Auto squelch, the S-meter's SNR) a fixed 2048-point spectrum (`_power_spectrum_dbfs`)

### Tuning: the VFO inside the span
- The LO (`_lo_hz`, the device's center frequency) is the middle of the plots; the tuned frequency (the VFO) can sit anywhere in the span. The receiver chain, decoders and S-meter work at `_vfo_offset_hz()`
- Tune through `set_frequency` / `_tune_to`, never the device's `set_frequency`. They move only the VFO while the channel fits the inner 80% of the span (`_USABLE_SPAN_FRACTION`) and retune the LO otherwise (always, with Radio > Keep Tuned Frequency Centered), to where `_lo_for` puts it: on the tuned frequency, except below half the sample rate, where it goes to half the sample rate so the span starts at 0 Hz (below about a tenth of the rate, only as high as keeps the channel in the usable span, so the span starts a little below 0 Hz). `_tune_to(recenter=True)` puts the LO exactly on the frequency (the Scanner's steps and the transmit path rely on it)
- A frequency out of the device's range, outside a recording that is playing (`_lock_lo`), asked for while transmitting, or outside the usable span while an I/Q recording has samples is refused, with the reason in the status bar and everything left on the frequency tuned (`_refuse_tuning`: red, but amber while transmitting, as every "not while transmitting" notice is). A take holds one center frequency, as it holds one sample rate and one device: while it has samples, tuning inside the span moves only the VFO (Keep Tuned Frequency Centered too), and Device > Connect..., Disconnect and Use Demo Device ask before they end it (`confirm_stop_recordings` in `device_controls.py`)
- After a hardware retune, `_retune_lo` drops the samples taken on the old frequency. Recordings are labeled with the LO (their center frequency), not the tuned frequency; a SigMF take's description names the tuned frequency and mode (`_take_details`)

### Never block the GUI thread
- Long jobs run in workers with progress and Cancel: `file_jobs.run_job(parent, title, label, work)` runs `work(report)` in a `QThread`, shows a progress dialog after 400 ms, holds all keyboard and mouse input except the dialog's own (Cancel) meanwhile (the spectrum keeps running) and returns a `JobResult`. Convert, save and decode recordings, Find Signals in Recording and Transmit Text > Save use it; a cancelled or failed write leaves no partial file. While a job runs, `file_jobs.busy_with()` gives its title, and the Decoder's and the audio's hints name that job, not the device or the sample rate, as what makes the display loop skip, also for as long after it as that skipping is still measured (`_busy_job_now`, `_JOB_SKIP_S`), then in the past tense ("Decode Recording kept this computer busy, ..."; `busy_with()` no longer names it). So title a job with its command's name ("Decode Recording", "Save Text Signal"): the hints read "Save Text Signal keeps this computer busy, so audio is paused meanwhile." (`TestEveryJobTitleReadsInTheHints` in `tests/test_final_finish.py` checks every `run_job` title)
- The transmitter streams from its own thread, polled by a timer; Identify and burst analysis run in a thread pool, and the bursts worker spends at most a quarter of the signal's time taking bursts apart (`ANALYSIS_SHARE`, timed in its own processor time; beyond it, a burst is listed without its analysis)
- Don't make the display loop wait for a thread that may not get a processor on a busy computer: the plots hand half of a large frame's FFTs to a helper thread, and the loop does that half itself when the helper hasn't started on it
- Widgets are touched only from the GUI thread

### Settings survive damaged values
- A settings file can hold anything (a hand edit, another version). `GuiSettings` getters give the default, logged once per key, for a value of the wrong type, a NaN or infinite float or an integer beyond 32 bits; saved bookmarks come back through `usable_bookmarks` (an entry that isn't a bookmark is skipped, one without a usable frequency is listed at 0 Hz)
- The getters can't know a key's range, its choices or the shape of JSON stored in it: check those where you read it and fall back to the default, or to the nearest valid value where that is harmless (the end of the tuning range; never the costliest FFT size or a squelch that never opens), and log it. The plots' saved display settings (JSON) show how: `apply_display_settings(settings, defaults=...)` reads each yes/no with `saved_flag` (never `bool()`, which reads "no" as yes) and each number with `saved_number`, and logs each damaged value with what it used instead (`log_damaged_setting` in `spectrum_widget.py`; an average or a history length the menus don't offer gives the nearest one). JSON of another shape than the one saved (a list or a number where an object belongs) is as damaged as JSON that can't be read: give the defaults and log it, as the plots' restores in both windows (`_saved_choices` in `second_receiver.py`) and the Ham ID, QRP and radio-preset restores do. Restore each value on its own (`restore()` in `_restore_audio_settings`): a damaged setting must never stop the others, the app from starting or a window from opening
- Safety settings fall back to the safe side: an unreadable `tx/confirm` asks before transmitting

### Both themes
- Every color comes from the `Palette` in `themes.py` (Catppuccin Mocha dark, Latte light). Tag widgets with `set_role` / `set_tone`; in `paintEvent` read `get_palette()` and drop cached pixmaps on `theme_notifier().theme_changed`. Never hard-code a color
- Check new UI in both themes (View > Theme), at the minimum window size, 1024x640, and at 1366x768 (a laptop screen); the Second Receiver window at its own minimum, 860x560. At 1024 px the status bar keeps 220 px for messages (`_MESSAGE_MIN_WIDTH`): a status-bar chip whose text can grow should give way rather than widen the window, as the decoder and scan chips (`DecoderChip`) do

### Transmit only through the transmitter
- Everything the GUI sends goes through `StationTx.transmit(TxJob)` -> `Transmitter`, the only GUI code that calls the HackRF's `start_tx`. Send ID Now and Radio > Send Station ID reach it through `StationTx.send_station_id`; Transmit Text and any other job through `SDRMainWindow._transmit_job(TxJob)`
- That path checks the device can transmit (the demo device and RTL-SDR refuse, with the reason), the license class, band segment and protected frequencies for the emission's own mode and occupied bandwidth (CW is allowed in every segment the class holds, FCC 97.305(a)), a callsign for amateur transmissions, and the power limits. It asks before each transmission until "Don't ask again" is ticked (`tx/confirm`; Ask before transmitting in Tools > Preferences turns the question back on), adds the automatic station IDs, shows ON AIR, pauses receiving (half duplex), puts the HackRF back afterwards and ends with the outcome in the status bar
- While on the air the transmit state wins: `_show_tx_state(True)` (called by `StationTx` when a transmission starts, and by `_refresh_state_ui` while `_tx_busy()`) turns Start and Radio > Start Receiving off and dims the plots with "Transmitting - receiving resumes when it ends", or "Transmitting - press Space to receive once it ends" when the receiver was stopped before (it stays stopped: `_tx_resumes_rx()`); `_refresh_state_ui` turns them back on afterwards. Update Start and the plots' state through `_refresh_state_ui`, so a transmission always shows
- A `TxJob`'s source can make its samples as they are read (`read(n)`, `total_samples`), so a long transmission never has to sit in memory
- Don't call `start_tx` / `write_samples` from GUI code, and don't loosen a check to make a test pass: use the fake HackRF backends

### Keys, menus and messages
- `_add_action` makes a menu action the only owner of its shortcut; `tests/test_gui_coherent_window.py` fails on a key bound twice, and `tests/test_final_window.py` on an access letter (`&`) used twice in one menu
- Keys that must leave a focused slider, list or text field alone (Space, Left/Right, `[` `]`, P, M, Home, Page Up/Down) are handled in `keyPressEvent`, not bound as shortcuts; a menu item shows such a key after a tab (`"&Start Receiving\tSpace"`). A window shortcut that a focused widget should keep is handed to it in `SDRMainWindow.event`, which accepts its `ShortcutOverride` (Ctrl+Up/Ctrl+Down: Volume Up/Down, but ten steps in a number field and a row up or down in a list)
- A menu item that changes a control panel setting follows its control: checked as it is, and disabled while it is (`follow()` in `_add_listening_menus`: Radio > Mode, Radio > Squelch, and Radio > Audio's Noise Reduction and Audio AGC)
- List every new key in `help_dialog.py`: a window key in `SHORTCUTS` (F1 checks the table against the real bindings), a key a focused plot handles in `PLOT_KEYS`, and a key a list, field or window handles itself in `PANEL_KEYS` with an id. `tests/test_gui_help_keys.py` presses each `PANEL_KEYS` row on its real widget in a `test_key_<id>` test and fails when a row has none
- A refusal says what and why ("2400.000 MHz is out of range: RTL-SDR tunes 0.5–1766 MHz."). Use `_show_status_message(text, tone, ms)` / `_show_status_error`. A long action ends with its outcome in the status bar, never a stale "Preparing...", and a message that stops being true is replaced or cleared (a refusal by the next good tune, "Recording armed..." once samples arrive)
- Tools > Error History lists problems only. It takes every WARNING (or worse) any logger writes, and the status bar logs what it shows in the "warning" tone or with `_show_status_error` unless you pass `log=False` (a refusal `_show_status_message` shows in "danger", such as a frequency out of range, is not logged): pass it when you logged already (a problem is listed once) and for a notice that is no problem, such as what to do first ("ID not sent: enter your callsign in Ham > Ham ID first."), what the app did as asked (a scan stopped by Space, Open Recording or a new sample rate; "Recording paused", "Recording armed...") or what a feature can't do in the mode chosen (Track Signal in SSB, I/Q or broadcast FM). A part that logs its own problems shows its messages with `log=False` (`WindowScannerHost.scan_message`: the scan logs a step that broke, and a receiver that returns no samples, whichever tab is shown). Warn only about what is in use: a paused decoder doesn't warn that the sample rate is too low for it. List the same problem once: a decoder that can't work at the rate is logged once for the device and once for a recording until the decoder or the rate changes (`_decoder_logged`), though the status bar, the Decoder line and its chip say it at every rebuild. Tests check that a notice stays out with `assertNoLogs(level="WARNING")`
- One action, one message: an action that sets the mode along the way (the band plan, a preset, a bookmark's or the Decoder's Tune to, the Signal tab's Apply Mode, going back when a recording closes) sets it quietly (`_quiet_mode_change`; presets: `_applying_preset`) and says once what it set; the mode change alone would first name the mode's last filter ("AM: 6 kHz"). A message that something said along the way would replace carries it instead: Open Recording in another band leads the band plan's line with "Playing fm8.sigmf-meta (98.9–101.3 MHz) · " (`_listen_by_band_plan(..., lead=)`, `playback.playing_line`), and a speed other than 1x says the audio's own hint at once and marks it said (`set_speed`). A closing message carries what its steps just said rather than replacing it (Close Recording: "Closed aprs2m.sigmf-data. Back to Demo Device (simulated signals). AX.25/APRS decoder off: ..."), and so does a message that would replace one just said: a tune that turned a preset's decoder off leads its own line with "ACARS decoder off · " (`_keeping_release_note`, once), and a scan stopped by a tune from another tab says "Scan stopped · " before the tune's own message (`_scan_frame`): one posted before the tune is done, back in the event loop or at the next frame (`_tune_said`), not one something else posts after it
- Show frequencies with `format_frequency` (`main_window.py`: kHz below 2 MHz, MHz with 3 to 6 decimals above; the Signal tab's `format_mhz` and the Scanner's `_say_freq` name them the same way, while a list's column in MHz stays in MHz) and sizes with `iq_files.format_size`; use American spelling (`TestTheWindowsWords` in `tests/test_final_window.py` checks every text the main window shows) and the controls' own names ("Filter", not "Bandwidth")

### General
- Hardware drivers are optional; graceful degradation if missing
- Closing the window releases and closes the device, discards the temporary recording file and stops the plots' FFT helper one step at a time (`quietly()` in `SDRMainWindow.closeEvent`): a step that fails is logged and the next still runs, so the device is closed. The parts' `shutdown()` calls before those steps are not wrapped: a part with background workers must stop them there without raising, and ignore frames after it (`SignalTools.shutdown`)
- Create a widget you hand to a dialog with the dialog as its parent (`QCheckBox("Don't ask again", box)` before `box.setCheckBox(...)`): PyQt's `QMessageBox.setCheckBox` leaves a parentless box to Python, which deletes it while the dialog still holds it, and a restyle (a theme change) then crashes
- mypy ignores errors in `sdr_module.gui.*` and `sdr_module.ham.gui.*`; the rest is type-checked in CI
- Use demo mode for development without physical SDR devices (`--demo`, or Device > Use Demo Device)
