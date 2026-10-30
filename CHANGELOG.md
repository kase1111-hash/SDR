# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Listening and tuning
- **Tuning inside the span** — clicks, the mouse wheel and the arrow keys move
  only the tuned channel; the receiver retunes near the span's edges, or on
  every tune with Radio → Keep Tuned Frequency Centered. The Info tab shows
  both Tuned and Center (LO).
- **Receiver bar** — a second toolbar row with MODE, FILTER and STEP;
  `Left`/`Right` (`Shift`: ten steps, `Ctrl`: 1 MHz) and the mouse wheel over
  the plots (`Shift`: ten steps) tune by STEP, and `[`/`]` (plots focused) or
  Radio → Tuning Step change it.
- The Frequency field (`Ctrl+L`) reads `433.92M`, `7074k`, `1.09 GHz` or a
  number in Hz and says why it can't use what you typed; `Esc` cancels.
- Scrolling over a digit of FREQ tunes that digit.
- Click-to-tune snaps to the signal's peak or center, and the hover readout
  shows where a click will land; `Alt`+click tunes exactly where you click.
- Radio → Tune Back / Tune Forward (`Alt+Left` / `Alt+Right`) and Recent
  Frequencies (the last 20, with their modes, kept between sessions).
- **Squelch Auto** (Demodulation group, on by default) keeps the threshold
  7 dB above the tuned channel's own noise; the squelch has an on/off box.
- **Level meter** — LEVEL shows the channel's power, its noise floor and a
  squelch threshold you can drag, with an OPEN / SQL / TONE / OFF / I/Q lamp
  (TONE: a signal without the chosen CTCSS tone, also with the audio off).
- The Filter choices and default widths follow the mode (USB/LSB 2.7 kHz, CW
  500 Hz, AM 10 kHz, narrow FM 25 kHz, broadcast FM 200 kHz), and each mode
  remembers yours; in SSB and CW the filter is the audio passband.
- CW pitch (400–1000 Hz) in the Demodulation group; the FM deviation choices
  say what each is for, and FM audio clipped by too small a deviation is
  reported.
- **Audio group** — Noise reduction (Off, Light, Strong), Audio AGC (Off, Fast,
  Medium, Slow), Tone squelch (the 50 CTCSS tones) and Tone heard.
- Receiver group: Remove DC spike, and Balance I/Q to remove the mirror
  images of strong signals (recordings stay raw).
- Radio → Audio: Mute (`Ctrl+M`, or `M` with the plots focused), Volume Up /
  Down (`Ctrl+Up` / `Ctrl+Down`) and Output Device...; the Audio button says
  Audio, Muted, No output or No audio.
- **RDS station names** next to FREQ and on the radio tuner while you listen
  to broadcast FM (RadioText in the tooltip), cleared when you tune away or
  change source; Radio → RDS Station Names turns them off.
- Radio → Choose Mode From Band Plan (on by default): a frequency typed in
  another band is heard in that band's usual mode, filter and FM deviation,
  never over a mode you chose.
- Radio → Track Signal (AFC) keeps a drifting narrow FM, AM or CW signal
  centered, shown as "AFC +1.2 kHz" in the status bar.

### Added — Spectrum and waterfall
- **Color-scale legend with Auto levels** — the waterfall header's COLORS
  bar shows which color each dBFS level gets. Auto (the app default) keeps
  the floor just under the noise floor and the ceiling over the strongest
  signal, so weak signals stand out instead of sitting in the bottom of a
  fixed 0 to −100 dBFS scale; drag the bar's ends (or scroll, or use the
  arrow keys) for manual levels. Level changes re-map colors in about
  1.5 ms instead of re-rendering the history (about 35 ms before).
- **History length** — Live, 1 min (the default), 5 min, 15 min or 1 h;
  slower rows keep the strongest level of their interval, so short bursts
  stay visible.
- **Pause** (`P`) to study the display while the receiver keeps running.
- **Measure** — drag a box for bandwidth, duration, center, peak and SNR;
  it follows its rows as they scroll. Clicking still tunes (now on release).
- **Right-click menu** — tune, bookmark or copy the frequency under the
  cursor, measurement actions, history, levels, colors, clear, and Save
  Image (a PNG with time and frequency axes).
- The hover readout adds SNR over the noise floor and a row crosshair; the
  waterfall labels its own frequency axis when the spectrum is hidden; its
  colors, history and levels are saved between sessions.
- Both plots use every sample of each frame (the spectrum shows the mean
  power, the waterfall the peak per bin), so short bursts show.
- **Zoom ×2–×16** — View → Zoom: Zoom In (`Ctrl+=`), Zoom Out (`Ctrl+-`),
  Full Span (`Ctrl+0`); `Ctrl`+wheel zooms both plots around the pointer.
- The spectrum's ⋯ button and right-click menu add FFT Size (1024–32768,
  with its RBW), Window (Hann, Blackman-Harris, Flat-top), Range (Auto,
  60–120 dB, Top at 0 dBFS), Min Hold and Bookmark Labels.
- Drag, scroll or double-click the spectrum's dB labels to move or step the
  scale, or go back to Auto.
- Right-click on the spectrum: Tune to, Bookmark, Copy Frequency, Zoom In
  Here and Identify Signal Here.
- Bookmarks in the span are labelled along the top of the spectrum; click a
  label to tune it.
- Waterfall rows keep the frequency they were received on, so readouts and
  measurements on older rows stay right after a retune.
- After Stop, Disconnect or a paused recording, or while transmitting, the
  plots dim and say so ("Stopped - press Space").
- A long waterfall history fills at the live rate first (about 17 s) instead
  of starting with a blank minute.
- The spectrum's display settings are remembered, and `P` works with either
  plot focused.

### Added — Decoding
- **Live decoding of POCSAG, FLEX, AX.25/APRS and ACARS** in the Decoder tab
  (only ADS-B decoded live before), each picking its own channel at the tuned
  frequency; messages that fail their checks are tinted red. FLEX at 3200
  and 6400 bps is not supported yet.
- Decoder → RDS lists the station name, program type, RadioText, clock and
  alternative frequencies.
- Decoder → CW (Morse) shows Morse code as text with its speed (5–60 WPM)
  in a Text tab, from ±250 Hz around the tuned frequency (tune the signal in
  CW mode). A fast, deep swing of the noise level can still print stray
  characters.
- **Aircraft tab** for ADS-B: one row per aircraft (flight, altitude, speed,
  track, squawk, position); click one to show only its messages. It stays
  responsive with hundreds of aircraft.
- Protocol: Auto picks the decoder from the tuned frequency (Morse code in CW
  mode) and follows you.
- A status line says what the decoder is doing ("Not tuned to POCSAG:
  listening on 100.000 MHz"); the Stats tab adds LIVE DECODER counts.
- The Decoder tab counts unread messages ("Decoder (12)"), and a status-bar
  chip ("ADS-B · 37 msgs") opens it.
- Readable rows (APRS positions and messages, ADS-B callsign, speed and
  climb, ACARS flight) with a detail card; CSV export adds the frequency and
  sample rate.
- "Tune to …" also for POCSAG, FLEX and RDS, each in a fitting mode (ADS-B in
  None (I/Q), so the speaker no longer plays pulse noise).
- File → Decode Recording... (or Decode File... in the Decoder tab) runs any
  decoder over a whole file faster than real time; each message carries the
  file's frequency and its place in the file ("at 2.6 s").
- ADS-B needs at least 2 MS/s and works best at 2.4 MS/s or more; lower rates
  are marked "(too low for ADS-B)" in the rate lists.

### Added — Recordings and playback
- **File → Open Recording...** (`Ctrl+O`) plays SigMF, WAV and raw I/Q captures
  through the receiver with a playback strip: Play/Pause (`Space`), position,
  Loop, 0.5×–4× speed (audio at 1× only) and Close. Recordings centered above
  6 GHz (such as QO-100) play at their own frequency.
- Big-endian or real-valued SigMF, and one-channel, 24-bit or 32-bit WAV
  files, are refused with the reason.
- An Import Options dialog asks how to read headerless captures (rtl_sdr,
  hackrf_transfer, `.bin`, `.raw`, `.iq`): format, rate and frequency, guessed
  from the extension, the data and the file name.
- File → Convert Recording... (`Ctrl+Shift+O`) converts in the background
  with progress and Cancel, without retuning; Save also writes cu8 and cs8.
- File → Recording Info... shows a file's format, rate, frequency, span,
  length and origin.
- **Radio → Record Audio** (`Ctrl+Shift+A`) records what you hear as a 48 kHz
  WAV (silence while the squelch is closed or a scan moves between channels);
  the Recording group's Record choice picks I/Q or audio.
- The Recording group shows the take ("Unsaved · 00:04 · 77 MB @ 100.100
  MHz") with Save... and Discard, a Folder, and Save automatically; REC
  WAITING shows while the receiver is stopped mid-recording.
- Raw and WAV recordings get a SigMF metadata file, so their rate and
  frequency are known when they are opened again (an unknown one is asked
  for, never taken from the receiver); names include the rate, and a paused
  recording keeps the start time of each stretch.
- A recording that had to drop samples says how much ("· 33 ms lost").
- Saving as SigMF, cf32 or raw on the same drive is an instant move and
  other saves run in the background; a save that doesn't fit is refused
  before it starts, a failed one leaves no partial files, and on a nearly
  full drive Save automatically keeps the take and says why. WAV holds at
  most 4 GiB (about 7 minutes at 2.4 MS/s).
- Recording file dialogs remember their folder for each kind of file,
  starting in Documents/SDR Recordings.

### Added — Scanner and bookmarks
- **Scanner tab** (Tools → Frequency Scanner, `Ctrl+F`, or `Ctrl+8`) replaces
  the dialog: the plots, audio and decoders keep running while it scans.
- Scan modes: Sweep once, Sweep continuously, Stop on signal (with Resume,
  Skip and Hold) and Scan bookmarks (each in its own mode, filter and CTCSS
  tone).
- A sweep steps about 80% of the sample rate at a time (the FM band takes
  about a second on the demo device); Threshold Auto is 12 dB over the noise.
- Scan results (level, SNR, mode, last heard) with Listen, Bookmark, Bookmark
  All and CSV export; when a scan ends the receiver goes back to where you
  were tuned.
- A scan stops and says why when you tune by hand, stop the receiver,
  disconnect, change the sample rate, open a recording, record I/Q or
  transmit.
- Bookmarks get a Mode choice when added, automatic names ("2m Calling"), an
  Edit... dialog (mode, deviation, filter, CTCSS tone, group, shift, Skip,
  comment), groups and a filter box.
- Right-click a bookmark for Tune, Edit..., Rename... (`F2`), Copy Frequency,
  Move Up / Down, Scan Bookmarks from Here and Remove... (`Delete`).
- Tuning a bookmark restores its mode, deviation, filter and tone squelch,
  and `Ctrl+B` selects an existing bookmark instead of adding a duplicate.

### Added — Signal tab
- **Signal tab** (`Ctrl+9`, or Tools → Signal) with Identify, Signals, Scope
  and Bursts pages.
- Tools → Identify Signal (`Ctrl+I`) says what the tuned signal is, how sure
  it is and how to listen to it; Apply Mode sets that up.
- On/off keyed remotes and sensors are taken apart like rtl_433's pulse
  analyzer; Copy Pulse Analysis gives `rtl_433 -A` style text.
- Signals lists the signals in view; `Home`, `Page Up` and `Page Down` (plots
  focused) tune to the strongest and to the next one up or down.
- Scope shows the tuned channel against time or as a constellation with a
  BPSK/QPSK/8PSK/16QAM overlay and its EVM (samples, not symbols).
- Bursts logs the transmissions that come and go, with the bits in hex for
  keyed data and CSV export.
- File → Find Signals in Recording... lists a file's signals; double-click one
  to play the recording from there.

### Added — Ham tools
- **SSTV** (Ham → SSTV): Tune to ISS (145.800 MHz), Mode Auto (VIS) or a
  fixed mode, Start Now, a tone meter, partial images kept, and Switch to FM
  (VHF/UHF) or USB/LSB (HF) when the mode can't carry SSTV.
- SSTV Received Images come back after a restart (the newest 50), with Open,
  Show in Folder and Delete File... on right-click.
- **S-meter**: peak hold, a 30-second level history, Copy (RST or contest
  report), Calibrate (an offset or a known S9) saved per device type, and a
  report format that follows the demodulation.
- **QRP**: Calculated Output through an amplifier chain (presets, and stages
  that saturate), the legal limit at the tuned frequency, a Miles Per Watt
  QSO list and a Contest Exchange, all remembered.
- **Radio tuner** (`Ctrl+R`): presets remembered and renamed or cleared by
  right-click, seek arrows (`Shift`+click: one channel) that also work on a
  receiver tuning only part of the band, STEREO from the real pilot.

### Added — Transmitting
- **One transmit engine** for the HackRF One: transmissions stream in the
  background, are checked for the emission actually sent, and show an ON AIR
  badge (click it, or Radio → Stop Transmitting, to stop).
- Receiving pauses while on the air (the HackRF is half duplex); closing the
  window or disconnecting stops a transmission first, and a recording can't
  be opened meanwhile. The demo device and the RTL-SDR can't transmit.
- A first-use confirmation gives the frequency, mode and estimated output;
  Tools → Preferences turns it back on.
- Ham → Ham ID says what Send ID Now would do or why it can't; Sent as Auto,
  CW or FM tone; Radio → Send Station ID sends it from the menu.
- TX Power (HackRF): TX gain 0–47 dB and the +14 dB amp with the estimated
  output, locked while on the air; more than the QRP or legal limit is
  refused.
- **Automatic ID** now works: at the start, every interval (at a word break)
  and at the end of a transmission, sent the way the transmission is (CW or
  FM tone); an ID that is due anyway goes first; the ID's Morse is sent at
  20 WPM at most.
- FM tone IDs and FM text ramp up and down instead of clicking.
- **Tools → Transmit Text...** (`Ctrl+Shift+T`) sends CW, RTTY, PSK31 or ASCII
  FSK on a HackRF One, with macros; Preview (through the speakers) and Save
  (WAV or I/Q) need no device.
- 50.0–50.1 and 144.0–144.1 MHz are CW only, data below 29.7 MHz may be at
  most 2.8 kHz wide, and amateur transmissions need a callsign.

### Added — Devices and settings
- Gain that matches the device: an RTL-SDR's own gain steps, a HackRF One's
  LNA, VGA and RF amp (+14 dB, kept on across Stop/Start), and Tuner AGC
  only where the hardware has it; each device type remembers its gain.
- Receiver → Sample rate (also Device → Sample Rate, or a click on the rate
  in the status bar) changes the rate live, without reconnecting (not while
  a recording plays, while recording I/Q or while transmitting); each device
  type remembers its own.
- RTL-SDR Correction (ppm) and Direct sampling, and a Bias tee switch for an
  RTL-SDR or HackRF One, in the Receiver group and Device → Connect...; the
  Info tab shows the real gain and a Front end row.
- Device → Use Demo Device also replaces an open device; Find Connected
  Devices... lists what is plugged in, or how to install a missing driver.
- **Tools → Preferences...** (`Ctrl+,`): audio output device, default folders,
  recording temp folder and the transmit confirmation.
- **Device → Second Receiver...** (`Ctrl+Shift+N`) opens a second receiver on
  another device (or a second demo device; the windows never share one) with
  its own plots, tuning, squelch, gain, I/Q recording and audio mix.
- The second receiver's Monitor TX and Follow transmissions tune it to the
  main window's HackRF transmissions, and it keeps receiving while the HackRF
  transmits (the demo device can't hear a real transmission).
- The demo device adds decodable APRS (144.390 MHz), POCSAG (152.240 MHz),
  FLEX (931.9375 MHz) and ACARS (131.550 MHz), RDS on 100.1 MHz, an SSTV
  test card on 145.800 MHz, a 100.0 Hz CTCSS repeater on 145.290 MHz, AM
  stations on 660–1260 kHz, an EV1527-style remote on 433.920 MHz and a
  drifting handheld on 438.500 MHz; its CW beacon on 99.820 MHz decodes as
  "VVV DE SDR".

### Added — Window, menus and help
- The tool tabs fit a 1024×640 window; View → Maximize Tool Tabs (or a
  double-click on a tab) gives them the whole column; the first launch opens
  on Bookmarks.
- Radio → Band Presets lists the favourites, then every preset by category;
  ADS-B, APRS, ACARS, POCSAG and FLEX presets turn their decoder on, and ISS
  SSTV/Voice shows the SSTV tab.
- New POCSAG (152.240 MHz) and FLEX (931.9375 MHz) presets.
- Tools also lists Protocol Decoder, S-Meter, SSTV Decoder, Ham ID (Station
  ID) and QRP Calculator, each showing its tab (with its `Ctrl`+number key),
  and SSTV: Tune to ISS (145.800 MHz), which tunes there and opens the SSTV
  tab.
- The status bar's state, device and rate are clickable, a click on the
  message area lists the last 20 messages, and its warnings also go to Tools
  → Error History.
- The window title shows the frequency, mode, device and REC while
  recording.
- The Info tab's Band plan row and FREQ's tooltip show the band plan at the
  tuned frequency; the tooltip adds what your license allows there.
- `F1` lists every key, also those of lists, fields and playback, and finds a
  key as you type it.
- Help → Online Documentation, Report a Problem... and About SDR Module...
  (with Copy Version Info).
- File → Save Screenshot... saves PNG, JPEG or BMP under a dated name.

### Added — CLI
- `sdr-scan scan` sweeps a connected RTL-SDR or HackRF One until `Ctrl+C`, or
  once with `--single` (new `--device`, `--rate`, `--gain` and
  `--threshold auto`); with no SDR it exits with an error, and a `--rate` the
  device can't run is replaced by one it can, with a note.
- `sdr-scan decode` uses the live decoders (APRS, ACARS and RDS now decode),
  adds `cw`, `--offset` and `--show-invalid`.
- `sdr-scan encode` sends standard RTTY (45.45 Bd, 170 Hz), full PSK31 and
  Morse; new `--baud` and `--shift`; it reports characters it can't send and
  refuses a `--carrier` that doesn't fit `--sample-rate`.

### Changed — GUI overhaul
- **Layout** — resizable splitters whose sizes persist (View → Reset Layout
  restores them); the window fits a 1366×768 screen and the control panel
  scrolls instead of crushing its sections. The tool tabs are **Decoder**,
  **Bookmarks**, **Ham** (S-Meter, Ham ID, SSTV and QRP sub-tabs),
  **Scanner**, **Signal** and **Info** (live receiver status and quick tips).
  The spectrum and waterfall keep their controls in header strips and share
  one frequency axis.
- **Menus** — a new **Radio** menu holds Start/Stop Receiving, Record I/Q,
  Record Audio, Audio, **Band Presets**, Enter Frequency (`Ctrl+L`) and
  Bookmark Current Frequency (band presets, audio output and bookmarking were
  under Tools). Save Screenshot moved to File and the theme to View → Theme;
  File → Open Recording now plays a recording and Convert Recording converts
  one, and Device → Refresh Devices is Find Connected Devices..., beside a
  new Use Demo Device. View adds Panels (`Ctrl+1`–`Ctrl+9`), Focus Spectrum
  (`F6`) and Reset Layout; Help adds Welcome and Quick Start. The `F1` list
  is built from the live menus, grouped by menu and filterable.
- **Toolbar and status bar** — an Audio button and volume slider join the
  toolbar; clicking the FREQ readout opens the frequency entry, and LEVEL
  shows the tuned channel in dBFS (green above squelch). The status bar
  shows a NO DEVICE / STOPPED / RUNNING badge, device and sample rate,
  self-clearing messages colored by tone, and while recording a REC badge
  (elapsed time, ARMED or PAUSED) with size and free disk space.
- **Theming** — one stylesheet rendered from per-theme palette tokens
  replaces two hand-written ones that had drifted apart. Widgets use style
  roles and tones instead of hard-coded colors; the spectrum, waterfall
  axes, S-meter and radio tuner paint from the palette and switch live.
  Light-theme text reaches at least 4.6:1 contrast, light-theme buttons no
  longer look disabled, and check boxes, radio buttons and spin-box arrows
  use generated icons.
- **Focus and accessibility** — keyboard focus is always visible (a 2 px
  ring on buttons, check boxes and slider handles, an accent outline on
  tabs and lists, a focus frame on the plots). `F6`/`Esc` return the focus
  to the spectrum; `Ctrl+1`–`Ctrl+9` jump to a panel's first control.
  Dialogs have a consistent tab order and Enter/Esc behavior, controls
  carry accessible names, and the mouse wheel no longer changes a control
  panel setting it merely scrolls past.
- **Radio tuner** — redrawn in theme colors with a slide-rule dial, seek
  arrows and presets stored by right-click or press-and-hold. It tunes the
  main receiver in AM or FM with the matching deviation and bandwidth,
  follows the receiver, and shows when the receiver is outside the
  broadcast bands.
- **Defaults** — with nothing saved, the GUI starts on 100.1 MHz in FM with
  audio output on.
- Renamed: the control panel's Bandwidth is **Filter**, the Ham Radio tab is
  **Ham**, the waterfall's LEVELS scale is **COLORS**, and Import/Export
  Channels are **Import/Export Bookmarks (CHIRP CSV)**.
- The License Profile moved from the control panel to Ham → Ham ID, with the
  other transmit settings.
- Start with no device looks for SDR hardware first, and offers Demo Mode as
  the default when there is none.

### Changed — library
- New modules (in `sdr_module`) for decoding: `dsp.live_decoders` (streaming
  POCSAG, FLEX, AX.25/APRS, ACARS, ADS-B, RDS and CW decoders), `dsp.morse`,
  `dsp.rds`, `dsp.channelizer` and `dsp.protocol_encoders` (test signals).
- New modules for analysis: `dsp.frame_spectrum`, `dsp.signal_id`,
  `dsp.signal_list`, `dsp.bursts`, `dsp.sweep` and `dsp.stereo_pilot`.
- New modules: `dsp.nco` (the tuning mixer), `core.band_plan`,
  `core.band_info`, `protocols.text_modes` (CW, RTTY, PSK31 and ASCII text
  signals), `ham.sstv_encoder` and `devices.polled.PolledDevice`.
- `protocols.ProtocolDetector` works: it tries every live decoder on the
  samples (it had none registered, and registering one made `detect()`
  raise).
- `DualSDRController.set_mode()` raises `NotImplementedError` for
  `WIDEBAND_SCAN` and `RELAY`, which aren't implemented (it returned True);
  `supported_modes()` and `use_devices()` are new.
- `HackRFDevice.start_tx()` and `write_samples()` take `bandwidth_hz` and
  `mode` for the legality check; `last_tx_error` and `wait_tx_done()` are new.
- `HackRFDevice.write_samples()` sends the whole block (the last ~0.15 s was
  cut); `set_gain_mode(True)` returns False; `set_bias_tee()` is new.
- `RTLSDRDevice.set_freq_correction()` and a public `set_direct_sampling()`
  are new, and `get_tuner_gains()` returns the open tuner's own gain table
  (it returned a fixed list); `DeviceCapability.AGC` and `FREQ_CORRECTION`
  are new.
- `SDRDevice.discard_queued_samples()` drops samples queued at an old rate.
- Text encoders send standard RTTY (ITA2 with figures, 45.45 Bd and 170 Hz by
  default), the full PSK31 Varicode and every Morse character, and ASCII FSK
  no longer crashes on non-ASCII text.
- Text encoders raise `ValueError` for a baud rate, shift or speed out of
  range, and report what they skip (`skipped`, `unsupported_chars()`).
- `SSTVDecoder` is rewritten; call the new `finish()` after the last block of
  a file.
- `NoiseReduction.flush()` and `process_clip()` are new; the spectral methods
  stream with a delay of `latency_samples`, and `estimate_noise()` accepts
  I/Q.
- `SignalMeter.set_noise_floor()` stays until the new `clear_noise_floor()`;
  `CompactSignalMeter.update_samples()` takes `noise_dbfs`.
- `CWDemodulator.decode_morse()` gives the same text for any block size;
  `CWDemodulator.flush()` is new.
- `dsp.recording` writes and reads SigMF sidecars for raw and WAV files, adds
  `check_iq_samples()` and `guess_sample_format()`, and reads `.bin` as cu8.
- `dsp.protocols`: the POCSAG, FLEX, AX.25 and ACARS decoders stream;
  `RDSDecoder.decode_bits()` and `parse_aprs_info()` are new.
- `FrequencyPreset.decoder` names the decoder a preset turns on;
  `ham.callsign.MorseIdSource` streams a CW or MCW station ID.

### Added — GUI
- **Passband display** — a shaded band on the spectrum marks the channel
  being demodulated; LEVEL, squelch and the S-meter now measure that
  channel instead of the strongest signal anywhere in the span.
- **Demo audio** — the demo device streams in real time and its strongest
  station, 100.1 MHz, plays a looping tune in stereo FM, so audio works
  without hardware. It also simulates airband AM, 2 m, 70 cm and NOAA FM,
  ISM bursts, a CW beacon, and real ADS-B frames at 1090 MHz
  (flight KLM1023 with position, speed and squawk) for the ADS-B decoder.
  A computer that can't keep up gets a hint and paused audio instead of a
  stutter, and audio resumes by itself once it catches up.
- **Recording to disk** — I/Q recordings stream to a temporary file in the
  user cache folder instead of memory (about 19 MB/s at 2.4 MS/s used to
  accumulate in RAM). Save moves or converts it, temporary files are
  removed on discard and exit, and recording stops with a warning when the
  drive has less than 500 MB free.
- **Decoder and bookmarks** — the Decoder tab offers a one-click "Tune to …"
  for ADS-B, APRS and ACARS and says whether the receiver is tuned where
  the protocol is heard. Bookmarks save the demodulation mode (and FM
  deviation) and restore it when tuned.
- **Plots** — hover readouts (frequency and level) on both plots; the
  waterfall puts the newest line at the top, with a time axis in seconds
  and a Clear button.
- **Persistence** — the filter width, FM deviation, recording format, audio
  volume, splitter sizes, open tool tab, license class and Ham ID settings
  are now remembered too.

### Fixed — GUI behavior
- Shortcuts bound twice (so neither fired) have one owner, and Space and
  the arrow keys no longer take over a focused slider, combo box, list, tab
  bar, button or text field.
- Click-to-tune updates the control panel, toolbar readout and plot axes
  together; AGC can be switched off again; the saved frequency and gain
  survive a launch; the Recording panel's Pause and format choice work; an
  unsaved recording is offered for saving before it would be lost.
- The SSTV decoder gets continuous FM audio whether or not the speaker is
  on; Ham ID transmits only with a HackRF One connected; reconnecting the
  open device changes its sample rate; hot-plug polling stays quiet without
  drivers; dialogs are freed after use instead of piling up as hidden
  windows; audio output says when no output device can be opened.
- Broadcast FM gets 75 µs de-emphasis and the 19 kHz stereo pilot is
  filtered out of the audio; the receiver resamples to exactly 48 kHz and
  needs about a quarter of the CPU it did at 8 and 20 MS/s. Every queued
  HackRF transfer is processed each frame, so high sample rates no longer
  fall behind and mute audio (a slow computer falls back to fewer samples
  per frame and keeps the window responsive).
- The channel width no longer drives the HackRF's analog baseband filter
  (any change clamped it to 1.75 MHz and dimmed most of the span); the
  filter follows 0.75 × the sample rate.
- Tuning a bookmark or the Decoder's "Tune to …" picks a channel width that
  fits the mode (25 kHz for 2 m FM, 10 kHz for AM, 2.7 kHz for SSB) instead
  of keeping 200 kHz.
- Hiding the focused plot moves the focus to the other plot instead of the
  Frequency field (where Space typed).
- The default layout never cuts a control panel group, also at 1024×640.
- The bookmark file dialogs start in Documents and Save Screenshot in the
  screenshots folder (then each in the folder last used), not in the folder
  the app was started from.
- Help → Welcome and Quick Start no longer says "Mode set to FM" before an
  I/Q band, or offers Demo Mode while it is running.

### Fixed — Listening and tuning
- The squelch no longer chatters on fading signals or between Morse
  characters, and fades the audio in and out instead of clicking.
- LEVEL no longer rises with the filter width on an empty channel.
- FM a few kHz off-tune no longer comes out offset and clipped; SSB keeps a
  carrier and the DC spike out of the audio.
- Changing the filter, mode, FM deviation or CW pitch no longer clicks.
- Audio no longer stutters after a slow frame and comes back by itself after
  a busy moment, also at the highest sample rates.
- An RTL-SDR at 1.0 MS/s plays without regular audio dropouts.
- A frequency the device can't tune is refused with the reason ("2400.000
  MHz is out of range: RTL-SDR tunes 0.5–1766 MHz.") instead of being shown
  as tuned.
- A preset whose frequency can't be tuned (outside a playing recording, or
  while transmitting) changes nothing else.
- The Demo Device chosen in Device → Connect... runs in real time, like Use
  Demo Device.
- Balance I/Q no longer fails on a recording that starts with silence.

### Fixed — Spectrum and waterfall
- After a hardware retune, the first waterfall row and the peak hold no
  longer show the old frequency's stations.
- The waterfall's SNR readouts no longer read 6–8 dB low.
- Waterfall → Save Image after Stop or Pause saves a clean picture, without
  the dimming or badge.
- The hover readout no longer covers a measurement's readout, and Disconnect
  clears the peak hold.

### Fixed — Decoding, drivers and band plan (found through the GUI)
- **ADS-B** accepts only CRC-valid DF11/17/18 frames, and parity formats
  (DF0/4/5/16/20/21) only from aircraft already seen, so receiver noise no
  longer produces a stream of phantom aircraft. Squawk codes from DF5/21
  are decoded and shown.
- **RTL-SDR** `set_bandwidth()` sets the tuner's IF filter and never
  changes the sample rate (the GUI's channel width used to become the
  sample rate).
- **License privileges**: higher classes inherit lower-class segments;
  General and Amateur Extra were refused the 10 m 28.3–28.5 MHz phone
  segment open to Technicians.
- The FM Broadcast preset is 100.1 MHz, the ADS-B preset turns on the ADS-B
  decoder, and the sample-rate and filter tooltips distinguish the span
  shown from the channel demodulated.
- Live decoders no longer lose bits between blocks, and POCSAG no longer
  locks onto the wrong baud rate, prints pages from noise or misreads numeric
  pages.
- ADS-B decodes replies that start between two samples (about half of the
  strong ones were lost at 2.4 MS/s) and needs less than half the CPU it did
  at 8 and 20 MS/s.
- ADS-B says when the sample rate is below 2 MS/s instead of decoding
  nothing, and no longer shows −64 ft/min for aircraft that send no climb
  rate.
- The HackRF One's gain reaches its full range (the slider stopped at 50 dB),
  and its AGC box no longer shows "Auto" for an AGC it doesn't have.
- Moving an RTL-SDR's gain slider no longer fills Error History with "gain
  clamped" warnings.
- An RTL-SDR applies your gain again after listening below 24 MHz (direct
  sampling) instead of running its own AGC.
- The demo device refuses a sample rate it can't run, which used to be saved
  and freeze every later launch.

### Fixed — Recordings
- I/Q recordings no longer lose most of the signal when the computer falls
  behind at high sample rates.
- An rtl_sdr `.bin` file is no longer read as 32-bit floats, and an empty
  file is refused.
- cu8, cs8 and cs16 saves round instead of truncating (no false spur at the
  center of a cu8 file).
- A converted capture whose rate or frequency is unknown can be saved (it
  failed with "float division by zero"), and keeps its own time and
  description.
- A recording on a drive that filled up can be saved to another drive.
- A paused recording stays shown as paused when Record is switched between
  I/Q and Audio.
- Long recordings use about 1 MB of memory an hour (14 MB before).

### Fixed — Ham tools and transmitting
- The SSTV decoder never recognised a standard transmission; rewritten, it
  decodes the Robot, Martin, Scottie, PD and Wraase modes, also from USB or
  LSB on HF.
- The S-meter read "S5, RST 55, SNR 23 dB" on an empty channel; it now
  measures the channel's real SNR.
- Send ID Now no longer freezes the window, cuts a long ID short or sends too
  fast at other sample rates.
- The transmit legality check uses the emission's own width and mode instead
  of the receiver's 1.75 MHz filter, which refused every HF band.
- The HackRF's bias tee is always off while transmitting, and a failed start
  no longer leaves the radio transmitting.
- The QRP chain starts from the SDR alone (1 mW) instead of a made-up 1 W,
  and the QRP tab gives a General licensee in USB on the 60 m channels their
  100 W PEP limit instead of "Not permitted ... SSB isn't allowed".

### Fixed — Scanner, bookmarks and CHIRP files
- CHIRP channels keep an 88.5 Hz TSQL tone when imported, and editing one no
  longer rewrites its CHIRP Mode (D-STAR and DMR became FM).
- Spreadsheet CSV rows whose Mode is FM are heard in FM; rows whose
  frequency or offset is "nan" or "inf" are skipped (they imported as
  "nan000 MHz"), and "inf" in another number column no longer fails the
  whole import.

### Removed — GUI
- The radio tuner's volume, tone, balance and power/mute controls, which
  never affected the audio: it plays through the main window's Audio
  button and volume slider. `set_volume()` / `set_muted()` remain for
  `process_samples()`.
- The status bar's buffer-fill gauge (the Info tab shows the buffer).
- The Frequency Scanner dialog, which froze the plots and audio while open
  (Tools → Frequency Scanner now shows the Scanner tab).

### Fixed — Hardware drivers now work against real libraries
- **RTL-SDR enumeration** called `RtlSdr.get_device_count()` /
  `get_device_serial()`, methods pyrtlsdr has never had, so device scanning
  always returned nothing and no RTL-SDR was ever detected. Rewritten on the
  real API (`get_device_serial_addresses()`,
  `librtlsdr.rtlsdr_get_device_count()`). `open()` no longer leaks a USB
  handle when a default setter fails, and a dead RX thread now clears the
  streaming flag and exposes the error instead of hanging silently.
- **HackRF driver** imported a `hackrf` package that does not exist on PyPI,
  so `pip install ".[hackrf]"` failed and no HackRF could be opened.
  Reimplemented on **python_hackrf** with correct RX/TX callback handling.
- **HackRF transmit safety**: `write_samples()` was a stub that returned
  success without transmitting (the GUI reported "TX complete" while nothing
  was sent); it is now a real blocking one-shot transmission. `start_tx(None)`
  is refused instead of radiating an uninitialised buffer, TX samples are
  clipped before scaling, and the TX frequency lockout is enforced in every
  tuning path.

### Fixed — DSP correctness
- **BPSK demodulation** decided on the quadrature sign, giving ~50% bit-error
  rate on a clean signal; it now slices on the in-phase axis (0 BER clean).
- **SSB demodulation**: USB and LSB produced byte-identical output and
  neither rejected the opposite sideband; sideband selection is now done in
  the frequency domain so USB and LSB differ and reject the other sideband.
- **Spectral noise reduction** crashed on complex I/Q (used `rfft`) and
  returned fewer samples than it consumed; both spectral-subtraction and
  Wiener paths now use a complex-safe, length-preserving overlap-add.
- Noise reduction no longer amplifies block edges by 20–45 dB.
- The CTCSS tone squelch no longer opens for the adjacent tones; DCS, not
  implemented, is refused instead of acting as a carrier squelch.
- AGC log mode no longer runs the gain to its maximum, and the NOISE squelch
  opens on a signal.
- I/Q imbalance correction no longer makes the mirror image worse.
- Signal classification no longer calls filtered noise "OOK data" with full
  confidence or mixes up narrow FM, AM and BPSK, and a wrong guess is never
  "confident".
- Burst detection finds bursts from the first second instead of after 12 s
  of nothing and then false ones.
- AFC settles on carriers and voice FM and follows a steady drift.
- The constellation scales its points by their average level, so a clean
  QPSK signal reads about 6% EVM (it read 14%).

### Fixed — Robustness
- `SampleBuffer.read()/peek()` raise on a negative count or one exceeding
  capacity instead of blocking forever or corrupting the ring buffer.
- `SDRConfig.save()` writes atomically (temp file + rename), so an interrupted
  write can no longer truncate a good config; `get_default_config_path()` no
  longer creates directories as a side effect.
- A damaged value in the settings no longer stops the app or a window from
  opening: the default is used and the problem logged. An unreadable
  transmit-confirmation setting asks again.

### Changed — Packaging, CI and tooling
- CI runs the PyQt6 GUI suite headless on every job, adds the antenna-array
  tests, spans Python 3.10-3.14 plus Windows and macOS, makes mypy a blocking
  gate, adds a Bandit + pip-audit security job and a CodeQL workflow, and
  smoke-tests the built wheel. Release tags are checked against the package
  version and CHANGELOG.
- `requires-python` raised to `>=3.10` (PyQt6 already required it; 3.9 is EOL).
  Version is single-sourced from `sdr_module.__version__`; both packages ship
  `py.typed`. The `[hackrf]` extra points at `python_hackrf`, `[dsp]` adds
  SciPy and `[images]` adds Pillow; `[dev]` includes PyQt6 so contributors run
  the GUI tests. Development status is now Beta.
- The PyInstaller spec, Windows installer and build scripts were corrected:
  a working entry point (running `cli.py` directly broke its relative
  imports), the GUI bundled by default, a real installer GUID, `LICENSE.md`,
  and version/`python -m pip` fixes.

### Added — Saved channels (CHIRP CSV)
- **`sdr_module.core.chirp_csv`** — read and write memory channels in
  [CHIRP's](https://chirpmyradio.com/) generic CSV format. Exports use CHIRP's
  exact 21-column header and value formatting (MHz with six decimals,
  zero-padded DTCS codes), so a file written here opens directly in CHIRP and
  can be uploaded to a radio; a CHIRP file re-exported unchanged is
  byte-identical. Imports are forgiving: headers matched case-insensitively,
  any subset of the optional columns, unknown columns ignored, UTF-8 BOM
  stripped, blank rows skipped. Repeater duplex/offset, CTCSS/DTCS tones,
  mode, tuning step, skip flag, power, and comments all survive a round trip.
  Pure Python — no PyQt6 or NumPy needed.
- **GUI** — a CSV button on the Bookmarks (memory channels) tab with
  Import / Export Bookmarks (CHIRP CSV), plus the same two commands in the
  File menu. Importing offers
  replace or append; imported mode, tone, and repeater shift are shown in the
  channel list. Channels saved before this change still load.
- **CLI** — `sdr-scan channels export|import|list`, including
  `channels export --presets` to dump the built-in RX preset list as a CHIRP
  CSV without needing the GUI.
- A bookmark's mode, FM deviation and filter survive a round trip through
  CHIRP, in a short tag in the Comment column that the app hides.
- Files with semicolons or tabs between columns, or decimal commas, import;
  Append skips channels already in the list.
- Bookmarks → CSV → Export Built-in Presets (CHIRP CSV)... saves the app's
  presets for CHIRP or a handheld.

## [0.2.0] - 2026-07-04

First release published to PyPI.

### Added — CLI
- **`sdr-scan decode`** — decode a protocol (POCSAG, FLEX, AX.25/APRS, RDS,
  ADS-B, ACARS) from a recorded I/Q file, exposing the shipped decoders without
  writing Python.
- **`sdr-scan scan --input FILE`** — offline signal detection on a recorded I/Q
  file via FFT peak detection (previously `scan` was a hardware-only stub that
  performed no analysis). Supports `--format`, `--sample-rate`, `--center`,
  `--fft-size`, and `--max-samples`.

### Fixed — Developer experience
- Added a `SessionStart` hook (`.claude/hooks/session-start.sh`) that installs
  the project and the bundled `sdr-antenna-array` package with the correct
  interpreter, so Claude Code on the web sessions are ready with no manual
  steps.
- Documented the `python -m pip` install requirement in the README to avoid
  interpreter/pip mismatches.

### Added — GUI usability
- **AM/FM Radio Tuner widget** with vintage car radio styling (LED frequency
  readout, analog tuning dial, 6 presets per band, volume/tone/balance,
  seek up/down, AM/FM band selector, power/mute).
- **Frequency scanner dialog** — non-blocking sweep with progress bar and
  live hit table.
- **Bookmarks / memory channels panel** (add, rename, remove, double-click to
  tune). Bookmarks persist across launches.
- **Band presets menu** for common bands (FM Broadcast, NOAA Weather, 2 m,
  70 cm, Airband AM, ADS-B, ISM 433/915).
- **Click-to-tune** on both the spectrum analyzer and the waterfall display.
- **Keyboard shortcuts** — Space (start/stop), Ctrl+Shift+R (record),
  arrows ±10/100/1000 kHz with Shift/Ctrl modifiers, Ctrl+O/S/P/Q,
  Ctrl+T (theme), Ctrl+B (bookmark), Ctrl+E (error history),
  Ctrl+F (scanner), F1 (help).
- **Audio output** — simple in-GUI demodulation (FM/AM/SSB/CW) driving
  `QAudioSink`; squelch-gated; toggleable via Tools → Audio Output.
- **Light theme** option alongside the existing dark theme; toggle with
  Ctrl+T. Theme, window geometry, frequency, gain, squelch, AGC, demod
  mode, and bookmarks all persist between launches via `QSettings`.
- **First-run wizard** that offers demo mode and a starting band.
- **Help dialog** listing every keyboard shortcut (F1).
- **Error history viewer** — ring-buffer log handler retains the last 500
  warning/error records for review.
- **Device hot-plug detection** — background polling notifies when a new
  RTL-SDR or HackRF is plugged in.
- **Recording indicators** — live HH:MM:SS elapsed time, size in MB, and
  free-space on the target volume in the status bar.
- **Screenshot export** of the main window (Tools → Save Screenshot, Ctrl+P).
- **`File → Open / Save Recording`** menu items now actually load/save
  I/Q files (cf32, cs16, raw, WAV) via `dsp.recording.load_iq_file` /
  `save_iq_file`.
- **`Device → Refresh Devices`** now actually rescans hardware and reports
  the result.

### Added — other
- Python 3.13 support.
- Covariant return type on `ProtocolDecoder.decode` (`Sequence[DecodedMessage]`)
  so concrete decoders returning `List[SubMessage]` pass type-checking.

### Changed — dependency bumps (landed as combined PRs #65, #66)
- GitHub Actions: `codecov-action@v5→v6`, `attest-build-provenance@v2→v4`,
  `upload-artifact@v6→v7`, `download-artifact@v7→v8`,
  `softprops/action-gh-release@v2→v3`. Workflow files are now internally
  consistent.
- Python tooling: `pytest>=8.4.2`, `mypy>=1.19.1`, `ruff>=0.15.10`,
  `PyQt6>=6.10.2`.
- Build requirement: `setuptools>=61.0,<77` (temporary upper bound —
  setuptools 77+ emits `Dynamic: license-file` / `License-Expression`
  metadata that twine 6.2.0 rejects; remove the cap once a twine release
  recognises PEP 639 fields).

### Fixed
- **ADS-B decoder hung forever** on any input below 1 MHz sample rate
  (`samples_per_bit` truncated to 0, so the preamble scan never advanced).
  All protocol decoders now validate their sample rate up front and raise
  a descriptive `ValueError` instead of hanging or crashing with
  `range() arg 3 must not be zero`; the CLI reports these as friendly
  errors.
- **`sdr-scan encode --output out.wav`** now writes a real playable 16-bit
  stereo (I/Q) WAV instead of raw float bytes in a `.wav`-named file.
- **`--input` accepts `.sigmf-meta` paths**: previously the JSON metadata
  file was misread as raw samples; the SigMF data/meta pair is now
  resolved from either filename.
- **README install instructions** led with `pip install sdr-module`
  before the package was published; now lead with the working
  from-source install.
- **Release workflow**: the SBOM job no longer tries to attach SBOMs to a
  GitHub release with a read-only token (that step failed every prior
  release run; the release job attaches artifacts itself).
- **Dead GUI test revived**: `TestSignalMeterWidgetLogic` imported a
  widget from a module path that never existed and silently skipped
  forever; it now tests the real `SignalMeterPanel`.
- **`sdr-antenna-array`**: fixed a Python 3.9 crash-on-import
  (`type | None` annotation without `__future__` import), a `None`
  default on a dict dataclass field, unused imports, and stale
  type-ignore comments; added the missing README the packaging
  referenced; aligned license metadata with the root package so
  `twine check` passes.
- **Waterfall widget crashed on first frame under NumPy 2.x**: bitshift of
  `uint8` colormap components into a 32-bit ARGB word overflowed. Cast
  components to `int` before the shift.
- **Waterfall widget crashed on empty spectrum arrays** (NumPy 2.x made
  `np.interp` stricter): now records a neutral min-dB row instead.
- **Demo mode displayed nothing**: the `MockDevice` was created but never
  started, so `read_samples()` returned `None` and the spectrum never
  updated. Now starts on entry.
- FIR filter normalization for highpass, bandpass, and bandstop filters
  (was dividing by near-zero sum for non-lowpass types).
- POCSAG decoder dropping all but the first message per batch
  (`_process_batch` returned only `messages[0]`).
- PSK31 encoder phase inversion (was shifting phase on 1-bits instead of
  0-bits per the PSK31 BPSK standard).
- Blackman-Harris and Flat-top window function denominators (used `n/N`
  instead of the correct `n/(N-1)`).
- `save_iq_file` missing `cf64` format support that `load_iq_file` already
  supported.
- AX25Frame / APRSMessage / RDSData dataclasses: default mutable list
  fields now use `field(default_factory=list)` instead of `None`.
- `__exit__` signatures in `core/device_manager.py`, `core/dual_sdr.py`,
  and `devices/base.py` no longer use Python 3.10 `type | None` syntax
  (the project targets 3.9) and return `None` instead of `bool`.

### Documentation
- Full re-alignment between README / SPEC_SHEET and the shipping codebase.
- Historical audit reports (`AUDIT_REPORT`, `VIBE_CODE_AUDIT`,
  `EVALUATION_REPORT`, `AGENTIC_SECURITY_AUDIT`,
  `AGENTIC_SECURITY_AUDIT_V3`) consolidated into a single
  `AUDIT_HISTORY.md` with a status-per-finding table.
- Planning docs (`PLAN.md`, `REFOCUS_PLAN.md`) removed — the work they
  described is complete; git history remains the reference.

## [0.1.0] - 2024-01-01

### Added
- Initial release.
- Dual-SDR support for RTL-SDR and HackRF One.
- Real-time spectrum analyzer with FFT-based visualization.
- Waterfall display with time-frequency representation.
- Signal classification for analog and digital modulations.
- Protocol detection framework.
- Demodulators: AM, FM, SSB, CW, OOK, FSK.
- Filter bank with low-pass, high-pass, band-pass, and notch filters.
- Automatic Frequency Control (AFC).
- Frequency scanning capability.
- I/Q recording and playback (WAV, raw, SigMF formats).
- Text encoding (RTTY, Morse, ASCII, PSK31).
- SSTV decoder for satellite image reception.
- Ham radio callsign identification.
- S-unit signal meter with RST reporting.
- QRP (low power) operations support.
- PyQt6 GUI application.
- Command-line interface (`sdr-scan`).
- Comprehensive configuration system with presets.
- Tooltip system for RF/DSP terminology.

### Hardware Support
- RTL-SDR (RX only): 500 kHz – 1.7 GHz, 2.4 MHz bandwidth.
- HackRF One (TX/RX): 1 MHz – 6 GHz, 20 MHz bandwidth.
- Dual-SDR operation modes: `DUAL_RX`, `FULL_DUPLEX`, `TX_MONITOR`,
  `WIDEBAND_SCAN`, `RELAY`.

### Protocols
- ISM band devices (433/868/915 MHz).
- Amateur radio (AX.25, APRS).
- Aviation (ADS-B, ACARS).
- Paging (POCSAG, FLEX).
- Broadcast (RDS).

[Unreleased]: https://github.com/kase1111-hash/SDR/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kase1111-hash/SDR/compare/v0.2.0-Beta...v0.2.0
[0.1.0]: https://github.com/kase1111-hash/SDR/releases/tag/v0.1.0
