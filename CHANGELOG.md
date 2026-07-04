# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
