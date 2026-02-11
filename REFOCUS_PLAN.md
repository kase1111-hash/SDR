# SDR Module Refocus Plan

This plan restructures the SDR Module from a sprawling 36,000+ line project into a focused dual-SDR toolkit. It is organized into four sequential phases, each leaving the codebase in a shippable state.

---

## Guiding Principle

**Ship a 0.1.0 that does five things well, not a 0.1.0 that claims twenty.**

The refocused product identity:

> A Python-native dual-SDR framework for RTL-SDR and HackRF One with signal processing, protocol decoding, and real-time visualization.

Everything that doesn't directly serve this identity gets cut, extracted, or deferred.

---

## Phase 1: Correct Over-Promises (Documentation Honesty)

**Goal:** Align all public-facing documentation with what actually exists in the codebase.
**Risk:** Zero. No code changes. Only text edits.
**Estimated scope:** ~50 lines of edits across 2 files.

### 1.1 Remove unimplemented protocol claims

The following protocols are listed in documentation but have no implementation:
- P25, DMR, TETRA (Trunking)
- LoRa, Zigbee, Z-Wave (IoT)
- ISS SSTV, Meteor-M2 (Space -- SSTV decoder exists but not satellite-specific)

**Files to edit:**

| File | Change |
|------|--------|
| `README.md:211-220` | Remove P25, DMR, TETRA, LoRa, Zigbee, Z-Wave rows from Supported Protocols table. Keep ISM Band, Aviation, Paging, Amateur Radio, Broadcast rows (these have real implementations). |
| `README.md:15` | Change `(ADS-B, POCSAG, LoRa, APRS, etc.)` to `(ADS-B, POCSAG, APRS, RDS, FLEX, ACARS)` -- list only what's implemented. |
| `SPEC_SHEET.md` | Audit every protocol reference. Remove or tag as "Planned" any protocol without a decoder class in `dsp/protocols.py`. |

### 1.2 Add performance disclaimer

**File:** `README.md` (new section after Hardware Support)

Add a "Known Limitations" section:
- Pure Python DSP cannot process the full 20 MHz HackRF bandwidth in real-time
- Effective real-time bandwidth depends on FFT size, demodulator complexity, and host CPU
- For wideband capture, record I/Q to disk and process offline
- No GPU acceleration

### 1.3 Remove plugin system from README

**File:** `README.md:250-276`

Remove the entire "Plugin System" section including the code example. The plugin system has no real plugins and will be extracted in Phase 2.

### 1.4 Deliverable

A README and SPEC_SHEET that accurately represent the codebase. Every feature listed has working code behind it.

---

## Phase 2: Cut Dead Weight (Remove Non-Core Code)

**Goal:** Remove code that provides zero user value or belongs in a different project.
**Risk:** Low. All removed components are isolated (verified via dependency analysis).
**Estimated scope:** ~4,200 lines removed.

### 2.1 Delete the plugin system

The plugin system (`plugins/`) is 2,476 lines of framework code with zero real plugins. It adds architectural complexity for no user benefit.

**Files to delete:**

| File | Lines |
|------|-------|
| `src/sdr_module/plugins/base.py` | 595 |
| `src/sdr_module/plugins/manager.py` | 849 |
| `src/sdr_module/plugins/registry.py` | 386 |
| `src/sdr_module/plugins/__init__.py` | 58 |
| `examples/plugins/noise_filter/plugin.py` | 199 |
| `examples/plugins/noise_filter/plugin.json` | - |
| `examples/plugins/weather_sensor/__init__.py` | 9 |
| `examples/plugins/weather_sensor/plugin.py` | 380 |
| `examples/plugins/weather_sensor/plugin.json` | - |
| `tests/test_plugins.py` | 358 |

**Dependency cleanup required:**
- `src/sdr_module/__init__.py` (lines 51-62): Remove all Plugin-related exports (PluginManager, PluginRegistry, Plugin, PluginMetadata, etc.)
- Remove `examples/plugins/` directory entirely

**Impact:** No core code imports the plugin system. Only tests and examples use it.

### 2.2 Delete MX-K2 Morse keyer driver

A USB driver for a specific Morse key peripheral. Not SDR functionality.

**Files to delete:**

| File | Lines |
|------|-------|
| `src/sdr_module/devices/mxk2_keyer.py` | 801 |
| `tests/test_mxk2_keyer.py` | 457 |

**Dependency cleanup required:**
- `src/sdr_module/devices/__init__.py`: Remove MXK2Keyer export
- `src/sdr_module/core/device_manager.py`: Remove any MX-K2 device scanning logic (if present)

**Impact:** Only its own test file imports it.

### 2.3 Remove phantom protocol color schemes

These UI color entries reference protocols with no decoder:

**File:** `src/sdr_module/ui/waterfall.py` (lines 76-83)

Remove entries for: `p25`, `dmr`, `tetra`, `lora`, `zigbee`, `zwave`

Keep entries for protocols with real decoders (pocsag, ax25, rds, adsb, flex, acars).

### 2.4 Deliverable

A codebase ~4,200 lines lighter with no dead code paths. All remaining code is reachable and serves a purpose.

---

## Phase 3: Extract Separate Products (Decouple Non-Core Subsystems)

**Goal:** Move the antenna array and ham radio features out of the core package so the main project stays focused. Code is preserved, not deleted.
**Risk:** Medium. Ham radio features are woven into `gui/main_window.py` and require refactoring.
**Estimated scope:** ~12,000 lines moved/refactored.

### 3.1 Extract antenna array to separate package

The antenna array subsystem (7,365 lines) is completely isolated -- no core, DSP, or GUI code imports it. Extraction is clean.

**Action:** Move `src/sdr_module/antenna_array/` to a new top-level package directory `packages/sdr-antenna-array/` (or a separate repository `sdr-antenna-array`).

**Files to move:**

| Source | Destination |
|--------|-------------|
| `src/sdr_module/antenna_array/*` (8 files, 5,131 lines) | `packages/sdr-antenna-array/src/sdr_antenna_array/` |
| `tests/test_antenna_array.py` (698 lines) | `packages/sdr-antenna-array/tests/` |
| `tests/test_antenna_array_phase2.py` (672 lines) | `packages/sdr-antenna-array/tests/` |
| `tests/test_antenna_array_phase3.py` (664 lines) | `packages/sdr-antenna-array/tests/` |

**Dependency cleanup:**
- `src/sdr_module/__init__.py` (lines 41-48): Remove AntennaArrayController, ArrayConfig, ArrayGeometry exports
- Create `packages/sdr-antenna-array/pyproject.toml` with `sdr-module` as optional dependency

### 3.2 Extract ham radio features to optional extras

Ham radio features (6,485 lines) are more deeply integrated -- `gui/main_window.py` directly creates panels for callsign, SSTV, signal meter, QRP, and radio tuner.

**Strategy:** Make ham radio panels **optional** in the GUI rather than moving them to a separate repo. This preserves the code but decouples it from the core product.

**Step 1: Refactor `gui/main_window.py` to support optional panels**

Current state: `main_window.py` unconditionally imports and creates all panels:
```python
# Lines ~182-195: Hard-coded panel creation
self._callsign_panel = CallsignPanel()
self._sstv_panel = SSTVPanel()
self._signal_meter_panel = SignalMeterPanel()
self._qrp_panel = QRPPanel()
```

Refactored approach: Use a try/except import pattern and feature flags:
```python
# Optional ham radio panels
try:
    from sdr_module.gui.radio_tuner import RadioTunerWidget
    from sdr_module.gui.callsign_panel import CallsignPanel
    from sdr_module.gui.sstv_panel import SSTVPanel
    from sdr_module.gui.signal_meter_widget import SignalMeterPanel
    from sdr_module.gui.qrp_panel import QRPPanel
    HAS_HAM_RADIO = True
except ImportError:
    HAS_HAM_RADIO = False
```

Then guard panel creation:
```python
if HAS_HAM_RADIO:
    self._callsign_panel = CallsignPanel()
    # ... etc
```

**Step 2: Move ham radio DSP backends to a subpackage**

| Source | Destination |
|--------|-------------|
| `src/sdr_module/dsp/qrp.py` | `src/sdr_module/ham/qrp.py` |
| `src/sdr_module/dsp/sstv.py` | `src/sdr_module/ham/sstv.py` |
| `src/sdr_module/dsp/signal_meter.py` | `src/sdr_module/ham/signal_meter.py` |
| `src/sdr_module/dsp/callsign.py` | `src/sdr_module/ham/callsign.py` |
| `src/sdr_module/gui/radio_tuner.py` | `src/sdr_module/ham/gui/radio_tuner.py` |
| `src/sdr_module/gui/qrp_panel.py` | `src/sdr_module/ham/gui/qrp_panel.py` |
| `src/sdr_module/gui/sstv_panel.py` | `src/sdr_module/ham/gui/sstv_panel.py` |
| `src/sdr_module/gui/signal_meter_widget.py` | `src/sdr_module/ham/gui/signal_meter_widget.py` |
| `src/sdr_module/gui/callsign_panel.py` | `src/sdr_module/ham/gui/callsign_panel.py` |

This creates a clear `sdr_module.ham` namespace that can be installed or omitted.

**Step 3: Update `dsp/__init__.py` and `gui/__init__.py`**

Remove ham radio exports from the core DSP and GUI namespaces. Add them to `ham/__init__.py` instead.

**Step 4: Update test imports**

Update `tests/test_radio_tuner.py`, `tests/test_gui.py`, `tests/test_ui_components.py` to import from `sdr_module.ham.*`.

### 3.3 Defer text encoders

Text encoders (549 lines) are lightly coupled. The only internal dependency is `dsp/callsign.py` importing `MorseEncoder` from `protocols/encoders.py`.

**Action:** Keep the encoders in-tree for now (they're small), but:
- Remove the `encode` subcommand from `cli.py` (TX encoding is premature without validated TX workflows)
- Move `tools/text_encoder.py` and `examples/text_encoding_example.py` to a `deferred/` directory
- De-emphasize in README (move from "Quick Start" to a "Utilities" section at the bottom)

### 3.4 Deliverable

The main package (`sdr_module`) contains only dual-SDR core, DSP, protocol decoders, and visualization. Ham radio features live in `sdr_module.ham` (optional). Antenna array is a separate package.

---

## Phase 4: Double Down (Harden the Core)

**Goal:** Make the retained core reliable, honest, and tested against reality.
**Risk:** Low-medium. Requires actual hardware testing and performance work.
**Estimated scope:** Net new work (not refactoring).

### 4.1 Validate protocol decoders against real-world captures

The six real protocol decoders (POCSAG, AX.25/APRS, RDS, ADS-B, FLEX, ACARS) have correct-looking algorithms but no evidence of testing against real RF captures.

**Action:**
- Source or create I/Q sample recordings for each protocol (publicly available test vectors exist for ADS-B, POCSAG, RDS)
- Add integration tests in `tests/` that decode known captures and verify output
- Document any protocol features that decode correctly vs. partially
- Add a `samples/` directory with small test recordings and expected outputs

**Priority order (by real-world utility):**
1. ADS-B (1090 MHz) -- most popular SDR use case, abundant test data
2. POCSAG -- well-documented protocol, easy to validate
3. RDS -- every FM radio broadcasts it, easy to capture
4. AX.25/APRS -- active amateur radio network
5. FLEX -- legacy but still used in some areas
6. ACARS -- aviation, requires proximity to airport

### 4.2 Performance profiling and bandwidth documentation

**Action:**
- Profile the DSP pipeline (`spectrum.py`, `demodulators.py`, `filters.py`) with realistic sample sizes
- Measure maximum sustainable throughput for each demodulator on a reference machine
- Document realistic bandwidth limits in SPEC_SHEET.md (e.g., "FM demodulation sustains 2.4 MHz real-time on Intel i5")
- Identify hot loops that could benefit from NumPy vectorization improvements
- Consider adding an optional Cython or Numba fast path for the spectrum analyzer FFT pipeline

### 4.3 Harden dual-SDR orchestration

The `DualSDRController` (`core/dual_sdr.py`, 524 lines) is the unique value proposition. It needs to handle real-world failure modes:

**Action:**
- Add device disconnection handling (USB hot-unplug)
- Add buffer overrun recovery (currently samples are silently dropped -- `devices/rtlsdr.py:333`)
- Add USB error retry logic with backoff
- Test all five operation modes (DUAL_RX, FULL_DUPLEX, TX_MONITOR, WIDEBAND_SCAN, RELAY) with actual or simulated devices
- Add integration tests using mock devices that simulate failure conditions (not just happy-path mocking)
- Document the actual state machine transitions in a diagram

### 4.4 Improve test quality

Current tests are mocking-heavy and often verify interface contracts rather than algorithm correctness.

**Action:**
- Add algorithmic validation tests for core demodulators:
  - Generate known FM signal, demodulate, verify audio matches expected
  - Generate known AM signal, demodulate, verify envelope matches expected
  - Generate known CW signal, decode, verify text matches expected
- Add filter response tests:
  - Verify FIR filter frequency response matches design specifications
  - Verify AGC converges to target level within expected time
- Reduce mock usage in `test_dual_sdr.py` -- use `FakeDevice` classes that simulate real timing and buffer behavior
- Target: every DSP module has at least one "round-trip" test (generate signal -> process -> verify output)

### 4.5 Rewrite README for the refocused product

**Action:** Rewrite README.md to reflect the refocused product:

**Structure:**
1. One-line description (dual-SDR framework, nothing else)
2. What it does (3-4 bullet points, only real features)
3. Hardware support table (keep as-is, it's accurate)
4. Installation (simplify -- drop mention of unused optional extras)
5. Quick Start: GUI, CLI, Python API (keep existing examples, they're good)
6. Supported protocols (only the six real ones)
7. Dual-SDR operation modes (keep, this is the differentiator)
8. Known limitations (new -- performance, hardware requirements)
9. Development (keep)
10. Optional: Ham radio features (brief mention, point to `sdr_module.ham`)
11. Optional: Antenna array (brief mention, point to separate package)

Remove: Plugin system section, protocol encoding from Quick Start, inflated protocol table.

### 4.6 Deliverable

A 0.2.0 release with validated protocol decoders, documented performance characteristics, hardened dual-SDR control, and honest documentation. Fewer features claimed, more features proven.

---

## Summary

| Phase | What | Lines Affected | Risk | Dependencies |
|-------|------|---------------|------|-------------|
| **1: Stop Lying** | Fix README + SPEC_SHEET | ~50 edits | Zero | None |
| **2: Cut Dead Weight** | Delete plugins, MX-K2, phantom protocols | ~4,200 deleted | Low | None |
| **3: Extract Products** | Move antenna array + ham radio | ~12,000 moved | Medium | Phase 2 |
| **4: Double Down** | Validate, profile, harden, rewrite docs | Net new work | Low-Med | Phase 1-3 |

### Execution order

Phases 1 and 2 can be done in a single session. Phase 3 requires careful refactoring of `gui/main_window.py` and should be its own PR. Phase 4 is ongoing work that can be broken into individual issues.

### What the codebase looks like after all four phases

```
sdr-module/
├── src/sdr_module/
│   ├── core/              # Device management, dual-SDR controller, config
│   │   ├── config.py
│   │   ├── device_manager.py
│   │   ├── dual_sdr.py        # THE differentiator
│   │   ├── frequency_manager.py
│   │   └── sample_buffer.py
│   ├── devices/           # Hardware drivers (RTL-SDR + HackRF only)
│   │   ├── base.py
│   │   ├── rtlsdr.py
│   │   └── hackrf.py
│   ├── dsp/               # Signal processing (focused)
│   │   ├── spectrum.py
│   │   ├── demodulators.py
│   │   ├── filters.py
│   │   ├── classifiers.py
│   │   ├── protocols.py       # 7 real protocol decoders
│   │   ├── recording.py
│   │   ├── scanner.py
│   │   ├── afc.py
│   │   └── frequency_lock.py
│   ├── gui/               # Core visualization only
│   │   ├── app.py
│   │   ├── main_window.py     # Refactored, optional panel support
│   │   ├── spectrum_widget.py
│   │   ├── waterfall_widget.py
│   │   ├── control_panel.py
│   │   ├── constellation.py
│   │   ├── decoder_panel.py
│   │   └── device_dialog.py
│   ├── ham/               # Optional ham radio extras
│   │   ├── callsign.py
│   │   ├── signal_meter.py
│   │   ├── sstv.py
│   │   ├── qrp.py
│   │   └── gui/
│   │       ├── radio_tuner.py
│   │       ├── callsign_panel.py
│   │       ├── sstv_panel.py
│   │       ├── signal_meter_widget.py
│   │       └── qrp_panel.py
│   ├── protocols/         # Encoder utilities (kept, de-emphasized)
│   │   ├── base.py
│   │   ├── encoder.py
│   │   ├── encoders.py
│   │   └── detector.py
│   ├── ui/                # Visualization components
│   │   ├── waterfall.py
│   │   ├── constellation.py
│   │   ├── signal_meter.py
│   │   ├── time_domain.py
│   │   └── packet_highlighter.py
│   └── utils/             # Helpers
│       ├── conversions.py
│       ├── iq.py
│       └── tooltips.py
├── packages/
│   └── sdr-antenna-array/     # Separate package
│       ├── pyproject.toml
│       ├── src/sdr_antenna_array/
│       │   ├── array_controller.py
│       │   ├── array_config.py
│       │   ├── beamformer.py
│       │   ├── adaptive_beamformer.py
│       │   ├── doa.py
│       │   ├── cross_correlator.py
│       │   ├── calibration.py
│       │   └── timestamped_buffer.py
│       └── tests/
├── samples/                   # NEW: Real-world I/Q test recordings
├── tests/                     # Focused test suite
└── pyproject.toml
```

**Lines before refocus:** ~36,000
**Lines after refocus (core):** ~18,000
**Lines preserved (extracted):** ~12,000 (antenna array + ham radio)
**Lines deleted (dead weight):** ~4,200 (plugins + MX-K2 + phantom refs)

The core is half the size, twice as honest, and everything it claims actually works.
