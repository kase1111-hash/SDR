# Remediation Plan — SDR Module Vibe-Code Audit

Based on the findings in `VIBE_CODE_AUDIT.md`, this plan addresses all 8 identified problems
organized into 6 work items, ordered by priority.

---

## Work Item 1: Fix Silent Exception Swallowing (High Priority)

**Problem:** 4 bare `except Exception:` clauses in `dsp/recording.py` catch errors without
even capturing the exception variable, making debugging impossible.

**Files to change:**
- `src/sdr_module/dsp/recording.py` — lines ~2490, ~2672, ~2682, ~2695

**Changes:**
1. Line ~2490 (`_has_wav_file`): Change `except Exception:` to `except Exception as e:` and add `logger.debug(f"WAV detection failed: {e}")`
2. Lines ~2672, ~2682, ~2695 (`_infer_sample_format`): Same pattern — capture `as e`, add `logger.debug()` for each format probe (UINT8, INT16, FLOAT32)

**Scope:** 4 edits in 1 file. No behavior change — only adds logging.

---

## Work Item 2: Narrow Broad Exception Catches (High Priority)

**Problem:** 50 `except Exception as e` clauses across 11 files. Hardware drivers are
defensible (hardware APIs throw unpredictable errors), but DSP/protocol code should
catch specific exceptions.

**Files to change:**
- `src/sdr_module/dsp/protocols.py` — callback handler: narrow to `(TypeError, ValueError, RuntimeError)`
- `src/sdr_module/dsp/recording.py` — 3 logged handlers: narrow to `(OSError, ValueError)` for file I/O
- `src/sdr_module/protocols/detector.py` — 2 handlers: narrow to `(ValueError, IndexError)`
- `src/sdr_module/ham/sstv.py` — 2 handlers: narrow to `(ValueError, IndexError, OSError)`

**Files to leave unchanged (defensible broad catches):**
- `devices/rtlsdr.py` (12) — hardware driver, unpredictable ctypes errors
- `devices/hackrf.py` (17) — hardware driver, same reason
- `core/device_manager.py` (4) — device enumeration probes
- `gui/*.py` (8) — top-level GUI error boundaries

**Scope:** ~8 edits across 4 files. Narrowing exception types only.

---

## Work Item 3: Add Exception Chaining (Medium Priority)

**Problem:** Zero uses of `raise X from e` in the entire codebase. When exceptions are
wrapped or re-raised, the original cause is lost.

**Files to change:**
- `src/sdr_module/core/config.py` — `ConfigValidationError` raises: add `from e`
- `src/sdr_module/devices/rtlsdr.py` — re-raises in `open()`/`close()`: add `from e`
- `src/sdr_module/devices/hackrf.py` — re-raises in `open()`/`close()`: add `from e`

**Approach:** Search for patterns like `raise SomeException(...)` inside `except` blocks
and add `from e` to preserve the chain. Only modify cases where a new exception is raised
within a catch block.

**Scope:** ~6-10 edits across 3 files.

---

## Work Item 4: Add GUI Error Notification System (High Priority)

**Problem:** Several error paths only log to console with `logger.error()` and never
notify the user. The status bar exists but is only used for operational state, not errors.

**Current state:**
- Status bar exists in `main_window.py` with device/rate/buffer/recording labels
- QMessageBox used for 15 explicit errors (TX, callsign, device selection)
- 3 `logger.error()` calls with NO user notification:
  - `main_window.py:714` — radio tuner frequency failure
  - `device_dialog.py:141,146` — device enumeration failures
  - `app.py:112` — application runtime error

**Files to change:**
- `src/sdr_module/gui/main_window.py`:
  - Add a `_show_status_error(message, duration_ms=5000)` helper that sets status bar text
    with a warning style (red/orange text) and auto-clears via QTimer
  - Wire it to the tuner error at line ~714
  - Expose it for child widgets to call

- `src/sdr_module/gui/device_dialog.py`:
  - Show enumeration failures in the device list (e.g., "RTL-SDR: scan failed") instead
    of silently logging

**Design:** Use status bar for transient errors (auto-clearing after 5s). Keep QMessageBox
for blocking errors that need user acknowledgment (TX failures, missing devices). No new
toast/notification widget — keep it simple.

**Scope:** ~3-4 methods added, ~4 call sites wired up.

---

## Work Item 5: Add WHY Comments for Magic Numbers (Medium Priority)

**Problem:** 28 magic numbers across 4 files lack explanatory comments. All comments are
WHAT-focused; domain decisions (thresholds, constants, coefficients) need WHY rationale.

**Files to change (28 inline comments total):**

1. `src/sdr_module/dsp/filters.py` — 10 magic numbers:
   - `1e-10` sinc zero-guard (line ~83), `1.0` center tap (line ~105),
     `1e-10` gain threshold (line ~115), `8.0` Kaiser beta (lines ~141, ~371),
     `1e-20` log floor (lines ~208, ~1027), `8` taps multiplier (line ~343),
     `0.8` Nyquist margin (line ~756)

2. `src/sdr_module/dsp/demodulators.py` — 8 magic numbers:
   - `0.001` DC alpha (line ~70), FM normalization (line ~119),
     `0.01` timing alpha (line ~304), Gaussian alpha (line ~323),
     `0.5` MSK index (line ~595), LLR normalization (line ~1027),
     `1.2` PARIS timing (line ~1328)

3. `src/sdr_module/dsp/protocols.py` — 7 magic numbers:
   - `0x7CD215D8` POCSAG sync (line ~117), `0x7A89C197` idle (line ~118),
     `576` preamble bits (line ~119), `0x769` BCH poly (line ~122),
     `0x7E` HDLC flag (line ~497), `0xFFFF` CRC init (line ~541),
     `0x5B9` RDS poly (line ~967)

4. `src/sdr_module/core/frequency_manager.py` — 3 magic numbers:
   - `1.5` power headroom (line ~32), `15e6` GPS guard (line ~157),
     `0.05` CTCSS threshold (line ~1468)

**Format:**
```python
# WHY: Kaiser beta=8 gives ~65dB stopband attenuation — good balance between
# transition width and rejection for anti-aliasing filters
window = np.kaiser(n, 8.0)
```

**Scope:** 28 comment-only additions. Zero code changes.

---

## Work Item 6: Add Property-Based & Parametrized Tests (Medium Priority)

**Problem:** Zero parametrized tests, zero property-based tests, zero fuzz tests. 6
demodulators have no tests at all (SSB, FSK, PSK, GFSK, MSK, QAM). 4 DSP modules
entirely untested (AFC, classifiers, frequency_lock, scanner).

**Changes:**

### 6a. Add `hypothesis` to dev dependencies
- Edit `pyproject.toml`: add `"hypothesis>=6.0.0"` to `[project.optional-dependencies.dev]`

### 6b. New test file: `tests/test_dsp_properties.py`
Property-based tests using hypothesis for DSP invariants:
- **Filter properties:** Output length matches input, energy conservation (passband),
  FIR filter linearity (superposition), symmetric impulse response for linear-phase
- **Demodulator round-trips:** FM modulate→demodulate recovers signal (within tolerance),
  AM modulate→demodulate recovers envelope
- **Conversion round-trips:** `linear_to_db(db_to_linear(x)) ≈ x`, frequency format
  round-trips

### 6c. Add parametrized tests to existing files

- `tests/test_filters.py`: Parametrize window functions `@pytest.mark.parametrize("window", ["hamming", "hann", "blackman", "kaiser", "rectangular"])`
- `tests/test_filters.py`: Parametrize sample rates `@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000, 96000])`
- `tests/test_dsp_roundtrip.py`: Parametrize over all demodulator types

### 6d. New test file: `tests/test_demodulators_extended.py`
Tests for the 6 untested demodulators:
- `SSBDemodulator` — USB/LSB tone recovery
- `FSKDemodulator` — binary FSK symbol recovery
- `PSKDemodulator` — BPSK/QPSK constellation recovery
- `GFSKDemodulator` — Gaussian-filtered FSK recovery
- `MSKDemodulator` — minimum shift keying recovery
- `QAMDemodulator` — 16-QAM/64-QAM constellation recovery

### 6e. New test file: `tests/test_protocol_fuzz.py`
Fuzz-style tests for protocol decoders using hypothesis:
- Feed random byte sequences to each decoder, assert no crashes
- Feed structured-but-invalid data (wrong CRC, truncated frames)
- Verify decoders return empty results (not exceptions) on garbage input

### 6f. New test file: `tests/test_untested_modules.py`
Basic coverage for 4 untested DSP modules:
- `AutomaticFrequencyControl` — frequency drift tracking
- `SignalClassifier` — analog/digital classification
- `FrequencyLocker` — lock acquisition and tracking
- `FrequencyScanner` — scan range coverage

**Scope:** 1 config edit, 4 new test files, 2 existing test files modified.

---

## Execution Order

| Step | Work Item | Priority | Est. Edits | Risk |
|------|-----------|----------|------------|------|
| 1 | WI-1: Fix silent exceptions | High | 4 | None — logging only |
| 2 | WI-2: Narrow broad catches | High | 8 | Low — could miss edge cases |
| 3 | WI-3: Exception chaining | Medium | 8 | None — additive |
| 4 | WI-4: GUI error notifications | High | 8 | Low — UI-only |
| 5 | WI-5: WHY comments | Medium | 28 | None — comments only |
| 6 | WI-6: Tests | Medium | ~400 LOC | None — test-only |

**Validation:** After each work item, run `pytest tests/ -v --tb=short` to ensure no
regressions. After WI-4, visually verify status bar error display. After WI-6, run
`pytest tests/ -v` to confirm new tests pass.
