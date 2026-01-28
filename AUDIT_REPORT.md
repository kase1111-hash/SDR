# SDR Module Software Audit Report

**Date:** 2026-01-28
**Repository:** kase1111-hash/SDR
**Branch:** claude/audit-software-correctness-EcSPc
**Audit Scope:** Full codebase correctness and fitness for purpose

---

## Executive Summary

This audit identified **67 issues** across the SDR software codebase:

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 12 | Issues that could cause safety hazards, data loss, or security breaches |
| **HIGH** | 18 | Issues that significantly impact functionality or reliability |
| **MEDIUM** | 24 | Issues that affect code quality or have limited impact |
| **LOW** | 13 | Minor issues or code style improvements |

### Key Findings

1. **Safety-Critical TX Lockout System is UNTESTED** - The frequency lockout system protecting GPS, aviation, and cellular bands has ZERO test coverage
2. **Thread Safety Issues** - Multiple race conditions in core components
3. **DSP Algorithm Bugs** - Critical errors in interpolator gain scaling and spectral subtraction
4. **Plugin Security** - Arbitrary code execution via unsandboxed plugin loading
5. **API Inconsistencies** - Mixed return types and error handling patterns

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [High Severity Issues](#2-high-severity-issues)
3. [Medium Severity Issues](#3-medium-severity-issues)
4. [Low Severity Issues](#4-low-severity-issues)
5. [Positive Findings](#5-positive-findings)
6. [Recommendations](#6-recommendations)

---

## 1. Critical Issues

### 1.1 Thread Safety - Race Conditions

#### CRIT-01: DualSDRController State Not Thread-Safe
**File:** `src/sdr_module/core/dual_sdr.py:147-402`
**Issue:** The `_state` object is accessed and modified without synchronization across multiple threads. Device callbacks execute in background threads while the main thread reads/writes state without locks.

**Impact:** State corruption, inconsistent view of device status, data races

**Fix Required:**
```python
self._state_lock = threading.Lock()
# Use lock for all _state access
```

#### CRIT-02: Callback Reference Mutation Without Synchronization
**File:** `src/sdr_module/core/dual_sdr.py:246-258`
**Issue:** Callback pointers are reassigned from main thread while device threads reference them, creating TOCTOU (time-of-check-time-of-use) race window.

#### CRIT-03: DeviceManager Registry Not Thread-Safe
**File:** `src/sdr_module/core/device_manager.py:49,164,179,190-191`
**Issue:** `_devices` dictionary accessed without synchronization. Concurrent calls to `open_device()`, `close_device()`, or `close_all()` can corrupt the dictionary.

#### CRIT-04: SampleBuffer.clear() Doesn't Wake Blocked Readers
**File:** `src/sdr_module/core/sample_buffer.py:260-266`
**Issue:** Clear operation notifies writers (`_not_full.notify_all()`) but not readers. Blocked readers on `_not_empty.wait()` will hang indefinitely.

**Fix Required:**
```python
def clear(self) -> None:
    with self._lock:
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._not_full.notify_all()
        self._not_empty.notify_all()  # ADD THIS
```

### 1.2 DSP Algorithm Errors

#### CRIT-05: Spectral Subtraction Overlap-Add Broken
**File:** `src/sdr_module/dsp/filters.py:2109-2118`
**Issue:** The overlap-add implementation does nothing - the loop continues and skips all iterations. Overlapping samples are never added to output.

```python
# Current broken code:
for j in range(min(len(self._overlap_buffer), len(frame_clean))):
    if j < len(output) - len(self._overlap_buffer):
        continue  # ALWAYS continues, never adds overlap
output.extend(frame_clean[: self._hop_size])
```

**Impact:** Noise reduction output has discontinuities and energy loss at frame boundaries.

#### CRIT-06: Interpolator/Resampler Filter Scaling Error
**File:** `src/sdr_module/dsp/filters.py:605-607, 787-788`
**Issue:** Incorrect filter coefficient normalization. Formula `h /= np.sum(h) / self._factor` is mathematically wrong - it doubles the gain correction.

```python
# Current buggy code:
h *= self._factor
h /= np.sum(h) / self._factor  # WRONG - results in h *= factor² / sum(h)

# Correct code:
h *= self._factor
h /= np.sum(h)  # Normalize to unity, then scale by factor
```

**Impact:** For factor=2, output will be 4x too loud.

#### CRIT-07: HackRF Gain Allocation Algorithm Bug
**File:** `src/sdr_module/devices/hackrf.py:246-269`
**Issue:** For gain values < 24 dB, the algorithm gives wrong gain (e.g., request 20 dB, get 40 dB).

```python
# Current code gives wrong results for low gains:
lna_gain = min(40, max(0, int(gain_db / 3) * 8))  # 20/3=6, *8=48, min(40,48)=40
remaining = gain_db - lna_gain                      # 20-40=-20
vga_gain = min(62, max(0, int(remaining / 2) * 2)) # max(0,-10)=0
# Result: 40 dB instead of 20 dB!
```

### 1.3 Security Vulnerabilities

#### CRIT-08: Arbitrary Code Execution via Plugin System
**File:** `src/sdr_module/plugins/manager.py:362`
**Issue:** Plugin loading uses `exec_module()` without sandboxing or code signing. Any `.py` file in plugin directories is executed with full privileges.

**Attack Vector:** Place malicious Python file in `~/.sdr_module/plugins/`

#### CRIT-09: Unbounded Serial Buffer Read (DoS)
**File:** `src/sdr_module/devices/mxk2_keyer.py:378`
**Issue:** `read_until(b"\r")` has no size limit. A malicious serial device can exhaust memory.

**Fix Required:**
```python
response = self._serial.read_until(b"\r", size=1024)  # Add size limit
```

### 1.4 Test Coverage Gap

#### CRIT-10: TX Lockout System Completely Untested
**File:** `src/sdr_module/core/frequency_manager.py` (1,467 lines, 0% tested)
**Issue:** The safety-critical TX frequency lockout system has ZERO test coverage:
- GPS lockout validation (L1, L2, L5, GLONASS, Galileo, BeiDou)
- Aviation emergency frequencies (121.5 MHz, 243 MHz)
- ADS-B/Mode S lockout (1090 MHz)
- Cellular band protection
- License class enforcement
- Power limit validation

**Risk:** Could inadvertently transmit on protected frequencies causing safety hazards and federal violations.

### 1.5 Resource Management

#### CRIT-11: Device Open Failure Resource Leak
**File:** `src/sdr_module/core/device_manager.py:150-156`
**Issue:** Device objects created but not cleaned up when `open()` fails.

```python
device = self.create_device(device_type)  # Object created
if not device.open(index):
    return None  # Device object discarded without close()!
```

#### CRIT-12: Serial Port Leak in MXK2Keyer
**File:** `src/sdr_module/devices/mxk2_keyer.py:261-332`
**Issue:** If exception occurs after `serial.Serial()` but before completion, port remains open.

---

## 2. High Severity Issues

### 2.1 Error Handling

#### HIGH-01: Silent Exception in HackRF stop_rx()
**File:** `src/sdr_module/devices/hackrf.py:384-385`
```python
except Exception:
    pass  # No logging!
```

#### HIGH-02: Silent Exception in HackRF stop_tx()
**File:** `src/sdr_module/devices/hackrf.py:472-473`
```python
except Exception:
    pass  # No logging!
```

#### HIGH-03: Silent Exception in Queue Operations
**Files:** `hackrf.py:359-360`, `rtlsdr.py:313-314`
**Issue:** Catches all exceptions when only `queue.Full` expected. Should be specific.

#### HIGH-04: Config File I/O Has No Error Handling
**File:** `src/sdr_module/core/config.py:139-154`
**Issue:** JSON save/load operations lack error handling. Exceptions propagate without context.

#### HIGH-05: Config Deserialization Has No Validation
**File:** `src/sdr_module/core/config.py:109-135`
**Issue:** No validation of loaded values. Invalid frequencies, sample rates, or gains accepted silently.

### 2.2 Thread Safety

#### HIGH-06: TOCTOU Race in DeviceManager.get_rtlsdr/get_hackrf()
**File:** `src/sdr_module/core/device_manager.py:250-264`
**Issue:** Time-of-check-time-of-use vulnerability in device caching. Same device could be opened twice.

#### HIGH-07: DualSDRController Partial Failure Handling
**File:** `src/sdr_module/core/dual_sdr.py:323-329`
**Issue:** Incomplete rollback on second device failure in `start_full_duplex()`. Callbacks remain set after error.

#### HIGH-08: No Thread Safety for Device State Updates
**Files:** `rtlsdr.py`, `hackrf.py` (multiple locations)
**Issue:** State object modified by multiple threads without locks.

### 2.3 DSP Issues

#### HIGH-09: AFC Unused Noise Floor Calculation
**File:** `src/sdr_module/dsp/afc.py:245`
**Issue:** Noise floor computed but result discarded (dead code).
```python
10 * np.log10(np.median(power) + 1e-12)  # Result not assigned!
```

#### HIGH-10: Polyphase Decimation Time-Reversal Issue
**File:** `src/sdr_module/dsp/filters.py:477-499`
**Issue:** Comment claims filter is "time-reversed for convolution" but no reversal occurs.

#### HIGH-11: Spectrum Window Normalization Incomplete
**File:** `src/sdr_module/dsp/spectrum.py:95,119,230`
**Issue:** Uses `sum(window)²` instead of proper ENBW correction. Power readings off by ~1.76 dB for Hann window.

### 2.4 Device Driver Issues

#### HIGH-12: RTL-SDR Missing Frequency Validation
**File:** `src/sdr_module/devices/rtlsdr.py:206-226`
**Issue:** No validation against spec limits (HackRF does this correctly).

#### HIGH-13: TX Lockout Only in start_tx(), Not set_frequency()
**File:** `src/sdr_module/devices/hackrf.py`
**Issue:** Lockout check occurs too late. Device state can reflect unsafe frequency.

### 2.5 Security

#### HIGH-14: Path Traversal in Plugin Template Creation
**File:** `src/sdr_module/plugins/manager.py:629-636`
**Issue:** User-controlled `output_dir` without boundary checking.

#### HIGH-15: Unsafe JSON Config Deserialization
**File:** `src/sdr_module/core/config.py:143-147`
**Issue:** Config files loaded without schema validation. Malicious values possible.

### 2.6 API Design

#### HIGH-16: Inconsistent Return Types for Success/Failure
**Issue:** Three different patterns used:
- Boolean (`set_frequency() -> bool`)
- Optional (`create_device() -> Optional[SDRDevice]`)
- Optional array (`read() -> Optional[np.ndarray]`)

#### HIGH-17: Mixed Exception vs. Return Value Error Handling
**Issue:** Some methods raise `NotImplementedError`, others return `False` with logging.

#### HIGH-18: Inconsistent Parameter Ordering
**Issue:** Constructor parameters have no consistent pattern across similar classes.

---

## 3. Medium Severity Issues

### 3.1 Error Handling (12 issues)

| ID | File | Line | Issue |
|----|------|------|-------|
| MED-01 | hackrf.py | 120-121 | Silent failure getting serial number |
| MED-02 | rtlsdr.py | 109-110 | Silent failure in get_device_serial |
| MED-03 | recording.py | 2486-2487 | Silent failure checking WAV file |
| MED-04 | recording.py | 2663-2687 | Silent failures in format detection |
| MED-05 | manager.py | 394-396 | Silent failure getting plugin metadata |
| MED-06 | detector.py | 161-162 | Silent failure in protocol decoding |
| MED-07 | protocols.py | 633-634 | Silent failure in string decoding |
| MED-08 | device_manager.py | 204-225 | apply_config continues on failure |
| MED-09 | dual_sdr.py | 159-169 | Callbacks may execute after shutdown |
| MED-10 | hackrf.py | 388, 476 | Thread join without exception handling |
| MED-11 | recording.py | 2620-2626 | Default format used without warning |
| MED-12 | frequency_manager.py | 1256-1301 | Large bandwidth edge case |

### 3.2 DSP Issues (4 issues)

| ID | File | Line | Issue |
|----|------|------|-------|
| MED-13 | frequency_lock.py | 476-481 | Drift rate assumes 10ms intervals |
| MED-14 | filters.py | 160-169 | FIR filter no input size validation |
| MED-15 | spectrum.py | Various | Missing ENBW correction |
| MED-16 | afc.py | 245 | Dead code (unused calculation) |

### 3.3 API Design (5 issues)

| ID | Issue |
|----|-------|
| MED-17 | Missing type hints (`dict` instead of `Dict[str, Any]`) |
| MED-18 | Inconsistent None vs Optional returns |
| MED-19 | Inconsistent `set_*` return types |
| MED-20 | Inconsistent callback signatures |
| MED-21 | State methods lack precondition docs |

### 3.4 Test Coverage (3 issues)

| ID | Module | Issue |
|----|--------|-------|
| MED-22 | frequency_lock.py | Complex state machine untested |
| MED-23 | demodulators.py | No validation of DSP correctness |
| MED-24 | device drivers | HackRF/RTL-SDR no direct tests |

---

## 4. Low Severity Issues

| ID | File | Line | Issue |
|----|------|------|-------|
| LOW-01 | rtlsdr.py | 253-269 | Gain clamping without warning |
| LOW-02 | mxk2_keyer.py | 581-606 | Silent input filtering |
| LOW-03 | rtlsdr.py | 311-314 | Bare except (queue.Full expected) |
| LOW-04 | sample_buffer.py | 100-109 | Stats property copy overhead |
| LOW-05 | plugins/base.py | 37-63 | Exception hierarchy not used consistently |
| LOW-06 | frequency_manager.py | Various | Large bandwidth edge case |
| LOW-07 | cli.py | 131-133 | Unrestricted file path in CLI |
| LOW-08 | recording.py | 767 | Integer overflow risk (theoretical) |
| LOW-09 | manager.py | 120 | Buffer exhaustion (large JSON) |
| LOW-10 | recording.py | 529,578 | TOCTOU race in file stat |
| LOW-11 | mxk2_keyer.py | 597-601 | Serial protocol input sanitization |
| LOW-12 | Various | Various | Missing thread-safety documentation |
| LOW-13 | Various | Various | Incomplete type annotations |

---

## 5. Positive Findings

The audit also identified several well-implemented aspects:

### 5.1 Safety Features
- **GPS/GNSS frequency lockout coverage is comprehensive** with conservative margins
- **TX lockout implementation exists** in HackRF driver (though timing could be better)
- **Hard-coded lockouts** for aviation, cellular, emergency beacons, ADS-B

### 5.2 Resource Management
- Device `close()` methods properly use try/finally patterns
- Context manager support (`__enter__`/`__exit__`) implemented
- MXK2Keyer uses proper threading locks for serial communication

### 5.3 Architecture
- Clean separation of concerns (core, devices, DSP, protocols, GUI, plugins)
- Plugin architecture is extensible
- Well-documented code with docstrings

### 5.4 Test Coverage (Good Areas)
- Sample buffer threading well tested
- Device manager creation/scanning tested
- Filter bank frequency response tested
- MXK2 keyer comprehensively tested
- I/Q conversions and utilities tested

---

## 6. Recommendations

### 6.1 Immediate Actions (Critical)

1. **Add threading locks** to DualSDRController for `_state` and callback references
2. **Add threading lock** to DeviceManager for `_devices` dictionary
3. **Fix SampleBuffer.clear()** to notify `_not_empty`
4. **Add device.close()** on open failure paths
5. **Fix interpolator/resampler** filter scaling formula
6. **Fix spectral subtraction** overlap-add implementation
7. **Fix HackRF gain allocation** algorithm
8. **Add size limit** to MXK2Keyer serial read

### 6.2 High Priority (This Sprint)

9. **Create test_frequency_manager.py** (~200+ test cases for TX lockouts)
10. **Add frequency validation** to RTL-SDR matching HackRF implementation
11. **Add logging** to HackRF stop methods
12. **Add error handling** to config file I/O
13. **Add validation layer** in Config.from_dict()
14. **Implement plugin sandboxing** or code signing

### 6.3 Medium Priority (Next Sprint)

15. Create test_frequency_lock.py (~100+ test cases)
16. Create test_demodulators.py (~150+ test cases)
17. Add TX frequency validation to `set_frequency()` in HackRF
18. Convert bare `except Exception: pass` to specific exceptions
19. Document thread-safety contracts for all classes
20. Standardize API return types

### 6.4 Long Term

21. Add fuzzing tests for parsers and encoders
22. Add stress tests for concurrent operations
23. Add real-world signal tests for demodulators
24. Complete type annotations across codebase
25. Add performance benchmarks

---

## Appendix A: Files Audited

```
src/sdr_module/
├── __init__.py
├── cli.py
├── config.py
├── core/
│   ├── dual_sdr.py
│   ├── device_manager.py
│   ├── sample_buffer.py
│   └── frequency_manager.py
├── devices/
│   ├── base.py
│   ├── rtlsdr.py
│   ├── hackrf.py
│   └── mxk2_keyer.py
├── dsp/
│   ├── filters.py
│   ├── spectrum.py
│   ├── afc.py
│   ├── frequency_lock.py
│   ├── recording.py
│   ├── protocols.py
│   ├── classifiers.py
│   └── [other DSP modules]
├── protocols/
│   ├── base.py
│   ├── detector.py
│   └── encoders.py
├── plugins/
│   ├── base.py
│   ├── manager.py
│   └── registry.py
└── gui/
    └── [GUI modules]
```

---

## Appendix B: Test Coverage Summary

| Module | Test File | Coverage |
|--------|-----------|----------|
| core/sample_buffer.py | test_sample_buffer.py | ~80% |
| core/device_manager.py | test_device_manager.py | ~70% |
| core/dual_sdr.py | test_dual_sdr.py | ~60% |
| core/frequency_manager.py | NONE | **0%** |
| devices/mxk2_keyer.py | test_mxk2_keyer.py | ~85% |
| devices/rtlsdr.py | NONE | **0%** |
| devices/hackrf.py | NONE | **0%** |
| dsp/filters.py | test_filters.py | ~50% |
| dsp/frequency_lock.py | NONE | **0%** |
| utils/conversions.py | test_conversions.py | ~90% |
| utils/iq.py | test_iq.py | ~85% |
| protocols/encoders.py | test_encoders.py | ~75% |

**Overall Estimated Coverage:** 30-40%

---

*Report generated by software audit process*
