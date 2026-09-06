# Audit History

This file consolidates five separate audit / evaluation reports that were
previously kept as individual documents. Each original report remains
available in git history; see the "source document" column in the table
below for the commit at which each was last present.

| Consolidated into | Source document | Date | Auditor | Verdict |
|---|---|---|---|---|
| §2 | `AUDIT_REPORT.md` | 2026-01-28 | Internal | 67 issues; 12 CRITICAL, 18 HIGH |
| §3 | `VIBE_CODE_AUDIT.md` | 2026-02-22 | Internal | 23% vibe-code confidence ("AI-Assisted") |
| §4 | `EVALUATION_REPORT.md` | 2026-02 | Internal | Feature creep on sound concept |
| §5 | `AGENTIC_SECURITY_AUDIT.md` | 2026-03-11 | claude-opus-4-6 | L1 WARN, L2/L3/L5 PASS, L4 WARN |
| §6 | `AGENTIC_SECURITY_AUDIT_V3.md` | 2026-03-11 | claude-opus-4-6 | L1 WARN, L2/L3/L4/L5 PASS |
| §7 | *(this PR)* | 2026-04-20 | Internal (doc consolidation) | Status roll-up |

To see the full text of an original report, `git log -- <filename>` on the
branch history and check out the pre-consolidation commit.

---

## 1. How to read this file

Findings below are grouped by severity. Each row shows:

- **ID** as it appeared in the original report.
- A one-line summary.
- **Current status** reflecting the state of `main` at the time of this
  consolidation (see `CHANGELOG.md` for the exact fixes).

Status legend:

- ✅ **Fixed** — remediation has landed on `main`.
- 🟡 **Partially fixed** — most of the issue is resolved; one subordinate
  concern remains noted in code or docs.
- ⚠️ **Open** — known issue, not yet addressed.
- ⛔ **Wontfix / obsolete** — no longer applies (e.g. audited component
  has been removed) or deliberately left alone.

---

## 2. AUDIT_REPORT.md (2026-01-28)

Full codebase correctness audit. 67 issues total: 12 CRITICAL, 18 HIGH,
24 MEDIUM, 13 LOW.

### 2.1 Critical

| ID | Summary | Status |
|---|---|---|
| CRIT-01 | `DualSDRController._state` not thread-safe | ✅ Fixed — `_lock: RLock` guards all `_state` access (`core/dual_sdr.py`). |
| CRIT-02 | Callback reassignment TOCTOU in `DualSDRController` | ✅ Fixed — same `_lock` covers callback references. |
| CRIT-03 | `DeviceManager._devices` not thread-safe | ✅ Fixed — `_lock: RLock` guards `_devices` dict. |
| CRIT-04 | `SampleBuffer.clear()` doesn't wake blocked readers | ✅ Fixed — now notifies both `_not_full` and `_not_empty`. |
| CRIT-05 | Spectral-subtraction overlap-add broken | ✅ Fixed — the `continue` loop was removed; see `dsp/filters.py` spectral subtraction path. |
| CRIT-06 | Interpolator/resampler filter scaling doubled gain | ✅ Fixed — formula is now `h *= factor / sum(h)` (`dsp/filters.py:634`). |
| CRIT-07 | HackRF gain allocation gave wrong output for < 24 dB | ✅ Fixed — new strategy clamps to valid range and uses LNA table lookup. |
| CRIT-08 | Plugin system allowed arbitrary code execution | ⛔ Obsolete — plugin system removed (REFOCUS_PLAN Phase 2). |
| CRIT-09 | Unbounded serial read in `mxk2_keyer.py` (DoS) | ⛔ Obsolete — `mxk2_keyer.py` removed. |
| CRIT-10 | TX lockout system had zero test coverage | ✅ Fixed — `tests/test_frequency_manager.py` has 58 tests covering GPS, aviation, ADS-B, cellular, license enforcement, and power limits. |
| CRIT-11 | Device-open failure resource leak | ✅ Fixed — `open_device` and `_open_device_unlocked` call `device.close()` in the failure path. |
| CRIT-12 | Serial port leak in `mxk2_keyer.py` | ⛔ Obsolete — `mxk2_keyer.py` removed. |

### 2.2 High severity

| ID | Summary | Status |
|---|---|---|
| HIGH-01 | Silent `except Exception: pass` in `hackrf.stop_rx()` | ✅ Fixed — logs. |
| HIGH-02 | Silent `except Exception: pass` in `hackrf.stop_tx()` | ✅ Fixed — logs. |
| HIGH-03 | Broad exception catches on queue operations | ✅ Fixed — narrowed to `queue.Full`. |
| HIGH-04 | Config file I/O had no error handling | ✅ Fixed — JSON save/load have explicit error handling with context. |
| HIGH-05 | Config deserialization had no validation | ✅ Fixed — `ConfigValidationError` raised for out-of-range values. |
| HIGH-06 | TOCTOU race in `DeviceManager.get_rtlsdr/get_hackrf()` | ✅ Fixed — `_open_device_unlocked` runs under held `_lock`. |
| HIGH-07 | Partial failure handling in `start_full_duplex()` | ✅ Fixed — incomplete rollback no longer leaves callbacks set. |
| HIGH-08 | No thread safety for device state updates | ✅ Fixed — `SDRDevice._state_lock` (`devices/base.py`). |
| HIGH-09 | AFC unused noise-floor calculation | ✅ Fixed — dead code removed. |
| HIGH-10 | Polyphase decimation time-reversal comment lied | ✅ Fixed — comment and code now match. |
| HIGH-11 | Spectrum window normalization incomplete (ENBW) | ✅ Fixed — uses `sum(window**2)` (`dsp/spectrum.py:97`). |
| HIGH-12 | RTL-SDR missing frequency validation | ✅ Fixed — validation mirrors HackRF. |
| HIGH-13 | TX lockout only in `start_tx()`, not `set_frequency()` | ✅ Fixed — `hackrf.set_frequency`, `set_bandwidth`, and `start_tx` all call `is_tx_allowed`. |
| HIGH-14 | Path traversal in plugin template creation | ⛔ Obsolete — plugin system removed. |
| HIGH-15 | Unsafe JSON config deserialization | ✅ Fixed — file-size cap + schema validation on load. |
| HIGH-16 | Inconsistent return types (bool/Optional/raise) | ⚠️ Open — API surface largely unchanged; revisit for 0.2. |
| HIGH-17 | Mixed exception vs. return-value error handling | ⚠️ Open — same as HIGH-16. |
| HIGH-18 | Inconsistent parameter ordering | ⚠️ Open — cosmetic; revisit for 0.2. |

### 2.3 Medium and low severity

Individual rows not reproduced here. Of the 24 MEDIUM + 13 LOW findings,
most silent-exception and dead-code items have been addressed alongside
the CRIT/HIGH work. The remaining items are API-surface polish (return
types, parameter ordering) earmarked for a future 0.2 release.

---

## 3. VIBE_CODE_AUDIT.md (2026-02-22)

Heuristic audit for AI-generated-code indicators. Final confidence score:
23% ("AI-Assisted, not blindly generated"). The audit identified eight
specific weaknesses; all are tracked by the remediation plan previously
kept as `PLAN.md`:

| WI | Summary | Status |
|---|---|---|
| WI-1 | Silent exception swallowing in `dsp/recording.py` | ✅ Fixed — `except Exception as e: logger.debug(...)` everywhere. |
| WI-2 | Narrow broad exception catches in DSP/protocol code | ✅ Fixed — narrowed to `(TypeError, ValueError, RuntimeError)` in decoders; hardware drivers intentionally left broad. |
| WI-3 | Add exception chaining (`raise X from e`) | ✅ Fixed in `core/config.py`, `devices/rtlsdr.py`, `devices/hackrf.py`. |
| WI-4 | GUI error notification system | ✅ Fixed — `_show_status_error` helper + status-bar toast + `ErrorLogDialog` (`Ctrl+E`). |
| WI-5 | Add WHY comments for magic numbers | ✅ Fixed — comments added across `dsp/filters.py`, `dsp/demodulators.py`, `dsp/protocols.py`, `core/frequency_manager.py`. |
| WI-6 | Property-based + parametrized + fuzz tests | ✅ Fixed — added `tests/test_dsp_properties.py`, `test_demodulators_extended.py`, `test_protocol_fuzz.py`, `test_untested_modules.py`; test count went from ~200 to 735. |

---

## 4. EVALUATION_REPORT.md (2026-02)

Product-level evaluation. Classification: **Feature Creep** with the
secondary tag **Underdeveloped (in claimed areas)**. Key verdicts:

- **Concept:** Sound. Python-native dual-SDR orchestration is a genuine
  gap, albeit with a narrow audience.
- **Execution:** Hardware drivers, DSP, and protocol decoders are real
  and mathematically sound. GUI is functional. README overstated the
  protocol catalogue (claimed P25/DMR/TETRA/LoRa/Zigbee/Z-Wave).
- **Recommendation:** Cut over-claims; extract the antenna-array and
  plugin systems; ship a smaller, honest 0.1.

| Recommendation | Status |
|---|---|
| Remove phantom protocol claims from README / SPEC_SHEET | ✅ Fixed — Supported Protocols table lists only implemented decoders. |
| Remove plugin system | ✅ Fixed — removed per REFOCUS_PLAN Phase 2. |
| Extract antenna array to separate package | ✅ Fixed — lives in `packages/sdr-antenna-array/`. |
| Document real-time bandwidth limitations | ✅ Fixed — "Known Limitations" section in README. |
| Focus 0.1.0 on dual-SDR + DSP + decoding + GUI | ✅ Done. |

---

## 5. AGENTIC_SECURITY_AUDIT.md (2026-03-11)

Five-layer agentic-security audit.

| Layer | Finding | Status |
|---|---|---|
| L1 Provenance | WARN — AI-generated, moderate human review | ℹ️ Acknowledged; nature of the project. |
| L2 Credentials | PASS — no secrets handling | ✅ Remains PASS. |
| L3 Agent Boundaries | PASS — no agentic features | ✅ Remains PASS. |
| L4 Supply Chain | PASS — minimal deps | ✅ Remains PASS; Dependabot bumps tracked by PRs #65, #66. |
| L5 Infrastructure | PASS — TX lockouts, input validation | ✅ Remains PASS; lockouts now test-covered. |

---

## 6. AGENTIC_SECURITY_AUDIT_V3.md (2026-03-11)

Second pass of the agentic audit at commit `b3e9a03`. Notable findings
and current status:

| Finding | Status |
|---|---|
| README listed unimplemented protocols (P25, DMR, TETRA, LoRa, Zigbee, Z-Wave) | ✅ Fixed — README and SPEC_SHEET reflect only implemented decoders. |
| Waterfall referenced phantom protocol colours | ⚠️ Open — `ui/waterfall.py` may still carry colour entries for removed protocols. Cosmetic; low priority. |
| No security tooling in CI (bandit/semgrep/CodeQL) | ⚠️ Open — consider for 0.2. |
| No `.env.example` | ⛔ Wontfix — desktop app with no secrets. |
| TX lockout untested | ✅ Fixed — 58 tests in `tests/test_frequency_manager.py`. |

---

## 7. Roll-up (2026-04-20)

At the time of this consolidation, `main` is at commit `1b155fe` (post
PR #65 GitHub-Actions bulk + PR #66 Python-tooling bulk + PRs #67/#68/#69
UX branch merge). The overall state:

- **12 CRITICAL findings:** 9 fixed, 3 obsolete (plugin / mxk2_keyer
  removed).
- **18 HIGH findings:** 15 fixed, 3 open (API-surface polish only:
  HIGH-16/17/18).
- **Vibe-code remediation (6 work items):** all completed.
- **Evaluation recommendations (5 items):** all completed.
- **Agentic security (5 layers):** no regressions.

Recent additions on top of those audits (see `CHANGELOG.md` `[Unreleased]`):

- Waterfall widget bug fixes (NumPy 2.x `uint8` bitshift overflow;
  empty-spectrum `np.interp` crash).
- Full GUI usability pass (bookmarks, click-to-tune, band presets,
  keyboard shortcuts, audio output, themes, first-run wizard, error
  history, screenshot export, device hot-plug polling, state
  persistence).
- Full mypy cleanup: 40 → 0 findings across `src/sdr_module`.
- `ruff`/`black`/`isort` checks all pass; `pytest` reports 735 passed,
  25 skipped, 0 failures.
- `twine check dist/*` passes (setuptools cap in `[build-system]` until
  twine supports PEP 639 `License-Expression` / `License-File` fields).

### Remaining known items

| Item | Severity | Tracked where |
|---|---|---|
| API-surface inconsistency (return types, param ordering) — HIGH-16/17/18 | Medium | This file §2.2, likely 0.2. |
| Phantom protocol colours in `ui/waterfall.py` | Low | This file §6. |
| No security tooling in CI (bandit/semgrep/CodeQL) | Low | This file §6. |
| `setuptools<77` build cap | Low | `pyproject.toml` has a TODO comment; remove when twine supports PEP 639. |
| NumPy 2.x bump (Dependabot PR #64) | — | Closed pending fresh recreate against post-UX main. |

---

## 8. Production-readiness pass (2026-09)

A 15-dimension production-readiness audit was run with adversarial
verification of each finding. Rate limits meant only the `core-runtime` and
`dsp-numerics` dimensions produced fully verified findings; the other 13
dimensions did not complete and should be re-run. The fixes that landed in
this pass:

### 8.1 Fixed

- **Hardware drivers (blocker).** RTL-SDR enumeration used pyrtlsdr methods
  that never existed, so no RTL-SDR was ever detected; the HackRF driver
  imported a non-existent `hackrf` PyPI package. Both were rewritten against
  the real libraries (`pyrtlsdr`, `python_hackrf`) with fake-backend tests
  that lock the library surface in. See `CHANGELOG.md`.
- **HackRF TX safety (blocker/high).** `write_samples()` returned success
  without transmitting; `start_tx(None)` radiated an uninitialised buffer;
  TX samples were wrapped instead of clipped. All fixed, with the TX lockout
  enforced in every tuning path.
- **DSP correctness (high).** BPSK sliced on the wrong axis (~0.5 BER); SSB
  USB and LSB were byte-identical; spectral noise reduction crashed on
  complex I/Q and dropped samples. The frequency scanner logged one hit per
  overlapping step at the tuned centre (21 hits for one station) instead of
  one hit at the true peak frequency. FrequencyLocker mapped bins with the
  unshifted-FFT convention while the spectra are fftshifted, mirroring the
  detected frequency. SignalClassifier reported a constant 0.5 confidence
  because it was never computed. The Resampler silently stayed 1:1 for ratios
  its hand-rolled search could not represent (now uses
  `Fraction.limit_denominator` and warns when a rate is unrepresentable). The
  block AGC/FastAGC applied the per-sample attack/decay coefficient once per
  block, stretching a 1 ms attack to hundreds of ms (now scaled to the block).
  AFC restarted its correction-tone phase every block, breaking phase
  continuity across blocks (now driven from a running NCO phase). The
  single-input LMS/NLMS noise reducer used the current sample as both filter
  input and desired output, so it predicted the sample from itself and
  cancelled the whole signal (output ~0); a decorrelation delay makes it a
  proper adaptive line enhancer that keeps a narrowband tone and suppresses
  broadband noise. The MSK demodulator's "coherent" matched-filter path had no
  carrier or symbol-timing recovery, so its bit decisions were effectively
  random (~0.5 BER on a clean signal); since MSK is CPFSK with modulation
  index 0.5, bit and soft-bit decisions now always come from the FM
  discriminator, and the `coherent` flag defaults to False. All fixed with
  regression tests.
- **Robustness (medium).** `SampleBuffer.read()/peek()` bounds-checking;
  atomic `SDRConfig.save()`; side-effect-free default-config-path getter;
  the packet-highlighter KeyError crash and its flaky (unseeded) test.
- **Packaging/CI/docs.** Real CI gates (blocking mypy, headless GUI suite,
  antenna-array tests, 3.10-3.14 + Windows/macOS, Bandit + pip-audit,
  CodeQL, wheel smoke test); PEP 639 metadata (the `setuptools<77` cap is
  gone); `py.typed`; single-sourced version; corrected PyInstaller spec,
  Windows installer and build scripts; docs realigned to the shipping code.

### 8.2 Remaining known items (verified but not yet fixed)

These `dsp-numerics` findings were verified as real and are documented here
rather than left implicit. They affect correctness of secondary features and
are candidates for the next pass:

| Item | File | Notes |
|---|---|---|
| SignalClassifier still mislabels some modulation *types* | `dsp/classifiers.py` | Confidence is now computed (fixed); the type-decision heuristics still need tuning. |

Fixed since an earlier revision of this list (now in §8.1): the scanner hit
dedup/frequency, the FrequencyLocker fftshift mismatch, the SignalClassifier
constant-0.5 confidence, the Resampler silent 1:1 fallback, the per-block AGC
attack-time stretch, the AFC per-block NCO phase reset, and the coherent MSK
demodulator's ~0.5 BER. The AFC PI loop gains, also listed earlier, were
checked with a closed-loop simulation and converge cleanly with no oscillation
(the reported oscillation was an open-loop windup artifact), so they were left
unchanged.

### 8.3 Not yet re-audited

The `recording-io`, `protocol-decoders`, `gui-core`, `gui-widgets`, `cli`,
`packaging-ci`, `docs-truth`, `security-tx-safety`, `ham-features`,
`demo-vs-real`, `tests-quality`, `antenna-array` and `ui-viz-utils`
dimensions did not complete in this pass and should be run before a 1.0.
