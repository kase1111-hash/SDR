# Agentic Security Audit v3.0

```
AUDIT METADATA
  Project:       SDR Module (Dual-SDR Framework)
  Repository:    github.com/kase1111-hash/SDR
  Date:          2026-03-11
  Auditor:       Claude Opus 4.6 (claude-opus-4-6)
  Commit:        b3e9a034593ee47578c75153eccfcf7faec55614
  Strictness:    STANDARD
  Context:       PROTOTYPE (Alpha 0.1.0)

PROVENANCE ASSESSMENT
  Vibe-Code Confidence:   23% (AI-Assisted, not blindly generated)
  Human Review Evidence:  MODERATE — PR-based workflow, multi-phase audit cycles

LAYER VERDICTS
  L1 Provenance:       WARN
  L2 Credentials:      PASS
  L3 Agent Boundaries: PASS
  L4 Supply Chain:     PASS
  L5 Infrastructure:   PASS
```

---

## L1: PROVENANCE & TRUST ORIGIN — WARN

### 1.1 Vibe-Code Detection

- [x] **No security config**: No `.env.example` — acceptable for desktop app with no secrets
- [x] **AI boilerplate**: Uniform docstring formatting, section dividers, zero stylistic variation across 57 source files
- [x] **Rapid commit history**: Large initial commits (2,500+ lines), formulaic commit messages (`"Phase N: verb noun"`)
- [ ] ~~No tests~~: **22 test files**, 605+ test functions, 1,034+ assertions, property-based + fuzz testing — STRONG
- [ ] ~~No security config~~: SECURITY.md (150 lines), TX frequency lockouts, input validation — PRESENT
- [ ] ~~Polished README, hollow codebase~~: README corrected via "Phase 1: Correct over-promises" — HONEST
- [ ] ~~Bloated deps~~: 1 required dep (numpy), 5 optional — MINIMAL

**Commit Authorship Breakdown:**
| Author | Commits | % |
|--------|---------|---|
| Claude (AI) | 95 | 63% |
| Kase (human) | 33 | 22% |
| Kase Branham | 17 | 11% |
| dependabot | 6 | 4% |

**Verdict:** AI-generated with genuine iterative refinement. Zero reverts, zero WIP commits, but multi-phase audit-fix cycles show human direction. The code is functionally sound despite synthetic origin.

### 1.2 Human Review Evidence

- [x] Security-focused commits: VIBE_CODE_AUDIT.md, EVALUATION_REPORT.md, "Phase 1-4" hardening cycles
- [x] Security tooling in CI/CD: Black, isort, ruff, mypy in `.github/workflows/ci.yml`
- [x] `.gitignore` excludes `.env`, `.venv`, `env/`, `venv/`, IDE files, build artifacts
- [ ] No dedicated secret scanning (gitleaks, truffleHog) in CI — LOW risk given no secrets in codebase

### 1.3 The "Tech Preview" Trap

- [x] **Alpha label**: `pyproject.toml` declares `"Development Status :: 3 - Alpha"`, version `0.1.0`
- [ ] No production traffic or real users depending on this
- [ ] No disclaimers shifting responsibility — limitations honestly documented

**L1 VERDICT: WARN** — AI provenance is transparent and well-documented. Code quality exceeds typical vibe-coded projects. Warning due to 63% AI authorship with limited human line-by-line review evidence.

---

## L2: CREDENTIAL & SECRET HYGIENE — PASS

### 2.1 Secret Storage

- [ ] ~~Plaintext credentials~~: **None found** — exhaustive search for `api_key`, `secret`, `password`, `token`, `Bearer`, `sk-`, `pk_`
- [ ] ~~API keys in client-side code~~: No web code, no embedded keys
- [ ] ~~Secrets in git history~~: No credential patterns in commit diffs
- [ ] ~~`.env` files committed~~: Properly gitignored
- [x] No `.env.example` — acceptable, project has no secrets to template

### 2.2 Credential Scoping & Lifecycle

- **N/A** — Desktop SDR application with no API keys, OAuth tokens, or user authentication
- Configuration uses dataclass-based validation (`core/config.py`) with JSON persistence
- No credential aggregation risk

### 2.3 Machine Credential Exposure

- [x] GitHub Actions uses OIDC for PyPI publishing (no hardcoded tokens in `.github/workflows/release.yml`)
- [x] Codecov integration with safe `fail_ci_if_error: false`
- [ ] ~~Billing attack surface~~: No cloud services, no pay-per-use APIs

**L2 VERDICT: PASS** — No credentials exist in the codebase. Configuration management is clean. CI/CD uses modern token-free auth (OIDC).

---

## L3: AGENT BOUNDARY ENFORCEMENT — PASS

### 3.1 Agent Permission Model

- **N/A** — No AI agents, LLMs, or agentic features in the application code
- No dynamic code execution: zero `eval()`, `exec()`, `__import__()`, `subprocess`, `os.system` calls
- File system access limited to:
  - Config read/write: `~/.config/sdr_module/config.json` (size-limited to 1MB)
  - I/Q recording: User-specified paths validated via `Path().expanduser().resolve()`
- Network access: **None** — pure desktop application communicating only with local USB SDR hardware

### 3.2 Prompt Injection Defense

- **N/A** — No LLM integration, no prompt construction, no natural language processing

### 3.3 Memory Poisoning

- **N/A** — No persistent AI memory or learning components

### 3.4 Agent-to-Agent Trust

- **N/A** — No inter-agent communication

**L3 VERDICT: PASS** — No agentic attack surface exists. The application is a traditional desktop program with no AI runtime components.

---

## L4: SUPPLY CHAIN & DEPENDENCY TRUST — PASS

### 4.1 Plugin/Skill Supply Chain

- **Former plugin system removed** in commit 80d8425 ("Phase 2: Remove plugin system and MX-K2 keyer driver — 4,308 lines")
- No dynamic plugin loading, no extension points for arbitrary code execution
- Antenna array and ham radio extracted into separate packages under `packages/` with explicit imports

### 4.2 MCP Server Trust

- **N/A** — No MCP servers configured or referenced

### 4.3 Dependency Audit

**Required dependencies (1):**
| Package | Version Spec | Pinning | Assessment |
|---------|-------------|---------|------------|
| numpy | `>=1.21.0` | Floor only | Acceptable for library |

**Optional dependencies (5):**
| Package | Version Spec | Purpose |
|---------|-------------|---------|
| pyrtlsdr | `>=0.2.92` | RTL-SDR hardware driver |
| hackrf | `>=1.0.0` | HackRF hardware driver |
| scipy | `>=1.7.0` | Advanced DSP |
| matplotlib | `>=3.4.0` | Plotting |
| PyQt6 | (unpinned) | GUI framework |

**Findings:**
- [ ] ~~Dependencies not updated in 12+ months~~: dependabot active (6 commits)
- [x] Versions use floor pins (`>=`), not exact pins — standard for Python libraries
- [x] No lock file (acceptable for library, would be needed for application deployment)
- [x] All dependencies are well-known, actively maintained packages from official PyPI
- [x] No private registry URLs, no git dependencies

**L4 VERDICT: PASS** — Minimal dependency footprint. Plugin system was proactively removed. All deps are mainstream, well-audited packages.

---

## L5: INFRASTRUCTURE & RUNTIME — PASS

### 5.1 Database Security

- **N/A** — No database. Persistence via JSON files only, using safe `json.load()`/`json.dump()`

### 5.2 BaaS Configuration

- **N/A** — No backend-as-a-service integration

### 5.3 Network & Hosting

- **N/A** — Desktop-only application with no network listeners, HTTP servers, or REST APIs
- SECURITY.md proactively warns: "Do not expose SDR control interfaces to untrusted networks"

### 5.4 Deployment Pipeline

- [x] CI/CD uses pinned action versions (`actions/checkout@v6`, `actions/setup-python@v6`)
- [x] Release workflow uses OIDC for PyPI — secrets not baked into artifacts
- [x] Multi-Python matrix testing (3.9, 3.10, 3.11, 3.12, 3.13)
- [ ] No explicit dev/staging/prod isolation — acceptable for library project

### 5.5 Input Validation & Safety

- [x] **TX frequency lockouts**: Hard-coded safety zones for GPS, aviation, emergency, ADS-B, cellular bands (`core/frequency_manager.py`)
- [x] **CLI input validation**: `argparse` with typed parameters, explicit choices
- [x] **Config validation**: `__post_init__` range checks — frequency (1 Hz–30 GHz), sample rate (1–100 MS/s), gain (-20–100 dB)
- [x] **Config file size limit**: 1MB max prevents DoS via malformed config
- [x] **No unsafe deserialization**: Zero `pickle`, zero `yaml.load()` — JSON only
- [x] **No shell execution**: Zero `subprocess`, `os.system`, `eval`, `exec`
- [x] **Error handling**: Logging module throughout, no stack traces exposed to users
- [x] **Thread safety**: `RLock` in device management, bounded queues (maxsize=100), proper cleanup

### 5.6 Regulatory Compliance

- [x] TX frequency lockouts align with FCC/ITU regulations for amateur radio
- [x] License-based TX permission system (NONE → TECHNICIAN → GENERAL → AMATEUR_EXTRA)
- [ ] No PII/medical/financial data handled — N/A

**L5 VERDICT: PASS** — No network attack surface. Strong input validation. Excellent RF transmission safety controls. Safe deserialization and error handling throughout.

---

## FINDINGS

### [MEDIUM] — Silent Exception Swallowing in Recording Module
```
Layer:     5
Location:  src/sdr_module/dsp/recording.py (3 instances)
Evidence:  Bare `except Exception:` clauses that silently discard errors without logging
Risk:      Masked failures during I/Q recording could cause data loss without user awareness
Fix:       Add `logger.error(f"Recording error: {e}")` to each bare except clause
```

### [LOW] — Broad Exception Catches Throughout Codebase
```
Layer:     5
Location:  54 instances across src/ (device drivers, DSP, GUI)
Evidence:  `except Exception as e` used instead of specific exception types
Risk:      May catch and suppress unexpected errors, complicating debugging
Fix:       Use specific exceptions (IOError, ValueError, etc.) where the failure mode is known
```

### [LOW] — Missing Exception Chaining
```
Layer:     5
Location:  All exception re-raise/wrap points
Evidence:  No `raise X from e` or `raise X from None` patterns used
Risk:      Lost traceback context when exceptions are wrapped
Fix:       Add `from e` to exception chains for debugging clarity
```

### [LOW] — Shallow GUI Test Coverage
```
Layer:     1
Location:  tests/test_gui.py, tests/test_ui_components.py
Evidence:  Widget creation tests with 0 assertions — verify instantiation but not behavior
Risk:      GUI regressions would not be caught by CI
Fix:       Add assertions for widget state, signal connections, and rendering
```

### [LOW] — No Secret Scanning in CI Pipeline
```
Layer:     2
Location:  .github/workflows/ci.yml
Evidence:  No gitleaks, truffleHog, or git-secrets step
Risk:      Future credential leaks would not be caught pre-merge
Fix:       Add `gitleaks detect` step to CI workflow
```

### [LOW] — All Dependencies Use Floating Minimum Versions — RESOLVED
```
Layer:     4
Location:  pyproject.toml
Status:    RESOLVED — Added major-version upper bounds (<3, <2, <4, <7) to all
           runtime and optional deps. Added PyQt6>=6.4.0,<7 as explicit gui extra.
```

### [LOW] — No Code Signing or SBOM in Distribution — RESOLVED
```
Layer:     4
Location:  .github/workflows/release.yml
Status:    RESOLVED — Added `attest` job with actions/attest-build-provenance@v2
           for sigstore attestation and anchore/sbom-action@v0 for SPDX SBOM
           generation. Both artifacts attached to GitHub releases.
```

---

## SECURITY STRENGTHS

1. **RF Transmission Safety** — Hard-coded frequency lockouts for GPS, aviation, emergency, ADS-B, cellular, and marine distress bands with license-based permission tiers. This is genuinely safety-critical code implemented correctly.

2. **Minimal Attack Surface** — Desktop-only, no network listeners, no database, no web framework, no dynamic code execution. The attack surface is essentially: local file I/O and USB hardware communication.

3. **Clean Dependency Graph** — Single required dependency (numpy). Optional deps are all well-known packages. Former plugin system was proactively removed (4,308 lines deleted).

4. **Transparent Provenance** — AI authorship explicitly documented. Multiple audit-fix cycles tracked in version control. No attempt to obscure synthetic origin.

5. **Property-Based & Fuzz Testing** — Hypothesis-based fuzz testing of protocol decoders with random data, NaN/inf values, truncated frames. This exceeds most human-written hobby projects.

6. **Safe Serialization** — JSON-only persistence. No pickle, no unsafe YAML, no eval-based deserialization anywhere in the codebase.

---

## RECOMMENDATIONS

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Fix silent exception swallowing in `recording.py` | 15 min |
| 2 | Add `gitleaks` to CI pipeline | 30 min |
| 3 | Generate dependency lock file (`pip-compile` or Poetry) | 30 min |
| 4 | Pin PyQt6 minimum version in `pyproject.toml` | 5 min |
| 5 | Add assertions to GUI tests | 2-4 hrs |
| 6 | Narrow broad `except Exception` to specific types in DSP code | 2-4 hrs |
| 7 | Add sigstore signing + SBOM to release pipeline | 1-2 hrs |

---

## CONCLUSION

This SDR codebase is a well-structured desktop application with **no critical or high-severity security findings**. The project's security posture is strong for its context (alpha hobbyist SDR framework):

- **No credentials to leak** — the application has no secrets, API keys, or authentication
- **No network attack surface** — purely local desktop + USB hardware
- **No agentic risks** — no AI runtime, no plugin system, no dynamic code execution
- **Genuine safety engineering** — TX frequency lockouts are a standout feature

The primary risk is **provenance-based**: 63% AI-authored code warrants careful human review of safety-critical paths (particularly `core/frequency_manager.py` TX validation logic) before any production or regulatory use. For its declared purpose as an alpha hobbyist tool, the security posture is above average.

---

*Audit performed using [Agentic Security Audit v3.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-check.md) methodology, aligned with OWASP Top 10 for Agentic Applications (2026).*
