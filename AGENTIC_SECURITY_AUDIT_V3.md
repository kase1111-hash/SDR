# Agentic Security Audit v3.0 — SDR Module

```
AUDIT METADATA
  Project:       SDR Module (sdr-module v0.1.0)
  Date:          2026-03-11
  Auditor:       claude-opus-4-6
  Commit:        b3e9a034593ee47578c75153eccfcf7faec55614
  Strictness:    STANDARD
  Context:       PROTOTYPE

PROVENANCE ASSESSMENT
  Vibe-Code Confidence:   77%
  Human Review Evidence:  MINIMAL

LAYER VERDICTS
  L1 Provenance:       WARN
  L2 Credentials:      PASS
  L3 Agent Boundaries: PASS (N/A — no agentic features)
  L4 Supply Chain:     WARN
  L5 Infrastructure:   PASS
```

---

## L1: PROVENANCE & TRUST ORIGIN

### 1.1 Vibe-Code Detection

| Indicator | Status | Evidence |
|-----------|--------|----------|
| No tests | **PASS** | 20 test files, 9,626 lines, 605 test functions |
| No security config | **WARN** | SECURITY.md exists; no `.env.example` (not needed); no CI security scanning |
| AI boilerplate | **FAIL** | Zero TODO/FIXME/HACK markers across 57 source files; uniform docstring formatting; identical private attribute naming (`_config`, `_state`, `_lock`) across all modules |
| Rapid commit history | **FAIL** | 95 commits by "Claude" (noreply@anthropic.com) vs. 50 human commits (65% AI authorship); 50 commits on day 1, 37 on day 2 — entire 28k-line framework generated in bulk |
| Polished README, hollow codebase | **WARN** | README lists P25, DMR, TETRA, LoRa, Zigbee, Z-Wave as supported protocols — none are implemented in source code; only UI color schemes exist for these |
| Bloated deps | **PASS** | Minimalist: only `numpy` as hard dependency; optional groups for hardware drivers and dev tools |

**Severity: WARN** — Multiple vibe-code indicators present but no credentials/PII/payments handled. Codebase is functional (real DSP algorithms, working hardware bindings, complete call chains) despite AI generation.

### 1.2 Human Review Evidence

- [x] Security-focused commits exist: `ee3ec25` "Fix high severity issues: validation, TX safety, path traversal", `318a420` "Fix critical and high severity issues from audit"
- [ ] No security tooling in CI/CD — no bandit, semgrep, CodeQL, or OWASP dependency scanning
- [x] `.gitignore` properly excludes `.env`, `venv/`, `.vscode/`, `.idea/`, IDE temp files

**Evidence level: MINIMAL** — Security fixes are audit-driven remediation (generate → audit → fix), not proactive prevention. Zero security-related git history before audit phase.

### 1.3 The "Tech Preview" Trap

- [ ] No production traffic — desktop application, local-only SDR tool
- [ ] No real credentials handled — pure signal processing
- [ ] Alpha label (v0.1.0) is consistent with project maturity

**Status: NOT APPLICABLE** — Prototype with no production exposure.

---

## L2: CREDENTIAL & SECRET HYGIENE

### 2.1 Secret Storage

- [x] **No plaintext credentials** in files, DB schemas, config, or env files
- [x] **No API keys** in client-side code (desktop app, no client/server split)
- [x] **No secrets in git history** — comprehensive search of patterns (`api_key`, `password`, `secret`, `token`, `Bearer`, `sk_`, `AKIA`) returned zero results
- [x] **No `.env` files committed** — properly excluded via `.gitignore`
- [x] **No database connection strings** — no database usage at all

### 2.2 Credential Scoping & Lifecycle

**NOT APPLICABLE** — Project has no credential management. All configuration is device settings (frequencies, sample rates, gains) stored in `~/.config/sdr_module/config.json`.

### 2.3 Machine Credential Exposure

**NOT APPLICABLE** — No OAuth, no API keys, no cloud services.

**Layer Verdict: PASS**

---

## L3: AGENT BOUNDARY ENFORCEMENT

**NOT APPLICABLE** — This is a desktop SDR application with no agentic features, no LLM integration, no plugin execution, no agent-to-agent communication, and no MCP servers.

The following items were verified as absent:
- No MCP configuration files (`.mcp.json`, `mcp.yaml`)
- No `eval()`, `exec()`, or dynamic code execution
- No plugin loading system in current codebase (was removed in Phase 2 cleanup)
- No LLM or AI model integration
- No network-facing API endpoints

**Layer Verdict: PASS (N/A)**

---

## L4: SUPPLY CHAIN & DEPENDENCY TRUST

### 4.1 Plugin/Skill Supply Chain

**NOT APPLICABLE** — No plugin system in current codebase.

### 4.2 MCP Server Trust

**NOT APPLICABLE** — No MCP servers.

### 4.3 Dependency Audit

```
[HIGH] — Floating Dependency Versions
Layer:     4
Location:  pyproject.toml:30-53
Evidence:  All dependencies use >= pins: numpy>=1.21.0, pyrtlsdr>=0.2.92,
           hackrf>=1.0.0, scipy>=1.7.0, matplotlib>=3.4.0, pytest>=7.0.0
           No lockfile (poetry.lock, requirements.lock, Pipfile.lock) exists.
Risk:      Undetected breaking changes in transitive dependencies. For an SDR
           application, numpy/scipy version differences can alter FFT/DSP
           numerical results. Vulnerable dependency versions could be pulled
           at install time without alerting maintainers.
Fix:       Add pip-compile generated requirements.lock or adopt poetry with
           poetry.lock. Pin transitive dependencies for reproducible builds.
```

```
[MEDIUM] — Unpinned GitHub Actions
Layer:     4
Location:  .github/workflows/ci.yml, .github/workflows/release.yml
Evidence:  All actions use major version tags: actions/checkout@v6,
           actions/setup-python@v6, codecov/codecov-action@v5,
           pypa/gh-action-pypi-publish@release/v1
Risk:      GitHub Actions can be compromised. Tag v6 can point to different
           commits across runs. PyPI publish action is highest risk — controls
           package publication credentials.
Fix:       Pin all uses: statements to full commit SHAs.
           Example: actions/checkout@5a4ac9002d0be2fb38bd78470171d3ada0e5231b
```

```
[MEDIUM] — No Dependency Vulnerability Scanning
Layer:     4
Location:  .github/workflows/ci.yml
Evidence:  CI runs lint (black, isort, ruff), typecheck (mypy), and tests
           (pytest) but no pip-audit, safety, or OWASP dependency-check.
Risk:      Known vulnerabilities in dependencies (numpy, scipy, PyQt6) would
           go undetected until manual review.
Fix:       Add pip-audit or safety to CI pipeline:
           pip install pip-audit && pip-audit
```

**Layer Verdict: WARN**

---

## L5: INFRASTRUCTURE & RUNTIME

### 5.1 Database Security

**NOT APPLICABLE** — No database usage. Zero imports of sqlite3, psycopg, pymongo, sqlalchemy, or any ORM.

### 5.2 BaaS Configuration

**NOT APPLICABLE** — Desktop application, no backend-as-a-service.

### 5.3 Network & Hosting

**NOT APPLICABLE** — No network requests (zero imports of requests, urllib, aiohttp). No CORS. No API endpoints. Pure local SDR hardware interaction.

### 5.4 Deployment Pipeline

```
[LOW] — Permissive Type Checking in CI
Layer:     5
Location:  .github/workflows/ci.yml (typecheck job)
Evidence:  mypy runs with --ignore-missing-imports --no-error-summary || true
           The || true means type errors never fail CI.
Risk:      Type errors (potential runtime crashes) pass CI undetected.
Fix:       Remove || true. Fix mypy errors or add targeted per-file overrides.
```

```
[LOW] — No Security Static Analysis in CI
Layer:     5
Location:  .github/workflows/ci.yml
Evidence:  CI has lint, typecheck, test, build jobs but no security scanning.
           No bandit (Python security linter), no semgrep, no CodeQL.
Risk:      Security anti-patterns (hardcoded secrets, SQL injection, command
           injection, path traversal) would not be caught automatically.
Fix:       Add bandit to lint job: pip install bandit && bandit -r src/
```

### 5.5 Regulatory Compliance

**Positive findings:**
- TX frequency lockouts are hard-coded and cannot be disabled (`core/frequency_manager.py`)
  - GPS/GNSS bands protected (1575.42, 1227.60, 1176.45 MHz)
  - Aviation frequencies locked (121.5, 243.0 MHz)
  - ADS-B/Mode S guarded (1030, 1090 MHz)
  - Emergency beacons blocked (406.0–406.1 MHz)
  - Marine distress protected (156.8 MHz)
  - Cellular bands locked (698–806, 824–894, 1850–1995 MHz)
- Ham license enforcement with Technician/General/Extra class restrictions
- QRP power limits enforced (5W CW, 10W SSB)
- FCC Part 15, CE Mark, ISED, TELEC, ACMA compliance documented

**Layer Verdict: PASS**

---

## ADDITIONAL FINDINGS

```
[MEDIUM] — Silent Exception Swallowing in Hardware Drivers
Layer:     5
Location:  src/sdr_module/devices/hackrf.py:384-385, 472-473
Evidence:  stop_rx() and stop_tx() catch Exception without logging.
           Bare except clauses hide hardware failures.
Risk:      Hardware errors during TX/RX stop could leave device in undefined
           state. Silent failures prevent debugging of intermittent issues.
Fix:       Add logger.warning() to exception handlers. Log device state on
           failure for post-incident analysis.
```

```
[MEDIUM] — README Over-Claims Unimplemented Protocols
Layer:     1
Location:  README.md (protocol support table)
Evidence:  README lists P25, DMR, TETRA, LoRa, Zigbee, Z-Wave as supported.
           Source code contains only UI color schemes for these — zero decoder
           implementation. Verified: no P25/DMR/TETRA/LoRa/Zigbee/Z-Wave
           classes or functions exist outside gui/waterfall theming.
Risk:      Users may rely on claimed protocol support that doesn't exist.
           For security-adjacent protocols (P25 used by law enforcement),
           false capabilities could lead to misplaced trust.
Fix:       Mark unimplemented protocols as "Planned" or remove from README.
```

```
[LOW] — Bare Exception Handlers in DSP Code
Layer:     5
Location:  src/sdr_module/dsp/recording.py:2490, 2682, 2695
Evidence:  except Exception: without as e — exceptions caught and silently
           discarded without any logging or handling.
Risk:      DSP processing errors (corrupted I/Q data, malformed recordings)
           silently produce incorrect results instead of failing visibly.
Fix:       Replace bare except with specific exception types. Log warnings
           for unexpected conditions.
```

---

## FINDING SUMMARY

| # | Severity | Title | Layer |
|---|----------|-------|-------|
| 1 | HIGH | Floating dependency versions, no lockfile | L4 |
| 2 | MEDIUM | Unpinned GitHub Actions (including PyPI publish) | L4 |
| 3 | MEDIUM | No dependency vulnerability scanning in CI | L4 |
| 4 | MEDIUM | Silent exception swallowing in HackRF drivers | L5 |
| 5 | MEDIUM | README over-claims unimplemented protocols | L1 |
| 6 | LOW | Permissive type checking (mypy \|\| true) | L5 |
| 7 | LOW | No security static analysis (bandit/semgrep) in CI | L5 |
| 8 | LOW | Bare exception handlers in DSP recording code | L5 |

**Critical findings: 0**
**High findings: 1**
**Medium findings: 4**
**Low findings: 3**

---

## PROVENANCE DEEP-DIVE

### Vibe-Code Confidence: 77%

**Evidence supporting AI generation:**

| Signal | Detail |
|--------|--------|
| Commit authorship | 95/145 commits (65%) by "Claude" (noreply@anthropic.com) |
| Velocity | 28,607 lines of source in 2 days (50 + 37 commits) |
| Zero organic markers | No TODO, FIXME, HACK, NOTE, WIP, or revert commits in entire history |
| Uniform structure | Identical docstring format, private attribute naming, method signatures across 57 files |
| Audit-then-fix pattern | All security improvements are reactive to audits (Phase 1→4 remediation cycle) |

**Evidence of genuine engineering substance:**

| Signal | Detail |
|--------|--------|
| Working DSP | Real FFT, demodulation, filter design — not stubs |
| Hardware integration | Functional pyrtlsdr/hackrf bindings with proper gain staging |
| Domain comments | `frequency_manager.py:13` explains 150% power headroom for TX chain losses — authentic RF knowledge |
| Thread safety | RLock protections, thread-safe sample buffers with proper locking |
| TX safety | Hard-coded frequency lockouts that cannot be bypassed — genuine safety engineering |
| Meaningful tests | Test suite generates synthetic BPSK/FSK/tone signals and validates bit recovery |

**Assessment:** This is AI-generated code with meaningful human architectural direction. The AI did the heavy lifting on implementation, but domain expertise guided the design. The development methodology is "generate → audit → fix → repeat" rather than incremental human development.

---

## RECOMMENDATIONS (Priority Order)

### Immediate
1. **Pin dependencies** — Add `pip-compile` lockfile for reproducible builds
2. **Pin GitHub Actions to SHAs** — Especially `pypa/gh-action-pypi-publish`

### Short-term
3. **Add `pip-audit` to CI** — Catch known vulnerabilities in dependencies
4. **Add `bandit` to CI** — Catch Python security anti-patterns
5. **Fix HackRF exception handlers** — Log hardware errors instead of swallowing
6. **Correct README protocol claims** — Mark P25/DMR/TETRA/LoRa/Zigbee/Z-Wave as "Planned"

### Medium-term
7. **Make mypy CI-blocking** — Remove `|| true` and fix type errors
8. **Add property-based testing** — hypothesis is in dev deps but unused for DSP validation
9. **Replace bare `except Exception:` blocks** — Use specific exception types

---

*Audit performed using the [Agentic Security Audit v3.0](https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-check.md) framework, aligned with OWASP Top 10 for Agentic Applications (2026).*
