#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs the project (with dev extras) and the bundled sdr-antenna-array
# package so tests, linters and type checks are ready with no manual steps.
#
# IMPORTANT: always install through `python -m pip`. In some environments the
# bare `pip` on PATH belongs to a different interpreter than `python`, which
# silently installs packages where `python` cannot import them (the classic
# "ModuleNotFoundError: No module named 'numpy'" after a successful install).
# Routing through `python -m pip` guarantees the two always agree.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

PY="$(command -v python3 || command -v python)"

echo "Using interpreter: $PY ($("$PY" --version 2>&1))"

# Best-effort pip upgrade; the system pip may be distro-managed and refuse to
# uninstall itself, which is fine — installs work with the existing version.
"$PY" -m pip install --upgrade pip || echo "pip upgrade skipped (distro-managed)"

# Main package + dev tooling (pytest, ruff, black, isort, mypy, hypothesis).
"$PY" -m pip install -e ".[dev]"

# Bundled antenna-array sub-package (separate distribution under packages/).
"$PY" -m pip install -e "packages/sdr-antenna-array"

echo "session-start: dependencies installed."
