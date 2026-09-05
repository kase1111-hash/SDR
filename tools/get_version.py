#!/usr/bin/env python3
"""Print the sdr_module version without importing the package.

Used by the build scripts and the release workflow so the version is read
from the single source of truth (``src/sdr_module/__init__.py``) even when
NumPy is not installed.
"""

import re
import sys
from pathlib import Path

INIT_FILE = (
    Path(__file__).resolve().parent.parent / "src" / "sdr_module" / "__init__.py"
)


def main() -> int:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print(f"error: no __version__ found in {INIT_FILE}", file=sys.stderr)
        return 1
    print(match.group(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
