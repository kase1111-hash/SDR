"""PyInstaller entry point for the ``sdr-scan`` executable (see sdr_module.spec).

Kept outside the package on purpose: PyInstaller runs the entry script as
``__main__``, which would break the relative imports inside ``sdr_module``.
"""

import sys

from sdr_module.cli import main

if __name__ == "__main__":
    sys.exit(main())
