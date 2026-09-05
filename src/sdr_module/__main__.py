"""Run the ``sdr-scan`` command line via ``python -m sdr_module``."""

import sys

from sdr_module.cli import main

if __name__ == "__main__":
    sys.exit(main())
