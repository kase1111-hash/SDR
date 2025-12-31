"""
Entry point for running the SDR GUI application.

Usage:
    python -m sdr_module.gui

This module can also be run directly:
    python -m sdr_module.gui --demo    # Run in demo mode without hardware
    python -m sdr_module.gui --help    # Show help
"""

import argparse
import logging
import sys

from .app import SDRApplication


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SDR Module GUI Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m sdr_module.gui              # Normal mode
    python -m sdr_module.gui --demo       # Demo mode with synthetic signals
    python -m sdr_module.gui -v           # Verbose logging
    python -m sdr_module.gui -f 144.8e6   # Start at specific frequency
        """,
    )

    parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        help="Run in demo mode with synthetic signals (no hardware required)",
    )

    parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=100e6,
        help="Initial frequency in Hz (default: 100 MHz)",
    )

    parser.add_argument(
        "--sample-rate",
        "-s",
        type=float,
        default=2.4e6,
        help="Sample rate in Hz (default: 2.4 MHz)",
    )

    parser.add_argument(
        "--gain", "-g", type=float, default=20.0, help="RF gain in dB (default: 20)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv, -vvv)",
    )

    parser.add_argument(
        "--log-file", type=str, default=None, help="Log to file instead of console"
    )

    return parser.parse_args()


def setup_logging(verbosity: int, log_file: str = None):
    """Setup logging based on verbosity level."""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG, logging.DEBUG]
    level = levels[min(verbosity, len(levels) - 1)]

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file:
        logging.basicConfig(
            level=level, format=format_str, filename=log_file, filemode="w"
        )
    else:
        logging.basicConfig(level=level, format=format_str)

    # Set third-party loggers to WARNING
    logging.getLogger("PyQt6").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose, args.log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting SDR Module GUI")

    # Create application
    app = SDRApplication()

    if not app.is_available():
        print("Error: PyQt6 is required for the GUI application.")
        print("Install it with: pip install PyQt6")
        sys.exit(1)

    # Configure initial settings
    settings = {
        "frequency": args.frequency,
        "sample_rate": args.sample_rate,
        "gain": args.gain,
        "demo_mode": args.demo,
    }

    # Run application
    result = app.run(settings)

    logger.info(f"Application exited with code {result}")
    sys.exit(result)


if __name__ == "__main__":
    main()
