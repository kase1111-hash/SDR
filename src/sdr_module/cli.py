#!/usr/bin/env python3
"""
SDR Module - Command Line Interface

Main entry point for the SDR Module application.
Provides access to scanning, encoding, and signal analysis tools.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__


def cmd_gui(args: argparse.Namespace) -> int:
    """Launch the graphical user interface."""
    try:
        from .gui.app import SDRApplication
    except ImportError:
        print("Error: GUI module not available.")
        print("Install PyQt6 with: pip install PyQt6")
        return 1

    app = SDRApplication()

    if not app.is_available():
        print("Error: PyQt6 is required for the GUI.")
        print("Install with: pip install PyQt6")
        return 1

    settings = {
        "demo_mode": args.demo,
        "frequency": args.frequency,
        "gain": args.gain,
    }

    return app.run(settings)


def cmd_info(args: argparse.Namespace) -> int:
    """Display module information."""
    print(f"SDR Module v{__version__}")
    print()
    print("Software Defined Radio Framework")
    print("================================")
    print()
    print("Supported Hardware:")
    print("  - RTL-SDR (RX only): 500 kHz - 1.7 GHz")
    print("  - HackRF One (TX/RX): 1 MHz - 6 GHz")
    print()
    print("Features:")
    print("  - Frequency scanning and signal detection")
    print("  - Spectrum analysis and visualization")
    print("  - Signal classification and protocol detection")
    print("  - Text encoding (RTTY, Morse, PSK31, ASCII)")
    print("  - I/Q recording and playback")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Run frequency scanner."""
    from .dsp.scanner import FrequencyScanner, ScanConfig, ScanMode

    config = ScanConfig(
        start_freq_hz=args.start * 1e6,
        end_freq_hz=args.end * 1e6,
        step_hz=args.step * 1e3,
        threshold_db=args.threshold,
        mode=ScanMode.SINGLE if args.single else ScanMode.CONTINUOUS,
    )

    FrequencyScanner(config)

    print(f"Scanning {args.start:.3f} - {args.end:.3f} MHz")
    print(f"Step: {args.step} kHz, Threshold: {args.threshold} dB")
    print()
    print("Note: This requires connected SDR hardware.")
    print("Use with RTL-SDR or HackRF One for actual scanning.")

    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    """Encode text to protocol."""
    from .protocols.encoder import EncoderConfig, ModulationType
    from .protocols.encoders import (
        ASCIIEncoder,
        MorseEncoder,
        PSK31Encoder,
        RTTYEncoder,
    )

    encoders = {
        "rtty": RTTYEncoder,
        "morse": MorseEncoder,
        "ascii": ASCIIEncoder,
        "psk31": PSK31Encoder,
    }

    if args.protocol not in encoders:
        print(f"Unknown protocol: {args.protocol}")
        print(f"Available: {', '.join(encoders.keys())}")
        return 1

    config = EncoderConfig(
        sample_rate=args.sample_rate,
        carrier_freq=args.carrier,
        baud_rate=300,
        modulation=ModulationType.FSK,
        amplitude=0.8,
        frequency_shift=1000,
    )

    encoder_class = encoders[args.protocol]
    if args.protocol == "morse":
        encoder = encoder_class(config, wpm=args.wpm)
    else:
        encoder = encoder_class(config)

    text = args.text or sys.stdin.read().strip()
    if not text:
        print("No text provided")
        return 1

    samples = encoder.encode_text(text)

    print(f"Encoded {len(text)} characters using {args.protocol.upper()}")
    print(f"Generated {len(samples)} samples ({len(samples)/args.sample_rate:.3f}s)")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            print(f"Warning: Overwriting existing file: {output_path}")
        samples.tofile(output_path)
        print(f"Saved to: {output_path}")

    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    """List available SDR devices."""
    from .core.device_manager import DeviceManager

    print("Scanning for SDR devices...")
    print()

    manager = DeviceManager()
    devices = manager.scan_devices()

    if not devices:
        print("No SDR devices found.")
        print()
        print("Supported devices:")
        print("  - RTL-SDR (rtl-sdr.com)")
        print("  - HackRF One (greatscottgadgets.com)")
        return 0

    print(f"Found {len(devices)} device(s):")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="sdr-scan",
        description="SDR Module - Software Defined Radio Framework",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display module information")
    info_parser.set_defaults(func=cmd_info)

    # Devices command
    devices_parser = subparsers.add_parser("devices", help="List available SDR devices")
    devices_parser.set_defaults(func=cmd_devices)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan frequency range")
    scan_parser.add_argument(
        "--start",
        type=float,
        default=88.0,
        help="Start frequency in MHz (default: 88.0)",
    )
    scan_parser.add_argument(
        "--end", type=float, default=108.0, help="End frequency in MHz (default: 108.0)"
    )
    scan_parser.add_argument(
        "--step", type=float, default=100.0, help="Step size in kHz (default: 100.0)"
    )
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=-60.0,
        help="Detection threshold in dB (default: -60.0)",
    )
    scan_parser.add_argument(
        "--single", action="store_true", help="Single scan (default: continuous)"
    )
    scan_parser.set_defaults(func=cmd_scan)

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to protocol")
    encode_parser.add_argument(
        "protocol", choices=["rtty", "morse", "ascii", "psk31"], help="Protocol to use"
    )
    encode_parser.add_argument(
        "--text", "-t", type=str, help="Text to encode (reads stdin if not provided)"
    )
    encode_parser.add_argument(
        "--output", "-o", type=str, help="Output file for I/Q samples"
    )
    encode_parser.add_argument(
        "--sample-rate",
        type=float,
        default=48000,
        help="Sample rate in Hz (default: 48000)",
    )
    encode_parser.add_argument(
        "--carrier",
        type=float,
        default=1000,
        help="Carrier frequency in Hz (default: 1000)",
    )
    encode_parser.add_argument(
        "--wpm", type=int, default=20, help="Words per minute for Morse (default: 20)"
    )
    encode_parser.set_defaults(func=cmd_encode)

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Launch graphical user interface")
    gui_parser.add_argument(
        "--demo",
        "-d",
        action="store_true",
        help="Run in demo mode (no hardware required)",
    )
    gui_parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=100e6,
        help="Initial frequency in Hz (default: 100 MHz)",
    )
    gui_parser.add_argument(
        "--gain", "-g", type=float, default=20.0, help="RF gain in dB (default: 20)"
    )
    gui_parser.set_defaults(func=cmd_gui)

    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point.

    Args:
        argv: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # No command specified - show info
        return cmd_info(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
