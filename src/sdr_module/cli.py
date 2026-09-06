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
        print('Install it with: python -m pip install "sdr-module[gui]"')
        return 1

    app = SDRApplication()

    if not app.is_available():
        print("Error: PyQt6 is required for the GUI.")
        print('Install it with: python -m pip install "sdr-module[gui]"')
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
    print("  - Frequency scanning and signal detection (live or from a recording)")
    print("  - Spectrum analysis and visualization")
    print("  - Protocol decoding (POCSAG, FLEX, AX.25/APRS, RDS, ADS-B, ACARS)")
    print("  - Text encoding (RTTY, Morse, PSK31, ASCII)")
    print("  - I/Q recording and playback")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a frequency range (live) or analyze a recorded I/Q file (offline)."""
    if args.input:
        return _scan_file(args)

    # Live sweep: requires hardware. Validate the config so bad ranges are
    # caught even without a device attached.
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
    print("Note: A live sweep requires connected SDR hardware.")
    print("To analyze a recorded capture offline, pass --input FILE.")

    return 0


def _scan_file(args: argparse.Namespace) -> int:
    """Detect signals in a recorded I/Q file via FFT peak detection."""
    from .dsp.recording import SampleFormat, load_iq_file
    from .dsp.spectrum import SpectrumAnalyzer

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return 1

    sample_format = SampleFormat(args.format) if args.format else None
    samples, metadata = load_iq_file(
        input_path, sample_format=sample_format, max_samples=args.max_samples
    )

    sample_rate = args.sample_rate or metadata.sample_rate
    if not sample_rate:
        print("Error: sample rate unknown for this file; specify --sample-rate (Hz).")
        return 1

    center_freq = args.center if args.center is not None else metadata.center_frequency

    if len(samples) < args.fft_size:
        print(
            f"Error: file has {len(samples)} samples, fewer than the FFT size "
            f"({args.fft_size}). Use a smaller --fft-size."
        )
        return 1

    analyzer = SpectrumAnalyzer(fft_size=args.fft_size)
    result = analyzer.compute_spectrum(
        samples, center_freq=center_freq, sample_rate=sample_rate
    )
    peaks = analyzer.find_peaks(
        result, threshold_db=args.threshold, min_distance_hz=args.step * 1e3
    )

    print(
        f"Analyzed {len(samples)} samples "
        f"({len(samples) / sample_rate:.3f}s at {sample_rate / 1e6:.3f} Msps)"
    )
    print(
        f"Center: {center_freq / 1e6:.4f} MHz, "
        f"RBW: {result.rbw:.1f} Hz, Threshold: {args.threshold} dB"
    )
    print()

    if not peaks:
        print("No signals above threshold.")
        return 0

    print(f"Detected {len(peaks)} signal(s):")
    for i, peak in enumerate(peaks):
        offset_khz = (peak.frequency - center_freq) / 1e3
        print(
            f"  [{i}] {peak.frequency / 1e6:11.4f} MHz "
            f"({offset_khz:+8.1f} kHz)  {peak.power_db:7.1f} dB"
        )

    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Decode a protocol from a recorded I/Q file."""
    from .dsp.protocols import (
        ProtocolType,
        create_protocol_decoder,
        demodulate_for_protocol,
    )
    from .dsp.recording import SampleFormat, load_iq_file

    protocol = ProtocolType(args.protocol)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return 1

    sample_format = SampleFormat(args.format) if args.format else None
    samples, metadata = load_iq_file(
        input_path, sample_format=sample_format, max_samples=args.max_samples
    )

    sample_rate = args.sample_rate or metadata.sample_rate
    if not sample_rate:
        print("Error: sample rate unknown for this file; specify --sample-rate (Hz).")
        return 1

    print(
        f"Loaded {len(samples)} samples "
        f"({len(samples) / sample_rate:.3f}s at {sample_rate / 1e6:.3f} Msps)"
    )

    try:
        decoder = create_protocol_decoder(protocol, sample_rate=sample_rate)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    baseband = demodulate_for_protocol(samples, protocol)
    if baseband is None:
        print(
            f"Error: decoding {protocol.value.upper()} from a raw I/Q capture "
            "is not supported yet (it needs subcarrier recovery)."
        )
        return 2
    messages = decoder.decode(baseband)

    print(f"Decoded {len(messages)} {args.protocol.upper()} message(s)")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg}")

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
        if output_path.suffix.lower() == ".wav":
            from .dsp.recording import FileFormat, SampleFormat, save_iq_file

            save_iq_file(
                output_path,
                samples,
                sample_rate=args.sample_rate,
                sample_format=SampleFormat.INT16,
                file_format=FileFormat.WAV,
            )
        else:
            samples.tofile(output_path)
        print(f"Saved to: {output_path}")

    return 0


def _load_saved_channels():
    """
    Read the GUI's saved channels (bookmarks) as `Channel` objects.

    Returns:
        Tuple of (channels, error_message). `error_message` is set when the
        store cannot be read (PyQt6 missing), in which case channels is empty.
    """
    from .core.chirp_csv import bookmarks_to_channels

    try:
        from .gui.settings_store import GuiSettings
    except ImportError:
        return [], "PyQt6 is required to read saved channels."

    try:
        settings = GuiSettings()
    except ImportError:
        return [], (
            "PyQt6 is required to read saved channels.\n"
            'Install it with: python -m pip install "sdr-module[gui]"'
        )
    return bookmarks_to_channels(settings.get_bookmarks()), None


def _print_channels(channels) -> None:
    """Print a channel table to stdout."""
    print(f"{'#':>4}  {'Frequency':>13}  {'Mode':<5} {'Tone':<6} {'Name'}")
    for i, ch in enumerate(channels):
        slot = ch.location if ch.location is not None else i
        shift = f" {ch.duplex}{abs(ch.offset_hz)/1e6:g}" if ch.duplex else ""
        print(
            f"{slot:>4}  {ch.frequency_hz/1e6:>9.4f} MHz  "
            f"{ch.mode:<5} {ch.tone_mode or '-':<6} {ch.name}{shift}"
        )


def cmd_channels_export(args: argparse.Namespace) -> int:
    """Export saved channels (or the built-in RX presets) to a CHIRP CSV."""
    from .core.chirp_csv import presets_to_channels, write_chirp_csv

    if args.presets:
        from .core.frequency_manager import RX_PRESETS

        channels = presets_to_channels(RX_PRESETS)
    else:
        channels, error = _load_saved_channels()
        if error:
            print(f"Error: {error}")
            return 1
        if not channels:
            print("No saved channels to export.")
            return 1

    output = Path(args.output).expanduser()
    count = write_chirp_csv(output, channels, renumber=True)
    print(f"Exported {count} channel(s) to {output}")
    print("This file opens directly in CHIRP (File → Open).")
    return 0


def cmd_channels_import(args: argparse.Namespace) -> int:
    """Import channels from a CHIRP CSV into the saved channel store."""
    from .core.chirp_csv import ChirpCsvError, channels_to_bookmarks, read_chirp_csv

    try:
        report = read_chirp_csv(Path(args.input).expanduser(), strict=not args.skip_bad)
    except ChirpCsvError as exc:
        print(f"Error: {exc}")
        return 1

    if not report.channels:
        print("No channels found in that file.")
        return 1

    for message in report.skipped:
        print(f"Warning: skipped {message}")

    try:
        from .gui.settings_store import GuiSettings

        settings = GuiSettings()
    except ImportError:
        print("Error: PyQt6 is required to write saved channels.")
        print('Install it with: python -m pip install "sdr-module[gui]"')
        return 1

    imported = channels_to_bookmarks(report.channels)
    if args.append:
        bookmarks = settings.get_bookmarks() + imported
    else:
        bookmarks = imported
    settings.set_bookmarks(bookmarks)
    settings.sync()

    action = "Appended" if args.append else "Imported"
    print(f"{action} {len(imported)} channel(s); {len(bookmarks)} saved in total.")
    return 0


def cmd_channels_list(args: argparse.Namespace) -> int:
    """List channels from a CHIRP CSV, or the saved channels."""
    from .core.chirp_csv import ChirpCsvError, read_chirp_csv

    if args.input:
        try:
            channels = read_chirp_csv(
                Path(args.input).expanduser(), strict=False
            ).channels
        except ChirpCsvError as exc:
            print(f"Error: {exc}")
            return 1
    else:
        channels, error = _load_saved_channels()
        if error:
            print(f"Error: {error}")
            return 1

    if not channels:
        print("No channels.")
        return 0

    _print_channels(channels)
    print()
    print(f"{len(channels)} channel(s)")
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
    scan_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Analyze a recorded I/Q file offline instead of a live sweep",
    )
    scan_parser.add_argument(
        "--format",
        choices=["cu8", "cs8", "cs16", "cf32", "cf64"],
        help="Sample format of --input file (auto-detect if omitted)",
    )
    scan_parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.0,
        help="Sample rate in Hz for --input (overrides file metadata)",
    )
    scan_parser.add_argument(
        "--center",
        type=float,
        default=None,
        help="Center frequency in Hz for --input (overrides file metadata)",
    )
    scan_parser.add_argument(
        "--fft-size",
        type=int,
        default=4096,
        help="FFT size for --input (default: 4096)",
    )
    scan_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to read from --input (default: all)",
    )
    scan_parser.set_defaults(func=cmd_scan)

    # Decode command
    decode_parser = subparsers.add_parser(
        "decode", help="Decode a protocol from a recorded I/Q file"
    )
    decode_parser.add_argument(
        "protocol",
        choices=["pocsag", "flex", "ax25", "aprs", "rds", "adsb", "acars"],
        help="Protocol to decode",
    )
    decode_parser.add_argument(
        "--input", "-i", type=str, required=True, help="Recorded I/Q file to decode"
    )
    decode_parser.add_argument(
        "--format",
        choices=["cu8", "cs8", "cs16", "cf32", "cf64"],
        help="Sample format of the input file (auto-detect if omitted)",
    )
    decode_parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.0,
        help="Sample rate in Hz (overrides file metadata)",
    )
    decode_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to read (default: all)",
    )
    decode_parser.set_defaults(func=cmd_decode)

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to protocol")
    encode_parser.add_argument(
        "protocol", choices=["rtty", "morse", "ascii", "psk31"], help="Protocol to use"
    )
    encode_parser.add_argument(
        "--text", "-t", type=str, help="Text to encode (reads stdin if not provided)"
    )
    encode_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (.wav writes a playable WAV; anything else raw cf32)",
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

    # Channels command (CHIRP-compatible CSV import/export)
    channels_parser = subparsers.add_parser(
        "channels",
        help="Import/export saved channels as CHIRP-compatible CSV",
        description=(
            "Move memory channels between this application and CHIRP. "
            "Files use CHIRP's generic CSV format, so they open directly in "
            "CHIRP and can be uploaded to a radio."
        ),
    )
    channels_sub = channels_parser.add_subparsers(dest="channels_command")
    channels_sub.required = True

    channels_export = channels_sub.add_parser(
        "export", help="Write saved channels to a CHIRP CSV file"
    )
    channels_export.add_argument(
        "output",
        nargs="?",
        default="channels.csv",
        help="Output CSV file (default: channels.csv)",
    )
    channels_export.add_argument(
        "--presets",
        action="store_true",
        help="Export the built-in RX presets instead of the saved channels",
    )
    channels_export.set_defaults(func=cmd_channels_export)

    channels_import = channels_sub.add_parser(
        "import", help="Load channels from a CHIRP CSV file"
    )
    channels_import.add_argument("input", help="CHIRP CSV file to read")
    channels_import.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing channels instead of replacing them",
    )
    channels_import.add_argument(
        "--skip-bad",
        action="store_true",
        help="Skip unreadable rows instead of aborting",
    )
    channels_import.set_defaults(func=cmd_channels_import)

    channels_list = channels_sub.add_parser(
        "list", help="Show channels in a CSV file, or the saved channels"
    )
    channels_list.add_argument(
        "input", nargs="?", help="CHIRP CSV file (default: the saved channels)"
    )
    channels_list.set_defaults(func=cmd_channels_list)

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
