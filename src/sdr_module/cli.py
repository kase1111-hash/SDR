#!/usr/bin/env python3
"""
SDR Module - Command Line Interface

Main entry point for the SDR Module application.
Provides access to scanning, encoding, and signal analysis tools.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
        "sample_rate": args.sample_rate,
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
    print("  - Protocol decoding (POCSAG, FLEX, AX.25/APRS, RDS, ADS-B, ACARS, CW)")
    print("  - Text encoding (RTTY, Morse, PSK31, ASCII)")
    print("  - I/Q recording and playback")
    return 0


#: ``--threshold`` not given: auto for a live scan, -60 dBFS for --input.
_DEFAULT_THRESHOLD: Any = object()
_FILE_THRESHOLD_DB = -60.0


def _threshold_arg(text: str) -> Optional[float]:
    """``--threshold``: a level in dBFS, or "auto" (None)."""
    if str(text).strip().lower() == "auto":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a level in dBFS or 'auto', not {text!r}"
        ) from exc


def _rates_arg(text: str) -> Tuple[float, ...]:
    """``--rate``: a sample rate in Hz, or two separated by a comma (one
    for each of --dual's receivers)."""
    parts = [part.strip() for part in str(text).split(",")]
    try:
        rates = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a sample rate in Hz, or two separated by a comma, not {text!r}"
        ) from exc
    if not 1 <= len(rates) <= 2 or not all(math.isfinite(r) and r > 0 for r in rates):
        raise argparse.ArgumentTypeError(
            f"expected a sample rate in Hz, or two separated by a comma, not {text!r}"
        )
    return rates


# Live sweep timing: after each retune, queued samples are thrown away and
# the tuner gets this long to settle (an RTL-SDR's transfer in flight, 0.11 s
# at 2.4 MS/s, still holds the old frequency); then at least this many
# samples are averaged, waiting at most this long for them.
_LIVE_SETTLE_S = 0.12
_LIVE_MEASURE_SAMPLES = 65536
_LIVE_READ_TIMEOUT_S = 1.0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a frequency range (live) or analyze a recorded I/Q file (offline)."""
    if args.input:
        return _scan_file(args)
    if getattr(args, "dual", False):
        return _scan_dual(args)
    if len(_rates(args)) > 1:
        print(
            "Error: two rates are for --dual's two receivers; give one --rate "
            "to scan with one."
        )
        return 1
    return _scan_live(args)


def _rates(args: argparse.Namespace) -> Tuple[float, ...]:
    """``--rate`` as a tuple (one rate, or one per --dual receiver)."""
    rate = args.rate
    return tuple(rate) if isinstance(rate, (tuple, list)) else (float(rate),)


def _applied_rate(device, requested: float) -> float:
    """The sample rate a device runs at after ``set_sample_rate`` (a driver
    clamps a rate it can't do into its range), else ``requested``."""
    import math

    state = getattr(device, "state", None)
    for value in (
        getattr(state, "sample_rate", None),
        getattr(device, "sample_rate", None),
    ):
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
    return float(requested)


def _drain(device) -> None:
    """Throw away the samples a device has queued."""
    for _ in range(1000):
        try:
            block = device.read_samples(4096, timeout=0.0)
        except Exception:  # pragma: no cover - driver specific
            return
        if block is None or len(block) == 0:
            return


def _open_first_receiver(args: argparse.Namespace):
    """``(manager, device, name)`` for the first SDR found (of
    ``--device``'s type), or ``(manager, None, error message)``."""
    manager, opened, error = _open_receivers(args, 1)
    if error:
        return manager, None, error
    device, name = opened[0]
    return manager, device, name


def _open_receivers(args: argparse.Namespace, count: int):
    """``(manager, [(device, name)], "")`` for the first ``count`` SDRs
    found (of ``--device``'s type), or ``(manager, [], error message)``
    (none of them left open)."""
    from .core.device_manager import DeviceManager

    manager = DeviceManager()
    detected = list(manager.scan_devices())
    types = {cls: key for key, cls in manager.DEVICE_TYPES.items()}
    if args.device:
        detected = [d for d in detected if types.get(d.device_class) == args.device]
    detected = [d for d in detected if d.is_available and d.device_class in types]
    which = {"rtlsdr": "RTL-SDR", "hackrf": "HackRF One"}.get(
        args.device or "", "SDR device"
    )
    if not detected:
        return (
            manager,
            [],
            (
                f"No {which} is connected, so there is nothing to scan with. Plug "
                "one in ('sdr-scan devices' lists what is found), or analyze a "
                "recording instead with --input FILE."
            ),
        )

    def name_of(found) -> str:
        return str(getattr(found.info, "name", "") or types[found.device_class])

    if len(detected) < count:
        kind = f" of that kind ({which})" if args.device else ""
        return (
            manager,
            [],
            (
                f"--dual needs two connected SDRs{kind}, but only "
                f"{name_of(detected[0])} is connected. Plug in another "
                "('sdr-scan devices' lists what is found), or scan with this "
                "one alone (without --dual)."
            ),
        )
    opened: List[Tuple[Any, str]] = []
    for found in detected[:count]:
        name = name_of(found)
        device = manager.open_device(types[found.device_class], found.info.index)
        if device is None:
            if opened:
                manager.close_all()
            return (
                manager,
                [],
                f"Could not open {name}. Check that no other program is using it.",
            )
        opened.append((device, name))
    return manager, opened, ""


def _scan_live(args: argparse.Namespace) -> int:
    """Sweep a connected receiver across the range and list the signals.

    The receiver steps by about 80 % of its sample rate; at each step the
    queued samples are dropped, the tuner settles, about 64k samples are
    averaged into a spectrum and every channel of that step is checked
    (see :mod:`sdr_module.dsp.sweep`). It sweeps once with ``--single``,
    else until Ctrl+C.
    """
    import time

    import numpy as np

    from .core.band_plan import band_for
    from .dsp.sweep import (
        AUTO_MARGIN_DB,
        SpectrumAverager,
        format_duration,
        measure_page,
        plan_sweep,
        same_signal,
    )

    start_hz, end_hz, step_hz = args.start * 1e6, args.end * 1e6, args.step * 1e3
    if step_hz <= 0 or end_hz <= start_hz:
        print("Error: --end must be above --start, and --step above 0.")
        return 1

    manager, device, name = _open_first_receiver(args)
    if device is None:
        print(f"Error: {name}")
        return 1

    hits: List[Any] = []
    try:
        requested = _rates(args)[0]
        if device.set_sample_rate(requested) is False:
            print(f"Error: {name} can't run at {requested / 1e6:g} MS/s (--rate).")
            return 1
        # Planned and measured at the rate it runs at: the steps and the
        # frequencies found depend on it.
        rate = _applied_rate(device, requested)
        if abs(rate - requested) >= 1.0:
            print(
                f"Note: {name} can't run at {requested / 1e6:g} MS/s (--rate); "
                f"scanning at {rate / 1e6:g} MS/s, the nearest rate it can."
            )
        if args.gain is None:
            device.set_gain_mode(True)
        else:
            device.set_gain_mode(False)
            device.set_gain(float(args.gain))
        spec = getattr(device, "spec", None)
        low = float(getattr(spec, "freq_min", 0.0) or 0.0)
        high = float(getattr(spec, "freq_max", 0.0) or 0.0)
        if high > low:
            if end_hz < low or start_hz > high:
                print(
                    f"Error: {name} tunes {low / 1e6:.3f} - {high / 1e6:.3f} MHz; "
                    "choose a range inside it."
                )
                return 1
            if start_hz < low or end_hz > high:
                start_hz, end_hz = max(start_hz, low), min(end_hz, high)
                print(
                    f"Note: limited to {name}'s range, {start_hz / 1e6:.3f} - "
                    f"{end_hz / 1e6:.3f} MHz."
                )
        pages = plan_sweep(start_hz, end_hz, step_hz, rate)
        per_page = _LIVE_SETTLE_S + _LIVE_MEASURE_SAMPLES / rate + 0.02
        channels = sum(p.count for p in pages)
        threshold = None if args.threshold is _DEFAULT_THRESHOLD else args.threshold
        print(
            f"Scanning {start_hz / 1e6:.3f} - {end_hz / 1e6:.3f} MHz with {name} "
            f"({rate / 1e6:g} MS/s): {channels:,} channels of {args.step:g} kHz "
            f"in {len(pages):,} steps, {format_duration(len(pages) * per_page)} "
            "per sweep."
        )
        print(
            "Threshold: "
            + (
                f"auto (the noise floor + {AUTO_MARGIN_DB:.0f} dB)"
                if threshold is None
                else f"{threshold:.1f} dBFS"
            )
            + (". Press Ctrl+C to stop." if not args.single else ".")
        )
        if device.start_rx() is False:
            print(f"Error: {name} did not start streaming.")
            return 1
        print()
        print(f"  {'Frequency':>14}  {'Level':>11}  {'SNR':>7}  Band")
        sweep = 0
        while True:
            sweep += 1
            began = time.monotonic()
            new = 0
            measured = 0  # steps of this sweep that delivered samples
            for page in pages:
                if device.set_frequency(page.lo_hz) is False:
                    continue
                _drain(device)
                time.sleep(_LIVE_SETTLE_S)
                _drain(device)
                averager = SpectrumAverager(max_segments=64)
                taken = 0
                deadline = time.monotonic() + _LIVE_READ_TIMEOUT_S
                while taken < _LIVE_MEASURE_SAMPLES and time.monotonic() < deadline:
                    block = device.read_samples(
                        _LIVE_MEASURE_SAMPLES, timeout=_LIVE_READ_TIMEOUT_S
                    )
                    if block is None or len(block) == 0:
                        continue
                    averager.add(np.asarray(block))
                    taken += len(block)
                power = averager.power_db()
                if power is None:
                    continue
                measured += 1
                readings, _noise, _thr = measure_page(
                    power, page.lo_hz, rate, page.channels, step_hz, threshold
                )
                for reading in readings:
                    if not (
                        start_hz - step_hz / 2
                        <= reading.frequency_hz
                        <= end_hz + step_hz / 2
                    ):
                        continue
                    known = next(
                        (
                            h
                            for h in hits
                            if same_signal(
                                h.frequency_hz,
                                h.width_hz,
                                reading.frequency_hz,
                                step_hz,
                            )
                        ),
                        None,
                    )
                    if known is not None:
                        if reading.peak_db > known.peak_db:
                            hits[hits.index(known)] = reading
                        continue
                    hits.append(reading)
                    new += 1
                    band = band_for(reading.frequency_hz)
                    print(
                        f"  {reading.frequency_hz / 1e6:10.4f} MHz  "
                        f"{reading.peak_db:6.1f} dBFS  {reading.snr_db:4.0f} dB  "
                        f"{band.name if band else ''}".rstrip(),
                        flush=True,
                    )
            if not measured:
                print(
                    f"Error: {name} returned no samples. Check that it is plugged "
                    "in and not used by another program."
                )
                return 1
            print(
                f"Sweep {sweep} done in {time.monotonic() - began:.1f} s: "
                f"{len(hits)} signal(s), {new} new.",
                flush=True,
            )
            if args.single:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            device.stop_rx()
        except Exception as exc:  # pragma: no cover - driver specific
            print(f"Note: {name} did not stop cleanly: {exc}")
        manager.close_all()

    if not hits:
        print("No signals above the threshold.")
    return 0


def _mhz_span(low_hz: float, high_hz: float) -> str:
    return f"{low_hz / 1e6:.3f} - {high_hz / 1e6:.3f} MHz"


def _hit_line(hit: Any, name_width: int) -> str:
    """A signal found by ``sdr-scan scan --dual``: frequency, level, SNR,
    the receiver that measured it and the band."""
    from .core.band_plan import band_for

    band = band_for(hit.frequency_hz)
    return (
        f"  {hit.frequency_hz / 1e6:10.4f} MHz  {hit.peak_db:6.1f} dBFS  "
        f"{hit.snr_db:4.0f} dB  {hit.receiver_name:<{name_width}}  "
        f"{band.name if band else ''}"
    ).rstrip()


def _say_dual_pass(report: Any, names: List[str]) -> None:
    """``sdr-scan scan --dual``: what a sweep found, what each receiver
    swept, and what it couldn't."""
    if not report.complete:
        return

    def steps(count: int) -> str:
        return f"{count} step{'' if count == 1 else 's'}"

    parts = []
    for index, name in enumerate(names):
        spans = ", ".join(_mhz_span(low, high) for low, high in report.covered[index])
        if not spans:
            parts.append(f"{name} swept nothing")
            continue
        verb = "swept " if not parts else ""
        parts.append(f"{name} {verb}{spans} ({steps(report.steps[index])})")
    print(
        f"Sweep {report.number} done in {report.seconds:.1f} s: "
        f"{report.found} signal(s), {report.new} new. {', '.join(parts)}.",
        flush=True,
    )
    for index, name in enumerate(names):
        others = [n for i, n in enumerate(names) if i != index]
        if report.stopped[index]:
            print(
                f"Note: {name} {report.stopped[index]}; "
                f"{' and '.join(others) or 'nothing else'} swept what it could "
                "of its part.",
                flush=True,
            )
        elif report.failed and report.failed[index]:
            print(
                f"Note: {name} couldn't measure {steps(report.failed[index])}; "
                f"{' and '.join(others)} measured what it tunes of them.",
                flush=True,
            )
    if report.missed:
        spans = ", ".join(_mhz_span(low, high) for low, high in report.missed)
        print(f"Not swept: {spans}.", flush=True)


def _scan_dual(args: argparse.Namespace) -> int:
    """Sweep with two connected receivers at once (``--dual``).

    Each sweeps part of the range (see :mod:`sdr_module.core.wideband_scan`):
    what only it tunes, then the rest from its end until they meet, the
    faster taking more. The signals are listed as found, and once more in
    frequency order at the end (or at Ctrl+C), each once.
    """
    from .core.wideband_scan import CANCELLED, COMPLETE, WidebandScan, device_range
    from .dsp.sweep import AUTO_MARGIN_DB, format_duration

    start_hz, end_hz, step_hz = args.start * 1e6, args.end * 1e6, args.step * 1e3
    if step_hz <= 0 or end_hz <= start_hz:
        print("Error: --end must be above --start, and --step above 0.")
        return 1
    rates = _rates(args)
    manager, opened, error = _open_receivers(args, 2)
    if error:
        print(f"Error: {error}")
        return 1

    names = [name for _device, name in opened]
    width = max(len("Receiver"), *(len(name) for name in names))
    scan: Optional[Any] = None
    try:
        devices = []
        for index, (device, name) in enumerate(opened):
            requested = rates[min(index, len(rates) - 1)]
            if device.set_sample_rate(requested) is False:
                print(f"Error: {name} can't run at {requested / 1e6:g} MS/s (--rate).")
                return 1
            rate = _applied_rate(device, requested)
            if abs(rate - requested) >= 1.0:
                print(
                    f"Note: {name} can't run at {requested / 1e6:g} MS/s "
                    f"(--rate); scanning at {rate / 1e6:g} MS/s, the nearest rate "
                    "it can."
                )
            if args.gain is None:
                device.set_gain_mode(True)
            else:
                device.set_gain_mode(False)
                device.set_gain(float(args.gain))
            devices.append(device)
        spans = [device_range(device) for device in devices]
        low = min(span[0] for span in spans)
        high = max(span[1] for span in spans)
        tunes = ", ".join(
            (
                f"{name} tunes {_mhz_span(*span)}"
                if math.isfinite(span[1])
                else f"{name} tunes anything"
            )
            for name, span in zip(names, spans, strict=True)
        )
        if end_hz < low or start_hz > high:
            print(
                f"Error: neither receiver tunes {_mhz_span(start_hz, end_hz)} "
                f"({tunes}); choose a range inside what they tune."
            )
            return 1
        if start_hz < low or end_hz > high:
            start_hz, end_hz = max(start_hz, low), min(end_hz, high)
            print(
                f"Note: limited to what {' and '.join(names)} tune, "
                f"{_mhz_span(start_hz, end_hz)}."
            )
        threshold = None if args.threshold is _DEFAULT_THRESHOLD else args.threshold
        try:
            scan = WidebandScan(
                devices,
                start_hz,
                end_hz,
                step_hz,
                threshold_db=threshold,
                continuous=not args.single,
                settle_s=_LIVE_SETTLE_S,
                measure_samples=_LIVE_MEASURE_SAMPLES,
                read_timeout_s=_LIVE_READ_TIMEOUT_S,
                names=names,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        names = list(scan.names)
        sweep = scan.sweep
        print(
            f"Scanning {_mhz_span(start_hz, end_hz)} with "
            + " and ".join(
                f"{name} ({rate / 1e6:g} MS/s)"
                for name, rate in zip(names, scan.rates, strict=True)
            )
            + f" at once: {sweep.covered_count:,} channels of {args.step:g} kHz, "
            f"{format_duration(scan.estimate_seconds())} per sweep."
        )
        for index, (name, rate) in enumerate(zip(names, scan.rates, strict=True)):
            plan = []
            only = sweep.only_tuned_by(index)
            if only:
                spans_text = ", ".join(_mhz_span(a, b) for a, b in only)
                plan.append(f"{spans_text} (only it tunes those)")
            shared = sweep.shared()
            if shared:
                up = sweep.sweeps_up(index)
                edge = shared[0][0] if up else shared[-1][1]
                plan.append(f"from {edge / 1e6:.3f} MHz {'up' if up else 'down'}")
            print(f"  {name} ({rate / 1e6:g} MS/s): " + ", then ".join(plan))
        if sweep.shared():
            print("They meet where their speeds put them.")
        for low_hz, high_hz in sweep.uncovered:
            print(f"Note: neither receiver tunes {_mhz_span(low_hz, high_hz)}.")
        print(
            "Threshold: "
            + (
                f"auto (the noise floor + {AUTO_MARGIN_DB:.0f} dB)"
                if threshold is None
                else f"{threshold:.1f} dBFS"
            )
            + (". Press Ctrl+C to stop." if not args.single else ".")
        )
        print()
        print(
            f"  {'Frequency':>14}  {'Level':>11}  {'SNR':>7}  "
            f"{'Receiver':<{width}}  Band"
        )
        scan.on_found = lambda hit: print(_hit_line(hit, width), flush=True)
        scan.on_pass = lambda report: _say_dual_pass(report, names)
        scan.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        manager.close_all()

    if scan is None:
        return 0
    hits = scan.hits
    # (Not an error: it swept, or was stopped as asked.)
    failed = scan.reason not in (COMPLETE, CANCELLED)
    if hits:
        print()
        print(f"{len(hits)} signal(s) in frequency order:")
        print(
            f"  {'Frequency':>14}  {'Level':>11}  {'SNR':>7}  "
            f"{'Receiver':<{width}}  Band"
        )
        for hit in hits:
            print(_hit_line(hit, width))
    elif not failed:
        print()
        print("No signals above the threshold.")
    if failed:
        print(f"Error: {scan.reason}")
        return 1
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
    threshold = args.threshold
    if threshold is _DEFAULT_THRESHOLD:
        threshold = _FILE_THRESHOLD_DB
    elif threshold is None:  # auto: the noise floor plus the usual margin
        from .dsp.sweep import AUTO_MARGIN_DB, noise_floor_db

        threshold = round(noise_floor_db(result.power_db) + AUTO_MARGIN_DB, 1)
    peaks = analyzer.find_peaks(
        result, threshold_db=threshold, min_distance_hz=args.step * 1e3
    )

    print(
        f"Analyzed {len(samples)} samples "
        f"({len(samples) / sample_rate:.3f}s at {sample_rate / 1e6:.3f} Msps)"
    )
    print(
        f"Center: {center_freq / 1e6:.4f} MHz, "
        f"RBW: {result.rbw:.1f} Hz, Threshold: {threshold} dB"
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
    """Decode a protocol from a recorded I/Q file.

    Uses the same streaming live decoders as the window's Decoder panel:
    the channel at ``--offset`` from the recording's centre is picked out,
    demodulated and decoded in one-second blocks. Only messages that pass
    their checks are counted; ``--show-invalid`` lists the others too.
    """
    from .dsp.live_decoders import (
        create_live_decoder,
        describe_message,
        format_rate,
        live_decoder_class,
    )
    from .dsp.recording import SampleFormat, load_iq_file

    name = args.protocol.upper()
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

    decoder_class = live_decoder_class(args.protocol)
    if decoder_class is None:
        print(
            f"Error: decoding {name} from a raw I/Q capture is not supported "
            "yet (it needs subcarrier recovery)."
        )
        return 2
    if sample_rate < decoder_class.MIN_SAMPLE_RATE:
        print(
            f"Error: the sample rate is too low for {decoder_class.name}: it "
            f"needs at least {format_rate(decoder_class.MIN_SAMPLE_RATE)}, the "
            f"recording is {format_rate(sample_rate)}."
        )
        return 1
    try:
        decoder = create_live_decoder(args.protocol, sample_rate, args.offset)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    if decoder.status.problem:
        print(f"Error: {decoder.status.problem}")
        return 1

    messages = []
    block = max(1, int(sample_rate))
    for start in range(0, len(samples), block):
        messages += decoder.process(samples[start : start + block])
    messages += decoder.flush()  # e.g. the last line of Morse code
    valid = [m for m in messages if getattr(m, "valid", True)]
    invalid = [m for m in messages if not getattr(m, "valid", True)]

    summary = f"Decoded {len(valid)} {name} message(s)"
    if invalid:
        summary += f" ({len(invalid)} failed their checks"
        summary += ")" if args.show_invalid else "; --show-invalid lists them)"

    def describe(msg: Any) -> Tuple[str, str]:
        # One message that can't be read must not end the listing.
        try:
            return describe_message(msg)
        except Exception as exc:
            return "", f"(could not be read: {exc})"

    print(summary)
    for i, msg in enumerate(valid):
        address, content = describe(msg)
        print(f"  [{i}] {'  '.join(p for p in (address, content) if p)}")
    if args.show_invalid:
        for msg in invalid:
            address, content = describe(msg)
            error = getattr(msg, "error_message", "") or "failed its checks"
            text = "  ".join(p for p in (address, content) if p)
            print(
                f"  [invalid] {text}  ({error})" if text else f"  [invalid] ({error})"
            )
    return 0


#: ``sdr-scan encode`` defaults per protocol: (baud, shift in Hz). RTTY is
#: the amateur standard; ASCII keeps the library's 1 kHz shift.
_ENCODE_FSK_DEFAULTS = {"rtty": (45.45, 170.0), "ascii": (300.0, 1000.0)}
_ENCODE_NAMES = {"rtty": "RTTY", "morse": "Morse", "ascii": "ASCII", "psk31": "PSK31"}


def cmd_encode(args: argparse.Namespace) -> int:
    """Encode text to protocol."""
    from .protocols.encoder import EncoderConfig, ModulationType
    from .protocols.encoders import (
        ASCIIEncoder,
        MorseEncoder,
        PSK31Encoder,
        RTTYEncoder,
    )
    from .protocols.text_modes import (
        BAUD_RANGE,
        SHIFT_RANGE,
        WPM_RANGE,
        format_number,
    )

    encoders = {
        "rtty": RTTYEncoder,
        "morse": MorseEncoder,
        "ascii": ASCIIEncoder,
        "psk31": PSK31Encoder,
    }

    protocol = args.protocol
    if protocol not in encoders:
        print(f"Unknown protocol: {protocol}")
        print(f"Available: {', '.join(encoders.keys())}")
        return 1
    name = _ENCODE_NAMES[protocol]
    if not args.sample_rate > 0:
        print("Error: --sample-rate must be positive")
        return 1

    # Each protocol its own configuration: one shared config used to make
    # every protocol 300 Bd with a 1 kHz shift, so RTTY wasn't RTTY.
    baud = getattr(args, "baud", None)
    shift = getattr(args, "shift", None)
    if protocol in _ENCODE_FSK_DEFAULTS:
        default_baud, default_shift = _ENCODE_FSK_DEFAULTS[protocol]
        baud = default_baud if baud is None else float(baud)
        shift = default_shift if shift is None else float(shift)
        if not BAUD_RANGE[0] <= baud <= BAUD_RANGE[1]:
            print(
                f"Error: --baud must be {format_number(BAUD_RANGE[0])} to "
                f"{format_number(BAUD_RANGE[1])}"
            )
            return 1
        if not SHIFT_RANGE[0] <= shift <= SHIFT_RANGE[1]:
            print(
                f"Error: --shift must be {format_number(SHIFT_RANGE[0])} to "
                f"{format_number(SHIFT_RANGE[1])} Hz"
            )
            return 1
        details = f"{format_number(baud)} Bd, {format_number(shift)} Hz shift"
        modulation = ModulationType.FSK
    else:
        if baud is not None:
            print(
                "Error: PSK31 always runs at 31.25 Bd; --baud is for rtty and ascii."
                if protocol == "psk31"
                else "Error: Morse speed is set with --wpm; --baud is for rtty "
                "and ascii."
            )
            return 1
        if shift is not None:
            print(
                f"Error: {name} has no frequency shift; --shift is for rtty and ascii."
            )
            return 1
        if protocol == "morse":
            if not WPM_RANGE[0] <= args.wpm <= WPM_RANGE[1]:
                print(
                    f"Error: --wpm must be {format_number(WPM_RANGE[0])} to "
                    f"{format_number(WPM_RANGE[1])}"
                )
                return 1
            details = f"{args.wpm} WPM"
            modulation = ModulationType.OOK
        else:
            details = "31.25 Bd"
            modulation = ModulationType.PSK

    config = EncoderConfig(
        sample_rate=args.sample_rate,
        carrier_freq=args.carrier,
        baud_rate=baud or 0,
        modulation=modulation,
        amplitude=0.8,
        frequency_shift=shift,
    )

    encoder_class = encoders[protocol]
    if protocol == "morse":
        encoder = encoder_class(config, wpm=args.wpm)
    else:
        encoder = encoder_class(config)

    # The signal must stay inside +-sample_rate/2, or it aliases (mark and
    # space swap, a carrier lands on 0 Hz). The output is I/Q: a negative
    # carrier is fine. Its edges (of its 99% power bandwidth) may reach
    # +-sample_rate/2, so what the error suggests is accepted.
    width = round(encoder.text_signal("").occupied_bandwidth_hz())
    half_rate = args.sample_rate / 2.0
    reach = abs(args.carrier) + width / 2.0
    if reach > half_rate + 1e-6:
        # (Rounded inwards to the 0.01 Hz shown, so each suggestion fits.)
        room = math.floor((half_rate - width / 2.0) * 100.0) / 100.0
        least = math.ceil(2.0 * reach * 100.0) / 100.0
        carriers = (
            f"a --carrier from {format_number(-room)} to {format_number(room)} Hz, or "
            if room > 0
            else ""
        )
        print(
            f"Error: the {name} signal ({format_number(width)} Hz wide) at "
            f"--carrier {format_number(args.carrier)} Hz reaches "
            f"{format_number(reach)} Hz, more than --sample-rate "
            f"{format_number(args.sample_rate)} can hold (±{format_number(half_rate)}"
            f" Hz): use {carriers}a --sample-rate of at least {format_number(least)}."
        )
        return 1

    text = args.text or sys.stdin.read().strip()
    if not text:
        print("No text provided")
        return 1

    samples = encoder.encode_text(text)
    check = encoder.last_check
    if check is not None and check.unsupported:
        count = check.skipped
        print(
            f"Skipped {count} character{'' if count == 1 else 's'} {name} can't "
            f"send: {check.describe_unsupported()}"
        )
    if not len(samples) or check is None:
        print(f"Error: nothing to send: {name} can't send any of these characters.")
        return 1

    print(
        f"Encoded {check.characters} characters using {protocol.upper()} "
        f"({details})"
    )
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
    from .core.chirp_csv import bookmarks_to_channels, list_order

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
    # In the order the Bookmarks tab lists them (each group, then the rest),
    # as its export writes them.
    bookmarks = settings.get_bookmarks()
    return bookmarks_to_channels([bookmarks[i] for i in list_order(bookmarks)]), None


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
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a frequency range with a connected SDR, or a recording",
        description=(
            "Sweep a connected RTL-SDR or HackRF One across a range and list "
            "the signals found (continuously until Ctrl+C, or once with "
            "--single), sweep with two at once (--dual), or find the signals "
            "in a recorded I/Q file with --input."
        ),
    )
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
        "--step",
        type=float,
        default=100.0,
        help="Channel spacing in kHz: each channel reports the strongest "
        "signal within half a step (default: 100.0)",
    )
    scan_parser.add_argument(
        "--threshold",
        type=_threshold_arg,
        default=_DEFAULT_THRESHOLD,
        help="Detection threshold in dBFS, or 'auto' for the noise floor + 12 dB "
        "(default: auto for a live scan, -60 for --input)",
    )
    scan_parser.add_argument(
        "--single", action="store_true", help="Single scan (default: continuous)"
    )
    scan_parser.add_argument(
        "--device",
        choices=["rtlsdr", "hackrf"],
        help="Live scan: which kind of SDR to use (default: the first found)",
    )
    scan_parser.add_argument(
        "--dual",
        action="store_true",
        help="Live scan with two connected SDRs at once (an RTL-SDR and a "
        "HackRF One, or two of a kind; with --device, two of that kind): each "
        "sweeps part of the range, the faster one more, and the signals they "
        "find are listed together in frequency order",
    )
    scan_parser.add_argument(
        "--rate",
        type=_rates_arg,
        default=(2.4e6,),
        help="Live scan: sample rate in Hz (default: 2.4e6); with --dual, one "
        "for both receivers or one each, separated by a comma (2.4e6,20e6 runs "
        "the second, a HackRF One, at 20 MS/s)",
    )
    scan_parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Live scan: RF gain in dB (default: automatic gain)",
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
        choices=["pocsag", "flex", "ax25", "aprs", "rds", "adsb", "acars", "cw"],
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
    decode_parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Where the signal is, in Hz from the centre of the recording "
        "(default: 0, the centre)",
    )
    decode_parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="Also list messages that failed their checks (CRC, BCH, parity)",
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
        help="Where the signal sits in Hz: the tone, or the center between the "
        "FSK tones (default: 1000)",
    )
    encode_parser.add_argument(
        "--wpm", type=int, default=20, help="Words per minute for Morse (default: 20)"
    )
    encode_parser.add_argument(
        "--baud",
        type=float,
        default=None,
        help="Symbol rate for rtty and ascii (default: 45.45 for rtty, the "
        "amateur standard; 300 for ascii)",
    )
    encode_parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Frequency shift between mark and space in Hz, for rtty and ascii "
        "(default: 170 for rtty; 1000 for ascii)",
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
    # Same as the GUI's DEFAULT_FREQUENCY_HZ (not imported: that would load
    # the GUI package for every command).
    gui_parser.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=100.1e6,
        help="Initial frequency in Hz (default: the last one used, or 100.1 MHz)",
    )
    gui_parser.add_argument(
        "--sample-rate",
        "-s",
        type=float,
        default=2.4e6,
        help=(
            "Sample rate in Hz for the demo device; also preselected in "
            "Device > Connect (default: 2.4 MHz)"
        ),
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
