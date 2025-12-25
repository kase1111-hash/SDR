#!/usr/bin/env python3
"""
Dual-SDR Receive Example

Demonstrates simultaneous reception using RTL-SDR and HackRF One.
RTL-SDR monitors 433 MHz ISM band while HackRF monitors 915 MHz.
"""

import numpy as np
import time
import signal
import sys

# Add src to path for development
sys.path.insert(0, '../src')

from sdr_module import DualSDRController
from sdr_module.core.config import SDRConfig, create_preset_dual_rx
from sdr_module.dsp.spectrum import SpectrumAnalyzer, WindowType, AveragingMode
from sdr_module.dsp.classifiers import SignalClassifier
from sdr_module.utils.conversions import freq_to_str, linear_to_db


# Global flag for clean shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for clean shutdown."""
    global running
    print("\nShutting down...")
    running = False


def print_spectrum_summary(name: str, result, peaks):
    """Print spectrum analysis summary."""
    print(f"\n=== {name} Spectrum ===")
    print(f"Center: {freq_to_str(result.center_freq)}")
    print(f"RBW: {result.rbw:.1f} Hz")
    print(f"Power range: {result.power_db.min():.1f} to {result.power_db.max():.1f} dB")

    if peaks:
        print(f"Top {min(3, len(peaks))} peaks:")
        for i, peak in enumerate(peaks[:3]):
            print(f"  {i+1}. {freq_to_str(peak.frequency)}: {peak.power_db:.1f} dB")


def main():
    """Main entry point."""
    global running

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("Dual-SDR Receive Example")
    print("=" * 60)
    print()

    # Create configuration
    config = create_preset_dual_rx()

    # Configure frequencies
    config.dual_sdr.rtlsdr.frequency = 433.92e6   # 433 MHz ISM
    config.dual_sdr.rtlsdr.sample_rate = 2.4e6
    config.dual_sdr.rtlsdr.gain = 40

    config.dual_sdr.hackrf.frequency = 915e6      # 915 MHz ISM
    config.dual_sdr.hackrf.sample_rate = 10e6
    config.dual_sdr.hackrf.gain = 30

    print(f"RTL-SDR: {freq_to_str(config.dual_sdr.rtlsdr.frequency)}")
    print(f"HackRF:  {freq_to_str(config.dual_sdr.hackrf.frequency)}")
    print()

    # Create spectrum analyzers
    rtl_spectrum = SpectrumAnalyzer(
        fft_size=4096,
        window=WindowType.BLACKMAN_HARRIS,
        averaging=AveragingMode.RMS,
        avg_count=10
    )

    hackrf_spectrum = SpectrumAnalyzer(
        fft_size=8192,
        window=WindowType.BLACKMAN_HARRIS,
        averaging=AveragingMode.RMS,
        avg_count=10
    )

    # Create classifiers
    rtl_classifier = SignalClassifier(config.dual_sdr.rtlsdr.sample_rate)
    hackrf_classifier = SignalClassifier(config.dual_sdr.hackrf.sample_rate)

    # Sample counters
    rtl_samples_total = 0
    hackrf_samples_total = 0

    def on_rtl_samples(samples):
        """Callback for RTL-SDR samples."""
        nonlocal rtl_samples_total
        rtl_samples_total += len(samples)

    def on_hackrf_samples(samples):
        """Callback for HackRF samples."""
        nonlocal hackrf_samples_total
        hackrf_samples_total += len(samples)

    # Initialize dual-SDR controller
    print("Initializing dual-SDR system...")

    try:
        controller = DualSDRController(config)

        if not controller.initialize():
            print("ERROR: Failed to initialize SDR devices")
            print("Make sure RTL-SDR and/or HackRF are connected.")
            return 1

        print("Devices initialized successfully!")
        print()

        # Print device status
        status = controller.get_status()
        print("Device Status:")
        print(f"  RTL-SDR connected: {status['rtlsdr']['connected']}")
        print(f"  HackRF connected:  {status['hackrf']['connected']}")
        print()

        # Start dual receive
        print("Starting dual receive mode...")
        if not controller.start_dual_rx(on_rtl_samples, on_hackrf_samples):
            print("ERROR: Failed to start dual receive")
            controller.shutdown()
            return 1

        print("Dual receive active. Press Ctrl+C to stop.")
        print()

        # Main processing loop
        last_report = time.time()
        report_interval = 2.0  # seconds

        while running:
            current_time = time.time()

            # Read samples from buffers
            rtl_samples = controller.read_rtlsdr_samples(256 * 1024, timeout=0.1)
            hackrf_samples = controller.read_hackrf_samples(512 * 1024, timeout=0.1)

            # Process RTL-SDR samples
            if rtl_samples is not None and len(rtl_samples) > 0:
                result = rtl_spectrum.compute_spectrum(
                    rtl_samples,
                    config.dual_sdr.rtlsdr.frequency,
                    config.dual_sdr.rtlsdr.sample_rate
                )
                peaks = rtl_spectrum.find_peaks(result, threshold_db=-50)

                # Classify signal
                classification = rtl_classifier.classify(rtl_samples)

            # Process HackRF samples
            if hackrf_samples is not None and len(hackrf_samples) > 0:
                result = hackrf_spectrum.compute_spectrum(
                    hackrf_samples,
                    config.dual_sdr.hackrf.frequency,
                    config.dual_sdr.hackrf.sample_rate
                )
                peaks = hackrf_spectrum.find_peaks(result, threshold_db=-50)

            # Periodic status report
            if current_time - last_report >= report_interval:
                elapsed = current_time - last_report

                rtl_rate = rtl_samples_total / elapsed / 1e6
                hackrf_rate = hackrf_samples_total / elapsed / 1e6

                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"RTL: {rtl_rate:.2f} MS/s | "
                      f"HackRF: {hackrf_rate:.2f} MS/s | "
                      f"Buffers: RTL {controller.rtlsdr_buffer.stats.fill_ratio*100:.0f}% "
                      f"HackRF {controller.hackrf_buffer.stats.fill_ratio*100:.0f}%")

                rtl_samples_total = 0
                hackrf_samples_total = 0
                last_report = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        print("\nShutting down...")
        controller.shutdown()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
