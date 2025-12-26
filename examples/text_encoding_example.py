#!/usr/bin/env python3
"""
Example: Text-to-Protocol Encoding

Demonstrates how to encode text into various radio protocols
programmatically without using the interactive UI.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sdr_module.protocols.encoder import EncoderConfig, ModulationType
from sdr_module.protocols.encoders import (
    RTTYEncoder,
    MorseEncoder,
    ASCIIEncoder,
    PSK31Encoder
)


def encode_morse_example():
    """Encode text to Morse code."""
    print("=" * 70)
    print("Morse Code Encoding Example")
    print("=" * 70)

    # Configuration
    config = EncoderConfig(
        sample_rate=48000,
        carrier_freq=700,  # 700 Hz tone
        baud_rate=0,       # Not used for Morse
        modulation=ModulationType.OOK,
        amplitude=0.8
    )

    # Create encoder (20 WPM)
    encoder = MorseEncoder(config, wpm=20)

    # Encode text
    text = "HELLO WORLD"
    print(f"Text: {text}")
    samples = encoder.encode_text(text)

    print(f"Generated {len(samples)} samples")
    print(f"Duration: {len(samples) / config.sample_rate:.2f} seconds")

    return samples, config.sample_rate


def encode_rtty_example():
    """Encode text to RTTY."""
    print("\n" + "=" * 70)
    print("RTTY Encoding Example")
    print("=" * 70)

    # Configuration
    config = EncoderConfig(
        sample_rate=8000,
        carrier_freq=2125,     # Standard RTTY center frequency
        baud_rate=45.45,       # Standard RTTY baud
        modulation=ModulationType.FSK,
        amplitude=0.8,
        frequency_shift=170    # Standard shift
    )

    # Create encoder
    encoder = RTTYEncoder(config)

    # Encode text
    text = "CQ CQ DE W1ABC"
    print(f"Text: {text}")
    samples = encoder.encode_text(text)

    print(f"Generated {len(samples)} samples")
    print(f"Duration: {len(samples) / config.sample_rate:.2f} seconds")

    return samples, config.sample_rate


def encode_ascii_example():
    """Encode text to ASCII FSK."""
    print("\n" + "=" * 70)
    print("ASCII FSK Encoding Example")
    print("=" * 70)

    # Configuration
    config = EncoderConfig(
        sample_rate=48000,
        carrier_freq=1500,
        baud_rate=300,
        modulation=ModulationType.FSK,
        amplitude=0.8,
        frequency_shift=1000
    )

    # Create encoder
    encoder = ASCIIEncoder(config)

    # Encode text
    text = "SDR Encoder Test 123"
    print(f"Text: {text}")
    samples = encoder.encode_text(text)

    print(f"Generated {len(samples)} samples")
    print(f"Duration: {len(samples) / config.sample_rate:.2f} seconds")

    return samples, config.sample_rate


def plot_signal(samples, sample_rate, title):
    """
    Plot the encoded signal.

    Args:
        samples: Complex I/Q samples
        sample_rate: Sample rate in Hz
        title: Plot title
    """
    # Time vector
    t = np.arange(len(samples)) / sample_rate

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot I and Q components
    ax1.plot(t[:1000], samples.real[:1000], label='I (Real)', alpha=0.7)
    ax1.plot(t[:1000], samples.imag[:1000], label='Q (Imag)', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Time Domain (first 1000 samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot magnitude
    magnitude = np.abs(samples)
    ax2.plot(t, magnitude)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'{title} - Magnitude Envelope')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TEXT-TO-PROTOCOL ENCODING EXAMPLES" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Example 1: Morse Code
    morse_samples, morse_sr = encode_morse_example()

    # Example 2: RTTY
    rtty_samples, rtty_sr = encode_rtty_example()

    # Example 3: ASCII FSK
    ascii_samples, ascii_sr = encode_ascii_example()

    # Save samples
    print("\n" + "=" * 70)
    print("Saving encoded samples...")
    print("=" * 70)

    morse_samples.tofile('morse_example.iq')
    print("✓ Saved: morse_example.iq")

    rtty_samples.tofile('rtty_example.iq')
    print("✓ Saved: rtty_example.iq")

    ascii_samples.tofile('ascii_example.iq')
    print("✓ Saved: ascii_example.iq")

    # Try to plot (optional, requires matplotlib)
    try:
        print("\n" + "=" * 70)
        print("Generating plots...")
        print("=" * 70)

        plot_signal(morse_samples, morse_sr, "Morse Code")
        plt.savefig('morse_plot.png')
        print("✓ Saved: morse_plot.png")

        plot_signal(rtty_samples, rtty_sr, "RTTY")
        plt.savefig('rtty_plot.png')
        print("✓ Saved: rtty_plot.png")

        plot_signal(ascii_samples, ascii_sr, "ASCII FSK")
        plt.savefig('ascii_plot.png')
        print("✓ Saved: ascii_plot.png")

        print("\nPlots saved successfully!")

    except ImportError:
        print("\n⚠️  Matplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\n⚠️  Plotting error: {e}")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Load the .iq files in SDR software (e.g., GNU Radio, SDR#)")
    print("  2. Transmit them using HackRF or other TX-capable SDR")
    print("  3. Analyze the waveforms in the saved plots")
    print()


if __name__ == '__main__':
    main()
