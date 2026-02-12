#!/usr/bin/env python3
"""
Text-to-Protocol Encoder Tool

Interactive tool for encoding text into various radio protocols.
Converts text input to modulated I/Q samples ready for transmission.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sdr_module.protocols.encoder import EncoderConfig, ModulationType
from sdr_module.protocols.encoders import (
    RTTYEncoder,
    MorseEncoder,
    ASCIIEncoder,
    PSK31Encoder
)


class TextEncoderUI:
    """Interactive text encoder with simple UI."""

    def __init__(self):
        """Initialize encoder UI."""
        self.sample_rate = 48000  # Default sample rate
        self.carrier_freq = 1000   # Default carrier frequency
        self.encoders = {
            '1': ('RTTY', RTTYEncoder, 'Radio Teletype (FSK, 45.45 baud, 170 Hz shift)'),
            '2': ('Morse', MorseEncoder, 'Morse Code (CW/OOK, adjustable WPM)'),
            '3': ('ASCII', ASCIIEncoder, 'ASCII FSK (8-N-1 framing, 300 baud)'),
            '4': ('PSK31', PSK31Encoder, 'PSK31 (BPSK, 31.25 baud, varicode)'),
        }

    def print_header(self):
        """Print application header."""
        print("=" * 70)
        print(" " * 15 + "TEXT-TO-PROTOCOL ENCODER")
        print("=" * 70)
        print()

    def print_menu(self):
        """Print protocol selection menu."""
        print("\n📻 Available Protocols:")
        print("-" * 70)
        for key, (name, _, desc) in self.encoders.items():
            print(f"  [{key}] {name:12} - {desc}")
        print(f"  [5] Settings  - Configure sample rate and carrier frequency")
        print(f"  [q] Quit      - Exit the encoder")
        print("-" * 70)

    def get_text_input(self) -> str:
        """Get text input from user."""
        print("\n✏️  Enter your message:")
        print("   (Press Enter on empty line to finish)\n")

        lines = []
        while True:
            line = input("   > ")
            if not line:
                break
            lines.append(line)

        return '\n'.join(lines) if lines else ''

    def configure_settings(self):
        """Configure encoder settings."""
        print("\n⚙️  Settings Configuration")
        print("-" * 70)

        try:
            rate = input(f"Sample Rate (current: {self.sample_rate} Hz): ")
            if rate.strip():
                self.sample_rate = float(rate)
                print(f"✓ Sample rate set to {self.sample_rate} Hz")

            freq = input(f"Carrier Frequency (current: {self.carrier_freq} Hz): ")
            if freq.strip():
                self.carrier_freq = float(freq)
                print(f"✓ Carrier frequency set to {self.carrier_freq} Hz")

        except ValueError:
            print("✗ Invalid input. Settings unchanged.")

    def encode_text(self, protocol_key: str, text: str):
        """
        Encode text using selected protocol.

        Args:
            protocol_key: Protocol selection key
            text: Text to encode
        """
        if not text:
            print("✗ No text provided!")
            return

        if protocol_key not in self.encoders:
            print("✗ Invalid protocol selection!")
            return

        name, encoder_class, _ = self.encoders[protocol_key]

        print(f"\n🔧 Encoding with {name}...")
        print("-" * 70)

        try:
            # Create encoder configuration
            config = EncoderConfig(
                sample_rate=self.sample_rate,
                carrier_freq=self.carrier_freq,
                baud_rate=300,  # Default, may be overridden by encoder
                modulation=ModulationType.FSK,
                amplitude=0.8,
                frequency_shift=1000
            )

            # Special handling for Morse (needs WPM setting)
            if protocol_key == '2':
                wpm_str = input("Words Per Minute (default 20): ")
                wpm = int(wpm_str) if wpm_str.strip() else 20
                encoder = encoder_class(config, wpm=wpm)
            else:
                encoder = encoder_class(config)

            # Encode the text
            samples = encoder.encode_text(text)

            # Display results
            print(f"\n✓ Encoding complete!")
            print(f"  Protocol:       {name}")
            print(f"  Text length:    {len(text)} characters")
            print(f"  Samples:        {len(samples)}")
            print(f"  Duration:       {len(samples) / self.sample_rate:.3f} seconds")
            print(f"  Sample rate:    {self.sample_rate} Hz")
            print(f"  Carrier freq:   {self.carrier_freq} Hz")

            # Ask to save
            save = input("\n💾 Save to file? (y/n): ")
            if save.lower() == 'y':
                self.save_samples(samples, name)

        except Exception as e:
            print(f"✗ Encoding failed: {e}")

    def save_samples(self, samples: np.ndarray, protocol: str):
        """
        Save encoded samples to file.

        Args:
            samples: I/Q samples to save
            protocol: Protocol name for filename
        """
        filename = input("Filename (without extension): ")
        if not filename:
            filename = f"{protocol.lower()}_output"

        try:
            # Save as complex64 binary
            filepath = f"{filename}.iq"
            samples.tofile(filepath)
            print(f"✓ Saved to: {filepath}")

            # Also save metadata
            meta_filepath = f"{filename}.txt"
            with open(meta_filepath, 'w') as f:
                f.write(f"Protocol: {protocol}\n")
                f.write(f"Sample Rate: {self.sample_rate} Hz\n")
                f.write(f"Carrier Frequency: {self.carrier_freq} Hz\n")
                f.write(f"Samples: {len(samples)}\n")
                f.write(f"Duration: {len(samples) / self.sample_rate:.3f} s\n")
                f.write(f"Format: complex64 (I/Q)\n")
            print(f"✓ Metadata saved to: {meta_filepath}")

        except Exception as e:
            print(f"✗ Save failed: {e}")

    def run(self):
        """Run the interactive encoder."""
        self.print_header()

        while True:
            self.print_menu()

            choice = input("\n🎯 Select protocol (or command): ").strip()

            if choice.lower() == 'q':
                print("\n👋 Goodbye!\n")
                break

            elif choice == '5':
                self.configure_settings()

            elif choice in self.encoders:
                text = self.get_text_input()
                if text:
                    self.encode_text(choice, text)
                else:
                    print("✗ No text entered.")

            else:
                print("✗ Invalid selection. Please try again.")


def main():
    """Main entry point."""
    try:
        ui = TextEncoderUI()
        ui.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
