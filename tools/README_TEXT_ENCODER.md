# Text-to-Protocol Encoder

A powerful, interactive tool for encoding text into various radio protocols. Perfect for SDR experimentation, amateur radio, and protocol testing.

## Features

- 🎯 **Interactive UI** - Simple text-based interface
- 📻 **Multiple Protocols** - RTTY, Morse, ASCII FSK, PSK31
- ⚙️ **Configurable** - Adjust sample rate, carrier frequency, and protocol parameters
- 💾 **Save Outputs** - Export encoded signals as I/Q files
- 🔧 **Programmatic API** - Use encoders in your own Python scripts

## Quick Start

### Interactive Mode

Run the text encoder tool:

```bash
cd tools
python3 text_encoder.py
```

You'll see an interactive menu:

```
======================================================================
               TEXT-TO-PROTOCOL ENCODER
======================================================================

📻 Available Protocols:
----------------------------------------------------------------------
  [1] RTTY         - Radio Teletype (FSK, 45.45 baud, 170 Hz shift)
  [2] Morse        - Morse Code (CW/OOK, adjustable WPM)
  [3] ASCII        - ASCII FSK (8-N-1 framing, 300 baud)
  [4] PSK31        - PSK31 (BPSK, 31.25 baud, varicode)
  [5] Settings     - Configure sample rate and carrier frequency
  [q] Quit         - Exit the encoder
----------------------------------------------------------------------

🎯 Select protocol (or command):
```

### Example Workflow

1. **Select a Protocol**
   ```
   🎯 Select protocol (or command): 2
   ```

2. **Enter Your Message**
   ```
   ✏️  Enter your message:
      (Press Enter on empty line to finish)

      > HELLO WORLD
      > 73 DE W1ABC
      >
   ```

3. **Configure Options** (if needed)
   ```
   Words Per Minute (default 20): 15
   ```

4. **View Results**
   ```
   ✓ Encoding complete!
     Protocol:       Morse
     Text length:    24 characters
     Samples:        72000
     Duration:       1.500 seconds
     Sample rate:    48000 Hz
     Carrier freq:   700 Hz
   ```

5. **Save to File**
   ```
   💾 Save to file? (y/n): y
   Filename (without extension): morse_message
   ✓ Saved to: morse_message.iq
   ✓ Metadata saved to: morse_message.txt
   ```

## Supported Protocols

### 1. RTTY (Radio Teletype)

Classic radioteletype using FSK modulation.

- **Baud Rate:** 45.45 baud (standard)
- **Shift:** 170 Hz (standard amateur radio)
- **Encoding:** 5-bit Baudot code
- **Framing:** 1 start bit, 5 data bits, 1.5 stop bits
- **Use Case:** Amateur radio RTTY contests, legacy communications

**Example:**
```
Protocol: RTTY
Message: "CQ CQ DE W1ABC"
Output: FSK modulated signal ready for transmission
```

### 2. Morse Code (CW)

International Morse code using On-Off Keying.

- **Timing:** Configurable WPM (Words Per Minute)
- **Modulation:** OOK/CW (carrier on/off)
- **Standard:** PARIS standard timing
- **Use Case:** Amateur radio CW, emergency communications, maritime

**Timing Units:**
- Dit: 1 unit
- Dah: 3 units
- Inter-element gap: 1 unit
- Inter-character gap: 3 units
- Inter-word gap: 7 units

**Example:**
```
Protocol: Morse
Message: "SOS"
WPM: 20
Output: ... --- ... (classic distress signal)
```

### 3. ASCII FSK

Simple ASCII encoding using Frequency Shift Keying.

- **Baud Rate:** 300 baud (configurable)
- **Shift:** 1000 Hz (configurable)
- **Encoding:** 8-bit ASCII
- **Framing:** 8-N-1 (8 data bits, no parity, 1 stop bit)
- **Use Case:** Custom digital modes, experimentation

**Example:**
```
Protocol: ASCII
Message: "Test 123"
Output: FSK signal with full ASCII character set support
```

### 4. PSK31

Popular digital mode for keyboard-to-keyboard communications.

- **Baud Rate:** 31.25 baud (fixed)
- **Modulation:** Binary Phase Shift Keying (BPSK)
- **Encoding:** Varicode (variable-length encoding)
- **Efficiency:** Optimized for common characters
- **Use Case:** Amateur radio digital communications

**Example:**
```
Protocol: PSK31
Message: "hello world"
Output: BPSK varicode encoded signal
```

## Configuration Options

Access settings menu by selecting option `[5]`:

### Sample Rate
- **Default:** 48000 Hz
- **Typical Values:** 8000, 22050, 44100, 48000 Hz
- **Impact:** Higher rates = better quality, larger files

### Carrier Frequency
- **Default:** 1000 Hz
- **Range:** Typically 300-3000 Hz for audio transmission
- **Impact:** Center frequency of the modulated signal

## Output Files

### I/Q File (.iq)
Binary file containing complex64 samples:
- **Format:** Interleaved I/Q (complex64)
- **Use:** Load into GNU Radio, SDR#, or transmit with HackRF
- **Structure:** Each sample is 8 bytes (4 bytes I + 4 bytes Q)

### Metadata File (.txt)
Human-readable file with encoding details:
```
Protocol: Morse
Sample Rate: 48000 Hz
Carrier Frequency: 700 Hz
Samples: 72000
Duration: 1.500 s
Format: complex64 (I/Q)
```

## Programmatic Usage

You can also use the encoders directly in your Python code:

```python
from sdr_module.protocols.encoder import EncoderConfig, ModulationType
from sdr_module.protocols.encoders import MorseEncoder

# Create configuration
config = EncoderConfig(
    sample_rate=48000,
    carrier_freq=700,
    baud_rate=0,
    modulation=ModulationType.OOK,
    amplitude=0.8
)

# Create encoder
encoder = MorseEncoder(config, wpm=20)

# Encode text
samples = encoder.encode_text("HELLO WORLD")

# Save to file
samples.tofile("output.iq")
```

See `examples/text_encoding_example.py` for more examples.

## Transmitting Encoded Signals

### With HackRF One

```bash
# Transmit at 433 MHz with 8 MHz sample rate
hackrf_transfer -t morse_message.iq -f 433000000 -s 8000000 -x 20
```

### With GNU Radio

1. Load the .iq file using "File Source" block
2. Set file type to "complex float (complex64)"
3. Connect to "USRP Sink" or "HackRF Sink"
4. Configure center frequency and sample rate to match metadata

### With gr-osmosdr

```python
from gnuradio import gr, blocks
from osmosdr import sink

# Create flowgraph
fg = gr.top_block()

# File source
src = blocks.file_source(gr.sizeof_gr_complex, "morse_message.iq", False)

# SDR sink
sdr = sink(args="hackrf=0")
sdr.set_sample_rate(48000)
sdr.set_center_freq(433e6)
sdr.set_gain(20)

# Connect and run
fg.connect(src, sdr)
fg.run()
```

## Tips and Best Practices

### 1. **Start Simple**
   - Begin with Morse code - it's the easiest to verify
   - Use low baud rates for cleaner signals
   - Start with short messages

### 2. **Verify Before Transmitting**
   - Always check your signal in a waterfall display first
   - Verify the encoded data looks correct
   - Test with receive-only setup before transmitting

### 3. **Legal Considerations**
   - Ensure you have appropriate license for transmission
   - Use correct amateur radio bands
   - Follow power limits and regulations
   - Include proper identification

### 4. **Testing**
   - Use a dummy load or low power for initial tests
   - Record and analyze your transmissions
   - Compare with known-good reference signals

### 5. **Optimization**
   - Match sample rate to your SDR capabilities
   - Use appropriate carrier frequencies for your target band
   - Adjust amplitude to prevent clipping

## Troubleshooting

### Issue: No output file created
**Solution:** Check file permissions in the current directory

### Issue: Signal sounds distorted
**Solution:**
- Lower the amplitude (try 0.5 instead of 0.8)
- Check sample rate matches your SDR
- Verify carrier frequency is within valid range

### Issue: Can't decode received signal
**Solution:**
- Verify exact same parameters used for encoding
- Check for timing drift
- Ensure proper tuning and frequency alignment

### Issue: Python import errors
**Solution:**
```bash
# From SDR project root:
pip install -e .
```

## Advanced Features

### Custom Protocols

Create your own protocol encoder:

```python
from sdr_module.protocols.encoder import ProtocolEncoder

class MyCustomEncoder(ProtocolEncoder):
    def encode_text(self, text: str) -> np.ndarray:
        # Your encoding logic here
        bits = self.text_to_bits(text)
        return self.bits_to_fsk(bits, mark_freq, space_freq)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        # Your byte encoding logic
        pass
```

### Batch Processing

Process multiple messages:

```python
messages = ["CQ DE W1ABC", "TEST 123", "73"]
for msg in messages:
    samples = encoder.encode_text(msg)
    samples.tofile(f"message_{messages.index(msg)}.iq")
```

## Requirements

- Python 3.9+
- NumPy
- sdr-module (this package)

Optional:
- Matplotlib (for plotting examples)
- GNU Radio (for advanced signal processing)
- SDR hardware (HackRF, RTL-SDR, etc.)

## License

MIT License - See project root LICENSE file

## Contributing

Contributions welcome! Consider adding:
- New protocol encoders (SSTV, APRS, etc.)
- GUI interface
- Real-time transmission mode
- Decoder integration
- Protocol validation tools

## References

- RTTY: [Amateur Radio RTTY](https://en.wikipedia.org/wiki/Radioteletype)
- Morse Code: [International Morse Code](https://en.wikipedia.org/wiki/Morse_code)
- PSK31: [PSK31 Specification](http://www.arrl.org/psk31-spec)
- FSK: [Frequency-shift keying](https://en.wikipedia.org/wiki/Frequency-shift_keying)
