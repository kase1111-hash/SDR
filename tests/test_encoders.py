"""Tests for protocol encoders."""

import numpy as np
import pytest

from sdr_module.protocols.encoder import (
    EncoderConfig,
    ModulationType,
)
from sdr_module.protocols.encoders import (
    ASCIIEncoder,
    MorseEncoder,
    PSK31Encoder,
    RTTYEncoder,
)


class TestEncoderConfig:
    """Tests for EncoderConfig dataclass."""

    def test_basic_config(self):
        """Test creating basic encoder configuration."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=300,
            modulation=ModulationType.FSK,
        )
        assert config.sample_rate == 48000
        assert config.carrier_freq == 1000
        assert config.baud_rate == 300
        assert config.modulation == ModulationType.FSK
        assert config.amplitude == 1.0

    def test_config_with_fsk_shift(self):
        """Test configuration with FSK frequency shift."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=45.45,
            modulation=ModulationType.FSK,
            frequency_shift=170,
        )
        assert config.frequency_shift == 170

    def test_config_with_amplitude(self):
        """Test configuration with custom amplitude."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=300,
            modulation=ModulationType.ASK,
            amplitude=0.5,
        )
        assert config.amplitude == 0.5


class TestModulationType:
    """Tests for ModulationType enum."""

    def test_modulation_types(self):
        """Test all modulation types exist."""
        assert ModulationType.FSK.value == "fsk"
        assert ModulationType.ASK.value == "ask"
        assert ModulationType.PSK.value == "psk"
        assert ModulationType.OOK.value == "ook"
        assert ModulationType.AFSK.value == "afsk"
        assert ModulationType.MSK.value == "msk"


class TestRTTYEncoder:
    """Tests for RTTY encoder."""

    @pytest.fixture
    def rtty_encoder(self):
        """Create RTTY encoder instance."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=0,  # Will be set to 45.45
            modulation=ModulationType.FSK,
        )
        return RTTYEncoder(config)

    def test_default_baud_rate(self, rtty_encoder):
        """Test default baud rate is set."""
        assert rtty_encoder.config.baud_rate == 45.45

    def test_default_shift(self, rtty_encoder):
        """Test default frequency shift is set."""
        assert rtty_encoder.config.frequency_shift == 170

    def test_encode_single_character(self, rtty_encoder):
        """Test encoding a single character."""
        signal = rtty_encoder.encode_text("A")
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex64
        assert len(signal) > 0

    def test_encode_text(self, rtty_encoder):
        """Test encoding text string."""
        signal = rtty_encoder.encode_text("CQ CQ CQ")
        assert isinstance(signal, np.ndarray)
        assert len(signal) > 0

    def test_encode_lowercase_converted(self, rtty_encoder):
        """Test lowercase is converted to uppercase."""
        signal_lower = rtty_encoder.encode_text("hello")
        signal_upper = rtty_encoder.encode_text("HELLO")
        # Both should produce same output
        assert len(signal_lower) == len(signal_upper)

    def test_encode_bytes(self, rtty_encoder):
        """Test encoding bytes."""
        signal = rtty_encoder.encode_bytes(b"TEST")
        assert isinstance(signal, np.ndarray)
        assert len(signal) > 0

    def test_encode_bytes_invalid(self, rtty_encoder):
        """Test encoding invalid bytes returns empty array."""
        signal = rtty_encoder.encode_bytes(b"\xff\xfe")
        assert len(signal) == 0

    def test_baudot_table(self):
        """Test Baudot code table contains expected characters."""
        assert "A" in RTTYEncoder.BAUDOT_LETTERS
        assert "Z" in RTTYEncoder.BAUDOT_LETTERS
        assert " " in RTTYEncoder.BAUDOT_LETTERS


class TestMorseEncoder:
    """Tests for Morse code encoder."""

    @pytest.fixture
    def morse_encoder(self):
        """Create Morse encoder instance."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=700,
            baud_rate=0,
            modulation=ModulationType.OOK,
        )
        return MorseEncoder(config, wpm=20)

    def test_wpm_setting(self, morse_encoder):
        """Test WPM is properly set."""
        assert morse_encoder._wpm == 20

    def test_dit_duration(self, morse_encoder):
        """Test dit duration calculation."""
        # PARIS standard: 1.2 / wpm seconds
        expected = 1.2 / 20
        assert abs(morse_encoder._dit_duration - expected) < 0.001

    def test_encode_single_letter(self, morse_encoder):
        """Test encoding single letter."""
        signal = morse_encoder.encode_text("E")  # Single dit
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex64

    def test_encode_word(self, morse_encoder):
        """Test encoding word."""
        signal = morse_encoder.encode_text("SOS")
        assert len(signal) > 0

    def test_encode_with_spaces(self, morse_encoder):
        """Test encoding text with spaces."""
        signal = morse_encoder.encode_text("CQ CQ")
        assert len(signal) > 0

    def test_morse_table(self):
        """Test Morse code table."""
        assert MorseEncoder.MORSE_CODE["S"] == "..."
        assert MorseEncoder.MORSE_CODE["O"] == "---"
        assert MorseEncoder.MORSE_CODE["1"] == ".----"

    def test_encode_bytes(self, morse_encoder):
        """Test encoding bytes."""
        signal = morse_encoder.encode_bytes(b"HI")
        assert len(signal) > 0

    def test_encode_bytes_invalid(self, morse_encoder):
        """Test encoding invalid bytes."""
        signal = morse_encoder.encode_bytes(b"\xff\xfe")
        assert len(signal) == 0

    def test_different_wpm(self):
        """Test different WPM settings."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=700,
            baud_rate=0,
            modulation=ModulationType.OOK,
        )
        slow = MorseEncoder(config, wpm=10)
        fast = MorseEncoder(config, wpm=30)

        # Slower WPM = longer dit duration
        assert slow._dit_duration > fast._dit_duration


class TestASCIIEncoder:
    """Tests for ASCII FSK encoder."""

    @pytest.fixture
    def ascii_encoder(self):
        """Create ASCII encoder instance."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1500,
            baud_rate=300,
            modulation=ModulationType.FSK,
        )
        return ASCIIEncoder(config)

    def test_default_shift(self, ascii_encoder):
        """Test default frequency shift is set."""
        assert ascii_encoder.config.frequency_shift == 1000

    def test_encode_text(self, ascii_encoder):
        """Test encoding text."""
        signal = ascii_encoder.encode_text("Hello")
        assert isinstance(signal, np.ndarray)
        assert len(signal) > 0

    def test_encode_bytes(self, ascii_encoder):
        """Test encoding bytes with framing."""
        signal = ascii_encoder.encode_bytes(b"AB")
        assert len(signal) > 0

    def test_framing_bits(self, ascii_encoder):
        """Test correct number of bits per byte."""
        # Each byte: 1 start + 8 data + 1 stop = 10 bits
        signal = ascii_encoder.encode_bytes(b"A")
        samples_per_bit = int(48000 / 300)
        expected_samples = 10 * samples_per_bit
        assert len(signal) == expected_samples


class TestPSK31Encoder:
    """Tests for PSK31 encoder."""

    @pytest.fixture
    def psk31_encoder(self):
        """Create PSK31 encoder instance."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=0,  # Will be set to 31.25
            modulation=ModulationType.FSK,  # Will be changed to PSK
        )
        return PSK31Encoder(config)

    def test_baud_rate_fixed(self, psk31_encoder):
        """Test baud rate is fixed at 31.25."""
        assert psk31_encoder.config.baud_rate == 31.25

    def test_modulation_type(self, psk31_encoder):
        """Test modulation type is PSK."""
        assert psk31_encoder.config.modulation == ModulationType.PSK

    def test_encode_text(self, psk31_encoder):
        """Test encoding text."""
        signal = psk31_encoder.encode_text("test")
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex64

    def test_varicode_table(self):
        """Test varicode table contains common characters."""
        assert " " in PSK31Encoder.VARICODE
        assert "e" in PSK31Encoder.VARICODE
        assert "t" in PSK31Encoder.VARICODE

    def test_encode_bytes(self, psk31_encoder):
        """Test encoding bytes."""
        signal = psk31_encoder.encode_bytes(b"hi")
        assert len(signal) > 0

    def test_encode_bytes_invalid(self, psk31_encoder):
        """Test encoding invalid bytes."""
        signal = psk31_encoder.encode_bytes(b"\xff\xfe")
        assert len(signal) == 0


class TestProtocolEncoderBase:
    """Tests for ProtocolEncoder base class methods."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance using RTTYEncoder."""
        config = EncoderConfig(
            sample_rate=48000,
            carrier_freq=1000,
            baud_rate=100,
            modulation=ModulationType.FSK,
            frequency_shift=500,
        )
        return RTTYEncoder(config)

    def test_text_to_bits(self, encoder):
        """Test text to bits conversion."""
        bits = encoder.text_to_bits("A")
        assert isinstance(bits, np.ndarray)
        # 'A' = 0x41 = 01000001
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_bits_to_fsk(self, encoder):
        """Test FSK modulation."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        signal = encoder.bits_to_fsk(bits, 1250, 750)

        samples_per_bit = int(48000 / 100)
        assert len(signal) == 4 * samples_per_bit
        assert signal.dtype == np.complex64

    def test_bits_to_ask(self, encoder):
        """Test ASK modulation."""
        bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        signal = encoder.bits_to_ask(bits)

        samples_per_bit = int(48000 / 100)
        assert len(signal) == 4 * samples_per_bit

        # Check that '0' bits have zero amplitude
        bit_1_power = np.mean(np.abs(signal[0:samples_per_bit]) ** 2)
        bit_0_power = np.mean(
            np.abs(signal[samples_per_bit : 2 * samples_per_bit]) ** 2
        )
        assert bit_1_power > 0
        assert bit_0_power < 0.01

    def test_add_preamble_fsk(self, encoder):
        """Test adding preamble with FSK."""
        signal = np.ones(100, dtype=np.complex64)
        preamble_bits = np.array([1, 0, 1, 0], dtype=np.uint8)

        result = encoder.add_preamble(signal, preamble_bits)

        samples_per_bit = int(48000 / 100)
        expected_preamble_len = 4 * samples_per_bit
        assert len(result) == expected_preamble_len + 100

    def test_config_property(self, encoder):
        """Test config property returns configuration."""
        config = encoder.config
        assert config.sample_rate == 48000
        assert config.carrier_freq == 1000
