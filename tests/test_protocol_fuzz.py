"""Fuzz-style property-based tests for protocol decoders using Hypothesis.

Validates that all protocol decoders handle arbitrary, malformed, and
edge-case inputs without crashing. Decoders must always return a list
(possibly empty) and never raise unhandled exceptions, regardless of
the input data.

Covers:
- Random byte sequences (complex64 and float32 samples)
- Structured-but-invalid data (wrong CRC, truncated frames, garbage)
- Edge cases (empty arrays, single-sample, very large values, NaN, inf)

Tested decoders: POCSAGDecoder, ADSBDecoder, RDSDecoder, AX25Decoder,
ACARSDecoder, FLEXDecoder.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sdr_module.dsp.protocols import (
    POCSAGDecoder,
    ADSBDecoder,
    RDSDecoder,
    AX25Decoder,
    ACARSDecoder,
    FLEXDecoder,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy: random array length between 10 and 10000
random_length = st.integers(min_value=10, max_value=10000)

# Strategy: generate a random complex64 numpy array of the given length.
# Uses flatmap to first pick a length, then build the array.
random_complex64_samples = random_length.flatmap(
    lambda n: st.just(n).map(
        lambda size: (
            np.random.randn(size) + 1j * np.random.randn(size)
        ).astype(np.complex64)
    )
)

# Strategy: generate a random float32 numpy array (real-valued) of random length.
random_float32_samples = random_length.flatmap(
    lambda n: st.just(n).map(
        lambda size: np.random.randn(size).astype(np.float32)
    )
)

# Strategy: generate random bytes then reinterpret as float32 samples.
# This produces truly arbitrary bit patterns including denormals, NaN, etc.
random_bytes_as_float32 = st.integers(min_value=10, max_value=5000).flatmap(
    lambda n: st.just(n).map(
        lambda size: np.frombuffer(
            np.random.bytes(size * 4), dtype=np.float32
        ).copy()
    )
)


# ---------------------------------------------------------------------------
# Decoder factory helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 2.4e6

DECODER_FACTORIES = [
    ("POCSAGDecoder", lambda: POCSAGDecoder(sample_rate=SAMPLE_RATE, baud_rate=1200)),
    ("ADSBDecoder", lambda: ADSBDecoder(sample_rate=SAMPLE_RATE)),
    ("RDSDecoder", lambda: RDSDecoder(sample_rate=SAMPLE_RATE)),
    ("AX25Decoder", lambda: AX25Decoder(sample_rate=SAMPLE_RATE, baud_rate=1200)),
    ("ACARSDecoder", lambda: ACARSDecoder(sample_rate=SAMPLE_RATE)),
    ("FLEXDecoder", lambda: FLEXDecoder(sample_rate=SAMPLE_RATE, baud_rate=1600)),
]


def _all_decoder_ids():
    return [name for name, _ in DECODER_FACTORIES]


def _all_decoders():
    return [factory() for _, factory in DECODER_FACTORIES]


# ---------------------------------------------------------------------------
# 1. Random byte sequences -- complex64 samples fed to each decoder
# ---------------------------------------------------------------------------


class TestRandomComplex64Samples:
    """Feed random complex64 arrays to every decoder.

    Decoders expect real-valued (float32) samples, so passing complex64
    is an intentional type mismatch. The decoder must either handle
    it gracefully or raise a controlled exception -- it must not segfault
    or produce an unhandled traceback.
    """

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(data=random_complex64_samples)
    @settings(max_examples=30, deadline=None)
    def test_complex64_no_crash(self, name, factory, data):
        """Decoder does not crash on random complex64 input."""
        decoder = factory()
        # The decoder may raise a TypeError/ValueError for complex input,
        # but it must not raise an unexpected exception or segfault.
        try:
            result = decoder.decode(data)
            assert isinstance(result, list)
        except (TypeError, ValueError):
            # Acceptable: decoder explicitly rejects complex input.
            pass


class TestRandomFloat32Samples:
    """Feed random float32 arrays to every decoder.

    This is the expected input dtype. Decoders must return a list and
    never crash, regardless of the content of the samples.
    """

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(data=random_float32_samples)
    @settings(max_examples=30, deadline=None)
    def test_float32_no_crash(self, name, factory, data):
        """Decoder returns a list (possibly empty) for random float32 input."""
        decoder = factory()
        result = decoder.decode(data)
        assert isinstance(result, list)


class TestRandomBytePatternsAsFloat32:
    """Reinterpret random bytes as float32.

    This produces arbitrary bit patterns including denormals, signaling NaN,
    quiet NaN, infinity, and very large/small magnitudes.
    """

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(data=random_bytes_as_float32)
    @settings(max_examples=30, deadline=None)
    def test_raw_bytes_as_float32_no_crash(self, name, factory, data):
        """Decoder does not crash on float32 data from random bytes."""
        decoder = factory()
        try:
            result = decoder.decode(data)
            assert isinstance(result, list)
        except (ValueError, FloatingPointError):
            # Acceptable if the decoder explicitly rejects bad float values.
            pass


# ---------------------------------------------------------------------------
# 2. Structured-but-invalid data
# ---------------------------------------------------------------------------


class TestStructuredInvalidData:
    """Feed data that looks somewhat like real protocol data but is
    intentionally broken: wrong CRC, truncated frames, corrupted sync
    words, etc.
    """

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_all_zeros(self, name, factory):
        """A buffer of all zeros must not crash any decoder."""
        decoder = factory()
        samples = np.zeros(10000, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_all_ones(self, name, factory):
        """A buffer of all ones (constant DC offset) must not crash."""
        decoder = factory()
        samples = np.ones(10000, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_alternating_high_low(self, name, factory):
        """Alternating +1/-1 pattern (simulates preamble-like data)."""
        decoder = factory()
        samples = np.array([1.0, -1.0] * 5000, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_square_wave_various_periods(self, name, factory):
        """Square wave at different periods to simulate wrong baud rates."""
        decoder = factory()
        for period in [2, 4, 8, 16, 50, 200]:
            half = period // 2
            one_cycle = np.concatenate([
                np.ones(half, dtype=np.float32),
                -np.ones(half, dtype=np.float32),
            ])
            samples = np.tile(one_cycle, 500)
            result = decoder.decode(samples)
            assert isinstance(result, list)

    def test_pocsag_corrupted_sync(self):
        """POCSAG sync word with flipped bits should not produce messages."""
        decoder = POCSAGDecoder(sample_rate=SAMPLE_RATE, baud_rate=1200)
        sps = int(SAMPLE_RATE / 1200)

        # Build a preamble (101010...) then a corrupted sync word
        preamble_bits = [1 if i % 2 == 0 else 0 for i in range(576)]
        # Real sync is 0x7CD215D8; flip several bits
        corrupted_sync = 0x7CD215D8 ^ 0xFF00FF00
        sync_bits = [(corrupted_sync >> (31 - i)) & 1 for i in range(32)]
        # Fill rest with random data
        rng = np.random.default_rng(42)
        filler_bits = [int(b) for b in rng.integers(0, 2, size=512)]
        all_bits = preamble_bits + sync_bits + filler_bits

        # Convert bits to samples
        samples = np.zeros(len(all_bits) * sps, dtype=np.float32)
        for i, bit in enumerate(all_bits):
            samples[i * sps : (i + 1) * sps] = 1.0 if bit else -1.0

        result = decoder.decode(samples)
        assert isinstance(result, list)
        # Corrupted sync should yield no valid messages
        assert len(result) == 0

    def test_adsb_truncated_message(self):
        """ADS-B decoder handles a preamble followed by too few data bits."""
        decoder = ADSBDecoder(sample_rate=SAMPLE_RATE)
        sps = int(SAMPLE_RATE / 1e6)

        # Build valid preamble pattern
        preamble_pattern = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        preamble_samples = np.zeros(16 * sps, dtype=np.float32)
        for i, val in enumerate(preamble_pattern):
            start = i * sps
            end = start + sps
            preamble_samples[start:end] = 1.0 if val else 0.1

        # Only 10 bits of message data instead of 56 or 112
        short_msg = np.full(10 * sps, 0.5, dtype=np.float32)
        signal = np.concatenate([
            np.full(50, 0.1, dtype=np.float32),
            preamble_samples,
            short_msg,
            np.full(50, 0.1, dtype=np.float32),
        ])

        result = decoder.decode(signal)
        assert isinstance(result, list)

    def test_adsb_wrong_crc(self):
        """ADS-B message with intentionally wrong CRC is rejected."""
        decoder = ADSBDecoder(sample_rate=SAMPLE_RATE)
        sps = int(SAMPLE_RATE / 1e6)

        preamble_pattern = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        preamble_samples = np.zeros(16 * sps, dtype=np.float32)
        for i, val in enumerate(preamble_pattern):
            start = i * sps
            end = start + sps
            preamble_samples[start:end] = 1.0 if val else 0.1

        # 112 bits of random data (will have wrong CRC with high probability)
        rng = np.random.default_rng(99)
        msg_bits = rng.integers(0, 2, size=112).tolist()
        msg_samples = np.zeros(112 * sps, dtype=np.float32)
        half = sps // 2
        for i, bit in enumerate(msg_bits):
            pos = i * sps
            if bit == 1:
                msg_samples[pos : pos + half] = 1.0
                msg_samples[pos + half : pos + sps] = 0.1
            else:
                msg_samples[pos : pos + half] = 0.1
                msg_samples[pos + half : pos + sps] = 1.0

        signal = np.concatenate([
            np.full(50, 0.1, dtype=np.float32),
            preamble_samples,
            msg_samples,
            np.full(200, 0.1, dtype=np.float32),
        ])

        result = decoder.decode(signal)
        assert isinstance(result, list)
        # DF=17 messages from random data should not pass CRC
        df17 = [m for m in result if m.downlink_format == 17]
        assert len(df17) == 0

    def test_ax25_no_closing_flag(self):
        """AX.25 frame with opening flag but no closing flag."""
        decoder = AX25Decoder(sample_rate=SAMPLE_RATE, baud_rate=1200)
        sps = int(SAMPLE_RATE / 1200)

        # Opening flag 0x7E = 01111110
        flag_bits = [0, 1, 1, 1, 1, 1, 1, 0]
        # Then random data (no closing flag)
        rng = np.random.default_rng(33)
        data_bits = [int(b) for b in rng.integers(0, 2, size=500)]
        all_bits = flag_bits * 3 + data_bits

        samples = np.zeros(len(all_bits) * sps, dtype=np.float32)
        for i, bit in enumerate(all_bits):
            samples[i * sps : (i + 1) * sps] = 1.0 if bit else -1.0

        result = decoder.decode(samples)
        assert isinstance(result, list)

    def test_rds_partial_group(self):
        """RDS decoder handles incomplete groups (fewer than 4 blocks)."""
        decoder = RDSDecoder(sample_rate=SAMPLE_RATE)
        # Just a short burst of random data -- not enough for a complete group
        rng = np.random.default_rng(77)
        samples = rng.standard_normal(500).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    def test_flex_corrupted_sync(self):
        """FLEX decoder with a sync-like pattern that has bit errors."""
        decoder = FLEXDecoder(sample_rate=SAMPLE_RATE, baud_rate=1600)
        sps = int(SAMPLE_RATE / 1600)

        # Corrupted FLEX sync pattern
        corrupted_sync1 = 0xA6C6AAAA ^ 0x0F0F0F0F
        sync_bits = [(corrupted_sync1 >> (31 - i)) & 1 for i in range(32)]
        # Second sync word also corrupted
        corrupted_sync2 = 0x5939AAAA ^ 0xF0F0F0F0
        sync2_bits = [(corrupted_sync2 >> (31 - i)) & 1 for i in range(32)]

        rng = np.random.default_rng(55)
        filler = [int(b) for b in rng.integers(0, 2, size=512)]
        all_bits = sync_bits + sync2_bits + filler

        samples = np.zeros(len(all_bits) * sps, dtype=np.float32)
        for i, bit in enumerate(all_bits):
            samples[i * sps : (i + 1) * sps] = 1.0 if bit else -1.0

        result = decoder.decode(samples)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_acars_no_soh(self):
        """ACARS decoder with preamble but no SOH character."""
        decoder = ACARSDecoder(sample_rate=SAMPLE_RATE)
        sps = int(SAMPLE_RATE / 2400)

        # Build preamble ++ (0x2B 0x2B) in NRZI bits, then garbage
        # We'll just create samples that encode 0x2B 0x2B followed by random
        rng = np.random.default_rng(11)
        random_bits = [int(b) for b in rng.integers(0, 2, size=800)]

        samples = np.zeros(len(random_bits) * sps, dtype=np.float32)
        for i, bit in enumerate(random_bits):
            samples[i * sps : (i + 1) * sps] = 1.0 if bit else -1.0

        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_impulse_noise(self, name, factory):
        """A signal with occasional extreme spikes must not crash decoders."""
        decoder = factory()
        rng = np.random.default_rng(88)
        samples = rng.standard_normal(5000).astype(np.float32) * 0.01
        # Insert random impulse spikes
        spike_positions = rng.integers(0, 5000, size=20)
        samples[spike_positions] = rng.choice(
            [1e6, -1e6, 1e10, -1e10, 1e38, -1e38], size=20
        )
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(
        num_chunks=st.integers(min_value=2, max_value=10),
        chunk_size=st.integers(min_value=10, max_value=2000),
    )
    @settings(max_examples=10, deadline=None)
    def test_successive_random_chunks(self, name, factory, num_chunks, chunk_size):
        """Successive calls to decode with random chunks must not crash."""
        decoder = factory()
        for _ in range(num_chunks):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            result = decoder.decode(chunk)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test boundary conditions and special floating-point values."""

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_empty_array(self, name, factory):
        """Empty array must return an empty list without crashing."""
        decoder = factory()
        result = decoder.decode(np.array([], dtype=np.float32))
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_single_sample(self, name, factory):
        """A single-sample array must not crash."""
        decoder = factory()
        result = decoder.decode(np.array([0.5], dtype=np.float32))
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_two_samples(self, name, factory):
        """Two samples -- minimum for any transition detection."""
        decoder = factory()
        result = decoder.decode(np.array([1.0, -1.0], dtype=np.float32))
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_very_large_values(self, name, factory):
        """Samples with very large magnitude must not crash."""
        decoder = factory()
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(1000).astype(np.float32) * 1e30
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_very_small_values(self, name, factory):
        """Samples with very small (subnormal) magnitude must not crash."""
        decoder = factory()
        samples = np.full(1000, 1e-38, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_all_nan(self, name, factory):
        """An array of all NaN values must not crash the decoder."""
        decoder = factory()
        samples = np.full(1000, np.nan, dtype=np.float32)
        try:
            result = decoder.decode(samples)
            assert isinstance(result, list)
        except (ValueError, FloatingPointError):
            # Acceptable: decoder explicitly rejects NaN input.
            pass

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_all_inf(self, name, factory):
        """An array of all +inf values must not crash the decoder."""
        decoder = factory()
        samples = np.full(1000, np.inf, dtype=np.float32)
        try:
            result = decoder.decode(samples)
            assert isinstance(result, list)
        except (ValueError, FloatingPointError, OverflowError):
            pass

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_all_negative_inf(self, name, factory):
        """An array of all -inf values must not crash the decoder."""
        decoder = factory()
        samples = np.full(1000, -np.inf, dtype=np.float32)
        try:
            result = decoder.decode(samples)
            assert isinstance(result, list)
        except (ValueError, FloatingPointError, OverflowError):
            pass

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_mixed_nan_inf_normal(self, name, factory):
        """Samples mixing NaN, inf, -inf, and normal values must not crash."""
        decoder = factory()
        rng = np.random.default_rng(12)
        samples = rng.standard_normal(1000).astype(np.float32)
        # Inject special values at random positions
        positions = rng.integers(0, 1000, size=30)
        specials = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1e38, -1e38,
                    np.finfo(np.float32).tiny, np.finfo(np.float32).max]
        for pos in positions:
            samples[pos] = rng.choice(specials)
        try:
            result = decoder.decode(samples)
            assert isinstance(result, list)
        except (ValueError, FloatingPointError):
            pass

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_negative_zero_array(self, name, factory):
        """An array of negative zeros must not crash."""
        decoder = factory()
        samples = np.array([-0.0] * 500, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_max_float32_values(self, name, factory):
        """Samples at the limits of float32 representation."""
        decoder = factory()
        finfo = np.finfo(np.float32)
        samples = np.array(
            [finfo.max, finfo.min, finfo.tiny, -finfo.tiny, finfo.eps, 0.0],
            dtype=np.float32,
        )
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_monotonically_increasing(self, name, factory):
        """Monotonically increasing ramp signal must not crash."""
        decoder = factory()
        samples = np.linspace(-1.0, 1.0, num=5000, dtype=np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_single_transition(self, name, factory):
        """Step function: half negative, half positive."""
        decoder = factory()
        samples = np.concatenate([
            np.full(2500, -1.0, dtype=np.float32),
            np.full(2500, 1.0, dtype=np.float32),
        ])
        result = decoder.decode(samples)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4. can_decode() robustness (confidence scoring)
# ---------------------------------------------------------------------------


class TestCanDecodeRobustness:
    """Test the can_decode() confidence scoring method if it exists.

    can_decode(samples) should return a float between 0.0 and 1.0
    representing confidence that the samples contain the protocol.
    It must never crash on arbitrary input.
    """

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_can_decode_exists(self, name, factory):
        """Verify can_decode method is present on the decoder."""
        decoder = factory()
        if not hasattr(decoder, "can_decode"):
            pytest.skip(f"{name} does not have a can_decode method")
        assert callable(decoder.can_decode)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(data=random_float32_samples)
    @settings(max_examples=20, deadline=None)
    def test_can_decode_random_returns_float(self, name, factory, data):
        """can_decode returns a float for random input."""
        decoder = factory()
        if not hasattr(decoder, "can_decode"):
            pytest.skip(f"{name} does not have a can_decode method")
        confidence = decoder.can_decode(data)
        assert isinstance(confidence, float)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_can_decode_empty(self, name, factory):
        """can_decode on empty input returns a float without crashing."""
        decoder = factory()
        if not hasattr(decoder, "can_decode"):
            pytest.skip(f"{name} does not have a can_decode method")
        confidence = decoder.can_decode(np.array([], dtype=np.float32))
        assert isinstance(confidence, float)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_can_decode_nan_input(self, name, factory):
        """can_decode on NaN input must not crash."""
        decoder = factory()
        if not hasattr(decoder, "can_decode"):
            pytest.skip(f"{name} does not have a can_decode method")
        samples = np.full(500, np.nan, dtype=np.float32)
        try:
            confidence = decoder.can_decode(samples)
            assert isinstance(confidence, float)
        except (ValueError, FloatingPointError):
            pass


# ---------------------------------------------------------------------------
# 5. Reset-then-decode stability
# ---------------------------------------------------------------------------


class TestResetStability:
    """Verify that calling reset() then decode() with garbage does not crash."""

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    @given(data=random_float32_samples)
    @settings(max_examples=15, deadline=None)
    def test_reset_then_decode(self, name, factory, data):
        """decode() after reset() must not crash on random data."""
        decoder = factory()
        # First pass
        decoder.decode(data)
        decoder.reset()
        # Second pass with same data
        result = decoder.decode(data)
        assert isinstance(result, list)

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_multiple_reset_cycles(self, name, factory):
        """Multiple reset-decode cycles with varying data must not crash."""
        decoder = factory()
        rng = np.random.default_rng(42)
        for _ in range(20):
            size = rng.integers(0, 3000)
            samples = rng.standard_normal(size).astype(np.float32)
            result = decoder.decode(samples)
            assert isinstance(result, list)
            decoder.reset()

    @pytest.mark.parametrize("name,factory", DECODER_FACTORIES, ids=_all_decoder_ids())
    def test_reset_on_fresh_decoder(self, name, factory):
        """Calling reset() on a brand-new decoder must not crash."""
        decoder = factory()
        decoder.reset()
        result = decoder.decode(np.zeros(100, dtype=np.float32))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 6. Decoder-specific targeted fuzz tests
# ---------------------------------------------------------------------------


class TestPOCSAGFuzz:
    """POCSAG-specific fuzz tests."""

    @given(
        length=st.integers(min_value=100, max_value=8000),
        amplitude=st.floats(min_value=0.001, max_value=100.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_random_amplitude_scaling(self, length, amplitude):
        """POCSAG decoder handles various signal amplitudes."""
        decoder = POCSAGDecoder(sample_rate=SAMPLE_RATE, baud_rate=1200)
        samples = np.random.randn(length).astype(np.float32) * amplitude
        result = decoder.decode(samples)
        assert isinstance(result, list)

    @given(baud=st.sampled_from([512, 1200, 2400]))
    @settings(max_examples=10, deadline=None)
    def test_all_baud_rates(self, baud):
        """POCSAG decoder at each standard baud rate handles garbage."""
        decoder = POCSAGDecoder(sample_rate=SAMPLE_RATE, baud_rate=baud)
        samples = np.random.randn(5000).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)


class TestADSBFuzz:
    """ADS-B-specific fuzz tests."""

    @given(length=st.integers(min_value=500, max_value=50000))
    @settings(max_examples=20, deadline=None)
    def test_varying_lengths(self, length):
        """ADS-B decoder handles various input lengths."""
        decoder = ADSBDecoder(sample_rate=SAMPLE_RATE)
        samples = np.random.randn(length).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)

    def test_repeated_preamble_pattern(self):
        """Repeated preamble-like patterns must not cause infinite loops."""
        decoder = ADSBDecoder(sample_rate=SAMPLE_RATE)
        sps = int(SAMPLE_RATE / 1e6)
        pattern = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        one_preamble = np.zeros(16 * sps, dtype=np.float32)
        for i, val in enumerate(pattern):
            one_preamble[i * sps : (i + 1) * sps] = 1.0 if val else 0.1
        # Repeat preamble many times without valid message data
        signal = np.tile(one_preamble, 50)
        result = decoder.decode(signal)
        assert isinstance(result, list)


class TestRDSFuzz:
    """RDS-specific fuzz tests."""

    @given(length=st.integers(min_value=100, max_value=20000))
    @settings(max_examples=20, deadline=None)
    def test_varying_lengths(self, length):
        """RDS decoder handles various input lengths."""
        decoder = RDSDecoder(sample_rate=SAMPLE_RATE)
        samples = np.random.randn(length).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)


class TestAX25Fuzz:
    """AX.25-specific fuzz tests."""

    @given(
        num_flags=st.integers(min_value=1, max_value=50),
        gap_size=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_many_flags_with_gaps(self, num_flags, gap_size):
        """Multiple flag sequences interspersed with random data."""
        decoder = AX25Decoder(sample_rate=SAMPLE_RATE, baud_rate=1200)
        sps = int(SAMPLE_RATE / 1200)
        flag_bits = [0, 1, 1, 1, 1, 1, 1, 0]
        all_bits = []
        rng = np.random.default_rng(77)
        for _ in range(num_flags):
            all_bits.extend(flag_bits)
            gap = [int(b) for b in rng.integers(0, 2, size=gap_size)]
            all_bits.extend(gap)

        if not all_bits:
            return

        samples = np.zeros(len(all_bits) * sps, dtype=np.float32)
        for i, bit in enumerate(all_bits):
            samples[i * sps : (i + 1) * sps] = 1.0 if bit else -1.0

        result = decoder.decode(samples)
        assert isinstance(result, list)


class TestACARSFuzz:
    """ACARS-specific fuzz tests."""

    @given(length=st.integers(min_value=100, max_value=20000))
    @settings(max_examples=20, deadline=None)
    def test_varying_lengths(self, length):
        """ACARS decoder handles various input lengths."""
        decoder = ACARSDecoder(sample_rate=SAMPLE_RATE)
        samples = np.random.randn(length).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)


class TestFLEXFuzz:
    """FLEX-specific fuzz tests."""

    @given(baud=st.sampled_from([1600, 3200, 6400]))
    @settings(max_examples=10, deadline=None)
    def test_all_baud_rates(self, baud):
        """FLEX decoder at each standard baud rate handles garbage."""
        decoder = FLEXDecoder(sample_rate=SAMPLE_RATE, baud_rate=baud)
        samples = np.random.randn(5000).astype(np.float32)
        result = decoder.decode(samples)
        assert isinstance(result, list)
