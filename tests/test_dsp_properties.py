"""Property-based tests for DSP invariants using Hypothesis.

Validates mathematical properties that must hold for any input:
- FIR filter structural and frequency-domain properties
- Demodulator round-trip fidelity (FM, AM)
- Conversion function round-trips (dB, frequency strings)
- Sample buffer integrity under arbitrary write/read sequences
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sdr_module.dsp.filters import FIRFilter, FilterSpec, FilterType
from sdr_module.dsp.demodulators import AMDemodulator, FMDemodulator
from sdr_module.utils.conversions import linear_to_db, db_to_linear, freq_to_str
from sdr_module.core.sample_buffer import SampleBuffer, BufferOverflowPolicy


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Odd tap counts keep the filter symmetric about its center sample.
odd_tap_counts = st.sampled_from([31, 51, 63, 101, 127])

# Window functions supported by FIRFilter._get_window
window_names = st.sampled_from(["hamming", "hann", "blackman", "kaiser", "rectangular"])


# ---------------------------------------------------------------------------
# 1. FIR filter properties
# ---------------------------------------------------------------------------

class TestFIRFilterProperties:
    """Property-based tests for FIR filter invariants."""

    @given(
        cutoff=st.floats(min_value=500.0, max_value=4000.0),
        num_taps=odd_tap_counts,
        window=window_names,
    )
    @settings(max_examples=50, deadline=None)
    def test_output_length_matches_input(self, cutoff, num_taps, window):
        """Filtering with mode='same' must preserve input length."""
        sample_rate = 10000.0
        assume(cutoff < sample_rate / 2)

        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=sample_rate,
            num_taps=num_taps,
            window=window,
        )
        fir = FIRFilter(spec)

        n_samples = 256
        signal = np.random.randn(n_samples)
        output = fir.filter(signal)

        assert output.shape == signal.shape, (
            f"Output length {output.shape} != input length {signal.shape}"
        )

    @given(
        cutoff=st.floats(min_value=500.0, max_value=4000.0),
        num_taps=odd_tap_counts,
        window=window_names,
    )
    @settings(max_examples=50, deadline=None)
    def test_taps_are_real_valued(self, cutoff, num_taps, window):
        """FIR filter taps designed by windowed-sinc must be real-valued."""
        sample_rate = 10000.0
        assume(cutoff < sample_rate / 2)

        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=sample_rate,
            num_taps=num_taps,
            window=window,
        )
        fir = FIRFilter(spec)
        taps = fir.taps

        assert np.isrealobj(taps), "Filter taps must be real-valued"
        assert not np.any(np.isnan(taps)), "Filter taps must not contain NaN"
        assert not np.any(np.isinf(taps)), "Filter taps must not contain Inf"

    @given(
        cutoff=st.floats(min_value=500.0, max_value=4000.0),
        num_taps=odd_tap_counts,
    )
    @settings(max_examples=50, deadline=None)
    def test_symmetric_taps_for_linear_phase(self, cutoff, num_taps):
        """Windowed-sinc lowpass filters with symmetric windows produce
        symmetric (linear-phase) tap coefficients."""
        sample_rate = 10000.0
        assume(cutoff < sample_rate / 2)

        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=sample_rate,
            num_taps=num_taps,
            window="hamming",
        )
        fir = FIRFilter(spec)
        taps = fir.taps

        np.testing.assert_allclose(
            taps,
            taps[::-1],
            atol=1e-10,
            err_msg="Lowpass filter taps must be symmetric for linear phase",
        )

    @given(
        cutoff=st.floats(min_value=500.0, max_value=4000.0),
        num_taps=odd_tap_counts,
        window=window_names,
    )
    @settings(max_examples=50, deadline=None)
    def test_lowpass_passes_dc(self, cutoff, num_taps, window):
        """A lowpass filter must have approximately unity gain at DC.

        DC gain equals the sum of all taps. After normalization the
        design code ensures sum(taps) == 1.0.
        """
        sample_rate = 10000.0
        assume(cutoff < sample_rate / 2)

        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=sample_rate,
            num_taps=num_taps,
            window=window,
        )
        fir = FIRFilter(spec)
        taps = fir.taps

        dc_gain = np.sum(taps)
        assert abs(dc_gain - 1.0) < 0.01, (
            f"DC gain should be ~1.0, got {dc_gain}"
        )


# ---------------------------------------------------------------------------
# 2. Demodulator round-trips
# ---------------------------------------------------------------------------

class TestDemodulatorRoundTrips:
    """Property-based tests for modulate-then-demodulate fidelity."""

    @given(
        tone_freq=st.floats(min_value=200.0, max_value=3000.0),
        deviation=st.floats(min_value=5000.0, max_value=75000.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_fm_round_trip_peak_frequency(self, tone_freq, deviation):
        """FM-modulate a single tone, demodulate, and verify the peak
        frequency of the recovered baseband matches the original tone
        within a reasonable tolerance.
        """
        sample_rate = 256000.0
        duration = 0.05  # 50 ms gives adequate spectral resolution
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples) / sample_rate

        # Generate baseband tone and FM-modulate it
        baseband = np.cos(2 * np.pi * tone_freq * t)
        phase = 2 * np.pi * deviation * np.cumsum(baseband) / sample_rate
        iq_signal = np.exp(1j * phase)

        # Demodulate
        demod = FMDemodulator(sample_rate=sample_rate, max_deviation=deviation)
        recovered = demod.demodulate(iq_signal)

        # Trim edges to avoid transient artefacts
        trim = len(recovered) // 10
        recovered_trimmed = recovered[trim:-trim]

        # Find peak frequency via FFT
        spectrum = np.abs(np.fft.rfft(recovered_trimmed))
        freqs = np.fft.rfftfreq(len(recovered_trimmed), 1.0 / sample_rate)
        peak_idx = np.argmax(spectrum[1:]) + 1  # skip DC bin
        detected_freq = freqs[peak_idx]

        freq_resolution = sample_rate / len(recovered_trimmed)
        tolerance = max(freq_resolution * 2, tone_freq * 0.05)

        assert abs(detected_freq - tone_freq) < tolerance, (
            f"Detected {detected_freq:.1f} Hz, expected {tone_freq:.1f} Hz "
            f"(tolerance {tolerance:.1f} Hz)"
        )

    @given(
        mod_depth=st.floats(min_value=0.3, max_value=0.9),
        tone_freq=st.floats(min_value=300.0, max_value=3000.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_am_round_trip_envelope_recovered(self, mod_depth, tone_freq):
        """AM-modulate a tone onto a carrier, demodulate with envelope
        detection, and verify the dominant frequency of the recovered
        envelope matches the modulating tone.
        """
        sample_rate = 48000.0
        duration = 0.1
        n_samples = int(sample_rate * duration)
        t = np.arange(n_samples) / sample_rate

        # AM modulation: carrier * (1 + m * cos(2*pi*f*t))
        carrier_freq = 10000.0
        modulating = np.cos(2 * np.pi * tone_freq * t)
        envelope = 1.0 + mod_depth * modulating
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        am_signal = (envelope * carrier).astype(np.complex128)

        # Demodulate (disable DC block so we get the raw envelope)
        demod = AMDemodulator(sample_rate=sample_rate, dc_block=False)
        recovered = demod.demodulate(am_signal)

        # Remove DC (mean) and analyse spectrum
        recovered_ac = recovered - np.mean(recovered)

        # Trim edges
        trim = len(recovered_ac) // 10
        recovered_trimmed = recovered_ac[trim:-trim]

        spectrum = np.abs(np.fft.rfft(recovered_trimmed))
        freqs = np.fft.rfftfreq(len(recovered_trimmed), 1.0 / sample_rate)
        peak_idx = np.argmax(spectrum[1:]) + 1
        detected_freq = freqs[peak_idx]

        freq_resolution = sample_rate / len(recovered_trimmed)
        tolerance = max(freq_resolution * 2, tone_freq * 0.05)

        assert abs(detected_freq - tone_freq) < tolerance, (
            f"Detected {detected_freq:.1f} Hz, expected {tone_freq:.1f} Hz "
            f"(tolerance {tolerance:.1f} Hz)"
        )


# ---------------------------------------------------------------------------
# 3. Conversion round-trips
# ---------------------------------------------------------------------------

class TestConversionRoundTrips:
    """Property-based tests for unit-conversion invertibility."""

    @given(
        x=st.floats(min_value=-200.0, max_value=200.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_linear_db_round_trip(self, x):
        """linear_to_db(db_to_linear(x)) should approximately equal x.

        The dB range is limited to [-200, 200] so that 10^(x/10)
        remains representable as a float64.

        Note: db_to_linear uses power convention (10^(x/10)) and
        linear_to_db uses 10*log10, so the round-trip is exact up to
        the 1e-20 floor constant in linear_to_db.
        """
        linear_val = db_to_linear(x)
        assume(np.isfinite(linear_val))
        assume(linear_val > 1e-15)  # avoid floor-constant domination

        round_tripped = linear_to_db(linear_val)
        assert np.isfinite(round_tripped), "Round-tripped value must be finite"

        np.testing.assert_allclose(
            round_tripped,
            x,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"linear_to_db(db_to_linear({x})) != {x}",
        )

    @given(
        freq=st.floats(min_value=0.1, max_value=30e9),
    )
    @settings(max_examples=50, deadline=None)
    def test_freq_to_str_produces_valid_string(self, freq):
        """freq_to_str must return a non-empty string containing a
        recognised unit suffix for any positive frequency.
        """
        result = freq_to_str(freq)
        assert isinstance(result, str), "freq_to_str must return a string"
        assert len(result) > 0, "freq_to_str must return a non-empty string"

        valid_suffixes = ("Hz", "kHz", "MHz", "GHz")
        assert any(result.endswith(suffix) for suffix in valid_suffixes), (
            f"freq_to_str({freq}) = '{result}' does not end with a "
            f"recognised unit suffix"
        )


# ---------------------------------------------------------------------------
# 4. Sample buffer properties
# ---------------------------------------------------------------------------

class TestSampleBufferProperties:
    """Property-based tests for SampleBuffer integrity."""

    @given(
        n_samples=st.integers(min_value=1, max_value=4096),
    )
    @settings(max_examples=50, deadline=None)
    def test_write_then_read_data_matches(self, n_samples):
        """Writing N samples then reading N samples must return
        identical data.
        """
        capacity = max(n_samples, 8192)
        buf = SampleBuffer(
            capacity=capacity,
            overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
        )

        # Generate deterministic complex samples
        real_part = np.arange(n_samples, dtype=np.float32)
        imag_part = np.arange(n_samples, dtype=np.float32) * 0.5
        samples = (real_part + 1j * imag_part).astype(np.complex64)

        written = buf.write(samples)
        assert written == n_samples, (
            f"Expected to write {n_samples}, wrote {written}"
        )

        read_back = buf.read(n_samples, timeout=0)
        assert read_back is not None, "Read returned None unexpectedly"
        assert len(read_back) == n_samples, (
            f"Read back {len(read_back)} samples, expected {n_samples}"
        )
        np.testing.assert_array_equal(
            read_back,
            samples,
            err_msg="Read-back data does not match written data",
        )

    @given(
        capacity=st.integers(min_value=16, max_value=4096),
        write_size=st.integers(min_value=1, max_value=8192),
        n_writes=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_buffer_never_exceeds_capacity(self, capacity, write_size, n_writes):
        """Regardless of how many samples are written, the number of
        available samples in the buffer must never exceed its capacity.
        """
        buf = SampleBuffer(
            capacity=capacity,
            overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
        )

        for _ in range(n_writes):
            samples = np.zeros(write_size, dtype=np.complex64)
            buf.write(samples)

            assert buf.available <= capacity, (
                f"Buffer available ({buf.available}) exceeds "
                f"capacity ({capacity})"
            )

        stats = buf.stats
        assert stats.current_fill <= capacity, (
            f"Buffer fill ({stats.current_fill}) exceeds capacity ({capacity})"
        )

    @given(
        capacity=st.integers(min_value=64, max_value=2048),
        n_writes=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_drop_newest_never_exceeds_capacity(self, capacity, n_writes):
        """With DROP_NEWEST policy the buffer must also respect its
        capacity bound and discard incoming samples that do not fit.
        """
        buf = SampleBuffer(
            capacity=capacity,
            overflow_policy=BufferOverflowPolicy.DROP_NEWEST,
        )

        for _ in range(n_writes):
            chunk = np.ones(capacity, dtype=np.complex64)
            buf.write(chunk)

            assert buf.available <= capacity, (
                f"Buffer available ({buf.available}) exceeds "
                f"capacity ({capacity}) under DROP_NEWEST"
            )

    @given(
        capacity=st.integers(min_value=32, max_value=1024),
        write_size=st.integers(min_value=1, max_value=512),
    )
    @settings(max_examples=50, deadline=None)
    def test_stats_bookkeeping_consistent(self, capacity, write_size):
        """After a series of writes, the buffer stats must satisfy:
        - current_fill + total_samples_out <= total_samples_in
          (you cannot read out or retain more than was admitted)
        - current_fill <= capacity (already tested, but reinforced here)
        """
        buf = SampleBuffer(
            capacity=capacity,
            overflow_policy=BufferOverflowPolicy.DROP_OLDEST,
        )

        for _ in range(3):
            chunk = np.zeros(write_size, dtype=np.complex64)
            buf.write(chunk)

        stats = buf.stats
        assert stats.current_fill + stats.total_samples_out <= stats.total_samples_in, (
            f"current_fill ({stats.current_fill}) + "
            f"total_samples_out ({stats.total_samples_out}) > "
            f"total_samples_in ({stats.total_samples_in})"
        )
        assert stats.current_fill <= capacity, (
            f"current_fill ({stats.current_fill}) exceeds "
            f"capacity ({capacity})"
        )
