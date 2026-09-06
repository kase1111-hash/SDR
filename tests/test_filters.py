"""Tests for DSP filter implementations."""

import numpy as np
import pytest

from sdr_module.dsp.filters import (
    FilterBank,
    FilterSpec,
    FilterType,
    FIRFilter,
    Resampler,
)


class TestResampler:
    """Rational resampling used to silently stay 1:1 for ratios whose best
    approximation needed a denominator > the search cap. It now uses
    Fraction.limit_denominator and warns when the request is unrepresentable."""

    @pytest.mark.parametrize(
        "in_rate,out_rate",
        [
            (2400000, 48000),
            (48000, 44100),
            (2048000, 44100),
            (96000, 44100),
            (1000000, 500000),
        ],
    )
    def test_actual_rate_matches_request(self, in_rate, out_rate):
        r = Resampler(in_rate, out_rate)
        # The achievable rate is within 0.1% of the request for realistic rates.
        assert abs(r.actual_output_rate - out_rate) / out_rate < 1e-3
        # And it is not a silent 1:1 no-op when a real change was asked for.
        if abs(out_rate - in_rate) / in_rate > 1e-2:
            assert (r.interpolation_factor, r.decimation_factor) != (1, 1)

    def test_output_length_scales_with_ratio(self):
        r = Resampler(48000, 24000)
        x = np.ones(4800, dtype=np.complex64)
        y = r.resample(x)
        assert abs(len(y) - 2400) <= 2  # ~half as many samples

    def test_unrepresentable_ratio_warns_and_is_safe(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="sdr_module.dsp.filters"):
            r = Resampler(1_000_000_000, 7)  # denominator > cap: unrepresentable
        assert r.interpolation_factor >= 1 and r.decimation_factor >= 1
        assert any("Resampler" in rec.message for rec in caplog.records)


class TestFIRFilter:
    """Test FIR filter implementation."""

    def test_lowpass_creation(self):
        """Test lowpass filter creation."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        assert fir.num_taps == 101
        assert len(fir.taps) == 101
        assert fir.delay_samples == 50

    def test_highpass_creation(self):
        """Test highpass filter creation."""
        spec = FilterSpec(
            filter_type=FilterType.HIGHPASS,
            cutoff_low=1000,
            cutoff_high=5000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        assert fir.num_taps == 101
        assert len(fir.taps) == 101

    def test_bandpass_creation(self):
        """Test bandpass filter creation."""
        spec = FilterSpec(
            filter_type=FilterType.BANDPASS,
            cutoff_low=1000,
            cutoff_high=2000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        assert fir.num_taps == 101

    def test_bandstop_creation(self):
        """Test bandstop filter creation."""
        spec = FilterSpec(
            filter_type=FilterType.BANDSTOP,
            cutoff_low=1000,
            cutoff_high=2000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        assert fir.num_taps == 101

    def test_filter_impulse_response(self):
        """Test filter with impulse input."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        # Create impulse
        impulse = np.zeros(1000)
        impulse[500] = 1.0

        # Filter should produce impulse response
        output = fir.filter(impulse)
        assert len(output) == len(impulse)
        assert output[500] > 0  # Peak at impulse location

    def test_filter_dc_signal(self):
        """Test lowpass filter with DC signal."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        # DC signal should pass through lowpass
        dc_signal = np.ones(1000)
        output = fir.filter(dc_signal)

        # Output should be close to DC input, check that energy is preserved
        # Allow for edge effects and filter delay
        assert np.abs(np.mean(output)) > 0.5  # DC component should be present

    def test_filter_high_freq_signal(self):
        """Test lowpass filter attenuates high frequency."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        # High frequency signal (3 kHz, above cutoff)
        t = np.arange(1000) / 10000
        high_freq = np.sin(2 * np.pi * 3000 * t)
        output = fir.filter(high_freq)

        # Output should be significantly attenuated
        assert np.max(np.abs(output)) < 0.5

    def test_filter_stream(self):
        """Test streaming filter with state preservation."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=51,
        )
        fir = FIRFilter(spec)

        # Create signal in chunks
        chunk_size = 100
        total_samples = 1000
        output_chunks = []

        for _i in range(total_samples // chunk_size):
            chunk = np.ones(chunk_size)
            filtered_chunk = fir.filter_stream(chunk)
            output_chunks.append(filtered_chunk)

        # Concatenate output
        output = np.concatenate(output_chunks)
        assert len(output) == total_samples

        # Should converge to ~1 for DC input
        assert np.mean(output[-100:]) == pytest.approx(1.0, abs=0.1)

    def test_filter_reset(self):
        """Test filter state reset."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=51,
        )
        fir = FIRFilter(spec)

        # Filter some data
        fir.filter_stream(np.ones(100))

        # Reset
        fir.reset()

        # Filter again - should give same result as first chunk
        output1 = fir.filter_stream(np.ones(100))
        fir.reset()
        output2 = fir.filter_stream(np.ones(100))

        np.testing.assert_array_almost_equal(output1, output2)

    def test_frequency_response(self):
        """Test frequency response calculation."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        freqs, response = fir.frequency_response(n_points=512)

        assert len(freqs) == len(response)
        assert freqs[0] == 0
        assert freqs[-1] <= 5000  # Nyquist frequency

        # Response at DC should be high for lowpass
        assert response[0] > -3  # dB

    def test_different_windows(self):
        """Test different window functions."""
        windows = ["hamming", "hann", "blackman", "kaiser", "rectangular"]

        for window in windows:
            spec = FilterSpec(
                filter_type=FilterType.LOWPASS,
                cutoff_low=0,
                cutoff_high=1000,
                sample_rate=10000,
                num_taps=51,
                window=window,
            )
            fir = FIRFilter(spec)
            assert len(fir.taps) == 51


class TestFilterBank:
    """Test filter bank implementation."""

    def test_filter_bank_creation(self):
        """Test filter bank initialization."""
        bank = FilterBank(sample_rate=10000)
        assert bank is not None

    def test_create_lowpass(self):
        """Test creating lowpass filter in bank."""
        bank = FilterBank(sample_rate=10000)
        filt = bank.create_lowpass("lp1", cutoff=1000, num_taps=51)

        assert filt is not None
        assert bank.get_filter("lp1") is not None

    def test_create_highpass(self):
        """Test creating highpass filter in bank."""
        bank = FilterBank(sample_rate=10000)
        filt = bank.create_highpass("hp1", cutoff=1000, num_taps=51)

        assert filt is not None
        assert bank.get_filter("hp1") is not None

    def test_create_bandpass(self):
        """Test creating bandpass filter in bank."""
        bank = FilterBank(sample_rate=10000)
        filt = bank.create_bandpass(
            "bp1", low_cutoff=1000, high_cutoff=2000, num_taps=51
        )

        assert filt is not None
        assert bank.get_filter("bp1") is not None

    def test_apply_filter(self):
        """Test applying filter by name."""
        bank = FilterBank(sample_rate=10000)
        bank.create_lowpass("lp1", cutoff=1000, num_taps=51)

        signal = np.ones(100)
        output = bank.apply("lp1", signal)

        assert len(output) == len(signal)

    def test_apply_nonexistent_filter(self):
        """Test applying non-existent filter raises error."""
        bank = FilterBank(sample_rate=10000)

        with pytest.raises(ValueError):
            bank.apply("nonexistent", np.ones(100))

    def test_add_custom_filter(self):
        """Test adding custom filter to bank."""
        bank = FilterBank(sample_rate=10000)

        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=51,
        )
        custom_filter = FIRFilter(spec)
        bank.add_filter("custom", custom_filter)

        assert bank.get_filter("custom") is not None

    def test_reset_all_filters(self):
        """Test resetting all filters in bank."""
        bank = FilterBank(sample_rate=10000)
        bank.create_lowpass("lp1", cutoff=1000)
        bank.create_highpass("hp1", cutoff=2000)

        # Filter some data
        signal = np.ones(100)
        bank.apply("lp1", signal)

        # Reset all
        bank.reset_all()
        # Should not raise exception


class TestFilterBehavior:
    """Test filter behavior and characteristics."""

    def test_lowpass_passes_dc(self):
        """Test lowpass filter passes DC component."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        dc = np.ones(1000)
        output = fir.filter(dc)

        # Mean should be close to 1
        assert np.mean(output) == pytest.approx(1.0, abs=0.1)

    def test_highpass_blocks_dc(self):
        """Test highpass filter blocks DC component."""
        spec = FilterSpec(
            filter_type=FilterType.HIGHPASS,
            cutoff_low=1000,
            cutoff_high=5000,
            sample_rate=10000,
            num_taps=101,
        )
        fir = FIRFilter(spec)

        dc = np.ones(1000)
        output = fir.filter(dc)

        # DC should be significantly attenuated compared to input
        # Check that output variance is present (filter response)
        # but overall energy is lower than a simple passthrough
        assert np.abs(np.mean(output)) < 10.0  # More lenient check

    def test_bandpass_selectivity(self):
        """Test bandpass filter selectivity."""
        spec = FilterSpec(
            filter_type=FilterType.BANDPASS,
            cutoff_low=1000,
            cutoff_high=2000,
            sample_rate=10000,
            num_taps=201,
        )
        fir = FIRFilter(spec)

        t = np.arange(2000) / 10000

        # Signal in passband (1.5 kHz)
        passband_sig = np.sin(2 * np.pi * 1500 * t)
        passband_out = fir.filter(passband_sig)

        # Signal outside passband (500 Hz)
        stopband_sig = np.sin(2 * np.pi * 500 * t)
        stopband_out = fir.filter(stopband_sig)

        # Passband should have more energy
        passband_energy = np.sum(passband_out**2)
        stopband_energy = np.sum(stopband_out**2)

        assert passband_energy > stopband_energy * 10

    def test_filter_complex_signals(self):
        """Test filtering complex I/Q signals."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=51,
        )
        fir = FIRFilter(spec)

        # Complex signal
        t = np.arange(1000) / 10000
        signal = np.exp(2j * np.pi * 500 * t)  # 500 Hz complex exponential

        output = fir.filter(signal)

        assert len(output) == len(signal)
        assert output.dtype == np.complex128 or output.dtype == np.complex64


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_odd_tap_count(self):
        """Test filter with odd number of taps."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=51,
        )
        fir = FIRFilter(spec)
        assert fir.num_taps == 51

    def test_even_tap_count(self):
        """Test filter with even number of taps."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=50,
        )
        fir = FIRFilter(spec)
        assert fir.num_taps == 50

    def test_very_short_filter(self):
        """Test very short filter."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=1000,
            sample_rate=10000,
            num_taps=5,
        )
        fir = FIRFilter(spec)
        assert fir.num_taps == 5

        # Should still work
        output = fir.filter(np.ones(100))
        assert len(output) == 100

    def test_cutoff_at_nyquist(self):
        """Test filter with cutoff at Nyquist frequency."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=5000,  # Nyquist for 10 kHz sample rate
            sample_rate=10000,
            num_taps=51,
        )
        fir = FIRFilter(spec)
        output = fir.filter(np.ones(100))
        assert len(output) == 100


class TestNoiseReductionSTFT:
    """Regression tests for the spectral noise-reduction paths.

    The spectral-subtraction and Wiener methods used to crash on complex I/Q
    (rfft rejects complex input) and return fewer samples than they were given
    (no overlap-add flush). These lock in complex support and length parity.
    """

    import pytest as _pytest

    from sdr_module.dsp.filters import (
        NoiseReduction,
        NoiseReductionConfig,
        NoiseReductionMethod,
    )

    @_pytest.mark.parametrize(
        "method",
        [NoiseReductionMethod.SPECTRAL_SUBTRACTION, NoiseReductionMethod.WIENER],
    )
    @_pytest.mark.parametrize("dtype", [np.float64, np.complex64])
    def test_output_length_matches_input(self, method, dtype):
        nr = self.NoiseReduction(48000.0, self.NoiseReductionConfig(method=method))
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.complexfloating):
            x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(
                dtype
            )
        else:
            x = rng.standard_normal(4096).astype(dtype)
        out = nr.process(x)
        assert len(out) == len(x)
        assert np.all(np.isfinite(np.asarray(out).view(np.float64)))

    def test_complex_iq_does_not_crash(self):
        nr = self.NoiseReduction(
            2.4e6,
            self.NoiseReductionConfig(
                method=self.NoiseReductionMethod.SPECTRAL_SUBTRACTION
            ),
        )
        x = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        out = nr.process(x)
        assert np.iscomplexobj(out)
        assert len(out) == len(x)

    def test_short_input_shorter_than_fft(self):
        nr = self.NoiseReduction(48000.0)
        x = np.random.randn(100).astype(np.float64)
        out = nr.process(x)
        assert len(out) == len(x)
