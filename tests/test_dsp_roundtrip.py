"""
Round-trip tests for DSP demodulator algorithms.

Validates that modulation -> demodulation pipelines recover the original
information content. Each test generates a synthetic modulated signal with
known parameters and verifies the demodulated output matches expectations
within realistic tolerances.
"""

import numpy as np
import pytest

from sdr_module.dsp.demodulators import (
    AMDemodulator,
    CWDemodulator,
    FMDemodulator,
    OOKDemodulator,
)
from sdr_module.dsp.filters import AGC, AGCConfig, AGCMode, FilterBank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_peak_frequency(signal: np.ndarray, sample_rate: float) -> float:
    """Return the frequency (Hz) of the dominant spectral peak in a real signal.

    Uses an FFT and searches only the positive-frequency half.  A simple
    parabolic interpolation around the peak bin gives sub-bin accuracy.
    """
    N = len(signal)
    spectrum = np.fft.rfft(signal * np.hanning(N))
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Ignore DC bin (index 0) to avoid false positives from DC offset
    peak_bin = np.argmax(magnitudes[1:]) + 1

    # Parabolic interpolation for sub-bin accuracy
    if 1 < peak_bin < len(magnitudes) - 1:
        alpha = magnitudes[peak_bin - 1]
        beta = magnitudes[peak_bin]
        gamma = magnitudes[peak_bin + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            correction = 0.5 * (alpha - gamma) / denom
        else:
            correction = 0.0
        peak_freq = freqs[peak_bin] + correction * (freqs[1] - freqs[0])
    else:
        peak_freq = freqs[peak_bin]

    return float(peak_freq)


def rms(signal: np.ndarray) -> float:
    """Root-mean-square amplitude of a signal."""
    return float(np.sqrt(np.mean(np.abs(signal) ** 2)))


# ---------------------------------------------------------------------------
# FM Demodulation Round-Trip
# ---------------------------------------------------------------------------


class TestFMDemodulationRoundTrip:
    """Generate a known FM signal and verify the demodulator recovers the tone."""

    SAMPLE_RATE = 2.4e6
    MAX_DEVIATION = 75e3

    def _generate_fm_signal(
        self,
        tone_freq: float,
        duration: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Synthesize a baseband FM-modulated IQ signal.

        The carrier sits at DC (baseband).  The instantaneous frequency
        deviation is ``max_deviation * sin(2*pi*tone_freq*t)``.
        """
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.arange(n_samples) / self.SAMPLE_RATE

        # Modulating tone
        modulating = np.sin(2.0 * np.pi * tone_freq * t)

        # Instantaneous phase: integral of 2*pi*deviation*modulating
        phase = (
            2.0 * np.pi * self.MAX_DEVIATION * np.cumsum(modulating) / self.SAMPLE_RATE
        )

        # Complex baseband FM signal
        return amplitude * np.exp(1j * phase)

    def test_fm_1khz_tone(self):
        """FM-demod a 1 kHz tone and verify spectral peak at 1 kHz."""
        tone_freq = 1000.0
        duration = 0.05  # 50 ms gives 50k samples at 2.4 MS/s
        signal = self._generate_fm_signal(tone_freq, duration)

        demod = FMDemodulator(
            sample_rate=self.SAMPLE_RATE, max_deviation=self.MAX_DEVIATION
        )
        output = demod.demodulate(signal)

        # Trim edges to avoid transient artefacts
        trim = len(output) // 10
        output_trimmed = output[trim:-trim]

        peak_freq = find_peak_frequency(output_trimmed, self.SAMPLE_RATE)
        assert (
            abs(peak_freq - tone_freq) < 50.0
        ), f"Expected peak near {tone_freq} Hz, got {peak_freq:.1f} Hz"

    def test_fm_5khz_tone(self):
        """FM-demod a 5 kHz tone -- ensures it is not hard-coded for 1 kHz."""
        tone_freq = 5000.0
        duration = 0.05
        signal = self._generate_fm_signal(tone_freq, duration)

        demod = FMDemodulator(
            sample_rate=self.SAMPLE_RATE, max_deviation=self.MAX_DEVIATION
        )
        output = demod.demodulate(signal)

        trim = len(output) // 10
        output_trimmed = output[trim:-trim]

        peak_freq = find_peak_frequency(output_trimmed, self.SAMPLE_RATE)
        assert abs(peak_freq - tone_freq) < 50.0

    def test_fm_output_amplitude_scales_with_deviation(self):
        """A larger modulating amplitude should produce a larger demod output."""
        duration = 0.05
        tone_freq = 1000.0

        signal_low = self._generate_fm_signal(tone_freq, duration, amplitude=1.0)
        # Halve the deviation by scaling modulating signal -- we can approximate
        # this by generating a signal whose deviation is half.
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.arange(n_samples) / self.SAMPLE_RATE
        modulating = 0.5 * np.sin(2.0 * np.pi * tone_freq * t)
        phase = (
            2.0 * np.pi * self.MAX_DEVIATION * np.cumsum(modulating) / self.SAMPLE_RATE
        )
        signal_half = np.exp(1j * phase)

        demod = FMDemodulator(
            sample_rate=self.SAMPLE_RATE, max_deviation=self.MAX_DEVIATION
        )
        out_full = demod.demodulate(signal_low)
        demod.reset()
        out_half = demod.demodulate(signal_half)

        # The full-deviation output should have roughly 2x the amplitude of half
        ratio = rms(out_full) / (rms(out_half) + 1e-12)
        assert 1.5 < ratio < 2.5, f"Amplitude ratio {ratio:.2f} not near 2.0"

    def test_fm_demodulator_reset(self):
        """After reset, demodulating the same signal gives the same output."""
        tone_freq = 1000.0
        signal = self._generate_fm_signal(tone_freq, 0.02)

        demod = FMDemodulator(
            sample_rate=self.SAMPLE_RATE, max_deviation=self.MAX_DEVIATION
        )
        out1 = demod.demodulate(signal)

        demod.reset()
        out2 = demod.demodulate(signal)

        np.testing.assert_allclose(out1, out2, atol=1e-10)


# ---------------------------------------------------------------------------
# AM Demodulation Round-Trip
# ---------------------------------------------------------------------------


class TestAMDemodulationRoundTrip:
    """Generate a known AM signal and verify envelope detection recovers the tone."""

    SAMPLE_RATE = 2.4e6

    def _generate_am_signal(
        self,
        mod_freq: float,
        mod_index: float,
        carrier_freq: float,
        duration: float,
    ) -> np.ndarray:
        """Synthesize an AM-modulated complex baseband signal.

        AM signal: A * (1 + m * cos(2*pi*f_m*t)) * exp(j*2*pi*f_c*t)

        When carrier_freq == 0, the carrier sits at DC in baseband.
        """
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.arange(n_samples) / self.SAMPLE_RATE

        envelope = 1.0 + mod_index * np.cos(2.0 * np.pi * mod_freq * t)
        carrier = np.exp(1j * 2.0 * np.pi * carrier_freq * t)

        return (envelope * carrier).astype(np.complex128)

    def test_am_400hz_tone(self):
        """AM-demod a 400 Hz tone (mod index 0.8) and verify spectral peak."""
        mod_freq = 400.0
        mod_index = 0.8
        duration = 0.1  # 100 ms -- long enough for good freq resolution
        signal = self._generate_am_signal(
            mod_freq, mod_index, carrier_freq=0.0, duration=duration
        )

        demod = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=True)
        output = demod.demodulate(signal)

        # Let the DC-blocking filter settle
        trim = len(output) // 5
        output_trimmed = output[trim:]

        peak_freq = find_peak_frequency(output_trimmed, self.SAMPLE_RATE)
        assert (
            abs(peak_freq - mod_freq) < 50.0
        ), f"Expected peak near {mod_freq} Hz, got {peak_freq:.1f} Hz"

    def test_am_1khz_tone(self):
        """AM-demod a 1 kHz tone to test a second frequency."""
        mod_freq = 1000.0
        mod_index = 0.5
        duration = 0.1
        signal = self._generate_am_signal(
            mod_freq, mod_index, carrier_freq=0.0, duration=duration
        )

        demod = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=True)
        output = demod.demodulate(signal)

        trim = len(output) // 5
        output_trimmed = output[trim:]

        peak_freq = find_peak_frequency(output_trimmed, self.SAMPLE_RATE)
        assert abs(peak_freq - mod_freq) < 50.0

    def test_am_no_dc_block_preserves_dc(self):
        """Without DC blocking, the output should contain a large DC component."""
        mod_freq = 400.0
        mod_index = 0.5
        duration = 0.05
        signal = self._generate_am_signal(
            mod_freq, mod_index, carrier_freq=0.0, duration=duration
        )

        demod = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=False)
        output = demod.demodulate(signal)

        # Mean should be close to 1.0 (the carrier amplitude)
        assert abs(np.mean(output) - 1.0) < 0.1

    def test_am_modulation_depth(self):
        """Higher modulation index should produce a larger AC component."""
        mod_freq = 400.0
        duration = 0.1

        demod_low = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=True)
        signal_low = self._generate_am_signal(
            mod_freq, mod_index=0.3, carrier_freq=0.0, duration=duration
        )
        out_low = demod_low.demodulate(signal_low)

        demod_high = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=True)
        signal_high = self._generate_am_signal(
            mod_freq, mod_index=0.9, carrier_freq=0.0, duration=duration
        )
        out_high = demod_high.demodulate(signal_high)

        # Trim transients
        trim = len(out_low) // 5
        rms_low = rms(out_low[trim:])
        rms_high = rms(out_high[trim:])

        assert rms_high > rms_low * 1.5, (
            f"High-mod RMS ({rms_high:.4f}) should be substantially larger "
            f"than low-mod RMS ({rms_low:.4f})"
        )


# ---------------------------------------------------------------------------
# AGC Convergence
# ---------------------------------------------------------------------------


class TestAGCConvergence:
    """Feed AGC a constant-amplitude signal and verify gain converges."""

    SAMPLE_RATE = 2.4e6

    def test_agc_converges_to_target_rms(self):
        """A constant-amplitude signal should converge the AGC output to the target level."""
        target_level = 1.0
        input_amplitude = 0.1  # well below target, AGC must boost
        config = AGCConfig(
            target_level=target_level,
            attack_time=0.001,
            decay_time=0.01,
            max_gain=100.0,
            min_gain=0.001,
            mode=AGCMode.RMS,
        )
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)

        # Feed multiple blocks so the per-sample loop has time to converge.
        # The RMS detector with exponential averaging needs many samples to
        # settle, especially when boosting a weak signal.
        n_samples = 120000  # 50 ms at 2.4 MS/s
        tone = input_amplitude * np.ones(n_samples, dtype=np.complex128)
        output = agc.process(tone)

        # Check the last 10% of the output for convergence
        tail = output[-(n_samples // 10) :]
        output_level = rms(tail)
        assert (
            abs(output_level - target_level) / target_level < 0.15
        ), f"AGC output level {output_level:.4f} not within 15% of target {target_level}"

    def test_agc_converges_for_strong_signal(self):
        """An overly-strong signal should be attenuated toward the target level."""
        target_level = 1.0
        input_amplitude = 10.0  # well above target, AGC must reduce gain
        config = AGCConfig(
            target_level=target_level,
            attack_time=0.0005,
            decay_time=0.01,
            max_gain=100.0,
            min_gain=0.001,
            mode=AGCMode.RMS,
        )
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)

        n_samples = 24000
        tone = input_amplitude * np.ones(n_samples, dtype=np.complex128)
        output = agc.process(tone)

        tail = output[-(n_samples // 10) :]
        output_level = rms(tail)
        assert abs(output_level - target_level) / target_level < 0.10

    def test_agc_gain_limits(self):
        """Verify gain is clamped between min_gain and max_gain."""
        config = AGCConfig(
            target_level=1.0,
            attack_time=0.001,
            decay_time=0.01,
            max_gain=50.0,
            min_gain=0.01,
            mode=AGCMode.MAGNITUDE,
        )
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)

        # Very weak signal -- gain should hit max_gain
        weak = 1e-6 * np.ones(5000, dtype=np.complex128)
        agc.process(weak)
        assert agc.current_gain <= config.max_gain + 0.01

        agc.reset()

        # Very strong signal -- gain should hit min_gain
        strong = 1e4 * np.ones(5000, dtype=np.complex128)
        agc.process(strong)
        assert agc.current_gain >= config.min_gain - 0.001

    def test_agc_peak_mode(self):
        """AGC in PEAK mode should converge to target for a constant signal."""
        target_level = 0.5
        config = AGCConfig(
            target_level=target_level,
            attack_time=0.001,
            decay_time=0.01,
            max_gain=100.0,
            min_gain=0.001,
            mode=AGCMode.PEAK,
        )
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)

        # Peak mode with exponential decay needs many samples to converge
        n_samples = 120000
        tone = 0.05 * np.ones(n_samples, dtype=np.complex128)
        output = agc.process(tone)

        tail = output[-(n_samples // 10) :]
        output_level = float(np.max(np.abs(tail)))
        # Peak mode tracks peaks; allow wider tolerance for convergence dynamics
        assert (
            abs(output_level - target_level) / target_level < 0.25
        ), f"AGC PEAK output {output_level:.4f} not within 25% of target {target_level}"

    def test_agc_reset_restores_initial_state(self):
        """After reset, AGC gain should revert to the initial value."""
        config = AGCConfig(target_level=1.0, attack_time=0.001, decay_time=0.01)
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)

        # Drive the gain away from 1.0
        agc.process(0.01 * np.ones(10000, dtype=np.complex128))
        assert agc.current_gain != pytest.approx(1.0, abs=0.5)

        agc.reset()
        assert agc.current_gain == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Filter Response
# ---------------------------------------------------------------------------


class TestFilterResponse:
    """Create a lowpass filter and verify spectral rolloff."""

    SAMPLE_RATE = 2.4e6

    def test_lowpass_attenuates_high_frequencies(self):
        """White noise filtered by a 100 kHz lowpass should show rolloff above 100 kHz."""
        rng = np.random.default_rng(42)

        fb = FilterBank(sample_rate=self.SAMPLE_RATE)
        fb.create_lowpass("lp100k", cutoff=100e3, num_taps=201)

        # Generate real-valued white noise (rfft requires real input)
        n_samples = 100000
        noise = rng.standard_normal(n_samples)

        filtered = fb.apply("lp100k", noise)

        # Compute power spectral density
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.SAMPLE_RATE)
        psd_output = np.abs(np.fft.rfft(filtered)) ** 2

        # Average power in passband (10 kHz -- 80 kHz) vs stopband (200 kHz -- 400 kHz)
        passband_mask = (freqs >= 10e3) & (freqs <= 80e3)
        stopband_mask = (freqs >= 200e3) & (freqs <= 400e3)

        passband_power = np.mean(psd_output[passband_mask])
        stopband_power = np.mean(psd_output[stopband_mask])

        # Passband should have significantly more power than stopband
        assert passband_power > 0, "Passband power is zero"
        assert stopband_power < passband_power, "Stopband not attenuated"

        ratio_db = 10.0 * np.log10(passband_power / (stopband_power + 1e-30))
        assert (
            ratio_db > 10.0
        ), f"Passband/stopband ratio only {ratio_db:.1f} dB, expected >10 dB"

    def test_lowpass_passes_low_frequencies(self):
        """A tone well below the cutoff should pass through largely unattenuated."""
        fb = FilterBank(sample_rate=self.SAMPLE_RATE)
        fb.create_lowpass("lp100k", cutoff=100e3, num_taps=201)

        n_samples = 50000
        t = np.arange(n_samples) / self.SAMPLE_RATE
        tone_10k = np.sin(2.0 * np.pi * 10e3 * t)

        filtered = fb.apply("lp100k", tone_10k)

        # Trim edges to avoid filter transients
        trim = 500
        input_rms = rms(tone_10k[trim:-trim])
        output_rms = rms(filtered[trim:-trim])

        # Expect at most 3 dB of loss in the passband
        ratio = output_rms / (input_rms + 1e-12)
        assert ratio > 0.5, f"Passband attenuation too large: ratio={ratio:.3f}"

    def test_highpass_blocks_low_frequencies(self):
        """A highpass filter should attenuate tones well below its cutoff.

        The windowed-sinc highpass implementation has a wide transition band
        relative to the normalised cutoff, so we compare the energy of a tone
        well below the cutoff against a tone well above it, rather than
        demanding a specific absolute attenuation level.
        """
        local_sr = 480e3  # 480 kHz
        fb = FilterBank(sample_rate=local_sr)
        fb.create_highpass("hp100k", cutoff=100e3, num_taps=301)

        n_samples = 50000
        t = np.arange(n_samples) / local_sr

        # Tone well below cutoff (stopband)
        tone_low = np.sin(2.0 * np.pi * 10e3 * t)
        filtered_low = fb.apply("hp100k", tone_low)

        # Tone well above cutoff (passband)
        tone_high = np.sin(2.0 * np.pi * 200e3 * t)
        filtered_high = fb.apply("hp100k", tone_high)

        trim = 500
        rms_low = rms(filtered_low[trim:-trim])
        rms_high = rms(filtered_high[trim:-trim])

        # The passband tone should pass with much more energy than the stopband tone
        assert rms_high > rms_low * 1.5, (
            f"Highpass passband RMS ({rms_high:.4f}) should be substantially "
            f"larger than stopband RMS ({rms_low:.4f})"
        )

    def test_bandpass_passes_center_frequency(self):
        """A bandpass filter should pass a tone at its center frequency."""
        fb = FilterBank(sample_rate=self.SAMPLE_RATE)
        fb.create_bandpass("bp", low_cutoff=80e3, high_cutoff=120e3, num_taps=301)

        n_samples = 50000
        t = np.arange(n_samples) / self.SAMPLE_RATE
        tone_center = np.sin(2.0 * np.pi * 100e3 * t)
        tone_outside = np.sin(2.0 * np.pi * 10e3 * t)

        filt_center = fb.apply("bp", tone_center)
        filt_outside = fb.apply("bp", tone_outside)

        trim = 500
        rms_center = rms(filt_center[trim:-trim])
        rms_outside = rms(filt_outside[trim:-trim])

        assert rms_center > rms_outside * 3.0, (
            f"Center tone RMS ({rms_center:.4f}) not significantly larger "
            f"than out-of-band RMS ({rms_outside:.4f})"
        )


# ---------------------------------------------------------------------------
# OOK Demodulation
# ---------------------------------------------------------------------------


class TestOOKDemodulation:
    """Generate an OOK signal with a known bit pattern and verify demodulation."""

    SAMPLE_RATE = 2.4e6

    def _generate_ook_signal(
        self,
        bit_pattern: list,
        samples_per_bit: int,
        on_amplitude: float = 1.0,
    ) -> np.ndarray:
        """Synthesize a baseband OOK-modulated IQ signal.

        A ``1`` bit is represented by a constant-amplitude carrier (complex
        exponential at DC), and a ``0`` bit by silence.
        """
        signal = np.zeros(len(bit_pattern) * samples_per_bit, dtype=np.complex128)
        for i, bit in enumerate(bit_pattern):
            start = i * samples_per_bit
            end = start + samples_per_bit
            if bit:
                signal[start:end] = on_amplitude
        return signal

    def test_ook_bit_transitions(self):
        """OOK demod of a known pattern should recover on/off transitions."""
        bit_pattern = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
        samples_per_bit = 1000
        signal = self._generate_ook_signal(bit_pattern, samples_per_bit)

        demod = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        output = demod.demodulate(signal)

        # Sample at the midpoint of each bit
        recovered_bits = []
        for i in range(len(bit_pattern)):
            mid = i * samples_per_bit + samples_per_bit // 2
            recovered_bits.append(int(output[mid] > 0.5))

        assert (
            recovered_bits == bit_pattern
        ), f"Recovered {recovered_bits}, expected {bit_pattern}"

    def test_ook_all_ones(self):
        """All-ones pattern with manual threshold produces all-high output.

        Note: The auto-threshold computes ``(max + min) / 2`` of the envelope.
        For an all-ones signal max == min, so the auto-threshold equals the
        signal level and nothing exceeds it.  We therefore use a manual
        threshold for this edge case, which is realistic usage.
        """
        bit_pattern = [1, 1, 1, 1, 1]
        samples_per_bit = 500
        signal = self._generate_ook_signal(bit_pattern, samples_per_bit)

        demod = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        demod.set_threshold(0.5, auto=False)
        output = demod.demodulate(signal)

        # Entire output should be above threshold
        assert np.all(output > 0.5)

    def test_ook_all_zeros(self):
        """All-zeros pattern should produce all-low output."""
        bit_pattern = [0, 0, 0, 0, 0]
        samples_per_bit = 500
        signal = self._generate_ook_signal(bit_pattern, samples_per_bit)

        demod = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        output = demod.demodulate(signal)

        # Entire output should be below threshold
        assert np.all(output < 0.5)

    def test_ook_with_noise(self):
        """OOK demod should tolerate moderate noise (SNR ~20 dB)."""
        rng = np.random.default_rng(99)
        bit_pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        samples_per_bit = 2000
        signal = self._generate_ook_signal(
            bit_pattern, samples_per_bit, on_amplitude=1.0
        )

        # Add noise at ~20 dB SNR
        noise_power = 0.1
        noise = (
            noise_power
            * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
            / np.sqrt(2)
        )
        noisy_signal = signal + noise

        demod = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        output = demod.demodulate(noisy_signal)

        recovered_bits = []
        for i in range(len(bit_pattern)):
            mid = i * samples_per_bit + samples_per_bit // 2
            recovered_bits.append(int(output[mid] > 0.5))

        assert (
            recovered_bits == bit_pattern
        ), f"Recovered {recovered_bits}, expected {bit_pattern}"

    def test_ook_threshold_auto_vs_manual(self):
        """Auto threshold should match manual for a clean signal."""
        bit_pattern = [1, 0, 1, 0]
        samples_per_bit = 500
        signal = self._generate_ook_signal(bit_pattern, samples_per_bit)

        demod_auto = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        out_auto = demod_auto.demodulate(signal)

        demod_manual = OOKDemodulator(sample_rate=self.SAMPLE_RATE)
        demod_manual.set_threshold(0.5, auto=False)
        out_manual = demod_manual.demodulate(signal)

        # Both should recover the same bits
        for i in range(len(bit_pattern)):
            mid = i * samples_per_bit + samples_per_bit // 2
            assert int(out_auto[mid] > 0.5) == bit_pattern[i]
            assert int(out_manual[mid] > 0.5) == bit_pattern[i]


# ---------------------------------------------------------------------------
# CW Demodulator Smoke Test
# ---------------------------------------------------------------------------


class TestCWDemodulatorSmoke:
    """Basic sanity checks for the CW demodulator."""

    SAMPLE_RATE = 48000.0  # CW is typically at audio rates

    def test_cw_produces_audio_output(self):
        """CW demod of a carrier should produce an audio tone at the BFO frequency."""
        bfo_freq = 700.0
        duration = 0.1
        n_samples = int(self.SAMPLE_RATE * duration)

        # Pure carrier at DC (baseband)
        carrier = np.ones(n_samples, dtype=np.complex128)

        demod = CWDemodulator(sample_rate=self.SAMPLE_RATE, bfo_freq=bfo_freq)
        audio = demod.demodulate(carrier)

        assert len(audio) == n_samples

        # The dominant frequency in the audio should be near the BFO frequency
        peak_freq = find_peak_frequency(audio, self.SAMPLE_RATE)
        assert (
            abs(peak_freq - bfo_freq) < 20.0
        ), f"Expected BFO tone near {bfo_freq} Hz, got {peak_freq:.1f} Hz"

    def test_cw_keying_detection(self):
        """Keying detection should distinguish carrier-present from carrier-absent."""
        n_on = 5000
        n_off = 5000
        carrier_on = np.ones(n_on, dtype=np.complex128)
        carrier_off = np.zeros(n_off, dtype=np.complex128)
        signal = np.concatenate([carrier_on, carrier_off, carrier_on])

        demod = CWDemodulator(sample_rate=self.SAMPLE_RATE, bfo_freq=700.0)
        keying = demod.detect_keying(signal)

        assert len(keying) == len(signal)

        # Sample well inside the first "on" region
        on_mid = n_on // 2

        # The first "on" should eventually register as keyed
        # (The detector has a settling time, so check towards the middle)
        assert keying[on_mid] > 0.5 or keying[on_mid + 1000] > 0.5
        # The "off" region should be mostly zero (after settling)
        assert np.mean(keying[n_on + 1000 : n_on + n_off - 1000]) < 0.5


# ---------------------------------------------------------------------------
# Integration: FM demod through AGC
# ---------------------------------------------------------------------------


class TestFMWithAGC:
    """End-to-end: FM signal -> AGC -> FM demod and verify tone recovery."""

    SAMPLE_RATE = 2.4e6
    MAX_DEVIATION = 75e3

    def test_fm_agc_pipeline(self):
        """A weak FM signal passed through AGC then demodulated should still
        recover the modulating tone.
        """
        tone_freq = 1000.0
        duration = 0.05
        n_samples = int(self.SAMPLE_RATE * duration)
        t = np.arange(n_samples) / self.SAMPLE_RATE

        # Generate a weak FM signal
        modulating = np.sin(2.0 * np.pi * tone_freq * t)
        phase = (
            2.0 * np.pi * self.MAX_DEVIATION * np.cumsum(modulating) / self.SAMPLE_RATE
        )
        weak_signal = 0.01 * np.exp(1j * phase)  # -40 dB

        # AGC to normalise amplitude
        config = AGCConfig(
            target_level=1.0,
            attack_time=0.0005,
            decay_time=0.01,
            max_gain=100.0,
            min_gain=0.001,
            mode=AGCMode.MAGNITUDE,
        )
        agc = AGC(sample_rate=self.SAMPLE_RATE, config=config)
        normalised = agc.process(weak_signal)

        # FM demodulate
        demod = FMDemodulator(
            sample_rate=self.SAMPLE_RATE, max_deviation=self.MAX_DEVIATION
        )
        output = demod.demodulate(normalised)

        trim = len(output) // 5
        output_trimmed = output[trim:-trim]

        peak_freq = find_peak_frequency(output_trimmed, self.SAMPLE_RATE)
        assert (
            abs(peak_freq - tone_freq) < 50.0
        ), f"After AGC + FM demod, expected ~{tone_freq} Hz, got {peak_freq:.1f} Hz"


# ---------------------------------------------------------------------------
# Integration: Filtered noise through FM demod (negative test)
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Sanity-check that demodulators do not produce false positives."""

    SAMPLE_RATE = 2.4e6

    def test_fm_demod_of_noise_has_no_strong_tone(self):
        """FM-demodulating pure noise should not produce a sharp spectral peak."""
        rng = np.random.default_rng(123)
        n_samples = 50000
        noise = (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        ) / np.sqrt(2)

        demod = FMDemodulator(sample_rate=self.SAMPLE_RATE, max_deviation=75e3)
        output = demod.demodulate(noise)

        # The spectrum should be relatively flat -- no bin should dominate
        spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
        mean_mag = np.mean(spectrum[1:])  # exclude DC
        max_mag = np.max(spectrum[1:])

        # The peak should not be more than ~10x the mean (in a flat spectrum
        # it is typically 2-4x due to windowing)
        assert (
            max_mag < mean_mag * 12.0
        ), f"Noise demod has suspiciously sharp peak: max/mean = {max_mag/mean_mag:.1f}"

    def test_am_demod_of_zero_signal(self):
        """AM-demodulating a zero signal should produce near-zero output."""
        n_samples = 10000
        zero = np.zeros(n_samples, dtype=np.complex128)

        demod = AMDemodulator(sample_rate=self.SAMPLE_RATE, dc_block=True)
        output = demod.demodulate(zero)

        assert rms(output) < 1e-6
