#!/usr/bin/env python3
"""
Regression tests for the audio DSP back ends the GUI's Audio section uses:

* FG-B06: NoiseReduction (spectral subtraction, Wiener) amplified the
  signal by 20-45 dB at every block edge and estimate_noise failed on I/Q.
* FG-B07: the CTCSS tone squelch opened for the adjacent EIA tones.
* FG-B08: AGC LOG mode drove the gain to its maximum, the NOISE squelch was
  inverted and DCS was only a carrier squelch.
* FG-B09: the I/Q imbalance estimate used the opposite gain convention to
  the correction, so "correcting" made the image worse.
"""

import numpy as np
import pytest

from sdr_module.dsp.filters import (
    AGC,
    CTCSS_FREQUENCIES,
    AGCConfig,
    AGCMode,
    CTCSSDetector,
    NoiseReduction,
    NoiseReductionConfig,
    NoiseReductionMethod,
    Squelch,
    SquelchConfig,
    SquelchMode,
    ctcss_label,
)
from sdr_module.utils.iq import (
    apply_dc_offset_correction,
    apply_iq_imbalance_correction,
    estimate_iq_imbalance,
)

FS = 48000.0
SPECTRAL = (NoiseReductionMethod.SPECTRAL_SUBTRACTION, NoiseReductionMethod.WIENER)


def _noisy_tone(seconds=4.0, tone=0.25, noise=0.2, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    clean = tone * np.sin(2 * np.pi * 1000 * t)
    return clean, clean + noise * rng.standard_normal(n)


def _snr_db(output, clean, latency):
    """SNR of ``output`` against the clean signal it should match, after
    the processor's latency, ignoring the first second (adaptation)."""
    start = int(FS)
    y = np.real(output[start + latency :])
    ref = clean[start : start + len(y)]
    # Least-squares gain so a small level change isn't counted as noise.
    gain = np.dot(y, ref) / np.dot(ref, ref)
    error = y - gain * ref
    return 10 * np.log10(np.sum((gain * ref) ** 2) / np.sum(error**2))


class TestNoiseReductionStreams:
    """FG-B06."""

    @pytest.mark.parametrize("method", SPECTRAL)
    def test_snr_improves_whole_and_in_blocks_without_spikes(self, method):
        clean, noisy = _noisy_tone()
        before = _snr_db(noisy, clean, 0)
        for blocks in (False, True):
            nr = NoiseReduction(FS, NoiseReductionConfig(method=method))
            if blocks:
                out = np.concatenate(
                    [
                        nr.process(noisy[i : i + 1600])
                        for i in range(0, len(noisy), 1600)
                    ]
                )
            else:
                out = nr.process(noisy)
            assert len(out) == len(noisy)
            assert np.max(np.abs(out)) <= 2 * np.max(np.abs(noisy))  # was 4446
            after = _snr_db(out, clean, nr.latency_samples)
            assert after > before + 3, (method, blocks, before, after)

    @pytest.mark.parametrize("method", SPECTRAL)
    def test_blocks_give_exactly_the_whole_output(self, method):
        _clean, noisy = _noisy_tone(seconds=1.0)
        whole = NoiseReduction(FS, NoiseReductionConfig(method=method)).process(noisy)
        nr = NoiseReduction(FS, NoiseReductionConfig(method=method))
        cuts = [0, 1, 700, 1601, 1602, 9000, 20000, len(noisy)]
        parts = np.concatenate(
            [nr.process(noisy[a:b]) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
        )
        np.testing.assert_allclose(parts, whole, atol=1e-9)

    @pytest.mark.parametrize("method", SPECTRAL)
    def test_a_clean_signal_keeps_its_level(self, method):
        t = np.arange(int(2 * FS)) / FS
        tone = 0.5 * np.sin(2 * np.pi * 700 * t)
        nr = NoiseReduction(FS, NoiseReductionConfig(method=method))
        out = nr.process(tone)
        y = out[int(FS) + nr.latency_samples :]
        ref = tone[int(FS) : int(FS) + len(y)]
        assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(ref**2)), 0.12)
        np.testing.assert_allclose(y, ref, atol=0.08)

    def test_a_steady_tone_is_not_learned_as_noise(self):
        # Without estimate_noise the old code took the first frames (tone
        # and all) as the noise and subtracted the tone away.
        clean, noisy = _noisy_tone(tone=0.3, noise=0.1)
        nr = NoiseReduction(
            FS, NoiseReductionConfig(method=NoiseReductionMethod.SPECTRAL_SUBTRACTION)
        )
        out = np.real(nr.process(noisy))
        start = int(2 * FS)
        y = out[start:]
        t = (np.arange(len(y)) + start - nr.latency_samples) / FS
        tone_level = 2 * abs(np.mean(y * np.exp(-2j * np.pi * 1000 * t)))
        assert tone_level == pytest.approx(0.3, rel=0.15)
        # ... and the output is not dominated by anything else.
        assert np.sqrt(np.mean(y**2)) < 1.2 * 0.3 / np.sqrt(2)

    def test_estimate_noise_accepts_iq_and_reduces_iq_noise(self):
        rng = np.random.default_rng(1)
        n = int(2 * FS)
        noise = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        nr = NoiseReduction(
            FS, NoiseReductionConfig(method=NoiseReductionMethod.SPECTRAL_SUBTRACTION)
        )
        nr.estimate_noise(noise)  # used to raise TypeError
        assert nr.noise_estimated
        out = nr.process(noise[: int(FS)])
        assert np.iscomplexobj(out)
        tail = out[nr.latency_samples + 4096 :]
        assert np.mean(np.abs(tail) ** 2) < 0.5 * np.mean(np.abs(noise) ** 2)

    def test_reset_stream_keeps_the_noise_estimate(self):
        _clean, noisy = _noisy_tone(seconds=2.0)
        nr = NoiseReduction(
            FS, NoiseReductionConfig(method=NoiseReductionMethod.WIENER)
        )
        nr.process(noisy)
        assert nr.noise_estimated
        nr.reset_stream()
        assert nr.noise_estimated
        nr.reset()
        assert not nr.noise_estimated


def _voice_and_tone(tone_hz, seconds=2.0, seed=4):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[(f < 300) | (f > 3000)] = 0
    voice = np.fft.irfft(spec, n)
    voice *= 0.3 / np.sqrt(np.mean(voice**2))
    voice *= 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 2.5 * t))
    if tone_hz:
        voice = voice + 0.1 * np.sin(2 * np.pi * tone_hz * t)
    return voice


class TestCtcss:
    """FG-B07."""

    def _opened(self, tone_hz, api="squelch"):
        audio = _voice_and_tone(tone_hz)
        if api == "detector":
            detector = CTCSSDetector(FS)
            flags = []
            for i in range(0, len(audio), 1600):
                detector.process(audio[i : i + 1600])
                flags.append(detector.is_present(100.0))
            return flags
        sq = Squelch(
            FS,
            SquelchConfig(
                mode=SquelchMode.CTCSS, ctcss_freq=100.0, threshold=0.05, tail_time=0.0
            ),
        )
        flags = []
        for i in range(0, len(audio), 1600):
            _out, is_open = sq.process_block(audio[i : i + 1600])
            flags.append(is_open)
        return flags

    @pytest.mark.parametrize("api", ["squelch", "detector"])
    def test_opens_only_for_its_own_tone(self, api):
        right = self._opened(100.0, api)
        # Opens within a second and stays open.
        assert all(right[30:]), right
        for wrong in (97.4, 103.5, 94.8, 107.2, 88.5, 123.0, 67.0, None):
            flags = self._opened(wrong, api)
            assert sum(flags) == 0, (wrong, sum(flags))  # 97.4 / 103.5 were 60/60

    def test_reports_the_tone_it_hears(self):
        for tone in (67.0, 69.3, 100.0, 159.8, 162.2, 254.1):
            detector = CTCSSDetector(FS)
            audio = _voice_and_tone(tone, seconds=1.5)
            for i in range(0, len(audio), 1600):
                detector.process(audio[i : i + 1600])
            assert detector.detected_tone == pytest.approx(tone), tone
        detector = CTCSSDetector(FS)
        detector.process(_voice_and_tone(None, seconds=1.5))
        assert detector.detected_tone is None

    def test_tone_table_and_labels(self):
        assert len(CTCSS_FREQUENCIES) == 50
        assert CTCSS_FREQUENCIES[0] == 67.0 and CTCSS_FREQUENCIES[-1] == 254.1
        assert ctcss_label(100.0) == "100.0 Hz (1Z)"
        assert ctcss_label(254.1) == "254.1 Hz"


class TestAgcLogAndSquelchModes:
    """FG-B08."""

    def _step(self):
        n = int(FS)
        t = np.arange(n) / FS
        amp = np.where(t < 0.5, 0.05, 0.9)
        return amp * np.exp(2j * np.pi * 1000 * t)

    @pytest.mark.parametrize("api", ["process", "process_block", "process_with_ramp"])
    def test_log_mode_levels_like_the_others(self, api):
        x = self._step()
        agc = AGC(FS, AGCConfig(target_level=0.5, mode=AGCMode.LOG))
        if api == "process_block":
            y = np.concatenate(
                [agc.process_block(x[i : i + 1600]) for i in range(0, len(x), 1600)]
            )
        else:
            y = getattr(agc, api)(x)
        quiet = np.abs(y[int(0.3 * FS) : int(0.5 * FS)])
        loud = np.abs(y[int(0.8 * FS) :])
        assert np.mean(quiet) == pytest.approx(0.5, abs=0.05)
        assert np.mean(loud) == pytest.approx(0.5, abs=0.05)  # was 63.6
        # The 25 dB step overshoots for the attack time only, as in the
        # other modes (it used to stay at 90, the maximum gain).
        assert np.max(np.abs(y[int(0.51 * FS) :])) < 1.5

    def test_log_mode_reference_offsets_the_target(self):
        x = np.full(int(FS), 0.1 + 0j)
        agc = AGC(FS, AGCConfig(target_level=0.5, mode=AGCMode.LOG, reference_level=-6))
        y = agc.process(x)
        assert np.abs(y[-1]) == pytest.approx(0.5 * 10 ** (-6 / 20), rel=0.05)

    def _blocks_open(self, mode, signal, threshold, api="process_block"):
        sq = Squelch(FS, SquelchConfig(mode=mode, threshold=threshold, tail_time=0.0))
        opened = 0
        for i in range(0, len(signal), 1600):
            block = signal[i : i + 1600]
            if api == "process_block":
                _out, is_open = sq.process_block(block)
            else:
                sq.process(block)
                is_open = sq.is_open
            opened += bool(is_open)
        return opened

    @pytest.mark.parametrize("api", ["process_block", "process"])
    def test_noise_squelch_opens_on_a_carrier_not_on_noise(self, api):
        rng = np.random.default_rng(8)
        m = int(FS)
        carrier = 0.5 * np.exp(2j * np.pi * 3000 * np.arange(m) / FS) + 0.016 * (
            rng.standard_normal(m) + 1j * rng.standard_normal(m)
        )
        noise = 0.3 * (rng.standard_normal(m) + 1j * rng.standard_normal(m))
        for threshold in (0.1, 0.5):
            assert self._blocks_open(SquelchMode.NOISE, carrier, threshold, api) >= 28
            assert self._blocks_open(SquelchMode.NOISE, noise, threshold, api) == 0

    def test_noise_squelch_ignores_voice_modulation(self):
        # 5 kHz deviation voice at 48 kHz: FM keeps the noise quiet.
        audio = _voice_and_tone(None, seconds=1.0)
        audio /= np.max(np.abs(audio))
        phase = 2 * np.pi * np.cumsum(5e3 * audio) / FS
        fm = np.exp(1j * phase)
        assert self._blocks_open(SquelchMode.NOISE, fm, 0.1) >= 28

    def test_dcs_is_not_pretended(self):
        with pytest.raises(NotImplementedError):
            Squelch(FS, SquelchConfig(mode=SquelchMode.DCS))
        sq = Squelch(FS)
        with pytest.raises(NotImplementedError):
            sq.set_config(SquelchConfig(mode=SquelchMode.DCS))


def _imbalanced_tone(eps_q=0.10, phi_deg=5.0, dc=0.0, n=1 << 16, cycles=8192):
    """A bin-centred tone with a Q gain error ``eps_q`` and a phase error."""
    k = np.arange(n)
    theta = 2 * np.pi * cycles * k / n
    phi = np.radians(phi_deg)
    i = np.cos(theta)
    q = (1 + eps_q) * np.sin(theta + phi)
    return (i + 1j * q + dc).astype(np.complex128), cycles


def _image_dbc(x, cycles):
    spectrum = np.abs(np.fft.fft(x)) ** 2
    return 10 * np.log10(spectrum[-cycles] / spectrum[cycles] + 1e-30)


class TestIqImbalance:
    """FG-B09."""

    def test_estimate_uses_the_correction_convention(self):
        x, _ = _imbalanced_tone()
        gain, phase = estimate_iq_imbalance(x)
        assert gain == pytest.approx(0.10, abs=0.005)  # was -0.091
        assert np.degrees(phase) == pytest.approx(5.0, abs=0.1)

    @pytest.mark.parametrize("eps_q, phi", [(0.10, 5.0), (-0.2, -3.0), (0.05, 0.0)])
    def test_round_trip_removes_the_image(self, eps_q, phi):
        x, cycles = _imbalanced_tone(eps_q, phi, dc=0.05 + 0.02j)
        before = _image_dbc(x, cycles)
        clean = apply_dc_offset_correction(x)
        gain, phase = estimate_iq_imbalance(clean)
        after = _image_dbc(apply_iq_imbalance_correction(clean, gain, phase), cycles)
        assert after < -40, (before, after)  # was -20.4 (worse than -23.8)
        assert after < before
