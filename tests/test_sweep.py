"""Sweep planning and measurement (sdr_module.dsp.sweep), shared by the
Scanner tab and ``sdr-scan scan``."""

import numpy as np
import pytest

from sdr_module.dsp.sweep import (
    FFT_SIZE,
    SpectrumAverager,
    channel_frequencies,
    format_duration,
    measure_page,
    noise_floor_db,
    plan_sweep,
    power_spectrum_dbfs,
    report_frequency,
    same_signal,
)

RATE = 2.4e6


def _capture(lo_hz, signals, n=FFT_SIZE * 32, noise=1e-4, seed=0):
    """Noise plus tones ``{freq: dBFS}`` (or ``{freq: (dBFS, width)}`` for a
    noise-like signal ``width`` Hz wide) around ``lo_hz``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * noise
    for freq, spec in signals.items():
        level, width = spec if isinstance(spec, tuple) else (spec, 0.0)
        amp = 10 ** (level / 20)
        offset = freq - lo_hz
        if width:
            # FM-like: a tone swept across the width.
            phase = 2 * np.pi * offset * t + (width / 2 / 400.0) * np.sin(
                2 * np.pi * 400.0 * t
            )
            x = x + amp * np.exp(1j * phase)
        else:
            x = x + amp * np.exp(2j * np.pi * offset * t)
    return x.astype(np.complex64)


def _spectrum(x):
    averager = SpectrumAverager(max_segments=64)
    averager.add(x)
    return averager.power_db()


class TestPlan:
    def test_channels_include_both_ends(self):
        chans = channel_frequencies(88e6, 108e6, 200e3)
        assert len(chans) == 101
        assert chans[0] == 88e6 and chans[-1] == pytest.approx(108e6)

    def test_pages_step_by_about_80_percent_of_the_rate(self):
        pages = plan_sweep(88e6, 108e6, 200e3, RATE)
        # Every channel is measured once, on a page whose usable span holds
        # its whole slice.
        all_channels = [c for p in pages for c in p.channels]
        assert all_channels == list(channel_frequencies(88e6, 108e6, 200e3))
        for page in pages:
            for c in page.channels:
                assert abs(c - page.lo_hz) + 100e3 <= 0.4 * RATE + 1
        steps = np.diff([p.lo_hz for p in pages[:-1]])
        assert np.all(steps <= 0.8 * RATE) and np.all(steps >= 0.6 * RATE)
        assert len(pages) == 12

    def test_lo_is_never_on_a_channel_centre(self):
        for start, end, step in (
            (88e6, 108e6, 200e3),
            (144e6, 148e6, 12.5e3),
            (162.4e6, 162.55e6, 25e3),
            (100e6, 100e6, 25e3),
        ):
            for page in plan_sweep(start, end, step, RATE):
                gap = min(abs(c - page.lo_hz) for c in page.channels)
                assert gap >= min(step / 2, 4 * RATE / FFT_SIZE) - 1

    def test_wide_steps_get_a_page_each(self):
        pages = plan_sweep(100e6, 120e6, 5e6, RATE)
        assert [len(p.channels) for p in pages] == [1, 1, 1, 1, 1]

    def test_page_count_matches_the_plan(self):
        from sdr_module.dsp.sweep import channel_count, page_count

        for start, end, step in (
            (88e6, 108e6, 200e3),
            (144e6, 148e6, 12.5e3),
            (100e6, 120e6, 5e6),
            (162.4e6, 162.55e6, 25e3),
            (100e6, 100e6, 25e3),
        ):
            pages = plan_sweep(start, end, step, RATE)
            assert page_count(start, end, step, RATE) == len(pages)
            assert channel_count(start, end, step) == sum(p.count for p in pages)
        assert page_count(108e6, 88e6, 200e3, RATE) == 0

    def test_a_huge_plan_holds_no_channel_lists(self):
        # 0.1-6000 MHz in 1 kHz steps: six million channels, planned without
        # listing them (the Scanner tab's hint is recomputed as you type).
        from sdr_module.dsp.sweep import channel_count

        pages = plan_sweep(0.1e6, 6000e6, 1e3, RATE)
        assert channel_count(0.1e6, 6000e6, 1e3) == 5_999_901
        assert len(pages) == 3127
        page = pages[100]
        chans = page.channels
        assert len(chans) == page.count
        assert chans[0] == page.first_hz
        assert chans[-1] == page.last_hz
        nearest = page.nearest_channel(page.lo_hz)
        assert abs(nearest - page.lo_hz) <= 500.0
        assert (nearest - page.first_hz) % 1e3 == pytest.approx(0.0, abs=1e-3)

    def test_bad_ranges_plan_nothing(self):
        assert plan_sweep(108e6, 88e6, 200e3, RATE) == []
        assert plan_sweep(88e6, 108e6, 0, RATE) == []
        assert plan_sweep(88e6, 108e6, 200e3, 0) == []


class TestMeasure:
    def test_finds_carriers_at_their_frequency(self):
        lo = 100.05e6
        x = _capture(lo, {99.5e6: -40.0, 100.3e6: -50.0, 100.712e6: -60.0})
        chans = channel_frequencies(99.2e6, 100.9e6, 12.5e3)
        readings, noise, threshold = measure_page(_spectrum(x), lo, RATE, chans, 12.5e3)
        assert [r.frequency_hz for r in readings] == [99.5e6, 100.3e6, 100.712e6]
        assert readings[0].peak_db == pytest.approx(-40.0, abs=1.0)
        assert threshold == pytest.approx(noise + 12.0)
        assert all(r.snr_db > 20 for r in readings)

    def test_a_wide_signal_is_reported_once_across_narrow_channels(self):
        lo = 100.05e6
        x = _capture(lo, {100.3e6: (-30.0, 150e3)})
        chans = channel_frequencies(99.7e6, 100.9e6, 12.5e3)
        readings, _noise, _thr = measure_page(_spectrum(x), lo, RATE, chans, 12.5e3)
        assert len(readings) == 1
        assert readings[0].frequency_hz == pytest.approx(100.3e6, abs=15e3)
        assert readings[0].width_hz > 100e3

    def test_broadcast_steps_snap_wide_signals_to_the_channel_raster(self):
        lo = 100.05e6
        x = _capture(lo, {100.3e6: (-30.0, 150e3), 99.62e6: -40.0})
        chans = channel_frequencies(99.3e6, 100.9e6, 200e3)
        readings, _noise, _thr = measure_page(_spectrum(x), lo, RATE, chans, 200e3)
        assert [r.frequency_hz for r in readings] == [99.62e6, 100.3e6]

    def test_skirt_of_a_neighbour_is_not_a_signal(self):
        # A strong carrier right next to a channel's slice: the slice's edge
        # bin is high, but it is the neighbour's skirt.
        lo = 100.05e6
        x = _capture(lo, {100.2125e6: -10.0})
        chans = [100.2e6]
        readings, _noise, _thr = measure_page(_spectrum(x), lo, RATE, chans, 25e3)
        assert readings == []

    def test_a_bump_in_a_stronger_signals_skirt_is_not_another_signal(self):
        # A broadcast station's skirt has bumps (its modulation's sidebands):
        # one that is the highest bin of the next channel's slice, a local
        # peak there, measures as the station itself (its 12 dB region runs
        # into the station, whose centroid it then takes) and was listed a
        # second time at the station's frequency (the demo's 100.1 MHz
        # station once its stereo followed the standard, K-S1).
        n = FFT_SIZE
        lo = 100.05e6
        bin_hz = RATE / n
        offsets = (np.arange(n) - n // 2) * bin_hz
        away = np.abs(lo + offsets - 100.1e6)
        power = np.maximum(-31.0 - np.maximum(away - 50e3, 0.0) * 0.8e-3, -87.0)
        power = power + np.where(np.arange(n) % 2, 0.2, -0.2)  # (noise)
        first = n // 2 + int(np.ceil((100.3e6 - lo - 100e3) / bin_hz - 1e-9))
        power[first] = power[first - 1] + 1.0  # (the bump)
        readings, _noise, _thr = measure_page(
            power, lo, RATE, [100.1e6, 100.3e6], 200e3
        )
        assert [r.frequency_hz for r in readings] == [100.1e6]

    def test_threshold_and_minimum_width(self):
        # The threshold is a peak level per bin, as the spectrum shows it: a
        # wide signal spreads its power (-30 dBFS peaks near -42 dBFS/bin).
        lo = 100.05e6
        x = _capture(lo, {99.5e6: -30.0, 100.3e6: (-30.0, 150e3)})
        chans = channel_frequencies(99.3e6, 100.9e6, 200e3)
        spec = _spectrum(x)
        readings, _n, thr = measure_page(spec, lo, RATE, chans, 200e3, threshold_db=-35)
        assert thr == -35
        assert [r.frequency_hz for r in readings] == [99.5e6]
        readings, _n, _t = measure_page(spec, lo, RATE, chans, 200e3, min_width_hz=30e3)
        assert [r.frequency_hz for r in readings] == [100.3e6]

    def test_dc_spike_at_the_lo_is_ignored(self):
        lo = 100.05e6
        x = _capture(lo, {lo: -20.0})
        chans = channel_frequencies(99.3e6, 100.9e6, 25e3)
        readings, _noise, _thr = measure_page(_spectrum(x), lo, RATE, chans, 25e3)
        assert readings == []

    def test_outside_the_usable_span_is_ignored(self):
        lo = 100e6
        x = _capture(lo, {lo + 1.1e6: -20.0})  # in the roll-off
        chans = channel_frequencies(lo + 1.0e6, lo + 1.15e6, 25e3)
        readings, _noise, _thr = measure_page(_spectrum(x), lo, RATE, chans, 25e3)
        assert readings == []

    def test_empty_input(self):
        assert measure_page(np.zeros(4), 100e6, RATE, [100e6], 25e3)[0] == []
        assert measure_page(np.zeros(2048), 100e6, RATE, [], 25e3)[0] == []


class TestSpectra:
    def test_levels_do_not_depend_on_the_block_size(self):
        rng = np.random.default_rng(1)
        floors = []
        for block in (8192, 262144):
            x = (rng.standard_normal(block) + 1j * rng.standard_normal(block)) * 1e-3
            averager = SpectrumAverager(max_segments=32)
            averager.add(x.astype(np.complex64))
            floors.append(noise_floor_db(averager.power_db()))
        assert floors[0] == pytest.approx(floors[1], abs=1.0)

    def test_averager_matches_the_display_scale(self):
        t = np.arange(FFT_SIZE * 4)
        tone = np.exp(2j * np.pi * (256 / FFT_SIZE) * t)  # on a bin
        averager = SpectrumAverager()
        averager.add(tone)
        assert np.max(averager.power_db()) == pytest.approx(0.0, abs=0.1)
        assert np.max(power_spectrum_dbfs(tone)) == pytest.approx(0.0, abs=0.1)

    def test_averager_accumulates_and_resets(self):
        averager = SpectrumAverager(max_segments=4)
        assert averager.power_db() is None
        assert averager.add(np.zeros(100, np.complex64)) == 0  # too short
        assert averager.add(np.ones(FFT_SIZE * 10, np.complex64)) == 4
        assert averager.add(np.ones(FFT_SIZE * 2, np.complex64)) == 2
        assert averager.segments == 6
        averager.reset()
        assert averager.segments == 0 and averager.power_db() is None

    def test_short_capture_uses_one_fft(self):
        assert len(power_spectrum_dbfs(np.ones(100, np.complex64))) == 100
        assert len(power_spectrum_dbfs(np.ones(0, np.complex64))) == 0


class TestHelpers:
    def test_report_frequency_rules(self):
        # Wide signal, broadcast-sized step: nearest 100 kHz channel.
        assert report_frequency(88.4861e6, 90e3, 200e3) == 88.5e6
        # A narrow carrier in the same sweep keeps its own frequency.
        assert report_frequency(433.9204e6, 5e3, 200e3) == 433.92e6
        # Voice-channel steps never snap to 100 kHz.
        assert report_frequency(146.5204e6, 90e3, 12.5e3) == 146.52e6
        assert report_frequency(146.5204e6, 90e3, None) == 146.52e6

    def test_same_signal(self):
        assert same_signal(100.0e6, 1e3, 100.1e6, 200e3)  # within 3/4 step
        assert not same_signal(100.0e6, 1e3, 100.2e6, 200e3)
        assert same_signal(100.0e6, 180e3, 100.08e6, 12.5e3)  # inside it
        assert not same_signal(146.52e6, 10e3, 146.535e6, 12.5e3)

    def test_format_duration(self):
        assert format_duration(0.2) == "about 1 s"
        assert format_duration(42) == "about 42 s"
        assert format_duration(150) == "about 2 min"
        assert format_duration(3 * 3600 + 25 * 60) == "about 3 h 25 min"
        assert format_duration(7200) == "about 2 h"
