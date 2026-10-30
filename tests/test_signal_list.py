"""Signals in view (T10): find_signals on a spectrum and SignalSurvey over
a recording (fg:FG-G17, rx:G-10, fg:FG-G02)."""

from __future__ import annotations

import numpy as np
import pytest

from sdr_module.dsp.signal_list import (
    SignalSurvey,
    find_signals,
    local_floor_db,
    survey_nfft,
)

SPAN = 2.4e6
N = 2048
BIN = SPAN / N


def _spectrum(seed=0, floor_db=-90.0):
    """A noise floor in dBFS per bin (a 30-frame average: about +/- 1 dB)."""
    rng = np.random.default_rng(seed)
    lin = rng.gamma(30.0, 1.0 / 30.0, N) * 10 ** (floor_db / 10)
    return lin


def _db(lin):
    return 10 * np.log10(lin)


class TestFindSignals:
    def test_a_carrier_and_a_narrow_fm_signal(self):
        lin = _spectrum()
        centre = N // 2
        lin[centre + 85] += 10 ** (-40 / 10)  # a carrier, 100 kHz up
        bump = np.arange(centre - 261, centre - 251)  # 11.7 kHz wide, 300 kHz down
        lin[bump] += 10 ** (-60 / 10)
        lin[centre] += 10 ** (-50 / 10)  # the DC spike
        found = find_signals(_db(lin), 100e6, SPAN)
        assert len(found) == 2, found
        nfm, carrier = found
        assert carrier.frequency_hz == pytest.approx(100e6 + 85 * BIN, abs=BIN)
        assert carrier.peak_dbfs == pytest.approx(-40, abs=0.5)
        assert carrier.snr_db == pytest.approx(50, abs=2)
        assert carrier.width_hz <= 3 * BIN
        assert nfm.frequency_hz == pytest.approx(100e6 - 256 * BIN, abs=BIN)
        assert nfm.snr_db == pytest.approx(30, abs=2)
        assert nfm.width_hz == pytest.approx(10 * BIN, abs=2 * BIN)

    def test_min_snr_and_the_edges(self):
        lin = _spectrum(1)
        centre = N // 2
        lin[centre + 299 : centre + 302] += 3e-9  # 6 dB up: under 10 dB
        lin[10] += 10 ** (-40 / 10)  # in the receiver's roll-off at the edge
        assert find_signals(_db(lin), 0.0, SPAN) == []
        weak = find_signals(_db(lin), 0.0, SPAN, min_snr_db=4.0)
        assert [round(s.offset_hz / BIN) for s in weak] == [300]

    def test_noise_alone_lists_nothing(self):
        for seed in range(5):
            assert find_signals(_db(_spectrum(seed)), 0.0, SPAN) == []

    def test_the_floor_follows_a_slope(self):
        slope = np.linspace(-95.0, -85.0, N)
        noise = 10 * np.log10(np.random.default_rng(3).exponential(1.0, N))
        floor = local_floor_db(slope + noise)  # one FFT: the noise's mean
        assert np.max(np.abs(floor[100:-100] - slope[100:-100])) < 1.5

    def test_a_broadcast_station_is_on_its_channel(self):
        """Wide FM's centroid wanders with the programme: it is listed on
        its 100 kHz channel."""
        lin = _spectrum(2)
        centre = N // 2
        # 170 kHz wide, lopsided, centred near 100.1 MHz - 9 kHz.
        idx = np.arange(centre + 77, centre + 222)
        lin[idx] += 10 ** (-30 / 10) * np.linspace(1.0, 0.2, len(idx))
        found = find_signals(_db(lin), 99.95e6, SPAN)
        assert [s.frequency_hz for s in found] == [100.1e6]


def test_the_demo_fm_band():
    """The demo at 100.3 MHz: the stations and the narrow signals in view
    are listed, each once."""
    from sdr_module.dsp.classifiers import welch_psd
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(SPAN)
    dev.set_frequency(100.3e6)
    dev.start_rx()
    x = np.concatenate([dev.read_samples(80_000) for _ in range(30)])
    _f, psd, _s = welch_psd(x, SPAN, nfft=N)
    db = 10 * np.log10(psd * SPAN * 1.5 / N)  # dBFS per Hann bin
    found = find_signals(db, 100.3e6, SPAN)
    freqs = [s.frequency_hz for s in found]
    for want, tol in ((99.5e6, 1e3), (100.1e6, 1e3), (101.1e6, 1e3), (99.82e6, 2e3)):
        assert sum(abs(f - want) <= tol for f in freqs) == 1, (want, freqs)
    strongest = max(found, key=lambda s: s.peak_dbfs)
    assert strongest.frequency_hz == 100.1e6


class TestSurvey:
    RATE = 240e3

    def test_first_and_last_heard(self):
        n = int(2.0 * self.RATE)
        t = np.arange(n) / self.RATE
        rng = np.random.default_rng(4)
        x = 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x += 0.1 * np.exp(2j * np.pi * 30e3 * t)  # on all the time
        on = (t >= 0.5) & (t < 1.0)
        x[on] += 0.1 * np.exp(2j * np.pi * -50e3 * t[on])  # half a second
        survey = SignalSurvey(self.RATE, 145e6)
        for k in range(0, n, 10_000):  # any chunk size
            survey.add(x[k : k + 10_000].astype(np.complex64))
        found = survey.finish()
        assert [round(s.frequency_hz - 145e6, -2) for s in found] == [-50e3, 30e3]
        burst, steady = found
        # To a segment (about 1 ms), not to a quarter-second slice.
        assert burst.first_s == pytest.approx(0.5, abs=0.005)
        assert burst.last_s == pytest.approx(1.0, abs=0.005)
        assert burst.duty == pytest.approx(0.25, abs=0.15)
        assert steady.first_s == pytest.approx(0.0, abs=0.01)
        assert steady.last_s == pytest.approx(2.0, abs=0.01)
        assert steady.duty == 1.0
        assert survey.position == n - n % survey.nfft or survey.position == n

    def test_resolution(self):
        assert survey_nfft(2.4e6) == 2048
        assert survey_nfft(240e3) == 256
