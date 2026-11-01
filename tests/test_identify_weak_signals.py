"""Identify near 10 dB SNR (K-L1): what the honest noise measure cost.

Measuring the noise's real spread (W-02: the noise's ripple no longer
counts as signal, so a call isn't joined to a station 385 kHz away) made
Identify about 1 dB less sensitive. An AM station near 10 dB read
"Unmodulated carrier" (its faint sidebands no longer reached far enough to
be cut out with the carrier), FSK at 8 dB was taken for OOK at 0.78 ("its
two tones are no longer joined across the dip between them") and Mode S
replies at 4-6 dB were more often noise. These tests pin the recovered
sensitivity; ``TestTheSignalAtTheTunedFrequency`` in
tests/test_signal_identify.py pins that the noise's ripple still doesn't
count as signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402

from sdr_module.dsp.classifiers import SignalClassifier  # noqa: E402
from sdr_module.dsp.signal_id import analysis_band, identify  # noqa: E402

FS = 48_000.0
N = int(0.5 * FS)


def classify(x, rate=FS, **kw):
    return SignalClassifier(rate).classify(np.asarray(x, dtype=np.complex64), **kw)


def am_voice(n=N, rate=FS, band=(300.0, 3000.0)):
    """AM, 70 % deep, voice (its sidebands hold a few per cent of the
    power, spread over +/- 3 kHz)."""
    return S.am(n, rate, 0.7, S.voice(n, rate, seed=1, band=band))


# --------------------------------------------------------------------------- #
# AM: a carrier with faint sidebands
# --------------------------------------------------------------------------- #


class TestWeakAmIsAm:
    @pytest.mark.parametrize("snr,least", [(10, 11), (11, 12)])
    def test_am_voice_near_10_db(self, snr, least):
        """AM at 10 and 11 dB (in its 6 kHz): it read "Unmodulated carrier"
        in 16 and 5 of 20 takes (the classifier's own 48 kHz band)."""
        kinds = []
        for seed in range(12):
            x, _ = S.add_noise(am_voice(), FS, snr, 6e3, seed=104729 * seed + snr + 3)
            kinds.append(classify(x).kind)
        assert kinds.count("am") >= least, kinds

    def test_through_identify_with_the_windows_am_band(self):
        """Identify as the window runs it while listening in AM with a
        10 kHz filter (a 100 kHz band): AM at 10 dB is AM, and Apply Mode
        listens in AM (it read "Unmodulated carrier" in 4 of 12 takes,
        whose Apply Mode outside an AM band is CW with 500 Hz)."""
        rate = 100e3
        n = int(0.5 * rate)
        sig = S.am(n, rate, 0.7, S.voice(n, rate, seed=1))
        _width, reach = analysis_band("AM", 10e3, 2.4e6, 50e6)
        for seed in range(12):
            x, _ = S.add_noise(sig, rate, 10, 6e3, seed=31337 * seed + 10)
            ident = identify(x, rate, 50e6, reach, band_hz=(-0.45 * rate, 0.45 * rate))
            assert ident.kind == "am", (seed, ident.summary(), ident.details)
            assert ident.mode == "AM"

    @pytest.mark.parametrize("snr", [10, 20, 40])
    def test_its_width_is_its_sidebands_and_its_snr_in_that(self, snr):
        """An AM station occupies what its voice spreads over (+/- 3 kHz
        here), not the 99 % of its power its carrier alone nearly holds:
        it read about 4 kHz wide at any strength (or, called a carrier,
        35 Hz and 32 dB at 10 dB), and its SNR 1.7-2 dB high."""
        for seed in range(4):
            x, _ = S.add_noise(am_voice(), FS, snr, 6e3, seed=11 * seed + snr)
            r = classify(x)
            assert r.kind == "am", (seed, r.kind)
            assert 5.0e3 <= r.bandwidth_hz <= 7.5e3, (seed, r.bandwidth_hz)
            assert abs(r.snr_db - snr) <= 1.5, (seed, r.snr_db)
            assert any("Occupies about" in e for e in r.evidence)

    def test_wider_voice_reads_wider(self):
        """+/- 4.5 kHz of audio (broadcast AM) reads about 9 kHz wide."""
        sig = am_voice(band=(300.0, 4500.0))
        for seed in range(3):
            x, _ = S.add_noise(sig, FS, 20, 9e3, seed=11 * seed + 20)
            r = classify(x)
            assert r.kind == "am"
            assert 8.0e3 <= r.bandwidth_hz <= 10.5e3, r.bandwidth_hz

    @pytest.mark.parametrize("side", [1, -1])
    def test_a_station_beside_it_is_not_its_sidebands(self, side):
        """Two AM stations 10 kHz apart, each +/- 4.5 kHz, the one beside
        the tuned one 10 dB stronger (Identify describes that one: its
        sidebands reach within the AM reach). Its width is its own, not the
        two stations': the spectrum mirrored about its carrier held the
        other station's sidebands too, on one side only, and they counted
        as its own (29.6 kHz wide, and Apply Mode's AM filter 15 kHz; 11.7
        kHz and 12.9 kHz before its sidebands were judged mirrored)."""
        rate = 100e3
        n = int(0.5 * rate)
        a = S.voice(n, rate, seed=1, band=(300.0, 4500.0))
        b = S.voice(n, rate, seed=2, band=(300.0, 4500.0))
        stronger = 10 ** (10 / 20) * S.am(n, rate, 0.7, b, offset=side * 10e3)
        sig = S.am(n, rate, 0.7, a) + stronger
        _width, reach = analysis_band("AM", 10e3, 2.4e6, 1.0e6)
        for seed in range(3):
            x, _ = S.add_noise(sig, rate, 30, 9e3, seed=97 * seed + 5)
            ident = identify(x, rate, 1.0e6, reach, band_hz=(-0.45 * rate, 0.45 * rate))
            r = ident.result
            assert ident.kind == "am", (seed, ident.summary())
            assert r.center_offset_hz * side > 8e3, r.center_offset_hz
            assert 8e3 <= r.bandwidth_hz <= 12e3, (seed, r.bandwidth_hz)
            assert ident.filter_hz <= 13e3, (seed, ident.filter_hz)

    @pytest.mark.parametrize("snr", [10, 20, 30, 45])
    def test_a_steady_carrier_is_still_a_carrier(self, snr):
        """(Guard) A carrier with nothing on it stays a carrier: judged
        over a wider band where there are sidebands, it never gains any."""
        for seed in range(4):
            x, _ = S.add_noise(S.carrier(N, FS, 150.0), FS, snr, 50.0, seed=seed)
            r = classify(x)
            assert r.kind == "carrier", (snr, seed, r.kind, r.features.get("scores"))


# --------------------------------------------------------------------------- #
# FSK: two tones about the tuned frequency
# --------------------------------------------------------------------------- #


class TestFskTonesArePaired:
    def fsk_at(self, snr, seed):
        sig = S.fsk(N, FS, 1200, 4500)
        return S.add_noise(sig, FS, snr, 12e3, seed=7919 * seed + snr + 17)[0]

    def test_fsk_at_8_db_is_fsk(self):
        """FSK at 8 dB was taken for OOK, "confident" 0.78, in 3 of 10
        takes: the noise in the dip between the tones joined to one tone's
        group, which then "held" the tuned frequency alone, and that tone,
        on half the time, looked keyed."""
        for seed in range(10):
            r = classify(self.fsk_at(8, seed))
            assert r.kind == "fsk", (seed, r.kind, round(r.confidence, 2))
            assert abs(r.center_offset_hz) < 1.5e3, r.center_offset_hz
            assert r.bandwidth_hz > 8e3, r.bandwidth_hz

    def test_unequal_humps_either_side_are_not_one_signal(self):
        """(Guard) Two carriers either side, one 10 dB weaker: not a pair
        of FSK tones (they must be about as strong)."""
        t = np.arange(N) / FS
        sig = np.exp(2j * np.pi * 4500 * t) + 0.316 * np.exp(-2j * np.pi * 4500 * t)
        x, _ = S.add_noise(sig, FS, 30, 100.0, seed=3)
        r = classify(x, search_hz=6e3)
        assert r.kind == "carrier", r.kind
        assert r.center_offset_hz == pytest.approx(4500, abs=100)

    def test_tones_beyond_the_reach_are_not_paired(self):
        """(Guard) FSK whose tones lie beyond the search reach: the tuned
        frequency is between two signals, not on one. (Their inner edges'
        symmetry joins them as before; this pins that the tones' own
        symmetry adds no pair the reach rules out.)"""
        from sdr_module.dsp.classifiers import _symmetric_pair

        excess = np.zeros(2048)
        excess[400:420] = 1.0  # a hump 604-624 bins below the middle
        excess[1628:1648] = 1.0  # and as far above
        starts = np.array([400, 1628])
        lengths = np.array([20, 20])
        powers = np.array([20.0, 20.0])
        lo, hi = np.array([400, 1628]), np.array([419, 1647])
        clear = np.ones(2, dtype=bool)
        args = (starts, lengths, excess, powers, lo, hi, clear)
        assert _symmetric_pair(*args, reach_bins=700.0, gap=30) == (0, 1)
        assert _symmetric_pair(*args, reach_bins=500.0, gap=30) is None


# --------------------------------------------------------------------------- #
# Short bursts near the noise (Mode S)
# --------------------------------------------------------------------------- #


class TestShortBurstsNearTheNoise:
    def mode_s_at(self, snr, seed):
        rate = 2.4e6
        sig = S.mode_s_replies(int(0.1 * rate), rate, 10)
        x, _ = S.add_noise(
            sig, rate, snr, 2e6, seed=104729 * seed + snr + 3, on_power=1.0
        )
        return x, rate

    @pytest.mark.parametrize("snr", [4, 5])
    def test_mode_s_replies_at_4_and_5_db(self, snr):
        """Ten replies in 0.1 s at 4-5 dB: each stands 16-18 dB above the
        noise once cut narrow, but the keying wanted ten times as many
        samples above its threshold as noise gives, plus 20 (61): ten
        replies hold 55-60. They read "Only a trace" in 14 and 6 of 20
        takes."""
        kinds = []
        for seed in range(10):
            x, rate = self.mode_s_at(snr, seed)
            kinds.append(classify(x, rate=rate).kind)
        assert kinds.count("pulsed") >= 9, kinds

    @pytest.mark.parametrize("seed", [0, 1])
    def test_a_weak_signal_always_there_is_not_keyed(self, seed):
        """(Guard) A signal that is always there, near the noise (BPSK at
        0 dB), crosses the keying threshold more often than noise does: it
        still needs ten times the noise's count (its peaks stand barely
        above the threshold), or it reads "OOK data bursts, probably"."""
        sig = S.psk(N, FS, 2400, 2)
        x, _ = S.add_noise(sig, FS, 0, 3.3e3, seed=1000 * seed)
        assert classify(x).kind != "ook"

    @pytest.mark.parametrize(
        "rate,n", [(48e3, 24000), (250e3, 100000), (2.4e6, 240000)]
    )
    def test_noise_is_never_keyed(self, rate, n):
        """(Guard) Noise alone, white or through a channel filter."""
        for seed in range(12):
            rng = np.random.default_rng(10_000 + seed)
            x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
                np.complex64
            )
            if seed % 2:
                x = S.channel_filter(x, rate, 0.3 * rate)
            assert classify(x, rate=rate).kind == "noise", seed
