"""Signal identification (T10): SignalClassifier and dsp.signal_id.

Regressions for FG-B11 (noise called OOK / pulsed with confidence 1.0 and
50-150 dB "SNRs", NFM called LSB or pulsed, BPSK called FM), a confusion
summary on synthesized signals, and the demo device's signals identified
the way the window does it.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402

from sdr_module.dsp.classifiers import (  # noqa: E402
    MAX_CONFIDENCE,
    SignalClassifier,
    SignalType,
    noise_psd_from_bin_db,
)
from sdr_module.dsp.demodulators import ModulationType  # noqa: E402
from sdr_module.dsp.signal_id import (  # noqa: E402
    WAIT_S,
    SignalIdentifier,
    analysis_band,
    better,
    confidence_words,
    identify,
    listening_for,
    takes_needed,
)

FS = 48_000.0
N = int(0.5 * FS)


def classify(x, rate=FS, **kw):
    return SignalClassifier(rate).classify(np.asarray(x, dtype=np.complex64), **kw)


# --------------------------------------------------------------------------- #
# FG-B11: the noise is measured honestly
# --------------------------------------------------------------------------- #


class TestNoiseIsNoise:
    def test_channel_filtered_noise_is_noise_not_ook(self):
        """FG-B11: noise through a channel filter read digital/OOK at
        confidence 1.00 and SNR 142 dB (the stopband taken for noise)."""
        x = S.channel_filter(S.noise(N, 1e-3), FS, 0.125 * FS)
        r = classify(x)
        assert r.kind == "noise"
        assert r.signal_type == SignalType.NOISE
        assert r.confidence <= MAX_CONFIDENCE < 1.0
        assert r.snr_db < 10.0  # not 50-150 dB

    def test_white_noise_is_noise(self):
        r = classify(S.noise(N, 1.0))
        assert r.kind == "noise"
        assert r.snr_db < 10.0

    @pytest.mark.parametrize("seed", range(6))
    def test_noise_never_called_a_signal(self, seed):
        x = S.noise(N, 1e-2, seed=seed)
        if seed % 2:
            x = S.channel_filter(x, FS, 0.2 * FS)
        assert classify(x).kind == "noise"

    def test_snr_of_a_filtered_signal_is_plausible(self):
        """A 20 dB NFM signal in a filtered channel reads about 20 dB, not
        the 148 dB the stopband gave."""
        sig = S.fm(N, FS, 3000, S.voice(N, FS, seed=2))
        x, _n0 = S.add_noise(sig, FS, 20.0, 10e3)
        r = classify(S.channel_filter(x, FS, 0.3 * FS))
        assert r.kind == "fm"
        assert 12.0 < r.snr_db < 30.0

    def test_confidence_is_never_one(self):
        r = classify(S.carrier(N, FS, 100.0))
        assert r.confidence <= MAX_CONFIDENCE

    def test_noise_from_the_caller(self):
        """The window can pass the noise density from its spectrum."""
        sig = S.fm(N, FS, 2500, S.tones(N, FS))
        x, n0 = S.add_noise(sig, FS, 25.0, 8e3)
        r = classify(x, noise_psd=n0)
        assert r.kind == "fm"
        assert r.features["noise_source"] == "given"
        assert 18.0 < r.snr_db < 32.0

    def test_noise_psd_from_display_bins(self):
        # White noise of total power s2 reads s2 * 1.5 / N per Hann bin.
        s2, n, rate = 1e-4, 2048, 2.4e6
        bin_db = 10 * np.log10(s2 * 1.5 / n)
        assert noise_psd_from_bin_db(bin_db, n, rate) == pytest.approx(s2 / rate)


# --------------------------------------------------------------------------- #
# Kinds of signal
# --------------------------------------------------------------------------- #


def _case(name: str):
    """(signal, rate, occupied Hz, on-power or None, expected kind)."""
    a = S.voice(N, FS, seed=1)
    if name == "carrier":
        return S.carrier(N, FS, 150.0), FS, 50.0, None, "carrier"
    if name == "cw":
        return S.cw(2 * N, FS, 80.0), FS, 100.0, 1.0, "cw"
    if name == "am":
        return S.am(N, FS, 0.7, a), FS, 6e3, None, "am"
    if name == "usb":
        audio = S.voice(N, FS, seed=3, band=(300, 2700))
        return S.ssb(N, FS, audio), FS, 2.5e3, None, "ssb"
    if name == "nfm":
        return S.fm(N, FS, 3000, a), FS, 10e3, None, "fm"
    if name == "fsk":
        return S.fsk(N, FS, 1200, 4500), FS, 12e3, None, "fsk"
    if name == "bpsk":
        return S.psk(N, FS, 2400, 2), FS, 3.3e3, None, "psk"
    if name == "qpsk":
        return S.psk(N, FS, 4800, 4), FS, 6.5e3, None, "psk"
    if name == "ook":
        rate = 250e3
        pulses = S.ev1527_pulses(0x5E3A1C)
        return S.ook_packet(int(0.4 * rate), rate, pulses, 6), rate, 8e3, 1.0, "ook"
    if name == "modes":
        rate = 2.4e6
        return S.mode_s_replies(int(0.1 * rate), rate, 10), rate, 2e6, 1.0, "pulsed"
    if name == "wfm":
        rate = 480e3
        n = int(0.5 * rate)
        audio = S.voice(n, rate, seed=5, band=(50, 15000))
        return S.fm(n, rate, 75e3, audio), rate, 180e3, None, "fm"
    raise KeyError(name)


KINDS = (
    "carrier",
    "cw",
    "am",
    "usb",
    "nfm",
    "fsk",
    "bpsk",
    "qpsk",
    "ook",
    "modes",
    "wfm",
)


def _run(snrs) -> Tuple[Dict[str, Counter], List[Tuple[str, int, str, float]]]:
    confusion: Dict[str, Counter] = {}
    rows = []
    for name in KINDS:
        sig, rate, occupied, on_power, want = _case(name)
        for snr in snrs:
            x, _n0 = S.add_noise(sig, rate, snr, occupied, seed=snr, on_power=on_power)
            r = SignalClassifier(rate).classify(x)
            confusion.setdefault(want, Counter())[r.kind] += 1
            rows.append((name, snr, r.kind, r.confidence))
    return confusion, rows


def _summary(confusion: Dict[str, Counter]) -> str:
    kinds = sorted({k for c in confusion.values() for k in c} | set(confusion))
    lines = ["true \\ said  " + " ".join(f"{k[:7]:>7s}" for k in kinds)]
    for want in sorted(confusion):
        row = confusion[want]
        lines.append(f"{want:12s} " + " ".join(f"{row.get(k, 0):7d}" for k in kinds))
    return "\n".join(lines)


class TestConfusion:
    def test_every_kind_at_15_and_30_db(self):
        """All eleven kinds are named right at 15 and 30 dB SNR (in the
        occupied bandwidth), each with at least "probably" confidence."""
        confusion, rows = _run((30, 15))
        print("\n" + _summary(confusion))
        wrong = [(n, s, k) for n, s, k, _c in rows if k != _case(n)[4]]
        assert not wrong, f"misnamed: {wrong}\n{_summary(confusion)}"
        unsure = [(n, s, round(c, 2)) for n, s, k, c in rows if c < 0.4]
        assert not unsure, unsure

    def test_mostly_right_at_10_db_and_honest_when_wrong(self):
        confusion, rows = _run((10,))
        right = sum(1 for n, _s, k, _c in rows if k == _case(n)[4])
        assert right >= 0.75 * len(rows), _summary(confusion)
        for name, _snr, kind, conf in rows:
            if kind != _case(name)[4]:
                # A wrong guess is never a confident one.
                assert conf < 0.75, (name, kind, conf)


class TestModulationLabels:
    def test_bpsk_and_qpsk_orders(self):
        sig, rate, occ, _p, _w = _case("bpsk")
        x, _ = S.add_noise(sig, rate, 25, occ)
        assert classify(x, rate).modulation == ModulationType.BPSK
        sig, rate, occ, _p, _w = _case("qpsk")
        x, _ = S.add_noise(sig, rate, 25, occ)
        assert classify(x, rate).modulation == ModulationType.QPSK

    def test_evidence_and_label(self):
        sig, rate, occ, _p, _w = _case("nfm")
        x, _ = S.add_noise(sig, rate, 25, occ)
        r = classify(x, rate)
        assert r.label == "FM"
        assert any("frequency swings" in e for e in r.evidence)

    def test_search_reach_keeps_a_neighbour_out(self):
        """Only a signal at the tuned frequency counts when asked: a strong
        carrier 15 kHz away is "nothing at the tuned frequency"."""
        x, _ = S.add_noise(S.carrier(N, FS, 15e3), FS, 30, 50.0)
        r = classify(x, search_hz=3e3)
        assert r.kind == "noise"
        assert r.features["nearest_signal_hz"] == pytest.approx(15e3, abs=500)
        assert "nearest signal" in r.evidence[0]

    def test_short_input_is_unknown(self):
        assert classify(np.ones(10, dtype=np.complex64)).kind == "unknown"


class TestTransmissionInTheCapture:
    """A voice transmission that starts or stops during the capture is that
    transmission, not Morse code: the clicks the noise makes in the quiet
    part are not keying. (An AM station keying up read "CW (Morse code)",
    confident 95 %, so Identify on an airband channel between overs said
    CW: the first capture with the signal holds its start.)"""

    @pytest.mark.parametrize(
        "on,snr", [((0.3, 1.0), 15), ((0.8, 1.0), 30), ((0.0, 0.6), 30)]
    )
    def test_am_keying_up_or_down_is_am(self, on, snr):
        n = int(1.0 * FS)
        sig = S.am(n, FS, 0.7, S.voice(n, FS, seed=4))
        gate = np.zeros(n)
        gate[int(on[0] * n) : int(on[1] * n)] = 1.0
        power = float(np.mean(np.abs(sig) ** 2))
        x, _ = S.add_noise(sig * gate, FS, snr, 6e3, seed=snr, on_power=power)
        r = classify(x)
        assert r.kind == "am", (r.kind, r.confidence, r.features.get("scores"))
        assert r.features["transmission"]


def test_demo_airband_keying_up_is_am():
    """The demo's 119.1 MHz airband station keys up about 0.7 s into the
    window's listening: AM (it read CW, 95 %)."""
    ident = identify_demo(119.1e6, 119.1e6, "AM", 10e3, start_s=5.0)
    assert ident.kind == "am", ident.summary()
    assert ident.mode == "AM"


# --------------------------------------------------------------------------- #
# Listening suggestions
# --------------------------------------------------------------------------- #


class TestListening:
    def test_narrow_fm(self):
        sig, rate, occ, _p, _w = _case("nfm")
        x, _ = S.add_noise(sig, rate, 25, occ)
        ident = listening_for(classify(x, rate), 162.55e6)
        assert ident.mode == "FM"
        assert ident.fm_deviation == "5 kHz"
        assert 12.5e3 <= ident.filter_hz <= 25e3
        assert ident.label.startswith("Narrow FM")

    def test_broadcast_fm(self):
        sig, rate, occ, _p, _w = _case("wfm")
        x, _ = S.add_noise(sig, rate, 30, occ)
        ident = listening_for(classify(x, rate), 100.1e6)
        assert (ident.label, ident.mode, ident.fm_deviation) == (
            "Broadcast FM",
            "FM",
            "75 kHz",
        )
        assert ident.filter_hz == 200e3

    def test_ssb_side_by_frequency(self):
        sig, rate, occ, _p, _w = _case("usb")
        x, _ = S.add_noise(sig, rate, 25, occ)
        r = classify(x, rate)
        assert listening_for(r, 14.2e6).mode == "USB"
        assert listening_for(r, 7.1e6).mode == "LSB"

    def test_mode_s_at_1090(self):
        sig, rate, occ, _p, _w = _case("modes")
        x, _ = S.add_noise(sig, rate, 25, occ, on_power=1.0)
        ident = listening_for(classify(x, rate), 1090e6)
        assert ident.label == "Mode S / ADS-B replies"
        assert ident.mode == "None (I/Q)"

    @pytest.mark.parametrize("mode,channel", [("FM", 12.5e3), ("None (I/Q)", 25e3)])
    def test_mode_s_through_a_narrow_filter(self, mode, channel):
        """Identify at 1090 MHz with the usual narrow filter: each reply
        shows as one burst, which read "OOK data bursts, as remote controls
        and sensors send", confident 95 %, with a remote's pulse analysis.
        It is Mode S, and Apply Mode widens the filter to see the pulses."""
        ident = identify_demo(1090e6, 1090e6, mode, channel)
        assert ident.label == "Mode S / ADS-B replies", ident.summary()
        assert (ident.mode, ident.filter_hz) == ("None (I/Q)", 2e6)
        assert ident.pulses is None
        assert not any("remote controls" in d for d in ident.details)

    @pytest.mark.parametrize("freq", [660e3, 1.2e6, 118.1e6, 125e6])
    def test_a_carrier_in_an_am_band_is_heard_in_am(self, freq):
        """A weak AM station's sidebands sink into the noise before its
        carrier does, so Identify can call it an unmodulated carrier. In a
        band heard in AM (broadcast, airband) Apply Mode then listens the
        band's way, not in CW with a 500 Hz filter that cuts the voice."""
        x, _ = S.add_noise(S.carrier(N, FS, 150.0), FS, 30, 50.0)
        r = classify(x, FS)
        assert r.kind == "carrier"
        ident = listening_for(r, freq)
        assert (ident.mode, ident.filter_hz) == ("AM", 10e3)

    @pytest.mark.parametrize("freq", [7.03e6, 99.82e6, 146.52e6, 433.92e6])
    def test_a_carrier_elsewhere_is_heard_in_cw(self, freq):
        """Outside the AM bands a steady carrier is listened to in CW, as
        before: a key held down (the demo's 99.82 MHz beacon), a tuning
        carrier, or a band whose own mode a carrier says nothing about."""
        x, _ = S.add_noise(S.carrier(N, FS, 150.0), FS, 30, 50.0)
        ident = listening_for(classify(x, FS), freq)
        assert (ident.mode, ident.filter_hz) == ("CW", 500.0)

    def test_confidence_words(self):
        assert confidence_words(0.9) == "confident"
        assert confidence_words(0.6) == "probably"
        assert confidence_words(0.35) == "maybe"
        assert confidence_words(0.1) == "can't tell"

    def test_analysis_band(self):
        assert analysis_band("FM", 12.5e3, 2.4e6) == (100e3, 9375.0)
        width, reach = analysis_band("None (I/Q)", 2.4e6, 2.4e6)
        assert width == 2.4e6 and reach == 25e3


class TestDeviationThatHoldsThePeak:
    """P-01: Identify suggested "2.5 kHz" for any FM swing under 3.5 kHz
    (the demo's 100.715 MHz call swings +/-3.4 kHz), the band's own
    deviation for swings up to 1.6 times it, and "5 kHz" for FSK data up
    to +/-6 kHz. Half a second after Apply Mode the window warned "FM
    audio is clipping: this station deviates more than 2.5 kHz". The
    suggestion is the smallest offered deviation that holds the measured
    peak with a margin; the band's only when it holds it too."""

    @staticmethod
    def result(peak_hz, width_hz, kind="fm", rms_hz=0.0, p95_hz=None):
        from sdr_module.dsp.classifiers import ClassificationResult

        features = {"freq_peak_hz": peak_hz, "freq_dev_hz": rms_hz}
        if p95_hz is not None:
            features["freq_p95_hz"] = p95_hz
        return ClassificationResult(
            SignalType.ANALOG, None, 0.9, width_hz, 0.0, 30.0, features, kind=kind
        )

    @pytest.mark.parametrize(
        "peak,want",
        [
            (1.5e3, "2.5 kHz"),
            (2.2e3, "2.5 kHz"),
            (2.4e3, "5 kHz"),
            (3.28e3, "5 kHz"),  # the demo's 100.715 MHz call, as measured
            (4.5e3, "5 kHz"),
            (4.7e3, "12.5 kHz"),
            (11e3, "12.5 kHz"),
            (12e3, "25 kHz"),
            (22e3, "25 kHz"),
            (24e3, "75 kHz"),
        ],
    )
    def test_the_smallest_choice_that_holds_the_peak(self, peak, want):
        # (300 MHz: no band plan there.)
        ident = listening_for(self.result(peak, 2.0 * peak + 6e3), 300e6)
        assert ident.mode == "FM"
        assert ident.fm_deviation == want

    @pytest.mark.parametrize(
        "freq,peak,want",
        [
            (462.6e6, 2.0e3, "2.5 kHz"),  # FRS: its 2.5 kHz holds it
            (462.6e6, 3.2e3, "5 kHz"),  # swings wider than FRS's 2.5 kHz
            (146.0e6, 1.5e3, "5 kHz"),  # 2 m: the band's 5 kHz holds it
            (146.0e6, 6.0e3, "12.5 kHz"),  # 2 m, but beyond its 5 kHz
            (100.715e6, 3.28e3, "5 kHz"),  # a narrow call in the FM band
        ],
    )
    def test_the_bands_deviation_only_when_it_holds_the_peak(self, freq, peak, want):
        ident = listening_for(self.result(peak, 2.0 * peak + 6e3), freq)
        assert ident.fm_deviation == want

    def test_fsk_data(self):
        """The demo's POCSAG pager (+/-4.5 kHz FSK; its rms swing reads
        4.4 kHz) was suggested 5 kHz, which clipped a fifth of its audio."""
        ident = listening_for(self.result(0.0, 12e3, "fsk", rms_hz=4.42e3), 152.24e6)
        assert (ident.mode, ident.fm_deviation) == ("FM", "12.5 kHz")
        ident = listening_for(self.result(0.0, 6e3, "fsk", rms_hz=1.4e3), 152.24e6)
        assert ident.fm_deviation == "2.5 kHz"

    def test_the_label_still_says_what_it_is(self):
        """Wide FM stays wide FM (heard at 75 kHz, which holds its swing);
        broadcast FM is still told by its width or a broadcast swing."""
        ident = listening_for(self.result(30e3, 66e3), 300e6)
        assert (ident.label, ident.fm_deviation) == ("Wide FM", "75 kHz")
        ident = listening_for(self.result(45e3, 90e3), 300e6)
        assert (ident.label, ident.fm_deviation, ident.filter_hz) == (
            "Broadcast FM",
            "75 kHz",
            200e3,
        )
        ident = listening_for(self.result(20e3, 60e3), 100.1e6)
        assert (ident.label, ident.fm_deviation) == ("Broadcast FM", "75 kHz")
        ident = listening_for(self.result(3.28e3, 8.9e3), 100.715e6)
        assert ident.label == "Narrow FM (voice or data)"
        assert (ident.fm_deviation, ident.filter_hz) == ("5 kHz", 25e3)

    def test_the_card_names_the_peak_the_deviation_holds(self):
        """The card's "the frequency swings about ±1.8 kHz" is the rms
        swing: next to "Listen with: FM · 5 kHz deviation" it read as if
        2.5 kHz would do. It also names the peak the deviation holds."""
        ident = listening_for(self.result(3.28e3, 8.9e3, rms_hz=1.77e3), 100.715e6)
        assert ident.fm_deviation == "5 kHz"
        assert "The swing peaks at about ±3.3 kHz" in ident.details
        # (Estimated from the rms swing, not measured: not said.)
        ident = listening_for(self.result(0.0, 8.9e3, rms_hz=1.77e3), 300e6)
        assert not [d for d in ident.details if "peaks" in d]

    @pytest.mark.parametrize("freq", [146.52e6, 300e6])
    @pytest.mark.parametrize("peak", [4.6e3, 4.8e3, 5.0e3])
    def test_voice_reaching_5_khz_now_and_then_is_heard_at_5_khz(self, freq, peak):
        """N-01: with 10 % to spare over the peak, a 2 m voice station
        whose radio limits it at +/-5 kHz (measured peak 4.6-5 kHz) was
        suggested 12.5 kHz: Apply Mode made it about 8 dB quieter. Voice
        reaches its peak only now and then (the swing exceeded 5 % of the
        time is well below it), so at 5 kHz it clips far less than the
        clipping warning's 5 % of the time."""
        ident = listening_for(self.result(peak, 15e3, p95_hz=0.75 * peak), freq)
        assert (ident.mode, ident.fm_deviation) == ("FM", "5 kHz")
        assert ident.filter_hz == 25e3
        assert f"The swing peaks at about ±{peak / 1e3:.1f} kHz" in ident.details

    def test_a_tone_near_a_choice_keeps_the_margin(self):
        """A tone stays near its peak (the swing exceeded 5 % of the time
        is the peak, near enough): measured up to 10 % low, just beyond the
        deviation it clips most of the time, which the clipping warning
        catches. So it keeps the margin, and the card says why."""
        ident = listening_for(self.result(4.8e3, 15e3, p95_hz=4.78e3), 146.52e6)
        assert ident.fm_deviation == "12.5 kHz"
        assert (
            "The swing peaks at about ±4.8 kHz and stays near it: 5 kHz could clip"
            in ident.details
        )
        # Below the margin: the smallest choice holds it.
        ident = listening_for(self.result(4.4e3, 15e3, p95_hz=4.38e3), 300e6)
        assert ident.fm_deviation == "5 kHz"
        assert "The swing peaks at about ±4.4 kHz" in ident.details

    def test_the_bands_choice_needs_no_explaining(self):
        """2 m's 5 kHz for voice peaking at 2.4 kHz: the band plan's choice,
        not because 2.5 kHz would clip (off any band it is suggested)."""
        result = self.result(2.4e3, 10e3, p95_hz=1.6e3)
        ident = listening_for(result, 146.52e6)
        assert ident.fm_deviation == "5 kHz"
        assert "The swing peaks at about ±2.4 kHz" in ident.details
        assert listening_for(result, 300e6).fm_deviation == "2.5 kHz"

    def test_voice_beyond_a_choice(self):
        """Voice peaking beyond 5 kHz isn't held by it: 12.5 kHz, and the
        card says why when the peak reads close to 5 kHz."""
        ident = listening_for(self.result(5.3e3, 16e3, p95_hz=3.6e3), 146.52e6)
        assert ident.fm_deviation == "12.5 kHz"
        assert "The swing peaks at about ±5.3 kHz: 5 kHz could clip" in ident.details
        ident = listening_for(self.result(7e3, 20e3, p95_hz=4.0e3), 300e6)
        assert ident.fm_deviation == "12.5 kHz"
        assert "The swing peaks at about ±7.0 kHz" in ident.details


# --------------------------------------------------------------------------- #
# The demo device's signals, identified as the window does
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _demo_never_skips(monkeypatch):
    """The demo device skips ahead when read slower than real time (a busy
    machine): here every sample is read, so the timing is fixed."""
    from sdr_module.gui import device_dialog

    monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)


def identify_demo(freq, lo, mode, channel_hz, rate=2.4e6, start_s=0.0):
    """The window's policy: listen in 33 ms blocks, judge every half
    second, stop at a confident signal (or after three judgements of one,
    or WAIT_S seconds). ``start_s``: seconds of the demo let go by first."""
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev._rng = np.random.default_rng(1)  # (the same noise every run)
    dev.set_sample_rate(rate)
    dev.set_frequency(lo)
    dev.start_rx()
    for _ in range(int(round(start_s * rate / 240000))):
        dev.read_samples(240000)
    width, reach = analysis_band(mode, channel_hz, rate, freq)
    ident = SignalIdentifier(rate, freq - lo, width)
    best, kinds = None, []
    while ident.seconds < WAIT_S:
        ident.feed(dev.read_samples(80000))
        if ident.ready():
            result = identify(
                ident.take(), ident.rate, freq, reach, band_hz=ident.passband_hz
            )
            best = better(best, result)
            kinds.append(result.kind)
            agree = kinds.count(best.kind)
            if best.is_signal and (
                (best.confidence >= 0.6 and agree >= takes_needed(best))
                or len(kinds) >= 3
            ):
                break
    return best


DEMO = [
    ("broadcast FM", 100.1e6, 100.3e6, "FM", 200e3, "fm"),
    ("NOAA narrow FM", 162.55e6, 162.5e6, "FM", 25e3, "fm"),
    ("airband AM", 125.0e6, 125.2e6, "AM", 10e3, "am"),
    ("CW beacon", 99.82e6, 100.1e6, "CW", 500.0, "cw"),
    ("OOK remote", 433.92e6, 433.92e6, "None (I/Q)", 500e3, "ook"),
    ("ADS-B", 1090e6, 1090e6, "None (I/Q)", 2.4e6, "pulsed"),
    ("empty channel", 433.5e6, 433.92e6, "None (I/Q)", 500e3, "noise"),
]


@pytest.mark.parametrize(
    "name,freq,lo,mode,channel,want", DEMO, ids=[d[0] for d in DEMO]
)
def test_demo_signals(name, freq, lo, mode, channel, want):
    ident = identify_demo(freq, lo, mode, channel)
    assert ident.kind == want, (
        name,
        ident.summary(),
        ident.result.features.get("scores"),
    )
    if want != "noise":
        assert ident.confidence >= 0.5, ident.summary()


def test_demo_ook_remote_is_taken_apart():
    """The demo's 433.92 MHz remote: pulse width modulation, 350/1050 us,
    and its code (rtl_433 counts a short pulse as 1)."""
    ident = identify_demo(433.92e6, 433.92e6, "None (I/Q)", 500e3)
    pulses = ident.pulses
    assert pulses is not None
    assert pulses.modulation == "OOK PWM"
    assert pulses.short_us == pytest.approx(350, abs=40)
    assert pulses.long_us == pytest.approx(1050, abs=60)
    bits, count = pulses.distinct_rows()[0]
    # 0x5E3A1C sent, short pulse = 1: inverted, plus the sync pulse.
    assert bits[:24] == format(0x5E3A1C ^ 0xFFFFFF, "024b")
    assert count >= 2


def test_demo_narrow_fm_call_is_heard_at_5_khz():
    """P-01: the demo's 100.715 MHz call swings +/-3.4 kHz: Identify said
    2.5 kHz, and Apply Mode led straight into "FM audio is clipping"."""
    ident = identify_demo(100.715e6, 100.715e6, "FM", 12.5e3)
    assert ident.kind == "fm", ident.summary()
    assert (ident.mode, ident.fm_deviation) == ("FM", "5 kHz")


def demo_takes(freq, lo, mode, channel_hz, start_s, seconds, rate=2.4e6):
    """Every take Identify judges in ``seconds`` of the demo from
    ``start_s`` (as identify_demo, but all of them, and the stream follows
    the samples read: never skipped ahead to the clock, however slowly
    this computer runs)."""
    import time

    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev._rng = np.random.default_rng(1)  # (the same noise every run)
    dev.set_sample_rate(rate)
    dev.set_frequency(lo)
    dev.start_rx()
    dev._stream_epoch = time.monotonic() + 1e9  # (always ahead of the clock)
    for _ in range(int(round(start_s * rate / 240000))):
        dev.read_samples(240000)
    width, reach = analysis_band(mode, channel_hz, rate, freq)
    ident = SignalIdentifier(rate, freq - lo, width)
    takes = []
    while ident.seconds < seconds:
        ident.feed(dev.read_samples(80000))
        if ident.ready():
            takes.append(
                identify(
                    ident.take(), ident.rate, freq, reach, band_hz=ident.passband_hz
                )
            )
    return takes


class TestTheSignalAtTheTunedFrequency:
    """W-02: typed into the FM band, 100.715 MHz (the demo's narrow FM
    call, on 5 s of every 9) is heard as broadcast FM, and Identify looks
    800 kHz wide with a 150 kHz reach. The noise's own ripple, taken for
    signal all over that band (its significance was reckoned as if every
    bin were independent), joined the call to the 101.1 MHz station 385
    kHz above: most takes with the call on described that as "Unknown
    signal", 382 kHz wide, "Centred 330.6 kHz above the tuned frequency"
    (and Identify said "Nothing identifiable"). With the call off, takes
    described a stretch of noise near the tuned frequency as "Unknown
    signal" instead of naming the station as the nearest signal."""

    CALL = (100.715e6, 100.1e6, "FM", 200e3)

    def test_the_call_while_it_is_on(self):
        # (On from 3.7 s to 8.7 s of the demo's stream.)
        takes = demo_takes(*self.CALL, start_s=4.0, seconds=4.5)
        assert len(takes) >= 8
        for ident in takes:
            assert ident.kind == "fm", (ident.summary(), ident.details)
            assert abs(ident.result.center_offset_hz) < 5e3, ident.details
            assert ident.label == "Narrow FM (voice or data)"

    def test_nothing_there_while_it_is_off(self):
        # (Off from 8.7 s to 12.7 s.)
        takes = demo_takes(*self.CALL, start_s=8.8, seconds=3.8)
        assert len(takes) >= 6
        for ident in takes:
            assert ident.kind == "noise", (ident.summary(), ident.details)
            # (The station, cut at the band's edge at 380 kHz: its power
            # starts about 330 kHz above.)
            nearest = ident.result.features.get("nearest_signal_hz")
            assert nearest is not None and 300e3 < nearest < 350e3, ident.details
            assert ident.details[0].startswith(
                "Nothing at the tuned frequency; the nearest signal is 3"
            ), ident.details

    @pytest.mark.parametrize("start_s", [11.9, 12.1])
    def test_the_call_keying_up(self, start_s):
        """W-03: Identify pressed just before the call keys up (12.7 s):
        joined to the noise around it, the call was judged as one signal
        about 70-150 kHz wide, uncut, and its start read as keying ("CW
        (Morse code) · maybe ... Occupies about 71.8 kHz", or OOK data
        bursts of 13 us pulses). It is the call, or nothing yet."""
        takes = demo_takes(*self.CALL, start_s=start_s, seconds=1.3)
        assert takes
        for ident in takes:
            assert ident.kind in ("fm", "noise"), (ident.summary(), ident.details)
        assert takes[-1].kind == "fm", takes[-1].summary()


# --------------------------------------------------------------------------- #
# The end-to-end journeys' problems (PLOTS-E2E-2, -3, -4)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("start_s", [5.0, 5.2, 6.0, 6.2, 6.4])
def test_demo_cw_beacon_keying_into_its_carrier_is_cw(start_s):
    """PLOTS-E2E-3: the 99.82 MHz beacon keys 'VVV DE SDR', then holds its
    key down for 2.4 s. Listening over its last letters, Identify said
    'Unmodulated carrier', confident 75 % (or AM), in about half the tries:
    it stopped at the first take, a long element or the key held down. Now
    keying heard in any take wins, and a carrier needs three takes. (From
    6.6 s on every take hears only silence and the key held down: a carrier
    switched on, nothing to tell it from one.)"""
    ident = identify_demo(99.82e6, 100.1e6, "CW", 500.0, start_s=start_s)
    assert ident.kind == "cw", ident.summary()


def test_keyed_into_a_held_key_is_cw_not_a_transmission():
    """Two dits, then the key held down: keyed, not a transmission
    starting (whose level, voice, varies)."""
    n = int(1.0 * FS)
    key = np.zeros(n)
    for start, stop in ((0.10, 0.18), (0.26, 0.34), (0.42, 1.0)):
        key[int(start * n) : int(stop * n)] = 1.0
    x, _ = S.add_noise(S.carrier(n, FS, 80.0) * key, FS, 30, 100.0, on_power=1.0)
    r = classify(x)
    assert not r.features["transmission"]
    assert r.kind == "cw", (r.kind, r.features.get("scores"))


def _switched(at, on=True, snr_db=30):
    """A narrow carrier switched on (or off) ``at`` (a fraction) into one
    second of signal, classified."""
    n = int(1.0 * FS)
    key = np.zeros(n)
    if on:
        key[int(at * n) :] = 1.0
    else:
        key[: int(at * n)] = 1.0
    x, _ = S.add_noise(S.carrier(n, FS, 80.0) * key, FS, snr_db, 100.0, on_power=1.0)
    return classify(x)


@pytest.mark.parametrize("at", [0.1, 0.2, 0.4])
def test_a_carrier_switched_on_is_never_confident(at):
    """A steady carrier keyed up while listening could be a CW station
    holding its key down: 'probably' at most."""
    r = _switched(at)
    assert r.features["transmission"]
    assert r.kind == "carrier"
    assert r.confidence <= 0.6


@pytest.mark.parametrize("on", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("at", [0.2, 0.4, 0.6])
def test_a_carrier_switched_once_is_not_keying(at, on):
    """Switched on or off once, a strong carrier is no Morse code: the
    channel is cut out through an FFT, whose ringing wraps the other end of
    the block round to this one, and that 'key-down' at the edge (20-30 ms
    at 40 dB) made it 'CW (Morse code)', confident 95 %. A key-down that
    counts lies whole inside the block."""
    r = _switched(at, on, snr_db=40)
    assert r.kind != "cw", (r.kind, r.confidence, r.features.get("on_runs_keyed"))


def test_keying_seen_once_is_kept():
    """Of several takes, keying wins over what it looks like between its
    keys (or across a gap in the stream), whatever their confidence."""
    n = int(0.5 * FS)
    carrier = identify(S.add_noise(S.carrier(n, FS, 80.0), FS, 30, 100.0)[0], FS, 1e6)
    cw = identify(
        S.add_noise(S.cw(n, FS, 80.0), FS, 30, 100.0, on_power=1.0)[0], FS, 1e6
    )
    assert (carrier.kind, cw.kind) == ("carrier", "cw")
    cw.result.confidence = 0.55
    assert better(carrier, cw) is cw
    assert better(cw, carrier) is cw
    assert takes_needed(carrier) == 3
    assert takes_needed(cw) == 1


def test_short_pulses_off_1090_wait_for_a_second_take():
    sig, rate, occ, power, _want = _case("modes")
    x, _ = S.add_noise(sig, rate, 25, occ, on_power=power)
    for freq, want in ((433.92e6, 2), (1090e6, 1)):
        ident = identify(x, rate, freq)
        if ident.kind == "pulsed":
            assert takes_needed(ident) == want


def test_a_gap_in_the_stream_starts_the_capture_afresh():
    """PLOTS-E2E-2: a capture spanning samples the receiver dropped (the
    demo skipping ahead after a stall) cut the remote's packets into what
    read 'Short pulses', the first take finishing on it. After a gap the
    held samples go, and the next capture is half a second of unbroken
    signal; how long Identify listened in all still counts."""
    rate = 240e3
    ident = SignalIdentifier(rate, 0.0, rate)
    block = np.ones(int(0.1 * rate), np.complex64)
    for _ in range(4):
        ident.feed(block)
    ident.gap()
    for _ in range(4):
        ident.feed(2 * block)
        assert not ident.ready()
    ident.feed(2 * block)
    assert ident.ready()
    capture = ident.take()
    assert len(capture) == int(0.5 * rate)
    assert np.all(capture == 2)
    assert ident.seconds == pytest.approx(0.9)


def test_1090_is_analysed_two_megahertz_wide_in_any_mode():
    """PLOTS-E2E-4: at 1090 MHz in FM with a 100 kHz filter the analysis
    was 400 kHz wide: each reply one burst, often 'Unknown' after 4.4 s of
    work. The replies' pulses show 2 MHz wide, whatever the mode."""
    assert analysis_band("FM", 100e3, 2.4e6, 1090e6)[0] == 2e6
    assert analysis_band("AM", 10e3, 2.4e6, 1090e6)[0] == 2e6
    assert analysis_band("FM", 100e3, 2.4e6, 100.1e6)[0] == 400e3
    assert analysis_band("FM", 100e3, 2.4e6)[0] == 400e3
    ident = identify_demo(1090e6, 1090e6, "FM", 100e3)
    assert ident.label == "Mode S / ADS-B replies", ident.summary()
    assert ident.seconds <= 1.0
