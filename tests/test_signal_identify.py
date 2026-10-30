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

    def test_confidence_words(self):
        assert confidence_words(0.9) == "confident"
        assert confidence_words(0.6) == "probably"
        assert confidence_words(0.35) == "maybe"
        assert confidence_words(0.1) == "can't tell"

    def test_analysis_band(self):
        assert analysis_band("FM", 12.5e3, 2.4e6) == (100e3, 9375.0)
        width, reach = analysis_band("None (I/Q)", 2.4e6, 2.4e6)
        assert width == 2.4e6 and reach == 25e3


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
    dev.set_sample_rate(rate)
    dev.set_frequency(lo)
    dev.start_rx()
    for _ in range(int(round(start_s * rate / 240000))):
        dev.read_samples(240000)
    width, reach = analysis_band(mode, channel_hz, rate)
    ident = SignalIdentifier(rate, freq - lo, width)
    best, takes = None, 0
    while ident.seconds < WAIT_S:
        ident.feed(dev.read_samples(80000))
        if ident.ready():
            result = identify(
                ident.take(), ident.rate, freq, reach, band_hz=ident.passband_hz
            )
            best = better(best, result)
            takes += 1
            if best.is_signal and (best.confidence >= 0.6 or takes >= 3):
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
