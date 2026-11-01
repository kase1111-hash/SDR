"""Bursts (T10): BurstDetector, the rtl_433-style pulse analysis, the
BurstWatcher on the demo device, and the reworked PacketHighlighter.

Regressions for FG-B14: LivePacketDisplay found nothing for about 12 s
(its noise floor started at -100 dB and rose 0.1 dB a frame), then flooded
out about 110 false "packets" at once and kept reporting fragments.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402

from sdr_module.dsp.bursts import (  # noqa: E402
    BurstDetector,
    BurstWatcher,
    analyze_pulses,
    analyze_report,
    bits_to_hex,
    format_analysis,
    frame_rows,
)

OOK_RATE = 250e3
CODE = 0x5E3A1C
# rtl_433 counts a short pulse as 1: the EV1527 code comes out inverted,
# with the sync pulse as a 25th bit.
CODE_BITS = format(CODE ^ 0xFFFFFF, "024b") + "1"


def _ook(pulses, repeats, snr_db=30.0, seed=7):
    sig = S.ook_packet(int(0.4 * OOK_RATE), OOK_RATE, pulses, repeats=repeats)
    x, _ = S.add_noise(sig, OOK_RATE, snr_db, 8e3, seed=seed, on_power=1.0)
    return x


# --------------------------------------------------------------------------- #
# Pulse analysis
# --------------------------------------------------------------------------- #


class TestPulseAnalysis:
    def test_ev1527_remote_is_pwm_with_its_code(self):
        a = analyze_pulses(_ook(S.ev1527_pulses(CODE), 6), OOK_RATE)
        assert a is not None
        assert a.modulation == "OOK PWM"
        assert a.short_us == pytest.approx(350, abs=30)
        assert a.long_us == pytest.approx(1050, abs=50)
        bits, count = a.distinct_rows()[0]
        assert bits == CODE_BITS
        assert count == 6
        assert a.summary.startswith("OOK PWM · ")
        assert "{25}a1c5e38 ×6" in a.summary

    def test_weather_sensor_is_ppm_with_its_code(self):
        code = 0x9A5C3E1F0
        a = analyze_pulses(_ook(S.ppm_sensor_pulses(code), 3), OOK_RATE)
        assert a is not None
        assert a.modulation == "OOK PPM"
        assert a.short_us == pytest.approx(1000, abs=60)
        assert a.long_us == pytest.approx(2000, abs=100)
        bits, count = a.distinct_rows()[0]
        assert bits == format(code, "036b")  # short gap = 0
        assert count == 3

    def test_still_right_at_15_db(self):
        a = analyze_pulses(_ook(S.ev1527_pulses(CODE), 6, snr_db=15.0), OOK_RATE)
        assert a is not None and a.modulation == "OOK PWM"
        assert a.distinct_rows()[0][0] == CODE_BITS

    def test_fsk_burst_is_fsk(self):
        x = S.fsk(int(0.1 * OOK_RATE), OOK_RATE, 9600, 20e3)
        sig = np.concatenate([np.zeros(5000), x, np.zeros(5000)])
        xx, _ = S.add_noise(sig, OOK_RATE, 25, 60e3, on_power=1.0)
        a = analyze_pulses(xx, OOK_RATE)
        assert a is not None
        assert a.modulation.startswith("FSK")
        assert a.fsk_shift_hz == pytest.approx(20e3, rel=0.25)

    def test_noise_has_no_pulses(self):
        assert analyze_pulses(S.noise(100_000, 1e-3), OOK_RATE) is None

    def test_written_up_like_rtl_433(self):
        a = analyze_pulses(_ook(S.ev1527_pulses(CODE), 6), OOK_RATE)
        text = format_analysis(a, 433.92e6)
        assert "Frequency: 433.920 MHz" in text
        assert "Pulse width distribution:" in text
        assert "Guessing modulation: Pulse width modulation" in text
        assert "-X 'n=name,m=OOK_PWM,s=3" in text
        assert "codes     : {25}a1c5e38  (x6)" in text

    def test_bits_to_hex(self):
        assert bits_to_hex("01011110") == "5e"
        assert bits_to_hex("1") == "8"  # padded to a whole nibble
        assert bits_to_hex("") == ""


# --------------------------------------------------------------------------- #
# The detector: no warm-up, no flood of noise bursts (FG-B14)
# --------------------------------------------------------------------------- #


def _noise_row(n=1024, seed=0, level_db=-80.0):
    rng = np.random.default_rng(seed)
    return level_db + 10 * np.log10(rng.exponential(1.0, n))


class TestBurstDetector:
    SPAN = 2.4e6

    def test_noise_alone_gives_no_bursts(self):
        det = BurstDetector()
        ended = []
        for i in range(600):  # 20 s of rows
            ended += det.process(_noise_row(seed=i), 433.92e6, self.SPAN, i / 30)
        ended += det.flush()
        assert ended == []

    def test_a_burst_in_the_second_row_is_found(self):
        """No warm-up: the floor is the first row."""
        det = BurstDetector()
        det.process(_noise_row(seed=1), 433.92e6, self.SPAN, 0.0)
        row = _noise_row(seed=2)
        row[600:640] = -40.0  # 94 kHz wide, 40 dB up, 252 kHz above centre
        ended = det.process(row, 433.92e6, self.SPAN, 1 / 30)
        assert ended == [] and len(det.active) == 1
        for i in range(3, 7):
            ended += det.process(_noise_row(seed=i), 433.92e6, self.SPAN, i / 30)
        assert len(ended) == 1
        burst = ended[0]
        bin_hz = self.SPAN / 1024
        assert burst.center_hz == pytest.approx(433.92e6 + 107.5 * bin_hz, abs=5e3)
        assert burst.width_hz == pytest.approx(40 * bin_hz, abs=3 * bin_hz)
        assert burst.snr_db > 30
        assert burst.peak_dbfs == pytest.approx(-40.0, abs=1)

    def test_a_station_on_from_the_start_is_no_burst(self):
        det = BurstDetector()
        ended = []
        for i in range(100):
            row = _noise_row(seed=i)
            row[300:340] = -30.0
            ended += det.process(row, 100e6, self.SPAN, i / 30)
        assert ended == [] and det.active == []

    def test_a_gain_change_is_no_burst(self):
        """The whole floor moving 10 dB (the gain raised) re-settles the
        floor instead of calling every bin a burst."""
        det = BurstDetector()
        ended = []
        for i in range(30):
            ended += det.process(_noise_row(seed=i), 100e6, self.SPAN, i / 30)
        for i in range(30, 60):
            row = _noise_row(seed=i, level_db=-70.0)
            ended += det.process(row, 100e6, self.SPAN, i / 30)
        ended += det.flush()
        assert ended == []

    def test_a_retune_starts_afresh(self):
        det = BurstDetector()
        row = _noise_row(seed=3)
        det.process(row, 433.92e6, self.SPAN, 0.0)
        row = row.copy()
        row[100:140] = -40.0
        det.process(row, 433.92e6, self.SPAN, 0.033)
        assert det.active
        ended = det.process(_noise_row(seed=4), 868e6, self.SPAN, 0.066)
        assert len(ended) == 1 and det.active == []

    @staticmethod
    def _station_row(i, rng, centre=320, core=12, reach=30):
        """A station always on (bins ``centre`` +/- ``core`` at -40 dBFS)
        whose modulation reaches out a random distance each row (-50
        dBFS), as a broadcast FM station's sidebands do with the music
        (on the noise of :meth:`test_noise_alone_gives_no_bursts`)."""
        row = _noise_row(seed=i)
        row[centre - core : centre + core + 1] = -40.0
        out = int(rng.integers(core, reach + 1))
        row[centre - out : centre - core] = -50.0
        row[centre + core + 1 : centre + out + 1] = -50.0
        return row

    def test_an_occupied_channel_coming_and_going_is_no_burst(self):
        """ROB-03: a station that never stops is not a burst, and neither is
        its modulation reaching further out now and then (the floor there
        is the station's average, above the noise). Before, the demo's
        100.1 MHz station gave 95 'bursts' in 20 s."""
        rng = np.random.default_rng(5)
        det = BurstDetector()
        ended = []
        for i in range(600):  # 20 s of rows
            row = self._station_row(i, rng)
            ended += det.process(row, 100.1e6, self.SPAN, i / 30)
        ended += det.flush()
        assert ended == []

    def test_a_burst_beside_an_occupied_channel_is_found(self):
        """A transmission that comes and goes in the next channel (14 kHz
        from the station's edge: no floor of its own) and one on the DC
        spike of the receiver are still bursts."""
        rng = np.random.default_rng(6)
        det = BurstDetector()
        ended = []
        for i in range(600):
            row = self._station_row(i, rng, reach=12)
            row[512] = -50.0  # the DC spike, always there
            if 300 <= i < 330:  # 1 s, from 10 s
                row[338:344] = -45.0  # 6 bins past the station's edge
            if 450 <= i < 455:
                row[510:515] = -35.0  # on the DC spike
            ended += det.process(row, 433.92e6, self.SPAN, i / 30)
        ended += det.flush()
        bins = sorted(
            (b.center_hz - 433.92e6) / (self.SPAN / 1024) + 512 for b in ended
        )
        assert len(ended) == 2, [(b.center_hz, b.duration_s) for b in ended]
        assert bins[0] == pytest.approx(340.5, abs=1.5)
        assert bins[1] == pytest.approx(512, abs=1.5)

    def test_occupied_db_zero_turns_it_off(self):
        """``occupied_db=0`` gives the detector as it was (every excursion
        10 dB over the floor is a burst)."""
        rng = np.random.default_rng(5)
        det = BurstDetector(occupied_db=0)
        ended = []
        for i in range(300):
            ended += det.process(self._station_row(i, rng), 100.1e6, self.SPAN, i / 30)
        assert ended


def test_frame_rows_show_a_short_burst_anywhere_in_the_block():
    rate = 2.4e6
    x = S.noise(80_000, 1e-6)
    t = np.arange(2048) / rate
    x[50_000:52_048] += 0.1 * np.exp(2j * np.pi * 300e3 * t)  # 0.85 ms
    top, mean = frame_rows(x, 1024)
    peak = int(np.argmax(top))
    assert abs((peak - 512) * rate / 1024 - 300e3) < 5e3
    assert top[peak] - mean[peak] > 10.0  # the mean dilutes it, the top not


def test_a_busy_band_cuts_out_a_few_bursts_at_once():
    """Each burst being cut out runs a channel filter over every block (up
    to 100 ms a block at 20 MS/s): six at once take four captures and the
    other two are listed without an analysis, so a busy band can't slow
    the display loop without bound."""
    rate, n = 2.4e6, 80_000
    t = np.arange(n) / rate
    watcher = BurstWatcher(defer=True, max_captures=4)
    reports, most = [], 0
    for k in range(20):
        x = S.noise(n, 1e-6, seed=k)
        if 5 <= k < 12:
            for f in (-900e3, -600e3, -300e3, 300e3, 600e3, 900e3):
                x = x + 0.05 * np.exp(2j * np.pi * f * (t + k * n / rate))
        reports += watcher.feed(x.astype(np.complex64), 433.92e6, rate)
        taking = [c for c in watcher._captures.values() if c.count < c.max]
        most = max(most, len(taking))
    reports += watcher.flush()
    assert len(reports) == 6
    assert most == 4
    assert sum(r.samples is not None for r in reports) == 4


# --------------------------------------------------------------------------- #
# On the demo device
# --------------------------------------------------------------------------- #


def _demo(freq=433.92e6, rate=2.4e6):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(freq)
    dev.start_rx()
    return dev


class TestDemo:
    RATE, CENTRE, FRAME = 2.4e6, 433.92e6, 80_000

    @pytest.fixture(autouse=True)
    def _demo_never_skips(self, monkeypatch):
        """The demo device skips ahead when read slower than real time (a
        busy machine): here every sample is read, so the timing is fixed."""
        from sdr_module.gui import device_dialog

        monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)

    def test_live_packet_display_finds_the_bursts_from_the_start(self):
        """FG-B14: 30 s of the demo at 433.92 MHz. The remote sends every
        2.5 s (12 times), 434.10 MHz's FSK burst every 6 s (5 times);
        nothing else is a burst."""
        from sdr_module.ui.packet_highlighter import LivePacketDisplay

        dev = _demo(self.CENTRE, self.RATE)
        disp = LivePacketDisplay(
            width=1024, height=256, sample_rate=self.RATE, center_freq=self.CENTRE
        )
        found = []
        frame = [0]
        disp.highlighter.set_on_packet_detected(
            lambda p: found.append((frame[0] * self.FRAME / self.RATE, p))
        )
        for i in range(int(30 * self.RATE / self.FRAME)):
            frame[0] = i
            disp.process_samples(dev.read_samples(self.FRAME))
        at = {433.92: [], 434.10: [], "other": []}
        for t, p in found:
            mhz = round(p.center_freq_hz / 1e6, 2)
            at.get(mhz, at["other"]).append(t)
        assert len(at[433.92]) == pytest.approx(12, abs=2)
        assert len(at[434.10]) == pytest.approx(5, abs=1)
        assert len(at["other"]) < 3
        assert found and found[0][0] < 1.5  # no 12 s warm-up
        # Its attributes stay (the FG-B14 repro script reads .spectrum).
        from sdr_module.dsp.spectrum import SpectrumAnalyzer

        assert isinstance(disp.spectrum, SpectrumAnalyzer)

    def test_burst_watcher_takes_the_remote_apart(self):
        dev = _demo(self.CENTRE, self.RATE)
        watcher = BurstWatcher()
        reports = []
        for _ in range(int(6 * self.RATE / self.FRAME)):
            reports += watcher.feed(
                dev.read_samples(self.FRAME), self.CENTRE, self.RATE
            )
        reports += watcher.flush()
        remote = [r for r in reports if abs(r.burst.center_hz - self.CENTRE) < 20e3]
        assert len(remote) >= 2
        for report in remote:
            assert report.kind_text == "OOK PWM"
            assert report.analysis.distinct_rows()[0][0] == CODE_BITS
            assert report.classification.kind == "ook"
        fsk = [r for r in reports if abs(r.burst.center_hz - 434.10e6) < 30e3]
        assert fsk and fsk[0].analysis is not None
        assert fsk[0].kind_text == "FSK PCM"  # both tones cut out together

    def _bursts(self, centre, seconds, frame=None):
        frame = frame or self.FRAME
        dev = _demo(centre, self.RATE)
        dev._rng = np.random.default_rng(11)  # (the same noise every run)
        watcher = BurstWatcher(analyze=False)
        reports = []
        for _ in range(int(seconds * self.RATE / frame)):
            reports += watcher.feed(dev.read_samples(frame), centre, self.RATE)
        return reports + watcher.flush()

    def test_broadcast_fm_stations_are_no_bursts(self):
        """ROB-03: on the first-launch tuning (100.1 MHz) the music
        station's sidebands came and went as 95 'bursts' in 20 s (and the
        99.5 / 101.1 MHz stations' too), flooding the list and its
        analysis. Stations that never stop are not bursts; the narrow FM
        station on 100.715 MHz, on for 5 s every 9 s, still is."""
        reports = self._bursts(100.1e6, 20.0)
        on = [r.burst.center_hz for r in reports]
        for station in (99.5e6, 100.1e6, 101.1e6):
            assert not [f for f in on if abs(f - station) < 150e3], (station, on)
        nfm = [r for r in reports if abs(r.burst.center_hz - 100.715e6) < 10e3]
        assert len(nfm) >= 2
        assert all(r.burst.duration_s > 2.0 for r in nfm)
        # Nothing else but the keyed CW beacon on 99.82 MHz, now and then.
        others = [f for f in on if abs(f - 100.715e6) >= 10e3]
        assert all(abs(f - 99.82e6) < 5e3 for f in others), on

    def test_the_remote_is_still_found_on_every_packet(self):
        """The occupancy rule leaves the 433.92 MHz remote (a packet every
        2.5 s: 8 in 20 s) and the FSK burst on 434.10 MHz alone."""
        reports = self._bursts(self.CENTRE, 20.0)
        remote = [r for r in reports if abs(r.burst.center_hz - self.CENTRE) < 20e3]
        fsk = [r for r in reports if abs(r.burst.center_hz - 434.10e6) < 30e3]
        assert len(remote) == 8
        assert len(fsk) == 3
        assert len(reports) == 11

    def test_deferred_analysis_for_a_worker(self):
        """``defer``: the reports come out with their samples but not taken
        apart (that takes 50-300 ms, too long for a display loop);
        analyze_report() does it later, on a worker."""
        dev = _demo(self.CENTRE, self.RATE)
        watcher = BurstWatcher(defer=True)
        reports = []
        for _ in range(int(3 * self.RATE / self.FRAME)):
            reports += watcher.feed(
                dev.read_samples(self.FRAME), self.CENTRE, self.RATE
            )
        reports += watcher.flush()
        remote = [r for r in reports if abs(r.burst.center_hz - self.CENTRE) < 20e3]
        assert remote
        for report in remote:
            assert report.analysis is None and report.classification is None
            assert report.samples is not None and len(report.samples)
            assert analyze_report(report) is report
            assert report.analysis.distinct_rows()[0][0] == CODE_BITS
            assert report.classification.kind == "ook"

    def test_the_t10_demo_signals_keep_read_samples_fast(self):
        """The OOK remote and the drifting 70 cm handheld are precomputed
        loops: reading stays well under 0.25 us a sample (about 0.04 us on
        an idle machine; computing the modulation would take over 1 us)."""
        for freq in (433.92e6, 438.5e6):
            dev = _demo(freq)
            dev.read_samples(self.FRAME)  # build the loops
            start = time.perf_counter()
            for _ in range(20):
                dev.read_samples(self.FRAME)
            per_sample = (time.perf_counter() - start) / (20 * self.FRAME)
            assert per_sample < 0.25e-6, (freq, per_sample)
