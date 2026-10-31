"""The transmit engine (gui/transmitter.py), the streaming station ID
(ham/callsign.py) and the HackRF driver's transmit checks.

No HackRF is attached: the driver runs on a fake ``python_hackrf`` backend
whose TX callback is pulled by its own thread (tests/hackrf_fakes.py), and
what it "transmits" is demodulated and measured.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.hackrf_fakes import analyse_cw, analyse_mcw, open_hackrf  # noqa: E402

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from sdr_module.core import frequency_manager as fm  # noqa: E402
from sdr_module.core.frequency_manager import LicenseClass  # noqa: E402
from sdr_module.gui import transmitter as tx_mod  # noqa: E402
from sdr_module.gui.transmitter import (  # noqa: E402
    AutoIdPolicy,
    Transmitter,
    TxJob,
    TxPower,
    describe_refusal,
)
from sdr_module.ham.callsign import (  # noqa: E402
    MorseIdSource,
    choose_id_emission,
    cw_occupied_bandwidth,
    mcw_occupied_bandwidth,
    morse_keying,
    morse_units,
)

_APP = QApplication.instance() or QApplication([])

# "DE W1AW" is 67 dit units long (PARIS timing): 4.02 s at 20 WPM.
_W1AW_UNITS = 67
# What was sent is measured at 1/32 of the 2 MS/s it was sent at.
_DEC = 32
_DEC_RATE = 2e6 / _DEC


@pytest.fixture(autouse=True)
def _license():
    """Each test sets the license it needs; put it back afterwards."""
    manager = fm.get_frequency_manager()
    before = manager.get_license_class()
    yield manager
    manager.set_license_class(before)


def _id_job(freq, emission="mcw", text="DE W1AW", wpm=20, rate=2e6, tone=700.0):
    source = MorseIdSource(text, wpm, rate, emission=emission, tone_hz=tone)
    return TxJob(
        source,
        rate,
        freq,
        source.mode,
        source.occupied_bandwidth_hz,
        text,
        is_id=True,
    )


def _run(engine, job, device, timeout=60.0, **kwargs):
    results, events = [], []
    engine.started.connect(lambda j: events.append("started"))
    engine.on_air_changed.connect(lambda on: events.append(("on_air", on)))
    engine.finished.connect(results.append)
    t0 = time.perf_counter()
    started = engine.start(job, device, **kwargs)
    elapsed = time.perf_counter() - t0
    assert started, engine.last_refusal
    assert engine.wait(timeout)
    return results[0], events, elapsed


# --------------------------------------------------------------------------- #
# The station ID itself
# --------------------------------------------------------------------------- #


class TestMorseId:
    def test_units_follow_paris_timing(self):
        units = sum(n for _on, n in morse_units("DE W1AW"))
        assert units == _W1AW_UNITS
        env = morse_keying("DE W1AW", 20, rate=48000.0)
        assert len(env) == pytest.approx(_W1AW_UNITS * 0.06 * 48000, abs=2)
        # Raised-cosine edges inside each element: no instant key clicks.
        assert env[0] < 0.05 and env.max() == pytest.approx(1.0)
        assert np.max(np.abs(np.diff(env))) < 0.02

    def test_source_streams_blocks_without_building_the_whole_id(self):
        source = MorseIdSource("DE VE3ABC/P", 5, 2e6, emission="mcw")
        # 26.6 s of keying plus lead and tail, at 2 MS/s...
        assert source.duration_s == pytest.approx(26.6 + 0.45, abs=0.1)
        # ...is held as a 48 kHz table (~10 MB), not 53 M I/Q samples (400 MB).
        assert source._table.nbytes < 15e6
        block = source.read(65536)
        assert block.dtype == np.complex64 and len(block) == 65536
        assert source.position == 65536
        while len(source.read(1 << 20)):
            pass
        assert source.position == source.total_samples
        assert len(source.read(10)) == 0

    def test_occupied_bandwidths(self):
        assert mcw_occupied_bandwidth(700, 2500) == pytest.approx(6400)
        assert cw_occupied_bandwidth(20) == pytest.approx(150)
        assert cw_occupied_bandwidth(50) == pytest.approx(5 * 50 / 1.2)

    def test_auto_emission_is_cw_on_hf_and_fm_tone_above(self):
        assert choose_id_emission(14.05e6) == "cw"
        assert choose_id_emission(28.4e6) == "cw"
        assert choose_id_emission(50.1e6) == "mcw"
        assert choose_id_emission(146.52e6) == "mcw"
        assert choose_id_emission(146.52e6, "cw") == "cw"
        assert choose_id_emission(7.03e6, "mcw") == "mcw"


# --------------------------------------------------------------------------- #
# The driver checks the emission, not the receive filter (HAM-03, HAM-09)
# --------------------------------------------------------------------------- #


def _source_once(samples):
    state = {"done": False}

    def gen():
        if state["done"]:
            return None
        state["done"] = True
        return samples

    return gen


class TestDriverChecksTheEmission:
    def test_without_the_emission_the_filter_width_is_checked(
        self, monkeypatch, _license
    ):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, _fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(144.39e6)  # 0.39 MHz above the band edge
        # The 1.75 MHz filter reaches below 144 MHz: an older caller that
        # doesn't say how wide its signal is stays refused...
        assert dev.start_tx(_source_once(np.ones(8, np.complex64))) is False
        assert "143.515" in dev.last_tx_error
        # ...while a 6.4 kHz FM ID is allowed.
        assert dev.start_tx(
            _source_once(np.ones(8, np.complex64)), bandwidth_hz=6.4e3, mode="FM"
        )
        dev.stop_tx()
        assert dev.last_tx_error is None

    def test_mode_is_checked(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.AMATEUR_EXTRA)
        dev, _fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(14.05e6)
        ok = dev.start_tx(
            _source_once(np.ones(8, np.complex64)), bandwidth_hz=6.4e3, mode="FM"
        )
        assert ok is False
        assert "Mode 'FM' not allowed" in dev.last_tx_error
        assert dev.start_tx(
            _source_once(np.ones(8, np.complex64)), bandwidth_hz=150, mode="CW"
        )
        dev.stop_tx()

    def test_every_mode_of_a_mixed_transmission_must_be_allowed(
        self, monkeypatch, _license
    ):
        _license.set_license_class(LicenseClass.GENERAL)
        dev, _fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(28.05e6)  # 10 m CW/data segment
        assert not dev.start_tx(
            _source_once(np.ones(8, np.complex64)),
            bandwidth_hz=6.4e3,
            mode=("CW", "FM"),
        )
        assert "FM" in dev.last_tx_error

    def test_retuning_while_transmitting_checks_the_emission(
        self, monkeypatch, _license
    ):
        _license.set_license_class(LicenseClass.GENERAL)
        dev, fake = open_hackrf(monkeypatch, rate=2e6, stall=True)
        dev.set_frequency(146.52e6)
        assert dev.start_tx(lambda: np.ones(64, np.complex64), 6.4e3, "FM")
        try:
            assert dev.set_frequency(146.94e6)  # still 2 m, FM allowed
            assert dev.set_frequency(28.05e6) is False  # FM in a CW segment
            assert dev.state.frequency == 146.94e6
            assert dev.set_amp(True)  # legal here
        finally:
            dev.stop_tx()

    def test_a_lockout_is_explained(self, monkeypatch, _license):
        dev, _fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(1090e6)
        assert dev.start_tx(lambda: None, bandwidth_hz=10e3, mode="FM") is False
        assert "ADS-B" in dev.last_tx_error

    def test_write_samples_waits_for_the_air_time(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2e6, pace=True)
        dev.set_frequency(146.52e6)
        samples = np.full(int(1.2 * 2e6), 0.5, np.complex64)  # 1.2 s on air
        # The timeout is the margin beyond the samples' own air time (a 0.3 s
        # timeout used to cut anything longer short).
        assert dev.write_samples(samples, timeout=0.3, bandwidth_hz=10e3, mode="FM")
        assert fake.tx_count() == len(samples)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


class TestTransmitter:
    @pytest.mark.parametrize("rx_rate", [2.0e6, 2.4e6, 10e6, 20e6])
    def test_id_goes_out_at_its_own_rate(self, monkeypatch, _license, rx_rate):
        """HAM-02: speed, tone, deviation and duration are right whatever
        rate the receiver ran at; the receiver's rate and filter come back."""
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=rx_rate)
        dev.set_frequency(146.52e6)
        rx_filter = dev.state.bandwidth
        engine = Transmitter()
        result, events, _elapsed = _run(engine, _id_job(146.52e6), dev)
        assert result.status == "sent", result.message
        assert fake.start_rate() == 2e6
        iq = fake.tx_iq(_DEC)
        on_air = analyse_mcw(iq, _DEC_RATE)
        assert on_air["text"] == "DE W1AW"
        assert 1.2 * _W1AW_UNITS / on_air["keyed_s"] == pytest.approx(20, abs=0.5)
        assert on_air["keyed_s"] == pytest.approx(4.02, rel=0.02)
        assert on_air["tone_hz"] == pytest.approx(700, abs=5)
        assert on_air["peak_dev_hz"] == pytest.approx(2500, rel=0.05)
        # MCW: the carrier stays on, rising and falling over 5 ms at the
        # ends (never on or off in one sample).
        ramp = int(0.006 * _DEC_RATE)
        assert np.abs(iq[ramp:-ramp]).min() > 0.8
        assert np.abs(iq[0]) < 0.1 and np.abs(iq[-1]) < 0.1
        # Back as the receiver had it.
        assert dev.state.sample_rate == rx_rate
        assert dev.state.bandwidth == rx_filter
        assert not dev.state.is_transmitting
        assert events == ["started", ("on_air", True), ("on_air", False)]

    def test_hf_id_is_real_cw(self, monkeypatch, _license):
        """HAM-09: on HF the ID is an on/off-keyed carrier, checked as CW."""
        _license.set_license_class(LicenseClass.AMATEUR_EXTRA)
        dev, fake = open_hackrf(monkeypatch, rate=2.4e6)
        dev.set_frequency(14.05e6)
        engine = Transmitter()
        result, _events, _t = _run(engine, _id_job(14.05e6, emission="cw"), dev)
        assert result.status == "sent", result.message
        on_air = analyse_cw(fake.tx_iq(_DEC), _DEC_RATE)
        assert on_air["text"] == "DE W1AW"
        assert on_air["carrier_off_level"] < 0.01  # key up: no carrier
        assert on_air["keyed_s"] == pytest.approx(4.02, rel=0.02)

    def test_fm_in_a_cw_segment_is_refused_with_the_reason(self, _license):
        _license.set_license_class(LicenseClass.GENERAL)
        engine = Transmitter()
        problem = tx_mod.legality_problem(28.05e6, 6.4e3, "FM")
        assert problem == ("FM isn't allowed in the 10m CW segment (CW and DATA only)")
        assert tx_mod.legality_problem(28.05e6, 150, "CW") is None
        assert engine.check(_id_job(28.05e6), object()) is not None

    def test_long_slow_id_is_sent_whole(self, monkeypatch, _license):
        """HAM-04: a 26.6 s ID is neither cut at a timeout nor built whole."""
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2.4e6)
        dev.set_frequency(146.52e6)
        job = _id_job(146.52e6, text="DE VE3ABC/P", wpm=5)
        engine = Transmitter()
        result, _events, elapsed = _run(engine, job, dev, timeout=120)
        assert elapsed < 0.2  # start() returns at once
        assert result.status == "sent"
        assert fake.tx_count() == job.total_samples
        assert analyse_mcw(fake.tx_iq(_DEC), _DEC_RATE)["text"] == "DE VE3ABC/P"

    def test_the_window_never_waits_and_stop_ends_it(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=10e6, pace=True)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        results, progress = [], []
        engine.finished.connect(results.append)
        engine.progress.connect(lambda done, total: progress.append((done, total)))
        job = _id_job(146.52e6, text="DE VE3ABC/P", wpm=5)
        t0 = time.perf_counter()
        assert engine.start(job, dev)
        assert time.perf_counter() - t0 < 0.2
        assert engine.busy
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            _APP.processEvents()
            time.sleep(0.01)
        assert progress and 0 < progress[-1][0] < progress[-1][1]
        engine.stop()
        assert engine.wait(5)
        assert results[0].status == "stopped"
        assert results[0].seconds < 2.0
        assert "Stopped DE VE3ABC/P on 146.520 MHz after" in results[0].message
        assert not dev.state.is_transmitting
        assert dev.state.sample_rate == 10e6
        assert not engine.busy

    def test_a_stalled_device_fails_cleanly(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        monkeypatch.setattr(tx_mod, "_STALL_S", 0.3)
        dev, fake = open_hackrf(monkeypatch, rate=2.4e6, stall=True)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        result, _events, _t = _run(engine, _id_job(146.52e6), dev, timeout=5)
        assert result.status == "failed"
        assert "stopped taking samples" in result.message
        assert not dev.state.is_transmitting
        assert dev.state.sample_rate == 2.4e6

    def test_shutdown_gives_up_on_a_hung_radio(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2.4e6)
        dev.set_frequency(146.52e6)
        release = threading.Event()
        real_stop = fake.pyhackrf_stop_tx

        def hung_stop():
            release.wait(5)
            real_stop()

        fake.pyhackrf_stop_tx = hung_stop
        engine = Transmitter()
        results = []
        engine.finished.connect(results.append)
        assert engine.start(_id_job(146.52e6, text="DE VE3ABC/P", wpm=5), dev)
        engine.shutdown(timeout=0.3)
        assert not engine.busy
        assert results[0].status == "failed"
        assert "didn't stop in time" in results[0].message
        release.set()

    def test_power_and_amp_are_set_and_the_amp_put_back(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2.4e6)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        amp_during = []
        engine.started.connect(lambda _job: amp_during.append(fake.amp))
        result, _e, _t = _run(
            engine, _id_job(146.52e6), dev, power=TxPower(gain_db=35, amp=True)
        )
        assert result.ok
        assert fake.txvga == 35 and amp_during == [True]
        assert fake.amp is False and dev.amp_enabled is False

    def test_power_estimate(self):
        power = TxPower(gain_db=47, amp=True)
        assert power.output_dbm(10e6) == pytest.approx(15.0)
        assert power.output_dbm(2150e6) == pytest.approx(5.0, abs=0.1)
        low = TxPower(gain_db=20, amp=False)
        assert low.output_dbm(146.52e6) == pytest.approx(
            power.output_dbm(146.52e6) - 27 - 14
        )
        assert "TX gain 20 dB, amp off: about" in low.describe(146.52e6)
        assert TxPower(gain_db=99).clamped().gain_db == 47


class TestRefusals:
    def test_devices_that_cant_transmit(self):
        from sdr_module.devices.rtlsdr import RTLSDRDevice
        from sdr_module.gui.device_dialog import MockDevice

        engine = Transmitter()
        job = _id_job(146.52e6)
        assert "no device is connected" in engine.check(job, None)
        assert "can only receive" in engine.check(job, MockDevice())
        assert "can only receive" in engine.check(job, RTLSDRDevice())

    def test_rate_the_hackrf_cant_send(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, _fake = open_hackrf(monkeypatch)
        job = _id_job(146.52e6, rate=48000.0)
        assert "2-20 MS/s" in Transmitter().check(job, dev)

    def test_license_reasons_read_plainly(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.NONE)
        dev, _fake = open_hackrf(monkeypatch)
        engine = Transmitter()
        assert engine.check(_id_job(100.1e6), dev) == (
            "your license (None) allows CB/MURS/FRS only"
        )
        _license.set_license_class(LicenseClass.TECHNICIAN)
        # (No part of 20 m is the Technician's: the band is named.)
        assert engine.check(_id_job(14.2e6, emission="cw"), dev) == (
            "your license (Technician) doesn't include 20 m (14.200 MHz)"
        )
        assert engine.check(_id_job(1575.42e6), dev).startswith(
            "1575.420 MHz is a protected frequency (GPS L1"
        )
        assert engine.check(_id_job(147.999e6), dev) == (
            "the signal's edge at 148.002 MHz is outside the amateur and "
            "license-free bands"
        )
        # (144.0-144.1 MHz is CW only: no FM tone ID there.)
        assert engine.check(_id_job(144.001e6), dev) == (
            "FM isn't allowed in the 2m CW segment (CW only)"
        )

    def test_describe_refusal_falls_back_to_the_text(self):
        assert describe_refusal("Something odd", 1e6) == "something odd"
        assert describe_refusal(None, 1e6) == "transmitting isn't allowed here"


class TestAutoId:
    def _carrier_job(self, seconds, rate=2e6, freq=146.52e6):
        samples = np.full(int(seconds * rate), 0.9, np.complex64)
        return TxJob(samples, rate, freq, "FM", 12.5e3, "a test carrier")

    def _make_id(self, job):
        return _id_job(job.center_hz, rate=job.sample_rate)

    def test_ids_before_and_after(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        policy = AutoIdPolicy(at_start=True, at_end=True, interval_s=None)
        job = self._carrier_job(1.0)
        result, _e, _t = _run(engine, job, dev, policy=policy, make_id=self._make_id)
        assert result.ok and result.ids_sent == 2
        id_len = _id_job(146.52e6).total_samples
        assert fake.tx_count() == 2 * id_len + job.total_samples
        iq = fake.tx_iq(_DEC)
        n_id, n_job = id_len // _DEC, job.total_samples // _DEC
        first = analyse_mcw(iq[:n_id], _DEC_RATE)
        last = analyse_mcw(iq[-n_id:], _DEC_RATE)
        assert first["text"] == last["text"] == "DE W1AW"
        middle = iq[n_id + 2 : n_id + n_job - 2]
        assert np.allclose(middle, round(0.9 * 127) / 127, atol=0.01)

    def test_interval_id_during_a_long_transmission(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        policy = AutoIdPolicy(at_start=False, at_end=False, interval_s=2.0)
        job = self._carrier_job(2.5)
        result, _e, _t = _run(engine, job, dev, policy=policy, make_id=self._make_id)
        assert result.ok and result.ids_sent == 1
        id_len = _id_job(146.52e6).total_samples
        gap = int(round(7 * 1.2 / 20 * 2e6))  # a word space at the ID's speed
        total = fake.tx_count()
        assert total == id_len + job.total_samples + 2 * gap
        # The ID began when it was due (2 s on the air, not at the next
        # block), a word space of key-up after the carrier faded out.
        start, end = int(2.0 * 2e6) // _DEC, (int(2.0 * 2e6) + id_len) // _DEC
        iq = fake.tx_iq(_DEC)
        assert analyse_mcw(iq[start:end], _DEC_RATE)["text"] == "DE W1AW"
        assert not np.any(iq[start - gap // _DEC : start])  # key up
        assert not np.any(iq[end + 1 : end + gap // _DEC])
        steps = np.abs(np.diff(np.abs(iq)))
        assert steps.max() < 0.02  # (no one-sample carrier steps)
        # The next transmission's interval runs from that ID.
        assert engine.seconds_since_last_id() < 5

    def test_an_id_job_gets_no_extra_ids(self, monkeypatch, _license):
        _license.set_license_class(LicenseClass.TECHNICIAN)
        dev, fake = open_hackrf(monkeypatch, rate=2e6)
        dev.set_frequency(146.52e6)
        engine = Transmitter()
        policy = AutoIdPolicy(at_start=True, at_end=True, interval_s=1.0)
        job = _id_job(146.52e6)
        result, _e, _t = _run(engine, job, dev, policy=policy, make_id=self._make_id)
        assert result.ids_sent == 1
        assert fake.tx_count() == job.total_samples

    def test_countdown(self):
        import threading

        job = self._carrier_job(30.0, rate=2e6)
        plan = tx_mod._Plan(
            job,
            AutoIdPolicy(False, False, 10.0),
            self._make_id,
            None,
            threading.Event(),
        )
        assert plan.next_id_in_s() == pytest.approx(10.0)
        for _ in range(31):  # about 1 s
            plan.read()
        assert plan.next_id_in_s() == pytest.approx(10.0 - 31 * 65536 / 2e6)
        plain = tx_mod._Plan(job, None, None, None, threading.Event())
        assert plain.next_id_in_s() is None
