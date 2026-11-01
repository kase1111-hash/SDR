"""Automatic frequency control (T10): the closed loop in dsp/afc.py and
SignalTracker, which Radio > Track Signal (AFC) uses.

Regressions for FG-G11 / UX-G15: the loop did not converge on voice NFM
(it moved the correction by the raw error every update, so a signal 1.2 kHz
off was over-corrected back and forth instead of settled), and nothing kept
a drifting signal centred in the window.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdr_module.dsp.afc import (
    TRACK_METHODS,
    AFCConfig,
    AFCMethod,
    AFCMode,
    AutomaticFrequencyControl,
    SignalTracker,
)

RATE = 48_000.0
BLOCK = 2_400  # 50 ms


def _stream(offset_of_t, seconds, rate=RATE, block=BLOCK, fm_dev=0.0, snr_db=30.0):
    """Blocks of a carrier (or voice-like FM) whose offset from 0 Hz follows
    ``offset_of_t(t)``, with noise ``snr_db`` below it."""
    rng = np.random.default_rng(3)
    phase = 0.0
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    for k in range(int(seconds * rate / block)):
        t = (k * block + np.arange(block)) / rate
        inst = offset_of_t(t)
        if fm_dev:
            inst = inst + fm_dev * (
                0.6 * np.sin(2 * np.pi * 820 * t)
                + 0.4 * np.sin(2 * np.pi * 1650 * t + 1)
            )
        ph = phase + 2 * np.pi * np.cumsum(inst) / rate
        phase = float(ph[-1])
        noise = sigma * (rng.standard_normal(block) + 1j * rng.standard_normal(block))
        yield t[-1], (np.exp(1j * ph) + noise).astype(np.complex64)


def _closed_loop(afc, offset_of_t, seconds, **kw):
    """Run the loop the way a receiver does: correct, then measure what is
    left. Returns ``[(t, offset, correction, locked)]``."""
    out = []
    for t, block in _stream(offset_of_t, seconds, **kw):
        afc.update(afc.apply_correction(block), time_delta_ms=1000.0 * BLOCK / RATE)
        out.append(
            (t, float(offset_of_t(np.array([t]))[0]), afc.correction, afc.is_locked)
        )
    return out


class TestClosedLoop:
    @pytest.mark.parametrize("method", [AFCMethod.FFT_PEAK, AFCMethod.PHASE_DIFF])
    def test_settles_on_a_carrier_1_2_khz_off(self, method):
        afc = AutomaticFrequencyControl(RATE, AFCConfig(method=method, fft_size=4096))
        run = _closed_loop(afc, lambda t: np.full_like(t, 1200.0), 6.0)
        tail = run[-20:]
        assert all(abs(c - 1200.0) < 15.0 for _t, _o, c, _l in tail), tail[-1]
        assert tail[-1][3]  # locked

    def test_settles_on_voice_nfm(self):
        """FG-G11: the loop never settled on voice NFM: with the FFT peak
        (not the carrier, for FM) it stayed 750 Hz off, with the mean
        frequency it swung +/- 25 Hz around it every update. FM is tracked
        with the mean frequency (TRACK_METHODS), and now settles."""
        assert TRACK_METHODS["FM"] == AFCMethod.PHASE_DIFF
        afc = AutomaticFrequencyControl(RATE, AFCConfig(method=AFCMethod.PHASE_DIFF))
        run = _closed_loop(afc, lambda t: np.full_like(t, -1500.0), 8.0, fm_dev=2500.0)
        corrections = np.array([c for _t, _o, c, _l in run[-40:]])
        assert np.all(np.abs(corrections + 1500.0) < 15.0), corrections[-5:]

    def test_follows_a_steady_drift_without_lag(self):
        """The integrator learns the drift: the error stays small."""
        afc = AutomaticFrequencyControl(
            RATE, AFCConfig(method=AFCMethod.PHASE_DIFF, max_rate_hz_per_sec=5e3)
        )
        run = _closed_loop(afc, lambda t: 200.0 + 100.0 * t, 15.0)
        errors = [abs(o - c) for _t, o, c, _l in run[-60:]]
        assert max(errors) < 10.0

    def test_noise_moves_nothing(self):
        afc = AutomaticFrequencyControl(RATE, AFCConfig(method=AFCMethod.PHASE_DIFF))
        rng = np.random.default_rng(5)
        for _ in range(100):
            noise = (rng.standard_normal(BLOCK) + 1j * rng.standard_normal(BLOCK)) * 0.1
            status = afc.update(noise.astype(np.complex64), time_delta_ms=50.0)
            assert not status.signal_present
        assert afc.correction == 0.0

    def test_hold_and_off_keep_the_correction(self):
        afc = AutomaticFrequencyControl(RATE, AFCConfig(method=AFCMethod.PHASE_DIFF))
        _closed_loop(afc, lambda t: np.full_like(t, 800.0), 4.0)
        held = afc.correction
        for mode in (AFCMode.HOLD, AFCMode.OFF):
            afc.config = AFCConfig(method=AFCMethod.PHASE_DIFF, mode=mode)
            _closed_loop(afc, lambda t: np.full_like(t, 2000.0), 1.0)
            assert afc.correction == held


# --------------------------------------------------------------------------- #
# SignalTracker: the VFO follows the signal (Radio > Track Signal)
# --------------------------------------------------------------------------- #

# A tenth of a 2.4 MS/s receiver's rate (and its 33 ms blocks): the same
# loop, ten times fewer samples to make.
WIDE = 240e3
FRAME = 8_000


def _track(tracker, signal_of_t, seconds, fm_dev=0.0, snr_db=25.0, vfo=100e3):
    """Feed ``tracker`` blocks at the device's rate with a signal at the
    offset ``signal_of_t(t)`` from the LO; move the VFO by every nudge.
    Returns ``[(t, signal - vfo, status)]``."""
    rng = np.random.default_rng(1)
    phase = 0.0
    # The noise per sample for snr_db in a 12.5 kHz channel.
    sigma = np.sqrt(10 ** (-snr_db / 10) * (WIDE / 12.5e3) / 2)
    out = []
    for k in range(int(seconds * WIDE / FRAME)):
        t = (k * FRAME + np.arange(FRAME)) / WIDE
        f_sig = signal_of_t(t)
        inst = f_sig
        if fm_dev:
            inst = inst + fm_dev * (
                0.6 * np.sin(2 * np.pi * 820 * t)
                + 0.4 * np.sin(2 * np.pi * 1650 * t + 1)
            )
        ph = phase + 2 * np.pi * np.cumsum(inst) / WIDE
        phase = float(ph[-1])
        x = np.exp(1j * ph) + sigma * (
            rng.standard_normal(FRAME) + 1j * rng.standard_normal(FRAME)
        )
        status = tracker.process(x.astype(np.complex64), FRAME / WIDE)
        vfo += status.nudge_hz
        out.append((float(t[-1]), float(f_sig[-1]) - vfo, status))
    return out


class TestSignalTracker:
    def test_a_carrier_1_2_khz_off_is_centred(self):
        """UX-G15: tuned 1.2 kHz off a carrier, Track Signal centres it and
        says how far it moved ("AFC +1.2 kHz")."""
        tracker = SignalTracker(WIDE, 100e3, "AM", 10e3)
        run = _track(tracker, lambda t: np.full_like(t, 101.2e3), 6.0)
        assert all(abs(err) <= 10.0 for _t, err, _s in run[-30:])
        assert tracker.correction_hz == pytest.approx(1200.0, abs=10.0)
        assert run[-1][2].locked and not run[-1][2].lost

    def test_a_drifting_carrier_is_followed(self):
        tracker = SignalTracker(WIDE, 100e3, "AM", 10e3)
        run = _track(tracker, lambda t: 100e3 + 150.0 * t, 20.0)
        assert max(abs(err) for _t, err, _s in run[-60:]) <= 15.0
        assert tracker.correction_hz == pytest.approx(3000.0, abs=50.0)

    def test_a_drifting_narrow_fm_signal_is_followed(self):
        """FG-G11: the old loop never settled on voice NFM; drifting 3 kHz in
        20 s it stays within a resolution step or so."""
        tracker = SignalTracker(WIDE, 100e3, "FM", 12.5e3)
        run = _track(tracker, lambda t: 100e3 + 150.0 * t, 20.0, fm_dev=2500.0)
        assert max(abs(err) for _t, err, _s in run[-60:]) <= 2 * tracker.resolution_hz
        assert all(s.locked for _t, _e, s in run[-60:])

    def test_cw_moves_in_small_steps(self):
        tracker = SignalTracker(WIDE, 100e3, "CW", 500.0)
        run = _track(tracker, lambda t: 100.2e3 + 20.0 * t, 10.0, snr_db=20.0)
        assert tracker.resolution_hz == 5.0
        assert max(abs(err) for _t, err, _s in run[-60:]) <= 10.0
        nudges = [s.nudge_hz for _t, _e, s in run if s.nudge_hz]
        assert all(abs(n) % 5.0 == 0 for n in nudges)

    def test_noise_is_lost_and_never_moves(self):
        tracker = SignalTracker(WIDE, 100e3, "FM", 12.5e3)
        rng = np.random.default_rng(2)
        status = None
        for _ in range(int(4.0 * WIDE / FRAME)):
            x = 0.01 * (rng.standard_normal(FRAME) + 1j * rng.standard_normal(FRAME))
            status = tracker.process(x.astype(np.complex64), FRAME / WIDE)
            assert status.nudge_hz == 0.0
        assert status.lost and not status.signal_present
        assert tracker.correction_hz == 0.0

    def test_retune_starts_over_and_an_lo_move_carries_on(self):
        tracker = SignalTracker(WIDE, 100e3, "AM", 10e3)
        _track(tracker, lambda t: np.full_like(t, 101e3), 4.0)
        assert tracker.correction_hz == pytest.approx(1000.0, abs=10.0)
        # The LO moved 500 kHz up, the VFO stayed on the signal.
        tracker.lo_moved(100e3 + 1000.0 - 500e3)
        assert tracker.correction_hz == pytest.approx(1000.0, abs=10.0)
        tracker.retuned(250e3)
        assert tracker.correction_hz == 0.0

    def test_only_modes_with_something_to_follow(self):
        assert set(TRACK_METHODS) == {"FM", "AM", "CW"}
        for mode in ("USB", "LSB", "None (I/Q)"):
            with pytest.raises(ValueError):
                SignalTracker(WIDE, 0.0, mode, 2.7e3)


def test_the_demo_drifting_handheld_is_followed(monkeypatch):
    """The demo's 70 cm handheld on 438.5 MHz drifts +/- 2.5 kHz over 40 s
    (2.5 kHz * sin(2 pi t / 40)): tracking from the start, the VFO follows
    it to within a resolution step or two."""
    from sdr_module.gui import device_dialog
    from sdr_module.gui.device_dialog import MockDevice

    monkeypatch.setattr(device_dialog, "_MAX_LAG_S", 1e9)  # every sample
    rate, frame = 2.4e6, 80_000
    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(438.5e6)
    dev.start_rx()
    tracker = SignalTracker(rate, 0.0, "FM", 12.5e3)
    vfo = 0.0
    errors = []
    for k in range(int(12.0 * rate / frame)):
        status = tracker.process(dev.read_samples(frame), frame / rate)
        vfo += status.nudge_hz
        t = (k + 1) * frame / rate
        drift = device_dialog._DRIFT_HZ * np.sin(
            2 * np.pi * t / device_dialog._DRIFT_PERIOD_S
        )
        if t > 3.0:
            errors.append(drift - vfo)
    assert max(abs(e) for e in errors) < 4 * tracker.resolution_hz, errors[-3:]
    assert tracker.correction_hz > 2000.0  # it moved about 2.4 kHz up
