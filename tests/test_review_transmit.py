"""Fixes from the code review of the transmit side (TX-1 to TX-12): where
automatic station IDs go and how they sound, the ID's emission and speed,
the HackRF driver's start and end of a transmission, and the transmit
window's settings and confirmation.

The plan of a transmission is read the way the HackRF driver reads it (a
block at a time); what goes on the air is decoded with the independent
demodulators in tests/text_demod.py and tests/hackrf_fakes.py.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests import text_demod as demod  # noqa: E402
from tests.hackrf_fakes import _MORSE, analyse_cw  # noqa: E402

pytest.importorskip("PyQt6")

from sdr_module.gui.transmitter import AutoIdPolicy, TxJob, _Plan  # noqa: E402
from sdr_module.ham.callsign import MorseIdSource  # noqa: E402
from sdr_module.protocols.text_modes import TextSettings, TextSignal  # noqa: E402

# Plan-level checks run at a low rate (the plan works at any rate; the
# HackRF would be at 2 MS/s): a minute of air time stays a few MB.
RATE = 48000.0


def _cw_id_factory(wpm=20, emission="cw"):
    """What the window's automatic ID makes: "DE W1AW" at the job's
    center and rate."""

    def make(job):
        source = MorseIdSource("DE W1AW", wpm, job.sample_rate, emission=emission)
        return TxJob(
            source,
            job.sample_rate,
            job.center_hz,
            source.mode,
            source.occupied_bandwidth_hz,
            "DE W1AW",
            is_id=True,
            identify=False,
        )

    return make


def _text_job(text, mode, freq=14.08e6, emission="direct", rate=RATE, **settings):
    signal = TextSignal(text, TextSettings.for_mode(mode, **settings))
    source = signal.source(rate, emission, amplitude=0.9)
    job = TxJob(source, rate, freq, source.mode, source.occupied_bandwidth_hz, mode)
    return job, signal


def _drain(plan, job):
    """Read ``plan`` to the end. Returns (everything sent, the sample where
    each ID began, the job's own runs: its samples between the IDs)."""
    blocks, id_starts, runs = [], [], []
    run, handed, last_id = [], 0, None
    while True:
        before = job.source.position
        block = plan.read()
        if block is None:
            break
        block = np.asarray(block)
        if plan.last_id_began_at is not None and plan.last_id_began_at != last_id:
            last_id = plan.last_id_began_at
            id_starts.append(last_id)
        if job.source.position != before:  # (the job's own samples)
            run.append(block)
        elif run:
            runs.append(np.concatenate(run))
            run = []
        blocks.append(block)
        handed += len(block)
    if run:
        runs.append(np.concatenate(run))
    return np.concatenate(blocks), id_starts, runs


def _decode_cw(iq, rate, wpm):
    """Morse text of a keyed carrier at a known speed (PARIS timing): a
    mark under 2 dits is a dot; a gap of 5 dits or more ends a word, of 2
    or more a letter."""
    dit = 1.2 / wpm
    win = max(1, int(rate * 0.001))
    count = len(iq) // win
    env = np.abs(iq[: count * win]).reshape(count, win).mean(axis=1)
    on = env > 0.45 * float(env.max() or 1.0)
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(on)]))
    text, symbol = "", ""
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        seconds = (b - a) * win / rate
        if on[a]:
            symbol += "." if seconds < 2 * dit else "-"
        elif seconds >= 5 * dit:
            text += (_MORSE.get(symbol, "~") if symbol else "") + " "
            symbol = ""
        elif seconds >= 2 * dit:
            text += _MORSE.get(symbol, "~") if symbol else ""
            symbol = ""
    if symbol:
        text += _MORSE.get(symbol, "~")
    return " ".join(text.split())


def _plan(job, policy, make_id=None, since=None):
    return _Plan(job, policy, make_id or _cw_id_factory(), since, threading.Event())


# --------------------------------------------------------------------------- #
# TX-1: interval IDs at the text's word breaks, shaped, never late
# --------------------------------------------------------------------------- #


class TestIntervalIdPlacement:
    @pytest.mark.parametrize("prefix", ["", "E ", "I ", "S ", "5 ", "EE ", "T "])
    def test_a_cw_text_is_split_between_words_only(self, prefix):
        """Wherever the interval falls, the ID goes between two words of
        the text, a word of its own ("TEST DE W1AW TEST"), never into an
        element or a letter ("TESZE W1AP")."""
        words = ["TEST"] * 24
        job, _signal = _text_job(prefix + " ".join(words), "CW", wpm=20)
        # (Short to keep the test quick, but more than twice the ID's air
        # time, as a real one is: at least half an interval of the job
        # always goes between two interval IDs.)
        interval = 12.0
        plan = _plan(job, AutoIdPolicy(True, True, interval))
        iq, id_starts, _runs = _drain(plan, job)
        decoded = _decode_cw(iq, RATE, 20).split()
        expected_words = prefix.split() + words
        text, ids = [], 0
        i = 0
        while i < len(decoded):
            if decoded[i : i + 2] == ["DE", "W1AW"]:
                ids += 1
                i += 2
                continue
            text.append(decoded[i])
            i += 1
        assert text == expected_words, " ".join(decoded)
        assert ids == plan.ids_sent >= 4  # start, 2+ interval, end
        # Never late: an ID begins within the interval of the one before.
        gaps = np.diff(id_starts) / RATE
        assert gaps.max() <= interval + 1e-6, gaps

    def test_no_one_sample_carrier_steps_at_a_splice(self):
        """A job that reports no breaks (a plain array) fades out over a few
        ms before the ID and back in after it: no key clicks."""
        carrier = np.full(int(30 * RATE), 0.9, np.complex64)
        job = TxJob(carrier, RATE, 146.52e6, "FM", 12.5e3, "a test carrier")
        plan = _plan(job, AutoIdPolicy(False, False, 10.0), _cw_id_factory())
        iq, id_starts, runs = _drain(plan, job)
        assert plan.ids_sent >= 2
        assert sum(len(r) for r in runs) == len(carrier)
        steps = np.abs(np.diff(np.abs(iq)))
        assert steps.max() < 0.02, steps.max()  # (0.9 in one sample before)
        # The ID begins by the time it is due (not at the next block).
        assert id_starts[0] / RATE <= 10.0 + 1e-6

    def test_rtty_resumes_in_step_after_the_id(self):
        """RTTY is split after a space, the carrier fades out and in, and
        the text starts again the way it began (idle mark, LTRS): each part
        decodes on its own and together they are the whole text."""
        text = "CQ CQ CQ DE W1AW W1AW PSE K " * 3
        job, signal = _text_job(text.strip(), "RTTY")
        plan = _plan(job, AutoIdPolicy(False, True, 6.0))
        _iq, _ids, runs = _drain(plan, job)
        assert len(runs) >= 2  # (split at least once)
        parts = [demod.rtty_decode(run, RATE) for run in runs]
        assert "".join(parts) == signal.check.sent, parts
        assert all(p.endswith(" ") for p in parts[:-1]), parts

    def test_psk31_resumes_in_step_after_the_id(self):
        text = "Hello de W1AW, the quick brown fox jumps over the lazy dog 73"
        job, signal = _text_job(text, "PSK31", freq=14.0703e6)
        plan = _plan(job, AutoIdPolicy(False, True, 8.0))
        _iq, _ids, runs = _drain(plan, job)
        assert len(runs) >= 2
        table = {code: char for char, code in _varicode().items()}
        parts = [demod.psk31_decode(run, RATE, table) for run in runs]
        assert "".join(parts) == signal.check.sent, parts

    def test_ascii_and_fm_audio_resume_in_step(self):
        """The same for ASCII on the carrier, and for PSK31 sent as audio on
        an FM carrier (the carrier fades, comes back, and the audio goes on
        in step), with FM tone IDs."""
        text = "the quick brown fox jumps over the lazy dog 0123456789 " * 2
        job, signal = _text_job(text.strip(), "ASCII")
        _iq, _ids, runs = _drain(_plan(job, AutoIdPolicy(False, True, 3.0)), job)
        assert len(runs) >= 2
        parts = [demod.ascii_decode(run, RATE).decode() for run in runs]
        assert "".join(parts) == signal.check.sent, parts

        text = "Hello de W1AW, the quick brown fox jumps over the lazy dog 73"
        job, signal = _text_job(text, "PSK31", freq=146.52e6, emission="fm")
        mcw = _cw_id_factory(emission="mcw")
        iq, _ids, runs = _drain(_plan(job, AutoIdPolicy(False, True, 8.0), mcw), job)
        assert len(runs) >= 2
        assert np.max(np.abs(np.diff(np.abs(iq)))) < 0.05
        table = {code: char for char, code in _varicode().items()}
        audio_hz = signal.settings.audio_hz
        parts = [
            demod.psk31_decode(
                demod.analytic(demod.instantaneous_frequency(run, RATE)),
                RATE,
                table,
                center_hz=audio_hz,
            )
            for run in runs
        ]
        assert "".join(parts) == signal.check.sent, parts

    def test_a_carried_over_id_goes_before_the_text(self):
        """The last ID went out 9 min 58 s ago and "At start" is off: the ID
        due 2 s into a 7 s text goes before it instead of into it."""
        text = "CQ CQ CQ DE W1AW W1AW PSE K"
        job, signal = _text_job(text, "RTTY")
        policy = AutoIdPolicy(at_start=False, at_end=True, interval_s=600.0)
        plan = _plan(job, policy, since=598.0)
        _iq, id_starts, runs = _drain(plan, job)
        assert id_starts[0] == 0  # the ID first
        assert plan.ids_sent == 2  # and the one at the end
        assert len(runs) == 1  # the text whole
        assert demod.rtty_decode(runs[0], RATE) == signal.check.sent


def _varicode():
    from sdr_module.protocols.text_modes import VARICODE

    return VARICODE


# --------------------------------------------------------------------------- #
# TX-5: FM carriers rise and fall, no key clicks where segments meet
# --------------------------------------------------------------------------- #


def _read_all(source, block=65536):
    parts = []
    while True:
        part = source.read(block)
        if not len(part):
            return np.concatenate(parts)
        parts.append(part)


class TestCarrierEdges:
    def test_fm_tone_id_and_fm_text_rise_and_fall(self):
        mcw = _read_all(MorseIdSource("DE W1AW", 20, RATE, emission="mcw"))
        _job, signal = _text_job("TEST", "CW")
        fm = _read_all(signal.source(RATE, "fm", amplitude=0.9))
        ramp = int(0.005 * RATE)
        for iq in (mcw, fm):
            env = np.abs(iq)
            assert env[0] < 0.01 and env[-1] < 0.01  # (0.9 before)
            assert np.allclose(env[ramp:-ramp], 0.9, atol=1e-3)
            assert np.max(np.abs(np.diff(env))) < 4 * 0.9 / ramp

    def test_a_keyed_text_meets_its_fm_tone_ids_without_a_step(self):
        """RTTY on the carrier with FM tone IDs (Sent as: FM tone): where
        the ID meets the text the carrier changes over a few ms, not in one
        sample (about -30 dB of key clicks beyond 12.5 kHz before)."""
        job, _signal = _text_job("RYRY CQ DE W1AW", "RTTY", freq=144.15e6)
        plan = _plan(
            job, AutoIdPolicy(True, True, None), _cw_id_factory(emission="mcw")
        )
        iq, _ids, _runs = _drain(plan, job)
        assert plan.ids_sent == 2
        assert np.max(np.abs(np.diff(np.abs(iq)))) < 0.05


# --------------------------------------------------------------------------- #
# TX-11: interval IDs aren't kept once sent
# --------------------------------------------------------------------------- #


class TestIdMemory:
    def test_interval_ids_are_let_go(self):
        carrier = np.full(int(40 * RATE), 0.9, np.complex64)
        job = TxJob(carrier, RATE, 146.52e6, "FM", 12.5e3, "a test carrier")
        plan = _plan(job, AutoIdPolicy(True, True, 10.0), _cw_id_factory(wpm=5))
        _drain(plan, job)
        assert plan.ids_sent >= 4
        assert len(plan.id_jobs) == 2  # the start and end IDs only

    def test_the_id_table_is_float32(self):
        source = MorseIdSource("DE VE3ABC/P", 5, 2e6, emission="mcw")
        assert source._table.dtype == np.float32
        assert source._table.nbytes < 6e6  # (10 MB as float64)


# --------------------------------------------------------------------------- #
# The window (a real SDRMainWindow, the HackRF on a fake backend)
# --------------------------------------------------------------------------- #

from tests.test_gui_station_tx import _TxCase  # noqa: E402


class TestConfirmationCountsTheIds(_TxCase):
    def _confirmation(self, job):
        """The first-use question's air time for ``job`` (answered Cancel)."""
        self.store.values["tx/confirm"] = True
        seen = []

        def answer(box):
            seen.append(box.informativeText())
            box.clickedButton = lambda: box.escapeButton()  # Cancel
            return 0

        self.win._exec_dialog = answer
        self.assertFalse(self.win._transmit_job(job))
        return seen[-1]

    def test_a_carried_over_id_is_counted(self):
        """TX-1: "At start" off, the last ID 9 min 58 s ago: the ID that
        goes before the job is in the air time the question gives."""
        import time

        self.panel.set_callsign("W1AW")
        self.panel._id_start_check.setChecked(False)
        self.hackrf()
        self.station.engine._last_id_at = time.monotonic() - 598.0
        carrier = np.full(int(7 * 2e6), 0.9, np.complex64)  # 7 s
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        id_s = self.station.make_id_job(146.52e6).duration_s
        text = self._confirmation(job)
        self.assertIn(f"about {7.0 + 2 * id_s:.1f} s with your station ID", text)
        # Without the carried-over interval, only the ID at the end.
        self.station.engine._last_id_at = time.monotonic() - 60.0
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        text = self._confirmation(job)
        self.assertIn(f"about {7.0 + id_s:.1f} s with your station ID", text)

    def test_transmit_text_says_the_id_goes_first(self):
        import time

        self.panel.set_callsign("W1AW")
        self.panel._id_start_check.setChecked(False)
        self.hackrf()
        self.win._show_transmit_text()
        dlg = self.win._transmit_text
        self.addCleanup(dlg.close)
        dlg.set_mode("CW")
        dlg.set_text("CQ CQ DE W1AW")
        self.station.engine._last_id_at = time.monotonic() - 598.0
        dlg.refresh()
        self.assertTrue(
            dlg._tx_label.text().endswith(
                "· ID DE W1AW (FM tone, 20 WPM) at the start (9 min since the "
                "last), every 10 min and at the end"
            ),
            dlg._tx_label.text(),
        )
        self.station.engine._last_id_at = time.monotonic() - 60.0
        dlg.refresh()
        self.assertTrue(
            dlg._tx_label.text().endswith(
                "· ID DE W1AW (FM tone, 20 WPM) every 10 min and at the end"
            ),
            dlg._tx_label.text(),
        )


# --------------------------------------------------------------------------- #
# TX-4: an automatic ID is sent the way the transmission it identifies is
# --------------------------------------------------------------------------- #


class TestCwOnlySegments:
    def test_the_vhf_cw_only_segments_refuse_fm(self):
        """47 CFR 97.305(c): 50.0-50.1 and 144.0-144.1 MHz are CW only."""
        from sdr_module.core import frequency_manager as fm
        from sdr_module.core.frequency_manager import LicenseClass

        manager = fm.get_frequency_manager()
        before = manager.get_license_class()
        manager.set_license_class(LicenseClass.TECHNICIAN)
        try:
            for freq in (50.05e6, 144.05e6):
                assert fm.is_tx_allowed(freq, 150.0, "CW")[0]
                for mode in ("FM", "DATA", "SSB"):
                    allowed, reason = fm.is_tx_allowed(freq, 6.4e3, mode)
                    assert not allowed and "CW" in reason, (freq, mode)
            for freq in (50.2e6, 144.2e6, 146.52e6):
                assert fm.is_tx_allowed(freq, 6.4e3, "FM")[0]
            # (A signal reaching down into it is refused too.)
            assert not fm.is_tx_allowed(144.102e6, 6.4e3, "FM")[0]
        finally:
            manager.set_license_class(before)


class TestIdEmissionFollowsTheJob(_TxCase):
    license = "general"

    def _job(self, mode, freq, width):
        carrier = np.full(int(0.5 * 2e6), 0.9, np.complex64)
        return TxJob(carrier, 2e6, freq, mode, width, f"a {mode} test")

    def _sent_ids(self, backend, job):
        """The start ID of what went on the air, as CW (keyed carrier)."""
        id_job = self.station.engine.last_result
        self.assertTrue(id_job.ok, id_job.message)
        n = self.station._id_factory()(job).total_samples // 32
        return analyse_cw(backend.tx_iq(32)[:n], 2e6 / 32)

    def test_data_on_the_carrier_above_30_mhz_gets_a_cw_id(self):
        """PSK31 on the carrier at 50.290 MHz: its listeners are in USB/CW,
        so its IDs are CW (a keyed carrier), not a 6 kHz FM tone."""
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=50.29e6)
        job = self._job("DATA", 50.29e6, 60.0)
        self.assertTrue(self.win._transmit_job(job), self.status()[0])
        self.assertEqual(self.station.engine._run.plan.modes, ("DATA", "CW"))
        self.wait_done(30)
        on_air = self._sent_ids(backend, job)
        self.assertEqual(on_air["text"], "DE W1AW")
        self.assertLess(on_air["carrier_off_level"], 0.01)  # key up: no carrier

    def test_cw_in_the_2m_cw_segment_gets_a_cw_id(self):
        """A CW text keyed on the carrier at 144.050 MHz (CW only there):
        its IDs are CW too; an FM tone ID would be an emission not allowed
        there."""
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=144.05e6)
        job = self._job("CW", 144.05e6, 150.0)
        self.assertTrue(self.win._transmit_job(job), self.status()[0])
        self.assertEqual(self.station.engine._run.plan.modes, ("CW",))
        self.wait_done(30)
        self.assertEqual(self._sent_ids(backend, job)["text"], "DE W1AW")

    def test_an_fm_job_keeps_its_fm_tone_id(self):
        self.panel.set_callsign("W1AW")
        self.hackrf(freq=146.52e6)
        job = self._job("FM", 146.52e6, 12.5e3)
        self.assertTrue(self.win._transmit_job(job), self.status()[0])
        self.assertEqual(self.station.engine._run.plan.modes, ("FM",))
        self.station.stop()
        self.wait_done(10)

    def test_send_id_now_in_a_cw_only_segment_is_cw(self):
        """Auto picks CW where FM isn't allowed; an FM tone chosen by hand
        is refused there, with the reason."""
        self.panel.set_callsign("W1AW")
        self.hackrf(freq=144.05e6)
        self.assertIn("· CW ID ·", self.panel.tx_status_text())
        self.panel._emission_combo.setCurrentIndex(
            self.panel._emission_combo.findData("mcw")
        )
        self.assertEqual(
            self.panel.tx_status_text(),
            "Can't send on 144.050 MHz: FM isn't allowed in the 2m CW segment "
            "(CW only).",
        )

    def test_transmit_text_says_how_its_ids_go_out(self):
        self.panel.set_callsign("W1AW")
        self.hackrf(freq=50.29e6)
        self.win._show_transmit_text()
        dlg = self.win._transmit_text
        self.addCleanup(dlg.close)
        dlg.set_mode("PSK31")
        dlg.set_text("CQ CQ DE W1AW")
        dlg.refresh()
        line = dlg._tx_label.text()
        self.assertTrue(
            line.endswith(
                "· ID DE W1AW (CW, 20 WPM) at the start, every 10 min and at the end"
            ),
            line,
        )


# --------------------------------------------------------------------------- #
# TX-3: an automatically keyed CW ID goes out at 20 WPM at most
# --------------------------------------------------------------------------- #


class TestIdSpeed(_TxCase):
    license = "general"

    def test_the_id_speed_tops_out_at_20_wpm(self):
        """47 CFR 97.119(b)(1): a CW ID keyed by an automatic device may be
        sent at 20 WPM at most; the Ham ID speed can't be set higher, and a
        saved faster speed comes back as 20."""
        spin = self.panel._wpm_spin
        spin.setValue(40)
        self.assertEqual(self.panel.cw_wpm(), 20)
        self.assertEqual(spin.maximum(), 20)
        self.assertIn("20 WPM", spin.toolTip())
        self.panel.set_settings({"cw_wpm": 35})
        self.assertEqual(self.panel.cw_wpm(), 20)

    def test_ids_go_out_at_20_wpm_whatever_the_panel_says(self):
        self.panel.set_callsign("W1AW")
        dev, backend = self.hackrf(freq=14.05e6)
        self.panel.cw_wpm = lambda: 40  # (a speed from elsewhere)
        job = self.station.make_id_job(14.05e6)
        self.assertEqual(job.source.wpm, 20)
        carrier = np.full(1000, 0.9, np.complex64)
        text_job = TxJob(carrier, 2e6, 14.05e6, "CW", 150.0, "a CW test")
        self.assertEqual(self.station._id_factory()(text_job).source.wpm, 20)
        self.assertIn("CW, 20 WPM", self.station.id_spec(14.05e6)[3])
        # On the air: "DE W1AW" keyed over 67 dits of 60 ms (20 WPM).
        self.assertTrue(self.station.send_station_id())
        self.wait_done(30)
        on_air = analyse_cw(backend.tx_iq(32), 2e6 / 32)
        self.assertEqual(on_air["text"], "DE W1AW")
        self.assertAlmostEqual(on_air["keyed_s"], 4.02, delta=0.1)


# --------------------------------------------------------------------------- #
# TX-2 and TX-12: the HackRF driver's start and end of a transmission
# --------------------------------------------------------------------------- #

import time  # noqa: E402

from tests.hackrf_fakes import (  # noqa: E402
    BUFFER_LENGTH,
    ThreadedFakeHackRF,
    _CInt,
    install_fake_backend,
)


class _StartFailsHackRF(ThreadedFakeHackRF):
    """pyhackrf_start_tx() puts the radio in TX and submits transfers (the
    pump pulls samples), then fails part-way (HACKRF_ERROR_LIBUSB)."""

    def __init__(self):
        super().__init__(pace=True)
        self.stops = 0
        self.rx_stops = 0

    def pyhackrf_start_tx(self):
        super().pyhackrf_start_tx()
        time.sleep(0.05)
        raise RuntimeError("pyhackrf_start_tx() failed: HACKRF_ERROR_LIBUSB (-1000)")

    def pyhackrf_stop_tx(self):
        self.stops += 1
        super().pyhackrf_stop_tx()

    def pyhackrf_start_rx(self):
        raise RuntimeError("pyhackrf_start_rx() failed: HACKRF_ERROR_LIBUSB (-1000)")

    def pyhackrf_stop_rx(self):
        self.rx_stops += 1


class _QueuedHackRF(ThreadedFakeHackRF):
    """libhackrf's TX transfers (hackrf.c): four of 131072 samples are filled
    ahead; the radio sends each at the sample rate and the callback refills
    it; once the callback has no more, the ones still queued go out unless
    pyhackrf_stop_tx() cancels them first."""

    def __init__(self):
        super().__init__()
        self.queue = []
        self.sent = 0
        self.cancelled = 0
        self._run = False

    def _fill(self):
        buf = np.zeros(BUFFER_LENGTH, dtype=np.int8)
        length = _CInt(0)
        rc = self._tx_cb(self, buf, BUFFER_LENGTH, length)
        return rc, buf[: int(length.value)]

    def pyhackrf_start_tx(self):
        self.queue, self._run, streaming = [], True, True
        for _ in range(4):
            rc, data = self._fill()
            if rc or not len(data):
                streaming = False
                break
            self.queue.append(data)
        self._streaming = streaming
        self._thread = threading.Thread(target=self._device, daemon=True)
        self._thread.start()

    def _device(self):
        rate = float(self.sample_rate or 2e6)
        while self._run and self.queue:
            time.sleep(len(self.queue[0]) / 2 / rate)  # (on the air)
            if not self._run:
                break
            self.sent += len(self.queue.pop(0)) // 2
            if self._streaming:
                rc, data = self._fill()
                if rc or not len(data):
                    self._streaming = False
                else:
                    self.queue.append(data)

    def pyhackrf_stop_tx(self):
        self._run = False
        self.cancelled += sum(len(d) for d in self.queue) // 2
        self.queue = []
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(5)


def _open(monkeypatch, fake, rate=2e6):
    from sdr_module.core import frequency_manager as fm
    from sdr_module.core.frequency_manager import LicenseClass
    from sdr_module.devices.hackrf import HackRFDevice

    # (The license, put back after the test.)
    manager = fm.get_frequency_manager()
    monkeypatch.setattr(manager, "_license_class", LicenseClass.TECHNICIAN)
    install_fake_backend(monkeypatch, fake)
    device = HackRFDevice()
    assert device.open(0)
    device.set_sample_rate(rate)
    device.set_bandwidth(1.75e6)
    device.set_frequency(146.52e6)
    return device


class TestDriverStartAndEnd:
    def test_a_failed_start_stops_the_radio(self, monkeypatch):
        """TX-2: if pyhackrf_start_tx() fails after the radio went into TX,
        the driver stops it: nothing more is pulled, the engine says it
        failed."""
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui.transmitter import Transmitter

        QApplication.instance() or QApplication([])
        fake = _StartFailsHackRF()
        device = _open(monkeypatch, fake)
        carrier = np.full(int(5 * 2e6), 0.5, np.complex64)
        job = TxJob(carrier, 2e6, 146.52e6, "FM", 12.5e3, "a test carrier")
        engine = Transmitter()
        results = []
        engine.finished.connect(results.append)
        assert engine.start(job, device)
        assert engine.wait(10)
        assert results[0].status == "failed"
        assert "didn't start transmitting" in results[0].message
        assert fake.stops >= 1  # (the radio left TX)
        pulled = fake.tx_count()
        time.sleep(0.3)
        assert fake.tx_count() == pulled  # (nothing more goes out)
        assert not device.state.is_transmitting

    def test_a_failed_receive_start_stops_the_radio(self, monkeypatch):
        fake = _StartFailsHackRF()
        device = _open(monkeypatch, fake)
        assert device.start_rx(lambda samples: None) is False
        assert fake.rx_stops == 1
        assert not device.state.is_streaming

    def test_write_samples_sends_the_whole_block(self, monkeypatch):
        """TX-12: the transfers libhackrf still holds when the callback runs
        dry go out before the radio is stopped (the last 0.15 s of an ID was
        cancelled: "DE W1AR")."""
        fake = _QueuedHackRF()
        device = _open(monkeypatch, fake)
        samples = np.full(400000, 0.5, np.complex64)  # 0.2 s
        assert device.write_samples(samples, bandwidth_hz=10e3, mode="FM")
        assert fake.cancelled == 0
        assert fake.sent == len(samples)


# --------------------------------------------------------------------------- #
# TX-6 to TX-10: power on the air, the confirmation, settings, a sent job
# --------------------------------------------------------------------------- #


class TestPowerWhileOnTheAir(_TxCase):
    def test_power_controls_wait_for_the_end(self):
        """TX-6: the HackRF keeps the power a transmission started with, so
        the TX gain and the amplifier can't be changed while it is on the
        air (the estimate showed a power that wasn't)."""
        panel = self.panel
        panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        controls = (panel._tx_gain_spin, panel._tx_gain_slider, panel._amp_check)
        tips = [w.toolTip() for w in controls]
        self.assertTrue(all(w.isEnabled() for w in controls))
        self.assertTrue(self.station.send_station_id())
        for widget in controls:
            self.assertFalse(widget.isEnabled())
            self.assertIn("while transmitting", widget.toolTip())
        self.station.stop()
        self.wait_done(10)
        self.assertTrue(all(w.isEnabled() for w in controls))
        self.assertEqual([w.toolTip() for w in controls], tips)

    def test_the_estimate_shows_the_power_on_the_air(self):
        """Even a setting changed some other way while on the air shows
        only once the transmission ends: the Output line and the QRP tab
        give the power that is going out."""
        panel = self.panel
        panel.set_callsign("W1AW")
        self.hackrf(pace=True)
        before = panel._power_label.text()
        qrp_before = self.win._qrp_panel._amp_calc.sdr_dbm()
        self.assertTrue(self.station.send_station_id())
        panel._tx_gain_spin.setValue(47)  # (programmatically: it's disabled)
        panel._amp_check.setChecked(True)
        self.assertEqual(panel._power_label.text(), before)
        self.assertEqual(self.win._qrp_panel._amp_calc.sdr_dbm(), qrp_before)
        self.station.stop()
        self.wait_done(10)
        self.assertEqual(panel._power_label.text(), "about 10 mW (+10 dBm)")
        self.assertEqual(self.station.power.gain_db, 47)  # (the next one's)


class TestConfirmationPower(_TxCase):
    def test_the_question_gives_the_station_output(self):
        """TX-7: through the QRP tab's 100 W linear, the question states the
        100 W the power check uses, and the HackRF's own 10 mW."""
        self.panel.set_callsign("W1AW")
        self.hackrf()
        self.panel._tx_gain_spin.setValue(47)
        self.panel._amp_check.setChecked(True)
        self.win._qrp_panel._amp_calc.apply_preset("HackRF → 100 W linear")
        self.store.values["tx/confirm"] = True
        seen = []

        def answer(box):
            seen.append(f"{box.text()} {box.informativeText()}")
            box.clickedButton = lambda: box.escapeButton()  # Cancel
            return 0

        self.win._exec_dialog = answer
        self.assertFalse(self.station.send_station_id())
        self.assertIn("about 100 W", seen[-1])
        self.assertIn("10 mW", seen[-1])


class TestTransmitTextSettings(_TxCase):
    def test_wrong_types_in_the_saved_settings_are_ignored(self):
        """TX-8: a tx/text value of the wrong shape doesn't stop Tools >
        Transmit Text from opening; the defaults are used."""
        from sdr_module.gui.transmit_text import TransmitTextDialog

        for raw in (
            '{"modes": [1]}',
            '{"modes": "CW"}',
            '{"save_format": ["wav"]}',
            '{"mode": {"x": 1}, "text": 5}',
            '{"modes": {"CW": {"wpm": [1], "emission": 3}}}',
        ):
            self.store.values["tx/text"] = raw
            dialog = TransmitTextDialog(self.win)
            try:
                self.assertEqual(dialog.mode(), "RTTY", raw)
                self.assertEqual(dialog.text(), "", raw)
            finally:
                dialog.close()
                dialog.deleteLater()


class TestConfirmSetting(_TxCase):
    def test_only_an_explicit_no_skips_the_question(self):
        """TX-9: an empty, unknown or unreadable tx/confirm asks (it turned
        the confirmation off); "Don't ask again" (False) still skips it."""
        from sdr_module.gui.preferences_dialog import PreferencesDialog

        self.panel.set_callsign("W1AW")
        self.hackrf()
        asked = []

        def answer(box):
            asked.append(True)
            box.clickedButton = lambda: box.escapeButton()  # Cancel
            return 0

        self.win._exec_dialog = answer
        job = self.station.make_id_job(146.52e6)
        for raw, asks in (
            ("", True),
            ("ja", True),
            (None, True),
            ("true", True),
            (False, False),
            ("false", False),
            ("0", False),
            ("Off", False),
        ):
            self.store.values["tx/confirm"] = raw
            asked.clear()
            self.assertEqual(self.station.confirm(job), not asks, raw)
            self.assertEqual(bool(asked), asks, raw)
            dialog = PreferencesDialog(self.store)
            self.assertEqual(dialog.preferences().tx_confirm, asks, raw)
            dialog.deleteLater()


class TestSentJobs:
    def test_a_job_sent_again_goes_out_again(self, monkeypatch):
        """TX-10: a job whose source was read is sent from its start again
        (it reported "Sent ... (0.0 s)" with nothing on the air)."""
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui.transmitter import Transmitter

        QApplication.instance() or QApplication([])
        fake = ThreadedFakeHackRF()
        device = _open(monkeypatch, fake)
        job = TxJob(
            MorseIdSource("DE W1AW", 20, 2e6, emission="mcw"),
            2e6,
            146.52e6,
            "FM",
            6.4e3,
            "DE W1AW",
            is_id=True,
        )
        engine = Transmitter()
        results = []
        engine.finished.connect(results.append)
        for _ in range(2):
            assert engine.start(job, device)
            assert engine.wait(30)
            assert fake.tx_count() == job.total_samples
            assert results[-1].ok and results[-1].seconds > 4.0

    def test_a_used_source_that_cant_start_again_is_refused(self, monkeypatch):
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui.transmitter import Transmitter

        QApplication.instance() or QApplication([])

        class OneShot:
            total_samples = 1000

            def __init__(self):
                self.position = 0

            def read(self, count):
                n = max(0, min(count, self.total_samples - self.position))
                self.position += n
                return np.full(n, 0.5, np.complex64)

        device = _open(monkeypatch, ThreadedFakeHackRF())
        source = OneShot()
        source.read(1000)
        job = TxJob(source, 2e6, 146.52e6, "FM", 12.5e3, "a used carrier")
        engine = Transmitter()
        assert engine.check(job, device) == (
            "this transmission was already sent: make a new one to send it again"
        )
        assert not engine.start(job, device)
