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
from tests.hackrf_fakes import _MORSE  # noqa: E402

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

