"""J-decode: the Mode S fit's work per second is bounded at 2 to 2.5 MS/s.

Below 2.5 MS/s (2.4 MS/s, the rate ADS-B is offered at, included) every
cluster of pulses that passed the preamble screen got the 48-start-time
fit, and every fit that looked like a reply a Viterbi search at five start
times. Mode A/C replies and pulse interference make many such clusters, so
a busy sky took 4-10 ms of the 33 ms display frame on the GUI thread and
dense pulse interference 14-23 ms (the slot reader before it: about 2.5).
Now a cheap test of the data block (a Mode S reply has a pulse in every
bit, fruit and stray pulses don't) and of both halves of the preamble
comes before the fit, a quick reading of each bit settles most replies
before the Viterbi search, and what is left of the work has a budget per
second of signal (the clearest preambles first), so no input can take more.
Work is counted, not timed, so the tests don't depend on the machine.
"""

import math
from contextlib import contextmanager

import numpy as np
import pytest

from sdr_module.dsp import mode_s
from sdr_module.dsp.protocols import ADSBDecoder
from sdr_module.gui import device_dialog as dd

GUI_BLOCK_S = 0.034  # one display frame at 30 Hz
RATES = [2.0e6, 2.048e6, 2.4e6]
# The bound on the work, per second of signal: preamble fits and timing
# hypotheses searched by the Viterbi search (the budget lets a burst use
# up to a twentieth of a second's worth at once).
FITS_PER_S = 20_000
SEARCHES_PER_S = 10_000
BURST_S = 0.05


@contextmanager
def _counting():
    """Count the preamble fits and the Viterbi searches (rows) run."""
    counts = {"fits": 0, "searches": 0}
    fit, search = mode_s.ModeSDemodulator._fit, mode_s.ModeSDemodulator._search

    def counted_fit(self, buf, first, last):
        counts["fits"] += len(first)
        return fit(self, buf, first, last)

    def counted_search(self, buf, start, level, floor):
        counts["searches"] += len(start)
        return search(self, buf, start, level, floor)

    mode_s.ModeSDemodulator._fit = counted_fit
    mode_s.ModeSDemodulator._search = counted_search
    try:
        yield counts
    finally:
        mode_s.ModeSDemodulator._fit = fit
        mode_s.ModeSDemodulator._search = search


def _noise(rate, seconds, rng):
    n = int(rate * seconds)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return x * np.sqrt(0.5)


def _place(x, rate, pulses, t, amplitude, rng):
    """Add a reply's pulses (an integrating ADC's samples) at ``t`` s."""
    first, env = dd._pulse_envelope(pulses, t * rate, rate)
    end = min(len(x), first + len(env))
    if first < len(x):
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
        x[first:end] += amplitude * env[: end - first] * phase


def _df11(icao):
    head = bytes([11 << 3 | 5]) + icao.to_bytes(3, "big")
    return head + dd._crc24(head).to_bytes(3, "big")


def _busy_sky(rate, seconds, seed, modes_per_s=2000, fruit_per_s=6000):
    """Mode S replies (DF17, DF11, DF4) from 50 aircraft and Mode A/C fruit
    at random times and strengths (12-35 dB above the noise in 2 MHz)."""
    rng = np.random.default_rng(seed)
    x = _noise(rate, seconds, rng) * math.sqrt(rate / 2e6)
    icaos = [int(a) for a in rng.integers(0x100000, 0xFFFFFF, 50)]
    for t in rng.uniform(0, seconds, int(fruit_per_s * seconds)):
        code = int(rng.integers(0, 1 << 12))
        _place(
            x, rate, dd._mode_ac_pulses(code), t, 10 ** (rng.uniform(12, 35) / 20), rng
        )
    for t in rng.uniform(0, seconds, int(modes_per_s * seconds)):
        icao = icaos[int(rng.integers(0, len(icaos)))]
        kind = rng.random()
        if kind < 0.5:
            frame = dd._extended_squitter(
                icao, (11 << 51) | int(rng.integers(0, 1 << 51))
            )
        elif kind < 0.75:
            frame = _df11(icao)
        else:
            frame = dd._address_parity_reply(
                4 << 27 | int(rng.integers(0, 1 << 13)), icao
            )
        amp = 10 ** (rng.uniform(12, 35) / 20)
        _place(x, rate, dd._modes_pulses(frame), t, amp, rng)
    return np.abs(x).astype(np.float32)


def _decode(decoder, mag, block):
    out = []
    for start in range(0, len(mag), block):
        out += [bytes(m.raw_bits) for m in decoder.decode(mag[start : start + block])]
    return out


@pytest.mark.parametrize("rate", RATES)
def test_pulse_interference_costs_little_work(rate):
    # 100,000 random 0.5 us pulses a second, 10-30 dB above the noise
    # (Before: about 3,500 fits per display frame, 100,000 a second.)
    rng = np.random.default_rng(9)
    seconds = 0.5
    x = _noise(rate, seconds, rng)
    width = max(1, round(0.5e-6 * rate))
    count = int(100_000 * seconds)
    starts = rng.integers(0, len(x) - width - 1, count)
    amps = 10 ** (rng.uniform(10, 30, count) / 20)
    for s, a in zip(starts.tolist(), amps.tolist(), strict=True):
        x[s : s + width] += a * np.exp(1j * rng.uniform(0, 2 * np.pi))
    mag = np.abs(x).astype(np.float32)
    with _counting() as counts:
        found = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    assert found == []
    assert counts["fits"] <= FITS_PER_S * (seconds + BURST_S)
    assert counts["searches"] <= SEARCHES_PER_S * (seconds + BURST_S)


@pytest.mark.parametrize("rate", RATES)
def test_mode_a_c_fruit_costs_little_work(rate):
    # 30,000 Mode A/C replies a second (no Mode S): the older replies'
    # pulses make preamble-like clusters but have no data block. (Before:
    # about 75,000 fits a second.)
    rng = np.random.default_rng(4)
    seconds = 0.5
    x = _noise(rate, seconds, rng) * math.sqrt(rate / 2e6)
    for t in rng.uniform(0, seconds, int(30_000 * seconds)):
        code = int(rng.integers(0, 1 << 12))
        _place(
            x, rate, dd._mode_ac_pulses(code), t, 10 ** (rng.uniform(10, 35) / 20), rng
        )
    mag = np.abs(x).astype(np.float32)
    with _counting() as counts:
        found = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    assert found == []
    assert counts["fits"] <= FITS_PER_S * (seconds + BURST_S)
    assert counts["searches"] <= SEARCHES_PER_S * (seconds + BURST_S)


def _garbage(rate, seconds, rng, db=(20, 30)):
    """Mode S-like replies with random bits (no valid parity) back to back,
    2 us apart: clean preambles, a full data block, and in it the
    preamble-like patterns random bits make."""
    x = _noise(rate, seconds, rng) * math.sqrt(rate / 2e6)
    t = 10e-6
    while t < seconds - 130e-6:
        frame = bytes(rng.integers(0, 256, 14, dtype=np.uint8).tolist())
        amp = 10 ** (rng.uniform(*db) / 20)
        _place(x, rate, dd._modes_pulses(frame), t + rng.uniform(0, 1e-6), amp, rng)
        t += 122e-6
    return x


@pytest.mark.parametrize("rate", RATES)
def test_no_input_takes_more_than_the_budget(rate):
    # Every reply passes the cheap tests and fits; the budget still holds.
    rng = np.random.default_rng(6)
    seconds = 0.4
    mag = np.abs(_garbage(rate, seconds, rng)).astype(np.float32)
    with _counting() as counts:
        _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    assert counts["fits"] <= FITS_PER_S * (seconds + BURST_S)
    assert counts["searches"] <= SEARCHES_PER_S * (seconds + BURST_S)


@pytest.mark.parametrize("rate", RATES)
def test_the_clearest_replies_survive_an_overload(rate):
    # Garbage like the above at 15-25 dB, every fifth reply a real
    # squitter at 35 dB instead: what the full search reads of them, the
    # budget (the clearest preambles first) reads too.
    rng = np.random.default_rng(8)
    seconds = 0.4
    x = _noise(rate, seconds, rng) * math.sqrt(rate / 2e6)
    real = []
    t, k = 10e-6, 0
    while t < seconds - 130e-6:
        if k % 5 == 0:
            frame = dd._extended_squitter(0x400000 + k, (11 << 51) | k)
            real.append(frame)
            amp = 10 ** (35 / 20)
        else:
            frame = bytes(rng.integers(0, 256, 14, dtype=np.uint8).tolist())
            amp = 10 ** (rng.uniform(15, 25) / 20)
        _place(x, rate, dd._modes_pulses(frame), t + rng.uniform(0, 1e-6), amp, rng)
        t, k = t + 122e-6, k + 1
    mag = np.abs(x).astype(np.float32)
    block = int(rate * GUI_BLOCK_S)
    full = ADSBDecoder(rate)
    full._fine.FITS_PER_S = full._fine.SEARCHES_PER_S = math.inf
    full._fine.reset()
    reference = set(_decode(full, mag, block)) & set(real)
    budget = ADSBDecoder(rate)
    found = set(_decode(budget, mag, block))
    assert len(reference) > 0.9 * len(real)
    assert reference <= found
    assert budget._fine.skipped > 0  # (the budget was reached)


@pytest.mark.parametrize("rate", RATES)
def test_a_busy_sky_is_read_in_full(rate):
    # 2,000 Mode S replies and 6,000 Mode A/C replies a second: the cheap
    # tests, the quick first reading and the budget read at least as many
    # replies as the full search at every start time, and the budget isn't
    # reached. (Where two replies overlap, the quick reading can read the
    # first where the search couldn't; the second, starting inside it, is
    # then skipped, as it always was after a reply read: a rare swap.)
    mag = _busy_sky(rate, 0.5, seed=int(rate) % 101)
    block = int(rate * GUI_BLOCK_S)
    full = ADSBDecoder(rate)
    full._fine.DATA = -math.inf  # no data-block test
    full._fine.BALANCE = -math.inf  # no preamble-halves test
    full._fine.HARD_PASS = False  # every reply searched at every start time
    full._fine.FITS_PER_S = full._fine.SEARCHES_PER_S = math.inf
    full._fine.reset()
    reference = _decode(full, mag, block)
    quick = ADSBDecoder(rate)
    found = _decode(quick, mag, block)
    assert len(reference) > 300
    assert len(set(found)) >= len(set(reference))
    assert len(set(reference) - set(found)) <= len(reference) // 100
    assert quick._fine.skipped == 0


@pytest.mark.parametrize("rate", RATES)
def test_a_busy_sky_in_one_block_reads_the_same(rate):
    # Decode Recording and the command line feed seconds of samples at
    # once: the budget for a block grows with it, so a busy sky read in
    # one block reads as it does a display frame at a time.
    mag = _busy_sky(rate, 0.5, seed=int(rate) % 103)
    framed = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    whole = ADSBDecoder(rate)
    at_once = _decode(whole, mag, len(mag))
    assert len(framed) > 300
    assert at_once == framed
    assert whole._fine.skipped == 0


def _spaced(rate, count, snr_db, seed):
    """``count`` DF17 squitters 400 us apart, at every chip phase."""
    rng = np.random.default_rng(seed)
    n = int((count + 2) * 400e-6 * rate)
    x = _noise(rate, n / rate, rng) * math.sqrt(10 ** (-snr_db / 10) * rate / 2e6)
    frames = []
    for k in range(count):
        frame = dd._extended_squitter(
            0x500000 + k, (11 << 51) | int(rng.integers(0, 1 << 51))
        )
        frames.append(frame)
        t = (math.floor((k + 1) * 400e-6 * rate) + (k + 0.5) / count) / rate
        _place(x, rate, dd._modes_pulses(frame), t, 1.0, rng)
    return frames, np.abs(x).astype(np.float32)


@pytest.mark.parametrize("rate", RATES)
def test_the_quick_first_reading_finds_what_the_search_finds(rate):
    # Each reply's bits are first read quickly at its fitted start time and
    # searched only when that frame doesn't check itself: it finds every
    # reply the search alone finds (and now and then one it misses).
    frames, mag = _spaced(rate, 60, 13.0, seed=21)
    found = []
    for quick in (True, False):
        decoder = ADSBDecoder(rate)
        decoder._fine.HARD_PASS = quick
        found.append(_decode(decoder, mag, len(mag)))
    assert set(found[1]) <= set(found[0]) <= set(frames)
    assert len(found[1]) > 20
    assert found[0] == sorted(found[0], key=frames.index)  # (in order, once)


@pytest.mark.parametrize("rate", RATES)
def test_dense_strong_replies_are_not_left_for_lack_of_budget(rate):
    # 3,300 strong replies a second (one every 300 us): the quick reading
    # settles few at 2.0-2.048 MS/s, where the phase drifts through half a
    # sample within each reply; the search at the fitted start time first
    # keeps them all within the budget.
    rng = np.random.default_rng(31)
    count = 200
    n = int((count + 2) * 300e-6 * rate)
    x = _noise(rate, n / rate, rng) * math.sqrt(0.01 * rate / 2e6)
    frames = []
    for k in range(count):
        frame = dd._extended_squitter(
            0x600000 + k, (11 << 51) | int(rng.integers(0, 1 << 51))
        )
        frames.append(frame)
        t = (k + 1) * 300e-6 + rng.uniform(0, 20e-6)
        _place(x, rate, dd._modes_pulses(frame), t, 1.0, rng)
    decoder = ADSBDecoder(rate)
    found = set(_decode(decoder, np.abs(x).astype(np.float32), int(rate * GUI_BLOCK_S)))
    assert decoder._fine.skipped == 0
    assert len(found & set(frames)) >= 0.98 * count
