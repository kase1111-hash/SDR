"""Two receivers sharing one sweep (``sdr_module.dsp.sweep.SharedSweep``),
the plan behind the dual-SDR wideband scan (K-X1).

Before, ``DualSDRController.set_mode(WIDEBAND_SCAN)`` refused: there was no
way to have two devices sweep complementary parts of one range. The plan is
driven here claim by claim, in the order a fast and a slow receiver would
claim pages, so how the range is shared doesn't depend on threads.
"""

import threading

import numpy as np
import pytest

from sdr_module.dsp.sweep import (
    USABLE_FRACTION,
    SharedSweep,
    SweepReceiver,
    channel_count,
    page_count,
    plan_sweep,
)

RTL = SweepReceiver(2.4e6, 0.5e6, 1766e6)
HACKRF = SweepReceiver(2.4e6, 1e6, 6e9)


def _drain(sweep, order):
    """Claim pages in ``order`` (receiver indices, repeated) until nothing
    is left; every page claimed is reported measured. Returns the pages per
    receiver."""
    pages = {i: [] for i in range(len(sweep.receivers))}
    idle = set()
    k = 0
    while len(idle) < len(set(order)):
        receiver = order[k % len(order)]
        k += 1
        if receiver in idle:
            continue
        page = sweep.claim(receiver)
        if page is None:
            idle.add(receiver)
            continue
        sweep.done(receiver, page)
        pages[receiver].append(page)
    return pages


def _indices(pages):
    return [p.first_index + i for p in pages for i in range(p.count)]


def _check_page_fits(page, receiver, step_hz):
    """The page fits the receiver: its LO and channels are inside what it
    tunes, and the page (half a step either side) inside the usable span."""
    channels = page.channels
    assert receiver.low_hz <= page.lo_hz <= receiver.high_hz
    assert receiver.low_hz <= channels.min() and channels.max() <= receiver.high_hz
    half_usable = 0.5 * USABLE_FRACTION * receiver.sample_rate
    assert channels.min() - step_hz / 2 >= page.lo_hz - half_usable - 1e-6
    assert channels.max() + step_hz / 2 <= page.lo_hz + half_usable + 1e-6


class TestSharing:
    def test_two_receivers_meet_in_the_middle_each_channel_once(self):
        sweep = SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF])
        total = channel_count(88e6, 108e6, 100e3)
        assert sweep.channel_total == total == 201
        assert sweep.covered_count == total
        assert not sweep.finished
        pages = _drain(sweep, [0, 1])
        assert sweep.finished
        first, second = _indices(pages[0]), _indices(pages[1])
        # Complementary: every channel once, nothing twice.
        assert sorted(first + second) == list(range(total))
        # Contiguous parts: one from the bottom up, the other from the top
        # down, meeting where they met.
        assert first == list(range(len(first)))
        assert sorted(second) == list(range(len(first), total))
        assert pages[0][0].first_index == 0
        assert pages[1][0].first_index + pages[1][0].count - 1 == total - 1
        for page in pages[0]:
            _check_page_fits(page, RTL, 100e3)
        for page in pages[1]:
            _check_page_fits(page, HACKRF, 100e3)
        assert sweep.progress == pytest.approx(1.0)
        # Each one's part, as frequency ranges.
        ((low, high),) = sweep.coverage(0)
        assert low == pytest.approx(88e6)
        ((low2, high2),) = sweep.coverage(1)
        assert high2 == pytest.approx(108e6)
        assert low2 == pytest.approx(high + 100e3)
        assert sweep.missed() == []

    def test_the_faster_receiver_takes_more(self):
        """Claims follow speed: one receiver claiming three pages for each
        of the other's ends up with about three times the channels (no
        timing model: the split falls where they meet)."""
        sweep = SharedSweep(400e6, 470e6, 100e3, [RTL, HACKRF])
        pages = _drain(sweep, [0, 1, 1, 1])
        slow, fast = len(_indices(pages[0])), len(_indices(pages[1]))
        assert slow + fast == 701
        assert 2.5 <= fast / slow <= 3.5

    def test_a_faster_rate_measures_wider_pages(self):
        fast = SweepReceiver(10e6, 1e6, 6e9)
        sweep = SharedSweep(88e6, 108e6, 100e3, [RTL, fast])
        pages = _drain(sweep, [0, 1])
        # 2.4 MS/s pages hold 19 channels of 100 kHz (as plan_sweep's),
        # 10 MS/s pages 79.
        assert max(p.count for p in pages[0]) == 19
        assert plan_sweep(88e6, 108e6, 100e3, 2.4e6)[0].count == 19
        assert max(p.count for p in pages[1]) == 79
        for page in pages[1]:
            _check_page_fits(page, fast, 100e3)
        # Equal turns: the wider pages cover more of the range.
        assert len(_indices(pages[1])) > 3 * len(_indices(pages[0]))

    def test_each_receiver_stays_inside_what_it_tunes(self):
        """1700-1900 MHz: the RTL-SDR tunes up to 1766 MHz only; above that
        is the HackRF's alone, and it claims that part first."""
        sweep = SharedSweep(1700e6, 1900e6, 100e3, [RTL, HACKRF])
        # The plan, as the CLI tells it.
        assert sweep.only_tuned_by(0) == []
        ((low, high),) = sweep.only_tuned_by(1)
        assert low == pytest.approx(1766.1e6) and high == pytest.approx(1900e6)
        ((low, high),) = sweep.shared()
        assert low == pytest.approx(1700e6) and high == pytest.approx(1766e6)
        assert sweep.sweeps_up(0) and not sweep.sweeps_up(1)
        first_claim = sweep.claim(1)
        assert first_claim.channels.min() > 1766e6
        sweep.done(1, first_claim)
        pages = _drain(sweep, [0, 1])
        pages[1].insert(0, first_claim)
        for page in pages[0]:
            _check_page_fits(page, RTL, 100e3)
            assert page.channels.max() <= 1766e6
        for page in pages[1]:
            _check_page_fits(page, HACKRF, 100e3)
        indices = sorted(_indices(pages[0]) + _indices(pages[1]))
        assert indices == list(range(sweep.channel_total))

    def test_what_only_one_tunes_comes_first(self):
        """Nested ranges: the outer receiver measures both of its own ends
        before it joins the middle, which the inner one starts on."""
        inner = SweepReceiver(2.4e6, 100e6, 200e6)
        outer = SweepReceiver(2.4e6, 1e6, 6e9)
        sweep = SharedSweep(50e6, 250e6, 1e6, [inner, outer])
        order = []
        for _ in range(200):
            page = sweep.claim(1)
            if page is None:
                break
            sweep.done(1, page)
            order.append(page)
            if page.channels.max() > 200e6 and page.channels.max() >= 250e6:
                break
        # The outer one's first pages: below 100 MHz, then above 200 MHz.
        below = [p for p in order if p.channels.max() < 100e6]
        above = [p for p in order if p.channels.min() > 200e6]
        assert below and above
        assert order.index(below[-1]) < order.index(above[0])
        # The inner one takes the shared middle from its own end.
        page = sweep.claim(0)
        assert 100e6 <= page.channels.min() and page.channels.max() <= 200e6
        sweep.done(0, page)
        pages = _drain(sweep, [0, 1])
        pages[1] = order + pages[1]
        pages[0].insert(0, page)
        indices = sorted(_indices(pages[0]) + _indices(pages[1]))
        assert indices == list(range(sweep.channel_total))
        for p in pages[0]:
            _check_page_fits(p, inner, 1e6)

    def test_channels_no_receiver_tunes_are_left_out_and_said(self):
        sweep = SharedSweep(0.1e6, 2e6, 10e3, [RTL, HACKRF])
        ((low, high),) = sweep.uncovered
        assert low == pytest.approx(0.1e6) and high == pytest.approx(0.49e6)
        assert sweep.covered_count == sweep.channel_total - 40
        pages = _drain(sweep, [0, 1])
        everything = _indices(pages[0]) + _indices(pages[1])
        assert sorted(everything) == list(range(40, sweep.channel_total))
        assert sweep.progress == pytest.approx(1.0)

    def test_one_receiver_plans_as_plan_sweep_does(self):
        sweep = SharedSweep(88e6, 108e6, 200e3, [SweepReceiver(2.4e6)])
        assert sweep.shared() == [] and sweep.only_tuned_by(0) == []
        assert sweep.sweeps_up(0)
        pages = _drain(sweep, [0])[0]
        planned = plan_sweep(88e6, 108e6, 200e3, 2.4e6)
        assert [(p.lo_hz, p.first_index, p.count) for p in pages] == [
            (p.lo_hz, p.first_index, p.count) for p in planned
        ]

    def test_the_lo_stays_inside_the_tuning_range(self):
        """A one-channel page right at the top of what a receiver tunes:
        its LO goes beside the channel on the side it can tune."""
        top = SweepReceiver(2.4e6, 24e6, 100e6)
        sweep = SharedSweep(99e6, 100e6, 1e6, [top])
        pages = _drain(sweep, [0])[0]
        assert [p.count for p in pages] == [1, 1]
        for page in pages:
            assert 24e6 <= page.lo_hz <= 100e6
            assert page.lo_hz != page.channels[0]  # (off the DC spike)


class TestFailures:
    def test_a_page_released_goes_to_the_other_receiver(self):
        sweep = SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF])
        page = sweep.claim(0)
        assert not sweep.finished  # (a page is being measured)
        sweep.release(0, page)
        # Not to the one that couldn't measure it ...
        again = sweep.claim(0)
        assert again.first_index > page.first_index + page.count - 1
        sweep.done(0, again)
        # ... but to the other, which measures it when it gets there.
        pages = _drain(sweep, [1])
        assert set(_indices([page])) <= set(_indices(pages[1]))
        assert sweep.missed() == []

    def test_released_with_nobody_else_to_take_it_is_missed(self):
        sweep = SharedSweep(1760e6, 1800e6, 100e3, [RTL, HACKRF])
        # Above 1766 MHz only the HackRF tunes: it can't measure a page.
        page = sweep.claim(1)
        assert page.channels.min() > 1766e6
        sweep.release(1, page)
        _drain(sweep, [0, 1])
        # Done: the page is left, but no one can measure it.
        assert sweep.finished
        assert sweep.progress == pytest.approx(1.0)
        missed = sweep.missed()
        assert missed == [(page.first_hz, page.last_hz)]

    def test_a_receiver_that_gives_up_leaves_the_shared_part_to_the_other(self):
        sweep = SharedSweep(1700e6, 1900e6, 100e3, [RTL, HACKRF])
        sweep.give_up(1)
        assert sweep.claim(1) is None
        pages = _drain(sweep, [0])
        # The RTL-SDR measured everything it tunes; above 1766 MHz is lost.
        assert _indices(pages[0]) == list(range(0, 661))
        assert sweep.progress == pytest.approx(1.0)
        ((low, high),) = sweep.missed()
        assert low == pytest.approx(1766.1e6) and high == pytest.approx(1900e6)

    def test_reset_starts_a_new_pass(self):
        sweep = SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF])
        sweep.give_up(1)
        _drain(sweep, [0])
        sweep.reset()
        assert sweep.progress == 0.0
        assert sweep.coverage(0) == [] and sweep.missed() == []
        pages = _drain(sweep, [0, 1])
        assert pages[1], "a receiver that gave up is back in the next pass"


class TestEdges:
    def test_bad_receivers_are_refused(self):
        with pytest.raises(ValueError):
            SharedSweep(88e6, 108e6, 100e3, [])
        with pytest.raises(ValueError):
            SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF, RTL])
        with pytest.raises(ValueError):
            SharedSweep(88e6, 108e6, 100e3, [SweepReceiver(0.0)])
        with pytest.raises(ValueError):
            SharedSweep(88e6, 108e6, 100e3, [SweepReceiver(float("nan"))])

    def test_an_empty_range_plans_nothing(self):
        sweep = SharedSweep(108e6, 88e6, 100e3, [RTL, HACKRF])
        assert sweep.channel_total == 0 and sweep.covered_count == 0
        assert sweep.claim(0) is None and sweep.claim(1) is None
        assert sweep.progress == 1.0

    def test_a_huge_range_costs_nothing_to_plan(self):
        """6 GHz in 1 kHz steps: six million channels, planned and claimed
        without channel lists (pages hold only where they start)."""
        sweep = SharedSweep(1e6, 6e9, 1e3, [RTL, SweepReceiver(20e6, 1e6, 6e9)])
        assert sweep.channel_total == channel_count(1e6, 6e9, 1e3)
        page = sweep.claim(1)
        assert page.count <= 20e6 * USABLE_FRACTION / 1e3
        assert sweep.estimate_seconds([0.15, 0.13]) > 0

    def test_estimate_follows_the_speeds(self):
        sweep = SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF])
        alone = SharedSweep(88e6, 108e6, 100e3, [RTL]).estimate_seconds([0.15])
        together = sweep.estimate_seconds([0.15, 0.15])
        assert together == pytest.approx(alone / 2, rel=0.15)
        # The estimate doesn't claim anything.
        assert sweep.progress == 0.0 and sweep.claim(0).first_index == 0

    def test_the_estimate_counts_each_receivers_pages(self):
        """(J-dual) The Scanner tab says how many steps a pass shared with
        a second receiver takes: as the plan hands them out."""
        plan = SharedSweep(88e6, 108e6, 100e3, [RTL, HACKRF])
        seconds, pages = plan.estimate([0.15, 0.15])
        assert seconds == plan.estimate_seconds([0.15, 0.15])
        # Each page of the plan once (RTL and HACKRF have the same rate
        # here: their pages are the same size), shared evenly.
        total = page_count(88e6, 108e6, 100e3, RTL.sample_rate)
        assert sum(pages) in (total, total + 1)
        assert abs(pages[0] - pages[1]) <= 1
        assert seconds == pytest.approx(max(pages) * 0.15)
        # The faster receiver (a quicker step) takes more of them.
        _seconds, uneven = plan.estimate([0.30, 0.10])
        assert uneven[1] > 2 * uneven[0]
        assert abs(sum(uneven) - sum(pages)) <= 1
        # The estimate doesn't claim anything.
        assert plan.progress == 0.0 and plan.claim(0).first_index == 0

    def test_threads_claim_each_channel_once(self):
        sweep = SharedSweep(24e6, 1766e6, 12.5e3, [RTL, HACKRF])
        got = {0: [], 1: []}

        def work(i):
            while True:
                page = sweep.claim(i)
                if page is None:
                    return
                got[i].append((page.first_index, page.count))
                sweep.done(i, page)

        threads = [threading.Thread(target=work, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        counts = np.zeros(sweep.channel_total, dtype=int)
        for i in (0, 1):
            for first, count in got[i]:
                counts[first : first + count] += 1
        assert counts.min() == 1 and counts.max() == 1
        assert sweep.progress == pytest.approx(1.0)
