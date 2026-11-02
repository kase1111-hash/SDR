"""Mode S replies from magnitude samples at 2 to 2.5 MS/s.

A Mode S reply (ADS-B on 1090 MHz) is sent in 0.5 us chips: an 8 us
preamble (pulses in chips 0, 2, 7 and 9), then 56 or 112 bits of 1 us,
each a pulse in its first chip (a 1) or its second (a 0). At 2 to 2.5 MS/s
a sample lasts 0.8 to 1 chip, so wherever the reply starts most samples
straddle two chips and hold a mix of both: at exactly 2 MS/s a reply that
starts half a sample off the sample grid gives every sample of a bit the
same value, and only the samples across the boundary of two bits tell the
bits apart (a 1 then a 0 leaves them dark, a 0 then a 1 bright). Reading
the magnitude in 0.5 us slots and comparing a bit's two halves loses such
replies, and at 2.048 MS/s every reply drifts through that phase several
times.

:class:`ModeSDemodulator` reads the samples as what they are:

1. A preamble screen: the samples over the four pulses (wherever in a
   sample the reply starts) against the samples that are certainly quiet.
2. Cheap tests of what passed: both halves of the preamble hold pulses,
   and the first 56 bits hold energy (a reply has a pulse in every bit;
   Mode A/C replies and stray pulses, which make most of what passes the
   screen in a busy sky, don't).
3. A least-squares fit of the preamble, as an integrating ADC samples it,
   at a grid of start times a fraction of a sample apart: the reply's start
   time, pulse level and noise floor.
4. A quick reading of each bit at the fitted start time: from its own
   samples and the samples it shares with the bits on either side (the
   other bit taken as unknown). A reply whose frame then checks itself is
   read; most strong ones are.
5. For the others, a Viterbi search for the bits: each sample's expected
   value is the pulse level times the share of the sample each of its two
   chips covers, so the bits are the sequence whose samples fit best (two
   states, the last bit). A short reply's search ends after bit 55. It
   runs at five start times around the fitted one, the fitted one first.
6. The CRC and format checks of the decoder that owns it (``accept``).

The fit and the search are vectorised across replies: a block's replies
are searched together, whatever their number. The fits and the searches
have a budget per second of signal (:attr:`ModeSDemodulator.FITS_PER_S`,
:attr:`~ModeSDemodulator.SEARCHES_PER_S`), so no input, however busy,
costs more than it allows: when there is more to do, the clearest
preambles are read first and the rest are counted in ``skipped``.
"""

import math
from typing import Callable, Collection, Dict, Iterable, Sequence, Tuple

import numpy as np

#: Sample rates this demodulator reads (the upper bound excluded): from 2.5
#: MS/s up a chip spans more than one and a quarter samples and the slot
#: reader in ``ADSBDecoder`` sees it well.
MIN_RATE = 2.0e6
MAX_RATE = 2.5e6

CHIP_S = 0.5e-6
LONG_BITS = 112
SHORT_BITS = 56
SHORT_FORMATS = (0, 4, 5, 11)
LONG_FORMATS = (16, 17, 18, 20, 21)
_PREAMBLE = np.zeros(16)
_PREAMBLE[[0, 2, 7, 9]] = 1.0

_SHORT = np.zeros(32, dtype=bool)
_SHORT[list(SHORT_FORMATS)] = True
_LONG = np.zeros(32, dtype=bool)
_LONG[list(LONG_FORMATS)] = True
_SQUITTER = np.zeros(32, dtype=bool)
_SQUITTER[[17, 18]] = True

_CRC_POLY = 0x1FFF409


def _crc24(data: bytes) -> int:
    """Mode S CRC-24 of ``data`` (bitwise)."""
    crc = 0
    for byte in data:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= _CRC_POLY
    return crc & 0xFFFFFF


def _residual_table(length: int) -> np.ndarray:
    """``table[i, v]``: the CRC residual (CRC of the data XOR the parity)
    that byte value ``v`` at position ``i`` of a ``length``-byte reply
    contributes; the residual is linear, so a reply's is the XOR of its
    bytes' entries (and a byte's of its bits')."""
    data_len = length - 3
    bit_residual = np.zeros((length, 8), dtype=np.int64)
    for i in range(length):
        for bit in range(8):
            frame = bytearray(length)
            frame[i] = 0x80 >> bit
            parity = int.from_bytes(bytes(frame[data_len:]), "big")
            bit_residual[i, bit] = _crc24(bytes(frame[:data_len])) ^ parity
    bits = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(bool)
    table = np.zeros((length, 256), dtype=np.int64)
    for bit in range(8):
        table[:, bits[:, bit]] ^= bit_residual[:, bit : bit + 1]
    return table


_RESIDUALS: Dict[int, np.ndarray] = {}


def crc_residuals(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CRC residuals of ``frames`` (rows of 14 bytes) read as short replies
    (the first 7 bytes) and as long ones."""
    if not _RESIDUALS:
        _RESIDUALS[7] = _residual_table(7)
        _RESIDUALS[14] = _residual_table(14)
    rows = frames.astype(np.int64)
    short = np.bitwise_xor.reduce(_RESIDUALS[7][np.arange(7), rows[:, :7]], axis=1)
    long = np.bitwise_xor.reduce(_RESIDUALS[14][np.arange(14), rows], axis=1)
    return short, long


def _traceback_vectorised(
    from0: np.ndarray, from1: np.ndarray, final: np.ndarray
) -> np.ndarray:
    """The states of the best paths, given which state each step's state 0
    (``from0``) and state 1 (``from1``) came from (True: state 1) and the
    last state; ``from0[k]`` is step ``k + 1``'s. Each step maps the next
    state to the one before as a constant, the identity or a negation, so
    a state is the next constant step's value XOR the negations between."""
    steps = from0.shape[0]
    const = from0 == from1
    negate = (from0 & ~from1).astype(np.int16)
    flips = np.zeros((steps + 1,) + final.shape, dtype=np.int16)
    flips[:steps] = np.cumsum(negate[::-1], axis=0, dtype=np.int16)[::-1] & 1
    index = np.where(const, np.arange(steps)[:, None], steps)
    anchor = np.minimum.accumulate(index[::-1], axis=0)[::-1]
    at = np.minimum(anchor, steps - 1)
    value = np.where(
        anchor < steps, np.take_along_axis(from0, at, axis=0), final[None, :]
    )
    states = np.empty((steps + 1,) + final.shape, dtype=bool)
    states[:steps] = value ^ (
        flips[:steps] ^ np.take_along_axis(flips, anchor, 0)
    ).astype(bool)
    states[steps] = final
    return states


def _traceback_loop(
    from0: np.ndarray, from1: np.ndarray, final: np.ndarray
) -> np.ndarray:
    """:func:`_traceback_vectorised` a step at a time (cheaper for many
    rows)."""
    steps = from0.shape[0]
    states = np.empty((steps + 1,) + final.shape, dtype=bool)
    state = final
    states[steps] = state
    for k in range(steps - 1, -1, -1):
        state = np.where(state, from1[k], from0[k])
        states[k] = state
    return states


def _traceback(from0: np.ndarray, from1: np.ndarray, final: np.ndarray) -> np.ndarray:
    if final.shape[0] <= 40:
        return _traceback_vectorised(from0, from1, final)
    return _traceback_loop(from0, from1, final)


class ModeSDemodulator:
    """Mode S replies in magnitude samples at 2 to 2.5 MS/s (see the module
    docstring). Feed the stream in blocks of any size with :meth:`feed`;
    replies that straddle two blocks are found once, and the replies found
    don't depend on the block sizes (while the budget isn't reached: then
    the clearest are read, and which those are depends on the blocks).

    Not thread-safe: one stream, one thread.
    """

    #: Start times tried around each reply's fitted one (samples), in the
    #: order they are believed: the fit's timing is good to about a tenth
    #: of a sample at 15 dB, and the bits need it to a few hundredths.
    OFFSETS = (0.0, -0.08, 0.08, -0.16, 0.16)
    #: Preamble screen: the pulse samples' mean over the quiet samples'.
    SCREEN = 2.2
    #: Fit: least correlation with the preamble, least pulse level over the
    #: noise floor (about 9 dB).
    FIT = 0.8
    LEVEL = 2.7
    #: Start times fitted per sample.
    FIT_STEPS = 8
    #: Fractional start times the bit tables are made for.
    PHASES = 32
    #: Samples scanned per pass (bounds the memory a huge block takes).
    PASS = 1 << 20
    #: Up to this many replies left to search in a pass, all start times
    #: are searched at once (see :meth:`_frames`): a search costs about
    #: 50 replies' worth of work to set up, so one costs less than two
    #: unless more than about 12 of them settle at the fitted start time.
    ONE_PASS = 12
    #: Cheap tests before the fit (see :meth:`_promising`): the weaker
    #: preamble half's pulse level over the noise floor, at least this
    #: share of the stronger's; the first 56 bits' mean level over the
    #: floor, at least this share of the preamble pulses' (a reply's is
    #: about a half over a half: every bit has a pulse in one of its two
    #: chips). Real replies, alone or overlapped, measured no lower than
    #: 0.37 and 0.33.
    BALANCE = 0.2
    DATA = 0.2
    #: Read each reply's bits quickly at its fitted start time first, and
    #: search (Viterbi, every start time) only those whose frame doesn't
    #: then check itself.
    HARD_PASS = True
    #: The budget per second of signal: preamble fits, and timing
    #: hypotheses searched by the Viterbi search. A busy sky (2,000 Mode S
    #: and 6,000 Mode A/C replies a second) needs a third of it at 2.0 and
    #: 2.4 MS/s, two thirds at 2.048 MS/s; 15,000 mixed replies a second
    #: fit in it, 25,000 read 2 % fewer replies than without it. Unused
    #: budget is kept for up to ``BUDGET_S`` of signal, for bursts.
    FITS_PER_S = 20_000.0
    SEARCHES_PER_S = 10_000.0
    BUDGET_S = 0.05
    _MARGIN = 4  # samples kept before the next start position
    _LONGEST_CLUSTER = 64  # screen positions in a row worth waiting for

    def __init__(self, sample_rate: float):
        if not MIN_RATE <= sample_rate < MAX_RATE:
            raise ValueError(
                f"ModeSDemodulator reads 2 to 2.5 MS/s, got {sample_rate:g} Hz"
            )
        self.sample_rate = float(sample_rate)
        r = self._r = self.sample_rate * CHIP_S  # samples per chip
        self._length = int(math.ceil(241 * r)) + 2  # a long reply's samples
        # Preamble screen regions (sample offsets from a position p, for a
        # reply starting anywhere in [p, p + 1)): pulse chips 0-2 and 7-9,
        # and the samples certainly inside quiet chips 3-6 and 10-15.
        pulses = [
            (0, int(math.ceil(1 + 3 * r))),
            (int(math.floor(7 * r)), int(math.ceil(1 + 10 * r))),
        ]
        quiet = [
            (int(math.ceil(1 + 3 * r)), int(math.floor(7 * r))),
            (int(math.ceil(1 + 10 * r)), int(math.floor(16 * r))),
        ]
        self._pulses, self._quiet = pulses, quiet
        self._n_pulse = sum(b - a for a, b in pulses)
        self._n_quiet = sum(b - a for a, b in quiet)
        self._widths = sorted({b - a for a, b in pulses + quiet})
        # The samples certainly inside the first 56 bits for a reply the fit
        # can put anywhere from 1.5 samples before a piece's first screen
        # position to 1.5 after its last (at most 3 later), as offsets from
        # the first: whole windows of the screen's widest sums.
        wide = self._widths[-1]
        data_a = int(math.ceil(4.5 + 16 * r))
        data_b = int(math.floor(-1.5 + 128 * r))
        self._data = data_a + wide * np.arange((data_b - data_a) // wide)
        self._plan_fit()
        self._plan_bits()
        self.searched = 0  # timing hypotheses searched (work done)
        self.fitted = 0  # preamble fits (work done)
        self.skipped = 0  # preambles neither fitted nor searched (budget)
        self.reset()

    def reset(self) -> None:
        """Forget the stream (start a new one)."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos0 = 0  # stream index of _buf[0]
        self._free = 0  # stream index before which replies are taken
        # The budget, full.
        self._fit_budget = self.FITS_PER_S * self.BUDGET_S
        self._search_budget = self.SEARCHES_PER_S * self.BUDGET_S

    # -- Plans ----------------------------------------------------------

    def _chips_at(self, start: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
        """For replies starting at ``start`` (samples after sample 0): each
        of the first ``count`` samples' first chip and the share of the
        sample it covers (the rest is the next chip's)."""
        r = self._r
        u = (np.arange(count) - np.asarray(start, dtype=np.float64)[..., None]) / r
        first = np.floor(u).astype(np.int64)
        return first, np.minimum((first + 1 - u) * r, 1.0)

    def _plan_fit(self) -> None:
        """Preamble templates at ``FIT_STEPS`` start times per sample, from
        1.5 samples before a screen cluster's first position to 4.5 after
        (each fit keeps the start times its cluster allows)."""
        steps = self.FIT_STEPS
        self._fit_offsets = -1.5 + np.arange(6 * steps) / steps
        self._fit_len = int(math.ceil(17 * self._r)) + 10
        # The window starts 4 samples before the cluster.
        first, share = self._chips_at(4.0 + self._fit_offsets, self._fit_len)

        def chip(k: np.ndarray) -> np.ndarray:
            inside = (k >= 0) & (k < 16)
            return np.where(inside, _PREAMBLE[np.clip(k, 0, 15)], 0.0)

        template = share * chip(first) + (1.0 - share) * chip(first + 1)
        # Samples whose chips are both known: 3 quiet chips before the
        # reply to the end of the preamble.
        mask = ((first >= -3) & (first + 1 < 16)).astype(np.float64)
        self._fit_mask = mask
        self._fit_tm = template * mask
        self._fit_count = mask.sum(axis=1)
        self._fit_sum = self._fit_tm.sum(axis=1)
        self._fit_var = (self._fit_tm * template).sum(axis=1) - (
            self._fit_sum**2 / self._fit_count
        )

    def _plan_bits(self) -> None:
        """Tables of which samples and weights make each bit's metrics, for
        ``PHASES`` fractional start times (see :meth:`_search`).

        With ``y2 = 2 (y - floor) / level`` and ``w`` the share of a sample
        in its first chip, every metric is a sum of ``coeff * y2 + const``
        terms over a few samples:

        - a bit's own samples (first chip ``16 + 2n``: the bit, then its
          complement): ``dU[n]``, the cost of a 1 over a 0;
        - the samples across bits ``n`` and ``n + 1`` (first chip
          ``17 + 2n``): ``a[n]`` and ``b[n]``, what ending ``n`` in a 1
          costs over a 0 when ``n + 1`` is 1 or 0, and ``c[n]``;
        - the sample across the end of a short reply (first chip 127).
        """
        phases, length = self.PHASES, self._length
        own_k, pair_k = 4, 2
        own_j = np.zeros((phases, LONG_BITS, own_k), dtype=np.int64)
        own_c = np.zeros((phases, LONG_BITS, own_k), dtype=np.float32)
        own_0 = np.zeros((phases, LONG_BITS))
        pair_j = np.zeros((phases, LONG_BITS, pair_k), dtype=np.int64)
        pair_a = np.zeros((phases, LONG_BITS, pair_k), dtype=np.float32)
        pair_b = np.zeros((phases, LONG_BITS, pair_k), dtype=np.float32)
        pair_c = np.zeros((phases, LONG_BITS, pair_k), dtype=np.float32)
        pair_a0 = np.zeros((phases, LONG_BITS))
        pair_b0 = np.zeros((phases, LONG_BITS))
        pair_c0 = np.zeros((phases, LONG_BITS))
        end_j = np.zeros((phases, 2), dtype=np.int64)
        end_c = np.zeros((phases, 2), dtype=np.float32)
        end_0 = np.zeros(phases)
        last_pair = 17 + 2 * (LONG_BITS - 1)  # first chip of the last bit's
        firsts, shares = self._chips_at((np.arange(phases) + 0.5) / phases, length)
        for g in range(phases):
            own_used = np.zeros(LONG_BITS, dtype=np.int64)
            pair_used = np.zeros(LONG_BITS, dtype=np.int64)
            end_used = 0
            pairs = zip(firsts[g].tolist(), shares[g].tolist(), strict=True)
            for j, (k, w) in enumerate(pairs):
                v = 1.0 - w
                if 16 <= k < 16 + 2 * LONG_BITS and not (k - 16) % 2:
                    n, coeff, const = (k - 16) // 2, 1.0 - 2.0 * w, 2.0 * w - 1.0
                elif k == 15:  # (quiet, bit 0)
                    n, coeff, const = 0, -v, v * v
                elif k == last_pair:  # (last bit's complement, quiet)
                    n, coeff, const = LONG_BITS - 1, w, -w * w
                else:
                    n = -1
                if n >= 0:
                    slot = own_used[n]
                    own_j[g, n, slot], own_c[g, n, slot] = j, coeff
                    own_0[g, n] += const
                    own_used[n] += 1
                if 17 <= k < last_pair and not (k - 17) % 2:
                    n = (k - 17) // 2
                    slot = pair_used[n]
                    pair_j[g, n, slot] = j
                    pair_a[g, n, slot], pair_a0[g, n] = -w, pair_a0[g, n] + w * (2 - w)
                    pair_b[g, n, slot], pair_b0[g, n] = -w, pair_b0[g, n] + w * w
                    pair_c[g, n, slot], pair_c0[g, n] = -v, pair_c0[g, n] + v * v
                    pair_used[n] += 1
                if k == 17 + 2 * (SHORT_BITS - 1):
                    end_j[g, end_used], end_c[g, end_used] = j, w
                    end_0[g] -= w * w
                    end_used += 1
        self._own = (own_j, own_c, own_0)
        self._pair = (pair_j, pair_a, pair_b, pair_c, pair_a0, pair_b0, pair_c0)
        self._end = (end_j, end_c, end_0)

    # -- Stream ---------------------------------------------------------

    def feed(
        self,
        mag: np.ndarray,
        accept: Callable[[Sequence[bytes]], int],
        known: Callable[[], Collection[int]] = frozenset,
    ) -> None:
        """Scan the stream's next magnitude samples.

        Args:
            mag: Magnitudes (float32; negative values count as their size).
            accept: Called for each reply found, in stream order, with its
                frames (14 bytes; a short reply's last 7 are zero) that
                pass the CRC, the more likely first; returns the bits of
                the one it took (56 or 112), or 0. Replies starting inside
                a taken one are skipped.
            known: The aircraft addresses an address/parity reply (DF0,
                4, 5, 16, 20, 21) may carry: only frames whose CRC gives
                one of them, or the address of a checked reply in the same
                block, reach ``accept``.
        """
        mag = np.abs(np.asarray(mag, dtype=np.float32).ravel())
        for start in range(0, len(mag), self.PASS):
            self._scan(mag[start : start + self.PASS], accept, known)

    def _scan(
        self,
        mag: np.ndarray,
        accept: Callable[[Sequence[bytes]], int],
        known: Callable[[], Collection[int]],
    ) -> None:
        margin = self._MARGIN
        if len(self._buf):
            buf = np.concatenate((self._buf, mag))
        else:
            # Quiet samples before the stream, so a reply at sample 0 is
            # found.
            buf = np.concatenate((np.zeros(margin, dtype=np.float32), mag))
            self._pos0 -= margin
        end = len(buf) - self._length - 8  # positions with a whole reply after
        if end <= margin + 4:
            self._buf = buf
            return
        positions, stats = self._screen(buf, end)
        first = last = positions[:0]
        if len(positions):
            new = np.diff(positions, prepend=-10) > 2
            firsts = np.flatnonzero(new)
            lasts = np.append(firsts[1:], len(positions)) - 1
            first, last = positions[firsts], positions[lasts]
            if last[-1] >= end - 3 and end - first[-1] <= self._LONGEST_CLUSTER:
                # A cluster that may go on past the end: next time (unless
                # it is longer than any preamble's, which would hold the
                # stream back for ever).
                end = int(first[-1])
                first, last = first[:-1], last[:-1]
        # The budget grows with the signal scanned; what is left over is
        # kept for up to BUDGET_S of signal (or this pass's, if longer).
        seconds = (end - margin) / self.sample_rate
        keep = max(seconds, self.BUDGET_S)
        self._fit_budget = min(
            self._fit_budget + seconds * self.FITS_PER_S, keep * self.FITS_PER_S
        )
        self._search_budget = min(
            self._search_budget + seconds * self.SEARCHES_PER_S,
            keep * self.SEARCHES_PER_S,
        )
        if len(first):
            self._replies(buf, first, last, accept, known, positions, stats)
        self._buf = buf[end - margin :].copy()
        self._pos0 += end - margin

    def _screen(
        self, buf: np.ndarray, end: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """Positions (``margin`` to ``end``) where a preamble may start:
        the pulse samples' mean is ``SCREEN`` times the quiet samples'.

        Also, for :meth:`_promising`: at each position the sums of the
        two preamble halves' samples and of the quiet samples, and the
        running sums of the widest region's width over ``buf``."""
        margin = self._MARGIN
        sums = {1: buf}
        running = buf
        for width in range(2, self._widths[-1] + 1):
            running = running[:-1] + buf[width - 1 :]
            sums[width] = running

        def part(a: int, b: int) -> np.ndarray:
            return sums[b - a][margin + a : end + a]

        (a1, b1), (a2, b2) = self._pulses
        half1, half2 = part(a1, b1), part(a2, b2)
        pulses = half1 + half2
        quiet = None
        for a, b in self._quiet:
            total = part(a, b)
            quiet = total.copy() if quiet is None else quiet + total
        assert quiet is not None
        ok = pulses * np.float32(self._n_quiet) > (
            np.float32(self.SCREEN * self._n_pulse) * quiet
        )
        index = np.flatnonzero(ok)
        stats = (half1[index], half2[index], quiet[index], sums[self._widths[-1]])
        return index + margin, stats

    def _replies(
        self,
        buf: np.ndarray,
        first: np.ndarray,
        last: np.ndarray,
        accept: Callable[[Sequence[bytes]], int],
        known: Callable[[], Collection[int]],
        positions: np.ndarray,
        stats: Tuple[np.ndarray, ...],
    ) -> None:
        """Fit the clusters of screen positions ``first..last`` (in pieces
        of up to four positions) that pass the cheap tests, as many as the
        budget allows (the clearest first), read the bits of those that fit
        and hand them to ``accept``."""
        # Long clusters in pieces, each fitted on its own.
        span = last - first + 1
        pieces = (span + 3) // 4
        piece_first = np.repeat(first, pieces) + 4 * (
            np.arange(pieces.sum()) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        )
        piece_last = np.minimum(piece_first + 3, np.repeat(last, pieces))
        keep = piece_first + self._pos0 >= self._free - 2
        piece_first, piece_last = piece_first[keep], piece_last[keep]
        if not len(piece_first):
            return
        good, clarity = self._promising(piece_first, piece_last, positions, stats)
        piece_first, piece_last, clarity = (
            piece_first[good],
            piece_last[good],
            clarity[good],
        )
        if len(piece_first) > self._fit_budget:
            allowed = max(0, int(self._fit_budget))
            self.skipped += len(piece_first) - allowed
            chosen = np.sort(np.argsort(-clarity, kind="stable")[:allowed])
            piece_first, piece_last = piece_first[chosen], piece_last[chosen]
        if not len(piece_first):
            return
        self._fit_budget -= len(piece_first)
        self.fitted += len(piece_first)
        start, level, floor = self._fit(buf, piece_first, piece_last)
        if not len(start):
            return
        frames, passes, first_sample = self._frames(buf, start, level, floor, known)
        r = self._r
        for i in np.argsort(start, kind="stable").tolist():
            if not passes[i].any():
                continue
            at = int(first_sample[i]) + self._pos0
            if at < self._free:
                continue  # inside a reply already taken
            bits = accept([frames[i, h].tobytes() for h in np.flatnonzero(passes[i])])
            if bits:
                self._free = at + int(math.ceil((16 + 2 * bits) * r))

    def _promising(
        self,
        piece_first: np.ndarray,
        piece_last: np.ndarray,
        positions: np.ndarray,
        stats: Tuple[np.ndarray, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Which pieces may be a reply's preamble, and how clear each is.

        At each piece's clearest screen position (most pulse level over
        the quiet samples): both halves of the preamble rise above the
        quiet samples alike (``BALANCE``), and so do the first 56 bits
        (``DATA``). Clarity: the pulse level over the quiet level."""
        half1, half2, quiet, wide = stats
        (a1, b1), (a2, b2) = self._pulses
        floor_at = quiet.astype(np.float64) / self._n_quiet
        excess_at = (half1.astype(np.float64) + half2) / self._n_pulse - floor_at
        # The screen positions in each piece: lo..hi-1 (one at least).
        lo = np.searchsorted(positions, piece_first)
        hi = np.searchsorted(positions, piece_last, side="right")
        best = lo
        for k in range(1, 4):
            at = np.minimum(lo + k, len(positions) - 1)
            best = np.where((lo + k < hi) & (excess_at[at] > excess_at[best]), at, best)
        floor, excess = floor_at[best], excess_at[best]
        upper = half1[best] / (b1 - a1) - floor
        lower = half2[best] / (b2 - a2) - floor
        balanced = np.minimum(upper, lower) > self.BALANCE * np.maximum(upper, lower)
        data = wide[piece_first[:, None] + self._data].sum(axis=1, dtype=np.float64)
        data /= self._data.size * self._widths[-1]
        busy = data - floor > self.DATA * excess
        clarity = excess / np.maximum(floor, 1e-30)
        return balanced & busy, clarity

    def _frames(
        self,
        buf: np.ndarray,
        start: np.ndarray,
        level: np.ndarray,
        floor: np.ndarray,
        known: Callable[[], Collection[int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each reply's frame at each of the ``OFFSETS`` start times
        (replies x offsets x 14 bytes), which of them pass
        :meth:`_checked`, and each reply's first sample.

        A reply whose quick reading (:meth:`_hard`) at the fitted start
        gives a frame that checks itself (see :meth:`_settled`) is that
        frame. The others are searched as before: at the fitted start time,
        and at the other start times only those whose frame there doesn't
        check itself, in a second pass when there are many (one pass costs
        less for a few); as many as the search budget allows, the clearest
        first. The frames that reach ``accept`` are the ones a search of
        every reply would offer, and a few more: replies the quick reading
        reads where the search couldn't."""
        offsets = np.asarray(self.OFFSETS)
        replies, count = len(start), len(offsets)
        frames = np.zeros((replies, count, 14), dtype=np.uint8)
        made = np.zeros((replies, count), dtype=bool)  # frames read
        first = np.floor(start).astype(np.int64)
        settled = np.zeros(replies, dtype=bool)
        if self.HARD_PASS:
            quick = self._hard(buf, start, level, floor)
            settled = self._settled(quick, known)
            frames[settled, 0] = quick[settled]
            made[settled, 0] = True
        rest = np.flatnonzero(~settled)
        if len(rest) <= self.ONE_PASS:
            steps = [slice(0, count)]
        else:
            steps = [slice(0, 1), slice(1, count)]
        for at in steps:
            step = offsets[at]
            rest = self._affordable(rest, len(step), level, floor)
            if not len(rest):
                break
            found, _first = self._search(
                buf,
                (start[rest, None] + step).ravel(),
                np.repeat(level[rest], len(step)),
                np.repeat(floor[rest], len(step)),
            )
            self._search_budget -= len(rest) * len(step)
            frames[rest, at] = found.reshape(len(rest), len(step), 14)
            made[rest, at] = True
            if at.start == 0:  # (the fitted start time read)
                settled[rest] = self._settled(frames[rest, 0], known)
                rest = rest[~settled[rest]]
        passes = self._checked(frames.reshape(-1, 14), known).reshape(replies, count)
        # Only frames read; a settled reply is its first frame.
        passes &= made
        passes[settled, 1:] = False
        return frames, passes, first

    def _affordable(
        self, rest: np.ndarray, each: int, level: np.ndarray, floor: np.ndarray
    ) -> np.ndarray:
        """The replies among ``rest`` (indices) the search budget lets
        search at ``each`` start times, the clearest first; the others are
        counted in ``skipped``."""
        if len(rest) * each <= self._search_budget:
            return rest
        allowed = max(0, int(self._search_budget // each))
        self.skipped += len(rest) - allowed
        clarity = level[rest] / np.maximum(floor[rest], 1e-30)
        return np.sort(rest[np.argsort(-clarity, kind="stable")[:allowed]])

    def _hard(
        self, buf: np.ndarray, start: np.ndarray, level: np.ndarray, floor: np.ndarray
    ) -> np.ndarray:
        """Frames (rows of 14 bytes; a short reply's last 7 zero) read a
        bit at a time at ``start``: from the bit's own samples, weighed as
        the search weighs them, and from the samples it shares with the
        bits before and after it, the other bit's chip taken as half lit.
        Cheap (no search); right for most strong replies, at 2.4 MS/s most
        of all."""
        first = np.floor(start).astype(np.int64)
        phase = np.minimum(
            ((start - first) * self.PHASES).astype(np.int64), self.PHASES - 1
        )
        scale = (2.0 / level).astype(np.float32)[:, None, None]
        offset = floor.astype(np.float32)[:, None, None]
        own_j, own_c, own_0 = self._own
        pair_j, pair_a, _pair_b, pair_c = self._pair[:4]
        at = first[:, None, None]
        y_own = (buf[own_j[phase] + at] - offset) * scale
        # A shared sample with w of it in a bit's second chip and v in the
        # next bit's first, y2 scaled as the search scales it: it weighs
        # w (y2 - 1) for a 1 over a 0 in the bit, -v (y2 - 1) in the next
        # (pair_a is -w, pair_c is -v; slots not used are 0).
        y_pair = (buf[pair_j[phase] + at] - offset) * scale - np.float32(1.0)
        cost = np.einsum("rnk,rnk->rn", y_own, own_c[phase]) + own_0[phase]
        cost -= np.einsum("rnk,rnk->rn", y_pair, pair_a[phase])
        cost[:, 1:] += np.einsum("rnk,rnk->rn", y_pair, pair_c[phase])[:, :-1]
        frames = np.packbits(cost < 0, axis=1)
        frames[_SHORT[frames[:, 0] >> 3], 7:] = 0
        return frames

    @staticmethod
    def _settled(
        frames: np.ndarray, known: Callable[[], Collection[int]]
    ) -> np.ndarray:
        """Frames that check themselves: a DF17/18 or a DF11 whose CRC
        residual is 0, or an address/parity reply whose residual is an
        address in ``known`` or that such a frame among ``frames`` carries
        (any decoder takes them as they are)."""
        df = frames[:, 0] >> 3
        short, long = crc_residuals(frames)
        residual = np.where(_SHORT[df], short, long)
        settled = (_SQUITTER[df] | (df == 11)) & (residual == 0)
        parity = (_SHORT[df] | _LONG[df]) & ~_SQUITTER[df] & (df != 11)
        if parity.any():
            addresses = _addresses(frames[settled], known())
            if len(addresses):
                settled |= parity & np.isin(residual, addresses)
        return settled

    def _fit(
        self, buf: np.ndarray, first: np.ndarray, last: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start time (samples into ``buf``), pulse level and noise floor of
        the clusters that look like a reply's preamble."""
        base = first - 4
        window = buf[base[:, None] + np.arange(self._fit_len)].astype(np.float64)
        # (einsum, not matmul: BLAS would spread these small products over
        # threads and cost more.)
        sum_y = np.einsum("cj,tj->ct", window, self._fit_mask)
        sum_yy = np.einsum("cj,tj->ct", window * window, self._fit_mask)
        sum_ty = np.einsum("cj,tj->ct", window, self._fit_tm)
        count = self._fit_count
        cov = sum_ty - sum_y * self._fit_sum / count
        var_y = sum_yy - sum_y * sum_y / count
        corr = cov / np.sqrt(np.maximum(self._fit_var * var_y, 1e-30))
        # Start times from 1.5 samples before the cluster to 1.5 after it.
        corr[self._fit_offsets[None, :] > (last - first)[:, None] + 1.5] = -2.0
        best = np.argmax(corr, axis=1)
        rows = np.arange(len(first))
        level = cov[rows, best] / self._fit_var[best]
        floor = (sum_y[rows, best] - level * self._fit_sum[best]) / count[best]
        good = (
            (corr[rows, best] > self.FIT)
            & (level > 0)
            & (level > self.LEVEL * np.maximum(floor, 0.0))
        )
        start = base + 4.0 + self._fit_offsets[best]
        return start[good], level[good], floor[good]

    def _search(
        self, buf: np.ndarray, start: np.ndarray, level: np.ndarray, floor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The most likely frames (rows of 14 bytes) for replies starting at
        ``start`` (samples into ``buf``) and each one's first sample."""
        rows = len(start)
        self.searched += rows
        first = np.floor(start).astype(np.int64)
        phase = np.minimum(
            ((start - first) * self.PHASES).astype(np.int64), self.PHASES - 1
        )
        scale = (2.0 / level).astype(np.float32)[:, None, None]
        offset = floor.astype(np.float32)[:, None, None]
        own_j, own_c, own_0 = self._own
        pair_j, pair_a, pair_b, pair_c, pair_a0, pair_b0, pair_c0 = self._pair
        end_j, end_c, end_0 = self._end
        at = first[:, None, None]
        y_own = (buf[own_j[phase] + at] - offset) * scale
        y_pair = (buf[pair_j[phase] + at] - offset) * scale
        y_end = (buf[end_j[phase] + first[:, None]] - offset[:, :, 0]) * scale[:, :, 0]
        d_own = np.einsum("rnk,rnk->nr", y_own, own_c[phase]) + own_0[phase].T
        a = np.einsum("rnk,rnk->nr", y_pair, pair_a[phase]) + pair_a0[phase].T
        b = np.einsum("rnk,rnk->nr", y_pair, pair_b[phase]) + pair_b0[phase].T
        c = np.einsum("rnk,rnk->nr", y_pair, pair_c[phase]) + pair_c0[phase].T
        end = (y_end * end_c[phase]).sum(axis=1) + end_0[phase]
        return self._viterbi(d_own, a, b, c, end), first

    @staticmethod
    def _viterbi(
        d_own: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        """Two-state Viterbi on ``D = cost(1) - cost(0)``: ``D[n] =
        min(a, D) - min(b, D) + c + d_own[n]`` (``a``, ``b``, ``c`` of the
        bits before); then the short replies' search again from bit 55."""
        rows = d_own.shape[1]
        inc = c[:-1] + d_own[1:]
        diff = np.empty((LONG_BITS, rows))
        diff[0] = d_own[0]
        low_a = np.empty(rows)
        low_b = np.empty(rows)
        for n in range(1, LONG_BITS):
            prev = diff[n - 1]
            np.minimum(a[n - 1], prev, out=low_a)
            np.minimum(b[n - 1], prev, out=low_b)
            out = diff[n]
            np.subtract(low_a, low_b, out=out)
            out += inc[n - 1]
        from1 = diff[:-1] < a[:-1]  # state 1 came from state 1
        from0 = diff[:-1] < b[:-1]  # state 0 came from state 1
        bits = _traceback(from0, from1, diff[-1] < 0)
        frames = np.packbits(bits.T, axis=1)
        short = _SHORT[frames[:, 0] >> 3]
        if short.any():
            last = SHORT_BITS - 1
            bits = _traceback(
                from0[:last, short],
                from1[:last, short],
                (diff[last, short] + end[short]) < 0,
            )
            frames[short, :7] = np.packbits(bits.T, axis=1)
            frames[short, 7:] = 0
        return frames

    def _checked(
        self, frames: np.ndarray, known: Callable[[], Collection[int]]
    ) -> np.ndarray:
        """Which frames may be replies: a DF17/18 with a CRC residual of 0,
        a DF11 with an interrogator code at most, an address/parity reply
        whose residual is an address in ``known`` or of a checked reply
        among ``frames``."""
        df = frames[:, 0] >> 3
        short, long = crc_residuals(frames)
        residual = np.where(_SHORT[df], short, long)
        checked = (_SQUITTER[df] & (residual == 0)) | (
            (df == 11) & ((residual & ~0x7F) == 0)
        )
        parity = (_SHORT[df] | _LONG[df]) & ~_SQUITTER[df] & (df != 11)
        if parity.any():
            addresses = _addresses(frames[checked], known())
            if len(addresses):
                checked |= parity & np.isin(residual, addresses)
        return checked


def _addresses(frames: np.ndarray, known: Iterable[int]) -> np.ndarray:
    """``known`` and the addresses the checked ``frames`` carry (int64)."""
    rows = frames[:, 1:4].astype(np.int64)
    carried = (rows[:, 0] << 16) | (rows[:, 1] << 8) | rows[:, 2]
    return np.concatenate((np.fromiter(known, dtype=np.int64), carried))
