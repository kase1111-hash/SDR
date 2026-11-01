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
2. A least-squares fit of the preamble, as an integrating ADC samples it,
   at a grid of start times a fraction of a sample apart: the reply's start
   time, pulse level and noise floor.
3. A Viterbi search for the bits: each sample's expected value is the
   pulse level times the share of the sample each of its two chips covers,
   so the bits are the sequence whose samples fit best (two states, the
   last bit). A short reply's search ends after bit 55. It runs at five
   start times around the fitted one, the fitted one first.
4. The CRC and format checks of the decoder that owns it (``accept``).

The fit and the search are vectorised across replies: a block's replies
are searched together, whatever their number.
"""

import math
from typing import Callable, Collection, Dict, Iterable, List, Sequence, Tuple

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
    don't depend on the block sizes.

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
    #: Up to this many replies in a pass, all start times are searched at
    #: once (see :meth:`_frames`).
    ONE_PASS = 6
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
        self._plan_fit()
        self._plan_bits()
        self.searched = 0  # timing hypotheses searched (work done)
        self.reset()

    def reset(self) -> None:
        """Forget the stream (start a new one)."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos0 = 0  # stream index of _buf[0]
        self._free = 0  # stream index before which replies are taken

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
        positions = self._screen(buf, end)
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
            if len(first):
                self._replies(buf, first, last, accept, known)
        self._buf = buf[end - margin :].copy()
        self._pos0 += end - margin

    def _screen(self, buf: np.ndarray, end: int) -> np.ndarray:
        """Positions (``margin`` to ``end``) where a preamble may start:
        the pulse samples' mean is ``SCREEN`` times the quiet samples'."""
        margin = self._MARGIN
        sums = {1: buf}
        running = buf
        for width in range(2, self._widths[-1] + 1):
            running = running[:-1] + buf[width - 1 :]
            sums[width] = running

        def region(parts: List[Tuple[int, int]]) -> np.ndarray:
            total = None
            for a, b in parts:
                part = sums[b - a][margin + a : end + a]
                total = part.copy() if total is None else total + part
            assert total is not None
            return total

        pulses, quiet = region(self._pulses), region(self._quiet)
        ok = pulses * np.float32(self._n_quiet) > (
            np.float32(self.SCREEN * self._n_pulse) * quiet
        )
        return np.flatnonzero(ok) + margin

    def _replies(
        self,
        buf: np.ndarray,
        first: np.ndarray,
        last: np.ndarray,
        accept: Callable[[Sequence[bytes]], int],
        known: Callable[[], Collection[int]],
    ) -> None:
        """Fit the clusters of screen positions ``first..last`` (in pieces
        of up to four positions), search the bits of those that fit and
        hand them to ``accept``."""
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

        A reply whose fitted start gives a frame that checks itself (a
        DF17/18 or a DF11 with a CRC residual of 0) needs no other start
        time: those are searched only for the others, in a second pass
        when there are many replies (one pass costs less for a few). The
        frames that reach ``accept`` are the same either way."""
        offsets = np.asarray(self.OFFSETS)
        replies, count = len(start), len(offsets)
        frames = np.zeros((replies, count, 14), dtype=np.uint8)
        if replies <= self.ONE_PASS:
            starts = (start[:, None] + offsets).ravel()
            found, first = self._search(
                buf, starts, np.repeat(level, count), np.repeat(floor, count)
            )
            frames[:] = found.reshape(replies, count, 14)
            settled = self._settled(frames[:, 0])
            first = first[::count]
        else:
            frames[:, 0], first = self._search(buf, start, level, floor)
            settled = self._settled(frames[:, 0])
            rest = np.flatnonzero(~settled)
            if len(rest):
                starts = (start[rest, None] + offsets[1:]).ravel()
                found, _first = self._search(
                    buf,
                    starts,
                    np.repeat(level[rest], count - 1),
                    np.repeat(floor[rest], count - 1),
                )
                frames[rest, 1:] = found.reshape(len(rest), count - 1, 14)
        passes = self._checked(frames.reshape(-1, 14), known).reshape(replies, count)
        # A settled reply is its first frame (the others weren't searched
        # in two passes, and aren't looked at in one).
        passes[settled, 1:] = False
        return frames, passes, first

    @staticmethod
    def _settled(frames: np.ndarray) -> np.ndarray:
        """Frames that check themselves: a DF17/18 or a DF11 whose CRC
        residual is 0 (any decoder takes them as they are)."""
        df = frames[:, 0] >> 3
        short, long = crc_residuals(frames)
        return (_SQUITTER[df] & (long == 0)) | ((df == 11) & (short == 0))

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
            if addresses:
                ok = np.fromiter(addresses, dtype=np.int64, count=len(addresses))
                checked |= parity & np.isin(residual, ok)
        return checked


def _addresses(frames: np.ndarray, known: Iterable[int]) -> set:
    """``known`` and the addresses the checked ``frames`` carry."""
    found = set(int(a) for a in known)
    for frame in frames:
        found.add(int.from_bytes(bytes(frame[1:4]), "big"))
    return found
