"""K-D4: FLEX at 3200 and 6400 bps decodes.

FLEX frames start with a 1600 bps 2-level header whose sync code names the
speed of the rest: 1600 bps 2-level (one phase, A), 3200 bps 2-level (3200
symbols/s; phases A and C on alternate symbols), 3200 bps 4-level (1600
symbols/s; a symbol carries a bit of phases A and B, Gray coded) and 6400
bps 4-level (3200 symbols/s; A and B, then C and D). Before, only 1600 bps
2-level decoded and the others were reported as "not supported yet". The
frames here are built by ``protocol_encoders.flex_frame_levels`` from the
same format (ARIB STD-43A, as multimon-ng reads it).
"""

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import FLEXLiveDecoder
from sdr_module.dsp.protocols import FLEXDecoder

SPEEDS = [(1600, 2), (3200, 2), (3200, 4), (6400, 4)]
RATE = 48000.0


def _pages(speed):
    """Different pages in every phase the speed carries."""
    phases = enc.FLEX_SPEEDS[speed][2]
    return {
        p: [(1000000 + 10 * i + ord(p), f"Phase {p} page {i} at {speed[0]} bps")]
        for i, p in enumerate(phases)
    }


def _discriminator(pages, speed, cycle=3, frame=17, invert=False, rate=RATE):
    """What an FM discriminator gives for the frame: its frequency (Hz)."""
    f = enc.flex_frequency(pages, rate, cycle, frame, speed=speed)
    return -f if invert else f


def _decode(decoder, x, block):
    messages = []
    for start in range(0, len(x), block):
        messages += decoder.decode(x[start : start + block])
    return messages


def _rows(messages):
    return [(m.capcode, m.content, m.phase, m.baud_rate, m.valid) for m in messages]


def _expected(pages, speed):
    return [
        (capcode, text, p, speed[0], True)
        for p in enc.FLEX_SPEEDS[speed][2]
        for capcode, text in pages[p]
    ]


@pytest.mark.parametrize("speed", SPEEDS)
def test_every_phase_decodes_at_every_speed(speed):
    pages = _pages(speed)
    x = np.concatenate([_discriminator(pages, speed), np.zeros(4800)])
    decoder = FLEXDecoder(RATE)
    messages = _decode(decoder, x, 1601)
    assert _rows(messages) == _expected(pages, speed)
    assert {(m.cycle, m.frame) for m in messages} == {(3, 17)}
    assert (decoder.frames_seen, decoder.frames_failed) == (1, 0)
    assert decoder.unsupported_frames == 0


@pytest.mark.parametrize("speed", SPEEDS)
def test_either_polarity(speed):
    pages = _pages(speed)
    x = np.concatenate([_discriminator(pages, speed, invert=True), np.zeros(4800)])
    assert _rows(_decode(FLEXDecoder(RATE), x, 2000)) == _expected(pages, speed)


@pytest.mark.parametrize("speed", [(3200, 2), (6400, 4)])
def test_blocks_of_any_size_give_the_same_pages(speed):
    pages = _pages(speed)
    x = np.concatenate([_discriminator(pages, speed), np.zeros(4800)])
    rows = [_rows(_decode(FLEXDecoder(RATE), x, block)) for block in (len(x), 997, 37)]
    assert rows[0] == _expected(pages, speed)
    assert rows[1] == rows[0] and rows[2] == rows[0]


def test_frames_at_different_speeds_one_after_another():
    # The decoder goes back to the 1600 bps header after each frame,
    # whatever speed the frame before used.
    x, expected = [], []
    for frame, speed in enumerate([(6400, 4), (1600, 2), (3200, 2), (3200, 4)]):
        pages = _pages(speed)
        x.append(_discriminator(pages, speed, frame=frame))
        expected += [row + (frame,) for row in _expected(pages, speed)]
    stream = np.concatenate(x + [np.zeros(4800)])
    messages = _decode(FLEXDecoder(RATE), stream, 4000)
    rows = [r + (m.frame,) for r, m in zip(_rows(messages), messages, strict=True)]
    assert rows == expected


@pytest.mark.parametrize("speed", SPEEDS)
def test_on_the_air_through_the_live_decoder(speed):
    # FM at 240 kS/s, 1.5 kHz off tune, with noise, in display-frame blocks.
    rate = 240e3
    pages = _pages(speed)
    rng = np.random.default_rng(int(speed[0] + speed[1]))
    x = enc.flex_baseband(pages, rate, 2, 40, speed=speed)
    t = np.arange(len(x)) / rate
    x = x * np.exp(2j * np.pi * 1500.0 * t)
    x = np.concatenate([np.zeros(int(0.3 * rate)), x, np.zeros(int(0.3 * rate))])
    noise = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    x = (x + 0.05 * noise).astype(np.complex64)
    decoder = FLEXLiveDecoder(rate)
    messages = []
    block = int(rate * 0.034)
    for start in range(0, len(x), block):
        messages += decoder.process(x[start : start + block])
    assert _rows(messages) == _expected(pages, speed)
    assert decoder.status.problem is None


def _channel(frames, rate=240e3, offset_hz=1500.0, noise=0.05, seed=0):
    """FM at ``rate`` sending ``frames`` (their frequencies, Hz) back to
    back, as a paging transmitter does: one phase-continuous signal,
    ``offset_hz`` off tune, with noise."""
    rng = np.random.default_rng(seed)
    gap = np.zeros(int(0.3 * rate))
    x = enc.fm_modulate(np.concatenate([gap, *frames, gap]) + offset_hz, rate)
    x = x + noise * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64)


def _live(x, rate=240e3):
    decoder = FLEXLiveDecoder(rate)
    block = int(rate * 0.034)
    messages = []
    for start in range(0, len(x), block):
        messages += decoder.process(x[start : start + block])
    return decoder, messages


@pytest.mark.parametrize("speed", SPEEDS)
def test_frames_back_to_back_all_decode(speed):
    # A paging channel sends one frame after another without a break, and
    # a frame with few pages is mostly idle words (all ones): a long run
    # at one frequency right before the next frame's header, and 4-level
    # data whose average is far from the centre. Before, after the first
    # frame the next header was read off centre and missed (1 of 8 frames
    # back to back decoded at 1600 bps).
    rate = 240e3
    frames, expected = [], []
    for frame in range(5):
        pages = {} if frame == 2 else _pages(speed)  # frame 2 is idle
        frames.append(enc.flex_frequency(pages, rate, 3, frame, speed=speed))
        if pages:
            expected += [row + (frame,) for row in _expected(pages, speed)]
    decoder, messages = _live(_channel(frames))
    rows = [r + (m.frame,) for r, m in zip(_rows(messages), messages, strict=True)]
    assert rows == expected
    assert decoder._decoder.frames_seen == 5
    assert decoder._decoder.frames_failed == 0


def test_every_speed_back_to_back():
    rate = 240e3
    order = [(6400, 4), (1600, 2), (3200, 4), (6400, 4), (3200, 2), (1600, 2)]
    frames, expected = [], []
    for frame, speed in enumerate(order):
        pages = _pages(speed)
        frames.append(enc.flex_frequency(pages, rate, 3, frame, speed=speed))
        expected += [row + (frame,) for row in _expected(pages, speed)]
    _decoder, messages = _live(_channel(frames, offset_hz=-2000.0, seed=1))
    rows = [r + (m.frame,) for r, m in zip(_rows(messages), messages, strict=True)]
    assert rows == expected


@pytest.mark.parametrize("speed", [(1600, 2), (6400, 4)])
def test_a_block_information_word_failing_its_checksum_gives_no_pages(speed):
    # A BIW that passes its BCH check but not its own checksum is a word
    # with more errors than BCH corrects, "corrected" into another one:
    # the pages it points to would be other words read as addresses and
    # vectors. At 4-level speeds, whose data needs a stronger signal than
    # the 2-level header, such frames gave rows of made-up "valid" pages
    # near the threshold. The phase now gives none and the frame counts
    # as failed.
    phases = enc.FLEX_SPEEDS[speed][2]
    values = {p: enc.flex_phase_values([(1234567, f"PAGE {p}")]) for p in phases}
    values["A"][0] ^= 0x10  # the BIW's checksum no longer adds up
    x = enc.flex_frequency(values, RATE, 3, 17, speed=speed)
    decoder = FLEXDecoder(RATE)
    messages = _decode(decoder, np.concatenate([x, np.zeros(4800)]), 2000)
    assert [m.phase for m in messages] == list(phases[1:])
    assert decoder.frames_failed == 1


@pytest.mark.parametrize("page_type", ["alpha", "numeric"])
def test_a_page_whose_message_runs_off_the_phase_is_skipped(page_type):
    # A vector word pointing past the phase's 88 words is damaged (as
    # multimon-ng reads FLEX): the page was shown cut short or made up.
    values = enc.flex_phase_values([(1234567, "LAST WORDS")])
    first_vector = (values[0] >> 10) & 0x3F
    vector = values[first_vector]
    kind = 5 if page_type == "alpha" else 3
    values[first_vector] = (vector & 0x1FC00F) | (kind << 4) | (84 << 7)
    if page_type == "numeric":
        values[first_vector] |= 7 << 14  # 8 words from 84
    decoder = FLEXDecoder(RATE)
    x = np.concatenate([_discriminator({"A": values}, (1600, 2)), np.zeros(4800)])
    assert _decode(decoder, x, 2000) == []


def test_a_rate_too_low_for_3200_symbols_says_so():
    # Below 3 samples a symbol the 3200 symbols/s frames can't be read:
    # they are counted as not decoded, with their speed, as before.
    rate = 8000.0
    pages = _pages((6400, 4))
    x = np.concatenate([_discriminator(pages, (6400, 4), rate=rate), np.zeros(800)])
    decoder = FLEXDecoder(rate)
    assert _decode(decoder, x, 500) == []
    assert decoder.unsupported_frames == 1
    assert decoder.last_unsupported == "6400 bps 4-level"
    # 1600 symbols/s still decode at that rate.
    pages = _pages((3200, 4))
    x = np.concatenate([_discriminator(pages, (3200, 4), rate=rate), np.zeros(800)])
    assert _rows(_decode(decoder, x, 500)) == _expected(pages, (3200, 4))


def test_the_demo_pager_sends_every_speed():
    # The demo's FLEX channel (931.9375 MHz) sends a frame at each speed in
    # its loop; all of them decode.
    from unittest import mock

    from sdr_module.gui import device_dialog as dd

    period = dd._DATA_SCHEDULE["flex"][0]
    with mock.patch.object(dd, "_MAX_LAG_S", 1e9):
        dev = dd.MockDevice()
        dev.set_sample_rate(1.0e6)
        dev.set_frequency(931.9375e6)
        dev._rng = np.random.default_rng(4)
        dev.start_rx()
        decoder = FLEXLiveDecoder(dev.sample_rate)
        block = int(dev.sample_rate * 0.034)
        messages = []
        for _ in range(int((period + 1.0) / 0.034)):
            messages += decoder.process(dev.read_samples(block))
    speeds = {m.baud_rate for m in messages}
    assert speeds == {1600, 3200, 6400}
    assert all(m.valid for m in messages)
    assert {m.phase for m in messages} == {"A", "B", "C", "D"}
