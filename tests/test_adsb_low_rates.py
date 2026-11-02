"""K-D1: ADS-B at 2.0 to 2.5 MS/s, where a sample spans most of a chip.

Mode S sends 0.5 us chips; at 2 MS/s a sample is one chip long and at
2.048 MS/s slightly shorter, so most samples straddle two chips and their
value is a mix of both. Before, the decoder read the magnitude in 0.5 us
slots and compared the two halves of each bit: at exactly 2 MS/s a reply
that started about half a sample off the sample grid could not be told
apart (strong replies decoded 34-36 of 40), and at 2.048 MS/s every reply
drifts through that phase several times, so most failed their CRC (14-16
of 40). Now each reply's timing and level are fitted from its preamble and
its bits are found by a Viterbi search over the samples' chip mixtures
(``sdr_module.dsp.mode_s``); the replies are synthesised the way the demo
device's integrating ADC samples them.
"""

import math

import numpy as np
import pytest

from sdr_module.dsp.protocols import ADSBDecoder
from sdr_module.gui import device_dialog as dd

GUI_BLOCK_S = 0.034  # one display frame at 30 Hz
REPLY_GAP_US = 400.0


def _squitter(icao: int, rng: np.random.Generator) -> bytes:
    """A DF17 airborne-position-like squitter with a random payload."""
    payload = int(rng.integers(0, 1 << 51))
    return dd._extended_squitter(icao, (11 << 51) | payload)


def _stream(rate, frames, offsets, snr_db, rng):
    """Complex samples holding ``frames`` REPLY_GAP_US apart, each starting
    ``offsets[k]`` of a sample after the sample grid, with complex noise
    ``snr_db`` below the pulse power (in 2 MHz)."""
    n = int((len(frames) + 2) * REPLY_GAP_US * 1e-6 * rate)
    sigma2 = 10 ** (-snr_db / 10) * rate / 2e6
    x = np.sqrt(sigma2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for k, (frame, offset) in enumerate(zip(frames, offsets, strict=True)):
        start = math.floor((k + 1) * REPLY_GAP_US * 1e-6 * rate) + offset
        first, env = dd._pulse_envelope(dd._modes_pulses(frame), start, rate)
        x[first : first + len(env)] += env * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return np.abs(x).astype(np.float32)


def _decode(decoder, mag, block):
    messages = []
    for start in range(0, len(mag), block):
        messages += decoder.decode(mag[start : start + block])
    return messages


def _replies(rate, count, snr_db, seed):
    rng = np.random.default_rng(seed)
    icaos = [int(a) for a in rng.choice(np.arange(0x100000, 0xFFFFFF), count, False)]
    frames = [_squitter(icao, rng) for icao in icaos]
    offsets = (np.arange(count) + 0.5) / count  # every chip phase
    return icaos, frames, _stream(rate, frames, offsets, snr_db, rng)


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6, 2.1e6])
def test_strong_replies_decode_at_every_chip_phase(rate):
    icaos, frames, mag = _replies(rate, 40, 30.0, seed=int(rate) % 9973)
    rows = []
    for block in (len(mag), int(rate * GUI_BLOCK_S), 997):
        messages = _decode(ADSBDecoder(rate), mag, block)
        rows.append([(m.icao_address, m.raw_bits) for m in messages])
    expected = [(f"{a:06X}", f) for a, f in zip(icaos, frames, strict=True)]
    assert rows[0] == expected
    # Frames that straddle two blocks decode the same, once.
    assert rows[1] == rows[0] and rows[2] == rows[0]


@pytest.mark.parametrize("rate, least", [(2.0e6, 80), (2.048e6, 78)])
def test_most_replies_15_db_above_the_noise_decode(rate, least):
    # Before: 55 of 100 at 2.0 MS/s (none that started 0.4-0.6 of a sample
    # off the grid) and 4 of 100 at 2.048 MS/s.
    icaos, _frames, mag = _replies(rate, 100, 15.0, seed=7)
    messages = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    found = {m.icao_address for m in messages}
    assert len(found & {f"{icao:06X}" for icao in icaos}) >= least
    assert found <= {f"{icao:06X}" for icao in icaos}


@pytest.mark.parametrize("rate", [2.4e6, 2.56e6, 3.2e6])
def test_2_4_ms_s_and_above_do_not_get_worse(rate):
    icaos, _frames, mag = _replies(rate, 100, 15.0, seed=7)
    messages = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    found = {m.icao_address for m in messages} & {f"{a:06X}" for a in icaos}
    # Measured before: 79 (2.4), 92 (2.56) and 98 (3.2) of 100.
    assert len(found) >= {2.4e6: 90, 2.56e6: 88, 3.2e6: 94}[rate]


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6])
def test_short_replies_decode_at_every_chip_phase(rate):
    # DF11 all-call replies and DF4/DF5 surveillance replies are 56 bits:
    # the bit search ends after bit 55 for them.
    rng = np.random.default_rng(3)
    icao = 0x4840D6
    frames = [dd._extended_squitter(icao, dd._ident_me("KLM1023"))]
    for k in range(24):
        if k % 3 == 0:
            head = bytes([11 << 3 | 5]) + icao.to_bytes(3, "big")
            frames.append(head + dd._crc24(head).to_bytes(3, "big"))
        elif k % 3 == 1:
            frames.append(dd._address_parity_reply(4 << 27 | 0x1F5, icao))
        else:
            frames.append(dd._address_parity_reply(5 << 27 | 0x456, icao))
    offsets = (np.arange(len(frames)) + 0.5) / len(frames)
    mag = _stream(rate, frames, offsets, 30.0, rng)
    messages = _decode(ADSBDecoder(rate), mag, int(rate * GUI_BLOCK_S))
    assert [m.raw_bits for m in messages] == [
        f[:7] if (f[0] >> 3) in (0, 4, 5, 11) else f for f in frames
    ]


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6])
def test_replies_back_to_back_both_decode(rate):
    # Each reply starts 2 us after the one before ends (and a little over
    # a third of a sample further along the sample grid).
    rng = np.random.default_rng(5)
    frames = [_squitter(0x100000 + k, rng) for k in range(20)]
    n = int(0.004 * rate)
    x = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for k, frame in enumerate(frames):
        start = 100.0 + k * (math.floor(122e-6 * rate) + 0.37)
        first, env = dd._pulse_envelope(dd._modes_pulses(frame), start, rate)
        x[first : first + len(env)] += env
    messages = _decode(ADSBDecoder(rate), np.abs(x).astype(np.float32), 997)
    assert [m.raw_bits for m in messages] == frames


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6])
def test_noise_decodes_nothing(rate):
    rng = np.random.default_rng(11)
    decoder = ADSBDecoder(rate)
    block = int(rate * GUI_BLOCK_S)
    for _ in range(150):  # 5 s
        iq = rng.standard_normal(block) + 1j * rng.standard_normal(block)
        assert decoder.decode(np.abs(iq).astype(np.float32)) == []


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6, 2.4e6])
def test_the_bit_search_on_noise_is_bounded(rate):
    # Count the work, not the time: the Viterbi search runs only for
    # preambles whose fit looks like a reply (a few dozen a second of
    # noise), so a frame of noise costs about a millisecond.
    rng = np.random.default_rng(12)
    decoder = ADSBDecoder(rate)
    block = int(rate * GUI_BLOCK_S)
    for _ in range(60):  # 2 s
        iq = rng.standard_normal(block) + 1j * rng.standard_normal(block)
        decoder.decode(np.abs(iq).astype(np.float32))
    demod = decoder._fine
    assert demod is not None
    assert demod.searched < 2 * 800  # timing hypotheses searched, per 2 s


@pytest.mark.parametrize("rate", [2.0e6, 2.048e6])
def test_one_or_two_search_passes_find_the_same_replies(rate):
    # With many replies in a pass the other start times are searched only
    # for replies whose first frame doesn't check itself; the result must
    # not depend on how many replies a block happens to hold.
    _icaos, _frames, mag = _replies(rate, 60, 13.0, seed=21)
    found = []
    for one_pass in (0, 1000):
        decoder = ADSBDecoder(rate)
        decoder._fine.ONE_PASS = one_pass
        found.append([m.raw_bits for m in _decode(decoder, mag, len(mag))])
    assert found[0] == found[1] and len(found[0]) > 20


def test_the_old_slot_reader_stays_from_2_5_ms_s():
    assert ADSBDecoder(2.4e6)._fine is not None
    assert ADSBDecoder(2.56e6)._fine is None
    assert ADSBDecoder(20e6)._fine is None


class TestDemoAirliner:
    """The demo device's airliner at 1090 MHz, through the window's path."""

    @staticmethod
    def _decode(rate, seconds):
        from unittest import mock

        from sdr_module.dsp.protocols import ProtocolType, demodulate_for_protocol

        with mock.patch.object(dd, "_MAX_LAG_S", 1e9):
            dev = dd.MockDevice()
            dev.set_sample_rate(rate)
            dev.set_frequency(1090e6)
            dev._rng = np.random.default_rng(5)
            dev.start_rx()
            decoder = ADSBDecoder(rate)
            messages, total = [], 0
            block = int(rate * GUI_BLOCK_S)
            while total < rate * seconds:
                iq = dev.read_samples(block)
                total += len(iq)
                baseband = demodulate_for_protocol(iq, ProtocolType.ADSB)
                messages += decoder.decode(baseband)
        return messages

    @pytest.mark.parametrize("rate", [2.0e6, 2.048e6])
    def test_every_message_decodes(self, rate):
        # Before: 73 of 80 messages in 20 s at 2.0 MS/s, 50 at 2.048.
        seconds = 10.0
        sent = seconds / dd._SQUITTER_CYCLE_S * len(dd._SQUITTERS)
        messages = self._decode(rate, seconds)
        assert {m.icao_address for m in messages} == {"4840D6"}
        assert len(messages) >= sent - 1
        assert "KLM1023" in {m.callsign for m in messages}
        assert "3472" in {m.squawk for m in messages}
