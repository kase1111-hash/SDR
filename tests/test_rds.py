"""RDS from FM broadcasts: the group decoder, the physical layer, the 'RDS'
live decoder and the demo station.

fg:FG-G05 / tl:G-06 / ux:UX-G04: RDSDecoder parsed groups from bits, but
nothing recovered the 57 kHz subcarrier, the decoder had no clock recovery
or error correction and dropped each call's tail, RDS was disabled in the
Decoder panel and the demo's 57 kHz carrier sent no data.
"""

import datetime

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import (
    RDSLiveDecoder,
    create_live_decoder,
    describe_message,
    live_decoder_class,
)
from sdr_module.dsp.protocols import (
    ProtocolType,
    RDSData,
    RDSDecoder,
    _mjd_date,
    rds_char,
)
from sdr_module.dsp.rds import RDSDemodulator

PI = 0xC0DE
NAME = "SDR DEMO"
TEXT = "Now playing: the SDR Module demo tune"


def _groups(text=TEXT, ab=0, pty=9):
    return (
        enc.rds_ps_groups(PI, NAME, pty=pty, af_mhz=(100.1, 99.5))
        + enc.rds_radiotext_groups(PI, text, pty=pty, ab=ab)
        + [enc.rds_clock_group(PI, 2026, 10, 30, 14, 23, -10, pty=pty)]
    )


def _updates(decoder, bits):
    return [(u.update, u) for u in decoder.decode_bits(bits, updates_only=True)]


def _flip(bits, *positions):
    out = list(bits)
    for p in positions:
        out[p] ^= 1
    return out


# --------------------------------------------------------------------------- #
# Group decoder
# --------------------------------------------------------------------------- #


class TestGroups:
    def test_station_name_text_type_and_clock(self):
        decoder = RDSDecoder(1187.5)
        updates = dict(_updates(decoder, enc.rds_bits(_groups())))
        assert updates["ps"].ps_name == NAME and updates["ps"].ps_complete
        assert updates["radio_text"].radio_text == TEXT
        assert updates["radio_text"].radio_text_complete
        assert updates["pty"].pty_name == "Top 40"  # RBDS without a country code
        # 14:23 UTC, local offset -5 h.
        assert updates["clock_time"].clock_time == "2026-10-30 09:23 (UTC-05:00)"
        station = decoder.station()
        assert station.pi_code == PI
        assert station.af_list == [99.5, 100.1]
        assert station.protocol is ProtocolType.RDS
        assert decoder.frames_seen == len(_groups()) and decoder.frames_failed == 0

    def test_bits_in_any_pieces_decode_alike(self):
        bits = enc.rds_bits(_groups()) * 2
        whole = RDSDecoder(1187.5).decode_bits(bits, updates_only=True)
        decoder = RDSDecoder(1187.5)
        pieces = []
        for i in range(0, len(bits), 37):
            pieces += decoder.decode_bits(bits[i : i + 37], updates_only=True)
        assert [(u.update, u.ps_name, u.radio_text) for u in pieces] == [
            (u.update, u.ps_name, u.radio_text) for u in whole
        ]

    def test_a_snapshot_after_every_group_by_default(self):
        decoder = RDSDecoder(1187.5)
        snapshots = decoder.decode_bits(enc.rds_bits(enc.rds_ps_groups(PI, NAME)))
        assert [s.group for s in snapshots] == ["0A"] * 4
        assert snapshots[-1].ps_name == NAME

    def test_a_new_name_needs_all_four_segments(self):
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(enc.rds_bits(enc.rds_ps_groups(PI, NAME)))
        once = _updates(decoder, enc.rds_bits(enc.rds_ps_groups(PI, "SDR TEST")))
        assert once == []  # segments 0-1 came before the change
        twice = _updates(decoder, enc.rds_bits(enc.rds_ps_groups(PI, "SDR TEST")))
        assert [(k, u.ps_name) for k, u in twice] == [("ps", "SDR TEST")]

    def test_radiotext_ab_flag_starts_a_new_text(self):
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(enc.rds_bits(_groups()))
        groups = enc.rds_radiotext_groups(PI, "Next", pty=9, ab=1)
        new = _updates(decoder, enc.rds_bits(groups))
        assert [(k, u.radio_text) for k, u in new] == [("radio_text", "Next")]

    def test_version_b_groups_take_the_pi_from_block_c_prime(self):
        # A type 2B group: block C' carries the PI, block D two characters.
        b = (0x05 << 11) | (9 << 5) | (1 << 4)  # 2B, PTY 9, A/B 1, segment 0
        chars = (ord("H") << 8) | ord("I")
        bits = enc.rds_bits([(PI, b, PI, chars), (PI, b + 1, PI, 0x0D20)])
        decoder = RDSDecoder(1187.5)
        updates = dict(_updates(decoder, bits * 2))
        assert decoder.station().pi_code == PI
        assert updates["radio_text"].radio_text == "HI"

    def test_country_code_picks_rds_program_type_names(self):
        decoder = RDSDecoder(1187.5)
        bits = enc.rds_bits([enc.rds_ecc_group(PI, 0xE0)] + _groups(pty=10))  # Germany
        decoder.decode_bits(bits)
        station = decoder.station()
        assert (station.ecc, station.rbds, station.pty_name) == (
            0xE0,
            False,
            "Pop Music",
        )
        us = RDSDecoder(1187.5)
        us.decode_bits(enc.rds_bits([enc.rds_ecc_group(PI, 0xA0)] + _groups(pty=10)))
        assert us.station().pty_name == "Country"
        assert RDSDecoder(1187.5).get_pty_name(5) == "Rock"

    def test_characters(self):
        assert rds_char(0x41) == "A"
        assert rds_char(0x91) == "ä" and rds_char(0x97) == "ö" and rds_char(0x99) == "ü"
        assert rds_char(0x8D) == "ß" and rds_char(0x82) == "é"
        assert rds_char(0x0A) == " " and rds_char(0x1F) == ""

    def test_modified_julian_day(self):
        mjd = (datetime.date(2026, 10, 30) - datetime.date(1858, 11, 17)).days
        assert _mjd_date(mjd) == (2026, 10, 30)
        assert _mjd_date(mjd + 62) == (2026, 12, 31)
        assert _mjd_date(mjd + 63) == (2027, 1, 1)
        assert _mjd_date(0) is None


class TestSyncAndCorrection:
    def test_errors_of_two_bits_in_a_block_are_corrected(self):
        bits = enc.rds_bits(_groups())
        decoder = RDSDecoder(1187.5)
        # One wrong bit on the air is two wrong data bits after differential
        # decoding: in blocks after sync (block A of the third group, the
        # second group's D).
        broken = _flip(bits, 2 * 104 + 5, 2 * 104 + 6, 104 + 3 * 26 + 10)
        updates = dict(_updates(decoder, broken))
        assert updates["ps"].ps_name == NAME
        assert decoder.blocks_corrected == 2 and decoder.blocks_failed == 0

    def test_a_corrected_block_does_not_change_the_name(self):
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(enc.rds_bits(enc.rds_ps_groups(PI, NAME)))
        b = (0x00 << 11) | (9 << 5) | 0  # 0A, segment 0
        group = [
            (PI, "A", False),
            (b, "B", False),
            (0, "C", False),
            (0x5858, "D", True),
        ]
        decoder._process_blocks(group)
        assert decoder._ps[:2] == ["S", "D"]  # not "XX"
        group[3] = (0x5858, "D", False)  # an exact block does change it,
        decoder._process_blocks(group)
        assert decoder.station().ps_name == NAME  # once the rest arrives:
        decoder.decode_bits(enc.rds_bits(enc.rds_ps_groups(PI, NAME)[1:]))
        assert decoder.station().ps_name == "XXR DEMO"

    def test_bad_blocks_lose_sync_and_it_comes_back(self):
        bits = enc.rds_bits(_groups())
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(bits)
        assert decoder.synced
        rng = np.random.default_rng(3)
        decoder.decode_bits(rng.integers(0, 2, 26 * 12).tolist())
        assert not decoder.synced
        assert decoder.blocks_failed >= RDSDecoder.SYNC_LOSS_BLOCKS
        decoder.decode_bits(bits)
        assert decoder.synced

    def test_a_slipped_bit_resyncs_quickly(self):
        bits = enc.rds_bits(_groups()) * 3
        cut = len(bits) // 3 + 40
        slipped = bits[:cut] + bits[cut + 1 :]  # one bit lost
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(slipped)
        # Lost: the group with the slip and the next while finding sync again.
        assert decoder.frames_seen - decoder.frames_failed >= 2 * len(_groups()) - 3
        assert decoder.synced

    @pytest.mark.parametrize("lost", [1, 2, 3, 5])
    def test_lost_blocks_resync_without_garbage(self, lost):
        # Whole blocks lost (a dropout of a multiple of 22 ms, e.g. samples
        # skipped while the window was busy): the blocks after it still
        # check, as other blocks of the group. Offset words A, B, C' and D
        # are a one- or two-bit burst apart, so they used to be "corrected"
        # into the blocks expected: the decoder stayed out of step, counted
        # garbage groups as good and read the PI code as a program type
        # ("Emergency Test").
        bits = enc.rds_bits(_groups()) * 4
        cut = len(bits) // 4 + 2 * 104 + 3 * 26  # before block D of a group
        stream = bits[:cut] + bits[cut + 26 * lost :]
        decoder = RDSDecoder(1187.5)
        updates = decoder.decode_bits(stream, updates_only=True)
        assert [u.pty for u in updates if u.update == "pty"] == [9]
        assert [u.ps_name for u in updates if u.update == "ps"] == [NAME]
        assert decoder.blocks_corrected == 0
        # Back in step at once: only the groups the dropout cut are lost.
        assert decoder.frames_seen - decoder.frames_failed >= 4 * len(_groups()) - 2

    def test_joining_mid_group_is_no_failure(self):
        # Sync found on blocks C and D: that group's A and B were before
        # the sync, so it is no failed group (it showed "1 failed check" on
        # every station tuned to).
        bits = enc.rds_bits(_groups())
        decoder = RDSDecoder(1187.5)
        decoder.decode_bits(bits[52:])  # from block C of the first group
        assert decoder.synced and decoder.frames_failed == 0
        assert decoder.frames_seen == len(_groups()) - 1

    def test_noise_finds_nothing(self):
        rng = np.random.default_rng(7)
        decoder = RDSDecoder(1187.5)
        updates = decoder.decode_bits(rng.integers(0, 2, 60_000).tolist(), True)
        assert [u for u in updates if u.update in ("ps", "radio_text")] == []


# --------------------------------------------------------------------------- #
# Physical layer
# --------------------------------------------------------------------------- #


def _decode_multiplex(multiplex, rate, block=None):
    demod = RDSDemodulator(rate)
    decoder = RDSDecoder(demod.work_rate)
    step = block or len(multiplex)
    updates = []
    for i in range(0, len(multiplex), step):
        updates += decoder.decode_bits(demod.process(multiplex[i : i + step]), True)
    return updates, decoder, demod


class TestDemodulator:
    @pytest.mark.parametrize("rate", [228e3, 240e3, 480e3])
    @pytest.mark.parametrize(
        "pilot,phase", [(True, 0.0), (True, np.pi / 2), (False, 0.7)]
    )
    def test_multiplex_to_station(self, rate, pilot, phase):
        bits = enc.rds_bits(_groups())
        rng = np.random.default_rng(1)
        multiplex = enc.fm_broadcast_composite(
            rate, 4.0, bits, pilot=pilot, rds_phase=phase
        )
        multiplex += 2000.0 * rng.standard_normal(len(multiplex))
        updates, decoder, demod = _decode_multiplex(multiplex, rate, int(rate / 30))
        found = {u.update: u for u in updates}
        assert found["ps"].ps_name == NAME
        assert found["radio_text"].radio_text == TEXT
        assert demod.pilot is pilot
        assert decoder.frames_failed == 0
        assert demod.quality > 0.5

    def test_blocks_decode_like_the_whole(self):
        rate = 240e3
        multiplex = enc.fm_broadcast_composite(rate, 3.0, enc.rds_bits(_groups()))
        whole, _d, _m = _decode_multiplex(multiplex, rate)
        streamed, _d2, _m2 = _decode_multiplex(multiplex, rate, 8000)
        key = [(u.update, u.ps_name, u.radio_text) for u in whole]
        assert key and key == [(u.update, u.ps_name, u.radio_text) for u in streamed]

    def test_rejects_a_rate_too_low_for_57_khz(self):
        with pytest.raises(ValueError, match="RDS needs"):
            RDSDemodulator(100e3)

    def test_audio_alone_gives_no_station(self):
        rate = 240e3
        multiplex = enc.fm_broadcast_composite(rate, 3.0, None)
        updates, decoder, demod = _decode_multiplex(multiplex, rate)
        assert updates == [] and demod.pilot

    def test_periodic_baseband_is_exact(self):
        # A period of a whole number of samples: the Fourier series and the
        # pulses summed directly agree.
        bits = enc.rds_bits(enc.rds_ps_groups(PI, NAME))  # even count of ones
        rate = 47500.0
        loop = enc.rds_baseband(bits, rate, periodic=True)
        three = enc.rds_baseband(bits * 3, rate, span_bits=40)
        assert np.max(np.abs(loop - three[len(loop) : 2 * len(loop)])) < 1e-3
        assert 0.9 < np.max(np.abs(loop)) < 1.2


# --------------------------------------------------------------------------- #
# The 'RDS' live decoder
# --------------------------------------------------------------------------- #


def _wfm(rate, seconds, offset, cnr_db=None, seed=2):
    """An RDS station ``offset`` from the centre of a span at ``rate``,
    with noise ``cnr_db`` below it in a 200 kHz channel."""
    multiplex = enc.fm_broadcast_composite(rate, seconds, enc.rds_bits(_groups()))
    t = np.arange(len(multiplex)) / rate
    phase = 2 * np.pi * np.cumsum(multiplex) / rate + 2 * np.pi * offset * t
    x = np.exp(1j * phase).astype(np.complex64)
    if cnr_db is not None:
        rng = np.random.default_rng(seed)
        sigma = np.sqrt(10 ** (-cnr_db / 10) * (rate / 200e3) / 2)
        noise = rng.standard_normal(2 * len(x), dtype=np.float32).view(np.complex64)
        x += np.float32(sigma) * noise
    return x


class TestRDSLiveDecoder:
    def test_registered(self):
        assert live_decoder_class("RDS") is RDSLiveDecoder
        assert live_decoder_class("rds") is RDSLiveDecoder
        with pytest.raises(ValueError, match="too low for RDS"):
            create_live_decoder("RDS", 100e3)

    def test_station_from_raw_iq_off_the_centre(self):
        rate, offset = 1.2e6, -350e3
        decoder = create_live_decoder("RDS", rate, offset)
        x = _wfm(rate, 3.0, offset, cnr_db=20)
        messages = []
        for i in range(0, len(x), int(rate / 30)):
            messages += decoder.process(x[i : i + int(rate / 30)])
        rows = [describe_message(m) for m in messages]
        assert rows[:3] == [
            ("PI C0DE", "Station name: SDR DEMO"),
            ("PI C0DE", "Program type: Top 40"),
            ("PI C0DE", f"Radio text: {TEXT}"),
        ]
        status = decoder.status
        assert status.detail == NAME and status.signal_present
        assert status.valid == status.frames > 20 and status.failed == 0
        assert status.last_time is not None

    def test_a_cheap_channel_from_20_ms_s(self):
        # A HackRF at 20 MS/s: the channel is decimated by 80 (16 x 5). It
        # was 81 (9 x 9), whose first stage cost twice as much: the decoder
        # took about 20 ms of each 33 ms frame.
        channelizer = RDSLiveDecoder(20e6, 300e3)._channelizer
        assert channelizer.output_rate == 250e3
        assert channelizer._stages[0].factor >= 16
        # From the usual rates the channel is as it was.
        assert RDSLiveDecoder(2.4e6).channel_rate == 240e3
        assert RDSLiveDecoder(2.048e6).channel_rate == 256e3

    def test_a_cheap_channel_from_8_ms_s(self):
        # DEC-07: from 8 MS/s the first stage only decimated by 4 (a fixed
        # rule kept 8 times the stopband edge as room), so the 183-tap
        # channel filter ran at 2 MS/s and the decoder took about 9 ms of
        # each 33 ms frame. The split is now chosen by its cost.
        channelizer = RDSLiveDecoder(8e6, 300e3)._channelizer
        assert channelizer.output_rate == 250e3
        assert channelizer._stages[0].factor >= 8

    def test_from_the_receivers_channel(self):
        # The receiver's broadcast FM channel: already at 0 Hz, 480 kS/s.
        rate = 480e3
        decoder = RDSLiveDecoder(rate, 0.0, channel_ready=True)
        x = _wfm(rate, 2.5, 0.0)
        messages = []
        for i in range(0, len(x), 16000):
            messages += decoder.process(x[i : i + 16000])
        assert ("PI C0DE", "Station name: SDR DEMO") in [
            describe_message(m) for m in messages
        ]
        assert decoder.channel_rate == rate

    def test_retuning_forgets_the_station(self):
        rate = 480e3
        decoder = RDSLiveDecoder(rate, 0.0, channel_ready=True)
        decoder.process(_wfm(rate, 2.5, 0.0))
        assert decoder.station.ps_name == NAME
        decoder.channel_changed()
        assert decoder.station.ps_name == "" and not decoder.synced
        assert decoder.status.detail == ""

    def test_a_station_without_rds(self):
        rate = 480e3
        multiplex = enc.fm_broadcast_composite(rate, 2.0, None)
        x = np.exp(2j * np.pi * np.cumsum(multiplex) / rate).astype(np.complex64)
        decoder = RDSLiveDecoder(rate, 0.0, channel_ready=True)
        assert decoder.process(x) == []
        status = decoder.status
        # The pilot is there: the panel says nothing decodes after a while.
        assert status.signal_present and status.undecoded_signal_s > 1.5

    def test_message_rows(self):
        def row(**fields):
            msg = RDSData(
                protocol=ProtocolType.RDS,
                timestamp=0.0,
                raw_bits=b"",
                valid=True,
                **fields,
            )
            return describe_message(msg)

        assert row(pi_code=0x1234, update="ps", ps_name="ROCK FM") == (
            "PI 1234",
            "Station name: ROCK FM",
        )
        assert row(update="clock_time", clock_time="2026-10-30 09:23 (UTC-05:00)") == (
            "",
            "Clock: 2026-10-30 09:23 (UTC-05:00)",
        )
        assert row(pi_code=0x1234, ps_name="A", radio_text="B") == (
            "PI 1234",
            "A  ·  B",
        )


class TestDemoStation:
    """The demo device's 100.1 MHz station sends real RDS."""

    def test_decodes_in_33_ms_blocks_like_whole(self):
        from sdr_module.gui.device_dialog import MockDevice

        rate, lo = 1.024e6, 100.0e6
        dev = MockDevice()
        dev.set_sample_rate(rate)
        dev.set_frequency(lo)
        dev.start_rx()
        dev._stream_pos = 0
        block = int(rate / 30)
        iq = np.concatenate([dev.read_samples(block) for _ in range(6 * 30)])
        results = []
        for step in (len(iq), block):
            decoder = create_live_decoder("RDS", rate, 100.1e6 - lo)
            rows = []
            for i in range(0, len(iq), step):
                rows += [describe_message(m) for m in decoder.process(iq[i : i + step])]
            results.append(rows)
        assert results[0] == results[1]
        assert results[0][:3] == [
            ("PI C0DE", "Station name: SDR DEMO"),
            ("PI C0DE", "Program type: Top 40"),
            (
                "PI C0DE",
                "Radio text: Now playing: the SDR Module demo tune (I-vi-IV-V in C)",
            ),
        ]
        assert decoder.status.failed == 0

    def test_the_loop_has_no_seam(self):
        from sdr_module.gui import device_dialog as dd

        bits = enc.rds_bits(dd._rds_groups())
        assert len(dd._rds_groups()) == dd._RDS_LOOP_GROUPS
        assert sum(bits) % 2 == 0  # differential coding repeats
        for rate in (2.4e6, 2.048e6, 1.024e6):
            loop = dd._rds_loop(rate)
            assert len(loop.values) * loop.factor == pytest.approx(
                rate * len(bits) / 1187.5
            )

    def test_other_stations_send_no_rds(self):
        from sdr_module.gui.device_dialog import MockDevice

        rate = 1.024e6
        dev = MockDevice()
        dev.set_sample_rate(rate)
        dev.set_frequency(99.5e6)
        dev.start_rx()
        decoder = create_live_decoder("RDS", rate, 0.0)
        for _ in range(60):
            decoder.process(dev.read_samples(int(rate / 30)))
        assert decoder.status.detail == "" and decoder.status.frames == 0
