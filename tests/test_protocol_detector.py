"""ProtocolDetector finds protocols with the real decoders.

fg:FG-B10 / fg:FG-G10 / mc:MC-G13: the detector's registry was empty, the
dsp decoders had no can_decode or protocol_info, so registering one made
detect() raise AttributeError, and it found nothing in 1 s of demo
1090 MHz data that the ADS-B decoder decodes.
"""

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.protocols import ProtocolDetector
from sdr_module.protocols.base import ProtocolType
from sdr_module.protocols.detector import LiveProtocolDecoder, live_protocol_decoder


def _demo(freq, rate, seconds, start_s=0.0):
    from sdr_module.gui.device_dialog import MockDevice

    dev = MockDevice()
    dev.set_sample_rate(rate)
    dev.set_frequency(freq)
    dev.start_rx()
    dev._stream_pos = int(start_s * rate)
    block = int(rate / 10)
    return np.concatenate([dev.read_samples(block) for _ in range(int(seconds * 10))])


def _place(baseband, rate, lead_s=0.2, tail_s=0.2):
    lead, tail = int(lead_s * rate), int(tail_s * rate)
    x = np.zeros(lead + len(baseband) + tail, dtype=np.complex64)
    x[lead : lead + len(baseband)] = baseband
    return x


class TestRegistry:
    def test_the_live_decoders_are_registered(self):
        detector = ProtocolDetector(2.4e6)
        assert set(detector.list_protocols()) == {
            "POCSAG",
            "FLEX",
            "AX.25/APRS",
            "ADS-B",
            "ACARS",
            "RDS",
            "CW (Morse)",
        }
        info = detector.get_protocol_info("ADS-B")
        assert (info.name, info.protocol_type, info.modulation) == (
            "ADS-B",
            ProtocolType.AVIATION,
            "PPM",
        )
        assert detector.get_protocol_info("RDS").symbol_rate == 1187.5

    def test_only_what_works_at_the_rate(self):
        names = ProtocolDetector(240e3).list_protocols()
        assert "ADS-B" not in names  # needs 2 MS/s
        assert "AX.25/APRS" in names and "RDS" in names

    def test_empty_on_request(self):
        assert ProtocolDetector(2.4e6, register_defaults=False).list_protocols() == []

    def test_a_decoder_that_cannot_detect_does_not_break_it(self):
        from sdr_module.dsp.protocols import ADSBDecoder

        detector = ProtocolDetector(2.4e6, register_defaults=False)
        detector.register_decoder("adsb", ADSBDecoder)  # no can_decode
        assert detector.detect(np.zeros(1000, dtype=np.complex64)) == []
        assert detector.detect_and_decode(np.zeros(1000, dtype=np.complex64)) == (
            [],
            [],
        )


class TestDetection:
    def test_demo_1090_mhz_is_ads_b(self):
        # The case of the audit: 1 s of the demo's 1090 MHz data.
        detector = ProtocolDetector(2.4e6)
        matches = detector.detect(_demo(1090e6, 2.4e6, 1.0))
        assert matches and matches[0].protocol_info.name == "ADS-B"
        assert matches[0].confidence == 1.0
        assert [m.protocol_info.name for m in matches] == ["ADS-B"]

    def test_aprs_packet(self):
        rate = 240e3
        x = _place(
            enc.aprs_baseband("N0CALL-9", "APZ001", "!4903.50N/07201.75W>Hi", rate),
            rate,
        )
        matches = ProtocolDetector(rate).detect(x)
        assert [m.protocol_info.name for m in matches] == ["AX.25/APRS"]
        frames = ProtocolDetector(rate).decode(x)
        assert len(frames) == 1 and frames[0].protocol == "AX.25/APRS"
        assert frames[0].is_valid and frames[0].data["source"] == "N0CALL-9"

    def test_rds_station_at_an_offset(self):
        rate, offset = 1.2e6, 150e3
        groups = enc.rds_ps_groups(0x1234, "TEST FM")
        multiplex = enc.fm_broadcast_composite(rate, 1.5, enc.rds_bits(groups))
        t = np.arange(len(multiplex)) / rate
        x = np.exp(2j * np.pi * (np.cumsum(multiplex) / rate + offset * t))
        matches = ProtocolDetector(rate, offset_hz=offset).detect(
            x.astype(np.complex64)
        )
        assert matches and matches[0].protocol_info.name == "RDS"

    def test_failed_checks_are_a_weaker_match(self):
        rate = 240e3
        frame = bytearray(enc.ax25_frame("N0CALL-9", "APZ001", "Damaged packet"))
        frame[-4] ^= 0x01  # the CRC fails
        line = enc.hdlc_bits(bytes(frame))
        bb = enc.fm_modulate(enc.APRS_DEVIATION_HZ * enc.afsk_audio(line, rate), rate)
        matches = ProtocolDetector(rate).detect(_place(bb, rate))
        assert [(m.protocol_info.name, m.confidence) for m in matches] == [
            ("AX.25/APRS", 0.6)
        ]
        assert ProtocolDetector(rate).detect(_place(bb, rate), min_confidence=0.8) == []

    def test_noise_is_nothing(self):
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(480_000) + 1j * rng.standard_normal(480_000)) * 0.1
        assert ProtocolDetector(240e3).detect(noise.astype(np.complex64)) == []

    def test_adapter_class(self):
        cls = live_protocol_decoder("ACARS", offset_hz=5e3)
        decoder = cls(240e3)
        assert isinstance(decoder, LiveProtocolDecoder)
        assert (decoder.live_name, decoder.offset_hz) == ("ACARS", 5e3)
        assert decoder.can_decode(np.zeros(24_000, dtype=np.complex64)) == 0.0

    def test_low_rate_ads_b_is_no_match(self):
        detector = ProtocolDetector(1.8e6, register_defaults=False)
        detector.register_decoder("ADS-B", live_protocol_decoder("ADS-B"))
        # The decoder can't work at 1.8 MS/s: no match, no exception.
        assert detector.detect(np.zeros(18_000, dtype=np.complex64)) == []


def test_morse_needs_a_few_characters():
    rate = 48e3
    x = enc.cw_baseband("CQ DE N0CALL", rate, 20)
    matches = ProtocolDetector(rate).detect(x)
    assert [m.protocol_info.name for m in matches] == ["CW (Morse)"]
    # One keyed burst reads as a letter or two: no match.
    burst = np.zeros(int(1.5 * rate), dtype=np.complex64)
    burst[int(0.4 * rate) : int(0.5 * rate)] = 1.0
    assert ProtocolDetector(rate).detect(burst) == []


@pytest.mark.parametrize("name", ["POCSAG", "FLEX", "ACARS"])
def test_each_protocol_is_recognised(name):
    rate = 240e3
    baseband = {
        "POCSAG": lambda: enc.pocsag_baseband([(1234567, 3, "TEST")], rate),
        "FLEX": lambda: enc.flex_baseband([(1234567, "FLEX TEST")], rate),
        "ACARS": lambda: enc.acars_baseband(enc.acars_block("N1", "H1", "TEST"), rate),
    }[name]()
    matches = ProtocolDetector(rate).detect(_place(baseband, rate))
    assert matches and matches[0].protocol_info.name == name
    assert matches[0].confidence == 1.0
