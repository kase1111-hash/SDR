"""FG-B12: the text encoders send standard RTTY, the full PSK31 Varicode,
ASCII FSK and Morse code, never drop a character silently, and the CLI's
``encode`` builds each protocol's own configuration.

Every check reads the signal back the way a receiver would (tests/
text_demod.py: FM discriminator and bit clock for RTTY/ASCII, symbol clock
and phase comparison for PSK31, the repo's Morse decoder for CW).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sdr_module.cli import main
from sdr_module.protocols.encoder import EncoderConfig, ModulationType
from sdr_module.protocols.encoders import (
    ASCIIEncoder,
    MorseEncoder,
    PSK31Encoder,
    RTTYEncoder,
)
from tests import text_demod as demod

FS = 48000.0

# sha256 of the 128 Varicode codes (ASCII 0-127, one per line) of G3PLX's
# PSK31 table, as fldigi ships it (src/psk/pskvaricode.cxx).
G3PLX_VARICODE_SHA256 = (
    "87dc639d0f500d147028fc3f4e6d847f5470125cf9a7ce47166aacb21435efdf"
)


def _rtty(carrier=1000.0, baud=0.0, shift=None, amplitude=0.8):
    return RTTYEncoder(
        EncoderConfig(FS, carrier, baud, ModulationType.FSK, amplitude, shift)
    )


def _psk(carrier=1000.0):
    return PSK31Encoder(EncoderConfig(FS, carrier, 0, ModulationType.PSK, 0.8))


def _varicode_table():
    return {code: char for char, code in PSK31Encoder.VARICODE.items()}


class TestRTTY:
    def test_digits_and_punctuation_go_out_with_figures_shifts(self):
        text = "CQ CQ DE N0CALL 599 TU. QTH: PARIS? 73"
        iq = _rtty().encode_text(text)
        assert demod.rtty_decode(iq, FS, center_hz=1000.0) == text

    def test_receivers_that_unshift_on_space_print_the_figures_too(self):
        text = "RST 599 599 73 73"
        iq = _rtty().encode_text(text)
        assert demod.rtty_decode(iq, FS, center_hz=1000.0, usos=True) == text
        assert demod.rtty_decode(iq, FS, center_hz=1000.0, usos=False) == text

    def test_us_tty_figures(self):
        text = "$5 & 'A' \"B\" #1 X!"
        iq = _rtty().encode_text(text)
        assert demod.rtty_decode(iq, FS, center_hz=1000.0) == text

    def test_standard_45_45_baud_170_hz_with_1_5_stop_bits(self):
        enc = _rtty(carrier=2210.0)
        assert enc.config.baud_rate == 45.45
        assert enc.config.frequency_shift == 170
        one = enc.encode_text("RY" * 10)
        two = enc.encode_text("RY" * 20)
        # Each character is 7.5 bits: start, five data bits, 1.5 stop bits.
        per_char = (len(two) - len(one)) / FS / 20
        assert per_char == pytest.approx(7.5 / 45.45, rel=1e-3)
        freqs = demod.tone_frequencies(np.real(two), FS)
        assert freqs[0] == pytest.approx(2125.0, abs=4)
        assert freqs[1] == pytest.approx(2295.0, abs=4)

    def test_mark_is_the_higher_tone(self):
        # Idle (mark) before the first start bit: the higher tone.
        iq = _rtty(carrier=0.0).encode_text("E")
        f = demod.instantaneous_frequency(iq[2000:8000], FS)
        assert np.median(f) == pytest.approx(85.0, abs=1.0)

    def test_phase_continuous(self):
        iq = _rtty(carrier=1000.0).encode_text("RYRYRY 123")
        f = demod.instantaneous_frequency(iq, FS)
        body = f[200:-200]
        # Two tones only, never a phase jump between them.
        assert np.all(np.abs(body - 1000.0) < 86.0)

    def test_characters_rtty_lacks_are_reported(self):
        enc = _rtty()
        iq = enc.encode_text("A@B%C@")
        assert enc.skipped == {"@": 2, "%": 1}
        assert demod.rtty_decode(iq, FS, center_hz=1000.0) == "ABC"
        assert enc.unsupported_chars("Hi @ there #1") == {"@": 1}

    def test_letters_go_out_as_capitals(self):
        enc = _rtty()
        assert np.array_equal(enc.encode_text("hello"), enc.encode_text("HELLO"))
        assert enc.skipped == {}


class TestPSK31:
    def test_the_varicode_is_g3plx_full_table(self):
        table = PSK31Encoder.VARICODE
        assert len(table) == 128
        codes = [table[chr(i)] for i in range(128)]
        digest = hashlib.sha256("\n".join(codes).encode()).hexdigest()
        assert digest == G3PLX_VARICODE_SHA256
        # Spot checks against G3PLX's published table.
        expected = {
            " ": "1",
            "e": "11",
            "t": "101",
            "o": "111",
            "a": "1011",
            "i": "1101",
            "n": "1111",
            "r": "10101",
            "s": "10111",
            "l": "11011",
            "h": "101011",
            "d": "101101",
            "c": "101111",
            "E": "1110111",
            "Q": "111011101",
            "0": "10110111",
            "\r": "11111",
            "\n": "11101",
        }
        for char, code in expected.items():
            assert table[char] == code, char

    def test_round_trip_keeps_case_digits_and_punctuation(self):
        text = "CQ CQ de N0CALL: Hello, 599 {ok} ~73!"
        iq = _psk().encode_text(text)
        assert demod.psk31_decode(iq, FS, _varicode_table(), center_hz=1000.0) == text

    def test_every_letter_is_sent(self):
        enc = _psk()
        assert not np.array_equal(enc.encode_text("CQ"), enc.encode_text("C"))
        assert not np.array_equal(enc.encode_text("Hello"), enc.encode_text("hello"))
        assert not np.array_equal(enc.encode_text("TEST 123"), enc.encode_text("test "))

    def test_reversals_are_amplitude_shaped(self):
        iq = _psk(carrier=0.0).encode_text("ee")
        env = np.abs(iq)
        # A cosine through zero at each reversal, never a step.
        assert np.max(np.abs(np.diff(env))) < 0.01
        assert env.min() < 0.02

    def test_characters_psk31_lacks_are_reported(self):
        enc = _psk()
        iq = enc.encode_text("café über")
        assert enc.skipped == {"é": 1, "ü": 1}
        assert demod.psk31_decode(iq, FS, _varicode_table(), 1000.0) == "caf ber"


class TestASCII:
    def test_non_ascii_is_reported_instead_of_raising(self):
        enc = ASCIIEncoder(EncoderConfig(FS, 1500, 300, ModulationType.FSK))
        iq = enc.encode_text("café")
        assert enc.skipped == {"é": 1}
        assert demod.ascii_decode(iq, FS, 1500.0, 300.0) == b"caf"

    def test_round_trip(self):
        enc = ASCIIEncoder(EncoderConfig(FS, 1500, 300, ModulationType.FSK))
        text = "Hello, World! 0123 ~{}"
        iq = enc.encode_text(text)
        assert demod.ascii_decode(iq, FS, 1500.0, 300.0) == text.encode()


class TestMorse:
    def _enc(self, wpm=20):
        return MorseEncoder(EncoderConfig(FS, 700, 0, ModulationType.OOK), wpm=wpm)

    def test_punctuation_is_sent(self):
        enc = self._enc()
        text = "CQ DE N0CALL/P QTH? 5NN = TU."
        iq = enc.encode_text(text)
        assert enc.skipped == {}
        assert demod.cw_decode(iq, FS, center_hz=700.0) == text

    def test_word_gap_is_seven_dits(self):
        enc = self._enc(wpm=20)
        dit = 1.2 / 20
        one = len(enc.encode_text("E E")) - len(enc.encode_text("E"))
        # "E E" adds a word gap (7 dits) and a dit.
        assert one / FS == pytest.approx(8 * dit, abs=0.002)

    def test_characters_morse_lacks_are_reported(self):
        enc = self._enc()
        enc.encode_text("50% #1")
        assert enc.skipped == {"%": 1, "#": 1}


class TestConfigsOutOfRange:
    """A configuration the encoders can't honour is refused, never quietly
    changed (2000 Hz of shift used to go out as 1000 Hz)."""

    def test_rtty_and_ascii_baud_and_shift(self):
        with pytest.raises(ValueError, match="shift"):
            _rtty(shift=2000.0)
        with pytest.raises(ValueError, match="baud"):
            _rtty(baud=5000.0)
        with pytest.raises(ValueError, match="baud"):
            ASCIIEncoder(EncoderConfig(FS, 1500, 9600, ModulationType.FSK))
        # In range: kept exactly.
        enc = _rtty(carrier=3000.0, baud=50.0, shift=850.0)
        iq = enc.encode_text("RYRY")
        low, high = demod.tone_frequencies(iq, FS)
        assert high - low == pytest.approx(850.0, abs=3)

    def test_morse_speed(self):
        with pytest.raises(ValueError, match="WPM"):
            MorseEncoder(EncoderConfig(FS, 700, 0, ModulationType.OOK), wpm=100)


class TestBitsToFsk:
    def test_phase_continuous_across_bits(self):
        enc = ASCIIEncoder(EncoderConfig(FS, 1500, 300, ModulationType.FSK))
        iq = enc.bits_to_fsk(np.array([0, 1, 0, 0, 1, 1, 0, 1]), 2000.0, 1000.0)
        f = demod.instantaneous_frequency(iq, FS)
        assert np.all((np.abs(f - 2000.0) < 1.0) | (np.abs(f - 1000.0) < 1.0))


class TestCliEncode:
    def test_rtty_is_45_45_baud_170_hz(self, capsys, tmp_path):
        out = tmp_path / "rtty.cf32"
        rc = main(["encode", "rtty", "--text", "CQ CQ DE N0CALL", "-o", str(out)])
        printed = capsys.readouterr().out
        assert rc == 0
        assert "Encoded 15 characters" in printed
        assert "45.45 Bd" in printed and "170 Hz shift" in printed
        iq = np.fromfile(out, np.complex64)
        assert demod.rtty_decode(iq, FS, center_hz=1000.0) == "CQ CQ DE N0CALL"
        freqs = demod.tone_frequencies(np.real(iq), FS)
        assert freqs[1] - freqs[0] == pytest.approx(170.0, abs=6)

    def test_baud_and_shift_options(self, capsys, tmp_path):
        out = tmp_path / "rtty75.cf32"
        rc = main(
            [
                "encode",
                "rtty",
                "--text",
                "RYRY 75 BAUD",
                "--baud",
                "75",
                "--shift",
                "850",
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert "75 Bd" in capsys.readouterr().out
        iq = np.fromfile(out, np.complex64)
        assert demod.rtty_decode(iq, FS, center_hz=1000.0, baud=75.0) == "RYRY 75 BAUD"
        freqs = demod.tone_frequencies(np.real(iq), FS)
        assert freqs[1] - freqs[0] == pytest.approx(850.0, abs=6)

    def test_skipped_characters_are_warned_and_not_counted(self, capsys):
        rc = main(["encode", "rtty", "--text", "A@B@C%"])
        printed = capsys.readouterr().out
        assert rc == 0
        assert "Encoded 3 characters" in printed
        assert "Skipped 3 characters RTTY can't send: @ (2x), %" in printed

    def test_psk31_keeps_case(self, capsys, tmp_path):
        out = tmp_path / "psk.cf32"
        rc = main(["encode", "psk31", "--text", "Hello DE N0CALL", "-o", str(out)])
        assert rc == 0
        assert "31.25 Bd" in capsys.readouterr().out
        iq = np.fromfile(out, np.complex64)
        assert demod.psk31_decode(iq, FS, _varicode_table(), 1000.0) == (
            "Hello DE N0CALL"
        )

    def test_ascii_takes_baud_and_shift(self, capsys, tmp_path):
        out = tmp_path / "ascii.cf32"
        rc = main(
            [
                "encode",
                "ascii",
                "--text",
                "hi there",
                "--baud",
                "110",
                "--shift",
                "200",
                "--carrier",
                "1170",
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert "110 Bd" in capsys.readouterr().out
        iq = np.fromfile(out, np.complex64)
        assert demod.ascii_decode(iq, FS, 1170.0, 110.0) == b"hi there"

    def test_baud_is_refused_for_morse_and_psk31(self, capsys):
        assert main(["encode", "psk31", "--text", "x", "--baud", "63"]) == 1
        assert "PSK31 always runs at 31.25" in capsys.readouterr().out

    def test_sample_rate_must_be_positive(self, capsys):
        assert main(["encode", "rtty", "--text", "RY", "--sample-rate", "0"]) == 1
        assert "--sample-rate must be positive" in capsys.readouterr().out

    def test_nothing_sendable_is_an_error(self, capsys):
        assert main(["encode", "rtty", "--text", "@@@"]) == 1
        assert "nothing to send" in capsys.readouterr().out.lower()
