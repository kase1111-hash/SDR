"""Streaming Morse (CW) decoding: sdr_module.dsp.morse, CWDemodulator and
the 'CW (Morse)' live decoder.

fg:FG-B05: CWDemodulator.detect_keying set its threshold from each block's
own max and min, and decode_morse padded every block with key-up at both
ends, so the same signal decoded whole gave "CQ CQ DE N0CALL PARIS 7" but
"?? ?? ?H ?????? ????? ?" in 33 ms blocks. fg:FG-G06: there was no way to
read CW as text in the window.
"""

import difflib

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.demodulators import MORSE_CODE, CWDemodulator
from sdr_module.dsp.live_decoders import (
    CWLiveDecoder,
    create_live_decoder,
    describe_message,
    live_decoder_class,
)
from sdr_module.dsp.morse import KeyingDetector, MorseDecoder, MorseMessage
from sdr_module.dsp.protocols import ProtocolType

MESSAGE = "CQ CQ DE N0CALL PARIS 73"


def _noisy(x, snr_db, seed, rate_bw=None):
    """``x`` (unit carrier) plus complex white noise ``snr_db`` below it
    (in the full bandwidth of the samples)."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    noise = sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return (x + noise).astype(np.complex64)


def _decode(x, rate, block=None):
    """Lines of text from a complex CW channel, fed ``block`` at a time."""
    keyer, morse = KeyingDetector(rate), MorseDecoder(rate)
    step = block or len(x)
    lines = []
    for i in range(0, len(x), step):
        lines += morse.process(keyer.process(np.abs(x[i : i + step])))
    lines += morse.flush()
    return [m.text for m in lines], morse


# --------------------------------------------------------------------------- #
# FG-B05: CWDemodulator streams
# --------------------------------------------------------------------------- #


class TestCWDemodulatorStreams:
    FS = 48000.0

    def _signal(self):
        # The verifier's case: 15 WPM, 5 ms key edges, about 40 dB SNR.
        rng = np.random.default_rng(5)
        key = enc.morse_keying(MESSAGE, self.FS, 15, edge_s=0.005)
        noise = 0.005 * (
            rng.standard_normal(len(key)) + 1j * rng.standard_normal(len(key))
        )
        return (0.5 * key + noise).astype(np.complex64)

    def _run(self, x, block=None):
        demod = CWDemodulator(self.FS, wpm=15)
        step = block or len(x)
        returned = ""
        for i in range(0, len(x), step):
            returned += demod.decode_morse(demod.detect_keying(x[i : i + step]))
        return demod, returned

    @pytest.mark.parametrize("block", [1600, 12000, 48000])
    def test_blocks_decode_like_the_whole_buffer(self, block):
        x = self._signal()
        whole, _ = self._run(x)
        streamed, returned = self._run(x, block)
        assert whole.decoded_text.strip() == MESSAGE
        assert streamed.decoded_text == whole.decoded_text
        assert returned == streamed.decoded_text

    def test_keying_is_the_same_in_blocks(self):
        x = self._signal()
        whole = CWDemodulator(self.FS).detect_keying(x)
        demod = CWDemodulator(self.FS)
        parts = [demod.detect_keying(x[i : i + 1600]) for i in range(0, len(x), 1600)]
        assert np.array_equal(np.concatenate(parts), whole)

    def test_speed_is_measured(self):
        demod, _ = self._run(self._signal(), 1600)
        assert demod.estimate_wpm(np.zeros(10)) == pytest.approx(15.0, abs=0.5)

    def test_reset_forgets_the_text(self):
        demod, _ = self._run(self._signal(), 1600)
        demod.reset()
        assert demod.decoded_text == ""
        assert demod._current_letter == ""

    def test_the_table_is_still_exported(self):
        assert MORSE_CODE["...---..."] == "SOS"
        assert MORSE_CODE[".-"] == "A"


# --------------------------------------------------------------------------- #
# MorseDecoder and KeyingDetector
# --------------------------------------------------------------------------- #


class TestSpeeds:
    @pytest.mark.parametrize("wpm", [15, 20, 25])
    def test_clean_keying(self, wpm):
        rate = 1000.0
        morse = MorseDecoder(rate)
        lines = morse.process(enc.morse_keying(MESSAGE, rate, wpm)) + morse.flush()
        assert [m.text for m in lines] == [MESSAGE]
        assert morse.wpm == pytest.approx(wpm, rel=0.03)
        assert lines[0].wpm == pytest.approx(wpm, rel=0.03)
        assert lines[0].protocol is ProtocolType.CW

    @pytest.mark.parametrize("wpm", [15, 20, 25])
    @pytest.mark.parametrize("snr_db", [10.0, 6.0])
    def test_with_noise(self, wpm, snr_db):
        # SNR in the 2 kHz channel: a +/-250 Hz filter would add 6 dB.
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, wpm, offset_hz=30.0)
        for seed in (1, 2):
            text, morse = _decode(_noisy(x, snr_db, seed), rate)
            assert text == [MESSAGE], (wpm, snr_db, seed)
            assert morse.wpm == pytest.approx(wpm, rel=0.08)

    @pytest.mark.parametrize("wpm", [15, 20, 25])
    def test_weak_signal_mostly_right(self, wpm):
        # 4 dB in 2 kHz (about 10 dB in a 500 Hz filter): a letter or two
        # may be lost, the rest reads.
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, wpm)
        ratios = []
        for seed in (7, 8, 9):
            text, _ = _decode(_noisy(x, 4.0, seed), rate)
            ratios.append(
                difflib.SequenceMatcher(None, " ".join(text), MESSAGE).ratio()
            )
        assert min(ratios) > 0.85 and np.mean(ratios) > 0.9

    @pytest.mark.parametrize("block", [66, 333, 2000])
    def test_noisy_stream_in_blocks_decodes_like_whole(self, block):
        rate = 2000.0
        x = _noisy(enc.cw_baseband(MESSAGE, rate, 20), 4.0, 11)
        whole, _ = _decode(x, rate)
        assert _decode(x, rate, block)[0] == whole

    def test_noise_alone_decodes_nothing(self):
        rate = 2000.0
        for seed in range(3):
            rng = np.random.default_rng(seed)
            noise = 0.3 * (
                rng.standard_normal(120_000) + 1j * rng.standard_normal(120_000)
            )
            text, morse = _decode(noise, rate)
            assert text == [] and morse.characters == 0, seed

    def test_silence_decodes_nothing(self):
        text, morse = _decode(np.zeros(20_000, dtype=np.complex64), 2000.0)
        assert text == [] and morse.characters == 0

    def test_a_carrier_at_the_start_keys_at_once(self):
        # Nothing before the buffer: a steady carrier is a key down within
        # a few tens of milliseconds, while noise is not.
        rate = 48000.0
        keyer = KeyingDetector(rate)
        key = keyer.process(np.ones(4800, dtype=np.complex64))
        assert key[int(0.04 * rate) :].all()


class TestTextLayout:
    RATE = 1000.0

    def keying(self, *parts):
        """Concatenate keying for (text) parts and (seconds,) silences or
        ('carrier', seconds)."""
        out = []
        for part in parts:
            if isinstance(part, str):
                out.append(enc.morse_keying(part, self.RATE, 20, lead_s=0, tail_s=0))
            elif part[0] == "carrier":
                out.append(np.ones(int(part[1] * self.RATE)))
            else:
                out.append(np.zeros(int(part[0] * self.RATE)))
        return np.concatenate(out)

    def lines(self, keying):
        morse = MorseDecoder(self.RATE)
        return [m.text for m in morse.process(keying)], morse

    def test_a_long_pause_ends_the_line(self):
        lines, morse = self.lines(
            self.keying((0.5,), "CQ DE TEST", (3.0,), "K", (0.3,))
        )
        assert lines == ["CQ DE TEST"]
        assert morse.line == "K"
        assert morse.text == "CQ DE TEST\nK"

    def test_a_long_carrier_ends_the_line_and_is_not_a_letter(self):
        lines, morse = self.lines(
            self.keying(
                (0.5,), "VVV DE SDR", (0.2,), ("carrier", 2.4), (0.4,), "VVV", (0.2,)
            )
        )
        assert lines == ["VVV DE SDR"]
        assert morse.line == "VVV"

    def test_long_lines_wrap_at_a_word(self):
        words = " ".join(["PARIS"] * 14)
        morse = MorseDecoder(self.RATE, line_chars=30)
        lines = morse.process(self.keying((0.5,), words, (3.0,)))
        assert [len(m.text) <= 35 for m in lines] == [True] * len(lines)
        assert " ".join(m.text for m in lines) == words

    def test_first_letters_wait_for_the_speed(self):
        # "TEST" alone has too few marks to measure the speed until the
        # pause after it; nothing is lost meanwhile.
        lines, _morse = self.lines(self.keying((0.5,), "TEST", (3.0,)))
        assert lines == ["TEST"]

    def test_glitches_are_not_elements(self):
        keying = self.keying((0.5,), "PARIS", (0.42,), "PARIS", (3.0,))
        # Spikes of 4 ms (shorter than a quarter dit at 20 WPM) in the word
        # gap (7 dits, from 3.08 s) and in a letter gap of the second word.
        word_gap = 500 + 43 * 60
        for start in (word_gap + 100, word_gap + 300, word_gap + 420 + 11 * 60 + 60):
            keying[start : start + 4] = 1.0
        lines, _morse = self.lines(keying)
        assert lines == ["PARIS PARIS"]

    def test_unknown_codes_show_a_star(self):
        morse = MorseDecoder(self.RATE)
        dit = int(0.06 * self.RATE)
        # Eight dits in a row: no Morse character.
        key = np.concatenate(
            [np.zeros(500)]
            + [np.r_[np.ones(dit), np.zeros(dit)]] * 8
            + [np.zeros(3000)]
        )
        lines = morse.process(key) + morse.flush()
        assert [m.text for m in lines] == ["*"]


# --------------------------------------------------------------------------- #
# The 'CW (Morse)' live decoder
# --------------------------------------------------------------------------- #


class TestCWLiveDecoder:
    # Long messages at 240 kS/s (a whole message at 2.4 MS/s is hundreds of
    # MB); test_at_2_4_ms_s checks the full rate with a short one.
    RATE = 240e3

    def span(self, offset, text=MESSAGE, wpm=20, seconds_extra=1.0, rate=None):
        """A CW signal at ``offset`` in the span, with noise and a much
        stronger FM station 30 kHz away."""
        rng = np.random.default_rng(4)
        rate = rate or self.RATE
        cw = enc.cw_baseband(text, rate, wpm, offset_hz=offset, amplitude=0.05)
        n = len(cw) + int(seconds_extra * rate)
        x = np.zeros(n, dtype=np.complex64)
        x[: len(cw)] = cw
        t = np.arange(n) / rate
        phase = 2 * np.pi * (offset + 30e3) * t + 2 * np.sin(2e3 * t)
        x.real += 0.5 * np.cos(phase).astype(np.float32)
        x.imag += 0.5 * np.sin(phase).astype(np.float32)
        noise = rng.standard_normal(2 * n, dtype=np.float32).view(np.complex64)
        x += np.float32(0.002) * noise
        return x

    def decode(self, x, offset, block=None, rate=None):
        decoder = create_live_decoder("CW (Morse)", rate or self.RATE, offset)
        step = block or len(x)
        messages = []
        for i in range(0, len(x), step):
            messages += decoder.process(x[i : i + step])
        return decoder, messages

    def test_registered(self):
        assert live_decoder_class("CW (Morse)") is CWLiveDecoder
        assert live_decoder_class("cw") is CWLiveDecoder
        assert live_decoder_class("morse") is CWLiveDecoder

    def test_decodes_at_the_vfo_in_33_ms_blocks_like_whole(self):
        offset = -80e3
        x = self.span(offset, seconds_extra=3.0)  # the pause ends the line
        whole, messages = self.decode(x, offset)
        _streamed, streamed = self.decode(x, offset, block=8_000)
        rows = [describe_message(m) for m in messages]
        assert rows == [("20 WPM", MESSAGE)]
        assert [describe_message(m) for m in streamed] == rows
        status = whole.status
        assert status.detail == "20 WPM"
        assert status.valid == len(MESSAGE.replace(" ", ""))
        assert status.last_time is not None

    def test_at_2_4_ms_s(self):
        offset, rate = 600e3, 2.4e6
        x = self.span(offset, "TEST DE N0CALL", wpm=30, seconds_extra=0.0, rate=rate)
        decoder, _messages = self.decode(x, offset, block=80_000, rate=rate)
        status = decoder.status
        assert (status.live_text, status.detail) == ("TEST DE N0CALL", "30 WPM")

    def test_line_in_progress_is_the_live_text(self):
        offset = 50e3
        x = self.span(offset, "CQ CQ", seconds_extra=0.3)
        decoder, messages = self.decode(x, offset, block=8_000)
        assert messages == []
        assert decoder.status.live_text == "CQ CQ"

    def test_retuning_ends_the_line(self):
        offset = 50e3
        x = self.span(offset, "CQ CQ", seconds_extra=0.3)
        decoder, _ = self.decode(x, offset, block=8_000)
        decoder.set_offset(offset + 10e3)  # another channel
        messages = decoder.process(np.zeros(8_000, dtype=np.complex64))
        assert [m.text for m in messages] == ["CQ CQ"]
        assert decoder.status.live_text == ""

    def test_message_rows(self):
        msg = MorseMessage(
            protocol=ProtocolType.CW,
            timestamp=0.0,
            raw_bits=b"",
            valid=True,
            text="CQ DE N0CALL",
            wpm=18.4,
        )
        assert describe_message(msg) == ("18 WPM", "CQ DE N0CALL")


class TestDemoBeacon:
    """The demo device's beacon on 99.82 MHz ('VVV DE SDR', 15 WPM)."""

    def test_decodes_in_33_ms_blocks_like_whole(self):
        from sdr_module.gui.device_dialog import MockDevice

        # 0.25 MS/s keeps 22 s of samples small (the decoder works the same
        # at any rate: it picks its 2 kHz channel out).
        rate = 0.25e6
        dev = MockDevice()
        dev.set_sample_rate(rate)
        dev.set_frequency(99.9e6)
        dev.start_rx()
        dev._stream_pos = int(2.0 * rate)  # mid-way through a cycle
        block = int(rate / 30)  # 33 ms
        iq = np.concatenate(
            [dev.read_samples(block) for _ in range(int(22 * rate / block))]
        )
        offset = 99.82e6 - 99.9e6
        results = []
        for step in (len(iq), block):
            decoder = create_live_decoder("CW (Morse)", rate, offset)
            lines = []
            for i in range(0, len(iq), step):
                lines += decoder.process(iq[i : i + step])
            results.append([m.text for m in lines])
        assert results[0] == results[1]
        assert "VVV DE SDR" in results[0]
        assert decoder.status.detail == "15 WPM"
