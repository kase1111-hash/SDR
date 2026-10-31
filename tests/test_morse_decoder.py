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


def _noisy(x, snr_db, seed, rate=2000.0):
    """``x`` (unit carrier) plus noise ``snr_db`` below it in 500 Hz: white
    noise band-limited to +/-250 Hz, as a CW channel filter leaves it."""
    rng = np.random.default_rng(seed)
    n = len(x)
    spectrum = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    spectrum[np.abs(np.fft.fftfreq(n, 1.0 / rate)) > 250.0] = 0.0
    noise = np.fft.ifft(spectrum)
    noise /= np.sqrt(np.mean(np.abs(noise) ** 2))
    return (np.sqrt(10 ** (snr_db / 10)) * x + noise).astype(np.complex64)


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

    @pytest.mark.parametrize("text", ["HI", "EEE", "TOM", "PARIS"])
    def test_a_short_complete_capture(self, text):
        # DEC-11: decode_morse waited for 6 marks of both kinds (or 12, or a
        # 1.5 s pause) before decoding anything, so one complete capture of
        # "HI", "EEE" or "TOM" with half a second of silence after it gave
        # "" (the old decoder gave "H", "EE" and "TO").
        rate = 8000.0
        key = enc.morse_keying(text, rate, 20, edge_s=0.0, lead_s=0.5, tail_s=0.5)
        demod = CWDemodulator(rate)
        assert demod.decode_morse(demod.detect_keying(key)) == text
        # A capture that stops right after its last letter: flush() ends it.
        key = enc.morse_keying(text, rate, 20, edge_s=0.0, lead_s=0.5, tail_s=0.05)
        demod = CWDemodulator(rate)
        returned = demod.decode_morse(demod.detect_keying(key))
        returned += demod.flush()
        assert returned.strip() == text
        assert demod.decoded_text == returned
        assert demod.flush() == ""


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

    @pytest.mark.parametrize("wpm", [5, 6, 7.5, 8, 8.5, 12, 40, 60])
    def test_slow_and_fast_code_in_blocks(self, wpm):
        # DEC-03: below about 8.5 WPM the dashes (0.45 s and longer) were
        # taken for a tuning carrier before the speed was known, so only
        # the dits were measured, as dashes: "CQ CQ DE W1AW K" at 5 WPM
        # read "T | T | T ..." at 15 WPM.
        rate = 2000.0
        text = "CQ CQ DE W1AW K"
        key = enc.morse_keying(text, rate, wpm, lead_s=1.0, tail_s=4.0)
        morse = MorseDecoder(rate)
        lines = []
        for i in range(0, len(key), int(rate / 30)):
            lines += morse.process(key[i : i + int(rate / 30)])
        lines += morse.flush()
        assert [m.text for m in lines] == [text]
        assert morse.wpm == pytest.approx(wpm, rel=0.05)

    @pytest.mark.parametrize("wpm", [5, 6, 8, 10])
    def test_slow_code_that_starts_with_dits(self, wpm):
        # Before the speed was known, marks all of one kind were taken for
        # dits or dashes by the default 20 WPM alone, so slow dits (0.15 s
        # and longer) were dashes. A pause of 7 of those "dits" (shorter
        # than a letter gap) then ended the transmission after the first
        # letter: "SOS DE W1AW" at 8 WPM read "TTT OS DE W1AW", and twelve
        # dits were twelve dashes ("HI HI" read "TTTT TT TTTT TT"). The
        # gaps between the marks tell: a letter's elements are a dit apart.
        rate = 2000.0
        for text in ("SOS DE W1AW", "HI HI DE W1AW", "5NN TU", "IS IT OK"):
            key = enc.morse_keying(text, rate, wpm, lead_s=1.0, tail_s=4.0)
            morse = MorseDecoder(rate)
            lines = []
            for i in range(0, len(key), int(rate / 30)):
                lines += morse.process(key[i : i + int(rate / 30)])
            lines += morse.flush()
            assert [m.text for m in lines] == [text], text
            assert morse.wpm == pytest.approx(wpm, rel=0.05), text

    @pytest.mark.parametrize("wpm", [5, 6, 8, 10, 12])
    def test_a_transmission_of_only_dashes(self, wpm):
        # Leftover 27: "TTT" at 8 WPM read "S". Marks all of one kind with
        # gaps as long as themselves were taken for dits; at the slowest
        # speed (5 WPM) a dit is 0.24 s, and these marks (0.3-0.72 s) were
        # measured against that clamped dit, which made them dits too.
        rate = 2000.0
        key = enc.morse_keying("TTT", rate, wpm, lead_s=1.0, tail_s=4.0)
        morse = MorseDecoder(rate)
        lines = []
        for i in range(0, len(key), int(rate / 30)):
            lines += morse.process(key[i : i + int(rate / 30)])
        lines += morse.flush()
        assert [m.text for m in lines] == ["TTT"]
        assert morse.wpm == pytest.approx(wpm, rel=0.05)

    @pytest.mark.parametrize("text", ["SOS DE N0CALL K", "HI HI DE N0CALL"])
    def test_slow_code_that_starts_with_dits_in_noise(self, text):
        rate = 2000.0
        x = enc.cw_baseband(text, rate, 8, offset_hz=30.0)
        x = np.concatenate([np.zeros(int(rate)), x, np.zeros(int(3 * rate))])
        for seed in (1, 2, 3):
            lines, morse = _decode(_noisy(x, 12.0, seed), rate, block=66)
            assert lines == [text], seed
            assert morse.wpm == pytest.approx(8, rel=0.05), seed

    @pytest.mark.parametrize("wpm", [8, 20])
    def test_a_tuning_carrier_before_the_first_letters(self, wpm):
        # A beacon's carrier of about a second, then its text: the carrier
        # is not a letter and does not count in the speed.
        rate = 1000.0
        key = np.concatenate(
            [
                np.zeros(500),
                np.ones(1100),
                enc.morse_keying("VVV DE SDR", rate, wpm, lead_s=0.6, tail_s=3.0),
            ]
        )
        morse = MorseDecoder(rate)
        lines = morse.process(key) + morse.flush()
        assert [m.text for m in lines] == ["VVV DE SDR"]
        assert morse.wpm == pytest.approx(wpm, rel=0.05)

    @pytest.mark.parametrize("wpm", [5, 8])
    def test_slow_code_through_the_live_decoder(self, wpm):
        # DEC-03 on the whole path: a carrier at the tuned frequency, 30 dB
        # above the noise in 500 Hz, in 33 ms blocks.
        rate = 48e3
        text = "CQ DE W1AW K"
        x = enc.cw_baseband(text, rate, wpm, amplitude=0.1)
        x = np.concatenate([x, np.zeros(int(3 * rate), dtype=np.complex64)])
        rng = np.random.default_rng(3)
        sigma = 0.1 * np.sqrt(rate / 500.0 / 10**3.0 / 2)
        x += (
            (sigma * rng.standard_normal(2 * len(x)))
            .view(np.complex128)
            .astype(np.complex64)
        )
        decoder = create_live_decoder("CW (Morse)", rate)
        lines = []
        for i in range(0, len(x), int(rate / 30)):
            lines += decoder.process(x[i : i + int(rate / 30)])
        lines += decoder.flush()
        assert [m.text for m in lines] == [text]
        assert decoder.status.detail == f"{wpm} WPM"

    @pytest.mark.parametrize("wpm", [15, 20, 25])
    @pytest.mark.parametrize("snr_db", [12.0, 8.0])
    def test_with_noise(self, wpm, snr_db):
        # SNR in 500 Hz (the CW channel).
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, wpm, offset_hz=30.0)
        for seed in (1, 2):
            text, morse = _decode(_noisy(x, snr_db, seed), rate)
            assert text == [MESSAGE], (wpm, snr_db, seed)
            assert morse.wpm == pytest.approx(wpm, rel=0.08)

    @pytest.mark.parametrize("wpm", [15, 20, 25])
    def test_weak_signal_mostly_right(self, wpm):
        # 6 dB in 500 Hz: a letter or two may be lost, the rest reads.
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, wpm)
        ratios = []
        for seed in (7, 8, 9):
            text, _ = _decode(_noisy(x, 6.0, seed), rate)
            ratios.append(
                difflib.SequenceMatcher(None, " ".join(text), MESSAGE).ratio()
            )
        assert min(ratios) > 0.6 and np.mean(ratios) > 0.8

    @pytest.mark.parametrize("block", [66, 333, 2000])
    def test_noisy_stream_in_blocks_decodes_like_whole(self, block):
        rate = 2000.0
        x = _noisy(enc.cw_baseband(MESSAGE, rate, 20), 6.0, 11)
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
            # As the CW channel leaves it, band-limited.
            text, morse = _decode(_noisy(np.zeros(120_000), 0.0, seed), rate)
            assert text == [] and morse.characters == 0, seed

    @staticmethod
    def _shaped_noise(seconds, level, seed, rate=2000.0):
        """Noise as the CW channel leaves it (band-limited to +/-250 Hz),
        its amplitude shaped by ``level`` (an array over time)."""
        rng = np.random.default_rng(seed)
        n = int(seconds * rate)
        white = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * level
        spectrum = np.fft.fft(white)
        spectrum[np.abs(np.fft.fftfreq(n, 1.0 / rate)) > 250.0] = 0.0
        return np.fft.ifft(spectrum).astype(np.complex64)

    @staticmethod
    def _crashes(seconds, seed, rate=2000.0):
        """Static crashes: 12 dB bursts of 20-120 ms about once a second."""
        rng = np.random.default_rng(seed)
        level = np.ones(int(seconds * rate))
        t = rng.uniform(0.5, 1.5)
        while t < seconds:
            length = rng.uniform(0.02, 0.12)
            level[int(t * rate) : int((t + length) * rate)] = 10 ** (12 / 20)
            t += length + rng.uniform(0.5, 1.5)
        return level

    def test_static_crashes_decode_next_to_nothing(self):
        # DEC-10: each crash of static keyed one mark, a gap of 1.5 s then
        # took whatever was queued for a transmission, and every crash
        # after that was an E or a T: about one character a second. Now a
        # lone E or T is not shown (but in code spaced letter by letter); a
        # crash that keys twice can still make an I or an N now and then.
        rate = 2000.0
        for seed in (1, 2, 3):
            x = self._shaped_noise(90.0, self._crashes(90.0, seed), seed)
            _text, morse = _decode(x, rate, block=66)
            assert morse.characters <= 5, seed

    def test_a_slow_swing_of_the_noise_decodes_nothing(self):
        # DEC-10: band noise rising and falling 4 dB over 6 s (fading, a
        # pumping AGC) outran the floor and keyed: 70 characters in 5 min.
        rate = 2000.0
        t = np.arange(int(180 * rate)) / rate
        level = 10 ** (4.0 * np.sin(2 * np.pi * t / 6.0) / 20)
        for seed in (1, 2):
            text, morse = _decode(self._shaped_noise(180.0, level, seed), rate, 66)
            assert text == [] and morse.characters == 0, seed

    def test_crashes_after_a_transmission_add_nothing(self):
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, 20, offset_hz=30.0)
        after = self._crashes(40.0, 3)
        level = np.concatenate([np.ones(len(x)), after])
        noise = self._shaped_noise(len(level) / rate, level, 3)
        signal = np.concatenate([x, np.zeros(len(after), dtype=np.complex64)])
        text, morse = _decode(np.sqrt(10**1.2) * signal + noise, rate, block=66)
        assert text == [MESSAGE]
        assert morse.characters == len(MESSAGE.replace(" ", ""))

    @pytest.mark.parametrize("seed", [2, 3, 4, 5, 6, 7, 8, 9])
    def test_crashes_just_before_and_after_a_message_add_nothing(self, seed):
        # Crashes of static less than a line's pause from the words key
        # lone Es and Ts next to them, in the message's line: next to
        # words of several letters they are static, not letters.
        rate = 2000.0
        x = enc.cw_baseband(MESSAGE, rate, 20, offset_hz=30.0)
        lead, tail = int(2.0 * rate), int(3.0 * rate)
        signal = np.concatenate(
            [np.zeros(lead, np.complex64), x, np.zeros(tail, np.complex64)]
        )
        level = self._crashes(len(signal) / rate, seed)[: len(signal)]
        level[lead - int(0.4 * rate) : lead + len(x)] = 1.0  # none in the text
        noise = self._shaped_noise(len(signal) / rate, level, seed)
        text, morse = _decode(np.sqrt(10**1.2) * signal + noise, rate, block=66)
        assert text == [MESSAGE], seed

    def test_a_blip_before_a_long_pause_is_not_a_letter(self):
        # One or two marks and then silence: noise, not a transmission.
        rate = 1000.0
        for marks in ([60], [60, 180]):
            key = np.zeros(500)
            for length in marks:
                key = np.concatenate([key, np.ones(length), np.zeros(60)])
            morse = MorseDecoder(rate)
            lines = morse.process(np.concatenate([key, np.zeros(3000)]))
            assert lines + morse.flush() == [] and morse.characters == 0, marks

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

    def test_flush_keeps_a_mark_cut_off_by_the_end(self):
        # A recording that stops while the key is down: its last dash
        # still counts ("TEST" ends in T, a dash).
        keying = self.keying((0.5,), "TEST")
        morse = MorseDecoder(self.RATE)
        morse.process(keying[:-20])
        assert [m.text for m in morse.flush()] == ["TEST"]

    def spaced(self, text, wpm, letter_gap, word_gap, tail_s=3.0):
        """Keying of ``text`` at ``wpm`` with its letters ``letter_gap``
        dits apart and its words ``word_gap`` dits (wider than the usual 3
        and 7: Farnsworth code practice, or a loose hand)."""
        codes = {char: code for code, char in MORSE_CODE.items() if len(char) == 1}
        dit = 1.2 / wpm * self.RATE
        out = [np.zeros(int(0.5 * self.RATE))]
        for w, word in enumerate(text.split()):
            if w:
                out.append(np.zeros(int(word_gap * dit)))
            for c, char in enumerate(word):
                if c:
                    out.append(np.zeros(int(letter_gap * dit)))
                for e, element in enumerate(codes[char]):
                    if e:
                        out.append(np.zeros(int(dit)))
                    out.append(np.ones(int((1 if element == "." else 3) * dit)))
        out.append(np.zeros(int(tail_s * self.RATE)))
        return np.concatenate(out)

    @pytest.mark.parametrize(
        "letter_gap, word_gap",
        [(5.5, 11.0), (9.3, 21.7)],  # a loose hand; 18 WPM letters at 10 WPM
    )
    def test_letters_spaced_wider_keep_their_es_and_ts(self, letter_gap, word_gap):
        # Letters spaced as wide as words are each a word to the decoder,
        # and every word that was a lone E or T was dropped as a crash of
        # static: "THE CAT SAT" read "H C A S A". Once the code reads
        # letter by letter they show (as do the Es and Ts that end it).
        morse = MorseDecoder(self.RATE)
        key = self.spaced("CQ THE CAT SAT MEET THE TEAM", 18, letter_gap, word_gap)
        lines = morse.process(key) + morse.flush()
        assert [m.text for m in lines] == [
            "C Q T H E C A T S A T M E E T T H E T E A M"
        ]
        key = self.spaced("CQ DE MEET", 18, letter_gap, word_gap)
        assert [m.text for m in morse.process(key) + morse.flush()] == [
            "C Q D E M E E T"
        ]

    def test_a_blip_after_letters_is_not_an_e(self):
        # Noise keys short blips once a signal stops: a third of a dit
        # alone, even after letters spaced one by one, is not an E.
        dit = 1.2 / 18 * self.RATE
        keying = np.concatenate(
            [
                self.spaced("CQ DE", 18, 9.3, 21.7, tail_s=0.0),
                np.zeros(int(12 * dit)),
                np.ones(int(dit / 3)),
                np.zeros(int(3.0 * self.RATE)),
            ]
        )
        lines, morse = self.lines(keying)
        assert lines + [m.text for m in morse.flush()] == ["C Q D E"]

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

    def test_flush_ends_the_line_and_counts_all_of_it(self):
        # The stream stops during the last dash: flush() gives the line,
        # and the status counts its last letter too.
        offset = 50e3
        key = enc.morse_keying("CQ CQ", self.RATE, 20)
        end = np.nonzero(key > 0.5)[0][-1] - int(0.02 * self.RATE)
        x = self.span(offset, "CQ CQ", seconds_extra=0.0)[:end]
        decoder, messages = self.decode(x, offset, block=8_000)
        assert decoder.status.live_text == "CQ C"
        assert messages == []
        assert [m.text for m in decoder.flush()] == ["CQ CQ"]
        assert decoder.status.valid == 4

    def test_retuning_ends_the_line(self):
        offset = 50e3
        x = self.span(offset, "CQ CQ", seconds_extra=0.3)
        decoder, _ = self.decode(x, offset, block=8_000)
        decoder.set_offset(offset + 10e3)  # another channel
        messages = decoder.process(np.zeros(8_000, dtype=np.complex64))
        assert [m.text for m in messages] == ["CQ CQ"]
        assert decoder.status.live_text == ""
        assert decoder.status.valid == 4  # its last letter counted too

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
