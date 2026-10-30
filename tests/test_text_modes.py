"""The text-mode signals behind Transmit Text (protocols/text_modes.py):
character sets and what is reported, exact timing, the audio, the baseband
and the streamed transmit source (on the carrier and on FM), the occupied
bandwidth the legality check uses, the symbol-rate rules and the exports.
Signals are read back with tests/text_demod.py."""

from __future__ import annotations

import json
import time
import wave

import numpy as np
import pytest

from sdr_module.protocols import text_modes as tm
from sdr_module.protocols.text_modes import (
    EMISSION_DIRECT,
    EMISSION_FM,
    MODE_ASCII,
    MODE_CW,
    MODE_PSK31,
    MODE_RTTY,
    TextSettings,
    TextSignal,
)
from tests import text_demod as demod

RATE = 48000.0


def _signal(text, mode, **values):
    return TextSignal(text, TextSettings.for_mode(mode, **values))


def _varicode_table():
    return {code: char for char, code in tm.VARICODE.items()}


def _read_all(source, block=65536):
    parts = []
    while True:
        chunk = source.read(block)
        if not len(chunk):
            break
        parts.append(chunk)
    return np.concatenate(parts) if parts else np.zeros(0, np.complex64)


def _fm_audio(iq, rate):
    """FM-demodulated audio of ``iq`` (Hz of deviation per sample)."""
    return demod.instantaneous_frequency(iq, rate)


class TestSettings:
    def test_mode_defaults(self):
        cw = TextSettings.for_mode("morse")
        assert (cw.mode, cw.wpm, cw.audio_hz) == (MODE_CW, 20.0, 700.0)
        rtty = TextSettings.for_mode("rtty")
        assert (rtty.baud, rtty.shift_hz, rtty.tones()) == (
            45.45,
            170.0,
            (2125.0, 2295.0),
        )
        psk = TextSettings.for_mode("psk31")
        assert (psk.baud, psk.audio_hz) == (31.25, 1000.0)
        ascii_ = TextSettings.for_mode("ascii fsk")
        assert ascii_.tones() == (1070.0, 1270.0)  # Bell 103 originate
        assert ascii_.baud == 300.0

    def test_values_are_kept_in_range(self):
        s = TextSettings(MODE_RTTY, wpm=1, baud=1e6, shift_hz=5, audio_hz=1e5).clamped()
        assert (s.wpm, s.baud, s.shift_hz, s.audio_hz) == (5.0, 1200.0, 20.0, 3500.0)
        assert TextSettings(MODE_PSK31, baud=100).clamped().baud == 31.25

    def test_describe(self):
        assert (
            TextSettings.for_mode(MODE_RTTY).describe() == "RTTY 45.45 Bd, 170 Hz shift"
        )
        assert TextSettings.for_mode(MODE_CW, wpm=25).describe() == "CW 25 WPM"
        assert TextSettings.for_mode(MODE_PSK31).describe() == "PSK31"
        assert (
            TextSettings.for_mode(MODE_ASCII).describe()
            == "ASCII FSK 300 Bd, 200 Hz shift"
        )

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            tm.normalize_mode("olivia")


class TestCharacterSets:
    def test_rtty_shifts(self):
        codes, check = tm.ita2_codes("N0CALL 73")
        ltrs, figs, space = 0b11111, 0b11011, 0b00100
        letters = {c: i for i, c in enumerate(tm.ITA2_LETTERS) if c}
        figures = {c: i for i, c in enumerate(tm.ITA2_FIGURES_US) if c}
        assert codes == [
            ltrs, letters["N"], figs, figures["0"], ltrs, letters["C"],
            letters["A"], letters["L"], letters["L"], space, figs,
            figures["7"], figures["3"],
        ]  # fmt: skip
        assert check.sent == "N0CALL 73" and check.characters == 9

    def test_rtty_repeats_figures_after_a_space(self):
        codes, _ = tm.ita2_codes("73 73")
        figs = 0b11011
        assert codes.count(figs) == 2
        # (and not after a space in letters)
        codes, _ = tm.ita2_codes("AB CD")
        assert codes.count(0b11111) == 1

    def test_rtty_new_line_is_cr_lf(self):
        codes, check = tm.ita2_codes("A\nB")
        assert codes[2:4] == [0b01000, 0b00010]
        assert check.sent == "A\nB" and check.characters == 3

    def test_rtty_figures_variants(self):
        assert tm.check_text("$5", MODE_RTTY).ok
        assert tm.check_text("1+1=2", MODE_RTTY).unsupported == {"+": 1, "=": 1}
        check = tm.ita2_codes("1+1=2", figures="ita2")[1]
        assert check.ok
        assert tm.ita2_codes("$5", figures="ita2")[1].unsupported == {"$": 1}

    def test_tab_is_a_space_in_rtty_and_cw(self):
        assert tm.check_text("A\tB", MODE_RTTY).sent == "A B"
        assert tm.check_text("A\tB", MODE_CW).sent == "A B"
        assert tm.check_text("A\tB", MODE_PSK31).sent == "A\tB"

    def test_typographic_characters_are_sent_as_ascii(self):
        check = tm.check_text("“Hi” — it’s 5…", MODE_PSK31)
        assert check.ok
        assert check.sent == '"Hi" - it\'s 5...'

    def test_unsupported_counts_in_order(self):
        check = tm.check_text("50% @ 1 % é", MODE_RTTY)
        assert list(check.unsupported.items()) == [("%", 2), ("@", 1), ("é", 1)]
        assert check.skipped == 4
        assert check.describe_unsupported() == "% (2x), @, é"
        assert tm.check_text("a\x01b", MODE_CW).describe_unsupported() == (
            "control character 0x01"
        )

    def test_cw_prosigns(self):
        words, check = tm.morse_elements("73 <SK> <AR>")
        assert words[1] == ["...-.-"] and words[2] == [".-.-."]
        assert check.sent == "73 <SK> <AR>" and check.characters == 4
        # An unknown or unclosed bracket is reported, its letters sent.
        check = tm.check_text("<SK", MODE_CW)
        assert check.unsupported == {"<": 1} and check.sent == "SK"
        assert tm.check_text("<#>", MODE_CW).unsupported == {"<": 1, "#": 1, ">": 1}

    def test_cw_punctuation_from_the_decoders_table(self):
        check = tm.check_text(
            "CQ? 5NN. TU! A/B (X) 'Q' \"Y\" & : ; = + - _ $ @", MODE_CW
        )
        assert check.ok, check.unsupported

    def test_psk31_takes_every_ascii_character(self):
        text = "".join(chr(i) for i in range(32, 127)) + "\t"
        assert tm.check_text(text, MODE_PSK31).ok
        assert tm.check_text(text, MODE_ASCII).ok
        assert tm.check_text("é", MODE_ASCII).unsupported == {"é": 1}

    def test_varicode_structure(self):
        codes = list(tm.VARICODE.values())
        assert len(codes) == 128 and len(set(codes)) == 128
        for code in codes:
            assert code[0] == "1" and code[-1] == "1" and "00" not in code


class TestTiming:
    def test_cw_paris_timing(self):
        a = _signal("PARIS", MODE_CW, wpm=20)
        b = _signal("PARIS PARIS", MODE_CW, wpm=20)
        # One PARIS plus its word gap is 50 dits; 20 WPM -> 3 s.
        assert b.duration_s - a.duration_s == pytest.approx(3.0, abs=1e-9)

    def test_rtty_character_time(self):
        a = _signal("RY" * 5, MODE_RTTY)
        b = _signal("RY" * 15, MODE_RTTY)
        assert (b.duration_s - a.duration_s) / 20 == pytest.approx(7.5 / 45.45)

    def test_psk31_symbol_time(self):
        a = _signal("e", MODE_PSK31)
        b = _signal("ee", MODE_PSK31)
        # "e" is 11 then 00: four symbols of 32 ms.
        assert b.duration_s - a.duration_s == pytest.approx(4 * 0.032)

    def test_ascii_frame_time(self):
        a = _signal("A", MODE_ASCII, baud=110)
        b = _signal("AB", MODE_ASCII, baud=110)
        assert b.duration_s - a.duration_s == pytest.approx(10 / 110.0)

    def test_empty(self):
        assert _signal("@@", MODE_RTTY).empty
        assert _signal("   ", MODE_CW).empty
        assert not _signal("e", MODE_PSK31).empty


class TestRoundTrips:
    TEXT = "CQ CQ DE N0CALL 599 TU"

    def test_baseband(self):
        rtty = _signal(self.TEXT, MODE_RTTY)
        assert demod.rtty_decode(rtty.baseband(rtty.times(RATE)), RATE) == self.TEXT
        psk = _signal("Hello de N0CALL 73", MODE_PSK31)
        iq = psk.baseband(psk.times(RATE))
        assert demod.psk31_decode(iq, RATE, _varicode_table()) == "Hello de N0CALL 73"
        asc = _signal("Hello, 8N1!", MODE_ASCII)
        iq = asc.baseband(asc.times(RATE))
        assert demod.ascii_decode(iq, RATE, 0.0, 300.0) == b"Hello, 8N1!"
        cw = _signal(self.TEXT, MODE_CW)
        assert demod.cw_decode(cw.baseband(cw.times(RATE)), RATE) == self.TEXT

    def test_audio_tones_sit_where_set(self):
        rtty = _signal(self.TEXT, MODE_RTTY)
        audio = rtty.audio(rtty.times(RATE))
        low, high = demod.tone_frequencies(audio, RATE)
        assert (low, high) == (pytest.approx(2125, abs=2), pytest.approx(2295, abs=2))
        # Mark is the higher audio tone (USB), so the text decodes with mark up.
        iq = demod.analytic(audio)
        assert demod.rtty_decode(iq, RATE, center_hz=2210.0) == self.TEXT
        psk = _signal("tone test", MODE_PSK31, audio_hz=1500.0)
        iq = demod.analytic(psk.audio(psk.times(RATE)))
        assert demod.psk31_decode(iq, RATE, _varicode_table(), 1500.0) == "tone test"
        cw = _signal("VVV", MODE_CW, audio_hz=600.0)
        audio = cw.audio(cw.times(RATE))
        spectrum = np.abs(np.fft.rfft(audio))
        peak = np.fft.rfftfreq(len(audio), 1 / RATE)[np.argmax(spectrum)]
        assert peak == pytest.approx(600.0, abs=2)
        assert np.max(np.abs(audio)) <= 1.0

    def test_cw_decodes_through_the_live_decoder(self):
        from sdr_module.dsp.live_decoders import create_live_decoder

        text = "VVV DE N0CALL/P = QTH? <SK>"
        cw = _signal(text, MODE_CW, wpm=18)
        rate = 48000.0
        iq = cw.baseband(cw.times(rate)).astype(np.complex64)
        pad = np.zeros(int(rate), np.complex64)
        iq = np.concatenate((pad, iq, pad, pad, pad))
        noise = np.random.default_rng(3).normal(0, 0.01, (len(iq), 2))
        iq = iq + (noise[:, 0] + 1j * noise[:, 1]).astype(np.complex64)
        decoder = create_live_decoder("CW (Morse)", rate)
        lines = []
        for start in range(0, len(iq), int(rate * 0.033)):
            lines += decoder.process(iq[start : start + int(rate * 0.033)])
        lines += decoder.flush()
        assert " ".join(m.text for m in lines) == text

    def test_rtty_usos_receiver(self):
        text = "RST 599 599 73"
        rtty = _signal(text, MODE_RTTY)
        iq = rtty.baseband(rtty.times(RATE))
        assert demod.rtty_decode(iq, RATE, usos=True) == text

    def test_new_lines(self):
        rtty = _signal("ABC\nDEF", MODE_RTTY)
        assert demod.rtty_decode(rtty.baseband(rtty.times(RATE)), RATE) == "ABC\r\nDEF"
        psk = _signal("abc\ndef", MODE_PSK31)
        iq = psk.baseband(psk.times(RATE))
        assert demod.psk31_decode(iq, RATE, _varicode_table()) == "abc\r\ndef"


class TestTxSource:
    def test_direct_matches_the_baseband_in_any_block_size(self):
        sig = _signal("RYRY 123", MODE_RTTY)
        a = _read_all(sig.source(RATE, amplitude=1.0), 1000)
        b = _read_all(sig.source(RATE, amplitude=1.0), 65536)
        assert np.array_equal(a, b)
        ref = sig.baseband(sig.times(RATE)).astype(np.complex64)
        assert np.allclose(a, ref, atol=1e-6)

    def test_fm_is_phase_continuous_across_blocks(self):
        sig = _signal("VVV", MODE_CW)
        a = _read_all(sig.source(RATE, EMISSION_FM), 777)
        b = _read_all(sig.source(RATE, EMISSION_FM), 65536)
        assert len(a) == len(b)
        assert np.allclose(a, b, atol=1e-5)

    def test_total_samples_and_duration(self):
        sig = _signal("TEST", MODE_PSK31)
        src = sig.source(2e6)
        assert src.total_samples == round(sig.duration_s * 2e6)
        assert len(_read_all(src)) == src.total_samples
        assert src.read(10).size == 0  # done
        src.reset()
        assert src.position == 0 and src.read(10).size == 10
        fm = sig.source(2e6, EMISSION_FM)
        assert fm.duration_s == pytest.approx(
            sig.duration_s + tm.FM_LEAD_S + tm.FM_TAIL_S, abs=1e-6
        )

    def test_transmit_rate_decodes_after_decimation(self):
        sig = _signal("CQ DE N0CALL", MODE_RTTY)
        iq = _read_all(sig.source(2e6))
        assert np.max(np.abs(iq)) <= 0.9 + 1e-6
        small = iq[: len(iq) // 50 * 50].reshape(-1, 50).mean(axis=1)
        assert demod.rtty_decode(small, 40000.0) == "CQ DE N0CALL"

    def test_fm_emission_carries_the_audio(self):
        sig = _signal("RYRY DE N0CALL", MODE_RTTY)
        src = sig.source(RATE, EMISSION_FM, deviation_hz=2500.0)
        iq = _read_all(src)
        assert np.allclose(np.abs(iq), 0.9, atol=1e-4)  # constant envelope
        audio = _fm_audio(iq, RATE) / 2500.0
        # The FM audio is the AFSK audio (after the unmodulated lead-in);
        # the discriminator's value k is the phase step into sample k + 1.
        lead = int(tm.FM_LEAD_S * RATE)
        expected = sig.audio(sig.times(RATE))
        n = min(len(expected), len(audio) - lead)
        got = audio[lead - 1 : lead - 1 + n]
        assert np.corrcoef(got, expected[:n])[0, 1] > 0.999
        assert np.max(np.abs(audio[: lead - 1])) < 1e-3  # unmodulated lead-in
        assert src.mode == "FM" and sig.tx_mode() == "DATA"

    def test_cw_mcw_and_modes(self):
        from tests.hackrf_fakes import analyse_cw, analyse_mcw

        sig = _signal("TEST", MODE_CW, wpm=20, audio_hz=700.0)
        rate = 200000.0
        direct = _read_all(sig.source(rate))
        assert analyse_cw(direct, rate)["text"] == "TEST"
        mcw = _read_all(sig.source(rate, EMISSION_FM))
        result = analyse_mcw(mcw, rate)
        assert result["text"] == "TEST"
        assert result["tone_hz"] == pytest.approx(700.0, abs=15)
        assert sig.tx_mode(EMISSION_DIRECT) == "CW"
        assert sig.tx_mode(EMISSION_FM) == "FM"

    def test_streaming_is_cheap_at_the_transmit_rate(self):
        sig = _signal("THE QUICK BROWN FOX 0123456789 " * 4, MODE_ASCII)
        for emission in (EMISSION_DIRECT, EMISSION_FM):
            src = sig.source(2e6, emission)
            start = time.perf_counter()
            for _ in range(10):
                src.read(65536)
            per_block = (time.perf_counter() - start) / 10
            # 65536 samples are 33 ms on the air: generating them must take
            # far less (the transmit callback must keep up).
            assert per_block < 0.025, per_block


class TestBandwidth:
    PATTERNS = (
        "RY" * 40,
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789",
        "".join(chr(c) for c in np.random.default_rng(5).integers(32, 127, 80)),
    )

    @pytest.mark.parametrize(
        "mode,values",
        [
            (MODE_RTTY, {}),
            (MODE_RTTY, {"baud": 75.0, "shift_hz": 850.0}),
            (MODE_RTTY, {"baud": 50.0, "shift_hz": 425.0}),
            (MODE_ASCII, {}),
            (MODE_ASCII, {"baud": 110.0, "shift_hz": 850.0}),
            (MODE_ASCII, {"baud": 1200.0, "shift_hz": 1000.0}),
            (MODE_ASCII, {"baud": 50.0, "shift_hz": 200.0}),
            (MODE_PSK31, {}),
            (MODE_CW, {"wpm": 30.0}),
        ],
    )
    def test_declared_bandwidth_holds_99_percent(self, mode, values):
        for text in self.PATTERNS:
            sig = _signal(text, mode, **values)
            if sig.empty:
                continue
            declared = sig.occupied_bandwidth_hz()
            rate = max(8000.0, 8.0 * declared)
            measured = demod.occupied_bandwidth(sig.baseband(sig.times(rate)), rate)
            assert measured <= declared, (text[:10], measured, declared)

    def test_fm_carson(self):
        sig = _signal("RYRY DE N0CALL", MODE_RTTY)
        declared = sig.occupied_bandwidth_hz(EMISSION_FM, 2500.0)
        assert declared == pytest.approx(2 * (2500 + 2295 + 45.45 / 2))
        iq = _read_all(sig.source(RATE, EMISSION_FM))
        assert demod.occupied_bandwidth(iq, RATE) <= declared

    def test_standard_values(self):
        assert _signal("A", MODE_PSK31).occupied_bandwidth_hz() == 60.0
        assert _signal("A", MODE_CW, wpm=20).occupied_bandwidth_hz() == 150.0
        rtty = _signal("A", MODE_RTTY).occupied_bandwidth_hz()
        assert 250.0 < rtty < 320.0


class TestRules:
    def test_symbol_rate_limits(self):
        fast = TextSettings.for_mode(MODE_ASCII, baud=1200.0)
        assert "300 Bd limit" in tm.symbol_rate_problem(14.08e6, fast)
        assert tm.symbol_rate_problem(29.0e6, fast) is None
        assert (
            tm.symbol_rate_problem(14.08e6, TextSettings.for_mode(MODE_ASCII)) is None
        )
        assert (
            tm.symbol_rate_problem(14.08e6, TextSettings.for_mode(MODE_PSK31)) is None
        )
        assert tm.symbol_rate_problem(3.58e6, TextSettings.for_mode(MODE_RTTY)) is None


class TestExports:
    def test_wav_is_mono_audio(self, tmp_path):
        sig = _signal("CQ DE N0CALL", MODE_RTTY)
        path = tmp_path / "rtty.wav"
        reports = []
        frames = tm.write_audio_wav(str(path), sig, report=reports.append)
        with wave.open(str(path)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (
                1,
                2,
                48000,
            )
            assert w.getnframes() == frames == sig.sample_count(48000)
            pcm = np.frombuffer(w.readframes(frames), "<i2") / 32767.0
        assert reports[0] == 0.0 and reports[-1] == pytest.approx(1.0)
        assert np.max(np.abs(pcm)) == pytest.approx(0.5, abs=0.01)
        assert demod.rtty_decode(demod.analytic(pcm), 48000.0, 2210.0) == "CQ DE N0CALL"

    def test_sigmf_keeps_rate_frequency_and_description(self, tmp_path):
        from sdr_module.dsp.recording import load_iq_file

        sig = _signal("Hello SigMF", MODE_PSK31)
        path = tmp_path / "psk.sigmf-data"
        count = tm.write_iq_file(
            str(path), sig.source(RATE), "sigmf", 14.0703e6, "PSK31: Hello SigMF"
        )
        meta = json.loads((tmp_path / "psk.sigmf-meta").read_text())
        assert meta["global"]["core:sample_rate"] == RATE
        assert meta["global"]["core:datatype"] == "cf32_le"
        assert meta["global"]["core:description"] == "PSK31: Hello SigMF"
        assert meta["captures"][0]["core:frequency"] == 14.0703e6
        samples, _meta = load_iq_file(path)
        assert len(samples) == count
        assert demod.psk31_decode(samples, RATE, _varicode_table()) == "Hello SigMF"

    def test_cf32_gets_a_sidecar(self, tmp_path):
        from sdr_module.dsp.recording import find_sidecar

        sig = _signal("ABC", MODE_ASCII)
        path = tmp_path / "ascii.cf32"
        tm.write_iq_file(str(path), sig.source(RATE), "cf32", 144.5e6, "ASCII: ABC")
        iq = np.fromfile(path, np.complex64)
        assert demod.ascii_decode(iq, RATE, 0.0, 300.0) == b"ABC"
        info = find_sidecar(path)
        assert info is not None
        assert info.sample_rate == RATE and info.center_frequency == 144.5e6

    def test_cancel_leaves_no_file(self, tmp_path):
        class Stop(Exception):
            pass

        def report(fraction):
            if fraction > 0.2:
                raise Stop()

        sig = _signal("THE QUICK BROWN FOX " * 3, MODE_RTTY)
        with pytest.raises(Stop):
            tm.write_audio_wav(str(tmp_path / "a.wav"), sig, report=report)
        with pytest.raises(Stop):
            tm.write_iq_file(
                str(tmp_path / "b.sigmf-data"), sig.source(RATE), "sigmf", report=report
            )
        with pytest.raises(Stop):
            tm.write_iq_file(
                str(tmp_path / "c.cf32"), sig.source(RATE), "cf32", report=report
            )
        assert list(tmp_path.iterdir()) == []
