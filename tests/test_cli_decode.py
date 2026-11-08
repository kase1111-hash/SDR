"""``sdr-scan decode`` uses the live decoders and counts only valid messages.

FG-B13 (decode part): the command printed every frame the decoder returned,
CRC failures included ("Decoded 6 AX25 message(s)" from pure noise), as raw
dataclass reprs; and like the window it fed an unfiltered discriminator, so
real AFSK APRS or ACARS never decoded.
"""

import numpy as np

from sdr_module.cli import main
from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

RATE = 240e3


def _save(path, samples, rate=RATE, center=144.39e6):
    save_iq_file(
        path,
        samples.astype(np.complex64),
        sample_rate=rate,
        center_frequency=center,
        sample_format=SampleFormat.FLOAT32,
        file_format=FileFormat.SIGMF,
    )
    return path


def _place(baseband, offset=0.0, lead_s=0.3, tail_s=0.3, rate=RATE):
    lead, tail = int(lead_s * rate), int(tail_s * rate)
    x = np.zeros(lead + len(baseband) + tail, dtype=np.complex64)
    x[lead : lead + len(baseband)] = baseband
    if offset:
        x *= np.exp(2j * np.pi * offset * np.arange(len(x)) / rate).astype(np.complex64)
    return x


def _aprs(info):
    return enc.aprs_baseband("N0CALL-9", "APZ001", info, RATE, ["WIDE1-1"])


def test_counts_only_valid_messages(tmp_path, capsys):
    good = _place(_aprs("!4903.50N/07201.75W>Good packet"))
    bad_frame = bytearray(enc.ax25_frame("N0CALL-9", "APZ001", "Damaged packet"))
    bad_frame[-4] ^= 0x01
    line = enc.hdlc_bits(bytes(bad_frame))
    bad = enc.fm_modulate(enc.APRS_DEVIATION_HZ * enc.afsk_audio(line, RATE), RATE)
    path = _save(tmp_path / "aprs.sigmf-data", np.concatenate([good, _place(bad)]))
    assert main(["decode", "aprs", "--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Decoded 1 APRS message(s) (1 failed their checks; --show-invalid" in out
    # The APRS report is read: position, symbol and comment.
    assert "  [0] N0CALL-9>APZ001  49.0583, -72.0292  ·  Car  ·  Good packet" in out
    assert "AX25Frame(" not in out  # readable lines, not reprs
    assert main(["decode", "aprs", "--input", str(path), "--show-invalid"]) == 0
    out = capsys.readouterr().out
    assert "[invalid] (CRC mismatch)" in out


def test_noise_decodes_nothing(tmp_path, capsys):
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(int(4 * RATE)) + 1j * rng.standard_normal(int(4 * RATE))
    path = _save(tmp_path / "noise.sigmf-data", noise)
    for protocol in ("ax25", "pocsag", "flex", "acars"):
        assert main(["decode", protocol, "--input", str(path)]) == 0
        out = capsys.readouterr().out
        assert f"Decoded 0 {protocol.upper()} message(s)\n" in out, protocol


def test_offset_picks_the_channel(tmp_path, capsys):
    capture = _place(
        enc.acars_baseband(
            enc.acars_block("N601SD", "H1", "CLI TEST", message_number="M01A",
                            flight_id="SD0601"),
            RATE,
        ),
        offset=-40e3,
    )  # fmt: skip
    path = _save(tmp_path / "acars.sigmf-data", capture, center=131.59e6)
    assert main(["decode", "acars", "--input", str(path)]) == 0
    assert "Decoded 0 ACARS message(s)" in capsys.readouterr().out
    assert main(["decode", "acars", "--input", str(path), "--offset", "-40000"]) == 0
    out = capsys.readouterr().out
    assert "Decoded 1 ACARS message(s)" in out
    assert "N601SD  Flight SD0601  ·  H1  ·  CLI TEST" in out  # with the flight


def test_ads_b_needs_2_ms_s(tmp_path, capsys):
    path = _save(tmp_path / "low.sigmf-data", np.zeros(48000, dtype=np.complex64))
    assert main(["decode", "adsb", "--input", str(path)]) == 1
    out = capsys.readouterr().out
    assert "Error: the sample rate is too low for ADS-B" in out
    assert "needs at least 2 MS/s" in out


def test_a_malformed_aprs_packet_does_not_stop_the_listing(tmp_path, capsys):
    # DEC-08: a CRC-valid packet whose compressed speed byte is not ASCII
    # raised OverflowError while it was described: a traceback, exit 1, and
    # neither packet listed.
    from sdr_module.dsp.protocols import AX25Decoder

    good = enc.ax25_frame("N0CALL-9", "APZ001", "PLACEHOLDER", ["WIDE1-1"])
    payload = good[: good.index(b"PLACEHOLDER")] + b"!/5L!!<*e7>!\xffA bad speed"
    fcs = AX25Decoder(9600)._compute_crc(payload)
    bad = payload + bytes([fcs & 0xFF, fcs >> 8])
    line = enc.hdlc_bits(bad)
    bad_bb = enc.fm_modulate(enc.APRS_DEVIATION_HZ * enc.afsk_audio(line, RATE), RATE)
    after = _aprs("!4903.50N/07201.75W>After the bad one")
    path = _save(
        tmp_path / "aprs.sigmf-data", np.concatenate([_place(bad_bb), _place(after)])
    )
    assert main(["decode", "aprs", "--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Decoded 2 APRS message(s)" in out
    assert "  [0] N0CALL-9>APZ001  49.5000, -72.7500" in out
    assert (
        "  [1] N0CALL-9>APZ001  49.0583, -72.0292  ·  Car  ·  After the bad one" in out
    )


class TestAudioRecordings:
    """R7L-2: ``sdr-scan decode`` read a one-channel WAV file at an audio
    rate (a receiver's sound) as I/Q, so only Morse decoded from it, while
    File > Decode Recording decodes such a file's sound as a receiver's
    audio. Now the command does as the window does (the same live decoders,
    fed the file's own samples), and says it read the file as audio."""

    @staticmethod
    def _wav(path, audio, rate):
        from tests.wav_builders import iq_wav

        iq_wav(path, audio, "int", 2, int(rate))
        return path

    def test_receiver_audio_decodes(self, tmp_path, capsys):
        from tests.test_audio_decoding import EXPECTED, _audio

        for name, protocol, rate in (
            ("AX.25/APRS", "aprs", 48000),
            ("POCSAG", "pocsag", 22050),
            ("FLEX", "flex", 48000),
            ("ACARS", "acars", 22050),
        ):
            path = self._wav(tmp_path / f"{protocol}.wav", _audio(name, rate), rate)
            assert main(["decode", protocol, "-i", str(path)]) == 0, protocol
            out = capsys.readouterr().out
            assert "one channel" in out and "as a receiver's audio" in out, out
            detector = "AM detector" if protocol == "acars" else "FM discriminator"
            assert detector in out, out
            expected = EXPECTED[name]
            assert f"Decoded {len(expected)} {protocol.upper()} message(s)" in out
            for address, content in expected:
                assert f"{address}  {content}" in out, (protocol, out)

    def test_a_rate_too_low_is_said(self, tmp_path, capsys):
        from tests.test_audio_decoding import _audio

        path = self._wav(tmp_path / "slow.wav", _audio("POCSAG", 8000)[::2], 4000)
        assert main(["decode", "pocsag", "-i", str(path)]) == 1
        out = capsys.readouterr().out
        assert "Error: POCSAG needs audio at 8 kS/s or more; this is 4 kS/s" in out

    def test_a_header_without_a_rate_takes_the_one_given(self, tmp_path, capsys):
        from tests.test_audio_decoding import _audio
        from tests.wav_builders import encode, fmt_body, write_wav

        path = tmp_path / "rate0.wav"
        write_wav(
            path, fmt_body(1, 0, 2), encode(_audio("AX.25/APRS", 48000), "int", 2)
        )
        assert main(["decode", "aprs", "-i", str(path)]) == 1
        assert "sample rate unknown" in capsys.readouterr().out
        argv = ["decode", "aprs", "-i", str(path), "--sample-rate", "48000"]
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "as a receiver's audio" in out
        assert "Decoded 1 APRS message(s)" in out

    def test_morse_and_fast_captures_are_still_read_as_iq(self, tmp_path, capsys):
        """(Morse decodes at its pitch from the I/Q, as in the window; a
        one-channel file above 200 kHz is an IF or RF sampled directly: the
        radio signal itself, read as I/Q.)"""
        rate = 48000
        tone = np.real(enc.cw_baseband("CQ DE N0CALL", rate, 20, offset_hz=700.0))
        pad = np.zeros(rate // 2)
        path = self._wav(
            tmp_path / "cw.wav", 0.5 * np.concatenate([pad, tone, pad]), rate
        )
        assert main(["decode", "cw", "-i", str(path), "--offset", "-11300"]) == 0
        out = capsys.readouterr().out
        assert "as a receiver's audio" not in out
        assert "CQ DE N0CALL" in out

        fast = 480000
        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>IF", fast, ["WIDE1-1"]
        )
        pad = np.zeros(int(0.2 * fast))
        z = np.concatenate([pad, 0.5 * burst, pad])
        real = np.real(z * np.exp(2j * np.pi * 150e3 * np.arange(len(z)) / fast))
        path = self._wav(tmp_path / "if.wav", real, fast)
        # (I/Q at 240 kS/s centered on 120 kHz: the packet 30 kHz above.)
        assert main(["decode", "aprs", "-i", str(path), "--offset", "30000"]) == 0
        out = capsys.readouterr().out
        assert "as a receiver's audio" not in out
        assert "Decoded 1 APRS message(s)" in out


def test_a_file_that_cant_be_read_is_said_plainly(tmp_path, capsys):
    """files-3: 24-bit samples in 32-bit words whose words fit neither
    layout (R8-11 reads the others: test_samples_in_32_bit_words_decode)
    are refused by the WAV reader (they read as noise before); the command
    says so, as the window does, instead of a traceback."""
    from tests.wav_builders import encode, fmt_body, write_wav

    path = tmp_path / "s24_le.wav"
    packed = encode(0.5 * np.sin(np.arange(800) / 3.0), "int", 3)
    write_wav(path, fmt_body(2, 48000, 4, bits=24), packed)
    for argv in (["decode", "aprs", "-i", str(path)], ["scan", "--input", str(path)]):
        assert main(argv) == 1
        out = capsys.readouterr().out
        assert "Error: s24_le.wav can't be read: its 24-bit samples are stored" in out


def test_samples_in_32_bit_words_decode(tmp_path, capsys):
    """R8-11: 24-bit samples in 32-bit words (ALSA's arecord's S24_LE, in
    each word's low three bytes) are read where the data shows them: an
    APRS packet decodes, as from packed samples (they were refused)."""
    from tests.wav_builders import fmt_body, full_scale, integers, interleave, write_wav

    x = _place(_aprs("!4903.50N/07201.75W>From S24_LE"), offset=-50e3)
    ints = np.rint(interleave(0.5 * x) * full_scale(3)).astype(np.int64)
    path = tmp_path / "aprs_s24_le.wav"
    write_wav(path, fmt_body(2, int(RATE), 4, bits=24), integers(ints, 4))
    assert main(["decode", "aprs", "-i", str(path), "--offset", "-50000"]) == 0
    out = capsys.readouterr().out
    assert "Decoded 1 APRS message(s)" in out, out
    assert "From S24_LE" in out


def test_a_one_channel_files_rate_is_its_own_read_as_iq_too(tmp_path, capsys):
    """files-2 on the command line: ``--sample-rate`` of a one-channel WAV
    file is its own rate, its samples' (twice the I/Q rate it is read at),
    when it is read as I/Q too (Morse, ``scan --input``), as when its sound
    is decoded as a receiver's audio (R7L-2) and as Import Options takes
    it. There it was taken as the I/Q rate: given the file's own 8 kHz,
    Morse at 700 Hz decoded as noise ("60 WPM IE EEIE ..."), and a file
    whose header gives no rate was scanned around a center of 0 Hz, its
    tone at -600 Hz. The command also says where a one-channel file's I/Q
    is centered: --offset counts from there."""
    from tests.wav_builders import encode, fmt_body, iq_wav, write_wav

    rate = 8000
    tone = np.real(enc.cw_baseband("CQ DE N0CALL", rate, 20, offset_hz=700.0))
    pad = np.zeros(rate // 2)
    x = 0.5 * np.concatenate([pad, tone, pad])
    with_rate = tmp_path / "cw.wav"
    iq_wav(with_rate, x, "int", 2, rate)
    without = tmp_path / "cw_rate0.wav"
    write_wav(without, fmt_body(1, 0, 2), encode(x, "int", 2))
    for path in (with_rate, without):
        argv = ["decode", "cw", "-i", str(path), "--offset", "-1300"]
        assert main(argv + ["--sample-rate", "8000"]) == 0
        out = capsys.readouterr().out
        assert "CQ DE N0CALL" in out, out
        assert f"({len(x) / rate:.3f}s at 0.004 Msps)" in out, out
        assert (
            "One channel: read as I/Q at half its rate, centered on 2 kHz, "
            "which --offset counts from." in out
        ), out
    assert main(["scan", "--input", str(without), "--sample-rate", "8000"]) == 0
    out = capsys.readouterr().out
    assert "Center: 0.0020 MHz" in out, out
    assert "     0.0007 MHz (    -1.3 kHz)" in out, out
