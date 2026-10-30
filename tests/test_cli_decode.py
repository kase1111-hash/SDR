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
