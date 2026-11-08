"""Decoding a receiver's audio (an audio recording) rather than its I/Q.

A one-channel WAV file of audio is usually a receiver's sound: its
speaker's (APRS tones), its FM discriminator's ("discriminator tap": pager
audio, POCSAG and FLEX) or its AM detector's (ACARS). Read as I/Q (M-F2) it
is heard as recorded, but the decoders demodulate their channel
themselves (FM for APRS, POCSAG and FLEX, AM for ACARS), so nothing in it
decoded. The live decoders now take such audio too
(``create_live_decoder(..., audio=True)``: the stages after the
discriminator or AM detector, at the audio's own rate), and File > Decode
Recording reads an audio recording's own samples for them
(``file_jobs.decode_recording``: ``DecodeResult.audio``).

The audio here is as a receiver gives it: band-limited, at levels from
well below to near full scale, either polarity (a discriminator's output
is inverted on many receivers), with a DC offset (a mistuned carrier) and
noise; at telephone rate (8 kHz), 22.05 kHz and 48 kHz.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import (
    create_live_decoder,
    describe_message,
    live_decoder_class,
)
from tests.wav_builders import iq_wav

APRS_INFO = "!4903.50N/07201.75W>Test station"
ACARS_BLOCK = enc.acars_block(
    "N601SD", "H1", "TEST MESSAGE", message_number="M01A", flight_id="SD0601"
)
POCSAG_PAGES = [(1234567, 3, "TEST PAGE ONE"), (1100004, 0, "555-0123")]
EXPECTED = {
    "POCSAG": [("1234567 (F3)", "TEST PAGE ONE"), ("1100004 (F0)", "555-0123")],
    "FLEX": [("1234567 (A, 1600 bps)", "FLEX TEST PAGE")],
    "AX.25/APRS": [("N0CALL-9>APZ001", "49.0583, -72.0292  ·  Car  ·  Test station")],
    "ACARS": [("N601SD", "Flight SD0601  ·  H1  ·  TEST MESSAGE")],
}
RATES = (8000, 22050, 48000)


def _lowpass(x, rate, cutoff):
    """``x`` through a windowed-sinc low-pass (a receiver's audio path)."""
    taps = 63
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / rate * n) * np.hamming(taps)
    return np.convolve(x, h / h.sum(), mode="same")


def _as_received(audio, rate, level, polarity, seed=1, cutoff=4000.0, dc=0.0):
    """Audio as a receiver gives it: band-limited, ``level`` peak, the
    polarity given, a DC offset (a fraction of the level) and noise 26 dB
    below it, with a quarter second of noise before and after."""
    pad = np.zeros(int(0.25 * rate))
    x = np.concatenate([pad, np.asarray(audio, dtype=np.float64), pad])
    x = _lowpass(x, rate, min(cutoff, 0.45 * rate))
    x = polarity * level * x / np.max(np.abs(x))
    rng = np.random.default_rng(seed)
    x += dc * level + 0.05 * level * rng.standard_normal(len(x))
    return x


def _audio(name, rate, level=0.3, polarity=1.0, seed=1):
    """``name``'s transmission as a receiver's audio: AFSK tones (APRS,
    from its speaker: de-emphasised, the space tone 4 dB quieter), the FM
    discriminator's output (POCSAG, FLEX: the NRZ levels themselves, a DC
    offset) or the AM detector's (ACARS: the MSK tones)."""
    if name == "AX.25/APRS":
        line = enc.hdlc_bits(
            enc.ax25_frame("N0CALL-9", "APZ001", APRS_INFO, ["WIDE1-1"]), 32
        )
        raw = enc.afsk_audio(line, rate, twist_db=-4.0)
        return _as_received(raw, rate, level, polarity, seed, cutoff=3000.0)
    if name == "POCSAG":
        raw = enc.bit_levels(enc.pocsag_bits(POCSAG_PAGES), rate, 1200)
        return _as_received(raw, rate, level, polarity, seed, dc=0.1)
    if name == "FLEX":
        raw = enc.flex_frequency([(1234567, "FLEX TEST PAGE")], rate)
        return _as_received(raw, rate, level, polarity, seed, dc=0.1)
    if name == "ACARS":
        raw = enc.msk_audio(enc.acars_bits(ACARS_BLOCK), rate)
        return _as_received(raw, rate, level, polarity, seed, cutoff=3500.0)
    raise ValueError(name)


def _rows(decoder, x, block):
    rows = []
    for start in range(0, len(x), block):
        for msg in decoder.process(x[start : start + block]):
            if getattr(msg, "valid", True):
                rows.append(describe_message(msg))
    return rows


class TestTheDecodersTakeAudio(unittest.TestCase):
    def test_every_one_at_every_rate_level_and_polarity(self):
        for name, expected in EXPECTED.items():
            for rate in RATES:
                if rate < live_decoder_class(name).MIN_AUDIO_RATE:
                    continue
                for level in (0.4, 0.04):
                    for polarity in (1.0, -1.0):
                        with self.subTest(name, rate=rate, level=level, pol=polarity):
                            decoder = create_live_decoder(name, rate, audio=True)
                            self.assertTrue(decoder.audio)
                            x = _audio(name, rate, level, polarity)
                            block = int(0.034 * rate)  # (a display frame's worth)
                            self.assertEqual(_rows(decoder, x, block), expected)

    def test_any_blocks_give_the_same_messages(self):
        rng = np.random.default_rng(3)
        for name in EXPECTED:
            with self.subTest(name):
                x = _audio(name, 48000)
                whole = _rows(create_live_decoder(name, 48000, audio=True), x, len(x))
                decoder = create_live_decoder(name, 48000, audio=True)
                rows, start = [], 0
                while start < len(x):
                    step = int(rng.integers(1, 5000))
                    for msg in decoder.process(x[start : start + step]):
                        if getattr(msg, "valid", True):
                            rows.append(describe_message(msg))
                    start += step
                self.assertEqual(rows, whole)
                self.assertEqual(whole, EXPECTED[name])

    def test_a_mistuned_receivers_dc(self):
        """(A discriminator's output carries the carrier's mistuning as DC:
        here 60 % of the deviation, a 2.7 kHz error on 4.5 kHz.)"""
        for name in ("POCSAG", "FLEX"):
            with self.subTest(name):
                if name == "POCSAG":
                    raw = enc.bit_levels(enc.pocsag_bits(POCSAG_PAGES), 22050, 1200)
                else:
                    raw = enc.flex_frequency([(1234567, "FLEX TEST PAGE")], 22050)
                x = _as_received(raw, 22050, 0.3, -1.0, dc=0.6)
                decoder = create_live_decoder(name, 22050, audio=True)
                self.assertEqual(_rows(decoder, x, 750), EXPECTED[name])

    def test_its_status_counts_the_audio(self):
        decoder = create_live_decoder("POCSAG", 22050, audio=True)
        x = _audio("POCSAG", 22050)
        decoder.process(x)
        status = decoder.status
        self.assertEqual(status.valid, 2)
        self.assertAlmostEqual(status.stream_s, len(x) / 22050, places=6)
        self.assertEqual(status.channel_rate, 22050)
        # (No channel to measure a signal in: the audio is the signal.)
        self.assertFalse(status.signal_measured)

    def test_iq_is_decoded_as_before(self):
        """(Without ``audio`` the decoders take I/Q, as they did.)"""
        for name in EXPECTED:
            with self.subTest(name):
                decoder = create_live_decoder(name, 48000)
                self.assertFalse(decoder.audio)
                if name == "POCSAG":
                    iq = enc.pocsag_baseband(POCSAG_PAGES, 48000)
                elif name == "FLEX":
                    iq = enc.flex_baseband([(1234567, "FLEX TEST PAGE")], 48000)
                elif name == "AX.25/APRS":
                    iq = enc.aprs_baseband(
                        "N0CALL-9", "APZ001", APRS_INFO, 48000, ["WIDE1-1"]
                    )
                else:
                    iq = enc.acars_baseband(ACARS_BLOCK, 48000)
                lead, tail = np.zeros(9600, np.complex64), np.zeros(12000, np.complex64)
                iq = np.concatenate([lead, iq, tail])
                self.assertEqual(_rows(decoder, iq, 1632), EXPECTED[name])


class TestWhatIsNotDecodedFromAudio(unittest.TestCase):
    def test_protocols_not_sent_as_receiver_audio(self):
        """CW decodes a tone in audio from its I/Q, at its pitch; RDS and
        ADS-B aren't in a receiver's audio."""
        for name in ("CW (Morse)", "RDS", "ADS-B"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "audio"):
                    create_live_decoder(name, 48000, audio=True)

    def test_a_rate_too_low_is_said_plainly(self):
        with self.assertRaisesRegex(ValueError, "ACARS needs audio at 9.6 kS/s"):
            create_live_decoder("ACARS", 8000, audio=True)
        with self.assertRaisesRegex(ValueError, "POCSAG needs audio at 8 kS/s"):
            create_live_decoder("POCSAG", 6000, audio=True)


class TestDecodeRecordingReadsAudioAsAudio(unittest.TestCase):
    """``file_jobs.decode_recording`` of a one-channel WAV file at an audio
    rate: its own samples go to the decoder's audio input."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="audio-decode-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def _decode(self, name, audio, rate, kind="int", width=2):
        from sdr_module.gui.file_jobs import decode_recording
        from sdr_module.gui.iq_files import describe_iq_file

        path = os.path.join(self.folder, "audio.wav")
        iq_wav(path, audio, kind, width, rate)
        options, _missing = describe_iq_file(path)
        self.assertTrue(options.audio)
        progress = []
        result = decode_recording(options, name, 0.0, progress.append)
        self.assertEqual(progress[-1], 1.0)
        return result

    def test_aprs_pocsag_flex_and_acars(self):
        for name, expected in EXPECTED.items():
            for rate in (11025, 48000):
                with self.subTest(name, rate=rate):
                    x = _audio(name, rate, polarity=-1.0)
                    result = self._decode(name, x, rate)
                    self.assertTrue(result.audio)
                    rows = [describe_message(m) for m, _t in result.messages]
                    self.assertEqual(rows, expected)
                    self.assertAlmostEqual(result.seconds, len(x) / rate, places=3)
                    # (Each message stamped where it ended in the file.)
                    for _msg, at in result.messages:
                        self.assertLessEqual(at, len(x) / rate + 1e-9)

    def test_wide_and_float_samples(self):
        x = _audio("AX.25/APRS", 22050)
        for kind, width in (("int", 3), ("float", 4), ("int", 1)):
            with self.subTest(kind=kind, width=width):
                result = self._decode("AX.25/APRS", x, 22050, kind, width)
                rows = [describe_message(m) for m, _t in result.messages]
                self.assertEqual(rows, EXPECTED["AX.25/APRS"])

    def test_samples_that_arent_numbers_are_read_as_silence(self):
        """(A damaged floating-point file: a few NaN and infinite samples
        before the transmission. Read as they are, they stopped ACARS for
        good: its phase, followed from block to block, stayed NaN.)"""
        x = _audio("ACARS", 22050)
        x[4410:4413] = np.nan
        x[4500] = np.inf
        result = self._decode("ACARS", x, 22050, "float", 4)
        rows = [describe_message(m) for m, _t in result.messages]
        self.assertEqual(rows, EXPECTED["ACARS"])

    def test_morse_still_decodes_at_its_pitch(self):
        """(CW reads the tone from the I/Q at its frequency, as before.)"""
        from sdr_module.gui.file_jobs import decode_recording
        from sdr_module.gui.iq_files import describe_iq_file
        from tests.test_gui_wav_formats import _cw_audio

        path = os.path.join(self.folder, "cw.wav")
        iq_wav(path, _cw_audio(700.0), "int", 2, 48000)
        options, _missing = describe_iq_file(path)
        result = decode_recording(
            options, "CW (Morse)", 700.0 - options.center_hz, lambda f: None
        )
        self.assertFalse(result.audio)
        text = " ".join(describe_message(m)[1] for m, _t in result.messages)
        self.assertIn("CQ CQ DE N0CALL K", text)

    def test_two_channel_and_high_rate_files_are_read_as_iq(self):
        """(Audio is a one-channel file of 200 kHz or less: an IF or RF
        sampled directly, faster, holds the radio signal itself.)"""
        from sdr_module.gui.file_jobs import decode_recording
        from sdr_module.gui.iq_files import describe_iq_file

        rate = 480000
        burst = enc.aprs_baseband("N0CALL-9", "APZ001", APRS_INFO, rate, ["WIDE1-1"])
        pad = np.zeros(int(0.2 * rate))
        z = np.concatenate([pad, 0.5 * burst, pad])
        real = np.real(z * np.exp(2j * np.pi * 150e3 * np.arange(len(z)) / rate))
        path = os.path.join(self.folder, "if.wav")
        iq_wav(path, real, "int", 2, rate)
        options, _missing = describe_iq_file(path)
        self.assertFalse(options.audio)
        result = decode_recording(
            options, "AX.25/APRS", 150e3 - options.center_hz, lambda f: None
        )
        self.assertFalse(result.audio)
        rows = [describe_message(m) for m, _t in result.messages]
        self.assertEqual(rows, EXPECTED["AX.25/APRS"])
