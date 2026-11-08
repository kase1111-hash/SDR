"""File > Decode Recording of an audio recording, in the window.

A one-channel WAV file at an audio rate (200 kHz or less, M-F2) is usually
a receiver's sound. Decode Recording read it as I/Q only, so a receiver's
audio decoded nothing but Morse tones ("a receiver's audio (APRS tones,
pager audio) isn't decoded as audio"). Now APRS, POCSAG, FLEX and ACARS
decode its sound as a receiver's audio (after the receiver's own FM
discriminator or AM detector), the Decode dialog says so and has no
frequency to ask for them, and the summary says the file was read as
audio. Morse still decodes at its pitch.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_audio_decoding import EXPECTED, _audio  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_wav_formats import _WavCase  # noqa: E402
from tests.wav_builders import iq_wav  # noqa: E402


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestDecodeAnAudioRecording(_WavCase):
    def _decode(self, name, path, edit=None):
        """Decode Recording of ``path`` with ``name`` chosen in the Decoder
        tab; ``edit(dialog)`` before Decode. Returns the dialog."""
        from sdr_module.gui import iq_files

        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText(name)
        self.settle(2)
        dialogs = []

        def fake_exec(dialog):
            self.boxes.append(dialog)
            self.assertIsInstance(dialog, iq_files.DecodeOptionsDialog)
            dialogs.append(dialog)
            if edit is not None:
                edit(dialog)
            dialog._on_accept()
            return dialog.result()

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        return dialogs[0]

    def _wav(self, name, audio, rate):
        path = self.path(name)
        iq_wav(path, audio, "int", 2, rate)
        return path

    def test_pager_and_aprs_audio_decode(self):
        win = self.win
        panel = win._decoder_panel
        for name, rate in (("POCSAG", 22050), ("AX.25/APRS", 8000), ("FLEX", 48000)):
            with self.subTest(name):
                before = len(panel._messages)
                path = self._wav(f"{name[:4]}.wav", _audio(name, rate, 0.3, -1.0), rate)
                dialog = self._decode(name, path)
                rows = [
                    (m["address"], m["content"]) for m in list(panel._messages)[before:]
                ]
                for address, content in EXPECTED[name]:
                    self.assertTrue(
                        any(a == address and c.startswith(content) for a, c in rows),
                        (address, content, rows),
                    )
                said = self.message()
                self.assertIn("read as a receiver's audio", said)
                self.assertIn(f"Decoded {len(EXPECTED[name])} {name}", said)
                # The dialog said so, and asked for no frequency.
                self.assertIn(
                    "decode its sound as a receiver's audio", dialog._note.text()
                )
                self.assertFalse(dialog._freq.isEnabled())

    def test_acars_audio_decodes(self):
        panel = self.win._decoder_panel
        path = self._wav("acars.wav", _audio("ACARS", 22050, 0.2, 1.0), 22050)
        self._decode("ACARS", path)
        rows = [(m["address"], m["content"]) for m in panel._messages]
        self.assertTrue(
            any(
                a == "N601SD" and c.startswith("Flight SD0601  ·  H1  ·  TEST MESSAGE")
                for a, c in rows
            ),
            rows,
        )
        self.assertIn("read as a receiver's audio", self.message())

    def test_nothing_found_says_what_the_audio_must_be(self):
        """(A speaker's audio of a pager: cut below 300 Hz, de-emphasised:
        the summary says pager audio must be the discriminator's.)"""
        rate = 22050
        rng = np.random.default_rng(2)
        noise = 0.05 * rng.standard_normal(rate * 2)
        path = self._wav("noise.wav", noise, rate)
        self._decode("POCSAG", path)
        said = self.message()
        self.assertIn(
            "No POCSAG messages in noise.wav, read as a receiver's audio", said
        )
        self.assertIn("discriminator", said)

    def test_morse_still_asks_for_its_pitch(self):
        from tests.test_gui_wav_formats import _cw_audio

        path = self._wav("cw.wav", _cw_audio(700.0), 48000)
        panel = self.win._decoder_panel

        def pitch(dialog):
            self.assertTrue(dialog._freq.isEnabled())
            dialog.set_frequency_text("700 Hz")

        dialog = self._decode("CW (Morse)", path, pitch)
        texts = " ".join(m["content"] for m in panel._messages)
        self.assertIn("CQ CQ DE N0CALL K", texts)
        self.assertIn(
            "Morse tones decode with CW (Morse) at their pitch", dialog._note.text()
        )
        self.assertNotIn("read as a receiver's audio", self.message())

    def test_choosing_another_protocol_in_the_dialog(self):
        """(The Frequency box is there for CW, not for the audio ones.)"""
        path = self._wav("a.wav", _audio("AX.25/APRS", 22050), 22050)

        def switch(dialog):
            self.assertFalse(dialog._freq.isEnabled())
            dialog._protocol.setCurrentText("CW (Morse)")
            self.assertTrue(dialog._freq.isEnabled())
            dialog._protocol.setCurrentText("AX.25/APRS")
            self.assertFalse(dialog._freq.isEnabled())
            # (A frequency typed that isn't in the file doesn't matter.)
            dialog.set_frequency_text("145.825 MHz")

        self._decode("AX.25/APRS", path, switch)
        self.assertIn("Decoded 1 AX.25/APRS", self.message())

    def test_a_two_channel_recording_is_decoded_as_before(self):
        """(I/Q: the frequency is asked for, the summary says where.)"""
        x = self._aprs_iq("Plain I/Q")
        path = self._wav_iq(x)
        dialog = self._decode(
            "AX.25/APRS", path, lambda d: d.set_frequency_text("144.39 MHz")
        )
        self.assertTrue(dialog._freq.isEnabled())
        self.assertNotIn("audio", dialog._note.text())
        said = self.message()
        self.assertIn("at 144.390 MHz", said)
        self.assertNotIn("receiver's audio", said)

    def _wav_iq(self, x):
        from tests.test_gui_recording_files import RATE

        return self._wav_file("iq.wav", x, RATE, 144.44e6)

    def _wav_file(self, name, x, rate, center):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        path = self.path(name)
        offset = iq_wav(path, x, "int", 2, int(rate))
        write_sidecar(path, SampleFormat.INT16, rate, center, header_bytes=offset)
        return path
