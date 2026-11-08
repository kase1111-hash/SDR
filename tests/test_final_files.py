"""The final review's findings on reading recordings, and on what the File
menu says about them (R7-files).

- files-3: a WAV file whose block align says its 24-bit samples lie in
  32-bit words (as ALSA's arecord wrote S24_LE) played and decoded as noise
  with no word of it: it is refused in plain words (Can't Read Recording).
- files-2: a one-channel WAV file whose header gives no sample rate: the
  rate its sidecar gives, or Import Options takes (prefilled from its
  name), was used as the I/Q rate, though it is the file's own (twice the
  I/Q's): every frequency came out doubled and shifted, Decode Recording
  read its sound at twice its rate. Now the I/Q is worked out from that
  rate as from a header's.
- R7W-06: Recording Info and Decode Recording use the center frequency the
  app reads from the file's name (SDR#, HDSDR...), as Open Recording does.
- ui-3: Convert Recording of a one-channel audio file offered "Tune There"
  to a quarter of its rate, "where the recording was made": no receiver
  was tuned there.

The files are packed by hand (``tests/wav_builders.py``).
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_recording_files import RATE  # noqa: E402
from tests.test_gui_wav_formats import _WavCase  # noqa: E402
from tests.wav_builders import (  # noqa: E402
    encode,
    fmt_body,
    full_scale,
    integers,
    interleave,
    iq_wav,
    write_wav,
)

#: A one-channel (audio) file's own rate.
AUDIO_RATE = 48000


def _tone(n, freq, rate, amplitude=0.4):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestSamplesInWiderWords(_WavCase):
    """files-3: 24-bit samples in 32-bit words (block align 8 for two
    channels) were read 3 bytes at a time: a third more samples than the
    file holds, all noise, with no warning; Decode Recording then blamed
    the content. Now every File menu item says it can't be read, and why."""

    def _s24_in_32(self, name="arecord_s24_le.wav"):
        x = _tone(int(0.2 * RATE), 30e3, RATE)
        ints = np.rint(interleave(x) * full_scale(3)).astype(np.int64)
        path = self.path(name)
        # (ALSA's S24_LE: each sample in the low three bytes of its word.)
        write_wav(path, fmt_body(2, int(RATE), 4, bits=24), integers(ints, 4))
        return path

    def _refused(self, warn):
        warn.assert_called_once()
        title, text = warn.call_args[0][1:3]
        self.assertEqual(title, "Can't Read Recording")
        self.assertIn("arecord_s24_le.wav can't be read", text)
        self.assertIn("24-bit samples are stored in 32-bit words", text)

    def test_open_recording_says_it_cant_be_read(self):
        path = self._s24_in_32()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertFalse(self.win._open_recording())
        self._refused(warn)
        self.assertFalse(self.win._playback.active)

    def test_decode_and_info_say_so_too(self):
        path = self._s24_in_32()
        for run in (self.win._decode_recording, self.win._show_recording_info):
            with self.open_dialog(path), self.warnings() as warn:
                self.assertFalse(run())
            self._refused(warn)
        self.assertEqual(self.boxes, [])  # (no dialog of what it holds)


def _rate0_wav(path, x):
    """``x`` as a one-channel 16-bit WAV file whose header gives no sample
    rate (0)."""
    write_wav(path, fmt_body(1, 0, 2), encode(x, "int", 2))
    return path


def _strongest(samples, rate, center):
    """Where the strongest signal of ``samples`` (I/Q at ``rate``, centered
    on ``center``) shows, in Hz, as the spectrum maps it."""
    y = np.asarray(samples[:8192], dtype=np.complex128)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(y * np.hanning(len(y)))))
    return center + (int(np.argmax(spectrum)) - len(y) // 2) * rate / len(y)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestAOneChannelFileWithoutARate(_WavCase):
    """files-2: a one-channel WAV file whose header gives no sample rate.
    Import Options asked for "Sample rate" prefilled with the file's own
    rate (from its name, tone_48ksps) and took it as the I/Q rate: a 1 kHz
    tone played at -10 kHz, the file lasted half its length, and Decode
    Recording read its sound at twice its rate (nothing decoded). Its
    center was asked too, with the receiver's frequency as a guess. Now the
    rate asked is the file's own (the tooltip says so), and its center
    follows from it: its own frequencies, 0 Hz to half its rate."""

    def _tone_file(self, name="tone_48ksps.wav", seconds=2.0, freq=1000.0):
        t = np.arange(int(seconds * AUDIO_RATE)) / AUDIO_RATE
        return _rate0_wav(self.path(name), 0.5 * np.cos(2 * np.pi * freq * t))

    def test_import_options_asks_for_its_own_rate(self):
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        options, missing = describe_iq_file(self._tone_file())
        self.assertIn("rate", missing)
        guesses = {"rate": 2.4e6, "center": 100.1e6}  # (the receiver's)
        dialog = ImportOptionsDialog(options, missing, "play", guesses)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog._rate_edit.text(), "48 kS/s")  # (its name)
        self.assertEqual(dialog._rate_hint.text(), "from the file name")
        self.assertIn("The file's own samples", dialog._rate_edit.toolTip())
        self.assertIn("half of it", dialog._rate_edit.toolTip())
        # Its center isn't asked (nor guessed from the receiver's): the
        # middle of its own frequencies, a quarter of its rate.
        self.assertFalse(dialog._center_edit.isEnabled())
        self.assertEqual(dialog._center_edit.text(), "0.012 MHz")
        self.assertEqual(
            dialog._center_hint.text(), "the middle of its own frequencies"
        )
        self.assertIn(
            "48,000 samples, 0:02.0 long, 0.000–0.024 MHz", dialog.check_text()
        )
        dialog._rate_edit.setText("8k")  # (another rate: the center follows)
        self.assertEqual(dialog._center_edit.text(), "0.002 MHz")
        self.assertIn("0:12.0 long, 0.000–0.004 MHz", dialog.check_text())
        dialog._rate_edit.setText("48 kS/s")
        dialog._on_accept()
        chosen = dialog.options()
        self.assertEqual(chosen.sample_rate, AUDIO_RATE / 2)  # (its I/Q's)
        self.assertEqual(chosen.frame_rate, AUDIO_RATE)
        self.assertEqual(chosen.center_hz, AUDIO_RATE / 4)
        self.assertAlmostEqual(chosen.duration_s, 2.0)

    def test_the_receivers_rate_is_no_guess_for_it(self):
        """(Its name giving none: asked, not the receiver's 2.4 MS/s.)"""
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        options, missing = describe_iq_file(self._tone_file("tone.wav"))
        dialog = ImportOptionsDialog(
            options, missing, "play", {"rate": 2.4e6, "center": 100.1e6}
        )
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog._rate_edit.text(), "")
        self.assertEqual(dialog._center_edit.text(), "")
        dialog._on_accept()
        self.assertIsNone(dialog.options())
        self.assertEqual(dialog.check_text(), "Enter the sample rate to play the file.")

    def test_open_recording_plays_its_own_frequencies(self):
        path = self._tone_file()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertEqual(
            [type(b).__name__ for b in self.boxes], ["ImportOptionsDialog"]
        )
        device = self.win._playback.device
        self.assertEqual(device.sample_rate, AUDIO_RATE / 2)
        self.assertAlmostEqual(self.win._lo_frequency(), AUDIO_RATE / 4, delta=0.5)
        self.assertAlmostEqual(device.total_samples / device.sample_rate, 2.0)
        device.seek(4000)
        shown = _strongest(device._player.read(8192), AUDIO_RATE / 2, AUDIO_RATE / 4)
        self.assertAlmostEqual(shown, 1000.0, delta=10.0)  # (it showed at -10 kHz)
        self.win._playback.close()

    def test_recording_info_doesnt_invent_its_frequencies(self):
        """(Its rate unknown: no "0 Hz to 0 kHz".)"""
        dialog, options, _missing = self.info_dialog(self._tone_file())
        self.assertTrue(options.real_signal)
        self.assertEqual(
            dialog.value("Sample rate"), "Unknown (not stored in the file)"
        )
        self.assertEqual(
            dialog.value("Channels"),
            "One: a real signal (audio, or an IF or RF sampled directly), played "
            "as I/Q at half its rate",
        )

    def test_a_sidecars_rate_is_its_own(self):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        path = self._tone_file("tone.wav")
        write_sidecar(path, SampleFormat.INT16, AUDIO_RATE, 0.0, header_bytes=44)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertEqual(self.boxes, [])  # (nothing to ask)
        device = self.win._playback.device
        self.assertEqual(device.sample_rate, AUDIO_RATE / 2)
        self.assertAlmostEqual(self.win._lo_frequency(), AUDIO_RATE / 4, delta=0.5)
        self.win._playback.close()

    def test_decode_recording_reads_its_sound_at_its_own_rate(self):
        from tests.test_audio_decoding import EXPECTED, _audio

        path = _rate0_wav(
            self.path("aprs_48ksps.wav"), _audio("AX.25/APRS", AUDIO_RATE, 0.3)
        )
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        said = self.message()
        self.assertIn("Decoded 1 AX.25/APRS message(s)", said)
        self.assertIn("read as a receiver's audio", said)
        rows = [(m["address"], m["content"]) for m in panel._messages]
        ((address, content),) = EXPECTED["AX.25/APRS"]
        self.assertTrue(
            any(a == address and c.startswith(content) for a, c in rows), rows
        )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheFrequencyInTheFileName(_WavCase):
    """R7W-06: a WAV I/Q file named the SDR# way, its center in its name
    and nowhere else. Open Recording's Import Options fills the center in
    "from the file name"; Recording Info said "Unknown (not stored in the
    file)", and Decode Recording said "The file doesn't say where it was
    tuned" and asked for an offset (its messages then had no frequency).
    Now both use the frequency the name gives, and say where it is from."""

    NAME = "SDRSharp_20261108_101500Z_144440000Hz_IQ.wav"

    def _named(self, name=NAME):
        """An APRS packet 50 kHz below the middle (144.390 MHz), 24-bit,
        with no sidecar: only the name says where it was tuned."""
        path = self.path(name)
        iq_wav(path, self._aprs_iq(), "int", 3, int(RATE))
        return path

    def test_recording_info_shows_it(self):
        dialog, options, missing = self.info_dialog(self._named())
        self.assertIn("center", missing)  # (not stored in the file)
        self.assertEqual(
            dialog.value("Center frequency"), "144.440 MHz (from the file name)"
        )
        self.assertEqual(dialog.value("Covers"), "144.320–144.560 MHz")
        self.assertEqual(dialog.value("Metadata"), "The WAV header")

    def test_decode_recording_decodes_at_its_frequency(self):
        from sdr_module.gui import iq_files

        path = self._named()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        dialogs = []

        def fake_exec(dialog):
            self.boxes.append(dialog)
            self.assertIsInstance(dialog, iq_files.DecodeOptionsDialog)
            dialogs.append(dialog)
            dialog._on_accept()
            return dialog.result()

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        (dialog,) = dialogs  # (no Import Options: nothing it needs is asked)
        # A frequency inside the recording, not an offset from its middle.
        self.assertEqual(dialog._freq.accessibleName(), "Frequency")
        self.assertEqual(dialog._freq.text(), "144.390 MHz")
        note = dialog._note.text()
        self.assertIn("covers 144.320–144.560 MHz", note)
        self.assertIn("from the file name", note)
        self.assertIn(
            "Decoded 1 AX.25/APRS message(s) from " f"{self.NAME} at 144.390 MHz",
            self.message(),
        )
        (row,) = list(panel._messages)
        self.assertAlmostEqual(row["freq_hz"], 144.39e6, delta=1.0)

    def test_a_center_cleared_in_import_options_stays_unknown(self):
        """(Asked, as for a headerless file, and the center the name gives
        cleared: the user's word, not the name's, so an offset is asked.)"""
        from sdr_module.gui import iq_files

        path = self.path("capture_144440000Hz_240ksps.cf32")
        self._aprs_iq().tofile(path)
        dialogs = []

        def fake_exec(dialog):
            self.boxes.append(dialog)
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                self.assertEqual(dialog._center_edit.text(), "144.440 MHz")
                dialog._center_edit.setText("")
                dialog._on_accept()
                return dialog.result()
            self.assertIsInstance(dialog, iq_files.DecodeOptionsDialog)
            dialogs.append(dialog)
            return 0  # (looked at, then cancelled)

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.assertFalse(self.win._decode_recording())
        warn.assert_not_called()
        (dialog,) = dialogs
        self.assertEqual(dialog._freq.accessibleName(), "Offset from the center")
        self.assertNotIn("from the file name", dialog._note.text())

    def test_a_frequency_no_recording_is_centered_on_isnt_taken(self):
        """(A frequency in the name that no recording is centered on, below
        1 kHz or above 300 GHz, as Import Options refuses it: a test tone's
        "440Hz". Taken as the center, Decode Recording refused every
        frequency, its own default too ("That frequency isn't inside the
        recording"), so the file couldn't be decoded at all, and Recording
        Info showed a center Open Recording then refused. It stays unknown,
        as before: the offset is asked, and the file decodes.)"""
        from sdr_module.dsp import protocol_encoders as enc

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>Tone name", RATE, ["WIDE1-1"]
        ).astype(np.complex64)
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, 0.5 * burst, pad])  # (in the middle)
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        for name in ("test_440Hz_IQ.wav", "SDRSharp_20261108_101500Z_400GHz_IQ.wav"):
            path = self.path(name)
            iq_wav(path, x, "int", 2, int(RATE))
            dialog, options, _missing = self.info_dialog(path)
            self.assertIs(options.with_name_center(), options)
            self.assertEqual(
                dialog.value("Center frequency"), "Unknown (not stored in the file)"
            )
            self.boxes.clear()
            self.answer()
            with self.open_dialog(path), self.warnings() as warn:
                self.assertTrue(self.win._decode_recording(), name)
            warn.assert_not_called()
            (decode,) = self.boxes
            self.assertEqual(decode._freq.accessibleName(), "Offset from the center")
            self.assertIn(
                f"Decoded 1 AX.25/APRS message(s) from {name}.", self.message()
            )

    def test_open_recording_still_asks(self):
        """(Open Recording confirms it as before: Import Options, filled in
        from the file name.)"""
        path = self._named()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        (dialog,) = self.boxes
        self.assertEqual(dialog._center_edit.text(), "144.440 MHz")
        self.assertEqual(dialog._center_hint.text(), "from the file name")
        self.assertAlmostEqual(self.win._lo_frequency(), 144.44e6, delta=1.0)
        self.win._playback.close()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestConvertingAnAudioFile(_WavCase):
    """ui-3: File > Convert Recording of a one-channel audio file (8 kHz)
    said "Sample rate: 0.004 MS/s" and "Center frequency: 2 kHz" (its I/Q's,
    unexplained) and offered Tune There, "Tune the receiver to 2 kHz, where
    the recording was made": a quarter of its rate, the middle of its own
    sound's frequencies, where no receiver was tuned (the live receiver
    went to 2 kHz; an RTL-SDR refuses it). Tune There is kept where a
    receiver was tuned: two-channel I/Q, a one-channel file its metadata
    places, a capture of RF sampled directly."""

    def _one_channel(self, name, rate, seconds, width=2):
        t = np.arange(int(seconds * rate)) / rate
        x = 0.4 * np.sin(2 * np.pi * (rate / 9.6) * t)
        path = self.path(name)
        return path, iq_wav(path, x, "int", width, int(rate)), len(x)

    def summary(self, path, *answers):
        self.answer(*answers)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._convert_recording())
        warn.assert_not_called()
        box = self.boxes[-1]
        return box, [b.text().replace("&", "") for b in box.buttons()]

    def test_audio_says_what_it_is_and_offers_no_tune_there(self):
        path, _offset, frames = self._one_channel("cw8k_s24.wav", 8000, 2.0, 3)
        before = self.win._current_frequency()
        box, names = self.summary(path, "Close")
        self.assertNotIn("Tune There", names)
        self.assertIn("Play", names)
        info = box.informativeText()
        self.assertIn("Sample format: 24-bit integer WAV, one channel", info)
        self.assertIn("Sample rate: 4 kS/s as I/Q (8 kS/s in the file)", info)
        self.assertIn(
            "Center frequency: 2 kHz (the middle of its own frequencies, 0 Hz to "
            "4 kHz)",
            info,
        )
        self.assertIn("Length: 2.0 s", info)
        self.assertEqual(
            box.text(), f"Loaded {frames // 2:,} samples from cw8k_s24.wav."
        )
        self.assertEqual(self.win._current_frequency(), before)  # (not retuned)

    def test_tune_there_where_a_receiver_was_tuned(self):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar
        from tests.test_gui_wav_formats import _noisy_tone

        x = _noisy_tone(int(0.2 * RATE), 30e3, RATE)
        iq = self._wav("iq.wav", x, "int", 2, "RIFF", 145e6)
        placed, offset, _n = self._one_channel("if.wav", 8000, 1.0)
        write_sidecar(placed, SampleFormat.INT16, 8000.0, 10.7e6, header_bytes=offset)
        rf, _offset, _n = self._one_channel("am_band.wav", 2.4e6, 0.05)
        for path, where in ((iq, 145e6), (placed, 10.7e6), (rf, 600e3)):
            box, names = self.summary(path, "Tune There")
            self.assertIn("Tune There", names)
            self.assertAlmostEqual(self.win._current_frequency(), where, delta=1)


if __name__ == "__main__":
    unittest.main()
