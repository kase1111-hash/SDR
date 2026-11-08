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

And the last cleanup's (R8-files):

- R8-10: Recording Info showed a center from the file name but not a rate
  from it ("Sample rate: Unknown"), though Import Options takes it.
- R8-11: 24-bit samples in 32-bit words (files-3) are read where their data
  shows them to be; only words that fit neither layout are refused.
- R8-13: a one-channel file at an audio rate that its sidecar places at an
  IF was decoded as a receiver's audio, though the other File menu items
  took it as placed there: one rule now (dsp.real_iq.is_audio).

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
    the content. R7 refused them (Can't Read Recording: the header doesn't
    say where in each word a sample lies). R8-11 reads them where their
    data shows them to be, in every File menu item; words that fit neither
    layout (packed samples under a wrong block align) are still refused,
    saying why."""

    def _s24_in_32(self, name="arecord_s24_le.wav", x=None, center=None):
        """``x`` as ALSA's S24_LE writes it: each sample in the low three
        bytes of its 32-bit word (a sidecar giving ``center``, if one)."""
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        if x is None:
            x = _tone(int(0.2 * RATE), 30e3, RATE)
        ints = np.rint(interleave(x) * full_scale(3)).astype(np.int64)
        path = self.path(name)
        offset = write_wav(path, fmt_body(2, int(RATE), 4, bits=24), integers(ints, 4))
        if center is not None:
            write_sidecar(path, SampleFormat.INT16, RATE, center, header_bytes=offset)
        return path, x

    def _neither(self):
        """Packed 24-bit samples under a header saying 32-bit words: its
        words fit neither layout."""
        x = interleave(_tone(int(0.2 * RATE), 30e3, RATE))
        path = self.path("arecord_s24_le.wav")
        write_wav(path, fmt_body(2, int(RATE), 4, bits=24), encode(x, "int", 3))
        return path

    def _refused(self, warn):
        warn.assert_called_once()
        title, text = warn.call_args[0][1:3]
        self.assertEqual(title, "Can't Read Recording")
        self.assertIn("arecord_s24_le.wav can't be read", text)
        self.assertIn("24-bit samples are stored in 32-bit words", text)
        self.assertIn("neither in each word's top three bytes nor", text)

    def test_open_recording_says_it_cant_be_read(self):
        """(Words that fit neither layout.)"""
        path = self._neither()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertFalse(self.win._open_recording())
        self._refused(warn)
        self.assertFalse(self.win._playback.active)

    def test_decode_and_info_say_so_too(self):
        """(Words that fit neither layout.)"""
        path = self._neither()
        for run in (self.win._decode_recording, self.win._show_recording_info):
            with self.open_dialog(path), self.warnings() as warn:
                self.assertFalse(run())
            self._refused(warn)
        self.assertEqual(self.boxes, [])  # (no dialog of what it holds)

    def test_open_recording_plays_them(self):
        path, x = self._s24_in_32(center=145e6)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertEqual(self.boxes, [])  # (nothing to ask)
        device = self.win._playback.device
        self.assertEqual(device.total_samples, len(x))
        device.seek(1000)
        np.testing.assert_allclose(
            device._player.read(2000), x[1000:3000], atol=1.0 / full_scale(3)
        )
        self.win._playback.close()

    def test_recording_info_says_how_they_are_stored(self):
        path, x = self._s24_in_32()
        dialog, _options, _missing = self.info_dialog(path)
        self.assertEqual(
            dialog.value("Sample format"),
            "24-bit integer, in the low three bytes of 32-bit words",
        )
        self.assertEqual(dialog.value("Samples"), f"{len(x):,}")
        self.assertIn("look right", dialog._check.text())

    def test_decode_recording_finds_the_packet(self):
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        path, _x = self._s24_in_32(
            "aprs_s24_le.wav", self._aprs_iq("From S24_LE"), self.CENTER
        )
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        self.assertIn("Decoded 1 AX.25/APRS message(s)", self.message())
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(any("From S24_LE" in t for t in texts), texts)


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
        # The file's own samples, at the rate asked: 96,000 at 48 kS/s, two
        # seconds (it said "48,000 samples", its I/Q's, next to 48 kS/s).
        self.assertIn(
            "96,000 samples, 0:02.0 long, 0.000–0.024 MHz", dialog.check_text()
        )
        dialog._rate_edit.setText("8k")  # (another rate: the center follows)
        self.assertEqual(dialog._center_edit.text(), "0.002 MHz")
        self.assertIn(
            "96,000 samples, 0:12.0 long, 0.000–0.004 MHz", dialog.check_text()
        )
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
        """(Its rate unknown, its name giving none either: no "0 Hz to 0
        kHz". A name that gives one: TestTheRateInTheFileName.)"""
        dialog, options, _missing = self.info_dialog(self._tone_file("tone.wav"))
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

    def test_its_rate_reads_as_recording_info_says_it(self):
        # The summary said "Sample rate: 0.24 MS/s" where Recording Info
        # (and an audio file's own line here) says "240 kS/s".
        from tests.test_gui_wav_formats import _noisy_tone

        x = _noisy_tone(int(0.2 * RATE), 30e3, RATE)
        iq = self._wav("iq240k.wav", x, "int", 2, "RIFF", 145e6)
        box, _names = self.summary(iq, "Close")
        self.assertIn("Sample rate: 240 kS/s\n", box.informativeText())
        dialog, _options, _missing = self.info_dialog(iq)
        self.assertEqual(dialog.value("Sample rate"), "240 kS/s")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheRateInTheFileName(_WavCase):
    """R8-10: a WAV file whose header gives no sample rate, its rate in its
    name (capture_100.1MHz_48ksps.wav). Recording Info showed the center the
    name gives "(from the file name)" but "Sample rate: Unknown" and
    "Duration: Unknown (needs the sample rate)", though Import Options and
    Open Recording take the name's rate. Now Info shows that rate too,
    marked, with the span and the length that follow from it; a rate no
    recording has (one Import Options refuses) isn't taken."""

    def _rate0_iq(self, name, seconds=0.5):
        """Two channels (I/Q) of 16-bit samples, the header's rate 0."""
        x = _tone(int(seconds * 48000), 5e3, 48000)
        path = self.path(name)
        write_wav(path, fmt_body(2, 0, 2), encode(interleave(x), "int", 2))
        return path

    def test_recording_info_shows_it(self):
        dialog, options, missing = self.info_dialog(
            self._rate0_iq("capture_100.1MHz_48ksps.wav")
        )
        self.assertIn("rate", missing)  # (not stored in the file)
        self.assertEqual(dialog.value("Sample rate"), "48 kS/s (from the file name)")
        self.assertEqual(
            dialog.value("Center frequency"), "100.100 MHz (from the file name)"
        )
        self.assertEqual(dialog.value("Covers"), "100.076–100.124 MHz")
        self.assertEqual(dialog.value("Duration"), "0:00.5")
        self.assertEqual(dialog.value("Samples"), "24,000")

    def test_a_one_channel_files_name_gives_its_own_rate(self):
        """(Its I/Q's is half of it, and its frequencies are its own: 0 Hz
        to half its rate, centered on a quarter of it.)"""
        t = np.arange(2 * AUDIO_RATE) / AUDIO_RATE
        path = _rate0_wav(
            self.path("tone_48ksps.wav"), 0.5 * np.cos(2 * np.pi * 1e3 * t)
        )
        dialog, options, _missing = self.info_dialog(path)
        self.assertTrue(options.real_signal)
        self.assertEqual(
            dialog.value("Sample rate"),
            "24 kS/s as I/Q (48 kS/s in the file, from the file name)",
        )
        self.assertEqual(dialog.value("Center frequency"), "0.012 MHz")
        self.assertEqual(dialog.value("Covers"), "0.000–0.024 MHz")
        self.assertEqual(dialog.value("Duration"), "0:02.0")
        self.assertIn("0 Hz to 24 kHz", dialog.value("Channels"))
        # (A sidecar placing it at an IF keeps it there, at the name's rate.)
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        write_sidecar(path, SampleFormat.INT16, 0.0, 10.7e6, header_bytes=44)
        dialog, options, _missing = self.info_dialog(path)
        self.assertEqual(dialog.value("Center frequency"), "10.700 MHz")
        self.assertEqual(dialog.value("Covers"), "10.688–10.712 MHz")

    def test_the_headers_rate_wins(self):
        path = self.path("capture_100.1MHz_48ksps.wav")
        iq_wav(path, _tone(9600, 5e3, 96000), "int", 2, 96000)
        dialog, _options, _missing = self.info_dialog(path)
        self.assertEqual(dialog.value("Sample rate"), "96 kS/s")
        self.assertEqual(dialog.value("Duration"), "0:00.1")

    def test_a_rate_no_recording_has_isnt_taken(self):
        """(Below 1 kS/s or above 200 MS/s, as Import Options refuses it.)"""
        for name in ("hum_100sps.wav", "capture_100.1MHz_500MSps.wav"):
            dialog, options, _missing = self.info_dialog(self._rate0_iq(name))
            self.assertIs(options.with_name_rate(), options)
            self.assertEqual(
                dialog.value("Sample rate"), "Unknown (not stored in the file)", name
            )
            self.assertEqual(
                dialog.value("Duration"), "Unknown (needs the sample rate)", name
            )

    def test_a_rate_cleared_in_import_options_stays_unknown(self):
        """(A headerless file is asked about, its rate prefilled from its
        name: cleared there, it is the user's word, not the name's.)"""
        from sdr_module.gui import iq_files

        path = self.path("capture_144440000Hz_240ksps.cf32")
        self._aprs_iq().tofile(path)

        def fake_exec(dialog):
            self.boxes.append(dialog)
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                self.assertEqual(dialog._rate_edit.text(), "240 kS/s")
                dialog._rate_edit.setText("")
                dialog._on_accept()
                return dialog.result()
            self.assertIsInstance(dialog, iq_files.RecordingInfoDialog)
            self.assertEqual(
                dialog.value("Sample rate"), "Unknown (not stored in the file)"
            )
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.win._show_recording_info()
        warn.assert_not_called()
        self.assertEqual(
            [type(b).__name__ for b in self.boxes],
            ["ImportOptionsDialog", "RecordingInfoDialog"],
        )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestAnAudioRateFileItsSidecarPlaces(_WavCase):
    """R8-13: a one-channel file at an audio rate whose sidecar places it at
    an IF (10.7 MHz): a receiver's IF recorded at 48 kS/s, not its sound.
    Open Recording played it there, Recording Info said "where its metadata
    places it" and Convert Recording offered Tune There to 10.7 MHz, but
    Decode Recording read its samples as a receiver's audio (its rate alone
    decided) and found nothing in the FM signal they hold. One rule now
    (dsp.real_iq.is_audio): a receiver's audio is at an audio rate and at
    its own frequencies; a file its metadata places elsewhere is decoded
    from its I/Q there, as any recording made there."""

    IF_HZ = 10.7e6

    def _if_file(self, name="if_aprs.wav", placed=True):
        """An APRS packet's FM signal on a 12 kHz carrier (the middle of the
        file's own frequencies) at 48 kS/s, one channel; its sidecar places
        that middle at 10.7 MHz."""
        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        rate = AUDIO_RATE
        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>From an IF", rate, ["WIDE1-1"]
        )
        pad = np.zeros(int(0.3 * rate), np.complex64)
        s = np.concatenate([pad, burst, pad])
        carrier = np.exp(2j * np.pi * (rate / 4) * np.arange(len(s)) / rate)
        path = self.path(name)
        offset = iq_wav(path, 0.5 * np.real(s * carrier), "int", 2, rate)
        if placed:
            write_sidecar(
                path, SampleFormat.INT16, rate, self.IF_HZ, header_bytes=offset
            )
        return path

    def test_decode_recording_decodes_it_where_it_is_placed(self):
        from sdr_module.gui import iq_files

        path = self._if_file()
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
        (dialog,) = dialogs
        # A frequency inside the IF it holds (no receiver's sound to read).
        self.assertTrue(dialog._freq.isEnabled())
        self.assertEqual(dialog._freq.text(), "10.700 MHz")
        self.assertNotIn("as a receiver's audio", dialog._note.text())
        self.assertIn(
            "Decoded 1 AX.25/APRS message(s) from if_aprs.wav at 10.700 MHz",
            self.message(),
        )
        (row,) = list(panel._messages)
        self.assertAlmostEqual(row["freq_hz"], self.IF_HZ, delta=1.0)

    def test_every_file_menu_item_agrees(self):
        from sdr_module.gui.iq_files import describe_iq_file

        placed = self._if_file()
        options = describe_iq_file(placed)[0]
        self.assertFalse(options.audio)
        self.assertFalse(options.decodes_as_audio("AX.25/APRS"))
        # (The same file without its sidecar: its sound, its own frequencies.)
        self.assertTrue(describe_iq_file(self._if_file("own.wav", False))[0].audio)
        dialog, _options, _missing = self.info_dialog(placed)
        self.assertIn("where its metadata places it", dialog.value("Channels"))
        # Open Recording plays it at 10.7 MHz as any recording made there:
        # not "Audio: listening in USB from 0 Hz", its lowest frequency.
        with self.open_dialog(placed), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertAlmostEqual(self.win._lo_frequency(), self.IF_HZ, delta=1.0)
        self.assertNotIn("Audio: listening in USB", self.message())
        mode = self.win._control_panel.get_demod_mode()
        self.assertNotEqual(self.win._mode_text(mode), "USB")
        self.win._playback.close()


if __name__ == "__main__":
    unittest.main()
