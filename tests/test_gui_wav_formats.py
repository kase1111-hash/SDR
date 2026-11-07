"""M-F1 and M-F2 in the window: wide, floating-point and one-channel WAV files.

Before, File > Open Recording, Recording Info, Convert Recording, Decode
Recording and Find Signals in Recording refused a 24- or 32-bit integer or
a floating-point WAV I/Q file ("its samples are 24-bit, and this app reads
8- and 16-bit I/Q WAV files"; "only integer (PCM) WAV files are read") and
a one-channel WAV file ("it has one channel: it is audio, not I/Q"). Now
they open: wide and floating-point samples at full scale 1.0, and a
one-channel file as the real signal it holds, made I/Q at half its rate
(its own frequencies, 0 Hz to half its rate, not mirrored), and the window
says what each is.
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_recording_files import RATE  # noqa: E402
from tests.test_review_recordings import _ReviewCase  # noqa: E402
from tests.wav_builders import fmt_body, iq_wav, write_wav  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QMessageBox

#: (kind, bytes per sample, container, how Recording Info names it)
FORMATS = [
    ("int", 3, "RIFF", "24-bit integer"),
    ("int", 4, "RF64", "32-bit integer"),
    ("float", 4, "W64", "32-bit float"),
    ("float", 8, "RIFF", "64-bit float"),
]
#: A one-channel (audio) file's rate.
AUDIO_RATE = 48000


def _step(kind, width):
    return 1e-6 if kind == "float" else 1.0 / (2 ** (8 * width - 1) - 1) + 1e-7


def _noisy_tone(n, freq, rate, amplitude=0.3, noise=1e-3, seed=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    x = noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return (x + amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def _cw_audio(pitch=700.0, rate=AUDIO_RATE, text="CQ CQ DE N0CALL K"):
    """Morse audio, as a receiver in CW mode plays it: a keyed tone."""
    code = {
        "C": "-.-.",
        "Q": "--.-",
        "D": "-..",
        "E": ".",
        "N": "-.",
        "0": "-----",
        "A": ".-",
        "L": ".-..",
        "K": "-.-",
    }
    dit = int(1.2 / 20 * rate)
    keyed = [0] * rate
    for word in text.split():
        for letter in word:
            for symbol in code[letter]:
                keyed += [1] * (dit if symbol == "." else 3 * dit) + [0] * dit
            keyed += [0] * (2 * dit)
        keyed += [0] * (4 * dit)
    keyed = np.array(keyed + [0] * rate, dtype=np.float64)
    t = np.arange(len(keyed)) / rate
    rng = np.random.default_rng(1)
    return 0.5 * keyed * np.sin(2 * np.pi * pitch * t) + 0.01 * rng.standard_normal(
        len(t)
    )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class _WavCase(_ReviewCase):
    CENTER = 144.44e6

    def _aprs_iq(self, comment="From a wide WAV"):
        """An APRS packet 50 kHz below the middle of an I/Q recording."""
        from sdr_module.dsp import protocol_encoders as enc

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", f"!4903.50N/07201.75W>{comment}", RATE, ["WIDE1-1"]
        ).astype(np.complex64)
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, 0.5 * burst, pad, pad])
        turn = np.exp(-2j * np.pi * 50e3 * np.arange(len(x)) / RATE)
        return (x * turn).astype(np.complex64)

    def _wav(self, name, x, kind, width, container="RIFF", center=None, rate=RATE):
        """``x`` as a WAV file (with a sidecar giving ``center``, as this
        app saves one beside a WAV file, unless None)."""
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        path = self.path(name)
        offset = iq_wav(path, x, kind, width, int(rate), container)
        if center is not None:
            write_sidecar(path, SampleFormat.INT16, rate, center, header_bytes=offset)
        return path

    def info_dialog(self, path):
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        options, missing = describe_iq_file(path)
        dialog = RecordingInfoDialog(options)
        self.addCleanup(dialog.deleteLater)
        return dialog, options, missing


class TestWideAndFloatRecordings(_WavCase):
    def test_open_recording_plays_every_format(self):
        x = _noisy_tone(int(0.5 * RATE), 30e3, RATE)
        for kind, width, container, _name in FORMATS:
            ext = ".w64" if container == "W64" else ".wav"
            path = self._wav(
                f"{kind}{8 * width}{ext}", x, kind, width, container, 145e6
            )
            with self.open_dialog(path), self.warnings() as warn:
                self.assertTrue(self.win._open_recording(), (kind, width))
            warn.assert_not_called()
            self.assertTrue(self.win._playback.active)
            self.assertAlmostEqual(self.win._lo_frequency(), 145e6, delta=1.0)
            device = self.win._playback.device
            self.assertEqual(device.total_samples, len(x))
            device.seek(1000)
            got = device._player.read(2000)
            np.testing.assert_allclose(got, x[1000:3000], atol=_step(kind, width))
            device.seek(0)
            start = device.position
            for _ in range(3):
                self.win._update_display()
            self.assertGreater(device.position, start)
            self.win._playback.close()

    def test_recording_info_says_how_the_samples_are_stored(self):
        x = _noisy_tone(int(0.2 * RATE), 30e3, RATE)
        for kind, width, container, name in FORMATS:
            ext = ".w64" if container == "W64" else ".wav"
            path = self._wav(
                f"i{kind}{8 * width}{ext}", x, kind, width, container, 145e6
            )
            dialog, options, missing = self.info_dialog(path)
            self.assertEqual(missing, set())
            self.assertEqual(dialog.value("Sample format"), name)
            self.assertIn("I/Q as stereo", dialog.value("Type"))
            if container == "RF64":
                self.assertIn("RF64", dialog.value("Type"))
            self.assertEqual(dialog.value("Samples"), f"{len(x):,}")
            self.assertEqual(dialog.value("Sample rate"), "240 kS/s")
            self.assertIn("look right", dialog._check.text())

    def test_convert_recording_loads_every_sample_and_names_the_format(self):
        x = _noisy_tone(int(0.3 * RATE), 30e3, RATE)
        path = self._wav("take.w64", x, "float", 4, "W64", 145e6)
        self.answer("Close")
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._convert_recording())
        warn.assert_not_called()
        count = self.win._buffer_sample_count()
        self.assertEqual(count, len(x))
        loaded = np.array(self.win._samples_buffer.samples()[:count])
        np.testing.assert_array_equal(loaded, x)
        summary = self.boxes[-1]
        self.assertIn("Sample format: 32-bit float WAV", summary.informativeText())

    def test_decode_recording_finds_the_packet(self):
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        for kind, width, container in (("int", 3, "RF64"), ("float", 8, "RIFF")):
            comment = f"From {kind}{8 * width}"
            x = self._aprs_iq(comment)
            path = self._wav(f"aprs{width}.wav", x, kind, width, container, self.CENTER)
            self.answer()
            with self.open_dialog(path), self.warnings() as warn:
                self.assertTrue(self.win._decode_recording())
            warn.assert_not_called()
            texts = [m["content"] for m in panel._messages]
            self.assertTrue(any(comment in t for t in texts), texts)

    def test_find_signals_in_recording_reads_them(self):
        x = _noisy_tone(int(2.0 * RATE), 30e3, RATE, amplitude=0.1)
        path = self._wav("tone.wav", x, "int", 4, "RIFF", 145e6)
        tools = self.win._signal_tools
        with self.warnings() as warn:
            self.assertTrue(tools.find_in_recording(path))
        warn.assert_not_called()
        found = [s.frequency_hz - 145e6 for s in tools.panel.signals()]
        self.assertTrue(any(abs(f - 30e3) < 2e3 for f in found), found)

    def test_import_options_shows_the_files_own_format(self):
        """A WAV file without a sidecar doesn't say its center: Import
        Options asks, showing the format the file states (not cf32)."""
        x = _noisy_tone(int(0.2 * RATE), 30e3, RATE)
        path = self._wav("nocenter.wav", x, "int", 3)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        dialog = self.boxes[0]
        combo = dialog._format_combo
        self.assertFalse(combo.isEnabled())
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())], ["24-bit integer WAV"]
        )
        self.assertIn("look right", dialog.check_text())
        self.assertEqual(self.win._playback.device.total_samples, len(x))
        self.win._playback.close()


class TestOneChannelRecordings(_WavCase):
    def _audio(self, name="audio.wav", seconds=1.0, kind="int", width=2):
        """Audio (a real signal) at 48 kHz: a 5 kHz tone in noise."""
        n = int(seconds * AUDIO_RATE)
        rng = np.random.default_rng(3)
        t = np.arange(n) / AUDIO_RATE
        x = 0.4 * np.sin(2 * np.pi * 5000.0 * t) + 0.002 * rng.standard_normal(n)
        path = self.path(name)
        iq_wav(path, x, kind, width, AUDIO_RATE)
        return path, x

    def test_open_recording_plays_it_as_iq(self):
        path, x = self._audio()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertEqual(self.boxes, [])  # (nothing to ask: it says it all)
        device = self.win._playback.device
        self.assertEqual(device.sample_rate, AUDIO_RATE / 2)
        self.assertEqual(device.total_samples, len(x) // 2)
        # Its own frequencies, 0 Hz to 24 kHz: centered on 12 kHz.
        self.assertAlmostEqual(self.win._lo_frequency(), AUDIO_RATE / 4, delta=0.5)
        self.assertIn("one channel", self.message())
        start = device.position
        for _ in range(3):
            self.win._update_display()
        self.assertGreater(device.position, start)
        self.win._playback.close()

    def test_recording_info_says_what_it_is(self):
        path, x = self._audio()
        dialog, options, missing = self.info_dialog(path)
        self.assertEqual(missing, set())
        self.assertTrue(options.real_signal)
        self.assertEqual(dialog.value("Type"), "WAV file, one channel (a real signal)")
        self.assertEqual(dialog.value("Sample format"), "16-bit integer")
        self.assertEqual(
            dialog.value("Sample rate"), "24 kS/s as I/Q (48 kS/s in the file)"
        )
        channels = dialog.value("Channels")
        self.assertIn("real signal", channels)
        self.assertIn("0 Hz to 24 kHz", channels)
        self.assertIn("not mirrored", channels)
        self.assertEqual(dialog.value("Samples"), f"{len(x) // 2:,}")
        self.assertEqual(dialog.value("Duration"), "0:01.0")
        self.assertIn("look right", dialog._check.text())

    def test_convert_loads_the_iq(self):
        path, x = self._audio(seconds=0.5)
        self.answer("Close")
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._convert_recording())
        warn.assert_not_called()
        self.assertEqual(self.win._buffer_sample_count(), len(x) // 2)
        self.assertEqual(self.win._buffer_meta, (AUDIO_RATE / 4, AUDIO_RATE / 2))
        summary = self.boxes[-1].informativeText()
        self.assertIn("Sample format: 16-bit integer WAV, one channel", summary)

    def test_find_signals_finds_the_tone_once(self):
        path, _x = self._audio(seconds=2.0)
        tools = self.win._signal_tools
        with self.warnings() as warn:
            self.assertTrue(tools.find_in_recording(path))
        warn.assert_not_called()
        found = [s.frequency_hz for s in tools.panel.signals()]
        self.assertTrue(any(abs(f - 5000.0) < 300 for f in found), found)
        # (Not mirrored: nothing at 24 - 5 = 19 kHz.)
        self.assertFalse(any(abs(f - 19000.0) < 1000 for f in found), found)

    def test_decode_recording_reads_morse_tones(self):
        """A CW tone in audio decodes with CW (Morse) at its frequency
        (below 1 kHz too); the dialog says audio from a receiver isn't
        decoded as audio."""
        from sdr_module.gui import iq_files

        path = self.path("cw.wav")
        iq_wav(path, _cw_audio(700.0), "int", 2, AUDIO_RATE)
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("CW (Morse)")
        self.settle(2)
        notes = []

        def fake_exec(dialog):
            self.boxes.append(dialog)
            self.assertIsInstance(dialog, iq_files.DecodeOptionsDialog)
            notes.append(dialog._note.text())
            dialog.set_frequency_text("700 Hz")
            dialog._on_accept()
            return dialog.result()

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        texts = " ".join(m["content"] for m in panel._messages)
        self.assertIn("CQ CQ DE N0CALL K", texts)
        self.assertIn("one channel", notes[0])
        self.assertIn("Morse", notes[0])
        self.assertIn("isn't decoded as audio", notes[0])

    def test_a_real_if_recording_decodes_like_any_recording(self):
        """A receiver's IF sampled directly (one channel at 960 kS/s): an
        APRS packet at 200 kHz in it decodes there."""
        from sdr_module.dsp import protocol_encoders as enc

        rate = 960e3
        burst = enc.aprs_baseband(
            "N0CALL-9",
            "APZ001",
            "!4903.50N/07201.75W>From a real IF",
            rate,
            ["WIDE1-1"],
        )
        pad = np.zeros(int(0.3 * rate))
        z = np.concatenate([pad, 0.5 * burst, pad])
        real = np.real(z * np.exp(2j * np.pi * 200e3 * np.arange(len(z)) / rate))
        path = self.path("if.wav")
        iq_wav(path, real, "int", 2, int(rate))
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)

        def fake_exec(dialog):
            self.boxes.append(dialog)
            dialog.set_frequency_text("200 kHz")
            dialog._on_accept()
            return dialog.result()

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(any("From a real IF" in t for t in texts), texts)
        # (Not audio at 960 kS/s: the dialog says nothing of audio.)
        note = self.boxes[0]._note.text()
        self.assertNotIn("audio", note)

    def test_a_direct_sampled_capture_is_heard(self):
        """A receiver's RF sampled directly (one channel at 1.2 MS/s: the
        AM band's 0-600 kHz): each station is heard at its own frequency
        in AM, its tone what the speaker gets."""
        from tests.test_gui_round3_main import _FakeClock
        from tests.test_gui_second_receiver import _FakeSink

        rate = 1.2e6
        t = np.arange(int(1.5 * rate)) / rate
        real = 0.002 * np.random.default_rng(8).standard_normal(len(t))
        for station, tone in ((450e3, 800.0), (200e3, 400.0)):
            envelope = 0.1 * (1 + 0.6 * np.sin(2 * np.pi * tone * t))
            real += envelope * np.cos(2 * np.pi * station * t)
        path = self.path("am_band.wav")
        iq_wav(path, real, "int", 2, int(rate))
        win = self.win
        sink = _FakeSink()
        sink.open = True
        win._audio, win._audio_enabled, win._audio_problem = sink, True, None
        # (As with an output device: a mode change asks the system again.)
        win._audio_output_problem = lambda: None
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        self.assertAlmostEqual(win._lo_frequency(), rate / 4, delta=0.5)
        clock = _FakeClock()
        win._set_demod_mode("AM", None, "10 kHz")
        for station, tone in ((450e3, 800.0), (200e3, 400.0)):
            self.assertTrue(win.set_frequency(station), self.message())
            sink.blocks.clear()
            with mock.patch.object(self.mw, "time", clock):
                win._pacer.restart()
                for _ in range(20):  # (0.6 s)
                    clock.advance(0.03)
                    win._update_display()
            heard = sink.samples()[-16384:]
            self.assertEqual(len(heard), 16384, station)
            spectrum = np.abs(np.fft.rfft(heard * np.hanning(len(heard))))
            freqs = np.fft.rfftfreq(len(heard), 1 / 48000)
            loudest = freqs[np.argmax(spectrum[20:]) + 20]
            self.assertAlmostEqual(loudest, tone, delta=10, msg=station)
        win._playback.close()

    def test_a_sidecar_that_places_it_elsewhere_is_said_so(self):
        """A sidecar may place a real IF recording at its RF: Recording
        Info then doesn't claim the file's own 0-24 kHz are what shows."""
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        path, _x = self._audio("if_10m.wav")
        write_sidecar(path, SampleFormat.INT16, AUDIO_RATE, 28.4e6, header_bytes=44)
        dialog, options, _missing = self.info_dialog(path)
        self.assertEqual(dialog.value("Center frequency"), "28.400 MHz")
        channels = dialog.value("Channels")
        self.assertIn("not mirrored", channels)
        self.assertNotIn("its own frequencies", channels)
        self.assertIn("where its metadata places it", channels)

    def test_telephone_rate_audio_plays_too(self):
        """8 kHz audio is 4 kS/s of I/Q (0 to 4 kHz): it plays in real time
        like any slow recording (a display block, 2048 samples, at a time:
        half a second here)."""
        from tests.test_gui_round3_main import _FakeClock

        n = 2 * 8000
        x = 0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / 8000)
        path = self.path("phone.wav")
        iq_wav(path, x, "int", 2, 8000)
        win = self.win
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        device = win._playback.device
        self.assertEqual((device.sample_rate, device.total_samples), (4000, n // 2))
        self.assertAlmostEqual(win._lo_frequency(), 2000, delta=0.5)
        clock = _FakeClock()
        with mock.patch.object(self.mw, "time", clock):
            win._pacer.restart()
            win._update_display()
            first = device.position
            for _ in range(50):  # (1.5 s)
                clock.advance(0.03)
                win._update_display()
        played = (device.position - first) / device.sample_rate
        self.assertGreater(played, 0.9)
        self.assertAlmostEqual(played, 1.5, delta=2048 / 4000 + 0.03)
        win._playback.close()

    def test_every_format_of_one_channel_opens(self):
        for kind, width in (("int", 1), ("int", 3), ("float", 4), ("float", 8)):
            path, x = self._audio(f"a{kind}{width}.wav", 0.2, kind, width)
            dialog, options, _missing = self.info_dialog(path)
            self.assertEqual(options.total_samples, len(x) // 2)
            self.assertIn("look right", dialog._check.text(), (kind, width))


class TestStillRefused(_WavCase):
    def _refused(self, path, said):
        with (
            self.open_dialog(path),
            mock.patch.object(QMessageBox, "warning") as warn,
            self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"),
        ):
            self.win._show_recording_info()
        self.assertIn(said, warn.call_args[0][2])
        self.assertFalse(self.win._playback.active)

    def test_more_than_two_channels(self):
        path = self.path("quad.wav")
        write_wav(path, fmt_body(4, 48000, 2), bytes(4 * 2 * 1000))
        self._refused(path, "it has 4 channels")

    def test_a_compressed_wav_file(self):
        import struct

        body = struct.pack("<HHIIHH", 7, 1, 8000, 8000, 1, 8)  # mu-law
        path = self.path("phone.wav")
        write_wav(path, body, bytes(800))
        self._refused(path, "WAV format 7")
