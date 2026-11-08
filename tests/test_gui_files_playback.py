"""The files lane (M-F1, M-F2) and the playback lane (M-P1) together, and
the bookmarks the files and squelch (M-Q3) lanes both changed.

* A one-channel WAV file is read as I/Q at half its rate (M-F2): 8 kHz
  audio is 4 kS/s of I/Q, so the display loop reads a display block (2048
  samples, half a second of it) every 16 frames at 1x and every 31 at
  0.5x. Its sound at another speed (M-P1, time-stretched) was given out
  only with each block, a quarter of a second at most, and the pace it
  follows was measured from those blocks alone (a frame more than a
  quarter of a second after the last counted as a stall): the speaker ran
  dry most of the time, 3.5 s of every 4 for 8 kHz audio at 0.5x, 2.8 s
  for 22.05 kHz audio, 2.2 s for 8 kHz audio at 2x.
* Bookmarks below 1 kHz and above 6 GHz keep their frequency (M-F2) and
  bookmarks keep a reverse tone squelch, CHIRP's TSQL-R (M-Q3): one that
  is both, made while a recording above 6 GHz plays, is edited, exported
  to CHIRP CSV and imported again, and tuned while the recording plays.
"""

import io
import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_playback_sound import AUDIO, _peak_hz, _SoundCase  # noqa: E402
from tests.test_gui_wav_formats import AUDIO_RATE, _WavCase  # noqa: E402
from tests.test_playback_sound import _Speaker  # noqa: E402
from tests.wav_builders import iq_wav  # noqa: E402

#: A recording above the receiver's range (6 GHz): the 3 cm band.
X_BAND = 10.368e9
#: Its rate (I/Q) and the station's offset from its middle.
X_RATE = 240000
X_OFFSET = 30e3


def _tone_wav(path, rate, seconds, freq=1000.0, bursts=None):
    """A one-channel WAV file of a ``freq`` tone at ``rate`` (a little
    noise), or of tone bursts ``(on_s, off_s)``."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    x = 0.3 * np.sin(2 * np.pi * freq * t)
    if bursts is not None:
        on, off = bursts
        x = np.where((t % (on + off)) < on, x, 0.0)
    x = x + 0.001 * np.random.default_rng(4).standard_normal(n)
    iq_wav(path, x, "int", 2, int(rate))
    return path


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestSlowRecordingsAtOtherSpeeds(_SoundCase):
    """Sound at 0.5x and 2x of recordings slower than a display block a
    frame: telephone-rate and other low-rate audio in one-channel files."""

    def open_audio(self, rate, seconds=6.0, bursts=None):
        path = _tone_wav(
            os.path.join(self.folder, f"audio{rate}.wav"), rate, seconds, bursts=bursts
        )
        self.play(path)
        win = self.win
        self.assertTrue(win._playback.active, self.message())
        # (Heard as recorded: USB from 0 Hz, the 1 kHz tone at 1 kHz.)
        self.assertEqual(win._control_panel.get_demod_mode(), "USB")
        win._playback_bar.set_loop(True)
        win._control_panel.set_squelch_enabled(False)
        win._control_panel.set_audio_agc_speed("off")
        win._rx_history.clear()
        win._pacer.restart()

    def test_low_rate_audio_plays_without_gaps(self):
        win = self.win
        for rate, speed in ((8000, 0.5), (8000, 2.0), (11025, 0.5), (22050, 0.5)):
            with self.subTest(rate=rate, speed=speed):
                self.heard.clear()
                self.open_audio(rate)
                win._playback_bar.set_speed(speed)
                self.frames(2.0)
                device = win._device
                before = device.position + device.loops * device.total_samples
                start = self.clock.now
                self.frames(4.0)
                gaps, dropped = self.troubles_since(start)
                self.assertLess(gaps, 0.01)
                self.assertEqual(dropped, 0.0)
                sound = self.sound(since=start)
                self.assertAlmostEqual(len(sound) / AUDIO, 4.0, delta=0.15)
                self.assertAlmostEqual(_peak_hz(sound[: int(AUDIO)]), 1000.0, delta=2.0)
                # At the speed chosen: 4 s of the recording at 0.5x, 2 s ...
                after = device.position + device.loops * device.total_samples
                played = (after - before) / device.sample_rate
                self.assertAlmostEqual(played, 4.0 * speed, delta=2048 / (rate / 2))
                win._playback.close()

    def test_a_sound_plays_on_between_blocks_from_its_start(self):
        """Tone bursts in 8 kHz audio at 0.5x with the squelch on: within
        each burst's sound (from the frame the squelch opens to the one it
        closes in) the speaker never runs dry, its first block included
        (taken for a pile-up of the stretch, it was rushed through)."""
        win = self.win
        self.open_audio(8000, seconds=10.0, bursts=(1.5, 1.0))
        panel = win._control_panel
        panel.set_squelch_auto(False)
        panel.set_squelch_enabled(True)
        panel.set_squelch_db(-40.0)
        update = win._update_display
        opened = []  # (fake time, the squelch open after the frame)

        def frame():
            update()
            opened.append((self.clock.now, win._squelch.is_open))

        win._update_display = frame
        win._playback_bar.set_speed(0.5)
        self.frames(1.0)
        start = self.clock.now
        self.frames(12.0)
        opens = sum(
            1
            for i in range(1, len(opened))
            if not opened[i - 1][1] and opened[i][1] and opened[i][0] >= start
        )
        self.assertGreaterEqual(opens, 2)
        model = _Speaker()
        model.listening = False
        state = dict(opened)
        for when, block in self.heard:
            if when < start:
                continue
            model.write(block, when)
            if len(block):
                model.listening = state.get(when, False)
        self.assertLess(model.gaps, 0.01)
        win._playback.close()

    def test_1x_still_passes_the_blocks_as_they_are(self):
        """(At 1x a recording that keeps up is heard as it is: the
        frames between its blocks add nothing.)"""
        win = self.win
        self.open_audio(8000)
        self.frames(1.0)
        self.heard.clear()
        self.frames(2.0)
        self.assertFalse(win._playback_sound.stretching)
        blocks = [b for _t, b in self.heard if len(b)]
        self.assertGreaterEqual(len(blocks), 3)
        # (Each a display block's sound or more: half a second of 8 kHz.)
        for block in blocks:
            self.assertGreaterEqual(len(block), int(2048 / 4000 * AUDIO))
        win._playback.close()

    def test_record_audio_seek_pause_and_loop(self):
        """8 kHz audio at 0.5x with Record Audio on, through a seek, a
        pause, the file's end (Loop) and 2x: the take keeps the
        recording's own pace (what was played of it, at its own pitch),
        the speaker plays on after each restart, and nothing is given out
        between blocks once the sound has stopped."""
        import shutil
        import tempfile
        import wave

        folder = tempfile.mkdtemp(prefix="slow-take-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.store.values["recording/folder"] = folder
        win = self.win
        self.open_audio(8000, seconds=3.0)
        device = win._device

        def played():
            return (device.position + device.loops * device.total_samples) / 4000

        win._playback_bar.set_speed(0.5)
        self.frames(1.0)
        self.assertTrue(win._start_audio_recording())
        start = played()
        self.frames(2.0)
        jump = -played()
        win._playback.seek(2.5)  # (a little before the end: it loops)
        jump += played()
        self.assertFalse(win._playback_sound.stretching)
        self.assertEqual(len(win._playback_sound.more(self.clock.now)), 0)
        resumed = self.clock.now
        self.frames(3.0)
        self.assertGreaterEqual(device.loops, 1)
        gaps, _dropped = self.troubles_since(resumed + 0.3)
        self.assertLess(gaps, 0.01)
        win._toggle_acquisition()  # (Space: pause)
        self.assertFalse(win._playback_sound.stretching)
        self.frames(0.5)
        win._toggle_acquisition()
        win._playback_bar.set_speed(2.0)
        self.frames(2.0)
        # (What played of the recording: the seek skipped what lay between.)
        heard = played() - start - jump
        path = win._stop_audio_recording()
        with wave.open(path, "rb") as take:
            frames = take.getnframes()
            data = np.frombuffer(take.readframes(frames), dtype=np.int16)
        self.assertAlmostEqual(frames / AUDIO, heard, delta=2048 / 4000 + 0.1)
        self.assertAlmostEqual(_peak_hz(data[4800:52800]), 1000.0, delta=2.0)
        win._playback.close()


def _nfm(seconds, ctcss_hz, seed=1, rate=X_RATE, offset=X_OFFSET):
    """Narrow FM (5 kHz) tones with a CTCSS tone at 15 % of the deviation,
    ``offset`` Hz from the middle of an I/Q recording."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    audio = 0.15 * np.sin(2 * np.pi * ctcss_hz * t)
    for freq in (420.0, 830.0, 1270.0, 2010.0):
        audio += 0.15 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(5e3 * audio + offset) / rate
    x = 0.03 * np.exp(1j * phase)
    x += 1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x.astype(np.complex64)


class TestBookmarksOfBothKinds(_WavCase):
    """A reverse tone squelch (TSQL-R) on a frequency the receiver's own
    range doesn't cover (10368.030 MHz), made while a recording that does
    plays; and a bookmark below 1 kHz in a one-channel recording."""

    def _sink(self):
        from tests.test_gui_second_receiver import _FakeSink

        sink = _FakeSink()
        sink.open = True
        win = self.win
        win._audio, win._audio_enabled, win._audio_problem = sink, True, None
        win._audio_output_problem = lambda: None
        return sink

    def _loudest(self, sink, seconds):
        """The loudest sample the speaker got in each frame of ``seconds``."""
        from tests.test_gui_round3_main import _FakeClock

        win = self.win
        clock = _FakeClock()
        peaks = []
        with mock.patch.object(self.mw, "time", clock):
            win._pacer.restart()
            for _ in range(int(seconds / 0.03)):
                before = len(sink.blocks)
                clock.advance(0.03)
                win._update_display()
                peaks.append(
                    max(
                        (float(np.max(np.abs(b))) for b in sink.blocks[before:]),
                        default=0.0,
                    )
                )
        return peaks

    def _round_trip(self, bookmarks):
        """Export to CHIRP CSV and import again in their place."""
        from sdr_module.core.chirp_csv import export_bookmarks_csv

        panel = self.win._bookmarks_panel
        out = io.StringIO()
        export_bookmarks_csv(out, bookmarks)
        path = self.path("bookmarks.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(out.getvalue())
        panel._ask_replace_or_append = lambda count: "replace"
        self.assertEqual(panel.import_csv(path), len(bookmarks))
        return out.getvalue()

    def test_a_reverse_tone_squelch_above_6_ghz(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog, bookmark_squelch

        win = self.win
        panel = win._bookmarks_panel
        cp = win._control_panel
        station = X_BAND + X_OFFSET
        x = np.concatenate([_nfm(2.0, 100.0), _nfm(2.0, 123.0, seed=2)])
        path = self._wav("x_band.wav", x, "int", 3, "RIFF", X_BAND, rate=X_RATE)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.assertTrue(win.set_frequency(station), self.message())
        cp.set_tone_squelch_hz(100.0)
        cp.set_tone_reverse(True)
        # Ctrl+B keeps both: the frequency and the reverse tone squelch.
        win._bookmark_current_frequency()
        self.assertIn("(FM, reverse tone squelch 100.0 Hz)", self.message())
        made = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual(made["freq_hz"], station)
        self.assertEqual((made["tone_mode"], made["ctone"]), ("TSQL-R", 100.0))
        # The Add box shows it (not 6000 MHz), and Add saves it.
        self.assertAlmostEqual(panel._freq_input.value() * 1e6, station, delta=1.0)
        panel._confirm_add_another = lambda *args: True
        panel._label_input.setText("Added")
        panel._add_current()
        added = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual(added["freq_hz"], station)
        # The editor keeps both.
        dialog = BookmarkEditDialog(made, ("FM", "5 kHz"), [])
        self.addCleanup(dialog.deleteLater)
        self.assertTrue(dialog._reverse.isChecked())
        edited = dialog.bookmark()
        self.assertEqual(edited["freq_hz"], station)
        self.assertEqual(edited["tone_mode"], "TSQL-R")
        # CHIRP CSV and back.
        text = self._round_trip([edited, added])
        self.assertIn("10368.030000", text)
        self.assertIn("TSQL-R", text)
        back = panel.bookmark(0)
        self.assertEqual(back["freq_hz"], station)
        self.assertEqual(bookmark_squelch(back), (100.0, None, True))
        # Tuned while the recording plays: muted while its tone is sent,
        # the station with another tone heard.
        win._set_demod_mode("AM")
        cp.set_tone_reverse(False)
        cp.set_tone_squelch_hz(0.0)
        panel._tune_index(0)
        self.assertIn("(FM, reverse tone squelch 100.0 Hz)", self.message())
        self.assertEqual(win._current_frequency(), station)
        self.assertEqual((cp.tone_squelch_hz(), cp.tone_reverse()), (100.0, True))
        sink = self._sink()
        cp.set_squelch_db(-80)
        peaks = self._loudest(sink, 3.9)
        self.assertLess(max(peaks[5:60]), 1e-3)  # (its tone: muted)
        self.assertGreater(max(peaks[80:]), 0.1)  # (another tone: heard)
        # Without the recording the receiver can't tune it: said so.
        win._playback.close()
        before = win._current_frequency()
        panel._tune_index(0)
        self.assertEqual(win._current_frequency(), before)
        self.assertIn("10368.030 MHz is out of range", self.message())

    def test_a_bookmark_below_1_khz_goes_to_chirp_and_back(self):
        win = self.win
        panel = win._bookmarks_panel
        path = _tone_wav(self.path("audio.wav"), AUDIO_RATE, 2.0, freq=700.0)
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(win._open_recording())
        warn.assert_not_called()
        win._set_demod_mode("CW", None, "500 Hz")
        self.assertTrue(win.set_frequency(700.0), self.message())
        win._bookmark_current_frequency()
        made = panel.bookmark(panel.bookmark_count() - 1)
        self.assertEqual((made["freq_hz"], made["demod"]), (700.0, "CW"))
        text = self._round_trip([made])
        self.assertIn("0.000700", text)
        self.assertEqual(panel.bookmark(0)["freq_hz"], 700.0)
        win._set_demod_mode("USB", None, "2.7 kHz")
        self.assertTrue(win.set_frequency(5000.0))
        panel._tune_index(0)
        self.assertEqual(win._current_frequency(), 700.0)
        self.assertEqual(win._control_panel.get_demod_mode(), "CW")
        win._playback.close()
        panel._tune_index(0)
        self.assertIn("0.7 kHz is out of range", self.message())
