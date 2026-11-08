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
    fmt_body,
    full_scale,
    integers,
    interleave,
    write_wav,
)


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


if __name__ == "__main__":
    unittest.main()
