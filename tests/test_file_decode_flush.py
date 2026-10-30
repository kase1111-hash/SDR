"""Decode Recording hands over what the decoder still holds at the end.

A live decoder can hold a message until it is complete (the last line of
Morse code waits for a long gap). The command line's ``decode`` flushes the
decoder after the last block; the window's Decode Recording must too, or a
recording that ends right after its last word loses that word.
"""

import os
import tempfile
import unittest

import numpy as np

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.recording import SampleFormat


class TestDecodeRecordingFlushes(unittest.TestCase):
    RATE = 240e3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _cw_file(self, text: str) -> str:
        """A cf32 recording of ``text`` in Morse at 20 WPM, the carrier at
        the centre, ending 50 ms after the last element (no closing gap)."""
        key = enc.morse_keying(text, self.RATE, 20, edge_s=0.005)
        # Trim the keying's own trailing silence, then keep 50 ms.
        last = int(np.flatnonzero(np.abs(key) > 0.01)[-1])
        key = key[: last + int(0.05 * self.RATE)]
        lead = np.zeros(int(0.3 * self.RATE), dtype=np.complex64)
        rng = np.random.default_rng(7)
        signal = np.concatenate([lead, 0.5 * key]).astype(np.complex64)
        noise = 0.002 * (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        )
        path = os.path.join(self.tmp.name, "cw.cf32")
        (signal + noise).astype(np.complex64).tofile(path)
        return path

    def test_the_last_word_of_morse_is_kept(self):
        from sdr_module.gui.file_jobs import decode_recording
        from sdr_module.gui.iq_files import IqFileOptions

        path = self._cw_file("CQ DE TEST")
        options = IqFileOptions(
            path=path,
            sample_format=SampleFormat.FLOAT32,
            sample_rate=self.RATE,
            center_hz=7.030e6,
            total_samples=os.path.getsize(path) // 8,
            source="test",
            confirmed=True,
        )
        result = decode_recording(options, "CW (Morse)", 0.0, lambda _f: None)
        text = " ".join(
            str(getattr(msg, "text", "") or getattr(msg, "content", ""))
            for msg, _t in result.messages
        )
        self.assertIn("TEST", text, f"decoded: {text!r}")


if __name__ == "__main__":
    unittest.main()
