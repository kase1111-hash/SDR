"""Final polish: small cross-area problems left open by the post-merge
checks (P-01 to P-10 of the final pass).

Each test names the problem it covers; most run the real window offscreen
on the demo device (or the part on its own where that is enough).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import require_pyqt6  # noqa: E402
from tests.test_gui_recording_files import _FilesCase  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402


def _save_sigmf(path, center, rate=1e6, seconds=0.2):
    from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

    rng = np.random.default_rng(7)
    n = int(rate * seconds)
    noise = 0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    save_iq_file(
        path,
        noise.astype(np.complex64),
        rate,
        center,
        SampleFormat.FLOAT32,
        FileFormat.SIGMF,
    )
    return path


_DEVIATIONS_HZ = (2.5e3, 5e3, 12.5e3, 25e3, 75e3)


def _hz(text):
    """ "12.5 kHz" -> 12500.0"""
    value, unit = text.split()
    return float(value) * {"Hz": 1.0, "kHz": 1e3}[unit]


class TestIdentifiedDeviationDoesNotClip(unittest.TestCase):
    """P-01: Identify suggested "2.5 kHz" for the demo's 100.715 MHz call
    (+/-3.4 kHz) and "5 kHz" for its POCSAG pager (+/-4.5 kHz FSK); half a
    second after Apply Mode the window warned "FM audio is clipping". Here
    the window's own receiver chain measures what the warning measures (the
    share of audio samples beyond full scale, warned about above 5 %) at
    the deviation Identify suggests: it doesn't clip, and the next smaller
    choice would (the suggestion is no wider than it needs to be)."""

    RATE = 240e3
    SECONDS = 1.5

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window

    def clip_fraction(self, x, deviation_hz, width_hz):
        """What the window's clipping warning measures, block by block (30
        blocks a second, as the display loop hands them over)."""
        chain = self.mw._ReceiverChain(self.RATE, "FM", width_hz)
        over = total = 0.0
        block = int(self.RATE / 30)
        for start in range(0, len(x) - block + 1, block):
            channel = chain.channelize(x[start : start + block])
            audio = chain.demodulate(channel, deviation_hz)
            over += chain.last_clip_fraction * len(audio)
            total += len(audio)
        return over / max(total, 1.0)

    def check(self, signal, occupied_hz, width_hz, snr_db=30.0):
        from sdr_module.dsp.signal_id import identify

        x, _ = S.add_noise(signal, self.RATE, snr_db, occupied_hz, seed=3)
        x = x.astype(np.complex64)
        ident = identify(
            x[: int(self.RATE)], self.RATE, 300e6, search_hz=0.75 * width_hz
        )
        self.assertEqual(ident.mode, "FM", ident.summary())
        deviation = _hz(ident.fm_deviation)
        limit = self.mw._CLIP_FRACTION
        clipped = self.clip_fraction(x, deviation, width_hz)
        self.assertLessEqual(clipped, limit, (ident.fm_deviation, clipped))
        smaller = [d for d in _DEVIATIONS_HZ if d < deviation]
        if smaller:
            self.assertGreater(
                self.clip_fraction(x, smaller[-1], width_hz),
                limit,
                f"{ident.fm_deviation} is wider than it needs to be",
            )
        return ident.fm_deviation

    def test_the_demo_calls_swing(self):
        """Two tones swinging +/-3.4 kHz together, as the demo's narrow FM
        calls do (2.2 kHz + 1.2 kHz)."""
        n = int(self.SECONDS * self.RATE)
        signal = S.fm(n, self.RATE, 3.4e3, S.tones(n, self.RATE))
        self.assertEqual(self.check(signal, 10e3, 12.5e3), "5 kHz")

    def test_a_swing_just_beyond_a_choice(self):
        n = int(self.SECONDS * self.RATE)
        signal = S.fm(n, self.RATE, 2.8e3, S.tones(n, self.RATE))
        self.assertEqual(self.check(signal, 9e3, 12.5e3), "5 kHz")

    def test_wide_voice(self):
        n = int(self.SECONDS * self.RATE)
        audio = S.voice(n, self.RATE, seed=7, band=(300.0, 2500.0))
        signal = S.fm(n, self.RATE, 12e3, audio)
        self.assertEqual(self.check(signal, 30e3, 50e3), "12.5 kHz")

    def test_pager_data(self):
        """FSK at +/-4.5 kHz, 1200 baud: POCSAG as the demo sends it."""
        n = int(self.SECONDS * self.RATE)
        signal = S.fsk(n, self.RATE, 1200, 4500)
        self.assertEqual(self.check(signal, 10.2e3, 25e3), "12.5 kHz")


class TestARecordingTooSlowForTheDecoder(_FilesCase):
    """P-05: with ADS-B chosen by hand, opening a 1 MS/s recording said "ADS-B
    needs at least 2 MS/s; the device runs at 1 MS/s. Choose 2.4 MS/s or
    more in Receiver > Sample rate (or click the rate in the status bar)",
    a rate a playing recording can't change. It says the recording's rate
    is too low, as Decode Recording does; the device keeps its advice."""

    def said(self):
        """Everywhere the decoder's problem shows: the status bar, the
        decoder chip's tooltip, the Decoder's status line and empty table."""
        win = self.win
        panel = win._decoder_panel
        return [
            win._message_label.text(),
            win._decoder_chip.toolTip(),
            panel.decoder_status_text(),
            panel._empty._text,
        ]

    def choose_ads_b(self):
        self.win._decoder_panel._proto_combo.setCurrentText("ADS-B")  # by hand
        self.settle(2)

    def open_file(self, name, rate):
        path = _save_sigmf(self.path(name), 1090e6, rate)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        self.settle(2)

    def test_a_recording_says_its_rate_is_too_low(self):
        win = self.win
        self.choose_ads_b()
        self.open_file("slow.sigmf-data", 1e6)
        self.assertTrue(win._decoder_panel.is_decoding())
        self.assertEqual(win._decoder_chip.text(), "ADS-B · can't decode")
        self.assertIn(
            "ADS-B needs at least 2 MS/s; this recording is 1 MS/s.",
            win._message_label.text(),
        )
        self.assertIn("this recording is 1 MS/s", win._decoder_chip.toolTip())
        self.assertIn("this recording is 1 MS/s", win._decoder_panel._empty._text)
        for text in self.said():
            self.assertNotIn("Sample rate", text)
            self.assertNotIn("the device", text)
        # Closed: the demo device (2.4 MS/s) decodes again.
        win._playback.close()
        self.settle(2)
        self.assertEqual(win._decoder_chip.text(), "ADS-B · 0 msgs")

    def test_the_device_and_a_recording_at_the_same_rate(self):
        """Each says it its own way, although the rate stays 1 MS/s."""
        win = self.win
        self.assertTrue(win.set_sample_rate(1e6))
        self.choose_ads_b()
        tip = win._decoder_chip.toolTip()
        self.assertIn("ADS-B needs at least 2 MS/s; the device runs at 1 MS/s", tip)
        self.assertIn("Receiver > Sample rate", tip)
        self.open_file("same.sigmf-data", 1e6)
        tip = win._decoder_chip.toolTip()
        self.assertIn("ADS-B needs at least 2 MS/s; this recording is 1 MS/s", tip)
        self.assertNotIn("Sample rate", tip)
        win._playback.close()
        self.settle(2)
        self.assertEqual(win._device_sample_rate(), 1e6)
        self.assertIn("the device runs at 1 MS/s", win._decoder_chip.toolTip())


if __name__ == "__main__":
    unittest.main()
