"""Decode Recording of the new file kinds, in the Decoder tab's views.

The files lane opened 24-bit, 32-bit and floating-point WAV I/Q files and
one-channel (audio) WAV files; round 1 made an audio recording decodable as
a receiver's audio in File > Decode Recording; the scan-decode lane gave a
recording's aircraft and Morse text lists of their own in the Aircraft and
Text views ("Showing:", timed by the recording's clock). Together (a
post-merge check):

- A 24-bit, 32-bit or float I/Q WAV file's aircraft and Morse text are
  listed under its name, timed into the file (a WAV file doesn't say when
  it was made), as a 16-bit or SigMF file's are; the summary counts what
  the view holds.
- A one-channel audio recording's Morse (at its pitch) is listed in the
  Text view under its name; its pages and packets (its sound read as a
  receiver's audio) go to the table, marked as the file's, with the time
  into it and its name; ADS-B isn't offered for it (no ADS-B from audio),
  so nothing goes to the Aircraft view. The summary and the views agree.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_audio_decoding import EXPECTED, _audio  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_wav_formats import _cw_audio, _WavCase  # noqa: E402
from tests.wav_builders import iq_wav  # noqa: E402

#: (kind, bytes a sample, container) of the wide and float formats.
WIDE = [
    ("int", 3, "RIFF"),
    ("int", 4, "RF64"),
    ("float", 4, "W64"),
    ("float", 8, "RIFF"),
]


def _adsb_iq(rate=2.4e6, seconds=0.25):
    """A quarter of a second of the demo airliner's replies (KLM1023: its
    identification, two positions, its velocity and a squawk)."""
    from sdr_module.gui import device_dialog as dd

    n = int(rate * seconds)
    rng = np.random.default_rng(3)
    x = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for k, message in enumerate(("ident", "even", "velocity", "odd", "squawk")):
        t = 0.02 + 0.04 * k
        pulses = dd._modes_pulses(dd._aircraft_frame(message, t))
        first, env = dd._pulse_envelope(pulses, t * rate, rate)
        x[first : first + len(env)] += 0.5 * env
    return x.astype(np.complex64)


def _cw_iq(rate=16e3):
    """ "CQ DE N0CALL", a pause, "TU 73" at 20 WPM, keyed at 0 Hz."""
    from sdr_module.dsp import protocol_encoders as enc

    first = enc.morse_keying("CQ DE N0CALL", rate, 20, lead_s=0.3, tail_s=3.5)
    second = enc.morse_keying("TU 73", rate, 20, lead_s=0.0, tail_s=0.5)
    key = np.concatenate([first, second])
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(key)) + 1j * rng.standard_normal(len(key))
    return (0.5 * key + 0.002 * noise).astype(np.complex64)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class _Case(_WavCase):
    def decode(self, protocol, path, edit=None):
        """File > Decode Recording of ``path``, ``protocol`` chosen in the
        Decoder tab; ``edit(dialog)`` before Decode. Returns the dialog."""
        from sdr_module.gui import iq_files

        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText(protocol)
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
        self.settle(2)
        return dialogs[0]

    def rows(self, name):
        return [m for m in self.win._decoder_panel._messages if m["source"] == name]


class TestWideAndFloatIQ(_Case):
    def test_their_aircraft_are_listed_under_their_names(self):
        from sdr_module.gui import decoder_views as dv

        panel = self.win._decoder_panel
        view = panel._aircraft_view
        names = []
        for kind, width, container in WIDE:
            name = f"adsb_{kind}{8 * width}.wav"
            with self.subTest(name):
                path = self._wav(
                    name, _adsb_iq(), kind, width, container, 1090e6, rate=2.4e6
                )
                self.decode("ADS-B", path)
                names.append(name)
                self.assertEqual(view.sources(), names)
                self.assertEqual(view.shown_source(), name)  # (the newest)
                aircraft = view.tracker_for(name).get("4840D6")
                self.assertEqual(
                    (aircraft.callsign, aircraft.altitude, aircraft.squawk),
                    ("KLM1023", 38000, "3472"),
                )
                # The summary counts what the view holds.
                self.assertIn(
                    f"Decoded {aircraft.messages} ADS-B message(s) from {name}",
                    self.message(),
                )
                self.assertEqual(len(self.rows(name)), aircraft.messages)
                # Timed into the file (a WAV file doesn't say when it was made).
                row = [
                    view.table.item(0, c).text()
                    for c in range(view.table.columnCount())
                ]
                self.assertEqual(row[dv.AC_SEEN], "0:00")
                self.assertIn(f"into {name}", view.detail_text("4840D6"))
        self.assertEqual(len(view.tracker), 0)  # (nothing heard live)

    def test_their_morse_text_is_listed_under_their_names(self):
        panel = self.win._decoder_panel
        view = panel._cw_view
        for kind, width, container in WIDE[:3]:
            name = f"cw_{kind}{8 * width}.wav"
            with self.subTest(name):
                path = self._wav(name, _cw_iq(), kind, width, container, 7.03e6, 16000)
                self.decode("CW (Morse)", path)
                self.assertEqual(view.shown_source(), name)
                lines = view.plain_text(name).split("\n")
                self.assertEqual(
                    [line.split("  ", 1)[1] for line in lines],
                    ["CQ DE N0CALL", "TU 73"],
                )
                self.assertEqual(
                    [line.split("  ")[0] for line in lines], ["0:10", "0:14"]
                )
                self.assertIn(
                    f"Decoded 2 CW (Morse) message(s) from {name}", self.message()
                )
                self.assertEqual(view.speed.text(), "20 WPM")
        self.assertEqual(view.plain_text(), "")  # (nothing heard live)


class TestAudioRecordings(_Case):
    def test_its_morse_is_listed_under_its_name(self):
        panel = self.win._decoder_panel
        for kind, width in (("int", 2), ("int", 3), ("float", 4)):
            name = f"cw_audio_{kind}{8 * width}.wav"
            with self.subTest(name):
                path = self.path(name)
                iq_wav(path, _cw_audio(700.0), kind, width, 48000)
                self.decode(
                    "CW (Morse)", path, lambda d: d.set_frequency_text("700 Hz")
                )
                view = panel._cw_view
                self.assertEqual(view.shown_source(), name)
                # Its line when the file ended (12.92 s), timed into it.
                self.assertEqual(view.plain_text(name), "0:12  CQ CQ DE N0CALL K")
                self.assertIn(
                    f"Decoded 1 CW (Morse) message(s) from {name}", self.message()
                )
                rows = self.rows(name)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["protocol"], "CW (Morse) (file)")
                self.assertIn(f"(from {name} at 12.9 s)", rows[0]["content"])
        self.assertFalse(panel._aircraft_view.has_recordings())

    def test_its_pages_and_packets_go_to_the_table(self):
        panel = self.win._decoder_panel
        for protocol, rate in (("POCSAG", 22050), ("AX.25/APRS", 8000)):
            name = f"{protocol[:4]}_audio.wav"
            with self.subTest(name):
                path = self.path(name)
                iq_wav(path, _audio(protocol, rate, 0.3, -1.0), "int", 2, rate)
                self.decode(protocol, path)
                rows = self.rows(name)
                self.assertEqual(
                    [(m["address"], m["content"].split("  (from")[0]) for m in rows],
                    EXPECTED[protocol],
                )
                for row in rows:
                    self.assertEqual(row["protocol"], f"{protocol} (file)")
                    self.assertRegex(row["content"], rf"\(from {name} at \d+\.\d s\)")
                    self.assertEqual(row["sample_rate"], rate)  # (the file's own)
                self.assertIn(
                    f"Decoded {len(rows)} {protocol} message(s) from {name}, read as "
                    "a receiver's audio.",
                    self.message(),
                )
        # Neither view holds anything of theirs.
        self.assertFalse(panel._aircraft_view.has_recordings())
        self.assertFalse(panel._cw_view.has_recordings())

    def test_ads_b_isnt_offered_for_it(self):
        panel = self.win._decoder_panel
        name = "audio.wav"
        path = self.path(name)
        iq_wav(path, _cw_audio(700.0), "int", 2, 48000)
        offered = []

        def look(dialog):
            index = dialog._protocol.findText("ADS-B")
            item = dialog._protocol.model().item(index)
            offered.append((item.isEnabled(), dialog._protocol.currentText()))

        self.decode("ADS-B", path, look)
        enabled, chosen = offered[0]
        self.assertFalse(enabled)  # (greyed out: its rate is far too low)
        self.assertNotEqual(chosen, "ADS-B")
        self.assertFalse(panel._aircraft_view.has_recordings())
        self.assertEqual(len(panel._aircraft_view.tracker), 0)


if __name__ == "__main__":
    unittest.main()
