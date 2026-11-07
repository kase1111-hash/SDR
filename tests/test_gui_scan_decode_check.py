"""The check of the scan-decode lane (M-D1, M-D2, M-S1): what it found.

- M-S1: a saved result whose "receiver" is a JSON list or object (a
  damaged settings file) stopped the window from opening ("unhashable
  type" while the Scanner tab restored its results). And a result was made
  with its receiver fields by position, after ``dcs``: a field another lane
  adds there at the same time (the reverse tone squelch's ``reverse``)
  would have taken them.
- M-D2: the busy-sky line, and the status bar's notice, came for a short
  burst of unread replies (half a second after it ended, while the sky was
  calm again) and for a few unread now and then at the edge of the budget
  (5 a second of 20,000 kept it up most of the time, with a notice every
  few seconds). Now replies must go unread for half a second, at 100 or
  more a second, before it is said; once said, it stays until it clears.
  And while a recording played at the device's own rate (its decoder isn't
  rebuilt then), the notice said to choose another sample rate.
- M-D1: a recording's aircraft never dim, however long the recording (the
  tests moved only the wall clock); a decode with another protocol keeps
  what the views hold of that recording (it started their lists over).
"""

import json
import os
import types
import unittest
from dataclasses import MISSING, field, fields, make_dataclass
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests import test_gui_recording_views as recording_views  # noqa: E402
from tests.test_adsb_skipped_replies import (  # noqa: E402
    BLOCK_S,
    _feed,
    _quiet,
    _sky,
    _status,
)
from tests.test_gui_chrome import _FakeSettings, require_pyqt6  # noqa: E402
from tests.test_gui_decoder_views import _adsb, _settle  # noqa: E402
from tests.test_gui_recording_files import _FilesCase  # noqa: E402
from tests.test_gui_recording_views import REC, WHEN, _klm, _ViewsCase  # noqa: E402

# --------------------------------------------------------------------------- #
# M-S1: the results' receiver, saved and made
# --------------------------------------------------------------------------- #


class TestDamagedSavedResults(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_a_receiver_of_another_kind_is_dropped_and_the_result_kept(self):
        from sdr_module.gui.scanner_panel import ScanHit

        for bad in ([], {}, ["main"], {"main": 1}, 7, None, True, "third"):
            data = {
                "frequency_hz": 1850e6,
                "peak_db": -40.0,
                "receiver": bad,
                "readings": {"main": -40.0},
            }
            hit = ScanHit.from_dict(data)
            self.assertIsNotNone(hit, bad)
            self.assertEqual((hit.receiver, hit.readings), ("", {}), bad)

    def test_the_scanner_tab_opens_with_them(self):
        from sdr_module.gui import scanner_panel as mod

        store = _FakeSettings()
        store.values["scanner/results"] = json.dumps(
            [
                {
                    "frequency_hz": 1850e6,
                    "peak_db": -40.0,
                    "receiver": ["second"],
                    "readings": {"second": -40.0},
                },
                {"frequency_hz": 98.1e6, "peak_db": -50.0, "receiver": {"x": 1}},
                {
                    "frequency_hz": 1720e6,
                    "peak_db": -45.0,
                    "receiver": "main",
                    "readings": {"main": -45.0},
                },
            ]
        )
        panel = mod.ScannerPanel(settings=store)
        self.addCleanup(panel.deleteLater)
        hits = sorted(panel.hits(), key=lambda h: h.frequency_hz)
        self.assertEqual([h.receiver for h in hits], ["", "main", ""])
        self.assertFalse(panel._table.isColumnHidden(mod.COL_RECEIVER))


def _with_a_field_after_dcs(cls):
    """``cls`` (ScanHit) with one more field after ``dcs``, as another lane
    adds one there (the reverse tone squelch's)."""
    spec = []
    for item in fields(cls):
        if item.default is not MISSING:
            spec.append((item.name, item.type, field(default=item.default)))
        elif item.default_factory is not MISSING:
            spec.append(
                (item.name, item.type, field(default_factory=item.default_factory))
            )
        else:
            spec.append((item.name, item.type))
        if item.name == "dcs":
            spec.append(("other_lane", bool, field(default=False)))
    namespace = {"listening": cls.listening, "read_by": cls.read_by}
    return make_dataclass(cls.__name__, spec, namespace=namespace)


class TestResultFields(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_its_receiver_is_named_when_a_result_is_made(self):
        from sdr_module.gui import scanner_panel as mod
        from sdr_module.gui.scan_engine import PARTNER, Found
        from tests.test_gui_scanner import FakeWindow

        merged = _with_a_field_after_dcs(mod.ScanHit)
        with mock.patch.object(mod, "ScanHit", merged):
            panel = mod.ScannerPanel(settings=_FakeSettings())
            self.addCleanup(panel.deleteLater)
            panel.set_host(FakeWindow())
            found = Found(1850e6, -40.0, -90.0, 12e3, ("FM", "5 kHz", "12.5 kHz"))
            found.receiver = PARTNER
            panel._on_found(found)
            hit = panel.hits()[0]
            self.assertEqual(
                (hit.receiver, hit.readings, hit.other_lane),
                ("second", {"second": -40.0}, False),
            )
            self.assertEqual(panel._table.item(0, mod.COL_RECEIVER).text(), "2nd")


# --------------------------------------------------------------------------- #
# M-D2: when a busy sky is worth saying
# --------------------------------------------------------------------------- #


class _ScriptedModeS:
    """Stands in for the ADS-B decoder: each block adds the next count of a
    script to its demodulator's unread replies."""

    frames_seen = 0
    frames_failed = 0

    def __init__(self, counts):
        self._fine = types.SimpleNamespace(skipped=0)
        self._counts = iter(counts)

    def decode(self, _magnitude):
        self._fine.skipped += next(self._counts, 0)
        return []

    def reset(self):
        pass


def _said(counts, rate=2.4e6):
    """``skipped_per_s`` after each block (BLOCK_S of signal) of a decoder
    whose blocks leave ``counts`` replies unread."""
    from sdr_module.dsp.live_decoders import ADSBLiveDecoder

    decoder = ADSBLiveDecoder(rate)
    decoder._decoder = _ScriptedModeS(counts)
    block = np.zeros(int(rate * BLOCK_S), np.complex64)
    said = []
    for _ in counts:
        decoder.process(block)
        said.append(decoder.status.skipped_per_s)
    return decoder, said


def _blocks(seconds):
    return int(round(seconds / BLOCK_S))


class TestWhenABusySkyIsSaid(unittest.TestCase):
    def test_a_short_burst_isnt_said(self):
        # A fifth of a second of a very busy sky between calm ones: some
        # replies go unread, but not for long enough to be worth a notice.
        from sdr_module.dsp.live_decoders import ADSBLiveDecoder

        rate = 2.4e6
        burst = _sky(rate)[: int(0.2 * rate)]
        mag = np.concatenate([_quiet(rate, 1.0), burst, _quiet(rate, 3.5, seed=12)])
        decoder = ADSBLiveDecoder(rate)
        statuses = _feed(decoder, mag, rate)
        self.assertGreater(decoder.skipped, 20)
        self.assertEqual(max(s.skipped_per_s for s in statuses), 0.0)

    def test_a_few_unread_now_and_then_arent_said(self):
        # At the edge of the budget: a block leaves 5 unread every 1.5 s.
        counts = ([5] + [0] * (_blocks(1.5) - 1)) * 20
        decoder, said = _said(counts)
        self.assertEqual(decoder.skipped, 100)
        self.assertEqual(max(said), 0.0)

    def test_a_few_unread_a_second_arent_said(self):
        counts = [1] * _blocks(10.0)  # (about 29 a second)
        decoder, said = _said(counts)
        self.assertEqual(decoder.skipped, len(counts))
        self.assertEqual(max(said), 0.0)

    def test_a_busy_spell_is_said_once_and_stays_until_it_clears(self):
        from sdr_module.dsp.live_decoders import (
            SKIPPED_SETTLE_S,
            SKIPPED_WINDOW_S,
        )

        calm, busy = _blocks(1.0), _blocks(2.0)
        sparse = ([1] + [0] * 14) * 4  # (then one now and then: 2 s)
        counts = [0] * calm + [30] * busy + sparse + [0] * _blocks(4.0)
        _decoder, said = _said(counts)
        shown = [i for i, rate in enumerate(said) if rate > 0]
        self.assertTrue(shown)
        # Once, from half a second into the busy spell ...
        self.assertEqual(shown, list(range(shown[0], shown[-1] + 1)))
        start = (shown[0] + 1 - calm) * BLOCK_S
        self.assertGreaterEqual(start, SKIPPED_SETTLE_S - 1e-9)
        self.assertLess(start, SKIPPED_SETTLE_S + 2 * BLOCK_S)
        self.assertGreater(said[shown[0]], 500)  # (30 a block: about 880/s)
        # ... while the rate falls, until none were left for the window.
        last_skip = calm + busy + 45  # (the block of the last one)
        self.assertAlmostEqual(
            (shown[-1] - last_skip) * BLOCK_S, SKIPPED_WINDOW_S, delta=2 * BLOCK_S
        )


class TestPlayingARecordingAtTheDevicesRate(_FilesCase):
    """A recording at the device's rate keeps the live decoder (it isn't
    rebuilt for it): the status bar's notice must still know a recording
    plays, whose rate can't be changed."""

    def test_the_notice_doesnt_say_to_choose_another_rate(self):
        from sdr_module.dsp.protocols import ProtocolType
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")  # (live first)
        rate = float(win._device_sample_rate())
        path = self.path("adsb_1090MHz.sigmf-data")
        rng = np.random.default_rng(1)
        noise = 0.01 * (rng.standard_normal(int(rate * 0.2)) + 0j)
        save_iq_file(
            path,
            noise.astype(np.complex64),
            rate,
            1090e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )
        with self.open_dialog(path):
            self.assertTrue(win._open_recording(path))
        self.addCleanup(win._playback.close, True)
        self.assertTrue(win._playback.active)
        decoder = mock.MagicMock()
        decoder.name = "ADS-B"
        decoder.process.return_value = []
        win._decoder = decoder
        win._decoder_protocol = ProtocolType.ADSB
        shown = []
        real = win._show_status_message

        def spy(text, *args, **kwargs):
            shown.append(text)
            return real(text, *args, **kwargs)

        with mock.patch.object(win, "_show_status_message", spy):
            for _ in range(3):
                decoder.status = _status(873.0)
                win._run_decoder(np.zeros(4096, np.complex64))
        said = [t for t in shown if "weren't read" in t]
        self.assertEqual(len(said), 1, shown)
        self.assertNotIn("MS/s or more", said[0])
        self.assertNotIn("MS/s or more", win._decoder_panel.decoder_status_text())


# --------------------------------------------------------------------------- #
# M-D1: a recording's lists
# --------------------------------------------------------------------------- #


class TestALongRecording(_ViewsCase):
    def test_its_aircraft_never_dim_however_long_it_is(self):
        from sdr_module.gui import decoder_views as dv
        from sdr_module.gui.themes import get_palette

        self.panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        source = "long.cs16"
        self.recorded(_klm(12000), source=source, at_s=5.0)
        dlh = _adsb(
            icao_address="3C6586", downlink_format=17, type_code=4, callsign="DLH9U"
        )
        self.recorded(dlh, source=source, at_s=3600.0)  # (an hour into it)
        view = self.panel._aircraft_view
        view.refresh()
        _settle(1)
        text = get_palette().qcolor("text")
        self.assertEqual(view.table.rowCount(), 2)
        for row in range(2):
            item = view.table.item(row, dv.AC_NAME)
            self.assertEqual(item.foreground().color(), text, item.text())
            self.assertNotIn("Not heard for over a minute", item.toolTip())
        self.assertEqual(len(view.tracker_for(source)), 2)


class TestDecodingWithAnotherProtocol(_ViewsCase):
    def test_the_other_views_keep_their_lists_of_it(self):
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        for protocol in ("CW (Morse)", "POCSAG"):
            p.recording_decode_started(REC, protocol)
            self.assertEqual(p._aircraft_view.sources(), [REC], protocol)
        p.recording_decode_started(REC, "ADS-B")  # (its own: anew)
        self.assertEqual(p._aircraft_view.sources(), [])


class TestDecodeAgainInTheWindow(_FilesCase):
    def test_a_decode_with_another_protocol_keeps_its_aircraft(self):
        from sdr_module.gui import iq_files

        # (The recording of M-D1's window test: KLM1023, 4840D6.)
        path = recording_views.TestDecodeRecordingInTheViews._adsb_file(self)
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("ADS-B")
        self.settle(2)
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        view = panel._aircraft_view
        self.assertEqual(view.sources(), [REC])
        answer = self.win._exec_dialog

        def pocsag(dialog):
            if isinstance(dialog, iq_files.DecodeOptionsDialog):
                dialog._protocol.setCurrentText("POCSAG")
            return answer(dialog)

        self.win._exec_dialog = pocsag
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        self.assertEqual(view.sources(), [REC])
        self.assertIsNotNone(view.tracker_for(REC).get("4840D6"))


if __name__ == "__main__":
    unittest.main()
