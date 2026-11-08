"""M-D2: ADS-B says when a very busy sky makes it skip replies.

Below 2.5 MS/s the Mode S decoder's work per second of signal is bounded
(J-decode: ``ModeSDemodulator.FITS_PER_S``, ``SEARCHES_PER_S``): in a very
busy sky it reads the clearest replies first and leaves the rest, counted
in ``ModeSDemodulator.skipped``, which nothing showed. Now the live decoder
exposes the count (``ADSBLiveDecoder.skipped``, and in its status
``skipped`` and ``skipped_per_s``, how many a second over the last few
seconds of signal); the Decoder tab's status line says so while it lasts,
in plain words (how many a second, that the clearest are read first, and
what helps), and clears a few seconds after it stops; the status bar says
it once per episode; Decode Recording's summary counts them.

The busy sky is synthetic (``tests/test_adsb_bounded_cost.py``'s): 16,000
Mode S and 24,000 Mode A/C replies a second at random times and levels.
"""

import math
import os
from unittest import mock

import numpy as np
import pytest

from sdr_module.dsp import live_decoders
from sdr_module.dsp.live_decoders import ADSBLiveDecoder, LiveDecoderStatus
from tests.test_adsb_bounded_cost import _busy_sky, _noise

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_recording_files import _FilesCase  # noqa: E402

BLOCK_S = 0.034  # one display frame


def _feed(decoder, mag, rate, block_s=BLOCK_S):
    """Feed ``mag`` a display frame at a time; the status after each."""
    block = int(rate * block_s)
    statuses = []
    x = np.asarray(mag, dtype=np.float32).astype(np.complex64)
    for start in range(0, len(x), block):
        decoder.process(x[start : start + block])
        statuses.append(decoder.status)
    return statuses


_SKIES = {}


def _sky(rate, seconds=0.6, seed=3):
    key = (rate, seconds, seed)
    if key not in _SKIES:
        _SKIES[key] = _busy_sky(
            rate, seconds, seed, modes_per_s=16000, fruit_per_s=24000
        )
    return _SKIES[key]


def _quiet(rate, seconds, seed=11):
    rng = np.random.default_rng(seed)
    return np.abs(_noise(rate, seconds, rng) * math.sqrt(rate / 2e6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# The live decoder counts them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rate", [2.0e6, 2.4e6])
def test_the_live_decoder_counts_the_replies_it_skips(rate):
    decoder = ADSBLiveDecoder(rate)
    statuses = _feed(decoder, _sky(rate), rate)
    skipped = decoder._decoder._fine.skipped
    assert skipped > 100  # (the budget was reached)
    assert decoder.skipped == skipped
    last = statuses[-1]
    assert last.skipped == skipped
    # How many a second: over the signal so far (0.6 s), within reason.
    assert 0.5 * skipped / 0.6 < last.skipped_per_s < 2.0 * skipped / 0.6
    assert all(s.skipped_per_s >= 0 for s in statuses)


def test_a_calm_sky_skips_nothing():
    rate = 2.4e6
    decoder = ADSBLiveDecoder(rate)
    sky = _busy_sky(rate, 0.5, seed=5)  # 2,000 Mode S + 6,000 Mode A/C
    statuses = _feed(decoder, sky, rate)
    assert decoder.skipped == 0
    assert all(s.skipped == 0 and s.skipped_per_s == 0 for s in statuses)


def test_it_clears_a_few_seconds_after_it_stops():
    rate = 2.4e6
    decoder = ADSBLiveDecoder(rate)
    busy = _feed(decoder, _sky(rate), rate)
    assert busy[-1].skipped_per_s > 0
    after = _feed(decoder, _quiet(rate, 4.0), rate)
    seconds = [(i + 1) * BLOCK_S for i, s in enumerate(after) if s.skipped_per_s > 0]
    # Said for a few seconds of signal after the last skip, then not.
    assert seconds and seconds[-1] <= live_decoders.SKIPPED_WINDOW_S + BLOCK_S
    assert seconds[-1] >= 1.0
    assert after[-1].skipped_per_s == 0
    assert after[-1].skipped == busy[-1].skipped  # (the count stays)


def test_a_busy_spell_says_its_own_rate_once_it_has_lasted_a_moment():
    # The status bar says it once, when it starts: the rate then must be
    # the busy sky's, not diluted by the calm seconds before it, nor by the
    # budget the decoder kept for a burst (the first blocks leave few).
    rate = 2.4e6
    decoder = ADSBLiveDecoder(rate)
    calm = _feed(decoder, _quiet(rate, 2.5), rate)
    assert calm[-1].skipped_per_s == 0
    busy = _feed(decoder, _sky(rate), rate)
    said = [i for i, s in enumerate(busy) if s.skipped_per_s > 0]
    assert said and said == list(range(said[0], len(busy)))  # (then all along)
    assert said[0] * BLOCK_S >= live_decoders.SKIPPED_SETTLE_S - BLOCK_S
    whole = busy[-1].skipped / 0.6  # (the busy sky's own rate)
    first = busy[said[0]].skipped_per_s
    assert 0.6 * whole < first < 1.5 * whole, (first, whole)


def test_from_2_5_ms_s_nothing_is_skipped():
    rate = 2.56e6
    decoder = ADSBLiveDecoder(rate)
    statuses = _feed(decoder, _sky(rate, 0.3), rate)
    assert decoder.skipped == 0
    assert statuses[-1].skipped_per_s == 0


def test_reset_counts_from_zero():
    rate = 2.4e6
    decoder = ADSBLiveDecoder(rate)
    _feed(decoder, _sky(rate), rate)
    assert decoder.skipped > 0
    decoder.reset()
    assert decoder.skipped == 0
    assert decoder.status.skipped_per_s == 0


def test_the_status_is_backwards_compatible():
    # Every decoder's status has the new fields, 0 by default.
    status = LiveDecoderStatus(name="POCSAG")
    assert (status.skipped, status.skipped_per_s) == (0, 0.0)
    pocsag = live_decoders.create_live_decoder("POCSAG", 1e6)
    assert (pocsag.status.skipped, pocsag.status.skipped_per_s) == (0, 0.0)
    slow = ADSBLiveDecoder(1.8e6)  # (too slow to decode: no demodulator)
    assert slow.skipped == 0 and slow.status.skipped_per_s == 0


# --------------------------------------------------------------------------- #
# What it says
# --------------------------------------------------------------------------- #


def test_the_words_say_how_many_that_the_clearest_were_read_and_what_helps():
    text = live_decoders.skipped_text(873.0, 2.4e6)
    assert "about 870 replies a second" in text
    assert "clearest" in text
    assert "2.56 MS/s or more" in text
    assert "Receiver > Sample rate" in live_decoders.skipped_tip(873.0, 2.4e6)
    # Below 2.4 MS/s: 2.4 MS/s already reads more of them.
    low = live_decoders.skipped_text(4519.0, 2.048e6)
    assert "about 4,500 replies a second" in low
    assert "2.4 MS/s" in low and "2.56 MS/s" in low
    # A recording's rate can't be changed: no advice to change it.
    played = live_decoders.skipped_text(873.0, 2.4e6, recording=True)
    assert "clearest" in played
    assert "Sample rate" not in live_decoders.skipped_tip(873.0, 2.4e6, recording=True)
    assert live_decoders.skipped_text(0.0, 2.4e6) == ""
    assert "about 5 replies a second" in live_decoders.skipped_text(4.6, 2.4e6)


# --------------------------------------------------------------------------- #
# The Decoder tab and the window
# --------------------------------------------------------------------------- #


def _status(per_s, skipped=100, **fields):
    base = dict(name="ADS-B", frames=40, valid=30, sample_rate=2.4e6)
    base.update(fields)
    return LiveDecoderStatus(skipped=skipped, skipped_per_s=per_s, **base)


@pytest.fixture
def panel():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from sdr_module.gui.decoder_panel import DecoderPanel

    app = QApplication.instance() or QApplication([])
    p = DecoderPanel()
    p._clock = lambda: 1000.0
    p.resize(420, 520)
    p.show()
    app.processEvents()
    p._proto_combo.setCurrentText("ADS-B")
    p.set_receiving(True)
    p.set_current_frequency(1090e6)
    yield p
    p.close()
    p.deleteLater()
    app.processEvents()


def test_the_status_line_says_so_while_it_lasts(panel):
    panel.set_decoder_status(_status(0.0, skipped=0))
    calm = panel.decoder_status_text()
    assert "weren't read" not in calm
    panel.set_decoder_status(_status(873.0))
    busy = panel.decoder_status_text()
    assert busy.startswith("Busy sky: about 870 replies a second weren't read")
    assert "clearest" in busy and "2.56 MS/s or more" in busy
    assert panel._status_label.property("tone") == "info"
    tip = panel._status_label.toolTip()
    assert "Receiver > Sample rate" in tip
    assert "30 decoded" in tip  # (the counts, in the tooltip meanwhile)
    # Clears with the decoder's rate (a few seconds of signal after it stops).
    panel.set_decoder_status(_status(0.0, skipped=900))
    assert panel.decoder_status_text() == calm


def test_a_recording_playing_isnt_told_to_change_the_rate(panel):
    panel.set_source("recording")
    panel.set_decoder_status(_status(873.0))
    assert "clearest" in panel.decoder_status_text()
    assert "Sample rate" not in panel._status_label.toolTip()


class TestTheWindow(_FilesCase):
    """The status bar says it once per episode; Decode Recording counts.
    (A window with the demo device, its dialogs answered.)"""

    def _run_with(self, statuses):
        """Run frames whose decoder status is each of ``statuses``."""
        from sdr_module.dsp.protocols import ProtocolType

        win = self.win
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("ADS-B")
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
            for status in statuses:
                decoder.status = status
                win._run_decoder(np.zeros(4096, np.complex64))
        return [t for t in shown if "weren't read" in t]

    def test_the_status_bar_says_it_once_per_episode(self):
        busy = [_status(873.0)] * 5
        calm = [_status(0.0)] * 3
        said = self._run_with(busy + calm + busy)
        self.assertEqual(len(said), 2, said)
        self.assertTrue(said[0].startswith("Busy sky: about 870 replies"))
        # Cleared from the status bar when it ends.
        self._run_with(calm)
        self.assertNotIn("weren't read", self.win._message_label.text())

    def test_a_new_decoder_starts_a_new_episode(self):
        # Turned off (or rebuilt: another rate, a recording starting) mid
        # spell: what the old decoder said goes, and the new one says it
        # again when its sky is busy.
        busy = [_status(873.0)] * 5
        self.assertEqual(len(self._run_with(busy)), 1)
        self.win._on_decoder_protocol_changed("Off")
        self.assertNotIn("weren't read", self.win._message_label.text())
        self.assertEqual(len(self._run_with(busy)), 1)

    def test_no_rate_to_raise_while_the_signal_is_skipped(self):
        """R7W-05 (final walk): while the demo device (or this computer, or
        the device) can't keep up, the Decoder tab's line says the decoder
        misses part of the signal and that a lower sample rate helps. The
        status bar's busy-sky notice said "2.56 MS/s or more reads every
        one" at the same time: the opposite advice, and more work for a
        computer already behind (from 2.5 MS/s the decoder's work has no
        limit). While the signal is skipped, the notice now says what it
        saw, with no rate to choose."""
        win = self.win
        panel = win._decoder_panel
        calm = [_status(0.0)] * 3
        self._run_with(calm)  # (the decoder runs, in a calm sky)
        with mock.patch.object(win, "_samples_skipped", return_value=True):
            win._sync_decoder_skipping()
        hint = panel.decoder_status_text()
        self.assertIn("A lower sample rate (Receiver > Sample rate) helps", hint)
        said = self._run_with([_status(873.0)] * 5)
        self.assertEqual(
            said,
            [
                "Busy sky: about 870 replies a second weren't read (the "
                "clearest are read first)."
            ],
        )
        self.assertEqual(panel.decoder_status_text(), hint)  # (as before)
        # Keeping up again, the next busy spell says what helps, as before.
        self._run_with(calm)
        win._skip_hint_until = 0.0  # (the hint's hold over)
        with mock.patch.object(win, "_samples_skipped", return_value=False):
            win._sync_decoder_skipping()
        self.assertNotIn("lower sample rate", panel.decoder_status_text())
        said = self._run_with([_status(873.0)] * 5)
        self.assertEqual(len(said), 1, said)
        self.assertTrue(said[0].endswith("2.56 MS/s or more reads every one."))

    def test_decode_recording_counts_them(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        rate = 2.4e6
        path = self.path("busy_sky.sigmf-data")
        sky = _sky(rate)
        save_iq_file(
            path,
            (sky / (1.05 * sky.max())).astype(np.complex64),  # (16-bit: in range)
            rate,
            1090e6,
            SampleFormat.INT16,
            FileFormat.SIGMF,
        )
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        self.answer()
        with self.open_dialog(path):
            self.assertTrue(win._decode_recording())
        message = win._message_label.text()
        self.assertIn("ADS-B message(s) from busy_sky.sigmf-data", message)
        self.assertRegex(message, r"[\d,]+ replies weren't read")
        self.assertIn("clearest", message)


def test_decode_recording_returns_the_count(tmp_path):
    from sdr_module.dsp.recording import SampleFormat
    from sdr_module.gui.file_jobs import decode_recording
    from sdr_module.gui.iq_files import IqFileOptions

    rate = 2.4e6
    path = str(tmp_path / "busy.cf32")
    _sky(rate).astype(np.complex64).tofile(path)
    options = IqFileOptions(
        path=path,
        sample_format=SampleFormat.FLOAT32,
        sample_rate=rate,
        center_hz=1090e6,
        total_samples=os.path.getsize(path) // 8,
        source="test",
        confirmed=True,
    )
    result = decode_recording(options, "ADS-B", 0.0, lambda _f: None)
    assert result.skipped > 100
    quiet = str(tmp_path / "quiet.cf32")
    _quiet(rate, 0.2).astype(np.complex64).tofile(quiet)
    options.path, options.total_samples = quiet, os.path.getsize(quiet) // 8
    assert decode_recording(options, "ADS-B", 0.0, lambda _f: None).skipped == 0
