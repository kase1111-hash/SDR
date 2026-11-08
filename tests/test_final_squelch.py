"""Final review (R7): a reverse tone squelch and the squelch's hang time.

A reverse tone squelch (CHIRP's TSQL-R and DTCS-R) mutes the stations that
send its tone or code and plays every other one, once it is sure the tone
isn't there (``ToneSquelch.REVERSE_DECIDE_S``, 0.55 s of audio). When the
station it mutes was heard only for the last half second of a transmission
(tuned then, or reached by Scan bookmarks then), its tone or code wasn't
recognised yet when its carrier dropped, and that wait ran out on the noise
the squelch's hang time (0.3 s) keeps open after a carrier drops:

* R7W-03: the speaker played that noise, the muted station's squelch tail
  (about 0.13 s, clipped), in both receivers;
* R7W-01: Scan bookmarks listed the TSQL-R bookmark as a hit on it, at
  about -84 dBFS and 4 dB, and stopped to listen to noise.

The hang's noise is no station without the tone: a reverse tone squelch
now opens only on a carrier really there (``SquelchGate.hanging``,
``ToneSquelch.process(audio, carrier)``). A station it plays still plays
through the hang (a fade, or its own squelch tail), as with no tone squelch,
and a plain tone squelch is as it was.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402
from tests.test_gui_squelch import pretend_audio_output  # noqa: E402
from tests.test_gui_tone_reverse import _on_at  # noqa: E402
from tests.voice_synth import pitched_voice  # noqa: E402

AUDIO = 48000.0
TEN_MS = 480
#: How long before a transmission ends it is tuned (s): its tone or code
#: isn't recognised before its carrier drops (it takes 0.35-0.5 s), and the
#: 0.55 s wait would end within the squelch's 0.3 s hang.
LEADS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)


# ---------------------------------------------------------------------------
# The decision (no window)
# ---------------------------------------------------------------------------


def _voice(seconds, seed=1):
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, AUDIO, seed=seed)
    return 0.6 * SosFilter(highpass_sos(300.0, AUDIO)).process(voice)


def _ending(seconds, ctcss_hz=None, dcs=None):
    """FM discriminator audio of the last ``seconds`` of a transmission:
    voice and its CTCSS tone ending in the reverse burst (as the demo's 2 m
    repeater), or its DCS code ending in the turn-off code (as the 70 cm
    one); with neither, voice alone."""
    from sdr_module.dsp.dcs import DCS_TURN_OFF_S, dcs_transmission
    from sdr_module.dsp.filters import CTCSS_BURST_S, ctcss_transmission

    n = int(round(seconds * AUDIO))
    if dcs is not None:
        end = int(round(DCS_TURN_OFF_S * AUDIO))
        sub = 0.15 * dcs_transmission(*dcs, rate=AUDIO, samples=n - end)
    elif ctcss_hz is not None:
        end = int(round(CTCSS_BURST_S * AUDIO))
        sub = 0.12 * ctcss_transmission(ctcss_hz, AUDIO, samples=n - end)
    else:
        end, sub = 0, np.zeros(n)
    voice = np.zeros(n)
    voice[: n - end] = _voice((n - end) / AUDIO)
    return voice + sub[:n]


def _dropped(seconds, seed=4):
    """The discriminator's audio once the carrier has dropped: noise."""
    return 0.5 * np.random.default_rng(seed).standard_normal(int(seconds * AUDIO))


def _squelch(tone_hz=None, dcs=None, reverse=True):
    from sdr_module.gui.audio_chain import ToneSquelch

    tones = ToneSquelch(AUDIO)
    tones.tone_hz, tones.dcs, tones.reverse = tone_hz, dcs, reverse
    return tones


def _run(tones, x, carrier=True):
    """Whether each 10 ms block may play (DCS fed first, as the window
    does); ``carrier``: the squelch's view (False: open for its hang)."""
    plays = []
    for pos in range(0, len(x), TEN_MS):
        piece = x[pos : pos + TEN_MS]
        tones.feed_dcs(piece)
        plays.append(bool(tones.process(piece, carrier=carrier)))
    return plays


class TestTheGateSaysWhenItHangs(unittest.TestCase):
    """``SquelchGate.hanging``: open only for its hang time (the level fell
    below its closing level: the carrier dropped, or faded away)."""

    def gate(self, **kwargs):
        from sdr_module.gui.audio_chain import SquelchGate

        return SquelchGate(manual_db=-60.0, auto=False, **kwargs)

    def test_while_open_and_below_its_closing_level(self):
        gate = self.gate()
        self.assertFalse(gate.hanging)  # (closed)
        gate.update(-40.0, None, 0.033)
        self.assertTrue(gate.is_open)
        self.assertFalse(gate.hanging)
        gate.update(-62.0, None, 0.033)  # (within the 3 dB hysteresis)
        self.assertFalse(gate.hanging)
        gate.update(-70.0, None, 0.033)  # the carrier dropped
        self.assertTrue(gate.is_open)
        self.assertTrue(gate.hanging)
        gate.update(-40.0, None, 0.033)  # back: a fade, or another station
        self.assertFalse(gate.hanging)
        for _ in range(9):
            gate.update(-70.0, None, 0.033)
        self.assertTrue(gate.is_open and gate.hanging)
        gate.update(-70.0, None, 0.033)  # (0.33 s: closed)
        self.assertFalse(gate.is_open)
        self.assertFalse(gate.hanging)

    def test_never_when_off_or_reset(self):
        gate = self.gate(enabled=False)
        for level in (-40.0, -70.0, -90.0):
            gate.update(level, None, 0.033)
            self.assertFalse(gate.hanging)
        gate = self.gate()
        gate.update(-40.0, None, 0.033)
        gate.update(-70.0, None, 0.033)
        gate.reset()
        self.assertFalse(gate.hanging)


class TestReverseOpensOnlyOnACarrier(unittest.TestCase):
    def test_the_muted_stations_end_isnt_heard_in_the_hang(self):
        """Heard only for the last ``lead`` seconds before its carrier
        dropped, then the hang's noise: never played (each played 0.04 to
        0.3 s of that noise before)."""
        for lead in LEADS:
            for tone_hz, dcs in ((100.0, None), (None, (23, False))):
                tones = _squelch(tone_hz, dcs)
                ending = _ending(lead, tone_hz, dcs)
                plays = _run(tones, ending) + _run(tones, _dropped(0.3), False)
                self.assertNotIn(True, plays, (lead, tone_hz, dcs))

    def test_a_station_it_plays_plays_through_the_hang(self):
        """A station without the tone, playing: a fade (or its own squelch
        tail) keeps playing for the hang, as with no tone squelch; when the
        level comes back it plays on, without waiting again."""
        for tone_hz, dcs in ((100.0, None), (None, (23, False))):
            tones = _squelch(tone_hz, dcs)
            voice = _voice(2.0, seed=2)
            self.assertTrue(_run(tones, voice[: int(1.2 * AUDIO)])[-1])
            self.assertNotIn(False, _run(tones, _dropped(0.2), False), dcs)
            self.assertNotIn(False, _run(tones, voice[int(1.2 * AUDIO) :]), dcs)

    def test_the_tone_it_mutes_stays_muted_in_the_hang(self):
        for burst in (True, False):
            tones = _squelch(100.0)
            x = (
                _ending(1.5, 100.0)
                if burst
                else _ending(1.5)
                + 0.12 * np.cos(2 * np.pi * 100.0 * np.arange(int(1.5 * AUDIO)) / AUDIO)
            )
            plays = _run(tones, x) + _run(tones, _dropped(0.3), False)
            self.assertNotIn(True, plays, burst)

    def test_a_plain_tone_squelch_doesnt_look_at_the_carrier(self):
        x = np.concatenate((_ending(1.5, 100.0), _dropped(0.3)))
        for tone_hz in (100.0, 123.0, None):
            with_carrier = _run(_squelch(tone_hz, reverse=False), x)
            without = _run(_squelch(tone_hz, reverse=False), x, False)
            self.assertEqual(with_carrier, without, tone_hz)
        code = np.concatenate((_ending(1.5, dcs=(23, False)), _dropped(0.3)))
        with_carrier = _run(_squelch(dcs=(23, False), reverse=False), code)
        without = _run(_squelch(dcs=(23, False), reverse=False), code, False)
        self.assertEqual(with_carrier, without)
        self.assertIn(True, with_carrier)

    def test_carrier_is_assumed_when_not_said(self):
        """(Backwards compatible: ``process(audio)`` as before.)"""
        tones = _squelch(100.0)
        x = _voice(1.5, seed=3)
        plays = []
        for pos in range(0, len(x), TEN_MS):
            tones.feed_dcs(x[pos : pos + TEN_MS])
            plays.append(bool(tones.process(x[pos : pos + TEN_MS])))
        self.assertTrue(plays[-1])
        self.assertFalse(plays[int(0.5 * AUDIO / TEN_MS)])


# ---------------------------------------------------------------------------
# The Second Receiver's signal path
# ---------------------------------------------------------------------------

RATE = 2.4e6
BLOCK = 80_000


def _iq(audio):
    """Narrow FM (5 kHz deviation) of 48 kHz discriminator audio at 2.4 MS/s
    with a little noise."""
    held = np.repeat(np.asarray(audio, dtype=np.float64), int(RATE / AUDIO))
    phase = 2 * np.pi * np.cumsum(5e3 * held) / RATE
    rng = np.random.default_rng(6)
    noise = 1e-3 * (
        rng.standard_normal(len(held)) + 1j * rng.standard_normal(len(held))
    )
    return (0.3 * np.exp(1j * phase) + noise).astype(np.complex64)


def _quiet(seconds, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    return (1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )


class _DspCase(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")

    @staticmethod
    def dsp(choice):
        from sdr_module.gui.second_receiver import ReceiverDsp

        dsp = ReceiverDsp()
        dsp.rate, dsp.mode, dsp.bandwidth_hz, dsp.deviation_hz = RATE, "FM", 25e3, 5e3
        dsp.tone_squelch = choice
        dsp.options.tone_squelch = True
        return dsp

    @staticmethod
    def listen(dsp, iq):
        """Each frame's (state, audio peak)."""
        out = []
        for start in range(0, len(iq) - BLOCK + 1, BLOCK):
            frame = dsp.process(iq[start : start + BLOCK], True)
            audio = frame.audio
            peak = (
                float(np.max(np.abs(audio)))
                if audio is not None and len(audio)
                else 0.0
            )
            out.append((frame.state, peak))
        return out


class TestTheSecondReceiversSignalPath(_DspCase):
    def test_the_muted_stations_end_isnt_heard_in_the_hang(self):
        """(It played 0.1 to 0.2 s of clipped noise, peaks 1.2 to 2.1, at
        leads of 0.3 to 0.45 s.)"""
        quiet = _quiet(1.5)
        for lead in LEADS:
            for choice, sub in (
                ((100.0, None, True), {"ctcss_hz": 100.0}),
                ((None, (23, False), True), {"dcs": (23, False)}),
            ):
                iq = np.concatenate(
                    (quiet[: 6 * BLOCK], _iq(_ending(lead, **sub)), quiet)
                )
                frames = self.listen(self.dsp(choice), iq)
                self.assertEqual([p for _s, p in frames if p > 1e-3], [], (lead, sub))
                self.assertNotIn("open", [s for s, _p in frames], (lead, sub))

    def test_a_station_it_plays_plays_through_a_fade(self):
        """A station without the tone dips into the noise for 0.13 s (the
        squelch's hang keeps it open): heard throughout, as before."""
        voice = _iq(_voice(3.0, seed=2))
        cut = 45 * BLOCK
        iq = np.concatenate((voice[:cut], _quiet(4 * BLOCK / RATE), voice[cut:]))
        frames = self.listen(self.dsp((100.0, None, True)), iq)
        states = [s for s, _p in frames]
        first = states.index("open")
        self.assertLessEqual(first * BLOCK / RATE, 0.6)
        self.assertEqual(set(states[first:]), {"open"})
        self.assertTrue(all(p > 1e-3 for _s, p in frames[first:]))


# ---------------------------------------------------------------------------
# The windows, on the demo's repeaters
# ---------------------------------------------------------------------------


def _demo_lag():
    from sdr_module.gui import device_dialog

    return mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheMainWindow(_WindowTestCase):
    """The main window tuned to a demo repeater with its tone squelch
    reversed on the tone or code that repeater sends, ``lead`` seconds before
    one of its transmissions ends (R7W-03: 0.13 s of clipped noise)."""

    demo = True

    def setUp(self):
        super().setUp()
        win = self.win
        pretend_audio_output(win)
        self.written = []
        win._audio.write = lambda a: self.written.append(np.asarray(a).copy())
        win._device._rng = np.random.default_rng(7)  # (its noise, repeatable)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")

    def frame(self):
        win = self.win
        self.written.clear()
        win._pacer.reset()
        win._process_block(
            win._device.read_samples(int(win._device_sample_rate() / 30))
        )
        return max(
            (float(np.max(np.abs(a))) for a in self.written if a.size), default=0.0
        )

    def tune_late(self, kind, freq_hz, lead):
        """Listen 1 s on a quiet channel, then tune ``freq_hz``, ``lead``
        seconds before a transmission of ``kind`` ends; (peak, lamp,
        counted by a scan) for each frame of the next 1.5 s."""
        win = self.win
        win._stop_acquisition()
        win._device.start_rx()
        rate = win._device_sample_rate()
        win._device._stream_pos = int((_on_at(kind, 8.0 - lead) - 1.0) * rate)
        rows = []
        with _demo_lag():
            self.assertTrue(win.set_frequency(freq_hz + 40e3))
            for _ in range(30):
                self.frame()
            self.assertTrue(win.set_frequency(freq_hz))
            for _ in range(45):
                peak = self.frame()
                lamp = win._squelch_state(True)
                rows.append((peak, lamp, win._scanner_host.scan_signal_present()))
        return rows

    def check(self, kind, freq_hz):
        for lead in LEADS:
            rows = self.tune_late(kind, freq_hz, lead)
            self.assertLess(max(p for p, _l, _c in rows), 1e-3, lead)
            self.assertNotIn("open", [lamp for _p, lamp, _c in rows], lead)
            self.assertNotIn(True, [counted for _p, _l, counted in rows], lead)
            self.assertIn("reverse", [lamp for _p, lamp, _c in rows], lead)

    def test_the_2m_repeater_reversed_on_its_tone(self):
        panel = self.win._control_panel
        panel.set_tone_squelch_hz(100.0)
        panel.set_tone_reverse(True)
        self.check("rpt", 145.29e6)

    def test_the_70cm_repeater_reversed_on_its_code(self):
        panel = self.win._control_panel
        panel.set_dcs_squelch(23)
        panel.set_tone_reverse(True)
        self.check("dcs", 442.1e6)

    def test_another_tone_still_plays(self):
        """Reversed on 123.0 Hz, the 100.0 Hz repeater plays from about
        half a second into its transmission, and its own reverse burst
        still closes it before its squelch tail."""
        panel = self.win._control_panel
        panel.set_tone_squelch_hz(123.0)
        panel.set_tone_reverse(True)
        rows = self.tune_late("rpt", 145.29e6, 1.0)
        heard = [i for i, (peak, _l, _c) in enumerate(rows) if peak > 0.05]
        self.assertTrue(heard)
        self.assertLessEqual(heard[0] / 30, 0.7)
        # The burst begins 0.85 s in, 0.15 s before the carrier drops (the
        # frame starting 1.0 s in): nothing is heard after that, its squelch
        # tail neither (it ends at 0.9 s; with Squelch Tail Elimination off
        # the tail plays to 1.33 s).
        self.assertLess(heard[-1], 30)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheMainWindowsLamp(_WindowTestCase):
    """The main window's LEVEL lamp says what each block's tone gate
    decided, as the Second Receiver's does. It was set before the gate ran,
    with the last block's verdict: a station the tone squelch mutes read
    OPEN for a frame as it keyed up, and the station a reverse tone squelch
    lets through read TONE as it began to play."""

    demo = True

    def setUp(self):
        super().setUp()
        win = self.win
        pretend_audio_output(win)
        self.written = []
        win._audio.write = lambda a: self.written.append(np.asarray(a).copy())
        win._device._rng = np.random.default_rng(7)  # (its noise, repeatable)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")

    def key_up(self, tone_hz, reverse):
        """(lamp, the gate's verdict, sound) for each frame from 0.2 s
        before a transmission of the demo's 2 m repeater (100.0 Hz) to 1 s
        into it."""
        win = self.win
        panel = win._control_panel
        panel.set_tone_squelch_hz(tone_hz)
        panel.set_tone_reverse(reverse)
        win._stop_acquisition()
        win._device.start_rx()
        self.assertTrue(win.set_frequency(145.29e6))
        rate = win._device_sample_rate()
        win._device._stream_pos = int((_on_at("rpt", 0.0) - 0.2) * rate)
        rows = []
        with _demo_lag():
            for _ in range(36):
                self.written.clear()
                win._pacer.reset()
                win._process_block(win._device.read_samples(int(rate / 30)))
                sound = any(np.max(np.abs(a)) > 1e-3 for a in self.written if a.size)
                rows.append((win._level_meter.state(), win._squelch_state(True), sound))
        return rows

    def test_a_station_it_mutes_never_reads_open(self):
        for tone_hz, reverse, muted in (
            (100.0, True, "reverse"),
            (123.0, False, "tone"),
        ):
            rows = self.key_up(tone_hz, reverse)
            lamps = [lamp for lamp, _gate, _sound in rows]
            self.assertIn(muted, lamps, tone_hz)
            self.assertNotIn("open", lamps, tone_hz)
            self.assertEqual(lamps, [gate for _lamp, gate, _sound in rows], tone_hz)
            self.assertNotIn(True, [sound for _lamp, _gate, sound in rows], tone_hz)

    def test_it_reads_open_as_the_station_begins_to_play(self):
        """Reversed on 123.0 Hz: held about half a second, then heard."""
        rows = self.key_up(123.0, True)
        lamps = [lamp for lamp, _gate, _sound in rows]
        heard = [sound for _lamp, _gate, sound in rows]
        self.assertIn(True, heard)
        self.assertIn("reverse", lamps)
        self.assertEqual(lamps, [gate for _lamp, gate, _sound in rows])
        self.assertEqual([lamp == "open" for lamp in lamps], heard)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestScanBookmarks(_WindowTestCase):
    """R7W-01: a bookmark scan that reaches a repeater in the last half
    second of a transmission, its only bookmark reversed on the tone or code
    the repeater sends: no hit, and it doesn't stop to listen to noise."""

    demo = True

    def scan(self, kind, freq_hz, lead, tone=0.0, code=None):
        from sdr_module.gui.scan_engine import LISTEN

        win, scanner = self.win, self.win._scanner_panel
        win._stop_acquisition()
        win._device.start_rx()
        win._device._rng = np.random.default_rng(11)
        rate = win._device_sample_rate()
        win._device._stream_pos = int(_on_at(kind, 8.0 - lead) * rate)
        bookmarks = win._bookmarks_panel
        for index in reversed(range(len(bookmarks.bookmarks()))):
            bookmarks._bookmarks.pop(index)
        index = bookmarks.add_bookmark("Reversed", freq_hz, "FM", "5 kHz")
        if code is not None:
            bookmarks.set_dcs(index, code, False, reverse=True)
        else:
            bookmarks.set_tone(index, tone, reverse=True)
        scanner.set_scan_mode("bookmarks")
        self.assertTrue(scanner.start_scan())
        listened = False
        with _demo_lag():
            for _ in range(int(2.5 * 30)):
                win._pacer.reset()
                win._process_block(win._device.read_samples(int(rate / 30)))
                listened = listened or scanner.engine.state == LISTEN
        hits = scanner.hits()
        scanner.stop_scan()
        return listened, hits

    def test_the_2m_repeater_reversed_on_its_tone(self):
        for lead in (0.35, 0.45):
            listened, hits = self.scan("rpt", 145.29e6, lead, tone=100.0)
            self.assertFalse(listened, lead)
            self.assertEqual(hits, [], lead)

    def test_the_70cm_repeater_reversed_on_its_code(self):
        for lead in (0.35, 0.45):
            listened, hits = self.scan("dcs", 442.1e6, lead, code=23)
            self.assertFalse(listened, lead)
            self.assertEqual(hits, [], lead)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheSecondReceiverWindow(_SecondCase):
    """The Second Receiver on the demo's 2 m repeater, reversed on its
    100.0 Hz, tuned ``lead`` seconds before a transmission ends: nothing
    played (R7W-03, the Second Receiver's side)."""

    def test_tuned_in_the_last_half_second_of_a_transmission(self):
        win = self.win
        main_sink = _FakeSink()
        main_sink.open = True
        win._audio = main_sink
        win._audio_enabled = True
        win._audio_problem = None
        speaker = self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.device._rng = np.random.default_rng(5)
        rx.tune(145.33e6, recenter=True)
        rx.set_mode("FM", "5 kHz", "25 kHz")
        rx._tone_combo.setCurrentIndex(rx._tone_combo.findData("100.0"))
        rx._tone_reverse_check.setChecked(True)
        rx.set_listen("second")
        rx._sync_audio_output()
        rate = rx.sample_rate()
        for lead in (0.3, 0.35, 0.4):
            rx.tune(145.33e6)
            rx.device._stream_pos = int((_on_at("rpt", 8.0 - lead) - 1.0) * rate)
            with _demo_lag():
                for _ in range(30):
                    rx._process(rx.device.read_samples(BLOCK))
                self.assertTrue(rx.tune(145.29e6))
                before = len(speaker.samples())
                lamps = []
                for _ in range(45):
                    rx._process(rx.device.read_samples(BLOCK))
                    lamps.append(rx._level_meter.state())
            heard = speaker.samples()[before:]
            peak = float(np.max(np.abs(heard))) if len(heard) else 0.0
            self.assertLess(peak, 1e-3, lead)
            self.assertNotIn("open", lamps, lead)


if __name__ == "__main__":
    unittest.main()
