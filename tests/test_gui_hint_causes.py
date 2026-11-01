"""K-W2: the status bar's skip and pause hints follow a change of cause.

The status bar says once per episode why a live decoder misses part of the
signal, and why audio is paused. Said while the demo device (or this
computer) couldn't keep up, it went on naming the device for up to 10 s
after Decode Recording started keeping the computer busy, and went on
saying "keeps this computer busy" after the job had ended. While the status
bar still shows the hint, it is now said again with the current cause:
once per change, once the new cause has held for a moment (a cause that
comes and goes is never said), and never the same text twice in a row
(no extra entries in the status history). A computer that is catching up
is not blamed on the device meanwhile.
"""

import os
import threading
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402


def _demo_take_case():
    if not HAS_PYQT6:  # pragma: no cover - no Qt
        return unittest.TestCase
    from tests.test_final_record import _DemoTakeCase

    return _DemoTakeCase


def _hardware_case():
    if not HAS_PYQT6:  # pragma: no cover - no Qt
        return unittest.TestCase
    from tests.test_review_mainloop import _HardwareCase

    return _HardwareCase


SKIP_DEVICE = (
    "The demo device can't keep up in real time right now, so the POCSAG "
    "decoder misses part of the signal. A lower sample rate (Receiver > "
    "Sample rate) helps."
)
SKIP_JOB = (
    "Decode Recording keeps this computer busy, so the POCSAG decoder misses "
    "part of the signal meanwhile."
)
SKIP_JOB_DONE = (
    "Decode Recording kept this computer busy, so the POCSAG decoder missed "
    "part of the signal meanwhile."
)
AUDIO_DEVICE = (
    "The demo device can't keep up in real time right now, so its audio is "
    "paused. It resumes by itself when the computer catches up (a lower "
    "sample rate in Receiver > Sample rate helps)."
)
AUDIO_JOB = "Decode Recording keeps this computer busy, so audio is paused meanwhile."
AUDIO_JOB_DONE = (
    "Decode Recording kept this computer busy, so audio is paused until it "
    "catches up."
)


def _runs(texts):
    """``texts`` with repeats in a row dropped."""
    runs = []
    for text in texts:
        if not runs or runs[-1] != text:
            runs.append(text)
    return runs


# --------------------------------------------------------------------------- #
# The rule on its own
# --------------------------------------------------------------------------- #


class TestHintFollower(unittest.TestCase):
    """status_history.HintFollower: when a hint the status bar shows is to
    be said again with its current wording."""

    def follower(self, settle=1.0):
        from sdr_module.gui.status_history import HintFollower

        return HintFollower(settle)

    def test_the_same_wording_is_not_said_again(self):
        follow = self.follower()
        for t in range(50):
            self.assertEqual(follow.follow("A", "A", "A", t * 0.1), "")

    def test_a_new_wording_is_said_once_it_has_held(self):
        follow = self.follower(1.0)
        said = [follow.follow("B", "A", "A", t * 0.1) for t in range(10)]
        self.assertEqual(said, [""] * 10)  # (not yet: 0.9 s)
        self.assertEqual(follow.follow("B", "A", "A", 1.0), "B")
        # Said: from then on "B" is what the status bar shows.
        self.assertEqual(follow.follow("B", "B", "B", 1.1), "")

    def test_a_cause_that_comes_and_goes_is_never_said(self):
        follow = self.follower(1.0)
        for t in range(100):
            wording = "B" if t % 4 < 2 else "A"  # (0.2 s each)
            self.assertEqual(follow.follow(wording, "A", "A", t * 0.1), "")
        for t in range(100, 120):
            wording = "B" if t % 4 < 2 else "C"
            self.assertEqual(follow.follow(wording, "A", "A", t * 0.1), "")

    def test_not_once_the_status_bar_says_something_else(self):
        follow = self.follower(1.0)
        follow.follow("B", "A", "A", 0.0)
        self.assertEqual(follow.follow("B", "A", "Tuned to 100.1 MHz", 2.0), "")
        # ...and it starts over if the hint shows again.
        self.assertEqual(follow.follow("B", "A", "A", 2.5), "")
        self.assertEqual(follow.follow("B", "A", "A", 3.5), "B")

    def test_nothing_said_nothing_followed(self):
        follow = self.follower(1.0)
        for t in range(30):
            self.assertEqual(follow.follow("B", "", "", t * 0.1), "")
            self.assertEqual(follow.follow("", "A", "A", t * 0.1), "")

    def test_reset_starts_over(self):
        follow = self.follower(1.0)
        follow.follow("B", "A", "A", 0.0)
        follow.reset()
        self.assertEqual(follow.follow("B", "A", "A", 1.5), "")
        self.assertEqual(follow.follow("B", "A", "A", 2.5), "B")


# --------------------------------------------------------------------------- #
# In the window: the demo device on a fake clock, ten frames a second
# --------------------------------------------------------------------------- #


class _HintCase(_demo_take_case()):
    """The demo device at 1 MS/s on a fake clock at ten frames a second,
    its processing ``load`` times real time (1.5: it can't keep up)."""

    FRAME_S = 0.1

    def setUp(self):
        require_pyqt6(self)
        super().setUp()
        self.said = []  # (what the status bar showed after each frame)

    def run_frames(self, seconds):
        end = self.clock.now + seconds
        while self.clock.now < end:
            tick = self.clock.now
            self.win._update_display()
            self.said.append(self.win._message_label.text())
            self.clock.now = max(self.clock.now, tick + self.FRAME_S)

    def a_busy_job(self, seconds):
        """Decode Recording's job (run_job's own event loop), with
        ``seconds`` of display frames meanwhile."""
        from PyQt6.QtCore import QTimer

        from sdr_module.gui.file_jobs import run_job

        done = threading.Event()

        def frames_meanwhile():
            try:
                self.run_frames(seconds)
            finally:
                done.set()

        QTimer.singleShot(0, frames_meanwhile)
        result = run_job(
            self.win, "Decode Recording", "Decoding…", lambda report: done.wait(20)
        )
        self.assertTrue(result.ok, result.error)

    def history(self, start):
        """The status history's entries from ``start`` on, oldest first,
        with how many times each was said in a row."""
        entries = list(reversed(self.win._status_history.entries()))
        return [(entry.text, entry.count) for entry in entries[start:]]

    def hints_said(self):
        """The bar's texts in order, the hints only."""
        hints = (SKIP_DEVICE, SKIP_JOB, SKIP_JOB_DONE)
        hints += (AUDIO_DEVICE, AUDIO_JOB, AUDIO_JOB_DONE)
        return [text for text in _runs(self.said) if text in hints]


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheDecodersHintFollowsItsCause(_HintCase):
    def setUp(self):
        super().setUp()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")
        self.run_frames(0.1)
        panel.set_receiving(True)
        self.said.clear()

    def test_a_job_that_starts_is_named(self):
        """The demo device was named; Decode Recording starts keeping the
        computer busy: the status bar names the job (it went on naming the
        device for up to 10 s)."""
        self.load = 1.5
        self.run_frames(1.5)
        self.assertEqual(self.win._message_label.text(), SKIP_DEVICE)
        start = len(self.win._status_history) - 1
        self.a_busy_job(2.5)
        self.assertEqual(self.win._message_label.text(), SKIP_JOB)
        self.assertEqual(self.hints_said(), [SKIP_DEVICE, SKIP_JOB])
        # Each said once: no repeats in the status history.
        self.assertEqual(self.history(start), [(SKIP_DEVICE, 1), (SKIP_JOB, 1)])

    def test_the_job_done_then_the_device_again(self):
        """Still too slow once the job is done: "kept ... busy" while what
        it made the loop skip still counts, then the device once more (the
        bar went on saying "keeps this computer busy")."""
        self.load = 1.5
        self.run_frames(1.0)
        start = len(self.win._status_history) - 1
        self.a_busy_job(2.0)
        self.run_frames(6.0)  # (still 1.5x real time)
        self.assertEqual(
            self.hints_said(), [SKIP_DEVICE, SKIP_JOB, SKIP_JOB_DONE, SKIP_DEVICE]
        )
        self.assertEqual(self.win._message_label.text(), SKIP_DEVICE)
        self.assertEqual(
            self.history(start),
            [(SKIP_DEVICE, 1), (SKIP_JOB, 1), (SKIP_JOB_DONE, 1), (SKIP_DEVICE, 1)],
        )

    def test_the_job_done_and_caught_up(self):
        """The computer keeps up once the job is done: "kept ... busy",
        then the hint goes (as before)."""
        self.load = 1.5
        self.run_frames(1.0)
        self.a_busy_job(2.0)
        self.load = 0.3
        self.run_frames(8.0)
        self.assertEqual(self.hints_said(), [SKIP_DEVICE, SKIP_JOB, SKIP_JOB_DONE])
        self.assertNotIn("missed", self.win._message_label.text())
        self.assertNotIn("misses", self.win._message_label.text())

    def test_not_said_over_another_message(self):
        """Once something else is in the status bar the hint is not brought
        back (the Decoder's line keeps saying why)."""
        win = self.win
        self.load = 1.5
        self.run_frames(1.5)
        win._show_status_message("Bookmarked POCSAG (152.240 MHz)", "success")
        self.a_busy_job(2.5)
        self.assertEqual(win._message_label.text(), "Bookmarked POCSAG (152.240 MHz)")
        self.assertEqual(win._decoder_panel.decoder_status_text(), SKIP_JOB)
        self.assertNotIn(SKIP_JOB, [text for text, _n in self.history(0)])

    def test_an_unchanged_cause_is_said_once(self):
        self.load = 1.5
        start = len(self.win._status_history)
        self.run_frames(8.0)
        self.assertEqual(self.history(start), [(SKIP_DEVICE, 1)])


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheAudioHintFollowsItsCause(_HintCase):
    """The speaker on (FM, the squelch open): audio pauses while the demo
    device can't keep up, and the status bar says why."""

    def setUp(self):
        super().setUp()
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._audio_enabled = True  # (no sound card here switched it off)
        win._control_panel.set_squelch_db(-120)
        self.run_frames(0.1)
        self.said.clear()

    def test_the_job_then_the_job_done_then_the_device(self):
        win = self.win
        self.load = 1.5
        self.run_frames(2.5)
        self.assertEqual(win._message_label.text(), AUDIO_DEVICE)
        start = len(win._status_history) - 1
        self.a_busy_job(2.5)
        self.assertEqual(win._message_label.text(), AUDIO_JOB)
        self.run_frames(6.0)  # (still 1.5x real time)
        self.assertEqual(
            self.hints_said(), [AUDIO_DEVICE, AUDIO_JOB, AUDIO_JOB_DONE, AUDIO_DEVICE]
        )
        self.assertEqual(
            self.history(start),
            [(AUDIO_DEVICE, 1), (AUDIO_JOB, 1), (AUDIO_JOB_DONE, 1), (AUDIO_DEVICE, 1)],
        )
        # (The "Audio resumed" logic follows what was said last.)
        self.assertEqual(win._realtime_hint_text, AUDIO_DEVICE)

    def test_audio_resumed_after_a_job(self):
        """Caught up after the job: "Audio resumed" (in place of the hint
        said last)."""
        win = self.win
        self.load = 1.5
        self.run_frames(2.5)
        self.a_busy_job(2.0)
        self.load = 0.3
        self.run_frames(8.0)
        self.assertEqual(self.hints_said(), [AUDIO_DEVICE, AUDIO_JOB, AUDIO_JOB_DONE])
        self.assertTrue(win._realtime())
        # "Audio resumed" took the place of the hint said last.
        runs = _runs(self.said)
        self.assertIn("Audio resumed", runs)
        self.assertEqual(runs[runs.index("Audio resumed") - 1], AUDIO_JOB_DONE)
        self.assertFalse(win._realtime_hint_shown)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestACatchingUpComputerIsNotBlamedOnTheDevice(_hardware_case()):
    """A HackRF-like device at 20 MS/s on a computer whose frames take 1.5x
    real time: audio pauses, "This computer can't process 20 MS/s ...".
    Once the computer keeps up again, the two seconds real time is judged
    over still hold the one-transfer frames: the audio's hint (the Audio
    button's tooltip, and the status bar following its cause) then said
    "The device isn't delivering samples in real time", blaming the device
    for what this computer dropped (the Decoder's line got this right)."""

    def test_the_computer_is_named_until_audio_resumes(self):
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._audio_enabled = True
        win._control_panel.set_squelch_db(-120)
        self.run_frames(60)
        self.assertTrue(win._pacer.overloaded)
        self.assertFalse(win._realtime())
        self.assertTrue(
            win._realtime_hint().startswith("This computer can't process 20 MS/s"),
            win._realtime_hint(),
        )
        self.LOAD = 0.3
        hints, bar = [], []
        for _ in range(300):
            self.run_frames(1)
            if not win._realtime():
                hints.append(win._realtime_hint())
            bar.append(win._message_label.text())
        self.assertFalse(win._pacer.overloaded)
        self.assertTrue(win._realtime())
        self.assertTrue(hints)
        blamed = [text for text in _runs(hints + bar) if "isn't delivering" in text]
        self.assertFalse(blamed, _runs(hints))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestANewRateIsJudgedAfresh(_hardware_case()):
    """Overloaded at 20 MS/s, the hints say a lower sample rate helps. At
    10 MS/s this computer keeps up, but the device delivers 40 % of it.
    The overload was at the old rate, before the seconds real time is
    judged over started again: the audio's hint (the status bar and the
    Audio button) said "This computer can't process 10 MS/s ..." for a
    second, then "The device isn't delivering ...", and the Decoder's line
    went on naming this computer once it judged the new rate."""

    def overloaded_then_10_ms(self):
        win = self.win
        self.run_frames(60)
        self.assertTrue(win._pacer.overloaded)
        self.LOAD = 0.3
        self.assertTrue(win.set_sample_rate(10e6))

    def test_the_audio_hint_names_the_device(self):
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._audio_enabled = True
        win._control_panel.set_squelch_db(-120)
        self.overloaded_then_10_ms()
        hints, bar = [], []
        for _ in range(150):
            self.run_frames(1, per_frame=1)  # (the device: 40 % of 10 MS/s)
            if not win._realtime():
                hints.append(win._realtime_hint())
            bar.append(win._message_label.text())
        self.assertFalse(win._realtime())
        self.assertTrue(hints)
        blamed = [text for text in _runs(hints + bar) if "This computer" in text]
        self.assertFalse(blamed, _runs(hints + bar))
        self.assertTrue(
            win._realtime_hint().startswith("The device isn't delivering"),
            win._realtime_hint(),
        )

    def test_the_decoders_line_names_the_device(self):
        win = self.win
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")
        self.run_frames(1)
        panel.set_receiving(True)
        self.overloaded_then_10_ms()
        judged = []  # (the line once the new rate is found short)
        for _ in range(150):
            self.run_frames(1, per_frame=1)
            if not win._realtime():
                judged.append(panel.decoder_status_text())
        self.assertTrue(judged)
        blamed = [text for text in _runs(judged) if text.startswith("This computer")]
        self.assertFalse(blamed, _runs(judged))
        self.assertTrue(
            judged[-1].startswith("The device isn't delivering 10 MS/s"), judged[-1]
        )


if __name__ == "__main__":
    unittest.main()
