"""
Play back an I/Q recording through the receiver.

File > Open Recording (Ctrl+O) makes a recording the receiver's source:
:class:`FilePlaybackDevice` has the demo device's API (``MockDevice``), so
the spectrum, waterfall, demodulator, squelch, S-meter, SSTV decoder, live
decoders and recorders work on the played samples exactly as on a live
signal. It reads the file from disk a chunk at a time (never the whole file
into memory), and the window paces it in real time like the demo device,
times the playback speed.

The LO is the file's center frequency, fixed while it plays (the window's
``_lock_lo``): tuning inside the recorded span moves only the tuned
frequency (the VFO), and frequencies outside it are refused with a status
message.

:class:`PlaybackBar` is the transport strip under the plots (Play/Pause,
position, time, Loop, speed, Close) and :class:`PlaybackController` switches
the window to a file and back to the device it was using.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from ..dsp.recording import SampleFormat
from .iq_files import IqFileOptions, format_mhz, format_rate

try:
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QSlider,
        QStyle,
        QStyleOptionSlider,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - the GUI needs PyQt6
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

#: Playback speeds offered (the strip's Speed list).
SPEEDS = (0.5, 1.0, 2.0, 4.0)
#: Seconds over which the speed a recording really plays at is measured.
_SPEED_CHECK_S = 3.0
#: The playback strip leaves out its caption where the position slider
#: would get less than _MIN_SLIDER px, and the file's name where that would
#: get less than _MIN_FILE_LABEL px.
_MIN_SLIDER = 200
_MIN_FILE_LABEL = 120


def format_clock(seconds: float, total: float = 0.0) -> str:
    """``75 -> "1:15"``; with hours (or a total of an hour or more)
    ``"1:02:05"``."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours or total >= 3600:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class FilePlaybackDevice:
    """A recording as a receiver: the demo device's (``MockDevice``) API
    over a file read from disk a chunk at a time.

    ``read_samples(n)`` gives the next ``n`` samples; at the end of the file
    it starts again with :attr:`loop`, else it stops there
    (:attr:`finished`). :attr:`paced` asks the window to pace the reads in
    real time times :attr:`speed`, as it does the demo device.

    The center frequency and sample rate are the file's: ``set_frequency``
    accepts only the center (the window tunes its VFO inside the span
    instead) and ``set_sample_rate`` only the file's rate; the gain is the
    one the recording was made with.
    """

    #: The window paces this device's reads in real time (see _read_block).
    paced = True

    class Info:
        def __init__(self, name: str):
            self.name = name
            self.serial = "FILE"

    def __init__(self, options: IqFileOptions):
        if not options.sample_rate > 0:
            raise ValueError("the recording's sample rate is unknown")
        self.options = options
        self._player = options.open_player()
        self._rate = float(options.sample_rate)
        self._center = float(options.center_hz)
        self._total = int(self._player.total_samples)
        self.info = self.Info(f"Playback: {options.name}")
        self._running = False
        self._finished = False
        #: Start again at the end of the file.
        self.loop = False
        #: Playback speed (the window reads this much faster than real time).
        self.speed = 1.0
        #: How often the file wrapped around (Loop).
        self.loops = 0
        #: Why reading stopped working (the file went away), if it did.
        self.error: Optional[str] = None
        #: Samples that weren't numbers were replaced by zeros.
        self.bad_samples = 0
        self._float_data = self._player.sample_format in (
            SampleFormat.FLOAT32,
            SampleFormat.FLOAT64,
        )

    # -- the MockDevice / driver API ------------------------------------------

    def set_frequency(self, freq: float) -> bool:
        """Only the file's center: a recording can't be retuned."""
        return abs(float(freq) - self._center) < 0.5

    def set_sample_rate(self, rate: float) -> bool:
        """Only the file's own rate."""
        return abs(float(rate) - self._rate) < 0.5

    def set_gain(self, gain: float) -> bool:
        return True  # recorded with its own gain

    def set_gain_mode(self, auto: bool) -> bool:
        return True

    def set_bandwidth(self, bw: float) -> bool:
        return True

    @property
    def frequency(self) -> float:
        return self._center

    @property
    def sample_rate(self) -> float:
        return self._rate

    @property
    def gain(self) -> float:
        return 0.0

    @property
    def gain_mode(self) -> str:
        return "manual"

    @property
    def is_streaming(self) -> bool:
        return self._running

    def start_rx(self) -> bool:
        """Play (from the start again once the end was reached)."""
        if self.error:
            return False
        if self._finished or self.position >= self._total:
            self.seek(0)
        self._running = True
        return True

    def stop_rx(self) -> bool:
        """Pause."""
        self._running = False
        return True

    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        if not self._running:
            return None
        n = int(num_samples)
        if n <= 0 or self._total <= 0:
            return np.zeros(0, dtype=np.complex64)
        try:
            pieces = [self._player.read(n)]
            got = len(pieces[0])
            while got < n and self.loop:
                self._player.seek(0)
                more = self._player.read(n - got)
                if not len(more):
                    break
                self.loops += 1
                pieces.append(more)
                got += len(more)
            out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        except (OSError, ValueError) as e:
            self.error = f"the recording can't be read any more ({e})"
            logger.warning(f"Playback of {self.options.path} stopped: {e}")
            self._running = False
            return None
        if not self.loop and self._player.position >= self._total:
            self._finished = True
        if self._float_data and len(out):
            finite = np.isfinite(out)
            if not finite.all():
                # NaN or infinity would poison the filters from here on.
                self.bad_samples += int(out.size - np.count_nonzero(finite))
                out = np.where(finite, out, 0).astype(np.complex64)
        return out

    def close(self) -> None:
        self._running = False
        try:
            self._player.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not close {self.options.path}: {e}")

    # -- playback -------------------------------------------------------------

    @property
    def finished(self) -> bool:
        """The end of the file was reached (and Loop is off)."""
        return self._finished

    @property
    def total_samples(self) -> int:
        return self._total

    @property
    def position(self) -> int:
        """The next sample to play."""
        return int(self._player.position)

    @property
    def duration_s(self) -> float:
        return self._total / self._rate

    @property
    def position_s(self) -> float:
        return self.position / self._rate

    def seek(self, sample: int) -> None:
        """Go to ``sample`` (clamped to the file)."""
        sample = min(max(0, int(sample)), self._total)
        self._player.seek(sample)
        self._finished = False

    def seek_seconds(self, seconds: float) -> None:
        self.seek(int(round(float(seconds) * self._rate)))


@dataclass
class _PreviousSource:
    """What the receiver used before a recording took over."""

    device: Any
    demo_mode: bool
    running: bool
    tuned_hz: float
    lo_hz: float
    name: str
    rate: float = 0.0  # the device's sample rate
    #: How it was heard: (mode, FM deviation, filter), as the window's
    #: _set_demod_mode takes them ("" for one that doesn't apply).
    listening: Tuple[str, str, str] = ("", "", "")


if HAS_PYQT6:

    class _FileLabel(QLabel):
        """The file's name and a suffix (its center frequency) on one line:
        when it doesn't fit, the name is shortened in the middle and the
        suffix kept. It never widens its row."""

        def __init__(self, parent: Any = None):
            super().__init__(parent)
            self._name = ""
            self._suffix = ""
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        def set_file(self, name: str, suffix: str) -> None:
            self._name, self._suffix = str(name or ""), str(suffix or "")
            self._elide()

        def full_text(self) -> str:
            return " · ".join(p for p in (self._name, self._suffix) if p)

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
            super().resizeEvent(event)
            self._elide()

        def _elide(self) -> None:
            width = self.contentsRect().width()
            metrics = self.fontMetrics()
            shown = self.full_text()
            if width > 0 and metrics.horizontalAdvance(shown) > width:
                tail = f" · {self._suffix}" if self._suffix else ""
                room = width - metrics.horizontalAdvance(tail)
                if room > metrics.horizontalAdvance("…") * 4:
                    shown = (
                        metrics.elidedText(
                            self._name, Qt.TextElideMode.ElideMiddle, room
                        )
                        + tail
                    )
                else:  # too narrow for both: the name (the rest is in the tooltip)
                    shown = metrics.elidedText(
                        self._name, Qt.TextElideMode.ElideMiddle, width
                    )
            super().setText(shown)

    class SeekSlider(QSlider):
        """A position slider that jumps to where it is clicked."""

        def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
            if event.button() == Qt.MouseButton.LeftButton:
                option = QStyleOptionSlider()
                self.initStyleOption(option)
                handle = self.style().subControlRect(
                    QStyle.ComplexControl.CC_Slider,
                    option,
                    QStyle.SubControl.SC_SliderHandle,
                    self,
                )
                if not handle.contains(event.position().toPoint()):
                    span = max(1, self.width() - handle.width())
                    x = int(event.position().x()) - handle.width() // 2
                    # (The value change seeks; the press goes on as a drag.)
                    self.setValue(
                        QStyle.sliderValueFromPosition(
                            self.minimum(), self.maximum(), x, span
                        )
                    )
            super().mousePressEvent(event)

    class PlaybackBar(QFrame):
        """The transport strip shown under the plots while a recording
        plays: Play/Pause, position, elapsed / total time, Loop, speed, the
        file, and Close (back to the device used before)."""

        play_clicked = pyqtSignal()
        seek_requested = pyqtSignal(float)  # seconds
        loop_changed = pyqtSignal(bool)
        speed_changed = pyqtSignal(float)
        close_clicked = pyqtSignal()
        #: Find Signals (T10): list the signals of the file playing.
        find_clicked = pyqtSignal()

        # Slider units: tenths of a second.
        _SCALE = 10.0

        def __init__(self, parent: Any = None):
            super().__init__(parent)
            set_role(self, "header-strip")
            self.setObjectName("playbackBar")
            self.setAccessibleName("Playback")
            self._duration = 0.0
            self._updating = False
            row = QHBoxLayout(self)
            row.setContentsMargins(6, 4, 6, 4)
            row.setSpacing(6)

            self._caption = QLabel("PLAYBACK")
            set_role(self._caption, "caption")
            row.addWidget(self._caption)

            self._play_button = QPushButton("▶ Play")
            set_role(self._play_button, "compact")
            self._play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._play_button.setAccessibleName("Play or pause")
            self._play_button.setMinimumWidth(
                self._play_button.fontMetrics().horizontalAdvance("❚❚ Pause") + 24
            )
            self._play_button.clicked.connect(lambda _c=False: self.play_clicked.emit())
            row.addWidget(self._play_button)

            self._elapsed = QLabel("0:00")
            self._elapsed.setAccessibleName("Elapsed time")
            self._elapsed.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(self._elapsed)

            self._slider = SeekSlider(Qt.Orientation.Horizontal)
            self._slider.setAccessibleName("Position")
            self._slider.setToolTip(
                "Where in the recording you are: drag or click to jump; the "
                "arrow keys skip 5 s, Page Up/Down 30 s."
            )
            self._slider.setSingleStep(50)
            self._slider.setPageStep(300)
            # Tab reaches it (arrows then skip); a click leaves the keyboard
            # focus on the plots, where Space plays and pauses.
            self._slider.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            self._slider.setMinimumWidth(100)
            self._slider.sliderMoved.connect(self._on_slider_moved)
            self._slider.sliderReleased.connect(self._on_slider_released)
            self._slider.valueChanged.connect(self._on_slider_value)
            row.addWidget(self._slider, 3)

            self._total = QLabel("0:00")
            self._total.setAccessibleName("Length")
            row.addWidget(self._total)

            self._loop = QPushButton("Loop")
            set_role(self._loop, "compact")
            self._loop.setCheckable(True)
            self._loop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._loop.setToolTip("Start again from the beginning at the end")
            self._loop.toggled.connect(self.loop_changed.emit)
            row.addWidget(self._loop)

            self._speed = QComboBox()
            self._speed.setAccessibleName("Playback speed")
            self._speed.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            for speed in SPEEDS:
                self._speed.addItem(f"{speed:g}x", speed)
            self._speed.setCurrentIndex(SPEEDS.index(1.0))
            self._speed.setToolTip(
                "Playback speed. Audio plays at 1x only; the spectrum, "
                "waterfall, decoders and recordings work at any speed."
            )
            self._speed.currentIndexChanged.connect(
                lambda _i: self.speed_changed.emit(float(self._speed.currentData()))
            )
            row.addWidget(self._speed)

            # The file and its center frequency (the name is shortened when
            # the strip is narrow; the rate is in the status bar).
            self._file = _FileLabel()
            set_role(self._file, "muted")
            self._file.setMinimumWidth(40)
            row.addWidget(self._file, 2)

            # (T10) The signals in the whole file, in the Signal tab.
            self._find = QPushButton("Signals…")
            set_role(self._find, "compact")
            self._find.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._find.setToolTip(
                "Find the signals in the whole recording: when each was heard, "
                "listed in the Signal tab (double-click one to play it)"
            )
            self._find.setAccessibleName("Find signals in the recording")
            self._find.clicked.connect(lambda _c=False: self.find_clicked.emit())
            row.addWidget(self._find)

            self._close = QPushButton("Close")
            set_role(self._close, "compact")
            self._close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._close.clicked.connect(lambda _c=False: self.close_clicked.emit())
            row.addWidget(self._close)

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
            super().resizeEvent(event)
            self._fit_row()

        def _fit_row(self) -> None:
            """Give the position slider room to seek with on a narrow strip
            (at 1024 px it got 100 px, and the file's name 40 px: "ap…ta").
            The PLAYBACK caption goes first, where the slider would get
            less than ``_MIN_SLIDER`` px; the file's name, where it would
            get less than ``_MIN_FILE_LABEL`` px (it shares the room with
            the slider 2:3). The status bar names the file, as Close's
            tooltip does."""
            row = self.layout()
            spacing = max(0, row.spacing())
            margins = row.contentsMargins()
            fixed = [
                w
                for w in (
                    self._play_button,
                    self._elapsed,
                    self._total,
                    self._loop,
                    self._speed,
                    self._find,
                    self._close,
                )
                if not w.isHidden()
            ]
            free = (
                self.width()
                - margins.left()
                - margins.right()
                - sum(w.sizeHint().width() + spacing for w in fixed)
            )
            caption = self._caption.sizeHint().width() + spacing
            show_caption = free - caption >= _MIN_SLIDER
            if show_caption:
                free -= caption
            show_file = 0.4 * (free - spacing) >= _MIN_FILE_LABEL
            for widget, show in (
                (self._caption, show_caption),
                (self._file, show_file),
            ):
                if show == widget.isHidden():
                    widget.setHidden(not show)

        # -- what is shown --------------------------------------------------

        def show_file(self, options: IqFileOptions, back_to: str) -> None:
            self._duration = max(0.0, options.duration_s)
            self._updating = True
            try:
                self._slider.setRange(0, int(round(self._duration * self._SCALE)))
                self._slider.setValue(0)
                self._loop.setChecked(False)
                self._speed.setCurrentIndex(SPEEDS.index(1.0))
            finally:
                self._updating = False
            self._total.setText(format_clock(self._duration, self._duration))
            self._elapsed.setText(format_clock(0, self._duration))
            self._file.set_file(options.name, format_mhz(options.center_hz))
            self._file.setToolTip(
                f"{options.path}\n{format_mhz(options.center_hz)}, "
                f"{format_rate(options.sample_rate)}"
            )
            self._close.setToolTip(
                f"Stop playing {options.name} and go back to {back_to}"
            )

        def set_playing(self, playing: bool, finished: bool = False) -> None:
            self._play_button.setText("❚❚ Pause" if playing else "▶ Play")
            if playing:
                tip = "Pause (Space)"
            elif finished:
                tip = "Play the recording again from the start (Space)"
            else:
                tip = "Play (Space)"
            self._play_button.setToolTip(tip)
            set_tone(self._elapsed, "muted" if finished else None)

        def set_position(self, seconds: float) -> None:
            """Show the playing position (not while the slider is dragged)."""
            if self._slider.isSliderDown():
                return
            self._updating = True
            try:
                self._slider.setValue(int(round(seconds * self._SCALE)))
            finally:
                self._updating = False
            self._elapsed.setText(format_clock(seconds, self._duration))

        def set_loop(self, on: bool) -> None:
            self._loop.setChecked(bool(on))

        def set_speed(self, speed: float) -> None:
            index = self._speed.findData(float(speed))
            if index >= 0:
                self._speed.setCurrentIndex(index)

        def loop(self) -> bool:
            return self._loop.isChecked()

        def speed(self) -> float:
            return float(self._speed.currentData())

        def elapsed_text(self) -> str:
            return self._elapsed.text()

        def file_text(self) -> str:
            return self._file.full_text()

        # -- the slider -------------------------------------------------------

        def _seconds(self, value: int) -> float:
            return value / self._SCALE

        def _on_slider_moved(self, value: int) -> None:
            self._elapsed.setText(format_clock(self._seconds(value), self._duration))

        def _on_slider_released(self) -> None:
            self.seek_requested.emit(self._seconds(self._slider.value()))

        def _on_slider_value(self, value: int) -> None:
            # Keys (arrows, Page Up/Down, Home/End) seek at once; a drag
            # seeks when released; the playing position isn't a seek.
            if self._updating or self._slider.isSliderDown():
                return
            self._elapsed.setText(format_clock(self._seconds(value), self._duration))
            self.seek_requested.emit(self._seconds(value))

    class PlaybackController(QObject):
        """Switches the window to a recording and back.

        :meth:`open` keeps the device the receiver used (stopped, not
        closed) and makes a :class:`FilePlaybackDevice` the window's device,
        with the LO fixed at the file's center; :meth:`close` goes back to
        the device, the frequency and the running state from before. While
        a file plays, Start/Stop (and Space) play and pause it.
        """

        _TICK_MS = 200

        def __init__(self, window: Any, bar: "PlaybackBar"):
            super().__init__(window)
            self._win = window
            self._bar = bar
            self._device: Optional[FilePlaybackDevice] = None
            self._previous: Optional[_PreviousSource] = None
            self._device_tip = ""
            #: The last open() changed the listening by the band plan (its
            #: status line is the one to keep).
            self.band_plan_applied = False
            # Where the file was (monotonic time, sample, loops) when the
            # speed it really plays at was last measured; and whether a
            # speed that can't be kept up was said.
            self._speed_mark: Optional[Tuple[float, int, int]] = None
            self._slow_said = False
            self._timer = QTimer(self)
            self._timer.setInterval(self._TICK_MS)
            self._timer.timeout.connect(self._tick)
            bar.setVisible(False)
            bar.play_clicked.connect(self._on_play_clicked)
            bar.seek_requested.connect(self.seek)
            bar.loop_changed.connect(self._on_loop_changed)
            bar.speed_changed.connect(self.set_speed)
            bar.close_clicked.connect(lambda: self.close(restore=True))

        # -- state --------------------------------------------------------------

        @property
        def active(self) -> bool:
            """A recording is the receiver's source."""
            return self._device is not None and self._win._device is self._device

        @property
        def device(self) -> Optional[FilePlaybackDevice]:
            return self._device if self.active else None

        @property
        def speed(self) -> float:
            return self._device.speed if self.active and self._device else 1.0

        def previous_name(self) -> str:
            prev = self._previous
            return prev.name if prev is not None and prev.device is not None else ""

        def previous_rate(self) -> float:
            """The sample rate of the device used before the recording (0
            if there was none)."""
            prev = self._previous
            return prev.rate if prev is not None and prev.device is not None else 0.0

        def back_to_text(self) -> str:
            name = self.previous_name()
            return name or "no device"

        # -- open / close ---------------------------------------------------------

        def open(self, options: IqFileOptions) -> bool:
            """Play ``options.path`` through the receiver. False (with the
            reason in the status bar) if it can't be opened."""
            win = self._win
            try:
                device = FilePlaybackDevice(options)
            except Exception as e:
                logger.warning(f"Could not open {options.path} for playback: {e}")
                win._show_status_message(
                    f"Can't play {options.name}: {e}", "danger", 8000
                )
                return False
            tuned_before = float(win._current_frequency())
            self.band_plan_applied = False
            self._speed_mark = None
            self._slow_said = False
            if self.active:
                # Another file: the device from before stays the one to go
                # back to.
                old = self._device
                win._stop_acquisition()
                if old is not None:
                    old.close()
            else:
                running = bool(win._is_running and win._device is not None)
                self._previous = _PreviousSource(
                    device=win._device,
                    demo_mode=bool(win._demo_mode),
                    running=running,
                    tuned_hz=float(win._current_frequency()),
                    lo_hz=float(win._lo_frequency()),
                    name=win._device_display_name() if win._device else "",
                    rate=float(win._device_sample_rate()) if win._device else 0.0,
                    listening=(
                        *win._listening_mode(),
                        win._control_panel._bw_combo.currentText(),
                    ),
                )
                if running:
                    win._stop_acquisition()
                self._set_gain_controls(False)
                self._device_tip = win._device_label.toolTip()
            self._device = device
            win._device = device
            win._demo_mode = False
            win._pacer.reset()
            win._rx_history.clear()
            win._realtime_hint_shown = False
            win._apply_tuning_range()
            win._lock_lo(options.center_hz, options.sample_rate, "the recording")
            # Heard the way its band usually is (a 2 m recording opened from
            # broadcast FM: narrow FM), as a frequency typed there would be;
            # the band plan's status line says so.
            self.band_plan_applied = bool(
                win._listen_by_band_plan(tuned_before, win._current_frequency())
            )
            win._reset_demodulator()
            self._bar.show_file(options, self.back_to_text())
            self._bar.setVisible(True)
            win._refresh_state_ui()
            win._start_acquisition()
            self._timer.start()
            self._tick()
            logger.info(f"Playing back {options.path}")
            return True

        def close(
            self, restore: bool = True, resume: bool = True, quiet: bool = False
        ) -> None:
            """Stop playing. With ``restore`` go back to the device used
            before (receiving again if it was, unless ``resume`` is False),
            else close that one too. Either way the tuning goes back to
            where it was before the recording."""
            if self._device is None:
                return
            win = self._win
            device = self._device
            name = device.options.name
            # What was being recorded came from the file: the recordings end
            # with it (an I/Q take stays in the Recording group to save).
            stopped = []
            if win._recording:
                win._toggle_recording(False)
                stopped.append("the I/Q recording")
            if win._audio_take is not None:
                win._stop_audio_recording(quiet=True)
                stopped.append("the audio recording")
            if win._is_running and win._device is device:
                win._stop_acquisition()
            self._device = None
            self._timer.stop()
            device.close()
            prev = self._previous
            self._previous = None
            if win._device is device:
                win._device = None
            win._demo_mode = False
            if prev is not None:
                if restore and prev.device is not None:
                    win._device = prev.device
                    win._demo_mode = prev.demo_mode
                elif prev.device is not None:
                    try:
                        prev.device.close()
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug(f"Device close failed: {e}")
            win._pacer.reset()
            win._rx_history.clear()
            win._realtime_hint_shown = False
            win._apply_tuning_range()
            if prev is not None:
                # Back where the receiver was before the recording, and
                # listening as it was there.
                if not win._retune_lo(prev.lo_hz):
                    win._lo_hz = float(prev.lo_hz)
                win._finish_tune(prev.tuned_hz)
                mode, deviation, width = prev.listening
                if mode:
                    win._set_demod_mode(mode, deviation or None, width or None)
            win._unlock_lo()  # the LO follows tuning again
            win._reset_demodulator()
            self._bar.setVisible(False)
            self._set_gain_controls(True)
            win._device_label.setToolTip(self._device_tip)
            win._refresh_state_ui()
            if resume and restore and prev is not None and prev.running and win._device:
                win._start_acquisition()
            if not quiet:
                back = f"Back to {prev.name}." if restore and prev and prev.name else ""
                ended = f" Stopped {' and '.join(stopped)} of it." if stopped else ""
                win._show_status_message(
                    f"Closed {name}.{ended} {back}".strip(),
                    "info" if stopped else None,
                    8000 if stopped else 4000,
                )
            logger.info(f"Playback of {name} closed")

        def shutdown(self) -> None:
            """The window closes: close the file and go back to the device
            and tuning from before (not receiving), so that is what is
            saved and closed with the window."""
            self.close(restore=True, resume=False, quiet=True)

        # -- transport ----------------------------------------------------------

        def toggle(self) -> None:
            """Play or pause (the strip's button, Start/Stop, Space)."""
            if self.active:
                self._win._toggle_acquisition()

        def _on_play_clicked(self) -> None:
            self.toggle()
            self._win._focus_plots()

        def seek(self, seconds: float) -> None:
            device = self.device
            if device is None:
                return
            device.seek_seconds(seconds)
            self._speed_mark = None
            # A jump in the stream: the demodulator starts afresh (no click).
            self._win._reset_demodulator()
            self._win._pacer.restart()
            self._bar.set_position(device.position_s)
            self._bar.set_playing(self._win._is_running, device.finished)

        def _on_loop_changed(self, on: bool) -> None:
            if self._device is not None:
                self._device.loop = bool(on)

        def set_speed(self, speed: float) -> None:
            device = self.device
            if device is None:
                return
            device.speed = float(speed)
            win = self._win
            # Whether frames keep up depends on the speed: find out anew.
            win._pacer.reset()
            win._rx_history.clear()
            win._realtime_hint_shown = False
            self._speed_mark = None
            self._slow_said = False
            if speed != 1.0:
                win._show_status_message(
                    f"Playing at {speed:g}x: audio plays at 1x only.", "info", 4000
                )
            else:  # (the note on another speed no longer applies)
                win._show_status_message("Playing at 1x.", "info", 2500)

        def _tick(self) -> None:
            """Follow the playing position; pause at the end of the file."""
            device = self.device
            if device is None:
                self._timer.stop()
                return
            win = self._win
            if device.error:
                win._stop_acquisition()
                # (Logged by the device: one Error History entry.)
                win._show_status_error(f"Playback stopped: {device.error}.", log=False)
                device.error = None
            elif device.finished and win._is_running:
                win._stop_acquisition()
                win._show_status_message(
                    f"End of {device.options.name}. Press Play (Space) to play "
                    "it again, or turn on Loop.",
                    "info",
                    8000,
                )
            self._bar.set_position(device.position_s)
            self._bar.set_playing(win._is_running, device.finished)
            self._check_speed(device)
            if device.bad_samples and not getattr(device, "_warned", False):
                device._warned = True  # type: ignore[attr-defined]
                win._show_status_error(
                    f"{device.options.name} has samples that aren't numbers "
                    "(NaN or infinity); they are played as silence. Is the "
                    "sample format right?"
                )

        def _check_speed(self, device: FilePlaybackDevice) -> None:
            """Faster than 1x, say once when the recording plays much slower
            than chosen: this computer can't read and show it that fast
            (frames that can't keep up take what they can), and the speed
            note said nothing of it."""
            now = time.monotonic()
            here = (now, device.position, device.loops)
            mark, self._speed_mark = self._speed_mark, here
            speed = float(device.speed)
            if (
                mark is None
                or speed <= 1.0
                or self._slow_said
                or not self._win._is_running
                or device.finished
                or mark[2] != device.loops  # (looped: the position wrapped)
                or here[1] < mark[1]
            ):
                return
            if now - mark[0] < _SPEED_CHECK_S:
                self._speed_mark = mark  # (measured over a few seconds)
                return
            real = (here[1] - mark[1]) / device.sample_rate / (now - mark[0])
            if real >= 0.6 * speed:
                return
            self._slow_said = True
            shown = f"{real:.1f}" if real >= 0.1 else "less than 0.1"
            self._win._show_status_message(
                f"Playing at about {shown}x, not {speed:g}x: this computer "
                "can't read and show the recording that fast right now. A "
                "lower speed helps.",
                "info",
                10000,
            )

        # -- the window's state shown as playback --------------------------------

        def sync_state_ui(self) -> None:
            """Show the state as playback (called by _refresh_state_ui)."""
            device = self.device
            if device is None:
                return
            win = self._win
            running = bool(win._is_running)
            badge = win._state_badge
            name = device.options.name
            if running:
                badge.setText("PLAYING")
                set_tone(badge, "success")
                badge.setToolTip(f"Playing {name}")
            elif device.finished:
                badge.setText("ENDED")
                set_tone(badge, "muted")
                badge.setToolTip(f"The end of {name}: Play (Space) starts it again")
            else:
                badge.setText("PAUSED")
                set_tone(badge, "warning")
                badge.setToolTip(f"{name} is paused: Play (Space) continues it")
            button = win._start_button
            button.setText("❚❚  Pause" if running else "▶  Play")
            button.setMinimumWidth(
                max(
                    button.minimumWidth(),
                    win._text_width(button, "❚❚  Pause", "▶  Play"),
                )
            )
            button.setToolTip(
                f"Pause {name} (Space)" if running else f"Play {name} (Space)"
            )
            win._start_action.setText(
                "&Pause Playback\tSpace" if running else "&Play Recording\tSpace"
            )
            win._device_label.setToolTip(
                f"{device.options.path}\nClose in the playback strip goes back "
                f"to {self.back_to_text()}."
            )
            self._bar.set_playing(running, device.finished)

        def info_state(self) -> Optional[str]:
            """The Info tab's State while a file plays (None: not playing)."""
            device = self.device
            if device is None:
                return None
            if self._win._is_running:
                return "Playing a recording"
            return "Playback ended" if device.finished else "Playback paused"

        def _set_gain_controls(self, enabled: bool) -> None:
            """The recording was made with its own gain: RF gain and AGC are
            off while it plays."""
            panel = self._win._control_panel
            widgets = [
                getattr(panel, "_gain_slider", None),
                getattr(panel, "_agc_check", None),
            ]
            for widget in widgets:
                if widget is None:
                    continue
                target = widget.parentWidget() if widget is widgets[0] else widget
                if target is None:
                    continue
                target.setEnabled(enabled)
                if enabled:
                    tip = getattr(target, "_tip_before_playback", None)
                    if tip is not None:
                        target.setToolTip(tip)
                else:
                    target._tip_before_playback = target.toolTip()
                    target.setToolTip(
                        "A recording plays with the gain it was made with."
                    )
