"""
Device > Second Receiver...: a second, lightweight receiver in its own window.

The dual-SDR uses of :mod:`sdr_module.core.dual_sdr`, with any two devices:

* **Dual watch**: a second device (an RTL-SDR, a HackRF One, or a second demo
  device to try it without hardware) next to the main window, with its own
  frequency (VFO), mode, filter, squelch, level meter, spectrum and
  waterfall. **Listen** decides whose audio plays: the main window's, this
  receiver's, or both mixed (each at its own volume).
* **TX monitor** and **full duplex**: while the main window's HackRF
  transmits (through the transmit engine, ``SDRMainWindow._transmit_job``),
  this receiver keeps receiving. **Monitor TX** tunes it to the transmit
  frequency, in the transmission's mode, in one click; **Follow
  transmissions** does that whenever a transmission starts.

The two windows never open the same device: the list here marks the one the
main window has as in use, and the main window's Device > Connect... refuses
the one this window holds (:meth:`SecondReceiverHost.device_problem`). A
stopped second receiver holds no device at all (Stop closes it).

It reuses the main window's bookmarks (tune from one, bookmark the frequency
heard here) and records its I/Q with the library's :class:`~sdr_module.dsp.
recording.IQRecorder` straight to a SigMF file in the recordings folder (the
main window's File > Open Recording plays it back), written by a thread of
its own as the main window's take is (:class:`~sdr_module.dsp.
background_writer.BackgroundWriter`: a block its queue has no room for is
counted as lost and said).

The signal path is the main window's: :class:`~.main_window._ReceiverChain`
(channel filter and demodulators), the squelch, level and noise-floor
helpers of :mod:`.audio_chain`, and the main window's display loop pacing
(:class:`~.main_window._Pacer`), in :class:`ReceiverDsp` (no Qt). Both
windows share the GUI thread, so each display frame's heavy part (the
signal path and the plots' spectrum, :class:`FrameWork`) runs on a worker
thread a frame behind the window (:class:`~.frame_pipeline.FramePipeline`):
the GUI thread reads, records, paints and plays.

Settings: ``device/second_receiver/*`` (device, sample rate per kind of
device, frequency, mode, filter, step per mode, squelch, Listen, volume,
gain, Follow transmissions) and ``layout/second_receiver/*`` (window
geometry, splitters and the waterfall's look).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import QByteArray, QObject, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QStyle,
        QStyleOptionComboBox,
        QStylePainter,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

from ..devices.base import SDRDevice
from ..dsp.background_writer import BackgroundWriter
from ..dsp.frame_spectrum import (
    WINDOW_LABELS,
    FrameSpectrumAnalyzer,
    normalize_fft_size,
    normalize_window,
)
from ..dsp.recording import FileFormat, IQRecorder, SampleFormat
from . import device_settings, file_dialogs
from .audio_chain import (
    AudioJoin,
    AudioOptions,
    LevelAverager,
    NoiseFloorTracker,
    SquelchGate,
    channel_power_db,
)
from .audio_sink import AudioSink
from .bookmarks_panel import default_bookmark_name, format_mhz
from .bookmarks_panel import listening as bookmark_listening
from .control_panel import (
    DEMOD_MODES,
    FM_DEVIATIONS,
    MAX_FREQUENCY_HZ,
    MIN_FREQUENCY_HZ,
    MODE_FILTER_DEFAULTS,
    SQUELCH_RANGE,
    FrequencyInput,
    SignedSpinBox,
    bandwidth_options,
    closest_bandwidth,
    filter_profile,
)
from .device_controls import format_ppm
from .device_settings import (
    AMP_TEXT,
    AMP_TIP,
    AMP_WARNING,
    DEFAULT_GAIN_MODEL,
    DIRECT_SAMPLING_CHOICES,
    DIRECT_SAMPLING_TIP,
    DIRECT_SAMPLING_VALUES,
    HACKRF_DEFAULT_LNA,
    HACKRF_DEFAULT_VGA,
    HACKRF_LNA_STEPS,
    HACKRF_VGA_STEPS,
    PPM_RANGE,
    PPM_TIP,
    GainModel,
    format_db,
    front_end_features,
    gain_model_for,
    gain_model_for_kind,
    load_front_end,
)
from .frame_pipeline import FramePipeline
from .iq_files import format_size
from .level_meter import LevelMeter

# The receiver chain and the display loop's helpers are the main window's,
# reused as they are (the same filters, demodulators, pacing and squelch
# timing as the main receiver).
from .main_window import (
    _AUDIBLE_MODES,
    _AUDIO_RATE,
    _CW_SQUELCH_HANG_S,
    _DEVICE_MODELS,
    _FREE_SPACE_CHECK_S,
    _MIN_FREE_BYTES,
    _PLOT_MAX_SAMPLES,
    _REALTIME_FRACTION,
    _REALTIME_WINDOW_S,
    _RETUNE_MAX_DROPPED,
    _RETUNE_SETTLE_S,
    _RETUNE_STALE_SAMPLES,
    _SQUELCH_HANG_S,
    _USABLE_SPAN_FRACTION,
    DISPLAY_BLOCK,
    SDRMainWindow,
    _capture_queue_bytes,
    _channel_edges,
    _clock_text,
    _ElidedLabel,
    _Pacer,
    _parse_hz,
    _ReceiverChain,
    _span_text,
    format_frequency,
    format_rate,
)
from .plot_view import PlotView
from .receiver_bar import FrequencyDigits
from .spectrum_widget import SpectrumWidget, format_hz
from .themes import set_role, set_tone
from .tuning import (
    STEP_CHOICES_HZ,
    default_step_hz,
    format_mhz_range,
    format_step,
    nearest_step,
    step_frequency,
    step_index,
    step_mode_key,
)
from .waterfall_widget import WaterfallWidget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings and constants
# ---------------------------------------------------------------------------

#: Settings of the second receiver (device, tuning, squelch, audio).
KEY = "device/second_receiver/"
#: Window geometry and splitter positions.
GEOMETRY_KEY = "layout/second_receiver/geometry"
SPLITTER_KEY = "layout/second_receiver/splitter"
PLOTS_KEY = "layout/second_receiver/plots"
#: The waterfall's colors, history and levels (JSON, as the widget gives it).
WATERFALL_KEY = "layout/second_receiver/waterfall"
#: The spectrum's display choices (Peak hold, averaging, range; JSON).
SPECTRUM_KEY = "layout/second_receiver/spectrum"
#: The spectrum settings kept: not its FFT size, window or zoom (the
#: frame analyzer's and the plot view's, set from the spectrum's menu).
_SPECTRUM_KEPT = (
    "avg",
    "peak_hold",
    "min_hold",
    "range_auto",
    "range_floor",
    "range_top",
    "bookmark_labels",
)

#: The sample rate a device starts at (every device offers it: RTL-SDR
#: 0.9-2.56 MS/s, HackRF 2-20 MS/s, the demo device 1-3.2 MS/s).
SAMPLE_RATE = 2.4e6
#: The sample rates offered for each kind of device: the main window's
#: (Device > Connect..., the Receiver group and the status bar's rate).
SAMPLE_RATES: Dict[str, Tuple[float, ...]] = {
    kind: device_settings.sample_rates_for(kind)
    for kind in ("demo", "rtlsdr", "hackrf")
}
#: Gain range of the RF gain slider (dB) with the demo device, and the
#: gain to start with (the demo device's reference gain). The slider steps
#: through the device's own gains (an RTL-SDR: its tuner's table); a HackRF
#: One has its LNA, VGA and RF amp instead, as in the main window.
GAIN_RANGE = (0, 50)
DEFAULT_GAIN_DB = 20

DEMO_KEY = "demo"
DEMO_NAME = "Demo Device (simulated signals)"  # (as the main window names it)

#: Whose audio plays: (key, item text, tooltip).
LISTEN_MAIN = "main"
LISTEN_SECOND = "second"
LISTEN_BOTH = "both"
LISTEN_CHOICES: Tuple[Tuple[str, str, str], ...] = (
    (
        LISTEN_MAIN,
        "Main window",
        "Hear the main window's receiver only; this one is silent",
    ),
    (
        LISTEN_SECOND,
        "This receiver",
        "Hear this receiver only: the main window's audio is muted while "
        "this window is open",
    ),
    (
        LISTEN_BOTH,
        "Both (mixed)",
        "Hear both receivers at once; balance them with this window's Volume "
        "and the main window's",
    ),
)
_LISTEN_KEYS = tuple(choice[0] for choice in LISTEN_CHOICES)
DEFAULT_LISTEN = LISTEN_BOTH

_DEFAULT_MODE = "FM"
_DEFAULT_FREQUENCY_HZ = 145.8e6  # (the main window starts on 100.1 MHz)

#: How a transmission is listened to: TxJob mode -> (Mode list name, FM
#: deviation value or "").
_TX_MODES = {
    "FM": ("FM", "5 kHz"),
    "AM": ("AM", ""),
    "CW": ("CW", ""),
    "USB": ("USB", ""),
    "SSB": ("USB", ""),
    "LSB": ("LSB", ""),
}
#: Data transmissions up to this wide are sent on the carrier (heard in CW).
_CW_DATA_HZ = 1e3
#: A transmission's mode in words, when its job doesn't say how it is sent.
_TX_MODE_WORDS = {"DATA": "data on the carrier", "SSB": "SSB"}
#: FM deviation for a transmission's occupied width: (up to width, value).
_FM_BY_WIDTH = (
    (10e3, "2.5 kHz"),  # the FM tone ID (2.5 kHz, 700 Hz tone: 6.4 kHz)
    (20e3, "5 kHz"),  # narrow voice
    (40e3, "12.5 kHz"),
    (100e3, "25 kHz"),
)

#: How long after this receiver's audio output failed to open it is tried
#: again (it is tried again at once when Listen or Start asks for it).
_AUDIO_RETRY_S = 5.0

_TX_WARNING = (
    "Keep this receiver's antenna well away from the transmitter, or put "
    "40-60 dB of attenuation in front of it: a strong signal close by "
    "overloads a receiver and can damage it."
)


def _number(value: Any) -> float:
    """``float(value)``, or 0 when it isn't a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _saved_choices(raw: str, plot: str) -> Optional[Dict[str, Any]]:
    """A plot's saved display choices (JSON, as the widget gives them), or
    None when ``raw`` can't be read as them: logged, as the main window's
    own are (a damaged settings file is listed in Tools > Error History),
    and the plot keeps the main window's defaults."""
    try:
        saved = json.loads(raw)
    except ValueError:
        saved = None
    if not isinstance(saved, dict):
        logger.warning(
            f"Ignoring the second receiver's unreadable {plot} settings: {raw!r}"
        )
        return None
    return saved


def _step_index(steps: Tuple[int, ...], value: Any, default: int) -> int:
    """Index of the step nearest ``value`` (``default``'s when it isn't a
    finite number)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return min(range(len(steps)), key=lambda i: abs(steps[i] - number))


def _device_call(device: Any, name: str, *args: Any) -> bool:
    """``device.name(*args)``; False if it refused, raised or has no such
    method (the driver logs why)."""
    method = getattr(device, name, None)
    if not callable(method):
        return False
    try:
        return method(*args) is not False
    except Exception as e:
        logger.warning(f"Second receiver: {name}{args} failed: {e}")
        return False


def _model_name(kind: str) -> str:
    return {"rtlsdr": "RTL-SDR", "hackrf": "HackRF One"}.get(kind, "Device")


def _file_and_folder(path: str) -> Tuple[str, str]:
    """A recording's file name and its folder, as messages say them: the
    name first, so that the window's status bar shows it at 860 px too
    (the folder was the start of a path that left no room for it)."""
    return os.path.basename(path), file_dialogs.display_path(os.path.dirname(path))


def device_kind(obj: Any) -> str:
    """``"rtlsdr"``, ``"hackrf"`` or ``"demo"`` for a device (or a driver
    class), by the driver class it is; ``""`` for anything else."""
    if obj is None:
        return ""
    cls = obj if isinstance(obj, type) else type(obj)
    for base in cls.__mro__:
        model = _DEVICE_MODELS.get(base.__name__)
        if model == "RTL-SDR":
            return "rtlsdr"
        if model == "HackRF One":
            return "hackrf"
    from .device_dialog import MockDevice

    if issubclass(cls, MockDevice):
        return "demo"
    return ""


def hardware_identity(device: Any) -> Optional[Tuple[str, int]]:
    """``(kind, index)`` of an open hardware device (None for the demo
    device, a recording or nothing): what two receivers must not share."""
    kind = device_kind(device)
    if kind not in ("rtlsdr", "hackrf"):
        return None
    info = getattr(device, "info", None)
    try:
        index = int(getattr(info, "index", 0) or 0)
    except (TypeError, ValueError):
        index = 0
    return kind, index


@dataclass
class Source:
    """A device the second receiver can use (one row of its Device list)."""

    key: str  # saved in the settings: "demo", "rtlsdr:<serial>", ...
    name: str  # "Demo device (simulated signals)", "RTL-SDR #1"
    kind: str  # "demo", "rtlsdr" or "hackrf"
    index: int = 0
    serial: str = ""
    device_class: Any = None
    #: Why it can't be used here ("" when it can): who has it.
    busy: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} (in use by {self.busy})" if self.busy else self.name


def main_window_devices(win: Any) -> List[Any]:
    """The devices the main window holds open: its own, and the one it
    returns to after a recording playing back."""
    devices = []
    device = getattr(win, "_device", None)
    if device is not None:
        devices.append(device)
    playback = getattr(win, "_playback", None)
    previous = getattr(playback, "_previous", None)
    held = getattr(previous, "device", None)
    if held is not None and held is not device:
        devices.append(held)
    return devices


def list_sources(win: Any) -> List[Source]:
    """The demo device, then each SDR plugged in (from the drivers that are
    installed), the main window's own marked as in use."""
    sources = [Source(DEMO_KEY, DEMO_NAME, "demo")]
    held = {
        identity
        for identity in map(hardware_identity, main_window_devices(win))
        if identity is not None
    }
    classes_of = getattr(win, "_hardware_driver_classes", None)
    try:
        classes = list(classes_of()) if callable(classes_of) else []
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Could not list the hardware drivers: {e}")
        classes = []
    seen: Dict[str, int] = {}
    for device_class in classes:
        kind = device_kind(device_class)
        if kind not in ("rtlsdr", "hackrf"):
            continue
        try:
            infos = device_class.list_devices() or []
        except Exception as e:
            logger.debug(f"{device_class.__name__} enumeration failed: {e}")
            continue
        for position, info in enumerate(infos):
            try:
                index = int(getattr(info, "index", position))
            except (TypeError, ValueError):
                index = position
            serial = str(getattr(info, "serial", "") or "")
            name = str(getattr(info, "name", "") or f"{_model_name(kind)} #{index}")
            busy = "the main window" if (kind, index) in held else ""
            # Cheap dongles often share one serial: keep them apart.
            key = f"{kind}:{serial or index}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                key = f"{key}#{seen[key]}"
            sources.append(
                Source(
                    key,
                    name,
                    kind,
                    index,
                    serial,
                    device_class,
                    busy,
                )
            )
    return sources


# ---------------------------------------------------------------------------
# Signal path (no Qt)
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """What one block of samples gave."""

    power_db: np.ndarray  # spectrum of the newest DISPLAY_BLOCK samples (dBFS)
    level_db: float  # the tuned channel's level (dBFS)
    noise_db: Optional[float]  # the channel's noise level (dBFS), once known
    threshold_db: float  # the squelch threshold in effect
    state: str  # the level meter's lamp (see level_meter.STATES)
    audio: Optional[np.ndarray]  # 48 kHz audio for the speaker, or None


class ReceiverDsp:
    """The second receiver's signal path: spectrum, channel filter, level,
    noise floor, squelch and demodulation, as the main window does them.

    Set :attr:`rate`, :attr:`mode`, :attr:`bandwidth_hz`,
    :attr:`deviation_hz` and :attr:`offset_hz` (the VFO minus the LO), then
    :meth:`process` each block. The channel filter and demodulator keep
    their state between blocks (rebuilt when the rate, mode or width
    changes), and the audio fades in and out with the squelch (and
    crossfades into a new chain), so it never clicks.
    """

    def __init__(self) -> None:
        self.rate = SAMPLE_RATE
        self.mode = _DEFAULT_MODE
        self.bandwidth_hz = 25e3
        self.deviation_hz = 5e3
        self.offset_hz = 0.0
        self.options = AudioOptions()
        self.squelch = SquelchGate(-80.0)
        self._level_average = LevelAverager()
        self._noise = NoiseFloorTracker()
        self._join = AudioJoin(_AUDIO_RATE)
        self._chain: Optional[_ReceiverChain] = None
        self._join_chain: Optional[_ReceiverChain] = None
        self._join_deviation = 0.0
        self._demod_running = False
        self._window: Optional[np.ndarray] = None
        self._window_gain = 1.0

    @property
    def audible(self) -> bool:
        return self.mode in _AUDIBLE_MODES

    @property
    def playing(self) -> bool:
        """Audio went to the speaker with the last block."""
        return self._join.playing

    def reset(self) -> None:
        """A new stream (started, another device): filters, squelch and
        level start afresh."""
        self._chain = None
        self._join_chain = None
        self._demod_running = False
        self.squelch.reset()
        self._level_average.reset()
        self._noise.reset()
        self._join.stop()

    def chain(self) -> _ReceiverChain:
        """The channel filter and demodulator for the current settings."""
        chain = self._chain
        wanted = _ReceiverChain.clamp_bandwidth(self.mode, self.bandwidth_hz, self.rate)
        if (
            chain is None
            or (chain.rate, chain.mode) != (self.rate, self.mode)
            or chain.bandwidth != wanted
        ):
            chain = self._chain = _ReceiverChain(
                self.rate, self.mode, self.bandwidth_hz, self.offset_hz
            )
            chain.options = self.options
        chain.set_offset(self.offset_hz)
        return chain

    def band_edges(self) -> Tuple[float, float]:
        """Offsets from the tuned frequency (Hz) of what is demodulated."""
        return _channel_edges(self.mode, self.bandwidth_hz, self.rate)

    def spectrum(self, samples: np.ndarray) -> np.ndarray:
        """Hann-windowed power spectrum in dBFS (as the main window's)."""
        n = len(samples)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if self._window is None or len(self._window) != n:
            self._window = np.hanning(n + 1)[:-1].astype(np.float64)
            self._window_gain = float(np.sum(self._window))
        spectrum = np.fft.fftshift(np.fft.fft(samples * self._window))
        magnitude = np.abs(spectrum) / max(self._window_gain, 1e-12)
        return (20.0 * np.log10(magnitude + 1e-12)).astype(np.float32)

    def noise_db(self, chain: _ReceiverChain) -> Optional[float]:
        return self._noise.channel_noise_db(chain.noise_bandwidth(), chain.rate)

    def process(self, samples: np.ndarray, play: bool) -> Frame:
        """One block: its spectrum, the channel's level and squelch, and
        (``play``: the speaker wants this receiver) the audio to play."""
        samples = np.asarray(samples, dtype=np.complex64)
        display = samples[-DISPLAY_BLOCK:] if len(samples) > DISPLAY_BLOCK else samples
        power = self.spectrum(display)
        self._noise.update(power, len(samples) / max(self.rate, 1.0))
        chain = self.chain()
        channel = chain.channelize(samples)
        seconds = len(channel) / max(chain.channel_rate, 1.0)
        if len(channel):
            level = channel_power_db(channel)
            gate_level = self._level_average.update(level, seconds)
        else:  # (an empty block: the strongest bin stands in)
            level = gate_level = float(np.max(power)) if len(power) else -150.0
        noise = self.noise_db(chain)
        gate = self.squelch
        gate.hang_s = _CW_SQUELCH_HANG_S if self.mode == "CW" else _SQUELCH_HANG_S
        audible = self.audible
        if audible:
            gate.update(gate_level, noise, seconds)
            state = ("open" if gate.is_open else "closed") if gate.enabled else "off"
        else:
            state = "idle"
        frame = Frame(power, level, noise, gate.threshold_db(noise), state, None)
        frame.audio = self._audio(chain, channel, play and audible and gate.is_open)
        return frame

    def _audio(
        self, chain: _ReceiverChain, channel: np.ndarray, speaker: bool
    ) -> Optional[np.ndarray]:
        join = self._join
        if not speaker and not join.playing:
            if self._demod_running:
                # Closed for real (not during the hang): start afresh.
                chain.reset_demodulator()
                self._demod_running = False
            return None
        if not self.audible or len(channel) == 0:
            # (A mode without audio, or nothing new: decay to silence.)
            self._demod_running = False
            return join.release()
        try:
            audio = chain.demodulate(channel, self.deviation_hz)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Second receiver demodulation failed: {e}")
            chain.reset_demodulator()
            self._demod_running = False
            return join.release()
        self._demod_running = True
        # A new chain or deviation: the audio doesn't continue the last
        # block played.
        new_stream = (
            chain is not self._join_chain or self.deviation_hz != self._join_deviation
        )
        if speaker:
            out = join.join(audio) if join.playing and new_stream else join.play(audio)
            self._join_chain, self._join_deviation = chain, self.deviation_hz
            return out
        # Stopping: fade this block out (or decay from the last sample).
        return join.release() if new_stream else join.fade_out(audio)

    def release(self) -> np.ndarray:
        """Audio is stopping (receiver stopped, Listen changed): a short
        decay from the last sample played (empty when nothing played)."""
        return self._join.release()


@dataclass
class FrameResult:
    """What :class:`FrameWork` made of one frame's samples."""

    frame: Frame  # the signal path's: level, squelch, audio
    spectrum: Any  # the plots' spectrum (a FrameSpectrum), or None
    axis: Tuple[float, float]  # the (LO, rate) the samples were taken at


class FrameWork:
    """One frame's heavy work, as the second receiver's DSP worker thread
    does it (no Qt): the signal path (:class:`ReceiverDsp`: channel filter,
    level, squelch, demodulation) and, when the plots want the frame, their
    whole-frame spectrum. While a frame runs, what the DSP and the analyzer
    keep (filters, squelch and level history, samples for the next FFT)
    and :attr:`axis` are the worker's: the window waits for the frame
    before it resets or reconfigures them (see :class:`~.frame_pipeline.
    FramePipeline`). The settings a user changes (mode, filter, tuning,
    squelch level) are plain values the window sets at any time: a frame
    running meanwhile takes them up where it reads them, so a change counts
    from that frame or the next."""

    def __init__(self, dsp: ReceiverDsp, analyzer: FrameSpectrumAnalyzer) -> None:
        self.dsp = dsp
        self.analyzer = analyzer
        #: The (LO, rate) the analyzer's samples kept from the last frame
        #: belong to (None: the next frame starts afresh).
        self.axis: Optional[Tuple[float, float]] = None

    def __call__(
        self, samples: np.ndarray, play: bool, plots: bool, axis: Tuple[float, float]
    ) -> FrameResult:
        frame = self.dsp.process(samples, play)
        spectrum = None
        if plots:
            if axis != self.axis:
                # Samples kept from the last frame belong to another span.
                self.analyzer.reset()
                self.axis = axis
            spectrum = self.analyzer.process(samples)
        else:
            self.axis = None  # (the first frame plotted again starts afresh)
        return FrameResult(frame, spectrum, axis)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def _show_note(label: Any, text: str, tone: Optional[str]) -> None:
    """Show ``text`` in the hint ``label``, colored by ``tone`` (None:
    plain). Only the tone changes, never the role: a role switched on a
    shown label doesn't get its padding. (Refreshed five times a second:
    nothing is touched unless it changed.)"""
    if label.text() != text:
        label.setText(text)
        label.updateGeometry()
    set_tone(label, tone)


if HAS_PYQT6:

    #: Item data role holding a shorter form of an item's text ("2.5 kHz"
    #: for "2.5 kHz (narrow, 12.5 kHz channels)"), shown by a
    #: :class:`_FittingComboBox` too narrow for the whole text.
    _SHORT_TEXT_ROLE = Qt.ItemDataRole.UserRole + 1

    class _FittingComboBox(QComboBox):
        """A combo box that never shows its text cut off mid-word: too
        narrow for the whole text (the controls column is), it shows the
        item's shorter form (:data:`_SHORT_TEXT_ROLE`), else the text
        ending in "…". The list shows every text whole."""

        def text_room(self) -> int:
            """The width (px) the shown text has in the box."""
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
            return self._room(option)

        def _room(self, option: "QStyleOptionComboBox") -> int:
            field = self.style().subControlRect(
                QStyle.ComplexControl.CC_ComboBox,
                option,
                QStyle.SubControl.SC_ComboBoxEditField,
                self,
            )
            width = field.width()
            if not option.currentIcon.isNull():
                width -= option.iconSize.width() + 4
            return max(0, width)

        def shown_text(self) -> str:
            """The text the box shows for the current item."""
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
            return self._fitted(option)

        def _fitted(self, option: "QStyleOptionComboBox") -> str:
            text = option.currentText
            room = self._room(option)
            metrics = self.fontMetrics()
            if metrics.horizontalAdvance(text) <= room:
                return text
            short = self.currentData(_SHORT_TEXT_ROLE)
            if isinstance(short, str) and short:
                if metrics.horizontalAdvance(short) <= room:
                    return short
            return metrics.elidedText(text, Qt.TextElideMode.ElideRight, room)

        def paintEvent(self, _event):  # noqa: N802 - Qt naming
            # As QComboBox paints itself, with the text fitted to the box.
            painter = QStylePainter(self)
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
            painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
            option.currentText = self._fitted(option)
            painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    class _SourceCombo(QComboBox):
        """The Device list; refreshed every time it opens (a device plugged
        in, or the main window's device changed, shows at once)."""

        def __init__(self, refresh: Any, parent: Any = None):
            super().__init__(parent)
            self._refresh = refresh

        def showPopup(self) -> None:  # noqa: N802 - Qt naming
            self._refresh()
            super().showPopup()

    class SecondReceiverWindow(QMainWindow):
        """The second receiver (see the module docstring)."""

        #: The Listen choice changed (its key).
        listen_changed = pyqtSignal(str)
        #: Started or stopped receiving.
        running_changed = pyqtSignal(bool)

        _START_TEXT = "▶  Start"
        _STOP_TEXT = "■  Stop"
        _RECORD_TEXT = "●  Record"
        _FRAME_MS = 33
        _STATUS_MS = 200

        def __init__(self, main_window: Any):
            # A window of its own (no Qt parent): a peer of the main window
            # that can go behind it and has its own taskbar entry. The
            # host closes it with the main window.
            super().__init__(None)
            self._win = main_window
            self._settings = main_window._settings
            self.setObjectName("secondReceiverWindow")
            self.setWindowTitle("Second Receiver")
            # The toolbar (Start, Record, FREQ, LEVEL) and the plots beside
            # the controls fit; a 1024x640 screen still holds it.
            self.setMinimumSize(860, 560)
            self._dsp = ReceiverDsp()
            self._device: Any = None
            self._source: Optional[Source] = None
            self._sources: List[Source] = []
            self._running = False
            self._pacer = _Pacer()
            self._rx_history: Deque[Tuple[float, int]] = deque()
            self._realtime_hint = False
            self._lo_hz = _DEFAULT_FREQUENCY_HZ
            self._tuned_hz = _DEFAULT_FREQUENCY_HZ
            self._step_hz = 12.5e3
            self._audio = AudioSink()
            self._audio_open = False
            self._audio_retry_at = 0.0  # (monotonic) after a failed open
            self._audio_failed = False
            self._listen = DEFAULT_LISTEN
            # I/Q recording: the recorder and its file, the LO it is
            # labelled with (tuning stays inside its span meanwhile).
            self._recorder: Optional[IQRecorder] = None
            # Its writer thread, and the samples it had no room for.
            self._record_writer: Optional[BackgroundWriter] = None
            self._record_lost = 0
            self._record_path = ""
            # What the last recording's end said (text, tone, ms).
            self._recording_said: Optional[Tuple[str, str, int]] = None
            self._record_lo: Optional[float] = None
            self._next_space_check = 0.0
            # Transmit monitor: tune here when the main window transmits.
            self._follow_tx = False
            # RF gain: how the device sets it, the RF gain slider's steps
            # and the gain chosen on it (dB; it carries over between devices
            # to the nearest step each has).
            self._gain_model: GainModel = DEFAULT_GAIN_MODEL
            self._slider_steps: Tuple[float, ...] = tuple(DEFAULT_GAIN_MODEL.steps)
            self._gain_db = float(DEFAULT_GAIN_DB)
            self._restoring = False
            self._shut_down = False
            self._main_gone = False  # (the main window was deleted)
            self._setup_ui()
            self._setup_actions()
            self._restore()
            self._frame_timer = QTimer(self)
            self._frame_timer.timeout.connect(self._frame)
            self._status_timer = QTimer(self)
            self._status_timer.timeout.connect(self._update_status)
            self._status_timer.start(self._STATUS_MS)
            self._connect_main_window()
            self._refresh_state()
            self._update_status()

        # ----------------------------------------------------------- building

        def _setup_ui(self) -> None:
            toolbar = QToolBar("Second receiver")
            toolbar.setObjectName("secondReceiverToolbar")
            toolbar.setMovable(False)
            toolbar.setFloatable(False)
            toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
            self.addToolBar(toolbar)
            self._toolbar = toolbar

            self._start_button = QPushButton(self._START_TEXT)
            self._start_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._start_button.setAccessibleName("Start or stop the second receiver")
            self._start_button.clicked.connect(self.toggle)
            self._start_button.setMinimumWidth(
                SDRMainWindow._text_width(
                    self._start_button, self._START_TEXT, self._STOP_TEXT
                )
            )
            toolbar.addWidget(self._start_button)

            self._record_button = QPushButton(self._RECORD_TEXT)
            self._record_button.setCheckable(True)
            self._record_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._record_button.setAccessibleName("Record I/Q of the second receiver")
            self._record_button.setMinimumWidth(
                SDRMainWindow._text_width(self._record_button, self._RECORD_TEXT)
            )
            self._record_button.clicked.connect(self._on_record_clicked)
            toolbar.addWidget(self._record_button)
            toolbar.addSeparator()

            caption = QLabel("FREQ")
            set_role(caption, "caption")
            toolbar.addWidget(caption)
            self._freq_label = QLabel(format_frequency(self._tuned_hz))
            set_role(self._freq_label, "lcd")
            self._freq_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._freq_label.setAccessibleName("Second receiver's tuned frequency")
            self._freq_label.setToolTip(
                "This receiver's tuned frequency. Scroll over a digit to tune "
                "that digit; type a frequency in Tuning > Frequency (Ctrl+L); "
                "click or scroll the spectrum and waterfall; Left/Right tune "
                "by the Step while they have the focus."
            )
            self._freq_digits = FrequencyDigits(self._freq_label)
            self._freq_digits.digit_tuned.connect(
                lambda delta: self.tune(self._tuned_hz + float(delta))
            )
            toolbar.addWidget(self._freq_label)
            toolbar.addSeparator()

            caption = QLabel("LEVEL")
            set_role(caption, "caption")
            toolbar.addWidget(caption)
            self._level_meter = LevelMeter()
            self._level_meter.setAccessibleName("Second receiver's level and squelch")
            self._level_meter.threshold_changed.connect(self._on_meter_threshold)
            toolbar.addWidget(self._level_meter)

            # Plots | controls, as in the main window.
            self._splitter = QSplitter(Qt.Orientation.Horizontal)
            self._splitter.setChildrenCollapsible(False)
            self._plots = QSplitter(Qt.Orientation.Vertical)
            self._plots.setChildrenCollapsible(False)
            self._spectrum = SpectrumWidget()
            # The main window's spectrum defaults: one trace (Peak hold is a
            # click away) and a range that follows the signals.
            self._spectrum.apply_display_settings(
                dict(getattr(self._win, "_SPECTRUM_DEFAULTS", {}))
            )
            self._waterfall = WaterfallWidget()
            # The main window's waterfall defaults: levels that follow the
            # noise floor and the strongest signal, a minute of history.
            self._waterfall.apply_display_settings(
                dict(getattr(self._win, "_WATERFALL_DEFAULTS", {"auto_levels": True}))
            )
            # As in the main window: both plots show the same part of the
            # span (zoom), and every sample of a frame (evenly spread
            # segments of one longer than _PLOT_MAX_SAMPLES) at the FFT size
            # and window chosen in the spectrum's menu (see FrameWork).
            self._plot_view = PlotView()
            self._frame_spectrum = FrameSpectrumAnalyzer(max_samples=_PLOT_MAX_SAMPLES)
            # Each frame's heavy work runs on a worker thread (see _feed).
            self._work = FrameWork(self._dsp, self._frame_spectrum)
            self._pipeline = FramePipeline("Second receiver DSP")
            # After a hardware retune the plots skip these samples and until
            # then (see _drop_stale_samples).
            self._plot_skip_samples = 0
            self._plot_skip_until = 0.0
            for plot in (self._spectrum, self._waterfall):
                plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                plot.set_view_model(self._plot_view)
                plot.frequency_clicked.connect(self.tune)
                plot.tune_steps.connect(self._tune_steps)
                plot.bookmark_requested.connect(self.bookmark_frequency)
                plot.message.connect(self._show_plot_message)
                plot.setMinimumHeight(140)
                self._plots.addWidget(plot)
            self._plots.setStretchFactor(0, 2)
            self._plots.setStretchFactor(1, 3)
            self._spectrum.resolution_changed.connect(self._apply_plot_resolution)
            self._spectrum.display_settings_changed.connect(
                self._save_spectrum_settings
            )
            self._waterfall.display_settings_changed.connect(
                self._save_waterfall_settings
            )
            self._splitter.addWidget(self._plots)

            controls = QWidget()
            column = QVBoxLayout(controls)
            column.setContentsMargins(6, 6, 6, 6)
            column.setSpacing(8)
            column.addWidget(self._build_device_group())
            column.addWidget(self._build_tuning_group())
            column.addWidget(self._build_audio_group())
            column.addWidget(self._build_bookmarks_group())
            column.addWidget(self._build_tx_group())
            column.addStretch(1)
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(controls)
            # Never narrower than the controls (no sideways scrolling).
            controls.ensurePolished()
            area.setMinimumWidth(
                controls.minimumSizeHint().width()
                + area.verticalScrollBar().sizeHint().width()
                + 4
            )
            self._controls_area = area
            self._splitter.addWidget(area)
            self._splitter.setStretchFactor(0, 1)
            self._splitter.setStretchFactor(1, 0)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.addWidget(self._splitter)
            self.setCentralWidget(central)
            self._setup_statusbar()

        @staticmethod
        def _form(group: "QGroupBox") -> "QFormLayout":
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            form.setLabelAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(8)
            return form

        @staticmethod
        def _combo(min_chars: int = 8) -> "_FittingComboBox":
            combo = _FittingComboBox()
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(min_chars)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return combo

        @staticmethod
        def _hint(text: str = "") -> "QLabel":
            label = QLabel(text)
            label.setWordWrap(True)
            set_role(label, "hint")
            return label

        @staticmethod
        def _value_label(text: str, widest: str) -> "QLabel":
            label = QLabel(text)
            set_role(label, "value")
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            label.ensurePolished()
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance(widest) + 4)
            return label

        def _build_device_group(self) -> "QGroupBox":
            group = QGroupBox("Device")
            form = self._form(group)
            self._source_combo = _SourceCombo(self._refresh_sources)
            self._source_combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._source_combo.setMinimumContentsLength(12)
            self._source_combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._source_combo.setAccessibleName("Second receiver's device")
            self._source_combo.setToolTip(
                "The device this receiver uses: any SDR the main window isn't "
                "using, or a second demo device (simulated signals) to try it "
                "without hardware. The list is refreshed every time it opens."
            )
            self._source_combo.activated.connect(self._on_source_activated)
            # The whole width of the group (titled Device): the demo device's
            # name, "Demo Device (simulated signals)", fits at the window's
            # minimum width (860 px) too, which a label beside it didn't
            # leave room for.
            form.addRow(self._source_combo)

            self._rate_combo = self._combo(8)
            self._rate_combo.setAccessibleName("Second receiver's sample rate")
            self._rate_combo.setToolTip(
                "How much spectrum this receiver sees at once (the span of its "
                "spectrum and waterfall). Higher rates cost more processing."
            )
            self._rate_combo.activated.connect(self._on_rate_activated)
            label = QLabel("Sample rate:")
            label.setBuddy(self._rate_combo)
            form.addRow(label, self._rate_combo)

            gain_row = QWidget()
            row = QHBoxLayout(gain_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            self._gain_slider = QSlider(Qt.Orientation.Horizontal)
            self._gain_slider.setRange(*GAIN_RANGE)
            self._gain_slider.setPageStep(5)
            self._gain_slider.setValue(DEFAULT_GAIN_DB)
            self._gain_slider.setAccessibleName("Second receiver's RF gain")
            self._gain_slider.setToolTip(
                "RF gain of this receiver's device. Turn it down when it "
                "monitors your own transmitter nearby."
            )
            self._gain_slider.valueChanged.connect(self._on_gain_changed)
            row.addWidget(self._gain_slider, 1)
            self._gain_label = self._value_label(f"{DEFAULT_GAIN_DB} dB", "49.6 dB")
            row.addWidget(self._gain_label)
            self._gain_auto = QCheckBox("Auto")
            self._gain_auto.setToolTip(
                "Let the device set its gain itself (on devices that can)"
            )
            self._gain_auto.toggled.connect(self._on_gain_auto)
            row.addWidget(self._gain_auto)
            label = QLabel("RF gain:")
            label.setBuddy(self._gain_slider)
            form.addRow(label, gain_row)

            # A HackRF One: its two gain stages and the RF amp instead of
            # the RF gain slider, as in the main window's Receiver group.
            self._lna_slider, self._lna_label = self._stage_row(
                form,
                "LNA gain:",
                "Second receiver's LNA gain",
                HACKRF_LNA_STEPS,
                HACKRF_DEFAULT_LNA,
                "The HackRF's RF (LNA) gain: 0-40 dB in 8 dB steps. Raise it "
                "first for weak signals, then the VGA gain; turn it down when "
                "this receiver monitors your own transmitter nearby.",
            )
            self._vga_slider, self._vga_label = self._stage_row(
                form,
                "VGA gain:",
                "Second receiver's VGA gain",
                HACKRF_VGA_STEPS,
                HACKRF_DEFAULT_VGA,
                "The HackRF's baseband (VGA) gain: 0-62 dB in 2 dB steps, after "
                "the LNA.",
            )
            self._amp_check = QCheckBox(AMP_TEXT)
            self._amp_check.setToolTip(AMP_TIP)
            self._amp_check.setAccessibleName("Second receiver's RF amp")
            self._amp_check.toggled.connect(self._on_amp_toggled)
            form.addRow("", self._amp_check)
            self._amp_warning = self._hint(AMP_WARNING)
            set_tone(self._amp_warning, "warning")
            form.addRow(self._amp_warning)

            # An RTL-SDR: how it receives HF, and its crystal's correction.
            # This receiver's own (its dongle is usually another unit than
            # the main window's).
            self._direct_combo = self._combo(8)
            for value, text, tip in DIRECT_SAMPLING_CHOICES:
                self._direct_combo.addItem(text, value)
                index = self._direct_combo.count() - 1
                self._direct_combo.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
                self._direct_combo.setItemData(
                    index, text.split(" (")[0], _SHORT_TEXT_ROLE
                )
            self._direct_combo.setAccessibleName("Second receiver's direct sampling")
            self._direct_combo.setToolTip(DIRECT_SAMPLING_TIP)
            self._direct_combo.activated.connect(self._on_direct_activated)
            label = QLabel("Direct sampling:")
            label.setBuddy(self._direct_combo)
            form.addRow(label, self._direct_combo)
            self._ppm_spin = SignedSpinBox()
            self._ppm_spin.setRange(*PPM_RANGE)
            self._ppm_spin.setSuffix(" ppm")
            self._ppm_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._ppm_spin.setKeyboardTracking(False)  # (applied on Enter)
            self._ppm_spin.setAccessibleName("Second receiver's frequency correction")
            self._ppm_spin.setToolTip(PPM_TIP)
            self._ppm_spin.valueChanged.connect(self._on_ppm_changed)
            label = QLabel("Correction:")
            label.setBuddy(self._ppm_spin)
            form.addRow(label, self._ppm_spin)

            self._device_note = self._hint()
            form.addRow(self._device_note)
            self._device_form = form
            return group

        def _stage_row(
            self,
            form: "QFormLayout",
            caption: str,
            name: str,
            steps: Tuple[int, ...],
            value: int,
            tip: str,
        ) -> Tuple["QSlider", "QLabel"]:
            """A HackRF gain stage's row: a slider over its steps (their
            index) and a dB readout."""
            widget = QWidget()
            row = QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, len(steps) - 1)
            slider.setPageStep(1 if len(steps) < 10 else 5)
            slider.setValue(steps.index(value))
            slider.setToolTip(tip)
            slider.setAccessibleName(name)
            slider.valueChanged.connect(self._on_stage_changed)
            row.addWidget(slider, 1)
            label = self._value_label(format_db(value), "49.6 dB")
            row.addWidget(label)
            text = QLabel(caption)
            text.setBuddy(slider)
            form.addRow(text, widget)
            return slider, label

        def _build_tuning_group(self) -> "QGroupBox":
            group = QGroupBox("Tuning")
            form = self._form(group)
            self._freq_input = FrequencyInput()
            self._freq_input.frequency_changed.connect(self._on_entry_frequency)
            self._freq_input.entry_rejected.connect(
                lambda reason: self._show_message(reason, "danger", 6000)
            )
            entry = getattr(self._freq_input, "_freq_input", None)
            if entry is not None:
                entry.setAccessibleName("Second receiver's frequency")
            label = QLabel("Frequency:")
            label.setBuddy(entry or self._freq_input)
            form.addRow(label, self._freq_input)

            self._mode_combo = self._combo(6)
            for text, _tip in DEMOD_MODES:
                self._mode_combo.addItem(text)
            self._mode_combo.setToolTip("How this receiver turns the signal into audio")
            self._mode_combo.setAccessibleName("Second receiver's mode")
            self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
            label = QLabel("Mode:")
            label.setBuddy(self._mode_combo)
            form.addRow(label, self._mode_combo)

            self._filter_combo = self._combo(6)
            self._filter_combo.setToolTip(
                "The channel this receiver demodulates (the filter width in "
                "USB, LSB and CW): what is heard, and what LEVEL and the "
                "squelch measure"
            )
            self._filter_combo.setAccessibleName("Second receiver's filter")
            self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
            label = QLabel("Filter:")
            label.setBuddy(self._filter_combo)
            form.addRow(label, self._filter_combo)

            self._deviation_combo = self._combo(6)
            for value, text, tip in FM_DEVIATIONS:
                self._deviation_combo.addItem(text, value)
                index = self._deviation_combo.count() - 1
                self._deviation_combo.setItemData(
                    index, tip, Qt.ItemDataRole.ToolTipRole
                )
                # (Shown when "2.5 kHz (narrow, 12.5 kHz channels)" is
                # too wide for the box.)
                self._deviation_combo.setItemData(index, value, _SHORT_TEXT_ROLE)
            self._deviation_combo.setToolTip("The FM deviation the station uses")
            self._deviation_combo.setAccessibleName("Second receiver's FM deviation")
            self._deviation_combo.currentIndexChanged.connect(
                lambda _i: self._on_deviation_changed()
            )
            label = QLabel("FM deviation:")
            label.setBuddy(self._deviation_combo)
            form.addRow(label, self._deviation_combo)

            self._step_combo = self._combo(6)
            for step in STEP_CHOICES_HZ:
                self._step_combo.addItem(format_step(step), step)
            self._step_combo.setToolTip(
                "Tuning step: what Left/Right and the mouse wheel over the "
                "plots tune this receiver by (Shift: ten steps)"
            )
            self._step_combo.setAccessibleName("Second receiver's tuning step")
            self._step_combo.currentIndexChanged.connect(self._on_step_index)
            label = QLabel("Step:")
            label.setBuddy(self._step_combo)
            form.addRow(label, self._step_combo)
            self._tuning_form = form
            return group

        def _build_audio_group(self) -> "QGroupBox":
            group = QGroupBox("Squelch and audio")
            form = self._form(group)
            self._squelch_check = QCheckBox("Squelch:")
            self._squelch_check.setChecked(True)
            self._squelch_check.setToolTip(
                "Squelch on: this receiver's audio is muted while its channel "
                "is quieter than the threshold. Off: everything plays, noise "
                "too."
            )
            self._squelch_check.toggled.connect(self._on_squelch_enabled)
            squelch_row = QWidget()
            row = QHBoxLayout(squelch_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            self._squelch_auto = QCheckBox("Auto")
            self._squelch_auto.setChecked(True)
            self._squelch_auto.setToolTip(
                "Sets the threshold a little (7 dB) above the channel's noise "
                "and keeps it there. Moving the slider (or dragging the "
                "threshold on LEVEL) sets it by hand instead."
            )
            self._squelch_auto.toggled.connect(self._on_squelch_auto)
            row.addWidget(self._squelch_auto)
            row.addStretch(1)
            form.addRow(self._squelch_check, squelch_row)

            threshold_row = QWidget()
            row = QHBoxLayout(threshold_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
            self._squelch_slider.setRange(*SQUELCH_RANGE)
            self._squelch_slider.setValue(-80)
            self._squelch_slider.setPageStep(5)
            self._squelch_slider.setAccessibleName(
                "Second receiver's squelch threshold"
            )
            self._squelch_slider.setToolTip(
                "Mutes this receiver while its channel is below this level "
                "(dBFS); LEVEL shows the level and the threshold"
            )
            self._squelch_slider.valueChanged.connect(self._on_squelch_slider)
            row.addWidget(self._squelch_slider, 1)
            self._squelch_label = self._value_label("-80 dBFS", "-120 dBFS")
            row.addWidget(self._squelch_label)
            label = QLabel("Threshold:")
            label.setBuddy(self._squelch_slider)
            form.addRow(label, threshold_row)

            self._listen_combo = self._combo(8)
            for key, text, tip in LISTEN_CHOICES:
                self._listen_combo.addItem(text, key)
                self._listen_combo.setItemData(
                    self._listen_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
                )
            self._listen_combo.setAccessibleName("Listen to")
            self._listen_combo.currentIndexChanged.connect(self._on_listen_index)
            label = QLabel("Listen:")
            label.setBuddy(self._listen_combo)
            form.addRow(label, self._listen_combo)

            volume_row = QWidget()
            row = QHBoxLayout(volume_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            self._volume_slider = QSlider(Qt.Orientation.Horizontal)
            self._volume_slider.setRange(0, 100)
            self._volume_slider.setPageStep(10)
            self._volume_slider.setValue(70)
            self._volume_slider.setAccessibleName("Second receiver's volume")
            self._volume_slider.setToolTip(
                "Volume of this receiver (the main window's toolbar sets its own)"
            )
            self._volume_slider.valueChanged.connect(self._on_volume_changed)
            row.addWidget(self._volume_slider, 1)
            self._volume_label = self._value_label("70%", "100%")
            row.addWidget(self._volume_label)
            label = QLabel("Volume:")
            label.setBuddy(self._volume_slider)
            form.addRow(label, volume_row)
            self._audio_note = self._hint()
            form.addRow(self._audio_note)
            return group

        def _build_bookmarks_group(self) -> "QGroupBox":
            group = QGroupBox("Bookmarks")
            layout = QVBoxLayout(group)
            layout.setSpacing(6)
            self._bookmark_combo = self._combo(10)
            self._bookmark_combo.setAccessibleName("Tune to a bookmark")
            self._bookmark_combo.setToolTip(
                "The main window's bookmarks: pick one to tune this receiver "
                "to it, in its mode and filter"
            )
            self._bookmark_combo.activated.connect(self._on_bookmark_activated)
            layout.addWidget(self._bookmark_combo)
            self._bookmark_button = QPushButton("Bookmark This Frequency")
            self._bookmark_button.setToolTip(
                "Add this receiver's frequency, in its mode and filter, to the "
                "main window's Bookmarks (Ctrl+B)"
            )
            self._bookmark_button.setAutoDefault(False)
            self._bookmark_button.clicked.connect(
                lambda _checked=False: self.bookmark_frequency(self._tuned_hz)
            )
            layout.addWidget(self._bookmark_button)
            return group

        def _build_tx_group(self) -> "QGroupBox":
            group = QGroupBox("Transmit monitor")
            group.setToolTip(
                "TX monitor and full duplex: this receiver keeps receiving "
                "while the main window's HackRF transmits"
            )
            layout = QVBoxLayout(group)
            layout.setSpacing(6)
            self._tx_label = self._hint()
            self._tx_label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(self._tx_label)
            row = QHBoxLayout()
            row.setSpacing(8)
            self._monitor_button = QPushButton("Monitor TX")
            self._monitor_button.setAutoDefault(False)
            self._monitor_button.clicked.connect(
                lambda _checked=False: self.monitor_tx()
            )
            row.addWidget(self._monitor_button)
            self._follow_check = QCheckBox("Follow transmissions")
            self._follow_check.setToolTip(
                "Whenever the main window starts transmitting, tune this "
                "receiver to the transmit frequency (and start it)"
            )
            self._follow_check.toggled.connect(self._on_follow_toggled)
            row.addWidget(self._follow_check)
            row.addStretch(1)
            layout.addLayout(row)
            self._tx_warning = QLabel(_TX_WARNING)
            self._tx_warning.setWordWrap(True)
            set_role(self._tx_warning, "callout", "warning")
            layout.addWidget(self._tx_warning)
            return group

        def _setup_statusbar(self) -> None:
            bar = QStatusBar()
            bar.setContentsMargins(8, 0, 8, 0)
            bar.setSizeGripEnabled(False)
            self.setStatusBar(bar)
            self._state_badge = QLabel("STOPPED")
            set_role(self._state_badge, "badge", "muted")
            self._state_badge.setAccessibleName("Second receiver state")
            bar.addWidget(self._state_badge)
            self._device_label = QLabel("")
            set_role(self._device_label, "value")
            bar.addWidget(self._device_label)
            self._rate_label = QLabel("")
            bar.addWidget(self._rate_label)
            self._message_label = _ElidedLabel("")
            bar.addWidget(self._message_label, 1)
            self._message_timer = QTimer(self)
            self._message_timer.setSingleShot(True)
            self._message_timer.timeout.connect(self._clear_message)
            self._rec_label = QLabel("REC")
            set_role(self._rec_label, "badge", "danger")
            self._rec_label.setAccessibleName("Second receiver recording time")
            self._rec_label.setVisible(False)
            bar.addPermanentWidget(self._rec_label)

        def _setup_actions(self) -> None:
            """Window-local commands (they act while this window is active)."""

            def add(text: str, slot: Any, keys: str, tip: str) -> "QAction":
                action = QAction(text, self)
                action.setShortcut(QKeySequence(keys))
                action.setToolTip(tip)
                action.setStatusTip(tip)
                action.triggered.connect(lambda _checked=False: slot())
                self.addAction(action)
                return action

            self._bookmark_action = add(
                "Bookmark This Frequency",
                lambda: self.bookmark_frequency(self._tuned_hz),
                "Ctrl+B",
                "Add this receiver's frequency to the main window's Bookmarks",
            )
            self._entry_action = add(
                "Enter Frequency",
                self._focus_entry,
                "Ctrl+L",
                "Type a frequency for this receiver",
            )
            self._record_action = add(
                "Record I/Q",
                lambda: self._set_recording(self._recorder is None),
                "Ctrl+Shift+R",
                "Record this receiver's I/Q samples",
            )
            self._close_action = add(
                "Close", self.close, "Ctrl+W", "Close the second receiver"
            )

        # ------------------------------------------------------- main window

        def _connect_main_window(self) -> None:
            station = getattr(self._win, "_tx_station", None)
            engine = getattr(station, "engine", None)
            if engine is not None:
                # (Bound methods: disconnected when this window is deleted.)
                engine.started.connect(self._on_tx_started)
                engine.finished.connect(self._on_tx_finished)
            panel = getattr(self._win, "_bookmarks_panel", None)
            if panel is not None:
                panel.bookmarks_changed.connect(self._refresh_bookmarks)
            self._refresh_bookmarks()
            # Never outlive the main window running (the host closes this
            # window first; this covers a main window deleted without it).
            destroyed = getattr(self._win, "destroyed", None)
            if destroyed is not None:
                destroyed.connect(self._on_main_window_destroyed)

        def _on_main_window_destroyed(self, *_args: Any) -> None:
            """The main window is gone: stop, finishing a recording and
            closing the device, without touching anything of the main
            window's."""
            self._main_gone = True
            self._frame_timer.stop()
            self._status_timer.stop()
            self._pipeline.close()  # (the frame in flight is dropped)
            self._running = False
            if self._recorder is not None:
                writer, self._record_writer = self._record_writer, None
                try:
                    if writer is not None:
                        writer.close()  # (what it holds is written first)
                    self._recorder.stop()
                except Exception as e:  # pragma: no cover - disk trouble
                    logger.warning(f"Could not finish {self._record_path}: {e}")
                self._recorder = None
            self._close_device()
            self._audio.stop()
            self._audio_open = False
            self._shut_down = True

        def _main_audio_on(self) -> bool:
            """The main window's Audio switch (the app's master mute)."""
            return bool(getattr(self._win, "_audio_enabled", True))

        def _audio_problem(self) -> str:
            """Why no audio can play at all ("" when it can): no
            QtMultimedia, or no output device (as the main window found)."""
            if not self._audio.available:
                return "Audio needs the PyQt6 QtMultimedia module"
            problem = getattr(self._win, "_audio_problem", None)
            return str(problem) if problem and not self._main_audio_on() else ""

        # ------------------------------------------------------------ state

        @property
        def running(self) -> bool:
            return self._running

        @property
        def device(self) -> Any:
            return self._device

        @property
        def source(self) -> Optional[Source]:
            return self._source

        def listen(self) -> str:
            return self._listen

        def tuned_frequency(self) -> float:
            return self._tuned_hz

        def lo_frequency(self) -> float:
            return self._lo_hz

        def mode(self) -> str:
            return self._mode_combo.currentText()

        def _rate(self) -> float:
            dev = self._device
            for value in (
                getattr(getattr(dev, "state", None), "sample_rate", None),
                getattr(dev, "sample_rate", None),
            ):
                try:
                    rate = float(value)
                except (TypeError, ValueError):
                    continue
                if rate > 0:
                    return rate
            return self._chosen_rate()

        def _chosen_rate(self) -> float:
            """The Sample rate choice (what the device is opened at)."""
            rate = (
                self._rate_combo.currentData() if hasattr(self, "_rate_combo") else None
            )
            try:
                value = float(rate)
            except (TypeError, ValueError):
                return SAMPLE_RATE
            return value if value > 0 else SAMPLE_RATE

        def _bandwidth_hz(self) -> float:
            return _parse_hz(self._filter_combo.currentText()) or 25e3

        def _deviation_value(self) -> str:
            value = self._deviation_combo.currentData()
            return str(value) if value else "5 kHz"

        def _deviation_hz(self) -> float:
            return _parse_hz(self._deviation_value()) or 5e3

        def _device_name(self) -> str:
            if self._source is not None:
                return self._source.name
            return "No device"

        def holds(self, kind: str, index: int) -> bool:
            """Whether this receiver has hardware device ``(kind, index)``
            open."""
            return hardware_identity(self._device) == (kind, int(index))

        # ---------------------------------------------------------- devices

        def _refresh_sources(self) -> None:
            """Rebuild the Device list (keeping the choice)."""
            current = self._selected_key()
            self._sources = list_sources(self._win)
            combo = self._source_combo
            combo.blockSignals(True)
            combo.clear()
            for source in self._sources:
                combo.addItem(source.label, source.key)
                index = combo.count() - 1
                if source.busy:
                    item = combo.model().item(index)
                    if item is not None:
                        item.setEnabled(False)
                    combo.setItemData(
                        index,
                        f"{source.name} is in use by {source.busy}: the two "
                        "receivers can't share a device",
                        Qt.ItemDataRole.ToolTipRole,
                    )
            # The device in use stays chosen; otherwise the saved (or last)
            # choice when it is free, else the demo device.
            wanted = self._source.key if self._source is not None else current
            index = self._source_index(wanted)
            if index < 0 or self._sources[index].busy:
                index = 0
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
            self._fill_rates()
            self._sync_device_controls()
            self._update_device_note()

        def _rate_kind(self) -> str:
            """The kind of device the Sample rate list is for: the one open,
            else the one chosen."""
            if self._source is not None:
                return self._source.kind
            source = self._selected_source()
            return source.kind if source is not None else "demo"

        def _fill_rates(self) -> None:
            """The rates the device offers, the one last used with that kind
            of device chosen (remembered per kind)."""
            kind = self._rate_kind()
            rates = SAMPLE_RATES.get(kind, SAMPLE_RATES["demo"])
            wanted = (
                self._rate()
                if self._device is not None
                else self._settings.get_float(f"{KEY}sample_rate/{kind}", SAMPLE_RATE)
            )
            if not (math.isfinite(wanted) and wanted > 0):
                wanted = SAMPLE_RATE
            combo = self._rate_combo
            combo.blockSignals(True)
            combo.clear()
            for rate in rates:
                combo.addItem(format_rate(rate), rate)
            best = min(range(len(rates)), key=lambda i: abs(rates[i] - wanted))
            combo.setCurrentIndex(best)
            combo.blockSignals(False)

        def _on_rate_activated(self, _index: int) -> None:
            rate = self._chosen_rate()
            kind = self._rate_kind()
            if self._recorder is not None:
                self._fill_rates()  # (back to the rate in use)
                self._show_message(
                    "Stop recording to change the sample rate: one file has one "
                    "rate.",
                    "warning",
                )
                return
            if self._device is None or abs(rate - self._rate()) < 1.0:
                self._save(f"{KEY}sample_rate/{kind}", rate)
                self._refresh_state()
                return
            self._settle_frames()  # (the frame in flight is the old span's)
            dev = self._device
            # Hardware takes the new rate between two streams, and what it
            # queued at the old rate is dropped (it would be taken for the
            # new rate: a wrong waterfall line, a blip of audio at the wrong
            # pitch), as in the main window.
            pause = isinstance(dev, SDRDevice) and self._running
            if pause:
                try:
                    dev.stop_rx()
                except Exception as e:  # pragma: no cover - driver-dependent
                    logger.debug(f"Second receiver: stop_rx for the rate: {e}")
            try:
                ok = dev.set_sample_rate(rate)
            except Exception as e:
                logger.warning(f"Second receiver: sample rate {rate}: {e}")
                ok = False
            self._apply_baseband_filter()  # (for the rate it runs at)
            if pause:
                dev.discard_queued_samples()
                try:
                    started = dev.start_rx() is not False
                except Exception as e:
                    logger.error(f"Second receiver: could not restart receiving: {e}")
                    started = False
                if not started:
                    name = self._device_name()
                    self.stop(quiet=True)
                    self._show_message(
                        f"{name} didn't restart receiving after the sample rate "
                        "change (details in Tools > Error History): press Start.",
                        "danger",
                        10000,
                    )
                    return
            if ok is False:
                # (Not remembered: it would be asked for at every open.)
                self._fill_rates()
                self._show_message(
                    f"{self._device_name()} didn't take {format_rate(rate)}.",
                    "danger",
                )
                return
            self._save(f"{KEY}sample_rate/{kind}", rate)
            # A new span: the filters, the noise floor and the pacing start
            # afresh (the audio fading out first); a narrower span may no
            # longer hold the channel.
            if self._audio_open:
                self._audio.write(self._dsp.release())
            self._dsp.reset()
            self._pacer.reset()
            self._rx_history.clear()
            self._refresh_state()
            self._ensure_vfo_fits()
            self._show_message(
                f"{format_rate(rate)}: {format_mhz_range(self._lo_hz - rate / 2, self._lo_hz + rate / 2)} "
                "in view.",
                None,
                3000,
            )

        def _apply_baseband_filter(self) -> None:
            """A HackRF's baseband filter follows its sample rate, as in the
            main window (0.75x the rate: the span without what would alias
            into it)."""
            dev = self._device
            if dev is None or not SDRMainWindow._has_baseband_filter(dev):
                return
            width = SDRMainWindow._BASEBAND_FILTER_FRACTION * self._rate()
            try:
                if dev.set_bandwidth(width) is False:
                    logger.warning("Could not set the HackRF baseband filter")
            except Exception as e:
                logger.warning(f"Could not set the HackRF baseband filter: {e}")

        def _source_index(self, key: str) -> int:
            for index, source in enumerate(self._sources):
                if source.key == key:
                    return index
            return -1

        def _selected_key(self) -> str:
            data = self._source_combo.currentData()
            return str(data) if data else DEMO_KEY

        def _selected_source(self) -> Optional[Source]:
            index = self._source_index(self._selected_key())
            return self._sources[index] if index >= 0 else None

        def _update_device_note(self) -> None:
            busy = [s for s in self._sources if s.busy]
            hardware = [s for s in self._sources if s.kind != "demo"]
            if not hardware:
                text = (
                    "No other SDR found: try it with the demo device, or plug "
                    "in a second RTL-SDR or HackRF One and open this list again."
                )
            elif busy and len(busy) == len(hardware):
                text = (
                    f"{busy[0].name} is the main window's. Plug in a second SDR "
                    "for two real receivers, or use the demo device here."
                )
            else:
                text = "Stop closes the device, so the main window can use it."
            self._device_note.setText(text)

        def _on_source_activated(self, _index: int) -> None:
            source = self._selected_source()
            if source is None:
                return
            if source.busy:
                self._show_message(
                    f"{source.name} is in use by {source.busy}: choose another "
                    "device.",
                    "warning",
                )
                self._refresh_sources()
                return
            self._save(KEY + "device", source.key)
            if self._source is not None and source.key == self._source.key:
                return
            if self._running:
                # Switch over: the new device receives at once, at a rate it
                # offers (the one last used with that kind of device).
                self.stop(quiet=True)
                self._fill_rates()
                self.start()
            else:
                self._fill_rates()  # (its rates; it opens on Start)
                self._sync_device_controls()
                self._refresh_state()

        def _open(self, source: Source) -> Optional[str]:
            """Open ``source`` as this receiver's device; why not, if it
            can't be."""
            if source.busy:
                return f"{source.name} is in use by {source.busy}"
            if source.kind == "demo":
                from .device_dialog import MockDevice

                device: Any = MockDevice()
            else:
                problem = self._host_problem(source)
                if problem:
                    return problem
                try:
                    device = source.device_class()
                    opened = device.open(source.index)
                except Exception as e:
                    logger.warning(f"Could not open {source.name}: {e}")
                    opened = False
                if not opened:
                    return (
                        f"{source.name} didn't open. Check that it is plugged "
                        "in and not in use by another program (details in "
                        "Tools > Error History)"
                    )
            try:
                device.set_sample_rate(self._chosen_rate())
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"{source.name}: could not set the sample rate: {e}")
            self._device = device
            self._source = source
            self._apply_baseband_filter()
            self._sync_device_controls()
            self._apply_gain()
            self._apply_front_end()  # (before tuning: HF depends on it)
            self._apply_tuning_range()
            note = self._sync_device_tuning()
            if note:
                self._show_message(note, "warning", 8000)
            return None

        def _host_problem(self, source: Source) -> str:
            """Why the main window's devices rule ``source`` out ("" if
            they don't): checked again just before opening."""
            held = {
                identity
                for identity in map(hardware_identity, main_window_devices(self._win))
                if identity is not None
            }
            if (source.kind, source.index) in held:
                return f"{source.name} is in use by the main window"
            return ""

        def _close_device(self) -> None:
            device, self._device = self._device, None
            self._source = None
            if device is None:
                return
            try:
                device.stop_rx()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Second receiver stop_rx failed: {e}")
            try:
                device.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Second receiver close failed: {e}")

        # ------------------------------------------------------ start / stop

        def toggle(self) -> None:
            if self._running:
                self.stop()
            else:
                self.start()

        def start(self) -> bool:
            """Open the chosen device (if needed) and start receiving."""
            if self._running:
                return True
            source = self._selected_source()
            if source is None:
                self._refresh_sources()
                source = self._selected_source()
            if source is None:  # pragma: no cover - the demo is always there
                return False
            if self._device is None or self._source is None:
                problem = self._open(source)
                if problem:
                    self._refresh_sources()
                    self._show_message(f"Can't start: {problem}.", "danger", 10000)
                    return False
            try:
                started = self._device.start_rx()
            except Exception as e:
                logger.error(f"The second receiver could not start: {e}")
                started = False
            if started is False:
                name = self._device_name()
                self._close_device()
                self._show_message(
                    f"{name} didn't start streaming (details in Tools > Error "
                    "History).",
                    "danger",
                    10000,
                )
                self._refresh_state()
                return False
            self._running = True
            self._dsp.reset()
            self._pacer.restart()
            self._rx_history.clear()
            self._realtime_hint = False
            self._frame_timer.start(self._FRAME_MS)
            self._audio_retry_at = 0.0
            self._sync_audio_output()
            self._refresh_state()
            self.running_changed.emit(True)
            logger.info(f"Second receiver started on {self._device_name()}")
            return True

        def stop(self, quiet: bool = False) -> None:
            """Stop receiving and close the device (so the main window can
            use it), ending a recording first. The frame the DSP worker has
            is played first (then the audio fades out)."""
            self._settle_frames()
            self._pipeline.close()
            was_running = self._running
            # (Where a recording that ends here was saved is said too.)
            self._recording_said = None
            if self._recorder is not None:
                self._set_recording(False)
            said, self._recording_said = self._recording_said, None
            self._running = False
            self._frame_timer.stop()
            if self._audio_open:
                self._audio.write(self._dsp.release())
            name = self._device_name()
            self._close_device()
            self._apply_tuning_range()  # (no device: any frequency)
            self._dsp.reset()
            self._sync_audio_output()
            self._level_meter.clear()
            self._refresh_state()
            if was_running:
                self.running_changed.emit(False)
                if not quiet and said is not None:
                    # Where the recording went first: at the window's
                    # minimum width the end of the line is cut off.
                    text, tone, duration = said
                    lead = "Stopped. " if tone == "success" else ""
                    self._show_message(
                        f"{lead}{text} {name} is closed.", tone, duration
                    )
                elif not quiet:
                    self._show_message(f"Stopped; {name} is closed.", None, 4000)

        # ------------------------------------------------------------ tuning

        def _tuning_limits(self) -> Tuple[float, float]:
            spec = getattr(self._device, "spec", None)
            low = getattr(spec, "freq_min", None)
            high = getattr(spec, "freq_max", None)
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if math.isfinite(low) and math.isfinite(high) and 0 < low < high:
                    return float(low), float(high)
            return MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ

        def _range_text(self) -> str:
            low, high = self._tuning_limits()
            kind = device_kind(self._device)
            who = _model_name(kind) if kind in ("rtlsdr", "hackrf") else "it"
            return f"{who} tunes {format_mhz_range(low, high)}"

        def _apply_tuning_range(self) -> None:
            low, high = self._tuning_limits()
            hardware = device_kind(self._device) in ("rtlsdr", "hackrf")
            self._freq_input.set_tuning_range(
                low, high, self._range_text() if hardware else ""
            )

        def _channel_fits(self, offset_hz: float, rate: Optional[float] = None) -> bool:
            """Whether the channel ``offset_hz`` from the LO lies inside the
            usable inner part of the span (as in the main window)."""
            if abs(offset_hz) < 0.5:
                return True
            rate = float(rate or self._rate())
            low, high = _channel_edges(self.mode(), self._bandwidth_hz(), rate)
            half = 0.5 * _USABLE_SPAN_FRACTION * rate
            return -half <= offset_hz + low and offset_hz + high <= half

        def _retune_lo(self, lo_hz: float) -> bool:
            self._settle_frames()  # (shown on the span it was taken on)
            dev = self._device
            if dev is not None:
                try:
                    ok = dev.set_frequency(float(lo_hz))
                except Exception as e:
                    logger.error(
                        f"The second receiver could not tune to "
                        f"{format_frequency(lo_hz)}: {e}"
                    )
                    ok = False
                if ok is False:
                    return False
            moved = abs(float(lo_hz) - self._lo_hz) >= 0.5
            self._lo_hz = float(lo_hz)
            rate = self._rate()
            self._spectrum.set_frequency_range(self._lo_hz, rate)
            self._waterfall.set_frequency_range(self._lo_hz, rate)
            if moved and isinstance(dev, SDRDevice):
                self._drop_stale_samples(dev)
            return True

        def _drop_stale_samples(self, dev: SDRDevice) -> None:
            """Hardware was retuned: drop what it queued on the old
            frequency, and have the plots skip the transfer in flight and
            the tuner's settling (as the main window does)."""
            now = time.monotonic()
            for _ in range(_RETUNE_MAX_DROPPED):
                try:
                    block = dev.read_samples(DISPLAY_BLOCK, timeout=0.0)
                except Exception as e:  # pragma: no cover - driver specific
                    logger.debug(f"Second receiver: can't drop samples: {e}")
                    break
                if block is None or len(block) == 0:
                    break
                self._rx_history.append((now, len(block)))  # (received)
            if getattr(getattr(dev, "state", None), "is_streaming", False):
                self._plot_skip_samples = _RETUNE_STALE_SAMPLES
                self._plot_skip_until = now + _RETUNE_SETTLE_S

        def tune(self, freq_hz: float, recenter: bool = False) -> bool:
            """Tune this receiver to ``freq_hz``: only the VFO moves while
            the channel fits the span, else the device is retuned. While
            recording, the device stays where the recording started (one
            file has one center frequency). False, with the reason in the
            status bar, when refused."""
            try:
                freq = float(freq_hz)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(freq):
                return False
            low, high = self._tuning_limits()
            if not low <= freq <= high:
                return self._refuse(
                    f"{format_frequency(freq)} is out of range: {self._range_text()}."
                )
            lo = self._lo_hz
            if self._record_lo is not None:
                if not self._channel_fits(freq - self._record_lo):
                    rate = self._rate()
                    covered = format_mhz_range(
                        self._record_lo - rate / 2, self._record_lo + rate / 2
                    )
                    return self._refuse(
                        f"Recording: {format_frequency(freq)} is outside the "
                        f"span being recorded ({covered}). Stop recording to "
                        "tune there."
                    )
            elif recenter or not self._channel_fits(freq - lo):
                lo = freq
            if abs(lo - self._lo_hz) >= 0.5 and not self._retune_lo(lo):
                return self._refuse(
                    f"{self._device_name()} didn't tune to {format_frequency(lo)}; "
                    f"it stays on {format_frequency(self._lo_hz)}.",
                    tone="danger",
                    log=True,
                )
            self._tuned_hz = freq
            self._on_tuned()
            return True

        def _refuse(self, reason: str, tone: str = "danger", log: bool = False) -> bool:
            self._freq_input.set_frequency(self._tuned_hz)
            self._show_message(reason, tone, 6000)
            if log:
                logger.warning(f"Second receiver: {reason}")
            return False

        def _on_tuned(self) -> None:
            """The tuned frequency changed: show it everywhere."""
            self._freq_input.set_frequency(self._tuned_hz)
            self._freq_label.setText(format_frequency(self._tuned_hz))
            self._dsp.offset_hz = self._tuned_hz - self._lo_hz
            self._update_passband()
            self._update_tx_group()

        def _sync_device_tuning(self) -> str:
            """Bring a device just opened to the LO and tuned frequency
            (into its range when they are outside); the note says why it
            moved ("" when it didn't)."""
            low, high = self._tuning_limits()
            wanted = self._tuned_hz
            note = ""
            if not low <= wanted <= high:
                moved = min(max(wanted, low), high)
                note = (
                    f"Tuned to {format_frequency(moved)}: {format_frequency(wanted)} "
                    f"is out of range ({self._range_text()})."
                )
                wanted = moved
            lo = self._lo_hz
            if not low <= lo <= high or not self._channel_fits(wanted - lo):
                lo = wanted
            if not self._retune_lo(lo):
                # The device says where it is: show that.
                actual = _number(
                    getattr(getattr(self._device, "state", None), "frequency", 0.0)
                )
                actual = actual or _number(getattr(self._device, "frequency", 0.0))
                if not low <= actual <= high:
                    actual = lo
                note = (
                    f"{self._device_name()} didn't tune to {format_frequency(lo)}; "
                    f"it is on {format_frequency(actual)}."
                )
                self._lo_hz = wanted = actual
                rate = self._rate()
                self._spectrum.set_frequency_range(actual, rate)
                self._waterfall.set_frequency_range(actual, rate)
            self._tuned_hz = wanted
            self._on_tuned()
            return note

        def _on_entry_frequency(self, freq_hz: float) -> None:
            self.tune(freq_hz)

        def _tune_steps(self, steps: int) -> None:
            self.tune(step_frequency(self._tuned_hz, self._step_hz, int(steps)))

        def _focus_entry(self) -> None:
            entry = getattr(self._freq_input, "_freq_input", self._freq_input)
            entry.setFocus(Qt.FocusReason.ShortcutFocusReason)
            select_all = getattr(entry, "selectAll", None)
            if callable(select_all):
                select_all()

        def _focus_plots(self) -> None:
            plot = self._spectrum if self._spectrum.isVisible() else self._waterfall
            plot.setFocus(Qt.FocusReason.ShortcutFocusReason)

        def _plots_focused(self) -> bool:
            focus = QApplication.focusWidget()
            return focus is not None and any(
                focus is plot or plot.isAncestorOf(focus)
                for plot in (self._spectrum, self._waterfall)
            )

        def _update_passband(self) -> None:
            low, high = self._dsp.band_edges()
            offset = self._tuned_hz - self._lo_hz
            for plot in (self._spectrum, self._waterfall):
                try:
                    plot.set_passband(offset + low, offset + high)
                    plot.set_tuned_frequency(self._tuned_hz)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"Could not show the passband: {e}")

        # ------------------------------------------------- mode, filter, step

        def _on_mode_changed(self, mode: str) -> None:
            """A mode brings its usual filter (and FM deviation), as in the
            main window."""
            self._fill_filters(mode)
            if not self._restoring:
                width, deviation = MODE_FILTER_DEFAULTS.get(
                    filter_profile(mode, self._tuned_hz), ("", "")
                )
                if width:
                    self._set_filter(width)
                if deviation:
                    self._set_deviation(deviation)
                self._load_step()
            self._tuning_form.setRowVisible(self._deviation_combo, mode == "FM")
            self._apply_dsp_settings()

        def _fill_filters(self, mode: str) -> None:
            combo = self._filter_combo
            current = combo.currentText()
            options = bandwidth_options(mode)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(options))
            if current in options:
                combo.setCurrentText(current)
            else:
                width = _parse_hz(current)
                choice = closest_bandwidth(options, width) if width else options[0]
                combo.setCurrentText(choice)
            combo.blockSignals(False)

        def _set_filter(self, width: str) -> None:
            combo = self._filter_combo
            index = combo.findText(width)
            if index < 0:
                hz = _parse_hz(width)
                if hz:
                    index = combo.findText(
                        closest_bandwidth(bandwidth_options(self.mode()), hz)
                    )
            if index >= 0:
                combo.setCurrentIndex(index)

        def _set_deviation(self, value: str) -> None:
            index = self._deviation_combo.findData(value)
            if index >= 0:
                self._deviation_combo.setCurrentIndex(index)

        def set_mode(self, mode: str, deviation: str = "", width: str = "") -> None:
            """Listen in ``mode`` (a Mode list name), with an FM
            ``deviation`` value and a filter ``width`` when given."""
            index = self._mode_combo.findText(mode)
            if index < 0:
                return
            self._mode_combo.setCurrentIndex(index)
            if mode == "FM" and deviation:
                self._set_deviation(deviation)
            if width:
                self._set_filter(width)
            self._apply_dsp_settings()

        def _on_filter_changed(self, _text: str) -> None:
            self._apply_dsp_settings()
            if not self._restoring:
                self._ensure_vfo_fits()

        def _on_deviation_changed(self) -> None:
            self._apply_dsp_settings()
            if not self._restoring:
                self._load_step()

        def _apply_dsp_settings(self) -> None:
            dsp = self._dsp
            dsp.rate = self._rate()
            dsp.mode = self.mode()
            dsp.bandwidth_hz = self._bandwidth_hz()
            dsp.deviation_hz = self._deviation_hz()
            dsp.offset_hz = self._tuned_hz - self._lo_hz
            self._update_passband()
            self._update_click_snap()

        def _ensure_vfo_fits(self) -> None:
            """A wider filter that no longer fits the span recenters."""
            if self._record_lo is None and not self._channel_fits(
                self._tuned_hz - self._lo_hz
            ):
                self.tune(self._tuned_hz, recenter=True)

        def _step_key(self) -> str:
            return step_mode_key(self.mode(), self._deviation_hz())

        def _load_step(self) -> None:
            key = self._step_key()
            step = self._settings.get_float(f"{KEY}step/{key}", default_step_hz(key))
            if not (math.isfinite(step) and step > 0):  # (a damaged setting)
                step = default_step_hz(key)
            self._set_step(step, save=False)

        def _set_step(self, step_hz: float, save: bool = True) -> None:
            step = nearest_step(step_hz)
            self._step_hz = step
            index = step_index(step)
            if self._step_combo.currentIndex() != index:
                self._step_combo.blockSignals(True)
                self._step_combo.setCurrentIndex(index)
                self._step_combo.blockSignals(False)
            self._freq_input.set_step_hz(step)
            self._update_click_snap()
            if save:
                self._save(f"{KEY}step/{self._step_key()}", step)

        def _on_step_index(self, index: int) -> None:
            step = self._step_combo.itemData(index)
            if step:
                self._set_step(float(step))

        def _change_step(self, delta: int) -> None:
            index = min(
                max(step_index(self._step_hz) + delta, 0), len(STEP_CHOICES_HZ) - 1
            )
            self._set_step(STEP_CHOICES_HZ[index])
            self._show_message(f"Step {format_step(self._step_hz)}", None, 2000)

        def _update_click_snap(self) -> None:
            for plot in (self._spectrum, self._waterfall):
                setter = getattr(plot, "set_click_snap", None)
                if callable(setter):
                    setter(self._step_hz, self._step_key())

        # ------------------------------------------------------------ gain

        def _gain_step_db(self, index: Optional[int] = None) -> float:
            """The RF gain slider's gain (dB): the step it is on."""
            steps = self._slider_steps
            if not steps:
                return 0.0
            if index is None:
                index = self._gain_slider.value()
            return float(steps[min(max(int(index), 0), len(steps) - 1)])

        def _hackrf_gains(self) -> Tuple[int, int, bool]:
            """``(lna, vga, amp)`` shown for a HackRF One."""
            lna = HACKRF_LNA_STEPS[self._lna_slider.value()]
            vga = HACKRF_VGA_STEPS[self._vga_slider.value()]
            return int(lna), int(vga), bool(self._amp_check.isChecked())

        def _sync_device_controls(self) -> None:
            """Show the controls of the device open, else of the one
            chosen: its own gain steps (an RTL-SDR's tuner table, the demo
            device's 1 dB steps) or a HackRF One's LNA, VGA and RF amp (the
            gain stays as close as the new steps allow), and an RTL-SDR's
            direct sampling and correction."""
            if self._device is not None:
                model = gain_model_for(self._device)
            else:
                model = gain_model_for_kind(self._rate_kind())
            hackrf = model.kind == "hackrf"
            form = self._device_form
            form.setRowVisible(self._gain_slider.parentWidget(), not hackrf)
            for widget in (
                self._lna_slider.parentWidget(),
                self._vga_slider.parentWidget(),
                self._amp_check,
            ):
                form.setRowVisible(widget, hackrf)
            self._gain_model = model
            if not hackrf and tuple(model.steps) != self._slider_steps:
                slider = self._gain_slider
                slider.blockSignals(True)
                self._slider_steps = tuple(model.steps)
                slider.setRange(0, max(0, len(model.steps) - 1))
                slider.setPageStep(max(1, len(model.steps) // 10))
                slider.setValue(model.nearest_index(self._gain_db))
                slider.blockSignals(False)
            self._show_gain()
            self._update_amp_warning()
            features = front_end_features(self._rate_kind())
            form.setRowVisible(self._direct_combo, "direct" in features)
            form.setRowVisible(self._ppm_spin, "ppm" in features)
            if self._device is None:
                self._show_direct_choice()

        def _show_gain(self) -> None:
            """The gain readouts: the gain the device applied (an RTL-SDR
            says which step of its tuner it took), else the step shown. With
            Auto: "Auto", or "Auto (25.9 dB)" where the device says what its
            automatic gain chose, as in the main window."""
            db = self._gain_step_db()
            dev = self._device
            lna, vga, _amp = self._hackrf_gains()
            self._lna_label.setText(format_db(lna))
            self._vga_label.setText(format_db(vga))
            if self._gain_auto.isChecked() and self._gain_model.kind != "hackrf":
                chosen = device_settings.auto_gain_db(dev)
                self._gain_label.setText(
                    "Auto"
                    if chosen is None
                    else f"Auto ({format_db(round(chosen, 1))})"
                )
                return
            if (
                isinstance(dev, SDRDevice)
                and self._gain_model.kind != "hackrf"
                and not self._gain_auto.isChecked()
            ):
                try:
                    state = dev.state
                    if state.gain_mode == "manual" and math.isfinite(state.gain):
                        db = float(state.gain)
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"Second receiver gain readout: {e}")
            self._gain_label.setText(format_db(db))

        def _update_amp_warning(self) -> None:
            show = self._gain_model.kind == "hackrf" and self._amp_check.isChecked()
            self._device_form.setRowVisible(self._amp_warning, show)

        def _on_gain_changed(self, value: int) -> None:
            if not self._restoring:
                self._gain_db = self._gain_step_db(value)
                self._save(KEY + "gain_db", self._gain_db)
            self._show_gain()
            self._apply_gain()

        def _on_stage_changed(self, _index: int) -> None:
            lna, vga, _amp = self._hackrf_gains()
            self._save(KEY + "hackrf_lna", lna)
            self._save(KEY + "hackrf_vga", vga)
            self._show_gain()
            self._apply_gain()

        def _on_amp_toggled(self, on: bool) -> None:
            self._save(KEY + "hackrf_amp", bool(on))
            self._update_amp_warning()
            self._apply_gain()

        def _on_gain_auto(self, on: bool) -> None:
            if on and not self._gain_model.agc:
                # (A HackRF One: no automatic gain.)
                name = (
                    self._device_name()
                    if self._device is not None
                    else self._gain_model.name or "This device"
                )
                self._gain_auto.blockSignals(True)
                self._gain_auto.setChecked(False)
                self._gain_auto.blockSignals(False)
                self._show_message(
                    f"{name} has no automatic gain: set it by hand.", "warning"
                )
                return
            self._gain_slider.setEnabled(not on)
            if not self._restoring:
                self._save(KEY + "gain_auto", bool(on))
            self._show_gain()
            self._apply_gain()

        def _apply_hackrf_gain(self, dev: Any) -> None:
            """Set a HackRF One's LNA, VGA and RF amp as shown."""
            lna, vga, amp = self._hackrf_gains()
            results = [  # (all three tried, whatever each says)
                _device_call(dev, "set_lna_gain", lna),
                _device_call(dev, "set_vga_gain", vga),
                _device_call(dev, "set_amp", amp),
            ]
            if not all(results):
                self._show_message(
                    f"{self._device_name()} didn't take LNA {lna} dB, VGA {vga} dB "
                    f"and the RF amp {'on' if amp else 'off'} (Tools > Error "
                    "History says why).",
                    "danger",
                )

        def _apply_gain(self) -> None:
            dev = self._device
            if dev is None:
                return
            if self._gain_model.kind == "hackrf":
                self._apply_hackrf_gain(dev)
                return
            auto = self._gain_auto.isChecked()
            try:
                ok = dev.set_gain_mode(auto)
            except Exception as e:
                logger.debug(f"Second receiver gain mode: {e}")
                ok = False
            if auto and ok is False:
                self._show_message(
                    f"{self._device_name()} has no automatic gain: set it by hand.",
                    "warning",
                )
                self._gain_auto.blockSignals(True)
                self._gain_auto.setChecked(False)
                self._gain_auto.blockSignals(False)
                self._gain_slider.setEnabled(True)
                auto = False
            if not auto:
                try:
                    dev.set_gain(self._gain_step_db())
                except Exception as e:
                    logger.debug(f"Second receiver gain: {e}")
            self._show_gain()

        # ------------------------------------------------------- front end

        def _direct_value(self) -> str:
            """The Direct sampling choice ("q", "i" or "off")."""
            value = self._direct_combo.currentData()
            return value if value in DIRECT_SAMPLING_VALUES else "q"

        def _show_direct_choice(self) -> None:
            """Show the Direct sampling choice: this receiver's own, else
            what the main window uses for an RTL-SDR (until one is chosen
            here: the two dongles may be different models)."""
            value = self._settings.get_str(KEY + "rtlsdr_direct_sampling", "")
            if value not in DIRECT_SAMPLING_VALUES:
                value = load_front_end(self._settings, "rtlsdr").direct_sampling
            self._direct_combo.setCurrentIndex(
                max(0, self._direct_combo.findData(value))
            )

        def _show_device_front_end(self) -> None:
            """Show what the RTL-SDR open uses (after a refused change)."""
            dev = self._device
            value = {0: "off", 1: "i", 2: "q"}.get(
                getattr(dev, "direct_sampling_branch", None)
            )
            if value is not None:
                self._direct_combo.setCurrentIndex(
                    max(0, self._direct_combo.findData(value))
                )
            ppm = getattr(dev, "freq_correction", None)
            if isinstance(ppm, int):
                self._ppm_spin.blockSignals(True)
                self._ppm_spin.setValue(ppm)
                self._ppm_spin.blockSignals(False)

        def _apply_front_end(self) -> None:
            """Set an RTL-SDR's direct sampling and correction as shown (when
            it opens, before it is tuned: direct sampling depends on where
            it tunes)."""
            dev = self._device
            features = front_end_features(self._rate_kind())
            if dev is None or not features:
                return
            self._show_direct_choice()
            failed: List[str] = []
            if "direct" in features and not _device_call(
                dev, "set_direct_sampling", self._direct_value()
            ):
                failed.append("direct sampling")
            ppm = int(self._ppm_spin.value())
            if "ppm" in features and not _device_call(dev, "set_freq_correction", ppm):
                failed.append(f"the {format_ppm(ppm)} correction")
            if failed:
                self._show_device_front_end()
                self._show_message(
                    f"{self._device_name()} didn't take {' and '.join(failed)} "
                    "(Tools > Error History says why).",
                    "danger",
                    8000,
                )

        def _on_direct_activated(self, _index: int) -> None:
            value = self._direct_value()
            dev = self._device
            if dev is not None and "direct" in front_end_features(self._rate_kind()):
                if not _device_call(dev, "set_direct_sampling", value):
                    self._show_device_front_end()
                    self._show_message(
                        f"{self._device_name()} didn't change its direct sampling "
                        "(Tools > Error History says why).",
                        "danger",
                    )
                    return
            self._save(KEY + "rtlsdr_direct_sampling", value)

        def _on_ppm_changed(self, ppm: int) -> None:
            dev = self._device
            if dev is not None and "ppm" in front_end_features(self._rate_kind()):
                if not _device_call(dev, "set_freq_correction", int(ppm)):
                    self._show_device_front_end()
                    self._show_message(
                        f"{self._device_name()} didn't take the {format_ppm(ppm)} "
                        "correction (Tools > Error History says why).",
                        "danger",
                    )
                    return
            self._save(KEY + "rtlsdr_ppm", int(ppm))

        # ---------------------------------------------------- squelch, audio

        def _on_squelch_enabled(self, on: bool) -> None:
            self._dsp.squelch.enabled = bool(on)
            self._squelch_auto.setEnabled(on)
            self._squelch_slider.setEnabled(on and not self._squelch_auto.isChecked())
            if not self._restoring:
                self._save(KEY + "squelch_on", bool(on))

        def _on_squelch_auto(self, auto: bool) -> None:
            gate = self._dsp.squelch
            if not auto and gate.last_threshold_db is not None:
                # Keep the threshold where Auto had it.
                self._set_threshold(gate.last_threshold_db)
            gate.auto = bool(auto)
            self._squelch_slider.setEnabled(
                self._squelch_check.isChecked() and not auto
            )
            if not self._restoring:
                self._save(KEY + "squelch_auto", bool(auto))

        def _on_squelch_slider(self, value: int) -> None:
            self._dsp.squelch.manual_db = float(value)
            self._squelch_label.setText(f"{int(value)} dBFS")
            if not self._restoring:
                self._save(KEY + "squelch_db", float(value))

        def _set_threshold(self, db: float) -> None:
            if not math.isfinite(db):  # (a damaged setting)
                db = -80.0
            value = int(round(min(max(db, SQUELCH_RANGE[0]), SQUELCH_RANGE[1])))
            self._squelch_slider.setValue(value)
            self._dsp.squelch.manual_db = float(value)

        def _on_meter_threshold(self, db: float) -> None:
            """A threshold dragged on LEVEL: the squelch is on and set by
            hand from now on (as on the main window's LEVEL)."""
            self._squelch_auto.setChecked(False)
            if not self._squelch_check.isChecked():
                self._squelch_check.setChecked(True)
            self._set_threshold(db)

        def _on_listen_index(self, index: int) -> None:
            key = self._listen_combo.itemData(index)
            if key not in _LISTEN_KEYS:
                return
            changed = key != self._listen
            self._listen = str(key)
            self._listen_combo.setToolTip(
                next(tip for k, _t, tip in LISTEN_CHOICES if k == key)
            )
            if not self._restoring:
                self._save(KEY + "listen", self._listen)
            self._audio_retry_at = 0.0  # (asked for: try the output at once)
            self._sync_audio_output()
            self._update_audio_note()
            if changed:
                self.listen_changed.emit(self._listen)

        def set_listen(self, key: str) -> None:
            """Choose whose audio plays (a key of :data:`LISTEN_CHOICES`)."""
            for index in range(self._listen_combo.count()):
                if self._listen_combo.itemData(index) == key:
                    self._listen_combo.setCurrentIndex(index)
                    return

        def _plays_here(self) -> bool:
            """Listen wants this receiver's audio (and audio is on)."""
            return (
                self._listen in (LISTEN_SECOND, LISTEN_BOTH) and self._main_audio_on()
            )

        def mutes_main(self) -> bool:
            """Listen plays only this receiver: the main window is silent."""
            return self._listen == LISTEN_SECOND and not self._shut_down

        def _sync_audio_output(self) -> None:
            """Open this receiver's speaker while it may play, close it
            otherwise."""
            want = self._running and self._plays_here() and self._dsp.audible
            if want and not self._audio_open:
                # (After a failure, tried again every few seconds, not on
                # every status tick.)
                now = time.monotonic()
                if now >= self._audio_retry_at:
                    self._follow_main_output()
                    self._audio_open = self._audio.start(int(_AUDIO_RATE))
                    if not self._audio_open:
                        self._audio_retry_at = now + _AUDIO_RETRY_S
                        if self._audio.available and not self._audio_failed:
                            logger.warning(
                                "The second receiver's audio output didn't open"
                            )
                        self._audio_failed = True
                    else:
                        self._audio_failed = False
            elif not want and self._audio_open:
                self._settle_frames()  # (its audio first, then the fade out)
                self._audio.write(self._dsp.release())
                self._audio.stop()
                self._audio_open = False
            self._update_audio_note()

        def _follow_main_output(self) -> None:
            """Play through the output the main window plays through (the
            one chosen in its preferences), where the sink can choose one
            (``output_id`` / ``set_output_device``)."""
            setter = getattr(self._audio, "set_output_device", None)
            main_sink = getattr(self._win, "_audio", None)
            device_id = getattr(main_sink, "output_id", None)
            if not callable(setter) or not isinstance(device_id, str):
                return
            if getattr(self._audio, "output_id", None) != device_id:
                setter(device_id)  # (an open stream reopens on it)

        def _on_volume_changed(self, value: int) -> None:
            self._audio.set_volume(int(value) / 100.0)
            self._volume_label.setText(f"{int(value)}%")
            if not self._restoring:
                self._save(KEY + "volume", int(value))

        def _output_failed(self) -> bool:
            """This receiver should play, but its audio output didn't open
            (it is tried again every few seconds)."""
            return (
                self._audio_failed
                and not self._audio_open
                and self._running
                and self._plays_here()
                and self._dsp.audible
            )

        def _update_audio_note(self) -> None:
            problem = self._audio_problem()
            if problem:
                text = f"{problem}: neither receiver can play."
                tone: Optional[str] = "warning"
            elif not self._main_audio_on():
                text = (
                    "Audio is off in the main window (its Audio button): "
                    "neither receiver plays."
                )
                tone = "warning"
            elif self._output_failed():
                # (An output that takes one stream at a time refuses the
                # second one.)
                text = (
                    "This receiver's audio output didn't open, so it is "
                    "silent; trying again every few seconds."
                )
                if self._listen == LISTEN_SECOND:
                    text += (
                        " The main window is muted too: choose Main window to "
                        "hear it."
                    )
                tone = "warning"
            elif self._listen == LISTEN_SECOND:
                text = "The main window is muted while this window is open."
                tone = "info"
            elif self._listen == LISTEN_MAIN:
                text = "Only the main window plays; this receiver is silent."
                tone = None
            else:
                text = "Both receivers play, each at its own volume."
                tone = None
            if not self._dsp.audible and self._listen != LISTEN_MAIN:
                text += " None (I/Q) mode has no audio."
            _show_note(self._audio_note, text, tone)

        # ------------------------------------------------------------ frames

        def _read(self) -> Optional[np.ndarray]:
            """The samples received since the last frame (as the main
            window's display loop takes them)."""
            dev = self._device
            if dev is None:
                return None
            if isinstance(dev, SDRDevice):
                limit = self._pacer.read_limit(self._rate(), time.monotonic())
                blocks: List[np.ndarray] = []
                total = 0
                while not blocks or total < limit:
                    block = dev.read_samples(DISPLAY_BLOCK, timeout=0.0)
                    if block is None or len(block) == 0:
                        break
                    blocks.append(block)
                    total += len(block)
                if len(blocks) <= 1:
                    return blocks[0] if blocks else None
                return np.concatenate(blocks)
            count = self._pacer.demo_request(self._rate(), time.monotonic())
            return dev.read_samples(count)

        def _frame(self) -> None:
            """One display frame: read, then hand the samples to the DSP
            worker and paint and play the last frame's result (_feed)."""
            if not self._running or self._device is None:
                return
            started = time.perf_counter()
            try:
                samples = self._read()
            except Exception as e:
                self._stream_failed(str(e))
                return
            if samples is None or len(samples) == 0:
                if self._pipeline.ready():  # (what the worker made: shown now)
                    self._present(self._pipeline.collect())
                self._check_stream()
                return
            self._feed(samples)
            self._pacer.measured(
                len(samples), self._rate(), time.perf_counter() - started
            )

        def _feed(self, samples: np.ndarray) -> None:
            """A display frame's samples: the DSP worker filters,
            demodulates and analyzes them (FrameWork) while this thread
            paints and plays the last frame's result, a frame behind. Only
            this thread touches Qt, and the worker is never more than a
            frame behind: collecting waits while its frame still runs (the
            pacer then sees slow frames)."""
            job = self._frame_job(samples)
            try:
                previous = self._pipeline.collect()
            finally:
                self._pipeline.submit(job)
            if previous is not None:
                self._present(previous)

        def _process(self, samples: np.ndarray) -> None:
            """One frame start to finish on this thread (a frame the DSP
            worker still has goes first). The display loop pipelines its
            frames through the worker instead (_feed)."""
            self._settle_frames()
            self._present(self._frame_job(samples)())

        def _frame_job(self, samples: np.ndarray) -> Callable[[], FrameResult]:
            """This thread's part of a frame (what was received, the audio's
            real-time hint, the recording, whether the plots want the frame)
            and the DSP's part, to run (FrameWork)."""
            now = time.monotonic()
            history = self._rx_history
            history.append((now, len(samples)))
            while history and now - history[0][0] > _REALTIME_WINDOW_S:
                history.popleft()
            self._dsp.rate = self._rate()
            play = self._audio_open and self._plays_here()
            realtime = self._realtime()
            if play and not realtime and not self._realtime_hint:
                self._realtime_hint = True
                self._show_message(
                    f"{self._device_name()} isn't delivering samples in real "
                    "time, so this receiver's audio is paused until it catches "
                    "up.",
                    "info",
                    8000,
                )
            elif realtime:
                self._realtime_hint = False
            plots = self._plots_want(samples)
            axis = (float(self._lo_hz), float(self._rate()))
            if self._recorder is not None:
                self._record(samples)
            return functools.partial(
                self._work, samples, play and realtime, plots, axis
            )

        def _plots_want(self, samples: np.ndarray) -> bool:
            """Whether the plots take this frame, as in the main window:
            every sample of it (of a frame longer than _PLOT_MAX_SAMPLES,
            about half a million, evenly spread FFT segments: every frame
            above 15 MS/s) at the FFT size and window chosen in the
            spectrum's menu (the mean power per bin for the spectrum, the
            strongest for the waterfall row, so a burst anywhere shows
            unless it is shorter than the gap between segments). Not the
            samples taken (partly) before the last retune
            (_drop_stale_samples), nor while nobody sees them (the receiver
            still plays; the first frame shown again starts afresh)."""
            if self._plot_skip_samples > 0 or time.monotonic() < self._plot_skip_until:
                self._plot_skip_samples = max(0, self._plot_skip_samples - len(samples))
                return False
            return not (self.isMinimized() or not self.isVisible())

        def _present(self, result: FrameResult) -> None:
            """Paint and play what the DSP made of a frame (this thread)."""
            frame, spectrum = result.frame, result.spectrum
            # (A frame from before a retune or a new rate is not painted on
            # the new span; one shorter than an FFT has no spectrum yet.)
            if spectrum is not None and result.axis == (
                float(self._lo_hz),
                float(self._rate()),
            ):
                self._spectrum.update_spectrum(spectrum.mean_db)
                # (As in the main window: while frames can't keep up their
                # length isn't the usual one, N-07.)
                self._waterfall.add_line(
                    spectrum.max_db,
                    noise_db=spectrum.noise_floor_db,
                    usual=not self._pacer.overloaded,
                    mean_db=spectrum.mean_db,  # (what the hover reads)
                )
            gate = self._dsp.squelch
            self._level_meter.set_reading(
                frame.level_db,
                frame.noise_db,
                frame.threshold_db,
                frame.state,
                gate.auto and frame.noise_db is not None,
            )
            if gate.auto and gate.last_threshold_db is not None:
                self._show_auto_threshold(gate.last_threshold_db)
            if frame.audio is not None and len(frame.audio) and self._audio_open:
                self._audio.write(frame.audio)

        def _settle_frames(self) -> None:
            """Paint and play the frame the DSP worker has (waiting for it),
            so none is in flight: before anything resets what the worker
            keeps (a new rate or FFT size, Stop) or moves the span (a
            retune)."""
            if not self._pipeline.pending:
                return
            try:
                result = self._pipeline.collect()
            except Exception:  # (the frame is lost; what comes next isn't)
                logger.exception("Second receiver: a frame's processing failed")
                return
            self._present(result)

        def _apply_plot_resolution(self, size: int, window: str) -> None:
            """The spectrum's FFT Size or Window menu: both plots use it."""
            size, window = normalize_fft_size(size), normalize_window(window)
            self._settle_frames()  # (the analyzer is the worker's meanwhile)
            if not self._frame_spectrum.configure(size, window):
                return
            self._spectrum.set_fft_size(size)
            self._spectrum.set_window_name(window)
            self._waterfall.set_fft_size(size)
            rbw = self._frame_spectrum.rbw(self._rate())
            self._show_message(
                f"Spectrum: {size}-point FFT, {WINDOW_LABELS[window]} window, "
                f"RBW {format_hz(rbw)}",
                None,
                4000,
            )

        def _show_plot_message(self, text: str, tone: str) -> None:
            """A plot's message (Copied ..., a measurement copied)."""
            self._show_message(text, tone or None, 3000)

        def _show_auto_threshold(self, db: float) -> None:
            """The slider shows the Auto threshold (without taking it)."""
            if not math.isfinite(db):
                return
            value = int(round(min(max(db, SQUELCH_RANGE[0]), SQUELCH_RANGE[1])))
            slider = self._squelch_slider
            if slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
                self._squelch_label.setText(f"{value} dBFS")

        def _realtime(self) -> bool:
            history = self._rx_history
            if len(history) < 2:
                return True
            span = history[-1][0] - history[0][0]
            if span < 1.0:
                return True
            received = sum(count for _t, count in list(history)[1:])
            return received >= _REALTIME_FRACTION * self._rate() * span

        def _check_stream(self) -> None:
            dev = self._device
            if not isinstance(dev, SDRDevice):
                return
            try:
                if dev.state.is_streaming:
                    return
            except Exception:  # pragma: no cover - defensive
                return
            self._stream_failed(str(getattr(dev, "rx_error", "") or ""))

        def _stream_failed(self, error: str) -> None:
            name = self._device_name()
            self.stop(quiet=True)
            detail = f": {error}" if error else ""
            message = (
                f"{name} stopped streaming{detail}. Check the USB connection, "
                "then press Start."
            )
            logger.warning(f"Second receiver: {message}")
            self._show_message(message, "danger", 10000)

        # -------------------------------------------------------- recording

        def _on_record_clicked(self) -> None:
            self._set_recording(self._record_button.isChecked())

        def _set_recording(self, on: bool) -> bool:
            """Start or stop recording this receiver's I/Q. True when the
            recording state is what was asked."""
            if on and self._recorder is None:
                ok = self._start_recording()
            elif not on and self._recorder is not None:
                self._stop_recording()
                ok = True
            else:
                ok = True
            self._sync_record_ui()
            return ok

        def _recording_name(self) -> str:
            rate = format_rate(self._rate()).replace(" ", "").replace("/s", "ps")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return f"iq_{self._lo_hz / 1e6:.3f}MHz_{rate}_{stamp}_rx2.sigmf-data"

        def _start_recording(self) -> bool:
            if not self._running:
                self._show_message(
                    "Start this receiver first (Start, or Space), then Record.",
                    "warning",
                )
                return False
            folder = file_dialogs.recordings_folder(self._settings)
            if not file_dialogs.ensure_folder(folder):
                self._show_message(
                    f"Can't record: the recordings folder {folder} can't be "
                    "created.",
                    "danger",
                    10000,
                )
                return False
            free = self._free_space(folder)
            if free is not None and free < _MIN_FREE_BYTES:
                self._show_message(
                    f"Can't record: only {format_size(free)} of disk space is left.",
                    "danger",
                    10000,
                )
                return False
            path = file_dialogs.unique_path(folder, self._recording_name())
            recorder = IQRecorder(
                self._rate(),
                center_frequency=self._lo_hz,
                sample_format=SampleFormat.FLOAT32,
                file_format=FileFormat.SIGMF,
            )
            recorder.set_metadata(
                hardware=self._device_name(),
                description=(
                    f"Second receiver, tuned to {format_frequency(self._tuned_hz)} "
                    f"({self.mode()})"
                ),
            )
            try:
                recorder.start(path)
            except OSError as e:
                self._show_message(f"Can't record to {path}: {e}", "danger", 10000)
                return False
            self._recorder = recorder
            # (Written from a thread of its own: the frames never wait for
            # the disk.)
            self._record_writer = BackgroundWriter(
                recorder.write,
                _capture_queue_bytes(self._rate()),
                name="Second receiver recording writer",
            )
            self._record_lost = 0
            self._record_path = path
            self._record_lo = self._lo_hz
            self._next_space_check = 0.0
            name, folder = _file_and_folder(path)
            self._show_message(
                f"Recording to {name} in {folder}; tuning stays inside the span "
                "while it records.",
                "info",
                6000,
            )
            logger.info(f"Second receiver recording to {path}")
            return True

        def _record(self, samples: np.ndarray) -> None:
            """Queue a copy of the frame's samples for the recording's
            writer thread (a write that failed meanwhile stops it)."""
            writer = self._record_writer
            if self._recorder is None or writer is None:
                return
            now = time.monotonic()
            if now >= self._next_space_check:
                self._next_space_check = now + _FREE_SPACE_CHECK_S
                free = self._free_space(self._record_path)
                if free is not None:  # (what the writer holds goes there too)
                    free -= writer.pending_bytes
                if free is not None and free < _MIN_FREE_BYTES:
                    self._stop_recording(
                        f"only {format_size(free)} of disk space is left"
                    )
                    self._sync_record_ui()
                    return
            try:
                queued = writer.put(samples)
            except (OSError, ValueError, RuntimeError) as e:
                self._stop_recording(f"the file can't be written ({e})")
                self._sync_record_ui()
                return
            if not queued:
                self._note_record_lost(len(samples))

        def _note_record_lost(self, count: int) -> None:
            """The writer had no room for a block (the disk can't keep up):
            a gap in the file, said once a recording (and when it ends)."""
            first = not self._record_lost
            self._record_lost += count
            if not first:
                return
            rate = self._rate()
            message = (
                f"The recording has gaps: {_span_text(count / max(rate, 1.0))} "
                "of signal was lost because this computer's disk couldn't keep "
                f"up with {format_rate(rate)}. A lower sample rate helps."
            )
            logger.warning(f"Second receiver: {message}")
            self._show_message(message, "warning", 15000)

        def _check_record_writer(self) -> None:
            """A write that failed on the writer thread stops the recording,
            also while no frame records (the status timer)."""
            writer = self._record_writer
            error = writer.error if writer is not None else None
            if error is not None:
                self._stop_recording(f"the file can't be written ({error})")
                self._sync_record_ui()

        @staticmethod
        def _free_space(path: str) -> Optional[int]:
            import os

            folder = path if os.path.isdir(path) else os.path.dirname(path)
            try:
                return int(shutil.disk_usage(folder or ".").free)
            except OSError:
                return None

        def _stop_recording(self, problem: str = "") -> None:
            recorder, self._recorder = self._recorder, None
            writer, self._record_writer = self._record_writer, None
            lost, self._record_lost = self._record_lost, 0
            path, self._record_path = self._record_path, ""
            self._record_lo = None
            if recorder is None:
                return
            if writer is not None:
                # What the writer holds is written first: the file is whole
                # before it is said to be saved.
                writer.close()
                error = writer.take_error()
                if error is not None and not problem:
                    problem = f"the file can't be written ({error})"
            rate = recorder.sample_rate
            gaps = f", {_span_text(lost / rate)} lost" if lost and rate > 0 else ""
            seconds = recorder.duration_seconds
            size = recorder.file_size_bytes
            name, folder = _file_and_folder(path)
            try:
                recorder.stop()
            except Exception as e:  # pragma: no cover - disk trouble
                logger.warning(f"Could not finish the recording {path}: {e}")
                problem = problem or f"its metadata couldn't be written ({e})"
                text = (
                    f"{name} in {folder} holds {_clock_text(seconds)} of "
                    "samples, but without its frequency and rate."
                )
            else:
                text = (
                    f"Saved {name} ({_clock_text(seconds)}, {format_size(size)}"
                    f"{gaps}) in {folder}. File > Open Recording in the main "
                    "window plays it."
                )
            if problem:
                text = f"Recording stopped: {problem}. {text}"
                logger.warning(text)
                said = (text, "danger", 20000)
            else:
                logger.info(text)
                said = (text, "success", 10000)
            self._recording_said = said
            self._show_message(*said)

        def _sync_record_ui(self) -> None:
            on = self._recorder is not None
            button = self._record_button
            button.blockSignals(True)
            button.setChecked(on)
            button.blockSignals(False)
            role = "danger" if on else ""
            if (button.property("role") or "") != role:
                set_role(button, role or None)
            button.setEnabled(self._running or on)
            if on:
                button.setToolTip("Stop recording (Ctrl+Shift+R)")
            elif self._running:
                button.setToolTip(
                    "Record this receiver's I/Q samples to a SigMF file in the "
                    "recordings folder (Ctrl+Shift+R)"
                )
            else:
                button.setToolTip("Start this receiver to record its I/Q samples")
            self._rec_label.setVisible(on)
            self._update_rec_label()

        def _update_rec_label(self) -> None:
            recorder = self._recorder
            if recorder is None:
                return
            size = format_size(recorder.file_size_bytes)
            self._rec_label.setText(
                f"REC {_clock_text(recorder.duration_seconds)} · {size}"
            )
            self._rec_label.setToolTip(f"Recording to {self._record_path}")

        # -------------------------------------------------------- bookmarks

        def _bookmarks_panel(self) -> Any:
            return getattr(self._win, "_bookmarks_panel", None)

        def _refresh_bookmarks(self) -> None:
            panel = self._bookmarks_panel()
            bookmarks = panel.bookmarks() if panel is not None else []
            combo = self._bookmark_combo
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(
                (
                    "Tune to a bookmark…"
                    if bookmarks
                    else "No bookmarks yet (Ctrl+B adds one)"
                ),
                None,
            )
            for index, bookmark in enumerate(bookmarks):
                try:
                    freq = float(bookmark.get("freq_hz") or 0.0)
                except (TypeError, ValueError):
                    continue
                label = str(bookmark.get("label") or "") or format_mhz(freq)
                group = str(bookmark.get("group") or "")
                text = f"{label} · {format_mhz(freq)}"
                if group:
                    text = f"{group} / {text}"
                combo.addItem(text, index)
            combo.setCurrentIndex(0)
            combo.setEnabled(bool(bookmarks))
            combo.blockSignals(False)

        def _on_bookmark_activated(self, index: int) -> None:
            data = self._bookmark_combo.itemData(index)
            self._bookmark_combo.setCurrentIndex(0)
            panel = self._bookmarks_panel()
            if data is None or panel is None:
                return
            bookmark = panel.bookmark(int(data))
            if bookmark is not None:
                self.tune_bookmark(bookmark)

        def tune_bookmark(self, bookmark: Dict[str, Any]) -> bool:
            """Tune to a bookmark, in its mode, deviation and filter."""
            try:
                freq = float(bookmark.get("freq_hz") or 0.0)
            except (TypeError, ValueError):
                return False
            if not self.tune(freq):
                return False
            demod, deviation, width, _tone = bookmark_listening(bookmark)
            if demod:
                self._listen_like_bookmark(demod, deviation, width)
            label = str(bookmark.get("label") or "") or format_frequency(freq)
            mode = "I/Q" if self.mode().startswith("None") else self.mode()
            self._show_message(f"Tuned to {label} ({mode})", "info", 3000)
            return True

        def _listen_like_bookmark(self, demod: str, deviation: str, width: str) -> None:
            """A bookmark's mode: its own deviation and width override the
            mode's defaults."""
            self.set_mode(demod, deviation, width)
            if not width and demod == "FM" and deviation:
                # The usual channel for that deviation (as the main window).
                hz = _parse_hz(deviation) or 5e3
                audio = 15e3 if hz >= 50e3 else 3e3
                need = 0.8 * 2.0 * (hz + audio)
                self._set_filter(closest_bandwidth(bandwidth_options("FM"), need))

        def bookmark_frequency(self, freq_hz: float) -> None:
            """Add ``freq_hz`` (this receiver's frequency, or a signal
            right-clicked on its waterfall) to the main window's bookmarks,
            in this receiver's mode and filter."""
            panel = self._bookmarks_panel()
            if panel is None:
                return
            freq = float(freq_hz)
            mode = self.mode()
            deviation = self._deviation_value() if mode == "FM" else ""
            width = self._filter_combo.currentText() if mode in _AUDIBLE_MODES else ""
            label = default_bookmark_name(freq)
            index, added = panel.add_unique(label, freq, mode, deviation, width)
            existing = panel.bookmark(index) or {}
            name = str(existing.get("label") or label)
            if added:
                short = "I/Q" if mode.startswith("None") else mode
                self._show_message(
                    f"Bookmarked {name} ({short}) in the main window's Bookmarks.",
                    "success",
                    4000,
                )
            else:
                self._show_message(f"Already bookmarked: {name}", "info", 3000)

        # ------------------------------------------------------- TX monitor

        def _tx_station(self) -> Any:
            return getattr(self._win, "_tx_station", None)

        def tx_target(self) -> Optional[Tuple[float, str, float, str]]:
            """``(frequency, TxJob mode, bandwidth, what)`` of the
            transmission on the air, or of the next station ID the main
            window would send; None when the main window can't transmit."""
            station = self._tx_station()
            if station is None:
                return None
            engine = getattr(station, "engine", None)
            job = getattr(engine, "job", None) if engine is not None else None
            if job is not None:
                return (
                    float(job.center_hz),
                    str(job.mode),
                    float(job.bandwidth_hz),
                    str(job.description),
                )
            try:
                if station.device_problem() is not None:
                    return None
                freq = float(self._win._current_frequency())
                _emission, mode, width, _summary = station.id_spec(freq)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"No transmit target: {e}")
                return None
            return freq, str(mode), float(width), "the next station ID"

        @staticmethod
        def tx_listening(mode: str, bandwidth_hz: float) -> Tuple[str, str, str]:
            """``(Mode, FM deviation, filter)`` to hear a transmission of
            TxJob ``mode`` and occupied ``bandwidth_hz``: FM with the
            deviation its width suggests (Carson's rule: about twice the
            deviation plus the audio), AM and FM in the narrowest channel
            that holds it; CW, USB and LSB with their own filter; narrow
            data (text modes sent on the carrier) in CW, wider data in FM."""
            mode = mode.upper()
            if mode == "DATA":
                if bandwidth_hz <= _CW_DATA_HZ:
                    # FSK or PSK on the carrier (RTTY, PSK31, ASCII FSK):
                    # the CW beat note makes its tones heard.
                    width = closest_bandwidth(
                        bandwidth_options("CW"), max(1.2 * bandwidth_hz, 250.0)
                    )
                    return "CW", "", width
                mode = "FM"  # (tones on an FM carrier)
            demod, deviation = _TX_MODES.get(mode, ("FM", "5 kHz"))
            if demod == "FM":
                deviation = next(
                    (value for limit, value in _FM_BY_WIDTH if bandwidth_hz <= limit),
                    "75 kHz",
                )
            if demod in ("FM", "AM"):
                width = closest_bandwidth(
                    bandwidth_options(demod), max(bandwidth_hz, 1.0)
                )
            else:
                width = ""  # (the mode's own filter)
            return demod, deviation, width

        def monitor_tx(self) -> bool:
            """Tune to the transmit frequency, in the transmission's mode,
            and start receiving (one click)."""
            target = self.tx_target()
            if target is None:
                self._show_message(
                    "Nothing to monitor: the main window transmits with a "
                    "HackRF One (Device > Connect... there).",
                    "warning",
                )
                return False
            freq, mode, bandwidth, what = target
            if not self.tune(freq, recenter=self._record_lo is None):
                return False
            demod, deviation, width = self.tx_listening(mode, bandwidth)
            self.set_mode(demod, deviation, width)
            if not self._running and not self.start():
                return False
            self._show_message(
                f"Monitoring {what} on {format_frequency(freq)} ({demod}).",
                "info",
                5000,
            )
            return True

        def _on_follow_toggled(self, on: bool) -> None:
            self._follow_tx = bool(on)
            if not self._restoring:
                self._save(KEY + "follow_tx", bool(on))

        def _on_tx_finished(self, _result: Any) -> None:
            if not self._main_gone:
                self._update_tx_group()

        def _on_tx_started(self, _job: Any) -> None:
            if self._main_gone:
                return
            if self._follow_tx and not self._shut_down:
                self.monitor_tx()
            self._update_tx_group()

        def _update_tx_group(self) -> None:
            station = self._tx_station()
            engine = getattr(station, "engine", None)
            on_air = bool(getattr(engine, "busy", False))
            target = self.tx_target()
            demo_here = self._source is not None and self._source.kind == "demo"
            if target is None:
                problem = None
                try:
                    problem = station.device_problem() if station is not None else None
                except Exception:  # pragma: no cover - defensive
                    problem = None
                text = (
                    "With a HackRF One in the main window, this receiver (another "
                    "device, such as an RTL-SDR) hears your transmissions and "
                    "keeps receiving while you transmit."
                )
                if problem:
                    text = (
                        f"The main window can't transmit: {problem}. Once it "
                        "can, this receiver (another device, such as an "
                        "RTL-SDR) hears your transmissions and keeps receiving "
                        "while you transmit."
                    )
                tone: Optional[str] = None
            else:
                freq, mode, _width, what = target
                where = format_frequency(freq)
                if on_air:
                    # How it is sent, in the main window's words ("45.45
                    # Bd, 170 Hz shift"), not the engine's job class.
                    job = getattr(engine, "job", None)
                    how = str(getattr(job, "summary", "") or "")
                    how = how or _TX_MODE_WORDS.get(mode.upper(), mode)
                    text = f"On air: {what} on {where} ({how})."
                    tone = "danger"
                else:
                    text = (
                        f"The main window transmits on {where} ({mode} for "
                        f"{what}). This receiver keeps receiving meanwhile "
                        "(full duplex)."
                    )
                    tone = None
                if demo_here:
                    text += (
                        " The demo device only simulates signals: it can't "
                        "hear a real transmission."
                    )
            _show_note(self._tx_label, text, tone)
            self._monitor_button.setEnabled(target is not None)
            if target is not None:
                self._monitor_button.setToolTip(
                    f"Tune this receiver to {format_frequency(target[0])} "
                    "and start it"
                )
            else:
                self._monitor_button.setToolTip(
                    "Available while the main window has a HackRF One that can "
                    "transmit"
                )

        # ------------------------------------------------------- presenting

        def _refresh_state(self) -> None:
            running = self._running
            button = self._start_button
            button.setText(self._STOP_TEXT if running else self._START_TEXT)
            role = "" if running else "primary"
            if (button.property("role") or "") != role:
                set_role(button, role or None)
            button.setToolTip(
                "Stop this receiver and close its device (Space)"
                if running
                else "Open the chosen device and start receiving (Space)"
            )
            if running:
                self._state_badge.setText("RUNNING")
                set_tone(self._state_badge, "success")
                self._device_label.setText(self._device_name())
                self._rate_label.setText(format_rate(self._rate()))
            else:
                self._state_badge.setText("STOPPED")
                set_tone(self._state_badge, "muted")
                source = self._selected_source()
                self._device_label.setText(source.name if source else "")
                self._rate_label.setText("")
            self._source_combo.setToolTip(
                "The device this receiver uses (choosing another while it "
                "runs switches to it). Any SDR the main window isn't using, "
                "or a second demo device to try it without hardware."
            )
            rate = self._rate()
            self._spectrum.set_frequency_range(self._lo_hz, rate)
            self._waterfall.set_frequency_range(self._lo_hz, rate)
            self._apply_dsp_settings()
            self._sync_record_ui()
            name = self._device_name() if running else ""
            self.setWindowTitle(
                f"Second Receiver — {name}" if name else "Second Receiver"
            )
            self._update_tx_group()
            self._update_audio_note()

        def _update_status(self) -> None:
            if self._main_gone:
                return
            if self._recorder is not None:
                self._check_record_writer()
                self._update_rec_label()
            self._update_tx_group()
            self._update_audio_note()
            if self._gain_auto.isChecked():
                self._show_gain()  # (what the automatic gain chose now)
            self._follow_main_output()  # (the output chosen meanwhile)
            want = self._running and self._plays_here() and self._dsp.audible
            if want != self._audio_open:
                self._sync_audio_output()

        def _show_message(
            self, text: str, tone: Optional[str] = None, duration_ms: int = 4000
        ) -> None:
            self._message_label.setText(text)
            self._message_label.setToolTip(text)
            set_tone(self._message_label, tone)
            self._message_timer.start(max(500, int(duration_ms)))

        def _clear_message(self) -> None:
            self._message_label.setText("")
            self._message_label.setToolTip("")
            set_tone(self._message_label, None)

        def message(self) -> str:
            return self._message_label.text()

        # ---------------------------------------------------------- keyboard

        def keyPressEvent(self, event):  # noqa: N802 - Qt naming
            """Space starts/stops this receiver, Left/Right tune it, [ and ]
            change its step (plots focused), Esc goes back to the plots: as
            in the main window, only when the focused widget doesn't use
            the key itself."""
            key = event.key()
            mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
            if key == Qt.Key.Key_Escape and mods == Qt.KeyboardModifier.NoModifier:
                entry = getattr(self._freq_input, "_freq_input", None)
                if entry is not None and entry.hasFocus():
                    self._freq_input.cancel_edit()
                self._focus_plots()
                event.accept()
                return
            if key == Qt.Key.Key_Space and mods == Qt.KeyboardModifier.NoModifier:
                if SDRMainWindow._focus_uses_key(key):
                    event.ignore()
                    return
                if not event.isAutoRepeat():
                    self.toggle()
                event.accept()
                return
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                if SDRMainWindow._focus_uses_key(key):
                    event.ignore()
                    return
                sign = 1 if key == Qt.Key.Key_Right else -1
                if mods == Qt.KeyboardModifier.NoModifier:
                    self._tune_steps(sign)
                elif mods == Qt.KeyboardModifier.ShiftModifier:
                    self._tune_steps(10 * sign)
                elif mods == Qt.KeyboardModifier.ControlModifier:
                    self.tune(self._tuned_hz + sign * 1e6)
                else:
                    super().keyPressEvent(event)
                    return
                event.accept()
                return
            if event.text() in ("[", "]") and self._plots_focused():
                self._change_step(1 if event.text() == "]" else -1)
                event.accept()
                return
            super().keyPressEvent(event)

        # ------------------------------------------------------ persistence

        def _save(self, key: str, value: Any) -> None:
            if self._restoring:
                return
            try:
                self._settings.set(key, value)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Could not save {key}: {e}")

        def _restore(self) -> None:
            s = self._settings
            self._restoring = True
            try:
                freq = s.get_float(KEY + "frequency_hz", _DEFAULT_FREQUENCY_HZ)
                if not (
                    math.isfinite(freq) and MIN_FREQUENCY_HZ <= freq <= MAX_FREQUENCY_HZ
                ):
                    freq = _DEFAULT_FREQUENCY_HZ
                lo = s.get_float(KEY + "lo_hz", freq)
                if not (
                    math.isfinite(lo) and MIN_FREQUENCY_HZ <= lo <= MAX_FREQUENCY_HZ
                ):
                    lo = freq
                mode = s.get_str(KEY + "mode", _DEFAULT_MODE)
                if self._mode_combo.findText(mode) < 0:
                    mode = _DEFAULT_MODE
                width, deviation = MODE_FILTER_DEFAULTS.get(
                    filter_profile(mode, freq), ("", "")
                )
                width = s.get_str(KEY + "bandwidth", width)
                deviation = s.get_str(KEY + "fm_deviation", deviation or "5 kHz")
                self._mode_combo.setCurrentText(mode)
                self._on_mode_changed(mode)  # (the filters, even if unchanged)
                self._set_deviation(deviation)
                if width:
                    self._set_filter(width)
                self._lo_hz, self._tuned_hz = lo, freq
                if not self._channel_fits(freq - lo):
                    self._lo_hz = freq
                self._squelch_check.setChecked(s.get_bool(KEY + "squelch_on", True))
                self._squelch_auto.setChecked(s.get_bool(KEY + "squelch_auto", True))
                self._set_threshold(s.get_float(KEY + "squelch_db", -80.0))
                self._on_squelch_enabled(self._squelch_check.isChecked())
                self._dsp.squelch.auto = self._squelch_auto.isChecked()
                gain = s.get_float(KEY + "gain_db", DEFAULT_GAIN_DB)
                self._gain_db = gain if math.isfinite(gain) else float(DEFAULT_GAIN_DB)
                self._gain_slider.setValue(
                    self._gain_model.nearest_index(self._gain_db)
                )
                self._gain_auto.setChecked(s.get_bool(KEY + "gain_auto", False))
                self._gain_slider.setEnabled(not self._gain_auto.isChecked())
                lna = s.get_float(KEY + "hackrf_lna", HACKRF_DEFAULT_LNA)
                vga = s.get_float(KEY + "hackrf_vga", HACKRF_DEFAULT_VGA)
                self._lna_slider.setValue(
                    _step_index(HACKRF_LNA_STEPS, lna, HACKRF_DEFAULT_LNA)
                )
                self._vga_slider.setValue(
                    _step_index(HACKRF_VGA_STEPS, vga, HACKRF_DEFAULT_VGA)
                )
                self._amp_check.setChecked(s.get_bool(KEY + "hackrf_amp", False))
                self._show_direct_choice()
                ppm = _number(s.get_float(KEY + "rtlsdr_ppm", 0.0))
                self._ppm_spin.setValue(
                    int(round(min(max(ppm, PPM_RANGE[0]), PPM_RANGE[1])))
                )
                volume = s.get_int(KEY + "volume", 70)
                self._volume_slider.setValue(min(max(volume, 0), 100))
                self._on_volume_changed(self._volume_slider.value())
                listen = s.get_str(KEY + "listen", DEFAULT_LISTEN)
                if listen not in _LISTEN_KEYS:
                    listen = DEFAULT_LISTEN
                self._listen = listen
                self.set_listen(listen)
                self._on_listen_index(self._listen_combo.currentIndex())
                self._follow_check.setChecked(s.get_bool(KEY + "follow_tx", False))
                self._follow_tx = self._follow_check.isChecked()
                self._load_step()
                self._refresh_sources()
                saved = s.get_str(KEY + "device", DEMO_KEY)
                index = self._source_index(saved)
                if index >= 0 and not self._sources[index].busy:
                    self._source_combo.setCurrentIndex(index)
                self._fill_rates()  # (the chosen device's rates)
                self._sync_device_controls()  # (and its gain controls)
            finally:
                self._restoring = False
            self._on_tuned()
            self._apply_dsp_settings()
            self._restore_waterfall_settings()
            self._restore_spectrum_settings()
            self._restore_geometry()

        def _save_spectrum_settings(self) -> None:
            if self._restoring:
                return
            shown = self._spectrum.display_settings()
            kept = {key: shown[key] for key in _SPECTRUM_KEPT if key in shown}
            try:
                self._save(SPECTRUM_KEY, json.dumps(kept))
            except (TypeError, ValueError) as e:  # pragma: no cover - defensive
                logger.debug(f"Could not save the spectrum settings: {e}")

        def _restore_spectrum_settings(self) -> None:
            raw = self._settings.get_str(SPECTRUM_KEY, "")
            if not raw:
                return
            saved = _saved_choices(raw, "spectrum")
            if saved is None:
                return
            kept = {key: saved[key] for key in _SPECTRUM_KEPT if key in saved}
            try:
                self._spectrum.apply_display_settings(kept)
            except Exception as e:  # (a damaged setting must not stop it)
                logger.warning(f"Ignoring the second receiver's spectrum settings: {e}")

        def _save_waterfall_settings(self) -> None:
            if self._restoring:
                return
            try:
                self._save(
                    WATERFALL_KEY, json.dumps(self._waterfall.display_settings())
                )
            except (TypeError, ValueError) as e:  # pragma: no cover - defensive
                logger.debug(f"Could not save the waterfall settings: {e}")

        def _restore_waterfall_settings(self) -> None:
            raw = self._settings.get_str(WATERFALL_KEY, "")
            if not raw:
                return
            saved = _saved_choices(raw, "waterfall")
            if saved is None:
                return
            try:
                # (A damaged value gives the one in use, logged.)
                self._waterfall.apply_display_settings(saved)
            except Exception as e:  # (a damaged setting must not stop it)
                logger.warning(
                    f"Ignoring the second receiver's waterfall settings: {e}"
                )

        def _restore_geometry(self) -> None:
            restored = False
            raw = self._settings.get_str(GEOMETRY_KEY, "")
            if raw:
                try:
                    restored = bool(
                        self.restoreGeometry(QByteArray.fromBase64(raw.encode("ascii")))
                    )
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"Could not restore the second receiver: {e}")
            if not restored:
                self._apply_default_geometry()
            for key, splitter in (
                (SPLITTER_KEY, self._splitter),
                (PLOTS_KEY, self._plots),
            ):
                raw = self._settings.get_str(key, "")
                ok = False
                if raw:
                    try:
                        ok = bool(
                            splitter.restoreState(
                                QByteArray.fromBase64(raw.encode("ascii"))
                            )
                        )
                    except Exception:  # pragma: no cover - defensive
                        ok = False
                if ok and min(splitter.sizes() or [0]) < 40:
                    ok = False
                if not ok and splitter is self._splitter:
                    width = max(self.width(), self.minimumWidth())
                    splitter.setSizes([width - 340, 340])

        def _apply_default_geometry(self) -> None:
            """1100x700 (smaller on a small screen), centred on the screen."""
            screen = self.screen() or QGuiApplication.primaryScreen()
            available = screen.availableGeometry() if screen is not None else None
            width, height = 1100, 700
            if available is not None:
                width = min(width, max(self.minimumWidth(), available.width() - 40))
                height = min(height, max(self.minimumHeight(), available.height() - 60))
            self.resize(width, height)
            if available is not None:
                self.move(
                    available.left() + max(0, (available.width() - width) // 2),
                    available.top() + max(0, (available.height() - height) // 2),
                )

        def persist(self) -> None:
            """Save the tuning, the device and the layout."""
            s = self._settings
            try:
                s.set(KEY + "frequency_hz", float(self._tuned_hz))
                s.set(KEY + "lo_hz", float(self._lo_hz))
                s.set(KEY + "mode", self.mode())
                s.set(KEY + "bandwidth", self._filter_combo.currentText())
                s.set(KEY + "fm_deviation", self._deviation_value())
                s.set(KEY + "device", self._selected_key())
                s.set(KEY + "listen", self._listen)
                s.set(
                    GEOMETRY_KEY,
                    bytes(self.saveGeometry().toBase64()).decode("ascii"),
                )
                for key, splitter in (
                    (SPLITTER_KEY, self._splitter),
                    (PLOTS_KEY, self._plots),
                ):
                    s.set(key, bytes(splitter.saveState().toBase64()).decode("ascii"))
                s.sync()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Could not save the second receiver: {e}")

        # ---------------------------------------------------------- closing

        def shutdown(self) -> None:
            """Stop (ending a recording, closing the device), save, and
            stop the timers: the window is closing, or the app is."""
            if self._shut_down or self._main_gone:
                return
            self.stop(quiet=True)  # (the DSP worker ends there)
            self.persist()
            self._status_timer.stop()
            self._frame_timer.stop()
            self._frame_spectrum.close()  # (its helper thread)
            self._audio.stop()
            self._audio_open = False
            self._shut_down = True
            # (Listen no longer mutes the main window.)
            self.listen_changed.emit(LISTEN_MAIN)

        def closeEvent(self, event):  # noqa: N802 - Qt naming
            self.shutdown()
            event.accept()

        def showEvent(self, event):  # noqa: N802 - Qt naming
            if self._main_gone:  # (an orphan: nothing to show it with)
                super().showEvent(event)
                return
            if self._shut_down:
                # Opened again: back in service.
                self._shut_down = False
                self._status_timer.start(self._STATUS_MS)
                self._refresh_sources()
                self.listen_changed.emit(self._listen)
            super().showEvent(event)
            self._refresh_state()

    class SecondReceiverHost(QObject):
        """The main window's side: Device > Second Receiver..., the one
        window (made on first use), and what the main window asks of it
        (whether Listen mutes it, whether a device is taken)."""

        def __init__(self, window: Any):
            super().__init__(window)
            self._win = window
            self._receiver: Optional[SecondReceiverWindow] = None
            self.action: Any = None
            self._muting = False  # (what the main window was last told)

        def add_menu_action(self, menu: Any) -> None:
            """Device > Second Receiver..."""
            self.action = self._win._add_action(
                menu,
                "&Second Receiver...",
                self.show,
                tip="Open a second receiver window on another device (or a "
                "second demo device): its own frequency, mode, squelch, "
                "spectrum and waterfall, and a choice of whose audio plays; "
                "it can also monitor the main window's transmissions",
            )

        @property
        def receiver(self) -> Optional["SecondReceiverWindow"]:
            return self._receiver

        def show(self) -> "SecondReceiverWindow":
            receiver = self._receiver
            if receiver is None:
                receiver = self._receiver = SecondReceiverWindow(self._win)
                receiver.listen_changed.connect(self._on_listen_changed)
            receiver.show()
            receiver.raise_()
            receiver.activateWindow()
            # Space and the arrow keys work at once (as in the main window).
            receiver._focus_plots()
            self._on_listen_changed(receiver.listen())
            return receiver

        def mutes_main(self) -> bool:
            """Listen in the second receiver plays only that one: the main
            window's speaker stays silent (while that window is open)."""
            receiver = self._receiver
            return bool(
                receiver is not None and receiver.isVisible() and receiver.mutes_main()
            )

        def device_problem(self, kind: str, index: int) -> str:
            """Why the main window can't open hardware device ``(kind,
            index)`` ("" when it can): the second receiver has it."""
            receiver = self._receiver
            if receiver is not None and receiver.holds(kind, index):
                name = receiver.source.name if receiver.source else _model_name(kind)
                return (
                    f"{name} is in use by the second receiver. Stop it there "
                    "(Device > Second Receiver...), or choose another device."
                )
            return ""

        def entry_problem(self, entry: Dict[str, Any]) -> str:
            """:meth:`device_problem` for a Device > Connect... entry
            (``{"type": "rtlsdr", "index": 0, ...}``)."""
            kind = str(entry.get("type", "") or "")
            if kind not in ("rtlsdr", "hackrf"):
                return ""
            try:
                index = int(entry.get("index", 0) or 0)
            except (TypeError, ValueError):
                index = 0
            return self.device_problem(kind, index)

        def _on_listen_changed(self, _key: str) -> None:
            """Tell the main window when Listen mutes (or unmutes) it: its
            Audio button says so ("Muted (2nd RX)"), and a status message."""
            muting = self.mutes_main()
            if muting == self._muting:
                return
            self._muting = muting
            sync = getattr(self._win, "_sync_audio_ui", None)
            if callable(sync):
                sync()
            if muting:
                text = (
                    "Main receiver muted: Listen in the Second Receiver window "
                    "plays only that receiver."
                )
            else:
                text = "Main receiver audio is back on."
            show = getattr(self._win, "_show_status_message", None)
            if callable(show):
                show(text, "info", 5000)

        def shutdown(self) -> None:
            """The main window is closing: stop, save and close the second
            receiver (a window of its own: it would stay open otherwise)."""
            receiver, self._receiver = self._receiver, None
            if receiver is not None:
                receiver.shutdown()
                receiver.close()
                receiver.deleteLater()


__all__ = [
    "DEMO_KEY",
    "Frame",
    "LISTEN_CHOICES",
    "ReceiverDsp",
    "SAMPLE_RATE",
    "Source",
    "device_kind",
    "hardware_identity",
    "list_sources",
    "main_window_devices",
]
if HAS_PYQT6:
    __all__ += ["SecondReceiverHost", "SecondReceiverWindow"]
