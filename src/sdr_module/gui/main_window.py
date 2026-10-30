"""
Main application window.

Provides the primary window with all panels and controls::

    +-----------------------------------------------------------------+
    | File  Device  Radio  View  Tools  Help                          |
    | [Start] [Record] | FREQ 100.000 MHz | LEVEL -40.0 dBFS          |
    +--------------------------------------+--------------------------+
    | spectrum                             | control panel (scrolls)  |
    |--------------------------------------|--------------------------|
    | waterfall                            | Decoder | Bookmarks | .. |
    +--------------------------------------+--------------------------+
    | [RUNNING] Demo device  2.4 MS/s   messages...           [REC]   |

All three splitters are user-resizable; their sizes persist between launches
and View > Reset Layout restores the defaults.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
import weakref
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

try:  # POSIX: a recording file is locked while a session uses it
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:
    from PyQt6.QtCore import (
        PYQT_VERSION_STR,
        QT_VERSION_STR,
        QEvent,
        QPoint,
        QPointF,
        Qt,
        QTimer,
    )
    from PyQt6.QtGui import (
        QAction,
        QActionGroup,
        QFont,
        QFontMetrics,
        QIcon,
        QKeySequence,
        QPainter,
        QPen,
        QPixmap,
        QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractItemView,
        QAbstractSlider,
        QAbstractSpinBox,
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QTabBar,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .. import __version__
from ..devices.base import SDRDevice
from ..dsp.frame_spectrum import (
    DEFAULT_FFT_SIZE,
    WINDOW_LABELS,
    FrameSpectrumAnalyzer,
    normalize_fft_size,
    normalize_window,
)
from ..dsp.nco import Nco
from ..dsp.stereo_pilot import StereoPilotDetector
from ..utils.tooltips import get_short_tip
from . import file_dialogs
from .audio_chain import (
    AGC_SPEEDS,
    AM_CARRIER_TIME_S,
    DEFAULT_CW_PITCH_HZ,
    AudioAgc,
    AudioJoin,
    AudioOptions,
    DcBlocker,
    IqConditioner,
    LevelAverager,
    NoiseFloorTracker,
    SquelchGate,
    ToneSquelch,
    channel_power_db,
    make_noise_reduction,
)
from .audio_recording import AudioTake, audio_file_name
from .audio_sink import AudioSink
from .bookmarks_panel import BookmarksPanel, default_bookmark_name
from .bookmarks_panel import listening as bookmark_listening
from .control_panel import (
    MAX_FREQUENCY_HZ,
    MIN_FREQUENCY_HZ,
    ControlPanel,
    bandwidth_options,
    closest_bandwidth,
)
from .decoder_panel import DecoderPanel
from .device_controls import (
    DeviceControls,
    ScanResultsDialog,
    confirm_switch_to_demo,
)
from .file_jobs import decode_recording, load_recording, run_job, write_recording
from .iq_files import (
    OPEN_FILTERS,
    REQUIRED,
    DecodeOptionsDialog,
    ImportOptionsDialog,
    IqFileOptions,
    RecordingInfoDialog,
    ask_recording_details,
    describe_iq_file,
    needs_options,
    split_name,
)
from .level_meter import LevelMeter
from .playback import PlaybackBar, PlaybackController
from .plot_view import PlotView, normalize_zoom
from .preferences_dialog import (
    KEY_AUDIO_OUTPUT,
    KEY_AUDIO_OUTPUT_NAME,
    PreferencesDialog,
    preferred_capture_folder,
)
from .rds_station import RdsListener, badge_text, badge_tooltip, tuner_text
from .receiver_bar import FrequencyDigits, ReceiverBar
from .scanner_host import WindowScannerHost
from .scanner_panel import ScannerPanel
from .settings_store import DEFAULT_FREQUENCY_HZ, GuiSettings
from .signal_panel import SignalPanel
from .signal_tools import SignalTools
from .spectrum_widget import SpectrumWidget, format_hz
from .themes import (
    apply_theme,
    current_theme,
    get_palette,
    normalize_theme,
    set_role,
    set_tone,
    theme_notifier,
)
from .transmitter import ArraySource, TxJob
from .tuning import (
    STEP_CHOICES_HZ,
    STEP_MODE_NAMES,
    TunedEntry,
    TuningHistory,
    default_step_hz,
    format_mhz_range,
    format_step,
    nearest_step,
    step_frequency,
    step_index,
    step_mode_key,
)
from .tx_station import StationTx
from .waterfall_widget import WaterfallWidget


class BandPreset(NamedTuple):
    """One entry of the Radio > Band Presets menu (and the first-run
    wizard's starting bands)."""

    label: str  # menu text; "&" marks the mnemonic
    frequency_hz: float
    mode: str  # Demodulation > Mode item text
    fm_deviation: Optional[str]  # FM deviation item text, None to keep
    bandwidth: str  # Bandwidth item text (channel width)


# Modes and channel widths match the Control Panel's presets (the frequency
# manager) where both list a band.
BAND_PRESETS: List[BandPreset] = [
    BandPreset("&FM Broadcast", 100.1e6, "FM", "75 kHz", "200 kHz"),
    BandPreset("&NOAA Weather", 162.55e6, "FM", "5 kHz", "25 kHz"),
    BandPreset("&2m Ham", 146.52e6, "FM", "5 kHz", "25 kHz"),
    BandPreset("&70cm Ham", 446.0e6, "FM", "5 kHz", "25 kHz"),
    BandPreset("&Airband AM", 125.0e6, "AM", None, "10 kHz"),
    BandPreset("A&DS-B", 1090e6, "None (I/Q)", None, "2.4 MHz"),
    BandPreset("ISM &433", 433.92e6, "None (I/Q)", None, "500 kHz"),
    BandPreset("ISM &915", 915e6, "None (I/Q)", None, "2 MHz"),
]
# Broadcast (wideband) FM: used by the AM/FM radio tuner and the defaults.
_BROADCAST_FM_DEVIATION = "75 kHz"
_BROADCAST_FM_BANDWIDTH = "200 kHz"
_BROADCAST_AM_BANDWIDTH = "10 kHz"
_FM_BROADCAST_BAND = (87.5e6, 108e6)


def _plain(label: str) -> str:
    """Menu text without its mnemonic marker: ``"A&DS-B"`` -> ``"ADS-B"``."""
    return label.replace("&&", "\0").replace("&", "").replace("\0", "&")


# Samples read per display frame (at least); also the length of the fixed
# measurement FFT (noise floor, S-meter). The plots' FFT size is the
# spectrum menu's choice (see dsp.frame_spectrum).
DISPLAY_BLOCK = 2048
# Equivalent noise bandwidth of the Hann window, in FFT bins.
_HANN_ENBW_BINS = 1.5
# Noise bandwidth of the channel filter relative to the channel width (its
# transition band, to 1.25x the edge, lets a little more noise through).
_CHANNEL_NOISE_WIDTH = 1.1
_DEFAULT_SAMPLE_RATE = 2.4e6
_AUDIBLE_MODES = ("AM", "FM", "USB", "LSB", "CW")
_AUDIO_RATE = 48000.0  # target rate of the demodulated audio
_NO_VALUE = "—"  # em dash for "not available"
_TX_UNAVAILABLE = "Connect a HackRF One to transmit an ID."
_DEFAULT_VOLUME = 70  # percent
# Audio is only played while the device delivers at least this fraction of
# its sample rate: less than that (a device or computer that can't keep up)
# would come out as a buzz of short bursts.
_REALTIME_FRACTION = 0.5
_REALTIME_WINDOW_S = 2.0
# The most signal one display frame takes from the device, in seconds (a
# stalled window catches up in steps), and the most until frames are known
# to keep up with real time (so finding out stalls a slow computer only
# briefly). See _Pacer.
_MAX_READ_S = 0.25
_PROBE_READ_S = 0.05
# A display frame (reading and processing its samples) that takes more than
# this fraction of the time its samples cover means this computer can't keep
# up; below the second fraction it can again. Frames shorter than
# _TIMED_FRAME samples aren't timed (their per-frame overhead dominates).
_OVERLOAD_RATIO = 0.8
_RECOVER_RATIO = 0.5
_TIMED_FRAME = 8 * DISPLAY_BLOCK
# While frames can't keep up, one frame this often reads (and times) a full
# frame's worth again, to find out when the computer keeps up once more.
_OVERLOAD_PROBE_S = 1.0
# After the LO of a hardware receiver moves, what it took on the old
# frequency is not drawn at the new one: the transfers it queued are
# dropped (at most this many), and the plots skip the samples of the
# transfer in flight (an RTL-SDR reads 256k samples at a time, 0.11 s at
# 2.4 MS/s; a HackRF 128k) and the tuner's settling time (as the Scanner,
# see scanner_host).
_RETUNE_MAX_DROPPED = 256
_RETUNE_STALE_SAMPLES = 256 * 1024
_RETUNE_SETTLE_S = 0.12

# Tuning (see SDRMainWindow.set_frequency): the tuned channel may sit
# anywhere in this inner fraction of the span (the receiver's anti-alias
# filter rolls off at its edges) before the LO is retuned to center it.
_USABLE_SPAN_FRACTION = 0.8
# A frequency joins the Back/Forward history and Recent Frequencies once
# the receiver has stayed on it this long (so stepping through a band
# doesn't fill them with every step).
_HISTORY_DWELL_MS = 1000
# Short model names, for "RTL-SDR tunes 0.5–1766 MHz".
_DEVICE_MODELS = {"RTLSDRDevice": "RTL-SDR", "HackRFDevice": "HackRF One"}

# Hardware drivers scanned for devices: (package, name the driver imports
# from it, driver module, driver class), matching the imports in
# devices/rtlsdr.py and devices/hackrf.py. A driver whose package can't be
# imported is skipped, so a missing package doesn't log an "is not
# installed" warning on every hot-plug poll.
_HARDWARE_DRIVERS = (
    ("RTL-SDR", "rtlsdr", "RtlSdr", "rtlsdr", "RTLSDRDevice"),
    ("HackRF", "python_hackrf", "pyhackrf", "hackrf", "HackRFDevice"),
)

# Optional ham radio panels
try:
    from ..ham.gui.callsign_panel import CallsignPanel
    from ..ham.gui.qrp_panel import QRPPanel
    from ..ham.gui.radio_tuner import RadioTunerWidget, band_step
    from ..ham.gui.signal_meter_widget import SignalMeterPanel
    from ..ham.gui.sstv_panel import ISS_FREQUENCY_HZ, SSTVPanel

    HAS_HAM_RADIO = True
except ImportError:
    HAS_HAM_RADIO = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_frequency(freq_hz: float) -> str:
    """Format a frequency in MHz with 3 to 6 decimals.

    ``100e6 -> "100.000 MHz"``, ``146.5125e6 -> "146.5125 MHz"``.
    """
    text = f"{max(0.0, float(freq_hz)) / 1e6:.6f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(3, '0')} MHz"


def format_rate(rate_hz: float) -> str:
    """Format a sample rate, e.g. ``"2.4 MS/s"``."""
    return f"{float(rate_hz) / 1e6:.3f}".rstrip("0").rstrip(".") + " MS/s"


def format_bandwidth(bw_hz: float) -> str:
    """Format a bandwidth in kHz below 1 MHz and in MHz above."""
    bw_hz = float(bw_hz)
    if bw_hz >= 1e6:
        return f"{bw_hz / 1e6:.3f}".rstrip("0").rstrip(".") + " MHz"
    return f"{bw_hz / 1e3:.2f} kHz"


def _parse_hz(text: str) -> Optional[float]:
    """``"25 kHz"`` -> 25000.0, ``"2.4 MHz"`` -> 2.4e6; None if unreadable."""
    value, _, unit = str(text).strip().partition(" ")
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit.strip().lower())
    try:
        hz = float(value) * (scale or 0.0)
    except ValueError:
        return None
    return hz if hz > 0 else None


# ---------------------------------------------------------------------------
# Receiver DSP: channel filter and demodulators
# ---------------------------------------------------------------------------

# SSB and CW filters. Their width is the Bandwidth control in those modes
# (its choices depend on the mode, see control_panel.MODE_BANDWIDTHS). The
# SSB passband starts _SSB_LOW_HZ from the carrier and is the Bandwidth
# wide; its lower edge falls off over _SSB_TRANSITION_HZ, so 50 Hz from the
# carrier and below (a carrier, or the hardware's DC spike, at the tuned
# frequency) is 60 dB down and never reaches the audio. CW keeps +/- half
# the Bandwidth around the carrier and adds a beat note (the CW pitch).
_SSB_LOW_HZ = 200.0
_SSB_TRANSITION_HZ = 150.0
_SSB_WIDTH_RANGE = (300.0, 10e3)
_CW_WIDTH_RANGE = (50.0, 3e3)
_NARROW_MODES = ("USB", "LSB", "CW")
# DC blocking of the demodulated audio: below the lowest broadcast bass
# for FM, above the hum of a fading carrier for the other modes.
_FM_DC_CORNER_HZ = 15.0
_FM_BROADCAST_DC_CORNER_HZ = 2.0
_AUDIO_DC_CORNER_HZ = 30.0
# AM with the AGC off: a fixed gain, so the loudness follows the signal.
_AM_FIXED_GAIN = 30.0
_AM_CARRIER_TIME_S = AM_CARRIER_TIME_S  # release, by AGC speed
_AM_ATTACK_S = 0.01
# The audio filters' start-up (their history fills): the DC blocker waits
# it out, so the noise in it doesn't linger (and a stream cut into blocks
# still gives exactly the audio of one long block).
_AUDIO_STARTUP_S = 0.012
# Options for chains nobody handed any (tests, one-off demodulation).
_DEFAULT_AUDIO_OPTIONS = AudioOptions()
# Squelch hang time: how long the level must stay below the threshold
# (minus the hysteresis) before the squelch closes; longer in CW, whose
# gaps between characters must not chop it.
_SQUELCH_HANG_S = 0.3
_CW_SQUELCH_HANG_S = 0.5
# FM audio clipping warning: more than this fraction of the samples beyond
# full scale for this long (the deviation is set too small), at most this
# often.
_CLIP_FRACTION = 0.05
_CLIP_MIN_SNR_DB = 15.0  # (noise clips whatever the deviation)
_CLIP_WARN_AFTER_S = 0.5
_CLIP_WARN_EVERY_S = 30.0
# Highest audio frequency kept after AM/FM detection (broadcast FM is 15 kHz),
# and where the audio low-pass stops: below broadcast FM's 19 kHz stereo
# pilot tone, which would otherwise come through as a faint whistle.
_MAX_AUDIO_HZ = 15e3
_AUDIO_STOP_HZ = 18.5e3
_MAX_TAPS = 2047
# Broadcast FM (an FM deviation of at least this much) is sent with 75 us
# pre-emphasis (50 us in Europe; the difference is a slight treble tilt),
# which the receiver undoes with the matching de-emphasis.
_BROADCAST_DEVIATION_HZ = 50e3
_DEEMPHASIS_S = 75e-6
# The de-emphasis runs at no less than this rate, where its FIR matches the
# analog response to within 0.1 dB up to 15 kHz.
_MIN_DEEMPHASIS_RATE = 192e3


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _polyphase_is_cheaper(groups: int, factor: int) -> bool:
    """Whether a decimating FIR of ``groups`` x ``factor`` taps costs less
    computed per kept output (one row-by-row dot product per group) than
    by FFT overlap-save. Measured per input sample: about 0.7 + 7 / factor
    units per group against 25 for the FFT."""
    return groups * (0.7 + 7.0 / max(1, factor)) <= 25.0


_VECDOT = getattr(np, "vecdot", None)  # numpy 2.0+


def _row_dot(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``rows @ weights`` for real ``weights``, without BLAS: a threaded
    BLAS spins its worker threads on every core after each call, which
    costs several times the work itself and stalls when other programs
    are busy."""
    if _VECDOT is not None:
        return _VECDOT(weights, rows)  # conj(weights) . row, weights are real
    return np.einsum("ij,j->i", rows, weights)


def _smooth_factor(ratio: float) -> int:
    """The largest integer <= ``ratio`` with no prime factor above 5 (at
    least 1). Decimating by such a factor splits into cheap short stages,
    unlike a prime like 167 (8 MS/s to 48 kHz), which needs one long filter
    at the full sample rate."""
    limit = max(1, int(np.floor(ratio + 1e-9)))
    best = 1
    power2 = 1
    while power2 <= limit:
        power3 = power2
        while power3 <= limit:
            value = power3
            while value * 5 <= limit:
                value *= 5
            best = max(best, value)
            power3 *= 3
        power2 *= 2
    return best


def _deemphasis_taps(rate_hz: float, tau_s: float = _DEEMPHASIS_S) -> np.ndarray:
    """The broadcast FM de-emphasis (a one-pole low-pass with time constant
    ``tau_s``) as an FIR: its exponential impulse response, sampled at
    ``rate_hz`` and cut off 80 dB down. At the channel rate (hundreds of
    kHz) this matches the analog response to within 0.1 dB up to 15 kHz,
    and as an FIR it folds into the audio filter, whose history carries it
    across blocks."""
    decay = float(np.exp(-1.0 / (tau_s * rate_hz)))
    length = int(np.ceil(np.log(1e-4) / np.log(decay))) + 1
    taps = decay ** np.arange(length)
    return (taps / np.sum(taps)).astype(np.float32)


def _lowpass_taps(
    pass_hz: float, stop_hz: float, rate_hz: float
) -> Optional[np.ndarray]:
    """Kaiser-windowed sinc low-pass: flat to ``pass_hz``, about 60 dB down
    from ``stop_hz``, unity gain at DC. None when there is nothing to remove
    (the passband reaches Nyquist)."""
    nyquist = rate_hz / 2.0
    if pass_hz >= 0.98 * nyquist:
        return None
    stop_hz = min(max(stop_hz, pass_hz * 1.05 + 1.0), nyquist)
    width = (stop_hz - pass_hz) / rate_hz  # transition, cycles per sample
    atten_db = 60.0
    taps = int(np.ceil((atten_db - 7.95) / (2.285 * 2.0 * np.pi * width))) + 1
    taps = min(max(taps, 15), _MAX_TAPS) | 1  # odd: symmetric, integer delay
    cutoff = (pass_hz + stop_hz) / 2.0 / rate_hz
    t = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * t)
    h *= np.kaiser(taps, 0.1102 * (atten_db - 8.7))
    return (h / np.sum(h)).astype(np.float32)


class _FirDecimator:
    """Streaming FIR filter followed by integer decimation.

    The filter history and the decimation phase carry over between blocks,
    so a stream split into blocks gives exactly the output of one long
    block.

    A decimating filter with few taps per kept sample only computes the
    samples it keeps (polyphase: the block, viewed as rows of ``factor``
    samples, dotted with the taps one group of ``factor`` taps at a time),
    several times cheaper than filtering every sample by FFT and discarding
    most. Longer filters, and filters that don't decimate, use FFT
    overlap-save.
    """

    def __init__(self, taps: Optional[np.ndarray], factor: int = 1):
        self._taps = None if taps is None else np.asarray(taps)
        self._factor = max(1, int(factor))
        self._history: Optional[np.ndarray] = None
        self._phase = 0  # index, in the next block, of the next kept sample
        self._spectra: Dict[Tuple[int, bool], np.ndarray] = {}
        # Polyphase: the reversed taps (zero-padded to whole groups) as
        # (groups, factor) rows, by sample dtype.
        self._groups = 0
        self._weights: Dict[Any, np.ndarray] = {}
        if self._taps is not None and self._factor > 1:
            groups = -(-len(self._taps) // self._factor)
            if _polyphase_is_cheaper(groups, self._factor):
                self._groups = groups

    def reset(self) -> None:
        """Forget the stream (history and decimation phase); keep the design."""
        self._history = None
        self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._groups and len(x):
            out = self._polyphase(x)
        else:
            if self._taps is not None and len(x):
                x = self._filter(x)
            out = x[self._phase :: self._factor]
        self._phase = (self._phase - len(x)) % self._factor
        return out

    def _polyphase(self, x: np.ndarray) -> np.ndarray:
        """The kept outputs of the filter, computed directly."""
        factor, groups = self._factor, self._groups
        length = groups * factor  # taps, padded with zeros
        dtype = np.result_type(x.dtype, np.float32)
        if self._history is None or self._history.dtype != dtype:
            self._history = np.zeros(length - 1, dtype=dtype)
        padded = np.concatenate((self._history, x.astype(dtype, copy=False)))
        self._history = padded[len(padded) - (length - 1) :].copy()
        phase = self._phase
        count = -(-(len(x) - phase) // factor) if len(x) > phase else 0
        if count <= 0:
            return np.empty(0, dtype=dtype)
        # Output j is padded[phase + j*factor : ... + length] . reversed taps,
        # i.e. the sum over groups g of row (j + g) times that group's taps.
        rows = count + groups - 1
        view = padded[phase : phase + rows * factor].reshape(rows, factor)
        weights = self._weights.get(dtype)
        if weights is None:
            padded_taps = np.zeros(length, dtype=np.float32)
            padded_taps[: len(self._taps)] = self._taps
            weights = padded_taps[::-1].reshape(groups, factor).astype(dtype)
            self._weights[dtype] = weights
        out = _row_dot(view[0:count], weights[0])
        for g in range(1, groups):
            out += _row_dot(view[g : g + count], weights[g])
        return out

    def _filter(self, x: np.ndarray) -> np.ndarray:
        taps = self._taps
        n_taps = len(taps)
        dtype = np.result_type(x.dtype, taps.dtype, np.float32)
        if self._history is None or self._history.dtype != dtype:
            self._history = np.zeros(n_taps - 1, dtype=dtype)
        padded = np.concatenate((self._history, x.astype(dtype, copy=False)))
        self._history = padded[len(padded) - (n_taps - 1) :]
        # Overlap-save: each FFT segment yields `step` valid outputs.
        n_out = len(x)
        nfft = max(1024, _next_pow2(4 * n_taps))
        nfft = min(nfft, max(_next_pow2(len(padded)), _next_pow2(n_taps)))
        step = nfft - n_taps + 1
        segments = -(-n_out // step)
        total = (segments - 1) * step + nfft
        if total > len(padded):
            padded = np.concatenate((padded, np.zeros(total - len(padded), dtype)))
        frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[::step]
        frames = frames[:segments]
        real = not np.iscomplexobj(padded)
        spectrum = self._spectrum(nfft, real)
        if real:
            y = np.fft.irfft(np.fft.rfft(frames, axis=1) * spectrum, nfft, axis=1)
        else:
            y = np.fft.ifft(np.fft.fft(frames, axis=1) * spectrum, axis=1)
        return y[:, n_taps - 1 :].reshape(-1)[:n_out].astype(dtype, copy=False)

    def _spectrum(self, nfft: int, real: bool) -> np.ndarray:
        key = (nfft, real)
        if key not in self._spectra:
            fft = np.fft.rfft if real else np.fft.fft
            self._spectra[key] = fft(self._taps, nfft)
        return self._spectra[key]


class _Resampler:
    """Streaming fractional resampler for real audio (numpy only).

    Each output sample is interpolated at its exact time in the input with a
    Kaiser-windowed sinc, whose taps come from a table of 256 fractional
    offsets (interpolated between neighbors). Flat to about 0.38x the input
    rate and 65 dB down above 0.6x, so audio that is already low-passed (to
    18.5 kHz at most) goes from a rate of 48-64 kHz to exactly 48 kHz
    without images or aliases. The time of the next output carries over
    between blocks, so the output rate is exact on average and the audio is
    continuous.
    """

    _TAPS = 24
    _PHASES = 256
    _BETA = 0.1102 * (60.0 - 8.7)  # Kaiser window for about 60 dB

    def __init__(self, in_rate: float, out_rate: float):
        self.step = float(in_rate) / float(out_rate)  # input samples per output
        taps, phases = self._TAPS, self._PHASES
        # Cutoff (cycles per input sample) a little inside half the lower rate.
        cutoff = 0.5 * 0.95 * min(1.0, 1.0 / self.step)
        frac = np.arange(phases + 1)[:, None] / phases
        tau = frac + (taps // 2 - 1) - np.arange(taps)[None, :]
        # Kaiser window, shifted to reach zero at its ends: the last row
        # (offset 1) is then exactly the first one a sample later.
        edge = np.sqrt(np.clip(1.0 - (tau / (taps / 2.0)) ** 2, 0.0, 1.0))
        window = (np.i0(self._BETA * edge) - 1.0) / (np.i0(self._BETA) - 1.0)
        table = 2.0 * cutoff * np.sinc(2.0 * cutoff * tau) * window
        table /= np.sum(table, axis=1, keepdims=True)  # unity gain at DC
        self._table = table.astype(np.float32)
        self.reset()

    def reset(self) -> None:
        """Start a new stream; the (costly) tap table is kept."""
        self._history = np.zeros(self._TAPS - 1, dtype=np.float32)
        # Position of the next output in the buffer (history + block).
        self._time = float(self._TAPS // 2 - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        taps, phases = self._TAPS, self._PHASES
        buffer = np.concatenate((self._history, np.asarray(x, dtype=np.float32)))
        # An output at time t uses buffer[floor(t) - taps/2 + 1 ...
        # floor(t) + taps/2]: those before `limit` are complete.
        limit = len(buffer) - taps // 2
        count = 0
        if self._time < limit:
            count = int(np.ceil((limit - self._time) / self.step))
        times = self._time + self.step * np.arange(count)
        times = times[times < limit]  # (rounding at the edge)
        count = len(times)
        whole = np.floor(times).astype(np.int64)
        position = (times - whole) * phases
        phase = np.minimum(position.astype(np.int64), phases - 1)
        blend = (position - phase).astype(np.float32)[:, None]
        weights = self._table[phase] * (1.0 - blend) + self._table[phase + 1] * blend
        index = (whole - (taps // 2 - 1))[:, None] + np.arange(taps)[None, :]
        out = np.sum(buffer[index] * weights, axis=1, dtype=np.float32)
        consumed = len(buffer) - (taps - 1)
        self._history = buffer[consumed:].copy()
        self._time = self._time + self.step * count - consumed
        return out.astype(np.float32, copy=False)


def _filter_width(mode: str, bandwidth: float, rate: float) -> float:
    """The Bandwidth control, limited to what ``mode`` can use: the SSB or
    CW filter width in those modes, else the channel width."""
    if mode in ("USB", "LSB"):
        low, high = _SSB_WIDTH_RANGE
        return min(max(float(bandwidth), low), high)
    if mode == "CW":
        low, high = _CW_WIDTH_RANGE
        return min(max(float(bandwidth), low), high)
    return min(max(float(bandwidth), 1e3), float(rate))


def _channel_edges(mode: str, bandwidth: float, rate: float) -> Tuple[float, float]:
    """Offsets from the tuned frequency (Hz) of what ``mode`` demodulates
    (and LEVEL, the squelch and the S-meter measure). In USB, LSB and CW
    ``bandwidth`` is the filter width: USB keeps 200 Hz to 200 Hz plus the
    width above the carrier, LSB the same below, CW +/- half the width."""
    width = _filter_width(mode, bandwidth, rate)
    if mode == "USB":
        return _SSB_LOW_HZ, _SSB_LOW_HZ + width
    if mode == "LSB":
        return -(_SSB_LOW_HZ + width), -_SSB_LOW_HZ
    half = width / 2.0
    return -half, half


def _noise_bandwidth(taps: Optional[np.ndarray], rate: float) -> float:
    """Equivalent noise bandwidth (Hz) of an FIR with unity gain in its
    passband, running at ``rate``."""
    if taps is None or len(taps) == 0:
        return float(rate)
    taps = np.asarray(taps)
    gain = abs(complex(np.sum(taps)))
    if gain <= 0.0:
        return float(rate)
    return float(rate * np.sum(np.abs(taps) ** 2) / gain**2)


def _channel_centroid_hz(power_db: np.ndarray, chain: Any) -> Optional[float]:
    """Where the tuned channel's power is centered, in Hz from the tuned
    frequency (from the spectrum, linear power weights), or None for an
    empty spectrum."""
    n = len(power_db)
    if n == 0:
        return None
    bin_hz = chain.rate / n
    low, high = chain.band_edges()
    offset = float(getattr(chain, "offset_hz", 0.0))
    center = n // 2  # fftshift puts the LO (0 Hz) here
    first = max(0, center + int(np.floor((offset + low) / bin_hz + 0.5)))
    last = min(n - 1, center + int(np.ceil((offset + high) / bin_hz - 0.5)))
    if last <= first:
        return 0.0
    weights = np.power(10.0, power_db[first : last + 1].astype(np.float64) / 10.0)
    freqs = (np.arange(first, last + 1) - center) * bin_hz - offset
    total = float(np.sum(weights))
    return float(np.dot(weights, freqs) / total) if total > 0 else 0.0


def _noise_per_bin(power_db: np.ndarray) -> float:
    """Mean noise power per bin (dB) of spectrum bins that are mostly noise.

    White-noise bins are exponentially distributed. Bins above three times
    the noise mean count as signals and are left out; the median of the
    rest, corrected for that cut (``1 - exp(-x) = (1 - exp(-3)) / 2``),
    gives the mean. Two rounds from the plain median: a band where
    stations fill a third of the bins reads about 0.1 dB above an empty
    one (the plain median read ~2.5 dB high).
    """
    linear = 10.0 ** (np.asarray(power_db, dtype=np.float64) / 10.0)
    cut = 3.0
    median_of_cut = -np.log(1.0 - 0.5 * (1.0 - np.exp(-cut)))
    mean = float(np.median(linear)) / np.log(2.0)
    for _ in range(2):
        kept = linear[linear < cut * mean]
        if len(kept) < 16:
            break
        mean = float(np.median(kept)) / median_of_cut
    return 10.0 * float(np.log10(mean + 1e-30))


class _ReceiverChain:
    """Selects the tuned channel and turns it into audio.

    ``channelize`` low-pass filters the I/Q block to the channel (the
    Bandwidth control) and decimates it; the S-meter measures that and
    ``demodulate`` turns it into mono audio at exactly 48 kHz. Without the
    channel filter the FM discriminator locks onto the strongest station
    anywhere in the 2.4 MHz span and the AM detector hears them all.

    The sample rate comes down to an intermediate rate of 48-64 kHz by an
    integer factor with no prime factor above 5, in short stages (so 8 or 20
    MS/s cost little more per sample than 2.4 MS/s), and a fractional
    resampler takes that to 48 kHz (not needed at 2.4 MS/s).

    AM and FM audio is low-passed to at most 15 kHz, 60 dB down by 18.5 kHz
    (below broadcast FM's 19 kHz stereo pilot); FM with a broadcast
    deviation (50 kHz or more) is also de-emphasized (75 us).

    In USB, LSB and CW the Bandwidth is the filter width, and the channel
    filter ends in that filter: USB/LSB keep 200 Hz to 200 Hz plus the width
    on their side of the carrier, CW +/- half the width around it (and the
    demodulator adds the CW pitch as a beat note). So the channel the
    S-meter, LEVEL and the squelch measure is exactly what is heard.

    All audio is DC-blocked (a mistuned FM station otherwise comes out
    offset and clips), and ``options`` (an :class:`~.audio_chain.
    AudioOptions`, shared with the window) sets the CW pitch, the AGC speed
    of AM, SSB and CW, and the noise reduction, all without rebuilding the
    chain.

    The tuned channel need not be at the center of the span: ``offset_hz``
    (set with :meth:`set_offset`) is the VFO's offset from the LO, and
    ``channelize`` first mixes every block down by it with a
    phase-continuous NCO. Changing it keeps the filters' state, so retuning
    inside the span doesn't click.
    """

    options: Optional[AudioOptions] = None

    @staticmethod
    def clamp_bandwidth(mode: str, bandwidth: float, rate: float) -> float:
        """The width a chain for ``mode`` uses for a Bandwidth of
        ``bandwidth`` (what ``chain.bandwidth`` will be)."""
        return _filter_width(mode, bandwidth, rate)

    def __init__(
        self, rate: float, mode: str, bandwidth: float, offset_hz: float = 0.0
    ):
        self.rate = float(rate)
        self.mode = mode
        self._nco = Nco(self.rate, offset_hz)
        self.bandwidth = self.clamp_bandwidth(mode, bandwidth, self.rate)
        self.audio_rate = _AUDIO_RATE
        total = _smooth_factor(self.rate / _AUDIO_RATE)
        self.intermediate_rate = self.rate / total
        half = self.bandwidth / 2.0
        if mode in ("USB", "LSB", "CW"):
            # Straight to the intermediate rate; the sideband is cut there.
            first = total
        else:
            # Keep the whole channel (plus a guard band) for AM/FM detection.
            need = max(1.25 * self.bandwidth, self.intermediate_rate)
            first = max(
                (d for d in range(1, total + 1) if total % d == 0),
                key=lambda d: d if self.rate / d >= need else 0,
            )
            if self.rate / first < need:
                first = 1
        self.channel_rate = self.rate / first
        # USB/LSB: complex band-pass for one sideband; CW: narrow low-pass.
        # The last stage of the channel filter (see _channel_stages).
        self._sideband_taps: Optional[np.ndarray] = None
        self._noise_bw = self.rate
        self._channel = self._channel_stages(first, half, mode)
        self._audio_factor = total // first
        self._audio_cutoff = min(half, _MAX_AUDIO_HZ, 0.45 * self.intermediate_rate)
        self._deemphasis = False  # set by demodulate() for broadcast FM
        self._design_demodulator()
        # Stream objects are built once and only reset after a gap: muted or
        # squelched frames reset the demodulator every frame, and rebuilding
        # the resampler table and filter designs each time cost ~0.6 ms.
        self._audio_cache: Dict[bool, List[_FirDecimator]] = {}
        self._sideband: Optional[_FirDecimator] = None  # (now in the channel)
        self._resampler: Optional[_Resampler] = None
        if abs(self.intermediate_rate - self.audio_rate) > 1e-6 * self.audio_rate:
            self._resampler = _Resampler(self.intermediate_rate, self.audio_rate)
        corner = _FM_DC_CORNER_HZ if mode == "FM" else _AUDIO_DC_CORNER_HZ
        self._dc_blocker = self._make_dc_blocker(corner)
        self._audio_agc = AudioAgc(self.audio_rate)
        # What the last demodulate() call saw: the fraction of FM audio
        # beyond full scale (clipped: the deviation is set too small), and
        # the audio before noise reduction (for the SSTV decoder).
        self.last_clip_fraction = 0.0
        self.pre_nr_audio = np.empty(0, dtype=np.float32)
        self.reset_demodulator()

    def _options(self) -> AudioOptions:
        options = self.options
        return options if options is not None else _DEFAULT_AUDIO_OPTIONS

    def _make_dc_blocker(self, corner_hz: float) -> DcBlocker:
        startup = int(round(_AUDIO_STARTUP_S * self.audio_rate))
        return DcBlocker(corner_hz, self.audio_rate, startup)

    def noise_bandwidth(self) -> float:
        """Equivalent noise bandwidth (Hz) of the channel: an empty channel
        measures the noise density times this."""
        return self._noise_bw

    def _channel_stages(
        self, factor: int, half: float, mode: str
    ) -> List[_FirDecimator]:
        """Filters that keep +/- ``half`` Hz (the channel) and decimate by
        ``factor`` to ``channel_rate``.

        The channel filter is sharp (stopband at 1.25x the channel edge), so
        a strong station next door doesn't leak into the S-meter or the
        demodulator. It runs after a coarse decimation stage where possible,
        which keeps it short.

        In USB, LSB and CW the decimation only needs to keep +/- a quarter
        of the channel rate (a lighter filter), and the SSB or CW filter at
        the channel rate ends the chain: the channel is the passband.
        """
        rate = self.rate
        narrow = mode in _NARROW_MODES
        if factor == 1 and half >= 0.45 * rate and not narrow:
            return []  # the channel is the whole span: nothing to remove
        if narrow:
            pass_hz = 0.25 * self.channel_rate
        else:
            pass_hz = min(half, 0.45 * self.channel_rate)
        stop_hz = min(pass_hz * 1.25, max(self.channel_rate - pass_hz, pass_hz * 1.05))
        if factor == 1:
            stop_hz = min(stop_hz, rate / 2.0)
        # Coarse stage: the largest divisor of the factor that still leaves
        # room (4x the stopband edge) for the sharp stage's transition band.
        coarse = max(
            (d for d in range(1, factor + 1) if factor % d == 0),
            key=lambda d: d if rate / d >= 4.0 * stop_hz else 0,
        )
        if rate / coarse < 4.0 * stop_hz:
            coarse = 1
        stages = []
        if coarse > 1:
            taps = _lowpass_taps(stop_hz, rate / coarse - pass_hz, rate)
            stages.append(_FirDecimator(taps, coarse))
        fine_rate = rate / coarse
        fine = _lowpass_taps(pass_hz, stop_hz, fine_rate)
        stages.append(_FirDecimator(fine, factor // coarse))
        self._noise_bw = _noise_bandwidth(fine, fine_rate)
        if narrow:
            prototype, self._sideband_taps = self._narrow_taps()
            stages.append(_FirDecimator(self._sideband_taps))
            self._noise_bw = _noise_bandwidth(prototype, self.channel_rate)
        return stages

    def _narrow_taps(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(low-pass prototype, filter)`` of the SSB or CW filter at the
        channel rate: the prototype shifted to the sideband for USB/LSB (a
        complex band-pass), the prototype itself for CW."""
        rate = self.channel_rate
        width = self.bandwidth
        if self.mode == "CW":
            # Centred on the edge, so the filter is ``width`` wide at -6 dB
            # (and in noise bandwidth); half the width at most, so narrow
            # filters still pass something flat.
            half = width / 2.0
            transition = min(max(0.5 * width, 60.0), 250.0)
            taps = _lowpass_taps(
                max(half - transition / 2.0, 1.0), half + transition / 2.0, rate
            )
            assert taps is not None  # a CW filter is far below Nyquist
            return taps, taps
        half = width / 2.0
        taps = _lowpass_taps(half, half + _SSB_TRANSITION_HZ, rate)
        assert taps is not None  # an SSB filter is far below Nyquist
        center = _SSB_LOW_HZ + half
        if self.mode == "LSB":
            center = -center
        t = np.arange(len(taps)) - (len(taps) - 1) / 2.0
        shift = np.exp(2j * np.pi * center * t / rate)
        return taps, (taps * shift).astype(np.complex64)

    def band_edges(self) -> Tuple[float, float]:
        """Offsets from the tuned frequency (Hz) of what is demodulated."""
        return _channel_edges(self.mode, self.bandwidth, self.rate)

    @property
    def offset_hz(self) -> float:
        """The tuned (VFO) frequency minus the LO, in Hz: where in the span
        the channel is."""
        return self._nco.frequency_hz

    @offset_hz.setter
    def offset_hz(self, value: float) -> None:
        self._nco.frequency_hz = value

    def set_offset(self, offset_hz: float) -> None:
        """Move the channel to ``offset_hz`` from the LO. The NCO's phase and
        the filters' state carry on, so the audio doesn't click."""
        self._nco.frequency_hz = offset_hz

    def _design_demodulator(self) -> None:
        """Design the filters after detection (once per chain)."""
        rate = self.channel_rate
        # AM/FM: an optional coarse decimation, then the audio low-pass (with
        # the de-emphasis, for broadcast FM) and decimation to the
        # intermediate rate.
        self._audio_coarse: Optional[np.ndarray] = None
        self._audio_coarse_factor = 1
        self._audio_taps: Optional[np.ndarray] = None
        self._audio_rate_in = rate  # the rate the audio low-pass runs at
        # (USB/LSB/CW: the sideband or CW filter ends the channel filter.)
        if self.mode in ("AM", "FM") and (
            self._audio_factor > 1 or self._audio_cutoff < 0.45 * rate
        ):
            self._design_audio_filters()

    def _design_audio_filters(self) -> None:
        """AM/FM audio: low-pass to ``_audio_cutoff`` (stopping below the
        19 kHz pilot) and decimate by ``_audio_factor``.

        A wide channel (a high channel rate) first comes down by a coarse
        stage, like the channel filter's, to no less than 192 kHz: the
        sharp filter is then short, and the de-emphasis, an FIR sampled at
        that rate, stays within 0.1 dB of the analog response.
        """
        rate, factor, cutoff = self.channel_rate, self._audio_factor, self._audio_cutoff
        floor = max(4.0 * _AUDIO_STOP_HZ, _MIN_DEEMPHASIS_RATE)
        coarse = max(
            (d for d in range(1, factor + 1) if factor % d == 0),
            key=lambda d: d if rate / d >= floor else 0,
        )
        if rate / coarse < floor:
            coarse = 1
        fine_rate, fine_factor = rate / coarse, factor // coarse
        if fine_factor > 1:
            stop = fine_rate / fine_factor - cutoff  # where aliases start
        else:
            stop = cutoff * 1.3
        stop = min(stop, _AUDIO_STOP_HZ)
        if coarse > 1:
            # Removes only what would alias below the sharp filter's stopband.
            self._audio_coarse = _lowpass_taps(stop, fine_rate - cutoff, rate)
            self._audio_coarse_factor = coarse
        self._audio_rate_in = fine_rate
        self._audio_taps = _lowpass_taps(cutoff, stop, fine_rate)

    def _audio_stages(self) -> List[_FirDecimator]:
        """The audio filters (with de-emphasis for broadcast FM) and the
        decimation to the intermediate rate, reset to a fresh stream.

        Built once per de-emphasis setting and reused."""
        cached = self._audio_cache.get(self._deemphasis)
        if cached is None:
            cached = self._audio_cache[self._deemphasis] = self._build_audio_stages()
        for stage in cached:
            stage.reset()
        return cached

    def _build_audio_stages(self) -> List[_FirDecimator]:
        stages = []
        if self._audio_coarse_factor > 1:
            stages.append(_FirDecimator(self._audio_coarse, self._audio_coarse_factor))
        taps = self._audio_taps
        if self._deemphasis:
            deemphasis = _deemphasis_taps(self._audio_rate_in)
            taps = (
                deemphasis
                if taps is None
                else np.convolve(taps, deemphasis).astype(np.float32)
            )
        fine_factor = self._audio_factor // self._audio_coarse_factor
        if taps is not None or fine_factor > 1:
            stages.append(_FirDecimator(taps, fine_factor))
        return stages

    def reset_demodulator(self) -> None:
        """Forget the audio state (after a gap); the channel filter keeps its."""
        self._prev: Optional[complex] = None
        self._level = 0.0  # AM carrier level
        self._bfo_phase = 0.0
        self._audio = self._audio_stages()
        if self._sideband is not None:
            self._sideband.reset()
        if self._resampler is not None:
            self._resampler.reset()
        self._dc_blocker.reset()
        self._audio_agc.reset()
        noise_reduction = self._options().noise_reduction
        if noise_reduction is not None:
            noise_reduction.reset_stream()

    def channelize(self, samples: np.ndarray) -> np.ndarray:
        """The tuned channel of an I/Q block, at ``channel_rate``."""
        # The VFO's channel to 0 Hz (nothing to do at the LO itself).
        out = self._nco.mix(np.asarray(samples, dtype=np.complex64))
        for stage in self._channel:
            out = stage.process(out)
        return out

    def demodulate(self, channel: np.ndarray, fm_deviation: float) -> np.ndarray:
        """Mono float32 audio in [-1, 1] at ``audio_rate`` (48 kHz)."""
        if len(channel) == 0:
            return np.empty(0, dtype=np.float32)
        if self.mode == "FM":
            broadcast = float(fm_deviation) >= _BROADCAST_DEVIATION_HZ
            if broadcast != self._deemphasis:
                self._deemphasis = broadcast
                self._audio = self._audio_stages()
                # Broadcast keeps its deepest bass (a 2 Hz corner still
                # removes a tuning offset within a few tenths of a second).
                self._dc_blocker = self._make_dc_blocker(
                    _FM_BROADCAST_DC_CORNER_HZ if broadcast else _FM_DC_CORNER_HZ
                )
            # Discriminator; the previous block's last sample keeps the phase
            # difference continuous. Scaled so +/- the deviation is full scale.
            if self._prev is not None:
                prior = np.concatenate(([self._prev], channel[:-1]))
            else:
                prior = np.concatenate((channel[:1], channel[:-1]))
            self._prev = complex(channel[-1])
            detected = np.angle(channel * np.conj(prior)).astype(np.float32)
            gain = self.channel_rate / (2.0 * np.pi * max(float(fm_deviation), 1.0))
            audio = self._to_audio_rate(detected * np.float32(gain))
            # A tuning error is a constant frequency offset: DC, removed.
            audio = self._dc_blocker.process(audio)
            if audio.size:
                over = np.count_nonzero(np.abs(audio) > 1.0)
                self.last_clip_fraction = float(over) / audio.size
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            self.pre_nr_audio = audio
            return self._reduce_noise(audio)
        options = self._options()
        if self.mode == "AM":
            audio = self._am_detect(channel, options.agc_speed)
            audio = self._dc_blocker.process(self._to_audio_rate(audio))
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            self.pre_nr_audio = audio
            return self._reduce_noise(audio)
        # USB / LSB / CW: the channel is already the sideband (or the CW
        # filter's passband); CW adds the beat note. Then the real part.
        narrow = channel
        if self.mode == "CW":
            pitch = min(max(float(options.cw_pitch_hz), 200.0), 2000.0)
            step = 2.0 * np.pi * pitch / self.channel_rate
            phase = self._bfo_phase + step * np.arange(len(narrow))
            self._bfo_phase = float(
                (self._bfo_phase + step * len(narrow)) % (2 * np.pi)
            )
            narrow = narrow * np.exp(1j * phase)
        audio = np.real(narrow).astype(np.float32)
        if self._resampler is not None:
            audio = self._resampler.process(audio)
        audio = self._dc_blocker.process(audio)
        self.pre_nr_audio = audio
        nr = options.noise_reduction
        if nr is not None:
            audio = np.asarray(nr.process(audio), dtype=np.float32)
        return self._agc(audio)

    def _am_detect(self, channel: np.ndarray, speed: str) -> np.ndarray:
        """AM envelope relative to the carrier (at the channel rate).

        The carrier level is tracked with the AGC speed's time constant and
        interpolated across the block (no steps between blocks), and the
        envelope is divided by it: the loudness follows the modulation
        depth, not the signal strength. With the AGC off it isn't divided,
        so the loudness follows the signal."""
        envelope = np.abs(channel).astype(np.float32)
        carrier = float(np.mean(envelope))
        seconds = len(channel) / max(self.channel_rate, 1.0)
        if carrier > self._level:
            tau = _AM_ATTACK_S  # a rising carrier is followed at once
        else:
            tau = _AM_CARRIER_TIME_S.get(speed, _AM_CARRIER_TIME_S["medium"])
        if self._level <= 0.0:
            start = end = carrier
        else:
            start = self._level
            end = start + (1.0 - np.exp(-seconds / tau)) * (carrier - start)
        self._level = float(end)
        if len(channel) > 1 and start != end:
            level = np.linspace(start, end, len(channel), dtype=np.float32)
        else:
            level = np.float32(end)
        if speed not in AGC_SPEEDS:  # AGC off
            return (envelope - level) * np.float32(_AM_FIXED_GAIN)
        return (envelope - level) / np.maximum(level, np.float32(1e-9))

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        nr = self._options().noise_reduction
        if nr is None:
            return audio
        out = np.asarray(nr.process(audio), dtype=np.float32)
        return np.clip(out, -1.0, 1.0)

    def _to_audio_rate(self, audio: np.ndarray) -> np.ndarray:
        """Audio filter, decimation and resampling to exactly 48 kHz."""
        for stage in self._audio:
            audio = stage.process(audio)
        if self._resampler is not None:
            audio = self._resampler.process(audio)
        return audio

    def _agc(self, audio: np.ndarray) -> np.ndarray:
        """Bring SSB/CW to a steady level: instant attack, a release that
        follows the AGC speed (see :class:`~.audio_chain.AudioAgc`; its
        floor, -80 dBFS, keeps an empty channel from being turned up into
        full-scale hiss)."""
        self._audio_agc.speed = self._options().agc_speed
        return self._audio_agc.process(audio)


class _Pacer:
    """Keeps the display loop in step with real time, as far as this
    computer can.

    Each display frame takes what the device delivered since the last one:
    every queued hardware transfer, or as many demo samples as the time
    since the last frame covers (:meth:`demo_request`), up to
    :meth:`read_limit`: ``_MAX_READ_S`` of signal, ``_PROBE_READ_S`` until
    frames are known to keep up.

    :meth:`measured` times whole frames (reading, filtering, demodulating)
    against the time their samples cover. While frames take more than
    ``_OVERLOAD_RATIO`` of it, this computer can't keep up (``overloaded``):
    each frame then reads one block, as it used to, so the window stays
    responsive, and as samples no longer arrive in real time, audio is
    muted with a hint. One-block frames are too short to time, so once
    every ``_OVERLOAD_PROBE_S`` a frame reads a full frame's worth again
    (a probe, timed like any frame; after a fast one the next frame probes
    too). Full frames resume below ``_RECOVER_RATIO``, so a passing hiccup
    (another program busy for a moment) doesn't mute the audio for good.
    Start (:meth:`restart`) finds out anew.
    """

    def __init__(self) -> None:
        self._last: Optional[float] = None
        self._ratios: Deque[float] = deque(maxlen=5)
        self.overloaded = False
        # While overloaded: when the next probe frame is due (None: not
        # scheduled yet), or probe on the next frame.
        self._next_probe: Optional[float] = None
        self._probe_now = False
        # Samples a paced source (a recording playing back) is owed but not
        # yet given: less than a block, or a little ahead (negative).
        self._owed = 0.0

    def reset(self) -> None:
        """Forget everything (new device or sample rate)."""
        self._last = None
        self._ratios.clear()
        self.overloaded = False
        self._next_probe = None
        self._probe_now = False
        self._owed = 0.0

    def restart(self) -> None:
        """Reading pauses (receiver stopped): the next demo read starts
        afresh, and whether frames keep up is found out anew."""
        self.reset()

    def read_limit(self, rate: float, now: Optional[float] = None) -> float:
        """The most samples one frame may take.

        While overloaded that is 0 (one block), except, given the monotonic
        time ``now``, for a probe frame: at least ``_TIMED_FRAME`` samples.
        """
        if self.overloaded:
            if now is not None and self._probe_due(now):
                return max(float(_TIMED_FRAME), rate * _PROBE_READ_S)
            return 0.0
        return rate * (_MAX_READ_S if self._keeps_up() else _PROBE_READ_S)

    def _probe_due(self, now: float) -> bool:
        if self._probe_now or (
            self._next_probe is not None and now >= self._next_probe
        ):
            self._probe_now = False
            self._next_probe = now + _OVERLOAD_PROBE_S
            return True
        if self._next_probe is None:
            self._next_probe = now + _OVERLOAD_PROBE_S
        return False

    def demo_request(self, rate: float, now: float) -> int:
        """Demo samples to read at monotonic time ``now``: the time since
        the previous read times the rate, at least one display block (and
        a probe frame at least ``_TIMED_FRAME``, so it is timed)."""
        last, self._last = self._last, now
        if last is None or rate <= 0:
            return DISPLAY_BLOCK
        wanted = round(rate * max(0.0, now - last))
        limit = self.read_limit(rate, now)
        smallest = _TIMED_FRAME if self.overloaded and limit > 0 else DISPLAY_BLOCK
        return int(max(smallest, min(wanted, limit)))

    def paced_request(self, rate: float, now: float) -> int:
        """Like :meth:`demo_request`, for a source that must not run ahead
        of real time (a recording playing back, ``rate`` times its speed):
        a frame owed less than a display block reads nothing (0) and the
        time carries over, so a source slower than a block a frame (a 48
        kHz recording, or half speed) keeps its own pace."""
        last, self._last = self._last, now
        if last is None or rate <= 0:
            self._owed = 0.0
            return DISPLAY_BLOCK
        owed = self._owed + rate * max(0.0, now - last)
        if owed < DISPLAY_BLOCK:
            self._owed = owed
            return 0
        limit = self.read_limit(rate, now)
        smallest = _TIMED_FRAME if self.overloaded and limit > 0 else DISPLAY_BLOCK
        count = int(max(smallest, min(owed, limit)))
        # What a frame couldn't take (a stall) is dropped, as for the demo;
        # a block read early is made up by the next frames.
        self._owed = min(owed - count, float(DISPLAY_BLOCK))
        return count

    def measured(self, count: int, rate: float, seconds: float) -> None:
        """A frame of ``count`` samples took ``seconds`` to read and process."""
        if count < _TIMED_FRAME or rate <= 0:
            return
        ratio = seconds * rate / count
        self._ratios.append(ratio)
        if self.overloaded and ratio < _RECOVER_RATIO:
            self._probe_now = True  # looks better: check again right away
        if len(self._ratios) < 3:
            return
        load = float(np.median(self._ratios))
        if load > _OVERLOAD_RATIO:
            if not self.overloaded:
                self._next_probe = None
                self._probe_now = False
            self.overloaded = True
        elif load < _RECOVER_RATIO:
            self.overloaded = False
            self._probe_now = False

    def _keeps_up(self) -> bool:
        """Frames are known to take under ``_OVERLOAD_RATIO`` of real time."""
        return (
            len(self._ratios) >= 3 and float(np.median(self._ratios)) <= _OVERLOAD_RATIO
        )


def _is_demo_device(device: Any) -> bool:
    """True for the simulated Demo Device, however it was opened (Device >
    Use Demo Device, or the Demo row of Device > Connect...)."""
    if device is None:
        return False
    from .device_dialog import MockDevice

    return isinstance(device, MockDevice)


def _driver_importable(package: str, name: str) -> bool:
    """True if ``from package import name`` works, checked without logging.

    Also False when the package is installed but its C library is missing
    (pyrtlsdr raises ImportError then; other bindings may raise OSError).
    """
    try:
        if importlib.util.find_spec(package) is None:
            return False
        module = importlib.import_module(package)
        if not hasattr(module, name):
            importlib.import_module(f"{package}.{name}")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Recording buffer
# ---------------------------------------------------------------------------

_CF32 = np.dtype("<c8")  # complex64, little-endian: the .cf32 layout
_RECORDING_PREFIX = "sdr-recording-"
_RECORDING_SUFFIX = ".cf32"
# A recording stops, with a warning, when the drive it is written to has
# less than this much free space left.
_MIN_FREE_BYTES = 500 * 1024 * 1024
_FREE_SPACE_CHECK_S = 0.5
_SAVE_CHUNK = 1 << 20  # samples converted and written per step when saving
_cleaned_folders: set = set()


def _recording_dir() -> str:
    """The folder recordings are captured into (created if needed): the
    user cache folder, which is on disk (a system temp folder is often in
    RAM), else the system temp folder."""
    candidates = []
    try:
        from PyQt6.QtCore import QStandardPaths

        cache = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.GenericCacheLocation
        )
        if cache:
            candidates.append(os.path.join(cache, "sdr-module", "recordings"))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"No user cache folder: {e}")
    candidates.append(os.path.join(tempfile.gettempdir(), "sdr-module-recordings"))
    for folder in candidates:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            continue
        if os.access(folder, os.W_OK):
            return folder
    return tempfile.gettempdir()


def _file_in_use(path: str) -> bool:
    """Whether a running session holds this recording file: it keeps it
    open and locked (POSIX). On Windows an open file can't be deleted, so
    the attempt to delete it tells."""
    if fcntl is None:
        return False
    try:
        with open(path, "rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        return True
    return False


def _remove_stale_recordings(folder: str) -> int:
    """Delete the temporary recordings of sessions that ended without
    cleaning up (a crash, a killed process); returns how many.

    Only this app's own files (``sdr-recording-*.cf32`` in its own folder)
    that no running session holds are touched.
    """
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if not (
            name.startswith(_RECORDING_PREFIX) and name.endswith(_RECORDING_SUFFIX)
        ):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or _file_in_use(path):
            continue
        try:
            os.remove(path)
        except OSError:  # in use (Windows) or not ours to delete
            continue
        removed += 1
        logger.info(f"Removed a stale temporary recording: {path}")
    return removed


def _close_and_delete(handle: Any, path: str) -> None:
    """Finalizer of a temporary recording: close and delete its file."""
    try:
        handle.close()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Could not close the temporary recording {path}: {e}")
    try:
        os.remove(path)
    except OSError as e:
        logger.debug(f"Could not delete the temporary recording {path}: {e}")


class _SampleStore:
    """The recording buffer's I/Q samples.

    A live recording streams to a temporary complex64 file (``.cf32``) as
    it is captured, so recording takes disk space rather than memory (19 MB
    a second at 2.4 MS/s, 160 MB at 20 MS/s); so does a file loaded by File >
    Convert Recording, a chunk at a time (:meth:`from_samples` holds samples
    in memory, for small ones). Either way the samples read back as the list
    of blocks they arrived in, like the list this replaces: ``len()`` is
    the number of blocks, and iterating or indexing gives each block (a
    read-only view of the file). :meth:`samples` gives them all and
    :meth:`chunks` a piece at a time.

    The temporary file stays open (and locked, on POSIX) until the store is
    discarded, which deletes it; so does the interpreter exiting, and
    :func:`_remove_stale_recordings` removes files a crash left behind.
    """

    def __init__(self, folder: Optional[str] = None):
        self._folder = folder
        self._path: Optional[str] = None
        self._file: Any = None
        self._finalizer: Any = None
        self._memory: Optional[np.ndarray] = None
        self._blocks: List[Tuple[int, int]] = []  # (first sample, length)
        self._count = 0
        self._map: Optional[np.ndarray] = None

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "_SampleStore":
        """A store holding ``samples`` in memory."""
        store = cls()
        data = np.asarray(samples)
        store._memory = data
        store._count = len(data)
        store._blocks = [(0, len(data))] if len(data) else []
        return store

    @property
    def sample_count(self) -> int:
        return self._count

    @property
    def nbytes(self) -> int:
        """Size of the samples as complex64."""
        return self._count * _CF32.itemsize

    @property
    def path(self) -> Optional[str]:
        """The file holding the samples (None in memory or empty)."""
        return self._path

    @property
    def folder(self) -> Optional[str]:
        """Where the samples are (or will be) written."""
        if self._path:
            return os.path.dirname(self._path)
        return self._folder

    def append(self, block: np.ndarray) -> None:
        """Add a block after the others. The file is created with the first
        one. Raises OSError when it can't be written (e.g. a full disk)."""
        data = np.ascontiguousarray(block, dtype=_CF32)
        if len(data) == 0:
            return
        if self._memory is not None:
            raise ValueError("an imported recording can't be added to")
        if self._file is None:
            self._create_file()
        self._file.write(data.data)
        self._blocks.append((self._count, len(data)))
        self._count += len(data)
        self._map = None

    def _create_file(self) -> None:
        folder = self._folder or _recording_dir()
        os.makedirs(folder, exist_ok=True)
        if folder not in _cleaned_folders:
            _cleaned_folders.add(folder)
            _remove_stale_recordings(folder)
        unique = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        path = os.path.join(folder, f"{_RECORDING_PREFIX}{unique}{_RECORDING_SUFFIX}")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags, 0o666)  # the usual permissions, once saved
        handle = os.fdopen(fd, "wb", buffering=1 << 20)
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:  # pragma: no cover - e.g. a filesystem without locks
                pass
        self._folder, self._path, self._file = folder, path, handle
        self._finalizer = weakref.finalize(self, _close_and_delete, handle, path)

    def finish(self) -> None:
        """Nothing more will be added for now: write out what is buffered."""
        if self._file is not None:
            self._file.flush()

    def samples(self) -> np.ndarray:
        """Every sample: the loaded array or a read-only map of the file."""
        if self._memory is not None:
            return self._memory
        if self._count == 0 or self._path is None:
            return np.empty(0, dtype=_CF32)
        if self._map is None:
            self.finish()
            self._map = np.memmap(
                self._path, dtype=_CF32, mode="r", shape=(self._count,)
            )
        return self._map

    def chunks(self, size: int = _SAVE_CHUNK):
        """The samples, ``size`` at a time (each read from disk as needed)."""
        data = self.samples()
        for start in range(0, len(data), size):
            yield np.asarray(data[start : start + size])

    def save_raw(self, filename: str, move: bool = False) -> None:
        """Put the samples in ``filename`` as raw complex64 (``.cf32``),
        the layout they are captured in, so no conversion is needed.

        With ``move`` (nothing more will be recorded to this store) the
        temporary file itself is moved there, instantly on the same drive,
        and the store reads from ``filename`` from then on. Otherwise, or
        when it can't be moved (another drive), it is copied.
        """
        if self._path is None:
            raise ValueError("the samples are not in a file")
        self.finish()
        if os.path.exists(filename) and os.path.samefile(filename, self._path):
            return
        if move and self._finalizer is not None and self._finalizer.alive:
            self._map = None
            try:
                os.replace(self._path, filename)
            except OSError as e:  # another drive, or open elsewhere (Windows)
                logger.debug(f"Copying the recording instead of moving it: {e}")
            else:
                # The file is the user's now: never delete it.
                self._finalizer.detach()
                self._finalizer = None
                self._file.close()
                self._file = None
                self._path = os.path.abspath(filename)
                return
        shutil.copyfile(self._path, filename)

    def discard(self) -> None:
        """Forget the samples, deleting the temporary file (not a saved one)."""
        self._map = None
        if self._finalizer is not None:
            self._finalizer()  # closes and deletes the temporary file
        elif self._file is not None:
            self._file.close()
        self._finalizer = self._file = self._path = None
        self._memory = None
        self._blocks = []
        self._count = 0

    # The list of blocks it replaces -----------------------------------------

    def __len__(self) -> int:
        return len(self._blocks)

    def __bool__(self) -> bool:
        return bool(self._blocks)

    def __iter__(self):
        data = self.samples()
        for start, length in self._blocks:
            yield data[start : start + length]

    def __getitem__(self, index: int) -> np.ndarray:
        start, length = self._blocks[index]
        return self.samples()[start : start + length]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return len(other) == len(self) and all(
                np.array_equal(a, b) for a, b in zip(self, other, strict=True)
            )
        return other is self

    __hash__ = object.__hash__


def _clock_text(seconds: float) -> str:
    """``12.3 -> "00:12"``, ``3725 -> "1:02:05"`` (recording lengths)."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Status bar message label
# ---------------------------------------------------------------------------


class _ElidedLabel(QLabel if HAS_PYQT6 else object):
    """A one-line label that ends in "…" instead of being cut off.

    ``text()`` returns the full text; only the painted text is shortened to
    the label's current width. It never asks for width of its own, so a long
    message can't raise the window's minimum width.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full_text = str(text or "")
        self._elide()

    def text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().changeEvent(event)
        # The stylesheet (theme switch) can change the font and padding.
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()

    def _elide(self) -> None:
        width = self.contentsRect().width()
        shown = self._full_text
        if width > 0 and self.fontMetrics().horizontalAdvance(shown) > width:
            shown = self.fontMetrics().elidedText(
                shown, Qt.TextElideMode.ElideRight, width
            )
        super().setText(shown)


# ---------------------------------------------------------------------------
# Info tab
# ---------------------------------------------------------------------------


class InfoPanel(QWidget if HAS_PYQT6 else object):
    """Read-only live summary of the receiver, shown in the Info tab."""

    # (group title, ((key, label, tooltip), ...))
    SECTIONS: Tuple[Tuple[str, Tuple[Tuple[str, str, str], ...]], ...] = (
        (
            "Receiver",
            (
                ("device", "Device", "The SDR samples are read from."),
                ("state", "State", "Whether the receiver is running."),
                (
                    "frequency",
                    "Tuned",
                    "The tuned frequency: what is demodulated, decoded and "
                    "measured (FREQ in the toolbar).",
                ),
                (
                    "lo",
                    "Center (LO)",
                    get_short_tip("center_frequency")
                    + " The middle of the spectrum and waterfall; recordings "
                    "are labelled with it.",
                ),
                ("sample_rate", "Sample rate", get_short_tip("sample_rate")),
                (
                    "span",
                    "Span",
                    "Width of the spectrum shown; equals the complex sample rate.",
                ),
                (
                    "rbw",
                    "RBW",
                    get_short_tip("resolution_bandwidth")
                    + " Choose the FFT size and window in the spectrum's ⋯ menu.",
                ),
                ("demod", "Demodulation", "Selected in the Demodulation panel."),
                (
                    "bandwidth",
                    "Bandwidth",
                    "Width of the tuned channel that is demodulated and measured "
                    "(Receiver > Bandwidth); in USB, LSB and CW the filter "
                    "width, with the audio passband it gives.",
                ),
                ("gain", "Gain", get_short_tip("gain")),
                (
                    "frontend",
                    "Front end",
                    "The device's frequency correction (ppm), direct sampling "
                    "(HF) and bias tee: Receiver group.",
                ),
                (
                    "squelch",
                    "Squelch",
                    "Audio is muted while the channel is below this level "
                    "(Demodulation > Squelch, or drag it on the level meter).",
                ),
                (
                    "level",
                    "Channel level",
                    "Power of the tuned channel (what is demodulated), in dBFS "
                    "(0 dBFS = full scale), and how far it is above the noise.",
                ),
            ),
        ),
        (
            "Audio && Recording",
            (
                (
                    "audio",
                    "Audio output",
                    "Toggle with the Audio button in the toolbar or Radio > "
                    "Audio Output.",
                ),
                (
                    "recording",
                    "Recording",
                    "Toggle with the Record button or Ctrl+Shift+R.",
                ),
                (
                    "buffer",
                    "Buffer",
                    "The recorded (or imported) I/Q samples. A recording is "
                    "kept in a temporary file until File > Save Recording "
                    "writes it where you choose.",
                ),
            ),
        ),
        (
            "Application",
            (
                ("version", "Version", "SDR Module version."),
                ("qt", "Qt / PyQt", "Versions of the GUI toolkit."),
                ("theme", "Theme", "Change with View > Theme or Ctrl+T."),
            ),
        ),
    )

    # (keys, what they do): shown as keycap rows like the first-run wizard's
    # Quick Tips.
    TIPS: Tuple[Tuple[str, str], ...] = (
        ("Click", "Tune to a signal on the spectrum or waterfall."),
        (
            "Wheel",
            "Scroll over the plots to tune by the step, or over a digit of the "
            "FREQ readout to tune that digit.",
        ),
        ("Ctrl+L", "Type a frequency (or click the FREQ readout): 145.8, 7074k."),
        (
            "← / →",
            "Tune by the step (STEP in the toolbar). Add Shift for ten steps, "
            "Ctrl for 1 MHz.",
        ),
        ("[ / ]", "Smaller / larger tuning step."),
        ("Alt+← / →", "Back / forward through the frequencies you tuned."),
        ("Space", "Start or stop the receiver."),
        ("F6", "Put the keyboard focus on the spectrum (Esc also returns there)."),
        ("Ctrl+B", "Bookmark the current frequency."),
        ("F1", "Show all keyboard shortcuts."),
    )

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self._values: Dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        for title, rows in self.SECTIONS:
            group = QGroupBox(title)
            form = QFormLayout(group)
            form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            # Top-aligned, so a label lines up with the first line of a
            # wrapped value.
            form.setLabelAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(4)
            for key, label_text, tip in rows:
                label = QLabel(label_text)
                set_role(label, "muted")
                value = QLabel(_NO_VALUE)
                value.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
                )
                value.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                # Long values (device names) wrap instead of forcing a
                # horizontal scrollbar in a narrow right column.
                value.setWordWrap(True)
                label.setToolTip(tip)
                value.setToolTip(tip)
                form.addRow(label, value)
                self._values[key] = value
            layout.addWidget(group)

        tips_group = QGroupBox("Quick Tips")
        tips = QGridLayout(tips_group)
        tips.setHorizontalSpacing(10)
        tips.setVerticalSpacing(6)
        tips.setColumnStretch(1, 1)
        top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        keycaps = []
        for row, (keys, text) in enumerate(self.TIPS):
            key_label = QLabel(keys)
            set_role(key_label, "keycap")
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tips.addWidget(key_label, row, 0, top_left)
            keycaps.append(key_label)
            text_label = QLabel(text)
            text_label.setWordWrap(True)
            set_role(text_label, "hint")
            tips.addWidget(text_label, row, 1, Qt.AlignmentFlag.AlignTop)
        # Equal-width keycaps keep the descriptions in one clean column.
        width = max(k.sizeHint().width() for k in keycaps)
        for key_label in keycaps:
            key_label.setFixedWidth(width)
        layout.addWidget(tips_group)
        layout.addStretch(1)

    def set_value(self, key: str, text: str, tone: Optional[str] = None) -> None:
        """Show ``text`` for row ``key`` (optionally colored with a tone)."""
        label = self._values.get(key)
        if label is None:
            return
        if label.text() != text:
            label.setText(text)
        set_tone(label, tone)

    def value(self, key: str) -> str:
        """Current text of row ``key`` (empty if unknown)."""
        label = self._values.get(key)
        return label.text() if label is not None else ""


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class SDRMainWindow(QMainWindow if HAS_PYQT6 else object):
    """
    Main SDR application window.

    Contains:
    - Spectrum analyzer display
    - Waterfall display
    - Control panel (frequency, gain, bandwidth)
    - Protocol decoder output and other tool panels
    - Recording controls
    """

    # How F1 names the arrow keys' steps (by modifier prefix): they follow
    # the tuning step.
    _TUNE_STEP_WORDS = {"": "one tuning step", "Shift+": "ten tuning steps"}

    @property
    def _TUNE_STEPS_HZ(self) -> Dict[Any, float]:  # noqa: N802 - existing name
        """How far Left/Right tune, by modifier: one tuning step, ten with
        Shift, 1 MHz with Ctrl. (F1 lists the arrow keys from this.)"""
        step = float(getattr(self, "_step_hz", 10e3))
        return {
            Qt.KeyboardModifier.NoModifier: step,
            Qt.KeyboardModifier.ShiftModifier: 10.0 * step,
            Qt.KeyboardModifier.ControlModifier: 1e6,
        }

    # Toolbar button captions.
    _START_TEXT = "▶  Start"
    _STOP_TEXT = "■  Stop"
    _RECORD_TEXT = "●  Record"
    # (A leading space: Qt leaves almost no gap after a button's icon.)
    _AUDIO_ON_TEXT = " Audio"
    _AUDIO_OFF_TEXT = " Muted"

    def __init__(self, parent=None, demo_mode: bool = False):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required for the GUI")

        super().__init__(parent)

        self._device = None
        self._is_running = False
        self._recording = False
        # The recording buffer: captured samples stream to a temporary file
        # (in _recording_folder, resolved on first use), imported ones are
        # held in memory.
        self._recording_folder: Optional[str] = None
        self._samples_buffer = _SampleStore()
        self._next_space_check = 0.0
        # What the buffer holds: (center frequency, sample rate) of the
        # capture or loaded file (None where unknown), and whether it holds
        # live samples that haven't been saved yet.
        self._buffer_meta: Tuple[Optional[float], Optional[float]] = (None, None)
        self._buffer_unsaved = False
        # The buffer holds a converted file (its name) rather than a take,
        # and the file the last take (I/Q or audio) was saved to.
        self._buffer_source = ""
        self._last_saved = ""
        # Record Audio: the take being written (None while not recording).
        self._audio_take: Optional[AudioTake] = None
        # When the take in the buffer started: UTC (ISO, for its metadata)
        # and local time (for its file name).
        self._take_started = ""
        self._take_stamp = ""
        # A new recording replaces the buffer only once samples arrive, so an
        # armed recording that captures nothing keeps the previous one.
        self._capture_pending = False
        self._retune_warned = False
        self._demo_mode = False  # set by _start_demo_mode()
        self._radio_tuner = None  # Pop-out radio tuner window
        self._transmit_text = None  # Tools > Transmit Text window (T14)
        # The tuner's seek and STEREO lamp (T7): blocks processed since the
        # last retune, whether the last one opened the squelch, and the
        # 19 kHz pilot detector (built while the tuner is open).
        self._tuner_blocks = 0
        self._tuner_signal = False
        self._tuner_centered = False
        self._pilot: Optional[StereoPilotDetector] = None
        self._squelch_db = -80.0  # the manual squelch threshold (dBFS)
        # What happens after the channel filter: the squelch (hysteresis,
        # hang time, Auto), the channel's noise floor, the audio options
        # the receiver chain reads, the fades that keep the speaker free
        # of clicks, the CTCSS tone squelch and the I/Q clean-up.
        self._squelch = SquelchGate(self._squelch_db)
        self._level_average = LevelAverager()
        self._noise_floor = NoiseFloorTracker()
        self._audio_options = AudioOptions()
        self._audio_join = AudioJoin(_AUDIO_RATE)
        self._join_chain: Optional["_ReceiverChain"] = None
        self._join_deviation = 0.0
        self._join_nr: Any = None  # the noise reduction the last block had
        self._tone_blocked = False  # the tone squelch is muting a signal
        self._demod_running = False
        self._tone_squelch = ToneSquelch(_AUDIO_RATE)
        self._tone_channel: Optional[float] = None  # where the tone was heard
        self._iq_conditioner = IqConditioner()
        self._clip_s = 0.0
        self._clip_warn_due = 0.0
        # The Scanner tab mutes the speaker while it moves between channels,
        # and has a bookmark's CTCSS tone checked even with the speaker off.
        self._scan_muted = False
        self._scan_checks_tone = False
        # Cached analysis window for the measurement spectrum (built lazily
        # to match the sample-block length): a fixed 2048-point Hann FFT of
        # the newest samples, whatever the plots show, so the noise floor
        # (Auto squelch), the S-meter's SNR and LEVEL never depend on the
        # display settings.
        self._spectrum_window: Optional[np.ndarray] = None
        self._spectrum_window_gain = 1.0
        # What the plots show: every sample of each frame, at the FFT size
        # and window chosen in the spectrum's menu, and the (LO, rate) the
        # samples carried over between frames belong to.
        self._frame_spectrum = FrameSpectrumAnalyzer()
        self._frame_axis: Optional[Tuple[float, float]] = None
        # After a hardware retune the plots skip these samples and until
        # then (they were taken on the old frequency; see _retune_lo).
        self._plot_skip_samples = 0
        self._plot_skip_until = 0.0
        # The waterfall rows' clock and when it last moved (see _plot_time).
        self._plot_clock = 0.0
        self._plot_clock_wall: Optional[float] = None
        # The part of the span both plots show (View > Zoom).
        self._plot_view = PlotView()
        self._agc_enabled = False
        self._recording_bytes = 0  # for free-space display
        self._recording_paused = False  # Pause in the Recording panel
        # Recording clock: seconds captured so far plus the start of the
        # current capturing stretch (None while paused or not receiving).
        self._rec_accum = 0.0
        self._rec_since: Optional[float] = None
        self._last_peak_db: Optional[float] = None
        # Hot-plug: devices seen by the last poll (None until the first one),
        # and the driver classes whose packages import (None until probed).
        self._known_devices: Optional[Dict[str, str]] = None
        self._hardware_classes: Optional[List[Any]] = None
        # Sample rate asked for on the command line (--sample-rate): used by
        # the demo device and preselected in Device > Connect.
        self._preferred_rate: Optional[float] = None
        # Channel filter + demodulator for the tuned channel (rebuilt when the
        # sample rate, mode or bandwidth changes). Its state carries between
        # I/Q blocks, so the audio (and the SSTV decoder's timing) is
        # continuous across block boundaries.
        self._rx_chain: Optional[_ReceiverChain] = None
        # (monotonic time, samples) of recent blocks: whether the device
        # delivers samples in real time (audio needs it).
        self._rx_history: Deque[Tuple[float, int]] = deque()
        self._realtime_hint_shown = False
        self._realtime_hint_text = ""
        self._pacer = _Pacer()  # how much each display frame reads
        self._entry_returns_focus = False  # Ctrl+L: Enter goes back to the plots
        self._restoring = False  # True while _restore_state applies settings
        self._layout_initialized = False
        self._splitters_restored = False
        self._panel_pages: Dict[str, QWidget] = {}
        self._panel_owners: Dict[str, QTabWidget] = {}
        self._panel_specs: List[Tuple[str, str, str]] = []

        # Live protocol decoder driven by the Decoder panel (None = "Off", no
        # active decoder). Rebuilt when the panel's protocol changes.
        self._decoder: Any = None
        self._decoder_protocol: Any = None
        self._decoder_rate = 0.0  # sample rate the live decoder was built for
        # Messages decoded while the Decoder tab was out of sight (its title
        # counts them), the background RDS of the tuned broadcast station
        # (Radio > RDS Station Names) and what its badge shows.
        self._decoder_unread = 0
        self._rds_listener = RdsListener()
        self._rds_shown: Tuple[str, str] = ("", "")

        # Persisted settings (frequency, gain, theme, bookmarks live here)
        self._settings = GuiSettings()

        # Tuning (see set_frequency). The LO is the device's center (the
        # middle of the plots); the tuned frequency (the VFO, which the
        # Frequency field shows) is what is demodulated, anywhere in the
        # span. _tuned_hz is the last frequency tuned, to go back to when a
        # new one is refused. _lo_lock fixes the LO: (center, span, why),
        # e.g. while a recording plays back.
        self._lo_hz = float(DEFAULT_FREQUENCY_HZ)
        self._tuned_hz = float(DEFAULT_FREQUENCY_HZ)
        self._lo_lock: Optional[Tuple[float, float, str]] = None
        # The tuning step and the mode it was chosen for (tuning/step/<key>).
        self._step_hz = 10e3
        self._step_key = ""
        # Typed text the Frequency field refused (keep the focus there).
        self._entry_rejected = False
        # Back/forward and Recent Frequencies (tuning/recent): a frequency
        # joins them once the receiver stays on it (_note_tuned).
        self._tuning_history = TuningHistory(
            TuningHistory.entries_from_json(self._settings.get_str("tuning/recent", ""))
        )
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.timeout.connect(self._commit_history_entry)
        # SDRApplication applies the persisted theme before the window exists;
        # mirror whatever is actually on screen.
        self._theme = current_theme()

        # Audio output: on by default, so a signal is heard as soon as the
        # receiver runs. Why it can't play (no output device), if it can't.
        self._audio = AudioSink()
        self._audio_enabled = self._settings.get_bool("audio_enabled", True)
        self._audio_problem: Optional[str] = None
        self._audio.set_volume(self._saved_volume() / 100.0)
        # The output chosen in Tools > Preferences (the default if none).
        self._audio.set_output_device(
            self._settings.get_str(KEY_AUDIO_OUTPUT, ""),
            self._settings.get_str(KEY_AUDIO_OUTPUT_NAME, ""),
        )

        self._setup_ui()
        # Transmitting (Send ID Now, Transmit Text): the transmit engine,
        # the ON AIR badge and the Ham ID panel's TX side (T7).
        self._tx_station = StationTx(self)
        # Device > Second Receiver... (T13): a second receiver window, whose
        # Listen choice can mute this one. (Imported here: it reuses this
        # module's receiver chain.)
        from .second_receiver import SecondReceiverHost

        self._second_rx = SecondReceiverHost(self)
        # The Receiver group's device settings: gain, sample rate, front end
        # (T11).
        self._device_controls = DeviceControls(self)
        self._setup_menus()
        self._setup_toolbar()
        self._setup_receiver_bar()
        self._setup_statusbar()
        self._setup_decoder_chip()
        self._setup_timers()

        # Connect signals
        self._connect_signals()
        theme_notifier().theme_changed.connect(self._on_theme_changed)

        # Restore persisted values
        self._restore_state()
        self._refresh_audio_output(announce=False)
        self._update_passband()
        self._refresh_state_ui()

        # Auto-start demo mode
        if demo_mode:
            self._start_demo_mode()

        # First-run wizard
        if self._settings.is_first_run():
            self._run_first_run_wizard()

        # Start with the plots focused so Space and the arrow keys work.
        self._spectrum.setFocus()

        logger.info("Main window initialized")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("SDR Module")
        # Small enough for 1366x768 laptops; panels scroll instead of crushing.
        self.setMinimumSize(1024, 640)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(0)

        # Main horizontal splitter: displays | controls
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self._main_splitter)

        # Left side: spectrum over waterfall
        self._display_splitter = QSplitter(Qt.Orientation.Vertical)
        self._display_splitter.setChildrenCollapsible(False)
        self._spectrum = SpectrumWidget()
        self._waterfall = WaterfallWidget()
        for plot in (self._spectrum, self._waterfall):
            # Tab, a click, F6 or Esc focuses a plot, so Space and Left/Right
            # work from the keyboard too (see keyPressEvent).
            plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._display_splitter.addWidget(plot)
        self._display_splitter.setStretchFactor(0, 2)
        self._display_splitter.setStretchFactor(1, 3)
        # The plots, with the playback strip under them while a recording
        # plays (File > Open Recording).
        display_column = QWidget()
        column = QVBoxLayout(display_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(self._display_splitter, 1)
        self._playback_bar = PlaybackBar()
        column.addWidget(self._playback_bar)
        self._playback = PlaybackController(self, self._playback_bar)
        self._main_splitter.addWidget(display_column)

        # Right side: control panel over the tabbed tool panels
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._control_panel = ControlPanel()
        self._right_splitter.addWidget(self._scrollable(self._control_panel))
        self._right_tabs = self._build_panel_tabs()
        self._right_splitter.addWidget(self._right_tabs)
        # The Signal tab's work and Radio > Track Signal (T10).
        self._signal_tools = SignalTools(self, self._signal_panel)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 1)
        self._main_splitter.addWidget(self._right_splitter)

        # Extra window width goes to the displays, not the controls.
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)

    @staticmethod
    def _text_width(button: "QPushButton", *texts: str) -> int:
        """Width that fits any of ``texts`` in bold plus the button padding,
        so a toolbar button doesn't change size when its caption changes."""
        font = QFont(button.font())
        font.setBold(True)
        metrics = QFontMetrics(font)
        return max(metrics.horizontalAdvance(t) for t in texts) + 36

    @staticmethod
    def _scrollable(widget: "QWidget") -> "QWidget":
        """Return ``widget`` inside a frameless, resizable scroll area.

        A widget that already scrolls internally is returned unchanged so it
        is never wrapped twice.
        """
        if isinstance(widget, QScrollArea) or widget.findChild(QScrollArea):
            return widget
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        area.setWidget(widget)
        return area

    def _build_panel_tabs(self) -> "QTabWidget":
        """Create the right-hand tab widget holding the tool panels."""
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        # The theme gives this bar slightly tighter tabs (T10), so all six
        # top-level tabs (with Signal) fit the column at 1024 px.
        tabs.tabBar().setObjectName("toolTabs")

        self._decoder_panel = DecoderPanel()
        self._add_panel(
            tabs,
            "decoder",
            self._decoder_panel,
            "Decoder",
            "Live protocol decoder output (POCSAG, FLEX, ADS-B, ACARS, "
            "AX.25/APRS, RDS, Morse code)",
        )

        self._bookmarks_panel = BookmarksPanel()
        self._add_panel(
            tabs,
            "bookmarks",
            self._bookmarks_panel,
            "Bookmarks",
            "Saved frequencies. Double-click one to tune; Ctrl+B adds the "
            "current frequency.",
        )

        # Optional ham radio panels, grouped under one tab so the top-level
        # tab bar fits a 360 px column without scroll arrows hiding tabs.
        if HAS_HAM_RADIO:
            self._ham_tabs = QTabWidget()
            self._ham_tabs.setDocumentMode(True)
            self._ham_tabs.setUsesScrollButtons(True)

            self._signal_meter_panel = SignalMeterPanel()
            self._add_panel(
                self._ham_tabs,
                "s_meter",
                self._signal_meter_panel,
                "S-Meter",
                "Signal strength in S-units with RST and signal report",
            )

            self._callsign_panel = CallsignPanel()
            self._add_panel(
                self._ham_tabs,
                "ham_id",
                self._callsign_panel,
                "Ham ID",
                "Station identification with your callsign (CW ID, HackRF only)",
            )

            self._sstv_panel = SSTVPanel()
            self._add_panel(
                self._ham_tabs,
                "sstv",
                self._sstv_panel,
                "SSTV",
                "Slow-scan TV image viewer, e.g. for ISS SSTV events",
            )

            self._qrp_panel = QRPPanel()
            self._add_panel(
                self._ham_tabs,
                "qrp",
                self._qrp_panel,
                "QRP",
                "Low-power (QRP) transmit power tools and compliance check",
            )

            index = tabs.addTab(self._ham_tabs, "Ham Radio")
            tabs.setTabToolTip(index, "S-meter, station ID, SSTV and QRP tools")

        self._info_panel = InfoPanel()
        self._add_panel(
            tabs,
            "info",
            self._info_panel,
            "Info",
            "Live receiver status, versions and quick tips",
        )

        # Scanner (T8): its panel spec comes after the others (Ctrl+8), its
        # tab before Info.
        self._scanner_panel = ScannerPanel(settings=self._settings)
        self._add_panel(
            tabs,
            "scanner",
            self._scanner_panel,
            "Scanner",
            "Sweep a band, stop on signals or scan your bookmarks, while you "
            "keep listening (Ctrl+F)",
        )
        tabs.tabBar().moveTab(
            tabs.indexOf(self._panel_pages["scanner"]),
            tabs.indexOf(self._panel_pages["info"]),
        )

        # Signal (T10): its panel spec after the Scanner's (Ctrl+9), its tab
        # before Info.
        self._signal_panel = SignalPanel()
        # Scrolling as a whole like the other panels (its Identify page
        # scrolls on its own too, which _scrollable would take for enough):
        # the column's split may leave it less than its pages need.
        signal_area = QScrollArea()
        signal_area.setWidgetResizable(True)
        signal_area.setFrameShape(QFrame.Shape.NoFrame)
        signal_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        signal_area.setWidget(self._signal_panel)
        self._add_panel(
            tabs,
            "signal",
            signal_area,
            "Signal",
            "Identify the tuned signal (Ctrl+I), list the signals in view, "
            "look at the channel on a scope and log the bursts in the span",
        )
        tabs.tabBar().moveTab(
            tabs.indexOf(self._panel_pages["signal"]),
            tabs.indexOf(self._panel_pages["info"]),
        )
        return tabs

    # View > Panels menu text (mnemonics) by panel key; the tab keeps its title.
    _PANEL_MENU_TEXT = {
        "decoder": "&Decoder",
        "bookmarks": "&Bookmarks",
        "s_meter": "S-&Meter",
        "ham_id": "&Ham ID",
        "sstv": "&SSTV",
        "qrp": "&QRP",
        "info": "&Info",
        "scanner": "S&canner",
        "signal": "Si&gnal",
    }

    def _add_panel(
        self, tabs: "QTabWidget", key: str, widget: "QWidget", title: str, tip: str
    ) -> None:
        """Add one tool panel as a (scrollable) tab of ``tabs``."""
        page = self._scrollable(widget)
        index = tabs.addTab(page, title)
        tabs.setTabToolTip(index, tip)
        self._panel_pages[key] = page
        self._panel_owners[key] = tabs
        self._panel_specs.append((key, title, tip))

    def _add_action(
        self,
        menu: Any,
        text: str,
        slot: Any = None,
        shortcut: Any = None,
        tip: str = "",
        checkable: bool = False,
        checked: bool = False,
    ) -> "QAction":
        """Create a menu action. Menu actions are the single owner of their
        keyboard shortcut, so no key is ever bound twice (which Qt treats as
        ambiguous and then fires neither)."""
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
            if slot is not None:
                action.toggled.connect(slot)
        elif slot is not None:
            action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # ---- File ----
        file_menu = menubar.addMenu("&File")
        self._open_recording_action = self._add_action(
            file_menu,
            "&Open Recording...",
            self._open_recording,
            QKeySequence.StandardKey.Open,
            "Play an I/Q recording through the receiver: spectrum, waterfall, "
            "audio and decoders, with a position slider, Loop and speed",
        )
        self._close_recording_action = self._add_action(
            file_menu,
            "C&lose Recording",
            lambda: self._playback.close(restore=True),
            tip="Stop playing the recording and go back to the device used before",
        )
        self._close_recording_action.setEnabled(False)
        self._add_action(
            file_menu,
            "Recording &Info...",
            self._show_recording_info,
            tip="Show what an I/Q file holds: format, sample rate, center "
            "frequency, length, and whether its samples look right",
        )
        self._add_action(
            file_menu,
            "&Decode Recording...",
            self._decode_recording,
            tip="Run the Decoder panel's protocol over a whole I/Q file, much "
            "faster than real time; the messages go to the Decoder tab",
        )
        self._add_action(
            file_menu,
            "Find Si&gnals in Recording...",
            lambda: self._signal_tools.find_in_recording(),
            tip="List the signals in a whole I/Q file (frequency, level, SNR, "
            "when heard) in the Signal tab; double-click one to play it there",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Con&vert Recording...",
            self._convert_recording,
            "Ctrl+Shift+O",
            "Load an I/Q file (any format, including rtl_sdr and HackRF "
            "captures) to save it in another format; the receiver isn't "
            "retuned",
        )
        self._save_recording_action = self._add_action(
            file_menu,
            "&Save Recording...",
            self._save_recording,
            QKeySequence.StandardKey.Save,
            "Write the recorded (or converted) I/Q samples to a file",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "&Import Channels (CHIRP CSV)...",
            self._import_channels_csv,
            tip="Load memory channels from a CHIRP-compatible CSV file",
        )
        self._add_action(
            file_menu,
            "&Export Channels (CHIRP CSV)...",
            self._export_channels_csv,
            tip="Save the bookmarks as a CHIRP-compatible CSV file",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Save S&creenshot...",
            self._save_screenshot,
            "Ctrl+P",
            "Save a PNG image of the window",
        )
        file_menu.addSeparator()
        # Some platforms (e.g. Windows) have no standard Quit key; use Ctrl+Q.
        quit_keys = QKeySequence.keyBindings(QKeySequence.StandardKey.Quit)
        self._add_action(
            file_menu,
            "E&xit",
            self.close,
            quit_keys[0] if quit_keys else "Ctrl+Q",
            "Close SDR Module",
        )

        # ---- Device ----
        device_menu = menubar.addMenu("&Device")
        self._connect_action = self._add_action(
            device_menu,
            "&Connect...",
            self._show_device_dialog,
            tip="Choose and open an RTL-SDR or HackRF One",
        )
        self._disconnect_action = self._add_action(
            device_menu,
            "&Disconnect",
            self._disconnect_device,
            tip="Stop receiving and close the current device",
        )
        device_menu.addSeparator()
        # The sample rate (span), changed live; the status bar's rate pops
        # up the same menu.
        device_menu.addMenu(self._device_controls.rate_menu)
        device_menu.addSeparator()
        self._demo_action = self._add_action(
            device_menu,
            "Use De&mo Device",
            tip="Explore the app with simulated signals (no hardware needed); "
            "it replaces a connected device",
            checkable=True,
        )
        self._demo_action.triggered.connect(lambda _c=False: self._use_demo_device())
        self._add_action(
            device_menu,
            "&Scan for Devices",
            self._refresh_devices,
            tip="List the SDR devices currently plugged in",
        )
        device_menu.addSeparator()
        self._second_rx.add_menu_action(device_menu)  # Second Receiver... (T13)

        # ---- Radio ----
        radio_menu = menubar.addMenu("&Radio")
        # Space is handled in keyPressEvent (so focused widgets keep it); the
        # "\tSpace" suffix only shows it in the menu's shortcut column.
        self._start_action = self._add_action(
            radio_menu,
            "&Start Receiving\tSpace",
            self._toggle_acquisition,
            tip="Start or stop receiving",
        )
        self._record_action = QAction("&Record I/Q", self)
        self._record_action.setCheckable(True)
        self._record_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._record_action.setStatusTip(
            "Capture raw I/Q samples (to a temporary file until you save them)"
        )
        self._record_action.triggered.connect(self._toggle_recording)
        radio_menu.addAction(self._record_action)
        self._record_audio_action = self._add_action(
            radio_menu,
            "Record &Audio",
            self._toggle_audio_recording,
            "Ctrl+Shift+A",
            "Record what you hear (the demodulated audio) to a WAV file in the "
            "recordings folder",
            checkable=True,
        )
        radio_menu.addSeparator()
        self._audio_action = self._add_action(
            radio_menu,
            "&Audio Output",
            self._set_audio_enabled,
            tip="Play the tuned signal (AM, FM, SSB or CW) through your "
            "speakers while it is above squelch",
            checkable=True,
            checked=self._audio_enabled and self._audio.available,
        )
        self._audio_action.setEnabled(self._audio.available)
        radio_menu.addSeparator()
        bands_menu = radio_menu.addMenu("Band &Presets")
        for preset in BAND_PRESETS:
            freq_text = format_frequency(preset.frequency_hz)
            act = QAction(f"{preset.label}\t{freq_text}", self)
            act.setStatusTip(
                f"Tune to {freq_text} in {preset.mode} mode, "
                f"{preset.bandwidth} bandwidth"
            )
            act.triggered.connect(
                lambda _c=False, p=preset: self._apply_band_preset(
                    p.frequency_hz, p.mode, p.label
                )
            )
            bands_menu.addAction(act)
        self._add_action(
            radio_menu,
            "Enter &Frequency",
            self._focus_frequency_entry,
            "Ctrl+L",
            "Jump to the Frequency field to type a new frequency",
        )
        self._add_action(
            radio_menu,
            "&Bookmark Current Frequency",
            self._bookmark_current_frequency,
            "Ctrl+B",
            "Add the current frequency to the Bookmarks panel",
        )
        radio_menu.addSeparator()
        self._back_action = self._add_action(
            radio_menu,
            "Tune Ba&ck",
            self._tune_back,
            "Alt+Left",
            "Go back to the frequency (and mode) you were on before",
        )
        self._forward_action = self._add_action(
            radio_menu,
            "Tune For&ward",
            self._tune_forward,
            "Alt+Right",
            "Go forward again after Tune Back",
        )
        self._recent_menu = radio_menu.addMenu("Recent Fre&quencies")
        self._recent_menu.setToolTipsVisible(True)
        self._step_menu = radio_menu.addMenu("Tuning S&tep")
        self._step_menu.setToolTipsVisible(True)
        # [ and ] are handled in keyPressEvent (plots focused only, like
        # Space); the "\t[" suffix shows the key in the menu and in F1.
        self._add_action(
            self._step_menu,
            "&Smaller Step\t[",
            lambda: self._change_step(-1),
            tip="Tune in finer steps (the [ key while the plots have the focus)",
        )
        self._add_action(
            self._step_menu,
            "&Larger Step\t]",
            lambda: self._change_step(1),
            tip="Tune in coarser steps (the ] key while the plots have the focus)",
        )
        self._step_menu.addSeparator()
        step_group = QActionGroup(self)
        step_group.setExclusive(True)
        self._step_actions: Dict[float, QAction] = {}
        for step in STEP_CHOICES_HZ:
            act = QAction(format_step(step), self)
            act.setCheckable(True)
            act.setStatusTip(
                f"Left/Right and the mouse wheel tune by {format_step(step)}"
            )
            act.triggered.connect(lambda _c=False, s=step: self._set_step(s))
            step_group.addAction(act)
            self._step_menu.addAction(act)
            self._step_actions[step] = act
        radio_menu.addSeparator()
        self._keep_centered_action = self._add_action(
            radio_menu,
            "&Keep Tuned Frequency Centered",
            self._set_keep_centered,
            tip="Retune the receiver for every new frequency, so the tuned "
            "frequency is always in the middle of the spectrum (off: tuning "
            "moves within the span and the receiver retunes only at its edges)",
            checkable=True,
            checked=self._settings.get_bool("tuning/keep_centered", False),
        )
        self._rds_auto_action = self._add_action(
            radio_menu,
            "&RDS Station Names",
            self._set_rds_auto,
            tip="While you listen to broadcast FM, decode the station's RDS in "
            "the background: its name next to FREQ (the RadioText in its "
            "tooltip) and on the radio tuner",
            checkable=True,
            checked=self._settings.get_bool("decoder/rds_auto", True),
        )
        self._track_action = self._add_action(
            radio_menu,
            "Track Signal (AF&C)",
            lambda on: self._signal_tools.set_tracking(on),
            tip="Keep a drifting signal centred: the tuned frequency follows it "
            "(narrow FM, AM and CW; the receiver retunes only at the span's "
            "edges). The status bar shows how far it moved",
            checkable=True,
        )
        radio_menu.addSeparator()
        self._tx_station.add_menu_actions(radio_menu)  # Send Station ID, Stop TX
        self._rebuild_recent_menu()
        self._update_history_actions()

        # ---- View ----
        view_menu = menubar.addMenu("&View")
        self._spectrum_action = self._add_action(
            view_menu,
            "&Spectrum",
            self._spectrum.setVisible,
            tip="Show or hide the spectrum plot",
            checkable=True,
            checked=True,
        )
        self._waterfall_action = self._add_action(
            view_menu,
            "&Waterfall",
            self._waterfall.setVisible,
            tip="Show or hide the waterfall",
            checkable=True,
            checked=True,
        )
        for action in (self._spectrum_action, self._waterfall_action):
            action.toggled.connect(self._sync_plot_actions)
        # Without the spectrum above it, the waterfall labels its frequencies.
        self._spectrum_action.toggled.connect(
            lambda shown: self._waterfall.set_frequency_axis_visible(not shown)
        )
        self._add_zoom_menu(view_menu)
        view_menu.addSeparator()
        panels_menu = view_menu.addMenu("&Panels")
        for index, (key, title, tip) in enumerate(self._panel_specs):
            shortcut = f"Ctrl+{index + 1}" if index < 9 else None
            self._add_action(
                panels_menu,
                self._PANEL_MENU_TEXT.get(key, title),
                lambda k=key: self._show_panel(k, focus=True),
                shortcut,
                tip,
            )
        self._add_action(
            view_menu,
            "&Focus Spectrum",
            self._focus_plots,
            "F6",
            "Put the keyboard focus on the spectrum, so Space and the arrow "
            "keys work",
        )
        theme_menu = view_menu.addMenu("&Theme")
        self._theme_actions: Dict[str, QAction] = {}
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for name, text in (("dark", "&Dark"), ("light", "&Light")):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setStatusTip(f"Use the {name} color theme")
            act.triggered.connect(lambda _c=False, n=name: self._set_theme(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act
        theme_menu.addSeparator()
        self._add_action(
            theme_menu,
            "&Toggle Theme",
            self._toggle_theme,
            "Ctrl+T",
            "Switch between the dark and light themes",
        )
        self._sync_theme_actions()
        view_menu.addSeparator()
        self._add_action(
            view_menu,
            "&Reset Layout",
            self._reset_layout,
            tip="Restore the default panel sizes and show both plots",
        )

        # ---- Tools ----
        tools_menu = menubar.addMenu("&Tools")
        self._add_action(
            tools_menu,
            "Frequency &Scanner",
            self._show_scanner,
            "Ctrl+F",
            "Show the Scanner tab: sweep a band, stop on signals or scan your "
            "bookmarks",
        )
        self._add_action(
            tools_menu,
            "Protocol &Decoder",
            self._show_decoder_config,
            tip="Show the Decoder panel",
        )
        if HAS_HAM_RADIO:
            self._add_action(
                tools_menu,
                "AM/FM Radio &Tuner...",
                self._show_radio_tuner,
                "Ctrl+R",
                "Open the broadcast radio tuner window",
            )
            self._add_action(
                tools_menu,
                "SSTV: Tune to &ISS (145.800 MHz)",
                self._tune_iss_from_menu,
                tip="Tune to the ISS SSTV downlink (FM, 5 kHz deviation), start "
                "receiving and listen for images in the SSTV panel",
            )
            self._add_action(
                tools_menu,
                "Transmit Te&xt...",
                self._show_transmit_text,
                "Ctrl+Shift+T",
                "Send text in Morse code (CW), RTTY, PSK31 or ASCII FSK: preview "
                "it, save it as audio or I/Q, or transmit it on a HackRF One",
            )
        self._add_action(
            tools_menu,
            "&Identify Signal",
            lambda: self._signal_tools.identify(),
            "Ctrl+I",
            "Say what the signal at the tuned frequency is (type, likely mode "
            "and width, SNR) in the Signal tab, with Apply Mode",
        )
        signal_menu = tools_menu.addMenu("Si&gnal")
        signal_menu.setToolTipsVisible(True)
        for text, page, tip in (
            ("Signals in &View", "signals", "List the signals in the span"),
            ("&Scope", "scope", "The tuned channel in time or as a constellation"),
            ("&Bursts", "bursts", "Log the transmissions that come and go"),
        ):
            self._add_action(
                signal_menu,
                text,
                lambda p=page: self._show_signal_page(p),
                tip=tip,
            )
        signal_menu.addSeparator()
        # Home, Page Up and Page Down are handled in keyPressEvent (plots
        # focused, like Space); the suffix shows the key here and in F1.
        for text, step, tip in (
            ("Tune to Strongest Signal\tHome", "strongest", "The strongest in view"),
            ("Next Signal Up\tPgUp", "up", "The next signal above the tuned one"),
            ("Next Signal Down\tPgDown", "down", "The next signal below"),
        ):
            self._add_action(
                signal_menu,
                text,
                lambda s=step: self._signal_tools.step(s),
                tip=tip + " (keys work while the plots have the focus)",
            )
        tools_menu.addSeparator()
        self._add_action(
            tools_menu,
            "&Error History...",
            self._show_error_history,
            "Ctrl+E",
            "Show recent warnings and errors",
        )
        preferences = self._add_action(
            tools_menu,
            "&Preferences...",
            self._show_preferences,
            "Ctrl+,",
            "Audio output device, default folders, the recording temp folder "
            "and the transmit confirmation",
        )
        preferences.setMenuRole(QAction.MenuRole.PreferencesRole)

        # ---- Help ----
        help_menu = menubar.addMenu("&Help")
        self._add_action(
            help_menu,
            "&Welcome and Quick Start...",
            self._show_welcome,
            tip="Show the welcome screen again: pick a starting band, start "
            "Demo Mode and see the quick tips",
        )
        self._add_action(
            help_menu,
            "&Keyboard Shortcuts...",
            self._show_help,
            "F1",
            "List every keyboard shortcut",
        )
        help_menu.addSeparator()
        self._add_action(help_menu, "&About SDR Module...", self._show_about)

    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        # No "hide toolbar" context menu: there would be no way to get it back.
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(toolbar)
        self._toolbar = toolbar

        # Start/Stop: accent call to action while idle, a plain "Stop" while
        # running. Red is kept for Record, so "Stop" next to a red Record
        # button can't be mistaken for "stop recording".
        self._start_button = QPushButton(self._START_TEXT)
        self._start_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._start_button.setAccessibleName("Start or stop receiving")
        self._start_button.clicked.connect(self._toggle_acquisition)
        set_role(self._start_button, "primary")
        self._start_button.setMinimumWidth(
            self._text_width(self._start_button, self._START_TEXT, self._STOP_TEXT)
        )
        toolbar.addWidget(self._start_button)

        # Record: a toggle, filled red while recording.
        self._record_button = QPushButton(self._RECORD_TEXT)
        self._record_button.setCheckable(True)
        self._record_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._record_button.setAccessibleName("Record I/Q")
        self._record_button.setMinimumWidth(
            self._text_width(self._record_button, self._RECORD_TEXT)
        )
        self._record_button.clicked.connect(
            lambda _checked=False: self._on_record_button()
        )
        toolbar.addWidget(self._record_button)
        self._update_record_button_tip()

        # ON AIR (red) while transmitting, beside Start and Record (so it
        # never falls into the toolbar's overflow); a click stops it.
        self._tx_station.add_toolbar_badge(toolbar)

        toolbar.addSeparator()

        # Audio on/off (the same switch as Radio > Audio Output) and volume.
        self._audio_button = QPushButton(self._AUDIO_ON_TEXT)
        self._audio_button.setCheckable(True)
        self._audio_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._audio_button.setAccessibleName("Audio output")
        self._audio_button.setMinimumWidth(
            self._text_width(
                self._audio_button, self._AUDIO_ON_TEXT, self._AUDIO_OFF_TEXT
            )
            + 20  # icon
        )
        self._audio_button.clicked.connect(self._audio_action.setChecked)
        toolbar.addWidget(self._audio_button)
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setPageStep(10)
        self._volume_slider.setValue(self._saved_volume())
        self._volume_slider.setFixedWidth(96)
        self._volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._volume_slider.setAccessibleName("Volume")
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        toolbar.addWidget(self._volume_slider)
        self._on_volume_changed(self._volume_slider.value(), save=False)

        toolbar.addSeparator()

        # Frequency readout: the tuned frequency, in a fixed width (the
        # toolbar after it never shifts); scrolling over a digit tunes it.
        freq_caption = QLabel("FREQ")
        set_role(freq_caption, "caption")
        toolbar.addWidget(freq_caption)
        self._freq_label = QLabel(format_frequency(DEFAULT_FREQUENCY_HZ))
        set_role(self._freq_label, "lcd")
        self._freq_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._freq_label.setToolTip(
            "Tuned frequency. Click here (or press Ctrl+L) to type a new one; "
            "scroll over a digit to tune that digit. You can also click the "
            "spectrum or waterfall, scroll over them, or use Left/Right while "
            "they have the focus (the STEP in the receiver bar; Shift: ten "
            "steps, Ctrl: 1 MHz)."
        )
        self._freq_label.setAccessibleName("Tuned frequency")
        # Clicking the readout jumps to the Frequency field.
        self._freq_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._freq_label.installEventFilter(self)
        self._freq_digits = FrequencyDigits(self._freq_label)
        self._freq_digits.digit_tuned.connect(self._nudge_frequency)
        toolbar.addWidget(self._freq_label)
        # The broadcast station's name from its RDS, while there is one.
        self._rds_badge = QLabel("")
        set_role(self._rds_badge, "badge", "info")
        # A received name is data, never markup ("<b>HITS" stays literal).
        self._rds_badge.setTextFormat(Qt.TextFormat.PlainText)
        self._rds_badge.setAccessibleName("Station name (RDS)")
        self._rds_badge_action = toolbar.addWidget(self._rds_badge)
        self._rds_badge_action.setVisible(False)

        toolbar.addSeparator()

        # Signal level meter: the tuned channel's level with the squelch
        # threshold (drag it to set the squelch) and whether it is open.
        level_caption = QLabel("LEVEL")
        set_role(level_caption, "caption")
        toolbar.addWidget(level_caption)
        self._level_meter = LevelMeter()
        self._level_meter.threshold_changed.connect(self._on_meter_threshold)
        # The readout's old name: text() / setText() work on the meter.
        self._level_label = self._level_meter
        toolbar.addWidget(self._level_meter)

    def _setup_receiver_bar(self) -> None:
        """Second toolbar row: MODE and FILTER (mirroring the control panel's
        Mode and Bandwidth) and the tuning STEP."""
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        panel = self._control_panel
        self._receiver_bar = ReceiverBar(panel._demod_combo, panel._bw_combo, self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._receiver_bar)
        self._receiver_bar.step_changed.connect(self._set_step)

    def _setup_statusbar(self):
        """Setup status bar."""
        statusbar = QStatusBar()
        # Keep the state badge off the window edge. The window edges resize
        # the window; the size grip only drew a stray box in the corner.
        statusbar.setContentsMargins(8, 0, 8, 0)
        statusbar.setSizeGripEnabled(False)
        self.setStatusBar(statusbar)

        # Device state badge + device name + sample rate (left)
        self._state_badge = QLabel("NO DEVICE")
        set_role(self._state_badge, "badge", "warning")
        self._state_badge.setAccessibleName("Receiver state")
        statusbar.addWidget(self._state_badge)

        self._device_label = QLabel("No device")
        set_role(self._device_label, "value")
        self._device_label.setToolTip(
            "Current device. Change it with Device > Connect..."
        )
        statusbar.addWidget(self._device_label)

        self._rate_label = QLabel("")
        self._rate_label.setToolTip(get_short_tip("sample_rate"))
        # A click opens the sample rate menu (Device > Sample Rate).
        self._rate_label.setAccessibleName("Sample rate")
        self._rate_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rate_label.installEventFilter(self)
        statusbar.addWidget(self._rate_label)

        # Transient messages (neutral, success or error), auto-clearing. A
        # long message ends in "…" instead of raising the window's minimum
        # width; the full text is also in the tooltip.
        self._message_label = _ElidedLabel("")
        statusbar.addWidget(self._message_label, 1)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_status_message)

        # Recording indicator (right, only while recording)
        self._recording_label = QLabel("REC")
        set_role(self._recording_label, "badge", "danger")
        self._recording_label.setToolTip("Recording raw I/Q samples")
        self._recording_label.setVisible(False)
        self._recording_label.setAccessibleName("Recording time")
        statusbar.addPermanentWidget(self._recording_label)
        self._recording_info_label = QLabel("")
        self._recording_info_label.setToolTip(
            "Recorded size, and the free space left on the drive the "
            "recording is written to (it stops below 500 MB)"
        )
        self._recording_info_label.setVisible(False)
        statusbar.addPermanentWidget(self._recording_info_label)
        # Record Audio: "AUDIO 00:12" while what you hear is recorded.
        self._audio_rec_label = QLabel("AUDIO")
        set_role(self._audio_rec_label, "badge", "danger")
        self._audio_rec_label.setAccessibleName("Audio recording time")
        self._audio_rec_label.setVisible(False)
        statusbar.addPermanentWidget(self._audio_rec_label)

    def _setup_timers(self):
        """Setup update timers."""
        # Display update timer (30 Hz)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._update_display)
        self._display_timer.start(33)

        # Status update timer (5 Hz)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(200)

        # Device hot-plug polling (0.5 Hz)
        self._hotplug_timer = QTimer(self)
        self._hotplug_timer.timeout.connect(self._poll_hotplug)
        self._hotplug_timer.start(2000)

    def _connect_signals(self):
        """Connect control panel signals."""
        # Typed, stepped, quick-tuned or preset frequencies: the tuning rules
        # of set_frequency apply (and a refused one is put back).
        self._control_panel.frequency_changed.connect(self._on_panel_frequency_changed)
        self._control_panel._freq_input.entry_rejected.connect(
            self._on_frequency_entry_rejected
        )
        self._control_panel.gain_changed.connect(self._on_gain_changed)
        self._control_panel.bandwidth_changed.connect(self._on_bandwidth_changed)
        self._control_panel.squelch_changed.connect(self._on_squelch_changed)
        self._control_panel.agc_changed.connect(self._on_agc_changed)
        self._control_panel.demod_changed.connect(self._on_demod_changed)
        # The control panel's Record button drives the same recording as the
        # toolbar action (its signals were previously connected to nothing).
        self._control_panel.recording_started.connect(self._on_panel_record_started)
        self._control_panel.recording_stopped.connect(self._on_panel_record_stopped)
        self._control_panel.recording_paused.connect(self._on_panel_record_paused)
        # The decoder panel's protocol selector drives a live decoder over the
        # acquired samples (the panel was previously fed no data at all).
        self._decoder_panel.protocol_changed.connect(self._on_decoder_protocol_changed)
        # Bookmarks and the Decoder panel's "Tune to ..." tune the receiver
        # (and select the mode they listen with); "Add" saves that mode.
        self._bookmarks_panel.tune_requested.connect(self._on_bookmark_tune)
        self._decoder_panel.tune_requested.connect(self._on_bookmark_tune)
        self._bookmarks_panel.set_mode_source(self._listening_mode)
        # The license class decides where transmitting is allowed; remember it.
        self._control_panel.license_changed.connect(self._on_license_changed)
        # Remembered between sessions, like the mode they belong to (the
        # value, "75 kHz", not the item text "75 kHz (broadcast)").
        self._control_panel._fm_dev_combo.currentTextChanged.connect(
            lambda _text: self._save_setting(
                "fm_deviation", self._control_panel.get_fm_deviation_text()
            )
        )
        # Broadcast and narrow FM remember their own tuning step.
        self._control_panel._fm_dev_combo.currentTextChanged.connect(
            lambda _text: self._load_step_for_mode()
        )
        self._connect_audio_controls()
        self._control_panel._format_combo.currentTextChanged.connect(
            lambda text: self._save_setting("recording_format", text)
        )
        # The Recording group: what Record records, the take after Stop,
        # the recordings folder and auto-save.
        panel = self._control_panel
        panel.record_kind_changed.connect(self._on_record_kind_changed)
        panel.take_save_requested.connect(self._on_take_save)
        panel.take_discard_requested.connect(self._on_take_discard)
        panel.open_folder_requested.connect(self._on_open_folder)
        panel.folder_change_requested.connect(self._on_change_recordings_folder)
        panel.auto_save_changed.connect(self._on_auto_save_changed)
        # The Decoder tab's Decode File... (File > Decode Recording).
        self._decoder_panel.decode_file_requested.connect(
            lambda: self._decode_recording()
        )
        # Apply Preset tunes silently; confirm it in the status bar. (The
        # panel's own handler is connected first, so this runs after it.)
        self._control_panel._apply_preset_btn.clicked.connect(
            self._on_panel_preset_applied
        )
        # After Ctrl+L, Enter in the Frequency field returns to the plots.
        entry = self._frequency_field()
        if hasattr(entry, "editingFinished"):
            entry.editingFinished.connect(self._on_frequency_entry_finished)
        if HAS_HAM_RADIO:
            self._callsign_panel.id_requested.connect(self._on_callsign_id_requested)
            self._callsign_panel.transmit_text_requested.connect(
                self._show_transmit_text
            )
            self._callsign_panel.callsign_changed.connect(self._save_ham_id_settings)
            self._callsign_panel.settings_changed.connect(self._save_ham_id_settings)
            self._tx_station.connect_panel()
            self._connect_qrp()
            self._sstv_panel.start_requested.connect(self._on_sstv_start_requested)
            self._connect_sstv_and_smeter()
        # Click-to-tune from spectrum and waterfall. set_frequency keeps the
        # control panel, toolbar readout and plot axes in step. The mouse
        # wheel over them tunes by the step.
        self._spectrum.frequency_clicked.connect(self.set_frequency)
        self._waterfall.frequency_clicked.connect(self.set_frequency)
        self._spectrum.tune_steps.connect(self._tune_steps)
        self._waterfall.tune_steps.connect(self._tune_steps)
        self._waterfall.bookmark_requested.connect(self._bookmark_frequency)
        self._waterfall.display_settings_changed.connect(self._save_waterfall_settings)
        self._connect_plots()
        self._right_tabs.currentChanged.connect(self._on_panel_tab_changed)
        self._connect_bookmarks_and_scanner()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if (
            obj is getattr(self, "_freq_label", None)
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._focus_frequency_entry()
            return True
        if (
            obj is getattr(self, "_rate_label", None)
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._device_controls.popup_rate_menu(self._rate_label)
            return True
        return super().eventFilter(obj, event)

    def _frequency_field(self) -> "QWidget":
        """The Frequency spin box of the control panel."""
        entry = self._control_panel._freq_input
        return getattr(entry, "_freq_input", entry)

    def _focus_frequency_entry(self) -> None:
        """Put the cursor in the Frequency field, ready to type. Enter then
        tunes and returns the focus to the plots."""
        field = self._frequency_field()
        area = self._control_panel.findChild(QScrollArea)
        if area is None and isinstance(self._right_splitter.widget(0), QScrollArea):
            area = self._right_splitter.widget(0)
        if area is not None:
            area.ensureWidgetVisible(field)
        field.setFocus(Qt.FocusReason.ShortcutFocusReason)
        select_all = getattr(field, "selectAll", None)
        if callable(select_all):
            select_all()
        self._entry_returns_focus = True

    def _on_frequency_entry_finished(self) -> None:
        """Enter in the Frequency field after Ctrl+L: back to the plots
        (unless what was typed was refused: then it stays, to retype)."""
        field = self._frequency_field()
        if self._entry_rejected:
            self._entry_rejected = False
            if field.hasFocus():
                select_all = getattr(field, "selectAll", None)
                if callable(select_all):
                    select_all()
                return
        returns = self._entry_returns_focus
        self._entry_returns_focus = False
        # editingFinished also fires when the field loses focus; only move
        # the focus when Enter was pressed (the field still has it).
        if returns and field.hasFocus():
            self._focus_plots()

    def _on_frequency_entry_rejected(self, reason: str) -> None:
        """The Frequency field dropped text it can't tune: say why."""
        self._entry_rejected = self._frequency_field().hasFocus()
        # A typing slip, not a fault: explained, not logged.
        self._show_status_message(reason, "danger", 6000)

    def _focus_plots(self) -> None:
        """Give the keyboard focus to the spectrum (or the waterfall when the
        spectrum is hidden), where Space and the arrow keys work."""
        plot = self._spectrum if self._spectrum.isVisible() else self._waterfall
        plot.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _plots_focused(self) -> bool:
        """Whether the keyboard focus is on (or inside) a plot."""
        focus = QApplication.focusWidget()
        return focus is not None and any(
            focus is plot or plot.isAncestorOf(focus)
            for plot in (self._spectrum, self._waterfall)
        )

    @staticmethod
    def _focus_uses_key(key: int) -> bool:
        """Whether the focused widget has its own use for ``key``, so the
        window must not act on it even though the widget let it through
        (Space on a slider or tab bar, Left/Right on a combo box or
        button)."""
        focus = QApplication.focusWidget()
        owners: Tuple[type, ...] = (
            QAbstractSlider,
            QTabBar,
            QComboBox,
            QAbstractItemView,
            QAbstractSpinBox,
            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
        )
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            owners += (QAbstractButton,)
        return isinstance(focus, owners)

    def keyPressEvent(self, event):
        """Space starts/stops receiving; Left/Right tune; [ and ] change the
        tuning step; Esc cancels typing in the Frequency field and returns
        the focus to the plots.

        These are handled here instead of as window-wide shortcuts, so they
        only act when the focused widget does not use the key itself: a
        focused slider, combo box, list, tab bar, button or text field keeps
        its normal arrow/Space behavior, and the key only reaches the window
        when nothing else consumed it (e.g. with the spectrum focused). Keys
        such a widget ignores (Space on a slider) are not acted on either.
        [ and ] act only while a plot has the focus.
        """
        key = event.key()
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if key == Qt.Key.Key_Escape and mods == Qt.KeyboardModifier.NoModifier:
            if self._frequency_field().hasFocus():
                # Esc cancels the edit (moving the focus would commit it).
                self._control_panel._freq_input.cancel_edit()
            self._entry_returns_focus = False
            self._entry_rejected = False
            self._focus_plots()
            event.accept()
            return
        if key == Qt.Key.Key_Space and mods == Qt.KeyboardModifier.NoModifier:
            if self._focus_uses_key(key):
                event.ignore()
                return
            if not event.isAutoRepeat():
                self._toggle_acquisition()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if self._focus_uses_key(key):
                event.ignore()
                return
            sign = 1 if key == Qt.Key.Key_Right else -1
            if mods == Qt.KeyboardModifier.NoModifier:
                self._tune_steps(sign)
            elif mods == Qt.KeyboardModifier.ShiftModifier:
                self._tune_steps(10 * sign)
            elif mods == Qt.KeyboardModifier.ControlModifier:
                self._nudge_frequency(sign * self._TUNE_STEPS_HZ[mods])
            else:
                super().keyPressEvent(event)
                return
            event.accept()
            return
        if event.text() in ("[", "]") and self._plots_focused():
            self._change_step(1 if event.text() == "]" else -1)
            event.accept()
            return
        if (
            key == Qt.Key.Key_P
            and mods == Qt.KeyboardModifier.NoModifier
            and self._plots_focused()
        ):
            # P pauses the waterfall from either plot (the waterfall takes P
            # itself when it has the focus); not while a control in a plot's
            # header (a combo box) has a use for it.
            if self._focus_uses_key(key):
                event.ignore()
                return
            self._waterfall.set_paused(not self._waterfall.is_paused())
            event.accept()
            return
        steps = {
            Qt.Key.Key_Home: "strongest",
            Qt.Key.Key_PageUp: "up",
            Qt.Key.Key_PageDown: "down",
        }
        if (
            key in steps
            and mods == Qt.KeyboardModifier.NoModifier
            and self._plots_focused()
        ):
            # (T10) The signals in view: the strongest, the next up / down.
            self._signal_tools.step(steps[key])
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Status bar messages
    # ------------------------------------------------------------------

    def _show_status_message(
        self, message: str, tone: Optional[str] = None, duration_ms: int = 4000
    ) -> None:
        """Show a transient message in the status bar.

        Args:
            message: Text to display
            tone: ``None`` (neutral), ``"success"``, ``"info"``, ``"warning"``
                or ``"danger"``
            duration_ms: How long to show before auto-clearing
        """
        self._message_label.setText(message)
        self._message_label.setToolTip(message)
        set_tone(self._message_label, tone)
        self._message_timer.start(max(500, int(duration_ms)))

    def _show_status_error(
        self, message: str, duration_ms: int = 6000, log: bool = True
    ) -> None:
        """Show a transient error message (red) in the status bar, and log
        it for Tools > Error History (``log=False``: the caller logged it
        already, so the event is listed once)."""
        self._show_status_message(message, "danger", duration_ms)
        if log:
            logger.warning(f"Status bar error: {message}")

    def _clear_status_message(self) -> None:
        self._message_label.setText("")
        self._message_label.setToolTip("")
        set_tone(self._message_label, None)

    # ------------------------------------------------------------------
    # State presentation
    # ------------------------------------------------------------------

    def _current_frequency(self) -> float:
        """The tuned frequency (the VFO): what is demodulated, decoded and
        measured, and what FREQ and the Frequency field show."""
        return float(self._control_panel._freq_input.get_frequency())

    def _lo_frequency(self) -> float:
        """The LO: the device's center frequency, the middle of the
        spectrum and waterfall."""
        return float(self._lo_hz)

    def _vfo_offset_hz(self) -> float:
        """The tuned frequency minus the LO: where in the span the tuned
        channel is."""
        return self._current_frequency() - self._lo_hz

    def _device_sample_rate(self) -> float:
        """Sample rate of the current device (default 2.4 MS/s)."""
        dev = self._device
        if dev is None:
            return _DEFAULT_SAMPLE_RATE
        state = getattr(dev, "state", None)
        for value in (
            getattr(state, "sample_rate", None),
            getattr(dev, "sample_rate", None),
        ):
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if rate > 0:
                return rate
        return _DEFAULT_SAMPLE_RATE

    def _device_display_name(self) -> str:
        dev = self._device
        if dev is None:
            return "No device"
        if self._demo_mode:
            return "Demo Device (simulated signals)"
        name = getattr(getattr(dev, "info", None), "name", "") or ""
        return str(name) or type(dev).__name__

    def _refresh_state_ui(self) -> None:
        """Sync toolbar, status bar, menus and title with the device state."""
        has_device = self._device is not None
        running = self._is_running and has_device

        # Toolbar Start/Stop
        button = self._start_button
        button.setText(self._STOP_TEXT if running else self._START_TEXT)
        role = "" if running else "primary"
        if (button.property("role") or "") != role:
            set_role(button, role or None)
        if running:
            button.setToolTip("Stop receiving (Space)")
        elif has_device:
            button.setToolTip("Start receiving (Space)")
        else:
            button.setToolTip(
                "Start receiving (Space). No device is connected yet: you can "
                "connect one or use the demo device."
            )
        self._start_action.setText(
            "&Stop Receiving\tSpace" if running else "&Start Receiving\tSpace"
        )

        # Status bar
        if not has_device:
            self._state_badge.setText("NO DEVICE")
            set_tone(self._state_badge, "warning")
            self._state_badge.setToolTip(
                "Use Device > Connect... or Device > Use Demo Device"
            )
        elif running:
            self._state_badge.setText("RUNNING")
            set_tone(self._state_badge, "success")
            self._state_badge.setToolTip("Receiving samples")
        else:
            self._state_badge.setText("STOPPED")
            set_tone(self._state_badge, "muted")
            self._state_badge.setToolTip("Press Start or Space to receive")
        if self._tx_busy():
            self._show_tx_state(True)
        if has_device:
            self._device_label.setText(self._device_display_name())
        else:
            self._device_label.setText(
                "Press Start to connect a device or run the demo"
            )
        label_role = "value" if has_device else "muted"
        if self._device_label.property("role") != label_role:
            set_role(self._device_label, label_role)
        rate = self._device_sample_rate()
        self._rate_label.setText(format_rate(rate) if has_device else "")

        # Menus. Use Demo Device stays available while a real device (or a
        # recording) is in use, and is ticked (and off) while the demo is.
        self._disconnect_action.setEnabled(has_device)
        demo_in_use = bool(self._demo_mode and has_device)
        self._demo_action.setChecked(demo_in_use)
        self._demo_action.setEnabled(not demo_in_use)
        self._demo_action.setStatusTip(
            "The demo device is in use"
            if demo_in_use
            else "Explore the app with simulated signals (no hardware needed); it "
            "replaces a connected device"
        )
        # The Receiver group's gain, sample rate and front end follow the
        # device (T11).
        self._device_controls.sync()

        # Level readout is only meaningful while samples flow
        if not running:
            self._last_peak_db = None
            self._level_label.setText(f"{_NO_VALUE} dBFS")
            set_tone(self._level_label, None)

        # Plot axes / click-to-tune span (centered on the LO), and the tuned
        # channel marked on them
        center = self._lo_frequency()
        self._spectrum.set_frequency_range(center, rate)
        self._waterfall.set_frequency_range(center, rate)
        self._update_passband()
        self._sync_plot_state(running, has_device)

        # Only a HackRF One can send the Ham ID: what Send ID Now would do
        # (or why it can't), and the QRP calculator's SDR output.
        self._tx_station.refresh()
        # An open radio tuner: what the device can tune; no STEREO when idle.
        self._sync_tuner_device()

        # A live decoder is built for one sample rate; rebuild it when the
        # device (and so the rate) changes, also after a build that failed
        # (e.g. one that needs a higher rate).
        if self._decoder_panel.protocol() is not None and self._decoder_rate != rate:
            self._rebuild_decoder()

        # The Decoder panel's hint says whether it is listening; its
        # messages note the sample rate, and Auto follows the mode.
        self._decoder_panel.set_receiving(running)
        self._decoder_panel.set_sample_rate(rate)
        self._decoder_panel.set_demod_mode(self._control_panel.get_demod_mode())
        self._update_decoder_chip()
        listener = self._own("_rds_listener")
        if not has_device and listener is not None:
            # No receiver: no station to name.
            listener.reset()
            self._show_rds_station(None)
        if HAS_HAM_RADIO:
            self._sync_signal_meter(running, has_device)

        if self._recording:
            self._update_rec_clock()
            self._update_recording_status()  # also armed / capturing
        else:
            self._sync_recording_armed()
        self._update_audio_recording_status()

        # A recording playing back: Start/Stop play and pause it.
        self._playback.sync_state_ui()
        self._close_recording_action.setEnabled(self._playback.active)
        signal_tools = self._own("_signal_tools")
        if signal_tools is not None:
            signal_tools.sync_state(running, has_device)  # (T10)

        self._update_window_title()
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _device_can_transmit(self) -> bool:
        """True when the current device is a (TX-capable) HackRF One."""
        if self._device is None:
            return False
        try:
            from ..devices.hackrf import HackRFDevice
        except ImportError:  # pragma: no cover - defensive
            return False
        return isinstance(self._device, HackRFDevice)

    def _update_window_title(self) -> None:
        if self._device is None:
            self.setWindowTitle("SDR Module")
        elif self._demo_mode:
            self.setWindowTitle("Demo Mode — SDR Module")
        else:
            self.setWindowTitle(f"{self._device_display_name()} — SDR Module")

    def _refresh_info_panel(self) -> None:
        """Fill the Info tab from the current state."""
        p = self._info_panel
        has_device = self._device is not None
        running = self._is_running and has_device
        rate = self._device_sample_rate()

        p.set_value(
            "device", self._device_display_name(), None if has_device else "warning"
        )
        playback_state = self._playback.info_state()
        if not has_device:
            p.set_value("state", "Not connected", "warning")
        elif playback_state:
            p.set_value("state", playback_state, "success" if running else "muted")
        elif running:
            p.set_value("state", "Receiving", "success")
        else:
            p.set_value("state", "Stopped", "muted")
        p.set_value("frequency", format_frequency(self._current_frequency()))
        p.set_value("lo", format_frequency(self._lo_frequency()))
        p.set_value("sample_rate", format_rate(rate) if has_device else _NO_VALUE)
        p.set_value("span", format_bandwidth(rate) if has_device else _NO_VALUE)
        p.set_value(
            "rbw",
            (
                f"{format_hz(self._frame_spectrum.rbw(rate))} "
                f"({self._frame_spectrum.size}-point FFT, "
                f"{WINDOW_LABELS[self._frame_spectrum.window]})"
                if has_device
                else _NO_VALUE
            ),
        )
        p.set_value("demod", self._control_panel._demod_combo.currentText())
        p.set_value("bandwidth", self._bandwidth_description())
        p.set_value("gain", self._device_controls.gain_text())
        p.set_value("frontend", self._device_controls.front_end_text())
        p.set_value("squelch", *self._squelch_description(running))
        if running and self._last_peak_db is not None:
            p.set_value("level", self._level_description())
        else:
            p.set_value("level", _NO_VALUE)

        p.set_value("audio", *self._audio_status())
        if not self._recording:
            p.set_value("recording", "Idle")
        elif self._recording_paused:
            p.set_value(
                "recording", f"Paused at {self._recording_elapsed_text()}", "warning"
            )
        elif not running:
            p.set_value("recording", "Armed, waiting for the receiver", "warning")
        else:
            p.set_value("recording", self._recording_elapsed_text(), "danger")
        take = self._audio_take
        if take is not None:  # Record Audio
            audio = f"Audio {_clock_text(take.seconds)}"
            iq = p.value("recording")
            p.set_value(
                "recording", audio if iq == "Idle" else f"{iq} · {audio}", "danger"
            )
        count = self._buffer_sample_count()
        if count:
            p.set_value("buffer", f"{count:,} samples ({count * 8 / 1e6:.1f} MB)")
        else:
            p.set_value("buffer", "Empty")

        p.set_value("version", __version__)
        p.set_value("qt", f"{QT_VERSION_STR} / {PYQT_VERSION_STR}")
        p.set_value("theme", self._theme.title())

    def _on_panel_tab_changed(self, _index: int) -> None:
        if self._right_tabs.currentWidget() is self._panel_pages.get("info"):
            self._refresh_info_panel()
        self._decoder_tab_seen()

    def _decoder_tab_seen(self) -> None:
        """The Decoder tab is on screen again (shown, or its window restored
        from the taskbar): its title stops counting new messages."""
        if self._own("_decoder_unread") and self._decoder_page_visible():
            self._decoder_unread = 0
            self._update_decoder_tab_title()

    def _show_panel(self, key: str, focus: bool = False) -> None:
        """Bring one of the right-hand tool panels to the front; with
        ``focus`` (View > Panels, Ctrl+1...) also move the keyboard focus to
        its first control."""
        page = self._panel_pages.get(key)
        if page is None:
            return
        owner = self._panel_owners.get(key, self._right_tabs)
        if owner is not self._right_tabs:
            # A nested panel (e.g. the Ham Radio group): show its group first.
            self._right_tabs.setCurrentWidget(owner)
        owner.setCurrentWidget(page)
        if focus:
            target = self._first_focusable(page)
            if target is None:
                target = owner.tabBar()  # read-only panel: its tab
            target.setFocus(Qt.FocusReason.ShortcutFocusReason)

    @staticmethod
    def _first_focusable(page: "QWidget") -> Optional["QWidget"]:
        """The first control inside ``page`` that Tab would reach."""
        widget = page.nextInFocusChain()
        for _ in range(500):
            if widget is None or widget is page:
                return None
            if (
                page.isAncestorOf(widget)
                and widget.isVisibleTo(page)
                and widget.isEnabled()
                and widget.focusPolicy() & Qt.FocusPolicy.TabFocus
            ):
                return widget
            widget = widget.nextInFocusChain()
        return None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if not self._layout_initialized:
            self._layout_initialized = True
            if not self._splitters_restored:
                self._apply_default_splitter_sizes()

    def _apply_default_splitter_sizes(self) -> None:
        """Default proportions: controls ~30% (380-440 px) wide, the control
        panel over ~55% of the right column, spectrum over 40% of the plots."""
        width = self._main_splitter.width()
        if width < 400:
            width = self.width() - 12
        right = int(min(440, max(380, width * 0.3)))
        self._main_splitter.setSizes([max(200, width - right), right])

        height = self._right_splitter.height()
        if height < 300:
            height = self.height() - 120
        top = self._snap_to_section_gap(int(height * 0.55), height)
        self._right_splitter.setSizes([top, max(120, height - top)])

        height = self._display_splitter.height()
        if height < 300:
            height = self.height() - 120
        spectrum = int(height * 0.4)
        self._display_splitter.setSizes([spectrum, max(120, height - spectrum)])

    def _snap_to_section_gap(self, top: int, height: int) -> int:
        """Move the control pane's lower edge into the gap after a section.

        A split that falls inside a group box leaves a sliver of its title or
        border showing above the tool tabs, which looks broken. Use the gap
        nearest to ``top`` that keeps the pane between 40% and 70% of the
        column; otherwise keep ``top``.
        """
        pane = self._right_splitter.widget(0)
        if pane is None:
            return top
        try:
            if pane.layout() is not None:
                pane.layout().activate()
            self._control_panel.layout().activate()
            gaps = []
            for group in self._control_panel.findChildren(QGroupBox):
                if not group.isVisibleTo(self._control_panel):
                    continue
                # Only top-level sections, not group boxes nested in one.
                outer = group.parentWidget()
                while outer is not None and outer is not self._control_panel:
                    if isinstance(outer, QGroupBox):
                        break
                    outer = outer.parentWidget()
                if isinstance(outer, QGroupBox):
                    continue
                bottom = group.mapTo(pane, QPoint(0, group.height())).y()
                gaps.append(bottom + 3)  # half the usual layout spacing
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not measure control panel sections: %s", exc)
            return top
        usable = [g for g in gaps if 0.4 * height <= g <= 0.7 * height]
        if not usable:
            return top
        return min(usable, key=lambda g: abs(g - top))

    def _sync_plot_actions(self, _checked: bool = True) -> None:
        """Keep at least one plot visible: the last visible plot's menu item
        is disabled so the display area can't be left empty."""
        actions = (self._spectrum_action, self._waterfall_action)
        shown = [a for a in actions if a.isChecked()]
        for action in actions:
            locked = len(shown) == 1 and action.isChecked()
            action.setEnabled(not locked)

    def _reset_layout(self) -> None:
        """Show both plots and restore the default splitter sizes."""
        for action in (self._spectrum_action, self._waterfall_action):
            action.setChecked(True)
        self._spectrum.setVisible(True)
        self._waterfall.setVisible(True)
        self._waterfall.set_frequency_axis_visible(False)
        self._apply_default_splitter_sizes()
        self._show_status_message("Layout reset")

    def _apply_default_geometry(self) -> None:
        """Size the window to 1400x900, or to fit a smaller screen."""
        width, height = 1400, 900
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(width, int(avail.width() * 0.95))
            height = min(height, int(avail.height() * 0.92))
        self.resize(max(width, self.minimumWidth()), max(height, self.minimumHeight()))

    # ------------------------------------------------------------------
    # Recording sync between toolbar, menu and control panel
    # ------------------------------------------------------------------

    def _record_kind(self) -> str:
        """What the Record buttons record: "iq" or "audio" (the Recording
        group's Record choice)."""
        try:
            return self._control_panel.record_kind()
        except AttributeError:  # pragma: no cover - partial windows in tests
            return "iq"

    def _on_record_button(self) -> None:
        """The toolbar's Record: records what the Recording group's Record
        choice says (Radio > Record I/Q or Record Audio)."""
        if self._record_kind() == "audio":
            self._record_audio_action.toggle()
        else:
            self._record_action.trigger()
        self._sync_record_buttons()

    def _on_panel_record_started(self, fmt: str) -> None:
        """Start recording from the control panel's Record button (I/Q or
        audio, as its Record choice says)."""
        if self._record_kind() == "audio":
            self._start_audio_recording()
        else:
            self._start_recording()
        self._sync_record_buttons()

    def _on_panel_record_stopped(self) -> None:
        """Stop recording from the control panel's Record button."""
        if self._record_kind() == "audio":
            self._stop_audio_recording()
        else:
            self._stop_recording()
        self._sync_record_buttons()

    def _on_panel_record_paused(self, paused: bool) -> None:
        """Pause or resume capturing from the control panel's Pause button."""
        if self._record_kind() == "audio":
            take = self._audio_take
            if take is None or bool(paused) == take.paused:
                return
            take.paused = bool(paused)
            self._update_audio_recording_status()
            self._show_status_message(
                "Audio recording paused" if paused else "Audio recording resumed",
                "warning" if paused else None,
                2500,
            )
            return
        paused = bool(paused) and self._recording
        if paused == self._recording_paused:
            return
        self._recording_paused = paused
        self._update_rec_clock()
        self._update_recording_status()
        self._show_status_message(
            "Recording paused" if paused else "Recording resumed",
            "warning" if paused else None,
            2500,
        )

    def _on_record_kind_changed(self, kind: str) -> None:
        """The Recording group's Record choice: the Record buttons now
        record that (and show its state)."""
        self._save_setting("recording/kind", kind)
        self._sync_record_buttons()
        if kind == "audio" and self._audio_take is not None:
            self._update_audio_recording_status()

    def _sync_record_ui(self, recording: bool) -> None:
        """The I/Q recording's state on Radio > Record I/Q, the REC badge
        and (when they record I/Q) the Record buttons."""
        self._record_action.blockSignals(True)
        self._record_action.setChecked(recording)
        self._record_action.blockSignals(False)
        self._recording_label.setVisible(recording)
        self._recording_info_label.setVisible(recording)
        self._sync_record_buttons()
        if recording:
            self._update_recording_status()
        else:
            self._sync_recording_armed()

    def _sync_record_buttons(self) -> None:
        """Show on the toolbar's and the Recording group's Record buttons
        whether what they record (I/Q or audio) is recording."""
        audio = self._record_kind() == "audio"
        on = (self._audio_take is not None) if audio else bool(self._recording)
        button = self._record_button
        button.blockSignals(True)
        button.setChecked(on)
        button.blockSignals(False)
        role = "danger" if on else ""
        if (button.property("role") or "") != role:
            set_role(button, role or None)
        button.setAccessibleName("Record audio" if audio else "Record I/Q")
        self._update_record_button_tip()
        panel = self._control_panel
        if panel._record_btn.isChecked() != on:
            panel.set_recording_state(on)
        if audio and self._audio_take is not None:
            panel.update_record_time(int(self._audio_take.seconds))
        self._sync_recording_armed()

    def _update_record_button_tip(self) -> None:
        if self._record_kind() == "audio":
            if self._audio_take is not None:
                tip = "Stop recording audio (Ctrl+Shift+A)"
            else:
                tip = (
                    "Record what you hear to a WAV file in the recordings "
                    "folder (Ctrl+Shift+A). The Recording group's Record choice "
                    "switches this button to I/Q."
                )
        elif self._recording:
            tip = "Stop recording (Ctrl+Shift+R)"
        else:
            tip = (
                "Record raw I/Q samples (Ctrl+Shift+R). "
                "Save them with File > Save Recording."
            )
        self._record_button.setToolTip(tip)

    # Panel labels -> ProtocolType for the live decoder.
    _DECODER_PROTOCOLS = {
        "POCSAG": "pocsag",
        "FLEX": "flex",
        "AX.25/APRS": "ax25",
        "ADS-B": "adsb",
        "ACARS": "acars",
        "RDS": "rds",
        "CW (Morse)": "cw",
    }

    def _on_decoder_protocol_changed(self, text: str) -> None:
        """Build (or clear) the live decoder for the panel's selected protocol.

        The live decoder (``sdr_module.dsp.live_decoders``) takes the raw
        blocks and picks out its own channel at the VFO, so it works
        whatever the Mode and Bandwidth are.
        """
        from ..dsp.live_decoders import create_live_decoder, live_decoder_class
        from ..dsp.protocols import ProtocolType

        proto_value = self._DECODER_PROTOCOLS.get(text)
        # The badge's station came from the old decoder (if it was RDS).
        self._show_rds_station(None)
        if proto_value is None:
            # "Off" (or unknown): no live decoder.
            self._decoder = None
            self._decoder_protocol = None
            self._decoder_panel.set_decoder_status(None)
            self._update_decoder_chip()
            return

        rate = self._device_sample_rate()
        self._decoder_panel.set_sample_rate(rate)
        # Remember the rate even if there is no decoder or its build fails,
        # so a later rate change retries it (and an unchanged rate doesn't).
        self._decoder_rate = rate
        if live_decoder_class(text) is None:
            # Listed, but without a live decoder (yet).
            self._decoder = None
            self._decoder_protocol = None
            self._decoder_panel.set_decoder_status(None)
            self._update_decoder_chip()  # (no decoder runs: no chip)
            return
        try:
            self._decoder = create_live_decoder(
                text, float(rate), self._decoder_offset_hz()
            )
            self._decoder_protocol = ProtocolType(proto_value)
        except Exception as exc:
            from ..dsp.live_decoders import LiveDecoderStatus

            self._decoder = None
            self._decoder_protocol = None
            problem = f"The {text} decoder could not start: {exc}"
            self._decoder_panel.set_decoder_status(
                LiveDecoderStatus(name=text, problem=problem)
            )
            self._update_decoder_chip()  # (no decoder runs: no chip)
            self._show_status_error(problem)  # (also logged)
            return
        self._decoder_error_logged = False
        status = self._decoder.status
        self._decoder_panel.set_decoder_status(status)
        self._update_decoder_chip()
        if status.problem:
            # E.g. ADS-B at 1.8 MS/s: say so instead of "listening" forever.
            logger.warning("%s decoder: %s", text, status.problem)
            remedy = f" {status.remedy}" if status.remedy else ""
            self._show_status_message(f"{status.problem}.{remedy}", "warning", 10000)

    def _decoder_offset_hz(self) -> float:
        """The tuned channel's offset from the LO (0 until the window tunes
        a VFO inside the span)."""
        try:
            return float(getattr(self, "_vfo_offset_hz", lambda: 0.0)())
        except Exception:  # pragma: no cover - defensive (partial windows)
            return 0.0

    def _decoder_retuned(self) -> bool:
        """Whether the tuned frequency moved (the LO, the VFO or both) since
        the last block the live decoder got."""
        try:
            tuned = self._current_frequency()
        except Exception:  # pragma: no cover - defensive (partial windows)
            return False
        last = getattr(self, "_decoder_tuned_hz", tuned)
        self._decoder_tuned_hz = tuned
        return tuned != last

    def _rebuild_decoder(self) -> None:
        """Recreate the live decoder for the current device's sample rate."""
        self._on_decoder_protocol_changed(self._decoder_panel.protocol() or "Off")

    def _run_decoder(self, samples: np.ndarray) -> None:
        """Feed a raw block to the live decoder and show what it decoded."""
        decoder = self._decoder
        if decoder is None or self._decoder_protocol is None:
            return
        if not self._decoder_panel._enabled_check.isChecked():
            return
        offset = self._decoder_offset_hz()
        if offset != getattr(decoder, "offset_hz", offset):
            decoder.set_offset(offset)
        if self._decoder_retuned():
            # What was measured on the old channel (signal, "nothing
            # decodes") says nothing about this one.
            decoder.channel_changed()
        try:
            messages = decoder.process(samples)
        except Exception as exc:  # pragma: no cover - defensive
            if not getattr(self, "_decoder_error_logged", False):
                self._decoder_error_logged = True
                logger.warning("The %s decoder failed: %s", decoder.name, exc)
            return
        for msg in messages:
            self._push_decoded_message(msg)
        self._decoder_panel.set_decoder_status(decoder.status)
        self._update_decoder_chip()
        if getattr(decoder, "name", "") == "RDS":
            # The Decoder's own RDS names the station (the background
            # listener rests meanwhile).
            self._show_rds_station(decoder.station)

    def _push_decoded_message(self, msg: Any, note: str = "") -> None:
        """Render one decoded message into the decoder panel.

        Rows use the protocol names of the panel's selector, and a message
        that failed its checks is added as invalid (tinted, counted under
        Invalid). A failed frame with nothing decoded in it (e.g. an AX.25
        CRC error, which noise produces about once a second) is left out.
        ``note`` is added to the message text (where it came from: File >
        Decode Recording names the file and the time in it).
        """
        try:
            address, content = self._describe_message(msg)
            if note and (address or content):
                content = f"{content}  ({note})" if content else f"({note})"
            valid = bool(getattr(msg, "valid", True))
            if not valid:
                if not (address or content):
                    logger.debug(
                        "Dropped an undecodable %s frame: %s",
                        type(msg).__name__,
                        getattr(msg, "error_message", ""),
                    )
                    return
                error = str(getattr(msg, "error_message", "") or "")
                if error:
                    content = f"{content}  ({error})" if content else error
            raw = getattr(msg, "raw_bits", b"")
            raw_text = raw[:32].hex() if isinstance(raw, (bytes, bytearray)) else ""
            from ..dsp.live_decoders import message_details

            self._decoder_panel.add_message(
                self._protocol_label(msg),
                address,
                content,
                valid,
                raw_text,
                message=msg,
                details=message_details(msg),
            )
            self._note_decoded()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not render decoded message: %s", exc)

    def _protocol_label(self, msg: Any) -> str:
        """The selector's name for a message's protocol (e.g. "AX.25/APRS")."""
        value = getattr(getattr(msg, "protocol", None), "value", "")
        if value == "aprs":
            value = "ax25"
        for label, proto in self._DECODER_PROTOCOLS.items():
            if proto == value:
                return label
        return str(value or type(msg).__name__).upper()

    @staticmethod
    def _describe_message(msg: Any) -> Tuple[str, str]:
        """(address, content) columns for a decoded message (shared with the
        command line: ``sdr_module.dsp.live_decoders.describe_message``)."""
        from ..dsp.live_decoders import describe_message

        return describe_message(msg)

    # ------------------------------------------------------------------
    # Decoder activity (unread count, status-bar chip) and the RDS station
    # ------------------------------------------------------------------

    def _setup_decoder_chip(self) -> None:
        """A status-bar chip while a decoder runs ("ADS-B · 37 msgs");
        clicking it opens the Decoder tab."""
        chip = QPushButton("")
        set_role(chip, "compact")
        chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        chip.setAccessibleName("Decoder activity")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.clicked.connect(lambda _checked=False: self._show_panel("decoder"))
        chip.hide()
        self.statusBar().addPermanentWidget(chip)
        self._decoder_chip = chip

    def _own(self, name: str) -> Any:
        """An attribute this window set, or None. (Reading a missing one
        through PyQt raises RuntimeError on a window whose Qt part is not
        built, as in tests that build parts of it.)"""
        return self.__dict__.get(name)

    def _decoder_page_visible(self) -> bool:
        """Whether the Decoder tab is on screen."""
        page = (self._own("_panel_pages") or {}).get("decoder")
        return bool(page is not None and page.isVisible() and not self.isMinimized())

    def _note_decoded(self) -> None:
        """A message was decoded: count it on the tab title if the Decoder
        tab is out of sight."""
        if self._own("_panel_pages") is None or self._decoder_page_visible():
            return
        self._decoder_unread += 1
        self._update_decoder_tab_title()

    def _update_decoder_tab_title(self) -> None:
        page = (self._own("_panel_pages") or {}).get("decoder")
        owner = (self._own("_panel_owners") or {}).get("decoder")
        if page is None or owner is None:
            return
        index = owner.indexOf(page)
        base = next(
            (t for k, t, _tip in self._panel_specs if k == "decoder"), "Decoder"
        )
        count = self._decoder_unread
        title = f"{base} ({count:,})" if count else base
        if owner.tabText(index) != title:
            owner.setTabText(index, title)
            owner.setTabToolTip(
                index,
                (
                    f"{count:,} new decoded message{'s' if count != 1 else ''}. "
                    if count
                    else ""
                )
                + next((tip for k, _t, tip in self._panel_specs if k == "decoder"), ""),
            )

    def _update_decoder_chip(self) -> None:
        """The status-bar chip: the protocol being decoded and what it has
        found, while one is (hidden otherwise)."""
        chip = self._own("_decoder_chip")
        if chip is None:
            return
        panel = self._decoder_panel
        proto = panel.protocol()
        if proto is None or self._decoder is None:
            if not chip.isHidden():
                chip.hide()
            return
        count = panel.count_for(proto)
        status = getattr(self._decoder, "status", None)
        detail = str(getattr(status, "detail", "") or "")
        parts = [proto]
        if detail:
            parts.append(detail)
        if not panel._enabled_check.isChecked():
            parts.append("paused")
        else:
            parts.append(f"{count:,} msg{'' if count == 1 else 's'}")
        # "&&": an "&" in a station's name is not a keyboard mnemonic.
        text = " · ".join(parts).replace("&", "&&")
        if chip.text() != text:
            chip.setText(text)
        # Its View > Panels shortcut (Ctrl+1 while it is the first panel).
        index = next(
            (i for i, spec in enumerate(self._panel_specs) if spec[0] == "decoder"),
            9,
        )
        tip = (
            f"Decoder: {proto}"
            + (" (chosen by Auto)" if panel.is_auto() else "")
            + f", {count:,} message{'' if count == 1 else 's'} since it started"
            + " (or since Clear). Click to open the Decoder tab"
            + (f" (Ctrl+{index + 1})." if index < 9 else ".")
        )
        if chip.toolTip() != tip:
            chip.setToolTip(tip)
        if chip.isHidden():
            chip.show()

    def _set_rds_auto(self, on: bool) -> None:
        """Radio > RDS Station Names."""
        self._settings.set("decoder/rds_auto", bool(on))
        if not on:
            self._rds_listener.reset()
            self._show_rds_station(None)

    def _listen_rds(self, channel: np.ndarray, chain: "_ReceiverChain") -> None:
        """Broadcast FM: decode the station's RDS from the receiver's own
        channel (Radio > RDS Station Names), unless the Decoder's RDS does."""
        action = self._own("_rds_auto_action")
        listener = self._own("_rds_listener")
        if listener is None:
            return
        panel = self._decoder_panel
        if (
            action is None
            or not action.isChecked()
            or chain.mode != "FM"
            or self._control_panel.get_fm_deviation() < _BROADCAST_DEVIATION_HZ
        ):
            if listener.decoder is not None:
                listener.reset()
                self._show_rds_station(None)
            return
        if (
            panel.protocol() == "RDS"
            and self._decoder is not None
            and panel.is_decoding()
        ):
            return  # the Decoder's RDS decoder names the station
        station = listener.process(
            channel, chain.channel_rate, self._current_frequency()
        )
        self._show_rds_station(station)

    def _show_rds_station(self, station: Any) -> None:
        """Name the station next to FREQ and on the radio tuner (None:
        no station known)."""
        badge = self._own("_rds_badge")
        if badge is None:
            return
        text = badge_text(station)
        tip = badge_tooltip(station) if text else ""
        if (text, tip) == self._rds_shown:
            return
        self._rds_shown = (text, tip)
        badge.setText(text)
        badge.setToolTip(tip)
        self._rds_badge_action.setVisible(bool(text))
        self._rds_station = station if text else None
        self._update_tuner_station()

    def _update_tuner_station(self) -> None:
        tuner = self._own("_radio_tuner")
        setter = getattr(tuner, "set_rds_text", None)
        if setter is not None:
            setter(tuner_text(self._own("_rds_station")))

    # ------------------------------------------------------------------
    # Control panel handlers
    # ------------------------------------------------------------------

    def _on_frequency_changed(self, freq_hz: float):
        """The tuned frequency is now ``freq_hz``: show it everywhere.

        Tuning itself (the LO, the device) is :meth:`_tune_to`'s; this
        updates what shows the tuned frequency.
        """
        # The readout, and the channel marked on the plots (their axes follow
        # the LO, see _retune_lo)
        self._freq_label.setText(format_frequency(freq_hz))
        self._update_passband()
        # The Bookmarks "Add" field offers the frequency being listened to;
        # the Decoder panel says whether it is where the protocol is heard.
        self._bookmarks_panel.set_current_frequency(freq_hz)
        self._decoder_panel.set_current_frequency(freq_hz)
        # An open radio tuner follows tuning (outside its broadcast bands it
        # shows that the receiver is elsewhere).
        tuner = self._radio_tuner
        if tuner is not None and tuner.isVisible():
            tuner.set_frequency(freq_hz)  # does not emit frequency_changed
        # Another station: its RDS name is not known yet.
        self._show_rds_station(None)
        self._reset_signal_meter_peak()
        # Ham ID: where Send ID Now would transmit, and whether it may; QRP:
        # the legal limit here.
        if HAS_HAM_RADIO:
            self._qrp_panel.set_station(freq_hz)
        self._tx_station.refresh()
        self._tuner_retuned()
        # Identify listens afresh; Track Signal starts over from here (T10).
        signal_tools = self._own("_signal_tools")
        if signal_tools is not None:
            signal_tools.frequency_changed(freq_hz)

        logger.debug(f"Frequency changed to {freq_hz/1e6:.3f} MHz")

    # ------------------------------------------------------------------
    # Tuning: the LO and the VFO
    # ------------------------------------------------------------------

    def _on_panel_frequency_changed(self, freq_hz: float) -> None:
        """The control panel asks for ``freq_hz`` (typed, the Frequency
        field's arrows, a quick-tune button, Apply Preset)."""
        self._tune_to(freq_hz)

    def _keep_centered(self) -> bool:
        """Radio > Keep Tuned Frequency Centered: every tune retunes the LO."""
        action = getattr(self, "_keep_centered_action", None)
        return bool(action is not None and action.isChecked())

    def _set_keep_centered(self, on: bool) -> None:
        on = bool(on)
        self._save_setting("tuning/keep_centered", on)
        if on:
            self._ensure_vfo_fits()  # center now, as every tune will
            text = (
                "Keep tuned frequency centered: the receiver retunes for every "
                "new frequency"
            )
        else:
            text = (
                "Tuning now moves within the span; the receiver retunes only "
                "near its edges"
            )
        self._show_status_message(text, None, 4000)

    def _channel_fits(self, offset_hz: float, rate: Optional[float] = None) -> bool:
        """Whether the tuned channel, ``offset_hz`` from the LO, lies inside
        the usable inner part (``_USABLE_SPAN_FRACTION``) of a span of
        ``rate`` Hz (the device's). At the LO itself it always does (a
        channel wider than that is best centered)."""
        if abs(offset_hz) < 0.5:
            return True
        rate = float(rate or self._device_sample_rate())
        low, high = _channel_edges(
            self._control_panel._demod_combo.currentText(),
            self._channel_bandwidth(),
            rate,
        )
        half = 0.5 * _USABLE_SPAN_FRACTION * rate
        return -half <= offset_hz + low and offset_hz + high <= half

    def _device_model_name(self) -> str:
        """``"RTL-SDR"``, ``"HackRF One"`` or the device's name, for messages."""
        dev = self._device
        if dev is None:
            return "The receiver"
        for cls in type(dev).__mro__:
            if cls.__name__ in _DEVICE_MODELS:
                return _DEVICE_MODELS[cls.__name__]
        return self._device_display_name()

    def _device_tuning_range(self) -> Optional[Tuple[float, float]]:
        """The connected device's tuning range (its driver's spec), if any."""
        if self._device is None:
            return None
        spec = getattr(self._device, "spec", None)
        low = getattr(spec, "freq_min", None)
        high = getattr(spec, "freq_max", None)
        if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            return None
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            return None
        return float(low), float(high)

    def _tuning_limits(self) -> Tuple[float, float]:
        """The frequencies that can be tuned now, in Hz."""
        return self._device_tuning_range() or (MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ)

    def _range_text(self) -> str:
        """``"RTL-SDR tunes 0.5–1766 MHz"``, for messages and the tooltip."""
        low, high = self._tuning_limits()
        if self._device_tuning_range() is None:
            return f"the receiver tunes {format_mhz_range(low, high)}"
        return f"{self._device_model_name()} tunes {format_mhz_range(low, high)}"

    def _apply_tuning_range(self) -> None:
        """Let the Frequency field accept the connected device's range (its
        tooltip and messages say what it is)."""
        low, high = self._tuning_limits()
        text = self._range_text() if self._device_tuning_range() else ""
        self._control_panel._freq_input.set_tuning_range(low, high, text)

    def _device_frequency(self) -> Optional[float]:
        """The frequency the device is actually tuned to, if it says."""
        dev = self._device
        if dev is None:
            return None
        for value in (
            getattr(getattr(dev, "state", None), "frequency", None),
            getattr(dev, "frequency", None),
        ):
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                return float(value)
        return None

    def _retune_lo(self, lo_hz: float) -> bool:
        """Tune the device (if any) to ``lo_hz`` and center the plots on it.

        False, leaving everything as it was, when the device refuses (its
        ``set_frequency`` returns False or raises: out of its range, or it
        can't right now).
        """
        lo = float(lo_hz)
        dev = self._device
        if dev is not None:
            try:
                ok = dev.set_frequency(lo)
            except Exception as e:
                logger.error(
                    f"Could not tune the device to {format_frequency(lo)}: {e}"
                )
                ok = False
            if ok is False:
                return False
        moved = abs(lo - self._lo_hz) >= 0.5
        self._lo_hz = lo
        self._spectrum.set_center_freq(lo)
        self._waterfall.set_center_freq(lo)
        if moved and isinstance(dev, SDRDevice):
            self._drop_stale_samples(dev)
        return True

    def _drop_stale_samples(self, dev: SDRDevice) -> None:
        """A hardware receiver was retuned: what it took before holds the
        old frequency. Drop the transfers it queued, and have the plots
        skip the one in flight and the tuner's settling, so no waterfall
        row or held peak shows the old frequency's signals at the new one.
        The dropped transfers count as received (see ``_realtime``)."""
        for _ in range(_RETUNE_MAX_DROPPED):
            try:
                block = dev.read_samples(DISPLAY_BLOCK, timeout=0.0)
            except Exception as e:  # pragma: no cover - driver specific
                logger.debug(f"Could not drop the queued samples: {e}")
                break
            if block is None or len(block) == 0:
                break
            self._note_block(len(block))
        try:
            streaming = bool(dev.state.is_streaming)
        except Exception:  # pragma: no cover - defensive
            streaming = False
        if streaming:
            self._plot_skip_samples = _RETUNE_STALE_SAMPLES
            self._plot_skip_until = time.monotonic() + _RETUNE_SETTLE_S

    def _tune_to(self, freq_hz: float, recenter: bool = False) -> bool:
        """Tune to ``freq_hz`` by the rules of :meth:`set_frequency`.

        ``recenter`` puts the LO on the tuned frequency whatever the span.
        True when tuned; otherwise the reason is in the status bar and
        everything stays on the frequency tuned.
        """
        try:
            freq = float(freq_hz)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(freq):
            return False
        if self._tx_busy():
            # The HackRF sends where it is: it can't move mid-transmission.
            return self._refuse_tuning(
                "Can't tune while transmitting: press Stop (or wait for the "
                "transmission to end) first."
            )
        low, high = self._tuning_limits()
        if not low <= freq <= high:
            return self._refuse_tuning(
                f"{format_frequency(freq)} is out of range: {self._range_text()}."
            )
        lo = self._lo_hz
        lock = self._lo_lock
        if lock is not None:
            center, span, why = lock
            if not self._channel_fits(freq - center, rate=span):
                covered = format_mhz_range(center - span / 2, center + span / 2)
                if abs(freq - center) < span / 2:
                    # Inside, but the channel (the Bandwidth) sticks out.
                    width = self._control_panel._bw_combo.currentText()
                    return self._refuse_tuning(
                        f"A {width} channel at {format_frequency(freq)} doesn't "
                        f"fit inside {why} ({covered}): choose a narrower "
                        "Bandwidth, or tune nearer its middle."
                    )
                return self._refuse_tuning(
                    f"{format_frequency(freq)} is outside {why}, which covers "
                    f"{covered}."
                )
            lo = center
        elif recenter or self._keep_centered() or not self._channel_fits(freq - lo):
            lo = freq
        if abs(lo - self._lo_hz) >= 0.5 and not self._retune_lo(lo):
            return self._refuse_tuning(
                f"{self._device_model_name()} didn't tune to "
                f"{format_frequency(lo)}; it stays on "
                f"{format_frequency(self._lo_hz)}. Details are in Tools > "
                "Error History.",
                device_problem=True,
            )
        self._finish_tune(freq)
        return True

    def _finish_tune(self, freq_hz: float) -> None:
        """``freq_hz`` is tuned: the Frequency field, every view of it, and
        (after a moment on it) the tuning history."""
        self._tuned_hz = float(freq_hz)
        self._control_panel.set_frequency(freq_hz)  # no signal
        self._on_frequency_changed(self._current_frequency())
        self._note_tuned()

    def _refuse_tuning(self, reason: str, device_problem: bool = False) -> bool:
        """A frequency can't be tuned: say why, and put the Frequency field
        (and anything that showed the new frequency) back on the one tuned.

        Only a device that said no is logged (Tools > Error History); a
        frequency out of range is just explained.
        """
        self._control_panel.set_frequency(self._tuned_hz)
        self._on_frequency_changed(self._current_frequency())
        if device_problem:
            self._show_status_error(reason)
        else:
            self._show_status_message(reason, "danger", 6000)
        return False

    def _ensure_vfo_fits(self) -> None:
        """Center the LO on the tuned frequency when the tuned channel no
        longer fits the usable span (a wider channel, a smaller sample
        rate), or Keep Tuned Frequency Centered asks for it."""
        if self._lo_lock is not None:
            return
        offset = self._vfo_offset_hz()
        if abs(offset) < 0.5:
            return
        if self._keep_centered() or not self._channel_fits(offset):
            self._tune_to(self._current_frequency(), recenter=True)

    def _lock_lo(self, center_hz: float, span_hz: float, why: str) -> None:
        """Fix the LO at ``center_hz`` with a ``span_hz`` span (e.g. while a
        recording plays back: ``why`` = "the recording"). Tuning then moves
        only the VFO, and frequencies outside the span are refused."""
        self._lo_lock = (float(center_hz), float(span_hz), str(why))
        self._lo_hz = float(center_hz)
        self._spectrum.set_center_freq(center_hz)
        self._waterfall.set_center_freq(center_hz)
        if not self._channel_fits(self._vfo_offset_hz(), rate=span_hz):
            self._finish_tune(center_hz)
        else:
            self._update_passband()
        self._tx_station.refresh()  # (no transmitting during playback)

    def _unlock_lo(self) -> None:
        """Let the LO follow tuning again (see :meth:`_lock_lo`)."""
        self._lo_lock = None
        self._ensure_vfo_fits()
        self._tx_station.refresh()

    def _sync_device_tuning(self) -> str:
        """Bring a device just connected to the plots' LO and the tuned
        frequency (recentering when the channel doesn't fit its span).

        When it can't tune there (out of its range, or refused) it stays
        where it is: the plots and the tuned frequency go there too, and the
        returned note says why ("" when all went to plan).
        """
        self._apply_tuning_range()
        wanted = self._current_frequency()
        low, high = self._tuning_limits()
        if low <= wanted <= high:
            lo = self._lo_hz
            if (
                not low <= lo <= high
                or self._keep_centered()
                or not self._channel_fits(wanted - lo)
            ):
                lo = wanted
            if self._retune_lo(lo):
                self._finish_tune(wanted)
                return ""
            problem = f"it didn't tune to {format_frequency(lo)}"
        else:
            problem = (
                f"{format_frequency(wanted)} is out of its range "
                f"({self._range_text()})"
            )
        actual = self._device_frequency()
        if actual is None or not low <= actual <= high:
            actual = min(max(wanted, low), high)
        if not self._retune_lo(actual):
            # The device says where it is; show that.
            self._lo_hz = actual
            self._spectrum.set_center_freq(actual)
            self._waterfall.set_center_freq(actual)
        self._finish_tune(actual)
        return f"Tuned to {format_frequency(actual)}: {problem}."

    def _nudge_frequency(self, offset_hz: float) -> None:
        """Tune ``offset_hz`` away from the tuned frequency."""
        self._tune_to(self._current_frequency() + float(offset_hz))

    def _tune_steps(self, steps: int) -> None:
        """Tune ``steps`` tuning steps up (negative: down): Left/Right, the
        mouse wheel over the plots."""
        steps = int(steps)
        if steps:
            self._tune_to(
                step_frequency(self._current_frequency(), self._step_hz, steps)
            )

    # -- tuning step ------------------------------------------------------------

    def _step_mode(self) -> str:
        """The key the current mode's tuning step is remembered under."""
        panel = self._control_panel
        return step_mode_key(panel.get_demod_mode(), panel.get_fm_deviation())

    def _load_step_for_mode(self) -> None:
        """Use the step the current mode remembers (or starts with)."""
        key = self._step_mode()
        if key == self._step_key:
            return
        self._step_key = key
        saved = self._settings.get_float(f"tuning/step/{key}", default_step_hz(key))
        self._apply_step(nearest_step(saved))

    def _apply_step(self, step_hz: float) -> None:
        """Show and use ``step_hz`` everywhere that tunes by the step."""
        self._step_hz = float(step_hz)
        self._receiver_bar.set_step(step_hz)
        action = self._step_actions.get(step_hz)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self._control_panel._freq_input.set_step_hz(step_hz)
        for plot in (self._spectrum, self._waterfall):
            snap = getattr(plot, "set_click_snap", None)
            if callable(snap):
                snap(step_hz, self._step_key)

    def _set_step(self, step_hz: float) -> None:
        """The user picked a step (receiver bar, Radio > Tuning Step, [ or
        ]): use it, and remember it for the current mode."""
        step = nearest_step(step_hz)
        if self._step_key != self._step_mode():
            self._load_step_for_mode()
        self._apply_step(step)
        self._save_setting(f"tuning/step/{self._step_key}", step)

    def _change_step(self, delta: int) -> None:
        """One step choice finer (-1) or coarser (+1): [ and ]."""
        index = step_index(self._step_hz) + int(delta)
        index = min(max(index, 0), len(STEP_CHOICES_HZ) - 1)
        step = STEP_CHOICES_HZ[index]
        if step != self._step_hz:
            self._set_step(step)
        mode = STEP_MODE_NAMES.get(self._step_key, self._step_key)
        self._show_status_message(
            f"Tuning step {format_step(self._step_hz)} (for {mode})", None, 2000
        )

    # -- tuning history -----------------------------------------------------------

    def _note_tuned(self) -> None:
        """A frequency was tuned: it joins the history once the receiver
        stays on it a moment (stepping through a band doesn't add each
        step)."""
        self._history_timer.start(_HISTORY_DWELL_MS)

    def _commit_history_entry(self) -> None:
        """Record where the receiver is (frequency and mode) now."""
        self._history_timer.stop()
        mode, deviation = self._listening_mode()
        entry = TunedEntry(self._current_frequency(), mode, deviation)
        if self._tuning_history.visit(entry):
            self._save_setting("tuning/recent", self._tuning_history.to_json())
            self._rebuild_recent_menu()
        self._update_history_actions()

    def _commit_pending_history(self) -> None:
        if self._history_timer.isActive():
            self._commit_history_entry()

    def _tune_entry(self, entry: "TunedEntry") -> bool:
        """Tune to a history or recent entry, in its mode."""
        if not self.set_frequency(entry.freq_hz):
            return False
        if entry.mode and self._listening_mode() != (entry.mode, entry.deviation):
            self._set_demod_mode(
                entry.mode,
                entry.deviation or None,
                self._bandwidth_for_mode(entry.mode, entry.deviation),
            )
        return True

    def _entry_text(self, entry: "TunedEntry") -> str:
        text = format_frequency(entry.freq_hz)
        if entry.mode:
            text += f" ({self._mode_text(entry.mode)})"
        return text

    def _tune_back(self) -> None:
        """Radio > Tune Back (Alt+Left): the frequency before this one."""
        self._commit_pending_history()
        entry = self._tuning_history.back()
        if entry is None:
            self._show_status_message("Nothing to go back to yet", None, 2500)
        elif self._tune_entry(entry):
            self._show_status_message(
                f"Back to {self._entry_text(entry)}", "info", 2500
            )
        else:
            # Refused (the status bar says why): the history stays where the
            # receiver is, so Forward and the next visit don't lose entries.
            self._tuning_history.forward()
        self._update_history_actions()

    def _tune_forward(self) -> None:
        """Radio > Tune Forward (Alt+Right): undo Tune Back."""
        self._commit_pending_history()
        entry = self._tuning_history.forward()
        if entry is None:
            self._show_status_message("Nothing to go forward to", None, 2500)
        elif self._tune_entry(entry):
            self._show_status_message(
                f"Forward to {self._entry_text(entry)}", "info", 2500
            )
        else:
            self._tuning_history.back()  # refused: stay where the receiver is
        self._update_history_actions()

    def _tune_recent(self, entry: "TunedEntry") -> None:
        """Radio > Recent Frequencies > an entry."""
        self._commit_pending_history()
        if self._tune_entry(entry):
            self._show_status_message(
                f"Tuned to {self._entry_text(entry)}", "info", 2500
            )

    def _clear_recent(self) -> None:
        self._tuning_history.clear_recent()
        self._save_setting("tuning/recent", self._tuning_history.to_json())
        self._rebuild_recent_menu()
        self._update_history_actions()
        self._show_status_message("Recent frequencies cleared", None, 2500)

    def _rebuild_recent_menu(self) -> None:
        """Radio > Recent Frequencies: the last frequencies tuned."""
        menu = self._recent_menu
        # Not menu.clear(): that deletes the actions at once, and this runs
        # from the triggered handler of one of them (an entry or Clear),
        # which Qt still uses after the handler returns.
        for old in menu.actions():
            menu.removeAction(old)
            if old.parent() is menu:
                old.deleteLater()
        entries = self._tuning_history.recent()
        if not entries:
            empty = menu.addAction("No recent frequencies yet")
            empty.setEnabled(False)
        for entry in entries:
            text = self._entry_text(entry)
            act = menu.addAction(text)
            act.setToolTip(f"Tune to {text}")
            act.setStatusTip(f"Tune to {text}")
            act.triggered.connect(lambda _c=False, e=entry: self._tune_recent(e))
        menu.addSeparator()
        clear = menu.addAction("&Clear Recent Frequencies")
        clear.setStatusTip("Forget the recent frequencies")
        clear.setEnabled(bool(entries))
        clear.triggered.connect(lambda _c=False: self._clear_recent())

    def _update_history_actions(self) -> None:
        """Enable Tune Back/Forward when there is somewhere to go, and say
        where in their tooltips."""
        history = self._tuning_history
        for action, can, delta, verb in (
            (self._back_action, history.can_go_back(), -1, "Go back to"),
            (self._forward_action, history.can_go_forward(), 1, "Go forward to"),
        ):
            action.setEnabled(can)
            entry = history.peek(delta) if can else None
            if entry is not None:
                tip = f"{verb} {self._entry_text(entry)}"
            elif delta < 0:
                tip = "Go back to the frequency you were on before (none yet)"
            else:
                tip = "Go forward again after Tune Back"
            action.setToolTip(tip)
            action.setStatusTip(tip)

    def _on_gain_changed(self, gain_db: float):
        """Apply a manual gain (the panel only emits this while AGC is off)."""
        if self._device:
            try:
                self._device.set_gain(gain_db)
            except Exception as e:
                self._show_status_error(f"Could not set the gain: {e}")
        if HAS_HAM_RADIO:
            self._signal_meter_panel.set_gain(self._signal_meter_gain())
        self._device_controls.gain_applied()  # remembered for the device type
        logger.debug(f"Gain changed to {gain_db:.1f} dB")

    def _apply_controls_to_device(self) -> str:
        """Bring a newly connected device to what the controls show:
        frequency (the Frequency field takes the device's range), then the
        AGC mode, then the manual gain (only while AGC is off; with AGC on
        the tuner owns the gain), and a HackRF's baseband filter to its
        sample rate.

        Returns a note for the user when the tuned frequency had to change
        (the device can't tune it), else "".
        """
        # The device's own gain controls, Tuner AGC where it has it and its
        # front end (ppm, direct sampling, bias tee) as that kind of device
        # last had them, set before tuning (direct sampling depends on where).
        device_note = self._device_controls.configure()
        note = self._sync_device_tuning()
        agc = self._control_panel.is_agc_enabled()
        self._on_agc_changed(agc)
        if not agc:
            self._device_controls.apply_manual_gain()
        self._apply_baseband_filter()
        return " ".join(text for text in (note, device_note) if text)

    def _on_bandwidth_changed(self, bw_hz: float):
        """The channel width: it sets the demodulator's channel filter and
        what LEVEL, the squelch and the S-meter measure.

        It is not sent to hardware: an RTL-SDR's or HackRF's
        ``set_bandwidth`` sets the analog filter in front of the ADC, which
        must pass the whole span (a HackRF rounded 25 kHz up to its
        narrowest filter, 1.75 MHz, and darkened most of a 10 MHz span).
        A HackRF's filter follows its sample rate instead
        (:meth:`_apply_baseband_filter`).
        """
        self._save_setting("bandwidth", self._control_panel._bw_combo.currentText())
        dev = self._device
        if dev is not None and self._forwards_channel_width(dev):
            rate = self._device_sample_rate()
            try:
                dev.set_bandwidth(bw_hz)
            except Exception as e:
                self._show_status_error(f"Could not set the bandwidth: {e}")
            if self._device_sample_rate() != rate:
                self._refresh_state_ui()  # axes, rate label, decoder
        self._rx_chain = None  # rebuilt for the new channel
        self._update_passband()
        self._ensure_vfo_fits()  # a wider channel may need the LO centered
        self._reset_signal_meter_peak()
        if self._info_panel.isVisible():
            self._refresh_info_panel()
        logger.debug(f"Bandwidth changed to {bw_hz/1e3:.1f} kHz")

    @staticmethod
    def _forwards_channel_width(device: Any) -> bool:
        """Whether ``device.set_bandwidth`` gets the channel width.

        Never for a hardware driver (RTL-SDR, HackRF): their filter is the
        analog anti-alias filter for the whole span, not the channel.
        """
        return not isinstance(device, SDRDevice)

    # A HackRF's baseband filter as a fraction of its sample rate, as
    # hackrf_transfer sets it: most of the span, while keeping out what
    # would alias into it.
    _BASEBAND_FILTER_FRACTION = 0.75

    @staticmethod
    def _has_baseband_filter(device: Any) -> bool:
        """True for a HackRF, whose baseband filter follows its sample rate."""
        try:
            from ..devices.hackrf import HackRFDevice
        except Exception:  # pragma: no cover - defensive
            return False
        return isinstance(device, HackRFDevice)

    def _apply_baseband_filter(self) -> None:
        """Set a HackRF's baseband filter to 0.75x its sample rate (the
        driver rounds it to a width the hardware has, 1.75-28 MHz)."""
        dev = self._device
        if dev is None or not self._has_baseband_filter(dev):
            return
        width = self._BASEBAND_FILTER_FRACTION * self._device_sample_rate()
        try:
            if dev.set_bandwidth(width) is False:
                logger.warning("Could not set the HackRF baseband filter")
        except Exception as e:
            logger.warning(f"Could not set the HackRF baseband filter: {e}")

    def _save_setting(self, key: str, value: Any) -> None:
        """Persist a control's value (not while restoring them)."""
        if not self._restoring:
            self._settings.set(key, value)

    def _on_squelch_changed(self, db: float):
        """A squelch threshold set by hand (the slider, or dragged on the
        level meter)."""
        self._squelch_db = float(db)
        self._squelch.manual_db = self._squelch_db
        self._save_setting("squelch_db", self._squelch_db)

    def _on_meter_threshold(self, db: float) -> None:
        """The threshold was dragged on the toolbar meter: the squelch is on
        and set by hand there (the slider follows)."""
        panel = self._control_panel
        if not panel.is_squelch_enabled():
            panel.set_squelch_enabled(True)
        panel.set_squelch_db(db)

    def _on_squelch_enabled(self, enabled: bool) -> None:
        self._squelch.enabled = bool(enabled)
        self._save_setting("squelch/enabled", bool(enabled))
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _on_squelch_auto(self, auto: bool) -> None:
        self._squelch.auto = bool(auto)
        self._save_setting("squelch/auto", bool(auto))

    def _connect_audio_controls(self) -> None:
        """The Demodulation, Audio and I/Q controls of the control panel."""
        panel = self._control_panel
        panel.squelch_enabled_changed.connect(self._on_squelch_enabled)
        panel.squelch_auto_changed.connect(self._on_squelch_auto)
        panel.filter_memory_changed.connect(self._save_filter_memory)
        panel.cw_pitch_changed.connect(self._on_cw_pitch_changed)
        panel.noise_reduction_changed.connect(self._on_noise_reduction_changed)
        panel.audio_agc_changed.connect(self._on_audio_agc_changed)
        panel.tone_squelch_changed.connect(self._on_tone_squelch_changed)
        panel.iq_options_changed.connect(self._on_iq_options_changed)

    def _save_filter_memory(self) -> None:
        try:
            text = json.dumps(self._control_panel.filter_memory())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        self._save_setting("audio/mode_filters", text)

    def _on_cw_pitch_changed(self, hz: int) -> None:
        self._audio_options.cw_pitch_hz = float(hz)
        self._save_setting("audio/cw_pitch", int(hz))
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _on_noise_reduction_changed(self, level: str) -> None:
        self._audio_options.noise_reduction = make_noise_reduction(level, _AUDIO_RATE)
        self._save_setting("audio/noise_reduction", level)

    def _on_audio_agc_changed(self, speed: str) -> None:
        self._audio_options.agc_speed = speed
        self._save_setting("audio/agc_speed", speed)

    def _on_tone_squelch_changed(self, hz: float) -> None:
        self._tone_squelch.tone_hz = float(hz) if hz else None
        self._save_setting("audio/tone_squelch", float(hz))

    def _on_iq_options_changed(self, remove_dc: bool, balance_iq: bool) -> None:
        conditioner = self._iq_conditioner
        conditioner.remove_dc = bool(remove_dc)
        conditioner.balance_iq = bool(balance_iq)
        conditioner.reset()
        self._save_setting("audio/remove_dc", bool(remove_dc))
        self._save_setting("audio/balance_iq", bool(balance_iq))

    def _restore_audio_settings(self) -> None:
        """Squelch, per-mode filters and the Audio / I/Q options."""
        panel = self._control_panel
        settings = self._settings
        raw = settings.get_str("audio/mode_filters", "")
        if raw:
            try:
                memory = json.loads(raw)
            except ValueError:
                logger.debug("Ignoring unreadable saved mode filters: %r", raw)
            else:
                if isinstance(memory, dict):
                    panel.set_filter_memory(memory)
        panel.set_squelch_auto(settings.get_bool("squelch/auto", True))
        panel.set_squelch_enabled(settings.get_bool("squelch/enabled", True))
        panel.set_cw_pitch(settings.get_int("audio/cw_pitch", DEFAULT_CW_PITCH_HZ))
        panel.set_noise_reduction(settings.get_str("audio/noise_reduction", "off"))
        panel.set_audio_agc_speed(settings.get_str("audio/agc_speed", "medium"))
        panel.set_tone_squelch_hz(settings.get_float("audio/tone_squelch", 0.0))
        panel.set_iq_options(
            settings.get_bool("audio/remove_dc", False),
            settings.get_bool("audio/balance_iq", False),
        )
        # (The panel's signals applied these; make sure even unchanged
        # defaults are in effect.)
        self._squelch.enabled = panel.is_squelch_enabled()
        self._squelch.auto = panel.is_squelch_auto()
        self._squelch.manual_db = self._squelch_db
        self._audio_options.cw_pitch_hz = float(panel.cw_pitch())
        self._audio_options.agc_speed = panel.audio_agc_speed()
        self._audio_options.noise_reduction = make_noise_reduction(
            panel.noise_reduction(), _AUDIO_RATE
        )
        hz = panel.tone_squelch_hz()
        self._tone_squelch.tone_hz = hz if hz else None
        remove_dc, balance_iq = panel.iq_options()
        self._iq_conditioner.remove_dc = remove_dc
        self._iq_conditioner.balance_iq = balance_iq

    def _persist_audio_settings(self) -> None:
        panel = self._control_panel
        settings = self._settings
        settings.set("squelch/enabled", panel.is_squelch_enabled())
        settings.set("squelch/auto", panel.is_squelch_auto())
        settings.set("audio/mode_filters", json.dumps(panel.filter_memory()))
        settings.set("audio/cw_pitch", panel.cw_pitch())
        settings.set("audio/noise_reduction", panel.noise_reduction())
        settings.set("audio/agc_speed", panel.audio_agc_speed())
        settings.set("audio/tone_squelch", panel.tone_squelch_hz())
        remove_dc, balance_iq = panel.iq_options()
        settings.set("audio/remove_dc", remove_dc)
        settings.set("audio/balance_iq", balance_iq)

    def _bandwidth_description(self) -> str:
        """The Info tab's Bandwidth: the width and, in SSB and CW, the
        passband it gives (``"2.7 kHz (USB, 200-2900 Hz)"``)."""
        panel = self._control_panel
        mode = panel.get_demod_mode()
        width = panel._bw_combo.currentText()
        if mode in ("USB", "LSB"):
            low, high = sorted(abs(e) for e in self._passband_edges())
            return f"{width} ({mode}, {low:.0f}-{high:.0f} Hz)"
        if mode == "CW":
            return f"{width} (CW, {panel.cw_pitch()} Hz tone)"
        return width

    def _squelch_description(self, running: bool) -> Tuple[str, Optional[str]]:
        """The Info tab's Squelch: on or off, the threshold, open or not."""
        mode = self._control_panel.get_demod_mode()
        gate = self._squelch
        if mode not in _AUDIBLE_MODES:
            return "Not used (no audio in this mode)", "muted"
        tone_hz = self._tone_squelch.tone_hz if mode == "FM" else None
        tone = f", tone {tone_hz:.1f} Hz" if tone_hz else ""
        if not gate.enabled:
            return f"Off (everything plays){tone}", None
        threshold = gate.last_threshold_db
        if not running or threshold is None:
            if gate.auto:
                return f"Auto (noise + {gate.margin_db:.0f} dB){tone}", None
            return f"{gate.manual_db:.0f} dBFS (set by hand){tone}", None
        how = "Auto" if gate.auto else "set by hand"
        if gate.is_open and not self._tone_blocked:
            return f"{threshold:.0f} dBFS ({how}), open{tone}", "success"
        if gate.is_open:
            return f"{threshold:.0f} dBFS ({how}), open{tone} (not heard)", None
        return f"{threshold:.0f} dBFS ({how}), closed{tone}", None

    def _level_description(self) -> str:
        level = self._last_peak_db
        text = f"{level:.1f} dBFS"
        noise = self._level_meter.noise_floor()
        if level is not None and noise is not None:
            text += f" (SNR {level - noise:.0f} dB)"
        return text

    def _on_agc_changed(self, enabled: bool):
        """Handle AGC toggle."""
        self._agc_enabled = bool(enabled)
        self._settings.set("agc_enabled", self._agc_enabled)
        if HAS_HAM_RADIO:
            self._signal_meter_panel.set_gain(self._signal_meter_gain())
        accepted = True
        if self._device and hasattr(self._device, "set_gain_mode"):
            try:
                # set_gain_mode(auto: bool). Passing the strings "auto"/"manual"
                # made "manual" truthy, so unticking AGC left AGC on.
                accepted = self._device.set_gain_mode(self._agc_enabled) is not False
            except Exception as e:
                logger.warning(f"Could not switch the device's automatic gain: {e}")
                accepted = False
        # A refusal (a HackRF has no AGC) is reported and AGC goes back off;
        # the readout shows what an automatic gain chose.
        self._device_controls.agc_changed(self._agc_enabled, accepted)

    def _on_demod_changed(self, mode: str):
        """Handle demodulator selection change. The control panel has
        already applied the mode's Bandwidth choices and the width (and FM
        deviation) last used with it; say what they are."""
        self._save_setting("demod_mode", mode)
        self._decoder_panel.set_demod_mode(mode)  # Auto: Morse code in CW
        self._refresh_audio_output(announce=not self._restoring)
        self._update_passband()
        # Each mode tunes by its own step; its channel may need the LO
        # centered; the tuning history remembers the mode.
        self._load_step_for_mode()
        self._ensure_vfo_fits()
        self._note_tuned()
        if not self._restoring and mode in _AUDIBLE_MODES:
            self._show_status_message(f"{mode}: {self._filter_summary()}", None, 3000)
        if mode != "FM" and self._sstv_listening():
            self._hint_sstv_needs_fm()
        if HAS_HAM_RADIO:
            self._signal_meter_panel.set_demod_mode(mode)
            self._qrp_panel.set_demod_mode(mode)  # QRP's Auto mode follows
            self._reset_signal_meter_peak()
            self._sstv_panel.show_fm_needed(
                self._mode_text(mode)
                if mode != "FM" and self._sstv_listening()
                else None
            )
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _filter_summary(self) -> str:
        """What the current mode listens with, e.g. ``"2.7 kHz filter,
        200-2900 Hz"`` or ``"200 kHz, 75 kHz deviation (broadcast)"``."""
        panel = self._control_panel
        mode = panel.get_demod_mode()
        width = panel._bw_combo.currentText()
        if mode in ("USB", "LSB"):
            low, high = sorted(abs(e) for e in self._passband_edges())
            return f"{width} filter, {low:.0f}-{high:.0f} Hz"
        if mode == "CW":
            return f"{width} filter, {panel.cw_pitch()} Hz tone"
        if mode == "FM":
            return f"{width}, {panel._fm_dev_combo.current_label()} deviation"
        return width

    def _passband_edges(self) -> Tuple[float, float]:
        """The demodulated channel's edges (Hz from the tuned frequency)."""
        return _channel_edges(
            self._control_panel._demod_combo.currentText(),
            self._channel_bandwidth(),
            self._device_sample_rate(),
        )

    def _set_demod_mode(
        self,
        mode: str,
        fm_deviation: Optional[str] = None,
        bandwidth: Optional[str] = None,
    ) -> None:
        """Select a demodulation mode (and, for FM, a deviation; and a
        channel bandwidth) in the control panel; its signals apply them.

        The mode goes first: it sets the Bandwidth choices (and the width
        and deviation last used with it), which ``fm_deviation`` and
        ``bandwidth`` then override. A width the mode doesn't offer picks
        the narrowest of its choices that holds it (else the widest)."""
        panel = self._control_panel
        index = panel._demod_combo.findText(mode)
        if index >= 0:
            panel._demod_combo.setCurrentIndex(index)
        if mode == "FM" and fm_deviation:
            index = panel._fm_dev_combo.findText(fm_deviation)
            if index >= 0:
                panel._fm_dev_combo.setCurrentIndex(index)
        if bandwidth:
            combo = panel._bw_combo
            index = combo.findText(bandwidth)
            if index < 0:
                width = _parse_hz(bandwidth)
                if width:
                    nearest = closest_bandwidth(panel.bandwidth_choices(), width)
                    index = combo.findText(nearest)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _on_license_changed(self, license_class: Any) -> None:
        """Remember the license class (it gates where TX is allowed)."""
        value = getattr(license_class, "value", None)
        if isinstance(value, str):
            self._settings.set("license_class", value)
        if HAS_HAM_RADIO:
            self._qrp_panel.refresh_license()  # the legal limit here
        self._tx_station.refresh()  # what Send ID Now may do now

    def _save_ham_id_settings(self, *_args) -> None:
        """Persist the Ham ID panel (callsign, CW speed/tone, auto-ID)."""
        if self._restoring or not HAS_HAM_RADIO:
            return
        try:
            self._settings.set(
                "ham_id", json.dumps(self._callsign_panel.get_settings())
            )
        except (TypeError, ValueError) as e:  # pragma: no cover - defensive
            logger.debug(f"Could not save the Ham ID settings: {e}")

    def _restore_ham_id_settings(self) -> None:
        if not HAS_HAM_RADIO:
            return
        raw = self._settings.get_str("ham_id", "")
        if not raw:
            return
        try:
            saved = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring unreadable saved Ham ID settings")
            return
        if isinstance(saved, dict):
            try:
                self._callsign_panel.set_settings(saved)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Could not restore the Ham ID settings: {e}")

    # -- QRP panel (T7) ----------------------------------------------------------

    def _connect_qrp(self) -> None:
        """The QRP panel follows the tuned frequency, the mode and the
        license; its settings and tally are saved (ham/qrp); its limit and
        chain gate transmitting; its exchange takes the S-meter's report."""
        qrp = self._qrp_panel
        qrp.state_changed.connect(self._save_qrp_settings)
        qrp.state_changed.connect(lambda _state: self._tx_station.refresh())
        qrp.exchange_copied.connect(
            lambda text: self._show_status_message(f"Copied: {text}", "success", 3000)
        )
        qrp.set_rst_source(self._smeter_rst)
        qrp.set_station(self._current_frequency(), self._control_panel.get_demod_mode())

    def _smeter_rst(self) -> Optional[str]:
        """The S-meter's RST for the signal received (None without one)."""
        panel = self._signal_meter_panel
        reading = getattr(panel, "_last_reading", None)
        if not getattr(panel, "_showing_data", False) or reading is None:
            return None
        if not getattr(reading, "signal_present", False):
            return None
        try:
            return str(panel.get_meter().get_rst())
        except Exception:  # pragma: no cover - defensive
            return None

    def _save_qrp_settings(self, *_args) -> None:
        if self._restoring or not HAS_HAM_RADIO:
            return
        try:
            self._settings.set("ham/qrp", json.dumps(self._qrp_panel.get_state()))
        except (TypeError, ValueError) as e:  # pragma: no cover - defensive
            logger.debug(f"Could not save the QRP settings: {e}")

    def _restore_qrp_settings(self) -> None:
        if not HAS_HAM_RADIO:
            return
        raw = self._settings.get_str("ham/qrp", "")
        if not raw:
            return
        try:
            saved = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring unreadable saved QRP settings")
            return
        if isinstance(saved, dict):
            self._qrp_panel.set_state(saved)

    def _start_or_stop_audio(self, mode: str) -> bool:
        """Open the speaker for audible modes, close it otherwise.

        Returns False when audio is on but the output can't be opened (no
        output device, or the sink failed); ``_audio_problem`` then says why.
        """
        if not self._audio_enabled or mode not in _AUDIBLE_MODES:
            self._audio.stop()
            return True
        problem = self._audio_output_problem()
        if problem is None and not self._audio.start(int(_AUDIO_RATE)):
            problem = "The audio output could not be opened"
        self._audio_problem = problem
        return problem is None

    def _audio_output_problem(self) -> Optional[str]:
        """Why audio can't play right now, or None if it can."""
        if not self._audio.available:
            return "Audio needs the PyQt6 QtMultimedia module"
        try:
            from PyQt6.QtMultimedia import QMediaDevices

            if QMediaDevices.defaultAudioOutput().isNull():
                return "No audio output device"
        except Exception as e:  # pragma: no cover - backend-dependent
            logger.debug(f"Audio device check failed: {e}")
        return None

    def _listening_mode(self) -> Tuple[str, str]:
        """The current ``(demod mode, FM deviation)``, as bookmarks save it
        (the deviation only in FM)."""
        panel = self._control_panel
        mode = panel.get_demod_mode()
        return mode, panel.get_fm_deviation_text() if mode == "FM" else ""

    def _bandwidth_for_mode(self, mode: str, deviation: str = "") -> Optional[str]:
        """The Bandwidth option to listen to ``mode`` with (None: keep).

        FM: the narrowest of FM's options at least 80% of Carson's
        bandwidth, twice the deviation plus the highest audio frequency (15
        kHz for broadcast, 3 kHz for voice): 200 kHz for broadcast FM, 50
        kHz at 25 kHz deviation, 25 kHz at 12.5 or 5 kHz (2 m, APRS, NOAA)
        and 12.5 kHz at 2.5 kHz. AM: 10 kHz (airband and broadcast AM).
        USB/LSB/CW: None, so the mode keeps its own filter (the width last
        used with it, or its default).
        """
        if mode == "FM":
            hz = _parse_hz(deviation) if deviation else None
            if hz is None:
                hz = float(self._control_panel.get_fm_deviation())
            audio = _MAX_AUDIO_HZ if hz >= _BROADCAST_DEVIATION_HZ else 3e3
            need = 0.8 * 2.0 * (hz + audio)
        elif mode == "AM":
            need = 10e3
        else:
            return None
        return closest_bandwidth(bandwidth_options(mode), need) or None

    @staticmethod
    def _mode_text(mode: str) -> str:
        """Short mode name for messages: ``"None (I/Q)"`` -> ``"I/Q"``."""
        return "I/Q" if mode.startswith("None") else mode

    def _on_bookmark_tune(
        self, freq_hz: float, label: str, mode: str = "", deviation: str = ""
    ) -> None:
        """Tune to a bookmark (or the Decoder panel's suggested channel), in
        its mode when it names one, with a channel width that suits it."""
        if not self.set_frequency(freq_hz):
            return  # refused: the status bar says why
        if not mode:
            self._show_status_message(f"Tuned to {label}", "info", 2500)
            return
        self._set_demod_mode(
            mode, deviation or None, self._bandwidth_for_mode(mode, deviation)
        )
        mode = self._mode_text(mode)
        if label.endswith(")"):  # "ACARS (131.550 MHz)" -> "(131.550 MHz, AM)"
            text = f"Tuned to {label[:-1]}, {mode})"
        else:
            text = f"Tuned to {label} ({mode})"
        self._show_status_message(text, "info", 2500)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _toggle_acquisition(self):
        """Toggle signal acquisition."""
        if self._is_running:
            self._stop_acquisition()
        else:
            self._start_acquisition()

    def _prompt_no_device(self) -> str:
        """Ask how to proceed when Start is pressed with no device.

        Returns ``"connect"``, ``"demo"`` or ``"cancel"``.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("No Device Connected")
        box.setText("No SDR device is connected.")
        box.setInformativeText(
            "Connect opens an RTL-SDR or HackRF One. Demo Mode explores the "
            "app with simulated signals, no hardware needed."
        )
        connect_btn = box.addButton("&Connect...", QMessageBox.ButtonRole.AcceptRole)
        set_role(connect_btn, "primary")  # the recommended path, as elsewhere
        demo_btn = box.addButton("&Demo Mode", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(connect_btn)
        box.setEscapeButton(cancel_btn)
        self._exec_dialog(box)
        clicked = box.clickedButton()
        if clicked is connect_btn:
            return "connect"
        if clicked is demo_btn:
            return "demo"
        return "cancel"

    def _start_acquisition(self):
        """Start signal acquisition."""
        if self._is_running:
            return
        if self._tx_busy():
            self._show_status_message(
                "Can't receive while transmitting (the HackRF is half duplex): "
                "receiving resumes when the transmission ends.",
                "warning",
            )
            return
        if not self._device:
            choice = self._prompt_no_device()
            if choice == "demo":
                self._start_demo_mode()
                return
            # The dialog starts receiving on the device it opens.
            if choice == "connect":
                self._show_device_dialog(start_after=True)
            return

        try:
            started = self._device.start_rx()
        except Exception as e:
            logger.error(f"Could not start receiving: {e}")
            self._show_status_error(f"Could not start receiving: {e}", log=False)
            return
        if started is False:
            self._show_status_error("The device did not start streaming")
            return

        self._is_running = True
        self._rx_history.clear()
        self._pacer.restart()
        self._refresh_state_ui()
        logger.info("Acquisition started")

    def _stop_acquisition(self):
        """Stop signal acquisition."""
        self._is_running = False
        self._reset_demodulator()  # the next samples don't follow on
        self._rx_history.clear()

        # (A transmission in progress keeps the Ham ID panel's TX state.)
        if HAS_HAM_RADIO and not self._tx_busy():
            self._callsign_panel.set_transmitting(False)

        if self._device:
            try:
                self._device.stop_rx()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"stop_rx failed: {e}")

        self._refresh_state_ui()
        logger.info("Acquisition stopped")

    def _on_callsign_id_requested(self):
        """Ham ID > Send ID Now (or Radio > Send Station ID): the station ID
        on the tuned frequency, through the transmit engine (checked, asked
        the first time, sent without blocking the window)."""
        self._tx_station.send_station_id()

    def _tx_busy(self) -> bool:
        """True while a transmission is on the air (or being set up)."""
        station = getattr(self, "_tx_station", None)
        return bool(station is not None and station.busy)

    def _show_tx_state(self, on_air: bool) -> None:
        """The status bar's state badge says ON AIR while transmitting (the
        state comes back with the next _refresh_state_ui)."""
        if on_air:
            self._state_badge.setText("ON AIR")
            set_tone(self._state_badge, "danger")
            self._state_badge.setToolTip(
                "Transmitting; receiving resumes when the transmission ends"
            )

    def _transmit_job(self, job: "TxJob", confirm: bool = True) -> bool:
        """Transmit ``job`` (see :class:`~.transmitter.TxJob`) on the HackRF:
        checked (license, mode, bandwidth, power), asked the first time,
        with the Ham ID panel's automatic IDs, without blocking the window.
        False (the status bar says why) when it isn't sent. Transmit Text
        and any other transmission come through here."""
        return self._tx_station.transmit(job, confirm=confirm)

    def _transmit_audio(
        self,
        iq_samples: np.ndarray,
        description: str = "audio",
        sample_rate: float = 2e6,
        mode: str = "FM",
        bandwidth_hz: float = 12.5e3,
    ) -> bool:
        """Transmit ready-made I/Q samples on the tuned frequency (kept for
        callers of the old blocking API; the samples are taken to be
        ``mode`` at ``sample_rate``, ``bandwidth_hz`` wide). Prefer
        :meth:`_transmit_job`."""
        job = TxJob(
            ArraySource(iq_samples),
            sample_rate,
            self._current_frequency(),
            mode,
            bandwidth_hz,
            description,
            identify=False,
        )
        return self._transmit_job(job)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _toggle_recording(self, checked: bool):
        """Toggle recording from the toolbar button / menu action."""
        if checked:
            if not self._start_recording():
                self._sync_record_ui(False)  # cancelled: stay stopped
        else:
            self._stop_recording()
        # Keep the Record buttons in sync (without re-emitting).
        self._sync_record_buttons()

    def _start_recording(self) -> bool:
        """Start (or arm) a recording. False if the user kept an unsaved one
        or the disk is nearly full.

        The buffer is replaced when the first samples arrive, so arming and
        stopping without capturing anything keeps the previous recording.
        """
        if self._recording:
            return True
        free = self._free_recording_space()
        if free is not None and free < _MIN_FREE_BYTES:
            self._show_status_error(
                f"Can't record: only {free / 1024**2:.0f} MB of disk space is "
                "left. Free up some space, then record again.",
                10000,
            )
            return False
        if not self._confirm_discard_recording("starting a new recording"):
            return False
        self._recording = True
        self._recording_paused = False
        self._capture_pending = True
        self._next_space_check = 0.0  # check the free space right away
        self._recording_bytes = 0
        self._rec_accum = 0.0
        self._rec_since = None
        self._update_rec_clock()
        self._sync_record_ui(True)
        if not self._is_running:
            self._show_status_message(
                "Recording armed: samples are captured once receiving starts "
                "(Space)",
                "warning",
                6000,
            )
        else:
            # Replaces any "Use File > Save Recording" message about the
            # previous recording.
            self._show_status_message(
                "Recording I/Q samples. Press Record again to stop.", "info", 3000
            )
        logger.info("Recording started")
        return True

    def _stop_recording(self):
        """Stop recording."""
        was_recording = self._recording
        captured = was_recording and not self._capture_pending
        self._recording = False
        self._recording_paused = False
        self._capture_pending = False
        self._update_rec_clock()
        self._sync_record_ui(False)
        try:
            self._samples_buffer.finish()
        except OSError as e:  # pragma: no cover - disk trouble
            logger.warning(f"Could not write the end of the recording: {e}")
        count = self._buffer_sample_count()
        if captured and count and self._control_panel.is_auto_save():
            if getattr(self, "_auto_save_later", False):
                # Stopped from inside a display frame (the disk filled up):
                # saved right after it, not in a nested event loop here.
                store = self._samples_buffer
                QTimer.singleShot(
                    0, lambda: self._auto_save_take(announce=False, store=store)
                )
            else:
                self._auto_save_take()
        elif captured and count:
            self._show_status_message(
                f"Recorded {count:,} samples. Use File > Save Recording (Ctrl+S) "
                "to write them to disk.",
                "success",
                8000,
            )
        elif was_recording:
            self._show_status_message(
                "Recording stopped. Nothing was captured because the receiver "
                "was not running.",
                "warning",
                6000,
            )
        self._update_take_panel()
        logger.info("Recording stopped")

    def _buffer_sample_count(self) -> int:
        return self._samples_buffer.sample_count

    def _replace_buffer(self, store: "_SampleStore") -> None:
        """Make ``store`` the recording buffer, deleting the old one's
        temporary file."""
        old, self._samples_buffer = self._samples_buffer, store
        if old is not store:
            old.discard()

    def _capture_folder(self) -> str:
        """The folder recordings are captured into: the one chosen in Tools
        > Preferences, else the user cache folder."""
        if self._recording_folder is None:
            self._recording_folder = (
                preferred_capture_folder(self._settings) or _recording_dir()
            )
        return self._recording_folder

    def _show_preferences(self) -> None:
        """Tools > Preferences (Ctrl+,): audio output, default folders, the
        recording temp folder and the transmit confirmation."""
        rate = self._device_sample_rate() if self._device else _DEFAULT_SAMPLE_RATE
        dialog = PreferencesDialog(
            self._settings, self, capture_default=_recording_dir(), sample_rate=rate
        )
        if not self._exec_dialog(dialog):
            return
        before = self._audio.output_id
        prefs = dialog.apply()
        self._audio.set_output_device(prefs.audio_output, prefs.audio_output_name)
        self._control_panel.set_recordings_folder(prefs.recordings_folder)
        # The next recording starts in the temp folder chosen (one running
        # goes on where it is).
        self._recording_folder = None
        # Audio on the new output (or off, saying why, if it can't open).
        self._refresh_audio_output(announce=True)
        if prefs.audio_output != before:
            self._show_status_message(
                f"Audio plays through {self._audio.output_name()}.", None, 5000
            )
        else:
            self._show_status_message("Preferences saved.", None, 3000)

    def _confirm_discard_recording(self, doing: str) -> bool:
        """Before ``doing`` (e.g. "closing") drops the unsaved recording,
        offer to save it. True to go ahead: saved, discarded or nothing to
        lose; False to cancel.

        A recording still capturing is included: Save and Don't Save stop
        it first, Cancel leaves it running.
        """
        if not (self._buffer_unsaved and self._samples_buffer):
            return True
        live = self._recording and not self._capture_pending
        if live and self._control_panel.is_auto_save():
            # Stopping saves it (Save automatically): nothing to ask.
            self._toggle_recording(False)
            if not self._buffer_unsaved:
                return True
            live = False
        count = self._buffer_sample_count()
        freq, _rate = self._buffer_meta
        # (A non-breaking space keeps "100.000 MHz" on one line.)
        where = (
            f" at {format_frequency(freq)}".replace(" MHz", "\u00a0MHz") if freq else ""
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved Recording")
        box.setText(f"Save the recording before {doing}?")
        if live:
            state = "paused" if self._recording_paused else "in progress"
            box.setInformativeText(
                f"A recording is {state}: {count:,} I/Q samples "
                f"({count * 8 / 1e6:.1f} MB) so far{where}. Save and Don't Save "
                "stop it; Cancel keeps it going."
            )
        else:
            box.setInformativeText(
                f"{count:,} I/Q samples ({count * 8 / 1e6:.1f} MB) "
                f"recorded{where} haven't been saved. If you don't save them, "
                "they are lost."
            )
        save_btn = box.addButton("&Save...", QMessageBox.ButtonRole.AcceptRole)
        set_role(save_btn, "primary")
        discard_btn = box.addButton(
            "&Don't Save", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(save_btn)
        box.setEscapeButton(cancel_btn)
        self._exec_dialog(box)
        clicked = box.clickedButton()
        if clicked not in (save_btn, discard_btn):
            return False  # Cancel: a running recording keeps going
        if self._recording:
            self._toggle_recording(False)
        if clicked is save_btn:
            return self._save_recording()
        self._discard_recording()
        return True

    def _discard_recording(self) -> None:
        """Empty the recording buffer (deleting its temporary file)."""
        self._replace_buffer(_SampleStore())
        self._buffer_meta = (None, None)
        self._buffer_unsaved = False
        self._buffer_source = ""
        self._clear_status_message()  # e.g. "Use File > Save Recording"
        self._update_take_panel()

    def _recording_capturing(self) -> bool:
        """True while samples are actually being added to the recording."""
        return (
            self._recording
            and not self._recording_paused
            and self._is_running
            and self._device is not None
        )

    def _recording_armed(self) -> bool:
        """True while a recording waits for the receiver to run (it captures
        nothing until then). A pause is shown as such, not as armed."""
        return self._recording and not (self._is_running and self._device is not None)

    def _sync_recording_armed(self) -> None:
        """Show on the Recording panel whether the recording is armed (or
        waiting for the receiver after capturing some) or capturing, as the
        status bar's REC (or AUDIO) badge does."""
        panel = self._control_panel
        if self._record_kind() == "audio":
            take = self._audio_take
            waiting = take is not None and not (
                self._is_running and self._device is not None
            )
            panel.set_recording_armed(waiting, waiting=bool(take and take.samples))
            return
        armed = self._recording_armed()
        panel.set_recording_armed(armed, waiting=armed and not self._capture_pending)

    def _update_rec_clock(self) -> None:
        """Bank the running stretch of the recording clock and restart it if
        capturing continues. Call whenever recording, pause or receiver state
        changes, so the clock only counts time that samples were captured."""
        now = time.monotonic()
        if self._rec_since is not None:
            self._rec_accum += max(0.0, now - self._rec_since)
            self._rec_since = None
        if self._recording_capturing():
            self._rec_since = now

    def _recording_elapsed(self) -> float:
        """Seconds spent capturing in the current recording."""
        elapsed = self._rec_accum
        if self._rec_since is not None:
            elapsed += max(0.0, time.monotonic() - self._rec_since)
        return elapsed

    def _recording_elapsed_text(self) -> str:
        elapsed = int(self._recording_elapsed())
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_recording_status(self) -> None:
        """Refresh the REC badge, size/free-space label and panel timer."""
        elapsed = self._recording_elapsed_text()
        badge = self._recording_label
        if self._recording_paused:
            badge.setText(f"PAUSED {elapsed}")
            set_tone(badge, "warning")
            badge.setToolTip("Recording paused. Resume it in the Recording panel.")
        elif not self._recording_capturing() and not self._capture_pending:
            # Stopped mid-recording: what was captured is kept.
            badge.setText(f"REC WAITING {elapsed}")
            set_tone(badge, "warning")
            badge.setToolTip(
                "The receiver is stopped: the recording resumes when it starts "
                "again (Space)"
            )
        elif not self._recording_capturing():
            badge.setText("REC ARMED")
            set_tone(badge, "warning")
            badge.setToolTip(
                "Recording is armed: samples are captured once receiving "
                "starts (Space)"
            )
        else:
            badge.setText(f"REC {elapsed}")
            set_tone(badge, "danger")
            badge.setToolTip("Recording raw I/Q samples")
        self._sync_recording_armed()
        if self._record_kind() != "audio":  # (else the group times the audio)
            self._control_panel.update_record_time(int(self._recording_elapsed()))

        mb = self._recording_bytes / (1024 * 1024)
        # Recorded size + free space on the drive the recording goes to
        free = self._free_recording_space()
        if free is None:
            info = f"{mb:.1f} MB"
        else:
            info = f"{mb:.1f} MB · {free / 1024**3:.1f} GB free"
        self._recording_info_label.setText(info)

    def _free_recording_space(self) -> Optional[int]:
        """Free bytes on the drive recordings are written to: the running
        take's own (a temp folder chosen in Tools > Preferences meanwhile is
        for the next one)."""
        folder = None
        if self._recording and not self._capture_pending:
            folder = getattr(self._samples_buffer, "folder", None)
        try:
            return int(shutil.disk_usage(folder or self._capture_folder()).free)
        except OSError:
            return None

    def _check_recording_space(self) -> bool:
        """Stop the recording, with a warning, when its drive is nearly
        full (checked twice a second). False when it was stopped."""
        now = time.monotonic()
        if now < self._next_space_check:
            return True
        self._next_space_check = now + _FREE_SPACE_CHECK_S
        free = self._free_recording_space()
        if free is None or free >= _MIN_FREE_BYTES:
            return True
        self._stop_recording_with_warning(
            f"only {free / 1024**2:.0f} MB of disk space is left"
        )
        return False

    def _stop_recording_with_warning(self, reason: str) -> None:
        """Stop recording because it can't go on (``reason``), keeping what
        was captured."""
        captured = self._recording and not self._capture_pending
        self._auto_save_later = True  # (inside a display frame)
        try:
            self._toggle_recording(False)
        finally:
            self._auto_save_later = False
        size = self._samples_buffer.nbytes / 1024**2
        if captured and self._samples_buffer and self._control_panel.is_auto_save():
            kept = f" The {size:,.0f} MB recorded so far are kept."
        elif captured and self._samples_buffer:
            kept = (
                f" The {size:,.0f} MB recorded so far are kept: save them with "
                "File > Save Recording (Ctrl+S)."
            )
        else:
            kept = ""
        message = f"Recording stopped: {reason}.{kept}"
        logger.warning(message)
        self._show_status_error(message, 20000, log=False)  # (logged once)

    # -- the take, the recordings folder and auto-save ---------------------

    def _auto_save_take(
        self, announce: bool = True, store: Optional["_SampleStore"] = None
    ) -> bool:
        """Save automatically is on: the take just stopped goes to the
        recordings folder with an automatic name, in the panel's format.
        With ``store``, only if that is still the recording buffer."""
        if store is not None and store is not self._samples_buffer:
            return False
        if not (self._samples_buffer and self._buffer_unsaved):
            return False
        folder = file_dialogs.recordings_folder(self._settings)
        if not file_dialogs.ensure_folder(folder):
            self._show_status_error(
                f"Couldn't save the recording automatically: the folder "
                f"{file_dialogs.display_path(folder)} can't be created. Use "
                "Save... in the Recording group."
            )
            return False
        ext = self._panel_recording_extension()
        path = file_dialogs.unique_path(folder, self._suggested_iq_name(ext))
        message = self._message_label.text()
        if not self._write_buffer(path, quiet=True):
            return False
        if announce:
            self._show_status_message(
                f"Saved the recording to {file_dialogs.display_path(path)}",
                "success",
                8000,
            )
        else:  # (a warning is on show: keep it)
            self._show_status_message(
                f"{message} It was saved to {file_dialogs.display_path(path)}.",
                "danger",
                20000,
            )
        return True

    def _take_summary(self) -> Tuple[str, Optional[str], str]:
        """(text, state, tooltip) of the Recording group's take line."""
        store = self._samples_buffer
        count = store.sample_count if store else 0
        if self._recording and not self._capture_pending:
            return "", None, ""  # the status line shows it
        if count and (self._buffer_unsaved or self._buffer_source):
            center, rate = self._buffer_meta
            parts = []
            if rate:
                parts.append(_clock_text(count / rate))
            parts.append(f"{count * 8 / 1e6:,.0f} MB")
            where = f" @ {format_frequency(center)}" if center else ""
            if self._buffer_source:
                text = f"Loaded {self._buffer_source} · " + " · ".join(parts) + where
                tip = (
                    f"{self._buffer_source} is loaded to be saved in another "
                    "format (File > Convert Recording)."
                )
                return text, "loaded", tip
            text = "Unsaved · " + " · ".join(parts) + where
            tip = "This recording is kept in a temporary file until you save it."
            return text, "unsaved", tip
        if self._last_saved and os.path.exists(self._last_saved):
            name = os.path.basename(self._last_saved)
            return f"Saved {name}", "saved", self._last_saved
        return "", None, ""

    def _update_take_panel(self) -> None:
        """Show the take (unsaved, loaded or saved) in the Recording group."""
        text, state, tip = self._take_summary()
        self._control_panel.show_take(text, state, tip)

    def _on_take_save(self) -> None:
        """The Recording group's Save... (Save As... for a loaded file)."""
        self._save_recording()

    def _on_take_discard(self) -> None:
        """The Recording group's Discard (Unload for a loaded file)."""
        if not self._samples_buffer:
            self._update_take_panel()
            return
        if self._buffer_unsaved and not self._buffer_source:
            count = self._buffer_sample_count()
            answer = QMessageBox.question(
                self,
                "Discard Recording",
                f"Delete the {count * 8 / 1e6:,.0f} MB recording without saving it?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        name = self._buffer_source
        self._discard_recording()
        self._show_status_message(
            f"Unloaded {name}." if name else "Recording discarded.", None, 3000
        )

    def _on_open_folder(self) -> None:
        """Show the last saved recording in the file manager."""
        target = self._last_saved or file_dialogs.recordings_folder(self._settings)
        if not file_dialogs.show_in_folder(target):
            self._show_status_message(
                f"Couldn't open the folder: {file_dialogs.display_path(target)}",
                "warning",
                6000,
            )

    def _on_change_recordings_folder(self) -> None:
        """Choose where audio recordings and auto-saved takes go."""
        current = file_dialogs.recordings_folder(self._settings)
        folder = file_dialogs.choose_folder(
            self,
            self._settings,
            "iq",
            "Recordings Folder",
            current if os.path.isdir(current) else "",
        )
        if not folder:
            return
        file_dialogs.set_recordings_folder(self._settings, folder)
        self._control_panel.set_recordings_folder(folder)
        self._show_status_message(
            f"Recordings go to {file_dialogs.display_path(folder)}", None, 4000
        )

    def _on_auto_save_changed(self, on: bool) -> None:
        self._save_setting("recording/auto_save", bool(on))
        if on and not self._restoring:
            folder = file_dialogs.display_path(
                file_dialogs.recordings_folder(self._settings)
            )
            self._show_status_message(
                f"I/Q recordings are saved to {folder} when they stop.", None, 5000
            )

    # -- Record Audio --------------------------------------------------------

    def _toggle_audio_recording(self, checked: bool) -> None:
        """Radio > Record Audio (Ctrl+Shift+A)."""
        if checked:
            self._start_audio_recording()
        else:
            self._stop_audio_recording()

    def _start_audio_recording(self) -> bool:
        """Record what you hear: the demodulated audio, as 48 kHz 16-bit
        WAV in the recordings folder, named after the frequency, mode and
        time. It records whether or not the speaker is on (squelched
        stretches as silence), also while a recording plays back."""
        if self._audio_take is not None:
            return True
        mode = self._control_panel.get_demod_mode()
        if mode not in _AUDIBLE_MODES:
            self._show_status_message(
                "None (I/Q) has no audio to record: choose AM, FM, USB, LSB or "
                "CW in Demodulation first.",
                "warning",
                8000,
            )
            self._sync_audio_record_ui()
            return False
        folder = file_dialogs.recordings_folder(self._settings)
        if not file_dialogs.ensure_folder(folder):
            self._show_status_error(
                f"Can't record audio: the folder {file_dialogs.display_path(folder)} "
                "can't be created. Choose another in the Recording group."
            )
            self._sync_audio_record_ui()
            return False
        freq = self._current_frequency()
        path = file_dialogs.unique_path(
            folder, audio_file_name(freq, self._mode_text(mode))
        )
        try:
            take = AudioTake(path, freq, mode, int(_AUDIO_RATE))
        except (OSError, RuntimeError) as e:
            logger.error(f"Could not start an audio recording in {path}: {e}")
            self._show_status_error(
                f"Can't record audio to {file_dialogs.display_path(path)}: {e}",
                log=False,
            )
            self._sync_audio_record_ui()
            return False
        self._audio_take = take
        self._sync_audio_record_ui()
        note = ""
        if not (self._is_running and self._device is not None):
            note = " It starts with the receiver (Space)."
        elif not self._audio_enabled:
            note = " (The speaker is off; the recording still gets the audio.)"
        self._show_status_message(
            f"Recording audio to {os.path.basename(path)}.{note}", "info", 5000
        )
        logger.info(f"Audio recording started: {path}")
        return True

    def _stop_audio_recording(self, quiet: bool = False) -> Optional[str]:
        """Finish the audio recording (it is saved as it records); returns
        its path, None if none was running."""
        take = self._audio_take
        if take is None:
            return None
        self._audio_take = None
        seconds = take.stop()
        if take.samples:
            self._last_saved = take.path
        else:
            # Nothing was received: no empty file left behind.
            try:
                os.remove(take.path)
            except OSError:
                pass
        self._sync_audio_record_ui()
        self._update_take_panel()
        if not quiet:
            if take.samples:
                self._show_status_message(
                    f"Saved {_clock_text(seconds)} of audio to "
                    f"{file_dialogs.display_path(take.path)}",
                    "success",
                    8000,
                )
            else:
                self._show_status_message(
                    "Audio recording stopped: nothing was received, so nothing "
                    "was saved.",
                    "warning",
                    6000,
                )
        logger.info(f"Audio recording stopped: {take.path} ({seconds:.1f} s)")
        return take.path

    def _sync_audio_record_ui(self) -> None:
        on = self._audio_take is not None
        action = self._record_audio_action
        if action.isChecked() != on:
            action.blockSignals(True)
            action.setChecked(on)
            action.blockSignals(False)
        self._update_audio_recording_status()
        self._sync_record_buttons()

    def _update_audio_recording_status(self) -> None:
        """The AUDIO badge ("AUDIO 00:12") and, when the Record buttons
        record audio, the Recording group's timer."""
        take = getattr(self, "_audio_take", None)
        label = getattr(self, "_audio_rec_label", None)
        if label is None:
            return
        label.setVisible(take is not None)
        if take is None:
            return
        clock = _clock_text(take.seconds)
        receiving = self._is_running and self._device is not None
        where = file_dialogs.display_path(take.path)
        if take.paused:
            label.setText(f"AUDIO PAUSED {clock}")
            set_tone(label, "warning")
            label.setToolTip(f"Audio recording paused ({where})")
        elif not receiving:
            label.setText(f"AUDIO {clock}")
            set_tone(label, "warning")
            label.setToolTip(
                f"Recording audio to {where}: waiting for the receiver to "
                "start (Space)"
            )
        else:
            label.setText(f"AUDIO {clock}")
            set_tone(label, "danger")
            label.setToolTip(
                f"Recording what you hear to {where}. Radio > Record Audio "
                "(Ctrl+Shift+A) stops it."
            )
        if self._record_kind() == "audio":
            self._control_panel.update_record_time(int(take.seconds))
            self._sync_recording_armed()

    def _record_audio_block(
        self, audio: Optional[np.ndarray], open_: bool, seconds: float, joined: bool
    ) -> None:
        """Give the audio recording this block: ``audio`` as heard while the
        squelch is ``open_`` (silence otherwise), or ``seconds`` of silence
        when nothing was demodulated."""
        take = self._audio_take
        if take is None:
            return
        try:
            if audio is None:
                take.silence(seconds)
            else:
                take.write(audio, open_, continuous=not joined)
        except Exception as e:
            self._audio_take = None
            try:
                take.stop()
            except Exception as stop_error:  # pragma: no cover - disk trouble
                logger.debug(f"Could not finish {take.path}: {stop_error}")
            message = (
                f"Audio recording stopped: {file_dialogs.display_path(take.path)} "
                f"can't be written ({e}). What was recorded is kept."
            )
            logger.warning(message)
            self._show_status_error(message, 20000, log=False)
            self._sync_audio_record_ui()

    # ------------------------------------------------------------------
    # Periodic updates
    # ------------------------------------------------------------------

    def _read_block(self) -> Optional[np.ndarray]:
        """The samples received since the last display frame, without
        stalling the window (None when nothing new has arrived).

        Hardware drivers queue whole USB transfers from a background thread,
        and ``read_samples`` waits up to a second for one by default, which
        froze the GUI: poll without waiting. Take every queued transfer, not
        just one: at 8 MS/s and more a HackRF delivers 60 to 150 transfers a
        second, more than the display frames, and reading one per frame fell
        behind (and muted the audio as "not real time").

        The demo device makes samples on request: it is asked for as many as
        the time since the last frame covers, so it runs in real time too,
        however it was opened (Use Demo Device, or its row in Device >
        Connect...). Other sources give one display block per frame.
        :class:`_Pacer` limits how much one frame takes.
        """
        dev = self._device
        if isinstance(dev, SDRDevice):
            return self._drain_device(dev)
        if getattr(dev, "paced", False):
            # A recording playing back: in real time times its speed, never
            # ahead of it (a slow one skips frames; an empty read is fine).
            rate = self._pacing_rate()
            return dev.read_samples(self._pacer.paced_request(rate, time.monotonic()))
        if self._demo_mode or _is_demo_device(dev):
            rate = self._pacing_rate()
            return dev.read_samples(self._pacer.demo_request(rate, time.monotonic()))
        return dev.read_samples(DISPLAY_BLOCK)

    def _pacing_rate(self) -> float:
        """Samples per second the display loop takes: the sample rate, times
        the speed of a recording playing back."""
        rate = self._device_sample_rate()
        speed = getattr(self._device, "speed", 1.0) if self._playback.active else 1.0
        try:
            return rate * max(0.01, float(speed))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return rate

    def _drain_device(self, dev: SDRDevice) -> Optional[np.ndarray]:
        """The transfers a hardware driver has queued, joined, up to the
        pacer's limit (the rest stays queued for the next frame)."""
        limit = self._pacer.read_limit(self._device_sample_rate(), time.monotonic())
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

    def _stream_died(self) -> bool:
        """Stop and tell the user if a hardware stream ended on its own
        (USB unplugged, read error); otherwise the window would sit on
        "RUNNING" with frozen plots."""
        dev = self._device
        if not isinstance(dev, SDRDevice):
            return False
        try:
            if dev.state.is_streaming:
                return False
        except Exception:  # pragma: no cover - defensive
            return False
        error = getattr(dev, "rx_error", None)
        self._stop_acquisition()
        detail = f": {error}" if error else ""
        self._show_status_error(
            f"The device stopped streaming{detail}. Check the USB connection, "
            "then press Start.",
            10000,
        )
        return True

    def _update_display(self):
        """Update spectrum and waterfall displays."""
        if not self._is_running or not self._device:
            return

        started = time.perf_counter()
        samples = self._read_block()
        if samples is None or len(samples) == 0:
            self._stream_died()
            return
        self._note_block(len(samples))
        self._process_block(samples)
        # How long the frame took against the time its samples cover.
        self._pacer.measured(
            len(samples), self._pacing_rate(), time.perf_counter() - started
        )

    def _process_block(self, samples: np.ndarray) -> None:
        """Plots, level, audio, recording, S-meter and decoder for one frame."""
        # Remove DC spike / Balance I/Q (Receiver group), when switched on;
        # a recording keeps the raw samples.
        raw = samples
        samples = self._iq_conditioner.process(samples, self._device_sample_rate())
        # The measurement line: a 2048-point Hann FFT of the newest samples
        # (dBFS), the same whatever the plots show. The noise floor (Auto
        # squelch), the S-meter's noise and the radio tuner use it.
        display = samples[-DISPLAY_BLOCK:] if len(samples) > DISPLAY_BLOCK else samples
        power = self._power_spectrum_dbfs(display)

        # The plots: every sample of the frame, at the chosen FFT size and
        # window (the mean power per bin for the spectrum, the strongest
        # for the waterfall row, so a short burst anywhere shows).
        self._plot_frame(samples)

        # Level of the tuned channel: a strong station elsewhere in the span
        # no longer holds the squelch open. (The meter and the squelch use
        # the power of the whole block's channel, see _route_audio; the
        # spectrum gives the noise floor for Auto squelch and the SNR.)
        chain = self._receiver_chain()
        level = self._channel_level(power, chain)
        self._last_peak_db = level
        self._track_noise_floor(power, len(samples))

        # The tuned channel, filtered out of the block: the S-meter measures
        # it and the demodulator listens to it.
        channel = chain.channelize(samples)

        # Audio: the speaker (while the signal is above squelch) and the SSTV
        # decoder (whenever it listens, speaker on or off).
        self._route_audio(channel, level)
        # Broadcast FM: the station's name from its RDS, in the background.
        self._listen_rds(channel, chain)
        # The AM/FM radio tuner's seek and STEREO lamp (only while it is open).
        self._tuner_block(channel, chain, power)
        # The Signal tab and Track Signal (T10): only what is on screen or on.
        self._signal_tools.frame(samples, channel, power, chain)

        # Record if active (and not paused). The REC badge and the Recording
        # panel's timer are refreshed by the status timer.
        if self._recording and not self._recording_paused:
            self._capture(raw)

        if HAS_HAM_RADIO and len(channel):
            try:
                self._signal_meter_panel.update_samples(
                    channel, self._channel_noise_dbfs(power, chain)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("S-meter update failed: %s", exc)

        # Feed the live protocol decoder (if the Decoder panel selected one).
        self._run_decoder(samples)

        # The Scanner tab's scan runs on the same blocks.
        self._scan_frame(samples)

    def _capture(self, samples: np.ndarray) -> None:
        """Add a block to the recording (the first one replaces the buffer)."""
        if not self._check_recording_space():
            return
        # The samples are centered on the LO (not the tuned frequency).
        tuned = (self._lo_frequency(), self._device_sample_rate())
        if self._capture_pending:
            self._capture_pending = False
            self._replace_buffer(_SampleStore(self._capture_folder()))
            self._buffer_meta = tuned
            self._buffer_source = ""
            self._take_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._take_stamp = time.strftime("%Y%m%d_%H%M%S")
            self._retune_warned = False
            self._sync_recording_armed()
            self._update_take_panel()  # (the status line shows the new take)
        elif not self._retune_warned and any(
            known is not None and abs(known - now) > 0.5
            for known, now in zip(self._buffer_meta, tuned, strict=True)
        ):
            # One file has one center frequency and rate: say which it keeps.
            self._retune_warned = True
            freq, rate = self._buffer_meta
            self._show_status_message(
                "Retuned while recording: the saved file will be labelled "
                f"{format_frequency(freq or 0)} at {format_rate(rate or 0)}, "
                "where the recording started. Stop and record again for a "
                "clean capture.",
                "warning",
                10000,
            )
        try:
            self._samples_buffer.append(samples)
        except (OSError, ValueError) as e:
            self._stop_recording_with_warning(
                f"the recording file can't be written ({e})"
            )
            return
        self._buffer_unsaved = True
        # complex64 = 8 bytes/sample
        self._recording_bytes += len(samples) * 8

    def _receiver_chain(self) -> "_ReceiverChain":
        """The channel filter and demodulator for the current settings."""
        rate = float(self._device_sample_rate())
        mode = self._control_panel._demod_combo.currentText()
        bandwidth = self._channel_bandwidth()
        chain = self._rx_chain
        if (
            chain is None
            or (chain.rate, chain.mode) != (rate, mode)
            or chain.bandwidth != _ReceiverChain.clamp_bandwidth(mode, bandwidth, rate)
        ):
            chain = self._rx_chain = _ReceiverChain(rate, mode, bandwidth)
        # The channel is where the VFO is (moving it keeps the filters).
        chain.set_offset(self._vfo_offset_hz())
        return chain

    def _channel_bandwidth(self) -> float:
        """The Bandwidth control, in Hz (the channel that is demodulated)."""
        hz = _parse_hz(self._control_panel._bw_combo.currentText())
        return hz if hz else 200e3

    def _channel_level(self, power_db: np.ndarray, chain: "_ReceiverChain") -> float:
        """Strongest spectrum bin inside the tuned channel (at the VFO's
        offset from the LO), in dBFS."""
        n = len(power_db)
        if n == 0:
            return -120.0
        bin_hz = chain.rate / n
        low, high = chain.band_edges()
        offset = float(getattr(chain, "offset_hz", 0.0))
        center = n // 2  # fftshift puts the LO (0 Hz) here
        first = max(0, center + int(np.floor((offset + low) / bin_hz + 0.5)))
        last = min(n - 1, center + int(np.ceil((offset + high) / bin_hz - 0.5)))
        if last < first:
            first = last = min(max(center + int(round(offset / bin_hz)), 0), n - 1)
        return float(np.max(power_db[first : last + 1]))

    def _channel_noise_dbfs(
        self, power_db: np.ndarray, chain: "_ReceiverChain"
    ) -> Optional[float]:
        """Noise power in the tuned channel (dBFS), for the S-meter's SNR.

        Measured from the spectrum bins outside the channel (and away from
        the span's edges, where the anti-alias filter rolls off) by
        :func:`_noise_per_bin`, robust to the stations among them. A
        Hann-windowed bin of white noise holds ``ENBW / N`` of its power, so
        the noise in a channel ``width`` wide is ``bin * N / ENBW * width /
        rate``.
        """
        n = len(power_db)
        if n < 64:
            return None
        rate = chain.rate
        low, high = chain.band_edges()
        offset = float(getattr(self, "_vfo_offset_hz", lambda: 0.0)())
        key = (n, rate, low, high, offset)
        cached = getattr(self, "_noise_mask", None)
        if cached is None or cached[0] != key:
            bin_hz = rate / n
            freqs = (np.arange(n) - n // 2) * bin_hz  # fftshift order
            guard = max(0.25 * (high - low), 2.0 * bin_hz)
            outside = (freqs < offset + low - guard) | (freqs > offset + high + guard)
            mask = np.nonzero(outside & (np.abs(freqs) <= 0.4 * rate))[0]
            width = max(high - low, bin_hz)
            scale = 10.0 * float(
                np.log10(n / _HANN_ENBW_BINS * _CHANNEL_NOISE_WIDTH * width / rate)
            )
            cached = self._noise_mask = (key, mask, scale)
        _key, mask, scale = cached
        if len(mask) < 32:
            return None
        return _noise_per_bin(power_db[mask]) + scale

    def _update_passband(self) -> None:
        """Mark the demodulated channel on the spectrum and the waterfall:
        the passband at the VFO's offset from the LO (the plots' center) and
        the tuned-frequency marker."""
        low, high = _channel_edges(
            self._control_panel._demod_combo.currentText(),
            self._channel_bandwidth(),
            self._device_sample_rate(),
        )
        offset = float(self._vfo_offset_hz())
        edges = (offset + low, offset + high)
        tuned = float(self._lo_frequency()) + offset
        for plot in (self._spectrum, self._waterfall):
            show = getattr(plot, "set_passband", None)
            mark = getattr(plot, "set_tuned_frequency", None)
            try:
                if callable(show):
                    show(*edges)
                if callable(mark):
                    mark(tuned)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Could not show the passband: %s", exc)

    def _note_block(self, count: int) -> None:
        """Remember when samples arrived (for ``_realtime``)."""
        now = time.monotonic()
        history = self._rx_history
        history.append((now, int(count)))
        while history and now - history[0][0] > _REALTIME_WINDOW_S:
            history.popleft()

    def _realtime(self) -> bool:
        """Whether the device delivers (close to) its full sample rate.

        True until there is a second of history to judge by. A recording
        playing faster or slower than real time isn't (its audio would
        stutter): audio plays at 1x only.
        """
        if self._playback.active and self._playback.speed != 1.0:
            return False
        history = self._rx_history
        if len(history) < 2:
            return True
        span = history[-1][0] - history[0][0]
        if span < 1.0:
            return True
        received = sum(count for _t, count in list(history)[1:])
        return received >= _REALTIME_FRACTION * self._device_sample_rate() * span

    def _power_spectrum_dbfs(self, samples: np.ndarray) -> np.ndarray:
        """Windowed power spectrum in dBFS (full-scale sinusoid -> 0 dB).

        A raw ``20*log10(|FFT|)`` of an N-point block scales with N (an N=2048
        FFT of a full-scale tone peaks near +66 dB), so every bin saturated the
        top of the (-120, 0) dB display and pinned the -80 dB squelch open.

        A Hann window suppresses spectral leakage, and dividing the magnitude by
        the window's coherent gain (the sum of its samples) references the
        result to dBFS: a full-scale complex sinusoid peaks at 0 dB and real
        captures land in the display/squelch range as intended.
        """
        n = len(samples)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if self._spectrum_window is None or self._spectrum_window.shape[0] != n:
            # Periodic Hann (the form used for spectral analysis).
            self._spectrum_window = np.hanning(n + 1)[:-1].astype(np.float64)
            self._spectrum_window_gain = float(np.sum(self._spectrum_window))
        windowed = samples * self._spectrum_window
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        magnitude = np.abs(spectrum) / max(self._spectrum_window_gain, 1e-12)
        return (20.0 * np.log10(magnitude + 1e-12)).astype(np.float32)

    def _sstv_listening(self) -> bool:
        """True while the SSTV panel's decoder waits for or receives audio."""
        return HAS_HAM_RADIO and self._sstv_panel.is_receiving()

    def _hint_sstv_needs_fm(self) -> None:
        self._show_status_message(
            "SSTV decoding needs FM audio: set Demodulation > Mode to FM.",
            "warning",
            8000,
        )

    def _on_sstv_start_requested(self) -> None:
        """Tell the user what the SSTV decoder still needs to get audio."""
        mode = self._control_panel._demod_combo.currentText()
        self._sstv_panel.show_fm_needed(None if mode == "FM" else self._mode_text(mode))
        if mode != "FM":
            self._hint_sstv_needs_fm()
        elif not (self._is_running and self._device is not None):
            self._show_status_message(
                "SSTV decoder ready. Start receiving (Space) to feed it audio.",
                "info",
                6000,
            )

    # -- SSTV and S-meter hooks ------------------------------------------

    def _connect_sstv_and_smeter(self) -> None:
        sstv = self._sstv_panel
        sstv.tune_iss_requested.connect(self._on_sstv_tune_iss)
        sstv.switch_to_fm_requested.connect(self._on_sstv_switch_to_fm)
        sstv.mode_choice_changed.connect(
            lambda text: self._save_setting("ham/sstv_mode", text)
        )
        meter = self._signal_meter_panel
        meter.set_frequency_source(self._current_frequency)
        meter.set_demod_mode(self._control_panel._demod_combo.currentText())
        meter.report_copied.connect(
            lambda text: self._show_status_message(
                text, "success" if text.startswith("Copied") else "info", 4000
            )
        )
        meter.calibration_changed.connect(self._save_signal_meter_calibration)
        self._smeter_device_key: Optional[str] = None

    def _restore_sstv_and_smeter(self) -> None:
        if not HAS_HAM_RADIO:
            return
        choice = self._settings.get_str("ham/sstv_mode", "")
        if choice:
            self._sstv_panel.set_mode_choice(choice)

    def _on_sstv_tune_iss(self) -> None:
        """Tune to ISS: 145.800 MHz, FM with narrowband deviation, then start
        the receiver and the SSTV decoder."""
        self.set_frequency(ISS_FREQUENCY_HZ)
        self._set_demod_mode("FM", "5 kHz", self._bandwidth_for_mode("FM", "5 kHz"))
        if self._device is not None and not self._is_running:
            self._start_acquisition()
        self._sstv_panel.start_listening()
        self._sstv_panel.show_fm_needed(None)
        if self._device is None:
            self._show_status_message(
                "Tuned to 145.800 MHz FM. No receiver: Device > Use Demo Device "
                "or Connect... to receive.",
                "warning",
                10000,
            )
        elif self._is_running:
            self._show_status_message(
                "Tuned to the ISS SSTV downlink (145.800 MHz FM). Listening "
                "for images.",
                "success",
                6000,
            )

    def _tune_iss_from_menu(self) -> None:
        """Tools > SSTV: Tune to ISS: the SSTV panel's button, from the menu."""
        self._show_panel("sstv")
        self._on_sstv_tune_iss()

    def _on_sstv_switch_to_fm(self) -> None:
        """The SSTV panel's "Switch to FM": narrowband FM, as SSTV is sent."""
        self._set_demod_mode("FM", "5 kHz", self._bandwidth_for_mode("FM", "5 kHz"))
        self._sstv_panel.show_fm_needed(None)
        self._show_status_message("Demodulation set to FM for SSTV.", "info", 3000)

    def _reset_signal_meter_peak(self) -> None:
        """A retune or a new mode or channel: the S-meter starts afresh."""
        if HAS_HAM_RADIO and hasattr(self, "_signal_meter_panel"):
            self._signal_meter_panel.reset_peak()

    def _signal_meter_gain(self) -> Optional[float]:
        """The RF gain for the S-meter's calibration (None under AGC: the
        gain the tuner picks isn't known)."""
        if self._agc_enabled:
            return None
        return float(self._control_panel.gain_db())

    def _smeter_device_type(self) -> str:
        """Key the S-meter calibration is stored under: one per device type."""
        if self._device is None:
            return ""
        name = type(self._device).__name__
        if self._demo_mode or name == "MockDevice":
            return "demo"
        return name.lower()

    def _sync_signal_meter(self, running: bool, has_device: bool) -> None:
        """Receiver state and the device type's calibration to the S-meter."""
        meter = self._signal_meter_panel
        meter.set_receiving(running, has_device)
        key = self._smeter_device_type()
        if key and key != getattr(self, "_smeter_device_key", None):
            self._smeter_device_key = key
            offset, gain = None, None
            raw = self._settings.get_str(f"ham/smeter_cal/{key}", "")
            if raw:
                try:
                    saved = json.loads(raw)
                    offset = float(saved["offset_db"])
                    gain = saved.get("gain_db")
                    gain = None if gain is None else float(gain)
                except (ValueError, KeyError, TypeError):
                    logger.warning("Ignoring unreadable S-meter calibration")
                    offset, gain = None, None
            meter.set_calibration(offset, gain)
            meter.set_gain(self._signal_meter_gain())

    def _save_signal_meter_calibration(self, offset: Any, gain: Any) -> None:
        key = self._smeter_device_type()
        if not key:
            self._show_status_message(
                "Connect a device to save the S-meter calibration for it.",
                "warning",
                6000,
            )
            return
        value = (
            ""
            if offset is None
            else json.dumps({"offset_db": float(offset), "gain_db": gain})
        )
        self._settings.set(f"ham/smeter_cal/{key}", value)
        if offset is None:
            self._show_status_message("S-meter calibration cleared.", "info", 3000)
        else:
            self._show_status_message(
                f"S-meter calibrated for this device type ({float(offset):+.1f} dB).",
                "success",
                4000,
            )

    def _route_audio(self, channel: np.ndarray, level_db: float) -> None:
        """Measure the tuned channel, run the squelch, demodulate once and
        hand the audio to its consumers (the speaker, the SSTV decoder).

        The level is the power of the whole block's channel (in USB, LSB
        and CW exactly the passband heard): the toolbar meter, the Info tab
        and the squelch all use it. ``level_db`` (the strongest spectrum
        bin of the newest samples) only stands in for an empty block.

        The squelch opens at its threshold and closes only after the level
        has stayed 3 dB below it for the hang time (0.3 s, 0.5 s in CW); the
        demodulator is reset only once it has closed. Audio fades in when
        the squelch opens, fades out when it closes and crossfades when a
        new receiver chain takes over, so none of these click.
        """
        chain = self._receiver_chain()
        chain.options = self._audio_options
        mode = chain.mode
        audible = mode in _AUDIBLE_MODES
        seconds = len(channel) / max(chain.channel_rate, 1.0)
        gate_level = level_db
        if len(channel):
            level_db = channel_power_db(channel)
            # The squelch decides on at least ~25 ms of signal, however
            # short the block (LEVEL shows the block's own level).
            gate_level = self._level_average.update(level_db, seconds)
        self._last_peak_db = level_db
        noise_db = self._channel_noise_db(chain)
        gate = self._squelch
        gate.hang_s = _CW_SQUELCH_HANG_S if mode == "CW" else _SQUELCH_HANG_S
        if audible:
            gate.update(gate_level, noise_db, seconds)
            if gate.auto and noise_db is not None:
                self._control_panel.show_auto_threshold(gate.last_threshold_db)
        self._update_level_meter(level_db, noise_db, audible)
        speaker = self._audio_enabled and audible and gate.is_open
        # (Muted while the Scanner moves between channels.)
        speaker = speaker and not self._scan_muted
        # (And while the second receiver's Listen plays only that one.)
        second = getattr(self, "_second_rx", None)
        speaker = speaker and not (second is not None and second.mutes_main())
        if speaker and not self._realtime():
            # A few ms of signal per display frame would play as a buzz.
            speaker = False
            if not self._realtime_hint_shown:
                self._realtime_hint_shown = True
                self._realtime_hint_text = self._realtime_hint()
                self._show_status_message(self._realtime_hint_text, "info", 10000)
        elif speaker and self._realtime_hint_shown:
            # Keeping up again (e.g. after another program hogged the
            # computer for a moment): say so, and hint again next time.
            self._realtime_hint_shown = False
            if self._message_label.text() == self._realtime_hint_text:
                self._show_status_message("Audio resumed", "success", 2500)
        # SSTV is FM; it gets every block (no squelch) so its line timing
        # stays intact, and it listens even with the speaker off.
        sstv = mode == "FM" and self._sstv_listening()
        # Record Audio gets every block too: what is heard, and silence
        # while the squelch is closed (or there is no audio in this mode).
        take = self._audio_take
        recording = take is not None and not take.paused
        # A bookmark scan judges a channel by its CTCSS tone, speaker on or
        # off (see scanner_host.WindowScannerHost.scan_signal_present).
        scan_tone = self._scan_checks_tone and mode == "FM" and gate.is_open
        # The block after the speaker stopped is still demodulated: it
        # fades out instead of breaking off.
        fading_out = self._audio_join.playing and not speaker
        demodulate = speaker or sstv or fading_out or recording or scan_tone
        if not demodulate or not audible:
            if self._demod_running:
                # Closed for real (not during the hang): start afresh.
                chain.reset_demodulator()
                self._tone_squelch.reset()
                self._demod_running = False
                self._clip_s = 0.0
            self._tone_blocked = False
            if self._audio_join.playing:
                # (Another mode with no audio: no block to fade out.)
                self._audio.write(self._audio_join.release())
            self._show_detected_tone()
            if recording:
                self._record_audio_block(None, False, seconds, False)
            return
        deviation = self._control_panel.get_fm_deviation()
        try:
            audio = chain.demodulate(channel, deviation)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Audio demod failed: {exc}")
            chain.reset_demodulator()
            self._demod_running = False
            if self._audio_join.playing:
                self._audio.write(self._audio_join.release())
            if recording:
                self._record_audio_block(None, False, seconds, False)
            return
        self._demod_running = True
        tone_ok = True
        if mode == "FM":
            # CTCSS: what tone the station sends, and the tone squelch
            # (broadcast stations send none: not looked for unless asked).
            tones = self._tone_squelch
            tuned = self._current_frequency()
            if tuned != self._tone_channel:
                # Another channel: its tone (if any) must be heard anew.
                # The last second of the old channel's audio kept a tone
                # squelch open on a channel without the tone for 0.9 s.
                self._tone_channel = tuned
                tones.reset()
            self._tone_blocked = False
            if self._tones_checked(deviation):
                if not tones.process(chain.pre_nr_audio):
                    tone_ok = False
                    self._tone_blocked = speaker or scan_tone
                    speaker = False
            elif tones.detected_hz is not None:
                tones.reset()
            snr = None if noise_db is None else level_db - noise_db
            self._check_fm_clipping(chain, seconds, snr)
        self._show_detected_tone()
        join = self._audio_join
        nr = self._audio_options.noise_reduction
        # A new chain, deviation or noise reduction: the audio doesn't
        # continue the last block played.
        new_stream = (
            chain is not self._join_chain
            or deviation != self._join_deviation
            or nr is not self._join_nr
        )
        if recording and take is not None:
            # (Its own crossfade when a new chain takes over.)
            last = take.stream_key
            joined = last is not None and not (
                last[0] is chain and last[1] == deviation and last[2] is nr
            )
            take.stream_key = (chain, deviation, nr)
            self._record_audio_block(audio, gate.is_open and tone_ok, seconds, joined)
        if speaker:
            if join.playing and new_stream:
                out = join.join(audio)  # crossfade from the last sample
            else:
                out = join.play(audio)  # fades in after silence
            self._join_chain, self._join_deviation = chain, deviation
            self._join_nr = nr
            self._audio.write(out)
        elif join.playing:
            # Stopping: fade this block out, or (if it doesn't continue the
            # last one) decay from the last sample played.
            self._audio.write(join.release() if new_stream else join.fade_out(audio))
        if sstv:
            try:
                self._sstv_panel.process_audio(
                    chain.pre_nr_audio, sample_rate=chain.audio_rate
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SSTV decoder failed: %s", exc)

    def _track_noise_floor(self, power: np.ndarray, count: int) -> None:
        """The spectrum's noise floor (for Auto squelch and the SNR)."""
        rate = self._device_sample_rate()
        self._noise_floor.update(power, count / max(rate, 1.0))

    def _channel_noise_db(self, chain: "_ReceiverChain") -> Optional[float]:
        """Noise power (dBFS) of the tuned channel: what an empty channel
        of this width reads. None until a spectrum arrived."""
        return self._noise_floor.channel_noise_db(chain.noise_bandwidth(), chain.rate)

    def _squelch_state(self, audible: bool) -> str:
        """The meter lamp's state (see level_meter.STATES)."""
        if not audible:
            return "idle"
        gate = self._squelch
        if gate.is_open and self._tone_blocked:
            return "tone"  # a signal, but not the chosen CTCSS tone
        if not gate.enabled:
            return "off"
        return "open" if gate.is_open else "closed"

    def _update_level_meter(
        self, level_db: float, noise_db: Optional[float], audible: bool
    ) -> None:
        gate = self._squelch
        threshold = gate.threshold_db(noise_db)
        self._level_meter.set_reading(
            level_db,
            noise_db,
            threshold,
            self._squelch_state(audible),
            gate.auto and noise_db is not None,
        )

    def _tones_checked(self, deviation: float) -> bool:
        """Whether FM audio is searched for a CTCSS tone: with a tone
        squelch set, or at a narrow deviation (broadcast stations send
        none, so they aren't searched unless asked)."""
        return bool(self._tone_squelch.tone_hz) or deviation < _BROADCAST_DEVIATION_HZ

    def _show_detected_tone(self) -> None:
        if self._tone_squelch is not None:
            # (Tones not searched for say so: "None" read as "this
            # repeater sends no tone".)
            checked = self._tones_checked(self._control_panel.get_fm_deviation())
            self._control_panel.set_detected_tone(
                self._tone_squelch.detected_hz, checked
            )

    def _check_fm_clipping(
        self, chain: "_ReceiverChain", seconds: float, snr_db: Optional[float]
    ) -> None:
        """Warn (now and then) while FM audio clips: the station deviates
        more than the FM deviation setting, which distorts it. Only a clear
        signal counts: noise (a squelch tail, an empty channel with the
        squelch off) clips whatever the setting."""
        clear = snr_db is not None and snr_db >= _CLIP_MIN_SNR_DB
        if clear and chain.last_clip_fraction > _CLIP_FRACTION:
            self._clip_s += seconds
        else:
            self._clip_s = max(0.0, self._clip_s - seconds)
            return
        now = time.monotonic()
        if self._clip_s < _CLIP_WARN_AFTER_S or now < self._clip_warn_due:
            return
        self._clip_warn_due = now + _CLIP_WARN_EVERY_S
        self._clip_s = 0.0
        current = self._control_panel.get_fm_deviation_text()
        self._show_status_message(
            f"FM audio is clipping: this station deviates more than "
            f"{current}. Choose a larger FM deviation in Demodulation, such as "
            "75 kHz (broadcast) for an FM radio station.",
            "warning",
            8000,
        )

    def _realtime_hint(self) -> str:
        if self._playback.active and self._playback.speed != 1.0:
            return (
                f"Playing at {self._playback.speed:g}x: audio plays at 1x only "
                "(the spectrum, decoders and recordings keep working)."
            )
        if self._demo_mode:
            return (
                "The demo device can't keep up in real time right now, so "
                "its audio is paused. It resumes by itself when the computer "
                "catches up (a lower sample rate in Receiver > Sample rate "
                "helps)."
            )
        if self._pacer.overloaded:
            return (
                "This computer can't process "
                f"{format_rate(self._device_sample_rate())} in real time right "
                "now, so audio is paused until it catches up. A lower sample "
                "rate (Receiver > Sample rate) helps."
            )
        return (
            "The device isn't delivering samples in real time, so audio is "
            "paused. Try a lower sample rate (Receiver > Sample rate)."
        )

    def _reset_demodulator(self) -> None:
        """Forget the channel filter and demodulator state (after a gap)."""
        self._rx_chain = None
        # The next samples don't follow on: the squelch starts closed, the
        # audio decays to silence (and fades in again) and the I/Q
        # corrections are measured anew.
        self._squelch.reset()
        self._level_average.reset()
        if self._audio_join.playing:
            self._audio.write(self._audio_join.release())
        self._join_chain = None
        self._demod_running = False
        self._tone_squelch.reset()
        self._tone_blocked = False
        self._iq_conditioner.reset()
        self._clip_s = 0.0

    def _demodulate(
        self, samples: np.ndarray, mode: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Channel-filter and demodulate an I/Q block to mono audio.

        Returns ``(audio, audio_rate)`` (about 48 kHz), or None for a
        non-audio mode. The filter and demodulator state carry over to the
        next block, so consecutive blocks give gap-free audio.
        """
        if mode not in _AUDIBLE_MODES or len(samples) == 0:
            return None
        chain = self._rx_chain
        rate = float(self._device_sample_rate())
        if chain is None or chain.mode != mode or chain.rate != rate:
            chain = self._rx_chain = _ReceiverChain(
                rate, mode, self._channel_bandwidth()
            )
        chain.set_offset(self._vfo_offset_hz())
        try:
            audio = chain.demodulate(
                chain.channelize(samples), self._control_panel.get_fm_deviation()
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Audio demod failed: {e}")
            self._reset_demodulator()
            return None
        return audio, chain.audio_rate

    def _update_status(self):
        """Update status bar (and the Info tab while it is visible)."""
        if self._recording:
            self._update_recording_status()
        if self._audio_take is not None:
            self._update_audio_recording_status()
        if self._agc_enabled:
            self._device_controls.show_auto_gain()  # "Auto (25.9 dB)"
        if self._info_panel.isVisible():
            self._refresh_info_panel()
        # (T6) Messages counted while the window was minimized are seen once
        # it is back with the Decoder tab in front.
        self._decoder_tab_seen()

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def _exec_dialog(self, dialog: Any) -> int:
        """Run a modal dialog, then schedule its deletion.

        A dialog parented to the window would otherwise stay alive, hidden,
        until the window closes, and every theme switch would restyle it.
        Its results can still be read until control returns to the event
        loop.
        """
        try:
            return dialog.exec()
        finally:
            dialog.deleteLater()

    def _show_device_dialog(self, start_after: bool = False) -> bool:
        """Show the device connection dialog. Returns True if one was opened.

        Cancel leaves the current device as it was. Connect replaces it: if
        the choice uses the same driver as the current device (typically the
        same dongle again, e.g. to change the sample rate), the current device
        is closed before the dialog opens the new one, or opening would fail
        as "in use". Receiving resumes on the new device if it was running,
        and starts on it with ``start_after`` (Start pressed with no device).
        """
        from .device_dialog import DeviceDialog

        was_running = self._is_running and self._device is not None
        released: List[str] = []
        ended: List[str] = []
        dialog = DeviceDialog(self)
        self._preselect_rate(dialog)
        open_device = getattr(dialog, "_open_device", None)
        driver_class = getattr(DeviceDialog, "_device_class", None)

        def end_playback() -> None:
            # Connecting a device ends a recording playing back (Cancel keeps
            # it playing): back to the device from before, which may be the
            # one chosen again and must then be free to open (below). The
            # new device receives if the file was playing.
            playing = self._playback.device
            if playing is not None:
                ended.append(playing.options.name)
                self._playback.close(restore=True, resume=False, quiet=True)

        def open_replacing_current(entry: Dict[str, Any]) -> Any:
            # (T13) The second receiver's device stays its own: refused
            # before anything here is closed (the dialog shows why).
            problem = self._second_rx.entry_problem(entry)
            if problem:
                raise RuntimeError(problem)
            end_playback()
            try:
                device_class = driver_class(str(entry.get("type", "")))
            except Exception:  # pragma: no cover - defensive
                device_class = None
            if (
                self._device is not None
                and isinstance(device_class, type)
                and isinstance(self._device, device_class)
            ):
                released.append(self._device_display_name())
                self._release_device()
                self._refresh_state_ui()
            return open_device(entry)

        if callable(open_device) and callable(driver_class):
            dialog._open_device = open_replacing_current
        accepted = self._exec_dialog(dialog)
        device = dialog.get_selected_device() if accepted else None
        if device is None:
            if released:
                self._show_status_message(
                    f"Disconnected from {released[0]}. No device is connected.",
                    "warning",
                    8000,
                )
            elif ended:  # (connecting failed after the file was closed)
                back = (
                    f"Press Start (Space) to receive from {self._device_display_name()}."
                    if self._device is not None
                    else "No device is connected."
                )
                self._show_status_message(f"Closed {ended[0]}. {back}", "warning", 8000)
            return False

        end_playback()
        if self._device is not None and device is not self._device:
            self._release_device()
        self._device = device
        # The front end chosen in the dialog is what the device type uses
        # from now on (applied again, with the gain, below).
        choice = getattr(dialog, "front_end_choice", None)
        if callable(choice):
            self._device_controls.remember_dialog_choice(*choice())
        # The dialog's Demo row opens the same simulated device as Device >
        # Use Demo Device: it is paced in real time, and named, the same way.
        self._demo_mode = _is_demo_device(device)
        if self._demo_mode:
            # A later Use Demo Device keeps the rate chosen here.
            self._preferred_rate = self._device_sample_rate()
        # A new device: whether it keeps up is found out afresh.
        self._pacer.reset()
        self._rx_history.clear()
        self._realtime_hint_shown = False
        note = self._apply_controls_to_device()
        name = self._device_display_name()
        logger.info(f"Connected to {name}")
        self._refresh_state_ui()
        if not (was_running or start_after):
            text = f"Connected to {name}. Press Start or Space to receive."
            if note:
                text = f"Connected to {name}. {note} Press Start or Space to receive."
            self._show_status_message(
                text, "warning" if note else "success", 10000 if note else 4000
            )
        else:
            self._start_acquisition()  # reports its own failure
            if self._is_running:
                text = f"Receiving from {name}" + (f". {note}" if note else "")
                self._show_status_message(
                    text, "warning" if note else "success", 10000 if note else 4000
                )
        return True

    def _preselect_rate(self, dialog: Any) -> None:
        """Preselect the current device's (or the --sample-rate) rate; each
        device type gets its own (the one it had last), and its front end."""
        set_preferences = getattr(dialog, "set_preferences", None)
        if callable(set_preferences):
            set_preferences(*self._device_controls.dialog_preferences())
            return
        combo = getattr(dialog, "_rate_combo", None)
        wanted: Optional[float]
        if self._playback.active:
            # Not the recording's rate: the device's from before it.
            wanted = self._playback.previous_rate() or self._preferred_rate
        elif self._device is not None:
            wanted = self._device_sample_rate()
        else:
            wanted = self._preferred_rate
        if combo is None or not wanted or combo.count() == 0:
            return
        rates = [float(combo.itemData(i) or 0.0) for i in range(combo.count())]
        combo.setCurrentIndex(
            min(range(len(rates)), key=lambda i: abs(rates[i] - wanted))
        )

    def _release_device(self) -> None:
        """Stop and close the current device (if any)."""
        # A transmission ends first (and puts the HackRF back as it was).
        self._tx_station.shutdown()
        if self._is_running:
            self._stop_acquisition()
        if self._device is not None:
            try:
                self._device.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Device close failed: {e}")
        self._device = None
        self._demo_mode = False
        self._realtime_hint_shown = False
        self._rx_history.clear()
        self._pacer.reset()
        self._frame_spectrum.reset()
        self._spectrum.reset_peak()  # (another device's levels)
        self._apply_tuning_range()  # back to every receiver's range

    def _disconnect_device(self):
        """Disconnect from device (while a recording plays: close it, and
        the device used before it too)."""
        if self._playback.active:
            device = self._playback.device
            name = device.options.name if device is not None else "the recording"
            before = self._playback.previous_name()
            self._playback.close(restore=False, quiet=True)
            self._show_status_message(
                f"Closed {name}"
                + (f" and disconnected from {before}." if before else ".")
            )
            return
        if self._device is None:
            return
        name = self._device_display_name()
        self._release_device()
        self._refresh_state_ui()
        self._show_status_message(f"Disconnected from {name}")
        logger.info("Device disconnected")

    def _hardware_driver_classes(self) -> List[Any]:
        """Driver classes whose Python package imports (probed once)."""
        if self._hardware_classes is None:
            classes = []
            for _label, package, name, module, class_name in _HARDWARE_DRIVERS:
                if not _driver_importable(package, name):
                    continue
                try:
                    driver = importlib.import_module(f"..devices.{module}", __package__)
                    classes.append(getattr(driver, class_name))
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug(f"{class_name} unavailable: {e}")
            self._hardware_classes = classes
        return self._hardware_classes

    def _missing_drivers(self) -> List[Tuple[str, str]]:
        """(label, install command) of the hardware drivers that aren't
        installed: the same commands as Device > Connect... and the welcome
        screen show."""
        from .device_dialog import install_command

        return [
            (label, install_command(module))
            for label, package, name, module, _cls in _HARDWARE_DRIVERS
            if not _driver_importable(package, name)
        ]

    def _scan_hardware(self) -> List[Tuple[str, str]]:
        """``(key, name)`` of each connected SDR, from installed drivers only.

        Unlike ``DeviceManager.scan_devices`` this logs nothing when a driver
        is missing or nothing is plugged in, so it can run every 2 s.
        """
        found: List[Tuple[str, str]] = []
        seen: Dict[str, int] = {}
        for device_class in self._hardware_driver_classes():
            try:
                infos = device_class.list_devices()
            except Exception as e:
                logger.debug(f"{device_class.__name__} enumeration failed: {e}")
                continue
            for i, info in enumerate(infos or []):
                name = str(getattr(info, "name", "") or f"Device #{i}")
                ident = str(getattr(info, "serial", "") or name)
                key = f"{device_class.__name__}:{ident}"
                # Cheap dongles often share one serial: keep them apart.
                seen[key] = seen.get(key, 0) + 1
                found.append((f"{key}#{seen[key]}", name))
        return found

    def _refresh_devices(self):
        """Device > Scan for Devices: list the SDR hardware plugged in, with
        Connect... (and, when nothing was found, Use Demo Device and the
        commands that install a missing driver, selectable)."""
        logger.info("Scanning for devices...")
        self._hardware_classes = None  # pick up a driver installed meanwhile
        devices = self._scan_hardware()
        self._known_devices = dict(devices)
        missing = [] if devices else self._missing_drivers()
        box = ScanResultsDialog([name for _key, name in devices], missing, self)
        self._exec_dialog(box)
        if box.choice == "connect":
            self._show_device_dialog()
        elif box.choice == "demo":
            self._use_demo_device()

    def _poll_hotplug(self) -> None:
        """Poll for device hot-plug changes and notify on new devices."""
        if not self._hardware_driver_classes():
            return  # no hardware driver installed: nothing can appear
        devices = dict(self._scan_hardware())
        if self._known_devices is None:
            # First poll after init: baseline silently. (An empty baseline is
            # valid, so a device plugged in after starting the app with none
            # attached is still announced.)
            self._known_devices = devices
            return
        added = [
            name for key, name in devices.items() if key not in self._known_devices
        ]
        self._known_devices = devices
        if added:
            for name in added:
                logger.info(f"Device connected: {name}")
            self._show_status_message(
                f"New device detected: {added[0]}. "
                "Use Device > Connect... to open it.",
                "info",
                8000,
            )

    def _use_demo_device(self) -> None:
        """Device > Use Demo Device: switch to the demo device (also from a
        connected SDR, asking first if a recording runs)."""
        if not (self._demo_mode and self._device is not None):
            self._start_demo_mode()
        self._refresh_state_ui()  # the menu item ticked (or not, if cancelled)

    def _start_demo_mode(self) -> bool:
        """Start demo mode with simulated signals. A connected SDR is
        released first; if a recording runs on it, the user is asked (the
        recording would go on with simulated samples). False if cancelled."""
        from .device_dialog import MockDevice

        if (
            self._device is not None
            and not self._demo_mode
            and not self._playback.active
            and not confirm_switch_to_demo(self)
        ):
            return False
        if self._playback.active:
            self._playback.close(restore=True, resume=False, quiet=True)
        if self._device is not None and not self._demo_mode:
            self._release_device()
        if self._device is None:
            self._device = MockDevice()
            # The rate asked for without a device (--sample-rate), else the
            # one the demo device had last time.
            rate = self._preferred_rate or self._device_controls.remembered_rate("demo")
            if rate:
                self._device.set_sample_rate(rate)
        self._demo_mode = True
        # The demo is about seeing (and demodulating) stations: leave the
        # "no audio" mode for FM.
        switched = self._control_panel._demod_combo.currentText() == "None (I/Q)"
        if switched:
            freq = self._current_frequency()
            broadcast = _FM_BROADCAST_BAND[0] <= freq <= _FM_BROADCAST_BAND[1]
            self._set_demod_mode(
                "FM",
                _BROADCAST_FM_DEVIATION if broadcast else None,
                _BROADCAST_FM_BANDWIDTH if broadcast else None,
            )
        self._apply_controls_to_device()
        self._start_acquisition()
        # (The status bar already names the "Demo Device (simulated signals)".)
        self._show_status_message(
            "Demo mode: no hardware needed. "
            + ("Mode set to FM. " if switched else "")
            + "Click a signal to tune to it and scroll to fine-tune; Device > "
            "Connect... opens a real SDR.",
            "info",
            8000,
        )
        logger.info("Demo mode started")
        return True

    def set_sample_rate(self, rate_hz: float) -> bool:
        """Set the sample rate of the current device, live (no reconnecting:
        the decoders, the receiver chain and the plots follow). Receiver >
        Sample rate, Device > Sample Rate and the status bar's rate use it,
        and ``--sample-rate``.

        Without a device (or with the demo device) the rate is also used by
        the demo device from now on; asked for without a device it is also
        preselected in Device > Connect for every device type. It is
        refused, with the reason in the status bar, while a recording plays
        back (the file sets the rate), while recording I/Q (one take holds
        one rate) or while transmitting (a running scan stops itself at the
        next frame, see _scan_frame). Returns True when applied.
        """
        try:
            rate = float(rate_hz)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(rate) and rate > 0):
            return False
        if self._device is not None:
            blocked = self._device_controls.rate_blocked()
            if blocked:
                self._show_status_message(blocked, "warning", 8000)
                return False
        if self._device is None:
            self._device_controls.requested_rate = rate
        if self._device is None or _is_demo_device(self._device):
            self._preferred_rate = rate
        self._tx_station.shutdown()  # (a transmission has its own rate)
        applied = True
        dev = self._device
        if dev is not None:
            # Hardware takes the new rate between two streams, and what it
            # queued at the old rate is dropped (the plots, decoders and
            # recordings would take it for the new one).
            pause = isinstance(dev, SDRDevice) and self._is_running
            if pause:
                try:
                    dev.stop_rx()
                except Exception as e:  # pragma: no cover - driver-dependent
                    logger.debug(f"stop_rx before the rate change failed: {e}")
            try:
                applied = dev.set_sample_rate(rate) is not False
                if not applied:
                    self._show_status_error(
                        f"{self._device_display_name()} didn't take "
                        f"{format_rate(rate)} (Tools > Error History says why)."
                    )
            except Exception as e:
                applied = False
                self._show_status_error(f"Could not set the sample rate: {e}")
            self._apply_baseband_filter()
            if pause:
                dev.discard_queued_samples()
                try:
                    started = dev.start_rx() is not False
                except Exception as e:
                    logger.error(f"Could not restart receiving: {e}")
                    started = False
                if not started:
                    self._is_running = False
                    self._show_status_error(
                        "The device didn't restart receiving after the sample "
                        "rate change: press Start (Space)."
                    )
        self._reset_demodulator()
        # Whether frames keep up depends on the rate: find out anew.
        self._pacer.reset()
        self._realtime_hint_shown = False
        self._rx_history.clear()
        self._refresh_state_ui()
        # A smaller span may no longer hold the tuned channel.
        self._ensure_vfo_fits()
        if self._device is not None and applied:
            # Remembered for the device type; the status bar says what it is.
            self._device_controls.rate_applied(self._device_sample_rate())
        return applied

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _choose_recording(self, caption: str) -> str:
        """Ask for an I/Q file (the dialog opens where the last one was)."""
        path, _filter = file_dialogs.open_file(
            self, self._settings, "iq", caption, OPEN_FILTERS
        )
        return path

    def _recording_options(
        self, path: str, purpose: str, options: Optional[IqFileOptions] = None
    ) -> Optional[IqFileOptions]:
        """How to read ``path`` for ``purpose`` ("play", "convert", "decode"
        or "info"): what its metadata says, or what the user tells Import
        Options when it doesn't (always for a headerless file). ``options``
        already chosen (Recording Info) are asked about only if ``purpose``
        needs more. None when the file can't be read, holds no samples, or
        the user cancelled."""
        name = os.path.basename(path)
        if options is None:
            try:
                options, missing = describe_iq_file(path)
            except (OSError, ValueError) as e:
                logger.warning(f"Can't read the recording {path}: {e}")
                QMessageBox.warning(
                    self, "Can't Read Recording", f"{name} can't be read:\n{e}"
                )
                return None
        else:
            missing = {
                key
                for key, value in (
                    ("rate", options.sample_rate),
                    ("center", options.center_hz),
                )
                if not value
            }
            if options.confirmed:
                # Already asked (Recording Info): only what this needs.
                missing &= REQUIRED.get(purpose, set())
        if options.total_samples <= 0:
            # (Before anything is replaced or retuned.)
            QMessageBox.warning(
                self,
                "Empty Recording",
                f"{name} holds no I/Q samples, so there is nothing to "
                f"{'show' if purpose == 'info' else purpose}.",
            )
            return None
        if needs_options(missing, purpose):
            # (An I/Q file's center is where the LO was: the receiver's is
            # the middle of the plots, not the tuned frequency.)
            guesses = {
                "rate": self._device_sample_rate() if self._device else 0.0,
                "center": self._lo_frequency(),
            }
            dialog = ImportOptionsDialog(options, missing, purpose, guesses, self)
            if not self._exec_dialog(dialog):
                return None
            options = dialog.options()
            if options is None:
                return None
        return options

    def _confirm_stop_recordings(self, name: str) -> bool:
        """Before a recording plays back: a recording in progress (I/Q or
        audio) would go on recording the file, so offer to stop it. True to
        go ahead (stopped, or nothing was recording)."""
        iq = self._recording
        audio = self._audio_take is not None
        if not (iq or audio):
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Recording in Progress")
        box.setText(f"Stop recording and play {name}?")
        detail = (
            "Playing a file switches the receiver to it, so the recording stops "
            "here."
        )
        if iq:
            detail += (
                " The I/Q samples recorded so far stay in the Recording group, "
                "to save."
            )
        if audio:
            detail += " The audio recording is saved as it is."
        box.setInformativeText(detail)
        play_btn = box.addButton("Stop and &Play", QMessageBox.ButtonRole.AcceptRole)
        set_role(play_btn, "primary")
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(play_btn)
        box.setEscapeButton(cancel_btn)
        self._exec_dialog(box)
        if box.clickedButton() is not play_btn:
            return False
        if self._recording:
            self._toggle_recording(False)
        self._stop_audio_recording()
        return True

    def _confirm_stop_auto_saved(self, doing: str, verb: str) -> bool:
        """Before ``doing`` (e.g. "convert x.cu8") replaces the recording
        buffer while an I/Q recording runs with Save automatically on: it
        is saved when it stops, but it does stop, so ask. True to go ahead
        (stopped and saved, or nothing running); Cancel keeps it going."""
        live = self._recording and not self._capture_pending
        if not (live and self._control_panel.is_auto_save()):
            return True
        folder = file_dialogs.display_path(
            file_dialogs.recordings_folder(self._settings)
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Recording in Progress")
        box.setText(f"Stop recording and {doing}?")
        box.setInformativeText(
            f"The recording stops here and is saved automatically to {folder}. "
            "Cancel keeps it going."
        )
        go_btn = box.addButton(f"Stop and {verb}", QMessageBox.ButtonRole.AcceptRole)
        set_role(go_btn, "primary")
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(go_btn)
        box.setEscapeButton(cancel_btn)
        self._exec_dialog(box)
        if box.clickedButton() is not go_btn:
            return False
        self._toggle_recording(False)  # saves it (Save automatically)
        return True

    def _open_recording(
        self, path: str = "", options: Optional[IqFileOptions] = None
    ) -> bool:
        """File > Open Recording (Ctrl+O): play an I/Q recording through the
        receiver. The spectrum and waterfall show the file's span; tuning
        inside it moves the tuned frequency, and the audio, squelch,
        S-meter, SSTV and live decoders work on it. The playback strip
        under the plots pauses, seeks, loops, changes the speed and closes
        it (back to the device from before)."""
        path = path or self._choose_recording("Open Recording")
        if not path:
            return False
        options = self._recording_options(path, "play", options)
        if options is None:
            return False
        if not self._confirm_stop_recordings(options.name):
            return False
        if not self._playback.open(options):
            return False
        half = options.sample_rate / 2.0
        self._show_status_message(
            f"Playing {options.name} "
            f"({format_mhz_range(options.center_hz - half, options.center_hz + half)})."
            " Click a signal to tune to it; Space pauses.",
            "info",
            8000,
        )
        return True

    def _convert_recording(
        self, path: str = "", options: Optional[IqFileOptions] = None
    ) -> bool:
        """File > Convert Recording (Ctrl+Shift+O): load an I/Q file (any
        format, also headerless captures) into the recording buffer, to save
        it in another format. It is read a chunk at a time into a temporary
        file in a background thread (progress, Cancel), never all at once
        into memory. The receiver isn't retuned; the summary offers Tune
        there and Play."""
        path = path or self._choose_recording("Convert Recording")
        if not path:
            return False
        options = self._recording_options(path, "convert", options)
        if options is None:
            return False
        # First the unsaved take (Cancel keeps a running recording going),
        # then the work: Cancel there costs nothing. A recording saved
        # automatically is no loss, but it stops: asked about too.
        if not self._confirm_stop_auto_saved(f"convert {options.name}", "&Convert"):
            return False
        if not self._confirm_discard_recording(f"converting {options.name}"):
            return False
        if self._recording:
            self._toggle_recording(False)  # armed: nothing captured
        need = options.total_samples * 8  # held as complex64
        free = self._free_recording_space()
        if free is not None and free - need < _MIN_FREE_BYTES:
            QMessageBox.warning(
                self,
                "Not Enough Disk Space",
                f"Converting {options.name} needs {need / 1024**2:,.0f} MB of "
                f"space in {file_dialogs.display_path(self._capture_folder())}, "
                f"which has {free / 1024**2:,.0f} MB free. Free up some space, "
                "then try again.",
            )
            return False
        store = _SampleStore(self._capture_folder())
        result = run_job(
            self,
            "Convert Recording",
            f"Reading {options.name}…",
            lambda report: load_recording(options, store.append, report),
        )
        if not result.ok:
            store.discard()
            if result.cancelled:
                self._show_status_message(
                    f"Stopped reading {options.name}; nothing was loaded.", None, 5000
                )
            else:
                logger.error(f"Could not convert {path}: {result.error}")
                QMessageBox.warning(
                    self,
                    "Convert Failed",
                    f"Could not read {options.name}:\n{result.error}",
                )
            return False
        store.finish()
        loaded = result.value
        self._replace_buffer(store)
        self._buffer_meta = (options.center_hz or None, options.sample_rate or None)
        self._buffer_unsaved = False  # the file is on disk already
        self._buffer_source = options.name
        self._take_started = ""
        self._update_take_panel()
        logger.info(
            f"Loaded {loaded.samples} samples from {path} "
            f"({options.sample_format.value}, rate={options.sample_rate}, "
            f"center={options.center_hz})"
        )
        self._show_status_message(
            f"Loaded {options.name}: Save Recording (Ctrl+S) writes it in another "
            "format.",
            "success",
            8000,
        )
        self._show_conversion_summary(options, loaded)
        return True

    def _show_conversion_summary(self, options: IqFileOptions, loaded: Any) -> None:
        """What was loaded, and what to do next: Save As (the conversion),
        Play, or tune the receiver there."""
        unknown = "unknown (not stored in the file)"
        rate = options.sample_rate
        center = options.center_hz
        lines = [
            f"Sample format: {options.sample_format.value}",
            f"Sample rate: {format_rate(rate) if rate else unknown}",
            f"Center frequency: {format_frequency(center) if center else unknown}",
        ]
        if rate:
            lines.append(f"Length: {loaded.samples / rate:.1f} s")
        text = "\n".join(lines) + (
            "\n\nSave As writes it in another format (SigMF keeps the frequency "
            "and rate). The receiver wasn't retuned."
        )
        if loaded.nonfinite:
            text += (
                f"\n\n{loaded.nonfinite:,} samples aren't numbers (NaN or "
                "infinity): the sample format may be wrong."
            )
        box = QMessageBox(self)
        box.setIcon(
            QMessageBox.Icon.Warning
            if loaded.nonfinite
            else QMessageBox.Icon.Information
        )
        box.setWindowTitle("Recording Loaded")
        box.setText(f"Loaded {loaded.samples:,} samples from {options.name}.")
        box.setInformativeText(text)
        save_btn = box.addButton("&Save As...", QMessageBox.ButtonRole.AcceptRole)
        set_role(save_btn, "primary")
        play_btn = box.addButton("&Play", QMessageBox.ButtonRole.ActionRole)
        if not (rate and center):
            play_btn.setToolTip(
                "Play the recording (you'll be asked for the sample rate and "
                "center frequency it doesn't say)"
            )
        tune_btn = None
        if center:
            tune_btn = box.addButton("&Tune There", QMessageBox.ButtonRole.ActionRole)
            tune_btn.setToolTip(
                f"Tune the receiver to {format_frequency(center)}, where the "
                "recording was made"
            )
        close_btn = box.addButton(QMessageBox.StandardButton.Close)
        box.setDefaultButton(save_btn)
        box.setEscapeButton(close_btn)
        self._exec_dialog(box)
        clicked = box.clickedButton()
        if clicked is save_btn:
            self._save_recording()
        elif clicked is play_btn:
            self._open_recording(options.path, options)
        elif tune_btn is not None and clicked is tune_btn:
            if self.set_frequency(center):
                self._show_status_message(
                    f"Tuned to {format_frequency(center)}", "info", 3000
                )

    def _show_recording_info(self, path: str = "") -> None:
        """File > Recording Info: what an I/Q file holds, whether its
        samples look right, and Play / Convert / Decode from there."""
        path = path or self._choose_recording("Recording Info")
        if not path:
            return
        options = self._recording_options(path, "info")
        if options is None:
            return
        dialog = RecordingInfoDialog(options, self)
        self._exec_dialog(dialog)
        action = dialog.action()
        if action == "play":
            self._open_recording(path, options)
        elif action == "convert":
            self._convert_recording(path, options)
        elif action == "decode":
            self._decode_recording(path, options)

    def _decode_default_frequency(self, protocol: str, options: IqFileOptions) -> float:
        """Where the Decode dialog starts: the protocol's usual channel, the
        tuned frequency or the file's center, the first inside the file."""
        from .decoder_panel import PROTOCOL_TUNE

        center, half = options.center_hz, 0.5 * options.sample_rate
        candidates = [
            PROTOCOL_TUNE.get(protocol, (0.0,))[0],
            self._current_frequency(),
        ]
        for freq in candidates:
            if freq and abs(freq - center) < 0.8 * half:
                return float(freq)
        return center

    def _decode_recording(
        self, path: str = "", options: Optional[IqFileOptions] = None
    ) -> bool:
        """File > Decode Recording (and the Decoder tab's Decode File...):
        run the Decoder panel's protocol over a whole I/Q file in a
        background thread, much faster than real time, with progress and
        Cancel. Each message goes to the Decoder tab as it is found, noted
        with the file and where in it."""
        from ..dsp.live_decoders import LIVE_DECODERS

        protocols = list(LIVE_DECODERS)
        path = path or self._choose_recording("Decode Recording")
        if not path:
            return False
        options = self._recording_options(path, "decode", options)
        if options is None:
            return False
        current = self._decoder_panel.protocol() or ""
        protocol = current if current in protocols else protocols[0]
        dialog = DecodeOptionsDialog(
            options,
            protocols,
            protocol,
            self._decode_default_frequency(protocol, options),
            self,
        )
        if not self._exec_dialog(dialog):
            return False
        choice = dialog.choice()
        if choice is None:
            return False
        protocol, offset = choice
        self._show_panel("decoder")
        counts = {"messages": 0}

        def show(batch: Any) -> None:
            for msg, when in batch:
                counts["messages"] += 1
                self._push_decoded_message(
                    msg, note=f"from {options.name} at {when:.1f} s"
                )

        result = run_job(
            self,
            "Decode Recording",
            f"Decoding {protocol} in {options.name}…",
            lambda report, partial: decode_recording(
                options, protocol, offset, report, partial
            ),
            on_partial=show,
        )
        where = (
            f" at {format_frequency(options.center_hz + offset)}"
            if options.center_hz
            else ""
        )
        if result.error is not None:
            logger.warning(f"Decoding {path} failed: {result.error}")
            QMessageBox.warning(
                self,
                "Decode Failed",
                f"Could not decode {protocol} in {options.name}:\n{result.error}",
            )
            return False
        if result.cancelled:
            self._show_status_message(
                f"Decoding stopped: {counts['messages']} message(s) from "
                f"{options.name} so far.",
                None,
                6000,
            )
            return False
        decoded = result.value
        if decoded.valid:
            failed = (
                f" ({decoded.failed} failed their checks)" if decoded.failed else ""
            )
            self._show_status_message(
                f"Decoded {decoded.valid} {protocol} message(s) from "
                f"{options.name}{where}{failed}.",
                "success",
                8000,
            )
        else:
            self._show_status_message(
                f"No {protocol} messages in {options.name}{where}. Check the "
                "protocol and the frequency (the signal must be in the file).",
                "warning",
                10000,
            )
        return True

    # Formats File > Save Recording writes:
    # (file dialog filter, extension, FileFormat name, SampleFormat name)
    _SAVE_FORMATS: Tuple[Tuple[str, str, str, str], ...] = (
        (
            "SigMF, keeps frequency and rate (*.sigmf-data)",
            ".sigmf-data",
            "SIGMF",
            "FLOAT32",
        ),
        ("Complex Float32 (*.cf32)", ".cf32", "RAW", "FLOAT32"),
        ("Complex Int16 (*.cs16)", ".cs16", "RAW", "INT16"),
        ("WAV, 16-bit I/Q (*.wav)", ".wav", "WAV", "INT16"),
        ("Raw I/Q, Float32 (*.raw)", ".raw", "RAW", "FLOAT32"),
        ("Complex Uint8, rtl_sdr (*.cu8)", ".cu8", "RAW", "UINT8"),
        ("Complex Int8, hackrf_transfer (*.cs8)", ".cs8", "RAW", "INT8"),
    )

    def _panel_recording_extension(self) -> str:
        """Extension matching the Recording panel's Format choice."""
        try:
            ext = self._control_panel.get_recording_extension()
            fmt = self._control_panel.get_recording_format()
        except Exception:  # pragma: no cover - defensive
            ext, fmt = "", ""
        if any(spec[1] == ext for spec in self._SAVE_FORMATS):
            return ext
        lower = (fmt or "").lower()
        if lower.startswith("wav"):
            return ".wav"
        if "sigmf" in lower:
            return ".sigmf-data"
        return ".cf32"

    @classmethod
    def _resolve_save_format(
        cls, filename: str, selected_filter: str = ""
    ) -> Tuple[str, Tuple[str, str, str, str]]:
        """Pick the save format for ``filename``.

        A known extension wins; otherwise the selected filter's format is used
        and its extension appended (Qt's static dialog doesn't add one).
        """
        lower = filename.lower()
        if lower.endswith(".sigmf-meta"):
            # SigMF is saved as the data file; its .sigmf-meta is written next to it.
            filename = filename[: -len(".sigmf-meta")] + ".sigmf-data"
            lower = filename.lower()
        for spec in cls._SAVE_FORMATS:
            if lower.endswith(spec[1]):
                return filename, spec
        spec = next(
            (f for f in cls._SAVE_FORMATS if f[0] == selected_filter),
            cls._SAVE_FORMATS[0],
        )
        return filename + spec[1], spec

    def _suggested_iq_name(self, ext: str) -> str:
        """``iq_100.100MHz_2.4MSps_20261030_142233.sigmf-data`` (the take's
        center, rate and start), or a converted file's own name."""
        if self._buffer_source:
            return split_name(self._buffer_source)[0] + ext
        center, rate = self._buffer_meta
        parts = ["iq"]
        if center:
            parts.append(f"{center / 1e6:.3f}MHz")
        if rate:
            parts.append(format_rate(rate).replace(" ", "").replace("/s", "ps"))
        parts.append(self._take_stamp or time.strftime("%Y%m%d_%H%M%S"))
        return "_".join(parts) + ext

    def _save_recording(self) -> bool:
        """Save the recording buffer to an I/Q file. True when saved.

        The file gets the center frequency and sample rate the samples were
        captured at (or read from), not what the receiver is set to now: a
        recording that doesn't say them (a converted headerless file) asks
        for them first. Raw and WAV files get a ``.sigmf-meta`` beside them
        so the rate and frequency survive."""
        if not self._samples_buffer:
            QMessageBox.information(
                self,
                "Nothing to Save",
                "No samples have been recorded yet.\n\n"
                "Start receiving, press Record (Ctrl+Shift+R), stop recording, "
                "then save.",
            )
            return False

        if not self._ask_missing_details():
            return False

        # Preselect the format chosen in the Recording panel.
        ext = self._panel_recording_extension()
        initial = next(f[0] for f in self._SAVE_FORMATS if f[1] == ext)
        filename, selected_filter = file_dialogs.save_file(
            self,
            self._settings,
            "iq",
            "Save Recording",
            self._suggested_iq_name(ext),
            ";;".join(f[0] for f in self._SAVE_FORMATS),
            initial,
        )
        if not filename:
            return False
        return self._write_buffer(filename, selected_filter)

    def _ask_missing_details(self) -> bool:
        """Ask for the buffer's unknown sample rate or center frequency
        (never stamped from the receiver unasked). False if cancelled."""
        center, rate = self._buffer_meta
        if center and rate:
            return True
        values = ask_recording_details(
            self,
            rate or 0.0,
            center or 0.0,
            self._device_sample_rate() if self._device is not None else 0.0,
            self._lo_frequency(),  # (an I/Q recording's center: the LO)
            exec_dialog=self._exec_dialog,
        )
        if values is None:
            return False
        new_rate, new_center = values
        self._buffer_meta = (new_center or None, new_rate or None)
        return True

    def _write_buffer(
        self, filename: str, selected_filter: str = "", quiet: bool = False
    ) -> bool:
        """Write the recording buffer to ``filename`` (its format from the
        extension or ``selected_filter``), a chunk at a time in a background
        thread; a raw or WAV file gets its metadata in a ``.sigmf-meta``
        beside it. True when saved; failures are reported."""
        from ..dsp.recording import (
            FileFormat,
            SampleFormat,
            wav_data_offset,
            write_sidecar,
        )

        filename, spec = self._resolve_save_format(filename, selected_filter)
        fmt, sample_fmt = FileFormat[spec[2]], SampleFormat[spec[3]]
        store = self._samples_buffer
        count = store.sample_count
        center, rate = self._buffer_meta
        center, rate = float(center or 0.0), float(rate or 0.0)
        name = os.path.basename(filename)

        error: Optional[BaseException] = None
        if (
            fmt is FileFormat.RAW
            and sample_fmt is SampleFormat.FLOAT32
            and store.path is not None
        ):
            # Already in this layout: move (or copy) the file itself.
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                store.save_raw(filename, move=not self._recording)
            except Exception as e:
                error = e
            finally:
                QApplication.restoreOverrideCursor()
        else:
            # The samples so far, mapped here on the GUI thread: the worker
            # only reads them, while a recording still running may go on
            # adding to the file. (So is when the take was recorded, for
            # SigMF's core:datetime.)
            snapshot = store.samples()[:count]
            recorded = self._take_started

            def chunks():
                for start in range(0, len(snapshot), _SAVE_CHUNK):
                    yield np.asarray(snapshot[start : start + _SAVE_CHUNK])

            result = run_job(
                self,
                "Save Recording",
                f"Writing {name}…",
                lambda report: write_recording(
                    chunks(),
                    count,
                    filename,
                    fmt,
                    sample_fmt,
                    rate,
                    center,
                    report,
                    start_time=recorded,
                ),
            )
            if result.cancelled:
                self._show_status_message(f"Not saved: {name} was cancelled.", None)
                return False
            error = result.error
        if error is None and fmt is not FileFormat.SIGMF:
            try:
                write_sidecar(
                    filename,
                    sample_fmt,
                    rate,
                    center,
                    header_bytes=(
                        wav_data_offset(filename) if fmt is FileFormat.WAV else 0
                    ),
                    start_time=self._take_started,
                )
            except OSError as e:
                logger.warning(f"Could not write the metadata of {filename}: {e}")
        if error is not None:
            logger.error(f"Failed to save recording: {error}")
            if quiet:
                self._show_status_error(
                    f"Could not save the recording to {name}: {error}", log=False
                )
            else:
                QMessageBox.warning(
                    self, "Save Failed", f"Could not save recording:\n{error}"
                )
            return False

        self._buffer_unsaved = False
        self._last_saved = filename
        logger.info(f"Saved {count} samples to {filename}")
        self._show_status_message(
            f"Saved {count:,} samples to {file_dialogs.display_path(filename)}",
            "success",
            6000,
        )
        self._update_take_panel()
        return True

    def _import_channels_csv(self) -> None:
        """Import memory channels from a CHIRP CSV into the bookmarks panel."""
        count = self._bookmarks_panel.import_csv()
        if count:
            self._show_panel("bookmarks")
            self._show_status_message(f"Imported {count} channel(s)", "success")

    def _export_channels_csv(self) -> None:
        """Export the saved channels to a CHIRP-compatible CSV file.

        The Bookmarks panel confirms the export (and reports failures) in a
        dialog of its own.
        """
        self._bookmarks_panel.export_csv()

    def _save_screenshot(self) -> None:
        """Save a PNG of the window (spectrum + waterfall + panels), in the
        screenshots folder (Tools > Preferences; then the last one used)."""
        name = (
            f"sdr_{self._current_frequency() / 1e6:.3f}MHz_"
            f"{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        folder = file_dialogs.start_folder(self._settings, "screenshot", create=True)
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", os.path.join(folder, name), "PNG (*.png)"
        )
        if not filename:
            return
        file_dialogs.remember_folder(self._settings, "screenshot", filename)
        # Qt's own file dialog doesn't add the filter's extension.
        if not os.path.splitext(filename)[1]:
            filename += ".png"
        pixmap = self.grab()
        if pixmap.save(filename, "PNG"):
            self._show_status_message(f"Screenshot saved to {filename}", "success")
        else:
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Could not save the screenshot to:\n{filename}\n\n"
                "Check that the folder exists and you can write to it.",
            )

    # ------------------------------------------------------------------
    # Tools and dialogs
    # ------------------------------------------------------------------

    def _show_signal_page(self, page: str) -> None:
        """Tools > Signal > ...: the Signal tab, on ``page`` (T10)."""
        self._show_panel("signal")
        self._signal_panel.show_page(page)

    def _show_scanner(self):
        """Tools > Frequency Scanner (Ctrl+F): bring up the Scanner tab.

        The scan runs in the display loop (see ``_scan_frame``), so the
        plots, the audio and the decoders keep running while it scans.
        """
        self._show_panel("scanner", focus=True)

    # ------------------------------------------------------------------
    # Scanner tab and bookmarks (T8)
    # ------------------------------------------------------------------

    def _connect_bookmarks_and_scanner(self) -> None:
        """Bookmarks: a tuned bookmark's filter and tone, status messages,
        the Add row's mode, Scan from Here. Scanner: its host (the window)."""
        bookmarks = self._bookmarks_panel
        bookmarks.bookmark_tuned.connect(self._on_bookmark_tuned)
        bookmarks.message.connect(
            lambda text, tone: self._show_status_message(text, tone or None, 3000)
        )
        bookmarks.set_width_source(lambda: self._control_panel._bw_combo.currentText())
        self._control_panel.demod_changed.connect(lambda _m: bookmarks.refresh_mode())
        self._control_panel._fm_dev_combo.currentTextChanged.connect(
            lambda _t: bookmarks.refresh_mode()
        )
        self._scanner_host = WindowScannerHost(self)
        self._scanner_panel.set_host(self._scanner_host)
        bookmarks.bookmarks_changed.connect(self._scanner_panel.bookmarks_changed)
        bookmarks.scan_requested.connect(self._scan_bookmarks_from)

    def _scan_frame(self, samples: np.ndarray) -> None:
        """Hand a block to a running scan (after everything else saw it).

        The scan stops when the user tunes elsewhere, or a recording starts
        (it keeps the receiver on one frequency).
        """
        panel = getattr(self, "_scanner_panel", None)
        if panel is None or not panel.engine.active:
            return
        host = self._scanner_host
        if self._device_sample_rate() != panel.engine.rate:
            # Its steps and measurements were planned for the old rate.
            panel.stop_scan(
                "Scan stopped: the sample rate changed. Start it again.",
                "warning",
                stay=True,
            )
            return
        if getattr(self, "_lo_lock", None) is not None:
            panel.stop_scan(
                "Scan stopped: a recording is playing.", "warning", stay=True
            )
            return
        tuned = self._current_frequency()
        if host.expected_hz is not None and abs(tuned - host.expected_hz) > 1.0:
            panel.stop_scan(
                f"Scan stopped: you tuned to {format_frequency(tuned)}.",
                "info",
                stay=True,
            )
            return
        if self._recording:
            panel.stop_scan(
                "Scan stopped: the recording keeps the receiver on one frequency.",
                "warning",
                stay=True,
            )
            return
        panel.engine.frame(samples)

    def _scan_bookmarks_from(self, index: int) -> None:
        """Bookmarks > Scan Bookmarks from Here: a bookmark scan from it."""
        self._show_panel("scanner")
        self._scanner_panel.scan_bookmarks_from(index)

    def _on_bookmark_tuned(self, bookmark: Dict[str, Any]) -> None:
        """After a bookmark's ``tune_requested``: its own filter width and
        CTCSS tone squelch (FM: its tone, or none)."""
        try:
            freq = float(bookmark.get("freq_hz") or 0.0)
        except (TypeError, ValueError):
            return
        if abs(self._current_frequency() - freq) >= 0.5:
            return  # refused: the status bar says why
        demod, deviation, width, tone = bookmark_listening(bookmark)
        if not demod:
            return  # it keeps the mode, filter and tone squelch
        if width:
            self._set_demod_mode(demod, deviation or None, width)
        if demod == "FM":
            self._control_panel.set_tone_squelch_hz(tone)
            if tone:
                label = str(bookmark.get("label") or format_frequency(freq))
                self._show_status_message(
                    f"Tuned to {label} (FM, tone squelch {tone:.1f} Hz)", "info", 3000
                )

    def _show_radio_tuner(self):
        """Show the AM/FM radio tuner pop-out window."""
        if self._radio_tuner is None:
            tuner = RadioTunerWidget(self, self._device_sample_rate())
            self._radio_tuner = tuner
            # Connect frequency change to main tuner
            tuner.frequency_changed.connect(self._on_radio_frequency_changed)
            # Stored presets (and names) are kept between sessions.
            self._restore_radio_presets()
            tuner.presets_changed.connect(self._save_radio_presets)
            # Seek steps until a signal opens the squelch.
            tuner.set_signal_probe(self._tuner_probe)
        self._sync_tuner_device()
        self._pilot = None  # (a fresh detector for the station shown)

        # Open on the station being received (or, outside the broadcast
        # bands, showing that the receiver is elsewhere), with its RDS name.
        self._radio_tuner.set_frequency(self._current_frequency())  # no emit
        self._update_tuner_station()

        self._radio_tuner.show()
        self._radio_tuner.raise_()
        self._radio_tuner.activateWindow()

    def _show_transmit_text(self) -> None:
        """Tools > Transmit Text (Ctrl+Shift+T): the text transmit window
        (CW, RTTY, PSK31, ASCII FSK; gui/transmit_text.py)."""
        if self._transmit_text is None:
            from .transmit_text import TransmitTextDialog

            self._transmit_text = TransmitTextDialog(self)
        self._transmit_text.show()
        self._transmit_text.raise_()
        self._transmit_text.activateWindow()
        self._transmit_text.focus_text()

    def _on_radio_frequency_changed(self, freq_hz: float, band: str):
        """Tune the receiver to the radio tuner's station, in its mode."""
        # set_frequency tunes the device (if any) and updates the control
        # panel, toolbar readout and plot axes together. A frequency the
        # device can't tune is refused (the status bar says why) and the
        # tuner shows the frequency still tuned.
        if not self.set_frequency(freq_hz):
            return
        if band == "FM":
            self._set_demod_mode("FM", _BROADCAST_FM_DEVIATION, _BROADCAST_FM_BANDWIDTH)
        elif band == "AM":
            self._set_demod_mode("AM", bandwidth=_BROADCAST_AM_BANDWIDTH)
        tuner = self._radio_tuner
        if tuner is None or not tuner.is_seeking():  # (seek says it itself)
            self._show_status_message(
                f"Tuned to {format_frequency(freq_hz)} ({band})", "info", 2500
            )
        logger.info(f"Tuned to {freq_hz/1e6:.3f} MHz ({band})")

    # -- AM/FM radio tuner: presets, seek, STEREO, device range (T7) -----------

    def _save_radio_presets(self) -> None:
        tuner = self._radio_tuner
        if tuner is None:
            return
        try:
            self._settings.set("ham/radio_presets", json.dumps(tuner.get_presets()))
        except (TypeError, ValueError) as e:  # pragma: no cover - defensive
            logger.debug(f"Could not save the radio presets: {e}")

    def _restore_radio_presets(self) -> None:
        raw = self._settings.get_str("ham/radio_presets", "")
        if not raw or self._radio_tuner is None:
            return
        try:
            saved = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring unreadable saved radio presets")
            return
        if isinstance(saved, dict):
            self._radio_tuner.set_presets(saved)

    def _sync_tuner_device(self) -> None:
        """Tell an open tuner what the device can tune (it explains the AM
        band when the device can't reach it) and whether it is receiving."""
        tuner = self._radio_tuner
        if tuner is None:
            return
        has_range = self._device_tuning_range() is not None
        low, high = self._tuning_limits()
        name = self._device_model_name() if self._device is not None else ""
        tuner.set_device_range(
            low if has_range else None, high if has_range else None, name
        )
        if not (self._is_running and self._device is not None):
            tuner.set_stereo(False)
            if self._pilot is not None:
                self._pilot.reset()

    def _tuner_retuned(self) -> None:
        """A retune: the seek waits for blocks of the new channel, and the
        pilot is listened for afresh."""
        self._tuner_blocks = 0
        self._tuner_signal = False
        if self._pilot is not None:
            self._pilot.reset()
        tuner = self._radio_tuner
        if tuner is not None and tuner.isVisible():
            tuner.set_stereo(False)

    def _tuner_block(
        self,
        channel: np.ndarray,
        chain: "_ReceiverChain",
        power: Optional[np.ndarray] = None,
    ) -> None:
        """Per block while the tuner is open: whether the tuned channel opens
        the squelch with a station centered on it (for seek), and the 19 kHz
        stereo pilot (broadcast FM)."""
        tuner = self._radio_tuner
        if tuner is None or not tuner.isVisible():
            return
        self._tuner_blocks += 1
        gate = self._squelch
        noise = self._channel_noise_db(chain)
        if gate.enabled or noise is None:
            threshold = gate.threshold_db(noise)
        else:  # squelch off: seek still needs a threshold (Auto's)
            threshold = noise + gate.margin_db
        level = self._last_peak_db
        self._tuner_signal = level is not None and level >= threshold
        # A station half in the channel (a 200 kHz FM filter tuned 100 kHz
        # beside one) opens the squelch too: seek stops only where the
        # channel's power is centered, within half a channel step.
        centroid = None if power is None else _channel_centroid_hz(power, chain)
        step = band_step(tuner.get_band())
        self._tuner_centered = centroid is None or abs(centroid) <= 0.5 * step
        stereo = False
        broadcast = (
            chain.mode == "FM"
            and self._control_panel.get_fm_deviation() >= _BROADCAST_DEVIATION_HZ
            and not tuner.is_off_band()
        )
        if broadcast and len(channel):
            pilot = self._pilot
            if pilot is None or pilot.rate != chain.channel_rate:
                pilot = self._pilot = StereoPilotDetector(chain.channel_rate)
            stereo = pilot.process(channel)
        elif self._pilot is not None:
            self._pilot.reset()
        tuner.set_stereo(stereo)

    def _tuner_probe(self) -> Optional[str]:
        """For the tuner's seek: "signal" (the tuned channel opens the
        squelch), "quiet", "wait" (too few blocks of it yet), "refused"
        (the receiver didn't tune where the tuner is) or None (not
        receiving: seek can't tell)."""
        tuner = self._radio_tuner
        if not self._is_running or self._device is None or tuner is None:
            return None
        if abs(self._current_frequency() - tuner.get_frequency()) > 1.0:
            return "refused"
        if self._tuner_blocks < 3:
            return "wait"
        return "signal" if self._tuner_signal and self._tuner_centered else "quiet"

    def _show_decoder_config(self):
        """Bring the Decoder panel to the front."""
        self._show_panel("decoder")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SDR Module",
            f"<h3>SDR Module {__version__}</h3>"
            "<p>Software-defined radio for signal visualization, frequency "
            "analysis and protocol decoding.</p>"
            "<p><b>Hardware:</b> RTL-SDR and HackRF One, or the built-in "
            "demo device.</p>"
            "<p><b>Features:</b> spectrum and waterfall displays; AM, FM, SSB "
            "and CW demodulation; POCSAG, FLEX, ADS-B, ACARS, AX.25/APRS, RDS "
            "and Morse code decoders; I/Q recording; ham radio tools.</p>"
            f"<p>Qt {QT_VERSION_STR} &middot; PyQt {PYQT_VERSION_STR}</p>",
        )

    def _show_help(self) -> None:
        """Open the keyboard shortcuts reference."""
        from .help_dialog import HelpDialog

        self._exec_dialog(HelpDialog(self))

    def _show_error_history(self) -> None:
        """Open the error history viewer."""
        from .error_log_dialog import ErrorLogDialog, install_history_handler

        install_history_handler()
        self._exec_dialog(ErrorLogDialog(self))

    # ------------------------------------------------------------------
    # Tuning helpers
    # ------------------------------------------------------------------

    def _apply_band_preset(self, freq_hz: float, mode: str, label: str) -> None:
        """Tune to a band preset in its mode, FM deviation and bandwidth."""
        if not self.set_frequency(freq_hz):
            return  # refused: the status bar says why
        preset = self._preset_for(freq_hz)
        if preset is not None and preset.mode == mode:
            self._set_demod_mode(mode, preset.fm_deviation, preset.bandwidth)
        else:
            self._set_demod_mode(mode)
        self._show_status_message(
            f"{_plain(label)}: {format_frequency(freq_hz)}, {mode}, "
            f"{self._control_panel._bw_combo.currentText()} bandwidth",
            "info",
            3000,
        )

    @staticmethod
    def _preset_for(freq_hz: float) -> Optional[BandPreset]:
        """The band preset tuned to exactly ``freq_hz``, if any."""
        return next(
            (p for p in BAND_PRESETS if abs(p.frequency_hz - freq_hz) < 1.0), None
        )

    def _on_panel_preset_applied(self) -> None:
        """Confirm the control panel's Apply Preset in the status bar."""
        panel = self._control_panel
        name = panel._preset_combo.currentText()
        if not name:
            return
        self._show_status_message(
            f"{name}: {format_frequency(self._current_frequency())}, "
            f"{panel._demod_combo.currentText()}, "
            f"{panel._bw_combo.currentText()} bandwidth",
            "info",
            3000,
        )

    def _bookmark_current_frequency(self) -> None:
        """Ctrl+B: bookmark the tuned frequency with the mode (and FM
        deviation) it is listened to in."""
        self._bookmark_frequency(self._current_frequency())

    def _bookmark_frequency(self, freq: float) -> None:
        """Bookmark ``freq`` (Hz) in the current listening mode, filter and
        tone squelch (Ctrl+B, or a signal right-clicked on the waterfall).

        It is named after its preset or band ("FM Broadcast 100.1"). A
        frequency already bookmarked in this mode isn't added twice: that
        bookmark is selected instead.
        """
        mode, deviation = self._listening_mode()
        panel = self._bookmarks_panel
        index = panel.find_bookmark(freq, mode, deviation)
        if index >= 0:
            panel.select_bookmark(index)
            existing = panel.bookmark(index) or {}
            name = str(existing.get("label") or format_frequency(freq))
            self._show_status_message(f"Already bookmarked: {name}", "info", 3000)
            return
        label = default_bookmark_name(freq)
        width = self._control_panel._bw_combo.currentText() if mode else ""
        index = panel.add_bookmark(label, freq, mode, deviation, width)
        # The tone squelch belongs to the channel listened to, not to
        # another signal right-clicked on the waterfall.
        tuned = abs(freq - self._current_frequency()) < 0.5
        tone = self._control_panel.tone_squelch_hz() if mode == "FM" and tuned else 0.0
        if tone:
            panel.set_tone(index, tone)
        self._show_status_message(
            f"Bookmarked {label} ({self._mode_text(mode)}). Rename it with F2 "
            "in the Bookmarks tab.",
            "success",
            3000,
        )

    # ------------------------------------------------------------------
    # Theme and audio
    # ------------------------------------------------------------------

    def _set_theme(self, name: str) -> None:
        """Apply and persist a theme ("dark" or "light")."""
        name = normalize_theme(name)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, name)  # emits theme_changed -> _on_theme_changed
        else:  # pragma: no cover - no QApplication
            self._on_theme_changed(name)
        self._settings.set("theme", name)

    def _toggle_theme(self) -> None:
        """Switch between the dark and light themes."""
        self._set_theme("light" if current_theme() == "dark" else "dark")
        self._show_status_message(f"{self._theme.title()} theme", None, 1500)

    def _on_theme_changed(self, name: str) -> None:
        try:
            self._theme = normalize_theme(name)
            self._sync_theme_actions()
            self._sync_audio_ui()  # the speaker icon is drawn in theme colors
            if self._info_panel.isVisible():
                self._refresh_info_panel()
        except RuntimeError:  # pragma: no cover - window already destroyed
            pass

    def _sync_theme_actions(self) -> None:
        action = self._theme_actions.get(self._theme)
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _set_audio_enabled(self, enabled: bool) -> None:
        """Turn the speaker on or off (toolbar Audio button, Radio > Audio
        Output) and remember the choice."""
        self._audio_enabled = bool(enabled)
        mode = self._control_panel._demod_combo.currentText()
        ok = self._start_or_stop_audio(mode)
        if self._audio_enabled and not ok:
            # Nothing can play: say why and switch back off, but keep the
            # saved preference so audio is on once a device is plugged in.
            self._audio_enabled = False
            self._sync_audio_ui()
            self._show_status_message(
                f"{self._audio_problem}. Plug in speakers or headphones, then "
                "turn Audio on again.",
                "warning",
                8000,
            )
            return
        self._settings.set("audio_enabled", self._audio_enabled)
        self._sync_audio_ui()
        if self._audio_enabled and mode not in _AUDIBLE_MODES:
            self._show_status_message(
                "Audio on. Choose AM, FM, USB, LSB or CW in Demodulation to hear "
                "signals.",
                "info",
                6000,
            )
        else:
            self._show_status_message(
                "Audio on" if self._audio_enabled else "Audio off", None, 2000
            )

    def _saved_volume(self) -> int:
        try:
            volume = int(self._settings.get_int("audio_volume", _DEFAULT_VOLUME))
        except (TypeError, ValueError):
            volume = _DEFAULT_VOLUME
        return min(100, max(0, volume))

    def _on_volume_changed(self, value: int, save: bool = True) -> None:
        """Toolbar volume slider (0-100 %)."""
        value = int(value)
        self._audio.set_volume(value / 100.0)
        self._volume_slider.setToolTip(f"Volume: {value}%")
        if save:
            self._save_setting("audio_volume", value)

    def _audio_status(self) -> Tuple[str, Optional[str]]:
        """Info tab text and tone for the audio output."""
        if not self._audio.available:
            return "Unavailable (needs QtMultimedia)", "muted"
        if self._audio_enabled:
            if self._is_running and not self._realtime():
                return "On (the signal isn't real time)", "warning"
            if self._control_panel.get_demod_mode() not in _AUDIBLE_MODES:
                return "On (no audio in I/Q mode)", "muted"
            return "On", None
        problem = self._audio_problem
        if problem is not None:
            return f"Unavailable ({problem[:1].lower()}{problem[1:]})", "warning"
        return "Off", None

    def _refresh_audio_output(self, announce: bool = True) -> None:
        """Open or close the speaker for the current mode. If it should be
        on but can't open, switch audio off for this session (the saved
        preference is kept) and, with ``announce``, say why."""
        mode = self._control_panel._demod_combo.currentText()
        if not self._start_or_stop_audio(mode) and self._audio_enabled:
            self._audio_enabled = False
            if announce:
                self._show_status_message(
                    f"{self._audio_problem}: audio is off.", "warning", 8000
                )
        self._sync_audio_ui()

    def _sync_audio_ui(self) -> None:
        """Show the audio state on the toolbar button and the menu item."""
        on = bool(self._audio_enabled and self._audio.available)
        for control in (self._audio_action, self._audio_button):
            if control.isChecked() != on:
                control.blockSignals(True)
                control.setChecked(on)
                control.blockSignals(False)
        button = self._audio_button
        button.setText(self._AUDIO_ON_TEXT if on else self._AUDIO_OFF_TEXT)
        button.setIcon(self._speaker_icon(on))
        available = self._audio.available
        button.setEnabled(available)
        self._volume_slider.setEnabled(available)
        if not available:
            tip = "Audio output needs the PyQt6 QtMultimedia module."
        elif on and self._control_panel.get_demod_mode() not in _AUDIBLE_MODES:
            tip = (
                "Audio is on, but None (I/Q) mode has no audio: choose AM, FM, "
                "USB, LSB or CW in Demodulation to hear the tuned signal. "
                "Click to mute (Radio > Audio Output)."
            )
        elif on:
            tip = (
                "Audio is on: the tuned signal plays through your speakers "
                "while it is above squelch. Click to mute (Radio > Audio Output)."
            )
        else:
            tip = "Audio is off. Click to play the tuned signal through your speakers."
            if self._audio_problem is not None:
                tip += f" ({self._audio_problem}.)"
        if available:
            # Which sound card plays it (Tools > Preferences chooses).
            tip += f"\nOutput: {self._audio.output_name()}"
        button.setToolTip(tip)
        if self._info_panel.isVisible():
            self._refresh_info_panel()

    def _speaker_icon(self, on: bool) -> "QIcon":
        """A speaker with sound waves (on) or a cross (muted), drawn in the
        color of the button's text for the current theme."""
        p = get_palette()
        color = p.qcolor("on_accent" if on else "text")
        icon = QIcon()
        for size in (16, 32):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            k = size / 16.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(1.5 * k, 5.5 * k),
                        QPointF(4.5 * k, 5.5 * k),
                        QPointF(8.5 * k, 2.0 * k),
                        QPointF(8.5 * k, 14.0 * k),
                        QPointF(4.5 * k, 10.5 * k),
                        QPointF(1.5 * k, 10.5 * k),
                    ]
                )
            )
            pen = QPen(color)
            pen.setWidthF(1.4 * k)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if on:
                for radius in (3.0, 5.5):
                    painter.drawArc(
                        int(round((8.5 - radius) * k)),
                        int(round((8.0 - radius) * k)),
                        int(round(2 * radius * k)),
                        int(round(2 * radius * k)),
                        -50 * 16,
                        100 * 16,
                    )
            else:
                painter.drawLine(QPointF(10.5 * k, 5.5 * k), QPointF(15 * k, 10.5 * k))
                painter.drawLine(QPointF(15 * k, 5.5 * k), QPointF(10.5 * k, 10.5 * k))
            painter.end()
            icon.addPixmap(pixmap)
        return icon

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore_state(self) -> None:
        """Restore persisted user settings on startup."""
        try:
            freq = self._settings.get_float("frequency_hz", DEFAULT_FREQUENCY_HZ)
            gain = self._settings.get_float("gain_db", 20.0)
            squelch = self._settings.get_float("squelch_db", -80.0)
            agc = self._settings.get_bool("agc_enabled", False)
            # FM by default, so a first launch can be heard right away.
            demod = self._settings.get_str("demod_mode", "FM")
            deviation = self._settings.get_str("fm_deviation", "")
            bandwidth = self._settings.get_str("bandwidth", "")
            rec_format = self._settings.get_str("recording_format", "")
            panel = self._settings.get_str("panel", "")
        except Exception as e:
            logger.debug(f"Settings restore failed: {e}")
            freq, gain, squelch, agc, demod = (
                DEFAULT_FREQUENCY_HZ,
                20.0,
                -80.0,
                False,
                "FM",
            )
            deviation = bandwidth = rec_format = panel = ""
        if not deviation and _FM_BROADCAST_BAND[0] <= freq <= _FM_BROADCAST_BAND[1]:
            deviation = _BROADCAST_FM_DEVIATION  # broadcast FM is wideband
        if math.isfinite(freq):  # (a damaged setting: the nearest end)
            freq = min(max(freq, MIN_FREQUENCY_HZ), MAX_FREQUENCY_HZ)
        # The view as it was left: the LO where it was, if the tuned
        # frequency is still inside its span (else centered on it).
        lo = self._settings.get_float("tuning/lo_hz", freq)
        if not (math.isfinite(lo) and lo > 0 and self._channel_fits(freq - lo)):
            lo = freq
        if math.isfinite(lo) and MIN_FREQUENCY_HZ <= lo <= MAX_FREQUENCY_HZ:
            self._lo_hz = float(lo)
        # set_frequency also updates the toolbar readout and plot axes.
        self.set_frequency(freq)
        self._control_panel.set_gain(gain)
        self._control_panel.set_agc_enabled(agc)
        if bandwidth not in bandwidth_options(demod):
            bandwidth = ""  # saved by a version whose choices differed
        self._restoring = True
        try:
            # The manual squelch threshold, then Auto and on/off; the widths
            # last used with each mode before the mode itself.
            self._control_panel.set_squelch_db(squelch)
            self._restore_audio_settings()
            # The mode (its choices), then its saved deviation and width.
            self._set_demod_mode(demod, deviation or None, bandwidth or None)
            combo = self._control_panel._format_combo
            if rec_format and combo.findText(rec_format) >= 0:
                combo.setCurrentText(rec_format)
            self._control_panel.set_record_kind(
                self._settings.get_str("recording/kind", "iq")
            )
            self._control_panel.set_auto_save(
                self._settings.get_bool("recording/auto_save", False)
            )
            self._control_panel.set_recordings_folder(
                file_dialogs.recordings_folder(self._settings)
            )
            if panel in self._panel_pages:
                self._show_panel(panel)
            self._restore_license_class()
            self._restore_ham_id_settings()
            self._restore_qrp_settings()
            self._restore_waterfall_settings()
            self._restore_spectrum_settings()
            self._load_step_for_mode()  # the restored mode's tuning step
            self._restore_sstv_and_smeter()
        finally:
            self._restoring = False

        # Window geometry
        geom = self._settings.load_geometry("main")
        restored = False
        if geom:
            try:
                restored = bool(self.restoreGeometry(geom))
            except Exception as e:
                logger.debug(f"Could not restore window geometry: {e}")
        if not restored:
            self._apply_default_geometry()

        # Splitter sizes (ignored if a saved pane is squeezed to nothing)
        restored_all = True
        for name, splitter in self._splitters().items():
            state = self._settings.load_geometry(name)
            ok = False
            if state:
                try:
                    ok = bool(splitter.restoreState(state))
                except Exception as e:
                    logger.debug(f"Could not restore {name}: {e}")
            if ok and min(splitter.sizes() or [0]) < 40:
                ok = False
            restored_all = restored_all and ok
        self._splitters_restored = restored_all

    #: Waterfall display defaults when nothing is saved: levels that follow
    #: the noise floor and the strongest signal, and a minute of history.
    _WATERFALL_DEFAULTS = {"auto_levels": True, "history_s": 60.0}

    def _restore_waterfall_settings(self) -> None:
        settings = dict(self._WATERFALL_DEFAULTS)
        raw = self._settings.get_str("waterfall_display", "")
        if raw:
            try:
                saved = json.loads(raw)
            except ValueError:
                logger.debug("Ignoring unreadable waterfall settings: %r", raw)
            else:
                if isinstance(saved, dict):
                    settings.update(saved)
        try:
            self._waterfall.apply_display_settings(settings)
        except Exception as e:
            # (A damaged setting must never stop the app from starting.)
            logger.warning(f"Ignoring the saved waterfall settings: {e}")
            self._waterfall.apply_display_settings(dict(self._WATERFALL_DEFAULTS))

    def _save_waterfall_settings(self) -> None:
        if getattr(self, "_restoring", False):
            return
        try:
            self._settings.set(
                "waterfall_display", json.dumps(self._waterfall.display_settings())
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not save waterfall settings: {e}")

    # ------------------------------------------------------------------
    # Spectrum and waterfall display (T9)
    # ------------------------------------------------------------------

    #: Spectrum display defaults when nothing is saved: a vertical range
    #: that follows the noise floor and the strongest signal, and one trace
    #: (Peak hold is a click away in the header; ux:UX-U09).
    _SPECTRUM_DEFAULTS: Dict[str, Any] = {"range_auto": True, "peak_hold": False}

    def _connect_plots(self) -> None:
        """Both plots show the same part of the span (zoom); the spectrum's
        menus (bookmark, FFT size, window, range, labels) and both plots'
        messages; bookmark labels on the spectrum."""
        sp, wf = self._spectrum, self._waterfall
        # A gain or AGC change moves the noise floor: peak hold starts over.
        sp.reset_on_gain_jump = True
        for plot in (sp, wf):
            plot.set_view_model(self._plot_view)
            plot.message.connect(self._show_plot_message)
            plot.view_changed.connect(self._on_plot_zoomed)
        sp.bookmark_requested.connect(self._bookmark_frequency)
        sp.bookmark_clicked.connect(self._tune_bookmark_label)
        sp.display_settings_changed.connect(self._save_spectrum_settings)
        sp.resolution_changed.connect(self._apply_plot_resolution)
        self._bookmarks_panel.bookmarks_changed.connect(self._sync_bookmark_labels)
        self._sync_bookmark_labels()

    def _plot_frame(self, samples: np.ndarray) -> None:
        """Hand one frame to the spectrum and the waterfall (see
        :class:`~sdr_module.dsp.frame_spectrum.FrameSpectrumAnalyzer`)."""
        rate = float(self._device_sample_rate())
        stamp = self._plot_time(len(samples), rate)
        if self._plot_skip_samples > 0 or time.monotonic() < self._plot_skip_until:
            # Taken (at least partly) on the frequency before the last
            # retune: not drawn at this one (see _drop_stale_samples).
            self._plot_skip_samples = max(0, self._plot_skip_samples - len(samples))
            return
        axis = (float(self._lo_hz), rate)
        if axis != self._frame_axis:
            # Samples kept from the last frame belong to another frequency.
            self._frame_spectrum.reset()
            self._frame_axis = axis
        frame = self._frame_spectrum.process(samples)
        if frame is None:
            return  # less than one FFT of samples yet (a large FFT size)
        self._spectrum.update_spectrum(frame.mean_db)
        self._waterfall.add_line(
            frame.max_db, timestamp=stamp, noise_db=frame.noise_floor_db
        )

    def _plot_time(self, count: int, rate: float) -> float:
        """The time (s) on the waterfall row of a frame of ``count``
        samples: wall-clock time while receiving, but a recording's own
        time while one plays (at 4x a second holds four of the recording),
        so its time axis, ages and measured durations are right at any
        speed. Pauses in reception still show as gaps."""
        now = time.monotonic()
        last, self._plot_clock_wall = self._plot_clock_wall, now
        if last is None:
            self._plot_clock = now
        elif self._playback.active:
            self._plot_clock += count / max(rate, 1.0)
        else:
            self._plot_clock += max(0.0, now - last)
        return self._plot_clock

    def _apply_plot_resolution(self, size: int, window: str, save: bool = True) -> None:
        """Use a ``size``-point FFT with ``window`` for both plots."""
        size, window = normalize_fft_size(size), normalize_window(window)
        changed = self._frame_spectrum.configure(size, window)
        self._spectrum.set_fft_size(size)
        self._spectrum.set_window_name(window)
        self._waterfall.set_fft_size(size)
        if self._info_panel.isVisible():
            self._refresh_info_panel()
        if save and changed:
            rbw = self._frame_spectrum.rbw(self._device_sample_rate())
            self._show_status_message(
                f"Spectrum: {size}-point FFT, {WINDOW_LABELS[window]} window, "
                f"RBW {format_hz(rbw)}",
                None,
                4000,
            )
            self._save_spectrum_settings()

    def _sync_plot_state(self, running: bool, has_device: bool) -> None:
        """Dim the plots, and say why, while nothing new arrives (their data
        is the last received, not live)."""
        playback = self._playback.device if self._playback.active else None
        if running:
            text = None
        elif self._tx_busy():
            text = "Transmitting - receiving resumes when it ends"
        elif not has_device:
            text = "No device - press Space to connect one"
        elif playback is not None and getattr(playback, "finished", False):
            text = "End of the recording - press Space to play it again"
        elif playback is not None:
            text = "Paused - press Space to play"
        else:
            text = "Stopped - press Space"
        for plot in (self._spectrum, self._waterfall):
            plot.set_stale(text)

    def _show_plot_message(self, text: str, tone: str) -> None:
        self._show_status_message(text, tone or None, 3000)

    def _tune_bookmark_label(self, index: int) -> None:
        """A bookmark label on the spectrum was clicked: tune it as the
        Bookmarks tab does (its mode, filter and tone)."""
        self._bookmarks_panel._tune_index(int(index))

    def _sync_bookmark_labels(self) -> None:
        items = []
        for index, bookmark in enumerate(self._bookmarks_panel.bookmarks()):
            try:
                freq = float(bookmark.get("freq_hz") or 0.0)
            except (TypeError, ValueError):
                continue
            if freq > 0:
                items.append((freq, str(bookmark.get("label") or ""), index))
        self._spectrum.set_bookmarks(items)

    # -- zoom ----------------------------------------------------------------
    def _add_zoom_menu(self, view_menu: Any) -> None:
        zoom_menu = view_menu.addMenu("&Zoom")
        self._zoom_in_action = self._add_action(
            zoom_menu,
            "Zoom &In",
            lambda: self._zoom_by(1),
            "Ctrl+=",
            "Show half as much of the span, around the tuned frequency, on "
            "both plots (Ctrl+wheel over a plot zooms around the pointer)",
        )
        self._zoom_in_action.setShortcuts(
            [QKeySequence("Ctrl+="), QKeySequence("Ctrl++")]
        )
        self._zoom_out_action = self._add_action(
            zoom_menu,
            "Zoom &Out",
            lambda: self._zoom_by(-1),
            "Ctrl+-",
            "Show twice as much of the span",
        )
        self._zoom_full_action = self._add_action(
            zoom_menu,
            "&Full Span",
            self._zoom_full,
            "Ctrl+0",
            "Show the whole span the receiver covers",
        )

    def _zoom_anchor(self) -> Optional[float]:
        """The tuned frequency, when it is inside the span (zooming keeps it
        in the middle)."""
        tuned = self._current_frequency()
        return tuned if self._plot_view.in_span(tuned) else None

    def _zoom_by(self, steps: int) -> None:
        view = self._plot_view
        before = view.zoom
        view.zoom_steps(int(steps), self._zoom_anchor(), 0.5)
        if view.zoom == before:
            limit = "in all the way" if steps > 0 else "out to the full span"
            self._show_status_message(f"Already zoomed {limit}", None, 2500)
            return
        self._on_plot_zoomed()

    def _zoom_full(self) -> None:
        if self._plot_view.set_zoom(1):
            self._on_plot_zoomed()

    def _on_plot_zoomed(self) -> None:
        """The zoom changed (menu, keys, Ctrl+wheel): say so, remember it."""
        view = self._plot_view
        if view.is_zoomed():
            text = (
                f"Zoom {view.describe()}: "
                f"{format_mhz_range(view.start, view.end)}. Ctrl+0 shows the "
                "full span."
            )
        else:
            text = "Full span"
        self._show_status_message(text, None, 3000)
        self._save_spectrum_settings()

    # -- settings ------------------------------------------------------------
    def _restore_spectrum_settings(self) -> None:
        settings = dict(self._SPECTRUM_DEFAULTS)
        raw = self._settings.get_str("spectrum_display", "")
        if raw:
            try:
                saved = json.loads(raw)
            except ValueError:
                logger.debug("Ignoring unreadable spectrum settings: %r", raw)
            else:
                if isinstance(saved, dict):
                    settings.update(saved)
        try:
            self._apply_spectrum_settings(settings)
        except Exception as e:
            # (A damaged setting must never stop the app from starting.)
            logger.warning(f"Ignoring the saved spectrum settings: {e}")
            self._apply_spectrum_settings(dict(self._SPECTRUM_DEFAULTS))

    def _apply_spectrum_settings(self, settings: Dict[str, Any]) -> None:
        self._spectrum.apply_display_settings(settings)
        self._apply_plot_resolution(
            settings.get("fft_size", DEFAULT_FFT_SIZE),
            settings.get("window", ""),
            save=False,
        )
        self._plot_view.set_zoom(
            normalize_zoom(settings.get("zoom", 1)), self._zoom_anchor(), 0.5
        )

    def _save_spectrum_settings(self) -> None:
        if getattr(self, "_restoring", False):
            return
        try:
            self._settings.set(
                "spectrum_display", json.dumps(self._spectrum.display_settings())
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not save spectrum settings: {e}")

    def _restore_license_class(self) -> None:
        from ..core.frequency_manager import LicenseClass

        saved = self._settings.get_str("license_class", LicenseClass.NONE.value)
        try:
            license_class = LicenseClass(saved)
        except ValueError:
            logger.warning(f"Unknown saved license class {saved!r}; using None")
            license_class = LicenseClass.NONE
        self._control_panel.set_license_class(license_class)

    def _current_panel_key(self) -> str:
        """Key of the right-hand panel on show (e.g. "bookmarks")."""
        page = self._right_tabs.currentWidget()
        if HAS_HAM_RADIO and page is self._ham_tabs:
            page = self._ham_tabs.currentWidget()
        return next((k for k, p in self._panel_pages.items() if p is page), "")

    def _splitters(self) -> Dict[str, "QSplitter"]:
        return {
            "main_splitter": self._main_splitter,
            "right_splitter": self._right_splitter,
            "display_splitter": self._display_splitter,
        }

    def _persist_state(self) -> None:
        """Save settings on exit."""
        try:
            freq = self._current_frequency()
            gain = float(self._control_panel.slider_gain_db())
            self._settings.set("frequency_hz", freq)
            self._settings.set("tuning/lo_hz", self._lo_frequency())
            self._settings.set("gain_db", gain)
            self._settings.set("squelch_db", self._squelch_db)
            self._settings.set("agc_enabled", self._agc_enabled)
            self._settings.set("theme", self._theme)
            panel = self._control_panel
            self._settings.set("demod_mode", panel._demod_combo.currentText())
            self._settings.set("fm_deviation", panel.get_fm_deviation_text())
            self._settings.set("bandwidth", panel._bw_combo.currentText())
            self._settings.set("recording_format", panel._format_combo.currentText())
            self._settings.set("audio_volume", self._volume_slider.value())
            self._persist_audio_settings()
            current = self._current_panel_key()
            if current:
                self._settings.set("panel", current)
            self._save_ham_id_settings()
            self._save_qrp_settings()
            self._save_radio_presets()
            self._save_waterfall_settings()
            self._save_spectrum_settings()
            self._settings.save_geometry("main", self.saveGeometry())
            if self._spectrum.isVisible() and self._waterfall.isVisible():
                for name, splitter in self._splitters().items():
                    self._settings.save_geometry(name, splitter.saveState())
            self._settings.sync()
        except Exception as e:
            logger.debug(f"Settings save failed: {e}")

    def _run_first_run_wizard(self) -> None:
        """Show the welcome wizard, then tune to the chosen starting band in
        the mode that suits it (starting Demo Mode first if asked)."""
        from .first_run_wizard import FirstRunWizard

        # A connected radio counts as hardware, so the wizard doesn't offer
        # to replace it with the demo device; the wizard names the radio.
        if self._device is not None and not self._demo_mode:
            hardware, device_name = True, self._device_display_name()
        else:
            found = self._scan_hardware()
            hardware, device_name = bool(found), (found[0][1] if found else "")
        wiz = FirstRunWizard(self, hardware_found=hardware, device_name=device_name)
        wiz.demo_mode_requested.connect(self._start_demo_mode)
        # "Connect ... and start receiving": open Device > Connect once the
        # starting band is tuned, so receiving starts there.
        connect_wanted: List[bool] = []
        connect_signal = getattr(wiz, "connect_requested", None)
        if connect_signal is not None:
            connect_signal.connect(lambda: connect_wanted.append(True))
        if self._exec_dialog(wiz):
            freq = wiz.selected_frequency()
            self.set_frequency(freq)
            preset = self._preset_for(freq)
            if preset is not None:
                self._set_demod_mode(preset.mode, preset.fm_deviation, preset.bandwidth)
        self._settings.mark_first_run_done()
        if connect_wanted and (self._device is None or self._demo_mode):
            self._show_device_dialog(start_after=True)

    def _show_welcome(self) -> None:
        """Help > Welcome and Quick Start: the first-run wizard again."""
        self._run_first_run_wizard()

    def closeEvent(self, event):
        """Handle window close (offering to save an unsaved recording).

        The prompt comes first and covers a recording still in progress:
        Save or Don't Save stop it, Cancel keeps it recording and the window
        open.
        """
        if not self._confirm_discard_recording("closing SDR Module"):
            event.ignore()
            return
        if self._recording:
            self._toggle_recording(False)  # armed, or nothing left to lose
        self._stop_audio_recording(quiet=True)  # (saved as it records)
        # A scan running: back to where it started, so that is saved.
        self._scanner_panel.stop_scan(stay=False)
        # A recording playing back: the tuning from before it is saved.
        self._playback.shutdown()
        # The second receiver: its recording finished, its device closed and
        # its settings saved (T13).
        self._second_rx.shutdown()
        self._signal_tools.shutdown()  # (T10) no identification left running
        self._commit_pending_history()  # the last frequency joins Recent
        if self._transmit_text is not None:
            self._transmit_text.shutdown()  # (its preview; the draft is kept)
        self._persist_state()
        for timer in (self._display_timer, self._status_timer, self._hotplug_timer):
            timer.stop()
        self._audio.stop()
        # Stops receiving and closes the device; a failing close() is logged
        # instead of leaving the window impossible to close.
        self._release_device()
        # Saved or declined by now: delete the temporary recording file.
        self._samples_buffer.discard()
        self._frame_spectrum.close()  # (its FFT helper thread)

        logger.info("Application closing")
        event.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frequency(self, freq_hz: float) -> bool:
        """Tune to ``freq_hz`` (the VFO). True when tuned.

        While the tuned channel fits inside the usable inner part of the span
        (``_USABLE_SPAN_FRACTION``), only the VFO moves: the channel slides
        across the plots, which stay put. Otherwise the LO (the device) is
        retuned so the VFO sits in the middle. With Radio > Keep Tuned
        Frequency Centered the LO always follows (every tune retunes the
        device). While the LO is locked (:meth:`_lock_lo`, a recording
        playing back) frequencies outside its span are refused.

        A frequency outside the connected device's range, or one the device
        rejects, is refused: the status bar says why ("RTL-SDR tunes
        0.5–1766 MHz") and the Frequency field, FREQ and the plots stay on
        the frequency actually tuned.
        """
        return self._tune_to(freq_hz)

    def set_gain(self, gain_db: float):
        """Set the RF gain slider.

        The panel applies it to the device only while AGC is off (with AGC
        on, the value takes effect when AGC is switched off).
        """
        self._control_panel.set_gain(gain_db)
