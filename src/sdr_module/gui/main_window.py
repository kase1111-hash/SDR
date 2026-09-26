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
import os
import shutil
import time
from collections import deque
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import (
        PYQT_VERSION_STR,
        QT_VERSION_STR,
        QEvent,
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
        QApplication,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from .. import __version__
from ..devices.base import SDRDevice
from ..utils.tooltips import get_short_tip
from .audio_sink import AudioSink
from .bookmarks_panel import BookmarksPanel
from .control_panel import ControlPanel
from .decoder_panel import DecoderPanel
from .settings_store import GuiSettings
from .spectrum_widget import SpectrumWidget
from .themes import (
    apply_theme,
    current_theme,
    get_palette,
    normalize_theme,
    set_role,
    set_tone,
    theme_notifier,
)
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
    BandPreset("&Airband AM", 125.0e6, "AM", None, "25 kHz"),
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


# Samples read per display frame; also the spectrum FFT length.
DISPLAY_BLOCK = 2048
# Equivalent noise bandwidth of the Hann window, in FFT bins.
_HANN_ENBW_BINS = 1.5
_DEFAULT_SAMPLE_RATE = 2.4e6
_AUDIBLE_MODES = ("AM", "FM", "USB", "LSB", "CW")
_AUDIO_RATE = 48000.0  # target rate of the demodulated audio
_NO_VALUE = "—"  # em dash for "not available"
_TX_UNAVAILABLE = "Connect a HackRF One to transmit an ID."
_DEFAULT_VOLUME = 70  # percent
# Audio is only played while the device delivers at least this fraction of
# its sample rate: less than that (e.g. the demo device, which simulates a
# few ms per display frame) would come out as a buzz of short bursts.
_REALTIME_FRACTION = 0.5
_REALTIME_WINDOW_S = 2.0

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
    from ..ham.gui.radio_tuner import RadioTunerWidget, band_for_frequency
    from ..ham.gui.signal_meter_widget import SignalMeterPanel
    from ..ham.gui.sstv_panel import SSTVPanel

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

# Audio passband of the SSB demodulator and the CW filter and beat note.
_SSB_LOW_HZ = 200.0
_SSB_HIGH_HZ = 2800.0
_CW_HALF_WIDTH_HZ = 250.0
_CW_PITCH_HZ = 700.0
# Highest audio frequency kept after AM/FM detection (broadcast FM is 15 kHz).
_MAX_AUDIO_HZ = 15e3
_MAX_TAPS = 2047


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


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
    block. Filtering uses FFT overlap-save, cheap enough for full-rate
    hardware blocks.
    """

    def __init__(self, taps: Optional[np.ndarray], factor: int = 1):
        self._taps = None if taps is None else np.asarray(taps)
        self._factor = max(1, int(factor))
        self._history: Optional[np.ndarray] = None
        self._phase = 0  # index, in the next block, of the next kept sample
        self._spectra: Dict[Tuple[int, bool], np.ndarray] = {}

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._taps is not None and len(x):
            x = self._filter(x)
        out = x[self._phase :: self._factor]
        self._phase = (self._phase - len(x)) % self._factor
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


def _channel_edges(mode: str, bandwidth: float, rate: float) -> Tuple[float, float]:
    """Offsets from the tuned frequency (Hz) of what ``mode`` demodulates
    (and LEVEL, the squelch and the S-meter measure)."""
    if mode == "USB":
        return 0.0, _SSB_HIGH_HZ
    if mode == "LSB":
        return -_SSB_HIGH_HZ, 0.0
    if mode == "CW":
        return -_CW_HALF_WIDTH_HZ, _CW_HALF_WIDTH_HZ
    half = min(max(float(bandwidth), 1e3), float(rate)) / 2.0
    return -half, half


class _ReceiverChain:
    """Selects the tuned channel and turns it into audio.

    ``channelize`` low-pass filters the I/Q block to the channel (the
    Bandwidth control) and decimates it; the S-meter measures that and
    ``demodulate`` turns it into mono audio at about 48 kHz. Without the
    channel filter the FM discriminator locks onto the strongest station
    anywhere in the 2.4 MHz span and the AM detector hears them all.

    USB/LSB keep 200-2800 Hz on their side of the carrier; CW keeps
    +/-250 Hz around it and adds a 700 Hz beat note (BFO).
    """

    def __init__(self, rate: float, mode: str, bandwidth: float):
        self.rate = float(rate)
        self.mode = mode
        self.bandwidth = min(max(float(bandwidth), 1e3), self.rate)
        total = max(1, int(round(self.rate / _AUDIO_RATE)))
        self.audio_rate = self.rate / total
        half = self.bandwidth / 2.0
        if mode in ("USB", "LSB", "CW"):
            first = total  # straight to the audio rate; the sideband is cut there
            need = self.audio_rate
        else:
            # Keep the whole channel (plus a guard band) for AM/FM detection.
            need = max(1.25 * self.bandwidth, self.audio_rate)
            first = max(
                (d for d in range(1, total + 1) if total % d == 0),
                key=lambda d: d if self.rate / d >= need else 0,
            )
            if self.rate / first < need:
                first = 1
        self.channel_rate = self.rate / first
        self._channel = self._channel_stages(first, half, mode)
        self._audio_factor = total // first
        self._audio_cutoff = min(half, _MAX_AUDIO_HZ, 0.45 * self.audio_rate)
        self._design_demodulator()
        self.reset_demodulator()

    def _channel_stages(
        self, factor: int, half: float, mode: str
    ) -> List[_FirDecimator]:
        """Filters that keep +/- ``half`` Hz (the channel) and decimate by
        ``factor`` to ``channel_rate``.

        The channel filter is sharp (stopband at 1.25x the channel edge), so
        a strong station next door doesn't leak into the S-meter or the
        demodulator. It runs after a coarse decimation stage where possible,
        which keeps it short.
        """
        rate = self.rate
        if factor == 1 and half >= 0.45 * rate:
            return []  # the channel is the whole span: nothing to remove
        # Narrow modes cut their sideband later, so this passband only needs
        # to clear that (a lighter filter than the full channel).
        edge = 0.25 if mode in ("USB", "LSB", "CW") else 0.45
        pass_hz = min(half, edge * self.channel_rate)
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
        stages.append(
            _FirDecimator(_lowpass_taps(pass_hz, stop_hz, fine_rate), factor // coarse)
        )
        return stages

    def band_edges(self) -> Tuple[float, float]:
        """Offsets from the tuned frequency (Hz) of what is demodulated."""
        return _channel_edges(self.mode, self.bandwidth, self.rate)

    def _design_demodulator(self) -> None:
        """Design the filters after detection (once per chain)."""
        rate = self.channel_rate
        # AM/FM: audio low-pass + decimation to the audio rate.
        self._audio_taps: Optional[np.ndarray] = None
        # USB/LSB: complex band-pass for one sideband; CW: narrow low-pass.
        self._sideband_taps: Optional[np.ndarray] = None
        if self.mode in ("AM", "FM") and (
            self._audio_factor > 1 or self._audio_cutoff < 0.45 * rate
        ):
            stop = rate / self._audio_factor - self._audio_cutoff
            if self._audio_factor == 1:
                stop = self._audio_cutoff * 1.3
            self._audio_taps = _lowpass_taps(self._audio_cutoff, stop, rate)
        elif self.mode in ("USB", "LSB"):
            width = (_SSB_HIGH_HZ - _SSB_LOW_HZ) / 2.0
            center = (_SSB_HIGH_HZ + _SSB_LOW_HZ) / 2.0
            if self.mode == "LSB":
                center = -center
            taps = _lowpass_taps(width, width + 300.0, rate)
            t = np.arange(len(taps)) - (len(taps) - 1) / 2.0
            shift = np.exp(2j * np.pi * center * t / rate)
            self._sideband_taps = (taps * shift).astype(np.complex64)
        elif self.mode == "CW":
            self._sideband_taps = _lowpass_taps(
                _CW_HALF_WIDTH_HZ, _CW_HALF_WIDTH_HZ + 250.0, rate
            )

    def reset_demodulator(self) -> None:
        """Forget the audio state (after a gap); the channel filter keeps its."""
        self._prev: Optional[complex] = None
        self._level = 0.0  # AGC / AM carrier level
        self._bfo_phase = 0.0
        self._audio: Optional[_FirDecimator] = None
        self._sideband: Optional[_FirDecimator] = None
        if self._audio_taps is not None or self._audio_factor > 1:
            self._audio = _FirDecimator(self._audio_taps, self._audio_factor)
        if self._sideband_taps is not None:
            self._sideband = _FirDecimator(self._sideband_taps)

    def channelize(self, samples: np.ndarray) -> np.ndarray:
        """The tuned channel of an I/Q block, at ``channel_rate``."""
        out = np.asarray(samples, dtype=np.complex64)
        for stage in self._channel:
            out = stage.process(out)
        return out

    def demodulate(self, channel: np.ndarray, fm_deviation: float) -> np.ndarray:
        """Mono float32 audio in [-1, 1] at ``audio_rate``."""
        if len(channel) == 0:
            return np.empty(0, dtype=np.float32)
        if self.mode == "FM":
            # Discriminator; the previous block's last sample keeps the phase
            # difference continuous. Scaled so +/- the deviation is full scale.
            if self._prev is not None:
                prior = np.concatenate(([self._prev], channel[:-1]))
            else:
                prior = np.concatenate((channel[:1], channel[:-1]))
            self._prev = complex(channel[-1])
            detected = np.angle(channel * np.conj(prior)).astype(np.float32)
            gain = self.channel_rate / (2.0 * np.pi * max(float(fm_deviation), 1.0))
            audio = self._to_audio_rate(detected * gain)
            return np.clip(audio, -1.0, 1.0).astype(np.float32)
        if self.mode == "AM":
            # Envelope relative to the carrier: loudness follows the
            # modulation depth, not the signal strength.
            envelope = np.abs(channel).astype(np.float32)
            carrier = float(np.mean(envelope))
            self._level = (
                carrier if self._level <= 0 else (0.8 * self._level + 0.2 * carrier)
            )
            audio = (envelope - self._level) / max(self._level, 1e-9)
            return np.clip(self._to_audio_rate(audio), -1.0, 1.0).astype(np.float32)
        # USB / LSB / CW: keep the sideband (or the CW filter), then real part.
        narrow = self._sideband.process(channel) if self._sideband else channel
        if self.mode == "CW":
            step = 2.0 * np.pi * _CW_PITCH_HZ / self.channel_rate
            phase = self._bfo_phase + step * np.arange(len(narrow))
            self._bfo_phase = float(
                (self._bfo_phase + step * len(narrow)) % (2 * np.pi)
            )
            narrow = narrow * np.exp(1j * phase)
        return self._agc(np.real(narrow).astype(np.float32))

    def _to_audio_rate(self, audio: np.ndarray) -> np.ndarray:
        return self._audio.process(audio) if self._audio else audio

    def _agc(self, audio: np.ndarray) -> np.ndarray:
        """Bring SSB/CW to a steady level: fast attack, ~1 s release."""
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        release = float(np.exp(-len(audio) / max(self.audio_rate, 1.0)))
        self._level = max(peak, self._level * release)
        # The floor (-80 dBFS) keeps an empty channel from being turned up
        # into full-scale hiss.
        out = audio * (0.5 / max(self._level, 1e-4))
        return np.clip(out, -1.0, 1.0).astype(np.float32)


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
                ("state", "State", "Whether samples are being acquired."),
                ("frequency", "Center frequency", get_short_tip("center_frequency")),
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
                    + f" ({DISPLAY_BLOCK}-point FFT, Hann window.)",
                ),
                ("demod", "Demodulation", "Selected in the Demodulation panel."),
                (
                    "bandwidth",
                    "Bandwidth",
                    "Width of the tuned channel that is demodulated and measured "
                    "(Receiver > Bandwidth).",
                ),
                ("gain", "Gain", get_short_tip("gain")),
                ("squelch", "Squelch", "Audio is muted below this level."),
                (
                    "level",
                    "Channel level",
                    "Strongest signal inside the tuned channel (Bandwidth), in "
                    "dBFS (0 dBFS = full scale).",
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
                    "I/Q samples held in memory. File > Save Recording writes "
                    "them to disk.",
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
        ("Ctrl+L", "Type a frequency (or click the FREQ readout)."),
        (
            "← / →",
            "Tune in 10 kHz steps; add Shift for 100 kHz, Ctrl for 1 MHz.",
        ),
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
            set_role(key_label, "badge", "muted")
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

    # Frequency step for Left/Right, by modifier.
    _TUNE_STEPS_HZ = (
        {
            Qt.KeyboardModifier.NoModifier: 10e3,
            Qt.KeyboardModifier.ShiftModifier: 100e3,
            Qt.KeyboardModifier.ControlModifier: 1e6,
        }
        if HAS_PYQT6
        else {}
    )

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
        self._samples_buffer: List[np.ndarray] = []
        # What the buffer holds: (center frequency, sample rate) of the
        # capture or loaded file (None where unknown), and whether it holds
        # live samples that haven't been saved yet.
        self._buffer_meta: Tuple[Optional[float], Optional[float]] = (None, None)
        self._buffer_unsaved = False
        # A new recording replaces the buffer only once samples arrive, so an
        # armed recording that captures nothing keeps the previous one.
        self._capture_pending = False
        self._retune_warned = False
        self._demo_mode = False  # set by _start_demo_mode()
        self._radio_tuner = None  # Pop-out radio tuner window
        self._squelch_db = -80.0  # applied to spectrum gating
        # Cached analysis window for the spectrum display (built lazily to
        # match the sample-block length).
        self._spectrum_window: Optional[np.ndarray] = None
        self._spectrum_window_gain = 1.0
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

        # Persisted settings (frequency, gain, theme, bookmarks live here)
        self._settings = GuiSettings()
        # SDRApplication applies the persisted theme before the window exists;
        # mirror whatever is actually on screen.
        self._theme = current_theme()

        # Audio output: on by default, so a signal is heard as soon as the
        # receiver runs. Why it can't play (no output device), if it can't.
        self._audio = AudioSink()
        self._audio_enabled = self._settings.get_bool("audio_enabled", True)
        self._audio_problem: Optional[str] = None
        self._audio.set_volume(self._saved_volume() / 100.0)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
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
        self._main_splitter.addWidget(self._display_splitter)

        # Right side: control panel over the tabbed tool panels
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._control_panel = ControlPanel()
        self._right_splitter.addWidget(self._scrollable(self._control_panel))
        self._right_tabs = self._build_panel_tabs()
        self._right_splitter.addWidget(self._right_tabs)
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

        self._decoder_panel = DecoderPanel()
        self._add_panel(
            tabs,
            "decoder",
            self._decoder_panel,
            "Decoder",
            "Live protocol decoder output (POCSAG, FLEX, ADS-B, ACARS, "
            "AX.25/APRS, RDS)",
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
        self._add_action(
            file_menu,
            "Import &Recording (convert format)...",
            self._open_recording,
            QKeySequence.StandardKey.Open,
            "Load an I/Q file so you can save it in another format with Save "
            "Recording (playback isn't supported yet)",
        )
        self._add_action(
            file_menu,
            "&Save Recording...",
            self._save_recording,
            QKeySequence.StandardKey.Save,
            "Write the recorded I/Q samples to a file",
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
        self._demo_action = self._add_action(
            device_menu,
            "Use De&mo Device",
            self._start_demo_mode,
            tip="Explore the app with simulated signals (no hardware needed)",
        )
        self._add_action(
            device_menu,
            "&Scan for Devices",
            self._refresh_devices,
            tip="List the SDR devices currently plugged in",
        )

        # ---- Radio ----
        radio_menu = menubar.addMenu("&Radio")
        # Space is handled in keyPressEvent (so focused widgets keep it); the
        # "\tSpace" suffix only shows it in the menu's shortcut column.
        self._start_action = self._add_action(
            radio_menu,
            "&Start Receiving\tSpace",
            self._toggle_acquisition,
            tip="Start or stop acquiring samples",
        )
        self._record_action = QAction("&Record I/Q", self)
        self._record_action.setCheckable(True)
        self._record_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._record_action.setStatusTip("Capture raw I/Q samples into memory")
        self._record_action.triggered.connect(self._toggle_recording)
        radio_menu.addAction(self._record_action)
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
            "Jump to the Frequency field to type a new center frequency",
        )
        self._add_action(
            radio_menu,
            "&Bookmark Current Frequency",
            self._bookmark_current_frequency,
            "Ctrl+B",
            "Add the current frequency to the Bookmarks panel",
        )

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
            "Frequency &Scanner...",
            self._show_scanner,
            "Ctrl+F",
            "Sweep a frequency range and list active channels",
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
        tools_menu.addSeparator()
        self._add_action(
            tools_menu,
            "&Error History...",
            self._show_error_history,
            "Ctrl+E",
            "Show recent warnings and errors",
        )

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
            lambda _checked=False: self._record_action.trigger()
        )
        toolbar.addWidget(self._record_button)
        self._update_record_button_tip()

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

        # Frequency readout
        freq_caption = QLabel("FREQ")
        set_role(freq_caption, "caption")
        toolbar.addWidget(freq_caption)
        self._freq_label = QLabel(format_frequency(100e6))
        set_role(self._freq_label, "lcd")
        self._freq_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._freq_label.setMinimumWidth(
            self._freq_label.fontMetrics().horizontalAdvance("0000.000 MHz") + 26
        )
        self._freq_label.setToolTip(
            "Center frequency. Click here (or press Ctrl+L) to type a new one. "
            "You can also click the spectrum or waterfall, or use Left/Right "
            "(10 kHz), Shift (100 kHz) and Ctrl (1 MHz) while the plots have "
            "focus."
        )
        self._freq_label.setAccessibleName("Center frequency")
        # Clicking the readout jumps to the Frequency field.
        self._freq_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._freq_label.installEventFilter(self)
        toolbar.addWidget(self._freq_label)

        toolbar.addSeparator()

        # Signal level readout
        level_caption = QLabel("LEVEL")
        set_role(level_caption, "caption")
        toolbar.addWidget(level_caption)
        self._level_label = QLabel(f"{_NO_VALUE} dBFS")
        set_role(self._level_label, "lcd-small")
        self._level_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._level_label.setMinimumWidth(
            self._level_label.fontMetrics().horizontalAdvance("-120.0 dBFS") + 20
        )
        self._level_label.setToolTip(
            "Strongest signal inside the tuned channel (Bandwidth), in dBFS "
            "(0 dBFS = full scale). Green while it is above the squelch "
            "threshold."
        )
        self._level_label.setAccessibleName("Signal level")
        toolbar.addWidget(self._level_label)

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
            "Recorded size and free disk space in the working directory"
        )
        self._recording_info_label.setVisible(False)
        statusbar.addPermanentWidget(self._recording_info_label)

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
        self._control_panel.frequency_changed.connect(self._on_frequency_changed)
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
        self._bookmarks_panel.tune_requested.connect(self._on_bookmark_tune)
        # The license class decides where transmitting is allowed; remember it.
        self._control_panel.license_changed.connect(self._on_license_changed)
        # Remembered between sessions, like the mode they belong to.
        self._control_panel._fm_dev_combo.currentTextChanged.connect(
            lambda text: self._save_setting("fm_deviation", text)
        )
        self._control_panel._format_combo.currentTextChanged.connect(
            lambda text: self._save_setting("recording_format", text)
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
            self._callsign_panel.callsign_changed.connect(self._save_ham_id_settings)
            self._callsign_panel.settings_changed.connect(self._save_ham_id_settings)
            self._sstv_panel.start_requested.connect(self._on_sstv_start_requested)
        # Click-to-tune from spectrum and waterfall. set_frequency keeps the
        # control panel, toolbar readout and plot axes in step.
        self._spectrum.frequency_clicked.connect(self.set_frequency)
        self._waterfall.frequency_clicked.connect(self.set_frequency)
        self._right_tabs.currentChanged.connect(self._on_panel_tab_changed)

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
        """Enter in the Frequency field after Ctrl+L: back to the plots."""
        returns = self._entry_returns_focus
        self._entry_returns_focus = False
        # editingFinished also fires when the field loses focus; only move
        # the focus when Enter was pressed (the field still has it).
        if returns and self._frequency_field().hasFocus():
            self._focus_plots()

    def _focus_plots(self) -> None:
        """Give the keyboard focus to the spectrum (or the waterfall when the
        spectrum is hidden), where Space and the arrow keys work."""
        plot = self._spectrum if self._spectrum.isVisible() else self._waterfall
        plot.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def keyPressEvent(self, event):
        """Space starts/stops receiving; Left/Right tune; Esc returns the
        focus to the plots.

        These are handled here instead of as window-wide shortcuts, so they
        only act when the focused widget does not use the key itself: a
        focused slider, combo box, list, tab bar, button or text field keeps
        its normal arrow/Space behavior, and the key only reaches the window
        when nothing else consumed it (e.g. with the spectrum focused).
        """
        key = event.key()
        mods = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if key == Qt.Key.Key_Escape and mods == Qt.KeyboardModifier.NoModifier:
            self._entry_returns_focus = False
            self._focus_plots()
            event.accept()
            return
        if key == Qt.Key.Key_Space and mods == Qt.KeyboardModifier.NoModifier:
            if not event.isAutoRepeat():
                self._toggle_acquisition()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = self._TUNE_STEPS_HZ.get(mods)
            if step is not None:
                self._nudge_frequency(step if key == Qt.Key.Key_Right else -step)
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

    def _show_status_error(self, message: str, duration_ms: int = 6000) -> None:
        """Show a transient error message (red) in the status bar."""
        self._show_status_message(message, "danger", duration_ms)
        logger.warning(f"Status bar error: {message}")

    def _clear_status_message(self) -> None:
        self._message_label.setText("")
        self._message_label.setToolTip("")
        set_tone(self._message_label, None)

    # ------------------------------------------------------------------
    # State presentation
    # ------------------------------------------------------------------

    def _current_frequency(self) -> float:
        return float(self._control_panel._freq_input.get_frequency())

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

        # Menus
        self._disconnect_action.setEnabled(has_device)
        self._demo_action.setEnabled(not has_device)

        # Level readout is only meaningful while samples flow
        if not running:
            self._last_peak_db = None
            self._level_label.setText(f"{_NO_VALUE} dBFS")
            set_tone(self._level_label, None)

        # Plot axes / click-to-tune span
        center = self._current_frequency()
        self._spectrum.set_frequency_range(center, rate)
        self._waterfall.set_frequency_range(center, rate)

        # Only a HackRF One can send the Ham ID.
        if HAS_HAM_RADIO:
            self._callsign_panel.set_tx_available(
                self._device_can_transmit(), _TX_UNAVAILABLE
            )

        # A live decoder is built for one sample rate; rebuild it when the
        # device (and so the rate) changes.
        if self._decoder is not None and self._decoder_rate != rate:
            self._rebuild_decoder()

        if self._recording:
            self._update_rec_clock()
            self._update_recording_status()

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
        if not has_device:
            p.set_value("state", "Not connected", "warning")
        elif running:
            p.set_value("state", "Receiving", "success")
        else:
            p.set_value("state", "Stopped", "muted")
        p.set_value("frequency", format_frequency(self._current_frequency()))
        p.set_value("sample_rate", format_rate(rate) if has_device else _NO_VALUE)
        p.set_value("span", format_bandwidth(rate) if has_device else _NO_VALUE)
        p.set_value(
            "rbw",
            (
                format_bandwidth(_HANN_ENBW_BINS * rate / DISPLAY_BLOCK)
                if has_device
                else _NO_VALUE
            ),
        )
        p.set_value("demod", self._control_panel._demod_combo.currentText())
        p.set_value("bandwidth", self._control_panel._bw_combo.currentText())
        if self._agc_enabled:
            p.set_value("gain", "Automatic (AGC)")
        else:
            p.set_value("gain", f"{self._control_panel._gain_slider.value()} dB")
        p.set_value("squelch", f"{self._squelch_db:.0f} dBFS")
        if running and self._last_peak_db is not None:
            p.set_value("level", f"{self._last_peak_db:.1f} dBFS")
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
        count = sum(len(block) for block in self._samples_buffer)
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
        top = int(height * 0.55)
        self._right_splitter.setSizes([top, max(120, height - top)])

        height = self._display_splitter.height()
        if height < 300:
            height = self.height() - 120
        spectrum = int(height * 0.4)
        self._display_splitter.setSizes([spectrum, max(120, height - spectrum)])

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

    def _on_panel_record_started(self, fmt: str) -> None:
        """Start recording from the control panel's Record button."""
        if not self._start_recording():
            self._control_panel.set_recording_state(False)

    def _on_panel_record_stopped(self) -> None:
        """Stop recording from the control panel's Record button."""
        self._stop_recording()

    def _on_panel_record_paused(self, paused: bool) -> None:
        """Pause or resume capturing from the control panel's Pause button."""
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

    def _sync_record_ui(self, recording: bool) -> None:
        for control in (self._record_action, self._record_button):
            control.blockSignals(True)
            control.setChecked(recording)
            control.blockSignals(False)
        role = "danger" if recording else ""
        if (self._record_button.property("role") or "") != role:
            set_role(self._record_button, role or None)
        self._update_record_button_tip()
        self._recording_label.setVisible(recording)
        self._recording_info_label.setVisible(recording)
        if recording:
            self._update_recording_status()

    def _update_record_button_tip(self) -> None:
        if self._recording:
            tip = "Stop recording (Ctrl+Shift+R)"
        else:
            tip = (
                "Record raw I/Q samples to memory (Ctrl+Shift+R). "
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
    }

    def _on_decoder_protocol_changed(self, text: str) -> None:
        """Build (or clear) the live decoder for the panel's selected protocol."""
        from ..dsp.protocols import ProtocolType, create_protocol_decoder

        proto_value = self._DECODER_PROTOCOLS.get(text)
        if proto_value is None:
            # "Off" (or unknown): no live decoder.
            self._decoder = None
            self._decoder_protocol = None
            return

        rate = self._device_sample_rate()
        try:
            protocol = ProtocolType(proto_value)
            self._decoder = create_protocol_decoder(protocol, sample_rate=float(rate))
            self._decoder_protocol = protocol
            self._decoder_rate = rate
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not create %s decoder: %s", text, exc)
            self._decoder = None
            self._decoder_protocol = None

    def _rebuild_decoder(self) -> None:
        """Recreate the live decoder for the current device's sample rate."""
        self._on_decoder_protocol_changed(
            self._decoder_panel._proto_combo.currentText()
        )

    def _run_decoder(self, samples: np.ndarray) -> None:
        """Feed demodulated samples to the active decoder and show any messages."""
        if self._decoder is None or self._decoder_protocol is None:
            return
        if not self._decoder_panel._enabled_check.isChecked():
            return

        from ..dsp.protocols import demodulate_for_protocol

        baseband = demodulate_for_protocol(samples, self._decoder_protocol)
        if baseband is None:
            return
        try:
            messages = self._decoder.decode(baseband)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Decoder error: %s", exc)
            return
        for msg in messages:
            self._push_decoded_message(msg)

    def _push_decoded_message(self, msg: Any) -> None:
        """Render one decoded message into the decoder panel.

        Rows use the protocol names of the panel's selector, and a message
        that failed its checks is added as invalid (tinted, counted under
        Invalid). A failed frame with nothing decoded in it (e.g. an AX.25
        CRC error, which noise produces about once a second) is left out.
        """
        try:
            address, content = self._describe_message(msg)
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
            self._decoder_panel.add_message(
                self._protocol_label(msg), address, content, valid, raw_text
            )
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
        """(address, content) columns for a decoded message."""
        name = type(msg).__name__

        def text(attr: str) -> str:
            return str(getattr(msg, attr, "") or "").strip()

        def position() -> str:
            lat = getattr(msg, "latitude", 0.0) or 0.0
            lon = getattr(msg, "longitude", 0.0) or 0.0
            return f"{lat:.4f}, {lon:.4f}" if lat and lon else ""

        def joined(*parts: str) -> str:
            return "  ·  ".join(p for p in parts if p)

        if name == "POCSAGMessage":
            return f"{msg.address} (F{msg.function})", text("content")
        if name == "FLEXMessage":
            return str(getattr(msg, "capcode", "") or ""), text("content")
        if name == "ADSBMessage":
            altitude = getattr(msg, "altitude", 0) or 0
            speed = getattr(msg, "velocity", 0.0) or 0.0
            return text("icao_address"), joined(
                text("callsign"),
                f"{altitude:,} ft" if altitude else "",
                position(),
                f"{speed:.0f} kt" if speed else "",
            )
        if name in ("AX25Frame", "APRSMessage"):
            source, dest = text("source"), text("destination")
            address = f"{source}>{dest}" if source or dest else ""
            return address, joined(position(), text("info") or text("comment"))
        if name == "ACARSMessage":
            return text("registration") or text("flight_id"), joined(
                text("label"), text("text")
            )
        if name == "RDSData":
            pi_code = getattr(msg, "pi_code", 0) or 0
            return (
                f"PI {pi_code:04X}" if pi_code else "",
                joined(text("ps_name"), text("radio_text")),
            )
        address = text("address") or text("icao_address")
        return address, text("content") or text("info")

    # ------------------------------------------------------------------
    # Control panel handlers
    # ------------------------------------------------------------------

    def _on_frequency_changed(self, freq_hz: float):
        """Handle frequency change."""
        if self._device:
            try:
                self._device.set_frequency(freq_hz)
            except Exception as e:
                self._show_status_error(f"Could not tune the device: {e}")

        # Update the readout and the plots' axes / click-to-tune mapping
        self._freq_label.setText(format_frequency(freq_hz))
        self._spectrum.set_center_freq(freq_hz)
        self._waterfall.set_center_freq(freq_hz)
        # The Bookmarks "Add" field offers the frequency being listened to.
        self._bookmarks_panel.set_current_frequency(freq_hz)
        # An open radio tuner follows tuning inside its broadcast bands.
        tuner = self._radio_tuner
        if (
            tuner is not None
            and tuner.isVisible()
            and band_for_frequency(freq_hz) is not None
            and tuner.get_frequency() != freq_hz
        ):
            tuner.set_frequency(freq_hz)  # does not emit frequency_changed

        logger.debug(f"Frequency changed to {freq_hz/1e6:.3f} MHz")

    def _on_gain_changed(self, gain_db: float):
        """Apply a manual gain (the panel only emits this while AGC is off)."""
        if self._device:
            try:
                self._device.set_gain(gain_db)
            except Exception as e:
                self._show_status_error(f"Could not set the gain: {e}")
        logger.debug(f"Gain changed to {gain_db:.1f} dB")

    def _apply_controls_to_device(self) -> None:
        """Bring a newly connected device to what the controls show:
        frequency, then the AGC mode, then the manual gain (only while AGC
        is off; with AGC on the tuner owns the gain)."""
        self._on_frequency_changed(self._current_frequency())
        agc = self._control_panel.is_agc_enabled()
        self._on_agc_changed(agc)
        if not agc:
            self._on_gain_changed(float(self._control_panel._gain_slider.value()))

    def _on_bandwidth_changed(self, bw_hz: float):
        """The channel width: it sets the demodulator's channel filter and
        what LEVEL, the squelch and the S-meter measure.

        A device with an analog baseband filter (HackRF) gets it too. An
        RTL-SDR's "bandwidth" is its sample rate, which is chosen in Device >
        Connect, so it is left alone (changing it silently narrowed the span
        while the axes, rate label and decoder kept the old rate).
        """
        self._save_setting("bandwidth", self._control_panel._bw_combo.currentText())
        dev = self._device
        if dev is not None and not self._bandwidth_is_sample_rate(dev):
            rate = self._device_sample_rate()
            try:
                dev.set_bandwidth(bw_hz)
            except Exception as e:
                self._show_status_error(f"Could not set the bandwidth: {e}")
            if self._device_sample_rate() != rate:
                self._refresh_state_ui()  # axes, rate label, decoder
        self._rx_chain = None  # rebuilt for the new channel
        self._update_passband()
        if self._info_panel.isVisible():
            self._refresh_info_panel()
        logger.debug(f"Bandwidth changed to {bw_hz/1e3:.1f} kHz")

    @staticmethod
    def _bandwidth_is_sample_rate(device: Any) -> bool:
        """True for drivers whose set_bandwidth changes the sample rate."""
        try:
            from ..devices.rtlsdr import RTLSDRDevice
        except Exception:  # pragma: no cover - defensive
            return False
        return isinstance(device, RTLSDRDevice)

    def _save_setting(self, key: str, value: Any) -> None:
        """Persist a control's value (not while restoring them)."""
        if not self._restoring:
            self._settings.set(key, value)

    def _on_squelch_changed(self, db: float):
        """Handle squelch threshold change."""
        self._squelch_db = float(db)
        self._settings.set("squelch_db", self._squelch_db)

    def _on_agc_changed(self, enabled: bool):
        """Handle AGC toggle."""
        self._agc_enabled = bool(enabled)
        self._settings.set("agc_enabled", self._agc_enabled)
        if self._device and hasattr(self._device, "set_gain_mode"):
            try:
                # set_gain_mode(auto: bool). Passing the strings "auto"/"manual"
                # made "manual" truthy, so unticking AGC left AGC on.
                self._device.set_gain_mode(self._agc_enabled)
            except Exception as e:
                logger.debug(f"Device AGC set failed: {e}")

    def _on_demod_changed(self, mode: str):
        """Handle demodulator selection change."""
        self._save_setting("demod_mode", mode)
        self._refresh_audio_output(announce=not self._restoring)
        self._update_passband()
        if mode != "FM" and self._sstv_listening():
            self._hint_sstv_needs_fm()

    def _set_demod_mode(
        self,
        mode: str,
        fm_deviation: Optional[str] = None,
        bandwidth: Optional[str] = None,
    ) -> None:
        """Select a demodulation mode (and, for FM, a deviation; and a
        channel bandwidth) in the control panel; its signals apply them."""
        panel = self._control_panel
        if bandwidth:
            index = panel._bw_combo.findText(bandwidth)
            if index >= 0:
                panel._bw_combo.setCurrentIndex(index)
        if mode == "FM" and fm_deviation:
            index = panel._fm_dev_combo.findText(fm_deviation)
            if index >= 0:
                panel._fm_dev_combo.setCurrentIndex(index)
        index = panel._demod_combo.findText(mode)
        if index >= 0:
            panel._demod_combo.setCurrentIndex(index)

    def _on_license_changed(self, license_class: Any) -> None:
        """Remember the license class (it gates where TX is allowed)."""
        value = getattr(license_class, "value", None)
        if isinstance(value, str):
            self._settings.set("license_class", value)

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

    def _start_or_stop_audio(self, mode: str) -> bool:
        """Open the speaker for audible modes, close it otherwise.

        Returns False when audio is on but the output can't be opened (no
        output device, or the sink failed); ``_audio_problem`` then says why.
        """
        if not self._audio_enabled or mode not in _AUDIBLE_MODES:
            self._audio.stop()
            return True
        problem = self._audio_output_problem()
        if problem is None:
            started = self._audio.start(int(_AUDIO_RATE))
            # AudioSink reports success even when Qt couldn't open the device
            # (it then has no output stream).
            if not started or getattr(self._audio, "_io", True) is None:
                self._audio.stop()
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

    def _on_bookmark_tune(self, freq_hz: float, label: str):
        """Tune to a bookmarked frequency."""
        self.set_frequency(freq_hz)
        self._show_status_message(f"Tuned to {label}", "info", 2500)

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
            self._show_status_error(f"Could not start receiving: {e}")
            return
        if started is False:
            self._show_status_error("The device did not start streaming")
            return

        self._is_running = True
        self._rx_history.clear()
        self._refresh_state_ui()
        logger.info("Acquisition started")

    def _stop_acquisition(self):
        """Stop signal acquisition."""
        self._is_running = False
        self._reset_demodulator()  # the next samples don't follow on
        self._rx_history.clear()

        # Update callsign panel
        if HAS_HAM_RADIO:
            self._callsign_panel.set_transmitting(False)

        if self._device:
            try:
                self._device.stop_rx()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"stop_rx failed: {e}")

        self._refresh_state_ui()
        logger.info("Acquisition stopped")

    def _on_callsign_id_requested(self):
        """Handle callsign ID request from the callsign panel."""
        callsign = self._callsign_panel.get_callsign()
        if not callsign:
            QMessageBox.warning(
                self, "No Callsign", "Please enter your callsign in the Ham ID panel."
            )
            self._show_panel("ham_id")
            return

        logger.info(f"Callsign ID requested: {callsign}")

        try:
            from ..ham.callsign import generate_tx_id

            settings = self._callsign_panel.get_settings()

            # Only CW ID is implemented for transmission. Refuse the other
            # modes rather than silently transmitting Morse under their label.
            mode = settings.get("mode", "CW")
            if mode != "CW":
                QMessageBox.information(
                    self,
                    "Mode Not Available",
                    f"{mode} identification is not implemented for transmission "
                    "yet; only CW (Morse) ID can be sent. Select CW (Morse) in "
                    "the Ham ID panel.",
                )
                return

            self._show_status_message(f"Preparing CW ID: DE {callsign}", "info")

            # Generate FM-modulated I/Q samples ready for transmission
            iq_samples = generate_tx_id(
                callsign,
                wpm=settings.get("cw_wpm", 20),
                tone_frequency=settings.get("cw_tone", 700),
                rf_sample_rate=2e6,
                fm_deviation=2500.0,  # Narrowband FM for CW
            )
            logger.info(f"Generated TX ID: {len(iq_samples)} I/Q samples")

            # Attempt transmission
            self._transmit_audio(iq_samples, callsign)

        except Exception as e:
            logger.error(f"Error generating callsign ID: {e}")
            QMessageBox.warning(
                self, "ID Error", f"Failed to generate callsign ID: {e}"
            )

    def _transmit_audio(self, iq_samples: np.ndarray, description: str = "audio"):
        """
        Transmit I/Q samples via HackRF.

        Args:
            iq_samples: Complex I/Q samples to transmit
            description: Description for logging/status
        """
        from ..core.frequency_manager import is_tx_allowed
        from ..devices.hackrf import HackRFDevice

        # Check if we have a TX-capable device
        if self._device is None:
            QMessageBox.warning(
                self,
                "No Device",
                "No SDR device connected. Connect a HackRF for transmission.",
            )
            return

        # Verify it's a HackRF (TX-capable)
        if not isinstance(self._device, HackRFDevice):
            QMessageBox.warning(
                self,
                "TX Not Supported",
                "Connected device does not support transmission.\n"
                "HackRF One is required for TX operations.",
            )
            return

        # Get current frequency for TX validation
        current_freq = self._current_frequency()
        bandwidth = 10e3  # Approximate CW bandwidth

        # Validate TX is allowed at this frequency
        allowed, reason = is_tx_allowed(current_freq, bandwidth)
        if not allowed:
            QMessageBox.critical(
                self,
                "TX Blocked",
                f"Transmission blocked at {current_freq/1e6:.3f} MHz:\n{reason}",
            )
            return

        # Stop RX if running (HackRF is half-duplex)
        was_running = self._is_running
        if was_running:
            self._stop_acquisition()

        try:
            # Update status
            self._show_status_message(f"Transmitting: {description}", "warning")
            if HAS_HAM_RADIO:
                self._callsign_panel.set_transmitting(True)

            # Configure TX gain
            self._device.set_tx_gain(20)  # Moderate TX power

            # Transmit the samples
            logger.info(
                f"Starting TX: {len(iq_samples)} samples at {current_freq/1e6:.3f} MHz"
            )

            # Use write_samples for one-shot transmission
            success = self._device.write_samples(iq_samples)

            if success:
                logger.info(f"TX complete: {description}")
                self._show_status_message(
                    f"Transmission complete: {description}", "success"
                )
            else:
                logger.error("TX failed")
                QMessageBox.warning(
                    self, "TX Failed", "Failed to transmit. Check device connection."
                )

        except Exception as e:
            logger.error(f"TX error: {e}")
            QMessageBox.warning(self, "TX Error", f"Transmission error: {e}")
        finally:
            if HAS_HAM_RADIO:
                self._callsign_panel.set_transmitting(False)

            # Restart RX if it was running
            if was_running:
                self._start_acquisition()

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
        # Keep the control panel's Record button in sync (without re-emitting).
        self._control_panel.set_recording_state(self._recording)

    def _start_recording(self) -> bool:
        """Start (or arm) a recording. False if the user kept an unsaved one.

        The buffer is replaced when the first samples arrive, so arming and
        stopping without capturing anything keeps the previous recording.
        """
        if self._recording:
            return True
        if not self._confirm_discard_recording("starting a new recording"):
            return False
        self._recording = True
        self._recording_paused = False
        self._capture_pending = True
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
        count = self._buffer_sample_count()
        if captured and count:
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
        logger.info("Recording stopped")

    def _buffer_sample_count(self) -> int:
        return sum(len(block) for block in self._samples_buffer)

    def _confirm_discard_recording(self, doing: str) -> bool:
        """Before ``doing`` (e.g. "closing") drops the unsaved recording,
        offer to save it. True to go ahead: saved, discarded or nothing to
        lose; False to cancel."""
        if not (self._buffer_unsaved and self._samples_buffer):
            return True
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
        box.setInformativeText(
            f"{count:,} I/Q samples ({count * 8 / 1e6:.1f} MB) recorded{where} "
            "haven't been saved. If you don't save them, they are lost."
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
        if clicked is save_btn:
            return self._save_recording()
        if clicked is discard_btn:
            self._discard_recording()
            return True
        return False

    def _discard_recording(self) -> None:
        """Empty the recording buffer."""
        self._samples_buffer = []
        self._buffer_meta = (None, None)
        self._buffer_unsaved = False
        self._clear_status_message()  # e.g. "Use File > Save Recording"

    def _recording_capturing(self) -> bool:
        """True while samples are actually being added to the recording."""
        return (
            self._recording
            and not self._recording_paused
            and self._is_running
            and self._device is not None
        )

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
        self._control_panel.update_record_time(int(self._recording_elapsed()))

        mb = self._recording_bytes / (1024 * 1024)
        # Recorded size + free space on the working-directory volume
        try:
            free_gb = shutil.disk_usage(".").free / (1024**3)
            info = f"{mb:.1f} MB · {free_gb:.1f} GB free"
        except OSError:
            info = f"{mb:.1f} MB"
        self._recording_info_label.setText(info)

    # ------------------------------------------------------------------
    # Periodic updates
    # ------------------------------------------------------------------

    def _read_block(self) -> Optional[np.ndarray]:
        """The next block of samples, without stalling the window.

        Hardware devices queue whole USB blocks from a background thread and
        ``read_samples`` waits up to a second for one by default. The display
        timer runs faster than blocks arrive, so waiting froze the GUI for
        most of every second; poll without waiting instead.
        """
        dev = self._device
        if isinstance(dev, SDRDevice):
            return dev.read_samples(DISPLAY_BLOCK, timeout=0.0)
        return dev.read_samples(DISPLAY_BLOCK)

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

        samples = self._read_block()
        if samples is None or len(samples) == 0:
            self._stream_died()
            return
        self._note_block(len(samples))

        # The plots show the newest DISPLAY_BLOCK samples (a fixed FFT size, so
        # the RBW shown in the Info tab holds); everything else gets the block.
        display = samples[-DISPLAY_BLOCK:] if len(samples) > DISPLAY_BLOCK else samples

        # Compute spectrum (windowed, referenced to dBFS)
        power = self._power_spectrum_dbfs(display)

        # Update spectrum widget
        self._spectrum.update_spectrum(power)

        # Update waterfall
        self._waterfall.add_line(power)

        # Level of the tuned channel (green while above the squelch
        # threshold): a strong station elsewhere in the span no longer
        # holds the squelch open.
        chain = self._receiver_chain()
        level = self._channel_level(power, chain)
        self._last_peak_db = level
        self._level_label.setText(f"{level:.1f} dBFS")
        set_tone(self._level_label, "success" if level >= self._squelch_db else None)

        # The tuned channel, filtered out of the block: the S-meter measures
        # it and the demodulator listens to it.
        channel = chain.channelize(samples)

        # Audio: the speaker (while the signal is above squelch) and the SSTV
        # decoder (whenever it listens, speaker on or off).
        self._route_audio(channel, level)

        # Record if active (and not paused). The REC badge and the Recording
        # panel's timer are refreshed by the status timer.
        if self._recording and not self._recording_paused:
            self._capture(samples)

        if HAS_HAM_RADIO and len(channel):
            try:
                self._signal_meter_panel.update_samples(channel)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("S-meter update failed: %s", exc)

        # Feed the live protocol decoder (if the Decoder panel selected one).
        self._run_decoder(samples)

    def _capture(self, samples: np.ndarray) -> None:
        """Add a block to the recording (the first one replaces the buffer)."""
        tuned = (self._current_frequency(), self._device_sample_rate())
        if self._capture_pending:
            self._capture_pending = False
            self._samples_buffer = []
            self._buffer_meta = tuned
            self._retune_warned = False
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
        self._samples_buffer.append(samples)
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
            or (chain.bandwidth != min(max(bandwidth, 1e3), rate))
        ):
            chain = self._rx_chain = _ReceiverChain(rate, mode, bandwidth)
        return chain

    def _channel_bandwidth(self) -> float:
        """The Bandwidth control, in Hz (the channel that is demodulated)."""
        hz = _parse_hz(self._control_panel._bw_combo.currentText())
        return hz if hz else 200e3

    def _channel_level(self, power_db: np.ndarray, chain: "_ReceiverChain") -> float:
        """Strongest spectrum bin inside the tuned channel, in dBFS."""
        n = len(power_db)
        if n == 0:
            return -120.0
        bin_hz = chain.rate / n
        low, high = chain.band_edges()
        center = n // 2  # fftshift puts 0 Hz here
        first = max(0, center + int(np.floor(low / bin_hz + 0.5)))
        last = min(n - 1, center + int(np.ceil(high / bin_hz - 0.5)))
        if last < first:
            first = last = center
        return float(np.max(power_db[first : last + 1]))

    def _update_passband(self) -> None:
        """Show the demodulated channel on the spectrum, when it can."""
        show = getattr(self._spectrum, "set_passband", None)
        if not callable(show):
            return
        try:
            show(
                *_channel_edges(
                    self._control_panel._demod_combo.currentText(),
                    self._channel_bandwidth(),
                    self._device_sample_rate(),
                )
            )
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

        True until there is a second of history to judge by.
        """
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
        if self._control_panel._demod_combo.currentText() != "FM":
            self._hint_sstv_needs_fm()
        elif not (self._is_running and self._device is not None):
            self._show_status_message(
                "SSTV decoder ready. Start receiving (Space) to feed it audio.",
                "info",
                6000,
            )

    def _route_audio(self, channel: np.ndarray, level_db: float) -> None:
        """Demodulate the tuned channel once and hand the audio to its
        consumers."""
        chain = self._receiver_chain()
        mode = chain.mode
        speaker = (
            self._audio_enabled
            and mode in _AUDIBLE_MODES
            and level_db >= self._squelch_db
        )
        if speaker and not self._realtime():
            # A few ms of signal per display frame would play as a buzz.
            speaker = False
            if not self._realtime_hint_shown:
                self._realtime_hint_shown = True
                self._show_status_message(self._realtime_hint(), "info", 10000)
        # SSTV is FM; it gets every block (no squelch) so its line timing
        # stays intact, and it listens even with the speaker off.
        sstv = mode == "FM" and self._sstv_listening()
        if not (speaker or sstv):
            chain.reset_demodulator()
            return
        if mode not in _AUDIBLE_MODES:
            return
        try:
            audio = chain.demodulate(channel, self._control_panel.get_fm_deviation())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Audio demod failed: {exc}")
            chain.reset_demodulator()
            return
        if speaker:
            self._audio.write(audio)
        if sstv:
            try:
                self._sstv_panel.process_audio(audio, sample_rate=chain.audio_rate)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SSTV decoder failed: %s", exc)

    def _realtime_hint(self) -> str:
        if self._demo_mode:
            return (
                "The demo device simulates only a few milliseconds of signal "
                "per screen update, too little to hear. Connect an RTL-SDR or "
                "HackRF One to listen."
            )
        return (
            "The device isn't delivering samples in real time, so audio is "
            "paused. Try a lower sample rate in Device > Connect..."
        )

    def _reset_demodulator(self) -> None:
        """Forget the channel filter and demodulator state (after a gap)."""
        self._rx_chain = None

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
        if self._info_panel.isVisible():
            self._refresh_info_panel()

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
        dialog = DeviceDialog(self)
        self._preselect_rate(dialog)
        open_device = getattr(dialog, "_open_device", None)
        driver_class = getattr(DeviceDialog, "_device_class", None)

        def open_replacing_current(entry: Dict[str, Any]) -> Any:
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
            return False

        if self._device is not None and device is not self._device:
            self._release_device()
        self._device = device
        self._demo_mode = False
        self._apply_controls_to_device()
        name = self._device_display_name()
        logger.info(f"Connected to {name}")
        self._refresh_state_ui()
        if not (was_running or start_after):
            self._show_status_message(
                f"Connected to {name}. Press Start or Space to receive.", "success"
            )
        else:
            self._start_acquisition()  # reports its own failure
            if self._is_running:
                self._show_status_message(f"Receiving from {name}", "success")
        return True

    def _preselect_rate(self, dialog: Any) -> None:
        """Preselect the current device's (or the --sample-rate) rate."""
        combo = getattr(dialog, "_rate_combo", None)
        if self._device is not None:
            wanted: Optional[float] = self._device_sample_rate()
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

    def _disconnect_device(self):
        """Disconnect from device."""
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
        """(label, pip extra) of the hardware drivers that aren't installed."""
        return [
            (label, module)
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
        """Device > Scan for Devices: list the SDR hardware plugged in."""
        logger.info("Scanning for devices...")
        self._hardware_classes = None  # pick up a driver installed meanwhile
        devices = self._scan_hardware()
        self._known_devices = dict(devices)
        if devices:
            names = "\n".join(f"  •  {name}" for _key, name in devices)
            QMessageBox.information(
                self,
                "Devices Found",
                f"Detected {len(devices)} device(s):\n\n{names}\n\n"
                "Use Device > Connect... to open one.",
            )
            return
        text = (
            "No SDR devices were detected.\n\n"
            "Plug in an RTL-SDR or HackRF One and scan again, or use "
            "Device > Use Demo Device to try the app without hardware."
        )
        missing = self._missing_drivers()
        if missing:
            text += "\n\nNot installed:\n" + "\n".join(
                f'  •  {label} support: python -m pip install "sdr-module[{extra}]"'
                for label, extra in missing
            )
        QMessageBox.information(self, "No Devices Found", text)

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

    def _start_demo_mode(self):
        """Start demo mode with simulated signals."""
        from .device_dialog import MockDevice

        if self._device is not None and not self._demo_mode:
            self._release_device()
        if self._device is None:
            self._device = MockDevice()
            if self._preferred_rate:
                self._device.set_sample_rate(self._preferred_rate)
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
            + "Click a peak to tune to it; Device > Connect... opens a real SDR.",
            "info",
            8000,
        )
        logger.info("Demo mode started")

    def set_sample_rate(self, rate_hz: float) -> None:
        """Set the sample rate of the current device.

        The rate is also used by the demo device from now on and preselected
        in Device > Connect (it is how ``--sample-rate`` is applied).
        """
        try:
            rate = float(rate_hz)
        except (TypeError, ValueError):
            return
        if not rate > 0:
            return
        self._preferred_rate = rate
        if self._device is not None:
            try:
                self._device.set_sample_rate(rate)
            except Exception as e:
                self._show_status_error(f"Could not set the sample rate: {e}")
        self._reset_demodulator()
        self._refresh_state_ui()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _open_recording(self):
        """Load an I/Q file into the recording buffer (to save it in another
        format; there is no playback)."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Recording",
            "",
            ";;".join(
                (
                    "I/Q Recordings (*.cf32 *.cs16 *.cs8 *.cu8 *.cf64 *.raw *.iq "
                    "*.bin *.sigmf-data *.sigmf-meta *.wav)",
                    "SigMF (*.sigmf-data *.sigmf-meta)",
                    "WAV Files (*.wav)",
                    "All Files (*)",
                )
            ),
        )
        if not filename:
            return

        from ..dsp.recording import load_iq_file

        try:
            samples, metadata = load_iq_file(filename)
        except Exception as e:
            logger.error(f"Failed to open recording: {e}")
            QMessageBox.warning(
                self, "Import Failed", f"Could not import the recording:\n{e}"
            )
            return

        if self._recording:
            # Don't mix live samples into the loaded file.
            self._toggle_recording(False)
        if not self._confirm_discard_recording("importing another file"):
            return
        self._samples_buffer = [samples]
        rate = float(metadata.sample_rate or 0.0)
        center = float(metadata.center_frequency or 0.0)
        # Saved with the file's own rate and frequency (not the receiver's).
        self._buffer_meta = (center if center > 0 else None, rate if rate > 0 else None)
        self._buffer_unsaved = False  # it is on disk already
        if center > 0:
            self.set_frequency(center)
        logger.info(
            f"Loaded {len(samples)} samples from {filename} "
            f"(rate={metadata.sample_rate}, freq={metadata.center_frequency})"
        )
        unknown = "not stored in the file"
        rate_text = format_rate(rate) if rate > 0 else unknown
        freq_text = format_frequency(center) if center > 0 else unknown
        self._show_status_message(
            f"Imported {len(samples):,} samples into the recording buffer", "success"
        )
        QMessageBox.information(
            self,
            "Recording Imported",
            f"Imported {len(samples):,} samples into the recording buffer.\n\n"
            f"Sample rate: {rate_text}\n"
            f"Center frequency: {freq_text}\n\n"
            "Use File > Save Recording to write it in another format.",
        )

    # (file dialog filter, extension, FileFormat name, SampleFormat name)
    _SAVE_FORMATS: Tuple[Tuple[str, str, str, str], ...] = (
        ("Complex Float32 (*.cf32)", ".cf32", "RAW", "FLOAT32"),
        ("Complex Int16 (*.cs16)", ".cs16", "RAW", "INT16"),
        ("SigMF (*.sigmf-data)", ".sigmf-data", "SIGMF", "FLOAT32"),
        ("WAV, 16-bit I/Q (*.wav)", ".wav", "WAV", "INT16"),
        ("Raw I/Q, Float32 (*.raw)", ".raw", "RAW", "FLOAT32"),
    )

    def _panel_recording_extension(self) -> str:
        """Extension matching the Recording panel's Format choice."""
        try:
            fmt = self._control_panel.get_recording_format().lower()
        except Exception:  # pragma: no cover - defensive
            fmt = ""
        if fmt.startswith("wav"):
            return ".wav"
        if "sigmf" in fmt:
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

    def _save_recording(self) -> bool:
        """Save the recording buffer to an I/Q file. True when saved.

        The file gets the center frequency and sample rate the samples were
        captured at (or read from), not what the receiver is set to now.
        """
        if not self._samples_buffer:
            QMessageBox.information(
                self,
                "Nothing to Save",
                "No samples have been recorded yet.\n\n"
                "Start receiving, press Record (Ctrl+Shift+R), stop recording, "
                "then save.",
            )
            return False

        center_freq, sample_rate = self._buffer_meta
        if not center_freq:
            center_freq = self._current_frequency()
        if not sample_rate:
            sample_rate = self._device_sample_rate()

        # Preselect the format chosen in the Recording panel.
        ext = self._panel_recording_extension()
        initial = next(f[0] for f in self._SAVE_FORMATS if f[1] == ext)
        suggested = (
            f"iq_{center_freq / 1e6:.3f}MHz_" f"{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        )
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            suggested,
            ";;".join(f[0] for f in self._SAVE_FORMATS),
            initial,
        )
        if not filename:
            return False

        from ..dsp.recording import FileFormat, SampleFormat, save_iq_file

        filename, spec = self._resolve_save_format(filename, selected_filter)
        fmt, sample_fmt = FileFormat[spec[2]], SampleFormat[spec[3]]

        samples = np.concatenate(self._samples_buffer).astype(np.complex64)

        try:
            save_iq_file(
                filename,
                samples,
                sample_rate=sample_rate,
                center_frequency=center_freq,
                sample_format=sample_fmt,
                file_format=fmt,
            )
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            QMessageBox.warning(self, "Save Failed", f"Could not save recording:\n{e}")
            return False

        self._buffer_unsaved = False
        logger.info(f"Saved {len(samples)} samples to {filename}")
        self._show_status_message(
            f"Saved {len(samples):,} samples to {filename}", "success", 6000
        )
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
        """Save a PNG of the window (spectrum + waterfall + panels)."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "sdr.png", "PNG (*.png)"
        )
        if not filename:
            return
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

    def _show_scanner(self):
        """Show the frequency scanner dialog.

        The sweep reads the device from a worker thread, so the display loop
        pauses while the dialog is open. Tuning to a result tunes the
        receiver.
        """
        from .scanner_dialog import ScannerDialog

        dialog = ScannerDialog(self, device=self._device)
        dialog.frequency_selected.connect(self.set_frequency)
        display_was_active = self._display_timer.isActive()
        self._display_timer.stop()
        try:
            self._exec_dialog(dialog)
        finally:
            if display_was_active:
                self._display_timer.start()

    def _show_radio_tuner(self):
        """Show the AM/FM radio tuner pop-out window."""
        if self._radio_tuner is None:
            self._radio_tuner = RadioTunerWidget(self, self._device_sample_rate())
            # Connect frequency change to main tuner
            self._radio_tuner.frequency_changed.connect(
                self._on_radio_frequency_changed
            )

        # Open on the station being received, when it is in a broadcast band.
        freq = self._current_frequency()
        if band_for_frequency(freq) is not None:
            self._radio_tuner.set_frequency(freq)  # does not emit

        self._radio_tuner.show()
        self._radio_tuner.raise_()
        self._radio_tuner.activateWindow()

    def _on_radio_frequency_changed(self, freq_hz: float, band: str):
        """Tune the receiver to the radio tuner's station, in its mode."""
        # set_frequency tunes the device (if any) and updates the control
        # panel, toolbar readout and plot axes together.
        self.set_frequency(freq_hz)
        if band == "FM":
            self._set_demod_mode("FM", _BROADCAST_FM_DEVIATION, _BROADCAST_FM_BANDWIDTH)
        elif band == "AM":
            self._set_demod_mode("AM", bandwidth=_BROADCAST_AM_BANDWIDTH)
        self._show_status_message(
            f"Tuned to {format_frequency(freq_hz)} ({band})", "info", 2500
        )
        logger.info(f"Tuned to {freq_hz/1e6:.3f} MHz ({band})")

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
            "and CW demodulation; POCSAG, FLEX, ADS-B, ACARS, AX.25/APRS and "
            "RDS decoders; I/Q recording; ham radio tools.</p>"
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

    def _nudge_frequency(self, offset_hz: float) -> None:
        """Adjust the center frequency by the given offset."""
        new_freq = max(0.0, self._current_frequency() + offset_hz)
        self.set_frequency(new_freq)

    def _apply_band_preset(self, freq_hz: float, mode: str, label: str) -> None:
        """Tune to a band preset in its mode, FM deviation and bandwidth."""
        self.set_frequency(freq_hz)
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
        """Save the current tuner frequency into bookmarks."""
        freq = self._current_frequency()
        label = format_frequency(freq)
        self._bookmarks_panel.add_bookmark(label, freq)
        self._show_status_message(f"Bookmarked {label}", "success", 2500)

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
        elif on:
            tip = (
                "Audio is on: the tuned signal plays through your speakers "
                "while it is above squelch. Click to mute (Radio > Audio Output)."
            )
        else:
            tip = "Audio is off. Click to play the tuned signal through your speakers."
            if self._audio_problem is not None:
                tip += f" ({self._audio_problem}.)"
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
            freq = self._settings.get_float("frequency_hz", 100e6)
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
            freq, gain, squelch, agc, demod = 100e6, 20.0, -80.0, False, "FM"
            deviation = bandwidth = rec_format = panel = ""
        if not deviation and _FM_BROADCAST_BAND[0] <= freq <= _FM_BROADCAST_BAND[1]:
            deviation = _BROADCAST_FM_DEVIATION  # broadcast FM is wideband
        # set_frequency also updates the toolbar readout and plot axes.
        self.set_frequency(freq)
        self._control_panel.set_gain(gain)
        self._control_panel.set_squelch_db(squelch)
        self._control_panel.set_agc_enabled(agc)
        self._restoring = True
        try:
            # Bandwidth and deviation first: they belong to the mode.
            self._set_demod_mode(demod, deviation or None, bandwidth or None)
            combo = self._control_panel._format_combo
            if rec_format and combo.findText(rec_format) >= 0:
                combo.setCurrentText(rec_format)
            if panel in self._panel_pages:
                self._show_panel(panel)
            self._restore_license_class()
            self._restore_ham_id_settings()
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
            gain = float(self._control_panel._gain_slider.value())
            self._settings.set("frequency_hz", freq)
            self._settings.set("gain_db", gain)
            self._settings.set("squelch_db", self._squelch_db)
            self._settings.set("agc_enabled", self._agc_enabled)
            self._settings.set("theme", self._theme)
            panel = self._control_panel
            self._settings.set("demod_mode", panel._demod_combo.currentText())
            self._settings.set("fm_deviation", panel._fm_dev_combo.currentText())
            self._settings.set("bandwidth", panel._bw_combo.currentText())
            self._settings.set("recording_format", panel._format_combo.currentText())
            self._settings.set("audio_volume", self._volume_slider.value())
            current = self._current_panel_key()
            if current:
                self._settings.set("panel", current)
            self._save_ham_id_settings()
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
        # to replace it with the demo device.
        hardware = (self._device is not None and not self._demo_mode) or bool(
            self._scan_hardware()
        )
        wiz = FirstRunWizard(self, hardware_found=hardware)
        wiz.demo_mode_requested.connect(self._start_demo_mode)
        if self._exec_dialog(wiz):
            freq = wiz.selected_frequency()
            self.set_frequency(freq)
            preset = self._preset_for(freq)
            if preset is not None:
                self._set_demod_mode(preset.mode, preset.fm_deviation, preset.bandwidth)
        self._settings.mark_first_run_done()

    def _show_welcome(self) -> None:
        """Help > Welcome and Quick Start: the first-run wizard again."""
        self._run_first_run_wizard()

    def closeEvent(self, event):
        """Handle window close (offering to save an unsaved recording)."""
        if self._recording:
            self._toggle_recording(False)
        if not self._confirm_discard_recording("closing SDR Module"):
            event.ignore()
            return
        self._persist_state()
        for timer in (self._display_timer, self._status_timer, self._hotplug_timer):
            timer.stop()
        self._audio.stop()
        # Stops receiving and closes the device; a failing close() is logged
        # instead of leaving the window impossible to close.
        self._release_device()

        logger.info("Application closing")
        event.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frequency(self, freq_hz: float):
        """Set the center frequency (clamped to the tuning range)."""
        self._control_panel.set_frequency(freq_hz)
        # Use the value the control panel accepted, so the device, toolbar
        # readout and plot axes never disagree with the Frequency field.
        self._on_frequency_changed(self._current_frequency())

    def set_gain(self, gain_db: float):
        """Set the RF gain slider.

        The panel applies it to the device only while AGC is off (with AGC
        on, the value takes effect when AGC is switched off).
        """
        self._control_panel.set_gain(gain_db)
