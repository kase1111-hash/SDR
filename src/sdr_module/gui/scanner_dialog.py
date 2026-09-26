"""
Frequency scanner dialog.

Runs a non-blocking sweep using the connected SDR and lists detected
signals. A real device is required: with no hardware the scan is disabled
rather than fabricating detections from synthetic noise.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QProgressBar,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import numpy as np

from ..utils.tooltips import get_short_tip

if HAS_PYQT6:
    from .themes import set_role, set_tone

logger = logging.getLogger(__name__)

# Scan presets: (label, start MHz, end MHz, step kHz).
SCAN_PRESETS: Tuple[Tuple[str, float, float, float], ...] = (
    ("FM Broadcast (88–108 MHz)", 88.0, 108.0, 200.0),
    ("Airband AM (118–137 MHz)", 118.0, 137.0, 25.0),
    ("2m Ham (144–148 MHz)", 144.0, 148.0, 12.5),
    ("NOAA Weather (162 MHz)", 162.4, 162.55, 25.0),
    ("ISM 433 MHz", 433.05, 434.79, 25.0),
    ("70cm Ham (420–450 MHz)", 420.0, 450.0, 25.0),
)
_CUSTOM = "Custom"
_NO_DEVICE_TIP = (
    "Connect a device first: close the scanner, choose Device > Connect... or "
    "Device > Use Demo Device, then open the scanner again."
)

# FFT length for level measurements. It matches the main window's spectrum
# (DISPLAY_BLOCK), so a threshold read off the spectrum means the same here.
_FFT_SIZE = 2048
# Up to this many FFT_SIZE segments are power-averaged per step (Welch), which
# steadies the noise floor without hiding steady carriers.
_MAX_SEGMENTS = 4
# Give up when this many steps in a row return no samples before any step
# has returned some: the device is not delivering data, and each empty read
# can wait a second on real hardware.
_EMPTY_STEPS_LIMIT = 2

# Worker threads still winding down after their dialog closed. Holding a
# reference keeps the QThread alive until run() returns, instead of it being
# destroyed while running (which aborts the process).
_ORPHANED_WORKERS: set = set()


def _release_orphan(worker) -> None:
    """Drop an orphaned worker once its thread has really exited.

    ``finished`` is delivered (queued) to the GUI thread while the worker
    thread may still be winding down; deleting the QThread before it exits
    aborts the process, so wait for it first (microseconds at this point).
    """
    worker.wait()
    _ORPHANED_WORKERS.discard(worker)


def _format_freq(freq_hz: float) -> str:
    """Always MHz, like the range fields (``1090.000 MHz``, not GHz)."""
    return f"{freq_hz / 1e6:.3f} MHz"


def _power_spectrum_dbfs(samples: np.ndarray) -> np.ndarray:
    """Hann-windowed power spectrum in dBFS (a full-scale tone reads 0 dB).

    Uses the main display's FFT length and averages up to ``_MAX_SEGMENTS``
    of the newest segments, so levels match the spectrum whatever block size
    the device delivers. Shorter captures use a single FFT of their length.
    """
    n = len(samples)
    if n >= _FFT_SIZE:
        segments = min(_MAX_SEGMENTS, n // _FFT_SIZE)
        blocks = samples[n - segments * _FFT_SIZE :].reshape(segments, _FFT_SIZE)
        size = _FFT_SIZE
    else:
        blocks = samples.reshape(1, n)
        size = n
    window = np.hanning(size + 1)[:-1]
    gain = float(np.sum(window)) or 1.0
    spec = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = np.mean(np.abs(spec) ** 2, axis=0) / (gain * gain)
    return 10.0 * np.log10(power + 1e-24)


def _is_streaming(device) -> Optional[bool]:
    """Whether ``device`` is delivering samples; ``None`` when unknown."""
    state = getattr(device, "state", None)
    for value in (
        getattr(state, "is_streaming", None),
        getattr(device, "is_streaming", None),
    ):
        if isinstance(value, bool):
            return value
    return None


def _tuning_range(device) -> Optional[Tuple[float, float]]:
    """The device's ``(min, max)`` tuning range in Hz, if it publishes one."""
    spec = getattr(device, "spec", None)
    low = getattr(spec, "freq_min", None)
    high = getattr(spec, "freq_max", None)
    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high > low:
        return float(low), float(high)
    return None


class _ScanWorker(QThread if HAS_PYQT6 else object):
    """Background sweep over a frequency range."""

    if HAS_PYQT6:
        hit = pyqtSignal(float, float)  # freq_hz, peak_db
        detected = pyqtSignal(float, float, float)  # freq_hz, peak_db, noise_db
        progress = pyqtSignal(int)  # percent
        stepped = pyqtSignal(float)  # freq_hz now being measured
        finished_scan = pyqtSignal()

    def __init__(self, device, start_hz, end_hz, step_hz, threshold_db, parent=None):
        super().__init__(parent)
        self._device = device
        self._start = start_hz
        self._end = end_hz
        self._step = step_hz
        self._threshold = threshold_db
        self._cancelled = False
        self.cancelled = False
        self.steps_measured = 0  # steps where the device returned samples
        self.steps_out_of_range = 0  # steps the device refused to tune to
        self.no_samples = False  # gave up: the device delivered no samples
        self.started_stream = False  # the sweep started the device itself
        self.noise_floor_db: Optional[float] = None  # median across the sweep
        self._last_failure = ""  # why the last _measure() returned None

    def cancel(self):
        self._cancelled = True

    def frequencies(self) -> np.ndarray:
        """The step frequencies, from start up to and including end."""
        return np.arange(self._start, self._end + self._step * 0.5, self._step)

    def run(self):
        freqs = self.frequencies()
        total = max(1, len(freqs))
        original = self._device_frequency()
        noise: List[float] = []
        empty_steps = 0
        try:
            self._ensure_streaming()
            for i, f in enumerate(freqs):
                if self._cancelled:
                    self.cancelled = True
                    break
                self.stepped.emit(float(f))
                measured = self._measure(f, self._step)
                if measured is not None:
                    peak_freq, peak_db, noise_db = measured
                    self.steps_measured += 1
                    empty_steps = 0
                    noise.append(noise_db)
                    if peak_db > self._threshold:
                        self.hit.emit(float(peak_freq), float(peak_db))
                        self.detected.emit(
                            float(peak_freq), float(peak_db), float(noise_db)
                        )
                elif self._last_failure == "samples" and not self.steps_measured:
                    empty_steps += 1
                    if empty_steps >= _EMPTY_STEPS_LIMIT:
                        self.no_samples = True
                        break
                self.progress.emit(int((i + 1) * 100 / total))
        finally:
            if noise:
                self.noise_floor_db = float(np.median(noise))
            # Put the receiver back where the user had it tuned.
            if original is not None:
                try:
                    self._device.set_frequency(original)
                except Exception as e:
                    logger.debug(f"Could not restore frequency after scan: {e}")
            if self.started_stream:
                try:
                    self._device.stop_rx()
                except Exception as e:
                    logger.debug(f"Could not stop the stream after scan: {e}")
            self.finished_scan.emit()

    def _ensure_streaming(self) -> None:
        """Start the device for the sweep when the receiver is stopped.

        The scanner is modal, so the user cannot press Start while it is
        open. A device the sweep starts is stopped again when it ends.
        """
        dev = self._device
        if dev is None or _is_streaming(dev) is not False:
            return
        try:
            self.started_stream = bool(dev.start_rx())
        except Exception as e:
            logger.debug(f"Could not start the device for the scan: {e}")

    def _device_frequency(self) -> Optional[float]:
        """The device's current center frequency, if it exposes one."""
        dev = self._device
        if dev is None:
            return None
        for value in (
            getattr(dev, "frequency", None),
            getattr(getattr(dev, "state", None), "frequency", None),
        ):
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return None

    def _sample_rate(self) -> float:
        dev = self._device
        for value in (
            getattr(dev, "sample_rate", None),
            getattr(getattr(dev, "state", None), "sample_rate", None),
        ):
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return 2.4e6

    def _measure_peak(self, freq_hz: float) -> Optional[float]:
        """Peak level (dBFS) anywhere in one capture at ``freq_hz``."""
        measured = self._measure(freq_hz, None)
        return None if measured is None else measured[1]

    def _measure(
        self, freq_hz: float, span_hz: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        """Tune, capture and measure: ``(peak_freq_hz, peak_dbfs, noise_dbfs)``.

        Only bins within ``span_hz / 2`` of ``freq_hz`` count toward the peak
        (``None`` = the whole capture), so each step reports the signals of
        its own slice instead of every strong carrier in the passband.
        """
        # A real sweep requires real hardware. Never synthesise samples here:
        # noise fed through an FFT clears a low threshold at every step and the
        # dialog would fill with fabricated "detections".
        self._last_failure = "device"
        if not (self._device and hasattr(self._device, "set_frequency")):
            return None
        try:
            if self._device.set_frequency(freq_hz) is False:
                # Outside the tuning range: the device is still on the last
                # frequency, so measuring now would report the wrong one.
                self._last_failure = "range"
                self.steps_out_of_range += 1
                return None
            # Discard one block: it may predate the retune (tuner settling
            # and buffered samples), which would report the wrong frequency.
            first = self._device.read_samples(_FFT_SIZE)
            if first is None or len(first) == 0:
                self._last_failure = "samples"
                return None
            samples = self._device.read_samples(_FFT_SIZE * _MAX_SEGMENTS)
        except Exception as e:
            logger.debug(f"Scan read failed at {freq_hz}: {e}")
            self._last_failure = "error"
            return None

        if samples is None or len(samples) == 0:
            self._last_failure = "samples"
            return None
        self._last_failure = ""

        power = _power_spectrum_dbfs(np.asarray(samples))
        n = len(power)
        offsets = (np.arange(n) - n // 2) * (self._sample_rate() / n)

        if span_hz:
            mask = np.abs(offsets) <= max(span_hz / 2.0, abs(offsets[1] - offsets[0]))
            if not np.any(mask):
                mask = np.ones(n, dtype=bool)
        else:
            mask = np.ones(n, dtype=bool)
        region = power[mask]
        region_offsets = offsets[mask]
        k = int(np.argmax(region))
        peak_db = float(region[k])

        # Report the signal's center, not just its hottest bin: a power-
        # weighted centroid of the contiguous bins within 12 dB of the peak
        # lands on the carrier of wide FM stations too.
        above = region >= peak_db - 12.0
        lo = k
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(region) - 1 and above[hi + 1]:
            hi += 1
        weights = 10.0 ** (region[lo : hi + 1] / 10.0)
        center = float(np.sum(region_offsets[lo : hi + 1] * weights) / np.sum(weights))
        return float(freq_hz + center), peak_db, float(np.median(power))


class _NumericItem(QTableWidgetItem if HAS_PYQT6 else object):
    """Table item that sorts by a numeric value instead of its text."""

    def __init__(self, text: str, value: float, align_right: bool = True):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, float(value))
        horizontal = (
            Qt.AlignmentFlag.AlignRight if align_right else Qt.AlignmentFlag.AlignLeft
        )
        self.setTextAlignment(horizontal | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        try:
            return float(self.data(Qt.ItemDataRole.UserRole)) < float(
                other.data(Qt.ItemDataRole.UserRole)
            )
        except (TypeError, ValueError):
            return super().__lt__(other)


class _ViewportResizeWatcher(QObject if HAS_PYQT6 else object):
    """Keeps an overlay label covering a table's viewport."""

    def __init__(self, viewport, overlay):
        super().__init__(viewport)
        self._overlay = overlay
        viewport.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self._overlay.setGeometry(obj.rect())
        return False


class ScannerDialog(QDialog if HAS_PYQT6 else object):
    """Non-blocking frequency sweep dialog."""

    if HAS_PYQT6:
        # Emitted when the user asks to tune to a result (Hz).
        frequency_selected = pyqtSignal(float)

    def __init__(self, parent=None, device=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._device = device
        self._worker: Optional[_ScanWorker] = None
        self._hits: List[tuple] = []
        self._scanning = False  # between Start Scan and the sweep's end
        self._stopping = False
        self._range_note = ""  # set when the sweep is clamped to the device

        self.setWindowTitle("Frequency Scanner")
        self.setMinimumSize(560, 520)
        self.resize(620, 600)
        self._setup_ui()
        self._update_steps_hint()
        self._update_controls()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Range controls
        range_group = QGroupBox("Scan Range")
        range_layout = QVBoxLayout(range_group)
        range_layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnMinimumWidth(2, 70)

        self._preset = QComboBox()
        self._preset.addItem(_CUSTOM)
        for label, *_rest in SCAN_PRESETS:
            self._preset.addItem(label)
        self._preset.setToolTip("Fill in the range and step for a common band")
        self._preset.currentIndexChanged.connect(self._on_preset_changed)
        self._add_field(grid, 0, 0, "&Band:", self._preset, span=3)

        self._start = self._spin(0.1, 6000.0, 3, 88.0, " MHz")
        self._start.setToolTip("First frequency to measure")
        self._add_field(grid, 1, 0, "&Start:", self._start)

        self._end = self._spin(0.1, 6000.0, 3, 108.0, " MHz")
        self._end.setToolTip("Last frequency to measure")
        self._add_field(grid, 2, 0, "&End:", self._end)

        self._step = self._spin(1.0, 10000.0, 1, 200.0, " kHz")
        self._step.setToolTip(
            "Distance between measurements. Each step reports the strongest "
            "signal within half a step either side.\nUse about the channel "
            "spacing: 200 kHz for FM broadcast, 12.5-25 kHz for voice."
        )
        self._add_field(grid, 1, 2, "St&ep:", self._step)

        self._threshold = self._spin(-140.0, 0.0, 1, -60.0, " dBFS")
        self._threshold.setToolTip(
            "Only signals whose peak is above this level are listed.\n"
            "Set it 10-20 dB above the noise floor shown on the spectrum."
        )
        self._add_field(grid, 2, 2, "T&hreshold:", self._threshold)
        range_layout.addLayout(grid)

        self._steps_hint = QLabel()
        set_role(self._steps_hint, "hint")
        range_layout.addWidget(self._steps_hint)

        for spin in (self._start, self._end, self._step):
            spin.valueChanged.connect(self._on_range_edited)
        self._threshold.valueChanged.connect(self._update_steps_hint)
        # FM broadcast, which matches the default range above.
        self._preset.blockSignals(True)
        self._preset.setCurrentIndex(1)
        self._preset.blockSignals(False)

        layout.addWidget(range_group)

        # Start/Stop, status and progress
        run_row = QHBoxLayout()
        run_row.setSpacing(10)
        self._start_btn = QPushButton()
        # Same width in both states, so the status text beside it stays put.
        widths = []
        for text, role in (("Start Scan", "primary"), ("Stop Scan", "danger")):
            self._start_btn.setText(text)
            set_role(self._start_btn, role)
            widths.append(self._start_btn.sizeHint().width())
        self._start_btn.setMinimumWidth(max(widths) + 8)
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self._toggle_scan)
        run_row.addWidget(self._start_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        run_row.addWidget(self._status, 1)
        layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(8)
        layout.addWidget(self._progress)

        # Results
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Frequency", "Peak Level", "SNR"])
        header = self._table.horizontalHeader()
        # Three numeric columns of equal width, right-aligned so the decimal
        # points line up.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        header_tips = (
            "Signal center frequency",
            "Strongest FFT bin, in dB relative to full scale (same scale as "
            "the spectrum)",
            get_short_tip("snr"),
        )
        for col, tip in enumerate(header_tips):
            item = self._table.horizontalHeaderItem(col)
            item.setToolTip(tip)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setToolTip("Double-click a signal to tune to it")
        self._table.itemSelectionChanged.connect(self._update_controls)
        self._table.itemDoubleClicked.connect(lambda _item: self._tune_selected())
        layout.addWidget(self._table, 1)

        # Empty-state overlay on the table
        self._empty = QLabel(self._table.viewport())
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        set_role(self._empty, "placeholder")
        self._empty_watcher = _ViewportResizeWatcher(
            self._table.viewport(), self._empty
        )

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._tune_btn = QPushButton("&Tune to Signal")
        self._tune_btn.setAutoDefault(False)
        self._tune_btn.clicked.connect(self._tune_selected)
        buttons.addButton(self._tune_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._clear_btn = QPushButton("C&lear Results")
        self._clear_btn.setAutoDefault(False)
        self._clear_btn.clicked.connect(self._clear_results)
        buttons.addButton(self._clear_btn, QDialogButtonBox.ButtonRole.ResetRole)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setAutoDefault(False)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        # Status / device-required notice
        if self._device is None:
            self._start_btn.setEnabled(False)
            self._start_btn.setToolTip(_NO_DEVICE_TIP)
            self._set_status("No device connected.", "warning")

    @staticmethod
    def _add_field(grid, row, col, text, field, span=1) -> None:
        label = QLabel(text)
        label.setBuddy(field)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label, row, col)
        grid.addWidget(field, row, col + 1, 1, span)

    @staticmethod
    def _spin(low, high, decimals, value, suffix) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.setKeyboardTracking(False)
        return spin

    def _set_status(self, text: str, tone: Optional[str] = None) -> None:
        self._status.setText(text)
        set_tone(self._status, tone)

    # ------------------------------------------------------------------ #
    # Range editing
    # ------------------------------------------------------------------ #
    def _on_preset_changed(self, index: int) -> None:
        if index <= 0:
            return
        _label, start, end, step = SCAN_PRESETS[index - 1]
        for spin, value in ((self._start, start), (self._end, end), (self._step, step)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_steps_hint()
        self._update_controls()

    def _on_range_edited(self) -> None:
        # Hand-edited values no longer match the chosen band.
        if self._preset.currentIndex() > 0:
            self._preset.blockSignals(True)
            self._preset.setCurrentIndex(0)
            self._preset.blockSignals(False)
        self._update_steps_hint()
        self._update_controls()

    def _range_valid(self) -> bool:
        return self._end.value() > self._start.value()

    def _planned_sweep(self) -> Tuple[Optional[Tuple[float, float]], str]:
        """The range that will actually be swept, in Hz, and a note.

        The range is kept inside what the device can tune (steps outside it
        would fail, and each failure is logged as an error). Returns
        ``(None, reason)`` when none of the range is tunable.
        """
        start_hz = self._start.value() * 1e6
        end_hz = self._end.value() * 1e6
        limits = _tuning_range(self._device)
        if limits is None:
            return (start_hz, end_hz), ""
        low, high = limits
        if end_hz < low or start_hz > high:
            return None, (
                f"This device tunes {_format_freq(low)} to {_format_freq(high)}. "
                "Choose a range inside it."
            )
        if start_hz < low or end_hz > high:
            start_hz, end_hz = max(start_hz, low), min(end_hz, high)
            return (start_hz, end_hz), (
                f"Limited to the device's range, {_format_freq(start_hz)} to "
                f"{_format_freq(end_hz)}."
            )
        return (start_hz, end_hz), ""

    def _update_steps_hint(self) -> None:
        if not self._range_valid():
            self._steps_hint.setText("End must be above Start.")
            set_tone(self._steps_hint, "danger")
            return
        sweep, note = self._planned_sweep()
        if sweep is None:
            self._steps_hint.setText(note)
            set_tone(self._steps_hint, "danger")
            return
        span_khz = (sweep[1] - sweep[0]) / 1e3
        steps = int(span_khz // self._step.value()) + 1
        noun = "step" if steps == 1 else "steps"
        text = (
            f"{steps:,} {noun}. Signals above {self._threshold.value():.0f} dBFS "
            "are listed below."
        )
        self._steps_hint.setText(f"{text} {note}" if note else text)
        set_tone(self._steps_hint, "warning" if note else None)

    def _update_controls(self) -> None:
        scanning = self._scanning
        for w in (self._preset, self._start, self._end, self._step, self._threshold):
            w.setEnabled(not scanning)

        if scanning:
            self._start_btn.setText("Stop Scan")
            set_role(self._start_btn, "danger")
            self._start_btn.setEnabled(True)
        else:
            self._start_btn.setText("Start Scan")
            set_role(self._start_btn, "primary")
            reason = ""
            if not self._range_valid():
                reason = "End must be above Start"
            elif self._device is not None and self._planned_sweep()[0] is None:
                reason = self._planned_sweep()[1]
            can_scan = self._device is not None and not reason
            self._start_btn.setEnabled(can_scan)
            if self._device is not None:
                self._start_btn.setToolTip(reason)

        has_rows = self._table.rowCount() > 0
        has_selection = bool(self._table.selectionModel().selectedRows())
        self._tune_btn.setEnabled(has_selection and not scanning)
        self._tune_btn.setToolTip(
            "Tune the receiver to the selected signal"
            if has_selection
            else "Select a signal in the list first"
        )
        self._clear_btn.setEnabled(has_rows and not scanning)
        self._update_empty_state(scanning)

    def _update_empty_state(self, scanning: bool) -> None:
        if self._table.rowCount():
            self._empty.hide()
            return
        if self._device is None:
            text = (
                "No device connected.\n"
                "Close the scanner, choose Device > Connect... or\n"
                "Device > Use Demo Device, then open it again."
            )
        elif scanning:
            text = "Scanning... signals above the threshold will appear here."
        elif self._worker is not None:
            text = (
                "No signals above the threshold.\n"
                "Lower the threshold or pick another range and scan again."
            )
        else:
            text = "No results yet.\nChoose a range and click Start Scan."
        self._empty.setText(text)
        self._empty.setGeometry(self._table.viewport().rect())
        self._empty.show()

    # ------------------------------------------------------------------ #
    # Scanning
    # ------------------------------------------------------------------ #
    def _toggle_scan(self):
        if self._scanning:
            if self._worker is not None:
                self._worker.cancel()
            self._stopping = True
            self._start_btn.setEnabled(False)
            self._set_status("Stopping...", "muted")
            return
        if self._worker is not None and self._worker.isRunning():
            return  # the previous sweep is still winding down

        if self._device is None:
            self._set_status("No device connected.", "warning")
            return

        step_hz = self._step.value() * 1e3
        if not self._range_valid():
            self._set_status("End must be above Start.", "danger")
            return
        sweep, note = self._planned_sweep()
        if sweep is None:
            self._set_status(note, "danger")
            return
        start_hz, end_hz = sweep
        self._range_note = f" {note}" if note else ""

        self._clear_results()
        self._progress.setValue(0)

        self._worker = _ScanWorker(
            self._device, start_hz, end_hz, step_hz, self._threshold.value(), self
        )
        self._worker.detected.connect(self._on_hit)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.stepped.connect(self._on_stepped)
        self._worker.finished_scan.connect(self._on_finished)
        self._scanning = True
        self._stopping = False
        self._worker.start()
        self._set_status(f"Scanning from {_format_freq(start_hz)}...", None)
        self._update_controls()

    def _on_stepped(self, freq_hz: float) -> None:
        if self._stopping:
            return
        count = len(self._hits)
        noun = "signal" if count == 1 else "signals"
        self._set_status(
            f"Scanning {_format_freq(freq_hz)}  ·  {count} {noun} found", None
        )

    def _on_hit(self, freq_hz: float, peak_db: float, noise_db: Optional[float] = None):
        snr = None if noise_db is None else peak_db - noise_db

        # A signal wider than the step is reported by neighbouring steps too:
        # keep one row per signal (the strongest reading).
        merge_hz = 0.75 * (self._step.value() * 1e3)
        if self._hits and abs(freq_hz - self._hits[-1][0]) < merge_hz:
            if peak_db <= self._hits[-1][1]:
                return
            self._hits[-1] = (freq_hz, peak_db, snr)
            row = self._table.rowCount() - 1
        else:
            self._hits.append((freq_hz, peak_db, snr))
            row = self._table.rowCount()
            self._table.insertRow(row)

        self._table.setItem(row, 0, _NumericItem(_format_freq(freq_hz), freq_hz))
        self._table.setItem(row, 1, _NumericItem(f"{peak_db:.1f} dBFS", peak_db))
        snr_item = _NumericItem("—" if snr is None else f"{snr:.1f} dB", snr or 0.0)
        self._table.setItem(row, 2, snr_item)
        self._update_empty_state(True)

    def _on_finished(self):
        self._scanning = False
        self._stopping = False
        worker = self._worker
        count = len(self._hits)
        noun = "signal" if count == 1 else "signals"

        noise = worker.noise_floor_db if worker is not None else None
        skipped = worker.steps_out_of_range if worker is not None else 0
        skipped_note = (
            f" {skipped:,} steps outside the device's range were skipped."
            if skipped
            else ""
        )
        if worker is not None and worker.steps_measured == 0 and not worker.cancelled:
            if skipped and not worker.no_samples:
                text = "The device could not tune to any step in this range."
            else:
                text = (
                    "The device returned no samples. Check that it is plugged "
                    "in and not used by another program, then scan again."
                )
            self._set_status(text, "warning")
            self._progress.setValue(0)
        elif worker is not None and worker.cancelled:
            self._set_status(f"Scan stopped. {count} {noun} found.", "muted")
        else:
            floor = f" Noise floor {noise:.0f} dBFS." if noise is not None else ""
            self._set_status(
                f"Scan complete: {count} {noun} found.{floor}"
                f"{self._range_note}{skipped_note}",
                "success" if count else None,
            )
            self._progress.setValue(100)

        self._table.setSortingEnabled(True)
        self._update_controls()

    def _clear_results(self) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._hits.clear()
        self._update_controls()

    # ------------------------------------------------------------------ #
    # Tuning to a result
    # ------------------------------------------------------------------ #
    def selected_frequency(self) -> Optional[float]:
        """Frequency (Hz) of the selected result row, if any."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return float(value) if value is not None else None

    def _tune_selected(self) -> None:
        if self._scanning:
            return
        freq = self.selected_frequency()
        if freq is None:
            return
        self.frequency_selected.emit(freq)
        # Without a listener, tune the parent window directly.
        if self.receivers(self.frequency_selected) == 0:
            parent = self.parent()
            tune = getattr(parent, "set_frequency", None)
            if callable(tune):
                tune(freq)
            elif self._device is not None:
                self._device.set_frequency(freq)
        self._set_status(f"Tuned to {_format_freq(freq)}.", "success")

    # ------------------------------------------------------------------ #
    # Closing
    # ------------------------------------------------------------------ #
    def reject(self):
        self._stop_worker()
        super().reject()

    def _stop_worker(self) -> None:
        worker = self._worker
        if not (worker and worker.isRunning()):
            return
        worker.cancel()
        if worker.wait(2000):
            return
        # Still inside a slow device read: let it finish on its own rather
        # than destroying a running thread along with this dialog.
        worker.setParent(None)
        _ORPHANED_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _release_orphan(w))
