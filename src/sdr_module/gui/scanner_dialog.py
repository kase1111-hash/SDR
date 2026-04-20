"""
Frequency scanner dialog.

Runs a non-blocking sweep using the connected SDR (or synthetic data
when none is connected) and lists detected signals.
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QGridLayout,
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

logger = logging.getLogger(__name__)


class _ScanWorker(QThread if HAS_PYQT6 else object):
    """Background sweep over a frequency range."""

    if HAS_PYQT6:
        hit = pyqtSignal(float, float)  # freq_hz, peak_db
        progress = pyqtSignal(int)  # percent
        finished_scan = pyqtSignal()

    def __init__(self, device, start_hz, end_hz, step_hz, threshold_db, parent=None):
        super().__init__(parent)
        self._device = device
        self._start = start_hz
        self._end = end_hz
        self._step = step_hz
        self._threshold = threshold_db
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        freqs = np.arange(self._start, self._end, self._step)
        total = max(1, len(freqs))
        for i, f in enumerate(freqs):
            if self._cancelled:
                break
            peak_db = self._measure_peak(f)
            if peak_db is not None and peak_db > self._threshold:
                self.hit.emit(float(f), float(peak_db))
            self.progress.emit(int((i + 1) * 100 / total))
        self.finished_scan.emit()

    def _measure_peak(self, freq_hz: float) -> Optional[float]:
        try:
            if self._device and hasattr(self._device, "set_frequency"):
                self._device.set_frequency(freq_hz)
                samples = self._device.read_samples(4096)
            else:
                # Synthetic: random noise plus a fake signal near band centers
                samples = (
                    np.random.randn(4096) + 1j * np.random.randn(4096)
                ) * 0.05
        except Exception as e:
            logger.debug(f"Scan read failed at {freq_hz}: {e}")
            return None

        if samples is None or len(samples) == 0:
            return None

        spec = np.fft.fft(samples)
        power = 20 * np.log10(np.abs(spec) + 1e-10)
        return float(np.max(power))


class ScannerDialog(QDialog if HAS_PYQT6 else object):
    """Non-blocking frequency sweep dialog."""

    def __init__(self, parent=None, device=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)

        self._device = device
        self._worker: Optional[_ScanWorker] = None
        self._hits: List[tuple] = []

        self.setWindowTitle("Frequency Scanner")
        self.setMinimumSize(600, 420)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Range controls
        grid = QGridLayout()
        grid.addWidget(QLabel("Start (MHz):"), 0, 0)
        self._start = QDoubleSpinBox()
        self._start.setRange(0.1, 6000.0)
        self._start.setDecimals(3)
        self._start.setValue(88.0)
        grid.addWidget(self._start, 0, 1)

        grid.addWidget(QLabel("End (MHz):"), 0, 2)
        self._end = QDoubleSpinBox()
        self._end.setRange(0.1, 6000.0)
        self._end.setDecimals(3)
        self._end.setValue(108.0)
        grid.addWidget(self._end, 0, 3)

        grid.addWidget(QLabel("Step (kHz):"), 1, 0)
        self._step = QDoubleSpinBox()
        self._step.setRange(1.0, 10000.0)
        self._step.setValue(200.0)
        grid.addWidget(self._step, 1, 1)

        grid.addWidget(QLabel("Threshold (dB):"), 1, 2)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(-120.0, 40.0)
        self._threshold.setValue(-20.0)
        grid.addWidget(self._threshold, 1, 3)

        layout.addLayout(grid)

        # Start/Stop button
        self._start_btn = QPushButton("Start Scan")
        self._start_btn.clicked.connect(self._toggle_scan)
        layout.addWidget(self._start_btn)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        # Results
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Frequency", "Peak (dB)"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._table, 1)

        # Close
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _toggle_scan(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            return

        self._table.setRowCount(0)
        self._hits.clear()
        self._progress.setValue(0)

        start_hz = self._start.value() * 1e6
        end_hz = self._end.value() * 1e6
        step_hz = self._step.value() * 1e3
        if end_hz <= start_hz:
            return

        self._worker = _ScanWorker(
            self._device, start_hz, end_hz, step_hz, self._threshold.value(), self
        )
        self._worker.hit.connect(self._on_hit)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.start()
        self._start_btn.setText("Stop Scan")

    def _on_hit(self, freq_hz: float, peak_db: float):
        self._hits.append((freq_hz, peak_db))
        row = self._table.rowCount()
        self._table.insertRow(row)
        if freq_hz >= 1e9:
            text = f"{freq_hz/1e9:.6f} GHz"
        else:
            text = f"{freq_hz/1e6:.3f} MHz"
        self._table.setItem(row, 0, QTableWidgetItem(text))
        self._table.setItem(row, 1, QTableWidgetItem(f"{peak_db:.1f}"))

    def _on_finished(self):
        self._start_btn.setText("Start Scan")

    def reject(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        super().reject()
