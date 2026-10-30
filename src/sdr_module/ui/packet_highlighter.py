"""
Packet highlighter for automatic detection and visualization.

Integrates signal detection with waterfall display to
automatically highlight detected packets with protocol colors.
"""

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Dict, List, Optional

import numpy as np

from ..dsp.bursts import Burst, BurstDetector, frame_rows
from ..dsp.classifiers import ClassificationResult, SignalClassifier, SignalType
from ..dsp.spectrum import SpectrumAnalyzer
from ..protocols.detector import ProtocolDetector
from .waterfall import PacketHighlight, WaterfallDisplay


class DetectionMode(Enum):
    """Packet detection modes."""

    THRESHOLD = "threshold"  # Simple power threshold
    ADAPTIVE = "adaptive"  # Adaptive noise floor tracking
    CLASSIFIER = "classifier"  # Use signal classifier
    PROTOCOL = "protocol"  # Full protocol detection


@dataclass
class DetectionConfig:
    """Configuration for packet detection."""

    mode: DetectionMode = DetectionMode.ADAPTIVE
    threshold_db: float = -50  # Detection threshold (dB)
    min_bandwidth_hz: float = 1000  # Minimum signal bandwidth
    max_bandwidth_hz: float = 1e6  # Maximum signal bandwidth
    min_duration_lines: int = 3  # Minimum packet duration
    max_duration_lines: int = 1000  # Maximum packet duration
    noise_floor_alpha: float = 0.01  # Noise floor averaging factor
    snr_threshold_db: float = 10  # SNR threshold for detection
    merge_gap_lines: int = 2  # Merge packets within this gap
    merge_gap_hz: float = 5000  # Merge packets within this freq gap


@dataclass
class DetectedPacket:
    """Detected packet information."""

    start_time: float  # Unix timestamp
    end_time: float
    center_freq_hz: float
    bandwidth_hz: float
    peak_power_db: float
    snr_db: float
    protocol: str
    classification: Optional[ClassificationResult] = None
    waterfall_lines: int = 0


class PacketHighlighter:
    """
    Automatic packet detection and highlighting.

    Analyzes spectrum data in real-time to detect signal bursts
    and highlights them on the waterfall display with protocol-
    specific colors.

    Detection (ADAPTIVE, the default) uses
    :class:`~sdr_module.dsp.bursts.BurstDetector`: the noise floor is taken
    from the first line and adapted quickly at first, so packets are found
    from the second line on (it used to start at -100 dB and rise 0.1 dB a
    line: nothing for about 12 s, then a flood of noise "packets"), and a
    packet must stand ``snr_threshold_db`` above the floor over a few
    neighbouring bins and be ``min_bandwidth_hz`` to ``max_bandwidth_hz``
    wide. Feed it dBFS lines (as :class:`LivePacketDisplay` does).
    THRESHOLD compares against ``threshold_db`` instead.
    """

    #: Samples of each line kept per packet for classification.
    KEEP_SAMPLES = 4096

    def __init__(
        self,
        waterfall: WaterfallDisplay,
        sample_rate: float,
        config: Optional[DetectionConfig] = None,
    ):
        """
        Initialize packet highlighter.

        Args:
            waterfall: Waterfall display to highlight on
            sample_rate: Sample rate for frequency calculations
            config: Detection configuration
        """
        self._waterfall = waterfall
        self._sample_rate = sample_rate
        self._config = config or DetectionConfig()
        self._detector = self._make_detector()
        self._lock = Lock()
        # Samples of the packets in progress (detector burst id -> blocks).
        self._samples: Dict[int, List[np.ndarray]] = {}
        self._lines = 0

        # Classifiers and detectors
        self._classifier = SignalClassifier(sample_rate)
        self._protocol_detector: Optional[ProtocolDetector] = None

        # Detected packets history
        self._detected_packets: List[DetectedPacket] = []
        self._max_history = 1000

        # Callbacks
        self._on_packet_detected: Optional[Callable[[DetectedPacket], None]] = None

        # Statistics
        self._total_packets = 0
        self._packets_by_protocol: Dict[str, int] = {}

    def _make_detector(self) -> BurstDetector:
        cfg = self._config
        return BurstDetector(
            snr_db=cfg.snr_threshold_db,
            min_width_hz=cfg.min_bandwidth_hz,
            max_width_hz=cfg.max_bandwidth_hz,
            gap_rows=cfg.merge_gap_lines,
            merge_hz=cfg.merge_gap_hz,
            max_duration_s=1e9,
        )

    @property
    def config(self) -> DetectionConfig:
        return self._config

    @config.setter
    def config(self, config: DetectionConfig) -> None:
        self._config = config
        self._detector = self._make_detector()

    @property
    def detected_packets(self) -> List[DetectedPacket]:
        return self._detected_packets.copy()

    @property
    def statistics(self) -> Dict:
        return {
            "total_packets": self._total_packets,
            "by_protocol": self._packets_by_protocol.copy(),
            "active_signals": len(self._detector.active),
        }

    @property
    def noise_floor_db(self) -> Optional[np.ndarray]:
        """The per-bin noise floor (dB), once a line has been seen."""
        return self._detector.noise_floor_db()

    def set_protocol_detector(self, detector: ProtocolDetector) -> None:
        """Set protocol detector for advanced detection."""
        self._protocol_detector = detector

    def set_on_packet_detected(
        self, callback: Callable[[DetectedPacket], None]
    ) -> None:
        """Set callback for packet detection events."""
        self._on_packet_detected = callback

    def process_spectrum(
        self,
        power_db: np.ndarray,
        samples: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[PacketHighlight]:
        """
        Process spectrum line and detect/highlight packets.

        Args:
            power_db: Power spectrum in dB (dBFS, 0 Hz in the middle)
            samples: Optional I/Q samples for classification
            timestamp: Optional timestamp

        Returns:
            List of new highlights created
        """
        timestamp = time.time() if timestamp is None else float(timestamp)
        new_highlights = []
        power = np.asarray(power_db, dtype=np.float64)
        with self._lock:
            self._lines += 1
            detector = self._detector
            if self._config.mode == DetectionMode.THRESHOLD:
                ended = self._threshold_line(power, timestamp)
            else:
                ended = detector.process(
                    power, self._waterfall_centre(), self._sample_rate, timestamp
                )
            if samples is not None and self._config.mode in (
                DetectionMode.CLASSIFIER,
                DetectionMode.PROTOCOL,
            ):
                for burst in detector.active:
                    kept = self._samples.setdefault(burst.ident, [])
                    if sum(len(k) for k in kept) < 16 * self.KEEP_SAMPLES:
                        kept.append(np.asarray(samples[: self.KEEP_SAMPLES]).copy())
            for burst in ended:
                kept = self._samples.pop(burst.ident, [])
                if burst.rows < self._config.min_duration_lines:
                    continue
                highlight = self._finalize_packet(burst, kept, timestamp)
                if highlight:
                    new_highlights.append(highlight)
        return new_highlights

    def _waterfall_centre(self) -> float:
        # The waterfall's frequency axis: its first bin is centre - rate/2.
        return float(self._waterfall.get_frequency_at_bin(self._waterfall.width // 2))

    def _threshold_line(self, power: np.ndarray, timestamp: float) -> List[Burst]:
        """THRESHOLD mode: a fixed floor ``snr_threshold_db`` below
        ``threshold_db``, so bins above ``threshold_db`` are packets."""
        detector = self._detector
        floor = self._config.threshold_db - self._config.snr_threshold_db
        level = np.full(len(power), 10.0 ** (floor / 10.0))
        detector._floor = level.copy()
        detector._rows = max(detector._rows, detector.settle_rows + 1)
        ended = detector.process(
            power, self._waterfall_centre(), self._sample_rate, timestamp
        )
        detector._floor = level
        return ended

    def _finalize_packet(
        self, burst: Burst, kept: List[np.ndarray], end_time: float
    ) -> Optional[PacketHighlight]:
        """
        Finalize a detected packet and create highlight.

        Args:
            burst: The burst the detector followed
            kept: Samples collected while it lasted (for classification)
            end_time: End timestamp

        Returns:
            PacketHighlight or None
        """
        freq_start = burst.center_hz - burst.offset_hz + burst.lo_hz
        freq_end = burst.center_hz - burst.offset_hz + burst.hi_hz
        bandwidth = freq_end - freq_start

        # Classify signal
        protocol = "unknown"
        classification = None

        if self._config.mode in (DetectionMode.CLASSIFIER, DetectionMode.PROTOCOL):
            if kept:
                all_samples = np.concatenate(kept)
                # The packet to 0 Hz (it is somewhere in the span).
                t = np.arange(len(all_samples)) / self._sample_rate
                mixed = all_samples * np.exp(-2j * np.pi * burst.offset_hz * t)
                classification = self._classifier.classify(
                    mixed, search_hz=0.5 * bandwidth + 5e3
                )

                if classification.signal_type == SignalType.ANALOG:
                    protocol = "analog"
                elif classification.signal_type in (
                    SignalType.DIGITAL,
                    SignalType.PULSED,
                ):
                    if classification.modulation:
                        protocol = classification.modulation.value
                    else:
                        protocol = "digital"

                # Try protocol detection
                if (
                    self._protocol_detector
                    and self._config.mode == DetectionMode.PROTOCOL
                ):
                    matches = self._protocol_detector.detect(
                        all_samples, min_confidence=0.6
                    )
                    if matches:
                        protocol = matches[0].protocol_info.name.lower()

        snr_db = float(burst.snr_db)

        # Create detected packet record
        packet = DetectedPacket(
            start_time=burst.start_s,
            end_time=end_time,
            center_freq_hz=burst.center_hz,
            bandwidth_hz=bandwidth,
            peak_power_db=burst.peak_dbfs,
            snr_db=snr_db,
            protocol=protocol,
            classification=classification,
            waterfall_lines=burst.rows,
        )

        self._detected_packets.append(packet)
        if len(self._detected_packets) > self._max_history:
            self._detected_packets.pop(0)

        self._total_packets += 1
        self._packets_by_protocol[protocol] = (
            self._packets_by_protocol.get(protocol, 0) + 1
        )

        # Callback
        if self._on_packet_detected:
            self._on_packet_detected(packet)

        # Create waterfall highlight
        highlight = self._waterfall.add_packet_highlight(
            freq_start_hz=freq_start,
            freq_end_hz=freq_end,
            duration_lines=burst.rows,
            protocol=protocol,
            label=f"{protocol.upper()} {snr_db:.1f}dB",
            confidence=min(1.0, snr_db / 30),
            metadata={
                "peak_power_db": burst.peak_dbfs,
                "snr_db": snr_db,
                "bandwidth_hz": bandwidth,
            },
        )

        return highlight

    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self._detector.reset()
            self._samples.clear()

    def clear_history(self) -> None:
        """Clear packet history."""
        self._detected_packets.clear()
        self._total_packets = 0
        self._packets_by_protocol.clear()


class LivePacketDisplay:
    """
    Live packet display combining waterfall and highlighter.

    Convenience class for real-time packet visualization.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 512,
        sample_rate: float = 2.4e6,
        center_freq: float = 433.92e6,
    ):
        """
        Initialize live packet display.

        Args:
            width: Display width
            height: Display height
            sample_rate: Sample rate in Hz
            center_freq: Center frequency in Hz
        """
        from .waterfall import ColorMap

        self.waterfall = WaterfallDisplay(
            width=width, height=height, colormap=ColorMap.TURBO
        )
        self.waterfall.set_frequency_range(center_freq, sample_rate)

        self.highlighter = PacketHighlighter(self.waterfall, sample_rate)
        # (Kept for callers that read it; the lines are frame_rows' now.)
        self.spectrum = SpectrumAnalyzer(fft_size=width)

        self._width = width
        self._sample_rate = sample_rate
        self._center_freq = center_freq

    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Process I/Q samples and return waterfall image.

        Args:
            samples: Complex I/Q samples

        Returns:
            RGBA image as numpy array
        """
        # dBFS per bin over the whole frame: the strongest of its FFTs, so
        # a short packet anywhere in it shows.
        line, _mean = frame_rows(samples, self._width)

        # Add to waterfall
        self.waterfall.add_spectrum_line(line)

        # Detect and highlight packets
        self.highlighter.process_spectrum(line, samples)

        return self.waterfall.image

    def set_frequency(self, center_freq: float) -> None:
        """Set center frequency."""
        self._center_freq = center_freq
        self.waterfall.set_frequency_range(center_freq, self._sample_rate)

    @property
    def detected_packets(self) -> List[DetectedPacket]:
        return self.highlighter.detected_packets

    @property
    def image(self) -> np.ndarray:
        return self.waterfall.image
