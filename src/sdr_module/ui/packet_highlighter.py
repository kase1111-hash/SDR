"""
Packet highlighter for automatic detection and visualization.

Integrates signal detection with waterfall display to
automatically highlight detected packets with protocol colors.
"""

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

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
    """

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

        # Detection state
        self._noise_floor = np.full(waterfall.width, -100.0)
        self._active_signals: Dict[int, Dict] = {}  # bin -> signal info
        self._lock = Lock()

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

    @property
    def config(self) -> DetectionConfig:
        return self._config

    @config.setter
    def config(self, config: DetectionConfig) -> None:
        self._config = config

    @property
    def detected_packets(self) -> List[DetectedPacket]:
        return self._detected_packets.copy()

    @property
    def statistics(self) -> Dict:
        return {
            "total_packets": self._total_packets,
            "by_protocol": self._packets_by_protocol.copy(),
            "active_signals": len(self._active_signals),
        }

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
            power_db: Power spectrum in dB
            samples: Optional I/Q samples for classification
            timestamp: Optional timestamp

        Returns:
            List of new highlights created
        """
        timestamp = timestamp or time.time()
        new_highlights = []

        with self._lock:
            # Update noise floor (adaptive)
            if self._config.mode == DetectionMode.ADAPTIVE:
                self._update_noise_floor(power_db)

            # Detect signals above threshold
            signals = self._detect_signals(power_db)

            # Track signal continuity
            for signal in signals:
                bin_start, bin_end, peak_power = signal
                center_bin = (bin_start + bin_end) // 2

                # Check if continuing existing signal
                matched = False
                for active_bin, info in list(self._active_signals.items()):
                    if abs(active_bin - center_bin) < 5:  # Allow some drift
                        # Continue existing signal
                        info["duration"] += 1
                        info["peak_power"] = max(info["peak_power"], peak_power)
                        info["bin_end"] = max(info["bin_end"], bin_end)
                        info["bin_start"] = min(info["bin_start"], bin_start)
                        matched = True
                        break

                if not matched:
                    # New signal
                    self._active_signals[center_bin] = {
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                        "start_time": timestamp,
                        "duration": 1,
                        "peak_power": peak_power,
                        "samples": [] if samples is not None else None,
                    }

            # Collect samples for active signals
            if samples is not None:
                for info in self._active_signals.values():
                    if info["samples"] is not None:
                        # Store subset of samples for classification
                        info["samples"].append(samples[:1024].copy())

            # Check for ended signals
            ended_signals = []
            for active_bin in list(self._active_signals.keys()):
                signal_found = False
                for signal in signals:
                    bin_start, bin_end, _ = signal
                    center_bin = (bin_start + bin_end) // 2
                    if abs(active_bin - center_bin) < 5:
                        signal_found = True
                        break

                if not signal_found:
                    info = self._active_signals[active_bin]
                    info["gap"] = info.get("gap", 0) + 1

                    if info["gap"] > self._config.merge_gap_lines:
                        ended_signals.append((active_bin, info))
                        del self._active_signals[active_bin]

            # Process ended signals
            for active_bin, info in ended_signals:
                if info["duration"] >= self._config.min_duration_lines:
                    highlight = self._finalize_packet(info, samples, timestamp)
                    if highlight:
                        new_highlights.append(highlight)

        return new_highlights

    def _update_noise_floor(self, power_db: np.ndarray) -> None:
        """Update adaptive noise floor estimate."""
        # Use minimum of current and exponential average
        alpha = self._config.noise_floor_alpha
        self._noise_floor = (
            alpha * np.minimum(power_db, self._noise_floor + 10)
            + (1 - alpha) * self._noise_floor
        )

    def _detect_signals(self, power_db: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect signals in spectrum.

        Returns:
            List of (bin_start, bin_end, peak_power) tuples
        """
        # Calculate threshold
        if self._config.mode == DetectionMode.ADAPTIVE:
            threshold = self._noise_floor + self._config.snr_threshold_db
        else:
            threshold = np.full_like(power_db, self._config.threshold_db)

        # Find bins above threshold
        above = power_db > threshold

        # Find contiguous regions
        signals: List[Tuple[int, int, float]] = []
        in_signal = False
        signal_start = 0
        peak_power: float = -200.0

        for i in range(len(above)):
            if above[i] and not in_signal:
                # Signal start
                in_signal = True
                signal_start = i
                peak_power = power_db[i]
            elif above[i] and in_signal:
                # Continue signal
                peak_power = max(peak_power, power_db[i])
            elif not above[i] and in_signal:
                # Signal end
                in_signal = False
                bandwidth_bins = i - signal_start
                bandwidth_hz = bandwidth_bins * self._sample_rate / len(power_db)

                if (
                    self._config.min_bandwidth_hz
                    <= bandwidth_hz
                    <= self._config.max_bandwidth_hz
                ):
                    signals.append((signal_start, i - 1, peak_power))

        # Handle signal at end
        if in_signal:
            bandwidth_bins = len(above) - signal_start
            bandwidth_hz = bandwidth_bins * self._sample_rate / len(power_db)
            if (
                self._config.min_bandwidth_hz
                <= bandwidth_hz
                <= self._config.max_bandwidth_hz
            ):
                signals.append((signal_start, len(above) - 1, peak_power))

        return signals

    def _finalize_packet(
        self, info: Dict, samples: Optional[np.ndarray], end_time: float
    ) -> Optional[PacketHighlight]:
        """
        Finalize a detected packet and create highlight.

        Args:
            info: Signal tracking info
            samples: Optional samples for classification
            end_time: End timestamp

        Returns:
            PacketHighlight or None
        """
        # Calculate frequencies
        center_bin = (info["bin_start"] + info["bin_end"]) // 2
        center_freq = self._waterfall.get_frequency_at_bin(center_bin)
        freq_start = self._waterfall.get_frequency_at_bin(info["bin_start"])
        freq_end = self._waterfall.get_frequency_at_bin(info["bin_end"])
        bandwidth = freq_end - freq_start

        # Classify signal
        protocol = "unknown"
        classification = None

        if self._config.mode in (DetectionMode.CLASSIFIER, DetectionMode.PROTOCOL):
            if info.get("samples") and len(info["samples"]) > 0:
                # Concatenate collected samples
                all_samples = np.concatenate(info["samples"])

                # Classify
                classification = self._classifier.classify(all_samples)

                if classification.signal_type == SignalType.ANALOG:
                    protocol = "analog"
                elif classification.signal_type == SignalType.DIGITAL:
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

        # Estimate SNR
        snr_db = info["peak_power"] - np.mean(
            self._noise_floor[info["bin_start"] : info["bin_end"] + 1]
        )

        # Create detected packet record
        packet = DetectedPacket(
            start_time=info["start_time"],
            end_time=end_time,
            center_freq_hz=center_freq,
            bandwidth_hz=bandwidth,
            peak_power_db=info["peak_power"],
            snr_db=snr_db,
            protocol=protocol,
            classification=classification,
            waterfall_lines=info["duration"],
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
            duration_lines=info["duration"],
            protocol=protocol,
            label=f"{protocol.upper()} {snr_db:.1f}dB",
            confidence=min(1.0, snr_db / 30),
            metadata={
                "peak_power_db": info["peak_power"],
                "snr_db": snr_db,
                "bandwidth_hz": bandwidth,
            },
        )

        return highlight

    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self._noise_floor.fill(-100)
            self._active_signals.clear()

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

        self.spectrum = SpectrumAnalyzer(fft_size=width)
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
        # Compute spectrum
        result = self.spectrum.compute_spectrum(
            samples, self._center_freq, self._sample_rate
        )

        # Add to waterfall
        self.waterfall.add_spectrum_line(result.power_db)

        # Detect and highlight packets
        self.highlighter.process_spectrum(result.power_db, samples)

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
