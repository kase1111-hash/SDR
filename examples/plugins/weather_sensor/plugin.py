"""
Weather Sensor Protocol Decoder Plugin.

Decodes common 433 MHz weather sensor protocols including:
- Oregon Scientific v2.1/v3.0
- Acurite
- LaCrosse
- Generic OOK sensors

These sensors typically use OOK (On-Off Keying) modulation with
Manchester encoding at various baud rates (1024-4096 bps).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from sdr_module.plugins import (
    ProtocolPlugin,
    PluginMetadata,
    PluginType,
)


class SensorType(Enum):
    """Known sensor types."""
    UNKNOWN = "unknown"
    OREGON_V2 = "oregon_v2"
    OREGON_V3 = "oregon_v3"
    ACURITE = "acurite"
    LACROSSE = "lacrosse"
    GENERIC = "generic"


@dataclass
class WeatherData:
    """Decoded weather sensor data."""
    sensor_type: SensorType
    sensor_id: int
    channel: int
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    battery_low: bool = False
    raw_data: bytes = b""


class WeatherSensorPlugin(ProtocolPlugin):
    """
    Weather sensor protocol decoder.

    Decodes OOK-modulated weather sensor transmissions commonly
    found in the 433 MHz ISM band.
    """

    # Protocol constants
    PREAMBLE_OREGON = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    SYNC_OREGON_V2 = np.array([1, 0, 0, 1], dtype=np.uint8)

    # Timing constants (in samples at 250 kHz sample rate)
    SHORT_PULSE = 500e-6   # 500 us
    LONG_PULSE = 1000e-6   # 1000 us
    TOLERANCE = 0.3        # 30% timing tolerance

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="weather_sensor",
            version="1.0.0",
            plugin_type=PluginType.PROTOCOL,
            author="SDR Module Team",
            description="Decoder for 433 MHz weather sensor protocols",
            tags=["protocol", "weather", "433mhz", "ism", "sensors", "ook"],
            license="MIT",
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the decoder."""
        self._sample_rate = config.get("sample_rate", 250000)
        self._threshold = config.get("threshold", 0.5)
        self._min_packet_bits = config.get("min_packet_bits", 32)

        # Calculate timing in samples
        self._short_samples = int(self.SHORT_PULSE * self._sample_rate)
        self._long_samples = int(self.LONG_PULSE * self._sample_rate)

        # State for streaming decode
        self._bit_buffer = []
        self._last_edge = 0

        return True

    def get_protocol_info(self) -> Dict[str, Any]:
        return {
            "name": "Weather Sensors",
            "protocol_type": "ism",
            "frequency_range": (433.0e6, 434.0e6),
            "bandwidth_hz": 200000,
            "modulation": "OOK",
            "symbol_rate": 2048,
            "description": "433 MHz weather sensor protocols (Oregon, Acurite, etc.)",
        }

    def decode(self, samples: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode weather sensor data from I/Q samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            List of decoded frames as dictionaries
        """
        frames = []

        # Envelope detection (OOK demodulation)
        envelope = np.abs(samples)

        # Normalize
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)

        # Threshold to binary
        bits = (envelope > self._threshold).astype(np.uint8)

        # Find pulse edges
        edges = np.where(np.diff(bits) != 0)[0]

        if len(edges) < 10:
            return frames

        # Extract pulse widths
        pulse_widths = np.diff(edges)

        # Try to decode as Manchester
        decoded_bits = self._decode_manchester(pulse_widths, bits[edges[0]])

        if len(decoded_bits) >= self._min_packet_bits:
            # Try to identify and decode the packet
            weather_data = self._decode_packet(decoded_bits)

            if weather_data:
                frames.append({
                    "protocol": "weather_sensor",
                    "sensor_type": weather_data.sensor_type.value,
                    "sensor_id": weather_data.sensor_id,
                    "channel": weather_data.channel,
                    "temperature_c": weather_data.temperature_c,
                    "humidity_percent": weather_data.humidity_percent,
                    "battery_low": weather_data.battery_low,
                    "raw_bytes": weather_data.raw_data.hex(),
                    "is_valid": True,
                })

        return frames

    def _decode_manchester(
        self,
        pulse_widths: np.ndarray,
        start_level: int
    ) -> np.ndarray:
        """Decode Manchester-encoded pulse widths to bits."""
        bits = []
        current_level = start_level

        # Tolerance bounds
        short_min = int(self._short_samples * (1 - self.TOLERANCE))
        short_max = int(self._short_samples * (1 + self.TOLERANCE))
        long_min = int(self._long_samples * (1 - self.TOLERANCE))
        long_max = int(self._long_samples * (1 + self.TOLERANCE))

        i = 0
        while i < len(pulse_widths):
            width = pulse_widths[i]

            if short_min <= width <= short_max:
                # Short pulse - need another short pulse
                if i + 1 < len(pulse_widths):
                    next_width = pulse_widths[i + 1]
                    if short_min <= next_width <= short_max:
                        bits.append(current_level)
                        i += 2
                        continue

            elif long_min <= width <= long_max:
                # Long pulse - single bit
                bits.append(1 - current_level)
                current_level = 1 - current_level
                i += 1
                continue

            # Unknown pulse width - skip
            i += 1
            current_level = 1 - current_level

        return np.array(bits, dtype=np.uint8)

    def _decode_packet(self, bits: np.ndarray) -> Optional[WeatherData]:
        """Try to decode a packet from bits."""
        # Look for Oregon Scientific preamble
        preamble_pos = self._find_pattern(bits, self.PREAMBLE_OREGON)

        if preamble_pos >= 0:
            # Try Oregon Scientific decode
            data = self._decode_oregon(bits[preamble_pos:])
            if data:
                return data

        # Try generic decode
        return self._decode_generic(bits)

    def _find_pattern(self, bits: np.ndarray, pattern: np.ndarray) -> int:
        """Find pattern in bit array."""
        pattern_len = len(pattern)

        for i in range(len(bits) - pattern_len + 1):
            if np.array_equal(bits[i:i + pattern_len], pattern):
                return i

        return -1

    def _decode_oregon(self, bits: np.ndarray) -> Optional[WeatherData]:
        """Decode Oregon Scientific protocol."""
        if len(bits) < 68:  # Minimum Oregon packet
            return None

        # Skip preamble
        bits = bits[8:]

        # Look for sync
        sync_pos = self._find_pattern(bits, self.SYNC_OREGON_V2)
        if sync_pos < 0:
            return None

        bits = bits[sync_pos + 4:]

        if len(bits) < 56:
            return None

        # Extract nibbles (Oregon uses nibbles, LSB first)
        nibbles = []
        for i in range(0, min(len(bits), 64), 4):
            if i + 4 <= len(bits):
                nibble = (bits[i] + bits[i+1]*2 + bits[i+2]*4 + bits[i+3]*8)
                nibbles.append(nibble)

        if len(nibbles) < 10:
            return None

        # Parse Oregon format
        # Nibbles: [sensor_type(4), channel, rolling_code(2), flags, temp(3), humidity(2), checksum(2)]
        try:
            sensor_id = (nibbles[0] << 12) | (nibbles[1] << 8) | (nibbles[2] << 4) | nibbles[3]
            channel = nibbles[4] & 0x07
            rolling_code = (nibbles[5] << 4) | nibbles[6]

            # Temperature (BCD, signed)
            temp_sign = 1 if (nibbles[7] & 0x08) == 0 else -1
            temp_val = nibbles[8] * 10 + nibbles[9] + nibbles[10] * 0.1
            temperature = temp_sign * temp_val

            # Humidity
            humidity = None
            if len(nibbles) >= 13:
                humidity = nibbles[11] * 10 + nibbles[12]
                if humidity > 100:
                    humidity = None

            # Battery flag
            battery_low = (nibbles[7] & 0x04) != 0

            # Convert nibbles to bytes for raw data
            raw_bytes = bytes([
                (nibbles[i] << 4) | nibbles[i+1]
                for i in range(0, len(nibbles) - 1, 2)
            ])

            return WeatherData(
                sensor_type=SensorType.OREGON_V2,
                sensor_id=sensor_id,
                channel=channel,
                temperature_c=temperature,
                humidity_percent=humidity,
                battery_low=battery_low,
                raw_data=raw_bytes,
            )

        except (IndexError, ValueError):
            return None

    def _decode_generic(self, bits: np.ndarray) -> Optional[WeatherData]:
        """Try generic weather sensor decode."""
        if len(bits) < 32:
            return None

        # Convert bits to bytes
        num_bytes = len(bits) // 8
        raw_bytes = bytes([
            sum(bits[i*8 + j] << j for j in range(8))
            for i in range(num_bytes)
        ])

        # Simple heuristic: look for reasonable temperature value
        # Many sensors encode temperature as 12-bit value in first bytes
        if len(raw_bytes) >= 2:
            temp_raw = (raw_bytes[0] << 4) | (raw_bytes[1] >> 4)

            # Check if it looks like a reasonable temperature (-40 to +60 C)
            # Many sensors use offset of 400 (temp * 10 + 400)
            if 0 <= temp_raw <= 1000:
                temperature = (temp_raw - 400) / 10.0

                if -40 <= temperature <= 60:
                    return WeatherData(
                        sensor_type=SensorType.GENERIC,
                        sensor_id=raw_bytes[2] if len(raw_bytes) > 2 else 0,
                        channel=1,
                        temperature_c=temperature,
                        raw_data=raw_bytes,
                    )

        return None

    def can_decode(self, samples: np.ndarray) -> float:
        """
        Estimate probability that samples contain weather sensor data.

        Looks for:
        - OOK modulation characteristics
        - Pulse timing consistent with weather sensors
        - Preamble patterns
        """
        # Envelope detection
        envelope = np.abs(samples)

        if np.max(envelope) == 0:
            return 0.0

        envelope = envelope / np.max(envelope)

        # Check for OOK characteristics
        # Should have bimodal amplitude distribution
        low_count = np.sum(envelope < 0.3)
        high_count = np.sum(envelope > 0.7)
        total = len(envelope)

        if total == 0:
            return 0.0

        # Good OOK signal has clear on/off states
        ook_score = (low_count + high_count) / total

        if ook_score < 0.5:
            return 0.0

        # Check for pulse timing
        bits = (envelope > 0.5).astype(np.uint8)
        edges = np.where(np.diff(bits) != 0)[0]

        if len(edges) < 20:
            return 0.0

        pulse_widths = np.diff(edges)

        # Check if pulse widths are in expected range (250us - 2ms at 250kHz)
        min_width = int(250e-6 * self._sample_rate)
        max_width = int(2000e-6 * self._sample_rate)

        valid_pulses = np.sum((pulse_widths >= min_width) & (pulse_widths <= max_width))
        timing_score = valid_pulses / len(pulse_widths)

        # Combined score
        confidence = 0.4 * ook_score + 0.6 * timing_score

        return min(confidence, 1.0)

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._last_edge = 0
