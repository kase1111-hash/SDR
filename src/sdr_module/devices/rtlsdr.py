"""
RTL-SDR device driver wrapper.

Provides interface for RTL-SDR Blog V3/V4 and compatible devices.
Specifications:
    - Frequency: 500 kHz - 1.766 GHz (24-1766 MHz native, HF via direct sampling)
    - Bandwidth: 2.4 MHz
    - Sample Rate: 2.56 MS/s (2.4 MS/s recommended)
    - ADC: 8-bit (RTL2832U)
    - RX Only
"""

import logging
from threading import Thread
from typing import Callable, List, Optional

import numpy as np

from .base import (
    DeviceCapability,
    DeviceInfo,
    DeviceSpec,
    SDRDevice,
)

logger = logging.getLogger(__name__)

# RTL-SDR specifications from spec sheet
RTLSDR_SPEC = DeviceSpec(
    freq_min=500e3,  # 500 kHz (direct sampling)
    freq_max=1.766e9,  # 1.766 GHz
    sample_rate_min=225001,  # ~225 kHz minimum
    sample_rate_max=2.56e6,  # 2.56 MS/s max (2.4 recommended)
    bandwidth_max=2.4e6,  # 2.4 MHz
    adc_bits=8,
    gain_min=0.0,
    gain_max=49.6,  # R820T2 max gain
    max_input_power=10.0,  # +10 dBm max input
)


class RTLSDRDevice(SDRDevice):
    """
    RTL-SDR device driver.

    Wraps the rtlsdr library for RTL2832U-based devices.
    """

    # Valid gain values for R820T2 tuner
    VALID_GAINS = [
        0.0,
        0.9,
        1.4,
        2.7,
        3.7,
        7.7,
        8.7,
        12.5,
        14.4,
        15.7,
        16.6,
        19.7,
        20.7,
        22.9,
        25.4,
        28.0,
        29.7,
        32.8,
        33.8,
        36.4,
        37.2,
        38.6,
        40.2,
        42.1,
        43.4,
        43.9,
        44.5,
        48.0,
        49.6,
    ]

    def __init__(self):
        super().__init__()
        self._device = None
        self._spec = RTLSDR_SPEC
        self._direct_sampling = False

    @staticmethod
    def get_device_count() -> int:
        """Get number of RTL-SDR devices connected."""
        try:
            from rtlsdr import RtlSdr

            return RtlSdr.get_device_count()
        except ImportError:
            logger.warning("rtlsdr library not installed")
            return 0
        except Exception as e:
            logger.error(f"Error getting device count: {e}")
            return 0

    @staticmethod
    def get_device_serial(index: int) -> Optional[str]:
        """Get serial number of device at index."""
        try:
            from rtlsdr import RtlSdr

            return RtlSdr.get_device_serial(index)
        except Exception:
            return None

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """List all available RTL-SDR devices."""
        devices = []
        try:
            from rtlsdr import RtlSdr

            count = RtlSdr.get_device_count()
            for i in range(count):
                serial = RtlSdr.get_device_serial(i) or f"rtlsdr_{i}"
                devices.append(
                    DeviceInfo(
                        name=f"RTL-SDR #{i}",
                        serial=serial,
                        manufacturer="RTL-SDR Blog",
                        product="RTL2832U",
                        index=i,
                        capabilities=[
                            DeviceCapability.RX,
                            DeviceCapability.BIAS_TEE,
                            DeviceCapability.DIRECT_SAMPLE,
                        ],
                    )
                )
        except ImportError:
            logger.warning("rtlsdr library not installed")
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
        return devices

    def open(self, index: int = 0) -> bool:
        """Open RTL-SDR device."""
        if self._is_open:
            logger.warning("Device already open")
            return True

        try:
            from rtlsdr import RtlSdr

            self._device = RtlSdr(device_index=index)
            self._is_open = True

            # Get device info
            serial = self.get_device_serial(index) or f"rtlsdr_{index}"
            self._info = DeviceInfo(
                name=f"RTL-SDR #{index}",
                serial=serial,
                manufacturer="RTL-SDR Blog",
                product="RTL2832U",
                index=index,
                capabilities=[
                    DeviceCapability.RX,
                    DeviceCapability.BIAS_TEE,
                    DeviceCapability.DIRECT_SAMPLE,
                ],
            )

            # Set defaults
            self._device.sample_rate = 2.4e6
            self._device.center_freq = 100e6
            self._device.gain = "auto"

            self._state.sample_rate = 2.4e6
            self._state.frequency = 100e6
            self._state.gain_mode = "auto"
            self._state.bandwidth = 2.4e6

            logger.info(f"Opened RTL-SDR device: {serial}")
            return True

        except ImportError:
            logger.error(
                "rtlsdr library not installed. Install with: pip install pyrtlsdr"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to open RTL-SDR: {e}")
            return False

    def close(self) -> None:
        """Close RTL-SDR device."""
        if self._state.is_streaming:
            self.stop_rx()

        if self._device is not None:
            try:
                self._device.close()
            except Exception as e:
                logger.error(f"Error closing device: {e}")
            finally:
                self._device = None
                self._is_open = False
                logger.info("RTL-SDR device closed")

    def set_frequency(self, freq_hz: float) -> bool:
        """Set center frequency."""
        if not self._is_open or self._device is None:
            return False

        # Check if we need direct sampling for HF
        if freq_hz < 24e6:
            if not self._direct_sampling:
                self._enable_direct_sampling(True)
        else:
            if self._direct_sampling:
                self._enable_direct_sampling(False)

        try:
            self._device.center_freq = freq_hz
            self._state.frequency = freq_hz
            logger.debug(f"Set frequency to {freq_hz/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False

    def set_sample_rate(self, rate_hz: float) -> bool:
        """Set sample rate."""
        if not self._is_open or self._device is None:
            return False

        # Clamp to valid range
        rate_hz = max(
            self._spec.sample_rate_min, min(rate_hz, self._spec.sample_rate_max)
        )

        try:
            self._device.sample_rate = rate_hz
            self._state.sample_rate = rate_hz
            self._state.bandwidth = rate_hz  # Bandwidth follows sample rate
            logger.debug(f"Set sample rate to {rate_hz/1e6:.3f} MS/s")
            return True
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}")
            return False

    def set_bandwidth(self, bw_hz: float) -> bool:
        """Set filter bandwidth (limited by sample rate in RTL-SDR)."""
        # RTL-SDR bandwidth is tied to sample rate
        return self.set_sample_rate(bw_hz)

    def set_gain(self, gain_db: float) -> bool:
        """Set gain value."""
        if not self._is_open or self._device is None:
            return False

        # Find nearest valid gain
        nearest_gain = min(self.VALID_GAINS, key=lambda x: abs(x - gain_db))

        try:
            self._device.gain = nearest_gain
            self._state.gain = nearest_gain
            self._state.gain_mode = "manual"
            logger.debug(f"Set gain to {nearest_gain} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set gain: {e}")
            return False

    def set_gain_mode(self, auto: bool) -> bool:
        """Set AGC mode."""
        if not self._is_open or self._device is None:
            return False

        try:
            if auto:
                self._device.gain = "auto"
                self._state.gain_mode = "auto"
            else:
                # Set to current gain value
                self._device.gain = self._state.gain
                self._state.gain_mode = "manual"
            return True
        except Exception as e:
            logger.error(f"Failed to set gain mode: {e}")
            return False

    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Start receiving samples."""
        if not self._is_open or self._device is None:
            return False

        if self._state.is_streaming:
            logger.warning("Already streaming")
            return True

        self._rx_callback = callback
        self._stop_event.clear()

        def rx_thread():
            """Background thread for receiving samples."""
            try:
                while not self._stop_event.is_set():
                    samples = self._device.read_samples(256 * 1024)
                    if self._rx_callback:
                        self._rx_callback(samples)
                    else:
                        try:
                            self._sample_queue.put_nowait(samples)
                        except Exception:
                            pass  # Queue full, drop samples
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"RX thread error: {e}")

        self._rx_thread = Thread(target=rx_thread, daemon=True)
        self._rx_thread.start()
        self._state.is_streaming = True
        logger.info("Started RX streaming")
        return True

    def stop_rx(self) -> bool:
        """Stop receiving samples."""
        if not self._state.is_streaming:
            return True

        self._stop_event.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None

        self._state.is_streaming = False
        logger.info("Stopped RX streaming")
        return True

    def set_bias_tee(self, enabled: bool) -> bool:
        """Enable/disable bias tee (V3+ only)."""
        if not self._is_open or self._device is None:
            return False

        try:
            self._device.set_bias_tee(enabled)
            self._state.bias_tee_enabled = enabled
            logger.info(f"Bias tee {'enabled' if enabled else 'disabled'}")
            return True
        except AttributeError:
            logger.warning("Bias tee not supported by this rtlsdr library version")
            return False
        except Exception as e:
            logger.error(f"Failed to set bias tee: {e}")
            return False

    def _enable_direct_sampling(self, enabled: bool) -> bool:
        """Enable/disable direct sampling mode for HF."""
        if not self._is_open or self._device is None:
            return False

        try:
            # 0 = disabled, 1 = I-branch, 2 = Q-branch
            mode = 2 if enabled else 0  # Q-branch for RTL-SDR Blog
            self._device.set_direct_sampling(mode)
            self._direct_sampling = enabled
            logger.info(
                f"Direct sampling {'enabled (Q-branch)' if enabled else 'disabled'}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set direct sampling: {e}")
            return False

    def get_tuner_gains(self) -> List[float]:
        """Get list of valid tuner gain values."""
        return self.VALID_GAINS.copy()
