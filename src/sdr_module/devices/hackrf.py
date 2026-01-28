"""
HackRF One device driver wrapper.

Provides interface for HackRF One from Great Scott Gadgets.
Specifications:
    - Frequency: 1 MHz - 6 GHz
    - Bandwidth: 20 MHz
    - Sample Rate: 8-20 MS/s
    - ADC/DAC: 8-bit (MAX5864)
    - Half-duplex TX/RX
    - TX Power: -10 to +15 dBm
"""

import logging
import queue
from threading import Thread
from typing import Callable, List, Optional

import numpy as np

from ..core.frequency_manager import is_tx_allowed
from .base import (
    DeviceCapability,
    DeviceInfo,
    DeviceSpec,
    SDRDevice,
)

logger = logging.getLogger(__name__)

# HackRF specifications from spec sheet
HACKRF_SPEC = DeviceSpec(
    freq_min=1e6,  # 1 MHz
    freq_max=6e9,  # 6 GHz
    sample_rate_min=8e6,  # 8 MS/s minimum recommended
    sample_rate_max=20e6,  # 20 MS/s max
    bandwidth_max=20e6,  # 20 MHz
    adc_bits=8,
    gain_min=0.0,
    gain_max=62.0,  # Combined LNA + VGA + AMP
    max_input_power=-5.0,  # -5 dBm max input (CAUTION!)
    tx_power_min=-10.0,  # -10 dBm
    tx_power_max=15.0,  # +15 dBm (frequency dependent)
)


class HackRFDevice(SDRDevice):
    """
    HackRF One device driver.

    Wraps the hackrf library for HackRF One transceiver.
    Supports both receive and transmit operations (half-duplex).
    """

    # Gain stages
    LNA_GAIN_VALUES = list(range(0, 41, 8))  # 0, 8, 16, 24, 32, 40 dB
    VGA_GAIN_VALUES = list(range(0, 63, 2))  # 0-62 dB in 2dB steps
    TX_VGA_GAIN_VALUES = list(range(0, 48, 1))  # 0-47 dB

    def __init__(self):
        super().__init__()
        self._device = None
        self._spec = HACKRF_SPEC
        self._lna_gain = 16
        self._vga_gain = 20
        self._tx_vga_gain = 20
        self._amp_enabled = False
        self._tx_callback: Optional[Callable[[], np.ndarray]] = None
        self._tx_thread: Optional[Thread] = None

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """List all available HackRF devices."""
        devices = []
        try:
            from hackrf import HackRF

            # Try to open and get info
            h = HackRF()
            serial = (
                h.get_serial_number() if hasattr(h, "get_serial_number") else "unknown"
            )
            devices.append(
                DeviceInfo(
                    name="HackRF One",
                    serial=serial,
                    manufacturer="Great Scott Gadgets",
                    product="HackRF One",
                    index=0,
                    capabilities=[
                        DeviceCapability.RX,
                        DeviceCapability.TX,
                        DeviceCapability.HALF_DUPLEX,
                        DeviceCapability.EXT_CLOCK,
                    ],
                )
            )
            h.close()
        except ImportError:
            logger.warning("hackrf library not installed")
        except Exception as e:
            logger.debug(f"No HackRF device found: {e}")
        return devices

    def open(self, index: int = 0) -> bool:
        """Open HackRF device."""
        if self._is_open:
            logger.warning("Device already open")
            return True

        try:
            from hackrf import HackRF

            self._device = HackRF()
            self._is_open = True

            # Get device info
            serial = "unknown"
            try:
                serial = self._device.get_serial_number()
            except Exception as e:
                logger.debug(f"Could not get HackRF serial number: {e}")

            self._info = DeviceInfo(
                name="HackRF One",
                serial=serial,
                manufacturer="Great Scott Gadgets",
                product="HackRF One",
                index=index,
                capabilities=[
                    DeviceCapability.RX,
                    DeviceCapability.TX,
                    DeviceCapability.HALF_DUPLEX,
                    DeviceCapability.EXT_CLOCK,
                ],
            )

            # Set defaults
            self.set_sample_rate(10e6)
            self.set_frequency(100e6)
            self.set_bandwidth(10e6)
            self.set_gain(30)

            logger.info(f"Opened HackRF device: {serial}")
            return True

        except ImportError:
            logger.error(
                "hackrf library not installed. Install with: pip install hackrf"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to open HackRF: {e}")
            return False

    def close(self) -> None:
        """Close HackRF device."""
        if self._state.is_streaming:
            self.stop_rx()
        if self._state.is_transmitting:
            self.stop_tx()

        if self._device is not None:
            try:
                self._device.close()
            except Exception as e:
                logger.error(f"Error closing device: {e}")
            finally:
                self._device = None
                self._is_open = False
                logger.info("HackRF device closed")

    def set_frequency(self, freq_hz: float) -> bool:
        """Set center frequency."""
        if not self._is_open or self._device is None or self._spec is None:
            return False

        # Validate range
        if freq_hz < self._spec.freq_min or freq_hz > self._spec.freq_max:
            logger.error(f"Frequency {freq_hz/1e6:.3f} MHz out of range")
            return False

        try:
            self._device.set_freq(int(freq_hz))
            self._state.frequency = freq_hz
            logger.debug(f"Set frequency to {freq_hz/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False

    def set_sample_rate(self, rate_hz: float) -> bool:
        """Set sample rate."""
        if not self._is_open or self._device is None or self._spec is None:
            return False

        # Clamp to valid range
        rate_hz = max(
            self._spec.sample_rate_min, min(rate_hz, self._spec.sample_rate_max)
        )

        try:
            self._device.set_sample_rate(rate_hz)
            self._state.sample_rate = rate_hz
            logger.debug(f"Set sample rate to {rate_hz/1e6:.3f} MS/s")
            return True
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}")
            return False

    def set_bandwidth(self, bw_hz: float) -> bool:
        """Set baseband filter bandwidth."""
        if not self._is_open or self._device is None:
            return False

        # HackRF has specific supported bandwidths
        supported_bw = [
            1.75e6,
            2.5e6,
            3.5e6,
            5e6,
            5.5e6,
            6e6,
            7e6,
            8e6,
            9e6,
            10e6,
            12e6,
            14e6,
            15e6,
            20e6,
            24e6,
            28e6,
        ]
        # Find nearest supported bandwidth
        bw_hz = min(supported_bw, key=lambda x: abs(x - bw_hz))

        try:
            self._device.set_baseband_filter_bandwidth(int(bw_hz))
            self._state.bandwidth = bw_hz
            logger.debug(f"Set bandwidth to {bw_hz/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set bandwidth: {e}")
            return False

    def set_gain(self, gain_db: float) -> bool:
        """Set combined RX gain (LNA + VGA)."""
        if not self._is_open or self._device is None:
            return False

        # Split gain between LNA and VGA
        # LNA: 0, 8, 16, 24, 32, 40 dB (discrete steps)
        # VGA: 0-62 dB in 2dB steps
        # Strategy: For best noise figure, maximize LNA first, then use VGA for remainder
        # For gains up to 40 dB: try to use LNA only if possible, otherwise add VGA
        # For gains above 40 dB: max LNA (40 dB) + VGA for remainder

        gain_db = max(0, min(102, gain_db))  # Clamp to valid range (LNA max 40 + VGA max 62)

        if gain_db <= 40:
            # For lower gains, find nearest LNA step and use VGA for fine adjustment
            lna_gain = min(self.LNA_GAIN_VALUES, key=lambda x: abs(x - gain_db))
            if lna_gain > gain_db:
                # LNA alone would be too much, try lower LNA + VGA
                lower_lna_values = [v for v in self.LNA_GAIN_VALUES if v <= gain_db]
                lna_gain = max(lower_lna_values) if lower_lna_values else 0
            vga_gain = int((gain_db - lna_gain) / 2) * 2  # Round to nearest 2 dB
            vga_gain = max(0, min(62, vga_gain))
        else:
            # For higher gains, max out LNA and use VGA
            lna_gain = 40
            remaining = gain_db - lna_gain
            vga_gain = int(remaining / 2) * 2  # Round to nearest 2 dB
            vga_gain = max(0, min(62, vga_gain))

        try:
            self._device.set_lna_gain(lna_gain)
            self._device.set_vga_gain(vga_gain)
            self._lna_gain = lna_gain
            self._vga_gain = vga_gain
            self._state.gain = lna_gain + vga_gain
            self._state.gain_mode = "manual"
            logger.debug(f"Set gain: LNA={lna_gain}dB, VGA={vga_gain}dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set gain: {e}")
            return False

    def set_lna_gain(self, gain_db: int) -> bool:
        """Set LNA gain directly."""
        if not self._is_open or self._device is None:
            return False

        gain_db = min(self.LNA_GAIN_VALUES, key=lambda x: abs(x - gain_db))
        try:
            self._device.set_lna_gain(gain_db)
            self._lna_gain = gain_db
            self._state.gain = self._lna_gain + self._vga_gain
            return True
        except Exception as e:
            logger.error(f"Failed to set LNA gain: {e}")
            return False

    def set_vga_gain(self, gain_db: int) -> bool:
        """Set VGA gain directly."""
        if not self._is_open or self._device is None:
            return False

        gain_db = min(62, max(0, gain_db))
        try:
            self._device.set_vga_gain(gain_db)
            self._vga_gain = gain_db
            self._state.gain = self._lna_gain + self._vga_gain
            return True
        except Exception as e:
            logger.error(f"Failed to set VGA gain: {e}")
            return False

    def set_gain_mode(self, auto: bool) -> bool:
        """HackRF doesn't have AGC, this is a no-op."""
        logger.warning("HackRF does not support automatic gain control")
        self._state.gain_mode = "manual"
        return True

    def set_amp(self, enabled: bool) -> bool:
        """Enable/disable RF amplifier (+14 dB)."""
        if not self._is_open or self._device is None:
            return False

        try:
            self._device.set_amp_enable(enabled)
            self._amp_enabled = enabled
            self._state.amp_enabled = enabled
            logger.info(f"RF amplifier {'enabled (+14dB)' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to set amp: {e}")
            return False

    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Start receiving samples."""
        if not self._is_open or self._device is None:
            return False

        if self._state.is_streaming:
            logger.warning("Already streaming RX")
            return True

        if self._state.is_transmitting:
            logger.error("Cannot RX while TX is active (half-duplex)")
            return False

        self._rx_callback = callback
        self._stop_event.clear()

        def rx_thread():
            """Background thread for receiving samples."""
            try:

                def rx_callback(hackrf_transfer):
                    """HackRF callback for received data."""
                    if self._stop_event.is_set():
                        return -1

                    # Convert to complex samples
                    data = hackrf_transfer.buffer
                    # HackRF uses signed 8-bit I/Q
                    iq = np.frombuffer(data, dtype=np.int8).astype(np.float32)
                    iq = iq.reshape(-1, 2)
                    samples = (iq[:, 0] + 1j * iq[:, 1]) / 128.0

                    if self._rx_callback:
                        self._rx_callback(samples)
                    else:
                        try:
                            self._sample_queue.put_nowait(samples)
                        except queue.Full:
                            pass  # Queue full, drop samples
                    return 0

                if self._device is not None:
                    self._device.start_rx(rx_callback)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"RX thread error: {e}")

        self._rx_thread = Thread(target=rx_thread, daemon=True)
        self._rx_thread.start()
        self._state.is_streaming = True
        logger.info("Started HackRF RX streaming")
        return True

    def stop_rx(self) -> bool:
        """Stop receiving samples."""
        if not self._state.is_streaming:
            return True

        self._stop_event.set()
        try:
            if self._device is not None:
                self._device.stop_rx()
        except Exception as e:
            logger.warning(f"Error stopping RX: {e}")

        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None

        self._state.is_streaming = False
        logger.info("Stopped HackRF RX streaming")
        return True

    def set_tx_gain(self, gain_db: float) -> bool:
        """Set transmit VGA gain."""
        if not self._is_open or self._device is None:
            return False

        gain_db = min(47, max(0, int(gain_db)))
        try:
            self._device.set_txvga_gain(gain_db)
            self._tx_vga_gain = gain_db
            logger.debug(f"Set TX VGA gain to {gain_db} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set TX gain: {e}")
            return False

    def start_tx(self, callback: Optional[Callable[[], np.ndarray]] = None) -> bool:
        """Start transmitting samples."""
        if not self._is_open or self._device is None:
            return False

        if self._state.is_transmitting:
            logger.warning("Already transmitting")
            return True

        if self._state.is_streaming:
            logger.error("Cannot TX while RX is active (half-duplex)")
            return False

        # SAFETY: Validate TX frequency against lockout bands
        allowed, reason = is_tx_allowed(self._state.frequency, self._state.bandwidth)
        if not allowed:
            logger.error(f"TX BLOCKED: {reason}")
            return False

        self._tx_callback = callback
        self._stop_event.clear()

        def tx_thread():
            """Background thread for transmitting samples."""
            try:

                def tx_callback(hackrf_transfer):
                    """HackRF callback for transmit data."""
                    if self._stop_event.is_set():
                        return -1

                    if self._tx_callback:
                        samples = self._tx_callback()
                        if samples is not None:
                            # Convert complex to signed 8-bit I/Q
                            iq = np.zeros(len(samples) * 2, dtype=np.int8)
                            iq[0::2] = (samples.real * 127).astype(np.int8)
                            iq[1::2] = (samples.imag * 127).astype(np.int8)
                            hackrf_transfer.buffer[: len(iq)] = iq
                    return 0

                if self._device is not None:
                    self._device.start_tx(tx_callback)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"TX thread error: {e}")

        self._tx_thread = Thread(target=tx_thread, daemon=True)
        self._tx_thread.start()
        self._state.is_transmitting = True
        logger.info("Started HackRF TX streaming")
        return True

    def stop_tx(self) -> bool:
        """Stop transmitting samples."""
        if not self._state.is_transmitting:
            return True

        self._stop_event.set()
        try:
            if self._device is not None:
                self._device.stop_tx()
        except Exception as e:
            logger.warning(f"Error stopping TX: {e}")

        if self._tx_thread:
            self._tx_thread.join(timeout=2.0)
            self._tx_thread = None

        self._state.is_transmitting = False
        logger.info("Stopped HackRF TX streaming")
        return True

    def write_samples(self, samples: np.ndarray) -> bool:
        """
        Write samples to transmit buffer.

        Note: For continuous TX, use start_tx with a callback.
        This method is for one-shot transmissions.
        """
        if not self._is_open or self._device is None:
            return False

        if self._state.is_streaming:
            logger.error("Cannot TX while RX is active")
            return False

        try:
            # Convert complex to signed 8-bit I/Q
            iq = np.zeros(len(samples) * 2, dtype=np.int8)
            iq[0::2] = (samples.real * 127).astype(np.int8)
            iq[1::2] = (samples.imag * 127).astype(np.int8)

            # This is a simplified implementation
            # Real implementation would need proper buffer management
            logger.debug(f"Writing {len(samples)} samples to TX")
            return True
        except Exception as e:
            logger.error(f"Failed to write samples: {e}")
            return False
