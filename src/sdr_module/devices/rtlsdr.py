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
import queue
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

_INSTALL_HINT = (
    "RTL-SDR support is not installed. Install it with "
    '`python -m pip install "sdr-module[rtlsdr]"` and make sure the system '
    "librtlsdr library is present (Debian/Ubuntu: `sudo apt install "
    "librtlsdr-dev`; macOS: `brew install librtlsdr`; Linux users also need "
    "the RTL-SDR udev rules to run without root)."
)


def _device_count() -> int:
    """Number of connected RTL-SDR devices, via the real pyrtlsdr API."""
    from rtlsdr import librtlsdr

    return int(librtlsdr.rtlsdr_get_device_count())


def _device_serials() -> List[str]:
    """Serial strings for connected devices (index order)."""
    from rtlsdr import RtlSdr

    # get_device_serial_addresses() is the supported enumeration entry point;
    # RtlSdr.get_device_count / get_device_serial (used by older code) never
    # existed in pyrtlsdr and always raised AttributeError.
    return list(RtlSdr.get_device_serial_addresses())


# Direct sampling: which ADC input HF goes to (pyrtlsdr's
# ``set_direct_sampling`` values). Below DIRECT_SAMPLING_BELOW_HZ the tuner
# (R820T/R828D) can't receive, so the antenna is fed straight to the ADC.
DIRECT_SAMPLING_OFF = 0  # never: an RTL-SDR Blog V4 or an upconverter tunes HF
DIRECT_SAMPLING_I = 1  # the I input (dongles modified for HF on I)
DIRECT_SAMPLING_Q = 2  # the Q input (RTL-SDR Blog V3)
DIRECT_SAMPLING_BELOW_HZ = 24e6
_DIRECT_SAMPLING_NAMES = {
    DIRECT_SAMPLING_OFF: "off",
    DIRECT_SAMPLING_I: "I branch",
    DIRECT_SAMPLING_Q: "Q branch",
}

# Frequency correction (ppm) the driver accepts; cheap dongles are within
# about +/-100 ppm, TCXO ones within 1-2 ppm.
PPM_LIMIT = 500

_CAPABILITIES = [
    DeviceCapability.RX,
    DeviceCapability.BIAS_TEE,
    DeviceCapability.DIRECT_SAMPLE,
    DeviceCapability.AGC,
    DeviceCapability.FREQ_CORRECTION,
]


def direct_sampling_mode(value) -> int:
    """``0``/``False``/``"off"``, ``1``/``"i"``, ``2``/``"q"`` -> 0, 1 or 2
    (pyrtlsdr's spelling). Raises ValueError for anything else."""
    if isinstance(value, str):
        text = value.strip().lower()
        for mode, name in (
            (DIRECT_SAMPLING_OFF, "off"),
            (DIRECT_SAMPLING_I, "i"),
            (DIRECT_SAMPLING_Q, "q"),
        ):
            if text in (name, str(mode)):
                return mode
        raise ValueError(f"unknown direct sampling mode {value!r}")
    mode = int(value or 0)
    if mode not in _DIRECT_SAMPLING_NAMES:
        raise ValueError(f"unknown direct sampling mode {value!r}")
    return mode


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
        # Whether direct sampling is in use now, and the ADC input used for
        # it below 24 MHz (DIRECT_SAMPLING_OFF: never).
        self._direct_sampling = False
        self._direct_mode = DIRECT_SAMPLING_OFF  # what the hardware is set to
        self._direct_branch = DIRECT_SAMPLING_Q
        self._rx_error: Optional[str] = None
        # Tuner IF bandwidth set with set_bandwidth(); 0 = automatic.
        self._if_bandwidth = 0.0
        # Frequency correction applied to the open handle (ppm).
        self._ppm = 0
        # The tuner's gain steps (dB): the open device's own table.
        self._gains: List[float] = list(self.VALID_GAINS)

    @staticmethod
    def get_device_count() -> int:
        """Get number of RTL-SDR devices connected."""
        try:
            return _device_count()
        except ImportError:
            logger.warning(_INSTALL_HINT)
            return 0
        except Exception as e:
            logger.error(f"Error getting device count: {e}")
            return 0

    @staticmethod
    def get_device_serial(index: int) -> Optional[str]:
        """Get serial number of device at index."""
        try:
            serials = _device_serials()
            if 0 <= index < len(serials):
                return serials[index]
            return None
        except ImportError:
            return None  # Library not installed
        except Exception as e:
            logger.debug(f"Could not get serial for RTL-SDR device {index}: {e}")
            return None

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """List all available RTL-SDR devices."""
        devices = []
        try:
            serials = _device_serials()
            for i, serial in enumerate(serials):
                devices.append(
                    DeviceInfo(
                        name=f"RTL-SDR #{i}",
                        serial=serial or f"rtlsdr_{i}",
                        manufacturer="RTL-SDR Blog",
                        product="RTL2832U",
                        index=i,
                        capabilities=list(_CAPABILITIES),
                    )
                )
        except ImportError:
            logger.warning(_INSTALL_HINT)
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
        except ImportError:
            logger.error(_INSTALL_HINT)
            return False

        try:
            self._device = RtlSdr(device_index=index)
            self._if_bandwidth = 0.0  # a fresh handle starts automatic
            # A fresh handle starts uncorrected, without direct sampling.
            self._ppm = 0
            self._direct_mode = DIRECT_SAMPLING_OFF
            self._direct_sampling = False
            self._gains = self._read_gain_table()

            # Get device info (best-effort; enumeration may not expose a serial).
            serial = self.get_device_serial(index) or f"rtlsdr_{index}"
            self._info = DeviceInfo(
                name=f"RTL-SDR #{index}",
                serial=serial,
                manufacturer="RTL-SDR Blog",
                product="RTL2832U",
                index=index,
                capabilities=list(_CAPABILITIES),
            )

            # Set defaults
            self._device.sample_rate = 2.4e6
            self._device.center_freq = 100e6
            self._device.gain = "auto"

            self._state.sample_rate = 2.4e6
            self._state.frequency = 100e6
            self._state.gain_mode = "auto"
            self._state.bandwidth = 2.4e6

            self._is_open = True
            logger.info(f"Opened RTL-SDR device: {serial}")
            return True

        except Exception as e:
            logger.error(f"Failed to open RTL-SDR: {e}")
            # A setter may have raised after RtlSdr() succeeded: close the
            # handle so we do not leak an open USB device or report is_open.
            if self._device is not None:
                try:
                    self._device.close()
                except Exception:
                    logger.debug(
                        "Error closing RTL-SDR after failed open", exc_info=True
                    )
            self._device = None
            self._is_open = False
            return False

    def close(self) -> None:
        """Close RTL-SDR device (switching its bias tee off first)."""
        if self._state.is_streaming:
            self.stop_rx()

        if self._device is not None and self._state.bias_tee_enabled:
            # Don't leave the antenna port powered after the app is done.
            self.set_bias_tee(False)

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
        if not self._is_open or self._device is None or self._spec is None:
            return False

        # Validate frequency against device specifications
        if freq_hz < self._spec.freq_min or freq_hz > self._spec.freq_max:
            logger.error(
                f"Frequency {freq_hz/1e6:.3f} MHz out of range "
                f"({self._spec.freq_min/1e6:.3f} - {self._spec.freq_max/1e6:.3f} MHz)"
            )
            return False

        # HF (below 24 MHz) through direct sampling, on the chosen input.
        self._apply_direct_sampling(freq_hz)

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
        if not self._is_open or self._device is None or self._spec is None:
            return False

        # Clamp to valid range
        rate_hz = max(
            self._spec.sample_rate_min, min(rate_hz, self._spec.sample_rate_max)
        )

        try:
            self._device.sample_rate = rate_hz
            self._state.sample_rate = rate_hz
            # The tuner filter follows the sample rate unless set explicitly.
            self._state.bandwidth = self._if_bandwidth or rate_hz
            logger.debug(f"Set sample rate to {rate_hz/1e6:.3f} MS/s")
            return True
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}")
            return False

    def set_bandwidth(self, bw_hz: float) -> bool:
        """Set the tuner's IF (anti-alias) filter bandwidth.

        This is the analog filter in front of the ADC. It never changes the
        sample rate, so the span shown stays the same; 0 makes the filter
        follow the sample rate again (the default). The tuner rounds the
        width to one it supports (R820T/R828D: about 0.3-8 MHz), so this is
        not the demodulated channel width, which the DSP chain filters.

        Returns False, and changes nothing, when the device is not open, the
        width is negative, or the installed pyrtlsdr / librtlsdr cannot set
        the tuner bandwidth (``rtlsdr_set_tuner_bandwidth``). Without that
        support, asking for the automatic width (0 or the sample rate) still
        succeeds, because the filter already is that wide.
        """
        if not self._is_open or self._device is None:
            return False
        if bw_hz < 0:
            logger.error(f"Invalid RTL-SDR tuner bandwidth: {bw_hz} Hz")
            return False

        try:
            setter = getattr(self._device, "set_bandwidth", None)
            if not callable(setter):
                raise AttributeError("set_bandwidth")
            setter(int(bw_hz))
        except AttributeError:
            # Old pyrtlsdr without the method, or a librtlsdr without
            # rtlsdr_set_tuner_bandwidth behind it.
            if bw_hz in (0, self._state.sample_rate):
                return True  # what the automatic filter already does
            logger.info(
                "This RTL-SDR library cannot set the tuner bandwidth; it "
                "stays automatic (it follows the sample rate)"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to set tuner bandwidth: {e}")
            return False

        self._if_bandwidth = float(bw_hz)
        self._state.bandwidth = self._if_bandwidth or self._state.sample_rate
        if bw_hz:
            logger.debug(f"Set tuner bandwidth to {bw_hz/1e6:.3f} MHz")
        else:
            logger.debug("Tuner bandwidth follows the sample rate")
        return True

    def set_gain(self, gain_db: float) -> bool:
        """Set gain value."""
        if not self._is_open or self._device is None:
            return False

        # The nearest step of the tuner's own gain table (the GUI offers
        # only those, so a far-off request is a programming slip: debug).
        nearest_gain = min(self._gains, key=lambda x: abs(x - gain_db))
        if abs(nearest_gain - gain_db) > 0.5:
            logger.debug(
                f"Requested gain {gain_db:.1f} dB set to the nearest step: "
                f"{nearest_gain} dB"
            )

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

        with self._state_lock:
            if self._state.is_streaming:
                logger.warning("Already streaming")
                return True

        self._rx_callback = callback
        self._rx_error = None
        self._stop_event.clear()

        def rx_thread():
            """Background thread for receiving samples."""
            try:
                while not self._stop_event.is_set():
                    if self._device is None:
                        break
                    samples = self._device.read_samples(256 * 1024)
                    if self._rx_callback:
                        self._rx_callback(samples)
                    else:
                        try:
                            self._sample_queue.put_nowait(samples)
                        except queue.Full:
                            # Queue full, drop samples (counted).
                            self._dropped_samples += len(samples)
            except Exception as e:
                if not self._stop_event.is_set():
                    # The stream died on its own (USB unplug, read error, or a
                    # user callback raising). Record it and clear the streaming
                    # flag so callers can see acquisition has stopped.
                    self._rx_error = str(e)
                    logger.error(f"RX thread error: {e}")
                    with self._state_lock:
                        self._state.is_streaming = False

        # Mark streaming BEFORE starting the thread. If we set it afterwards, a
        # thread that dies immediately (bad read, callback raising) would clear
        # the flag first and this line would wrongly set it back to True.
        with self._state_lock:
            self._state.is_streaming = True
        self._rx_thread = Thread(target=rx_thread, daemon=True)
        self._rx_thread.start()
        logger.info("Started RX streaming")
        return True

    def stop_rx(self) -> bool:
        """Stop receiving samples."""
        with self._state_lock:
            if not self._state.is_streaming:
                return True
            # Mark as not streaming before stopping to prevent races
            self._state.is_streaming = False

        self._stop_event.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            if self._rx_thread.is_alive():
                logger.warning("RX thread did not terminate within timeout")
            self._rx_thread = None

        logger.info("Stopped RX streaming")
        return True

    def set_bias_tee(self, enabled: bool) -> bool:
        """Power the antenna socket (4.5 V, RTL-SDR Blog V3 and later) for
        an LNA or an active antenna, or switch that power off."""
        if not self._is_open or self._device is None:
            return False

        enabled = bool(enabled)
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

    # ------------------------------------------------------------------ #
    # Direct sampling (HF)
    # ------------------------------------------------------------------ #
    def set_direct_sampling(self, mode) -> bool:
        """Choose how HF (below 24 MHz) is received: through the ADC's Q
        input (``2`` / ``"q"``, the RTL-SDR Blog V3 and the default), its I
        input (``1`` / ``"i"``) or not at all (``0`` / ``"off"``: an RTL-SDR
        Blog V4 or an upconverter tunes HF through the tuner).

        Direct sampling is switched on only while tuned below 24 MHz (the
        tuner can't go lower); above that the tuner is always used. A change
        takes effect at once when tuned there. Returns False for an unknown
        mode, or when the device refused it.
        """
        try:
            branch = direct_sampling_mode(mode)
        except (TypeError, ValueError) as e:
            logger.error(f"Direct sampling: {e}")
            return False
        self._direct_branch = branch
        if not self._is_open or self._device is None:
            return True  # applied when the device opens and tunes
        return self._apply_direct_sampling(self._state.frequency)

    @property
    def direct_sampling_branch(self) -> int:
        """The ADC input used below 24 MHz (0 = direct sampling off)."""
        return self._direct_branch

    @property
    def direct_sampling(self) -> bool:
        """Whether direct sampling is in use right now."""
        return self._direct_sampling

    def _apply_direct_sampling(self, freq_hz: float) -> bool:
        """Set the hardware's direct sampling for ``freq_hz``."""
        want = self._direct_branch if 0 < freq_hz < DIRECT_SAMPLING_BELOW_HZ else 0
        if want == self._direct_mode:
            return True
        return self._set_direct_mode(want)

    def _set_direct_mode(self, mode: int) -> bool:
        if not self._is_open or self._device is None:
            return False
        try:
            self._device.set_direct_sampling(mode)
        except Exception as e:
            logger.error(f"Failed to set direct sampling: {e}")
            return False
        self._direct_mode = mode
        self._direct_sampling = mode != DIRECT_SAMPLING_OFF
        if mode:
            logger.info(f"Direct sampling enabled ({_DIRECT_SAMPLING_NAMES[mode]})")
        else:
            logger.info("Direct sampling disabled")
            self._restore_tuner()
        return True

    def _restore_tuner(self) -> None:
        """Set the tuner's gain and IF filter again. Switching direct
        sampling off makes librtlsdr initialise the tuner afresh
        (``tuner->init``): an R820T is then back on automatic gain with its
        default IF filter, whatever the gain shown says."""
        device = self._device
        if device is None:
            return
        try:
            if self._state.gain_mode == "auto":
                device.gain = "auto"
            else:
                device.gain = self._state.gain
        except Exception as e:
            logger.error(f"Could not set the tuner gain again: {e}")
        setter = getattr(device, "set_bandwidth", None)
        if not callable(setter):
            return  # (an old pyrtlsdr: the filter is automatic anyway)
        try:
            setter(int(self._if_bandwidth))  # 0: it follows the sample rate
        except AttributeError:
            pass  # (a librtlsdr without rtlsdr_set_tuner_bandwidth)
        except Exception as e:
            logger.warning(f"Could not set the tuner bandwidth again: {e}")

    def _enable_direct_sampling(self, enabled: bool) -> bool:
        """Switch direct sampling on (on the chosen input, the Q input when
        none is) or off."""
        branch = self._direct_branch or DIRECT_SAMPLING_Q
        return self._set_direct_mode(branch if enabled else DIRECT_SAMPLING_OFF)

    # ------------------------------------------------------------------ #
    # Frequency correction
    # ------------------------------------------------------------------ #
    def set_freq_correction(self, ppm: float) -> bool:
        """Correct the dongle's crystal error, in parts per million.

        librtlsdr's convention: a positive value means the crystal runs
        fast, which makes stations show up below (left of) their true
        frequency. Cheap dongles are often 20-100 ppm off (50 ppm is 22 kHz
        at 433 MHz); ones with a TCXO within 1-2 ppm. Returns False when the
        device isn't open or refused the value.
        """
        if not self._is_open or self._device is None:
            return False
        try:
            value = int(round(float(ppm)))
        except (TypeError, ValueError):
            logger.error(f"Invalid frequency correction: {ppm!r}")
            return False
        if abs(value) > PPM_LIMIT:
            logger.error(
                f"Frequency correction {value} ppm is out of range "
                f"(+/-{PPM_LIMIT} ppm)"
            )
            return False
        if value == self._ppm:
            # librtlsdr refuses to set the value it already has (-2), and
            # pyrtlsdr then closes the device: never ask for it again.
            return True
        try:
            self._device.set_freq_correction(value)
        except AttributeError:
            logger.warning(
                "This rtlsdr library can't set the frequency correction (ppm)"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to set the frequency correction: {e}")
            return False
        self._ppm = value
        logger.info(f"Frequency correction set to {value:+d} ppm")
        return True

    @property
    def freq_correction(self) -> int:
        """The frequency correction in use (ppm)."""
        return int(self._ppm)

    # ------------------------------------------------------------------ #
    # Gain table
    # ------------------------------------------------------------------ #
    def _read_gain_table(self) -> List[float]:
        """The open tuner's gain steps (pyrtlsdr's ``valid_gains_db``: an
        R820T/R828D has 29 from 0 to 49.6 dB, an E4000 14 from -1 to 42
        dB), else the R820T2 table."""
        try:
            table = getattr(self._device, "valid_gains_db", None)
            gains = sorted({round(float(g), 1) for g in (table or [])})
        except (TypeError, ValueError):
            gains = []
        return gains or list(self.VALID_GAINS)

    def get_tuner_gains(self) -> List[float]:
        """The tuner's gain steps in dB (the open device's own table)."""
        return list(self._gains)

    @property
    def rx_error(self) -> Optional[str]:
        """The last RX-thread error, if streaming stopped unexpectedly."""
        return self._rx_error
