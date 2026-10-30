"""
HackRF One device driver wrapper.

Provides an interface for the HackRF One from Great Scott Gadgets, built on
the ``python_hackrf`` binding (``pip install "sdr-module[hackrf]"``). That
binding is a thin wrapper over the system ``libhackrf`` library, so the host
also needs libhackrf installed (Debian/Ubuntu: ``libhackrf-dev`` and a C
compiler; macOS: ``brew install hackrf``).

Specifications:
    - Frequency: 1 MHz - 6 GHz
    - Bandwidth: up to 20 MHz (baseband filter)
    - Sample Rate: 2-20 MS/s
    - ADC/DAC: 8-bit (MAX5864)
    - Half-duplex TX/RX
    - TX Power: -10 to +15 dBm (frequency dependent)

Threading model:
    ``python_hackrf`` runs the RX/TX transfer loop on its own USB thread and
    invokes the registered callback from there; ``pyhackrf_start_rx`` /
    ``pyhackrf_start_tx`` return immediately. This driver therefore does not
    spawn its own streaming thread. Callbacks never raise: an error inside a
    callback is logged once, marks the stream stopped, and returns a non-zero
    code so libhackrf tears the transfer down.
"""

import logging
import queue
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.frequency_manager import is_tx_allowed
from .base import (
    DeviceCapability,
    DeviceInfo,
    DeviceSpec,
    SDRDevice,
)

logger = logging.getLogger(__name__)

# HackRF specifications from the spec sheet.
HACKRF_SPEC = DeviceSpec(
    freq_min=1e6,  # 1 MHz
    freq_max=6e9,  # 6 GHz
    sample_rate_min=2e6,  # 2 MS/s (libhackrf accepts down to ~2 MS/s)
    sample_rate_max=20e6,  # 20 MS/s max
    bandwidth_max=28e6,  # widest baseband filter
    adc_bits=8,
    gain_min=0.0,
    gain_max=116.0,  # LNA (0-40) + VGA (0-62) + amp (14)
    max_input_power=-5.0,  # -5 dBm max input (CAUTION!)
    tx_power_min=-10.0,  # -10 dBm
    tx_power_max=15.0,  # +15 dBm (frequency dependent)
)

_INSTALL_HINT = (
    "HackRF support is not installed. Install it with "
    '`python -m pip install "sdr-module[hackrf]"` and make sure the system '
    "libhackrf library is present (Debian/Ubuntu: `sudo apt install "
    "libhackrf-dev` plus a C compiler; macOS: `brew install hackrf`)."
)

# What a HackRF One has: no automatic gain (no AGC), antenna port power
# (the "bias tee", 3.3 V at most 50 mA).
_CAPABILITIES = [
    DeviceCapability.RX,
    DeviceCapability.TX,
    DeviceCapability.HALF_DUPLEX,
    DeviceCapability.EXT_CLOCK,
    DeviceCapability.BIAS_TEE,
]

# libhackrf must be initialised once per process before any device is opened.
_init_lock = threading.Lock()
_initialized = False


def _load_backend():
    """Import the python_hackrf backend, or return None if unavailable."""
    try:
        from python_hackrf import pyhackrf
    except ImportError:
        return None
    return pyhackrf


def _ensure_initialized(pyhackrf) -> None:
    """Call pyhackrf_init() exactly once per process (thread-safe)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if not _initialized:
            pyhackrf.pyhackrf_init()
            _initialized = True


class HackRFDevice(SDRDevice):
    """
    HackRF One device driver (RX and half-duplex TX).

    Wraps ``python_hackrf`` for the HackRF One transceiver.
    """

    # Gain stages.
    LNA_GAIN_VALUES = list(range(0, 41, 8))  # 0, 8, 16, 24, 32, 40 dB
    VGA_GAIN_VALUES = list(range(0, 63, 2))  # 0-62 dB in 2 dB steps
    TX_VGA_GAIN_VALUES = list(range(0, 48, 1))  # 0-47 dB
    # The front-end amplifier's gain (RX and TX), nominal.
    AMP_GAIN_DB = 14.0

    # Supported baseband filter bandwidths (Hz).
    SUPPORTED_BANDWIDTHS = [
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

    def __init__(self) -> None:
        super().__init__()
        self._device: Any = None
        self._spec = HACKRF_SPEC
        self._lna_gain = 16
        self._vga_gain = 20
        self._tx_vga_gain = 20
        self._amp_enabled = False
        # Antenna port power while receiving (the firmware switches it off
        # whenever the HackRF goes idle, so start_rx switches it on again).
        self._bias_tee = False
        self._tx_callback: Optional[Callable[[], Optional[np.ndarray]]] = None
        # Leftover int8 I/Q bytes not yet handed to a TX transfer.
        self._tx_leftover: np.ndarray = np.empty(0, dtype=np.int8)
        self._tx_done = threading.Event()
        self._rx_error: Optional[str] = None
        # The emission being transmitted: its occupied bandwidth (None: not
        # given, so the baseband filter's width is checked) and mode(s).
        # The legality checks during a transmission use these.
        self._tx_bandwidth: Optional[float] = None
        self._tx_modes: Tuple[str, ...] = ()
        # Why the last transmission was refused or failed (None: it wasn't).
        self._last_tx_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Enumeration
    # ------------------------------------------------------------------ #
    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        """List connected HackRF devices without opening them."""
        pyhackrf = _load_backend()
        if pyhackrf is None:
            logger.warning(_INSTALL_HINT)
            return []

        devices: List[DeviceInfo] = []
        try:
            _ensure_initialized(pyhackrf)
            device_list = pyhackrf.pyhackrf_device_list()
            count = getattr(device_list, "device_count", 0) or 0
            serials = list(getattr(device_list, "serial_numbers", []) or [])
            for i in range(count):
                serial = serials[i] if i < len(serials) else f"hackrf_{i}"
                devices.append(
                    DeviceInfo(
                        name="HackRF One" if count == 1 else f"HackRF One #{i}",
                        serial=serial or f"hackrf_{i}",
                        manufacturer="Great Scott Gadgets",
                        product="HackRF One",
                        index=i,
                        capabilities=list(_CAPABILITIES),
                    )
                )
        except Exception as e:
            logger.debug(f"HackRF enumeration failed: {e}")
        return devices

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self, index: int = 0) -> bool:
        """Open a HackRF device by index."""
        if self._is_open:
            logger.warning("Device already open")
            return True

        pyhackrf = _load_backend()
        if pyhackrf is None:
            logger.error(_INSTALL_HINT)
            return False

        try:
            _ensure_initialized(pyhackrf)
            if index == 0:
                device = pyhackrf.pyhackrf_open()
            else:
                device_list = pyhackrf.pyhackrf_device_list()
                device = pyhackrf.pyhackrf_device_list_open(device_list, index)
            if device is None:
                logger.error(f"No HackRF device found at index {index}")
                return False

            self._device = device
            self._is_open = True

            serial = "unknown"
            try:
                serial = self._device.pyhackrf_serialno_read()
            except Exception as e:
                logger.debug(f"Could not read HackRF serial number: {e}")

            self._info = DeviceInfo(
                name="HackRF One",
                serial=serial,
                manufacturer="Great Scott Gadgets",
                product="HackRF One",
                index=index,
                capabilities=list(_CAPABILITIES),
            )

            # Apply defaults. If any of these fail, the open has not really
            # succeeded, so tear the handle back down rather than leaving a
            # half-configured device claiming to be open.
            ok = (
                self.set_sample_rate(10e6)
                and self.set_frequency(100e6)
                and self.set_bandwidth(10e6)
                and self.set_gain(30)
            )
            if not ok:
                logger.error("HackRF opened but failed to apply default settings")
                self.close()
                return False

            logger.info(f"Opened HackRF device: {serial}")
            return True

        except Exception as e:
            logger.error(f"Failed to open HackRF: {e}")
            # Do not leak a half-open handle.
            try:
                if self._device is not None:
                    self._device.pyhackrf_close()
            except Exception:
                logger.debug("Error closing HackRF after failed open", exc_info=True)
            self._device = None
            self._is_open = False
            return False

    def close(self) -> None:
        """Close the HackRF device and release resources."""
        if self._state.is_streaming:
            self.stop_rx()
        if self._state.is_transmitting:
            self.stop_tx()
        if self._device is not None and self._bias_tee:
            self.set_bias_tee(False)  # don't leave the antenna port powered

        if self._device is not None:
            try:
                self._device.pyhackrf_close()
            except Exception as e:
                logger.error(f"Error closing device: {e}")
            finally:
                self._device = None
                self._is_open = False
                logger.info("HackRF device closed")

    # ------------------------------------------------------------------ #
    # Tuning / gain
    # ------------------------------------------------------------------ #
    def set_frequency(self, freq_hz: float) -> bool:
        """Set the center frequency."""
        if not self._is_open or self._device is None or self._spec is None:
            return False

        if freq_hz < self._spec.freq_min or freq_hz > self._spec.freq_max:
            logger.error(f"Frequency {freq_hz/1e6:.3f} MHz out of range")
            return False

        # SAFETY: while transmitting, a retune must not cross into a locked band.
        if self._state.is_transmitting:
            allowed, reason = self._check_tx(freq_hz)
            if not allowed:
                logger.error(f"TX frequency change BLOCKED: {reason}")
                return False

        try:
            self._device.pyhackrf_set_freq(int(freq_hz))
            self._state.frequency = freq_hz
            logger.debug(f"Set frequency to {freq_hz/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False

    def set_sample_rate(self, rate_hz: float) -> bool:
        """Set the sample rate."""
        if not self._is_open or self._device is None or self._spec is None:
            return False

        rate_hz = max(
            self._spec.sample_rate_min, min(rate_hz, self._spec.sample_rate_max)
        )

        try:
            self._device.pyhackrf_set_sample_rate(float(rate_hz))
            self._state.sample_rate = rate_hz
            logger.debug(f"Set sample rate to {rate_hz/1e6:.3f} MS/s")
            return True
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}")
            return False

    def set_bandwidth(self, bw_hz: float) -> bool:
        """Set the baseband filter bandwidth (rounded to a supported value)."""
        if not self._is_open or self._device is None:
            return False

        bw_hz = min(self.SUPPORTED_BANDWIDTHS, key=lambda x: abs(x - bw_hz))

        # SAFETY: while transmitting, a wider filter must not straddle a lock
        # (when the emission's own bandwidth is unknown, the filter is the
        # widest it can be).
        if self._state.is_transmitting:
            allowed, reason = self._check_tx(self._state.frequency, bw_hz)
            if not allowed:
                logger.error(f"TX bandwidth change BLOCKED: {reason}")
                return False

        try:
            pyhackrf = _load_backend()
            actual = int(bw_hz)
            if pyhackrf is not None and hasattr(
                pyhackrf, "pyhackrf_compute_baseband_filter_bw"
            ):
                actual = int(pyhackrf.pyhackrf_compute_baseband_filter_bw(int(bw_hz)))
            self._device.pyhackrf_set_baseband_filter_bandwidth(actual)
            self._state.bandwidth = float(actual)
            logger.debug(f"Set bandwidth to {actual/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set bandwidth: {e}")
            return False

    def set_gain(self, gain_db: float) -> bool:
        """Set the combined RX gain, split across the LNA and VGA stages."""
        if not self._is_open or self._device is None:
            return False

        # LNA: 0-40 dB in 8 dB steps; VGA: 0-62 dB in 2 dB steps.
        gain_db = max(0.0, min(102.0, gain_db))
        if gain_db <= 40:
            lower = [v for v in self.LNA_GAIN_VALUES if v <= gain_db]
            lna_gain = max(lower) if lower else 0
        else:
            lna_gain = 40
        vga_gain = int((gain_db - lna_gain) / 2) * 2
        vga_gain = max(0, min(62, vga_gain))

        try:
            self._device.pyhackrf_set_lna_gain(lna_gain)
            self._device.pyhackrf_set_vga_gain(vga_gain)
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
        """Set the LNA gain directly (rounded to a valid 8 dB step)."""
        if not self._is_open or self._device is None:
            return False

        gain_db = min(self.LNA_GAIN_VALUES, key=lambda x: abs(x - gain_db))
        try:
            self._device.pyhackrf_set_lna_gain(gain_db)
            self._lna_gain = gain_db
            self._state.gain = self._lna_gain + self._vga_gain
            return True
        except Exception as e:
            logger.error(f"Failed to set LNA gain: {e}")
            return False

    def set_vga_gain(self, gain_db: int) -> bool:
        """Set the VGA gain directly (0-62 dB, rounded to a 2 dB step)."""
        if not self._is_open or self._device is None:
            return False

        gain_db = max(0, min(62, int(gain_db)))
        gain_db -= gain_db % 2
        try:
            self._device.pyhackrf_set_vga_gain(gain_db)
            self._vga_gain = gain_db
            self._state.gain = self._lna_gain + self._vga_gain
            return True
        except Exception as e:
            logger.error(f"Failed to set VGA gain: {e}")
            return False

    def set_gain_mode(self, auto: bool) -> bool:
        """The HackRF One has no automatic gain (AGC): the gain is always
        set by hand (LNA and VGA), so asking for automatic gain is refused
        (False) and the gain stays as it was."""
        if not self._is_open or self._device is None:
            return False
        self._state.gain_mode = "manual"
        if auto:
            logger.info(
                "The HackRF One has no automatic gain; set the LNA and VGA "
                "gains by hand"
            )
            return False
        return True

    @property
    def lna_gain(self) -> int:
        """The RX LNA ("IF") gain in use: 0-40 dB in 8 dB steps."""
        return int(self._lna_gain)

    @property
    def vga_gain(self) -> int:
        """The RX VGA (baseband) gain in use: 0-62 dB in 2 dB steps."""
        return int(self._vga_gain)

    def set_bias_tee(self, enabled: bool) -> bool:
        """Power the antenna port (3.3 V, at most 50 mA) for an LNA or an
        active antenna while receiving, or switch that power off.

        The HackRF's firmware switches it off whenever the HackRF stops
        receiving or transmitting, so :meth:`start_rx` switches it on again;
        it stays off while transmitting (an LNA must not be fed the
        transmitter's power: :meth:`start_tx` switches it off), and
        switching it on mid-transmission is refused.
        """
        if not self._is_open or self._device is None:
            return False
        enabled = bool(enabled)
        if enabled and self._state.is_transmitting:
            logger.error("Antenna power refused: the HackRF is transmitting")
            return False
        try:
            self._device.pyhackrf_set_antenna_enable(enabled)
        except AttributeError:
            logger.warning(
                "This python_hackrf can't switch the antenna port power "
                "(pyhackrf_set_antenna_enable)"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to set the antenna port power: {e}")
            return False
        self._bias_tee = enabled
        self._state.bias_tee_enabled = enabled
        logger.info(f"Antenna port power {'on' if enabled else 'off'}")
        return True

    def set_amp(self, enabled: bool) -> bool:
        """Enable/disable the +14 dB RF front-end amplifier."""
        if not self._is_open or self._device is None:
            return False

        # SAFETY: the +14 dB amp raises TX ERP. Refuse to enable it mid-transmit
        # unless the current frequency/bandwidth is transmit-legal.
        if enabled and self._state.is_transmitting:
            allowed, reason = self._check_tx(self._state.frequency)
            if not allowed:
                logger.error(f"TX amplifier enable BLOCKED: {reason}")
                return False

        try:
            self._device.pyhackrf_set_amp_enable(bool(enabled))
            self._amp_enabled = enabled
            self._state.amp_enabled = enabled
            logger.info(f"RF amplifier {'enabled (+14dB)' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to set amp: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Receive
    # ------------------------------------------------------------------ #
    def start_rx(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """Start receiving samples (non-blocking; libhackrf drives the loop)."""
        if not self._is_open or self._device is None:
            return False

        with self._state_lock:
            if self._state.is_streaming:
                logger.warning("Already streaming RX")
                return True
            if self._state.is_transmitting:
                logger.error("Cannot RX while TX is active (half-duplex)")
                return False

        self._rx_callback = callback
        self._rx_error = None
        self._stop_event.clear()

        def rx_callback(device, buffer, buffer_length, valid_length) -> int:
            """Called by libhackrf's USB thread with int8 I/Q data."""
            if self._stop_event.is_set():
                return -1
            try:
                n = int(valid_length)
                if n <= 0:
                    return 0
                raw = np.asarray(buffer[:n], dtype=np.int8)
                # Drop a trailing odd byte so I/Q stays paired.
                if raw.size % 2:
                    raw = raw[:-1]
                iq = raw.astype(np.float32).reshape(-1, 2)
                samples = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64) / 128.0
                if self._rx_callback:
                    self._rx_callback(samples)
                else:
                    try:
                        self._sample_queue.put_nowait(samples)
                    except queue.Full:
                        # Consumer too slow; drop this transfer (counted).
                        self._dropped_samples += len(samples)
                return 0
            except Exception as e:
                # Never raise out of a C callback: log once and stop the stream.
                self._rx_error = str(e)
                logger.error(f"HackRF RX callback error: {e}")
                with self._state_lock:
                    self._state.is_streaming = False
                return -1

        # The firmware bypassed the RF amp, and switched the antenna power
        # off, when the HackRF last went idle (stop_rx/stop_tx: OFF mode),
        # and never puts them back itself: set both again for this
        # reception.
        self._send_amp()
        if self._bias_tee:
            try:
                self._device.pyhackrf_set_antenna_enable(True)
            except Exception as e:
                logger.warning(f"Could not switch the antenna port power on: {e}")
        try:
            self._device.set_rx_callback(rx_callback)
            self._device.pyhackrf_start_rx()
        except Exception as e:
            logger.error(f"Failed to start RX: {e}")
            return False

        with self._state_lock:
            self._state.is_streaming = True
        logger.info("Started HackRF RX streaming")
        return True

    def stop_rx(self) -> bool:
        """Stop receiving samples."""
        with self._state_lock:
            if not self._state.is_streaming:
                return True
            self._state.is_streaming = False

        self._stop_event.set()
        try:
            if self._device is not None:
                self._device.pyhackrf_stop_rx()
        except Exception as e:
            logger.warning(f"Error stopping RX: {e}")

        logger.info("Stopped HackRF RX streaming")
        return True

    @property
    def rx_error(self) -> Optional[str]:
        """The last RX callback error, if the stream died unexpectedly."""
        return self._rx_error

    def _send_amp(self) -> None:
        """Send the RF amp setting again before a stream starts (the
        firmware bypasses the amp whenever the HackRF goes idle)."""
        try:
            self._device.pyhackrf_set_amp_enable(bool(self._amp_enabled))
        except Exception as e:
            logger.warning(f"Could not set the RF amplifier again: {e}")

    def _antenna_power_off_for_tx(self) -> bool:
        """Switch the antenna port power off before transmitting. The
        firmware does that only when a stream stops, not when the HackRF
        goes from idle straight to transmitting, so a bias tee switched on
        while idle would stay powered on the air. False when it is on and
        couldn't be switched off (``_bias_tee`` stays as it is: start_rx
        powers the port again)."""
        try:
            self._device.pyhackrf_set_antenna_enable(False)
        except AttributeError:
            pass  # (a python_hackrf that can't switch it never powered it)
        except Exception as e:
            if self._bias_tee:
                logger.error(f"Could not switch the antenna port power off: {e}")
                return False
            logger.debug(f"Could not switch the antenna port power off: {e}")
        return True

    # ------------------------------------------------------------------ #
    # Transmit
    # ------------------------------------------------------------------ #
    def set_tx_gain(self, gain_db: float) -> bool:
        """Set the transmit VGA gain (0-47 dB)."""
        if not self._is_open or self._device is None:
            return False

        gain_db = min(47, max(0, int(gain_db)))
        try:
            self._device.pyhackrf_set_txvga_gain(gain_db)
            self._tx_vga_gain = gain_db
            logger.debug(f"Set TX VGA gain to {gain_db} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set TX gain: {e}")
            return False

    @property
    def tx_gain(self) -> int:
        """The transmit VGA gain last set (0-47 dB)."""
        return int(self._tx_vga_gain)

    @property
    def amp_enabled(self) -> bool:
        """Whether the +14 dB front-end amplifier is on."""
        return bool(self._amp_enabled)

    @property
    def last_tx_error(self) -> Optional[str]:
        """Why the last transmission was refused or failed (None if it
        wasn't), e.g. the frequency manager's ``"LICENSE: ..."`` reason."""
        return self._last_tx_error

    @staticmethod
    def _mode_list(mode: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
        """The emission's mode(s) as a tuple (``""`` for "not given")."""
        if mode is None:
            return ("",)
        if isinstance(mode, str):
            return (mode,)
        modes = tuple(str(m) for m in mode)
        return modes or ("",)

    def _check_tx(
        self, freq_hz: float, filter_hz: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Whether sending the current emission at ``freq_hz`` is allowed.

        With the emission's occupied bandwidth and mode(s) given to
        :meth:`start_tx` / :meth:`write_samples`, those are checked (each
        mode must be allowed). Otherwise (a caller that didn't say) the
        baseband filter's width, ``filter_hz`` or the current one, stands in
        as the widest the signal can be, as before.
        """
        if self._tx_bandwidth is None:
            width = self._state.bandwidth if filter_hz is None else filter_hz
            return is_tx_allowed(freq_hz, width)
        for mode in self._tx_modes or ("",):
            allowed, reason = is_tx_allowed(freq_hz, self._tx_bandwidth, mode)
            if not allowed:
                return False, reason
        return True, None

    @staticmethod
    def _to_int8_iq(samples: np.ndarray) -> np.ndarray:
        """Convert complex samples to clipped, interleaved int8 I/Q bytes."""
        samples = np.asarray(samples)
        real = np.clip(samples.real, -1.0, 1.0)
        imag = np.clip(samples.imag, -1.0, 1.0)
        iq = np.empty(samples.size * 2, dtype=np.int8)
        iq[0::2] = np.round(real * 127.0).astype(np.int8)
        iq[1::2] = np.round(imag * 127.0).astype(np.int8)
        return iq

    def start_tx(
        self,
        callback: Optional[Callable[[], Optional[np.ndarray]]] = None,
        bandwidth_hz: Optional[float] = None,
        mode: Union[str, Sequence[str], None] = "",
    ) -> bool:
        """
        Start transmitting.

        ``callback`` is called (with no arguments) each time libhackrf needs
        more data and must return the next block of complex samples, or an
        empty array / None to end the transmission. ``callback`` is required;
        ``start_tx(None)`` is refused so the transmitter never radiates the
        contents of an uninitialised buffer.

        ``bandwidth_hz`` and ``mode`` describe the emission: its occupied
        bandwidth (e.g. 6.4 kHz for an FM tone ID, 150 Hz for CW) and its
        mode ("CW", "FM", "SSB", ...; a sequence when the transmission mixes
        modes, e.g. an FM message with a CW ID: each must be allowed). The
        legality check (lockouts and license privileges) uses them, here and
        whenever the frequency, filter or amplifier changes while
        transmitting. Without ``bandwidth_hz`` the baseband filter's width is
        checked instead (the widest the signal can be), which refuses
        anything near a band edge. Why a start was refused is in
        :attr:`last_tx_error`.
        """
        if not self._is_open or self._device is None:
            self._last_tx_error = "The HackRF is not open"
            return False

        if callback is None:
            logger.error("start_tx requires a sample-producing callback")
            self._last_tx_error = "No samples to transmit"
            return False

        with self._state_lock:
            if self._state.is_transmitting:
                logger.warning("Already transmitting")
                return True
            if self._state.is_streaming:
                logger.error("Cannot TX while RX is active (half-duplex)")
                self._last_tx_error = (
                    "The HackRF is receiving; it can't transmit at the same time"
                )
                return False
            freq = self._state.frequency

        # SAFETY: validate the emission against the lockout bands and the
        # license (its own bandwidth and mode, when given).
        self._tx_bandwidth = (
            None if bandwidth_hz is None else max(0.0, float(bandwidth_hz))
        )
        self._tx_modes = self._mode_list(mode)
        allowed, reason = self._check_tx(freq)
        if not allowed:
            logger.error(f"TX BLOCKED: {reason}")
            self._last_tx_error = reason or "Transmitting is not allowed here"
            self._tx_bandwidth, self._tx_modes = None, ()
            return False

        # SAFETY: the antenna port is never powered on the air (an LNA or
        # active antenna there must not get the transmitter's power).
        if not self._antenna_power_off_for_tx():
            self._last_tx_error = (
                "The antenna port power (bias tee) couldn't be switched off, so "
                "the HackRF didn't transmit"
            )
            self._tx_bandwidth, self._tx_modes = None, ()
            return False
        # The amp as it is set (the firmware bypassed it when the HackRF
        # last went idle).
        self._send_amp()
        self._last_tx_error = None

        self._tx_callback = callback
        self._tx_leftover = np.empty(0, dtype=np.int8)
        self._tx_done.clear()
        self._stop_event.clear()

        def tx_callback(device, buffer, buffer_length, valid_length) -> int:
            """Fill ``buffer`` with up to ``buffer_length`` int8 I/Q bytes."""
            if self._stop_event.is_set():
                valid_length.value = 0
                self._tx_done.set()
                return -1
            try:
                capacity = int(buffer_length)
                out = self._tx_leftover
                # Pull fresh blocks until we have enough or the source is spent.
                while out.size < capacity:
                    block = self._tx_callback() if self._tx_callback else None
                    if block is None or len(block) == 0:
                        break
                    out = np.concatenate([out, self._to_int8_iq(block)])
                if out.size == 0:
                    valid_length.value = 0
                    self._tx_done.set()
                    return 1  # Nothing left to send: stop.
                n = min(out.size, capacity)
                buffer[:n] = out[:n]
                self._tx_leftover = out[n:]
                valid_length.value = n
                return 0
            except Exception as e:
                logger.error(f"HackRF TX callback error: {e}")
                self._last_tx_error = f"Transmit error: {e}"
                valid_length.value = 0
                self._tx_done.set()
                return -1

        try:
            self._device.set_tx_callback(tx_callback)
            self._device.pyhackrf_start_tx()
        except Exception as e:
            logger.error(f"Failed to start TX: {e}")
            self._last_tx_error = f"The HackRF didn't start transmitting: {e}"
            self._tx_bandwidth, self._tx_modes = None, ()
            return False

        with self._state_lock:
            self._state.is_transmitting = True
        logger.info("Started HackRF TX streaming")
        return True

    def wait_tx_done(self, timeout: Optional[float] = None) -> bool:
        """Wait until the transmission has handed its last samples to the
        hardware (the callback returned no more) or was stopped. True when
        that happened within ``timeout`` seconds."""
        return self._tx_done.wait(timeout=timeout)

    def stop_tx(self) -> bool:
        """Stop transmitting samples."""
        with self._state_lock:
            if not self._state.is_transmitting:
                return True
            self._state.is_transmitting = False

        self._stop_event.set()
        self._tx_done.set()
        try:
            if self._device is not None:
                self._device.pyhackrf_stop_tx()
        except Exception as e:
            logger.warning(f"Error stopping TX: {e}")
        self._tx_bandwidth, self._tx_modes = None, ()

        logger.info("Stopped HackRF TX streaming")
        return True

    def write_samples(
        self,
        samples: np.ndarray,
        timeout: float = 10.0,
        bandwidth_hz: Optional[float] = None,
        mode: Union[str, Sequence[str], None] = "",
    ) -> bool:
        """
        Transmit a fixed block of complex samples once, blocking until done.

        Returns True only after the samples have been handed to libhackrf and
        transmission has stopped. Returns False if TX is not currently allowed
        (device closed, RX active, or a frequency/bandwidth lockout; the
        reason is in :attr:`last_tx_error`).

        ``bandwidth_hz`` and ``mode`` describe the emission, as for
        :meth:`start_tx`. ``timeout`` is the extra time allowed beyond the
        samples' own air time at the current sample rate, so a long block is
        never cut short. (The GUI streams through
        :class:`sdr_module.gui.transmitter.Transmitter` instead, which does
        not block.)
        """
        if not self._is_open or self._device is None:
            self._last_tx_error = "The HackRF is not open"
            return False

        samples = np.asarray(samples)
        if samples.size == 0:
            return True

        with self._state_lock:
            if self._state.is_streaming:
                logger.error("Cannot TX while RX is active (half-duplex)")
                self._last_tx_error = (
                    "The HackRF is receiving; it can't transmit at the same time"
                )
                return False
            if self._state.is_transmitting:
                logger.error("Already transmitting")
                self._last_tx_error = "The HackRF is already transmitting"
                return False
            rate = self._state.sample_rate

        sent = threading.Event()
        state = {"done": False}

        def one_shot() -> Optional[np.ndarray]:
            if state["done"]:
                sent.set()
                return None
            state["done"] = True
            return samples

        if not self.start_tx(one_shot, bandwidth_hz=bandwidth_hz, mode=mode):
            return False

        # Wait for the source to be fully consumed, then for the transfer to
        # end: the samples' air time plus the margin.
        air_time = samples.size / rate if rate and rate > 0 else 0.0
        limit = air_time + max(0.0, float(timeout))
        sent.wait(timeout=limit)
        finished = self._tx_done.wait(timeout=limit)
        self.stop_tx()
        if not finished:
            logger.warning("write_samples timed out waiting for TX to drain")
            self._last_tx_error = "The HackRF stopped taking samples (timed out)"
        return finished
