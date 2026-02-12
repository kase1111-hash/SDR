"""
Configuration management for SDR module.

Handles device configuration, DSP settings, and persistence.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration values are invalid."""

    pass


@dataclass
class DeviceConfig:
    """Configuration for a single SDR device."""

    device_type: str = "rtlsdr"  # "rtlsdr" or "hackrf"
    device_index: int = 0
    frequency: float = 100e6  # 100 MHz default
    sample_rate: float = 2.4e6  # 2.4 MS/s default
    bandwidth: float = 2.4e6
    gain: float = 30.0
    gain_mode: str = "manual"  # "auto" or "manual"
    bias_tee: bool = False
    amp_enabled: bool = False
    # HackRF specific
    lna_gain: float = 16.0
    vga_gain: float = 20.0
    tx_vga_gain: float = 20.0

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields."""
        if self.device_type not in ("rtlsdr", "hackrf"):
            raise ConfigValidationError(
                f"Invalid device_type: {self.device_type}. "
                "Must be 'rtlsdr' or 'hackrf'"
            )
        if self.device_index < 0:
            raise ConfigValidationError(
                f"device_index must be non-negative, got {self.device_index}"
            )
        if not (1 <= self.frequency <= 30e9):
            raise ConfigValidationError(
                f"frequency must be between 1 Hz and 30 GHz, got {self.frequency}"
            )
        if not (1 <= self.sample_rate <= 100e6):
            raise ConfigValidationError(
                f"sample_rate must be between 1 and 100 MS/s, got {self.sample_rate}"
            )
        if self.bandwidth <= 0:
            raise ConfigValidationError(
                f"bandwidth must be positive, got {self.bandwidth}"
            )
        if not (-20 <= self.gain <= 100):
            raise ConfigValidationError(
                f"gain must be between -20 and 100 dB, got {self.gain}"
            )
        if self.gain_mode not in ("auto", "manual"):
            raise ConfigValidationError(
                f"gain_mode must be 'auto' or 'manual', got {self.gain_mode}"
            )
        if not (0 <= self.lna_gain <= 40):
            raise ConfigValidationError(
                f"lna_gain must be between 0 and 40 dB, got {self.lna_gain}"
            )
        if not (0 <= self.vga_gain <= 62):
            raise ConfigValidationError(
                f"vga_gain must be between 0 and 62 dB, got {self.vga_gain}"
            )
        if not (0 <= self.tx_vga_gain <= 47):
            raise ConfigValidationError(
                f"tx_vga_gain must be between 0 and 47 dB, got {self.tx_vga_gain}"
            )


@dataclass
class DualSDRConfig:
    """Configuration for dual-SDR operation."""

    rtlsdr: DeviceConfig = field(
        default_factory=lambda: DeviceConfig(device_type="rtlsdr")
    )
    hackrf: DeviceConfig = field(
        default_factory=lambda: DeviceConfig(
            device_type="hackrf", sample_rate=10e6, bandwidth=10e6
        )
    )
    # Operation mode
    mode: str = "dual_rx"  # "dual_rx", "full_duplex", "tx_monitor", "wideband_scan"
    # Synchronization
    sync_enabled: bool = False
    sync_method: str = "software"  # "software", "external_clock", "gps"

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields."""
        valid_modes = ("dual_rx", "full_duplex", "tx_monitor", "wideband_scan", "relay")
        valid_sync_methods = ("software", "external_clock", "gps")

        if self.mode not in valid_modes:
            raise ConfigValidationError(
                f"mode must be one of {valid_modes}, got {self.mode}"
            )
        if self.sync_method not in valid_sync_methods:
            raise ConfigValidationError(
                f"sync_method must be one of {valid_sync_methods}, got {self.sync_method}"
            )


@dataclass
class DSPConfig:
    """Configuration for DSP processing."""

    fft_size: int = 4096
    fft_window: str = (
        "hann"  # "hann", "hamming", "blackman", "blackman-harris", "flat-top"
    )
    fft_overlap: float = 0.5  # 50% overlap
    averaging_mode: str = "rms"  # "rms", "peak_hold", "min_hold", "linear"
    averaging_count: int = 10
    dc_removal: bool = True
    iq_correction: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields."""
        valid_windows = (
            "hann",
            "hamming",
            "blackman",
            "blackman-harris",
            "flat-top",
            "rectangular",
        )
        valid_avg_modes = ("rms", "peak_hold", "min_hold", "linear", "exponential")

        # FFT size must be power of 2 and reasonable
        if self.fft_size < 64 or self.fft_size > 1048576:
            raise ConfigValidationError(
                f"fft_size must be between 64 and 1048576, got {self.fft_size}"
            )
        if self.fft_size & (self.fft_size - 1) != 0:
            raise ConfigValidationError(
                f"fft_size must be a power of 2, got {self.fft_size}"
            )
        if self.fft_window.lower() not in valid_windows:
            raise ConfigValidationError(
                f"fft_window must be one of {valid_windows}, got {self.fft_window}"
            )
        if not (0.0 <= self.fft_overlap < 1.0):
            raise ConfigValidationError(
                f"fft_overlap must be between 0.0 and 1.0 (exclusive), got {self.fft_overlap}"
            )
        if self.averaging_mode.lower() not in valid_avg_modes:
            raise ConfigValidationError(
                f"averaging_mode must be one of {valid_avg_modes}, got {self.averaging_mode}"
            )
        if self.averaging_count < 1 or self.averaging_count > 10000:
            raise ConfigValidationError(
                f"averaging_count must be between 1 and 10000, got {self.averaging_count}"
            )


@dataclass
class RecordingConfig:
    """Configuration for recording."""

    output_dir: str = "./recordings"
    format: str = "cf32"  # "cu8", "cs8", "cs16", "cf32"
    include_metadata: bool = True  # SigMF metadata
    max_file_size_mb: int = 1024  # 1 GB default

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields."""
        valid_formats = ("cu8", "cs8", "cs16", "cf32", "wav")

        if self.format.lower() not in valid_formats:
            raise ConfigValidationError(
                f"format must be one of {valid_formats}, got {self.format}"
            )
        if self.max_file_size_mb < 1 or self.max_file_size_mb > 102400:
            raise ConfigValidationError(
                f"max_file_size_mb must be between 1 and 102400 (100 GB), got {self.max_file_size_mb}"
            )


@dataclass
class SDRConfig:
    """Main configuration container."""

    dual_sdr: DualSDRConfig = field(default_factory=DualSDRConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SDRConfig":
        """Create configuration from dictionary."""
        config = cls()

        if "dual_sdr" in data:
            ds = data["dual_sdr"]
            if "rtlsdr" in ds:
                config.dual_sdr.rtlsdr = DeviceConfig(**ds["rtlsdr"])
            if "hackrf" in ds:
                config.dual_sdr.hackrf = DeviceConfig(**ds["hackrf"])
            if "mode" in ds:
                config.dual_sdr.mode = ds["mode"]
            if "sync_enabled" in ds:
                config.dual_sdr.sync_enabled = ds["sync_enabled"]
            if "sync_method" in ds:
                config.dual_sdr.sync_method = ds["sync_method"]

        if "dsp" in data:
            config.dsp = DSPConfig(**data["dsp"])

        if "recording" in data:
            config.recording = RecordingConfig(**data["recording"])

        return config

    def save(self, path: str) -> bool:
        """Save configuration to JSON file.

        Args:
            path: File path to save configuration to

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {path}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize configuration: {e}")
            return False

    # Maximum config file size (1 MB should be more than enough)
    MAX_CONFIG_FILE_SIZE = 1024 * 1024

    @classmethod
    def load(cls, path: str) -> Optional["SDRConfig"]:
        """Load configuration from JSON file.

        Args:
            path: File path to load configuration from

        Returns:
            SDRConfig instance or None if loading failed
        """
        try:
            # Check file size before reading to prevent DoS
            file_path = Path(path)
            if file_path.exists():
                file_size = file_path.stat().st_size
                if file_size > cls.MAX_CONFIG_FILE_SIZE:
                    logger.error(
                        f"Configuration file too large ({file_size} bytes > {cls.MAX_CONFIG_FILE_SIZE}): {path}"
                    )
                    return None

            with open(path, "r") as f:
                data = json.load(f)
            config = cls.from_dict(data)
            logger.info(f"Configuration loaded from {path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {path}")
            return None
        except (OSError, IOError) as e:
            logger.error(f"Failed to read configuration from {path}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {path}: {e}")
            return None
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed in {path}: {e}")
            return None
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Invalid configuration format in {path}: {e}")
            return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get default configuration file path."""
        config_dir = Path.home() / ".config" / "sdr_module"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"

    def save_default(self) -> None:
        """Save to default configuration path."""
        self.save(str(self.get_default_config_path()))

    @classmethod
    def load_default(cls) -> "SDRConfig":
        """Load from default configuration path, or create new if not found or invalid."""
        path = cls.get_default_config_path()
        if path.exists():
            config = cls.load(str(path))
            if config is not None:
                return config
            logger.warning("Using default configuration due to load failure")
        return cls()


# Preset configurations for common use cases
PRESETS: Dict[str, SDRConfig] = {}


def create_preset_dual_rx() -> SDRConfig:
    """Preset for dual RX monitoring."""
    config = SDRConfig()
    config.dual_sdr.mode = "dual_rx"
    config.dual_sdr.rtlsdr.frequency = 433e6  # ISM band
    config.dual_sdr.hackrf.frequency = 915e6  # ISM band
    return config


def create_preset_full_duplex() -> SDRConfig:
    """Preset for full-duplex transceiver operation."""
    config = SDRConfig()
    config.dual_sdr.mode = "full_duplex"
    config.dual_sdr.rtlsdr.frequency = 146.52e6  # 2m calling
    config.dual_sdr.hackrf.frequency = 146.52e6  # TX same freq
    return config


def create_preset_adsb() -> SDRConfig:
    """Preset for ADS-B reception."""
    config = SDRConfig()
    config.dual_sdr.mode = "dual_rx"
    config.dual_sdr.rtlsdr.frequency = 1090e6  # ADS-B
    config.dual_sdr.rtlsdr.sample_rate = 2.4e6
    config.dual_sdr.rtlsdr.gain = 40.0
    config.dual_sdr.hackrf.frequency = 131.55e6  # ACARS
    return config


def create_preset_wideband_scan() -> SDRConfig:
    """Preset for wideband spectrum scanning."""
    config = SDRConfig()
    config.dual_sdr.mode = "wideband_scan"
    config.dsp.fft_size = 8192
    config.dsp.averaging_mode = "peak_hold"
    return config


# Register presets
PRESETS["dual_rx"] = create_preset_dual_rx()
PRESETS["full_duplex"] = create_preset_full_duplex()
PRESETS["adsb"] = create_preset_adsb()
PRESETS["wideband_scan"] = create_preset_wideband_scan()


def get_preset(name: str) -> Optional[SDRConfig]:
    """Get a preset configuration by name."""
    return PRESETS.get(name)


def list_presets() -> List[str]:
    """List available preset names."""
    return list(PRESETS.keys())
