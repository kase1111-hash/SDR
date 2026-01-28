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

    device_type: str = "rtlsdr"  # "rtlsdr", "hackrf", or "mxk2_keyer"
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
        if self.device_type not in ("rtlsdr", "hackrf", "mxk2_keyer"):
            raise ConfigValidationError(
                f"Invalid device_type: {self.device_type}. "
                "Must be 'rtlsdr', 'hackrf', or 'mxk2_keyer'"
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
class KeyerConfig:
    """Configuration for CW keyer devices (e.g., MX-K2)."""

    device_type: str = "mxk2_keyer"
    device_index: int = 0
    port: str = ""  # Serial port (e.g., "/dev/ttyUSB0" or "COM3")
    baud_rate: int = 1200  # Default baud rate for MX-K2
    wpm: int = 20  # Words per minute (5-50)
    sidetone_freq: int = 700  # Sidetone frequency in Hz (300-1200)
    sidetone_enabled: bool = True
    paddle_mode: str = "iambic_b"  # "iambic_a", "iambic_b", "ultimatic", "bug", "straight"
    paddle_swap: bool = False  # Swap dit/dah paddles
    weight: int = 50  # Dit/dah weight (25-75, 50 = standard 1:3)
    ptt_lead_time_ms: int = 50  # PTT lead time before keying
    ptt_tail_time_ms: int = 100  # PTT hang time after keying
    auto_space: bool = False  # Automatic inter-character spacing

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration fields."""
        valid_paddle_modes = ("iambic_a", "iambic_b", "ultimatic", "bug", "straight")
        if self.device_index < 0:
            raise ConfigValidationError(
                f"device_index must be non-negative, got {self.device_index}"
            )
        if self.baud_rate <= 0:
            raise ConfigValidationError(
                f"baud_rate must be positive, got {self.baud_rate}"
            )
        if not (5 <= self.wpm <= 50):
            raise ConfigValidationError(
                f"wpm must be between 5 and 50, got {self.wpm}"
            )
        if not (300 <= self.sidetone_freq <= 1200):
            raise ConfigValidationError(
                f"sidetone_freq must be between 300 and 1200 Hz, got {self.sidetone_freq}"
            )
        if self.paddle_mode not in valid_paddle_modes:
            raise ConfigValidationError(
                f"paddle_mode must be one of {valid_paddle_modes}, got {self.paddle_mode}"
            )
        if not (25 <= self.weight <= 75):
            raise ConfigValidationError(
                f"weight must be between 25 and 75, got {self.weight}"
            )
        if self.ptt_lead_time_ms < 0:
            raise ConfigValidationError(
                f"ptt_lead_time_ms must be non-negative, got {self.ptt_lead_time_ms}"
            )
        if self.ptt_tail_time_ms < 0:
            raise ConfigValidationError(
                f"ptt_tail_time_ms must be non-negative, got {self.ptt_tail_time_ms}"
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


@dataclass
class RecordingConfig:
    """Configuration for recording."""

    output_dir: str = "./recordings"
    format: str = "cf32"  # "cu8", "cs8", "cs16", "cf32"
    include_metadata: bool = True  # SigMF metadata
    max_file_size_mb: int = 1024  # 1 GB default


@dataclass
class SDRConfig:
    """Main configuration container."""

    dual_sdr: DualSDRConfig = field(default_factory=DualSDRConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    keyer: KeyerConfig = field(default_factory=KeyerConfig)

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

        if "keyer" in data:
            config.keyer = KeyerConfig(**data["keyer"])

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

    @classmethod
    def load(cls, path: str) -> Optional["SDRConfig"]:
        """Load configuration from JSON file.

        Args:
            path: File path to load configuration from

        Returns:
            SDRConfig instance or None if loading failed
        """
        try:
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
