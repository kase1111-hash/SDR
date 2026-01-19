"""
Configuration management for SDR module.

Handles device configuration, DSP settings, and persistence.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SDRConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

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
        """Load from default configuration path, or create new."""
        path = cls.get_default_config_path()
        if path.exists():
            return cls.load(str(path))
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
