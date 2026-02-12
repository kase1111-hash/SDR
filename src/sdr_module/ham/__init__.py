"""
Ham radio features for SDR Module.

Optional subpackage providing amateur radio functionality:
- Callsign identification (CW ID)
- SSTV image decoding (ISS reception)
- Signal meter (S-units / RST reporting)
- QRP (low power) operations
- AM/FM radio tuner (vintage car radio style)
"""

from .callsign import (
    CallsignConfig,
    CallsignIdentifier,
    IdentificationMode,
    MorseEncoder,
    audio_to_fm_iq,
    generate_cw_id,
    generate_tx_id,
)
from .qrp import (
    QRP_LIMITS,
    AmplifierStage,
    PowerChainResult,
    QRPClass,
    QRPController,
    QRPLimits,
    dbm_to_mw,
    dbm_to_watts,
    format_power,
    format_power_verbose,
    get_qro_amplifier_chain,
    get_qrp_amplifier_chain,
    mw_to_dbm,
    watts_to_dbm,
)
from .signal_meter import (
    DB_PER_S_UNIT,
    S9_DBM,
    S_METER_REFERENCE,
    SignalHistory,
    SignalMeter,
    SignalMode,
    SignalReading,
)
from .sstv import (
    SSTV_MODES,
    SSTVDecoder,
    SSTVImageViewer,
    SSTVMode,
    SSTVModeSpec,
    SSTVState,
)

__all__ = [
    # Callsign identification
    "IdentificationMode",
    "CallsignConfig",
    "CallsignIdentifier",
    "MorseEncoder",
    "generate_cw_id",
    "audio_to_fm_iq",
    "generate_tx_id",
    # SSTV
    "SSTVMode",
    "SSTVModeSpec",
    "SSTV_MODES",
    "SSTVState",
    "SSTVDecoder",
    "SSTVImageViewer",
    # Signal Meter
    "SignalMode",
    "SignalReading",
    "SignalMeter",
    "SignalHistory",
    "S_METER_REFERENCE",
    "S9_DBM",
    "DB_PER_S_UNIT",
    # QRP
    "dbm_to_watts",
    "watts_to_dbm",
    "dbm_to_mw",
    "mw_to_dbm",
    "format_power",
    "format_power_verbose",
    "QRPClass",
    "QRPLimits",
    "QRP_LIMITS",
    "AmplifierStage",
    "PowerChainResult",
    "QRPController",
    "get_qrp_amplifier_chain",
    "get_qro_amplifier_chain",
]
