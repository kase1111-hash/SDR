"""
DSP module - Signal processing components.
"""

from .spectrum import SpectrumAnalyzer
from .filters import (
    FilterBank,
    Decimator,
    Interpolator,
    Resampler,
    AGC,
    AGCConfig,
    AGCMode,
    FastAGC,
)
from .demodulators import Demodulator, CWDemodulator, GFSKDemodulator, MSKDemodulator, QAMDemodulator, MORSE_CODE
from .classifiers import SignalClassifier
from .frequency_lock import (
    FrequencyLocker,
    LockState,
    LockMode,
    LockStatus,
    LockConfig,
    LockTarget,
)
from .afc import (
    AutomaticFrequencyControl,
    AFCMode,
    AFCMethod,
    AFCStatus,
    AFCConfig,
)
from .scanner import (
    FrequencyScanner,
    ScanMode,
    ScanState,
    ScanDirection,
    ScanConfig,
    ScanStatus,
    ScanResult,
    SignalHit,
)
from .protocols import (
    ProtocolType,
    ProtocolDecoder,
    DecodedMessage,
    POCSAGDecoder,
    POCSAGMessage,
    AX25Decoder,
    AX25Frame,
    APRSMessage,
    RDSDecoder,
    RDSData,
    create_protocol_decoder,
)

__all__ = [
    "SpectrumAnalyzer",
    "FilterBank",
    "Decimator",
    "Interpolator",
    "Resampler",
    "AGC",
    "AGCConfig",
    "AGCMode",
    "FastAGC",
    "Demodulator",
    "CWDemodulator",
    "GFSKDemodulator",
    "MSKDemodulator",
    "QAMDemodulator",
    "MORSE_CODE",
    "SignalClassifier",
    "FrequencyLocker",
    "LockState",
    "LockMode",
    "LockStatus",
    "LockConfig",
    "LockTarget",
    "AutomaticFrequencyControl",
    "AFCMode",
    "AFCMethod",
    "AFCStatus",
    "AFCConfig",
    "FrequencyScanner",
    "ScanMode",
    "ScanState",
    "ScanDirection",
    "ScanConfig",
    "ScanStatus",
    "ScanResult",
    "SignalHit",
    "ProtocolType",
    "ProtocolDecoder",
    "DecodedMessage",
    "POCSAGDecoder",
    "POCSAGMessage",
    "AX25Decoder",
    "AX25Frame",
    "APRSMessage",
    "RDSDecoder",
    "RDSData",
    "create_protocol_decoder",
]
