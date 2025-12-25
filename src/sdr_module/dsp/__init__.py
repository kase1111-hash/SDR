"""
DSP module - Signal processing components.
"""

from .spectrum import SpectrumAnalyzer
from .filters import FilterBank
from .demodulators import Demodulator
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

__all__ = [
    "SpectrumAnalyzer",
    "FilterBank",
    "Demodulator",
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
]
