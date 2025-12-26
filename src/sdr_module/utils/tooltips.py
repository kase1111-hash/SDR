"""
Tooltips and help text for SDR module features.

Provides user-friendly explanations for complex SDR concepts
and feature descriptions for UI integration.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Tooltip:
    """Tooltip with title and description."""
    title: str
    short: str          # One-liner for quick display
    detailed: str       # Full explanation
    warning: str = ""   # Optional warning message


# =============================================================================
# RF & Hardware Tooltips
# =============================================================================

RF_TOOLTIPS: Dict[str, Tooltip] = {
    "center_frequency": Tooltip(
        title="Center Frequency",
        short="The frequency your SDR is tuned to receive or transmit.",
        detailed="""The center frequency is the middle of your receive/transmit
bandwidth. For example, if tuned to 100 MHz with 2 MHz bandwidth, you'll
see signals from 99-101 MHz. The RTL-SDR covers 500 kHz to 1.7 GHz,
while HackRF covers 1 MHz to 6 GHz.""",
    ),

    "sample_rate": Tooltip(
        title="Sample Rate",
        short="How many samples per second the SDR captures (determines bandwidth).",
        detailed="""Sample rate directly determines your visible bandwidth. At 2.4 MS/s,
you see 2.4 MHz of spectrum. Higher rates show more spectrum but require
more CPU and USB bandwidth. RTL-SDR maxes at 2.56 MS/s, HackRF at 20 MS/s.
Nyquist theorem: sample rate must be ≥2× highest frequency of interest.""",
        warning="Rates above 2.4 MS/s on RTL-SDR may drop samples on some systems."
    ),

    "gain": Tooltip(
        title="RF Gain",
        short="Amplification applied to received signals (in dB).",
        detailed="""Gain amplifies weak signals but also amplifies noise. Too little
gain = weak signals buried in noise. Too much gain = distortion and
intermodulation from strong signals. Start with AGC or ~30dB, then
adjust based on signal quality. RTL-SDR: 0-49.6dB, HackRF: 0-62dB.""",
        warning="Excessive gain causes signal clipping and spurious signals."
    ),

    "bandwidth": Tooltip(
        title="Filter Bandwidth",
        short="The width of the frequency range being received.",
        detailed="""Bandwidth determines how much spectrum you capture. Narrower
bandwidth = less noise but may cut off wideband signals. For FM broadcast,
use 200kHz+. For SSB voice, 2.4-3kHz. For digital modes, match the
signal's occupied bandwidth. HackRF has selectable filters from 1.75-28 MHz.""",
    ),

    "bias_tee": Tooltip(
        title="Bias Tee",
        short="Supplies DC power through the antenna cable for active antennas/LNAs.",
        detailed="""The bias tee injects DC voltage (typically 4.5V) onto the center
conductor of the coax to power external devices like LNAs (Low Noise
Amplifiers) or active antennas. This eliminates the need for separate
power cables. RTL-SDR V3+ provides 4.5V at up to 180mA.""",
        warning="Only enable if your antenna/LNA requires power. May damage passive antennas!"
    ),

    "direct_sampling": Tooltip(
        title="Direct Sampling Mode",
        short="Bypasses the tuner to receive HF frequencies (below 24 MHz).",
        detailed="""Normal RTL-SDR uses the R820T2 tuner which only works above 24 MHz.
Direct sampling bypasses this, feeding the ADC directly to receive HF
(500 kHz - 24 MHz). Q-branch mode is recommended for RTL-SDR Blog devices.
Sensitivity is reduced compared to a dedicated HF receiver.""",
        warning="Performance below 24 MHz is limited. Consider an upconverter for better HF reception."
    ),

    "max_input_power": Tooltip(
        title="Maximum Input Power",
        short="The strongest signal level your SDR can safely receive.",
        detailed="""Exceeding max input power can permanently damage the SDR's frontend.
RTL-SDR can handle +10 dBm (10 mW), but HackRF is only rated for -5 dBm
(0.3 mW). When transmitting nearby or monitoring your own TX, ALWAYS use
appropriate attenuation (30-60 dB or more).""",
        warning="HackRF max input is -5 dBm! Exceeding this WILL cause permanent damage!"
    ),

    "noise_figure": Tooltip(
        title="Noise Figure",
        short="How much noise the receiver adds to the signal (lower is better).",
        detailed="""Noise figure (NF) measures receiver sensitivity degradation. A 3dB NF
means the receiver doubles the noise. RTL-SDR: ~6-8dB, HackRF: ~10-15dB.
Adding an external LNA with 1-2dB NF before the SDR significantly improves
weak signal reception. Place LNA close to antenna for best results.""",
    ),

    "dynamic_range": Tooltip(
        title="Dynamic Range",
        short="The ratio between the strongest and weakest signals you can receive simultaneously.",
        detailed="""Dynamic range is limited by ADC bit depth. 8-bit ADCs (RTL-SDR, HackRF)
provide ~42-48dB range. This means if a strong signal is present, weak
signals within 42dB may be masked. Higher-bit SDRs like SDRPlay (14-bit)
offer ~84dB range. Use attenuation or filtering when strong signals are nearby.""",
    ),
}

# =============================================================================
# Dual-SDR Operation Tooltips
# =============================================================================

DUAL_SDR_TOOLTIPS: Dict[str, Tooltip] = {
    "dual_rx": Tooltip(
        title="Dual Receive Mode",
        short="Both SDRs receive simultaneously on different frequencies.",
        detailed="""Monitor two completely separate frequency bands at once. RTL-SDR
and HackRF operate independently - tune each to different frequencies.
Example uses: ADS-B (1090 MHz) + ACARS (131 MHz), or monitoring two
ISM bands (433 + 915 MHz) simultaneously.""",
    ),

    "full_duplex": Tooltip(
        title="Full-Duplex Mode",
        short="Transmit with HackRF while receiving with RTL-SDR.",
        detailed="""Since HackRF is half-duplex (can't TX and RX simultaneously), we use
RTL-SDR for receiving while HackRF transmits. This enables transceiver
applications, repeaters, and protocol testing. Ensure adequate frequency
separation or filtering to prevent RTL-SDR overload from TX signal.""",
        warning="Use bandpass filters to prevent TX signal from overloading RTL-SDR receiver!"
    ),

    "tx_monitor": Tooltip(
        title="TX Monitoring Mode",
        short="Monitor your own transmissions for quality assurance.",
        detailed="""Use RTL-SDR to observe HackRF's transmitted signal in real-time.
Verify spectral purity, modulation quality, and spurious emissions.
RTL-SDR must be heavily attenuated (40-60dB) to prevent damage and
allow proper signal analysis. Use a directional coupler for best results.""",
        warning="ALWAYS use 40-60dB attenuation between HackRF TX and RTL-SDR input!"
    ),

    "wideband_scan": Tooltip(
        title="Wideband Scan Mode",
        short="Cover more spectrum by scanning different ranges with each SDR.",
        detailed="""Split the spectrum between devices for faster scanning. RTL-SDR
handles 0-1.7 GHz while HackRF covers 1.7-6 GHz. Coordinate scanning
to avoid overlap and maximize coverage. Useful for spectrum surveys
and signal hunting across wide frequency ranges.""",
    ),

    "clock_sync": Tooltip(
        title="Clock Synchronization",
        short="Aligning timing between the two SDRs.",
        detailed="""RTL-SDR and HackRF use independent clocks, so samples aren't
time-aligned. For most applications this doesn't matter. For precise
timing (TDOA, coherent processing), use HackRF's external clock input
with a GPSDO or shared reference. Software correlation can align
streams to ~1ms accuracy.""",
        warning="Precise synchronization requires external hardware (GPSDO, shared 10MHz reference)."
    ),
}

# =============================================================================
# DSP & Spectrum Analysis Tooltips
# =============================================================================

DSP_TOOLTIPS: Dict[str, Tooltip] = {
    "fft_size": Tooltip(
        title="FFT Size",
        short="Number of points in the frequency transform (larger = finer resolution).",
        detailed="""FFT size determines frequency resolution (RBW = sample_rate / FFT_size).
At 2.4 MS/s with 4096-point FFT, RBW = 586 Hz. Larger FFTs give finer
resolution but require more CPU and have slower update rates. Common
sizes: 1024 (fast), 4096 (balanced), 8192+ (high resolution).""",
    ),

    "window_function": Tooltip(
        title="Window Function",
        short="Reduces spectral leakage artifacts in the FFT.",
        detailed="""Raw FFT assumes the signal repeats infinitely, causing 'leakage'
where sharp signals spread across bins. Windows taper the signal edges:
• Rectangular: No tapering, best resolution, worst leakage
• Hann: Good general purpose, moderate resolution/leakage
• Blackman-Harris: Excellent leakage rejection, reduced resolution
• Flat-top: Accurate amplitude measurement, poor resolution""",
    ),

    "averaging_rms": Tooltip(
        title="RMS Averaging",
        short="Averages power values to reduce noise fluctuations.",
        detailed="""RMS (Root Mean Square) averaging reduces noise variance while
preserving signal levels. Higher averaging count = smoother display
but slower response to changes. 10-20 averages is typical. Power
values are averaged, then converted to dB for display.""",
    ),

    "averaging_peak_hold": Tooltip(
        title="Peak Hold",
        short="Displays the maximum value seen at each frequency.",
        detailed="""Peak hold remembers and displays the highest power level ever seen
at each frequency bin. Useful for catching intermittent signals or
measuring maximum interference levels. Clear/reset to start fresh.""",
    ),

    "dc_offset": Tooltip(
        title="DC Offset Removal",
        short="Removes the spike at center frequency caused by hardware imperfections.",
        detailed="""Most SDRs show a false signal at exactly the center frequency due
to DC bias in the I/Q signal path. DC offset removal subtracts the
average I and Q values to eliminate this spike. Some residual may
remain; offsetting your tuned frequency slightly can help.""",
    ),

    "iq_correction": Tooltip(
        title="I/Q Imbalance Correction",
        short="Fixes image signals caused by gain/phase mismatch between I and Q.",
        detailed="""Imperfect analog hardware causes the I and Q channels to have
slightly different gains and phases, creating mirror images of strong
signals on the opposite side of center frequency. I/Q correction
estimates and compensates for these imbalances, typically reducing
images by 20-40dB.""",
    ),

    "resolution_bandwidth": Tooltip(
        title="Resolution Bandwidth (RBW)",
        short="The smallest frequency difference the analyzer can distinguish.",
        detailed="""RBW = Sample Rate ÷ FFT Size. Smaller RBW resolves closer-spaced
signals but requires larger FFT or lower sample rate. For measuring
noise floor, use small RBW. For fast scanning, use larger RBW.
Spectrum analyzers typically display RBW with measurements.""",
    ),
}

# =============================================================================
# Modulation & Demodulation Tooltips
# =============================================================================

MODULATION_TOOLTIPS: Dict[str, Tooltip] = {
    "am": Tooltip(
        title="AM (Amplitude Modulation)",
        short="Audio encoded by varying signal strength.",
        detailed="""The oldest modulation type - audio information is carried in the
amplitude envelope of the carrier. Simple to demodulate (envelope
detection) but inefficient and susceptible to noise. Used for
broadcast AM radio (530-1700 kHz) and aircraft communications.""",
    ),

    "fm": Tooltip(
        title="FM (Frequency Modulation)",
        short="Audio encoded by varying the frequency.",
        detailed="""Audio shifts the carrier frequency up/down proportionally. More
resistant to noise and interference than AM. Wideband FM (broadcast,
±75kHz deviation) sounds great; narrowband FM (amateur, ±5kHz) fits
more channels. Demodulated by measuring instantaneous frequency.""",
    ),

    "ssb": Tooltip(
        title="SSB (Single Sideband)",
        short="AM with carrier and one sideband removed for efficiency.",
        detailed="""SSB transmits only the upper (USB) or lower (LSB) sideband,
eliminating the carrier and redundant sideband. 3× more power
efficient than AM and uses half the bandwidth. Requires accurate
tuning (±50Hz). USB used above 10MHz, LSB below. Standard for
amateur HF and maritime communications.""",
    ),

    "ook": Tooltip(
        title="OOK (On-Off Keying)",
        short="Digital data transmitted by turning carrier on/off.",
        detailed="""Simplest digital modulation - carrier present = 1, absent = 0.
Used by many ISM band devices: garage door openers, car key fobs,
weather sensors, 433MHz remotes. Easy to decode by measuring
signal envelope and applying threshold detection.""",
    ),

    "fsk": Tooltip(
        title="FSK (Frequency Shift Keying)",
        short="Digital data encoded as frequency shifts.",
        detailed="""Binary FSK shifts between two frequencies to represent 0 and 1.
More robust than OOK in noisy environments. Common in pagers (POCSAG),
AX.25 packet radio, and many digital modes. Deviation and symbol rate
vary by protocol. GFSK (Gaussian FSK) smooths transitions for
narrower bandwidth (Bluetooth, DECT).""",
    ),

    "psk": Tooltip(
        title="PSK (Phase Shift Keying)",
        short="Digital data encoded as phase changes.",
        detailed="""Data represented by shifting carrier phase. BPSK uses 2 phases
(0°, 180°) for 1 bit/symbol. QPSK uses 4 phases for 2 bits/symbol.
More bandwidth-efficient than FSK but requires coherent detection.
Common in digital TV (DVB), WiFi, and satellite communications.""",
    ),

    "qam": Tooltip(
        title="QAM (Quadrature Amplitude Modulation)",
        short="Data encoded in both amplitude and phase for high throughput.",
        detailed="""Combines amplitude and phase modulation for maximum data density.
16-QAM = 4 bits/symbol, 256-QAM = 8 bits/symbol. Used in cable modems,
digital TV, WiFi (up to 1024-QAM). Requires high SNR - cable/fiber
only, not suitable for noisy RF channels.""",
    ),
}

# =============================================================================
# Signal Classification Tooltips
# =============================================================================

CLASSIFICATION_TOOLTIPS: Dict[str, Tooltip] = {
    "snr": Tooltip(
        title="SNR (Signal-to-Noise Ratio)",
        short="How much stronger the signal is compared to noise (higher = better).",
        detailed="""SNR in dB tells you signal quality. <3dB: signal barely visible.
10dB: clearly visible, may have errors. 20dB: good quality. 30dB+:
excellent. Measured by comparing signal peak power to noise floor
power. Improve SNR with: better antenna, LNA, narrower bandwidth,
or removing interference sources.""",
    ),

    "spectral_flatness": Tooltip(
        title="Spectral Flatness",
        short="Measures how noise-like a signal is (1.0 = pure noise).",
        detailed="""Also called Wiener entropy. Ratio of geometric to arithmetic mean
of spectrum. Pure tone = 0, white noise = 1. Used to distinguish
signals from noise floor. Values 0.8+ suggest noise or very wideband
signal. Values <0.3 indicate narrowband signal present.""",
    ),

    "crest_factor": Tooltip(
        title="Crest Factor",
        short="Peak-to-average ratio indicating signal dynamics.",
        detailed="""High crest factor (>4) suggests bursty/pulsed signals. Low crest
factor (~1.4) indicates continuous carrier. AM voice: ~4-5. FM: ~1.4.
OOK digital: high (pulsed). Helps distinguish between analog and
digital modulations and identify pulsed signals.""",
    ),

    "bandwidth_estimation": Tooltip(
        title="Bandwidth Estimation",
        short="Automatically measures the signal's occupied bandwidth.",
        detailed="""Measures frequency span containing 99% of signal power (99% OBW).
Helps identify signal type: narrowband FM ~15kHz, broadcast FM ~200kHz,
WiFi ~20MHz. Also useful for filter design - set filter bandwidth
slightly wider than occupied bandwidth.""",
    ),
}


# =============================================================================
# Lookup Functions
# =============================================================================

ALL_TOOLTIPS: Dict[str, Tooltip] = {
    **RF_TOOLTIPS,
    **DUAL_SDR_TOOLTIPS,
    **DSP_TOOLTIPS,
    **MODULATION_TOOLTIPS,
    **CLASSIFICATION_TOOLTIPS,
}


def get_tooltip(key: str) -> Optional[Tooltip]:
    """Get tooltip by key."""
    return ALL_TOOLTIPS.get(key)


def get_short_tip(key: str) -> str:
    """Get short tooltip text."""
    tip = ALL_TOOLTIPS.get(key)
    return tip.short if tip else ""


def get_detailed_tip(key: str) -> str:
    """Get detailed tooltip text."""
    tip = ALL_TOOLTIPS.get(key)
    return tip.detailed if tip else ""


def list_tooltips() -> list:
    """List all available tooltip keys."""
    return list(ALL_TOOLTIPS.keys())


def get_tooltips_by_category(category: str) -> Dict[str, Tooltip]:
    """Get tooltips by category name."""
    categories = {
        "rf": RF_TOOLTIPS,
        "dual_sdr": DUAL_SDR_TOOLTIPS,
        "dsp": DSP_TOOLTIPS,
        "modulation": MODULATION_TOOLTIPS,
        "classification": CLASSIFICATION_TOOLTIPS,
    }
    return categories.get(category.lower(), {})
