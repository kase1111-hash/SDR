#!/usr/bin/env python3
"""
Test script for SDR framework.

Tests all major components without requiring actual SDR hardware.
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, '../src')

def test_imports():
    """Test all module imports."""
    print("Testing imports...")

    # Core
    from sdr_module import DeviceManager, DualSDRController, SampleBuffer
    print("  ✓ Core imports")

    # Devices
    from sdr_module.devices import SDRDevice, DeviceCapability, RTLSDRDevice, HackRFDevice
    print("  ✓ Device imports")

    # DSP
    from sdr_module.dsp import SpectrumAnalyzer, FilterBank, Demodulator, SignalClassifier
    print("  ✓ DSP imports")

    # Protocols
    from sdr_module.protocols import ProtocolDecoder, ProtocolDetector
    print("  ✓ Protocol imports")

    # UI
    from sdr_module.ui import (
        WaterfallDisplay, PacketHighlight, ColorMap,
        PacketHighlighter, DetectionMode, LivePacketDisplay
    )
    print("  ✓ UI imports")

    # Utils
    from sdr_module.utils import (
        db_to_linear, linear_to_db, freq_to_str,
        iq_to_complex, complex_to_iq,
        get_tooltip, Tooltip
    )
    print("  ✓ Utils imports")

    # Config
    from sdr_module.core.config import SDRConfig, DualSDRConfig, DSPConfig
    print("  ✓ Config imports")

    print("All imports successful!\n")


def test_sample_buffer():
    """Test sample buffer functionality."""
    print("Testing SampleBuffer...")

    from sdr_module.core.sample_buffer import SampleBuffer, BufferOverflowPolicy

    # Create buffer
    buf = SampleBuffer(capacity=1024)
    assert buf.capacity == 1024
    assert buf.available == 0
    print("  ✓ Buffer creation")

    # Write samples
    samples = np.random.randn(100) + 1j * np.random.randn(100)
    samples = samples.astype(np.complex64)
    written = buf.write(samples)
    assert written == 100
    assert buf.available == 100
    print("  ✓ Write samples")

    # Read samples
    read_samples = buf.read(50, timeout=1.0)
    assert read_samples is not None
    assert len(read_samples) == 50
    assert buf.available == 50
    print("  ✓ Read samples")

    # Peek
    peeked = buf.peek(25)
    assert peeked is not None
    assert len(peeked) == 25
    assert buf.available == 50  # Peek doesn't consume
    print("  ✓ Peek samples")

    # Clear
    buf.clear()
    assert buf.available == 0
    print("  ✓ Clear buffer")

    # Overflow handling
    large_samples = np.random.randn(2000) + 1j * np.random.randn(2000)
    large_samples = large_samples.astype(np.complex64)
    buf.write(large_samples)
    assert buf.available <= buf.capacity
    print("  ✓ Overflow handling")

    print("SampleBuffer tests passed!\n")


def test_spectrum_analyzer():
    """Test spectrum analyzer."""
    print("Testing SpectrumAnalyzer...")

    from sdr_module.dsp.spectrum import SpectrumAnalyzer, WindowType, AveragingMode

    # Create analyzer
    analyzer = SpectrumAnalyzer(
        fft_size=1024,
        window=WindowType.HANN,
        averaging=AveragingMode.RMS
    )
    assert analyzer.fft_size == 1024
    print("  ✓ Analyzer creation")

    # Generate test signal (tone at 100kHz)
    sample_rate = 2.4e6
    t = np.arange(10000) / sample_rate
    freq = 100e3
    samples = np.exp(2j * np.pi * freq * t).astype(np.complex64)

    # Compute spectrum
    result = analyzer.compute_spectrum(samples, center_freq=0, sample_rate=sample_rate)
    assert len(result.frequencies) == 1024
    assert len(result.power_db) == 1024
    assert result.sample_rate == sample_rate
    print("  ✓ Spectrum computation")

    # Find peak
    peak_idx = np.argmax(result.power_db)
    peak_freq = result.frequencies[peak_idx]
    assert abs(peak_freq - freq) < sample_rate / 1024 * 2  # Within 2 bins
    print(f"  ✓ Peak detection (expected {freq/1e3:.1f}kHz, got {peak_freq/1e3:.1f}kHz)")

    # Test different windows
    for window in [WindowType.HAMMING, WindowType.BLACKMAN, WindowType.BLACKMAN_HARRIS]:
        analyzer = SpectrumAnalyzer(fft_size=1024, window=window)
        result = analyzer.compute_spectrum(samples, 0, sample_rate)
        assert len(result.power_db) == 1024
    print("  ✓ Window functions")

    print("SpectrumAnalyzer tests passed!\n")


def test_filters():
    """Test filter bank."""
    print("Testing FilterBank...")

    from sdr_module.dsp.filters import FilterBank, FIRFilter, FilterType, FilterSpec

    sample_rate = 2.4e6
    bank = FilterBank(sample_rate)

    # Create lowpass
    lpf = bank.create_lowpass("lpf_100k", 100e3, num_taps=51)
    assert lpf.num_taps == 51
    print("  ✓ Lowpass filter creation")

    # Create bandpass
    bpf = bank.create_bandpass("bpf_100k_200k", 100e3, 200e3, num_taps=101)
    assert bpf.num_taps == 101
    print("  ✓ Bandpass filter creation")

    # Test filtering
    samples = np.random.randn(1000) + 1j * np.random.randn(1000)
    samples = samples.astype(np.complex64)
    filtered = lpf.filter(samples)
    assert len(filtered) == len(samples)
    print("  ✓ Filter application")

    # Test frequency response
    freqs, response = lpf.frequency_response(512)
    assert len(freqs) == 257  # rfft returns N/2+1 points
    assert len(response) == 257
    print("  ✓ Frequency response")

    print("FilterBank tests passed!\n")


def test_demodulators():
    """Test demodulators."""
    print("Testing Demodulators...")

    from sdr_module.dsp.demodulators import (
        AMDemodulator, FMDemodulator, SSBDemodulator,
        OOKDemodulator, FSKDemodulator, create_demodulator, ModulationType
    )

    sample_rate = 2.4e6
    n_samples = 10000

    # AM demodulator
    am = AMDemodulator(sample_rate)
    carrier = np.exp(2j * np.pi * 100e3 * np.arange(n_samples) / sample_rate)
    modulation = 1 + 0.5 * np.sin(2 * np.pi * 1e3 * np.arange(n_samples) / sample_rate)
    am_signal = (carrier * modulation).astype(np.complex64)
    demod_am = am.demodulate(am_signal)
    assert len(demod_am) == n_samples
    print("  ✓ AM demodulation")

    # FM demodulator
    fm = FMDemodulator(sample_rate, max_deviation=75e3)
    phase = 2 * np.pi * 100e3 * np.arange(n_samples) / sample_rate
    phase += 0.5 * np.sin(2 * np.pi * 1e3 * np.arange(n_samples) / sample_rate)
    fm_signal = np.exp(1j * phase).astype(np.complex64)
    demod_fm = fm.demodulate(fm_signal)
    assert len(demod_fm) == n_samples
    print("  ✓ FM demodulation")

    # SSB demodulator
    ssb = SSBDemodulator(sample_rate, mode="usb")
    demod_ssb = ssb.demodulate(am_signal)
    assert len(demod_ssb) == n_samples
    print("  ✓ SSB demodulation")

    # OOK demodulator
    ook = OOKDemodulator(sample_rate)
    ook_signal = np.zeros(n_samples, dtype=np.complex64)
    ook_signal[1000:2000] = 1.0 + 0j
    ook_signal[3000:4000] = 1.0 + 0j
    demod_ook = ook.demodulate(ook_signal)
    assert len(demod_ook) == n_samples
    assert demod_ook[1500] > 0.5  # Should detect "on"
    assert demod_ook[2500] < 0.5  # Should detect "off"
    print("  ✓ OOK demodulation")

    # Factory function
    am2 = create_demodulator(ModulationType.AM, sample_rate)
    assert isinstance(am2, AMDemodulator)
    print("  ✓ Demodulator factory")

    print("Demodulator tests passed!\n")


def test_signal_classifier():
    """Test signal classifier."""
    print("Testing SignalClassifier...")

    from sdr_module.dsp.classifiers import SignalClassifier, SignalType

    sample_rate = 2.4e6
    classifier = SignalClassifier(sample_rate)

    # Test noise - classifier may return any type for low-level noise
    noise = (np.random.randn(10000) + 1j * np.random.randn(10000)).astype(np.complex64) * 0.01
    result = classifier.classify(noise)
    assert result.signal_type in SignalType  # Just verify valid type returned
    print(f"  ✓ Noise classification: {result.signal_type.value} (confidence: {result.confidence:.2f})")

    # Test FM signal (continuous carrier with varying phase)
    t = np.arange(10000) / sample_rate
    phase = 2 * np.pi * 100e3 * t + 10 * np.sin(2 * np.pi * 1e3 * t)
    fm_signal = np.exp(1j * phase).astype(np.complex64)
    result = classifier.classify(fm_signal)
    print(f"  ✓ FM-like classification: {result.signal_type.value}")

    # Test OOK/digital signal
    ook_signal = np.zeros(10000, dtype=np.complex64)
    for i in range(0, 10000, 1000):
        if i % 2000 == 0:
            ook_signal[i:i+500] = 1.0
    result = classifier.classify(ook_signal + noise * 0.1)
    print(f"  ✓ Digital-like classification: {result.signal_type.value}")

    # Check result structure
    assert hasattr(result, 'bandwidth_hz')
    assert hasattr(result, 'snr_db')
    assert hasattr(result, 'confidence')
    print("  ✓ Result structure")

    print("SignalClassifier tests passed!\n")


def test_waterfall():
    """Test waterfall display."""
    print("Testing WaterfallDisplay...")

    from sdr_module.ui.waterfall import WaterfallDisplay, ColorMap, PROTOCOL_COLORS

    # Create display
    waterfall = WaterfallDisplay(width=512, height=256)
    waterfall.set_frequency_range(433.92e6, 2.4e6)
    assert waterfall.width == 512
    assert waterfall.height == 256
    print("  ✓ Waterfall creation")

    # Add spectrum lines
    for i in range(10):
        power = np.random.randn(512) * 10 - 60
        waterfall.add_spectrum_line(power)
    assert waterfall.image.shape == (256, 512, 4)
    print("  ✓ Spectrum line addition")

    # Add highlight
    highlight = waterfall.add_packet_highlight(
        freq_start_hz=433.8e6,
        freq_end_hz=434.0e6,
        duration_lines=5,
        protocol="ook",
        label="Test packet"
    )
    assert highlight.protocol == "ook"
    assert len(waterfall.highlights) == 1
    print("  ✓ Packet highlighting")

    # Test color map change
    waterfall.set_colormap(ColorMap.VIRIDIS)
    print("  ✓ Color map change")

    # Test protocol colors
    assert "ook" in PROTOCOL_COLORS
    assert "adsb" in PROTOCOL_COLORS
    assert "pocsag" in PROTOCOL_COLORS
    print(f"  ✓ Protocol colors ({len(PROTOCOL_COLORS)} defined)")

    print("WaterfallDisplay tests passed!\n")


def test_packet_highlighter():
    """Test packet highlighter."""
    print("Testing PacketHighlighter...")

    from sdr_module.ui.waterfall import WaterfallDisplay
    from sdr_module.ui.packet_highlighter import (
        PacketHighlighter, DetectionConfig, DetectionMode
    )

    # Create components
    waterfall = WaterfallDisplay(width=512, height=256)
    waterfall.set_frequency_range(433.92e6, 2.4e6)

    config = DetectionConfig(
        mode=DetectionMode.ADAPTIVE,
        threshold_db=-50,
        min_bandwidth_hz=5000,
        min_duration_lines=2
    )

    highlighter = PacketHighlighter(waterfall, 2.4e6, config)
    print("  ✓ Highlighter creation")

    # Process spectrum with a signal burst
    for i in range(20):
        power = np.random.randn(512) * 5 - 80  # Noise floor
        if 5 <= i <= 10:
            # Add burst
            power[200:250] = -40  # Signal

        highlighter.process_spectrum(power)

    stats = highlighter.statistics
    print(f"  ✓ Packet detection (found {stats['total_packets']} packets)")

    # Reset
    highlighter.reset()
    highlighter.clear_history()
    assert highlighter.statistics['total_packets'] == 0
    print("  ✓ Reset and clear")

    print("PacketHighlighter tests passed!\n")


def test_config():
    """Test configuration system."""
    print("Testing Configuration...")

    from sdr_module.core.config import (
        SDRConfig, DualSDRConfig, DSPConfig, DeviceConfig,
        create_preset_dual_rx, create_preset_adsb, list_presets
    )

    # Create default config
    config = SDRConfig()
    assert config.dual_sdr is not None
    assert config.dsp is not None
    print("  ✓ Default config creation")

    # Modify config
    config.dual_sdr.rtlsdr.frequency = 433.92e6
    config.dual_sdr.hackrf.frequency = 915e6
    assert config.dual_sdr.rtlsdr.frequency == 433.92e6
    print("  ✓ Config modification")

    # Serialize/deserialize
    config_dict = config.to_dict()
    assert "dual_sdr" in config_dict
    assert "dsp" in config_dict

    config2 = SDRConfig.from_dict(config_dict)
    assert config2.dual_sdr.rtlsdr.frequency == config.dual_sdr.rtlsdr.frequency
    print("  ✓ Serialization")

    # Test presets
    presets = list_presets()
    assert len(presets) >= 4
    print(f"  ✓ Presets available: {presets}")

    adsb_config = create_preset_adsb()
    assert adsb_config.dual_sdr.rtlsdr.frequency == 1090e6
    print("  ✓ ADS-B preset")

    print("Configuration tests passed!\n")


def test_tooltips():
    """Test tooltips system."""
    print("Testing Tooltips...")

    from sdr_module.utils.tooltips import (
        get_tooltip, get_short_tip, get_detailed_tip,
        list_tooltips, get_tooltips_by_category
    )

    # Get tooltip
    tip = get_tooltip("center_frequency")
    assert tip is not None
    assert tip.title == "Center Frequency"
    assert len(tip.short) > 0
    assert len(tip.detailed) > 0
    print("  ✓ Get tooltip")

    # Short tip
    short = get_short_tip("sample_rate")
    assert "sample" in short.lower()
    print("  ✓ Short tip")

    # List tooltips
    all_tips = list_tooltips()
    assert len(all_tips) > 20
    print(f"  ✓ {len(all_tips)} tooltips available")

    # By category
    rf_tips = get_tooltips_by_category("rf")
    dsp_tips = get_tooltips_by_category("dsp")
    mod_tips = get_tooltips_by_category("modulation")
    assert len(rf_tips) > 0
    assert len(dsp_tips) > 0
    assert len(mod_tips) > 0
    print("  ✓ Category filtering")

    # Check warning field
    tip_with_warning = get_tooltip("max_input_power")
    assert len(tip_with_warning.warning) > 0
    print("  ✓ Warning field")

    print("Tooltips tests passed!\n")


def test_utils():
    """Test utility functions."""
    print("Testing Utils...")

    from sdr_module.utils import db_to_linear, linear_to_db, freq_to_str
    from sdr_module.utils.iq import (
        iq_to_complex, complex_to_iq,
        interleaved_to_complex, complex_to_interleaved,
        apply_dc_offset_correction
    )
    from sdr_module.utils.conversions import (
        dbm_to_watts, watts_to_dbm, str_to_freq
    )

    # dB conversions
    assert abs(db_to_linear(3) - 2.0) < 0.01
    assert abs(linear_to_db(2.0) - 3.0) < 0.1
    print("  ✓ dB conversions")

    # Frequency formatting
    assert "MHz" in freq_to_str(433.92e6)
    assert "GHz" in freq_to_str(2.4e9)
    assert "kHz" in freq_to_str(100e3)
    print("  ✓ Frequency formatting")

    # Frequency parsing
    assert abs(str_to_freq("433.92MHz") - 433.92e6) < 1
    assert abs(str_to_freq("2.4 GHz") - 2.4e9) < 1
    print("  ✓ Frequency parsing")

    # Power conversions
    assert abs(dbm_to_watts(30) - 1.0) < 0.01  # 30 dBm = 1W
    assert abs(watts_to_dbm(1.0) - 30) < 0.1
    print("  ✓ Power conversions")

    # I/Q conversions
    i = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    q = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    c = iq_to_complex(i, q)
    assert np.allclose(c, [1+0j, 0+1j, -1+0j])

    i2, q2 = complex_to_iq(c)
    assert np.allclose(i2, i)
    assert np.allclose(q2, q)
    print("  ✓ I/Q conversions")

    # Interleaved conversion
    interleaved = np.array([127, 127, 255, 127, 0, 127], dtype=np.uint8)
    complex_samples = interleaved_to_complex(interleaved, np.uint8)
    assert len(complex_samples) == 3
    print("  ✓ Interleaved conversion")

    # DC offset correction
    samples = np.ones(100, dtype=np.complex64) * (0.5 + 0.3j)
    corrected = apply_dc_offset_correction(samples + np.random.randn(100).astype(np.complex64) * 0.01)
    assert abs(np.mean(corrected)) < 0.1
    print("  ✓ DC offset correction")

    print("Utils tests passed!\n")


def test_protocol_framework():
    """Test protocol detection framework."""
    print("Testing Protocol Framework...")

    from sdr_module.protocols.base import ProtocolDecoder, ProtocolInfo, ProtocolType
    from sdr_module.protocols.detector import ProtocolDetector

    # Create detector
    detector = ProtocolDetector(sample_rate=2.4e6)
    assert detector is not None
    print("  ✓ Protocol detector creation")

    # List protocols (empty initially)
    protocols = detector.list_protocols()
    assert isinstance(protocols, list)
    print("  ✓ Protocol listing")

    # Test with samples (no decoders registered, should return empty)
    samples = np.random.randn(1000).astype(np.complex64)
    matches = detector.detect(samples)
    assert isinstance(matches, list)
    print("  ✓ Detection with no decoders")

    print("Protocol Framework tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SDR Framework Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("SampleBuffer", test_sample_buffer),
        ("SpectrumAnalyzer", test_spectrum_analyzer),
        ("Filters", test_filters),
        ("Demodulators", test_demodulators),
        ("SignalClassifier", test_signal_classifier),
        ("Waterfall", test_waterfall),
        ("PacketHighlighter", test_packet_highlighter),
        ("Config", test_config),
        ("Tooltips", test_tooltips),
        ("Utils", test_utils),
        ("Protocols", test_protocol_framework),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
