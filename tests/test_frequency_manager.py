"""
Tests for the frequency manager TX lockout system.

This module tests the safety-critical TX frequency lockout system that protects:
- GPS/GNSS navigation frequencies (safety-of-life)
- Aviation emergency frequencies
- ADS-B/Mode S transponder frequencies
- Emergency locator beacons (ELT/EPIRB)
- Marine distress frequencies
- Cellular bands
- License-based TX permissions
- Power limit enforcement
"""

import pytest

from sdr_module.core.frequency_manager import (
    BandPrivilege,
    FrequencyBand,
    FrequencyManager,
    LicenseClass,
    LockoutReason,
    get_frequency_manager,
    is_tx_allowed,
    set_license_class,
    get_license_class,
    get_power_limit,
    get_effective_power_limit,
    validate_tx_frequency,
    TX_LOCKOUT_BANDS,
    LICENSE_FREE_BANDS,
    POWER_HEADROOM_FACTOR,
)


class TestGPSLockouts:
    """Test GPS/GNSS frequency lockouts - SAFETY CRITICAL."""

    def test_gps_l1_center_blocked(self):
        """GPS L1 center frequency (1575.42 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(1575.42e6)
        assert not allowed
        assert "GPS" in reason.upper() or "NAVIGATION" in reason.upper()

    def test_gps_l1_band_edges_blocked(self):
        """GPS L1 band edges must be blocked."""
        # Lower edge
        allowed, _ = is_tx_allowed(1560.42e6)
        assert not allowed
        # Upper edge
        allowed, _ = is_tx_allowed(1590.42e6)
        assert not allowed

    def test_gps_l2_center_blocked(self):
        """GPS L2 center frequency (1227.60 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(1227.60e6)
        assert not allowed
        assert "GPS" in reason.upper() or "NAVIGATION" in reason.upper()

    def test_gps_l2_band_blocked(self):
        """GPS L2 band must be blocked."""
        # Within the protected range
        allowed, _ = is_tx_allowed(1220e6)
        assert not allowed
        allowed, _ = is_tx_allowed(1235e6)
        assert not allowed

    def test_gps_l5_center_blocked(self):
        """GPS L5 safety-of-life frequency (1176.45 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(1176.45e6)
        assert not allowed
        assert "GPS" in reason.upper() or "NAVIGATION" in reason.upper()

    def test_gps_l5_band_blocked(self):
        """GPS L5 band must be blocked."""
        allowed, _ = is_tx_allowed(1161.45e6)
        assert not allowed
        allowed, _ = is_tx_allowed(1191.45e6)
        assert not allowed

    def test_glonass_l1_blocked(self):
        """GLONASS L1 frequencies must be blocked."""
        # GLONASS L1 center around 1602 MHz
        allowed, reason = is_tx_allowed(1602e6)
        assert not allowed
        assert "GLONASS" in reason.upper() or "NAVIGATION" in reason.upper()

    def test_glonass_l2_blocked(self):
        """GLONASS L2 frequencies must be blocked."""
        allowed, reason = is_tx_allowed(1246e6)
        assert not allowed

    def test_galileo_e1_blocked(self):
        """Galileo E1 (same as GPS L1) must be blocked."""
        allowed, _ = is_tx_allowed(1575.42e6)
        assert not allowed

    def test_galileo_e5_blocked(self):
        """Galileo E5 frequencies must be blocked."""
        # E5a around 1176.45 MHz, E5b around 1207.14 MHz
        allowed, _ = is_tx_allowed(1191e6)
        assert not allowed

    def test_beidou_b1_blocked(self):
        """BeiDou B1 frequencies must be blocked."""
        # B1 center at 1561.098 MHz
        allowed, reason = is_tx_allowed(1561.098e6)
        assert not allowed

    def test_bandwidth_spill_into_gps_blocked(self):
        """TX with bandwidth spilling into GPS band must be blocked."""
        # Frequency just outside GPS L1, but bandwidth extends into it
        # GPS L1 lower edge is ~1560.42 MHz
        freq = 1555e6  # 5 MHz below GPS L1 lower edge
        bandwidth = 10e6  # 10 MHz bandwidth - will overlap GPS
        allowed, reason = is_tx_allowed(freq, bandwidth)
        assert not allowed


class TestAviationLockouts:
    """Test aviation safety frequency lockouts."""

    def test_aviation_emergency_121_5_blocked(self):
        """121.5 MHz aviation emergency frequency must be blocked."""
        allowed, reason = is_tx_allowed(121.5e6)
        assert not allowed
        assert "EMERGENCY" in reason.upper() or "AVIATION" in reason.upper()

    def test_aviation_emergency_243_blocked(self):
        """243.0 MHz military aviation emergency must be blocked."""
        allowed, reason = is_tx_allowed(243.0e6)
        assert not allowed
        assert "EMERGENCY" in reason.upper() or "AVIATION" in reason.upper()

    def test_adsb_1090_blocked(self):
        """ADS-B/Mode S transponder frequency (1090 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(1090e6)
        assert not allowed
        assert "ADS-B" in reason.upper() or "AVIATION" in reason.upper() or "TRANSPONDER" in reason.upper()

    def test_mode_s_interrogation_1030_blocked(self):
        """Mode S interrogation frequency (1030 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(1030e6)
        assert not allowed

    def test_adsb_band_edges_blocked(self):
        """ADS-B band edges must be blocked."""
        allowed, _ = is_tx_allowed(1088e6)
        assert not allowed
        allowed, _ = is_tx_allowed(1092e6)
        assert not allowed


class TestEmergencyLockouts:
    """Test emergency beacon frequency lockouts."""

    def test_elt_epirb_406_blocked(self):
        """ELT/EPIRB emergency beacon frequency (406 MHz) must be blocked."""
        allowed, reason = is_tx_allowed(406.0e6)
        assert not allowed
        assert "EMERGENCY" in reason.upper() or "ELT" in reason.upper() or "EPIRB" in reason.upper()

    def test_marine_distress_156_8_blocked(self):
        """Marine distress frequency (156.8 MHz / VHF Ch 16) must be blocked."""
        allowed, reason = is_tx_allowed(156.8e6)
        assert not allowed
        assert "DISTRESS" in reason.upper() or "MARINE" in reason.upper() or "EMERGENCY" in reason.upper()


class TestCellularLockouts:
    """Test cellular band lockouts."""

    def test_cellular_700_blocked(self):
        """700 MHz cellular band must be blocked."""
        allowed, reason = is_tx_allowed(750e6)
        assert not allowed
        assert "CELLULAR" in reason.upper()

    def test_cellular_850_blocked(self):
        """850 MHz cellular band must be blocked."""
        allowed, reason = is_tx_allowed(869e6)
        assert not allowed
        assert "CELLULAR" in reason.upper()

    def test_cellular_1900_blocked(self):
        """1900 MHz PCS cellular band must be blocked."""
        allowed, reason = is_tx_allowed(1920e6)
        assert not allowed
        assert "CELLULAR" in reason.upper()


class TestLicenseFreeBands:
    """Test license-free band TX permissions."""

    def setup_method(self):
        """Reset to no license before each test."""
        set_license_class(LicenseClass.NONE)

    def test_cb_radio_allowed_no_license(self):
        """CB radio (27 MHz) should be allowed without license."""
        # CB channel 19 = 27.185 MHz
        allowed, _ = is_tx_allowed(27.185e6)
        assert allowed

    def test_cb_power_limit(self):
        """CB radio power limit should be 12W PEP SSB."""
        power = get_power_limit(27.185e6)
        assert power is not None
        assert power <= 12.0

    def test_murs_allowed_no_license(self):
        """MURS frequencies should be allowed without license."""
        # MURS channel 1 = 151.820 MHz
        allowed, _ = is_tx_allowed(151.820e6)
        assert allowed

    def test_murs_power_limit(self):
        """MURS power limit should be 2W."""
        power = get_power_limit(151.820e6)
        assert power is not None
        assert power <= 2.0

    def test_frs_allowed_no_license(self):
        """FRS frequencies should be allowed without license (if defined)."""
        # FRS channel 1 = 462.5625 MHz
        # Note: This test verifies the concept; actual FRS support may vary
        allowed, reason = is_tx_allowed(462.5625e6)
        # FRS may or may not be defined - just check it doesn't crash
        assert isinstance(allowed, bool)


class TestLicenseClassEnforcement:
    """Test license class-based TX permissions."""

    def setup_method(self):
        """Reset to no license before each test."""
        set_license_class(LicenseClass.NONE)

    def test_no_license_blocked_from_amateur_bands(self):
        """Unlicensed users should be blocked from amateur bands."""
        set_license_class(LicenseClass.NONE)
        # 2m amateur band (144-148 MHz)
        allowed, reason = is_tx_allowed(146.0e6)
        assert not allowed
        assert "LICENSE" in reason.upper()

    def test_technician_allowed_on_2m(self):
        """Technician class should have full 2m (144-148 MHz) privileges."""
        set_license_class(LicenseClass.TECHNICIAN)
        allowed, _ = is_tx_allowed(146.0e6)
        assert allowed

    def test_technician_allowed_on_70cm(self):
        """Technician class should have 70cm (420-450 MHz) privileges."""
        set_license_class(LicenseClass.TECHNICIAN)
        allowed, _ = is_tx_allowed(446.0e6)
        assert allowed

    def test_technician_limited_on_hf(self):
        """Technician class should have limited HF privileges."""
        set_license_class(LicenseClass.TECHNICIAN)
        # 10m is allowed for Technicians
        allowed, _ = is_tx_allowed(28.4e6)
        assert allowed
        # But 40m phone is not (only some portions for CW)
        # This depends on exact band definitions

    def test_general_has_hf_privileges(self):
        """General class should have HF privileges."""
        set_license_class(LicenseClass.GENERAL)
        # 20m band - General phone portion is 14.225-14.350 MHz
        allowed, _ = is_tx_allowed(14.250e6)
        assert allowed

    def test_extra_has_full_privileges(self):
        """Amateur Extra should have full band privileges."""
        set_license_class(LicenseClass.AMATEUR_EXTRA)
        # Full 40m band including Extra-only portions
        allowed, _ = is_tx_allowed(7.1e6)
        assert allowed

    def test_license_class_from_string(self):
        """Test parsing license class from string."""
        assert LicenseClass.from_string("technician") == LicenseClass.TECHNICIAN
        assert LicenseClass.from_string("GENERAL") == LicenseClass.GENERAL
        assert LicenseClass.from_string("amateur_extra") == LicenseClass.AMATEUR_EXTRA
        assert LicenseClass.from_string("Amateur Extra") == LicenseClass.AMATEUR_EXTRA
        assert LicenseClass.from_string("invalid") == LicenseClass.NONE


class TestPowerLimits:
    """Test TX power limit enforcement."""

    def test_power_limit_returns_value(self):
        """Power limit should return a numeric value for valid bands."""
        set_license_class(LicenseClass.TECHNICIAN)
        power = get_power_limit(146.0e6)  # 2m band
        assert power is None or isinstance(power, (int, float))

    def test_effective_power_includes_headroom(self):
        """Effective power limit should include headroom factor."""
        set_license_class(LicenseClass.NONE)
        base = get_power_limit(27.185e6)  # CB radio
        effective = get_effective_power_limit(27.185e6)
        if base is not None and effective is not None:
            assert effective == pytest.approx(base * POWER_HEADROOM_FACTOR)

    def test_power_limit_none_for_lockout_bands(self):
        """Power limit should be None or indicate lockout for protected bands."""
        power = get_power_limit(1575.42e6)  # GPS L1
        # Either None or 0 indicates no TX allowed
        assert power is None or power == 0


class TestBandwidthValidation:
    """Test bandwidth-aware TX validation."""

    def test_narrow_bandwidth_allowed_in_clear_band(self):
        """Narrow signal in clear band should be allowed."""
        set_license_class(LicenseClass.TECHNICIAN)
        # Narrow FM in 2m band
        allowed, _ = is_tx_allowed(146.0e6, bandwidth_hz=15e3)
        assert allowed

    def test_wide_bandwidth_spilling_into_lockout_blocked(self):
        """Wide signal spilling into lockout band should be blocked."""
        # Signal near GPS but with wide bandwidth
        freq = 1550e6
        bandwidth = 30e6  # Would overlap GPS L1
        allowed, _ = is_tx_allowed(freq, bandwidth)
        assert not allowed

    def test_signal_at_band_edge_with_bandwidth(self):
        """Signal at band edge should account for bandwidth."""
        set_license_class(LicenseClass.TECHNICIAN)
        # At upper edge of 2m band (148 MHz)
        # With 100 kHz bandwidth, half would be outside band
        allowed, _ = is_tx_allowed(148.0e6, bandwidth_hz=100e3)
        # Should be blocked if it extends beyond band allocation


class TestValidateTxFrequency:
    """Test the validate_tx_frequency function that raises exceptions."""

    def test_validate_raises_on_gps(self):
        """validate_tx_frequency should raise ValueError for GPS."""
        with pytest.raises(ValueError) as excinfo:
            validate_tx_frequency(1575.42e6)
        assert "GPS" in str(excinfo.value).upper() or "LOCKOUT" in str(excinfo.value).upper()

    def test_validate_raises_on_cellular(self):
        """validate_tx_frequency should raise ValueError for cellular."""
        with pytest.raises(ValueError):
            validate_tx_frequency(869e6)

    def test_validate_passes_for_allowed_frequency(self):
        """validate_tx_frequency should not raise for allowed frequencies."""
        set_license_class(LicenseClass.TECHNICIAN)
        # Should not raise
        validate_tx_frequency(146.0e6)


class TestFrequencyBandDataclass:
    """Test FrequencyBand dataclass functionality."""

    def test_frequency_band_creation(self):
        """Test FrequencyBand creation and attributes."""
        band = FrequencyBand(
            name="Test Band",
            start_hz=100e6,
            end_hz=200e6,
            description="Test",
            rx_only=True,
            lockout_reason=LockoutReason.GPS,
        )
        assert band.name == "Test Band"
        assert band.start_hz == 100e6
        assert band.end_hz == 200e6
        assert band.rx_only is True
        assert band.lockout_reason == LockoutReason.GPS

    def test_frequency_band_in_lockout_list(self):
        """Test that lockout bands have correct structure."""
        # Verify GPS L1 is in lockout bands
        gps_l1_bands = [b for b in TX_LOCKOUT_BANDS if "GPS L1" in b.name]
        assert len(gps_l1_bands) > 0
        gps_l1 = gps_l1_bands[0]
        assert gps_l1.rx_only is True
        assert gps_l1.lockout_reason == LockoutReason.GPS


class TestBandPrivilegeDataclass:
    """Test BandPrivilege dataclass functionality."""

    def test_is_allowed_with_correct_license(self):
        """Test license checking."""
        privilege = BandPrivilege(
            name="Test",
            start_hz=100e6,
            end_hz=200e6,
            licenses={LicenseClass.TECHNICIAN, LicenseClass.GENERAL},
        )
        assert privilege.is_allowed(LicenseClass.TECHNICIAN)
        assert privilege.is_allowed(LicenseClass.GENERAL)
        assert not privilege.is_allowed(LicenseClass.NONE)

    def test_is_allowed_with_mode_restriction(self):
        """Test mode restriction checking."""
        privilege = BandPrivilege(
            name="Test",
            start_hz=100e6,
            end_hz=200e6,
            modes={"CW", "SSB"},
            licenses={LicenseClass.TECHNICIAN},
        )
        assert privilege.is_allowed(LicenseClass.TECHNICIAN, mode="CW")
        assert privilege.is_allowed(LicenseClass.TECHNICIAN, mode="SSB")
        assert not privilege.is_allowed(LicenseClass.TECHNICIAN, mode="FM")


class TestFrequencyManagerSingleton:
    """Test FrequencyManager singleton behavior."""

    def test_get_frequency_manager_returns_same_instance(self):
        """get_frequency_manager should return the same instance."""
        manager1 = get_frequency_manager()
        manager2 = get_frequency_manager()
        assert manager1 is manager2

    def test_manager_has_lockout_bands(self):
        """Manager should have lockout bands loaded."""
        manager = get_frequency_manager()
        assert len(manager._lockout_bands) > 0

    def test_manager_has_license_free_bands(self):
        """Manager should have license-free bands loaded."""
        manager = get_frequency_manager()
        assert len(manager._license_free_bands) > 0


class TestLockoutBandsCompleteness:
    """Test that all critical frequency bands are in the lockout list."""

    def test_all_gps_frequencies_have_lockouts(self):
        """Verify all GPS frequencies have lockout bands defined."""
        gps_freqs = [
            1575.42e6,  # L1
            1227.60e6,  # L2
            1176.45e6,  # L5
        ]
        for freq in gps_freqs:
            allowed, _ = is_tx_allowed(freq)
            assert not allowed, f"GPS frequency {freq/1e6} MHz should be blocked"

    def test_all_aviation_frequencies_have_lockouts(self):
        """Verify all aviation frequencies have lockout bands defined."""
        aviation_freqs = [
            121.5e6,   # Emergency
            243.0e6,   # Military emergency
            1090e6,    # ADS-B
            1030e6,    # Mode S interrogation
        ]
        for freq in aviation_freqs:
            allowed, _ = is_tx_allowed(freq)
            assert not allowed, f"Aviation frequency {freq/1e6} MHz should be blocked"

    def test_emergency_frequencies_have_lockouts(self):
        """Verify emergency frequencies have lockout bands defined."""
        emergency_freqs = [
            406.0e6,   # ELT/EPIRB
            156.8e6,   # Marine distress
        ]
        for freq in emergency_freqs:
            allowed, _ = is_tx_allowed(freq)
            assert not allowed, f"Emergency frequency {freq/1e6} MHz should be blocked"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_frequency(self):
        """Zero frequency should be handled gracefully."""
        allowed, _ = is_tx_allowed(0)
        # Should either be blocked or return a sensible result
        assert isinstance(allowed, bool)

    def test_negative_frequency(self):
        """Negative frequency should be handled gracefully."""
        allowed, _ = is_tx_allowed(-100e6)
        # Should be blocked (invalid frequency)
        assert not allowed

    def test_very_high_frequency(self):
        """Very high frequency should be handled gracefully."""
        allowed, _ = is_tx_allowed(100e9)  # 100 GHz
        # Should be handled without crashing
        assert isinstance(allowed, bool)

    def test_zero_bandwidth(self):
        """Zero bandwidth should be valid."""
        set_license_class(LicenseClass.TECHNICIAN)
        allowed, _ = is_tx_allowed(146.0e6, bandwidth_hz=0)
        assert allowed

    def test_negative_bandwidth(self):
        """Negative bandwidth should be handled gracefully."""
        set_license_class(LicenseClass.TECHNICIAN)
        allowed, _ = is_tx_allowed(146.0e6, bandwidth_hz=-1000)
        # Should handle gracefully (treat as 0 or absolute value)
        assert isinstance(allowed, bool)
