"""Round-3 review fixes in non-GUI code that the GUI shows.

- ADS-B: noise no longer produces "aircraft"; replies are validated the way
  dump1090 does it (CRC residual, recovered addresses of known aircraft).
- RTL-SDR: set_bandwidth sets the tuner IF filter and never the sample rate.
- Frequency manager: presets, and higher license classes keep every
  lower-class privilege.
- Tooltips: "sample rate" (the span) and "bandwidth" (the channel) differ.
"""

import sys
import types

import numpy as np
import pytest

from sdr_module.core.frequency_manager import (
    AMATEUR_BAND_PRIVILEGES,
    FrequencyManager,
    LicenseClass,
)
from sdr_module.devices.rtlsdr import RTLSDRDevice
from sdr_module.dsp.protocols import (
    ADSBDecoder,
    ProtocolType,
    demodulate_for_protocol,
)
from sdr_module.utils.tooltips import get_detailed_tip, get_short_tip

# ---------------------------------------------------------------------------
# ADS-B
# ---------------------------------------------------------------------------

# Published example frames ("The 1090 MHz Riddle" / pyModeS test vectors).
IDENT_KLM1023 = "8D4840D6202CC371C32CE0576098"  # DF17 TC4, ICAO 4840D6
POS_ODD = "8D40621D58C386435CC412692AD6"  # DF17 TC11, ICAO 40621D, odd CPR
POS_EVEN = "8D40621D58C382D690C8AC2863A7"  # DF17 TC11, ICAO 40621D, even CPR

_PREAMBLE = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
_FLOOR = 0.05


def _frame_slots(hex_frame: str, level: float = 1.0) -> np.ndarray:
    """A Mode S frame as 0.5 us slots (2 MHz): preamble + PPM bits."""
    n_bits = len(hex_frame) * 4
    bits = [int(b) for b in bin(int(hex_frame, 16))[2:].zfill(n_bits)]
    slots = list(_PREAMBLE)
    for bit in bits:
        slots += (1, 0) if bit else (0, 1)
    return _FLOOR + level * np.asarray(slots, dtype=np.float32)


def _signal(*frames: str, gap: int = 300) -> np.ndarray:
    """Frames separated by quiet gaps, at 2 MHz."""
    quiet = np.full(gap, _FLOOR, dtype=np.float32)
    parts = [quiet]
    for frame in frames:
        parts += [_frame_slots(frame), quiet]
    return np.concatenate(parts)


def _at_2_4_mhz(slots: np.ndarray) -> np.ndarray:
    """The 2 MHz slot signal as a 2.4 MS/s capture (each sample averages
    its 1/2.4 us period, like an ADC's anti-alias filter)."""
    fine = np.repeat(slots, 6)  # 12 MHz
    fine = fine[: len(fine) // 5 * 5]
    return fine.reshape(-1, 5).mean(axis=1)


def _crc(data: bytes) -> int:
    return ADSBDecoder(2e6)._compute_crc(data)


def _address_parity(df: int, icao: int, field13: int, first3: int = 0) -> str:
    """A 56-bit DF0/4/5 reply whose parity carries ``icao`` (CRC XOR AA)."""
    word = (df << 27) | (first3 << 24) | (0 << 13) | (field13 & 0x1FFF)
    payload = word.to_bytes(4, "big")
    parity = _crc(payload) ^ icao
    return (payload + parity.to_bytes(3, "big")).hex().upper()


def _all_call(icao: int, iid: int = 0, ca: int = 5) -> str:
    """A 56-bit DF11 all-call reply; ``iid`` is overlaid on the parity."""
    payload = bytes([(11 << 3) | ca]) + icao.to_bytes(3, "big")
    parity = _crc(payload) ^ iid
    return (payload + parity.to_bytes(3, "big")).hex().upper()


def _ac13(altitude_ft: int) -> int:
    """25 ft AC13 code (M=0, Q=1) for ``altitude_ft``."""
    n = (altitude_ft + 1000) // 25
    return ((n & 0x7E0) << 2) | ((n & 0x010) << 1) | 0x10 | (n & 0x0F)


class TestADSBNoise:
    def test_one_second_of_receiver_noise_yields_no_aircraft(self):
        """The GUI path: complex noise at the 2.4 MS/s band-preset rate."""
        rng = np.random.default_rng(2026)
        rate = 2.4e6
        decoder = ADSBDecoder(rate)
        messages = []
        block = 262144  # the RTL-SDR driver's read size
        for _ in range(int(rate) // block + 1):
            iq = (rng.standard_normal(block) + 1j * rng.standard_normal(block)) / 3
            baseband = demodulate_for_protocol(iq, ProtocolType.ADSB)
            messages += decoder.decode(baseband)
        assert len(messages) <= 1, [m.icao_address for m in messages]

    def test_one_second_of_real_noise_at_2_mhz(self):
        rng = np.random.default_rng(5)
        decoder = ADSBDecoder(2e6)
        messages = decoder.decode(rng.standard_normal(2_000_000).astype(np.float32))
        assert len(messages) <= 1

    def test_decoding_a_second_is_fast(self):
        """The live decoder must keep up (it used to take ~9 s per second)."""
        import time

        rng = np.random.default_rng(1)
        decoder = ADSBDecoder(2e6)
        noise = np.abs(
            rng.standard_normal(2_000_000) + 1j * rng.standard_normal(2_000_000)
        )
        started = time.perf_counter()
        decoder.decode(noise)
        assert time.perf_counter() - started < 1.0

    def test_random_frames_after_a_valid_preamble_are_dropped(self):
        rng = np.random.default_rng(11)
        decoder = ADSBDecoder(2e6)
        for _ in range(200):
            frame = rng.integers(0, 256, 14, dtype=np.uint8).tobytes().hex()
            assert decoder.decode(_signal(frame)) == []


class TestADSBValidation:
    def test_extended_squitter_decodes(self):
        decoder = ADSBDecoder(2e6)
        received = []
        decoder.add_callback(received.append)
        (msg,) = decoder.decode(_signal(IDENT_KLM1023))
        assert msg.valid is True
        assert msg.downlink_format == 17
        assert msg.icao_address == "4840D6"
        assert msg.type_code == 4
        assert msg.callsign == "KLM1023"
        # The raw column shows the frame in the usual hex form.
        assert msg.raw_bits.hex().upper() == IDENT_KLM1023
        assert received == [msg]

    def test_one_bit_error_in_an_extended_squitter_is_dropped(self):
        corrupted = f"{int(IDENT_KLM1023, 16) ^ (1 << 40):028X}"
        assert ADSBDecoder(2e6).decode(_signal(corrupted)) == []

    def test_airborne_position_pair_decodes(self):
        decoder = ADSBDecoder(2e6)
        messages = decoder.decode(_signal(POS_ODD, POS_EVEN))
        assert [m.icao_address for m in messages] == ["40621D", "40621D"]
        assert messages[-1].altitude == 38000
        assert messages[-1].latitude == pytest.approx(52.2572, abs=1e-4)
        assert messages[-1].longitude == pytest.approx(3.91937, abs=1e-4)

    def test_address_parity_reply_needs_a_known_aircraft(self):
        icao = 0x4840D6
        reply = _address_parity(4, icao, _ac13(38000))
        decoder = ADSBDecoder(2e6)
        # Unknown address: CRC XOR parity gives *some* address for any
        # frame, so on its own this proves nothing.
        assert decoder.decode(_signal(reply)) == []
        # After a CRC-checked squitter from that aircraft it is accepted.
        decoder.decode(_signal(IDENT_KLM1023))
        (msg,) = decoder.decode(_signal(reply))
        assert (msg.downlink_format, msg.icao_address) == (4, "4840D6")
        assert msg.altitude == 38000
        assert msg.valid is True
        # A different aircraft's address is still refused.
        assert decoder.decode(_signal(_address_parity(4, 0x123456, 0))) == []

    def test_known_aircraft_expire(self):
        decoder = ADSBDecoder(2e6)
        decoder.ICAO_TIMEOUT_S = 0.005
        decoder.decode(_signal(IDENT_KLM1023))
        decoder.decode(np.full(20_000, _FLOOR, dtype=np.float32))  # 10 ms
        reply = _address_parity(4, 0x4840D6, _ac13(1000))
        assert decoder.decode(_signal(reply)) == []

    def test_squawk_from_identity_reply(self):
        decoder = ADSBDecoder(2e6)
        decoder.decode(_signal(IDENT_KLM1023))
        # 7000 -> A=7 B=0 C=0 D=0: A4 A2 A1 at bits 7, 9, 11.
        id13 = (1 << 7) | (1 << 9) | (1 << 11)
        (msg,) = decoder.decode(_signal(_address_parity(5, 0x4840D6, id13)))
        assert msg.downlink_format == 5
        assert msg.squawk == "7000"

    def test_field_decoders_match_published_vectors(self):
        decoder = ADSBDecoder(2e6)
        assert decoder._decode_ac13(bytes.fromhex("A02014B400000000000000F9D514")) == (
            32300
        )
        assert decoder._decode_squawk(
            bytes.fromhex("A800292DFFBBA9383FFCEB903D01")
        ) == ("1346")

    def test_all_call_reply(self):
        decoder = ADSBDecoder(2e6)
        # A non-zero interrogator code from an unknown aircraft is refused.
        assert decoder.decode(_signal(_all_call(0xABCDEF, iid=0x16))) == []
        # IID 0 is self-checking and makes the aircraft known ...
        (msg,) = decoder.decode(_signal(_all_call(0xABCDEF)))
        assert (msg.downlink_format, msg.icao_address) == (11, "ABCDEF")
        # ... after which replies to other interrogators are accepted.
        (msg,) = decoder.decode(_signal(_all_call(0xABCDEF, iid=0x16)))
        assert msg.icao_address == "ABCDEF"
        # A residual above the low 7 bits is a CRC error.
        assert decoder.decode(_signal(_all_call(0xABCDEF, iid=0x80))) == []
        # The DF11 made the address known for address/parity replies too.
        assert decoder.decode(_signal(_address_parity(0, 0xABCDEF, 0)))

    def test_unsupported_format_is_dropped(self):
        payload = bytes([(19 << 3) | 0, 0x48, 0x40, 0xD6]) + bytes(7)
        frame = payload + _crc(payload).to_bytes(3, "big")
        assert ADSBDecoder(2e6).decode(_signal(frame.hex())) == []

    def test_reset_forgets_known_aircraft(self):
        decoder = ADSBDecoder(2e6)
        decoder.decode(_signal(IDENT_KLM1023))
        decoder.reset()
        assert decoder.decode(_signal(_address_parity(4, 0x4840D6, 0))) == []


class TestADSBStream:
    def test_frame_split_across_blocks_is_decoded_once(self):
        signal = _signal(IDENT_KLM1023, POS_ODD, POS_EVEN)
        for block in (97, 250, 1000):
            decoder = ADSBDecoder(2e6)
            messages = []
            for start in range(0, len(signal), block):
                messages += decoder.decode(signal[start : start + block])
            assert [m.icao_address for m in messages] == [
                "4840D6",
                "40621D",
                "40621D",
            ], block

    def test_decodes_at_2_4_msps(self):
        """The Band Presets menu's ADS-B entry uses 2.4 MS/s."""
        capture = _at_2_4_mhz(_signal(IDENT_KLM1023, POS_ODD))
        decoder = ADSBDecoder(2.4e6)
        messages = decoder.decode(capture)
        assert [m.callsign for m in messages][:1] == ["KLM1023"]
        assert [m.icao_address for m in messages] == ["4840D6", "40621D"]

    def test_2_4_msps_in_odd_sized_blocks(self):
        capture = _at_2_4_mhz(_signal(IDENT_KLM1023, POS_ODD))
        decoder = ADSBDecoder(2.4e6)
        messages = []
        for start in range(0, len(capture), 333):
            messages += decoder.decode(capture[start : start + 333])
        assert [m.icao_address for m in messages] == ["4840D6", "40621D"]

    def test_complex_input_is_taken_as_magnitude(self):
        signal = _signal(IDENT_KLM1023).astype(np.complex64) * np.exp(1j * 0.7)
        (msg,) = ADSBDecoder(2e6).decode(signal)
        assert msg.callsign == "KLM1023"

    def test_non_finite_samples_do_not_crash(self):
        decoder = ADSBDecoder(2.4e6)
        bad = np.array([np.nan, np.inf, -np.inf] * 1000, dtype=np.float32)
        assert decoder.decode(bad) == []
        assert decoder.decode(_at_2_4_mhz(_signal(IDENT_KLM1023)))


# ---------------------------------------------------------------------------
# RTL-SDR set_bandwidth
# ---------------------------------------------------------------------------


class _FakeRtlSdr:
    """Minimal pyrtlsdr stand-in; ``with_bandwidth`` adds set_bandwidth."""

    def __init__(self, device_index=0):
        self.sample_rate = 2.048e6
        self.center_freq = 100e6
        self.gain = "auto"
        self.tuner_bandwidth = None

    @staticmethod
    def get_device_serial_addresses():
        return ["00000001"]

    def close(self):
        pass


class _FakeRtlSdrWithBandwidth(_FakeRtlSdr):
    def set_bandwidth(self, bw):
        self.tuner_bandwidth = bw


class _FakeRtlSdrOldLibrary(_FakeRtlSdr):
    def set_bandwidth(self, bw):
        # pyrtlsdr has the method, librtlsdr lacks rtlsdr_set_tuner_bandwidth.
        raise AttributeError("undefined symbol: rtlsdr_set_tuner_bandwidth")


def _open_rtlsdr(monkeypatch, fake_class) -> RTLSDRDevice:
    module = types.ModuleType("rtlsdr")
    module.RtlSdr = fake_class
    monkeypatch.setitem(sys.modules, "rtlsdr", module)
    device = RTLSDRDevice()
    assert device.open()
    return device


class TestRtlSdrBandwidth:
    def test_sets_the_tuner_filter_not_the_sample_rate(self, monkeypatch):
        device = _open_rtlsdr(monkeypatch, _FakeRtlSdrWithBandwidth)
        assert device.set_bandwidth(25e3) is True
        assert device._device.tuner_bandwidth == 25000
        assert device._device.sample_rate == 2.4e6
        assert device.state.sample_rate == 2.4e6
        assert device.state.bandwidth == 25e3
        # A later rate change keeps the explicit filter width.
        device.set_sample_rate(2.048e6)
        assert device.state.bandwidth == 25e3
        # 0 goes back to automatic (the filter follows the rate).
        assert device.set_bandwidth(0) is True
        assert device.state.bandwidth == 2.048e6

    def test_unsupported_library_is_a_no_op(self, monkeypatch):
        for fake in (_FakeRtlSdr, _FakeRtlSdrOldLibrary):
            device = _open_rtlsdr(monkeypatch, fake)
            assert device.set_bandwidth(25e3) is False
            assert device._device.sample_rate == 2.4e6
            assert device.state.sample_rate == 2.4e6
            assert device.state.bandwidth == 2.4e6
            # Asking for what the automatic filter does is not a failure
            # (DeviceManager.apply_config sets bandwidth = sample rate).
            assert device.set_bandwidth(2.4e6) is True
            assert device.set_bandwidth(0) is True
            assert device._device.sample_rate == 2.4e6

    def test_closed_device_and_bad_width(self, monkeypatch):
        assert RTLSDRDevice().set_bandwidth(1e6) is False
        device = _open_rtlsdr(monkeypatch, _FakeRtlSdrWithBandwidth)
        assert device.set_bandwidth(-1) is False
        assert device._device.tuner_bandwidth is None


# ---------------------------------------------------------------------------
# Frequency manager
# ---------------------------------------------------------------------------

_HAM = (LicenseClass.TECHNICIAN, LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA)


def _manager(license_class: LicenseClass) -> FrequencyManager:
    manager = FrequencyManager()
    manager.set_license_class(license_class)
    return manager


class TestPresets:
    def test_fm_broadcast_is_a_real_channel(self):
        preset = FrequencyManager().get_preset_by_name("FM Broadcast")
        assert preset.frequency_hz == 100.1e6

    def test_adsb_preset_points_at_the_decoder(self):
        preset = FrequencyManager().get_preset_by_name("ADS-B")
        assert "dump1090" not in preset.description
        assert "Decoder" in preset.description

    def test_preset_names_are_unique(self):
        names = [p.name for p in FrequencyManager().get_rx_presets()]
        assert len(names) == len(set(names))


class TestLicensePrivileges:
    def test_10m_phone_for_every_class(self):
        """Regression: 28.36 MHz USB was refused for General and Extra."""
        for license_class in _HAM:
            allowed, reason = _manager(license_class).is_tx_allowed(
                28.36e6, 2700, "USB"
            )
            assert allowed, (license_class, reason)

    def test_higher_classes_keep_every_lower_class_privilege(self):
        """Every segment granted to a class is open to the classes above it."""
        for band in AMATEUR_BAND_PRIVILEGES:
            centre = (band.start_hz + band.end_hz) / 2
            modes = sorted(band.modes) or [""]
            for lower_index, lower in enumerate(_HAM):
                if lower not in band.licenses:
                    continue
                for higher in _HAM[lower_index:]:
                    manager = _manager(higher)
                    for mode in modes:
                        allowed, reason = manager.is_tx_allowed(centre, 0, mode)
                        assert allowed, (band.name, higher, mode, reason)

    def test_classes_include_lower_ones(self):
        assert LicenseClass.AMATEUR_EXTRA.includes(LicenseClass.TECHNICIAN)
        assert LicenseClass.GENERAL.includes(LicenseClass.GENERAL)
        assert not LicenseClass.TECHNICIAN.includes(LicenseClass.GENERAL)
        assert not LicenseClass.NONE.includes(LicenseClass.TECHNICIAN)
        assert LicenseClass.NONE.includes(LicenseClass.NONE)
        assert not LicenseClass.GENERAL.includes(LicenseClass.NONE)

    def test_privilege_lists_grow_with_the_class(self):
        names = {
            lc: {b.name for b in _manager(lc).get_license_privileges()} for lc in _HAM
        }
        assert names[LicenseClass.TECHNICIAN] <= names[LicenseClass.GENERAL]
        assert names[LicenseClass.GENERAL] <= names[LicenseClass.AMATEUR_EXTRA]

    def test_own_class_sets_the_power_limit(self):
        # Technicians are limited to 200 W PEP on their HF segments ...
        tech = _manager(LicenseClass.TECHNICIAN)
        assert tech.get_power_limit(28.36e6, "USB") == 200.0
        assert tech.get_power_limit(28.06e6, "CW") == 200.0
        assert tech.get_power_limit(7.05e6, "CW") == 200.0
        # ... a General operator on the same frequencies is not.
        general = _manager(LicenseClass.GENERAL)
        assert general.get_power_limit(28.36e6, "USB") is None
        assert general.get_power_limit(28.06e6, "CW") is None
        assert general.get_power_limit(7.05e6, "CW") is None

    def test_technician_hf_limits(self):
        tech = _manager(LicenseClass.TECHNICIAN)
        assert not tech.is_tx_allowed(29.6e6, 15e3, "FM")[0]  # above 28.5 MHz
        assert not tech.is_tx_allowed(28.1e6, 2700, "USB")[0]  # data segment
        assert tech.is_tx_allowed(28.1e6, 0, "DATA")[0]
        allowed, reason = tech.is_tx_allowed(7.05e6, 2700, "USB")
        assert not allowed and "Mode 'USB' not allowed" in reason
        allowed, reason = tech.is_tx_allowed(14.25e6, 2700, "USB")
        assert not allowed and "higher license class" in reason

    def test_general_has_all_of_17m(self):
        general = _manager(LicenseClass.GENERAL)
        assert general.is_tx_allowed(18.08e6, 0, "CW")[0]
        assert general.is_tx_allowed(18.13e6, 2700, "USB")[0]

    def test_extra_only_segments_stay_extra_only(self):
        general = _manager(LicenseClass.GENERAL)
        for freq, mode in ((3.51e6, "CW"), (7.15e6, "USB"), (14.2e6, "USB")):
            allowed, reason = general.is_tx_allowed(freq, 0, mode)
            assert not allowed and "higher license class" in reason, freq
            assert _manager(LicenseClass.AMATEUR_EXTRA).is_tx_allowed(freq, 0, mode)[0]


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------


class TestTooltips:
    def test_sample_rate_is_the_span(self):
        assert get_short_tip("sample_rate") == (
            "Samples per second the SDR captures. Sets the span, the width of "
            "spectrum shown (2.4 MS/s = 2.4 MHz)."
        )
        assert "Nyquist" not in get_detailed_tip("sample_rate")

    def test_bandwidth_is_the_channel(self):
        assert get_short_tip("bandwidth") == (
            "Width of the channel around the tuned frequency that is "
            "demodulated. It does not change the span shown in the spectrum."
        )
        detailed = get_detailed_tip("bandwidth")
        # No advice the Bandwidth list cannot follow (it starts at 10 kHz).
        assert "2.4-3kHz" not in detailed
        assert "10 kHz" in detailed and "200 kHz" in detailed

    def test_bandwidth_advice_matches_the_options(self):
        pytest.importorskip("PyQt6")
        from sdr_module.gui.control_panel import BANDWIDTH_OPTIONS

        detailed = " ".join(get_detailed_tip("bandwidth").split())
        for width in ("10 kHz", "25 kHz", "200 kHz"):
            assert width in BANDWIDTH_OPTIONS
            assert width in detailed
