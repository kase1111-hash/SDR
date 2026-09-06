"""Tests for the SSTV mode table."""

from sdr_module.ham.sstv import SSTV_MODES


class TestSSTVVisCodes:
    """The VIS-code table must match the SSTV standard."""

    # Standard PD-mode VIS codes.
    STANDARD_PD_CODES = {
        "PD 90": 99,  # 0x63
        "PD 120": 95,  # 0x5F
        "PD 160": 98,  # 0x62
        "PD 180": 96,  # 0x60
        "PD 240": 97,  # 0x61
        "PD 290": 94,  # 0x5E
    }

    def test_dict_keys_match_vis_codes(self):
        """Every table key must equal the spec's own vis_code."""
        for key, spec in SSTV_MODES.items():
            assert (
                spec.vis_code == key
            ), f"{spec.name}: key {key} != vis_code {spec.vis_code}"

    def test_pd_modes_use_standard_codes(self):
        """PD 90 and PD 290 were swapped/wrong (93/99 instead of 99/94)."""
        by_name = {spec.name: key for key, spec in SSTV_MODES.items()}
        for name, code in self.STANDARD_PD_CODES.items():
            assert by_name.get(name) == code, f"{name} should use VIS {code}"

    def test_pd90_and_pd290_lookups(self):
        """A real PD90 (VIS 99) and PD290 (VIS 94) resolve to the right mode."""
        assert SSTV_MODES[99].name == "PD 90"
        assert SSTV_MODES[99].width == 320
        assert SSTV_MODES[94].name == "PD 290"
        assert SSTV_MODES[94].width == 800
