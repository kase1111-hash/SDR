"""Tests for AM/FM Radio Tuner widget."""

from sdr_module.gui.radio_tuner import (
    AM_RANGE,
    FM_RANGE,
    RadioBand,
    RadioPreset,
)


class TestRadioBand:
    """Tests for RadioBand enum."""

    def test_am_band_value(self):
        """Test AM band has correct value."""
        assert RadioBand.AM.value == "AM"

    def test_fm_band_value(self):
        """Test FM band has correct value."""
        assert RadioBand.FM.value == "FM"


class TestRadioPreset:
    """Tests for RadioPreset dataclass."""

    def test_create_fm_preset(self):
        """Test creating FM preset."""
        preset = RadioPreset(101.1e6, RadioBand.FM, "Rock Station")
        assert preset.frequency_hz == 101.1e6
        assert preset.band == RadioBand.FM
        assert preset.name == "Rock Station"

    def test_create_am_preset(self):
        """Test creating AM preset."""
        preset = RadioPreset(880e3, RadioBand.AM, "News")
        assert preset.frequency_hz == 880e3
        assert preset.band == RadioBand.AM
        assert preset.name == "News"

    def test_preset_default_name(self):
        """Test preset with default empty name."""
        preset = RadioPreset(95.5e6, RadioBand.FM)
        assert preset.name == ""


class TestFrequencyRanges:
    """Tests for frequency range constants."""

    def test_am_range_valid(self):
        """Test AM range is valid (530 kHz - 1700 kHz)."""
        assert AM_RANGE[0] == 530e3
        assert AM_RANGE[1] == 1700e3
        assert AM_RANGE[0] < AM_RANGE[1]

    def test_fm_range_valid(self):
        """Test FM range is valid (87.5 MHz - 108 MHz)."""
        assert FM_RANGE[0] == 87.5e6
        assert FM_RANGE[1] == 108e6
        assert FM_RANGE[0] < FM_RANGE[1]

    def test_ranges_dont_overlap(self):
        """Test AM and FM ranges don't overlap."""
        assert AM_RANGE[1] < FM_RANGE[0]


class TestRadioTunerImport:
    """Tests for radio tuner module imports."""

    def test_import_radio_tuner_widget(self):
        """Test RadioTunerWidget can be imported."""
        from sdr_module.gui.radio_tuner import RadioTunerWidget

        assert RadioTunerWidget is not None

    def test_import_show_radio_tuner(self):
        """Test show_radio_tuner function exists."""
        from sdr_module.gui.radio_tuner import show_radio_tuner

        assert callable(show_radio_tuner)

    def test_import_frequency_display(self):
        """Test FrequencyDisplay can be imported."""
        from sdr_module.gui.radio_tuner import FrequencyDisplay

        assert FrequencyDisplay is not None

    def test_import_tuning_dial(self):
        """Test TuningDial can be imported."""
        from sdr_module.gui.radio_tuner import TuningDial

        assert TuningDial is not None

    def test_import_preset_button(self):
        """Test PresetButton can be imported."""
        from sdr_module.gui.radio_tuner import PresetButton

        assert PresetButton is not None

    def test_import_volume_knob(self):
        """Test VolumeKnob can be imported."""
        from sdr_module.gui.radio_tuner import VolumeKnob

        assert VolumeKnob is not None


class TestRadioTunerDefaults:
    """Tests for default preset values."""

    def test_default_fm_presets_count(self):
        """Test default FM presets list has correct count."""
        from sdr_module.gui.radio_tuner import RadioTunerWidget

        assert len(RadioTunerWidget.DEFAULT_FM_PRESETS) == 6

    def test_default_am_presets_count(self):
        """Test default AM presets list has correct count."""
        from sdr_module.gui.radio_tuner import RadioTunerWidget

        assert len(RadioTunerWidget.DEFAULT_AM_PRESETS) == 6

    def test_default_fm_presets_in_range(self):
        """Test all default FM presets are within FM range."""
        from sdr_module.gui.radio_tuner import RadioTunerWidget

        for preset in RadioTunerWidget.DEFAULT_FM_PRESETS:
            assert FM_RANGE[0] <= preset.frequency_hz <= FM_RANGE[1]
            assert preset.band == RadioBand.FM

    def test_default_am_presets_in_range(self):
        """Test all default AM presets are within AM range."""
        from sdr_module.gui.radio_tuner import RadioTunerWidget

        for preset in RadioTunerWidget.DEFAULT_AM_PRESETS:
            assert AM_RANGE[0] <= preset.frequency_hz <= AM_RANGE[1]
            assert preset.band == RadioBand.AM


class TestGuiExports:
    """Tests for GUI module exports."""

    def test_radio_tuner_in_gui_exports(self):
        """Test RadioTunerWidget is exported from gui module."""
        from sdr_module.gui import RadioTunerWidget

        assert RadioTunerWidget is not None

    def test_radio_band_in_gui_exports(self):
        """Test RadioBand is exported from gui module."""
        from sdr_module.gui import RadioBand

        assert RadioBand is not None

    def test_radio_preset_in_gui_exports(self):
        """Test RadioPreset is exported from gui module."""
        from sdr_module.gui import RadioPreset

        assert RadioPreset is not None

    def test_show_radio_tuner_in_gui_exports(self):
        """Test show_radio_tuner is exported from gui module."""
        from sdr_module.gui import show_radio_tuner

        assert callable(show_radio_tuner)
