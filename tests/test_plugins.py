"""Tests for plugin system."""

import pytest

from sdr_module.plugins import (
    DemodulatorPlugin,
    DevicePlugin,
    PluginManager,
    PluginRegistry,
    ProcessorPlugin,
    ProtocolPlugin,
)
from sdr_module.plugins.base import (
    PluginError,
    PluginInitError,
    PluginLoadError,
    PluginMetadata,
    PluginNotFoundError,
    PluginState,
    PluginType,
)


class TestPluginType:
    """Tests for PluginType enum."""

    def test_all_types_exist(self):
        """Test all plugin types are defined."""
        assert PluginType.PROTOCOL is not None
        assert PluginType.DEMODULATOR is not None
        assert PluginType.DEVICE is not None
        assert PluginType.PROCESSOR is not None
        assert PluginType.UI_WIDGET is not None

    def test_type_values_unique(self):
        """Test all type values are unique."""
        values = [t.value for t in PluginType]
        assert len(values) == len(set(values))


class TestPluginState:
    """Tests for PluginState enum."""

    def test_all_states_exist(self):
        """Test all plugin states are defined."""
        assert PluginState.DISCOVERED is not None
        assert PluginState.LOADED is not None
        assert PluginState.INITIALIZED is not None
        assert PluginState.ENABLED is not None
        assert PluginState.DISABLED is not None
        assert PluginState.ERROR is not None

    def test_state_values_unique(self):
        """Test all state values are unique."""
        values = [s.value for s in PluginState]
        assert len(values) == len(set(values))


class TestPluginErrors:
    """Tests for plugin error classes."""

    def test_plugin_error(self):
        """Test base PluginError."""
        error = PluginError("Test error", plugin_name="test_plugin")
        assert str(error) == "Test error"
        assert error.plugin_name == "test_plugin"
        assert error.cause is None

    def test_plugin_error_with_cause(self):
        """Test PluginError with cause."""
        cause = ValueError("Original error")
        error = PluginError("Wrapped error", cause=cause)
        assert error.cause is cause

    def test_plugin_load_error(self):
        """Test PluginLoadError."""
        error = PluginLoadError("Load failed", plugin_name="bad_plugin")
        assert isinstance(error, PluginError)
        assert error.plugin_name == "bad_plugin"

    def test_plugin_init_error(self):
        """Test PluginInitError."""
        error = PluginInitError("Init failed")
        assert isinstance(error, PluginError)

    def test_plugin_not_found_error(self):
        """Test PluginNotFoundError."""
        error = PluginNotFoundError("Not found", plugin_name="missing")
        assert isinstance(error, PluginError)


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_basic_metadata(self):
        """Test creating basic metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            plugin_type=PluginType.PROTOCOL,
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.PROTOCOL

    def test_full_metadata(self):
        """Test creating full metadata."""
        metadata = PluginMetadata(
            name="full_plugin",
            version="2.0.0",
            plugin_type=PluginType.DEMODULATOR,
            author="Test Author",
            description="A test plugin",
            dependencies=["dep1", "dep2"],
            min_api_version="1.5.0",
            tags=["test", "example"],
            homepage="https://example.com",
            license="MIT",
        )
        assert metadata.author == "Test Author"
        assert metadata.description == "A test plugin"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.min_api_version == "1.5.0"
        assert "test" in metadata.tags
        assert metadata.homepage == "https://example.com"
        assert metadata.license == "MIT"

    def test_default_values(self):
        """Test default values are set."""
        metadata = PluginMetadata(
            name="minimal",
            version="1.0.0",
            plugin_type=PluginType.PROCESSOR,
        )
        assert metadata.author == ""
        assert metadata.description == ""
        assert metadata.dependencies == []
        assert metadata.min_api_version == "1.0.0"
        assert metadata.tags == []

    def test_to_dict(self):
        """Test converting to dictionary."""
        metadata = PluginMetadata(
            name="dict_test",
            version="1.0.0",
            plugin_type=PluginType.PROTOCOL,
            author="Author",
        )
        result = metadata.to_dict()

        assert result["name"] == "dict_test"
        assert result["version"] == "1.0.0"
        assert result["plugin_type"] == "PROTOCOL"
        assert result["author"] == "Author"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "from_dict_test",
            "version": "2.0.0",
            "plugin_type": "DEMODULATOR",
            "author": "Dict Author",
            "description": "Created from dict",
        }
        metadata = PluginMetadata.from_dict(data)

        assert metadata.name == "from_dict_test"
        assert metadata.version == "2.0.0"
        assert metadata.plugin_type == PluginType.DEMODULATOR
        assert metadata.author == "Dict Author"

    def test_from_dict_with_enum(self):
        """Test from_dict with PluginType enum."""
        data = {
            "name": "enum_test",
            "plugin_type": PluginType.DEVICE,
        }
        metadata = PluginMetadata.from_dict(data)
        assert metadata.plugin_type == PluginType.DEVICE

    def test_roundtrip(self):
        """Test dict roundtrip conversion."""
        original = PluginMetadata(
            name="roundtrip",
            version="3.0.0",
            plugin_type=PluginType.UI_WIDGET,
            author="RT Author",
            tags=["a", "b"],
        )
        data = original.to_dict()
        recovered = PluginMetadata.from_dict(data)

        assert recovered.name == original.name
        assert recovered.version == original.version
        assert recovered.plugin_type == original.plugin_type
        assert recovered.author == original.author
        assert recovered.tags == original.tags


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry instance."""
        return PluginRegistry()

    def test_registry_creation(self, registry):
        """Test registry can be created."""
        assert registry is not None

    def test_registry_initially_empty(self, registry):
        """Test registry starts empty."""
        assert len(registry.list_plugins()) == 0


class TestPluginManager:
    """Tests for PluginManager."""

    @pytest.fixture
    def manager(self):
        """Create plugin manager instance."""
        return PluginManager()

    def test_manager_creation(self, manager):
        """Test manager can be created."""
        assert manager is not None


class TestPluginBaseClasses:
    """Tests for plugin base classes."""

    def test_protocol_plugin_exists(self):
        """Test ProtocolPlugin class exists."""
        assert ProtocolPlugin is not None

    def test_demodulator_plugin_exists(self):
        """Test DemodulatorPlugin class exists."""
        assert DemodulatorPlugin is not None

    def test_device_plugin_exists(self):
        """Test DevicePlugin class exists."""
        assert DevicePlugin is not None

    def test_processor_plugin_exists(self):
        """Test ProcessorPlugin class exists."""
        assert ProcessorPlugin is not None


class TestConcretePlugin:
    """Tests using a concrete plugin implementation."""

    def test_create_simple_plugin(self):
        """Test creating a simple protocol plugin."""

        class TestProtocolPlugin(ProtocolPlugin):
            @classmethod
            def get_metadata(cls):
                return PluginMetadata(
                    name="test_protocol",
                    version="1.0.0",
                    plugin_type=PluginType.PROTOCOL,
                    description="Test protocol decoder",
                )

            def initialize(self, config):
                self._config = config
                return True

            def cleanup(self):
                pass

            def decode(self, samples):
                return []

            def get_protocol_info(self):
                return {"name": "test", "protocol_type": "test"}

            def can_decode(self, samples):
                return 0.0

        plugin = TestProtocolPlugin()
        metadata = plugin.get_metadata()

        assert metadata.name == "test_protocol"
        assert metadata.plugin_type == PluginType.PROTOCOL

    def test_plugin_state_tracking(self):
        """Test plugin state is tracked."""

        class StateTestPlugin(ProcessorPlugin):
            @classmethod
            def get_metadata(cls):
                return PluginMetadata(
                    name="state_test",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config):
                return True

            def cleanup(self):
                pass

            def process(self, samples):
                return samples

            def get_processor_info(self):
                return {"name": "state_test", "category": "test"}

        plugin = StateTestPlugin()
        # Initial state after creation should be LOADED
        assert plugin._state == PluginState.LOADED

    def test_plugin_config_storage(self):
        """Test plugin stores configuration."""

        class ConfigTestPlugin(ProcessorPlugin):
            @classmethod
            def get_metadata(cls):
                return PluginMetadata(
                    name="config_test",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR,
                )

            def initialize(self, config):
                self._config = config
                return True

            def cleanup(self):
                pass

            def process(self, samples):
                return samples

            def get_processor_info(self):
                return {"name": "config_test", "category": "test"}

        plugin = ConfigTestPlugin()
        test_config = {"sample_rate": 48000, "gain": 10}
        plugin.initialize(test_config)

        assert plugin._config == test_config
