"""
Plugin manager for discovery, loading, and lifecycle management.

Handles the complete plugin lifecycle from discovery to unloading,
with support for multiple plugin directories and hot-reloading.
"""

import sys
import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field

from .base import (
    Plugin,
    PluginMetadata,
    PluginType,
    PluginLoadError,
    check_api_compatibility,
    PLUGIN_API_VERSION,
)
from .registry import PluginRegistry


logger = logging.getLogger(__name__)


@dataclass
class PluginSource:
    """Information about a discovered plugin source."""
    path: Path
    name: str
    is_package: bool
    metadata_file: Optional[Path] = None
    module_name: Optional[str] = None
    load_error: Optional[str] = None


@dataclass
class PluginConfig:
    """Configuration for plugin manager."""
    plugin_dirs: List[Path] = field(default_factory=list)
    auto_enable: bool = True
    strict_mode: bool = False  # Fail on any plugin error
    allow_reload: bool = True
    config_file: Optional[Path] = None


class PluginManager:
    """
    Central plugin manager.

    Handles plugin discovery, loading, initialization, and lifecycle
    management for all plugin types.

    Usage:
        manager = PluginManager()

        # Add plugin directories
        manager.add_plugin_dir("~/.sdr_module/plugins")
        manager.add_plugin_dir("./plugins")

        # Discover and load all plugins
        manager.discover_plugins()
        manager.load_all()

        # Use plugins
        protocols = manager.get_plugins_by_type(PluginType.PROTOCOL)
        for proto in protocols:
            if proto.is_enabled:
                frames = proto.decode(samples)
    """

    # Plugin entry point file names
    ENTRY_POINTS = ["plugin.py", "__init__.py"]
    METADATA_FILE = "plugin.json"

    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize plugin manager.

        Args:
            config: Optional configuration
        """
        self._config = config or PluginConfig()
        self._registry = PluginRegistry()
        self._sources: Dict[str, PluginSource] = {}
        self._loaded_modules: Dict[str, Any] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

        # Add default plugin directories
        self._init_default_dirs()

        # Load plugin configs if available
        if self._config.config_file and self._config.config_file.exists():
            self._load_plugin_configs()

    def _init_default_dirs(self) -> None:
        """Initialize default plugin directories."""
        # User plugin directory
        user_plugin_dir = Path.home() / ".sdr_module" / "plugins"
        if user_plugin_dir.exists():
            self._config.plugin_dirs.append(user_plugin_dir)

        # Local plugin directory
        local_plugin_dir = Path.cwd() / "plugins"
        if local_plugin_dir.exists():
            self._config.plugin_dirs.append(local_plugin_dir)

    def _load_plugin_configs(self) -> None:
        """Load plugin configurations from file."""
        try:
            with open(self._config.config_file, "r") as f:
                self._plugin_configs = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load plugin configs: {e}")

    def _save_plugin_configs(self) -> None:
        """Save plugin configurations to file."""
        if not self._config.config_file:
            return

        try:
            self._config.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config.config_file, "w") as f:
                json.dump(self._plugin_configs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save plugin configs: {e}")

    @property
    def registry(self) -> PluginRegistry:
        """Get the plugin registry."""
        return self._registry

    def add_plugin_dir(self, path: str) -> bool:
        """
        Add a plugin directory to search.

        Args:
            path: Path to plugin directory

        Returns:
            True if directory was added
        """
        plugin_dir = Path(path).expanduser().resolve()

        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return False

        if not plugin_dir.is_dir():
            logger.warning(f"Not a directory: {plugin_dir}")
            return False

        if plugin_dir not in self._config.plugin_dirs:
            self._config.plugin_dirs.append(plugin_dir)
            logger.info(f"Added plugin directory: {plugin_dir}")
            return True

        return False

    def remove_plugin_dir(self, path: str) -> bool:
        """
        Remove a plugin directory.

        Args:
            path: Path to remove

        Returns:
            True if directory was removed
        """
        plugin_dir = Path(path).expanduser().resolve()

        if plugin_dir in self._config.plugin_dirs:
            self._config.plugin_dirs.remove(plugin_dir)
            logger.info(f"Removed plugin directory: {plugin_dir}")
            return True

        return False

    def discover_plugins(self, path: Optional[str] = None) -> List[PluginSource]:
        """
        Discover plugins in configured directories.

        Args:
            path: Optional specific path to search (uses all dirs if None)

        Returns:
            List of discovered plugin sources
        """
        discovered = []

        dirs_to_search = [Path(path).expanduser().resolve()] if path else self._config.plugin_dirs

        for plugin_dir in dirs_to_search:
            if not plugin_dir.exists():
                continue

            # Look for plugin packages (directories with plugin.py or __init__.py)
            for item in plugin_dir.iterdir():
                if item.is_dir() and not item.name.startswith((".", "_")):
                    source = self._discover_package(item)
                    if source:
                        discovered.append(source)
                        self._sources[source.name] = source

                # Look for single-file plugins
                elif item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                    source = self._discover_file(item)
                    if source:
                        discovered.append(source)
                        self._sources[source.name] = source

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _discover_package(self, path: Path) -> Optional[PluginSource]:
        """Discover a plugin package."""
        # Look for entry point
        entry_point = None
        for ep in self.ENTRY_POINTS:
            candidate = path / ep
            if candidate.exists():
                entry_point = candidate
                break

        if not entry_point:
            return None

        # Look for metadata
        metadata_file = path / self.METADATA_FILE
        if not metadata_file.exists():
            metadata_file = None

        # Derive name from directory
        name = path.name

        return PluginSource(
            path=path,
            name=name,
            is_package=True,
            metadata_file=metadata_file,
            module_name=f"sdr_plugins.{name}",
        )

    def _discover_file(self, path: Path) -> Optional[PluginSource]:
        """Discover a single-file plugin."""
        name = path.stem

        return PluginSource(
            path=path,
            name=name,
            is_package=False,
            module_name=f"sdr_plugins.{name}",
        )

    def load_plugin(self, name: str) -> bool:
        """
        Load a discovered plugin.

        Args:
            name: Plugin name to load

        Returns:
            True if loaded successfully
        """
        if name not in self._sources:
            logger.error(f"Plugin not discovered: {name}")
            return False

        source = self._sources[name]

        try:
            # Load the module
            module = self._load_module(source)
            if not module:
                return False

            self._loaded_modules[name] = module

            # Find plugin classes in the module
            plugin_classes = self._find_plugin_classes(module)

            if not plugin_classes:
                logger.warning(f"No plugin classes found in: {name}")
                return False

            # Register all found plugin classes
            for plugin_class in plugin_classes:
                metadata = plugin_class.get_metadata()

                # Check API compatibility
                if not check_api_compatibility(metadata.min_api_version):
                    logger.error(
                        f"Plugin '{metadata.name}' requires API v{metadata.min_api_version}, "
                        f"but v{PLUGIN_API_VERSION} is installed"
                    )
                    if self._config.strict_mode:
                        return False
                    continue

                self._registry.register_class(plugin_class)

            logger.info(f"Loaded plugin: {name} ({len(plugin_classes)} classes)")
            return True

        except Exception as e:
            source.load_error = str(e)
            logger.error(f"Failed to load plugin '{name}': {e}")

            if self._config.strict_mode:
                raise PluginLoadError(str(e), name, e)

            return False

    def _load_module(self, source: PluginSource) -> Optional[Any]:
        """Load a plugin module."""
        try:
            if source.is_package:
                # Add parent directory to path
                parent_dir = str(source.path.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)

                # Load package
                spec = importlib.util.spec_from_file_location(
                    source.module_name,
                    source.path / "__init__.py" if (source.path / "__init__.py").exists()
                    else source.path / "plugin.py"
                )
            else:
                # Load single file
                spec = importlib.util.spec_from_file_location(
                    source.module_name,
                    source.path
                )

            if not spec or not spec.loader:
                logger.error(f"Could not create module spec for: {source.name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[source.module_name] = module
            spec.loader.exec_module(module)

            return module

        except Exception as e:
            logger.error(f"Module load error for '{source.name}': {e}")
            return None

    def _find_plugin_classes(self, module: Any) -> List[Type[Plugin]]:
        """Find all Plugin subclasses in a module."""
        plugin_classes = []

        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Check if it's a class that inherits from Plugin
            if (isinstance(obj, type) and
                issubclass(obj, Plugin) and
                obj is not Plugin and
                hasattr(obj, "get_metadata")):

                # Make sure it's not a base class
                try:
                    # Try to get metadata to verify it's a concrete implementation
                    metadata = obj.get_metadata()
                    if metadata and metadata.name:
                        plugin_classes.append(obj)
                except Exception:
                    # Skip abstract classes that don't implement get_metadata
                    pass

        return plugin_classes

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name to unload

        Returns:
            True if unloaded successfully
        """
        if name not in self._loaded_modules:
            return False

        try:
            # Remove from registry
            self._registry.unregister_class(name)

            # Remove module
            self._loaded_modules.pop(name)
            source = self._sources.get(name)

            if source and source.module_name in sys.modules:
                del sys.modules[source.module_name]

            logger.info(f"Unloaded plugin: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload plugin '{name}': {e}")
            return False

    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin.

        Args:
            name: Plugin name to reload

        Returns:
            True if reloaded successfully
        """
        if not self._config.allow_reload:
            logger.warning("Plugin reload is disabled")
            return False

        # Unload then load
        self.unload_plugin(name)
        return self.load_plugin(name)

    def load_all(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.

        Returns:
            Dictionary mapping plugin names to load success
        """
        results = {}

        for name in self._sources:
            results[name] = self.load_plugin(name)

        return results

    def unload_all(self) -> None:
        """Unload all plugins."""
        for name in list(self._loaded_modules.keys()):
            self.unload_plugin(name)

    def initialize_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """
        Initialize a plugin instance.

        Args:
            name: Plugin name
            config: Optional configuration

        Returns:
            Plugin instance or None on failure
        """
        # Merge with saved config
        merged_config = self._plugin_configs.get(name, {}).copy()
        if config:
            merged_config.update(config)

        instance = self._registry.create_instance(name, merged_config)

        if instance and self._config.auto_enable:
            instance.enable()

        return instance

    def initialize_all(
        self,
        plugin_type: Optional[PluginType] = None
    ) -> Dict[str, Plugin]:
        """
        Initialize all loaded plugins.

        Args:
            plugin_type: Optional filter by type

        Returns:
            Dictionary of initialized plugin instances
        """
        results = {}

        for metadata in self._registry.list_plugins(plugin_type=plugin_type):
            instance = self.initialize_plugin(metadata.name)
            if instance:
                results[metadata.name] = instance

        return results

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._registry.get_instance(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """
        Get all plugin instances of a type.

        Args:
            plugin_type: Plugin type to filter by

        Returns:
            List of plugin instances
        """
        plugins = []

        for name in self._registry.list_by_type(plugin_type):
            instance = self._registry.get_instance(name)
            if instance:
                plugins.append(instance)

        return plugins

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None
    ) -> List[PluginMetadata]:
        """
        List all registered plugins.

        Args:
            plugin_type: Optional filter by type

        Returns:
            List of plugin metadata
        """
        return self._registry.list_plugins(plugin_type=plugin_type)

    def list_discovered(self) -> List[PluginSource]:
        """
        List all discovered plugin sources.

        Returns:
            List of plugin sources
        """
        return list(self._sources.values())

    def set_plugin_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Set configuration for a plugin.

        Args:
            name: Plugin name
            config: Configuration dictionary
        """
        self._plugin_configs[name] = config
        self._save_plugin_configs()

    def get_plugin_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a plugin.

        Args:
            name: Plugin name

        Returns:
            Configuration dictionary
        """
        return self._plugin_configs.get(name, {})

    def get_status(self) -> Dict[str, Any]:
        """
        Get plugin system status.

        Returns:
            Status dictionary
        """
        return {
            "api_version": PLUGIN_API_VERSION,
            "plugin_dirs": [str(d) for d in self._config.plugin_dirs],
            "discovered": len(self._sources),
            "loaded": len(self._loaded_modules),
            "registry": self._registry.get_stats(),
            "sources": {
                name: {
                    "path": str(source.path),
                    "is_package": source.is_package,
                    "loaded": name in self._loaded_modules,
                    "error": source.load_error,
                }
                for name, source in self._sources.items()
            },
        }

    def create_plugin_template(
        self,
        name: str,
        plugin_type: PluginType,
        output_dir: Optional[str] = None
    ) -> Path:
        """
        Create a plugin template.

        Args:
            name: Plugin name
            plugin_type: Type of plugin to create
            output_dir: Output directory (uses first plugin dir if None)

        Returns:
            Path to created plugin directory
        """
        if output_dir:
            base_dir = Path(output_dir).expanduser().resolve()
        elif self._config.plugin_dirs:
            base_dir = self._config.plugin_dirs[0]
        else:
            base_dir = Path.cwd() / "plugins"

        plugin_dir = base_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Create plugin.json
        metadata = {
            "name": name,
            "version": "0.1.0",
            "plugin_type": plugin_type.name,
            "author": "",
            "description": f"A {plugin_type.name.lower()} plugin",
            "tags": [],
            "license": "MIT",
        }

        with open(plugin_dir / "plugin.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create plugin.py with template
        template = self._get_plugin_template(name, plugin_type)
        with open(plugin_dir / "plugin.py", "w") as f:
            f.write(template)

        logger.info(f"Created plugin template: {plugin_dir}")
        return plugin_dir

    def _get_plugin_template(self, name: str, plugin_type: PluginType) -> str:
        """Get template code for a plugin type."""
        class_name = "".join(word.capitalize() for word in name.split("_"))

        templates = {
            PluginType.PROTOCOL: f'''"""
{name} - Protocol decoder plugin.
"""

import numpy as np
from typing import Dict, Any, List

from sdr_module.plugins import (
    ProtocolPlugin,
    PluginMetadata,
    PluginType,
)


class {class_name}Plugin(ProtocolPlugin):
    """Protocol decoder for {name}."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="0.1.0",
            plugin_type=PluginType.PROTOCOL,
            author="Your Name",
            description="Decodes {name} protocol",
            tags=["protocol"],
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._sample_rate = config.get("sample_rate", 2.4e6)
        return True

    def get_protocol_info(self) -> Dict[str, Any]:
        return {{
            "name": "{name}",
            "protocol_type": "ism",
            "frequency_range": (433e6, 434e6),
            "bandwidth_hz": 200e3,
            "modulation": "OOK",
            "description": "{name} protocol decoder",
        }}

    def decode(self, samples: np.ndarray) -> List[Dict[str, Any]]:
        # Implement decoding logic here
        frames = []
        return frames

    def can_decode(self, samples: np.ndarray) -> float:
        # Return confidence score 0.0 to 1.0
        return 0.0
''',

            PluginType.DEMODULATOR: f'''"""
{name} - Demodulator plugin.
"""

import numpy as np
from typing import Dict, Any

from sdr_module.plugins import (
    DemodulatorPlugin,
    PluginMetadata,
    PluginType,
)


class {class_name}Plugin(DemodulatorPlugin):
    """Custom demodulator for {name}."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="0.1.0",
            plugin_type=PluginType.DEMODULATOR,
            author="Your Name",
            description="Custom {name} demodulator",
            tags=["demodulator"],
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._sample_rate = config.get("sample_rate", 2.4e6)
        return True

    def get_modulation_info(self) -> Dict[str, Any]:
        return {{
            "name": "{name}",
            "modulation_type": "digital",
            "supported_variants": [],
            "description": "Custom {name} demodulation",
        }}

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        # Implement demodulation logic here
        return np.abs(samples)
''',

            PluginType.PROCESSOR: f'''"""
{name} - Signal processor plugin.
"""

import numpy as np
from typing import Dict, Any, Optional

from sdr_module.plugins import (
    ProcessorPlugin,
    PluginMetadata,
    PluginType,
)


class {class_name}Plugin(ProcessorPlugin):
    """Signal processor for {name}."""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="0.1.0",
            plugin_type=PluginType.PROCESSOR,
            author="Your Name",
            description="Custom {name} signal processor",
            tags=["processor"],
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._config = config
        return True

    def get_processor_info(self) -> Dict[str, Any]:
        return {{
            "name": "{name}",
            "category": "filter",
            "input_type": "complex",
            "output_type": "complex",
            "description": "Custom {name} processor",
        }}

    def process(self, samples: np.ndarray) -> np.ndarray:
        # Implement processing logic here
        return samples
''',
        }

        return templates.get(plugin_type, templates[PluginType.PROCESSOR])
