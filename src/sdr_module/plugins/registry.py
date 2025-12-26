"""
Plugin registry for managing loaded plugins.

Provides centralized storage and access to all loaded plugins,
with support for filtering by type, tags, and capabilities.
"""

from typing import Dict, List, Optional, Type, Callable, Any
from threading import Lock
import logging

from .base import (
    Plugin,
    PluginMetadata,
    PluginType,
    PluginState,
    PluginError,
    PluginNotFoundError,
)


logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all loaded plugins.

    Thread-safe storage and retrieval of plugin instances.
    Supports filtering, querying, and lifecycle management.
    """

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._lock = Lock()
        self._listeners: List[Callable[[str, str], None]] = []

    def register_class(self, plugin_class: Type[Plugin]) -> bool:
        """
        Register a plugin class.

        Args:
            plugin_class: Plugin class to register

        Returns:
            True if registered successfully
        """
        try:
            metadata = plugin_class.get_metadata()
            name = metadata.name

            with self._lock:
                if name in self._plugin_classes:
                    logger.warning(f"Plugin class '{name}' already registered, replacing")

                self._plugin_classes[name] = plugin_class
                logger.info(f"Registered plugin class: {name} v{metadata.version}")
                self._notify_listeners(name, "registered")

            return True

        except Exception as e:
            logger.error(f"Failed to register plugin class: {e}")
            return False

    def unregister_class(self, name: str) -> bool:
        """
        Unregister a plugin class.

        Args:
            name: Plugin name to unregister

        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if name in self._plugin_classes:
                # Also remove any instance
                if name in self._plugins:
                    self._plugins[name].cleanup()
                    del self._plugins[name]

                del self._plugin_classes[name]
                logger.info(f"Unregistered plugin class: {name}")
                self._notify_listeners(name, "unregistered")
                return True

        return False

    def create_instance(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """
        Create and initialize a plugin instance.

        Args:
            name: Plugin name
            config: Optional configuration dictionary

        Returns:
            Plugin instance or None on failure
        """
        with self._lock:
            if name not in self._plugin_classes:
                logger.error(f"Plugin class not found: {name}")
                return None

            try:
                plugin_class = self._plugin_classes[name]
                plugin = plugin_class()

                if plugin.initialize(config or {}):
                    plugin._state = PluginState.INITIALIZED
                    self._plugins[name] = plugin
                    logger.info(f"Created plugin instance: {name}")
                    return plugin
                else:
                    logger.error(f"Plugin initialization failed: {name}")
                    return None

            except Exception as e:
                logger.error(f"Failed to create plugin instance '{name}': {e}")
                return None

    def get_instance(self, name: str) -> Optional[Plugin]:
        """
        Get a plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        with self._lock:
            return self._plugins.get(name)

    def get_or_create(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Plugin]:
        """
        Get existing instance or create new one.

        Args:
            name: Plugin name
            config: Configuration for new instance

        Returns:
            Plugin instance or None on failure
        """
        instance = self.get_instance(name)
        if instance:
            return instance
        return self.create_instance(name, config)

    def remove_instance(self, name: str) -> bool:
        """
        Remove a plugin instance.

        Args:
            name: Plugin name

        Returns:
            True if removed successfully
        """
        with self._lock:
            if name in self._plugins:
                self._plugins[name].cleanup()
                del self._plugins[name]
                logger.info(f"Removed plugin instance: {name}")
                return True
        return False

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.

        Args:
            name: Plugin name

        Returns:
            PluginMetadata or None if not found
        """
        with self._lock:
            if name in self._plugin_classes:
                return self._plugin_classes[name].get_metadata()
        return None

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        state: Optional[PluginState] = None,
        tag: Optional[str] = None
    ) -> List[PluginMetadata]:
        """
        List registered plugins with optional filtering.

        Args:
            plugin_type: Filter by plugin type
            state: Filter by state (requires instance)
            tag: Filter by tag

        Returns:
            List of matching plugin metadata
        """
        results = []

        with self._lock:
            for name, plugin_class in self._plugin_classes.items():
                metadata = plugin_class.get_metadata()

                # Apply filters
                if plugin_type and metadata.plugin_type != plugin_type:
                    continue

                if tag and tag not in metadata.tags:
                    continue

                if state:
                    instance = self._plugins.get(name)
                    if not instance or instance.state != state:
                        continue

                results.append(metadata)

        return results

    def list_by_type(self, plugin_type: PluginType) -> List[str]:
        """
        List plugin names by type.

        Args:
            plugin_type: Plugin type to filter by

        Returns:
            List of plugin names
        """
        return [m.name for m in self.list_plugins(plugin_type=plugin_type)]

    def get_protocols(self) -> List[str]:
        """Get all protocol plugin names."""
        return self.list_by_type(PluginType.PROTOCOL)

    def get_demodulators(self) -> List[str]:
        """Get all demodulator plugin names."""
        return self.list_by_type(PluginType.DEMODULATOR)

    def get_devices(self) -> List[str]:
        """Get all device plugin names."""
        return self.list_by_type(PluginType.DEVICE)

    def get_processors(self) -> List[str]:
        """Get all processor plugin names."""
        return self.list_by_type(PluginType.PROCESSOR)

    def search(self, query: str) -> List[PluginMetadata]:
        """
        Search plugins by name, description, or tags.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching plugin metadata
        """
        query = query.lower()
        results = []

        with self._lock:
            for plugin_class in self._plugin_classes.values():
                metadata = plugin_class.get_metadata()

                if (query in metadata.name.lower() or
                    query in metadata.description.lower() or
                    any(query in tag.lower() for tag in metadata.tags)):
                    results.append(metadata)

        return results

    def enable_plugin(self, name: str) -> bool:
        """
        Enable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if enabled successfully
        """
        instance = self.get_instance(name)
        if instance:
            result = instance.enable()
            if result:
                self._notify_listeners(name, "enabled")
            return result
        return False

    def disable_plugin(self, name: str) -> bool:
        """
        Disable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if disabled successfully
        """
        instance = self.get_instance(name)
        if instance:
            result = instance.disable()
            if result:
                self._notify_listeners(name, "disabled")
            return result
        return False

    def add_listener(self, callback: Callable[[str, str], None]) -> None:
        """
        Add a plugin event listener.

        Args:
            callback: Function(plugin_name, event_type) to call on events
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, str], None]) -> None:
        """
        Remove a plugin event listener.

        Args:
            callback: Previously added callback to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self, plugin_name: str, event: str) -> None:
        """Notify all listeners of a plugin event."""
        for listener in self._listeners:
            try:
                listener(plugin_name, event)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def clear(self) -> None:
        """Remove all plugins and instances."""
        with self._lock:
            for plugin in self._plugins.values():
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")

            self._plugins.clear()
            self._plugin_classes.clear()
            logger.info("Plugin registry cleared")

    @property
    def plugin_count(self) -> int:
        """Get number of registered plugin classes."""
        return len(self._plugin_classes)

    @property
    def instance_count(self) -> int:
        """Get number of active plugin instances."""
        return len(self._plugins)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            type_counts = {}
            for plugin_class in self._plugin_classes.values():
                ptype = plugin_class.get_metadata().plugin_type.name
                type_counts[ptype] = type_counts.get(ptype, 0) + 1

            state_counts = {}
            for plugin in self._plugins.values():
                state = plugin.state.name
                state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_classes": self.plugin_count,
            "total_instances": self.instance_count,
            "by_type": type_counts,
            "by_state": state_counts,
        }
