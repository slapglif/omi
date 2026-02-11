"""
Plugin Discovery and Loading System

Provides plugin discovery via Python entry points for embedding providers,
storage backends, and event handlers. Enables third-party packages to extend
OMI without modifying the core codebase.

Entry point groups:
- omi.embedding_providers: Custom embedding providers (EmbeddingProvider subclasses)
- omi.storage_backends: Custom storage backends (StorageBackend subclasses)
- omi.event_handlers: Custom event handlers (EventHandler subclasses)

Usage:
    # Discover all installed plugins
    plugins = discover_plugins()

    # Load a specific plugin
    provider = load_plugin('omi.embedding_providers', 'my-custom-embedder')

    # Use the plugin registry
    registry = PluginRegistry()
    registry.discover_all()
    embedding_providers = registry.get_plugins('omi.embedding_providers')
"""

import logging
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass
from datetime import datetime
from abc import ABC

logger = logging.getLogger(__name__)

# Import for entry point discovery (Python 3.10+)
try:
    from importlib.metadata import entry_points, EntryPoint
except ImportError:
    # Fallback for Python < 3.10
    from importlib_metadata import entry_points, EntryPoint  # type: ignore


# Plugin entry point group names
PLUGIN_GROUPS = {
    'embedding_providers': 'omi.embedding_providers',
    'storage_backends': 'omi.storage_backends',
    'event_handlers': 'omi.event_handlers',
}


class PluginError(Exception):
    """Base exception for plugin-related errors"""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load"""
    pass


class PluginValidationError(PluginError):
    """Raised when a plugin fails validation checks"""
    pass


@dataclass
class PluginInfo:
    """Metadata for a discovered plugin"""
    name: str  # Entry point name
    group: str  # Entry point group (e.g., 'omi.embedding_providers')
    module: str  # Module path (e.g., 'my_plugin.provider')
    attr: str  # Attribute name (e.g., 'MyEmbedder')
    version: Optional[str] = None  # Plugin version (if available)
    interface_version: Optional[str] = None  # Plugin interface version
    loaded: bool = False  # Whether plugin has been successfully loaded
    error: Optional[str] = None  # Error message if loading failed
    instance: Optional[Any] = None  # Loaded plugin instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "group": self.group,
            "module": self.module,
            "attr": self.attr,
            "version": self.version,
            "interface_version": self.interface_version,
            "loaded": self.loaded,
            "error": self.error,
        }


def discover_plugins(group: Optional[str] = None) -> Dict[str, List[EntryPoint]]:
    """
    Discover plugins via Python entry points

    Args:
        group: Optional entry point group to filter by (e.g., 'omi.embedding_providers').
               If None, discovers all OMI plugin groups.

    Returns:
        Dictionary mapping group names to lists of entry points

    Example:
        # Discover all plugins
        all_plugins = discover_plugins()

        # Discover only embedding providers
        providers = discover_plugins('omi.embedding_providers')
    """
    discovered: Dict[str, List[EntryPoint]] = {}

    # Determine which groups to discover
    groups_to_discover = [group] if group else list(PLUGIN_GROUPS.values())

    for group_name in groups_to_discover:
        try:
            # Python 3.10+ returns EntryPoints object with select() method
            eps = entry_points()
            if hasattr(eps, 'select'):
                group_eps = list(eps.select(group=group_name))
            else:
                # Fallback for older versions where entry_points() returns dict
                group_eps = eps.get(group_name, [])

            discovered[group_name] = group_eps
            logger.debug(f"Discovered {len(group_eps)} plugins in group '{group_name}'")
        except Exception as e:
            logger.warning(f"Failed to discover plugins in group '{group_name}': {e}")
            discovered[group_name] = []

    return discovered


def load_plugin(
    group: str,
    name: str,
    validate: bool = True,
    expected_base_class: Optional[Type] = None,
) -> Any:
    """
    Load a specific plugin by name from an entry point group

    Args:
        group: Entry point group (e.g., 'omi.embedding_providers')
        name: Plugin entry point name
        validate: Whether to validate the plugin after loading
        expected_base_class: Expected base class for validation (if validate=True)

    Returns:
        The loaded plugin class or instance

    Raises:
        PluginLoadError: If plugin cannot be loaded
        PluginValidationError: If plugin fails validation

    Example:
        from omi.embeddings import EmbeddingProvider
        provider_cls = load_plugin(
            'omi.embedding_providers',
            'my-embedder',
            expected_base_class=EmbeddingProvider
        )
        provider = provider_cls()
    """
    try:
        # Discover plugins in the specified group
        plugins = discover_plugins(group)
        group_plugins = plugins.get(group, [])

        # Find the plugin by name
        plugin_ep = None
        for ep in group_plugins:
            if ep.name == name:
                plugin_ep = ep
                break

        if plugin_ep is None:
            raise PluginLoadError(
                f"Plugin '{name}' not found in group '{group}'. "
                f"Available: {[ep.name for ep in group_plugins]}"
            )

        # Load the plugin
        logger.info(f"Loading plugin '{name}' from group '{group}'")
        plugin = plugin_ep.load()

        # Validate if requested
        if validate:
            validate_plugin(plugin, expected_base_class, group)

        return plugin

    except PluginValidationError:
        raise  # Re-raise validation errors as-is
    except Exception as e:
        raise PluginLoadError(f"Failed to load plugin '{name}' from '{group}': {e}") from e


def validate_plugin(
    plugin: Any,
    expected_base_class: Optional[Type] = None,
    group: Optional[str] = None,
) -> None:
    """
    Validate a loaded plugin

    Checks:
    1. Plugin has interface_version attribute
    2. Plugin inherits from expected base class (if provided)
    3. interface_version is compatible (currently only '1.0' is supported)

    Args:
        plugin: The loaded plugin class or instance
        expected_base_class: Expected base class (ABC) that plugin should inherit from
        group: Optional entry point group name (for better error messages)

    Raises:
        PluginValidationError: If validation fails

    Example:
        from omi.embeddings import EmbeddingProvider
        validate_plugin(MyEmbedder, EmbeddingProvider, 'omi.embedding_providers')
    """
    plugin_name = getattr(plugin, '__name__', str(plugin))
    group_info = f" in group '{group}'" if group else ""

    # Check for interface_version attribute
    if not hasattr(plugin, 'interface_version'):
        raise PluginValidationError(
            f"Plugin '{plugin_name}'{group_info} is missing required 'interface_version' attribute. "
            f"Add 'interface_version = \"1.0\"' to your plugin class."
        )

    interface_version = getattr(plugin, 'interface_version')

    # Validate interface version (currently only 1.0 is supported)
    supported_versions = ['1.0']
    if interface_version not in supported_versions:
        raise PluginValidationError(
            f"Plugin '{plugin_name}'{group_info} has incompatible interface_version '{interface_version}'. "
            f"Supported versions: {supported_versions}"
        )

    # Check base class if provided
    if expected_base_class is not None:
        # Handle both class and instance
        plugin_class = plugin if isinstance(plugin, type) else type(plugin)

        if not issubclass(plugin_class, expected_base_class):
            raise PluginValidationError(
                f"Plugin '{plugin_name}'{group_info} must inherit from {expected_base_class.__name__}. "
                f"Found: {plugin_class.__bases__}"
            )

    logger.debug(f"Plugin '{plugin_name}'{group_info} passed validation (interface_version={interface_version})")


class PluginRegistry:
    """
    Central registry for managing discovered and loaded plugins

    The registry discovers plugins on initialization and provides methods
    to access, load, and manage plugins across different entry point groups.

    Usage:
        registry = PluginRegistry()
        registry.discover_all()

        # Get all embedding providers
        providers = registry.get_plugins('omi.embedding_providers')

        # Load a specific plugin
        provider = registry.load('omi.embedding_providers', 'my-embedder')
    """

    def __init__(self) -> None:
        """Initialize empty plugin registry"""
        self._plugins: Dict[str, Dict[str, PluginInfo]] = {
            group: {} for group in PLUGIN_GROUPS.values()
        }
        self._discovered: bool = False

    def discover_all(self) -> None:
        """
        Discover all plugins in all entry point groups

        Populates the registry with plugin metadata without loading them.
        Plugins are loaded lazily when first accessed via load().
        """
        logger.info("Discovering plugins in all entry point groups...")

        for group in PLUGIN_GROUPS.values():
            try:
                plugins = discover_plugins(group)
                group_eps = plugins.get(group, [])

                for ep in group_eps:
                    plugin_info = PluginInfo(
                        name=ep.name,
                        group=group,
                        module=ep.value.split(':')[0] if ':' in ep.value else ep.value,
                        attr=ep.value.split(':')[1] if ':' in ep.value else '',
                    )
                    self._plugins[group][ep.name] = plugin_info

                logger.debug(f"Discovered {len(group_eps)} plugins in '{group}'")
            except Exception as e:
                logger.warning(f"Failed to discover plugins in '{group}': {e}")

        self._discovered = True
        total_count = sum(len(plugins) for plugins in self._plugins.values())
        logger.info(f"Plugin discovery complete: {total_count} plugins found")

    def discover_group(self, group: str) -> None:
        """
        Discover plugins in a specific entry point group

        Args:
            group: Entry point group name (e.g., 'omi.embedding_providers')
        """
        if group not in PLUGIN_GROUPS.values():
            logger.warning(f"Unknown plugin group: {group}")
            return

        try:
            plugins = discover_plugins(group)
            group_eps = plugins.get(group, [])

            self._plugins[group] = {}
            for ep in group_eps:
                plugin_info = PluginInfo(
                    name=ep.name,
                    group=group,
                    module=ep.value.split(':')[0] if ':' in ep.value else ep.value,
                    attr=ep.value.split(':')[1] if ':' in ep.value else '',
                )
                self._plugins[group][ep.name] = plugin_info

            logger.info(f"Discovered {len(group_eps)} plugins in '{group}'")
        except Exception as e:
            logger.warning(f"Failed to discover plugins in '{group}': {e}")

    def get_plugins(self, group: str) -> Dict[str, PluginInfo]:
        """
        Get all discovered plugins in a specific group

        Args:
            group: Entry point group name

        Returns:
            Dictionary mapping plugin names to PluginInfo objects
        """
        if not self._discovered:
            self.discover_all()

        return self._plugins.get(group, {})

    def load(
        self,
        group: str,
        name: str,
        validate: bool = True,
        expected_base_class: Optional[Type] = None,
    ) -> Any:
        """
        Load a specific plugin from the registry

        Args:
            group: Entry point group name
            name: Plugin name
            validate: Whether to validate the plugin
            expected_base_class: Expected base class for validation

        Returns:
            Loaded plugin class or instance

        Raises:
            PluginLoadError: If plugin cannot be loaded
            PluginValidationError: If plugin fails validation
        """
        if not self._discovered:
            self.discover_all()

        # Check if plugin exists in registry
        plugin_info = self._plugins.get(group, {}).get(name)
        if plugin_info is None:
            available = list(self._plugins.get(group, {}).keys())
            raise PluginLoadError(
                f"Plugin '{name}' not found in registry for group '{group}'. "
                f"Available: {available}"
            )

        # Return cached instance if already loaded
        if plugin_info.loaded and plugin_info.instance is not None:
            return plugin_info.instance

        # Load the plugin
        try:
            plugin = load_plugin(group, name, validate, expected_base_class)

            # Update plugin info
            plugin_info.loaded = True
            plugin_info.instance = plugin
            plugin_info.interface_version = getattr(plugin, 'interface_version', None)

            return plugin
        except Exception as e:
            plugin_info.loaded = False
            plugin_info.error = str(e)
            raise

    def list_all(self) -> Dict[str, List[PluginInfo]]:
        """
        List all discovered plugins grouped by entry point group

        Returns:
            Dictionary mapping group names to lists of PluginInfo objects
        """
        if not self._discovered:
            self.discover_all()

        return {
            group: list(plugins.values())
            for group, plugins in self._plugins.items()
        }

    def is_loaded(self, group: str, name: str) -> bool:
        """
        Check if a specific plugin has been loaded

        Args:
            group: Entry point group name
            name: Plugin name

        Returns:
            True if plugin is loaded, False otherwise
        """
        plugin_info = self._plugins.get(group, {}).get(name)
        return plugin_info is not None and plugin_info.loaded
