# OMI Plugin Development Guide

> *"Extend the palace without touching its foundation."*

OMI's plugin system enables third-party packages to extend functionality through Python entry points — the same pattern that made pytest, Flask, and Datasette successful ecosystems.

## What Plugins Can Do

OMI supports three plugin types:

| Plugin Type | Entry Point Group | Base Class | Purpose |
|-------------|-------------------|------------|---------|
| **Embedding Providers** | `omi.embedding_providers` | `EmbeddingProvider` | Custom embedding models (Cohere, OpenAI, HuggingFace, etc.) |
| **Storage Backends** | `omi.storage_backends` | `StorageBackend` | Alternative storage systems (PostgreSQL, Redis, Neo4j) |
| **Event Handlers** | `omi.event_handlers` | `EventHandler` | React to memory lifecycle events (webhooks, logging, analytics) |

## Quick Start: Create a Plugin

### 1. Package Structure

```
my-omi-plugin/
├── pyproject.toml              # Package metadata + entry point registration
├── README.md
└── src/
    └── my_omi_plugin/
        ├── __init__.py
        └── provider.py         # Your plugin implementation
```

### 2. Implement the Plugin

```python
# src/my_omi_plugin/provider.py
from typing import List
from omi.embeddings import EmbeddingProvider

class MyCustomEmbedder(EmbeddingProvider):
    """Custom embedding provider using my-service API"""

    # REQUIRED: Interface version for compatibility tracking
    interface_version = "1.0"

    def __init__(self, api_key: str, model: str = "my-model-v2"):
        self.api_key = api_key
        self.model = model
        self.dimension = 1536  # Model-specific

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Your implementation here
        response = self._call_api([text])
        return response[0]

    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.embed(text) for text in texts]

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        import numpy as np
        vec1, vec2 = np.array(embedding1), np.array(embedding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        return self.dimension

    def get_model_name(self) -> str:
        """Return model identifier"""
        return self.model

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Internal API call logic"""
        # Implementation details...
        pass
```

### 3. Register the Entry Point

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-omi-plugin"
version = "0.1.0"
description = "Custom embedding provider for OMI"
requires-python = ">=3.10"
dependencies = [
    "omi-openclaw>=0.1.0",
    "numpy>=1.24.0",
]

# CRITICAL: Entry point registration
[project.entry-points."omi.embedding_providers"]
my-embedder = "my_omi_plugin.provider:MyCustomEmbedder"
```

**Entry Point Format:**
- **Key** (`my-embedder`): Plugin name shown in `omi plugins list`
- **Value** (`my_omi_plugin.provider:MyCustomEmbedder`): Import path to your plugin class

### 4. Install and Test

```bash
# Install in development mode
cd my-omi-plugin
pip install -e .

# Verify OMI discovers your plugin
omi plugins list

# Expected output:
# Embedding Providers:
#   my-embedder (MyCustomEmbedder) - v0.1.0 [interface: 1.0] ✓
```

## Plugin Types in Detail

### Embedding Providers

**Use case:** Integrate custom embedding models (Cohere, Anthropic, OpenAI, HuggingFace, local models).

**Required methods:**

```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    interface_version: str  # Must be "1.0"

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate similarity score (-1.0 to 1.0)"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier"""
        pass
```

**Example entry point:**
```toml
[project.entry-points."omi.embedding_providers"]
cohere = "omi_cohere.provider:CohereEmbedder"
openai = "omi_openai.provider:OpenAIEmbedder"
```

### Storage Backends

**Use case:** Replace SQLite with PostgreSQL, Neo4j, Redis, or distributed databases.

**Required methods:**

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class StorageBackend(ABC):
    interface_version: str  # Must be "1.0"

    @abstractmethod
    def store_memory(self, memory: Dict[str, Any]) -> str:
        """Store a memory and return its ID"""
        pass

    @abstractmethod
    def recall(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by query"""
        pass

    @abstractmethod
    def add_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        """Create relationship between memories"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources"""
        pass
```

**Example entry point:**
```toml
[project.entry-points."omi.storage_backends"]
postgres = "omi_postgres.backend:PostgresBackend"
neo4j = "omi_neo4j.backend:Neo4jBackend"
```

### Event Handlers

**Use case:** React to memory lifecycle events — webhooks, logging, analytics, integrations.

**Required methods:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class EventHandler(ABC):
    interface_version: str  # Must be "1.0"

    @abstractmethod
    def handle(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Handle a memory event

        Args:
            event_type: Event name (e.g., 'memory.stored', 'session.started')
            event_data: Event payload (memory content, metadata, etc.)
        """
        pass
```

**Event types:**
- `memory.stored` — New memory added to Graph Palace
- `memory.recalled` — Memory retrieved via recall
- `session.started` — Session initialization
- `session.ended` — Session cleanup
- `checkpoint.created` — Pre-compression checkpoint saved
- `belief.updated` — Belief confidence changed

**Example entry point:**
```toml
[project.entry-points."omi.event_handlers"]
webhook = "omi_webhook.handler:WebhookHandler"
slack = "omi_slack.handler:SlackNotifier"
analytics = "omi_analytics.handler:AnalyticsTracker"
```

## Interface Versioning

Every plugin **must** declare `interface_version = "1.0"`.

OMI validates plugins on load:

```python
# Plugin validation checks:
# 1. interface_version attribute exists
# 2. Version is compatible (currently only "1.0" is supported)
# 3. Plugin inherits from the correct base class
# 4. All abstract methods are implemented

from omi.plugins import validate_plugin, PluginValidationError

try:
    validate_plugin(MyPlugin, expected_base_class=EmbeddingProvider)
except PluginValidationError as e:
    print(f"Plugin validation failed: {e}")
```

**Version compatibility:**
- `1.0` — Current stable interface (supported)
- Future versions may add optional methods while maintaining backward compatibility

**Best practice:** Pin `omi-openclaw` dependency in your plugin to ensure compatibility:

```toml
dependencies = [
    "omi-openclaw>=0.1.0,<0.2.0",  # Lock to compatible minor version
]
```

## Plugin Discovery and Loading

### Discovery

OMI discovers plugins via Python's `importlib.metadata` entry points:

```python
from omi.plugins import discover_plugins

# Discover all OMI plugins
all_plugins = discover_plugins()
# Returns: {'omi.embedding_providers': [...], 'omi.storage_backends': [...], ...}

# Discover only embedding providers
providers = discover_plugins('omi.embedding_providers')
```

### Loading

Load plugins with automatic validation:

```python
from omi.plugins import load_plugin, PluginLoadError
from omi.embeddings import EmbeddingProvider

try:
    provider_cls = load_plugin(
        'omi.embedding_providers',
        'my-embedder',
        expected_base_class=EmbeddingProvider
    )
    provider = provider_cls(api_key="...")
except PluginLoadError as e:
    print(f"Failed to load plugin: {e}")
```

### Plugin Registry

The registry provides centralized plugin management:

```python
from omi.plugins import PluginRegistry

registry = PluginRegistry()
registry.discover_all()

# List all embedding providers
providers = registry.get_plugins('omi.embedding_providers')

# Load and cache a plugin
provider = registry.load('omi.embedding_providers', 'my-embedder')

# Check if plugin is loaded
if registry.is_loaded('omi.embedding_providers', 'my-embedder'):
    print("Plugin already loaded (cached)")
```

## Testing Your Plugin

### Unit Tests

```python
# tests/test_my_embedder.py
import pytest
from my_omi_plugin.provider import MyCustomEmbedder
from omi.embeddings import EmbeddingProvider

def test_plugin_interface():
    """Verify plugin implements EmbeddingProvider"""
    assert issubclass(MyCustomEmbedder, EmbeddingProvider)
    assert MyCustomEmbedder.interface_version == "1.0"

def test_embed_single():
    """Test single text embedding"""
    embedder = MyCustomEmbedder(api_key="test-key")
    result = embedder.embed("Hello world")

    assert isinstance(result, list)
    assert len(result) == embedder.get_dimension()
    assert all(isinstance(x, float) for x in result)

def test_embed_batch():
    """Test batch embedding"""
    embedder = MyCustomEmbedder(api_key="test-key")
    texts = ["First text", "Second text", "Third text"]
    results = embedder.embed_batch(texts)

    assert len(results) == len(texts)
    assert all(len(r) == embedder.get_dimension() for r in results)

def test_similarity():
    """Test cosine similarity calculation"""
    embedder = MyCustomEmbedder(api_key="test-key")
    vec1 = embedder.embed("identical text")
    vec2 = embedder.embed("identical text")

    similarity = embedder.similarity(vec1, vec2)
    assert 0.99 <= similarity <= 1.0  # Should be nearly identical
```

### Integration Tests

```python
def test_plugin_discovery():
    """Verify OMI can discover the plugin"""
    from omi.plugins import discover_plugins

    plugins = discover_plugins('omi.embedding_providers')
    plugin_names = [ep.name for ep in plugins['omi.embedding_providers']]

    assert 'my-embedder' in plugin_names

def test_plugin_loading():
    """Verify OMI can load and validate the plugin"""
    from omi.plugins import load_plugin
    from omi.embeddings import EmbeddingProvider

    provider_cls = load_plugin(
        'omi.embedding_providers',
        'my-embedder',
        expected_base_class=EmbeddingProvider
    )

    assert provider_cls is MyCustomEmbedder
```

## Best Practices

### 1. Fail Gracefully

Provide clear error messages:

```python
def embed(self, text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    try:
        response = self._call_api([text])
    except APIError as e:
        raise RuntimeError(f"Embedding API failed: {e}") from e

    return response[0]
```

### 2. Cache Expensive Operations

```python
class MyCustomEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._model_cache = None  # Cache model metadata

    def get_dimension(self) -> int:
        if self._model_cache is None:
            self._model_cache = self._fetch_model_info()
        return self._model_cache['dimension']
```

### 3. Support Batching

Batch operations are critical for performance:

```python
def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Process texts in batches to avoid API rate limits"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = self._call_api(batch)
        results.extend(batch_results)
    return results
```

### 4. Document Configuration

Provide clear examples in your README:

```markdown
## Configuration

Configure the plugin via `config.yaml`:

\`\`\`yaml
embedding:
  provider: my-embedder
  model: my-model-v2
  api_key: ${MY_SERVICE_API_KEY}
  batch_size: 16
\`\`\`

Or programmatically:

\`\`\`python
from my_omi_plugin.provider import MyCustomEmbedder

embedder = MyCustomEmbedder(
    api_key=os.getenv("MY_SERVICE_API_KEY"),
    model="my-model-v2"
)
\`\`\`
```

### 5. Add Type Hints

Full type annotations improve IDE support and catch errors:

```python
from typing import List, Optional, Dict, Any

def embed_batch(
    self,
    texts: List[str],
    batch_size: int = 8,
    timeout: Optional[float] = None
) -> List[List[float]]:
    """Type hints make your plugin easier to use"""
    pass
```

## Example: Complete Plugin

See [`examples/omi-embedding-example/`](../examples/omi-embedding-example/) for a complete working plugin demonstrating:

- Package structure
- Entry point registration
- Abstract method implementation
- Interface versioning
- Error handling
- Unit tests

**Install the example:**

```bash
cd examples/omi-embedding-example
pip install -e .
omi plugins list  # Verify DummyEmbedder appears
```

## Publishing Your Plugin

### 1. Package Checklist

Before publishing to PyPI:

- [ ] `interface_version = "1.0"` is set
- [ ] All abstract methods are implemented
- [ ] Entry points are registered in `pyproject.toml`
- [ ] Unit tests cover all methods
- [ ] Integration tests verify plugin discovery
- [ ] README includes installation and configuration examples
- [ ] Version follows semantic versioning (e.g., `0.1.0`)

### 2. Build and Publish

```bash
# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Or TestPyPI for testing
python -m twine upload --repository testpypi dist/*
```

### 3. Naming Convention

Follow the pattern: `omi-{plugin-type}-{name}`

**Examples:**
- `omi-embedding-cohere` (Cohere embedding provider)
- `omi-storage-postgres` (PostgreSQL storage backend)
- `omi-event-webhook` (Webhook event handler)

This makes plugins easy to discover via `pip search omi-`.

## Debugging Plugins

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from omi.plugins import discover_plugins, load_plugin

# You'll see:
# DEBUG:omi.plugins:Discovered 3 plugins in group 'omi.embedding_providers'
# INFO:omi.plugins:Loading plugin 'my-embedder' from group 'omi.embedding_providers'
# DEBUG:omi.plugins:Plugin 'MyCustomEmbedder' passed validation (interface_version=1.0)
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `PluginValidationError: missing 'interface_version'` | Forgot to set `interface_version` attribute | Add `interface_version = "1.0"` to plugin class |
| `PluginValidationError: incompatible interface_version` | Using unsupported version | Use `interface_version = "1.0"` (only supported version) |
| `PluginLoadError: Plugin 'foo' not found` | Entry point not registered or plugin not installed | Check `pyproject.toml` entry points, run `pip install -e .` |
| `TypeError: Can't instantiate abstract class` | Missing abstract method implementations | Implement all methods from base class |

## CLI Commands

### List All Plugins

```bash
omi plugins list
```

**Output:**
```
Embedding Providers:
  nim (NIMEmbedder) - v0.1.0 [interface: 1.0] ✓
  ollama (OllamaEmbedder) - v0.1.0 [interface: 1.0] ✓
  my-embedder (MyCustomEmbedder) - v0.1.0 [interface: 1.0] ✓

Storage Backends:
  (none installed)

Event Handlers:
  (none installed)
```

## Community Plugins

Know of a useful OMI plugin? Submit a PR to add it here:

- **omi-embedding-cohere** — Cohere embedding API integration
- **omi-storage-postgres** — PostgreSQL storage backend with pgvector
- **omi-event-webhook** — HTTP webhook notifications for memory events

## Support

- **Issues:** [GitHub Issues](https://github.com/slapglif/omi/issues)
- **Discussions:** [GitHub Discussions](https://github.com/slapglif/omi/discussions)
- **Examples:** [`examples/`](../examples/)

---

*Built with the OMI plugin system. Extend freely.*
