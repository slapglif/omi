# OMI Embedding Example Plugin

This is an example plugin demonstrating how to create a custom embedding provider for the OMI (OpenClaw Memory Infrastructure) system.

## What This Demonstrates

This example shows:

1. **Plugin Package Structure** - How to structure a Python package as an OMI plugin
2. **Entry Point Registration** - Using `pyproject.toml` to register your plugin with OMI
3. **Interface Implementation** - Implementing the `EmbeddingProvider` abstract base class
4. **Interface Versioning** - Setting `interface_version = '1.0'` for compatibility tracking
5. **Plugin Discovery** - How OMI automatically discovers and loads your plugin

## Installation

Install the plugin in development mode:

```bash
cd examples/omi-embedding-example
pip install -e .
```

## Verification

After installation, verify that OMI can discover your plugin:

```bash
omi plugins list
```

You should see `DummyEmbedder` listed under "Embedding Providers".

## Plugin Structure

```
omi-embedding-example/
├── pyproject.toml              # Package metadata and entry point registration
├── README.md                   # This file
└── src/
    └── omi_embedding_example/
        ├── __init__.py         # Package initialization
        └── provider.py         # DummyEmbedder implementation
```

## Key Files

### pyproject.toml

The entry point registration is the critical piece:

```toml
[project.entry-points."omi.embedding_providers"]
dummy = "omi_embedding_example.provider:DummyEmbedder"
```

This tells OMI:
- **Group**: `omi.embedding_providers` (plugin type)
- **Name**: `dummy` (plugin identifier)
- **Location**: `omi_embedding_example.provider:DummyEmbedder` (import path)

### provider.py

The implementation must:
1. Inherit from `omi.embeddings.EmbeddingProvider`
2. Set `interface_version = '1.0'`
3. Implement all abstract methods: `embed()`, `embed_batch()`, `similarity()`, `get_dimension()`, `get_model_name()`

## Creating Your Own Plugin

To create your own embedding provider plugin:

1. **Copy this structure** as a starting point
2. **Rename the package** (e.g., `omi-my-embedder`)
3. **Implement your embedder** in `provider.py`
4. **Update entry points** in `pyproject.toml`
5. **Install and test** with `pip install -e .`

For more details, see the [Plugin Development Guide](../../docs/PLUGINS.md).

## Plugin Types

OMI supports three plugin types:

- **Embedding Providers** (`omi.embedding_providers`) - Custom embedding models
- **Storage Backends** (`omi.storage_backends`) - Alternative storage systems
- **Event Handlers** (`omi.event_handlers`) - React to memory lifecycle events

This example demonstrates an embedding provider. The pattern is similar for other plugin types.

## License

MIT License - Same as the main OMI project.
