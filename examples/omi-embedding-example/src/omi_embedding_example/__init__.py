"""
OMI Embedding Example Plugin

This is an example plugin demonstrating how to create a custom embedding provider
for the OMI (OpenClaw Memory Infrastructure) system.

Usage:
    1. Install this package: pip install -e .
    2. The plugin will be automatically discovered by OMI
    3. View installed plugins: omi plugins list
    4. Configure OMI to use this provider in config.yaml

For more information on creating OMI plugins, see docs/PLUGINS.md
"""

__version__ = "0.1.0"

# The actual DummyEmbedder implementation will be imported from provider.py
# when it's created in the next subtask
__all__ = ["__version__"]
