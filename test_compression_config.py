#!/usr/bin/env python3
"""Simple verification of compression config"""
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
from click.testing import CliRunner

# Import directly from omi module
from omi import cli as omi_cli

# Create temp dir
tmpdir = tempfile.mkdtemp()
print(f"Testing in: {tmpdir}")

# Run init command
runner = CliRunner()
result = runner.invoke(omi_cli.init, ['--data-dir', tmpdir])
print(f"Init result: {result.exit_code}")
if result.exit_code != 0:
    print(f"Error: {result.output}")
    sys.exit(1)

# Check config exists
config_path = Path(tmpdir) / 'config.yaml'
if not config_path.exists():
    print("FAIL: config.yaml not created")
    sys.exit(1)

# Load and check compression section
config = yaml.safe_load(config_path.read_text())
if 'compression' not in config:
    print("FAIL: compression section missing")
    sys.exit(1)

compression = config['compression']
print(f"Compression config: {compression}")

# Verify expected keys
required_keys = ['enabled', 'provider', 'model', 'max_summary_tokens']
for key in required_keys:
    if key not in compression:
        print(f"FAIL: compression.{key} missing")
        sys.exit(1)

print("OK - All compression config keys present")
