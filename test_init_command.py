#!/usr/bin/env python3
"""Test the init command directly"""
import sys
import tempfile
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import cli module - the actual cli.py file has the Click commands
from omi import cli as cli_module

# But we need to import from the actual cli.py file, not the cli/ directory
# Let's import it directly using importlib
import importlib.util
cli_py_path = Path(__file__).parent / "src" / "omi" / "cli.py"
spec = importlib.util.spec_from_file_location("omi_cli_commands", cli_py_path)
cli_commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_commands)

from click.testing import CliRunner

# Create temp dir
tmpdir = tempfile.mkdtemp()
print(f"Testing in: {tmpdir}")

# Run init command
runner = CliRunner()
result = runner.invoke(cli_commands.init, ['--data-dir', tmpdir], obj={'data_dir': Path(tmpdir)})

print(f"Exit code: {result.exit_code}")
print(f"Output:\n{result.output}")

if result.exit_code != 0:
    if result.exception:
        import traceback
        print("Exception:")
        print(''.join(traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__)))
    sys.exit(1)

# Check config exists
config_path = Path(tmpdir) / 'config.yaml'
if not config_path.exists():
    print("FAIL: config.yaml not created")
    sys.exit(1)

# Load and check compression section
config = yaml.safe_load(config_path.read_text())
if 'compression' not in config:
    print("FAIL: compression section missing from config")
    sys.exit(1)

compression = config['compression']
print(f"\nCompression config keys: {list(compression.keys())}")
print(f"Compression config:\n{yaml.dump({'compression': compression})}")

# Verify expected keys
required_keys = ['enabled', 'provider', 'model', 'max_summary_tokens']
missing = [k for k in required_keys if k not in compression]
if missing:
    print(f"FAIL: Missing keys: {missing}")
    sys.exit(1)

print("\n✓ OK - All compression config keys present")
