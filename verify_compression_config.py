#!/usr/bin/env python
"""Verify compression config exists in init command"""
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, "./src")

from omi.cli import init
from click.testing import CliRunner
import yaml

# Create temp dir
tmpdir = tempfile.mkdtemp()

# Run init command
runner = CliRunner()
result = runner.invoke(init, ['--data-dir', tmpdir])

# Check config exists
config_path = Path(tmpdir) / 'config.yaml'
assert config_path.exists(), "config.yaml not created"

# Load and check compression section
config = yaml.safe_load(config_path.read_text())
assert 'compression' in config, "compression section missing from config"

print("OK")
