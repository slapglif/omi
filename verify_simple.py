#!/usr/bin/env python3
"""Simple check that compression config template exists"""
import sys
from pathlib import Path

# Read the cli.py file
cli_path = Path("./src/omi/cli.py")
content = cli_path.read_text()

# Check if compression section exists in config_template
if 'compression:' not in content:
    print("FAIL: 'compression:' not found in cli.py")
    sys.exit(1)

# Check for required keys in the compression section
required_keys = ['enabled:', 'provider:', 'model:', 'max_summary_tokens:']
for key in required_keys:
    if key not in content:
        print(f"FAIL: compression.{key} not found in config template")
        sys.exit(1)

print("OK - Compression section with all required keys found in config template")
