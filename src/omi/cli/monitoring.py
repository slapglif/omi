"""OMI CLI - Monitoring Commands

Status reporting and security audit commands.
"""
import os
import sys
from pathlib import Path
from typing import Optional
import click

# OMI imports
from omi import GraphPalace
from omi.security import PoisonDetector

# Local CLI imports
from .common import (
    get_base_path,
    echo_normal,
    echo_verbose,
    echo_quiet,
    VERBOSITY_NORMAL,
)


@click.group()
@click.pass_context
def monitoring_group(ctx):
    """Monitoring and security commands."""
    ctx.ensure_object(dict)


@monitoring_group.command()
@click.pass_context
def status(ctx) -> None:
    """Show OMI health and size statistics."""
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(click.style("OMI Status Report", fg="cyan", bold=True), verbosity)
    echo_normal("=" * 50, verbosity)

    # Base path info
    echo_normal(f"\nBase Path: {click.style(str(base_path), fg='cyan')}", verbosity)

    # Check files
    files_to_check = [
        ("Config", base_path / "config.yaml"),
        ("Database", base_path / "palace.sqlite"),
        ("NOW.md", base_path / "NOW.md"),
    ]

    echo_normal(f"\nFiles:", verbosity)
    for name, path in files_to_check:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status_symbol = "✓" if exists else "✗"
        status_color = "green" if exists else "red"
        echo_normal(f"  {name:12} {click.style(status_symbol, fg=status_color)} {path.name}", verbosity)
        if size > 0:
            echo_verbose(f"    Size: {size / 1024:.1f} KB", verbosity)

    # Database stats
    db_path = base_path / "palace.sqlite"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        mem_count = cursor.fetchone()[0]

        # Memory types breakdown
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        type_counts = dict(cursor.fetchall())

        # Database size
        db_size = db_path.stat().st_size / 1024  # KB

        echo_normal(f"\nDatabase:", verbosity)
        echo_normal(f"  Size: {click.style(f'{db_size:.1f} KB', fg='cyan')}", verbosity)
        echo_normal(f"  Memories: {click.style(str(mem_count), fg='cyan')}", verbosity)
        if type_counts:
            echo_verbose(f"  Breakdown:", verbosity)
            for mem_type, count in type_counts.items():
                echo_verbose(f"    {mem_type}: {count}", verbosity)

        conn.close()

    # Integrity check
    echo_normal(f"\nIntegrity:", verbosity)
    try:
        from omi.security import IntegrityChecker
        integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
        now_ok = integrity_checker.check_now_md()
        mem_ok = integrity_checker.check_memory_md()
        now_color = "green" if now_ok else "red"
        mem_color = "green" if mem_ok else "red"
        echo_normal(f"  NOW.md: {click.style('✓ OK' if now_ok else '✗ Failed', fg=now_color)}", verbosity)
        echo_normal(f"  MEMORY.md: {click.style('✓ OK' if mem_ok else '✗ Failed', fg=mem_color)}", verbosity)
    except Exception as e:
        echo_normal(f"  {click.style(f'Check failed: {e}', fg='yellow')}", verbosity)

    # Overall health
    echo_quiet(f"\n" + click.style("Overall: ", bold=True) + click.style("HEALTHY ✓", fg="green", bold=True), verbosity)


@monitoring_group.command()
@click.pass_context
def audit(ctx) -> None:
    """Run security audit.

    Checks:
    - File integrity (NOW.md, MEMORY.md)
    - Graph topology (orphan nodes, sudden cores)
    - Git history for suspicious modifications
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(click.style("Running Security Audit...", fg="cyan", bold=True), verbosity)
    echo_normal("=" * 50, verbosity)

    db_path = base_path / "palace.sqlite"
    try:
        detector = PoisonDetector(base_path, GraphPalace(db_path) if db_path.exists() else None)
        results = detector.full_security_audit()

        # File integrity
        echo_normal(f"\n{click.style('File Integrity:', bold=True)}", verbosity)
        file_ok = results.get('file_integrity', False)
        file_status = "✓ VERIFIED" if file_ok else "✗ FAILED"
        file_color = "green" if file_ok else "red"
        echo_normal(f"  Status: {click.style(file_status, fg=file_color)}", verbosity)

        # Topology
        echo_normal(f"\n{click.style('Graph Topology:', bold=True)}", verbosity)
        orphans = results.get('orphan_nodes', [])
        cores = results.get('sudden_cores', [])

        if orphans:
            echo_normal(click.style(f"  ⚠ {len(orphans)} orphan nodes detected", fg="yellow"), verbosity)
        else:
            echo_normal(click.style(f"  ✓ No orphan nodes", fg="green"), verbosity)

        if cores:
            echo_normal(click.style(f"  ⚠ {len(cores)} sudden cores detected", fg="yellow"), verbosity)
        else:
            echo_normal(click.style(f"  ✓ No sudden cores", fg="green"), verbosity)

        # Git audit
        echo_normal(f"\n{click.style('Git History:', bold=True)}", verbosity)
        git_check = results.get('git_audit', {})
        if 'error' in git_check:
            echo_normal(click.style(f"  ⚠ {git_check['error']}", fg="yellow"), verbosity)
        else:
            commits = git_check.get('recent_commits', 0)
            suspicious = git_check.get('suspicious', [])
            echo_verbose(f"  Recent commits: {commits}", verbosity)
            if suspicious:
                echo_normal(click.style(f"  ⚠ {len(suspicious)} suspicious commits", fg="yellow"), verbosity)
            else:
                echo_normal(click.style(f"  ✓ No suspicious commits", fg="green"), verbosity)

        # Overall
        overall = results.get('overall_safe', False)
        if overall:
            echo_quiet(f"\n" + click.style("Overall Safety: ", bold=True) + click.style("SAFE ✓", fg="green", bold=True), verbosity)
        else:
            echo_quiet(f"\n" + click.style("Overall Safety: ", bold=True) + click.style("ATTENTION REQUIRED ⚠", fg="yellow", bold=True), verbosity)

    except Exception as e:
        echo_quiet(click.style(f"Error: Audit failed: {e}", fg="red"), verbosity)
        echo_verbose("Traceback:", verbosity)
        if verbosity >= 2:  # Only show traceback in verbose mode
            import traceback
            traceback.print_exc()
        sys.exit(1)
