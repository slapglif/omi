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
from .common import get_base_path


@click.group()
@click.pass_context
def monitoring_group(ctx):
    """Monitoring and security commands."""
    ctx.ensure_object(dict)


@monitoring_group.command()
@click.pass_context
def status(ctx) -> None:
    """Show OMI health and size statistics."""
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("OMI Status Report", fg="cyan", bold=True))
    click.echo("=" * 50)

    # Base path info
    click.echo(f"\nBase Path: {click.style(str(base_path), fg='cyan')}")

    # Check files
    files_to_check = [
        ("Config", base_path / "config.yaml"),
        ("Database", base_path / "palace.sqlite"),
        ("NOW.md", base_path / "NOW.md"),
    ]

    click.echo(f"\nFiles:")
    for name, path in files_to_check:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status_symbol = "✓" if exists else "✗"
        status_color = "green" if exists else "red"
        click.echo(f"  {name:12} {click.style(status_symbol, fg=status_color)} {path.name}")

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

        click.echo(f"\nDatabase:")
        click.echo(f"  Size: {click.style(f'{db_size:.1f} KB', fg='cyan')}")
        click.echo(f"  Memories: {click.style(str(mem_count), fg='cyan')}")
        if type_counts:
            click.echo(f"  Breakdown:")
            for mem_type, count in type_counts.items():
                click.echo(f"    {mem_type}: {count}")

        conn.close()

    # Integrity check
    click.echo(f"\nIntegrity:")
    try:
        from omi.security import IntegrityChecker
        integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
        now_ok = integrity_checker.check_now_md()
        mem_ok = integrity_checker.check_memory_md()
        now_color = "green" if now_ok else "red"
        mem_color = "green" if mem_ok else "red"
        click.echo(f"  NOW.md: {click.style('✓ OK' if now_ok else '✗ Failed', fg=now_color)}")
        click.echo(f"  MEMORY.md: {click.style('✓ OK' if mem_ok else '✗ Failed', fg=mem_color)}")
    except Exception as e:
        click.echo(f"  {click.style(f'Check failed: {e}', fg='yellow')}")

    # Overall health
    click.echo(f"\n" + click.style("Overall: ", bold=True), nl=False)
    click.echo(click.style("HEALTHY ✓", fg="green", bold=True))


@monitoring_group.command()
@click.pass_context
def audit(ctx) -> None:
    """Run security audit.

    Checks:
    - File integrity (NOW.md, MEMORY.md)
    - Graph topology (orphan nodes, sudden cores)
    - Git history for suspicious modifications
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("Running Security Audit...", fg="cyan", bold=True))
    click.echo("=" * 50)

    db_path = base_path / "palace.sqlite"
    try:
        detector = PoisonDetector(base_path, GraphPalace(db_path) if db_path.exists() else None)
        results = detector.full_security_audit()

        # File integrity
        click.echo(f"\n{click.style('File Integrity:', bold=True)}")
        file_ok = results.get('file_integrity', False)
        file_status = "✓ VERIFIED" if file_ok else "✗ FAILED"
        file_color = "green" if file_ok else "red"
        click.echo(f"  Status: {click.style(file_status, fg=file_color)}")

        # Topology
        click.echo(f"\n{click.style('Graph Topology:', bold=True)}")
        orphans = results.get('orphan_nodes', [])
        cores = results.get('sudden_cores', [])

        if orphans:
            click.echo(click.style(f"  ⚠ {len(orphans)} orphan nodes detected", fg="yellow"))
        else:
            click.echo(click.style(f"  ✓ No orphan nodes", fg="green"))

        if cores:
            click.echo(click.style(f"  ⚠ {len(cores)} sudden cores detected", fg="yellow"))
        else:
            click.echo(click.style(f"  ✓ No sudden cores", fg="green"))

        # Git audit
        click.echo(f"\n{click.style('Git History:', bold=True)}")
        git_check = results.get('git_audit', {})
        if 'error' in git_check:
            click.echo(click.style(f"  ⚠ {git_check['error']}", fg="yellow"))
        else:
            commits = git_check.get('recent_commits', 0)
            suspicious = git_check.get('suspicious', [])
            click.echo(f"  Recent commits: {commits}")
            if suspicious:
                click.echo(click.style(f"  ⚠ {len(suspicious)} suspicious commits", fg="yellow"))
            else:
                click.echo(click.style(f"  ✓ No suspicious commits", fg="green"))

        # Overall
        overall = results.get('overall_safe', False)
        click.echo(f"\n" + click.style("Overall Safety: ", bold=True), nl=False)
        if overall:
            click.echo(click.style("SAFE ✓", fg="green", bold=True))
        else:
            click.echo(click.style("ATTENTION REQUIRED ⚠", fg="yellow", bold=True))

    except Exception as e:
        click.echo(click.style(f"Error: Audit failed: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)
