"""Policy management commands for OMI CLI."""
import sys
from pathlib import Path
import click

# OMI imports
from omi import GraphPalace
from omi.policies import (
    PolicyEngine,
    get_default_policies,
    load_policies_from_config,
    PolicyAction
)

# Local CLI imports
from .common import (
    get_base_path,
    echo_quiet,
    echo_normal,
    echo_verbose,
    VERBOSITY_NORMAL
)


@click.group()
def policy_group():
    """Policy management commands."""
    pass


@policy_group.command("dry-run")
@click.option('--policy', '-p', help='Specific policy name to run (runs all if not specified)')
@click.pass_context
def dry_run(ctx, policy: str) -> None:
    """Preview what policies would do without executing them.

    Runs all enabled policies in dry-run mode to show which memories
    would be affected by each policy action (archive, delete, etc.)
    without actually making any changes.

    Args:
        --policy: Optional policy name to run (runs all if not specified)

    Examples:
        omi policy dry-run
        omi policy dry-run --policy archive-old-memories
        omi policy dry-run -p cleanup-low-confidence
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        echo_quiet(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    config_path = base_path / "config.yaml"

    echo_normal(click.style("Policy Dry Run", fg="cyan", bold=True), verbosity)
    echo_normal("=" * 60, verbosity)
    echo_verbose(f"Loading policies from {config_path if config_path.exists() else 'defaults'}...", verbosity)

    # Load policies from config or use defaults
    try:
        if config_path.exists():
            policies = load_policies_from_config(config_path)
            if not policies:
                echo_verbose("No policies in config.yaml, using defaults", verbosity)
                policies = get_default_policies()
        else:
            echo_verbose("No config.yaml found, using default policies", verbosity)
            policies = get_default_policies()
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to load policies: {e}", fg="red"), verbosity)
        sys.exit(1)

    # Filter to specific policy if requested
    if policy:
        policies = [p for p in policies if p.name == policy]
        if not policies:
            echo_quiet(click.style(f"Error: Policy '{policy}' not found", fg="red"), verbosity)
            sys.exit(1)

    # Initialize PolicyEngine
    try:
        palace = GraphPalace(db_path)
        engine = PolicyEngine(palace)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to initialize policy engine: {e}", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(f"\nFound {len(policies)} {'policy' if len(policies) == 1 else 'policies'} to evaluate\n", verbosity)

    # Execute each policy in dry-run mode
    total_affected = 0
    for pol in policies:
        if not pol.enabled:
            echo_verbose(f"Skipping disabled policy: {pol.name}", verbosity)
            continue

        echo_normal(click.style(f"Policy: {pol.name}", fg="cyan", bold=True), verbosity)
        if pol.description:
            echo_normal(f"  {pol.description}", verbosity)

        try:
            results = engine.execute(pol, dry_run=True)

            for result in results:
                if result.error:
                    echo_normal(click.style(f"  ✗ Error: {result.error}", fg="red"), verbosity)
                    continue

                count = len(result.affected_memory_ids)
                total_affected += count

                # Color code based on action type
                action_colors = {
                    PolicyAction.ARCHIVE: "yellow",
                    PolicyAction.DELETE: "red",
                    PolicyAction.COMPRESS: "blue",
                    PolicyAction.PROMOTE: "green",
                    PolicyAction.DEMOTE: "magenta"
                }
                action_color = action_colors.get(result.action, "white")

                action_name = result.action.value.upper()
                echo_normal(
                    f"  • {click.style(action_name, fg=action_color)}: "
                    f"{click.style(str(count), bold=True)} {'memory' if count == 1 else 'memories'}",
                    verbosity
                )

                if count > 0 and verbosity >= 2:  # Show IDs in verbose mode
                    for mem_id in result.affected_memory_ids[:5]:  # Show first 5
                        echo_verbose(f"    - {mem_id[:16]}...", verbosity)
                    if count > 5:
                        echo_verbose(f"    ... and {count - 5} more", verbosity)

        except Exception as e:
            echo_quiet(click.style(f"  Error executing policy: {e}", fg="red"), verbosity)
            continue

        echo_normal("", verbosity)  # Blank line between policies

    # Summary
    echo_normal("─" * 60, verbosity)
    echo_normal(
        click.style(f"Dry run complete: {total_affected} memories would be affected",
                   fg="green" if total_affected > 0 else "yellow", bold=True),
        verbosity
    )
    echo_normal(click.style("No changes were made (dry-run mode)", fg="cyan"), verbosity)
