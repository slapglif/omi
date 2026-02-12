"""
Policy management commands for OMI CLI.

This module provides CLI commands for managing memory lifecycle policies:
- show: Display all configured policies and their status
- dry-run: Preview policy execution without making changes
- execute: Run policies and apply their actions to memories

All commands support verbosity levels and work with both configured and default policies.
"""
import sys
from pathlib import Path
from typing import Optional
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
def policy_group() -> None:
    """
    Policy management commands for memory lifecycle automation.

    This command group provides tools for managing, previewing, and executing
    memory lifecycle policies that control automatic archival, deletion, and
    other memory management actions.

    Available commands:
        - show: Display all configured policies and their status
        - dry-run: Preview policy effects without making changes
        - execute: Run policies and apply their actions

    See 'omi policy COMMAND --help' for more information on a specific command.
    """
    pass


@policy_group.command("show")
@click.pass_context
def show(ctx: click.Context) -> None:
    """
    Display all policies and their status.

    Lists all configured policies (or defaults if no config exists),
    showing their enabled/disabled status, descriptions, and conditions.

    Args:
        ctx: Click context object containing verbosity and data_dir settings

    Returns:
        None. Outputs policy information to stdout and exits with code 0 on success,
        1 on error (e.g., OMI not initialized).

    Examples:
        omi policy show
        omi policy show -v  # Verbose output with conditions
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    config_path = base_path / "config.yaml"

    echo_normal(click.style("Policy Configuration", fg="cyan", bold=True), verbosity)
    echo_normal("=" * 60, verbosity)
    echo_verbose(f"Loading policies from {config_path if config_path.exists() else 'defaults'}...", verbosity)

    # Load policies from config or use defaults
    try:
        if config_path.exists():
            policies = load_policies_from_config(config_path)
            if not policies:
                echo_verbose("No policies in config.yaml, using defaults", verbosity)
                policies = get_default_policies()
                source = "defaults"
            else:
                source = str(config_path)
        else:
            echo_verbose("No config.yaml found, using default policies", verbosity)
            policies = get_default_policies()
            source = "defaults"
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to load policies: {e}", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(f"Source: {source}", verbosity)
    echo_normal(f"Total policies: {len(policies)}\n", verbosity)

    # Display each policy
    enabled_count = 0
    for pol in policies:
        if pol.enabled:
            enabled_count += 1
            status = click.style("ENABLED", fg="green", bold=True)
        else:
            status = click.style("DISABLED", fg="red")

        echo_normal(click.style(f"• {pol.name}", fg="cyan", bold=True), verbosity)
        echo_normal(f"  Status: {status}", verbosity)

        if pol.description:
            echo_normal(f"  Description: {pol.description}", verbosity)

        # Show rules in verbose mode
        if verbosity >= 2 and pol.rules:
            echo_verbose(f"  Rules: {len(pol.rules)}", verbosity)
            for rule in pol.rules:
                rule_status = "enabled" if rule.enabled else "disabled"
                echo_verbose(f"    - {rule.name} ({rule_status})", verbosity)
                if rule.conditions:
                    for key, value in rule.conditions.items():
                        echo_verbose(f"      • {key}: {value}", verbosity)

        echo_normal("", verbosity)  # Blank line between policies

    # Summary
    echo_normal("─" * 60, verbosity)
    disabled_count = len(policies) - enabled_count
    echo_normal(
        f"{click.style(str(enabled_count), fg='green', bold=True)} enabled, "
        f"{click.style(str(disabled_count), fg='red')} disabled",
        verbosity
    )


@policy_group.command("dry-run")
@click.option('--policy', '-p', help='Specific policy name to run (runs all if not specified)')
@click.pass_context
def dry_run(ctx: click.Context, policy: Optional[str]) -> None:
    """
    Preview what policies would do without executing them.

    Runs all enabled policies in dry-run mode to show which memories
    would be affected by each policy action (archive, delete, etc.)
    without actually making any changes.

    Args:
        ctx: Click context object containing verbosity and data_dir settings
        policy: Optional policy name to run. If None, runs all enabled policies.

    Returns:
        None. Outputs dry-run results to stdout and exits with code 0 on success,
        1 on error (e.g., OMI not initialized, policy not found).

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


@policy_group.command("execute")
@click.option('--policy', '-p', help='Specific policy name to run (runs all if not specified)')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def execute(ctx: click.Context, policy: Optional[str], yes: bool) -> None:
    """
    Execute policies and apply their actions to memories.

    Runs all enabled policies and applies their actions (archive, delete, etc.)
    to the memory system. Prompts for confirmation unless --yes flag is provided.
    Use dry-run first to safely preview changes before execution.

    Args:
        ctx: Click context object containing verbosity and data_dir settings
        policy: Optional policy name to run. If None, runs all enabled policies.
        yes: If True, skip confirmation prompt and execute immediately

    Returns:
        None. Outputs execution results to stdout and exits with code 0 on success,
        1 on error (e.g., OMI not initialized, policy not found).

    Safety:
        - Shows warning prompt before execution (unless --yes is provided)
        - Only executes enabled policies
        - Reports detailed results including success counts and errors
        - Locked memories are automatically exempt from all actions

    Examples:
        omi policy execute
        omi policy execute --policy archive-old-memories
        omi policy execute -p cleanup-low-confidence --yes
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

    echo_normal(click.style("Policy Execution", fg="cyan", bold=True), verbosity)
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

    # Filter to enabled policies only
    enabled_policies = [p for p in policies if p.enabled]
    if not enabled_policies:
        echo_normal(click.style("No enabled policies to execute", fg="yellow"), verbosity)
        sys.exit(0)

    echo_normal(f"\nFound {len(enabled_policies)} enabled {'policy' if len(enabled_policies) == 1 else 'policies'}\n", verbosity)

    # Confirmation prompt (unless --yes flag is set)
    if not yes:
        echo_normal(click.style("⚠ Warning: This will modify your memory system", fg="yellow", bold=True), verbosity)
        echo_normal("Run 'omi policy dry-run' first to preview changes.", verbosity)
        if not click.confirm("\nProceed with policy execution?"):
            echo_normal(click.style("Execution cancelled", fg="yellow"), verbosity)
            sys.exit(0)
        echo_normal("", verbosity)

    # Initialize PolicyEngine
    try:
        palace = GraphPalace(db_path)
        engine = PolicyEngine(palace)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to initialize policy engine: {e}", fg="red"), verbosity)
        sys.exit(1)

    # Execute each policy
    total_affected = 0
    total_errors = 0
    for pol in enabled_policies:
        echo_normal(click.style(f"Policy: {pol.name}", fg="cyan", bold=True), verbosity)
        if pol.description:
            echo_normal(f"  {pol.description}", verbosity)

        try:
            results = engine.execute(pol, dry_run=False)

            for result in results:
                if result.error:
                    echo_normal(click.style(f"  ✗ Error: {result.error}", fg="red"), verbosity)
                    total_errors += 1
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
                    f"  ✓ {click.style(action_name, fg=action_color)}: "
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
            total_errors += 1
            continue

        echo_normal("", verbosity)  # Blank line between policies

    # Summary
    echo_normal("─" * 60, verbosity)
    if total_errors > 0:
        echo_normal(
            click.style(f"Execution complete with errors: {total_affected} memories affected, {total_errors} errors",
                       fg="yellow", bold=True),
            verbosity
        )
    else:
        echo_normal(
            click.style(f"Execution complete: {total_affected} memories affected",
                       fg="green", bold=True),
            verbosity
        )
    echo_normal(click.style("Changes have been applied to the memory system", fg="cyan"), verbosity)
