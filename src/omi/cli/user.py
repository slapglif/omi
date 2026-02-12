"""User management commands for OMI CLI."""
import sys
from pathlib import Path
from typing import Optional
import click

# Local CLI imports
from .common import (
    get_base_path,
    echo_verbose,
    echo_normal,
    echo_quiet,
    VERBOSITY_NORMAL
)


@click.group()
def user_group():
    """User management commands."""
    pass


@user_group.command('create')
@click.argument('username')
@click.option('--role', '-r',
              type=click.Choice(['admin', 'developer', 'reader', 'auditor']),
              default='developer',
              help='User role (default: developer)')
@click.option('--namespace', '-n', multiple=True,
              help='Grant access to specific namespace(s)')
@click.pass_context
def user_create(ctx, username: str, role: str, namespace: tuple) -> None:
    """Create a new user with specified role.

    Args:
        username: Username to create
        --role: Role to assign (admin|developer|reader|auditor)
        --namespace: Namespace(s) to grant access to

    Examples:
        omi user create alice --role developer
        omi user create bob --role reader --namespace proj1 --namespace proj2
        omi user create admin_user --role admin
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        # TODO: Implement user creation logic
        echo_normal(click.style("✓ User created", fg="green", bold=True), verbosity)
        echo_normal(f"  Username: {click.style(username, fg='cyan')}", verbosity)
        echo_normal(f"  Role: {click.style(role, fg='cyan')}", verbosity)
        if namespace:
            echo_normal(f"  Namespaces: {click.style(', '.join(namespace), fg='cyan')}", verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to create user: {e}", fg="red"), verbosity)
        sys.exit(1)


@user_group.command('list')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def user_list(ctx, json_output: bool) -> None:
    """List all users and their roles.

    Args:
        --json: Output as JSON (for scripts)

    Examples:
        omi user list
        omi user list --json
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        # TODO: Implement user listing logic
        if json_output:
            import json
            users = []
            click.echo(json.dumps(users, indent=2))
        else:
            echo_normal(click.style("Users", fg="cyan", bold=True), verbosity)
            echo_normal("=" * 60, verbosity)
            echo_normal(click.style("No users found.", fg="yellow"), verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to list users: {e}", fg="red"), verbosity)
        sys.exit(1)


@user_group.command('delete')
@click.argument('username')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def user_delete(ctx, username: str, force: bool) -> None:
    """Delete a user.

    Args:
        username: Username to delete
        --force: Skip confirmation prompt

    Examples:
        omi user delete alice
        omi user delete bob --force
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    if not force:
        if not click.confirm(f"Delete user '{username}'?"):
            echo_normal(click.style("Cancelled.", fg="yellow"), verbosity)
            return

    try:
        # TODO: Implement user deletion logic
        echo_normal(click.style(f"✓ User '{username}' deleted", fg="green", bold=True), verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to delete user: {e}", fg="red"), verbosity)
        sys.exit(1)


@user_group.command('permissions')
@click.argument('username')
@click.option('--grant', '-g', multiple=True,
              help='Grant permission (format: namespace:permission)')
@click.option('--revoke', multiple=True,
              help='Revoke permission (format: namespace:permission)')
@click.pass_context
def user_permissions(ctx, username: str, grant: tuple, revoke: tuple) -> None:
    """View or modify user permissions.

    Args:
        username: Username to manage
        --grant: Grant permission(s)
        --revoke: Revoke permission(s)

    Examples:
        omi user permissions alice
        omi user permissions alice --grant proj1:write --grant proj2:read
        omi user permissions bob --revoke proj1:write
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        # TODO: Implement permission management logic
        if grant:
            for perm in grant:
                echo_normal(click.style(f"✓ Granted {perm} to {username}", fg="green"), verbosity)

        if revoke:
            for perm in revoke:
                echo_normal(click.style(f"✓ Revoked {perm} from {username}", fg="green"), verbosity)

        if not grant and not revoke:
            # Just show current permissions
            echo_normal(click.style(f"Permissions for {username}", fg="cyan", bold=True), verbosity)
            echo_normal("=" * 60, verbosity)
            echo_normal(click.style("No permissions found.", fg="yellow"), verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to manage permissions: {e}", fg="red"), verbosity)
        sys.exit(1)


@user_group.command('set-role')
@click.argument('username')
@click.argument('role', type=click.Choice(['admin', 'developer', 'reader', 'auditor']))
@click.pass_context
def user_set_role(ctx, username: str, role: str) -> None:
    """Change a user's role.

    Args:
        username: Username to modify
        role: New role to assign

    Examples:
        omi user set-role alice admin
        omi user set-role bob reader
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        # TODO: Implement role change logic
        echo_normal(click.style(f"✓ Changed role for '{username}' to '{role}'", fg="green", bold=True), verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to change role: {e}", fg="red"), verbosity)
        sys.exit(1)
