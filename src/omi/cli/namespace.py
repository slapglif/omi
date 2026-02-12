"""Namespace management CLI commands"""
import click


@click.group()
@click.pass_context
def namespace_group(ctx: click.Context) -> None:
    """Namespace management commands."""
    ctx.ensure_object(dict)
