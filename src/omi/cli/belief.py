"""Belief management CLI commands."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click

from .common import get_base_path


@click.group()
@click.pass_context
def belief_group(ctx: click.Context) -> None:
    """Belief management commands."""
    ctx.ensure_object(dict)


@belief_group.command("evidence")
@click.argument("belief_id")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def belief_evidence(ctx: click.Context, belief_id: str, json_output: bool) -> None:
    """Display evidence chain for a belief."""
    from omi.storage.graph_palace import GraphPalace
    from omi.belief import BeliefNetwork

    base_path = get_base_path(ctx.obj.get("data_dir"))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style("Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        palace = GraphPalace(db_path)
        belief_network = BeliefNetwork(palace)

        belief = palace.get_belief(belief_id)
        if not belief:
            click.echo(click.style(f"Error: Belief '{belief_id}' not found.", fg="red"))
            palace.close()
            sys.exit(1)

        evidence_chain = belief_network.get_evidence_chain(belief_id)

        if json_output:
            output = [
                {
                    "memory_id": e.memory_id,
                    "supports": e.supports,
                    "strength": e.strength,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in evidence_chain
            ]
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo(click.style(f"Belief: {belief.get('content', '')}", fg="cyan", bold=True))
            click.echo(f"Current Confidence: {belief.get('confidence', 0.0):.2f}")
            click.echo("=" * 60)
            if not evidence_chain:
                click.echo(click.style("\nNo evidence entries found.", fg="yellow"))
            else:
                click.echo(click.style(f"\nEvidence Chain ({len(evidence_chain)} entries)", fg="cyan", bold=True))
                for i, e in enumerate(evidence_chain, 1):
                    support_text = "SUPPORTS" if e.supports else "CONTRADICTS"
                    color = "green" if e.supports else "red"
                    click.echo(f"{i}. {click.style(support_text, fg=color, bold=True)}")
                    click.echo(f"   Memory ID: {e.memory_id}")
                    click.echo(f"   Strength: {e.strength:.2f}")
                    click.echo(f"   Timestamp: {e.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    try:
                        memory = palace.get_memory(e.memory_id)
                        if memory:
                            content = memory.get("content", "")[:80]
                            click.echo(f"   Content: {content}")
                    except Exception:
                        pass

        palace.close()
    except Exception as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        sys.exit(1)


@belief_group.command("update")
@click.argument("belief_id")
@click.option("--evidence", required=True, help="Memory ID to use as evidence")
@click.option("--supports", "evidence_type", flag_value="supports", default=True,
              help="Evidence supports the belief (default)")
@click.option("--contradicts", "evidence_type", flag_value="contradicts",
              help="Evidence contradicts the belief")
@click.option("--strength", type=float, default=0.8, help="Evidence strength (0.0-1.0)")
@click.pass_context
def belief_update(
    ctx: click.Context, belief_id: str, evidence: str, evidence_type: str, strength: float
) -> None:
    """Update a belief with new evidence."""
    from omi.storage.graph_palace import GraphPalace
    from omi.belief import BeliefNetwork, Evidence

    base_path = get_base_path(ctx.obj.get("data_dir"))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style("Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    if not 0.0 <= strength <= 1.0:
        click.echo(click.style(f"Error: Strength must be between 0.0 and 1.0, got {strength}", fg="red"))
        sys.exit(1)

    try:
        palace = GraphPalace(db_path)
        belief_network = BeliefNetwork(palace)

        belief = palace.get_belief(belief_id)
        if not belief:
            click.echo(click.style(f"Error: Belief '{belief_id}' not found.", fg="red"))
            palace.close()
            sys.exit(1)

        memory = palace.get_memory(evidence)
        if not memory:
            click.echo(click.style(f"Error: Evidence memory '{evidence}' not found.", fg="red"))
            palace.close()
            sys.exit(1)

        supports = evidence_type == "supports"
        evidence_obj = Evidence(
            memory_id=evidence,
            supports=supports,
            strength=strength,
            timestamp=datetime.now(),
        )

        old_confidence = belief.get("confidence", 0.5)
        new_confidence = belief_network.update_with_evidence(belief_id, evidence_obj)

        click.echo(click.style(f"Belief: {belief.get('content', '')}", fg="cyan", bold=True))
        click.echo("=" * 60)
        click.echo(f"Evidence Type: {'SUPPORTING' if supports else 'CONTRADICTING'}")
        click.echo(f"Evidence ID:   {evidence}")
        click.echo(f"Strength:      {strength:.2f}")
        click.echo("Confidence Update:")
        click.echo(f"  Old: {old_confidence:.4f}")
        click.echo(f"  New: {new_confidence:.4f}")

        palace.close()
    except SystemExit:
        raise
    except Exception as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"))
        sys.exit(1)
