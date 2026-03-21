"""OMI CLI - Vector index management commands."""
import sys
import sqlite3
import struct
from pathlib import Path
from typing import Dict, List
import click
from .common import get_base_path


@click.group()
@click.pass_context
def index_group(ctx: click.Context) -> None:
    """Vector index management commands."""
    ctx.ensure_object(dict)


@index_group.command(name='rebuild')
@click.option('--force', is_flag=True, help='Rebuild even if index already exists')
@click.pass_context
def rebuild(ctx: click.Context, force: bool) -> None:
    """Rebuild ANN index from existing embeddings.

    Fetches all memories with embeddings from the database and rebuilds
    the HNSW index for fast similarity search. Automatically handles
    multiple embedding dimensions (768 for Ollama, 1024 for NIM).

    Examples:
        omi index rebuild
        omi index rebuild --force
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    db_path = base_path / "palace.sqlite"

    if not db_path.exists():
        click.echo(click.style("Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        from omi.storage.ann_index import ANNIndex
    except ImportError as e:
        click.echo(click.style(f"Error: Failed to import ANNIndex: {e}", fg="red"))
        click.echo("Make sure hnswlib is installed: pip install hnswlib")
        sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Fetch all memories with embeddings
    click.echo("📊 Fetching memories with embeddings for index rebuild...")
    cursor.execute("""
        SELECT id, embedding
        FROM memories
        WHERE embedding IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        click.echo(click.style("⚠️  No memories with embeddings found. Index rebuild skipped.", fg="yellow"))
        return

    click.echo(f"Found {len(rows)} memories with embeddings")

    # Decode embeddings and group by dimension
    click.echo("🔍 Analyzing embedding dimensions...")
    embeddings_by_dim: Dict[int, List[tuple]] = {}

    for memory_id, embedding_blob in rows:
        # Decode embedding from binary blob
        embedding = list(struct.unpack(f'{len(embedding_blob) // 4}f', embedding_blob))
        dim = len(embedding)

        if dim not in embeddings_by_dim:
            embeddings_by_dim[dim] = []
        embeddings_by_dim[dim].append((memory_id, embedding))

    # Rebuild index for each dimension
    for dim, embeddings_list in embeddings_by_dim.items():
        click.echo(f"\n🔧 Rebuilding {dim}-dimensional index ({len(embeddings_list)} vectors)...")

        # Create ANNIndex
        index = ANNIndex(str(db_path), dim=dim, enable_persistence=True)

        # Rebuild from embeddings
        index.rebuild_from_embeddings(embeddings_list)

        # Save to disk
        index.save()

        click.echo(click.style(f"✓ {dim}-dim index rebuilt successfully", fg="green"))

    click.echo(click.style(f"\n✓ Index rebuild complete!", fg="green"))
