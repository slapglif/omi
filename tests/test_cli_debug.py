"""Tests for OMI CLI debug command

Uses Click's test runner for command testing.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest
from click.testing import CliRunner


class TestDebug:
    """Tests for 'omi debug' command."""

    def test_debug_requires_init(self):
        """Test that debug requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_debug_requires_database(self):
        """Test that debug requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 1
            assert "database not found" in result.output.lower()

    def test_debug_recall_with_embedding_failure(self):
        """Test debug recall when embedding generation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder to raise an exception
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.side_effect = Exception("API key not set")
                mock_embedder.return_value = mock_embedder_instance

                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 1
            assert "failed to generate embedding" in result.output.lower()
            assert "tip:" in result.output.lower()

    def test_debug_recall_success(self):
        """Test debug recall with successful execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.storage.graph_palace import Memory
            from datetime import datetime

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create mock memory
            mock_memory = Memory(
                id="test-id-123",
                content="Test memory content for debugging",
                memory_type="experience",
                confidence=None,
                embedding=None,
                created_at=datetime.now(),
                access_count=0,
                last_accessed=None
            )

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'recall', return_value=[(mock_memory, 0.95)]):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 0
            assert "=== DEBUG: Recall Operation ===" in result.output
            assert "Step 1: Generating embedding" in result.output
            assert "Step 2: Searching for candidate memories" in result.output
            assert "Step 3: Scoring results" in result.output
            assert "Step 4: Final Results" in result.output
            assert "test memory content" in result.output.lower()

    def test_debug_recall_no_results(self):
        """Test debug recall when no memories are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'recall', return_value=[]):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 0
            assert "no memories found" in result.output.lower()

    def test_debug_recall_with_custom_limit(self):
        """Test debug recall with custom limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'recall', return_value=[]) as mock_recall:
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["debug", "--operation", "recall", "test query", "--limit", "5"])

                    # Verify that recall was called with the correct limit
                    mock_recall.assert_called_once()
                    call_kwargs = mock_recall.call_args[1]
                    assert call_kwargs['limit'] == 5

            assert result.exit_code == 0

    def test_debug_store_success(self):
        """Test debug store with successful execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'store_memory', return_value="test-memory-id-123"):
                    with patch.object(GraphPalace, 'recall', return_value=[]):
                        with patch.object(GraphPalace, 'close'):
                            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                                result = runner.invoke(cli, [
                                    "debug",
                                    "--operation", "store",
                                    "Test memory content",
                                    "--type", "experience"
                                ])

            assert result.exit_code == 0
            assert "=== DEBUG: Store Operation ===" in result.output
            assert "Step 1: Input Validation" in result.output
            assert "Step 2: Content Hash Generation" in result.output
            assert "Step 3: Generating Embedding" in result.output
            assert "Step 4: Database Insertion" in result.output
            assert "Step 5: Edge Creation" in result.output
            assert "Step 6: Confirmation" in result.output
            assert "test memory content" in result.output.lower()

    def test_debug_store_with_belief_and_confidence(self):
        """Test debug store with belief type and confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'store_memory', return_value="test-id"):
                    with patch.object(GraphPalace, 'recall', return_value=[]):
                        with patch.object(GraphPalace, 'close'):
                            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                                result = runner.invoke(cli, [
                                    "debug",
                                    "--operation", "store",
                                    "Test belief",
                                    "--type", "belief",
                                    "--confidence", "0.85"
                                ])

            assert result.exit_code == 0
            assert "belief" in result.output.lower()
            assert "0.85" in result.output

    def test_debug_store_with_confidence_warning(self):
        """Test debug store shows warning when confidence used with non-belief type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'store_memory', return_value="test-id"):
                    with patch.object(GraphPalace, 'recall', return_value=[]):
                        with patch.object(GraphPalace, 'close'):
                            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                                result = runner.invoke(cli, [
                                    "debug",
                                    "--operation", "store",
                                    "Test fact",
                                    "--type", "fact",
                                    "--confidence", "0.9"
                                ])

            assert result.exit_code == 0
            assert "warning" in result.output.lower()
            assert "typically used with --type belief" in result.output.lower()

    def test_debug_store_with_similar_memories(self):
        """Test debug store creates edges to similar memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.storage.graph_palace import Memory
            from datetime import datetime

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create mock similar memory
            mock_similar_memory = Memory(
                id="similar-memory-id",
                content="Similar memory content",
                memory_type="experience",
                confidence=None,
                embedding=None,
                created_at=datetime.now(),
                access_count=0,
                last_accessed=None
            )

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                # Create a mock palace instance with add_edge method
                mock_palace = MagicMock()
                mock_palace.store_memory.return_value = "new-memory-id"
                mock_palace.recall.return_value = [(mock_similar_memory, 0.85)]
                mock_palace.add_edge = MagicMock()
                mock_palace.close = MagicMock()

                with patch.object(GraphPalace, '__init__', return_value=None):
                    with patch('omi.cli.GraphPalace', return_value=mock_palace):
                        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                            result = runner.invoke(cli, [
                                "debug",
                                "--operation", "store",
                                "Test memory",
                                "--type", "experience"
                            ])

                        # Verify edge was created
                        mock_palace.add_edge.assert_called_once()

            assert result.exit_code == 0
            assert "RELATED_TO" in result.output
            assert "similar memory" in result.output.lower()

    def test_debug_store_with_embedding_failure(self):
        """Test debug store when embedding generation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder to raise an exception
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.side_effect = Exception("API error")
                mock_embedder.return_value = mock_embedder_instance

                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, [
                        "debug",
                        "--operation", "store",
                        "Test content",
                        "--type", "experience"
                    ])

            assert result.exit_code == 1
            assert "failed to generate embedding" in result.output.lower()

    def test_debug_store_with_database_failure(self):
        """Test debug store when database insertion fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'store_memory', side_effect=Exception("Database error")):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, [
                            "debug",
                            "--operation", "store",
                            "Test content",
                            "--type", "experience"
                        ])

            assert result.exit_code == 1
            assert "failed to store memory" in result.output.lower()

    def test_debug_store_all_memory_types(self):
        """Test debug store with all memory types."""
        memory_types = ['fact', 'experience', 'belief', 'decision']

        for mem_type in memory_types:
            with tempfile.TemporaryDirectory() as tmpdir:
                runner = CliRunner()
                base_path = Path(tmpdir) / "omi"

                sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
                from omi.cli import cli
                from omi import GraphPalace

                # Initialize first
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    runner.invoke(cli, ["init"])

                # Mock NIMEmbedder and GraphPalace
                with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                    mock_embedder_instance = MagicMock()
                    mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                    mock_embedder.return_value = mock_embedder_instance

                    with patch.object(GraphPalace, 'store_memory', return_value="test-id"):
                        with patch.object(GraphPalace, 'recall', return_value=[]):
                            with patch.object(GraphPalace, 'close'):
                                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                                    result = runner.invoke(cli, [
                                        "debug",
                                        "--operation", "store",
                                        f"Test {mem_type} content",
                                        "--type", mem_type
                                    ])

                assert result.exit_code == 0, f"Failed for memory type: {mem_type}"
                assert mem_type in result.output.lower()

    def test_debug_recall_search_failure(self):
        """Test debug recall when search operation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NIMEmbedder and GraphPalace
            with patch('omi.embeddings.NIMEmbedder') as mock_embedder:
                mock_embedder_instance = MagicMock()
                mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedder.return_value = mock_embedder_instance

                with patch.object(GraphPalace, 'recall', side_effect=Exception("Search error")):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["debug", "--operation", "recall", "test query"])

            assert result.exit_code == 1
            assert "search failed" in result.output.lower()

    def test_debug_operation_required(self):
        """Test that debug command requires --operation flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["debug", "test content"])

            assert result.exit_code != 0
            assert "missing option" in result.output.lower() or "required" in result.output.lower()
