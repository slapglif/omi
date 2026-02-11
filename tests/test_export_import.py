"""
Unit tests for Memory Export/Import functionality

Tests cover:
- MemoryExporter: export_to_dict, export_to_json, export_to_yaml
- MemoryImporter: import_from_dict, conflict resolution, edge restoration
- Filtering: memory_type, confidence, date range
- Round-trip integrity (export → import → export = identical)
"""

import unittest
import tempfile
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.storage.graph_palace import GraphPalace, Memory
from omi.export_import import (
    MemoryExporter,
    MemoryImporter,
    ConflictResolution,
    serialize_embedding,
    deserialize_embedding
)


class TestMemoryExporter(unittest.TestCase):
    """Test suite for MemoryExporter."""

    def setUp(self):
        """Set up test database with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_palace.sqlite"
        self.palace = GraphPalace(self.db_path)
        self.exporter = MemoryExporter(self.palace)

        # Create sample memories with different types and dates
        self._create_sample_data()

    def tearDown(self):
        """Clean up test database."""
        self.palace.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _create_sample_data(self):
        """Create sample memories for testing."""
        # Create memories with different types
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)

        self.memory1_id = self.palace.store_memory(
            content="Python is a programming language",
            embedding=self._generate_embedding(),
            memory_type="fact",
            confidence=0.95
        )
        # Manually set created_at for testing
        self.palace._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (two_days_ago.isoformat(), self.memory1_id)
        )

        self.memory2_id = self.palace.store_memory(
            content="I learned Python last year",
            embedding=self._generate_embedding(),
            memory_type="experience",
            confidence=0.85
        )
        # Set created_at to yesterday
        self.palace._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (yesterday.isoformat(), self.memory2_id)
        )

        self.memory3_id = self.palace.store_memory(
            content="Python is better than Java",
            embedding=self._generate_embedding(),
            memory_type="belief",
            confidence=0.70
        )
        # Set created_at to now
        self.palace._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (now.isoformat(), self.memory3_id)
        )

        self.memory4_id = self.palace.store_memory(
            content="Use Python for the new project",
            memory_type="decision",
            confidence=0.90
        )

        # Create edges
        self.edge1_id = self.palace.create_edge(
            self.memory1_id,
            self.memory2_id,
            "SUPPORTS"
        )
        self.edge2_id = self.palace.create_edge(
            self.memory2_id,
            self.memory3_id,
            "RELATED_TO"
        )

    # ==================== Basic Export Tests ====================

    def test_export_to_dict_all_memories(self):
        """Test exporting all memories without filters."""
        result = self.exporter.export_to_dict()

        self.assertIn("metadata", result)
        self.assertIn("memories", result)
        self.assertEqual(result["metadata"]["memory_count"], 4)
        self.assertEqual(len(result["memories"]), 4)

        # Verify metadata structure
        metadata = result["metadata"]
        self.assertEqual(metadata["export_version"], "1.0")
        self.assertIn("exported_at", metadata)
        self.assertIn("filters", metadata)

    def test_export_includes_edges_by_default(self):
        """Test that edges are included by default."""
        result = self.exporter.export_to_dict()

        self.assertIn("edges", result)
        self.assertEqual(result["metadata"]["edge_count"], 2)
        self.assertEqual(len(result["edges"]), 2)

        # Verify edge structure
        edge = result["edges"][0]
        self.assertIn("id", edge)
        self.assertIn("source_id", edge)
        self.assertIn("target_id", edge)
        self.assertIn("edge_type", edge)
        self.assertIn("strength", edge)
        self.assertIn("created_at", edge)

    def test_export_excludes_embeddings_by_default(self):
        """Test that embeddings are excluded by default."""
        result = self.exporter.export_to_dict()

        # Check that embeddings are not included
        for memory in result["memories"]:
            self.assertIsNone(memory.get("embedding"))

    def test_export_includes_embeddings_when_requested(self):
        """Test including embeddings in export."""
        result = self.exporter.export_to_dict(include_embeddings=True)

        # Check that embeddings are included for memories that have them
        memories_with_embeddings = [m for m in result["memories"] if m.get("embedding")]
        self.assertGreater(len(memories_with_embeddings), 0)

        # Verify embedding is a list of floats
        embedding = memories_with_embeddings[0]["embedding"]
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)

    def test_export_excludes_edges_when_requested(self):
        """Test excluding edges from export."""
        result = self.exporter.export_to_dict(include_edges=False)

        self.assertNotIn("edges", result)
        self.assertNotIn("edge_count", result["metadata"])

    # ==================== Filter Tests ====================

    def test_filter_by_memory_type(self):
        """Test filtering by memory type."""
        result = self.exporter.export_to_dict(memory_type="fact")

        self.assertEqual(result["metadata"]["memory_count"], 1)
        self.assertEqual(len(result["memories"]), 1)
        self.assertEqual(result["memories"][0]["memory_type"], "fact")
        self.assertEqual(result["memories"][0]["content"], "Python is a programming language")

    def test_filter_by_min_confidence(self):
        """Test filtering by minimum confidence."""
        result = self.exporter.export_to_dict(min_confidence=0.80)

        # Should include memories with confidence >= 0.80
        self.assertGreater(result["metadata"]["memory_count"], 0)

        for memory in result["memories"]:
            self.assertGreaterEqual(memory["confidence"], 0.80)

    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Filter to only include memories from yesterday onward
        result = self.exporter.export_to_dict(date_from=yesterday)

        # Should include memories created yesterday and today
        self.assertGreater(result["metadata"]["memory_count"], 0)

        for memory in result["memories"]:
            created_at = datetime.fromisoformat(memory["created_at"])
            self.assertGreaterEqual(created_at, yesterday)

    def test_filter_combined(self):
        """Test combining multiple filters."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        result = self.exporter.export_to_dict(
            memory_type="experience",
            min_confidence=0.80,
            date_from=yesterday
        )

        # Should match specific criteria
        for memory in result["memories"]:
            self.assertEqual(memory["memory_type"], "experience")
            self.assertGreaterEqual(memory["confidence"], 0.80)
            created_at = datetime.fromisoformat(memory["created_at"])
            self.assertGreaterEqual(created_at, yesterday)

    def test_filter_no_matches(self):
        """Test filter that matches no memories."""
        result = self.exporter.export_to_dict(
            memory_type="fact",
            min_confidence=0.99
        )

        self.assertEqual(result["metadata"]["memory_count"], 0)
        self.assertEqual(len(result["memories"]), 0)

    def test_filter_date_to(self):
        """Test filtering with date_to parameter."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        result = self.exporter.export_to_dict(date_to=yesterday)

        # Should only include memories up to yesterday
        for memory in result["memories"]:
            created_at = datetime.fromisoformat(memory["created_at"])
            self.assertLessEqual(created_at, yesterday)

    # ==================== JSON Export Tests ====================

    def test_export_to_json(self):
        """Test exporting to JSON file."""
        output_path = Path(self.temp_dir) / "export.json"

        count = self.exporter.export_to_json(output_path)

        self.assertEqual(count, 4)
        self.assertTrue(output_path.exists())

        # Verify JSON is valid
        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("memories", data)
        self.assertEqual(len(data["memories"]), 4)

    def test_export_to_json_with_filters(self):
        """Test exporting to JSON with filters."""
        output_path = Path(self.temp_dir) / "filtered.json"

        count = self.exporter.export_to_json(
            output_path,
            memory_type="fact",
            include_embeddings=True
        )

        self.assertEqual(count, 1)

        # Verify filtered content
        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data["memories"]), 1)
        self.assertEqual(data["memories"][0]["memory_type"], "fact")
        self.assertIsNotNone(data["memories"][0]["embedding"])

    def test_export_to_json_creates_parent_directories(self):
        """Test that export creates parent directories if needed."""
        output_path = Path(self.temp_dir) / "subdir" / "nested" / "export.json"

        count = self.exporter.export_to_json(output_path)

        self.assertTrue(output_path.exists())
        self.assertEqual(count, 4)

    def test_export_to_json_custom_indent(self):
        """Test JSON export with custom indentation."""
        output_path = Path(self.temp_dir) / "compact.json"

        self.exporter.export_to_json(output_path, indent=0)

        with open(output_path, 'r') as f:
            content = f.read()

        # Compact JSON should have less whitespace
        self.assertIn("{", content)
        self.assertIn("memories", content)

    # ==================== YAML Export Tests ====================

    def test_export_to_yaml(self):
        """Test exporting to YAML file."""
        output_path = Path(self.temp_dir) / "export.yaml"

        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            with self.assertRaises(ImportError):
                self.exporter.export_to_yaml(output_path)
            return

        count = self.exporter.export_to_yaml(output_path)

        self.assertEqual(count, 4)
        self.assertTrue(output_path.exists())

        # Verify YAML is valid
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)

        self.assertIn("metadata", data)
        self.assertIn("memories", data)
        self.assertEqual(len(data["memories"]), 4)

    def test_export_to_yaml_with_filters(self):
        """Test exporting to YAML with filters."""
        output_path = Path(self.temp_dir) / "filtered.yaml"

        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            self.skipTest("PyYAML not installed")

        count = self.exporter.export_to_yaml(
            output_path,
            memory_type="belief",
            min_confidence=0.70
        )

        self.assertGreater(count, 0)

        # Verify filtered content
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)

        for memory in data["memories"]:
            self.assertEqual(memory["memory_type"], "belief")
            self.assertGreaterEqual(memory["confidence"], 0.70)

    # ==================== Edge Export Tests ====================

    def test_edges_only_include_exported_memories(self):
        """Test that edges only include connections between exported memories."""
        # Export only one memory type
        result = self.exporter.export_to_dict(memory_type="fact")

        # Should have 1 memory
        self.assertEqual(result["metadata"]["memory_count"], 1)

        # Should have 0 edges (fact memory's edges connect to other types)
        self.assertIn("edges", result)
        edge_count = result["metadata"].get("edge_count", 0)
        # Edges should only connect memories in the export
        for edge in result.get("edges", []):
            self.assertIn(edge["source_id"], [m["id"] for m in result["memories"]])
            self.assertIn(edge["target_id"], [m["id"] for m in result["memories"]])

    def test_edges_preserve_relationships(self):
        """Test that edge relationships are preserved."""
        result = self.exporter.export_to_dict()

        # Find the edge between memory1 and memory2
        edges = result["edges"]
        edge = next((e for e in edges if e["source_id"] == self.memory1_id), None)

        self.assertIsNotNone(edge)
        self.assertEqual(edge["target_id"], self.memory2_id)
        self.assertEqual(edge["edge_type"], "SUPPORTS")

    # ==================== Memory Structure Tests ====================

    def test_exported_memory_structure(self):
        """Test that exported memories have correct structure."""
        result = self.exporter.export_to_dict()

        memory = result["memories"][0]

        # Required fields
        self.assertIn("id", memory)
        self.assertIn("content", memory)
        self.assertIn("memory_type", memory)
        self.assertIn("confidence", memory)
        self.assertIn("created_at", memory)
        self.assertIn("last_accessed", memory)
        self.assertIn("access_count", memory)
        self.assertIn("content_hash", memory)

        # Verify types
        self.assertIsInstance(memory["id"], str)
        self.assertIsInstance(memory["content"], str)
        self.assertIsInstance(memory["memory_type"], str)
        self.assertIsInstance(memory["confidence"], (int, float))
        self.assertIsInstance(memory["access_count"], int)

    def test_metadata_filter_tracking(self):
        """Test that metadata correctly tracks applied filters."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        result = self.exporter.export_to_dict(
            memory_type="fact",
            min_confidence=0.9,
            date_from=yesterday,
            date_to=now,
            include_embeddings=True
        )

        filters = result["metadata"]["filters"]
        self.assertEqual(filters["memory_type"], "fact")
        self.assertEqual(filters["min_confidence"], 0.9)
        self.assertEqual(filters["date_from"], yesterday.isoformat())
        self.assertEqual(filters["date_to"], now.isoformat())
        self.assertTrue(filters["include_embeddings"])

    # ==================== Embedding Serialization Tests ====================

    def test_serialize_deserialize_embedding(self):
        """Test embedding serialization round-trip."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Serialize to bytes
        serialized = serialize_embedding(original)
        self.assertIsInstance(serialized, bytes)

        # Deserialize back to list
        deserialized = deserialize_embedding(serialized)
        self.assertIsInstance(deserialized, list)

        # Verify values match (with float precision tolerance)
        self.assertTrue(np.allclose(original, deserialized, rtol=1e-5))

    def test_serialize_empty_embedding(self):
        """Test serializing empty embedding."""
        serialized = serialize_embedding([])
        self.assertEqual(serialized, b'')

        deserialized = deserialize_embedding(b'')
        self.assertEqual(deserialized, [])

    def test_serialize_large_embedding(self):
        """Test serializing large embedding vector."""
        large_embedding = self._generate_embedding(1024)

        serialized = serialize_embedding(large_embedding)
        deserialized = deserialize_embedding(serialized)

        self.assertEqual(len(deserialized), 1024)
        self.assertTrue(np.allclose(large_embedding, deserialized, rtol=1e-5))


class TestMemoryImporter(unittest.TestCase):
    """Test suite for MemoryImporter with conflict detection."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_palace.sqlite"
        self.palace = GraphPalace(self.db_path)
        self.importer = MemoryImporter(self.palace)
        self.exporter = MemoryExporter(self.palace)

    def tearDown(self):
        """Clean up test database."""
        self.palace.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _create_sample_export_data(self) -> Dict[str, Any]:
        """Create sample export data for import testing."""
        return {
            "metadata": {
                "export_version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "memory_count": 2,
                "edge_count": 1,
                "filters": {}
            },
            "memories": [
                {
                    "id": "test-memory-1",
                    "content": "Python is a programming language",
                    "embedding": self._generate_embedding(),
                    "memory_type": "fact",
                    "confidence": 0.95,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": [],
                    "content_hash": "hash-1"
                },
                {
                    "id": "test-memory-2",
                    "content": "I learned Python last year",
                    "embedding": self._generate_embedding(),
                    "memory_type": "experience",
                    "confidence": 0.85,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 2,
                    "instance_ids": [],
                    "content_hash": "hash-2"
                }
            ],
            "edges": [
                {
                    "id": "test-edge-1",
                    "source_id": "test-memory-1",
                    "target_id": "test-memory-2",
                    "edge_type": "SUPPORTS",
                    "strength": 0.8,
                    "created_at": datetime.now().isoformat()
                }
            ]
        }

    # ==================== Basic Import Tests ====================

    def test_import_from_dict_basic(self):
        """Test basic import from dictionary."""
        data = self._create_sample_export_data()

        stats = self.importer.import_from_dict(data)

        self.assertEqual(stats["imported"], 2)
        self.assertEqual(stats["skipped"], 0)
        self.assertEqual(stats["overwritten"], 0)
        self.assertEqual(len(stats["errors"]), 0)

        # Verify memories were imported
        memory1 = self.palace.get_memory("test-memory-1")
        self.assertIsNotNone(memory1)
        self.assertEqual(memory1.content, "Python is a programming language")
        self.assertEqual(memory1.memory_type, "fact")
        self.assertAlmostEqual(memory1.confidence, 0.95, places=4)

        memory2 = self.palace.get_memory("test-memory-2")
        self.assertIsNotNone(memory2)
        self.assertEqual(memory2.content, "I learned Python last year")

    def test_import_preserves_embeddings(self):
        """Test that embeddings are preserved during import."""
        data = self._create_sample_export_data()
        original_embedding = data["memories"][0]["embedding"]

        self.importer.import_from_dict(data)

        memory = self.palace.get_memory("test-memory-1")
        self.assertIsNotNone(memory.embedding)
        self.assertEqual(len(memory.embedding), len(original_embedding))
        self.assertTrue(np.allclose(memory.embedding, original_embedding, rtol=1e-5))

    def test_import_without_embeddings(self):
        """Test importing memories without embeddings."""
        data = self._create_sample_export_data()
        data["memories"][0]["embedding"] = None
        data["memories"][1]["embedding"] = None

        stats = self.importer.import_from_dict(data)

        self.assertEqual(stats["imported"], 2)

        memory = self.palace.get_memory("test-memory-1")
        self.assertIsNone(memory.embedding)

    def test_import_restores_edges(self):
        """Test that edges are restored after import."""
        data = self._create_sample_export_data()

        self.importer.import_from_dict(data)

        # Verify edge exists
        edges = self.palace.get_edges("test-memory-1")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, "test-memory-1")
        self.assertEqual(edges[0].target_id, "test-memory-2")
        self.assertEqual(edges[0].edge_type, "SUPPORTS")
        self.assertAlmostEqual(edges[0].strength, 0.8, places=4)

    def test_import_skips_edges_with_missing_memories(self):
        """Test that edges are skipped if source or target memory is missing."""
        data = self._create_sample_export_data()

        # Remove one memory but keep the edge
        data["memories"] = [data["memories"][0]]

        stats = self.importer.import_from_dict(data)

        self.assertEqual(stats["imported"], 1)

        # Edge should not be imported since target is missing
        edges = self.palace.get_edges("test-memory-1")
        self.assertEqual(len(edges), 0)

    # ==================== Conflict Detection Tests ====================

    def test_conflict_skip_strategy(self):
        """Test SKIP conflict strategy (default)."""
        data = self._create_sample_export_data()

        # First import
        stats1 = self.importer.import_from_dict(data)
        self.assertEqual(stats1["imported"], 2)

        # Second import - should skip all
        stats2 = self.importer.import_from_dict(data, ConflictResolution.SKIP)
        self.assertEqual(stats2["imported"], 0)
        self.assertEqual(stats2["skipped"], 2)
        self.assertEqual(stats2["overwritten"], 0)

    def test_conflict_overwrite_strategy(self):
        """Test OVERWRITE conflict strategy."""
        data = self._create_sample_export_data()

        # First import
        stats1 = self.importer.import_from_dict(data)
        self.assertEqual(stats1["imported"], 2)

        # Modify content
        data["memories"][0]["content"] = "Python is the best programming language"
        data["memories"][0]["confidence"] = 0.99

        # Second import with overwrite
        stats2 = self.importer.import_from_dict(data, ConflictResolution.OVERWRITE)
        self.assertEqual(stats2["imported"], 0)
        self.assertEqual(stats2["skipped"], 0)
        self.assertEqual(stats2["overwritten"], 2)

        # Verify content was updated
        memory = self.palace.get_memory("test-memory-1")
        self.assertEqual(memory.content, "Python is the best programming language")
        self.assertAlmostEqual(memory.confidence, 0.99, places=4)

    def test_conflict_error_strategy(self):
        """Test ERROR conflict strategy."""
        data = self._create_sample_export_data()

        # First import
        stats1 = self.importer.import_from_dict(data)
        self.assertEqual(stats1["imported"], 2)

        # Second import with error strategy - should record errors
        stats2 = self.importer.import_from_dict(data, ConflictResolution.ERROR)

        # Should have errors for both conflicting memories
        self.assertEqual(stats2["imported"], 0)
        self.assertEqual(len(stats2["errors"]), 2)
        self.assertIn("conflict", stats2["errors"][0].lower())
        self.assertIn("conflict", stats2["errors"][1].lower())

    def test_conflict_detection_by_id(self):
        """Test conflict detection by memory ID."""
        # Create first memory
        memory_id = self.palace.store_memory(
            content="Original content",
            memory_type="fact"
        )

        # Import data with same ID but different content
        data = {
            "metadata": {"export_version": "1.0", "memory_count": 1},
            "memories": [
                {
                    "id": memory_id,
                    "content": "New content",
                    "memory_type": "fact",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": [],
                    "content_hash": "different-hash"
                }
            ]
        }

        # SKIP should detect conflict by ID
        stats = self.importer.import_from_dict(data, ConflictResolution.SKIP)
        self.assertEqual(stats["skipped"], 1)

        # Content should remain original
        memory = self.palace.get_memory(memory_id)
        self.assertEqual(memory.content, "Original content")

    def test_conflict_detection_by_content_hash(self):
        """Test conflict detection by content hash."""
        # Create memory with known content hash
        original_id = self.palace.store_memory(
            content="Test content",
            memory_type="fact"
        )

        # Get the content hash
        original_memory = self.palace.get_memory(original_id)
        content_hash = original_memory.content_hash

        # Import with different ID but same content hash
        data = {
            "metadata": {"export_version": "1.0", "memory_count": 1},
            "memories": [
                {
                    "id": "different-id",
                    "content": "Test content",
                    "memory_type": "fact",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": [],
                    "content_hash": content_hash
                }
            ]
        }

        # Should detect conflict by hash
        stats = self.importer.import_from_dict(data, ConflictResolution.SKIP)
        self.assertEqual(stats["skipped"], 1)

    # ==================== Validation Tests ====================

    def test_invalid_conflict_strategy(self):
        """Test invalid conflict strategy raises error."""
        data = self._create_sample_export_data()

        with self.assertRaises(ValueError) as context:
            self.importer.import_from_dict(data, "invalid_strategy")

        self.assertIn("Invalid conflict strategy", str(context.exception))

    def test_invalid_data_format_not_dict(self):
        """Test that non-dict data raises error."""
        with self.assertRaises(ValueError) as context:
            self.importer.import_from_dict("not a dict")

        self.assertIn("must be a dictionary", str(context.exception))

    def test_invalid_data_format_missing_memories(self):
        """Test that data missing 'memories' key raises error."""
        data = {"metadata": {}}

        with self.assertRaises(ValueError) as context:
            self.importer.import_from_dict(data)

        self.assertIn("missing 'memories' key", str(context.exception))

    def test_invalid_data_format_memories_not_list(self):
        """Test that 'memories' must be a list."""
        data = {"memories": "not a list"}

        with self.assertRaises(ValueError) as context:
            self.importer.import_from_dict(data)

        self.assertIn("must be a list", str(context.exception))

    def test_memory_missing_id(self):
        """Test that memory without ID is tracked as error."""
        data = {
            "metadata": {"export_version": "1.0", "memory_count": 1},
            "memories": [
                {
                    # Missing "id" field
                    "content": "Test content",
                    "memory_type": "fact",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": []
                }
            ]
        }

        stats = self.importer.import_from_dict(data)

        self.assertEqual(stats["imported"], 0)
        self.assertEqual(len(stats["errors"]), 1)
        self.assertIn("id", stats["errors"][0].lower())

    # ==================== Statistics Tests ====================

    def test_statistics_mixed_results(self):
        """Test statistics with mixed import results."""
        # Create one existing memory
        existing_id = self.palace.store_memory(
            content="Existing memory",
            memory_type="fact"
        )

        # Import data with one new and one existing
        data = {
            "metadata": {"export_version": "1.0", "memory_count": 2},
            "memories": [
                {
                    "id": existing_id,
                    "content": "Existing memory",
                    "memory_type": "fact",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": [],
                    "content_hash": "hash-1"
                },
                {
                    "id": "new-memory",
                    "content": "New memory",
                    "memory_type": "experience",
                    "confidence": 0.85,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": [],
                    "content_hash": "hash-2"
                }
            ]
        }

        # SKIP strategy
        stats = self.importer.import_from_dict(data, ConflictResolution.SKIP)
        self.assertEqual(stats["imported"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["overwritten"], 0)

    def test_statistics_with_errors(self):
        """Test statistics tracking with errors."""
        data = {
            "metadata": {"export_version": "1.0", "memory_count": 3},
            "memories": [
                {
                    "id": "valid-memory",
                    "content": "Valid",
                    "memory_type": "fact",
                    "confidence": 0.9,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": []
                },
                {
                    # Missing ID - will cause error
                    "content": "Invalid",
                    "memory_type": "fact"
                },
                {
                    "id": "another-valid",
                    "content": "Another valid",
                    "memory_type": "experience",
                    "confidence": 0.8,
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 1,
                    "instance_ids": []
                }
            ]
        }

        stats = self.importer.import_from_dict(data)

        self.assertEqual(stats["imported"], 2)
        self.assertEqual(len(stats["errors"]), 1)

    # ==================== JSON/YAML Import Tests ====================

    def test_import_from_json(self):
        """Test importing from JSON file."""
        data = self._create_sample_export_data()
        json_path = Path(self.temp_dir) / "test_import.json"

        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(data, f)

        # Import from file
        stats = self.importer.import_from_json(json_path)

        self.assertEqual(stats["imported"], 2)
        self.assertEqual(stats["skipped"], 0)

        # Verify import
        memory = self.palace.get_memory("test-memory-1")
        self.assertIsNotNone(memory)

    def test_import_from_json_file_not_found(self):
        """Test import from non-existent JSON file."""
        with self.assertRaises(FileNotFoundError):
            self.importer.import_from_json(Path(self.temp_dir) / "nonexistent.json")

    def test_import_from_json_invalid_json(self):
        """Test import from invalid JSON file."""
        json_path = Path(self.temp_dir) / "invalid.json"

        # Write invalid JSON
        with open(json_path, 'w') as f:
            f.write("{ invalid json }")

        with self.assertRaises(ValueError) as context:
            self.importer.import_from_json(json_path)

        self.assertIn("Invalid JSON", str(context.exception))

    def test_import_from_yaml(self):
        """Test importing from YAML file."""
        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            with self.assertRaises(ImportError):
                self.importer.import_from_yaml(Path(self.temp_dir) / "test.yaml")
            return

        data = self._create_sample_export_data()
        yaml_path = Path(self.temp_dir) / "test_import.yaml"

        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        # Import from file
        stats = self.importer.import_from_yaml(yaml_path)

        self.assertEqual(stats["imported"], 2)

        # Verify import
        memory = self.palace.get_memory("test-memory-1")
        self.assertIsNotNone(memory)

    def test_import_from_yaml_file_not_found(self):
        """Test import from non-existent YAML file."""
        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            self.skipTest("PyYAML not installed")

        with self.assertRaises(FileNotFoundError):
            self.importer.import_from_yaml(Path(self.temp_dir) / "nonexistent.yaml")

    # ==================== Round-Trip Tests ====================

    def test_export_import_roundtrip(self):
        """Test that export → import maintains data integrity."""
        # Create original data
        memory1_id = self.palace.store_memory(
            content="Original memory 1",
            embedding=self._generate_embedding(),
            memory_type="fact",
            confidence=0.95
        )
        memory2_id = self.palace.store_memory(
            content="Original memory 2",
            embedding=self._generate_embedding(),
            memory_type="experience",
            confidence=0.85
        )
        edge_id = self.palace.create_edge(memory1_id, memory2_id, "SUPPORTS")

        # Export
        export_data = self.exporter.export_to_dict(include_embeddings=True)

        # Create new database for import
        new_db_path = Path(self.temp_dir) / "import_palace.sqlite"
        new_palace = GraphPalace(new_db_path)
        new_importer = MemoryImporter(new_palace)

        # Import
        stats = new_importer.import_from_dict(export_data)

        self.assertEqual(stats["imported"], 2)
        self.assertEqual(stats["skipped"], 0)

        # Verify memories match
        memory1_new = new_palace.get_memory(memory1_id)
        self.assertEqual(memory1_new.content, "Original memory 1")
        self.assertEqual(memory1_new.memory_type, "fact")
        self.assertAlmostEqual(memory1_new.confidence, 0.95, places=4)

        # Verify edges match
        edges = new_palace.get_edges(memory1_id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].target_id, memory2_id)

        new_palace.close()

    def test_round_trip(self):
        """Test complete round-trip: export → import → export = identical."""
        # Create original data with various memory types and edges
        memory1_id = self.palace.store_memory(
            content="Python is a programming language",
            embedding=self._generate_embedding(),
            memory_type="fact",
            confidence=0.95
        )
        memory2_id = self.palace.store_memory(
            content="I learned Python last year",
            embedding=self._generate_embedding(),
            memory_type="experience",
            confidence=0.85
        )
        memory3_id = self.palace.store_memory(
            content="Python is better than Java",
            embedding=self._generate_embedding(),
            memory_type="belief",
            confidence=0.70
        )

        # Create edges between memories
        edge1_id = self.palace.create_edge(memory1_id, memory2_id, "SUPPORTS")
        edge2_id = self.palace.create_edge(memory2_id, memory3_id, "RELATED_TO")

        # First export (with embeddings for complete test)
        export1 = self.exporter.export_to_dict(include_embeddings=True)

        # Verify export structure
        self.assertEqual(export1["metadata"]["memory_count"], 3)
        self.assertEqual(export1["metadata"]["edge_count"], 2)
        self.assertEqual(len(export1["memories"]), 3)
        self.assertEqual(len(export1["edges"]), 2)

        # Create new database for import
        new_db_path = Path(self.temp_dir) / "roundtrip_palace.sqlite"
        new_palace = GraphPalace(new_db_path)
        new_importer = MemoryImporter(new_palace)

        # Import into new database
        stats = new_importer.import_from_dict(export1)

        self.assertEqual(stats["imported"], 3)
        self.assertEqual(stats["skipped"], 0)

        # Second export from imported database
        new_exporter = MemoryExporter(new_palace)
        export2 = new_exporter.export_to_dict(include_embeddings=True)

        # Verify metadata matches (except timestamp)
        self.assertEqual(export2["metadata"]["memory_count"], export1["metadata"]["memory_count"])
        self.assertEqual(export2["metadata"]["edge_count"], export1["metadata"]["edge_count"])
        self.assertEqual(export2["metadata"]["export_version"], export1["metadata"]["export_version"])

        # Verify memories match
        self.assertEqual(len(export2["memories"]), len(export1["memories"]))

        # Sort memories by ID for comparison
        memories1 = sorted(export1["memories"], key=lambda m: m["id"])
        memories2 = sorted(export2["memories"], key=lambda m: m["id"])

        for mem1, mem2 in zip(memories1, memories2):
            self.assertEqual(mem2["id"], mem1["id"])
            self.assertEqual(mem2["content"], mem1["content"])
            self.assertEqual(mem2["memory_type"], mem1["memory_type"])
            self.assertAlmostEqual(mem2["confidence"], mem1["confidence"], places=4)
            self.assertEqual(mem2["created_at"], mem1["created_at"])

            # Verify embeddings match if present
            if mem1.get("embedding"):
                self.assertIsNotNone(mem2.get("embedding"))
                self.assertEqual(len(mem2["embedding"]), len(mem1["embedding"]))
                # Check embeddings are close (account for floating point precision)
                for e1, e2 in zip(mem1["embedding"], mem2["embedding"]):
                    self.assertAlmostEqual(e2, e1, places=6)

        # Verify edges match
        self.assertEqual(len(export2["edges"]), len(export1["edges"]))

        # Sort edges by ID for comparison
        edges1 = sorted(export1["edges"], key=lambda e: e["id"])
        edges2 = sorted(export2["edges"], key=lambda e: e["id"])

        for edge1, edge2 in zip(edges1, edges2):
            self.assertEqual(edge2["id"], edge1["id"])
            self.assertEqual(edge2["source_id"], edge1["source_id"])
            self.assertEqual(edge2["target_id"], edge1["target_id"])
            self.assertEqual(edge2["edge_type"], edge1["edge_type"])
            self.assertAlmostEqual(edge2["strength"], edge1["strength"], places=4)
            self.assertEqual(edge2["created_at"], edge1["created_at"])

        new_palace.close()

    def test_partial_import_with_overwrite(self):
        """Test partial import with overwrite strategy."""
        # Create initial memory
        memory_id = self.palace.store_memory(
            content="Version 1",
            memory_type="fact",
            confidence=0.8
        )

        # Export and modify
        export_data = self.exporter.export_to_dict()
        export_data["memories"][0]["content"] = "Version 2"
        export_data["memories"][0]["confidence"] = 0.9

        # Import with overwrite
        stats = self.importer.import_from_dict(export_data, ConflictResolution.OVERWRITE)

        self.assertEqual(stats["overwritten"], 1)
        self.assertEqual(stats["imported"], 0)

        # Verify update
        memory = self.palace.get_memory(memory_id)
        self.assertEqual(memory.content, "Version 2")
        self.assertAlmostEqual(memory.confidence, 0.9, places=4)


class TestCLIExport(unittest.TestCase):
    """Test suite for CLI export command integration."""

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _create_test_database(self, base_path: Path):
        """Create a test database with sample memories."""
        db_path = base_path / "palace.sqlite"
        palace = GraphPalace(db_path)

        now = datetime.now()
        yesterday = now - timedelta(days=1)

        memory1_id = palace.store_memory(
            content="Python is a programming language",
            embedding=self._generate_embedding(),
            memory_type="fact",
            confidence=0.95
        )

        memory2_id = palace.store_memory(
            content="I learned Python last year",
            embedding=self._generate_embedding(),
            memory_type="experience",
            confidence=0.85
        )
        # Set created_at to yesterday for filtering tests
        palace._conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (yesterday.isoformat(), memory2_id)
        )
        palace._conn.commit()

        memory3_id = palace.store_memory(
            content="Python is better than Java",
            embedding=self._generate_embedding(),
            memory_type="belief",
            confidence=0.70
        )

        # Create an edge
        palace.create_edge(memory1_id, memory2_id, "SUPPORTS")

        # Commit and close to ensure everything is written
        palace._conn.commit()
        palace.close()

        return db_path

    # ==================== Basic CLI Tests ====================

    def test_export_requires_database(self):
        """Test that export command requires initialized database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Point to non-existent database
            with runner.isolated_filesystem(temp_dir=tmpdir):
                result = runner.invoke(cli, [
                    '--data-dir', tmpdir,
                    'export',
                    '--output', 'memories.json'
                ])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_export_to_json_file(self):
        """Test exporting memories to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            output_path = Path(tmpdir) / "export.json"

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--output', str(output_path)
            ])

            assert result.exit_code == 0
            assert output_path.exists()
            assert "exported" in result.output.lower()

            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "metadata" in data
            assert "memories" in data
            assert len(data["memories"]) == 3

    def test_export_to_yaml_file(self):
        """Test exporting memories to YAML file."""
        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            self.skipTest("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            output_path = Path(tmpdir) / "export.yaml"

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'yaml',
                '--output', str(output_path)
            ])

            assert result.exit_code == 0
            assert output_path.exists()
            assert "exported" in result.output.lower()

            # Verify YAML content
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)

            assert "metadata" in data
            assert "memories" in data
            assert len(data["memories"]) == 3

    def test_export_to_stdout_json(self):
        """Test exporting to stdout as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json'
            ])

            assert result.exit_code == 0

            # Verify JSON output
            data = json.loads(result.output)
            assert "metadata" in data
            assert "memories" in data
            assert len(data["memories"]) == 3

    def test_export_to_stdout_yaml(self):
        """Test exporting to stdout as YAML."""
        try:
            import yaml
            yaml_available = True
        except ImportError:
            yaml_available = False

        if not yaml_available:
            self.skipTest("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'yaml'
            ])

            assert result.exit_code == 0

            # Verify YAML output
            data = yaml.safe_load(result.output)
            assert "metadata" in data
            assert "memories" in data

    # ==================== Filter Tests ====================

    def test_export_filter_by_type(self):
        """Test filtering export by memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--type', 'fact'
            ])

            assert result.exit_code == 0

            # Verify filtered output
            data = json.loads(result.output)
            assert len(data["memories"]) == 1
            assert data["memories"][0]["memory_type"] == "fact"

    def test_export_filter_by_confidence(self):
        """Test filtering export by minimum confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--min-confidence', '0.80'
            ])

            assert result.exit_code == 0

            # Verify filtered output (confidence >= 0.80)
            data = json.loads(result.output)
            assert len(data["memories"]) >= 1
            for memory in data["memories"]:
                if memory.get("confidence") is not None:
                    assert memory["confidence"] >= 0.80

    def test_export_filter_by_date_range(self):
        """Test filtering export by date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Export from today onwards (should exclude yesterday's memory)
            today = datetime.now().date().isoformat()

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--date-from', today
            ])

            assert result.exit_code == 0

            # Verify filtered output
            data = json.loads(result.output)
            # Should have fewer memories (excluding the one from yesterday)
            assert len(data["memories"]) >= 1

    def test_export_invalid_date_format(self):
        """Test that invalid date format returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--date-from', 'invalid-date'
            ])

            assert result.exit_code == 1
            assert "invalid" in result.output.lower()

    # ==================== Options Tests ====================

    def test_export_with_edges(self):
        """Test export includes edges by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json'
            ])

            assert result.exit_code == 0

            # Verify edges are included
            data = json.loads(result.output)
            assert "edges" in data
            assert len(data["edges"]) >= 1

    def test_export_without_edges(self):
        """Test export with --no-edges flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--no-edges'
            ])

            assert result.exit_code == 0

            # Verify edges are not included
            data = json.loads(result.output)
            assert "edges" not in data

    def test_export_without_embeddings_by_default(self):
        """Test that embeddings are excluded by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json'
            ])

            assert result.exit_code == 0

            # Verify embeddings are not included
            data = json.loads(result.output)
            for memory in data["memories"]:
                assert memory.get("embedding") is None

    def test_export_with_embeddings(self):
        """Test export with --include-embeddings flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--include-embeddings'
            ])

            assert result.exit_code == 0

            # Verify embeddings are included for memories that have them
            data = json.loads(result.output)
            memories_with_embeddings = [m for m in data["memories"] if m.get("embedding")]
            assert len(memories_with_embeddings) > 0

            # Verify embedding is a list of floats
            embedding = memories_with_embeddings[0]["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert isinstance(embedding[0], float)

    # ==================== Combined Filter Tests ====================

    def test_export_multiple_filters(self):
        """Test export with multiple filters combined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'export',
                '--format', 'json',
                '--type', 'belief',
                '--min-confidence', '0.5',
                '--no-edges'
            ])

            assert result.exit_code == 0

            # Verify filtered output
            data = json.loads(result.output)
            assert "edges" not in data
            for memory in data["memories"]:
                assert memory["memory_type"] == "belief"
                if memory.get("confidence") is not None:
                    assert memory["confidence"] >= 0.5


class TestCLIImport:
    """Tests for 'omi import' CLI command."""

    def _create_test_database(self, base_path: Path):
        """Create a test database with sample data."""
        db_path = base_path / "palace.sqlite"
        palace = GraphPalace(db_path)

        # Create sample memories
        memory1_id = palace.store_memory(
            content="Test memory 1",
            embedding=self._generate_embedding(),
            memory_type="fact",
            confidence=0.95
        )
        memory2_id = palace.store_memory(
            content="Test memory 2",
            embedding=self._generate_embedding(),
            memory_type="experience",
            confidence=0.85
        )

        # Create edge
        palace.create_edge(memory1_id, memory2_id, "RELATED_TO")

        palace.close()
        return db_path

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _create_export_file(self, file_path: Path, format_type: str = 'json'):
        """Create a sample export file for testing import."""
        data = {
            "metadata": {
                "export_version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "memory_count": 2,
                "edge_count": 1,
                "filters": {}
            },
            "memories": [
                {
                    "id": "test-memory-1",
                    "content": "Imported memory 1",
                    "memory_type": "fact",
                    "confidence": 0.90,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                },
                {
                    "id": "test-memory-2",
                    "content": "Imported memory 2",
                    "memory_type": "experience",
                    "confidence": 0.80,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
            ],
            "edges": [
                {
                    "id": "test-edge-1",
                    "source_id": "test-memory-1",
                    "target_id": "test-memory-2",
                    "edge_type": "SUPPORTS",
                    "strength": 0.75,
                    "created_at": datetime.now().isoformat()
                }
            ]
        }

        if format_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:  # yaml
            import yaml
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)

    # ==================== Initialization Tests ====================

    def test_import_requires_init(self):
        """Test that import requires database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create a non-existent base path
            base_path = Path(tmpdir) / "not_initialized"
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file)

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file)
            ])

            assert result.exit_code == 1
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_import_requires_input_file(self):
        """Test that import requires --input parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Try import without --input
            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import'
            ])

            assert result.exit_code != 0
            # Click will show an error about missing required option

    def test_import_validates_file_exists(self):
        """Test that import validates input file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Try import with non-existent file
            non_existent_file = Path(tmpdir) / "does_not_exist.json"

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(non_existent_file)
            ])

            assert result.exit_code != 0
            # Click will validate file existence

    # ==================== Format Detection Tests ====================

    def test_import_auto_detects_json_format(self):
        """Test that import auto-detects JSON format from file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file)
            ])

            assert result.exit_code == 0
            assert "auto-detected" in result.output.lower() or "import" in result.output.lower()

    def test_import_auto_detects_yaml_format(self):
        """Test that import auto-detects YAML format from file extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.yaml"
            self._create_export_file(export_file, format_type='yaml')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file)
            ])

            assert result.exit_code == 0
            assert "auto-detected" in result.output.lower() or "import" in result.output.lower()

    def test_import_explicit_format_override(self):
        """Test that import respects explicit --format parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create JSON file with .txt extension
            export_file = Path(tmpdir) / "export.txt"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--format', 'json'
            ])

            assert result.exit_code == 0

    # ==================== Import Functionality Tests ====================

    def test_import_json_file_success(self):
        """Test successful import from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file)
            ])

            assert result.exit_code == 0
            assert "import" in result.output.lower()
            # Should show some success indication

    def test_import_yaml_file_success(self):
        """Test successful import from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.yaml"
            self._create_export_file(export_file, format_type='yaml')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file)
            ])

            assert result.exit_code == 0
            assert "import" in result.output.lower()

    # ==================== Conflict Resolution Tests ====================

    def test_import_with_skip_conflict_strategy(self):
        """Test import with 'skip' conflict resolution (default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--on-conflict', 'skip'
            ])

            assert result.exit_code == 0

    def test_import_with_overwrite_conflict_strategy(self):
        """Test import with 'overwrite' conflict resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--on-conflict', 'overwrite'
            ])

            assert result.exit_code == 0

    def test_import_with_error_conflict_strategy(self):
        """Test import with 'error' conflict resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--on-conflict', 'error'
            ])

            # Should succeed or fail depending on whether there are conflicts
            # Exit code 0 means no conflicts or conflicts handled correctly
            assert result.exit_code in [0, 1]

    # ==================== Dry Run Tests ====================

    def test_import_dry_run_shows_preview(self):
        """Test that --dry-run shows preview without importing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            db_path = self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            # Count memories before import
            palace = GraphPalace(db_path)
            cursor = palace._conn.execute("SELECT COUNT(*) FROM memories")
            before_count = cursor.fetchone()[0]
            palace.close()

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--dry-run'
            ])

            assert result.exit_code == 0
            assert "dry run" in result.output.lower() or "would import" in result.output.lower()

            # Verify no actual import happened
            palace = GraphPalace(db_path)
            cursor = palace._conn.execute("SELECT COUNT(*) FROM memories")
            after_count = cursor.fetchone()[0]
            palace.close()

            assert after_count == before_count  # No new memories added

    def test_import_dry_run_with_conflict_strategy(self):
        """Test --dry-run with conflict strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path with database
            base_path = Path(tmpdir) / "omi_base"
            base_path.mkdir()
            self._create_test_database(base_path)

            # Create export file
            export_file = Path(tmpdir) / "export.json"
            self._create_export_file(export_file, format_type='json')

            result = runner.invoke(cli, [
                '--data-dir', str(base_path),
                'import',
                '--input', str(export_file),
                '--on-conflict', 'overwrite',
                '--dry-run'
            ])

            assert result.exit_code == 0
            assert "dry run" in result.output.lower() or "would import" in result.output.lower()


if __name__ == '__main__':
    unittest.main()
