"""
Memory Export/Import - Portable formats for OMI memories
Support for JSON and YAML export/import with filtering capabilities.

Enables:
- Inspection and debugging of memory stores
- Sharing curated memory sets between agents
- Version control of key memories
- Migration between OMI instances
"""

import json
import struct
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .storage.graph_palace import GraphPalace, Memory, Edge


def serialize_embedding(embedding: List[float]) -> bytes:
    """
    Convert embedding list to binary blob (float32).

    Args:
        embedding: List of float values representing the embedding vector

    Returns:
        Binary blob containing float32 representation of embedding
    """
    if not embedding:
        return b''
    return struct.pack(f'{len(embedding)}f', *embedding)


def deserialize_embedding(blob: bytes) -> List[float]:
    """
    Convert binary blob to embedding list (float32).

    Args:
        blob: Binary blob containing float32 embedding data

    Returns:
        List of float values representing the embedding vector
    """
    if not blob:
        return []
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


class MemoryExporter:
    """
    Export memories to portable formats (JSON, YAML).

    Features:
    - Filter by memory type, confidence, date range
    - Include or exclude embeddings
    - Include or exclude edges
    - Human-readable output
    """

    def __init__(self, palace: GraphPalace):
        """
        Initialize exporter with GraphPalace instance.

        Args:
            palace: GraphPalace storage instance
        """
        self.palace = palace

    def export_to_dict(self,
                      memory_type: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      include_embeddings: bool = False,
                      include_edges: bool = True) -> Dict[str, Any]:
        """
        Export memories to dictionary with optional filters.

        Args:
            memory_type: Filter by type (fact|experience|belief|decision)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            date_from: Start date for filtering (inclusive)
            date_to: End date for filtering (inclusive)
            include_embeddings: Include embedding vectors in export
            include_edges: Include relationship edges in export

        Returns:
            Dictionary with metadata and filtered memories
        """
        # Build SQL query with filters
        query = """
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories
            WHERE 1=1
        """
        params = []

        # Apply filters
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if date_from:
            query += " AND created_at >= ?"
            params.append(date_from.isoformat())

        if date_to:
            query += " AND created_at <= ?"
            params.append(date_to.isoformat())

        # Order by created_at for consistent output
        query += " ORDER BY created_at ASC"

        # Execute query
        cursor = self.palace._conn.execute(query, params)

        # Convert rows to Memory objects and serialize
        memories = []
        memory_ids = set()

        for row in cursor:
            memory_id = row[0]
            memory_ids.add(memory_id)

            # Parse embedding if needed
            embedding = None
            if include_embeddings and row[2]:
                embedding = self.palace._blob_to_embed(row[2])

            # Create Memory object
            memory = Memory(
                id=memory_id,
                content=row[1],
                embedding=embedding,
                memory_type=row[3],
                confidence=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
                access_count=row[7],
                instance_ids=json.loads(row[8]) if row[8] else [],
                content_hash=row[9]
            )

            memories.append(memory.to_dict())

        # Build result dict
        result = {
            "metadata": {
                "export_version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "memory_count": len(memories),
                "filters": {
                    "memory_type": memory_type,
                    "min_confidence": min_confidence,
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None,
                    "include_embeddings": include_embeddings
                }
            },
            "memories": memories
        }

        # Add edges if requested
        if include_edges and memory_ids:
            edges = self._export_edges(memory_ids)
            result["edges"] = edges
            result["metadata"]["edge_count"] = len(edges)

        return result

    def _export_edges(self, memory_ids: set) -> List[Dict[str, Any]]:
        """
        Export edges for given memory IDs.

        Args:
            memory_ids: Set of memory IDs to export edges for

        Returns:
            List of edge dictionaries
        """
        # Build query for edges connecting exported memories
        placeholders = ','.join('?' * len(memory_ids))
        query = f"""
            SELECT id, source_id, target_id, edge_type, strength, created_at
            FROM edges
            WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
            ORDER BY created_at ASC
        """

        params = list(memory_ids) + list(memory_ids)
        cursor = self.palace._conn.execute(query, params)

        edges = []
        for row in cursor:
            edge = {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "edge_type": row[3],
                "strength": row[4],
                "created_at": row[5]
            }
            edges.append(edge)

        return edges

    def export_to_json(self,
                      output_path: Path,
                      memory_type: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      include_embeddings: bool = False,
                      include_edges: bool = True,
                      indent: int = 2) -> int:
        """
        Export memories to JSON file.

        Args:
            output_path: Path to output JSON file
            memory_type: Filter by type (fact|experience|belief|decision)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            date_from: Start date for filtering (inclusive)
            date_to: End date for filtering (inclusive)
            include_embeddings: Include embedding vectors in export
            include_edges: Include relationship edges in export
            indent: JSON indentation for readability (default: 2)

        Returns:
            Number of memories exported
        """
        data = self.export_to_dict(
            memory_type=memory_type,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            include_embeddings=include_embeddings,
            include_edges=include_edges
        )

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return data["metadata"]["memory_count"]

    def export_to_yaml(self,
                      output_path: Path,
                      memory_type: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      include_embeddings: bool = False,
                      include_edges: bool = True) -> int:
        """
        Export memories to YAML file.

        Args:
            output_path: Path to output YAML file
            memory_type: Filter by type (fact|experience|belief|decision)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            date_from: Start date for filtering (inclusive)
            date_to: End date for filtering (inclusive)
            include_embeddings: Include embedding vectors in export
            include_edges: Include relationship edges in export

        Returns:
            Number of memories exported

        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML export. Install with: pip install pyyaml")

        data = self.export_to_dict(
            memory_type=memory_type,
            min_confidence=min_confidence,
            date_from=date_from,
            date_to=date_to,
            include_embeddings=include_embeddings,
            include_edges=include_edges
        )

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return data["metadata"]["memory_count"]


class ConflictResolution:
    """Conflict resolution strategies for import."""
    SKIP = "skip"          # Skip conflicting memories
    OVERWRITE = "overwrite"  # Replace existing memories
    ERROR = "error"        # Raise error on conflict


class MemoryImporter:
    """
    Import memories from portable formats (JSON, YAML).

    Features:
    - Detect conflicts by ID and content hash
    - Multiple conflict resolution strategies
    - Preserve relationships (edges)
    - Validation of import data
    """

    def __init__(self, palace: GraphPalace):
        """
        Initialize importer with GraphPalace instance.

        Args:
            palace: GraphPalace storage instance
        """
        self.palace = palace

    def import_from_dict(self,
                        data: Dict[str, Any],
                        conflict_strategy: str = ConflictResolution.SKIP) -> Dict[str, Any]:
        """
        Import memories from dictionary with conflict detection.

        Args:
            data: Dictionary containing memories and metadata (from export)
            conflict_strategy: How to handle conflicts ('skip'|'overwrite'|'error')

        Returns:
            Dictionary with import statistics:
            {
                "imported": int,
                "skipped": int,
                "overwritten": int,
                "errors": List[str]
            }

        Raises:
            ValueError: If conflict_strategy is 'error' and conflicts are detected
            ValueError: If data format is invalid
        """
        # Validate conflict strategy
        valid_strategies = {ConflictResolution.SKIP, ConflictResolution.OVERWRITE, ConflictResolution.ERROR}
        if conflict_strategy not in valid_strategies:
            raise ValueError(f"Invalid conflict strategy: {conflict_strategy}. Must be one of {valid_strategies}")

        # Validate data format
        if not isinstance(data, dict):
            raise ValueError("Import data must be a dictionary")
        if "memories" not in data:
            raise ValueError("Import data missing 'memories' key")
        if not isinstance(data["memories"], list):
            raise ValueError("Import data 'memories' must be a list")

        # Initialize stats
        stats = {
            "imported": 0,
            "skipped": 0,
            "overwritten": 0,
            "errors": []
        }

        # Track memory IDs for edge restoration
        imported_ids = set()

        # Import memories
        for mem_data in data["memories"]:
            try:
                result = self._import_memory(mem_data, conflict_strategy)

                if result == "imported":
                    stats["imported"] += 1
                    imported_ids.add(mem_data["id"])
                elif result == "skipped":
                    stats["skipped"] += 1
                elif result == "overwritten":
                    stats["overwritten"] += 1
                    imported_ids.add(mem_data["id"])

            except Exception as e:
                error_msg = f"Error importing memory {mem_data.get('id', 'unknown')}: {str(e)}"
                stats["errors"].append(error_msg)

        # Import edges if present
        if "edges" in data and isinstance(data["edges"], list):
            for edge_data in data["edges"]:
                try:
                    # Only import edges where both source and target were imported
                    if edge_data["source_id"] in imported_ids and edge_data["target_id"] in imported_ids:
                        self._import_edge(edge_data)
                except Exception as e:
                    error_msg = f"Error importing edge {edge_data.get('id', 'unknown')}: {str(e)}"
                    stats["errors"].append(error_msg)

        return stats

    def _import_memory(self, mem_data: Dict[str, Any], conflict_strategy: str) -> str:
        """
        Import a single memory with conflict detection.

        Args:
            mem_data: Memory dictionary
            conflict_strategy: How to handle conflicts

        Returns:
            'imported' | 'skipped' | 'overwritten'

        Raises:
            ValueError: If conflict_strategy is 'error' and conflict detected
        """
        memory_id = mem_data.get("id")
        if not memory_id:
            raise ValueError("Memory missing 'id' field")

        content_hash = mem_data.get("content_hash")

        # Check for conflicts
        existing_by_id = self._check_memory_exists_by_id(memory_id)
        existing_by_hash = None
        if content_hash:
            existing_by_hash = self._check_memory_exists_by_hash(content_hash)

        # Determine if there's a conflict
        has_conflict = existing_by_id or existing_by_hash

        if has_conflict:
            if conflict_strategy == ConflictResolution.ERROR:
                conflict_type = "ID" if existing_by_id else "content hash"
                raise ValueError(f"Memory conflict detected: {conflict_type} already exists for memory {memory_id}")

            elif conflict_strategy == ConflictResolution.SKIP:
                return "skipped"

            elif conflict_strategy == ConflictResolution.OVERWRITE:
                # Delete existing memory (cascade will handle edges)
                if existing_by_id:
                    self.palace._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                elif existing_by_hash:
                    # Delete by hash if different ID
                    self.palace._conn.execute("DELETE FROM memories WHERE content_hash = ?", (content_hash,))

        # Insert memory
        self._insert_memory(mem_data)

        return "overwritten" if has_conflict else "imported"

    def _check_memory_exists_by_id(self, memory_id: str) -> bool:
        """Check if memory with given ID exists."""
        cursor = self.palace._conn.execute(
            "SELECT 1 FROM memories WHERE id = ? LIMIT 1",
            (memory_id,)
        )
        return cursor.fetchone() is not None

    def _check_memory_exists_by_hash(self, content_hash: str) -> bool:
        """Check if memory with given content hash exists."""
        cursor = self.palace._conn.execute(
            "SELECT 1 FROM memories WHERE content_hash = ? LIMIT 1",
            (content_hash,)
        )
        return cursor.fetchone() is not None

    def _insert_memory(self, mem_data: Dict[str, Any]) -> None:
        """Insert memory into database."""
        # Convert embedding if present
        embedding_blob = None
        if mem_data.get("embedding"):
            embedding_blob = self.palace._embed_to_blob(mem_data["embedding"])

        # Parse instance_ids
        instance_ids_json = json.dumps(mem_data.get("instance_ids", []))

        # Insert into database
        self.palace._conn.execute("""
            INSERT INTO memories (
                id, content, embedding, memory_type, confidence,
                created_at, last_accessed, access_count, instance_ids, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mem_data["id"],
            mem_data["content"],
            embedding_blob,
            mem_data.get("memory_type"),
            mem_data.get("confidence"),
            mem_data.get("created_at"),
            mem_data.get("last_accessed"),
            mem_data.get("access_count", 0),
            instance_ids_json,
            mem_data.get("content_hash")
        ))

        # Update in-memory embedding cache if present
        if mem_data.get("embedding"):
            self.palace._embedding_cache[mem_data["id"]] = mem_data["embedding"]

    def _import_edge(self, edge_data: Dict[str, Any]) -> None:
        """Import a single edge into database."""
        # Check if edge already exists
        cursor = self.palace._conn.execute(
            "SELECT 1 FROM edges WHERE id = ? LIMIT 1",
            (edge_data["id"],)
        )

        if cursor.fetchone() is not None:
            # Edge already exists, skip or update based on strategy
            return

        # Insert edge
        self.palace._conn.execute("""
            INSERT INTO edges (
                id, source_id, target_id, edge_type, strength, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            edge_data["id"],
            edge_data["source_id"],
            edge_data["target_id"],
            edge_data["edge_type"],
            edge_data.get("strength"),
            edge_data.get("created_at")
        ))

    def import_from_json(self,
                        file_path: Path,
                        conflict_strategy: str = ConflictResolution.SKIP) -> Dict[str, Any]:
        """
        Import memories from JSON file.

        Args:
            file_path: Path to JSON file
            conflict_strategy: How to handle conflicts ('skip'|'overwrite'|'error')

        Returns:
            Dictionary with import statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid JSON
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")

        return self.import_from_dict(data, conflict_strategy)

    def import_from_yaml(self,
                        file_path: Path,
                        conflict_strategy: str = ConflictResolution.SKIP) -> Dict[str, Any]:
        """
        Import memories from YAML file.

        Args:
            file_path: Path to YAML file
            conflict_strategy: How to handle conflicts ('skip'|'overwrite'|'error')

        Returns:
            Dictionary with import statistics

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid YAML
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML import. Install with: pip install pyyaml")

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML file: {e}")

        return self.import_from_dict(data, conflict_strategy)
