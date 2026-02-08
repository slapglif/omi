"""Pytest fixtures for OMI MCP integration tests"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime


@pytest.fixture(scope="session")
def omi_db():
    """Single test database for all tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_omi_setup(tmp_path):
    """Create temporary OMI instance for testing.
    
    Returns a dict with:
        - base_path: Path to temp directory
        - db_path: Path to SQLite database
        - now_path: Path to NOW.md
    """
    base_path = tmp_path / "omi"
    base_path.mkdir(parents=True, exist_ok=True)
    
    db_path = base_path / "palace.sqlite"
    now_path = base_path / "NOW.md"
    memory_path = base_path / "MEMORY.md"
    
    # Create required directories
    (base_path / "memory").mkdir(exist_ok=True)
    (base_path / "embeddings").mkdir(exist_ok=True)
    
    return {
        "base_path": base_path,
        "db_path": db_path,
        "now_path": now_path,
        "memory_path": memory_path,
    }


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns consistent embeddings for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    # Return a consistent 768-dim embedding
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.return_value = [[0.1] * 768]
    
    def mock_similarity(e1, e2):
        # Simple cosine similarity - identical vectors = 1.0
        if e1 == e2:
            return 1.0
        return 0.85  # Decent similarity for testing
    
    mock.similarity = mock_similarity
    return mock


@pytest.fixture
def mock_embedding_cache(tmp_path, mock_embedder):
    """Create a mock embedding cache."""
    from omi.embeddings import EmbeddingCache
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EmbeddingCache(cache_dir, mock_embedder)


@pytest.fixture
def persistence_stores(temp_omi_setup):
    """Create persistence stores for testing."""
    from omi.persistence import NOWStore, DailyLogStore, GraphPalace
    
    now_store = NOWStore(temp_omi_setup["base_path"])
    daily_store = DailyLogStore(temp_omi_setup["base_path"])
    palace = GraphPalace(temp_omi_setup["db_path"])
    
    return {
        "now_store": now_store,
        "daily_store": daily_store,
        "palace": palace,
    }


@pytest.fixture
def mock_vault():
    """Mock vault backup."""
    from unittest.mock import MagicMock
    
    vault = MagicMock()
    vault.backup.return_value = "backup_id_12345"
    vault.restore.return_value = "restored_memory_content"
    return vault


@pytest.fixture
def belief_network_setup(temp_omi_setup, mock_embedder, mock_embedding_cache):
    """Setup for belief network tests."""
    from omi.persistence import GraphPalace
    from omi.belief import BeliefNetwork, ContradictionDetector
    
    palace = GraphPalace(temp_omi_setup["db_path"])
    belief_network = BeliefNetwork(palace)
    detector = ContradictionDetector()
    
    return {
        "palace": palace,
        "belief_network": belief_network,
        "detector": detector,
    }


@pytest.fixture
def security_setup(temp_omi_setup, persistence_stores):
    """Setup for security tests."""
    from omi.security import IntegrityChecker, TopologyVerifier
    
    integrity = IntegrityChecker(temp_omi_setup["base_path"])
    topology = TopologyVerifier(persistence_stores["palace"])
    
    return {
        "integrity": integrity,
        "topology": topology,
        "base_path": temp_omi_setup["base_path"],
    }


@pytest.fixture
def sample_memories():
    """Sample memory content for testing."""
    return {
        "python_1": "Learned Python debugging using pdb and logging",
        "python_2": "Discovered Python decorators are powerful for code reuse",
        "python_3": "Found that Python list comprehensions are faster than loops",
        "js_1": "JavaScript promises simplify async code handling",
        "js_2": "JavaScript closures are confusing but useful",
        "sqlalchemy": "Learned that SQLAlchemy is slow when not using proper indexing",
    }


@pytest.fixture
def sample_beliefs():
    """Sample beliefs for testing."""
    return {
        "high_confidence": "Python is the best language for data science",
        "low_confidence": "Maybe JavaScript will replace Python someday",
    }
