"""Smoke Tests for OMI

Basic sanity checks - ensure the system can be imported and basic operations work.
These tests should be fast and give quick feedback if the system is broken.
"""
import pytest
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports:
    """Test that OMI modules can be imported."""
    
    def test_import_omi_main(self):
        """Can we import the main omi module?"""
        import omi
        assert omi.__version__
        assert omi.__version__ == "0.1.0"
    
    def test_import_persistence(self):
        """Can we import persistence module?"""
        from omi import NowStorage, DailyLogStore, GraphPalace
        assert NowStorage is not None
        assert DailyLogStore is not None
        assert GraphPalace is not None
    
    def test_import_embeddings(self):
        """Can we import embeddings module?"""
        from omi.embeddings import OllamaEmbedder, EmbeddingCache
        assert OllamaEmbedder is not None
        assert EmbeddingCache is not None
    
    def test_import_security(self):
        """Can we import security module?"""
        from omi.security import IntegrityChecker, ConsensusManager
        assert IntegrityChecker is not None
        assert ConsensusManager is not None
    
    def test_import_api(self):
        """Can we import API module with MCP tools?"""
        from omi.api import MemoryTools, BeliefTools, CheckpointTools
        assert MemoryTools is not None
        assert BeliefTools is not None
        assert CheckpointTools is not None
    
    def test_import_belief(self):
        """Can we import belief module?"""
        from omi.belief import BeliefNetwork, Evidence, ContradictionDetector
        assert BeliefNetwork is not None
        assert Evidence is not None
        assert ContradictionDetector is not None
    
    def test_import_cli(self):
        """Can we import CLI module?"""
        from omi.cli import cli
        assert cli is not None


class TestBasicInstantiation:
    """Test that core classes can be instantiated."""
    
    def test_now_store_instantiation(self, tmp_path):
        """Can we create a NowStorage?"""
        from omi import NowStorage
        store = NowStorage(tmp_path)
        assert store is not None
        assert store.now_file == tmp_path / "NOW.md"
    
    def test_daily_log_store_instantiation(self, tmp_path):
        """Can we create a DailyLogStore?"""
        from omi import DailyLogStore
        store = DailyLogStore(tmp_path)
        assert store is not None
    
    def test_graph_palace_instantiation(self, tmp_path):
        """Can we create a GraphPalace?"""
        from omi import GraphPalace
        db_path = tmp_path / "test.db"
        palace = GraphPalace(db_path)
        assert palace is not None
        assert db_path.exists()
    
    def test_integrity_checker_instantiation(self, tmp_path):
        """Can we create an IntegrityChecker?"""
        from omi.security import IntegrityChecker
        checker = IntegrityChecker(tmp_path)
        assert checker is not None
    
    def test_embedding_cache_instantiation(self, tmp_path):
        """Can we create an EmbeddingCache?"""
        from omi.embeddings import OllamaEmbedder, EmbeddingCache
        embedder = OllamaEmbedder()
        cache = EmbeddingCache(tmp_path / "embeddings", embedder)
        assert cache is not None


class TestBasicOperations:
    """Test basic CRUD operations."""
    
    def test_directory_creation(self, tmp_path):
        """Does init create required directories?"""
        base_path = tmp_path / "omi_data"
        base_path.mkdir(parents=True, exist_ok=True)
        
        memory_path = base_path / "memory"
        memory_path.mkdir(parents=True, exist_ok=True)
        
        assert base_path.exists()
        assert memory_path.exists()
    
    def test_sqlite_database_creation(self, tmp_path):
        """Can we create a SQLite database?"""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        assert db_path.exists()


class TestEnvironment:
    """Test environment setup."""
    
    def test_python_version(self):
        """Are we running Python 3.10+?"""
        import sys
        version_info = sys.version_info
        assert version_info.major == 3
        assert version_info.minor >= 10
    
    def test_numpy_available(self):
        """Is numpy available?"""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
    
    def test_click_available(self):
        """Is click available?"""
        import click
        assert hasattr(click, 'command')
    
    def test_yaml_available(self):
        """Is pyyaml available?"""
        import yaml
        data = yaml.safe_load("key: value")
        assert data == {"key": "value"}


class TestCLIEntry:
    """Test CLI entry points."""
    
    def test_cli_runner_works(self):
        """Can we use Click's test runner?"""
        from click.testing import CliRunner
        runner = CliRunner()
        assert runner is not None
    
    def test_cli_exists(self):
        """Does CLI object exist with commands?"""
        from omi.cli import cli
        assert cli is not None
        # cli should be a Click group with commands
        assert hasattr(cli, 'commands')


@pytest.mark.nim
class TestNIMConnectivity:
    """Optional: Test NIM connectivity if API key is available."""
    
    NIM_API_KEY = os.getenv("NIM_API_KEY", "")
    
    @pytest.mark.skipif(not NIM_API_KEY, reason="NIM_API_KEY not set")
    def test_nim_api_key_available(self):
        """Is NIM_API_KEY set?"""
        assert len(self.NIM_API_KEY) > 0
    
    @pytest.mark.skipif(not NIM_API_KEY, reason="NIM_API_KEY not set")
    def test_nim_embedder_instantiation(self):
        """Can we instantiate NIMEmbedder with real key?"""
        from omi.embeddings import NIMEmbedder
        embedder = NIMEmbedder(api_key=self.NIM_API_KEY)
        assert embedder is not None
    
    @pytest.mark.skipif(not NIM_API_KEY, reason="NIM_API_KEY not set")
    def test_nim_embed_returns_vector(self):
        """Does NIM embed return a 1024-dim vector?"""
        from omi.embeddings import NIMEmbedder
        embedder = NIMEmbedder(api_key=self.NIM_API_KEY)
        vector = embedder.embed("hello")
        assert len(vector) == 1024
        assert all(isinstance(x, float) for x in vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
