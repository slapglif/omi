"""
Comprehensive tests for security features (IntegrityChecker, TopologyVerifier, PoisonDetector, ConsensusManager)
"""
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
import hashlib

from omi.security import (
    IntegrityChecker,
    TopologyVerifier,
    PoisonDetector,
    ConsensusManager,
    AnomalyReport,
)


class TestIntegrityChecker:
    """Test IntegrityChecker functionality"""

    def test_hash_file(self, tmp_path):
        """Test SHA-256 hashing of file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        checker = IntegrityChecker(tmp_path)
        file_hash = checker.hash_file(test_file)

        # Verify hash is correct
        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert file_hash == expected_hash

    def test_check_now_md_first_run(self, tmp_path):
        """Test checking NOW.md on first run (no hash stored)"""
        now_file = tmp_path / "NOW.md"
        now_file.write_text("# NOW\nCurrent context")

        checker = IntegrityChecker(tmp_path)
        result = checker.check_now_md()

        # First run should pass and create hash file
        assert result is True
        assert (tmp_path / ".now.hash").exists()

    def test_check_now_md_unchanged(self, tmp_path):
        """Test checking NOW.md when unchanged"""
        now_file = tmp_path / "NOW.md"
        now_file.write_text("# NOW\nCurrent context")

        checker = IntegrityChecker(tmp_path)
        # First check creates hash
        checker.check_now_md()

        # Second check should pass (file unchanged)
        result = checker.check_now_md()
        assert result is True

    def test_check_now_md_modified(self, tmp_path):
        """Test checking NOW.md when modified"""
        now_file = tmp_path / "NOW.md"
        now_file.write_text("# NOW\nOriginal content")

        checker = IntegrityChecker(tmp_path)
        # First check creates hash
        checker.check_now_md()

        # Modify file
        now_file.write_text("# NOW\nModified content")

        # Check should fail (file changed)
        result = checker.check_now_md()
        assert result is False

    def test_check_now_md_missing(self, tmp_path):
        """Test checking NOW.md when file doesn't exist"""
        checker = IntegrityChecker(tmp_path)
        result = checker.check_now_md()

        # Should pass if file doesn't exist
        assert result is True

    def test_check_memory_md(self, tmp_path):
        """Test checking MEMORY.md integrity"""
        memory_file = tmp_path / "MEMORY.md"
        memory_file.write_text("# MEMORY\nLong-term memory")

        checker = IntegrityChecker(tmp_path)
        # First check should pass
        result = checker.check_memory_md()
        assert result is True

        # Second check should pass (unchanged)
        result = checker.check_memory_md()
        assert result is True

    def test_update_hashes(self, tmp_path):
        """Test updating stored hashes after modification"""
        now_file = tmp_path / "NOW.md"
        now_file.write_text("Original content")

        checker = IntegrityChecker(tmp_path)
        checker.check_now_md()  # Create initial hash

        # Modify file
        now_file.write_text("Modified content")

        # Update hash
        checker.update_hashes()

        # Check should now pass
        result = checker.check_now_md()
        assert result is True

    def test_audit_git_history_success(self, tmp_path):
        """Test auditing git history when git is available"""
        checker = IntegrityChecker(tmp_path)

        with patch('subprocess.run') as mock_run:
            # Mock git log output
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "abc123 Commit message\ndef456 Another commit"
            mock_run.return_value = mock_result

            result = checker.audit_git_history()

            assert result is not None
            assert "recent_commits" in result

    def test_audit_git_history_not_available(self, tmp_path):
        """Test audit when git is not available"""
        checker = IntegrityChecker(tmp_path)

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            result = checker.audit_git_history()

            assert result is not None
            assert "error" in result


class TestTopologyVerifier:
    """Test TopologyVerifier functionality"""

    def test_find_orphan_nodes_empty(self, tmp_path):
        """Test finding orphan nodes in empty graph"""
        mock_palace = MagicMock()
        mock_palace.get_stats.return_value = {"memory_count": 0}

        verifier = TopologyVerifier(mock_palace)
        orphans = verifier.find_orphan_nodes()

        assert orphans == []

    def test_find_orphan_nodes_with_db_path(self, tmp_path):
        """Test finding orphan nodes with SQL query"""
        db_path = tmp_path / "test.db"

        # Create mock palace with db_path
        mock_palace = MagicMock()
        mock_palace.db_path = str(db_path)

        # Create in-memory SQLite for testing
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, source_id TEXT, target_id TEXT)")
        conn.execute("INSERT INTO memories VALUES ('orphan_1')")
        conn.execute("INSERT INTO memories VALUES ('connected_1')")
        conn.execute("INSERT INTO edges VALUES (1, 'connected_1', 'connected_1')")
        conn.commit()
        conn.close()

        verifier = TopologyVerifier(mock_palace)
        orphans = verifier.find_orphan_nodes()

        assert len(orphans) == 1
        assert "orphan_1" in orphans

    def test_find_sudden_cores(self, tmp_path):
        """Test finding sudden core memories"""
        db_path = tmp_path / "test.db"

        mock_palace = MagicMock()
        mock_palace.db_path = str(db_path)

        # Create test database
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, access_count INTEGER)")
        conn.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, source_id TEXT, target_id TEXT)")
        # Create a "sudden core" - high in-degree but low access count
        conn.execute("INSERT INTO memories VALUES ('sudden_core', 'Important memory', 1)")
        for i in range(6):
            conn.execute(f"INSERT INTO edges VALUES ({i}, 'other_{i}', 'sudden_core')")
        conn.commit()
        conn.close()

        verifier = TopologyVerifier(mock_palace)
        sudden_cores = verifier.find_sudden_cores(min_in_edges=5)

        assert len(sudden_cores) >= 1
        assert any(sc["id"] == "sudden_core" for sc in sudden_cores)

    def test_check_embedding_drift(self):
        """Test checking for embedding drift"""
        mock_palace = MagicMock()
        mock_palace.get_memory.return_value = {
            "embedding": [0.1] * 768,
            "content": "Test content"
        }

        verifier = TopologyVerifier(mock_palace)

        with patch('omi.security.OllamaEmbedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = [0.1] * 768
            mock_embedder.similarity.return_value = 0.85  # Below 0.9 threshold
            mock_embedder_class.return_value = mock_embedder

            result = verifier.check_embedding_drift("memory_id")

            assert result is not None
            assert result["id"] == "memory_id"
            assert "warning" in result

    def test_check_embedding_drift_no_memory(self):
        """Test drift check when memory doesn't exist"""
        mock_palace = MagicMock()
        mock_palace.get_memory.return_value = None

        verifier = TopologyVerifier(mock_palace)
        result = verifier.check_embedding_drift("nonexistent")

        assert result is None

    def test_find_hash_mismatches(self, tmp_path):
        """Test finding memories with hash mismatches"""
        db_path = tmp_path / "test.db"

        mock_palace = MagicMock()
        mock_palace.db_path = str(db_path)

        # Create test database with mismatched hash
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, content_hash TEXT)")

        # Valid hash
        valid_content = "valid content"
        valid_hash = hashlib.sha256(valid_content.encode()).hexdigest()
        conn.execute("INSERT INTO memories VALUES ('valid', ?, ?)", (valid_content, valid_hash))

        # Invalid hash
        invalid_content = "actual content"
        wrong_hash = "wrong_hash_value"
        conn.execute("INSERT INTO memories VALUES ('invalid', ?, ?)", (invalid_content, wrong_hash))

        conn.commit()
        conn.close()

        verifier = TopologyVerifier(mock_palace)
        mismatches = verifier.find_hash_mismatches()

        assert len(mismatches) == 1
        assert "invalid" in mismatches

    def test_full_topology_audit(self, tmp_path):
        """Test full topology audit"""
        db_path = tmp_path / "test.db"

        mock_palace = MagicMock()
        mock_palace.db_path = str(db_path)

        # Create minimal test database
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, access_count INTEGER, content_hash TEXT)")
        conn.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, source_id TEXT, target_id TEXT)")
        conn.commit()
        conn.close()

        verifier = TopologyVerifier(mock_palace)
        report = verifier.full_topology_audit()

        assert isinstance(report, AnomalyReport)
        assert isinstance(report.orphan_nodes, list)
        assert isinstance(report.sudden_cores, list)
        assert isinstance(report.hash_mismatches, list)
        assert isinstance(report.timestamp, datetime)


class TestConsensusManager:
    """Test ConsensusManager functionality"""

    def test_init(self):
        """Test ConsensusManager initialization"""
        mock_palace = MagicMock()
        manager = ConsensusManager("instance_1", mock_palace, required_instances=3)

        assert manager.instance_id == "instance_1"
        assert manager.palace == mock_palace
        assert manager.required_instances == 3

    def test_propose_foundation_memory(self):
        """Test proposing a foundational memory"""
        mock_palace = MagicMock()
        mock_palace.store_memory.return_value = "memory_id_123"
        mock_palace.get_consensus_votes.return_value = 1

        manager = ConsensusManager("instance_1", mock_palace, required_instances=3)
        memory_id = manager.propose_foundation_memory("Important fact")

        assert memory_id == "memory_id_123"
        mock_palace.store_memory.assert_called_once()
        mock_palace.add_consensus_vote.assert_called_once()

    def test_propose_foundation_memory_reaches_consensus(self):
        """Test proposing memory that reaches consensus"""
        mock_palace = MagicMock()
        mock_palace.store_memory.return_value = "memory_id_123"
        mock_palace.get_consensus_votes.return_value = 3  # Meets threshold

        manager = ConsensusManager("instance_1", mock_palace, required_instances=3)
        memory_id = manager.propose_foundation_memory("Important fact")

        # Should mark as foundational
        mock_palace.mark_as_foundational.assert_called_once_with("memory_id_123")

    def test_support_memory(self):
        """Test supporting an existing memory"""
        mock_palace = MagicMock()
        manager = ConsensusManager("instance_2", mock_palace)

        manager.support_memory("memory_id_123")

        mock_palace.add_consensus_vote.assert_called_once_with(
            memory_id="memory_id_123",
            instance_id="instance_2",
            votes_for=1
        )

    def test_check_consensus(self):
        """Test checking consensus status"""
        mock_palace = MagicMock()
        mock_palace.get_consensus_votes.return_value = 2

        manager = ConsensusManager("instance_1", mock_palace, required_instances=3)
        status = manager.check_consensus("memory_id_123")

        assert status["memory_id"] == "memory_id_123"
        assert status["votes_for"] == 2
        assert status["required"] == 3
        assert status["is_foundational"] is False

    def test_check_consensus_met(self):
        """Test consensus when threshold is met"""
        mock_palace = MagicMock()
        mock_palace.get_consensus_votes.return_value = 3

        manager = ConsensusManager("instance_1", mock_palace, required_instances=3)
        status = manager.check_consensus("memory_id_123")

        assert status["is_foundational"] is True


class TestPoisonDetector:
    """Test PoisonDetector unified detection"""

    def test_init_without_palace(self, tmp_path):
        """Test initialization without palace store"""
        detector = PoisonDetector(tmp_path)

        assert detector.integrity is not None
        assert detector.topology is None

    def test_init_with_palace(self, tmp_path):
        """Test initialization with palace store"""
        mock_palace = MagicMock()
        detector = PoisonDetector(tmp_path, mock_palace)

        assert detector.integrity is not None
        assert detector.topology is not None

    def test_full_security_audit_without_palace(self, tmp_path):
        """Test full audit without topology checks"""
        now_file = tmp_path / "NOW.md"
        now_file.write_text("# NOW")

        detector = PoisonDetector(tmp_path)

        with patch.object(detector.integrity, 'audit_git_history', return_value={"recent_commits": 0}):
            result = detector.full_security_audit()

            assert "file_integrity" in result
            assert result["orphan_nodes"] == []
            assert result["sudden_cores"] == []
            assert "git_audit" in result
            assert "overall_safe" in result

    def test_full_security_audit_with_palace(self, tmp_path):
        """Test full audit with topology checks"""
        mock_palace = MagicMock()
        detector = PoisonDetector(tmp_path, mock_palace)

        # Mock topology audit
        mock_audit = AnomalyReport(
            orphan_nodes=["orphan_1"],
            sudden_cores=[],
            semantic_anomalies=[],
            hash_mismatches=[],
            timestamp=datetime.now()
        )

        with patch.object(detector.topology, 'full_topology_audit', return_value=mock_audit):
            with patch.object(detector.integrity, 'audit_git_history', return_value={}):
                result = detector.full_security_audit()

                assert len(result["orphan_nodes"]) == 1
                assert result["orphan_nodes"][0] == "orphan_1"

    def test_full_security_audit_unsafe(self, tmp_path):
        """Test audit that detects unsafe conditions"""
        mock_palace = MagicMock()
        detector = PoisonDetector(tmp_path, mock_palace)

        # Create many orphan nodes (unsafe condition)
        mock_audit = AnomalyReport(
            orphan_nodes=["orphan_" + str(i) for i in range(10)],
            sudden_cores=[],
            semantic_anomalies=[],
            hash_mismatches=[],
            timestamp=datetime.now()
        )

        with patch.object(detector.topology, 'full_topology_audit', return_value=mock_audit):
            with patch.object(detector.integrity, 'audit_git_history', return_value={}):
                with patch.object(detector.integrity, 'check_now_md', return_value=True):
                    with patch.object(detector.integrity, 'check_memory_md', return_value=True):
                        result = detector.full_security_audit()

                        # Should be unsafe due to many orphans
                        assert result["overall_safe"] is False


class TestAnomalyReport:
    """Test AnomalyReport dataclass"""

    def test_anomaly_report_creation(self):
        """Test creating AnomalyReport"""
        report = AnomalyReport(
            orphan_nodes=["node1", "node2"],
            sudden_cores=[{"id": "core1"}],
            semantic_anomalies=[{"id": "anomaly1"}],
            hash_mismatches=["mismatch1"],
            timestamp=datetime(2024, 1, 1)
        )

        assert len(report.orphan_nodes) == 2
        assert len(report.sudden_cores) == 1
        assert len(report.semantic_anomalies) == 1
        assert len(report.hash_mismatches) == 1
        assert report.timestamp == datetime(2024, 1, 1)
