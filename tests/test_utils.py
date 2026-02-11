"""Unit Tests for Utils Module

Tests: cosine_similarity, hash_file, hash_content
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        from src.omi.utils import cosine_similarity

        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        from src.omi.utils import cosine_similarity

        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]

        result = cosine_similarity(v1, v2)
        assert abs(result - 0.0) < 0.0001

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        from src.omi.utils import cosine_similarity

        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]

        result = cosine_similarity(v1, v2)
        assert abs(result - (-1.0)) < 0.0001

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 (edge case)."""
        from src.omi.utils import cosine_similarity

        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]

        result = cosine_similarity(v1, v2)
        assert result == 0.0

    def test_both_zero_vectors(self):
        """Two zero vectors should return 0.0."""
        from src.omi.utils import cosine_similarity

        v1 = [0.0, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0]

        result = cosine_similarity(v1, v2)
        assert result == 0.0

    def test_numpy_arrays(self):
        """Should work with numpy arrays."""
        from src.omi.utils import cosine_similarity

        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001

    def test_mixed_list_and_array(self):
        """Should work with mixed list and numpy array."""
        from src.omi.utils import cosine_similarity

        v1 = [1.0, 2.0, 3.0]
        v2 = np.array([1.0, 2.0, 3.0])

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001

    def test_normalized_vectors(self):
        """Pre-normalized vectors should work correctly."""
        from src.omi.utils import cosine_similarity

        # Unit vectors at 45 degrees
        v1 = [1.0, 0.0]
        v2 = [0.707107, 0.707107]

        result = cosine_similarity(v1, v2)
        assert abs(result - 0.707107) < 0.001

    def test_different_magnitudes_same_direction(self):
        """Vectors with same direction but different magnitude should have similarity 1.0."""
        from src.omi.utils import cosine_similarity

        v1 = [1.0, 2.0, 3.0]
        v2 = [2.0, 4.0, 6.0]

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001

    def test_negative_values(self):
        """Should handle negative values correctly."""
        from src.omi.utils import cosine_similarity

        v1 = [-1.0, -2.0, -3.0]
        v2 = [-1.0, -2.0, -3.0]

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001

    def test_high_dimensional_vectors(self):
        """Should work with high-dimensional vectors."""
        from src.omi.utils import cosine_similarity

        v1 = np.random.rand(1536)  # OpenAI embedding dimension
        v2 = v1.copy()

        result = cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 0.0001


class TestHashFile:
    """Tests for file hashing function."""

    def test_hash_file_creates_64_char_hash(self, tmp_path):
        """Hash should be 64 character hex string (SHA-256)."""
        from src.omi.utils import hash_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = hash_file(test_file)

        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_same_content_same_hash(self, tmp_path):
        """Same content should produce same hash."""
        from src.omi.utils import hash_file

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("identical content")
        file2.write_text("identical content")

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path):
        """Different content should produce different hash."""
        from src.omi.utils import hash_file

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A")
        file2.write_text("content B")

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        assert hash1 != hash2

    def test_empty_file(self, tmp_path):
        """Empty file should produce valid hash."""
        from src.omi.utils import hash_file

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = hash_file(empty_file)

        assert len(result) == 64
        # Known SHA-256 hash of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_large_file(self, tmp_path):
        """Should handle large files (tests chunked reading)."""
        from src.omi.utils import hash_file

        large_file = tmp_path / "large.txt"
        # Create file larger than chunk size (8192 bytes)
        large_file.write_text("x" * 100000)

        result = hash_file(large_file)

        assert len(result) == 64

    def test_binary_file(self, tmp_path):
        """Should handle binary files."""
        from src.omi.utils import hash_file

        binary_file = tmp_path / "binary.dat"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe\xfd')

        result = hash_file(binary_file)

        assert len(result) == 64

    def test_unicode_content(self, tmp_path):
        """Should handle unicode content correctly."""
        from src.omi.utils import hash_file

        unicode_file = tmp_path / "unicode.txt"
        unicode_file.write_text("Hello 世界 🌍", encoding='utf-8')

        result = hash_file(unicode_file)

        assert len(result) == 64

    def test_file_not_found_raises_error(self):
        """Non-existent file should raise error."""
        from src.omi.utils import hash_file

        non_existent = Path("/tmp/does_not_exist_12345.txt")

        with pytest.raises(FileNotFoundError):
            hash_file(non_existent)


class TestHashContent:
    """Tests for content hashing function."""

    def test_hash_content_creates_64_char_hash(self):
        """Hash should be 64 character hex string (SHA-256)."""
        from src.omi.utils import hash_content

        result = hash_content("test content")

        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        from src.omi.utils import hash_content

        hash1 = hash_content("identical content")
        hash2 = hash_content("identical content")

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        from src.omi.utils import hash_content

        hash1 = hash_content("content A")
        hash2 = hash_content("content B")

        assert hash1 != hash2

    def test_empty_string(self):
        """Empty string should produce valid hash."""
        from src.omi.utils import hash_content

        result = hash_content("")

        assert len(result) == 64
        # Known SHA-256 hash of empty string
        assert result == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_unicode_content(self):
        """Should handle unicode content correctly."""
        from src.omi.utils import hash_content

        result = hash_content("Hello 世界 🌍")

        assert len(result) == 64

    def test_whitespace_sensitivity(self):
        """Hash should be sensitive to whitespace differences."""
        from src.omi.utils import hash_content

        hash1 = hash_content("hello world")
        hash2 = hash_content("hello  world")  # Double space
        hash3 = hash_content("hello\nworld")  # Newline

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_case_sensitivity(self):
        """Hash should be case sensitive."""
        from src.omi.utils import hash_content

        hash1 = hash_content("Hello")
        hash2 = hash_content("hello")

        assert hash1 != hash2

    def test_multiline_content(self):
        """Should handle multiline content correctly."""
        from src.omi.utils import hash_content

        content = """Line 1
Line 2
Line 3"""

        result = hash_content(content)

        assert len(result) == 64

    def test_special_characters(self):
        """Should handle special characters correctly."""
        from src.omi.utils import hash_content

        content = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"

        result = hash_content(content)

        assert len(result) == 64

    def test_long_content(self):
        """Should handle long content correctly."""
        from src.omi.utils import hash_content

        # Create large content
        content = "x" * 100000

        result = hash_content(content)

        assert len(result) == 64

    def test_json_like_content(self):
        """Should handle JSON-like string content."""
        from src.omi.utils import hash_content

        content = '{"key": "value", "number": 42}'

        result = hash_content(content)

        assert len(result) == 64


class TestHashConsistency:
    """Tests for consistency between hash_file and hash_content."""

    def test_file_and_content_hash_match(self, tmp_path):
        """hash_file and hash_content should produce same result for same data."""
        from src.omi.utils import hash_file, hash_content

        content_str = "test content for consistency"

        # Create file with content
        test_file = tmp_path / "test.txt"
        test_file.write_text(content_str, encoding='utf-8')

        file_hash = hash_file(test_file)
        content_hash = hash_content(content_str)

        assert file_hash == content_hash

    def test_multiline_consistency(self, tmp_path):
        """Multiline content should hash consistently."""
        from src.omi.utils import hash_file, hash_content

        content_str = """First line
Second line
Third line"""

        test_file = tmp_path / "multiline.txt"
        test_file.write_text(content_str, encoding='utf-8')

        file_hash = hash_file(test_file)
        content_hash = hash_content(content_str)

        assert file_hash == content_hash

    def test_unicode_consistency(self, tmp_path):
        """Unicode content should hash consistently."""
        from src.omi.utils import hash_file, hash_content

        content_str = "Hello 世界 🌍"

        test_file = tmp_path / "unicode.txt"
        test_file.write_text(content_str, encoding='utf-8')

        file_hash = hash_file(test_file)
        content_hash = hash_content(content_str)

        assert file_hash == content_hash
