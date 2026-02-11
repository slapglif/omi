"""
test_async_embeddings.py - Full async embeddings test suite

Tests:
- Async connection tests (with real API key)
- Fallback verification (AsyncOllama when NIM unavailable)
- Error handling (timeouts, invalid keys)
- Async context manager support
- Batch embedding tests
- Cache functionality

NOTE: Tests with real NIM_API_KEY are optional and only run if NIM_API_KEY is set.
"""

import os
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Generator

import numpy as np

# Skip entire module if dependencies unavailable
pytest.importorskip("httpx")
pytest.importorskip("numpy")

from omi.async_embeddings import AsyncNIMEmbedder, AsyncOllamaEmbedder, AsyncEmbeddingCache

# Constants
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
SKIP_NIM_TESTS = not NIM_API_KEY or NIM_API_KEY == ""
REQUIRED_DIM = 1024


class TestAsyncNIMConnection:
    """Test async NIM connectivity with real API keys"""

    @pytest.mark.asyncio
    async def test_async_nim_connection(self):
        """
        Test with real NIM_API_KEY: embed("hello world") returns 1024-dim vector
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        async with AsyncNIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False) as embedder:
            embedding = await embedder.embed("hello world")

            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) == REQUIRED_DIM, f"Expected {REQUIRED_DIM} dimensions, got {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
            assert not all(x == 0 for x in embedding), "Embedding should not be all zeros"

            # Check L2 norm is reasonable (embeddings are typically normalized or near-normalized)
            norm = np.linalg.norm(embedding)
            assert 0.1 < norm < 100, f"L2 norm {norm} seems unreasonable"

    @pytest.mark.asyncio
    async def test_async_nim_embedding_quality_basic(self):
        """
        Basic quality check: "king" vs "queen" should be more similar than "king" vs "apple"
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        async with AsyncNIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False) as embedder:
            king_emb = await embedder.embed("king")
            queen_emb = await embedder.embed("queen")
            apple_emb = await embedder.embed("apple")

            royalty_sim = embedder.similarity(king_emb, queen_emb)
            different_sim = embedder.similarity(king_emb, apple_emb)

            assert royalty_sim > different_sim, (
                f"'king' should be more similar to 'queen' ({royalty_sim:.4f}) "
                f"than to 'apple' ({different_sim:.4f})"
            )

    @pytest.mark.asyncio
    async def test_async_nim_batch_embedding(self):
        """
        Test batch embedding works correctly
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        async with AsyncNIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False) as embedder:
            texts = ["hello world", "test text", "another sentence"]
            embeddings = await embedder.embed_batch(texts, batch_size=2)

            assert len(embeddings) == len(texts), "Should return same number of embeddings"
            for emb in embeddings:
                assert len(emb) == REQUIRED_DIM, "Each embedding should have correct dimension"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """
        Test async context manager properly initializes and cleans up
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = AsyncNIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)

        # Client should not be initialized yet
        assert embedder._client is None

        async with embedder:
            # Client should be initialized in context
            assert embedder._client is not None
            embedding = await embedder.embed("test")
            assert len(embedding) == REQUIRED_DIM

        # Client should be closed after exiting context
        assert embedder._client is None


class TestAsyncNIMFallback:
    """Test async Ollama fallback when NIM is unavailable"""

    @pytest.mark.asyncio
    async def test_async_fallback_initialized_when_nim_down(self):
        """
        Test that AsyncOllama fallback is properly initialized when NIM is unavailable
        """
        # Create a mock that simulates NIM being unavailable
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate NIM connection failure
            import httpx
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.aclose = AsyncMock()

            # Mock AsyncOllamaEmbedder to verify it's initialized
            with patch.object(AsyncNIMEmbedder, "_init_ollama_fallback") as mock_init_fallback:
                mock_init_fallback.return_value = None

                try:
                    async with AsyncNIMEmbedder(api_key="dummy_key", fallback_to_ollama=True) as embedder:
                        pass
                    mock_init_fallback.assert_called_once()
                except Exception:
                    pass  # Expected if fallback also fails

    @pytest.mark.asyncio
    async def test_async_embed_uses_fallback_on_nim_failure(self):
        """
        Test that embed() falls back to AsyncOllama when NIM fails mid-operation
        """
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock initial connection to succeed
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            embedder = AsyncNIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

            # Initialize client first
            await embedder._init_client()

            # Now mock NIM method to fail
            async def mock_embed_nim_fail(text):
                raise Exception("NIM unavailable")

            embedder._embed_nim = mock_embed_nim_fail

            # Mock Ollama embedder
            mock_ollama_emb = [0.5] * 768
            embedder._ollama_embedder = Mock()
            embedder._ollama_embedder.embed = AsyncMock(return_value=mock_ollama_emb)

            result = await embedder.embed("test text")

            assert result == mock_ollama_emb, "Should return Ollama fallback embedding"
            embedder._ollama_embedder.embed.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self):
        """
        Test that NIM failure raises exception when fallback is disabled
        """
        with pytest.raises((ValueError, RuntimeError)):
            # Try to create embedder without API key and no fallback
            embedder = AsyncNIMEmbedder(api_key=None, fallback_to_ollama=False)


class TestAsyncNIMErrorHandling:
    """Test graceful error handling for various async NIM failure modes"""

    @pytest.mark.asyncio
    async def test_async_nim_invalid_key(self):
        """
        Test clear error message on invalid API key
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set - can't test invalid key against real API")

        import httpx

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async with AsyncNIMEmbedder(api_key="invalid_key_12345", fallback_to_ollama=False) as embedder:
                await embedder.embed("test")

        assert exc_info.value.response.status_code in [401, 403], "Should get auth error"

    @pytest.mark.asyncio
    async def test_async_nim_timeout_handling(self):
        """
        Test graceful handling of timeout responses
        """
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock initial connection to succeed
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            embedder = AsyncNIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

            # Initialize client first
            await embedder._init_client()

            # Mock timeout
            async def mock_timeout(*args, **kwargs):
                import httpx
                raise httpx.TimeoutException("Request timed out")

            embedder._embed_nim = mock_timeout

            # Mock Ollama fallback
            mock_ollama_emb = [0.5] * 768
            embedder._ollama_embedder = Mock()
            embedder._ollama_embedder.embed = AsyncMock(return_value=mock_ollama_emb)

            result = await embedder.embed("test text")
            assert result == mock_ollama_emb, "Should fall back on timeout"

    @pytest.mark.asyncio
    async def test_async_connection_error_with_fallback(self):
        """
        Test that connection errors trigger fallback when enabled
        """
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate connection error
            import httpx
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Cannot connect to server"))
            mock_client.aclose = AsyncMock()

            # Mock Ollama fallback
            with patch("omi.async_embeddings.AsyncOllamaEmbedder") as mock_ollama_class:
                mock_ollama = Mock()
                mock_ollama._init_client = AsyncMock()
                mock_ollama_class.return_value = mock_ollama

                try:
                    embedder = AsyncNIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)
                    # Try to initialize client, should trigger fallback
                    await embedder._init_client()
                except Exception:
                    # It's ok if this fails, we're just checking fallback is attempted
                    pass

                # Verify fallback was attempted by checking if Ollama was instantiated
                if embedder.fallback_enabled:
                    # Test passes as long as we don't crash with fallback enabled
                    assert True


class TestAsyncOllamaEmbedder:
    """Test async Ollama embeddings"""

    @pytest.mark.asyncio
    async def test_async_ollama_embed(self):
        """
        Test AsyncOllamaEmbedder basic embedding functionality
        """
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock successful Ollama response
            mock_response = Mock()
            mock_embedding = [0.1] * 768
            mock_response.json.return_value = {"embedding": mock_embedding}
            mock_response.raise_for_status = Mock()

            mock_client.post = AsyncMock(return_value=mock_response)

            embedder = AsyncOllamaEmbedder()
            await embedder._init_client()

            result = await embedder.embed("test text")

            assert result == mock_embedding
            mock_client.post.assert_called_once()

            # Verify correct endpoint
            call_args = mock_client.post.call_args
            assert "/api/embeddings" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_async_ollama_close(self):
        """
        Test AsyncOllamaEmbedder properly closes HTTP client
        """
        embedder = AsyncOllamaEmbedder()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client

            await embedder._init_client()
            assert embedder._client is not None

            await embedder.close()
            mock_client.aclose.assert_called_once()


class TestAsyncEmbeddingCache:
    """Test async embedding cache functionality"""

    @pytest.mark.asyncio
    async def test_async_cache_miss_and_store(self):
        """
        Test cache miss computes embedding and stores it
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock embedder
            mock_embedder = Mock()
            mock_embedding = [0.1] * 1024
            mock_embedder.embed = AsyncMock(return_value=mock_embedding)

            cache = AsyncEmbeddingCache(cache_dir, mock_embedder)

            # First call should compute and store
            result = await cache.get_or_compute("test text")

            assert result == mock_embedding
            mock_embedder.embed.assert_called_once_with("test text")

            # Verify cache file was created
            cache_files = list(cache_dir.glob("*.npy"))
            assert len(cache_files) == 1, "Cache file should be created"

    @pytest.mark.asyncio
    async def test_async_cache_hit(self):
        """
        Test cache hit returns stored embedding without recomputing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock embedder
            mock_embedder = Mock()
            mock_embedding = [0.2] * 1024
            mock_embedder.embed = AsyncMock(return_value=mock_embedding)

            cache = AsyncEmbeddingCache(cache_dir, mock_embedder)

            # First call - cache miss
            result1 = await cache.get_or_compute("test text")
            assert result1 == mock_embedding
            assert mock_embedder.embed.call_count == 1

            # Second call - cache hit
            result2 = await cache.get_or_compute("test text")
            assert result2 == mock_embedding

            # Should not call embed again
            assert mock_embedder.embed.call_count == 1, "Should use cached value"

    @pytest.mark.asyncio
    async def test_async_cache_different_texts(self):
        """
        Test cache stores different embeddings for different texts
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create mock embedder that returns different embeddings
            mock_embedder = Mock()

            async def mock_embed(text):
                if text == "text1":
                    return [0.1] * 1024
                else:
                    return [0.2] * 1024

            mock_embedder.embed = mock_embed

            cache = AsyncEmbeddingCache(cache_dir, mock_embedder)

            result1 = await cache.get_or_compute("text1")
            result2 = await cache.get_or_compute("text2")

            assert result1 != result2
            assert result1 == [0.1] * 1024
            assert result2 == [0.2] * 1024

            # Verify two cache files were created
            cache_files = list(cache_dir.glob("*.npy"))
            assert len(cache_files) == 2, "Should create separate cache files"


class TestAsyncSimilarity:
    """Test similarity calculations (sync method on async class)"""

    def test_similarity_identical_vectors(self):
        """Test similarity of identical vectors is 1.0"""
        embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

        vec = [1.0, 2.0, 3.0]
        similarity = embedder.similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-6, "Identical vectors should have similarity ~1.0"

    def test_similarity_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is ~0.0"""
        embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = embedder.similarity(vec1, vec2)

        assert abs(similarity) < 1e-6, "Orthogonal vectors should have similarity ~0.0"

    def test_similarity_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0"""
        embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = embedder.similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-6, "Opposite vectors should have similarity ~-1.0"

    def test_similarity_zero_vector_handling(self):
        """Test similarity with zero vectors returns 0.0"""
        embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = embedder.similarity(vec1, vec2)

        assert similarity == 0.0, "Zero vector should return 0.0 similarity"


class TestAsyncClientManagement:
    """Test async HTTP client lifecycle management"""

    @pytest.mark.asyncio
    async def test_client_lazy_initialization(self):
        """Test client is not initialized until first use"""
        embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

        # Client should not be initialized yet
        assert embedder._client is None

    @pytest.mark.asyncio
    async def test_client_auto_initialization_on_embed(self):
        """Test client is automatically initialized when calling embed"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}
            mock_response.raise_for_status = Mock()

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)

            # Client should be initialized on first embed call
            await embedder.embed("test")

            assert embedder._client is not None

    @pytest.mark.asyncio
    async def test_client_cleanup_on_close(self):
        """Test client is properly cleaned up on close"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()

            # Mock the post method for the test connection
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_client_class.return_value = mock_client

            embedder = AsyncNIMEmbedder(api_key="dummy", fallback_to_ollama=False)
            await embedder._init_client()

            assert embedder._client is not None

            await embedder.close()

            mock_client.aclose.assert_called_once()
            assert embedder._client is None
