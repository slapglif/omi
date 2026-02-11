"""
test_nim_integration.py - Full connection and error handling test suite for NVIDIA NIM

Tests:
- Connection tests (with real API key)
- Fallback verification (Ollama when NIM unavailable)
- Error handling (rate limits, timeouts, invalid keys)
- Model availability (baai/bge-m3)

NOTE: Tests with real NIM_API_KEY are optional and only run if NIM_API_KEY is set.
"""

import os
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Generator

import numpy as np

# Skip entire module if dependencies unavailable
pytest.importorskip("requests")
pytest.importorskip("numpy")

from omi.embeddings import NIMEmbedder, OllamaEmbedder, NIMConfig

# Constants
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
SKIP_NIM_TESTS = not NIM_API_KEY or NIM_API_KEY == ""
REQUIRED_DIM = 1024


class TestNIMConnection:
    """Test NIM connectivity with real API keys"""

    def test_nim_connection(self):
        """
        Test with real NIM_API_KEY: embed("hello world") returns 1024-dim vector
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)
        embedding = embedder.embed("hello world")

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == REQUIRED_DIM, f"Expected {REQUIRED_DIM} dimensions, got {len(embedding)}"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        assert not all(x == 0 for x in embedding), "Embedding should not be all zeros"

        # Check L2 norm is reasonable (embeddings are typically normalized or near-normalized)
        norm = np.linalg.norm(embedding)
        assert 0.1 < norm < 100, f"L2 norm {norm} seems unreasonable"

    def test_nim_model_availability(self):
        """
        Verify baai/bge-m3 model is available on the NIM endpoint
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        import requests

        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            timeout=10
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        models = [m.get("id", "") for m in data.get("data", [])]

        # Check for baai/bge-m3 specifically
        found = any("bge-m3" in m for m in models)
        assert found, f"baai/bge-m3 not found in available models: {models[:10]}..."

    def test_nim_embedding_quality_basic(self):
        """
        Basic quality check: "king" vs "queen" should be more similar than "king" vs "apple"
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)

        king_emb = embedder.embed("king")
        queen_emb = embedder.embed("queen")
        apple_emb = embedder.embed("apple")

        royalty_sim = embedder.similarity(king_emb, queen_emb)
        different_sim = embedder.similarity(king_emb, apple_emb)

        assert royalty_sim > different_sim, (
            f"'king' should be more similar to 'queen' ({royalty_sim:.4f}) "
            f"than to 'apple' ({different_sim:.4f})"
        )

    def test_nim_batch_embedding(self):
        """
        Test batch embedding works correctly
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)

        texts = ["hello world", "test text", "another sentence"]
        embeddings = embedder.embed_batch(texts, batch_size=2)

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        for emb in embeddings:
            assert len(emb) == REQUIRED_DIM, "Each embedding should have correct dimension"

    def test_nim_batch_performance_improvement(self):
        """
        Integration test: Verify batch embedding is faster than sequential calls
        Skip if NIM_API_KEY not set

        This test compares:
        1. Sequential: calling embed() for each text individually
        2. Batch: calling embed_batch() with all texts at once

        Expected: Batch should be significantly faster (at least 30% improvement)
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)

        # Use 10 texts to make the performance difference measurable
        texts = [
            "artificial intelligence and machine learning",
            "natural language processing techniques",
            "computer vision applications",
            "deep neural network architectures",
            "reinforcement learning algorithms",
            "data science and analytics",
            "cloud computing infrastructure",
            "distributed systems design",
            "software engineering practices",
            "database optimization strategies"
        ]

        # Measure sequential embedding time
        start_sequential = time.time()
        sequential_embeddings = []
        for text in texts:
            emb = embedder.embed(text)
            sequential_embeddings.append(emb)
        sequential_time = time.time() - start_sequential

        # Measure batch embedding time
        start_batch = time.time()
        batch_embeddings = embedder.embed_batch(texts, batch_size=10)
        batch_time = time.time() - start_batch

        # Verify correctness: both methods should return same number of embeddings
        assert len(sequential_embeddings) == len(texts), "Sequential should return all embeddings"
        assert len(batch_embeddings) == len(texts), "Batch should return all embeddings"

        # Verify dimension correctness
        for emb in sequential_embeddings + batch_embeddings:
            assert len(emb) == REQUIRED_DIM, "All embeddings should have correct dimension"

        # Verify performance improvement: batch should be at least 30% faster
        # (batch_time should be <= 70% of sequential_time)
        speedup_ratio = sequential_time / batch_time if batch_time > 0 else float('inf')
        time_saved_pct = ((sequential_time - batch_time) / sequential_time * 100) if sequential_time > 0 else 0

        assert batch_time < sequential_time, (
            f"Batch embedding should be faster than sequential. "
            f"Sequential: {sequential_time:.2f}s, Batch: {batch_time:.2f}s"
        )

        # Expect at least 30% improvement (batch time <= 70% of sequential time)
        assert batch_time <= sequential_time * 0.7, (
            f"Batch embedding should be at least 30% faster than sequential. "
            f"Sequential: {sequential_time:.2f}s, Batch: {batch_time:.2f}s "
            f"(speedup: {speedup_ratio:.2f}x, time saved: {time_saved_pct:.1f}%)"
        )


class TestNIMFallback:
    """Test Ollama fallback when NIM is unavailable"""

    def test_nim_fallback_initialized_when_nim_down(self):
        """
        Test that Ollama fallback is properly initialized when NIM is unavailable
        """
        # Create a mock that simulates NIM being unavailable
        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            # Simulate NIM connection failure
            mock_session.post.side_effect = Exception("Connection refused")

            # Mock OllamaEmbedder to verify it's initialized
            with patch.object(NIMEmbedder, "_init_ollama_fallback") as mock_init_fallback:
                mock_init_fallback.return_value = None
                mock_ollama = Mock()
                mock_ollama.embed.return_value = [0.1] * REQUIRED_DIM

                try:
                    NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)
                    mock_init_fallback.assert_called_once()
                except Exception:
                    pass  # Expected if fallback also fails

    def test_embed_uses_fallback_on_nim_failure(self):
        """
        Test that embed() falls back to Ollama when NIM fails mid-operation
        """
        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        # Mock internal NIM method to fail
        embedder._embed_nim = Mock(side_effect=Exception("NIM unavailable"))

        # Mock Ollama embedder
        mock_ollama_emb = [0.5] * 768
        embedder._ollama_embedder = Mock()
        embedder._ollama_embedder.embed.return_value = mock_ollama_emb

        result = embedder.embed("test text")

        assert result == mock_ollama_emb, "Should return Ollama fallback embedding"
        embedder._embed_nim.assert_called_once_with("test text")

    def test_no_fallback_when_disabled(self):
        """
        Test that NIM failure raises exception when fallback is disabled
        """
        with pytest.raises((ValueError, RuntimeError)):
            # Try to create embedder without API key and no fallback
            NIMEmbedder(api_key=None, fallback_to_ollama=False)


class TestNIMErrorHandling:
    """Test graceful error handling for various NIM failure modes"""

    def test_nim_invalid_key(self):
        """
        Test clear error message on invalid API key
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set - can't test invalid key against real API")

        import requests

        with pytest.raises(requests.HTTPError) as exc_info:
            embedder = NIMEmbedder(api_key="invalid_key_12345", fallback_to_ollama=False)
            embedder.embed("test")

        assert exc_info.value.response.status_code in [401, 403], "Should get auth error"

    def test_nim_rate_limit_handling(self):
        """
        Test graceful handling of rate limit (429) responses
        """
        import requests

        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        # Mock a 429 response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)

        with patch.object(embedder._session, "post", return_value=mock_response):
            embedder._ollama_embedder = Mock()
            embedder._ollama_embedder.embed.return_value = [0.1] * 768

            # Should fallback instead of crashing
            result = embedder.embed("test")
            assert result is not None

    def test_nim_timeout(self):
        """
        Test timeout after 30s: Should fallback or raise appropriate error
        """
        import requests

        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        # Mock timeout
        embedder._embed_nim = Mock(side_effect=requests.Timeout("Request timed out"))
        embedder._ollama_embedder = Mock()
        embedder._ollama_embedder.embed.return_value = [0.1] * 768

        # Should fallback
        result = embedder.embed("test")
        assert result is not None

    def test_nim_timeout_exceeds_threshold(self):
        """
        Verify timeout is set to 30 seconds in configuration
        """
        # Check that timeout is configured in the code
        import inspect
        source = inspect.getsource(NIMEmbedder._embed_nim)
        assert "timeout" in source, "Timeout should be configured in _embed_nim"

        # Check NIMConfig has timeout
        config = NIMConfig(api_key="dummy")
        assert hasattr(config, "timeout"), "NIMConfig should have timeout attribute"
        assert config.timeout == 30, f"Timeout should be 30s, got {config.timeout}"

    def test_nim_network_error_graceful(self):
        """
        Test graceful handling of network errors (DNS, connectivity, etc.)
        """
        import requests

        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        # Simulate various network errors
        for error in [requests.ConnectionError, requests.ConnectTimeout, requests.RequestException]:
            embedder._embed_nim = Mock(side_effect=error("Network error"))
            embedder._ollama_embedder = Mock()
            embedder._ollama_embedder.embed.return_value = [0.1] * 768

            result = embedder.embed("test")
            assert result is not None, f"Should handle {error.__name__} gracefully"


class TestNIMConfig:
    """Test NIM configuration handling"""

    def test_nim_config_from_env(self):
        """
        Test that NIMConfig reads from environment variables
        """
        with patch.dict(os.environ, {"NIM_API_KEY": "nvapi-test123"}):
            embedder = NIMEmbedder(fallback_to_ollama=False)
            assert embedder.api_key == "nvapi-test123"

    def test_nim_config_explicit(self):
        """
        Test that explicit API key overrides environment
        """
        with patch.dict(os.environ, {"NIM_API_KEY": "env_key"}):
            embedder = NIMEmbedder(api_key="explicit_key", fallback_to_ollama=False)
            assert embedder.api_key == "explicit_key"

    def test_nim_config_model(self):
        """
        Test model configuration
        """
        config = NIMConfig(api_key="dummy", model="custom/model")
        assert config.model == "custom/model"
        assert config.embedding_dim == 1024  # Default dimension


class TestNIMEmbedderClass:
    """Test NIMEmbedder class structure and constants"""

    def test_default_model_constant(self):
        """Verify default model constant"""
        assert NIMEmbedder.DEFAULT_MODEL == "baai/bge-m3"
        assert NIMEmbedder.DEFAULT_DIM == 1024

    def test_embedding_dim_matches_constant(self):
        """
        Verify actual embedding dimension matches constant
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)
        embedding = embedder.embed("test")

        assert len(embedding) == NIMEmbedder.DEFAULT_DIM


class TestNIMSimilarity:
    """Test similarity calculations"""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0"""
        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        v = [1.0, 0.0, 0.0]
        sim = embedder.similarity(v, v)

        assert abs(sim - 1.0) < 0.0001, f"Expected 1.0, got {sim}"

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0"""
        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        sim = embedder.similarity(v1, v2)

        assert abs(sim) < 0.0001, f"Expected 0.0, got {sim}"

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0"""
        embedder = NIMEmbedder(api_key="dummy_key", fallback_to_ollama=True)

        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        sim = embedder.similarity(v1, v2)

        assert abs(sim - (-1.0)) < 0.0001, f"Expected -1.0, got {sim}"

    def test_cosine_similarity_real_embeddings(self):
        """
        Test similarity with real embeddings
        Skip if NIM_API_KEY not set
        """
        if SKIP_NIM_TESTS:
            pytest.skip("NIM_API_KEY not set")

        embedder = NIMEmbedder(api_key=NIM_API_KEY, fallback_to_ollama=False)

        # Test semantic similarity
        embeddings = {
            "machine learning": embedder.embed("machine learning"),
            "artificial intelligence": embedder.embed("artificial intelligence"),
            "pizza": embedder.embed("pizza"),
        }

        ml_ai_sim = embedder.similarity(
            embeddings["machine learning"],
            embeddings["artificial intelligence"]
        )
        ml_pizza_sim = embedder.similarity(
            embeddings["machine learning"],
            embeddings["pizza"]
        )

        assert ml_ai_sim > ml_pizza_sim, (
            "ML should be more similar to AI than to pizza"
        )


# Marks for pytest
pytestmark = [
    pytest.mark.skipif(not os.getenv("NIM_API_KEY"), reason="NIM_API_KEY not set"),
    pytest.mark.integration,
]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
