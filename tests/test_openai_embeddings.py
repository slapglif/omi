"""
test_openai_embeddings.py - Full connection and error handling test suite for OpenAI embeddings

Tests:
- Connection tests (with real API key)
- Model availability (text-embedding-3-small, text-embedding-3-large)
- Error handling (rate limits, timeouts, invalid keys)
- Embedding quality and dimensions

NOTE: Tests with real OPENAI_API_KEY are optional and only run if OPENAI_API_KEY is set.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

# Skip entire module if dependencies unavailable
pytest.importorskip("requests")
pytest.importorskip("numpy")

from src.omi.embeddings import OpenAIEmbedder, OpenAIConfig

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SKIP_OPENAI_TESTS = not OPENAI_API_KEY or OPENAI_API_KEY == ""
SMALL_MODEL_DIM = 1536
LARGE_MODEL_DIM = 3072


class TestOpenAIConnection:
    """Test OpenAI connectivity with real API keys"""

    @pytest.mark.openai
    def test_openai_connection(self):
        """
        Test with real OPENAI_API_KEY: embed("hello world") returns 1536-dim vector
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
        embedding = embedder.embed("hello world")

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == SMALL_MODEL_DIM, f"Expected {SMALL_MODEL_DIM} dimensions, got {len(embedding)}"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        assert not all(x == 0 for x in embedding), "Embedding should not be all zeros"

        # Check L2 norm is reasonable (OpenAI embeddings are normalized)
        norm = np.linalg.norm(embedding)
        assert 0.9 < norm < 1.1, f"L2 norm {norm} should be close to 1.0 (normalized)"

    @pytest.mark.openai
    def test_openai_large_model(self):
        """
        Test text-embedding-3-large model returns 3072-dim vector
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY, model="text-embedding-3-large")
        embedding = embedder.embed("test text")

        assert len(embedding) == LARGE_MODEL_DIM, f"Expected {LARGE_MODEL_DIM} dimensions, got {len(embedding)}"

        # Check L2 norm is reasonable
        norm = np.linalg.norm(embedding)
        assert 0.9 < norm < 1.1, f"L2 norm {norm} should be close to 1.0 (normalized)"

    @pytest.mark.openai
    def test_openai_embedding_quality_basic(self):
        """
        Basic quality check: "king" vs "queen" should be more similar than "king" vs "apple"
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)

        king_emb = embedder.embed("king")
        queen_emb = embedder.embed("queen")
        apple_emb = embedder.embed("apple")

        royalty_sim = embedder.similarity(king_emb, queen_emb)
        different_sim = embedder.similarity(king_emb, apple_emb)

        assert royalty_sim > different_sim, (
            f"'king' should be more similar to 'queen' ({royalty_sim:.4f}) "
            f"than to 'apple' ({different_sim:.4f})"
        )

    @pytest.mark.openai
    def test_openai_batch_embedding(self):
        """
        Test batch embedding works correctly
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)

        texts = ["hello world", "test text", "another sentence"]
        embeddings = embedder.embed_batch(texts, batch_size=2)

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        for emb in embeddings:
            assert len(emb) == SMALL_MODEL_DIM, "Each embedding should have correct dimension"

    @pytest.mark.openai
    def test_openai_dimensions_property(self):
        """
        Test dimensions property returns correct values for different models
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder_small = OpenAIEmbedder(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        assert embedder_small.dimensions == SMALL_MODEL_DIM

        embedder_large = OpenAIEmbedder(api_key=OPENAI_API_KEY, model="text-embedding-3-large")
        assert embedder_large.dimensions == LARGE_MODEL_DIM


class TestOpenAIErrorHandling:
    """Test graceful error handling for various OpenAI failure modes"""

    def test_openai_missing_api_key(self):
        """
        Test clear error message when API key is not provided
        """
        with patch.dict(os.environ, {}, clear=True):
            # Mock _test_connection to avoid actual API call
            with patch.object(OpenAIEmbedder, '_test_connection'):
                with pytest.raises(ValueError) as exc_info:
                    OpenAIEmbedder(api_key=None)

                assert "OPENAI_API_KEY required" in str(exc_info.value)

    @pytest.mark.openai
    def test_openai_invalid_key(self):
        """
        Test clear error message on invalid API key
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set - can't test invalid key against real API")

        import requests

        with pytest.raises(requests.HTTPError) as exc_info:
            embedder = OpenAIEmbedder(api_key="sk-invalid_key_12345")
            embedder.embed("test")

        assert exc_info.value.response.status_code in [401, 403], "Should get auth error"

    def test_openai_rate_limit_handling(self):
        """
        Test graceful handling of rate limit (429) responses
        """
        import requests

        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            # Mock a 429 response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)

            with patch.object(embedder._session, "post", return_value=mock_response):
                with pytest.raises(requests.HTTPError):
                    embedder.embed("test")

    def test_openai_timeout(self):
        """
        Test timeout handling: Should raise appropriate error
        """
        import requests

        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            # Mock timeout
            with patch.object(embedder._session, "post", side_effect=requests.Timeout("Request timed out")):
                with pytest.raises(requests.Timeout):
                    embedder.embed("test")

    def test_openai_network_error_graceful(self):
        """
        Test graceful handling of network errors (DNS, connectivity, etc.)
        """
        import requests

        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            # Simulate various network errors
            for error in [requests.ConnectionError, requests.ConnectTimeout, requests.RequestException]:
                with patch.object(embedder._session, "post", side_effect=error("Network error")):
                    with pytest.raises(error):
                        embedder.embed("test")


class TestOpenAIConfig:
    """Test OpenAI configuration handling"""

    def test_openai_config_from_env(self):
        """
        Test that OpenAIEmbedder reads from environment variables
        """
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            # Mock _test_connection to avoid actual API call during init
            with patch.object(OpenAIEmbedder, '_test_connection'):
                embedder = OpenAIEmbedder()
                assert embedder.api_key == "sk-test123"

    def test_openai_config_explicit(self):
        """
        Test that explicit API key overrides environment
        """
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env_key"}):
            # Mock _test_connection to avoid actual API call during init
            with patch.object(OpenAIEmbedder, '_test_connection'):
                embedder = OpenAIEmbedder(api_key="sk-explicit_key")
                assert embedder.api_key == "sk-explicit_key"

    def test_openai_config_model(self):
        """
        Test model configuration
        """
        config = OpenAIConfig(api_key="sk-dummy", model="text-embedding-3-large")
        assert config.model == "text-embedding-3-large"
        assert config.embedding_dim == 1536  # Default dimension in config

    def test_openai_config_base_url(self):
        """
        Test custom base URL configuration
        """
        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(
                api_key="sk-dummy",
                base_url="https://api.custom.com/v1/"
            )
            assert embedder.base_url == "https://api.custom.com/v1"  # Trailing slash stripped


class TestOpenAIEmbedderClass:
    """Test OpenAIEmbedder class structure and constants"""

    def test_default_model_constant(self):
        """Verify default model constant"""
        assert OpenAIEmbedder.DEFAULT_MODEL == "text-embedding-3-small"
        assert OpenAIEmbedder.DEFAULT_DIM == 1536

    def test_model_dimensions_mapping(self):
        """Verify model dimensions mapping"""
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-small"] == 1536
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-large"] == 3072

    @pytest.mark.openai
    def test_embedding_dim_matches_constant(self):
        """
        Verify actual embedding dimension matches constant
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
        embedding = embedder.embed("test")

        assert len(embedding) == OpenAIEmbedder.DEFAULT_DIM


class TestOpenAISimilarity:
    """Test similarity calculations"""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0"""
        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            v = [1.0, 0.0, 0.0]
            sim = embedder.similarity(v, v)

            assert abs(sim - 1.0) < 0.0001, f"Expected 1.0, got {sim}"

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0"""
        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            v1 = [1.0, 0.0, 0.0]
            v2 = [0.0, 1.0, 0.0]
            sim = embedder.similarity(v1, v2)

            assert abs(sim) < 0.0001, f"Expected 0.0, got {sim}"

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0"""
        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            v1 = [1.0, 0.0, 0.0]
            v2 = [-1.0, 0.0, 0.0]
            sim = embedder.similarity(v1, v2)

            assert abs(sim - (-1.0)) < 0.0001, f"Expected -1.0, got {sim}"

    @pytest.mark.openai
    def test_cosine_similarity_real_embeddings(self):
        """
        Test similarity with real embeddings
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)

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


class TestOpenAIBatchProcessing:
    """Test batch embedding functionality"""

    @pytest.mark.openai
    def test_large_batch_embedding(self):
        """
        Test batch embedding with large batch size
        Skip if OPENAI_API_KEY not set
        """
        if SKIP_OPENAI_TESTS:
            pytest.skip("OPENAI_API_KEY not set")

        embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)

        # Create a larger batch
        texts = [f"test text {i}" for i in range(10)]
        embeddings = embedder.embed_batch(texts, batch_size=5)

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        for emb in embeddings:
            assert len(emb) == SMALL_MODEL_DIM, "Each embedding should have correct dimension"

    def test_batch_empty_list(self):
        """
        Test batch embedding with empty list
        """
        # Mock _test_connection to avoid actual API call during init
        with patch.object(OpenAIEmbedder, '_test_connection'):
            embedder = OpenAIEmbedder(api_key="sk-dummy_key")

            with patch.object(embedder._session, "post") as mock_post:
                embeddings = embedder.embed_batch([], batch_size=10)
                assert embeddings == [], "Empty input should return empty list"
                mock_post.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
