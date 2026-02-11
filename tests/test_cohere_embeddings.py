"""
test_cohere_embeddings.py - Full connection and error handling test suite for Cohere embeddings

Tests:
- Connection tests (with real API key)
- Model availability (embed-english-v3.0, embed-multilingual-v3.0)
- Error handling (rate limits, timeouts, invalid keys)
- Embedding quality and dimensions

NOTE: Tests with real COHERE_API_KEY are optional and only run if COHERE_API_KEY is set.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

# Skip entire module if dependencies unavailable
pytest.importorskip("requests")
pytest.importorskip("numpy")

from src.omi.embeddings import CohereEmbedder

# Constants
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
SKIP_COHERE_TESTS = not COHERE_API_KEY or COHERE_API_KEY == ""
REQUIRED_DIM = 1024


class TestCohereConnection:
    """Test Cohere connectivity with real API keys"""

    @pytest.mark.cohere
    def test_cohere_connection(self):
        """
        Test with real COHERE_API_KEY: embed("hello world") returns 1024-dim vector
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)
        embedding = embedder.embed("hello world")

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == REQUIRED_DIM, f"Expected {REQUIRED_DIM} dimensions, got {len(embedding)}"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        assert not all(x == 0 for x in embedding), "Embedding should not be all zeros"

        # Check L2 norm is reasonable (embeddings are typically normalized or near-normalized)
        norm = np.linalg.norm(embedding)
        assert 0.1 < norm < 100, f"L2 norm {norm} seems unreasonable"

    @pytest.mark.cohere
    def test_cohere_multilingual_model(self):
        """
        Test embed-multilingual-v3.0 model returns 1024-dim vector
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY, model="embed-multilingual-v3.0")
        embedding = embedder.embed("hello world")

        assert len(embedding) == REQUIRED_DIM, f"Expected {REQUIRED_DIM} dimensions, got {len(embedding)}"

        # Check L2 norm is reasonable
        norm = np.linalg.norm(embedding)
        assert 0.1 < norm < 100, f"L2 norm {norm} seems unreasonable"

    @pytest.mark.cohere
    def test_cohere_embedding_quality_basic(self):
        """
        Basic quality check: "king" vs "queen" should be more similar than "king" vs "apple"
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

        king_emb = embedder.embed("king")
        queen_emb = embedder.embed("queen")
        apple_emb = embedder.embed("apple")

        royalty_sim = embedder.similarity(king_emb, queen_emb)
        different_sim = embedder.similarity(king_emb, apple_emb)

        assert royalty_sim > different_sim, (
            f"'king' should be more similar to 'queen' ({royalty_sim:.4f}) "
            f"than to 'apple' ({different_sim:.4f})"
        )

    @pytest.mark.cohere
    def test_cohere_batch_embedding(self):
        """
        Test batch embedding works correctly
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

        texts = ["hello world", "test text", "another sentence"]
        embeddings = embedder.embed_batch(texts, batch_size=2)

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        for emb in embeddings:
            assert len(emb) == REQUIRED_DIM, "Each embedding should have correct dimension"

    @pytest.mark.cohere
    def test_cohere_input_type_search_query(self):
        """
        Test input_type parameter works correctly with search_query
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY, input_type="search_query")
        embedding = embedder.embed("what is machine learning?")

        assert len(embedding) == REQUIRED_DIM, "Should return correct dimension"


class TestCohereErrorHandling:
    """Test graceful error handling for various Cohere failure modes"""

    def test_cohere_invalid_key(self):
        """
        Test clear error message on invalid API key
        """
        import requests

        with pytest.raises(requests.HTTPError) as exc_info:
            embedder = CohereEmbedder(api_key="invalid_key_12345")
            embedder.embed("test")

        assert exc_info.value.response.status_code in [401, 403], "Should get auth error"

    def test_cohere_missing_key(self):
        """
        Test that missing API key raises ValueError
        """
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                CohereEmbedder(api_key=None)

            assert "COHERE_API_KEY" in str(exc_info.value), "Should mention COHERE_API_KEY in error"

    def test_cohere_timeout(self):
        """
        Test timeout handling: Should raise appropriate error
        """
        import requests

        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set - can't test timeout against real API")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

        # Mock timeout
        original_post = embedder._session.post

        def mock_timeout(*args, **kwargs):
            raise requests.Timeout("Request timed out")

        embedder._session.post = mock_timeout

        with pytest.raises(requests.Timeout):
            embedder.embed("test")

    def test_cohere_network_error_graceful(self):
        """
        Test graceful handling of network errors (DNS, connectivity, etc.)
        """
        import requests

        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set - can't test network errors")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

        # Simulate various network errors
        for error in [requests.ConnectionError, requests.ConnectTimeout, requests.RequestException]:
            original_post = embedder._session.post

            def mock_error(*args, **kwargs):
                raise error("Network error")

            embedder._session.post = mock_error

            with pytest.raises(error):
                embedder.embed("test")

            embedder._session.post = original_post


class TestCohereConfig:
    """Test Cohere configuration handling"""

    @pytest.mark.cohere
    def test_cohere_config_from_env(self):
        """
        Test that CohereEmbedder reads from environment variables
        """
        with patch.dict(os.environ, {"COHERE_API_KEY": "co-test123"}):
            with patch.object(CohereEmbedder, "_test_connection", return_value=None):
                embedder = CohereEmbedder()
                assert embedder.api_key == "co-test123"

    @pytest.mark.cohere
    def test_cohere_config_explicit(self):
        """
        Test that explicit API key overrides environment
        """
        with patch.dict(os.environ, {"COHERE_API_KEY": "env_key"}):
            with patch.object(CohereEmbedder, "_test_connection", return_value=None):
                embedder = CohereEmbedder(api_key="explicit_key")
                assert embedder.api_key == "explicit_key"

    def test_cohere_config_model(self):
        """
        Test model configuration
        """
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy", model="embed-multilingual-v3.0")
            assert embedder.model == "embed-multilingual-v3.0"
            assert embedder.dimensions == 1024  # Default dimension for multilingual model

    def test_cohere_config_input_type(self):
        """
        Test input_type configuration
        """
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy", input_type="classification")
            assert embedder.input_type == "classification"

    def test_cohere_config_base_url(self):
        """
        Test custom base_url configuration
        """
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            custom_url = "https://custom.cohere.ai/v1"
            embedder = CohereEmbedder(api_key="dummy", base_url=custom_url)
            assert embedder.base_url == custom_url


class TestCohereEmbedderClass:
    """Test CohereEmbedder class structure and constants"""

    def test_default_model_constant(self):
        """Verify default model constant"""
        assert CohereEmbedder.DEFAULT_MODEL == "embed-english-v3.0"
        assert CohereEmbedder.DEFAULT_DIM == 1024

    def test_model_dimensions_mapping(self):
        """Verify model dimensions mapping"""
        expected_dims = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }
        assert CohereEmbedder.MODEL_DIMENSIONS == expected_dims

    @pytest.mark.cohere
    def test_embedding_dim_matches_constant(self):
        """
        Verify actual embedding dimension matches constant
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)
        embedding = embedder.embed("test")

        assert len(embedding) == CohereEmbedder.DEFAULT_DIM


class TestCohereSimilarity:
    """Test similarity calculations"""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0"""
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy_key")

            v = [1.0, 0.0, 0.0]
            sim = embedder.similarity(v, v)

            assert abs(sim - 1.0) < 0.0001, f"Expected 1.0, got {sim}"

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0"""
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy_key")

            v1 = [1.0, 0.0, 0.0]
            v2 = [0.0, 1.0, 0.0]
            sim = embedder.similarity(v1, v2)

            assert abs(sim) < 0.0001, f"Expected 0.0, got {sim}"

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0"""
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy_key")

            v1 = [1.0, 0.0, 0.0]
            v2 = [-1.0, 0.0, 0.0]
            sim = embedder.similarity(v1, v2)

            assert abs(sim - (-1.0)) < 0.0001, f"Expected -1.0, got {sim}"

    @pytest.mark.cohere
    def test_cosine_similarity_real_embeddings(self):
        """
        Test similarity with real embeddings
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

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


class TestCohereBatchProcessing:
    """Test batch processing capabilities"""

    @pytest.mark.cohere
    def test_batch_size_limit(self):
        """
        Test that batch size is respected (Cohere supports up to 96 texts per batch)
        Skip if COHERE_API_KEY not set
        """
        if SKIP_COHERE_TESTS:
            pytest.skip("COHERE_API_KEY not set")

        embedder = CohereEmbedder(api_key=COHERE_API_KEY)

        # Create 100 texts (more than batch size)
        texts = [f"text {i}" for i in range(100)]
        embeddings = embedder.embed_batch(texts, batch_size=50)

        assert len(embeddings) == 100, "Should return all embeddings"
        for emb in embeddings:
            assert len(emb) == REQUIRED_DIM, "Each embedding should have correct dimension"

    def test_batch_empty_list(self):
        """
        Test batch embedding with empty list
        """
        with patch.object(CohereEmbedder, "_test_connection", return_value=None):
            embedder = CohereEmbedder(api_key="dummy_key")

            embeddings = embedder.embed_batch([])
            assert embeddings == [], "Empty list should return empty list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
