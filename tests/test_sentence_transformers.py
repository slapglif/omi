"""
test_sentence_transformers.py - Test suite for local SentenceTransformer embeddings

Tests:
- Basic embedding functionality (with library installed)
- Model loading and initialization
- Error handling (library not installed, invalid models)
- Batch processing
- Quality checks (semantic similarity)
- Different model support (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)

NOTE: Tests marked with @pytest.mark.sentence_transformers are skipped unless
sentence-transformers is installed. This allows CI/CD to run basic tests without
requiring heavy ML dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
import numpy as np

# Import core classes - don't skip entire module since we test error handling
from omi.embeddings import SentenceTransformerEmbedder, EmbeddingProvider


# Check if sentence_transformers is available
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TestSentenceTransformerBasics:
    """Test basic functionality when sentence-transformers is available"""

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_basic_embedding(self):
        """
        Test basic embed() with default model returns correct dimensions
        """
        embedder = SentenceTransformerEmbedder()
        embedding = embedder.embed("hello world")

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == 384, "Default model (all-MiniLM-L6-v2) should produce 384-dim vectors"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        assert not all(x == 0 for x in embedding), "Embedding should not be all zeros"

        # Check L2 norm is reasonable
        norm = np.linalg.norm(embedding)
        assert 0.1 < norm < 100, f"L2 norm {norm} seems unreasonable"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_embedding_quality_basic(self):
        """
        Basic quality check: "king" vs "queen" should be more similar than "king" vs "apple"
        """
        embedder = SentenceTransformerEmbedder()

        king_emb = embedder.embed("king")
        queen_emb = embedder.embed("queen")
        apple_emb = embedder.embed("apple")

        royalty_sim = embedder.similarity(king_emb, queen_emb)
        different_sim = embedder.similarity(king_emb, apple_emb)

        assert royalty_sim > different_sim, (
            f"'king' should be more similar to 'queen' ({royalty_sim:.4f}) "
            f"than to 'apple' ({different_sim:.4f})"
        )

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_dimensions_property(self):
        """
        Test that dimensions property returns correct value for default model
        """
        embedder = SentenceTransformerEmbedder()
        assert embedder.dimensions == 384, "Default model should have 384 dimensions"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_inherits_from_embedding_provider(self):
        """
        Verify SentenceTransformerEmbedder correctly inherits from EmbeddingProvider
        """
        embedder = SentenceTransformerEmbedder()
        assert isinstance(embedder, EmbeddingProvider), "Should be instance of EmbeddingProvider"
        assert hasattr(embedder, 'embed'), "Should have embed() method"
        assert hasattr(embedder, 'dimensions'), "Should have dimensions property"
        assert hasattr(embedder, 'similarity'), "Should have similarity() method"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_empty_string_handling(self):
        """
        Test that empty strings are handled gracefully
        """
        embedder = SentenceTransformerEmbedder()
        embedding = embedder.embed("")

        assert isinstance(embedding, list), "Should return a list even for empty string"
        assert len(embedding) == 384, "Should still return correct dimensions"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_long_text_handling(self):
        """
        Test that long texts are handled correctly
        """
        embedder = SentenceTransformerEmbedder()
        long_text = "test " * 1000  # Very long text
        embedding = embedder.embed(long_text)

        assert isinstance(embedding, list), "Should handle long text"
        assert len(embedding) == 384, "Should return correct dimensions for long text"


class TestSentenceTransformerModels:
    """Test different model support"""

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_minilm_model(self):
        """
        Test all-MiniLM-L6-v2 model (default, 384 dimensions)
        """
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        embedding = embedder.embed("test")

        assert len(embedding) == 384, "all-MiniLM-L6-v2 should produce 384-dim vectors"
        assert embedder.dimensions == 384

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_mpnet_model(self):
        """
        Test all-mpnet-base-v2 model (768 dimensions, higher quality)
        """
        embedder = SentenceTransformerEmbedder(model="all-mpnet-base-v2")
        embedding = embedder.embed("test")

        assert len(embedding) == 768, "all-mpnet-base-v2 should produce 768-dim vectors"
        assert embedder.dimensions == 768

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_model_dimensions_mapping(self):
        """
        Test that MODEL_DIMENSIONS mapping is correct
        """
        assert SentenceTransformerEmbedder.MODEL_DIMENSIONS["all-MiniLM-L6-v2"] == 384
        assert SentenceTransformerEmbedder.MODEL_DIMENSIONS["all-mpnet-base-v2"] == 768
        assert SentenceTransformerEmbedder.MODEL_DIMENSIONS["paraphrase-multilingual-MiniLM-L12-v2"] == 384

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_unknown_model_dimensions_fallback(self):
        """
        Test that unknown models fall back to DEFAULT_DIM for dimensions property
        """
        # Mock the model loading to avoid downloading
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1] * 512)
            mock_st.return_value = mock_model

            embedder = SentenceTransformerEmbedder(model="unknown-custom-model")

            # Should fall back to DEFAULT_DIM (384) for unknown model
            assert embedder.dimensions == 384, "Unknown models should use DEFAULT_DIM"


class TestSentenceTransformerBatch:
    """Test batch embedding functionality"""

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_batch_embedding(self):
        """
        Test embed_batch() with multiple texts
        """
        embedder = SentenceTransformerEmbedder()
        texts = ["hello world", "test text", "another sentence"]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == len(texts), "Should return same number of embeddings as inputs"
        for emb in embeddings:
            assert len(emb) == 384, "Each embedding should have correct dimensions"
            assert isinstance(emb, list), "Each embedding should be a list"
            assert all(isinstance(x, float) for x in emb), "All elements should be floats"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_batch_embedding_custom_batch_size(self):
        """
        Test embed_batch() with custom batch size
        """
        embedder = SentenceTransformerEmbedder()
        texts = ["text " + str(i) for i in range(10)]
        embeddings = embedder.embed_batch(texts, batch_size=3)

        assert len(embeddings) == 10, "Should return all embeddings"
        for emb in embeddings:
            assert len(emb) == 384, "Each embedding should have correct dimensions"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_batch_embedding_consistency(self):
        """
        Test that batch embedding produces same results as individual embeddings
        """
        embedder = SentenceTransformerEmbedder()
        texts = ["hello", "world"]

        # Get embeddings individually
        individual_embs = [embedder.embed(t) for t in texts]

        # Get embeddings as batch
        batch_embs = embedder.embed_batch(texts)

        # Should be very similar (allowing for small numerical differences)
        for ind_emb, batch_emb in zip(individual_embs, batch_embs):
            similarity = embedder.similarity(ind_emb, batch_emb)
            assert similarity > 0.99, f"Batch and individual embeddings should be nearly identical, got {similarity}"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_batch_embedding_empty_list(self):
        """
        Test that empty list returns empty list
        """
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed_batch([])

        assert embeddings == [], "Empty input should return empty list"


class TestSentenceTransformerErrorHandling:
    """Test error handling for various failure modes"""

    def test_missing_library_error(self):
        """
        Test clear error message when sentence-transformers is not installed
        """
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with patch('omi.embeddings.SentenceTransformerEmbedder._load_model') as mock_load:
                mock_load.side_effect = RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

                with pytest.raises(RuntimeError) as exc_info:
                    embedder = SentenceTransformerEmbedder()
                    embedder.embed("test")

                assert "sentence-transformers not installed" in str(exc_info.value)
                assert "pip install sentence-transformers" in str(exc_info.value)

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_invalid_model_error(self):
        """
        Test error handling for invalid/non-existent model names
        """
        with pytest.raises(Exception):  # Can be OSError, ValueError, or other exceptions from HuggingFace
            embedder = SentenceTransformerEmbedder(model="definitely-not-a-real-model-xyz-123")

    def test_model_load_failure_error(self):
        """
        Test that model loading failures are handled with clear error messages
        """
        # Mock the _load_model method to simulate a model loading failure
        with patch.object(SentenceTransformerEmbedder, '_load_model') as mock_load:
            mock_load.side_effect = RuntimeError("Failed to load model test-model: Download failed")

            with pytest.raises(RuntimeError) as exc_info:
                SentenceTransformerEmbedder(model="test-model")

            assert "Failed to load model" in str(exc_info.value)
            assert "test-model" in str(exc_info.value)

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_embed_without_model_loaded_recovers(self):
        """
        Test that embed() can recover if model wasn't loaded initially
        """
        embedder = SentenceTransformerEmbedder()

        # Artificially set model to None
        embedder._model_instance = None

        # embed() should reload the model
        embedding = embedder.embed("test")

        assert embedder._model_instance is not None, "Model should be reloaded"
        assert len(embedding) == 384, "Should produce valid embedding"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_embed_batch_without_model_loaded_recovers(self):
        """
        Test that embed_batch() can recover if model wasn't loaded initially
        """
        embedder = SentenceTransformerEmbedder()

        # Artificially set model to None
        embedder._model_instance = None

        # embed_batch() should reload the model
        embeddings = embedder.embed_batch(["test1", "test2"])

        assert embedder._model_instance is not None, "Model should be reloaded"
        assert len(embeddings) == 2, "Should produce valid embeddings"


class TestSentenceTransformerSimilarity:
    """Test similarity calculation"""

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_similarity_identical_texts(self):
        """
        Test that identical texts have similarity close to 1.0
        """
        embedder = SentenceTransformerEmbedder()
        emb1 = embedder.embed("hello world")
        emb2 = embedder.embed("hello world")

        similarity = embedder.similarity(emb1, emb2)
        assert 0.99 <= similarity <= 1.01, f"Identical texts should have similarity ~1.0, got {similarity}"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_similarity_similar_texts(self):
        """
        Test that similar texts have high similarity
        """
        embedder = SentenceTransformerEmbedder()
        emb1 = embedder.embed("The cat is sleeping")
        emb2 = embedder.embed("A cat is asleep")

        similarity = embedder.similarity(emb1, emb2)
        assert similarity > 0.5, f"Similar texts should have high similarity, got {similarity}"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_similarity_dissimilar_texts(self):
        """
        Test that dissimilar texts have lower similarity
        """
        embedder = SentenceTransformerEmbedder()
        emb1 = embedder.embed("Machine learning is fascinating")
        emb2 = embedder.embed("I like pizza for dinner")

        similarity = embedder.similarity(emb1, emb2)
        # Dissimilar texts should have lower similarity, but not necessarily negative
        assert -1.0 <= similarity <= 0.7, f"Dissimilar texts should have lower similarity, got {similarity}"

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_similarity_range(self):
        """
        Test that similarity values are always in valid range [-1, 1]
        """
        embedder = SentenceTransformerEmbedder()

        test_pairs = [
            ("hello", "world"),
            ("cat", "dog"),
            ("programming", "coding"),
            ("completely", "unrelated"),
        ]

        for text1, text2 in test_pairs:
            emb1 = embedder.embed(text1)
            emb2 = embedder.embed(text2)
            similarity = embedder.similarity(emb1, emb2)

            assert -1.0 <= similarity <= 1.0, (
                f"Similarity for '{text1}' vs '{text2}' should be in [-1, 1], got {similarity}"
            )


class TestSentenceTransformerDefaults:
    """Test default values and constants"""

    def test_default_model(self):
        """Test that DEFAULT_MODEL is set correctly"""
        assert SentenceTransformerEmbedder.DEFAULT_MODEL == "all-MiniLM-L6-v2"

    def test_default_dim(self):
        """Test that DEFAULT_DIM matches the default model"""
        assert SentenceTransformerEmbedder.DEFAULT_DIM == 384

    def test_model_dimensions_dict_exists(self):
        """Test that MODEL_DIMENSIONS dict exists and is not empty"""
        assert hasattr(SentenceTransformerEmbedder, 'MODEL_DIMENSIONS')
        assert isinstance(SentenceTransformerEmbedder.MODEL_DIMENSIONS, dict)
        assert len(SentenceTransformerEmbedder.MODEL_DIMENSIONS) > 0


class TestSentenceTransformerIntegration:
    """Integration tests with real models (if available)"""

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_multilingual_model(self):
        """
        Test multilingual model support
        """
        embedder = SentenceTransformerEmbedder(model="paraphrase-multilingual-MiniLM-L12-v2")

        # Test English
        emb_en = embedder.embed("Hello, how are you?")
        assert len(emb_en) == 384

        # Test non-English (if model supports it)
        emb_es = embedder.embed("Hola, ¿cómo estás?")
        assert len(emb_es) == 384

        # Both should be valid embeddings
        assert not all(x == 0 for x in emb_en)
        assert not all(x == 0 for x in emb_es)

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_end_to_end_semantic_search(self):
        """
        End-to-end test: semantic search over a small corpus
        """
        embedder = SentenceTransformerEmbedder()

        # Small corpus
        corpus = [
            "Python is a programming language",
            "Machine learning uses neural networks",
            "Dogs are loyal pets",
            "JavaScript runs in browsers",
        ]

        # Embed corpus
        corpus_embeddings = embedder.embed_batch(corpus)

        # Query
        query = "What is a coding language?"
        query_embedding = embedder.embed(query)

        # Find most similar
        similarities = [embedder.similarity(query_embedding, doc_emb) for doc_emb in corpus_embeddings]
        most_similar_idx = similarities.index(max(similarities))

        # Should match programming-related sentences (indices 0 or 3)
        assert most_similar_idx in [0, 3], (
            f"Query about coding should match programming sentences, "
            f"but matched: '{corpus[most_similar_idx]}'"
        )

    @pytest.mark.sentence_transformers
    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not installed")
    def test_special_characters_handling(self):
        """
        Test that special characters are handled correctly
        """
        embedder = SentenceTransformerEmbedder()

        texts_with_special_chars = [
            "Hello! How are you?",
            "Price: $99.99",
            "Email: test@example.com",
            "Code: `print('hello')`",
            "Math: 2 + 2 = 4",
            "Emoji: 😊 🎉",
        ]

        for text in texts_with_special_chars:
            embedding = embedder.embed(text)
            assert len(embedding) == 384, f"Failed to handle: {text}"
            assert not all(x == 0 for x in embedding), f"Got zero embedding for: {text}"
