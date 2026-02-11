"""
test_embedding_providers.py - Test suite for abstract interface and factory

Tests:
- Abstract base class (EmbeddingProvider) interface compliance
- Provider registry completeness
- Factory pattern and provider instantiation
- Inheritance verification for all providers

NOTE: These are unit tests for the abstract interface and factory.
Provider-specific integration tests are in separate files.
"""

import pytest
from typing import List
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

# Import core classes
from omi.embeddings import (
    EmbeddingProvider,
    NIMEmbedder,
    OpenAIEmbedder,
    CohereEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    EmbeddingProviderFactory,
    PROVIDER_REGISTRY
)


class TestEmbeddingProviderInterface:
    """Test the abstract EmbeddingProvider interface"""

    def test_embedding_provider_is_abstract(self):
        """
        Verify EmbeddingProvider is an abstract base class
        """
        assert issubclass(EmbeddingProvider, ABC), "EmbeddingProvider should be an ABC"

        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            EmbeddingProvider()

    def test_embedding_provider_has_required_methods(self):
        """
        Verify EmbeddingProvider defines required abstract methods
        """
        # Check that embed() and dimensions are abstract
        assert hasattr(EmbeddingProvider, 'embed'), "Should have embed() method"
        assert hasattr(EmbeddingProvider, 'dimensions'), "Should have dimensions property"
        assert hasattr(EmbeddingProvider, 'similarity'), "Should have similarity() method"

    def test_similarity_is_concrete(self):
        """
        Verify similarity() is a concrete method with default implementation
        """
        # Create a minimal concrete implementation
        class MinimalProvider(EmbeddingProvider):
            def embed(self, text: str) -> List[float]:
                return [1.0, 0.0, 0.0]

            @property
            def dimensions(self) -> int:
                return 3

        provider = MinimalProvider()

        # Test that similarity() works without override
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity = provider.similarity(emb1, emb2)

        assert isinstance(similarity, float), "Similarity should return float"
        assert 0.99 <= similarity <= 1.01, "Identical vectors should have similarity ~1.0"

    def test_similarity_cosine_calculation(self):
        """
        Test that similarity() correctly calculates cosine similarity
        """
        class MinimalProvider(EmbeddingProvider):
            def embed(self, text: str) -> List[float]:
                return [1.0, 0.0, 0.0]

            @property
            def dimensions(self) -> int:
                return 3

        provider = MinimalProvider()

        # Orthogonal vectors should have 0 similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = provider.similarity(emb1, emb2)
        assert -0.01 <= similarity <= 0.01, "Orthogonal vectors should have ~0 similarity"

        # Opposite vectors should have -1 similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [-1.0, 0.0, 0.0]
        similarity = provider.similarity(emb1, emb2)
        assert -1.01 <= similarity <= -0.99, "Opposite vectors should have ~-1 similarity"

    def test_similarity_handles_zero_vectors(self):
        """
        Test that similarity() handles zero vectors gracefully
        """
        class MinimalProvider(EmbeddingProvider):
            def embed(self, text: str) -> List[float]:
                return [1.0, 0.0, 0.0]

            @property
            def dimensions(self) -> int:
                return 3

        provider = MinimalProvider()

        # Zero vector should return 0 similarity
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 0.0, 0.0]
        similarity = provider.similarity(emb1, emb2)
        assert similarity == 0.0, "Zero vector should have 0 similarity"


class TestProviderInheritance:
    """Test that all providers correctly inherit from EmbeddingProvider"""

    def test_nim_embedder_inheritance(self):
        """Verify NIMEmbedder inherits from EmbeddingProvider"""
        assert issubclass(NIMEmbedder, EmbeddingProvider), "NIMEmbedder should inherit from EmbeddingProvider"
        assert hasattr(NIMEmbedder, 'embed'), "NIMEmbedder should have embed() method"
        assert hasattr(NIMEmbedder, 'dimensions'), "NIMEmbedder should have dimensions property"

    def test_openai_embedder_inheritance(self):
        """Verify OpenAIEmbedder inherits from EmbeddingProvider"""
        assert issubclass(OpenAIEmbedder, EmbeddingProvider), "OpenAIEmbedder should inherit from EmbeddingProvider"
        assert hasattr(OpenAIEmbedder, 'embed'), "OpenAIEmbedder should have embed() method"
        assert hasattr(OpenAIEmbedder, 'dimensions'), "OpenAIEmbedder should have dimensions property"

    def test_cohere_embedder_inheritance(self):
        """Verify CohereEmbedder inherits from EmbeddingProvider"""
        assert issubclass(CohereEmbedder, EmbeddingProvider), "CohereEmbedder should inherit from EmbeddingProvider"
        assert hasattr(CohereEmbedder, 'embed'), "CohereEmbedder should have embed() method"
        assert hasattr(CohereEmbedder, 'dimensions'), "CohereEmbedder should have dimensions property"

    def test_ollama_embedder_inheritance(self):
        """Verify OllamaEmbedder inherits from EmbeddingProvider"""
        assert issubclass(OllamaEmbedder, EmbeddingProvider), "OllamaEmbedder should inherit from EmbeddingProvider"
        assert hasattr(OllamaEmbedder, 'embed'), "OllamaEmbedder should have embed() method"
        assert hasattr(OllamaEmbedder, 'dimensions'), "OllamaEmbedder should have dimensions property"

    def test_sentence_transformer_embedder_inheritance(self):
        """Verify SentenceTransformerEmbedder inherits from EmbeddingProvider"""
        assert issubclass(SentenceTransformerEmbedder, EmbeddingProvider), (
            "SentenceTransformerEmbedder should inherit from EmbeddingProvider"
        )
        assert hasattr(SentenceTransformerEmbedder, 'embed'), "SentenceTransformerEmbedder should have embed() method"
        assert hasattr(SentenceTransformerEmbedder, 'dimensions'), (
            "SentenceTransformerEmbedder should have dimensions property"
        )


class TestProviderRegistry:
    """Test the PROVIDER_REGISTRY completeness and correctness"""

    def test_registry_exists(self):
        """Verify PROVIDER_REGISTRY exists and is a dict"""
        assert isinstance(PROVIDER_REGISTRY, dict), "PROVIDER_REGISTRY should be a dict"
        assert len(PROVIDER_REGISTRY) > 0, "PROVIDER_REGISTRY should not be empty"

    def test_registry_contains_all_providers(self):
        """Verify PROVIDER_REGISTRY contains all provider types"""
        expected_providers = {
            "nim": NIMEmbedder,
            "openai": OpenAIEmbedder,
            "cohere": CohereEmbedder,
            "ollama": OllamaEmbedder,
            "sentence_transformers": SentenceTransformerEmbedder,
        }

        for provider_name, provider_class in expected_providers.items():
            assert provider_name in PROVIDER_REGISTRY, f"{provider_name} should be in PROVIDER_REGISTRY"
            assert PROVIDER_REGISTRY[provider_name] == provider_class, (
                f"PROVIDER_REGISTRY[{provider_name}] should map to {provider_class.__name__}"
            )

    def test_registry_supports_hyphenated_names(self):
        """Verify PROVIDER_REGISTRY supports both underscore and hyphen naming"""
        # sentence-transformers should work with both _ and -
        assert "sentence_transformers" in PROVIDER_REGISTRY, "Should support sentence_transformers"
        assert "sentence-transformers" in PROVIDER_REGISTRY, "Should support sentence-transformers"
        assert (
            PROVIDER_REGISTRY["sentence_transformers"] == PROVIDER_REGISTRY["sentence-transformers"]
        ), "Both names should map to same class"

    def test_all_registry_values_are_providers(self):
        """Verify all values in PROVIDER_REGISTRY are EmbeddingProvider subclasses"""
        for provider_name, provider_class in PROVIDER_REGISTRY.items():
            assert issubclass(provider_class, EmbeddingProvider), (
                f"{provider_name} should map to an EmbeddingProvider subclass"
            )


class TestEmbeddingProviderFactory:
    """Test the EmbeddingProviderFactory"""

    def test_factory_initialization(self):
        """Test factory can be initialized with and without default config"""
        # Without default config
        factory = EmbeddingProviderFactory()
        assert factory.default_config == {}, "Default config should be empty dict"

        # With default config
        default_config = {"provider": "ollama", "model": "nomic-embed-text"}
        factory = EmbeddingProviderFactory(default_config=default_config)
        assert factory.default_config == default_config, "Default config should be stored"

    def test_factory_has_provider_map(self):
        """Verify factory has PROVIDER_MAP with all providers"""
        factory = EmbeddingProviderFactory()
        assert hasattr(factory, 'PROVIDER_MAP'), "Factory should have PROVIDER_MAP"
        assert len(factory.PROVIDER_MAP) > 0, "PROVIDER_MAP should not be empty"

        # Should contain all main providers
        assert "nim" in factory.PROVIDER_MAP
        assert "openai" in factory.PROVIDER_MAP
        assert "cohere" in factory.PROVIDER_MAP
        assert "ollama" in factory.PROVIDER_MAP
        assert "sentence-transformers" in factory.PROVIDER_MAP

    def test_factory_default_provider(self):
        """Verify factory has a default provider"""
        factory = EmbeddingProviderFactory()
        assert hasattr(factory, 'DEFAULT_PROVIDER'), "Factory should have DEFAULT_PROVIDER"
        assert factory.DEFAULT_PROVIDER in factory.PROVIDER_MAP, "Default provider should be in PROVIDER_MAP"

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_creates_ollama_provider(self, mock_init):
        """Test factory creates OllamaEmbedder from config"""
        mock_init.return_value = None

        factory = EmbeddingProviderFactory()
        config = {"provider": "ollama", "model": "nomic-embed-text"}

        provider = factory.get_provider(config)

        # Verify correct provider type
        assert isinstance(provider, OllamaEmbedder), "Should create OllamaEmbedder"

        # Verify __init__ was called with correct params
        mock_init.assert_called_once_with(model="nomic-embed-text")

    @patch('omi.embeddings.SentenceTransformerEmbedder.__init__')
    def test_factory_creates_sentence_transformer_provider(self, mock_init):
        """Test factory creates SentenceTransformerEmbedder from config"""
        mock_init.return_value = None

        factory = EmbeddingProviderFactory()
        config = {"provider": "sentence-transformers", "model": "all-MiniLM-L6-v2"}

        provider = factory.get_provider(config)

        # Verify correct provider type
        assert isinstance(provider, SentenceTransformerEmbedder), (
            "Should create SentenceTransformerEmbedder"
        )

        # Verify __init__ was called with correct params
        mock_init.assert_called_once_with(model="all-MiniLM-L6-v2")

    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises ValueError for unknown provider type"""
        factory = EmbeddingProviderFactory()
        config = {"provider": "unknown_provider_xyz"}

        with pytest.raises(ValueError) as exc_info:
            factory.get_provider(config)

        assert "Unknown provider type" in str(exc_info.value), "Should mention unknown provider"
        assert "unknown_provider_xyz" in str(exc_info.value), "Should mention the specific provider"
        assert "Available providers" in str(exc_info.value), "Should list available providers"

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_merges_default_config(self, mock_init):
        """Test factory merges default config with provided config"""
        mock_init.return_value = None

        default_config = {"provider": "ollama", "model": "default-model"}
        factory = EmbeddingProviderFactory(default_config=default_config)

        # Override just the model
        config = {"model": "custom-model"}

        provider = factory.get_provider(config)

        # Should use provider from default but model from config
        assert isinstance(provider, OllamaEmbedder), "Should use default provider (ollama)"
        mock_init.assert_called_once_with(model="custom-model")

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_provided_config_overrides_default(self, mock_init):
        """Test provided config overrides default config"""
        mock_init.return_value = None

        default_config = {"provider": "nim", "model": "default-model"}
        factory = EmbeddingProviderFactory(default_config=default_config)

        # Override provider and model
        config = {"provider": "ollama", "model": "custom-model"}

        provider = factory.get_provider(config)

        # Should use everything from provided config
        assert isinstance(provider, OllamaEmbedder), "Should use provided provider (ollama)"
        mock_init.assert_called_once_with(model="custom-model")

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_filters_provider_key_from_params(self, mock_init):
        """Test factory doesn't pass 'provider' key to provider __init__"""
        mock_init.return_value = None

        factory = EmbeddingProviderFactory()
        config = {"provider": "ollama", "model": "test-model", "base_url": "http://localhost:11434"}

        factory.get_provider(config)

        # 'provider' should not be in the call args
        call_kwargs = mock_init.call_args[1]
        assert "provider" not in call_kwargs, "'provider' should be filtered from params"
        assert "model" in call_kwargs, "'model' should be passed"
        assert "base_url" in call_kwargs, "'base_url' should be passed"

    def test_factory_handles_initialization_errors(self):
        """Test factory handles provider initialization errors gracefully"""
        factory = EmbeddingProviderFactory()

        # Try to create NIM provider without API key (will fail)
        config = {"provider": "nim"}

        with pytest.raises(RuntimeError) as exc_info:
            factory.get_provider(config)

        assert "Failed to initialize" in str(exc_info.value), "Should mention initialization failure"
        assert "nim" in str(exc_info.value), "Should mention the provider that failed"

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_get_provider_without_config(self, mock_init):
        """Test factory can create provider with None config (uses defaults)"""
        mock_init.return_value = None

        # Set up factory with default config
        default_config = {"provider": "ollama", "model": "default-model"}
        factory = EmbeddingProviderFactory(default_config=default_config)

        # Call without config
        provider = factory.get_provider()

        assert isinstance(provider, OllamaEmbedder), "Should create provider from default config"

    @patch('omi.embeddings.OllamaEmbedder.__init__')
    def test_factory_uses_default_provider_when_not_specified(self, mock_init):
        """Test factory uses DEFAULT_PROVIDER when provider not in config"""
        mock_init.return_value = None

        factory = EmbeddingProviderFactory()

        # If DEFAULT_PROVIDER is not 'ollama', mock the correct one
        if factory.DEFAULT_PROVIDER == "ollama":
            config = {"model": "test-model"}
            provider = factory.get_provider(config)
            assert isinstance(provider, OllamaEmbedder), "Should use default provider"
        else:
            # Just verify that get_provider works without 'provider' key
            # The actual provider will depend on DEFAULT_PROVIDER
            pytest.skip("DEFAULT_PROVIDER is not 'ollama', skipping this specific test")


class TestProviderDimensionConsistency:
    """Test that provider dimensions are correctly defined"""

    def test_nim_embedder_dimensions(self):
        """Verify NIMEmbedder has correct default dimensions"""
        assert hasattr(NIMEmbedder, 'DEFAULT_DIM'), "NIMEmbedder should have DEFAULT_DIM"
        assert NIMEmbedder.DEFAULT_DIM == 1024, "NIM default should be 1024 (baai/bge-m3)"

    def test_openai_embedder_dimensions(self):
        """Verify OpenAIEmbedder has correct model dimensions"""
        assert hasattr(OpenAIEmbedder, 'MODEL_DIMENSIONS'), "OpenAIEmbedder should have MODEL_DIMENSIONS"
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-small"] == 1536
        assert OpenAIEmbedder.MODEL_DIMENSIONS["text-embedding-3-large"] == 3072

    def test_cohere_embedder_dimensions(self):
        """Verify CohereEmbedder has correct model dimensions"""
        assert hasattr(CohereEmbedder, 'MODEL_DIMENSIONS'), "CohereEmbedder should have MODEL_DIMENSIONS"
        assert CohereEmbedder.MODEL_DIMENSIONS["embed-english-v3.0"] == 1024
        assert CohereEmbedder.MODEL_DIMENSIONS["embed-multilingual-v3.0"] == 1024

    def test_ollama_embedder_dimensions(self):
        """Verify OllamaEmbedder has correct model dimensions"""
        assert hasattr(OllamaEmbedder, 'MODEL_DIMENSIONS'), "OllamaEmbedder should have MODEL_DIMENSIONS"
        assert OllamaEmbedder.MODEL_DIMENSIONS["nomic-embed-text"] == 768
        assert OllamaEmbedder.MODEL_DIMENSIONS["mxbai-embed-large"] == 1024

    def test_sentence_transformer_embedder_dimensions(self):
        """Verify SentenceTransformerEmbedder has correct model dimensions"""
        assert hasattr(SentenceTransformerEmbedder, 'MODEL_DIMENSIONS'), (
            "SentenceTransformerEmbedder should have MODEL_DIMENSIONS"
        )
        assert SentenceTransformerEmbedder.MODEL_DIMENSIONS["all-MiniLM-L6-v2"] == 384
        assert SentenceTransformerEmbedder.MODEL_DIMENSIONS["all-mpnet-base-v2"] == 768
