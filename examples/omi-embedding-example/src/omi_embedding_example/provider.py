"""
DummyEmbedder - Example embedding provider for OMI

This is a demonstration plugin showing how to implement a custom embedding provider.
It generates deterministic embeddings based on text hashing - NOT suitable for production use.

For production embedding providers, see NIMEmbedder or OllamaEmbedder in omi.embeddings.
"""

import hashlib
from typing import List
import numpy as np

from omi.embeddings import EmbeddingProvider


class DummyEmbedder(EmbeddingProvider):
    """
    Dummy embedding provider for testing and demonstration

    Generates deterministic 128-dimensional embeddings based on text hashing.
    Each unique input text produces a consistent embedding vector.

    This is NOT suitable for production - use for testing plugin architecture only.

    Example:
        >>> embedder = DummyEmbedder()
        >>> vec = embedder.embed("hello world")
        >>> len(vec)
        128
        >>> embedder.embed("hello world") == embedder.embed("hello world")
        True
    """

    interface_version = "1.0"

    def __init__(self, dimension: int = 128):
        """
        Initialize DummyEmbedder

        Args:
            dimension: Embedding vector dimension (default: 128)
        """
        self.dimension = dimension
        self.model_name = f"dummy-hash-{dimension}d"

    def embed(self, text: str) -> List[float]:
        """
        Generate deterministic embedding for a single text

        Uses SHA-256 hash to generate a reproducible embedding vector.
        The hash is expanded to the desired dimension using numpy's random
        generator with the hash as seed.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats (normalized to unit length)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate deterministic seed from text hash
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()
        seed = int.from_bytes(text_hash[:4], byteorder='big')

        # Generate deterministic random vector
        rng = np.random.RandomState(seed)
        vector = rng.randn(self.dimension).astype(np.float32)

        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch (ignored for this simple implementation)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        return [self.embed(text) for text in texts]

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (range -1.0 to 1.0)

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimension mismatch: {len(embedding1)} != {len(embedding2)}"
            )

        # Convert to numpy arrays
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider

        Returns:
            Number of dimensions in embedding vectors
        """
        return self.dimension

    def get_model_name(self) -> str:
        """
        Get the name/identifier of the embedding model

        Returns:
            Model name string
        """
        return self.model_name
