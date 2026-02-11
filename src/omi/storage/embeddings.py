"""
Embedding Utilities for OMI Storage
Binary serialization and similarity calculations for vector embeddings.
"""

import struct
import numpy as np
from typing import List


def embed_to_blob(embedding: List[float]) -> bytes:
    """
    Convert embedding list to binary blob (float32).

    Args:
        embedding: List of floats representing the embedding vector

    Returns:
        Binary blob representation (packed float32 values)
    """
    return struct.pack(f'{len(embedding)}f', *embedding)


def blob_to_embed(blob: bytes) -> List[float]:
    """
    Convert binary blob to embedding list (float32).

    Args:
        blob: Binary blob containing packed float32 values

    Returns:
        List of floats representing the embedding vector
    """
    if not blob:
        return []
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if not v1 or not v2:
        return 0.0
    arr1 = np.array(v1, dtype=np.float32)
    arr2 = np.array(v2, dtype=np.float32)
    dot = np.dot(arr1, arr2)
    norm = np.linalg.norm(arr1) * np.linalg.norm(arr2)
    return float(dot / norm) if norm > 0 else 0.0
