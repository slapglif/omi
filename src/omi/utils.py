"""
Shared utility functions for OMI
Pattern: Centralized utilities to avoid duplication
"""

from typing import List, Union
from pathlib import Path
import hashlib
import numpy as np


def cosine_similarity(embedding1: Union[List[float], np.ndarray],
                     embedding2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical), with 0 meaning orthogonal.

    Args:
        embedding1: First vector (list or numpy array)
        embedding2: Second vector (list or numpy array)

    Returns:
        Cosine similarity score (float)

    Examples:
        >>> cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        1.0
        >>> cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        0.0
        >>> cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        -1.0
    """
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def hash_file(file_path: Path) -> str:
    """
    Generate SHA-256 hash of file contents.

    Args:
        file_path: Path to file to hash

    Returns:
        SHA-256 hash as hexadecimal string (64 characters)

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        >>> _ = f.write('test')
        >>> f.close()
        >>> h = hash_file(Path(f.name))
        >>> len(h)
        64
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_content(content: str) -> str:
    """
    Generate SHA-256 hash of string content.

    Args:
        content: String content to hash

    Returns:
        SHA-256 hash as hexadecimal string (64 characters)

    Examples:
        >>> h = hash_content('test')
        >>> len(h)
        64
        >>> hash_content('hello') == hash_content('hello')
        True
    """
    sha256 = hashlib.sha256()
    sha256.update(content.encode('utf-8'))
    return sha256.hexdigest()
