"""
Async Embeddings via NVIDIA NIM (baai/bge-m3)
Pattern: Cloud API with local caching, async version using httpx
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Union
from io import BytesIO
import numpy as np


class AsyncNIMEmbedder:
    """
    Async NVIDIA NIM embeddings (baai/bge-m3)

    Model: baai/bge-m3 (proven in MEMORY.md)
    Dimensions: 1024
    Quality: > nomic-embed-text (768 dim)

    Fallback: AsyncOllama (local) for airgapped environments
    """

    DEFAULT_MODEL = "baai/bge-m3"
    DEFAULT_DIM = 1024

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 model: str = DEFAULT_MODEL,
                 fallback_to_ollama: bool = True):
        """
        Args:
            api_key: NVIDIA NIM API key (or NIM_API_KEY env var)
            base_url: NIM endpoint
            model: baai/bge-m3 (recommended) or other
            fallback_to_ollama: Use local Ollama if NIM unavailable
        """
        self.api_key = api_key or os.getenv("NIM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_enabled = fallback_to_ollama
        self._ollama_embedder = None
        self._client = None

        if not self.api_key:
            raise ValueError("NIM_API_KEY required or set NIM_API_KEY env var")

    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _init_client(self) -> None:
        """Initialize HTTP client"""
        try:
            import httpx
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            await self._test_connection()
        except Exception as e:
            if self.fallback_enabled:
                await self._init_ollama_fallback()
            else:
                raise

    async def _test_connection(self) -> None:
        """Test NIM connection"""
        test_response = await self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": "test"},
            timeout=10.0
        )
        test_response.raise_for_status()

    async def _init_ollama_fallback(self) -> None:
        """Initialize Ollama fallback"""
        try:
            self._ollama_embedder = AsyncOllamaEmbedder()
            await self._ollama_embedder._init_client()
        except Exception:
            raise RuntimeError("NIM unavailable and Ollama fallback failed")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding with fallback"""
        # Ensure client is initialized
        if self._client is None:
            await self._init_client()

        try:
            return await self._embed_nim(text)
        except Exception as e:
            if self._ollama_embedder:
                return await self._ollama_embedder.embed(text)
            raise

    async def _embed_nim(self, text: str) -> List[float]:
        """Generate embedding via NIM"""
        response = await self._client.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": text[:512],  # Truncate to max tokens
                "encoding_format": "float"
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [await self.embed(t) for t in batch]
            results.extend(batch_results)
        return results

    def similarity(self, embedding1: List[float],
                  embedding2: List[float]) -> float:
        """Cosine similarity (sync method, no I/O)"""
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._ollama_embedder:
            await self._ollama_embedder.close()


class AsyncOllamaEmbedder:
    """
    Async Local Ollama fallback embeddings

    Models:
    - nomic-embed-text (768 dim, fast)
    - mxbai-embed-large (1024 dim, quality)
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIM = 768

    def __init__(self, model: str = DEFAULT_MODEL,
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._client = None

    async def _init_client(self) -> None:
        """Initialize HTTP client"""
        import httpx
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> List[float]:
        """Generate embedding via Ollama"""
        if self._client is None:
            await self._init_client()

        response = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()['embedding']

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


class AsyncEmbeddingCache:
    """Async disk cache for embeddings"""

    def __init__(self, cache_dir: Path, embedder: Union[AsyncNIMEmbedder, AsyncOllamaEmbedder]):
        self.cache_dir = Path(cache_dir)
        self.embedder = embedder

    async def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        import aiofiles.os
        try:
            await aiofiles.os.makedirs(self.cache_dir, exist_ok=True)
        except FileExistsError:
            pass

    async def get_or_compute(self, text: str) -> List[float]:
        """Get from cache or compute and store"""
        import aiofiles
        import aiofiles.os

        # Ensure cache directory exists
        await self._ensure_cache_dir()

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"{content_hash}.npy"

        # Check if cache file exists
        try:
            await aiofiles.os.stat(cache_path)
            # File exists, load it
            async with aiofiles.open(cache_path, 'rb') as f:
                content = await f.read()
            # Load numpy array from bytes
            buffer = BytesIO(content)
            return np.load(buffer).tolist()
        except FileNotFoundError:
            # File doesn't exist, compute embedding
            embedding = await self.embedder.embed(text)

            # Save to cache
            buffer = BytesIO()
            np.save(buffer, np.array(embedding))
            buffer.seek(0)

            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(buffer.read())

            return embedding


__all__ = ['AsyncNIMEmbedder', 'AsyncOllamaEmbedder', 'AsyncEmbeddingCache']
