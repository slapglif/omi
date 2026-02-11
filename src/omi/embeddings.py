"""
Embeddings via NVIDIA NIM (baai/bge-m3)
Pattern: Cloud API with local caching, consistent with MEMORY.md setup
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding providers must implement:
    - embed(): Generate embeddings for text
    - dimensions: Return embedding dimensionality
    - similarity(): Calculate similarity between embeddings
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for input text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Return the dimensionality of embeddings produced by this provider.

        Returns:
            Embedding vector dimension (e.g., 768, 1024)
        """
        pass

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


@dataclass
class NIMConfig:
    """NVIDIA NIM configuration"""
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "baai/bge-m3"  # Proven deployment from MEMORY.md
    embedding_dim: int = 1024
    timeout: int = 30
    max_tokens: int = 512


class NIMEmbedder(EmbeddingProvider):
    """
    NVIDIA NIM embeddings (baai/bge-m3)

    Model: baai/bge-m3 (proven in MEMORY.md)
    Dimensions: 1024
    Quality: > nomic-embed-text (768 dim)

    Fallback: Ollama (local) for airgapped environments
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
        
        if not self.api_key:
            raise ValueError("NIM_API_KEY required or set NIM_API_KEY env var")
        
        # Initialize HTTP session
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            self._test_connection()
        except Exception as e:
            if self.fallback_enabled:
                self._init_ollama_fallback()
            else:
                raise
    
    def _test_connection(self) -> None:
        """Test NIM connection"""
        test_response = self._session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": "test"},
            timeout=10
        )
        test_response.raise_for_status()
    
    def _init_ollama_fallback(self) -> None:
        """Initialize Ollama fallback"""
        try:
            from .ollama_fallback import OllamaEmbedder
            self._ollama_embedder = OllamaEmbedder()
        except Exception:
            raise RuntimeError("NIM unavailable and Ollama fallback failed")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding with fallback"""
        try:
            return self._embed_nim(text)
        except Exception as e:
            if self._ollama_embedder:
                return self._ollama_embedder.embed(text)
            raise

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality (1024 for baai/bge-m3)"""
        return self.DEFAULT_DIM

    def _embed_nim(self, text: str) -> List[float]:
        """Generate embedding via NIM"""
        response = self._session.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": text[:512],  # Truncate to max tokens
                "encoding_format": "float"
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return data["data"][0]["embedding"]
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.embed(t) for t in batch]
            results.extend(batch_results)
        return results
    
    def similarity(self, embedding1: List[float], 
                  embedding2: List[float]) -> float:
        """Cosine similarity"""
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)


@dataclass
class OpenAIConfig:
    """OpenAI embeddings configuration"""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    timeout: int = 30
    max_tokens: int = 8191


class OpenAIEmbedder(EmbeddingProvider):
    """
    OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)

    Models:
    - text-embedding-3-small: 1536 dimensions (cost-efficient)
    - text-embedding-3-large: 3072 dimensions (highest quality)

    Proven quality for semantic search and RAG applications
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIM = 1536

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            base_url: OpenAI API endpoint
            model: text-embedding-3-small (default) or text-embedding-3-large
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required or set OPENAI_API_KEY env var")

        # Initialize HTTP session
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self._test_connection()

    def _test_connection(self) -> None:
        """Test OpenAI API connection"""
        test_response = self._session.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": "test"},
            timeout=10
        )
        test_response.raise_for_status()

    def embed(self, text: str) -> List[float]:
        """Generate embedding via OpenAI API"""
        response = self._session.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model,
                "input": text,
                "encoding_format": "float"
            },
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data["data"][0]["embedding"]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality based on model"""
        return self.MODEL_DIMENSIONS.get(self.model, self.DEFAULT_DIM)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        OpenAI supports up to 2048 inputs per batch, but we use 100 as default
        for better reliability and rate limit management
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._session.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float"
                },
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            batch_results = [item["embedding"] for item in data["data"]]
            results.extend(batch_results)

        return results


@dataclass
class CohereConfig:
    """Cohere embeddings configuration"""
    api_key: str
    base_url: str = "https://api.cohere.ai/v1"
    model: str = "embed-english-v3.0"
    embedding_dim: int = 1024
    timeout: int = 30
    input_type: str = "search_document"


class CohereEmbedder(EmbeddingProvider):
    """
    Cohere embeddings (embed-english-v3.0)

    Models:
    - embed-english-v3.0: 1024 dimensions (recommended)
    - embed-multilingual-v3.0: 1024 dimensions (multilingual support)

    High-quality embeddings optimized for semantic search and retrieval
    """

    DEFAULT_MODEL = "embed-english-v3.0"
    DEFAULT_DIM = 1024

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.cohere.ai/v1",
                 model: str = DEFAULT_MODEL,
                 input_type: str = "search_document"):
        """
        Args:
            api_key: Cohere API key (or COHERE_API_KEY env var)
            base_url: Cohere API endpoint
            model: embed-english-v3.0 (default) or other Cohere embedding model
            input_type: search_document (default), search_query, classification, or clustering
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.input_type = input_type

        if not self.api_key:
            raise ValueError("COHERE_API_KEY required or set COHERE_API_KEY env var")

        # Initialize HTTP session
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        self._test_connection()

    def _test_connection(self) -> None:
        """Test Cohere API connection"""
        test_response = self._session.post(
            f"{self.base_url}/embed",
            json={
                "model": self.model,
                "texts": ["test"],
                "input_type": self.input_type
            },
            timeout=10
        )
        test_response.raise_for_status()

    def embed(self, text: str) -> List[float]:
        """Generate embedding via Cohere API"""
        response = self._session.post(
            f"{self.base_url}/embed",
            json={
                "model": self.model,
                "texts": [text],
                "input_type": self.input_type
            },
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data["embeddings"][0]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality based on model"""
        return self.MODEL_DIMENSIONS.get(self.model, self.DEFAULT_DIM)

    def embed_batch(self, texts: List[str], batch_size: int = 96) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Cohere supports up to 96 texts per batch
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._session.post(
                f"{self.base_url}/embed",
                json={
                    "model": self.model,
                    "texts": batch,
                    "input_type": self.input_type
                },
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            results.extend(data["embeddings"])

        return results


class OllamaEmbedder(EmbeddingProvider):
    """
    Local Ollama fallback embeddings

    Models:
    - nomic-embed-text (768 dim, fast)
    - mxbai-embed-large (1024 dim, quality)
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIM = 768

    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024
    }
    
    def __init__(self, model: str = DEFAULT_MODEL, 
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self._use_client = True
        except ImportError:
            import requests
            self._use_client = False
            self._session = requests.Session()
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding via Ollama"""
        if self._use_client:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']
        else:
            import requests
            response = self._session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()['embedding']

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality based on model"""
        return self.MODEL_DIMENSIONS.get(self.model, self.DEFAULT_DIM)


class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    Local Sentence Transformers embeddings

    Models:
    - all-MiniLM-L6-v2 (384 dim, fast, default)
    - all-mpnet-base-v2 (768 dim, quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)

    Fully offline, no API keys required
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIM = 384

    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-MiniLM-L12-v2": 384
    }

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Args:
            model: Model name (default: all-MiniLM-L6-v2)
        """
        self.model = model
        self._model_instance = None
        self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model}: {e}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding via sentence-transformers"""
        if self._model_instance is None:
            self._load_model()

        embedding = self._model_instance.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def dimensions(self) -> int:
        """Return embedding dimensionality based on model"""
        return self.MODEL_DIMENSIONS.get(self.model, self.DEFAULT_DIM)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        sentence-transformers handles batching efficiently internally
        """
        if self._model_instance is None:
            self._load_model()

        embeddings = self._model_instance.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        return [emb.tolist() for emb in embeddings]


# Provider registry mapping provider names to classes
PROVIDER_REGISTRY = {
    "nim": NIMEmbedder,
    "openai": OpenAIEmbedder,
    "cohere": CohereEmbedder,
    "ollama": OllamaEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


class EmbeddingCache:
    """Disk cache for embeddings"""

    def __init__(self, cache_dir: Path, embedder: Union[NIMEmbedder, OpenAIEmbedder, CohereEmbedder, OllamaEmbedder, SentenceTransformerEmbedder]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
    
    def get_or_compute(self, text: str) -> List[float]:
        """Get from cache or compute and store"""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"{content_hash}.npy"
        
        if cache_path.exists():
            return np.load(cache_path).tolist()
        
        embedding = self.embedder.embed(text)
        np.save(cache_path, np.array(embedding))
        return embedding


class NIMInference:
    """
    NVIDIA NIM inference for agent operations
    
    Use cases:
    - Micro-model verification (state capsule validation)
    - Contradiction detection (semantic comparison)
    - Memory classification (fact vs belief)
    """
    
    CAPSULE_VERIFY_MODEL = "baai/bge-m3"
    CONTRADICTION_MODEL = "baai/bge-m3"
    
    def __init__(self, config: Optional[NIMConfig] = None):
        self.config = config or NIMConfig(
            api_key=os.getenv("NIM_API_KEY", "")
        )
        self._session = None
        self._init_session()
    
    def _init_session(self) -> None:
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
    
    def verify_capsule_state(self, 
                           capsule_state: str, 
                           retrieved_clips: List[str],
                           threshold: float = 0.85) -> dict:
        """
        Micro-model verification of state capsule
        
        Returns: accept | patch | reject
        """
        # Embed capsule and clips
        capsule_emb = self._embed(capsule_state)
        clip_embs = [self._embed(clip) for clip in retrieved_clips]
        
        # Calculate semantic coherence
        sims = [self._similarity(capsule_emb, ce) for ce in clip_embs]
        avg_sim = sum(sims) / len(sims) if sims else 0
        
        if avg_sim >= threshold:
            return {"verdict": "accept", "confidence": avg_sim}
        elif avg_sim >= threshold * 0.8:
            return {"verdict": "patch", "confidence": avg_sim, "drift": 1 - avg_sim}
        else:
            return {"verdict": "reject", "confidence": avg_sim}
    
    def _embed(self, text: str) -> List[float]:
        response = self._session.post(
            f"{self.config.base_url}/embeddings",
            json={"model": self.config.model, "input": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    def _similarity(self, e1: List[float], e2: List[float]) -> float:
        import numpy as np
        a, b = np.array(e1), np.array(e2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def classify_memory_type(self, content: str) -> str:
        """
        Classify memory as: fact | experience | belief | decision
        
        Uses semantic similarity to type examples
        """
        type_examples = {
            "fact": "LanceDB uses ANN for search with O(log n) complexity",
            "experience": "I fixed the bug by checking the null pointer first",
            "belief": "I think this approach works better than alternatives",
            "decision": "We chose to use PostgreSQL over SQLite"
        }
        
        content_emb = self._embed(content)
        
        best_type = "experience"  # default
        best_sim = 0
        
        for mem_type, example in type_examples.items():
            type_emb = self._embed(example)
            sim = self._similarity(content_emb, type_emb)
            if sim > best_sim:
                best_sim = sim
                best_type = mem_type
        
        return best_type


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity"""
    import numpy as np
    a, b = np.array(v1), np.array(v2)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers based on configuration.

    Supports all available providers:
    - nim: NVIDIA NIM (baai/bge-m3)
    - openai: OpenAI embeddings (text-embedding-3-small/large)
    - cohere: Cohere embeddings (embed-english-v3.0)
    - ollama: Local Ollama embeddings (nomic-embed-text)
    - sentence-transformers: Local sentence-transformers (all-MiniLM-L6-v2)

    Example:
        # Default NIM provider
        factory = EmbeddingProviderFactory()
        provider = factory.get_provider()

        # OpenAI provider with config
        config = {
            "provider": "openai",
            "api_key": "sk-...",
            "model": "text-embedding-3-large"
        }
        provider = factory.get_provider(config)
    """

    PROVIDER_MAP = {
        "nim": NIMEmbedder,
        "openai": OpenAIEmbedder,
        "cohere": CohereEmbedder,
        "ollama": OllamaEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "sentence_transformers": SentenceTransformerEmbedder,
    }

    DEFAULT_PROVIDER = "nim"

    def __init__(self, default_config: Optional[Dict] = None):
        """
        Initialize factory with optional default configuration.

        Args:
            default_config: Default configuration for provider selection
        """
        self.default_config = default_config or {}

    def get_provider(self, config: Optional[Dict] = None) -> EmbeddingProvider:
        """
        Create an embedding provider based on configuration.

        Args:
            config: Provider configuration dict with keys:
                - provider: Provider type (nim, openai, cohere, ollama, sentence-transformers)
                - Additional provider-specific parameters

        Returns:
            Configured EmbeddingProvider instance

        Raises:
            ValueError: If provider type is unknown or configuration is invalid
        """
        # Merge with default config
        merged_config = {**self.default_config, **(config or {})}

        # Extract provider type
        provider_type = merged_config.get("provider", self.DEFAULT_PROVIDER)

        # Get provider class
        provider_class = self.PROVIDER_MAP.get(provider_type)
        if provider_class is None:
            available = ", ".join(self.PROVIDER_MAP.keys())
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available providers: {available}"
            )

        # Extract provider-specific parameters (exclude 'provider' key)
        provider_params = {
            k: v for k, v in merged_config.items() if k != "provider"
        }

        # Instantiate provider
        try:
            return provider_class(**provider_params)
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration for {provider_type} provider: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {provider_type} provider: {e}"
            )


# Backwards compatibility
OllamaFallbackEmbedder = OllamaEmbedder
