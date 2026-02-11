"""
Embeddings via NVIDIA NIM (baai/bge-m3)
Pattern: Cloud API with local caching, consistent with MEMORY.md setup
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    import ollama  # type: ignore[import-not-found]


@dataclass
class NIMConfig:
    """NVIDIA NIM configuration"""
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "baai/bge-m3"  # Proven deployment from MEMORY.md
    embedding_dim: int = 1024
    timeout: int = 30
    max_tokens: int = 512


class NIMEmbedder:
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
        self.api_key: Optional[str] = api_key or os.getenv("NIM_API_KEY")
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.fallback_enabled: bool = fallback_to_ollama
        self._ollama_embedder: Optional['OllamaEmbedder'] = None
        self._session: Any  # requests.Session
        
        if not self.api_key:
            raise ValueError("NIM_API_KEY required or set NIM_API_KEY env var")
        
        # Initialize HTTP session
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            self._session = session
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
            from .ollama_fallback import OllamaEmbedder  # type: ignore[import-not-found]
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
        embedding: List[float] = data["data"][0]["embedding"]
        return embedding
    
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
        v1: Any = np.array(embedding1)  # type: ignore[attr-defined]
        v2: Any = np.array(embedding2)  # type: ignore[attr-defined]

        dot: Any = np.dot(v1, v2)  # type: ignore[attr-defined]
        norm1: Any = np.linalg.norm(v1)
        norm2: Any = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        result: Any = dot / (norm1 * norm2)
        return float(result)


class OllamaEmbedder:
    """
    Local Ollama fallback embeddings
    
    Models:
    - nomic-embed-text (768 dim, fast)
    - mxbai-embed-large (1024 dim, quality)
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIM = 768
    
    def __init__(self, model: str = DEFAULT_MODEL,
                 base_url: str = "http://localhost:11434"):
        self.model: str = model
        self.base_url: str = base_url
        self.client: Any  # ollama.Client
        self._use_client: bool
        self._session: Any  # requests.Session

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
            embedding: List[float] = response['embedding']
            return embedding
        else:
            import requests
            resp = self._session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            resp.raise_for_status()
            embedding_result: List[float] = resp.json()['embedding']
            return embedding_result


class EmbeddingCache:
    """Disk cache for embeddings"""

    def __init__(self, cache_dir: Path, embedder: Union[NIMEmbedder, OllamaEmbedder]):
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder: Union[NIMEmbedder, OllamaEmbedder] = embedder
    
    def get_or_compute(self, text: str) -> List[float]:
        """Get from cache or compute and store"""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_path = self.cache_dir / f"{content_hash}.npy"

        if cache_path.exists():
            loaded: Any = np.load(cache_path)  # type: ignore[attr-defined]
            result: List[float] = loaded.tolist()
            return result

        embedding = self.embedder.embed(text)
        np.save(cache_path, np.array(embedding))  # type: ignore[attr-defined]
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
        self.config: NIMConfig = config or NIMConfig(
            api_key=os.getenv("NIM_API_KEY", "")
        )
        self._session: Any  # requests.Session
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
                           threshold: float = 0.85) -> Dict[str, Any]:
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
        embedding: List[float] = response.json()["data"][0]["embedding"]
        return embedding
    
    def _similarity(self, e1: List[float], e2: List[float]) -> float:
        a: Any = np.array(e1)  # type: ignore[attr-defined]
        b: Any = np.array(e2)  # type: ignore[attr-defined]
        result: Any = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # type: ignore[attr-defined]
        return float(result)
    
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
        best_sim: float = 0.0

        for mem_type, example in type_examples.items():
            type_emb = self._embed(example)
            sim = self._similarity(content_emb, type_emb)
            if sim > best_sim:
                best_sim = sim
                best_type = mem_type

        return best_type


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity"""
    a: Any = np.array(v1)  # type: ignore[attr-defined]
    b: Any = np.array(v2)  # type: ignore[attr-defined]
    dot: Any = np.dot(a, b)  # type: ignore[attr-defined]
    norm: Any = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# Backwards compatibility
OllamaFallbackEmbedder = OllamaEmbedder
