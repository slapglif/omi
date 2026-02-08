"""
Local embeddings via Ollama
Pattern: Local GPU via Ollama, nothing leaves the machine
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Union
import numpy as np


class OllamaEmbedder:
    """
    Local embedding generation via Ollama
    
    Models:
    - nomic-embed-text (fast, good quality)
    - mxbai-embed-large (higher quality, slower)
    
    Default: nomic-embed-text (768 dimensions)
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIM = 768
    
    def __init__(self, model: str = DEFAULT_MODEL, 
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        # Try to import ollama client, fall back to requests
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self._use_client = True
        except ImportError:
            import requests
            self._use_client = False
            self._session = requests.Session()
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
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
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.embed(t) for t in texts]
    
    def similarity(self, embedding1: List[float], 
                  embedding2: List[float]) -> float:
        """Cosine similarity between two embeddings"""
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        return dot / (norm1 * norm2)
    
    def search_similar(self, query: str, 
                      candidates: List[tuple], 
                      top_k: int = 10,
                      threshold: float = 0.7) -> List[tuple]:
        """
        Find most similar candidates to query
        
        Args:
            query: Search text
            candidates: List of (id, text, embedding) tuples
            top_k: Number of results
            threshold: Minimum similarity score
        
        Returns:
            List of (id, similarity_score) tuples, sorted by similarity
        """
        query_embedding = self.embed(query)
        
        results = []
        for cid, text, embedding in candidates:
            sim = self.similarity(query_embedding, embedding)
            if sim >= threshold:
                results.append((cid, sim, text))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class EmbeddingCache:
    """
    Disk cache for embeddings to avoid recomputing
    
    Pattern: Hash content -> store embedding as .npy file
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, content: str) -> Path:
        """Get cache file path for content"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.cache_dir / f"{content_hash}.npy"
    
    def get_or_compute(self, content: str, embedder: OllamaEmbedder) -> List[float]:
        """Get from cache or compute and store"""
        cache_path = self._get_cache_path(content)
        
        if cache_path.exists():
            # Load from cache
            return np.load(cache_path).tolist()
        
        # Compute and cache
        embedding = embedder.embed(content)
        np.save(cache_path, np.array(embedding))
        
        return embedding
    
    def invalidate(self, content_hash: str) -> None:
        """Remove cached embedding"""
        cache_path = self.cache_dir / f"{content_hash}.npy"
        if cache_path.exists():
            cache_path.unlink()
    
    def clear_old(self, max_age_days: int = 30) -> None:
        """Clear embeddings older than specified days"""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for file_path in self.cache_dir.glob("*.npy"):
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                file_path.unlink()


class HybridSearch:
    """
    Hybrid search: semantic + keyword
    
    Pattern: Get candidates via embedding similarity,
    then rerank with BM25 or keyword matching
    """
    
    def __init__(self, embedder: OllamaEmbedder,
                 alpha: float = 0.7):
        """
        Args:
            embedder: OllamaEmbedder instance
            alpha: Weight for semantic vs keyword (0-1)
                   0.7 = 70% semantic, 30% keyword
        """
        self.embedder = embedder
        self.alpha = alpha
    
    def search(self, query: str, 
              candidates: List[dict],
              top_k: int = 10) -> List[dict]:
        """
        Hybrid search over candidates
        
        Candidate format:
        {
            'id': str,
            'text': str,
            'embedding': List[float]
        }
        """
        # Semantic scores
        query_emb = self.embedder.embed(query)
        
        results = []
        for cand in candidates:
            semantic = self.embedder.similarity(query_emb, 
                                                cand['embedding'])
            
            # Simple keyword score (could use BM25 instead)
            keyword = self._keyword_score(query, cand['text'])
            
            # Combined score
            combined = (self.alpha * semantic + 
                       (1 - self.alpha) * keyword)
            
            results.append({
                'id': cand['id'],
                'text': cand['text'],
                'semantic_score': semantic,
                'keyword_score': keyword,
                'combined_score': combined
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def _keyword_score(self, query: str, text: str) -> float:
        """Simple keyword overlap score"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words & text_words)
        return matches / len(query_words)


def cosine_similarity(v1: Union[List[float], np.ndarray],
                     v2: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(v1)
    b = np.array(v2)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)
