"""
Semantic Cache for LLM Responses.

Provides embedding-based semantic caching to reduce LLM costs
by caching responses for semantically similar inputs.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass, field
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class CacheEntry:
    """Single cache entry with embedding and metadata."""
    embedding: np.ndarray
    description: str
    result: Any  # ClassificationResult
    prompt_hash: str
    timestamp: float
    hits: int = 0


class SemanticCache:
    """
    Embedding-based semantic cache for LLM responses.
    
    Uses sentence transformers to find semantically similar cached results.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        max_size: int = 10000,
        ttl_seconds: int = 86400,  # 24 hours
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize semantic cache.
        
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Cosine similarity threshold for cache hits
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
            persist_path: Optional path to persist cache to disk
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required for semantic cache. "
                "Install with: pip install sentence-transformers"
            )
        
        self.encoder = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        
        self._cache: Dict[str, CacheEntry] = {}  # key -> entry
        self._embeddings: List[np.ndarray] = []  # For fast similarity search
        self._keys: List[str] = []  # Parallel to _embeddings
        
        # Load persisted cache if available
        if persist_path and persist_path.exists():
            self.load(persist_path)
    
    def _make_key(self, description: str, prompt_hash: str) -> str:
        """Create exact-match cache key."""
        desc_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
        return f"{prompt_hash}:{desc_hash}"
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding."""
        emb = self.encoder.encode(text, normalize_embeddings=True)
        return emb.astype(np.float32)
    
    def get(self, description: str, prompt_hash: str) -> Optional[Any]:
        """
        Get cached result for description (exact or semantic match).
        
        Args:
            description: Input description
            prompt_hash: Hash of the system prompt
            
        Returns:
            Cached ClassificationResult or None
        """
        # First try exact match
        exact_key = self._make_key(description, prompt_hash)
        if exact_key in self._cache:
            entry = self._cache[exact_key]
            if time.time() - entry.timestamp < self.ttl_seconds:
                entry.hits += 1
                entry.result.cache_hit = True
                return entry.result
            else:
                self._remove_entry(exact_key)
        
        # Semantic similarity search
        if not self._embeddings:
            return None
        
        query_emb = self._encode(description)
        
        # Compute cosine similarities (embeddings are normalized)
        similarities = np.dot(self._embeddings, query_emb)
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            key = self._keys[best_idx]
            entry = self._cache[key]
            if entry.prompt_hash == prompt_hash and time.time() - entry.timestamp < self.ttl_seconds:
                entry.hits += 1
                entry.result.cache_hit = True
                return entry.result
            elif time.time() - entry.timestamp >= self.ttl_seconds:
                self._remove_entry(key)
        
        return None
    
    def set(self, description: str, prompt_hash: str, result: Any) -> None:
        """
        Store result in cache.
        
        Args:
            description: Input description
            prompt_hash: Hash of the system prompt
            result: ClassificationResult to cache
        """
        # Evict old entries if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._make_key(description, prompt_hash)
        embedding = self._encode(description)
        
        entry = CacheEntry(
            embedding=embedding,
            description=description,
            result=result,
            prompt_hash=prompt_hash,
            timestamp=time.time(),
        )
        
        self._cache[key] = entry
        self._embeddings.append(embedding)
        self._keys.append(key)
        
        # Persist if path configured
        if self.persist_path:
            self.save(self.persist_path)
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            idx = self._keys.index(key)
            del self._cache[key]
            del self._embeddings[idx]
            del self._keys[idx]
    
    def _evict_oldest(self) -> None:
        """Evict least recently used / oldest entries."""
        # Sort by timestamp, remove oldest 10%
        sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        num_to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:num_to_remove]:
            self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._embeddings.clear()
        self._keys.clear()
    
    def save(self, path: Path) -> None:
        """Persist cache to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [],
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
            }
        }
        for key, entry in self._cache.items():
            data["entries"].append({
                "key": key,
                "embedding": entry.embedding.tolist(),
                "description": entry.description,
                "result": entry.result.to_dict() if hasattr(entry.result, 'to_dict') else str(entry.result),
                "prompt_hash": entry.prompt_hash,
                "timestamp": entry.timestamp,
                "hits": entry.hits,
            })
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load cache from disk."""
        with open(path, "r") as f:
            data = json.load(f)
        
        for entry_data in data.get("entries", []):
            key = entry_data["key"]
            embedding = np.array(entry_data["embedding"], dtype=np.float32)
            entry = CacheEntry(
                embedding=embedding,
                description=entry_data["description"],
                result=entry_data["result"],  # Will be dict, not ClassificationResult
                prompt_hash=entry_data["prompt_hash"],
                timestamp=entry_data["timestamp"],
                hits=entry_data.get("hits", 0),
            )
            self._cache[key] = entry
            self._embeddings.append(embedding)
            self._keys.append(key)
        
        print(f"Loaded semantic cache: {len(self._cache)} entries")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "similarity_threshold": self.similarity_threshold,
            "ttl_hours": self.ttl_seconds / 3600,
        }


class HybridCache:
    """
    Combines exact-match cache (fast) with semantic cache (broad coverage).
    """
    
    def __init__(
        self,
        exact_cache: Optional[Any] = None,  # LLMCache
        semantic_cache: Optional[SemanticCache] = None,
    ):
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
    
    def get(self, description: str, prompt_hash: str) -> Optional[Any]:
        """Try exact cache first, then semantic."""
        # Exact match
        if self.exact_cache:
            result = self.exact_cache.get(description, prompt_hash)
            if result:
                return result
        
        # Semantic match
        if self.semantic_cache:
            result = self.semantic_cache.get(description, prompt_hash)
            if result:
                return result
        
        return None
    
    def set(self, description: str, prompt_hash: str, result: Any) -> None:
        """Store in both caches."""
        if self.exact_cache:
            self.exact_cache.set(description, prompt_hash, result)
        if self.semantic_cache:
            self.semantic_cache.set(description, prompt_hash, result)
    
    def stats(self) -> Dict[str, Any]:
        stats = {}
        if self.exact_cache and hasattr(self.exact_cache, '_cache'):
            stats["exact_size"] = len(self.exact_cache._cache)
        if self.semantic_cache:
            stats["semantic"] = self.semantic_cache.stats()
        return stats