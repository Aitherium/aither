"""
MemoryManager - Agent Memory with Unified Embedding
====================================================

Simple memory manager for agents. Uses AitherMindClient for embeddings
when available (shared cache, consistent embeddings) with fallback to
Google Gemini API.

Usage:
    from aither_adk.memory.memory import MemoryManager
    
    memory = MemoryManager("memory/long_term.json")
    memory.add_memory("User prefers dark mode", source="preference")
    results = memory.search("dark mode")
"""

import os
import json
import time
import math
from typing import List, Dict, Optional, Any

# ===============================================================================
# UNIFIED EMBEDDING - Try AitherMindClient first, fallback to Gemini
# ===============================================================================
_MIND_CLIENT_AVAILABLE = False
_mind_client = None

try:
    # AitherMindClient provides unified embeddings across all AitherOS services
    from lib.AitherMindClient import get_mind_client, cosine_similarity, embed_sync
    _MIND_CLIENT_AVAILABLE = True
except ImportError:
    # Fallback - define our own cosine similarity
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

# Lazy import for Gemini fallback
_gemini_client = None


def _get_gemini_client():
    """Lazy-load Gemini client only when needed."""
    global _gemini_client
    if _gemini_client is None:
        from google.genai import Client
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        _gemini_client = Client(api_key=api_key)
    return _gemini_client


class MemoryManager:
    """
    Simple memory manager with semantic search.
    
    Uses AitherMindClient for embeddings when AitherMind service is running,
    otherwise falls back to Google Gemini API.
    """
    
    def __init__(self, memory_file: str, use_mind_client: bool = True):
        """
        Initialize memory manager.
        
        Args:
            memory_file: Path to JSON file for memory persistence
            use_mind_client: If True, try AitherMindClient before Gemini
        """
        self.memory_file = memory_file
        self.memories = self._load_memories()
        self.use_mind_client = use_mind_client and _MIND_CLIENT_AVAILABLE

    def _load_memories(self) -> List[Dict]:
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[warning] Failed to load memories: {e}")
        return []

    def _save_memories(self):
        if self.memory_file:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using unified MindClient or Gemini fallback.
        """
        # Try AitherMindClient first (unified embeddings, shared cache)
        if self.use_mind_client and _MIND_CLIENT_AVAILABLE:
            try:
                embedding = embed_sync(text)
                if embedding:
                    return embedding
            except Exception as e:
                print(f"[debug] MindClient unavailable, falling back to Gemini: {e}")
        
        # Fallback to Gemini API
        try:
            client = _get_gemini_client()
            result = client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"[error] Embedding failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return cosine_similarity(vec1, vec2)

    def add_memory(self, text: str, source: str = "user", tags: List[str] = None) -> Optional[Dict]:
        """Adds a new memory with embedding."""
        embedding = self._get_embedding(text)
        if not embedding:
            return None

        memory = {
            "id": str(time.time()),
            "text": text,
            "embedding": embedding,
            "timestamp": time.time(),
            "source": source,
            "tags": tags or []
        }
        self.memories.append(memory)
        self._save_memories()
        return memory

    def search(self, query: str, limit: int = 5, threshold: float = 0.65) -> List[Dict]:
        """Searches for relevant memories."""
        if not self.memories:
            return []

        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []

        results = []
        for mem in self.memories:
            if "embedding" not in mem:
                continue
            score = self._cosine_similarity(query_embedding, mem["embedding"])
            if score >= threshold:
                # Return a copy without the embedding to save space/logs
                mem_copy = mem.copy()
                del mem_copy["embedding"]
                mem_copy["score"] = score
                results.append(mem_copy)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_context_string(self, query: str, limit: int = 3) -> str:
        """Returns a formatted string of relevant memories."""
        memories = self.search(query, limit=limit)
        if not memories:
            return ""

        context = "\n[RELEVANT MEMORIES]\n"
        for mem in memories:
            context += f"- {mem['text']} (Source: {mem['source']})\n"
        context += "[END MEMORIES]\n"
        return context
