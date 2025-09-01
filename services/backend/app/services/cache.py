"""Simple cache service for L1 (Redis) and L2 (semantic) caching."""

import json
from typing import Any, Dict, Optional

import chromadb
import redis.asyncio as redis
from app.core.config import settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


class CacheService:
    """Simple dual-layer cache: Redis L1 + ChromaDB L2."""
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.chroma: Optional[ClientAPI] = None
        self.cache_collection: Optional[Collection] = None
    
    async def connect(self):
        """Connect to Redis and ChromaDB."""
        # Redis L1 cache
        self.redis = redis.from_url(settings.redis_url)
        
        # ChromaDB L2 semantic cache
        from urllib.parse import urlparse
        parsed_url = urlparse(settings.chroma_host)
        self.chroma = chromadb.HttpClient(
            host=parsed_url.hostname or "localhost",
            port=parsed_url.port or 8000
        )
        try:
            self.cache_collection = self.chroma.get_collection("query_cache")
        except:
            self.cache_collection = self.chroma.create_collection("query_cache")
    
    async def get(self, question: str) -> Optional[str]:
        """Get cached response - try L1 first, then L2."""
        # Try L1 cache (exact match)
        if self.redis:
            cached = await self.redis.get(f"q:{question}")
            if cached:
                return json.loads(cached)["answer"]
        
        # Try L2 semantic cache
        if self.cache_collection:
            try:
                from app.services.embeddings import get_embedding
                query_embedding = await get_embedding(question)
                
                results = self.cache_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1
                )
                
                if (results and 
                    results.get("documents") is not None and 
                    results.get("distances") is not None and 
                    results.get("metadatas") is not None):
                    
                    documents = results["documents"]
                    distances = results["distances"]
                    metadatas = results["metadatas"]
                    
                    if (documents and len(documents) > 0 and len(documents[0]) > 0 and
                        distances and len(distances) > 0 and len(distances[0]) > 0 and
                        metadatas and len(metadatas) > 0 and len(metadatas[0]) > 0):
                        
                        distance = distances[0][0]
                        similarity = 1 - distance  # Convert distance to similarity
                        
                        if similarity >= settings.semantic_threshold:
                            metadata = metadatas[0][0]
                            if isinstance(metadata, dict) and "answer" in metadata:
                                answer = metadata["answer"]
                                if isinstance(answer, str):
                                    return answer
            except Exception as e:
                print(f"L2 cache error: {e}")
        
        return None
    
    async def set(self, question: str, answer: str):
        """Cache the response in both L1 and L2."""
        # L1 cache (exact match)
        if self.redis:
            cache_data = {"question": question, "answer": answer}
            await self.redis.setex(
                f"q:{question}",
                settings.cache_ttl,
                json.dumps(cache_data)
            )
        
        # L2 semantic cache
        if self.cache_collection:
            try:
                from app.services.embeddings import get_embedding
                query_embedding = await get_embedding(question)
                
                self.cache_collection.add(
                    embeddings=[query_embedding],
                    documents=[question],
                    metadatas=[{"answer": answer}],
                    ids=[f"q_{hash(question)}"]
                )
            except Exception as e:
                print(f"L2 cache store error: {e}")
    
    async def close(self):
        """Close connections."""
        if self.redis:
            await self.redis.close()
