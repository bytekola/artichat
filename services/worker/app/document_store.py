"""
Redis Document Store implementation for the "Best of Breed" architecture.

This module handles:
1. Storing complete article and content in Redis (Document Store)
2. Generating hashed keys for efficient storage and retrieval
3. Providing methods to store/retrieve full article data

The complementary Vector Store (ChromaDB) stores chunks with references to this document store.
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

from pydantic import HttpUrl, ValidationError

from .config import get_redis_client, settings
from .schemas import ArticleMetadata, CachedQuery, StoredArticle

logger = logging.getLogger(__name__)


class DocumentStore:
    """Redis-based document store for complete article data."""
    
    @staticmethod
    def _generate_article_key(url: str) -> str:
        """
        Generate a consistent, safe Redis key for an article URL.
        Uses SHA256 hash to ensure clean keys regardless of URL complexity.
        
        Args:
            url: The article URL
            
        Returns:
            Redis key in format: article:<hash>
        """
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        return f"article:{url_hash}"
    
    @staticmethod
    def _generate_cache_key(query: str) -> str:
        """
        Generate a consistent cache key for user queries.
        
        Args:
            query: The user's question/prompt
            
        Returns:
            Redis key in format: cache:<hash>
        """
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        return f"cache:{query_hash}"
    
    @staticmethod
    def store_article(
        url: str,
        title: str,
        full_text: str,
        metadata: ArticleMetadata,
        summary: Optional[str] = None,
    ) -> str:
        """
        Store complete article data in Redis Document Store.
        
        Args:
            url: Article source URL
            title: Article title
            full_text: Complete article text content
            metadata: Structured article metadata
            summary: AI-generated summary (if available)
            ttl_hours: How long to keep the article in Redis (hours)
            
        Returns:
            The Redis key used for storage
        """
        try:
            key = DocumentStore._generate_article_key(url)
            
            # Create the document data structure using Pydantic model
            document_data = StoredArticle(
                url=HttpUrl(url),
                title=title,
                full_text=full_text,
                metadata=metadata,
                summary=summary,
            )
            # Store as JSON in Redis using Pydantic's serialization
            redis_client = get_redis_client()
            redis_client.set(
                key, 
                document_data.model_dump_json()
            )
            
            logger.info(f"📄 Stored article document: {key} ({len(full_text)} chars)")
            return key
            
        except ValidationError as e:
            logger.error(f"ERROR: Pydantic validation failed for article {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"ERROR: Failed to store article document for {url}: {e}")
            raise
    
    @staticmethod
    def get_article(url: str) -> Optional[StoredArticle]:
        """
        Retrieve complete article data from Redis Document Store.
        
        Args:
            url: Article source URL
            
        Returns:
            StoredArticle object or None if not found
        """
        try:
            key = DocumentStore._generate_article_key(url)
            redis_client = get_redis_client()
            json_data = redis_client.get(key)

            if json_data is None:
                logger.debug(f"📄 Article not found in document store: {url}")
                return None

            document = StoredArticle.model_validate_json(str(json_data))
            logger.debug(f"Retrieved article document: {key}")
            return document
        except ValidationError as e:
            logger.error(f"ERROR: Pydantic validation failed on retrieval for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"ERROR: Failed to retrieve article document for {url}: {e}")
            return None
    
    @staticmethod
    def get_article_by_key(key: str) -> Optional[StoredArticle]:
        """
        Retrieve article data by Redis key (useful for chunk references).
        
        Args:
            key: Redis key (format: article:<hash>)
            
        Returns:
            StoredArticle object or None if not found
        """
        try:
            redis_client = get_redis_client()
            json_data = redis_client.get(key)

            if json_data is None:
                logger.debug(f"📄 Document not found: {key}")
                return None

            document = StoredArticle.model_validate_json(str(json_data))
            logger.debug(f"Retrieved document: {key}")
            return document
        except ValidationError as e:
            logger.error(f"ERROR: Pydantic validation failed on retrieval for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"ERROR: Failed to retrieve document: {key} - {e}")
            return None
    
    @staticmethod
    def cache_query_response(query: str, response: str, ttl_hours: int = 1) -> str:
        """
        Cache a query response for fast repeated requests.
        
        Args:
            query: The user's original question
            response: The generated answer
            ttl_hours: How long to cache the response (hours)
            
        Returns:
            The Redis key used for caching
        """
        try:
            key = DocumentStore._generate_cache_key(query)
            
            cache_data = CachedQuery(query=query, response=response)
            
            redis_client = get_redis_client()
            redis_client.setex(
                key,
                ttl_hours * 3600,  # Convert hours to seconds
                cache_data.model_dump_json()
            )
            
            logger.info(f"Query response cached: {key[:16]}...")
            return key
            
        except ValidationError as e:
            logger.error(f"ERROR: Pydantic validation failed for query cache: {e}")
            raise
        except Exception as e:
            logger.error(f"ERROR: Failed to cache query response: {e}")
            raise
        
    @staticmethod
    def health_check() -> bool:
        """
        Check if Redis document store is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            redis_client = get_redis_client()
            redis_client.ping()
            logger.debug("Document store health check: OK")
            return True
        except Exception as e:
            logger.error(f"ERROR: Document store health check failed: {e}")
            return False
