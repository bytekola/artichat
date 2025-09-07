"""
Redis Document Store implementation for ingestion worker.

This module handles storing complete article and content in Redis (Document Store)

The complementary Vector Store (ChromaDB) stores chunks with references to this document store.
"""

import hashlib
import logging
from typing import Optional

from pydantic import HttpUrl, ValidationError

from .config import get_redis_client, settings
from .schemas import ArticleMetadata, StoredArticle

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
    def article_exists(url: str) -> bool:
        """
        Check if an article already exists in the document store.
        
        Args:
            url: Article source URL
            
        Returns:
            True if the article exists, False otherwise
        """
        try:
            key = DocumentStore._generate_article_key(url)
            redis_client = get_redis_client()
            exists = redis_client.exists(key)
            return bool(exists)
        except Exception as e:
            logger.error(f"ERROR: Failed to check article existence for {url}: {e}")
            return False
    
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
