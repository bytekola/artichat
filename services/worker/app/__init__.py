"""
Article Ingestion Worker Package

This package provides functionality for asynchronously processing article URLs,
extracting content, performing NLP enrichment, and storing results in a vector database.

Main components:
- worker: Main job processing logic
- loader: Article content extraction from URLs
- enrichment: NLP processing (entities, keywords, sentiment)
- storage: Vector database storage with embeddings
- schemas: Pydantic models for type safety
- config: Configuration management
"""

from .config import settings
from .enrichment import run_analysis
from .loader import load_document_from_url
from .schemas import (
    ArticleMetadata,
    DocumentChunk,
    IngestionJob,
    IngestionResult,
    NLPAnalysisResult,
)
from .storage import chunk_embed_and_store
from .worker import process_article_job

__version__ = "1.0.0"

__all__ = [
    "settings",
    "IngestionJob",
    "ArticleMetadata", 
    "DocumentChunk",
    "IngestionResult",
    "NLPAnalysisResult",
    "process_article_job",
    "load_document_from_url",
    "run_analysis",
    "chunk_embed_and_store",
]
