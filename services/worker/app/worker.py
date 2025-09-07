"""
Simplified LangChain-powered article ingestion worker.

Processes article ingestion jobs from RabbitMQ:
1. Load article content using LangChain WebBaseLoader
2. Run configurable NLP analysis (deterministic or LLM)  
3. Chunk text and store embeddings in ChromaDB via LangChain
4. Track status in Redis cache
"""

import logging
from datetime import datetime
from typing import Optional

import redis
from pydantic import ValidationError

from app.config import get_redis_client, settings
from app.document_store import DocumentStore
from app.enrichment import run_analysis
from app.loader import load_document_from_url
from app.schemas import ArticleMetadata, IngestionJob, IngestionResult, JobStatus
from app.storage import chunk_embed_and_store

logger = logging.getLogger(__name__)


class StatusTracker:
    """Simple status tracking helper with deduplication support."""
    
    @staticmethod
    def acquire_processing_lock(url: str, timeout_seconds: int = 300) -> bool:
        """
        Acquire a distributed lock for processing this URL.
        Prevents multiple workers from processing the same article simultaneously.
        
        Args:
            url: The article URL to lock
            timeout_seconds: Lock timeout in seconds (default 5 minutes)
            
        Returns:
            True if lock acquired, False if already locked by another worker
        """
        lock_key = f"processing_lock:{url}"
        try:
            redis_client = get_redis_client()
            result = redis_client.set(lock_key, "locked", nx=True, ex=timeout_seconds)
            if result:
                logger.info(f"🔒 Acquired processing lock for {url}")
                return True
            else:
                logger.warning(f"🔒 Processing lock already exists for {url} - skipping")
                return False
        except redis.RedisError as e:
            logger.warning(f"⚠️ Failed to acquire lock for {url}: {e}")
            # In case of Redis failure, proceed without lock to avoid blocking
            return True
    
    @staticmethod
    def release_processing_lock(url: str):
        """Release the processing lock for this URL."""
        lock_key = f"processing_lock:{url}"
        try:
            redis_client = get_redis_client()
            redis_client.delete(lock_key)
            logger.info(f"🔓 Released processing lock for {url}")
        except redis.RedisError as e:
            logger.warning(f"⚠️ Failed to release lock for {url}: {e}")
    
    @staticmethod
    def update(url: str, status: str, chunks_created: int = 0, error_message: Optional[str] = None):
        """Update job status in Redis."""
        try:
            status_data = JobStatus(
                url=url,
                status=status,
                chunks_created=chunks_created,
                error_message=error_message,
            )
            redis_client = get_redis_client()
            redis_client.set(f"ingestion:{url}", status_data.model_dump_json())
            logger.info(f"Status: {url} → {status}")
        except (ValidationError, redis.RedisError) as e:
            logger.warning(f"⚠️ Status update failed: {e}")


def process_article_job(raw_body: bytes) -> Optional[IngestionResult]:
    """
    Process a single article ingestion job with distributed locking.
    
    Pipeline: Parse job → Acquire lock → Load content → Run analysis → Store chunks → Return result
    
    Args:
        raw_body: Raw RabbitMQ message containing the ingestion job
        
    Returns:
        IngestionResult with processing details
    """
    start_time = datetime.now()
    url = "unknown"
    lock_acquired = False
    
    try:
        # 1. Parse and validate job
        job = IngestionJob.model_validate_json(raw_body)
        url = str(job.url)
        job_id = f"job-{hash(url)}"
        
        logger.info(f"📥 Processing {job_id} for {url}")
        
        # 2. Acquire distributed processing lock to prevent duplicate processing
        if not StatusTracker.acquire_processing_lock(url):
            logger.info(f"⏭️  Skipping {url} - already being processed by another worker")
            return IngestionResult(
                url=url,
                success=True,  # Not a failure, just skipped
                error_message="Skipped - already being processed",
                chunks_created=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata=None
            )
        
        lock_acquired = True
        StatusTracker.update(url, "processing")
        
        # 2.5. Check if article already exists (unless overwrite is requested)
        if not job.overwrite and DocumentStore.article_exists(url):
            logger.info(f"⏭️  Skipping {url} - article already exists (overwrite=False)")
            StatusTracker.release_processing_lock(url)
            lock_acquired = False
            return IngestionResult(
                url=url,
                success=True,
                error_message="Skipped - article already exists",
                chunks_created=0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                metadata=None
            )
        
        # 3. Load article content
        content, metadata_dict = load_document_from_url(url)
        if not content:
            return _fail_job(url, "Failed to load article content", start_time)
        
        # Create metadata object
        metadata_dict = metadata_dict or {}
        article_metadata = ArticleMetadata(
            source_url=url,
            title=metadata_dict.get('title', 'Unknown Title'),
            language=metadata_dict.get('language', 'en'),
            base_sentiment_label="neutral",
            base_sentiment_score=0.0
        )
        
        title_preview = (article_metadata.title or "")[:100]
        logger.info(f"Document loaded: {title_preview}... ({len(content)} chars)")
        
        # 4. Run NLP analysis
        logger.info(f"Running {settings.analysis_mode} analysis")
        nlp_result = run_analysis(
            full_text=content,
            external_request_id=job_id,
            extra_context={"url": url, "title": article_metadata.title or ""}
        )
        
        # Merge analysis results into metadata
        article_metadata.extracted_keywords = nlp_result.keywords or []
        article_metadata.base_sentiment_label = nlp_result.sentiment_label
        article_metadata.base_sentiment_score = nlp_result.sentiment_score
        if nlp_result.language_detected:
            article_metadata.language = nlp_result.language_detected
            
        logger.info(f"Analysis completed: {len(nlp_result.keywords)} keywords, {nlp_result.sentiment_label}")
        
        # 5. Store article chunks
        logger.info("🗄️ Storing in Redis + ChromaDB")
        chunks_created = chunk_embed_and_store(
            full_text=content,
            metadata=article_metadata,
            url=url,
            summary=nlp_result.summary,
            overwrite=job.overwrite or False
        )
        
        if chunks_created > 0:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processing completed: {chunks_created} chunks stored in {processing_time:.2f}s")
            StatusTracker.update(url, "completed", chunks_created=chunks_created)
            return IngestionResult(
                url=url,
                success=True,
                chunks_created=chunks_created,
                metadata=article_metadata,
                processing_time_seconds=processing_time,
                error_message=None
            )
        else:
            return _fail_job(url, "No chunks created during storage", start_time)
            
    except ValidationError as e:
        logger.error(f"ERROR: Invalid job format: {e}")
        return IngestionResult(
            url="unknown",
            success=False,
            error_message=f"Invalid job format: {e}",
            chunks_created=0,
            processing_time_seconds=(datetime.now() - start_time).total_seconds(),
            metadata=None
        )
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Processing failed: {str(e)}"
        logger.error(f"💥 {error_msg} for {url} after {processing_time:.2f}s", exc_info=True)
        StatusTracker.update(url, "failed", error_message=error_msg)
        return IngestionResult(
            url=url,
            success=False,
            error_message=error_msg,
            chunks_created=0,
            processing_time_seconds=processing_time,
            metadata=None
        )
    finally:
        # Always release the lock when done
        if lock_acquired:
            StatusTracker.release_processing_lock(url)


def _fail_job(url: str, error_message: str, start_time: datetime) -> IngestionResult:
    """Helper to create a failed job result."""
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.warning(f"⚠️ Job failed: {error_message} for {url}")
    StatusTracker.update(url, "failed", error_message=error_message)
    return IngestionResult(
        url=url,
        success=False,
        error_message=error_message,
        chunks_created=0,
        processing_time_seconds=processing_time,
        metadata=None
    )


