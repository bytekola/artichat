from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, validator


class IngestionJob(BaseModel):
    """
    Pydantic schema for validating the job message from the queue.
    """
    url: HttpUrl = Field(..., description="URL of the article to ingest")
    priority: Optional[int] = Field(default=0, ge=0, le=10, description="Job priority (0-10)")
    retry_count: Optional[int] = Field(default=0, ge=0, description="Number of retry attempts")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Job creation timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ArticleMetadata(BaseModel):
    """
    Schema for article metadata extracted during processing.
    """
    source_url: str = Field(..., description="Original URL of the article")
    title: Optional[str] = Field(None, description="Article title")
    extracted_keywords: List[str] = Field(default_factory=list, description="Keywords extracted from the article")
    base_sentiment_label: Optional[str] = Field(None, description="Overall sentiment label (POSITIVE/NEGATIVE)")
    base_sentiment_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sentiment confidence score")
    language: Optional[str] = Field(default="en", description="Detected language of the article")
    
    @field_validator('base_sentiment_label')
    def validate_sentiment_label(cls, v):
        if v is not None:
            # Normalize to uppercase
            v = v.upper()
            if v not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                raise ValueError('Sentiment label must be POSITIVE, NEGATIVE, or NEUTRAL')
        return v


class DocumentChunk(BaseModel):
    """
    Schema for individual document chunks with embeddings.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    chunk_index: int = Field(..., ge=0, description="Index of the chunk within the document")
    source_url: str = Field(..., description="URL of the source article")
    metadata: ArticleMetadata = Field(..., description="Metadata for the parent article")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    
    class Config:
        arbitrary_types_allowed = True


class IngestionResult(BaseModel):
    """
    Schema for the result of article ingestion.
    """
    url: str = Field(..., description="URL that was processed")
    success: bool = Field(..., description="Whether ingestion was successful")
    chunks_created: Optional[int] = Field(None, ge=0, description="Number of chunks created")
    metadata: Optional[ArticleMetadata] = Field(None, description="Extracted metadata")
    error_message: Optional[str] = Field(None, description="Error message if ingestion failed")
    processing_time_seconds: Optional[float] = Field(None, ge=0.0, description="Time taken to process")
    created_at: datetime = Field(default_factory=datetime.now, description="Processing completion timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NLPAnalysisResult(BaseModel):
    """
    Schema for NLP analysis results.
    """
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted")
    sentiment_label: Optional[str] = Field(None, description="Sentiment classification")
    sentiment_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sentiment confidence")
    language_detected: Optional[str] = Field(None, description="Detected language")
    summary: Optional[str] = Field(None, description="AI-generated summary of the article")
    
    @validator('sentiment_label')
    def validate_sentiment(cls, v):
        if v is not None:
            # Normalize to uppercase
            v = v.upper()
            if v not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                raise ValueError('Invalid sentiment label')
        return v


class StoredArticle(BaseModel):
    """Schema for the full article document stored in Redis."""
    url: HttpUrl
    title: str
    full_text: str
    metadata: ArticleMetadata
    summary: Optional[str] = None
    stored_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CachedQuery(BaseModel):
    """Schema for a cached query-response pair in Redis."""
    query: str
    response: str
    cached_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobStatus(BaseModel):
    """Schema for job status tracking in Redis."""
    url: str
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    chunks_created: int = 0
    error_message: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VectorStoreChunkMetadata(BaseModel):
    """Schema for the metadata associated with a vector chunk in ChromaDB."""
    source_url: str
    document_key: str  # Key reference to Redis document
    chunk_index: int
    chunk_id: str
    chunk_size: int
    title: str
    sentiment_label: Optional[str] = None
    language: str = "en"