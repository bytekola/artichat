"""Pydantic models for Artichat API."""

from typing import Optional

from pydantic import BaseModel, HttpUrl


class ChatRequest(BaseModel):
    """Chat request - compatible with Go backend."""
    question: str


class ChatResponse(BaseModel):
    """Chat response - compatible with Go backend."""
    answer: str
    cached: bool = False


class IngestRequest(BaseModel):
    """Ingest request."""
    url: HttpUrl
    force: bool = False


class IngestResponse(BaseModel):
    """Ingest response."""
    message: str
    url: str
    status: str


class StatusResponse(BaseModel):
    """Status response."""
    url: str
    status: str  # processing, completed, failed
    chunks_created: int = 0
    error_message: Optional[str] = None
