"""API routes for Artichat backend."""

from typing import Optional

from app.core.config import settings
from app.models.schemas import (ChatRequest, ChatResponse, IngestRequest,
                                IngestResponse, StatusResponse)
from fastapi import APIRouter, Depends, Header, HTTPException

router = APIRouter()


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """API key verification."""
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def get_cache_service():
    """Get cache service dependency."""
    from app.main import cache_service
    return cache_service


def get_agent_service():
    """Get agent service dependency."""
    from app.main import agent_service
    return agent_service


def get_ingestion_service():
    """Get ingestion service dependency."""
    from app.main import ingestion_service
    return ingestion_service


@router.post("/query", response_model=ChatResponse)
async def query(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
    cache_service=Depends(get_cache_service),
    agent_service=Depends(get_agent_service)
):
    """Answer a question with caching."""
    question = request.question
    
    # Try cache first
    cached_answer = await cache_service.get(question)
    if cached_answer:
        return ChatResponse(answer=cached_answer, cached=True)
    
    # Get answer from agent
    answer = await agent_service.process_question(question)
    
    # Cache the answer
    await cache_service.set(question, answer)
    
    return ChatResponse(answer=answer, cached=False)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    api_key: str = Depends(verify_api_key),
    ingestion_service=Depends(get_ingestion_service)
):
    """Queue article for processing by worker."""
    url = str(request.url)
    result = await ingestion_service.queue_job(url, request.force)
    
    return IngestResponse(
        message=result["message"],
        url=result["url"],
        status=result["status"]
    )


@router.get("/ingestion/status", response_model=StatusResponse)
async def ingestion_status(
    url: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
    ingestion_service=Depends(get_ingestion_service)
):
    """Get processing status."""
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter required")
    
    status = await ingestion_service.get_status(url)
    
    return StatusResponse(
        url=status["url"],
        status=status["status"],
        chunks_created=status["chunks_created"],
        error_message=status["error_message"]
    )
