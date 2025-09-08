"""
Artichat Python Backend

FastAPI service that provides:
1. /v1/query - Answer questions with caching  
2. /v1/ingest - Queue articles for worker processing
3. /v1/ingestion/status - Check processing status
4. /health - Health check

Architecture:
- FastAPI for REST API
- Redis for L1 cache (exact matches)
- ChromaDB for L2 semantic cache + article retrieval
- RabbitMQ to communicate with existing worker service
- OpenAI for embeddings and LLM
"""

import time
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.config import settings
from app.middleware.security import SecurityMiddleware
from app.services.agent import ArticleChatAgent
from app.services.cache import CacheService
from app.services.ingestion import IngestionService
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Global services
cache_service = CacheService()
agent_service = ArticleChatAgent()
ingestion_service = IngestionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 Starting Artichat Python Backend...")
    
    try:
        await cache_service.connect()
        await agent_service.initialize()
        await ingestion_service.connect()
        print("✅ All services initialized")
        
        # Queue predefined articles for processing
        predefined_urls = [
            "https://techcrunch.com/2025/07/26/astronomer-winks-at-viral-notoriety-with-temporary-spokesperson-gwyneth-paltrow/",
            "https://techcrunch.com/2025/07/26/allianz-life-says-majority-of-customers-personal-data-stolen-in-cyberattack/",
            "https://techcrunch.com/2025/07/27/itch-io-is-the-latest-marketplace-to-crack-down-on-adult-games/",
            "https://techcrunch.com/2025/07/26/tesla-vet-says-that-reviewing-real-products-not-mockups-is-the-key-to-staying-innovative/",
            "https://techcrunch.com/2025/07/25/meta-names-shengjia-zhao-as-chief-scientist-of-ai-superintelligence-unit/",
            "https://techcrunch.com/2025/07/26/dating-safety-app-tea-breached-exposing-72000-user-images/",
            "https://techcrunch.com/2025/07/25/sam-altman-warns-theres-no-legal-confidentiality-when-using-chatgpt-as-a-therapist/",
            "https://techcrunch.com/2025/07/25/intel-is-spinning-off-its-network-and-edge-group/",
            "https://techcrunch.com/2025/07/27/wizard-of-oz-blown-up-by-ai-for-giant-sphere-screen/",
            "https://techcrunch.com/2025/07/27/doge-has-built-an-ai-tool-to-slash-federal-regulations/",
            "https://edition.cnn.com/2025/07/27/business/us-china-trade-talks-stockholm-intl-hnk",
            "https://edition.cnn.com/2025/07/27/business/trump-us-eu-trade-deal",
            "https://edition.cnn.com/2025/07/27/business/eu-trade-deal",
            "https://edition.cnn.com/2025/07/26/tech/daydream-ai-online-shopping",
            "https://edition.cnn.com/2025/07/25/tech/meta-ai-superintelligence-team-who-its-hiring",
            "https://edition.cnn.com/2025/07/25/tech/sequoia-islamophobia-maguire-mamdani",
            "https://edition.cnn.com/2025/07/24/tech/intel-layoffs-15-percent-q2-earnings"
        ]
        
        print(f"📥 Queueing {len(predefined_urls)} predefined articles for processing...")
        queued_count = 0
        
        for url in predefined_urls:
            try:
                result = await ingestion_service.queue_job(url, overwrite=False)
                if result.get("status") == "accepted":
                    queued_count += 1
                    print(f"✅ Queued: {url}")
                else:
                    print(f"⚠️ Skipped: {url} - {result.get('message', 'Unknown status')}")
            except Exception as e:
                print(f"❌ Failed to queue {url}: {e}")
        
        print(f"🎉 Successfully queued {queued_count}/{len(predefined_urls)} predefined articles for processing")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown  
    print("🔄 Shutting down...")
    await cache_service.close()
    await ingestion_service.close()
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Artichat API",
    description="REST API for AI-powered article question-answering system",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(SecurityMiddleware)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Request logging middleware."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions."""
    print(f"❌ Error in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "artichat-python"}


# Include routes
app.include_router(router, prefix="/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8082, reload=True)
