"""Security middleware for the ArticleChat backend."""

import time
from typing import Optional

from app.core.config import settings
from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware providing:
    - Security headers
    - Request size limiting
    - Basic security validations
    """
    
    def __init__(self, app, max_request_size: int = 64 * 1024):  # 64KB default
        super().__init__(app)
        self.max_request_size = max_request_size
        self.settings = settings
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware."""
        
        # Check request size
        if hasattr(request, "headers"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                print(f"⚠️  Request too large: {content_length} bytes (max: {self.max_request_size})")
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request entity too large", "max_size": self.max_request_size}
                )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent content type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy (restrictive for API)
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none';"
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]


class APIKeyMiddleware:
    """API key authentication middleware."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def __call__(self, request: Request, call_next):
        """Check API key authentication."""
        
        # Skip auth for certain endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Check API key
        provided_key = request.headers.get("X-API-Key")
        
        if not provided_key:
            print(f"⚠️  Missing API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"error": "API key required", "details": "Include X-API-Key header"}
            )
        
        if provided_key != self.api_key:
            print(f"⚠️  Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )
        
        # API key is valid, proceed
        return await call_next(request)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no auth required)."""
        public_endpoints = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics"
        ]
        
        return any(path.startswith(endpoint) for endpoint in public_endpoints)
