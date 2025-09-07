"""Configuration for Artichat backend."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API
    api_key: str = os.getenv("API_KEY", "your-api-key-here")
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Azure OpenAI Configuration
    azure_openai_enabled: bool = os.getenv("AZURE_OPENAI_ENABLED", "false").lower() in ["1", "true", "yes"]
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_openai_embeddings_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "")
    azure_openai_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    
    # Storage
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    chroma_host: str = os.getenv("CHROMA_HOST", "http://localhost:8000")
    
    # RabbitMQ (for worker communication)
    rabbitmq_url: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    rabbitmq_queue: str = os.getenv("RABBITMQ_QUEUE", "ingestion_queue")
    
    # Cache
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))
    
    # Langfuse Configuration
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    class Config:
        env_file = ".env"


settings = Settings()
