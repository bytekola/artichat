"""Configuration settings for the LangChain-powered ingestion worker."""
import os
from functools import lru_cache

import redis
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables with LangChain optimizations."""
    
    # RabbitMQ Configuration
    rabbitmq_uri: str = Field(
        default_factory=lambda: os.getenv("RABBITMQ_URI", "amqp://guest:guest@localhost:5672/"),
        description="RabbitMQ connection URI"
    )
    rabbitmq_queue: str = Field(
        default_factory=lambda: os.getenv("RABBITMQ_QUEUE", "ingestion_queue"),
        description="RabbitMQ queue name for ingestion jobs"
    )
    
    # ChromaDB Configuration
    chroma_host: str = Field(
        default_factory=lambda: os.getenv("CHROMA_HOST", "http://localhost:8000"),
        description="ChromaDB host URL"
    )
    chroma_articles_collection: str = Field(
        default_factory=lambda: os.getenv("CHROMA_ARTICLES_COLLECTION", "articles"),
        description="ChromaDB collection name for articles"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        description="Redis connection URL for status tracking"
    )
    
    # OpenAI Configuration
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")),
        description="OpenAI API key for embeddings and LLM"
    )
    
    # Azure OpenAI Configuration
    azure_openai_enabled: bool = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENABLED", "false").lower() in ["1", "true", "yes"],
        description="Enable Azure OpenAI for embeddings and LLM"
    )
    azure_openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AZURE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
        description="Azure OpenAI API key (or reuse OPENAI_API_KEY)"
    )
    azure_openai_endpoint: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        description="Azure OpenAI endpoint, e.g. https://your-resource.openai.azure.com"
    )
    azure_openai_api_version: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        description="Azure OpenAI API version"
    )
    azure_openai_embeddings_deployment: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", ""),
        description="Azure OpenAI embeddings deployment name"
    )

    # Analysis Mode Configuration
    analysis_mode: str = Field(
        default_factory=lambda: os.getenv("ANALYSIS_MODE", "deterministic").lower(),
        description="Analysis mode: 'deterministic' (fast, local) or 'llm' (AI-powered, requires API key)"
    )
    
    # LLM Configuration (only used when analysis_mode = 'llm')
    openai_llm_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        description="OpenAI model for LLM analysis (when not using Azure)"
    )
    openai_summary_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
        description="OpenAI model for summary generation"
    )
    openai_embeddings_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
        description="OpenAI model for text embeddings"
    )
    azure_openai_chat_deployment: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""),
        description="Azure OpenAI chat/completions deployment name for LLM analysis"
    )
    azure_openai_summary_deployment: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_SUMMARY_DEPLOYMENT", ""),
        description="Azure OpenAI deployment name for summary generation (fallback to chat deployment)"
    )
    llm_max_input_chars: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_INPUT_CHARS", "20000")),
        description="Maximum characters of input text to send to the LLM"
    )
    
    # LangChain Text Splitting Configuration
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")),
        description="Size of text chunks for LangChain text splitter"
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")),
        description="Overlap between text chunks for LangChain text splitter"
    )
    
    # Performance Configuration
    max_concurrent_jobs: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_JOBS", "1")),
        description="Maximum number of concurrent jobs to process"
    )
    job_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("JOB_TIMEOUT_SECONDS", "300")),
        description="Timeout for individual job processing in seconds"
    )
    
    class Config:
        env_file = ".env"
    
    def model_post_init(self, __context):
        """Validate configuration after initialization."""
        valid_modes = ["deterministic", "llm"]
        if self.analysis_mode not in valid_modes:
            raise ValueError(f"Invalid analysis_mode '{self.analysis_mode}'. Must be one of: {valid_modes}")


settings = Settings()


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """Get a cached Redis client instance."""
    return redis.from_url(settings.redis_url, decode_responses=True)
