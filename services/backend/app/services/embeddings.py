"""Simple OpenAI embeddings service with Azure OpenAI support."""

from app.core.config import settings


async def get_embedding(text: str) -> list[float]:
    """Get embedding for text using OpenAI or Azure OpenAI."""
    # Choose client based on configuration
    if settings.azure_openai_enabled and settings.azure_openai_api_key and settings.azure_openai_endpoint:
        # Use Azure OpenAI
        from openai import AsyncAzureOpenAI
        
        client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version
        )
        model = settings.azure_openai_embeddings_deployment
    else:
        # Use regular OpenAI
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = settings.embedding_model
    
    response = await client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding
