import logging
from typing import Optional
from urllib.parse import urlparse

import chromadb
from chromadb.api import ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import ValidationError

from .config import settings
from .document_store import DocumentStore
from .schemas import ArticleMetadata, VectorStoreChunkMetadata


def _create_chroma_client() -> ClientAPI:
    """
    Build a Chroma HttpClient from CHROMA_HOST env which may include scheme and port.
    Examples:
      - http://chroma:8000
      - https://chroma.internal:8443
      - chroma (defaults to http://chroma:8000)
    """
    # Fallback to a default if the setting is not provided.
    raw_host = getattr(settings, "chroma_host", "http://localhost:8000") or "http://localhost:8000"

    # Ensure a scheme is present for consistent parsing.
    if "://" not in raw_host:
        raw_host = f"http://{raw_host}"

    try:
        parsed = urlparse(raw_host)
        
        if not parsed.hostname:
            raise ValueError("Hostname could not be parsed from the provided CHROMA_HOST.")

        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 8000)
        
        use_ssl = parsed.scheme == "https"

        return chromadb.HttpClient(host=host, port=port, ssl=use_ssl)
    except (ValueError, TypeError) as e:
        logging.warning(
            f"CHROMA_HOST '{getattr(settings, 'chroma_host', None)}' could not be parsed, "
            f"falling back to http://localhost:8000. Error: {e}"
        )
        return chromadb.HttpClient(host="localhost", port=8000, ssl=False)


def _create_optimized_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create an optimized text splitter using LangChain for article content.
    Uses separators optimized for news articles and web content.
    Configuration values are loaded from settings.
    """
    # Optimized separators for news articles
    separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamations
        "? ",    # Questions
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Spaces
        "",      # Characters
    ]
    
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


# Initialize clients once to be reused
chroma_client = _create_chroma_client()

# Select embeddings implementation (Azure or OpenAI) with improved error handling
def _initialize_embeddings():
    """Initialize embeddings with proper fallback handling."""
    try:
        if settings.azure_openai_enabled:
            if not (settings.azure_openai_endpoint and settings.azure_openai_embeddings_deployment and settings.azure_openai_api_key.get_secret_value()):
                logging.warning("AZURE_OPENAI_ENABLED is true, but endpoint/deployment/api_key are not fully set. Falling back to OpenAIEmbeddings.")
                return OpenAIEmbeddings(model=settings.openai_embeddings_model, api_key=settings.openai_api_key)
            else:
                return AzureOpenAIEmbeddings(
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    azure_deployment=settings.azure_openai_embeddings_deployment,
                )
        else:
            return OpenAIEmbeddings(model=settings.openai_embeddings_model, api_key=settings.openai_api_key)
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
        # Fallback to basic OpenAI embeddings with configured model
        return OpenAIEmbeddings(model=settings.openai_embeddings_model)

embeddings_model = _initialize_embeddings()

# Create optimized text splitter
text_splitter = _create_optimized_text_splitter()

# Use LangChain's Chroma vector store (consistent LangChain usage)
vectorstore = Chroma(
    collection_name=settings.chroma_articles_collection,
    embedding_function=embeddings_model,
    client=chroma_client,
)

def chunk_embed_and_store(
    full_text: str, 
    metadata: ArticleMetadata, 
    url: str,
    summary: Optional[str] = None,
    overwrite: bool = False
) -> int:
    """    
    1. Store complete article in Redis Document Store
    2. Break article into semantic chunks 
    3. Store chunks in ChromaDB Vector Store with references to Redis document
    
    This provides:
    - Fast whole-document retrieval (Redis)
    - Precise semantic search (ChromaDB chunks)
    - Efficient metadata linking
    
    Args:
        full_text: The complete article text
        metadata: Structured metadata object for the article
        url: The source URL of the article
        summary: AI-generated summary (optional)
        overwrite: If True, delete existing chunks before storing new ones (default: False)
        
    Returns:
        Number of chunks successfully stored
    """
    if not full_text or full_text.isspace():
        logging.warning(f"Skipping storage for empty content from URL: {url}")
        return 0    
    try:
        # Step 1: Store complete article in Redis Document Store
        document_key = DocumentStore.store_article(
            url=url,
            title=metadata.title or "Unknown Title",
            full_text=full_text,
            metadata=metadata,
            summary=summary
        )
        logging.info(f"📄 Stored complete article in Redis: {document_key}")
        
        # Step 2: Create semantic chunks using LangChain
        source_document = Document(
            page_content=full_text,
            metadata={
                "title": metadata.title,
                "language": metadata.language,
                "document_key": document_key,
            }
        )
        
        # Use LangChain text splitter to create semantic chunks
        document_chunks = text_splitter.split_documents([source_document])
        
        if not document_chunks:
            logging.warning(f"LangChain text splitting resulted in zero chunks for URL: {url}")
            return 0

        logging.info(f"✂️  Created {len(document_chunks)} semantic chunks for {url}")

        # Step 3: Store chunks in ChromaDB Vector Store with Redis references
        texts = []
        metadatas = []
        ids = []
        
        # Create Document objects with metadata for filtering
        documents_for_filtering = []
        
        for i, doc_chunk in enumerate(document_chunks):
            try:
                # Lightweight chunk metadata with Redis document reference
                chunk_metadata = VectorStoreChunkMetadata(
                    source_url=url,
                    document_key=document_key,
                    chunk_index=i,
                    chunk_id=f"{url}#{i}",
                    chunk_size=len(doc_chunk.page_content),
                    title=metadata.title or "",
                    sentiment_label=metadata.base_sentiment_label,
                    language=metadata.language or "en",
                )
                
                # Create Document with metadata for LangChain filtering
                chunk_metadata_dict = chunk_metadata.model_dump()
                filtered_doc = Document(
                    page_content=doc_chunk.page_content,
                    metadata=chunk_metadata_dict
                )
                documents_for_filtering.append(filtered_doc)
                ids.append(f"{url}#{i}")

            except ValidationError as e:
                logging.warning(f"Skipping chunk {i} due to validation error: {e}")
                continue
        
        if not documents_for_filtering:
            logging.error(f"All chunks failed validation for {url}. Aborting storage.")
            return 0
        
        try:
            filtered_documents = filter_complex_metadata(documents_for_filtering)
            texts = [doc.page_content for doc in filtered_documents]
            metadatas = [doc.metadata for doc in filtered_documents]
        except Exception as e:
            logging.warning(f"Error filtering metadata with LangChain utility: {e}. Using manual filtering.")
            # Fallback to manual filtering
            texts = []
            metadatas = []
            for doc in documents_for_filtering:
                texts.append(doc.page_content)
                # Manual filtering as fallback
                filtered_metadata = {k: v for k, v in doc.metadata.items() if v is not None and isinstance(v, (str, int, float, bool))}
                metadatas.append(filtered_metadata)
        
        if overwrite:
            try:
                existing_results = vectorstore.similarity_search(
                    query="", 
                    k=1000, 
                    filter={"source_url": url}
                )
                existing_ids = []
                for result in existing_results:
                    chunk_id = result.metadata.get("chunk_id")
                    if chunk_id:
                        existing_ids.append(chunk_id)
                
                if existing_ids:
                    vectorstore.delete(ids=existing_ids)
                    logging.info(f"🗑️  Cleaned up {len(existing_ids)} existing chunks for overwrite: {url}")
                else:
                    logging.info(f"🗑️  No existing chunks found to clean up for: {url}")
            except Exception as e:
                logging.warning(f"⚠️  Failed to clean up existing chunks (continuing anyway): {e}")
        
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        logging.info(f"Successfully stored {len(document_chunks)} chunks with Redis references")
        logging.info(f"Architecture: Redis Document ({document_key}) + ChromaDB Chunks ({len(document_chunks)})")
        
        return len(document_chunks)
        
    except Exception as e:
        logging.error(f"💥 Error in Best of Breed storage for {url}: {e}", exc_info=True)
        return 0

def verify_storage_health() -> bool:
    """
    Verify that the ChromaDB connection and vectorstore are healthy.
    
    Returns:
        True if storage is healthy, False otherwise
    """
    try:
        # Test ChromaDB connection
        heartbeat = chroma_client.heartbeat()
        
        # Test vectorstore operations
        collection = vectorstore._collection
        count = collection.count()
        
        logging.info(f"💖 Storage health check passed: ChromaDB heartbeat={heartbeat}, collection count={count}")
        return True
        
    except Exception as e:
        logging.error(f"Storage health check failed: {e}")
        return False
