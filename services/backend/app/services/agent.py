"""
Semantic-search-first agent for article analysis and question answering.
This class uses centralized intelligence to reduce LLM calls by separating
data retrieval from data synthesis.
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional

import chromadb
import redis.asyncio as redis
from app.core.config import settings
from chromadb.api import ClientAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field


class EnrichmentInput(BaseModel):
    source_urls: List[str] = Field(description="The list of complete article URLs to fetch metadata for.")
    fields_to_include: List[str] = Field(
        description="The specific metadata fields to retrieve. Valid options are: 'title', 'summary', 'keywords', 'sentiment', 'content'."
    )


class ArticleChatAgent:
    """
    Intelligent agent for article analysis using a Retrieval-Augmented Generation (RAG) strategy.
    The agent first retrieves relevant text chunks and then can enrich this context with
    full document metadata—including the full article content—before synthesizing a final answer.
    """

    def __init__(self):
        self.llm: Optional[BaseLanguageModel] = None
        self.embeddings: Optional[Embeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.chroma_client: Optional[ClientAPI] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.redis_client: Optional[redis.Redis] = None
        self.langfuse_handler = CallbackHandler()


    def _generate_article_key(self, url: str) -> str:
        url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        return f"article:{url_hash}"

    async def _get_article_from_redis_async(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.redis_client:
                return None
            article_key = self._generate_article_key(url)
            article_data_raw = await self.redis_client.get(article_key)
            if not article_data_raw:
                return None
            import json
            if isinstance(article_data_raw, bytes):
                return json.loads(article_data_raw.decode('utf-8'))
            return json.loads(article_data_raw)
        except Exception as e:
            print(f"Error retrieving article from Redis: {e}")
            return None

    async def initialize(self):
        """Initialize the agent with LangChain components and Redis connection."""
        try:
            self.redis_client = await redis.from_url(settings.redis_url)

            if settings.azure_openai_enabled and settings.azure_openai_api_key:
                import os
                os.environ["AZURE_OPENAI_API_KEY"] = settings.azure_openai_api_key
                os.environ["AZURE_OPENAI_ENDPOINT"] = settings.azure_openai_endpoint
                os.environ["AZURE_OPENAI_API_VERSION"] = settings.azure_openai_api_version
                from langchain_openai import (AzureChatOpenAI,
                                              AzureOpenAIEmbeddings)
                self.llm = AzureChatOpenAI(azure_deployment=settings.azure_openai_chat_deployment, temperature=0.1, api_version=settings.azure_openai_api_version)
                self.embeddings = AzureOpenAIEmbeddings(azure_deployment=settings.azure_openai_embeddings_deployment, api_version=settings.azure_openai_api_version)
            else:
                if settings.openai_api_key:
                    import os
                    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                self.embeddings = OpenAIEmbeddings()

            chroma_url = settings.chroma_host.rstrip('/')
            if '://' in chroma_url:
                chroma_url = chroma_url.split('://', 1)[1]
            if ':' in chroma_url:
                chroma_host, chroma_port_str = chroma_url.split(':', 1)
                chroma_port = int(chroma_port_str)
            else:
                chroma_host = chroma_url
                chroma_port = 8000
            self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self.vectorstore = Chroma(client=self.chroma_client, collection_name="articles", embedding_function=self.embeddings)

            tools = self._create_tools()
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert article analysis assistant. Your goal is to provide accurate answers by following a structured RAG (Retrieval-Augmented Generation) process.

METHODOLOGY:
1.  **Step 1: Search for Context.** For any user query, your first action is ALWAYS to use the `semantic_search_for_chunks` tool.
2.  **Step 2: Analyze and Decide.** Review the search results. If the information is insufficient, proceed to the next step.
3.  **Step 3: Enrich with Metadata (If Necessary).** Use the `enrich_chunks_with_metadata` tool if you need more details. You MUST provide both the `source_urls` and the `fields_to_include` arguments.
4.  **Step 4: Synthesize Final Answer.** Combine all gathered information to construct a comprehensive answer to the original query: '{input}'."""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])

            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10, max_execution_time=None)
            print("ArticleChatAgent initialized successfully.")
        except Exception as e:
            print(f"Error initializing ArticleChatAgent: {e}")
            raise

    async def _semantic_search_for_chunks(self, query: str) -> str:
        """
        Performs semantic search to find relevant text chunks from articles. This is the first step for any query.
        """
        if not self.vectorstore:
            return "Vectorstore is not available."
        try:
            results = await asyncio.to_thread(self.vectorstore.similarity_search, query, k=5)
            if not results:
                return "No relevant article chunks were found for your query."
            context_parts = [
                f"--- Chunk {i} ---\nSource URL: {doc.metadata.get('source_url', 'Unknown URL')}\nRelevant Content: \"{doc.page_content}\"\n"
                for i, doc in enumerate(results, 1)
            ]
            return "\n".join(context_parts)
        except Exception as e:
            return f"Error during semantic search: {str(e)}"

    async def _enrich_chunks_with_metadata(self, source_urls: List[str], fields_to_include: List[str]) -> str:
        """
        Retrieves specific metadata fields for a given list of source URLs from the document store.
        Use this tool only after 'semantic_search_for_chunks' if the initial chunks are insufficient.
        """

        if not source_urls:
            return "No source URLs provided for enrichment."

        allowed_fields = {'title', 'summary', 'keywords', 'sentiment', 'content'}
        invalid_fields = set(fields_to_include) - allowed_fields
        if invalid_fields:
            return f"Error: Invalid field(s) requested: {', '.join(invalid_fields)}. Allowed fields are: {', '.join(allowed_fields)}"

        tasks = [self._get_article_from_redis_async(url) for url in source_urls]
        articles_data = await asyncio.gather(*tasks)
        enriched_parts = []
        for i, data in enumerate(articles_data, 1):
            url = source_urls[i-1]
            if not data:
                enriched_parts.append(f"--- Article from URL {url} ---\nCould not retrieve data.\n")
                continue
            part = f"--- Enriched Data for URL: {url} ---\n"
            for field in fields_to_include:
                # Handle special field mappings
                if field == 'sentiment':
                    # Extract sentiment from metadata
                    metadata = data.get('metadata', {})
                    sentiment_label = metadata.get('base_sentiment_label', 'Unknown')
                    sentiment_score = metadata.get('base_sentiment_score', 'Unknown')
                    value = f"{sentiment_label} (score: {sentiment_score})"
                elif field == 'keywords':
                    # Extract keywords from metadata
                    metadata = data.get('metadata', {})
                    keywords_list = metadata.get('extracted_keywords', [])
                    value = ', '.join(keywords_list) if keywords_list else 'No keywords available'
                elif field == 'content':
                    # Map to full_text field
                    value = data.get('full_text', 'Not available')
                else:
                    # Direct field access for title, summary
                    value = data.get(field, 'Not available')
                
                part += f"  - {field.capitalize()}: {value}\n"
            enriched_parts.append(part)
        return "\n".join(enriched_parts)

    async def process_question(self, question: str) -> str:
        """The agent uses a multi-step RAG process to answer questions."""
        try:
            if not self.agent_executor:
                return "Agent not initialized. Please try again later."
            result = await self.agent_executor.ainvoke({"input": question}, config={"callbacks": [self.langfuse_handler]})
            return result.get("output", "Sorry, I couldn't process your question.")
        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def _create_tools(self) -> List[BaseTool]:
        """Creates a list of tools for the agent."""
        return [
            Tool(
                name="semantic_search_for_chunks",
                func=self._semantic_search_for_chunks,
                coroutine=self._semantic_search_for_chunks,
                description="Searches for relevant text chunks. Always use this tool first.",
            ),
            StructuredTool.from_function(
                func=self._enrich_chunks_with_metadata,
                coroutine=self._enrich_chunks_with_metadata,
                name="enrich_chunks_with_metadata",
                description="Retrieves specific metadata fields (Allowed values are: summary, content, sentiment, keywords, title) for a list of URLs.",
                args_schema=EnrichmentInput,
            )
        ]