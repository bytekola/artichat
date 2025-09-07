"""
Semantic-search-first agent for article analysis and question answering.
This class uses centralized intelligence to reduce LLM calls by separating
data retrieval from data synthesis.
"""

import asyncio
import hashlib
import re
from ast import Str
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import redis.asyncio as redis
from app.core.config import settings
from chromadb.api import ClientAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langfuse.langchain import CallbackHandler


class ArticleChatAgent:
    """
    Intelligent agent for article analysis using an semantic-search-first strategy.
    The agent first retrieves all necessary context and then synthesizes an answer in a single step.
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
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"article:{url_hash}"
    
    def _extract_urls_from_query(self, query: str) -> List[str]:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, query)
    
    async def _get_article_from_redis_async(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.redis_client:
                return None
            article_key = self._generate_article_key(url)
            article_data = await self.redis_client.hgetall(article_key) # type: ignore
            if not article_data:
                return None
            return {k.decode() if isinstance(k, bytes) else k: 
                   v.decode() if isinstance(v, bytes) else v 
                   for k, v in article_data.items()}
        except Exception as e:
            print(f"Error retrieving article from Redis: {e}")
            return None

    async def _semantic_search_with_documents(self, query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any], Document]]:
        try:
            if not self.vectorstore:
                return []
            
            # Perform the similarity search
            results = await asyncio.to_thread(self.vectorstore.similarity_search, query, k)

            # Gather unique URLs from the search results
            urls = list(set(chunk_doc.metadata.get('source_url', chunk_doc.metadata.get('source', '')) for chunk_doc in results))
            
            # Fetch all article data from Redis concurrently
            tasks = [self._get_article_from_redis_async(url) for url in urls if url]
            articles_data = await asyncio.gather(*tasks)
            articles = {url: data for url, data in zip(urls, articles_data) if data}

            # Enrich the search results with the full article data
            enriched_results = []
            for chunk_doc in results:
                url = chunk_doc.metadata.get('source_url', chunk_doc.metadata.get('source', ''))
                if url in articles:
                    enriched_results.append((url, articles[url], chunk_doc))
            print(enriched_results)
            return enriched_results
        except Exception as e:
            print(f"Error during semantic search with documents: {e}")
            return []

    async def initialize(self):
        """Initialize the agent with LangChain components and Redis connection."""
        try:
            # Initialize Redis connection
            self.redis_client = await redis.from_url(settings.redis_url)
            
            # Initialize OpenAI components
            if settings.azure_openai_enabled and settings.azure_openai_api_key:
                # Use Azure OpenAI - set environment variables for LangChain to pick up
                import os
                os.environ["AZURE_OPENAI_API_KEY"] = settings.azure_openai_api_key
                os.environ["AZURE_OPENAI_ENDPOINT"] = settings.azure_openai_endpoint
                os.environ["AZURE_OPENAI_API_VERSION"] = settings.azure_openai_api_version
                
                from langchain_openai import (AzureChatOpenAI,
                                              AzureOpenAIEmbeddings)
                
                self.llm = AzureChatOpenAI(
                    azure_deployment=settings.azure_openai_chat_deployment,
                    temperature=0.1,
                    api_version=settings.azure_openai_api_version
                )
                
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=settings.azure_openai_embeddings_deployment,
                    api_version=settings.azure_openai_api_version
                )
            else:
                # Use regular OpenAI
                if settings.openai_api_key:
                    import os
                    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
                
                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1
                )
                
                self.embeddings = OpenAIEmbeddings()
            
            # Initialize Chroma client and vector store
            # Parse CHROMA_HOST to extract host and port
            chroma_url = settings.chroma_host.rstrip('/')
            if '://' in chroma_url:
                # Remove protocol if present
                chroma_url = chroma_url.split('://', 1)[1]
            
            # Split host and port (default to 8000 if not specified)
            if ':' in chroma_url:
                chroma_host, chroma_port = chroma_url.split(':', 1)
                chroma_port = int(chroma_port)
            else:
                chroma_host = chroma_url
                chroma_port = 8000
                
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="articles",
                embedding_function=self.embeddings
            )
            
            # Create set of tools
            tools = self._create_tools()
            
            # Create the system prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert article analysis assistant. Your goal is to provide the most relevant and accurate answer to the user.

METHODOLOGY:
1. For any user query (e.g., to summarize, compare, analyze sentiment), your first and primary step is to use the `retrieve_article_context` tool. Do not try to answer from memory.
2. This tool will provide you with several pieces of information for each relevant article: a semantically relevant excerpt, the full content, and any cached data like a general summary, keywords, or sentiment.
3. Once you have this context, you MUST synthesize a final, comprehensive answer by reasoning about the user's original query, which is: '{input}'

YOUR REASONING PROCESS:
- **For Summaries**: If the user asks for a general summary, first check if a `Cached Summary` is available and adequate. If it is, present that summary. If the user asks a specific question or for a focused summary (e.g., 'what does it say about topic X?'), the `Cached Summary` is likely too general. In this case, you MUST generate a new, custom answer based on the `Semantically Relevant Excerpt` and `Full Content`.
- **For Other Analyses (Sentiment, Keywords, Comparison)**: Use all the provided context (relevant excerpts, full content, and cached data) to formulate a complete answer that directly addresses the user's request.
- **Synthesize, Don't Just Report**: Your final answer should be a well-written synthesis of the information, not just a list of the data you received.

Always provide a direct, comprehensive answer based on the context you've been given."""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=7, max_execution_time=150)
            
            print("ArticleChatAgent initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing ArticleChatAgent: {e}")
            raise

    async def _get_article_by_url(self, url: str) -> str:
        """Get complete article content and metadata for a specific URL. Use this ONLY when a user provides a full, specific URL to retrieve a single document."""
        try:
            article_data = await self._get_article_from_redis_async(url)
            
            if not article_data:
                return f"Article not found in document store for URL: {url}"
            
            title = article_data.get('title', 'Unknown Title')
            content = article_data.get('content', '')
            summary = article_data.get('summary', '')
            keywords = article_data.get('keywords', [])
            
            result = f"**{title}**\nURL: {url}\n\n"
            if summary:
                result += f"**Summary:** {summary}\n\n"
            
            if keywords:
                keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                result += f"**Keywords:** {keywords_str}\n\n"
            
            result += f"**Content:**\n{content[:2000]}" + ("..." if len(content) > 2000 else "")
            return result
        except Exception as e:
            return f"Error retrieving article: {str(e)}"
        
    async def _retrieve_article_context(self, query: str) -> str:
        """
        Searches for and retrieves context from articles based on a user's query. 
        This is the primary tool and should be used for almost all questions, including requests 
        to summarize, compare, analyze sentiment, or find entities.
        """
        try:
            results = await self._semantic_search_with_documents(query, k=5)
            if not results:
                return "No articles were found that match your query."

            context_parts = []
            for i, (url, article_data, chunk_doc) in enumerate(results, 1):
                title = article_data.get('title', 'Unknown Title')
                summary = article_data.get('summary', 'Not available.')
                keywords = article_data.get('keywords', 'Not available.')
                sentiment = article_data.get('sentiment', 'Not available.')
                content = article_data.get('content', '')

                context_parts.append(
                    f"--- Context from Article {i} ---\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Semantically Relevant Excerpt (Reason for retrieval): {chunk_doc.page_content}\n"
                    f"Cached Summary: {summary}\n"
                    f"Cached Keywords: {keywords}\n"
                    f"Cached Sentiment: {sentiment}\n"
                    f"Full Content: {content}...\n"
                )
            
            return "\n".join(context_parts)
        except Exception as e:
            return f"An error occurred while retrieving article context: {str(e)}"

    async def process_question(self, question: str) -> str:
        """
        The agent uses a single data retrieval tool and a powerful LLM prompt for synthesis.
        """
        try:
            if not self.agent_executor:
                return "Agent not initialized. Please try again later."
            
            result = await self.agent_executor.ainvoke({"input": question}, config={"callbacks": [self.langfuse_handler]})
            return result.get("output", "Sorry, I couldn't process your question.")
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"


    def _create_tools(self) -> List[Tool]:
        """Creates list of tools for the agent, explicitly wrapping instance methods."""
        return [
            Tool.from_function(
                func=self._retrieve_article_context,
                coroutine=self._retrieve_article_context,
                name="retrieve_article_context",
                description=(
                    "Searches for and retrieves context from articles based on a user's query. "
                    "Use this for summarizing, comparing, analyzing sentiment, or finding entities."
                ),
            ),
            Tool.from_function(
                func=self._get_article_by_url,
                coroutine=self._get_article_by_url,
                name="get_article_by_url",
                description=(
                    "Get complete article content and metadata for a specific URL. "
                    "Use this ONLY when a user provides a full, specific URL to retrieve a single document."
                )
            )
        ]