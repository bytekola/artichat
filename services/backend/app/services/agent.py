"""
Semantic-search-first agent for article analysis and question answering.
"""

import asyncio
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import redis.asyncio as redis
from app.core.config import settings
from chromadb.api import ClientAPI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ArticleChatAgent:
    """Intelligent agent for article analysis using semantic-search-first strategy."""
    
    def __init__(self):
        self.llm: Optional[BaseLanguageModel] = None
        self.embeddings: Optional[Embeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.chroma_client: Optional[ClientAPI] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.redis_client: Optional[redis.Redis] = None
        
    def _generate_article_key(self, url: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"article:{url_hash}"
    
    def _extract_urls_from_query(self, query: str) -> List[str]:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, query)
    
    def _get_article_from_redis_sync(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.redis_client:
                return None
                
            article_key = self._generate_article_key(url)
            import redis as sync_redis
            sync_client = sync_redis.from_url(settings.redis_url)
            article_data = sync_client.hgetall(article_key)
            
            if not article_data:
                return None
            
            if not isinstance(article_data, dict):
                print(f"Unexpected data type from Redis: {type(article_data)}")
                return None
                
            return {k.decode() if isinstance(k, bytes) else k: 
                   v.decode() if isinstance(v, bytes) else v 
                   for k, v in article_data.items()}
        except Exception as e:
            print(f"Error retrieving article from Redis: {e}")
            return None

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
            return None

    async def _semantic_search_with_documents(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any], Document]]:
        try:
            if not self.vectorstore:
                return []
            
            results = await asyncio.to_thread(self.vectorstore.similarity_search, query, k)
            
            urls = [chunk_doc.metadata.get('source_url', chunk_doc.metadata.get('source', '')) for chunk_doc in results]
            articles = {}
            for url in urls:
                if url:
                    article_data = await self._get_article_from_redis_async(url)
                    if article_data:
                        articles[url] = article_data
            
            enriched_results = []
            for chunk_doc, url in zip(results, urls):
                if url and url in articles:
                    enriched_results.append((url, articles[url], chunk_doc))
                else:
                    fallback_data = {
                        'title': chunk_doc.metadata.get('title', 'Unknown'),
                        'content': chunk_doc.page_content,
                        'url': url,
                        'summary': chunk_doc.metadata.get('summary', ''),
                        'keywords': chunk_doc.metadata.get('keywords', []),
                        'sentiment': chunk_doc.metadata.get('sentiment', '')
                    }
                    enriched_results.append((url, fallback_data, chunk_doc))
            
            return enriched_results
        except Exception:
            return []

    async def initialize(self):
        """Initialize the agent with LangChain components and Redis connection."""
        try:
            # Initialize Redis connection
            self.redis_client = await redis.from_url(settings.redis_url)
            
            # Initialize OpenAI components
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1
            )
            
            self.embeddings = OpenAIEmbeddings()
            
            # Initialize Chroma client and vector store
            self.chroma_client = chromadb.HttpClient(
                host="localhost",
                port=8000
            )
            
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name="articles",
                embedding_function=self.embeddings
            )
            
            # Create tools and agent
            tools = self._create_tools()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an intelligent article analysis assistant using semantic-search-first approach.

METHODOLOGY:
1. Start with semantic search to find relevant content chunks
2. For each relevant chunk, access the complete document from Redis
3. Synthesize answers using both semantic relevance and full document context

Your tools provide:
- Semantic relevance (why content matches the query)
- Complete document data (full text, cached metadata)
- Rich synthesis combining both contexts

Always provide comprehensive, well-contextualized answers that show both the specific relevance and broader document context."""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            agent = create_openai_tools_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            print("ArticleChatAgent initialized successfully with semantic-search-first strategy")
            
        except Exception as e:
            print(f"Error initializing ArticleChatAgent: {e}")
            raise

    async def _get_article_by_url(self, url: str) -> str:
        try:
            article_data = await self._get_article_from_redis_async(url)
            
            if not article_data:
                return f"Article not found in document store for URL: {url}"
            
            title = article_data.get('title', 'Unknown Title')
            content = article_data.get('content', article_data.get('full_text', ''))
            summary = article_data.get('summary', '')
            keywords = article_data.get('keywords', [])
            sentiment = article_data.get('sentiment', '')
            
            result = f"**{title}**\nURL: {url}\n\n"
            
            if summary:
                result += f"**Summary:** {summary}\n\n"
            
            if keywords:
                if isinstance(keywords, list):
                    result += f"**Keywords:** {', '.join(keywords)}\n\n"
                else:
                    result += f"**Keywords:** {keywords}\n\n"
            
            if sentiment:
                result += f"**Sentiment:** {sentiment}\n\n"
            
            result += f"**Content:**\n{content[:2000]}" + ("..." if len(content) > 2000 else "")
            
            return result
            
        except Exception as e:
            return f"Error retrieving article: {str(e)}"

    async def _get_article_summary_semantic_first(self, query: str) -> str:
        try:
            urls = self._extract_urls_from_query(query)
            if urls:
                summaries = []
                for url in urls:
                    article_data = await self._get_article_from_redis_async(url)
                    if article_data:
                        title = article_data.get('title', 'Unknown Title')
                        summary = article_data.get('summary', '')
                        if summary:
                            summaries.append(f"**{title}**\nURL: {url}\n\n{summary}")
                        else:
                            summaries.append(f"**{title}**\nURL: {url}\n\nNo cached summary available.")
                if summaries:
                    return "\n\n---\n\n".join(summaries)
            
            results = await self._semantic_search_with_documents(query, k=3)
            if not results:
                return "No articles found matching your query."
            
            summaries = []
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                cached_summary = article_data.get('summary', '')
                
                result = f"**{title}**\nURL: {url}\n"
                result += f"**Why relevant:** {chunk_doc.page_content[:200]}...\n\n"
                
                if cached_summary:
                    result += f"**Summary:** {cached_summary}"
                else:
                    full_content = article_data.get('content', article_data.get('full_text', ''))
                    if self.llm and full_content:
                        prompt = f"""Summarize this article focusing on \"{query}\":\n\nTitle: {title}\nRelevant excerpt: {chunk_doc.page_content}\nFull content: {full_content[:1500]}\n\nProvide a focused summary:"""
                        try:
                            response = self.llm.invoke(prompt)
                            summary = response.content if hasattr(response, 'content') else str(response)
                            result += f"**Summary:** {summary}"
                        except Exception as e:
                            result += f"**Summary:** Error generating summary: {str(e)}"
                    else:
                        result += f"**Summary:** No summary available."
                
                summaries.append(result)
            
            return "\n\n---\n\n".join(summaries)
            
        except Exception as e:
            return f"Error getting summaries: {str(e)}"

    async def _analyze_sentiment_semantic_first(self, query: str) -> str:
        try:
            urls = self._extract_urls_from_query(query)
            if urls:
                analyses = []
                for url in urls:
                    article_data = await self._get_article_from_redis_async(url)
                    if article_data:
                        title = article_data.get('title', 'Unknown Title')
                        sentiment = article_data.get('sentiment', '')
                        if sentiment:
                            analyses.append(f"**{title}**\nURL: {url}\n\nSentiment: {sentiment}")
                        else:
                            analyses.append(f"**{title}**\nURL: {url}\n\nNo cached sentiment analysis available.")
                if analyses:
                    return "\n\n---\n\n".join(analyses)
            
            # Semantic search first approach
            results = await self._semantic_search_with_documents(query, k=3)
            if not results:
                return "No articles found for sentiment analysis."
            
            analyses = []
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                cached_sentiment = article_data.get('sentiment', '')
                
                result = f"**{title}**\nURL: {url}\n"
                result += f"**Relevant excerpt:** {chunk_doc.page_content[:200]}...\n\n"
                
                if cached_sentiment:
                    result += f"**Sentiment:** {cached_sentiment}"
                else:
                    # Generate contextual sentiment analysis
                    full_content = article_data.get('content', article_data.get('full_text', ''))
                    if self.llm and full_content:
                        prompt = f"""Analyze sentiment focusing on \"{query}\":\n\nTitle: {title}\nRelevant excerpt: {chunk_doc.page_content}\nFull content: {full_content[:1000]}\n\nProvide sentiment analysis:"""
                        try:
                            response = self.llm.invoke(prompt)
                            sentiment = response.content if hasattr(response, 'content') else str(response)
                            result += f"**Sentiment:** {sentiment}"
                        except Exception as e:
                            result += f"**Sentiment:** Error analyzing sentiment: {str(e)}"
                    else:
                        result += f"**Sentiment:** No sentiment analysis available."
                
                analyses.append(result)
            
            return "\n\n---\n\n".join(analyses)
            
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"

    async def _extract_keywords_semantic_first(self, query: str) -> str:
        try:
            urls = self._extract_urls_from_query(query)
            if urls:
                analyses = []
                for url in urls:
                    article_data = await self._get_article_from_redis_async(url)
                    if article_data:
                        title = article_data.get('title', 'Unknown Title')
                        keywords = article_data.get('keywords', [])
                        if keywords:
                            keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                            analyses.append(f"**{title}**\nURL: {url}\n\nKeywords: {keywords_str}")
                        else:
                            analyses.append(f"**{title}**\nURL: {url}\n\nNo cached keywords available.")
                if analyses:
                    return "\n\n---\n\n".join(analyses)
            
            # Semantic search first approach
            results = await self._semantic_search_with_documents(query, k=3)
            if not results:
                return "No articles found for keyword extraction."
            
            analyses = []
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                cached_keywords = article_data.get('keywords', [])
                
                result = f"**{title}**\nURL: {url}\n"
                result += f"**Relevant excerpt:** {chunk_doc.page_content[:200]}...\n\n"
                
                if cached_keywords:
                    keywords_str = ", ".join(cached_keywords) if isinstance(cached_keywords, list) else str(cached_keywords)
                    result += f"**Keywords:** {keywords_str}"
                else:
                    # Generate contextual keywords
                    full_content = article_data.get('content', article_data.get('full_text', ''))
                    if self.llm and full_content:
                        prompt = f"""Extract keywords focusing on \"{query}\":\n\nTitle: {title}\nRelevant excerpt: {chunk_doc.page_content}\nFull content: {full_content[:1000]}\n\nExtract key terms and concepts:"""
                        try:
                            response = self.llm.invoke(prompt)
                            keywords = response.content if hasattr(response, 'content') else str(response)
                            result += f"**Keywords:** {keywords}"
                        except Exception as e:
                            result += f"**Keywords:** Error extracting keywords: {str(e)}"
                    else:
                        result += f"**Keywords:** No keywords available."
                
                analyses.append(result)
            
            return "\n\n---\n\n".join(analyses)
            
        except Exception as e:
            return f"Error extracting keywords: {str(e)}"

    async def _compare_articles_semantic_first(self, query: str) -> str:
        try:
            results = await self._semantic_search_with_documents(query, k=5)
            if len(results) < 2:
                return "Need at least 2 articles to compare. Please refine your search."
            
            articles_info = []
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                summary = article_data.get('summary', '')
                keywords = article_data.get('keywords', [])
                sentiment = article_data.get('sentiment', '')
                
                info = f"**{title}**\nURL: {url}"
                info += f"\n**Relevant excerpt:** {chunk_doc.page_content[:300]}..."
                
                if summary:
                    info += f"\n**Summary:** {summary}"
                if keywords:
                    keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                    info += f"\n**Keywords:** {keywords_str}"
                if sentiment:
                    info += f"\n**Sentiment:** {sentiment}"
                
                articles_info.append(info)
            
            if not articles_info:
                return "No article data available for comparison."
            
            # Generate comparison
            comparison_prompt = f"""Compare these {len(articles_info)} articles about \"{query}\":\n\n{chr(10).join([f'Article {i+1}:\n{article}\n' for i, article in enumerate(articles_info)])}\n\nProvide comparative analysis focusing on:\n1. Common themes and differences\n2. How each article's relevant excerpt relates to \"{query}\"\n3. Different perspectives or approaches\n4. Complementary insights\n"""
            
            if not self.llm:
                return f"Found {len(articles_info)} articles but LLM not initialized for comparison."
            
            response = self.llm.invoke(comparison_prompt)
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return f"**Article Comparison Analysis**\n\n{analysis}"
            
        except Exception as e:
            return f"Error comparing articles: {str(e)}"

    async def _search_articles_semantic_first(self, query: str) -> str:
        try:
            results = await self._semantic_search_with_documents(query, k=5)
            if not results:
                return "No articles found matching your search query."
            
            search_results = []
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                summary = article_data.get('summary', '')
                
                result = f"**{title}**\nURL: {url}\n"
                result += f"**Why relevant:** {chunk_doc.page_content[:300]}...\n"
                
                if summary:
                    result += f"**Summary:** {summary}"
                else:
                    content = article_data.get('content', article_data.get('full_text', ''))
                    result += f"**Content excerpt:** {content[:200]}..." if content else "No content available"
                
                search_results.append(result)
            
            return f"**Search Results ({len(results)} articles found)**\n\n" + "\n\n---\n\n".join(search_results)
            
        except Exception as e:
            return f"Error searching articles: {str(e)}"

    async def _find_entities_semantic_first(self, query: str) -> str:
        try:
            results = await self._semantic_search_with_documents(query, k=5)
            if not results:
                return "No articles found for entity extraction."
            
            semantic_chunks = []
            full_excerpts = []
            
            for url, article_data, chunk_doc in results:
                title = article_data.get('title', 'Unknown Title')
                content = article_data.get('content', article_data.get('full_text', ''))
                
                semantic_chunks.append(f"From '{title}': {chunk_doc.page_content}")
                full_excerpts.append(f"{title}: {content[:400]}" if content else f"{title}: No content")
            
            combined_semantic = " ".join(semantic_chunks)
            combined_full = " ".join(full_excerpts)
            
            entity_prompt = f"""Extract named entities from articles about \"{query}\":\n\n**Semantically relevant sections:**\n{combined_semantic[:1500]}...\n\n**Full article excerpts:**\n{combined_full[:1000]}...\n\nIdentify and categorize:\n- People (names, roles)\n- Organizations (companies, institutions)  \n- Locations (countries, cities)\n- Technologies/Products\n- Key dates or events\n- Most frequently mentioned entities\n"""
            
            if not self.llm:
                return f"Found {len(results)} articles but LLM not initialized for entity extraction."
            
            response = self.llm.invoke(entity_prompt)
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return f"**Named Entity Analysis (from {len(results)} articles)**\n\n{analysis}"
            
        except Exception as e:
            return f"Error finding entities: {str(e)}"

    async def process_question(self, question: str) -> str:
        """
        Process a user question using the intelligent agent.
        
        The agent uses semantic-search-first approach for comprehensive analysis.
        """
        try:
            if not self.agent_executor:
                return "Agent not initialized. Please try again later."
            
            # Execute the agent using async method
            result = await self.agent_executor.ainvoke({"input": question})
            
            return result.get("output", "Sorry, I couldn't process your question.")
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_article_by_url",
                description="Get complete article content and metadata for a specific URL. Use when user mentions a specific URL.",
                func=self._get_article_by_url,
                coroutine=True
            ),
            Tool(
                name="get_article_summary",
                description="Get summaries using semantic search first, then complete documents. Shows both relevant excerpts and full summaries.",
                func=self._get_article_summary_semantic_first,
                coroutine=True
            ),
            Tool(
                name="analyze_sentiment",
                description="Analyze sentiment using semantic search + complete documents. Shows relevant excerpts and sentiment analysis.",
                func=self._analyze_sentiment_semantic_first,
                coroutine=True
            ),
            Tool(
                name="extract_keywords",
                description="Extract keywords using semantic search + complete documents. Shows relevant excerpts and comprehensive keywords.",
                func=self._extract_keywords_semantic_first,
                coroutine=True
            ),
            Tool(
                name="compare_articles",
                description="Compare articles using semantic search + complete documents for rich comparison context.",
                func=self._compare_articles_semantic_first,
                coroutine=True
            ),
            Tool(
                name="search_articles",
                description="Search articles using semantic search first, then enrich with complete document data.",
                func=self._search_articles_semantic_first,
                coroutine=True
            ),
            Tool(
                name="find_entities",
                description="Find entities using semantic search + complete documents for comprehensive entity extraction.",
                func=self._find_entities_semantic_first,
                coroutine=True
            )
        ]
