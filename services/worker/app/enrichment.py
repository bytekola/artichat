"""Simplified NLP analysis with configurable modes."""
import json
import logging
from functools import lru_cache
from typing import Dict, Optional

import yake
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import ValidationError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import settings
from .schemas import NLPAnalysisResult


@lru_cache(maxsize=1)
def _create_llm_client(for_summary: bool = False):
    """Create and cache LLM client based on configuration."""
    model_kwargs = {}
    
    if for_summary:
        model_kwargs = {"temperature": 0.2, "max_tokens": 400}
    else:
        model_kwargs = {"response_format": {"type": "json_object"}, "temperature": 0.1, "max_tokens": 1000}
    
    if settings.azure_openai_enabled and settings.azure_openai_chat_deployment:
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            model_kwargs=model_kwargs,
        )
    else:
        return ChatOpenAI(
            model=settings.openai_llm_model, 
            api_key=settings.openai_api_key, 
            model_kwargs=model_kwargs
        )


@lru_cache(maxsize=1)
def get_nlp_tools():
    """Initialize and cache NLP tools."""
    logging.info("🔧 Loading NLP tools...")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    keyword_extractor = yake.KeywordExtractor(n=10, dedupLim=0.9, top=15, features=None)
    return sentiment_analyzer, keyword_extractor


def run_analysis(full_text: str, extra_context: Optional[Dict] = None) -> NLPAnalysisResult:
    """
    Run analysis based on configured mode: 'deterministic' (fast) or 'llm' (AI-powered).
    Summary is ALWAYS generated using LLM regardless of analysis mode.
    """
    if not full_text or full_text.isspace():
        logging.warning("Empty text provided for analysis")
        return NLPAnalysisResult(
            summary=None,
            sentiment_label=None,
            sentiment_score=None,
            language_detected=None,
            keywords=[],
        )
    
    mode = settings.analysis_mode.lower()
    logging.info(f"Running {mode} analysis")
    
    # ALWAYS generate summary using LLM (regardless of analysis mode)
    summary = _generate_summary(full_text)
    
    if mode == "llm":
        # Check if we have API credentials for LLM mode
        if not settings.openai_api_key.get_secret_value() and not (settings.azure_openai_enabled and settings.azure_openai_api_key.get_secret_value()):
            logging.error("LLM mode selected but no API key configured. Falling back to deterministic.")
            result = _run_deterministic_analysis(full_text)
            result.summary = summary
            return result
            
        result = _run_llm_analysis(full_text, extra_context)
        if result:
            result.summary = summary
            logging.info("LLM analysis completed")
            return result
        logging.warning("LLM failed, falling back to deterministic")
    
    # Deterministic mode: local analysis for sentiment/keywords + LLM summary
    result = _run_deterministic_analysis(full_text)
    result.summary = summary
    return result


def _generate_summary(full_text: str) -> Optional[str]:
    """Generate a summary using configurable model regardless of analysis mode."""
    try:
        # Handle long articles by truncating to max input chars
        max_chars = settings.llm_max_input_chars
        truncated_text = full_text[:max_chars]
        
        # Log if article was truncated
        if len(full_text) > max_chars:
            logging.info(f"📄 Article truncated from {len(full_text)} to {max_chars} chars for summary")
        
        # Create LLM client using centralized factory
        llm = _create_llm_client(for_summary=True)
        
        prompt = f"""Summarize this article in 2-3 sentences. Focus on the main points and key takeaways.

        Article: {truncated_text}"""
        
        response = llm.invoke(prompt)
        summary = getattr(response, "content", "").strip()
        
        if summary:
            logging.info("Summary generated successfully")
            return summary
        else:
            logging.warning("Empty summary returned")
            return None
            
    except Exception as e:
        logging.warning(f"Summary generation failed: {e}")
        return None


def _run_llm_analysis(full_text: str, extra_context: Optional[Dict] = None) -> Optional[NLPAnalysisResult]:
    """LLM analysis using OpenAI/Azure OpenAI."""
    try:
        # Create LLM client using centralized factory
        llm = _create_llm_client(for_summary=False)
        
        # Create analysis prompt
        prompt = f"""Analyze this article and return a JSON object with these exact fields:
- keywords: array of top 15 keywords/phrases
- sentiment_label: "POSITIVE", "NEGATIVE", or "NEUTRAL"  
- sentiment_score: confidence 0.0-1.0
- language_detected: ISO code like "en"

Article: {full_text[:settings.llm_max_input_chars]}"""
        
        # Get response and parse
        response = llm.invoke(prompt)
        content = getattr(response, "content", "")
        
        # Clean and parse JSON
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        data = json.loads(cleaned)
        
        # Return validated result
        return NLPAnalysisResult.model_validate(data)
        
    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"LLM analysis output parsing failed: {e}")
        return None
    except Exception as e:
        logging.warning(f"LLM analysis failed: {e}")
        return None


def _run_deterministic_analysis(full_text: str) -> NLPAnalysisResult:
    """Fast local analysis using YAKE + VADER."""
    sentiment_analyzer, keyword_extractor = get_nlp_tools()
    
    # Extract keywords
    keywords_tuples = keyword_extractor.extract_keywords(full_text)
    keywords = [kw for kw, score in keywords_tuples if len(kw) > 2]
    
    # Analyze sentiment
    sentiment_score = 0.5
    sentiment_label = "NEUTRAL"
    try:
        scores = sentiment_analyzer.polarity_scores(full_text[:5000])
        compound = scores.get("compound", 0.0)
        sentiment_score = round(abs(compound), 4)
        if compound >= 0.05:
            sentiment_label = "POSITIVE"
        elif compound <= -0.05:
            sentiment_label = "NEGATIVE"
    except Exception as e:
        logging.warning(f"Sentiment analysis failed: {e}")
    
    logging.info(f"Deterministic analysis: {len(keywords)} keywords, {sentiment_label}")
    return NLPAnalysisResult(
        keywords=keywords,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        language_detected="en",
        summary=None
    )

