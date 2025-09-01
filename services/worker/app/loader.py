import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean up text extracted from web pages by removing excessive whitespace,
    normalizing newlines, removing navigation/UI elements, and improving readability.
    
    Args:
        text: Raw text from web page
        
    Returns:
        Cleaned text with normalized whitespace and filtered content
    """
    if not text:
        return ""
    
    # Common navigation and UI patterns to remove
    ui_patterns = [
        # Navigation and menu items
        r'Sign in.*?account',
        r'Subscribe.*?newsletters?',
        r'Follow.*?(?:CNN|Facebook|Twitter|Instagram)',
        r'Ad Feedback',
        r'CNN values your feedback',
        r'Watch.*?Listen.*?Live TV',
        r'Markets.*?Tech.*?Media.*?Calculators',
        r'Edition.*?US.*?International.*?Arabic.*?Español',
        r'World.*?Africa.*?Americas.*?Asia.*?Australia.*?China.*?Europe',
        r'Business.*?Tech.*?Media.*?Calculators.*?Videos',
        r'Health.*?Life, But Better.*?Fitness.*?Food.*?Sleep',
        r'Entertainment.*?Movies.*?Television.*?Celebrity',
        r'Travel.*?Destinations.*?Food & Drink.*?Stay',
        r'Sports.*?Football.*?Tennis.*?Golf.*?Motorsport',
        r'Science.*?Space.*?Life.*?Unearthed.*?Climate',
        
        # Ads and feedback forms
        r'How relevant is this ad to you\?',
        r'Did you encounter any technical issues\?',
        r'Video player was slow to load content',
        r'Video content never loaded',
        r'Ad froze or did not finish loading',
        r'Audio on ad was too loud',
        r'Ad never loaded',
        r'Other issues.*?Cancel.*?Submit',
        r'Thank You!.*?Your effort and contribution.*?appreciated',
        
        # Social media and sharing
        r'Facebook.*?Tweet.*?Email.*?Link.*?Link Copied!',
        r'See all topics',
        
        # Footer and legal text
        r'Terms of Use.*?Privacy Policy.*?Ad Choices',
        r'© \d{4}.*?All Rights Reserved',
        r'Most stock quote data provided by.*?All rights reserved',
        
        # Page structure elements
        r'Close icon',
        r'Updated.*?\d{4},.*?Published.*?\d{4},',
        r'\d+ min read',
        
        # Common repeated phrases
        r'Sign in to your CNN account.*?(?=\w)',
        r'My Account.*?Settings.*?Newsletters.*?Topics you follow.*?Sign out',
        r'Live TV.*?Listen.*?Watch',
    ]
    
    # Apply UI pattern removal
    for pattern in ui_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive newlines (3+ consecutive newlines become 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove lines that are just single words or very short (likely navigation)
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and (len(line.split()) > 2 or len(line) > 20):  # Keep substantial content
            lines.append(line)
    
    # Remove empty lines at the beginning and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    # Join lines back together
    text = '\n'.join(lines)
    
    # Final cleanup - remove any remaining excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()


def validate_url(url: str) -> bool:
    """
    Validates if the given URL is properly formatted and accessible.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if URL appears valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def load_document_from_url(url: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Uses LangChain's WebBaseLoader to fetch and parse the main content
    of a web page into a standardized Document format.

    Args:
        url: The URL to load and parse
        
    Returns:
        A tuple containing the page content (str) and its metadata (dict).
        Returns (None, None) on failure.
    """
    if not validate_url(url):
        logging.error(f"Invalid URL format: {url}")
        return None, None
        
    logging.info(f"Loading document from URL: {url}")
    try:
        loader = WebBaseLoader(
            web_path=url,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        docs: List[Document] = loader.load()
        
        if not docs:
            logging.warning(f"No documents found for URL: {url}")
            return None, None

        # The main content is in the page_content attribute
        full_text = docs[0].page_content
        source_metadata = docs[0].metadata
        
        # Validate content quality before cleaning
        if not full_text or len(full_text.strip()) < 100:
            logging.warning(f"Content too short or empty for URL: {url}")
            return None, None
        
        # Clean up the text - remove excessive newlines and normalize whitespace
        full_text = clean_text(full_text)
        
        # Validate content quality after cleaning
        if len(full_text) < 50:
            logging.warning(f"Content too short after cleaning for URL: {url}")
            return None, None
        
        # Enhance metadata with additional info
        if source_metadata:
            source_metadata["content_length"] = len(full_text)
            source_metadata["word_count"] = len(full_text.split())
        
        logging.info(f"Successfully loaded {len(full_text)} characters from {url}")
        return full_text, source_metadata

    except Exception as e:
        logging.error(f"Failed to load document from {url}. Error: {e}", exc_info=True)
        return None, None