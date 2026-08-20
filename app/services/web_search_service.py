from typing import Any, Dict, List
from duckduckgo_search import DDGS

from app.config import settings
from app.core.logging import logger


class WebSearchService:
    """Provides fallback web search functionality when vector store retrieval fails."""

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.tavily_api_key = settings.TAVILY_API_KEY

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes web search using Tavily (if configured) or DuckDuckGo (zero-cost fallback).
        """
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_api_key)
                response = client.search(query=query, max_results=self.max_results)
                results = []
                for item in response.get("results", []):
                    results.append({
                        "title": item.get("title", "Web Result"),
                        "content": item.get("content", ""),
                        "url": item.get("url", ""),
                        "source": f"web:{item.get('url', '')}"
                    })
                logger.info(f"Retrieved {len(results)} search results via Tavily for query: '{query}'")
                return results
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to DuckDuckGo search.")

        # Zero-cost DuckDuckGo fallback
        try:
            results = []
            with DDGS() as ddgs:
                ddg_gen = ddgs.text(query, max_results=self.max_results)
                for item in ddg_gen:
                    results.append({
                        "title": item.get("title", "DuckDuckGo Result"),
                        "content": item.get("body", ""),
                        "url": item.get("href", ""),
                        "source": f"web:{item.get('href', '')}"
                    })
            logger.info(f"Retrieved {len(results)} search results via DuckDuckGo for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []


# Global singleton instance
web_search_service = WebSearchService()
