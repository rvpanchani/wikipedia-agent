#!/usr/bin/env python3
"""
Wikipedia search functionality for the Wikipedia Agent v2.
"""

import wikipedia
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SearchResult:
    """Container for Wikipedia search result data."""
    term: str
    url: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    found_relevant_info: bool = False

    def has_content(self) -> bool:
        """Check if this search result has content."""
        return bool(self.content or self.summary)


class WikipediaSearcher:
    """Handles all Wikipedia search operations."""

    def __init__(self, language: str = "en"):
        """
        Initialize the Wikipedia searcher.

        Args:
            language: Wikipedia language code (default: "en")
        """
        self.language = language
        wikipedia.set_lang(language)

    def search(self, search_term: str, max_results: int = 3) -> List[str]:
        """
        Search Wikipedia for pages matching the search term.

        Args:
            search_term: Term to search for
            max_results: Maximum number of results to return

        Returns:
            List of page titles matching the search
        """
        try:
            return wikipedia.search(search_term, results=max_results)
        except Exception:
            return []

    def get_page_content(
        self,
        search_term: str,
        max_content_length: int = 3000
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get Wikipedia page content for a search term.

        Args:
            search_term: Term to search for
            max_content_length: Maximum content length to return

        Returns:
            Tuple of (content, url) or (None, None) if not found
        """
        try:
            search_results = wikipedia.search(search_term, results=3)
            if not search_results:
                return None, None

            page = wikipedia.page(search_results[0])

            # Return summary + additional content (limited)
            content = page.summary
            if len(page.content) > len(page.summary):
                additional_content = page.content[len(page.summary):max_content_length]
                content = page.summary + "\n\n" + additional_content

            return content, page.url

        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0])
                return page.summary, page.url
            except Exception:
                return None, None
        except wikipedia.exceptions.PageError:
            return None, None
        except Exception:
            return None, None

    def get_search_result(
        self,
        search_term: str,
        max_content_length: int = 3000
    ) -> SearchResult:
        """
        Get a SearchResult object for a search term.

        Args:
            search_term: Term to search for
            max_content_length: Maximum content length to include

        Returns:
            SearchResult object with search results
        """
        content, url = self.get_page_content(search_term, max_content_length)

        return SearchResult(
            term=search_term,
            url=url,
            content=content,
            summary=content[:500] if content else None,
            found_relevant_info=bool(content)
        )

    def search_multiple_terms(
        self,
        terms: List[str],
        max_content_length: int = 3000
    ) -> List[SearchResult]:
        """
        Search Wikipedia for multiple terms and return results.

        Args:
            terms: List of search terms
            max_content_length: Maximum content length per result

        Returns:
            List of SearchResult objects
        """
        results = []
        for term in terms:
            result = self.get_search_result(term, max_content_length)
            results.append(result)
        return results

    def get_formula_content(
        self,
        topic: str,
        additional_terms: List[str] = None
    ) -> Optional[str]:
        """
        Search for mathematical/scientific formula content.

        Args:
            topic: Main topic to search for
            additional_terms: Additional search terms to try

        Returns:
            Combined formula content from Wikipedia
        """
        search_terms = [topic]
        if additional_terms:
            search_terms.extend(additional_terms)

        # Add formula-specific variations
        search_terms.extend([
            f"{topic} formula",
            f"{topic} equation",
            f"{topic} mathematics",
        ])

        combined_content = []
        seen_urls = set()

        for term in search_terms[:5]:  # Limit to 5 terms
            content, url = self.get_page_content(term, max_content_length=2000)
            if content and url and url not in seen_urls:
                combined_content.append(f"=== From: {term} ===\n{content}")
                seen_urls.add(url)

        return "\n\n".join(combined_content) if combined_content else None
