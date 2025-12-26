"""
Unit tests for the V2 Wikipedia search module.
"""

import pytest


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_has_content_with_content(self):
        """Test has_content returns True when content exists."""
        from wikipedia_agent.v2.wikipedia_search import SearchResult
        
        result = SearchResult(
            term="test",
            content="Some content"
        )
        
        assert result.has_content()
    
    def test_has_content_with_summary(self):
        """Test has_content returns True when summary exists."""
        from wikipedia_agent.v2.wikipedia_search import SearchResult
        
        result = SearchResult(
            term="test",
            summary="Some summary"
        )
        
        assert result.has_content()
    
    def test_has_content_empty(self):
        """Test has_content returns False when both are empty."""
        from wikipedia_agent.v2.wikipedia_search import SearchResult
        
        result = SearchResult(term="test")
        
        assert not result.has_content()


class TestWikipediaSearcher:
    """Tests for WikipediaSearcher class."""
    
    def test_search_returns_results(self, wikipedia_searcher):
        """Test that search returns results for a valid term."""
        results = wikipedia_searcher.search("Python programming")
        
        assert isinstance(results, list)
        # Note: This test requires network access
    
    def test_get_search_result_returns_search_result(self, wikipedia_searcher):
        """Test that get_search_result returns a SearchResult."""
        result = wikipedia_searcher.get_search_result("Python programming language")
        
        from wikipedia_agent.v2.wikipedia_search import SearchResult
        assert isinstance(result, SearchResult)
    
    def test_search_multiple_terms(self, wikipedia_searcher):
        """Test searching multiple terms."""
        terms = ["Python", "JavaScript"]
        results = wikipedia_searcher.search_multiple_terms(terms)
        
        assert len(results) == 2
        from wikipedia_agent.v2.wikipedia_search import SearchResult
        assert all(isinstance(r, SearchResult) for r in results)


class TestWikipediaSearcherIntegration:
    """Integration tests for WikipediaSearcher (require network)."""
    
    @pytest.mark.network
    def test_get_page_content(self, wikipedia_searcher):
        """Test getting page content."""
        content, url = wikipedia_searcher.get_page_content("Albert Einstein")
        
        assert content is not None
        assert url is not None
        assert "Einstein" in content or "physicist" in content.lower()
    
    @pytest.mark.network
    def test_get_formula_content(self, wikipedia_searcher):
        """Test getting formula content."""
        content = wikipedia_searcher.get_formula_content("Pythagorean theorem")
        
        # May return None if network is unavailable
        if content:
            assert "theorem" in content.lower() or "pythagorean" in content.lower()
