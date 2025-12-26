"""
Integration tests for the V1 agent (deprecated).
"""

import pytest
import warnings


@pytest.mark.integration
class TestWikipediaAgentV1:
    """Integration tests for WikipediaAgent V1 (deprecated)."""
    
    def test_deprecation_warning_on_import(self):
        """Test that importing V1 emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from wikipedia_agent.v1 import WikipediaAgent
            
            # Check that deprecation warning was emitted
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    
    def test_process_query_returns_tuple(self, v1_agent):
        """Test that process_query returns (answer, sources) tuple."""
        answer, sources = v1_agent.process_query("What is the capital of France?")
        
        assert isinstance(answer, str)
        assert isinstance(sources, list)
    
    def test_process_query_basic(self, v1_agent):
        """Test basic query processing."""
        answer, sources = v1_agent.process_query("What is photosynthesis?")
        
        assert answer
        assert len(answer) > 20
    
    def test_generate_search_terms(self, v1_agent):
        """Test search term generation."""
        terms = v1_agent.generate_search_terms("Who invented the telephone?")
        
        assert isinstance(terms, list)
        assert len(terms) > 0
    
    def test_search_wikipedia(self, v1_agent):
        """Test Wikipedia search."""
        content, url = v1_agent.search_wikipedia("Python programming")
        
        # Note: May return None if network is unavailable
        if content:
            assert isinstance(content, str)
            assert isinstance(url, str)
