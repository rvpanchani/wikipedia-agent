"""
Integration tests for the V2 agent.
"""

import pytest


@pytest.mark.integration
class TestWikipediaAgentV2:
    """Integration tests for WikipediaAgentV2."""
    
    def test_query_direct_question(self, v2_agent):
        """Test a direct knowledge question."""
        response = v2_agent.query("What is gravity?")
        
        assert response.answer
        assert len(response.answer) > 20
    
    def test_query_computation(self, v2_agent):
        """Test a computation question."""
        response = v2_agent.query("What is 17% of 8450?")
        
        assert response.answer
        # Should contain the result (approximately 1436.5)
        assert "1436" in response.answer or "1437" in response.answer
    
    def test_query_wikipedia_lookup(self, v2_agent):
        """Test a Wikipedia lookup question."""
        response = v2_agent.query("When was the Eiffel Tower built?")
        
        assert response.answer
        # Should mention 1889 or construction dates
        assert "1889" in response.answer or "18" in response.answer
    
    def test_ask_returns_formatted_string(self, v2_agent):
        """Test that ask() returns a formatted string."""
        result = v2_agent.ask("What is 2 + 2?")
        
        assert isinstance(result, str)
        assert "ANSWER" in result


@pytest.mark.integration
class TestAgentResponse:
    """Tests for AgentResponse formatting."""
    
    def test_format_output_basic(self):
        """Test basic output formatting."""
        from wikipedia_agent.v2.agent import AgentResponse
        from wikipedia_agent.v2.prompts import QueryCategory
        
        response = AgentResponse(
            answer="The answer is 42.",
            sources=["https://wikipedia.org/wiki/Test"],
            category=QueryCategory.WIKIPEDIA
        )
        
        output = response.format_output()
        
        assert "ANSWER" in output
        assert "The answer is 42." in output
        assert "Sources" in output
        assert "wikipedia.org" in output
    
    def test_format_output_with_code(self):
        """Test output formatting with code."""
        from wikipedia_agent.v2.agent import AgentResponse
        from wikipedia_agent.v2.prompts import QueryCategory
        
        response = AgentResponse(
            answer="The result is 42.",
            sources=[],
            category=QueryCategory.COMPUTE,
            code_executed="result = 6 * 7",
            code_result="42"
        )
        
        output = response.format_output()
        
        assert "CODE EXECUTED" in output
        assert "result = 6 * 7" in output
        assert "42" in output
