"""
Unit tests for the V2 prompts module.
"""

import pytest


class TestQueryCategory:
    """Tests for QueryCategory enum."""
    
    def test_category_values(self):
        """Test that all expected categories exist."""
        from wikipedia_agent.v2.prompts import QueryCategory
        
        assert QueryCategory.DIRECT.value == "DIRECT"
        assert QueryCategory.WIKIPEDIA.value == "WIKIPEDIA"
        assert QueryCategory.COMPUTE.value == "COMPUTE"
        assert QueryCategory.COMBINED.value == "COMBINED"


class TestPromptTemplates:
    """Tests for PromptTemplates class."""
    
    def test_agent_system_prompt_exists(self):
        """Test that agent system prompt is defined."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        assert PromptTemplates.AGENT_SYSTEM
        assert "knowledgeable assistant" in PromptTemplates.AGENT_SYSTEM
    
    def test_code_generator_system_prompt_exists(self):
        """Test that code generator system prompt is defined."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        assert PromptTemplates.CODE_GENERATOR_SYSTEM
        assert "Python" in PromptTemplates.CODE_GENERATOR_SYSTEM
    
    def test_classify_query_prompt(self):
        """Test classify_query_prompt generation."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.classify_query_prompt("What is 2 + 2?")
        
        assert "What is 2 + 2?" in prompt
        assert "DIRECT" in prompt
        assert "WIKIPEDIA" in prompt
        assert "COMPUTE" in prompt
        assert "COMBINED" in prompt
    
    def test_direct_answer_prompt(self):
        """Test direct_answer_prompt generation."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.direct_answer_prompt("What is gravity?")
        
        assert "What is gravity?" in prompt
        assert "Answer" in prompt
    
    def test_generate_search_term_prompt(self):
        """Test generate_search_term_prompt generation."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.generate_search_term_prompt("Who invented the telephone?")
        
        assert "Who invented the telephone?" in prompt
        assert "Wikipedia" in prompt
    
    def test_generate_search_terms_prompt_with_previous(self):
        """Test generate_search_terms_prompt with previous attempts."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.generate_search_terms_prompt(
            "Who invented the telephone?",
            previous_attempts=["telephone", "invention"]
        )
        
        assert "telephone" in prompt
        assert "invention" in prompt
    
    def test_extract_answer_from_context_prompt(self):
        """Test extract_answer_from_context_prompt generation."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.extract_answer_from_context_prompt(
            "What is the capital?",
            "Paris is the capital of France."
        )
        
        assert "What is the capital?" in prompt
        assert "Paris is the capital of France." in prompt
    
    def test_generate_computation_code_prompt(self):
        """Test generate_computation_code_prompt generation."""
        from wikipedia_agent.v2.prompts import PromptTemplates
        
        prompt = PromptTemplates.generate_computation_code_prompt("Calculate sin(45)")
        
        assert "Calculate sin(45)" in prompt
        assert "result" in prompt.lower()


class TestParseClassification:
    """Tests for parse_classification function."""
    
    def test_parse_direct_category(self):
        """Test parsing DIRECT category."""
        from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory
        
        assert PromptTemplates.parse_classification("A: This is common knowledge") == QueryCategory.DIRECT
        assert PromptTemplates.parse_classification("DIRECT: No tools needed") == QueryCategory.DIRECT
    
    def test_parse_wikipedia_category(self):
        """Test parsing WIKIPEDIA category."""
        from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory
        
        assert PromptTemplates.parse_classification("B: Needs lookup") == QueryCategory.WIKIPEDIA
        assert PromptTemplates.parse_classification("WIKIPEDIA: Factual query") == QueryCategory.WIKIPEDIA
    
    def test_parse_compute_category(self):
        """Test parsing COMPUTE category."""
        from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory
        
        assert PromptTemplates.parse_classification("C: Pure calculation") == QueryCategory.COMPUTE
        assert PromptTemplates.parse_classification("COMPUTE: Math problem") == QueryCategory.COMPUTE
    
    def test_parse_combined_category(self):
        """Test parsing COMBINED category."""
        from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory
        
        assert PromptTemplates.parse_classification("D: Needs lookup and calculation") == QueryCategory.COMBINED
        assert PromptTemplates.parse_classification("COMBINED: Formula application") == QueryCategory.COMBINED
    
    def test_parse_defaults_to_direct(self):
        """Test that unclear responses default to DIRECT."""
        from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory
        
        assert PromptTemplates.parse_classification("I'm not sure") == QueryCategory.DIRECT
        assert PromptTemplates.parse_classification("") == QueryCategory.DIRECT
