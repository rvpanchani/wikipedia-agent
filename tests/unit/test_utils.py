"""
Unit tests for the core utilities module.
"""

import pytest


class TestCleanLLMResponse:
    """Tests for clean_llm_response function."""
    
    def test_removes_thinking_tokens(self):
        """Test that thinking tokens are removed."""
        from wikipedia_agent.core.utils import clean_llm_response
        
        response = "<think>Let me think about this...</think>The answer is 42."
        cleaned = clean_llm_response(response)
        
        assert "<think>" not in cleaned
        assert "The answer is 42." in cleaned
    
    def test_removes_unclosed_thinking_tokens(self):
        """Test that unclosed thinking tokens are removed."""
        from wikipedia_agent.core.utils import clean_llm_response
        
        response = "Some text<think>Thinking continues..."
        cleaned = clean_llm_response(response)
        
        assert "<think>" not in cleaned
        assert "Some text" in cleaned
    
    def test_removes_endoftext_marker(self):
        """Test that endoftext markers are removed."""
        from wikipedia_agent.core.utils import clean_llm_response
        
        response = "The answer is 42.<|endoftext|>More text"
        cleaned = clean_llm_response(response)
        
        assert "<|endoftext|>" not in cleaned
        assert "More text" not in cleaned
    
    def test_splits_on_human_marker(self):
        """Test that content after Human: is removed."""
        from wikipedia_agent.core.utils import clean_llm_response
        
        response = "The answer is 42.\n\nHuman: Another question"
        cleaned = clean_llm_response(response)
        
        assert "Human:" not in cleaned
        assert "Another question" not in cleaned
    
    def test_strips_whitespace(self):
        """Test that result is stripped."""
        from wikipedia_agent.core.utils import clean_llm_response
        
        response = "  \n  The answer is 42.  \n  "
        cleaned = clean_llm_response(response)
        
        assert cleaned == "The answer is 42."


class TestExtractNumbers:
    """Tests for extract_numbers function."""
    
    def test_extracts_integers(self):
        """Test extraction of integers."""
        from wikipedia_agent.core.utils import extract_numbers
        
        numbers = extract_numbers("There are 42 apples and 7 oranges")
        
        assert 42.0 in numbers
        assert 7.0 in numbers
    
    def test_extracts_floats(self):
        """Test extraction of floating point numbers."""
        from wikipedia_agent.core.utils import extract_numbers
        
        numbers = extract_numbers("Pi is approximately 3.14159")
        
        assert 3.14159 in numbers
    
    def test_extracts_negative_numbers(self):
        """Test extraction of negative numbers."""
        from wikipedia_agent.core.utils import extract_numbers
        
        numbers = extract_numbers("Temperature is -42 degrees")
        
        assert -42.0 in numbers
    
    def test_returns_empty_for_no_numbers(self):
        """Test that empty list is returned when no numbers."""
        from wikipedia_agent.core.utils import extract_numbers
        
        numbers = extract_numbers("No numbers here")
        
        assert numbers == []


class TestFormatNumber:
    """Tests for format_number function."""
    
    def test_formats_regular_number(self):
        """Test formatting of regular numbers."""
        from wikipedia_agent.core.utils import format_number
        
        assert format_number(3.14159, 2) == "3.14"
    
    def test_formats_large_number_scientific(self):
        """Test scientific notation for large numbers."""
        from wikipedia_agent.core.utils import format_number
        
        result = format_number(1000000)
        assert "e" in result or "E" in result
    
    def test_formats_small_number_scientific(self):
        """Test scientific notation for small numbers."""
        from wikipedia_agent.core.utils import format_number
        
        result = format_number(0.00001)
        assert "e" in result or "E" in result or result == "0"
    
    def test_formats_zero(self):
        """Test formatting of zero."""
        from wikipedia_agent.core.utils import format_number
        
        assert format_number(0.0) == "0"
    
    def test_removes_trailing_zeros(self):
        """Test that trailing zeros are removed."""
        from wikipedia_agent.core.utils import format_number
        
        assert format_number(3.0, 2) == "3"


class TestDetectAngleUnit:
    """Tests for detect_angle_unit function."""
    
    def test_detects_degrees_word(self):
        """Test detection of 'degrees' keyword."""
        from wikipedia_agent.core.utils import detect_angle_unit
        
        assert detect_angle_unit("Calculate sin of 45 degrees") == "degrees"
    
    def test_detects_degree_symbol(self):
        """Test detection of degree symbol."""
        from wikipedia_agent.core.utils import detect_angle_unit
        
        assert detect_angle_unit("The angle is 90°") == "degrees"
    
    def test_detects_radians(self):
        """Test detection of radians."""
        from wikipedia_agent.core.utils import detect_angle_unit
        
        assert detect_angle_unit("Calculate cos(π/4 radians)") == "radians"
    
    def test_returns_none_when_no_unit(self):
        """Test that None is returned when no unit detected."""
        from wikipedia_agent.core.utils import detect_angle_unit
        
        assert detect_angle_unit("Calculate sin(x)") is None


class TestExtractCodeBlock:
    """Tests for extract_code_block function."""
    
    def test_extracts_python_code_block(self):
        """Test extraction of Python code block."""
        from wikipedia_agent.core.utils import extract_code_block
        
        text = """
Here's the code:
```python
x = 42
print(x)
```
"""
        code = extract_code_block(text)
        
        assert "x = 42" in code
        assert "print(x)" in code
    
    def test_extracts_generic_code_block(self):
        """Test extraction of generic code block."""
        from wikipedia_agent.core.utils import extract_code_block
        
        text = """
```
result = 10
```
"""
        code = extract_code_block(text)
        
        assert "result = 10" in code
    
    def test_returns_none_when_no_code_block(self):
        """Test that None is returned when no code block."""
        from wikipedia_agent.core.utils import extract_code_block
        
        assert extract_code_block("No code here") is None


class TestIsScientificQuestion:
    """Tests for is_scientific_question function."""
    
    def test_detects_calculation_keywords(self):
        """Test detection of calculation keywords."""
        from wikipedia_agent.core.utils import is_scientific_question
        
        assert is_scientific_question("Calculate sin(45 degrees)")
        assert is_scientific_question("Compute the derivative")
        assert is_scientific_question("Solve for x")
    
    def test_detects_physics_keywords(self):
        """Test detection of physics keywords."""
        from wikipedia_agent.core.utils import is_scientific_question
        
        assert is_scientific_question("What is the velocity?")
        assert is_scientific_question("Calculate the force")
        assert is_scientific_question("Find the acceleration")
    
    def test_returns_false_for_non_scientific(self):
        """Test that non-scientific questions return False."""
        from wikipedia_agent.core.utils import is_scientific_question
        
        assert not is_scientific_question("Who was Napoleon?")
        assert not is_scientific_question("What is the capital of France?")


class TestTruncateText:
    """Tests for truncate_text function."""
    
    def test_truncates_long_text(self):
        """Test that long text is truncated."""
        from wikipedia_agent.core.utils import truncate_text
        
        text = "a" * 100
        truncated = truncate_text(text, max_length=50)
        
        assert len(truncated) == 50
        assert truncated.endswith("...")
    
    def test_preserves_short_text(self):
        """Test that short text is not modified."""
        from wikipedia_agent.core.utils import truncate_text
        
        text = "Short text"
        truncated = truncate_text(text, max_length=100)
        
        assert truncated == text
    
    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        from wikipedia_agent.core.utils import truncate_text
        
        text = "a" * 100
        truncated = truncate_text(text, max_length=50, suffix="…")
        
        assert truncated.endswith("…")
