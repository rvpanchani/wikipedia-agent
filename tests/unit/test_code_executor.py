"""
Unit tests for the V2 code executor module.
"""

import pytest
import math


class TestCodeExecutor:
    """Tests for the CodeExecutor class."""
    
    def test_simple_calculation(self, code_executor):
        """Test simple arithmetic calculation."""
        result = code_executor.execute("result = 2 + 3")
        
        assert result.success
        assert result.variables.get('result') == 5
    
    def test_trigonometric_calculation(self, code_executor):
        """Test trigonometric functions are available."""
        code = """
angle_deg = 45
angle_rad = radians(angle_deg)
result = sin(angle_rad)
"""
        result = code_executor.execute(code)
        
        assert result.success
        assert abs(result.variables.get('result') - math.sqrt(2)/2) < 0.0001
    
    def test_complex_formula(self, code_executor):
        """Test complex formula execution."""
        code = """
# Quadratic formula: x = (-b ± sqrt(b² - 4ac)) / 2a
a, b, c = 1, -5, 6
discriminant = b**2 - 4*a*c
x1 = (-b + sqrt(discriminant)) / (2*a)
x2 = (-b - sqrt(discriminant)) / (2*a)
result = (x1, x2)
"""
        result = code_executor.execute(code)
        
        assert result.success
        assert result.variables.get('result') == (3.0, 2.0)
    
    def test_print_output_captured(self, code_executor):
        """Test that print statements are captured."""
        code = """
print("Hello, World!")
result = 42
"""
        result = code_executor.execute(code)
        
        assert result.success
        assert "Hello, World!" in result.output
    
    def test_statistical_functions(self, code_executor):
        """Test statistical helper functions."""
        code = """
data = [1, 2, 3, 4, 5]
avg = mean(data)
med = median(data)
result = (avg, med)
"""
        result = code_executor.execute(code)
        
        assert result.success
        assert result.variables.get('result') == (3.0, 3)
    
    def test_math_constants_available(self, code_executor):
        """Test that math constants are available."""
        code = """
result = pi
"""
        result = code_executor.execute(code)
        
        assert result.success
        assert abs(result.variables.get('result') - math.pi) < 0.0001
    
    def test_dangerous_function_blocked(self, code_executor):
        """Test that dangerous functions are blocked."""
        result = code_executor.execute("exec('print(1)')")
        
        assert not result.success
        assert "not allowed" in result.error
    
    def test_syntax_error_reported(self, code_executor):
        """Test that syntax errors are reported."""
        result = code_executor.execute("def incomplete(")
        
        assert not result.success
        assert result.error is not None


class TestCodeExtraction:
    """Tests for code extraction from LLM responses."""
    
    def test_extract_python_code_block(self, code_executor):
        """Test extraction of Python code block."""
        response = """
Here's the code:

```python
angle = 45
result = sin(radians(angle))
```

This calculates sine.
"""
        code = code_executor.extract_code_from_response(response)
        
        assert code is not None
        assert "sin" in code
        assert "radians" in code
    
    def test_extract_generic_code_block(self, code_executor):
        """Test extraction of generic code block."""
        response = """
```
result = 42
```
"""
        code = code_executor.extract_code_from_response(response)
        
        assert code is not None
        assert "result = 42" in code
    
    def test_cleans_thinking_tokens(self, code_executor):
        """Test that thinking tokens are cleaned before extraction."""
        response = """
<think>Let me think...</think>

```python
result = 42
```
"""
        code = code_executor.extract_code_from_response(response)
        
        assert code is not None
        assert "<think>" not in code
    
    def test_returns_none_for_no_code(self, code_executor):
        """Test that None is returned when no code found."""
        code = code_executor.extract_code_from_response("Just text, no code")
        
        assert code is None


class TestSafeCodeValidator:
    """Tests for the SafeCodeValidator class."""
    
    def test_valid_code_passes(self):
        """Test that valid code passes validation."""
        from wikipedia_agent.v2.code_executor import SafeCodeValidator
        
        validator = SafeCodeValidator()
        is_valid, errors = validator.validate("x = 10\nresult = x * 2")
        
        assert is_valid
        assert len(errors) == 0
    
    def test_import_detected(self):
        """Test that imports are detected."""
        from wikipedia_agent.v2.code_executor import SafeCodeValidator
        
        validator = SafeCodeValidator()
        is_valid, errors = validator.validate("import os")
        
        assert not is_valid
        assert any("Import" in e for e in errors)
    
    def test_from_import_detected(self):
        """Test that from imports are detected."""
        from wikipedia_agent.v2.code_executor import SafeCodeValidator
        
        validator = SafeCodeValidator()
        is_valid, errors = validator.validate("from os import path")
        
        assert not is_valid
        assert any("Import" in e for e in errors)
    
    def test_forbidden_function_detected(self):
        """Test that forbidden functions are detected."""
        from wikipedia_agent.v2.code_executor import SafeCodeValidator
        
        validator = SafeCodeValidator()
        is_valid, errors = validator.validate("eval('1 + 1')")
        
        assert not is_valid
        assert any("eval" in e for e in errors)
    
    def test_syntax_error_reported(self):
        """Test that syntax errors are reported."""
        from wikipedia_agent.v2.code_executor import SafeCodeValidator
        
        validator = SafeCodeValidator()
        is_valid, errors = validator.validate("def incomplete(")
        
        assert not is_valid
        assert any("Syntax" in e for e in errors)
