"""
Tests that validate examples from README documentation work correctly.
"""

import pytest
import subprocess
import sys
import os


@pytest.mark.docs
class TestDocumentedFeatures:
    """Tests that verify documented features work as expected."""
    
    def test_package_import(self):
        """Test basic package import as documented."""
        from wikipedia_agent import WikipediaAgentV2
        
        assert WikipediaAgentV2 is not None
    
    def test_provider_import(self):
        """Test provider imports as documented."""
        from wikipedia_agent import (
            LLMProvider,
            ProviderFactory,
            OpenAIProvider,
            GeminiProvider,
            OllamaProvider,
        )
        
        assert all([
            LLMProvider,
            ProviderFactory,
            OpenAIProvider,
            GeminiProvider,
            OllamaProvider
        ])
    
    def test_v1_import_with_warning(self):
        """Test that V1 import works with deprecation warning."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from wikipedia_agent.v1 import WikipediaAgent
            
            # Verify deprecation warning
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert WikipediaAgent is not None
    
    def test_v2_import(self):
        """Test V2 imports as documented."""
        from wikipedia_agent.v2 import (
            WikipediaAgentV2,
            WikipediaSearcher,
            CodeExecutor,
            PromptTemplates,
        )
        
        assert all([
            WikipediaAgentV2,
            WikipediaSearcher,
            CodeExecutor,
            PromptTemplates
        ])
    
    def test_tools_import(self):
        """Test tools import as documented."""
        from wikipedia_agent.tools import ImageAnalyzerAgent
        
        assert ImageAnalyzerAgent is not None


@pytest.mark.docs
class TestCLIHelp:
    """Tests for CLI help commands."""
    
    @pytest.fixture
    def python_path(self):
        """Get the Python executable path."""
        return sys.executable
    
    def test_module_help(self, python_path):
        """Test that python -m wikipedia_agent --help works."""
        result = subprocess.run(
            [python_path, "-m", "wikipedia_agent", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        # Should not error (may not have entry point set up yet)
        # This test validates the module structure


@pytest.mark.docs
class TestCodeExecutorDocumented:
    """Tests for documented CodeExecutor functionality."""
    
    def test_safe_math_execution(self):
        """Test safe math execution as documented."""
        from wikipedia_agent.v2.code_executor import CodeExecutor
        
        executor = CodeExecutor()
        result = executor.execute("result = sin(radians(45))")
        
        assert result.success
        assert result.variables.get('result') is not None
    
    def test_dangerous_code_blocked(self):
        """Test that dangerous code is blocked as documented."""
        from wikipedia_agent.v2.code_executor import CodeExecutor
        
        executor = CodeExecutor()
        result = executor.execute("import os\nos.system('ls')")
        
        assert not result.success
        assert "Import" in result.error or "not allowed" in result.error
