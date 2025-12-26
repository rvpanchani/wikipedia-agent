"""
Test configuration and shared fixtures for Wikipedia Agent tests.
"""

import os
import sys
import pytest
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def available_provider():
    """
    Fixture that detects and returns an available LLM provider.
    
    Returns a tuple of (provider_name, provider_config) or skips test if none available.
    """
    providers = []
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", {"api_key": os.getenv("OPENAI_API_KEY")}))
    
    # Check Gemini
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini", {"api_key": os.getenv("GEMINI_API_KEY")}))
    
    # Check Azure OpenAI
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        providers.append(("azure", {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
        }))
    
    # Check Hugging Face
    if os.getenv("HUGGINGFACE_API_KEY"):
        providers.append(("huggingface", {"api_key": os.getenv("HUGGINGFACE_API_KEY")}))
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            providers.append(("ollama", {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "model": os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
            }))
    except Exception:
        pass
    
    if not providers:
        pytest.skip("No LLM provider available for testing")
    
    return providers[0]


@pytest.fixture
def v2_agent(available_provider):
    """Create a WikipediaAgentV2 instance for testing."""
    from wikipedia_agent.v2 import WikipediaAgentV2
    
    provider_name, provider_config = available_provider
    return WikipediaAgentV2(
        provider_name=provider_name,
        max_search_attempts=1,
        verbose=False,
        **provider_config
    )


@pytest.fixture
def v1_agent(available_provider):
    """Create a WikipediaAgent V1 instance for testing (with suppressed deprecation warning)."""
    from wikipedia_agent.v1 import WikipediaAgent
    
    provider_name, provider_config = available_provider
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return WikipediaAgent(
            provider_name=provider_name,
            max_iterations=1,
            **provider_config
        )


@pytest.fixture
def code_executor():
    """Create a CodeExecutor instance for testing."""
    from wikipedia_agent.v2.code_executor import CodeExecutor
    return CodeExecutor()


@pytest.fixture
def wikipedia_searcher():
    """Create a WikipediaSearcher instance for testing."""
    from wikipedia_agent.v2.wikipedia_search import WikipediaSearcher
    return WikipediaSearcher()
