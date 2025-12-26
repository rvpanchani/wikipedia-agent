"""
Wikipedia Agent - A unified package for Wikipedia-based Q&A agents.

This package provides:
- V2 Agent: Modern agent with query classification and code execution (recommended)
- V1 Agent: Legacy agent with iterative search (deprecated, maintained for compatibility)
- Core modules: Shared providers and utilities
- Tools: Image analysis and other utilities

Usage:
    # Recommended: Use V2 agent
    from wikipedia_agent import WikipediaAgentV2
    agent = WikipediaAgentV2()
    response = agent.query("What is the speed of light?")

    # Legacy: V1 agent (deprecated)
    from wikipedia_agent.v1 import WikipediaAgent
    agent = WikipediaAgent()
    answer, sources = agent.process_query("What is photosynthesis?")
"""

__version__ = "2.0.0"

# Import main components for convenient access
from wikipedia_agent.core.providers import (
    LLMProvider,
    OpenAIProvider,
    AzureOpenAIProvider,
    GeminiProvider,
    OllamaProvider,
    HuggingFaceProvider,
    ProviderFactory,
    get_provider_config_from_env,
)

from wikipedia_agent.v2 import WikipediaAgentV2

__all__ = [
    # Version
    "__version__",
    # Main agent (V2)
    "WikipediaAgentV2",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "ProviderFactory",
    "get_provider_config_from_env",
]
