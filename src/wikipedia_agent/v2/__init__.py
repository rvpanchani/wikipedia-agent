"""
Wikipedia Agent V2 - Enhanced Wikipedia agent with scientific computation capabilities.

This package provides an intelligent agent that can:
1. Classify queries into categories (DIRECT/WIKIPEDIA/COMPUTE/COMBINED)
2. Search Wikipedia for information
3. Generate and execute Python code for scientific calculations
4. Use Wikipedia to find formulas and apply them programmatically

Design Principles:
- Tool Minimalism: Use tools only when they add value
- Single Classification: One decision determines the execution path
- Predictable Behavior: Clear, testable execution patterns
"""

from wikipedia_agent.v2.agent import WikipediaAgentV2, AgentResponse
from wikipedia_agent.v2.wikipedia_search import WikipediaSearcher, SearchResult
from wikipedia_agent.v2.code_executor import CodeExecutor, ExecutionResult
from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory

__version__ = "2.0.0"

__all__ = [
    # Main agent
    "WikipediaAgentV2",
    "AgentResponse",
    # Wikipedia search
    "WikipediaSearcher",
    "SearchResult",
    # Code execution
    "CodeExecutor",
    "ExecutionResult",
    # Prompts
    "PromptTemplates",
    "QueryCategory",
]
