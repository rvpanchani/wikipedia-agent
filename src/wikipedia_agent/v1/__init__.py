"""
Wikipedia Agent V1 - Legacy iterative search agent.

⚠️  DEPRECATED: This module is deprecated and will be removed in a future version.
    Please migrate to WikipediaAgentV2 from wikipedia_agent.v2

The V1 agent uses an iterative search approach with LLM-based validation.
While still functional, V2 provides better query classification and code execution.
"""

import warnings

from wikipedia_agent.v1.agent import WikipediaAgent, SearchResult

# Emit deprecation warning on import
warnings.warn(
    "wikipedia_agent.v1 is deprecated and will be removed in a future version. "
    "Please migrate to WikipediaAgentV2 from wikipedia_agent.v2",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "WikipediaAgent",
    "SearchResult",
]
