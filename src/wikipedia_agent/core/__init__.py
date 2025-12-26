"""
Core modules shared between V1 and V2 agents.

This package contains:
- providers: LLM provider abstraction layer
- utils: Common utility functions
"""

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

from wikipedia_agent.core.utils import (
    clean_llm_response,
    extract_numbers,
    format_number,
    detect_angle_unit,
    parse_math_expression,
    extract_code_block,
    truncate_text,
    is_scientific_question,
)

__all__ = [
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "ProviderFactory",
    "get_provider_config_from_env",
    # Utils
    "clean_llm_response",
    "extract_numbers",
    "format_number",
    "detect_angle_unit",
    "parse_math_expression",
    "extract_code_block",
    "truncate_text",
    "is_scientific_question",
]
