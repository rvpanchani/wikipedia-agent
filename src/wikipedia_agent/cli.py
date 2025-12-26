#!/usr/bin/env python3
"""
Unified CLI for Wikipedia Agent package.

This module provides command-line entry points for:
- wikipedia-agent: Main V2 agent (recommended)
- wikipedia-agent-v1: Legacy V1 agent (deprecated)
- wikipedia-agent-image: Image analysis tool

Usage:
    wikipedia-agent "What is photosynthesis?"
    wikipedia-agent --provider ollama --model qwen3:0.6b "Calculate sin(45°)"
"""

import os
import sys
import argparse
from dotenv import load_dotenv


def main_v2():
    """
    Main entry point for Wikipedia Agent V2.

    This is the recommended agent with query classification and code execution.
    """
    from wikipedia_agent.v2.agent import WikipediaAgentV2
    from wikipedia_agent.core.providers import ProviderFactory

    parser = argparse.ArgumentParser(
        description="Wikipedia Agent - Answer questions using Wikipedia and AI with scientific computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple question
  wikipedia-agent "Who was Albert Einstein?"

  # Scientific calculation
  wikipedia-agent "Calculate sin(45 degrees) using the definition"

  # Complex formula application
  wikipedia-agent "Using Bernoulli's equation, if water flows at 2 m/s at pressure 100000 Pa, what is the pressure when speed is 4 m/s?"

  # With specific provider
  wikipedia-agent --provider ollama --model qwen3:0.6b "What is the quadratic formula?"

  # Verbose mode for debugging
  wikipedia-agent --verbose "Calculate the area of a circle with radius 5"

Supported providers: openai, azure, gemini, ollama, huggingface
        """
    )

    parser.add_argument(
        "question",
        help="The question to answer"
    )

    parser.add_argument(
        "--provider",
        choices=['openai', 'azure', 'azure-openai', 'gemini', 'google', 'ollama', 'huggingface', 'hf'],
        help="LLM provider to use (auto-detected if not specified)"
    )

    parser.add_argument(
        "--model",
        help="Model to use (provider-specific)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 2.0, default: 0.7)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
    )

    parser.add_argument(
        "--max-search-attempts",
        type=int,
        default=2,
        help="Maximum Wikipedia search attempts (default: 2)"
    )

    parser.add_argument(
        "--api-key",
        help="API key for the provider"
    )

    parser.add_argument(
        "--no-code-execution",
        action="store_true",
        help="Disable automatic code generation and execution"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output showing intermediate steps"
    )

    # Provider-specific options
    parser.add_argument(
        "--azure-endpoint",
        help="Azure OpenAI endpoint URL"
    )

    parser.add_argument(
        "--azure-api-version",
        help="Azure OpenAI API version"
    )

    parser.add_argument(
        "--ollama-base-url",
        help="Ollama base URL (default: http://localhost:11434)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Also check for verbose env var
    if os.getenv("WIKIPEDIA_AGENT_VERBOSE", "").lower() == "true":
        args.verbose = True

    # Prepare provider arguments
    provider_kwargs = {}
    if args.azure_endpoint:
        provider_kwargs['azure_endpoint'] = args.azure_endpoint
    if args.azure_api_version:
        provider_kwargs['api_version'] = args.azure_api_version
    if args.ollama_base_url:
        provider_kwargs['base_url'] = args.ollama_base_url

    try:
        # Initialize agent
        agent = WikipediaAgentV2(
            provider_name=args.provider,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_search_attempts=args.max_search_attempts,
            enable_code_execution=not args.no_code_execution,
            verbose=args.verbose,
            **provider_kwargs
        )

        # Process query
        output = agent.ask(args.question)
        print("\n" + output)

    except Exception as e:
        _print_provider_error(str(e))
        sys.exit(1)


def main_v1():
    """
    Entry point for Wikipedia Agent V1 (deprecated).

    This is the legacy agent with iterative search.
    """
    import warnings
    warnings.warn(
        "wikipedia-agent-v1 is deprecated. Please use wikipedia-agent (V2) instead.",
        DeprecationWarning
    )

    from wikipedia_agent.v1.agent import WikipediaAgent
    from wikipedia_agent.core.providers import ProviderFactory

    parser = argparse.ArgumentParser(
        description="Wikipedia Agent V1 (DEPRECATED) - Answer questions using Wikipedia and AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  DEPRECATED: This command is deprecated. Please use 'wikipedia-agent' instead.

Examples:
  wikipedia-agent-v1 "Who was the first person to walk on the moon?"
  wikipedia-agent-v1 --provider openai "What is the capital of France?"

Supported providers: openai, azure, gemini, ollama, huggingface
        """
    )

    parser.add_argument(
        "question",
        help="The question to answer"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of search iterations (default: 3)"
    )

    parser.add_argument(
        "--provider",
        choices=['openai', 'azure', 'azure-openai', 'gemini', 'google', 'ollama', 'huggingface', 'hf'],
        help="LLM provider to use (auto-detected if not specified)"
    )

    parser.add_argument(
        "--model",
        help="Model to use (provider-specific)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 to 2.0, default: 0.7)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
    )

    parser.add_argument(
        "--api-key",
        help="API key for the provider"
    )

    # Provider-specific options
    parser.add_argument(
        "--azure-endpoint",
        help="Azure OpenAI endpoint URL"
    )

    parser.add_argument(
        "--azure-api-version",
        help="Azure OpenAI API version"
    )

    parser.add_argument(
        "--ollama-base-url",
        help="Ollama base URL (default: http://localhost:11434)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Prepare provider arguments
    provider_kwargs = {}
    if args.azure_endpoint:
        provider_kwargs['azure_endpoint'] = args.azure_endpoint
    if args.azure_api_version:
        provider_kwargs['api_version'] = args.azure_api_version
    if args.ollama_base_url:
        provider_kwargs['base_url'] = args.ollama_base_url

    try:
        # Initialize agent (suppress deprecation warning since we already warned)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = WikipediaAgent(
                provider_name=args.provider,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_iterations=args.max_iterations,
                **provider_kwargs
            )

        # Process query
        verbose_mode = os.getenv("WIKIPEDIA_AGENT_VERBOSE", "false").lower() == "true"
        answer, citation_urls = agent.process_query(args.question, verbose=verbose_mode)

        # Display clean results
        print("\n" + "=" * 60)
        print("📝 ANSWER:")
        print("=" * 60)
        print(answer)

        # Display Wikipedia sources
        if citation_urls:
            print("\n📚 Sources:")
            for i, url in enumerate(citation_urls, 1):
                print(f"  {i}. {url}")
        else:
            print("\n📚 No specific Wikipedia sources found.")

    except Exception as e:
        _print_provider_error(str(e))
        sys.exit(1)


def main_image():
    """Entry point for the image analyzer tool."""
    from wikipedia_agent.tools.image_analyser import main
    main()


def _print_provider_error(error_msg: str) -> None:
    """Print a helpful error message for provider configuration issues."""
    if "No LLM provider" in error_msg or "not available" in error_msg:
        print("❌ Error: No properly configured LLM provider found.")
        print("\n🔧 Configuration Help:")
        print("Set one of these environment variables:")
        print("  - OPENAI_API_KEY for OpenAI")
        print("  - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure")
        print("  - GEMINI_API_KEY for Google Gemini")
        print("  - HUGGINGFACE_API_KEY for Hugging Face")
        print("  - Or run Ollama locally at http://localhost:11434")
        print("\nOr use command line arguments:")
        print("  wikipedia-agent --provider openai --api-key YOUR_KEY \"your question\"")
    else:
        print(f"❌ Error: {error_msg}")


# Aliases for backwards compatibility
main = main_v2


if __name__ == "__main__":
    main()
