#!/usr/bin/env python3
"""
Wikipedia Agent - A simple command line agent to answer natural language questions
using Wikipedia as knowledge source and various LLM providers (OpenAI, Azure, Gemini, Ollama, etc.).
"""

import os
import sys
import argparse
import wikipedia
from dotenv import load_dotenv
from typing import List, Optional, Tuple
from providers import ProviderFactory, get_provider_config_from_env


class WikipediaAgent:
    """A simple agent that answers questions using Wikipedia and various LLM providers."""
    
    def __init__(self, provider_name: str = None, api_key: str = None, model: str = None, max_iterations: int = 3, **provider_kwargs):
        """
        Initialize the Wikipedia agent.
        
        Args:
            provider_name: LLM provider name (openai, azure, gemini, ollama, huggingface)
            api_key: API key for the provider (if required)
            model: Model name to use (provider-specific)
            max_iterations: Maximum number of search iterations
            **provider_kwargs: Additional provider-specific arguments
        """
        self.max_iterations = max_iterations
        
        # Auto-detect provider if not specified
        if not provider_name:
            provider_name = ProviderFactory.auto_detect_provider()
            if not provider_name:
                raise ValueError(
                    "No LLM provider detected. Please specify a provider and configure API keys.\n"
                    "Available providers: " + ", ".join(ProviderFactory.get_available_providers())
                )
        
        # Get provider configuration from environment if not provided
        provider_config = get_provider_config_from_env(provider_name)
        if api_key:
            provider_config['api_key'] = api_key
        if model:
            provider_config['model'] = model
        provider_config.update(provider_kwargs)
        
        # Create provider
        try:
            self.provider = ProviderFactory.create_provider(provider_name, **provider_config)
            if not self.provider.is_available():
                raise RuntimeError(f"Provider '{provider_name}' is not properly configured or available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize provider '{provider_name}': {e}")
        
        self.provider_name = provider_name
        
        # Wikipedia configuration
        wikipedia.set_lang("en")
    
    def generate_search_terms(self, question: str, previous_attempts: List[str] = None) -> List[str]:
        """
        Generate Wikipedia search terms for a given question using the LLM provider.
        
        Args:
            question: The user's question
            previous_attempts: List of previously tried search terms
            
        Returns:
            List of search terms to try
        """
        previous_info = ""
        if previous_attempts:
            previous_info = f"\nPrevious search terms that didn't work: {', '.join(previous_attempts)}"
        
        prompt = f"""
You are an expert at finding information on Wikipedia. Given a question, suggest 3-5 specific Wikipedia search terms that are most likely to contain the answer.

Question: {question}{previous_info}

Instructions:
- Focus on key entities, concepts, and topics mentioned in the question
- Use specific names, places, events, or technical terms when possible
- Avoid overly broad or generic terms
- If previous terms failed, try different angles or more specific/general variations
- Return only the search terms, one per line, without numbering or bullets

Search terms:"""

        try:
            response = self.provider.generate_content(prompt)
            search_terms = [term.strip() for term in response.strip().split('\n') if term.strip()]
            return search_terms[:5]  # Limit to 5 terms
        except Exception as e:
            print(f"Error generating search terms: {e}")
            # Fallback: extract key words from the question
            return [question]
    
    def search_wikipedia(self, search_term: str) -> Optional[str]:
        """
        Search Wikipedia for a given term and return the content.
        
        Args:
            search_term: Term to search for
            
        Returns:
            Wikipedia page content or None if not found
        """
        try:
            # Search for pages
            search_results = wikipedia.search(search_term, results=3)
            if not search_results:
                return None
            
            # Try to get the first result
            page = wikipedia.page(search_results[0])
            
            # Return summary + first few paragraphs (limit content size)
            content = page.summary
            if len(page.content) > len(page.summary):
                # Add more content but limit total size
                additional_content = page.content[len(page.summary):2000]
                content = page.summary + "\n\n" + additional_content
            
            return content
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option
            try:
                page = wikipedia.page(e.options[0])
                return page.summary
            except:
                return None
        except wikipedia.exceptions.PageError:
            return None
        except Exception as e:
            print(f"Error searching Wikipedia for '{search_term}': {e}")
            return None
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Generate an answer to the question using the Wikipedia context.
        
        Args:
            question: The user's question
            context: Wikipedia content as context
            
        Returns:
            Generated answer
        """
        prompt = f"""
Based on the following Wikipedia content, answer the user's question accurately and concisely.

Wikipedia Content:
{context}

Question: {question}

Instructions:
- Answer directly and factually based only on the provided Wikipedia content
- If the content doesn't contain enough information to answer the question, say so
- Keep the answer concise but complete
- Cite specific facts from the Wikipedia content when relevant

Answer:"""

        try:
            response = self.provider.generate_content(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def process_query(self, question: str) -> Tuple[str, List[str]]:
        """
        Process a user query through iterative Wikipedia search.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (answer, search_terms_used)
        """
        print(f"🤔 Processing question: {question}")
        
        previous_attempts = []
        search_terms_used = []
        
        for iteration in range(self.max_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate search terms
            search_terms = self.generate_search_terms(question, previous_attempts)
            print(f"🔍 Generated search terms: {', '.join(search_terms)}")
            
            # Try each search term
            for term in search_terms:
                if term in previous_attempts:
                    continue
                    
                print(f"   Searching Wikipedia for: {term}")
                content = self.search_wikipedia(term)
                
                if content:
                    print(f"   ✅ Found content ({len(content)} characters)")
                    search_terms_used.append(term)
                    
                    # Generate answer
                    answer = self.answer_question(question, content)
                    
                    # Simple check if answer seems complete
                    if len(answer) > 50 and "don't have enough information" not in answer.lower():
                        return answer, search_terms_used
                else:
                    print(f"   ❌ No content found")
                
                previous_attempts.append(term)
            
            # If we didn't find a good answer, continue to next iteration
            print(f"   No satisfactory answer found in iteration {iteration + 1}")
        
        # If we've exhausted all iterations
        return "I couldn't find a satisfactory answer to your question after searching Wikipedia with multiple terms. Please try rephrasing your question or asking about a different topic.", search_terms_used


def main():
    """Main entry point for the Wikipedia agent."""
    parser = argparse.ArgumentParser(
        description="Wikipedia Agent - Answer questions using Wikipedia and AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wikipedia_agent.py "Who was the first person to walk on the moon?"
  python wikipedia_agent.py --provider openai "What is the capital of France?"
  python wikipedia_agent.py --provider azure --model gpt-4 "How does photosynthesis work?"
  python wikipedia_agent.py --provider ollama --model llama2 "What is quantum computing?"
  python wikipedia_agent.py --max-iterations 5 "How does photosynthesis work?"

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
        help="Model to use (provider-specific, uses default if not specified)"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for the provider (can also be set via environment variables)"
    )
    
    # Legacy Gemini support for backward compatibility
    parser.add_argument(
        "--gemini-api-key",
        help="Google Gemini API key (deprecated, use --api-key with --provider gemini)"
    )
    
    # Azure-specific options
    parser.add_argument(
        "--azure-endpoint",
        help="Azure OpenAI endpoint URL"
    )
    
    parser.add_argument(
        "--azure-api-version",
        help="Azure OpenAI API version"
    )
    
    # Ollama-specific options
    parser.add_argument(
        "--ollama-base-url",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Handle legacy Gemini API key for backward compatibility
    if args.gemini_api_key and not args.provider:
        args.provider = 'gemini'
        args.api_key = args.gemini_api_key
        print("⚠️  Using deprecated --gemini-api-key. Please use --provider gemini --api-key instead.")
    
    # Handle legacy GEMINI_API_KEY environment variable
    if not args.provider and not args.api_key and os.getenv("GEMINI_API_KEY"):
        args.provider = 'gemini'
        args.api_key = os.getenv("GEMINI_API_KEY")
        print("⚠️  Using GEMINI_API_KEY environment variable. Consider using OPENAI_API_KEY or other provider-specific variables.")
    
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
        agent = WikipediaAgent(
            provider_name=args.provider,
            api_key=args.api_key,
            model=args.model,
            max_iterations=args.max_iterations,
            **provider_kwargs
        )
        
        print(f"🤖 Using {agent.provider_name} provider with model: {agent.provider.model}")
        
        # Process query
        answer, search_terms = agent.process_query(args.question)
        
        # Display results
        print("\n" + "="*60)
        print("📝 ANSWER:")
        print("="*60)
        print(answer)
        
        if search_terms:
            print(f"\n🔍 Search terms used: {', '.join(search_terms)}")
        
    except Exception as e:
        error_msg = str(e)
        if "No LLM provider detected" in error_msg or "not properly configured" in error_msg:
            print("❌ Error: No properly configured LLM provider found.")
            print("\n🔧 Configuration Help:")
            print("Set one of these environment variables:")
            print("  - OPENAI_API_KEY for OpenAI")
            print("  - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure OpenAI")
            print("  - GEMINI_API_KEY for Google Gemini")
            print("  - HUGGINGFACE_API_KEY for Hugging Face")
            print("  - Or run Ollama locally at http://localhost:11434")
            print("\nOr use command line arguments:")
            print("  python wikipedia_agent.py --provider openai --api-key YOUR_KEY \"your question\"")
            print("\nGet API keys from:")
            print("  - OpenAI: https://platform.openai.com/api-keys")
            print("  - Azure: https://portal.azure.com")
            print("  - Gemini: https://makersuite.google.com/app/apikey")
            print("  - Hugging Face: https://huggingface.co/settings/tokens")
        else:
            print(f"❌ Error: {error_msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()