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


class SearchResult:
    """Container for search result data."""
    def __init__(self, term: str, url: str = None, summary: str = None, found_relevant_info: bool = False):
        self.term = term
        self.url = url
        self.summary = summary
        self.found_relevant_info = found_relevant_info


class WikipediaAgent:
    """An intelligent agent that answers questions using Wikipedia with iterative search and validation."""
    
    def __init__(self, provider_name: str = None, api_key: str = None, model: str = None, 
                 temperature: float = 0.7, max_tokens: int = 1000, 
                 max_iterations: int = 3, **provider_kwargs):
        """
        Initialize the Wikipedia agent.
        
        Args:
            provider_name: LLM provider name (openai, azure, gemini, ollama, huggingface)
            api_key: API key for the provider (if required)
            model: Model name to use (provider-specific)
            temperature: Sampling temperature for generation (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
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
        
        # Set generation parameters
        provider_config['temperature'] = temperature
        provider_config['max_tokens'] = max_tokens
        
        # Set default system prompt for Wikipedia agent
        provider_config['system_prompt'] = (
            "You are a helpful AI assistant that helps users find information from Wikipedia. "
            "You are knowledgeable, precise, and always provide accurate information based on the content provided to you. "
            "When generating search terms, focus on specific entities, events, people, and concepts. "
            "When answering questions, be clear, concise, and factual."
        )
        
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
        
        search_system_prompt = (
            "You are an expert at finding information on Wikipedia. Your task is to generate the most "
            "effective search terms for finding relevant Wikipedia articles. Be specific and precise."
        )
        
        prompt = f"""Given a question, suggest 3-5 specific Wikipedia search terms that are most likely to contain the answer.

Question: {question}{previous_info}

Instructions:
- Focus on key entities, concepts, and topics mentioned in the question
- Use specific names, places, events, or technical terms when possible
- Avoid overly broad or generic terms
- If previous terms failed, try different angles or more specific/general variations
- Return only the search terms, one per line, without numbering or bullets

Search terms:"""

        try:
            response = self.provider.generate_content(prompt, system_prompt=search_system_prompt)
            
            # Clean response from thinking tokens and other noise
            cleaned_response = response
            # Remove thinking tokens if present
            if '<think>' in response:
                # Extract content after </think> or filter out thinking content
                import re
                cleaned_response = re.sub(
                    r'<think>.*?</think>', '', response, flags=re.DOTALL
                )
                # Also handle unclosed think tags
                cleaned_response = re.sub(
                    r'<think>.*', '', cleaned_response, flags=re.DOTALL
                )
            
            # Remove common prefixes and clean up
            cleaned_response = (cleaned_response.replace('- "', '')
                              .replace('"', '').replace('- ', ''))
            search_terms = [
                term.strip() for term in cleaned_response.strip().split('\n') 
                if term.strip()
            ]
            # Filter out empty terms and non-search term content
            search_terms = [
                term for term in search_terms 
                if term and not term.startswith('Search terms:')
            ]
            return search_terms[:5]  # Limit to 5 terms
        except Exception as e:
            print(f"Error generating search terms: {e}")
            # Fallback: extract key words from the question
            return [question]
    
    def search_wikipedia(self, search_term: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Search Wikipedia for a given term and return the content and URL.
        
        Args:
            search_term: Term to search for
            
        Returns:
            Tuple of (content, url) or (None, None) if not found
        """
        try:
            # Search for pages
            search_results = wikipedia.search(search_term, results=3)
            if not search_results:
                return None, None
            
            # Try to get the first result
            page = wikipedia.page(search_results[0])
            
            # Return summary + first few paragraphs (limit content size)
            content = page.summary
            if len(page.content) > len(page.summary):
                # Add more content but limit total size
                additional_content = page.content[len(page.summary):2000]
                content = page.summary + "\n\n" + additional_content
            
            return content, page.url
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option
            try:
                page = wikipedia.page(e.options[0])
                return page.summary, page.url
            except Exception:
                return None, None
        except wikipedia.exceptions.PageError:
            return None, None
        except Exception as e:
            print(f"Error searching Wikipedia for '{search_term}': {e}")
            return None, None
    
    def validate_answer_completeness(self, question: str, answer: str) -> bool:
        """
        Validate if the generated answer adequately addresses the question.
        
        Args:
            question: The original question
            answer: The generated answer
            
        Returns:
            True if answer is satisfactory, False otherwise
        """
        # Clean the answer first
        cleaned_answer = self._clean_llm_response(answer)
        
        # Simple heuristic validation (more reliable than LLM validation for this model)
        if len(cleaned_answer) < 20:
            return False
            
        # Check for obvious non-answers
        non_answer_phrases = [
            "don't have enough information",
            "not enough information", 
            "cannot answer",
            "insufficient information",
            "more information needed",
            "unable to determine"
        ]
        
        cleaned_lower = cleaned_answer.lower()
        for phrase in non_answer_phrases:
            if phrase in cleaned_lower:
                return False
                
        # If it's a reasonable length and doesn't contain non-answer phrases, consider it valid
        return True
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Generate an answer to the question using the Wikipedia context.
        
        Args:
            question: The user's question
            context: Wikipedia content as context
            
        Returns:
            Generated answer
        """
        answer_system_prompt = (
            "You are a knowledgeable assistant that provides accurate answers based on Wikipedia content. "
            "Always base your answers strictly on the provided information and be precise and factual. "
            "Do not include thinking tokens or reasoning in your response - provide only the direct answer."
        )
        
        prompt = f"""Based on the following Wikipedia content, answer the user's question accurately and concisely.

Wikipedia Content:
{context}

Question: {question}

Instructions:
- Answer directly and factually based only on the provided Wikipedia content
- If the content doesn't contain enough information to answer the question, say so
- Keep the answer concise but complete
- Cite specific facts from the Wikipedia content when relevant
- Do not include any thinking process or reasoning tags in your response

Answer:"""

        try:
            response = self.provider.generate_content(prompt, system_prompt=answer_system_prompt)
            # Clean up thinking tokens and other artifacts
            cleaned_response = self._clean_llm_response(response)
            return cleaned_response.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _clean_llm_response(self, response: str) -> str:
        """
        Clean up LLM response by removing thinking tokens and other artifacts.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response
        """
        import re
        
        # First, try to extract content between thinking tags and actual answer
        if "<think>" in response and "Human:" in response:
            # Extract the part after the thinking section and before "Human:"
            parts = response.split("Human:")
            if len(parts) > 1:
                # Look for "Answer:" in the first part after thinking
                first_part = parts[0]
                if "Answer:" in first_part:
                    answer_parts = first_part.split("Answer:")
                    if len(answer_parts) > 1:
                        cleaned = answer_parts[-1].strip()
                        if len(cleaned) > 10:
                            return cleaned
        
        # Remove thinking tokens
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        
        # Remove other common artifacts
        cleaned = re.sub(r'<\|endoftext\|>.*', '', cleaned, flags=re.DOTALL)
        
        # Split on "Human:" and keep only the first part
        if "Human:" in cleaned:
            cleaned = cleaned.split("Human:")[0]
        
        # Look for "Answer:" pattern and extract what follows
        if "Answer:" in cleaned:
            answer_parts = cleaned.split("Answer:")
            if len(answer_parts) > 1:
                cleaned = answer_parts[-1].strip()
        
        # Clean up multiple newlines and whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = cleaned.strip()
        
        # If response is still empty or too short after cleaning, try a different approach
        if len(cleaned) < 10:
            # Look for any substantial text after common prefixes
            lines = response.split('\n')
            substantial_lines = [line.strip() for line in lines if len(line.strip()) > 20 and 
                               not line.strip().startswith('<') and 
                               'think>' not in line.lower() and
                               'human:' not in line.lower()]
            if substantial_lines:
                return substantial_lines[0]
        
        return cleaned if len(cleaned) >= 10 else response.strip()
    
    def summarize_search_finding(self, question: str, search_term: str, content: str) -> str:
        """
        Generate a brief summary of what was found for a search term in relation to the question.
        
        Args:
            question: The original question
            search_term: The search term used
            content: The Wikipedia content found
            
        Returns:
            Brief summary of relevant findings
        """
        summary_system_prompt = (
            "You are an expert at extracting and summarizing relevant information. "
            "Create concise summaries that capture the key points relevant to the question."
        )
        
        prompt = f"""Summarize what information from this Wikipedia content is relevant to answering the question. Be very brief (2-3 sentences max).

Question: {question}
Search term: {search_term}

Wikipedia Content:
{content}

Brief summary of relevant findings:"""

        try:
            response = self.provider.generate_content(prompt, system_prompt=summary_system_prompt)
            return response.strip()
        except Exception as e:
            return f"Found content for '{search_term}' but could not summarize."
    
    def process_query(self, question: str, verbose: bool = False) -> Tuple[str, List[str]]:
        """
        Process a user query through iterative Wikipedia search with answer validation.
        
        Args:
            question: The user's question
            verbose: Whether to print intermediate steps
            
        Returns:
            Tuple of (answer, citation_urls)
        """
        if verbose:
            print(f"🤔 Processing question: {question}")
        
        search_history = []  # List of SearchResult objects
        previous_attempts = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate search terms based on question and previous attempts
            search_terms = self.generate_search_terms(question, previous_attempts)
            if verbose:
                print(f"🔍 Generated search terms: {', '.join(search_terms)}")
            
            # Try each search term
            for term in search_terms:
                if term in previous_attempts:
                    continue
                    
                if verbose:
                    print(f"   Searching Wikipedia for: {term}")
                
                # Search Wikipedia
                content, url = self.search_wikipedia(term)
                
                if content and url:
                    if verbose:
                        print(f"   ✅ Found content ({len(content)} characters)")
                    
                    # Generate answer from this content
                    answer = self.answer_question(question, content)
                    
                    # Validate if this answer is satisfactory
                    is_satisfactory = self.validate_answer_completeness(question, answer)
                    
                    if is_satisfactory:
                        # We found a satisfactory answer!
                        search_result = SearchResult(
                            term=term, 
                            url=url, 
                            summary=self.summarize_search_finding(question, term, content),
                            found_relevant_info=True
                        )
                        search_history.append(search_result)
                        
                        # Return the final answer with citation
                        citation_urls = list(dict.fromkeys([result.url for result in search_history if result.url]))
                        return answer, citation_urls
                    else:
                        # Answer not satisfactory, but record what we found
                        search_result = SearchResult(
                            term=term,
                            url=url,
                            summary=self.summarize_search_finding(question, term, content),
                            found_relevant_info=False
                        )
                        search_history.append(search_result)
                        
                        if verbose:
                            print(f"   ℹ️ Found some info but answer not complete, continuing search...")
                else:
                    if verbose:
                        print(f"   ❌ No content found")
                
                previous_attempts.append(term)
            
            # If we didn't find a satisfactory answer, continue to next iteration
            if verbose:
                print(f"   No complete answer found in iteration {iteration + 1}")
        
        # If we've exhausted all iterations, try to combine information from all searches
        if search_history:
            combined_context = "\n\n".join([
                f"From '{result.term}': {result.summary}" 
                for result in search_history 
                if result.summary
            ])
            
            if combined_context:
                final_answer = self.answer_question(question, combined_context)
                citation_urls = list(dict.fromkeys([result.url for result in search_history if result.url]))
                return final_answer, citation_urls
        
        # If we still couldn't find anything useful
        return ("I couldn't find a satisfactory answer to your question after searching Wikipedia with multiple terms. "
                "Please try rephrasing your question or asking about a different topic."), []


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
        default=10,
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
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (0.0 to 2.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate (default: 1000)"
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
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_iterations=args.max_iterations,
            **provider_kwargs
        )
        
        # Process query (verbose mode for debugging if needed)
        verbose_mode = os.getenv("WIKIPEDIA_AGENT_VERBOSE", "false").lower() == "true"
        answer, citation_urls = agent.process_query(args.question, verbose=verbose_mode)
        
        # Display clean results
        print("\n" + "="*60)
        print("📝 ANSWER:")
        print("="*60)
        print(answer)
        
        # Display Wikipedia sources
        if citation_urls:
            print(f"\n� Sources:")
            for i, url in enumerate(citation_urls, 1):
                print(f"  {i}. {url}")
        else:
            print("\n📚 No specific Wikipedia sources found.")
        
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