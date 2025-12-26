#!/usr/bin/env python3
"""
Wikipedia Agent V1 - Iterative Search Agent (DEPRECATED).

⚠️  DEPRECATED: This module is deprecated and will be removed in a future version.
    Please migrate to WikipediaAgentV2 from wikipedia_agent.v2

This agent uses an iterative Wikipedia search approach with LLM-based validation
to answer questions. While still functional, V2 provides better query classification
and code execution capabilities.

Usage (deprecated):
    from wikipedia_agent.v1 import WikipediaAgent

    agent = WikipediaAgent(provider_name="ollama", model="qwen3:0.6b")
    answer, sources = agent.process_query("Who was Albert Einstein?")
"""

import warnings
import wikipedia
from typing import List, Optional, Tuple

from wikipedia_agent.core.providers import ProviderFactory, get_provider_config_from_env
from wikipedia_agent.core.utils import clean_llm_response, extract_final_answer


class SearchResult:
    """Container for search result data."""

    def __init__(
        self,
        term: str,
        url: str = None,
        summary: str = None,
        found_relevant_info: bool = False
    ):
        self.term = term
        self.url = url
        self.summary = summary
        self.found_relevant_info = found_relevant_info


class WikipediaAgent:
    """
    An intelligent agent that answers questions using Wikipedia with iterative search and validation.

    ⚠️  DEPRECATED: Use WikipediaAgentV2 instead for better query classification
        and code execution capabilities.
    """

    def __init__(
        self,
        provider_name: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_iterations: int = 3,
        **provider_kwargs
    ):
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
        # Emit deprecation warning
        warnings.warn(
            "WikipediaAgent (V1) is deprecated. Please migrate to WikipediaAgentV2 "
            "from wikipedia_agent.v2 for improved functionality.",
            DeprecationWarning,
            stacklevel=2
        )

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
        provider_config['system_prompt'] = self._get_system_prompt()
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

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Wikipedia agent."""
        return """You are a Wikipedia information assistant. Your role is to help users find and understand information from Wikipedia articles.

# Core Responsibilities
- Search for relevant Wikipedia content based on user queries
- Provide accurate, factual information drawn directly from Wikipedia sources
- Cite specific articles when referencing information
- Clarify when information is not available in the provided Wikipedia content

# Search Term Guidelines
When generating Wikipedia search queries:
- Use specific proper nouns (names of people, places, organizations)
- Focus on discrete topics and concepts rather than broad categories
- Prefer Wikipedia article titles over generic phrases
- For multi-part questions, break them into separate targeted searches

Examples:
- Good: "Marie Curie", "French Revolution", "photosynthesis"
- Avoid: "famous scientists", "important historical events", "science topics"

# Response Guidelines
- Be concise: Provide direct answers without unnecessary elaboration
- Be precise: Use exact terminology and figures from Wikipedia sources
- Be factual: Never speculate or add information beyond what Wikipedia provides
- Indicate uncertainty: If Wikipedia content is ambiguous or incomplete, say so explicitly

# When Wikipedia Lacks Information
If the requested information isn't available in Wikipedia:
- State this clearly and directly
- Suggest related topics that might be available
- Do not fabricate or infer information from incomplete sources
"""

    def generate_search_terms(
        self,
        question: str,
        previous_attempts: List[str] = None
    ) -> List[str]:
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

            # Clean response
            cleaned_response = clean_llm_response(response)

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
        Validate if the generated answer adequately addresses the question using pattern recognition.

        Args:
            question: The original question
            answer: The generated answer with structured patterns

        Returns:
            True if answer is satisfactory, False otherwise
        """
        # Clean the answer first
        cleaned_answer = clean_llm_response(answer)

        # Check for structured response patterns
        if cleaned_answer.startswith("ANSWER_FOUND:"):
            # Extract the actual answer part
            actual_answer = cleaned_answer[len("ANSWER_FOUND:"):].strip()
            # Ensure the answer has substantial content
            return len(actual_answer) >= 10 and actual_answer.lower() not in [
                "none", "unknown", "unclear", "not specified", "not mentioned"
            ]
        elif cleaned_answer.startswith("INSUFFICIENT_INFO:"):
            return False
        else:
            # Fallback to LLM-based validation for non-structured responses
            return self._validate_answer_with_llm(question, cleaned_answer)

    def answer_question(self, question: str, context: str) -> str:
        """
        Generate an answer to the question using the Wikipedia context with structured response patterns.

        Args:
            question: The user's question
            context: Wikipedia content as context

        Returns:
            Generated answer with structured patterns for validation
        """
        answer_system_prompt = (
            "You are a knowledgeable assistant that provides accurate answers based on Wikipedia content. "
            "You must respond using specific patterns to indicate whether you found sufficient information. "
            "Always base your answers strictly on the provided information and be precise and factual."
        )

        prompt = f"""Based on the following Wikipedia content, answer the user's question using the specified response pattern.

Wikipedia Content:
{context}

Question: {question}

CRITICAL: You MUST respond using EXACTLY one of these two formats. Do not add any other text:

Format 1 - If the Wikipedia content contains sufficient information to answer the question:
ANSWER_FOUND: [your complete factual answer based on the Wikipedia content]

Format 2 - If the Wikipedia content does NOT contain sufficient information:
INSUFFICIENT_INFO: [brief explanation of what information is missing]

IMPORTANT:
- Start your response with either "ANSWER_FOUND:" or "INSUFFICIENT_INFO:"
- Do not include any reasoning, thinking, or additional commentary
- Do not use any other format
- Base your determination strictly on the provided Wikipedia content

Response:"""

        try:
            response = self.provider.generate_content(prompt, system_prompt=answer_system_prompt)
            cleaned_response = clean_llm_response(response)
            return cleaned_response.strip()
        except Exception as e:
            return f"INSUFFICIENT_INFO: Error generating answer: {e}"

    def _validate_answer_with_llm(self, question: str, answer: str) -> bool:
        """
        Use LLM to validate if an answer adequately addresses the question (fallback method).

        Args:
            question: The original question
            answer: The generated answer

        Returns:
            True if answer is satisfactory, False otherwise
        """
        validation_system_prompt = (
            "You are an expert at evaluating whether answers adequately address questions. "
            "Respond with exactly 'VALID' or 'INVALID' based on the criteria provided."
        )

        prompt = f"""Evaluate whether this answer adequately addresses the given question.

Question: {question}
Answer: {answer}

Criteria for VALID answer:
- Directly addresses the question asked
- Contains specific, factual information
- Is not vague or evasive
- Provides meaningful content (not just "I don't know" type responses)

Criteria for INVALID answer:
- Doesn't address the question
- Contains only vague or non-specific information
- Explicitly states lack of information without providing any relevant details
- Is too short or lacks substance

Respond with exactly one word: VALID or INVALID

Evaluation:"""

        try:
            response = self.provider.generate_content(prompt, system_prompt=validation_system_prompt)
            cleaned_response = response.strip().upper()
            return cleaned_response == "VALID"
        except Exception:
            # Fallback to basic length check if LLM validation fails
            return len(answer.strip()) >= 20

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
        except Exception:
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
                        citation_urls = list(dict.fromkeys(
                            [result.url for result in search_history if result.url]
                        ))
                        clean_answer = extract_final_answer(answer)
                        return clean_answer, citation_urls
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
                            print("   ℹ️ Found some info but answer not complete, continuing search...")
                else:
                    if verbose:
                        print("   ❌ No content found")

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
                citation_urls = list(dict.fromkeys(
                    [result.url for result in search_history if result.url]
                ))
                clean_answer = extract_final_answer(final_answer)
                return clean_answer, citation_urls

        # If we still couldn't find anything useful
        return (
            "I couldn't find a satisfactory answer to your question after searching Wikipedia with multiple terms. "
            "Please try rephrasing your question or asking about a different topic."
        ), []
