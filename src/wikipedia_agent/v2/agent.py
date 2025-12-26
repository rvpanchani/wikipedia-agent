#!/usr/bin/env python3
"""
Wikipedia Agent v2 - Redesigned with Tool Minimalism Architecture.

This agent follows a strict decision flow:
1. Classify query into one of four categories
2. Execute ONLY the tools required for that category
3. Synthesize and return the response

Categories:
- DIRECT: Answer from internal knowledge (no tools)
- WIKIPEDIA: Factual lookup only
- COMPUTE: Pure calculation only
- COMBINED: Wikipedia + reasoning + code

Design Principles:
- Tool Minimalism: Use tools only when they add value
- Single Classification: One decision determines the path
- Predictable Behavior: Clear, testable execution paths
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from wikipedia_agent.core.providers import (
    ProviderFactory,
    get_provider_config_from_env,
)
from wikipedia_agent.v2.wikipedia_search import WikipediaSearcher, SearchResult
from wikipedia_agent.v2.code_executor import CodeExecutor, ExecutionResult
from wikipedia_agent.v2.prompts import PromptTemplates, QueryCategory


@dataclass
class AgentResponse:
    """Container for agent response data."""
    answer: str
    sources: List[str]
    category: QueryCategory = QueryCategory.DIRECT
    code_executed: Optional[str] = None
    code_result: Optional[str] = None

    def format_output(self) -> str:
        """Format the response for display."""
        output_parts = []

        output_parts.append("=" * 60)
        output_parts.append("📝 ANSWER:")
        output_parts.append("=" * 60)
        output_parts.append(self.answer)

        if self.code_executed:
            output_parts.append("\n" + "-" * 40)
            output_parts.append("🔢 CODE EXECUTED:")
            output_parts.append("-" * 40)
            output_parts.append(f"```python\n{self.code_executed}\n```")

            if self.code_result:
                output_parts.append(f"\n📊 Result: {self.code_result}")

        if self.sources:
            output_parts.append("\n📚 Sources:")
            for i, url in enumerate(self.sources, 1):
                output_parts.append(f"  {i}. {url}")

        return "\n".join(output_parts)


class WikipediaAgentV2:
    """
    Wikipedia Agent with Tool Minimalism Architecture.

    Execution Flow:
    1. Classify → Determine category (DIRECT/WIKIPEDIA/COMPUTE/COMBINED)
    2. Execute → Use minimal tools for that category
    3. Return → Synthesized response
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_search_attempts: int = 2,
        enable_code_execution: bool = True,
        verbose: bool = False,
        **provider_kwargs
    ):
        """
        Initialize the Wikipedia Agent v2.

        Args:
            provider_name: LLM provider name (openai, azure, gemini, ollama, etc.)
            api_key: API key for the provider
            model: Model name to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            max_search_attempts: Maximum Wikipedia search attempts (default: 2)
            enable_code_execution: Whether to enable code execution
            verbose: Whether to print intermediate steps
            **provider_kwargs: Additional provider-specific arguments
        """
        self.max_search_attempts = max_search_attempts
        self.enable_code_execution = enable_code_execution
        self.verbose = verbose

        # Initialize prompts
        self.prompts = PromptTemplates()

        # Initialize provider
        self._init_provider(
            provider_name, api_key, model, temperature,
            max_tokens, **provider_kwargs
        )

        # Initialize components (lazy - only used when needed)
        self._searcher: Optional[WikipediaSearcher] = None
        self._code_executor: Optional[CodeExecutor] = None

    @property
    def searcher(self) -> WikipediaSearcher:
        """Lazy initialization of Wikipedia searcher."""
        if self._searcher is None:
            self._searcher = WikipediaSearcher()
        return self._searcher

    @property
    def code_executor(self) -> CodeExecutor:
        """Lazy initialization of code executor."""
        if self._code_executor is None:
            self._code_executor = CodeExecutor()
        return self._code_executor

    def _init_provider(
        self,
        provider_name: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        **provider_kwargs
    ) -> None:
        """Initialize the LLM provider."""
        if not provider_name:
            provider_name = ProviderFactory.auto_detect_provider()
            if not provider_name:
                raise ValueError(
                    "No LLM provider detected. Please specify a provider.\n"
                    f"Available: {', '.join(ProviderFactory.get_available_providers())}"
                )

        provider_config = get_provider_config_from_env(provider_name)
        if api_key:
            provider_config['api_key'] = api_key
        if model:
            provider_config['model'] = model

        provider_config['temperature'] = temperature
        provider_config['max_tokens'] = max_tokens
        provider_config['system_prompt'] = self.prompts.AGENT_SYSTEM
        provider_config.update(provider_kwargs)

        try:
            self.provider = ProviderFactory.create_provider(provider_name, **provider_config)
            if not self.provider.is_available():
                raise RuntimeError(f"Provider '{provider_name}' not available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize provider: {e}")

        self.provider_name = provider_name

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _clean_response(self, response: str) -> str:
        """Clean LLM response from thinking tokens and artifacts."""
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<\|endoftext\|>.*', '', cleaned, flags=re.DOTALL)

        if "Human:" in cleaned:
            cleaned = cleaned.split("Human:")[0]

        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()

    # ==========================================================================
    # STEP 1: QUERY CLASSIFICATION (Mandatory First Step)
    # ==========================================================================

    def _classify_query(self, question: str) -> QueryCategory:
        """
        Classify the query into one of four categories.

        This is the ONLY decision point that determines tool usage.
        """
        prompt = self.prompts.classify_query_prompt(question)

        try:
            response = self.provider.generate_content(prompt)
            cleaned = self._clean_response(response)
            self._log(f"📋 Classification: {cleaned}")

            category = self.prompts.parse_classification(cleaned)

            # Override: If code execution is disabled, downgrade COMPUTE/COMBINED
            if not self.enable_code_execution:
                if category == QueryCategory.COMPUTE:
                    self._log("⚠️ Code execution disabled, using DIRECT")
                    return QueryCategory.DIRECT
                elif category == QueryCategory.COMBINED:
                    self._log("⚠️ Code execution disabled, using WIKIPEDIA")
                    return QueryCategory.WIKIPEDIA

            return category

        except Exception as e:
            self._log(f"⚠️ Classification failed: {e}, defaulting to DIRECT")
            return QueryCategory.DIRECT

    # ==========================================================================
    # STEP 2: CATEGORY-SPECIFIC EXECUTION
    # ==========================================================================

    def _execute_direct(self, question: str) -> AgentResponse:
        """
        Category A: Answer directly from LLM knowledge.

        No tools used - pure internal knowledge.
        """
        self._log("🧠 Using internal knowledge (no tools)")

        prompt = self.prompts.direct_answer_prompt(question)
        response = self.provider.generate_content(prompt)
        answer = self._clean_response(response)

        return AgentResponse(
            answer=answer,
            sources=[],
            category=QueryCategory.DIRECT
        )

    def _execute_wikipedia(self, question: str) -> AgentResponse:
        """
        Category B: Factual lookup from Wikipedia.

        Tools used: Wikipedia search only.
        """
        self._log("📖 Using Wikipedia search")

        sources = []

        # Generate search term
        term_prompt = self.prompts.generate_search_term_prompt(question)
        term_response = self.provider.generate_content(term_prompt)
        search_term = self._clean_response(term_response).strip().strip('"').strip("'")

        self._log(f"🔍 Searching: {search_term}")

        # Try to find content
        content = None
        url = None
        attempted_terms = [search_term]

        for attempt in range(self.max_search_attempts):
            result = self.searcher.get_search_result(
                attempted_terms[-1],
                max_content_length=3000
            )

            if result.has_content():
                content = result.content
                url = result.url
                self._log(f"✅ Found content ({len(content)} chars)")
                break

            # Generate alternative term
            if attempt < self.max_search_attempts - 1:
                alt_prompt = self.prompts.generate_search_terms_prompt(
                    question, attempted_terms
                )
                alt_response = self.provider.generate_content(alt_prompt)
                new_terms = [
                    t.strip().strip('"').strip("'")
                    for t in self._clean_response(alt_response).split('\n')
                    if t.strip() and t.strip() not in attempted_terms
                ]
                if new_terms:
                    attempted_terms.append(new_terms[0])
                    self._log(f"🔍 Trying: {new_terms[0]}")

        if not content:
            self._log("❌ No Wikipedia content found")
            # Fall back to direct answer with disclaimer
            return self._execute_direct(question)

        if url:
            sources.append(url)

        # Extract answer from context
        answer_prompt = self.prompts.extract_answer_from_context_prompt(
            question, content
        )
        response = self.provider.generate_content(answer_prompt)
        answer = self._clean_response(response)

        return AgentResponse(
            answer=answer,
            sources=sources,
            category=QueryCategory.WIKIPEDIA
        )

    def _execute_compute(self, question: str) -> AgentResponse:
        """
        Category C: Pure computation.

        Tools used: Code generation + execution only.
        """
        self._log("🔢 Generating computation code")

        # Generate code
        code_prompt = self.prompts.generate_computation_code_prompt(question)
        code_response = self.provider.generate_content(
            code_prompt,
            system_prompt=self.prompts.CODE_GENERATOR_SYSTEM
        )

        code = self.code_executor.extract_code_from_response(code_response)

        if not code:
            self._log("⚠️ No code extracted, falling back to direct")
            return self._execute_direct(question)

        self._log(f"💻 Executing:\n{code}")

        # Execute code
        exec_result = self.code_executor.execute(code)

        if not exec_result.success:
            self._log(f"❌ Execution failed: {exec_result.error}")
            # Fall back to direct with error note
            direct_response = self._execute_direct(question)
            direct_response.answer += f"\n\n(Note: Automatic calculation failed: {exec_result.error})"
            return direct_response

        # Format result
        result_str = self._format_execution_result(exec_result)
        self._log(f"✅ Result: {result_str}")

        # Generate explanation
        explain_prompt = self.prompts.explain_computation_result_prompt(
            question, code, result_str
        )
        explanation = self._clean_response(
            self.provider.generate_content(explain_prompt)
        )

        return AgentResponse(
            answer=explanation,
            sources=[],
            category=QueryCategory.COMPUTE,
            code_executed=code,
            code_result=result_str
        )

    def _execute_combined(self, question: str) -> AgentResponse:
        """
        Category D: Wikipedia research + reasoning + computation.

        Tools used: Wikipedia search + code generation/execution.
        """
        self._log("🔬 Combined: Wikipedia + Computation")

        sources = []

        # Step 1: Identify concept to lookup
        concept_prompt = self.prompts.identify_concepts_prompt(question)
        concept_response = self.provider.generate_content(concept_prompt)
        concept = self._clean_response(concept_response).strip()

        self._log(f"🔍 Looking up concept: {concept}")

        # Step 2: Search Wikipedia for formula/concept
        result = self.searcher.get_search_result(concept, max_content_length=3000)

        if not result.has_content():
            # Try alternative search terms
            result = self.searcher.get_search_result(
                f"{concept} formula",
                max_content_length=3000
            )

        if not result.has_content():
            self._log("⚠️ No formula found, trying as pure computation")
            return self._execute_compute(question)

        formula_context = result.content
        if result.url:
            sources.append(result.url)

        self._log(f"📖 Found formula context ({len(formula_context)} chars)")

        # Step 3: Generate code with context
        code_prompt = self.prompts.generate_code_with_context_prompt(
            question, formula_context
        )
        code_response = self.provider.generate_content(
            code_prompt,
            system_prompt=self.prompts.CODE_GENERATOR_SYSTEM
        )

        code = self.code_executor.extract_code_from_response(code_response)

        if not code:
            # Fall back to text answer
            self._log("⚠️ No code generated, providing text answer")
            answer_prompt = self.prompts.extract_answer_from_context_prompt(
                question, formula_context
            )
            answer = self._clean_response(
                self.provider.generate_content(answer_prompt)
            )
            return AgentResponse(
                answer=answer,
                sources=sources,
                category=QueryCategory.COMBINED
            )

        self._log(f"💻 Executing:\n{code}")

        # Step 4: Execute code
        exec_result = self.code_executor.execute(code)

        if not exec_result.success:
            self._log(f"❌ Execution failed: {exec_result.error}")
            # Provide text answer with error note
            answer_prompt = self.prompts.extract_answer_from_context_prompt(
                question, formula_context
            )
            answer = self._clean_response(
                self.provider.generate_content(answer_prompt)
            )
            answer += f"\n\n(Note: Automatic calculation failed: {exec_result.error})"
            return AgentResponse(
                answer=answer,
                sources=sources,
                category=QueryCategory.COMBINED
            )

        result_str = self._format_execution_result(exec_result)
        self._log(f"✅ Result: {result_str}")

        # Step 5: Synthesize final answer
        synth_prompt = self.prompts.synthesize_combined_answer_prompt(
            question, formula_context, code, result_str
        )
        answer = self._clean_response(
            self.provider.generate_content(synth_prompt)
        )

        return AgentResponse(
            answer=answer,
            sources=sources,
            category=QueryCategory.COMBINED,
            code_executed=code,
            code_result=result_str
        )

    def _format_execution_result(self, exec_result: ExecutionResult) -> str:
        """Format execution result for display."""
        result_str = ""
        if exec_result.output:
            result_str = exec_result.output.strip()
        if exec_result.return_value is not None:
            if result_str:
                result_str += f"\nFinal value: {exec_result.return_value}"
            else:
                result_str = str(exec_result.return_value)
        return result_str or "No output"

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def query(self, question: str) -> AgentResponse:
        """
        Process a user query using the tool-minimalism architecture.

        Flow:
        1. Classify → Determine category
        2. Execute → Use minimal tools for that category
        3. Return → Synthesized response

        Args:
            question: The user's question

        Returns:
            AgentResponse with the answer and metadata
        """
        self._log(f"\n{'='*60}")
        self._log(f"🤔 Query: {question}")
        self._log('='*60)

        # Step 1: Classify
        category = self._classify_query(question)
        self._log(f"📌 Category: {category.value}")

        # Step 2: Execute based on category
        if category == QueryCategory.DIRECT:
            return self._execute_direct(question)
        elif category == QueryCategory.WIKIPEDIA:
            return self._execute_wikipedia(question)
        elif category == QueryCategory.COMPUTE:
            return self._execute_compute(question)
        elif category == QueryCategory.COMBINED:
            return self._execute_combined(question)
        else:
            # Fallback (should never happen)
            return self._execute_direct(question)

    def ask(self, question: str) -> str:
        """
        Simple interface to ask a question and get a string answer.

        Args:
            question: The user's question

        Returns:
            Formatted string answer
        """
        response = self.query(question)
        return response.format_output()

    # ==========================================================================
    # LEGACY COMPATIBILITY (Deprecated methods - will be removed)
    # ==========================================================================

    def process_calculation_query(self, question: str) -> AgentResponse:
        """
        DEPRECATED: Use query() instead.

        Kept for backwards compatibility.
        """
        self._log("⚠️ process_calculation_query is deprecated, use query()")
        return self._execute_combined(question)

    def process_info_query(self, question: str) -> AgentResponse:
        """
        DEPRECATED: Use query() instead.

        Kept for backwards compatibility.
        """
        self._log("⚠️ process_info_query is deprecated, use query()")
        return self._execute_wikipedia(question)
