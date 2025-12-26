#!/usr/bin/env python3
"""
Prompt templates for the Wikipedia Agent v2.
Centralized location for all LLM prompts.

Architecture: Single classification → Minimal tool execution → Response synthesis

Design Principles:
- Single classification prompt determines the execution path
- No redundant prompts - each has a clear purpose
- Minimal tool usage - tools only when they add value
"""

from typing import List, Optional
from enum import Enum


class QueryCategory(str, Enum):
    """Query classification categories."""
    DIRECT = "DIRECT"           # Answer from internal knowledge
    WIKIPEDIA = "WIKIPEDIA"     # Factual lookup required
    COMPUTE = "COMPUTE"         # Pure calculation
    COMBINED = "COMBINED"       # Wikipedia + reasoning + code


class PromptTemplates:
    """
    Consolidated prompt templates for the Wikipedia Agent.

    Removed redundant prompts from previous version:
    - SEARCH_TERM_GENERATOR_SYSTEM (overly complex, replaced with simpler approach)
    - ANSWER_VALIDATOR_SYSTEM (unnecessary validation step)
    - needs_calculation_prompt (replaced by unified classification)
    - extract_calculation_details_prompt (merged into code generation)
    - validate_answer_prompt (removed - trust the answer)
    - summarize_search_prompt (unused)
    """

    # ==========================================================================
    # SYSTEM PROMPTS (Role Definitions)
    # ==========================================================================

    AGENT_SYSTEM = """You are a knowledgeable assistant that answers questions accurately and concisely.

You have access to:
1. Your internal knowledge for common facts and concepts
2. Wikipedia search for specific factual lookups when needed
3. Python code execution for mathematical calculations

Guidelines:
- Be direct and precise
- Use tools only when they add value
- Never speculate or hallucinate facts
- Cite sources when using external information"""

    CODE_GENERATOR_SYSTEM = """You are a Python code generator for scientific calculations.

Rules:
1. Use ONLY Python standard library (math functions available globally)
2. NO external imports (numpy, scipy are NOT available)
3. Assign the final answer to a variable called 'result'
4. Include brief comments explaining steps
5. Use print() for intermediate values if helpful

Available: sin, cos, tan, sqrt, log, exp, pi, e, radians, degrees, and all math functions.

Output ONLY the Python code in a code block. No explanations outside."""

    # Legacy aliases for backwards compatibility
    WIKIPEDIA_ASSISTANT_SYSTEM = AGENT_SYSTEM

    # ==========================================================================
    # QUERY CLASSIFICATION (Critical First Step)
    # ==========================================================================

    @staticmethod
    def classify_query_prompt(question: str) -> str:
        """
        Single classification prompt - the decision layer.

        This is the ONLY prompt that determines which tools (if any) to use.
        Replaces the old needs_calculation_prompt and implicit Wikipedia defaulting.
        """
        return f"""Classify this query into exactly ONE category:

Query: "{question}"

Categories:
A. DIRECT - Common knowledge, short factual answers, no computation or lookup needed
   Examples: "What is gravity?", "Who was Einstein?", "What does CPU stand for?"

B. WIKIPEDIA - Specific facts requiring lookup (dates, statistics, technical details)
   Examples: "When was the Eiffel Tower completed?", "What is the GDP of Japan?"

C. COMPUTE - Pure calculation, no external knowledge needed
   Examples: "What is 17% of 8450?", "Solve 3x + 7 = 25", "Calculate sin(45°)"

D. COMBINED - Needs both concept lookup AND calculation
   Examples: "Calculate option price using Black-Scholes", "Apply Bayes theorem to..."

Respond with ONLY the category letter (A, B, C, or D) and a brief reason.

Format: [CATEGORY]: [one-line reason]

Response:"""

    # ==========================================================================
    # DIRECT ANSWER (Category A - No Tools)
    # ==========================================================================

    @staticmethod
    def direct_answer_prompt(question: str) -> str:
        """Generate a direct answer without tools."""
        return f"""Answer this question directly and concisely using your knowledge.

Question: {question}

Provide a clear, accurate answer. Be informative but not verbose.

Answer:"""

    # ==========================================================================
    # WIKIPEDIA SEARCH (Category B)
    # ==========================================================================

    @staticmethod
    def generate_search_term_prompt(question: str) -> str:
        """
        Generate a SINGLE optimal search term for Wikipedia.

        Simplified from previous version - we now generate ONE term at a time.
        """
        return f"""Generate ONE Wikipedia search term for this question.

Question: {question}

Return ONLY the search term (no quotes, no explanation).
Use the exact phrase that would match a Wikipedia article title.

Search term:"""

    @staticmethod
    def generate_search_terms_prompt(
        question: str,
        previous_attempts: Optional[List[str]] = None
    ) -> str:
        """Generate multiple search terms (for fallback scenarios)."""
        previous_info = ""
        if previous_attempts:
            previous_info = f"\nPrevious terms that didn't work: {', '.join(previous_attempts)}"

        return f"""Generate 2-3 Wikipedia search terms for this question.{previous_info}

Question: {question}

Return ONLY the search terms, one per line, no explanations.

Search terms:"""

    @staticmethod
    def extract_answer_from_context_prompt(question: str, context: str) -> str:
        """Extract answer from Wikipedia context."""
        return f"""Answer this question using ONLY the provided Wikipedia content.

Wikipedia Content:
{context}

Question: {question}

If the content answers the question, provide a clear response.
If the content is insufficient, state what specific information is missing.

Answer:"""

    # Legacy alias
    answer_question_prompt = extract_answer_from_context_prompt

    # ==========================================================================
    # PURE COMPUTATION (Category C)
    # ==========================================================================

    @staticmethod
    def generate_computation_code_prompt(question: str) -> str:
        """Generate code for pure computation (no Wikipedia context needed)."""
        return f"""Generate Python code to solve this calculation.

Question: {question}

Requirements:
1. No imports needed - math functions (sin, cos, sqrt, pi, radians, etc.) are available
2. Assign final answer to 'result'
3. Use print() to show the calculation steps
4. Handle edge cases

```python
# Your code here
```"""

    @staticmethod
    def explain_computation_result_prompt(
        question: str,
        code: str,
        result: str
    ) -> str:
        """Explain a pure computation result."""
        return f"""Provide a brief answer for this calculation.

Question: {question}

Code executed:
```python
{code}
```

Result: {result}

Provide a concise answer stating the result with any relevant units or context.

Answer:"""

    # ==========================================================================
    # COMBINED PROCESSING (Category D)
    # ==========================================================================

    @staticmethod
    def identify_concepts_prompt(question: str) -> str:
        """Identify what concepts need to be looked up for a combined query."""
        return f"""Identify the key concept or formula needed to answer this question.

Question: {question}

What specific concept, formula, or definition must be looked up?
Respond with ONE term suitable for Wikipedia search.

Concept to lookup:"""

    @staticmethod
    def generate_code_with_context_prompt(
        question: str,
        formula_context: str
    ) -> str:
        """Generate code using Wikipedia-sourced formulas/concepts."""
        return f"""Generate Python code to solve this using the provided formula/concept.

Question: {question}

Formula/Concept from Wikipedia:
{formula_context}

Requirements:
1. Apply the formula correctly with given values
2. No imports - math functions available globally
3. Assign final answer to 'result'
4. Show calculation steps with print()

```python
# Your code here
```"""

    # Legacy alias for backwards compatibility
    @staticmethod
    def generate_code_prompt(
        question: str,
        formula_context: str,
        calculation_details: str = ""
    ) -> str:
        """Legacy method - redirects to generate_code_with_context_prompt."""
        return PromptTemplates.generate_code_with_context_prompt(question, formula_context)

    @staticmethod
    def synthesize_combined_answer_prompt(
        question: str,
        concept_context: str,
        code: str,
        result: str
    ) -> str:
        """Synthesize final answer for combined queries."""
        return f"""Provide a complete answer combining the concept and calculation.

Question: {question}

Concept/Formula (from Wikipedia):
{concept_context}

Calculation Performed:
```python
{code}
```

Result: {result}

Provide a clear explanation that:
1. Briefly explains the concept/formula used
2. Shows how it was applied
3. States the final answer with appropriate units

Answer:"""

    # Legacy alias
    @staticmethod
    def explain_result_prompt(
        question: str,
        code: str,
        result: str,
        formula_context: str
    ) -> str:
        """Legacy method - redirects to synthesize_combined_answer_prompt."""
        return PromptTemplates.synthesize_combined_answer_prompt(
            question, formula_context, code, result
        )

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    @staticmethod
    def parse_classification(response: str) -> QueryCategory:
        """
        Parse the classification response into a QueryCategory.

        Args:
            response: Raw LLM response from classify_query_prompt

        Returns:
            QueryCategory enum value
        """
        response_upper = response.upper().strip()

        # Check for explicit category markers
        # Note: COMBINED must be checked before COMPUTE (substring match issue)
        if response_upper.startswith('A') or 'DIRECT' in response_upper:
            return QueryCategory.DIRECT
        elif response_upper.startswith('B') or 'WIKIPEDIA' in response_upper:
            return QueryCategory.WIKIPEDIA
        elif response_upper.startswith('D') or 'COMBINED' in response_upper:
            return QueryCategory.COMBINED
        elif response_upper.startswith('C') or 'COMPUTE' in response_upper:
            return QueryCategory.COMPUTE

        # Heuristic fallback based on keywords in response
        if any(kw in response_upper for kw in ['CALCULATION', 'MATH', 'FORMULA']):
            if 'LOOKUP' in response_upper or 'SEARCH' in response_upper:
                return QueryCategory.COMBINED
            return QueryCategory.COMPUTE

        if any(kw in response_upper for kw in ['LOOKUP', 'SEARCH', 'SPECIFIC', 'DATE', 'STATISTIC']):
            return QueryCategory.WIKIPEDIA

        # Default to DIRECT if unclear (tool minimalism principle)
        return QueryCategory.DIRECT
