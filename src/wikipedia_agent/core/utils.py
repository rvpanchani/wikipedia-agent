#!/usr/bin/env python3
"""
Utility functions shared between V1 and V2 Wikipedia agents.

This module provides common utilities for:
- Cleaning LLM responses
- Extracting and formatting numbers
- Detecting mathematical expressions
- Code block extraction
"""

import re
from typing import List, Optional


def clean_llm_response(response: str) -> str:
    """
    Clean an LLM response by removing thinking tokens and artifacts.

    Handles common artifacts from various LLM providers including:
    - <think>...</think> blocks (Qwen thinking mode)
    - <|endoftext|> markers
    - "Human:" continuation markers

    Args:
        response: Raw response from LLM

    Returns:
        Cleaned response text
    """
    # Remove thinking tokens
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)

    # Remove end of text markers
    cleaned = re.sub(r'<\|endoftext\|>.*', '', cleaned, flags=re.DOTALL)

    # Split on "Human:" and keep only first part
    if "Human:" in cleaned:
        cleaned = cleaned.split("Human:")[0]

    # Clean multiple newlines
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

    return cleaned.strip()


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a text string.

    Args:
        text: Input text

    Returns:
        List of extracted numbers
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def format_number(value: float, precision: int = 6) -> str:
    """
    Format a number for display.

    Uses scientific notation for very large or small numbers,
    otherwise uses standard decimal notation with trailing zeros removed.

    Args:
        value: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if abs(value) < 1e-10:
        return "0"
    elif abs(value) >= 1e6 or abs(value) < 1e-4:
        return f"{value:.{precision}e}"
    else:
        formatted = f"{value:.{precision}f}"
        # Remove trailing zeros
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted


def detect_angle_unit(text: str) -> Optional[str]:
    """
    Detect whether angles in text are in degrees or radians.

    Args:
        text: Input text to analyze

    Returns:
        'degrees', 'radians', or None if not detected
    """
    text_lower = text.lower()

    if any(word in text_lower for word in ['degree', 'degrees', '°']):
        return 'degrees'
    elif any(word in text_lower for word in ['radian', 'radians', 'rad']):
        return 'radians'

    return None


def parse_math_expression(expr: str) -> str:
    """
    Parse a mathematical expression and convert to Python syntax.

    Handles common mathematical notation like:
    - ^ → **  (exponentiation)
    - × → *   (multiplication)
    - √ → sqrt (square root)
    - π → pi  (pi constant)

    Args:
        expr: Mathematical expression

    Returns:
        Python-compatible expression
    """
    replacements = [
        (r'\^', '**'),           # Exponentiation
        (r'×', '*'),             # Multiplication
        (r'÷', '/'),             # Division
        (r'√', 'sqrt'),          # Square root
        (r'π', 'pi'),            # Pi
        (r'sin⁻¹', 'asin'),      # Inverse sine
        (r'cos⁻¹', 'acos'),      # Inverse cosine
        (r'tan⁻¹', 'atan'),      # Inverse tangent
        (r'ln', 'log'),          # Natural log
        (r'log₁₀', 'log10'),     # Base 10 log
        (r'log₂', 'log2'),       # Base 2 log
    ]

    result = expr
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    return result


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract a code block from text.

    Looks for code blocks in the following order:
    1. ```python ... ```
    2. ``` ... ```

    Args:
        text: Text potentially containing code blocks

    Returns:
        Extracted code or None
    """
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'```(.*?)```',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

    return None


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def is_scientific_question(question: str) -> bool:
    """
    Check if a question is likely scientific/mathematical.

    Uses keyword detection to identify questions that may require
    mathematical computation or scientific knowledge.

    Args:
        question: The question to check

    Returns:
        True if likely scientific
    """
    scientific_keywords = [
        'calculate', 'compute', 'solve', 'find', 'derive',
        'formula', 'equation', 'theorem', 'law',
        'sin', 'cos', 'tan', 'log', 'exp', 'sqrt',
        'velocity', 'acceleration', 'force', 'energy',
        'pressure', 'temperature', 'mass', 'volume',
        'area', 'circumference', 'radius', 'diameter',
        'integral', 'derivative', 'limit',
        'probability', 'statistics', 'mean', 'variance',
    ]

    question_lower = question.lower()
    return any(kw in question_lower for kw in scientific_keywords)


def extract_final_answer(structured_response: str) -> str:
    """
    Extract the final answer from structured response patterns.

    Handles responses with ANSWER_FOUND: or INSUFFICIENT_INFO: prefixes.

    Args:
        structured_response: Response with structured patterns

    Returns:
        Clean answer text for display to user
    """
    cleaned_response = clean_llm_response(structured_response)

    if cleaned_response.startswith("ANSWER_FOUND:"):
        return cleaned_response[len("ANSWER_FOUND:"):].strip()
    elif cleaned_response.startswith("INSUFFICIENT_INFO:"):
        reason = cleaned_response[len("INSUFFICIENT_INFO:"):].strip()
        return f"I couldn't find sufficient information to answer your question. {reason}"
    else:
        # Fallback for non-structured responses
        return cleaned_response
