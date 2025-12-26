#!/usr/bin/env python3
"""
Safe code execution module for scientific calculations.
Executes Python code in a restricted environment without external library imports.
"""

import ast
import math
import cmath
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from io import StringIO
import sys


@dataclass
class ExecutionResult:
    """Container for code execution results."""
    success: bool
    output: str
    error: Optional[str] = None
    return_value: Any = None
    variables: Dict[str, Any] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}


class SafeCodeValidator(ast.NodeVisitor):
    """
    AST-based validator to ensure code safety.
    Prevents imports, file operations, and dangerous operations.
    """

    FORBIDDEN_BUILTINS = {
        'exec', 'eval', 'compile', '__import__', 'open', 'input',
        'breakpoint', 'help', 'license', 'credits', 'quit', 'exit',
        'vars', 'dir', 'globals', 'locals', 'memoryview', 'bytearray',
    }

    FORBIDDEN_NAMES = {
        '__builtins__', '__import__', '__file__', '__doc__',
        'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'socket',
        'requests', 'urllib', 'http', 'ftplib', 'smtplib',
    }

    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block all import statements."""
        module_names = [alias.name for alias in node.names]
        self.errors.append(f"Import not allowed: {', '.join(module_names)}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block all from ... import statements."""
        self.errors.append(f"Import from '{node.module}' not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for forbidden functions."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_BUILTINS:
                self.errors.append(f"Function '{node.func.id}' not allowed")
        elif isinstance(node.func, ast.Attribute):
            # Check for file operations
            if node.func.attr in {'read', 'write', 'open', 'system', 'popen'}:
                self.errors.append(f"Method '{node.func.attr}' not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for forbidden names."""
        if node.id in self.FORBIDDEN_NAMES:
            self.errors.append(f"Name '{node.id}' not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for dangerous attribute access."""
        if node.attr.startswith('_'):
            self.errors.append(f"Private attribute '{node.attr}' access not allowed")
        self.generic_visit(node)

    def validate(self, code: str) -> tuple:
        """
        Validate code for safety.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return len(self.errors) == 0, self.errors
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]


class CodeExecutor:
    """
    Safe code executor for scientific calculations.
    Provides a sandboxed environment with math functions available.
    """

    # Safe built-in functions for calculations
    SAFE_BUILTINS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'reversed': reversed,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        'frozenset': frozenset,
        'str': str,
        'int': int,
        'float': float,
        'complex': complex,
        'bool': bool,
        'pow': pow,
        'divmod': divmod,
        'bin': bin,
        'hex': hex,
        'oct': oct,
        'chr': chr,
        'ord': ord,
        'all': all,
        'any': any,
        'isinstance': isinstance,
        'type': type,
        'print': print,
        'True': True,
        'False': False,
        'None': None,
    }

    def __init__(self, timeout: float = 10.0):
        """
        Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.validator = SafeCodeValidator()

    def _get_math_namespace(self) -> Dict[str, Any]:
        """
        Get a namespace with math functions available.

        Returns:
            Dictionary of safe math functions and constants
        """
        namespace = {}

        # Add safe builtins
        namespace.update(self.SAFE_BUILTINS)

        # Add math module functions and constants
        math_functions = {
            # Trigonometric functions
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'asinh': math.asinh,
            'acosh': math.acosh,
            'atanh': math.atanh,

            # Angular conversion
            'degrees': math.degrees,
            'radians': math.radians,

            # Power and logarithmic functions
            'exp': math.exp,
            'expm1': math.expm1,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'log1p': math.log1p,
            'sqrt': math.sqrt,
            'isqrt': math.isqrt,

            # Special functions
            'factorial': math.factorial,
            'gamma': math.gamma,
            'lgamma': math.lgamma,
            'erf': math.erf,
            'erfc': math.erfc,

            # Number-theoretic functions
            'gcd': math.gcd,
            'lcm': getattr(math, 'lcm', lambda a, b: abs(a * b) // math.gcd(a, b)),
            'comb': getattr(math, 'comb', None),
            'perm': getattr(math, 'perm', None),

            # Floating point functions
            'ceil': math.ceil,
            'floor': math.floor,
            'trunc': math.trunc,
            'fabs': math.fabs,
            'copysign': math.copysign,
            'fmod': math.fmod,
            'modf': math.modf,
            'frexp': math.frexp,
            'ldexp': math.ldexp,
            'fsum': math.fsum,
            'prod': getattr(math, 'prod', lambda x: __import__('functools').reduce(lambda a, b: a * b, x, 1)),

            # Classification functions
            'isfinite': math.isfinite,
            'isinf': math.isinf,
            'isnan': math.isnan,

            # Constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,
        }

        # Remove None values (for functions not available in older Python)
        math_functions = {k: v for k, v in math_functions.items() if v is not None}
        namespace.update(math_functions)

        # Add complex math functions
        complex_functions = {
            'csin': cmath.sin,
            'ccos': cmath.cos,
            'ctan': cmath.tan,
            'cexp': cmath.exp,
            'clog': cmath.log,
            'csqrt': cmath.sqrt,
            'phase': cmath.phase,
            'polar': cmath.polar,
            'rect': cmath.rect,
        }
        namespace.update(complex_functions)

        # Add helper functions for common operations
        def mean(numbers):
            """Calculate the arithmetic mean."""
            nums = list(numbers)
            return sum(nums) / len(nums)

        def median(numbers):
            """Calculate the median."""
            nums = sorted(numbers)
            n = len(nums)
            mid = n // 2
            if n % 2 == 0:
                return (nums[mid - 1] + nums[mid]) / 2
            return nums[mid]

        def variance(numbers, sample=True):
            """Calculate variance."""
            nums = list(numbers)
            n = len(nums)
            m = sum(nums) / n
            ss = sum((x - m) ** 2 for x in nums)
            return ss / (n - 1) if sample else ss / n

        def stdev(numbers, sample=True):
            """Calculate standard deviation."""
            return math.sqrt(variance(numbers, sample))

        def hypot(*args):
            """Calculate Euclidean distance."""
            return math.sqrt(sum(x ** 2 for x in args))

        namespace.update({
            'mean': mean,
            'median': median,
            'variance': variance,
            'stdev': stdev,
            'hypot': hypot,
        })

        return namespace

    def extract_code_from_response(self, response: str) -> Optional[str]:
        """
        Extract Python code from an LLM response.

        Args:
            response: LLM response potentially containing code blocks

        Returns:
            Extracted code or None
        """
        # First, clean thinking tokens
        cleaned_response = response
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<think>.*', '', cleaned_response, flags=re.DOTALL)

        # Try to find code blocks with ```python or ``` markers
        patterns = [
            r'```python\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
            r'```(.*?)```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                # Validate it looks like Python code
                if self._is_valid_python(code):
                    return code

        # If no code blocks, try to extract Python-like lines
        lines = cleaned_response.strip().split('\n')
        code_lines = []
        in_code_section = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and obvious non-code
            if not stripped:
                if in_code_section:
                    code_lines.append('')
                continue

            # Skip lines with non-ASCII characters (often LLM explanations)
            if any(ord(c) > 127 for c in stripped):
                continue

            # Skip lines that look like natural language explanations
            if any(phrase in stripped.lower() for phrase in [
                'let me', 'which is', 'the result', 'so the', 'this gives',
                'approximately', 'divided by', 'then ', 'first,', 'compute that'
            ]):
                continue

            # Check if line looks like Python code
            is_code = (
                stripped.startswith('#') or
                stripped.startswith('def ') or
                stripped.startswith('for ') or
                stripped.startswith('if ') or
                stripped.startswith('while ') or
                stripped.startswith('return ') or
                stripped.startswith('print(') or
                stripped.startswith('result') or
                ('=' in stripped and not stripped.endswith('=')) or
                stripped.startswith('import ') or
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', stripped) or
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(', stripped)
            )

            if is_code:
                in_code_section = True
                code_lines.append(line)
            elif in_code_section and stripped.startswith((' ', '\t')):
                # Continuation of indented block
                code_lines.append(line)

        if code_lines:
            code = '\n'.join(code_lines).strip()
            if self._is_valid_python(code):
                return code

        return None

    def _is_valid_python(self, code: str) -> bool:
        """
        Check if code is syntactically valid Python.

        Args:
            code: Code string to check

        Returns:
            True if valid Python syntax
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code to fix common LLM output issues.

        Args:
            code: Raw code from LLM

        Returns:
            Preprocessed code
        """
        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip import statements (math functions are already available)
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue

            # Remove math. prefix since functions are already in namespace
            line = re.sub(r'\bmath\.', '', line)

            # Remove numpy. prefix
            line = re.sub(r'\bnp\.', '', line)
            line = re.sub(r'\bnumpy\.', '', line)

            # Remove cmath. prefix (we have csin, ccos, etc.)
            line = re.sub(r'\bcmath\.', 'c', line)

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code safely and return results.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output, errors, and variables
        """
        # Preprocess code to fix common issues
        code = self._preprocess_code(code)

        # Validate code
        self.validator = SafeCodeValidator()  # Reset validator
        is_valid, errors = self.validator.validate(code)

        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {'; '.join(errors)}"
            )

        # Set up the execution environment
        namespace = self._get_math_namespace()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        result_value = None

        try:
            # Parse the code
            tree = ast.parse(code)

            # Execute all but the last statement
            if tree.body:
                # Separate expressions that could be results
                last_node = tree.body[-1]

                if len(tree.body) > 1:
                    module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(module, '<string>', 'exec'), namespace)

                # Handle the last statement
                if isinstance(last_node, ast.Expr):
                    # It's an expression, evaluate and capture result
                    expr = ast.Expression(body=last_node.value)
                    result_value = eval(compile(expr, '<string>', 'eval'), namespace)
                else:
                    # It's a statement, execute it
                    module = ast.Module(body=[last_node], type_ignores=[])
                    exec(compile(module, '<string>', 'exec'), namespace)

            output = captured_output.getvalue()

            # Extract user-defined variables
            user_vars = {
                k: v for k, v in namespace.items()
                if not k.startswith('_') and k not in self.SAFE_BUILTINS
                and k not in self._get_math_namespace()
            }

            # If there's a 'result' variable, use it as the return value
            if 'result' in namespace and result_value is None:
                result_value = namespace['result']

            return ExecutionResult(
                success=True,
                output=output,
                return_value=result_value,
                variables=user_vars
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=str(e)
            )
        finally:
            sys.stdout = old_stdout

    def execute_with_context(
        self,
        code: str,
        context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute code with additional context variables.

        Args:
            code: Python code to execute
            context: Additional variables to include in the namespace

        Returns:
            ExecutionResult with output and results
        """
        # Validate code first
        self.validator = SafeCodeValidator()
        is_valid, errors = self.validator.validate(code)

        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {'; '.join(errors)}"
            )

        namespace = self._get_math_namespace()
        if context:
            namespace.update(context)

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        result_value = None

        try:
            tree = ast.parse(code)

            if tree.body:
                last_node = tree.body[-1]

                if len(tree.body) > 1:
                    module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(module, '<string>', 'exec'), namespace)

                if isinstance(last_node, ast.Expr):
                    expr = ast.Expression(body=last_node.value)
                    result_value = eval(compile(expr, '<string>', 'eval'), namespace)
                else:
                    module = ast.Module(body=[last_node], type_ignores=[])
                    exec(compile(module, '<string>', 'exec'), namespace)

            output = captured_output.getvalue()

            if 'result' in namespace and result_value is None:
                result_value = namespace['result']

            return ExecutionResult(
                success=True,
                output=output,
                return_value=result_value,
                variables=namespace
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output=captured_output.getvalue(),
                error=str(e)
            )
        finally:
            sys.stdout = old_stdout
