#!/usr/bin/env python3
"""
Integration tests for the Wikipedia Agent that validate all documented functionality.
These tests require a real GEMINI_API_KEY to work properly.
"""

import os
import sys
import subprocess
import time
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wikipedia_agent import WikipediaAgent


class IntegrationTestRunner:
    """Integration test runner for Wikipedia Agent."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for integration tests")
        
        self.passed = 0
        self.total = 0
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and track results."""
        print(f"\n🧪 Running: {test_name}")
        print("-" * 60)
        
        self.total += 1
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.passed += 1
                return True
            else:
                print(f"❌ FAILED: {test_name}")
                return False
        except Exception as e:
            print(f"❌ FAILED: {test_name} - Exception: {e}")
            return False
    
    def test_agent_initialization_with_real_api(self) -> bool:
        """Test agent initialization with real API key."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=3)
            assert agent.max_iterations == 3
            assert agent.model is not None
            print("Agent initialized successfully with real API key")
            return True
        except Exception as e:
            print(f"Failed to initialize agent: {e}")
            return False
    
    def test_wikipedia_search_functionality(self) -> bool:
        """Test Wikipedia search functionality."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=1)
            
            # Test searching for a well-known topic
            content = agent.search_wikipedia("Python programming language")
            if not content:
                print("Failed to find content for Python programming language")
                return False
            
            if len(content) < 100:
                print(f"Content too short: {len(content)} characters")
                return False
            
            print(f"Successfully retrieved Wikipedia content ({len(content)} characters)")
            print(f"Content preview: {content[:200]}...")
            return True
        except Exception as e:
            print(f"Wikipedia search failed: {e}")
            return False
    
    def test_search_term_generation(self) -> bool:
        """Test search term generation using Gemini AI."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=1)
            
            question = "Who was the first person to walk on the moon?"
            search_terms = agent.generate_search_terms(question)
            
            if not search_terms or len(search_terms) == 0:
                print("No search terms generated")
                return False
            
            print(f"Generated search terms: {search_terms}")
            
            # Check if terms are reasonable (should contain relevant keywords)
            relevant_keywords = ["neil", "armstrong", "apollo", "moon", "astronaut", "nasa"]
            terms_text = " ".join(search_terms).lower()
            
            if not any(keyword in terms_text for keyword in relevant_keywords):
                print(f"Generated terms don't seem relevant: {search_terms}")
                return False
            
            return True
        except Exception as e:
            print(f"Search term generation failed: {e}")
            return False
    
    def test_answer_generation(self) -> bool:
        """Test answer generation using Gemini AI."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=1)
            
            question = "What is the capital of France?"
            # Use a simple Wikipedia context
            context = "France is a country in Western Europe. Paris is the capital and most populous city of France."
            
            answer = agent.answer_question(question, context)
            
            if not answer or len(answer) < 10:
                print(f"Generated answer too short: '{answer}'")
                return False
            
            # Check if answer mentions Paris
            if "paris" not in answer.lower():
                print(f"Answer doesn't mention Paris: '{answer}'")
                return False
            
            print(f"Generated answer: {answer}")
            return True
        except Exception as e:
            print(f"Answer generation failed: {e}")
            return False
    
    def test_complete_workflow_simple_question(self) -> bool:
        """Test complete workflow with a simple question."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=3)
            
            question = "What is the capital of Japan?"
            answer, search_terms = agent.process_query(question)
            
            if not answer or len(answer) < 20:
                print(f"Answer too short: '{answer}'")
                return False
            
            if not search_terms or len(search_terms) == 0:
                print("No search terms used")
                return False
            
            # Check if answer mentions Tokyo
            if "tokyo" not in answer.lower():
                print(f"Answer doesn't mention Tokyo: '{answer}'")
                return False
            
            print(f"Answer: {answer}")
            print(f"Search terms used: {search_terms}")
            return True
        except Exception as e:
            print(f"Complete workflow test failed: {e}")
            return False
    
    def test_complete_workflow_complex_question(self) -> bool:
        """Test complete workflow with a more complex question."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=3)
            
            question = "Who invented the telephone?"
            answer, search_terms = agent.process_query(question)
            
            if not answer or len(answer) < 30:
                print(f"Answer too short: '{answer}'")
                return False
            
            if not search_terms or len(search_terms) == 0:
                print("No search terms used")
                return False
            
            # Check if answer mentions Alexander Graham Bell
            answer_lower = answer.lower()
            if "bell" not in answer_lower and "alexander" not in answer_lower:
                print(f"Answer doesn't mention Bell or Alexander: '{answer}'")
                return False
            
            print(f"Answer: {answer}")
            print(f"Search terms used: {search_terms}")
            return True
        except Exception as e:
            print(f"Complex workflow test failed: {e}")
            return False
    
    def test_max_iterations_parameter(self) -> bool:
        """Test max iterations parameter functionality."""
        try:
            # Test with 1 iteration
            agent = WikipediaAgent(self.api_key, max_iterations=1)
            question = "What is quantum computing?"
            
            answer, search_terms = agent.process_query(question)
            
            if not answer:
                print("No answer generated with max_iterations=1")
                return False
            
            print(f"With max_iterations=1, got answer: {answer[:100]}...")
            return True
        except Exception as e:
            print(f"Max iterations test failed: {e}")
            return False
    
    def test_cli_basic_functionality(self) -> bool:
        """Test CLI functionality with subprocess calls."""
        try:
            # Test basic CLI call
            env = os.environ.copy()
            env["GEMINI_API_KEY"] = self.api_key
            
            result = subprocess.run([
                "python", "wikipedia_agent.py", 
                "What is the capital of Italy?"
            ], 
            capture_output=True, 
            text=True, 
            timeout=60,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                print(f"CLI call failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                return False
            
            output = result.stdout
            if "ANSWER:" not in output:
                print(f"Output doesn't contain expected format: {output}")
                return False
            
            if "rome" not in output.lower():
                print(f"Answer doesn't mention Rome: {output}")
                return False
            
            print("CLI basic functionality test passed")
            print(f"Output sample: {output[:200]}...")
            return True
        except subprocess.TimeoutExpired:
            print("CLI call timed out")
            return False
        except Exception as e:
            print(f"CLI test failed: {e}")
            return False
    
    def test_cli_with_api_key_parameter(self) -> bool:
        """Test CLI with --api-key parameter."""
        try:
            result = subprocess.run([
                "python", "wikipedia_agent.py", 
                "--api-key", self.api_key,
                "What is HTML?"
            ], 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                print(f"CLI with --api-key failed: {result.stderr}")
                return False
            
            output = result.stdout
            if "ANSWER:" not in output:
                print(f"Output doesn't contain expected format: {output}")
                return False
            
            print("CLI with --api-key parameter test passed")
            return True
        except subprocess.TimeoutExpired:
            print("CLI call with --api-key timed out")
            return False
        except Exception as e:
            print(f"CLI --api-key test failed: {e}")
            return False
    
    def test_cli_max_iterations_parameter(self) -> bool:
        """Test CLI with --max-iterations parameter."""
        try:
            env = os.environ.copy()
            env["GEMINI_API_KEY"] = self.api_key
            
            result = subprocess.run([
                "python", "wikipedia_agent.py", 
                "--max-iterations", "2",
                "What is CSS?"
            ], 
            capture_output=True, 
            text=True, 
            timeout=60,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                print(f"CLI with --max-iterations failed: {result.stderr}")
                return False
            
            output = result.stdout
            if "ANSWER:" not in output:
                print(f"Output doesn't contain expected format: {output}")
                return False
            
            print("CLI with --max-iterations parameter test passed")
            return True
        except subprocess.TimeoutExpired:
            print("CLI call with --max-iterations timed out")
            return False
        except Exception as e:
            print(f"CLI --max-iterations test failed: {e}")
            return False
    
    def test_error_handling_invalid_api_key(self) -> bool:
        """Test error handling with invalid API key."""
        try:
            agent = WikipediaAgent("invalid_api_key", max_iterations=1)
            question = "What is Python?"
            
            answer, search_terms = agent.process_query(question)
            
            # Should either return an error message or fallback gracefully
            if not answer:
                print("No answer returned for invalid API key")
                return False
            
            print(f"With invalid API key, got: {answer[:100]}...")
            return True
        except Exception as e:
            print(f"Error handling test completed with expected exception: {e}")
            return True  # Expected to fail
    
    def test_readme_examples(self) -> bool:
        """Test examples from README documentation."""
        try:
            agent = WikipediaAgent(self.api_key, max_iterations=3)
            
            # Test README examples
            examples = [
                "Who invented the telephone?",
                "What is the capital of Japan?",
                "When was the Internet created?"
            ]
            
            for question in examples:
                print(f"  Testing README example: {question}")
                answer, search_terms = agent.process_query(question)
                
                if not answer or len(answer) < 20:
                    print(f"    Failed for question: {question}")
                    return False
                
                if not search_terms:
                    print(f"    No search terms for: {question}")
                    return False
                
                print(f"    ✅ Success: {len(answer)} chars, {len(search_terms)} terms")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
            
            return True
        except Exception as e:
            print(f"README examples test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🚀 Starting Wikipedia Agent Integration Tests")
        print("=" * 80)
        
        tests = [
            ("Agent Initialization", self.test_agent_initialization_with_real_api),
            ("Wikipedia Search", self.test_wikipedia_search_functionality),
            ("Search Term Generation", self.test_search_term_generation),
            ("Answer Generation", self.test_answer_generation),
            ("Complete Workflow - Simple", self.test_complete_workflow_simple_question),
            ("Complete Workflow - Complex", self.test_complete_workflow_complex_question),
            ("Max Iterations Parameter", self.test_max_iterations_parameter),
            ("CLI Basic Functionality", self.test_cli_basic_functionality),
            ("CLI API Key Parameter", self.test_cli_with_api_key_parameter),
            ("CLI Max Iterations Parameter", self.test_cli_max_iterations_parameter),
            ("Error Handling", self.test_error_handling_invalid_api_key),
            ("README Examples", self.test_readme_examples)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(0.5)  # Small delay between tests
        
        print("\n" + "=" * 80)
        print(f"📊 INTEGRATION TEST RESULTS: {self.passed}/{self.total} PASSED")
        
        if self.passed == self.total:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            print("❌ SOME INTEGRATION TESTS FAILED")
            return False


def main():
    """Main entry point for integration tests."""
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is required for integration tests")
        print("   These tests validate the actual functionality with real API calls")
        sys.exit(1)
    
    runner = IntegrationTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()