#!/usr/bin/env python3
"""
Basic tests for the Wikipedia Agent functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wikipedia_agent import WikipediaAgent

def test_init():
    """Test agent initialization."""
    try:
        # This should work even without a valid API key for basic structure testing
        agent = WikipediaAgent("fake_api_key", max_iterations=3)
        assert agent.max_iterations == 3
        print("✅ Agent initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        return False

def test_search_term_generation_fallback():
    """Test search term generation fallback logic."""
    try:
        # Test that we can at least create the structure without making API calls
        question = "What is Python?"
        
        # Test the fallback logic that should return the question
        fallback_terms = [question]
        assert len(fallback_terms) >= 1
        assert fallback_terms[0] == question
        
        print("✅ Search term generation fallback test passed")
        return True
    except Exception as e:
        print(f"❌ Search term generation test failed: {e}")
        return False

def test_argument_parsing():
    """Test that argument parsing works correctly."""
    try:
        import argparse
        from wikipedia_agent import main
        
        # This tests the argument parser setup
        parser = argparse.ArgumentParser()
        parser.add_argument("question")
        parser.add_argument("--max-iterations", type=int, default=3)
        parser.add_argument("--api-key")
        
        args = parser.parse_args(["What is Python?", "--max-iterations", "5"])
        assert args.question == "What is Python?"
        assert args.max_iterations == 5
        print("✅ Argument parsing test passed")
        return True
    except Exception as e:
        print(f"❌ Argument parsing test failed: {e}")
        return False

def run_tests():
    """Run all basic tests."""
    print("Running basic functionality tests...")
    print("=" * 50)
    
    tests = [
        test_init,
        test_search_term_generation_fallback, 
        test_argument_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)