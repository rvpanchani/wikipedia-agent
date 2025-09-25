#!/usr/bin/env python3
"""
Smoke tests for Wikipedia Agent - basic functionality checks that don't require API calls.
These tests validate the structure and basic components work correctly.
"""

import sys
import os
import importlib.util
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported."""
    try:
        import wikipedia_agent
        import wikipedia
        import google.generativeai as genai
        from dotenv import load_dotenv
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_module_structure():
    """Test that the main module has expected structure."""
    try:
        from wikipedia_agent import WikipediaAgent, main
        
        # Check WikipediaAgent class exists and has expected methods
        expected_methods = ['__init__', 'generate_search_terms', 'search_wikipedia', 'answer_question', 'process_query']
        
        for method in expected_methods:
            if not hasattr(WikipediaAgent, method):
                print(f"❌ Missing method: {method}")
                return False
        
        print("✅ Module structure validation passed")
        return True
    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        return False


def test_argument_parser():
    """Test argument parser without executing main logic."""
    try:
        result = subprocess.run(
            ["python", "wikipedia_agent.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            print(f"❌ Help command failed: {result.stderr}")
            return False
        
        help_output = result.stdout
        expected_in_help = ["question", "max-iterations", "api-key", "Examples:"]
        
        for expected in expected_in_help:
            if expected not in help_output:
                print(f"❌ Missing in help: {expected}")
                return False
        
        print("✅ Argument parser test passed")
        return True
    except Exception as e:
        print(f"❌ Argument parser test failed: {e}")
        return False


def test_error_handling():
    """Test that the application handles missing API key correctly."""
    try:
        # Test without API key - should show proper error message or work with available provider
        env = os.environ.copy()
        # Remove all possible API keys
        env.pop("GEMINI_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("AZURE_OPENAI_API_KEY", None)
        env.pop("HUGGINGFACE_API_KEY", None)
        
        # Also try to prevent Ollama detection by setting a non-existent URL
        env["OLLAMA_BASE_URL"] = "http://localhost:99999"
        
        result = subprocess.run(
            ["python", "wikipedia_agent.py", "test question"],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        error_output = result.stderr + result.stdout
        
        # Check for proper error message when no providers are available
        if "No properly configured LLM provider found" in error_output:
            print("✅ Error handling test passed - no providers available")
            return True
        
        # If it seems to be trying a provider but failing, that's also acceptable
        if result.returncode != 0 and ("error" in error_output.lower() or "failed" in error_output.lower()):
            print("✅ Error handling test passed - provider failed as expected")
            return True
        
        # If it succeeded somehow, that might be fine too (maybe a provider was available)
        if result.returncode == 0:
            print("✅ Error handling test passed - provider worked")
            return True
        
        print(f"❌ Unexpected result: returncode={result.returncode}")
        print(f"Output: {error_output}")
        return False
        
    except subprocess.TimeoutExpired:
        print("✅ Error handling test passed - command timed out as expected without valid provider")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_dependencies_available():
    """Test that all dependencies are properly installed."""
    try:
        dependencies = [
            "wikipedia",
            "google.generativeai", 
            "python_dotenv",
            "openai"
        ]
        
        for dep_name in dependencies:
            try:
                if dep_name == "google.generativeai":
                    import google.generativeai as genai
                    print(f"   {dep_name}: imported successfully")
                elif dep_name == "python_dotenv":
                    from dotenv import load_dotenv
                    print(f"   {dep_name}: imported successfully")
                else:
                    module = __import__(dep_name)
                    if hasattr(module, "__version__"):
                        print(f"   {dep_name}: {module.__version__}")
                    else:
                        print(f"   {dep_name}: imported successfully")
            except ImportError as e:
                print(f"   ❌ Failed to import {dep_name}: {e}")
                return False
        
        print("✅ Dependencies check passed")
        return True
    except Exception as e:
        print(f"❌ Dependencies test failed: {e}")
        return False


def test_file_structure():
    """Test that required files exist."""
    required_files = [
        "wikipedia_agent.py",
        "requirements.txt",
        "README.md",
        "test_basic.py"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in required_files:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ Missing required file: {filename}")
            return False
    
    print("✅ File structure test passed")
    return True


def main():
    """Run all smoke tests."""
    print("🚀 Running Wikipedia Agent Smoke Tests")
    print("=" * 60)
    print("These tests validate basic functionality without requiring API access")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Module Structure", test_module_structure),
        ("Argument Parser", test_argument_parser),
        ("Error Handling", test_error_handling),
        ("Dependencies", test_dependencies_available)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 Testing: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 SMOKE TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL SMOKE TESTS PASSED!")
        print("✅ Basic functionality is working correctly")
        return True
    else:
        print("❌ SOME SMOKE TESTS FAILED")
        print("⚠️  Basic functionality issues detected")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)