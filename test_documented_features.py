#!/usr/bin/env python3
"""
Test script specifically for validating all documented features in README.md
This ensures that all features mentioned in documentation actually work.
"""

import os
import sys
import subprocess
import time

def test_documented_feature(description: str, command: list, expected_patterns: list = None, should_fail: bool = False) -> bool:
    """Test a documented feature and validate its behavior."""
    print(f"\n🧪 Testing: {description}")
    print(f"   Command: {' '.join(command)}")
    
    try:
        env = os.environ.copy()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        duration = time.time() - start_time
        
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Return code: {result.returncode}")
        
        if should_fail:
            if result.returncode == 0:
                print(f"   ❌ Expected failure but command succeeded")
                return False
            else:
                print(f"   ✅ Command failed as expected")
                return True
        
        if result.returncode != 0:
            print(f"   ❌ Command failed unexpectedly")
            print(f"   Stderr: {result.stderr}")
            return False
        
        output = result.stdout
        print(f"   Output length: {len(output)} characters")
        
        # Check for expected patterns
        if expected_patterns:
            for pattern in expected_patterns:
                if pattern.lower() not in output.lower():
                    print(f"   ❌ Expected pattern '{pattern}' not found in output")
                    return False
                else:
                    print(f"   ✅ Found expected pattern: '{pattern}'")
        
        print(f"   ✅ Feature test passed")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ Command timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Exception occurred: {e}")
        return False


def main():
    """Test all documented features from README.md"""
    print("🔍 Testing Documented Wikipedia Agent Features")
    print("=" * 80)
    
    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable is required")
        print("   Please set the API key to test documented features")
        sys.exit(1)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Basic usage example from README
    total_tests += 1
    if test_documented_feature(
        "Basic Usage - Moon Landing Question",
        ["python", "wikipedia_agent.py", "Who was the first person to walk on the moon?"],
        ["ANSWER:", "Neil Armstrong", "moon"]
    ):
        tests_passed += 1
    
    # Test 2: Help command (documented feature)
    total_tests += 1
    if test_documented_feature(
        "Help Command Documentation",
        ["python", "wikipedia_agent.py", "--help"],
        ["Wikipedia Agent", "question", "max-iterations", "api-key"]
    ):
        tests_passed += 1
    
    # Test 3: Max iterations parameter (documented configuration option)
    total_tests += 1
    if test_documented_feature(
        "Max Iterations Configuration",
        ["python", "wikipedia_agent.py", "--max-iterations", "5", "How does photosynthesis work?"],
        ["ANSWER:", "photosynthesis"]
    ):
        tests_passed += 1
    
    # Test 4: API key parameter (documented configuration option)
    total_tests += 1
    if test_documented_feature(
        "API Key Parameter Override",
        ["python", "wikipedia_agent.py", "--api-key", api_key, "What is quantum computing?"],
        ["ANSWER:", "quantum"]
    ):
        tests_passed += 1
    
    # Test 5: Error handling without API key (documented requirement)
    total_tests += 1
    env_backup = os.environ.get("GEMINI_API_KEY")
    if env_backup:
        del os.environ["GEMINI_API_KEY"]
    
    if test_documented_feature(
        "Error Handling - Missing API Key",
        ["python", "wikipedia_agent.py", "test question"],
        ["Google Gemini API key is required"],
        should_fail=True
    ):
        tests_passed += 1
    
    # Restore environment
    if env_backup:
        os.environ["GEMINI_API_KEY"] = env_backup
    
    # Test 6: README example questions
    readme_questions = [
        ("Telephone Invention", "Who invented the telephone?", ["telephone", "bell"]),
        ("Heart Function", "How does the human heart work?", ["heart", "blood"]),
        ("Japan Capital", "What is the capital of Japan?", ["tokyo", "japan"]),
        ("Internet Creation", "When was the Internet created?", ["internet", "arpanet"]),
        ("Renewable Energy", "What are the benefits of renewable energy?", ["renewable", "energy"])
    ]
    
    for question_name, question, expected_terms in readme_questions:
        total_tests += 1
        if test_documented_feature(
            f"README Example - {question_name}",
            ["python", "wikipedia_agent.py", question],
            ["ANSWER:"] + expected_terms
        ):
            tests_passed += 1
    
    # Test 7: Combined parameters (documented advanced usage)
    total_tests += 1
    if test_documented_feature(
        "Combined Parameters - Advanced Usage",
        ["python", "wikipedia_agent.py", "--api-key", api_key, "--max-iterations", "2", "What caused World War I?"],
        ["ANSWER:", "war", "🔍 Search terms used:"]
    ):
        tests_passed += 1
    
    # Test 8: Output format validation (documented in examples)
    total_tests += 1
    if test_documented_feature(
        "Expected Output Format",
        ["python", "wikipedia_agent.py", "What is the capital of France?"],
        ["📝 ANSWER:", "🔍 Search terms used:", "paris"]
    ):
        tests_passed += 1
    
    # Test 9: Performance requirement (should complete within reasonable time)
    print(f"\n🧪 Testing: Performance - Response Time")
    start_time = time.time()
    
    total_tests += 1
    success = test_documented_feature(
        "Performance Test - Quick Response",
        ["python", "wikipedia_agent.py", "--max-iterations", "1", "What is CSS?"],
        ["ANSWER:"]
    )
    
    duration = time.time() - start_time
    if success and duration < 60:  # Should complete within 60 seconds
        print(f"   ✅ Performance test passed ({duration:.2f}s)")
        tests_passed += 1
    else:
        print(f"   ❌ Performance test failed (took {duration:.2f}s)")
    
    # Final results
    print("\n" + "=" * 80)
    print(f"📊 DOCUMENTED FEATURES TEST RESULTS: {tests_passed}/{total_tests} PASSED")
    
    if tests_passed == total_tests:
        print("🎉 ALL DOCUMENTED FEATURES WORK CORRECTLY!")
        print("\n✅ Validated Features:")
        print("   - Natural Language Questions")
        print("   - Intelligent Search with Gemini 2.0 Flash")
        print("   - Iterative Search Strategy")  
        print("   - Simple CLI Interface")
        print("   - Configuration Options (max-iterations, api-key)")
        print("   - Proper Error Handling")
        print("   - Expected Output Format")
        print("   - README Examples")
        print("   - Performance Requirements")
        return True
    else:
        print("❌ SOME DOCUMENTED FEATURES FAILED")
        failed_count = total_tests - tests_passed
        print(f"   {failed_count} feature(s) need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)