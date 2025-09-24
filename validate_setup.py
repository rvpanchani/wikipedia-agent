#!/usr/bin/env python3
"""
Setup validation script for Wikipedia Agent integration tests.
This script verifies that all testing components are properly configured.
"""

import os
import sys
import yaml
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False


def check_github_actions_workflow() -> bool:
    """Validate GitHub Actions workflow configuration."""
    workflow_path = ".github/workflows/integration-tests.yml"
    
    if not check_file_exists(workflow_path, "GitHub Actions workflow"):
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check essential workflow components
        required_keys = ['name', 'jobs']
        for key in required_keys:
            if key not in workflow:
                print(f"❌ Missing required key in workflow: {key}")
                return False
        
        # Check for 'on' key (which YAML might parse as True)
        if True not in workflow and 'on' not in workflow:
            print("❌ Missing 'on' trigger configuration in workflow")
            return False
        
        # Check jobs
        if 'basic-tests' not in workflow['jobs']:
            print("❌ Missing 'basic-tests' job")
            return False
        
        if 'integration-tests' not in workflow['jobs']:
            print("❌ Missing 'integration-tests' job")
            return False
        
        # Check for GEMINI_API_KEY usage
        workflow_str = str(workflow)
        if 'GEMINI_API_KEY' not in workflow_str:
            print("❌ GEMINI_API_KEY not referenced in workflow")
            return False
        
        print("✅ GitHub Actions workflow validation passed")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in workflow file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating workflow: {e}")
        return False


def check_test_files() -> bool:
    """Check that all test files exist and are executable."""
    test_files = [
        ("test_basic.py", "Basic functionality tests"),
        ("test_smoke.py", "Smoke tests (no API key)"),
        ("test_integration.py", "Integration tests (with API key)"),
        ("test_documented_features.py", "Documented features tests")
    ]
    
    all_exist = True
    for filename, description in test_files:
        if not check_file_exists(filename, description):
            all_exist = False
        else:
            # Check if file is executable
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                    if 'def main(' in content and '__name__ == "__main__"' in content:
                        print(f"   ✅ {filename} is executable")
                    else:
                        print(f"   ⚠️  {filename} may not be executable")
            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")
                all_exist = False
    
    return all_exist


def check_dependencies() -> bool:
    """Check that requirements.txt exists and contains expected dependencies."""
    if not check_file_exists("requirements.txt", "Requirements file"):
        return False
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        expected_deps = ["wikipedia", "google-generativeai", "python-dotenv"]
        missing_deps = []
        
        for dep in expected_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies in requirements.txt: {missing_deps}")
            return False
        
        print("✅ All required dependencies found in requirements.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False


def check_main_application() -> bool:
    """Check that the main application file exists and has expected structure."""
    if not check_file_exists("wikipedia_agent.py", "Main application"):
        return False
    
    try:
        with open("wikipedia_agent.py", 'r') as f:
            content = f.read()
        
        expected_components = [
            "class WikipediaAgent",
            "def main(",
            "GEMINI_API_KEY",
            "generate_search_terms",
            "search_wikipedia",
            "answer_question",
            "process_query"
        ]
        
        missing_components = []
        for component in expected_components:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components in main application: {missing_components}")
            return False
        
        print("✅ Main application structure validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading main application: {e}")
        return False


def check_documentation() -> bool:
    """Check that documentation files exist."""
    docs = [
        ("README.md", "Main documentation"),
        ("TESTING.md", "Testing documentation")
    ]
    
    all_exist = True
    for filename, description in docs:
        if not check_file_exists(filename, description):
            all_exist = False
    
    return all_exist


def main():
    """Run all validation checks."""
    print("🔍 Validating Wikipedia Agent Integration Test Setup")
    print("=" * 70)
    
    checks = [
        ("File Structure", check_test_files),
        ("Dependencies", check_dependencies), 
        ("Main Application", check_main_application),
        ("GitHub Actions Workflow", check_github_actions_workflow),
        ("Documentation", check_documentation)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🧪 Checking: {check_name}")
        print("-" * 40)
        
        if check_func():
            passed += 1
            print(f"✅ {check_name} validation passed")
        else:
            print(f"❌ {check_name} validation failed")
    
    print("\n" + "=" * 70)
    print(f"📊 SETUP VALIDATION RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL SETUP VALIDATIONS PASSED!")
        print("\n✅ Your integration test setup is ready!")
        print("\nNext steps:")
        print("1. Ensure GEMINI_API_KEY is added to GitHub repository secrets")
        print("2. Push changes to trigger GitHub Actions workflow")
        print("3. Monitor workflow execution in Actions tab")
        return True
    else:
        print("❌ SETUP VALIDATION ISSUES DETECTED")
        print(f"\n⚠️  Please fix {total - passed} issue(s) before proceeding")
        return False


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML for workflow validation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])
        import yaml
    
    success = main()
    sys.exit(0 if success else 1)