#!/usr/bin/env python3
"""
Docker integration tests for Wikipedia Agent.
Tests Docker setup, Ollama integration, and container functionality.
"""

import os
import sys
import subprocess
import time
import requests
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DockerIntegrationTestRunner:
    """Docker integration test runner for Wikipedia Agent."""
    
    def __init__(self):
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
    
    def test_docker_build(self) -> bool:
        """Test Docker image build process."""
        try:
            print("Building Docker image...")
            result = subprocess.run(
                ["docker", "build", "-t", "wikipedia-agent-test", "."],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                print(f"Docker build failed: {result.stderr}")
                return False
            
            print("✅ Docker image built successfully")
            return True
        except subprocess.TimeoutExpired:
            print("❌ Docker build timed out")
            return False
        except Exception as e:
            print(f"Docker build error: {e}")
            return False
    
    def test_docker_help_command(self) -> bool:
        """Test Docker container help command."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "wikipedia-agent-test", "--help"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Help command failed: {result.stderr}")
                return False
            
            # Check if help output contains expected content
            output = result.stdout
            expected_strings = ["Wikipedia Agent", "question", "max-iterations", "provider"]
            
            for expected in expected_strings:
                if expected not in output:
                    print(f"Help output missing '{expected}'")
                    return False
            
            print("✅ Help command works correctly")
            return True
        except Exception as e:
            print(f"Help command test error: {e}")
            return False
    
    def test_ollama_container_startup(self) -> bool:
        """Test Ollama container startup and model availability."""
        container_name = "wikipedia-agent-ollama-test"
        
        try:
            # Start container with Ollama in background
            print("Starting Ollama container...")
            subprocess.run(
                ["docker", "run", "-d", "--name", container_name, 
                 "-p", "11435:11434",  # Use different port to avoid conflicts
                 "-e", "USE_OLLAMA=true",
                 "-e", "OLLAMA_MODEL=qwen3:0.6b",
                 "wikipedia-agent-test", "sleep", "300"],
                check=True,
                capture_output=True
            )
            
            # Wait for container to be ready
            print("Waiting for Ollama to be ready...")
            max_wait = 120  # 2 minutes
            for i in range(max_wait):
                try:
                    response = requests.get("http://localhost:11435/api/tags", timeout=5)
                    if response.status_code == 200:
                        print("✅ Ollama server is ready")
                        break
                except requests.RequestException:
                    pass
                
                if i == max_wait - 1:
                    print("❌ Ollama server failed to start")
                    return False
                
                time.sleep(1)
            
            # Test qwen3:0.6b model availability
            try:
                response = requests.get("http://localhost:11435/api/tags", timeout=10)
                if response.status_code == 200:
                    models = response.json()
                    model_names = [model.get('name', '') for model in models.get('models', [])]
                    if any('qwen3:0.6b' in name for name in model_names):
                        print("✅ qwen3:0.6b model is available")
                        return True
                    else:
                        print(f"❌ qwen3:0.6b model not found. Available models: {model_names}")
                        return False
            except Exception as e:
                print(f"Error checking models: {e}")
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"Container startup failed: {e}")
            return False
        except Exception as e:
            print(f"Ollama test error: {e}")
            return False
        finally:
            # Cleanup container
            try:
                subprocess.run(["docker", "rm", "-f", container_name], 
                             capture_output=True, check=False)
            except:
                pass
    
    def test_docker_compose_validation(self) -> bool:
        """Test docker-compose.yml file validation."""
        try:
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"docker-compose validation failed: {result.stderr}")
                return False
            
            # Check if key services are defined
            output = result.stdout
            required_services = ["wikipedia-agent-ollama", "wikipedia-agent-cloud", "wikipedia-agent-dev"]
            
            for service in required_services:
                if service not in output:
                    print(f"Service {service} not found in docker-compose.yml")
                    return False
            
            print("✅ docker-compose.yml is valid")
            return True
        except subprocess.TimeoutExpired:
            print("❌ docker-compose validation timed out")
            return False
        except Exception as e:
            print(f"docker-compose validation error: {e}")
            return False
    
    def test_docker_entrypoint_script(self) -> bool:
        """Test Docker entrypoint script functionality."""
        try:
            # Test entrypoint script exists and is executable
            result = subprocess.run(
                ["docker", "run", "--rm", "wikipedia-agent-test", "bash", "-c", 
                 "test -x /home/app/docker-entrypoint.sh && echo 'EXECUTABLE'"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 or "EXECUTABLE" not in result.stdout:
                print("Entrypoint script is not executable")
                return False
            
            print("✅ Entrypoint script is executable")
            return True
        except Exception as e:
            print(f"Entrypoint script test error: {e}")
            return False
    
    def test_docker_environment_variables(self) -> bool:
        """Test Docker environment variable handling."""
        try:
            # Test default environment variables
            result = subprocess.run(
                ["docker", "run", "--rm", "wikipedia-agent-test", "bash", "-c",
                 "echo \"OLLAMA_MODEL=$OLLAMA_MODEL\" && echo \"OLLAMA_BASE_URL=$OLLAMA_BASE_URL\""],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Environment variable test failed: {result.stderr}")
                return False
            
            output = result.stdout
            if "OLLAMA_MODEL=qwen3:0.6b" not in output:
                print(f"OLLAMA_MODEL not set correctly. Output: {output}")
                return False
            
            if "OLLAMA_BASE_URL=http://localhost:11434" not in output:
                print(f"OLLAMA_BASE_URL not set correctly. Output: {output}")
                return False
            
            print("✅ Environment variables are set correctly")
            return True
        except Exception as e:
            print(f"Environment variable test error: {e}")
            return False
    
    def cleanup_test_images(self):
        """Clean up test Docker images."""
        try:
            subprocess.run(
                ["docker", "rmi", "-f", "wikipedia-agent-test"],
                capture_output=True,
                check=False
            )
        except:
            pass
    
    def run_all_tests(self):
        """Run all Docker integration tests."""
        print("🐳 Running Docker Integration Tests for Wikipedia Agent")
        print("=" * 80)
        print("These tests validate Docker setup, Ollama integration, and container functionality")
        print()
        
        tests = [
            ("Docker Image Build", self.test_docker_build),
            ("Docker Help Command", self.test_docker_help_command),
            ("Docker Compose Validation", self.test_docker_compose_validation),
            ("Docker Entrypoint Script", self.test_docker_entrypoint_script),
            ("Docker Environment Variables", self.test_docker_environment_variables),
            ("Ollama Container Startup", self.test_ollama_container_startup),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()
        
        print("=" * 80)
        print(f"📊 DOCKER INTEGRATION TEST RESULTS: {self.passed}/{self.total} PASSED")
        
        # Cleanup
        self.cleanup_test_images()
        
        if self.passed == self.total:
            print("🎉 ALL DOCKER INTEGRATION TESTS PASSED!")
            print("✅ Docker setup is working correctly")
            return True
        else:
            print("❌ SOME DOCKER INTEGRATION TESTS FAILED")
            failed_count = self.total - self.passed
            print(f"⚠️  {failed_count} test(s) need attention")
            return False


def main():
    """Main function to run Docker integration tests."""
    runner = DockerIntegrationTestRunner()
    success = runner.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)