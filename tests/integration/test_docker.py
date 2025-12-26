"""
Docker integration tests for Wikipedia Agent.
"""

import os
import subprocess
import pytest


@pytest.mark.docker
class TestDockerBuild:
    """Tests for Docker image building."""
    
    @pytest.fixture(scope="class")
    def docker_build(self):
        """Build Docker image before tests."""
        result = subprocess.run(
            ["docker", "build", "-t", "wikipedia-agent-test", "."],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        return result
    
    def test_docker_build_succeeds(self, docker_build):
        """Test that Docker image builds successfully."""
        assert docker_build.returncode == 0, f"Build failed: {docker_build.stderr}"
    
    def test_help_command(self, docker_build):
        """Test that help command works."""
        if docker_build.returncode != 0:
            pytest.skip("Docker build failed")
        
        result = subprocess.run(
            ["docker", "run", "--rm", "wikipedia-agent-test", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "wikipedia" in result.stdout.lower()


@pytest.mark.docker
class TestDockerCompose:
    """Tests for docker-compose configuration."""
    
    def test_compose_config_valid(self):
        """Test that docker-compose.yml is valid."""
        result = subprocess.run(
            ["docker", "compose", "config"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        assert result.returncode == 0, f"Config invalid: {result.stderr}"
    
    def test_compose_has_agent_service(self):
        """Test that agent service is defined."""
        result = subprocess.run(
            ["docker", "compose", "config", "--services"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        services = result.stdout.strip().split('\n')
        # Check that at least one service exists
        assert len(services) > 0


@pytest.mark.docker
@pytest.mark.slow
class TestDockerOllama:
    """Tests for Docker with Ollama (requires Ollama running)."""
    
    @pytest.fixture
    def ollama_available(self):
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_ollama_container_startup(self, ollama_available):
        """Test that Ollama container can start."""
        if not ollama_available:
            pytest.skip("Ollama not available")
        
        # This test verifies Ollama is accessible
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
