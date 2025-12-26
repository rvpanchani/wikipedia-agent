"""
Unit tests for the core providers module.
"""

import pytest
import os


class TestProviderFactory:
    """Tests for ProviderFactory class."""
    
    def test_get_available_providers(self):
        """Test that available providers list is returned."""
        from wikipedia_agent.core.providers import ProviderFactory
        
        providers = ProviderFactory.get_available_providers()
        
        assert 'openai' in providers
        assert 'azure' in providers
        assert 'gemini' in providers
        assert 'ollama' in providers
        assert 'huggingface' in providers
    
    def test_create_unknown_provider_raises(self):
        """Test that creating unknown provider raises ValueError."""
        from wikipedia_agent.core.providers import ProviderFactory
        
        with pytest.raises(ValueError) as excinfo:
            ProviderFactory.create_provider("unknown_provider")
        
        assert "Unknown provider" in str(excinfo.value)
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        from wikipedia_agent.core.providers import ProviderFactory, OpenAIProvider
        
        provider = ProviderFactory.create_provider(
            "openai",
            api_key="test_key"
        )
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-3.5-turbo"
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        from wikipedia_agent.core.providers import ProviderFactory, OllamaProvider
        
        provider = ProviderFactory.create_provider(
            "ollama",
            model="qwen3:0.6b"
        )
        
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen3:0.6b"


class TestLLMProvider:
    """Tests for LLMProvider base class."""
    
    def test_prepare_messages_with_system_prompt(self):
        """Test message preparation with system prompt."""
        from wikipedia_agent.core.providers import OpenAIProvider
        
        provider = OpenAIProvider(
            api_key="test",
            system_prompt="You are a helpful assistant."
        )
        
        messages = provider._prepare_messages("Hello", None)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    def test_prepare_messages_override_system_prompt(self):
        """Test message preparation with overridden system prompt."""
        from wikipedia_agent.core.providers import OpenAIProvider
        
        provider = OpenAIProvider(
            api_key="test",
            system_prompt="Original prompt"
        )
        
        messages = provider._prepare_messages("Hello", "Override prompt")
        
        assert messages[0]["content"] == "Override prompt"
    
    def test_prepare_messages_no_system_prompt(self):
        """Test message preparation without system prompt."""
        from wikipedia_agent.core.providers import OpenAIProvider
        
        provider = OpenAIProvider(api_key="test")
        
        messages = provider._prepare_messages("Hello")
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestOllamaProvider:
    """Tests for OllamaProvider specific features."""
    
    def test_default_model(self):
        """Test default model setting."""
        from wikipedia_agent.core.providers import OllamaProvider
        
        assert OllamaProvider.get_default_model() == "qwen3:0.6b"
    
    def test_base_url_from_env(self):
        """Test base URL can be set from environment."""
        from wikipedia_agent.core.providers import OllamaProvider
        
        # This tests the default value
        provider = OllamaProvider()
        assert provider.base_url == os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class TestGetProviderConfigFromEnv:
    """Tests for get_provider_config_from_env function."""
    
    def test_openai_config(self, monkeypatch):
        """Test OpenAI config from environment."""
        from wikipedia_agent.core.providers import get_provider_config_from_env
        
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        
        config = get_provider_config_from_env("openai")
        
        assert config["api_key"] == "test_api_key"
        assert config["model"] == "gpt-4"
    
    def test_ollama_config(self, monkeypatch):
        """Test Ollama config from environment."""
        from wikipedia_agent.core.providers import get_provider_config_from_env
        
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "llama2")
        
        config = get_provider_config_from_env("ollama")
        
        assert config["base_url"] == "http://custom:11434"
        assert config["model"] == "llama2"
    
    def test_config_removes_none_values(self, monkeypatch):
        """Test that None values are removed from config."""
        from wikipedia_agent.core.providers import get_provider_config_from_env
        
        # Clear any existing env vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        
        config = get_provider_config_from_env("openai")
        
        assert "api_key" not in config
        assert "model" not in config
