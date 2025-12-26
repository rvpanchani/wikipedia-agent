#!/usr/bin/env python3
"""
Provider abstraction layer for different LLM services.

This module provides a unified interface for working with various LLM providers:
- OpenAI (GPT models)
- Azure OpenAI (GPT models via Azure)
- Google Gemini
- Ollama (local models)
- Hugging Face (via OpenAI-compatible API)

Usage:
    from wikipedia_agent.core.providers import ProviderFactory

    # Auto-detect available provider
    provider = ProviderFactory.create_provider(
        ProviderFactory.auto_detect_provider()
    )

    # Or specify explicitly
    provider = ProviderFactory.create_provider(
        "ollama",
        model="qwen3:0.6b"
    )

    response = provider.generate_content("What is Python?")
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.kwargs = kwargs

    @abstractmethod
    def generate_content(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content from a prompt with optional system prompt override."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass

    @classmethod
    @abstractmethod
    def get_default_model(cls) -> str:
        """Get the default model for this provider."""
        pass

    def _prepare_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for chat completion APIs with proper system prompt handling."""
        messages = []
        effective_system_prompt = system_prompt or self.system_prompt
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, system_prompt, **kwargs)
        self.model = model or self.get_default_model()
        self.client = None
        if self.is_available():
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            messages = self._prepare_messages(prompt, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-3.5-turbo"


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, system_prompt, **kwargs)
        self.model = model or self.get_default_model()
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.client = None
        if self.is_available():
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content using Azure OpenAI API."""
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")

        try:
            messages = self._prepare_messages(prompt, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {e}")

    def is_available(self) -> bool:
        """Check if Azure OpenAI is available."""
        try:
            import openai  # noqa: F401
            return bool(self.api_key and self.azure_endpoint)
        except ImportError:
            return False

    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-35-turbo"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, system_prompt, **kwargs)
        self.model = model or self.get_default_model()
        self.client = None
        if self.is_available():
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }

            if self.system_prompt:
                self.client = genai.GenerativeModel(
                    self.model,
                    generation_config=generation_config,
                    system_instruction=self.system_prompt
                )
            else:
                self.client = genai.GenerativeModel(
                    self.model,
                    generation_config=generation_config
                )

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content using Gemini API."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        try:
            if system_prompt and system_prompt != self.system_prompt:
                import google.generativeai as genai
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                temp_client = genai.GenerativeModel(
                    self.model,
                    generation_config=generation_config,
                    system_instruction=system_prompt
                )
                response = temp_client.generate_content(prompt)
            else:
                response = self.client.generate_content(prompt)

            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def is_available(self) -> bool:
        """Check if Gemini is available."""
        try:
            import google.generativeai  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    @classmethod
    def get_default_model(cls) -> str:
        return "gemini-2.0-flash-exp"


class OllamaProvider(LLMProvider):
    """Ollama local API provider using native ollama Python client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, system_prompt, **kwargs)
        self.model = model or self.get_default_model()
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.stream = stream
        self.client = None
        if self.is_available():
            import ollama
            if self.base_url != "http://localhost:11434":
                self.client = ollama.Client(host=self.base_url)
            else:
                self.client = ollama.Client()

    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """Generate content using native Ollama API with advanced features."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")

        try:
            options = {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
            options.update(kwargs)

            if enable_thinking and "qwen" in self.model.lower():
                options["enable_thinking"] = True

            messages = self._prepare_messages(prompt, system_prompt)

            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=self.stream,
                options=options
            )

            if self.stream:
                content = ""
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content += chunk['message']['content']
                return content.strip()
            else:
                if 'message' in response and 'content' in response['message']:
                    return response['message']['content'].strip()
                else:
                    raise RuntimeError("Unexpected response format from Ollama")

        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_with_thinking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate content with thinking mode enabled (for compatible models)."""
        return self.generate_content(prompt, system_prompt, enable_thinking=True)

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")

        try:
            response = self.client.list()
            return response.get('models', [])
        except Exception as e:
            raise RuntimeError(f"Failed to list Ollama models: {e}")

    def pull_model(self, model_name: str) -> bool:
        """Pull a model to make it available in Ollama."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")

        try:
            self.client.pull(model_name)
            return True
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}")
            return False

    def check_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama."""
        try:
            models = self.list_models()
            return any(m.get('name', '').startswith(model_name) for m in models)
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import ollama
            if self.base_url != "http://localhost:11434":
                test_client = ollama.Client(host=self.base_url)
            else:
                test_client = ollama.Client()
            test_client.list()
            return True
        except (ImportError, Exception):
            return False

    @classmethod
    def get_default_model(cls) -> str:
        return "qwen3:0.6b"


class HuggingFaceProvider(LLMProvider):
    """Hugging Face API provider (using OpenAI-compatible endpoint)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, model, temperature, max_tokens, system_prompt, **kwargs)
        self.model = model or self.get_default_model()
        self.base_url = base_url or os.getenv("HUGGINGFACE_BASE_URL", "https://api.huggingface.co/v1")
        self.client = None
        if self.is_available():
            import openai
            self.client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

    def generate_content(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate content using Hugging Face API."""
        if not self.client:
            raise RuntimeError("Hugging Face client not initialized")

        try:
            messages = self._prepare_messages(prompt, system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Hugging Face API error: {e}")

    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        try:
            import openai  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    @classmethod
    def get_default_model(cls) -> str:
        return "microsoft/DialoGPT-medium"


class ProviderFactory:
    """Factory for creating LLM providers."""

    PROVIDERS = {
        'openai': OpenAIProvider,
        'azure': AzureOpenAIProvider,
        'azure-openai': AzureOpenAIProvider,
        'gemini': GeminiProvider,
        'google': GeminiProvider,
        'ollama': OllamaProvider,
        'huggingface': HuggingFaceProvider,
        'hf': HuggingFaceProvider,
    }

    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> LLMProvider:
        """Create a provider instance."""
        provider_name = provider_name.lower().strip()
        if provider_name not in cls.PROVIDERS:
            available = ', '.join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {available}"
            )

        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(**kwargs)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def auto_detect_provider(cls) -> Optional[str]:
        """Auto-detect the best available provider based on environment variables."""
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            return "azure"
        elif os.getenv("GEMINI_API_KEY"):
            return "gemini"
        elif os.getenv("HUGGINGFACE_API_KEY"):
            return "huggingface"

        # Check for local Ollama
        try:
            import ollama
            client = ollama.Client()
            client.list()
            return "ollama"
        except Exception:
            pass

        return None


def get_provider_config_from_env(provider_name: str) -> Dict[str, Any]:
    """Get provider configuration from environment variables."""
    config = {}

    if provider_name in ['openai']:
        config['api_key'] = os.getenv('OPENAI_API_KEY')
        config['model'] = os.getenv('OPENAI_MODEL')

    elif provider_name in ['azure', 'azure-openai']:
        config['api_key'] = os.getenv('AZURE_OPENAI_API_KEY')
        config['azure_endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
        config['api_version'] = os.getenv('AZURE_OPENAI_API_VERSION')
        config['model'] = os.getenv('AZURE_OPENAI_MODEL')

    elif provider_name in ['gemini', 'google']:
        config['api_key'] = os.getenv('GEMINI_API_KEY')
        config['model'] = os.getenv('GEMINI_MODEL')

    elif provider_name == 'ollama':
        config['base_url'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        config['model'] = os.getenv('OLLAMA_MODEL')

    elif provider_name in ['huggingface', 'hf']:
        config['api_key'] = os.getenv('HUGGINGFACE_API_KEY')
        config['base_url'] = os.getenv('HUGGINGFACE_BASE_URL')
        config['model'] = os.getenv('HUGGINGFACE_MODEL')

    return {k: v for k, v in config.items() if v is not None}
