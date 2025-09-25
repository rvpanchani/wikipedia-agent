#!/usr/bin/env python3
"""
Provider abstraction layer for different LLM services.
Supports OpenAI, Azure OpenAI, Gemini, Ollama, and Hugging Face models.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import warnings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
    
    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generate content from a prompt."""
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


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or self.get_default_model()
        self.client = None
        if self.is_available():
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return bool(self.api_key)
        except ImportError:
            return False
    
    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-3.5-turbo"


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 azure_endpoint: Optional[str] = None, api_version: Optional[str] = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
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
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using Azure OpenAI API."""
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI is available."""
        try:
            import openai
            return bool(self.api_key and self.azure_endpoint)
        except ImportError:
            return False
    
    @classmethod
    def get_default_model(cls) -> str:
        return "gpt-35-turbo"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or self.get_default_model()
        self.client = None
        if self.is_available():
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini API."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        try:
            import google.generativeai
            return bool(self.api_key)
        except ImportError:
            return False
    
    @classmethod
    def get_default_model(cls) -> str:
        return "gemini-2.0-flash-exp"


class OllamaProvider(LLMProvider):
    """Ollama local API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or self.get_default_model()
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = None
        if self.is_available():
            import openai
            # Ollama uses OpenAI-compatible API
            self.client = openai.OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama"  # Ollama doesn't require real API key
            )
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using Ollama API."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import openai
            import requests
            # Check if Ollama server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (ImportError, requests.RequestException):
            return False
    
    @classmethod
    def get_default_model(cls) -> str:
        return "llama2"


class HuggingFaceProvider(LLMProvider):
    """Hugging Face API provider (using OpenAI-compatible endpoint)."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.model = model or self.get_default_model()
        self.base_url = base_url or os.getenv("HUGGINGFACE_BASE_URL", "https://api.huggingface.co/v1")
        self.client = None
        if self.is_available():
            import openai
            self.client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using Hugging Face API."""
        if not self.client:
            raise RuntimeError("Hugging Face client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Hugging Face API error: {e}")
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        try:
            import openai
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
            raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")
        
        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers."""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def auto_detect_provider(cls) -> Optional[str]:
        """Auto-detect the best available provider based on environment variables."""
        # Check for API keys in order of preference
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
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return "ollama"
        except:
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
    
    # Remove None values
    return {k: v for k, v in config.items() if v is not None}