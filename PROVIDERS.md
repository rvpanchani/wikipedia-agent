# Provider Configuration Guide

Wikipedia Agent now supports multiple LLM providers for maximum flexibility. This guide explains how to configure and use different providers.

## Supported Providers

| Provider | Description | Cost | Setup Difficulty |
|----------|-------------|------|------------------|
| **OpenAI** | GPT models via OpenAI API | Paid | Easy |
| **Azure OpenAI** | GPT models via Azure | Paid | Medium |
| **Google Gemini** | Google's Gemini models | Free tier available | Easy |
| **Ollama** | Local open-source models | Free | Medium |
| **Hugging Face** | Various open-source models | Free/Paid | Medium |

## Quick Start

### 1. OpenAI (Recommended)

Most reliable and fastest option.

```bash
# Set environment variable
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# Run with auto-detection
python wikipedia_agent.py "Who invented the telephone?"

# Or explicitly specify
python wikipedia_agent.py --provider openai "Who invented the telephone?"
```

**Get API key:** https://platform.openai.com/api-keys

### 2. Google Gemini

Good free option with generous limits.

```bash
# Set environment variable
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run (backward compatible)
python wikipedia_agent.py "What is quantum computing?"

# Or explicitly specify
python wikipedia_agent.py --provider gemini "What is quantum computing?"
```

**Get API key:** https://makersuite.google.com/app/apikey

### 3. Ollama (Local)

Run models locally for complete privacy.

```bash
# First, install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull a model
ollama pull llama2

# Run Wikipedia Agent
python wikipedia_agent.py --provider ollama --model llama2 "Explain photosynthesis"
```

**More info:** https://ollama.ai/

## Detailed Configuration

### OpenAI Configuration

```bash
# Environment variables
export OPENAI_API_KEY="sk-your-key-here"
export OPENAI_MODEL="gpt-3.5-turbo"  # Optional, defaults to gpt-3.5-turbo

# Command line
python wikipedia_agent.py --provider openai --model gpt-4 "your question"
```

**Available models:**
- `gpt-3.5-turbo` (default, fastest, cheapest)
- `gpt-4` (more capable, slower, more expensive)
- `gpt-4-turbo` (balance of speed and capability)

### Azure OpenAI Configuration

```bash
# Environment variables
export AZURE_OPENAI_API_KEY="your-azure-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # Optional
export AZURE_OPENAI_MODEL="gpt-35-turbo"  # Optional

# Command line
python wikipedia_agent.py --provider azure \
    --azure-endpoint "https://your-resource.openai.azure.com/" \
    --model "gpt-35-turbo" \
    "your question"
```

**Setup:**
1. Create Azure OpenAI resource in Azure Portal
2. Deploy a model (e.g., GPT-3.5-turbo)
3. Get endpoint URL and API key

### Google Gemini Configuration

```bash
# Environment variables
export GEMINI_API_KEY="your-gemini-key-here"
export GEMINI_MODEL="gemini-2.0-flash-exp"  # Optional

# Command line (legacy support)
python wikipedia_agent.py --gemini-api-key "your-key" "your question"

# Command line (new way)
python wikipedia_agent.py --provider gemini --model "gemini-pro" "your question"
```

**Available models:**
- `gemini-2.0-flash-exp` (default, latest experimental)
- `gemini-pro` (stable, production-ready)
- `gemini-pro-vision` (supports images, not needed for text)

### Ollama Configuration

```bash
# Environment variables
export OLLAMA_BASE_URL="http://localhost:11434"  # Optional, this is default
export OLLAMA_MODEL="llama2"  # Optional

# First time setup
ollama pull llama2  # or codellama, mistral, etc.

# Command line
python wikipedia_agent.py --provider ollama --model llama2 "your question"
```

**Popular models:**
- `llama2` (7B parameters, good balance)
- `llama2:13b` (larger, more capable)
- `codellama` (specialized for code)
- `mistral` (7B, efficient)
- `phi` (3B, fast, smaller)

### Hugging Face Configuration

```bash
# Environment variables
export HUGGINGFACE_API_KEY="hf_your-token-here"
export HUGGINGFACE_MODEL="microsoft/DialoGPT-medium"  # Optional

# Command line
python wikipedia_agent.py --provider huggingface "your question"
```

**Get token:** https://huggingface.co/settings/tokens

## Provider Auto-Detection

The agent automatically detects available providers in this order:

1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Azure OpenAI** (if both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set)
3. **Google Gemini** (if `GEMINI_API_KEY` is set)
4. **Hugging Face** (if `HUGGINGFACE_API_KEY` is set)
5. **Ollama** (if local server is running at `http://localhost:11434`)

## Advanced Usage

### Using .env Files

Create a `.env` file in your project directory:

```bash
# Copy example and edit
cp .env.example .env

# Edit with your preferred editor
nano .env
```

Example `.env` file:
```env
# Use OpenAI as primary
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Fallback to Gemini
GEMINI_API_KEY=your-gemini-key-here
```

### Multiple Providers Setup

You can configure multiple providers and switch between them:

```bash
# Set up multiple providers
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="your-gemini-key"

# Use OpenAI
python wikipedia_agent.py --provider openai "Question 1"

# Use Gemini  
python wikipedia_agent.py --provider gemini "Question 2"

# Use auto-detection (will pick OpenAI first)
python wikipedia_agent.py "Question 3"
```

### Provider-Specific Options

Each provider supports additional configuration:

```bash
# OpenAI with specific model
python wikipedia_agent.py --provider openai --model gpt-4 "complex question"

# Azure with custom endpoint
python wikipedia_agent.py --provider azure \
    --azure-endpoint "https://custom.openai.azure.com/" \
    --azure-api-version "2024-02-15-preview" \
    "your question"

# Ollama with custom URL
python wikipedia_agent.py --provider ollama \
    --ollama-base-url "http://remote-server:11434" \
    --model "mistral" \
    "your question"
```

## Troubleshooting

### Common Issues

1. **"No properly configured LLM provider found"**
   - Check that you've set at least one API key
   - Verify API key format and validity
   - For Azure, ensure both API key and endpoint are set

2. **"Provider 'xyz' is not properly configured"**
   - Verify your API key is correct
   - Check network connectivity
   - For Azure, verify endpoint URL format

3. **Ollama connection errors**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is installed: `ollama list`
   - Verify port 11434 is accessible

4. **Rate limiting errors**
   - Implement delays between requests
   - Upgrade to paid tier if using free tier
   - Switch to a different provider temporarily

### Getting Help

```bash
# Show all available options
python wikipedia_agent.py --help

# Test provider configuration
python -c "from providers import ProviderFactory; print(ProviderFactory.auto_detect_provider())"
```

## Cost Comparison

| Provider | Model | Cost per 1K tokens | Free Tier |
|----------|-------|-------------------|-----------|
| OpenAI | GPT-3.5-turbo | $0.001-0.002 | $5 credit |
| OpenAI | GPT-4 | $0.03-0.06 | $5 credit |
| Azure OpenAI | GPT-3.5-turbo | Same as OpenAI | $200 Azure credit |
| Google Gemini | Gemini Pro | Free up to 60 req/min | Yes |
| Ollama | Any model | $0 (local compute) | Yes |
| Hugging Face | Various | $0.0002-0.0008 | Limited free |

## Security Considerations

- **Never commit API keys to version control**
- **Use environment variables or .env files**
- **Rotate API keys regularly**
- **For production, use secrets management systems**
- **Ollama provides complete data privacy (local processing)**

## Performance Tips

1. **Use GPT-3.5-turbo for speed and cost efficiency**
2. **Use GPT-4 only for complex reasoning tasks**
3. **Ollama is slower but private and free**
4. **Gemini offers good balance of free tier and performance**
5. **Implement request caching for repeated questions**