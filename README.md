# Wikipedia Agent

A simple command line agent to answer natural language questions using Wikipedia as a knowledge source and various LLM providers (OpenAI, Azure OpenAI, Google Gemini, Ollama, Hugging Face) for intelligent search term generation and answer synthesis.

## Features

- **Multiple LLM Providers**: Support for OpenAI, Azure OpenAI, Google Gemini, Ollama, and Hugging Face
- **Auto-Detection**: Automatically detects available providers based on your configuration
- **Natural Language Questions**: Ask questions in plain English
- **Intelligent Search**: Uses AI to generate relevant Wikipedia search terms
- **Iterative Search**: Automatically tries multiple search strategies until it finds a satisfactory answer
- **Simple CLI**: Easy-to-use command line interface with extensive configuration options
- **Backward Compatible**: Existing Gemini setups continue to work
- **No Complex Frameworks**: Built with minimal dependencies for maximum reliability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rvpanchani/wikipedia-agent.git
cd wikipedia-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Choose and configure an LLM provider:

### Quick Setup Options

**Option A: OpenAI (Recommended)**
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
```
Get your API key: https://platform.openai.com/api-keys

**Option B: Google Gemini (Free tier available)**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```
Get your API key: https://makersuite.google.com/app/apikey

**Option C: Ollama (Local/Private)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama2
```

**Option D: Azure OpenAI**
```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**Option E: Docker (Recommended for easy setup)**
```bash
# Use Ollama (no API keys required)
docker-compose --profile ollama up wikipedia-agent-ollama

# Use cloud providers (API keys required)
export OPENAI_API_KEY="your-key-here"
docker-compose --profile cloud up wikipedia-agent-cloud
```
📦 **For comprehensive Docker documentation, see [DOCKER.md](DOCKER.md)**

**Option F: Use .env file**
```bash
cp .env.example .env
# Edit .env with your preferred provider settings
```

For detailed provider configuration, see [PROVIDERS.md](PROVIDERS.md).

## Usage

### Docker Usage (Recommended)

**Quick Start with Ollama (No API keys needed):**
```bash
# Build and run with Ollama using qwen3:0.6b model
docker-compose --profile ollama up wikipedia-agent-ollama

# Ask a question
docker run --rm wikipedia-agent "Who was the first person to walk on the moon?"

# Interactive mode
docker-compose --profile dev up wikipedia-agent-dev
```

**With Cloud Providers:**
```bash
# Set your API keys
export OPENAI_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"

# Run with cloud providers
docker-compose --profile cloud up wikipedia-agent-cloud
```

### Basic Usage

```bash
# Auto-detects available provider
python wikipedia_agent.py "Who was the first person to walk on the moon?"
```

### Provider-Specific Usage

```bash
# Use OpenAI explicitly
python wikipedia_agent.py --provider openai "What is quantum computing?"

# Use specific model
python wikipedia_agent.py --provider openai --model gpt-4 "Explain photosynthesis"

# Use Azure OpenAI
python wikipedia_agent.py --provider azure --model gpt-35-turbo "How does AI work?"

# Use local Ollama with qwen3:0.6b (new default model)
python wikipedia_agent.py --provider ollama --model qwen3:0.6b "What is machine learning?"

# Use Gemini (legacy compatibility)
python wikipedia_agent.py --provider gemini "What causes earthquakes?"
```

### Advanced Usage

```bash
# Increase search iterations for complex questions
python wikipedia_agent.py --max-iterations 5 "How does photosynthesis work in detail?"

# Provider with custom configuration
python wikipedia_agent.py --provider azure \
    --azure-endpoint "https://custom.openai.azure.com/" \
    --model "gpt-4" \
    "Complex scientific question"

# API key from command line (any provider)
python wikipedia_agent.py --provider openai --api-key "your_key" "What is quantum computing?"
```

### Help and Options

```bash
# Show all available options
python wikipedia_agent.py --help

# Supported providers: openai, azure, gemini, ollama, huggingface
```

### Example Output

```
🤖 Using openai provider with model: gpt-3.5-turbo
🤔 Processing question: Who was the first person to walk on the moon?

📍 Iteration 1/3
🔍 Generated search terms: Neil Armstrong, Apollo 11, Moon landing, First moonwalk, Lunar surface
   Searching Wikipedia for: Neil Armstrong
   ✅ Found content (2847 characters)

============================================================
📝 ANSWER:
============================================================
Neil Armstrong was the first person to walk on the moon. He was an American astronaut and aeronautical engineer who became the first person to step onto the lunar surface on July 20, 1969, during the Apollo 11 mission. Armstrong famously said "That's one small step for man, one giant leap for mankind" as he stepped onto the Moon's surface.

🔍 Search terms used: Neil Armstrong
```

## How It Works

1. **Question Processing**: The agent receives your natural language question
2. **Search Term Generation**: Uses Google Gemini 2.0 Flash to generate relevant Wikipedia search terms
3. **Wikipedia Search**: Searches Wikipedia for each generated term
4. **Content Analysis**: Retrieves and analyzes Wikipedia content
5. **Answer Generation**: Uses Gemini to synthesize a comprehensive answer from the Wikipedia content
6. **Iterative Refinement**: If the answer isn't satisfactory, tries different search terms (up to max iterations)

## Configuration Options

- `--max-iterations`: Number of search iterations (default: 3)
- `--api-key`: Google Gemini API key (overrides environment variable)

## Dependencies

- `wikipedia==1.4.0`: Wikipedia API access
- `google-generativeai==0.8.3`: Google Gemini API client
- `python-dotenv==1.0.1`: Environment variable management

## Requirements

- Python 3.7 or higher
- Google Gemini API key
- Internet connection

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Gemini API key is correctly set
2. **Network Issues**: Ensure you have a stable internet connection
3. **Wikipedia Disambiguation**: If Wikipedia returns multiple pages for a term, the agent will try the first option

### Getting Help

```bash
python wikipedia_agent.py --help
```

## Examples of Questions You Can Ask

- "Who invented the telephone?"
- "What caused World War I?"
- "How does the human heart work?"
- "What is the capital of Japan?"
- "When was the Internet created?"
- "What are the benefits of renewable energy?"

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
