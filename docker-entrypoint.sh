#!/bin/bash
set -e

# Docker entrypoint script for Wikipedia Agent with Ollama support

echo "🐳 Starting Wikipedia Agent Docker Container"

# Function to check if Ollama is needed
needs_ollama() {
    # Check if user explicitly wants Ollama or if no other provider is configured
    if [[ "$USE_OLLAMA" == "true" ]] || [[ "$1" == "--provider" && "$2" == "ollama" ]]; then
        return 0
    fi
    
    # Check if no other providers are configured (auto-detect Ollama usage)
    if [[ -z "$OPENAI_API_KEY" && -z "$GEMINI_API_KEY" && -z "$AZURE_OPENAI_API_KEY" && -z "$HUGGINGFACE_API_KEY" ]]; then
        echo "📡 No API keys detected for cloud providers, enabling Ollama for local inference"
        return 0
    fi
    
    return 1
}

# Function to setup Ollama
setup_ollama() {
    echo "🦙 Setting up Ollama..."
    
    # Start Ollama server in background
    echo "🔄 Starting Ollama server..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Ollama server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Pull the qwen3:0.6b model if not already present
    echo "📥 Checking for qwen3:0.6b model..."
    if ! ollama list | grep -q "qwen3:0.6b"; then
        echo "📦 Pulling qwen3:0.6b model (523MB)..."
        ollama pull qwen3:0.6b
        echo "✅ qwen3:0.6b model ready"
    else
        echo "✅ qwen3:0.6b model already available"
    fi
    
    # Store Ollama PID for cleanup
    echo $OLLAMA_PID > /tmp/ollama.pid
}

# Function to cleanup Ollama on exit
cleanup_ollama() {
    if [[ -f /tmp/ollama.pid ]]; then
        local pid=$(cat /tmp/ollama.pid)
        if kill -0 $pid 2>/dev/null; then
            echo "🔄 Stopping Ollama server..."
            kill $pid
            wait $pid 2>/dev/null || true
        fi
        rm -f /tmp/ollama.pid
    fi
}

# Setup signal handlers for cleanup
trap cleanup_ollama EXIT SIGTERM SIGINT

# Check if Ollama setup is needed
if needs_ollama "$@"; then
    setup_ollama
fi

# If the first argument starts with a dash, assume it's a Wikipedia Agent argument
if [[ "$1" == -* ]]; then
    echo "🚀 Running Wikipedia Agent with arguments: $@"
    exec python wikipedia_agent.py "$@"
fi

# If the first argument is a command, run it directly
case "$1" in
    "bash"|"sh"|"/bin/bash"|"/bin/sh")
        echo "🔧 Starting interactive shell"
        exec "$@"
        ;;
    "test")
        echo "🧪 Running tests"
        shift
        case "$1" in
            "smoke")
                exec python test_smoke.py
                ;;
            "basic")
                exec python test_basic.py
                ;;
            "integration")
                exec python test_integration.py
                ;;
            *)
                echo "Available tests: smoke, basic, integration"
                exit 1
                ;;
        esac
        ;;
    "wikipedia_agent.py"|"python")
        echo "🚀 Running Wikipedia Agent: $@"
        exec "$@"
        ;;
    *)
        # Treat as a question for the Wikipedia Agent
        echo "🤔 Treating as question: $@"
        exec python wikipedia_agent.py "$@"
        ;;
esac