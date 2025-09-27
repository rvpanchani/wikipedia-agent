#!/bin/bash
set -e

# Docker entrypoint script for Wikipedia Agent with Ollama support

echo "🐳 Starting Wikipedia Agent Docker Container"

# Function to check if Ollama is needed
needs_ollama() {
    # Don't use Ollama for help commands
    local args=("$@")
    for arg in "${args[@]}"; do
        if [[ "$arg" == "--help" || "$arg" == "help" ]]; then
            return 1
        fi
    done
    
    # Check if user explicitly wants Ollama
    if [[ "$USE_OLLAMA" == "true" ]]; then
        return 0
    fi
    
    # Check if --provider ollama is specified in arguments
    for i in "${!args[@]}"; do
        if [[ "${args[$i]}" == "--provider" && "${args[$((i+1))]}" == "ollama" ]]; then
            return 0
        fi
    done
    
    # Don't auto-enable Ollama if any cloud provider is configured
    if [[ -n "$OPENAI_API_KEY" || -n "$GEMINI_API_KEY" || -n "$AZURE_OPENAI_API_KEY" || -n "$HUGGINGFACE_API_KEY" ]]; then
        return 1
    fi
    
    # For actual questions (not help), enable Ollama if no cloud providers configured
    if [[ "$1" != -* ]]; then
        echo "📡 No API keys detected for cloud providers, enabling Ollama for local inference"
        return 0
    fi
    
    # For other CLI arguments that need a provider, check if Ollama is available
    if command -v ollama >/dev/null 2>&1; then
        echo "📡 No API keys detected for cloud providers, enabling Ollama for local inference"
        return 0
    fi
    
    return 1
}

# Function to setup Ollama
setup_ollama() {
    echo "🦙 Setting up Ollama..."
    
    # Check if Ollama is installed
    if ! command -v ollama >/dev/null 2>&1; then
        echo "❌ Ollama is not installed in this container"
        echo "💡 To use Ollama, rebuild the container with: docker build --build-arg INSTALL_OLLAMA=true ."
        echo "💡 Or use a cloud provider with API keys instead"
        exit 1
    fi
    
    # Ensure Ollama data directory exists and has proper permissions
    mkdir -p ~/.ollama
    
    # Start Ollama server in background
    echo "🔄 Starting Ollama server..."
    OLLAMA_HOST=0.0.0.0 ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama server to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama server is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            echo "❌ Ollama server failed to start within 60 seconds"
            echo "📋 Debug info:"
            echo "   - Ollama PID: $OLLAMA_PID"
            echo "   - Process status: $(ps aux | grep ollama 2>/dev/null || echo 'No ollama process found')"
            echo "   - Port status: $(ss -tlnp 2>/dev/null | grep 11434 || echo 'Port 11434 not in use')"
            
            # Try to get Ollama logs
            if kill -0 $OLLAMA_PID 2>/dev/null; then
                echo "   - Ollama process is still running, may need more time"
            else
                echo "   - Ollama process has exited"
            fi
            exit 1
        fi
        sleep 1
    done
    
    # Pull the default model if not already present
    local model="${OLLAMA_MODEL:-qwen3:0.6b}"
    echo "📥 Checking for $model model..."
    if ! ollama list | grep -q "$model"; then
        echo "📦 Pulling $model model (this may take several minutes)..."
        if ollama pull "$model"; then
            echo "✅ $model model ready"
        else
            echo "❌ Failed to pull $model model"
            echo "📋 Available models:"
            ollama list || echo "   Could not list models"
            echo "💡 You can manually pull the model with: ollama pull $model"
            exit 1
        fi
    else
        echo "✅ $model model already available"
    fi
    
    # Store Ollama PID for cleanup
    echo $OLLAMA_PID > /tmp/ollama.pid
    echo "📝 Ollama server running with PID: $OLLAMA_PID"
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
    "--help"|"help")
        echo "🚀 Showing Wikipedia Agent help"
        exec python wikipedia_agent.py --help
        ;;
    *)
        # Treat as a question for the Wikipedia Agent
        echo "🤔 Treating as question: $@"
        exec python wikipedia_agent.py "$@"
        ;;
esac