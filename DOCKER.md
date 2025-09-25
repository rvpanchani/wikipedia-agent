# Docker Guide for Wikipedia Agent

This guide provides comprehensive instructions for running Wikipedia Agent using Docker with support for all providers, including Ollama with the optimized qwen3:0.6b model.

## Quick Start

### Option 1: Using Ollama (No API Keys Required) ⭐ **Recommended**

The easiest way to get started - uses local qwen3:0.6b model (523MB, 40K context window):

```bash
# Build and run with automatic Ollama setup
docker-compose --profile ollama up wikipedia-agent-ollama

# Or run directly with Docker
docker build -t wikipedia-agent .
docker run --rm -p 11434:11434 wikipedia-agent "What is artificial intelligence?"
```

### Option 2: Using Cloud Providers

If you have API keys for cloud providers:

```bash
# Set your API keys
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"

# Run with cloud providers
docker-compose --profile cloud up wikipedia-agent-cloud
```

### Option 3: Development Environment

Full setup with all providers for development:

```bash
docker-compose --profile dev up wikipedia-agent-dev
```

## Docker Compose Profiles

The project includes three Docker Compose profiles:

- **`ollama`**: Local inference with Ollama (no API keys needed)
- **`cloud`**: Cloud providers only (requires API keys)
- **`dev`**: Complete development environment

## Environment Variables

### Ollama Configuration
```bash
USE_OLLAMA=true                    # Enable Ollama auto-setup
OLLAMA_MODEL=qwen3:0.6b           # Model to use (default)
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL
```

### Cloud Provider Configuration
```bash
# OpenAI
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Google Gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-1.5-flash

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_MODEL=gpt-35-turbo

# Hugging Face
HUGGINGFACE_API_KEY=your-key-here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
```

## Usage Examples

### Basic Questions
```bash
# Simple question
docker run --rm wikipedia-agent "Who was the first person to walk on the moon?"

# Historical question
docker run --rm wikipedia-agent "When did World War II end?"

# Scientific question
docker run --rm wikipedia-agent "How does photosynthesis work?"
```

### Advanced Usage
```bash
# Specify provider and model
docker run --rm \
  -e OPENAI_API_KEY="your-key" \
  wikipedia-agent \
  --provider openai --model gpt-4 \
  "Explain quantum computing"

# Use more search iterations
docker run --rm wikipedia-agent \
  --max-iterations 5 \
  "How does the human brain process memories?"

# Adjust AI parameters
docker run --rm wikipedia-agent \
  --temperature 0.3 --max-tokens 500 \
  "What is machine learning?"
```

### Interactive Mode
```bash
# Start interactive session
docker run -it --rm wikipedia-agent bash

# Inside container:
python wikipedia_agent.py "your question here"
```

## Ollama Model: qwen3:0.6b

The Docker image uses **qwen3:0.6b** as the default Ollama model:

- **Size**: 523MB (compact and efficient)
- **Context Window**: 40,000 tokens
- **Performance**: Optimized for general knowledge tasks
- **Speed**: Fast inference on CPU
- **Privacy**: Runs completely locally

### Model Features
- Efficient 0.6B parameter model
- Excellent for factual questions
- Good balance of size vs. capability
- No GPU required
- Suitable for resource-constrained environments

## Docker Image Details

### Multi-Stage Build
The Dockerfile uses a multi-stage build for optimization:
1. **Builder stage**: Installs build dependencies and Python packages
2. **Runtime stage**: Minimal production image with only runtime dependencies

### Image Size Optimization
- Uses `python:3.11-slim` as base
- Multi-stage build reduces final image size
- `.dockerignore` excludes unnecessary files
- Virtual environment for clean Python setup

### Security Features
- Runs as non-root user (`app`)
- Minimal attack surface
- Latest security patches in base image

## Health Checks

The container includes health checks for Ollama:

```bash
# Check if container is healthy
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl -f http://localhost:11434/api/tags || exit 1
```

## Persistent Data

Ollama models are stored in a named volume for persistence:

```bash
# View volume
docker volume ls | grep ollama

# Remove volume (will re-download models)
docker volume rm wikipedia-agent_ollama-data
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use different port
   docker run --rm -p 11435:11434 wikipedia-agent "question"
   ```

2. **Ollama model download fails**
   ```bash
   # Check available disk space
   df -h
   
   # Check container logs
   docker logs container-name
   ```

3. **Out of memory**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Reduce context window
   docker run --rm -e OLLAMA_CONTEXT_LENGTH=2048 wikipedia-agent "question"
   ```

4. **Container startup timeout**
   ```bash
   # Increase timeout for model download
   docker run --rm --timeout 300 wikipedia-agent "question"
   ```

### Debugging

```bash
# Run with debug output
docker run --rm -e OLLAMA_DEBUG=true wikipedia-agent "question"

# Interactive debugging
docker run -it --rm --entrypoint bash wikipedia-agent

# Check Ollama server status
docker exec container-name curl http://localhost:11434/api/tags
```

## Performance Optimization

### CPU Optimization
```bash
# Use all available CPU cores
docker run --rm --cpus="4" wikipedia-agent "question"

# Set CPU affinity
docker run --rm --cpuset-cpus="0-3" wikipedia-agent "question"
```

### Memory Optimization
```bash
# Limit memory usage
docker run --rm --memory="2g" wikipedia-agent "question"

# Set memory swap limit
docker run --rm --memory="2g" --memory-swap="4g" wikipedia-agent "question"
```

### Model Caching
```bash
# Pre-pull model to speed up subsequent runs
docker run --rm wikipedia-agent bash -c "ollama pull qwen3:0.6b"
```

## Development

### Building Custom Images
```bash
# Build with custom model
docker build --build-arg OLLAMA_MODEL=llama2 -t wikipedia-agent-llama2 .

# Build for different architecture
docker buildx build --platform linux/amd64,linux/arm64 -t wikipedia-agent .
```

### Testing
```bash
# Run Docker integration tests
python test_docker_integration.py

# Test specific functionality
docker run --rm wikipedia-agent test smoke
docker run --rm wikipedia-agent test basic
```

## Production Deployment

### Docker Swarm
```yaml
version: '3.8'
services:
  wikipedia-agent:
    image: wikipedia-agent
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    volumes:
      - ollama-data:/home/app/.ollama
volumes:
  ollama-data:
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wikipedia-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wikipedia-agent
  template:
    metadata:
      labels:
        app: wikipedia-agent
    spec:
      containers:
      - name: wikipedia-agent
        image: wikipedia-agent
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: ollama-data
          mountPath: /home/app/.ollama
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-pvc
```

## FAQ

**Q: Can I use GPU acceleration?**
A: Currently optimized for CPU. GPU support can be added by modifying the Dockerfile to include CUDA libraries.

**Q: How do I update the qwen3:0.6b model?**
A: Remove the volume and restart: `docker volume rm wikipedia-agent_ollama-data`

**Q: Can I use multiple models simultaneously?**
A: Yes, modify the docker-compose.yml to run multiple containers with different models.

**Q: What's the minimum system requirements?**
A: 2GB RAM, 1GB disk space for the model, any modern CPU.

**Q: Is it production-ready?**
A: Yes, the image follows Docker best practices and includes health checks, security measures, and optimization.