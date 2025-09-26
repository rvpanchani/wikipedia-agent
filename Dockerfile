# Multi-stage Dockerfile for Wikipedia Agent with Ollama support
# Optimized for small image size and best practices

# Stage 1: Build stage with all build dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage - minimal production image
FROM python:3.11-slim

# Install runtime dependencies and Ollama
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy application code
COPY --chown=app:app . .

# Copy entrypoint script
COPY --chown=app:app docker-entrypoint.sh /home/app/
RUN chmod +x /home/app/docker-entrypoint.sh

# Expose Ollama port
EXPOSE 11434

# Environment variables with defaults
ENV OLLAMA_BASE_URL="http://localhost:11434"
ENV OLLAMA_MODEL="qwen3:0.6b"
ENV PYTHONPATH="/home/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=5)" || exit 1

# Default entrypoint
ENTRYPOINT ["/home/app/docker-entrypoint.sh"]

# Default command - show help
CMD ["--help"]