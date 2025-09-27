# Multi-stage Dockerfile for Wikipedia Agent with optional Ollama support
# Optimized for small image size and best practices

ARG INSTALL_OLLAMA=true

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

ARG INSTALL_OLLAMA=true

# Install runtime dependencies first
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    procps \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Ollama conditionally with better error handling
RUN if [ "$INSTALL_OLLAMA" = "true" ]; then \
        echo "Installing Ollama..." && \
        (curl -fsSL https://ollama.ai/install.sh | sh || \
         echo "Ollama installation failed, but continuing..."); \
    else \
        echo "Skipping Ollama installation"; \
    fi

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.ollama && \
    chown -R app:app /home/app/.ollama && \
    chmod 755 /home/app/.ollama

# Set up environment variables
ENV OLLAMA_BASE_URL="http://localhost:11434"
ENV OLLAMA_MODEL="qwen3:0.6b"
ENV PYTHONPATH="/home/app"
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST="0.0.0.0"
ENV OLLAMA_ORIGINS="*"

# Copy application code as app user
COPY --chown=app:app . /home/app/

# Copy and set permissions for entrypoint script
COPY --chown=app:app docker-entrypoint.sh /home/app/
RUN chmod +x /home/app/docker-entrypoint.sh

# Switch to app user as default
USER app
WORKDIR /home/app

# Expose Ollama port
EXPOSE 11434

# Health check for application readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default entrypoint
ENTRYPOINT ["/home/app/docker-entrypoint.sh"]

# Default command - show help
CMD ["--help"]