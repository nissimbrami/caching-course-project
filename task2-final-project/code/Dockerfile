# Cost-Aware GDSF Eviction for GPTCache
# Multi-stage build for minimal image size
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash researcher
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Ensure scripts are executable
RUN chmod +x scripts/run_all.sh scripts/run_ablation.py scripts/generate_plots.py

# Create results directories
RUN mkdir -p results/plots results/benchmarks results/ablation

# Switch to non-root user
RUN chown -R researcher:researcher /app
USER researcher

# Default command: run full benchmark suite
CMD ["bash", "scripts/run_all.sh"]
