# FixWurx Deployment Dockerfile
# Multi-stage build for optimized container size

# Stage 1: Base Python image with dependencies
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Add metadata
LABEL maintainer="FixWurx Team"
LABEL version="1.0.0"
LABEL description="FixWurx Automated Bug Fixing System"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM base AS runtime

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p .triangulum/neural_matrix/patterns \
    .triangulum/neural_matrix/weights \
    .triangulum/neural_matrix/history \
    .triangulum/neural_matrix/connections \
    .triangulum/neural_matrix/test_data

# Initialize neural matrix
RUN python neural_matrix_init.py

# Create non-root user for security
RUN useradd -m fixwurx
RUN chown -R fixwurx:fixwurx /app
USER fixwurx

# Expose ports
# API server
EXPOSE 8000
# Dashboard
EXPOSE 8001

# Set entry point
ENTRYPOINT ["python", "main.py"]

# Default command - can be overridden
CMD ["--mode", "server"]
