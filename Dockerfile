# =============================================================================
# STAGE 1: Base - Common dependencies
# =============================================================================
FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# =============================================================================
# STAGE 2: Builder - Install Python dependencies
# =============================================================================
FROM base AS builder

# Install pip-tools for dependency management
RUN pip install --no-cache-dir pip-tools

# Copy dependency files
COPY requirements.in requirements.txt ./
COPY pyproject.toml ./

# Compile and install dependencies
RUN pip-compile requirements.in --output-file requirements.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# STAGE 3: Runtime - Final image
# =============================================================================
FROM base AS runtime

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser prompts/ ./prompts/
COPY --chown=appuser:appuser semantic_anchors.json ./
COPY --chown=appuser:appuser start.py ./
COPY --chown=appuser:appuser pyproject.toml ./

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Expose port
EXPOSE 8000

# Entry point
ENTRYPOINT ["python", "start.py"]