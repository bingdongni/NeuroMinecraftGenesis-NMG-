# NeuroMinecraft Genesis Dockerfile
# Multi-stage build for production and development

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================
# Development Stage
# ============================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pip==23.3.1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the project in development mode
RUN pip install -e .

# Expose ports
EXPOSE 8501  # Streamlit
EXPOSE 8888  # Jupyter
EXPOSE 8080  # API Server

# Default command for development
CMD ["python", "-m", "streamlit", "run", "utils/visualization/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================
# Production Stage
# ============================================
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir pip==23.3.1

# Copy requirements and install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --no-warn-script-location -r requirements.txt

# Copy only necessary project files
COPY core/ /app/core/
COPY agents/ /app/agents/
COPY worlds/ /app/worlds/
COPY utils/ /app/utils/
COPY config/ /app/config/
COPY __init__.py /app/
COPY setup.py /app/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

# Production command
CMD ["python", "-m", "streamlit", "run", "utils/visualization/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ============================================
# Testing Stage
# ============================================
FROM development as testing

# Copy test files
COPY tests/ /app/tests/

# Run tests
CMD ["pytest", "-v", "/app/tests/", "--tb=short"]

# ============================================
# GPU Support (CUDA)
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "utils/visualization/dashboard.py", "--server.port=8501"]
