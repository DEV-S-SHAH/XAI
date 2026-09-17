# ==============================================================================
# Dockerfile for XAI for IOT Anomaly Detection Platform
# Multi-stage production container supporting FastAPI, Streamlit, and Flower FL
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir onnx onnxscript && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source tree
COPY . /app

# Ensure output directories exist
RUN mkdir -p /app/models_saved/checkpoints /app/models_saved/onnx /app/models_saved/quantized \
             /app/paper_assets/tables /app/paper_assets/figures \
             /app/data/synthetic /app/data/real

# Expose API (8000), Streamlit Dashboard (8501), and Flower Server (8080)
EXPOSE 8000 8501 8080

# Default entrypoint starts the Split-Brain FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
