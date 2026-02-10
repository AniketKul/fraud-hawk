# GPU Fraud Detection Pipeline
# Base: NVIDIA RAPIDS with CUDA 12
FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies and fix nvtx compatibility with newer cudf
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'nvtx>=0.2.11'

# Copy source code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Default command
CMD ["python", "scripts/run_pipeline.py", "--help"]
