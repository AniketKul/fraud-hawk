# GPU Fraud Detection Pipeline
# Base: NVIDIA RAPIDS with CUDA 12 (includes cudf, cuml, xgboost, cupy)
FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install additional Python dependencies (RAPIDS packages come from base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Default command
CMD ["python", "scripts/run_pipeline.py", "--help"]
