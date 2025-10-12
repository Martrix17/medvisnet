# Base PyTorch image with CUDA 12.9 (for RTX 2700)
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Set working directory
WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git cmake clang-format wget unzip vim less \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-commit hooks
RUN pip install pre-commit

# Create non-root developer user
RUN useradd -m developer && chown -R developer /workspace
USER developer

# Environment variables
ENV PYTHONPATH=/workspace
