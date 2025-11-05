# Base PyTorch image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# System dependencies
RUN apt-get update && apt-get install -y \
    git cmake clang-format wget unzip vim less \
    build-essential pkg-config \
    libopencv-dev \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime CPU
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.20.1.tgz \
    && mv onnxruntime-linux-x64-1.20.1 /opt/onnxruntime \
    && rm onnxruntime-linux-x64-1.20.1.tgz

# Set ONNX Runtime environment variables
ENV ONNXRUNTIME_DIR=/opt/onnxruntime
ENV LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH

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
