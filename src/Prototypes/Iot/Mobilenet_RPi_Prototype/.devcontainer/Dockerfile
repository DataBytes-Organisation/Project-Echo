# Multi-stage Dockerfile for MobileNet Bird Detector
# Compatible with Raspberry Pi OS Lite and other ARM64/ARM32 systems

# Use Debian Bullseye as base (same as Raspberry Pi OS Lite)
FROM debian:bullseye-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    # Audio dependencies
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libsndfile1-dev \
    libjack-jackd2-dev \
    alsa-utils \
    pulseaudio \
    sox \
    # Media processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    # Python and development
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # System utilities
    vim \
    nano \
    htop \
    tree \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Create working directory
WORKDIR /workspace

# Create necessary directories
RUN mkdir -p /workspace/recordings \
    && mkdir -p /workspace/logs \
    && mkdir -p /workspace/test_audio \
    && mkdir -p /workspace/Model

# Copy application files
COPY . /workspace/

# Set up git safe directory
RUN git config --global --add safe.directory /workspace || true

# Make Python scripts executable
RUN chmod +x /workspace/*.py

# Set up audio permissions (for when running with audio devices)
RUN usermod -a -G audio root || true

# Expose port for Jupyter if needed
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tensorflow as tf; import librosa; import pyaudio; print('Dependencies OK')" || exit 1

# Default command
CMD ["python3", "mobilenet_bird_detector.py"]
