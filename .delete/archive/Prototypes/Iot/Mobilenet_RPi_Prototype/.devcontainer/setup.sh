#!/bin/bash

# Setup script for the development container
# This script installs all necessary dependencies for the MobileNet Bird Detector

set -e

echo "Setting up MobileNet Bird Detector development environment..."

# Configure apt to handle network issues
echo "Configuring package manager..."
export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Add retry logic for network operations
apt_retry() {
    local retries=3
    local delay=5
    local count=0
    
    while [ $count -lt $retries ]; do
        if "$@"; then
            return 0
        else
            count=$((count + 1))
            if [ $count -lt $retries ]; then
                echo "Retrying in ${delay} seconds... (attempt $count/$retries)"
                sleep $delay
                delay=$((delay * 2))
            fi
        fi
    done
    
    echo "Failed after $retries attempts: $*"
    return 1
}

# Update package lists with retry
echo "Updating package lists..."
apt_retry apt-get update -y

# Install system dependencies
echo "Installing system dependencies..."
apt_retry apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libsndfile1-dev \
    python3-dev \
    python3-pip \
    python3-venv

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel --no-warn-script-location

# Install TensorFlow Lite Runtime first (smaller, faster)
echo "Installing TensorFlow Lite Runtime..."
pip3 install tflite-runtime==2.13.0 --no-warn-script-location || echo "TFLite Runtime installation failed, will use TensorFlow"

# Install requirements with retry logic
pip_retry() {
    local retries=3
    local delay=5
    local count=0
    
    while [ $count -lt $retries ]; do
        if pip3 install "$@" --no-warn-script-location; then
            return 0
        else
            count=$((count + 1))
            if [ $count -lt $retries ]; then
                echo "Retrying pip install in ${delay} seconds... (attempt $count/$retries)"
                sleep $delay
            fi
        fi
    done
    
    echo "Pip install failed after $retries attempts: $*"
    return 1
}

if [ -f "/workspace/requirements.txt" ]; then
    echo "Installing Python packages from requirements.txt..."
    pip_retry -r /workspace/requirements.txt
else
    echo "requirements.txt not found, installing basic packages..."
    pip_retry \
        tflite-runtime==2.13.0 \
        tensorflow==2.13.0 \
        librosa==0.10.1 \
        pyaudio==0.2.11 \
        numpy==1.24.3 \
        scipy==1.11.1 \
        matplotlib==3.7.2 \
        soundfile==0.12.1 \
        scikit-learn==1.3.0
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p /workspace/recordings
mkdir -p /workspace/logs
mkdir -p /workspace/test_audio

# Set up git configuration (if not already configured)
echo "Setting up git configuration..."
git config --global --add safe.directory /workspace

# Install additional audio tools for testing (Windows/WSL2 compatible)
echo "Installing basic audio tools..."
apt_retry apt-get install -y \
    alsa-utils \
    sox || echo "Some audio tools failed to install, continuing..."

echo "Note: Audio device access is limited in Windows containers."
echo "   For audio testing, use the host system or a Linux environment."

# Set permissions
echo "Setting up permissions..."
chmod +x /workspace/*.py

# Clean up
echo "Cleaning up..."
apt-get autoremove -y || true
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true

echo "Setup completed successfully!"
echo ""
echo "You can now run the bird detector with:"
echo "   python3 mobilenet_bird_detector.py"
echo ""
echo "If you have an H5 model, convert it to TFLite for better performance:"
echo "   python3 convert_model.py Model/checkpoint_MobileNetV3-Large.hdf5"
echo ""
echo "Recordings will be saved to: /workspace/recordings/"
echo "Model files should be in: /workspace/Model/"
echo ""
echo "To test audio input, run:"
echo "   arecord -l  # List audio devices"
echo "   alsamixer   # Adjust audio levels"
echo ""
