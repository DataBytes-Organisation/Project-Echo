#!/bin/bash
# filepath: .devcontainer/setup.sh

set -e

echo "Setting up Project Echo development environment..."

# Set TensorFlow environment variable to disable oneDNN optimizations
export TF_ENABLE_ONEDNN_OPTS=0

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip

# Downgrade NumPy to be compatible with TensorFlow 2.10.0
echo "Installing compatible NumPy version..."
pip3 install --upgrade pip
pip3 install "numpy<2.0"

# Install other Python dependencies
pip3 install \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyter \
    jupyterlab \
    tensorboard \
    black \
    flake8 \
    isort

# Verify TensorFlow installation
echo "Verifying TensorFlow installation..."
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Create necessary directories
mkdir -p /workspace/data
mkdir -p /workspace/logs
mkdir -p /workspace/models
mkdir -p /tmp/cache

# Set permissions
chmod -R 755 /workspace

echo "Setup completed successfully!"