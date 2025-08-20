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

echo "Installing additional Python dependencies..."
pip3 install --upgrade pip

# Install TensorFlow 2.10.0 with compatible NumPy first
echo "Installing TensorFlow 2.10.0 and compatible NumPy..."
pip3 install "numpy>=1.19.2,<1.24.0"
pip3 install "tensorflow==2.10.0"

# Verify TensorFlow installation
echo "Verifying TensorFlow installation..."
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Install other packages with --no-deps to prevent conflicts
echo "Installing other packages..."
pip3 install --no-deps \
    "pandas==1.5.3" \
    "matplotlib==3.6.3" \
    "seaborn==0.12.2" \
    "scikit-learn==1.1.3" \
    "black==22.12.0" \
    "flake8==6.0.0" \
    "isort==5.12.0" \
    "tensorflow-hub==0.12.0" \
    "librosa==0.9.2" \
    "soundfile==0.12.1" \
    "audiomentations==0.42.0" \
    "diskcache==5.4.0" \
    "ipywidgets==8.0.4" \
    "widgetsnbextension==4.0.5" \
    "jupyter-widgets==1.1.1" \
    "tensorboard==2.10.1"

# Install missing dependencies for the packages above
echo "Installing missing dependencies..."
pip3 install \
    "python-dateutil==2.8.2" \
    "pytz==2022.7.1" \
    "six==1.16.0" \
    "pyparsing==3.0.9" \
    "cycler==0.11.0" \
    "fonttools==4.38.0" \
    "kiwisolver==1.4.4" \
    "pillow==9.4.0" \
    "contourpy==1.0.7" \
    "click>=8.0.0" \
    "mypy-extensions>=0.4.3" \
    "pathspec>=0.9.0" \
    "tomli>=1.1.0" \
    "typing-extensions>=4.0.1" \
    "resampy>=0.2.2" \
    "numpy-minmax>=0.3.0" \
    "numpy-rms>=0.4.2" \
    "python-stretch>=0.3.1" \
    "joblib>=1.0.0,<1.3.0" \
    "threadpoolctl>=2.0.0,<4.0.0" \
    "scipy>=1.7.0,<1.10.0" \
    "numba>=0.56.0,<0.58.0" \
    "cffi>=1.15.0" \
    "decorator>=4.4.0" \
    "audioread>=2.1.9" \
    "pooch>=1.6.0" \
    "lazy-loader>=0.1" \
    "soxr>=0.3.2" \
    "tzdata>=2022.1"

# Final verification
echo "Final verification..."
python3 -c "
import sys
print('Python version:', sys.version)

try:
    import numpy as np
    print('NumPy version:', np.__version__)
except ImportError as e:
    print('NumPy import error:', e)

try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    print('GPU available:', tf.config.list_physical_devices('GPU'))
    print('Built with CUDA:', tf.test.is_built_with_cuda())
except ImportError as e:
    print('TensorFlow import error:', e)

try:
    import ipywidgets
    print('ipywidgets version:', ipywidgets.__version__)
    from IPython.display import display
    print('IPython.display imported successfully')
except ImportError as e:
    print('Widgets import error:', e)

try:
    import tensorboard
    print('TensorBoard version:', tensorboard.__version__)
except ImportError as e:
    print('TensorBoard import error:', e)
"

# Create necessary directories
mkdir -p /workspace/data
mkdir -p /workspace/logs
mkdir -p /workspace/models
mkdir -p /tmp/cache

# Enable Jupyter widgets extensions
echo "Enabling Jupyter widget extensions..."
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter nbextension install --py widgetsnbextension --sys-prefix

# Set permissions
chmod -R 755 /workspace

echo "Setup completed successfully!"