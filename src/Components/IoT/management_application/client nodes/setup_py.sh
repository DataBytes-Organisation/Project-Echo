#!/bin/bash

# Variables
VENV_DIR="$HOME/echo_venv"

echo "=== Starting setup for EchoPi ==="

# Create SSH directory and generate SSH key pair
mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating RSA SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
else
    echo "SSH key already exists, skipping generation."
fi

chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

# Install python3-venv if not installed
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing packages..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

pip install psutil sounddevice requests numpy

echo "Setup complete! Virtual environment located at $VENV_DIR"
echo "To activate it, run: source $VENV_DIR/bin/activate"
