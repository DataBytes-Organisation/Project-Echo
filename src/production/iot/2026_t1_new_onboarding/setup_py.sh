#!/bin/bash
# Exit immediately if any command fails
set -e

# Variables
VENV_DIR="$HOME/echo_venv"
SCRIPT_NAME="client_pi.py"
SERVICE_NAME="client_pi.service"
PI_HOME="/home/pi/Project-Echo/src/Components/IoT/management_application"
SCRIPT_PATH="$PI_HOME/$SCRIPT_NAME"
PYTHON_BIN="python3"

echo "=== Starting setup for EchoPi ==="

# Generate SSH key pair for remote access by the management server
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

# Install python3-venv if not already present
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment if it doesn't exist yet
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

# Activate venv and install required packages
echo "Activating virtual environment and installing packages..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install opencv-python-headless psutil sounddevice requests pyserial gps3 paho-mqtt librosa tflite-runtime

# Pin numpy to avoid compatibility issues with librosa and tflite-runtime
pip install --force-reinstall "numpy==1.26.4"

# Create destination directory if it doesn't exist
mkdir -p "$PI_HOME"

# Copy client script to destination
echo "Copying $SCRIPT_NAME to $PI_HOME..."
cp "$SCRIPT_NAME" "$SCRIPT_PATH"
chmod +x "$SCRIPT_PATH"

# Create systemd service to run client_pi.py on boot
echo "[INFO] Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null << EOL
[Unit]
Description=Client Pi Service
After=network.target sound.target

[Service]
# Use venv Python so all installed packages are available
ExecStart=$VENV_DIR/bin/python3 $SCRIPT_PATH
WorkingDirectory=$PI_HOME
StandardOutput=journal
StandardError=journal
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd and enable the service to start on boot
echo "[INFO] Enabling and starting $SERVICE_NAME..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "=== Setup complete! ==="
echo "Virtual environment located at $VENV_DIR"
echo "To activate it manually, run: source $VENV_DIR/bin/activate"
echo "$SERVICE_NAME will auto-start on boot"