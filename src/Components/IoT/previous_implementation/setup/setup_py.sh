#!/bin/bash
set -e

# Variables
VENV_DIR="$HOME/echo_venv"

SCRIPT_NAME="client_pi.py"
SERVICE_NAME="client_pi.service"
PI_HOME="/home/pi/Project-Echo/src/Components/IoT/management_application"
SCRIPT_PATH="$PI_HOME/$SCRIPT_NAME"
PYTHON_BIN="python3"

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

chmod 600 ~/.ssh/id_rsa.ssh/id_rsa -N ""
else
    echo "SSH key already exists, skipping generation."
fi

chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

# Install python3-venv if not installed
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt-get update
    sudo apt-get install -y python3
fi

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists, skipping creation."
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing packages..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install opencv-python-headless psutil sounddevice requests pyserial gps3 flask paho-mqtt librosa tflite-runtime
pip install --force-reinstall "numpy==1.26.4"

# Copy client script to home
echo "Copying $SCRIPT_NAME to $HOME..."
cp $SCRIPT_NAME $SCRIPT_PATH
chmod +x $SCRIPT_NAME

# Create systemd service
echo "[INFO] Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null << EOL
[Unit]
Description=Client Pi Service
After=network.target sound.target

[Services]
ExecStart=/usr/bin/python3 $SCRIPT_PATH
WorkingDirectory=$PI_HOME
StandardOutput=journal
StandardError=journal
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOL

# Enable and start service
echo "[INFO] Enabling and Starting $SERVICE_NAME..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Setup complete! Virtual environment located at $VENV_DIR"
echo "To activate it, run: source $VENV_DIR/bin/activate"
echo "$SERVICE_NAME will auto-start on boot"
 
