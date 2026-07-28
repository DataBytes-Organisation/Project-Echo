# MobileNet Bird Sound Detector

A real-time bird sound detection system using MobileNet neural networks and TensorFlow Lite, designed to run efficiently on Raspberry Pi devices.

## Overview

This system continuously monitors audio input, processes 1-second chunks into mel spectrograms, and uses a trained MobileNet model to detect bird sounds. When a bird is detected above the confidence threshold, it automatically saves the audio clip with species and confidence information.

## Prerequisites

- Raspberry Pi 4 (recommended) or Raspberry Pi 3B+
- MicroSD card (32GB+ recommended)
- USB microphone or audio input device
- Raspberry Pi OS (latest version)
- Internet connection for initial setup

## Setting Up Your Raspberry Pi

### 1. SSH Connection

#### Connect from your computer:
```bash
# Replace 'pi' with your username and 'raspberry-pi-ip' with actual IP
ssh pi@raspberry-pi-ip

# Example:
ssh pi@192.168.1.100
```

### 2. File Transfer and Setup

#### Create the project directory:
```bash
# On the Raspberry Pi
mkdir -p /home/pi/bird-detector
cd /home/pi/bird-detector
```

#### Copy files using SCP:
```bash
# From your computer, copy the Python files
scp mobilenet_bird_detector.py pi@raspberry-pi-ip:/home/pi/bird-detector/
scp run_detector.sh pi@raspberry-pi-ip:/home/pi/bird-detector/
scp test_accuracy.py pi@raspberry-pi-ip:/home/pi/bird-detector/
scp requirements.txt pi@raspberry-pi-ip:/home/pi/bird-detector/
scp config.json pi@raspberry-pi-ip:/home/pi/bird-detector/

# Copy the config directory
scp -r config/ pi@raspberry-pi-ip:/home/pi/bird-detector/

```

### 3. Model Files Setup

#### Download the TFLite model:
The trained model files are available in the Microsoft Teams group folder.

1. Download the following files from Teams (this should be in the IoT shared folder):
   - `Model.tflite` (the TensorFlow Lite model)
   - `class_names.json` (species classification labels)

2. Transfer to Raspberry Pi:
```bash
# Create Model directory on Pi
ssh pi@raspberry-pi-ip "mkdir -p /home/pi/bird-detector/Model"

# Copy model files
scp Model.tflite pi@raspberry-pi-ip:/home/pi/bird-detector/Model/
scp class_names.json pi@raspberry-pi-ip:/home/pi/bird-detector/Model/
```

### 4. Directory Structure

Create the complete folder structure on your Raspberry Pi:

```bash
# On the Raspberry Pi
cd /home/pi/bird-detector

# Create required directories
mkdir -p Model
mkdir -p recordings
mkdir -p logs
mkdir -p test_audio
mkdir -p spectrograms
mkdir -p config

# Verify structure
tree .
```

Expected directory structure:
```
bird-detector/
├── mobilenet_bird_detector.py    # Main detection script
├── run_detector.sh               # Launcher script
├── test_accuracy.py              # Accuracy testing
├── requirements.txt              # Python dependencies
├── config.json                   # Configuration file
├── config/
│   └── system_config.py          # System configuration
├── Model/
│   ├── Model.tflite              # TensorFlow Lite model
│   └── class_names.json          # Species labels
├── recordings/                   # Saved audio files (created automatically)
├── logs/                         # Log files
├── test_audio/                   # Test audio samples
└── spectrograms/                 # Generated spectrograms
```

## Installation

### 1. System Dependencies

The following commands can be run from your pc using ssh, or in the terminal on the Rasberry Pi

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    ffmpeg \
    libsndfile1-dev \
    git \
    alsa-utils
```

### 2. Python Environment

```bash
# Create virtual environment
cd /home/pi/bird-detector
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Audio Setup

```bash
# List audio devices
arecord -l

# Test microphone (Ctrl+C to stop)
arecord -D plughw:1,0 -f cd test.wav

# Adjust audio levels if needed
alsamixer
```

## Running the Detector

### 1. Basic Usage

```bash
# Make the script executable
chmod +x run_detector.sh

# Run with default settings
./run_detector.sh start

# Run with specific audio device
./run_detector.sh device 1

# Run with top-5 prediction logging (Most Useful)
./run_detector.sh log-all
```

### 2. Available Commands

```bash
# Show all available options
./run_detector.sh help

# Test setup and audio devices
./run_detector.sh test
./run_detector.sh audio

# Show current configuration
./run_detector.sh config
```

### 3. Direct Python Usage

```bash
# Activate virtual environment first
source .venv/bin/activate

# List audio devices
python3 mobilenet_bird_detector.py --list-devices

# Run with specific device
python3 mobilenet_bird_detector.py --device 1

# Run with verbose logging
python3 mobilenet_bird_detector.py --log-all
```

## Running as a Service (Auto-start on boot)

### 1. Create systemd service:

```bash
sudo nano /etc/systemd/system/bird-detector.service
```

Add the following content:

```ini
[Unit]
Description=MobileNet Bird Detector
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/bird-detector
ExecStart=/home/pi/bird-detector/.venv/bin/python /home/pi/bird-detector/mobilenet_bird_detector.py --log-all
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### 2. Enable and start the service:

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable bird-detector.service

# Start the service now
sudo systemctl start bird-detector.service

# Check service status
sudo systemctl status bird-detector.service

# View live logs
sudo journalctl -u bird-detector.service -f
```

## Configuration

### Audio Settings

Edit `config.json` to adjust audio parameters:

```json
{
  "AUDIO_SAMPLE_RATE": 22050,
  "CHUNK_SIZE": 22050,
  "CHUNK_DURATION": 1.0,
  "AUDIO_MELS": 224,
  "AUDIO_NFFT": 2048,
  "DETECTION_THRESHOLD": 0.7
}
```

### Detection Parameters

- `DETECTION_THRESHOLD`: Confidence threshold for saving recordings (0.0-1.0)
- `SAMPLE_RATE`: Audio sample rate in Hz (22050 recommended)
- `CHUNK_SIZE`: Audio chunk size in samples (1 second recommended)

## Troubleshooting

### Audio Issues

```bash
# Check audio devices
arecord -l
lsusb  # For USB microphones

# Test microphone
arecord -D plughw:1,0 -f cd -t wav -d 5 test.wav
aplay test.wav

# Check ALSA configuration
cat /proc/asound/cards
```

### Permission Issues

```bash
# Add user to audio group
sudo usermod -a -G audio pi

# Check audio device permissions
ls -la /dev/snd/
```

### Service Issues

```bash
# Check service logs
sudo journalctl -u bird-detector.service --no-pager

# Restart service
sudo systemctl restart bird-detector.service

# Stop service
sudo systemctl stop bird-detector.service
```

### Model Issues

```bash
# Verify model files exist and are readable
ls -la Model/
file Model/Model.tflite
cat Model/class_names.json | head
```

## Output Files

- **Audio recordings**: Saved to `recordings/` directory
- **Filename format**: `{species}_{confidence}_{timestamp}.wav`
- **Example**: `robin_0.874_20250917_143052.wav`

## Performance Notes

- **CPU Usage**: Moderate on Raspberry Pi 4, higher on Pi 3B+
- **Memory Usage**: ~200-500MB RAM
- **Storage**: Each detection saves ~1-3MB audio file
- **Real-time Processing**: Processes 1-second chunks continuously

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review system logs: `sudo journalctl -u bird-detector.service`
3. Test audio setup: `./run_detector.sh audio`
4. Verify model files are present and readable

## Disclaimer

This project is for educational and research purposes. Please ensure you have appropriate permissions for audio recording in your location.