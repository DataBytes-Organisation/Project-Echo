#!/bin/bash

# MobileNet Bird Detector Launcher Script
# This script provides an easy way to run the bird detector with various options

set -e

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

# Function to print coloured output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3 is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
}

# Check if required files exist
check_files() {
    if [ ! -f "mobilenet_bird_detector.py" ]; then
        print_error "mobilenet_bird_detector.py not found"
        exit 1
    fi
    local TFLITE_MODEL="Model/Model.tflite"
    local CLASS_NAMES="Model/class_names.json"
    if [ ! -f "$TFLITE_MODEL" ]; then
        print_error "Required TFLite model not found: $TFLITE_MODEL"
        exit 1
    fi
    if [ ! -f "$CLASS_NAMES" ]; then
        print_error "Class names file not found: $CLASS_NAMES"
        exit 1
    fi
}

# Install dependencies if needed
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        print_warning "requirements.txt not found (skipping dependency verification)"
        return
    fi
    python3 - <<'EOF'
try:
    import numpy, librosa, pyaudio, soundfile  # noqa
    try:
        import tflite_runtime.interpreter  # type: ignore
    except Exception:
        import tensorflow.lite  # noqa
except Exception:
    raise SystemExit(99)
EOF
    if [ $? -ne 0 ]; then
        print_error "Dependencies missing. Please run: pip install -r requirements.txt"
        return
    fi
}

# Run setup test
run_test() {
    print_status "Running setup test..."
    python3 test_setup.py
}

# Show usage information
show_usage() {
    echo "MobileNet Bird Detector Launcher"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  start, run     Start the bird detector (default)"
    echo "  test           Run setup tests only"
    echo "  install        Install dependencies"
    echo "  audio          Test audio devices"
    echo "  config         Show current configuration"
    # convert        (removed) Model conversion no longer performed here
    echo "  device <N>     Start with specific audio device N"
    echo "  log-all        Start with top-5 prediction logging enabled"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start       # Start detection with default device"
    echo "  $0 device 1    # Start with audio device 1"
    echo "  $0 test        # Test setup"
    echo "  $0 audio       # List audio devices"
    # $0 convert     # (conversion removed)
}

# Test audio devices
test_audio() {
    print_status "Testing audio devices..."
    python3 mobilenet_bird_detector.py --list-devices
}

# Conversion function removed (TFLite model must already exist)

# Show configuration
show_config() {
    if [ -f "config.json" ]; then
        print_status "Current configuration:"
        cat config.json | python3 -m json.tool
    else
        print_warning "config.json not found"
    fi
}

# Main function
main() {
    print_header "MobileNet Bird Detector Launcher"
    print_header "=================================="
    
    # Check basic requirements
    check_python
    
    # Parse command line argument
    case "${1:-start}" in
        "start"|"run")
            check_files
            install_deps
            print_status "Starting MobileNet Bird Detector..."
            print_status "Press Ctrl+C to stop"
            echo ""
            python3 mobilenet_bird_detector.py
            ;;
        "device")
            if [ -z "$2" ]; then
                print_error "Device index required. Use: $0 device <N>"
                echo "Run '$0 audio' to list available devices"
                exit 1
            fi
            check_files
            install_deps
            print_status "Starting MobileNet Bird Detector with device $2..."
            print_status "Press Ctrl+C to stop"
            echo ""
            python3 mobilenet_bird_detector.py --device "$2"
            ;;
        "log-all")
            check_files
            install_deps
            print_status "Starting MobileNet Bird Detector with top-5 logging enabled..."
            print_status "Press Ctrl+C to stop"
            echo ""
            python3 mobilenet_bird_detector.py --log-all
            ;;
        "test")
            run_test
            ;;
        "install")
            install_deps
            ;;
        "audio")
            test_audio
            ;;
        "config")
            show_config
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
