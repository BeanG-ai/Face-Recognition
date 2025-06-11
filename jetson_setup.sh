#!/bin/bash
# Setup script for facial authentication system on Jetson Orin Nano X
# This script installs all required dependencies and prepares the system

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Facial Authentication System - Jetson Setup"
echo "================================================"

# Check if running on Jetson
if [ ! -d "/sys/devices/platform/host1x" ]; then
    echo "WARNING: This does not appear to be a Jetson device."
    echo "         Some optimizations may not work correctly."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Uncomment Jetson-specific packages in requirements.txt
sed -i 's/# jetson-stats/jetson-stats/' requirements.txt
pip install jetson-stats

# Setup ONNX Runtime with CUDA support for Jetson
echo "Setting up optimized ONNX Runtime..."
# Remove CPU-only version
pip uninstall -y onnxruntime
# Install CUDA version if available from NVIDIA repos
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 onnxruntime-gpu

# Install TensorRT if not already installed
if ! pip show tensorrt &> /dev/null; then
    echo "Installing TensorRT..."
    sudo apt-get update
    sudo apt-get install -y tensorrt
    # Link TensorRT to the Python environment
    python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__} installed successfully')"
fi

# Create necessary directories
echo "Setting up project directories..."
python setup_project.py

# Run system check
echo "Running system check..."
cat << EOF > system_check.py
import cv2
import numpy as np
import onnxruntime as ort

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("ONNX Runtime version:", ort.__version__)
print("ONNX Runtime providers:", ort.get_available_providers())

# Check camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Camera check: OK")
        h, w = frame.shape[:2]
        print(f"Camera resolution: {w}x{h}")
    else:
        print("Camera check: Failed to read frame")
    cap.release()
else:
    print("Camera check: Failed to open camera")

# Check GPU
try:
    import jetson.utils
    print("Jetson Utils version: Available")
    # Check Jetson temperature
    import jtop
    print("Jetson temperature monitoring: Available")
except ImportError:
    print("Jetson-specific packages: Not available")

print("System check complete")
EOF

python system_check.py
rm system_check.py

echo
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo
echo "To run the facial authentication demo:"
echo "  source venv/bin/activate"
echo "  python main.py --mode authentication --monitor --report"
echo
echo "For continuous authentication mode:"
echo "  python main.py --mode authentication --auth-mode continuous -duration 60 --monitor --report"
echo 
echo "For best performance on Jetson Orin Nano X:"
echo "  sudo jetson_clocks --fan"
echo "  python main.py --mode authentication --threshold 0.7 --monitor"
