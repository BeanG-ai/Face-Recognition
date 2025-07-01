#!/bin/bash
# Setup script for facial authentication system on Jetson Orin Nano X
# This script installs all required dependencies and prepares thprint("System check complete")

print("\n=== HARDWARE CHECK ===\n")
# Check Jetson-specific hardware
try:
    import jetson.utils
    print("Jetson Utils: Available")
    try:
        import jtop
        print("Jetson Stats: Available")
        print("\nYou can monitor your Jetson device with: jtop")
    except ImportError:
        print("Jetson Stats: Not installed")
except ImportError:
    print("Jetson-specific packages: Not available")

print("\n=== SYSTEM CHECK COMPLETE ===\n")
EOF-e  # Exit on error

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

# Install FAISS with GPU support for Jetson
echo "Setting up FAISS with GPU support..."
if python -c "import faiss" &> /dev/null; then
    echo "FAISS is already installed. Checking for GPU support..."
    
    # Check if GPU support is available
    if python -c "import faiss; print(hasattr(faiss, 'GpuIndexFlatL2'))" 2>/dev/null | grep -q "True"; then
        echo "FAISS GPU support is available."
    else
        echo "FAISS is installed but without GPU support. Attempting to upgrade..."
        pip uninstall -y faiss-cpu
        
        # Try installing faiss-gpu
        if pip install faiss-gpu; then
            echo "Successfully installed FAISS with GPU support."
        else
            echo "Failed to install FAISS with GPU support. Installing CPU version as fallback..."
            pip install faiss-cpu
            echo "For optimal performance on Jetson, please install FAISS with GPU support manually."
        fi
    fi
else
    echo "Installing FAISS..."
    # Try GPU version first
    if pip install faiss-gpu; then
        echo "Successfully installed FAISS with GPU support."
    else
        echo "Failed to install FAISS with GPU support. Installing CPU version as fallback..."
        pip install faiss-cpu
        echo "For optimal performance on Jetson, please install FAISS with GPU support manually."
    fi
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
import sys

print("\n=== SYSTEM CHECK REPORT ===\n")

print("Python version:", sys.version.split()[0])
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("ONNX Runtime version:", ort.__version__)
print("ONNX Runtime providers:", ort.get_available_providers())

# Check TensorFlow
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow: Not installed")

# Check FAISS
try:
    import faiss
    print("FAISS version: Available")
    # Check for GPU support
    if hasattr(faiss, 'GpuIndexFlatL2'):
        print("FAISS GPU support: Available")
        # Try to create a GPU resource to verify it's working
        try:
            res = faiss.StandardGpuResources()
            print("FAISS GPU resources: Successfully initialized")
        except Exception as e:
            print(f"FAISS GPU resources: Failed to initialize ({str(e)})")
    else:
        print("FAISS GPU support: Not available (CPU only)")
        print("For optimal performance on Jetson, consider installing FAISS with GPU support.")
except ImportError:
    print("FAISS: Not installed")
    print("Warning: FAISS is required for face recognition. Please install it.")

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
echo "The Face Recognition system is now set up on your Jetson device."
echo
echo "USAGE GUIDE:"
echo "-----------"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo
echo "2. Run the application in different modes:"
echo "   - Registration:  python main.py --mode registration --user <username>"
echo "   - Recognition:   python main.py --mode recognition"
echo "   - Authentication: python main.py --mode authentication"
echo
echo "3. For continuous authentication mode:"
echo "   python main.py --mode authentication --auth-mode continuous --duration 60"
echo
echo "4. Add monitoring and reporting:"
echo "   python main.py --mode authentication --monitor --report"
echo 
echo "5. For best performance on Jetson Orin Nano X:"
echo "   sudo jetson_clocks --fan"
echo "   python main.py --mode authentication --threshold 0.7 --monitor"
echo
echo "For detailed instructions, see FLOW_GUIDE_EN.md or FLOW_GUIDE_VI.md"
echo "================================================"
