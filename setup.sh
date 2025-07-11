#!/bin/bash
# Setup script for Face Recognition project
echo "Setting up Face Recognition project..."

# Install dependencies
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install FAISS for vector database
echo "Setting up FAISS vector database..."
if python -c "import faiss" &> /dev/null; then
    echo "FAISS is already installed."
else
    echo "Installing FAISS..."
    pip install faiss-cpu
    
    # Verify installation
    if python -c "import faiss" &> /dev/null; then
        echo "FAISS installed successfully."
    else
        echo "Failed to install FAISS. Please install it manually:"
        echo "  pip install faiss-cpu"
        echo "Or for systems with NVIDIA GPU:"
        echo "  pip install faiss-gpu"
    fi
fi

# Check for TensorRT (Optional)
echo "Checking for TensorRT (optional acceleration)..."
if python -c "import tensorrt" &> /dev/null; then
    echo "TensorRT is already installed."
    
    # Check if PyCUDA is installed
    if python -c "import pycuda" &> /dev/null; then
        echo "PyCUDA is already installed."
    else
        echo "Installing PyCUDA (required for TensorRT)..."
        pip install pycuda
    fi
else
    echo "TensorRT not detected. For GPU acceleration, you can install TensorRT:"
    echo "  1. Install CUDA and cuDNN for your GPU"
    
    if [ -f "/etc/nv_tegra_release" ]; then
        # This is a Jetson device
        echo "  2. On Jetson: sudo apt-get install -y tensorrt"
        echo "  3. Run: pip install nvidia-pyindex pycuda"
        echo "  4. Optimize models: ./optimize_models.sh"
    else
        # Regular Linux
        echo "  2. Run: pip install nvidia-pyindex"
        echo "  3. Run: pip install nvidia-tensorrt pycuda"
        echo "  4. Optimize models: ./optimize_models.sh"
    fi
    
    echo "System will run with ONNX Runtime without TensorRT."
fi

# Create project directories
echo "Creating project structure..."
python setup_project.py

# Check for model files
echo "Checking for model files..."
if [ ! -f "Project/models/blaze_face_short_range.tflite" ]; then
    echo "Downloading MediaPipe Blaze Face model..."
    mkdir -p Project/models
    wget -O Project/models/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite
fi

if [ ! -f "Project/models/inception_resnet_v1.onnx" ]; then
    echo "Face embedding model not found. Please download it manually and place it in Project/models/ directory."
    mkdir -p Project/models
fi

# Run a quick system check
echo "Running system check..."
cat << EOF > system_check.py
import cv2
import numpy as np
import sys

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Check FAISS
try:
    import faiss
    print("FAISS version: Available")
except ImportError:
    print("FAISS: Not installed")
    print("Warning: FAISS is required for face recognition. Please install it.")

# Check TensorFlow
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow: Not installed")

# Check camera
try:
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
except Exception as e:
    print(f"Camera check: Error - {str(e)}")

print("System check complete")
EOF

python system_check.py
rm system_check.py

# Setup DeepFace portable models
echo "Setting up DeepFace portable models..."
python setup_deepface.py
if [ $? -eq 0 ]; then
    echo "DeepFace setup completed successfully."
else
    echo "DeepFace setup encountered issues. Check logs above."
fi

echo
echo "Setup complete! You can now use the system."
echo "To run the main application: python main.py"
echo "To register a new user: python main.py --mode registration --user <username>"
echo "To start face recognition: python main.py --mode recognition"
echo "For authentication: python main.py --mode authentication"
