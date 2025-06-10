#!/bin/bash
# Setup script for Face Recognition project on Jetson Orin Nano X

echo "Setting up Face Recognition project..."

# Install dependencies
echo "Installing required packages..."
pip install -r requirements.txt

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

echo "Setup complete! You can now use the system."
echo "To register a new user: python face_recognition_cli.py register <username>"
echo "To start face recognition: python face_recognition_cli.py recognize"
