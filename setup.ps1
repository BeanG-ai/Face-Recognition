# Setup script for Face Recognition project on Windows

Write-Host "Setting up Face Recognition project..." -ForegroundColor Green

# Install dependencies
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# Install FAISS for vector database
Write-Host "Setting up FAISS vector database..." -ForegroundColor Cyan
try {
    python -c "import faiss" | Out-Null
    Write-Host "FAISS is already installed." -ForegroundColor Green
}
catch {
    Write-Host "Installing FAISS..." -ForegroundColor Yellow
    pip install faiss-cpu
    Write-Host "FAISS installed successfully." -ForegroundColor Green
}

# Check for TensorRT (Optional)
Write-Host "Checking for TensorRT (optional acceleration)..." -ForegroundColor Cyan
try {
    python -c "import tensorrt" | Out-Null
    Write-Host "TensorRT is already installed." -ForegroundColor Green
    
    # Check if PyCUDA is installed
    try {
        python -c "import pycuda" | Out-Null
        Write-Host "PyCUDA is already installed." -ForegroundColor Green
    }
    catch {
        Write-Host "Installing PyCUDA (required for TensorRT)..." -ForegroundColor Yellow
        pip install pycuda
    }
}
catch {
    Write-Host "TensorRT not detected. For GPU acceleration, you can install TensorRT:" -ForegroundColor Yellow
    Write-Host "  1. Install CUDA and cuDNN for your GPU" -ForegroundColor Yellow
    Write-Host "  2. Run: pip install nvidia-pyindex" -ForegroundColor Yellow
    Write-Host "  3. Run: pip install nvidia-tensorrt pycuda" -ForegroundColor Yellow
    Write-Host "  4. Optimize models: .\optimize_models.ps1" -ForegroundColor Yellow
    Write-Host "System will run with ONNX Runtime without TensorRT." -ForegroundColor Yellow
}

# Create project directories and setup DeepFace
Write-Host "Creating project structure and setting up DeepFace..." -ForegroundColor Cyan
python setup_project.py

# Check for model files
Write-Host "Checking for model files..." -ForegroundColor Cyan
if (-not (Test-Path "Project\models\blaze_face_short_range.tflite")) {
    Write-Host "Downloading MediaPipe Blaze Face model..." -ForegroundColor Yellow
    $webClient = New-Object System.Net.WebClient
    New-Item -ItemType Directory -Path "Project\models" -Force | Out-Null
    $webClient.DownloadFile("https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite", "Project\models\blaze_face_short_range.tflite")
}

if (-not (Test-Path "Project\models\inception_resnet_v1.onnx")) {
    Write-Host "Face embedding model not found. Please download it manually and place it in Project\models\ directory." -ForegroundColor Red
    New-Item -ItemType Directory -Path "Project\models" -Force | Out-Null
}

# Setup DeepFace portable models using integrated setup_project.py
Write-Host "Setting up DeepFace portable models..." -ForegroundColor Cyan
python setup_project.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "DeepFace setup completed successfully." -ForegroundColor Green
} else {
    Write-Host "DeepFace setup encountered issues. Check logs above." -ForegroundColor Yellow
}

# Run a quick system check
Write-Host "Running system check..." -ForegroundColor Cyan
$systemCheck = @'
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
'@

$systemCheck | Out-File -FilePath "system_check.py" -Encoding utf8
python system_check.py
Remove-Item "system_check.py"

Write-Host "`nSetup complete! You can now use the system." -ForegroundColor Green
Write-Host "`nUsage:" -ForegroundColor White
Write-Host "  To run the main application: python main.py" -ForegroundColor Cyan
Write-Host "  To register a new user: python main.py --mode registration --user <username>" -ForegroundColor Cyan
Write-Host "  To start face recognition: python main.py --mode recognition" -ForegroundColor Cyan
Write-Host "  For authentication: python main.py --mode authentication" -ForegroundColor Cyan
