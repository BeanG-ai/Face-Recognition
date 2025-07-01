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

# Create project directories
Write-Host "Creating project structure..." -ForegroundColor Cyan
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
