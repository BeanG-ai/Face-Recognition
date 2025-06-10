# Setup script for Face Recognition project on Windows

Write-Host "Setting up Face Recognition project..." -ForegroundColor Green

# Install dependencies
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install -r requirements.txt

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

Write-Host "Setup complete! You can now use the system." -ForegroundColor Green
Write-Host "To register a new user: python face_recognition_cli.py register <username>" -ForegroundColor Cyan
Write-Host "To start face recognition: python face_recognition_cli.py recognize" -ForegroundColor Cyan
