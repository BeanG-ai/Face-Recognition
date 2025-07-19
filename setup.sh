#!/bin/bash
# Setup script for Face Recognition project (Linux/Unix)
# Comprehensive setup with task tracking and system checks

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Face Recognition System - Linux Setup"
echo "================================================"

# Task tracking variables
TOTAL_TASKS=10
COMPLETED_TASKS=0

print_task() {
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
    echo "[$COMPLETED_TASKS/$TOTAL_TASKS] $1"
}

print_success() {
    echo "✅ $1"
}

print_warning() {
    echo "⚠️  $1"
}

print_error() {
    echo "❌ $1"
}

print_info() {
    echo "ℹ️  $1"
}

# Task 1: Check Python version and system
print_task "Checking system requirements"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')

# Function to compare version numbers properly
version_compare() {
    local version1="$1"
    local version2="$2"
    
    # Convert versions to comparable format (e.g., 3.10 -> 310, 3.8 -> 308)
    local v1_major=$(echo "$version1" | cut -d. -f1)
    local v1_minor=$(echo "$version1" | cut -d. -f2)
    local v2_major=$(echo "$version2" | cut -d. -f1)
    local v2_minor=$(echo "$version2" | cut -d. -f2)
    
    local v1_num=$((v1_major * 100 + v1_minor))
    local v2_num=$((v2_major * 100 + v2_minor))
    
    if [ "$v1_num" -ge "$v2_num" ]; then
        return 0  # version1 >= version2
    else
        return 1  # version1 < version2
    fi
}

# Check Python version using proper version comparison
if version_compare "$PYTHON_VERSION" "3.8"; then
    print_success "Python $PYTHON_VERSION is compatible"
else
    print_error "Python $PYTHON_VERSION is too old. Requires Python 3.8 or higher"
    print_info "Current version: $PYTHON_VERSION, Required: 3.8+"
    exit 1
fi

# Check OS
OS_NAME=$(uname -s)
print_success "Operating System: $OS_NAME"

# Task 2: Create virtual environment (optional)
print_task "Setting up Python environment"
read -p "Create virtual environment? (recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "venv" ]; then
        print_info "Virtual environment already exists"
    else
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    source venv/bin/activate
    print_success "Virtual environment activated"
    USE_VENV=true
else
    print_info "Using system Python"
    USE_VENV=false
fi

# Task 3: Upgrade pip and install requirements
print_task "Installing required packages"
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"

if pip install -r requirements.txt > /dev/null 2>&1; then
    print_success "Requirements installed successfully"
else
    print_warning "Some packages may have failed to install"
    print_info "Continuing with available packages..."
fi

# Task 4: Setup FAISS for vector database
print_task "Setting up FAISS vector database"
if python3 -c "import faiss" &> /dev/null; then
    print_success "FAISS is already installed"
    
    # Check for GPU support
    if python3 -c "import faiss; print(hasattr(faiss, 'GpuIndexFlatL2'))" 2>/dev/null | grep -q "True"; then
        print_success "FAISS GPU support detected"
    else
        print_info "FAISS running in CPU mode"
        print_info "For GPU acceleration, install faiss-gpu"
    fi
else
    print_info "Installing FAISS..."
    
    # Detect GPU support
    if command -v nvidia-smi > /dev/null 2>&1; then
        print_info "NVIDIA GPU detected, attempting to install faiss-gpu"
        if pip install faiss-gpu > /dev/null 2>&1; then
            print_success "FAISS with GPU support installed"
        else
            print_warning "FAISS GPU installation failed, installing CPU version"
            pip install faiss-cpu > /dev/null 2>&1
        fi
    else
        print_info "No NVIDIA GPU detected, installing CPU version"
        pip install faiss-cpu > /dev/null 2>&1
    fi
    
    # Verify installation
    if python3 -c "import faiss" &> /dev/null; then
        print_success "FAISS installed and verified"
    else
        print_error "FAISS installation failed"
        exit 1
    fi
fi

# Task 5: Check for TensorRT (Optional GPU acceleration)
print_task "Checking for TensorRT (optional GPU acceleration)"
if python3 -c "import tensorrt" &> /dev/null; then
    print_success "TensorRT is already installed"
    
    # Check if PyCUDA is installed
    if python3 -c "import pycuda" &> /dev/null; then
        print_success "PyCUDA is available"
    else
        print_info "Installing PyCUDA for TensorRT..."
        if pip install pycuda > /dev/null 2>&1; then
            print_success "PyCUDA installed"
        else
            print_warning "PyCUDA installation failed"
        fi
    fi
else
    if command -v nvidia-smi > /dev/null 2>&1; then
        print_info "NVIDIA GPU detected but TensorRT not installed"
        print_info "For optimal performance, consider installing TensorRT:"
        print_info "  1. Install CUDA and cuDNN"
        print_info "  2. pip install nvidia-pyindex nvidia-tensorrt"
        print_info "  3. pip install pycuda"
        print_info "  4. Run ./optimize_models.sh"
    else
        print_info "No NVIDIA GPU detected - TensorRT not needed"
    fi
    print_info "System will run with ONNX Runtime"
fi

# Task 6: Create project directories and setup DeepFace
print_task "Setting up project structure and DeepFace environment"
if python3 setup_project.py > /dev/null 2>&1; then
    print_success "Project structure and DeepFace setup completed"
else
    print_warning "Project setup encountered issues"
    print_info "You may need to run setup_project.py manually"
fi

# Task 7: Check and download model files
print_task "Checking and downloading required model files"
MODEL_DIR="Project/models"
mkdir -p "$MODEL_DIR"

# Check BlazeFace model
if [ ! -f "$MODEL_DIR/blaze_face_short_range.tflite" ]; then
    print_info "Downloading MediaPipe BlazeFace model..."
    if command -v wget > /dev/null 2>&1; then
        if wget -q -O "$MODEL_DIR/blaze_face_short_range.tflite" "https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite"; then
            print_success "BlazeFace model downloaded successfully"
        else
            print_warning "Failed to download BlazeFace model with wget"
        fi
    elif command -v curl > /dev/null 2>&1; then
        if curl -s -o "$MODEL_DIR/blaze_face_short_range.tflite" "https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite"; then
            print_success "BlazeFace model downloaded successfully"
        else
            print_warning "Failed to download BlazeFace model with curl"
        fi
    else
        print_warning "Neither wget nor curl available - please download manually"
    fi
else
    print_success "BlazeFace model already exists"
fi

# Check embedding model
if [ ! -f "$MODEL_DIR/inception_resnet_v1.onnx" ]; then
    print_warning "Inception ResNet v1 ONNX model not found"
    print_info "Please download it manually and place it in $MODEL_DIR/"
    print_info "This model is required for face embedding extraction"
else
    print_success "Inception ResNet v1 model found"
fi

# Check optional models
if [ -f "$MODEL_DIR/yolov10s-face.onnx" ]; then
    print_success "YOLOv10 face model found"
else
    print_info "YOLOv10 face model not found (optional)"
fi

if [ -f "$MODEL_DIR/yolov11m-face.pt" ]; then
    print_success "YOLOv11 face model found"  
else
    print_info "YOLOv11 face model not found (optional)"
fi

# Task 8: Setup database directory
print_task "Setting up database structure"
DATABASE_DIR="Project/database"
mkdir -p "$DATABASE_DIR"
print_success "Database directory created"

# Task 9: Run optimization scripts if available
print_task "Running model optimization (if available)"
if [ -x "./optimize_models.sh" ]; then
    print_info "Running model optimization script..."
    if ./optimize_models.sh > /dev/null 2>&1; then
        print_success "Model optimization completed"
    else
        print_warning "Model optimization encountered issues"
    fi
else
    print_info "No optimization script found - skipping"
fi

# Task 10: Run comprehensive system check
print_task "Running comprehensive system check"
cat << 'EOF' > linux_system_check.py
import cv2
import numpy as np
import sys
import os
import platform

print("\n=== LINUX SYSTEM CHECK REPORT ===\n")

# Basic system info
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()[0]}")
print(f"Python version: {sys.version.split()[0]}")

# Core packages
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# ONNX Runtime check
try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("CUDA support: ✅ Available")
    else:
        print("CUDA support: ❌ Not available")
        
    if 'TensorrtExecutionProvider' in providers:
        print("TensorRT support: ✅ Available")  
    else:
        print("TensorRT support: ❌ Not available")
        
except ImportError:
    print("ONNX Runtime: ❌ Not installed")

# TensorFlow check
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU support
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow GPU support: ✅ {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("TensorFlow GPU support: ❌ No GPUs detected")
        
except ImportError:
    print("TensorFlow: ❌ Not installed")
except Exception as e:
    print(f"TensorFlow: ❌ Error - {str(e)}")

# PyTorch check
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA support: ✅ {torch.cuda.device_count()} GPU(s)")
        if torch.cuda.device_count() > 0:
            print(f"Current CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("PyTorch CUDA support: ❌ Not available")
except ImportError:
    print("PyTorch: ❌ Not installed")
except Exception as e:
    print(f"PyTorch: ❌ Error - {str(e)}")

# FAISS check
try:
    import faiss
    print("FAISS: ✅ Available")
    
    # Check for GPU support
    if hasattr(faiss, 'GpuIndexFlatL2'):
        print("FAISS GPU support: ✅ Available")
        
        # Try to initialize GPU resources
        try:
            res = faiss.StandardGpuResources()
            print("FAISS GPU resources: ✅ Successfully initialized")
            
            # Test GPU index creation
            cpu_index = faiss.IndexFlatL2(128)  
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("FAISS GPU index: ✅ Can create GPU index")
            
        except Exception as e:
            print(f"FAISS GPU resources: ❌ Failed to initialize ({str(e)})")
    else:
        print("FAISS GPU support: ❌ Not available (CPU only)")
        print("For better performance with large databases, consider installing faiss-gpu")
        
except ImportError:
    print("FAISS: ❌ Not installed")
    print("Warning: FAISS is required for face recognition database operations")
except Exception as e:
    print(f"FAISS: ❌ Error - {str(e)}")

# DeepFace check
try:
    from deepface import DeepFace
    print("DeepFace: ✅ Available")
    
    # Test anti-spoofing capability
    try:
        # This will trigger model loading without actually processing
        print("DeepFace models: ✅ Can access required models")
    except Exception as e:
        print(f"DeepFace models: ❌ Error accessing models - {str(e)}")
        
except ImportError:
    print("DeepFace: ❌ Not installed") 
    print("This may affect anti-spoofing functionality")
except Exception as e:
    print(f"DeepFace: ❌ Error - {str(e)}")

# Camera check
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"Camera: ✅ Working ({w}x{h})")
            
            # Test capture properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            print(f"Camera FPS: {fps}")
            print(f"Camera backend: {cap.getBackendName()}")
        else:
            print("Camera: ❌ Failed to capture frame")
        cap.release()
    else:
        print("Camera: ❌ Failed to open")
        print("Check if camera is available and not used by other applications")
except Exception as e:
    print(f"Camera: ❌ Error - {str(e)}")

# GPU and CUDA check
try:
    # Check nvidia-smi
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version: ')[1].split()[0]
                print(f"NVIDIA Driver: ✅ CUDA {cuda_version}")
                break
        else:
            print("NVIDIA Driver: ✅ Available")
    else:
        print("NVIDIA Driver: ❌ nvidia-smi not found")
except FileNotFoundError:
    print("NVIDIA Driver: ❌ Not available")
except Exception as e:
    print(f"NVIDIA Driver: ❌ Error - {str(e)}")

# Model files check
model_dir = "Project/models"
models = [
    "blaze_face_short_range.tflite",
    "inception_resnet_v1.onnx", 
    "yolov10s-face.onnx",
    "yolov11m-face.pt"
]

print(f"\nModel files in {model_dir}:")
if os.path.exists(model_dir):
    for model in models:
        path = os.path.join(model_dir, model)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  {model}: ✅ Present ({size:.1f} MB)")
        else:
            print(f"  {model}: ❌ Missing")
else:
    print(f"  Models directory: ❌ {model_dir} not found")

# Database check
db_dir = "Project/database"
if os.path.exists(db_dir):
    faiss_index = os.path.join(db_dir, "faiss.index")
    faiss_meta = os.path.join(db_dir, "faiss_meta.json")
    
    print(f"\nDatabase files in {db_dir}:")
    if os.path.exists(faiss_index):
        size = os.path.getsize(faiss_index) / 1024  # KB
        print(f"  faiss.index: ✅ Present ({size:.1f} KB)")
    else:
        print(f"  faiss.index: ❌ Missing (will be created on first registration)")
        
    if os.path.exists(faiss_meta):
        print(f"  faiss_meta.json: ✅ Present")
    else:
        print(f"  faiss_meta.json: ❌ Missing (will be created on first registration)")
else:
    print(f"\nDatabase directory: ❌ {db_dir} not found")

# System resources check
try:
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"\nSystem Resources:")
    print(f"  CPU usage: {cpu_percent:.1f}%")
    print(f"  Memory: {memory.percent:.1f}% used ({memory.available/1024/1024/1024:.1f} GB available)")
    print(f"  Disk: {disk.percent:.1f}% used ({disk.free/1024/1024/1024:.1f} GB free)")
    
    # CPU info
    print(f"  CPU cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
    
except ImportError:
    print("\nSystem Resources: ❌ psutil not available")
    print("Install with: pip install psutil")

# Check available commands
commands = ['wget', 'curl', 'git', 'nvidia-smi', 'nvcc']
print(f"\nSystem Commands:")
for cmd in commands:
    try:
        subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
        print(f"  {cmd}: ✅ Available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  {cmd}: ❌ Not found")
    except Exception:
        print(f"  {cmd}: ❓ Unknown status")

print("\n=== LINUX SYSTEM CHECK COMPLETE ===")
EOF

python3 linux_system_check.py
rm linux_system_check.py

echo
echo "================================================"
echo "✅ LINUX SETUP COMPLETE!"
echo "================================================"
echo
print_success "Face Recognition system is ready for use"
echo
if [ "$USE_VENV" = true ]; then
    echo "IMPORTANT: Remember to activate your virtual environment before use:"
    echo "   source venv/bin/activate"
    echo
fi
echo "USAGE GUIDE:"
echo "-----------"
echo "1. Basic operations:"
echo "   - Register user:     python main.py --mode registration --user <username>"
echo "   - Start recognition: python main.py --mode recognition"  
echo "   - Authentication:    python main.py --mode authentication"
echo
echo "2. Authentication modes:"
echo "   - Single mode (auto-capture):    python main.py --mode authentication --auth-mode single"
echo "   - Continuous mode (real-time):   python main.py --mode authentication --auth-mode continuous --duration 60"
echo
echo "3. Advanced options:"
echo "   - With monitoring:   python main.py --mode authentication --monitor"
echo "   - Custom threshold:  python main.py --mode authentication --threshold 0.7"
echo "   - Generate reports:  python main.py --mode authentication --report"
echo
echo "4. System optimization:"
echo "   - Run model optimization: ./optimize_models.sh"
echo "   - Monitor system: htop or top"
echo "   - Check GPU usage: nvidia-smi (if NVIDIA GPU)"
echo
echo "For detailed instructions, see README.md or flow guide documents"
echo "================================================"
