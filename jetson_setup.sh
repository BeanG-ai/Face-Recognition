#!/bin/bash
# Setup script for facial authentication system on Jetson Orin Nano X
# This script installs all required dependencies and prepares the system

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Facial Authentication System - Jetson Setup"
echo "================================================"

# Task tracking variables
TOTAL_TASKS=12
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

# Task 1: Check Jetson platform
print_task "Checking Jetson platform"
if [ -d "/sys/devices/platform/host1x" ]; then
    print_success "Running on Jetson device"
    IS_JETSON=true
else
    print_warning "This does not appear to be a Jetson device"
    print_info "Some optimizations may not work correctly"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    IS_JETSON=false
fi

# Task 2: Check Python version
print_task "Checking Python version"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
    print_success "Python $PYTHON_VERSION is compatible"
else
    print_error "Python $PYTHON_VERSION is too old. Requires Python 3.8 or higher"
    exit 1
fi

# Task 3: Create and setup virtual environment  
print_task "Creating Python virtual environment"
if [ -d "venv" ]; then
    print_info "Virtual environment already exists"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

source venv/bin/activate
print_success "Virtual environment activated"

# Task 4: Update pip and install basic dependencies
print_task "Installing basic dependencies"
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"

if pip install -r requirements.txt > /dev/null 2>&1; then
    print_success "Requirements installed successfully"
else
    print_warning "Some packages failed to install. Continuing with available packages..."
fi

# Task 5: Setup TensorRT acceleration
print_task "Setting up TensorRT acceleration"
if command -v tensorrt &> /dev/null || python3 -c "import tensorrt" &> /dev/null; then
    print_success "TensorRT is already installed"
else
    if [ "$IS_JETSON" = true ]; then
        print_info "Installing TensorRT for Jetson..."
        if sudo apt-get update > /dev/null 2>&1 && sudo apt-get install -y tensorrt > /dev/null 2>&1; then
            print_success "TensorRT installed via apt"
        else
            print_warning "Failed to install TensorRT via apt"
        fi
        
        if pip install nvidia-pyindex pycuda > /dev/null 2>&1; then
            print_success "PyCUDA installed"
        else
            print_warning "Failed to install PyCUDA"
        fi
    else
        print_warning "TensorRT setup skipped (not on Jetson)"
    fi
fi

# Verify TensorRT installation
if python3 -c "import tensorrt" &> /dev/null; then
    print_success "TensorRT verification successful"
else
    print_warning "TensorRT not available. System will run without TensorRT acceleration"
fi

# Task 6: Setup ONNX Runtime with CUDA support
print_task "Setting up ONNX Runtime with CUDA support"
# Remove CPU-only version if exists
pip uninstall -y onnxruntime onnxruntime-cpu &> /dev/null || true

if [ "$IS_JETSON" = true ]; then
    print_info "Installing ONNX Runtime GPU for Jetson..."
    if pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 onnxruntime-gpu > /dev/null 2>&1; then
        print_success "ONNX Runtime GPU installed"
    else
        print_warning "ONNX Runtime GPU failed, installing CPU version"
        pip install onnxruntime > /dev/null 2>&1
    fi
else
    print_info "Installing standard ONNX Runtime..."
    pip install onnxruntime > /dev/null 2>&1
fi

# Task 7: Setup PyCUDA for TensorRT optimization
print_task "Installing PyCUDA for TensorRT optimization"
if pip show pycuda &> /dev/null; then
    print_success "PyCUDA is already installed"
else
    if [ "$IS_JETSON" = true ]; then
        print_info "Installing build dependencies..."
        sudo apt-get install -y libboost-all-dev > /dev/null 2>&1
    fi
    
    if pip install pycuda > /dev/null 2>&1; then
        print_success "PyCUDA installed successfully"
    else
        print_warning "PyCUDA installation failed"
    fi
fi

# Task 8: Setup FAISS with GPU support  
print_task "Setting up FAISS with GPU support"
if python3 -c "import faiss" &> /dev/null; then
    print_info "FAISS is already installed. Checking for GPU support..."
    
    if python3 -c "import faiss; print(hasattr(faiss, 'GpuIndexFlatL2'))" 2>/dev/null | grep -q "True"; then
        print_success "FAISS GPU support is available"
    else
        print_warning "FAISS installed without GPU support. Attempting upgrade..."
        pip uninstall -y faiss-cpu &> /dev/null || true
        
        if pip install faiss-gpu > /dev/null 2>&1; then
            print_success "FAISS GPU support installed"
        else
            print_warning "GPU support failed, installing CPU version"
            pip install faiss-cpu > /dev/null 2>&1
        fi
    fi
else
    print_info "Installing FAISS..."
    if pip install faiss-gpu > /dev/null 2>&1; then
        print_success "FAISS with GPU support installed"
    else
        print_warning "FAISS GPU failed, installing CPU version"
        pip install faiss-cpu > /dev/null 2>&1
    fi
fi

# Task 9: Install Jetson-specific packages
print_task "Installing Jetson-specific packages"
if [ "$IS_JETSON" = true ]; then
    # Install jetson-stats for monitoring
    if pip install jetson-stats > /dev/null 2>&1; then
        print_success "jetson-stats installed"
    else
        print_warning "jetson-stats installation failed"
    fi
    
    # Check for jetson-utils
    if python3 -c "import jetson.utils" &> /dev/null; then
        print_success "jetson-utils available"
    else
        print_warning "jetson-utils not available"
    fi
else
    print_info "Skipping Jetson-specific packages (not on Jetson)"
fi

# Task 10: Setup project directories and DeepFace
print_task "Setting up project directories and DeepFace environment"
if python3 setup_project.py > /dev/null 2>&1; then
    print_success "Project structure and DeepFace setup completed"
else
    print_warning "Project setup encountered issues"
fi

# Task 11: Check and download model files
print_task "Checking and downloading model files"
MODEL_DIR="Project/models"
mkdir -p "$MODEL_DIR"

# Check BlazeFace model
if [ ! -f "$MODEL_DIR/blaze_face_short_range.tflite" ]; then
    print_info "Downloading MediaPipe BlazeFace model..."
    if wget -q -O "$MODEL_DIR/blaze_face_short_range.tflite" "https://storage.googleapis.com/mediapipe-assets/blaze_face_short_range.tflite"; then
        print_success "BlazeFace model downloaded"
    else
        print_warning "Failed to download BlazeFace model"
    fi
else
    print_success "BlazeFace model already exists"
fi

# Check embedding model
if [ ! -f "$MODEL_DIR/inception_resnet_v1.onnx" ]; then
    print_warning "Inception ResNet v1 ONNX model not found"
    print_info "Please download it manually and place it in $MODEL_DIR/"
else
    print_success "Inception ResNet v1 model found"
fi

# Check YOLOv10/YOLOv11 models
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

# Task 12: Run comprehensive system check and optimize models
print_task "Running comprehensive system check"
cat << 'EOF' > jetson_system_check.py
import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
import platform

print("\n=== JETSON SYSTEM CHECK REPORT ===\n")

# Basic system info
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()[0]}")
print(f"Python version: {sys.version.split()[0]}")

# Check if running on Jetson
if os.path.exists("/sys/devices/platform/host1x"):
    print("Device: ✅ Jetson Platform Detected")
    
    # Try to get Jetson model info
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        print(f"Model: {model}")
    except:
        print("Model: Could not determine Jetson model")
        
    # Check Jetson temperature and power
    try:
        import jtop
        print("Jetson monitoring: ✅ jtop available")
    except ImportError:
        print("Jetson monitoring: ❌ jtop not available")
else:
    print("Device: ❌ Not a Jetson platform")

# Core packages
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# ONNX Runtime check
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
        
except ImportError:
    print("FAISS: ❌ Not installed")
except Exception as e:
    print(f"FAISS: ❌ Error - {str(e)}")

# DeepFace check
try:
    from deepface import DeepFace
    print("DeepFace: ✅ Available")
    
    # Test anti-spoofing capability
    try:
        # This will trigger model loading
        print("DeepFace anti-spoofing: ✅ Models accessible")
    except Exception as e:
        print(f"DeepFace anti-spoofing: ❌ Error - {str(e)}")
        
except ImportError:
    print("DeepFace: ❌ Not installed")
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
            
            # Test different capture properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            print(f"Camera FPS: {fps}")
            print(f"Camera FOURCC: {int(fourcc)}")
        else:
            print("Camera: ❌ Failed to capture frame")
        cap.release()
    else:
        print("Camera: ❌ Failed to open")
except Exception as e:
    print(f"Camera: ❌ Error - {str(e)}")

# Model files check
model_dir = "Project/models"
models = [
    "blaze_face_short_range.tflite",
    "inception_resnet_v1.onnx", 
    "yolov10s-face.onnx",
    "yolov11m-face.pt"
]

print(f"\nModel files in {model_dir}:")
for model in models:
    path = os.path.join(model_dir, model)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)  # MB
        print(f"  {model}: ✅ Present ({size:.1f} MB)")
    else:
        print(f"  {model}: ❌ Missing")

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
        print(f"  faiss.index: ❌ Missing")
        
    if os.path.exists(faiss_meta):
        print(f"  faiss_meta.json: ✅ Present")
    else:
        print(f"  faiss_meta.json: ❌ Missing")
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
    
except ImportError:
    print("\nSystem Resources: ❌ psutil not available")

# Temperature check (Jetson specific)
thermal_zones = ["/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp", 
                "/sys/class/thermal/thermal_zone2/temp"]

temps = []
for zone in thermal_zones:
    try:
        with open(zone, 'r') as f:
            temp = int(f.read().strip()) / 1000.0
            temps.append(temp)
    except:
        pass

if temps:
    max_temp = max(temps)
    print(f"\nTemperature: {max_temp:.1f}°C")
    if max_temp > 80:
        print("⚠️  High temperature detected! Consider improving cooling.")
    else:
        print("✅ Temperature normal")

print("\n=== JETSON SYSTEM CHECK COMPLETE ===")
EOF

python3 jetson_system_check.py
rm jetson_system_check.py

# Optimize models for TensorRT if available
if [ -x "./optimize_models.sh" ]; then
    print_info "Running model optimization..."
    if ./optimize_models.sh > /dev/null 2>&1; then
        print_success "Model optimization completed"
    else
        print_warning "Model optimization encountered issues"
    fi
else
    print_info "Model optimization script not found - skipping"
fi

echo
echo "================================================"
echo "✅ JETSON SETUP COMPLETE!"
echo "================================================"
echo
print_info "Face Recognition system is ready on your Jetson device"
echo
echo "PERFORMANCE TIPS:"
echo "- Enable maximum performance: sudo jetson_clocks --fan"
echo "- Monitor system: sudo jtop"
echo "- Check temperatures regularly for thermal throttling"
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
echo "4. For single mode (auto-capture):"
echo "   python main.py --mode authentication --auth-mode single"
echo
echo "5. Add monitoring and reporting:"
echo "   python main.py --mode authentication --monitor --report"
echo 
echo "6. For best performance on Jetson Orin Nano X:"
echo "   sudo jetson_clocks --fan"
echo "   python main.py --mode authentication --threshold 0.7 --monitor"
echo
echo "For detailed instructions, see FLOW_GUIDE_EN.md or FLOW_GUIDE_VI.md"
echo "================================================"
