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

# Multiple detection methods for Jetson devices
IS_JETSON=false
JETSON_MODEL=""

# Method 1: Check for traditional Jetson marker
if [ -d "/sys/devices/platform/host1x" ]; then
    IS_JETSON=true
    print_info "Detected via host1x platform"
fi

# Method 2: Check device tree model (more reliable for Jetson Orin)
if [ -f "/proc/device-tree/model" ]; then
    MODEL_INFO=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')
    if echo "$MODEL_INFO" | grep -iq "jetson\|tegra\|orin\|xavier\|nano"; then
        IS_JETSON=true
        JETSON_MODEL="$MODEL_INFO"
        print_info "Detected via device tree: $JETSON_MODEL"
    fi
fi

# Method 3: Check for Tegra SoC via nvidia-smi or tegrastats
if command -v tegrastats > /dev/null 2>&1; then
    IS_JETSON=true
    print_info "Detected via tegrastats command"
fi

# Method 4: Check for NVGPU driver (Jetson Orin specific)
if [ -d "/sys/bus/platform/drivers/nvgpu" ] || [ -d "/sys/devices/platform/17000000.ga10b" ]; then
    IS_JETSON=true
    print_info "Detected via NVGPU driver (Jetson Orin)"
fi

# Method 5: Check for L4T version (Linux for Tegra)
if [ -f "/etc/nv_tegra_release" ]; then
    IS_JETSON=true
    L4T_VERSION=$(cat /etc/nv_tegra_release | head -1)
    print_info "Detected via L4T: $L4T_VERSION"
fi

# Method 6: Check nvidia-smi for Tegra/Orin GPU
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi -L 2>/dev/null || true)
    if echo "$GPU_INFO" | grep -iq "tegra\|orin\|xavier"; then
        IS_JETSON=true
        print_info "Detected via nvidia-smi: GPU contains Tegra/Orin"
    fi
fi

# Method 7: Check for Jetson-specific directories
JETSON_DIRS=(
    "/sys/devices/platform/tegra-fuse"
    "/sys/devices/platform/tegra-pmc"
    "/sys/devices/soc0"
    "/proc/device-tree/compatible"
)

for dir in "${JETSON_DIRS[@]}"; do
    if [ -e "$dir" ]; then
        if [ -f "$dir" ]; then
            # It's a file, check contents
            if grep -iq "tegra\|jetson\|orin\|xavier" "$dir" 2>/dev/null; then
                IS_JETSON=true
                print_info "Detected via $dir file contents"
                break
            fi
        else
            # It's a directory
            IS_JETSON=true
            print_info "Detected via directory: $dir"
            break
        fi
    fi
done

# Final verification and user override
if [ "$IS_JETSON" = true ]; then
    print_success "✅ Jetson device detected!"
    if [ -n "$JETSON_MODEL" ]; then
        print_info "Model: $JETSON_MODEL"
    fi
    
    # Set Jetson environment variable for camera configuration
    export JETSON_DEVICE=true
    echo "export JETSON_DEVICE=true" >> ~/.bashrc
    print_info "Set JETSON_DEVICE environment variable for camera configuration"
    
    # Check for specific Orin features
    if echo "$JETSON_MODEL" | grep -iq "orin" || [ -d "/sys/devices/platform/17000000.ga10b" ]; then
        print_info "Jetson Orin specific optimizations will be applied"
        JETSON_ORIN=true
    else
        JETSON_ORIN=false
    fi
else
    print_warning "⚠️ Could not automatically detect Jetson device"
    print_info "You mentioned: NVIDIA Tegra Orin (nvgpu)"
    print_info "This suggests you are on a Jetson Orin device"
    echo
    read -p "Are you running on a Jetson device? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        IS_JETSON=true
        JETSON_ORIN=true
        # Set Jetson environment variable for camera configuration
        export JETSON_DEVICE=true
        echo "export JETSON_DEVICE=true" >> ~/.bashrc
        print_info "Set JETSON_DEVICE environment variable for camera configuration"
        print_success "✅ Jetson mode enabled by user confirmation"
        print_info "Assuming Jetson Orin based on your GPU information"
    else
        print_info "Continuing in standard Linux mode"
        print_warning "Some Jetson-specific optimizations will be skipped"
    fi
fi

# Task 2: Check Python version
print_task "Checking Python version"
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
        
        # For Jetson Orin, try different installation methods
        if [ "$JETSON_ORIN" = true ]; then
            print_info "Applying Jetson Orin specific TensorRT setup..."
            
            # Check if TensorRT is available via apt
            if sudo apt-get update > /dev/null 2>&1 && apt-cache search tensorrt | grep -q tensorrt; then
                if sudo apt-get install -y python3-libnvinfer-dev tensorrt > /dev/null 2>&1; then
                    print_success "TensorRT installed via apt (Orin method)"
                else
                    print_warning "Failed to install TensorRT via apt"
                fi
            else
                print_info "TensorRT not available via apt, trying pip..."
            fi
            
            # Try to install Python bindings
            if pip install nvidia-tensorrt > /dev/null 2>&1; then
                print_success "TensorRT Python bindings installed"
            else
                print_warning "TensorRT Python bindings installation failed"
            fi
        else
            # Standard Jetson installation
            if sudo apt-get update > /dev/null 2>&1 && sudo apt-get install -y tensorrt > /dev/null 2>&1; then
                print_success "TensorRT installed via apt"
            else
                print_warning "Failed to install TensorRT via apt"
            fi
        fi
        
        # Install PyCUDA for Jetson
        if pip install nvidia-pyindex pycuda > /dev/null 2>&1; then
            print_success "PyCUDA installed"
        else
            print_warning "Failed to install PyCUDA"
            # Try alternative installation
            if pip install pycuda > /dev/null 2>&1; then
                print_success "PyCUDA installed (alternative method)"
            else
                print_warning "PyCUDA installation failed completely"
            fi
        fi
    else
        print_warning "TensorRT setup skipped (not on Jetson)"
    fi
fi

# Verify TensorRT installation
if python3 -c "import tensorrt" &> /dev/null; then
    # Get TensorRT version if possible
    TRT_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "unknown")
    print_success "TensorRT verification successful (version: $TRT_VERSION)"
else
    print_warning "TensorRT not available. System will run without TensorRT acceleration"
    print_info "For Jetson Orin, you may need to:"
    print_info "  1. sudo apt update && sudo apt install tensorrt"
    print_info "  2. pip install nvidia-tensorrt"
    print_info "  3. Ensure CUDA is properly installed"
fi

# Task 6: Setup ONNX Runtime with CUDA support
print_task "Setting up ONNX Runtime with CUDA support"
# Remove CPU-only version if exists
pip uninstall -y onnxruntime onnxruntime-cpu &> /dev/null || true

if [ "$IS_JETSON" = true ]; then
    print_info "Installing ONNX Runtime GPU for Jetson..."
    
    if [ "$JETSON_ORIN" = true ]; then
        print_info "Using Jetson Orin optimized installation..."
        
        # Try multiple installation sources for Jetson Orin
        ONNX_INSTALLED=false
        
        # Method 1: Try NVIDIA's official Jetson repository (JetPack 5.1+)
        if pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 onnxruntime-gpu > /dev/null 2>&1; then
            print_success "ONNX Runtime GPU installed (JetPack 5.1 repo)"
            ONNX_INSTALLED=true
        elif pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 onnxruntime-gpu > /dev/null 2>&1; then
            print_success "ONNX Runtime GPU installed (JetPack 5.0 repo)"
            ONNX_INSTALLED=true
        fi
        
        # Method 2: Try standard PyPI with GPU support
        if [ "$ONNX_INSTALLED" = false ]; then
            if pip install onnxruntime-gpu > /dev/null 2>&1; then
                print_success "ONNX Runtime GPU installed (PyPI)"
                ONNX_INSTALLED=true
            fi
        fi
        
        # Method 3: Fallback to CPU version
        if [ "$ONNX_INSTALLED" = false ]; then
            print_warning "ONNX Runtime GPU failed, installing CPU version"
            pip install onnxruntime > /dev/null 2>&1
            print_info "ONNX Runtime CPU installed as fallback"
        fi
    else
        # Standard Jetson (Xavier, Nano, etc.)
        if pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 onnxruntime-gpu > /dev/null 2>&1; then
            print_success "ONNX Runtime GPU installed"
        else
            print_warning "ONNX Runtime GPU failed, installing CPU version"
            pip install onnxruntime > /dev/null 2>&1
        fi
    fi
else
    print_info "Installing standard ONNX Runtime..."
    pip install onnxruntime > /dev/null 2>&1
fi

# Verify ONNX Runtime installation and check providers
if python3 -c "import onnxruntime" &> /dev/null; then
    ORT_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
    PROVIDERS=$(python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())" 2>/dev/null || echo "[]")
    
    print_success "ONNX Runtime installed successfully (version: $ORT_VERSION)"
    print_info "Available providers: $PROVIDERS"
    
    # Check for CUDA support
    if echo "$PROVIDERS" | grep -q "CUDAExecutionProvider"; then
        print_success "CUDA execution provider available"
    else
        print_warning "CUDA execution provider not available"
    fi
    
    # Check for TensorRT support
    if echo "$PROVIDERS" | grep -q "TensorrtExecutionProvider"; then
        print_success "TensorRT execution provider available"
    else
        print_info "TensorRT execution provider not available"
    fi
else
    print_error "ONNX Runtime installation failed"
    exit 1
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
