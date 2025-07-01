#!/bin/bash
# Optimize face recognition models with TensorRT in FP16 precision

echo "===================================================="
echo "Face Recognition Model Optimizer for TensorRT"
echo "===================================================="

# Check if running on a Jetson device
if [ -d "/sys/devices/platform/host1x" ]; then
    echo "Jetson device detected. Setting maximum performance..."
    if command -v jetson_clocks &> /dev/null; then
        sudo jetson_clocks --fan
    else
        echo "Warning: jetson_clocks not found. Install jetson-stats for better performance."
    fi
fi

# Make sure we have required packages
echo "Checking dependencies..."
pip install --quiet nvidia-pyindex onnx

# Check if TensorRT is installed
if ! python -c "import tensorrt" &> /dev/null; then
    echo "TensorRT not found. Installing required packages..."
    if [ -d "/sys/devices/platform/host1x" ]; then
        # Jetson device
        sudo apt-get update
        sudo apt-get install -y tensorrt
        pip install --quiet pycuda
    else
        # Regular NVIDIA GPU system
        pip install --quiet nvidia-tensorrt pycuda
    fi
fi

# Optimize with FP16 precision (faster and works on most hardware)
echo
echo "===================================================="
echo "Optimizing models with FP16 precision"
echo "===================================================="
python optimize_face_models.py --precision fp16 --benchmark

echo
echo "===================================================="
echo "Optimization complete!"
echo "===================================================="
echo
echo "The following optimized models are now available:"
echo "  FP16 models:"
echo "    - Project/models/blazeface_fp16.trt"
echo "    - Project/models/inception_resnet_v1_fp16.trt"
echo
echo "To use these optimized models in your application, the code"
echo "will automatically use TensorRT for acceleration if available."
echo
echo "For higher accuracy but slower inference, you can also optimize with FP32:"
echo "    python optimize_face_models.py --precision fp32"
echo "===================================================="
