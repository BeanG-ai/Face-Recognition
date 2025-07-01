# PowerShell script to optimize face recognition models with TensorRT
# in FP16 precision

Write-Host "====================================================" -ForegroundColor Green
Write-Host "Face Recognition Model Optimizer for TensorRT" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

# Check if TensorRT is installed
try {
    $null = python -c "import tensorrt"
    Write-Host "TensorRT is already installed." -ForegroundColor Green
}
catch {
    Write-Host "TensorRT not found. Installing required packages..." -ForegroundColor Yellow
    pip install --quiet nvidia-pyindex
    pip install --quiet nvidia-tensorrt pycuda
}

# Make sure we have required packages
Write-Host "Checking dependencies..." -ForegroundColor Cyan
pip install --quiet onnx

# Optimize with FP16 precision (faster and works on most hardware)
Write-Host "`n====================================================" -ForegroundColor Green
Write-Host "Optimizing models with FP16 precision" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
python optimize_face_models.py --precision fp16 --benchmark

Write-Host "`n====================================================" -ForegroundColor Green
Write-Host "Optimization complete!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host "`nThe following optimized models are now available:"
Write-Host "  FP16 models:" -ForegroundColor Cyan
Write-Host "    - Project\models\blazeface_fp16.trt" -ForegroundColor White
Write-Host "    - Project\models\inception_resnet_v1_fp16.trt" -ForegroundColor White
Write-Host "`nTo use these optimized models in your application, the code"
Write-Host "will automatically use TensorRT for acceleration if available."
Write-Host "`nFor higher accuracy but slower inference, you can also optimize with FP32:"
Write-Host "    python optimize_face_models.py --precision fp32" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Green
