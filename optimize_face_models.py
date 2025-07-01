#!/usr/bin/env python3
"""
Face Recognition Model Optimizer for TensorRT

This script optimizes the face detection and recognition models used in the 
face authentication system to improve inference speed using TensorRT.

Supported models:
- BlazeFace (face detection)
- Inception ResNet v1 (face embedding extraction)

Usage:
    python optimize_face_models.py --precision <fp16|fp32> [--model <model_name>] [--benchmark]

Example:
    # Optimize all models with FP16 precision
    python optimize_face_models.py --precision fp16 --benchmark
    
    # Optimize only the BlazeFace model with FP32 precision
    python optimize_face_models.py --precision fp32 --model blazeface
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Face-Model-Optimizer')

# Define models
MODELS = {
    'blazeface': {
        'path': 'Project/models/blazeface.onnx',
        'description': 'BlazeFace face detection model',
        'input_shape': '1,3,128,128'
    },
    'inception_resnet_v1': {
        'path': 'Project/models/inception_resnet_v1.onnx',
        'description': 'Inception ResNet v1 face embedding model',
        'input_shape': '1,3,160,160'
    }
}

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import tensorrt
        logger.info(f"TensorRT version: {tensorrt.__version__}")
    except ImportError:
        logger.error("TensorRT is not installed. Please install it first.")
        logger.error("For Jetson: sudo apt-get install -y tensorrt")
        logger.error("For other platforms: pip install nvidia-pyindex && pip install nvidia-tensorrt")
        return False
    
    try:
        import pycuda
        logger.info(f"PyCUDA is installed")
    except ImportError:
        logger.error("PyCUDA is not installed. Please install it first.")
        logger.error("pip install pycuda")
        return False
    
    try:
        import onnx
        logger.info(f"ONNX version: {onnx.__version__}")
    except ImportError:
        logger.error("ONNX is not installed. Please install it first.")
        logger.error("pip install onnx")
        return False
    
    return True

def optimize_model(model_info, precision='fp16', benchmark=False):
    """
    Optimize a model using TensorRT.
    
    Args:
        model_info: Dictionary containing model information
        precision: Precision to use (fp16 or fp32)
        benchmark: Whether to benchmark the optimized model
        
    Returns:
        True if optimization succeeded, False otherwise
    """
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} does not exist")
        return False
    
    # Determine output path
    model_name = Path(model_path).stem
    output_path = os.path.join(os.path.dirname(model_path), f"{model_name}_{precision}.trt")
    
    # Build optimization command
    cmd = [
        sys.executable, 'optimize_tensorrt.py',
        '--model', model_path,
        '--output', output_path,
        '--precision', precision
    ]
    
    if benchmark:
        cmd.append('--benchmark')
    
    logger.info(f"Optimizing {model_info['description']} with {precision} precision")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the optimization
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Log the output
        for line in process.stdout.split('\n'):
            if line.strip():
                if 'ERROR' in line:
                    logger.error(line)
                elif 'WARNING' in line:
                    logger.warning(line)
                else:
                    logger.info(line)
        
        logger.info(f"Successfully optimized {model_info['description']}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to optimize {model_info['description']}: {e}")
        if e.stdout:
            logger.info(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False
    
    except Exception as e:
        logger.error(f"Error optimizing {model_info['description']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Optimize face recognition models with TensorRT')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp16',
                       help='Precision to use for optimization (default: fp16)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark the optimized models')
    parser.add_argument('--model', choices=list(MODELS.keys()) + ['all'], default='all',
                       help='Which model to optimize (default: all)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine which models to optimize
    models_to_optimize = []
    if args.model == 'all':
        models_to_optimize = list(MODELS.items())
    else:
        if args.model in MODELS:
            models_to_optimize = [(args.model, MODELS[args.model])]
        else:
            logger.error(f"Unknown model: {args.model}")
            sys.exit(1)
    
    # Optimize each model
    success = True
    for name, model_info in models_to_optimize:
        result = optimize_model(
            model_info=model_info,
            precision=args.precision,
            benchmark=args.benchmark
        )
        
        if not result:
            success = False
    
    if success:
        logger.info("All models successfully optimized")
        logger.info("\nTo use the optimized models in your code, you can:")
        logger.info("1. Use TensorRT API directly")
        logger.info("2. Update your model loading code to use the .trt files")
        logger.info("\nThe optimized models are saved in the Project/models directory")
        
        # Print model paths
        for name, model_info in models_to_optimize:
            model_name = Path(model_info['path']).stem
            trt_path = f"Project/models/{model_name}_{args.precision}.trt"
            logger.info(f"  - {name}: {trt_path}")
    else:
        logger.error("Some models failed to optimize")
        sys.exit(1)

if __name__ == '__main__':
    main()
