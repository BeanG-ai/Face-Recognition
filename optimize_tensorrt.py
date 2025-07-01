#!/usr/bin/env python3
"""
TensorRT Model Optimizer for Face Recognition Models

This script converts ONNX models to TensorRT format with FP16 or FP32 precision
to significantly improve inference speed on NVIDIA GPUs, especially Jetson Orin.

Usage:
    python optimize_tensorrt.py --model <model_path> --output <output_path> --precision <fp16|fp32>
    
Example:
    # Convert to FP16 precision
    python optimize_tensorrt.py --model Project/models/inception_resnet_v1.onnx --precision fp16
    
    # Convert to FP32 precision
    python optimize_tensorrt.py --model Project/models/blazeface.onnx --precision fp32
"""

import os
import argparse
import logging
import time
import sys
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TensorRT-Optimizer')

def check_tensorrt():
    """Check if TensorRT is available and return the module if it is."""
    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
        return trt
    except ImportError:
        logger.error("TensorRT is not installed. Please install it before continuing.")
        logger.error("On Jetson: sudo apt-get install -y tensorrt")
        logger.error("Then: pip install nvidia-pyindex && pip install nvidia-tensorrt")
        sys.exit(1)

def check_pycuda():
    """Check if PyCUDA is available."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        logger.info("PyCUDA is available")
        return True
    except ImportError:
        logger.error("PyCUDA is not installed. Please install it before continuing.")
        logger.error("pip install pycuda")
        sys.exit(1)

def build_engine(trt, model_path, output_path, precision, workspace_size=1):
    """
    Build a TensorRT engine from an ONNX model.
    
    Args:
        trt: TensorRT module
        model_path: Path to the ONNX model
        output_path: Path to save the TensorRT engine
        precision: Precision to use (fp32 or fp16)
        workspace_size: Maximum workspace size in GB
        
    Returns:
        Path to the saved TensorRT engine
    """
    import pycuda.driver as cuda
    
    logger.info(f"Building TensorRT engine for {model_path} with {precision} precision")
    
    # Create TensorRT logger
    trt_logger = trt.Logger(trt.Logger.INFO)
    
    # Create builder and network
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * (1 << 30)  # Convert GB to bytes
    
    # Set precision flags
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            logger.info("Using FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 is not supported on this platform, using FP32 instead")
    else:
        logger.info("Using FP32 precision")
    
    # Parse ONNX model
    logger.info("Parsing ONNX model...")
    parser = trt.OnnxParser(network, trt_logger)
    
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parsing error: {parser.get_error(error)}")
            raise RuntimeError(f"Failed to parse ONNX file {model_path}")
    
    # Build engine
    logger.info("Building TensorRT engine (this may take a while)...")
    start_time = time.time()
    engine = builder.build_engine(network, config)
    build_time = time.time() - start_time
    
    if not engine:
        raise RuntimeError("Failed to build TensorRT engine")
    
    logger.info(f"Engine built in {build_time:.2f} seconds")
    
    # Serialize engine and save to disk
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"TensorRT engine saved to {output_path}")
    logger.info(f"Engine size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    
    return output_path

def benchmark_engine(trt, engine_path, input_shape=None, num_warmup=5, num_inference=50):
    """
    Benchmark a TensorRT engine.
    
    Args:
        trt: TensorRT module
        engine_path: Path to the TensorRT engine
        input_shape: Shape of the input data (batch_size, C, H, W)
        num_warmup: Number of warmup iterations
        num_inference: Number of inference iterations
        
    Returns:
        Average inference time in milliseconds
    """
    import pycuda.driver as cuda
    
    logger.info(f"Benchmarking TensorRT engine: {engine_path}")
    
    # Load engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if not engine:
        logger.error(f"Failed to load TensorRT engine from {engine_path}")
        return None
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Determine input and output shapes
    if input_shape is None:
        # Get input shape from engine
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                input_shape = tuple(context.get_binding_shape(i))
                break
    
    if input_shape is None:
        logger.error("Could not determine input shape")
        return None
    
    logger.info(f"Using input shape: {input_shape}")
    
    # Allocate memory for input and output
    h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
    
    # Generate random input data
    np.copyto(h_input, np.random.random(input_shape).astype(np.float32))
    
    # Allocate device memory for input
    d_input = cuda.mem_alloc(h_input.nbytes)
    
    # Allocate output memory
    output_shapes = []
    output_dtypes = []
    d_outputs = []
    h_outputs = []
    
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            output_shape = tuple(context.get_binding_shape(i))
            output_shapes.append(output_shape)
            
            # Determine output data type
            output_dtype = np.float32  # Default
            d_outputs.append(cuda.mem_alloc(np.empty(output_shape, dtype=output_dtype).nbytes))
            h_outputs.append(cuda.pagelocked_empty(output_shape, dtype=output_dtype))
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Warm up
    logger.info(f"Warming up with {num_warmup} iterations...")
    for _ in range(num_warmup):
        # Copy input data to device
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        # Run inference
        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Copy outputs back to host
        for i, d_output in enumerate(d_outputs):
            cuda.memcpy_dtoh_async(h_outputs[i], d_output, stream)
        
        # Synchronize
        stream.synchronize()
    
    # Benchmark
    logger.info(f"Running {num_inference} iterations for benchmarking...")
    
    # Create events for timing
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    # Time inference
    inference_times = []
    
    for _ in range(num_inference):
        # Record start time
        start_event.record(stream)
        
        # Copy input data to device
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        # Run inference
        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Copy outputs back to host
        for i, d_output in enumerate(d_outputs):
            cuda.memcpy_dtoh_async(h_outputs[i], d_output, stream)
        
        # Record end time
        end_event.record(stream)
        stream.synchronize()
        
        # Calculate inference time
        inference_time = start_event.time_till(end_event)
        inference_times.append(inference_time)
    
    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    throughput = 1000 / avg_time  # FPS
    
    logger.info(f"Inference statistics:")
    logger.info(f"  Average time: {avg_time:.2f} ms")
    logger.info(f"  Min time: {min_time:.2f} ms")
    logger.info(f"  Max time: {max_time:.2f} ms")
    logger.info(f"  Throughput: {throughput:.2f} FPS")
    
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Optimize ONNX models with TensorRT')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the ONNX model to optimize')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the optimized TensorRT engine')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp16',
                       help='Precision to use for optimization (default: fp16)')
    parser.add_argument('--workspace', type=int, default=1,
                       help='Maximum workspace size in GB (default: 1)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark the optimized model')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file {args.model} does not exist")
        return 1
    
    # Check if model is an ONNX model
    if not args.model.lower().endswith('.onnx'):
        logger.error(f"Model file {args.model} is not an ONNX model")
        return 1
    
    # Set default output path if not specified
    if args.output is None:
        model_dir = os.path.dirname(args.model)
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = os.path.join(model_dir, f"{model_name}_{args.precision}.trt")
    
    # Check dependencies
    trt = check_tensorrt()
    check_pycuda()
    
    try:
        # Build TensorRT engine
        engine_path = build_engine(
            trt=trt,
            model_path=args.model,
            output_path=args.output,
            precision=args.precision,
            workspace_size=args.workspace
        )
        
        # Benchmark the engine if requested
        if args.benchmark and engine_path:
            benchmark_engine(trt=trt, engine_path=engine_path)
        
        logger.info("Optimization completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
