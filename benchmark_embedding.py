#!/usr/bin/env python3
"""
Benchmark script for comparing embedding speed between FP32 and FP16 models
"""

import os
import cv2
import time
import numpy as np
import onnxruntime as ort
import argparse
from tqdm import tqdm

def preprocess_face(img, target_size=(160, 160)):
    """Preprocess face image for the embedding model"""
    # Resize to target size
    img_resized = cv2.resize(img, target_size)
    
    # Convert to RGB (from BGR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Scale pixel values from [0, 255] to [-1, 1]
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    
    # Transpose from HWC to CHW format (height, width, channels) -> (channels, height, width)
    img_transposed = img_normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch

def benchmark_model(model_path, image_path, num_runs=100, use_fp16_input=False):
    """Benchmark model inference speed"""
    # Load model
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    input_data = preprocess_face(img)
    
    # Convert to FP16 if required
    if use_fp16_input:
        input_data = input_data.astype(np.float16)
    
    # Warm up
    for _ in range(10):
        session.run(None, {'input': input_data})
    
    # Benchmark
    times = []
    for _ in tqdm(range(num_runs), desc=f"Benchmarking {os.path.basename(model_path)}"):
        start_time = time.perf_counter()
        session.run(None, {'input': input_data})
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_dev = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        'avg_time': avg_time,
        'std_dev': std_dev,
        'min_time': min_time,
        'max_time': max_time
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark face embedding models')
    parser.add_argument('--image', type=str, default=None, 
                      help='Path to a face image for benchmarking')
    parser.add_argument('--runs', type=int, default=100,
                      help='Number of inference runs (default: 100)')
    args = parser.parse_args()
    
    # Set model paths
    fp32_model_path = os.path.join("Project", "models", "inception_resnet_v1.onnx")
    fp16_model_path = os.path.join("Project", "models", "inception_resnet_v1_fp16.onnx")
    
    # Find a sample image if not provided
    if args.image is None:
        # Try to find an image in the database directory
        database_dir = os.path.join("Project", "database", "images")
        if os.path.exists(database_dir):
            for root, dirs, files in os.walk(database_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        args.image = os.path.join(root, file)
                        break
                if args.image:
                    break
    
    if args.image is None or not os.path.exists(args.image):
        print("No suitable image found. Please provide an image path using --image")
        return
    
    print(f"Using image: {args.image}")
    print(f"Running {args.runs} inferences for each model")
    
    # Benchmark FP32 model
    if os.path.exists(fp32_model_path):
        print("\n===== Benchmarking FP32 model =====")
        fp32_results = benchmark_model(fp32_model_path, args.image, args.runs)
        print(f"FP32 Results:")
        print(f"  Average time: {fp32_results['avg_time']:.2f} ms")
        print(f"  Standard deviation: {fp32_results['std_dev']:.2f} ms")
        print(f"  Min time: {fp32_results['min_time']:.2f} ms")
        print(f"  Max time: {fp32_results['max_time']:.2f} ms")
    else:
        print(f"FP32 model not found at: {fp32_model_path}")
        fp32_results = None
    
    # Benchmark FP16 model
    if os.path.exists(fp16_model_path):
        print("\n===== Benchmarking FP16 model =====")
        fp16_results = benchmark_model(fp16_model_path, args.image, args.runs, use_fp16_input=True)
        print(f"FP16 Results:")
        print(f"  Average time: {fp16_results['avg_time']:.2f} ms")
        print(f"  Standard deviation: {fp16_results['std_dev']:.2f} ms")
        print(f"  Min time: {fp16_results['min_time']:.2f} ms")
        print(f"  Max time: {fp16_results['max_time']:.2f} ms")
    else:
        print(f"FP16 model not found at: {fp16_model_path}")
        fp16_results = None
    
    # Compare results if both models were benchmarked
    if fp32_results and fp16_results:
        speedup = fp32_results['avg_time'] / fp16_results['avg_time']
        print("\n===== Comparison =====")
        print(f"FP32 average time: {fp32_results['avg_time']:.2f} ms")
        print(f"FP16 average time: {fp16_results['avg_time']:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Performance improvement: {(speedup - 1) * 100:.2f}%")
    
if __name__ == '__main__':
    main() 