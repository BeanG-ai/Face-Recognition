"""
TensorRT Utilities for Face Recognition System

This module provides functions to load and use TensorRT-optimized models
for face detection and recognition.
"""

import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def is_tensorrt_available():
    """Check if TensorRT is available in the current environment."""
    try:
        import tensorrt as trt
        return True
    except ImportError:
        return False

def is_pycuda_available():
    """Check if PyCUDA is available in the current environment."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True
    except ImportError:
        return False

class TensorRTModelLoader:
    """
    A class for loading and running TensorRT optimized models.
    
    This class simplifies the use of TensorRT models in the face recognition system.
    It supports loading models optimized with different precision (FP16, FP32) and
    automatically falls back to ONNX Runtime if TensorRT is not available.
    """
    
    def __init__(self):
        """Initialize the TensorRT model loader."""
        self.models = {}
        self.can_use_tensorrt = is_tensorrt_available() and is_pycuda_available()
        
        if not self.can_use_tensorrt:
            logger.warning("TensorRT or PyCUDA not available. Will use ONNX Runtime instead.")
    
    def _load_tensorrt_engine(self, engine_path):
        """
        Load a TensorRT engine from the given path.
        
        Args:
            engine_path: Path to the TensorRT engine
            
        Returns:
            A tuple of (engine, context, input_binding, output_binding)
        """
        import tensorrt as trt
        import pycuda.driver as cuda
        
        # Initialize TensorRT engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
            if not engine:
                logger.error(f"Failed to load TensorRT engine: {engine_path}")
                return None
            
            context = engine.create_execution_context()
            
            # Get input and output binding information
            input_binding = None
            output_bindings = []
            
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    input_binding = i
                else:
                    output_bindings.append(i)
            
            if input_binding is None:
                logger.error(f"No input binding found in engine: {engine_path}")
                return None
            
            logger.info(f"TensorRT engine loaded successfully: {engine_path}")
            
            return engine, context, input_binding, output_bindings
    
    def _load_onnx_model(self, model_path):
        """
        Load an ONNX model using ONNX Runtime.
        
        Args:
            model_path: Path to the ONNX model
            
        Returns:
            An ONNX Runtime inference session
        """
        import onnxruntime as ort
        
        logger.info(f"Loading ONNX model: {model_path}")
        
        # Get available providers
        providers = ort.get_available_providers()
        
        # Use CUDA if available
        if 'CUDAExecutionProvider' in providers:
            logger.info("Using CUDA for ONNX Runtime")
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        else:
            logger.info("Using CPU for ONNX Runtime")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        return session
    
    def load_model(self, model_name, precision='fp16'):
        """
        Load a model for inference.
        
        Args:
            model_name: Name of the model (e.g., 'blazeface', 'inception_resnet_v1')
            precision: Precision to use ('fp16' or 'fp32')
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        model_dir = os.path.join('Project', 'models')
        model_key = f"{model_name}_{precision}"
        
        # Check if model is already loaded
        if model_key in self.models:
            logger.info(f"Model {model_key} already loaded")
            return True
        
        # Try to load TensorRT model
        if self.can_use_tensorrt:
            trt_path = os.path.join(model_dir, f"{model_name}_{precision}.trt")
            
            if os.path.exists(trt_path):
                try:
                    result = self._load_tensorrt_engine(trt_path)
                    
                    if result:
                        engine, context, input_binding, output_bindings = result
                        
                        # Prepare CUDA memory for input and output
                        import pycuda.driver as cuda
                        
                        # Get shapes
                        input_shape = tuple(context.get_binding_shape(input_binding))
                        output_shapes = [tuple(context.get_binding_shape(i)) for i in output_bindings]
                        
                        # Allocate memory
                        h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
                        d_input = cuda.mem_alloc(h_input.nbytes)
                        
                        h_outputs = [cuda.pagelocked_empty(shape, dtype=np.float32) for shape in output_shapes]
                        d_outputs = [cuda.mem_alloc(output.nbytes) for output in h_outputs]
                        
                        # Create bindings list
                        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
                        
                        # Create CUDA stream
                        stream = cuda.Stream()
                        
                        # Store model information
                        self.models[model_key] = {
                            'type': 'tensorrt',
                            'engine': engine,
                            'context': context,
                            'input_shape': input_shape,
                            'output_shapes': output_shapes,
                            'h_input': h_input,
                            'd_input': d_input,
                            'h_outputs': h_outputs,
                            'd_outputs': d_outputs,
                            'bindings': bindings,
                            'stream': stream
                        }
                        
                        logger.info(f"Loaded TensorRT model {model_key} with {precision} precision")
                        logger.info(f"Input shape: {input_shape}")
                        for i, shape in enumerate(output_shapes):
                            logger.info(f"Output {i} shape: {shape}")
                        
                        return True
                except Exception as e:
                    logger.error(f"Error loading TensorRT model {trt_path}: {e}")
        
        # Fall back to ONNX Runtime
        onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
        
        if os.path.exists(onnx_path):
            try:
                session = self._load_onnx_model(onnx_path)
                
                # Store model information
                self.models[model_key] = {
                    'type': 'onnx',
                    'session': session,
                    'inputs': session.get_inputs(),
                    'outputs': session.get_outputs()
                }
                
                logger.info(f"Loaded ONNX model {onnx_path} (fallback from TensorRT)")
                return True
            except Exception as e:
                logger.error(f"Error loading ONNX model {onnx_path}: {e}")
        
        logger.error(f"Could not load model {model_name} with precision {precision}")
        return False
    
    def run_inference(self, model_name, input_data, precision='fp16'):
        """
        Run inference on the given input data.
        
        Args:
            model_name: Name of the model (e.g., 'blazeface', 'inception_resnet_v1')
            input_data: Input data as a numpy array
            precision: Precision to use ('fp16' or 'fp32')
            
        Returns:
            Output data as a numpy array or list of arrays
        """
        model_key = f"{model_name}_{precision}"
        
        # Load model if not already loaded
        if model_key not in self.models:
            if not self.load_model(model_name, precision):
                raise RuntimeError(f"Failed to load model {model_name} with precision {precision}")
        
        model_info = self.models[model_key]
        
        # Run inference with TensorRT
        if model_info['type'] == 'tensorrt':
            import pycuda.driver as cuda
            
            # Prepare input
            input_shape = model_info['input_shape']
            
            # Reshape input if needed
            if input_data.shape != input_shape:
                logger.debug(f"Reshaping input from {input_data.shape} to {input_shape}")
                input_data = input_data.reshape(input_shape)
            
            # Copy input to host memory
            np.copyto(model_info['h_input'], input_data)
            
            # Copy input to device
            cuda.memcpy_htod_async(model_info['d_input'], model_info['h_input'], model_info['stream'])
            
            # Run inference
            model_info['context'].execute_async_v2(
                bindings=model_info['bindings'],
                stream_handle=model_info['stream'].handle
            )
            
            # Copy outputs back to host
            for i, d_output in enumerate(model_info['d_outputs']):
                cuda.memcpy_dtoh_async(model_info['h_outputs'][i], d_output, model_info['stream'])
            
            # Synchronize
            model_info['stream'].synchronize()
            
            # Return outputs
            if len(model_info['h_outputs']) == 1:
                return model_info['h_outputs'][0].copy()
            else:
                return [output.copy() for output in model_info['h_outputs']]
        
        # Run inference with ONNX Runtime
        elif model_info['type'] == 'onnx':
            # Prepare input
            input_name = model_info['inputs'][0].name
            
            # Run inference
            outputs = model_info['session'].run(None, {input_name: input_data})
            
            # Return outputs
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        
        else:
            raise RuntimeError(f"Unknown model type: {model_info['type']}")

# Create a global instance for easy access
tensorrt_model_loader = TensorRTModelLoader()

def get_optimized_model_path(model_name, precision='fp16'):
    """
    Get the path to the optimized TensorRT model if available, or fall back to the ONNX model.
    
    Args:
        model_name: Name of the model (e.g., 'blazeface', 'inception_resnet_v1')
        precision: Precision to use ('fp16' or 'fp32')
        
    Returns:
        Path to the model file and a boolean indicating if it's a TensorRT model
    """
    base_dir = os.path.join('Project', 'models')
    
    # Check for TensorRT model
    trt_path = os.path.join(base_dir, f"{model_name}_{precision}.trt")
    if os.path.exists(trt_path) and is_tensorrt_available() and is_pycuda_available():
        logger.info(f"Using optimized TensorRT model: {trt_path}")
        return trt_path, True
    
    # Fall back to ONNX model
    onnx_path = os.path.join(base_dir, f"{model_name}.onnx")
    if os.path.exists(onnx_path):
        logger.info(f"Using ONNX model: {onnx_path}")
        return onnx_path, False
    
    # If neither is found, return None
    logger.error(f"No model found for {model_name}")
    return None, False

def load_optimized_model(model_name, precision='fp16'):
    """
    Load an optimized model for face detection or recognition.
    
    This function handles loading either a TensorRT optimized model or falling back
    to the original ONNX model if TensorRT is not available.
    
    Args:
        model_name: Name of the model (e.g., 'blazeface', 'inception_resnet_v1')
        precision: Precision to use ('fp16' or 'fp32')
        
    Returns:
        A function that takes input data and returns the model output
    """
    # Only support fp16 and fp32
    if precision not in ['fp16', 'fp32']:
        logger.warning(f"Unsupported precision: {precision}, falling back to fp16")
        precision = 'fp16'
        
    # Check if TensorRT is available
    if not is_tensorrt_available() or not is_pycuda_available():
        logger.warning("TensorRT or PyCUDA not available. Using ONNX Runtime instead.")
        
        # Load ONNX model using ONNX Runtime
        import onnxruntime as ort
        
        onnx_path = os.path.join('Project', 'models', f"{model_name}.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Get available providers
        providers = ort.get_available_providers()
        
        # Use CUDA if available
        if 'CUDAExecutionProvider' in providers:
            logger.info("Using CUDA for ONNX Runtime")
            session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        else:
            logger.info("Using CPU for ONNX Runtime")
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        
        def infer_function(input_data):
            return session.run(None, {input_name: input_data})[0]
        
        return infer_function
    
    # Load TensorRT model
    return lambda input_data: tensorrt_model_loader.run_inference(model_name, input_data, precision)
