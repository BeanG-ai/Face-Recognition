# Unified Face Recognition and Authentication System

A comprehensive facial recognition and authentication system optimized for NVIDIA Jetson Orin Nano X, featuring real-time detection, multi-angle registration, and secure authentication with visual guidance.

## Features

- **Three Operating Modes**:
  - **Registration**: Multi-angle face capture for improved recognition accuracy
  - **Recognition**: Real-time face detection and identification of multiple faces
  - **Authentication**: Secure verification with positioning guidance and feedback

- **Core Technologies**:
  - **FAISS Vector Search**: Fast similarity matching of face embeddings
  - **Face Detection**: Optimized BlazeFace model for real-time detection
  - **Face Recognition**: Inception ResNet v1 for feature extraction
  - **Enhanced Security**: Guide box interface, liveness detection, and multi-angle verification

- **Performance Optimization**:
  - TensorRT acceleration with FP16/FP32 precision options
  - CUDA acceleration on NVIDIA hardware
  - System performance monitoring and reporting

## System Requirements

- NVIDIA Jetson Orin Nano X (recommended) or other computing platforms
- Camera module (built-in or USB)
- Python 3.8 or later
- Required Python packages (see `requirements.txt`)

## Quick Start

### Installation    

1. Clone this repository:
   ```bash
   git clone https://github.com/BeanG-ai/Face-Recognition.git
   cd face-recognition
   ```

2. Setup the project (includes DeepFace and camera configuration):
   - On Jetson: `./jetson_setup.sh` (sets camera to `/dev/video0`)
   - On Windows: `.\setup.ps1` (sets camera to device index 0)
   - On Linux: `bash setup.sh` (sets camera to device index 0)
   - Manual setup: `pip install -r requirements.txt && python setup_project.py`

   **Note**: The setup process will automatically:
   - Install all required dependencies including DeepFace
   - Configure DeepFace models in the project directory
   - Download necessary anti-spoofing models
   - Set up the complete project structure
   - Configure camera device for the specific platform

3. Optimize models with TensorRT (optional, recommended for performance):
   - On Windows: `.\optimize_models.ps1`
   - On Linux/Jetson: `./optimize_models.sh`

### Running the Application

#### 1. User Registration
```bash
python main.py --mode registration --username <name>
```
- Guides user to rotate their face in different directions
- Captures face images from multiple angles
- Extracts and stores face embeddings for improved recognition

#### 2. Face Recognition
```bash
python main.py --mode recognition --threshold 0.65
```
- Detects faces in real-time from camera feed
- Matches detected faces against the database
- Displays recognition results with confidence scores

#### 3. Face Authentication
```bash
# Single authentication
python main.py --mode authentication --auth-mode single 

# Continuous authentication with no frame skipping (best responsiveness)
python main.py --mode authentication --auth-mode continuous --duration 60 --skip-frames 0

# Continuous authentication with default frame skipping (balanced performance)
python main.py --mode authentication --auth-mode continuous --duration 60 --liveness
```
- Displays a guide box for correct face positioning
- Provides real-time positioning feedback
- Verifies identity with multiple consecutive matches


#### Alternative CLI Interface
```bash
# Register a user
python face_recognition_cli.py register <username>

# Recognize faces
python face_recognition_cli.py recognize
```

## TensorRT Acceleration

This system is optimized for inference speed using NVIDIA TensorRT with FP16/FP32 precision options.

### Using TensorRT
```bash
# Enable TensorRT with FP16 precision (default)
python main.py --mode recognition --use-tensorrt --precision fp16

# Use FP32 precision for higher accuracy
python main.py --mode recognition --use-tensorrt --precision fp32
```

### Optimizing Models for Different Hardware

#### NVIDIA Jetson Devices
```bash
sudo apt-get install -y tensorrt
pip install nvidia-pyindex pycuda
./optimize_models.sh
```

#### NVIDIA GPUs on PC/Server
```bash
pip install nvidia-pyindex nvidia-tensorrt pycuda
# Windows: .\optimize_models.ps1
# Linux: ./optimize_models.sh
```

## Command-Line Options

### Main Application Options
- `--mode`: Operating mode (registration, recognition, authentication)
- `--threshold`: Recognition threshold (0.0-1.0)
- `--username`: Username for registration
- `--auth-mode`: Authentication mode (single, continuous)
- `--duration`: Duration for continuous mode in seconds
- `--skip-frames`: Frame skipping for performance (0=no skip, 2=default, higher=faster but less responsive)
- `--liveness`: Enable liveness detection
- `--matches`: Required consecutive matches for authentication
- `--use-tensorrt`: Enable TensorRT acceleration
- `--precision`: Precision mode for TensorRT models (fp16, fp32)
- `--monitor`: Enable performance monitoring
- `--report`: Generate detailed report

## Troubleshooting

### TensorRT Issues
- **TensorRT Not Found**: Install with `sudo apt-get install -y tensorrt` (Jetson) or `pip install nvidia-pyindex nvidia-tensorrt pycuda`
- **CUDA Out of Memory**: Use FP16 precision or reduce image resolution
- **Failed to Build Engine**: Verify ONNX model compatibility and CUDA installation

### ONNX Runtime Issues
- **No CUDA Provider**: Install `onnxruntime-gpu` and verify CUDA installation

## Verifying TensorRT Usage
```bash
python main.py --mode recognition --use-tensorrt --precision fp16
```
- Look for "Using TensorRT optimized..." messages
- Verify faster inference times compared to ONNX Runtime

## Performance Monitoring
```bash
python system_monitor.py
```

For detailed reporting during authentication:
```bash
python main.py --mode authentication --auth-mode continuous --monitor --report
```

## Camera Configuration

The system automatically detects the platform and configures the camera device appropriately:

- **Jetson platforms**: Uses `/dev/video0` (automatic detection)
- **Other platforms**: Uses device index `0` (Windows, Linux, macOS)

To test camera configuration:
```bash
python test_camera_config.py --interactive
```

For detailed camera configuration information, see [CAMERA_CONFIG.md](CAMERA_CONFIG.md).

## System Architecture

The system consists of three main flows:
1. **User Registration**: Multi-angle face capture and embedding creation
2. **Face Recognition**: Real-time face detection and recognition
3. **Facial Authentication**: Secure authentication with guide box interface

For detailed information on system architecture, see [FLOW_GUIDE_EN.md](FLOW_GUIDE_EN.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MediaPipe**: For BlazeFace real-time face detection model
- **Google**: For the Inception ResNet v1 model
- **NVIDIA**: For TensorRT and Jetson platform support
- **OpenCV**: For computer vision algorithms and utilities
- **FAISS**: For efficient similarity search and clustering of dense vectors



