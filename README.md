# Unified Face Recognition and Authentication System

A comprehensive facial recognition and authentication system optimized for NVIDIA Jetson Orin Nano X, featuring real-time detection, multi-angle registration, and secure authentication with visual guidance.

## Features

- **Three Operating Modes**:
  - **Registration**: Multi-angle face capture for improved recognition accuracy
  - **Recognition**: Real-time face detection and identification of multiple faces
  - **Authentication**: Secure verification with positioning guidance and feedback

- **Vector Similarity Search**:
  - Integrated FAISS for efficient and scalable nearest neighbor search of face embeddings
  - Fast retrieval and matching in large-scale face databases

- **Face Detection & Recognition**:
  - Fast and accurate face detection using optimized models
  - Feature extraction using Inception ResNet v1
  - Cosine similarity matching for identification

- **Enhanced Authentication**:
  - **Guide Box Interface**: Visual guide for proper face positioning
  - **Real-time Feedback**: Directional guidance (move left/right/up/down/closer/back)
  - **Progressive Authentication**: Multiple consecutive matches for enhanced security
  - **Liveness Detection**: Optional protection against spoofing attacks

- **Performance Optimization**:
  - CUDA acceleration on Jetson hardware
  - Real-time FPS monitoring
  - System resource tracking (CPU, memory, temperature)
  - Performance reporting for analysis

## Project Structure

The project has been reorganized with the following structure:

```
face-recognition/
│
├── main.py                # Main entry point with all operating modes
├── face_recognition_cli.py # Command-line interface
├── setup_project.py       # Project setup script
├── setup.ps1              # Windows setup script
├── setup.sh               # Linux setup script
├── jetson_setup.sh        # Jetson-specific setup script
├── system_monitor.py      # Performance monitoring tool
├── requirements.txt       # Dependencies
├── FLOW_GUIDE_EN.md       # Detailed system flow documentation
│
└── Project/               # Core project directory
    ├── run.py             # Core application runner
    │
    ├── app/               # Application components
    │   ├── registration.py # User registration
    │   └── recognition.py  # Face recognition 
    │
    ├── utils/             # Utility modules
    │   ├── database_utils.py # Database operations
    │   ├── face_utils.py     # Face processing
    │   ├── Detector.py       # Face detection
    │   ├── HeadPoseModel.py  # Multi-angle face capture
    │   └── authentication.py # Facial authentication system
         └── vector_store.py # FAISS vector database utilities
    │
    ├── models/            # Model files
    │   ├── blaze_face_short_range.tflite # Detection model
    │   └── inception_resnet_v1.onnx      # Recognition model
    │
    └── database/          # User database
        ├── user_db.json   # User information
        ├── embeddings/    # Face embeddings
        └── images/        # Face images
        ├── faiss.index    # FAISS index file storing vector embeddings for fast similarity search
         └── faiss_meta.json    # Metadata mapping FAISS vector IDs to user information
```

## System Requirements

- NVIDIA Jetson Orin Nano X (recommended) or other computing platforms
- Camera module (built-in or USB)
- Python 3.8 or later
- Required Python packages (see `requirements.txt`)

## Quick Start

### Installationn    

1. Clone this repository:
   ```bash
   git clone https://github.com/BeanG-ai/Face-Recognition.git
   cd face-recognition
   ```

2. Setup on Jetson Orin Nano:
   ```bash
   chmod +x jetson_setup.sh
   ./jetson_setup.sh
   ```

3. For non-Jetson platforms:
   ```bash
   pip install -r requirements.txt
   python setup_project.py
   ```

### Running the Application

The system offers three main operating modes:

#### 1. User Registration

Register a new user with multi-angle face capture:

```bash
python main.py --mode registration --username <name>
```

This mode will:
- Guide the user to rotate their face in different directions
- Capture face images from multiple angles
- Extract and store face embeddings for improved recognition
- Collect optional metadata (age, major, course, etc.)

#### 2. Face Recognition

Run real-time face recognition:

```bash
python main.py --mode recognition --threshold 0.65
```

This mode will:
- Detect faces in real-time from camera feed
- Match detected faces against the database
- Display recognition results with confidence scores
- Show processing performance metrics

#### 3. Face Authentication

Run facial authentication:

```bash
# Single authentication mode
python main.py --mode authentication --auth-mode single --threshold 0.65

# Continuous authentication mode
python main.py --mode authentication --auth-mode continuous --duration 60 --threshold 0.7 --liveness
```

This mode will:
- Display a guide box for correct face positioning
- Provide real-time positioning feedback
- Verify identity with multiple consecutive matches
- Support both single-time and continuous authentication modes

#### Advanced Usage with Performance Monitoring

Monitor system performance during authentication:

```bash
python main.py --mode authentication --auth-mode continuous --monitor --report
```

### Command-Line Interface (Alternative)

The system also provides a command-line interface for basic operations:

```bash
# Register a new user
python face_recognition_cli.py register <username>

# Recognize faces
python face_recognition_cli.py recognize
```

## System Architecture

The system consists of three main flows:
1. **User Registration**: Multi-angle face capture and embedding creation
2. **Face Recognition**: Real-time face detection and recognition
3. **Facial Authentication**: Secure authentication with guide box interface

For detailed information on system architecture, see [FLOW_GUIDE_EN.md](FLOW_GUIDE_EN.md).

## Performance Optimization for Jetson Orin Nano X

For optimal performance on Jetson devices:

1. Enable maximum clock speeds:
   ```bash
   sudo jetson_clocks --fan
   ```

2. Use performance governor:
   ```bash
   sudo nvpmodel -m 0
   ```

3. Monitor system performance:
   ```bash
   python system_monitor.py
   ```

## Command-Line Options

### Main Application

```
python main.py --help
```

#### General Options:
- `--mode`: Operating mode (registration, recognition, authentication)
- `--threshold`: Recognition threshold (0.0-1.0)

#### Registration Options:
- `--username`: Username for registration

#### Authentication Options:
- `--auth-mode`: Authentication mode (single, continuous)
- `--duration`: Duration for continuous mode in seconds
- `--liveness`: Enable liveness detection
- `--matches`: Required consecutive matches for authentication
- `--timeout`: Authentication timeout in seconds
- `--attempts`: Maximum authentication attempts

#### Monitoring Options:
- `--monitor`: Enable performance monitoring
- `--report`: Generate detailed report
- `--output`: Directory for monitoring data and reports

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Example Pipeline

- Capture video from camera
- Detect faces in each frame
- Align and preprocess faces
- Extract face embeddings
- Compare embeddings with known faces
- Display results with bounding boxes and names

This pipeline is implemented in `src/main.py`.

## Setup and Usage

### Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - opencv-python
  - numpy
  - mediapipe
  - onnxruntime

### Getting Started

1. First, make sure all required directories exist:
   ```
   python setup_project.py
   ```

2. Download necessary models:
   - Face detection model (MediaPipe Blaze Face)
   - Face recognition model (Inception ResNet v1)

3. Run the face recognition system:

   a. To register a new user:
   ```
   python main.py --mode registration
   ```
   
   b. To run face recognition:
   ```
   python main.py --mode recognition
   ```
   
   c. To run facial authentication:
   ```
   python main.py --mode authentication 
   ```
   
   d. To run continuous authentication (for 60 seconds):
   ```
   python main.py --mode authentication --auth-mode continuous --duration 60
   ```

### Facial Authentication System

The system now includes an enhanced facial authentication component with the following features:

1. **Guide Box Interface**: Visual guidance helps users position their face correctly
2. **Real-time Feedback**: Provides directional instructions (move left/right/up/down/closer/back)
3. **Enhanced Security**: Requires multiple consecutive successful matches for authentication
4. **Two Authentication Modes**:
   - **Single Authentication**: One-time verification for secure access
   - **Continuous Authentication**: Ongoing verification for extended sessions

The authentication system is implemented in `Project/utils/authentication.py` and can be tested using the `facial_auth_demo.py` script.

## Command Line Interface

For ease of use, a command-line interface (CLI) is provided:

### Setup the project
```
python face_recognition_cli.py setup
```

### Register a new user
```
python face_recognition_cli.py register <username>
```

### Start face recognition
```
python face_recognition_cli.py recognize
```

## Deployment on Jetson Orin Nano X

When deploying on the Jetson Orin Nano X device:

1. Clone this repository to the device
2. Install the required dependencies
3. Download and place the required model files
4. Use the CLI to interact with the system

For optimal performance on the Jetson Orin Nano X, consider using TensorRT acceleration by running:
```
python tools/optimize_model.py
```

## Quick Start Guide

### Setting Up

1. Clone the repository and navigate to the project directory:
   ```
   git clone https://github.com/yourusername/Face-Recognition.git
   cd Face-Recognition
   ```

2. Set up the project (creates directories and downloads required models):
   ```
   python setup_project.py
   ```
   
   Alternatively, you can use the setup scripts:
   - Windows: `.\setup.ps1`
   - Linux/Jetson: `bash setup.sh`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Using the System

#### Direct Method

1. To register a new user:
   ```
   python main.py --mode registration --username <user_name>
   ```

2. To recognize faces:
   ```
   python main.py --mode recognition
   ```

#### CLI Method

The system provides a convenient command-line interface:

1. To register a new user:
   ```
   python face_recognition_cli.py register <username>
   ```

2. To start face recognition:
   ```
   python face_recognition_cli.py recognize
   ```

3. To set up the project:
   ```
   python face_recognition_cli.py setup
   ```

## Project Components

- `main.py`: Main entry point for the application
- `Detector.py`: Face detection using MediaPipe
- `HeadPoseModel.py`: Multi-angle face registration
- `Project/app/registration.py`: User registration workflow
- `Project/app/recognition.py`: Real-time recognition workflow
- `Project/utils/`: Utility functions for database and face processing



