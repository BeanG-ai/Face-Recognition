# Face-Recognition
Face recognition on Orin nano

## Project Structure

The project has been reorganized with the following structure:

```
face-recognition/
│
├── main.py                # Main entry point
├── face_recognition_cli.py # Command-line interface
├── setup_project.py       # Project setup script
├── setup.ps1              # Windows setup script
├── setup.sh               # Linux setup script
├── requirements.txt       # Dependencies
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
    │   └── HeadPoseModel.py  # Multi-angle face capture
    │
    ├── models/            # Model files
    │   ├── blaze_face_short_range.tflite # Detection model
    │   └── inception_resnet_v1.onnx      # Recognition model
    │
    └── database/          # User database
        ├── user_db.json   # User information
        ├── embeddings/    # Face embeddings
        └── images/        # Face images
```

## Real-Time Face Recognition

This project is designed to perform real-time face recognition on the Orin Nano X device. The typical pipeline includes:

1. **Face Detection:** Detect faces in video frames using a fast detection model (e.g., MTCNN, SSD, or YOLO).
2. **Face Alignment (optional):** Align detected faces to improve recognition accuracy.
3. **Face Recognition:** Extract features from detected faces and compare them to a database of known faces.
4. **Result Display:** Annotate and display the recognition results in real time.

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

### For Jetson Orin Nano X

When deploying on the Jetson Orin Nano X device, follow these additional steps:

1. Install the required CUDA dependencies
2. Optimize models using TensorRT:
   ```
   python Project/utils/optimize_model.py
   ```

## Project Components

- `main.py`: Main entry point for the application
- `Detector.py`: Face detection using MediaPipe
- `HeadPoseModel.py`: Multi-angle face registration
- `Project/app/registration.py`: User registration workflow
- `Project/app/recognition.py`: Real-time recognition workflow
- `Project/utils/`: Utility functions for database and face processing



