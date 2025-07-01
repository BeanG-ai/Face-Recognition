# Complete Flow of Face Recognition System

## 1. System Overview

The face recognition system on Jetson Orin Nano X has been consolidated into a unified application with three operating modes:
1. **Registration Mode**: Captures face images from multiple angles and creates embeddings
2. **Recognition Mode**: Detects and recognizes faces in real-time
3. **Authentication Mode**: Provides secure authentication with guide box and real-time feedback

All three modes are now integrated into a single unified system accessible through the `main.py` entry point, providing a consistent user experience and shared codebase.

## 2. User Registration Flow

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Initialize      │     │ Display face      │     │ User rotates face    │
│ camera + UI     ├────►│ rotation guide    ├────►│ (left/right/up/front)│
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Save user info  │     │ Extract embeddings│     │ Save captured face   │
│ to database     │◄────┤ from captured     │◄────┤ images from each     │
│                 │     │ images            │     │ rotation angle       │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Technical Details**:
1. Uses `HeadPoseEnrollment` class to guide the user to rotate their face in 4 directions
2. Uses `MediaPipe FaceMesh` to estimate head rotation angles (yaw, pitch)
3. When the required angle is achieved, the system automatically captures the face image
4. Uses `inception_resnet_v1.onnx` model to extract embeddings from captured images
5. Averages the embeddings to create a feature vector for the user
6. Saves the embedding and user information to the database

## 3. Real-Time Face Recognition Flow

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Initialize      │     │ Detect faces      │     │ Crop and preprocess  │
│ camera+detector ├────►│ in video frame    ├────►│ face images          │
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Display results │     │ Compare with      │     │ Extract embeddings   │
│ on screen       │◄────┤ database and      │◄────┤ from face images     │
│                 │     │ calculate match   │     │                      │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Technical Details**:
1. Uses `FaceDetector` (MediaPipe) to detect faces in video frames
2. For each detected face:
   - Crops the face region from the frame
   - Preprocesses the image (resize, normalization)
   - Extracts embedding using the `inception_resnet_v1.onnx` model
3. Compares embedding with the database:
   - Calculates cosine similarity between current embedding and stored embeddings
   - If similarity exceeds threshold, identifies the user
4. Displays results:
   - Draws bounding box around face
   - Shows user name if recognized
   - Shows "Unknown" if not recognized

## 4. Facial Authentication Flow

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Initialize      │     │ Display face      │     │ Provide real-time    │
│ auth system     ├────►│ guide box         ├────►│ position feedback    │
└─────────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                             │
┌─────────────────┐     ┌───────────────────┐     ┌──────────▼───────────┐
│ Display auth    │     │ Compare faces     │     │ Capture and process  │
│ result          │◄────┤ with database     │◄────┤ face when positioned │
│                 │     │ (multi-check)     │     │ correctly            │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Technical Details**:
1. Uses `FacialAuthenticationSystem` to guide users through authentication
2. Provides **two authentication modes**:
   - **Single Authentication**: One-time verification for secure access
   - **Continuous Authentication**: Real-time monitoring for ongoing verification
3. **Performance optimizations** for Jetson Orin Nano X:
   - Automatic CUDA detection for optimized inference
   - FPS monitoring to track system performance
   - Efficient frame processing
4. **Enhanced security features**:
   - **Consecutive Match Verification**: Requires multiple consecutive matches for enhanced security
   - **Optional Liveness Detection**: Detects blinks to prevent photo spoofing attacks
   - **Authentication Timeout**: Automatically expires authentication after a configurable period
5. **User interface features**:
   - **Guide Box Interface**: Visual guide to help users position their face
   - **Real-time Position Feedback**: Directional guidance (move left/right/up/down/closer/back)
   - **Visual Feedback**: Color-coded interface shows authentication status
   - **Welcome Effect**: Visual confirmation on successful authentication
   - **Authentication Progress Tracking**: Shows progress toward required matches

### Single Authentication Mode
- Guides user to position their face correctly
- Requires multiple consecutive successful matches to authenticate
- Optional liveness detection through blink detection
- Returns authentication result (success/failure) and user information
- Configurable timeout and maximum attempts

### Continuous Authentication Mode
- Monitors for faces continuously for a specified duration
- Can identify multiple users during a session
- Provides real-time feedback on authentication status
- Authentication expires after configurable period of inactivity
- Returns list of all authenticated users during the session

**Authentication Demo Usage**:
```bash
# Run facial authentication (single mode)
python main.py --mode authentication --auth-mode single --threshold 0.65

# Run continuous authentication for 60 seconds
python main.py --mode authentication --auth-mode continuous --duration 60 --threshold 0.65

# Enable enhanced security features
python main.py --mode authentication --auth-mode single --threshold 0.65 --liveness --matches 5

# Run in continuous mode indefinitely (until manually stopped)
python main.py --mode authentication --auth-mode continuous --duration 0 --threshold 0.7
```

**Advanced Configuration Options**:
```
--mode        : Operating mode (registration, recognition, authentication)
--auth-mode   : Authentication mode (single, continuous)
--duration    : Duration for continuous mode in seconds (0 for infinite)
--threshold   : Recognition threshold (0.0-1.0)
--liveness    : Enable liveness detection for enhanced security
--matches     : Number of consecutive matches required for authentication
--timeout     : Authentication timeout in seconds
--attempts    : Maximum number of authentication attempts
--monitor     : Enable system performance monitoring
--report      : Generate detailed authentication report
```

## 5. Project Structure and Execution Flow

```
┌─────────────────┐      ┌───────────────────┐      ┌──────────────────────┐
│ CLI Interface   │      │ main.py           │      │ Core Application     │
│ (command line)  ├─────►│ (unified entry    ├─────►│ (mode-specific       │
└─────────────────┘      │  point)           │      │  implementations)    │
                         └───────────────────┘      └──────────┬───────────┘
                                                               │
                          ┌────────────────────────────────────┼────────────────┐
                          │                                    │                │
                     ┌────▼─────────┐                    ┌─────▼──────────┐     │
                     │ Registration │                    │ Recognition    │     │
                     │ Mode         │                    │ Mode           │     │
                     └──────┬───────┘                    └────────┬───────┘     │
                            │                                     │             │ 
                     ┌──────▼───────┐                    ┌────────▼───────┐     │
                     │ HeadPoseModel│                    │ Detector       │     │    ┌──────────────┐
                     │ (multi-angle │                    │ (face          │     │    │Authentication│
                     │  capture)    │                    │  detection)    │     ├───►│Mode          │
                     └──────┬───────┘                    └────────┬───────┘     │    └───────┬──────┘
                            │                                     │             │            │
                            └─────────────┬─────────────┬─────────┘             │      ┌─────▼─────┐
                                          │             │                       │      │ Auth      │
                                    ┌─────▼─────┐ ┌─────▼──────┐                │      │ System    │
                                    │ face_utils│ │ database   │◄───────────────┘      │           │
                                    │ (image    │ │ (data      │                       └───────────┘
                                    │  process) │ │  storage)  │
                                    └───────────┘ └────────────┘
```

The consolidated system architecture now features:

- **Unified Entry Point**: `main.py` serves as the single entry point for all three operating modes
- **Shared Components**: Core utilities like face detection, database access, and image processing are shared across all modes
- **Mode-Specific Implementations**: Each mode (registration, recognition, authentication) has specialized code while leveraging common components
- **Command-Line Interface**: `face_recognition_cli.py` provides a simplified interface to the main functionality

## 6. Usage

### Using Command Line Interface:

```bash
# Register a new user
python face_recognition_cli.py register <username>

# Recognize faces
python face_recognition_cli.py recognize

# Authenticate a user (single mode)
python face_recognition_cli.py authenticate

# Authenticate continuously for 60 seconds
python face_recognition_cli.py authenticate --mode continuous --duration 60
```

### Using Python commands directly:

```bash
# Register a new user
python main.py --mode registration --username <username>

# Recognize faces
python main.py --mode recognition

# Authenticate a user (single mode)
python main.py --mode authentication --auth-mode single

# Authenticate continuously for 60 seconds
python main.py --mode authentication --auth-mode continuous --duration 60

# Authenticate with enhanced security features
python main.py --mode authentication --liveness --matches 5
```

## 7. Deployment on Jetson Orin Nano X

1. **Environment Setup**:
   - Install required libraries: `pip install -r requirements.txt`
   - Create directory structure: `python setup_project.py`
   - For Jetson-specific setup: `./jetson_setup.sh`

2. **Model Optimization**:
   - Automatically uses CUDA acceleration when available
   - TensorRT conversion for improved inference performance
   - Adjust recognition threshold parameters for the deployment environment

3. **Performance Monitoring**:
   - Real-time FPS monitoring during authentication
   - System resource tracking (CPU, memory, temperature)
   - Performance reporting for optimization
   ```bash
   # Monitor system performance during authentication
   python main.py --mode authentication --monitor --report
   ```

4. **Hardware Optimization**:
   - Enable maximum performance on Jetson:
   ```bash
   # Set maximum clock speeds
   sudo jetson_clocks --fan
   
   # Set performance mode
   sudo nvpmodel -m 0
   
   # Monitor system temperatures
   sudo jtop
   ```


## 8. Advanced Usage

### System Performance Monitoring

The system includes performance monitoring tools specifically designed for Jetson Orin Nano X:

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│ Initialize      │     │ Collect system    │     │ Plot performance     │
│ monitoring      ├────►│ metrics           ├────►│ data                 │
└─────────────────┘     └───────────────────┘     └──────────────────────┘
```

**Features**:
- Real-time monitoring of CPU, memory, and temperature
- Performance tracking during authentication sessions
- Visualization of system metrics
- CSV data export for analysis

**Usage**:
```bash
# Standalone system monitoring
python system_monitor.py --interval 0.5 --output ./monitoring_data

# Monitoring during authentication
python main.py --mode authentication --monitor --report
```

### Authentication Reporting

The advanced authentication demo can generate detailed reports on authentication sessions:

```bash
# Generate authentication report
python main.py --mode authentication --report --output ./reports
```

This creates a JSON report containing:
- Authentication configuration
- Success/failure results
- User information
- Session duration and timestamps
