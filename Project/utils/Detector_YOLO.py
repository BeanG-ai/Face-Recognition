import cv2
import numpy as np
from typing import List, NamedTuple
from ultralytics import YOLO
import os

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray

class FaceDetector:
    """Face detector using YOLOv11."""
    
    def __init__(self, model_path: str = None, use_gpu: bool = True, precision: str = "fp16"):
        """Initialize FaceDetector with YOLOv11."""
        self.use_gpu = use_gpu
        self.min_detection_confidence = 0.65
        self.is_onnx_model = False
        
        # If no model_path provided, try to find YOLO face model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov11m-face.onnx')
        
        # Check if the provided model exists
        if os.path.exists(model_path):
            # Check if it's a YOLO face model (onnx or pt)
            if model_path.endswith(('.onnx', '.pt')) and 'yolo' in model_path.lower():
                print(f"Loading YOLOv11 model from: {model_path}")
                self.model = YOLO(model_path)
                self.use_face_model = True
                
                # Check if it's ONNX model
                if model_path.endswith('.onnx'):
                    self.is_onnx_model = True
                    print("ONNX model detected - using predict mode for inference")
            else:
                # If not YOLO model, fallback to default YOLO with person detection
                print(f"Model {model_path} is not a YOLO model, using yolo11n with person detection")
                self.model = YOLO('yolo11n.pt')
                self.use_face_model = False
        else:
            print("YOLOv11 face model not found, using yolo11n with person detection")
            self.model = YOLO('yolo11n.pt')
            self.use_face_model = False
        
        # Set device - only for PyTorch models, not ONNX
        if not self.is_onnx_model:
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        print("GPU acceleration enabled for face detection")
                        self.device = 'cuda'
                    else:
                        self.model.to('cpu')
                        print("CUDA not available, using CPU for face detection")
                        self.device = 'cpu'
                except Exception as e:
                    print(f"GPU setup failed: {e}, using CPU for face detection")
                    self.model.to('cpu')
                    self.device = 'cpu'
            else:
                self.model.to('cpu')
                print("Using CPU for face detection")
                self.device = 'cpu'
        else:
            # For ONNX models, set device for predict method
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.device = 0  # CUDA device 0
                        print("GPU acceleration enabled for ONNX face detection")
                    else:
                        self.device = 'cpu'
                        print("CUDA not available, using CPU for ONNX face detection")
                except Exception as e:
                    print(f"GPU setup failed: {e}, using CPU for ONNX face detection")
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
                print("Using CPU for ONNX face detection")
        
    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Detect faces in frame."""
        if frame is None:
            return DetectionResult([], frame)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
            
        bboxes = []
        annotated_frame = frame.copy()
        
        try:
            # Run YOLO inference with appropriate method
            if self.is_onnx_model:
                # For ONNX models, use predict with device parameter
                results = self.model.predict(frame, verbose=False, device=self.device)
            else:
                # For PyTorch models, use direct call
                results = self.model(frame, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence score
                        confidence = float(box.conf[0])
                        
                        # Filter by confidence threshold
                        if confidence < self.min_detection_confidence:
                            continue
                        
                        # If using general model, filter for person class only
                        if not self.use_face_model:
                            class_id = int(box.cls[0])
                            # Class 0 is 'person' in COCO dataset
                            if class_id != 0:
                                continue
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to (x, y, width, height) format to match original interface
                        x = int(x1)
                        y = int(y1)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        
                        # Ensure valid bounding box
                        h, w, _ = frame.shape
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        if width > 0 and height > 0:
                            # For person detection, estimate face region as top 25% of person bbox
                            if not self.use_face_model:
                                face_height = int(height * 0.25)
                                face_y = y
                                face_x = x + int(width * 0.15)  # Center face horizontally
                                face_width = int(width * 0.7)
                                
                                if face_height > 30 and face_width > 30:  # Minimum face size
                                    bboxes.append((face_x, face_y, face_width, face_height))
                                    # Draw face bounding box
                                    cv2.rectangle(annotated_frame, (face_x, face_y), 
                                                (face_x + face_width, face_y + face_height), (0, 255, 0), 2)
                            else:
                                bboxes.append((x, y, width, height))
                                # Draw bounding box
                                cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return DetectionResult(bboxes, annotated_frame)
    
    def close(self):
        """Close resources."""
        pass











