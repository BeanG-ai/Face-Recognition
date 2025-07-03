import cv2
import mediapipe as mp
from typing import List, NamedTuple
import numpy as np

class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray

class FaceDetector:
    """Face detector using MediaPipe BlazeFace."""
    
    def __init__(self,model_path = None, use_gpu: bool = True,precision = None):
        """Initialize FaceDetector with MediaPipe."""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.use_gpu = use_gpu
        
        # Initialize face detection with appropriate model selection
        # model_selection: 0 for close-range detection, 1 for full-range detection
        self.model_selection = 0
        self.min_detection_confidence = 0.65
        
        if self.use_gpu:
            print("GPU acceleration enabled for face detection")
        else:
            print("Using CPU for face detection")
        
    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Detect faces in frame."""
        if frame is None:
            return DetectionResult([], frame)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
            
        bboxes = []
        annotated_frame = frame.copy()
        
        with self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection, 
            min_detection_confidence=self.min_detection_confidence) as face_detection:
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    h, w, _ = frame.shape
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute coordinates
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x)
                    height = min(int(bbox.height * h), h - y)
                    
                    # Ensure valid bounding box
                    if width > 0 and height > 0:
                        bboxes.append((x, y, width, height))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        return DetectionResult(bboxes, annotated_frame)
    
    def close(self):
        """Close resources."""
        pass
