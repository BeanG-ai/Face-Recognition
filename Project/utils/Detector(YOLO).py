import cv2
import numpy as np
from typing import List, NamedTuple
from ultralytics import YOLO
import os
import torch
# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray
class FaceDetector:
    """Face detector using YOLOv11."""

    def __init__(self, model_path: str = None, use_gpu: bool = True, use_tensorrt = None, precision = None):
        """Initialize FaceDetector with YOLOv11."""
        self.use_gpu = use_gpu
        self.min_detection_confidence = 0.65
        self.is_pt_model = False  # True nếu là .pt
    
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov11m-face.onnx')

        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)

            if model_path.endswith('.pt'):
                self.is_pt_model = True
                print("Detected PyTorch (.pt) model")
            else:
                self.is_pt_model = False
                print("Detected non-PyTorch model (.onnx/.engine/etc)")
        else:
            print("Model not found, fallback to default yolo11n.pt")
            self.model = YOLO("yolo11n.pt")
            self.is_pt_model = True

        # Set device
        if self.use_gpu:
            try:
                if torch.cuda.is_available():
                    self.device = 'cuda' if self.is_pt_model else 0
                    if self.is_pt_model:
                        self.model.to('cuda')
                    print("GPU enabled")
                else:
                    self.device = 'cpu'
                    if self.is_pt_model:
                        self.model.to('cpu')
                    print("CUDA not available, using CPU")
            except Exception as e:
                print(f"Error checking GPU: {e}")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            if self.is_pt_model:
                self.model.to('cpu')
            print("Forced to use CPU")

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Detect faces in a frame."""
        if frame is None:
            return DetectionResult([], frame)

        frame = cv2.flip(frame, 1)
        annotated_frame = frame.copy()
        bboxes = []

        try:
            if self.is_pt_model:
                results = self.model(frame, verbose=False)
            else:
                results = self.model.predict(frame, verbose=False, device=self.device)

            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < self.min_detection_confidence:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    frame_h, frame_w, _ = frame.shape
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)

                    if w > 0 and h > 0:
                        bboxes.append((x, y, w, h))
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        except Exception as e:
            print(f"Detection error: {e}")

        return DetectionResult(bboxes, annotated_frame)

    def close(self):
        pass
