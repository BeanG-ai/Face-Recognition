import cv2
import mediapipe as mp
import time
from typing import List, NamedTuple
import numpy as np
import os

# Import TensorRT utilities
from Project.utils.tensorrt_utils import load_optimized_model

# Sử dụng NamedTuple để có cấu trúc kết quả rõ ràng
class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray

class FaceDetector:
    """
    Một class để phát hiện khuôn mặt sử dụng BlazeFace model.
    """
    def __init__(self, model_path: str, use_tensorrt: bool = True, precision: str = 'fp16'):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to the model file
            use_tensorrt: Whether to use TensorRT optimization if available
            precision: Precision to use for TensorRT ('fp16' or 'fp32')
        """
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        
        # Try to use TensorRT optimized model if requested
        if use_tensorrt:
            try:
                # Load optimized model
                self.model = load_optimized_model('blazeface', precision=precision)
                self.using_tensorrt = True
                print(f"Using TensorRT optimized BlazeFace model with {precision} precision")
            except Exception as e:
                print(f"Failed to load TensorRT model: {e}")
                print("Falling back to MediaPipe detector")
                self.using_tensorrt = False
                self._init_mediapipe(model_path)
        else:
            # Use MediaPipe detector
            self.using_tensorrt = False
            self._init_mediapipe(model_path)
    
    def _init_mediapipe(self, model_path: str):
        """Initialize MediaPipe face detector as fallback."""
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        
        self.latest_result = None
        
        # Callback để nhận kết quả từ detector
        def result_callback(result: mp.tasks.vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_result = result

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback
        )
        
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        self.timestamp = 0
    
    def _process_blazeface_output(self, output, frame):
        """Process BlazeFace model output to get bounding boxes."""
        # BlazeFace outputs detection boxes and scores
        if isinstance(output, tuple) or isinstance(output, list):
            # Unpack multiple outputs if needed
            boxes = output[0]  # Assuming first output is boxes
            scores = output[1]  # Assuming second output is scores
        else:
            # Single output tensor that includes both boxes and scores
            # This depends on your specific model structure
            boxes = output[:, :4]  # First 4 values are box coordinates
            scores = output[:, 4]  # 5th value is confidence score
        
        # Convert normalized coordinates to pixel values
        height, width = frame.shape[:2]
        bboxes = []
        
        # Detection threshold
        threshold = 0.7
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < threshold:
                continue
                
            # Convert coordinates based on your model's output format
            # This may need adjustment based on how your model outputs coordinates
            y_min, x_min, y_max, x_max = box
            
            # Convert normalized coordinates [0,1] to pixel values
            x = int(x_min * width)
            y = int(y_min * height)
            w = int((x_max - x_min) * width)
            h = int((y_max - y_min) * height)
            
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Phát hiện khuôn mặt trong một khung hình và trả về kết quả.

        Args:
            frame (np.ndarray): Khung hình đầu vào (định dạng BGR).

        Returns:
            DetectionResult: Một đối tượng chứa danh sách các hộp giới hạn (bboxes)
                             và khung hình đã được chú thích (annotated_frame).
        """
        if self.using_tensorrt:
            # Process with TensorRT BlazeFace model
            # Preprocess image for BlazeFace model
            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(input_img, (128, 128))  # BlazeFace typically uses 128x128
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))  # HWC to CHW format
            input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
            
            # Run inference
            output = self.model(input_img)
            
            # Process output to get bounding boxes
            bboxes = self._process_blazeface_output(output, frame)
        else:
            # Process with MediaPipe detector (fallback)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            frame_timestamp_ms = int(time.time() * 1000)
            self.detector.detect_async(mp_image, frame_timestamp_ms)
            
            bboxes = []
            
            if self.latest_result:
                for detection in self.latest_result.detections:
                    bbox = detection.bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    bboxes.append((x, y, w, h))
        
        # Create annotated frame
        annotated_frame = frame.copy()
        for x, y, w, h in bboxes:
            # Vẽ lên khung hình
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return DetectionResult(bboxes=bboxes, annotated_frame=annotated_frame)

    def close(self):
        """Giải phóng tài nguyên detector."""
        if hasattr(self, 'detector') and not self.using_tensorrt:
            self.detector.close()