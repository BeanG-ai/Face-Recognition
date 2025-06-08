

import cv2
import mediapipe as mp
import time
from typing import List, NamedTuple
import numpy as np
# Sử dụng NamedTuple để có cấu trúc kết quả rõ ràng
class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray

class FaceDetector:
    """
    Một class để phát hiện khuôn mặt sử dụng MediaPipe Face Detector.
    """
    def __init__(self, model_path: str):
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

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Phát hiện khuôn mặt trong một khung hình và trả về kết quả.

        Args:
            frame (np.ndarray): Khung hình đầu vào (định dạng BGR).

        Returns:
            DetectionResult: Một đối tượng chứa danh sách các hộp giới hạn (bboxes)
                             và khung hình đã được chú thích (annotated_frame).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, frame_timestamp_ms)
        
        bboxes = []
        annotated_frame = frame.copy()
        
        if self.latest_result:
            for detection in self.latest_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                bboxes.append((x, y, w, h))
                
                # Vẽ lên khung hình
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        return DetectionResult(bboxes=bboxes, annotated_frame=annotated_frame)

    def close(self):
        """Giải phóng tài nguyên detector."""
        self.detector.close()