import cv2
import mediapipe as mp
import numpy as np
from typing import List, NamedTuple

# Sử dụng NamedTuple để có cấu trúc kết quả rõ ràng
class DetectionResult(NamedTuple):
    bboxes: List[tuple]
    annotated_frame: np.ndarray
    cropped_faces: List[np.ndarray] # THAY ĐỔI MỚI: Thêm trường cho ảnh đã crop

class FaceDetector:
    """
    Một class để phát hiện khuôn mặt sử dụng MediaPipe Face Detector cho ảnh tĩnh.
    """
    def __init__(self, model_path: str):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence = 0.9,
            
        )

        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """
        Phát hiện khuôn mặt trong một ảnh tĩnh và trả về kết quả.

        Args:
            image (np.ndarray): Ảnh đầu vào (định dạng BGR).

        Returns:
            DetectionResult: Danh sách hộp giới hạn, ảnh đã chú thích, và danh sách ảnh đã crop.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = self.detector.detect(mp_image)

        bboxes = []
        cropped_faces = [] # Danh sách mới để lưu ảnh đã crop
        annotated_image = image.copy() # Tạo một bản sao để chú thích mà không làm thay đổi ảnh gốc

        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                bboxes.append((x, y, w, h))

                # --- THAY ĐỔI MỚI: Cắt ảnh khuôn mặt ---
                # Đảm bảo tọa độ hợp lệ để tránh lỗi Index Out Of Bounds
                # Kiểm tra ranh giới của ảnh
                img_h, img_w, _ = image.shape
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)

                # Cắt ảnh khuôn mặt từ ảnh gốc
                # image[y1:y2, x1:x2]
                cropped_face = image[y1:y2, x1:x2].copy() # .copy() để đảm bảo đây là một bản sao độc lập
                cropped_faces.append(cropped_face)

                # Vẽ hộp giới hạn trên ảnh đã chú thích
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return DetectionResult(bboxes=bboxes, annotated_frame=annotated_image, cropped_faces=cropped_faces)

    def close(self):
        """Giải phóng tài nguyên detector."""
        self.detector.close()

detector = FaceDetector(model_path=r"C:\Users\Acer\Downloads\blaze_face_short_range.tflite")
image = cv2.imread(r"C:\Users\Acer\Pictures\z4467235177351_f4549c0844fa94af50f3f5de293b297b.jpg")
result = detector.detect_image(image)

