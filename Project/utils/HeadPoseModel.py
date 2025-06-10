# tools/register_face.py

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

class HeadPoseEnrollment:
    """
    Class để thực hiện quy trình đăng ký khuôn mặt theo nhiều góc độ (Face ID style).
    """
    def __init__(self, save_path='data/known_faces/new_person', yaw_threshold=20.0, pitch_threshold=15.0):
        self.SAVE_PATH = save_path
        self.YAW_THRESHOLD = yaw_threshold
        self.PITCH_THRESHOLD = pitch_threshold

        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.STAGES = ['LEFT', 'RIGHT', 'UP', 'FORWARD']
        self.stage_instructions = {
            'LEFT': 'Vui long xoay dau sang TRAI',
            'RIGHT': 'Tiep theo, xoay dau sang PHAI',
            'UP': 'Tuyet voi! Gio hay NHIN LEN',
            'FORWARD': 'Cuoi cung, nhin THANG vao camera',
            'COMPLETED': 'HOAN TAT! Cam on ban.'
        }
        self.current_stage_index = 0
        self.captured_images = {}

    def _estimate_head_pose(self, face_landmarks, image_shape):
        h, w, _ = image_shape
        face_3d, face_2d = [], []
        landmark_indices = [1, 33, 263, 61, 291, 199]
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in landmark_indices:
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y]); face_3d.append([x, y, lm.z])
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
        success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1)))
        if not success: return 0, 0
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[1] * 360, angles[0] * 360  # yaw, pitch

    def _save_image(self, image, bbox, stage):
        x, y, w, h = bbox
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(image.shape[1], x + w + pad_w), min(image.shape[0], y + h + pad_h)
        cropped_face = image[y1:y2, x1:x2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.SAVE_PATH, f"face_{timestamp}_{stage}.jpg")
        cv2.imwrite(filename, cropped_face)
        self.captured_images[stage] = filename
        print(f"Da chup va luu anh cho buoc [{stage}]: {filename}")

    def _draw_ui(self, frame):
        h, w, _ = frame.shape
        center_x, center_y, radius = w // 2, h // 2, int(min(w, h) * 0.3)
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 10)
        progress_arc_map = {'LEFT':(150,270), 'RIGHT':(30,-90), 'UP':(30,150)}
        for i in range(self.current_stage_index):
            stage = self.STAGES[i]
            if stage in progress_arc_map:
                start_angle, end_angle = progress_arc_map[stage]
                cv2.ellipse(frame, (center_x, center_y), (radius,radius), 0, start_angle, end_angle, (0,255,0), 10)
        if self.current_stage_index >= len(self.STAGES):
            instruction = self.stage_instructions['COMPLETED']
            cv2.ellipse(frame, (center_x, center_y), (radius,radius), 0, 0, 360, (0,255,0), 10)
        else:
            instruction = self.stage_instructions[self.STAGES[self.current_stage_index]]
        (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(frame, instruction, (center_x - text_w // 2, center_y + radius + 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Xử lý khung hình đầu vào cho quy trình đăng ký.
        Returns:
            Khung hình đã được vẽ UI.
        """
        ui_frame = self._draw_ui(frame.copy())
        
        if self.is_completed():
            return ui_frame

        original_frame = frame.copy()
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            yaw, pitch = self._estimate_head_pose(face_landmarks, original_frame.shape)
            
            current_stage = self.STAGES[self.current_stage_index]
            condition_met = False
            if current_stage == 'LEFT' and yaw < -self.YAW_THRESHOLD: condition_met = True
            elif current_stage == 'RIGHT' and yaw > self.YAW_THRESHOLD: condition_met = True
            elif current_stage == 'UP' and pitch > self.PITCH_THRESHOLD: condition_met = True
            elif current_stage == 'FORWARD' and abs(yaw) < 5 and abs(pitch) < 5: condition_met = True

            if condition_met:
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                bbox = (int(min(x_coords)*original_frame.shape[1]), int(min(y_coords)*original_frame.shape[0]), 
                        int((max(x_coords)-min(x_coords))*original_frame.shape[1]), int((max(y_coords)-min(y_coords))*original_frame.shape[0]))
                self._save_image(original_frame, bbox, current_stage)
                self.current_stage_index += 1
                time.sleep(0.5)

        return ui_frame

    def is_completed(self) -> bool:
        """Kiểm tra xem quy trình đã hoàn tất chưa."""
        return self.current_stage_index >= len(self.STAGES)

    def close(self):
        """Giải phóng tài nguyên."""
        self.face_mesh.close()
        print("Da thoat chuong trinh. Cac anh da chup:", self.captured_images)