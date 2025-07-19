import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
import os
from datetime import datetime

class HeadPoseEnrollment:
    def __init__(self, save_path='data/known_faces/new_person', yaw_threshold=10.0, pitch_threshold=10.0):
        """Face Enrollment với hiệu năng Production """
        
        # Backward compatibility cho registration pipeline
        self.SAVE_PATH = save_path
        self.YAW_THRESHOLD = yaw_threshold  
        self.PITCH_THRESHOLD = pitch_threshold
        self.captured_images = {}
        
        # Tạo thư mục nếu chưa có
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        
        # MediaPipe với config Production
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Smoothing buffer cho độ ổn định
        self.angle_history = {
            'x': deque(maxlen=5),
            'y': deque(maxlen=5),
            'z': deque(maxlen=5)
        }
        
        # Production landmarks (12 điểm)
        self.important_landmarks = [1, 33, 133, 362, 263, 61, 291, 199, 10, 151, 172, 397]
        
        # Enrollment states
        self.enrollment_steps = [
            {"name": "CENTER", "angle_range": {"x": (-5, 5), "y": (-5, 5)}, "duration": 2},
            {"name": "UP", "angle_range": {"x": (4, 20), "y": (-10, 10)}, "duration": 2},
            {"name": "DOWN", "angle_range": {"x": (-20, -8), "y": (-10, 10)}, "duration": 2},
            {"name": "LEFT", "angle_range": {"x": (-10, 10), "y": (-25, -8)}, "duration": 2},
            {"name": "RIGHT", "angle_range": {"x": (-10, 10), "y": (8, 25)}, "duration": 2}
        ]
        
        self.current_step = 0
        self.step_start_time = None
        self.step_progress = 0
        self.captured_poses = []
        self.enrollment_complete = False
        
        # Camera setup - CHỈ khởi tạo cho standalone mode
        self.cap = None
        
    def smooth_angle(self, new_x, new_y, new_z):
        """Làm mượt góc với Production algorithm"""
        self.angle_history['x'].append(new_x)
        self.angle_history['y'].append(new_y) 
        self.angle_history['z'].append(new_z)
        
        return (
            np.mean(self.angle_history['x']),
            np.mean(self.angle_history['y']),
            np.mean(self.angle_history['z'])
        )
    
    def get_head_pose(self, image):
        """Production-grade head pose detection"""
        img_h, img_w = image.shape[:2]
        
        # Process với MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d = []
                face_2d = []
                
                # Thu thập 12 landmarks quan trọng
                for idx in self.important_landmarks:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                
                if len(face_2d) >= 6:
                    # Production camera matrix
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    
                    focal_length = 1.5 * img_w
                    cam_matrix = np.array([
                        [focal_length, 0, img_w / 2],
                        [0, focal_length, img_h / 2],
                        [0, 0, 1]
                    ])
                    
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    
                    # Solve PnP với ITERATIVE method
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, dist_matrix,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        
                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360
                        
                        # Apply Production smoothing
                        smooth_x, smooth_y, smooth_z = self.smooth_angle(x, y, z)
                        
                        return smooth_x, smooth_y, smooth_z, True
        
        return 0, 0, 0, False
    
    def check_pose_in_range(self, x, y, target_range):
        """Kiểm tra góc trong phạm vi yêu cầu"""
        x_min, x_max = target_range["x"]
        y_min, y_max = target_range["y"]
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def draw_enrollment_ui(self, image, x, y, z, detected):
        """Vẽ UI enrollment với Production styling"""
        img_h, img_w = image.shape[:2]
        center_x, center_y = img_w // 2, img_h // 2
        
        if self.enrollment_complete:
            # Enrollment hoàn thành
            cv2.putText(image, "ENROLLMENT COMPLETE!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(image, "Press 'S' to save or ESC to exit", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return
        
        if not detected:
            cv2.putText(image, "No Face Detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        
        # Current step info
        current_step_info = self.enrollment_steps[self.current_step]
        step_name = current_step_info["name"]
        
        # Progress circle
        radius = 100
        cv2.circle(image, (center_x, center_y), radius, (100, 100, 100), 3)
        
        # Progress arc
        if self.step_start_time:
            elapsed = time.time() - self.step_start_time
            progress = min(elapsed / current_step_info["duration"], 1.0)
            self.step_progress = progress
            
            # Vẽ arc progress
            arc_angle = int(360 * progress)
            if arc_angle > 0:
                # Tạo mask cho arc
                overlay = image.copy()
                cv2.ellipse(overlay, (center_x, center_y), (radius, radius), 
                           -90, 0, arc_angle, (0, 255, 0), 8)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Face position indicator - HIDDEN
        # face_x = center_x + int(y * 3)  # y axis for left/right
        # face_y = center_y - int(x * 3)  # x axis for up/down (inverted)
        # cv2.circle(image, (face_x, face_y), 15, (255, 255, 0), -1)
        
        # Target position cho step hiện tại - HIDDEN
        # if step_name == "CENTER":
        #     target_x, target_y = center_x, center_y
        #     color = (0, 255, 255)
        # elif step_name == "UP":
        #     target_x, target_y = center_x, center_y - 60
        #     color = (255, 0, 255)
        # elif step_name == "DOWN":
        #     target_x, target_y = center_x, center_y + 60
        #     color = (255, 255, 0)
        # elif step_name == "LEFT":
        #     target_x, target_y = center_x - 80, center_y
        #     color = (0, 255, 0)
        # elif step_name == "RIGHT":
        #     target_x, target_y = center_x + 80, center_y
        #     color = (255, 0, 0)
        
        # Vẽ target - HIDDEN
        # cv2.circle(image, (target_x, target_y), 25, color, 3)
        # cv2.putText(image, step_name, (target_x - 30, target_y - 35), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Instructions
        cv2.putText(image, f"Step {self.current_step + 1}/5: Look {step_name}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Progress percentage
        progress_text = f"Progress: {int(self.step_progress * 100)}%"
        cv2.putText(image, progress_text, (20, img_h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Step progress indicator
        for i in range(len(self.enrollment_steps)):
            x_pos = 20 + i * 40
            color = (0, 255, 0) if i < self.current_step else (100, 100, 100)
            if i == self.current_step:
                color = (0, 255, 255)
            cv2.circle(image, (x_pos, img_h - 100), 15, color, -1)
            cv2.putText(image, str(i+1), (x_pos-5, img_h-95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def run_enrollment(self):
        """Chạy quá trình enrollment"""
        print("🎯 Face Enrollment với Production Performance")
        print("📋 Hướng dẫn:")
        print("   1. Nhìn thẳng để bắt đầu")
        print("   2. Làm theo hướng dẫn trên màn hình")
        print("   3. Giữ pose trong 2 giây cho mỗi hướng")
        print("   4. Nhấn ESC để thoát, 'S' để lưu khi hoàn thành")
        
        # Initialize camera for standalone mode
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        while True:
            start_time = time.time()
            success, image = self.cap.read()
            if not success:
                continue
            
            # Flip image
            image = cv2.flip(image, 1)
            
            # Get head pose với Production performance
            x, y, z, detected = self.get_head_pose(image)
            
            # Process enrollment logic
            if detected and not self.enrollment_complete:
                current_step_info = self.enrollment_steps[self.current_step]
                
                # Check if pose is in target range
                if self.check_pose_in_range(x, y, current_step_info["angle_range"]):
                    if self.step_start_time is None:
                        self.step_start_time = time.time()
                        print(f"✅ Detected {current_step_info['name']} pose!")
                    
                    # Check if duration completed
                    elapsed = time.time() - self.step_start_time
                    if elapsed >= current_step_info["duration"]:
                        # Capture pose data
                        pose_data = {
                            "step": current_step_info["name"],
                            "angles": {"x": x, "y": y, "z": z},
                            "timestamp": time.time()
                        }
                        self.captured_poses.append(pose_data)
                        print(f"📸 Captured {current_step_info['name']} pose!")
                        
                        # Move to next step
                        self.current_step += 1
                        self.step_start_time = None
                        self.step_progress = 0
                        
                        if self.current_step >= len(self.enrollment_steps):
                            self.enrollment_complete = True
                            print("🎉 Enrollment completed successfully!")
                else:
                    # Reset timer if not in range
                    self.step_start_time = None
                    self.step_progress = 0
            
            # Draw UI
            self.draw_enrollment_ui(image, x, y, z, detected)
            
            # Calculate and display FPS
            fps = 1 / (time.time() - start_time + 1e-6)
            cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Enrollment - Production Performance', image)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s') and self.enrollment_complete:
                self.save_enrollment_data()
            elif key == ord('r'):  # Reset
                self.reset_enrollment()
        
        self.cleanup()
    
    def save_enrollment_data(self):
        """Lưu dữ liệu enrollment"""
        if not self.captured_poses:
            print("❌ No data to save!")
            return
        
        filename = f"face_enrollment_{int(time.time())}.txt"
        with open(filename, 'w') as f:
            f.write("Face Enrollment Data\n")
            f.write("=" * 50 + "\n")
            for pose in self.captured_poses:
                f.write(f"Step: {pose['step']}\n")
                f.write(f"Angles: X={pose['angles']['x']:.2f}, Y={pose['angles']['y']:.2f}, Z={pose['angles']['z']:.2f}\n")
                f.write(f"Timestamp: {pose['timestamp']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"💾 Enrollment data saved to: {filename}")
    
    def reset_enrollment(self):
        """Reset quá trình enrollment"""
        self.current_step = 0
        self.step_start_time = None
        self.step_progress = 0
        self.captured_poses = []
        self.enrollment_complete = False
        self.angle_history['x'].clear()
        self.angle_history['y'].clear()
        self.angle_history['z'].clear()
        print("🔄 Enrollment reset!")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    # Backward compatibility methods for registration.py
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame for registration app compatibility
        Returns: Frame with UI drawn
        """
        # Flip frame horizontally to fix mirror effect
        frame = cv2.flip(frame, 1)
        
        if self.is_completed():
            # Show completion message
            cv2.putText(frame, "ENROLLMENT COMPLETE!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Registration finished successfully!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return frame

        # Get head pose
        x, y, z, detected = self.get_head_pose(frame)
        
        # Process enrollment logic
        if detected and not self.enrollment_complete:
            current_step_info = self.enrollment_steps[self.current_step]
            
            # Check if pose is in target range
            if self.check_pose_in_range(x, y, current_step_info["angle_range"]):
                if self.step_start_time is None:
                    self.step_start_time = time.time()
                    print(f"✅ Detected {current_step_info['name']} pose!")
                
                # Check if duration completed
                elapsed = time.time() - self.step_start_time
                if elapsed >= current_step_info["duration"]:
                    # Save image for compatibility
                    self._save_image_compatibility(frame, current_step_info["name"])
                    
                    # Capture pose data
                    pose_data = {
                        "step": current_step_info["name"],
                        "angles": {"x": x, "y": y, "z": z},
                        "timestamp": time.time()
                    }
                    self.captured_poses.append(pose_data)
                    print(f"📸 Captured {current_step_info['name']} pose!")
                    
                    # Move to next step
                    self.current_step += 1
                    self.step_start_time = None
                    self.step_progress = 0
                    
                    if self.current_step >= len(self.enrollment_steps):
                        self.enrollment_complete = True
                        print("🎉 Enrollment completed successfully!")
            else:
                # Reset timer if not in range
                self.step_start_time = None
                self.step_progress = 0
        
        # Draw UI
        self.draw_enrollment_ui(frame, x, y, z, detected)
        
        return frame
    
    def _save_image_compatibility(self, image, stage):
        """Lưu ảnh khuôn mặt đã cắt giống bản đơn giản (Simple version)"""
        from datetime import datetime

        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Tính toạ độ bounding box
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            x_min = int(min(x_coords) * img_w)
            y_min = int(min(y_coords) * img_h)
            x_max = int(max(x_coords) * img_w)
            y_max = int(max(y_coords) * img_h)
            
            # Thêm padding
            w, h = x_max - x_min, y_max - y_min
            pad_w, pad_h = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x_min - pad_w)
            y1 = max(0, y_min - pad_h)
            x2 = min(img_w, x_max + pad_w)
            y2 = min(img_h, y_max + pad_h)
            
            cropped_face = image[y1:y2, x1:x2]
        else:
            # Fallback: lưu nguyên frame nếu không tìm được mặt
            cropped_face = image

        # Lưu ảnh
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.SAVE_PATH, f"face_{timestamp}_{stage}.jpg")
        cv2.imwrite(filename, cropped_face)
        self.captured_images[stage] = filename
        print(f"📸 Da chup va luu anh da CROP buoc [{stage}]: {filename}")
    
    def is_completed(self) -> bool:
        """Check if enrollment is completed - for compatibility"""
        return self.enrollment_complete
    
    def close(self):
        """Close resources - for compatibility"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        print("Da thoat chuong trinh. Cac anh da chup:", self.captured_images)

