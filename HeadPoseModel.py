import cv2
import mediapipe as mp
import numpy as np
import time
import os
from PIL import ImageFont, ImageDraw, Image

class HeadPoseEstimator:
    def __init__(self, output_dir="captured_faces"):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.capture_order = ['Trái', 'Phải', 'Thẳng']
        self.captured = {d: False for d in self.capture_order}
        self.current_target = 0

    def draw_text_vietnamese(self, img, text, org, font_size=32, color=(0, 255, 0)):
        # Chuyển OpenCV image sang PIL image để vẽ chữ Unicode
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("arial.ttf", font_size)  # Thay bằng font có tiếng Việt, ví dụ Arial Unicode

        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_progress_circle(self, image, radius=60):
        center = (image.shape[1] - 100, 100)
        colors = [(0, 255, 0) if self.captured[dir] else (180, 180, 180) for dir in self.capture_order]
        angle_step = 360 / len(self.capture_order)

        for i in range(len(self.capture_order)):
            start_angle = int(i * angle_step)
            end_angle = int((i + 1) * angle_step)
            cv2.ellipse(image, center, (radius, radius), -90, start_angle, end_angle, colors[i], thickness=10)

        cv2.circle(image, center, 2, (0, 0, 0), -1)
        return image

    def estimate_pose(self):
        cap = cv2.VideoCapture(0)
        text_map = {'Trái': '📢 Quay đầu sang trái',
                    'Phải': '📢 Quay đầu sang phải',
                    'Thẳng': '📢 Nhìn thẳng vào camera'}

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            start = time.time()
            image = cv2.flip(image, 1)
            img_h, img_w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            face_3d = []
            face_2d = []
            detected_direction = ""

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])
                            if idx == 1:
                                nose_2d = (x, y)

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, *_ = cv2.RQDecomp3x3(rmat)

                    x_angle = angles[0] * 360
                    y_angle = angles[1] * 360

                    if y_angle < -10:
                        detected_direction = "Trái"
                    elif y_angle > 10:
                        detected_direction = "Phải"
                    elif abs(x_angle) < 10:
                        detected_direction = "Thẳng"
                    else:
                        detected_direction = "Khác"

                    # Kiểm tra hướng cần chụp
                    if self.current_target < len(self.capture_order):
                        target = self.capture_order[self.current_target]
                        if detected_direction == target and not self.captured[target]:
                            filename = os.path.join(self.output_dir, f"{target}.jpg")
                            cv2.imwrite(filename, image)
                            self.captured[target] = True
                            print(f"✅ Đã chụp hướng: {target}")
                            self.current_target += 1

            # Vẽ vòng tròn tiến độ
            image = self.draw_progress_circle(image)

            # Hiển thị hướng dẫn
            if self.current_target < len(self.capture_order):
                hint_text = text_map[self.capture_order[self.current_target]]
                image = self.draw_text_vietnamese(image, hint_text, (30, 400), font_size=36, color=(255, 255, 0))
                image = self.draw_text_vietnamese(image, f"Đang nhận diện: {detected_direction}",
                                                  (30, 50), font_size=28, color=(0, 255, 0))
            else:
                image = self.draw_text_vietnamese(image, "🎉 Hoàn thành chụp 3 hướng!", (30, 400), font_size=36,
                                                  color=(0, 255, 0))

            fps = int(1 / (time.time() - start))
            cv2.putText(image, f'FPS: {fps}', (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('📸 Hướng dẫn chụp khuôn mặt', image)
            if cv2.waitKey(5) & 0xFF == 27 or self.current_target >= len(self.capture_order):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Đã chụp xong tất cả các hướng.")

# Gọi hàm
estimator = HeadPoseEstimator()
estimator.estimate_pose()
