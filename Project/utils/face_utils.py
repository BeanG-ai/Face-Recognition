import cv2
import numpy as np
import os
from PIL import Image
from Project.utils.database_utils import IMAGES_DIR

def preprocess_face(face_img):
    """
    Chuẩn hóa face_img để feed vào model
    """
    if face_img is None or face_img.size == 0:
        raise ValueError("Invalid face image")
        
    img = cv2.resize(face_img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis, ...]

def get_face(frame):
    """
    Simple face detection using OpenCV's Haar cascade.
    For more advanced detection, use the FaceDetector class.
    """
    # Use a simple Haar cascade for quick detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None
        
    # Get the first face
    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    return face, (x, y, w, h)

def save_face_image(face_img, user_id):
    """
    Lưu ảnh mặt user
    """
    path = os.path.join(IMAGES_DIR, f'user_{user_id}.jpg')
    cv2.imwrite(path, face_img)
    return path
