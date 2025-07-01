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


def save_face_image(face_img, user_id):
    """
    Lưu ảnh mặt user
    """
    path = os.path.join(IMAGES_DIR, f'user_{user_id}.jpg')
    cv2.imwrite(path, face_img)
    return path
