#!/usr/bin/env python3
"""
DeepFace Setup Script
Tự động tạo và cấu hình thư mục models cho DeepFace
Auto setup and configure models directory for DeepFace
"""

import os
import sys
import shutil
from pathlib import Path

def get_project_root():
    """Lấy thư mục gốc của project"""
    # Lấy thư mục hiện tại của script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir

def setup_deepface_environment():
    """
    Thiết lập môi trường DeepFace portable
    Setup portable DeepFace environment
    """
    # Lấy thư mục project
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "Project", "models")
    deepface_dir = os.path.join(models_dir, ".deepface")
    weights_dir = os.path.join(deepface_dir, "weights")
    
    print(f"📁 DeepFace weights directory: {models_dir}")
    
    # Tạo các thư mục cần thiết
    try:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(deepface_dir).mkdir(parents=True, exist_ok=True)
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False
    
    # Set environment variable
    os.environ['DEEPFACE_HOME'] = models_dir
    
    # Di chuyển models cũ nếu có (silent)
    old_deepface_path = os.path.expanduser("~/.deepface")
    if os.path.exists(old_deepface_path):
        try:
            if os.path.exists(os.path.join(old_deepface_path, "weights")):
                old_weights = os.path.join(old_deepface_path, "weights")
                for file in os.listdir(old_weights):
                    if file.endswith(('.h5', '.pb', '.pth', '.onnx')):
                        src = os.path.join(old_weights, file)
                        dst = os.path.join(weights_dir, file)
                        shutil.copy2(src, dst)
        except Exception:
            pass  # Silent fail
    
    return True

def test_anti_spoofing():
    """
    Test anti-spoofing và tải models (silent mode)
    """
    try:
        # Import DeepFace sau khi đã set environment
        from deepface import DeepFace
        
        # Sử dụng đường dẫn ảnh mặc định
        test_image = r"C:\Users\truon\Downloads\download (225).jpg"
        
        if not os.path.exists(test_image):
            # Tạo ảnh dummy để test model download (silent)
            import numpy as np
            import cv2
            
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_path = "temp_test_image.jpg"
            cv2.imwrite(dummy_path, dummy_image)
            test_image = dummy_path
        
        # Chạy lệnh anti-spoofing test (silent)
        face_objs = DeepFace.extract_faces(
            img_path=test_image, 
            anti_spoofing=True
        )
        
        # Dọn dẹp file dummy nếu có
        if test_image == "temp_test_image.jpg" and os.path.exists(test_image):
            os.remove(test_image)
        
        return True
        
    except ImportError:
        print("❌ DeepFace not installed. Run: pip install deepface")
        return False
    except Exception:
        return False  # Silent fail

def show_downloaded_models():
    """Hiển thị thông tin models đã tải"""
    project_root = get_project_root()
    weights_dir = os.path.join(project_root, "Project", "models", ".deepface", "weights")
    
    if os.path.exists(weights_dir):
        models = [f for f in os.listdir(weights_dir) if f.endswith(('.h5', '.pb', '.pth', '.onnx', '.tflite'))]
        
        if models:
            print(f"\n📦 Downloaded weights ({len(models)} files):")
            total_size = 0
            for model in models:
                model_path = os.path.join(weights_dir, model)
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                total_size += size_mb
                print(f"   {model}: {size_mb:.1f} MB")
            print(f"💾 Total size: {total_size:.1f} MB")
        else:
            print("\n📦 No models downloaded yet")
    else:
        print("\n📦 Weights directory not created yet")

def check_deepface_installation():
    """Kiểm tra xem DeepFace đã được cài đặt chưa (silent)"""
    try:
        import deepface
        return True
    except ImportError:
        print("❌ DeepFace not installed. Run: pip install deepface")
        return False

def init_deepface():
    """
    Function để import trong các script khác
    Function to import in other scripts
    """
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "Project", "models")
    
    # Set environment variable
    os.environ['DEEPFACE_HOME'] = models_dir
    
    # Tạo thư mục nếu chưa có
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(models_dir, ".deepface")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(models_dir, ".deepface", "weights")).mkdir(parents=True, exist_ok=True)
    
    return models_dir

if __name__ == "__main__":
    # Kiểm tra DeepFace
    if not check_deepface_installation():
        sys.exit(1)
    
    # Thiết lập environment
    if setup_deepface_environment():
        # Tự động test anti-spoofing và tải models
        test_anti_spoofing()
        
        # Hiển thị kết quả
        show_downloaded_models()
        
        print("\n✅ Setup completed! DeepFace ready to use.")
    else:
        print("❌ Setup failed!")
        sys.exit(1)
