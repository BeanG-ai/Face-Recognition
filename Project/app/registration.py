import os
import cv2
import numpy as np
import time
import shutil
from Project.utils.database_utils import add_user, find_user_by_embedding
from Project.utils.face_utils import save_face_image
import sys
import onnxruntime as ort
import uuid
from Project.utils.camera_config import get_working_camera_device  # ADDED: import camera config

# Import TensorRT utilities
from Project.utils.tensorrt_utils import load_optimized_model

# Import HeadPoseModel and Detector from utils
from Project.utils.HeadPoseModel import HeadPoseEnrollment
from Project.utils.Detector import FaceDetector
from Project.utils.face_utils import preprocess_face
from Project.utils.vector_store import FaissStore


try:
    from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

class UserInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter User Information")
        self.setFixedWidth(350)
        layout = QVBoxLayout(self)
        self.fields = {}
        for label in ["Name", "Age", "Major", "Course", "Gmail", "Phone"]:
            row = QHBoxLayout()
            lbl = QLabel(label+":")
            edit = QLineEdit()
            row.addWidget(lbl)
            row.addWidget(edit)
            layout.addLayout(row)
            self.fields[label.lower()] = edit
        btn_row = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    # def get_values(self):
    #     return {k: v.text() for k, v in self.fields.items()}
    def get_values(self):
        values = {k: v.text().strip() for k, v in self.fields.items()}
    
    # Kiểm tra bắt buộc các trường không bỏ trống
        for field, value in values.items():
            if not value:
                raise ValueError(f"Trường '{field.capitalize()}' không được để trống.")
        
        # Ràng buộc định dạng cụ thể (có thể tùy chỉnh)
        if '@' not in values['gmail']:
            raise ValueError("Gmail không hợp lệ.")
        
        if not values['phone'].isdigit():
            raise ValueError("Số điện thoại phải là số.")

        return values


class Registrar:
    def __init__(self, vector_store, similarity_threshold: float = 0.7):
        """
        vector_store: instance của FaissStore
        similarity_threshold: ngưỡng inner‐product (cosine similarity) giữa two unit embeddings
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        return emb / np.linalg.norm(emb)

    def match(self, emb: np.ndarray):
        """
        Trả về (user_id, score) nếu người gần nhất có score >= threshold,
        ngược lại trả (None, None).
        """
        emb_norm = self._normalize(emb)
        # Lưu ý: truyền theo vị trí, không dùng k=
        results = self.vector_store.search(emb_norm, 1)
        if results:
            user_id, score = results[0]
            if score >= self.similarity_threshold:
                return user_id, score
        return None, None

    def register_new(self, face_img, emb, name=None, age=None, major=None, course=None, gmail=None, phone=None):
        print('=== Register New User ===')

        emb_norm = self._normalize(emb)

        # 1) nếu đã có thì thôi
        existing_uid, score = self.match(emb_norm)
        if existing_uid is not None:
            print(f"User đã tồn tại (UID={existing_uid}, similarity={score:.4f}), không thêm mới.")
            return existing_uid
        # 2) nếu chưa có thì yêu cầu nhập thông tin
        if PYQT5_AVAILABLE:
            app = QApplication.instance() or QApplication(sys.argv)
            dialog = UserInfoDialog()
            if dialog.exec_() == QDialog.Accepted:
                try:
                    values = dialog.get_values()
                except ValueError as e:
                    print(f"Lỗi nhập liệu: {e}")
                    return None,None  # hoặc lặp lại dialog nếu muốn

                name = values['name']
                age = values['age']
                major = values['major']
                course = values['course']
                gmail = values['gmail']
                phone = values['phone']

            else:
                print("Registration cancelled by user.")
                return None,None  # hoặc lặp lại dialog nếu muốn
        else:
            print("WARNING: PyQt5 not available, falling back to terminal input.")
            if name is None:
                name = input('Name: ')
            if age is None:
                age = input('Age: ')
            if major is None:
                major = input('Major: ')
            if course is None:
                course = input('Course: ')
            if gmail is None:
                gmail = input('Gmail: ')
            if phone is None:
                phone = input('Phone: ')
        
        uid = add_user(name, age, major, course, gmail, phone, emb)
        save_face_image(face_img, uid)

        # Normalize embedding and add to FAISS
        emb_norm = emb.astype(np.float32)  # ADDED: prepare embedding for FAISS
        emb_norm /= np.linalg.norm(emb_norm)  # ADDED: normalize embedding
        self.vector_store.add(uid, emb_norm)  # ADDED: add embedding to FAISS

        print(f'Registered #{uid}: {name}')
        return uid, name

class FaceRegistrationApp:
    """
    Application for registering new users with multi-angle face captures.
    """
    def __init__(self, detector_model_path, embedding_model_path, database_path, username=None, use_tensorrt=True, precision='fp16'):
        self.detector_model_path = detector_model_path
        self.embedding_model_path = embedding_model_path
        self.database_path = database_path
        self.username = username
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        # Initialize embedding model with TensorRT if available
        if use_tensorrt:
            try:
                # Load optimized face recognition model
                self.model = load_optimized_model('inception_resnet_v1_fp16', precision=precision)
                self.using_tensorrt = True
                print(f"Using TensorRT optimized face embedding model (FP16) with {precision} precision")
                
                # For FAISS, we need embedding dimension - default is 512 for Inception ResNet v1
                emb_dim = 512
            except Exception as e:
                print(f"Failed to load TensorRT model: {e}")
                print("Falling back to ONNX Runtime")
                self.using_tensorrt = False
                self.session = ort.InferenceSession(embedding_model_path, providers=['CPUExecutionProvider'])
                emb_dim = self.session.get_outputs()[0].shape[1]
        else:
            # Use ONNX Runtime
            self.using_tensorrt = False
            self.session = ort.InferenceSession(embedding_model_path, providers=['CPUExecutionProvider'])
            emb_dim = self.session.get_outputs()[0].shape[1]
            
        # Initialize FAISS vector store
        faiss_index = os.path.join(database_path, 'faiss.index')  # ADDED: define index path
        faiss_meta = os.path.join(database_path, 'faiss_meta.json')  # ADDED: define metadata path
        self.vector_store = FaissStore(dim=emb_dim, index_path=faiss_index, meta_path=faiss_meta)  # ADDED: init FAISS store

        # Initialize registrar with vector store
        self.registrar = Registrar(self.vector_store)  # ADDED: pass vector_store to registrar
        
    def extract_embedding(self, face_img):
        """Extract face embedding using the embedding model."""
        inp = preprocess_face(face_img)
        
        # Run inference with TensorRT or ONNX Runtime
        if self.using_tensorrt:
            emb = self.model(inp)
            # Check if the model returns a tuple/list and get the first element
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            # The TensorRT model might return a batch, get the first item
            if len(emb.shape) > 1:
                emb = emb[0]
        else:
            # Convert input to float16 for FP16 model
            inp_fp16 = inp.astype(np.float16)
            emb = self.session.run(None, {'input': inp_fp16})[0][0]
            
        return emb
        
    def run(self):
        """Run the face registration application."""
        print("Starting Face Registration Process...")
        # Tạo thư mục tạm cho ảnh
        temp_id = str(uuid.uuid4())  # Hoặc dùng time.time(), random…
        user_dir = os.path.join(self.database_path, "images", temp_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        # Create directory if it doesn't exist
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Initialize face detector with TensorRT if available
        detector = FaceDetector(self.detector_model_path, 
                               use_tensorrt=self.use_tensorrt, 
                               precision=self.precision)
        
        # Initialize head pose enrollment
        enrollment = HeadPoseEnrollment(save_path=user_dir)
        
        # Open webcam with appropriate device for platform
        camera_device = get_working_camera_device()
        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            print(f"Error: Could not open webcam device: {camera_device}")
            return
        
        print("Face Registration started. Follow the on-screen instructions.")
        
        while cap.isOpened() and not enrollment.is_completed():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
    
            # Process frame with head pose enrollment
            ui_frame = enrollment.process_frame(frame)
            
            # Display the UI frame
            cv2.imshow("Face Registration", ui_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Once enrollment is completed or terminated
        cap.release()
        detector.close()
        enrollment.close()
        cv2.destroyAllWindows()
        
        # Nếu người dùng thoát giữa chừng hoặc không hoàn thành đăng ký
        if not enrollment.is_completed():
            print("Da thoat chuong trinh. Cac anh da chup:", enrollment.captured_images)
            
            # Xóa ảnh đã chụp nếu có
            if enrollment.captured_images:
                user_dir = os.path.dirname(list(enrollment.captured_images.values())[0])
                for img_path in enrollment.captured_images.values():
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except OSError:
                        pass
                
                # Xóa thư mục nếu rỗng
                try:
                    if os.path.isdir(user_dir) and not os.listdir(user_dir):
                        os.rmdir(user_dir)
                except OSError:
                    pass
                    
                print("✅ Đã xóa dữ liệu tạm do thoát giữa chừng.")
            return
        
        if enrollment.is_completed():
            print("🎉 Enrollment completed successfully!")

            image_paths = list(enrollment.captured_images.values())
            if not image_paths:
                print("❌ Error: No face images were captured!")
                return

            # Lưu thư mục tạm để dễ dàng xóa nếu có lỗi
            user_dir = os.path.dirname(image_paths[0])
            
            # Hàm tiện ích xóa ảnh và thư mục tạm
            def cleanup_temp_images():
                print("🧹 Cleaning up temporary images...")
                for img_path in enrollment.captured_images.values():
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            print(f"  - Deleted: {os.path.basename(img_path)}")
                    except OSError as e:
                        print(f"  - Failed to delete {os.path.basename(img_path)}: {e}")
                        
                # Xóa thư mục nếu rỗng
                try:
                    if os.path.isdir(user_dir) and not os.listdir(user_dir):
                        os.rmdir(user_dir)
                        print(f"  - Removed empty directory: {os.path.basename(user_dir)}")
                except OSError as e:
                    print(f"  - Failed to remove directory {os.path.basename(user_dir)}: {e}")

            # 1) Tạo danh sách embeddings
            all_embeddings = []
            for img_path in enrollment.captured_images.values():
                img = cv2.imread(img_path)
                if img is not None:
                    try:
                        emb = self.extract_embedding(img)
                        all_embeddings.append(emb)
                    except Exception as e:
                        print(f"❌ Error extracting embedding: {e}")
            
            if not all_embeddings:
                print("❌ Error: Failed to generate embeddings.")
                cleanup_temp_images()
                return

            avg_embedding = np.mean(all_embeddings, axis=0)
            
            first_image = cv2.imread(image_paths[0])
            if first_image is None:
                print("❌ Error: Failed to read the first image.")
                cleanup_temp_images()
                return

            # 2) Kiểm tra xem user đã tồn tại chưa
            existing_uid, score = self.registrar.match(avg_embedding)
            if existing_uid is not None:
                print(f"⚠️ User đã tồn tại (UID={existing_uid}, similarity={score:.4f}), không thêm mới.")
                
                # Xóa ảnh tạm khi user đã tồn tại
                cleanup_temp_images()
                print("✅ Đã xóa các ảnh tạm do user đã tồn tại.")
                return

            # 4) Nếu chưa có, đăng ký mới
            try:
                new_user_id, real_name = self.registrar.register_new(first_image, avg_embedding)
                if new_user_id and real_name:
                    # Xử lý tên thư mục không dấu, không chứa ký tự đặc biệt
                    folder_name = real_name.strip().replace(" ", "_")
                    real_user_dir = os.path.join(self.database_path, "images", folder_name)
                    
                    # Đảm bảo thư mục đích không tồn tại trước khi đổi tên
                    if os.path.exists(real_user_dir):
                        print(f"⚠️ Thư mục đích {folder_name} đã tồn tại, tạo tên duy nhất...")
                        folder_name = f"{folder_name}_{str(int(time.time()))}"
                        real_user_dir = os.path.join(self.database_path, "images", folder_name)
                    
                    try:
                        os.rename(user_dir, real_user_dir)
                        print(f"✅ User '{real_name}' đã được đăng ký với folder ảnh: {folder_name}")
                        print(f"📸 Đã lưu {len(enrollment.captured_images)} ảnh vào thư mục: {folder_name}")
                    except OSError as e:
                        print(f"❌ Lỗi khi đổi tên thư mục: {e}")
                        # Không xóa ảnh vì chúng vẫn hợp lệ, chỉ có lỗi đổi tên
                else:
                    print("❌ Không nhận được thông tin người dùng hợp lệ.")
                    cleanup_temp_images()
            except Exception as e:
                print(f"❌ Lỗi khi đăng ký người dùng: {e}")
                cleanup_temp_images()
                print("✅ Đã xóa toàn bộ dữ liệu tạm thời do lỗi đăng ký.")
           

