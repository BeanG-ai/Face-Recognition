import os
import cv2
import numpy as np
import time
from Project.utils.database_utils import add_user, find_user_by_embedding
from Project.utils.face_utils import save_face_image
import sys
import onnxruntime as ort


# Import HeadPoseModel and Detector from utils
from Project.utils.HeadPoseModel import HeadPoseEnrollment
from Project.utils.Detector import FaceDetector
from Project.utils.face_utils import preprocess_face
from Project.utils.vector_store import FaissStore

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
        return uid

class FaceRegistrationApp:
    """
    Application for registering new users with multi-angle face captures.
    """
    def __init__(self, detector_model_path, embedding_model_path, database_path, username=None):
        self.detector_model_path = detector_model_path
        self.embedding_model_path = embedding_model_path
        self.database_path = database_path
        self.username = username
        
        # Initialize embedding model
        self.session = ort.InferenceSession(embedding_model_path, providers=['CPUExecutionProvider'])
        # Determine embedding dimension from model output
        emb_dim = self.session.get_outputs()[0].shape[1]  # ADDED: get embedding dimension
         # Initialize FAISS vector store
        faiss_index = os.path.join(database_path, 'faiss.index')  # ADDED: define index path
        faiss_meta = os.path.join(database_path, 'faiss_meta.json')  # ADDED: define metadata path
        self.vector_store = FaissStore(dim=emb_dim, index_path=faiss_index, meta_path=faiss_meta)  # ADDED: init FAISS store

        # Initialize registrar with vector store
        self.registrar = Registrar(self.vector_store)  # ADDED: pass vector_store to registrar
        
        
    def extract_embedding(self, face_img):
        """Extract face embedding using the embedding model."""
        inp = preprocess_face(face_img)
        emb = self.session.run(None, {'input': inp})[0][0]
        return emb
        
    def run(self):
        """Run the face registration application."""
        print("Starting Face Registration Process...")
        
        # Get user name for directory creation
        user_name = self.username or input("Enter user name for registration: ")
        user_dir = os.path.join(self.database_path, "images", user_name)
        
        # Create directory if it doesn't exist
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Initialize face detector
        detector = FaceDetector(self.detector_model_path)
        
        # Initialize head pose enrollment
        enrollment = HeadPoseEnrollment(save_path=user_dir)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
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
        

        if enrollment.is_completed():
            print("Registration completed successfully!")

            # 1) Tạo danh sách embeddings
            all_embeddings = []
            for img_path in enrollment.captured_images.values():
                img = cv2.imread(img_path)
                if img is not None:
                    emb = self.extract_embedding(img)
                    all_embeddings.append(emb)

            if not all_embeddings:
                print("Error: Failed to generate embeddings.")
                return

            avg_embedding = np.mean(all_embeddings, axis=0)
            first_image = cv2.imread(list(enrollment.captured_images.values())[0])

            # 2) Kiểm tra xem user đã tồn tại chưa
            existing_uid, score = self.registrar.match(avg_embedding)
            if existing_uid is not None:
                print(f"User đã tồn tại (UID={existing_uid}, similarity={score:.4f}), không thêm mới.")
                
                # 3) Xóa toàn bộ ảnh đã capture (tránh sao chép)
                for img_path in enrollment.captured_images.values():
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass
                # (tuỳ chọn) xóa cả thư mục nếu rỗng:
                user_dir = os.path.dirname(list(enrollment.captured_images.values())[0])
                if os.path.isdir(user_dir) and not os.listdir(user_dir):
                    os.rmdir(user_dir)

                print("Đã xóa các ảnh tạm do user đã tồn tại.")
                return

            # 4) Nếu chưa có, đăng ký mới như trước
            self.registrar.register_new(first_image, avg_embedding, name=user_name)
            print(f"User {user_name} has been registered successfully with multi-angle face data.")
        else:
            print("Registration was not completed.")