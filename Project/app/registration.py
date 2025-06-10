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

class Registrar:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def match(self, emb):
        user, score = find_user_by_embedding(emb, threshold=self.threshold)
        return user, score

    def register_new(self, face_img, emb, name=None, age=None, major=None, course=None, gmail=None, phone=None):
        print('=== Register New User ===')
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
        self.registrar = Registrar()
        
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
            
            # Process captured images to extract embeddings
            all_embeddings = []
            for stage, image_path in enrollment.captured_images.items():
                img = cv2.imread(image_path)
                if img is not None:
                    # Generate embedding
                    emb = self.extract_embedding(img)
                    all_embeddings.append(emb)
            
            # Average the embeddings for a more robust representation
            if all_embeddings:
                avg_embedding = np.mean(all_embeddings, axis=0)
                
                # Register the new user with the average embedding
                # Using the first captured image as the face image
                first_image = cv2.imread(list(enrollment.captured_images.values())[0])
                self.registrar.register_new(first_image, avg_embedding, name=user_name)
                
                print(f"User {user_name} has been registered successfully with multi-angle face data.")
            else:
                print("Error: Failed to generate embeddings from captured images.")
        else:
            print("Registration was not completed.")