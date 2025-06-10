import time
import cv2
import numpy as np
import onnxruntime as ort
from Project.utils.face_utils import get_face, preprocess_face
import sys
import os
from Project.utils.database_utils import find_user_by_embedding

# Import Detector from utils
from Project.utils.Detector import FaceDetector

class Recognizer:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.det_times, self.emb_times = [], []

    def detect_and_embed(self, frame):
        # Detect face
        t0 = time.perf_counter()
        face, box = get_face(frame)
        dt = time.perf_counter() - t0
        self.det_times.append(dt)
        if face is None:
            return None, None, dt, None

        # Embed
        inp = preprocess_face(face)
        t1 = time.perf_counter()
        emb = self.session.run(None, {'input': inp})[0][0]
        et = time.perf_counter() - t1
        self.emb_times.append(et)

        return face, emb, dt, et

    def summary(self):
        if self.det_times:
            return {
                'avg_detect_ms': np.mean(self.det_times)*1000,
                'avg_embed_ms': np.mean(self.emb_times)*1000
            }
        return {}

class FaceRecognitionApp:
    """
    Application for real-time face detection and recognition.
    """
    def __init__(self, detector_model_path, embedding_model_path, database_path, threshold=0.6):
        self.detector_model_path = detector_model_path
        self.embedding_model_path = embedding_model_path
        self.database_path = database_path
        self.threshold = threshold
        
        # Initialize embedding model
        self.session = ort.InferenceSession(embedding_model_path, providers=['CPUExecutionProvider'])
        
    def extract_embedding(self, face_img):
        """Extract face embedding using the embedding model."""
        inp = preprocess_face(face_img)
        emb = self.session.run(None, {'input': inp})[0][0]
        return emb
        
    def run(self):
        """Run the face recognition application."""
        print("Starting Face Recognition System...")
        
        # Initialize face detector
        detector = FaceDetector(self.detector_model_path)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Face Recognition started. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Detect faces in the frame
            detection_result = detector.detect_frame(frame)
            bboxes = detection_result.bboxes
            annotated_frame = detection_result.annotated_frame
            
            # Process each detected face
            for i, (x, y, w, h) in enumerate(bboxes):
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Get embedding
                try:
                    emb = self.extract_embedding(face_img)
                    
                    # Find matching user in database
                    user, score = find_user_by_embedding(emb, threshold=self.threshold)
                    
                    # Display result
                    if user:
                        # Found a match
                        label = f"{user['name']} ({score:.2f})"
                        color = (0, 255, 0)  # Green for match
                    else:
                        # No match found
                        label = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                        
                    # Draw label on the frame
                    cv2.putText(annotated_frame, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            # Display the frame
            cv2.imshow("Face Recognition", annotated_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Face Recognition system closed.")