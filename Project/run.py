import os
import cv2
import argparse
import sys
from Project.app.recognition import Recognizer, FaceRecognitionApp
from Project.app.registration import Registrar, FaceRegistrationApp
from Project.utils.database_utils import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'inception_resnet_v1.onnx')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'blaze_face_short_range.tflite')

def main():
    """
    Main entry point for the face recognition application.
    Supports two modes: registration and recognition.
    """
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, default='recognition', 
                        choices=['recognition', 'registration'],
                        help='Application mode: recognition or registration')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for registration (only used in registration mode)')
    args = parser.parse_args()
    
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f'Embedding model not found: {MODEL_PATH}')
        return
        
    if not os.path.exists(DETECTOR_MODEL_PATH):
        print(f'Detector model not found: {DETECTOR_MODEL_PATH}')
        return
    
    # Initialize the appropriate application based on the selected mode
    if args.mode == 'registration':
        username = args.username or input("Enter username for registration: ")
        app = FaceRegistrationApp(
            detector_model_path=DETECTOR_MODEL_PATH,
            embedding_model_path=MODEL_PATH,
            database_path=os.path.join(BASE_DIR, "database"),
            username=username
        )
        print(f"Starting face registration for user: {username}")
    else:  # recognition mode
        app = FaceRecognitionApp(
            detector_model_path=DETECTOR_MODEL_PATH,
            embedding_model_path=MODEL_PATH,
            database_path=os.path.join(BASE_DIR, "database")
        )
        print("Starting face recognition system")
    
    # Run the application
    app.run()

if __name__ == '__main__':
    main()