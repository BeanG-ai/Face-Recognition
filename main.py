import os
import cv2
import numpy as np
import argparse
import sys

# Add the current directory to the Python path so we can find Project module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Project.app.registration import FaceRegistrationApp
from Project.app.recognition import FaceRecognitionApp

def main():
    """
    Main application entry point with command-line argument handling.
    """
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, default='recognition', 
                        choices=['recognition', 'registration'],
                        help='Application mode: recognition or registration')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for registration (only used in registration mode)')
    args = parser.parse_args()
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Project", "models", "blaze_face_short_range.tflite")
    embedding_model_path = os.path.join(base_dir, "Project", "models", "inception_resnet_v1.onnx")
    db_path = os.path.join(base_dir, "Project", "database")
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Error: Detector model not found at {model_path}")
        return
    
    if not os.path.exists(embedding_model_path):
        print(f"Error: Embedding model not found at {embedding_model_path}")
        return
    
    # Initialize the appropriate application based on the selected mode
    if args.mode == 'registration':
        username = args.username
        app = FaceRegistrationApp(
            detector_model_path=model_path,
            embedding_model_path=embedding_model_path,
            database_path=db_path,
            username=username
        )
    else:  # recognition mode
        app = FaceRecognitionApp(
            detector_model_path=model_path,
            embedding_model_path=embedding_model_path,
            database_path=db_path
        )
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()
