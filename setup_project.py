import os
import json

def ensure_project_structure():
    """
    Ensures that the project directory structure exists.
    Creates necessary directories and files if they don't exist.
    """
    # Define the base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "Project")
    
    # Define directories to create
    directories = [
        os.path.join(project_dir, "database", "embeddings"),
        os.path.join(project_dir, "database", "images"),
        os.path.join(project_dir, "models"),
        os.path.join(project_dir, "utils"),
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Initialize user database JSON if it doesn't exist
    db_file = os.path.join(project_dir, "database", "user_db.json")
    if not os.path.exists(db_file):
        with open(db_file, 'w') as f:
            json.dump({"users": {}}, f)
        print(f"Created user database file: {db_file}")
      # Check for necessary model files
    model_files = [
        os.path.join(project_dir, "models", "inception_resnet_v1.onnx"),
        os.path.join(project_dir, "models", "blaze_face_short_range.tflite")
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        print("Warning: The following model files are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("Please ensure these models are downloaded and placed in the correct directories.")
    
    print("Project structure check completed.")

if __name__ == "__main__":
    ensure_project_structure()
