import os
import json
import sys
from datetime import datetime

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
    
    # Initialize FAISS database metadata if it doesn't exist
    faiss_meta_file = os.path.join(project_dir, "database", "faiss_meta.json")
    faiss_index_file = os.path.join(project_dir, "database", "faiss.index")
    
    # Update or create metadata file
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(faiss_meta_file):
        with open(faiss_meta_file, 'w') as f:
            json.dump({"user_ids": [], "last_updated": current_date}, f)
        print(f"Created FAISS metadata file: {faiss_meta_file}")
    else:
        # Update the last_updated field in existing metadata
        try:
            with open(faiss_meta_file, 'r') as f:
                metadata = json.load(f)
            metadata["last_updated"] = current_date
            with open(faiss_meta_file, 'w') as f:
                json.dump(metadata, f)
            print(f"Updated FAISS metadata last_updated timestamp: {current_date}")
        except Exception as e:
            print(f"Warning: Could not update FAISS metadata: {str(e)}")
    
    # Check for FAISS index file and create empty placeholder if needed
    if not os.path.exists(faiss_index_file):
        try:
            import faiss
            import numpy as np
            
            # Create an empty FAISS index (dimension 512 for InceptionResNet embeddings)
            dimension = 512
            
            # Check if GPU support is available
            try:
                if hasattr(faiss, 'GpuIndexFlatL2'):
                    # Create a GPU resource
                    print("Initializing FAISS with GPU support...")
                    res = faiss.StandardGpuResources()
                    empty_index = faiss.GpuIndexFlatL2(res, dimension)
                    # Convert back to CPU for storage
                    cpu_index = faiss.index_gpu_to_cpu(empty_index)
                    faiss.write_index(cpu_index, faiss_index_file)
                else:
                    # Create CPU index
                    empty_index = faiss.IndexFlatL2(dimension)
                    faiss.write_index(empty_index, faiss_index_file)
                print(f"Created empty FAISS index file: {faiss_index_file}")
            except Exception:
                # Fallback to CPU if GPU initialization fails
                empty_index = faiss.IndexFlatL2(dimension)
                faiss.write_index(empty_index, faiss_index_file)
                print(f"Created empty FAISS index file (CPU): {faiss_index_file}")
                
        except ImportError:
            print("Warning: FAISS library not found. Empty index file will not be created.")
            print("Please install FAISS: pip install faiss-cpu or pip install faiss-gpu")
        except Exception as e:
            print(f"Warning: Could not create FAISS index file: {str(e)}")
            print("You will need to initialize the FAISS index when registering the first user.")
    
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

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    required_packages = [
        "numpy", "opencv-python", "onnxruntime", "mediapipe", 
        "faiss-cpu", "pillow", "tensorflow"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            package_name = package.replace("-", "_").split(">=")[0]
            if package_name == "faiss_cpu":
                # Special case for FAISS which might be installed as faiss-gpu
                try:
                    __import__("faiss")
                except ImportError:
                    missing_packages.append(package)
            else:
                __import__(package_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Warning: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("All required dependencies are installed.")
    return True


if __name__ == "__main__":
    check_dependencies()
    ensure_project_structure()
