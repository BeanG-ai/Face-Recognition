import os
import json
import sys
import shutil
from datetime import datetime
from pathlib import Path

def setup_deepface_environment():
    """
    Thiết lập môi trường DeepFace portable
    Setup portable DeepFace environment
    """
    # Lấy thư mục project
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "Project")
    models_dir = os.path.join(project_dir, "models")
    deepface_dir = os.path.join(models_dir, ".deepface")
    weights_dir = os.path.join(deepface_dir, "weights")
    
    print(f"📁 Setting up DeepFace environment in: {models_dir}")
    
    # Tạo các thư mục cần thiết
    try:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(deepface_dir).mkdir(parents=True, exist_ok=True)
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating DeepFace directories: {e}")
        return False
    
    # Set environment variable
    os.environ['DEEPFACE_HOME'] = models_dir
    
    # Di chuyển models cũ nếu có (silent)
    old_deepface_path = os.path.expanduser("~/.deepface")
    if os.path.exists(old_deepface_path):
        try:
            if os.path.exists(os.path.join(old_deepface_path, "weights")):
                old_weights = os.path.join(old_deepface_path, "weights")
                moved_count = 0
                for file in os.listdir(old_weights):
                    if file.endswith(('.h5', '.pb', '.pth', '.onnx')):
                        src = os.path.join(old_weights, file)
                        dst = os.path.join(weights_dir, file)
                        if not os.path.exists(dst):  # Only copy if not exists
                            shutil.copy2(src, dst)
                            moved_count += 1
                if moved_count > 0:
                    print(f"✅ Moved {moved_count} existing DeepFace models to project directory")
        except Exception as e:
            print(f"⚠️ Warning: Could not migrate existing DeepFace models: {e}")
    
    print("✅ DeepFace environment setup completed")
    return True

def test_deepface_installation():
    """
    Test DeepFace installation and download required models
    """
    try:
        # Check if DeepFace is installed
        try:
            from deepface import DeepFace
            print("✅ DeepFace is installed")
        except ImportError:
            print("❌ DeepFace not installed. Run: pip install deepface")
            return False
        
        # Test with a dummy image to trigger model downloads
        print("📦 Testing DeepFace and downloading required models...")
        
        # Create a dummy image for testing
        import numpy as np
        import cv2
        
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_path = "temp_test_image.jpg"
        cv2.imwrite(dummy_path, dummy_image)
        
        try:
            # Test anti-spoofing (will download required models)
            face_objs = DeepFace.extract_faces(
                img_path=dummy_path, 
                anti_spoofing=True,
                enforce_detection=False  # Don't fail if no face detected in dummy image
            )
            print("✅ Anti-spoofing models downloaded successfully")
        except Exception as e:
            print(f"⚠️ Anti-spoofing test failed (models may still be downloaded): {e}")
        
        # Cleanup dummy image
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
        
        return True
        
    except Exception as e:
        print(f"❌ DeepFace test failed: {e}")
        return False

def show_deepface_models():
    """Hiển thị thông tin models DeepFace đã tải"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "Project", "models", ".deepface", "weights")
    
    if os.path.exists(weights_dir):
        models = [f for f in os.listdir(weights_dir) if f.endswith(('.h5', '.pb', '.pth', '.onnx', '.tflite'))]
        
        if models:
            print(f"\n📦 DeepFace models downloaded ({len(models)} files):")
            total_size = 0
            for model in models:
                model_path = os.path.join(weights_dir, model)
                try:
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    total_size += size_mb
                    print(f"   {model}: {size_mb:.1f} MB")
                except:
                    print(f"   {model}: Size unknown")
            print(f"💾 Total DeepFace models size: {total_size:.1f} MB")
        else:
            print("\n📦 No DeepFace models downloaded yet")
    else:
        print("\n📦 DeepFace weights directory not created yet")

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
        "faiss-cpu", "pillow", "tensorflow", "deepface"
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
            elif package_name == "opencv_python":
                __import__("cv2")
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
    
    print("✅ All required dependencies are installed.")
    return True


if __name__ == "__main__":
    print("🚀 Setting up Face Recognition Project...")
    print("=" * 50)
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Step 2: Ensure project structure
    print("\n2. Setting up project structure...")
    ensure_project_structure()
    
    # Step 3: Setup DeepFace environment
    print("\n3. Setting up DeepFace environment...")
    deepface_ok = setup_deepface_environment()
    
    # Step 4: Test DeepFace and download models
    if deps_ok and deepface_ok:
        print("\n4. Testing DeepFace installation...")
        deepface_test_ok = test_deepface_installation()
        
        # Step 5: Show DeepFace models
        if deepface_test_ok:
            show_deepface_models()
    
    print("\n" + "=" * 50)
    print("🎉 Project setup completed!")
    print("\nNext steps:")
    print("1. Place your model files in Project/models/:")
    print("   - inception_resnet_v1.onnx")
    print("   - blaze_face_short_range.tflite")
    print("2. Run: python main.py --mode registration --username <name>")
    print("3. Run: python main.py --mode authentication --auth-mode continuous")
