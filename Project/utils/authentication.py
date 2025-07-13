import cv2
import numpy as np
import time
import os
import tempfile
import uuid
import platform

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Setup DeepFace environment automatically
def _setup_deepface_environment():
    """Auto-setup DeepFace environment"""
    try:
        # Get project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        models_dir = os.path.join(project_root, "Project", "models")
        
        # Set DEEPFACE_HOME to project models directory
        os.environ['DEEPFACE_HOME'] = models_dir
        
        # Create deepface directory structure if not exists
        from pathlib import Path
        deepface_dir = os.path.join(models_dir, ".deepface")
        weights_dir = os.path.join(deepface_dir, "weights")
        Path(deepface_dir).mkdir(parents=True, exist_ok=True)
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception:
        return False

# Auto-setup DeepFace environment
_setup_deepface_environment()

from collections import deque
from threading import Lock
import threading

# DeepFace only for anti-spoofing detection (not embedding)
try:
    # Ensure PyTorch is properly imported first
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded successfully")
    
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully (anti-spoofing only)")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print(f"⚠️ DeepFace/PyTorch not available: {e}")
    print("   Falling back to detector-only mode.")

# ONNX Runtime for embedding model (like recognition.py)
import onnxruntime as ort

# Core imports using recognition.py pattern
from Project.utils.Detector import FaceDetector
from Project.utils.face_utils import preprocess_face
from Project.utils.database_utils import get_user_info  # Only for metadata, not embedding search
from Project.utils.vector_store import FaissStore  # Use FAISS like recognition.py

# Import TensorRT utilities
from Project.utils.tensorrt_utils import load_optimized_model


class OptimizedFaceInference:
    """
    TURBO Face Inference System - Based on working notebook code
    Optimized for both Windows and Jetson platforms
    """
    
    def __init__(self, 
                 batch_size=1,
                 preload_models=True,
                 face_confidence_threshold=0.7,
                 antispoof_confidence_threshold=0.6,
                 optimize_for_speed=True,
                 use_tensorrt=True,
                 precision='fp16',
                 verbose=False):
        
        self.batch_size = batch_size
        self.face_confidence_threshold = face_confidence_threshold
        self.antispoof_confidence_threshold = antispoof_confidence_threshold
        self.optimize_for_speed = optimize_for_speed
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.verbose = verbose
        
        # Platform detection
        self.is_jetson = self._detect_jetson()
        self.is_windows = platform.system() == "Windows"
        
        # GPU support detection
        self.gpu_available = self._check_gpu_support()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.process_times = deque(maxlen=100)
        
        # Optimization caches
        self.frame_resize_cache = {}
        self.file_lock = Lock()
        self.models_cached = False
        
        # Temp directory for optimized I/O
        if optimize_for_speed:
            self.temp_dir = tempfile.mkdtemp(prefix="turbo_face_")
            
        # Initialize embedding model like recognition.py
        self._init_embedding_model()
        
        # Initialize FAISS vector store like recognition.py
        self._init_vector_store()
        
        # Model preloading
        if preload_models and DEEPFACE_AVAILABLE:
            self._preload_models()
        
        if self.verbose:
            print(f"🚀 OptimizedFaceInference initialized")
            print(f"   Platform: {'Jetson' if self.is_jetson else 'Windows' if self.is_windows else 'Linux'}")
            print(f"   GPU Support: {'✅' if self.gpu_available else '❌'}")
            print(f"   TURBO Mode: {'✅' if optimize_for_speed else '❌'}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Embedding Model: {'TensorRT' if self.using_tensorrt else 'ONNX'}")
            print(f"   FAISS Vector Store: {'✅' if hasattr(self, 'vector_store') else '❌'}")
    
    def _detect_jetson(self):
        """Detect if running on Jetson platform"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return 'tegra' in f.read().lower()
        except:
            return False
    
    def _check_gpu_support(self):
        """Check GPU availability"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except:
            return False
    
    def _init_embedding_model(self):
        """Initialize embedding model EXACTLY like recognition.py"""
        try:
            # Path to embedding model - FIXED: correct path structure
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Project", "models")
            model_path = os.path.join(models_dir, "inception_resnet_v1.onnx")
            
            if self.verbose:
                print(f"🔍 Loading embedding model from: {model_path}")
                print(f"📁 Model exists: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                self.using_tensorrt = False
                self.embedding_model = None
                self.embedding_session = None
                return
            
            # Initialize both attributes to None first
            self.embedding_model = None
            self.embedding_session = None
            self.using_tensorrt = False
            
            if self.use_tensorrt:
                try:
                    # Load optimized model
                    self.embedding_model = load_optimized_model('inception_resnet_v1', precision=self.precision)
                    self.using_tensorrt = True
                    if self.verbose:
                        print(f"✅ TensorRT embedding model loaded with {self.precision} precision")
                    return  # Success with TensorRT
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ TensorRT model loading failed: {e}")
                        print("🔄 Falling back to ONNX Runtime")
                    self.using_tensorrt = False
                    self.embedding_model = None
            
            # Use ONNX Runtime - EXACTLY like recognition.py
            if self.verbose:
                print("🔄 Loading ONNX Runtime session...")
            providers = ['CPUExecutionProvider']  # Same as recognition.py
            
            try:
                self.embedding_session = ort.InferenceSession(model_path, providers=providers)
                if self.verbose:
                    print(f"✅ ONNX embedding session loaded with providers: {self.embedding_session.get_providers()}")
                
                # Validate session with dummy input
                dummy_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
                self.embedding_session.run(None, {'input': dummy_input})
                if self.verbose:
                    print("✅ ONNX session validation successful")
                
            except Exception as onnx_error:
                print(f"❌ ONNX Runtime loading failed: {onnx_error}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                self.embedding_session = None
                
        except Exception as e:
            print(f"❌ Failed to initialize embedding model: {e}")
            import traceback
            traceback.print_exc()
            self.using_tensorrt = False
            self.embedding_model = None
            self.embedding_session = None
    
    def _init_vector_store(self):
        """Initialize FAISS vector store EXACTLY like recognition.py"""
        try:
            # Database path - FIXED: correct path structure
            database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Project", "database")
            if self.verbose:
                print(f"🔍 Database path: {database_path}")
                print(f"📁 Database exists: {os.path.exists(database_path)}")
            
            # Embedding dimension (512 for Inception ResNet v1) - same as recognition.py
            emb_dim = 512
            faiss_index = os.path.join(database_path, 'faiss.index')
            faiss_meta = os.path.join(database_path, 'faiss_meta.json')
            
            if self.verbose:
                print(f"🔍 FAISS index path: {faiss_index}")
                print(f"📁 FAISS index exists: {os.path.exists(faiss_index)}")
                print(f"🔍 FAISS meta path: {faiss_meta}")
                print(f"📁 FAISS meta exists: {os.path.exists(faiss_meta)}")
            
            self.vector_store = FaissStore(dim=emb_dim,
                                          index_path=faiss_index,
                                          meta_path=faiss_meta)
            print("✅ FAISS vector store initialized")
            
            # Check if vector store has data
            if hasattr(self.vector_store, 'index') and self.vector_store.index is not None:
                print(f"📊 FAISS index size: {self.vector_store.index.ntotal}")
            else:
                print("⚠️ FAISS index is empty or not loaded")
            
        except Exception as e:
            print(f"❌ Failed to initialize vector store: {e}")
            import traceback
            traceback.print_exc()
            self.vector_store = None
    
    def _preload_models(self):
        """
        Preload only DeepFace anti-spoofing models (not embedding)
        Note: Embedding model (ONNX/TensorRT) is already loaded in _init_embedding_model()
        """
        try:
            print("📦 Preloading anti-spoofing models (embedding uses ONNX/TensorRT)...")
            
            # Check PyTorch installation
            try:
                import torch
                print(f"✅ PyTorch {torch.__version__} verified")
            except ImportError:
                print("❌ PyTorch not found - installing...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'])
                import torch
                print(f"✅ PyTorch {torch.__version__} installed and loaded")
            
            # Create a dummy small image for model loading
            dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
            temp_path = os.path.join(self.temp_dir if hasattr(self, 'temp_dir') else ".", "dummy.jpg")
            cv2.imwrite(temp_path, dummy_img)
            
            # Preload face detection and anti-spoofing with error handling
            try:
                DeepFace.extract_faces(
                    img_path=temp_path,
                    anti_spoofing=True,
                    enforce_detection=False
                )
                print("✅ Anti-spoofing model loaded")
            except Exception as spoof_error:
                print(f"⚠️ Anti-spoofing model loading failed: {spoof_error}")
                # Try without anti-spoofing
                DeepFace.extract_faces(
                    img_path=temp_path,
                    anti_spoofing=False,
                    enforce_detection=False
                )
                print("⚠️ Using face detection without anti-spoofing")
            
            # Note: Using ONNX/TensorRT embedding model (already loaded in _init_embedding_model)
            print("✅ Anti-spoofing model loaded - Using ONNX/TensorRT for embedding")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            self.models_cached = True
            print("✅ Models cached successfully")
            
        except Exception as e:
            print(f"⚠️ Model preloading failed: {e}")
            print("   Will try to load models on-demand")
            self.models_cached = False
    
    def _resize_frame_optimized(self, frame, target_size=(320, 240)):
        """Optimized frame resizing with caching"""
        h, w = frame.shape[:2]
        cache_key = f"{w}x{h}_to_{target_size[0]}x{target_size[1]}"
        
        if cache_key not in self.frame_resize_cache:
            scale_x = target_size[0] / w
            scale_y = target_size[1] / h
            self.frame_resize_cache[cache_key] = (scale_x, scale_y)
        
        scale_x, scale_y = self.frame_resize_cache[cache_key]
        
        # Only resize if necessary
        if abs(scale_x - 1.0) > 0.1 or abs(scale_y - 1.0) > 0.1:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame
    
    def extract_embedding(self, face_img):
        """
        Extract face embedding using ONNX/TensorRT model EXACTLY like recognition.py
        """
        try:
            print(f"🔍 Extracting embedding from face image shape: {face_img.shape}")
            
            # Validate input
            if face_img is None or face_img.size == 0:
                print("❌ Invalid face image: None or empty")
                return None
            
            if len(face_img.shape) != 3 or face_img.shape[2] != 3:
                print(f"❌ Face image must be 3-channel color image, got shape: {face_img.shape}")
                return None
            
            # Check if embedding model is available
            if self.using_tensorrt and hasattr(self, 'embedding_model') and self.embedding_model is not None:
                print("� Using TensorRT model...")
                inp = preprocess_face(face_img)
                print(f"✅ Preprocessed input shape: {inp.shape}")
                
                emb = self.embedding_model(inp)
                # Check if the model returns a tuple/list and get the first element
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                    print(f"📦 TensorRT returned tuple, using first element")
                # The TensorRT model might return a batch, get the first item
                if len(emb.shape) > 1:
                    emb = emb[0]
                    print(f"📦 TensorRT returned batch, using first item")
                print(f"✅ TensorRT embedding shape: {emb.shape}")
                
            elif hasattr(self, 'embedding_session') and self.embedding_session is not None:
                print("🔄 Using ONNX Runtime session...")
                inp = preprocess_face(face_img)
                print(f"✅ Preprocessed input shape: {inp.shape}")
                
                result = self.embedding_session.run(None, {'input': inp})
                emb = result[0][0]  # EXACTLY like recognition.py
                print(f"✅ ONNX embedding shape: {emb.shape}")
                
            else:
                print("❌ No embedding model available")
                print(f"   - using_tensorrt: {getattr(self, 'using_tensorrt', 'NOT_SET')}")
                print(f"   - has embedding_model: {hasattr(self, 'embedding_model')}")
                print(f"   - embedding_model is None: {getattr(self, 'embedding_model', None) is None}")
                print(f"   - has embedding_session: {hasattr(self, 'embedding_session')}")
                print(f"   - embedding_session is None: {getattr(self, 'embedding_session', None) is None}")
                return None
            
            final_embedding = np.array(emb, dtype=np.float32)
            print(f"✅ Final embedding shape: {final_embedding.shape}, norm: {np.linalg.norm(final_embedding):.3f}")
            return final_embedding
            
        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search_user_by_embedding(self, embedding, threshold=0.6):
        """
        Search user in FAISS vector store EXACTLY like recognition.py
        Returns: user info dict or None
        """
        try:
            print(f"🔍 Searching FAISS with embedding shape: {embedding.shape}, threshold: {threshold}")
            
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                print("❌ Vector store not available")
                return None
            
            # Normalize embedding EXACTLY like recognition.py
            emb_norm = embedding.astype(np.float32)
            emb_norm /= np.linalg.norm(emb_norm)
            print(f"✅ Normalized embedding norm: {np.linalg.norm(emb_norm):.3f}")
            
            # Search FAISS - EXACTLY like recognition.py
            results = self.vector_store.search(emb_norm, top_k=1)
            print(f"🔍 FAISS search results: {results}")
            
            if results and len(results) > 0:
                user_id, score = results[0]
                print(f"📊 Found user_id: {user_id}, score: {score:.3f}")
                
                if score >= threshold:
                    # Get user metadata - EXACTLY like recognition.py
                    user_info = get_user_info(user_id)
                    print(f"👤 User info: {user_info}")
                    
                    result = {
                        'user_id': user_id,
                        'confidence': score,
                        'name': user_info.get('name', user_id) if user_info else user_id
                    }
                    print(f"✅ Authentication result: {result}")
                    return result
                else:
                    print(f"❌ Score {score:.3f} below threshold {threshold}")
            else:
                print("❌ No FAISS results returned")
            
            return None
            
        except Exception as e:
            print(f"❌ User search error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_frame(self, frame):
        """
        Process single frame with TURBO optimizations
        Based on working notebook implementation
        """
        start_time = time.time()
        
        try:
            # Check if DeepFace is available
            if not DEEPFACE_AVAILABLE:
                print("❌ DeepFace not available")
                return {'faces': [], 'error': 'DeepFace not available', 'process_time': 0}
            
            # TURBO optimization: resize frame for faster processing
            if self.optimize_for_speed:
                small_frame = self._resize_frame_optimized(frame, (320, 240))
                
                # Skip similar frames
                if hasattr(self, 'last_frame_hash'):
                    current_hash = hash(small_frame.tobytes())
                    if abs(current_hash - self.last_frame_hash) < 1000:
                        return self.last_results if hasattr(self, 'last_results') else {'faces': [], 'process_time': 0.001}
                    self.last_frame_hash = current_hash
                else:
                    self.last_frame_hash = hash(small_frame.tobytes())
                
                frame_to_process = small_frame
            else:
                frame_to_process = frame
            
            # Create temp file with unique name
            if self.optimize_for_speed:
                with self.file_lock:
                    temp_filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
                    temp_path = os.path.join(self.temp_dir, temp_filename)
            else:
                temp_path = f"temp_frame_{int(time.time() * 1000)}.jpg"
            
            # Save with optimized JPEG quality for speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70] if self.optimize_for_speed else []
            cv2.imwrite(temp_path, frame_to_process, encode_param)
            
            # Run DeepFace with integrated detection + anti-spoofing
            process_start = time.time()
            
            try:
                # Try with anti-spoofing first
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    anti_spoofing=True,
                    enforce_detection=False
                )
            except Exception as spoof_error:
                print(f"⚠️ Anti-spoofing failed: {spoof_error}")
                try:
                    # Fallback to detection without anti-spoofing
                    print("🔄 Trying without anti-spoofing...")
                    faces = DeepFace.extract_faces(
                        img_path=temp_path,
                        anti_spoofing=False,
                        enforce_detection=False
                    )
                    # Mark all faces as potentially real since we can't verify
                    for face_data in faces:
                        if isinstance(face_data, dict):
                            face_data['is_real'] = True
                            face_data['antispoof_score'] = 0.5  # Neutral score
                            face_data['confidence'] = 0.8  # Default confidence
                except Exception as detect_error:
                    print(f"❌ Face detection also failed: {detect_error}")
                    # Cleanup and return empty result
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return {'faces': [], 'error': f'Detection failed: {detect_error}', 'process_time': 0}
            
            process_time = time.time() - process_start
            
            # Cleanup immediately
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Scale back bounding boxes if frame was resized
            if self.optimize_for_speed and frame.shape != frame_to_process.shape:
                scale_x = frame.shape[1] / frame_to_process.shape[1]
                scale_y = frame.shape[0] / frame_to_process.shape[0]
                
                for face_data in faces:
                    if isinstance(face_data, dict) and 'facial_area' in face_data:
                        bbox = face_data['facial_area']
                        bbox['x'] = int(bbox.get('x', 0) * scale_x)
                        bbox['y'] = int(bbox.get('y', 0) * scale_y)
                        bbox['w'] = int(bbox.get('w', 0) * scale_x)
                        bbox['h'] = int(bbox.get('h', 0) * scale_y)
            
            result = {
                'faces': faces,
                'process_time': process_time,
                'timestamp': time.time(),
                'thresholds': {
                    'face_detection': self.face_confidence_threshold,
                    'anti_spoofing': self.antispoof_confidence_threshold
                }
            }
            
            # Cache result for optimization
            if self.optimize_for_speed:
                self.last_results = result
            
            # Track performance
            total_time = time.time() - start_time
            self.process_times.append(total_time)
            
            return result
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            # Check if it's the PyTorch error
            if "You must install torch" in str(e) or "pytorch" in str(e).lower():
                print("🔧 Attempting to fix PyTorch installation...")
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'torch', 'torchvision', 'torchaudio'])
                    print("✅ PyTorch reinstalled - please restart the application")
                except:
                    print("❌ Failed to reinstall PyTorch")
            
            return {'faces': [], 'error': str(e), 'process_time': 0}
    
    def update_fps(self):
        """Update FPS counter with smoothing"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 0.5:  # Update every 0.5s for responsiveness
            current_fps = self.fps_counter / elapsed
            
            # Smooth FPS calculation
            if hasattr(self, 'smoothed_fps'):
                self.current_fps = 0.7 * self.smoothed_fps + 0.3 * current_fps
                self.smoothed_fps = self.current_fps
            else:
                self.current_fps = current_fps
                self.smoothed_fps = current_fps
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def get_avg_process_time(self):
        """Get average processing time"""
        if self.process_times:
            return sum(self.process_times) / len(self.process_times)
        return 0
    
    def set_thresholds(self, face_threshold=None, antispoof_threshold=None):
        """Update detection thresholds"""
        if face_threshold is not None:
            self.face_confidence_threshold = face_threshold
        if antispoof_threshold is not None:
            self.antispoof_confidence_threshold = antispoof_threshold
    
    def cleanup(self):
        """Cleanup resources and CUDA contexts"""
        try:
            # Cleanup ONNX session
            if hasattr(self, 'embedding_session'):
                del self.embedding_session
            
            # Cleanup TensorRT model
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            
            # Cleanup temp directory
            if self.optimize_for_speed and hasattr(self, 'temp_dir'):
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup"""
        self.cleanup()


class TurboAuthenticationSystem:
    """
    TURBO Authentication System với 2 flow chính:
    1. Single Mode: Detector guidance -> Manual capture -> DeepFace verify -> Show result
    2. Continuous Mode: DeepFace realtime detection + anti-spoofing -> Verify -> Show realtime
    """
    
    def __init__(self, face_threshold=0.6, antispoof_threshold=0.6, verbose=False):
        self.face_threshold = face_threshold
        self.antispoof_threshold = antispoof_threshold
        self.verbose = verbose
        
        # Initialize inference system với settings từ notebook
        self.inference_system = OptimizedFaceInference(
            batch_size=2,
            preload_models=True,
            face_confidence_threshold=face_threshold,
            antispoof_confidence_threshold=antispoof_threshold,
            optimize_for_speed=True,  # TURBO mode
            verbose=verbose
        )
        
        # Initialize fallback detector for single mode guidance
        try:
            # Find model path for detector
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
            detector_model_path = os.path.join(models_dir, "blaze_face_short_range.tflite")
            
            if os.path.exists(detector_model_path):
                self.fallback_detector = FaceDetector(detector_model_path, use_tensorrt=False)
                self.detector_available = True
                self.use_opencv_detector = False
                if self.verbose:
                    print("✅ Fallback detector (MediaPipe BlazeFace) initialized")
            else:
                # Try with OpenCV Haar cascade detector as backup
                self.opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if not self.opencv_face_cascade.empty():
                    self.detector_available = True
                    self.use_opencv_detector = True
                    if self.verbose:
                        print("✅ Fallback detector (OpenCV Haar) initialized")
                else:
                    self.detector_available = False
                    if self.verbose:
                        print("⚠️ No face detector available - guidance disabled")
        except Exception as e:
            # Final fallback to OpenCV
            try:
                self.opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if not self.opencv_face_cascade.empty():
                    self.detector_available = True
                    self.use_opencv_detector = True
                    print("✅ Fallback detector (OpenCV Haar) initialized after error")
                else:
                    self.detector_available = False
                    print(f"⚠️ All detectors failed: {e}")
            except:
                self.detector_available = False
                print(f"⚠️ Fallback detector initialization failed: {e}")
                print("   Single mode will work without face guidance")
        
        print(f"🔐 TurboAuthenticationSystem initialized")
        print(f"   Face threshold: {face_threshold}")
        print(f"   Anti-spoof threshold: {antispoof_threshold}")
    
    def single_mode_authentication(self, timeout=None):
        """
        Single Mode Flow (Updated):
        1. Sử dụng detector model đã code sẵn để guidance - lặp vô hạn
        2. Hiển thị hướng dẫn đưa mặt vào khung (thông báo khi detect được mặt)
        3. Không sử dụng timeout - chỉ thoát bằng ESC
        4. Bấm SPACE để capture frame -> dừng real-time
        5. Đưa frame vào DeepFace để kiểm tra thật/giả
        6. Nếu thật -> xử lý embedding và so sánh database
        7. Hiển thị kết quả processed frame cho đến khi bấm nút đóng
        """
        print("\n🔍 SINGLE MODE - Authentication with Detector Guidance")
        print("   Position your face in the guide box")
        print("   Press SPACE to capture, ESC to exit")
        print("   No timeout - runs until you exit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        captured_frame = None
        verification_result = None
        show_result = False
        processed_frame = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                if not show_result:
                    # Phase 1: Continuous Detector Guidance (No timeout)
                    center_x, center_y = w // 2, h // 2
                    box_size = min(w, h) // 3
                    
                    # Draw guide box
                    cv2.rectangle(display_frame, 
                                (center_x - box_size//2, center_y - box_size//2),
                                (center_x + box_size//2, center_y + box_size//2),
                                (0, 255, 0), 3)
                    
                    # Face detection guidance using pre-built detector
                    face_detected = False
                    face_in_position = False
                    face_count = 0
                    
                    if self.detector_available:
                        try:
                            detections = []
                            
                            if hasattr(self, 'use_opencv_detector') and self.use_opencv_detector:
                                # Use OpenCV Haar cascade detector
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                faces = self.opencv_face_cascade.detectMultiScale(gray, 1.1, 4)
                                detections = [(x, y, w, h) for (x, y, w, h) in faces]
                            else:
                                # Use MediaPipe BlazeFace detector
                                detection_result = self.fallback_detector.detect_frame(frame)
                                detections = detection_result.bboxes
                            
                            if detections:
                                face_count = len(detections)
                                face_detected = True
                                
                                for detection in detections:
                                    x, y, w_face, h_face = detection
                                    face_center_x = x + w_face // 2
                                    face_center_y = y + h_face // 2
                                    
                                    # Check if face is in guide box
                                    if (center_x - box_size//2 < face_center_x < center_x + box_size//2 and
                                        center_y - box_size//2 < face_center_y < center_y + box_size//2):
                                        face_in_position = True
                                        cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
                                        cv2.putText(display_frame, "READY TO CAPTURE", 
                                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
                                        cv2.putText(display_frame, "MOVE TO CENTER", 
                                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        except Exception as e:
                            print(f"⚠️ Detector error: {e}")
                            # Fallback to simple detection
                            face_detected = False
                    
                    # Status display
                    if face_detected:
                        if face_in_position:
                            status = f"✅ FACE IN POSITION ({face_count} detected) - Press SPACE"
                            color = (0, 255, 0)
                        else:
                            status = f"⚠️ MOVE TO GREEN BOX ({face_count} detected)"
                            color = (0, 255, 255)
                    else:
                        status = "❌ NO FACE DETECTED - Show your face"
                        color = (0, 0, 255)
                    
                    cv2.putText(display_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_frame, "SPACE: Capture | ESC: Exit", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, "No timeout - runs until exit", (50, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                else:
                    # Phase 2: Show processed verification result
                    if verification_result:
                        # Use processed frame if available, otherwise original captured frame
                        if processed_frame is not None:
                            display_frame = processed_frame.copy()
                        else:
                            display_frame = captured_frame.copy()
                        
                        h, w = display_frame.shape[:2]
                        
                        if verification_result['success']:
                            user_name = verification_result.get('name', verification_result['user_id'])
                            status = f"✅ AUTHENTICATED: {user_name}"
                            color = (0, 255, 0)
                            
                            # Add detailed info
                            cv2.putText(display_frame, f"FAISS Score: {verification_result['confidence']:.3f}", 
                                      (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(display_frame, f"Face Confidence: {verification_result.get('face_confidence', 0):.3f}", 
                                      (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(display_frame, f"Anti-spoof Score: {verification_result.get('antispoof_score', 0):.3f}", 
                                      (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(display_frame, f"User ID: {verification_result['user_id']}", 
                                      (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        else:
                            status = "❌ AUTHENTICATION FAILED"
                            color = (0, 0, 255)
                            reason = verification_result.get('reason', 'Unknown')
                            cv2.putText(display_frame, f"Reason: {reason}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        cv2.putText(display_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(display_frame, "ESC: Exit | SPACE: Try Again", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Single Mode Authentication', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):
                    if not show_result:
                        # SPACE to capture in guidance phase
                        if face_in_position:  # Only capture if face is in position
                            print("📸 Capturing frame...")
                            captured_frame = frame.copy()
                            
                            # Stop real-time and verify with DeepFace
                            print("⏸️ Real-time stopped - Processing with DeepFace...")
                            verification_result, processed_frame = self._verify_frame_with_deepface(captured_frame)
                            show_result = True
                        else:
                            print("⚠️ Please position face in green box first")
                    else:
                        # SPACE to try again in result phase
                        print("🔄 Trying again...")
                        show_result = False
                        verification_result = None
                        processed_frame = None
                        captured_frame = None
            
            cv2.destroyAllWindows()
            return verification_result
            
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
            return None
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def continuous_mode_authentication(self, duration=60, skip_frames=2):
        """
        Continuous Mode Flow (Updated):
        1. Mỗi frame sử dụng DeepFace làm detector chính + anti-spoofing
        2. Khi xác định mặt thật -> crop dựa trên bounding box
        3. Đưa vào embedding model và so sánh database
        4. Hiển thị realtime liên tục cho đến khi nhấn nút đóng
        5. Có frame skip và cache để giảm lag
        """
        print("\n🔄 CONTINUOUS MODE - Real-time Authentication with DeepFace")
        print(f"   Running for {duration} seconds")
        print("   Press 'q' or ESC to exit")
        print("   Press '1'/'2' to adjust frame skip")
        print(f"   Skip frames: {skip_frames} (0=no skip, higher=faster)")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start_time = time.time()
        frame_count = 0
        # Use parameter instead of hardcoded value
        verification_results = []
        
        # Cache for performance
        last_detection_result = None
        cache_valid_frames = 3  # Use cache for 3 frames
        cache_frame_count = 0
        
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    print(f"⏰ Time limit ({duration}s) reached")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Frame skipping with cache optimization
                should_process = (frame_count % (skip_frames + 1)) == 0
                use_cache = False
                
                if not should_process and last_detection_result and cache_frame_count < cache_valid_frames:
                    # Use cached result for performance
                    results = last_detection_result
                    use_cache = True
                    cache_frame_count += 1
                elif should_process:
                    # Process new frame with DeepFace
                    results = self.inference_system.process_frame(frame)
                    last_detection_result = results
                    cache_frame_count = 0
                else:
                    # Skip this frame completely
                    results = None
                
                if results and results.get('faces'):
                    faces = results['faces']
                    
                    for face_idx, face_data in enumerate(faces):
                        if isinstance(face_data, dict) and 'facial_area' in face_data:
                            # Get confidence scores
                            face_confidence = face_data.get('confidence', 0.0)
                            antispoof_score = face_data.get('antispoof_score', 0.0)
                            is_real = face_data.get('is_real', True)
                            
                            # Get bounding box
                            facial_area = face_data['facial_area']
                            x = max(0, facial_area.get('x', 0))
                            y = max(0, facial_area.get('y', 0))
                            w_face = min(facial_area.get('w', 0), frame.shape[1] - x)
                            h_face = min(facial_area.get('h', 0), frame.shape[0] - y)
                            
                            # Apply thresholds and check if real
                            if (face_confidence >= self.face_threshold and 
                                antispoof_score >= self.antispoof_threshold and 
                                is_real and w_face > 50 and h_face > 50):
                                
                                # Face is real and above thresholds
                                # Crop face for embedding
                                face_crop = frame[y:y+h_face, x:x+w_face]
                                
                                # Extract embedding and verify with FAISS (only for new detections)
                                if not use_cache:
                                    embedding = self.inference_system.extract_embedding(face_crop)
                                    if embedding is not None:
                                        # Search in FAISS vector store like recognition.py
                                        db_result = self.inference_system.search_user_by_embedding(embedding, threshold=0.6)
                                        if db_result:
                                            # Authenticated user found
                                            verification_results.append({
                                                'timestamp': time.time(),
                                                'user_id': db_result.get('user_id', 'Unknown'),
                                                'confidence': db_result.get('confidence', 0),
                                                'face_confidence': face_confidence,
                                                'antispoof_score': antispoof_score,
                                                'bbox': (x, y, w_face, h_face),
                                                'name': db_result.get('name', db_result.get('user_id', 'Unknown'))
                                            })
                                            
                                            # Draw authenticated face
                                            cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 3)
                                            cv2.putText(display_frame, f"✅ {db_result.get('name', db_result['user_id'])}", 
                                                      (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                            cv2.putText(display_frame, f"FAISS: {db_result['confidence']:.3f}", 
                                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                            cv2.putText(display_frame, f"AS: {antispoof_score:.3f}", 
                                                      (x, y + h_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        else:
                                            # Unknown person (real but not in database)
                                            cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 255, 255), 2)
                                            cv2.putText(display_frame, "❓ Unknown Person", 
                                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                            cv2.putText(display_frame, f"Real: {antispoof_score:.3f}", 
                                                      (x, y + h_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    else:
                                        # Embedding extraction failed
                                        cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (128, 128, 128), 2)
                                        cv2.putText(display_frame, "⚠️ Embed Error", 
                                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                                else:
                                    # Using cached result - draw previous detection
                                    cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (255, 255, 0), 2)
                                    cv2.putText(display_frame, "📋 Cached", 
                                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            elif is_real and (face_confidence < self.face_threshold or antispoof_score < self.antispoof_threshold):
                                # Real face but below confidence thresholds
                                cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (100, 100, 100), 2)
                                cv2.putText(display_frame, "⚡ Low Confidence", 
                                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                                cv2.putText(display_frame, f"F:{face_confidence:.2f} A:{antispoof_score:.2f}", 
                                          (x, y + h_face + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                            
                            elif not is_real:
                                # Fake/spoof detected
                                cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 0, 255), 2)
                                cv2.putText(display_frame, "🚫 FAKE DETECTED", 
                                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.putText(display_frame, f"Spoof: {antispoof_score:.3f}", 
                                          (x, y + h_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update FPS
                self.inference_system.update_fps()
                
                # Display real-time info
                remaining = max(0, duration - elapsed)
                cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {self.inference_system.current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Verified: {len(verification_results)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Skip: {skip_frames}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Cache status
                cache_status = "CACHE" if use_cache else "LIVE"
                cache_color = (255, 255, 0) if use_cache else (0, 255, 255)
                cv2.putText(display_frame, cache_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cache_color, 2)
                
                # Controls
                cv2.putText(display_frame, "q/ESC: Exit | 1/2: Skip±", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Continuous Mode Authentication', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    print("\n👋 Exiting continuous mode...")
                    break
                elif key == ord('1'):  # Decrease skip frames
                    skip_frames = max(0, skip_frames - 1)
                    print(f"Skip frames: {skip_frames}")
                    last_detection_result = None  # Clear cache
                elif key == ord('2'):  # Increase skip frames
                    skip_frames = min(10, skip_frames + 1)
                    print(f"Skip frames: {skip_frames}")
                    last_detection_result = None  # Clear cache
            
            cv2.destroyAllWindows()
            
            # Print summary
            if verification_results:
                print(f"\n📊 Continuous Mode Summary:")
                print(f"   Total verifications: {len(verification_results)}")
                user_counts = {}
                for result in verification_results:
                    user_id = result['user_id']
                    if user_id not in user_counts:
                        user_counts[user_id] = {
                            'count': 0,
                            'avg_confidence': 0,
                            'confidences': []
                        }
                    user_counts[user_id]['count'] += 1
                    user_counts[user_id]['confidences'].append(result['confidence'])
                
                for user_id, data in user_counts.items():
                    avg_conf = sum(data['confidences']) / len(data['confidences'])
                    print(f"   {user_id}: {data['count']} times (avg conf: {avg_conf:.3f})")
            else:
                print("\n📊 No verifications recorded")
            
            return verification_results
            
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
            return verification_results
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _verify_frame_with_deepface(self, frame):
        """
        Verify frame using DeepFace detection + anti-spoofing + ONNX/TensorRT embedding + FAISS
        Returns: (verification_result, processed_frame)
        """
        try:
            print("🔍 Verifying with DeepFace...")
            
            # Process frame với DeepFace
            results = self.inference_system.process_frame(frame)
            
            if not results or not results.get('faces'):
                return {'success': False, 'reason': 'No face detected'}, None
            
            faces = results['faces']
            
            # Find best face (highest confidence real face)
            best_face = None
            best_score = 0
            
            for face_data in faces:
                if isinstance(face_data, dict) and 'facial_area' in face_data:
                    face_confidence = face_data.get('confidence', 0.0)
                    antispoof_score = face_data.get('antispoof_score', 0.0)
                    is_real = face_data.get('is_real', True)
                    
                    if (is_real and 
                        face_confidence >= self.face_threshold and 
                        antispoof_score >= self.antispoof_threshold):
                        
                        combined_score = (face_confidence + antispoof_score) / 2
                        if combined_score > best_score:
                            best_score = combined_score
                            best_face = face_data
            
            if not best_face:
                return {'success': False, 'reason': 'No valid face (fake or low confidence)'}, None
            
            # Create processed frame showing detection results
            processed_frame = frame.copy()
            
            # Crop face
            facial_area = best_face['facial_area']
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            
            # Ensure valid crop
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w < 50 or h < 50:
                return {'success': False, 'reason': 'Face too small'}, None
            
            # Draw detection on processed frame
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(processed_frame, "DETECTED FACE", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Real: {best_face.get('antispoof_score', 0):.3f}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            face_crop = frame[y:y+h, x:x+w]
            
            # Extract embedding
            embedding = self.inference_system.extract_embedding(face_crop)
            if embedding is None:
                # Mark as extraction failed on processed frame
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(processed_frame, "EMBEDDING FAILED", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return {'success': False, 'reason': 'Embedding extraction failed'}, processed_frame
            
            # Verify with FAISS vector store
            db_result = self.inference_system.search_user_by_embedding(embedding, threshold=0.6)
            if db_result:
                # Mark as authenticated on processed frame
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(processed_frame, f"AUTHENTICATED: {db_result.get('name', db_result.get('user_id', 'Unknown'))}", 
                          (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"FAISS Score: {db_result.get('confidence', 0):.3f}", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                return {
                    'success': True,
                    'user_id': db_result.get('user_id', 'Unknown'),
                    'confidence': db_result.get('confidence', 0),
                    'face_confidence': best_face.get('confidence', 0),
                    'antispoof_score': best_face.get('antispoof_score', 0),
                    'name': db_result.get('name', db_result.get('user_id', 'Unknown'))
                }, processed_frame
            else:
                # Mark as unknown on processed frame
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(processed_frame, "UNKNOWN PERSON", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(processed_frame, "Not in FAISS database", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                return {'success': False, 'reason': 'User not found in FAISS database'}, processed_frame
                
        except Exception as e:
            print(f"❌ Verification error: {e}")
            # Create error frame if possible
            error_frame = frame.copy() if frame is not None else None
            if error_frame is not None:
                h, w = error_frame.shape[:2]
                cv2.putText(error_frame, f"ERROR: {str(e)}", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return {'success': False, 'reason': f'Error: {str(e)}'}, error_frame
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'inference_system'):
            self.inference_system.cleanup()


def simple_realtime_inference(duration=100, skip_frames=0, face_threshold=0.6, antispoof_threshold=0, turbo_mode=True, verbose=False):
    """
    Compatibility function - based on working notebook code
    Automatically routes to appropriate TURBO mode
    """
    if verbose:
        print(f"\n📞 simple_realtime_inference called:")
        print(f"   duration={duration}, skip_frames={skip_frames}")
        print(f"   face_threshold={face_threshold}, antispoof_threshold={antispoof_threshold}")
        print(f"   turbo_mode={turbo_mode}")
    
    # Initialize TURBO system
    auth_system = TurboAuthenticationSystem(
        face_threshold=face_threshold,
        antispoof_threshold=antispoof_threshold,
        verbose=verbose
    )
    
    # Determine mode based on duration
    if duration <= 30:
        print("🔍 Using Single Mode (short duration)")
        result = auth_system.single_mode_authentication(timeout=duration)
    else:
        print("🔄 Using Continuous Mode (long duration)")
        result = auth_system.continuous_mode_authentication(duration=duration)
    
    # Cleanup
    auth_system.cleanup()
    
    return result


def turbo_realtime_inference(duration=100, mode="continuous", face_threshold=0.6, antispoof_threshold=0, skip_frames=2):
    """
    Main TURBO function với explicit mode selection
    """
    print(f"\n🚀 TURBO Real-time Inference:")
    print(f"   Mode: {mode.upper()}")
    print(f"   Duration: {duration}s")
    print(f"   Face threshold: {face_threshold}")
    print(f"   Anti-spoof threshold: {antispoof_threshold}")
    print(f"   Skip frames: {skip_frames}")
    
    # Initialize system
    auth_system = TurboAuthenticationSystem(
        face_threshold=face_threshold,
        antispoof_threshold=antispoof_threshold
    )
    
    try:
        if mode.lower() == "single":
            result = auth_system.single_mode_authentication(timeout=duration)
        elif mode.lower() == "continuous":
            result = auth_system.continuous_mode_authentication(duration=duration, skip_frames=skip_frames)
        else:
            print(f"❌ Unknown mode: {mode}")
            return None
        
        return result
    
    finally:
        auth_system.cleanup()


# ===== BACKWARD COMPATIBILITY =====

class FacialAuthenticationSystem:
    """
    Original FacialAuthenticationSystem class for backward compatibility
    This class wraps the TURBO system while maintaining the original interface
    """
    
    def __init__(self, detector_model_path=None, embedding_model_path=None, 
                 threshold=0.6, max_attempts=3, timeout=30, use_tensorrt=False,
                 precision="fp16", stability_level=2, camera_mode="flat", fisheye_correction=False):
        
        print("🔄 Initializing FacialAuthenticationSystem (TURBO-backed)")
        
        # Store parameters
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.camera_mode = camera_mode
        
        # Initialize TURBO system
        self.turbo_system = TurboAuthenticationSystem(
            face_threshold=threshold,
            antispoof_threshold=0.6  # Default antispoof threshold
        )
        
        # Compatibility attributes
        self.required_matches = 3
        self.liveness_required = False
        
        print(f"✅ FacialAuthenticationSystem ready (TURBO-powered)")
    
    def authenticate(self):
        """
        Single authentication - compatible with original interface
        Returns: (success: bool, user: dict or None)
        """
        print("🔍 Running single authentication (TURBO)")
        
        try:
            result = self.turbo_system.single_mode_authentication(timeout=self.timeout)
            
            if result and result.get('success'):
                # Convert TURBO result to original format
                user_data = {
                    'name': result.get('user_id', 'Unknown'),
                    'id': result.get('user_id', 'Unknown'),
                    'confidence': result.get('confidence', 0)
                }
                return True, user_data
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False, None
    
    def authenticate_continuous(self, duration=60, auth_duration=300):
        """
        Continuous authentication - compatible with original interface
        Returns: list of authenticated users
        """
        print(f"🔄 Running continuous authentication (TURBO) for {duration}s")
        
        try:
            results = self.turbo_system.continuous_mode_authentication(duration=duration)
            
            if results and isinstance(results, list):
                # Convert TURBO results to original format
                authenticated_users = []
                user_counts = {}
                
                for verification in results:
                    user_id = verification.get('user_id', 'Unknown')
                    
                    if user_id not in user_counts:
                        user_counts[user_id] = {
                            'name': user_id,
                            'id': user_id,
                            'count': 0,
                            'max_confidence': 0
                        }
                    
                    user_counts[user_id]['count'] += 1
                    user_counts[user_id]['max_confidence'] = max(
                        user_counts[user_id]['max_confidence'],
                        verification.get('confidence', 0)
                    )
                
                # Convert to list format
                for user_data in user_counts.values():
                    if user_data['count'] >= self.required_matches:
                        authenticated_users.append({
                            'name': user_data['name'],
                            'id': user_data['id'],
                            'confidence': user_data['max_confidence'],
                            'verification_count': user_data['count']
                        })
                
                return authenticated_users
            else:
                return []
                
        except Exception as e:
            print(f"❌ Continuous authentication error: {e}")
            return []
    
    def close(self):
        """Cleanup resources"""
        if self.verbose:
            print("🔄 Closing FacialAuthenticationSystem")
        if hasattr(self, 'turbo_system'):
            self.turbo_system.cleanup()


# ===== CLEAN CODE - PURE UTILITY MODULE =====
