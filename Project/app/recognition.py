import time
import cv2
import numpy as np
import onnxruntime as ort
from Project.utils.face_utils import preprocess_face
import sys
import os
from Project.utils.database_utils import find_user_by_embedding
from Project.utils.database_utils import get_user_info  # UPDATED: use database_utils for metadata lookup
from Project.utils.vector_store import FaissStore  # ADDED: import FAISS vector store
from Project.utils.Detector import FaceDetector


# Import the TensorRT utilities
from Project.utils.tensorrt_utils import load_optimized_model

class Recognizer:
    def __init__(self, model_path, use_tensorrt=True, precision='fp16'):
        self.database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
        # Use optimized model if requested
        if use_tensorrt:
            try:
                # Load optimized face recognition model
                self.model = load_optimized_model('inception_resnet_v1_fp16', precision=precision)
                self.using_tensorrt = True
                print(f"Using TensorRT optimized face embedding model (FP16) with {precision} precision")
            except Exception as e:
                print(f"Failed to load TensorRT model: {e}")
                print("Falling back to ONNX Runtime")
                self.using_tensorrt = False
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            # Use ONNX Runtime
            self.using_tensorrt = False
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
        # ADDED: initialize FAISS
        # Get embedding dimension (512 for Inception ResNet v1)
        emb_dim = 512  # Default for Inception ResNet v1
        faiss_index = os.path.join(self.database_path, 'faiss.index')
        faiss_meta = os.path.join(self.database_path, 'faiss_meta.json')
        self.vector_store = FaissStore(dim=emb_dim,
                               index_path=faiss_index,
                               meta_path=faiss_meta)
        # END OF ADDED

        self.det_times, self.emb_times = [], []
        
        # Initialize the detector with TensorRT optimization
        detector_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           "models", "blaze_face_short_range.tflite")
        self.detector = FaceDetector(model_path=detector_model_path, 
                                     use_tensorrt=use_tensorrt, 
                                     precision=precision)

    def detect_and_embed(self, frame):
        # Detect face using optimized detector
        t0 = time.perf_counter()
        detection_result = self.detector.detect_frame(frame)
        bboxes = detection_result.bboxes
        dt = time.perf_counter() - t0
        self.det_times.append(dt)
        
        if not bboxes:
            return None, None, dt, None
            
        # Get the first face
        x, y, w, h = bboxes[0]
        face = frame[y:y+h, x:x+w]

        # Embed the face
        t1 = time.perf_counter()
        
        # Preprocess the face image
        inp = preprocess_face(face)
        
        # Run embedding inference with TensorRT or ONNX Runtime
        if self.using_tensorrt:
            emb = self.model(inp)
            # Check if the model returns a tuple/list and get the first element
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
        else:
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
        
    def close(self):
        """Release detector resources"""
        if hasattr(self, 'detector'):
            self.detector.close()

class FaceRecognitionApp:
    """
    Application for real-time face detection and recognition.
    """
    def __init__(self, detector_model_path, embedding_model_path, database_path, threshold=0.6, use_tensorrt=True, precision='fp16'):
        self.detector_model_path = detector_model_path
        self.embedding_model_path = embedding_model_path
        self.database_path = database_path
        self.threshold = threshold
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        if use_tensorrt:
            try:
                # Load optimized model
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
        faiss_index = os.path.join(database_path, 'faiss.index')  # ADDED: FAISS index path
        faiss_meta = os.path.join(database_path, 'faiss_meta.json')  # ADDED: FAISS metadata path
        self.vector_store = FaissStore(dim=emb_dim, index_path=faiss_index, meta_path=faiss_meta)  # ADDED: init FAISS store
        
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
        """Run the face recognition application."""
        print("Starting Face Recognition System...")
        
        # Initialize face detector with TensorRT if available
        detector = FaceDetector(self.detector_model_path, 
                                use_tensorrt=self.use_tensorrt,
                                precision=self.precision)
        
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
            frame = cv2.flip(frame, 1)
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
                    # Normalize and search FAISS
                    emb_norm = emb.astype(np.float32)  # ADDED: prepare embedding for FAISS
                    emb_norm /= np.linalg.norm(emb_norm)  # ADDED: normalize embedding
                    results = self.vector_store.search(emb_norm, top_k=1)  # ADDED: FAISS search
                    if results and results[0][1] >= self.threshold:
                        user_id, score = results[0]  # ADDED: unpack FAISS result
                        user = get_user_info(user_id)  # UPDATED: fetch user metadata
                        label = f"{user['name']} ({score:.2f})"
                        color = (0, 255, 0)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                    
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