import cv2
import numpy as np
import time
import onnxruntime as ort
from Project.utils.Detector import FaceDetector
from Project.utils.face_utils import preprocess_face
from Project.utils.database_utils import find_user_by_embedding

# Import TensorRT utilities
from Project.utils.tensorrt_utils import load_optimized_model
# Import fisheye correction
from Project.utils.FishEyeCalibrate import Defisheye
from Project.utils.defisheye_config import get_defisheye_params

class FacialAuthenticationSystem:
    """
    Hệ thống xác thực khuôn mặt với khung định vị.
    Người dùng đặt khuôn mặt vào khung được chỉ định để xác thực.
    
    Facial authentication system with a guide box interface.
    Users position their face within the designated box for authentication.
    """
    def __init__(self, detector_model_path, embedding_model_path, threshold=0.65, max_attempts=3, timeout=10, use_tensorrt=True, precision='fp16', stability_level=1,camera_mode='flat'):
        """
        Khởi tạo hệ thống xác thực khuôn mặt.
        
        Args:
            detector_model_path: Đường dẫn đến model phát hiện khuôn mặt
            embedding_model_path: Đường dẫn đến model embedding
            threshold: Ngưỡng nhận diện
            max_attempts: Số lần thử tối đa
            timeout: Thời gian tối đa cho mỗi lần xác thực (giây)
            use_tensorrt: Sử dụng TensorRT tăng tốc nếu có thể
            precision: Độ chính xác cho TensorRT (fp16 hoặc fp32)
            stability_level: Mức độ ổn định của xác thực (1-3, cao hơn = khó mất xác thực hơn)
            camera_mode: Chế độ camera ('flat' cho camera thường, 'fisheye' cho camera fisheye với hiệu chỉnh)
            
        Initialize the facial authentication system.
        
        Args:
            detector_model_path: Path to face detection model
            embedding_model_path: Path to embedding model
            threshold: Recognition threshold
            max_attempts: Maximum number of attempts
            timeout: Maximum time for authentication (seconds)
            use_tensorrt: Whether to use TensorRT acceleration if available
            precision: Precision to use for TensorRT models ('fp16' or 'fp32')
            stability_level: Stability level of authentication (1-3, higher = more resilient to failures)
            camera_mode: Camera mode ('flat' for normal camera, 'fisheye' for fisheye camera with correction)
        """
        # Initialize face detector with TensorRT if requested
        self.detector = FaceDetector(detector_model_path, use_tensorrt=use_tensorrt, precision=precision)
        
        # Initialize embedding model with TensorRT if requested
        if use_tensorrt:
            try:
                # Load optimized face recognition model
                self.model = load_optimized_model('inception_resnet_v1', precision=precision)
                self.using_tensorrt = True
                print(f"Using TensorRT optimized face embedding model with {precision} precision for authentication")
            except Exception as e:
                print(f"Failed to load TensorRT model: {e}")
                print("Falling back to ONNX Runtime")
                self.using_tensorrt = False
                self._init_onnx_model(embedding_model_path)
        else:
            # Use ONNX Runtime
            self.using_tensorrt = False
            self._init_onnx_model(embedding_model_path)
        
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.timeout = timeout
        
        # Kích thước khung định vị (% của kích thước frame)
        # Guide box size (% of frame size)
        self.face_box_ratio = 0.4
        
        # Authentication tracking
        self.consecutive_matches = 0
        self.consecutive_fails = 0
        self.last_match = None
        self.required_matches = 3  # Number of consecutive matches required for authentication
        
        # Configure stability based on the provided level (1-3)
        self._configure_stability(stability_level)
        
        # Liveness detection 
        self.blink_count = 0
        self.liveness_required = False  # Set to True to enable liveness detection
        self.last_eye_state = None
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        self.camera_mode = camera_mode
        if camera_mode == 'fisheye':
            # Initialize fisheye correction parameters
            self.fisheye_corrector = Defisheye(**get_defisheye_params())
            print("Fisheye correction enabled")
        else:
            self.fisheye_corrector = None
            print("Using flat camera mode (no fisheye correction)")
    def _init_onnx_model(self, embedding_model_path):
        """Initialize ONNX Runtime model as fallback"""
        # Performance optimization for Jetson Orin Nano - use CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self._is_cuda_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(embedding_model_path, providers=providers)
        
    def _is_cuda_available(self):
        """Check if CUDA is available for optimized inference"""
        try:
            providers = ort.get_available_providers()
            return 'CUDAExecutionProvider' in providers
        except:
            return False
        
    def extract_embedding(self, face_img):
        """
        Trích xuất embedding từ ảnh khuôn mặt.
        Extract embedding from face image.
        """
        inp = preprocess_face(face_img)
        
        # Run inference with TensorRT or ONNX Runtime
        if hasattr(self, 'using_tensorrt') and self.using_tensorrt:
            emb = self.model(inp)
            # Check if the model returns a tuple/list and get the first element
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            # The TensorRT model might return a batch, get the first item
            if len(emb.shape) > 1:
                emb = emb[0]
        else:
            emb = self.session.run(None, {'input': inp})[0][0]
            
        return emb
    
    def _draw_face_guide(self, frame):
        """
        Vẽ khung hướng dẫn để định vị khuôn mặt.
        Draw guide box for face positioning.
        """
        h, w = frame.shape[:2]
        
        # Tính kích thước khung dựa trên kích thước frame
        box_size = int(min(w, h) * self.face_box_ratio)
        
        # Tính toạ độ khung chính giữa
        center_x, center_y = w // 2, h // 2
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Vẽ khung định vị
        color = (0, 255, 255)  # Màu vàng
        thickness = 2
        
        # Vẽ 4 góc của khung thay vì toàn bộ khung
        corner_length = box_size // 4
        
        # Góc trên bên trái
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        
        # Góc trên bên phải
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        
        # Góc dưới bên trái
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        
        # Góc dưới bên phải
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
        return frame, (x1, y1, x2, y2)
    
    def _check_face_in_position(self, bboxes, guide_box):
        """
        Kiểm tra xem khuôn mặt có nằm trong khung định vị không.
        Check if a face is within the guide box.
        
        Returns:
            (bool, tuple, str): (Is face positioned correctly, face bbox, position feedback)
        """
        if not bboxes:
            return False, None, "No face detected"
        
        x1, y1, x2, y2 = guide_box
        guide_center_x = (x1 + x2) // 2
        guide_center_y = (y1 + y2) // 2
        guide_width = x2 - x1
        guide_height = y2 - y1
        
        best_face = None
        best_match_score = 0
        best_feedback = ""
        
        for i, (x, y, w, h) in enumerate(bboxes):
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Calculate face position relative to guide box
            center_diff_x = (face_center_x - guide_center_x) / guide_width
            center_diff_y = (face_center_y - guide_center_y) / guide_height
            
            # Calculate face size ratio relative to guide box
            size_ratio = (w * h) / (guide_width * guide_height)
            
            # Generate position feedback
            feedback = []
            
            # Horizontal position feedback
            if center_diff_x < -0.15:
                feedback.append("Move right")
            elif center_diff_x > 0.15:
                feedback.append("Move left")
                
            # Vertical position feedback
            if center_diff_y < -0.15:
                feedback.append("Move down")
            elif center_diff_y > 0.15:
                feedback.append("Move up")
                
            # Size feedback
            if size_ratio < 0.3:
                feedback.append("Move closer")
            elif size_ratio > 0.9:
                feedback.append("Move back")
                
            position_feedback = ", ".join(feedback) if feedback else "Good position"
            
            # Face should be within 50-90% of guide box size
            if size_ratio < 0.3 or size_ratio > 1.0:
                match_score = 0.3  # Poor match
            else:
                # Calculate match score (higher is better)
                match_score = 1.0 - (abs(center_diff_x) + abs(center_diff_y) + abs(0.7 - size_ratio))
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_face = (x, y, w, h)
                best_feedback = position_feedback
        
        if best_match_score > 0.7:  # Higher threshold for better positioning
            return True, best_face, "Good position"
        else:
            return False, best_face, best_feedback
    
    def _detect_blink(self, face_img):
        """
        Simple blink detection to enhance liveness detection.
        Returns True if blink detected.
        """
        # This is a placeholder for actual blink detection logic
        # In a real implementation, you would use a specialized model or algorithm
        # For now, we'll simulate blink detection with a random probability
        if np.random.random() < 0.05:  # 5% chance of detecting a blink
            return True
        return False
    
    def _show_status(self, frame, status, face_detected=False, guide_box=None, position_feedback=None):
        """
        Hiển thị trạng thái và hướng dẫn trên khung hình.
        Display status and guidance on the frame.
        """
        h, w = frame.shape[:2]
        
        # Hiển thị thông tin phản hồi về vị trí ở phía trên
        # Display position feedback at the top
        if position_feedback:
            # Determine color based on feedback
            if position_feedback == "Good position":
                feedback_color = (0, 255, 0)  # Green
            else:
                feedback_color = (0, 165, 255)  # Orange
            
            # Draw background for text
            text_size, _ = cv2.getTextSize(position_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_w, text_h = text_size
            
            cv2.rectangle(frame, (w//2 - text_w//2 - 10, 20), 
                          (w//2 + text_w//2 + 10, 20 + text_h + 20), 
                          (0, 0, 0), -1)
                          
            cv2.putText(frame, position_feedback, (w//2 - text_w//2, 20 + text_h + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
        
        # Hiển thị trạng thái ở phía dưới
        # Display status at the bottom
        
        # Set color based on status message content, not just face detection
        if "AUTHENTICATED" in status:
            status_color = (0, 255, 0)  # Green for authenticated
        elif "Unknown face" in status:
            status_color = (0, 0, 255)  # Red for unknown face
        elif "Recognizing" in status or "Verifying" in status:
            status_color = (0, 255, 255)  # Yellow for in-progress recognition
        else:
            # Default: Green if a face is detected in the correct position, red if not
            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        
        # Draw background for text
        text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        
        cv2.rectangle(frame, (w//2 - text_w//2 - 10, h - 60), 
                      (w//2 + text_w//2 + 10, h - 60 + text_h + 20), 
                      (0, 0, 0), -1)
                      
        cv2.putText(frame, status, (w//2 - text_w//2, h - 60 + text_h + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Vẽ hiệu ứng highlight cho khung khi phát hiện khuôn mặt
        # Draw highlight effect for the guide box when face is detected
        if guide_box:
            x1, y1, x2, y2 = guide_box
            
            if face_detected:
                # Draw green border for guide box when face is properly positioned
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            elif position_feedback == "Good position":
                # Draw yellow border when position is good but not yet authenticated
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            else:
                # Draw orange border when position needs adjustment
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
        
        # Display FPS for performance monitoring
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter for performance monitoring"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def _configure_stability(self, stability_level):
        """Configure stability parameters based on the stability level.
        
        Args:
            stability_level: 1=strict, 2=balanced, 3=tolerant
        """
        # Default values for balanced mode (level 2)
        self.fail_tolerance = 5      # How many consecutive fails before resetting matches
        self.auth_fail_tolerance = 10  # How many fails before invalidating authentication
        self.no_face_tolerance = 15    # How many frames without a face before resetting
        
        # Adjust based on stability level
        if stability_level == 1:  # Strict - Quick to invalidate authentication
            self.fail_tolerance = 3
            self.auth_fail_tolerance = 5
            self.no_face_tolerance = 8
        elif stability_level == 3:  # Tolerant - Very resistant to temporary failures
            self.fail_tolerance = 8
            self.auth_fail_tolerance = 15
            self.no_face_tolerance = 25
            
        # Store the current stability level
        self.stability_level = stability_level
    
    def _reset_excessive_failure_count(self):
        """Reset failure count if it's excessive to prevent display issues and overflow"""
        if self.consecutive_fails > self.auth_fail_tolerance * 2:
            previous_fails = self.consecutive_fails
            self.consecutive_fails = self.auth_fail_tolerance  # Reset to the tolerance level
            print(f"Excessive failure count reset: {previous_fails} -> {self.consecutive_fails}")
            return True
        return False
    
    def authenticate(self):
        """
        Tiến hành quá trình xác thực khuôn mặt.
        Perform facial authentication process.
        
        Returns:
            (bool, dict): (Success status, User information if successful)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return False, None
            
        attempts = 0
        start_time = time.time()
        matched_user = None
        
        # Reset authentication tracking
        self.consecutive_matches = 0
        self.consecutive_fails = 0
        self.last_match = None
        self.blink_count = 0
        
        print("Facial Authentication started. Please position your face in the guide box.")
        
        while attempts < self.max_attempts and time.time() - start_time < self.timeout:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Lật ngang frame để tạo hiệu ứng gương (Mirror effect)
            frame = cv2.flip(frame, 1)
            # Apply fisheye correction if enabled
            if self.camera_mode == 'fisheye' and self.fisheye_corrector:
                frame = self.fisheye_corrector.undistort(frame)
            # Update FPS counter
            self._update_fps()
                
            # Vẽ khung định vị (Draw guide box)
            frame_with_guide, guide_box = self._draw_face_guide(frame)
            
            # Phát hiện khuôn mặt (Detect faces)
            detection_result = self.detector.detect_frame(frame)
            bboxes = detection_result.bboxes
            
            # Kiểm tra khuôn mặt có nằm đúng vị trí không (Check if face is properly positioned)
            face_positioned, face_bbox, position_feedback = self._check_face_in_position(bboxes, guide_box)
            
            if face_positioned:
                x, y, w, h = face_bbox
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    # Check for blink if liveness detection is required
                    if self.liveness_required:
                        blink_detected = self._detect_blink(face_img)
                        if blink_detected:
                            self.blink_count += 1
                            position_feedback = f"Blink detected ({self.blink_count}/2)"
                    
                    # Lấy embedding (Extract embedding)
                    emb = self.extract_embedding(face_img)
                    
                    # So sánh với database (Compare with database)
                    user, score = find_user_by_embedding(emb, threshold=self.threshold)
                    
                    if user:
                        # Update consecutive matches tracking
                        if self.last_match == user['name']:
                            self.consecutive_matches += 1
                        else:
                            self.consecutive_matches = 1
                            self.last_match = user['name']
                        
                        self.consecutive_fails = 0
                        
                        # Check if we have enough consecutive matches and liveness check passed
                        liveness_ok = not self.liveness_required or self.blink_count >= 2
                        
                        if self.consecutive_matches >= self.required_matches and liveness_ok:
                            # Authentication successful!
                            status = f"AUTHENTICATED: {user['name']}"
                            frame = self._show_status(frame_with_guide, status, True, guide_box, position_feedback)
                            cv2.imshow("Face Authentication", frame)
                            cv2.waitKey(1500)  # Show result for 1.5 seconds
                            matched_user = user
                            break
                        else:
                            # Show progress toward authentication
                            auth_progress = f"{self.consecutive_matches}/{self.required_matches}"
                            status = f"Recognizing: {user['name']} ({auth_progress})"
                            frame = self._show_status(frame_with_guide, status, True, guide_box, position_feedback)
                    else:
                        # No match found
                        self.consecutive_fails += 1
                        self.consecutive_matches = 0
                        self.last_match = None
                        
                        if self.consecutive_fails >= 3:
                            status = f"Face not recognized ({self.consecutive_fails} fails)"
                            attempts += 1
                        else:
                            status = "Verifying..."
                except Exception as e:
                    print(f"Error processing face: {e}")
                    status = "Error processing face. Try again."
                    attempts += 1
            else:
                # Reset authentication progress when face is not in position
                self.consecutive_matches = 0
                
                # Hướng dẫn người dùng (Guide the user)
                if not bboxes:
                    status = "No face detected. Please look at the camera."
                else:
                    status = "Position your face in the guide box."
            
            # Hiển thị trạng thái (Display status)
            frame = self._show_status(frame_with_guide, status, face_positioned, guide_box, position_feedback)
            
            # Hiển thị frame (Display frame)
            cv2.imshow("Face Authentication", frame)
            
            # Kiểm tra nút thoát (Check exit key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Dọn dẹp (Cleanup)
        cap.release()
        cv2.destroyAllWindows()
        
        # Trả về kết quả (Return result)
        if matched_user:
            print(f"Authentication successful! Welcome, {matched_user['name']}.")
            return True, matched_user
        else:
            print("Authentication failed.")
            return False, None
    
    def authenticate_continuous(self, duration=30, auth_duration=10):
        """
        Continuous authentication mode for real-time face recognition.
        This mode runs continuously for the specified duration, providing real-time
        authentication feedback.
        
        Args:
            duration: Duration in seconds to run the continuous authentication (0 for infinite)
            auth_duration: How long each authentication remains valid (seconds)
            
        Returns:
            List of authenticated users during the session
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return []
            
        start_time = time.time()
        authenticated_users = []
        
        # Reset authentication tracking
        self.consecutive_matches = 0
        self.consecutive_fails = 0
        self.last_match = None
        
        authenticated_status = False
        auth_display_time = None
        auth_timeout = None  # Time when current authentication expires
        
        print(f"Continuous Authentication started. Auth valid for {auth_duration}s. Press 'q' to quit.")
        
        while True:
            # Check if duration has elapsed (if not infinite)
            if duration > 0 and time.time() - start_time > duration:
                break
                
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Update FPS counter
            self._update_fps()
                
            # Mirror effect
            frame = cv2.flip(frame, 1)
            if self.camera_mode == 'fisheye' and self.fisheye_corrector:
                frame = self.fisheye_corrector.undistort(frame)
 
            # Draw guide box
            frame_with_guide, guide_box = self._draw_face_guide(frame)
            
            # Detect faces
            detection_result = self.detector.detect_frame(frame)
            bboxes = detection_result.bboxes
            
            # Check if face is properly positioned
            face_positioned, face_bbox, position_feedback = self._check_face_in_position(bboxes, guide_box)
            
            status = "Position your face in the guide box"
            face_recognized = False
            
            # Check if current authentication has expired
            if auth_timeout and time.time() > auth_timeout:
                authenticated_status = False
                auth_timeout = None
                self.consecutive_matches = 0
                self.consecutive_fails = 0
                print("Authentication timeout expired")
                
            # Reset consecutive failures count if we haven't seen a face for a while
            # This prevents the counter from growing indefinitely
            if not authenticated_status and time.time() % 5 < 0.1:
                # Every 5 seconds, check if we need to reset excessive failure counts
                self._reset_excessive_failure_count()
                
                # If no face is positioned, fully reset the counter periodically
                if not face_positioned and self.consecutive_fails > self.auth_fail_tolerance:
                    print(f"No face positioned. Resetting consecutive failures counter from {self.consecutive_fails} to 0")
                    self.consecutive_fails = 0
            
            if face_positioned:
                x, y, w, h = face_bbox
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    # Extract embedding
                    emb = self.extract_embedding(face_img)
                    
                    # Compare with database
                    user, score = find_user_by_embedding(emb, threshold=self.threshold)
                    
                    if user:
                        # Reset excessive fail count if we've found a valid user
                        if self.consecutive_fails > self.auth_fail_tolerance:
                            print(f"Valid user detected. Resetting consecutive failures from {self.consecutive_fails} to 0")
                            self.consecutive_fails = 0
                        # Update consecutive matches tracking
                        if self.last_match == user['name']:
                            self.consecutive_matches += 1
                        else:
                            # Only reset consecutive matches if we consistently see a different user
                            # Higher stability level = more tolerant of temporary mismatches
                            if self.consecutive_fails > self.stability_level:
                                self.consecutive_matches = 1
                                self.last_match = user['name']
                            else:
                                # Temporary mismatch, maintain previous match count
                                self.consecutive_fails += 1
                                # Print diagnostic message about mismatch but maintain auth
                                print(f"Recognition mismatch: Expected {self.last_match}, got {user['name']} (fail {self.consecutive_fails}/{self.fail_tolerance})")
                        
                        # Reset consecutive fails counter on successful recognition
                        if self.last_match == user['name']:
                            self.consecutive_fails = 0
                        
                        # Check if we have enough consecutive matches
                        if self.consecutive_matches >= self.required_matches:
                            # Authentication successful!
                            if not authenticated_status:
                                authenticated_status = True
                                auth_display_time = time.time()
                                auth_timeout = time.time() + auth_duration
                                
                                # Add to authenticated users list if not already there
                                if user['name'] not in [u['name'] for u in authenticated_users]:
                                    authenticated_users.append(user)
                                    print(f"New user authenticated: {user['name']}")
                            else:
                                # Implement smarter timeout extension logic based on stability level
                                # Higher stability = less frequent timeout extensions
                                time_remaining = auth_timeout - time.time() if auth_timeout else 0
                                
                                # Calculate the threshold for timeout extension based on stability level
                                # Level 1 (strict): Extend when 70% of time has passed
                                # Level 2 (balanced): Extend when 50% of time has passed
                                # Level 3 (tolerant): Extend when 30% of time has passed
                                extension_threshold = auth_duration * (0.7 - (self.stability_level - 1) * 0.2)
                                
                                if auth_timeout is not None and time_remaining < extension_threshold:
                                    auth_timeout = time.time() + auth_duration
                                    print(f"Extended auth timeout for {user['name']} (remaining: {time_remaining:.1f}s)")
                            
                            # Draw face box with label
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{user['name']} ({score:.2f})", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Show remaining authentication time
                            if auth_timeout:
                                remaining = max(0, auth_timeout - time.time())
                                status = f"AUTHENTICATED: {user['name']} ({int(remaining)}s)"
                            else:
                                status = f"AUTHENTICATED: {user['name']}"
                        else:
                            # Show progress toward authentication
                            status = f"Recognizing: {user['name']} ({self.consecutive_matches}/{self.required_matches})"
                            
                            # Draw face box with progress label
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            cv2.putText(frame, f"{user['name']} ({score:.2f})", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        face_recognized = True
                    else:
                        # No match found
                        self.consecutive_fails += 1
                        
                        # Cap the consecutive fails to avoid excessively large numbers
                        self.consecutive_fails = min(self.consecutive_fails, self.auth_fail_tolerance + 5)
                        
                        # Don't immediately reset consecutive matches on temporary recognition failures
                        # Only reset after several consecutive failures
                        if self.consecutive_fails >= self.fail_tolerance:
                            self.consecutive_matches = 0
                            self.last_match = None
                        
                        # Draw red face box
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        if self.consecutive_fails >= 3:
                            status = f"Unknown face"
                            if authenticated_status:
                                # Only invalidate authentication after several consistent fails
                                # This adds stability against temporary recognition failures
                                if self.consecutive_fails >= self.auth_fail_tolerance:
                                    authenticated_status = False
                                    auth_timeout = None
                                    print(f"Authentication invalidated after {self.consecutive_fails} consecutive failures")
                        else:
                            status = "Verifying..."
                except Exception as e:
                    print(f"Error processing face: {e}")
                    status = "Error processing face"
                    # Don't reset matches on processing errors
            else:
                # Don't immediately reset matches/fails counters when face is temporarily out of position
                # Only if no face at all is detected, and only after a grace period
                if not bboxes:
                    # Increment fails counter but prevent it from growing excessively
                    self.consecutive_fails = min(self.consecutive_fails + 1, self.no_face_tolerance * 2)
                    
                    # Only reset authentication after a longer period with no face
                    if authenticated_status and self.consecutive_fails >= self.no_face_tolerance:
                        self.consecutive_matches = 0
                        print(f"Authentication progress reset: No face detected for {self.consecutive_fails} frames (tolerance: {self.no_face_tolerance})")
                    
                    # Show a countdown for how many more frames without a face will reset auth
                    if authenticated_status and self.consecutive_fails > (self.no_face_tolerance / 2):
                        frames_remaining = self.no_face_tolerance - self.consecutive_fails
                        frames_remaining = max(0, frames_remaining)  # Ensure it's not negative
                        status = f"No face detected ({frames_remaining} frames before reset)"
                    else:
                        status = "No face detected"
                else:
                    # Face is detected but not in position - just provide guidance
                    # Don't increase fail counter, but also don't reset it
                    status = "Position your face in the guide box"
            
            # Display authentication effect when first authenticated
            if authenticated_status and auth_display_time and time.time() - auth_display_time < 2:
                # Create overlay effect
                overlay = frame_with_guide.copy()
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame_with_guide, 0.7, 0, frame_with_guide)
                
                # Display welcome message
                welcome_text = f"Welcome, {self.last_match}!"
                text_size, _ = cv2.getTextSize(welcome_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                text_x = (frame_with_guide.shape[1] - text_size[0]) // 2
                text_y = (frame_with_guide.shape[0] + text_size[1]) // 2
                
                cv2.putText(frame_with_guide, welcome_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Display position feedback and status
            frame = self._show_status(frame_with_guide, status, face_positioned, guide_box, position_feedback)
            
            # Display time elapsed if duration is set
            if duration > 0:
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                time_text = f"Time remaining: {int(remaining)}s"
                cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Continuous Authentication", frame)
            
            # Check exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Authentication session completed. {len(authenticated_users)} users authenticated.")
        return authenticated_users
        
    def close(self):
        """
        Giải phóng tài nguyên.
        Release resources.
        """
        if hasattr(self, 'detector'):
            self.detector.close()