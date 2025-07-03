# main.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import json
from datetime import datetime
import cv2

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Project.app.registration import FaceRegistrationApp
from Project.app.recognition import FaceRecognitionApp
from Project.utils.authentication import FacialAuthenticationSystem
from Project.utils.defisheye_config import create_defisheye_instance

# Optional monitoring
try:
    from system_monitor import SystemMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Info: system_monitor.py not found. Performance monitoring will be disabled.")

from Project.utils.tensorrt_utils import is_tensorrt_available, is_pycuda_available


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Face System')
    parser.add_argument('--mode', choices=['recognition','registration','authentication'], required=True)
    parser.add_argument('--username', type=str)
    parser.add_argument('--auth-mode', choices=['single','continuous'], default='single')
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--stability', type=int, choices=[1,2,3], default=2)
    parser.add_argument('--liveness', action='store_true')
    parser.add_argument('--matches', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--timeout', type=int, default=2)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--use-tensorrt', action='store_true')
    parser.add_argument('--precision', choices=['fp16','fp32'], default='fp16')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--camera-mode', choices=['flat','fisheye'], default='flat')
    return parser.parse_args()


def check_models_exist(det_path, emb_path):
    errors = []
    if not os.path.exists(det_path):
        errors.append(f"Detector model missing: {det_path}")
    if not os.path.exists(emb_path):
        errors.append(f"Embedding model missing: {emb_path}")
    if errors:
        print("\nERROR(s):")
        for e in errors:
            print(f" - {e}")
        return False
    return True


def check_tensorrt(base, prec):
    trt_ok = is_tensorrt_available() and is_pycuda_available()
    if trt_ok:
        print("TensorRT & PyCUDA available.")
    else:
        print("TensorRT/PyCUDA not available, falling back to ONNX.")
    return trt_ok


def generate_auth_report(args, results, session_dur, out_dir=None):
    # unchanged report generation logic
    # Export results to JSON or other formats as needed
    report = {
        'mode': args.auth_mode,
        'threshold': args.threshold,
        'duration': session_dur,
        'results': results
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, f"auth_report_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")
    else:
        print(json.dumps(report, indent=2))


def run_authentication(args, det_model, emb_model, monitor=None):
    print("\n===== Facial Authentication =====")
    print(f"Mode: {args.auth_mode}, Threshold: {args.threshold}")

    # Create defisheye correction if needed
    undistorter = None
    if args.camera_mode == 'fisheye':
        undistorter = create_defisheye_instance()
        print("Fisheyecorrection enabled.")

    # Initialize authentication system
    auth_sys = FacialAuthenticationSystem(
        detector_model_path=det_model,
        embedding_model_path=emb_model,
        threshold=args.threshold,
        max_attempts=args.attempts,
        timeout=args.timeout
    )
    # Set optional attributes
    auth_sys.stability_level = args.stability
    auth_sys.use_tensorrt = args.use_tensorrt
    auth_sys.precision = args.precision
    auth_sys.camera_mode = args.camera_mode
    auth_sys.required_matches = args.matches
    auth_sys.liveness_required = args.liveness

    # Start monitor if requested
    if monitor:
        monitor.start_monitoring()

    try:
        start_time = time.time()
        auth_results = {}

        if args.auth_mode == 'continuous':
            print(f"Starting continuous authentication for {args.duration}s (Press Ctrl+C to stop early)")
            authenticated_users = auth_sys.authenticate_continuous(args.duration)
            auth_results = {'authenticated_users': authenticated_users}
            if authenticated_users:
                print("\nAuthenticated users:")
                for i, user in enumerate(authenticated_users, 1):
                    print(f"  {i}. {user['name']} (ID: {user['id']})")
            else:
                print("\nNo users were authenticated during the session.")
        else:
            print("Starting single authentication mode...")
            success, user = auth_sys.authenticate()
            auth_results = {'success': success, 'user': user if success else None}

        session_duration = time.time() - start_time
        print(f"Session time: {session_duration:.1f}s")

        if args.report:
            generate_auth_report(args, auth_results, session_duration, args.output)

    except KeyboardInterrupt:
        print("Authentication interrupted.")
    finally:
        if monitor:
            monitor.stop_monitoring()
        auth_sys.close()
        print("Facial Authentication closed.")


def main():
    args = parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    det_path = os.path.join(base, "Project/models/blaze_face_short_range.tflite")
    emb_path = os.path.join(base, "Project/models/inception_resnet_v1.onnx")
    db_path = os.path.join(base, "Project/database")

    if not check_models_exist(det_path, emb_path):
        return
    if args.use_tensorrt:
        check_tensorrt(base, args.precision)

    monitor = None
    if args.monitor and MONITORING_AVAILABLE and args.mode == 'authentication':
        monitor = SystemMonitor(output_dir=args.output, sampling_interval=0.5)

    if args.mode == 'registration':
        app = FaceRegistrationApp(det_path, emb_path, db_path, args.username)
        app.camera_mode = args.camera_mode
        app.run()
    elif args.mode == 'recognition':
        app = FaceRecognitionApp(det_path, emb_path, db_path, threshold=args.threshold)
        app.camera_mode = args.camera_mode
        app.run()
    else:
        run_authentication(args, det_path, emb_path, monitor)

# Usage examples:
# python main.py --mode registration --username Alice --camera-mode fisheye
# python main.py --mode recognition --camera-mode fisheye
# python main.py --mode authentication --username Alice --auth-mode single --camera-mode fisheye --report
# python main.py --mode authentication --username Alice --auth-mode continuous --duration 30 --camera-mode fisheye

if __name__ == '__main__':
    main()
