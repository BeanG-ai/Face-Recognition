#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Face Recognition and Authentication System
This application combines face registration, recognition, and authentication modes
into a single unified system optimized for the Jetson Orin Nano X.

Features:
- User Registration: Multi-angle face capture for accurate face models
- Face Recognition: Real-time detection and recognition of multiple faces
- Face Authentication: Secure verification with positioning guidance
- Performance Optimization: CUDA acceleration on Jetson hardware
- System Monitoring: Resource tracking for performance analysis
- TensorRT Acceleration: Optimized inference with FP16 precision
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from Project.app.registration import FaceRegistrationApp
from Project.app.recognition import FaceRecognitionApp
from Project.utils.authentication import FacialAuthenticationSystem

# Import TURBO Authentication System
from Project.utils.authentication import TurboAuthenticationSystem, turbo_realtime_inference

# Try to import system monitoring (optional)
try:
    from system_monitor import SystemMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Info: system_monitor.py not found. Performance monitoring will be disabled.")

# Check TensorRT availability
from Project.utils.tensorrt_utils import is_tensorrt_available, is_pycuda_available

def parse_args():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description='Unified Face Recognition and Authentication System')
    
    # Main application mode
    parser.add_argument('--mode', type=str, default='recognition', 
                        choices=['recognition', 'registration', 'authentication'],
                        help='Application mode: recognition, registration, or authentication')
    
    # Registration options
    parser.add_argument('--username', type=str, default=None,
                        help='Username for registration (only used in registration mode)')
    
    # Authentication options
    parser.add_argument('--auth-mode', type=str, default='single', choices=['single', 'continuous'],
                       help='Authentication mode: single (one-time) or continuous')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for continuous authentication mode in seconds (0 for infinite)')
    parser.add_argument('--auth-duration', type=int, default=10,
                       help='How long each authentication remains valid in continuous mode (seconds)')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Number of frames to skip for performance (0=no skip, 2=default, higher=faster but less responsive)')
    parser.add_argument('--stability', type=int, default=2, choices=[1, 2, 3],
                       help='Stability level: 1=strict, 2=balanced, 3=tolerant. Affects how the system handles unstable recognition')
    parser.add_argument('--liveness', action='store_true',
                       help='Enable liveness detection for enhanced security')
    parser.add_argument('--matches', type=int, default=3,
                       help='Number of consecutive matches required for authentication')
    
    # General options
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Recognition threshold (0.0-1.0)')
    parser.add_argument('--timeout', type=int, default=2,
                       help='Authentication timeout in seconds')
    parser.add_argument('--attempts', type=int, default=3,
                       help='Maximum number of authentication attempts')
    
    # TensorRT optimization options
    parser.add_argument('--use-tensorrt', action='store_true', default=False,
                       help='Use TensorRT acceleration if available')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'],
                       help='Precision to use for TensorRT models')
    
    # System monitoring and reporting
    parser.add_argument('--monitor', action='store_true',
                       help='Enable system performance monitoring')
    parser.add_argument('--output', type=str, default=None,
                       help='Directory to save monitoring data and reports')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed authentication report')
    return parser.parse_args()

def check_models_exist(detector_path, embedding_path):
    """Check if model files exist and provide helpful error messages."""
    missing_models = []
    
    if not os.path.exists(detector_path):
        missing_models.append(f"Detector model not found at: {detector_path}")
    
    if not os.path.exists(embedding_path):
        missing_models.append(f"Embedding model not found at: {embedding_path}")
    
    if missing_models:
        print("\nERROR: Required model files are missing:")
        for msg in missing_models:
            print(f"  - {msg}")
        print("\nPlease ensure the model files are in the correct location.")
        return False
    
    return True

def check_tensorrt_models(base_dir, precision):
    """Check if TensorRT optimized models exist and inform the user."""
    models_dir = os.path.join(base_dir, "Project", "models")
    
    # Check for optimized TensorRT models
    blazeface_trt = os.path.join(models_dir, f"blazeface_{precision}.trt")
    inception_trt = os.path.join(models_dir, f"inception_resnet_v1_{precision}.trt")
    
    trt_available = is_tensorrt_available() and is_pycuda_available()
    
    if trt_available:
        print("\nTensorRT and PyCUDA are available on this system.")
        
        if os.path.exists(blazeface_trt) and os.path.exists(inception_trt):
            print(f"✓ Optimized TensorRT models with {precision} precision are available.")
        else:
            print(f"! Optimized TensorRT models with {precision} precision are not found.")
            print("  The system will attempt to use ONNX models as fallback.")
            print("  To optimize models, run: python optimize_face_models.py")
    else:
        print("\nTensorRT or PyCUDA is not available on this system.")
        print("The system will use ONNX Runtime instead.")
        
    return trt_available

def generate_auth_report(args, auth_results, session_duration, output_dir=None):
    """Generate a detailed authentication report for TURBO system."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(output_dir, exist_ok=True)
    
    # Create report timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"turbo_auth_report_{timestamp_str}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    # Prepare report data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "system": "TURBO Authentication System",
        "session_info": {
            "mode": args.auth_mode,
            "duration": session_duration,
            "threshold": args.threshold,
            "liveness_enabled": args.liveness,
            "tensorrt_enabled": args.use_tensorrt,
            "precision": args.precision if args.use_tensorrt else "N/A",
            "continuous_duration": args.duration if args.auth_mode == "continuous" else "N/A",
            "timeout": args.timeout if args.auth_mode == "single" else "N/A",
        },
        "results": {}
    }
    
    # Process TURBO results
    if "turbo_result" in auth_results:
        turbo_result = auth_results["turbo_result"]
        
        if args.auth_mode == "continuous" and isinstance(turbo_result, list):
            # Continuous mode results
            user_stats = {}
            total_verifications = len(turbo_result)
            
            for verification in turbo_result:
                user_id = verification.get('user_id', 'Unknown')
                confidence = verification.get('confidence', 0)
                
                if user_id not in user_stats:
                    user_stats[user_id] = {
                        "verifications": 0,
                        "avg_confidence": 0,
                        "max_confidence": 0,
                        "min_confidence": 1
                    }
                
                user_stats[user_id]["verifications"] += 1
                user_stats[user_id]["avg_confidence"] = (
                    (user_stats[user_id]["avg_confidence"] * (user_stats[user_id]["verifications"] - 1) + confidence) /
                    user_stats[user_id]["verifications"]
                )
                user_stats[user_id]["max_confidence"] = max(user_stats[user_id]["max_confidence"], confidence)
                user_stats[user_id]["min_confidence"] = min(user_stats[user_id]["min_confidence"], confidence)
            
            report_data["results"] = {
                "total_verifications": total_verifications,
                "unique_users": len(user_stats),
                "user_statistics": user_stats,
                "success_rate": 1.0 if total_verifications > 0 else 0.0
            }
        
        elif args.auth_mode == "single":
            # Single mode results
            if turbo_result:
                report_data["results"] = {
                    "success": True,
                    "user_id": turbo_result.get('user_id', 'Unknown'),
                    "confidence": turbo_result.get('confidence', 0),
                    "similarity": turbo_result.get('similarity', 0)
                }
            else:
                report_data["results"] = {
                    "success": False,
                    "reason": "No face authenticated"
                }
    
    # Handle fallback results
    elif "authenticated_users" in auth_results:
        # Original continuous mode
        users = auth_results["authenticated_users"]
        report_data["results"] = {
            "fallback_mode": True,
            "authenticated_users": len(users) if users else 0,
            "users": users if users else []
        }
    elif "success" in auth_results:
        # Original single mode
        report_data["results"] = {
            "fallback_mode": True,
            "success": auth_results["success"],
            "user": auth_results["user"]
        }
    
    # Write report to file
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 TURBO Authentication report generated: {report_path}")
        
        # Print summary to console
        print(f"📊 Report Summary:")
        if args.auth_mode == "continuous":
            if "total_verifications" in report_data["results"]:
                print(f"   Total verifications: {report_data['results']['total_verifications']}")
                print(f"   Unique users: {report_data['results']['unique_users']}")
                print(f"   Session duration: {session_duration:.1f}s")
            elif "authenticated_users" in report_data["results"]:
                print(f"   Authenticated users (fallback): {report_data['results']['authenticated_users']}")
        else:
            if "success" in report_data["results"]:
                success = report_data["results"]["success"]
                print(f"   Authentication: {'✅ Success' if success else '❌ Failed'}")
                if success and "user_id" in report_data["results"]:
                    print(f"   User: {report_data['results']['user_id']}")
        
        return report_path
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return None
    
    # Basic report info
    report = {
        "timestamp": timestamp_str,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_duration_seconds": session_duration,
        "configuration": {
            "mode": args.auth_mode,
            "threshold": args.threshold,
            "liveness_detection": args.liveness,
            "required_matches": args.matches,
            "max_attempts": args.attempts,
            "timeout": args.timeout,
            "continuous_duration": args.duration if args.auth_mode == "continuous" else "N/A",
            "tensorrt_enabled": args.use_tensorrt,
            "precision": args.precision
        },
        "results": auth_results
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, f"auth_report_{timestamp_str}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report summary
    print(f"\n===== Authentication Report =====")
    print(f"Date: {report['date']}")
    print(f"Duration: {session_duration:.1f} seconds")
    print(f"Mode: {args.auth_mode}")
    
    # Print authentication results
    if args.auth_mode == "single":
        if auth_results.get("success", False):
            print(f"\nAuthentication SUCCESSFUL")
            print(f"User: {auth_results['user']['name']} (ID: {auth_results['user']['id']})")
        else:
            print(f"\nAuthentication FAILED")
    else:  # continuous mode
        users = auth_results.get("authenticated_users", [])
        print(f"\nAuthenticated {len(users)} user(s):")
        for i, user in enumerate(users, 1):
            print(f"  {i}. {user['name']} (ID: {user['id']})")
    
    print(f"\nDetailed report saved to: {report_path}")
    return report_path

def run_authentication(args, detector_model_path, embedding_model_path, monitor=None):
    """Run the TURBO authentication mode of the application."""
    print("\n" + "="*60)
    print("  🚀 TURBO Facial Authentication System")
    print("  Optimized for High Performance")
    print("="*60)
    
    print(f"\nMode: {'Continuous' if args.auth_mode == 'continuous' else 'Single Authentication'}")
    print(f"Security threshold: {args.threshold}")
    print(f"Required consecutive matches: {args.matches}")
    print(f"Liveness detection: {'Enabled' if args.liveness else 'Disabled'}")
    print(f"TensorRT acceleration: {'Enabled' if args.use_tensorrt else 'Disabled'}")
    if args.use_tensorrt:
        print(f"Precision: {args.precision}")
    
    if args.auth_mode == 'continuous':
        print(f"Duration: {args.duration}s {'(infinite)' if args.duration == 0 else ''}")
    else:
        print(f"Timeout: {args.timeout}s")
        print(f"Max attempts: {args.attempts}")
    
    print(f"Performance monitoring: {'Enabled' if monitor else 'Disabled'}")
    print("="*60 + "\n")
    
    # Start monitoring if enabled
    if monitor:
        monitor.start_monitoring()
    
    try:
        start_time = time.time()
        auth_results = {}
        
        # Determine appropriate settings based on arguments
        face_threshold = args.threshold
        antispoof_threshold = 0  # Default antispoof threshold
        
        if args.auth_mode == 'continuous':
            print(f"🔄 Starting TURBO continuous authentication for {args.duration}s")
            print("   Real-time processing with optimization")
            print("   Press 'q' to exit at any time")
            print(f"   Skip frames: {args.skip_frames} (0=no skip for best responsiveness)")
            
            # Use TURBO continuous mode
            result = turbo_realtime_inference(
                duration=args.duration if args.duration > 0 else 100,  # Default to 100s if infinite
                mode="continuous",
                face_threshold=face_threshold,
                antispoof_threshold=antispoof_threshold,
                skip_frames=args.skip_frames
            )
            
            auth_results = {"turbo_result": result}
            
            if result and isinstance(result, list) and len(result) > 0:
                print("\n✅ TURBO Continuous Authentication Summary:")
                # Group results by user_id
                user_counts = {}
                for verification in result:
                    user_id = verification.get('user_id', 'Unknown')
                    if user_id in user_counts:
                        user_counts[user_id] += 1
                    else:
                        user_counts[user_id] = 1
                
                for i, (user_id, count) in enumerate(user_counts.items(), 1):
                    print(f"  {i}. {user_id} (verified {count} times)")
            else:
                print("\n❌ No users were authenticated during the TURBO session.")
        
        else:
            print("🔍 Starting TURBO single authentication mode")
            print("   Optimized capture and verification")
            print("   Position your face in the guide box")
            
            # Use TURBO single mode with timeout as duration
            duration = min(args.timeout, 30)  # Max 30s for single mode
            result = turbo_realtime_inference(
                duration=duration,
                mode="single",
                face_threshold=face_threshold,
                antispoof_threshold=antispoof_threshold
            )
            
            auth_results = {"turbo_result": result}
            
            if result:
                print(f"\n✅ TURBO Authentication successful!")
                print(f"  User: {result.get('user_id', 'Unknown')}")
                if 'confidence' in result:
                    print(f"  Confidence: {result['confidence']:.3f}")
                if 'similarity' in result:
                    print(f"  Similarity: {result['similarity']:.3f}")
            else:
                print("\n❌ TURBO Authentication failed. Please try again.")
        
        session_duration = time.time() - start_time
        print(f"\n📊 TURBO Session completed in {session_duration:.1f} seconds")
        
        # Generate authentication report if requested
        if args.report:
            generate_auth_report(args, auth_results, session_duration, args.output)
        
        return auth_results
        
    except KeyboardInterrupt:
        print("\n\n⏹️ TURBO Authentication session interrupted by user.")
        return None
    except Exception as e:
        print(f"\n\n❌ TURBO Error: {e}")
        print("🔄 Falling back to original authentication system...")
        
        # Fallback to original system
        try:
            auth_system = FacialAuthenticationSystem(
                detector_model_path=detector_model_path,
                embedding_model_path=embedding_model_path,
                threshold=args.threshold,
                max_attempts=args.attempts,
                timeout=args.timeout,
                use_tensorrt=args.use_tensorrt,
                precision=args.precision,
                stability_level=args.stability
            )
            
            auth_system.required_matches = args.matches
            auth_system.liveness_required = args.liveness
            
            if args.auth_mode == 'continuous':
                authenticated_users = auth_system.authenticate_continuous(
                    duration=args.duration,
                    auth_duration=args.auth_duration
                )
                return {"authenticated_users": authenticated_users}
            else:
                success, user = auth_system.authenticate()
                return {"success": success, "user": user if success else None}
                
        except Exception as fallback_error:
            print(f"❌ Fallback system also failed: {fallback_error}")
            return None
        finally:
            if 'auth_system' in locals():
                auth_system.close()
    
    finally:
        if monitor:
            monitor.stop_monitoring()
        print("\n🚀 TURBO Authentication System session ended.")

def main():
    """Main application entry point with command-line argument handling."""
    args = parse_args()
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    detector_model_path = os.path.join(base_dir, "Project", "models", "blaze_face_short_range.tflite")
    embedding_model_path = os.path.join(base_dir, "Project", "models", "inception_resnet_v1.onnx")
    db_path = os.path.join(base_dir, "Project", "database")
    
    # Check if model files exist
    if not check_models_exist(detector_model_path, embedding_model_path):
        return
    
    # Check for TensorRT models
    if args.use_tensorrt:
        check_tensorrt_models(base_dir, args.precision)
    
    # Setup performance monitoring if requested
    monitor = None
    if args.monitor and MONITORING_AVAILABLE and args.mode == 'authentication':
        monitor = SystemMonitor(output_dir=args.output, sampling_interval=0.5)
    
    # Initialize and run the appropriate application based on the selected mode
    try:
        if args.mode == 'registration':
            print("\n=== Face Registration Mode ===")
            # if not args.username:
            #     print("ERROR: Username is required for registration mode.")
            #     print("Usage: python main.py --mode registration --username <name>")
            #     return
                
            app = FaceRegistrationApp(
                detector_model_path=detector_model_path,
                embedding_model_path=embedding_model_path,
                database_path=db_path,
                username=args.username,
                use_tensorrt=args.use_tensorrt,
                precision=args.precision
            )
            app.run()
            
        elif args.mode == 'recognition':
            print("\n=== Face Recognition Mode ===")
            app = FaceRecognitionApp(
                detector_model_path=detector_model_path,
                embedding_model_path=embedding_model_path,
                database_path=db_path,
                threshold=args.threshold,
                use_tensorrt=args.use_tensorrt,
                precision=args.precision
            )
            app.run()
            
        elif args.mode == 'authentication':
            # Run authentication mode
            run_authentication(args, detector_model_path, embedding_model_path, monitor)
            
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop monitoring if enabled
        if monitor:
            monitor.stop_monitoring()
        print("\nApplication closed.")

if __name__ == "__main__":
    main()
