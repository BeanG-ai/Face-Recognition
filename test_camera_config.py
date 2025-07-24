#!/usr/bin/env python3
"""
Camera Configuration Test Script
Tests camera configuration and platform detection
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from Project.utils.camera_config import (
    get_camera_device, 
    get_working_camera_device, 
    is_jetson_platform,
    print_camera_info,
    test_camera,
    set_jetson_mode
)

def main():
    print("🎥 Camera Configuration Test")
    print("=" * 50)
    
    # Print current configuration
    print_camera_info()
    print()
    
    # Test camera
    print("📸 Testing camera...")
    working = test_camera()
    print()
    
    # Test working camera device detection
    print("🔍 Finding working camera device...")
    working_device = get_working_camera_device()
    print(f"Working device: {working_device}")
    print()
    
    # Platform info
    print("🖥️ Platform Information:")
    print(f"   Is Jetson: {is_jetson_platform()}")
    print(f"   Primary device: {get_camera_device()}")
    print(f"   OS Environment JETSON_DEVICE: {os.environ.get('JETSON_DEVICE', 'Not set')}")
    print()
    
    # Manual testing options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-jetson":
            print("🧪 Testing Jetson mode manually...")
            set_jetson_mode(True)
            print_camera_info()
        elif sys.argv[1] == "--test-standard":
            print("🧪 Testing standard mode manually...")
            set_jetson_mode(False)
            print_camera_info()
        elif sys.argv[1] == "--interactive":
            interactive_test()
    else:
        print("💡 Usage options:")
        print("   python test_camera_config.py --test-jetson    # Test Jetson mode")
        print("   python test_camera_config.py --test-standard  # Test standard mode")
        print("   python test_camera_config.py --interactive    # Interactive test")

def interactive_test():
    """Interactive camera test"""
    import cv2
    
    print("🎮 Interactive Camera Test")
    print("=" * 30)
    
    # Test different devices
    test_devices = [0, 1, "/dev/video0", "/dev/video1"]
    
    for device in test_devices:
        print(f"\n📱 Testing device: {device}")
        try:
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"   ✅ Device {device} working - Frame shape: {frame.shape}")
                    
                    # Show a quick preview
                    response = input(f"   Show preview for device {device}? (y/n): ")
                    if response.lower() == 'y':
                        cv2.imshow(f'Camera Test - Device {device}', frame)
                        print("   Press any key to continue...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print(f"   ❌ Device {device} opened but failed to read frame")
                cap.release()
            else:
                print(f"   ❌ Device {device} failed to open")
        except Exception as e:
            print(f"   ❌ Device {device} error: {e}")
    
    print("\n✅ Interactive test completed")

if __name__ == "__main__":
    main()
