"""
Camera Configuration Module
Automatically detects platform and sets appropriate camera device
"""

import os
import platform
import cv2


class CameraConfig:
    """
    Camera configuration class that automatically detects platform
    and sets appropriate camera device
    """
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        self.camera_device = self._get_camera_device()
        
    def _detect_jetson(self):
        """Detect if running on Jetson platform"""
        try:
            # Check for Jetson-specific files/directories
            jetson_indicators = [
                '/proc/device-tree/model',
                '/proc/device-tree/nvidia,dtsfilename',
                '/sys/module/tegra_fuse',
                '/dev/nvhost-ctrl'
            ]
            
            for indicator in jetson_indicators:
                if os.path.exists(indicator):
                    return True
                    
            # Check for Jetson in device tree model
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'jetson' in model or 'nvidia' in model:
                        return True
            except:
                pass
                
            # Check for NVIDIA Tegra in CPU info
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'tegra' in cpuinfo or 'nvidia' in cpuinfo:
                        return True
            except:
                pass
                
            # Check environment variable (can be set manually)
            if os.environ.get('JETSON_DEVICE', '').lower() in ['true', '1', 'yes']:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _get_camera_device(self):
        """Get appropriate camera device based on platform"""
        if self.is_jetson:
            # For Jetson, prefer /dev/video0
            jetson_devices = ["/dev/video0", "/dev/video1"]
            for device in jetson_devices:
                if os.path.exists(device):
                    return device
            # Fallback to index 0 if /dev/video* not found
            return 0
        else:
            # For other platforms (Windows, Linux PC, etc.), use index 0
            return 0
    
    def get_camera_device(self):
        """Get the camera device for cv2.VideoCapture"""
        return self.camera_device
    
    def test_camera_device(self, device=None):
        """Test if camera device is working"""
        test_device = device if device is not None else self.camera_device
        
        try:
            cap = cv2.VideoCapture(test_device)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception:
            return False
    
    def get_working_camera_device(self):
        """Get a working camera device, testing multiple options"""
        # Test primary device first
        if self.test_camera_device(self.camera_device):
            return self.camera_device
            
        # Test alternative devices
        test_devices = []
        
        if self.is_jetson:
            test_devices = ["/dev/video0", "/dev/video1", "/dev/video2", 0, 1, 2]
        else:
            test_devices = [0, 1, 2, "/dev/video0"]
            
        for device in test_devices:
            if device != self.camera_device and self.test_camera_device(device):
                print(f"⚠️ Primary camera device {self.camera_device} failed, using {device}")
                return device
                
        print(f"❌ No working camera device found. Primary device: {self.camera_device}")
        return self.camera_device  # Return primary device anyway
    
    def print_info(self):
        """Print camera configuration info"""
        print(f"🎥 Camera Configuration:")
        print(f"   Platform: {'Jetson' if self.is_jetson else 'Standard'}")
        print(f"   Camera Device: {self.camera_device}")
        print(f"   Device Working: {self.test_camera_device()}")


# Global instance
_camera_config = CameraConfig()

def get_camera_device():
    """Get the appropriate camera device for current platform"""
    return _camera_config.get_camera_device()

def get_working_camera_device():
    """Get a working camera device, testing multiple options if needed"""
    return _camera_config.get_working_camera_device()

def is_jetson_platform():
    """Check if running on Jetson platform"""
    return _camera_config.is_jetson

def print_camera_info():
    """Print camera configuration information"""
    _camera_config.print_info()

def test_camera():
    """Test camera device and print results"""
    device = get_camera_device()
    working = _camera_config.test_camera_device(device)
    print(f"🎥 Camera Test:")
    print(f"   Device: {device}")
    print(f"   Status: {'✅ Working' if working else '❌ Failed'}")
    return working

# Set environment variable helper
def set_jetson_mode(enable=True):
    """Manually set Jetson mode (useful for testing)"""
    os.environ['JETSON_DEVICE'] = 'true' if enable else 'false'
    global _camera_config
    _camera_config = CameraConfig()  # Reinitialize
    print(f"🎥 Jetson mode {'enabled' if enable else 'disabled'}")
    print_camera_info()
