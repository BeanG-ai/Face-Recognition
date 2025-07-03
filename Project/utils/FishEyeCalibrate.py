import cv2
import numpy as np
import os
import sys
import mediapipe as mp


# Get the path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the default config if available, otherwise use hardcoded defaults
try:
    from Project.utils.defisheye_config import DEFISHEYE_CONFIG
    DEFAULT_CONFIG = DEFISHEYE_CONFIG
except (ImportError, ModuleNotFoundError):
    # Fallback defaults if config module not available
    DEFAULT_CONFIG = {
        "fov": 130,
        "pfov": 95,
        "xcenter": -1,
        "ycenter": -1,
        "radius": -1,
        "angle": -1,
        "dtype": "linear",
        "format": "fullframe",
        "pad": 0
    }

class Defisheye:
    def __init__(self, fov=None, pfov=None, xcenter=None, ycenter=None, radius=None, 
                 angle=None, dtype=None, format=None, pad=None):
        # Use provided parameters or defaults from config
        self.fov = fov if fov is not None else DEFAULT_CONFIG["fov"]
        self.pfov = pfov if pfov is not None else DEFAULT_CONFIG["pfov"]
        self.xcenter = xcenter if xcenter is not None else DEFAULT_CONFIG["xcenter"]
        self.ycenter = ycenter if ycenter is not None else DEFAULT_CONFIG["ycenter"]
        self.radius = radius if radius is not None else DEFAULT_CONFIG["radius"]
        self.angle = angle if angle is not None else DEFAULT_CONFIG["angle"]
        self.dtype = dtype if dtype is not None else DEFAULT_CONFIG["dtype"]
        self.format = format if format is not None else DEFAULT_CONFIG["format"]
        self.pad = pad if pad is not None else DEFAULT_CONFIG["pad"]

    def undistort(self, img):
        # Optional: You can use the defisheye library for more advanced correction.
        # Here is a simple defisheye using OpenCV remap for illustration.
        height, width = img.shape[:2]

        if self.xcenter == -1:
            self.xcenter = width // 2
        if self.ycenter == -1:
            self.ycenter = height // 2
        if self.radius == -1:
            self.radius = min(self.xcenter, self.ycenter, width - self.xcenter, height - self.ycenter)
        
        # Setup for remapping
        K = np.array([[self.radius, 0, self.xcenter],
                      [0, self.radius, self.ycenter],
                      [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        
        # FOV to focal length
        focal_length = self.radius / np.sin(np.radians(self.fov/2))
        K[0, 0] = K[1, 1] = focal_length

        new_K = K.copy()
        if self.pad > 0:
            new_K[0, 2] += self.pad
            new_K[1, 2] += self.pad

        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), 5)
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undistorted
