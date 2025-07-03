"""
Configuration file for Defisheye settings.
These settings will be used globally throughout the application.
"""

# Default Defisheye configuration
DEFISHEYE_CONFIG = {
    "fov": 180,           # Field of view (degrees)
    "pfov": 120,           # Projected field of view (degrees)
    "xcenter": -1,        # X center (-1 for auto)
    "ycenter": -1,        # Y center (-1 for auto)
    "radius": -1,         # Radius (-1 for auto)
    "angle": -1,          # Rotation angle (-1 for auto)
    "dtype": "linear",    # Distortion type
    "format": "fullframe", # Output format
    "pad": 0            # Padding
}

def get_defisheye_params():
    """
    Get the current Defisheye parameters.
    
    Returns:
        dict: Dictionary with all Defisheye parameters
    """
    return DEFISHEYE_CONFIG.copy()

def update_defisheye_params(new_params):
    """
    Update the Defisheye parameters.
    
    Args:
        new_params (dict): Dictionary with parameters to update
        
    Example:
        update_defisheye_params({"fov": 140, "pfov": 100})
    """
    global DEFISHEYE_CONFIG
    DEFISHEYE_CONFIG.update(new_params)
    print(f"Updated Defisheye config: {DEFISHEYE_CONFIG}")
    
def create_defisheye_instance():
    """
    Create a new Defisheye instance with the current configuration.
    
    Returns:
        Defisheye: Configured Defisheye instance
    """
    from Project.utils.FishEyeCalibrate import Defisheye
    return Defisheye(**DEFISHEYE_CONFIG)
