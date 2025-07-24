# Camera Configuration

This document explains how the camera configuration system works across different platforms.

## Overview

The camera configuration system automatically detects the platform and sets the appropriate camera device:

- **Jetson platforms**: Uses `/dev/video0` (or `/dev/video1` as fallback)
- **Other platforms**: Uses device index `0` (or `1`, `2` as fallback)

## How It Works

### Automatic Detection

The system automatically detects Jetson platforms by checking:

1. Device tree model files (`/proc/device-tree/model`)
2. Jetson-specific system files and directories
3. NVIDIA Tegra-related indicators
4. Environment variable `JETSON_DEVICE`

### Setup Scripts

Different setup scripts configure the environment appropriately:

#### `jetson_setup.sh`
- Detects Jetson platform automatically
- Sets `JETSON_DEVICE=true` environment variable
- Adds to `~/.bashrc` for persistence

#### `setup.sh` (Linux)
- Standard Linux setup
- Ensures `JETSON_DEVICE` is not set
- Uses camera device index 0

#### `setup.ps1` (Windows)
- Windows setup
- Ensures `JETSON_DEVICE` is not set
- Uses camera device index 0

## Usage

### In Python Code

```python
from Project.utils.camera_config import get_working_camera_device

# Get appropriate camera device for current platform
camera_device = get_working_camera_device()
cap = cv2.VideoCapture(camera_device)
```

### Testing Camera Configuration

```bash
# Test current configuration
python test_camera_config.py

# Test Jetson mode manually
python test_camera_config.py --test-jetson

# Test standard mode manually
python test_camera_config.py --test-standard

# Interactive camera device testing
python test_camera_config.py --interactive
```

### Manual Override

You can manually set Jetson mode:

```bash
# Enable Jetson mode
export JETSON_DEVICE=true

# Disable Jetson mode
unset JETSON_DEVICE
```

Or in Python:

```python
from Project.utils.camera_config import set_jetson_mode

# Enable Jetson mode
set_jetson_mode(True)

# Disable Jetson mode
set_jetson_mode(False)
```

## Camera Device Testing

The system automatically tests camera devices and falls back to alternatives if the primary device fails:

### Jetson Fallback Order
1. `/dev/video0`
2. `/dev/video1`
3. `/dev/video2`
4. Device index `0`
5. Device index `1`
6. Device index `2`

### Standard Platform Fallback Order
1. Device index `0`
2. Device index `1`
3. Device index `2`
4. `/dev/video0` (Linux)

## Files Modified

The following files have been updated to use the camera configuration system:

- `Project/utils/authentication.py` - TURBO authentication system
- `Project/app/recognition.py` - Face recognition app
- `Project/app/registration.py` - Face registration app
- `Project/utils/HeadPoseModel.py` - Head pose model for enrollment

## Environment Variables

- `JETSON_DEVICE`: Set to `true` on Jetson platforms, unset or `false` on others
- Automatically set by `jetson_setup.sh`
- Manually controllable for testing

## Benefits

1. **Automatic Platform Detection**: No manual configuration needed
2. **Fallback Support**: Automatically finds working camera devices
3. **Cross-Platform**: Works on Jetson, Linux, Windows, and macOS
4. **Testing Tools**: Easy to test different configurations
5. **Override Capability**: Manual control when needed

## Troubleshooting

### Camera Not Working

1. Run camera test:
   ```bash
   python test_camera_config.py --interactive
   ```

2. Check permissions (Linux):
   ```bash
   sudo usermod -a -G video $USER
   # Log out and log back in
   ```

3. List available cameras (Linux):
   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices
   ```

### Wrong Platform Detection

1. Check environment variable:
   ```bash
   echo $JETSON_DEVICE
   ```

2. Manual override:
   ```bash
   export JETSON_DEVICE=true    # For Jetson
   unset JETSON_DEVICE          # For standard platforms
   ```

3. Check detection logic:
   ```python
   from Project.utils.camera_config import print_camera_info
   print_camera_info()
   ```
