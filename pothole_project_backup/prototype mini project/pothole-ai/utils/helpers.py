"""
helpers.py
==========
Utility functions for validation, file handling, and timestamp generation.

Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System
"""

import os
from datetime import datetime
from typing import Tuple, Optional, Any
from PIL import Image


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_directories(*dirs: str) -> None:
    """
    Ensures that the required directories exist on disk.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_current_timestamp() -> str:
    """
    Returns the current local date and time in readable format.
    Example: 2026-09-26 11:50:30
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_image_file(uploaded_file) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded image file type and integrity.

    Returns:
        Tuple:
        - is_valid: bool
        - error_message: str if invalid, else None
        - pil_image: PIL Image object if valid, else None
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    # Check extension
    filename = uploaded_file.name.lower()
    ext = os.path.splitext(filename)[1]
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG.",
            None,
        )

    # Check image content with PIL
    try:
        image = Image.open(uploaded_file)
        image.verify()  # Verifies file integrity

        # Image.verify leaves the stream at the end, so re-open for actual usage
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()  # Load image data into memory
        return True, None, image
    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}", None


def validate_coordinates(
    lat_val: Any,
    lon_val: Any
) -> Tuple[bool, Optional[str], float, float]:
    """
    Validates latitude and longitude values.
    Latitude must be between -90.0 and 90.0.
    Longitude must be between -180.0 and 180.0.
    """
    try:
        lat = float(lat_val)
        lon = float(lon_val)
    except (ValueError, TypeError):
        return False, "Latitude and Longitude must be valid decimal numbers.", 0.0, 0.0

    if not (-90.0 <= lat <= 90.0):
        return False, f"Latitude {lat} is out of bounds (-90.0 to 90.0).", lat, lon

    if not (-180.0 <= lon <= 180.0):
        return False, f"Longitude {lon} is out of bounds (-180.0 to 180.0).", lat, lon

    return True, None, lat, lon
