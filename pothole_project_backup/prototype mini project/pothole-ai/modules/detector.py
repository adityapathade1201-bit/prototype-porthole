"""
detector.py
===========
Module for loading the YOLO model and executing pothole object detection.

Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System
"""

import os
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

# Global / Default path to the trained pothole model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pothole.pt")


class ModelNotFoundError(Exception):
    """Exception raised when the trained pothole model file does not exist."""
    pass


def check_model_exists(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """
    Verifies whether a model file exists at the specified path.
    Does NOT download or fabricate any model silently.
    """
    return os.path.isfile(model_path)


def load_detector(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads the YOLO model from the provided weights path.

    Raises:
        ModelNotFoundError: If the model file does not exist at model_path.
    """
    if not check_model_exists(model_path):
        raise ModelNotFoundError(
            f"No trained pothole model found. Please place the trained model at {model_path}."
        )

    # Import ultralytics lazily so module can be imported even if ultralytics is warming up
    from ultralytics import YOLO

    model = YOLO(model_path)
    return model


def run_inference(
    model: Any,
    image_input: Any,
    confidence_threshold: float = 0.25
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Runs YOLO object detection on the provided image input.

    Args:
        model: Ultralytics YOLO model instance.
        image_input: PIL Image, numpy array (BGR or RGB), or image filepath.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0).

    Returns:
        Tuple containing:
        - detections: List of dictionaries with keys:
            - 'box': [x1, y1, x2, y2] in integer pixel coordinates
            - 'confidence': float between 0.0 and 1.0
            - 'class_id': integer class index
            - 'class_name': string label (e.g., 'pothole')
        - annotated_image: numpy array (RGB) with bounding boxes drawn.
    """
    # Convert PIL Image to RGB numpy array if needed
    if isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        # Assume RGB or convert if needed
        image_rgb = image_input.copy()
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        image_rgb = np.array(pil_img)
    else:
        raise ValueError("Unsupported image input type. Provide PIL Image, numpy array, or file path.")

    # Execute inference
    results = model.predict(source=image_rgb, conf=confidence_threshold, verbose=False)

    detections: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, "pothole") if hasattr(model, "names") else "pothole"

                detections.append({
                    "box": [round(coord) for coord in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": str(class_name)
                })

        # Ultralytics results[0].plot() returns BGR numpy array
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    else:
        annotated_rgb = image_rgb.copy()

    return detections, annotated_rgb


def annotate_image_custom(
    image_rgb: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Renders custom bounding boxes and annotations onto the image.
    Uses distinctive colors based on severity if available in detection dict.
    """
    annotated = image_rgb.copy()
    h, w, _ = annotated.shape

    color_map = {
        "LOW": (46, 204, 113),      # Green
        "MEDIUM": (243, 156, 18),   # Orange
        "HIGH": (231, 76, 60),      # Red
        "DEFAULT": (52, 152, 219)   # Blue
    }

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        conf = det.get("confidence", 0.0)
        sev = det.get("severity", "DEFAULT")
        color = color_map.get(sev, color_map["DEFAULT"])

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Label text
        label = f"#{idx} Pothole ({conf * 100:.1f}%)"
        if "severity" in det:
            label += f" [{sev}]"

        # Background tag for readability
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y1 = max(0, y1 - text_h - baseline - 6)
        tag_y2 = y1
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 10, tag_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return annotated
