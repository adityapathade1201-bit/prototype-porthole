"""
report.py
=========
Module for generating, formatting, and exporting road inspection reports.

Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System
"""

import io
import csv
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List


def generate_report_id() -> str:
    """
    Generates a human-readable, unique report identifier.
    Example: RPT-20260926-115500-A1B2
    """
    timestamp_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_suffix = uuid.uuid4().hex[:4].upper()
    return f"RPT-{timestamp_prefix}-{random_suffix}"


def build_report_data(
    report_id: str,
    image_filename: str,
    processed_image_filename: str,
    pothole_count: int,
    severity: str,
    latitude: float,
    longitude: float,
    location_name: str,
    timestamp: str,
    detections: List[Dict[str, Any]],
    model_version: str = "YOLO (Custom Pothole)"
) -> Dict[str, Any]:
    """
    Constructs a structured report dictionary containing all detection and location metadata.
    """
    return {
        "report_id": report_id,
        "image_filename": image_filename,
        "processed_image_filename": processed_image_filename,
        "pothole_count": pothole_count,
        "severity": severity,
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name or "Unspecified"
        },
        "timestamp": timestamp,
        "system_metadata": {
            "project_name": "AI-Based Intelligent Pothole Detection and Road Accountability System",
            "model": model_version,
            "severity_method": "Prototype Bounding-Box Surface Area Heuristic"
        },
        "detections": detections
    }


def export_report_to_json(report_data: Dict[str, Any]) -> str:
    """
    Serializes report data into a well-formatted JSON string.
    """
    return json.dumps(report_data, indent=4)


def export_report_to_csv(report_data: Dict[str, Any]) -> str:
    """
    Converts individual pothole detection items along with report headers into CSV format.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header section
    writer.writerow(["Report ID", report_data.get("report_id", "")])
    writer.writerow(["Timestamp", report_data.get("timestamp", "")])
    writer.writerow(["Severity", report_data.get("severity", "")])
    writer.writerow(["Pothole Count", report_data.get("pothole_count", 0)])
    writer.writerow(["Latitude", report_data.get("location", {}).get("latitude", "")])
    writer.writerow(["Longitude", report_data.get("location", {}).get("longitude", "")])
    writer.writerow(["Location Name", report_data.get("location", {}).get("location_name", "")])
    writer.writerow(["Original Image", report_data.get("image_filename", "")])
    writer.writerow([])  # Blank line separator

    # Detections table
    writer.writerow(["Detection #", "Confidence", "Relative Area (%)", "Severity", "Bounding Box [x1, y1, x2, y2]"])
    detections = report_data.get("detections", [])
    if detections:
        for idx, det in enumerate(detections, start=1):
            writer.writerow([
                idx,
                det.get("confidence", ""),
                det.get("relative_area_pct", ""),
                det.get("severity", ""),
                str(det.get("box", []))
            ])
    else:
        writer.writerow(["No potholes detected in this image.", "", "", "", ""])

    return output.getvalue()
