"""
severity.py
===========
Module for prototype severity classification of detected potholes.

Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System

NOTE:
This module implements a PROTOTYPE severity estimation heuristic.
Severity is derived from the bounding-box surface area relative to the overall
image frame dimensions. In future production stages, this can be combined with
depth sensors, stereoscopic vision, and road surface profiling.
"""

from typing import List, Dict, Any, Tuple

# Threshold constants for relative area ratio (bbox_area / total_image_area)
# LOW: Less than 2% of camera image frame
# MEDIUM: Between 2% and 7% of camera image frame
# HIGH: 7% or more of camera image frame
THRESHOLD_LOW_UPPER = 0.02
THRESHOLD_MEDIUM_UPPER = 0.07


def calculate_bbox_area(box: List[int]) -> int:
    """
    Computes the area of a bounding box in pixels.
    box: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def calculate_relative_area_ratio(box: List[int], image_width: int, image_height: int) -> float:
    """
    Computes the ratio of the bounding-box area relative to the entire image area.
    Returns a float between 0.0 and 1.0.
    """
    total_image_area = image_width * image_height
    if total_image_area <= 0:
        return 0.0

    box_area = calculate_bbox_area(box)
    return float(box_area) / float(total_image_area)


def classify_pothole_severity(relative_area_ratio: float) -> str:
    """
    Classifies a single detected pothole into a prototype severity category:
    - LOW: Small surface impact (< 2% of frame)
    - MEDIUM: Moderate road damage (2% to 7% of frame)
    - HIGH: Severe road hazard (>= 7% of frame)
    """
    if relative_area_ratio < THRESHOLD_LOW_UPPER:
        return "LOW"
    elif relative_area_ratio < THRESHOLD_MEDIUM_UPPER:
        return "MEDIUM"
    else:
        return "HIGH"


def assess_pothole_severities(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    """
    Enriches each detection with its relative area percentage and individual severity.
    Determines an overall severity grade for the road inspection report.

    Args:
        detections: List of detection dictionaries containing 'box' and 'confidence'.
        image_width: Width of the input image in pixels.
        image_height: Height of the input image in pixels.

    Returns:
        Tuple:
        - overall_severity: "NONE", "LOW", "MEDIUM", or "HIGH"
        - enriched_detections: Updated list with 'relative_area_pct' and 'severity' fields
        - severity_counts: Dict with counts of {"LOW": int, "MEDIUM": int, "HIGH": int}
    """
    if not detections:
        return "NONE", [], {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    enriched = []
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    max_severity_rank = 0  # 1: LOW, 2: MEDIUM, 3: HIGH

    rank_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    for det in detections:
        ratio = calculate_relative_area_ratio(det["box"], image_width, image_height)
        pct = round(ratio * 100, 2)
        sev = classify_pothole_severity(ratio)

        counts[sev] += 1
        rank = rank_map[sev]
        if rank > max_severity_rank:
            max_severity_rank = rank

        det_copy = dict(det)
        det_copy["relative_area_pct"] = pct
        det_copy["severity"] = sev
        enriched.append(det_copy)

    # Determine overall severity based on highest individual severity or cumulative density
    inv_rank_map = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    overall_severity = inv_rank_map.get(max_severity_rank, "LOW")

    # If there are multiple medium potholes (>=3), escalate to HIGH due to cluster hazard
    if counts["MEDIUM"] >= 3 and overall_severity != "HIGH":
        overall_severity = "HIGH"
    elif counts["LOW"] >= 4 and overall_severity == "LOW":
        overall_severity = "MEDIUM"

    return overall_severity, enriched, counts
