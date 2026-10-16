"""
database.py
===========
Module for SQLite persistence of pothole reports, metadata, and analytics.

Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System
"""

import os
import sqlite3
import json
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "potholes.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Ensures that the directory structure exists.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database schema if the tables do not already exist.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                image_filename TEXT NOT NULL,
                processed_image_filename TEXT,
                pothole_count INTEGER NOT NULL,
                severity TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                location_name TEXT,
                timestamp TEXT NOT NULL,
                confidence_information TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_report(report_dict: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Inserts a new road pothole report into the SQLite database.

    Args:
        report_dict: Dictionary containing required report attributes:
            - report_id (str)
            - image_filename (str)
            - processed_image_filename (str, optional)
            - pothole_count (int)
            - severity (str)
            - latitude (float)
            - longitude (float)
            - location_name (str, optional)
            - timestamp (str)
            - confidence_information (str or list/dict)
        db_path: Path to the SQLite database file.

    Returns:
        bool: True if insert was successful, False otherwise.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    # Serialize confidence_information if provided as list or dict
    conf_info = report_dict.get("confidence_information", "")
    if isinstance(conf_info, (list, dict)):
        conf_info = json.dumps(conf_info)

    query = """
    INSERT INTO reports (
        report_id,
        image_filename,
        processed_image_filename,
        pothole_count,
        severity,
        latitude,
        longitude,
        location_name,
        timestamp,
        confidence_information
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        report_dict.get("report_id"),
        report_dict.get("image_filename"),
        report_dict.get("processed_image_filename", ""),
        int(report_dict.get("pothole_count", 0)),
        report_dict.get("severity", "LOW"),
        float(report_dict.get("latitude", 0.0)),
        float(report_dict.get("longitude", 0.0)),
        report_dict.get("location_name", "Unspecified"),
        report_dict.get("timestamp"),
        conf_info
    )

    try:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"[Database Error] Failed to insert report: {e}")
        raise e
    finally:
        conn.close()


def get_all_reports(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves all reports stored in the database as a pandas DataFrame.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        query = "SELECT * FROM reports ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_report_by_id(report_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Fetches a specific report record by its report_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            # Try parsing confidence_information back into Python structure
            try:
                data["confidence_information"] = json.loads(data["confidence_information"])
            except (json.JSONDecodeError, TypeError):
                pass
            return data
        return None
    finally:
        conn.close()


def get_database_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Calculates aggregate summary statistics for the dashboard.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    stats = {
        "total_reports": 0,
        "total_potholes": 0,
        "severity_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "NONE": 0},
        "avg_potholes_per_report": 0.0
    }
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(pothole_count), 0) FROM reports")
        total_reports, total_potholes = cursor.fetchone()

        stats["total_reports"] = total_reports or 0
        stats["total_potholes"] = total_potholes or 0

        if stats["total_reports"] > 0:
            stats["avg_potholes_per_report"] = round(stats["total_potholes"] / stats["total_reports"], 2)

        # Severity breakdown
        cursor.execute("SELECT severity, COUNT(*) FROM reports GROUP BY severity")
        for sev, count in cursor.fetchall():
            if sev in stats["severity_counts"]:
                stats["severity_counts"][sev] = count
            else:
                stats["severity_counts"][sev] = count

        return stats
    finally:
        conn.close()
