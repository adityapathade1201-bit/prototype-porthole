"""
app.py
======
Main Streamlit application for:
"AI-Based Intelligent Pothole Detection and Road Accountability System"

A modular college prototype MVP for automated road defect assessment,
severity estimation, geotagged reporting, and accountability tracking.
"""

import os
import sys
import json
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="AI Pothole Detection & Accountability System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project modules are discoverable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.detector import (
    check_model_exists,
    load_detector,
    run_inference,
    annotate_image_custom,
    ModelNotFoundError,
    DEFAULT_MODEL_PATH
)
from modules.severity import assess_pothole_severities
from modules.database import (
    init_db,
    save_report,
    get_all_reports,
    get_report_by_id,
    get_database_stats,
    DEFAULT_DB_PATH
)
from modules.report import (
    generate_report_id,
    build_report_data,
    export_report_to_json,
    export_report_to_csv
)
from utils.helpers import (
    ensure_directories,
    get_current_timestamp,
    validate_image_file,
    validate_coordinates
)

# Initialize project folders
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ensure_directories(UPLOADS_DIR, MODELS_DIR, DATA_DIR)
init_db(DEFAULT_DB_PATH)


# --- Custom Styling for Clean Academic Interface ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .severity-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .severity-badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .severity-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .prototype-note {
        font-size: 0.85rem;
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 8px 12px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Navigation & Model Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/road.png", width=64)
    st.markdown("### Navigation")
    menu = st.radio(
        "Go to page:",
        [
            "🏠 Dashboard",
            "🔍 Detect & Report",
            "🗄️ Report History",
            "ℹ️ Architecture & About"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    custom_model_path = st.text_input(
        "YOLO Model Path",
        value=os.path.relpath(DEFAULT_MODEL_PATH, BASE_DIR),
        help="Path to the trained YOLO weights file (.pt)"
    )

    resolved_model_path = os.path.abspath(
        os.path.join(BASE_DIR, custom_model_path) if not os.path.isabs(custom_model_path) else custom_model_path
    )

    model_present = check_model_exists(resolved_model_path)
    if model_present:
        st.success("✅ Model File Found")
    else:
        st.error("⚠️ Model File Not Found")
        st.caption("Place model at `models/pothole.pt`")

    conf_slider = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Filter predictions below this confidence score."
    )

    st.markdown("---")
    st.caption("College Prototype MVP v1.0")
    st.caption("AI-Based Road Accountability")


# ==============================================================================
# 1. HOME / DASHBOARD PAGE
# ==============================================================================
if menu == "🏠 Dashboard":
    st.markdown('<div class="main-title">AI-Based Intelligent Pothole Detection and Road Accountability System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">College Academic Prototype • Automated Road Inspection & Geotagged Hazard Management</div>', unsafe_allow_html=True)

    # Fetch aggregate database metrics
    stats = get_database_stats(DEFAULT_DB_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Inspection Reports", stats["total_reports"])
    with col2:
        st.metric("Total Potholes Detected", stats["total_potholes"])
    with col3:
        high_sev = stats["severity_counts"].get("HIGH", 0)
        st.metric("High Severity Hazards", high_sev, delta=f"{high_sev} critical" if high_sev else "None", delta_color="inverse")
    with col4:
        st.metric("Avg Potholes / Report", stats["avg_potholes_per_report"])

    st.markdown("---")

    # Severity distribution summary
    st.subheader("📊 Severity Breakdown Overview")
    col_chart, col_summary = st.columns([2, 1])

    with col_chart:
        sev_data = {
            "Severity": ["HIGH", "MEDIUM", "LOW", "NONE"],
            "Count": [
                stats["severity_counts"].get("HIGH", 0),
                stats["severity_counts"].get("MEDIUM", 0),
                stats["severity_counts"].get("LOW", 0),
                stats["severity_counts"].get("NONE", 0)
            ]
        }
        df_sev = pd.DataFrame(sev_data)
        st.bar_chart(df_sev.set_index("Severity"))

    with col_summary:
        st.markdown("**Severity Level Reference:**")
        st.markdown("- **HIGH (🔴)**: Major road hazard (rel. area $\\ge 7\\%$)")
        st.markdown("- **MEDIUM (🟠)**: Moderate road defect (rel. area $2\\% - 7\\%$)")
        st.markdown("- **LOW (🟢)**: Minor road imperfection (rel. area $< 2\\%$)")
        st.markdown(
            '<div class="prototype-note"><b>Prototype Notice:</b> Severity classification is a heuristic '
            'derived from bounding-box area relative to image resolution.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Recent reports preview
    st.subheader("🕒 Recent Road Inspection Reports")
    recent_df = get_all_reports(DEFAULT_DB_PATH)
    if not recent_df.empty:
        display_cols = ["report_id", "timestamp", "location_name", "pothole_count", "severity", "latitude", "longitude"]
        existing_cols = [c for c in display_cols if c in recent_df.columns]
        st.dataframe(recent_df[existing_cols].head(5), use_container_width=True)
    else:
        st.info("No reports logged yet. Navigate to 'Detect & Report' to analyze your first road image.")


# ==============================================================================
# 2. IMAGE UPLOAD, INFERENCE & REPORT PAGE
# ==============================================================================
elif menu == "🔍 Detect & Report":
    st.markdown('<div class="main-title">Pothole Detection & Field Report Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload road surface imagery, execute AI inference, and document geotagged road defects.</div>', unsafe_allow_html=True)

    # Model validation alert
    if not model_present:
        st.error(
            f"**No trained pothole model found.** Please place the trained model at `{resolved_model_path}`."
        )
        st.info(
            "💡 **How to proceed:**\n\n"
            "1. Train or obtain a YOLO model weights file for pothole detection (e.g., YOLOv8).\n"
            "2. Save the `.pt` file as `models/pothole.pt` (or configure the path in the sidebar).\n"
            "3. Once placed, reload this page to run live inference on uploaded road images."
        )

    # Step 1: Upload Image
    st.subheader("1. Road Imagery Input")
    uploaded_file = st.file_uploader(
        "Choose a road photo (Supported: JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload clear street-level or dashcam imagery."
    )

    if uploaded_file is not None:
        # Validate uploaded file format and integrity
        is_valid, err_msg, pil_img = validate_image_file(uploaded_file)
        if not is_valid:
            st.error(f"Image Error: {err_msg}")
        else:
            img_w, img_h = pil_img.size

            # Step 2: Location and Metadata Input
            st.subheader("2. Geolocation & Inspection Metadata")
            st.caption("Enter the manual inspection coordinates and location identifier.")

            loc_col1, loc_col2, loc_col3 = st.columns(3)
            with loc_col1:
                lat_input = st.number_input("Latitude (°N/S)", value=19.0760, format="%.6f", min_value=-90.0, max_value=90.0)
            with loc_col2:
                lon_input = st.number_input("Longitude (°E/W)", value=72.8777, format="%.6f", min_value=-180.0, max_value=180.0)
            with loc_col3:
                loc_name_input = st.text_input("Location Name / Landmark", value="Main Highway Sector 4")

            current_time = get_current_timestamp()
            st.text(f"Auto-generated Inspection Timestamp: {current_time}")

            st.markdown("---")

            # Step 3: Run Detection
            st.subheader("3. AI Detection & Severity Analysis")

            if st.button("🚀 Run Pothole Detection", type="primary", use_container_width=True):
                if not model_present:
                    st.error("Cannot run detection: No trained pothole model found at specified path.")
                else:
                    with st.spinner("Analyzing image with YOLO model..."):
                        try:
                            # Load YOLO model
                            detector_model = load_detector(resolved_model_path)
                            
                            # Execute inference
                            raw_detections, annotated_rgb = run_inference(
                                detector_model,
                                pil_img,
                                confidence_threshold=conf_slider
                            )

                            # Assess severity based on relative bounding-box area heuristic
                            overall_sev, enriched_detections, sev_counts = assess_pothole_severities(
                                raw_detections,
                                image_width=img_w,
                                image_height=img_h
                            )

                            # Re-annotate with customized severity color cues
                            custom_annotated = annotate_image_custom(annotated_rgb, enriched_detections)

                            # Store results in session state to allow report saving and exporting
                            report_id = generate_report_id()
                            
                            # Save uploaded and annotated images to uploads/
                            img_filename = f"{report_id}_orig.jpg"
                            proc_filename = f"{report_id}_annotated.jpg"
                            
                            pil_img.save(os.path.join(UPLOADS_DIR, img_filename))
                            Image.fromarray(custom_annotated).save(os.path.join(UPLOADS_DIR, proc_filename))

                            st.session_state["current_analysis"] = {
                                "report_id": report_id,
                                "orig_image": pil_img,
                                "annotated_image": custom_annotated,
                                "image_filename": img_filename,
                                "proc_filename": proc_filename,
                                "detections": enriched_detections,
                                "pothole_count": len(enriched_detections),
                                "overall_severity": overall_sev,
                                "severity_counts": sev_counts,
                                "latitude": float(lat_input),
                                "longitude": float(lon_input),
                                "location_name": loc_name_input.strip() or "Unspecified Location",
                                "timestamp": current_time,
                                "img_dimensions": (img_w, img_h)
                            }
                            st.success(f"Inference completed. Report ID: {report_id}")

                        except ModelNotFoundError as mne:
                            st.error(str(mne))
                        except Exception as ex:
                            st.error(f"Inference Failure: {str(ex)}")

            # Display Analysis Results if available in session state
            if "current_analysis" in st.session_state:
                analysis = st.session_state["current_analysis"]

                # Side-by-side Image Visualization
                st.markdown("#### Visual Inspection Comparison")
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(analysis["orig_image"], caption=f"Original Input ({img_w}x{img_h})", use_container_width=True)
                with col_img2:
                    st.image(analysis["annotated_image"], caption="Annotated Detection Output", use_container_width=True)

                # Key Detection Metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Potholes Detected", analysis["pothole_count"])
                with mcol2:
                    sev = analysis["overall_severity"]
                    st.metric("Overall Road Severity", sev)
                with mcol3:
                    avg_conf = (
                        sum(d["confidence"] for d in analysis["detections"]) / len(analysis["detections"])
                        if analysis["detections"] else 0.0
                    )
                    st.metric("Avg Detection Confidence", f"{avg_conf * 100:.1f}%")
                with mcol4:
                    st.metric("Report ID", analysis["report_id"])

                st.markdown(
                    '<div class="prototype-note"><b>Prototype Severity Estimation:</b> Evaluated based on pothole bounding-box '
                    'area relative to camera frame dimensions.</div>',
                    unsafe_allow_html=True
                )

                # Detection list breakdown
                if analysis["pothole_count"] == 0:
                    st.info("ℹ️ No potholes detected in this image matching the confidence threshold.")
                else:
                    st.markdown("#### Detection Findings Breakdown")
                    table_rows = []
                    for idx, det in enumerate(analysis["detections"], start=1):
                        table_rows.append({
                            "Pothole #": idx,
                            "Confidence": f"{det['confidence'] * 100:.2f}%",
                            "Rel Area (%)": f"{det.get('relative_area_pct', 0.0):.2f}%",
                            "Severity": det.get("severity", "LOW"),
                            "Bounding Box [x1, y1, x2, y2]": str(det["box"])
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.markdown("---")

                # Step 4: Save and Download Report
                st.subheader("4. Accountability Report Generation & Persistence")

                # Prepare structured report payload
                report_payload = build_report_data(
                    report_id=analysis["report_id"],
                    image_filename=analysis["image_filename"],
                    processed_image_filename=analysis["proc_filename"],
                    pothole_count=analysis["pothole_count"],
                    severity=analysis["overall_severity"],
                    latitude=analysis["latitude"],
                    longitude=analysis["longitude"],
                    location_name=analysis["location_name"],
                    timestamp=analysis["timestamp"],
                    detections=analysis["detections"]
                )

                action_col1, action_col2, action_col3 = st.columns(3)

                with action_col1:
                    if st.button("💾 Save Report to SQLite Database", type="secondary", use_container_width=True):
                        try:
                            # Validate coordinates before saving
                            c_valid, c_err, v_lat, v_lon = validate_coordinates(analysis["latitude"], analysis["longitude"])
                            if not c_valid:
                                st.error(f"Cannot save: {c_err}")
                            else:
                                save_report(report_payload, DEFAULT_DB_PATH)
                                st.success(f"Report `{analysis['report_id']}` successfully saved into SQLite database!")
                        except Exception as e:
                            st.error(f"Database error while saving report: {str(e)}")

                with action_col2:
                    json_data = export_report_to_json(report_payload)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=json_data,
                        file_name=f"{analysis['report_id']}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with action_col3:
                    csv_data = export_report_to_csv(report_payload)
                    st.download_button(
                        label="📥 Download Report (CSV)",
                        data=csv_data,
                        file_name=f"{analysis['report_id']}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ==============================================================================
# 3. REPORT HISTORY PAGE
# ==============================================================================
elif menu == "🗄️ Report History":
    st.markdown('<div class="main-title">Road Inspection Report History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Archived inspection logs, geographic defect locations, and severity tracking.</div>', unsafe_allow_html=True)

    df_reports = get_all_reports(DEFAULT_DB_PATH)

    if df_reports.empty:
        st.info("No reports currently logged in SQLite database.")
    else:
        # Filtering controls
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            severity_filter = st.selectbox("Filter by Severity:", ["ALL", "HIGH", "MEDIUM", "LOW", "NONE"])
        with filter_col2:
            search_query = st.text_input("Search by Location Name or Report ID:", "")

        filtered_df = df_reports.copy()
        if severity_filter != "ALL":
            filtered_df = filtered_df[filtered_df["severity"] == severity_filter]
        if search_query.strip():
            query_lower = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["location_name"].str.lower().str.contains(query_lower, na=False) |
                filtered_df["report_id"].str.lower().str.contains(query_lower, na=False)
            ]

        st.caption(f"Showing {len(filtered_df)} of {len(df_reports)} total records.")
        st.dataframe(filtered_df, use_container_width=True)

        # Full database export
        csv_full = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Filtered History (CSV)",
            data=csv_full,
            file_name=f"pothole_reports_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("🔎 Inspect Individual Report Details")
        selected_id = st.selectbox("Select Report ID to inspect:", filtered_df["report_id"].tolist())

        if selected_id:
            report_detail = get_report_by_id(selected_id, DEFAULT_DB_PATH)
            if report_detail:
                d_col1, d_col2 = st.columns([1, 1])

                with d_col1:
                    st.markdown(f"**Report ID:** `{report_detail['report_id']}`")
                    st.markdown(f"**Timestamp:** {report_detail['timestamp']}")
                    st.markdown(f"**Location:** {report_detail.get('location_name', 'N/A')}")
                    st.markdown(f"**Coordinates:** {report_detail['latitude']}, {report_detail['longitude']}")
                    st.markdown(f"**Potholes Detected:** {report_detail['pothole_count']}")
                    st.markdown(f"**Overall Severity:** `{report_detail['severity']}`")

                with d_col2:
                    proc_img_name = report_detail.get("processed_image_filename")
                    orig_img_name = report_detail.get("image_filename")
                    
                    target_img = proc_img_name if proc_img_name else orig_img_name
                    img_path = os.path.join(UPLOADS_DIR, target_img) if target_img else ""

                    if img_path and os.path.isfile(img_path):
                        st.image(img_path, caption=f"Report Image ({target_img})", use_container_width=True)
                    else:
                        st.caption("Archived image file not available in uploads directory.")


# ==============================================================================
# 4. ARCHITECTURE & ABOUT PAGE
# ==============================================================================
elif menu == "ℹ️ Architecture & About":
    st.markdown('<div class="main-title">Project Architecture & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Based Intelligent Pothole Detection and Road Accountability System</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Prototype Overview
    This prototype serves as the foundational engineering MVP for an automated road condition inspection system.
    It combines computer vision with structured spatial reporting to create an auditable defect log for municipal road monitoring.

    ### 2. Modular Architecture
    The codebase is split into isolated modules:
    - **`modules/detector.py`**: Handles YOLO model verification, weights loading, inference execution, and bounding box drawing.
    - **`modules/severity.py`**: Implements the prototype heuristic severity estimation (LOW, MEDIUM, HIGH) derived from bounding-box relative surface area.
    - **`modules/database.py`**: Manages SQLite persistence, schema initialization, and statistical aggregation.
    - **`modules/report.py`**: Coordinates report ID generation, JSON formatting, and CSV export.
    - **`utils/helpers.py`**: Validates uploaded image format and integrity, geographic coordinate bounds, and timestamp formatting.
    - **`app.py`**: Streamlit frontend providing the academic interface, dashboard, inspection workflow, and history viewer.

    ### 3. Severity Classification Heuristic
    $$\\text{Relative Area Ratio} = \\frac{\\text{Bounding Box Area}}{\\text{Total Image Frame Area}}$$
    - **LOW**: $\\text{Ratio} < 0.02$ ($< 2\\%$ of image frame)
    - **MEDIUM**: $0.02 \\le \\text{Ratio} < 0.07$ ($2\\% - 7\\%$ of image frame)
    - **HIGH**: $\\text{Ratio} \\ge 0.07$ ($\\ge 7\\%$ of image frame)

    *Note: Clearly marked as a prototype heuristic for college demonstration.*

    ### 4. Road Accountability Foundation
    Every report records an immutable timestamp, geographic coordinates, image evidence, and detection confidence,
    forming the data layer for future contractor accountability, tendering audit trails, and automated repair ticketing.
    """)
