# AI-Based Intelligent Pothole Detection and Road Accountability System

An academic college project prototype MVP designed for automated road surface inspection, defect detection using computer vision, heuristic severity classification, and geotagged accountability record management.

---

## 1. Project Overview
Road surface deterioration and unmonitored potholes pose severe safety hazards to motorists and pedestrians. This prototype demonstrates an intelligent road monitoring system that:
- Ingests street-level road surface imagery (JPG, JPEG, PNG).
- Detects potholes using an Ultralytics YOLO object detection pipeline.
- Extracts bounding box coordinates and detection confidence scores.
- Computes prototype severity categories (**LOW**, **MEDIUM**, **HIGH**) based on relative bounding-box area heuristic.
- Pairs detections with manual GPS coordinates and auto-generated inspection timestamps.
- Persists all inspection records in an embedded SQLite database.
- Provides an interactive Streamlit dashboard for real-time review and JSON/CSV data export.

---

## 2. Project Structure

```
pothole-ai/
│
├── app.py                     # Main Streamlit web application & UI
├── requirements.txt           # Project dependencies specification
├── README.md                  # Comprehensive setup and usage documentation
│
├── models/
│   └── README.md              # Instructions for placing YOLO weights (pothole.pt)
│
├── data/
│   └── potholes.db            # SQLite database for persistent report storage
│
├── modules/
│   ├── detector.py            # YOLO inference, weights loading, bounding box annotation
│   ├── severity.py            # Heuristic severity estimation (LOW, MEDIUM, HIGH)
│   ├── database.py            # SQLite database connection, queries, and statistics
│   └── report.py              # Report ID generation and JSON/CSV export formatting
│
├── utils/
│   └── helpers.py             # File validation, coordinate bounds checking, timestamps
│
└── uploads/                   # Storage for original input and annotated output images
```

---

## 3. Installation & Virtual Environment Setup

### Prerequisites
- Python 3.11+ installed on your system.

### Step A: Clone or Navigate to Project Directory
```bash
cd pothole-ai
```

### Step B: Create a Virtual Environment
Using Python's standard `venv` module:

**On Windows (PowerShell / Command Prompt):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Dependency Installation

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Core Libraries Used:
- `streamlit`: Clean interactive web dashboard and user interface.
- `ultralytics`: YOLO object detection engine.
- `opencv-python`: Image manipulation and bounding box visualization.
- `pillow`: Image I/O and format validation.
- `pandas`: Tabular data handling and CSV exports.

---

## 5. Model Placement Requirement

> [!IMPORTANT]
> The system **does not** silently download or fabricate any AI model weights.

1. Obtain or train a custom YOLO model checkpoint (e.g. YOLOv8/v11 trained on a pothole dataset).
2. Rename or save the model file as:
   ```
   models/pothole.pt
   ```
3. If no model is placed at this path, the application will display a clear advisory:
   ```
   No trained pothole model found. Please place the trained model at models/pothole.pt.
   ```
4. For flexible development, you can also adjust the model path dynamically in the Streamlit sidebar under **⚙️ Model Configuration**.

---

## 6. Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Application Features:
1. **🏠 Dashboard**: Displays total inspection reports, total detected potholes, severity breakdown chart, and recent reports.
2. **🔍 Detect & Report**: Upload road images, set coordinates, view detections and confidence scores, evaluate severity, save to SQLite, and download reports (JSON or CSV).
3. **🗄️ Report History**: Query and filter archived reports by severity or location, inspect individual detections, and export data.
4. **ℹ️ Architecture & About**: Academic documentation explaining the underlying modules, heuristics, and technical workflow.

---

## 7. Prototype Severity Estimation Heuristic

Severity is computed using the ratio of bounding-box surface area to total image frame area:

$$\text{Relative Area Ratio} = \frac{\text{Bounding Box Area (px)}}{\text{Image Width} \times \text{Image Height (px)}}$$

| Severity Grade | Relative Area Condition | Practical Interpretation |
| :--- | :--- | :--- |
| **LOW** (🟢) | $< 2\%$ of image frame | Small surface imperfection or minor depression |
| **MEDIUM** (🟠) | $2\% - 7\%$ of image frame | Moderate pothole requiring road maintenance |
| **HIGH** (🔴) | $\ge 7\%$ of image frame | Severe hazard requiring immediate road remediation |

*Note: In the MVP, this serves as a clear prototype heuristic. Future versions can incorporate depth camera sensors or stereoscopic vision.*

---

## 8. Known Limitations (MVP Scope)
- **Manual Geolocation Input**: GPS coordinates are entered manually to ensure accurate simulation without mock browser geolocation.
- **2D Area Estimation**: Severity is estimated strictly from 2D pixel area; depth profile is not measured in this MVP.
- **Single-Image Inference**: Designed for photo uploads; live video streaming and dashcam feeds will be introduced in subsequent milestones.
- **Accountability Layer**: Database schema captures audit timestamps and geographic coordinates; contractor assignment and tender matching are omitted from this MVP stage as requested.
