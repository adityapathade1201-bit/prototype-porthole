"""
Root Launcher for Streamlit
===========================
Academic Project: AI-Based Intelligent Pothole Detection and Road Accountability System

This launcher allows running `streamlit run app.py` directly from the workspace root
as well as from within the `pothole-ai/` directory.
"""

import os
import sys

# Target path to the primary app.py inside pothole-ai directory
SUBDIR_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pothole-ai", "app.py")

if os.path.isfile(SUBDIR_APP):
    # Switch working directory to pothole-ai so relative paths (data/, models/, uploads/) resolve cleanly
    pothole_ai_dir = os.path.dirname(SUBDIR_APP)
    os.chdir(pothole_ai_dir)
    if pothole_ai_dir not in sys.path:
        sys.path.insert(0, pothole_ai_dir)

    with open(SUBDIR_APP, "r", encoding="utf-8") as f:
        code_str = f.read()
    exec(compile(code_str, SUBDIR_APP, "exec"), globals())
else:
    raise FileNotFoundError(f"Could not locate main application at {SUBDIR_APP}")
