# Models Directory

This directory stores the trained YOLO model weights for pothole detection.

## Required File Placement
To enable automated pothole inference, place your trained YOLO model file here with the following filename:
```
models/pothole.pt
```

## Model Guidelines
- **Framework**: Ultralytics YOLO (YOLOv8, YOLOv9, YOLOv10, or YOLOv11)
- **Format**: PyTorch `.pt` model checkpoint
- **Class Label**: The model should output detections with class `pothole` (or class 0).
- **Architecture Compatibility**: Standard Ultralytics YOLO object detection weights (`yolov8n.pt`, `yolov8s.pt`, custom trained `best.pt` renamed to `pothole.pt`).

## Application Behavior When Model is Missing
If `models/pothole.pt` is not present, the system does **not** silently download or fabricate any model. Instead, it informs the user:
```
No trained pothole model found. Please place the trained model at models/pothole.pt.
```

You may also configure a custom model path in the application UI settings or sidebar.
