# AI Face Mask Detection & Real-Time Safety Monitoring System

A professional, real-time AI Face Mask Detection and Safety Verification System powered by **Ultralytics YOLO** object detection, persistent **IoU Face Tracking**, **EWMA Temporal Smoothing**, and a **Hysteresis Decision Machine**.

---

## 🌟 Key Features

- **Direct YOLO Object Detection**: Single-stage real-time detection eliminating legacy cascade + CNN crop classifier bottlenecks.
- **Fixed Hugging Face Dataset**: Built exclusively on `hlydecker/face-masks` dataset with verified class mapping (`0: with_mask`, `1: without_mask`).
- **Multi-Face Real-Time Tracking**: Detects and tracks multiple faces independently with IoU-based persistent track IDs (`Face #1`, `Face #2`...).
- **Temporal Prediction Stability**: Exponentially Weighted Moving Average (EWMA) smoothing eliminates frame-to-frame label flickering.
- **Hysteresis & Uncertainty Handling**: 3-state classification system (🟢 **`SAFE — 97.4%`**, 🔴 **`UNSAFE — 95.8%`**, 🟡 **`CHECKING — 65.0%`**).
- **Full-Stack Architecture**: Python FastAPI AI Backend + React (Vite) Cyberpunk Dark Dashboard Frontend.

---

## 📐 Project Directory Structure

```
face_mask_detection_project/
│
├── data/
│   ├── raw/face-masks/              # Downloaded Hugging Face raw dataset
│   └── processed/face-masks-yolo/   # Converted Ultralytics YOLO format & data.yaml
│
├── models/
│   └── best.pt                      # Trained Ultralytics YOLO model weights
│
├── training/
│   └── train.py                     # YOLO model training script
│
├── evaluation/
│   ├── evaluate.py                  # Model test evaluation script
│   └── reports/                     # Precision, Recall, mAP50, dataset_report.json
│
├── inference/
│   ├── image_inference.py           # Static image inference script
│   ├── webcam.py                    # Real-time OpenCV webcam detection script
│   ├── tracker.py                   # IoU multi-face tracking engine
│   └── stability.py                 # EWMA temporal smoothing & hysteresis engine
│
├── tools/
│   ├── download_dataset.py          # Hugging Face dataset downloader
│   ├── dataset_report.py            # Dataset inspection & report generator
│   ├── visualize_dataset.py        # Bounding box sample visualizer
│   └── prepare_yolo_dataset.py      # YOLO format converter
│
├── backend/
│   └── main.py                      # FastAPI web server endpoint router
│
├── frontend/                        # React (Vite) web dashboard application
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── components/Camera.jsx
│       └── index.css
│
├── tests/
│   └── test_system.py               # Automated unit testing suite
│
├── config.yaml                      # System-wide configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # Master documentation
```

---

## ⚡ Quick Start Guide

### 1. Installation
```bash
# Clone repository and enter project directory
cd "New folder (2)"

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download & Conversion
```bash
# Download Hugging Face dataset (hlydecker/face-masks)
python tools/download_dataset.py

# Generate Dataset Metadata Report
python tools/dataset_report.py

# Convert Dataset to Ultralytics YOLO Format
python tools/prepare_yolo_dataset.py
```

### 3. Model Training & Evaluation
```bash
# Train Ultralytics YOLO Model
python training/train.py

# Evaluate Model Metrics (mAP50, mAP50-95, Precision, Recall)
python evaluation/evaluate.py
```

### 4. Direct Inference Testing
```bash
# Test on a Static Image
python inference/image_inference.py --image path/to/image.jpg

# Test Live OpenCV Webcam Feed
python inference/webcam.py
```

### 5. Web Application Launch

#### Terminal 1 (FastAPI AI Backend):
```bash
python run_backend.py
```

#### Terminal 2 (React Frontend):
```bash
cd frontend
npm run dev
```

Open browser at **`http://localhost:5173`**!

---

## 🛡️ License & Credits
Built with Ultralytics YOLO, PyTorch, OpenCV, FastAPI, and React. Dataset provided by `hlydecker/face-masks` on Hugging Face.
