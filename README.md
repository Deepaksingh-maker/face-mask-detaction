# Face Mask Detection & Safety Monitoring System

A full-stack computer-vision application for real-time face-mask safety monitoring. The system captures webcam frames in a React dashboard, detects faces and mask status with a Python API, and presents stable per-person safety labels and live analytics.

> **Project status:** Educational / experimental. Validate the model on representative data before using it for safety, compliance, or access-control decisions.

## Highlights

- Real-time browser webcam experience with bounding-box overlays.
- Three-state safety status: **SAFE**, **UNSAFE**, and **CHECKING**.
- Per-face IoU tracking to retain identities across nearby frames.
- EWMA smoothing and hysteresis thresholds to reduce label flicker.
- FastAPI prediction API with interactive documentation.
- YOLO training, evaluation, static-image inference, and webcam inference utilities.
- React + Vite dashboard with live counts and overall safety status.

## Architecture

```text
Browser webcam
    |
    v
React + Vite dashboard (localhost:5173)
    |  POST base64 JPEG frames
    v
FastAPI service (localhost:8000)
    |
    +-- face detection and mask classification
    +-- IoU face tracking
    +-- temporal smoothing / hysteresis
    |
    v
Face-level results, analytics, and overall status
```

## Technology

| Area | Tools |
| --- | --- |
| Frontend | React, Vite |
| Backend | FastAPI, Uvicorn, Pydantic |
| Computer vision | OpenCV, PyTorch, Ultralytics YOLO |
| Data tooling | Hugging Face Datasets, Pandas, Matplotlib |
| Testing | `unittest`, Pytest |

## Repository layout

```text
.
+-- backend/                 # FastAPI API and face-mask detection components
+-- frontend/                # React/Vite webcam dashboard
+-- inference/               # Image, webcam, tracking, and stability utilities
+-- training/                # YOLO training script
+-- evaluation/              # Evaluation script and generated reports
+-- tools/                   # Dataset download, conversion, reporting, visualization
+-- tests/                   # Tracker and temporal-stability tests
+-- config.yaml              # Central project configuration
+-- requirements.txt         # Python dependencies
`-- run_backend.py           # Backend launcher
```

## Prerequisites

- Python 3.10 or later
- Node.js 18 or later (includes npm)
- A webcam and browser permission to access it, for live monitoring

## Quick start

From the repository root, create and activate a virtual environment, then install the Python packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Start the API in one terminal:

```powershell
python run_backend.py
```

Start the dashboard in a second terminal:

```powershell
cd frontend
npm run dev
```

Open the local URL Vite prints (normally `http://localhost:5173`), select **Start Camera**, and grant camera permission. API documentation is available at `http://localhost:8000/docs`.

## API

The dashboard calls the versioned endpoint below. Send a JSON object with an `image` data URL (for example, a `data:image/jpeg;base64,...` payload).

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/mask-detection/health` | Service and detector status |
| `POST` | `/api/v1/mask-detection/predict` | Analyze one image frame |

Example request:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

The prediction response includes each face's position, tracking ID, label, confidence, face counts, source dimensions, and an overall status.

## Model workflow

The project configuration expects the `hlydecker/face-masks` dataset with the following mapping:

| Class ID | Dataset label | Dashboard meaning |
| --- | --- | --- |
| `0` | `with_mask` | SAFE |
| `1` | `without_mask` | UNSAFE |

### Prepare data

```powershell
python tools/download_dataset.py
python tools/dataset_report.py
python tools/prepare_yolo_dataset.py
```

### Train and evaluate

```powershell
python training/train.py
python evaluation/evaluate.py
```

Training reads settings from [`config.yaml`](config.yaml), including model, image size, batch size, epochs, and target weights path. Generated data, runs, and model weights are excluded from Git by default.

### Run inference

```powershell
# Annotates an input image and writes annotated_<input-file-name> in the current directory
python inference/image_inference.py --image path\to\image.jpg

# Opens an OpenCV webcam inference window
python inference/webcam.py
```

## Validation

Run the system test suite from the repository root:

```powershell
python -m unittest tests/test_system.py
```

To produce an optimized production frontend bundle:

```powershell
cd frontend
npm run build
```

## Configuration

[`config.yaml`](config.yaml) centralizes dataset locations, class names, training options, inference thresholds, and temporal-stability settings. Tune the following values for your environment and model:

- `training.device` - use `cpu` or an appropriate accelerator setting.
- `inference.conf_threshold` and `inference.iou_threshold` - detection filtering sensitivity.
- `stability.*` - smoothing, transition thresholds, and stale-track timeout.
- `server.cors_origins` - allowed dashboard origins for deployment.

## Troubleshooting

- **Camera will not start:** confirm the browser has camera permission and no other application has exclusive access to the device.
- **Dashboard cannot analyze frames:** verify the API is running at `http://localhost:8000` and inspect `/api/v1/mask-detection/health`.
- **Missing model or dataset files:** complete the data preparation and training workflow, or update paths in `config.yaml`.
- **Slow inference:** lower the camera capture resolution, use hardware acceleration where supported, or run the browser and API on the same machine.

## License and acknowledgements

This repository does not currently declare a license. Add one before distributing or reusing the code outside its intended educational context.

Built with FastAPI, React, OpenCV, PyTorch, Ultralytics YOLO, and the `hlydecker/face-masks` dataset.
