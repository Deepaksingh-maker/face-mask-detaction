import os
import sys
import base64
import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.face_mask_detector.detect_webcam import load_face_detector
from backend.app.face_mask_detector.model import LightweightNumpyMaskClassifier
from inference.tracker import IoUFaceTracker
from inference.stability import TemporalStabilityEngine

app = FastAPI(
    title="AI Face Mask Detection API",
    description="Real-Time COVID Safety Monitoring System",
    version="1.0.0"
)

# Enable CORS for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config_path = os.path.join(root_dir, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load Face Detector Cascade & High-Precision Edge Classifier
face_cascade = load_face_detector()
classifier = LightweightNumpyMaskClassifier()

# Persistent IoU Face Tracker Engine & Temporal Stability
tracker = IoUFaceTracker(
    iou_threshold=config["inference"].get("iou_threshold", 0.30),
    max_stale_seconds=config["stability"].get("max_stale_seconds", 1.5)
)
stability_engine = TemporalStabilityEngine(config_path)

class FrameRequest(BaseModel):
    image: str

def non_max_suppression_boxes(boxes, overlapThresh=0.35):
    """
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes around the same face.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype="float")
    pick = []

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    w = boxes_arr[:, 2]
    h = boxes_arr[:, 3]
    x2 = x1 + w
    y2 = y1 + h

    area = (w + 1) * (h + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w_overlap = np.maximum(0, xx2 - xx1 + 1)
        h_overlap = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w_overlap * h_overlap) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return [boxes[i] for i in pick]

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Face Mask Detection & Real-Time Safety System",
        "detector_loaded": face_cascade is not None,
        "model_mode": "edge_classifier"
    }

@app.get("/health")
@app.get("/api/v1/mask-detection/health")
async def health_check():
    return {
        "status": "online",
        "detector_loaded": face_cascade is not None,
        "model_mode": "edge_classifier"
    }

@app.post("/predict")
@app.post("/api/v1/mask-detection/predict")
async def predict_frame(req: FrameRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Image data URL is required.")

    if face_cascade is None:
        raise HTTPException(status_code=500, detail="Face detector is not initialized.")

    try:
        encoded_data = req.image.split(",")[-1]
        img_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image payload.")

        img_h, img_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Multi-scale face detection
        raw_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(40, 40)
        )

        boxes_list = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in raw_faces]

        # 2. Apply Non-Maximum Suppression (NMS) to eliminate duplicate boxes on same person
        nms_boxes = non_max_suppression_boxes(boxes_list, overlapThresh=0.30)

        current_raw_detections = []

        for (x, y, w, h) in nms_boxes:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Direct High-Precision Classifier Inference
            raw_cls_id, raw_conf = classifier.predict(face_crop)

            current_raw_detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "cls_id": raw_cls_id,
                "conf": raw_conf
            })

        # Process through IoU Tracker & EWMA Temporal Stability Engine
        tracked_items = tracker.update(current_raw_detections)
        face_results = stability_engine.process_tracks(tracked_items)
        analytics = stability_engine.compute_analytics_and_global_status(face_results)

        return {
            "faces": face_results,
            "total_faces": analytics["total_faces"],
            "masked_count": analytics["masked_count"],
            "unmasked_count": analytics["unmasked_count"],
            "checking_count": analytics["checking_count"],
            "average_confidence": analytics["average_confidence"],
            "overall_status": analytics["overall_status"],
            "source_width": img_w,
            "source_height": img_h
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
