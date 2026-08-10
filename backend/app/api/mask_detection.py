import sys
import os
import time
import base64
import cv2
import numpy as np
from collections import deque
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.face_mask_detector.detect_webcam import load_face_detector, load_mask_model
from backend.app.face_mask_detector.model import preprocess_face, HAS_TORCH

router = APIRouter()

face_cascade = load_face_detector()
model, device, mode = load_mask_model()

CLASS_MAPPING = {
    0: "with_mask",     # SAFE
    1: "without_mask"   # UNSAFE
}

def compute_iou(boxA, boxB):
    """Computes Intersection-over-Union (IoU) of two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class FaceTrackerEngine:
    """
    Persistent Per-Face Temporal Tracking & Smoothing Engine.
    Maintains face identity across camera frames using IoU matching.
    Applies EWMA temporal smoothing and hysteresis thresholds to prevent label flipping.
    """
    def __init__(self, history_len=7, max_stale_seconds=1.5):
        self.history_len = history_len
        self.max_stale_seconds = max_stale_seconds
        self.next_track_id = 1
        self.tracks = {}

    def process_detections(self, current_detections):
        now = time.time()
        
        # 1. Clean up stale tracks
        stale_ids = [tid for tid, tr in self.tracks.items() if (now - tr['last_seen']) > self.max_stale_seconds]
        for tid in stale_ids:
            del self.tracks[tid]

        # 2. Match current detections to existing tracks via IoU
        matched_track_ids = set()
        results = []

        for det in current_detections:
            curr_box = det['bbox']
            best_iou = 0.0
            best_tid = None

            for tid, tr in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = compute_iou(curr_box, tr['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= 0.25:
                track = self.tracks[best_tid]
                matched_track_ids.add(best_tid)
            else:
                best_tid = self.next_track_id
                self.next_track_id += 1
                track = {
                    'bbox': curr_box,
                    'history': deque(maxlen=self.history_len),
                    'ewma_prob': 0.50,
                    'state': 'CHECKING',
                    'last_seen': now
                }
                self.tracks[best_tid] = track

            track['bbox'] = curr_box
            track['last_seen'] = now

            # Raw mask probability: Class 0 (with_mask) -> raw_conf, Class 1 -> (1.0 - raw_conf)
            raw_mask_prob = det['raw_conf'] if det['raw_class_id'] == 0 else (1.0 - det['raw_conf'])
            track['history'].append(raw_mask_prob)

            # Exponentially Weighted Moving Average (EWMA) smoothing across recent frames
            alpha = 0.35
            track['ewma_prob'] = alpha * raw_mask_prob + (1.0 - alpha) * track['ewma_prob']
            smoothed_p = track['ewma_prob']

            # HYSTERESIS STATE MACHINE & UNCERTAINTY THRESHOLDS
            #   SAFE transition  : smoothed_p >= 0.70
            #   UNSAFE transition: smoothed_p <= 0.35
            #   CHECKING state   : 0.35 < smoothed_p < 0.70 when stabilizing
            curr_state = track['state']

            if curr_state == 'SAFE':
                new_state = 'UNSAFE' if smoothed_p <= 0.35 else 'SAFE'
            elif curr_state == 'UNSAFE':
                new_state = 'SAFE' if smoothed_p >= 0.70 else 'UNSAFE'
            else:
                if smoothed_p >= 0.70:
                    new_state = 'SAFE'
                elif smoothed_p <= 0.35:
                    new_state = 'UNSAFE'
                else:
                    new_state = 'CHECKING'

            track['state'] = new_state

            # Compute label & confidence percentage
            if new_state == 'SAFE':
                displayed_label = 'SAFE'
                is_masked = True
                conf_val = smoothed_p
            elif new_state == 'UNSAFE':
                displayed_label = 'UNSAFE'
                is_masked = False
                conf_val = 1.0 - smoothed_p
            else:
                displayed_label = 'CHECKING'
                is_masked = False
                conf_val = 0.65 # Neutral stabilizing confidence

            print(f"[DEBUG TRACK #{best_tid}] Raw Prob: {raw_mask_prob:.3f} | Smoothed EWMA: {smoothed_p:.3f} | State: {new_state} ({conf_val*100:.1f}%)")

            results.append({
                'track_id': best_tid,
                'x': curr_box[0],
                'y': curr_box[1],
                'width': curr_box[2],
                'height': curr_box[3],
                'label': displayed_label,
                'is_masked': is_masked,
                'confidence': round(float(conf_val), 4),
                'bbox': curr_box
            })

        return results

# Global Face Tracker Engine Instance
face_tracker_engine = FaceTrackerEngine()

class FrameRequest(BaseModel):
    image: str

@router.get("/mask-detection/health")
async def health_check():
    return {
        "status": "online",
        "detector_loaded": face_cascade is not None,
        "model_mode": mode
    }

@router.post("/mask-detection/predict")
async def predict_mask(req: FrameRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Image data URL is required.")

    try:
        encoded_data = req.image.split(",")[-1]
        img_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image payload.")

        img_h, img_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(35, 35)
        )

        current_raw_detections = []

        for (x, y, w, h) in detected_faces:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            if mode == "pytorch":
                import torch
                import torch.nn.functional as F
                input_tensor = preprocess_face(face_crop).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)[0]
                    raw_class_id = torch.argmax(probs).item()
                    raw_conf = float(probs[raw_class_id].item())
            else:
                raw_class_id, raw_conf = model.predict(face_crop)

            current_raw_detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "raw_class_id": raw_class_id,
                "raw_conf": raw_conf
            })

        # Process detections through Temporal Face Tracker & Hysteresis Engine
        face_results = face_tracker_engine.process_detections(current_raw_detections)

        total_faces = len(face_results)
        mask_count = sum(1 for f in face_results if f['label'] == 'SAFE')
        nomask_count = sum(1 for f in face_results if f['label'] == 'UNSAFE')
        checking_count = sum(1 for f in face_results if f['label'] == 'CHECKING')

        if total_faces == 0:
            overall_status = "NO_FACE"
        elif nomask_count > 0:
            overall_status = "UNSAFE"
        elif checking_count > 0:
            overall_status = "CHECKING"
        else:
            overall_status = "SAFE"

        return {
            "faces": face_results,
            "total_faces": total_faces,
            "masked_count": mask_count,
            "unmasked_count": nomask_count,
            "checking_count": checking_count,
            "overall_status": overall_status,
            "source_width": img_w,
            "source_height": img_h
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
