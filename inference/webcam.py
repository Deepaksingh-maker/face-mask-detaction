"""
Phase 9: Real-Time OpenCV Webcam Detection Script
Pipeline: Camera -> OpenCV Frame -> Ultralytics YOLO -> IoU Tracker -> EWMA Stability -> Live Display
"""
import os
import sys
import time
import cv2
import numpy as np
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inference.tracker import IoUFaceTracker
from inference.stability import TemporalStabilityEngine

def run_webcam_yolo(camera_index=0, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    best_weights_path = os.path.abspath(config["training"]["best_weights"])

    print("="*60)
    print("     PHASE 9: REAL-TIME WEBCAM DETECTION (YOLO)")
    print("="*60)

    try:
        from ultralytics import YOLO
        if os.path.exists(best_weights_path):
            model = YOLO(best_weights_path)
            print(f"[SUCCESS] Loaded custom trained weights: {best_weights_path}")
        else:
            print("[INFO] Custom weights not found. Loading pretrained YOLO nano model...")
            model = YOLO("yolov8n.pt")
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run pip install ultralytics.")
        return

    tracker = IoUFaceTracker(
        iou_threshold=config["inference"].get("iou_threshold", 0.25),
        max_stale_seconds=config["stability"].get("max_stale_seconds", 1.5)
    )
    stability_engine = TemporalStabilityEngine(config_path)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_title = "AI Face Mask Detection - Real-time Safety System"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1024, 600)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            # YOLO Object Detection Inference
            results = model.predict(frame, conf=config["inference"].get("conf_threshold", 0.45), verbose=False)[0]

            raw_detections = []
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                raw_detections.append({
                    "bbox": [x1, y1, w, h],
                    "cls_id": cls_id,
                    "conf": conf
                })

            # Multi-Face Tracking & EWMA Stability Engine
            tracked_items = tracker.update(raw_detections)
            face_results = stability_engine.process_tracks(tracked_items)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            analytics = stability_engine.compute_analytics_and_global_status(face_results, fps)

            # Draw Detections & Label Pills
            for face in face_results:
                bx, by, bw, bh = face["bbox"]
                lbl = face["label"]
                conf_pct = face["confidence"] * 100.0

                if lbl == "SAFE":
                    color = (135, 255, 0) # Green
                    pill_str = f"SAFE - {conf_pct:.1f}%"
                elif lbl == "UNSAFE":
                    color = (84, 46, 255) # Red
                    pill_str = f"UNSAFE - {conf_pct:.1f}%"
                else:
                    color = (21, 204, 250) # Yellow
                    pill_str = f"CHECKING - {conf_pct:.1f}%"

                # Draw Bounding Box
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 3)

                # Draw Label Pill
                (tw, th), _ = cv2.getTextSize(pill_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (bx, max(0, by - 26)), (bx + tw + 10, max(0, by)), color, -1)
                cv2.putText(frame, pill_str, (bx + 5, max(18, by - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 14, 20), 2, cv2.LINE_AA)

            # Draw HUD Analytics Banner
            cv2.rectangle(frame, (0, 0), (img_w, 40), (11, 14, 20), -1)
            hud_text = f"FACES: {analytics['total_faces']} | SAFE: {analytics['masked_count']} | UNSAFE: {analytics['unmasked_count']} | FPS: {analytics['fps']:.1f} | STATUS: {analytics['overall_status']}"
            cv2.putText(frame, hud_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 135), 2, cv2.LINE_AA)

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_yolo()
