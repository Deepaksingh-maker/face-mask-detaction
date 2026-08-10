"""
Phase 8: Static Image Inference Script
Performs direct YOLO object detection on a static image file and saves annotated output.
"""
import os
import sys
import argparse
import cv2
import numpy as np
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

CLASS_NAMES = {
    0: "SAFE",
    1: "UNSAFE"
}

CLASS_COLORS = {
    0: (135, 255, 0),  # Green for SAFE (with_mask)
    1: (84, 46, 255)   # Red for UNSAFE (without_mask)
}

def run_image_inference(image_path, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    best_weights_path = os.path.abspath(config["training"]["best_weights"])

    print("="*60)
    print("        PHASE 8: STATIC IMAGE INFERENCE (YOLO)")
    print("="*60)
    print(f"Loading Image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image at '{image_path}'.")
        return

    try:
        from ultralytics import YOLO
        if os.path.exists(best_weights_path):
            model = YOLO(best_weights_path)
        else:
            model = YOLO("yolov8n.pt")

        results = model.predict(frame, conf=config["inference"].get("conf_threshold", 0.45))[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            is_masked = (cls_id == 0)
            label = "SAFE" if is_masked else "UNSAFE"

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            status_icon = "🟢" if is_masked else "🔴"
            pill_text = f"{status_icon} {label} — {conf * 100:.1f}%"

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw Pill Text Label above box
            (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + tw + 12, max(0, y1)), color, -1)
            cv2.putText(frame, pill_text, (x1 + 6, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 14, 20), 2, cv2.LINE_AA)

            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "class_id": cls_id,
                "label": label,
                "confidence": round(conf, 4)
            })

            print(f"  - Detected Face: {label} (Class {cls_id}) | Conf: {conf*100:.1f}% | Box: [{x1}, {y1}, {x2}, {y2}]")

        out_path = f"annotated_{os.path.basename(image_path)}"
        cv2.imwrite(out_path, frame)
        print(f"\n[SUCCESS] Saved annotated result image to: {out_path}")
        return detections

    except Exception as e:
        print(f"[ERROR] Image inference failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO Inference on a Static Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    args = parser.parse_args()

    run_image_inference(args.image)
