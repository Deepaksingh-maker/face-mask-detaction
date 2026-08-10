"""
Standalone Model Verification Script for Face Mask Detection.
Usage:
    python test_model.py --image path/to/image.jpg
"""
import os
import sys
import argparse
import cv2
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.face_mask_detector.detect_webcam import load_face_detector, load_mask_model
from backend.app.face_mask_detector.model import preprocess_face, HAS_TORCH

CLASS_MAPPING = {
    0: "with_mask (SAFE 🟢)",
    1: "without_mask (UNSAFE 🔴)"
}

def test_image(image_path, model_path="mask_detector_model.pth"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file '{image_path}' not found.")
        return

    print("\n" + "="*60)
    print("        STANDALONE FACE MASK CLASSIFICATION TEST")
    print("="*60)
    print(f"Loading image: {os.path.abspath(image_path)}")

    frame = cv2.imread(image_path)
    if frame is None:
        print("[ERROR] Could not read image.")
        return

    face_cascade = load_face_detector()
    model, device, mode = load_mask_model(model_path)
    print(f"Model Engine: {mode.upper()} | PyTorch Available: {HAS_TORCH}")

    img_h, img_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30)
    )

    print(f"Detected {len(faces)} face(s) in image.\n")

    if len(faces) == 0:
        # Evaluate whole crop if no face box detected by cascade
        faces = [(0, 0, img_w, img_h)]

    for idx, (x, y, w, h) in enumerate(faces):
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
                class_id = torch.argmax(probs).item()
                confidence = float(probs[class_id].item())
        else:
            class_id, confidence = model.predict(face_crop)

        label = "SAFE 🟢" if class_id == 0 else "UNSAFE 🔴"
        conf_percent = confidence * 100.0

        print(f"--- Face #{idx + 1} at (x={x1}, y={y1}, w={x2-x1}, h={y2-y1}) ---")
        print(f"  Raw Class ID : {class_id}")
        print(f"  Class Name   : {CLASS_MAPPING.get(class_id, 'Unknown')}")
        print(f"  Final Label  : {label}")
        print(f"  Confidence   : {conf_percent:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Face Mask Detection Model on an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    args = parser.parse_args()

    test_image(args.image)
