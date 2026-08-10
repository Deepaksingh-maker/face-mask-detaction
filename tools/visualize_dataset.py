"""
Phase 4: Bounding Box Visualization Script
Selects 20 sample images from data/processed/face-masks-yolo/, draws bounding boxes & labels, and saves to evaluation/reports/visualizations/
"""
import os
import sys
import random
import cv2
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

CLASS_COLORS = {
    0: (0, 255, 135),  # Green for with_mask (SAFE)
    1: (84, 46, 255)   # Red for without_mask (UNSAFE)
}

CLASS_NAMES = {
    0: "with_mask",
    1: "without_mask"
}

def visualize_sample_annotations(config_path="config.yaml", num_samples=20):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = os.path.abspath(config["dataset"]["processed_dir"])
    train_img_dir = os.path.join(processed_dir, "images", "train")
    train_lbl_dir = os.path.join(processed_dir, "labels", "train")

    output_dir = os.path.join(root_dir, "evaluation", "reports", "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("      PHASE 4: VISUALIZING BOUNDING BOX ANNOTATIONS")
    print("="*60)

    if not os.path.exists(train_img_dir):
        print(f"[ERROR] Processed image directory not found at '{train_img_dir}'. Run copy_hf_snapshot_yolo.py first.")
        return

    img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(img_files) == 0:
        print("[ERROR] No image files found in dataset.")
        return

    sample_files = random.sample(img_files, min(num_samples, len(img_files)))
    saved_count = 0

    for fname in sample_files:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(train_img_dir, fname)
        lbl_path = os.path.join(train_lbl_dir, f"{stem}.txt")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cat_id = int(parts[0])
                    xc, yc, nw, nh = [float(v) for v in parts[1:5]]

                    bw = int(nw * w)
                    bh = int(nh * h)
                    bx = int((xc * w) - (bw / 2.0))
                    by = int((yc * h) - (bh / 2.0))

                    bx = max(0, bx)
                    by = max(0, by)

                    color = CLASS_COLORS.get(cat_id, (255, 255, 255))
                    cname = CLASS_NAMES.get(cat_id, f"Class {cat_id}")

                    # Draw Bounding Box
                    cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), color, 2)

                    # Draw Label Pill
                    text = f"{cname}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_bgr, (bx, max(0, by - 20)), (bx + tw + 6, max(0, by)), color, -1)
                    cv2.putText(img_bgr, text, (bx + 3, max(12, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_file = os.path.join(output_dir, f"sample_{saved_count + 1:02d}.jpg")
        cv2.imwrite(out_file, img_bgr)
        saved_count += 1

    print(f"[SUCCESS] Saved {saved_count} visualization images to: {output_dir}")

if __name__ == "__main__":
    visualize_sample_annotations()
