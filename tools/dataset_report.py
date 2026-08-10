"""
Phase 3: Dataset Inspection & Report Generator
Inspects processed YOLO dataset splits, image counts, class distributions, and bounding box counts,
and outputs evaluation/reports/dataset_report.json
"""
import os
import sys
import json
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def generate_dataset_report(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = os.path.abspath(config["dataset"]["processed_dir"])
    report_dir = os.path.join(root_dir, "evaluation", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_json_path = os.path.join(report_dir, "dataset_report.json")

    print("="*60)
    print("      PHASE 3: DATASET INSPECTION & METADATA REPORT")
    print("="*60)

    splits = ["train", "val", "test"]
    report_data = {
        "dataset_repo": config["dataset"]["hf_repo"],
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "class_distribution": {
            "0 (with_mask / SAFE)": 0,
            "1 (without_mask / UNSAFE)": 0
        }
    }

    for split in splits:
        img_dir = os.path.join(processed_dir, "images", split)
        lbl_dir = os.path.join(processed_dir, "labels", split)

        if not os.path.exists(img_dir):
            continue

        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')] if os.path.exists(lbl_dir) else []

        ann_count = 0
        for lbl_file in labels:
            lbl_path = os.path.join(lbl_dir, lbl_file)
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                ann_count += len(lines)
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = parts[0]
                        if cid == "0":
                            report_data["class_distribution"]["0 (with_mask / SAFE)"] += 1
                        elif cid == "1":
                            report_data["class_distribution"]["1 (without_mask / UNSAFE)"] += 1

        report_data["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "bounding_boxes": ann_count
        }

        report_data["total_images"] += len(images)
        report_data["total_annotations"] += ann_count

    print("\n--- Dataset Report Summary ---")
    print(f"Total Images     : {report_data['total_images']}")
    print(f"Total BBoxes     : {report_data['total_annotations']}")
    print("Splits Distribution:")
    for k, v in report_data["splits"].items():
        print(f"  - {k}: {v['images']} images, {v['bounding_boxes']} bounding boxes")
    print("Class Distribution:")
    for cname, count in report_data["class_distribution"].items():
        print(f"  - {cname}: {count} instances")

    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[SUCCESS] Dataset report saved to: {report_json_path}")
    return report_data

if __name__ == "__main__":
    generate_dataset_report()
