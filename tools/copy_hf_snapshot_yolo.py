"""
Hugging Face Snapshot YOLO Dataset Copier.
Copies images and label files from huggingface_hub snapshot directly into data/processed/face-masks-yolo/
"""
import os
import sys
import shutil
import yaml
from huggingface_hub import snapshot_download

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def populate_yolo_from_snapshot(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    repo_id = config["dataset"]["hf_repo"]
    processed_dir = os.path.abspath(config["dataset"]["processed_dir"])

    print("="*60)
    print(f"   POPULATING YOLO DATASET FROM HF SNAPSHOT: {repo_id}")
    print("="*60)

    try:
        snapshot_dir = snapshot_download(repo_id, repo_type="dataset")
        print(f"Snapshot directory: {snapshot_dir}")

        splits = [("train", "train"), ("valid", "val"), ("test", "test")]

        for hf_split, yolo_split in splits:
            hf_img_dir = os.path.join(snapshot_dir, hf_split, "images")
            hf_lbl_dir = os.path.join(snapshot_dir, hf_split, "labels")

            target_img_dir = os.path.join(processed_dir, "images", yolo_split)
            target_lbl_dir = os.path.join(processed_dir, "labels", yolo_split)

            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_lbl_dir, exist_ok=True)

            if os.path.exists(hf_img_dir):
                for f in os.listdir(hf_img_dir):
                    shutil.copy2(os.path.join(hf_img_dir, f), os.path.join(target_img_dir, f))
            
            if os.path.exists(hf_lbl_dir):
                for f in os.listdir(hf_lbl_dir):
                    shutil.copy2(os.path.join(hf_lbl_dir, f), os.path.join(target_lbl_dir, f))

            print(f"  - Copied split '{hf_split}' -> '{yolo_split}'")

        # Generate standard data.yaml
        data_yaml_content = {
            "path": processed_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {
                0: "with_mask",
                1: "without_mask"
            }
        }

        data_yaml_path = os.path.join(processed_dir, "data.yaml")
        with open(data_yaml_path, "w") as f_yaml:
            yaml.dump(data_yaml_content, f_yaml, sort_keys=False)

        print(f"\n[SUCCESS] Populated data.yaml at: {data_yaml_path}")
        print("[SUCCESS] Processed YOLO dataset ready for training!")
        return True

    except Exception as e:
        print(f"[ERROR] Failed copying HF snapshot: {e}")
        return False

if __name__ == "__main__":
    populate_yolo_from_snapshot()
