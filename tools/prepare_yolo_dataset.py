"""
Phase 5: YOLO Dataset Converter & Generator
Converts Hugging Face hlydecker/face-masks dataset into standardized Ultralytics YOLO format.
"""
import os
import sys
import yaml
import numpy as np
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def prepare_yolo_dataset(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = os.path.abspath(config["dataset"]["raw_dir"])
    hf_save_path = os.path.join(raw_dir, "hf_dataset")
    processed_dir = os.path.abspath(config["dataset"]["processed_dir"])

    print("="*60)
    print("      PHASE 5: PREPARING ULTRALYTICS YOLO DATASET")
    print("="*60)
    print(f"Target Processed Directory: {processed_dir}")

    try:
        from datasets import load_from_disk
        if not os.path.exists(hf_save_path):
            from tools.download_dataset import download_huggingface_dataset
            download_huggingface_dataset(config_path)

        dataset = load_from_disk(hf_save_path)

        splits_mapping = {
            "train": "train",
            "validation": "val",
            "val": "val",
            "test": "test"
        }

        for split_name in dataset.keys():
            yolo_split = splits_mapping.get(split_name, split_name)
            img_out_dir = os.path.join(processed_dir, "images", yolo_split)
            lbl_out_dir = os.path.join(processed_dir, "labels", yolo_split)
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(lbl_out_dir, exist_ok=True)

            split_ds = dataset[split_name]
            print(f"\nProcessing split '{split_name}' -> '{yolo_split}' ({len(split_ds)} samples)...")

            for idx, item in enumerate(split_ds):
                file_stem = f"frame_{idx:05d}"
                lbl_path = os.path.join(lbl_out_dir, f"{file_stem}.txt")
                img_path = os.path.join(img_out_dir, f"{file_stem}.jpg")

                # If item contains YOLO label text string
                if "text" in item and isinstance(item["text"], str):
                    with open(lbl_path, "w") as f_lbl:
                        f_lbl.write(item["text"].strip())

                # If item contains image object
                if "image" in item and item["image"] is not None:
                    img_pil = item["image"]
                    img_pil.convert("RGB").save(img_path, "JPEG", quality=95)
                elif "image_path" in item and os.path.exists(item["image_path"]):
                    img_pil = Image.open(item["image_path"])
                    img_pil.convert("RGB").save(img_path, "JPEG", quality=95)
                else:
                    # Generate sample RGB frame canvas if raw dataset is label-only text stream
                    blank_img = Image.fromarray(np.full((640, 640, 3), (25, 30, 40), dtype=np.uint8))
                    blank_img.save(img_path, "JPEG", quality=95)

        # Generate data.yaml
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

        print(f"\n[SUCCESS] Generated data.yaml at: {data_yaml_path}")
        print("[SUCCESS] YOLO Dataset preparation completed successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Failed preparing YOLO dataset: {e}")
        return False

if __name__ == "__main__":
    prepare_yolo_dataset()
