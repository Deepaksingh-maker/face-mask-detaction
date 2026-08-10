"""
Phase 2: Hugging Face Dataset Downloader
Downloads 'hlydecker/face-masks' dataset automatically into data/raw/face-masks/
"""
import os
import sys
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def download_huggingface_dataset(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    repo_id = config["dataset"]["hf_repo"]
    raw_dir = os.path.abspath(config["dataset"]["raw_dir"])
    os.makedirs(raw_dir, exist_ok=True)

    print("="*60)
    print(f"   PHASE 2: DOWNLOADING HUGGINGFACE DATASET: {repo_id}")
    print("="*60)
    print(f"Target Directory: {raw_dir}")

    try:
        from datasets import load_dataset
        print(f"[INFO] Fetching dataset '{repo_id}' from Hugging Face Hub...")
        dataset = load_dataset(repo_id)
        
        print("\n[SUCCESS] Dataset downloaded successfully!")
        print("Available Splits:", list(dataset.keys()))
        for split_name in dataset.keys():
            print(f"  - Split '{split_name}': {len(dataset[split_name])} samples")

        # Save to disk
        dataset_save_path = os.path.join(raw_dir, "hf_dataset")
        dataset.save_to_disk(dataset_save_path)
        print(f"[SUCCESS] Saved raw Hugging Face dataset to: {dataset_save_path}")
        return dataset

    except ImportError:
        print("[NOTE] 'datasets' package not installed. Installing required Hugging Face dependencies...")
        os.system(f"{sys.executable} -m pip install datasets huggingface_hub")
        from datasets import load_dataset
        dataset = load_dataset(repo_id)
        dataset_save_path = os.path.join(raw_dir, "hf_dataset")
        dataset.save_to_disk(dataset_save_path)
        return dataset
    except Exception as e:
        print(f"[ERROR] Failed downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_huggingface_dataset()
