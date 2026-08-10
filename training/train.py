"""
Phase 6: Ultralytics YOLO Training Script
Trains a lightweight YOLO model on data/processed/face-masks-yolo/data.yaml and saves models/best.pt
"""
import os
import sys
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def train_yolo_model(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_yaml = os.path.abspath(config["dataset"]["yolo_yaml"])
    models_dir = os.path.abspath(config["training"]["model_output_dir"])
    os.makedirs(models_dir, exist_ok=True)
    best_weights_path = os.path.abspath(config["training"]["best_weights"])

    print("="*60)
    print("         PHASE 6: ULTRALYTICS YOLO MODEL TRAINING")
    print("="*60)
    print(f"Data Config: {data_yaml}")
    print(f"Target Output Weights: {best_weights_path}")

    try:
        from ultralytics import YOLO
        base_model_name = config["training"].get("base_model", "yolov8n.pt")
        print(f"[INFO] Initializing YOLO model from base: {base_model_name}")
        
        model = YOLO(base_model_name)

        # Train model
        results = model.train(
            data=data_yaml,
            epochs=config["training"].get("epochs", 15),
            imgsz=config["training"].get("img_size", 640),
            batch=config["training"].get("batch_size", 16),
            workers=config["training"].get("workers", 4),
            project=models_dir,
            name="train_run",
            exist_ok=True,
            verbose=True
        )

        # Save best.pt directly into models/best.pt
        run_best_pth = os.path.join(models_dir, "train_run", "weights", "best.pt")
        if os.path.exists(run_best_pth):
            import shutil
            shutil.copy(run_best_pth, best_weights_path)
            print(f"[SUCCESS] Trained best.pt weights copied to: {best_weights_path}")
        else:
            # Fallback: Save model state
            model.save(best_weights_path)
            print(f"[SUCCESS] Model saved to: {best_weights_path}")

        print("[SUCCESS] YOLO Model Training Completed!")
        return model

    except ImportError:
        print("[NOTE] 'ultralytics' package not installed. Installing required Ultralytics dependencies...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.train(data=data_yaml, epochs=config["training"].get("epochs", 15), imgsz=640)
        return model
    except Exception as e:
        print(f"[ERROR] YOLO training failed: {e}")
        return None

if __name__ == "__main__":
    train_yolo_model()
