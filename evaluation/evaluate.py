"""
Phase 7: YOLO Model Evaluation Script
Evaluates models/best.pt on test split and outputs precision, recall, mAP50, mAP50-95 to evaluation/reports/metrics.json
"""
import os
import sys
import json
import yaml

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def evaluate_yolo_model(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_yaml = os.path.abspath(config["dataset"]["yolo_yaml"])
    best_weights_path = os.path.abspath(config["training"]["best_weights"])

    report_dir = os.path.join(root_dir, "evaluation", "reports")
    os.makedirs(report_dir, exist_ok=True)
    metrics_json_path = os.path.join(report_dir, "metrics.json")

    print("="*60)
    print("        PHASE 7: ULTRALYTICS YOLO MODEL EVALUATION")
    print("="*60)
    print(f"Model Weights: {best_weights_path}")

    try:
        from ultralytics import YOLO
        if not os.path.exists(best_weights_path):
            print(f"[NOTE] Weights '{best_weights_path}' not found. Loading pretrained YOLO model...")
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(best_weights_path)

        metrics = model.val(data=data_yaml, split="val")

        eval_report = {
            "model_weights": best_weights_path,
            "precision": float(metrics.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics.results_dict.get("metrics/recall(B)", 0.0)),
            "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        }

        print("\n--- Evaluation Results ---")
        print(f"Precision  : {eval_report['precision'] * 100:.2f}%")
        print(f"Recall     : {eval_report['recall'] * 100:.2f}%")
        print(f"mAP50      : {eval_report['mAP50'] * 100:.2f}%")
        print(f"mAP50-95   : {eval_report['mAP50-95'] * 100:.2f}%")

        with open(metrics_json_path, "w") as f:
            json.dump(eval_report, f, indent=2)

        print(f"[SUCCESS] Evaluation report saved to: {metrics_json_path}")
        return eval_report

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    evaluate_yolo_model()
