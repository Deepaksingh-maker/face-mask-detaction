"""
Phase 12 - 16: Temporal EWMA Prediction Stability, Hysteresis, & Real-Time Analytics Engine
"""
import yaml

class TemporalStabilityEngine:
    """
    Temporal Probability Smoothing & Hysteresis State Machine Engine.
    Prevents frame-to-frame prediction flickering.
    """
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        stab_cfg = config.get("stability", {})
        self.alpha = stab_cfg.get("ewma_alpha", 0.35)
        self.hysteresis_safe = stab_cfg.get("hysteresis_safe_threshold", 0.70)
        self.hysteresis_unsafe = stab_cfg.get("hysteresis_unsafe_threshold", 0.35)

    def process_tracks(self, tracked_items):
        """
        Input: list of tracked dicts from IoUFaceTracker
        Returns: list of stabilized face prediction dicts
        """
        results = []

        for item in tracked_items:
            track = item["track"]
            raw_prob = item["raw_mask_prob"]

            # Exponentially Weighted Moving Average (EWMA)
            track["ewma_prob"] = self.alpha * raw_prob + (1.0 - self.alpha) * track["ewma_prob"]
            smoothed_p = track["ewma_prob"]

            curr_state = track["state"]

            # HYSTERESIS STATE MACHINE & UNCERTAINTY THRESHOLDS
            if curr_state == "SAFE":
                new_state = "UNSAFE" if smoothed_p <= self.hysteresis_unsafe else "SAFE"
            elif curr_state == "UNSAFE":
                new_state = "SAFE" if smoothed_p >= self.hysteresis_safe else "UNSAFE"
            else:
                if smoothed_p >= self.hysteresis_safe:
                    new_state = "SAFE"
                elif smoothed_p <= self.hysteresis_unsafe:
                    new_state = "UNSAFE"
                else:
                    new_state = "CHECKING"

            track["state"] = new_state

            if new_state == "SAFE":
                label = "SAFE"
                is_masked = True
                conf_val = smoothed_p
            elif new_state == "UNSAFE":
                label = "UNSAFE"
                is_masked = False
                conf_val = 1.0 - smoothed_p
            else:
                label = "CHECKING"
                is_masked = False
                conf_val = 0.65

            results.append({
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "x": track["bbox"][0],
                "y": track["bbox"][1],
                "width": track["bbox"][2],
                "height": track["bbox"][3],
                "label": label,
                "is_masked": is_masked,
                "confidence": round(float(conf_val), 4),
                "raw_prob": round(float(raw_prob), 4),
                "smoothed_prob": round(float(smoothed_p), 4)
            })

        return results

    def compute_analytics_and_global_status(self, face_results, fps=30.0):
        total_faces = len(face_results)
        masked_count = sum(1 for f in face_results if f["label"] == "SAFE")
        unmasked_count = sum(1 for f in face_results if f["label"] == "UNSAFE")
        checking_count = sum(1 for f in face_results if f["label"] == "CHECKING")

        avg_conf = (sum(f["confidence"] for f in face_results) / total_faces) if total_faces > 0 else 0.0

        if total_faces == 0:
            overall_status = "NO_FACE"
        elif unmasked_count > 0:
            overall_status = "UNSAFE"
        elif checking_count > 0:
            overall_status = "CHECKING"
        else:
            overall_status = "SAFE"

        return {
            "total_faces": total_faces,
            "masked_count": masked_count,
            "unmasked_count": unmasked_count,
            "checking_count": checking_count,
            "average_confidence": round(float(avg_conf), 4),
            "fps": round(float(fps), 1),
            "overall_status": overall_status
        }
