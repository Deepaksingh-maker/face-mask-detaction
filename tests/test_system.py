"""
Phase 22: Automated Test Suite for YOLO Face Mask Detection System
"""
import os
import sys
import unittest
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inference.tracker import IoUFaceTracker, compute_iou
from inference.stability import TemporalStabilityEngine

class TestFaceMaskDetectionSystem(unittest.TestCase):

    def test_iou_computation(self):
        boxA = [10, 10, 50, 50]
        boxB = [10, 10, 50, 50]
        iou_exact = compute_iou(boxA, boxB)
        self.assertAlmostEqual(iou_exact, 1.0, places=3)

        boxC = [100, 100, 50, 50]
        iou_zero = compute_iou(boxA, boxC)
        self.assertEqual(iou_zero, 0.0)

    def test_tracker_association(self):
        tracker = IoUFaceTracker(iou_threshold=0.25)
        det_frame1 = [{"bbox": [10, 10, 50, 50], "cls_id": 0, "conf": 0.95}]
        res1 = tracker.update(det_frame1)

        self.assertEqual(len(res1), 1)
        first_id = res1[0]["track"]["track_id"]

        det_frame2 = [{"bbox": [12, 12, 50, 50], "cls_id": 0, "conf": 0.96}]
        res2 = tracker.update(det_frame2)
        second_id = res2[0]["track"]["track_id"]

        self.assertEqual(first_id, second_id, "Tracker should keep same track_id across overlapping frames")

    def test_temporal_stability_hysteresis(self):
        stability = TemporalStabilityEngine()
        
        # Test SAFE transition
        tracked_safe = [{
            "track": {"track_id": 1, "bbox": [10, 10, 50, 50], "ewma_prob": 0.85, "state": "SAFE"},
            "raw_mask_prob": 0.90
        }]
        res_safe = stability.process_tracks(tracked_safe)
        self.assertEqual(res_safe[0]["label"], "SAFE")

        # Test UNSAFE transition
        tracked_unsafe = [{
            "track": {"track_id": 2, "bbox": [100, 100, 50, 50], "ewma_prob": 0.10, "state": "UNSAFE"},
            "raw_mask_prob": 0.05
        }]
        res_unsafe = stability.process_tracks(tracked_unsafe)
        self.assertEqual(res_unsafe[0]["label"], "UNSAFE")

if __name__ == "__main__":
    unittest.main()
