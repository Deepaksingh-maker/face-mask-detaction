"""
Phase 10 & 11: IoU Object / Face Tracker Engine
Maintains persistent track IDs (Face #1, Face #2...) across video frames using IoU bounding box matching.
"""
import time
from collections import deque

def compute_iou(boxA, boxB):
    """Computes Intersection-over-Union (IoU) between two boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class IoUFaceTracker:
    """
    Lightweight IoU Face Tracker Engine.
    Associates current frame detections with persistent face tracks.
    """
    def __init__(self, iou_threshold=0.25, max_stale_seconds=1.5, history_len=7):
        self.iou_threshold = iou_threshold
        self.max_stale_seconds = max_stale_seconds
        self.history_len = history_len
        self.next_track_id = 1
        self.tracks = {}

    def update(self, current_detections):
        """
        Input: current_detections = list of dicts: [{ bbox: [x, y, w, h], cls_id: int, conf: float }]
        Returns: list of updated track objects with track_id and history queue.
        """
        now = time.time()

        # Remove stale tracks
        stale_ids = [tid for tid, tr in self.tracks.items() if (now - tr['last_seen']) > self.max_stale_seconds]
        for tid in stale_ids:
            del self.tracks[tid]

        matched_track_ids = set()
        updated_results = []

        for det in current_detections:
            curr_box = det['bbox']
            best_iou = 0.0
            best_tid = None

            for tid, tr in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = compute_iou(curr_box, tr['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_tid]
                matched_track_ids.add(best_tid)
            else:
                best_tid = self.next_track_id
                self.next_track_id += 1
                track = {
                    'track_id': best_tid,
                    'bbox': curr_box,
                    'history': deque(maxlen=self.history_len),
                    'ewma_prob': 0.50,
                    'state': 'CHECKING',
                    'first_seen': now,
                    'last_seen': now
                }
                self.tracks[best_tid] = track

            track['bbox'] = curr_box
            track['last_seen'] = now

            # Compute mask probability: Class 0 (with_mask) -> conf, Class 1 -> (1.0 - conf)
            raw_mask_prob = det['conf'] if det['cls_id'] == 0 else (1.0 - det['conf'])
            track['history'].append(raw_mask_prob)

            updated_results.append({
                'track': track,
                'det': det,
                'raw_mask_prob': raw_mask_prob
            })

        return updated_results
