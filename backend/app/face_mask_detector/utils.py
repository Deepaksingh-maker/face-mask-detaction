import cv2
import numpy as np

COLOR_GREEN = (50, 205, 50)
COLOR_RED = (50, 50, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 20, 20)
COLOR_BLUE = (255, 144, 30)

def draw_hud_header(frame, title="AI FACE MASK DETECTOR - REALTIME SAFETY MONITOR"):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, 50), (w, 50), COLOR_BLUE, 2)
    cv2.putText(frame, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)

def draw_face_box(frame, x, y, w, h, is_masked, confidence):
    color = COLOR_GREEN if is_masked else COLOR_RED
    status_text = "WEARING MASK - SAFE COVID" if is_masked else "NO MASK - UNSAFE!"
    sub_text = "STATUS: SAFE" if is_masked else "WARNING: WEAR A MASK"
    
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    corner_len = min(20, (x2 - x1) // 4)
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)
    
    label = f"{status_text} ({confidence * 100:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    badge_y1 = max(60, y1 - text_h - 20)
    badge_y2 = badge_y1 + text_h + 15
    badge_x2 = min(frame.shape[1], x1 + text_w + 20)
    
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(frame, label, (x1 + 10, badge_y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)
    
    sub_badge_y1 = y2 + 5
    sub_badge_y2 = sub_badge_y1 + 25
    cv2.rectangle(frame, (x1, sub_badge_y1), (x1 + 240, sub_badge_y2), COLOR_DARK_BG, -1)
    cv2.rectangle(frame, (x1, sub_badge_y1), (x1 + 240, sub_badge_y2), color, 1)
    cv2.putText(frame, sub_text, (x1 + 8, sub_badge_y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_status_summary(frame, total_faces, mask_count, nomask_count, fps=0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, h - 60), (w, h - 60), COLOR_BLUE, 1)
    
    if total_faces == 0:
        summary_text = "SEARCHING FOR FACE... Align your face in front of the camera"
        status_color = COLOR_WHITE
    elif nomask_count > 0:
        summary_text = "ALERT: UNPROTECTED FACE DETECTED - PLEASE WEAR A MASK!"
        status_color = COLOR_RED
    else:
        summary_text = "SAFE STATUS: MASK ON - YOU ARE PROTECTED & SAFE FROM COVID"
        status_color = COLOR_GREEN
        
    cv2.putText(frame, summary_text, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    stats_str = f"Faces: {total_faces} | Masked: {mask_count} | Unmasked: {nomask_count} | FPS: {fps:.1f}"
    (sw, sh), _ = cv2.getTextSize(stats_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, stats_str, (w - sw - 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
