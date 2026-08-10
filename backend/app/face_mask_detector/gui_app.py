"""
Modern Tkinter / CustomTkinter Desktop GUI Application for AI Face Mask Detection.
Features:
 - Dark mode GUI layout with embedded live camera stream
 - Real-time COVID Safety Status Badge (Safe vs Warning)
 - Live face count statistics and FPS counter
 - Start / Stop Camera and Screenshot capture buttons
"""
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    USE_CTK = True
except ImportError:
    import tkinter as ctk
    USE_CTK = False

# Import face detector modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
grandparent_dir = os.path.dirname(parent_dir)

for path in [curr_dir, parent_dir, grandparent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from face_mask_detector.detect_webcam import load_face_detector, load_mask_model
    from face_mask_detector.model import preprocess_face
    from face_mask_detector.utils import draw_face_box, draw_hud_header
except ImportError:
    try:
        from backend.app.face_mask_detector.detect_webcam import load_face_detector, load_mask_model
        from backend.app.face_mask_detector.model import preprocess_face
        from backend.app.face_mask_detector.utils import draw_face_box, draw_hud_header
    except ImportError:
        from detect_webcam import load_face_detector, load_mask_model
        from model import preprocess_face
        from utils import draw_face_box, draw_hud_header

class MaskDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Mask Detector - COVID Safety Desktop App")
        self.root.geometry("1100x700")

        # Camera & Model State
        self.cap = None
        self.is_running = False
        self.face_cascade = load_face_detector()
        self.model, self.device, self.mode = load_mask_model()
        self.prev_time = time.time()

        self._build_ui()

    def _build_ui(self):
        if USE_CTK:
            # Main Layout
            self.root.grid_rowconfigure(1, weight=1)
            self.root.grid_columnconfigure(0, weight=3)
            self.root.grid_columnconfigure(1, weight=1)

            # Top Title Header
            self.header = ctk.CTkFrame(self.root, fg_color="#1E1E2E", corner_radius=0, height=60)
            self.header.grid(row=0, column=0, columnspan=2, sticky="ew")
            
            self.title_lbl = ctk.CTkLabel(
                self.header, 
                text="😷 AI FACE MASK DETECTOR & COVID SAFETY MONITOR", 
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color="#FFFFFF"
            )
            self.title_lbl.pack(side="left", padx=20, pady=15)

            # Left Panel: Live Camera Feed Container
            self.cam_frame = ctk.CTkFrame(self.root, fg_color="#11111B", corner_radius=12)
            self.cam_frame.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

            self.video_label = ctk.CTkLabel(
                self.cam_frame, 
                text="📷 Click 'Start Camera' to launch video stream...", 
                font=ctk.CTkFont(size=16),
                text_color="#A6ADC8"
            )
            self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

            # Right Panel: Controls & Status Cards
            self.side_panel = ctk.CTkFrame(self.root, fg_color="#1E1E2E", corner_radius=12)
            self.side_panel.grid(row=1, column=1, padx=(0, 15), pady=15, sticky="nsew")

            # Status Badge Card
            self.badge_card = ctk.CTkFrame(self.side_panel, fg_color="#313244", corner_radius=10)
            self.badge_card.pack(fill="x", padx=15, pady=15)

            self.badge_title = ctk.CTkLabel(
                self.badge_card, 
                text="CURRENT SAFETY STATUS", 
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#BAC2DE"
            )
            self.badge_title.pack(pady=(12, 5))

            self.badge_status = ctk.CTkLabel(
                self.badge_card, 
                text="SYSTEM READY", 
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#89B4FA"
            )
            self.badge_status.pack(pady=(0, 12))

            # Live Statistics Card
            self.stats_card = ctk.CTkFrame(self.side_panel, fg_color="#313244", corner_radius=10)
            self.stats_card.pack(fill="x", padx=15, pady=10)

            self.stats_title = ctk.CTkLabel(
                self.stats_card, 
                text="LIVE ANALYTICS", 
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#BAC2DE"
            )
            self.stats_title.pack(pady=(10, 5))

            self.lbl_total = ctk.CTkLabel(self.stats_card, text="Total Faces: 0", font=ctk.CTkFont(size=14))
            self.lbl_total.pack(anchor="w", padx=15, pady=3)

            self.lbl_masked = ctk.CTkLabel(self.stats_card, text="Masked (Safe): 0", font=ctk.CTkFont(size=14), text_color="#A6E3A1")
            self.lbl_masked.pack(anchor="w", padx=15, pady=3)

            self.lbl_unmasked = ctk.CTkLabel(self.stats_card, text="Unmasked (Unsafe): 0", font=ctk.CTkFont(size=14), text_color="#F38BA8")
            self.lbl_unmasked.pack(anchor="w", padx=15, pady=3)

            self.lbl_fps = ctk.CTkLabel(self.stats_card, text="FPS: 0.0", font=ctk.CTkFont(size=13), text_color="#CDD6F4")
            self.lbl_fps.pack(anchor="w", padx=15, pady=(3, 10))

            # Action Control Buttons
            self.btn_start = ctk.CTkButton(
                self.side_panel, 
                text="▶ Start Camera", 
                command=self.start_camera,
                fg_color="#A6E3A1", 
                hover_color="#94E2D5", 
                text_color="#11111B",
                font=ctk.CTkFont(size=15, weight="bold"),
                height=42
            )
            self.btn_start.pack(fill="x", padx=15, pady=10)

            self.btn_stop = ctk.CTkButton(
                self.side_panel, 
                text="⏹ Stop Camera", 
                command=self.stop_camera,
                fg_color="#F38BA8", 
                hover_color="#EBA0AC", 
                text_color="#11111B",
                font=ctk.CTkFont(size=15, weight="bold"),
                height=42,
                state="disabled"
            )
            self.btn_stop.pack(fill="x", padx=15, pady=5)

            self.btn_snap = ctk.CTkButton(
                self.side_panel, 
                text="📸 Save Screenshot", 
                command=self.take_screenshot,
                fg_color="#89B4FA", 
                hover_color="#B4BEFE", 
                text_color="#11111B",
                font=ctk.CTkFont(size=14, weight="bold"),
                height=38,
                state="disabled"
            )
            self.btn_snap.pack(fill="x", padx=15, pady=10)

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                self.badge_status.configure(text="❌ CAMERA ERROR", text_color="#F38BA8")
                return

            self.is_running = True
            if USE_CTK:
                self.btn_start.configure(state="disabled")
                self.btn_stop.configure(state="normal")
                self.btn_snap.configure(state="normal")
                self.badge_status.configure(text="SCANNING FACES...", text_color="#89B4FA")

            self.update_video()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        if USE_CTK:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.btn_snap.configure(state="disabled")
            self.video_label.configure(image="", text="📷 Camera Stopped.")
            self.badge_status.configure(text="CAMERA STOPPED", text_color="#BAC2DE")

    def take_screenshot(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            shot_path = f"mask_snapshot_gui_{int(time.time())}.png"
            cv2.imwrite(shot_path, self.current_frame)
            if USE_CTK:
                self.badge_status.configure(text="📸 SCREENSHOT SAVED!", text_color="#A6E3A1")

    def update_video(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            total_faces = len(faces)
            mask_count = 0
            nomask_count = 0

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue

                if self.mode == "pytorch":
                    import torch
                    import torch.nn.functional as F
                    input_tensor = preprocess_face(face_crop).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = F.softmax(outputs, dim=1)[0]
                        prediction = torch.argmax(probs).item()
                        confidence = probs[prediction].item()
                        is_masked = (prediction == 0)
                else:
                    prediction, confidence = self.model.predict(face_crop)
                    is_masked = (prediction == 0)

                if is_masked:
                    mask_count += 1
                else:
                    nomask_count += 1

                draw_face_box(frame, x, y, w, h, is_masked=is_masked, confidence=confidence)

            # Update FPS
            curr_time = time.time()
            fps = 1.0 / max((curr_time - self.prev_time), 1e-5)
            self.prev_time = curr_time

            # Update GUI Labels
            if USE_CTK:
                self.lbl_total.configure(text=f"Total Faces: {total_faces}")
                self.lbl_masked.configure(text=f"Masked (Safe): {mask_count}")
                self.lbl_unmasked.configure(text=f"Unmasked (Unsafe): {nomask_count}")
                self.lbl_fps.configure(text=f"FPS: {fps:.1f}")

                if total_faces == 0:
                    self.badge_status.configure(text="SEARCHING FOR FACE...", text_color="#CDD6F4")
                    self.badge_card.configure(fg_color="#313244")
                elif nomask_count > 0:
                    self.badge_status.configure(text="⚠️ NO MASK - UNSAFE!", text_color="#F38BA8")
                    self.badge_card.configure(fg_color="#451E2E")
                else:
                    self.badge_status.configure(text="✅ WEARING MASK - SAFE!", text_color="#A6E3A1")
                    self.badge_card.configure(fg_color="#1E3A2E")

            # Convert Frame to Tkinter ImageTk
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
            # Resize image to fit GUI frame
            img_w = self.cam_frame.winfo_width() - 20
            img_h = self.cam_frame.winfo_height() - 20
            if img_w > 100 and img_h > 100:
                pil_img = pil_img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

        self.root.after(15, self.update_video)

def main():
    if USE_CTK:
        root = ctk.CTk()
    else:
        import tkinter as tk
        root = tk.Tk()
        
    app = MaskDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
