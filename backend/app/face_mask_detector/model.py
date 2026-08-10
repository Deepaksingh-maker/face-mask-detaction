import sys
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CLASS_NAMES = {
    0: "with_mask",    # SAFE
    1: "without_mask"  # UNSAFE
}

if HAS_TORCH:
    class MaskDetectorNet(nn.Module):
        """
        PyTorch Deep Convolutional Neural Network for Face Mask Classification.
        Input: Tensor of shape (B, 3, 128, 128)
        Output: Logits of shape (B, 2)
           - Index 0: with_mask    (SAFE)
           - Index 1: without_mask (UNSAFE)
        """
        def __init__(self):
            super(MaskDetectorNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.4)
            self.fc1 = nn.Linear(128 * 16 * 16, 64)
            self.fc2 = nn.Linear(64, 2)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
else:
    class MaskDetectorNet:
        """Placeholder class when PyTorch is not present."""
        def __init__(self):
            pass

def preprocess_face(face_bgr, target_size=(128, 128)):
    """Preprocesses OpenCV BGR face crop image for neural network inference."""
    rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_face, target_size)
    normalized = resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    standardized = (normalized - mean) / std
    
    chw = np.transpose(standardized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    
    if HAS_TORCH:
        return torch.tensor(batch, dtype=torch.float32)
    return batch

class LightweightNumpyMaskClassifier:
    """
    High-Precision Mask / Rumal / Gaiter Classifier Engine.
    Accurately isolates Mouth & Chin Zone (int(h * 0.50) to int(h * 0.85)) and compares against Forehead Skin:
     - Class 0: with_mask / rumal / black gaiter / cloth wrap ("SAFE" 🟢)
     - Class 1: clear unmasked face ("UNSAFE" 🔴)
    """
    def __init__(self):
        pass

    def predict(self, face_bgr):
        h, w = face_bgr.shape[:2]
        if h < 10 or w < 10:
            return 1, 0.50

        # Region Segmentation:
        # 1. Forehead & Eyes Zone: top 10% to 45% (Measures natural skin tone)
        # 2. Mouth & Chin Zone: middle-lower 50% to 85% (Exact region covered by mask/gaiter/rumal)
        forehead_zone = face_bgr[int(h * 0.10):int(h * 0.45), :]
        mouth_chin_zone = face_bgr[int(h * 0.50):int(h * 0.85), :]
        
        hsv_forehead = cv2.cvtColor(forehead_zone, cv2.COLOR_BGR2HSV)
        hsv_mouth_chin = cv2.cvtColor(mouth_chin_zone, cv2.COLOR_BGR2HSV)
        
        # Robust Facial Skin Color Ranges in HSV Space
        lower_skin1 = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([165, 20, 50], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

        mouth_skin_pixels = np.sum((cv2.inRange(hsv_mouth_chin, lower_skin1, upper_skin1) > 0) | 
                                   (cv2.inRange(hsv_mouth_chin, lower_skin2, upper_skin2) > 0))
        
        forehead_skin_pixels = np.sum((cv2.inRange(hsv_forehead, lower_skin1, upper_skin1) > 0) | 
                                       (cv2.inRange(hsv_forehead, lower_skin2, upper_skin2) > 0))

        total_mouth_chin = mouth_chin_zone.shape[0] * mouth_chin_zone.shape[1] + 1e-5
        total_forehead = forehead_zone.shape[0] * forehead_zone.shape[1] + 1e-5

        mouth_skin_ratio = mouth_skin_pixels / total_mouth_chin
        forehead_skin_ratio = forehead_skin_pixels / total_forehead
        skin_diff = forehead_skin_ratio - mouth_skin_ratio

        # Fabric & Mask Color Detection in Mouth/Chin Zone
        # 1. Surgical Blue/Cyan mask
        blue_mask = cv2.inRange(hsv_mouth_chin, np.array([85, 40, 40]), np.array([135, 255, 255]))
        # 2. White/Light Gray N95 mask
        white_mask = cv2.inRange(hsv_mouth_chin, np.array([0, 0, 160]), np.array([180, 55, 255]))
        # 3. Black Mask / Black Neck Gaiter / Dark Cloth (V up to 135)
        black_mask = cv2.inRange(hsv_mouth_chin, np.array([0, 0, 0]), np.array([180, 120, 135]))
        # 4. Red/Patterned Cloth
        red_mask1 = cv2.inRange(hsv_mouth_chin, np.array([0, 60, 40]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv_mouth_chin, np.array([170, 60, 40]), np.array([180, 255, 255]))
        
        fabric_pixels = np.sum((blue_mask > 0) | (white_mask > 0) | (black_mask > 0) | (red_mask1 > 0) | (red_mask2 > 0))
        fabric_ratio = fabric_pixels / total_mouth_chin

        # DECISION ENGINE:
        # If mouth & chin zone contains exposed skin (mouth_skin_ratio >= 0.22 AND skin_diff < 0.20),
        # the mouth and lips are clearly visible -> UNMASKED (UNSAFE 🔴)!
        if mouth_skin_ratio >= 0.22 and skin_diff < 0.20:
            # Class 1: clear unmasked face ("UNSAFE" 🔴)
            confidence = min(0.99, float(0.80 + mouth_skin_ratio * 0.19))
            return 1, confidence
        else:
            # Class 0: with_mask / rumal / black gaiter / cloth wrap ("SAFE" 🟢)
            confidence = min(0.99, float(0.88 + max(fabric_ratio, skin_diff) * 0.11))
            return 0, confidence
