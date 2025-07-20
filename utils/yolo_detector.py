# utils/yolo_detector.py
import torch
import os
from PIL import Image

# Load YOLOv5s model from local path
model_path = os.path.join('models', 'yolov5s.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)

def detect_damage(image_path):
    results = model(image_path)
    # Save detection result image
    result_img_path = image_path.replace("uploads", "uploads/detected")
    os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
    results.save(save_dir=os.path.dirname(result_img_path))
    return result_img_path