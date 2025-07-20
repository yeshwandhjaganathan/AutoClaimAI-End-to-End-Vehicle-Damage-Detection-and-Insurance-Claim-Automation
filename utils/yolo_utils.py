# YOLO utilities
import torch
from PIL import Image

def load_yolo_model(model_path='yolov5s.pt'):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def detect_damage(model, image_path):
    results = model(Image.open(image_path))
    return results.pandas().xyxy[0].to_dict(orient="records")