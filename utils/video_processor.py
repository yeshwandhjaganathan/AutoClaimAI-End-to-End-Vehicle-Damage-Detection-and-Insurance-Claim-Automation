import cv2
import os
import uuid
from utils.damage_predictor import predict_damage
from utils.duplicate_checker import is_duplicate
from utils.explainability import generate_shap_image
from utils.yolo_detector import detect_damage  # ✅ YOLO integration

def extract_frames(video_path, frame_folder, max_frames=5):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = []

    while count < max_frames:
        success, frame = vidcap.read()
        if not success:
            break
        frame_name = f"frame_{uuid.uuid4().hex[:8]}.jpg"
        frame_path = os.path.join(frame_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        saved_frames.append(frame_path)
        count += 1
    vidcap.release()
    return saved_frames

def process_video_frames(frame_paths, model, upload_folder, shap_folder):
    results = []
    for path in frame_paths:
        severity = predict_damage(path, model)
        duplicate = is_duplicate(path)
        shap_img = generate_shap_image(path, model, shap_folder)
        yolo_result_img = detect_damage(path)  # ✅ YOLO detection

        results.append({
            'frame': os.path.basename(path),
            'severity': severity,
            'duplicate': 'Yes' if duplicate else 'No',
            'shap_image': shap_img,
            'yolo_detected': os.path.basename(yolo_result_img)  # ✅ Include YOLO result
        })
    return results