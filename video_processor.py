import cv2
import os
import uuid
from utils.damage_predictor import predict_damage
from utils.duplicate_checker import is_duplicate
from utils.explainability import generate_shap_image

def extract_frames(video_path, output_dir, max_frames=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames * interval:
            break
        if frame_count % interval == 0:
            frame_filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
            full_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(full_path, frame)
            saved_frames.append(full_path)
        frame_count += 1

    cap.release()
    return saved_frames

def process_video_frames(frame_paths, model, upload_dir, shap_dir):
    results = []

    for frame_path in frame_paths:
        filename = os.path.basename(frame_path)
        severity = predict_damage(frame_path, model)
        duplicate = is_duplicate(frame_path)
        shap_img_name = generate_shap_image(frame_path, model, shap_dir)
        results.append({
            'frame': filename,
            'severity': severity,
            'duplicate': duplicate,
            'shap_image': shap_img_name
        })

    return results