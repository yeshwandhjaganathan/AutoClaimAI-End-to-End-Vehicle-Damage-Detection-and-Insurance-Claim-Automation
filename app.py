import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from utils.damage_predictor import load_model, predict_damage
from utils.duplicate_checker import is_duplicate
from utils.explainability import generate_shap_image
from utils.video_processor import extract_frames, process_video_frames
import csv
import uuid
from utils.severity_utils import predict_severity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SHAP_FOLDER'] = 'static/shap'
app.config['FRAME_FOLDER'] = 'static/frames'
app.config['CSV_FOLDER'] = 'static/csv'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        if ext == 'mp4':
            frame_paths = extract_frames(save_path, app.config['FRAME_FOLDER'], max_frames=5)
            results = process_video_frames(frame_paths, model, app.config['UPLOAD_FOLDER'], app.config['SHAP_FOLDER'])

            os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)
            csv_filename = f"summary_{uuid.uuid4().hex[:8]}.csv"
            csv_path = os.path.join(app.config['CSV_FOLDER'], csv_filename)
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['frame', 'severity', 'duplicate', 'shap_image'])
                writer.writeheader()
                writer.writerows(results)

            return render_template('video_result.html', results=results, csv_file=csv_filename)

        else:
            severity = predict_damage(save_path, model)
            duplicate = is_duplicate(save_path)
            shap_img = generate_shap_image(save_path, model, app.config['SHAP_FOLDER'])

            return render_template('result.html',
                                   severity=severity,
                                   duplicate=duplicate,
                                   filename=unique_name,
                                   shap_image=shap_img)
    else:
        return "Unsupported file format", 400

@app.route('/download/<path:filename>')
def download_file(filename):
    folder = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(os.path.join('static', folder), file, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SHAP_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CSV_FOLDER'], exist_ok=True)
    app.run(debug=True)
