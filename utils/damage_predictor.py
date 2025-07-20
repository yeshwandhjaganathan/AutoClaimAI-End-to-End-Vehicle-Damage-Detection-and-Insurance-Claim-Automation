import os
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image

def download_model_from_gdrive():
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
        import gdown

    model_dir = "models"
    model_path = os.path.join(model_dir, "densenet_model.h5")
    
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print("[INFO] Downloading model from Google Drive...")
        gdown.download("https://drive.google.com/uc?id=1R7SqrIkfJETSrovnRCRrzoN9AZBKaWwU", model_path, quiet=False)
    return model_path

def load_model():
    model_path = download_model_from_gdrive()
    return keras_load_model(model_path)

def predict_damage(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0]
    severity_classes = ['Minor', 'Moderate', 'Severe']
    predicted_index = np.argmax(prediction)
    return severity_classes[predicted_index]