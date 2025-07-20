# Severity classification logic
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_severity_model(model_path):
    return load_model(model_path)

def predict_severity(model, image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    classes = ['Minor', 'Moderate', 'Severe']
    return classes[class_idx], prediction[0][class_idx]