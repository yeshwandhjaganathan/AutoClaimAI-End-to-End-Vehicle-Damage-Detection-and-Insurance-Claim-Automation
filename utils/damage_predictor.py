from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def load_model():
    model = keras_load_model("models/densenet121.h5")
    return model

def predict_damage(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0]
    severity_classes = ['Minor', 'Moderate', 'Severe']
    predicted_index = np.argmax(prediction)
    return severity_classes[predicted_index]