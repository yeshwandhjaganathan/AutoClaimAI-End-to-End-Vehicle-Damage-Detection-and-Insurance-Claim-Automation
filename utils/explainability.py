import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from PIL import Image

def generate_shap_image(img_path, model, shap_folder):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    background = np.random.rand(1, 224, 224, 3)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_array)

    output_path = os.path.join(shap_folder, f"shap_{os.path.basename(img_path)}.png")
    plt.figure()
    shap.image_plot(shap_values, img_array, show=False)
    plt.savefig(output_path)
    plt.close()
    return 
    os.path.basename(output_path)