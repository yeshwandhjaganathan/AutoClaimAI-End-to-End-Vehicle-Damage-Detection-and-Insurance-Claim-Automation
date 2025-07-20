# SHAP explainability logic
import shap
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def explain_with_shap(model, image_path, save_path='static/shap_output.png'):
    image = Image.open(image_path).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    explainer = shap.GradientExplainer(model, img_input)
    shap_values, indexes = explainer.shap_values(img_input)

    shap.image_plot(shap_values, img_input, show=False)
    plt.savefig(save_path)
    return save_path