import os
from PIL import Image
import imagehash

UPLOAD_FOLDER = 'static/uploads'

def is_duplicate(upload_path):
    uploaded_hash = imagehash.average_hash(Image.open(upload_path))

    for fname in os.listdir(UPLOAD_FOLDER):
        if fname == os.path.basename(upload_path):
            continue
        existing_path = os.path.join(UPLOAD_FOLDER, fname)
        try:
            existing_hash = imagehash.average_hash(Image.open(existing_path))
            if uploaded_hash - existing_hash < 5:
                return True
        except:
            continue
    return False