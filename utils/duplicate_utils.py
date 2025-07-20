# Duplicate detection logic
from imagehash import phash
from PIL import Image

def is_duplicate(image1_path, image2_path, threshold=5):
    hash1 = phash(Image.open(image1_path))
    hash2 = phash(Image.open(image2_path))
    return abs(hash1 - hash2) <= threshold