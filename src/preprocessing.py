"""
Image preprocessing for PhytoGuard AI.

The same steps as the notebook: decode the image as RGB, resize it to 224 x 224
and apply MobileNetV2 normalisation (pixel values scaled from 0-255 to -1 to 1).

This module only needs NumPy and Pillow, so the deployed API stays small enough
for a free cloud instance. The TensorFlow training pipeline lives in the notebook
and in src/model.py.
"""

import io
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import IMG_SIZE, VALID_EXTENSIONS


def decode_image_bytes(image_bytes):
    """Decode JPEG or PNG bytes into an RGB PIL image. Raises ValueError for invalid files."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The file is not a valid JPEG or PNG image.") from error
    return image.convert("RGB")


def preprocess_pil_image(image):
    """Resize to 224 x 224 and scale pixels to [-1, 1] (MobileNetV2 preprocessing)."""
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return array / 127.5 - 1.0


def preprocess_image_bytes(image_bytes):
    """Turn raw uploaded image bytes into a model-ready batch of shape (1, 224, 224, 3)."""
    return preprocess_pil_image(decode_image_bytes(image_bytes))[np.newaxis, ...]


def preprocess_image_file(image_path):
    """Read an image from disk and return a model-ready batch of shape (1, 224, 224, 3)."""
    with open(image_path, "rb") as f:
        return preprocess_image_bytes(f.read())


def load_labelled_folder(folder, class_names):
    """
    Collect image paths and integer labels from a folder of class subfolders.

    Folders that do not match a known class name are skipped, so a mislabelled
    upload cannot break retraining.
    """
    paths, labels, skipped = [], [], []

    if not os.path.isdir(folder):
        return paths, labels, skipped

    for class_name in sorted(os.listdir(folder)):
        class_dir = os.path.join(folder, class_name)

        if not os.path.isdir(class_dir):
            continue

        if class_name not in class_names:
            skipped.append(class_name)
            continue

        label = class_names.index(class_name)
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(VALID_EXTENSIONS):
                paths.append(os.path.join(class_dir, file_name))
                labels.append(label)

    return paths, labels, skipped
