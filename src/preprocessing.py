"""
Image preprocessing for PhytoGuard AI.

These functions apply exactly the same steps used in the notebook:
decode the image, resize it to 224 x 224 and apply MobileNetV2 normalisation
(pixel values scaled from 0-255 to -1 to 1). Training images are also augmented.
"""

import os

import numpy as np
import tensorflow as tf

from src.config import BATCH_SIZE, IMG_SIZE, VALID_EXTENSIONS

# Same augmentation as the notebook: flip, rotation and zoom
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


def preprocess_image_tensor(image):
    """Resize a decoded RGB image tensor and apply MobileNetV2 normalisation."""
    image = tf.image.resize(image, IMG_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def decode_image_bytes(image_bytes):
    """Decode JPEG or PNG bytes into an RGB tensor. Raises ValueError for invalid files."""
    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    except (tf.errors.InvalidArgumentError, ValueError) as error:
        raise ValueError("The file is not a valid JPEG or PNG image.") from error
    return image


def preprocess_image_bytes(image_bytes):
    """Turn raw uploaded image bytes into a model-ready batch of shape (1, 224, 224, 3)."""
    image = decode_image_bytes(image_bytes)
    image = preprocess_image_tensor(image)
    return np.expand_dims(image.numpy(), axis=0)


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


def _load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return preprocess_image_tensor(image), label


def _augment(image, label):
    image = data_augmentation(tf.expand_dims(image, axis=0), training=True)
    return tf.squeeze(image, axis=0), label


def build_dataset(paths, labels, training=False, batch_size=BATCH_SIZE):
    """Create a batched tf.data pipeline from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    if training:
        dataset = dataset.shuffle(len(paths), seed=42)

    dataset = dataset.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
