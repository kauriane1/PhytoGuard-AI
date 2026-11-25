"""
Model creation, loading and export for PhytoGuard AI.

Training (notebook and offline retraining) uses the full TensorFlow/Keras model:
MobileNetV2 pretrained on ImageNet, global average pooling, dropout and a
38-class softmax layer, saved as models/phytoguard_ai_final.h5.

For deployment the trained model is split into two parts that give exactly the
same predictions but need far less memory:

1. Feature extractor: MobileNetV2 + global average pooling, converted to
   TensorFlow Lite (models/phytoguard_feature_extractor.tflite).
2. Classification head: the weights of the final softmax layer, stored with
   NumPy (models/phytoguard_head.npz).

Export the serving files after training with:
    python -m src.model --export
"""

import json

import numpy as np

from src.config import (
    CLASS_NAMES_PATH, FEATURE_EXTRACTOR_PATH, FINE_TUNE_LAYERS, HEAD_PATH, IMG_SIZE,
    MODEL_PATH,
)


# ---------------------------------------------------------------------------
# Training model (TensorFlow is imported only when these functions are used)
# ---------------------------------------------------------------------------

def build_model(num_classes, learning_rate=1e-4):
    """Build a new MobileNetV2 transfer-learning model with a frozen base."""
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return compile_model(model, learning_rate)


def compile_model(model, learning_rate):
    """Compile with the same optimiser, loss and metric used in the notebook."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def prepare_for_fine_tuning(model, fine_tune_layers=FINE_TUNE_LAYERS, learning_rate=1e-5):
    """Unfreeze the top layers of the MobileNetV2 base and recompile with a small learning rate."""
    import tensorflow as tf

    base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    return compile_model(model, learning_rate)


def load_keras_model(model_path=MODEL_PATH):
    """Load the full saved Keras model."""
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Serving files
# ---------------------------------------------------------------------------

def export_for_serving(model_path=MODEL_PATH):
    """Split the trained Keras model into a TFLite feature extractor and a NumPy head."""
    import tensorflow as tf

    model = load_keras_model(model_path)

    # Everything up to global average pooling becomes the feature extractor
    pooling = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    extractor = tf.keras.Model(model.input, pooling.output)
    converter = tf.lite.TFLiteConverter.from_keras_model(extractor)
    with open(FEATURE_EXTRACTOR_PATH, "wb") as f:
        f.write(converter.convert())

    # The final Dense layer becomes the head (dropout is inactive at inference)
    dense = model.layers[-1]
    weights, bias = dense.get_weights()
    save_head(weights, bias)
    return FEATURE_EXTRACTOR_PATH, HEAD_PATH


def save_head(weights, bias, path=HEAD_PATH):
    np.savez(path, weights=weights.astype(np.float32), bias=bias.astype(np.float32))


def load_head(path=HEAD_PATH):
    data = np.load(path)
    return data["weights"], data["bias"]


def load_class_names(class_names_path=CLASS_NAMES_PATH):
    """Load the class names in the order the model outputs them."""
    with open(class_names_path) as f:
        return json.load(f)


def softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


class FeatureExtractor:
    """Runs the TFLite MobileNetV2 feature extractor (1280 features per image)."""

    def __init__(self, path=FEATURE_EXTRACTOR_PATH, num_threads=None):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, image_batch):
        features = []
        for image in image_batch:
            self.interpreter.set_tensor(self.input_index, image[np.newaxis].astype(np.float32))
            self.interpreter.invoke()
            features.append(self.interpreter.get_tensor(self.output_index)[0].copy())
        return np.stack(features)


if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        paths = export_for_serving()
        print("Saved:", *paths, sep="\n  ")
    else:
        print("Usage: python -m src.model --export")
