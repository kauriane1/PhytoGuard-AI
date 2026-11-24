"""
Model creation and loading for PhytoGuard AI.

The architecture matches the notebook: MobileNetV2 pretrained on ImageNet,
followed by global average pooling, dropout and a 38-class softmax layer.
"""

import json

import tensorflow as tf

from src.config import CLASS_NAMES_PATH, FINE_TUNE_LAYERS, IMG_SIZE, MODEL_PATH


def build_model(num_classes, learning_rate=1e-4):
    """Build a new MobileNetV2 transfer-learning model with a frozen base."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    """Compile with the same optimiser, loss and metric used in the notebook."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def prepare_for_fine_tuning(model, fine_tune_layers=FINE_TUNE_LAYERS, learning_rate=1e-5):
    """Unfreeze the top layers of the MobileNetV2 base and recompile with a small learning rate."""
    base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base_model.trainable = True

    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    return compile_model(model, learning_rate)


def load_model(model_path=MODEL_PATH):
    """Load a saved model from disk."""
    return tf.keras.models.load_model(model_path)


def load_class_names(class_names_path=CLASS_NAMES_PATH):
    """Load the class names in the order the model outputs them."""
    with open(class_names_path) as f:
        return json.load(f)
